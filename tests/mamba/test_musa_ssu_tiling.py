"""Tiled SSU must mask dimension tails and preserve cache addressing."""

import os
import struct

import pytest
import torch

from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton

from .test_philox_cpu_oracle import cvt_rs_f16_bits, philox4x32_words

pytestmark = pytest.mark.skipif(
    os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa",
    reason="requires a MUSA runtime",
)


@pytest.mark.parametrize("dim", [3, 17, 65])
@pytest.mark.parametrize("vectors", [False, True])
@pytest.mark.parametrize("tied_hdim", [False, True])
def test_dimension_tail_slots_and_strided_storage(dim, vectors, tied_hdim):
    batch, heads, groups, n = 3, 4, 2, 96
    storage = torch.full(
        (4, heads, 2 * dim, 2 * n), 99.0, device="musa", dtype=torch.float16
    )
    state = storage[:, :, ::2, ::2]
    state.fill_(0.25)
    x = torch.full((batch, heads, dim), 0.5, device="musa", dtype=torch.bfloat16)
    if tied_hdim:
        dt = torch.full((batch, heads, 1), 0.125, device="musa").expand_as(x)
        a = torch.zeros(heads, 1, 1, device="musa").expand(heads, dim, n)
    else:
        dt = torch.full_like(x, 0.125)
        a = torch.zeros(heads, dim, n, device="musa")
    b = torch.full((batch, groups, n), 0.5, device="musa", dtype=x.dtype)
    c = torch.full_like(b, 0.25)
    shape = (heads,) if vectors else (heads, dim)
    if tied_hdim and not vectors:
        skip = torch.full((heads, 1), 0.5, device="musa").expand(heads, dim)
        bias = torch.full((heads, 1), 0.25, device="musa").expand(heads, dim)
    else:
        skip = torch.full(shape, 0.5, device="musa")
        bias = torch.full(shape, 0.25, device="musa")
    src = torch.tensor([1, 1, -1], device="musa", dtype=torch.int32)
    dst = torch.tensor([2, -1, 3], device="musa", dtype=torch.int32)
    output_storage = torch.full(
        (batch, heads, 2 * dim), -99.0, device="musa", dtype=x.dtype
    )
    out = output_storage[:, :, ::2]

    result = ssu_one_token_musa_triton(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        src,
        dst_state_batch_indices=dst,
        dt_bias=bias,
        z=torch.ones_like(x),
        out=out,
    )
    assert result is out
    expected_state = torch.full_like(state, 0.25)
    expected_state[2] = 0.34375
    expected_state[3] = 0.09375
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    assert torch.all(storage[:, :, 1::2, :] == 99)
    assert torch.all(storage[:, :, :, 1::2] == 99)
    assert torch.all(output_storage[:, :, 1::2] == -99)
    gate = torch.sigmoid(torch.tensor(1.0)).item()
    expected = torch.zeros_like(out)
    expected[0] = (0.34375 * 0.25 * n + 0.25) * gate
    # A destination slot of -1 suppresses the cache write, but the output is
    # still produced from the valid source state.
    expected[1] = (0.34375 * 0.25 * n + 0.25) * gate
    expected[2] = (0.09375 * 0.25 * n + 0.25) * gate
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [17, 65])
@pytest.mark.parametrize("n", [64, 128])
@pytest.mark.parametrize("rounds", [5, 10])
def test_dimension_tail_stochastic_counter_bits(dim, n, rounds):
    heads = 2
    storage = torch.zeros(3, heads, 2 * dim, 2 * n, device="musa", dtype=torch.float16)
    state = storage[:, :, ::2, ::2]
    x = torch.ones(1, heads, dim, device="musa", dtype=torch.bfloat16)
    dt = torch.tensor([[1.00048828125, -1.00048828125]], device="musa")[
        :, :, None
    ].expand_as(x)
    a = torch.zeros(heads, 1, 1, device="musa").expand(heads, dim, n)
    b = torch.ones(1, 1, n, device="musa", dtype=x.dtype)
    c = torch.zeros_like(b)
    skip = torch.zeros(heads, dim, device="musa")
    seed = 42 + 2**40
    ssu_one_token_musa_triton(
        state,
        x,
        dt,
        a,
        b,
        c,
        skip,
        torch.tensor([1], device="musa", dtype=torch.int32),
        dst_state_batch_indices=torch.tensor([2], device="musa", dtype=torch.int32),
        rand_seed=torch.tensor([seed], device="musa", dtype=torch.int64),
        philox_rounds=rounds,
    )
    expected = []
    group = 2 if n == 64 else 4
    for head, value in enumerate([1.00048828125, -1.00048828125]):
        value_bits = struct.unpack("<I", struct.pack("<f", value))[0]
        for d in range(dim):
            offset = state.stride(0) + head * state.stride(1) + d * state.stride(2)
            for j in range(0, n, group):
                words = philox4x32_words(seed, offset + j * state.stride(3), rounds)
                expected.extend(
                    cvt_rs_f16_bits(value_bits, words[k]) for k in range(group)
                )
    actual = (
        state[2].contiguous().cpu().view(torch.int16).to(torch.int32).flatten() & 0xFFFF
    )
    torch.testing.assert_close(
        actual, torch.tensor(expected, dtype=torch.int32), rtol=0, atol=0
    )
    assert torch.all(storage[:, :, 1::2, :] == 0)
    assert torch.all(storage[:, :, :, 1::2] == 0)
