"""Runtime gate for the opt-in native S5000 Simple-STP extension."""

import os

import pytest
import torch
import struct

from flashinfer.mamba.musa_ssu_native import musa_ssu_one_token_native
from flashinfer.mamba.musa_ssu_triton import ssu_one_token_musa_triton
from flashinfer.mamba.selective_state_update import selective_state_update
from .test_philox_cpu_oracle import cvt_rs_f16_bits, philox4x32_words

pytestmark = [
    pytest.mark.skipif(
        os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa",
        reason="requires a MUSA runtime",
    ),
    pytest.mark.skipif(
        os.environ.get("FLASHINFER_MUSA_SIMPLE_STP_NATIVE") != "1",
        reason="native extension is opt-in",
    ),
]


def test_native_simple_stp_matches_triton():
    torch.manual_seed(23)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    native_state, triton_state = state.clone(), state.clone()
    native_out, triton_out = torch.empty_like(x), torch.empty_like(x)
    args = (native_state, x, dt, a, b, c, d, slot, slot, None, None, True, -1, native_out, None, 0)
    musa_ssu_one_token_native(*args)
    ssu_one_token_musa_triton(
        triton_state, x, dt, a, b, c, d, slot,
        dt_softplus=True, out=triton_out,
    )
    torch.testing.assert_close(native_out, triton_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(native_state, triton_state, atol=3e-2, rtol=3e-2)


def test_native_simple_stp_rejects_materialized_a():
    torch.manual_seed(24)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    # Negating after expand materializes a full contiguous tensor, which the
    # native kernel must reject instead of treating as a tied broadcast.
    a = -torch.rand((64, 1, 1), device="musa", dtype=torch.float32).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="tied A/dt"):
        musa_ssu_one_token_native(
            state, x, dt, a, b, c, d, slot, slot, None, None, True, -1, None, None, 0
        )


def test_native_simple_stp_stochastic_matches_triton():
    torch.manual_seed(29)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    seed = torch.tensor([9123], device="musa", dtype=torch.int64)
    native_state, triton_state = state.clone(), state.clone()
    native_out, triton_out = torch.empty_like(x), torch.empty_like(x)
    musa_ssu_one_token_native(
        native_state, x, dt, a, b, c, d, slot, slot, None, None, True, -1,
        native_out, seed, 5,
    )
    ssu_one_token_musa_triton(
        triton_state, x, dt, a, b, c, d, slot, dt_softplus=True,
        out=triton_out, rand_seed=seed, philox_rounds=5,
    )
    torch.testing.assert_close(native_out, triton_out, atol=3e-2, rtol=3e-2)
    torch.testing.assert_close(native_state, triton_state, atol=3e-2, rtol=3e-2)


def test_public_selective_state_update_routes_native():
    torch.manual_seed(31)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    result = selective_state_update(
        state,
        x,
        dt,
        a,
        b,
        c,
        d,
        dt_softplus=True,
        state_batch_indices=slot,
        backend="flashinfer",
    )
    assert result.shape == x.shape
    assert result.dtype == x.dtype


def test_public_native_accepts_single_token_cu_seqlens():
    torch.manual_seed(37)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], device="musa", dtype=torch.int32)
    result = selective_state_update(
        state, x, dt, a, b, c, d, dt_softplus=True,
        state_batch_indices=slot, cu_seqlens=cu_seqlens, backend="flashinfer",
    )
    assert result.shape == x.shape


def test_public_native_rejects_noncanonical_single_token_cu_seqlens():
    torch.manual_seed(38)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    malformed = torch.tensor([10, 11], device="musa", dtype=torch.int32)
    with pytest.raises(ValueError, match=r"canonical \[0, 1\]"):
        selective_state_update(
            state,
            x,
            dt,
            a,
            b,
            c,
            d,
            dt_softplus=True,
            state_batch_indices=slot,
            cu_seqlens=malformed,
            backend="flashinfer",
        )


def test_native_direct_rejects_malformed_z_shape():
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)
    z = torch.randn((1, 64, 1), device="musa", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="z must match"):
        musa_ssu_one_token_native(
            state, x, dt, a, b, c, d, slot, slot, None, z, True, -1, None, None, 0
        )


def test_native_direct_rejects_multi_element_slot_indices():
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64,), device="musa", dtype=torch.bfloat16)
    slots = torch.tensor([0, 1], device="musa", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="one-element"):
        musa_ssu_one_token_native(
            state,
            x,
            dt,
            a,
            b,
            c,
            d,
            slots,
            slots,
            None,
            None,
            True,
            -1,
            None,
            None,
            0,
        )


def test_public_native_torch_compile_capture():
    torch.manual_seed(41)
    state = torch.randn((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.randn((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.randn((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (-torch.rand((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.randn((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.randn_like(b)
    d = torch.randn((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    slot = torch.zeros((1,), device="musa", dtype=torch.int32)

    def fn(s):
        return selective_state_update(
            s, x, dt, a, b, c, d, dt_softplus=True,
            state_batch_indices=slot, backend="flashinfer",
        )

    compiled = torch.compile(fn, backend="eager", fullgraph=False)
    result = compiled(state)
    assert result.shape == x.shape


def test_native_public_stochastic_cache_matches_cpu_oracle():
    state = torch.zeros((2, 64, 64, 128), device="musa", dtype=torch.float16)
    x = torch.ones((1, 64, 64), device="musa", dtype=torch.bfloat16)
    dt = torch.ones((1, 64, 1), device="musa", dtype=torch.float32).expand(1, 64, 64)
    a = (torch.zeros((64, 1, 1), device="musa", dtype=torch.float32)).expand(64, 64, 128)
    b = torch.ones((1, 8, 128), device="musa", dtype=torch.bfloat16)
    c = torch.zeros_like(b)
    d = torch.zeros((64, 1), device="musa", dtype=torch.float32).expand(64, 64)
    src = torch.zeros((1,), device="musa", dtype=torch.int32)
    dst = torch.ones((1,), device="musa", dtype=torch.int32)
    seed_value = 42 + 2**40
    seed = torch.tensor([seed_value], device="musa", dtype=torch.int64)
    selective_state_update(
        state, x, dt, a, b, c, d, state_batch_indices=src,
        dst_state_batch_indices=dst, rand_seed=seed, philox_rounds=5,
        dt_softplus=False, backend="flashinfer",
    )
    expected = []
    for head in range(64):
        bits = struct.unpack("<I", struct.pack("<f", 1.0))[0]
        for d_idx in range(64):
            base = head * 64 * 128 + d_idx * 128
            for offset in range(0, 128, 4):
                words = philox4x32_words(seed_value, base + offset, 5)
                expected.extend(cvt_rs_f16_bits(bits, word) for word in words)
    actual = state[1].view(torch.int16).cpu().to(torch.int32).flatten() & 0xFFFF
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.int32))
