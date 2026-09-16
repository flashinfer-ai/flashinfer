"""Native MUSA SSD checks for ragged sequences and cache/output contracts."""

import os

import pytest

torch = pytest.importorskip("torch")
if os.environ.get("FLASHINFER_MAMBA_TEST_DEVICE") != "musa":
    pytest.skip("set FLASHINFER_MAMBA_TEST_DEVICE=musa", allow_module_level=True)

from flashinfer.mamba.musa_reference import (  # noqa: E402
    ssd_combined_fwd_varlen_musa_reference,
)
from flashinfer.mamba.ssd_combined import ssd_combined_fwd_varlen  # noqa: E402


def _tail_chunk_metadata(lengths, chunk_size, device):
    """Build packed physical-chunk metadata, including a final short chunk."""
    cu_seqlens = [0]
    cu_chunks = [0]
    sequence_ids = []
    last_chunks = []
    total = 0
    for sequence, length in enumerate(lengths):
        cu_seqlens.append(cu_seqlens[-1] + length)
        sequence_end = total + length
        while total < sequence_end:
            total = min(total + chunk_size, sequence_end)
            cu_chunks.append(total)
            sequence_ids.append(sequence)
        last_chunks.append(len(sequence_ids) - 1)
    return (
        torch.tensor(cu_seqlens, device=device, dtype=torch.int32),
        torch.tensor(cu_chunks, device=device, dtype=torch.int32),
        torch.tensor(last_chunks, device=device, dtype=torch.int32),
        torch.tensor(sequence_ids, device=device, dtype=torch.int32),
    )


@pytest.mark.parametrize("intermediate", [False, True])
@pytest.mark.parametrize("state_dtype", [torch.float16, torch.float32])
def test_ragged_states_and_strided_output(monkeypatch, intermediate, state_dtype):
    monkeypatch.setenv("VLLM_MUSA_FLASHINFER_SSD", "1")
    torch.manual_seed(11)
    device = torch.device("musa")
    tokens, heads, dim, dstate, groups, chunk = 26, 4, 16, 32, 2, 16
    x = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16) * 0.2
    dt = torch.randn(tokens, heads, device=device) * 0.05
    A = -torch.rand(heads, device=device) - 1
    B = torch.randn(tokens, groups, dstate, device=device, dtype=torch.bfloat16) * 0.2
    C = torch.randn_like(B) * 0.2
    z = torch.randn_like(x)
    D = torch.randn(heads, dim, device=device)
    bias = torch.randn(heads, device=device)
    initial = (
        torch.randn(2, heads, dim, dstate, device=device, dtype=torch.float16) * 0.1
    )
    cu_seqlens = torch.tensor([0, 19, 26], device=device, dtype=torch.int32)
    cu_chunks = torch.tensor([0, 16, 19, 26], device=device, dtype=torch.int32)
    last_chunks = torch.tensor([1, 2], device=device, dtype=torch.int32)
    sequence_ids = torch.tensor([0, 0, 1], device=device, dtype=torch.int32)
    output_storage = torch.empty(tokens, heads, dim * 2, device=device, dtype=x.dtype)
    output = output_storage[..., ::2]
    expected_output = torch.empty_like(x)
    args = (x, dt, A, B, C, chunk, cu_seqlens, cu_chunks, last_chunks, sequence_ids)
    kwargs = dict(
        D=D,
        z=z,
        dt_bias=bias,
        dt_softplus=True,
        initial_states=initial,
        state_dtype=state_dtype,
        return_intermediate_states=intermediate,
    )
    expected_states = ssd_combined_fwd_varlen_musa_reference(
        *args, out=expected_output, **kwargs
    )
    states = ssd_combined_fwd_varlen(*args, out=output, **kwargs)
    torch.musa.synchronize()
    assert states.dtype == state_dtype
    torch.testing.assert_close(output, expected_output, rtol=0.05, atol=0.02)
    torch.testing.assert_close(states, expected_states, rtol=0.05, atol=0.02)


@pytest.mark.parametrize("dstate", [32, 256])
def test_no_initial_state_and_strided_metadata(monkeypatch, dstate):
    monkeypatch.setenv("VLLM_MUSA_FLASHINFER_SSD", "1")
    torch.manual_seed(17)
    device = torch.device("musa")
    tokens, heads, dim, groups, chunk = 26, 4, 16, 2, 16
    x = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16) * 0.1
    dt = torch.randn(tokens, heads, device=device) * 0.05
    A = -torch.ones(heads, device=device)
    B = torch.randn(tokens, groups, dstate, device=device, dtype=x.dtype) * 0.1
    C = torch.randn_like(B) * 0.1
    D = torch.randn(heads, device=device)

    def strided(values):
        tensor = torch.tensor(values, device=device, dtype=torch.int32)
        return torch.stack((tensor, torch.full_like(tensor, 999)), dim=1).flatten()[::2]

    metadata = [strided(v) for v in ([0, 19, 26], [0, 16, 19, 26], [1, 2], [0, 0, 1])]
    args = (x, dt, A, B, C, chunk, *metadata)
    kwargs = dict(D=D, dt_softplus=True, state_dtype=torch.float32)
    output = torch.empty_like(x, dtype=torch.float32)
    expected_output = torch.empty_like(output)
    expected = ssd_combined_fwd_varlen_musa_reference(
        *args, out=expected_output, **kwargs
    )
    actual = ssd_combined_fwd_varlen(*args, out=output, **kwargs)
    torch.musa.synchronize()
    torch.testing.assert_close(output, expected_output, rtol=0.05, atol=0.02)
    torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.02)


@pytest.mark.parametrize("lengths", [[127, 128, 3841], [4095], [4097]])
def test_tail_chunks_match_reference(monkeypatch, lengths):
    """Keep packed tail chunks and final/intermediate state boundaries correct."""
    monkeypatch.setenv("VLLM_MUSA_FLASHINFER_SSD", "1")
    torch.manual_seed(7000 + sum(lengths))
    device = torch.device("musa")
    chunk_size = 128
    tokens, heads, dim, dstate, groups = sum(lengths), 4, 8, 16, 2
    cu, cu_chunks, last_chunks, sequence_ids = _tail_chunk_metadata(
        lengths, chunk_size, device
    )
    x = torch.randn(tokens, heads, dim, device=device, dtype=torch.bfloat16)
    dt = torch.randn(tokens, heads, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(heads, device=device, dtype=torch.float32) - 1.0
    B = torch.randn(tokens, groups, dstate, device=device, dtype=torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(heads, dim, device=device, dtype=torch.float32)
    z = torch.randn_like(x)
    bias = torch.randn(heads, device=device, dtype=torch.float32) - 2.0
    initial = torch.randn(
        len(lengths), heads, dim, dstate, device=device, dtype=torch.bfloat16
    )
    args = (x, dt, A, B, C, chunk_size, cu, cu_chunks, last_chunks, sequence_ids)
    for return_intermediate_states in (False, True):
        expected_out = torch.empty_like(x)
        expected = ssd_combined_fwd_varlen_musa_reference(
            *args,
            out=expected_out,
            D=D,
            z=z,
            dt_bias=bias,
            dt_softplus=True,
            initial_states=initial,
            return_intermediate_states=return_intermediate_states,
            state_dtype=initial.dtype,
        )
        actual_out = torch.empty_like(x)
        actual = ssd_combined_fwd_varlen(
            *args,
            out=actual_out,
            D=D,
            z=z,
            dt_bias=bias,
            dt_softplus=True,
            initial_states=initial,
            return_intermediate_states=return_intermediate_states,
            state_dtype=initial.dtype,
        )
        torch.musa.synchronize()
        torch.testing.assert_close(actual_out, expected_out, rtol=0.05, atol=0.03)
        torch.testing.assert_close(actual, expected, rtol=0.05, atol=0.03)
