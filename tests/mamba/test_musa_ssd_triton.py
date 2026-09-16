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
