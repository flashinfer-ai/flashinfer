import torch
import functools
import os
from flashinfer.utils import GPUArchitectureError
import pytest
import gc


@functools.cache
def get_device_properties(device: torch.device):
    return torch.cuda.get_device_properties(device)


def skip_on_gpu_arch_error(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GPUArchitectureError as e:
            pytest.skip(str(e))

    return wrapper


def clear_cuda_cache(device: torch.device) -> None:
    total_memory = get_device_properties(device).total_memory
    reserved_memory = torch.cuda.memory_reserved()

    # FLASHINFER_TEST_MEMORY_THRESHOLD: threshold for PyTorch reserved memory usage (default: 0.9)
    threshold = float(os.environ.get("FLASHINFER_TEST_MEMORY_THRESHOLD", "0.9"))

    if reserved_memory > threshold * total_memory:
        gc.collect()
        torch.cuda.empty_cache()


def assert_close_with_mismatch_tolerance(
    actual: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    max_mismatched_elements: int = 0,
):
    """
    Asserts that two tensors are close, allowing for a specified number of mismatched elements.
    This function correctly implements the same logic as torch.isclose.
    """
    # Ensure tensors are float for comparison
    actual_float = actual.float()
    expected_float = expected.float()

    # This is the core logic from torch.isclose
    # A mismatch occurs if the difference is greater than the combined tolerance
    mismatched = torch.abs(actual_float - expected_float) > (
        atol + rtol * torch.abs(expected_float)
    )

    num_mismatched = torch.sum(mismatched).item()

    if num_mismatched > max_mismatched_elements:
        # For a helpful error message, let's find the worst offenders
        actual_flat = actual_float.flatten()
        expected_flat = expected_float.flatten()
        abs_diff = torch.abs(actual_flat - expected_flat)

        # Calculate relative difference only where expected is not zero to avoid division by zero
        # Add a small epsilon to the denominator for stability
        rel_diff = abs_diff / (torch.abs(expected_flat) + 1e-12)

        total_elements = actual_flat.numel()

        raise AssertionError(
            f"Tensors are not close enough!\n"
            f"Mismatched elements: {num_mismatched} / {total_elements} "
            f"({100.0 * num_mismatched / total_elements:.2f}%)\n"
            f"Allowed mismatched elements: {max_mismatched_elements}, but found {num_mismatched}.\n"
            f"Greatest absolute difference: {torch.max(abs_diff).item():.4g} (atol={atol})\n"
            f"Greatest relative difference: {torch.max(rel_diff).item():.4g} (rtol={rtol})"
        )


def assert_close_chunked(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    chunk_rows: int = 4096,
    **kwargs,
):
    """Memory-frugal drop-in for torch.testing.assert_close on large tensors.

    torch.testing.assert_close allocates several full-size temporaries inside
    torch.isclose; on multi-GiB operands that transient spike is enough to OOM
    a 24 GB CI GPU. Comparing in row chunks along dim 0 bounds the transient
    to the chunk size while keeping identical pass/fail semantics. Extra
    keyword arguments (e.g. equal_nan, check_dtype) are forwarded to
    torch.testing.assert_close.
    """
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} vs {expected.shape}")
    if actual.ndim == 0 or actual.numel() == 0:
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, **kwargs)
        return
    for start in range(0, actual.shape[0], chunk_rows):
        end = min(start + chunk_rows, actual.shape[0])
        torch.testing.assert_close(
            actual[start:end],
            expected[start:end],
            rtol=rtol,
            atol=atol,
            msg=lambda m, s=start, e=end: f"rows [{s}:{e}]: {m}",
            **kwargs,
        )


def ref_single_prefill(q, k, v, causal=False):
    """FP64 reference of single-request prefill attention (NHD layout).

    Returns (output, lse) where lse is the base-2 logsumexp; fully masked rows
    get lse = -inf and zero output.
    """
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    group_size = num_qo_heads // num_kv_heads
    scale = head_dim**-0.5

    q64 = q.float().to(torch.float64)
    k64 = k.float().to(torch.float64).repeat_interleave(group_size, dim=1)
    v64 = v.float().to(torch.float64).repeat_interleave(group_size, dim=1)

    scores = torch.einsum("qhd,khd->hqk", q64, k64) * scale
    if causal:
        q_pos = torch.arange(qo_len, dtype=torch.float64, device=q.device)
        k_pos = torch.arange(kv_len, dtype=torch.float64, device=q.device)
        mask = k_pos[None, :] - (kv_len - qo_len) > q_pos[:, None]
        scores = scores.masked_fill(mask[None, :, :], float("-inf"))

    row_max = scores.amax(dim=-1)
    weights = torch.softmax(scores, dim=-1)
    # softmax yields NaN on fully masked rows (all -inf scores); their output
    # must be the zero vector.
    weights = torch.nan_to_num(weights, nan=0.0)
    out = torch.einsum("hqk,khd->qhd", weights.to(torch.float64), v64)
    sum_exp = torch.exp(scores - row_max[..., None]).sum(dim=-1)
    lse = row_max + torch.log2(sum_exp)
    lse = torch.where(
        torch.isneginf(row_max), torch.full_like(row_max, float("-inf")), lse
    )
    return out.to(q.dtype), lse.t().to(torch.float32)
