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
