"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

import flashinfer


def compute_topk_accuracy(test_indices, ref_indices, batch_size, k):
    """Compute accuracy as intersection ratio between test and reference top-k indices."""
    total_intersection = 0
    for i in range(batch_size):
        ref_set = set(ref_indices[i].cpu().numpy())
        test_set = set(test_indices[i].cpu().numpy())
        total_intersection += len(ref_set & test_set)
    return total_intersection / (batch_size * k)


def verify_topk_correctness(logits, values, indices, k):
    """Verify that all returned values are truly in the top-k.

    Returns True if all values are >= the k-th largest value in each row.
    This is a more robust check than comparing indices, since tie-breaking
    can differ between implementations.
    """
    batch_size = logits.size(0)
    for i in range(batch_size):
        # Get the k-th largest value (ground truth threshold)
        kth_largest = torch.kthvalue(-logits[i], k).values.item() * -1
        # All returned values should be >= this threshold
        if values[i].min().item() < kth_largest - 1e-6:
            return False
    return True


@pytest.mark.parametrize("batch_size", [1, 16, 64])
@pytest.mark.parametrize("vocab_size", [32000, 65536, 128512])
@pytest.mark.parametrize("k", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k(batch_size, vocab_size, k, dtype):
    """Test top_k returns correct values and indices."""
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=dtype)

    # flashinfer top_k
    values, indices = flashinfer.top_k(logits, k)

    # Reference: torch.topk
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    # Check output shapes
    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Check dtypes
    assert values.dtype == dtype
    assert indices.dtype == torch.int64

    # Verify values match the gathered indices
    gathered_values = torch.gather(logits, dim=-1, index=indices)
    torch.testing.assert_close(values, gathered_values)

    # Check accuracy of indices
    accuracy = compute_topk_accuracy(indices.int(), ref_indices.int(), batch_size, k)
    # Accuracy depends on vocab size, k, and data distribution
    # Random Gaussian data can have many values close to each other at boundaries
    min_accuracy = 0.98
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("vocab_size", [32000, 65536])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_top_k_sorted(batch_size, vocab_size, k, dtype):
    """Test top_k with sorted=True returns sorted values."""
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=dtype)

    # flashinfer top_k with sorted=True
    values, indices = flashinfer.top_k(logits, k, sorted=True)

    # Reference: torch.topk with sorted=True
    ref_values, ref_indices = torch.topk(logits, k, dim=-1, sorted=True)

    # Check output shapes
    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Verify values are sorted in descending order
    for i in range(batch_size):
        row_values = values[i]
        assert torch.all(row_values[:-1] >= row_values[1:]), (
            f"Row {i} values not sorted in descending order"
        )

    # Verify values match the gathered indices
    gathered_values = torch.gather(logits, dim=-1, index=indices)
    torch.testing.assert_close(values, gathered_values)

    # Check accuracy of indices
    accuracy = compute_topk_accuracy(indices.int(), ref_indices.int(), batch_size, k)
    min_accuracy = 0.90
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("vocab_size", [32000, 65536])
@pytest.mark.parametrize("k", [256])
def test_top_k_single_batch(vocab_size, k):
    """Test top_k with batch_size=1 (common inference case)."""
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k
    values, indices = flashinfer.top_k(logits, k)

    # Reference: torch.topk
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    # Check output shape
    assert values.shape == (1, k)
    assert indices.shape == (1, k)

    # Check accuracy
    accuracy = compute_topk_accuracy(indices, ref_indices, 1, k)
    assert accuracy >= 0.99, f"Accuracy {accuracy:.4f} < 0.99"


@pytest.mark.parametrize("batch_size", [64, 128])
@pytest.mark.parametrize("vocab_size", [65536, 128512])
@pytest.mark.parametrize("k", [256])
def test_top_k_large_batch(batch_size, vocab_size, k):
    """Test top_k with large batch sizes (multi-CTA path)."""
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k (should use multi-CTA path for large vocab)
    values, indices = flashinfer.top_k(logits, k)

    # Reference: torch.topk
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    # Check output shape
    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Check accuracy
    accuracy = compute_topk_accuracy(indices, ref_indices, batch_size, k)
    assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} < 0.98"


@pytest.mark.parametrize("k", [256, 1024, 2048])
def test_top_k_large_k(k):
    """Test top_k with larger k values."""
    batch_size = 4
    vocab_size = 32000

    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k
    values, indices = flashinfer.top_k(logits, k)

    # Reference: torch.topk
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    # Check output shape
    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Check accuracy
    accuracy = compute_topk_accuracy(indices, ref_indices, batch_size, k)
    assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} < 0.98"


def test_top_k_vs_torch_topk_compatibility():
    """Test that flashinfer.top_k can be used as a drop-in replacement for torch.topk."""
    batch_size = 4
    vocab_size = 32000
    k = 256

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k
    fi_values, fi_indices = flashinfer.top_k(logits, k, sorted=True)

    # torch.topk
    torch_values, torch_indices = torch.topk(logits, k, dim=-1, sorted=True)

    # Check shapes match
    assert fi_values.shape == torch_values.shape
    assert fi_indices.shape == torch_indices.shape

    # Check dtypes
    assert fi_values.dtype == torch_values.dtype
    # Note: flashinfer returns int64, torch returns int64
    assert fi_indices.dtype == torch_indices.dtype

    # Check that the selected values are the same (may be in different order for unsorted)
    # For sorted case, the order should match for identical values
    accuracy = compute_topk_accuracy(
        fi_indices.int(), torch_indices.int(), batch_size, k
    )
    assert accuracy >= 0.98


if __name__ == "__main__":
    test_top_k(4, 32000, 256, torch.float32)
    test_top_k_sorted(4, 32000, 256, torch.float32)
    test_top_k_large_batch(64, 128512, 256)
