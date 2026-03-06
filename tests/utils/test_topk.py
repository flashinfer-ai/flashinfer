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

import os

import pytest
import torch

import flashinfer
from flashinfer.topk import can_implement_filtered_topk
from flashinfer.utils import get_compute_capability


@pytest.fixture
def set_topk_algo():
    """Fixture to set and reset FLASHINFER_TOPK_ALGO environment variable."""
    original_value = os.environ.get("FLASHINFER_TOPK_ALGO", None)

    def _set_algo(algo: str):
        if algo == "auto":
            os.environ.pop("FLASHINFER_TOPK_ALGO", None)
        else:
            os.environ["FLASHINFER_TOPK_ALGO"] = algo

    yield _set_algo

    # Restore original value
    if original_value is None:
        os.environ.pop("FLASHINFER_TOPK_ALGO", None)
    else:
        os.environ["FLASHINFER_TOPK_ALGO"] = original_value


def compute_topk_accuracy(test_indices, ref_indices, batch_size, k):
    """Compute accuracy as intersection ratio between test and reference top-k indices."""
    total_intersection = 0
    for i in range(batch_size):
        ref_set = set(ref_indices[i].cpu().numpy())
        test_set = set(test_indices[i].cpu().numpy())
        total_intersection += len(ref_set & test_set)
    return total_intersection / (batch_size * k)


def _require_sm80_for_bf16():
    major, _ = get_compute_capability(torch.device("cuda"))
    if major < 8:
        pytest.skip("BF16 requires SM80+")


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


# ===================== Fused TopK Transform Tests =====================


def reference_page_table_transform(
    scores: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_to_batch: torch.Tensor = None,
) -> torch.Tensor:
    """Reference implementation for page table transform using torch.topk."""
    num_rows = scores.size(0)
    scores.size(1)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    for i in range(num_rows):
        length = lengths[i].item()
        batch_idx = row_to_batch[i].item() if row_to_batch is not None else i

        if length <= k:
            # Trivial case: just copy first `length` entries
            output[i, :length] = src_page_table[batch_idx, :length]
        else:
            # Get top-k indices
            row_scores = scores[i, :length]
            _, topk_indices = torch.topk(row_scores.float(), k)
            # Gather from page table
            output[i] = src_page_table[batch_idx, topk_indices.long()]

    return output


def reference_ragged_transform(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Reference implementation for ragged transform using torch.topk."""
    num_rows = scores.size(0)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    for i in range(num_rows):
        length = lengths[i].item()
        offset = offsets[i].item()

        if length <= k:
            # Trivial case: indices are [offset, offset+1, ..., offset+length-1]
            output[i, :length] = torch.arange(
                offset, offset + length, dtype=torch.int32, device=device
            )
        else:
            # Get top-k indices
            row_scores = scores[i, :length]
            _, topk_indices = torch.topk(row_scores.float(), k)
            # Add offset
            output[i] = topk_indices.int() + offset

    return output


def compute_transform_accuracy(test_output, ref_output, num_rows, k):
    """Compute accuracy for transform outputs, handling -1 padding correctly."""
    total_matches = 0
    total_valid = 0

    for i in range(num_rows):
        # Get valid entries (not -1)
        test_valid_mask = test_output[i] != -1
        ref_valid_mask = ref_output[i] != -1

        # Both should have same number of valid entries
        test_set = set(test_output[i][test_valid_mask].cpu().numpy())
        ref_set = set(ref_output[i][ref_valid_mask].cpu().numpy())

        # Count intersection
        total_matches += len(test_set & ref_set)
        total_valid += len(ref_set)

    return total_matches / total_valid if total_valid > 0 else 1.0


@pytest.mark.parametrize("num_rows", [1, 8, 32])
@pytest.mark.parametrize("max_len", [1024, 4096, 8192])
@pytest.mark.parametrize("k", [64, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_page_table_transform(num_rows, max_len, k, dtype):
    """Test top_k_page_table_transform returns correct page table entries."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    torch.manual_seed(42)
    device = "cuda"

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table (values 0 to 10000)
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    # Reference implementation
    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)

    # Check output shape
    assert output.shape == (num_rows, k), (
        f"Expected shape {(num_rows, k)}, got {output.shape}"
    )
    assert output.dtype == torch.int32

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [1, 8, 32])
@pytest.mark.parametrize("max_len", [1024, 4096, 8192])
@pytest.mark.parametrize("k", [64, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_ragged_transform(num_rows, max_len, k, dtype):
    """Test top_k_ragged_transform returns correct indices with offsets."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    torch.manual_seed(42)
    device = "cuda"

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate offsets (cumulative sum style)
    offsets = torch.arange(
        0, num_rows * max_len, max_len, device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Reference implementation
    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    # Check output shape
    assert output.shape == (num_rows, k), (
        f"Expected shape {(num_rows, k)}, got {output.shape}"
    )
    assert output.dtype == torch.int32

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [1, 8, 32])
@pytest.mark.parametrize("max_len", [1024, 4096, 8192])
@pytest.mark.parametrize("k", [64, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_top_k_ragged_transform_out_of_length(num_rows, max_len, k, dtype):
    """Test top_k_ragged_transform returns correct indices with offsets."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    torch.manual_seed(42)
    device = "cuda"

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate naive offsets (cumulative sum style)
    offsets = torch.zeros(num_rows, device=device, dtype=torch.int32)

    # Random in [1, max_len]
    lengths = torch.randint(
        1, max_len + 1, (num_rows,), device=device, dtype=torch.int32
    )

    # Test flashinfer implementation
    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Reference implementation
    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    # Check output shape
    assert output.shape == (num_rows, k), (
        f"Expected shape {(num_rows, k)}, got {output.shape}"
    )
    assert output.dtype == torch.int32

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"
    # Check out of length
    valid_min = offsets
    valid_max = offsets + lengths
    output = output.clamp_min(0)
    assert torch.all((output >= valid_min[:, None]) & (output < valid_max[:, None])), (
        f"Out of length Error. {valid_min=}, {valid_max=}, {output.max(dim=1).values=}, {output.min(dim=1).values=}"
    )


@pytest.mark.parametrize("num_rows", [4, 16])
@pytest.mark.parametrize("max_len", [2048])
@pytest.mark.parametrize("k", [256, 512])
def test_page_table_transform_trivial_case(num_rows, max_len, k):
    """Test page table transform when length <= k (trivial copy case)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )

    # Set lengths less than or equal to k
    lengths = torch.randint(1, k + 1, (num_rows,), device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    # Verify manually
    for i in range(num_rows):
        length = lengths[i].item()
        # First `length` entries should match page table
        expected = src_page_table[i, :length]
        actual_valid = output[i, :length]
        assert torch.equal(expected, actual_valid), (
            f"Row {i}: expected {expected}, got {actual_valid}"
        )
        # Remaining should be -1
        if length < k:
            remaining = output[i, length:]
            assert torch.all(remaining == -1), f"Row {i}: padding should be -1"


@pytest.mark.parametrize("num_rows", [4, 16])
@pytest.mark.parametrize("max_len", [2048])
@pytest.mark.parametrize("k", [256, 512])
def test_ragged_transform_trivial_case(num_rows, max_len, k):
    """Test ragged transform when length <= k (trivial sequential indices case)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate offsets
    offsets = torch.arange(0, num_rows * 1000, 1000, device=device, dtype=torch.int32)

    # Set lengths less than or equal to k
    lengths = torch.randint(1, k + 1, (num_rows,), device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Verify manually
    for i in range(num_rows):
        length = lengths[i].item()
        offset = offsets[i].item()
        # First `length` entries should be [offset, offset+1, ..., offset+length-1]
        expected = torch.arange(
            offset, offset + length, dtype=torch.int32, device=device
        )
        actual_valid = output[i, :length]
        assert torch.equal(expected, actual_valid), (
            f"Row {i}: expected {expected}, got {actual_valid}"
        )
        # Remaining should be -1
        if length < k:
            remaining = output[i, length:]
            assert torch.all(remaining == -1), f"Row {i}: padding should be -1"


@pytest.mark.parametrize("num_rows", [8])
@pytest.mark.parametrize("max_len", [4096])
@pytest.mark.parametrize("k", [256])
def test_page_table_transform_with_row_to_batch(num_rows, max_len, k):
    """Test page table transform with row_to_batch mapping."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Batch size is smaller than num_rows (multiple rows per batch)
    batch_size = num_rows // 2

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table for batch_size
    src_page_table = torch.randint(
        0, 10000, (batch_size, max_len), device=device, dtype=torch.int32
    )

    # Map rows to batches (2 rows per batch)
    row_to_batch = torch.arange(
        batch_size, device=device, dtype=torch.int32
    ).repeat_interleave(2)

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k, row_to_batch
    )

    # Reference implementation
    ref_output = reference_page_table_transform(
        scores, src_page_table, lengths, k, row_to_batch
    )

    # Check output shape
    assert output.shape == (num_rows, k)

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [4, 16])
@pytest.mark.parametrize("max_len", [4096, 131072])
@pytest.mark.parametrize("k", [256])
def test_page_table_transform_variable_lengths(num_rows, max_len, k):
    """Test page table transform with variable sequence lengths."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )

    # Variable lengths - some trivial (< k), some large (> k)
    lengths = torch.tensor(
        [k // 2, max_len, k, max_len // 2, k * 2] * (num_rows // 5 + 1),
        device=device,
        dtype=torch.int32,
    )[:num_rows]
    lengths = lengths.clamp(max=max_len)

    # Test flashinfer implementation
    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    # Reference implementation
    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [4, 16])
@pytest.mark.parametrize("max_len", [4096])
@pytest.mark.parametrize("k", [256])
def test_ragged_transform_variable_lengths(num_rows, max_len, k):
    """Test ragged transform with variable sequence lengths."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate variable offsets
    offsets = torch.tensor(
        [i * 5000 for i in range(num_rows)], device=device, dtype=torch.int32
    )

    # Variable lengths - some trivial (< k), some large (> k)
    lengths = torch.tensor(
        [k // 2, max_len, k, max_len // 2, k * 2] * (num_rows // 5 + 1),
        device=device,
        dtype=torch.int32,
    )[:num_rows]
    lengths = lengths.clamp(max=max_len)

    # Test flashinfer implementation
    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Reference implementation
    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [64, 128])
@pytest.mark.parametrize("max_len", [8192, 16384])
@pytest.mark.parametrize("k", [256, 512])
def test_page_table_transform_large_scale(num_rows, max_len, k):
    """Test page table transform with large inputs (multi-CTA path)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    # Reference implementation
    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)

    # Check output shape
    assert output.shape == (num_rows, k)

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("num_rows", [64, 128])
@pytest.mark.parametrize("max_len", [8192, 16384])
@pytest.mark.parametrize("k", [256, 512])
def test_ragged_transform_large_scale(num_rows, max_len, k):
    """Test ragged transform with large inputs (multi-CTA path)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float16

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate offsets
    offsets = torch.arange(
        0, num_rows * max_len, max_len, device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # Test flashinfer implementation
    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Reference implementation
    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    # Check output shape
    assert output.shape == (num_rows, k)

    # Check accuracy
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


def test_page_table_transform_single_row():
    """Test page table transform with single row (batch_size=1)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    max_len = 4096
    k = 256

    scores = torch.randn(1, max_len, device=device, dtype=dtype)
    src_page_table = torch.randint(
        0, 10000, (1, max_len), device=device, dtype=torch.int32
    )
    lengths = torch.tensor([max_len], device=device, dtype=torch.int32)

    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)

    accuracy = compute_transform_accuracy(output, ref_output, 1, k)
    assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} < 0.98"


def test_ragged_transform_single_row():
    """Test ragged transform with single row (batch_size=1)."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    max_len = 4096
    k = 256

    scores = torch.randn(1, max_len, device=device, dtype=dtype)
    offsets = torch.tensor([0], device=device, dtype=torch.int32)
    lengths = torch.tensor([max_len], device=device, dtype=torch.int32)

    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    accuracy = compute_transform_accuracy(output, ref_output, 1, k)
    assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} < 0.98"


def test_page_table_transform_correctness_exact():
    """Test that page table transform produces values that exist in the page table."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    num_rows = 8
    max_len = 2048
    k = 256

    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)
    # Use unique page table values for easy verification
    src_page_table = torch.arange(
        num_rows * max_len, device=device, dtype=torch.int32
    ).reshape(num_rows, max_len)
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)

    # Verify all output values exist in the corresponding page table row
    for i in range(num_rows):
        page_values = set(src_page_table[i].cpu().numpy())
        output_values = output[i][output[i] != -1].cpu().numpy()
        for val in output_values:
            assert val in page_values, f"Row {i}: output value {val} not in page table"


def test_ragged_transform_offset_correctness():
    """Test that ragged transform correctly applies offsets."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32
    num_rows = 8
    max_len = 2048
    k = 256

    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)
    offsets = torch.tensor(
        [i * 10000 for i in range(num_rows)], device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # Verify all output values are in the correct range [offset, offset + length)
    for i in range(num_rows):
        offset = offsets[i].item()
        length = lengths[i].item()
        output_values = output[i][output[i] != -1].cpu().numpy()
        for val in output_values:
            assert offset <= val < offset + length, (
                f"Row {i}: output value {val} not in range [{offset}, {offset + length})"
            )


# ===================== SGLang-style Reference Implementation =====================


def sglang_style_topk_page_table_transform(
    scores: torch.Tensor,
    src_page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    cu_seqlens_q: torch.Tensor = None,
) -> torch.Tensor:
    """
    Reference implementation mimicking SGLang's topk_transform_prefill logic.

    SGLang uses a radix-based top-k selection followed by page table gather.
    This PyTorch implementation follows the same semantic behavior.

    For prefill mode with cu_seqlens_q:
      - scores shape: [expanded_bs, max_len]
      - src_page_table shape: [prefill_bs, max_len]
      - Need to map expanded rows to prefill batch indices via cu_seqlens_q
    """
    num_rows = scores.size(0)
    scores.size(1)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    if cu_seqlens_q is not None:
        # Prefill mode: map expanded rows back to prefill batch
        prefill_bs = cu_seqlens_q.size(0) - 1
        cu_seqlens = cu_seqlens_q.cpu().numpy()

        for i in range(num_rows):
            length = lengths[i].item()
            # Find which prefill batch this row belongs to
            batch_idx = 0
            for b in range(prefill_bs):
                if cu_seqlens[b] <= i < cu_seqlens[b + 1]:
                    batch_idx = b
                    break

            if length <= k:
                output[i, :length] = src_page_table[batch_idx, :length]
            else:
                row_scores = scores[i, :length].float()
                _, topk_indices = torch.topk(row_scores, k)
                output[i] = src_page_table[batch_idx, topk_indices.long()]
    else:
        # Decode mode: 1:1 mapping
        for i in range(num_rows):
            length = lengths[i].item()
            if length <= k:
                output[i, :length] = src_page_table[i, :length]
            else:
                row_scores = scores[i, :length].float()
                _, topk_indices = torch.topk(row_scores, k)
                output[i] = src_page_table[i, topk_indices.long()]

    return output


def sglang_style_topk_ragged_transform(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Reference implementation mimicking SGLang's topk_transform_prefill_ragged logic.

    For each row:
      output[i] = topk_indices[i] + offsets[i]
    """
    num_rows = scores.size(0)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    for i in range(num_rows):
        length = lengths[i].item()
        offset = offsets[i].item()

        if length <= k:
            output[i, :length] = torch.arange(
                offset, offset + length, dtype=torch.int32, device=device
            )
        else:
            row_scores = scores[i, :length].float()
            _, topk_indices = torch.topk(row_scores, k)
            output[i] = topk_indices.int() + offset

    return output


@pytest.mark.parametrize("num_rows", [8, 32])
@pytest.mark.parametrize("max_len", [4096, 8192])
@pytest.mark.parametrize("k", [256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_compare_with_sglang_style_page_table(num_rows, max_len, k, dtype):
    """Compare flashinfer's page table transform with SGLang-style reference."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    torch.manual_seed(42)
    device = "cuda"

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate random page table
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # FlashInfer implementation
    fi_output = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k
    )

    # SGLang-style reference
    sgl_output = sglang_style_topk_page_table_transform(
        scores, src_page_table, lengths, k
    )

    # Compare results
    accuracy = compute_transform_accuracy(fi_output, sgl_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, (
        f"FlashInfer vs SGLang-style accuracy {accuracy:.4f} < {min_accuracy}"
    )


@pytest.mark.parametrize("num_rows", [8, 32])
@pytest.mark.parametrize("max_len", [4096, 8192])
@pytest.mark.parametrize("k", [256, 512, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_compare_with_sglang_style_ragged(num_rows, max_len, k, dtype):
    """Compare flashinfer's ragged transform with SGLang-style reference."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    torch.manual_seed(42)
    device = "cuda"

    # Generate random scores
    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)

    # Generate offsets
    offsets = torch.arange(
        0, num_rows * max_len, max_len, device=device, dtype=torch.int32
    )

    # All rows have full length
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    # FlashInfer implementation
    fi_output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)

    # SGLang-style reference
    sgl_output = sglang_style_topk_ragged_transform(scores, offsets, lengths, k)

    # Compare results
    accuracy = compute_transform_accuracy(fi_output, sgl_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, (
        f"FlashInfer vs SGLang-style accuracy {accuracy:.4f} < {min_accuracy}"
    )


@pytest.mark.parametrize("num_rows", [8])
@pytest.mark.parametrize("max_len", [4096])
@pytest.mark.parametrize("k", [256, 512])
def test_compare_with_sglang_style_prefill_mode(num_rows, max_len, k):
    """Compare with SGLang-style in prefill mode with cu_seqlens_q mapping."""
    torch.manual_seed(42)
    device = "cuda"
    dtype = torch.float32

    # Expanded batch: multiple query heads per prefill batch
    prefill_bs = num_rows // 2
    expanded_bs = num_rows

    # Generate random scores for expanded batch
    scores = torch.randn(expanded_bs, max_len, device=device, dtype=dtype)

    # Page table is per prefill batch
    src_page_table = torch.randint(
        0, 10000, (prefill_bs, max_len), device=device, dtype=torch.int32
    )

    # cu_seqlens_q: maps expanded rows to prefill batch
    # Each prefill batch has 2 rows in expanded batch
    cu_seqlens_q = torch.arange(0, expanded_bs + 1, 2, device=device, dtype=torch.int32)

    # Lengths for all rows
    lengths = torch.full((expanded_bs,), max_len, device=device, dtype=torch.int32)

    # FlashInfer: use row_to_batch mapping
    row_to_batch = torch.arange(
        prefill_bs, device=device, dtype=torch.int32
    ).repeat_interleave(2)
    fi_output = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k, row_to_batch
    )

    # SGLang-style reference with cu_seqlens_q
    sgl_output = sglang_style_topk_page_table_transform(
        scores, src_page_table, lengths, k, cu_seqlens_q
    )

    # Compare results
    accuracy = compute_transform_accuracy(fi_output, sgl_output, expanded_bs, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, (
        f"FlashInfer vs SGLang-style prefill mode accuracy {accuracy:.4f} < {min_accuracy}"
    )


# ===================== Algorithm-specific Tests =====================


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("vocab_size", [4096, 32000])
@pytest.mark.parametrize("k", [256, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_top_k_algorithms(algo, batch_size, vocab_size, k, dtype, set_topk_algo):
    """Test top_k with different algorithms (auto, multi_cta, filtered)."""
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    set_topk_algo(algo)

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=dtype)

    # flashinfer top_k with specified algorithm
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

    # Check accuracy
    accuracy = compute_topk_accuracy(indices.int(), ref_indices.int(), batch_size, k)
    min_accuracy = 0.98
    assert accuracy >= min_accuracy, (
        f"Algorithm {algo}: Accuracy {accuracy:.4f} < {min_accuracy}"
    )


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
@pytest.mark.parametrize("num_rows", [1, 8])
@pytest.mark.parametrize("max_len", [4096, 8192])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_page_table_transform_algorithms(
    algo, num_rows, max_len, k, dtype, set_topk_algo
):
    """Test page table transform with different algorithms."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    set_topk_algo(algo)

    torch.manual_seed(42)
    device = "cuda"

    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)
    src_page_table = torch.randint(
        0, 10000, (num_rows, max_len), device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    output = flashinfer.top_k_page_table_transform(scores, src_page_table, lengths, k)
    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)

    assert output.shape == (num_rows, k)
    assert output.dtype == torch.int32

    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, (
        f"Algorithm {algo}: Accuracy {accuracy:.4f} < {min_accuracy}"
    )


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
@pytest.mark.parametrize("num_rows", [1, 8])
@pytest.mark.parametrize("max_len", [4096, 8192])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ragged_transform_algorithms(algo, num_rows, max_len, k, dtype, set_topk_algo):
    """Test ragged transform with different algorithms."""
    if k > max_len:
        pytest.skip("k should be less than max_len")

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    set_topk_algo(algo)

    torch.manual_seed(42)
    device = "cuda"

    scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)
    offsets = torch.arange(
        0, num_rows * max_len, max_len, device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, k)
    ref_output = reference_ragged_transform(scores, offsets, lengths, k)

    assert output.shape == (num_rows, k)
    assert output.dtype == torch.int32

    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    min_accuracy = 0.95
    assert accuracy >= min_accuracy, (
        f"Algorithm {algo}: Accuracy {accuracy:.4f} < {min_accuracy}"
    )


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
def test_algorithms_produce_same_topk_set(algo, set_topk_algo):
    """Test that all algorithms produce valid top-k sets."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    torch.manual_seed(42)
    batch_size = 4
    vocab_size = 32000
    k = 256
    dtype = torch.float32
    device = "cuda"

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)

    set_topk_algo(algo)
    values, indices = flashinfer.top_k(logits, k)

    # Verify all returned values are truly top-k
    for i in range(batch_size):
        kth_largest = torch.kthvalue(-logits[i], k).values.item() * -1
        assert values[i].min().item() >= kth_largest - 1e-6, (
            f"Algorithm {algo}, row {i}: min value {values[i].min().item()} < "
            f"kth largest {kth_largest}"
        )


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
def test_algorithms_with_large_k(algo, set_topk_algo):
    """Test algorithms with large k values (stress test)."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    torch.manual_seed(42)
    batch_size = 2
    vocab_size = 4096
    k = 2048  # 50% of vocab_size
    dtype = torch.float32
    device = "cuda"

    set_topk_algo(algo)

    logits = torch.randn(batch_size, vocab_size, device=device, dtype=dtype)
    values, indices = flashinfer.top_k(logits, k)

    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    assert values.shape == (batch_size, k)
    accuracy = compute_topk_accuracy(indices.int(), ref_indices.int(), batch_size, k)
    assert accuracy >= 0.98, f"Algorithm {algo}: Accuracy {accuracy:.4f} < 0.98"


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
def test_bf16_long_seq_regression_across_algorithms(algo, set_topk_algo):
    """Regression for bf16 long-seq topk across algorithm overrides."""
    _require_sm80_for_bf16()
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    set_topk_algo(algo)

    logits, _, _, _, _, k, _ = _build_bf16_long_seq_bucket_inputs()

    values, indices = flashinfer.top_k(logits, k, sorted=True)
    ref_values, _ = torch.topk(logits, k, dim=-1, sorted=True)

    # Value set must match torch.topk; ties make index order non-unique.
    torch.testing.assert_close(values, ref_values)
    gathered_values = torch.gather(logits, dim=-1, index=indices)
    torch.testing.assert_close(values, gathered_values)


def _build_bf16_long_seq_bucket_inputs():
    """Construct a tie-heavy bf16 workload used by long-sequence regression tests."""
    batch_size = 4
    vocab_size = 65536
    k = 1024
    device = "cuda"

    logits = (
        ((torch.arange(vocab_size, device=device, dtype=torch.float32) % 64) / 64.0)
        .unsqueeze(0)
        .repeat(batch_size, 1)
        .to(torch.bfloat16)
    )
    lengths = torch.full((batch_size,), vocab_size, device=device, dtype=torch.int32)
    expected = (
        torch.arange(63, vocab_size, 64, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(batch_size, 1)
    )
    return logits, lengths, expected, batch_size, vocab_size, k, device


def _build_fp32_long_seq_overflow_inputs():
    """Construct a float32 case that overflows FilteredTopK refine candidate buffer.

    Values are crafted so that all elements share the same coarse/first-refine bucket,
    while the true top-k elements are concentrated in the tail. This triggers candidate
    truncation if overflow fallback is missing in multi-round refine.
    """
    batch_size = 1
    vocab_size = 65536
    k = 1024
    device = "cuda"

    idx = torch.arange(vocab_size, device=device, dtype=torch.int32)
    bits = torch.full((vocab_size,), 0x3F800000, device=device, dtype=torch.int32) + idx
    logits = bits.view(torch.float32).unsqueeze(0).contiguous()
    return logits, batch_size, vocab_size, k


def _build_fp32_long_seq_pivot_mismatch_inputs():
    """Construct a float32 case that exposes pivot reconstruction mismatch.

    This bit pattern keeps a dense coarse bucket while making the coarse threshold
    byte differ from the first FP32 ordered-byte threshold. If pivot reconstruction
    incorrectly mixes the coarse bin into high 8 bits, filtered top-k output drifts
    from torch.topk.
    """
    batch_size = 1
    vocab_size = 65536
    k = 1024
    device = "cuda"

    base = 0x3805ED27
    idx = torch.arange(vocab_size, device=device, dtype=torch.int64)
    bits = (
        (torch.tensor(base, device=device, dtype=torch.int64) + idx) & 0xFFFFFFFF
    ).to(torch.int32)
    logits = bits.view(torch.float32).unsqueeze(0).contiguous()
    return logits, batch_size, vocab_size, k


def _assert_unordered_indices_match(output, expected):
    """Compare index sets row-wise while ignoring order under ties."""
    output_sorted = torch.sort(output, dim=-1).values.to(torch.long)
    expected_sorted = torch.sort(expected, dim=-1).values.to(torch.long)
    assert torch.equal(
        output_sorted,
        expected_sorted,
    )


def _run_transform_with_identity_mapping(logits, k, transform_mode):
    """Run transform API with identity mapping so output equals selected indices."""
    batch_size, vocab_size = logits.shape
    device = logits.device
    lengths = torch.full((batch_size,), vocab_size, device=device, dtype=torch.int32)

    if transform_mode == "page_table":
        src_page_table = (
            torch.arange(vocab_size, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .contiguous()
        )
        return flashinfer.top_k_page_table_transform(logits, src_page_table, lengths, k)

    offsets = torch.zeros((batch_size,), device=device, dtype=torch.int32)
    return flashinfer.top_k_ragged_transform(logits, offsets, lengths, k)


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_bf16_long_seq_transform_regression_filtered(transform_mode, set_topk_algo):
    """Regression for bf16 long-seq transform APIs under filtered algorithm."""
    _require_sm80_for_bf16()
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")

    logits, lengths, expected, batch_size, vocab_size, k, device = (
        _build_bf16_long_seq_bucket_inputs()
    )

    if transform_mode == "page_table":
        # Identity page table: output should match selected local indices.
        src_page_table = (
            torch.arange(vocab_size, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .contiguous()
        )
        output = flashinfer.top_k_page_table_transform(
            logits, src_page_table, lengths, k
        )
    else:
        offsets = torch.zeros((batch_size,), device=device, dtype=torch.int32)
        output = flashinfer.top_k_ragged_transform(logits, offsets, lengths, k)

    _assert_unordered_indices_match(output, expected)


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
def test_fp32_long_seq_refine_overflow_regression_across_algorithms(
    algo, set_topk_algo
):
    """Regression for float32 long-seq refine overflow across algorithms."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    logits, batch_size, _, k = _build_fp32_long_seq_overflow_inputs()

    values, indices = flashinfer.top_k(logits, k, sorted=True)
    ref_values, ref_indices = torch.topk(logits, k, dim=-1, sorted=True)

    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)
    torch.testing.assert_close(values, ref_values)
    assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize("algo", ["auto", "multi_cta", "filtered"])
@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_fp32_long_seq_refine_overflow_transform_regression_across_algorithms(
    algo, transform_mode, set_topk_algo
):
    """Regression for fp32 long-seq overflow on transform APIs."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    logits, _, _, k = _build_fp32_long_seq_overflow_inputs()

    output = _run_transform_with_identity_mapping(logits, k, transform_mode)
    ref_indices = torch.topk(logits, k, dim=-1, sorted=True).indices.to(torch.int32)
    _assert_unordered_indices_match(output, ref_indices)


def test_fp32_long_seq_pivot_rebuild_regression_filtered(set_topk_algo):
    """Regression for pivot reconstruction in float32 overflow fallback."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    logits, batch_size, _, k = _build_fp32_long_seq_pivot_mismatch_inputs()

    values, indices = flashinfer.top_k(logits, k, sorted=True)
    ref_values, ref_indices = torch.topk(logits, k, dim=-1, sorted=True)

    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)
    torch.testing.assert_close(values, ref_values)
    assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_fp32_long_seq_pivot_rebuild_transform_regression_filtered(
    transform_mode, set_topk_algo
):
    """Regression for fp32 pivot reconstruction in filtered transform APIs."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    logits, _, _, k = _build_fp32_long_seq_pivot_mismatch_inputs()

    output = _run_transform_with_identity_mapping(logits, k, transform_mode)
    ref_indices = torch.topk(logits, k, dim=-1, sorted=True).indices.to(torch.int32)
    _assert_unordered_indices_match(output, ref_indices)


def test_top_k_deterministic_bool_repeatability_smoke():
    """deterministic=True should produce repeatable results."""
    batch_size = 2
    vocab_size = 8192
    k = 256
    device = "cuda"
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)

    values_a, indices_a = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    values_b, indices_b = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    assert torch.equal(values_a, values_b)
    assert torch.equal(indices_a, indices_b)


def test_top_k_deterministic_repeatability():
    """deterministic=True should be bitwise identical across repeated runs."""
    batch_size = 4
    vocab_size = 16384
    k = 256
    num_runs = 20
    device = "cuda"

    pattern = (torch.arange(vocab_size, device=device, dtype=torch.float32) % 32) / 32.0
    logits = pattern.unsqueeze(0).repeat(batch_size, 1).contiguous()

    ref_values, ref_indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)


def test_top_k_deterministic_repeatability_multi_cta(set_topk_algo):
    """deterministic=True should remain repeatable when forcing Radix multi-CTA."""
    set_topk_algo("multi_cta")

    batch_size = 1
    vocab_size = 131072
    k = 1024
    num_runs = 20
    device = "cuda"

    pattern = (torch.arange(vocab_size, device=device, dtype=torch.float32) % 64) / 64.0
    logits = pattern.unsqueeze(0).repeat(batch_size, 1).to(torch.bfloat16).contiguous()

    ref_values, ref_indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize("pattern", ["tie_heavy", "pivot_tie"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_top_k_deterministic_repeatability_filtered_tie_cases(
    pattern, dtype, set_topk_algo
):
    """Filtered deterministic top-k should be repeatable under tie pressure."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    batch_size = 4
    vocab_size = 16384
    k = 256
    num_runs = 20
    device = "cuda"

    if pattern == "tie_heavy":
        base = (torch.arange(vocab_size, device=device, dtype=torch.float32) % 32) / 32.0
        logits = base.unsqueeze(0).repeat(batch_size, 1).to(dtype).contiguous()
    else:  # pivot_tie
        logits = torch.ones(batch_size, vocab_size, device=device, dtype=dtype)
        gt_count = max(1, min(k // 4, vocab_size // 8))
        logits[:, vocab_size - gt_count :] = 2.0

    ref_values, ref_indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_deterministic_repeatability_filtered_tie_heavy(
    transform_mode, set_topk_algo
):
    """Filtered deterministic transform APIs should be repeatable under tie-heavy input."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    num_rows = 4
    max_len = 16384
    k = 256
    num_runs = 20
    device = "cuda"

    base = (torch.arange(max_len, device=device, dtype=torch.float32) % 32) / 32.0
    scores = base.unsqueeze(0).repeat(num_rows, 1).to(torch.bfloat16).contiguous()

    if transform_mode == "page_table":
        src_page_table = (
            torch.arange(max_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(num_rows, 1)
            .contiguous()
        )
        lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
        ref = flashinfer.top_k_page_table_transform(
            scores, src_page_table, lengths, k, deterministic=True
        )
        for _ in range(num_runs - 1):
            out = flashinfer.top_k_page_table_transform(
                scores, src_page_table, lengths, k, deterministic=True
            )
            assert torch.equal(out, ref)
    else:
        offsets = torch.arange(
            0, num_rows * max_len, max_len, device=device, dtype=torch.int32
        )
        lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
        ref = flashinfer.top_k_ragged_transform(
            scores, offsets, lengths, k, deterministic=True
        )
        for _ in range(num_runs - 1):
            out = flashinfer.top_k_ragged_transform(
                scores, offsets, lengths, k, deterministic=True
            )
            assert torch.equal(out, ref)


def test_top_k_invalid_deterministic_type():
    with pytest.raises(TypeError):
        flashinfer.top_k(
            torch.randn(1, 128, device="cuda", dtype=torch.float32),
            16,
            deterministic="invalid",
        )
    with pytest.raises(TypeError):
        flashinfer.top_k(
            torch.randn(1, 128, device="cuda", dtype=torch.float32),
            16,
            deterministic=1,
        )


def test_top_k_deterministic_bitwise_repeatability():
    """Deterministic top-k should be bitwise identical across repeated runs."""
    batch_size = 8
    vocab_size = 32768
    k = 512
    num_runs = 50
    device = "cuda"

    # Tie-heavy logits: repeated value buckets to stress tie handling.
    pattern = (torch.arange(vocab_size, device=device, dtype=torch.float32) % 64) / 64.0
    logits = pattern.unsqueeze(0).repeat(batch_size, 1).contiguous()

    ref_values, ref_indices = flashinfer.top_k(
        logits, k, deterministic=True, sorted=False
    )
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)


def test_top_k_page_table_transform_deterministic_repeatability():
    """Deterministic page-table transform should be bitwise identical across runs."""
    num_rows = 8
    max_len = 8192
    k = 512
    num_runs = 30
    device = "cuda"

    pattern = (torch.arange(max_len, device=device, dtype=torch.float32) % 32) / 32.0
    scores = pattern.unsqueeze(0).repeat(num_rows, 1).contiguous()
    src_page_table = (
        torch.arange(max_len, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(num_rows, 1)
        .contiguous()
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    ref = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k, deterministic=True
    )
    for _ in range(num_runs - 1):
        out = flashinfer.top_k_page_table_transform(
            scores, src_page_table, lengths, k, deterministic=True
        )
        assert torch.equal(out, ref)


def test_top_k_ragged_transform_deterministic_repeatability():
    """Deterministic ragged transform should be bitwise identical across runs."""
    num_rows = 8
    max_len = 8192
    k = 512
    num_runs = 30
    device = "cuda"

    pattern = (torch.arange(max_len, device=device, dtype=torch.float32) % 32) / 32.0
    scores = pattern.unsqueeze(0).repeat(num_rows, 1).contiguous()
    offsets = torch.arange(
        0, num_rows * max_len, max_len, device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)

    ref = flashinfer.top_k_ragged_transform(
        scores, offsets, lengths, k, deterministic=True
    )
    for _ in range(num_runs - 1):
        out = flashinfer.top_k_ragged_transform(
            scores, offsets, lengths, k, deterministic=True
        )
        assert torch.equal(out, ref)


if __name__ == "__main__":
    # Basic tests
    test_top_k(4, 32000, 256, torch.float32)
    test_top_k_sorted(4, 32000, 256, torch.float32)
    test_top_k_large_batch(64, 128512, 256)

    # Fused transform tests
    print("Testing page table transform...")
    test_top_k_page_table_transform(8, 4096, 256, torch.float32)
    test_top_k_page_table_transform(8, 4096, 256, torch.float16)
    print("Testing ragged transform...")
    test_top_k_ragged_transform(8, 4096, 256, torch.float32)
    test_top_k_ragged_transform(8, 4096, 256, torch.float16)
    print("Testing trivial cases...")
    test_page_table_transform_trivial_case(8, 2048, 256)
    test_ragged_transform_trivial_case(8, 2048, 256)
    print("Testing variable lengths...")
    test_page_table_transform_variable_lengths(8, 4096, 256)
    test_ragged_transform_variable_lengths(8, 4096, 256)
    print("Testing large scale...")
    test_page_table_transform_large_scale(64, 8192, 256)
    test_ragged_transform_large_scale(64, 8192, 256)

    # SGLang-style comparison tests
    print("\nTesting SGLang-style comparisons...")
    test_compare_with_sglang_style_page_table(8, 4096, 256, torch.float32)
    test_compare_with_sglang_style_ragged(8, 4096, 256, torch.float32)
    test_compare_with_sglang_style_prefill_mode(8, 4096, 256)

    print("\nAll tests passed!")
