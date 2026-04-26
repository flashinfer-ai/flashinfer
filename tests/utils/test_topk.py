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
import flashinfer.utils as flashinfer_utils
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
        assert len(ref_set) == len(test_set)
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


def _get_cached_topk_row_states_buffer(device: torch.device):
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (f"radix_topk_row_states_{device}", device)
    return flashinfer_utils._cache_buf.get(key)


def _clear_cached_topk_row_states_buffer(device: torch.device):
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    key = (f"radix_topk_row_states_{device}", device)
    flashinfer_utils._cache_buf.pop(key, None)


def _build_strictly_descending_logits(
    num_rows: int, vocab_size: int, device: torch.device
) -> torch.Tensor:
    base = torch.arange(vocab_size, 0, -1, device=device, dtype=torch.float32)
    return base.unsqueeze(0).repeat(num_rows, 1).contiguous()


@pytest.mark.parametrize("batch_size", [1, 16, 64])
@pytest.mark.parametrize("vocab_size", [32000, 65536, 128512])
@pytest.mark.parametrize("k", [256, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "tie_break",
    [
        flashinfer.TopKTieBreak.NONE,
        flashinfer.TopKTieBreak.SMALL,
        flashinfer.TopKTieBreak.LARGE,
    ],
)
def test_top_k(batch_size, vocab_size, k, dtype, tie_break):
    """Test top_k returns correct values and indices."""
    if tie_break != flashinfer.TopKTieBreak.NONE and not can_implement_filtered_topk():
        pytest.skip("Tie-break modes require filtered top-k support on this device")
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=dtype)

    # flashinfer top_k
    values, indices = flashinfer.top_k(logits, k, tie_break=tie_break)

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
    min_accuracy = 0.97
    assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("vocab_size", [32000, 65536])
@pytest.mark.parametrize("k", [256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    "tie_break",
    [
        flashinfer.TopKTieBreak.NONE,
        flashinfer.TopKTieBreak.SMALL,
        flashinfer.TopKTieBreak.LARGE,
    ],
)
def test_top_k_sorted(batch_size, vocab_size, k, dtype, tie_break):
    """Test top_k with sorted=True returns sorted values."""
    if tie_break != flashinfer.TopKTieBreak.NONE and not can_implement_filtered_topk():
        pytest.skip("Tie-break modes require filtered top-k support on this device")
    if k > vocab_size:
        pytest.skip("k should be less than vocab_size")

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=dtype)

    # flashinfer top_k with sorted=True
    values, indices = flashinfer.top_k(logits, k, sorted=True, tie_break=tie_break)

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
@pytest.mark.parametrize(
    "tie_break",
    [
        flashinfer.TopKTieBreak.NONE,
        flashinfer.TopKTieBreak.SMALL,
        flashinfer.TopKTieBreak.LARGE,
    ],
)
def test_top_k_single_batch(vocab_size, k, tie_break):
    """Test top_k with batch_size=1 (common inference case)."""
    if tie_break != flashinfer.TopKTieBreak.NONE and not can_implement_filtered_topk():
        pytest.skip("Tie-break modes require filtered top-k support on this device")
    torch.manual_seed(42)
    logits = torch.randn(1, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k
    values, indices = flashinfer.top_k(logits, k, tie_break=tie_break)

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
@pytest.mark.parametrize("det", [True, False])
@pytest.mark.parametrize(
    "tie_break",
    [
        flashinfer.TopKTieBreak.NONE,
        flashinfer.TopKTieBreak.SMALL,
        flashinfer.TopKTieBreak.LARGE,
    ],
)
def test_top_k_large_batch(batch_size, vocab_size, k, det, tie_break):
    """Test top_k with large batch sizes (multi-CTA path)."""
    if tie_break != flashinfer.TopKTieBreak.NONE and not can_implement_filtered_topk():
        pytest.skip("Tie-break modes require filtered top-k support on this device")
    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float32)

    # flashinfer top_k (should use multi-CTA path for large vocab)
    values, indices = flashinfer.top_k(
        logits, k, deterministic=det, tie_break=tie_break
    )

    # Reference: torch.topk
    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    # Check output shape
    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Check accuracy
    accuracy = compute_topk_accuracy(indices, ref_indices, batch_size, k)
    assert accuracy >= 0.98, f"Accuracy {accuracy:.4f} < 0.98"


@pytest.mark.parametrize("api_kind", ["top_k", "page_table", "ragged"])
@pytest.mark.parametrize(
    ("first_deterministic", "second_deterministic"),
    [(False, True), (True, False)],
)
def test_multi_cta_reuses_dirty_cached_row_states_buffer_across_mode_transitions(
    api_kind, set_topk_algo, first_deterministic, second_deterministic
):
    set_topk_algo("multi_cta")
    device = torch.device("cuda")
    _clear_cached_topk_row_states_buffer(device)

    batch_size = 4
    vocab_size = 131072
    k = 512
    logits = _build_strictly_descending_logits(batch_size, vocab_size, device)

    if api_kind == "top_k":
        expected_values = logits[:, :k]
        expected_indices = (
            torch.arange(k, device=device, dtype=torch.int64)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        values_a, indices_a = flashinfer.top_k(
            logits, k, sorted=True, deterministic=first_deterministic
        )
        torch.testing.assert_close(values_a, expected_values)
        assert torch.equal(indices_a, expected_indices)
        buf_a = _get_cached_topk_row_states_buffer(device)
        assert buf_a is not None

        values_b, indices_b = flashinfer.top_k(
            logits, k, sorted=True, deterministic=second_deterministic
        )
        torch.testing.assert_close(values_b, expected_values)
        assert torch.equal(indices_b, expected_indices)
    else:
        lengths = torch.full(
            (batch_size,), vocab_size, device=device, dtype=torch.int32
        )
        expected = torch.arange(k, device=device, dtype=torch.int32).unsqueeze(0)
        expected = expected.expand(batch_size, -1)
        src_page_table = None
        offsets = None

        if api_kind == "ragged":
            offsets = torch.arange(
                0, batch_size * vocab_size, vocab_size, device=device, dtype=torch.int32
            )
            expected = offsets.unsqueeze(1) + expected

        output_a = _run_transform(
            logits,
            k,
            api_kind,
            lengths=lengths,
            deterministic=first_deterministic,
            src_page_table=src_page_table,
            offsets=offsets,
        )
        _assert_unordered_indices_match(output_a, expected)
        buf_a = _get_cached_topk_row_states_buffer(device)
        assert buf_a is not None

        output_b = _run_transform(
            logits,
            k,
            api_kind,
            lengths=lengths,
            deterministic=second_deterministic,
            src_page_table=src_page_table,
            offsets=offsets,
        )
        _assert_unordered_indices_match(output_b, expected)

    buf_b = _get_cached_topk_row_states_buffer(device)
    assert buf_b is buf_a


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
    row_starts: torch.Tensor = None,
) -> torch.Tensor:
    """Reference implementation for page table transform using torch.topk."""
    num_rows = scores.size(0)
    scores.size(1)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    for i in range(num_rows):
        length = lengths[i].item()
        row_start = row_starts[i].item() if row_starts is not None else 0
        batch_idx = row_to_batch[i].item() if row_to_batch is not None else i

        if length <= k:
            # Trivial case: just copy first `length` entries
            output[i, :length] = src_page_table[
                batch_idx, row_start : row_start + length
            ]
        else:
            # Get top-k indices
            row_scores = scores[i, row_start : row_start + length]
            _, topk_indices = torch.topk(row_scores.float(), k)
            # Gather from page table
            output[i] = src_page_table[batch_idx, row_start + topk_indices.long()]

    return output


def reference_ragged_transform(
    scores: torch.Tensor,
    offsets: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    row_starts: torch.Tensor = None,
) -> torch.Tensor:
    """Reference implementation for ragged transform using torch.topk."""
    num_rows = scores.size(0)
    device = scores.device

    output = torch.full((num_rows, k), -1, dtype=torch.int32, device=device)

    for i in range(num_rows):
        length = lengths[i].item()
        row_start = row_starts[i].item() if row_starts is not None else 0
        offset = offsets[i].item()

        if length <= k:
            # Trivial case: indices are [offset, offset+1, ..., offset+length-1]
            output[i, :length] = torch.arange(
                offset, offset + length, dtype=torch.int32, device=device
            )
        else:
            # Get top-k indices
            row_scores = scores[i, row_start : row_start + length]
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


@pytest.mark.parametrize("algo", ["multi_cta", "filtered"])
@pytest.mark.parametrize("dsa_graph_safe", [False, True])
@pytest.mark.parametrize(
    "num_rows,max_len,k",
    [
        (2, 128 * 1024, 2048),
        (1, 256 * 1024, 1024),
        (74, 16 * 1024, 512),
    ],
)
def test_top_k_transform_with_row_starts(
    algo, dsa_graph_safe, num_rows, max_len, k, set_topk_algo
):
    """Transform APIs should honor row_starts windowing with local-index semantics."""
    if (algo == "filtered" or dsa_graph_safe) and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    device = "cuda"

    base = -torch.arange(max_len, device=device, dtype=torch.float32)
    scores = base.unsqueeze(0).repeat(num_rows, 1).contiguous()

    max_start = max_len - (k + 1)
    start_stride = max(1, max_start // max(1, num_rows - 1))
    row_starts = (
        torch.arange(num_rows, device=device, dtype=torch.int32) * start_stride
    ).clamp(max=max_start)
    max_windows = max_len - row_starts
    lengths = torch.minimum(
        max_windows,
        k + 1 + (torch.arange(num_rows, device=device, dtype=torch.int32) % 4),
    )
    offsets = torch.arange(num_rows, device=device, dtype=torch.int32) * 100
    row_to_batch = torch.arange(num_rows - 1, -1, -1, device=device, dtype=torch.int32)

    src_page_table = (
        torch.arange(max_len, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .repeat(num_rows, 1)
    )
    src_page_table = (
        src_page_table
        + 1000 * torch.arange(num_rows, device=device, dtype=torch.int32).unsqueeze(1)
    ).contiguous()

    output_page = flashinfer.top_k_page_table_transform(
        scores,
        src_page_table,
        lengths,
        k,
        row_to_batch=row_to_batch,
        row_starts=row_starts,
        deterministic=True,
        dsa_graph_safe=dsa_graph_safe,
    )
    output_ragged = flashinfer.top_k_ragged_transform(
        scores,
        offsets,
        lengths,
        k,
        row_starts=row_starts,
        deterministic=True,
        dsa_graph_safe=dsa_graph_safe,
    )
    ref_page = reference_page_table_transform(
        scores,
        src_page_table,
        lengths,
        k,
        row_to_batch=row_to_batch,
        row_starts=row_starts,
    )

    ref_ragged = reference_ragged_transform(
        scores, offsets, lengths, k, row_starts=row_starts
    )
    output_page_sorted, _ = torch.sort(output_page, dim=-1)
    ref_page_sorted, _ = torch.sort(ref_page, dim=-1)
    assert torch.equal(output_page_sorted, ref_page_sorted)

    output_ragged_sorted, _ = torch.sort(output_ragged, dim=-1)
    ref_ragged_sorted, _ = torch.sort(ref_ragged, dim=-1)
    assert torch.equal(output_ragged_sorted, ref_ragged_sorted)


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


@pytest.mark.parametrize("num_rows", [4, 8])
@pytest.mark.parametrize("top_k", [256, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_ragged_transform_multi_cta_short_rows(num_rows, top_k, dtype):
    """Regression test for uint32 underflow in multi-CTA chunk_size calculation."""
    torch.manual_seed(42)
    device = "cuda"

    max_len = 131072

    # Force multi_cta path so the test exercises the vulnerable code path
    # regardless of the heuristic.
    old_algo = os.environ.get("FLASHINFER_TOPK_ALGO", None)
    os.environ["FLASHINFER_TOPK_ALGO"] = "multi_cta"

    try:
        scores = torch.randn(num_rows, max_len, device=device, dtype=dtype)
        offsets = torch.zeros(num_rows, device=device, dtype=torch.int32)

        # Mix short and long rows. Short rows (4K-8K) are well below chunk_size
        # on any GPU, so CTAs beyond the first will have chunk_start > length.
        lengths_list = []
        for i in range(num_rows):
            if i % 2 == 0:
                lengths_list.append(max_len)
            else:
                lengths_list.append(torch.randint(4000, 8000, (1,)).item())
        lengths = torch.tensor(lengths_list, device=device, dtype=torch.int32)

        output = flashinfer.top_k_ragged_transform(scores, offsets, lengths, top_k)
        ref_output = reference_ragged_transform(scores, offsets, lengths, top_k)

        assert output.shape == (num_rows, top_k)
        assert output.dtype == torch.int32

        accuracy = compute_transform_accuracy(output, ref_output, num_rows, top_k)
        min_accuracy = 0.90
        assert accuracy >= min_accuracy, f"Accuracy {accuracy:.4f} < {min_accuracy}"

        # Verify indices stay within [offset, offset + length) for each row
        for i in range(num_rows):
            length = lengths[i].item()
            row_out = output[i]
            valid = row_out[row_out >= 0]
            assert torch.all(valid < length), (
                f"Row {i}: index out of bounds (max={valid.max().item()}, length={length})"
            )
    finally:
        if old_algo is None:
            os.environ.pop("FLASHINFER_TOPK_ALGO", None)
        else:
            os.environ["FLASHINFER_TOPK_ALGO"] = old_algo


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


def _assert_top_k_matches_torch(
    logits: torch.Tensor, k: int, *, deterministic: bool = False, sorted: bool = True
):
    """Assert FlashInfer top_k matches torch.topk for exact-order cases."""
    values, indices = flashinfer.top_k(
        logits, k, deterministic=deterministic, sorted=sorted
    )
    ref_values, ref_indices = torch.topk(logits, k, dim=-1, sorted=sorted)

    assert values.shape == ref_values.shape
    assert indices.shape == ref_indices.shape
    torch.testing.assert_close(values, ref_values)
    assert torch.equal(indices, ref_indices)


def _run_transform(
    logits,
    k,
    transform_mode,
    *,
    lengths: torch.Tensor | None = None,
    deterministic: bool = False,
    src_page_table: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
):
    """Run a transform API with either explicit or default identity metadata."""
    batch_size, vocab_size = logits.shape
    device = logits.device
    if lengths is None:
        lengths = torch.full(
            (batch_size,), vocab_size, device=device, dtype=torch.int32
        )

    if transform_mode == "page_table":
        if src_page_table is None:
            src_page_table = (
                torch.arange(vocab_size, device=device, dtype=torch.int32)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .contiguous()
            )
        return flashinfer.top_k_page_table_transform(
            logits, src_page_table, lengths, k, deterministic=deterministic
        )

    if offsets is None:
        offsets = torch.zeros((batch_size,), device=device, dtype=torch.int32)
    return flashinfer.top_k_ragged_transform(
        logits, offsets, lengths, k, deterministic=deterministic
    )


def _run_transform_with_identity_mapping(
    logits, k, transform_mode, deterministic: bool = False
):
    """Run transform API with identity mapping so output equals selected indices."""
    return _run_transform(logits, k, transform_mode, deterministic=deterministic)


def _assert_transform_identity_matches_torch(
    logits, k, transform_mode, deterministic: bool = False
):
    """Assert transform output matches torch.topk indices under identity mapping."""
    output = _run_transform_with_identity_mapping(
        logits, k, transform_mode, deterministic=deterministic
    )
    ref_indices = torch.topk(logits, k, dim=-1, sorted=True).indices.to(torch.int32)
    _assert_unordered_indices_match(output, ref_indices)


def _assert_repeatable_transform_output(
    logits,
    k,
    transform_mode,
    *,
    num_runs: int,
    deterministic: bool = True,
    lengths: torch.Tensor | None = None,
    src_page_table: torch.Tensor | None = None,
    offsets: torch.Tensor | None = None,
):
    """Assert a transform API produces bitwise-identical output across repeated runs."""
    ref = _run_transform(
        logits,
        k,
        transform_mode,
        lengths=lengths,
        deterministic=deterministic,
        src_page_table=src_page_table,
        offsets=offsets,
    )
    for _ in range(num_runs - 1):
        out = _run_transform(
            logits,
            k,
            transform_mode,
            lengths=lengths,
            deterministic=deterministic,
            src_page_table=src_page_table,
            offsets=offsets,
        )
        assert torch.equal(out, ref)
    return ref


def _assert_repeatable_valid_identity_transform_selection(
    output_a: torch.Tensor,
    output_b: torch.Tensor,
    vocab_size: int,
    k: int,
    gt_count: int = 0,
):
    """Assert deterministic transform outputs are repeatable and form a valid top-k set."""
    assert torch.equal(output_a, output_b)
    output = output_a[0]
    assert output.numel() == k
    assert torch.unique(output).numel() == k
    assert torch.all((output >= 0) & (output < vocab_size))

    if gt_count > 0:
        gt_indices = torch.arange(
            vocab_size - gt_count,
            vocab_size,
            device=output.device,
            dtype=torch.int32,
        )
        gt_mask = torch.isin(output, gt_indices)
        assert gt_mask.sum().item() == gt_count
        assert torch.all(torch.isin(gt_indices, output))
        tie_selected = output[~gt_mask]
        assert tie_selected.numel() == k - gt_count
        assert torch.all(tie_selected < vocab_size - gt_count)


def _assert_repeatable_valid_topk_selection(
    logits: torch.Tensor,
    values_a: torch.Tensor,
    indices_a: torch.Tensor,
    values_b: torch.Tensor,
    indices_b: torch.Tensor,
    k: int,
    gt_count: int = 0,
):
    """Assert deterministic top-k outputs are repeatable and form a valid selected set."""
    assert torch.equal(values_a, values_b)
    assert torch.equal(indices_a, indices_b)

    gathered_values = torch.gather(logits, 1, indices_a)
    torch.testing.assert_close(values_a, gathered_values)

    vocab_size = logits.size(1)
    for output in indices_a:
        assert output.numel() == k
        assert torch.unique(output).numel() == k
        assert torch.all((output >= 0) & (output < vocab_size))

        if gt_count > 0:
            gt_indices = torch.arange(
                vocab_size - gt_count,
                vocab_size,
                device=output.device,
                dtype=output.dtype,
            )
            gt_mask = torch.isin(output, gt_indices)
            assert gt_mask.sum().item() == gt_count
            assert torch.all(torch.isin(gt_indices, output))
            tie_selected = output[~gt_mask]
            assert tie_selected.numel() == k - gt_count
            assert torch.all(tie_selected < vocab_size - gt_count)


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


@pytest.mark.parametrize(
    ("builder", "algo"),
    [
        (_build_fp32_long_seq_overflow_inputs, "auto"),
        (_build_fp32_long_seq_overflow_inputs, "multi_cta"),
        (_build_fp32_long_seq_overflow_inputs, "filtered"),
        (_build_fp32_long_seq_pivot_mismatch_inputs, "filtered"),
    ],
    ids=[
        "refine_overflow-auto",
        "refine_overflow-multi_cta",
        "refine_overflow-filtered",
        "pivot_rebuild-filtered",
    ],
)
@pytest.mark.parametrize("api_kind", ["top_k", "page_table", "ragged"])
def test_fp32_long_seq_regression_matrix(builder, algo, api_kind, set_topk_algo):
    """Long-sequence fp32 regressions should remain exact across supported APIs."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    logits, _, _, k = builder()
    if api_kind == "top_k":
        _assert_top_k_matches_torch(logits, k, sorted=True)
    else:
        _assert_transform_identity_matches_torch(logits, k, api_kind)


@pytest.mark.parametrize(
    ("builder", "case_name"),
    [
        (_build_fp32_long_seq_overflow_inputs, "refine_overflow"),
        (_build_fp32_long_seq_pivot_mismatch_inputs, "pivot_rebuild"),
    ],
)
@pytest.mark.parametrize("api_kind", ["top_k", "page_table", "ragged"])
def test_fp32_long_seq_filtered_deterministic_regression_matrix(
    builder, case_name, api_kind, set_topk_algo
):
    """Filtered deterministic long-sequence fallback paths should remain exact."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    logits, _, _, k = builder()
    if api_kind == "top_k":
        _assert_top_k_matches_torch(logits, k, deterministic=True, sorted=True)
    else:
        _assert_transform_identity_matches_torch(
            logits, k, api_kind, deterministic=True
        )


def test_top_k_deterministic_across_streams():
    """deterministic=True should be repeatable across CUDA streams.

    This runs the same deterministic top-k on two non-default streams (sequentially)
    and checks for bitwise-identical results.
    """
    batch_size = 4
    vocab_size = 16384
    k = 256
    device = "cuda"

    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)

    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()

    with torch.cuda.stream(s1):
        values_a, indices_a = flashinfer.top_k(
            logits, k, deterministic=True, sorted=False
        )
    s1.synchronize()

    with torch.cuda.stream(s2):
        values_b, indices_b = flashinfer.top_k(
            logits, k, deterministic=True, sorted=False
        )
    s2.synchronize()

    assert torch.equal(values_a, values_b)
    assert torch.equal(indices_a, indices_b)


@pytest.mark.parametrize(
    ("algo", "batch_size", "vocab_size", "k", "dtype", "pattern_mod"),
    [
        ("auto", 4, 16384, 256, torch.float32, 32),
        # A 4096-wide fp32 row keeps ctas_per_group == 1 even under the multi_cta
        # override, so this still exercises the radix single-CTA branch.
        ("multi_cta", 4, 4096, 256, torch.float32, 32),
        ("multi_cta", 1, 131072, 1024, torch.bfloat16, 64),
        ("filtered", 4, 16384, 256, torch.float32, 32),
    ],
)
def test_top_k_deterministic_repeatability_matrix(
    algo, batch_size, vocab_size, k, dtype, pattern_mod, set_topk_algo
):
    """deterministic=True should be bitwise identical across routing modes."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")
    if dtype == torch.bfloat16:
        _require_sm80_for_bf16()

    set_topk_algo(algo)

    num_runs = 20
    device = "cuda"
    pattern = (
        torch.arange(vocab_size, device=device, dtype=torch.float32) % pattern_mod
    ) / float(pattern_mod)
    logits = pattern.unsqueeze(0).repeat(batch_size, 1).to(dtype).contiguous()

    ref_values, ref_indices = flashinfer.top_k(
        logits, k, deterministic=True, sorted=False
    )
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        assert torch.equal(values, ref_values)
        assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize(
    ("algo", "batch_size", "vocab_size", "k"),
    [
        ("auto", 4, 16384, 256),
        # A 4096-wide fp32 row keeps ctas_per_group == 1 even under the multi_cta
        # override, so this still exercises the radix single-CTA branch.
        ("multi_cta", 4, 4096, 256),
        ("multi_cta", 1, 131072, 1024),
        ("filtered", 4, 16384, 256),
    ],
)
def test_top_k_deterministic_sorted_matches_stable_sort(
    algo, batch_size, vocab_size, k, set_topk_algo
):
    """sorted=True should be repeatable, valid, and descending."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    device = "cuda"
    pattern = (torch.arange(vocab_size, device=device, dtype=torch.float32) % 32) / 32.0
    logits = pattern.unsqueeze(0).repeat(batch_size, 1).contiguous()

    sorted_values_a, sorted_indices_a = flashinfer.top_k(
        logits, k, deterministic=True, sorted=True
    )
    sorted_values_b, sorted_indices_b = flashinfer.top_k(
        logits, k, deterministic=True, sorted=True
    )

    _assert_repeatable_valid_topk_selection(
        logits, sorted_values_a, sorted_indices_a, sorted_values_b, sorted_indices_b, k
    )
    assert torch.all(sorted_values_a[:, :-1] >= sorted_values_a[:, 1:])


@pytest.mark.parametrize(
    ("algo", "vocab_size"),
    [
        ("auto", 16384),
        # A 4096-wide fp32 row keeps ctas_per_group == 1 even under the multi_cta
        # override, so this still exercises the radix single-CTA branch.
        ("multi_cta", 4096),
        ("multi_cta", 131072),
        ("filtered", 16384),
    ],
)
@pytest.mark.parametrize(("pattern", "k"), [("all_equal", 8), ("pivot_tie", 6)])
def test_top_k_deterministic_sorted_repeatable_valid_selection_under_ties(
    algo, vocab_size, pattern, k, set_topk_algo
):
    """Deterministic sorted top-k should remain repeatable under tie pressure."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    device = "cuda"
    logits = torch.ones((1, vocab_size), device=device, dtype=torch.float16)
    gt_count = 0

    if pattern == "all_equal":
        expected_values = torch.ones((1, k), device=device, dtype=torch.float16)
    else:
        gt_count = 2
        logits[:, vocab_size - gt_count :] = 2.0
        expected_values = torch.cat(
            [
                torch.full((1, gt_count), 2.0, device=device, dtype=torch.float16),
                torch.ones((1, k - gt_count), device=device, dtype=torch.float16),
            ],
            dim=-1,
        )

    values_a, indices_a = flashinfer.top_k(logits, k, deterministic=True, sorted=True)
    values_b, indices_b = flashinfer.top_k(logits, k, deterministic=True, sorted=True)

    torch.testing.assert_close(values_a, expected_values)
    _assert_repeatable_valid_topk_selection(
        logits, values_a, indices_a, values_b, indices_b, k, gt_count=gt_count
    )


@pytest.mark.parametrize(
    ("algo", "batch_size", "vocab_size", "k"),
    [
        ("filtered", 2, 128 * 1024, 2048),
        ("filtered", 1, 1024 * 1024, 1024),
        ("filtered", 74, 16 * 1024, 512),
    ],
    ids=[
        "filtered_b2_l128k_k2048",
        "filtered_b1_l1m_k1024",
        "filtered_b74_l16k_k512",
    ],
)
def test_top_k_tie_break_modes(algo, batch_size, vocab_size, k, set_topk_algo):
    """tie_break=1|2 should select row-global smallest/largest pivot indices."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    device = "cuda"
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    logits = (
        torch.randn(
            (batch_size, 1), device=device, dtype=torch.float32, generator=generator
        )
        .expand(batch_size, vocab_size)
        .contiguous()
    )

    values_small, indices_small = flashinfer.top_k(logits, k, tie_break=1)
    values_large, indices_large = flashinfer.top_k(logits, k, tie_break=2)

    expected_small = (
        torch.arange(k, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    expected_large = (
        torch.arange(vocab_size - k, vocab_size, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )
    expected_values = logits[:, :1].expand(batch_size, k).contiguous()

    torch.testing.assert_close(values_small, expected_values)
    torch.testing.assert_close(values_large, expected_values)
    _assert_unordered_indices_match(indices_small, expected_small)
    _assert_unordered_indices_match(indices_large, expected_large)


@pytest.mark.parametrize(
    ("algo", "num_rows", "max_len", "k"),
    [
        ("filtered", 2, 128 * 1024, 2048),
        ("filtered", 1, 1024 * 1024, 1024),
        ("filtered", 74, 16 * 1024, 512),
    ],
    ids=[
        "filtered_rows2_l128k_k2048",
        "filtered_rows1_l1m_k1024",
        "filtered_rows74_l16k_k512",
    ],
)
def test_top_k_tie_break_modes_transform_apis(
    algo, num_rows, max_len, k, set_topk_algo
):
    """Transform APIs should honor tie_break selection before remapping outputs."""
    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    device = "cuda"

    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    scores = (
        torch.randn(
            (num_rows, 1), device=device, dtype=torch.float32, generator=generator
        )
        .expand(num_rows, max_len)
        .contiguous()
    )
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
    src_page_table = (
        torch.arange(max_len, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_rows, -1)
        .contiguous()
    )
    offsets = torch.zeros((num_rows,), device=device, dtype=torch.int32)

    page_small = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k, tie_break=1
    )
    page_large = flashinfer.top_k_page_table_transform(
        scores, src_page_table, lengths, k, tie_break=2
    )
    ragged_small = flashinfer.top_k_ragged_transform(
        scores, offsets, lengths, k, tie_break=1
    )
    ragged_large = flashinfer.top_k_ragged_transform(
        scores, offsets, lengths, k, tie_break=2
    )

    expected_small = (
        torch.arange(k, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_rows, -1)
    )
    expected_large = (
        torch.arange(max_len - k, max_len, device=device, dtype=torch.int32)
        .unsqueeze(0)
        .expand(num_rows, -1)
    )

    assert torch.equal(page_small, expected_small)
    assert torch.equal(page_large, expected_large)
    assert torch.equal(ragged_small, expected_small)
    assert torch.equal(ragged_large, expected_large)


@pytest.mark.parametrize(
    ("algo", "vocab_size", "k"),
    [
        ("auto", 131072, 4096),
        ("multi_cta", 131072, 4096),
        # Keep one filtered-specific large-k coverage row that still satisfies
        # FILTERED_TOPK_MAX_K and therefore actually routes to FilteredTopK.
        ("filtered", 131072, 2048),
    ],
    ids=[
        "auto_k4096",
        "multi_cta_k4096",
        "filtered_k2048",
    ],
)
def test_top_k_deterministic_sorted_large_k_matches_torch_by_algo(
    algo, vocab_size, k, set_topk_algo
):
    """Deterministic sorted output should match torch.topk across routed large-k cases."""
    set_topk_algo(algo)

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("GPU does not support filtered topk (requires 128KB shared memory)")

    batch_size = 1
    device = "cuda"

    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float32)

    values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=True)
    ref_values, ref_indices = torch.topk(logits, k, dim=-1, sorted=True)

    torch.testing.assert_close(values, ref_values)
    assert torch.equal(indices, ref_indices)


@pytest.mark.parametrize("algo", ["auto", "multi_cta"])
def test_top_k_deterministic_trivial_k_equals_length_by_algo(algo, set_topk_algo):
    """Deterministic k==length fast paths should remain exact across auto/radix routing."""
    set_topk_algo(algo)

    batch_size = 2
    vocab_size = 131072
    k = vocab_size
    device = "cuda"

    torch.manual_seed(0)
    logits = torch.randn(batch_size, vocab_size, device=device, dtype=torch.float16)

    values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
    expected_indices = (
        torch.arange(vocab_size, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch_size, -1)
    )

    assert torch.equal(indices, expected_indices)
    torch.testing.assert_close(values, logits)


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_multi_cta_deterministic_trivial_lengths(
    transform_mode, set_topk_algo
):
    """Deterministic radix transform should handle length == k and length < k fast paths."""
    set_topk_algo("multi_cta")

    num_rows = 2
    max_len = 131072
    k = 256
    device = "cuda"

    torch.manual_seed(0)
    scores = torch.randn(num_rows, max_len, device=device, dtype=torch.float16)
    lengths = torch.tensor([k, k // 2], device=device, dtype=torch.int32)

    if transform_mode == "page_table":
        src_page_table = (
            torch.arange(max_len, device=device, dtype=torch.int32)
            .mul(3)
            .add(7)
            .unsqueeze(0)
            .repeat(num_rows, 1)
            .contiguous()
        )
        output = flashinfer.top_k_page_table_transform(
            scores, src_page_table, lengths, k, deterministic=True
        )
        expected = torch.full((num_rows, k), -1, device=device, dtype=torch.int32)
        expected[0] = src_page_table[0, :k]
        expected[1, : k // 2] = src_page_table[1, : k // 2]
    else:
        offsets = torch.tensor([0, 1000], device=device, dtype=torch.int32)
        output = flashinfer.top_k_ragged_transform(
            scores, offsets, lengths, k, deterministic=True
        )
        expected = torch.full((num_rows, k), -1, device=device, dtype=torch.int32)
        expected[0] = torch.arange(k, device=device, dtype=torch.int32)
        expected[1, : k // 2] = offsets[1] + torch.arange(
            k // 2, device=device, dtype=torch.int32
        )

    assert torch.equal(output, expected)


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
@pytest.mark.parametrize(("pattern", "k"), [("all_equal", 8), ("pivot_tie", 6)])
def test_top_k_transform_filtered_deterministic_valid_selection_under_ties(
    transform_mode, pattern, k, set_topk_algo
):
    """Filtered deterministic transform APIs should be repeatable and select a valid top-k set."""
    if not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo("filtered")
    device = "cuda"
    vocab_size = 16384
    logits = torch.ones((1, vocab_size), device=device, dtype=torch.float16)
    gt_count = 0

    if pattern == "all_equal":
        pass
    else:
        gt_count = 2
        logits[:, vocab_size - gt_count :] = 2.0

    output_a = _run_transform_with_identity_mapping(
        logits, k, transform_mode, deterministic=True
    )
    output_b = _run_transform_with_identity_mapping(
        logits, k, transform_mode, deterministic=True
    )
    _assert_repeatable_valid_identity_transform_selection(
        output_a, output_b, vocab_size, k, gt_count=gt_count
    )


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_deterministic_repeatability_multi_cta_all_equal(
    transform_mode, set_topk_algo
):
    """Force radix multi-CTA all-equal transform path where later CTAs take zero eq quota."""
    set_topk_algo("multi_cta")

    device = "cuda"
    vocab_size = 131072
    k = 256
    logits = torch.ones((1, vocab_size), device=device, dtype=torch.float16)

    output_a = _run_transform_with_identity_mapping(
        logits, k, transform_mode, deterministic=True
    )
    output_b = _run_transform_with_identity_mapping(
        logits, k, transform_mode, deterministic=True
    )
    _assert_repeatable_valid_identity_transform_selection(
        output_a, output_b, vocab_size, k
    )


@pytest.mark.parametrize("algo", ["auto", "filtered", "multi_cta"])
@pytest.mark.parametrize("pattern", ["tie_heavy", "pivot_tie"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_top_k_deterministic_repeatability_tie_cases_by_algo(
    algo, pattern, dtype, set_topk_algo
):
    """Deterministic top-k should be repeatable and valid under tie pressure."""
    if dtype == torch.bfloat16:
        _require_sm80_for_bf16()

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    batch_size = 4
    vocab_size = 16384
    k = 256
    num_runs = 20
    device = "cuda"
    gt_count = 0

    if pattern == "tie_heavy":
        base = (
            torch.arange(vocab_size, device=device, dtype=torch.float32) % 32
        ) / 32.0
        logits = base.unsqueeze(0).repeat(batch_size, 1).to(dtype).contiguous()
    else:  # pivot_tie
        logits = torch.ones(batch_size, vocab_size, device=device, dtype=dtype)
        gt_count = max(1, min(k // 4, vocab_size // 8))
        logits[:, vocab_size - gt_count :] = 2.0

    ref_values, ref_indices = flashinfer.top_k(
        logits, k, deterministic=True, sorted=False
    )
    for _ in range(num_runs - 1):
        values, indices = flashinfer.top_k(logits, k, deterministic=True, sorted=False)
        _assert_repeatable_valid_topk_selection(
            logits, ref_values, ref_indices, values, indices, k, gt_count=gt_count
        )


@pytest.mark.parametrize("algo", ["filtered", "multi_cta"])
@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_deterministic_repeatability_tie_heavy_by_algo(
    algo, transform_mode, set_topk_algo
):
    """Deterministic transform APIs should be repeatable under tie-heavy input."""
    _require_sm80_for_bf16()

    if algo == "filtered" and not can_implement_filtered_topk():
        pytest.skip("Filtered top-k not supported on this device")

    set_topk_algo(algo)
    num_rows = 4
    max_len = 16384
    k = 256
    num_runs = 20
    device = "cuda"

    base = (torch.arange(max_len, device=device, dtype=torch.float32) % 32) / 32.0
    scores = base.unsqueeze(0).repeat(num_rows, 1).to(torch.bfloat16).contiguous()
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
    offsets = None
    if transform_mode == "ragged":
        offsets = torch.arange(
            0, num_rows * max_len, max_len, device=device, dtype=torch.int32
        )

    _assert_repeatable_transform_output(
        scores,
        k,
        transform_mode,
        num_runs=num_runs,
        deterministic=True,
        lengths=lengths,
        offsets=offsets,
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


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_deterministic_repeatability(transform_mode):
    """Deterministic transform APIs should be bitwise identical across runs."""
    num_rows = 8
    max_len = 8192
    k = 512
    num_runs = 30
    device = "cuda"

    pattern = (torch.arange(max_len, device=device, dtype=torch.float32) % 32) / 32.0
    scores = pattern.unsqueeze(0).repeat(num_rows, 1).contiguous()
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
    src_page_table = None
    offsets = None
    if transform_mode == "ragged":
        offsets = torch.arange(
            0, num_rows * max_len, max_len, device=device, dtype=torch.int32
        )

    _assert_repeatable_transform_output(
        scores,
        k,
        transform_mode,
        num_runs=num_runs,
        deterministic=True,
        lengths=lengths,
        src_page_table=src_page_table,
        offsets=offsets,
    )


@pytest.mark.parametrize("transform_mode", ["page_table", "ragged"])
def test_top_k_transform_deterministic_k1_remap(transform_mode):
    """Deterministic transform APIs must remap local top-1 positions correctly."""
    num_rows = 4
    max_len = 257
    device = "cuda"

    torch.manual_seed(0)
    scores = torch.randn(num_rows, max_len, device=device, dtype=torch.float32)
    lengths = torch.full((num_rows,), max_len, device=device, dtype=torch.int32)
    ref_idx = torch.topk(scores, 1, dim=-1).indices.to(torch.int32)
    src_page_table = None
    offsets = None

    if transform_mode == "page_table":
        src_page_table = (
            torch.arange(max_len, device=device, dtype=torch.int32)
            .unsqueeze(0)
            .repeat(num_rows, 1)
            .mul(3)
            .add(7)
            .contiguous()
        )
        ref = torch.gather(src_page_table, 1, ref_idx)
    else:
        offsets = torch.tensor([5, 1000, 2000, 3000], device=device, dtype=torch.int32)
        ref = ref_idx + offsets.unsqueeze(1)

    out = _run_transform(
        scores,
        1,
        transform_mode,
        lengths=lengths,
        deterministic=True,
        src_page_table=src_page_table,
        offsets=offsets,
    )
    assert torch.equal(out, ref)


def test_top_k_uint32_pointer_overflow():
    """Test top_k with batch*vocab > 2^32 bytes"""
    batch_size = 32769
    vocab_size = 131072
    k = 256

    required_bytes = batch_size * vocab_size * 2  # fp16
    free_mem = torch.cuda.mem_get_info("cuda")[0]
    if free_mem < int(required_bytes * 1.15):
        pytest.skip(
            f"Insufficient GPU memory: {free_mem / 1e9:.1f}GB free, "
            f"need ~{required_bytes / 1e9:.1f}GB"
        )

    torch.manual_seed(42)
    logits = torch.randn(batch_size, vocab_size, device="cuda", dtype=torch.float16)

    values, indices = flashinfer.top_k(logits, k)

    assert values.shape == (batch_size, k)
    assert indices.shape == (batch_size, k)

    # Only check the last row: its element offset (row_idx * vocab_size)
    # exceeds 2^32, so a uint32 overflow bug would corrupt this region.
    row_idx = batch_size - 1
    gathered = torch.gather(
        logits[row_idx : row_idx + 1], -1, indices[row_idx : row_idx + 1]
    )
    torch.testing.assert_close(values[row_idx : row_idx + 1], gathered)

    _, ref_indices = torch.topk(logits[row_idx : row_idx + 1], k, dim=-1)
    accuracy = compute_topk_accuracy(
        indices[row_idx : row_idx + 1].int(), ref_indices.int(), 1, k
    )
    assert accuracy >= 0.98, f"Last row accuracy {accuracy:.4f} < 0.98"


# ===================== topk_clusters_exact Tests =====================


def _require_sm100_or_sm103():
    major, minor = get_compute_capability(torch.device("cuda"))
    cc = major * 10 + minor
    if cc not in [100, 103]:
        pytest.skip("topk_clusters_exact requires SM100 or SM103 (Blackwell)")


@pytest.mark.parametrize("batch_size", [1, 16, 64])
@pytest.mark.parametrize("seq_len", [4096, 16384, 65536])
@pytest.mark.parametrize("k", [256, 1024, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output_values", [False, True])
@pytest.mark.parametrize("out_dtype", [torch.int32, torch.int64])
def test_topk_clusters_exact_correctness(
    batch_size, seq_len, k, dtype, output_values, out_dtype
):
    """Test topk_clusters_exact returns indices (and optionally values) matching torch.topk."""
    _require_sm100_or_sm103()
    if k > seq_len:
        pytest.skip("k should be less than seq_len")

    torch.manual_seed(42)
    device = "cuda"
    logits = torch.randn(batch_size, seq_len, device=device, dtype=dtype)

    indices, values = flashinfer.topk.topk_clusters_exact(
        logits, k, output_values=output_values, out_dtype=out_dtype
    )

    assert indices.shape == (batch_size, k)
    assert indices.dtype == out_dtype

    ref_values, ref_indices = torch.topk(logits, k, dim=-1)

    if output_values:
        assert values is not None
        assert values.shape == (batch_size, k)
        assert values.dtype == dtype

        abs_err = 0.125 if dtype == torch.bfloat16 else 1e-5
        rel_err = 0.1 if dtype == torch.bfloat16 else 1e-5
        torch.testing.assert_close(
            values.min(dim=-1).values,
            ref_values.min(dim=-1).values,
            rtol=rel_err,
            atol=abs_err,
        )
        torch.testing.assert_close(
            values.max(dim=-1).values,
            ref_values.max(dim=-1).values,
            rtol=rel_err,
            atol=abs_err,
        )
    else:
        assert values is None

    accuracy = compute_topk_accuracy(indices, ref_indices.int(), batch_size, k)
    acc = 0.95 if dtype == torch.bfloat16 else 0.99
    assert accuracy >= acc, f"Accuracy {accuracy:.4f} < {acc}"


@pytest.mark.parametrize("batch_size", [1, 16])
@pytest.mark.parametrize("seq_len", [4096, 16384])
@pytest.mark.parametrize("k", [256, 2048])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk_clusters_exact_variable_seq_lens(batch_size, seq_len, k, dtype):
    """Test topk_clusters_ragged_transform respects per-row variable seq_lens."""
    _require_sm100_or_sm103()
    if k > seq_len // 2:
        pytest.skip("k should be well below seq_len")

    torch.manual_seed(42)
    device = "cuda"
    logits = torch.randn(batch_size, seq_len, device=device, dtype=dtype)
    # Variable lengths: half rows get seq_len, half get seq_len // 2
    lengths_list = [seq_len if i % 2 == 0 else seq_len // 2 for i in range(batch_size)]
    seq_lens = torch.tensor(lengths_list, device=device, dtype=torch.int32)
    # Zero offsets: output indices are positions within each row
    offsets = torch.zeros(batch_size, device=device, dtype=torch.int32)

    indices = flashinfer.topk.topk_clusters_ragged_transform(
        logits, seq_lens, offsets, k
    )

    assert indices.shape == (batch_size, k)
    assert indices.dtype == torch.int32

    # Verify all indices are within [0, row_len) for each row
    for i in range(batch_size):
        row_len = lengths_list[i]
        assert torch.all(indices[i] >= 0) and torch.all(indices[i] < row_len), (
            f"Row {i}: indices out of [0, {row_len})"
        )

    # Verify accuracy per row
    for i in range(batch_size):
        row_len = lengths_list[i]
        ref = torch.topk(logits[i, :row_len], k).indices
        test_set = set(indices[i].cpu().numpy())
        ref_set = set(ref.cpu().numpy())
        row_accuracy = len(test_set & ref_set) / k
        acc = 0.95 if dtype == torch.bfloat16 else 0.99
        assert row_accuracy >= acc, f"Row {i} accuracy {row_accuracy:.4f} < {acc}"


@pytest.mark.parametrize("num_rows", [1, 16, 64])
@pytest.mark.parametrize("seq_len", [4096, 16384])
@pytest.mark.parametrize("k", [256, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk_clusters_page_table_transform(num_rows, seq_len, k, dtype):
    """Test topk_clusters_page_table_transform returns correct page table entries."""
    _require_sm100_or_sm103()
    if k > seq_len:
        pytest.skip("k should be less than seq_len")

    torch.manual_seed(42)
    device = "cuda"

    scores = torch.randn(num_rows, seq_len, device=device, dtype=dtype)
    src_page_table = torch.randint(
        0, 10000, (num_rows, seq_len), device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)

    output = flashinfer.topk.topk_clusters_page_table_transform(
        scores, lengths, src_page_table, k
    )

    assert output.shape == (num_rows, k)
    assert output.dtype == torch.int32

    ref_output = reference_page_table_transform(scores, src_page_table, lengths, k)
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)

    acc = 0.95 if dtype == torch.bfloat16 else 0.99
    assert accuracy >= acc, f"Accuracy {accuracy:.4f} < {acc}"


@pytest.mark.parametrize("num_rows", [1, 16, 64])
@pytest.mark.parametrize("seq_len", [4096, 16384])
@pytest.mark.parametrize("k", [256, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_topk_clusters_ragged_transform(num_rows, seq_len, k, dtype):
    """Test topk_clusters_ragged_transform returns correct indices with offsets."""
    _require_sm100_or_sm103()
    if k > seq_len:
        pytest.skip("k should be less than seq_len")

    torch.manual_seed(42)
    device = "cuda"

    scores = torch.randn(num_rows, seq_len, device=device, dtype=dtype)
    offsets = torch.arange(
        0, num_rows * seq_len, seq_len, device=device, dtype=torch.int32
    )
    lengths = torch.full((num_rows,), seq_len, device=device, dtype=torch.int32)

    output = flashinfer.topk.topk_clusters_ragged_transform(scores, lengths, offsets, k)

    assert output.shape == (num_rows, k)
    assert output.dtype == torch.int32

    ref_output = reference_ragged_transform(scores, offsets, lengths, k)
    accuracy = compute_transform_accuracy(output, ref_output, num_rows, k)
    acc = 0.95 if dtype == torch.bfloat16 else 0.99
    assert accuracy >= acc, f"Accuracy {accuracy:.4f} < {acc}"


if __name__ == "__main__":
    # Basic tests
    test_top_k(4, 32000, 256, torch.float32, flashinfer.TopKTieBreak.NONE)
    test_top_k_sorted(4, 32000, 256, torch.float32, flashinfer.TopKTieBreak.NONE)
    test_top_k_large_batch(64, 128512, 256, False, flashinfer.TopKTieBreak.NONE)

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
