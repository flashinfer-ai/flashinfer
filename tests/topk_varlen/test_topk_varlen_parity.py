# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parity coverage for the radix-backed ``top_k_varlen`` transform APIs."""

import pytest
import torch

try:
    import flashinfer
    from flashinfer.topk import can_implement_filtered_topk
    from flashinfer.topk_varlen.topk_varlen import _CUTE_DSL_AVAILABLE
    from flashinfer.utils import BackendSupportedError, get_compute_capability

    _FLASHINFER_AVAILABLE = True
except ImportError:
    _FLASHINFER_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(
        not _FLASHINFER_AVAILABLE,
        reason="flashinfer not installed",
    ),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA"),
]


def _require_filtered_topk() -> None:
    if not can_implement_filtered_topk():
        pytest.skip("Tie-break modes require filtered top-k support")


def _make_boundary_tie_case():
    """Return logits with strict winners followed by a large K-th-value tie."""
    num_rows, width, top_k = 2, 64, 8
    logits = torch.zeros((num_rows, width), device="cuda", dtype=torch.float32)
    strict_indices = (11, 37)
    logits[:, strict_indices[0]] = 3
    logits[:, strict_indices[1]] = 2
    seq_lens = torch.full((num_rows,), width, device="cuda", dtype=torch.int32)
    return logits, seq_lens, top_k, strict_indices


def _expected_boundary_indices(width, top_k, strict_indices, prefer_large):
    boundary_candidates = [i for i in range(width) if i not in strict_indices]
    num_boundary = top_k - len(strict_indices)
    chosen = (
        boundary_candidates[-num_boundary:]
        if prefer_large
        else boundary_candidates[:num_boundary]
    )
    return torch.tensor(
        sorted((*strict_indices, *chosen)),
        device="cuda",
        dtype=torch.int32,
    )


@pytest.mark.parametrize(
    ("tie_break", "prefer_large"),
    [
        (1, False),
        (2, True),
    ],
    ids=["small", "large"],
)
def test_top_k_varlen_exact_kth_boundary_tie_set(tie_break, prefer_large):
    """SMALL/LARGE choose the exact local-index subset at the K-th boundary."""
    _require_filtered_topk()
    logits, seq_lens, top_k, strict_indices = _make_boundary_tie_case()

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        tie_break=tie_break,
        backend="auto",
    )

    expected = _expected_boundary_indices(
        logits.shape[1], top_k, strict_indices, prefer_large
    ).expand(logits.shape[0], -1)
    assert values is None
    assert torch.equal(indices.sort(dim=-1).values, expected)


@pytest.mark.parametrize(
    ("tie_break", "prefer_large"),
    [
        (1, False),
        (2, True),
    ],
    ids=["small", "large"],
)
def test_top_k_varlen_deterministic_tie_output_order(tie_break, prefer_large):
    """Deterministic mode emits the selected local indices canonically ordered."""
    _require_filtered_topk()
    logits, seq_lens, top_k, strict_indices = _make_boundary_tie_case()

    indices_a, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        deterministic=True,
        tie_break=tie_break,
        backend="auto",
    )
    indices_b, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        deterministic=True,
        tie_break=tie_break,
        backend="auto",
    )

    expected = _expected_boundary_indices(
        logits.shape[1], top_k, strict_indices, prefer_large
    ).expand(logits.shape[0], -1)
    assert torch.equal(indices_a, expected)
    assert torch.equal(indices_b, expected)


def test_top_k_varlen_skip_check_auto_still_routes_tie_semantics():
    """skip_check bypasses validation, not auto routing for CUTLASS-only modes."""
    _require_filtered_topk()
    logits, seq_lens, top_k, strict_indices = _make_boundary_tie_case()

    indices, _ = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        deterministic=True,
        tie_break=flashinfer.TopKTieBreak.SMALL,
        skip_check=True,
    )

    expected = _expected_boundary_indices(
        logits.shape[1], top_k, strict_indices, prefer_large=False
    ).expand(logits.shape[0], -1)
    assert torch.equal(indices, expected)


def test_top_k_varlen_row_starts_values_and_auto_fallback():
    """Values use score-window starts while returned indices remain window-local."""
    top_k, width = 4, 32
    logits = torch.full((2, width), -100, device="cuda", dtype=torch.float32)
    row_starts = torch.tensor([3, 9], device="cuda", dtype=torch.int32)
    windows = [
        torch.tensor([1, 9, 3, 7, 2, 8, 0, 6], device="cuda", dtype=logits.dtype),
        torch.tensor([5, 0, 4, 10, 2, 7, 1], device="cuda", dtype=logits.dtype),
    ]
    seq_lens = torch.tensor(
        [window.numel() for window in windows], device="cuda", dtype=torch.int32
    )
    for row, window in enumerate(windows):
        start = int(row_starts[row].item())
        logits[row, start : start + window.numel()] = window

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        return_values=True,
        row_starts=row_starts,
        backend="auto",
    )

    expected_indices = torch.stack(
        [torch.topk(window, top_k, sorted=False).indices.int() for window in windows]
    )
    assert torch.equal(
        indices.sort(dim=-1).values,
        expected_indices.sort(dim=-1).values,
    )
    gather_indices = row_starts[:, None].long() + indices.long()
    expected_values = torch.gather(logits, 1, gather_indices)
    torch.testing.assert_close(values, expected_values)


def test_top_k_varlen_row_window_clamp_and_empty_value_padding():
    """Score windows stop at row width and empty-window gathers stay in bounds."""
    width, top_k = 16, 2
    logits = torch.arange(2 * width, device="cuda", dtype=torch.float32).view(2, width)
    seq_lens = torch.tensor([10, 7], device="cuda", dtype=torch.int32)
    row_starts = torch.tensor([12, width], device="cuda", dtype=torch.int32)

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        top_k,
        return_values=True,
        row_starts=row_starts,
        backend="auto",
    )

    assert set(indices[0].cpu().tolist()) == {2, 3}
    torch.testing.assert_close(
        values[0].sort().values,
        logits[0, 14:16].sort().values,
    )
    assert torch.equal(indices[1], torch.full_like(indices[1], -1))
    assert torch.equal(values[1], torch.zeros_like(values[1]))


def test_top_k_varlen_next_n_negative_effective_lengths_clamp_to_zero():
    """Speculative rows before an empty request never wrap length to uint32."""
    logits = torch.randn((2, 16), device="cuda", dtype=torch.float32)
    seq_lens = torch.zeros(1, device="cuda", dtype=torch.int32)

    indices, values = flashinfer.top_k_varlen(
        logits,
        seq_lens,
        2,
        next_n=2,
        return_values=True,
        backend="radix_cutlass",
    )

    assert torch.equal(indices, torch.full_like(indices, -1))
    assert torch.equal(values, torch.zeros_like(values))


def test_top_k_varlen_compact_page_table_raw_values_and_outputs():
    """Compact physical, raw, and value outputs stay positionally aligned."""
    num_rows, width, top_k, page_size = 2, 20, 5, 4
    logits = torch.full((num_rows, width), -100, device="cuda", dtype=torch.float32)
    selected = ((1, 6, 11, 12, 16), (0, 4, 5, 8, 12))
    for row, row_indices in enumerate(selected):
        logits[row, torch.tensor(row_indices, device="cuda")] = torch.arange(
            top_k, 0, -1, device="cuda", dtype=logits.dtype
        )
    seq_lens = torch.tensor([17, 13], device="cuda", dtype=torch.int32)
    src_page_table = torch.tensor(
        [[9, 2, 17, 5, 11], [30, 25, 40, 35, 50]],
        device="cuda",
        dtype=torch.int32,
    )
    out = torch.empty((num_rows, top_k), device="cuda", dtype=torch.int32)
    out_raw = torch.empty_like(out)
    out_values = torch.empty_like(out, dtype=logits.dtype)

    physical, values = flashinfer.top_k_varlen_page_table_transform(
        logits,
        src_page_table,
        seq_lens,
        top_k,
        return_values=True,
        out=out,
        out_values=out_values,
        page_size=page_size,
        out_raw_indices=out_raw,
        backend="auto",
    )

    expected_raw = torch.tensor(selected, device="cuda", dtype=torch.int32)
    assert physical.data_ptr() == out.data_ptr()
    assert values.data_ptr() == out_values.data_ptr()
    assert torch.equal(
        out_raw.sort(dim=-1).values,
        expected_raw.sort(dim=-1).values,
    )
    physical_pages = torch.gather(
        src_page_table,
        1,
        (out_raw // page_size).long(),
    )
    expected_physical = physical_pages * page_size + out_raw % page_size
    assert torch.equal(physical, expected_physical)
    torch.testing.assert_close(values, torch.gather(logits, 1, out_raw.long()))


def test_top_k_varlen_page_table_independent_score_and_page_starts():
    """Score starts and compact page-table starts keep their distinct units."""
    num_rows, width, top_k, page_size = 2, 24, 3, 4
    logits = torch.full((num_rows, width), -100, device="cuda", dtype=torch.float32)
    row_starts = torch.tensor([3, 5], device="cuda", dtype=torch.int32)
    page_table_row_starts = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    row_to_batch = torch.tensor([1, 0], device="cuda", dtype=torch.int32)
    selected = ((1, 6, 9), (0, 4, 8))
    for row, local_indices in enumerate(selected):
        absolute = row_starts[row].long() + torch.tensor(local_indices, device="cuda")
        logits[row, absolute] = torch.tensor(
            [3, 2, 1], device="cuda", dtype=logits.dtype
        )
    seq_lens = torch.tensor([12, 10], device="cuda", dtype=torch.int32)
    src_page_table = torch.tensor(
        [[90, 13, 44, 27, 61, 8], [70, 32, 55, 19, 81, 6]],
        device="cuda",
        dtype=torch.int32,
    )
    out_raw = torch.empty((num_rows, top_k), device="cuda", dtype=torch.int32)

    physical, values = flashinfer.top_k_varlen_page_table_transform(
        logits,
        src_page_table,
        seq_lens,
        top_k,
        row_to_batch=row_to_batch,
        return_values=True,
        row_starts=row_starts,
        page_table_row_starts=page_table_row_starts,
        page_size=page_size,
        out_raw_indices=out_raw,
    )

    expected_raw = torch.tensor(selected, device="cuda", dtype=torch.int32)
    assert torch.equal(
        out_raw.sort(dim=-1).values,
        expected_raw.sort(dim=-1).values,
    )
    page_columns = page_table_row_starts[:, None] + out_raw // page_size
    physical_pages = src_page_table[row_to_batch[:, None], page_columns.long()]
    assert torch.equal(physical, physical_pages * page_size + out_raw % page_size)
    absolute = row_starts[:, None].long() + out_raw.long()
    torch.testing.assert_close(values, torch.gather(logits, 1, absolute))


def test_top_k_varlen_page_table_next_n_default_row_to_batch():
    """Omitted row_to_batch maps speculative row i to request i // next_n."""
    num_requests, next_n, width, top_k, page_size = 2, 2, 16, 3, 4
    num_rows = num_requests * next_n
    logits = torch.full((num_rows, width), -100, device="cuda", dtype=torch.float32)
    selected = torch.tensor([0, 5, 10], device="cuda")
    logits[:, selected] = torch.tensor([3, 2, 1], device="cuda", dtype=logits.dtype)
    seq_lens = torch.tensor([16, 14], device="cuda", dtype=torch.int32)
    src_page_table = torch.tensor(
        [[10, 11, 12, 13], [100, 101, 102, 103]],
        device="cuda",
        dtype=torch.int32,
    )
    out_raw = torch.empty((num_rows, top_k), device="cuda", dtype=torch.int32)

    physical, values = flashinfer.top_k_varlen_page_table_transform(
        logits,
        src_page_table,
        seq_lens,
        top_k,
        next_n=next_n,
        page_size=page_size,
        out_raw_indices=out_raw,
        backend="auto",
    )

    expected_raw = selected.int().expand(num_rows, -1)
    assert values is None
    assert torch.equal(
        out_raw.sort(dim=-1).values,
        expected_raw.sort(dim=-1).values,
    )
    request_ids = torch.arange(num_rows, device="cuda") // next_n
    physical_pages = src_page_table[
        request_ids[:, None],
        (out_raw // page_size).long(),
    ]
    expected_physical = physical_pages * page_size + out_raw % page_size
    assert torch.equal(physical, expected_physical)


def _native_backend_available(backend: str) -> bool:
    if not _CUTE_DSL_AVAILABLE:
        return False
    major, minor = get_compute_capability(torch.device("cuda"))
    return flashinfer.top_k_varlen.is_backend_supported(backend, major * 10 + minor)


@pytest.mark.parametrize("backend", ["radix", "gvr"])
@pytest.mark.parametrize("feature", ["deterministic", "tie_break", "row_starts"])
def test_top_k_varlen_explicit_native_backend_rejects_radix_only_features(
    backend, feature
):
    """Explicit native backends reject options implemented only by radix_cutlass."""
    if not _native_backend_available(backend):
        pytest.skip(f"{backend} is not available on this device")

    num_rows, width, top_k = 2, 1024, 512
    logits = torch.randn((num_rows, width), device="cuda", dtype=torch.bfloat16)
    seq_lens = torch.full((num_rows,), width - 1, device="cuda", dtype=torch.int32)
    pre_idx = torch.zeros((num_rows, top_k), device="cuda", dtype=torch.int32)
    kwargs = {"pre_idx": pre_idx}
    if feature == "deterministic":
        kwargs["deterministic"] = True
    elif feature == "tie_break":
        kwargs["tie_break"] = flashinfer.TopKTieBreak.SMALL
    else:
        kwargs["row_starts"] = torch.ones(num_rows, device="cuda", dtype=torch.int32)

    with pytest.raises(
        (BackendSupportedError, ValueError),
        match="not supported|does not yet support",
    ):
        flashinfer.top_k_varlen(
            logits,
            seq_lens,
            top_k,
            backend=backend,
            **kwargs,
        )
