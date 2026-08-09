# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for variable-length decode-step top-K APIs."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _top_k_varlen_reference(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    compress_ratio=1,
    next_n=1,
    deterministic=False,
    tie_break=0,
    row_starts=None,
    **_unused,
):
    """Per-row top-K with decode-length, score-window, and tie semantics."""
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")

    num_rows, max_seq_len = logits.shape
    if num_rows != seq_lens.shape[0] * next_n:
        raise ValueError(
            "logits.shape[0] must equal seq_lens.shape[0] * next_n, got "
            f"{num_rows} and {seq_lens.shape[0]} * {next_n}"
        )
    if next_n > 1:
        lengths = seq_lens.repeat_interleave(next_n)
        row_offsets = torch.arange(
            next_n, dtype=torch.int32, device=logits.device
        ).repeat(seq_lens.shape[0])
        lengths = (lengths - next_n + row_offsets + 1) // compress_ratio
    else:
        lengths = seq_lens // compress_ratio
    lengths = lengths.clamp(min=0, max=max_seq_len).to(torch.int32)
    if row_starts is not None:
        max_window_lengths = (max_seq_len - row_starts).clamp(min=0, max=max_seq_len)
        lengths = torch.minimum(lengths, max_window_lengths)

    indices = torch.full((num_rows, top_k), -1, dtype=torch.int32, device=logits.device)
    logits_f32 = logits.to(torch.float32)
    for r in range(num_rows):
        length = int(lengths[r].item())
        if length <= 0:
            continue
        row_start = int(row_starts[r].item()) if row_starts is not None else 0
        row = logits_f32[r, row_start : row_start + length]
        selected_count = min(top_k, length)
        if top_k >= length:
            idx = torch.arange(selected_count, dtype=torch.int32, device=logits.device)
        elif int(tie_break) == 1:
            idx = torch.argsort(row, descending=True, stable=True)[:selected_count].to(
                torch.int32
            )
        elif int(tie_break) == 2:
            reverse_idx = torch.argsort(
                torch.flip(row, dims=(0,)), descending=True, stable=True
            )[:selected_count]
            idx = (length - 1 - reverse_idx).to(torch.int32)
        else:
            idx = torch.topk(
                row, selected_count, largest=True, sorted=False
            ).indices.to(torch.int32)
        if deterministic:
            idx = torch.sort(idx).values
        indices[r, :selected_count] = idx
    return indices


def _top_k_varlen_check(
    reference_outputs,
    actual_outputs,
    logits=None,
    seq_lens=None,
    top_k=None,
    compress_ratio=1,
    next_n=1,
    deterministic=False,
    tie_break=0,
    row_starts=None,
    **_unused,
):
    """Tie-safe value check: every selected value must be >= the row's K-th largest.

    Exact set equality is not required because ties at the K-th boundary may be
    broken differently by the kernel vs the reference. Instead we verify that
    each selected index points to a value no smaller than the true K-th largest
    (matching the check used in tests/topk_varlen/test_topk_varlen.py::_check_correct).
    This requires the original logits and seq_lens, which the template passes
    via **_unused from the check call in the test.
    """
    act = (
        actual_outputs
        if isinstance(actual_outputs, torch.Tensor)
        else next(
            (item for item in actual_outputs if isinstance(item, torch.Tensor)),
            None,
        )
    )
    if act is None:
        return False
    if logits is None or seq_lens is None or top_k is None:
        ref = (
            reference_outputs
            if isinstance(reference_outputs, torch.Tensor)
            else next(
                (item for item in reference_outputs if isinstance(item, torch.Tensor)),
                None,
            )
        )
        if ref is None or ref.shape != act.shape:
            return False
        return torch.equal(
            torch.sort(ref, dim=-1).values,
            torch.sort(act, dim=-1).values,
        )

    num_rows, max_seq_len = logits.shape
    if next_n <= 0 or compress_ratio <= 0:
        return False
    if num_rows != seq_lens.shape[0] * next_n or act.shape != (num_rows, top_k):
        return False
    if next_n > 1:
        lengths = seq_lens.repeat_interleave(next_n)
        row_offsets = torch.arange(
            next_n, dtype=torch.int32, device=logits.device
        ).repeat(seq_lens.shape[0])
        lengths = (lengths - next_n + row_offsets + 1) // compress_ratio
    else:
        lengths = seq_lens // compress_ratio
    lengths = lengths.clamp(min=0, max=max_seq_len).to(torch.int32)
    if row_starts is not None:
        max_window_lengths = (max_seq_len - row_starts).clamp(min=0, max=max_seq_len)
        lengths = torch.minimum(lengths, max_window_lengths)

    logits_f32 = logits.to(torch.float32)
    for r in range(act.shape[0]):
        length = int(lengths[r].item())
        selected_count = min(top_k, max(length, 0))
        selected = act[r, :selected_count].to(torch.long)
        if not torch.all(act[r, selected_count:] == -1):
            return False
        if selected_count == 0:
            continue
        if torch.any(selected < 0) or torch.any(selected >= length):
            return False
        if torch.unique(selected).numel() != selected_count:
            return False
        if deterministic and not torch.all(selected[:-1] <= selected[1:]):
            return False
        if length <= top_k:
            expected = torch.arange(length, dtype=torch.long, device=act.device)
            if not torch.equal(selected, expected):
                return False
            continue

        if int(tie_break) != 0:
            ref = (
                reference_outputs
                if isinstance(reference_outputs, torch.Tensor)
                else next(
                    (
                        item
                        for item in reference_outputs
                        if isinstance(item, torch.Tensor)
                    ),
                    None,
                )
            )
            if ref is None or not torch.equal(
                torch.sort(selected).values,
                torch.sort(ref[r, :selected_count].to(torch.long)).values,
            ):
                return False
            continue

        row_start = int(row_starts[r].item()) if row_starts is not None else 0
        row = logits_f32[r, row_start : row_start + length]
        kth = torch.topk(row, top_k).values[-1]
        if torch.any(row[selected] < kth):
            return False
    return True


def _top_k_varlen_init(
    *,
    batch_size: int,
    num_requests=None,
    max_seq_len: int = 8192,
    top_k: int = 1024,
    compress_ratio: int = 1,
    next_n: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.top_k_varlen`` (radix_cutlass backend, no pre_idx).

    Uses the ``radix_cutlass`` (masked CUTLASS radix) backend so the example
    runs on any GPU.
    ``seq_lens`` is randomised so every derived score-row length has at least
    ``top_k`` valid entries under the requested compression and ``next_n``.
    ``max_seq_len`` is padded to the next multiple of 8 (fp16/bf16 alignment).
    """
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if num_requests is None:
        if batch_size % next_n != 0:
            raise ValueError("batch_size must be divisible by next_n")
        num_requests = batch_size // next_n
    if batch_size != num_requests * next_n:
        raise ValueError("batch_size must equal num_requests * next_n")

    max_seq_len = (max_seq_len + 7) // 8 * 8
    torch.manual_seed(seed)
    logits = torch.randn(batch_size, max_seq_len, dtype=torch.bfloat16, device=device)
    min_effective_len = min(top_k + 1, max_seq_len)
    if min_effective_len < max_seq_len:
        effective_lens = torch.randint(
            min_effective_len,
            max_seq_len + 1,
            (num_requests,),
            dtype=torch.int32,
            device=device,
        )
    else:
        effective_lens = torch.full(
            (num_requests,), max_seq_len, dtype=torch.int32, device=device
        )
    seq_lens = effective_lens * compress_ratio + next_n - 1
    return {
        "logits": logits,
        "seq_lens": seq_lens,
        "top_k": top_k,
        "compress_ratio": compress_ratio,
        "next_n": next_n,
        "backend": "radix_cutlass",
    }


top_k_varlen_trace = TraceTemplate(
    op_type="topk",
    name_prefix="top_k_varlen",
    description=(
        "Decode-step top-K selection over batched logits with per-request seq_lens. "
        "GVR (Blackwell sm_100+) or masked-radix fallback backend."
    ),
    axes={
        "batch_size": Var(description="Number of decode score rows."),
        "num_requests": Var(description="Number of request-level lengths."),
        "max_seq_len": Const(abbrev="n", description="Logits row width (padded)."),
        "top_k": Const(abbrev="k", description="Number of top elements per row."),
    },
    inputs={
        "logits": Tensor(
            ["batch_size", "max_seq_len"],
            description="Decode-step attention logits (bfloat16 / float16 / float32).",
        ),
        "seq_lens": Tensor(
            ["num_requests"],
            dtype="int32",
            description="Effective KV-cache length per request.",
        ),
        "top_k": Scalar("int32", description="K — number of top elements to select."),
        "pre_idx": Tensor(
            ["num_requests", "top_k"],
            dtype="int32",
            optional=True,
            description="Previous-step top-K indices (GVR warm-start hint).",
        ),
        "compress_ratio": Scalar("int32", optional=True),
        "next_n": Scalar("int32", optional=True),
        "deterministic": Scalar("bool", optional=True),
        "tie_break": Scalar("int32", optional=True),
        "row_starts": Tensor(["batch_size"], dtype="int32", optional=True),
    },
    outputs={
        "indices": Tensor(
            ["batch_size", "top_k"],
            dtype="int32",
            description="Selected top-K indices per row.",
        ),
    },
    tags=["status:verified"],
    reference=_top_k_varlen_reference,
    check=_top_k_varlen_check,
    init=_top_k_varlen_init,
)


# ── Variable-length top-K + compact page-table transform ──────────────────


@torch.no_grad()
def _top_k_varlen_page_table_transform_reference(
    logits,
    src_page_table,
    seq_lens,
    top_k,
    row_to_batch=None,
    compress_ratio=1,
    next_n=1,
    deterministic=False,
    tie_break=0,
    row_starts=None,
    page_table_row_starts=None,
    page_size=1,
    **_unused,
):
    """Select window-local indices, then translate them through compact pages."""
    if page_size > 1 and row_starts is not None and page_table_row_starts is None:
        raise ValueError(
            "page_table_row_starts is required with page_size > 1 and row_starts"
        )
    raw_indices = _top_k_varlen_reference(
        logits,
        seq_lens,
        top_k,
        compress_ratio=compress_ratio,
        next_n=next_n,
        deterministic=deterministic,
        tie_break=tie_break,
        row_starts=row_starts,
    )
    output = torch.full_like(raw_indices, -1)
    for row_idx in range(logits.shape[0]):
        selected = raw_indices[row_idx]
        selected = selected[selected >= 0]
        if selected.numel() == 0:
            continue
        page_start = (
            int(page_table_row_starts[row_idx].item())
            if page_table_row_starts is not None
            else (int(row_starts[row_idx].item()) if row_starts is not None else 0)
        )
        batch_idx = (
            int(row_to_batch[row_idx].item())
            if row_to_batch is not None
            else row_idx // next_n
        )
        physical_pages = src_page_table[
            batch_idx, page_start + selected.to(torch.long) // page_size
        ]
        output[row_idx, : selected.numel()] = (
            physical_pages * page_size + selected % page_size
        )
    return output


_top_k_varlen_page_table_transform_reference._trace_reference_dependencies = (
    _top_k_varlen_reference,
)


def _top_k_varlen_page_table_transform_check(
    reference_outputs, actual_outputs, **_unused
):
    """Compare the selected physical-index multisets, ignoring output order."""

    def first_tensor(outputs):
        if isinstance(outputs, torch.Tensor):
            return outputs
        if isinstance(outputs, (tuple, list)):
            return next(
                (item for item in outputs if isinstance(item, torch.Tensor)), None
            )
        return None

    reference = first_tensor(reference_outputs)
    actual = first_tensor(actual_outputs)
    if reference is None or actual is None or reference.shape != actual.shape:
        return False
    return torch.equal(
        torch.sort(reference, dim=-1).values,
        torch.sort(actual, dim=-1).values,
    )


def _top_k_varlen_page_table_transform_init(
    *,
    num_rows: int,
    num_requests=None,
    batch_size=None,
    max_pages_per_seq=None,
    max_seq_len: int = 8192,
    top_k: int = 1024,
    page_size: int = 64,
    next_n: int = 1,
    compress_ratio: int = 1,
    device: str = "cuda",
    seed: int = 0,
):
    """Build a compact-page native-radix trace without side outputs."""
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if compress_ratio <= 0:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if num_requests is None:
        if num_rows % next_n != 0:
            raise ValueError("num_rows must be divisible by next_n")
        num_requests = num_rows // next_n
    if num_rows != num_requests * next_n:
        raise ValueError("num_rows must equal num_requests * next_n")
    if batch_size is None:
        batch_size = num_requests

    min_pages = (max_seq_len + page_size - 1) // page_size
    if max_pages_per_seq is None:
        max_pages_per_seq = min_pages
    else:
        max_pages_per_seq = max(max_pages_per_seq, min_pages)

    torch.manual_seed(seed)
    logits = torch.randn(num_rows, max_seq_len, dtype=torch.float32, device=device)
    min_effective_len = min(top_k + 1, max_seq_len)
    if min_effective_len < max_seq_len:
        effective_lens = torch.randint(
            min_effective_len,
            max_seq_len + 1,
            (num_requests,),
            dtype=torch.int32,
            device=device,
        )
    else:
        effective_lens = torch.full(
            (num_requests,), max_seq_len, dtype=torch.int32, device=device
        )
    seq_lens = effective_lens * compress_ratio + next_n - 1

    num_pages = batch_size * max_pages_per_seq
    src_page_table = torch.randperm(
        num_pages, dtype=torch.int32, device=device
    ).reshape(batch_size, max_pages_per_seq)
    result = {
        "logits": logits,
        "src_page_table": src_page_table,
        "seq_lens": seq_lens,
        "top_k": top_k,
        "compress_ratio": compress_ratio,
        "next_n": next_n,
        "page_size": page_size,
        "backend": "radix",
    }
    if batch_size != num_requests:
        result["row_to_batch"] = (
            torch.arange(num_rows, dtype=torch.int32, device=device) // next_n
        ) % batch_size
    return result


class _TopKVarlenPageTableTransformTraceTemplate(TraceTemplate):
    """Keep the compact-page API's default page size locally extractable."""

    def _build_axis_extractors(self):
        extractors = super()._build_axis_extractors()

        def extract_page_size(kwargs):
            page_size = kwargs.get("page_size", 1)
            return 1 if page_size is None else int(page_size)

        extractors["page_size"] = extract_page_size
        return extractors


top_k_varlen_page_table_transform_trace = _TopKVarlenPageTableTransformTraceTemplate(
    op_type="topk",
    name_prefix="top_k_varlen_page_table_transform",
    description=(
        "Variable-length decode-step top-k selection followed by compact "
        "page-table translation from window-local to physical indices."
    ),
    axes={
        "num_rows": Var(description="Number of score rows."),
        "num_requests": Var(description="Number of request-level lengths."),
        "max_seq_len": Const(abbrev="n", description="Logits row width."),
        "batch_size": Var(description="Number of source page-table rows."),
        "max_pages_per_seq": Var(description="Source page-table row width."),
        "top_k": Const(abbrev="k", description="Selections per score row."),
        "page_size": Const(abbrev="ps", description="Positions per page."),
    },
    inputs={
        "logits": Tensor(["num_rows", "max_seq_len"]),
        "src_page_table": Tensor(["batch_size", "max_pages_per_seq"], dtype="int32"),
        "seq_lens": Tensor(["num_requests"], dtype="int32"),
        "top_k": Scalar("int32"),
        "row_to_batch": Tensor(["num_rows"], dtype="int32", optional=True),
        "compress_ratio": Scalar("int32", optional=True),
        "next_n": Scalar("int32", optional=True),
        "deterministic": Scalar("bool", optional=True),
        "tie_break": Scalar("int32", optional=True),
        "row_starts": Tensor(["num_rows"], dtype="int32", optional=True),
        "page_table_row_starts": Tensor(["num_rows"], dtype="int32", optional=True),
        "page_size": Scalar("int32", optional=True),
    },
    outputs={
        "physical_indices": Tensor(["num_rows", "top_k"], dtype="int32"),
    },
    tags=["status:verified", "sparse"],
    reference=_top_k_varlen_page_table_transform_reference,
    check=_top_k_varlen_page_table_transform_check,
    init=_top_k_varlen_page_table_transform_init,
)


def top_k_varlen_page_table_transform_trace_dispatch(
    save_dir=None, name=None, **kwargs
):
    """Trace only the primary physical-index return without side outputs."""
    if kwargs.get("return_values", False) or kwargs.get("out_raw_indices") is not None:
        return None
    return top_k_varlen_page_table_transform_trace


top_k_varlen_page_table_transform_trace_dispatch.templates = (  # type: ignore[attr-defined]
    top_k_varlen_page_table_transform_trace,
)
