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

"""TraceTemplate for top_k_varlen (per-row varlen top-K: decode rows or
prefill windows)."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _top_k_varlen_reference(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    row_starts=None,
    absolute_indices=False,
    page_table=None,
    page_size=None,
    next_n=1,
    **_unused,
):
    """Per-row top-K with seq_lens masking (window ``[row_starts[r],
    row_starts[r] + seq_lens[r])`` with window-local indices when
    ``row_starts`` is given, absolute columns with ``absolute_indices``).
    Reference uses torch.topk on each row."""
    page_size = 1 if page_size is None else page_size
    num_rows, N = logits.shape
    indices = torch.empty(num_rows, top_k, dtype=torch.int32, device=logits.device)
    logits_f32 = logits.to(torch.float32)
    for r in range(num_rows):
        # the kernel clamps a negative start to 0 and a start past the width
        # to the width (an empty window); the window end is clamped to the width
        s = min(max(int(row_starts[r].item()), 0), N) if row_starts is not None else 0
        n = max(min(s + int(seq_lens[r].item()), N) - s, 0)
        row = logits_f32[r, s : s + n]
        _, idx = torch.topk(row, min(top_k, n), largest=True, sorted=False)
        idx = idx.to(torch.int32) + (s if absolute_indices else 0)
        if page_table is not None:
            pt = page_table[r // next_n].to(torch.int64)
            i64 = idx.to(torch.int64)
            idx = ((pt[i64 // page_size] * page_size) + i64 % page_size).to(torch.int32)
        indices[r, : len(idx)] = idx
        if len(idx) < top_k:
            indices[r, len(idx) :] = -1
    return indices


def _top_k_varlen_check(
    reference_outputs,
    actual_outputs,
    logits=None,
    seq_lens=None,
    top_k=None,
    row_starts=None,
    absolute_indices=False,
    page_table=None,
    page_size=None,
    next_n=1,
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
    act = actual_outputs if not isinstance(actual_outputs, list) else actual_outputs[0]
    if logits is None or seq_lens is None or top_k is None:
        # Fallback to exact set equality when logits are unavailable.
        ref = (
            reference_outputs
            if not isinstance(reference_outputs, list)
            else reference_outputs[0]
        )
        if ref.shape != act.shape:
            return False
        for r in range(ref.shape[0]):
            ref_set = set(ref[r].cpu().tolist())
            act_set = set(act[r].cpu().tolist())
            ref_set.discard(-1)
            act_set.discard(-1)
            if ref_set != act_set:
                return False
        return True

    logits_f32 = logits.to(torch.float32)
    N = logits_f32.shape[1]
    if page_table is not None:
        ps = 1 if page_size is None else int(page_size)
        pt64 = page_table.to(torch.int64)
    for r in range(act.shape[0]):
        s = min(max(int(row_starts[r].item()), 0), N) if row_starts is not None else 0
        n = max(min(s + int(seq_lens[r].item()), N) - s, 0)
        sel = act[r].long()  # window-local indices (absolute: shift back by s)
        if page_table is not None:
            # paged output holds physical KV slots: map every valid slot back to
            # its logits column through the request's page map (one physical
            # page per logical page, so the map inverts), then validate the
            # columns exactly like the other frames -- ties at the K-th value
            # may be broken differently from the reference and still be correct.
            # Only the pages the row's length spans take part: later table
            # entries are padding (and may repeat a live page id). A slot
            # outside the spanned pages is a failure.
            row_pt = pt64[r // next_n][: (n + ps - 1) // ps]
            valid = sel >= 0
            slots = sel[valid]
            # an all-pad row (an empty request, or a full row wrongly padded)
            # has nothing to map; the pad checks below decide it. A slot in a
            # row that spans no pages is a failure, not an empty reduction.
            if slots.numel():
                if row_pt.numel() == 0:
                    return False
                phys = slots // ps
                if bool((phys > row_pt.max()).any()):
                    return False
                inv = torch.full(
                    (int(row_pt.max().item()) + 1,),
                    -1,
                    dtype=torch.int64,
                    device=sel.device,
                )
                inv[row_pt] = torch.arange(row_pt.numel(), device=sel.device)
                logical = inv[phys]
                if bool((logical < 0).any()):
                    return False
                sel = sel.clone()
                sel[valid] = logical * ps + slots % ps
        if absolute_indices:
            sel = torch.where(sel >= 0, sel - s, sel)  # -1 pads stay -1
        if n < top_k:
            # short window: every window position exactly once (the identity
            # set, in any order) and top_k - n pads of -1 -- never skipped
            valid = sel[sel >= 0]
            if int((sel < 0).sum()) != top_k - n or valid.numel() != n:
                return False
            if n and not torch.equal(
                torch.sort(valid).values, torch.arange(n, device=sel.device)
            ):
                return False
            continue
        row = logits_f32[r, s : s + n]
        kth = torch.topk(row, top_k).values[-1].item()
        # a -1 pad in a full row, or any index outside the window, is a
        # failure -- not a wrap-around gather or a device-side index assert
        if bool((sel < 0).any()) or bool((sel >= n).any()):
            return False
        if (row[sel] < kth - 1e-5).any():
            return False
    return True


def _top_k_varlen_init(
    *,
    batch_size: int,
    max_seq_len: int = 8192,
    top_k: int = 1024,
    max_pages: int = 0,  # paged-output axis; the example never builds a page table
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.top_k_varlen`` (radix_cutlass backend, no pre_idx).

    Uses the ``radix_cutlass`` (masked CUTLASS radix) backend so the example
    runs on any GPU.
    ``seq_lens`` is randomised in ``[top_k + 1, max_seq_len]`` to guarantee
    that every row has at least ``top_k`` valid entries.
    ``max_seq_len`` is padded to the next multiple of 8 (fp16/bf16 alignment).
    """
    max_seq_len = (max_seq_len + 7) // 8 * 8
    torch.manual_seed(seed)
    logits = torch.randn(batch_size, max_seq_len, dtype=torch.bfloat16, device=device)
    seq_lens = torch.randint(
        top_k + 1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    return {
        "logits": logits,
        "seq_lens": seq_lens,
        "top_k": top_k,
        "backend": "radix_cutlass",
    }


top_k_varlen_trace = TraceTemplate(
    op_type="topk",
    name_prefix="top_k_varlen",
    description=(
        "Per-row top-K over batched logits with per-request seq_lens: decode rows, "
        "or windows of a packed prefill-indexer layout when row_starts is given. "
        "Backends: gvr, gvr_2, radix, radix_filter (CuTe DSL; see top_k_varlen for "
        "each one's GPU list) and the masked-radix radix_cutlass fallback (every GPU)."
    ),
    axes={
        "batch_size": Var(
            description=(
                "Number of rows: requests in decode mode, query tokens in windowed "
                "(prefill) mode."
            )
        ),
        # the width axis is num_cols (the logits shape), distinct from the API's
        # length bound max_window_len (a scalar input below) and from
        # init()'s max_seq_len, which sizes the example buffer
        "num_cols": Const(abbrev="n", description="Logits row width (padded)."),
        "top_k": Const(abbrev="k", description="Number of top elements per row."),
        "max_pages": Var(
            description="Page-table entries per request (paged output only)."
        ),
    },
    inputs={
        "logits": Tensor(
            ["batch_size", "num_cols"],
            description="Scores, one row per query token (bfloat16 / float16 / float32).",
        ),
        "seq_lens": Tensor(
            ["batch_size"],
            dtype="int32",
            description=(
                "Decode mode: each request's KV length in tokens. Windowed (prefill) "
                "mode: each row's window length in columns."
            ),
        ),
        "top_k": Scalar("int32", description="K — number of top elements to select."),
        "pre_idx": Tensor(
            ["batch_size", "top_k"],
            dtype="int32",
            optional=True,
            description=(
                "Decode mode only: a hint of each request's top-k indices, a warm "
                "start for the backends that take one."
            ),
        ),
        "row_starts": Tensor(
            ["batch_size"],
            dtype="int32",
            optional=True,
            description=(
                "Windowed (prefill) mode: row r ranks "
                "logits[r, row_starts[r] : row_starts[r] + seq_lens[r]] and "
                "returns window positions (logits columns with "
                "absolute_indices=True)."
            ),
        ),
        "absolute_indices": Scalar(
            "bool",
            optional=True,
            description=(
                "Windowed (prefill) mode: return logits columns instead of window "
                "positions."
            ),
        ),
        "max_window_len": Scalar(
            "int32",
            optional=True,
            description=(
                "Upper bound on every seq_lens entry, in its units (window columns "
                "in windowed mode, KV tokens in decode mode; defaults to what the "
                "logits width holds); fixed for a captured graph, so it must hold "
                "at every replay."
            ),
        ),
        "page_table": Tensor(
            ["batch_size", "max_pages"],
            dtype="int32",
            optional=True,
            description="Paged decode output: indices leave as page_table[req, c // page_size] * page_size + c % page_size (gvr_2 only).",
        ),
        "page_size": Scalar(
            "int32",
            optional=True,
            description="Columns per page_table entry (power of two).",
        ),
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
