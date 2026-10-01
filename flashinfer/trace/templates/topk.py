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

"""TraceTemplate for top_k_varlen (GVR / radix decode-step top-K)."""

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _paged_slots(page_table, page_size, cols):
    """Physical KV slots of logical columns ``cols`` (one row of the page table):
    ``page_table[c // page_size] * page_size + c % page_size``."""
    pt = page_table.to(torch.int64)
    return pt[cols // page_size] * page_size + cols % page_size


@torch.no_grad()
def _top_k_varlen_reference(
    logits, seq_lens, top_k, pre_idx=None, page_table=None, page_size=1, **_unused
):
    """Per-row top-K with seq_lens masking. Reference uses torch.topk on each row.

    With ``page_table`` the selected columns are reported as physical KV slots
    (the fused paged-output contract), otherwise as logical column indices.
    """
    num_rows, N = logits.shape
    indices = torch.empty(num_rows, top_k, dtype=torch.int32, device=logits.device)
    logits_f32 = logits.to(torch.float32)
    for r in range(num_rows):
        n = min(int(seq_lens[r].item()), N)
        if page_table is not None:
            # the kernels clamp a row to the pages the table covers
            n = min(n, page_table.shape[1] * page_size)
        row = logits_f32[r, :n]
        _, idx = torch.topk(row, min(top_k, n), largest=True, sorted=False)
        if page_table is not None:
            idx = _paged_slots(page_table[r], page_size, idx.to(torch.int64))
        indices[r, : len(idx)] = idx.to(torch.int32)
        if len(idx) < top_k:
            indices[r, len(idx) :] = -1
    return indices


def _top_k_varlen_check(
    reference_outputs,
    actual_outputs,
    logits=None,
    seq_lens=None,
    top_k=None,
    page_table=None,
    page_size=1,
    **_unused,
):
    """Tie-safe value check: every selected value must be >= the row's K-th largest.

    Exact set equality is not required because ties at the K-th boundary may be
    broken differently by the kernel vs the reference. Instead we verify that
    each selected index points to a value no smaller than the true K-th largest
    (matching the check used in tests/topk_varlen/test_topk_varlen.py::_check_correct).
    This requires the original logits and seq_lens, which the template passes
    via **_unused from the check call in the test.  Paged outputs (physical
    slots) are mapped back to logical columns through the inverse page table
    before the value check.
    """
    act = actual_outputs if not isinstance(actual_outputs, list) else actual_outputs[0]
    if page_table is not None and logits is not None:
        # inverse table per row: slot -> logical column (-1 where no column
        # maps); the table may be narrower than the logits width (rows past
        # its coverage are clamped by the kernels), so only covered columns map
        num_rows, N = logits.shape
        covered = min(N, page_table.shape[1] * page_size)
        cols = torch.arange(covered, device=logits.device, dtype=torch.int64)
        mapped = torch.full_like(act, -1)
        for r in range(num_rows):
            slots = _paged_slots(page_table[r], page_size, cols)
            inv = torch.full(
                (int(slots.max().item()) + 1,),
                -1,
                dtype=torch.int64,
                device=logits.device,
            )
            inv[slots] = cols
            sel = act[r].to(torch.int64)
            valid = sel >= 0
            mapped[r][valid] = inv[sel[valid]].to(act.dtype)
        act = mapped
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
    for r in range(act.shape[0]):
        n = int(seq_lens[r].item())
        if n < top_k:
            continue
        row = logits_f32[r, :n]
        kth = torch.topk(row, top_k).values[-1].item()
        sel = act[r].long()
        if (logits_f32[r][sel] < kth - 1e-5).any():
            return False
    return True


def _top_k_varlen_init(
    *,
    batch_size: int,
    max_pages: int = 0,
    page_size: int = 1,
    max_seq_len: int = 8192,
    top_k: int = 1024,
    device: str = "cuda",
    seed: int = 0,
):
    """Build inputs for ``flashinfer.top_k_varlen`` (no pre_idx).

    ``max_pages == 0``: unpaged call on the ``radix_cutlass`` (masked CUTLASS
    radix) backend, so the example runs on any GPU.  ``max_pages > 0``: fused
    paged output -- a random ``(batch_size, max_pages)`` page table at
    ``page_size`` columns per page (the recorded ``ps`` axis; ``1`` asks for
    the smallest power of two that covers ``max_seq_len``) and the
    ``walkfirst_primitives`` backend, which emits physical slots from the
    selection launch.  Logits are float32, the dtype the recorded examples
    carry.
    ``seq_lens`` is randomised in ``[top_k + 1, max_seq_len]`` to guarantee
    that every row has at least ``top_k`` valid entries.
    ``max_seq_len`` is padded to the next multiple of 8 (the 16-byte row
    alignment the fp16/bf16 kernels need).
    """
    max_seq_len = (max_seq_len + 7) // 8 * 8
    torch.manual_seed(seed)
    logits = torch.randn(batch_size, max_seq_len, dtype=torch.float32, device=device)
    seq_lens = torch.randint(
        top_k + 1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )
    inputs = {
        "logits": logits,
        "seq_lens": seq_lens,
        "top_k": top_k,
        "backend": "radix_cutlass",
    }
    if max_pages > 0:
        if page_size <= 1:
            page_size = 1
            while page_size * max_pages < max_seq_len:
                page_size *= 2
        if page_size & (page_size - 1) or page_size * max_pages < max_seq_len:
            raise ValueError(
                f"page_size ({page_size}) must be a power of two covering "
                f"max_seq_len with max_pages ({max_pages}) pages"
            )
        page_table = torch.randperm(batch_size * max_pages, device=device).to(
            torch.int32
        )
        inputs.update(
            page_table=page_table.view(batch_size, max_pages),
            page_size=page_size,
            backend="walkfirst_primitives",
        )
    return inputs


top_k_varlen_trace = TraceTemplate(
    op_type="topk",
    name_prefix="top_k_varlen",
    description=(
        "Decode-step top-K selection over batched logits with per-request seq_lens. "
        "GVR (Blackwell sm_100+), the radix / walk-first primitives backends "
        "(Ampere+), or the masked-radix fallback; with page_table the selected "
        "columns are emitted as physical KV slots (fused paged output)."
    ),
    axes={
        "batch_size": Var(description="Number of decode requests."),
        "max_seq_len": Const(abbrev="n", description="Logits row width (padded)."),
        "top_k": Const(abbrev="k", description="Number of top elements per row."),
        "max_pages": Var(
            description="Page-table width in pages per request (page_table only)."
        ),
        "page_size": Const(
            abbrev="ps",
            description=(
                "Columns per page-table entry for the fused paged output "
                "(1 when unpaged); names the paged definitions apart."
            ),
        ),
    },
    inputs={
        "logits": Tensor(
            ["batch_size", "max_seq_len"],
            description="Decode-step attention logits (bfloat16 / float16 / float32).",
        ),
        "seq_lens": Tensor(
            ["batch_size"],
            dtype="int32",
            description="Effective KV-cache length per request.",
        ),
        "top_k": Scalar("int32", description="K — number of top elements to select."),
        "pre_idx": Tensor(
            ["batch_size", "top_k"],
            dtype="int32",
            optional=True,
            description="Previous-step top-K indices (GVR warm-start hint).",
        ),
        "page_table": Tensor(
            ["batch_size", "max_pages"],
            dtype="int32",
            optional=True,
            description=(
                "Per-request KV page table; when given, indices are physical slots "
                "page_table[r, c // page_size] * page_size + c % page_size."
            ),
        ),
        "page_size": Scalar(
            "int32",
            description=(
                "KV page size (power of two) for the paged output; 1 (the API "
                "default) when unpaged.  Sources the ps axis."
            ),
        ),
    },
    outputs={
        "indices": Tensor(
            ["batch_size", "top_k"],
            dtype="int32",
            description="Selected top-K indices per row (physical slots with page_table).",
        ),
    },
    tags=["status:verified"],
    reference=_top_k_varlen_reference,
    check=_top_k_varlen_check,
    init=_top_k_varlen_init,
)
