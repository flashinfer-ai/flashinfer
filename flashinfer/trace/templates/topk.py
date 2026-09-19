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


@torch.no_grad()
def _top_k_varlen_reference(
    logits,
    seq_lens,
    top_k,
    pre_idx=None,
    row_starts=None,
    absolute_indices=False,
    **_unused,
):
    """Per-row top-K with seq_lens masking (window ``[row_starts[r],
    row_starts[r] + seq_lens[r])`` with window-local indices when
    ``row_starts`` is given, absolute columns with ``absolute_indices``).
    Reference uses torch.topk on each row."""
    num_rows, N = logits.shape
    indices = torch.empty(num_rows, top_k, dtype=torch.int32, device=logits.device)
    logits_f32 = logits.to(torch.float32)
    for r in range(num_rows):
        s = int(row_starts[r].item()) if row_starts is not None else 0
        n = max(min(s + int(seq_lens[r].item()), N) - s, 0)
        row = logits_f32[r, s : s + n]
        _, idx = torch.topk(row, min(top_k, n), largest=True, sorted=False)
        indices[r, : len(idx)] = idx.to(torch.int32) + (s if absolute_indices else 0)
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
    for r in range(act.shape[0]):
        s = int(row_starts[r].item()) if row_starts is not None else 0
        n = max(min(s + int(seq_lens[r].item()), N) - s, 0)
        if n < top_k:
            continue
        row = logits_f32[r, s : s + n]
        kth = torch.topk(row, top_k).values[-1].item()
        sel = act[r].long()  # window-local indices (absolute: shift back by s)
        if absolute_indices:
            sel = sel - s
        if (row[sel] < kth - 1e-5).any():
            return False
    return True


def _top_k_varlen_init(
    *,
    batch_size: int,
    max_seq_len: int = 8192,
    top_k: int = 1024,
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
        "Decode-step top-K selection over batched logits with per-request seq_lens. "
        "GVR (Blackwell sm_100+) or masked-radix fallback backend."
    ),
    axes={
        "batch_size": Var(description="Number of decode requests."),
        "max_seq_len": Const(abbrev="n", description="Logits row width (padded)."),
        "top_k": Const(abbrev="k", description="Number of top elements per row."),
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
        "row_starts": Tensor(
            ["batch_size"],
            dtype="int32",
            optional=True,
            description=(
                "Windowed (prefill) mode: row r ranks "
                "logits[r, row_starts[r] : row_starts[r] + seq_lens[r]] and "
                "returns window-local indices (gvr_2 only; absolute columns "
                "with absolute_indices=True)."
            ),
        ),
        "absolute_indices": Scalar(
            "bool",
            optional=True,
            description="Windowed mode: return absolute logits columns instead of window-local indices.",
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
