"""
Copyright (c) 2026 by FlashInfer team.

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

from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_experimental_api


@flashinfer_experimental_api
def dsa_indexer_topk(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    prefix_logits: torch.Tensor,
    cu_end: torch.Tensor,
    top_k: int = 2048,
    *,
    cand_cap: int = 49152,
    out: Optional[torch.Tensor] = None,
    status: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused DSA indexer top-k (LiteTopK) for SM100 GPUs (compute capability 10.0).

    Selects, per query, the ``top_k`` KV positions with the largest indexer logits
    ``sum_h weights[i, h] * relu(q[i, h] . kv[j]) * kv_scales[j]`` (the
    ``deep_gemm.fp8_mqa_logits`` definition) over ``j < cu_end[i]``, without
    materializing the ``[num_q, seq_kv]`` logits: an fp8 MQA scan keeps positions above
    a per-row threshold in a fixed-size candidate buffer, and an exact selector picks
    the top-k from it. Requires the ``deep_gemm`` package, whose SM100 headers are
    compiled into the kernel.

    The threshold is seeded from ``prefix_logits``, the logits of ``kv[:P]`` computed
    by the caller. The prefix positions are candidates too; the scan covers
    ``[P, cu_end[i])``. It is fastest when ``kv[:P]`` holds likely winners, such as the
    previous prefill chunk's selection. A caller that reorders ``kv`` must keep every
    query's visible set (hot positions below ``cu_end.min()`` first, the rest in
    order) and map the returned indices back.

    Parameters
    ----------
    q : torch.Tensor
        ``[num_q, 32, 128]`` float8_e4m3fn queries.
    kv : torch.Tensor
        ``[seq_kv, 128]`` float8_e4m3fn keys, ``seq_kv <= 2**20``.
    kv_scales : torch.Tensor
        float32 key scales with at least ``seq_kv`` rounded up to 4 elements.
    weights : torch.Tensor
        ``[num_q, 32]`` float32 head weights.
    prefix_logits : torch.Tensor
        ``[num_q, P]`` float32 with ``2048 <= P <= 8192`` or ``P == 12288``; rows may be
        padded to a stride divisible by 4. Requires ``P <= cu_end.min()`` (unchecked).
    cu_end : torch.Tensor
        ``[num_q]`` int32 exclusive end of each query's visible KV range.
    top_k : int
        Must be 2048.
    cand_cap : int
        Candidates per row, in ``[49152, 2**20]``; ``6 * num_q * cand_cap`` bytes.
    out, status : Optional[torch.Tensor]
        Preallocated ``[num_q, top_k]`` and ``[num_q]`` int32 outputs.

    Returns
    -------
    indices : torch.Tensor
        ``[num_q, top_k]`` int32 KV positions, unordered.
    status : torch.Tensor
        ``[num_q]`` int32; nonzero marks an invalid row, e.g. when more than
        ``cand_cap`` positions passed a threshold seeded from a poor prefix.
    """
    from .experimental.dsa_indexer import run

    return run(
        q, kv, kv_scales, weights, prefix_logits, cu_end, top_k, cand_cap, out, status
    )
