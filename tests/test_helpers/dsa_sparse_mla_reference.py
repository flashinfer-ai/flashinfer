# Copyright (c) 2026 by FlashInfer team.
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

"""Pure-torch FP32 reference of DSA sparse MLA decode (GLM-5.3 / DeepSeek-V3.2)."""

from __future__ import annotations


import torch

HEAD_DIM = 576
V_DIM = 512


@torch.no_grad()
def sparse_mla_reference(
    query: torch.Tensor,
    kv_rows: torch.Tensor,
    indices: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float,
    top_k_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 reference of DSA sparse MLA over the KV rows.

    ``query`` ``[T, H, 576]`` (any float dtype; e4m3 is upcast), ``kv_rows`` fp32 ``[N, 576]``,
    ``indices`` ``[T, K]`` with ``-1`` = masked anywhere. Returns
    ``(out bf16 [T, H, 512], lse fp32 natural-log [T, H])``; fully masked rows give 0 / -inf.
    """
    T, H, _ = query.shape
    K = indices.shape[1]
    q = query.float()
    valid = indices >= 0
    if top_k_lens is not None:
        valid &= torch.arange(K, device=indices.device)[None, :] < top_k_lens[:, None]
    safe = torch.where(valid, indices, torch.zeros_like(indices))
    kv = kv_rows[safe.reshape(-1)].view(T, K, HEAD_DIM)
    s = torch.einsum("thd,tkd->thk", q, kv) * bmm1_scale  # natural-log domain logits
    s = s.masked_fill(~valid[:, None, :], float("-inf"))
    m = s.amax(dim=-1, keepdim=True)
    m_safe = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
    p = torch.exp(s - m_safe)
    l = p.sum(dim=-1, keepdim=True)
    o = (
        torch.einsum("thk,tkd->thd", p, kv[:, :, :V_DIM])
        / l.clamp(min=1e-30)
        * bmm2_scale
    )
    o = torch.where(l > 0, o, torch.zeros_like(o))
    lse = m_safe.squeeze(-1) + torch.log(l.squeeze(-1))
    lse = torch.where(l.squeeze(-1) > 0, lse, torch.full_like(lse, float("-inf")))
    return o.to(torch.bfloat16), lse
