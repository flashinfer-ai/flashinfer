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

"""Pure-torch reference for the TRT-LLM NVFP4 MLA KV cache format (DSA / GLM-5.3 / DSV3.2).

Mirrors ``TensorRT-LLM/cpp/tensorrt_llm/kernels/mlaKernels.cu::quantizeAndWriteMlaFp4[Residual]``
and the dequantization in ``nvfp4MlaKvCacheGather.cu``:

* Row layout, R = residual_dim in {0, 64} (R applies to the LAST R dims = the RoPE part):
    data  pool: uint8 ``[N, (576 + R) // 2]``
    scale pool: e4m3 ``[N, (576 + R) // 16]``
  Dims ``[0, 576 - R)`` are plain NVFP4: 16-element groups, one e4m3 block scale each,
  two e2m1 per byte with the even element in the low nibble.
  Dims ``[576 - R, 576)`` (R = 64) are stored as main + residual: for every 16-element group
  ``[main 8 B][residual 8 B]`` in the data row and ``[mainScale][residualScale]`` in the scale row.
* Second-level scale: ``s_global = 448 * 6 / amax`` (TRT-LLM default amax = 100).
  ``x' = x * s_global``; ``blockScale = e4m3(max|x'| / 6)`` (floored at 2^-9); ``e2m1 = x' / blockScale``.
  Dequant: ``x = e2m1 * blockScale * s_dequant`` with ``s_dequant = amax / (448 * 6)``.

Differences from the CUDA writer (documented, acceptable for a reference): exact fp32 division
instead of ``reciprocal_approximate_ftz``; e2m1 round-to-nearest-even implemented in torch.
"""

from __future__ import annotations

import math
import torch

HEAD_DIM = 576
V_DIM = 512
GROUP = 16
NVFP4_MAX_BLOCK_SCALE = 448.0
NVFP4_MAX_VALUE = 6.0
DEFAULT_AMAX = 100.0
MIN_FP4_SCALE = 2.0**-9

_E2M1_VALUES = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def global_scale(amax: float = DEFAULT_AMAX) -> float:
    """``s_global`` applied before block quantization (TRT-LLM ``_nvfp4_mla_kv_scale_orig_quant``)."""
    return NVFP4_MAX_BLOCK_SCALE * NVFP4_MAX_VALUE / amax


def dequant_scale(amax: float = DEFAULT_AMAX) -> float:
    """``s_dequant`` folded into bmm1/bmm2 (TRT-LLM ``_nvfp4_mla_kv_scale_quant_orig``)."""
    return amax / (NVFP4_MAX_BLOCK_SCALE * NVFP4_MAX_VALUE)


def row_bytes(residual_dim: int) -> tuple[int, int]:
    """Return ``(data_bytes, scale_bytes)`` per token."""
    assert residual_dim in (0, 64)
    return (HEAD_DIM + residual_dim) // 2, (HEAD_DIM + residual_dim) // GROUP


def _to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round fp32 to the nearest E2M1 value (ties to even, saturate at 6) and return 4-bit codes."""
    values = _E2M1_VALUES.to(x.device)
    mag = x.abs().clamp(max=NVFP4_MAX_VALUE)
    # Midpoints between representable magnitudes; ties-to-even on the code index.
    mids = (values[:-1] + values[1:]) / 2  # 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5
    code = torch.bucketize(
        mag, mids, right=False
    )  # number of midpoints strictly below... see below
    # bucketize(right=False) counts mids < mag ... but ties (mag == mid) must round to even code.
    tie = (mag.unsqueeze(-1) == mids).any(-1)
    # On a tie, bucketize(right=False) returns the index of the first mid >= mag, i.e. the lower code.
    # Lower code is even -> keep; odd -> go up.
    code = torch.where(tie & (code % 2 == 1), code + 1, code)
    sign = (x < 0) & (code > 0)
    return (code | (sign.to(code.dtype) << 3)).to(torch.uint8)


def _from_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    values = _E2M1_VALUES.to(codes.device)
    mag = values[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def _pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """``[..., 2k]`` codes -> ``[..., k]`` bytes, even element in the low nibble."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0xF
    hi = packed >> 4
    return torch.stack((lo, hi), dim=-1).flatten(-2)


def _quant_groups(xg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``xg`` fp32 ``[..., G, 16]`` already multiplied by s_global.

    Returns ``(codes uint8 [..., G, 16], scale_e4m3 [..., G], dequant fp32 [..., G, 16])``.
    """
    amax = xg.abs().amax(dim=-1)
    scale = (amax / NVFP4_MAX_VALUE).clamp(min=MIN_FP4_SCALE).to(torch.float8_e4m3fn)
    scale_f = scale.float()
    inv = torch.where(scale_f == 0, torch.zeros_like(scale_f), 1.0 / scale_f)
    codes = _to_e2m1_codes(xg * inv.unsqueeze(-1))
    deq = _from_e2m1_codes(codes) * scale_f.unsqueeze(-1)
    return codes, scale, deq


@torch.no_grad()
def quantize_rows(
    x: torch.Tensor, residual_dim: int = 64, amax: float = DEFAULT_AMAX
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize fp32/bf16 rows ``[N, 576]`` into ``(data_pool uint8, scale_pool e4m3)``."""
    assert x.shape[-1] == HEAD_DIM
    n = x.shape[0]
    xs = x.float() * global_scale(amax)
    main_dim = HEAD_DIM - residual_dim
    g_main = main_dim // GROUP
    codes_main, scale_main, _ = _quant_groups(xs[:, :main_dim].view(n, g_main, GROUP))
    data = [_pack_nibbles(codes_main).view(n, -1)]
    scales = [scale_main]
    if residual_dim:
        g_res = residual_dim // GROUP
        xr = xs[:, main_dim:].view(n, g_res, GROUP)
        codes_m, scale_m, deq_m = _quant_groups(xr)
        codes_r, scale_r, _ = _quant_groups(xr - deq_m)
        # per group: [main 8 B][residual 8 B]; scales: [main][residual]
        dm = _pack_nibbles(codes_m)  # [n, g_res, 8]
        dr = _pack_nibbles(codes_r)
        data.append(torch.stack((dm, dr), dim=2).view(n, -1))
        scales.append(torch.stack((scale_m, scale_r), dim=2).view(n, -1))
    data_pool = torch.cat(data, dim=1).contiguous()
    scale_pool = torch.cat(scales, dim=1).contiguous()
    db, sb = row_bytes(residual_dim)
    assert data_pool.shape == (n, db) and scale_pool.shape == (n, sb)
    return data_pool, scale_pool


@torch.no_grad()
def dequantize_rows(
    data_pool: torch.Tensor,
    scale_pool: torch.Tensor,
    residual_dim: int = 64,
    amax: float = DEFAULT_AMAX,
    *,
    apply_global: bool = True,
) -> torch.Tensor:
    """Inverse of :func:`quantize_rows`; fp32 ``[N, 576]``.

    ``apply_global=False`` leaves out ``s_dequant`` (what the kernel feeds its e4m3 MMA, with the
    global factor folded into bmm scales, and what TRT-LLM's gather writes when dequantScale == 1).
    """
    n = data_pool.shape[0]
    main_dim = HEAD_DIM - residual_dim
    g_main = main_dim // GROUP
    main_bytes = main_dim // 2
    codes = _unpack_nibbles(data_pool[:, :main_bytes]).view(n, g_main, GROUP)
    out = [
        (_from_e2m1_codes(codes) * scale_pool[:, :g_main].float().unsqueeze(-1)).view(
            n, -1
        )
    ]
    if residual_dim:
        g_res = residual_dim // GROUP
        d = data_pool[:, main_bytes:].view(n, g_res, 2, 8)
        s = scale_pool[:, g_main:].view(n, g_res, 2).float()
        cm = _unpack_nibbles(d[:, :, 0])
        cr = _unpack_nibbles(d[:, :, 1])
        v = _from_e2m1_codes(cm) * s[:, :, 0:1] + _from_e2m1_codes(cr) * s[:, :, 1:2]
        out.append(v.view(n, -1))
    x = torch.cat(out, dim=1)
    if apply_global:
        x = x * dequant_scale(amax)
    return x


@torch.no_grad()
def quantize_q_nvfp4(q: torch.Tensor) -> torch.Tensor:
    """FP32 image of Q after the kernel's on-chip NVFP4 quantization.

    Per 16-element block: scale = e4m3(amax / 6), values rounded to the nearest
    E2M1 (ties to even), saturating at 6 (the native FP4 QK path).
    """
    shape = q.shape
    x = q.float().reshape(-1, 16)
    amax = x.abs().amax(dim=1, keepdim=True)
    sf = (amax / NVFP4_MAX_VALUE).to(torch.float8_e4m3fn).float()
    inv = torch.where(sf > 0, 1.0 / sf, torch.zeros_like(sf))
    y = (x * inv).clamp(-NVFP4_MAX_VALUE, NVFP4_MAX_VALUE)
    grid = _E2M1_VALUES.to(q.device)
    mag = y.abs()
    idx_hi = (
        torch.searchsorted(grid, mag.reshape(-1).contiguous())
        .clamp(max=7)
        .reshape(mag.shape)
    )
    idx_lo = (idx_hi - 1).clamp(min=0)
    d_lo = (mag - grid[idx_lo]).abs()
    d_hi = (grid[idx_hi] - mag).abs()
    pick_hi = (d_hi < d_lo) | ((d_hi == d_lo) & (idx_hi % 2 == 0))
    code = torch.where(pick_hi, idx_hi, idx_lo)
    return (grid[code] * torch.sign(y) * sf).reshape(shape)


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
    """fp32 reference of DSA sparse MLA over dequantized rows.

    ``query`` ``[T, H, 576]`` (any float dtype; e4m3 is upcast), ``kv_rows`` fp32 ``[N, 576]``
    (dequantized **without** the global scale, which the caller folds into the bmm scales exactly
    like the kernel), ``indices`` ``[T, K]`` with ``-1`` = masked anywhere. Returns
    ``(out bf16 [T, H, 512], lse fp32 base-2 [T, H])``; fully masked rows give 0 / -inf.
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
    lse = (m_safe.squeeze(-1) + torch.log(l.squeeze(-1))) / math.log(2.0)
    lse = torch.where(l.squeeze(-1) > 0, lse, torch.full_like(lse, float("-inf")))
    return o.to(torch.bfloat16), lse


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1024, HEAD_DIM, device=dev) * 2.0
    for r in (0, 64):
        d, s = quantize_rows(x, r)
        y = dequantize_rows(d, s, r)
        err = (y - x).abs().max().item()
        rel = ((y - x).norm() / x.norm()).item()
        print(
            f"R={r}: data {tuple(d.shape)} scale {tuple(s.shape)} max_abs_err {err:.4f} rel_l2 {rel:.4e}"
        )
