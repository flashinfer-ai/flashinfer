# SPDX-License-Identifier: Apache-2.0
"""One Triton launch that stages a W4A4 NVFP4 MegaMoE call from BF16 tokens.

From ``x [n, H]`` BF16 it writes, in one read of ``x``:

* the routed NVFP4 input: packed E2M1 ``[n, H/2]`` (element ``2j`` in the low
  nibble) and one E4M3 scale per 16 values, plain row-major ``[n, H/16]``, against
  the per-layer static global scale ``norm`` (= 1 / input_scale): scale
  ``e4m3(absmax * norm / 6)``, data ``x * norm / scale`` (round to nearest even,
  saturating);
* the shared-expert MXFP8 input: E4M3 ``[n, H]`` and one UE8M0 exponent per 32
  values (the power of two at or above ``absmax / 448``), written in the kernel's
  flat 32x4x4 swizzled scale layout;
* the routing: ids as int32 and weights as fp32 for the ``rows`` rows of the
  kernel variant, ``-1`` / ``0`` for rows ``>= n`` and for rows vLLM marks as
  padding. With EPLB, the logical ids are mapped to physical replicas and the
  expert load is recorded on the way, by vLLM's own ``eplb_map_and_record``
  exactly as vLLM's MegaMoE staging calls it (``prepare_megamoe._store_topk``).

``x`` is read as 32-bit words (two BF16 each), so every 16- and 32-value group is
a ``[2, 8]``-word tile held by one program, and FP4 pairs are packed with the
hardware ``cvt.rn.satfinite.e2m1x2.f32``.
"""

import torch
import triton
import triton.language as tl
from .submission_eplb import eplb_map_and_record

INERT_EPLB_ARGS = (None,) * 5 + (0,) * 3

E4M3_MAX_C = tl.constexpr(448.0)


@triton.jit
def _stage_kernel(
    xw_ptr, ids_ptr, w_ptr, pad_ptr,
    act_ptr, act_sf_ptr, sh_ptr, sh_sf_ptr, ids_out_ptr, w_out_ptr,
    norm, scale_mul, num_tokens, num_rows,
    stride_xw, stride_ids, stride_w,
    l2p_ptr, replica_ptr, load_ptr, record_ptr, unpadded_ptr, num_logical, map_stride, num_physical,
    H: tl.constexpr, TOPK: tl.constexpr, BLOCK_TOPK: tl.constexpr,
    BLOCK_M: tl.constexpr, GROUPS: tl.constexpr, HAS_PAD: tl.constexpr, HAS_EPLB: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row0 = pid_m * BLOCK_M
    if row0 < num_tokens:
        BN: tl.constexpr = BLOCK_M * GROUPS
        n = tl.arange(0, BN)
        rows = row0 + n // GROUPS
        gcol = pid_k * GROUPS + n % GROUPS  # 32-value group index within the row
        row_ok = rows < num_tokens
        h = tl.arange(0, 2)
        w = tl.arange(0, 8)
        word = gcol[:, None, None] * 16 + h[None, :, None] * 8 + w[None, None, :]
        mask = row_ok[:, None, None]
        words = tl.load(xw_ptr + rows[:, None, None] * stride_xw + word, mask=mask, other=0)
        xe = (words << 16).to(tl.float32, bitcast=True)
        xo = (words & -65536).to(tl.float32, bitcast=True)

        # shared expert: MXFP8, one UE8M0 per 32 values = per [2, 8]-word tile
        a2 = tl.maximum(tl.abs(xe), tl.abs(xo))
        amax16 = tl.max(a2, axis=2)
        amax32 = tl.max(amax16, axis=1)
        y = tl.maximum(amax32, 1e-30) / E4M3_MAX_C
        bits = y.to(tl.int32, bitcast=True)
        exp = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
        exp = tl.minimum(tl.maximum(exp, -127), 127)
        inv = ((127 - exp) << 23).to(tl.float32, bitcast=True)
        q8e = tl.clamp(xe * inv[:, None, None], -E4M3_MAX_C, E4M3_MAX_C).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        q8o = tl.clamp(xo * inv[:, None, None], -E4M3_MAX_C, E4M3_MAX_C).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        pair8 = q8e.to(tl.uint16) | (q8o.to(tl.uint16) << 8)
        tl.store(sh_ptr + rows[:, None, None] * (H // 2) + word, pair8, mask=mask)
        # flat 32x4x4 swizzle of the [rows, H/32] scale plane (H/128 atoms per atom row)
        sf_off = (((rows // 128) * (H // 128) + gcol // 4) * 32 + rows % 32) * 16 + ((rows % 128) // 32) * 4 + gcol % 4
        tl.store(sh_sf_ptr + sf_off, (exp + 127).to(tl.uint8), mask=row_ok)

        # routed experts: NVFP4, one E4M3 per 16 values = per [8]-word half tile
        sf = tl.minimum(amax16 * scale_mul, E4M3_MAX_C).to(tl.float8e4nv)
        sf_f = sf.to(tl.float32)
        rcp = tl.where(sf_f > 0, norm / sf_f, 0.0)
        ve = xe * rcp[:, :, None]
        vo = xo * rcp[:, :, None]
        packed = tl.inline_asm_elementwise(
            "{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, $2, $1;\n cvt.u32.u8 $0, t;\n}",
            "=r,f,f", [ve, vo], dtype=tl.int32, is_pure=True, pack=1,
        ).to(tl.uint8)
        tl.store(act_ptr + rows[:, None, None] * (H // 2) + word, packed, mask=mask)
        tl.store(act_sf_ptr + rows[:, None] * (H // 16) + gcol[:, None] * 2 + h[None, :], sf, mask=row_ok[:, None])

    # routing for the variant's rows (first column block only)
    if pid_k == 0:
        rr = row0 + tl.arange(0, BLOCK_M)
        live = rr < num_tokens
        kk = tl.arange(0, BLOCK_TOPK)
        kmask = kk[None, :] < TOPK
        lmask = live[:, None] & kmask
        ids = tl.load(ids_ptr + rr[:, None] * stride_ids + kk[None, :], mask=lmask, other=-1).to(tl.int32)
        wts = tl.load(w_ptr + rr[:, None] * stride_w + kk[None, :], mask=lmask, other=0.0).to(tl.float32)
        if HAS_PAD:
            pad = tl.load(pad_ptr + rr, mask=live, other=1) != 0
            ids = tl.where(pad[:, None], -1, ids)
            wts = tl.where(pad[:, None], 0.0, wts)
        if HAS_EPLB:
            # Logical -> physical replica (Knuth hash of the row index picks among the replicas) and one load count
            # per live, unpadded (row, k) with a valid id; padding and rows >= num_tokens are -1 and not counted.
            # The same call, token index and mask as vLLM's staging, so the staged ids and the recorded load are
            # bit-equal to it. This block runs once per row (first column block only), so each route counts once.
            ids = eplb_map_and_record(
                ids.to(tl.int64), rr[:, None], lmask, l2p_ptr, replica_ptr, num_logical, map_stride, load_ptr,
                num_physical, record_ptr, unpadded_ptr,
            ).to(tl.int32)
        omask = (rr < num_rows)[:, None] & kmask
        tl.store(ids_out_ptr + rr[:, None] * TOPK + kk[None, :], ids, mask=omask)
        tl.store(w_out_ptr + rr[:, None] * TOPK + kk[None, :], wts, mask=omask)


def _tile(num_tokens: int) -> tuple[int, int, int]:
    """(BLOCK_M, GROUPS, num_warps) measured fastest on Rubin per token range."""
    if 3600 < num_tokens < 5600:
        return 32, 8, 8
    return 16, 8, 4


def scale_multiplier(norm: float) -> float:
    """fp32(norm / 6), the block-scale multiplier of the NVFP4 quantization."""
    return float(torch.tensor(norm / 6.0, dtype=torch.float32))


def stage_nvfp4_mxfp8(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    is_padding: torch.Tensor | None,
    *,
    norm: float,
    scale_mul: float,
    rows: int,
    act: torch.Tensor,
    act_sf: torch.Tensor,
    shared_act: torch.Tensor,
    shared_act_sf: torch.Tensor,
    ids_out: torch.Tensor,
    weights_out: torch.Tensor,
    eplb: tuple | None = None,
) -> None:
    """Stage ``x`` and the routing for a kernel variant of ``rows`` token rows.

    ``act`` ``[>= n, H/2]`` uint8, ``act_sf`` ``[>= n, H/16]`` float8_e4m3fn,
    ``shared_act`` ``[>= n, H]`` float8_e4m3fn, ``shared_act_sf`` flat uint8 of at
    least ``round_up(n, 128) * H / 32``, ``ids_out`` / ``weights_out`` ``[>= rows,
    topk]`` int32 / float32, all contiguous.

    ``eplb`` is ``eplb.eplb_kernel_args(layer.eplb_state)``. Without it (or when it is
    inert) the kernel specializes to the plain staging (``HAS_EPLB`` False): the
    staged bytes are exactly those of a build without EPLB support.
    """
    num_tokens, hidden = x.shape
    topk = topk_ids.shape[1]
    if num_tokens > rows:
        raise ValueError(f"{num_tokens} tokens exceed the variant's {rows} rows")
    if hidden % 128 or x.stride(1) != 1:
        raise ValueError("x must be row-major with a hidden size that is a multiple of 128")
    xw = x.view(torch.int32)
    block_m, groups, warps = _tile(num_tokens)
    grid = (triton.cdiv(rows, block_m), hidden // (32 * groups))
    has_pad = is_padding is not None
    pad = is_padding.view(torch.uint8) if has_pad else ids_out
    if eplb is None:
        eplb = INERT_EPLB_ARGS
    has_eplb = eplb[0] is not None
    _stage_kernel[grid](
        xw, topk_ids, topk_weights, pad,
        act, act_sf, shared_act.view(torch.uint16), shared_act_sf, ids_out, weights_out,
        norm, scale_mul, num_tokens, rows,
        xw.stride(0), topk_ids.stride(0), topk_weights.stride(0),
        *eplb,
        H=hidden, TOPK=topk, BLOCK_TOPK=triton.next_power_of_2(topk),
        BLOCK_M=block_m, GROUPS=groups, HAS_PAD=has_pad, HAS_EPLB=has_eplb, num_warps=warps,
    )
