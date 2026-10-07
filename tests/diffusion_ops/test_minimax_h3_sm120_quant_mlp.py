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
"""SM120 (GB202) FP8 / NVFP4 fused MiniMax-H3 MLP (RMSNorm + AdaLN + FC1 + SwiGLU + FC2 + gated
residual) against an exact torch emulation.

The operator quantizes twice (the modulated activation before FC1, ``y`` before FC2).  Both are
threshold operations on values the kernel and torch accumulate in different orders, so each quantized
intermediate is checked against its *definition* (scale and dequantized value within derived bounds of
the reference value, bounded violation budgets) and the next stage of the reference consumes the
kernel's own quantized operand:

* stage 1 (``a``): the K4 rules -- FP8 per-token scale within ``2^-7`` of ``RN(amax / 448)`` and values
  within half an E4M3 step plus the BF16 round points; NVFP4 codes and scales against the
  ``nvfp4_quantize`` recipe emulation up to bounded non-tie mismatches;
* stage 2 (``y``): FP8 -- the kernel's BF16 ``y`` against the reference ``y`` on its ``a_q`` (K4 rule),
  the per-token scale exactly ``RN(max(amax |y|, 1e-12) / 448)`` of the kernel's own ``y`` and the codes
  the nearest E4M3 of ``y / scale``; NVFP4 (``y`` is never materialized) -- the block scales within one
  UE4M3 code of the recipe applied to the reference ``y``, the dequantized values within half an E2M1
  step plus one BF16 ulp plus ``atol + rtol |y|`` (``max(4, 2e-7 numel)`` violations allowed);
* output: ``out = BF16(residual + BF16(gate[i] * BF16(FC2(y_q))))`` on the kernel's ``y_q`` within
  ``atol + rtol max(|ref|, |p|)`` (1e-2 / 1.6e-2) plus one BF16 ulp of ``o`` scaled by the gate and the
  round points of ``p`` and ``out`` (the FC2 accumulation order flips ``o`` by an ulp on a ~1e-6 fraction
  of the elements, and ``|o|`` may exceed ``|out|``), with ``max(4, 2e-7 numel)`` violations; rows whose
  AdaLN index is out of range reproduce the residual exactly.
"""

import math
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import (
    minimax_h3_mlp_fp8_sm120,
    minimax_h3_mlp_nvfp4_sm120,
    prepare_minimax_h3_fc1_weight_fp8,
    prepare_minimax_h3_fc2_weight_fp8,
    prepare_minimax_h3_fc2_weight_nvfp4_sm120,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu import (
    E2M1_MAX,
    E4M3_MAX,
    MINIMAX_H3_ADALN_ROWS,
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_HIDDEN,
    MINIMAX_H3_SF_BLOCK,
    NVFP4_PACKED_COLS,
    NVFP4_SF_COLS,
    deinterleave_minimax_h3_fc1_rows_sm120,
    fp8_scale_from_amax,
    prepare_minimax_h3_fc1_weight_nvfp4_sm120,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_mlp import (
    FFN_PACKED_COLS,
    FFN_SF_COLS,
    FP8_MMA_FORM_LEGACY,
    FP8_MMA_FORM_MXF8F6F4,
    MINIMAX_H3_FC2_BLOCK_M,
    minimax_h3_mlp_fc2_flags_sm120,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    _unswizzle_sf_128x4,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
)
from flashinfer.utils import get_compute_capability

ATOL = 1e-2
RTOL = 1.6e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4
MAX_E2M1_MISMATCH_FRACTION = 4e-6
MAX_E2M1_MISMATCH_FLOOR = 4
MAX_NV_SCALE_MISMATCH_FRACTION = 8e-6
MAX_SCALE_MISMATCH_FLOOR = 2
TIE_REL_TOL = 2.0**-18
# Tails around the 128-row FC1 tile and the 256-row FC2 tile, plus a multi-tile shape.
ROWS = [1, 127, 128, 129, 255, 257, 4097]
REFERENCE_CHUNK_ROWS = 1024
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _minor = get_compute_capability(torch.device("cuda:0"))
    return major == 12


requires_sm120 = pytest.mark.skipif(
    not _supported(), reason="requires a CUDA GPU with compute capability 12.x (GB202)"
)


# --------------------------------------------------------------------------------------------
# Deterministic synthetic model + inputs
# --------------------------------------------------------------------------------------------


def make_model(device: torch.device, seed: int = 4618) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)

    def uniform(shape, lo, hi):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).uniform_(
            lo, hi, generator=g
        )

    def normal(shape, std):
        return torch.empty(shape, dtype=torch.bfloat16, device=device).normal_(
            0.0, std, generator=g
        )

    return {
        "x_norm_weight": uniform((MINIMAX_H3_HIDDEN,), 0.9, 1.1),
        "adaln_scale": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "gate": uniform((MINIMAX_H3_ADALN_ROWS, MINIMAX_H3_HIDDEN), -1.0, 1.0),
        "fc1_weight": normal((MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), 0.02),
        "fc2_weight": normal((MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN), 0.02),
    }


def make_inputs(rows: int, device: torch.device, seed: int = 4618):
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * rows)
    x = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.5, generator=g)
    residual = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 1.0, generator=g)
    # "production segments": nine contiguous AdaLN segments over the rows (int64 index).
    r = torch.arange(rows, dtype=torch.int64, device=device)
    idx = torch.div(r * MINIMAX_H3_ADALN_ROWS, rows, rounding_mode="floor").clamp_max(
        MINIMAX_H3_ADALN_ROWS - 1
    )
    return x, residual, idx


# --------------------------------------------------------------------------------------------
# Reference math
# --------------------------------------------------------------------------------------------


def _no_tf32():
    class _Guard:
        def __enter__(self):
            self.prev = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False

        def __exit__(self, *exc):
            torch.backends.cuda.matmul.allow_tf32 = self.prev

    return _Guard()


def reference_modulated(x, model, adaln_index, eps=MINIMAX_H3_EPS):
    norm = F.rms_norm(x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=eps).to(
        torch.bfloat16
    )
    idx = adaln_index.long()
    rows = model["adaln_scale"].shape[0]
    valid = (idx >= 0) & (idx < rows)
    safe = idx.clamp(0, rows - 1)
    a = torch.addcmul(
        model["adaln_shift"].index_select(0, safe),
        norm,
        (model["adaln_scale"].index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def gate_rows(model, adaln_index):
    idx = adaln_index.long()
    rows = model["gate"].shape[0]
    valid = (idx >= 0) & (idx < rows)
    g = model["gate"].index_select(0, idx.clamp(0, rows - 1))
    return torch.where(valid[:, None], g, torch.zeros_like(g))


def swiglu_from_scaled(a, w_t, a_scale=None, w_scale=None, alpha=None):
    """``y = BF16(BF16(silu(h[:, :FFN])) * h[:, FFN:])`` with ``h = BF16(((a @ w^T) * a_scale) *
    w_scale)`` (FP8) or ``h = BF16(alpha * (a @ w^T))`` (NVFP4); FP32 GEMM, row chunks."""
    with _no_tf32():
        out = torch.empty(
            (a.shape[0], MINIMAX_H3_FFN), dtype=torch.bfloat16, device=a.device
        )
        for r0 in range(0, a.shape[0], REFERENCE_CHUNK_ROWS):
            r1 = min(a.shape[0], r0 + REFERENCE_CHUNK_ROWS)
            h = a[r0:r1].float() @ w_t
            if a_scale is not None:
                h = h * a_scale[r0:r1, None]
            if w_scale is not None:
                h = h * w_scale[None, :]
            if alpha is not None:
                h = h * float(alpha)
            h = h.to(torch.bfloat16)
            gate, up = h.chunk(2, dim=-1)
            out[r0:r1] = (F.silu(gate) * up).to(torch.bfloat16)
        return out


def fc2_gated_residual(y_scaled_fn, w2_t, model, idx, residual, y_scale=None, w_scale=None, alpha=None):
    """``o = BF16(FC2)``, ``p = BF16(gate[i] * o)``, ``out = BF16(residual + p)`` in row chunks.
    ``y_scaled_fn(r0, r1)`` yields the FP32 FC2 A operand rows (codes, or ``q * sf`` for NVFP4)."""
    rows = residual.shape[0]
    g = gate_rows(model, idx)
    with _no_tf32():
        out = torch.empty_like(residual)
        o_all = torch.empty_like(residual)
        p_all = torch.empty_like(residual)
        for r0 in range(0, rows, REFERENCE_CHUNK_ROWS):
            r1 = min(rows, r0 + REFERENCE_CHUNK_ROWS)
            o = y_scaled_fn(r0, r1) @ w2_t
            if y_scale is not None:
                o = o * y_scale[r0:r1, None]
            if w_scale is not None:
                o = o * w_scale[None, :]
            if alpha is not None:
                o = o * float(alpha)
            o = o.to(torch.bfloat16)
            p = (g[r0:r1] * o).to(torch.bfloat16)
            out[r0:r1] = (residual[r0:r1] + p).to(torch.bfloat16)
            o_all[r0:r1] = o
            p_all[r0:r1] = p
        return {"out": out, "o": o_all, "p": p_all, "gate": g}


def violation_stats(y: torch.Tensor, ref: torch.Tensor) -> Dict[str, float]:
    diff = (y.float() - ref.float()).abs()
    bad = diff > (ATOL + RTOL * ref.float().abs())
    numel = diff.numel()
    return {
        "numel": numel,
        "violations": int(bad.sum().item()),
        "budget": max(
            MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * numel))
        ),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
    }


def assert_within_budget(y, ref, what: str) -> Dict[str, float]:
    assert y.shape == ref.shape and y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all(), f"{what}: non-finite output"
    stats = violation_stats(y, ref)
    assert stats["violations"] <= stats["budget"], f"{what}: {stats}"
    return stats


def assert_output_within_budget(out, ref: Dict[str, torch.Tensor], what: str) -> Dict[str, float]:
    """``out`` against the reference math on the kernel's ``y_q``: within ``atol + rtol * max(|ref|, |p|)``
    plus one BF16 ulp of ``o`` scaled by the gate (the FC2 accumulation order flips ``o`` by an ulp on a
    ~1e-6 fraction of the elements; ``|o|`` can exceed ``|out|``) plus the round points of ``p`` and
    ``out``, with ``max(4, 2e-7 numel)`` violations allowed."""
    assert out.shape == ref["out"].shape and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all(), f"{what}: non-finite output"
    r = ref["out"].float()
    p = ref["p"].float()
    diff = (out.float() - r).abs()
    bound = (
        ATOL
        + RTOL * torch.maximum(r.abs(), p.abs())
        + ref["gate"].float().abs() * bf16_ulp(ref["o"])
        + bf16_ulp(p)
        + bf16_ulp(r)
    )
    bad = diff > bound
    numel = diff.numel()
    stats = {
        "numel": numel,
        "violations": int(bad.sum().item()),
        "budget": max(MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * numel))),
        "max_abs_err": float(diff.max().item()),
        "mean_abs_err": float(diff.mean().item()),
        "strict_violations": int((diff > (ATOL + RTOL * r.abs())).sum().item()),
    }
    assert stats["violations"] <= stats["budget"], f"{what}: {stats}"
    return stats


def bf16_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    mag = magnitude.float().abs().clamp_min(2.0**-126)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7.0)


def e4m3_step(magnitude: torch.Tensor) -> torch.Tensor:
    mag = magnitude.float().abs().clamp_min(2.0**-6)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 3.0).clamp_min(2.0**-9)


# ---- FP8 -----------------------------------------------------------------------------------


def quantize_fp8_rows(t: torch.Tensor, chunk_rows: int = 2048):
    """Per-row E4M3: ``scale = RN(max(absmax, 1e-12) / 448)``, ``q = RN_sat(t / scale)``."""
    q = torch.empty(t.shape, dtype=torch.float8_e4m3fn, device=t.device)
    scale = torch.empty((t.shape[0],), dtype=torch.float32, device=t.device)
    for r0 in range(0, t.shape[0], chunk_rows):
        r1 = min(t.shape[0], r0 + chunk_rows)
        rows = t[r0:r1].float()
        s = fp8_scale_from_amax(rows.abs().amax(dim=1).clamp_min(1e-12))
        scale[r0:r1] = s
        q[r0:r1] = (
            (rows / s[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        )
    return q, scale


def assert_fp8_stage1(a_q, a_scale, x, model, idx, a_ref) -> None:
    """K4 derived bound of the kernel's per-token FP8 activation against the reference activation."""
    _q_ref, scale_ref = quantize_fp8_rows(a_ref)
    scale_c = a_scale.float()
    scale_rel = ((scale_c / scale_ref) - 1.0).abs()
    assert bool((scale_rel <= 2.0**-7).all()), (
        f"fp8 per-token scale off: max rel err {float(scale_rel.max()):.4g}"
    )
    n_ref = F.rms_norm(
        x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=MINIMAX_H3_EPS
    ).to(torch.bfloat16)
    safe = idx.long().clamp(0, model["adaln_scale"].shape[0] - 1)
    sp1 = (
        (model["adaln_scale"].index_select(0, safe).float() + 1.0)
        .to(torch.bfloat16)
        .float()
        .abs()
    )
    ref32 = a_ref.float()
    deq = a_q.float() * scale_c[:, None]
    bound = (
        0.5 * e4m3_step(a_q.float()) * scale_c[:, None]
        + bf16_ulp(n_ref) * sp1
        + bf16_ulp(torch.maximum(ref32.abs(), deq.abs()))
        + 2.0**-20 * ref32.abs()
    )
    err = (deq - ref32).abs()
    assert torch.isfinite(deq).all(), "non-finite dequantized FP8 activation"
    bad = ~(err <= bound)
    assert not bool(bad.any()), (
        f"fp8 stage-1 activation beyond the derived bound: {int(bad.sum())}/{bad.numel()} "
        f"elements, max err {float(err.max()):.4g}"
    )


def assert_fp8_stage2(y_q, y_scale, y_kernel, y_ref) -> Dict[str, float]:
    """The FP8 operator's ``y`` path, checked against definitions:

    * the kernel's BF16 ``y`` (``workspace_y``, the FC1 + SwiGLU output) against the reference ``y`` on
      the kernel's own ``a_q`` -- the K4 rule (``atol + rtol |ref|``, ``max(4, 2e-7 numel)`` violations);
    * the per-token scale is exactly ``RN(max(amax |y|, 1e-12) / 448)`` of the kernel's own ``y`` (a
      threshold quantity: computing it from the reference ``y`` instead would move it by a BF16 ulp
      whenever the row maximum differs by one);
    * the codes dequantize to within half an E4M3 step of the kernel's own ``y`` (nearest rounding;
      the shared violation budget covers rounding ties of the in-kernel reciprocal) and to within half
      a step plus one BF16 ulp plus ``atol + rtol |y|`` of the reference ``y``."""
    stats_y = assert_within_budget(y_kernel, y_ref, "fp8 y (FC1 + SwiGLU output)")
    _q_own, scale_own = quantize_fp8_rows(y_kernel)
    assert torch.equal(y_scale, scale_own), (
        "fp8 y per-token scale is not RN(max(amax, 1e-12) / 448) of the kernel's y: "
        f"max rel err {float(((y_scale.float() / scale_own) - 1.0).abs().max()):.4g}"
    )
    scale_c = y_scale.float()
    own_violations = 0
    ref_violations = 0
    max_err_own = 0.0
    max_err_ref = 0.0
    for r0 in range(0, y_ref.shape[0], REFERENCE_CHUNK_ROWS):
        r1 = min(y_ref.shape[0], r0 + REFERENCE_CHUNK_ROWS)
        own32 = y_kernel[r0:r1].float()
        ref32 = y_ref[r0:r1].float()
        q32 = y_q[r0:r1].float()
        deq = q32 * scale_c[r0:r1, None]
        assert torch.isfinite(deq).all(), "non-finite dequantized FP8 y"
        half = 0.5 * e4m3_step(q32) * scale_c[r0:r1, None]
        err_own = (deq - own32).abs()
        own_violations += int((err_own > half + 2.0**-20 * own32.abs()).sum().item())
        max_err_own = max(max_err_own, float(err_own.max().item()))
        err_ref = (deq - ref32).abs()
        ref_violations += int((err_ref > half + bf16_ulp(ref32) + ATOL + RTOL * ref32.abs()).sum().item())
        max_err_ref = max(max_err_ref, float(err_ref.max().item()))
    budget = max(MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * y_ref.numel())))
    assert own_violations <= budget, (
        f"fp8 y codes are not the nearest E4M3 of the kernel's y / scale: {own_violations} violations "
        f"(budget {budget}), max err {max_err_own:.4g}"
    )
    assert ref_violations <= budget, (
        f"fp8 stage-2 y beyond its bound against the reference y: {ref_violations} violations "
        f"(budget {budget}), max err {max_err_ref:.4g}"
    )
    return {
        "y_violations": stats_y["violations"],
        "own_violations": own_violations,
        "ref_violations": ref_violations,
        "budget": budget,
    }


# ---- NVFP4 (FlashInfer nvfp4_quantize recipe emulation) -------------------------------------


def nvfp4_scale_prerounded(absmax: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    rcp6 = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=absmax.device)
    return g.float().reshape(()) * (absmax.float() * rcp6)


def nvfp4_scale_values(absmax, g):
    sf = torch.clamp(nvfp4_scale_prerounded(absmax, g), max=E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    return sf.view(torch.uint8), sf.float()


def nvfp4_output_scale(sf_f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    inv_g = torch.ones((), dtype=torch.float32, device=sf_f.device) / g.float().reshape(
        ()
    )
    return torch.where(sf_f != 0, 1.0 / (sf_f * inv_g), torch.zeros_like(sf_f))


def _near_midpoint(v: torch.Tensor, grid_values, rel_tol: float = TIE_REL_TOL):
    grid = torch.tensor(grid_values, dtype=torch.float32, device=v.device)
    mids = (grid[1:] + grid[:-1]) * 0.5
    mag = v.abs()
    idx = torch.clamp(torch.bucketize(mag, mids), max=mids.numel() - 1)
    lo = mids[torch.clamp(idx - 1, min=0)]
    hi = mids[idx]
    return ((mag - lo).abs() <= rel_tol * lo.clamp(min=1e-30)) | (
        (mag - hi).abs() <= rel_tol * hi.clamp(min=1e-30)
    )


def _e4m3_positive_grid(device):
    codes = torch.arange(1, 0x7F, dtype=torch.uint8, device=device)
    return torch.sort(codes.view(torch.float8_e4m3fn).float()).values.tolist()


def e2m1_encode(v: torch.Tensor) -> torch.Tensor:
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=v.device)
    mids = (grid[1:] + grid[:-1]) * 0.5
    mag = torch.clamp(v.abs(), max=E2M1_MAX)
    code = torch.bucketize(mag, mids, right=False)
    tie = (code < 7) & (mag == mids[torch.clamp(code, max=6)]) & (code % 2 == 1)
    code = torch.where(tie, code + 1, code)
    return (code + torch.where(torch.signbit(v), 8, 0)).to(torch.uint8)


def e2m1_pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()


def nvfp4_quantize_emulated(t: torch.Tensor, g: torch.Tensor):
    """``(packed E2M1 uint8 [R, K/2], UE4M3 uint8 [R, K/16], code_tie [R, K], scale_tie [R, K/16])``."""
    rows, cols = t.shape
    x = t.float()
    blocks = x.abs().reshape(rows, cols // MINIMAX_H3_SF_BLOCK, MINIMAX_H3_SF_BLOCK)
    absmax = blocks.amax(dim=-1)
    sf_pre = nvfp4_scale_prerounded(absmax, g)
    sf_bytes, sf_f = nvfp4_scale_values(absmax, g)
    out_scale = nvfp4_output_scale(sf_f, g)
    scaled = (
        x.reshape(rows, cols // MINIMAX_H3_SF_BLOCK, MINIMAX_H3_SF_BLOCK)
        * out_scale[..., None]
    ).reshape(rows, cols)
    scale_tie = _near_midpoint(sf_pre, _e4m3_positive_grid(t.device)) & (
        sf_pre < E4M3_MAX
    )
    code_tie = _near_midpoint(scaled, _E2M1_VALUES) & (scaled.abs() < E2M1_MAX)
    code_tie = code_tie | scale_tie[..., None].expand(
        rows, cols // MINIMAX_H3_SF_BLOCK, MINIMAX_H3_SF_BLOCK
    ).reshape(rows, cols)
    return e2m1_pack(e2m1_encode(scaled)), sf_bytes.contiguous(), code_tie, scale_tie


def e2m1_unpack(packed: torch.Tensor):
    """Packed E2M1 bytes -> (signed FP32 code values ``[R, K]``, code magnitudes)."""
    rows = packed.shape[0]
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.empty(
        (rows, 2 * packed.shape[1]), dtype=torch.uint8, device=packed.device
    )
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    mag = grid[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag), mag


def nvfp4_dequantize_scaled(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 ``[R, K/2]`` + UE4M3 bytes ``[R, K/16]`` -> FP32 ``q * sf`` (the MMA operand)."""
    rows = packed.shape[0]
    vals, _mag = e2m1_unpack(packed)
    sf_f = sf.view(torch.float8_e4m3fn).float()
    return (vals.reshape(rows, -1, MINIMAX_H3_SF_BLOCK) * sf_f[..., None]).reshape(
        rows, -1
    )


def assert_nvfp4_stage1(a_q, a_sf, a_ref, g_a) -> Dict[str, int]:
    q_ref, sf_ref, code_tie, scale_tie = nvfp4_quantize_emulated(a_ref, g_a)
    byte_tie = code_tie.reshape(code_tie.shape[0], -1, 2).any(-1)
    q_mismatch = int(((a_q != q_ref) & ~byte_tie).sum().item())
    sf_mismatch = int(((a_sf != sf_ref) & ~scale_tie).sum().item())
    q_budget = max(
        MAX_E2M1_MISMATCH_FLOOR,
        int(math.ceil(MAX_E2M1_MISMATCH_FRACTION * a_q.numel())),
    )
    sf_budget = max(
        MAX_SCALE_MISMATCH_FLOOR,
        int(math.ceil(MAX_NV_SCALE_MISMATCH_FRACTION * a_sf.numel())),
    )
    assert q_mismatch <= q_budget and sf_mismatch <= sf_budget, (
        f"nvfp4 stage-1 activation differs from the nvfp4_quantize emulation: "
        f"{q_mismatch} packed bytes (budget {q_budget}), {sf_mismatch} scale bytes (budget {sf_budget})"
    )
    return {"q_mismatch": q_mismatch, "sf_mismatch": sf_mismatch}


def assert_nvfp4_stage2(y_q, y_sf, y_ref, g_y) -> Dict[str, float]:
    """The kernel's block-16 NVFP4 ``y`` against the recipe applied to the reference ``y`` (on the
    kernel's ``a_q``): every block scale equal or one UE4M3 code apart (the block amax may sit on a
    rounding edge), the dequantized values ``code * sf / g`` within half an E2M1 step plus one BF16 ulp
    plus ``atol + rtol |y|`` with the shared violation budget."""
    rows = y_ref.shape[0]
    scale_violations = 0
    value_violations = 0
    max_err = 0.0
    g = g_y.float().reshape(())
    for r0 in range(0, rows, REFERENCE_CHUNK_ROWS):
        r1 = min(rows, r0 + REFERENCE_CHUNK_ROWS)
        ref32 = y_ref[r0:r1].float()
        absmax = ref32.abs().reshape(r1 - r0, -1, MINIMAX_H3_SF_BLOCK).amax(dim=-1)
        _sf_ref_bytes, sf_ref = nvfp4_scale_values(absmax, g_y)
        sf_k = y_sf[r0:r1].view(torch.float8_e4m3fn).float()
        # one UE4M3 code apart: |sf_k - sf_ref| <= e4m3 step at max(sf_k, sf_ref) (zero scales must agree)
        step = e4m3_step(torch.maximum(sf_k, sf_ref))
        scale_bad = (sf_k - sf_ref).abs() > step
        scale_bad |= (sf_k == 0) != (sf_ref == 0)
        scale_violations += int(scale_bad.sum().item())
        vals, mag = e2m1_unpack(y_q[r0:r1])
        half_step = torch.where(mag < 2.0, 0.25, torch.where(mag < 4.0, 0.5, 1.0))
        unit = (sf_k / g)[..., None].expand(r1 - r0, -1, MINIMAX_H3_SF_BLOCK).reshape(r1 - r0, -1)
        deq = vals * unit
        bound = half_step * unit + bf16_ulp(ref32) + ATOL + RTOL * ref32.abs()
        err = (deq - ref32).abs()
        value_violations += int((err > bound).sum().item())
        max_err = max(max_err, float(err.max().item()))
    budget = max(MAX_VIOLATIONS_FLOOR, int(math.ceil(MAX_VIOLATION_FRACTION * y_ref.numel())))
    assert scale_violations <= MAX_SCALE_MISMATCH_FLOOR, (
        f"nvfp4 stage-2 y block scales off by more than one UE4M3 code: {scale_violations}"
    )
    assert value_violations <= budget, (
        f"nvfp4 stage-2 y beyond its bound: {value_violations} violations (budget {budget}), "
        f"max err {max_err:.4g}"
    )
    return {"scale_violations": scale_violations, "violations": value_violations, "max_err": max_err}


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def model(device):
    return make_model(device)


@pytest.fixture(scope="module")
def prepared_fp8(model):
    """SM120 FP8 weights of both GEMMs plus the linear-order FC1 quantization for the reference."""
    w1_q, w1_scale = prepare_minimax_h3_fc1_weight_fp8(model["fc1_weight"])
    lin_q, lin_scale = quantize_fp8_rows(model["fc1_weight"])
    w2_q, w2_scale = prepare_minimax_h3_fc2_weight_fp8(model["fc2_weight"])
    return {
        "w1_q": w1_q,
        "w1_scale": w1_scale,
        "w1_lin_t": lin_q.float().t().contiguous(),
        "w1_lin_scale": lin_scale,
        "w2_q": w2_q,
        "w2_scale": w2_scale,
        "w2_t": w2_q.float().t().contiguous(),
    }


@pytest.fixture(scope="module")
def prepared_nvfp4(model):
    """SM120 NVFP4 weights of both GEMMs plus ``w_q * w_sf`` (FP32, linear order) for the reference."""
    g_w1 = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    w1_q, w1_sf = prepare_minimax_h3_fc1_weight_nvfp4_sm120(model["fc1_weight"], g_w1)
    w1_scaled = deinterleave_minimax_h3_fc1_rows_sm120(
        nvfp4_dequantize_scaled(
            w1_q, _unswizzle_sf_128x4(w1_sf, MINIMAX_H3_FC1_ROWS, NVFP4_SF_COLS)
        )
    )
    g_w2 = minimax_h3_nvfp4_global_scale(model["fc2_weight"])
    w2_q, w2_sf = prepare_minimax_h3_fc2_weight_nvfp4_sm120(model["fc2_weight"], g_w2)
    w2_scaled = nvfp4_dequantize_scaled(
        w2_q, _unswizzle_sf_128x4(w2_sf, MINIMAX_H3_HIDDEN, FFN_SF_COLS)
    )
    return {
        "g_w1": g_w1,
        "w1_q": w1_q,
        "w1_sf": w1_sf,
        "w1_scaled_t": w1_scaled.t().contiguous(),
        "g_w2": g_w2,
        "w2_q": w2_q,
        "w2_sf": w2_sf,
        "w2_scaled_t": w2_scaled.t().contiguous(),
    }


# --------------------------------------------------------------------------------------------
# Weight preparation
# --------------------------------------------------------------------------------------------


@requires_sm120
def test_prepare_fc2_weight_fp8_is_row_quantization(model, prepared_fp8):
    lin_q, lin_scale = quantize_fp8_rows(model["fc2_weight"])
    assert prepared_fp8["w2_q"].dtype == torch.float8_e4m3fn
    assert tuple(prepared_fp8["w2_q"].shape) == (MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN)
    assert torch.equal(prepared_fp8["w2_q"].view(torch.uint8), lin_q.view(torch.uint8))
    assert torch.equal(prepared_fp8["w2_scale"], lin_scale)


@requires_sm120
def test_prepare_fc2_weight_nvfp4_matches_nvfp4_quantize(model, prepared_nvfp4):
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    q, sf = nvfp4_quantize(
        model["fc2_weight"],
        prepared_nvfp4["g_w2"],
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    assert tuple(prepared_nvfp4["w2_q"].shape) == (MINIMAX_H3_HIDDEN, FFN_PACKED_COLS)
    assert prepared_nvfp4["w2_sf"].numel() == MINIMAX_H3_HIDDEN * FFN_SF_COLS
    assert torch.equal(
        prepared_nvfp4["w2_q"],
        q.view(torch.uint8).reshape(MINIMAX_H3_HIDDEN, FFN_PACKED_COLS),
    )
    assert torch.equal(prepared_nvfp4["w2_sf"], sf.view(torch.uint8).reshape(-1))
    # The swizzle round-trips: the linear scale rows match the recipe emulation of the weight.
    _q_ref, sf_ref, _ct, scale_tie = nvfp4_quantize_emulated(
        model["fc2_weight"], prepared_nvfp4["g_w2"]
    )
    sf_lin = _unswizzle_sf_128x4(prepared_nvfp4["w2_sf"], MINIMAX_H3_HIDDEN, FFN_SF_COLS)
    assert int(((sf_lin != sf_ref) & ~scale_tie).sum()) <= MAX_SCALE_MISMATCH_FLOOR


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


def _workspaces_fp8(rows, device):
    return {
        "workspace_a_q": torch.empty((rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device),
        "workspace_a_scale": torch.empty((rows,), dtype=torch.float32, device=device),
        "workspace_y": torch.empty((rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device),
        "workspace_y_q": torch.empty((rows, MINIMAX_H3_FFN), dtype=torch.float8_e4m3fn, device=device),
        "workspace_y_scale": torch.empty((rows,), dtype=torch.float32, device=device),
        "workspace_flags": torch.empty((minimax_h3_mlp_fc2_flags_sm120(rows),), dtype=torch.int32, device=device),
    }


def _workspaces_nvfp4(rows, device):
    return {
        "workspace_a_q": torch.empty((rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device),
        "workspace_a_sf": torch.empty((rows, NVFP4_SF_COLS), dtype=torch.uint8, device=device),
        "workspace_y_q": torch.empty((rows, FFN_PACKED_COLS), dtype=torch.uint8, device=device),
        "workspace_y_sf": torch.empty((rows, FFN_SF_COLS), dtype=torch.uint8, device=device),
        "workspace_flags": torch.empty((minimax_h3_mlp_fc2_flags_sm120(rows),), dtype=torch.int32, device=device),
    }


def run_fp8_case(rows, model, prepared, device, idx=None, fp8_mma_form=-1, out=None):
    x, residual, default_idx = make_inputs(rows, device)
    idx = default_idx if idx is None else idx
    ws = _workspaces_fp8(rows, device)
    returned = minimax_h3_mlp_fp8_sm120(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["gate"],
        residual,
        prepared["w1_q"],
        prepared["w1_scale"],
        prepared["w2_q"],
        prepared["w2_scale"],
        out=out,
        fp8_mma_form=fp8_mma_form,
        **ws,
    )
    torch.cuda.synchronize()
    a_ref = reference_modulated(x, model, idx)
    assert_fp8_stage1(ws["workspace_a_q"], ws["workspace_a_scale"], x, model, idx, a_ref)
    y_ref = swiglu_from_scaled(
        ws["workspace_a_q"],
        prepared["w1_lin_t"],
        a_scale=ws["workspace_a_scale"],
        w_scale=prepared["w1_lin_scale"],
    )
    assert_fp8_stage2(ws["workspace_y_q"], ws["workspace_y_scale"], ws["workspace_y"], y_ref)
    y_q = ws["workspace_y_q"]
    ref = fc2_gated_residual(
        lambda r0, r1: y_q[r0:r1].float(),
        prepared["w2_t"],
        model,
        idx,
        residual,
        y_scale=ws["workspace_y_scale"],
        w_scale=prepared["w2_scale"],
    )
    stats = assert_output_within_budget(returned, ref, f"fp8 M={rows} form={fp8_mma_form}")
    return returned, ref["out"], residual, stats


def nvfp4_scales(rows, model, prepared, device, x=None, idx=None):
    """Calibrated static global scales of ``a`` and ``y`` from the independent reference chain."""
    if x is None:
        x, _residual, idx = make_inputs(rows, device)
    a_ref = reference_modulated(x, model, idx)
    g_a = minimax_h3_nvfp4_global_scale(a_ref)
    a_q, a_sf, _ct, _st = nvfp4_quantize_emulated(a_ref, g_a)
    alpha1 = minimax_h3_nvfp4_alpha(g_a, prepared["g_w1"])
    y_chain = swiglu_from_scaled(
        nvfp4_dequantize_scaled(a_q, a_sf), prepared["w1_scaled_t"], alpha=alpha1.item()
    )
    g_y = minimax_h3_nvfp4_global_scale(y_chain)
    return g_a, g_y


def run_nvfp4_case(rows, model, prepared, device, idx=None, out=None):
    x, residual, default_idx = make_inputs(rows, device)
    idx = default_idx if idx is None else idx
    g_a, g_y = nvfp4_scales(rows, model, prepared, device, x=x, idx=idx)
    alpha1 = minimax_h3_nvfp4_alpha(g_a, prepared["g_w1"])
    alpha2 = minimax_h3_nvfp4_alpha(g_y, prepared["g_w2"])
    ws = _workspaces_nvfp4(rows, device)
    returned = minimax_h3_mlp_nvfp4_sm120(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["gate"],
        residual,
        g_a,
        prepared["w1_q"],
        prepared["w1_sf"],
        alpha1,
        g_y,
        prepared["w2_q"],
        prepared["w2_sf"],
        alpha2,
        out=out,
        **ws,
    )
    torch.cuda.synchronize()
    a_ref = reference_modulated(x, model, idx)
    assert_nvfp4_stage1(ws["workspace_a_q"], ws["workspace_a_sf"], a_ref, g_a)
    y_ref = swiglu_from_scaled(
        nvfp4_dequantize_scaled(ws["workspace_a_q"], ws["workspace_a_sf"]),
        prepared["w1_scaled_t"],
        alpha=alpha1.item(),
    )
    assert_nvfp4_stage2(ws["workspace_y_q"], ws["workspace_y_sf"], y_ref, g_y)
    y_q, y_sf = ws["workspace_y_q"], ws["workspace_y_sf"]
    ref = fc2_gated_residual(
        lambda r0, r1: nvfp4_dequantize_scaled(y_q[r0:r1], y_sf[r0:r1]),
        prepared["w2_scaled_t"],
        model,
        idx,
        residual,
        alpha=alpha2.item(),
    )
    stats = assert_output_within_budget(returned, ref, f"nvfp4 M={rows}")
    return returned, ref["out"], residual, stats


@requires_sm120
@pytest.mark.parametrize("rows", ROWS)
def test_minimax_h3_sm120_mlp_fp8(rows, model, prepared_fp8, device):
    run_fp8_case(rows, model, prepared_fp8, device)


@requires_sm120
@pytest.mark.parametrize("rows", ROWS)
def test_minimax_h3_sm120_mlp_nvfp4(rows, model, prepared_nvfp4, device):
    run_nvfp4_case(rows, model, prepared_nvfp4, device)


@requires_sm120
def test_minimax_h3_sm120_mlp_fp8_mma_forms_are_bitwise_identical(model, prepared_fp8, device):
    """The legacy FP8 ``mma.sync`` form and the ``kind::mxf8f6f4`` form with unit UE8M0 scales
    (the GeForce GB202 dispatch) accumulate the same products in FP32: identical outputs."""
    legacy, _ref, _res, _s = run_fp8_case(
        257, model, prepared_fp8, device, fp8_mma_form=FP8_MMA_FORM_LEGACY
    )
    unit_scaled, _ref, _res, _s = run_fp8_case(
        257, model, prepared_fp8, device, fp8_mma_form=FP8_MMA_FORM_MXF8F6F4
    )
    auto, _ref, _res, _s = run_fp8_case(257, model, prepared_fp8, device)
    assert torch.equal(legacy, unit_scaled)
    assert torch.equal(legacy, auto)


@requires_sm120
def test_minimax_h3_sm120_mlp_invalid_index_rows_reproduce_residual(model, prepared_fp8, prepared_nvfp4, device):
    rows = 300
    _x, _residual, idx = make_inputs(rows, device)
    idx = idx.clone()
    idx[:3] = torch.tensor([-1, MINIMAX_H3_ADALN_ROWS, -(2**63)], dtype=torch.int64, device=device)
    idx[257] = 2**40
    for run in (run_fp8_case, run_nvfp4_case):
        kwargs = {"idx": idx}
        prepared = prepared_fp8 if run is run_fp8_case else prepared_nvfp4
        out, _ref, residual, _stats = run(rows, model, prepared, device, **kwargs)
        for r in (0, 1, 2, 257):
            assert torch.equal(out[r], residual[r])


@requires_sm120
def test_minimax_h3_sm120_mlp_out_may_alias_residual(model, prepared_fp8, prepared_nvfp4, device):
    rows = 129
    for run, prepared in ((run_fp8_case, prepared_fp8), (run_nvfp4_case, prepared_nvfp4)):
        separate, _ref, _res, _s = run(rows, model, prepared, device)
        x, residual, idx = make_inputs(rows, device)
        aliased = residual.clone()
        returned, _ref, _res, _s = run(rows, model, prepared, device, out=aliased)
        assert returned.data_ptr() == aliased.data_ptr()
        assert torch.equal(aliased, separate)


@requires_sm120
def test_minimax_h3_sm120_mlp_strided_tables(model, prepared_fp8, device):
    """Column chunks of a wider modulation projection serve as the table views without a copy."""
    rows = 257
    wide = torch.zeros((MINIMAX_H3_ADALN_ROWS, 3 * MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device)
    wide[:, :MINIMAX_H3_HIDDEN] = model["adaln_shift"]
    wide[:, MINIMAX_H3_HIDDEN : 2 * MINIMAX_H3_HIDDEN] = model["adaln_scale"]
    wide[:, 2 * MINIMAX_H3_HIDDEN :] = model["gate"]
    strided = dict(model)
    strided["adaln_shift"] = wide[:, :MINIMAX_H3_HIDDEN]
    strided["adaln_scale"] = wide[:, MINIMAX_H3_HIDDEN : 2 * MINIMAX_H3_HIDDEN]
    strided["gate"] = wide[:, 2 * MINIMAX_H3_HIDDEN :]
    assert strided["gate"].stride(0) == 3 * MINIMAX_H3_HIDDEN
    out_strided, _ref, _res, _s = run_fp8_case(rows, strided, prepared_fp8, device)
    out_dense, _ref, _res, _s = run_fp8_case(rows, model, prepared_fp8, device)
    assert torch.equal(out_strided, out_dense)


@requires_sm120
def test_minimax_h3_sm120_mlp_default_workspaces(model, prepared_fp8, device):
    rows = 130
    x, residual, idx = make_inputs(rows, device)
    args = (
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        model["gate"],
        residual,
        prepared_fp8["w1_q"],
        prepared_fp8["w1_scale"],
        prepared_fp8["w2_q"],
        prepared_fp8["w2_scale"],
    )
    out = minimax_h3_mlp_fp8_sm120(*args)
    explicit, _ref, _res, _s = run_fp8_case(rows, model, prepared_fp8, device)
    torch.cuda.synchronize()
    assert torch.equal(out, explicit)


@requires_sm120
def test_minimax_h3_sm120_mlp_rejects_bad_inputs(model, prepared_fp8, prepared_nvfp4, device):
    rows = 8
    x, residual, idx = make_inputs(rows, device)
    tables = (model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx, model["gate"], residual)
    w8 = (prepared_fp8["w1_q"], prepared_fp8["w1_scale"], prepared_fp8["w2_q"], prepared_fp8["w2_scale"])
    with pytest.raises(ValueError):
        minimax_h3_mlp_fp8_sm120(x.float(), *tables, *w8)
    with pytest.raises(ValueError):
        minimax_h3_mlp_fp8_sm120(x[:, :64], *tables, *w8)
    with pytest.raises(ValueError):  # int32 index
        minimax_h3_mlp_fp8_sm120(x, *tables[:3], idx.int(), *tables[4:], *w8)
    with pytest.raises(ValueError):  # wrong FC2 weight dtype
        minimax_h3_mlp_fp8_sm120(x, *tables, *w8[:2], prepared_fp8["w2_q"].view(torch.uint8), w8[3])
    with pytest.raises(ValueError):
        minimax_h3_mlp_fp8_sm120(x, *tables, *w8, eps=0.0)
    with pytest.raises(ValueError):
        minimax_h3_mlp_fp8_sm120(x, *tables, *w8, fp8_mma_form=1)
    with pytest.raises(ValueError):  # flags workspace too small
        minimax_h3_mlp_fp8_sm120(
            x, *tables, *w8, workspace_flags=torch.zeros((0,), dtype=torch.int32, device=device)
        )
    with pytest.raises(ValueError):  # gate row count differs from the AdaLN tables
        minimax_h3_mlp_fp8_sm120(x, *tables[:4], model["gate"][:4], residual, *w8)
    g_a, g_y = nvfp4_scales(rows, model, prepared_nvfp4, device)
    w4 = (
        g_a,
        prepared_nvfp4["w1_q"],
        prepared_nvfp4["w1_sf"],
        1.0,
        g_y,
        prepared_nvfp4["w2_q"],
        prepared_nvfp4["w2_sf"],
        1.0,
    )
    with pytest.raises(ValueError):  # host float global scale
        minimax_h3_mlp_nvfp4_sm120(x, *tables, 1.0, *w4[1:])
    with pytest.raises(ValueError):  # truncated FC2 scales
        minimax_h3_mlp_nvfp4_sm120(x, *tables, *w4[:6], prepared_nvfp4["w2_sf"][:-1], 1.0)
