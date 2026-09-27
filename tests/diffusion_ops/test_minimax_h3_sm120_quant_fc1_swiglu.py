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
"""SM120 (GB202) FP8 / NVFP4 fused MiniMax-H3 RMSNorm + AdaLN + FC1 + SwiGLU against an exact
torch emulation.

Acceptance rule (shared with the SM100/SM103 operator tests): the operator rounds to BF16 three
times after the FP32 accumulation (h, silu(gate), the product) and silu amplifies a 1-ulp change of
a strongly negative gate several times, so two correct implementations with different FP32
accumulation orders legitimately disagree by a few BF16 ulps on a ~1e-6 fraction of the elements.
The check therefore bounds the number of elements outside ``atol + rtol * |ref|`` (1e-2 / 1.6e-2)
by ``max(4, 2e-7 * numel)``; a wrong tile or row produces thousands of violations.

Quantizing the BF16 activation is a threshold operation (the kernel and torch reduce the FP32 RMS
statistic in different orders, so single activations may differ by one BF16 ulp and flip a code or
a scale), so the reference GEMM consumes the kernel's *own* quantized activation and that activation
is checked separately against its definition:

* FP8: the per-token scale is ``RN(amax / 448)`` within one BF16 ulp of the row absmax and every
  dequantized value lies within half an E4M3 step plus the BF16 ulps of the norm/AdaLN round points
  of the reference activation (derived bound, zero violations).
* NVFP4: an exact torch emulation of FlashInfer's ``nvfp4_quantize`` recipe (same operation order,
  approximate reciprocals excluded through the rounding-tie masks) must match the kernel's codes and
  scales up to a bounded number of non-tie mismatches.
"""

import math
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import (
    minimax_h3_fc1_swiglu_fp8,
    minimax_h3_fc1_swiglu_nvfp4,
    prepare_minimax_h3_fc1_weight_fp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
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
    _minimax_h3_fc1_swiglu_nvfp4_sm120,
    deinterleave_minimax_h3_fc1_rows_sm120,
    fp8_scale_from_amax,
    interleave_minimax_h3_fc1_rows_sm120,
    prepare_minimax_h3_fc1_weight_nvfp4_sm120,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    _unswizzle_sf_128x4,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
    nvfp4_activation_scale_workspace_bytes,
)
from flashinfer.utils import get_compute_capability

ATOL = 1e-2
RTOL = 1.6e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4
# Bounded non-tie mismatch budgets of the kernel's NVFP4 activation against the recipe emulation.
MAX_E2M1_MISMATCH_FRACTION = 4e-6
MAX_E2M1_MISMATCH_FLOOR = 4
MAX_NV_SCALE_MISMATCH_FRACTION = 8e-6
MAX_SCALE_MISMATCH_FLOOR = 2
TIE_REL_TOL = 2.0**-18
# Tail lengths around the 128-row tile, plus multi-tile shapes (memory-lean: M <= 4097).
ROWS = [1, 127, 128, 129, 257, 4097]
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


def make_model(device: torch.device, seed: int = 4611) -> Dict[str, torch.Tensor]:
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
        "fc1_weight": normal((MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), 0.02),
    }


def make_inputs(
    rows: int, device: torch.device, seed: int = 4611
) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * rows)
    x = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.5, generator=g)
    # "production segments": nine contiguous AdaLN segments over the rows.
    r = torch.arange(rows, dtype=torch.int64, device=device)
    idx = torch.div(r * MINIMAX_H3_ADALN_ROWS, rows, rounding_mode="floor").clamp_max(
        MINIMAX_H3_ADALN_ROWS - 1
    )
    return x, idx.to(torch.int32)


# --------------------------------------------------------------------------------------------
# Reference math
# --------------------------------------------------------------------------------------------


def reference_modulated(x, model, adaln_index, eps=MINIMAX_H3_EPS):
    norm = F.rms_norm(x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=eps).to(
        torch.bfloat16
    )
    idx = adaln_index.long()
    valid = (idx >= 0) & (idx < MINIMAX_H3_ADALN_ROWS)
    safe = idx.clamp(0, MINIMAX_H3_ADALN_ROWS - 1)
    a = torch.addcmul(
        model["adaln_shift"].index_select(0, safe),
        norm,
        (model["adaln_scale"].index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


def swiglu_from_scaled(
    a: torch.Tensor,
    w_t: torch.Tensor,
    a_scale=None,
    w_scale=None,
    alpha=None,
) -> torch.Tensor:
    """``y = BF16(BF16(silu(h[:, :FFN])) * h[:, FFN:])`` with ``h = BF16(((a @ w^T) * a_scale) *
    w_scale)`` (FP8, the kernel's operation order) or ``h = BF16(alpha * (a @ w^T))`` (NVFP4),
    FP32 GEMM with TF32 disabled, evaluated in row chunks."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
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
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


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


def assert_within_budget(
    y: torch.Tensor, ref: torch.Tensor, what: str
) -> Dict[str, float]:
    assert y.shape == ref.shape and y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all(), f"{what}: non-finite output"
    stats = violation_stats(y, ref)
    assert stats["violations"] <= stats["budget"], f"{what}: {stats}"
    return stats


# ---- FP8 -----------------------------------------------------------------------------------


def quantize_fp8_rows(
    t: torch.Tensor, chunk_rows: int = 2048
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row E4M3: ``scale = RN(max(absmax, 1e-12) / 448)`` (true division), ``q = RN_sat(t / scale)``."""
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


def bf16_ulp(magnitude: torch.Tensor) -> torch.Tensor:
    mag = magnitude.float().abs().clamp_min(2.0**-126)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 7.0)


def e4m3_step(magnitude: torch.Tensor) -> torch.Tensor:
    """E4M3 spacing at ``|magnitude|`` (3 mantissa bits; subnormal floor 2^-9)."""
    mag = magnitude.float().abs().clamp_min(2.0**-6)
    return torch.pow(2.0, torch.floor(torch.log2(mag)) - 3.0).clamp_min(2.0**-9)


def assert_fp8_stage1(a_q, a_scale, x, model, idx, a_ref) -> None:
    """Derived-bound check of the kernel's per-token FP8 activation against the reference
    activation: ``|scale / scale_ref - 1| <= 2^-7`` and ``|a_q * scale - a_ref| <= 0.5 *
    e4m3_step(a_q) * scale + bf16_ulp(n_ref) * |1 + adaln_scale| + bf16_ulp(max(|a_ref|, |deq|))
    + 2^-20 |a_ref|`` (``shift + n * (1 + scale)`` can cancel, so a one-ulp flip of the normalized
    value is not small relative to ``a``)."""
    _q_ref, scale_ref = quantize_fp8_rows(a_ref)
    scale_c = a_scale.float()
    scale_rel = ((scale_c / scale_ref) - 1.0).abs()
    assert bool((scale_rel <= 2.0**-7).all()), (
        f"fp8 per-token scale off: max rel err {float(scale_rel.max()):.4g}"
    )
    n_ref = F.rms_norm(
        x, (MINIMAX_H3_HIDDEN,), model["x_norm_weight"], eps=MINIMAX_H3_EPS
    ).to(torch.bfloat16)
    safe = idx.long().clamp(0, MINIMAX_H3_ADALN_ROWS - 1)
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


# ---- NVFP4 (FlashInfer nvfp4_quantize recipe emulation) -------------------------------------


def nvfp4_scale_prerounded(absmax: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """FP32 ``g * (absmax * fp32(1/6))`` (FlashInfer operation order, before the UE4M3 rounding)."""
    rcp6 = torch.tensor(1.0 / 6.0, dtype=torch.float32, device=absmax.device)
    return g.float().reshape(()) * (absmax.float() * rcp6)


def nvfp4_scale_values(absmax, g) -> Tuple[torch.Tensor, torch.Tensor]:
    sf = torch.clamp(nvfp4_scale_prerounded(absmax, g), max=E4M3_MAX).to(
        torch.float8_e4m3fn
    )
    return sf.view(torch.uint8), sf.float()


def nvfp4_output_scale(sf_f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """FP32 ``1 / (sf * (1 / g))`` (FlashInfer operation order); 0 where the scale is 0."""
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
    """Round-to-nearest-even E2M1 codes (uint8 0..15, sign in bit 3) with saturation."""
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=v.device)
    mids = (grid[1:] + grid[:-1]) * 0.5
    mag = torch.clamp(v.abs(), max=E2M1_MAX)
    code = torch.bucketize(mag, mids, right=False)  # lower code on an exact tie
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


def nvfp4_dequantize_scaled(packed: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """Packed E2M1 ``[R, K/2]`` + UE4M3 bytes ``[R, K/16]`` -> FP32 ``q * sf`` (the operand the MMA
    consumes; the global scales enter through alpha)."""
    rows = packed.shape[0]
    grid = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=packed.device)
    codes = torch.empty(
        (rows, 2 * packed.shape[1]), dtype=torch.uint8, device=packed.device
    )
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    mag = grid[(codes & 7).long()]
    vals = torch.where((codes & 8) != 0, -mag, mag)
    sf_f = sf.view(torch.float8_e4m3fn).float()
    return (vals.reshape(rows, -1, MINIMAX_H3_SF_BLOCK) * sf_f[..., None]).reshape(
        rows, -1
    )


def assert_nvfp4_stage1(a_q, a_sf, a_ref, g_a) -> Dict[str, int]:
    """The kernel's NVFP4 activation against the recipe emulation of the reference activation:
    mismatches outside the rounding-tie positions must stay within bounded budgets (a layout or
    math error mismatches >10 % of the bytes)."""
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
    """Prepared (SM120 row order) FP8 weight plus its linear (gate rows; up rows) quantization."""
    w_q, w_scale = prepare_minimax_h3_fc1_weight_fp8(model["fc1_weight"])
    lin_q, lin_scale = quantize_fp8_rows(model["fc1_weight"])
    return {"w_q": w_q, "w_scale": w_scale, "lin_q": lin_q, "lin_scale": lin_scale}


@pytest.fixture(scope="module")
def prepared_nvfp4(model):
    """Prepared (SM120) NVFP4 weight plus ``w_q * w_sf`` (FP32) in the linear row order."""
    g_w = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    w_q, w_sf = prepare_minimax_h3_fc1_weight_nvfp4_sm120(model["fc1_weight"], g_w)
    sf_rows = _unswizzle_sf_128x4(w_sf, MINIMAX_H3_FC1_ROWS, NVFP4_SF_COLS)
    w_scaled = deinterleave_minimax_h3_fc1_rows_sm120(
        nvfp4_dequantize_scaled(w_q, sf_rows)
    )
    return {"g_w": g_w, "w_q": w_q, "w_sf": w_sf, "w_scaled": w_scaled}


# --------------------------------------------------------------------------------------------
# Weight preparation: the documented row permutation of the linear quantization
# --------------------------------------------------------------------------------------------


@requires_sm120
def test_prepare_fp8_weight_is_interleaved_row_quantization(prepared_fp8):
    assert prepared_fp8["w_q"].dtype == torch.float8_e4m3fn
    assert tuple(prepared_fp8["w_q"].shape) == (MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN)
    assert tuple(prepared_fp8["w_scale"].shape) == (MINIMAX_H3_FC1_ROWS,)
    assert torch.equal(
        prepared_fp8["w_q"].view(torch.uint8),
        interleave_minimax_h3_fc1_rows_sm120(prepared_fp8["lin_q"]).view(torch.uint8),
    )
    assert torch.equal(
        prepared_fp8["w_scale"],
        interleave_minimax_h3_fc1_rows_sm120(prepared_fp8["lin_scale"]),
    )
    assert torch.equal(
        deinterleave_minimax_h3_fc1_rows_sm120(prepared_fp8["w_scale"]),
        prepared_fp8["lin_scale"],
    )
    # Row 16 * (c // 8) + (c % 8) holds gate column c, row + 8 the matching up column.
    c = 1234
    gate_row = 16 * (c // 8) + (c % 8)
    assert torch.equal(prepared_fp8["w_q"][gate_row], prepared_fp8["lin_q"][c])
    assert torch.equal(
        prepared_fp8["w_q"][gate_row + 8], prepared_fp8["lin_q"][MINIMAX_H3_FFN + c]
    )


@requires_sm120
def test_prepare_nvfp4_weight_is_interleaved_row_quantization(model, prepared_nvfp4):
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    lin_q, lin_sf = nvfp4_quantize(
        model["fc1_weight"],
        prepared_nvfp4["g_w"],
        sfLayout=SfLayout.layout_128x4,
        do_shuffle=False,
    )
    lin_q = lin_q.view(torch.uint8).reshape(MINIMAX_H3_FC1_ROWS, NVFP4_PACKED_COLS)
    lin_sf = _unswizzle_sf_128x4(
        lin_sf.view(torch.uint8).reshape(-1), MINIMAX_H3_FC1_ROWS, NVFP4_SF_COLS
    )
    assert tuple(prepared_nvfp4["w_q"].shape) == (
        MINIMAX_H3_FC1_ROWS,
        NVFP4_PACKED_COLS,
    )
    assert prepared_nvfp4["w_sf"].numel() == MINIMAX_H3_FC1_ROWS * NVFP4_SF_COLS
    assert torch.equal(
        prepared_nvfp4["w_q"], interleave_minimax_h3_fc1_rows_sm120(lin_q)
    )
    assert torch.equal(
        _unswizzle_sf_128x4(prepared_nvfp4["w_sf"], MINIMAX_H3_FC1_ROWS, NVFP4_SF_COLS),
        interleave_minimax_h3_fc1_rows_sm120(lin_sf),
    )
    # The public (dispatching) preparation returns the SM120 layout on this device.
    pub_q, pub_sf = prepare_minimax_h3_fc1_weight_nvfp4(
        model["fc1_weight"], prepared_nvfp4["g_w"]
    )
    assert torch.equal(pub_q, prepared_nvfp4["w_q"])
    assert torch.equal(pub_sf.reshape(-1), prepared_nvfp4["w_sf"])


# --------------------------------------------------------------------------------------------
# Operators
# --------------------------------------------------------------------------------------------


def run_fp8_case(rows: int, model, prepared, device, idx=None) -> Dict[str, float]:
    x, default_idx = make_inputs(rows, device)
    idx = default_idx if idx is None else idx
    workspace_q = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    workspace_scale = torch.empty((rows,), dtype=torch.float32, device=device)
    out = minimax_h3_fc1_swiglu_fp8(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        prepared["w_q"],
        prepared["w_scale"],
        workspace_q=workspace_q,
        workspace_scale=workspace_scale,
    )
    torch.cuda.synchronize()
    a_ref = reference_modulated(x, model, idx)
    assert_fp8_stage1(workspace_q, workspace_scale, x, model, idx, a_ref)
    ref = swiglu_from_scaled(
        workspace_q,
        prepared["lin_q"].float().t(),
        a_scale=workspace_scale,
        w_scale=prepared["lin_scale"],
    )
    stats = assert_within_budget(out, ref, f"fp8 M={rows}")
    return out, stats


def run_nvfp4_case(rows: int, model, prepared, device) -> Dict[str, float]:
    x, idx = make_inputs(rows, device)
    a_ref = reference_modulated(x, model, idx)
    # Static activation global scale calibrated from the reference activation of this shape.
    g_a = minimax_h3_nvfp4_global_scale(a_ref)
    alpha = minimax_h3_nvfp4_alpha(g_a, prepared["g_w"])
    workspace_q = torch.empty(
        (rows, NVFP4_PACKED_COLS), dtype=torch.uint8, device=device
    )
    # Caller buffer sized by the shared upper bound; SM120 uses its first M * 336 bytes densely.
    workspace_sf = torch.zeros(
        (nvfp4_activation_scale_workspace_bytes(rows),),
        dtype=torch.uint8,
        device=device,
    )
    out = _minimax_h3_fc1_swiglu_nvfp4_sm120(
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        g_a,
        prepared["w_q"],
        prepared["w_sf"],
        alpha,
        workspace_q=workspace_q,
        workspace_sf=workspace_sf,
    )
    torch.cuda.synchronize()
    a_sf = workspace_sf[: rows * NVFP4_SF_COLS].view(rows, NVFP4_SF_COLS)
    assert bool((workspace_sf[rows * NVFP4_SF_COLS :] == 0).all()), (
        "bytes beyond the dense [M, 336] scales were written"
    )
    assert_nvfp4_stage1(workspace_q, a_sf, a_ref, g_a)
    ref = swiglu_from_scaled(
        nvfp4_dequantize_scaled(workspace_q, a_sf),
        prepared["w_scaled"].t(),
        alpha=alpha.item(),
    )
    stats = assert_within_budget(out, ref, f"nvfp4 M={rows}")
    return out, stats


@requires_sm120
@pytest.mark.parametrize("rows", ROWS)
def test_minimax_h3_sm120_fc1_swiglu_fp8(rows, model, prepared_fp8, device):
    run_fp8_case(rows, model, prepared_fp8, device)


@requires_sm120
@pytest.mark.parametrize("rows", ROWS)
def test_minimax_h3_sm120_fc1_swiglu_nvfp4(rows, model, prepared_nvfp4, device):
    run_nvfp4_case(rows, model, prepared_nvfp4, device)


@requires_sm120
def test_minimax_h3_sm120_fc1_swiglu_nvfp4_public_dispatch(
    model, prepared_nvfp4, device
):
    """The public ``minimax_h3_fc1_swiglu_nvfp4`` routes to the SM120 kernels on this device."""
    rows = 257
    x, idx = make_inputs(rows, device)
    g_a = minimax_h3_nvfp4_global_scale(reference_modulated(x, model, idx))
    alpha = minimax_h3_nvfp4_alpha(g_a, prepared_nvfp4["g_w"])
    args = (
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
        g_a,
        prepared_nvfp4["w_q"],
        prepared_nvfp4["w_sf"],
        alpha,
    )
    out = torch.full(
        (rows, MINIMAX_H3_FFN), float("nan"), dtype=torch.bfloat16, device=device
    )
    returned = minimax_h3_fc1_swiglu_nvfp4(*args, out=out)
    direct = _minimax_h3_fc1_swiglu_nvfp4_sm120(*args)
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    assert torch.isfinite(out.float()).all()
    assert torch.equal(out, direct)


@requires_sm120
def test_minimax_h3_sm120_fc1_swiglu_invalid_index_rows_are_zero(
    model, prepared_fp8, device
):
    rows = 200
    _x, idx = make_inputs(rows, device)
    idx = idx.clone()
    idx[:3] = torch.tensor(
        [-1, MINIMAX_H3_ADALN_ROWS, -(2**31)], dtype=torch.int32, device=device
    )
    out, _stats = run_fp8_case(rows, model, prepared_fp8, device, idx=idx)
    assert (out[:3] == 0).all()


@requires_sm120
def test_minimax_h3_sm120_fc1_swiglu_rejects_bad_inputs(
    model, prepared_fp8, prepared_nvfp4, device
):
    rows = 8
    x, idx = make_inputs(rows, device)
    norm = (model["x_norm_weight"], model["adaln_scale"], model["adaln_shift"], idx)
    fp8_w = (prepared_fp8["w_q"], prepared_fp8["w_scale"])
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(x.float(), *norm, *fp8_w)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(x[:, :64], *norm, *fp8_w)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(x, *norm[:3], idx.long(), *fp8_w)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(
            x, *norm, prepared_fp8["w_q"].view(torch.uint8), fp8_w[1]
        )
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(x, *norm, *fp8_w, eps=0.0)
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(
            x,
            *norm,
            *fp8_w,
            out=torch.empty((rows, 64), dtype=torch.bfloat16, device=device),
        )
    with pytest.raises(ValueError):
        minimax_h3_fc1_swiglu_fp8(
            x,
            *norm,
            *fp8_w,
            workspace_scale=torch.empty(
                (rows + 1,), dtype=torch.float32, device=device
            ),
        )
    g_a = minimax_h3_nvfp4_global_scale(reference_modulated(x, model, idx))
    alpha = minimax_h3_nvfp4_alpha(g_a, prepared_nvfp4["g_w"])
    nv_w = (prepared_nvfp4["w_q"], prepared_nvfp4["w_sf"])
    with pytest.raises(ValueError):
        _minimax_h3_fc1_swiglu_nvfp4_sm120(
            x, *norm, g_a, nv_w[0][:, :64], nv_w[1], alpha
        )
    with pytest.raises(ValueError):
        _minimax_h3_fc1_swiglu_nvfp4_sm120(x, *norm, g_a, nv_w[0], nv_w[1][:-1], alpha)
    with pytest.raises(ValueError):
        _minimax_h3_fc1_swiglu_nvfp4_sm120(x, *norm, float(g_a.item()), *nv_w, alpha)
    with pytest.raises(ValueError):
        _minimax_h3_fc1_swiglu_nvfp4_sm120(
            x,
            *norm,
            g_a,
            *nv_w,
            alpha,
            workspace_sf=torch.empty(
                (rows * NVFP4_SF_COLS - 1,), dtype=torch.uint8, device=device
            ),
        )
