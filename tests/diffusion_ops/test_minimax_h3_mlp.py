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
"""Tests for the MiniMax-H3 full MLP block operator (BF16 / MXFP8 / NVFP4): RMSNorm + indexed
AdaLN + FC1 + SwiGLU + FC2 + indexed gate + residual in three launches.

Reference = the segmented chain built from FlashInfer's own pieces: ``minimax_h3_fc1_swiglu*``
for ``y`` (the fused kernels reuse that FC1 GEMM core), FlashInfer's ``mxfp8_quantize`` /
``nvfp4_quantize`` of that ``y`` for the quantized variants, an FP32 ``torch`` FC2 GEMM (TF32
disabled) rounded to BF16, the indexed gate product rounded to BF16 and the residual sum rounded
to BF16.

Acceptance rule (the MiniMax-H3 out-stage rule, shared by the three variants and by the SM120 out-proj test): the
operator rounds to BF16 three times after the FC2 accumulation (o, gate * o, residual + p), so two
correct implementations with different FP32 accumulation orders legitimately disagree by one BF16
step of ``o``.  Every element must satisfy ``|out - ref| <= atol + rtol * mag + 2 bf16_ulp(mag)``
(1e-2 / 1e-2, ``mag = max(|ref|, |gate * o|)``) -- zero budget; a wrong tile, row or K half
produces thousands of violations.  The out stage is judged on the operator's own ``y`` workspace
(the y stage attributes FC1 separately).  The strict ``atol + rtol * |ref|`` count is reported as
a diagnostic only: where the gated projection cancels against the residual (``|p| >> |out|``) a
one-step flip of ``p`` exceeds ``rtol * |out|`` (15 / 1.4 M elements at M = 257 and 235 / 26 M at
M = 4824 on B200, all within two steps of ``mag``).
Rows whose int64 index lies outside the table must equal ``residual`` bit-exactly (the guard zeroes
the modulated activation and the gate).  The intermediate stages are checked as well: the BF16
``y`` workspace against FlashInfer's FC1 operator under the FC1 contract rule (rtol 1.6e-2), the
quantized ``a`` workspaces bit-exactly against FlashInfer's own quantizer of this operator's BF16
activation (and within the activation budget of the FC1 operator's, which is a fast-math build), and the
quantized ``y`` written by the FC1 epilogue against FlashInfer's quantizer applied to the FC1
operator's ``y`` under the FC1 operator's mismatch budgets on the rows whose quantized activation both
operators agree on (a few E4M3 / E2M1 codes may flip where the two FC1 kernels round an FP32
accumulator differently); the rare rows where they do not are judged against FlashInfer's quantizer
of the FP32 FC1 oracle on this operator's own quantized activation (the Cake contract's rule).
"""

import math
from typing import Dict, Tuple

import pytest
import torch
import torch.nn.functional as F

from flashinfer.diffusion_ops import (
    minimax_h3_fc1_swiglu,
    minimax_h3_fc1_swiglu_mxfp8,
    minimax_h3_fc1_swiglu_nvfp4,
    minimax_h3_mlp,
    minimax_h3_mlp_mxfp8,
    minimax_h3_mlp_nvfp4,
    prepare_minimax_h3_fc1_weight_mxfp8,
    prepare_minimax_h3_fc1_weight_nvfp4,
    prepare_minimax_h3_fc2_weight_mxfp8,
    prepare_minimax_h3_fc2_weight_nvfp4,
)
from flashinfer.diffusion_ops.minimax_h3_fc1_swiglu import (
    _unswizzle_sf_128x4,
    minimax_h3_nvfp4_alpha,
    minimax_h3_nvfp4_global_scale,
)
from flashinfer.diffusion_ops.minimax_h3_mlp import (
    MINIMAX_H3_EPS,
    MINIMAX_H3_FC1_ROWS,
    MINIMAX_H3_FFN,
    MINIMAX_H3_GATE_ROWS,
    MINIMAX_H3_HIDDEN,
    MXFP8_A_SF_COLS,
    MXFP8_A_SF_K_TILES,
    MXFP8_BLOCK,
    MXFP8_FC2_SCALE_TILE_BYTES,
    MXFP8_Y_SF_COLS,
    MXFP8_Y_SF_K_TILES,
    NVFP4_A_PACKED_COLS,
    NVFP4_A_SF_COLS,
    NVFP4_A_SF_K_TILES,
    NVFP4_BLOCK,
    NVFP4_FC2_SCALE_TILE_BYTES,
    NVFP4_Y_PACKED_COLS,
    NVFP4_Y_SF_COLS,
    NVFP4_Y_SF_K_TILES,
    _fc2_tail_workspace,
    mxfp8_a_scale_workspace_bytes,
    mxfp8_y_scale_workspace_bytes,
    nvfp4_a_scale_workspace_bytes,
    nvfp4_y_scale_workspace_bytes,
)
from flashinfer.diffusion_ops.minimax_h3_out_proj import (
    _reference_from_operands,
    _weight_scale_tiles,
)
from flashinfer.utils import get_compute_capability

ATOL = 1e-2
RTOL = 1e-2
# The y stage is judged with the FC1 contract's rtol (PyTorch's BF16 comparison default).
Y_RTOL = 1.6e-2
MAX_VIOLATION_FRACTION = 2.0e-7
MAX_VIOLATIONS_FLOOR = 4
# The FC1 operator's quantized-activation budgets for the y stage (codes / scale bytes).
MAX_CODE_MISMATCH_FRACTION = 4.0e-6
MAX_CODE_MISMATCH_FLOOR = 4
MAX_SCALE_MISMATCH_FRACTION = 1.0e-6
MAX_SCALE_MISMATCH_FLOOR = 2
# Rows on which the FC1 operator (FlashInfer's default fast-math build) quantizes the
# modulated activation differently from this operator's precise build (one-ulp division rounding).
MAX_FASTMATH_ROW_FRACTION = 4.0e-3
MAX_FASTMATH_ROW_FLOOR = 4
# One row, one partial pair of 128-row tiles (129 -> 2 tiles, 257 -> 3 tiles padded to 4) and the
# production token count of one rank at sequence-parallel degree 8.  On a 148-SM / 160-SM part the
# FC2 tail split-K is active for M = 1, 129 and 4824 (21 / 21 / 399 pair tiles) and off for 257.
M_VALUES = [1, 129, 257, 4824]
# Engine modulation projection: the three tables are the shift_mlp / scale_mlp / gate_mlp column
# chunks (3 / 4 / 5) of a [rows, 6 * 5376] buffer with rows = 3 x unique timesteps (3 or 6 in the
# t2va pipeline); 12 exercises rows above 9.
ENGINE_TABLE_CHUNKS = 6
ENGINE_SHIFT_CHUNK, ENGINE_SCALE_CHUNK, ENGINE_GATE_CHUNK = 3, 4, 5
ENGINE_TABLE_ROWS = [3, 6, 12]
_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return tuple(get_compute_capability(torch.device("cuda:0"))) in {(10, 0), (10, 3)}


requires_blackwell = pytest.mark.skipif(
    not _supported(), reason="requires a CUDA GPU with compute capability 10.0 or 10.3"
)


# --------------------------------------------------------------------------------------------
# Deterministic inputs
# --------------------------------------------------------------------------------------------


def make_model(device: torch.device, seed: int = 4612) -> Dict[str, torch.Tensor]:
    """Parameters scaled like the sibling tests: ``fc1_weight`` ~ N(0, 0.02) (FC1 recipe), ``gate``
    ~ U(-1, 1) and ``fc2_weight`` ~ N(0, 0.01) (out-proj recipe)."""
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
        "adaln_scale": uniform((MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "adaln_shift": uniform((MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), -0.05, 0.05),
        "fc1_weight": normal((MINIMAX_H3_FC1_ROWS, MINIMAX_H3_HIDDEN), 0.02),
        "gate": uniform((MINIMAX_H3_GATE_ROWS, MINIMAX_H3_HIDDEN), -1.0, 1.0),
        "fc2_weight": normal((MINIMAX_H3_HIDDEN, MINIMAX_H3_FFN), 0.01),
    }


def make_engine_model(
    model: Dict[str, torch.Tensor],
    table_rows: int,
    device: torch.device,
    seed: int = 4613,
) -> Dict[str, torch.Tensor]:
    """``model`` with its three tables replaced by column chunks 3 (shift), 4 (scale) and 5 (gate)
    of ONE ``[table_rows, 6 * 5376]`` modulation projection: row stride ``6 * 5376``, not
    contiguous, one index for all three."""
    g = torch.Generator(device=device)
    g.manual_seed(seed + table_rows)
    proj = torch.empty(
        (table_rows, ENGINE_TABLE_CHUNKS * MINIMAX_H3_HIDDEN),
        dtype=torch.bfloat16,
        device=device,
    ).uniform_(-0.05, 0.05, generator=g)

    def chunk(index):
        return proj[:, index * MINIMAX_H3_HIDDEN : (index + 1) * MINIMAX_H3_HIDDEN]

    gate = chunk(ENGINE_GATE_CHUNK)
    gate.uniform_(-1.0, 1.0, generator=g)
    assert gate.stride() == (ENGINE_TABLE_CHUNKS * MINIMAX_H3_HIDDEN, 1)
    assert not gate.is_contiguous()
    return {
        **model,
        "adaln_shift": chunk(ENGINE_SHIFT_CHUNK),
        "adaln_scale": chunk(ENGINE_SCALE_CHUNK),
        "gate": gate,
        "_engine_backing": proj,
    }


def make_index(
    rows: int, device: torch.device, table_rows: int = MINIMAX_H3_GATE_ROWS
) -> torch.Tensor:
    """``table_rows`` contiguous segments over the rows (int64), with out-of-range indices planted
    at a few rows (``-1`` where ``row % 101 == 50``, ``table_rows`` where ``row % 103 == 60``) so
    the device-side guard (``a = 0``, ``gate = 0``, i.e. ``out = residual``) is exercised."""
    r = torch.arange(rows, dtype=torch.int64, device=device)
    idx = torch.div(r * table_rows, rows, rounding_mode="floor").clamp_max(
        table_rows - 1
    )
    idx = torch.where(r % 101 == 50, torch.full_like(idx, -1), idx)
    idx = torch.where(r % 103 == 60, torch.full_like(idx, table_rows), idx)
    return idx


def make_inputs(
    rows: int,
    device: torch.device,
    seed: int = 4612,
    table_rows: int = MINIMAX_H3_GATE_ROWS,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(x [M, 5376], adaln_index [M], residual [M, 5376])``."""
    g = torch.Generator(device=device)
    g.manual_seed(seed + 7919 * rows)
    x = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 0.5, generator=g)
    residual = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    ).normal_(0.0, 1.0, generator=g)
    return x, make_index(rows, device, table_rows), residual


# --------------------------------------------------------------------------------------------
# Reference math
# --------------------------------------------------------------------------------------------


def reference_modulated(
    x, x_norm_weight, adaln_scale, adaln_shift, adaln_index, eps=MINIMAX_H3_EPS
):
    norm = F.rms_norm(x, (MINIMAX_H3_HIDDEN,), x_norm_weight, eps=eps).to(
        torch.bfloat16
    )
    table_rows = int(adaln_scale.shape[0])
    idx = adaln_index.long()
    valid = (idx >= 0) & (idx < table_rows)
    safe = idx.clamp(0, table_rows - 1)
    a = torch.addcmul(
        adaln_shift.index_select(0, safe),
        norm,
        (adaln_scale.index_select(0, safe) + 1.0).to(torch.bfloat16),
    ).to(torch.bfloat16)
    return torch.where(valid[:, None], a, torch.zeros_like(a))


REFERENCE_CHUNK_ROWS = 1024


def _fc1_reference_from_operands(
    a: torch.Tensor, w: torch.Tensor, alpha=None
) -> torch.Tensor:
    """``y = BF16(silu(BF16(alpha * a @ w^T)[:, :FFN]) * BF16(...)[:, FFN:])`` with an FP32 GEMM
    (TF32 disabled) over any float-convertible operands -- the FC1 contract oracle of
    ``test_minimax_h3_fc1_swiglu.swiglu_from_operands``, evaluated in row chunks."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        w_t = w.float().t()
        out = torch.empty(
            (a.shape[0], MINIMAX_H3_FFN), dtype=torch.bfloat16, device=a.device
        )
        for r0 in range(0, a.shape[0], REFERENCE_CHUNK_ROWS):
            r1 = min(a.shape[0], r0 + REFERENCE_CHUNK_ROWS)
            h = a[r0:r1].float() @ w_t
            if alpha is not None:
                h = h * float(alpha)
            h = h.to(torch.bfloat16)
            gate, up = h.chunk(2, dim=-1)
            out[r0:r1] = (F.silu(gate) * up).to(torch.bfloat16)
        return out
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


def mxfp8_dequantize(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    """E4M3 ``[R, K]`` + UE8M0 bytes ``[R, K/32]`` -> FP32 ``[R, K]``."""
    rows, cols = q.shape
    scale = torch.exp2(sf.float() - 127.0)
    return (
        q.float().reshape(rows, cols // MXFP8_BLOCK, MXFP8_BLOCK) * scale[:, :, None]
    ).reshape(rows, cols)


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
    return (vals.reshape(rows, -1, NVFP4_BLOCK) * sf_f[..., None]).reshape(rows, -1)


def violation_stats(
    y: torch.Tensor, ref: torch.Tensor, atol: float = ATOL, rtol: float = RTOL
) -> Dict[str, float]:
    diff = (y.float() - ref.float()).abs()
    bad = diff > (atol + rtol * ref.float().abs())
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
    y: torch.Tensor,
    ref: torch.Tensor,
    what: str,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> Dict[str, float]:
    assert y.shape == ref.shape and y.dtype == torch.bfloat16
    assert torch.isfinite(y.float()).all(), f"{what}: non-finite output"
    stats = violation_stats(y, ref, atol, rtol)
    assert stats["violations"] <= stats["budget"], f"{what}: {stats}"
    return stats


def bf16_ulp(x: torch.Tensor) -> torch.Tensor:
    magnitude = x.float().abs().clamp_min(2.0**-126)
    return torch.exp2(torch.floor(torch.log2(magnitude)) - 7)


def assert_out_within_rule(
    out: torch.Tensor, ref: torch.Tensor, residual: torch.Tensor, what: str
) -> Dict[str, float]:
    """The out stage's shared MiniMax-H3 rule (``test_minimax_h3_sm120_quant_out_proj.assert_matches``):
    ``|out - ref| <= atol + rtol * mag + 2 * bf16_ulp(mag)`` with ``mag = max(|ref|, |p_ref|)``,
    ``p_ref = ref - residual`` (the gated projection before the residual sum, recovered up to one
    BF16 step).  One BF16 flip of ``o`` (FP32 accumulation order) moves ``gate * o`` by at most two
    steps at ``|p|``, the rounding of ``p`` adds one and the final ``residual + p`` rounding one
    more: at most four steps at ``mag`` against ``rtol * mag >= 2.56`` steps plus the two steps of
    slack, so any element outside the bound is a real defect (zero budget).  The strict
    ``atol + rtol * |ref|`` count is kept as a diagnostic: where ``|p| >> |out|`` (cancellation
    against the residual) a one-step flip of ``p`` legitimately exceeds ``rtol * |out|``."""
    assert out.shape == ref.shape and out.dtype == torch.bfloat16
    out32 = out.float()
    assert torch.isfinite(out32).all(), f"{what}: non-finite output"
    ref32 = ref.float()
    mag = torch.maximum(ref32.abs(), (ref32 - residual.float()).abs())
    bound = ATOL + RTOL * mag + 2.0 * bf16_ulp(mag)
    err = (out32 - ref32).abs()
    bad = err > bound
    stats = violation_stats(out, ref)
    stats["strict_violations"] = stats.pop("violations")
    stats["strict_budget"] = stats.pop("budget")
    stats["violations"] = int(bad.sum().item())
    stats["budget"] = 0
    assert stats["violations"] == 0, f"{what}: {stats}"
    return stats


def assert_guard_rows(
    out: torch.Tensor,
    residual: torch.Tensor,
    idx: torch.Tensor,
    table_rows: int,
    what: str,
) -> int:
    """Rows with an index outside ``[0, table_rows)`` pass the residual through bit-exactly."""
    invalid = (idx < 0) | (idx >= table_rows)
    count = int(invalid.sum().item())
    if count:
        assert torch.equal(out[invalid], residual[invalid]), (
            f"{what}: {count} guard rows must equal residual bit-exactly"
        )
    return count


def assert_quantized_activation_within_budget(
    q: torch.Tensor,
    sf: torch.Tensor,
    ref_q: torch.Tensor,
    ref_sf: torch.Tensor,
    what: str,
) -> Dict[str, int]:
    q_mismatch = int((q.view(torch.uint8) != ref_q.view(torch.uint8)).sum().item())
    sf_mismatch = int((sf != ref_sf).sum().item())
    q_budget = max(
        MAX_CODE_MISMATCH_FLOOR, int(math.ceil(MAX_CODE_MISMATCH_FRACTION * q.numel()))
    )
    sf_budget = max(
        MAX_SCALE_MISMATCH_FLOOR,
        int(math.ceil(MAX_SCALE_MISMATCH_FRACTION * sf.numel())),
    )
    assert q_mismatch <= q_budget and sf_mismatch <= sf_budget, (
        f"{what}: {q_mismatch} code bytes differ (budget {q_budget}), "
        f"{sf_mismatch} scale bytes differ (budget {sf_budget})"
    )
    return {"code_mismatches": q_mismatch, "scale_mismatches": sf_mismatch}


def flashinfer_mxfp8_quantize(
    t: torch.Tensor, sf_cols: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's own MXFP8 quantization -> (E4M3 ``[R, K]``, linear UE8M0 ``[R, K/32]``)."""
    from flashinfer.quantization.fp8_quantization import mxfp8_quantize

    q, sf = mxfp8_quantize(t.contiguous(), is_sf_swizzled_layout=True)
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), t.shape[0], sf_cols)
    return q.view(torch.float8_e4m3fn), sf


def flashinfer_nvfp4_quantize(
    t: torch.Tensor, g: torch.Tensor, packed_cols: int, sf_cols: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """FlashInfer's own NVFP4 quantization -> (packed E2M1 ``[R, K/2]``, linear UE4M3 ``[R, K/16]``)."""
    from flashinfer.quantization.fp4_quantization import nvfp4_quantize
    from flashinfer.tllm_enums import SfLayout

    q, sf = nvfp4_quantize(
        t.contiguous(), g, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    rows = t.shape[0]
    q = q.view(torch.uint8).reshape(rows, packed_cols)
    sf = _unswizzle_sf_128x4(sf.view(torch.uint8).reshape(-1), rows, sf_cols)
    return q, sf


def kernel_sf_linear(
    workspace_sf: torch.Tensor, rows: int, sf_cols: int, k_tiles: int
) -> torch.Tensor:
    """Swizzled 128x4 scale workspace (row tiles padded to an even count) -> linear ``[rows, sf_cols]``."""
    padded_rows = workspace_sf.numel() // (k_tiles * 512) * 128
    return _unswizzle_sf_128x4(
        workspace_sf[: padded_rows * sf_cols], padded_rows, sf_cols
    )[:rows]


# --------------------------------------------------------------------------------------------
# Variant checks
# --------------------------------------------------------------------------------------------


def _table_rows(model) -> int:
    return int(model["adaln_scale"].shape[0])


def _bf16_activation_from_operator(model, x, idx, residual, device) -> torch.Tensor:
    """This operator's own BF16 modulated activation (the norm + AdaLN kernel of the BF16 variant,
    the same arithmetic the quantizing norm kernels apply before their quantize epilogue)."""
    rows = x.shape[0]
    workspace_a = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    workspace_y = torch.empty(
        (rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device
    )
    minimax_h3_mlp(
        *_norm_args(model, x, idx),
        model["fc1_weight"],
        model["fc2_weight"],
        model["gate"],
        residual.clone(),
        workspace_a=workspace_a,
        workspace_y=workspace_y,
    )
    torch.cuda.synchronize()
    return workspace_a


def assert_a_stage(
    workspace_a_q: torch.Tensor,
    a_sf: torch.Tensor,
    ref_q: torch.Tensor,
    ref_sf: torch.Tensor,
    fc1_q: torch.Tensor,
    fc1_sf: torch.Tensor,
    valid_rows: torch.Tensor,
    what: str,
) -> torch.Tensor:
    """The a stage is bit-exact against FlashInfer's own quantizer applied to this operator's BF16
    activation (this module is built without ``-use_fast_math``, like the Cake build the contract
    validated), and its guard rows (``valid_rows`` false) are all-zero codes and scale bytes.  The
    FC1 operator is FlashInfer's default fast-math build, whose norm differs from the precise one
    on rare division-rounding elements, so against its quantized activation the a stage is held to
    the activation budget instead of bit-exactness.  That comparison covers the valid rows only:
    the FC1 operator's guard-row workspace bytes are not reliable under CUDA 12.9, whose ptxas
    folds that kernel's constant-zero FP8 pack into an ``F2FP`` merge with an undefined register
    and leaves stale bytes in the high half of every zero code word (its scale byte is 0, so its y
    and out are unaffected).  Returns the valid-row mask on which both operators' quantized
    activations are identical."""
    assert torch.equal(workspace_a_q.view(torch.uint8), ref_q.view(torch.uint8)), (
        f"{what}: quantized a differs from FlashInfer's quantizer of this operator's activation"
    )
    assert torch.equal(a_sf, ref_sf), (
        f"{what}: a scales differ from FlashInfer's quantizer of this operator's activation"
    )
    guard = ~valid_rows
    if bool(guard.any()):
        assert not bool(workspace_a_q[guard].view(torch.uint8).any()), (
            f"{what}: guard rows must quantize to all-zero codes"
        )
        assert not bool(a_sf[guard].any()), (
            f"{what}: guard rows must have all-zero scale bytes"
        )
    assert_quantized_activation_within_budget(
        workspace_a_q[valid_rows],
        a_sf[valid_rows],
        fc1_q[valid_rows],
        fc1_sf[valid_rows],
        f"{what} a stage vs the FC1 operator (valid rows)",
    )
    same_rows = (workspace_a_q.view(torch.uint8) == fc1_q.view(torch.uint8)).all(dim=1)
    same_rows &= (a_sf == fc1_sf).all(dim=1)
    same_rows &= valid_rows
    rows = int(valid_rows.sum().item())
    budget = max(
        MAX_FASTMATH_ROW_FLOOR, int(math.ceil(MAX_FASTMATH_ROW_FRACTION * rows))
    )
    other = rows - int(same_rows.sum().item())
    assert other <= budget, (
        f"{what}: {other} rows quantize the activation differently from the FC1 operator (budget {budget})"
    )
    return same_rows


def assert_y_stage(
    y_q: torch.Tensor,
    y_sf: torch.Tensor,
    fc1_q: torch.Tensor,
    fc1_sf: torch.Tensor,
    same_rows: torch.Tensor,
    valid_rows: torch.Tensor,
    what: str,
    oracle_check,
    guard_check,
) -> Dict[str, int]:
    """Quantized y against FlashInfer's quantizer of the FC1 operator's y under the FC1 operator's code /
    scale budgets on the rows whose quantized activation both operators agree on (same operand,
    same GEMM arithmetic).  On the few rows where the fast-math FC1 norm quantizes the activation
    differently from this operator's precise one the two FC1 inputs differ by a quantization step,
    so no agreement with the FC1 operator's y is expected; ``oracle_check(rows)`` judges those rows
    the way the Cake contract judges every row: the kernel's quantized y against FlashInfer's
    quantizer of the FP32 FC1 oracle evaluated on this operator's own quantized activation, under
    the same code / scale budgets.  Guard rows (``valid_rows`` false) are not compared with the FC1
    operator (see ``assert_a_stage``); ``guard_check(rows)`` holds the kernel's quantized y on them
    bit-exact to FlashInfer's quantizer of a zero y."""
    stats = assert_quantized_activation_within_budget(
        y_q[same_rows], y_sf[same_rows], fc1_q[same_rows], fc1_sf[same_rows], what
    )
    other = valid_rows & ~same_rows
    stats["rows_with_differing_activation"] = int(other.sum().item())
    if stats["rows_with_differing_activation"]:
        stats["oracle_rows"] = oracle_check(other)
    guard = ~valid_rows
    stats["guard_rows_y"] = int(guard.sum().item())
    if stats["guard_rows_y"]:
        guard_check(guard)
    return stats


def _valid_rows(idx: torch.Tensor, table_rows: int) -> torch.Tensor:
    """Row mask of the indices inside ``[0, table_rows)`` (the complement are the guard rows)."""
    return (idx >= 0) & (idx < table_rows)


def _assert_guard_rows_zero_quantized(
    y_q: torch.Tensor,
    y_sf: torch.Tensor,
    guard: torch.Tensor,
    zero_q: torch.Tensor,
    zero_sf: torch.Tensor,
    what: str,
) -> None:
    """The kernel's quantized y on the guard rows is bit-exact to FlashInfer's quantizer of zeros."""
    assert torch.equal(y_q[guard].view(torch.uint8), zero_q.view(torch.uint8)), (
        f"{what}: guard rows' quantized y must be the quantizer's zero codes"
    )
    assert torch.equal(y_sf[guard], zero_sf), (
        f"{what}: guard rows' y scale bytes must be the quantizer's zero scales"
    )


def _norm_args(model, x, idx):
    return (
        x,
        model["x_norm_weight"],
        model["adaln_scale"],
        model["adaln_shift"],
        idx,
    )


def run_bf16_case(
    rows: int, model, device, alias_out: bool = False
) -> Dict[str, float]:
    x, idx, residual = make_inputs(rows, device, table_rows=_table_rows(model))
    residual_ref = residual.clone() if alias_out else residual
    # FlashInfer's own FC1 operator provides the chain's y.
    y_fi = minimax_h3_fc1_swiglu(*_norm_args(model, x, idx), model["fc1_weight"])
    workspace_a = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.bfloat16, device=device
    )
    workspace_y = torch.empty(
        (rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device
    )
    out = minimax_h3_mlp(
        *_norm_args(model, x, idx),
        model["fc1_weight"],
        model["fc2_weight"],
        model["gate"],
        residual,
        out=residual if alias_out else None,
        workspace_a=workspace_a,
        workspace_y=workspace_y,
    )
    torch.cuda.synchronize()
    if alias_out:
        assert out.data_ptr() == residual.data_ptr()
    what = f"bf16 M={rows}" + (" alias" if alias_out else "")
    y_stats = assert_within_budget(workspace_y, y_fi, f"{what} y stage", ATOL, Y_RTOL)
    # Out stage on the operator's own y (the y stage above attributes FC1; the FC2 + epilogue
    # stage is judged on the operand it actually consumed, as the Cake contract does).
    ref = _reference_from_operands(
        workspace_y, model["fc2_weight"], model["gate"], idx, residual_ref
    )
    stats = assert_out_within_rule(out, ref, residual_ref, what)
    stats["y_violations"] = y_stats["violations"]
    stats["guard_rows"] = assert_guard_rows(
        out, residual_ref, idx, _table_rows(model), what
    )
    return stats


def run_mxfp8_case(rows: int, model, prepared, device) -> Dict[str, float]:
    x, idx, residual = make_inputs(rows, device, table_rows=_table_rows(model))
    w1_q, w1_tiles, w2_q, w2_tiles, w2_deq, w1_deq = prepared
    # FlashInfer's own FC1 operator: its y and its quantized a (same norm arithmetic as the MLP's).
    a_fi_q = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    a_fi_sf = torch.zeros(
        (mxfp8_a_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    y_fi = minimax_h3_fc1_swiglu_mxfp8(
        *_norm_args(model, x, idx),
        w1_q,
        w1_tiles,
        workspace_q=a_fi_q,
        workspace_sf=a_fi_sf,
    )
    y_fi_q, y_fi_sf = flashinfer_mxfp8_quantize(y_fi, MXFP8_Y_SF_COLS)
    workspace_a_q = torch.empty(
        (rows, MINIMAX_H3_HIDDEN), dtype=torch.float8_e4m3fn, device=device
    )
    workspace_a_sf = torch.zeros(
        (mxfp8_a_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    workspace_y_q = torch.empty(
        (rows, MINIMAX_H3_FFN), dtype=torch.float8_e4m3fn, device=device
    )
    workspace_y_sf = torch.zeros(
        (mxfp8_y_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    out = minimax_h3_mlp_mxfp8(
        *_norm_args(model, x, idx),
        w1_q,
        w1_tiles,
        w2_q,
        w2_tiles,
        model["gate"],
        residual,
        workspace_a_q=workspace_a_q,
        workspace_a_sf=workspace_a_sf,
        workspace_y_q=workspace_y_q,
        workspace_y_sf=workspace_y_sf,
    )
    torch.cuda.synchronize()
    what = f"mxfp8 M={rows}"
    valid_rows = _valid_rows(idx, _table_rows(model))
    a_same = _bf16_activation_from_operator(model, x, idx, residual, device)
    assert not bool(a_same[~valid_rows].view(torch.int16).any()), (
        f"{what}: the BF16 operator's guard-row activation must be +0.0"
    )
    a_ref_q, a_ref_sf = flashinfer_mxfp8_quantize(a_same, MXFP8_A_SF_COLS)
    a_sf = kernel_sf_linear(workspace_a_sf, rows, MXFP8_A_SF_COLS, MXFP8_A_SF_K_TILES)
    same_rows = assert_a_stage(
        workspace_a_q,
        a_sf,
        a_ref_q,
        a_ref_sf,
        a_fi_q,
        kernel_sf_linear(a_fi_sf, rows, MXFP8_A_SF_COLS, MXFP8_A_SF_K_TILES),
        valid_rows,
        what,
    )
    y_sf = kernel_sf_linear(workspace_y_sf, rows, MXFP8_Y_SF_COLS, MXFP8_Y_SF_K_TILES)

    def oracle_check(other: torch.Tensor) -> Dict[str, int]:
        y_ref = _fc1_reference_from_operands(
            mxfp8_dequantize(workspace_a_q[other], a_sf[other]), w1_deq
        )
        ref_q, ref_sf = flashinfer_mxfp8_quantize(y_ref, MXFP8_Y_SF_COLS)
        return assert_quantized_activation_within_budget(
            workspace_y_q[other],
            y_sf[other],
            ref_q,
            ref_sf,
            f"{what} y stage (rows with a differing activation, vs the FC1 oracle on a_q)",
        )

    def guard_check(guard: torch.Tensor) -> None:
        zero_q, zero_sf = flashinfer_mxfp8_quantize(
            torch.zeros_like(y_fi[guard]), MXFP8_Y_SF_COLS
        )
        _assert_guard_rows_zero_quantized(
            workspace_y_q, y_sf, guard, zero_q, zero_sf, f"{what} y stage"
        )

    y_stats = assert_y_stage(
        workspace_y_q,
        y_sf,
        y_fi_q,
        y_fi_sf,
        same_rows,
        valid_rows,
        what,
        oracle_check,
        guard_check,
    )
    ref = _reference_from_operands(
        mxfp8_dequantize(workspace_y_q, y_sf), w2_deq, model["gate"], idx, residual
    )
    stats = assert_out_within_rule(out, ref, residual, what)
    stats.update(y_stats)
    stats["guard_rows"] = assert_guard_rows(
        out, residual, idx, _table_rows(model), what
    )
    return stats


def run_nvfp4_case(rows: int, model, prepared, device) -> Dict[str, float]:
    x, idx, residual = make_inputs(rows, device, table_rows=_table_rows(model))
    w1_q, w1_tiles, g_w1, w2_q, w2_tiles, w2_scaled, g_w2, w1_scaled = prepared
    # Static activation global scales calibrated from this shape's reference a and the FC1
    # operator's y (outside the operator, as the engine would).
    a_ref = reference_modulated(*_norm_args(model, x, idx))
    g_a = minimax_h3_nvfp4_global_scale(a_ref)
    alpha1 = minimax_h3_nvfp4_alpha(g_a, g_w1)
    a_fi_q = torch.empty((rows, NVFP4_A_PACKED_COLS), dtype=torch.uint8, device=device)
    a_fi_sf = torch.zeros(
        (nvfp4_a_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    y_fi = minimax_h3_fc1_swiglu_nvfp4(
        *_norm_args(model, x, idx),
        g_a,
        w1_q,
        w1_tiles,
        alpha1,
        workspace_q=a_fi_q,
        workspace_sf=a_fi_sf,
    )
    g_y = minimax_h3_nvfp4_global_scale(y_fi)
    alpha2 = minimax_h3_nvfp4_alpha(g_y, g_w2)
    y_fi_q, y_fi_sf = flashinfer_nvfp4_quantize(
        y_fi, g_y, NVFP4_Y_PACKED_COLS, NVFP4_Y_SF_COLS
    )
    workspace_a_q = torch.empty(
        (rows, NVFP4_A_PACKED_COLS), dtype=torch.uint8, device=device
    )
    workspace_a_sf = torch.zeros(
        (nvfp4_a_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    workspace_y_q = torch.empty(
        (rows, NVFP4_Y_PACKED_COLS), dtype=torch.uint8, device=device
    )
    workspace_y_sf = torch.zeros(
        (nvfp4_y_scale_workspace_bytes(rows),), dtype=torch.uint8, device=device
    )
    out = minimax_h3_mlp_nvfp4(
        *_norm_args(model, x, idx),
        g_a,
        w1_q,
        w1_tiles,
        alpha1,
        g_y,
        w2_q,
        w2_tiles,
        alpha2,
        model["gate"],
        residual,
        workspace_a_q=workspace_a_q,
        workspace_a_sf=workspace_a_sf,
        workspace_y_q=workspace_y_q,
        workspace_y_sf=workspace_y_sf,
    )
    torch.cuda.synchronize()
    what = f"nvfp4 M={rows}"
    valid_rows = _valid_rows(idx, _table_rows(model))
    a_same = _bf16_activation_from_operator(model, x, idx, residual, device)
    assert not bool(a_same[~valid_rows].view(torch.int16).any()), (
        f"{what}: the BF16 operator's guard-row activation must be +0.0"
    )
    a_ref_q, a_ref_sf = flashinfer_nvfp4_quantize(
        a_same, g_a, NVFP4_A_PACKED_COLS, NVFP4_A_SF_COLS
    )
    a_sf = kernel_sf_linear(workspace_a_sf, rows, NVFP4_A_SF_COLS, NVFP4_A_SF_K_TILES)
    same_rows = assert_a_stage(
        workspace_a_q,
        a_sf,
        a_ref_q,
        a_ref_sf,
        a_fi_q,
        kernel_sf_linear(a_fi_sf, rows, NVFP4_A_SF_COLS, NVFP4_A_SF_K_TILES),
        valid_rows,
        what,
    )
    y_sf = kernel_sf_linear(workspace_y_sf, rows, NVFP4_Y_SF_COLS, NVFP4_Y_SF_K_TILES)

    def oracle_check(other: torch.Tensor) -> Dict[str, int]:
        y_ref = _fc1_reference_from_operands(
            nvfp4_dequantize_scaled(workspace_a_q[other], a_sf[other]),
            w1_scaled,
            alpha=alpha1.item(),
        )
        ref_q, ref_sf = flashinfer_nvfp4_quantize(
            y_ref, g_y, NVFP4_Y_PACKED_COLS, NVFP4_Y_SF_COLS
        )
        return assert_quantized_activation_within_budget(
            workspace_y_q[other],
            y_sf[other],
            ref_q,
            ref_sf,
            f"{what} y stage (rows with a differing activation, vs the FC1 oracle on a_q)",
        )

    def guard_check(guard: torch.Tensor) -> None:
        zero_q, zero_sf = flashinfer_nvfp4_quantize(
            torch.zeros_like(y_fi[guard]), g_y, NVFP4_Y_PACKED_COLS, NVFP4_Y_SF_COLS
        )
        _assert_guard_rows_zero_quantized(
            workspace_y_q, y_sf, guard, zero_q, zero_sf, f"{what} y stage"
        )

    y_stats = assert_y_stage(
        workspace_y_q,
        y_sf,
        y_fi_q,
        y_fi_sf,
        same_rows,
        valid_rows,
        what,
        oracle_check,
        guard_check,
    )
    ref = _reference_from_operands(
        nvfp4_dequantize_scaled(workspace_y_q, y_sf),
        w2_scaled,
        model["gate"],
        idx,
        residual,
        alpha=alpha2.item(),
    )
    stats = assert_out_within_rule(out, ref, residual, what)
    stats.update(y_stats)
    stats["guard_rows"] = assert_guard_rows(
        out, residual, idx, _table_rows(model), what
    )
    return stats


def prepare_mxfp8(model):
    """Prepared weights plus the FP32 dequantization of the FC2 weight (from FlashInfer's own
    quantization, cross-checked against the prepared tiles)."""
    w1_q, w1_tiles = prepare_minimax_h3_fc1_weight_mxfp8(model["fc1_weight"])
    w2_q, w2_tiles = prepare_minimax_h3_fc2_weight_mxfp8(model["fc2_weight"])
    fi_q, fi_sf = flashinfer_mxfp8_quantize(model["fc2_weight"], MXFP8_Y_SF_COLS)
    assert torch.equal(w2_q.view(torch.uint8), fi_q.view(torch.uint8))
    assert torch.equal(w2_tiles, _weight_scale_tiles(fi_sf))
    assert w2_tiles.numel() == MXFP8_FC2_SCALE_TILE_BYTES
    w1_fi_q, w1_fi_sf = flashinfer_mxfp8_quantize(model["fc1_weight"], MXFP8_A_SF_COLS)
    return (
        w1_q,
        w1_tiles,
        w2_q,
        w2_tiles,
        mxfp8_dequantize(fi_q, fi_sf),
        mxfp8_dequantize(w1_fi_q, w1_fi_sf),
    )


def prepare_nvfp4(model):
    g_w1 = minimax_h3_nvfp4_global_scale(model["fc1_weight"])
    w1_q, w1_tiles = prepare_minimax_h3_fc1_weight_nvfp4(model["fc1_weight"], g_w1)
    g_w2 = minimax_h3_nvfp4_global_scale(model["fc2_weight"])
    w2_q, w2_tiles = prepare_minimax_h3_fc2_weight_nvfp4(model["fc2_weight"], g_w2)
    fi_q, fi_sf = flashinfer_nvfp4_quantize(
        model["fc2_weight"], g_w2, NVFP4_Y_PACKED_COLS, NVFP4_Y_SF_COLS
    )
    assert torch.equal(w2_q, fi_q)
    assert torch.equal(w2_tiles, _weight_scale_tiles(fi_sf))
    assert w2_tiles.numel() == NVFP4_FC2_SCALE_TILE_BYTES
    w1_fi_q, w1_fi_sf = flashinfer_nvfp4_quantize(
        model["fc1_weight"], g_w1, NVFP4_A_PACKED_COLS, NVFP4_A_SF_COLS
    )
    return (
        w1_q,
        w1_tiles,
        g_w1,
        w2_q,
        w2_tiles,
        nvfp4_dequantize_scaled(fi_q, fi_sf),
        g_w2,
        nvfp4_dequantize_scaled(w1_fi_q, w1_fi_sf),
    )


# --------------------------------------------------------------------------------------------
# pytest entry points
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def model(device):
    return make_model(device)


@pytest.fixture(scope="module")
def prepared_mxfp8(model):
    return prepare_mxfp8(model)


@pytest.fixture(scope="module")
def prepared_nvfp4(model):
    return prepare_nvfp4(model)


def test_minimax_h3_fc2_weight_scale_tile_layout():
    """CPU check of the combined 256-row FC2 scale-tile order (K = 14336) against its byte-offset
    formula, for the MXFP8 (112 K sets) and NVFP4 (224 K sets) column counts."""
    gen = torch.Generator().manual_seed(0)
    for cols, expected_bytes in (
        (MXFP8_Y_SF_COLS, MXFP8_FC2_SCALE_TILE_BYTES),
        (NVFP4_Y_SF_COLS, NVFP4_FC2_SCALE_TILE_BYTES),
    ):
        k_sets = cols // 4
        sf = torch.arange(MINIMAX_H3_HIDDEN * cols, dtype=torch.int64) % 251
        sf = sf.to(torch.uint8).reshape(MINIMAX_H3_HIDDEN, cols)
        tiles = _weight_scale_tiles(sf)
        assert (
            tiles.numel()
            == expected_bytes
            == (MINIMAX_H3_HIDDEN // 256) * k_sets * 1024
        )
        for _ in range(256):
            n = int(torch.randint(0, MINIMAX_H3_HIDDEN, (1,), generator=gen))
            c = int(torch.randint(0, cols, (1,), generator=gen))
            n_tile, within = divmod(n, 256)
            half, r = divmod(within, 128)
            k_set, kk = divmod(c, 4)
            offset = (
                ((n_tile * k_sets + k_set) * 2 + half) * 512
                + (r % 32) * 16
                + (r // 32) * 4
                + kk
            )
            assert int(tiles[offset]) == int(sf[n, c])


def test_minimax_h3_mlp_workspace_sizes():
    """CPU check of the scale-workspace sizing (row tiles padded to the even count the paired
    GEMMs consume)."""
    for rows, m_tiles in ((1, 2), (128, 2), (129, 2), (256, 2), (257, 4), (4824, 38)):
        assert (
            mxfp8_a_scale_workspace_bytes(rows)
            == m_tiles * (MXFP8_A_SF_COLS // 4) * 512
        )
        assert (
            nvfp4_a_scale_workspace_bytes(rows)
            == m_tiles * (NVFP4_A_SF_COLS // 4) * 512
        )
        assert mxfp8_y_scale_workspace_bytes(rows) == m_tiles * MXFP8_Y_SF_K_TILES * 512
        assert nvfp4_y_scale_workspace_bytes(rows) == m_tiles * NVFP4_Y_SF_K_TILES * 512


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_mlp_fc2_tail_workspace_geometry(device):
    """The per-device FC2 tail split-K workspace follows the SM count and starts zeroed."""
    partial, flags = _fc2_tail_workspace(device)
    clusters = torch.cuda.get_device_properties(device).multi_processor_count // 2
    slots = clusters // 2 + 1
    assert partial.dtype == torch.float32 and partial.numel() == slots * 2 * 128 * 256
    assert flags.dtype == torch.int32 and flags.numel() == slots * 2
    assert partial.device == flags.device == torch.device("cuda", device.index or 0)
    assert int(flags.abs().sum().item()) == 0
    assert _fc2_tail_workspace(device)[0].data_ptr() == partial.data_ptr()


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_mlp_bf16(rows, model, device):
    run_bf16_case(rows, model, device)


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_mlp_mxfp8(rows, model, prepared_mxfp8, device):
    run_mxfp8_case(rows, model, prepared_mxfp8, device)


@requires_blackwell
@pytest.mark.parametrize("rows", M_VALUES)
def test_minimax_h3_mlp_nvfp4(rows, model, prepared_nvfp4, device):
    run_nvfp4_case(rows, model, prepared_nvfp4, device)


@requires_blackwell
@pytest.mark.parametrize("table_rows", ENGINE_TABLE_ROWS)
def test_minimax_h3_mlp_bf16_engine_tables(table_rows, model, device):
    engine = make_engine_model(model, table_rows, device)
    run_bf16_case(257, engine, device)
    # The strided views were passed through, not copied.
    assert engine["gate"].stride(0) == ENGINE_TABLE_CHUNKS * MINIMAX_H3_HIDDEN


@requires_blackwell
def test_minimax_h3_mlp_mxfp8_engine_tables(model, prepared_mxfp8, device):
    run_mxfp8_case(257, make_engine_model(model, 6, device), prepared_mxfp8, device)


@requires_blackwell
def test_minimax_h3_mlp_nvfp4_engine_tables(model, prepared_nvfp4, device):
    run_nvfp4_case(257, make_engine_model(model, 6, device), prepared_nvfp4, device)


@requires_blackwell
@pytest.mark.parametrize("rows", [129, 4824])
def test_minimax_h3_mlp_bf16_out_aliases_residual(rows, model, device):
    """In-place hidden-state update: ``out`` is ``residual``."""
    run_bf16_case(rows, model, device, alias_out=True)


@requires_blackwell
def test_minimax_h3_mlp_fc2_flags_rearmed(model, device):
    """The tail split-K hand-off counters are back at zero after a launch that used them (M = 1:
    21 pair tiles, one partial wave), so CUDA-graph replay and back-to-back launches are safe."""
    run_bf16_case(1, model, device)
    _partial, flags = _fc2_tail_workspace(device)
    torch.cuda.synchronize()
    assert int(flags.abs().sum().item()) == 0


@requires_blackwell
@pytest.mark.parametrize("table_rows", [3, MINIMAX_H3_GATE_ROWS])
def test_minimax_h3_mlp_invalid_index_rows_pass_residual(table_rows, model, device):
    rows = 300
    if table_rows != MINIMAX_H3_GATE_ROWS:
        model = make_engine_model(model, table_rows, device)
    x, idx, residual = make_inputs(rows, device, table_rows=table_rows)
    idx = idx.clone()
    int64 = torch.iinfo(torch.int64)
    bad = [-1, table_rows, table_rows + 1, -(2**40), 2**40, int64.min, int64.max]
    idx[: len(bad)] = torch.tensor(bad, dtype=torch.int64, device=device)
    out = torch.full(
        (rows, MINIMAX_H3_HIDDEN), float("nan"), dtype=torch.bfloat16, device=device
    )
    workspace_y = torch.empty(
        (rows, MINIMAX_H3_FFN), dtype=torch.bfloat16, device=device
    )
    returned = minimax_h3_mlp(
        *_norm_args(model, x, idx),
        model["fc1_weight"],
        model["fc2_weight"],
        model["gate"],
        residual,
        out=out,
        workspace_y=workspace_y,
    )
    torch.cuda.synchronize()
    assert returned.data_ptr() == out.data_ptr()
    assert torch.equal(out[: len(bad)], residual[: len(bad)])
    assert not torch.equal(out[len(bad) :], residual[len(bad) :])
    y_fi = minimax_h3_fc1_swiglu(*_norm_args(model, x, idx), model["fc1_weight"])
    assert_within_budget(
        workspace_y, y_fi, "bf16 invalid-index probe y stage", ATOL, Y_RTOL
    )
    ref = _reference_from_operands(
        workspace_y, model["fc2_weight"], model["gate"], idx, residual
    )
    assert_out_within_rule(out, ref, residual, "bf16 invalid-index probe")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_minimax_h3_mlp_rejects_bad_inputs(model, device):
    x, idx, residual = make_inputs(8, device)

    def call(
        x=x,
        adaln_scale=model["adaln_scale"],
        adaln_shift=model["adaln_shift"],
        index=idx,
        fc1_weight=model["fc1_weight"],
        fc2_weight=model["fc2_weight"],
        gate=model["gate"],
        residual=residual,
        **kwargs,
    ):
        minimax_h3_mlp(
            x,
            model["x_norm_weight"],
            adaln_scale,
            adaln_shift,
            index,
            fc1_weight,
            fc2_weight,
            gate,
            residual,
            **kwargs,
        )

    with pytest.raises(ValueError):
        call(x=x.float())
    with pytest.raises(ValueError):
        call(x=x[:, :64])
    with pytest.raises(ValueError):
        call(fc1_weight=model["fc1_weight"][:64])
    with pytest.raises(ValueError):
        call(fc2_weight=model["fc2_weight"][:, :64])
    with pytest.raises(ValueError):
        call(residual=residual[:4])
    with pytest.raises(ValueError):
        call(out=torch.empty((8, 64), dtype=torch.bfloat16, device=device))
    with pytest.raises(ValueError):
        call(workspace_y=torch.empty((8, 64), dtype=torch.bfloat16, device=device))
    # int32 indices are no longer part of the contract.
    with pytest.raises(ValueError, match="adaln_index"):
        call(index=idx.int())
    with pytest.raises(ValueError, match="adaln_index"):
        call(index=idx[:4])
    # Tables: wrong dtype, mismatched row counts, non-unit last stride, odd row pitch,
    # misaligned base pointer, a gate table that does not share the AdaLN geometry.
    with pytest.raises(ValueError, match="adaln_scale"):
        call(adaln_scale=model["adaln_scale"].half())
    with pytest.raises(ValueError, match="same row count"):
        call(adaln_scale=model["adaln_scale"][:3])
    with pytest.raises(ValueError, match="gate"):
        call(gate=model["gate"][:3])
    with pytest.raises(ValueError, match="gate"):
        call(gate=model["gate"].half())
    with pytest.raises(ValueError, match="unit last stride"):
        call(
            gate=torch.empty(
                (MINIMAX_H3_HIDDEN, 9), dtype=torch.bfloat16, device=device
            ).t()
        )
    with pytest.raises(ValueError, match="row pitch"):
        call(
            adaln_shift=torch.empty(
                (9, MINIMAX_H3_HIDDEN + 4), dtype=torch.bfloat16, device=device
            )[:, :MINIMAX_H3_HIDDEN]
        )
    storage = torch.empty(
        9 * MINIMAX_H3_HIDDEN + 8, dtype=torch.bfloat16, device=device
    )
    misaligned = storage[4 : 4 + 9 * MINIMAX_H3_HIDDEN].view(9, MINIMAX_H3_HIDDEN)
    assert misaligned.data_ptr() % 16 == 8
    with pytest.raises(ValueError, match="16-byte aligned"):
        call(gate=misaligned)
    # An engine gate table with contract AdaLN tables: the row strides disagree.
    engine = make_engine_model(model, 9, device)
    with pytest.raises(ValueError, match="row stride"):
        call(gate=engine["gate"])
    # A complete engine-layout operand set passes the validation layer (no launch here).
    from flashinfer.diffusion_ops.minimax_h3_mlp import _check_operands

    _check_operands(
        x,
        model["x_norm_weight"],
        engine["adaln_scale"],
        engine["adaln_shift"],
        idx,
        engine["gate"],
        residual,
        1e-6,
    )
