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

Torch-only reference of the Cake StepFun fused MoE
(``flashinfer.experimental.cake_stepfun_moe``): the case encodings, the
per-family tolerances and comparators, the host routing reference, the kernel
weight / scale layouts, the per-stage fp32 references and the error-table row
builder shared by ``tests/experimental/test_cake_stepfun_moe.py`` (backend
suite, package only), ``tests/experimental/test_cake_stepfun_moe_api.py``
(public route, Cake vs native) and ``benchmarks/bench_swiglu_stepfun.py``.

This module imports ``torch`` and the standard library only, so the backend
suite stays vendorable (nothing of ``flashinfer`` is named here).

Conventions (one place for both suites):

* ``TOL`` is keyed by the kernel family: bf16 ``1e-2 / 1e-2``; fp8 and mxfp8
  ``0.1 / 0.1`` with the e4m3 one-ulp block rule on e4m3-valued tensors; nvfp4
  (and the per-token bf16-output variant) ``1.0 / 0.1`` after dequantization,
  with the e2m1 one-code block rule on e2m1-valued tensors.  Routing tables
  are compared bitwise (permutation maps canonically above 16 tokens).
  ``max_rel`` is taken over the elements with ``|ref| >= atol``.
* Limit modes follow the previous test's ``_limits`` mechanics: ``mixed`` is
  7 / 16 by expert parity, every other mode is ``float(mode)`` for all 64
  experts -- ``inf`` included.  ``float("inf")`` flows as data into the kernel
  view and the reference; the two validated public sites (``SwiGLUStep(limit=)``
  and ``CakeStepFunConfig.prepare_weights(step_limits=)``) take ``FLT_MAX``,
  the Cake "no clamp" value.
* Routing regimes ``uniform`` / ``ragged`` / ``randperm`` are the previous
  test's; ``empty`` masks experts 13 and 41 out of the ragged logits so they
  receive no token.
* Routing tables built here (``reference_routing``, ``routing_from_counts``)
  carry the backend's extents for the activation count ``T`` the stage entries
  derive their geometry from: ``max_num_ctas(T, top_k, E, tile)`` CTA entries
  (the live CTAs first, ``CTA_TABLE_PAD`` after them) and ``max_padded + 1``
  permutation slots (``PERMUTED_IDX_PAD`` outside the live rows), while
  ``num_non_exiting_ctas`` / ``total_num_padded_tokens`` stay the live values.
* The whole graph is gated hop by hop on the kernel's own upstream rows
  (:func:`graph_hops`): the FC1 rows, the requantized rows, the FC2 rows and
  the output each against the reference of that stage fed with the kernel's
  rows, at the unchanged ``TOL[family]``; the end-to-end error against
  :func:`graph_reference` is recorded, never asserted (kernel and reference
  round every stage independently, and finalize carries a flipped row ulp
  into outputs that are much smaller).
* :func:`requant_reference` mirrors the requant kernel's fp32 arithmetic
  (``SFValue = (1 / token_scale) * (vec_max * fp32(1 / 6))``, ``outputScale =
  1 / (e4m3(SFValue) * token_scale)``) and marks the blocks whose ``SFValue``
  lies on an e4m3 rounding edge (``REQUANT_SF_EDGE``), where the kernel's
  ``rcp.approx`` reciprocals may round the scale to the neighbouring code.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import torch

# --------------------------------------------------------------------------- geometry and encodings

HIDDEN_SIZE = 4096
INTERMEDIATE_SIZE = 1536
NUM_EXPERTS = 64
TOP_K = 8
TOKENS = (8, 64, 512, 2048)
HIDDEN_STATE_AMPLITUDE = 12.0
FAMILIES = ("nvfp4", "nvfp4_bf16tok", "bf16", "fp8", "mxfp8")
# public precision name -> kernel family of the non-per-token route
PRECISION_FAMILY = {"nvfp4": "nvfp4", "bf16": "bf16", "fp8": "fp8", "mxfp8": "mxfp8"}
LIMIT_MODES = ("7", "16", "mixed", "0.5", "inf")
ROUTING_REGIMES = ("uniform", "ragged", "randperm", "empty")
EMPTY_EXPERTS = (13, 41)
ROUTING_SEED = 20261004
INPUT_SETS = (0, 1)
FLT_MAX = float(torch.finfo(torch.float32).max)
F32_MIN_NORMAL = float(torch.finfo(torch.float32).tiny)
DYN_BLOCK_MAX_TOKENS = 16
E4M3_MAX = 448.0
E2M1_MAX = 6.0
NVFP4_SF_VEC = 16
MXFP8_SF_VEC = 32
GLOBAL_SCALE_INV = 1.0 / (448.0 * 6.0)
# The requant kernel's ``rcp.approx.ftz(6)`` stand-in: ``1 / E2M1_MAX`` rounded to fp32.
F32_SIXTH = float(torch.tensor(1.0 / E2M1_MAX, dtype=torch.float32))
# Relative half-width of the e4m3 rounding-midpoint band of ``requant_reference``: the kernel's
# ``SFValue`` differs from the reference's by at most five fp32 ulps (two ``rcp.approx.ftz`` of
# <= 1 ulp against IEEE reciprocals, two product roundings), below ``2**-20`` relative.
REQUANT_SF_EDGE = 2.0**-20
FC1_OUTPUT_ROWS_PER_CTA = 64
FC2_OUTPUT_ROWS_PER_CTA = 128
# Per-tensor FP8 calibration of the FC1 output (the previous test's harness: StepFun is bounded
# by L^2 <= 256 for the finite modes); the unbounded ``inf`` mode keeps 4096 physical units.
FP8_INTERMEDIATE_SCALE = {"finite": E4M3_MAX / 256.0, "inf": E4M3_MAX / 4096.0}
# Stage-level routing fixtures (per-expert token counts on the first experts, zeros elsewhere).
COUNTS_E2E_FC2 = (3, 0, 65, 132)
COUNTS_E2E_FC1 = (8, 5, 17, 0)
COUNTS_E2E_FC1_MXFP8 = (8, 5, 0, 11)
# Whole-graph cells of the backend suite: (limit mode, routing regime).
BACKEND_GRAPH_CELLS = (
    ("mixed", "ragged"),
    ("0.5", "ragged"),
    ("inf", "empty"),
    ("7", "uniform"),
    ("16", "randperm"),
)
# Whole-graph cells of the api suite (Cake vs native): every limit mode on the ragged and the
# empty regime, plus the two balanced regimes once; 12 cells per (precision, T).
API_GRAPH_CELLS = tuple(
    (mode, regime) for mode in LIMIT_MODES for regime in ("ragged", "empty")
) + (("7", "uniform"), ("16", "randperm"))
# (T, limit mode, regime, input set) cells of the unfinalized (permuted FC2 output) comparison.
API_FC2_PERMUTED_CELLS = (
    (8, "mixed", "ragged", 0),
    (512, "mixed", "ragged", 0),
    (8, "0.5", "ragged", 1),
    (512, "0.5", "ragged", 1),
    (512, "mixed", "empty", 0),
)
API_REQUANT_CELLS = ((8, "ragged"), (512, "ragged"), (512, "empty"))
# Keyword subsets of a case view consumed by the FC1 / FC2 stage entries.
FC1_VIEW_KEYS = (
    "hidden_states",
    "hidden_states_scale",
    "gemm1_weights",
    "gemm1_weights_scale",
    "output1_scale_scalar",
    "output1_scale_gate_scalar",
    "gemm1_clamp_limit",
    "per_token_scale",
)
FC2_VIEW_KEYS = ("gemm2_weights", "gemm2_weights_scale", "output2_scale_scalar")

TOL: dict[str, tuple[float, float]] = {
    "bf16": (1e-2, 1e-2),
    "fp8": (0.1, 0.1),
    "mxfp8": (0.1, 0.1),
    "nvfp4": (1.0, 0.1),
    "nvfp4_bf16tok": (1.0, 0.1),
}
BF16_BAND = (1e-2, 1e-2)
ERROR_TABLE_SCHEMA = "cake_stepfun_error_table/1"
ERROR_TABLE_FILENAME = "cake_stepfun_error_table.{arch}.json"
ERROR_ROW_REQUIRED_KEYS = (
    "stage",
    "family",
    "T",
    "limit_mode",
    "regime",
    "input_set",
    "arch",
    "cake_max_abs",
    "cake_max_rel",
    "native_max_abs",
    "native_max_rel",
    "atol",
    "rtol",
    "pass",
)
ERROR_STAGES = ("routing", "fc1", "requant", "fc2_permuted", "finalize", "output")


def limit_values(mode: str) -> list[float]:
    """One logical limit per routed expert; ``inf`` is ``float("inf")`` (data, never validated)."""
    if mode == "mixed":
        return [7.0 if expert % 2 == 0 else 16.0 for expert in range(NUM_EXPERTS)]
    if mode not in LIMIT_MODES:
        raise KeyError(mode)
    return [float(mode)] * NUM_EXPERTS


def limit_scalar(mode: str) -> float:
    """The finite nominal limit of the public route (``SwiGLUStep(limit=...)``): ``FLT_MAX`` for ``inf``."""
    value = limit_values(mode)[0]
    return FLT_MAX if math.isinf(value) else value


def limits_tensor(mode: str, device) -> torch.Tensor:
    return torch.tensor(limit_values(mode), dtype=torch.float32, device=device)


def public_step_limits(mode: str, device) -> torch.Tensor:
    """``step_limits`` of ``CakeStepFunConfig.prepare_weights``: finite and positive (``FLT_MAX`` for ``inf``)."""
    values = [FLT_MAX if math.isinf(v) else v for v in limit_values(mode)]
    return torch.tensor(values, dtype=torch.float32, device=device)


def routing_logits(
    num_tokens: int, regime: str, seed: int = ROUTING_SEED, device="cuda"
) -> torch.Tensor:
    """bf16 ``[T, 64]`` routing logits of a load regime.

    ``uniform``: i.i.d. normal logits.  ``ragged``: token ``t`` prefers the experts
    around ``(t * E / T)^2 / E`` (expert 0 heavy, high experts sparse).  ``randperm``:
    the ragged rows in a random token order.  ``empty``: the ragged logits with
    experts 13 and 41 set to the bf16 minimum, so top-k never selects them.
    """
    generator = torch.Generator(device=device).manual_seed(seed)
    base = torch.randn(num_tokens, NUM_EXPERTS, device=device, generator=generator)
    if regime == "uniform":
        return base.to(torch.bfloat16)
    centers = (
        torch.arange(num_tokens, device=device, dtype=torch.float32)
        * NUM_EXPERTS
        / num_tokens
    )
    centers = (centers * centers / NUM_EXPERTS).floor()
    experts = torch.arange(NUM_EXPERTS, device=device, dtype=torch.float32)
    logits = base - 0.25 * (experts[None, :] - centers[:, None]).abs()
    if regime == "randperm":
        logits = logits[torch.randperm(num_tokens, device=device, generator=generator)]
    elif regime == "empty":
        logits = logits.to(torch.bfloat16)
        logits[:, list(EMPTY_EXPERTS)] = torch.finfo(torch.bfloat16).min
        return logits
    elif regime != "ragged":
        raise KeyError(regime)
    return logits.to(torch.bfloat16)


# --------------------------------------------------------------------------- host plan formulas


def max_num_ctas(num_tokens: int, top_k: int, num_experts: int, tile: int) -> int:
    """trtllm-gen ``Routing::getMaxNumCtasInBatchDim``."""
    remaining = num_tokens * top_k
    filled = min(num_experts, remaining)
    ctas = filled
    remaining -= filled
    if remaining > 0:
        ctas += remaining // tile
    return ctas


def max_padded_tokens(num_tokens: int, top_k: int, num_experts: int, tile: int) -> int:
    return max_num_ctas(num_tokens, top_k, num_experts, tile) * tile


def gemm2_output_rows(max_padded: int, hidden_size_output: int) -> int:
    """Row count of the unfinalized ``gemm2_output``: native ``maybeGetMinTokenCount(n, H, 16)``."""
    return max(max_padded, -(-(128 * 1024 * 8) // (hidden_size_output * 16)))


def route_tile(
    num_tokens: int, top_k: int = TOP_K, num_experts: int = NUM_EXPERTS
) -> int:
    """FlashInfer's centre tile for ``num_tokens``: the first tile covering ``T * top_k / E``."""
    average = num_tokens * top_k / num_experts
    for tile in (8, 16, 32, 64, 128, 256):
        if tile >= average:
            return tile
    return 256


def finalize_variant(hidden_size: int, num_tokens: int) -> str:
    """The native finalize dispatch: the scalar kernel below 1184 CTAs, the vector kernel otherwise."""
    blocks = -(-hidden_size // 256) * min(8192, num_tokens)
    return "scalar" if blocks < 1184 else "vector"


def compute_log2(value: int) -> int:
    if value <= 0 or value & (value - 1):
        raise ValueError(
            f"tile_tokens_dim must be a positive power of two, got {value}"
        )
    return value.bit_length() - 1


# --------------------------------------------------------------------------- quantization primitives

_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_TIES = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)


def e4m3_round(x: torch.Tensor) -> torch.Tensor:
    """fp32 -> fp32 on the e4m3 grid (round to nearest even, saturating at 448)."""
    return x.float().clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).float()


def e4m3_rounding_edge(
    value: torch.Tensor, band: float = REQUANT_SF_EDGE
) -> torch.Tensor:
    """Mask of the positive fp32 ``value`` elements within ``band`` (relative) of a midpoint between two
    adjacent e4m3 values -- where an arithmetic twin that differs by a few fp32 ulps rounds to the
    neighbouring e4m3 code.  Exactly representable values (and 0) are never on an edge."""
    v = value.float()
    r = e4m3_round(v)
    code = r.to(torch.float8_e4m3fn).view(torch.uint8).int()
    up = torch.where(code < 126, code + 1, code).to(torch.uint8)
    down = torch.where(code > 0, code - 1, code).to(torch.uint8)
    up = up.view(torch.float8_e4m3fn).float()
    down = down.view(torch.float8_e4m3fn).float()
    near_up = (up != r) & ((v - (r + up) / 2).abs() <= band * v)
    near_down = (down != r) & ((v - (r + down) / 2).abs() <= band * v)
    return (v > 0) & (near_up | near_down)


def e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Signed e2m1 codes (``uint8`` 0..15, bit 3 = sign) of ``x`` by round-to-nearest-even, saturating at 6."""
    mag = x.float().abs()
    ties = torch.tensor(_E2M1_TIES, dtype=torch.float32, device=x.device)
    # number of tie points strictly below the magnitude = the lower candidate code
    lower = torch.bucketize(mag, ties, right=True)
    on_tie = (lower < 7) & (mag == ties[lower.clamp(max=6)])
    # a magnitude exactly on a tie rounds to the even code of the pair
    code = torch.where(on_tie & (lower % 2 == 1), lower + 1, lower).clamp(max=7)
    return (code + 8 * (x < 0).to(code.dtype)).to(torch.uint8)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1_VALUES, dtype=torch.float32, device=codes.device)
    magnitude = table[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -magnitude, magnitude)


def e2m1_pack(codes: torch.Tensor) -> torch.Tensor:
    """``[..., K]`` codes -> ``[..., K / 2]`` bytes, the even element in the low nibble."""
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def e2m1_unpack(packed: torch.Tensor) -> torch.Tensor:
    low, high = packed & 15, packed >> 4
    return torch.stack((low, high), dim=-1).reshape(
        *packed.shape[:-1], packed.shape[-1] * 2
    )


def ue8m0_codes(amax: torch.Tensor) -> torch.Tensor:
    """UE8M0 scale codes of per-block magnitudes (the trtllm-gen MXFP8 epilogue rule: round the exponent up)."""
    amax = torch.where(amax >= F32_MIN_NORMAL, amax, torch.zeros_like(amax))
    numerator = torch.where(amax == 0.0, torch.full_like(amax, E4M3_MAX), amax)
    sf = (numerator / E4M3_MAX).contiguous()
    bits = sf.view(torch.int32)
    code = ((bits >> 23) & 255) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    return code.clamp(0, 254).to(torch.uint8)


@dataclass(frozen=True)
class Nvfp4:
    """An nvfp4-quantized ``[rows, K]`` matrix: packed codes, linear e4m3 scale bytes, dequantized values."""

    packed: torch.Tensor  # uint8 [rows, K / 2]
    scale: torch.Tensor  # uint8 (e4m3 bytes) [rows, K / 16]
    dequant: torch.Tensor  # float32 [rows, K]


def nvfp4_quantize(x: torch.Tensor) -> Nvfp4:
    """16-element blocks with an e4m3 scale ``e4m3(amax / 6)`` (global scale 1); codes by round-to-nearest-even.

    ``dequant = code * sf``: the value the block-scaled MMA consumes.
    """
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // NVFP4_SF_VEC, NVFP4_SF_VEC)
    sf = e4m3_round(blocks.abs().amax(dim=-1) / E2M1_MAX)
    scaled = torch.where(
        sf[..., None] > 0, blocks / sf[..., None], torch.zeros_like(blocks)
    )
    codes = e2m1_codes(scaled)
    dequant = (e2m1_values(codes) * sf[..., None]).reshape(rows, k)
    return Nvfp4(
        packed=e2m1_pack(codes.reshape(rows, k)),
        scale=sf.to(torch.float8_e4m3fn).view(torch.uint8),
        dequant=dequant,
    )


def nvfp4_dequantize(packed: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
    """``[rows, K / 2]`` codes with linear ``[rows, K / 16]`` e4m3 bytes -> fp32 ``[rows, K]``."""
    rows = packed.shape[0]
    codes = e2m1_unpack(packed)
    k = codes.shape[-1]
    sf = (
        scale_bytes.reshape(rows, k // NVFP4_SF_VEC)
        .contiguous()
        .view(torch.float8_e4m3fn)
        .float()
    )
    values = e2m1_values(codes).reshape(rows, k // NVFP4_SF_VEC, NVFP4_SF_VEC)
    return (values * sf[..., None]).reshape(rows, k)


@dataclass(frozen=True)
class Mxfp8:
    """An mxfp8-quantized ``[rows, K]`` matrix: e4m3 codes, linear UE8M0 codes, dequantized values."""

    codes: torch.Tensor  # float8_e4m3fn [rows, K]
    scale: torch.Tensor  # uint8 [rows, K / 32]
    dequant: torch.Tensor  # float32 [rows, K]


def mxfp8_quantize(x: torch.Tensor) -> Mxfp8:
    rows, k = x.shape
    blocks = x.float().reshape(rows, k // MXFP8_SF_VEC, MXFP8_SF_VEC)
    code = ue8m0_codes(blocks.abs().amax(dim=-1))
    q = (blocks * torch.exp2(127.0 - code.float())[..., None]).to(torch.float8_e4m3fn)
    dequant = q.float() * torch.exp2(code.float() - 127.0)[..., None]
    return Mxfp8(codes=q.reshape(rows, k), scale=code, dequant=dequant.reshape(rows, k))


def mxfp8_dequantize(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    rows, k = codes.shape
    values = (
        codes.view(torch.float8_e4m3fn)
        .float()
        .reshape(rows, k // MXFP8_SF_VEC, MXFP8_SF_VEC)
    )
    sf = torch.exp2(scale.reshape(rows, k // MXFP8_SF_VEC).float() - 127.0)
    return (values * sf[..., None]).reshape(rows, k)


def fp8_per_tensor_quantize(x: torch.Tensor, quant_scale: float) -> torch.Tensor:
    """e4m3 codes of ``x * quant_scale`` (``quant_scale = 448 / amax`` of the calibration)."""
    return (x.float() * quant_scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)


# --------------------------------------------------------------------------- kernel layouts

_SHUFFLE_BLOCK32 = tuple((n % 8) * 4 + (n % 32) // 8 for n in range(32))


def fc1_weight_row_permutation(m_out: int) -> list[int]:
    """Physical row order of the packed ``[linear; gate]`` FC1 weights (FlashInfer
    ``reorder_rows_for_gated_act_gemm`` + ``shuffle_matrix_a(epilogue_tile_m=128)``): row ``p`` holds
    canonical row ``perm[p]`` (``[0, M_out)`` linear / up, ``[M_out, 2 M_out)`` gate)."""
    if m_out % 64:
        raise ValueError(f"M_out must be a multiple of 64, got {m_out}")
    perm: list[int] = []
    for m_tile in range(m_out // 64):
        for warp in range(4):
            lin = m_tile * 64 + warp * 16
            gate = m_out + lin
            perm.extend(range(lin, lin + 16, 2))
            perm.extend(range(gate, gate + 16, 2))
            perm.extend(range(lin + 1, lin + 16, 2))
            perm.extend(range(gate + 1, gate + 16, 2))
    return perm


def fc2_weight_row_permutation(rows: int) -> list[int]:
    """FlashInfer ``shuffle_matrix_a(epilogue_tile_m=128)`` of the FC2 weight rows: row ``p`` holds
    canonical row ``(p // 32) * 32 + (p % 8) * 4 + (p % 32) // 8``."""
    if rows % 32:
        raise ValueError(f"rows must be a multiple of 32, got {rows}")
    return [(p // 32) * 32 + _SHUFFLE_BLOCK32[p % 32] for p in range(rows)]


def to_block_major_k(weights: torch.Tensor, block: int = 64) -> torch.Tensor:
    """``[E, rows, K]`` -> ``WeightLayout.BlockMajorK`` ``[E, K / block, rows, block]`` (contiguous)."""
    num_experts, rows, k = weights.shape
    if k % block:
        raise ValueError(f"K must be a multiple of {block}, got {k}")
    return (
        weights.view(num_experts, rows, k // block, block)
        .permute(0, 2, 1, 3)
        .contiguous()
    )


def swizzle_sf_128x4(sf: torch.Tensor) -> torch.Tensor:
    """The 128x4 block-scale interleave (FlashInfer ``block_scale_interleave``) of ``[rows, cols]`` bytes
    (``rows % 128 == 0``, ``cols % 4 == 0``): byte ``(r // 128) * (cols / 4 * 512) + (c // 4) * 512 +
    (r % 32) * 16 + ((r % 128) // 32) * 4 + c % 4``; returned flat."""
    rows, cols = sf.shape
    if rows % 128 or cols % 4:
        raise ValueError(
            f"128x4 swizzle needs rows % 128 == 0 and cols % 4 == 0, got {rows} x {cols}"
        )
    return (
        sf.reshape(rows // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .reshape(-1)
    )


def swizzle_sf_8x4(sf: torch.Tensor) -> torch.Tensor:
    """The 8x4 interleave: byte ``(r // 8) * (cols / 4 * 32) + (c // 4) * 32 + (r % 8) * 4 + c % 4``."""
    rows, cols = sf.shape
    if rows % 8 or cols % 4:
        raise ValueError(
            f"8x4 swizzle needs rows % 8 == 0 and cols % 4 == 0, got {rows} x {cols}"
        )
    return (
        sf.reshape(rows // 8, 8, cols // 4, 4)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


def swizzle_sf(sf: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "r8c4":
        return swizzle_sf_8x4(sf)
    if layout == "r128c4":
        return swizzle_sf_128x4(sf)
    if layout == "linear":
        return sf.contiguous().reshape(-1)
    raise KeyError(layout)


def sf_indices(rows: int, cols: int, layout: str, device) -> torch.Tensor:
    """Byte offsets ``[rows, cols]`` of the logical block scales inside a swizzled buffer of ``layout``
    (``"linear"`` / ``"r8c4"`` / ``"r128c4"``; the column count is padded to a multiple of 4 as the
    buffers are)."""
    col_groups = -(-cols // 4)
    row = torch.arange(rows, device=device)[:, None]
    col = torch.arange(cols, device=device)[None, :]
    if layout == "linear":
        return row * cols + col
    if layout == "r8c4":
        return (
            (row // 8) * (col_groups * 32) + (col // 4) * 32 + (row % 8) * 4 + col % 4
        )
    if layout == "r128c4":
        return (
            (row // 128) * (col_groups * 512)
            + (col // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + col % 4
        )
    raise KeyError(layout)


def unswizzle_sf(flat: torch.Tensor, rows: int, cols: int, layout: str) -> torch.Tensor:
    """Logical ``[rows, cols]`` scale bytes out of a swizzled buffer (``flat`` may be longer than needed)."""
    return flat.reshape(-1)[sf_indices(rows, cols, layout, flat.device)]


SF_BLOCK_ROWS = {"linear": 1, "r8c4": 8, "r128c4": 128}


def swizzled_sf_bytes(rows: int, cols: int, layout: str) -> int:
    block_rows = SF_BLOCK_ROWS[layout]
    return -(-rows // block_rows) * block_rows * (-(-cols // 4) * 4)


def interleave_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Per-expert 128x4 interleave of ``[E, rows, cols]`` scale bytes, returned ``[E, rows, cols]`` (the
    FlashInfer view shape of the interleaved bytes; the kernels read it as ``(E * rows / 128, cols / 4, 2, 256)``)."""
    num_experts, rows, cols = scales.shape
    return torch.stack(
        [swizzle_sf_128x4(scales[e]) for e in range(num_experts)]
    ).reshape(num_experts, rows, cols)


# --------------------------------------------------------------------------- StepFun math


def stepfun(up: torch.Tensor, gate: torch.Tensor, limit) -> torch.Tensor:
    """``clamp(up, -L, L) * min(silu(gate), L)`` in fp32; ``limit`` is a scalar or a per-row column tensor."""
    swish = gate * torch.sigmoid(gate)
    cap = torch.as_tensor(limit, dtype=swish.dtype, device=swish.device)
    return up.clamp(min=-limit, max=limit) * torch.minimum(swish, cap)


def stepfun_rows(
    acc: torch.Tensor, limit: float, *, scale_gate: float = 1.0, scale_c: float = 1.0
) -> torch.Tensor:
    """The FC1 epilogue of one expert on raw accumulators ``acc`` ``[n, 2 * M_out]`` (linear half first):
    physical ``up = acc_up * scale_gate``, ``gate = acc_gate * scale_gate``, output ``stepfun(up, gate, L) *
    scale_c / scale_gate`` (the per-expert output scale of the nvfp4 / fp8 families; 1 / 1 for bf16 / mxfp8)."""
    m_out = acc.shape[1] // 2
    up = acc[:, :m_out] * scale_gate
    gate = acc[:, m_out:] * scale_gate
    return stepfun(up, gate, limit) * (scale_c / scale_gate)


# --------------------------------------------------------------------------- routing reference


def _twiddle_keys(logits: torch.Tensor) -> torch.Tensor:
    """Device-order sort keys of the reference top-k: twiddled bits ``<< 16 | (65535 - idx)``."""
    _t, e = logits.shape
    idx = torch.arange(e, device=logits.device, dtype=torch.int64)
    idx_part = (65535 - idx) & 0xFFFF
    if logits.dtype == torch.bfloat16:
        bits = logits.view(torch.int16).to(torch.int64) & 0xFFFF
        tw = torch.where(bits & 0x8000 != 0, (~bits) & 0xFFFF, bits | 0x8000)
    else:
        bits = logits.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
        tw = torch.where(
            bits & 0x80000000 != 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000
        )
    return (tw << 16) | idx_part.unsqueeze(0)


# Values of the table entries the device router never writes: the CTA tables at and beyond
# ``num_non_exiting_ctas`` (those CTAs never act on their entry: surplus CTAs exit at the tile check,
# and a prologue load of the tail is ignored) and the permutation slots outside the live padded rows.  ``-1`` is the router's own slot padding and the ``routing_buffers``
# pre-fill, so the ``+ 1`` slack entry reads ``-1`` on both sides.
CTA_TABLE_PAD = 0
PERMUTED_IDX_PAD = -1


def tables_from_counts(
    counts,
    *,
    tile: int,
    num_tokens: int,
    local_expert_offset: int = 0,
    device="cuda",
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
) -> dict[str, Any]:
    """CTA descriptors of per-expert token counts (zeros allowed) at the backend's table extents.

    ``cta_idx_xy_to_batch_idx`` / ``cta_idx_xy_to_mn_limit`` hold the live CTAs first (absolute row
    limit ``min((offset + cta + 1) * tile, offset * tile + count)``), then :data:`CTA_TABLE_PAD` up
    to ``max_num_ctas(num_tokens, top_k, num_experts, tile)`` entries -- the extent the stage
    entries check (``>= max_num_ctas``) and the CTA count of the GEMM grids; ``num_tokens`` is the
    activation count the backend derives its geometry from.  ``num_non_exiting_ctas`` and
    ``total_num_padded_tokens`` (ints) are the live values; ``_max_num_ctas`` / ``_max_padded``
    the extents."""
    counts = [int(c) for c in counts]
    num_cta = [-(-c // tile) for c in counts]
    offsets, total = [], 0
    for n in num_cta:
        offsets.append(total)
        total += n
    capacity = max_num_ctas(num_tokens, top_k, num_experts, tile)
    if total > capacity:
        raise ValueError(
            f"{total} live CTAs exceed max_num_ctas({num_tokens}, {top_k}, {num_experts}, "
            f"{tile}) = {capacity}"
        )
    batch_idx = [CTA_TABLE_PAD] * capacity
    mn_limit = [CTA_TABLE_PAD] * capacity
    for expert, (count, n, offset) in enumerate(
        zip(counts, num_cta, offsets, strict=True)
    ):
        for cta in range(n):
            batch_idx[offset + cta] = expert - local_expert_offset
            mn_limit[offset + cta] = min(
                (offset + cta + 1) * tile, offset * tile + count
            )
    return {
        "cta_idx_xy_to_batch_idx": torch.tensor(
            batch_idx, dtype=torch.int32, device=device
        ),
        "cta_idx_xy_to_mn_limit": torch.tensor(
            mn_limit, dtype=torch.int32, device=device
        ),
        "num_non_exiting_ctas": total,
        "total_num_padded_tokens": total * tile,
        "_cta_offset": offsets,
        "_max_num_ctas": capacity,
        "_max_padded": capacity * tile,
    }


def reference_routing(
    logits: torch.Tensor,
    *,
    top_k: int = TOP_K,
    tile: int,
    local_expert_offset: int = 0,
    local_num_experts: int = NUM_EXPERTS,
) -> dict[str, Any]:
    """Host re-implementation of the trtllm-gen Renormalize router: exact for the top-k selection and every
    integer table; the bf16 weights follow the device within one bf16 ulp (fast-math ``exp`` / ``div`` have
    no bit-exact host twin).  The permutation is token-major: exact for the single-block kernels (T <= 16),
    the canonical form for the cluster / cooperative kernels (atomic-arrival slot order).  The tables carry
    the router's extents for ``T = logits.shape[0]`` (:func:`tables_from_counts`; ``permuted_idx_to_token_idx``
    holds ``max_padded + 1`` slots, :data:`PERMUTED_IDX_PAD` outside the live rows)."""
    num_tokens, num_experts = logits.shape
    keys = _twiddle_keys(logits)
    order = torch.argsort(keys, dim=1, descending=True)[:, :top_k]
    selected = torch.gather(logits.float(), 1, order)
    m = selected.max(dim=1, keepdim=True).values
    e = torch.exp(selected - m)
    weights = (e / e.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    ids = order.to(torch.int32)
    local = ids - local_expert_offset
    is_local = (local >= 0) & (local < local_num_experts)
    counts = torch.zeros(num_experts, dtype=torch.int64, device=logits.device)
    selected_ids = ids.reshape(-1).long()[is_local.reshape(-1)]
    counts.scatter_add_(0, selected_ids, torch.ones_like(selected_ids))
    tables = tables_from_counts(
        counts.tolist(),
        tile=tile,
        num_tokens=num_tokens,
        local_expert_offset=local_expert_offset,
        device=logits.device,
        top_k=top_k,
        num_experts=num_experts,
    )
    running = [0] * num_experts
    ids_cpu = ids.cpu().tolist()
    local_cpu = is_local.cpu().tolist()
    offsets = tables["_cta_offset"]
    e2p = torch.full((num_tokens, top_k), -1, dtype=torch.int32)
    p2t = torch.full((tables["_max_padded"] + 1,), PERMUTED_IDX_PAD, dtype=torch.int32)
    for t in range(num_tokens):
        for k in range(top_k):
            if local_cpu[t][k]:
                expert = ids_cpu[t][k]
                slot = offsets[expert] * tile + running[expert]
                running[expert] += 1
                e2p[t, k] = slot
                p2t[slot] = t
    return {
        "topk_ids": ids,
        "topk_weights": weights,
        "expert_weights": weights,
        "expert_counts": counts.to(torch.int32),
        "total_num_padded_tokens": tables["total_num_padded_tokens"],
        "num_non_exiting_ctas": tables["num_non_exiting_ctas"],
        "cta_idx_xy_to_batch_idx": tables["cta_idx_xy_to_batch_idx"],
        "cta_idx_xy_to_mn_limit": tables["cta_idx_xy_to_mn_limit"],
        "expanded_idx_to_permuted_idx": e2p.to(logits.device),
        "permuted_idx_to_token_idx": p2t.to(logits.device),
        "tile": tile,
    }


def routing_from_counts(
    counts, *, tile: int, device="cuda", num_tokens: int
) -> dict[str, Any]:
    """Routing tables of a stage-level fixture: expert ``e`` receives ``counts[e]`` distinct tokens (token ids
    ascending through the experts, all below ``num_tokens``, the activation count the backend sees), slots
    token-major with ``-1`` padding; ``expanded_idx_to_permuted_idx`` is the flat slot list of the
    ``sum(counts)`` routed rows (one row per (token, k) pair).  Extents as :func:`reference_routing`:
    ``max_num_ctas(num_tokens, TOP_K, NUM_EXPERTS, tile)`` CTA entries and ``max_padded + 1`` slots."""
    counts = [int(c) for c in counts] + [0] * (NUM_EXPERTS - len(counts))
    tables = tables_from_counts(counts, tile=tile, num_tokens=num_tokens, device=device)
    total_rows = sum(counts)
    if total_rows > num_tokens:
        raise ValueError(f"{total_rows} routed tokens exceed num_tokens {num_tokens}")
    p2t = torch.full((tables["_max_padded"] + 1,), PERMUTED_IDX_PAD, dtype=torch.int32)
    e2p = torch.full((total_rows,), -1, dtype=torch.int32)
    token = 0
    for expert, count in enumerate(counts):
        start = tables["_cta_offset"][expert] * tile
        for row in range(count):
            p2t[start + row] = token
            e2p[token] = start + row
            token += 1
    return {
        "expert_counts": torch.tensor(counts, dtype=torch.int32, device=device),
        "total_num_padded_tokens": tables["total_num_padded_tokens"],
        "num_non_exiting_ctas": tables["num_non_exiting_ctas"],
        "cta_idx_xy_to_batch_idx": tables["cta_idx_xy_to_batch_idx"],
        "cta_idx_xy_to_mn_limit": tables["cta_idx_xy_to_mn_limit"],
        "expanded_idx_to_permuted_idx": e2p.to(device),
        "permuted_idx_to_token_idx": p2t.to(device),
        "num_rows": total_rows,
        "tile": tile,
    }


def routing_buffers(
    num_tokens: int,
    tile: int,
    device,
    *,
    top_k: int = TOP_K,
    num_experts: int = NUM_EXPERTS,
    fill: int = -1,
) -> dict[str, torch.Tensor]:
    """The routing table set the backend's ``prepare_routing`` writes (the routing module's ``TABLE_NAMES``
    extents), pre-filled with ``fill`` so untouched entries are visible."""
    ctas = max_num_ctas(num_tokens, top_k, num_experts, tile)

    def table(*shape):
        return torch.full(shape, fill, dtype=torch.int32, device=device)

    return {
        "topk_packed": table(num_tokens, top_k),
        "topk_weights": torch.zeros(
            num_tokens, top_k, dtype=torch.bfloat16, device=device
        ),
        "expert_count_histogram": table(max(2 * num_experts, 512)),
        "total_num_padded_tokens": table(1),
        "expanded_idx_to_permuted_idx": table(num_tokens, top_k),
        "permuted_idx_to_token_idx": table(ctas * tile + 1),
        "cta_idx_xy_to_batch_idx": table(ctas),
        "cta_idx_xy_to_mn_limit": table(ctas),
        "num_non_exiting_ctas": table(1),
    }


def live_tables(buffers: dict[str, torch.Tensor], *, tile: int) -> dict[str, Any]:
    """The comparable view of written routing buffers: scalars as ints, tables trimmed to the live extents,
    ``expert_weights`` aliased to ``topk_weights``."""
    nnec = int(buffers["num_non_exiting_ctas"].reshape(-1)[0].item())
    padded = nnec * tile
    return {
        "expert_weights": buffers["topk_weights"],
        "topk_weights": buffers["topk_weights"],
        "topk_packed": buffers["topk_packed"],
        "num_non_exiting_ctas": nnec,
        "total_num_padded_tokens": int(
            buffers["total_num_padded_tokens"].reshape(-1)[0].item()
        ),
        "cta_idx_xy_to_batch_idx": buffers["cta_idx_xy_to_batch_idx"].reshape(-1)[
            :nnec
        ],
        "cta_idx_xy_to_mn_limit": buffers["cta_idx_xy_to_mn_limit"].reshape(-1)[:nnec],
        "expanded_idx_to_permuted_idx": buffers["expanded_idx_to_permuted_idx"],
        "permuted_idx_to_token_idx": buffers["permuted_idx_to_token_idx"].reshape(-1)[
            :padded
        ],
        "tile": tile,
    }


def canonical_permutation(
    e2p,
    p2t,
    cta_idx_xy_to_batch_idx,
    *,
    tile: int,
    top_k: int,
    num_non_exiting_ctas: int,
):
    """Order-insensitive view of a permutation whose in-expert slot order is atomic-arrival order: per expert
    the sorted expanded indices it holds, the ``-1`` padding mask of ``p2t`` and the e2p / p2t consistency."""
    e2p = e2p.reshape(-1).cpu().long()
    p2t = p2t.reshape(-1).cpu().long()
    batch = cta_idx_xy_to_batch_idx.reshape(-1).cpu().long()[:num_non_exiting_ctas]
    active = e2p >= 0
    tokens = torch.arange(e2p.numel()) // top_k
    consistent = (
        bool((p2t[e2p[active]] == tokens[active]).all().item())
        if bool(active.any())
        else True
    )
    per_expert: dict[int, list[int]] = {}
    for ex in torch.nonzero(active).flatten().tolist():
        per_expert.setdefault(int(batch[int(e2p[ex]) // tile]), []).append(ex)
    padded = num_non_exiting_ctas * tile
    pad_mask = tuple((p2t[:padded] == -1).tolist())
    return consistent, {k: tuple(sorted(v)) for k, v in per_expert.items()}, pad_mask


DETERMINISTIC_ROUTING_TABLES = (
    "expert_weights",
    "cta_idx_xy_to_batch_idx",
    "cta_idx_xy_to_mn_limit",
    "num_non_exiting_ctas",
    "total_num_padded_tokens",
)


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    return int(value)


def expert_ids_of(
    tables: dict[str, Any], *, num_tokens: int, top_k: int
) -> torch.Tensor:
    """``[T, top_k]`` expert of every (token, k) pair through the permutation and the CTA table (-1 = not local)."""
    e2p = tables["expanded_idx_to_permuted_idx"].reshape(-1).long()
    batch = tables["cta_idx_xy_to_batch_idx"].reshape(-1)
    tile = int(tables["tile"])
    ids = torch.full_like(e2p, -1, dtype=torch.int32)
    live = e2p >= 0
    ids[live] = batch[e2p[live] // tile].to(torch.int32)
    return ids.reshape(num_tokens, top_k)


def assert_routing_tables_match(
    ours: dict,
    theirs: dict,
    *,
    num_tokens: int,
    tile: int,
    top_k: int,
    context: str,
    weights_ulp: Optional[float] = None,
) -> bool:
    """Routing tables bitwise on the deterministic tables; the permutation maps bitwise for ``T <= 16`` and in
    canonical form above (locality mask, padding mask, e2p / p2t consistency, slot uniqueness, (token, k) ->
    expert).  ``weights_ulp`` allows that many bf16 ulps on ``expert_weights`` (the host reference's
    fast-math gap); ``None`` means bitwise.  Returns ``True`` when every table was bitwise equal.  Each dict
    carries ``expert_weights``, the tables of ``DETERMINISTIC_ROUTING_TABLES`` (scalars as ints or
    one-element tensors), ``expanded_idx_to_permuted_idx`` and ``permuted_idx_to_token_idx``."""
    nnec = _as_int(ours["num_non_exiting_ctas"])
    assert nnec == _as_int(theirs["num_non_exiting_ctas"]), (
        f"{context}: num_non_exiting_ctas differs"
    )
    assert _as_int(ours["total_num_padded_tokens"]) == _as_int(
        theirs["total_num_padded_tokens"]
    ), f"{context}: total_num_padded_tokens differs"
    bitwise = True
    w_a, w_b = ours["expert_weights"].reshape(-1), theirs["expert_weights"].reshape(-1)
    if weights_ulp is None:
        assert torch.equal(w_a, w_b), (
            f"{context}: expert_weights differ (ours {w_a[:8].tolist()} vs theirs {w_b[:8].tolist()})"
        )
    else:
        ulp = torch.clamp(w_b.float().abs() * 2.0**-7, min=2.0**-133)
        gap = (w_a.float() - w_b.float()).abs()
        assert bool((gap <= weights_ulp * ulp).all()), (
            f"{context}: expert_weights differ by more than {weights_ulp} bf16 ulp "
            f"(max gap {gap.max().item():.3e})"
        )
        bitwise = bitwise and bool(torch.equal(w_a, w_b))
    for name in ("cta_idx_xy_to_batch_idx", "cta_idx_xy_to_mn_limit"):
        a, b = ours[name].reshape(-1)[:nnec], theirs[name].reshape(-1)[:nnec]
        assert torch.equal(a, b), (
            f"{context}: {name} differs (ours {a[:8].tolist()} vs theirs {b[:8].tolist()})"
        )
    e2p_a = ours["expanded_idx_to_permuted_idx"].reshape(-1)
    e2p_b = theirs["expanded_idx_to_permuted_idx"].reshape(-1)
    padded = nnec * tile
    p2t_a = ours["permuted_idx_to_token_idx"].reshape(-1)[:padded]
    p2t_b = theirs["permuted_idx_to_token_idx"].reshape(-1)[:padded]
    maps_bitwise = bool(torch.equal(e2p_a, e2p_b) and torch.equal(p2t_a, p2t_b))
    if num_tokens <= DYN_BLOCK_MAX_TOKENS:
        assert maps_bitwise, (
            f"{context}: permutation maps differ (T <= {DYN_BLOCK_MAX_TOKENS} is bitwise)"
        )
        return bitwise
    if maps_bitwise:
        return bitwise
    assert torch.equal(e2p_a < 0, e2p_b < 0), f"{context}: locality mask differs"
    assert torch.equal(p2t_a < 0, p2t_b < 0), f"{context}: padding mask differs"
    for label, e2p, p2t in (("ours", e2p_a, p2t_a), ("theirs", e2p_b, p2t_b)):
        valid = (e2p >= 0).nonzero().flatten()
        slots = e2p[valid].long()
        assert torch.equal(p2t[slots], (valid // top_k).to(p2t.dtype)), (
            f"{context}: {label} permutation maps are inconsistent"
        )
        assert slots.unique().numel() == slots.numel(), (
            f"{context}: {label} slots collide"
        )
    experts_a = ours["cta_idx_xy_to_batch_idx"].reshape(-1)[
        e2p_a[e2p_a >= 0].long() // tile
    ]
    experts_b = theirs["cta_idx_xy_to_batch_idx"].reshape(-1)[
        e2p_b[e2p_b >= 0].long() // tile
    ]
    assert torch.equal(experts_a, experts_b), (
        f"{context}: (token, k) -> expert assignment differs"
    )
    return False


def assert_empty_experts(
    tables: dict, *, tile: int, experts=EMPTY_EXPERTS, context: str
) -> None:
    """The ``empty`` regime: the masked experts count zero tokens, no live CTA names them and the padding of
    ``permuted_idx_to_token_idx`` is ``-1``."""
    nnec = _as_int(tables["num_non_exiting_ctas"])
    batch = tables["cta_idx_xy_to_batch_idx"].reshape(-1)[:nnec]
    for expert in experts:
        assert not bool((batch == expert).any()), (
            f"{context}: a live CTA names empty expert {expert}"
        )
    if "expert_counts" in tables:
        counts = tables["expert_counts"].reshape(-1)
        for expert in experts:
            assert int(counts[expert]) == 0, (
                f"{context}: expert {expert} received {int(counts[expert])} tokens"
            )
    p2t = tables["permuted_idx_to_token_idx"].reshape(-1)[: nnec * tile]
    e2p = tables["expanded_idx_to_permuted_idx"].reshape(-1)
    live = torch.zeros(nnec * tile, dtype=torch.bool, device=p2t.device)
    live[e2p[e2p >= 0].long()] = True
    assert bool((p2t[~live] == -1).all()), (
        f"{context}: tail padding of permuted_idx_to_token_idx is not -1"
    )


# --------------------------------------------------------------------------- comparators


def errors(
    actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float
) -> dict[str, Any]:
    """Elementwise error record: ``exceeds = |a - e| > atol + rtol |e|``; ``max_rel`` over ``|e| >= atol``."""
    a, e = actual.float(), expected.float()
    diff = (a - e).abs()
    exceeds = diff > atol + rtol * e.abs()
    floor = e.abs() >= atol
    max_rel = (diff[floor] / e.abs()[floor]).max().item() if bool(floor.any()) else 0.0
    return dict(
        max_abs=float(diff.max().item()) if diff.numel() else 0.0,
        max_rel=float(max_rel),
        n_outside=int(exceeds.sum().item()),
        numel=int(diff.numel()),
        rel_floor=atol,
        exceeds=exceeds,
    )


def e4m3_block_flips(codes_a, codes_e, sf_a, sf_e, block: int) -> torch.Tensor:
    """Mask of the elements that are one e4m3 ulp apart under an equal block scale code and sign (fp32
    accumulation-order flips across a rounding edge); ``sf_* = None`` for the per-tensor fp8 family."""
    codes_a = codes_a.reshape(codes_e.shape).contiguous().view(torch.uint8).int()
    codes_e = codes_e.contiguous().view(torch.uint8).int()
    same_sign = (codes_a >> 7) == (codes_e >> 7)
    one_ulp = ((codes_a & 127) - (codes_e & 127)).abs() == 1
    if sf_a is None:
        return same_sign & one_ulp
    same_sf = (
        (sf_a.reshape(sf_e.shape) == sf_e).repeat_interleave(block, dim=-1)
    ).reshape(codes_e.shape)
    return same_sf & same_sign & one_ulp


def e2m1_block_flips(
    codes_a,
    codes_e,
    sf_a,
    sf_e,
    block: int = NVFP4_SF_VEC,
    *,
    sf_edge: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """The e2m1 analogue: unpacked codes (0..15) one magnitude code apart under the same block scale byte and
    the same sign.  A KERNEL zero code is excused whichever sign bit it carries (its dequantised value is
    zero either way, one e2m1 step from the reference's ``+-0.5``); a reference zero keeps the input's sign
    (:func:`e2m1_codes`), and so does every non-zero kernel code (``cvt.rn.satfinite.e2m1x2`` of the same
    product), so a kernel ``+-0.5`` of the opposite sign next to a reference zero is a sign error, never
    excused.  ``sf_edge`` (``[rows, K / 16]`` bool, the reference's e4m3
    rounding-edge mask of :func:`requant_reference`) adds the kernel's legitimate one-ulp scale difference:
    in a block whose two scale bytes are adjacent e4m3 codes and whose reference ``SFValue`` lies on the
    rounding edge, every element whose magnitude code differs by at most one (sign as above) is excused.  A
    one-ulp scale change (ratio <= 9 / 8 for normal e4m3 scales; a denormal scale's block values lie far
    below ``atol``) moves at most one e2m1 rounding boundary (adjacent boundaries are >= 1.4 apart), so a
    larger code difference is never legitimate."""
    codes_a = codes_a.reshape(codes_e.shape).int()
    codes_e = codes_e.int()
    mag_a, mag_e = codes_a & 7, codes_e & 7
    sign_ok = ((codes_a >> 3) == (codes_e >> 3)) | (mag_a == 0)
    step = (mag_a - mag_e).abs()
    sf_a = sf_a.reshape(sf_e.shape)

    def per_element(block_mask: torch.Tensor) -> torch.Tensor:
        return block_mask.repeat_interleave(block, dim=-1).reshape(codes_e.shape)

    flips = per_element(sf_a == sf_e) & sign_ok & (step == 1)
    if sf_edge is not None:
        adjacent = (sf_a.int() - sf_e.int()).abs() == 1
        edge = adjacent & sf_edge.reshape(sf_e.shape).bool()
        flips = flips | (per_element(edge) & sign_ok & (step <= 1))
    return flips


def assert_family_close(
    actual,
    expected,
    family: str,
    *,
    label: str,
    excuse: Optional[torch.Tensor] = None,
    tolerance: Optional[tuple[float, float]] = None,
) -> dict[str, Any]:
    """The option-A gate: every element within ``TOL[family]`` (``tolerance`` overrides).  ``excuse`` is the
    one-ulp block mask (:func:`e4m3_block_flips` / :func:`e2m1_block_flips`) of the elements the rule removes
    from ``exceeds``.  Returns the error record plus ``one_ulp_block_flips`` (the excused elements that
    were outside the tolerance)."""
    actual_f = actual.float()
    assert torch.isfinite(actual_f).all(), f"{label}: non-finite values"
    atol, rtol = TOL[family] if tolerance is None else tolerance
    err = errors(actual_f, expected, atol, rtol)
    exceeds = err.pop("exceeds")
    flips = 0
    if excuse is not None:
        excuse = excuse.reshape(exceeds.shape)
        flips = int((exceeds & excuse).sum().item())
        exceeds = exceeds & ~excuse
    err["one_ulp_block_flips"] = flips
    err["n_outside"] = int(exceeds.sum().item())
    assert not bool(exceeds.any()), (
        f"{label}: {err['n_outside']}/{err['numel']} outside atol={atol} rtol={rtol} "
        f"(max_abs {err['max_abs']:.4g}, max_rel {err['max_rel']:.4g}, 1-ulp flips {flips})"
    )
    return err


def bf16_band_count(actual, expected) -> int:
    """Information column: elements outside the bf16 band ``1e-2 / 1e-2`` (never a gate for fp8 / nvfp4)."""
    return errors(actual, expected, *BF16_BAND)["n_outside"]


def nvfp4_rows_pair(
    *,
    packed_a: torch.Tensor,
    sf_a: torch.Tensor,
    packed_e: torch.Tensor,
    sf_e: torch.Tensor,
    token_scale_a: Optional[torch.Tensor] = None,
    token_scale_e: Optional[torch.Tensor] = None,
    sf_edge_e: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """``(dequant_a, dequant_e, excuse, info)`` of nvfp4 rows (packed codes ``[n, K / 2]`` with their logical
    scale bytes ``[n, K / 16]`` and optional per-row scales): the dequantized products ``code * sf
    (* token_scale)``, the e2m1 one-code block mask (:func:`e2m1_block_flips`; with the one-ulp scale rule
    when the reference's ``sf_edge_e`` is given) and the information columns ``sf_mismatch_bytes``,
    ``e2m1_one_code_flips``, ``bitwise`` (plus ``token_scale_bitwise`` and ``sf_edge_blocks``)."""
    deq_a = nvfp4_dequantize(packed_a, sf_a)
    deq_e = nvfp4_dequantize(packed_e, sf_e)
    if token_scale_a is not None:
        deq_a = deq_a * token_scale_a.float()[:, None]
        deq_e = deq_e * token_scale_e.float()[:, None]
    codes_a, codes_e = e2m1_unpack(packed_a), e2m1_unpack(packed_e)
    excuse = e2m1_block_flips(codes_a, codes_e, sf_a, sf_e, sf_edge=sf_edge_e)
    info: dict[str, Any] = {
        "sf_mismatch_bytes": int((sf_a.reshape(-1) != sf_e.reshape(-1)).sum().item()),
        "e2m1_one_code_flips": int(
            (((codes_a.int() & 7) - (codes_e.int() & 7)).abs() == 1).sum().item()
        ),
        "bitwise": bool(
            torch.equal(packed_a, packed_e)
            and torch.equal(sf_a.reshape(-1), sf_e.reshape(-1))
        ),
    }
    if token_scale_a is not None:
        info["token_scale_bitwise"] = bool(
            torch.equal(
                token_scale_a.contiguous().view(torch.int32),
                token_scale_e.contiguous().view(torch.int32),
            )
        )
        info["bitwise"] = info["bitwise"] and info["token_scale_bitwise"]
    if sf_edge_e is not None:
        info["sf_edge_blocks"] = int(sf_edge_e.sum().item())
    return deq_a, deq_e, excuse, info


def nvfp4_rows_compare(
    *,
    packed_a: torch.Tensor,
    sf_a: torch.Tensor,
    packed_e: torch.Tensor,
    sf_e: torch.Tensor,
    token_scale_a: Optional[torch.Tensor] = None,
    token_scale_e: Optional[torch.Tensor] = None,
    sf_edge_e: Optional[torch.Tensor] = None,
    label: str,
) -> dict[str, Any]:
    """nvfp4 rows compared as the dequantized product at ``TOL["nvfp4"]`` under the e2m1 block rule of
    :func:`nvfp4_rows_pair`; its information columns are returned, never gated."""
    deq_a, deq_e, excuse, info = nvfp4_rows_pair(
        packed_a=packed_a,
        sf_a=sf_a,
        packed_e=packed_e,
        sf_e=sf_e,
        token_scale_a=token_scale_a,
        token_scale_e=token_scale_e,
        sf_edge_e=sf_edge_e,
    )
    err = assert_family_close(deq_a, deq_e, "nvfp4", label=label, excuse=excuse)
    err.update(info)
    return err


# --------------------------------------------------------------------------- stage references


def fc1_reference(
    act_rows: torch.Tensor,
    w1_expert: torch.Tensor,
    limit: float,
    *,
    scale_gate: float = 1.0,
    scale_c: float = 1.0,
    token_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """fp32 FC1 output rows of one expert in the family's output units.  ``act_rows [n, H]`` and
    ``w1_expert [2 I, H]`` (canonical row order: linear half first) are in the accumulator units of
    :class:`StepFunWeights` (dequantized values; physical e4m3 codes for fp8), so ``acc`` is the kernel's raw
    accumulator and ``scale_gate`` / ``scale_c`` apply as the kernel applies them.  ``token_scale`` (per-token
    nvfp4 family) multiplies both accumulator halves before the clamp / gate math, as the kernel does."""
    acc = act_rows.float() @ w1_expert.float().T
    if token_scale is not None:
        acc = acc * token_scale.float()[:, None]
    return stepfun_rows(acc, limit, scale_gate=scale_gate, scale_c=scale_c)


def fc2_reference(
    fc1_rows: torch.Tensor,
    w2_expert: torch.Tensor,
    *,
    scale_c: float = 1.0,
    token_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """fp32 FC2 rows of one expert: ``fc1_rows [n, I]`` (the family's quantized FC1 output as FC2 reads it,
    ``QuantizedRows.dequant``) and ``w2_expert [H, I]`` in the accumulator units of :class:`StepFunWeights`;
    ``scale_c`` is the kernel's per-expert FC2 output scale on the raw product."""
    out = (fc1_rows.float() @ w2_expert.float().T) * scale_c
    if token_scale is not None:
        out = out * token_scale.float()[:, None]
    return out


def finalize_reference(
    gemm2_output: torch.Tensor, expert_weights: torch.Tensor, e2p: torch.Tensor
) -> torch.Tensor:
    """bf16 ``[T, H]``: fp32 sum over the routed rows in ``k`` order, ``-1`` slots skipped."""
    num_tokens, top_k = expert_weights.shape
    e2p = e2p.reshape(num_tokens, top_k).long()
    out = torch.zeros(
        num_tokens,
        gemm2_output.shape[1],
        dtype=torch.float32,
        device=gemm2_output.device,
    )
    rows = gemm2_output.float()
    for k in range(top_k):
        slot = e2p[:, k]
        valid = slot >= 0
        gathered = rows[slot.clamp(min=0)] * expert_weights[:, k].float()[:, None]
        out = out + torch.where(valid[:, None], gathered, torch.zeros_like(gathered))
    return out.to(torch.bfloat16)


@dataclass
class QuantizedRows:
    """Rows in the kernel form of one family plus ``dequant``, the fp32 value the next stage's reference
    consumes in its accumulator units: the dequantized rows for bf16 / mxfp8 / nvfp4, the e4m3 codes for fp8,
    ``code * sf`` without the ``token_scale`` for the requantized per-token rows."""

    dequant: torch.Tensor  # float32 [n, K]; fp8: the e4m3 codes as float
    bf16: Optional[torch.Tensor] = None
    e4m3: Optional[torch.Tensor] = None  # fp8 / mxfp8 codes [n, K]
    scale: Optional[torch.Tensor] = (
        None  # mxfp8 UE8M0 [n, K/32] | nvfp4 e4m3 bytes [n, K/16] (logical)
    )
    packed: Optional[torch.Tensor] = None  # nvfp4 [n, K/2]
    token_scale: Optional[torch.Tensor] = None  # requantized per-token family [n]
    # requant: the blocks [n, K/16] whose SFValue lies on an e4m3 rounding edge
    sf_edge: Optional[torch.Tensor] = None


def quantize_fc1_output(family: str, values: torch.Tensor) -> QuantizedRows:
    """The FC1 epilogue's output quantization per family on fp32 ``values [n, I]``."""
    if family in ("bf16", "nvfp4_bf16tok"):
        out = values.to(torch.bfloat16)
        return QuantizedRows(dequant=out.float(), bf16=out)
    if family == "fp8":
        q = values.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return QuantizedRows(dequant=q.float(), e4m3=q)
    if family == "mxfp8":
        q = mxfp8_quantize(values)
        return QuantizedRows(dequant=q.dequant, e4m3=q.codes, scale=q.scale)
    if family == "nvfp4":
        q = nvfp4_quantize(values)
        return QuantizedRows(dequant=q.dequant, packed=q.packed, scale=q.scale)
    raise KeyError(family)


def requant_reference(
    fc1_rows: torch.Tensor, global_scale_inv: float = GLOBAL_SCALE_INV
) -> QuantizedRows:
    """The per-token nvfp4 requantization of bf16 FC1 rows, mirroring the fp32 arithmetic of the requant
    kernel (``nvfp4QuantAndPerTokenScaleKernel``, default recipe; the Cake twin
    ``trtllm_moe_nvfp4_per_token_requant``) with IEEE reciprocals in place of its ``rcp.approx.ftz``:

    * ``token_scale = row_amax * global_scale_inv`` (one fp32 product; bitwise with the kernel),
    * per 16-block ``SFValue = (1 / token_scale) * (vec_max * fp32(1 / 6))``, ``sf = e4m3_rn(SFValue)``,
    * ``outputScale = 1 / (sf * (1 / (1 / token_scale)))`` (0 for an all-zero block),
    * codes ``e2m1_rn_even(x * outputScale)``, saturating at 6.

    ``dequant`` is ``code * sf``, the FC2 accumulator operand; the FC2 epilogue applies ``token_scale`` to the
    product (:func:`fc2_reference` ``token_scale``), so the physical row is ``dequant * token_scale``.
    ``sf_edge`` marks the blocks whose ``SFValue`` lies within :data:`REQUANT_SF_EDGE` of an e4m3 rounding
    midpoint, where the kernel's approximate reciprocals can legitimately round the scale to the
    neighbouring e4m3 code (:func:`e2m1_block_flips` ``sf_edge``)."""
    rows = fc1_rows.float()
    n, k = rows.shape
    blocks = rows.reshape(n, k // NVFP4_SF_VEC, NVFP4_SF_VEC)
    token_scale = rows.abs().amax(dim=1) * global_scale_inv
    encode = torch.where(
        token_scale != 0, 1.0 / token_scale, torch.zeros_like(token_scale)
    )
    vec_max = blocks.abs().amax(dim=-1)
    sf_value = encode[:, None] * (vec_max * F32_SIXTH)
    sf = e4m3_round(sf_value)
    output_scale = torch.where(
        vec_max != 0, 1.0 / (sf * (1.0 / encode[:, None])), torch.zeros_like(sf)
    )
    codes = e2m1_codes(blocks * output_scale[..., None])
    dequant = (e2m1_values(codes) * sf[..., None]).reshape(n, k)
    return QuantizedRows(
        dequant=dequant,
        packed=e2m1_pack(codes.reshape(n, k)),
        scale=sf.to(torch.float8_e4m3fn).view(torch.uint8),
        token_scale=token_scale,
        sf_edge=e4m3_rounding_edge(sf_value),
    )


def fc2_input_from_rows(
    family: str, rows: torch.Tensor, sf_layout: str
) -> tuple[dict[str, Any], torch.Tensor]:
    """``(prepare_fc2 keywords, QuantizedRows.dequant)`` for fp32 rows ``[max_padded, I]`` in the kernel form of the
    family's FC2 input: ``gemm2_input``, ``gemm2_input_scale`` (flat, swizzled to ``sf_layout``) and
    ``per_token_scale`` (per-token family; the returned ``dequant`` excludes it -- pass it to
    :func:`fc2_reference` ``token_scale``).  The row count must be a multiple of the layout's block rows."""
    if family == "nvfp4_bf16tok":
        q = requant_reference(rows.to(torch.bfloat16))
        return (
            {
                "gemm2_input": q.packed,
                "gemm2_input_scale": swizzle_sf(q.scale, sf_layout),
                "per_token_scale": q.token_scale.contiguous(),
            },
            q.dequant,
        )
    q = quantize_fc1_output(family, rows)
    if family == "nvfp4":
        return {
            "gemm2_input": q.packed,
            "gemm2_input_scale": swizzle_sf(q.scale, sf_layout),
        }, q.dequant
    if family == "mxfp8":
        return {
            "gemm2_input": q.e4m3,
            "gemm2_input_scale": swizzle_sf(q.scale, sf_layout),
        }, q.dequant
    if family == "fp8":
        return {"gemm2_input": q.e4m3}, q.dequant
    return {"gemm2_input": q.bf16}, q.dequant


# Storage dtypes a family's ``gemm1_output`` may be declared with (``unit["arg_dtypes"]["C"]``).
FC1_OUTPUT_STORAGE = {
    "nvfp4": (torch.uint8,),
    "nvfp4_bf16tok": (torch.bfloat16,),
    "bf16": (torch.bfloat16,),
    "fp8": (torch.uint8, torch.float8_e4m3fn),
    "mxfp8": (torch.uint8, torch.float8_e4m3fn),
}


def fc1_output_buffers(
    family: str,
    max_padded: int,
    sf_layout: str,
    device,
    *,
    output_dtype: torch.dtype,
) -> dict[str, Any]:
    """``gemm1_output`` / ``gemm1_output_scale`` buffers of ``prepare_fc1`` for ``max_padded`` rows in the
    kernel form of the family's FC1 output.  ``output_dtype`` is the unit's declared storage dtype of ``C``
    (``cake_jit.unit(program)["arg_dtypes"]["C"]``, the dtype the backend checks): ``uint8`` for the nvfp4
    codes, ``bfloat16`` for the bf16 / per-token rows, a one-byte e4m3 storage for fp8 / mxfp8 (``uint8`` at
    the pin; :func:`fc1_output_rows` views it as e4m3).  The scale buffer is flat in ``sf_layout``."""
    i = INTERMEDIATE_SIZE
    if output_dtype not in FC1_OUTPUT_STORAGE[family]:
        raise ValueError(
            f"{family} FC1 output storage must be one of {FC1_OUTPUT_STORAGE[family]}, "
            f"got {output_dtype}"
        )
    cols = i // 2 if family == "nvfp4" else i
    buffers = {
        "gemm1_output": torch.zeros(max_padded, cols, dtype=output_dtype, device=device)
    }
    block = {"nvfp4": NVFP4_SF_VEC, "mxfp8": MXFP8_SF_VEC}.get(family)
    if block is not None:
        buffers["gemm1_output_scale"] = torch.zeros(
            swizzled_sf_bytes(max_padded, i // block, sf_layout),
            dtype=torch.uint8,
            device=device,
        )
    return buffers


def e4m3_codes(tensor: torch.Tensor) -> torch.Tensor:
    """A one-byte e4m3 storage (``uint8`` bytes, the declared storage of the fp8 / mxfp8 FC1 output, or
    ``float8_e4m3fn``) viewed as ``float8_e4m3fn``."""
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor
    if tensor.dtype != torch.uint8:
        raise TypeError(
            f"e4m3 codes need uint8 or float8_e4m3fn storage, got {tensor.dtype}"
        )
    return tensor.view(torch.float8_e4m3fn)


def fc1_output_rows(
    family: str,
    buffers: dict[str, Any],
    rows: torch.Tensor,
    max_padded: int,
    sf_layout: str,
) -> QuantizedRows:
    """The kernel's FC1 output at ``rows`` (a long index tensor into the permuted buffer) read back into the
    family's :class:`QuantizedRows` form (scales un-swizzled to the logical layout; the fp8 / mxfp8 bytes
    viewed as e4m3 through :func:`e4m3_codes`)."""
    out = buffers["gemm1_output"]
    i = INTERMEDIATE_SIZE
    if family == "nvfp4":
        sf = unswizzle_sf(
            buffers["gemm1_output_scale"], max_padded, i // NVFP4_SF_VEC, sf_layout
        )
        packed = out[rows]
        return QuantizedRows(
            dequant=nvfp4_dequantize(packed, sf[rows]), packed=packed, scale=sf[rows]
        )
    if family == "mxfp8":
        sf = unswizzle_sf(
            buffers["gemm1_output_scale"], max_padded, i // MXFP8_SF_VEC, sf_layout
        )
        codes = e4m3_codes(out)[rows]
        return QuantizedRows(
            dequant=mxfp8_dequantize(codes, sf[rows]), e4m3=codes, scale=sf[rows]
        )
    if family == "fp8":
        codes = e4m3_codes(out)[rows]
        return QuantizedRows(dequant=codes.float(), e4m3=codes)
    values = out[rows]
    return QuantizedRows(dequant=values.float(), bf16=values)


def fc1_rows_pair(
    family: str, actual: QuantizedRows, expected: QuantizedRows
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], dict[str, Any]]:
    """``(actual, expected, excuse, info)`` of the FC1 stage gate per family: the dequantized values with the
    one-ulp block mask of the family's output code (e4m3 for fp8 / mxfp8, e2m1 for nvfp4), bf16 outputs
    plain (``excuse = None``)."""
    if family == "nvfp4":
        return nvfp4_rows_pair(
            packed_a=actual.packed,
            sf_a=actual.scale,
            packed_e=expected.packed,
            sf_e=expected.scale,
        )
    excuse, info = None, {}
    if family == "fp8":
        excuse = e4m3_block_flips(actual.e4m3, expected.e4m3, None, None, 1)
    elif family == "mxfp8":
        excuse = e4m3_block_flips(
            actual.e4m3, expected.e4m3, actual.scale, expected.scale, MXFP8_SF_VEC
        )
        info["sf_mismatch_bytes"] = int((actual.scale != expected.scale).sum().item())
    return actual.dequant, expected.dequant, excuse, info


def assert_fc1_rows_close(
    family: str, actual: QuantizedRows, expected: QuantizedRows, *, label: str
) -> dict[str, Any]:
    """The FC1 stage gate per family (:func:`fc1_rows_pair`) at ``TOL[family]``."""
    a, e, excuse, info = fc1_rows_pair(family, actual, expected)
    err = assert_family_close(a, e, family, label=label, excuse=excuse)
    err.update(info)
    return err


# --------------------------------------------------------------------------- canonical data -> kernel view


@dataclass
class StepFunWeights:
    """Canonical bf16 experts of one seed quantized into the kernel form of ``family`` (FlashInfer layouts:
    shuffled FC1 / FC2 rows, BlockMajorK for bf16, 128x4-interleaved weight scales).  ``w1_ref(e)`` /
    ``w2_ref(e)`` return the fp32 reference operands ``[2 I, H]`` / ``[H, I]`` in canonical row order and in
    the units of the kernel's accumulator: the dequantized values for bf16 / mxfp8 / nvfp4 (whose per-expert
    scales are 1), the physical e4m3 codes for fp8 (the kernel scales the raw code products by
    ``scale_gate`` / ``scale_c`` in FC1 and by ``scale_c2`` in FC2).  Recomputed per call: the operands are
    not kept."""

    family: str
    seed: int
    view: dict[str, torch.Tensor]
    w1_quant_scale: Optional[torch.Tensor] = (
        None  # fp8: per-expert quantization multiplier of w1
    )
    w2_quant_scale: Optional[torch.Tensor] = None
    _w1: Callable[[int], torch.Tensor] = field(repr=False, default=None)
    _w2: Callable[[int], torch.Tensor] = field(repr=False, default=None)

    def w1_ref(self, expert: int) -> torch.Tensor:
        return self._w1(expert)

    def w2_ref(self, expert: int) -> torch.Tensor:
        return self._w2(expert)


@dataclass
class StepFunCase:
    """One (family, T, limit mode) case: the keyword view of ``prepare_stepfun_moe`` (weights, scales,
    activations, ``gemm1_clamp_limit`` in kernel units) and the fp32 reference operands: ``act_ref`` /
    ``w1_ref`` / ``w2_ref`` in the family's accumulator units (see :class:`StepFunWeights`) and the
    per-expert scales the kernels apply to the raw accumulators."""

    family: str
    num_tokens: int
    limit_mode: str
    limits: torch.Tensor  # fp32 [E], logical (inf allowed)
    weights: StepFunWeights
    view: dict[str, torch.Tensor]
    act_ref: torch.Tensor  # fp32 [T, H], accumulator units (fp8: e4m3 codes); per-token family: without the token scale
    per_token_scale: Optional[torch.Tensor]
    scale_gate: Optional[torch.Tensor]  # per-expert FC1 accumulator scale (None -> 1)
    scale_c: Optional[torch.Tensor]  # per-expert FC1 output scale (None -> 1)
    scale_c2: Optional[torch.Tensor]  # per-expert FC2 output scale (None -> 1)

    def w1_ref(self, expert: int) -> torch.Tensor:
        return self.weights.w1_ref(expert)

    def w2_ref(self, expert: int) -> torch.Tensor:
        return self.weights.w2_ref(expert)

    def sg(self, expert: int) -> float:
        return 1.0 if self.scale_gate is None else float(self.scale_gate[expert])

    def sc(self, expert: int) -> float:
        return 1.0 if self.scale_c is None else float(self.scale_c[expert])

    def sc2(self, expert: int) -> float:
        return 1.0 if self.scale_c2 is None else float(self.scale_c2[expert])

    def fc1_view(self) -> dict[str, torch.Tensor]:
        return {k: self.view[k] for k in FC1_VIEW_KEYS if k in self.view}

    def fc2_view(self) -> dict[str, torch.Tensor]:
        return {k: self.view[k] for k in FC2_VIEW_KEYS if k in self.view}


def canonical_weights(seed: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """The previous test's experts (``run_moe_test``): bf16 weights scaled by ``1 / sqrt(K)``."""
    generator = torch.Generator(device=device).manual_seed(seed)
    w1 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    w2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        device=device,
        dtype=torch.float32,
        generator=generator,
    )
    w1 = (w1 / math.sqrt(HIDDEN_SIZE)).to(torch.bfloat16)
    w2 = (w2 / math.sqrt(INTERMEDIATE_SIZE)).to(torch.bfloat16)
    return w1, w2


def canonical_hidden(num_tokens: int, seed: int, device) -> torch.Tensor:
    """Amplitude-12 bf16 activations ``[T, H]`` (the previous test's data; seeded per input set)."""
    generator = torch.Generator(device=device).manual_seed(seed + 7919 * num_tokens)
    return (
        HIDDEN_STATE_AMPLITUDE
        * torch.randn(
            num_tokens,
            HIDDEN_SIZE,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
    ).to(torch.bfloat16)


def build_weights(family: str, seed: int, device) -> StepFunWeights:
    """Quantize the canonical experts of ``seed`` into the kernel view of ``family`` (quantized with this
    module's quantizers; the layouts are the FlashInfer ones)."""
    w1, w2 = canonical_weights(seed, device)
    perm1 = torch.tensor(
        fc1_weight_row_permutation(INTERMEDIATE_SIZE), dtype=torch.long, device=device
    )
    perm2 = torch.tensor(
        fc2_weight_row_permutation(HIDDEN_SIZE), dtype=torch.long, device=device
    )
    view: dict[str, torch.Tensor] = {}
    if family == "bf16":
        view["gemm1_weights"] = to_block_major_k(w1[:, perm1, :].contiguous())
        view["gemm2_weights"] = to_block_major_k(w2[:, perm2, :].contiguous())
        return StepFunWeights(
            family,
            seed,
            view,
            _w1=lambda e: w1[e].float(),
            _w2=lambda e: w2[e].float(),
        )
    if family == "fp8":
        w1_scale = E4M3_MAX / w1.float().abs().amax(dim=(1, 2))
        w2_scale = E4M3_MAX / w2.float().abs().amax(dim=(1, 2))
        w1_q = torch.stack(
            [
                fp8_per_tensor_quantize(w1[e], float(w1_scale[e]))
                for e in range(NUM_EXPERTS)
            ]
        )
        w2_q = torch.stack(
            [
                fp8_per_tensor_quantize(w2[e], float(w2_scale[e]))
                for e in range(NUM_EXPERTS)
            ]
        )
        view["gemm1_weights"] = w1_q[:, perm1, :].contiguous()
        view["gemm2_weights"] = w2_q[:, perm2, :].contiguous()
        return StepFunWeights(
            family,
            seed,
            view,
            w1_quant_scale=w1_scale.contiguous(),
            w2_quant_scale=w2_scale.contiguous(),
            _w1=lambda e: w1_q[e].float(),
            _w2=lambda e: w2_q[e].float(),
        )
    if family == "mxfp8":
        w1_q = [mxfp8_quantize(w1[e]) for e in range(NUM_EXPERTS)]
        w2_q = [mxfp8_quantize(w2[e]) for e in range(NUM_EXPERTS)]
        w1_codes = torch.stack([q.codes for q in w1_q])
        w1_sf = torch.stack([q.scale for q in w1_q])
        w2_codes = torch.stack([q.codes for q in w2_q])
        w2_sf = torch.stack([q.scale for q in w2_q])
        view["gemm1_weights"] = w1_codes[:, perm1, :].contiguous()
        view["gemm1_weights_scale"] = interleave_weight_scales(w1_sf[:, perm1, :])
        view["gemm2_weights"] = w2_codes[:, perm2, :].contiguous()
        view["gemm2_weights_scale"] = interleave_weight_scales(w2_sf[:, perm2, :])
        return StepFunWeights(
            family,
            seed,
            view,
            _w1=lambda e: mxfp8_dequantize(w1_codes[e], w1_sf[e]),
            _w2=lambda e: mxfp8_dequantize(w2_codes[e], w2_sf[e]),
        )
    if family in ("nvfp4", "nvfp4_bf16tok"):
        w1_q = [nvfp4_quantize(w1[e]) for e in range(NUM_EXPERTS)]
        w2_q = [nvfp4_quantize(w2[e]) for e in range(NUM_EXPERTS)]
        w1_packed = torch.stack([q.packed for q in w1_q])
        w1_sf = torch.stack([q.scale for q in w1_q])
        w2_packed = torch.stack([q.packed for q in w2_q])
        w2_sf = torch.stack([q.scale for q in w2_q])
        view["gemm1_weights"] = w1_packed[:, perm1, :].contiguous()
        view["gemm1_weights_scale"] = interleave_weight_scales(w1_sf[:, perm1, :]).view(
            torch.float8_e4m3fn
        )
        view["gemm2_weights"] = w2_packed[:, perm2, :].contiguous()
        view["gemm2_weights_scale"] = interleave_weight_scales(w2_sf[:, perm2, :]).view(
            torch.float8_e4m3fn
        )
        return StepFunWeights(
            family,
            seed,
            view,
            _w1=lambda e: nvfp4_dequantize(w1_packed[e], w1_sf[e]),
            _w2=lambda e: nvfp4_dequantize(w2_packed[e], w2_sf[e]),
        )
    raise KeyError(family)


def build_case(
    weights: StepFunWeights,
    num_tokens: int,
    limit_mode: str,
    *,
    seed: int = 0,
    device="cuda",
) -> StepFunCase:
    """The activations of (T, seed) quantized for ``weights.family``, the per-expert scales and the
    ``gemm1_clamp_limit`` of ``limit_mode`` in the units the kernels clamp (raw accumulator units for nvfp4 /
    fp8: logical limit over the FC1 gate dequant scale; physical units for bf16 / mxfp8).  ``act_ref`` is the
    activation operand of the reference in the family's accumulator units (fp8: the e4m3 codes)."""
    family = weights.family
    hidden = canonical_hidden(num_tokens, seed, device)
    limits = limits_tensor(limit_mode, device)
    view: dict[str, torch.Tensor] = dict(weights.view)
    ones = torch.ones(NUM_EXPERTS, dtype=torch.float32, device=device)
    scale_gate = scale_c = scale_c2 = per_token_scale = None
    if family == "bf16":
        view["hidden_states"] = hidden.contiguous()
        act_ref = hidden.float()
    elif family == "fp8":
        input_scale = float(E4M3_MAX / hidden.float().abs().amax())
        act_q = fp8_per_tensor_quantize(hidden, input_scale)
        act_ref = act_q.float()
        intermediate_scale = FP8_INTERMEDIATE_SCALE[
            "inf" if limit_mode == "inf" else "finite"
        ]
        scale_gate = (1.0 / (weights.w1_quant_scale * input_scale)).contiguous()
        scale_c = (
            intermediate_scale / (weights.w1_quant_scale * input_scale)
        ).contiguous()
        scale_c2 = (1.0 / (intermediate_scale * weights.w2_quant_scale)).contiguous()
        view.update(
            hidden_states=act_q.contiguous(),
            output1_scale_scalar=scale_c,
            output1_scale_gate_scalar=scale_gate,
            output2_scale_scalar=scale_c2,
        )
    elif family == "mxfp8":
        act_q = mxfp8_quantize(hidden)
        act_ref = act_q.dequant
        view.update(
            hidden_states=act_q.codes.contiguous(),
            hidden_states_scale=act_q.scale.contiguous(),
        )
    elif family in ("nvfp4", "nvfp4_bf16tok"):
        if family == "nvfp4":
            act_q = nvfp4_quantize(hidden)
        else:
            # per token: token_scale = amax_t / (448 * 6); the codes quantize hidden / token_scale and the
            # kernel multiplies the accumulators by token_scale (act_ref excludes it).
            per_token_scale = (
                hidden.float().abs().amax(dim=1) * GLOBAL_SCALE_INV
            ).contiguous()
            act_q = nvfp4_quantize(hidden.float() / per_token_scale[:, None])
        act_ref = act_q.dequant
        scale_gate, scale_c, scale_c2 = ones, ones, ones
        view.update(
            output1_scale_scalar=scale_c,
            output1_scale_gate_scalar=scale_gate,
            output2_scale_scalar=scale_c2,
            hidden_states=act_q.packed.contiguous(),
            hidden_states_scale=act_q.scale.contiguous().view(torch.float8_e4m3fn),
        )
        if per_token_scale is not None:
            view["per_token_scale"] = per_token_scale
    else:
        raise KeyError(family)
    if scale_gate is None:
        view["gemm1_clamp_limit"] = limits.clone()
    else:
        view["gemm1_clamp_limit"] = (limits / scale_gate).contiguous()
    return StepFunCase(
        family=family,
        num_tokens=num_tokens,
        limit_mode=limit_mode,
        limits=limits,
        weights=weights,
        view=view,
        act_ref=act_ref,
        per_token_scale=per_token_scale,
        scale_gate=scale_gate,
        scale_c=scale_c,
        scale_c2=scale_c2,
    )


def expert_segments(
    routing: dict[str, Any], *, tile: int
) -> dict[int, tuple[int, int]]:
    """``{expert: (first row, end row)}`` of the permuted buffers (an expert's CTAs are contiguous)."""
    nnec = _as_int(routing["num_non_exiting_ctas"])
    batch = routing["cta_idx_xy_to_batch_idx"].reshape(-1)[:nnec].tolist()
    limit = routing["cta_idx_xy_to_mn_limit"].reshape(-1)[:nnec].tolist()
    segments: dict[int, tuple[int, int]] = {}
    for cta, expert in enumerate(batch):
        start, _end = segments.get(expert, (cta * tile, cta * tile))
        segments[expert] = (start, limit[cta])
    return {e: span for e, span in segments.items() if span[1] > span[0]}


def fc1_stage_reference(
    case: StepFunCase, routing: dict[str, Any], *, tile: int
) -> dict[int, tuple[torch.Tensor, torch.Tensor, QuantizedRows]]:
    """Per expert ``(rows, fp32 FC1 output, quantized FC1 output)`` of the routing fixture."""
    p2t = routing["permuted_idx_to_token_idx"].reshape(-1)
    result = {}
    for expert, (start, end) in expert_segments(routing, tile=tile).items():
        tokens = p2t[start:end].long()
        token_scale = (
            case.per_token_scale[tokens] if case.per_token_scale is not None else None
        )
        act = fc1_reference(
            case.act_ref[tokens],
            case.w1_ref(expert),
            float(case.limits[expert]),
            scale_gate=case.sg(expert),
            scale_c=case.sc(expert),
            token_scale=token_scale,
        )
        rows = torch.arange(start, end, device=p2t.device)
        result[expert] = (rows, act, quantize_fc1_output(case.family, act))
    return result


def graph_reference(
    case: StepFunCase, routing: dict[str, Any], *, tile: int
) -> dict[str, torch.Tensor]:
    """The fp32 reference of the whole graph on the routing ``routing`` (tables of :func:`reference_routing`
    or the kernel's own): per expert FC1 -> the family's output quantization (-> requant for the per-token
    family) -> FC2 -> bf16, then finalize with ``routing["topk_weights"]``.  Returns ``output`` (bf16 ``[T, H]``),
    ``gemm2_output`` (bf16 ``[rows, H]``, permuted) and ``fc1_rows`` (fp32 ``[rows, I]``, the quantized FC1 rows
    as the FC2 accumulator reads them: ``QuantizedRows.dequant``, without the per-token scale), where ``rows =
    num_non_exiting_ctas * tile`` is the live extent; the kernel's buffers hold the geometry's
    ``max_padded_tokens`` rows, so compare live rows only."""
    nnec = _as_int(routing["num_non_exiting_ctas"])
    max_padded = nnec * tile
    device = case.act_ref.device
    fc1 = torch.zeros(max_padded, INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
    gemm2 = torch.zeros(max_padded, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    for expert, (rows, _act, quantized) in fc1_stage_reference(
        case, routing, tile=tile
    ).items():
        fc2_token_scale = None
        if case.family == "nvfp4_bf16tok":
            quantized = requant_reference(quantized.bf16)
            fc2_token_scale = quantized.token_scale
        fc1[rows] = quantized.dequant
        gemm2[rows] = fc2_reference(
            quantized.dequant,
            case.w2_ref(expert),
            scale_c=case.sc2(expert),
            token_scale=fc2_token_scale,
        ).to(torch.bfloat16)
    output = finalize_reference(
        gemm2, routing["topk_weights"], routing["expanded_idx_to_permuted_idx"]
    )
    return {"output": output, "gemm2_output": gemm2, "fc1_rows": fc1}


def concat_quantized_rows(parts: list[QuantizedRows]) -> QuantizedRows:
    """Row concatenation of :class:`QuantizedRows` of one family (every field set on all parts)."""

    def cat(name: str) -> Optional[torch.Tensor]:
        values = [getattr(part, name) for part in parts]
        return None if any(v is None for v in values) else torch.cat(values)

    return QuantizedRows(
        **{
            name: cat(name)
            for name in (
                "dequant",
                "bf16",
                "e4m3",
                "scale",
                "packed",
                "token_scale",
                "sf_edge",
            )
        }
    )


@dataclass
class GraphHop:
    """One hop of the whole-graph gate (:func:`graph_hops`): the kernel's rows of a stage (``actual``) against
    the reference of that stage computed from the kernel's OWN upstream rows (``expected``), both in the
    units :func:`assert_family_close` compares at ``TOL[family]``; ``excuse`` is the stage's one-ulp block
    mask (or ``None``), ``info`` its information columns, ``rows`` the permuted rows of the compared
    elements (``None`` for the ``[T, H]`` output)."""

    name: str
    family: str
    actual: torch.Tensor
    expected: torch.Tensor
    excuse: Optional[torch.Tensor]
    info: dict[str, Any]
    rows: Optional[torch.Tensor]


def graph_hops(
    case: StepFunCase,
    buffers: dict[str, torch.Tensor],
    routing: dict[str, Any],
    *,
    tile: int,
    sf_layout: str,
    output: Optional[torch.Tensor] = None,
) -> dict[str, GraphHop]:
    """The hop-wise whole-graph comparison on the kernel's own buffers (``launch.buffers`` after a launch,
    ``routing`` = :func:`live_tables` of them), at the unchanged per-family tolerances:

    * ``fc1``: the kernel's FC1 rows (``gemm1_output`` / ``gemm1_output_scale``, dequantized) against the
      per-expert FC1 reference (:func:`fc1_stage_reference`) with the family's block rule
      (:func:`fc1_rows_pair`);
    * ``requant`` (per-token family): the kernel's requantized rows (``requant_output`` /
      ``requant_output_scale`` / ``requant_per_token_scale``, dequantized ``code * sf * token_scale``) against
      :func:`requant_reference` of the KERNEL's own bf16 FC1 rows, e2m1 one-code rule with the one-ulp scale
      rule on the reference's rounding edges;
    * ``fc2``: the kernel's ``gemm2_output`` live rows against :func:`fc2_reference` of the KERNEL's own FC2
      input (its dequantized FC1 rows, or its requant codes with its per-token scale);
    * ``finalize`` (when ``output`` is given): the kernel's output against :func:`finalize_reference` of the
      KERNEL's own ``gemm2_output`` with the device ``topk_weights`` / ``expanded_idx_to_permuted_idx``.

    Why hop-wise: kernel and reference round the FC1 rows and the FC2 rows independently (bf16 or the
    family's codes); a sub-ulp fp32 difference (fast-math sigmoid, MMA accumulation order) flips a rounding
    in a small fraction of elements and finalize carries ``w_k * ulp(row_k)`` into outputs that are much
    smaller (cancellation), so the end-to-end output misses the family tolerance without any stage being
    wrong; each hop bounds exactly one rounding stage.  Rows are the live permuted rows in expert-segment
    order; the kernel's buffers hold the geometry's ``max_padded`` rows."""
    family = case.family
    device = case.act_ref.device
    max_padded = int(buffers["gemm1_output"].shape[0])
    stage = fc1_stage_reference(case, routing, tile=tile)
    if not stage:
        raise ValueError("the routing routes no token")
    experts = list(stage)
    rows = torch.cat([stage[e][0] for e in experts])
    actual = fc1_output_rows(family, buffers, rows, max_padded, sf_layout)
    expected = concat_quantized_rows([stage[e][2] for e in experts])
    a, e, excuse, info = fc1_rows_pair(family, actual, expected)
    hops = {"fc1": GraphHop("fc1", family, a, e, excuse, info, rows)}
    fc2_input, fc2_token_scale = actual.dequant, None
    if family == "nvfp4_bf16tok":
        own = requant_reference(actual.bf16)
        sf = unswizzle_sf(
            buffers["requant_output_scale"],
            max_padded,
            INTERMEDIATE_SIZE // NVFP4_SF_VEC,
            sf_layout,
        )[rows]
        packed = buffers["requant_output"][rows]
        token_scale = buffers["requant_per_token_scale"][rows]
        a, e, excuse, info = nvfp4_rows_pair(
            packed_a=packed,
            sf_a=sf,
            packed_e=own.packed,
            sf_e=own.scale,
            token_scale_a=token_scale,
            token_scale_e=own.token_scale,
            sf_edge_e=own.sf_edge,
        )
        hops["requant"] = GraphHop("requant", family, a, e, excuse, info, rows)
        fc2_input, fc2_token_scale = nvfp4_dequantize(packed, sf), token_scale
    expected_fc2 = torch.empty(
        rows.numel(), HIDDEN_SIZE, dtype=torch.float32, device=device
    )
    start = 0
    for expert in experts:
        stop = start + stage[expert][0].numel()
        expected_fc2[start:stop] = fc2_reference(
            fc2_input[start:stop],
            case.w2_ref(expert),
            scale_c=case.sc2(expert),
            token_scale=(
                None if fc2_token_scale is None else fc2_token_scale[start:stop]
            ),
        )
        start = stop
    hops["fc2"] = GraphHop(
        "fc2", family, buffers["gemm2_output"][rows], expected_fc2, None, {}, rows
    )
    if output is not None:
        expected_output = finalize_reference(
            buffers["gemm2_output"][:max_padded],
            routing["topk_weights"],
            routing["expanded_idx_to_permuted_idx"],
        )
        hops["finalize"] = GraphHop(
            "finalize", family, output, expected_output, None, {}, None
        )
    return hops


def assert_graph_hops_close(
    hops: dict[str, GraphHop], *, label: str
) -> dict[str, dict[str, Any]]:
    """Gate every hop of :func:`graph_hops` with :func:`assert_family_close` at ``TOL[hop.family]`` under the
    hop's excuse mask; returns the error records with the hops' information columns."""
    records = {}
    for name, hop in hops.items():
        err = assert_family_close(
            hop.actual,
            hop.expected,
            hop.family,
            label=f"{label} hop {name}",
            excuse=hop.excuse,
        )
        err.update(hop.info)
        records[name] = err
    return records


def end_to_end_record(
    output: torch.Tensor, expected_output: torch.Tensor, family: str
) -> dict[str, Any]:
    """The recorded, never gated, end-to-end error of the kernel output against
    ``graph_reference(...)["output"]`` at ``TOL[family]`` (no mask)."""
    err = errors(output, expected_output, *TOL[family])
    err.pop("exceeds")
    return err


# --------------------------------------------------------------------------- error table


def error_row(
    *,
    stage: str,
    family: str,
    T: int,
    limit_mode: str,
    regime: str,
    input_set: int,
    arch: str,
    status: str = "gated",
    cake=None,
    native=None,
    cake_vs_native=None,
    atol=None,
    rtol=None,
    passed=None,
    **extra,
) -> dict[str, Any]:
    """One flat row of ``cake_stepfun_error_table/1``: the required keys always present, ``cake_*`` = Cake vs
    the fp32 reference, ``native_*`` = native vs the same reference, ``pass`` = the option-A gate (Cake vs
    native within ``(atol, rtol)``; routing rows ``bitwise``).  ``cake`` / ``native`` / ``cake_vs_native`` are
    :func:`errors` records (``exceeds`` dropped); ``status`` is ``gated`` / ``info`` / ``n/a``."""
    if stage not in ERROR_STAGES:
        raise KeyError(stage)
    if status not in ("gated", "info", "n/a"):
        raise ValueError(status)

    def pick(record, key):
        return None if record is None else record.get(key)

    row: dict[str, Any] = {
        "stage": stage,
        "family": family,
        "T": int(T),
        "limit_mode": str(limit_mode),
        "regime": str(regime),
        "input_set": int(input_set),
        "arch": arch,
        "cake_max_abs": pick(cake, "max_abs"),
        "cake_max_rel": pick(cake, "max_rel"),
        "native_max_abs": pick(native, "max_abs"),
        "native_max_rel": pick(native, "max_rel"),
        "atol": atol,
        "rtol": rtol,
        "pass": None if status != "gated" else bool(passed),
        "status": status,
        "cake_n_outside": pick(cake, "n_outside"),
        "native_n_outside": pick(native, "n_outside"),
        "cake_vs_native_max_abs": pick(cake_vs_native, "max_abs"),
        "cake_vs_native_max_rel": pick(cake_vs_native, "max_rel"),
        "cake_vs_native_n_outside": pick(cake_vs_native, "n_outside"),
        "one_ulp_block_flips": pick(cake_vs_native, "one_ulp_block_flips"),
    }
    row.update(extra)
    for key in ERROR_ROW_REQUIRED_KEYS:
        if key not in row:
            raise KeyError(key)
    return row


def error_key(row: dict[str, Any]) -> tuple:
    return (
        row["stage"],
        row["family"],
        row["T"],
        row["limit_mode"],
        row["regime"],
        row["input_set"],
    )


def error_table_document(
    header: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    document = {"schema": ERROR_TABLE_SCHEMA}
    document.update(header)
    document["tolerances"] = {family: list(tol) for family, tol in TOL.items()}
    document["rows"] = list(rows)
    return document


def write_error_table(
    directory, arch: str, header: dict[str, Any], rows: list[dict[str, Any]]
) -> Path:
    """Write ``cake_stepfun_error_table.<arch>.json`` into ``directory``; returns the path."""
    path = Path(directory) / ERROR_TABLE_FILENAME.format(arch=arch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(error_table_document(header, rows), indent=1, sort_keys=True) + "\n"
    )
    return path


def expected_error_table_keys() -> set[tuple]:
    """The ``(stage, family, T, limit_mode, regime, input_set)`` keys the api suite records per arch: the
    whole-graph cells of :data:`API_GRAPH_CELLS` on two input sets, the routing rows per (T, regime), the
    unfinalized FC2 cells, the finalize and requant rows, and the ``n/a`` fc1 rows."""
    keys: set[tuple] = set()
    for _precision, family in PRECISION_FAMILY.items():
        for t in TOKENS:
            for mode, regime in API_GRAPH_CELLS:
                for input_set in INPUT_SETS:
                    keys.add(("output", family, t, mode, regime, input_set))
            keys.add(("fc1", family, t, "mixed", "ragged", 0))
        for t, mode, regime, input_set in API_FC2_PERMUTED_CELLS:
            keys.add(("fc2_permuted", family, t, mode, regime, input_set))
    for t in TOKENS:
        for regime in ROUTING_REGIMES:
            keys.add(("routing", "bf16", t, "7", regime, 0))
        keys.add(("finalize", "bf16", t, "7", "ragged", 0))
    for t, regime in API_REQUANT_CELLS:
        keys.add(("requant", "nvfp4_bf16tok", t, "7", regime, 0))
    return keys
