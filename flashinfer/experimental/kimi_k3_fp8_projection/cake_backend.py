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

Cake backend: Kimi-K3 serialized ``FP8_PB_WO`` projection GEMMs on SM100 / SM103.

The operator is one TP-local projection of ``nvidia/Kimi-K3-NVFP4`` whose
weight is serialized as ``FP8_PB_WO`` (E4M3 ``[N_pad128, K]`` plus the ModelOpt
128x128 FP32 ``weight_scale``): ``q_proj``, ``fused_qkvg``, ``in_proj_qkvgfab``,
``f_a``, ``f_b``, ``b_proj`` (KDA) and ``kv_a``, ``fused_qkv_a``, ``q_b``,
``kv_b``, ``o_proj`` (MLA), TP8 and TP1.  The complete call::

    a_q[m, k] = E4M3_rn(x[m, k] / s_a[m, k // 128]),  s_a = 2^ceil(log2(max(amax_128(x[m]), 1e-4) / 448))
    w2, s2    = requant_ue8m0(weight, weight_scale)   (once, at preparation; vLLM requant_weight_ue8m0)
    out       = bf16(sum_k a_q[m, k] s_a[m, k // 128] * w2[n, k] s2[n // 128, k // 128])   [M, n_valid]

runs as one or two generated Cake programs on the current stream:

* ``quant:u<units>`` -- the per-token 1x128 E4M3 quantization launch (DeepGEMM
  ``per_token_cast_to_fp8(use_ue8m0=True)`` bit-exact) writing the E4M3
  activation and its swizzled UE8M0 scale tiles into the caller-owned workspace;
* ``gemm_tstore`` / ``gemm`` -- the persistent 2-CTA block-scaled tcgen05 GEMM
  (M > 256 unless tabulated below) with the TMA-store epilogue for a 16-byte
  aligned output view whose row stride is a multiple of 8 elements, or the
  register epilogue for other strides (``_n192``: the 192-wide N-tile instance
  of the tabulated ``gemm_bn`` rows, e.g. the N = 576 ``kv_a`` family at
  M > 256, whose three 192-column tiles stream no padded columns; ``_sk``: the
  ordered stream-K instance of the tabulated ``gemm_sk`` rows, whose head CTA
  pairs run the first K half of the fractional last wave's tiles and hand the
  FP32 partial to the tail pair -- bit-exact with the plain schedule; ``_skf``:
  the fix-up stream-K instance of the tabulated ``gemm_skf`` rows, whose last
  wave plus fraction is cut into one contiguous K range per resident pair and
  whose finishing pair adds the other pairs' FP32 partials in ordinal order --
  the FP32 reduction order of those tiles differs from the plain schedule;
  ``_st``: the static one-tile instance of the tabulated ``gemm_static`` rows'
  192-wide 128-K TMA-store / staged-register programs, round 11 -- one tile per
  CTA pair, no cluster-launch-control work ring, admissible when the launch's
  CTA-pair tiles fit the resident pairs; bit-identical output; ``_gs``: the
  static grid-stride instance of the tabulated ``gemm_gs`` rows' 128-K
  TMA-store programs at bn 192 / 256, round 11 -- pair p of 2 min(T, resident
  pairs) CTAs owns tiles p, p + P', p + 2 P', ..., no cluster-launch-control
  work ring; bit-identical output), or
* ``decode:t<tok>_p<stages>[_fused][_res]`` -- the swap-AB split-K decode kernel
  (M <= 256, the single-N-tile families the measured table routes here up to
  16384 rows, and the mid-M rows -- 512 / 1024 / 2048 tokens of most K = 7168
  families -- whose under-filled 2-CTA GEMM grid the table measured slower
  than the token-tiled decode kernel; the ``_fused`` instances quantize the
  token tile in-CTA, so the quantization launch is skipped).

Host work is split exactly like the Cake production launcher:

* :func:`prepare_kimi_k3_fp8_projection_weights` -- once per weight: UE8M0
  requantization, 256-row padding, the 128x128 E4M3 tile order the TMA streams
  (one contiguous 32 KB box per pipeline stage) and the swizzled weight scale
  tiles.
* :func:`allocate_kimi_k3_fp8_projection_workspace` -- once per ``M``: the E4M3
  activation ``[M, K]`` and the auxiliary byte workspace (activation scale
  tiles and self-resetting per-tile counters, zero initialised once; split-K
  partial tiles, written before they are read).
* :func:`prepare_kimi_k3_fp8_projection` -- resolves the route once from the
  measured dispatch table (:mod:`.decode_table`; one table for both
  architectures with per-architecture overrides) and the device's cached
  architecture / SM count, and binds the launch sequence to the generated
  argument plans.  The returned runner's ``launch()`` performs no allocation
  and no host synchronisation and is CUDA-graph capturable.

The host dispatch reproduces the Cake dispatcher's measured table for the
representative ``(N, K)`` families; the generated-program export checks route,
configuration and bitwise output parity on every contract row.  See
``README.md`` in this package and flashinfer-ai/flashinfer#4568 (tracker #4254).
"""

from __future__ import annotations

import functools
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple, Optional

import torch
import tvm_ffi

from .cake_jit import (
    GEMM_KERNEL_KEY,
    GEMM_RSTAGED_KERNEL_KEY,
    GEMM_TSTORE_KERNEL_KEY,
    MODULES,
    Defines,
    decode_kernel_key,
    gemm_kernel_key,
    kernel_program,
    load_cake_kimi_k3_fp8_projection_module,
    quant_kernel_key,
    route_available,
)
from .decode_table import ARCHES, decode_table

# ---------------------------------------------------------------------------
# Fixed geometry of the generated programs
# ---------------------------------------------------------------------------

BLOCK = 128  # scale granularity along K (activations) and along N x K (weights)
BLOCK_M = 128  # activation rows per GEMM CTA (256 per CTA pair)
BLOCK_N = 256  # output columns per GEMM CTA pair
GEMM_BLOCK_N_NARROW = (
    192  # round 6 (lever L5): the narrow N-tile GEMM instance (table key ``gemm_bn``)
)
GEMM_N192_SUFFIX = (
    "_n192"  # kernel-key suffix of the 192-wide instance of any GEMM epilogue program
)
GEMM_K128_SUFFIX = "_k128"  # round 8 (lever L8): the 128-K operand hand-off instance of the TMA-store / staged-register programs (table key ``gemm_bk``; stem position after _n192, before the stream-K suffixes)
GEMM_BLOCK_N_WIDE = 384  # round 10 (CAKE-985 S3): the 128 x 384 pair-tile GEMM instance (table key ``gemm_bn`` 384; TMA-store / staged-register programs only, always the 128-K ring)
GEMM_N384_SUFFIX = "_n384"  # kernel-key suffix of the 384-wide instance (stem position of _n192; always followed by _k128, never by the stage / stream-K / prefetch tokens)
GEMM_G64_SUFFIX = "_g64"  # round 10 (CAKE-985 S3): the 64-row-tile raster group of the TMA-store / staged-register programs (table key ``gemm_gm``; stem position after the bn token, before _k128)
GEMM_GROUP_M = (
    32  # raster group (row tiles per group) of every program without the _g64 token
)
GEMM_GROUP_M_FORMS = (
    32,
    64,
)  # the raster groups the Cake GEMM instance supports (``gemm_ir``: gm)
GEMM_PDLB_SUFFIX = "_pdlb"  # round 11 (CAKE-985 T1a, lever _pdlb): the PDL-wait-after-weights form of the TMA-store / staged-register programs (table key ``gemm_pdlb``; the last stem token -- after the stream-K / _skp slot, before _pf<D>; no 384-wide form)
GEMM_ST_SUFFIX = "_st"  # round 11 (CAKE-985 T2-e, lever _st): the static one-tile form of the 192-wide 128-K TMA-store / staged-register programs (table key ``gemm_static``; the ``_cs<S>`` slot after _k128; no stream-K / stage / prefetch / _pdlb composition)
GEMM_GS_SUFFIX = "_gs"  # round 11 (CAKE-985 T6, lever _gs): the static grid-stride form of the 128-K TMA-store programs at bn 192 / 256 (table key ``gemm_gs``; the ``_cs<S>`` / ``_st`` slot after _k128; no stream-K / stage / prefetch / _pdlb / _st composition)
GEMM_N384_CELL_CONFLICTS = (
    "gemm_pf",
    "gemm_sk",
    "gemm_skf",
    "gemm_sks",
    "gemm_skp",
    "gemm_stages",
    "gemm_ts_slots",
    "gemm_pdlb",
    "gemm_static",
    "gemm_gs",
)  # round 10 (S3): GEMM-cell keys with no 384-wide form (the Cake launcher's ``N384_CELL_CONFLICTS``; ``gemm_bk`` 256 too; round 11 T1a: ``gemm_pdlb``; round 11 T2-e: ``gemm_static``; round 11 T6: ``gemm_gs``)
GEMM_STATIC_CELL_CONFLICTS = (
    "gemm_pf",
    "gemm_sk",
    "gemm_skf",
    "gemm_sks",
    "gemm_skp",
    "gemm_cs",
    "gemm_stages",
    "gemm_ts_slots",
    "gemm_pdlb",
    "gemm_gs",
)  # round 11 (CAKE-985 T2-e): GEMM-cell keys with no _st composition (the Cake launcher's ``GEMM_STATIC_CELL_CONFLICTS``; ``gemm_bn`` other than 192 and ``gemm_bk`` other than 128 too; round 11 T6: ``gemm_gs``)
GEMM_GS_CELL_CONFLICTS = (
    "gemm_pf",
    "gemm_sk",
    "gemm_skf",
    "gemm_sks",
    "gemm_skp",
    "gemm_cs",
    "gemm_stages",
    "gemm_ts_slots",
    "gemm_pdlb",
    "gemm_static",
)  # round 11 (CAKE-985 T6): GEMM-cell keys with no _gs composition (the Cake launcher's ``GEMM_GS_CELL_CONFLICTS``; ``gemm_bn`` other than 192 / 256 and ``gemm_bk`` other than 128 too)
GEMM_K128_RSTAGED_BUCKETS = (
    8,
    1024,
    2048,
    16384,
)  # round 8 (lever L8): the table buckets that hold a contract row with the 8-byte-aligned (``stride_pad`` 4) output stride, i.e. a staged-register launch (M 3 / 513 / 1000 / 1025 / 4097)
GEMM_K128_RSTAGED_STRIDE_PAD = 4  # round 8 (lever L8): the output row stride of those rows, in BF16 elements past ``n_valid`` (``gemm_k128_cell_epilogues``)
# Round 10 (CAKE-985, S1): the cluster split-K instance of the 192-wide 128-K TMA-store /
# staged-register programs (table key ``gemm_cs`` = CTA pairs per output tile; the Cake
# launcher's ``gemm_cluster_split_plan``).  Token ``_cs<S>`` (column form) / ``_cs<S>r``
# (row form) after ``_k128``; the first build has no stream-K, staging-token or prefetch
# form of it.
GEMM_CS_SUFFIX = "_cs"
# The row form S = 4 (``_cs4r``) is designed (gemm_cs_row_form) but not built in round 10.
GEMM_CS_FORMS = (2, 3)  # pair counts the Cake GEMM instance supports (gemm_ir cs)
GEMM_CS_PAD_WORDS = 4  # padded row stride of the exchanged FP32 blocks, in 32-bit words
GEMM_CS_TILE_ROWS = BLOCK_M  # rows of one CTA's accumulator half-tile
GEMM_CS_TS_COLS = 32  # columns per staged store chunk (column form: whole chunks)
GEMM_NUM_STAGES = 3  # operand stages of the untokened GEMM programs (round 9, K6 arm G1: 2 on the tabulated ``gemm_stages`` rows, token ``_s2``)
GEMM_TS_SLOTS = 1  # staging slots per epilogue group of the untokened GEMM programs (round 9: 4 on the tabulated ``gemm_ts_slots`` rows, token ``_ts4``; TMA-store programs only)
GEMM_STAGES_FORMS = (
    2,
    3,
)  # the operand-stage counts the Cake GEMM instance supports (``gemm_ir``)
GEMM_TS_SLOT_FORMS = (
    1,
    4,
)  # the staging-slot counts the Cake GEMM instance supports (4 needs the TMA-store epilogue)
_GEMM_INSTANCE_TOKEN = re.compile(
    r"(?:_s(?P<s>[0-9]+)|_ts(?P<ts>[0-9]+))(?=_|$)"
)  # ``_s<n>`` / ``_ts<n>`` (never ``_sk`` / ``_skf<n>`` / ``_sks<n>`` / ``_tstore``)
GEMM_SK_SUFFIX = "_sk"  # round 6 continuation 12 (lever SKO): the ordered stream-K instance of any GEMM epilogue program (table key ``gemm_sk``)
GEMM_SKF_SUFFIX = "_skf"  # round 6 continuation 17/18 (lever SKF): the fix-up stream-K instance of the TMA-store GEMM programs (table key ``gemm_skf``)
GEMM_SKF_MIN_ITERS = 4  # lever SKF: a pair's range of fewer K iterations cannot pay for the partial hand-offs
GEMM_SKF_MAX_SLOTS = 2  # lever SKF: contributor partial slots per tile the finisher adds (more contributors: plain schedule)
GEMM_SKF_SLOT_FORMS = (
    2,
    4,
    8,
)  # round 7: slot bounds of the shipped fix-up forms (``_skf`` = 2, ``_skf4`` / ``_skf8``)
GEMM_SKS_SUFFIX = "_sks"  # round 7 (lever SKS): the sliced fix-up stream-K instance (table key ``gemm_sks``; ``_sks<slots>``)
# round 10 (CAKE-985, lever SKP): the prefetched stream-K instance of the TMA-store GEMM programs (table key
# ``gemm_skp``: the slot bound 1 / 4 / 8 names ``_skp`` / ``_skp4`` / ``_skp8``; precedence ``_sk`` > ``_skp`` >
# ``_skf`` as the Cake launcher)
GEMM_SKP_SUFFIX = "_skp"
GEMM_SKP_SLOT_FORMS = (1, 4, 8)
# lever SKP: default iterations a contributor piece ends before its tile's finisher (table key ``gemm_skp_slack``)
GEMM_SKP_SLACK = 4
GEMM_SK_MIN_K_ITERS = 4  # lever SKO: a head / tail chunk of fewer than two K iterations cannot pay for the partial hand-off
GEMM_SK_FLAG_BYTES = 2048  # stream-K flag area: 512 u32 flags >= 2 x the largest tile count of a split region (SKO: pairs / 2 tail tiles; SKF: < 2 pairs SK tiles)
# lever SKP: one u32 flag per (SK tile, contributor ordinal, CTA rank) inside the stream-K flag area
GEMM_SKP_MAX_FLAGS = GEMM_SK_FLAG_BYTES // 4
GEMM_EPI_WARPS = (
    8  # epilogue warps per GEMM CTA (the head partial record is written per warp)
)
GEMM_SK_TILE_BYTES = (
    2 * GEMM_EPI_WARPS * 32 * 32 * 16
)  # lever SKO: one head partial tile (2 CTAs x 8 warps x 32 vectors x 32 lanes x 16 B = 256 KiB)
BLOCK_K = 256  # E4M3 elements per pipeline stage = two 128-K scale sets
CTA_GROUP = 2
WEIGHT_TILE_ROWS = 2 * BLOCK  # one CTA-pair N tile
WIDE_TILE_ROWS = (
    3 * BLOCK
)  # round 10 (S3): the 384-wide pair tile -- the widest ``bn`` of ``weight_scale_tiles_bn``
SF_TILE_ROWS = 128
SF_TILE_BYTES = 512
E4M3_MAX = 448.0
AMAX_FLOOR = 1e-4  # DeepGEMM per_token_cast_to_fp8 clamp
QUANT_WARPS = 4  # warps per quantization CTA; a half warp quantizes one 128-element block per unit
DECODE_MAX_M = 256  # rows above this use the persistent 2-CTA GEMM
DECODE_TABLE_BUCKETS = (
    1,
    8,
    64,
    256,
    512,
    1024,
    2048,
    4096,
    16384,
)  # M buckets of the measured dispatch table (buckets above DECODE_MAX_M
#    list only the families whose measured route differs from the plain GEMM)
DEC_W_BYTES = 128 * BLOCK_K  # one 128-row weight tile x 256 K per stage
DEC_SF_BYTES = 2048  # 2 K-sets x 512 B per operand
DEC_SMEM_CAP = 230400  # decode SMEM pool budget
DEC_MAX_STAGES = 4
DEC_XB_MAX_STAGES = 8  # BF16 ring depth cap of the decoupled fused decode variant
DEC_XQ_MAX_STAGES = 8  # FP8 token-ring depth cap (round 4 lever F; table key ``xq_stages``, adopted in round 7 on the 1,28,256 cells)
DEC_QUANT_WARPS = (
    8  # quantizing warps of the fused decode instance (narrow-unit divisibility rule)
)
DEC_RES_SLOTS = 4  # work items per CTA the resident decode instance can hold
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}


class DeviceFacts(NamedTuple):
    """The two device properties the host dispatch depends on (queried once per device)."""

    arch: str
    sm_count: int


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(device_index: int) -> DeviceFacts:
    """Architecture and SM count of ``cuda:device_index`` from one property query, cached per device."""
    props = torch.cuda.get_device_properties(device_index)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get((int(props.major), int(props.minor)))
    if arch is None:
        raise ValueError(
            "the Kimi-K3 FP8 projection requires compute capability 10.0 or 10.3 "
            f"(got {props.major}.{props.minor})"
        )
    return DeviceFacts(arch, int(props.multi_processor_count))


def device_facts(device: torch.device) -> DeviceFacts:
    return _device_facts(_device_index(device))


# Contract families (n_valid, K) of the Kimi-K3 FP8_PB_WO projections: TP8 first, then TP1.
PROJECTION_FAMILIES: dict[str, dict[str, tuple[int, int]]] = {
    "tp8": {
        "q_proj": (1536, 7168),
        "fused_qkvg": (6144, 7168),
        "in_proj_qkvgfab": (6284, 7168),
        "f_a": (12, 7168),
        "f_b": (1536, 128),
        "b_proj": (12, 7168),
        "kv_a": (576, 7168),
        "fused_qkv_a": (2112, 7168),
        "q_b": (2304, 1536),
        "kv_b": (3072, 512),
        "o_proj": (7168, 1536),
    },
    "tp1": {
        "q_proj": (12288, 7168),
        "fused_qkvg": (49152, 7168),
        "in_proj_qkvgfab": (49376, 7168),
        "f_a": (96, 7168),
        "f_b": (12288, 128),
        "b_proj": (96, 7168),
        "kv_a": (576, 7168),
        "fused_qkv_a": (2112, 7168),
        "q_b": (18432, 1536),
        "kv_b": (24576, 512),
        "o_proj": (7168, 12288),
    },
}


# ---------------------------------------------------------------------------
# Layout helpers (exact mirrors of the Cake reference recipe)
# ---------------------------------------------------------------------------


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round a positive FP32 tensor up to the next power of two (DeepGEMM ``ceil_to_ue8m0``)."""
    bits = x.abs().float().contiguous().view(torch.int32)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float32)


def ue8m0_byte(sf_pow2: torch.Tensor) -> torch.Tensor:
    """Biased exponent byte of a power-of-two FP32 scale (``sf == 2 ** (byte - 127)``)."""
    return (sf_pow2.contiguous().view(torch.int32) >> 23).to(torch.uint8)


def n_padded(n_valid: int) -> int:
    """Storage rows of a TP-local weight: the 128-row block padding rounded up to the 256-row pair tile."""
    return -(-int(n_valid) // WEIGHT_TILE_ROWS) * WEIGHT_TILE_ROWS


def n_padded_128(n_valid: int) -> int:
    """ModelOpt's serialized padding (128-row blocks)."""
    return -(-int(n_valid) // BLOCK) * BLOCK


def k_sets(K: int) -> int:
    """Scale K-sets per row block as the GEMM consumes them: two per 256-K stage, zero padded."""
    if K % BLOCK:
        raise ValueError(f"K must be a multiple of {BLOCK}, got {K}")
    return 2 * (-(-K // (2 * BLOCK)))


def swizzle_sf_128x4(sf_linear: torch.Tensor) -> torch.Tensor:
    """``[R, C]`` scale bytes -> flat 512-byte scale tiles (rows padded to 128, columns to a multiple of 4)."""
    R, C = sf_linear.shape
    Rp = -(-R // SF_TILE_ROWS) * SF_TILE_ROWS
    Cp = -(-C // 4) * 4
    padded = torch.zeros(
        (Rp // SF_TILE_ROWS, SF_TILE_ROWS, Cp),
        dtype=torch.uint8,
        device=sf_linear.device,
    )
    padded[:, :, :C] = torch.nn.functional.pad(sf_linear, (0, 0, 0, Rp - R)).reshape(
        Rp // SF_TILE_ROWS, SF_TILE_ROWS, C
    )
    # (tile, r_hi(4), r_lo(32), c_hi, c_lo(4)) -> (tile, c_hi, r_lo, r_hi, c_lo)
    v = padded.reshape(Rp // SF_TILE_ROWS, 4, 32, Cp // 4, 4).permute(0, 3, 2, 1, 4)
    return v.reshape(-1).contiguous()


def unswizzle_sf_128x4(sf_swizzled: torch.Tensor, R: int, C: int) -> torch.Tensor:
    """Inverse of :func:`swizzle_sf_128x4` -> ``[R, C]``."""
    Rp = -(-R // SF_TILE_ROWS) * SF_TILE_ROWS
    Cp = -(-C // 4) * 4
    v = sf_swizzled.reshape(Rp // SF_TILE_ROWS, Cp // 4, 32, 4, 4).permute(
        0, 3, 2, 1, 4
    )
    return v.reshape(Rp, Cp)[:R, :C]


def requant_weight_ue8m0(
    weight: torch.Tensor, scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """vLLM ``requant_weight_ue8m0`` on a serialized FP8_PB_WO weight: ``s2 = ceil_pow2(scale)``,
    ``w2 = E4M3_rn(weight * (scale / s2))``.  Returns ``(w2 float8_e4m3fn [N, K], s2 fp32 [N/128, K/128])``."""
    n, k = weight.shape
    sf = scale.reshape(n // BLOCK, k // BLOCK).float()
    s2 = ceil_to_ue8m0(sf)
    ratio = (sf / s2).view(n // BLOCK, 1, k // BLOCK, 1)
    w2 = (
        (weight.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK) * ratio)
        .to(torch.float8_e4m3fn)
        .view(n, k)
    )
    return w2, s2


def weight_scale_tiles(sf_bytes_128: torch.Tensor, K: int) -> torch.Tensor:
    """``[N_pad/128, K/128]`` UE8M0 bytes -> flat CTA-pair scale tiles ``[n_tiles][k_sets][2 halves][512 B]``.

    Every 32-element MX block of a 128x128 weight block carries the block's byte (the block-scaled MMA applies
    the same scale to the four K-groups of a 128-K set); ``n_tiles = N_pad / 256``, ``k_sets = k_sets(K)``."""
    n_blocks, kb = sf_bytes_128.shape
    if n_blocks % 2:
        raise ValueError(
            "N_pad must be a multiple of 256 (two 128-row blocks per CTA pair)"
        )
    ks = k_sets(K)
    rows = sf_bytes_128.repeat_interleave(BLOCK, dim=0)  # [N_pad, K/128]
    cols = torch.zeros((rows.shape[0], ks * 4), dtype=torch.uint8, device=rows.device)
    cols[:, : kb * 4] = rows.repeat_interleave(4, dim=1)
    tiles = swizzle_sf_128x4(cols)  # flat (128-row tile, k_set, 512 B)
    return (
        tiles.reshape(n_blocks // 2, 2, ks, SF_TILE_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


def weight_scale_tiles_bn(sf_bytes_128: torch.Tensor, K: int, bn: int) -> torch.Tensor:
    """Round 6 (lever L5): CTA-pair scale tiles of the GEMM instance with ``bn`` output columns per pair
    (192; 256 reproduces :func:`weight_scale_tiles`; round 10 S3: 384): ``[n_tiles][k_sets][blocks][512 B]`` with
    ``blocks = ceil(bn / 128)`` (two for 192 / 256, three for 384).

    The block-scaled MMA reads the B scales of logical N rows ``[128 b, 128 b + 128)`` from TMEM column block
    ``b``, so tile ``t`` carries the per-row bytes of weight rows ``[t bn, t bn + bn)`` in N order across its
    128-lane blocks (lanes past ``bn`` zero); rows past the stored ``N_pad`` are zero, ``n_tiles = ceil(N_pad / bn)``.
    The 192 / 256 layouts are byte-identical to the round-6 two-block form (``test_weight_scale_tiles_bn_layout``)."""
    if bn <= 0 or bn > WIDE_TILE_ROWS or bn % 32:
        raise ValueError(
            f"bn must be a positive multiple of 32 up to {WIDE_TILE_ROWS}, got {bn}"
        )
    blocks = -(-bn // BLOCK)
    tile_rows = blocks * BLOCK
    n_blocks, kb = sf_bytes_128.shape
    n_rows = n_blocks * BLOCK
    ks = k_sets(K)
    n_tiles = -(-n_rows // bn)
    rows = sf_bytes_128.repeat_interleave(BLOCK, dim=0)  # [N_pad, K/128]
    per_row = torch.zeros((n_tiles * bn, ks * 4), dtype=torch.uint8, device=rows.device)
    per_row[:n_rows, : kb * 4] = rows.repeat_interleave(4, dim=1)
    tiles_rows = torch.zeros(
        (n_tiles, tile_rows, ks * 4), dtype=torch.uint8, device=rows.device
    )
    tiles_rows[:, :bn] = per_row.view(n_tiles, bn, ks * 4)
    tiles = swizzle_sf_128x4(
        tiles_rows.reshape(n_tiles * tile_rows, ks * 4)
    )  # flat (128-lane block, k_set, 512 B)
    return (
        tiles.reshape(n_tiles, blocks, ks, SF_TILE_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


def weight_scale_sf_rows(bn: int) -> int:
    """Rows of 128 B per K set in the B scale-tile view of the ``bn``-wide GEMM instance (the kernel's SFB TMA box):
    ``4 * ceil(bn / 128)`` -- 8 for 192 / 256, 12 for 384 (round 10 S3)."""
    return 4 * (-(-int(bn) // BLOCK))


def activation_sf_rows(M: int, sf_rows: int = SF_TILE_ROWS) -> int:
    """Row slots of the activation scale-tile workspace: one 128-slot tile per ``sf_rows`` activation rows,
    rounded up to an even tile count."""
    tiles = -(-int(M) // int(sf_rows))
    return (tiles + tiles % 2) * SF_TILE_ROWS


def activation_sf_workspace_bytes(M: int, K: int, sf_rows: int = SF_TILE_ROWS) -> int:
    return activation_sf_rows(M, sf_rows) * k_sets(K) * 4


# ---------------------------------------------------------------------------
# Dispatch rules (mirrors of the Cake dispatcher)
# ---------------------------------------------------------------------------


def quant_units(M: int, k_blocks: int, sm_count: int) -> int:
    """K blocks per half warp of the quantization launch: 1 for M <= 256 (maximal parallelism), 4 / 2 / 8
    for large M when they divide the K blocks and keep at least four CTAs per SM of the device busy."""
    if M <= DECODE_MAX_M:
        return 1
    for units in (4, 2, 8):
        if (
            k_blocks % units == 0
            and (M * k_blocks) // units >= int(sm_count) * QUANT_WARPS * 2 * 4
        ):
            return units
    return 1


def table_quant_units(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 8 (table key ``quant_units``): the quantization launch width a measured cell pins for every two-launch
    route of the cell (decode and GEMM routes alike), 0 when the cell carries none (the ``quant_units`` rule applies)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    return int((entry or {}).get("quant_units", 0))


def launch_quant_units(M: int, k_blocks: int, table_units: int, sm_count: int) -> int:
    """K blocks per half warp of the quantization launch of a two-launch route: the shape's cell width ``table_units``
    when it pins one (host mirror of the Cake launcher: the width must divide the K blocks), the ``quant_units`` rule
    otherwise."""
    units = int(table_units or 0)
    if units:
        if int(k_blocks) % units:
            raise ValueError(
                f"decode table entry quant_units {units} does not divide {k_blocks} K blocks"
            )
        return units
    return quant_units(M, k_blocks, sm_count)


def _decode_stage_geometry(
    tok: int, fused: bool, resident: bool, xb_stages: int, xq_stages: int
) -> tuple[int, int, int, int]:
    """``(stage_bytes, xb_ring_bytes, xq_ring_bytes, res_bytes)`` of a decode instance (the Cake ``decode_ir`` SMEM rule).

    ``xb_stages`` > 0 is the decoupled fused variant: the BF16 token tiles stream through their own ``xb_stages``-deep
    ring instead of sharing the W / X stage; ``xq_stages`` > 0 (round 4 lever F, table key ``xq_stages``; needs the
    BF16 ring) moves the FP8 token tile and its scales out of the W stage into an ``xq_stages``-deep ring, so a W stage
    holds the weight tile plus one 1 KB scale set."""
    tok_rows = max(tok, 32)
    xb_bytes = tok * 512 if (fused and not resident) else 0
    xb_ring = bool(fused and not resident and xb_stages > 0)
    xq_ring = bool(fused and not resident and xq_stages > 0)
    stage_x_bytes = 0 if (resident or xq_ring) else tok_rows * BLOCK_K
    stage_bytes = (
        DEC_W_BYTES
        + stage_x_bytes
        + (1024 if xq_ring else DEC_SF_BYTES)
        + (0 if xb_ring else xb_bytes)
    )
    xb_ring_bytes = xb_stages * xb_bytes if xb_ring else 0
    xq_ring_bytes = xq_stages * (tok_rows * BLOCK_K + 1024) if xq_ring else 0
    res_bytes = DEC_RES_SLOTS * (tok_rows * BLOCK_K + 1024) if resident else 0
    return stage_bytes, xb_ring_bytes, xq_ring_bytes, res_bytes


def decode_module_stages(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    xb_stages: int = 0,
    epi_chunk: int = 32,
    xq_stages: int = 0,
    epi_groups: int = 1,
) -> int:
    """Pipeline depth of the physical decode instance after its SMEM clamp (the Cake ``decode_ir`` rule); see
    ``_decode_stage_geometry`` for the ring forms.  Round 8 (table key ``epi_groups``): a second epilogue warp group
    doubles the epilogue staging bytes on the 64/128-token tiles (one group below 64 tokens, as the dispatcher resolves it)."""
    stage_bytes, xb_ring_bytes, xq_ring_bytes, res_bytes = _decode_stage_geometry(
        tok, fused, resident, xb_stages, xq_stages
    )
    epi_bytes = (
        (1 if tok < 64 else int(epi_groups)) * min(int(epi_chunk), 32, tok) * 128 * 4
    )
    return max(
        1,
        min(
            stages,
            (DEC_SMEM_CAP - epi_bytes - res_bytes - xb_ring_bytes - xq_ring_bytes)
            // stage_bytes,
        ),
    )


# Co-resident cluster capacity of the cluster split-K decode instances per cluster size
# (``cuOccupancyMaxActiveClusters`` of the ``_cs<C>`` instance; one CTA per SM, so it depends only on the GPC topology).
# A persistent cluster grid larger than ``C x capacity`` serialises whole clusters into a second pass.  Measured on
# B200 / B300; mirrors the source repository's table.
DECODE_MAX_ACTIVE_CLUSTERS: dict[str, dict[int, int]] = {
    # 9..16 = non-portable clusters (round 6, lever C16), measured on the t16 fused small-inbox instance on both GPUs
    "sm_100a": {
        2: 74,
        3: 45,
        4: 33,
        5: 26,
        6: 22,
        7: 15,
        8: 15,
        9: 15,
        10: 11,
        12: 7,
        14: 7,
        16: 7,
    },  # B200, 148 SMs
    "sm_103a": {
        2: 74,
        3: 45,
        4: 33,
        5: 26,
        6: 22,
        7: 15,
        8: 15,
        9: 15,
        10: 11,
        12: 7,
        14: 7,
        16: 7,
    },  # B300 / GB300 (152 SMs; same per-GPC cluster capacity as B200 -- measured values)
}


def decode_cs_alias_fits(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    csplit: int,
    xb_stages: int = 0,
    xq_stages: int = 0,
    epi_groups: int = 1,
) -> bool:
    """Round 5: True when the one-round cluster exchange inbox + staging blocks
    ``(2C - 1) x 128 x (chunk | 1) x 4`` fit inside the physical instance's pipeline
    stages (host mirror of the Cake ``decode_cs_alias_fits`` rule); the fused view of a
    wide tile (1 stage at t128) does not fit and keeps the small-inbox exchange."""
    if csplit < 2:
        return False
    stage_bytes, _xb_ring_bytes, _xq_ring_bytes, _res_bytes = _decode_stage_geometry(
        tok, fused, resident, xb_stages, xq_stages
    )
    module_stages = decode_module_stages(
        tok,
        stages,
        fused,
        resident,
        xb_stages,
        xq_stages=xq_stages,
        epi_groups=epi_groups,
    )
    tok_per_cta = -(-tok // csplit)  # ceil(tok / csplit)
    chunk = (
        -(-tok_per_cta // 4) * 4
    )  # round the per-CTA row count up to a multiple of 4
    return (2 * csplit - 1) * 128 * (chunk | 1) * 4 <= module_stages * stage_bytes


def decode_cs_inbox_stages(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    csplit: int,
    xb_stages: int = 0,
    epi_chunk: int = 32,
    xq_stages: int = 0,
    epi_groups: int = 1,
) -> int:
    """Physical pipeline depth of the small-inbox (non-aliased) cluster split-K instance: the clamped depth of
    ``decode_module_stages`` reduced until a 4-row inbox round, ``(2C - 1) x 128 x 5 x 4`` bytes, fits next to the
    stages (host mirror of the Cake ``decode_cs_inbox_stages`` rule; the 7..16-wide clusters of round 6 give up one
    t16 stage)."""
    if csplit < 2:
        return decode_module_stages(
            tok, stages, fused, resident, xb_stages, epi_chunk, xq_stages, epi_groups
        )
    stage_bytes, xb_ring_bytes, xq_ring_bytes, res_bytes = _decode_stage_geometry(
        tok, fused, resident, xb_stages, xq_stages
    )
    module_stages = decode_module_stages(
        tok, stages, fused, resident, xb_stages, epi_chunk, xq_stages, epi_groups
    )
    need = 5 * (2 * csplit - 1) * 128 * 4
    while (
        module_stages > 1
        and DEC_SMEM_CAP
        - module_stages * stage_bytes
        - res_bytes
        - xb_ring_bytes
        - xq_ring_bytes
        < need
    ):
        module_stages -= 1
    return module_stages


def decode_cs_small_inbox_rounds(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    csplit: int,
    xb_stages: int = 0,
    xq_stages: int = 0,
    epi_groups: int = 1,
) -> int:
    """Round 5: exchange rounds of the small (non-aliased) cluster inbox next to the
    physical pipeline stages (host mirror of the Cake ``decode_cs_small_inbox_rounds``
    rule).  The aliased one-round exchange is selected only when this is > 1: with 4-8
    rows per rank the small inbox already holds the owner range in one round, and the
    all-rank ordering the aliased exchange needs costs ~1.6 us at C8."""
    if csplit < 2:
        return 1
    stage_bytes, xb_ring_bytes, xq_ring_bytes, res_bytes = _decode_stage_geometry(
        tok, fused, resident, xb_stages, xq_stages
    )
    module_stages = decode_module_stages(
        tok,
        stages,
        fused,
        resident,
        xb_stages,
        xq_stages=xq_stages,
        epi_groups=epi_groups,
    )
    need = 5 * (2 * csplit - 1) * 128 * 4
    while (
        module_stages > 1
        and DEC_SMEM_CAP
        - module_stages * stage_bytes
        - res_bytes
        - xb_ring_bytes
        - xq_ring_bytes
        < need
    ):
        module_stages -= 1
    budget = (
        DEC_SMEM_CAP
        - module_stages * stage_bytes
        - res_bytes
        - xb_ring_bytes
        - xq_ring_bytes
    )
    tpc = -(-tok // csplit)
    chunk = min(-(-tpc // 4) * 4, (budget // ((2 * csplit - 1) * 128 * 4) - 1) // 4 * 4)
    return -(-tpc // max(chunk, 4))


def decode_cluster_capacity(arch: str, csplit: int) -> int:
    """Co-resident cluster capacity of ``arch`` for cluster size ``csplit`` (tabulated; raises when not measured)."""
    table = DECODE_MAX_ACTIVE_CLUSTERS.get(arch, {})
    if csplit not in table:
        raise ValueError(
            f"cluster capacity of {arch} for csplit {csplit} is not tabulated (DECODE_MAX_ACTIVE_CLUSTERS)"
        )
    return int(table[csplit])


@dataclass(frozen=True)
class DecodeConfig:
    """Resolved decode route of one ``(M, n_tiles128, num_k_iters)`` on one architecture."""

    tok: int
    split: int
    fused: bool
    resident: bool
    persist: bool
    stages: int  # requested pipeline depth (host rule)
    module_stages: int  # physical instance depth after the SMEM clamp
    m_tiles: int
    tiles: int
    total_work: int
    grid: int
    tok_per_cta: int
    xb_stages: int = (
        0  # decoupled BF16 ring depth (fused, non-resident); 0 = coupled staging
    )
    xq_stages: int = 0  # round 4 (lever F) FP8 token-ring depth (table key ``xq_stages``; needs the BF16 ring); 0 = FP8 tile inside the W stage
    qlanes: int = (
        16  # lanes per quantization unit (16 = half-warp units, 8 / 4 = narrow units)
    )
    csplit: int = 1  # round 5: K split across the CTAs of one cluster (== split); the partials meet in SMEM (DSM)
    cs_alias: bool = False  # round 5: the DSM inbox aliases the dead pipeline stages (one exchange round); only when every CTA owns one work item
    epi_chunk: int = 32  # round 6: epilogue staging rows per flush (table key ``epi_chunk``; 16 frees SMEM for the 5-stage t32 ring)
    pf: int = 0  # round 6 (lever P): weight-tile L2 prefetch distance in stages (table key ``pf``; 0 = off)
    pw: int = 0  # round 8 (lever PW; adopted round 11 C20 on the sm_103a 192,2,64 cell): the first work item's W / SFW stream is L2-prefetched ``pw`` stages deep before the PDL wait (table key ``pw``; 0 = off; names the ``_pw<D>`` program; no data-path, launch-argument or SMEM effect)
    mc: int = 1  # round 6 (lever M): m tiles of one N tile per cluster sharing the W stage through TMA multicast (table key ``mc``; 1 = off)
    pfx: int = 0  # round 6 next loop (lever PX): BF16 token-tile L2 prefetch distance in stages (table key ``pfx``; fused, non-resident rows only; 0 = off)
    tstore: bool = False  # round 6 (lever E1): the split-1 epilogue stores BF16 through TMA (table key ``tstore``; the launch still needs a 16-byte-aligned output view)
    dpf: bool = False  # round 6 (lever Q1; adopted round 8): the tensor-map descriptors are prefetched in the kernel prelude before the first TMA issue (table key ``dpf``; names the ``_dpf`` program; no data-path, launch-argument or SMEM effect)
    pfi: int = 0  # round 6 continuation 7 (lever PI-W): cross-item L2 prefetch of the next work item's first W/SFW stages by the load warp (table key ``pfi``; multi-item rows with pf > 0; 0 = off)
    qwarps: int = DEC_QUANT_WARPS  # round 6 continuation 8 (lever QW16): quantizing warps of the fused instance (table key ``qwarps``; 8 = the round-3 default, 16 on the fused 16384 rows)
    xbh: bool = False  # round 6 continuation 9 (lever XBH): half-slot BF16 ring -- the two 128-K blocks of a stage are loaded and released separately (table key ``xbh``; fused ring rows with narrow units)
    qer: bool = False  # round 6 continuation 10 (lever QER): early half-slot release -- each quantizing warp frees its BF16 half slot right after its register loads, before the conversion (table key ``qer``; xbh rows only)
    xp: bool = False  # round 7 (lever XP): the cluster split-K exchange issues its DSM copies from one lane per peer in parallel (table key ``xp``; ``csplit`` > 1 rows only; bit-exact)
    acc_bufs: int = 1  # round 8 (lever ACC_BUFS): TMEM accumulator sets of the instance (table key ``acc_bufs``; 1 or 2; names the ``_a<n>`` program; no SMEM effect)
    epi_groups: int = 1  # round 8 (lever EPI_GROUPS): epilogue warp groups (table key ``epi_groups``; 1, or 2 on the 64/128-token tiles; names the ``_e<n>`` program and doubles the epilogue staging bytes of the SMEM clamp)
    quant_units: int = 0  # round 8 (lever QUANT_UNITS): the cell's quantization launch width for a two-launch route (table key ``quant_units``; 0 = the ``quant_units`` rule); no effect on the decode program

    @property
    def tok_rows(self) -> int:
        return max(self.tok, 32)

    @property
    def kernel_key(self) -> str:
        """Kernel key of the register-epilogue program (the table row's instance without the output-view-dependent TMA store)."""
        return self.kernel_key_for(False)

    def kernel_key_for(self, tma_store: bool) -> str:
        """Kernel key of the launched program: ``_tso`` when the row's ``tstore`` applies to a TMA-eligible output view."""
        return decode_kernel_key(
            self.tok,
            self.module_stages,
            self.fused,
            self.resident,
            self.xb_stages,
            self.qlanes,
            self.csplit,
            cs_alias=self.cs_alias,
            xq_stages=self.xq_stages,
            epi_chunk=self.epi_chunk,
            pf=self.pf,
            pw=self.pw,
            mc=self.mc,
            pfx=self.pfx,
            tstore=bool(tma_store)
            and self.tstore
            and self.split == 1
            and self.csplit == 1,
            dpf=self.dpf,
            pfi=self.pfi,
            qwarps=self.qwarps,
            xbh=self.xbh,
            qer=self.qer,
            xp=self.xp,
            acc_bufs=self.acc_bufs,
            epi_groups=self.epi_groups,
        )


def decode_table_entry(
    M: int, n_tiles128: int, num_k_iters: int, arch: str
) -> Optional[dict[str, Any]]:
    """Measured table entry for the shape, or ``None`` when the architecture / family is not tabulated."""
    if arch not in ARCHES:
        return None
    per_arch = decode_table(arch)
    bucket = next((b for b in DECODE_TABLE_BUCKETS if b >= M), None)
    if bucket is None:
        return None
    return per_arch.get(f"{int(n_tiles128)},{int(num_k_iters)},{bucket}")


def decode_config(
    M: int, n_tiles128: int, num_k_iters: int, arch: str, sm_count: int
) -> Optional[DecodeConfig]:
    """Decode route of the shape from the measured table (``None`` = quantization launch + GEMM).

    ``(N, K)`` families the table does not cover take the GEMM route (the Cake dispatcher falls back to its
    calibrated cost model for uncovered families up to ``DECODE_MAX_M`` rows; every representative Kimi-K3 family
    is covered on both architectures).  Above ``DECODE_MAX_M`` rows the decode route is taken only where the table
    says so: the narrow-N families up to 16384 rows (whose activation stream the 2-CTA GEMM cannot spread over the
    SMs) and the tabulated mid-M rows (512 / 1024 / 2048 tokens, round 7) whose GEMM grid is under-filled."""
    M = int(M)
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "decode":
        return None
    tok, split, fused = int(entry["tok"]), int(entry["split"]), bool(entry["fused"])
    persist = bool(entry.get("persist", True))
    split = max(1, min(split, int(num_k_iters)))
    # Table key ``csplit`` (round 5): the K split runs across the C CTAs of one cluster and the FP32 partials meet
    # in the owning CTA's shared memory (no gmem partials, no counters); ``split`` == C and the persistent grid is a
    # whole number of clusters.
    csplit = int(entry.get("csplit", 1))
    if csplit > 1:
        if csplit > 16 or csplit > tok or csplit > int(num_k_iters):
            raise ValueError(
                f"decode table entry csplit {csplit} needs 2 <= C <= min(16, tok {tok}, num_k_iters {num_k_iters})"
            )
        split = csplit
    tok_rows = max(tok, 32)
    # Table key ``epi_chunk`` (round 6): epilogue staging rows per flush (32, 16 or 8; a smaller chunk frees SMEM for a
    # deeper ring at the cost of more flush barriers per item).
    epi_chunk = min(int(entry.get("epi_chunk", 32)), tok)
    epi_bytes = epi_chunk * 128 * 4
    xb_bytes = tok * 512
    stage_bytes_ring = DEC_W_BYTES + tok_rows * BLOCK_K + DEC_SF_BYTES
    # Fused variant: the BF16 token tiles either share the W / X stage (coupled, ``xb_stages`` 0) or stream through
    # their own ring (table key ``xb_stages`` = "auto" | N): "auto" picks the (stages, ring) split of the SMEM budget
    # with the largest smaller depth (the W-stage round trip and the activation fetch are both depth-bound).
    xb_mode = entry.get("xb_stages", 0) if fused else 0
    xb_stages = 0
    stages = 0
    if xb_mode not in (0, "0", "", "none", None):
        if xb_mode == "auto":
            best = None
            for cand_stages in range(2, DEC_MAX_STAGES + 1):
                cand_xb = min(
                    DEC_XB_MAX_STAGES,
                    (DEC_SMEM_CAP - epi_bytes - cand_stages * stage_bytes_ring)
                    // xb_bytes,
                )
                if cand_xb < 1:
                    continue
                score = (min(cand_stages, cand_xb), cand_stages)
                if best is None or score > best[0]:
                    best = (score, cand_stages, cand_xb)
            if best is not None:
                _, stages, xb_stages = best
        else:
            xb_stages = int(xb_mode)
            stages = max(
                2,
                min(
                    DEC_MAX_STAGES,
                    (DEC_SMEM_CAP - epi_bytes - xb_stages * xb_bytes)
                    // stage_bytes_ring,
                ),
            )
    if xb_stages == 0:
        stage_bytes = stage_bytes_ring + (xb_bytes if fused else 0)
        stages = max(2, min(DEC_MAX_STAGES, (DEC_SMEM_CAP - epi_bytes) // stage_bytes))
    # Table key ``xq_stages`` (round 4 lever F, adopted in round 7 on the t16 cluster split-K M = 256 cells): the FP8 token
    # tiles leave the W stage for their own ring ("auto" | N; needs the BF16 ring).  "auto" splits the budget over the three
    # rings for the largest smallest depth, then the deepest W ring, then the most slots (host mirror of the Cake
    # ``decode_config`` rule; a table ``stages`` pin is the Cake ``force_stages``).
    xq_mode = entry.get("xq_stages", 0) if (fused and xb_stages > 0) else 0
    xq_stages = 0
    if xq_mode not in (0, "0", "", "none", None):
        xq_bytes = tok_rows * BLOCK_K + 1024
        w_only_bytes = DEC_W_BYTES + 1024
        budget = DEC_SMEM_CAP - epi_bytes
        force_stages = int(entry["stages"]) if entry.get("stages") else 0
        if xq_mode == "auto":
            best3 = None
            for cw in range(2, DEC_MAX_STAGES + 1):
                if force_stages and cw != force_stages:
                    continue
                for cq in range(1, DEC_XQ_MAX_STAGES + 1):
                    cb = min(
                        DEC_XB_MAX_STAGES,
                        (budget - cw * w_only_bytes - cq * xq_bytes) // xb_bytes,
                    )
                    if xb_mode != "auto":
                        if cb < int(xb_mode):
                            continue  # an explicit BF16 depth must fit next to this (W, FP8) pair
                        cb = int(xb_mode)
                    if cb < 1:
                        continue
                    score3 = (min(cw, cq, cb), cw, cq + cb)
                    if best3 is None or score3 > best3[0]:
                        best3 = (score3, cw, cq, cb)
            if best3 is not None:
                _, stages, xq_stages, xb_stages = best3
        else:
            xq_stages = int(xq_mode)
            if force_stages:
                stages = force_stages
            else:
                stages = max(
                    2,
                    min(
                        DEC_MAX_STAGES,
                        (
                            budget
                            - xq_stages * xq_bytes
                            - (xb_stages if xb_mode != "auto" else 1) * xb_bytes
                        )
                        // w_only_bytes,
                    ),
                )
            if xb_mode == "auto":
                xb_stages = min(
                    DEC_XB_MAX_STAGES,
                    (budget - stages * w_only_bytes - xq_stages * xq_bytes) // xb_bytes,
                )
            if (
                xb_stages < 1
                or stages * w_only_bytes + xq_stages * xq_bytes + xb_stages * xb_bytes
                > budget
            ):
                raise ValueError(
                    f"decode table entry: W {stages} x {w_only_bytes} + FP8 ring {xq_stages} x {xq_bytes} + BF16 ring "
                    f"{xb_stages} x {xb_bytes} B exceed the {budget} B budget ({entry})"
                )
    # Table key ``stages`` (round 6, lever D): the row pins the ring depth past ``DEC_MAX_STAGES`` (5 x 43008 B + an 8 KB
    # epilogue chunk fit the pool at t32); the physical instance still applies the SMEM clamp (``decode_module_stages``).
    if entry.get("stages"):
        stages = int(entry["stages"])
    # Table key ``pf`` (round 6, lever P): the weight tiles (+ weight scales) are prefetched into L2 ``pf`` stages ahead.
    pf = int(entry.get("pf", 0))
    # Table key ``pw`` (round 8, lever PW; adopted in round 11 C20 on the sm_103a 192,2,64 cell): the first work item's W / SFW
    # stream is L2-prefetched ``pw`` stages deep before the PDL wait (host mirror of the Cake ``decode_config`` rule: any decode
    # row, no constraint); the lever names its own program (``_pw<D>``) and changes no data path, launch argument or SMEM byte.
    pw = int(entry.get("pw", 0))
    m_tiles = -(-M // tok)
    tiles = int(n_tiles128) * m_tiles
    # Table key ``mc`` (round 6, lever M): the C m tiles of one N tile run as a cluster and share the W stage through
    # TMA multicast (each rank streams 1 / C of the weight bytes).  Whole m-tile groups, an unsplit K, no cluster split-K;
    # one work item = (N tile, m-tile group).  Host mirror of the Cake ``decode_config`` rule.
    mc = int(entry.get("mc", 1))
    if mc > 1 and (mc not in (2, 4, 8) or csplit > 1 or split != 1):
        raise ValueError(
            f"decode table entry mc {mc} needs C in (2, 4, 8), split 1 and no csplit (got {entry})"
        )
    if mc > 1 and m_tiles % mc:
        # A bucket spans row counts with different m-tile counts (the 256 bucket serves M = 65..256 at t128): rows without
        # whole m-tile groups take the plain instance of the route (no multicast cluster, no prefetch; host mirror of the
        # Cake rule -- the exported program set only carries the instances the contract rows exercise).
        mc = 1
        pf = 0
    total_work = tiles * split if mc == 1 else int(n_tiles128) * (m_tiles // mc)
    # Table key ``grid`` (round 4): a balanced persistent CTA count (e.g. 128 CTAs for 256 work items) instead of one
    # CTA per SM; the round-4 A/B of the 16384-row buckets preferred 128 x 2 items over 148 x 1.73.
    grid = (
        min(total_work, int(entry.get("grid") or sm_count)) if persist else total_work
    )
    if csplit > 1:
        # Whole clusters, and no more clusters than the GPCs co-schedule (a second pass of clusters doubles the time).
        grid = csplit * max(
            1, min(grid // csplit, decode_cluster_capacity(arch, csplit))
        )
    if mc > 1:
        # Lever M: ``grid`` counted cluster items so far; C CTAs per cluster, no more clusters than co-schedule.
        grid = mc * max(
            1, min(grid, int(sm_count) // mc, decode_cluster_capacity(arch, mc))
        )
    resident = (
        bool(entry.get("resident", False))
        and fused
        and int(num_k_iters) == 1
        and split == 1
        and tok <= 64
        and -(-total_work // grid) <= DEC_RES_SLOTS
        and csplit == 1
        and mc == 1
    )
    if resident:
        xb_stages = 0  # resident tiles are fetched once; no ring
        xq_stages = 0
    # Narrow quantization units (table key ``qlanes`` 4 / 8): every lane group must own a unit each stage, so the
    # width is doubled until the units divide evenly over the quantizing warps.
    qlanes = int(entry.get("qlanes", 16)) if fused and not resident else 16
    # Table key ``qwarps`` (round 6 continuation 8, lever QW16): quantizing warps of the fused instance (host mirror of the Cake
    # ``decode_config`` rule: fused rows only; the narrow-unit rule below divides by the row's warp count).
    qwarps = int(entry.get("qwarps", DEC_QUANT_WARPS)) if fused else DEC_QUANT_WARPS
    while qlanes < 16 and (2 * tok) % (qwarps * (32 // qlanes)):
        qlanes *= 2
    # Table key ``xbh`` (round 6 continuation 9, lever XBH): half-slot BF16 ring (host mirror of the Cake ``decode_config`` rule:
    # fused rows with a decoupled ring and narrow units only).
    xbh = (
        bool(entry.get("xbh", False))
        and fused
        and not resident
        and xb_stages > 0
        and qlanes != 16
    )
    # Table key ``qer`` (round 6 continuation 10, lever QER): early half-slot release (host mirror of the Cake ``decode_config`` rule:
    # xbh rows only; the kernel instance validates the one-unit-per-lane-group split).
    qer = bool(entry.get("qer", False)) and xbh
    # Table key ``xp`` (round 7, lever XP): the cluster split-K exchange issues its DSM copies from one lane per peer in
    # parallel instead of one lane walking the peers (host mirror of the Cake ``decode_config`` rule: csplit > 1 rows only;
    # the same partials land in the same inbox slots, so the reduction is bit-exact with the serial issue).
    xp = bool(entry.get("xp", False)) and csplit > 1
    # Table key ``tstore`` (round 6, lever E1): the split-1 epilogue stores BF16 through TMA; the instance has no TMA path
    # for the split-K / cluster reductions.
    tstore = bool(entry.get("tstore", False))
    if tstore and (split != 1 or csplit > 1):
        raise ValueError(
            f"decode table entry tstore needs split 1 and no cluster split-K (got {entry})"
        )
    # Table key ``dpf`` (round 6, lever Q1; adopted in round 8): the tensor-map descriptors are prefetched in the kernel
    # prelude before the first TMA issue (host mirror of the Cake ``decode_config`` rule: any decode row, no constraint);
    # the lever names its own program (``_dpf``) and changes no data path, launch argument or SMEM byte.
    dpf = bool(entry.get("dpf", False))
    # Table key ``pfx`` (round 6 next loop, lever PX): the BF16 token tile of the fused variant is prefetched into L2
    # ``pfx`` stages ahead of its TMA load (its DRAM access then precedes the weight burst instead of queueing behind
    # it); fused, non-resident rows only (host mirror of the Cake ``decode_config`` rule).
    pfx = int(entry.get("pfx", 0)) if (fused and not resident) else 0
    # Table key ``pfi`` (round 6 continuation 7, lever PI-W): during the last ``pf`` stages of a work item the load warp
    # also prefetches the NEXT item's first W/SFW tiles into L2 (host mirror of the Cake ``decode_config`` rule: pf > 0).
    pfi = int(entry.get("pfi", 0)) if pf > 0 else 0
    # Table keys ``acc_bufs`` / ``epi_groups`` (round 8, levers ACC_BUFS / EPI_GROUPS): TMEM accumulator sets of the instance
    # and epilogue warp groups (host mirror of the Cake ``decode_config`` / ``decode_ir`` rules: 1 or 2 each; two epilogue
    # groups only on the 64/128-token tiles -- forced to 1 below 64 tokens; the second group doubles the epilogue staging
    # bytes of the instance SMEM clamp, the accumulator sets live in TMEM).  Each names its own program (``_a<n>`` / ``_e<n>``).
    acc_bufs = int(entry.get("acc_bufs", 1))
    epi_groups = int(entry.get("epi_groups", 1)) if tok >= 64 else 1
    if acc_bufs not in (1, 2) or epi_groups not in (1, 2):
        raise ValueError(
            f"decode table entry acc_bufs / epi_groups must be 1 or 2 (got {entry})"
        )
    cs_alias = (
        csplit > 1
        and grid == total_work
        and decode_cs_alias_fits(
            tok,
            stages,
            fused,
            resident,
            csplit,
            xb_stages,
            xq_stages,
            epi_groups=epi_groups,
        )
        and decode_cs_small_inbox_rounds(
            tok,
            stages,
            fused,
            resident,
            csplit,
            xb_stages,
            xq_stages,
            epi_groups=epi_groups,
        )
        > 1
    )
    return DecodeConfig(
        tok=tok,
        split=split,
        fused=fused,
        resident=resident,
        persist=persist,
        tstore=tstore,
        dpf=dpf,
        stages=stages,
        # The small-inbox cluster exchange gives up pipeline depth until the inbox fits (round 6: one t16 stage at C7..C16).
        module_stages=decode_cs_inbox_stages(
            tok,
            stages,
            fused,
            resident,
            csplit,
            xb_stages,
            epi_chunk,
            xq_stages,
            epi_groups,
        )
        if csplit > 1 and not cs_alias
        else decode_module_stages(
            tok, stages, fused, resident, xb_stages, epi_chunk, xq_stages, epi_groups
        ),
        m_tiles=m_tiles,
        tiles=tiles,
        total_work=total_work,
        grid=grid,
        tok_per_cta=-(-tok // split),
        xb_stages=xb_stages,
        xq_stages=xq_stages,
        qlanes=qlanes,
        csplit=csplit,
        epi_chunk=epi_chunk,
        pf=pf,
        pw=pw,
        mc=mc,
        pfx=pfx,
        pfi=pfi,
        qwarps=qwarps,
        xbh=xbh,
        qer=qer,
        cs_alias=cs_alias,
        xp=xp,
        acc_bufs=acc_bufs,
        epi_groups=epi_groups,
        quant_units=int(entry.get("quant_units", 0)),
    )


def gemm_prefetch_distance(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 6 (lever GP): weight-tile L2 prefetch distance of the GEMM route from the shape's table row (key
    ``gemm_pf``; 0 = off). Only tabulated GEMM-routed rows prefetch (e.g. the 48-pair ``tp1:q_proj:256`` whose 3-stage
    weight stream is HBM-latency-bound: 1.14x on both architectures)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return 0
    return int(entry.get("gemm_pf", 0))


def gemm_n192_admissible(n_valid: int, n_pad: int, n_tiles128: int) -> bool:
    """Round 11 (CAKE-985 T3, unit U-T3-1): whether the 192-wide pair tile (``_n192``) may run over a prepared weight
    of ``n_valid`` stored columns, ``n_pad`` 256-padded storage rows and ``n_tiles128 = ceil(n_valid / 128)`` stored
    128-row tiles that carry columns -- the Cake launcher's ``gemm_n192_admissible`` rule, the one place the host
    mirror decides it (``gemm_block_n`` and the cluster split-K launch instance read it).

    Two ways in: the round-6 storage-fit rule, ``ceil(n_valid / 192) x 192 <= n_pad`` (every B box of every N tile is a
    stored row); or the OOB-safe rule, ``n_valid % 32 == 0 and n_pad == n_tiles128 x 128``.  Under the second the
    partial last N tile reaches past the storage, but the 192-wide instance streams B in 32-row boxes (three per
    96-row CTA half; part ``j`` of CTA rank ``r`` of tile ``t`` starts at weight row ``192 t + 96 r + 32 j``), the valid
    columns end on a 32-row boundary and the storage ends with the last tile that carries columns, so every box of
    the partial tile is stored valid columns, the zero padding of the last stored tile (rows ``[n_valid, n_pad)``) or
    a box at storage-tile index ``>= n_tiles128`` = the outer extent of the B tensor map: fully out of range, zero-filled
    and credited by the TMA unit (the 384-wide instance's mechanism; no box straddles the storage edge).  The B scales
    are the host-repacked :func:`weight_scale_tiles_bn` layout of ``ceil(n_pad / 192) >= ceil(n_valid / 192)`` tiles
    (the real UE8M0 bytes of every stored row, zeros past ``n_pad``), so the valid columns of the partial tile read
    their stored weights and scales, the columns past ``n_valid`` are exact zeros in accumulator columns no epilogue
    stores (TMA store clipped at ``n_valid``, register epilogues predicated on ``col < n_valid``), and the per-element
    K order does not depend on ``bn``: bit-identical to the 256-wide program (Tier A).  N = 7168 (``o_proj``, 56 tiles):
    38 x 192 = 7296 > 7168 failed the storage-fit rule; tile 37 now streams parts 7104 / 7136 (stored) and 7168 (OOB)
    on rank 0 and 7200 / 7232 / 7264 (OOB) on rank 1."""
    n_valid, n_pad, n_tiles128 = int(n_valid), int(n_pad), int(n_tiles128)
    if -(-n_valid // GEMM_BLOCK_N_NARROW) * GEMM_BLOCK_N_NARROW <= n_pad:
        return True
    return n_valid % 32 == 0 and n_pad == n_tiles128 * BLOCK


def gemm_block_n(
    M: int, n_tiles128: int, num_k_iters: int, arch: str, n_valid: int, n_pad: int
) -> int:
    """Round 6 (lever L5): output columns per CTA pair of the GEMM route, from the shape's table row (key
    ``gemm_bn``: 192 on the tabulated N = 576 ``kv_a`` rows at M > 256, whose 256-wide tiles stream and multiply
    768 padded columns; round 10 S3: 384, the 128 x 384 pair tile; 256 otherwise).  192 is taken only where
    :func:`gemm_n192_admissible` holds (the 192-padded N fits the stored 256-padded weight rows, or -- round 11 T3 --
    every 32-row B box of the partial last tile is stored or fully out of range), 256 otherwise; 384 keeps no
    storage-fit rule (the parts of its partial last N tile past the stored 128-row weight tiles are fully
    out-of-range TMA boxes, zero-filled, whose accumulator columns are never stored)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return BLOCK_N
    bn = int(entry.get("gemm_bn", BLOCK_N))
    if bn not in (GEMM_BLOCK_N_NARROW, BLOCK_N, GEMM_BLOCK_N_WIDE):
        raise ValueError(
            f"decode table entry gemm_bn must be 192, 256 or 384 (got {entry})"
        )
    if bn == GEMM_BLOCK_N_NARROW and not gemm_n192_admissible(
        n_valid, n_pad, n_tiles128
    ):
        return BLOCK_N
    return bn


def gemm_group_m(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 10 (CAKE-985 S3, ``_g64``): raster group (row tiles per group) of the GEMM route's TMA-store /
    staged-register launches, from the shape's table row (key ``gemm_gm``: 64 on the tabulated rows, whose programs
    carry the ``_g64`` token -- the weight stream is read from HBM once per group, so half as often at M = 16384;
    same output bits; 32 otherwise).  The Cake launcher's ``gemm_group_m`` rule."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return GEMM_GROUP_M
    gm = int(entry.get("gemm_gm", GEMM_GROUP_M))
    if gm not in GEMM_GROUP_M_FORMS:
        raise ValueError(
            f"decode table entry gemm_gm must be one of {GEMM_GROUP_M_FORMS} (got {entry})"
        )
    return gm


def _entry_gemm_pdlb(entry: dict[str, Any]) -> bool:
    """The ``gemm_pdlb`` key of a table cell (0 / 1; round 11, CAKE-985 T1a): whether the cell's TMA-store / staged-register
    GEMM launches take the PDL-wait-after-weights ``_pdlb`` programs (``route_plan`` / ``required_kernel_keys``; a cluster
    split-K launch reads its cell through this helper directly, as the Cake launcher does)."""
    on = int(entry.get("gemm_pdlb", 0))
    if on not in (0, 1):
        raise ValueError(f"decode table entry gemm_pdlb must be 0 or 1 (got {entry})")
    return bool(on)


def gemm_pdlb(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> bool:
    """Round 11 (CAKE-985 T1a, ``_pdlb``): whether the GEMM route's TMA-store / staged-register launches of a shape take the
    PDL-wait-after-weights programs, from the shape's table row (key ``gemm_pdlb``: 1 on the tabulated rows, whose programs
    carry the ``_pdlb`` token -- the loader issues every stage's weight boxes before its activation boxes and waits for the
    quantization launch once per CTA, right before its first activation box, so the first stage's weights, the weight L2
    prefetches and the first CLC request land under the quantization kernel's drain; same output bits; 0 otherwise).  The
    Cake launcher's ``gemm_pdlb_enabled`` rule for the two-launch GEMM route; the register epilogue and the 384-wide program
    keep the record's order (``route_plan``)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return False
    return _entry_gemm_pdlb(entry)


def gemm_n384_cell_conflicts(entry: Optional[dict[str, Any]]) -> list[str]:
    """Round 10 (S3): the GEMM-cell keys a ``gemm_bn`` 384 cell may not carry -- the 384-wide instance implies
    ``gemm_bk`` 128, the plain CLC schedule, no weight prefetch and its own ring / staging geometry, so a non-default
    ``gemm_pf`` / ``gemm_sk`` / ``gemm_skf`` / ``gemm_sks`` / ``gemm_stages`` / ``gemm_ts_slots`` / ``gemm_pdlb`` (round 11
    T1a) or a ``gemm_bk`` of 256 names no program (the Cake launcher's ``gemm_n384_cell_conflicts``; both hosts refuse such
    a cell)."""
    entry = entry or {}
    defaults = {"gemm_stages": GEMM_NUM_STAGES, "gemm_ts_slots": GEMM_TS_SLOTS}
    conflicts = [
        k
        for k in GEMM_N384_CELL_CONFLICTS
        if entry.get(k) is not None and int(entry[k]) != defaults.get(k, 0)
    ]
    if int(entry.get("gemm_bk", 128)) != 128:
        conflicts.append("gemm_bk")
    return conflicts


def _g64_stem(stem: str) -> str:
    """The ``_g64`` raster form of a TMA-store / staged-register program stem: the token sits after the bn token
    (``_n192`` / ``_n384``) and before ``_k128`` and the later tokens."""
    for base in (GEMM_TSTORE_KERNEL_KEY, GEMM_RSTAGED_KERNEL_KEY):
        if stem.startswith(base):
            rest = stem[len(base) :]
            for bn in (GEMM_N192_SUFFIX, GEMM_N384_SUFFIX):
                if rest.startswith(bn):
                    return base + bn + GEMM_G64_SUFFIX + rest[len(bn) :]
            return base + GEMM_G64_SUFFIX + rest
    raise ValueError(f"GEMM program {stem!r} has no _g64 form (register epilogue)")


def _pdlb_stem(stem: str) -> str:
    """The ``_pdlb`` form of a TMA-store / staged-register program stem (round 11, CAKE-985 T1a): the token closes the stem,
    after the stream-K / ``_skp`` slot (``gemm_kernel_key`` appends ``_pf<D>`` behind it); the register epilogue and the
    384-wide programs have none."""
    if not stem.startswith((GEMM_TSTORE_KERNEL_KEY, GEMM_RSTAGED_KERNEL_KEY)):
        raise ValueError(f"GEMM program {stem!r} has no _pdlb form (register epilogue)")
    if GEMM_N384_SUFFIX in stem:
        raise ValueError(f"GEMM program {stem!r} has no _pdlb form (384-wide program)")
    if stem.endswith(GEMM_ST_SUFFIX):
        raise ValueError(
            f"GEMM program {stem!r} has no _pdlb form (static one-tile program)"
        )
    if stem.endswith(GEMM_GS_SUFFIX):
        raise ValueError(
            f"GEMM program {stem!r} has no _pdlb form (static grid-stride program)"
        )
    return stem + GEMM_PDLB_SUFFIX


def _entry_gemm_static(entry: dict[str, Any]) -> bool:
    """The ``gemm_static`` key of a table cell (0 / 1; round 11, CAKE-985 T2-e): whether the cell's plain 192-wide 128-K
    TMA-store / staged-register GEMM launches take the static one-tile ``_st`` programs (``route_plan`` /
    ``required_kernel_keys``)."""
    on = int(entry.get("gemm_static", 0))
    if on not in (0, 1):
        raise ValueError(f"decode table entry gemm_static must be 0 or 1 (got {entry})")
    return bool(on)


def gemm_static_cell_conflicts(entry: Optional[dict[str, Any]]) -> list[str]:
    """Round 11 (CAKE-985 T2-e): the GEMM-cell keys a ``gemm_static`` 1 cell may not carry -- the static one-tile form is the
    plain 192-wide 128-K schedule, so a non-default ``gemm_pf`` / ``gemm_sk`` / ``gemm_skf`` / ``gemm_sks`` / ``gemm_skp`` /
    ``gemm_cs`` / ``gemm_stages`` / ``gemm_ts_slots`` / ``gemm_pdlb``, a ``gemm_bn`` other than 192 or a ``gemm_bk`` other
    than 128 (absent = 256 / 256) names no program (the Cake launcher's ``gemm_static_cell_conflicts``; both hosts refuse
    such a cell)."""
    entry = entry or {}
    defaults = {"gemm_stages": GEMM_NUM_STAGES, "gemm_ts_slots": GEMM_TS_SLOTS}
    conflicts = [
        k
        for k in GEMM_STATIC_CELL_CONFLICTS
        if entry.get(k) is not None and int(entry[k]) != defaults.get(k, 0)
    ]
    if int(entry.get("gemm_bn", BLOCK_N)) != GEMM_BLOCK_N_NARROW:
        conflicts.append("gemm_bn")
    if int(entry.get("gemm_bk", BLOCK_K)) != 128:
        conflicts.append("gemm_bk")
    return conflicts


def gemm_static_tile(
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    sm_count: int,
    m_tiles: int,
    gemm_n_tiles: int,
    bn: int,
) -> bool:
    """Round 11 (CAKE-985 T2-e, ``_st``): whether the GEMM route's TMA-store / staged-register launch of a shape takes the
    static one-tile program, from the shape's table row (key ``gemm_static``: 1 on the tabulated rows, whose plain 192-wide
    128-K programs carry the ``_st`` token -- one output tile per CTA pair over a grid of 2 x tiles CTAs, no cluster-launch-
    control work ring; same MMA sequence, bit-identical output; 0 otherwise).  The Cake launcher's
    ``gemm_static_tile_enabled`` rule: the cell may carry no key the form has no composition with
    (``gemm_static_cell_conflicts``), the launch must resolve to the 192-wide program (``bn``) and its CTA-pair tiles
    (``(m_tiles // 2) x gemm_n_tiles``) must fit the resident pairs (``sm_count // 2``); a cell asking for the form elsewhere is
    refused on both hosts, never routed around."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm" or not _entry_gemm_static(entry):
        return False
    conflicts = gemm_static_cell_conflicts(entry)
    if conflicts:
        raise ValueError(
            f"{arch}: decode table cell of M={M} n_tiles128={n_tiles128} num_k_iters={num_k_iters} carries gemm_static "
            f"together with {conflicts}, which names no GEMM instance (the static one-tile form is the plain 192-wide "
            "128-K schedule)"
        )
    if int(bn) != GEMM_BLOCK_N_NARROW:
        raise ValueError(
            f"{arch}: decode table cell of M={M} n_tiles128={n_tiles128} num_k_iters={num_k_iters} carries gemm_static "
            f"but the launch resolves to the {bn}-wide program (the 192-wide tile is not admissible on the stored weight rows: gemm_n192_admissible)"
        )
    tiles = (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles)
    pairs = int(sm_count) // CTA_GROUP
    if tiles > pairs:
        raise ValueError(
            f"{arch}: decode table gemm_static on a launch of {tiles} CTA-pair tiles > {pairs} resident pairs (M={M}: the "
            "static one-tile form needs exactly one tile per pair; keep the cell's persistent program)"
        )
    return True


def _st_eligible(stem: str) -> bool:
    """Whether a program stem has a static one-tile form (round 11, CAKE-985 T2-e): the plain ``_n192_k128`` (or
    ``_n192_g64_k128``) TMA-store / staged-register stems."""
    return stem.startswith(
        (GEMM_TSTORE_KERNEL_KEY, GEMM_RSTAGED_KERNEL_KEY)
    ) and stem.endswith(
        (
            GEMM_N192_SUFFIX + GEMM_K128_SUFFIX,
            GEMM_N192_SUFFIX + GEMM_G64_SUFFIX + GEMM_K128_SUFFIX,
        )
    )


def _st_stem(stem: str) -> str:
    """The ``_st`` static one-tile form of a plain ``_n192_k128`` TMA-store / staged-register program stem (round 11,
    CAKE-985 T2-e): the token takes the ``_cs<S>`` slot, right after ``_k128``; every other stem has none."""
    if not _st_eligible(stem):
        raise ValueError(
            f"GEMM program {stem!r} has no _st form (the static one-tile form is the plain _n192_k128 TMA-store / staged-register program)"
        )
    return stem + GEMM_ST_SUFFIX


def _entry_gemm_gs(entry: dict[str, Any]) -> bool:
    """The ``gemm_gs`` key of a table cell (0 / 1; round 11, CAKE-985 T6): whether the cell's plain 128-K TMA-store GEMM
    launches at bn 192 / 256 take the static grid-stride ``_gs`` programs (``route_plan`` / ``required_kernel_keys``)."""
    on = int(entry.get("gemm_gs", 0))
    if on not in (0, 1):
        raise ValueError(f"decode table entry gemm_gs must be 0 or 1 (got {entry})")
    return bool(on)


def gemm_gs_cell_conflicts(entry: Optional[dict[str, Any]]) -> list[str]:
    """Round 11 (CAKE-985 T6): the GEMM-cell keys a ``gemm_gs`` 1 cell may not carry -- the static grid-stride form is the
    plain 128-K TMA-store schedule at bn 192 / 256, so a non-default ``gemm_pf`` / ``gemm_sk`` / ``gemm_skf`` / ``gemm_sks`` /
    ``gemm_skp`` / ``gemm_cs`` / ``gemm_stages`` / ``gemm_ts_slots`` / ``gemm_pdlb`` / ``gemm_static``, a ``gemm_bn`` other than
    192 / 256 or a ``gemm_bk`` other than 128 (absent = 256) names no program (the Cake launcher's ``gemm_gs_cell_conflicts``;
    both hosts refuse such a cell)."""
    entry = entry or {}
    defaults = {"gemm_stages": GEMM_NUM_STAGES, "gemm_ts_slots": GEMM_TS_SLOTS}
    conflicts = [
        k
        for k in GEMM_GS_CELL_CONFLICTS
        if entry.get(k) is not None and int(entry[k]) != defaults.get(k, 0)
    ]
    if int(entry.get("gemm_bn", BLOCK_N)) not in (GEMM_BLOCK_N_NARROW, BLOCK_N):
        conflicts.append("gemm_bn")
    if int(entry.get("gemm_bk", BLOCK_K)) != 128:
        conflicts.append("gemm_bk")
    return conflicts


def _gs_pairs(sm_count: int, m_tiles: int, gemm_n_tiles: int) -> int:
    """Round 11 (CAKE-985 T6, ``_gs``): the launched CTA pairs P' of a static grid-stride launch = min(T, resident pairs), T =
    ``(m_tiles // 2) x gemm_n_tiles`` pair tiles, resident pairs = ``sm_count // 2`` (the Cake launcher's ``gemm_gs_pairs``: the
    measured co-resident 2-CTA cluster capacity at the GEMM footprint, 74 on B200 / 76 on GB300); the ``sk_pairs`` scalar of the
    launch and half its grid.  Pair p owns the tiles p, p + P', p + 2 P', ... (ceil((T - p) / P') of them), a bijection onto the T
    tiles, so every tile is computed exactly once."""
    tiles = (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles)
    if tiles < 1:
        raise ValueError(
            f"_gs_pairs: a GEMM launch has at least one pair tile (got {tiles})"
        )
    return min(tiles, int(sm_count) // CTA_GROUP)


def gemm_gs_launch_pairs(
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    sm_count: int,
    m_tiles: int,
    gemm_n_tiles: int,
    applicable: bool,
) -> int:
    """Round 11 (CAKE-985 T6, ``_gs``): the launched pairs P' of the GEMM route's TMA-store launch of a shape when its table row
    asks for the static grid-stride program (key ``gemm_gs``: 1 on the tabulated rows, whose plain 128-K TMA-store programs at bn
    192 / 256 carry the ``_gs`` token -- pair p of 2 P' CTAs owns tiles p + i P', no cluster-launch-control work ring; same MMA
    sequence, bit-identical output), 0 otherwise.  The Cake launcher's ``gemm_gs_enabled`` rule: the cell may carry no key the
    form has no composition with (``gemm_gs_cell_conflicts``) and the launch must be the plain 128-K TMA-store program at bn 192
    / 256 (``applicable``; the staged-register and register epilogues keep the record's program, as for the stream-K forms); a
    cell asking for the form elsewhere is refused on both hosts, never routed around.  The Cake launch additionally asserts P'
    against the driver's co-resident cluster capacity (``gemm_gs_capacity_check``); the mirror applies the measured rule."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm" or not _entry_gemm_gs(entry):
        return 0
    conflicts = gemm_gs_cell_conflicts(entry)
    if conflicts:
        raise ValueError(
            f"{arch}: decode table cell of M={M} n_tiles128={n_tiles128} num_k_iters={num_k_iters} carries gemm_gs "
            f"together with {conflicts}, which names no GEMM instance (the static grid-stride form is the plain 128-K "
            "TMA-store schedule at bn 192 / 256)"
        )
    if not applicable:
        raise ValueError(
            f"{arch}: decode table cell of M={M} n_tiles128={n_tiles128} num_k_iters={num_k_iters} carries gemm_gs "
            "but the launch is not the plain 128-K TMA-store program at bn 192 / 256 (the static grid-stride form has no "
            "other instance)"
        )
    return _gs_pairs(sm_count, m_tiles, gemm_n_tiles)


def _gs_eligible(stem: str) -> bool:
    """Whether a program stem has a static grid-stride form (round 11, CAKE-985 T6): the plain ``_k128`` / ``_n192_k128`` (or
    their ``_g64`` raster twins) TMA-store stems; never the 384-wide, staged-register or register-epilogue programs."""
    return (
        stem.startswith(GEMM_TSTORE_KERNEL_KEY)
        and GEMM_N384_SUFFIX not in stem
        and stem.endswith(GEMM_K128_SUFFIX)
        and stem[len(GEMM_TSTORE_KERNEL_KEY) :]
        in (
            GEMM_K128_SUFFIX,
            GEMM_N192_SUFFIX + GEMM_K128_SUFFIX,
            GEMM_G64_SUFFIX + GEMM_K128_SUFFIX,
            GEMM_N192_SUFFIX + GEMM_G64_SUFFIX + GEMM_K128_SUFFIX,
        )
    )


def _gs_stem(stem: str) -> str:
    """The ``_gs`` static grid-stride form of a plain ``_k128`` / ``_n192_k128`` TMA-store program stem (round 11, CAKE-985 T6):
    the token takes the ``_cs<S>`` / ``_st`` slot, right after ``_k128``; every other stem has none."""
    if not _gs_eligible(stem):
        raise ValueError(
            f"GEMM program {stem!r} has no _gs form (the static grid-stride form is the plain _k128 / _n192_k128 TMA-store program)"
        )
    return stem + GEMM_GS_SUFFIX


def gemm_block_k(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 8 (lever L8): K elements per operand hand-off of the GEMM route, from the shape's table row (key ``gemm_bk``:
    128 on the tabulated rows, whose TMA-store / staged-register programs carry the ``_k128`` suffix -- every 128-K half of
    an operand stage is loaded and multiplied under its own barrier pair, same output bits; 256 otherwise)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return BLOCK_K
    bk = int(entry.get("gemm_bk", BLOCK_K))
    if bk not in (128, BLOCK_K):
        raise ValueError(f"decode table entry gemm_bk must be 128 or 256 (got {entry})")
    return bk


def gemm_stages_ts_slots(
    M: int, n_tiles128: int, num_k_iters: int, arch: str
) -> tuple[int, int]:
    """Round 9 (K6 arm G1): ``(operand stages, staging slots per epilogue group)`` of the GEMM route's TMA-store launch,
    from the shape's table row (keys ``gemm_stages`` / ``gemm_ts_slots``: (2, 4) on the tabulated large-M ``f_b`` / ``kv_b``
    rows, whose TMA-store programs carry the ``_s2_ts4`` tokens -- two operand stages and four staging slots, so a
    32-column store chunk waits for the store issued one tile earlier instead of the previous chunk's read; same output
    bits; (3, 1) otherwise).  TMA-store launches only: the register epilogues have no multi-slot staging and keep the
    untokened programs (the Cake launcher's ``gemm_short_k_instance`` rule)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return GEMM_NUM_STAGES, GEMM_TS_SLOTS
    return _entry_stages_ts_slots(entry)


def _entry_stages_ts_slots(entry: dict[str, Any]) -> tuple[int, int]:
    stages = int(entry.get("gemm_stages", GEMM_NUM_STAGES))
    slots = int(entry.get("gemm_ts_slots", GEMM_TS_SLOTS))
    if stages not in GEMM_STAGES_FORMS or slots not in GEMM_TS_SLOT_FORMS:
        raise ValueError(
            f"decode table entry gemm_stages must be one of {GEMM_STAGES_FORMS} and gemm_ts_slots one of "
            f"{GEMM_TS_SLOT_FORMS} (got {entry})"
        )
    return stages, slots


def gemm_instance_suffix(num_stages: int, ts_slots: int) -> str:
    """Program tokens of a GEMM operand-stage / staging-slot instance (round 9, K6 arm G1): ``_s<n>`` when the stages
    differ from 3, ``_ts<n>`` when the slots differ from 1 -- the Cake symbol spelling, placed after ``_k128`` and before
    the stream-K suffixes in the kernel key; empty for the (3, 1) form."""
    return (f"_s{int(num_stages)}" if int(num_stages) != GEMM_NUM_STAGES else "") + (
        f"_ts{int(ts_slots)}" if int(ts_slots) != GEMM_TS_SLOTS else ""
    )


def _gemm_key_instance(key: str) -> tuple[int, int]:
    """``(operand stages, staging slots)`` a GEMM kernel key names through its ``_s<n>`` / ``_ts<n>`` tokens
    (``_pf<D>`` stripped first); the (3, 1) defaults when the key carries none."""
    stages, slots = GEMM_NUM_STAGES, GEMM_TS_SLOTS
    for m in _GEMM_INSTANCE_TOKEN.finditer(key.split("_pf")[0]):
        if m.group("s") is not None:
            stages = int(m.group("s"))
        else:
            slots = int(m.group("ts"))
    return stages, slots


def gemm_k128_cell_epilogues(
    n_tiles128: int, num_k_iters: int, bucket: int
) -> tuple[bool, bool]:
    """Round 8 (lever L8): which ``_k128`` epilogue programs the contract rows of a GEMM-route table cell launch, as
    ``(TMA store, staged register)``; round 9 reads the TMA-store flag for the ``_s<n>`` / ``_ts<n>`` programs of a
    ``gemm_stages`` / ``gemm_ts_slots`` cell the same way.  The Cake export plan ships only the programs its rows launch, so the enumeration
    cannot read the cell's flags alone: the bucket's perf row (a contiguous output, ``ldo == n_valid``) takes the TMA-store
    epilogue when the family's column edge is 16-byte aligned (``gemm_tma_store_eligible``) and otherwise the staged
    register epilogue when its rows are 8-byte aligned (``gemm_reg_staged_eligible``; the tp8 ``in_proj_qkvgfab`` edge,
    ``6284 % 8 == 4``, refuses the TMA store on every row of the family), and the 8-byte-stride correctness rows of the
    ``GEMM_K128_RSTAGED_BUCKETS`` buckets take the staged epilogue.  The family is resolved from the cell's geometry
    through ``PROJECTION_FAMILIES``; a cell no family reaches, or whose families disagree, is refused, not guessed."""
    n_tiles128, num_k_iters, bucket = int(n_tiles128), int(num_k_iters), int(bucket)
    forms: dict[tuple[bool, bool], list[str]] = {}
    for tp, modules in PROJECTION_FAMILIES.items():
        for name, (n_valid, K) in modules.items():
            if -(-n_valid // BLOCK) != n_tiles128 or num_k_iters != -(-K // BLOCK_K):
                continue
            tstore = gemm_tma_store_eligible(0, n_valid, n_valid)
            rstaged = gemm_reg_staged_eligible(tstore, _store_vec(0, n_valid))
            if bucket in GEMM_K128_RSTAGED_BUCKETS:
                ldo = n_valid + GEMM_K128_RSTAGED_STRIDE_PAD
                rstaged = rstaged or gemm_reg_staged_eligible(
                    gemm_tma_store_eligible(0, ldo, n_valid), _store_vec(0, ldo)
                )
            forms.setdefault((tstore, rstaged), []).append(f"{tp}:{name}")
    if not forms:
        raise ValueError(
            f"decode table cell {n_tiles128},{num_k_iters},{bucket} carries gemm_bk 128 but its geometry matches no "
            "contract family (PROJECTION_FAMILIES)"
        )
    if len(forms) > 1:
        raise ValueError(
            f"decode table cell {n_tiles128},{num_k_iters},{bucket} carries gemm_bk 128 but its geometry matches contract "
            f"families whose output alignment selects different epilogues: {forms}"
        )
    return next(iter(forms))


def gemm_n_tiles(prepared: PreparedProjectionWeight, bn: int) -> int:
    """CTA-pair N tiles of the GEMM instance with ``bn`` output columns per pair over ``prepared``."""
    return prepared.n_tiles if int(bn) == BLOCK_N else -(-prepared.n_valid // int(bn))


# ---------------------------------------------------------------------------
# Round 10 (CAKE-985, S1): cluster split-K GEMM form (table key ``gemm_cs``)
# ---------------------------------------------------------------------------
# One output tile per cluster of 2S CTAs (S cta_group::2 pairs): pair p streams the K
# iterations ``[nk p // S, nk (p + 1) // S)`` into its own TMEM accumulator; the finished
# 128 x bn FP32 half-tiles meet through DSMEM (one bulk copy per (source pair, owner pair),
# landing in the dead operand ring) and every owner adds its S terms in ascending pair =
# ascending K order before the BF16 rounding.  Host mirror of the Cake launcher's
# ``gemm_cs_*`` arithmetic and ``gemm_cluster_split_plan`` (same rules, same errors).

# Co-resident cluster capacity of the cluster split-K GEMM instances per architecture and
# cluster size 2S (``cuOccupancyMaxActiveClusters``; one CTA per SM at the GEMM SMEM pool,
# so the count follows the GPC topology).  Round-10 M1 readings on the same-footprint
# decode instances: B200 148 SMs {4: 33, 6: 22, 8: 15}; GB300 152 SMs {4: 36, 6: 23,
# 8: 15}.  Mirrors the source repository's ``GEMM_MAX_ACTIVE_CLUSTERS`` (measured values
# only; an untabulated size is refused, never guessed from the SM count).
GEMM_MAX_ACTIVE_CLUSTERS: dict[str, dict[int, int]] = {
    "sm_100a": {4: 33, 6: 22, 8: 15},
    "sm_103a": {4: 36, 6: 23, 8: 15},
}
GEMM_CS_A_STAGE_BYTES = BLOCK_M * BLOCK_K  # the A tile of one operand stage (32 KiB)
GEMM_CS_SF_STAGE_BYTES = 6 * SF_TILE_BYTES  # SFA (2 x 512 B) + SFB (2 x 1024 B)


def gemm_cluster_capacity(arch: str, cluster_size: int) -> int:
    """Co-resident cluster capacity of ``arch`` for ``cluster_size``-CTA GEMM clusters
    (tabulated; raises when not measured)."""
    table = GEMM_MAX_ACTIVE_CLUSTERS.get(arch, {})
    if cluster_size not in table:
        raise ValueError(
            f"cluster capacity of {arch} for {cluster_size}-CTA GEMM clusters is not "
            "tabulated (GEMM_MAX_ACTIVE_CLUSTERS)"
        )
    return int(table[cluster_size])


def gemm_cs_row_form(cs: int, bn: int = GEMM_BLOCK_N_NARROW) -> bool:
    """Form of the ``_cs<S>`` instance: the column form (``_cs<S>``: pair p owns columns
    ``[bn p / S, bn (p + 1) / S)`` of all 128 rows) when ``bn / S`` is a whole number of
    32-column store chunks, else the row form (``_cs<S>r``, S even: pair
    ``p = (S / 2) g + h`` owns epilogue column group ``g`` x rows
    ``[128 h / (S / 2), 128 (h + 1) / (S / 2))``); raises when neither form exists."""
    cs, bn = int(cs), int(bn)
    if cs < 2:
        raise ValueError(
            f"cluster split-K needs at least two CTA pairs per tile, got S = {cs}"
        )
    if bn % cs == 0 and (bn // cs) % GEMM_CS_TS_COLS == 0:
        return False
    if cs % 2 == 0 and GEMM_CS_TILE_ROWS % (cs // 2) == 0:
        return True
    raise ValueError(
        f"cluster split-K S = {cs} has no form at bn {bn}: {bn} / {cs} columns is not "
        f"a multiple of {GEMM_CS_TS_COLS} and S is odd"
    )


def gemm_cs_ownership(
    cs: int, bn: int = GEMM_BLOCK_N_NARROW
) -> tuple[tuple[int, int, int, int], ...]:
    """Per pair p of the ``_cs<S>`` instance: ``(col_lo, ncols, row_lo, nrows)`` of the
    finished 128 x ``bn`` FP32 half-tile (one per CTA of the pair) that pair p owns,
    sums and stores (column form: a column slice of every row; row form: a row slice of
    one epilogue column group)."""
    cs, bn = int(cs), int(bn)
    if gemm_cs_row_form(cs, bn):
        groups = GEMM_EPI_WARPS // 4  # epilogue column groups (96 columns at bn 192)
        slices = cs // groups
        cols, rows = bn // groups, GEMM_CS_TILE_ROWS // slices
        return tuple(
            (g * cols, cols, h * rows, rows)
            for g in range(groups)
            for h in range(slices)
        )
    ncols = bn // cs
    return tuple((p * ncols, ncols, 0, GEMM_CS_TILE_ROWS) for p in range(cs))


def gemm_cs_block_bytes(cs: int, bn: int = GEMM_BLOCK_N_NARROW) -> int:
    """Bytes of one exchanged FP32 block: the owned slice at the padded row stride
    ``ncols + GEMM_CS_PAD_WORDS`` words."""
    return max(
        nrows * (ncols + GEMM_CS_PAD_WORDS) * 4
        for _col, ncols, _row, nrows in gemm_cs_ownership(cs, bn)
    )


def gemm_cs_alias_bytes(cs: int, bn: int = GEMM_BLOCK_N_NARROW) -> int:
    """SMEM the exchange aliases over the dead operand ring: S - 1 staging blocks (my
    slices for the other owners) plus S - 1 inbox slots (their slices for me)."""
    return 2 * (int(cs) - 1) * gemm_cs_block_bytes(cs, bn)


def gemm_cs_ring_bytes(
    bn: int = GEMM_BLOCK_N_NARROW, num_stages: int = GEMM_NUM_STAGES
) -> int:
    """Bytes of the operand ring (``num_stages`` x the A / B / scale stage) the exchange
    may alias once every MMA of the pair has consumed it."""
    return int(num_stages) * (
        GEMM_CS_A_STAGE_BYTES + (int(bn) // 2) * BLOCK_K + GEMM_CS_SF_STAGE_BYTES
    )


def gemm_cs_alias_fits(
    cs: int, bn: int = GEMM_BLOCK_N_NARROW, num_stages: int = GEMM_NUM_STAGES
) -> bool:
    """True when the staging + inbox blocks of the exchange fit inside the dead ring."""
    return gemm_cs_alias_bytes(cs, bn) <= gemm_cs_ring_bytes(bn, num_stages)


def gemm_cs_k_ranges(cs: int, num_k_iters: int) -> tuple[tuple[int, int], ...]:
    """K iterations ``[nk p // S, nk (p + 1) // S)`` streamed by pair p (ascending pair
    = ascending K)."""
    cs, nk = int(cs), int(num_k_iters)
    return tuple((nk * p // cs, nk * (p + 1) // cs) for p in range(cs))


def gemm_cs_suffix(cs: int, bn: int = GEMM_BLOCK_N_NARROW) -> str:
    """Program-key token of the instance: ``_cs<S>`` (column form) or ``_cs<S>r`` (row
    form), the Cake symbol spelling; after ``_k128``."""
    return f"{GEMM_CS_SUFFIX}{int(cs)}" + ("r" if gemm_cs_row_form(cs, bn) else "")


class ClusterSplitPlan(NamedTuple):
    """Round 10 (CAKE-985, S1): the cluster split-K form of one GEMM launch (host mirror
    of the Cake ``gemm_cluster_split_plan`` dict): ``cs`` pairs per output tile, the
    ``rows`` form flag, ``cluster`` = 2 x ``cs`` CTAs, ``tiles`` output tiles, ``grid``
    = ``tiles x cluster`` launched CTAs (one cluster per tile, no CLC ring), the
    tabulated ``capacity`` the grid must not exceed and the exchanged block / alias
    bytes."""

    cs: int
    rows: bool
    cluster: int
    tiles: int
    grid: int
    capacity: int
    block_bytes: int
    alias_bytes: int

    @property
    def program_suffix(self) -> str:
        """Program key token of the launched form: ``_cs<S>`` or ``_cs<S>r``."""
        return f"{GEMM_CS_SUFFIX}{self.cs}" + ("r" if self.rows else "")


def gemm_cluster_split(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 10 (CAKE-985, S1): the S of the cluster split-K GEMM form the shape's table
    cell asks for (key ``gemm_cs``; decode- or GEMM-route cell), 0 when it carries none.
    The view rule (TMA-store eligible or 8-byte-aligned output views take the form,
    every other view keeps the cell's route) is ``route_plan``'s, as in the Cake
    launcher."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    return int((entry or {}).get("gemm_cs", 0))


def _cell_gemm_n_tiles(n_tiles128: int, num_k_iters: int, bn: int) -> int:
    """CTA-pair N tiles at ``bn`` columns of the contract families a table cell's
    geometry reaches (``PROJECTION_FAMILIES``); a cell no family reaches, or whose
    families disagree, is refused, not guessed."""
    tiles = {
        -(-n_valid // int(bn))
        for modules in PROJECTION_FAMILIES.values()
        for n_valid, K in modules.values()
        if -(-n_valid // BLOCK) == int(n_tiles128)
        and int(num_k_iters) == -(-K // BLOCK_K)
    }
    if len(tiles) != 1:
        raise ValueError(
            f"decode table cell {n_tiles128},{num_k_iters} carries gemm_cs but its "
            f"geometry matches {'no' if not tiles else 'disagreeing'} contract "
            f"families (N tiles at bn {bn}: {sorted(tiles)})"
        )
    return tiles.pop()


def gemm_cluster_split_plan(
    cs: int,
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    m_tiles: int,
    gemm_n_tiles: int,
    bn: int,
    bk: int,
    stages: int = GEMM_NUM_STAGES,
    slots: int = GEMM_TS_SLOTS,
    pf: int = 0,
) -> Optional[ClusterSplitPlan]:
    """The cluster split-K plan of a GEMM launch with ``cs`` pairs per tile, or ``None``
    for ``cs`` 0 (today's dispatch).  A requested form the launch cannot take raises, as
    the Cake launcher does (a table cell asking for it is a table error, never a silent
    fallback): first-build instance (``gemm_bn`` 192, ``gemm_bk`` 128, three operand
    stages, one staging slot, no weight prefetch), exclusive with the stream-K keys
    (``gemm_sk`` / ``gemm_skf`` / ``gemm_sks``), at least two K iterations per pair, the
    exchange blocks inside the dead ring, and the whole grid co-resident:
    ``tiles <= GEMM_MAX_ACTIVE_CLUSTERS[arch][2 S]``."""
    cs = int(cs)
    if cs == 0:
        return None
    if cs not in GEMM_CS_FORMS:
        raise ValueError(
            f"decode table gemm_cs must be one of {GEMM_CS_FORMS}, got {cs}"
        )
    bn, bk, stages, slots, pf = int(bn), int(bk), int(stages), int(slots), int(pf)
    if (bn, bk) != (GEMM_BLOCK_N_NARROW, 128):
        raise ValueError(
            "the cluster split-K GEMM form is built on the 192-wide 128-K instance: "
            f"the cell must select gemm_bn 192 and gemm_bk 128 (got {bn} / {bk})"
        )
    if (stages, slots) != (GEMM_NUM_STAGES, 1):
        raise ValueError(
            "the cluster split-K GEMM form has no gemm_stages / gemm_ts_slots instance "
            f"(got {stages} / {slots}; the first build keeps ({GEMM_NUM_STAGES}, 1))"
        )
    if pf:
        raise ValueError(
            "the cluster split-K GEMM form has no weight-prefetch instance "
            f"(gemm_pf {pf})"
        )
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch) or {}
    row_sk = [
        k for k in ("gemm_sk", "gemm_skf", "gemm_sks", "gemm_skp") if entry.get(k)
    ]
    if row_sk:
        raise ValueError(
            f"gemm_cs is exclusive with the stream-K levers (cell keys {row_sk})"
        )
    nk = int(num_k_iters)
    if nk < 2 * cs:
        raise ValueError(
            f"cluster split-K S = {cs} needs at least two K iterations per pair "
            f"(num_k_iters {nk})"
        )
    rows = gemm_cs_row_form(cs, bn)
    if not gemm_cs_alias_fits(cs, bn, stages):
        raise ValueError(
            f"cluster split-K S = {cs}: the exchange blocks "
            f"({gemm_cs_alias_bytes(cs, bn)} B) do not fit the dead operand ring "
            f"({gemm_cs_ring_bytes(bn, stages)} B)"
        )
    tiles = (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles)
    cluster = CTA_GROUP * cs
    cap = gemm_cluster_capacity(arch, cluster)
    if tiles > cap:
        raise ValueError(
            f"cluster split-K S = {cs}: {tiles} output tiles exceed the {cap} "
            f"co-resident {cluster}-CTA clusters of {arch} (a second cluster pass "
            "serialises; choose a smaller S or keep the cell's route)"
        )
    return ClusterSplitPlan(
        cs,
        rows,
        cluster,
        tiles,
        tiles * cluster,
        cap,
        gemm_cs_block_bytes(cs, bn),
        gemm_cs_alias_bytes(cs, bn),
    )


def _cs_launch_instance(
    entry: dict[str, Any], prepared: PreparedProjectionWeight, tma_store: bool
) -> tuple[int, int, int, int, int]:
    """``(bn, bk, stages, slots, pf)`` a cluster split-K launch of ``entry`` reads from
    its cell, as the Cake launcher resolves them for a GEMM launch: ``gemm_bn`` (192
    only where :func:`gemm_n192_admissible` holds, as ``gemm_block_n``), ``gemm_bk``, and
    -- on the TMA-store epilogue only -- ``gemm_stages`` / ``gemm_ts_slots`` and ``gemm_pf``.
    The plan then refuses every value outside the first-build instance."""
    bn = int(entry.get("gemm_bn", BLOCK_N))
    if bn == GEMM_BLOCK_N_NARROW and not gemm_n192_admissible(
        prepared.n_valid, prepared.n_pad, prepared.n_tiles128
    ):
        bn = BLOCK_N
    bk = int(entry.get("gemm_bk", BLOCK_K))
    stages, slots = (
        _entry_stages_ts_slots(entry) if tma_store else (GEMM_NUM_STAGES, GEMM_TS_SLOTS)
    )
    pf = int(entry.get("gemm_pf", 0)) if tma_store else 0
    return bn, bk, stages, slots, pf


class StreamKPlan(NamedTuple):
    """Round 6 continuation 12 (lever SKO): the ordered stream-K split of one GEMM launch (host mirror of the Cake
    ``gemm_stream_k_plan`` dict): ``pairs`` resident CTA pairs, ``rem`` tail tiles (one head + one tail chunk each),
    ``ksplit`` head K iterations, ``dp`` data-parallel tiles, ``grid`` launched CTAs (2 x ``dp``: one CTA pair per
    data-parallel tile, as the plain schedule; the tail tiles are taken after them)."""

    pairs: int
    rem: int
    ksplit: int
    dp: int
    grid: int


def gemm_stream_k(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> bool:
    """Round 6 continuation 12 (lever SKO): True when the shape's table row asks for the ordered stream-K GEMM instance
    (key ``gemm_sk``; GEMM-routed rows only)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    return bool(
        entry is not None and entry.get("route") == "gemm" and entry.get("gemm_sk", 0)
    )


def gemm_stream_k_ksplit(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 6 continuation 13 (lever SKO-ksplit): the table row's head-chunk length of the ordered stream-K split
    (key ``gemm_sk_ksplit``), or 0 for the default ceil(K / 2)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return 0
    return int(entry.get("gemm_sk_ksplit", 0))


def gemm_stream_k_plan(
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    sm_count: int,
    m_tiles: int,
    gemm_n_tiles: int,
) -> Optional[StreamKPlan]:
    """The ordered stream-K split of a GEMM launch (round 6 continuation 12, lever SKO), or ``None`` for the plain
    CLC schedule: taken when the table row asks for it AND the launch has at least one full wave of CTA pairs plus a
    fractional one that is at most half a wave (one head and one tail chunk per pair), with at least
    ``GEMM_SK_MIN_K_ITERS`` K iterations to split.  The head chunk takes the larger half of the K iterations."""
    if (
        not gemm_stream_k(M, n_tiles128, num_k_iters, arch)
        or int(num_k_iters) < GEMM_SK_MIN_K_ITERS
    ):
        return None
    pairs = int(sm_count) // CTA_GROUP
    tiles = (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles)
    full = tiles // pairs
    rem = tiles - full * pairs
    if full < 1 or rem == 0 or rem > pairs - rem:
        return None
    # Round 6 continuation 13 (lever SKO-ksplit): the table row may ask for a head-heavy split (key ``gemm_sk_ksplit``
    # = head K iterations; 0 / absent = ceil(K / 2)); both chunks keep at least two K iterations, as in the Cake host.
    ksplit = (
        gemm_stream_k_ksplit(M, n_tiles128, num_k_iters, arch)
        or (int(num_k_iters) + 1) // 2
    )
    ksplit = min(max(ksplit, 2), int(num_k_iters) - 2)
    return StreamKPlan(
        pairs,
        rem,
        ksplit,
        full * pairs,
        full * pairs * CTA_GROUP,
    )


class StreamKFixupPlan(NamedTuple):
    """Round 6 continuation 17/18 (lever SKF): the fix-up stream-K split of one GEMM launch (host mirror of the Cake
    ``gemm_stream_k_fixup_plan`` dict): ``pairs`` resident CTA pairs, ``total`` K iterations of the SK region (the last
    full wave plus the fractional one), ``slots`` contributor partial slots per SK tile, ``first`` SK tile (= the
    data-parallel tiles before the region), ``sk_tiles`` tiles in the region, ``grid`` launched CTAs and the partial
    workspace bytes.  The kernel takes ``total`` / ``slots`` / ``first`` through the ``sk_rem`` / ``sk_ksplit`` /
    ``sk_dp`` parameters."""

    pairs: int
    total: int
    slots: int
    first: int
    sk_tiles: int
    grid: int
    partial_bytes: int
    inst_slots: int = (
        GEMM_SKF_MAX_SLOTS  # round 7: slot bound of the launched instance form
    )
    sliced: bool = False  # round 7 (lever SKS): sliced range mapping (``pairs`` = tiles, ``total`` = slices)

    @property
    def program_suffix(self) -> str:
        """Program key suffix of the launched form: ``_skf`` (round 6), ``_skf4`` / ``_skf8``, or ``_sks<slots>``."""
        if self.sliced:
            return f"{GEMM_SKS_SUFFIX}{self.inst_slots}"
        return GEMM_SKF_SUFFIX + (
            "" if self.inst_slots == GEMM_SKF_MAX_SLOTS else str(self.inst_slots)
        )


def gemm_stream_k_fixup(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> bool:
    """Round 6 continuation 17/18 (lever SKF): True when the shape's table row asks for a fix-up stream-K GEMM
    instance (key ``gemm_skf``, or the sliced form's ``gemm_sks``; GEMM-routed rows only)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    return bool(
        entry is not None
        and entry.get("route") == "gemm"
        and (entry.get("gemm_skf", 0) or entry.get("gemm_sks", 0))
    )


def gemm_skf_tile_bytes(bn: int) -> int:
    """One FP32 partial tile record of the fix-up stream-K (2 CTAs x 8 epilogue warps x (bn / 2) / 4 vectors x 32
    lanes x 16 B; 256 KiB at bn 256, 192 KiB at bn 192)."""
    return 2 * GEMM_EPI_WARPS * ((int(bn) // 2) // 4) * 32 * 16


def gemm_stream_k_fixup_plan(
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    sm_count: int,
    m_tiles: int,
    gemm_n_tiles: int,
    gemm_bn: int,
) -> Optional[StreamKFixupPlan]:
    """The fix-up stream-K split of a GEMM launch (round 6 continuation 17/18, lever SKF), or ``None`` for the plain
    CLC schedule: taken when the table row asks for it AND the launch has a fractional wave.  The SK region is the
    last full wave plus the fractional one; its ``sk_tiles x num_k_iters`` K iterations are cut into ``pairs`` equal
    contiguous ranges (pair ``p`` owns ``[p total / pairs, (p + 1) total / pairs)``, at least ``GEMM_SKF_MIN_ITERS``
    each).  A pair that does not finish a tile stores its FP32 partial into slot (tile, ordinal); the finishing pair
    adds the slots in ordinal order before rounding (Cake host mirror, including the contributor count bound)."""
    if not gemm_stream_k_fixup(M, n_tiles128, num_k_iters, arch):
        return None
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch) or {}
    skf_key = int(entry.get("gemm_skf", 0))
    slot_cap = skf_key if skf_key in GEMM_SKF_SLOT_FORMS else GEMM_SKF_MAX_SLOTS
    sks_want = int(entry.get("gemm_sks", 0))
    pairs = int(sm_count) // CTA_GROUP
    tiles = (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles)
    if sks_want:
        # round 7 (lever SKS): S slices per tile on tiles x S resident pairs (Cake host mirror)
        slices = min(
            pairs // tiles,
            int(num_k_iters) // GEMM_SKF_MIN_ITERS,
            slot_cap + 1,
            sks_want if sks_want > 1 else pairs,
        )
        if slices < 2:
            return None
        inst = next(f for f in GEMM_SKF_SLOT_FORMS if f >= slices - 1)
        return StreamKFixupPlan(
            tiles,
            slices,
            slices - 1,
            0,
            tiles,
            tiles * slices * CTA_GROUP,
            tiles * (slices - 1) * gemm_skf_tile_bytes(gemm_bn),
            inst,
            True,
        )
    if not skf_key:
        return None
    full, rem = divmod(tiles, pairs)
    if rem == 0:
        return None
    first, sk_tiles = ((full - 1) * pairs, pairs + rem) if full >= 1 else (0, tiles)
    total = sk_tiles * int(num_k_iters)
    if total // pairs < GEMM_SKF_MIN_ITERS:
        return None
    nk = int(num_k_iters)

    def pair_of(i: int) -> int:
        return ((i + 1) * pairs - 1) // total

    maxc = max(pair_of((t + 1) * nk - 1) - pair_of(t * nk) + 1 for t in range(sk_tiles))
    if maxc - 1 > slot_cap:
        return None
    slots = max(maxc - 1, 1)
    return StreamKFixupPlan(
        pairs,
        total,
        slots,
        first,
        sk_tiles,
        (first if first > 0 else pairs) * CTA_GROUP,
        sk_tiles * slots * gemm_skf_tile_bytes(gemm_bn),
        next(f for f in GEMM_SKF_SLOT_FORMS if f >= slots),
        False,
    )


# SKP_CORE_BEGIN (round 10, CAKE-985, lever SKP: the four pure-integer planner functions
# below are byte-identical to the Cake launcher's; the Cake CPU parity test asserts it)
def gemm_skp_groups(value: Any) -> tuple[int, int] | None:
    """Round 10 (CAKE-985, lever SKP): the regime-U group override ``"a:b"`` (table key
    ``gemm_skp_groups`` / ``LOOM_KIMI_GEMM_SKP_GROUPS``: ``a`` tiles per group on ``b``
    CTA pairs) as ``(a, b)``, or ``None`` when absent."""
    if value is None or value == "":
        return None
    parts = str(value).split(":")
    if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
        raise ValueError(
            f"gemm_skp_groups must be 'a:b' (tiles:pairs per group), got {value!r}"
        )
    a, b = int(parts[0]), int(parts[1])
    if a < 1 or b < 1:
        raise ValueError(f"gemm_skp_groups must be positive, got {value!r}")
    return a, b


def gemm_skp_cuts(
    a: int, b: int, nk: int, slack: int, L: int
) -> tuple[int, ...] | None:
    """Round 10 (CAKE-985, lever SKP) regime U: the ``b + 1`` cut points of one group's
    ``a * nk`` K iterations (tile-major) at makespan ``L`` CTA-pair iterations, or
    ``None`` when the recurrence does not partition them into ``b`` non-empty ranges.
    ``c_0 = 0``; range ``j`` takes ``L`` iterations, or ``L - slack`` when ``c_j + L``
    does not reach the end of the tile holding ``c_j`` (a pure contributor piece); the
    last cut is clipped to ``a * nk``.  The host planner and the kernel share this
    recurrence (pure integers)."""
    total = a * nk
    cuts = [0]
    for _ in range(b):
        c = cuts[-1]
        tile_end = (c // nk + 1) * nk
        length = L if c + L >= tile_end else L - slack
        if length < 1 or c >= total:
            return None
        cuts.append(min(c + length, total))
    return tuple(cuts) if cuts[-1] == total else None


def gemm_skp_group_plan(a: int, b: int, nk: int, slack: int) -> dict[str, Any] | None:
    """Round 10 (CAKE-985, lever SKP) regime U: the minimum-makespan partition of one
    group (``a`` tiles on ``b > a`` CTA pairs): the smallest ``L`` in ``[ceil(a nk / b),
    nk]`` whose cuts (``gemm_skp_cuts``) give every pair at most one contributor piece
    (its FIRST piece, the head of the higher tile) and one finisher piece (its LAST
    piece, holding its tile's last iteration), every piece at least
    ``GEMM_SKF_MIN_ITERS`` iterations (the SKF hand-off floor), every tile exactly one
    finisher, and every contributor piece ending at least ``slack`` iterations before
    its tile's finisher ends (times counted from each pair's start).  Returns
    ``{"L", "cuts", "slots"}`` (``slots`` = the most contributors of one tile) or
    ``None`` when no ``L`` qualifies."""
    total = a * nk
    if a < 1 or b <= a or nk < 1 or slack < 0:
        return None
    for L in range(-(-total // b), nk + 1):
        cuts = gemm_skp_cuts(a, b, nk, slack, L)
        if cuts is None:
            continue
        fin_end = [0] * a
        contrib_end: list[list[int]] = [[] for _ in range(a)]
        valid = True
        for j in range(b):
            lo, hi = cuts[j], cuts[j + 1]
            t = lo // nk
            t_end = (t + 1) * nk
            if hi <= t_end:
                piece = hi - lo
                if piece < GEMM_SKF_MIN_ITERS:
                    valid = False
                    break
                if hi == t_end:
                    fin_end[t] = piece
                else:
                    contrib_end[t].append(piece)
            elif hi < t_end + nk:
                upper, lower = hi - t_end, t_end - lo
                if upper < GEMM_SKF_MIN_ITERS or lower < GEMM_SKF_MIN_ITERS:
                    valid = False
                    break
                contrib_end[t + 1].append(upper)
                fin_end[t] = upper + lower
            else:
                valid = False
                break
        if not valid or not all(fin_end):
            continue
        if any(fin_end[t] - end < slack for t in range(a) for end in contrib_end[t]):
            continue
        return {"L": L, "cuts": cuts, "slots": max(len(c) for c in contrib_end)}
    return None


def gemm_skp_plan_core(
    tiles: int,
    pairs: int,
    nk: int,
    tile_bytes: int,
    cap: int,
    slack: int,
    groups: tuple[int, int] | None,
) -> dict[str, Any] | None:
    """Round 10 (CAKE-985, lever SKP): the prefetched stream-K plan of a GEMM launch of
    ``tiles`` CTA-pair tiles on ``pairs`` resident pairs with ``nk`` K iterations per
    tile, or ``None`` for the plain CLC schedule (pure integers; the FlashInfer host
    mirrors this function verbatim).  ``cap`` is the slot bound of the instance
    (``GEMM_SKP_SLOT_FORMS``: ``_skp`` / ``_skp4`` / ``_skp8``), ``slack`` the regime-U
    slack in iterations, ``groups`` the optional regime-U ``(a, b)`` override.

    Regime U (under-filled, ``tiles < pairs``): ``a | tiles`` tiles per group of ``b``
    pairs, ``g = tiles / a`` groups; the planner takes the minimum makespan ``L``
    (``gemm_skp_group_plan``) subject to ``g b <= pairs``, ``slots <= cap`` and
    ``tiles * slots * 2 <= GEMM_SKP_MAX_FLAGS``; ties go to fewer slots, then fewer
    pairs, then smaller groups.  ``grid`` is exactly ``2 g b`` CTAs, no CLC phase.
    Regime F (fractional last wave, ``tiles = full * pairs + rem``, ``rem > 0``):
    ``S = min(pairs // rem, cap + 1, nk // 3)`` pieces per tail tile (``S >= 3``; two
    pieces stay the bit-exact ordered form): tail tile ``i`` has contributor pieces
    ``j < S - 1`` on pairs ``i (S - 1) + j`` (EARLY, K range ``[j nk // S, (j + 1) nk
    // S)``) and its finisher on pair ``pairs - 1 - i`` (LATE, ``[(S - 1) nk // S,
    nk)`` after exactly ``dp_cap = full`` data-parallel tiles); the first
    ``full * pairs`` tiles stream through the CLC ring as the ordered form does
    (``grid = 2 full pairs``).  ``makespan`` counts CTA-pair iterations: ``L`` in regime
    U, the finisher's fixed path ``full nk + late piece`` in regime F.  ``inst_slots``
    is the cap (the table key names the instance); ``cuts`` are the group's cut points
    (U) or the tail tile's piece bounds (F)."""
    if cap not in GEMM_SKP_SLOT_FORMS:
        raise ValueError(
            f"gemm_skp slot bound must be one of {GEMM_SKP_SLOT_FORMS}, got {cap}"
        )
    if slack < 0:
        raise ValueError(f"gemm_skp_slack must be >= 0, got {slack}")
    if tiles < 1 or pairs < 1 or nk < 1:
        return None
    if tiles < pairs:
        if groups is not None:
            options = [groups] if tiles % groups[0] == 0 else []
        else:
            options = [
                (a, b)
                for a in range(1, tiles + 1)
                if tiles % a == 0
                for b in range(a + 1, pairs // (tiles // a) + 1)
            ]
        best = None
        for a, b in options:
            g = tiles // a
            if g * b > pairs:
                continue
            group = gemm_skp_group_plan(a, b, nk, slack)
            if group is None or not 1 <= group["slots"] <= cap:
                continue
            if tiles * group["slots"] * 2 > GEMM_SKP_MAX_FLAGS:
                continue
            key = (group["L"], group["slots"], g * b, b, a)
            if best is None or key < best[0]:
                best = (key, a, b, group)
        if best is None:
            return None
        _key, a, b, group = best
        slots = group["slots"]
        launched = (tiles // a) * b
        return {
            "regime": "U",
            "pairs": launched,
            "grid": launched * CTA_GROUP,
            "sk_tiles": tiles,
            "slots": slots,
            "inst_slots": cap,
            "first": 0,
            "a": a,
            "b": b,
            "L": group["L"],
            "slack": slack,
            "S": 0,
            "dp_cap": 0,
            "cuts": group["cuts"],
            "tile_bytes": tile_bytes,
            "partial_bytes": tiles * slots * tile_bytes,
            "flag_count": tiles * slots * 2,
            "makespan": group["L"],
        }
    full, rem = divmod(tiles, pairs)
    if rem == 0:
        return None
    S = min(pairs // rem, cap + 1, nk // 3)
    if S < 3:
        return None
    slots = S - 1
    if rem * slots * 2 > GEMM_SKP_MAX_FLAGS:
        return None
    cuts = tuple((j * nk) // S for j in range(S)) + (nk,)
    return {
        "regime": "F",
        "pairs": pairs,
        "grid": full * pairs * CTA_GROUP,
        "sk_tiles": rem,
        "slots": slots,
        "inst_slots": cap,
        "first": full * pairs,
        "a": 0,
        "b": 0,
        "L": 0,
        "slack": 0,
        "S": S,
        "dp_cap": full,
        "cuts": cuts,
        "tile_bytes": tile_bytes,
        "partial_bytes": rem * slots * tile_bytes,
        "flag_count": rem * slots * 2,
        "makespan": full * nk + nk - cuts[S - 1],
    }


# SKP_CORE_END


class StreamKPrefetchPlan(NamedTuple):
    """Round 10 (CAKE-985, lever SKP): the prefetched stream-K plan of one GEMM launch (host mirror of the Cake
    ``gemm_stream_k_prefetch_plan`` dict, field for field): regime ``"U"`` (under-filled grid: ``a`` tiles per group
    of ``b`` pairs at makespan ``L`` with ``slack``; ``pairs`` = the launched ``g b`` pairs, ``grid`` = ``2 pairs``,
    no CLC phase) or ``"F"`` (fractional last wave: ``S`` pieces per tail tile, finishers capped at ``dp_cap``
    data-parallel tiles, ``first`` = the first tail tile = the data-parallel tiles, ``grid`` = ``2 first``);
    ``sk_tiles`` split tiles, ``slots`` contributor partial slots per split tile, ``inst_slots`` the instance's slot
    bound (``_skp`` / ``_skp4`` / ``_skp8``), ``cuts`` the piece bounds, ``partial_bytes`` / ``flag_count`` the
    hand-off area, ``makespan`` in CTA-pair iterations."""

    regime: str
    pairs: int
    grid: int
    sk_tiles: int
    slots: int
    inst_slots: int
    first: int
    a: int
    b: int
    L: int
    slack: int
    S: int
    dp_cap: int
    cuts: tuple[int, ...]
    tile_bytes: int
    partial_bytes: int
    flag_count: int
    makespan: int

    @property
    def program_suffix(self) -> str:
        """Program key suffix of the launched form: ``_skp``, ``_skp4`` or ``_skp8``."""
        return GEMM_SKP_SUFFIX + ("" if self.inst_slots == 1 else str(self.inst_slots))


def gemm_stream_k_prefetch(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 10 (CAKE-985, lever SKP): the shape's table row's ``gemm_skp`` slot bound (1 / 4 / 8) when the GEMM-routed
    row asks for the prefetched stream-K instance, else 0.  The key is exclusive with ``gemm_sk`` / ``gemm_skf`` /
    ``gemm_sks`` on one cell and with the 384-wide form (``gemm_bn`` 384, round 10 S3: no ``_skp`` instance of it);
    the Cake table loader refuses such a table, the mirror refuses the cell."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return 0
    cap = int(entry.get("gemm_skp", 0))
    if cap and (
        any(entry.get(key) for key in ("gemm_sk", "gemm_skf", "gemm_sks"))
        or int(entry.get("gemm_bn", BLOCK_N)) == GEMM_BLOCK_N_WIDE
    ):
        raise ValueError(
            "decode table: gemm_skp is exclusive with gemm_sk / gemm_skf / gemm_sks "
            f"and with gemm_bn 384 (got {entry})"
        )
    return cap


def gemm_stream_k_prefetch_plan(
    M: int,
    n_tiles128: int,
    num_k_iters: int,
    arch: str,
    sm_count: int,
    m_tiles: int,
    gemm_n_tiles: int,
    gemm_bn: int,
) -> Optional[StreamKPrefetchPlan]:
    """The prefetched stream-K plan of a GEMM launch (round 10, CAKE-985, lever SKP), or ``None`` for the plain CLC
    schedule: taken when the table row asks for it (``gemm_skp``; ``gemm_skp_groups`` / ``gemm_skp_slack`` pin the
    regime-U groups and slack) and the geometry admits a plan (``gemm_skp_plan_core``, the Cake planner verbatim).
    Changes the FP32 reduction order of the split tiles (fixed order, deterministic)."""
    cap = gemm_stream_k_prefetch(M, n_tiles128, num_k_iters, arch)
    if not cap:
        return None
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch) or {}
    plan = gemm_skp_plan_core(
        (int(m_tiles) // CTA_GROUP) * int(gemm_n_tiles),
        int(sm_count) // CTA_GROUP,
        int(num_k_iters),
        gemm_skf_tile_bytes(gemm_bn),
        cap,
        int(entry.get("gemm_skp_slack", GEMM_SKP_SLACK)),
        gemm_skp_groups(entry.get("gemm_skp_groups")),
    )
    return None if plan is None else StreamKPrefetchPlan(**plan)


def _sk_layout(
    prepared: PreparedProjectionWeight, M: int, arch: str, sm_count: int
) -> Optional[tuple[int, int, int]]:
    """``(flags_off, partials_off, partials_bytes)`` of the stream-K hand-off area (lever SKO) inside the auxiliary
    workspace of a GEMM-path launch, or ``None`` when the shape takes the plain CLC schedule."""
    bn = gemm_block_n(
        M,
        prepared.n_tiles128,
        prepared.num_k_iters,
        arch,
        prepared.n_valid,
        prepared.n_pad,
    )
    plan = gemm_stream_k_plan(
        M,
        prepared.n_tiles128,
        prepared.num_k_iters,
        arch,
        sm_count,
        _m_tiles(M),
        gemm_n_tiles(prepared, bn),
    )
    c_off, _c_bytes, _p_off, _p_bytes = reduction_layout(prepared, M, None)
    if plan is not None:
        return c_off, c_off + GEMM_SK_FLAG_BYTES, plan.rem * GEMM_SK_TILE_BYTES
    # round 10 (CAKE-985, lever SKP): the prefetched form's flags + partial slots, before the fix-up form (launcher precedence)
    skp = gemm_stream_k_prefetch_plan(
        M,
        prepared.n_tiles128,
        prepared.num_k_iters,
        arch,
        sm_count,
        _m_tiles(M),
        gemm_n_tiles(prepared, bn),
        bn,
    )
    if skp is not None:
        return c_off, c_off + GEMM_SK_FLAG_BYTES, skp.partial_bytes
    # round 6 continuation 17/18 (lever SKF): the fix-up form's partial slots when the ordered form does not apply
    skf = gemm_stream_k_fixup_plan(
        M,
        prepared.n_tiles128,
        prepared.num_k_iters,
        arch,
        sm_count,
        _m_tiles(M),
        gemm_n_tiles(prepared, bn),
        bn,
    )
    if skf is None:
        return None
    return c_off, c_off + GEMM_SK_FLAG_BYTES, skf.partial_bytes


_SK_DUMMY: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _sk_dummy(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero partial / flag buffers for the ``sk_*`` pointers of a launch without stream-K phases (never accessed)."""
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SK_DUMMY:
        _SK_DUMMY[idx] = (
            torch.zeros((64,), dtype=torch.float32, device=f"cuda:{idx}"),
            torch.zeros((128,), dtype=torch.uint32, device=f"cuda:{idx}"),
        )
    return _SK_DUMMY[idx]


def _skp_scalars(skp: StreamKPrefetchPlan) -> dict[str, int]:
    """Round 10 (CAKE-985, lever SKP): the seven scalar parameters of the ``_skp`` programs (the Cake launcher's
    ``skp_scalars`` verbatim): regime U -> ``sk_pairs`` = launched pairs, ``sk_rem`` = a (tiles per group),
    ``sk_ksplit`` = slots, ``sk_dp`` = 0, ``sk_l`` = L, ``sk_slack`` = slack, ``sk_b`` = b; regime F -> ``sk_pairs`` =
    resident pairs, ``sk_rem`` = tail tiles, ``sk_ksplit`` = slots, ``sk_dp`` = data-parallel tiles, ``sk_l`` = S,
    ``sk_slack`` = dp_cap, ``sk_b`` = 0."""
    regime_u = skp.regime == "U"
    return dict(
        sk_pairs=int(skp.pairs),
        sk_rem=int(skp.a if regime_u else skp.sk_tiles),
        sk_ksplit=int(skp.slots),
        sk_dp=int(skp.first),
        sk_l=int(skp.L if regime_u else skp.S),
        sk_slack=int(skp.slack if regime_u else skp.dp_cap),
        sk_b=int(skp.b if regime_u else 0),
    )


def _sk_launch_kwargs(
    plan: ProjectionPlan, sf: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    """The two stream-K hand-off tensors of a GEMM program (lever SKO): the partial / flag areas of the workspace on the
    stream-K forms (``plan.gemm_sk`` / ``gemm_skf`` / ``gemm_skp``), zero dummies on every other GEMM form (every GEMM
    program takes the pointers).  The split scalars (``sk_pairs`` / ``sk_rem`` / ``sk_ksplit`` / ``sk_dp``, the three more
    of the ``_skp`` programs) are stage arguments (``_sk_scalar_kwargs`` in ``_stage_kwargs``), never bindings: the
    call-bound dictionary is applied over the stage arguments, and a binding of the same name would shadow them -- round
    11 (CAKE-985 T6, ``_gs``): the launched pairs P' travel in ``sk_pairs`` of a form without a hand-off area, so a zero
    ``sk_pairs`` binding here would launch the grid-stride program with no tiles (every output row left unwritten)."""
    sk = plan.gemm_sk
    skf = plan.gemm_skf
    skp = plan.gemm_skp
    if sk is None and skp is not None:
        # round 10 (CAKE-985, lever SKP): the SKF record layout of the partials, one flag per (SK tile, ordinal, CTA rank)
        p_bytes = skp.partial_bytes
    elif sk is None and skf is not None:
        # round 6 continuation 17/18 (lever SKF): the fix-up form's partial area
        p_bytes = skf.partial_bytes
    elif sk is None:
        partials, flags = _sk_dummy(device)
        return dict(sk_partials=partials, sk_flags=flags)
    else:
        p_bytes = sk.rem * GEMM_SK_TILE_BYTES
    f_off = plan.counters_offset
    p_off = f_off + GEMM_SK_FLAG_BYTES
    if sf.numel() < p_off + p_bytes:
        raise ValueError(
            f"workspace.sf must hold the stream-K partial area: >= {p_off + p_bytes} bytes "
            "(allocate_kimi_k3_fp8_projection_workspace)"
        )
    return dict(
        sk_partials=sf[p_off : p_off + p_bytes].view(torch.float32),
        sk_flags=sf[f_off : f_off + GEMM_SK_FLAG_BYTES].view(torch.uint32),
    )


def required_kernel_keys(arch: str, sm_count: int) -> tuple[str, ...]:
    """Every logical kernel the dispatch table can select on ``arch`` (all buckets of every tabulated family)
    plus the GEMM and the quantization widths the large-M rule can pick (``u8`` needs a K-block count divisible
    by 8 but not by 4, which does not exist)."""
    keys: list[str] = [
        quant_kernel_key(1),
        quant_kernel_key(2),
        quant_kernel_key(4),
        GEMM_KERNEL_KEY,
        GEMM_TSTORE_KERNEL_KEY,
        GEMM_RSTAGED_KERNEL_KEY,
    ]
    for key, entry in decode_table(arch).items():
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        # round 10 (CAKE-985, S1): table key ``gemm_cs`` (decode- or GEMM-route cell) -- the
        # cluster split-K programs the bucket's contract rows launch, derived from the rows'
        # epilogues like the ``_k128`` programs (``gemm_k128_cell_epilogues``: the TMA-store
        # program at the perf row, the staged-register program at the 8-byte-stride rows),
        # validated at both bucket edges as ``route_plan`` validates a launch (a cell the form
        # cannot serve is refused here, at build time); the cell's plain GEMM programs are
        # launched by no row (every TMA-store eligible / 8-byte-aligned view takes the form)
        cs_form = int(entry.get("gemm_cs", 0))
        if cs_form:
            gbn = int(entry.get("gemm_bn", BLOCK_N))
            cs_plan = None
            for M in _bucket_rows(bucket):
                cs_plan = gemm_cluster_split_plan(
                    cs_form,
                    M,
                    n_tiles128,
                    num_k_iters,
                    arch,
                    _m_tiles(M),
                    _cell_gemm_n_tiles(n_tiles128, num_k_iters, gbn),
                    gbn,
                    int(entry.get("gemm_bk", BLOCK_K)),
                    *_entry_stages_ts_slots(entry),
                    int(entry.get("gemm_pf", 0)),
                )
            assert cs_plan is not None
            cs_tstore, cs_rstaged = gemm_k128_cell_epilogues(
                n_tiles128, num_k_iters, bucket
            )
            # round 11 (CAKE-985 T1a, _pdlb): a ``gemm_pdlb`` cell's cluster split-K launches take the _pdlb form (the
            # token closes the stem); the record-order _cs<S> programs are then launched by no row of the cell
            cs_stem = (
                GEMM_N192_SUFFIX
                + GEMM_K128_SUFFIX
                + cs_plan.program_suffix
                + (GEMM_PDLB_SUFFIX if _entry_gemm_pdlb(entry) else "")
            )
            for base, launched in (
                (GEMM_TSTORE_KERNEL_KEY, cs_tstore),
                (GEMM_RSTAGED_KERNEL_KEY, cs_rstaged),
            ):
                if launched and base + cs_stem not in keys:
                    keys.append(base + cs_stem)
        if entry.get("route") == "gemm" and not cs_form:
            gpf, gbn = int(entry.get("gemm_pf", 0)), int(entry.get("gemm_bn", BLOCK_N))
            # round 10 (S3, _g64): the raster twin of every TMA-store / staged-register program of a ``gemm_gm`` 64 cell
            # (token after the bn token, before ``_k128``); the register epilogue keeps GROUP_M 32
            ggm = int(entry.get("gemm_gm", GEMM_GROUP_M))
            if ggm not in GEMM_GROUP_M_FORMS:
                raise ValueError(
                    f"decode table entry gemm_gm must be one of {GEMM_GROUP_M_FORMS} (got {entry})"
                )
            gsfx = GEMM_G64_SUFFIX if ggm == 64 else ""
            # round 11 (CAKE-985 T1a, _pdlb): the _pdlb form of every TMA-store / staged-register program of a ``gemm_pdlb``
            # cell (the token closes the stem, before ``_pf<D>``); the register epilogue keeps the record's order, a
            # 384-wide cell naming it is refused below (``gemm_n384_cell_conflicts``)
            psfx = GEMM_PDLB_SUFFIX if _entry_gemm_pdlb(entry) else ""
            # round 11 (CAKE-985 T2-e, _st): the static one-tile form of the plain 192-wide 128-K programs of a ``gemm_static``
            # cell (the token takes the ``_cs<S>`` slot; the cell may carry no other GEMM lever: ``gemm_static_cell_conflicts``),
            # admissible at both bucket edges (tiles <= resident pairs) as ``route_plan`` validates a launch -- a cell the form
            # cannot serve is refused here, at build time; the cell's plain _n192_k128 programs are then launched by no
            # TMA-store eligible / 8-byte-aligned view of the bucket (the ``_st`` keys replace them below)
            ssfx = GEMM_ST_SUFFIX if _entry_gemm_static(entry) else ""
            if ssfx:
                for M in _bucket_rows(bucket):
                    gemm_static_tile(
                        M,
                        n_tiles128,
                        num_k_iters,
                        arch,
                        sm_count,
                        _m_tiles(M),
                        _cell_gemm_n_tiles(n_tiles128, num_k_iters, gbn),
                        gbn,
                    )
            # round 11 (CAKE-985 T6, _gs): the static grid-stride form of the plain 128-K TMA-store program of a ``gemm_gs`` cell
            # (the token takes the ``_cs<S>`` / ``_st`` slot; the cell may carry no other GEMM lever: ``gemm_gs_cell_conflicts``;
            # TMA-store epilogue only -- the staged-register program of the bucket's 8-byte-stride rows keeps the record's
            # program, as for the stream-K forms), validated at both bucket edges as ``route_plan`` validates a launch
            gssfx = GEMM_GS_SUFFIX if _entry_gemm_gs(entry) else ""
            if gssfx:
                for M in _bucket_rows(bucket):
                    gemm_gs_launch_pairs(
                        M,
                        n_tiles128,
                        num_k_iters,
                        arch,
                        sm_count,
                        _m_tiles(M),
                        _cell_gemm_n_tiles(n_tiles128, num_k_iters, gbn),
                        int(entry.get("gemm_bk", BLOCK_K)) == 128
                        and gbn in (GEMM_BLOCK_N_NARROW, BLOCK_N),
                    )
            if gbn == GEMM_BLOCK_N_WIDE:
                # round 10 (S3): the 384-wide pair-tile programs of the epilogues the cell's rows launch -- always the
                # ``_k128`` form, never with stage / stream-K / prefetch tokens (a cell naming one is refused, as in the
                # Cake launcher); the fallbacks stay registered through the base list
                conflicts = gemm_n384_cell_conflicts(entry)
                if conflicts:
                    raise ValueError(
                        f"decode table cell {key} carries gemm_bn 384 together with {conflicts}, which names no GEMM "
                        f"instance (got {entry})"
                    )
                n384_tstore, n384_rstaged = gemm_k128_cell_epilogues(
                    n_tiles128, num_k_iters, bucket
                )
                for base, launched in (
                    (GEMM_TSTORE_KERNEL_KEY, n384_tstore),
                    (GEMM_RSTAGED_KERNEL_KEY, n384_rstaged),
                ):
                    gkey = base + GEMM_N384_SUFFIX + gsfx + GEMM_K128_SUFFIX
                    if launched and gkey not in keys:
                        keys.append(gkey)
            else:
                sfx = GEMM_N192_SUFFIX if gbn != BLOCK_N else ""
                tsfx = (
                    sfx + gsfx
                )  # the bn token then the raster token (stem order; round 10 S3)
                # round 8 (lever L8): the 128-K operand hand-off form of the cell's programs (table key ``gemm_bk``); the token
                # sits in the stem after ``_n192`` and before the stream-K suffixes
                gbk = int(entry.get("gemm_bk", BLOCK_K))
                if gbk not in (128, BLOCK_K):
                    raise ValueError(
                        f"decode table entry gemm_bk must be 128 or 256 (got {entry})"
                    )
                ksfx = GEMM_K128_SUFFIX if gbk == 128 else ""
                # round 9 (K6 arm G1): the operand-stage / staging-slot tokens of the cell's TMA-store programs (table keys
                # ``gemm_stages`` / ``gemm_ts_slots``; after ``_k128``, before the stream-K suffixes); the register epilogues of
                # the bucket's padded-stride rows keep the untokened programs (Cake launcher rule)
                isfx = gemm_instance_suffix(*_entry_stages_ts_slots(entry))
                gsk = bool(entry.get("gemm_sk", 0))
                gskf = bool(entry.get("gemm_skf", 0))
                gsks = bool(entry.get("gemm_sks", 0))
                gskp = int(
                    entry.get("gemm_skp", 0)
                )  # round 10 (CAKE-985, lever SKP): the slot bound names the form
                gkeys: list[str] = []
                if ksfx:
                    # round 8 (lever L8): only the _k128 programs some contract row of the cell's bucket launches, derived from
                    # the rows' epilogues (``gemm_k128_cell_epilogues``) rather than from the cell's flags and bucket alone --
                    # the perf row's TMA-store program at the cell's prefetch distance when the family's column edge admits
                    # the TMA store and no stream-K lever names that row's program (the ``_sk`` / ``_skf`` / ``_sks`` keys
                    # below carry the token instead), the staged-register program when the perf row or an 8-byte-stride row
                    # of the bucket launches it; the 256-K fallbacks stay registered through the base list
                    k128_tstore, k128_rstaged = gemm_k128_cell_epilogues(
                        n_tiles128, num_k_iters, bucket
                    )
                    if k128_tstore and not (gsk or gskf or gsks or gskp):
                        gkeys.append(
                            gemm_kernel_key(
                                GEMM_TSTORE_KERNEL_KEY
                                + tsfx
                                + ksfx
                                + (ssfx or gssfx)
                                + isfx
                                + psfx,
                                gpf,
                            )
                        )
                    if k128_rstaged:
                        gkeys.append(
                            GEMM_RSTAGED_KERNEL_KEY + tsfx + ksfx + ssfx + psfx
                        )
                elif isfx:
                    # round 9 (K6 arm G1): the cell's TMA-store program in its tokened form (the perf row's launch when the
                    # family's column edge admits the TMA store and no stream-K lever names that program -- the keys below
                    # carry the tokens instead); the staged-register program of a 192-wide cell stays as today, the
                    # untokened TMA-store programs are launched by no row of this cell
                    if gemm_k128_cell_epilogues(n_tiles128, num_k_iters, bucket)[
                        0
                    ] and not (gsk or gskf or gsks or gskp):
                        gkeys.append(
                            gemm_kernel_key(
                                GEMM_TSTORE_KERNEL_KEY + tsfx + isfx + psfx, gpf
                            )
                        )
                    if sfx:
                        gkeys.append(GEMM_RSTAGED_KERNEL_KEY + tsfx + psfx)
                else:
                    if gpf > 0:
                        # round 6 (lever GP): the prefetching GEMM program of the tabulated row's production epilogue (16-byte
                        # aligned views: TMA store); other views fall back to the non-prefetching program (``route_plan``)
                        gkeys.append(
                            gemm_kernel_key(GEMM_TSTORE_KERNEL_KEY + tsfx + psfx, gpf)
                        )
                    if sfx:
                        # round 6 (lever L5): the 192-wide programs of the row's production epilogues (16-byte aligned views:
                        # TMA store; 8-byte aligned rows: staged register epilogue); other views fall back to the 256-wide program.
                        # Round 11 (CAKE-985 C17): the plain 192-wide TMA-store program only when the cell neither prefetches nor
                        # names a stream-K lever -- the Cake export plan (``stage_kernels``) launches the ``_pf<D>`` form on every
                        # TMA-store row of a prefetching cell (``gpf`` is the cell's key on each of them) and the ``_sk`` / ``_skp`` /
                        # ``_skf`` form where the lever engages, so no row of such a cell launches the plain program (the gate of
                        # the ``_k128`` branch above); the staged-register program reads neither key and stays listed
                        if gpf == 0 and not (gsk or gskf or gsks or gskp):
                            gkeys.append(GEMM_TSTORE_KERNEL_KEY + tsfx + psfx)
                        gkeys.append(GEMM_RSTAGED_KERNEL_KEY + tsfx + psfx)
                if gsk:
                    # round 6 continuation 12 (lever SKO): the ordered stream-K instance of the row's production epilogue (16-byte
                    # aligned views: TMA store, with the row's prefetch distance -- the only stream-K program the Cake export plan
                    # ships); other views and launches outside the stream-K wave window fall back to the plain programs
                    # (``route_plan``)
                    gkeys.append(
                        gemm_kernel_key(
                            GEMM_TSTORE_KERNEL_KEY
                            + tsfx
                            + ksfx
                            + isfx
                            + GEMM_SK_SUFFIX
                            + psfx,
                            gpf,
                        )
                    )
                if gskp:
                    # round 10 (CAKE-985, lever SKP): the prefetched stream-K instance of the row's TMA-store program
                    # (``_skp`` / ``_skp4`` / ``_skp8`` by the cell's slot bound); launches whose geometry admits no
                    # plan fall back to the plain programs (``route_plan``)
                    if gskp not in GEMM_SKP_SLOT_FORMS:
                        raise ValueError(
                            f"decode table entry gemm_skp must be one of {GEMM_SKP_SLOT_FORMS} (got {entry})"
                        )
                    gkeys.append(
                        gemm_kernel_key(
                            GEMM_TSTORE_KERNEL_KEY
                            + tsfx
                            + ksfx
                            + isfx
                            + GEMM_SKP_SUFFIX
                            + ("" if gskp == 1 else str(gskp))
                            + psfx,
                            gpf,
                        )
                    )
                if gskf:
                    # round 6 continuation 17/18 (lever SKF): the fix-up stream-K instance of the row's TMA-store program (the
                    # only fix-up program the Cake export plan ships); other views fall back to the plain programs.  Round 7:
                    # a slot bound of 4 / 8 in the cell names the wider form.
                    skf_key = int(entry.get("gemm_skf", 0))
                    skf_sfx = GEMM_SKF_SUFFIX + (
                        str(skf_key)
                        if skf_key in GEMM_SKF_SLOT_FORMS
                        and skf_key != GEMM_SKF_MAX_SLOTS
                        else ""
                    )
                    gkeys.append(
                        gemm_kernel_key(
                            GEMM_TSTORE_KERNEL_KEY
                            + tsfx
                            + ksfx
                            + isfx
                            + skf_sfx
                            + psfx,
                            gpf,
                        )
                    )
                if gsks:
                    # round 7 (lever SKS): the sliced fix-up forms the row may launch (the slot form follows the
                    # row count inside the bucket: every form is listed, unavailable ones fall back in ``route_plan``)
                    for form in GEMM_SKF_SLOT_FORMS:
                        gkeys.append(
                            gemm_kernel_key(
                                GEMM_TSTORE_KERNEL_KEY
                                + tsfx
                                + ksfx
                                + isfx
                                + f"{GEMM_SKS_SUFFIX}{form}"
                                + psfx,
                                gpf,
                            )
                        )
                if gsfx:
                    # round 10 (S3, _g64): the raster twin of the plain programs the cell's rows launch (the base list holds
                    # only the GROUP_M 32 programs): the TMA-store program of the cell's form when the family's column edge
                    # admits the TMA store and no stream-K lever names that program, the staged-register program when the
                    # perf row or an 8-byte-stride row of the bucket launches it
                    g64_tstore, g64_rstaged = gemm_k128_cell_epilogues(
                        n_tiles128, num_k_iters, bucket
                    )
                    if g64_tstore and not (gsk or gskf or gsks or gskp):
                        gkeys.append(
                            gemm_kernel_key(
                                GEMM_TSTORE_KERNEL_KEY
                                + tsfx
                                + ksfx
                                + (ssfx or gssfx)
                                + isfx
                                + psfx,
                                gpf,
                            )
                        )
                    if g64_rstaged:
                        gkeys.append(
                            GEMM_RSTAGED_KERNEL_KEY + tsfx + ksfx + ssfx + psfx
                        )
                if psfx:
                    # round 11 (CAKE-985 T1a, _pdlb): the _pdlb form of the plain programs the cell's rows launch (the base
                    # list holds only the record-order programs): the TMA-store program of the cell's form when the family's
                    # column edge admits the TMA store and no stream-K lever names that program, the staged-register program
                    # when the perf row or an 8-byte-stride row of the bucket launches it
                    pdlb_tstore, pdlb_rstaged = gemm_k128_cell_epilogues(
                        n_tiles128, num_k_iters, bucket
                    )
                    if pdlb_tstore and not (gsk or gskf or gsks or gskp):
                        gkeys.append(
                            gemm_kernel_key(
                                GEMM_TSTORE_KERNEL_KEY + tsfx + ksfx + isfx + psfx, gpf
                            )
                        )
                    if pdlb_rstaged:
                        gkeys.append(GEMM_RSTAGED_KERNEL_KEY + tsfx + ksfx + psfx)
                for gkey in gkeys:
                    if gkey not in keys:
                        keys.append(gkey)
        # round 8 (table key ``quant_units``): the quantization width a two-launch cell pins (decode or GEMM route)
        cell_units = int(entry.get("quant_units", 0))
        if cell_units and quant_kernel_key(cell_units) not in keys:
            keys.append(quant_kernel_key(cell_units))
        for M in _bucket_rows(bucket):
            cfg = decode_config(M, n_tiles128, num_k_iters, arch, sm_count)
            if cfg is None:
                continue
            if cfg.kernel_key not in keys:
                keys.append(cfg.kernel_key)
            # round 6 (lever E1): tstore rows launch the ``_tso`` program on 16-byte-aligned output views and fall back to
            # the register-epilogue program otherwise, so both programs belong to the plan.  Round 10 (CAKE-985, S1): a
            # ``gemm_cs`` cell's 16-byte-aligned views take the cluster split-K GEMM form, so its ``_tso`` program is
            # launched by no view and stays out of the plan.
            tso_key = cfg.kernel_key_for(True)
            if tso_key not in keys and not cs_form:
                keys.append(tso_key)
    return tuple(keys)


def _bucket_rows(bucket: int) -> tuple[int, ...]:
    """Row counts of one table bucket that can change the physical instance (the resident eligibility depends on
    the item count per CTA, so both ends of the bucket are enumerated)."""
    lower = (
        DECODE_TABLE_BUCKETS[DECODE_TABLE_BUCKETS.index(bucket) - 1] + 1
        if bucket > 1
        else 1
    )
    return tuple(sorted({lower, bucket}))


# ---------------------------------------------------------------------------
# Weight preparation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PreparedProjectionWeight:
    """A serialized ``FP8_PB_WO`` projection prepared for the generated programs (device tensors are
    caller-owned once returned; the runner reads them in place)."""

    n_valid: int
    K: int
    n_pad: int  # 256-row padded storage rows
    n_tiles: int  # n_pad / 256 CTA-pair tiles (GEMM)
    n_tiles128: (
        int  # ceil(n_valid / 128) weight tiles that carry stored columns (decode)
    )
    num_k_iters: int  # ceil(K / 256)
    sf_k_sets: int  # k_sets(K)
    k_blocks_pad: int  # 2 * num_k_iters (128-K blocks per row tile, zero padded)
    weight_tiles: torch.Tensor = field(
        repr=False
    )  # float8_e4m3fn [n_pad/128 * k_blocks_pad, 128, 128]
    scale_tiles: torch.Tensor = field(
        repr=False
    )  # uint8 flat swizzled weight scale tiles
    weight_scale_ue8m0: torch.Tensor = field(
        repr=False
    )  # fp32 [N_pad128/128, K/128] power-of-two scales
    sf_bytes: torch.Tensor = field(
        repr=False
    )  # uint8 [n_pad/128, K/128] stored UE8M0 bytes (source of the per-instance scale tiles)
    splits: Optional[tuple[int, ...]] = None
    _scale_tiles_by_bn: dict[int, torch.Tensor] = field(
        default_factory=dict, repr=False, compare=False
    )  # round 6 (lever L5): scale tiles of the narrow GEMM instance, built on first use

    @property
    def device(self) -> torch.device:
        return self.weight_tiles.device

    def scale_tiles_for(self, bn: int) -> torch.Tensor:
        """Flat swizzled weight scale tiles of the GEMM instance with ``bn`` output columns per CTA pair:
        ``scale_tiles`` for 256; the 192-wide layout (round 6, lever L5) is built once per prepared weight from the
        stored UE8M0 bytes when a binding first routes to it (a preparation-time allocation, never a launch-time one)."""
        bn = int(bn)
        if bn == BLOCK_N:
            return self.scale_tiles
        tiles = self._scale_tiles_by_bn.get(bn)
        if tiles is None:
            tiles = weight_scale_tiles_bn(self.sf_bytes, self.K, bn)
            self._scale_tiles_by_bn[bn] = tiles
        return tiles

    @property
    def weight_q(self) -> torch.Tensor:
        """Row-major ``[n_pad, K]`` view of the stored (requantized, tiled) weight, materialised on demand."""
        K = self.K
        tiles = self.weight_tiles.view(
            self.n_pad // BLOCK, self.k_blocks_pad, BLOCK, BLOCK
        )[:, : K // BLOCK]
        return tiles.permute(0, 2, 1, 3).reshape(self.n_pad, K)

    def output_views(self, out: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Per-branch views of a fused output ``[M, n_valid]`` (arbitrary row stride) at the split points."""
        if self.splits is None:
            return (out,)
        views, c = [], 0
        for s in self.splits:
            views.append(out[:, c : c + s])
            c += s
        return tuple(views)


def prepare_kimi_k3_fp8_projection_weights(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    n_valid: Optional[int] = None,
    *,
    splits: Optional[Sequence[int]] = None,
) -> PreparedProjectionWeight:
    """Prepare a serialized ``FP8_PB_WO`` projection weight (once per weight).

    ``weight`` is the contiguous ``float8_e4m3fn [N_pad128, K]`` checkpoint tensor (128-row block padding as
    serialized), ``weight_scale`` the ModelOpt FP32 ``[N_pad128/128, 1, K/128, 1]`` block scale (or its 2-D
    view), ``n_valid`` the stored output columns (defaults to ``N_pad128``; even), ``splits`` the optional
    branch widths of a fused projection (they must sum to ``n_valid``).  The weight is requantized to UE8M0
    power-of-two scales (``s2 = 2^ceil(log2 scale)``, ``w2 = E4M3_rn(weight * scale / s2)``), padded to the
    256-row CTA-pair tile and stored as 128x128 E4M3 tiles ``[n_tile128][k_block (padded to 2 per stage)]``
    so every TMA box of the generated programs is one contiguous 32 KB chunk."""
    if (
        weight.dtype != torch.float8_e4m3fn
        or weight.dim() != 2
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "weight must be a contiguous float8_e4m3fn [N, K] tensor (serialized FP8_PB_WO)"
        )
    if weight.device.type != "cuda":
        raise ValueError("weight must be a CUDA tensor")
    n_rows, K = (int(s) for s in weight.shape)
    if K % BLOCK:
        raise ValueError(f"K must be a multiple of {BLOCK}, got {K}")
    if n_rows % BLOCK:
        raise ValueError(
            "serialized FP8_PB_WO weights carry 128-row block padding; pad N to a multiple of 128"
        )
    if n_valid is None:
        n_valid = n_rows
    n_valid = int(n_valid)
    if not (0 < n_valid <= n_rows) or n_valid % 2:
        raise ValueError(f"n_valid must be even and in (0, {n_rows}], got {n_valid}")
    if weight_scale.device != weight.device:
        raise ValueError("weight_scale must live on the weight's device")
    if weight_scale.dim() == 4:
        if tuple(weight_scale.shape) != (n_rows // BLOCK, 1, K // BLOCK, 1):
            raise ValueError(
                f"4-D weight_scale must be [{n_rows // BLOCK}, 1, {K // BLOCK}, 1], got {tuple(weight_scale.shape)}"
            )
    elif weight_scale.dim() != 2 or tuple(weight_scale.shape) != (
        n_rows // BLOCK,
        K // BLOCK,
    ):
        raise ValueError(
            f"weight_scale must be [{n_rows // BLOCK}, 1, {K // BLOCK}, 1] or [{n_rows // BLOCK}, {K // BLOCK}]"
        )
    if splits is not None:
        splits = tuple(int(s) for s in splits)
        if any(s <= 0 for s in splits) or sum(splits) != n_valid:
            raise ValueError(
                f"splits {splits} must be positive and sum to n_valid {n_valid}"
            )
    n_pad = n_padded(n_valid)
    num_k_iters = -(-K // BLOCK_K)
    w2, s2 = requant_weight_ue8m0(
        weight, weight_scale.reshape(n_rows // BLOCK, K // BLOCK)
    )
    w_pad = torch.zeros((n_pad, K), dtype=torch.float8_e4m3fn, device=weight.device)
    w_pad[:n_rows] = w2
    sf_bytes = torch.zeros(
        (n_pad // BLOCK, K // BLOCK), dtype=torch.uint8, device=weight.device
    )
    sf_bytes[: n_rows // BLOCK] = ue8m0_byte(s2)
    k_blocks_pad = 2 * num_k_iters
    w_tiles = torch.zeros(
        (n_pad // BLOCK, k_blocks_pad, BLOCK, BLOCK),
        dtype=torch.float8_e4m3fn,
        device=weight.device,
    )
    w_tiles[:, : K // BLOCK] = w_pad.view(
        n_pad // BLOCK, BLOCK, K // BLOCK, BLOCK
    ).permute(0, 2, 1, 3)
    return PreparedProjectionWeight(
        n_valid=n_valid,
        K=K,
        n_pad=n_pad,
        n_tiles=n_pad // BLOCK_N,
        n_tiles128=-(-n_valid // BLOCK),
        num_k_iters=num_k_iters,
        sf_k_sets=k_sets(K),
        k_blocks_pad=k_blocks_pad,
        weight_tiles=w_tiles.reshape(-1, BLOCK, BLOCK).contiguous(),
        scale_tiles=weight_scale_tiles(sf_bytes, K),
        weight_scale_ue8m0=s2,
        sf_bytes=sf_bytes,
        splits=splits,
    )


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


class ProjectionWorkspace(NamedTuple):
    """Caller-owned workspaces of one ``M``: the E4M3 activation ``q [M, K]`` and the auxiliary byte buffer
    ``sf`` (activation scale tiles + decode split-K partials + per-tile counters)."""

    q: torch.Tensor
    sf: torch.Tensor


def reduction_layout(
    prepared: PreparedProjectionWeight, M: int, cfg: Optional[DecodeConfig]
) -> tuple[int, int, int, int]:
    """``(counters_off, counters_bytes, partials_off, partials_bytes)`` of the decode split-K area inside the
    auxiliary workspace (after the activation scale tiles; sized for twice the dispatcher's split)."""
    sf_bytes = activation_sf_workspace_bytes(
        M, prepared.K, cfg.tok if cfg is not None else SF_TILE_ROWS
    )
    c_off = -(-sf_bytes // 256) * 256
    if cfg is None:
        return c_off, 0, c_off, 0
    if cfg.csplit > 1:
        return (
            c_off,
            256,
            c_off + 256,
            0,
        )  # the cluster exchange keeps its partials in SMEM
    split_cap = min(prepared.num_k_iters, max(cfg.split * 2, 1))
    c_bytes = -(-(cfg.tiles * 2 * 4) // 256) * 256
    p_off = c_off + c_bytes
    p_bytes = cfg.tiles * split_cap * cfg.tok * 128 * 4
    return c_off, c_bytes, p_off, p_bytes


def workspace_sf_bytes(
    prepared: PreparedProjectionWeight, M: int, arch: str, sm_count: int
) -> int:
    """Bytes of the auxiliary workspace for ``M`` rows on ``arch``."""
    cfg = decode_config(M, prepared.n_tiles128, prepared.num_k_iters, arch, sm_count)
    _c_off, _c_bytes, p_off, p_bytes = reduction_layout(prepared, M, cfg)
    if cfg is None:
        # round 6 continuation 12 / 17 (levers SKO / SKF): the GEMM path's stream-K flags + partial tiles (zero-initialised once)
        sk = _sk_layout(prepared, M, arch, sm_count)
        if sk is not None:
            return sk[1] + sk[2]
    return p_off + p_bytes


def allocate_kimi_k3_fp8_projection_workspace(
    prepared: PreparedProjectionWeight, M: int
) -> ProjectionWorkspace:
    """Allocate the caller-owned workspaces for ``M`` activation rows on the weight's device (no launch).

    The auxiliary buffer is sized for the resolved route (activation scale tiles, decode split-K counters +
    partials, or the stream-K flags + partial tiles of a GEMM-route table row) and zero initialised as a whole: the
    scale padding rows must read as zero scales, the counters / flags start at zero, and the programs leave them
    reset, so one workspace serves every launch of this ``M``."""
    M = int(M)
    if M < 1:
        raise ValueError("M must be positive")
    device = prepared.device
    facts = device_facts(device)
    q = torch.empty((M, prepared.K), dtype=torch.float8_e4m3fn, device=device)
    # Zero initialised as a whole: the activation scale padding rows must read as zero scales, the decode per-tile
    # counters and the stream-K flags (levers SKO / SKF) start at zero, and the programs leave them reset.
    sf = torch.zeros(
        (workspace_sf_bytes(prepared, M, facts.arch, facts.sm_count),),
        dtype=torch.uint8,
        device=device,
    )
    return ProjectionWorkspace(q, sf)


# ---------------------------------------------------------------------------
# Route plan and launch binding
# ---------------------------------------------------------------------------


def _m_tiles(M: int) -> int:
    tiles = (M + BLOCK_M - 1) // BLOCK_M
    return tiles + (tiles % CTA_GROUP)


def _gemm_grid(m_tiles: int, n_tiles: int) -> int:
    return (m_tiles // CTA_GROUP) * n_tiles * CTA_GROUP


def _decode_store_vec(data_ptr: int, ldo: int, n_valid: int) -> int:
    """Widest BF16 vector store (elements) of the decode epilogue for the ``(out, ldo, n_valid)`` triple."""
    for vec in (8, 4):
        if data_ptr % (2 * vec) == 0 and ldo % vec == 0 and n_valid % vec == 0:
            return vec
    return 2


def _store_vec(data_ptr: int, ldo: int) -> int:
    """Widest BF16 vector store (elements) for which every ``row * ldo + 2k`` element address stays aligned."""
    for vec in (16, 4):
        if data_ptr % (2 * vec) == 0 and ldo % vec == 0:
            return vec
    return 2


def gemm_reg_staged_eligible(tma_store: bool, store_vec: int) -> bool:
    """Staged row-coalesced register epilogue for output views the TMA store cannot address but whose rows are
    at least 8-byte aligned (``store_vec >= 4``; e.g. ``n_valid = 6284``): the tile is staged in SMEM and copied
    out row by row so every store instruction writes whole 32-byte sectors."""
    return not tma_store and int(store_vec) >= 4


def gemm_tma_store_eligible(data_ptr: int, ldo: int, n_valid: int) -> bool:
    """The output view can be a TMA tensor map: 16-byte base, a row stride that is a multiple of 16 bytes and a
    16-byte column edge.  The TMA unit bounds-checks the inner (contiguous) axis of a store at 16-byte granularity,
    so a map whose inner extent is not a multiple of 16 bytes writes the rest of the edge chunk (``n_valid = 6284``
    with a 16-byte row stride stored columns 6284..6287 of the caller's padding); such views keep the predicated
    register epilogue."""
    return data_ptr % 16 == 0 and ldo % 8 == 0 and n_valid % 8 == 0


GEMM_TS_COLS = (
    128,
    32,
)  # [rows, BF16 columns] of the ``OUT`` descriptor placeholder of the register-epilogue GEMM


@functools.cache
def _placeholder_out_map(device_index: int) -> torch.Tensor:
    """The tensor behind the ``OUT`` descriptor of the register-epilogue GEMM programs, which never access it:
    one per device, allocated on first use and shared by every prepared runner."""
    return torch.zeros(
        GEMM_TS_COLS, dtype=torch.bfloat16, device=torch.device("cuda", device_index)
    )


@dataclass(frozen=True)
class ProjectionPlan:
    """The resolved route of one ``(x, prepared, out, workspace)`` binding."""

    arch: str
    sm_count: int
    M: int
    route: str  # "decode" or "gemm"
    decode: Optional[DecodeConfig]
    gemm_tma_store: bool  # GEMM route: TMA-store epilogue (aligned output view) instead of the register epilogue
    gemm_reg_staged: bool  # GEMM route: staged row-coalesced register epilogue (8-byte aligned rows the TMA store cannot address)
    decode_tma_store: bool  # decode route: the row's ``tstore`` instance on a 16-byte-aligned output view (round 6, lever E1)
    gemm_bn: int  # GEMM route: output columns per CTA pair of the launched program (256, or 192 on the tabulated ``gemm_bn`` rows; round 6, lever L5)
    gemm_bk: int  # GEMM route: K elements per operand hand-off of the launched program (256, or 128 on the tabulated ``gemm_bk`` rows; round 8, lever L8)
    gemm_stages: int  # GEMM route: operand stages of the launched program (3, or 2 on the tabulated ``gemm_stages`` rows' TMA-store launches; round 9, K6 arm G1)
    gemm_ts_slots: int  # GEMM route: staging slots per epilogue group of the launched program (1, or 4 on the tabulated ``gemm_ts_slots`` rows' TMA-store launches; round 9, K6 arm G1)
    gemm_gm: int  # GEMM route: raster group (row tiles per group) of the launched program (32, or 64 on the tabulated ``gemm_gm`` rows' TMA-store / staged-register launches; round 10 S3, ``_g64``)
    gemm_pdlb: bool  # GEMM route: the PDL-wait-after-weights form of the launched program (True on the tabulated ``gemm_pdlb`` rows' TMA-store / staged-register launches; round 11 T1a, ``_pdlb``)
    gemm_static: bool  # GEMM route: the static one-tile form of the launched program (True on the tabulated ``gemm_static`` rows' plain 192-wide 128-K TMA-store / staged-register launches at tiles <= resident pairs; round 11 T2-e, ``_st``)
    gemm_gs: bool  # GEMM route: the static grid-stride form of the launched program (True on the tabulated ``gemm_gs`` rows' plain 128-K TMA-store launches at bn 192 / 256; round 11 T6, ``_gs``)
    gemm_gs_pairs: int  # GEMM route: the launched pairs P' = min(T, sm_count // 2) of a ``_gs`` launch (its ``sk_pairs`` scalar, half its grid); 0 on every other program
    gemm_n_tiles: int  # GEMM route: CTA-pair N tiles of the launched program (``n_pad / 256`` or ``ceil(n_valid / 192)``)
    gemm_sk: Optional[
        StreamKPlan
    ]  # GEMM route: the ordered stream-K split of the launched ``_sk`` program, None for the plain CLC schedule (round 6 continuation 12, lever SKO)
    gemm_skf: Optional[
        StreamKFixupPlan
    ]  # GEMM route: the fix-up stream-K split of the launched ``_skf`` program, None otherwise (round 6 continuation 17/18, lever SKF)
    gemm_skp: Optional[
        StreamKPrefetchPlan
    ]  # GEMM route: the prefetched stream-K plan of the launched ``_skp`` program, None otherwise (round 10, CAKE-985, lever SKP)
    gemm_cs: Optional[
        ClusterSplitPlan
    ]  # GEMM route: the cluster split-K form of the launched ``_cs<S>[r]`` program (one 2S-CTA cluster per output tile), None otherwise (round 10, CAKE-985, S1)
    quant_units: Optional[int]  # None when the decode instance quantizes in-CTA
    sf_rows: int
    kernels: tuple[str, ...]  # logical kernel key per launch, in launch order
    programs: tuple[
        tuple[str, Defines], ...
    ]  # (registered program, compile-line defines) per launch
    grids: tuple[int, ...]
    counters_offset: int
    counters_bytes: int
    partials_offset: int
    partials_bytes: int

    @property
    def workspace_sf_bytes(self) -> int:
        return self.partials_offset + self.partials_bytes


def route_plan(
    prepared: PreparedProjectionWeight,
    M: int,
    arch: str,
    sm_count: int,
    cfg: Optional[DecodeConfig],
    *,
    gemm_tma_store: bool = True,
    gemm_reg_staged: bool = False,
    decode_tma_store: Optional[bool] = None,
) -> ProjectionPlan:
    """Resolve the launch sequence of ``M`` rows from the resolved decode route ``cfg`` (``None`` = quantization
    launch + GEMM) without touching device memory.

    ``gemm_tma_store`` / ``gemm_reg_staged`` select the GEMM epilogue program (``gemm_tma_store_eligible`` /
    ``gemm_reg_staged_eligible`` of the output view); ``decode_tma_store`` (default = ``gemm_tma_store``) tells whether the
    output view admits the decode TMA-store epilogue of a ``tstore`` table row (round 6, lever E1)."""
    if decode_tma_store is None:
        decode_tma_store = bool(gemm_tma_store)
    M = int(M)
    # round 10 (CAKE-985, S1): table key ``gemm_cs`` -- a TMA-store eligible or 8-byte-aligned
    # output view of a cell carrying it takes the GEMM route with the cluster split-K program
    # (``_cs<S>[r]`` of the 192-wide 128-K TMA-store / staged-register epilogue, one cluster of
    # 2S CTAs per output tile); every other view keeps the cell's route (Cake launcher rule).
    # The plan validates the cell / launch as the launcher does (raises otherwise); a program
    # this checkout does not register keeps today's dispatch.
    cs_plan: Optional[ClusterSplitPlan] = None
    cs_key = ""
    cs_form = (
        gemm_cluster_split(M, prepared.n_tiles128, prepared.num_k_iters, arch)
        if (gemm_tma_store or gemm_reg_staged)
        else 0
    )
    if cs_form:
        entry = (
            decode_table_entry(M, prepared.n_tiles128, prepared.num_k_iters, arch) or {}
        )
        cs_bn, cs_bk, cs_st, cs_ts, cs_pf = _cs_launch_instance(
            entry, prepared, bool(gemm_tma_store)
        )
        cs_plan = gemm_cluster_split_plan(
            cs_form,
            M,
            prepared.n_tiles128,
            prepared.num_k_iters,
            arch,
            _m_tiles(M),
            gemm_n_tiles(prepared, cs_bn),
            cs_bn,
            cs_bk,
            cs_st,
            cs_ts,
            cs_pf,
        )
        assert cs_plan is not None
        cs_key = (
            (GEMM_TSTORE_KERNEL_KEY if gemm_tma_store else GEMM_RSTAGED_KERNEL_KEY)
            + GEMM_N192_SUFFIX
            + GEMM_K128_SUFFIX
            + cs_plan.program_suffix
        )
        # round 11 (CAKE-985 T1a, _pdlb): the cell's _pdlb form first (the token closes the
        # stem), then the record-order program, as the plain GEMM candidates below
        cs_keys = (_pdlb_stem(cs_key), cs_key) if _entry_gemm_pdlb(entry) else (cs_key,)
        cs_key = next((k for k in cs_keys if route_available(arch, (k,))), "")
        if cs_key:
            cfg = None
        else:
            cs_plan = None
    c_off, c_bytes, p_off, p_bytes = reduction_layout(prepared, M, cfg)
    kernels: list[str] = []
    grids: list[int] = []
    units: Optional[int] = None
    if cfg is None or not cfg.fused:
        k_blocks = prepared.K // BLOCK
        # round 8 (table key ``quant_units``): the cell's launch width when it pins one (the decode config carries it; a
        # GEMM-route row reads its cell), the ``quant_units`` rule otherwise (host mirror of the Cake launcher)
        units = launch_quant_units(
            M,
            k_blocks,
            cfg.quant_units
            if cfg is not None
            else table_quant_units(M, prepared.n_tiles128, prepared.num_k_iters, arch),
            sm_count,
        )
        units_per_row = k_blocks // units
        kernels.append(quant_kernel_key(units))
        grids.append(-(-(M * units_per_row) // (QUANT_WARPS * 2)))
    if cfg is None and cs_plan is not None:
        # round 10 (CAKE-985, S1): the cluster split-K program (one 2S-CTA cluster per output
        # tile; the ``sk_*`` scalars stay zero, no workspace area)
        key = cs_key
        gbn, gbk = GEMM_BLOCK_N_NARROW, 128
        gst, gts = GEMM_NUM_STAGES, GEMM_TS_SLOTS
        ggm = GEMM_GROUP_M  # the _cs<S> programs keep GROUP_M 32 (round 10 S3, _g64)
        gpdlb = (
            GEMM_PDLB_SUFFIX in key
        )  # the launched program's form (round 11 T1a, _pdlb)
        gstatic = False  # the _cs<S> programs keep their cluster schedule (round 11 T2-e, _st: exclusive by construction)
        ggs, ggs_pairs = (
            False,
            0,
        )  # ... and no static grid-stride form (round 11 T6, _gs: exclusive by construction)
        sk_plan = None
        skf_plan = None
        skp_plan = None
        kernels.append(key)
        grids.append(cs_plan.grid)
    elif cfg is None:
        base = (
            GEMM_TSTORE_KERNEL_KEY
            if gemm_tma_store
            else GEMM_RSTAGED_KERNEL_KEY
            if gemm_reg_staged
            else GEMM_KERNEL_KEY
        )
        gpf = (
            gemm_prefetch_distance(M, prepared.n_tiles128, prepared.num_k_iters, arch)
            if gemm_tma_store
            else 0
        )
        gbn = gemm_block_n(
            M,
            prepared.n_tiles128,
            prepared.num_k_iters,
            arch,
            prepared.n_valid,
            prepared.n_pad,
        )
        # round 10 (S3): the 384-wide pair-tile form the table row asks for (TMA-store and staged-register epilogues only;
        # the predicated register epilogue keeps the 256-wide program).  It implies the 128-K ring, one staging tile, no
        # weight prefetch and the plain CLC schedule; a cell naming a conflicting key is refused, as in the Cake launcher.
        n384 = gbn == GEMM_BLOCK_N_WIDE and base != GEMM_KERNEL_KEY
        if n384:
            conflicts = gemm_n384_cell_conflicts(
                decode_table_entry(M, prepared.n_tiles128, prepared.num_k_iters, arch)
            )
            if conflicts:
                raise ValueError(
                    f"{arch}: decode table cell of M={M} n_tiles128={prepared.n_tiles128} num_k_iters="
                    f"{prepared.num_k_iters} carries gemm_bn 384 together with {conflicts}, which names no GEMM instance"
                )
            gpf, gbk, isfx = 0, 128, ""
        else:
            # round 8 (lever L8): the 128-K operand hand-off form the table row asks for (TMA-store and staged-register
            # epilogues only, like the 192-wide form; the predicated register epilogue keeps the 256-K program)
            gbk = gemm_block_k(M, prepared.n_tiles128, prepared.num_k_iters, arch)
            # round 9 (K6 arm G1): the operand-stage / staging-slot form the table row asks for (TMA-store epilogue only: the
            # four-slot staging has no other form; the register epilogues keep the (3, 1) programs, Cake launcher rule)
            isfx = (
                gemm_instance_suffix(
                    *gemm_stages_ts_slots(
                        M, prepared.n_tiles128, prepared.num_k_iters, arch
                    )
                )
                if base == GEMM_TSTORE_KERNEL_KEY
                else ""
            )
        # round 10 (S3, _g64): the raster group the table row asks for (TMA-store and staged-register epilogues only, the
        # Cake launcher's ``gemm_group_m`` rule; the predicated register epilogue keeps GROUP_M 32)
        ggm = (
            gemm_group_m(M, prepared.n_tiles128, prepared.num_k_iters, arch)
            if base != GEMM_KERNEL_KEY
            else GEMM_GROUP_M
        )
        # round 11 (CAKE-985 T1a, _pdlb): the PDL-wait-after-weights form the table row asks for (TMA-store and
        # staged-register epilogues of the two-launch route only, never the 384-wide program: the Cake launcher's
        # ``gemm_pdlb_enabled`` applicability; the predicated register epilogue keeps the record's order)
        pdlb = (
            gemm_pdlb(M, prepared.n_tiles128, prepared.num_k_iters, arch)
            if base != GEMM_KERNEL_KEY and not n384
            else False
        )
        # round 11 (CAKE-985 T2-e, _st): the static one-tile form the table row asks for (TMA-store and staged-register
        # epilogues of the plain 192-wide 128-K program only, never the 384-wide program: the Cake launcher's
        # ``gemm_static_tile_enabled`` rule -- a cell naming a key the form has no composition with, a launch that does not
        # resolve to the 192-wide program or one of more CTA-pair tiles than resident pairs is refused; the predicated
        # register epilogue keeps the record's program)
        static = (
            gemm_static_tile(
                M,
                prepared.n_tiles128,
                prepared.num_k_iters,
                arch,
                sm_count,
                _m_tiles(M),
                gemm_n_tiles(prepared, gbn),
                gbn,
            )
            if base != GEMM_KERNEL_KEY and not n384
            else False
        )
        # round 6 (lever GP): the prefetching program ships with the TMA-store epilogue only (Cake rule); use it when
        # registered.  Round 6 (lever L5): the 192-wide instance of the row's epilogue program when the table row asks
        # for it and the program is registered; every fallback keeps the row on a registered program.
        stems: list[tuple[str, int, int]] = []  # (program stem, prefetch distance, bn)
        if n384:
            # round 10 (S3): the only 384-wide program of the epilogue (always ``_k128``), then the 256-wide fallbacks
            stems.append((base + GEMM_N384_SUFFIX + GEMM_K128_SUFFIX, 0, gbn))
        elif gbn != BLOCK_N and base != GEMM_KERNEL_KEY:
            # round 7: the 192-wide form ships for the TMA-store and staged-register epilogues only (Cake
            # launcher rule); the predicated register epilogue keeps the 256-wide program
            stems.append((base + GEMM_N192_SUFFIX, gpf, gbn))
            if gpf:
                stems.append((base + GEMM_N192_SUFFIX, 0, gbn))
        if gpf:
            stems.append((base, gpf, BLOCK_N))
        stems.append((base, 0, BLOCK_N))
        if ggm == 64:
            # round 10 (S3, _g64): each stem in its raster form first, then as today, so a missing program falls back at
            # the same geometry onto a registered program (the token sits after the bn token, before ``_k128``)
            stems = [
                v for s, pf, bn in stems for v in ((_g64_stem(s), pf, bn), (s, pf, bn))
            ]
        if gbk == 128 and base != GEMM_KERNEL_KEY and not n384:
            # round 8 (lever L8): each stem in its _k128 form first, then as today, so a missing program falls back at the
            # same geometry onto a base-list program
            stems = [
                v
                for s, pf, bn in stems
                for v in ((s + GEMM_K128_SUFFIX, pf, bn), (s, pf, bn))
            ]
        if static:
            # round 11 (CAKE-985 T2-e, _st): the static one-tile form of the plain _n192_k128 stems first (the token takes the
            # _cs<S> slot, after _k128), then as today, so a missing program falls back at the same geometry onto a
            # registered program (a gemm_static cell carries no stage / stream-K / prefetch / _pdlb key: the candidates
            # below are the plain forms)
            stems = [
                v
                for s, pf, bn in stems
                for v in (
                    (((_st_stem(s), pf, bn),) if _st_eligible(s) else ())
                    + ((s, pf, bn),)
                )
            ]
        # round 11 (CAKE-985 T6, _gs): the static grid-stride form the table row asks for (the plain 128-K TMA-store program at bn
        # 192 / 256 of the two-launch route only: the Cake launcher's ``gemm_gs_enabled`` rule -- a cell naming a key the form has
        # no composition with, or a launch that is not that program, is refused; the staged-register and register epilogues keep
        # the record's program, as for the stream-K forms)
        gs_pairs = (
            gemm_gs_launch_pairs(
                M,
                prepared.n_tiles128,
                prepared.num_k_iters,
                arch,
                sm_count,
                _m_tiles(M),
                gemm_n_tiles(prepared, gbn),
                gbk == 128
                and gbn in (GEMM_BLOCK_N_NARROW, BLOCK_N)
                and not isfx
                and not gpf
                and not pdlb
                and not static,
            )
            if base == GEMM_TSTORE_KERNEL_KEY and not n384
            else 0
        )
        if gs_pairs:
            # the _gs form of the plain _k128 / _n192_k128 stems first (the token takes the _cs<S> / _st slot, after _k128), then
            # as today, so a missing program falls back at the same geometry onto a registered program (a gemm_gs cell carries
            # no stage / stream-K / prefetch / _pdlb / _st key: the candidates below are the plain forms)
            stems = [
                v
                for s, pf, bn in stems
                for v in (
                    (((_gs_stem(s), pf, bn),) if _gs_eligible(s) else ())
                    + ((s, pf, bn),)
                )
            ]
        if isfx:
            # round 9 (K6 arm G1): each stem in its ``_s<n>`` / ``_ts<n>`` form first, then as today, so a missing program
            # falls back at the same geometry onto a registered program
            stems = [
                v for s, pf, bn in stems for v in ((s + isfx, pf, bn), (s, pf, bn))
            ]
        # round 6 continuation 12 (lever SKO): the ordered stream-K instance (``_sk`` before ``_pf``) of each candidate when
        # the table row asks for it and the launch sits in the stream-K wave window (one full wave plus at most half a
        # wave of tail tiles), falling back to the plain program of the same stem
        candidates: list[
            tuple[
                str,
                int,
                Optional[StreamKPlan],
                Optional[StreamKFixupPlan],
                Optional[StreamKPrefetchPlan],
            ]
        ] = []

        def pdlb_forms(stem: str) -> tuple[str, ...]:
            # round 11 (CAKE-985 T1a, _pdlb): each candidate in its _pdlb form first (the token closes the stem, after
            # the stream-K / _skp suffix and before ``_pf<D>``), then as today, so a missing program falls back at the
            # same geometry onto a registered program; the register epilogue keeps the record's order
            return (_pdlb_stem(stem), stem) if pdlb else (stem,)

        for stem, pf, bn in stems:
            # round 7: the ordered stream-K form ships for the TMA-store epilogue only, like the fix-up form
            # (Cake launcher rule); the register epilogues keep the plain schedule
            sk = (
                gemm_stream_k_plan(
                    M,
                    prepared.n_tiles128,
                    prepared.num_k_iters,
                    arch,
                    sm_count,
                    _m_tiles(M),
                    gemm_n_tiles(prepared, bn),
                )
                if stem.startswith(GEMM_TSTORE_KERNEL_KEY)
                else None
            )
            if sk is not None:
                for form in pdlb_forms(stem + GEMM_SK_SUFFIX):
                    candidates.append((gemm_kernel_key(form, pf), bn, sk, None, None))
            else:
                # round 10 (CAKE-985, lever SKP): the prefetched stream-K instance (``_skp`` / ``_skp4`` / ``_skp8``) when the
                # table row asks for it and the geometry admits a plan, before the fix-up form (launcher precedence);
                # TMA-store stems only
                skp = (
                    gemm_stream_k_prefetch_plan(
                        M,
                        prepared.n_tiles128,
                        prepared.num_k_iters,
                        arch,
                        sm_count,
                        _m_tiles(M),
                        gemm_n_tiles(prepared, bn),
                        bn,
                    )
                    if stem.startswith(GEMM_TSTORE_KERNEL_KEY)
                    else None
                )
                if skp is not None:
                    for form in pdlb_forms(stem + skp.program_suffix):
                        candidates.append(
                            (gemm_kernel_key(form, pf), bn, None, None, skp)
                        )
                # round 6 continuation 17/18 (lever SKF): the fix-up stream-K instance (``_skf``) when the table row asks
                # for it and the launch has a fractional wave the ordered form cannot serve; TMA-store stems only
                skf = (
                    gemm_stream_k_fixup_plan(
                        M,
                        prepared.n_tiles128,
                        prepared.num_k_iters,
                        arch,
                        sm_count,
                        _m_tiles(M),
                        gemm_n_tiles(prepared, bn),
                        bn,
                    )
                    if stem.startswith(GEMM_TSTORE_KERNEL_KEY) and skp is None
                    else None
                )
                if skf is not None:
                    for form in pdlb_forms(stem + skf.program_suffix):
                        candidates.append(
                            (gemm_kernel_key(form, pf), bn, None, skf, None)
                        )
            for form in pdlb_forms(stem):
                candidates.append((gemm_kernel_key(form, pf), bn, None, None, None))
        key, gbn, sk_plan, skf_plan, skp_plan = next(
            (
                (k, b, s, f, p)
                for k, b, s, f, p in candidates
                if route_available(arch, (k,))
            ),
            (base, BLOCK_N, None, None, None),
        )
        gbk = (
            128 if GEMM_K128_SUFFIX in key.split("_pf")[0] else BLOCK_K
        )  # the launched program's hand-off (a fallback may have dropped the _k128 form)
        ggm = (
            64 if GEMM_G64_SUFFIX in key.split("_pf")[0] else GEMM_GROUP_M
        )  # the launched program's raster group (a fallback may have dropped the _g64 form; round 10 S3)
        gpdlb = (
            GEMM_PDLB_SUFFIX in key.split("_pf")[0]
        )  # the launched program's form (a fallback may have dropped the _pdlb token; round 11 T1a)
        gstatic = key.split(
            "_pf"
        )[
            0
        ].endswith(
            GEMM_ST_SUFFIX
        )  # the launched program's form (a fallback may have dropped the _st token; round 11 T2-e)
        ggs = key.split(
            "_pf"
        )[
            0
        ].endswith(
            GEMM_GS_SUFFIX
        )  # the launched program's form (a fallback may have dropped the _gs token; round 11 T6)
        ggs_pairs = (
            _gs_pairs(sm_count, _m_tiles(M), gemm_n_tiles(prepared, gbn)) if ggs else 0
        )  # P' at the launched program's bn (grid 2 P' CTAs, the sk_pairs scalar)
        gst, gts = _gemm_key_instance(
            key
        )  # the launched program's stages / slots (a fallback may have dropped the tokens)
        kernels.append(key)
        grids.append(
            sk_plan.grid
            if sk_plan is not None
            else skp_plan.grid
            if skp_plan is not None
            else skf_plan.grid
            if skf_plan is not None
            else CTA_GROUP * ggs_pairs
            if ggs
            else _gemm_grid(_m_tiles(M), gemm_n_tiles(prepared, gbn))
        )
    else:
        gbn = BLOCK_N
        gbk = BLOCK_K
        gst, gts = GEMM_NUM_STAGES, GEMM_TS_SLOTS
        ggm = GEMM_GROUP_M
        gpdlb = False
        gstatic = False
        ggs, ggs_pairs = False, 0
        sk_plan = None
        skf_plan = None
        skp_plan = None
    dec_ts = False
    if cfg is not None:
        key = cfg.kernel_key_for(bool(decode_tma_store))
        dec_ts = key != cfg.kernel_key
        if dec_ts and not route_available(arch, (key,)):
            key, dec_ts = (
                cfg.kernel_key,
                False,
            )  # the TMA-store program is not registered for this arch: register epilogue
        kernels.append(key)
        grids.append(cfg.grid)
    return ProjectionPlan(
        arch=arch,
        sm_count=int(sm_count),
        M=M,
        route="decode" if cfg is not None else "gemm",
        decode=cfg,
        gemm_tma_store=cfg is None and bool(gemm_tma_store),
        gemm_reg_staged=cfg is None and not gemm_tma_store and bool(gemm_reg_staged),
        decode_tma_store=dec_ts,
        gemm_bn=gbn,
        gemm_bk=gbk,
        gemm_stages=gst,
        gemm_ts_slots=gts,
        gemm_gm=ggm,
        gemm_pdlb=gpdlb,
        gemm_static=gstatic,
        gemm_gs=ggs,
        gemm_gs_pairs=ggs_pairs,
        gemm_n_tiles=gemm_n_tiles(prepared, gbn),
        gemm_sk=sk_plan,
        gemm_skf=skf_plan,
        gemm_skp=skp_plan,
        gemm_cs=cs_plan,
        quant_units=units,
        sf_rows=cfg.tok if cfg is not None else SF_TILE_ROWS,
        kernels=tuple(kernels),
        programs=tuple(kernel_program(arch, key) for key in kernels),
        grids=tuple(grids),
        counters_offset=c_off,
        counters_bytes=c_bytes,
        partials_offset=p_off,
        partials_bytes=p_bytes,
    )


def generated_program_available(
    device: torch.device,
    M: Optional[int] = None,
    prepared: Optional[PreparedProjectionWeight] = None,
) -> bool:
    """True when this checkout registers the programs for ``device``.

    With ``M`` and ``prepared`` the check names the exact launch sequence of that call; without them it asks
    whether every kernel the dispatch table can select on the device's architecture is registered."""
    try:
        facts = device_facts(device)
    except ValueError:
        return False
    if M is None and prepared is None:
        return bool(MODULES) and route_available(
            facts.arch, required_kernel_keys(facts.arch, facts.sm_count)
        )
    if M is None or prepared is None:
        raise ValueError("pass both M and prepared or neither")
    cfg = decode_config(
        int(M), prepared.n_tiles128, prepared.num_k_iters, facts.arch, facts.sm_count
    )
    try:
        route_plan(prepared, int(M), facts.arch, facts.sm_count, cfg)
    except NotImplementedError:
        return False
    return True


def _bind(
    program: tuple[str, Defines], arch: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``program`` and load its entry for ``arch``."""
    name, defines = program
    record = MODULES[name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, argument in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[argument])
        elif argument in kwargs:
            arguments.append(kwargs[argument])
        else:
            raise KeyError(
                f"generated program {name!r} expects argument {argument!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_fp8_projection_module(name, arch, defines)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


@dataclass(frozen=True)
class KimiK3Fp8ProjectionRunner:
    """The prepared launch sequence of one ``(x, prepared, out, workspace)`` binding.

    ``launch()`` runs the quantization (unless fused) and the GEMM / decode program on the current torch stream
    into the caller-owned ``out`` with no CUDA allocation and no host synchronisation and returns ``out``; it is
    CUDA-graph capturable (capture belongs to the caller).  The programs read ``x`` on device at every launch, so
    the same runner (or a graph capturing it) stays valid when the caller writes new activations into ``x``.
    Prepare a new runner when ``M``, the weight or a tensor binding changes."""

    plan: ProjectionPlan
    prepared: PreparedProjectionWeight
    x: torch.Tensor
    out: torch.Tensor
    workspace: ProjectionWorkspace
    launches: tuple[tuple[Callable[..., Any], tuple], ...] = field(repr=False)

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            for entry, arguments in self.launches:
                entry(*arguments)
        return self.out

    __call__ = launch

    @property
    def launch_count(self) -> int:
        return len(self.launches)


def _activation_rows(x: torch.Tensor, prepared: PreparedProjectionWeight) -> int:
    """``M`` of a valid activation binding (the kernels read ``x`` row-major with row stride ``K``)."""
    if (
        x.dtype != torch.bfloat16
        or x.dim() != 2
        or not x.is_contiguous()
        or int(x.shape[1]) != prepared.K
    ):
        raise ValueError(f"x must be a contiguous bf16 [M, {prepared.K}] tensor")
    if x.data_ptr() % 16:
        # The quantization programs read x with 16-byte vector loads and the fused decode programs map it as a TMA
        # tensor (16-byte base); the driver would otherwise refuse the map with an opaque CUresult.
        raise ValueError("x must be 16-byte aligned")
    M = int(x.shape[0])
    if M < 1:
        raise ValueError("M must be positive")
    return M


def validate_kimi_k3_fp8_projection_inputs(
    x: torch.Tensor,
    prepared: PreparedProjectionWeight,
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
    *,
    sf_bytes: int,
) -> int:
    """Shape / dtype / alignment validation of one call (``sf_bytes``: the auxiliary workspace the resolved route
    needs); returns ``M``."""
    M = _activation_rows(x, prepared)
    if x.device != prepared.device or out.device != prepared.device:
        raise ValueError("x, out and the prepared weight must be on one CUDA device")
    if (
        out.dtype != torch.bfloat16
        or out.dim() != 2
        or tuple(out.shape) != (M, prepared.n_valid)
        or out.stride(1) != 1
    ):
        raise ValueError(
            f"out must be a bf16 [{M}, {prepared.n_valid}] view with unit column stride"
        )
    ldo = int(out.stride(0))
    if ldo % 2 or ldo < prepared.n_valid:
        raise ValueError(
            f"out row stride must be even and >= {prepared.n_valid}, got {ldo}"
        )
    if (out.storage_offset() * 2) % 4:
        raise ValueError("out must be 4-byte aligned")
    q, sf = workspace
    if (
        tuple(q.shape) != (M, prepared.K)
        or q.dtype != torch.float8_e4m3fn
        or not q.is_contiguous()
    ):
        raise ValueError(
            f"workspace.q must be a contiguous float8_e4m3fn [{M}, {prepared.K}] tensor"
        )
    needed = int(sf_bytes)
    if (
        sf.dtype != torch.uint8
        or sf.dim() != 1
        or sf.numel() < needed
        or not sf.is_contiguous()
    ):
        raise ValueError(
            f"workspace.sf must be a contiguous uint8 tensor of >= {needed} bytes "
            "(allocate_kimi_k3_fp8_projection_workspace)"
        )
    if q.device != prepared.device or sf.device != prepared.device:
        raise ValueError("the workspace must be on the weight's device")
    if sf.data_ptr() % 256:
        raise ValueError("workspace.sf must be 256-byte aligned")
    return M


def _resolve_plan(
    prepared: PreparedProjectionWeight, M: int, out: torch.Tensor, facts: DeviceFacts
) -> tuple[ProjectionPlan, int]:
    """Route plan of ``M`` rows into the output view ``out`` (its row stride and address alignment select the
    epilogue program); returns ``(plan, ldo)``."""
    cfg = decode_config(
        M, prepared.n_tiles128, prepared.num_k_iters, facts.arch, facts.sm_count
    )
    ldo = int(out.stride(0)) if out.dim() == 2 else 0
    tma_store = gemm_tma_store_eligible(out.data_ptr(), ldo, prepared.n_valid)
    plan = route_plan(
        prepared,
        M,
        facts.arch,
        facts.sm_count,
        cfg,
        gemm_tma_store=tma_store,
        gemm_reg_staged=gemm_reg_staged_eligible(
            tma_store, _store_vec(out.data_ptr(), ldo)
        ),
        decode_tma_store=tma_store,
    )
    return plan, ldo


# Launch arguments that bind the call's tensors (activations, output view, workspace); every other argument of a
# stage is a function of the prepared weight and the resolved plan alone.
# The call-bound arguments a launch receipt re-binds per call -> the index of the call tensor they bind: 0 = ``x``
# (the quantization input / the fused instance's ``XB`` descriptor), 1 = ``out`` (the raw output pointer / the
# TMA-store instances' ``OUT`` descriptor).  Up to four positions per program (``run_launch`` takes four slots;
# ``run_chain`` takes the same pairs packed into one integer per receipt, see ``_slot_codes``).
_REBIND_SOURCES = {"x": 0, "XB": 0, "out": 1, "OUT": 1}
_REBIND_SLOTS = 4


def _slot_codes(rebinds: Sequence[tuple[int, int]]) -> int:
    """The re-bind pairs of one receipt as the packed ``slots`` word of ``run_chain``: four 16-bit lanes, each 0
    (unused) or ``1 + (FFI position << 1 | call tensor index)``, the live pairs in order (padding skipped)."""
    codes = 0
    lane = 0
    for position, source in rebinds:
        if position < 0:
            continue
        codes |= (1 + ((position << 1) | source)) << (16 * lane)
        lane += 1
    return codes


_CALL_ARGS = frozenset(
    (
        "x",
        "XB",
        "out",
        "OUT",
        "a_q",
        "a_sf",
        "A",
        "SFA",
        "X",
        "SFX",
        "partials",
        "counters",
        "sk_partials",
        "sk_flags",
    )
)


def _workspace_bindings(
    plan: ProjectionPlan,
    prepared: PreparedProjectionWeight,
    M: int,
    workspace: ProjectionWorkspace,
    device: torch.device,
) -> dict[str, Any]:
    """The workspace-derived launch arguments of ``plan`` (views of ``workspace`` for ``M`` rows)."""
    q, sf = workspace
    a_u8 = q.view(torch.uint8)
    sfa = sf[: activation_sf_workspace_bytes(M, prepared.K, plan.sf_rows)].view(
        -1, 4, 128
    )
    bound: dict[str, Any] = dict(a_q=q, a_sf=sf, A=a_u8, SFA=sfa, X=a_u8, SFX=sfa)
    if plan.decode is None:
        bound.update(_sk_launch_kwargs(plan, sf, device))
    else:
        c_off, c_bytes, p_off, p_bytes = (
            plan.counters_offset,
            plan.counters_bytes,
            plan.partials_offset,
            plan.partials_bytes,
        )
        bound["partials"] = sf[p_off : p_off + p_bytes].view(torch.float32)
        bound["counters"] = sf[c_off : c_off + c_bytes].view(torch.uint32)
    return bound


def _output_bindings(
    plan: ProjectionPlan,
    prepared: PreparedProjectionWeight,
    M: int,
    ldo: int,
    x: torch.Tensor,
    out: torch.Tensor,
    device_index: int,
) -> dict[str, Any]:
    """The activation / output launch arguments of one call: ``x`` (also ``XB`` of the fused decode instance), the
    flat output view and the ``OUT`` TMA-store descriptor source (the output view for the TMA-store programs, the
    per-device placeholder for the register-epilogue GEMM programs; the register-epilogue decode programs do not
    take it)."""
    out_flat = torch.as_strided(
        out, (ldo * (M - 1) + prepared.n_valid,), (1,), out.storage_offset()
    )
    bound: dict[str, Any] = dict(x=x, XB=x, out=out_flat)
    if plan.decode is None:
        bound["OUT"] = (
            out if plan.gemm_tma_store else _placeholder_out_map(device_index)
        )
    elif plan.decode_tma_store:
        bound["OUT"] = out
    return bound


def _stage_kwargs(
    plan: ProjectionPlan,
    prepared: PreparedProjectionWeight,
    facts: DeviceFacts,
    M: int,
    ldo: int,
    out_ptr: int,
) -> list[tuple[tuple[str, Defines], dict[str, Any]]]:
    """``(program, fixed kwargs)`` per launch of ``plan`` -- every argument except the call-bound ones (_CALL_ARGS)."""
    stages: list[tuple[tuple[str, Defines], dict[str, Any]]] = []
    stage = 0
    cfg = plan.decode
    if plan.quant_units is not None:
        k_blocks = prepared.K // BLOCK
        stages.append(
            (
                plan.programs[stage],
                dict(
                    M=M,
                    K=prepared.K,
                    units_per_row=k_blocks // plan.quant_units,
                    sf_k_sets=prepared.sf_k_sets,
                    sf_rows=plan.sf_rows,
                    grid=(plan.grids[stage], 1, 1),
                ),
            )
        )
        stage += 1
    if cfg is None:
        stages.append(
            (
                plan.programs[stage],
                dict(
                    B=prepared.weight_tiles.view(torch.uint8),
                    SFB=prepared.scale_tiles_for(plan.gemm_bn).view(
                        -1, weight_scale_sf_rows(plan.gemm_bn), 128
                    ),
                    K=prepared.K,
                    M=M,
                    m_tiles=_m_tiles(M),
                    n_tiles=plan.gemm_n_tiles,
                    n_valid=prepared.n_valid,
                    ldo=ldo,
                    store_vec=_store_vec(out_ptr, ldo),
                    num_k_iters=prepared.num_k_iters,
                    sf_k_tiles=prepared.sf_k_sets,
                    **_sk_scalar_kwargs(plan),
                    grid=(plan.grids[stage], 1, 1),
                ),
            )
        )
    else:
        stages.append(
            (
                plan.programs[stage],
                dict(
                    W=prepared.weight_tiles.view(torch.uint8),
                    SFW=prepared.scale_tiles.view(-1, 4, 128),
                    M=M,
                    n_tiles=prepared.n_tiles128,
                    n_valid=prepared.n_valid,
                    ldo=ldo,
                    num_k_iters=prepared.num_k_iters,
                    sf_k_tiles=prepared.sf_k_sets,
                    split=cfg.split,
                    tok_per_cta=cfg.tok_per_cta,
                    total_work=cfg.total_work,
                    store_vec=_decode_store_vec(out_ptr, ldo, prepared.n_valid),
                    K=prepared.K,
                    grid=(plan.grids[stage], 1, 1),
                ),
            )
        )
    return stages


def _sk_scalar_kwargs(plan: ProjectionPlan) -> dict[str, int]:
    """The four stream-K split scalars of a GEMM program (the two hand-off tensors are workspace bindings); seven on the
    ``_skp`` programs (round 10, CAKE-985, lever SKP)."""
    sk = plan.gemm_sk
    skf = plan.gemm_skf
    if sk is None and plan.gemm_skp is not None:
        return _skp_scalars(plan.gemm_skp)
    if sk is None and skf is not None:
        return dict(
            sk_pairs=skf.pairs, sk_rem=skf.total, sk_ksplit=skf.slots, sk_dp=skf.first
        )
    if sk is None and plan.gemm_gs:
        # round 11 (CAKE-985 T6, _gs): the launched pairs P' travel in ``sk_pairs`` (the static grid-stride form has no other scalar)
        return dict(sk_pairs=plan.gemm_gs_pairs, sk_rem=0, sk_ksplit=0, sk_dp=0)
    if sk is None:
        return dict(sk_pairs=0, sk_rem=0, sk_ksplit=0, sk_dp=0)
    return dict(sk_pairs=sk.pairs, sk_rem=sk.rem, sk_ksplit=sk.ksplit, sk_dp=sk.dp)


def prepare_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: PreparedProjectionWeight,
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
) -> KimiK3Fp8ProjectionRunner:
    """Validate the binding, select the route and bind the launch sequence.

    ``out`` is a ``[M, n_valid]`` BF16 view with unit column stride and an even row stride (any view into a
    wider buffer); ``workspace`` comes from :func:`allocate_kimi_k3_fp8_projection_workspace` for this ``M``
    (its scale-tile and counter bytes zero initialised once; the programs keep the counters reset).  The route
    is resolved once here from the device's cached architecture / SM count; the JIT modules of the route are
    built and loaded here, so prepare outside CUDA Graph capture.  For many launches of one weight with changing
    activations / output views (an engine's prefill), :class:`KimiK3Fp8ProjectionLauncher` caches the route and
    the workspaces per ``M`` and binds only the call's tensors."""
    device = prepared.device
    device_index = _device_index(device)
    facts = device_facts(device)
    M = _activation_rows(x, prepared)
    plan, ldo = _resolve_plan(prepared, M, out, facts)
    validate_kimi_k3_fp8_projection_inputs(
        x, prepared, out, workspace, sf_bytes=plan.workspace_sf_bytes
    )
    bound = _workspace_bindings(plan, prepared, M, workspace, device)
    bound.update(_output_bindings(plan, prepared, M, ldo, x, out, device_index))
    launches: list[tuple[Callable[..., Any], tuple]] = []
    with torch.cuda.device(device_index):
        for program, fixed in _stage_kwargs(
            plan, prepared, facts, M, ldo, out.data_ptr()
        ):
            kwargs = dict(fixed)
            kwargs.update(bound)
            launches.append(_bind(program, facts.arch, kwargs))
    return KimiK3Fp8ProjectionRunner(plan, prepared, x, out, workspace, tuple(launches))


def kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: PreparedProjectionWeight,
    out: Optional[torch.Tensor] = None,
    *,
    workspace: Optional[ProjectionWorkspace] = None,
) -> torch.Tensor:
    """``out[:, :n_valid] = bf16(x @ dequant(w2, s2).T)`` in one call (allocates ``out`` / the workspace when
    omitted; prefer :func:`prepare_kimi_k3_fp8_projection` for repeated launches and CUDA graphs)."""
    if out is None:
        out = torch.empty(
            (int(x.shape[0]), prepared.n_valid),
            dtype=torch.bfloat16,
            device=prepared.device,
        )
    if workspace is None:
        workspace = allocate_kimi_k3_fp8_projection_workspace(prepared, int(x.shape[0]))
    return prepare_kimi_k3_fp8_projection(x, prepared, out, workspace)()


@dataclass(frozen=True)
class _StageEntry:
    """One launch of a template: the program's ``run_plan`` / ``run_launch`` / ``run_chain`` entries, its argument
    slots with the call-bound arguments (_CALL_ARGS) left as names (``(is_call_arg, value_or_name)`` per FFI
    argument) and the ``(FFI position, call tensor index)`` pairs a call re-binds (_REBIND_SOURCES; the ``OUT``
    position of a register-epilogue GEMM program is not among them: it carries the per-device placeholder
    descriptor), padded with ``(-1, 0)`` to the four slots ``run_launch`` takes."""

    plan: Callable[..., Any]
    launch: Callable[..., Any]
    chain: Callable[..., Any]
    slots: tuple[tuple[bool, Any], ...]
    rebinds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _LaunchTemplate:
    """The launch sequence of one ``(M, output row stride, output address class)``."""

    plan: ProjectionPlan
    ldo: int
    stages: tuple[_StageEntry, ...]


@dataclass
class _WorkspaceState:
    workspace: ProjectionWorkspace
    bindings: dict[tuple[str, ...], dict[str, Any]]
    pinned: bool = False


# Host bytes per launch receipt: the generated ``run_plan`` lays out at most 25 kernel arguments at 64-byte
# alignment behind a small header (< 2 KiB) and rejects a shorter buffer with the exact requirement.
RECEIPT_BYTES = 8192

_BF16 = torch.bfloat16

# The per-call host path of the launcher (``KimiK3Fp8ProjectionLauncher.__call__``) is interpreter-bound (W4
# budget: <= 10 us per call on every supported host); it binds the two CUDA state queries it needs directly to
# their C entries instead of the ``torch.cuda`` wrappers (``current_stream`` constructs a ``Stream`` object per
# call) and falls back to the wrappers on a torch build without them.
try:
    # the current stream's cudaStream_t of a device, as an integer
    _current_raw_stream = torch._C._cuda_getCurrentRawStream
except AttributeError:  # pragma: no cover - older torch

    def _current_raw_stream(device_index: int) -> int:
        return torch.cuda.current_stream(device_index).cuda_stream


try:
    _current_stream_capturing = torch._C._cuda_isCurrentStreamCapturing
except AttributeError:  # pragma: no cover - older torch
    _current_stream_capturing = torch.cuda.is_current_stream_capturing


@dataclass(frozen=True)
class _Receipt:
    """The frozen launches of one ``(M, output row stride, output alignment class)``: per stage the program's
    ``run_launch`` entry, the planned host receipt (kernel-argument bytes, grid, device) and the re-bind pairs;
    for a two-stage template also the one-crossing form -- the first stage's ``run_chain`` entry with both receipts
    and their packed slot codes (``_slot_codes``) -- which ``__call__`` issues instead of two ``run_launch`` calls."""

    stages: tuple[
        tuple[Callable[..., Any], torch.Tensor, tuple[tuple[int, int], ...]], ...
    ]
    chain: tuple[Callable[..., Any], torch.Tensor, int, torch.Tensor, int] | None
    state: _WorkspaceState
    # the single-stage form flattened for ``__call__``: ``(launch, record, i0, s0, i1, s1, i2, s2, i3, s3)``
    single: tuple[Any, ...] | None = None


class KimiK3Fp8ProjectionLauncher:
    """Allocation-light repeated launches of one prepared weight (an engine's linear layer).

    ``launcher(x, out)`` runs ``out[:, :n_valid] = bf16(x @ dequant(w2, s2).T)`` for any ``M = x.shape[0]``: the
    route plan and the per-stage argument templates are resolved once per ``(M, row stride, output address class)``,
    the workspace of each ``M`` is allocated once and reused (the programs leave it reusable), and the launches of
    one ``(M, row stride, output address class)`` are planned once into caller-owned host *launch receipts* (the
    generated programs' ``run_plan``: the complete argument validation, descriptor encoding and scalar marshalling of
    a launch, frozen into a host byte tensor).  A call then only re-binds its activation and its output view into
    those receipts (the raw pointers and, for the fused / TMA-store instances, the ``XB`` / ``OUT`` descriptors) and
    issues the launches: a single-launch route through ``run_launch`` (one FFI crossing), a two-launch route through
    the generated programs' ``run_chain`` (one FFI crossing replaying both receipts in order: every receipt names the
    callees of the program that planned it, so the quantization program's entry launches the GEMM / decode program's
    receipt through that program's own launch code).  No route resolution, validation of the prepared weight,
    workspace allocation, descriptor re-encoding of the weights / workspace or host synchronisation happens per
    call.  The receipts hold the kernel-argument bytes only, so the path is CUDA-graph capturable.

    ``max_workspaces`` bounds the per-``M`` workspace cache (least recently used ``M`` evicted first, with its
    receipts); an ``M`` used inside CUDA-graph capture (first seen there or resolved eagerly before) is pinned,
    because the captured graph keeps referencing its workspace.  ``max_receipts`` bounds the receipt cache (least
    recently used ``(M, row stride, output address class)`` evicted first).  One launcher serves one stream order: concurrent launches of the
    same ``M`` from several streams would share a workspace.  Resolve a new ``(M, out)`` once eagerly before capturing
    a graph of it (the JIT modules of the route load on first use)."""

    def __init__(
        self,
        prepared: PreparedProjectionWeight,
        *,
        max_workspaces: int = 64,
        max_receipts: int = 256,
    ):
        if int(max_workspaces) < 1:
            raise ValueError("max_workspaces must be positive")
        if int(max_receipts) < 1:
            raise ValueError("max_receipts must be positive")
        self.prepared = prepared
        self.device = prepared.device
        self._device_index = _device_index(prepared.device)
        self._K = int(prepared.K)
        self._n_valid = int(prepared.n_valid)
        # the most recently used receipt key (skips the recency move of a repeated call)
        self._mru: tuple[int, int, int] | None = None
        self._facts = device_facts(prepared.device)
        self._max_workspaces = int(max_workspaces)
        self._max_receipts = int(max_receipts)
        self._templates: dict[tuple[int, int, int], _LaunchTemplate] = {}
        self._workspaces: dict[int, _WorkspaceState] = {}  # insertion order = recency
        self._receipts: dict[
            tuple[int, int, int], _Receipt
        ] = {}  # (M, ldo, out address % 32); insertion order = recency

    # -- caches ---------------------------------------------------------------------------------------------------

    def workspace(self, M: int) -> ProjectionWorkspace:
        """The cached workspace of ``M`` rows (allocated on first use)."""
        return self._workspace_state(int(M)).workspace

    def _workspace_state(self, M: int) -> _WorkspaceState:
        state = self._workspaces.get(M)
        if state is not None:
            if next(reversed(self._workspaces)) != M:
                self._workspaces[M] = self._workspaces.pop(M)  # most recently used last
            if not state.pinned and torch.cuda.is_current_stream_capturing():
                state.pinned = True  # a graph captured after the eager resolution keeps referencing this workspace
            return state
        while len(self._workspaces) >= self._max_workspaces:
            victim = next(
                (m for m, st in self._workspaces.items() if not st.pinned), None
            )
            if victim is None:
                break  # every cached workspace is pinned by a captured graph: grow instead of evicting
            del self._workspaces[victim]
            for key in [key for key in self._receipts if key[0] == victim]:
                del self._receipts[
                    key
                ]  # the receipts froze the evicted workspace's addresses
                if key == self._mru:
                    self._mru = None
        workspace = allocate_kimi_k3_fp8_projection_workspace(self.prepared, M)
        state = _WorkspaceState(
            workspace, {}, pinned=bool(torch.cuda.is_current_stream_capturing())
        )
        self._workspaces[M] = state
        return state

    def _template(self, M: int, out: torch.Tensor) -> _LaunchTemplate:
        ldo = int(out.stride(0))
        key = (M, ldo, out.data_ptr() % 32)
        template = self._templates.get(key)
        if template is None:
            plan, _ldo = _resolve_plan(self.prepared, M, out, self._facts)
            stages = []
            with torch.cuda.device(self._device_index):
                for program, fixed in _stage_kwargs(
                    plan, self.prepared, self._facts, M, ldo, out.data_ptr()
                ):
                    name, defines = program
                    record = MODULES[name]
                    grid = dict(
                        zip(("grid_x", "grid_y", "grid_z"), fixed["grid"], strict=True)
                    )
                    slots: list[tuple[bool, Any]] = []
                    for kind, argument in record["arg_plan"]:
                        if kind == "grid":
                            slots.append((False, grid[argument]))
                        elif argument in _CALL_ARGS:
                            slots.append((True, argument))
                        elif argument in fixed:
                            slots.append((False, fixed[argument]))
                        else:
                            raise KeyError(
                                f"generated program {name!r} expects argument {argument!r} ({kind}); "
                                f"host binding provides {sorted(fixed) + sorted(_CALL_ARGS)}"
                            )
                    module = load_cake_kimi_k3_fp8_projection_module(
                        name, self._facts.arch, defines
                    )
                    # The register-epilogue GEMM programs take ``OUT`` as the per-device placeholder descriptor
                    # (see ``_output_bindings``); re-encoding it from the caller's view would fail for row strides
                    # that are not multiples of 16 bytes, so only the TMA-store programs re-bind ``OUT``.
                    real_out = (
                        plan.gemm_tma_store
                        if plan.decode is None
                        else plan.decode_tma_store
                    )
                    rebinds = [
                        (index, _REBIND_SOURCES[argument])
                        for index, (_kind, argument) in enumerate(record["arg_plan"])
                        if argument in _REBIND_SOURCES
                        and (real_out or argument != "OUT")
                    ]
                    if not 1 <= len(rebinds) <= _REBIND_SLOTS:
                        raise ValueError(
                            f"generated program {name!r} binds the call tensors at {len(rebinds)} positions; "
                            f"run_launch re-binds one to {_REBIND_SLOTS}"
                        )
                    rebinds += [(-1, 0)] * (_REBIND_SLOTS - len(rebinds))
                    entry = record["ffi_entry"]
                    stages.append(
                        _StageEntry(
                            getattr(module, f"{entry}_plan"),
                            getattr(module, f"{entry}_launch"),
                            getattr(module, f"{entry}_chain"),
                            tuple(slots),
                            tuple(rebinds),
                        )
                    )
            template = _LaunchTemplate(plan, ldo, tuple(stages))
            self._templates[key] = template
        return template

    # -- launch ---------------------------------------------------------------------------------------------------

    def __call__(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Hot path: one attribute query per fact the call needs (the shape and the strides once as tuples, the
        # device as its index), no ``torch.device`` / ``Stream`` object construction, no re-query of a value
        # already in hand.  The checks are the same as before; only their cost changed.
        shape = x.shape
        if (
            x.dtype is not _BF16
            or len(shape) != 2
            or shape[1] != self._K
            or not x.is_contiguous()
        ):
            raise ValueError(f"x must be a contiguous bf16 [M, {self._K}] tensor")
        if x.data_ptr() % 16:
            raise ValueError("x must be 16-byte aligned")  # see _activation_rows
        M = shape[0]
        if M < 1:
            raise ValueError("M must be positive")
        n_valid = self._n_valid
        device_index = self._device_index
        if not x.is_cuda or x.get_device() != device_index:
            raise ValueError(
                "x, out and the prepared weight must be on one CUDA device"
            )
        if out is None:
            out = torch.empty((M, n_valid), dtype=_BF16, device=self.device)
            ldo = n_valid
        else:
            shape = out.shape
            strides = out.stride()
            if (
                out.dtype is not _BF16
                or len(shape) != 2
                or shape[0] != M
                or shape[1] != n_valid
                or strides[1] != 1
                or strides[0] % 2
                or strides[0] < n_valid
                or out.storage_offset() % 2
            ):
                raise ValueError(
                    f"out must be a 4-byte aligned bf16 [{M}, {n_valid}] view with unit column stride and an "
                    f"even row stride >= {n_valid}"
                )
            if not out.is_cuda or out.get_device() != device_index:
                raise ValueError(
                    "x, out and the prepared weight must be on one CUDA device"
                )
            ldo = strides[0]
        key = (M, ldo, out.data_ptr() % 32)
        receipt = self._receipts.get(key)
        if receipt is None:
            receipt = self._plan_receipt(key, x, out)
        elif key != self._mru:
            self._receipts[key] = self._receipts.pop(key)  # most recently used last
            self._mru = key
        state = receipt.state
        if not state.pinned and _current_stream_capturing():
            state.pinned = True  # a graph captured after the eager resolution keeps referencing this workspace
        stream = _current_raw_stream(device_index)
        chain = receipt.chain
        if chain is not None:
            entry, record_0, codes_0, record_1, codes_1 = chain
            entry(stream, x, out, record_0, codes_0, record_1, codes_1)
        else:
            launch, record, i0, s0, i1, s1, i2, s2, i3, s3 = receipt.single
            bind = (x, out)
            launch(
                record, stream, i0, bind[s0], i1, bind[s1], i2, bind[s2], i3, bind[s3]
            )
        return out

    def _plan_receipt(
        self, key: tuple[int, int, int], x: torch.Tensor, out: torch.Tensor
    ) -> _Receipt:
        """Plan the launches of ``(M, ldo, out address class)`` into fresh host receipts (the full ``run`` argument
        validation and marshalling, once) and cache them, evicting the least recently used receipts."""
        template, call = self._call_bindings(key[0], x, out)
        state = self._workspaces[key[0]]
        stages = []
        for stage in template.stages:
            record = torch.empty(RECEIPT_BYTES, dtype=torch.uint8)
            stage.plan(
                record,
                *[call[value] if dynamic else value for dynamic, value in stage.slots],
            )
            stages.append((stage.launch, record, stage.rebinds))
        single = None
        if len(stages) == 1:
            chain = None
            launch, record, ((i0, s0), (i1, s1), (i2, s2), (i3, s3)) = stages[0]
            single = (launch, record, i0, s0, i1, s1, i2, s2, i3, s3)
        elif len(stages) == 2:
            chain = (
                template.stages[0].chain,
                stages[0][1],
                _slot_codes(stages[0][2]),
                stages[1][1],
                _slot_codes(stages[1][2]),
            )
        else:
            raise ValueError(
                f"route of M = {key[0]} launches {len(stages)} programs; the launcher chains one or two"
            )
        while len(self._receipts) >= self._max_receipts:
            del self._receipts[next(iter(self._receipts))]
        receipt = _Receipt(tuple(stages), chain, state, single)
        self._receipts[key] = receipt
        self._mru = key
        return receipt

    def _call_bindings(
        self, M: int, x: torch.Tensor, out: torch.Tensor
    ) -> tuple[_LaunchTemplate, dict[str, Any]]:
        """The template of ``(M, out)`` and the complete argument bindings of this call (workspace bindings cached
        per ``(M, route)``, output / activation bindings per call)."""
        template = self._template(M, out)
        state = self._workspace_state(M)
        plan = template.plan
        bound = state.bindings.get(plan.kernels)
        if bound is None:
            bound = _workspace_bindings(
                plan, self.prepared, M, state.workspace, self.device
            )
            state.bindings[plan.kernels] = bound
        call = dict(bound)
        call.update(
            _output_bindings(
                plan, self.prepared, M, template.ldo, x, out, self._device_index
            )
        )
        return template, call

    def pin(self, M: int) -> None:
        """Pin the workspace of ``M`` rows ahead of a CUDA-graph capture (so no per-call capture query is needed)."""
        self._workspace_state(int(M)).pinned = True

    def plan(self, x: torch.Tensor, out: torch.Tensor) -> ProjectionPlan:
        """The route plan a call with these bindings launches (resolving / caching its template)."""
        return self._template(int(x.shape[0]), out).plan

    @property
    def cached_rows(self) -> tuple[int, ...]:
        """The ``M`` values with a cached workspace, least recently used first."""
        return tuple(self._workspaces)

    @property
    def cached_receipts(self) -> int:
        """The number of planned ``(M, output row stride, output address class)`` launch receipts."""
        return len(self._receipts)


def kimi_k3_fp8_projection_launcher(
    prepared: PreparedProjectionWeight,
    *,
    max_workspaces: int = 64,
    max_receipts: int = 256,
) -> KimiK3Fp8ProjectionLauncher:
    """A :class:`KimiK3Fp8ProjectionLauncher` for ``prepared`` (one per engine linear layer)."""
    return KimiK3Fp8ProjectionLauncher(
        prepared, max_workspaces=max_workspaces, max_receipts=max_receipts
    )
