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
  the FP32 reduction order of those tiles differs from the plain schedule), or
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
from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Sequence

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
GEMM_SK_SUFFIX = "_sk"  # round 6 continuation 12 (lever SKO): the ordered stream-K instance of any GEMM epilogue program (table key ``gemm_sk``)
GEMM_SKF_SUFFIX = "_skf"  # round 6 continuation 17/18 (lever SKF): the fix-up stream-K instance of the TMA-store GEMM programs (table key ``gemm_skf``)
GEMM_SKF_MIN_ITERS = 4  # lever SKF: a pair's range of fewer K iterations cannot pay for the partial hand-offs
GEMM_SKF_MAX_SLOTS = 2  # lever SKF: contributor partial slots per tile the finisher adds (more contributors: plain schedule)
GEMM_SKF_SLOT_FORMS = (
    2,
    4,
    8,
)  # round 7 (CAKE-985): slot bounds of the shipped fix-up forms (``_skf`` = 2, ``_skf4`` / ``_skf8``)
GEMM_SKS_SUFFIX = "_sks"  # round 7 (CAKE-985, lever SKS): the sliced fix-up stream-K instance (table key ``gemm_sks``; ``_sks<slots>``)
GEMM_SK_MIN_K_ITERS = 4  # lever SKO: a head / tail chunk of fewer than two K iterations cannot pay for the partial hand-off
GEMM_SK_FLAG_BYTES = 2048  # stream-K flag area: 512 u32 flags >= 2 x the largest tile count of a split region (SKO: pairs / 2 tail tiles; SKF: < 2 pairs SK tiles)
GEMM_EPI_WARPS = (
    8  # epilogue warps per GEMM CTA (the head partial record is written per warp)
)
GEMM_SK_TILE_BYTES = (
    2 * GEMM_EPI_WARPS * 32 * 32 * 16
)  # lever SKO: one head partial tile (2 CTAs x 8 warps x 32 vectors x 32 lanes x 16 B = 256 KiB)
BLOCK_K = 256  # E4M3 elements per pipeline stage = two 128-K scale sets
CTA_GROUP = 2
WEIGHT_TILE_ROWS = 2 * BLOCK  # one CTA-pair N tile
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
    (192; 256 reproduces :func:`weight_scale_tiles`): ``[n_tiles][k_sets][2 blocks][512 B]``.

    The block-scaled MMA reads the B scales of logical N rows ``[128 b, 128 b + 128)`` from TMEM column block
    ``b``, so tile ``t`` carries the per-row bytes of weight rows ``[t bn, t bn + bn)`` in N order across its two
    128-lane blocks (lanes past ``bn`` zero); rows past the stored ``N_pad`` are zero, ``n_tiles = ceil(N_pad / bn)``."""
    if bn <= 0 or bn > WEIGHT_TILE_ROWS or bn % 32:
        raise ValueError(
            f"bn must be a positive multiple of 32 up to {WEIGHT_TILE_ROWS}, got {bn}"
        )
    n_blocks, kb = sf_bytes_128.shape
    n_rows = n_blocks * BLOCK
    ks = k_sets(K)
    n_tiles = -(-n_rows // bn)
    rows = sf_bytes_128.repeat_interleave(BLOCK, dim=0)  # [N_pad, K/128]
    per_row = torch.zeros((n_tiles * bn, ks * 4), dtype=torch.uint8, device=rows.device)
    per_row[:n_rows, : kb * 4] = rows.repeat_interleave(4, dim=1)
    tiles_rows = torch.zeros(
        (n_tiles, WEIGHT_TILE_ROWS, ks * 4), dtype=torch.uint8, device=rows.device
    )
    tiles_rows[:, :bn] = per_row.view(n_tiles, bn, ks * 4)
    tiles = swizzle_sf_128x4(
        tiles_rows.reshape(n_tiles * WEIGHT_TILE_ROWS, ks * 4)
    )  # flat (128-lane block, k_set, 512 B)
    return (
        tiles.reshape(n_tiles, 2, ks, SF_TILE_BYTES)
        .permute(0, 2, 1, 3)
        .contiguous()
        .reshape(-1)
    )


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


def decode_module_stages(
    tok: int,
    stages: int,
    fused: bool,
    resident: bool,
    xb_stages: int = 0,
    epi_chunk: int = 32,
) -> int:
    """Pipeline depth of the physical decode instance after its SMEM clamp (the Cake ``decode_ir`` rule).

    ``xb_stages`` > 0 is the decoupled fused variant: the BF16 token tiles stream through their own
    ``xb_stages``-deep ring instead of sharing the W / X stage."""
    tok_rows = max(tok, 32)
    xb_bytes = tok * 512 if (fused and not resident) else 0
    xb_ring = bool(fused and not resident and xb_stages > 0)
    stage_x_bytes = 0 if resident else tok_rows * BLOCK_K
    stage_bytes = (
        DEC_W_BYTES + stage_x_bytes + DEC_SF_BYTES + (0 if xb_ring else xb_bytes)
    )
    xb_ring_bytes = xb_stages * xb_bytes if xb_ring else 0
    res_bytes = DEC_RES_SLOTS * (tok_rows * BLOCK_K + 1024) if resident else 0
    epi_bytes = min(int(epi_chunk), 32, tok) * 128 * 4
    return max(
        1,
        min(
            stages,
            (DEC_SMEM_CAP - epi_bytes - res_bytes - xb_ring_bytes) // stage_bytes,
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
    },  # B300 / GB300 (152 SMs on the JHB nodes; same per-GPC cluster capacity as B200 -- measured values)
}


def decode_cs_alias_fits(
    tok: int, stages: int, fused: bool, resident: bool, csplit: int, xb_stages: int = 0
) -> bool:
    """Round 5: True when the one-round cluster exchange inbox + staging blocks
    ``(2C - 1) x 128 x (chunk | 1) x 4`` fit inside the physical instance's pipeline
    stages (host mirror of the Cake ``decode_cs_alias_fits`` rule); the fused view of a
    wide tile (1 stage at t128) does not fit and keeps the small-inbox exchange."""
    if csplit < 2:
        return False
    tok_rows = max(tok, 32)
    xb_bytes = tok * 512 if (fused and not resident) else 0
    xb_ring = bool(fused and not resident and xb_stages > 0)
    stage_x_bytes = 0 if resident else tok_rows * BLOCK_K
    stage_bytes = (
        DEC_W_BYTES + stage_x_bytes + DEC_SF_BYTES + (0 if xb_ring else xb_bytes)
    )
    module_stages = decode_module_stages(tok, stages, fused, resident, xb_stages)
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
) -> int:
    """Physical pipeline depth of the small-inbox (non-aliased) cluster split-K instance: the clamped depth of
    ``decode_module_stages`` reduced until a 4-row inbox round, ``(2C - 1) x 128 x 5 x 4`` bytes, fits next to the
    stages (host mirror of the Cake ``decode_cs_inbox_stages`` rule; the 7..16-wide clusters of round 6 give up one
    t16 stage)."""
    if csplit < 2:
        return decode_module_stages(tok, stages, fused, resident, xb_stages, epi_chunk)
    tok_rows = max(tok, 32)
    xb_bytes = tok * 512 if (fused and not resident) else 0
    xb_ring = bool(fused and not resident and xb_stages > 0)
    stage_x_bytes = 0 if resident else tok_rows * BLOCK_K
    stage_bytes = (
        DEC_W_BYTES + stage_x_bytes + DEC_SF_BYTES + (0 if xb_ring else xb_bytes)
    )
    xb_ring_bytes = xb_stages * xb_bytes if xb_ring else 0
    res_bytes = DEC_RES_SLOTS * (tok_rows * BLOCK_K + 1024) if resident else 0
    module_stages = decode_module_stages(
        tok, stages, fused, resident, xb_stages, epi_chunk
    )
    need = 5 * (2 * csplit - 1) * 128 * 4
    while (
        module_stages > 1
        and DEC_SMEM_CAP - module_stages * stage_bytes - res_bytes - xb_ring_bytes
        < need
    ):
        module_stages -= 1
    return module_stages


def decode_cs_small_inbox_rounds(
    tok: int, stages: int, fused: bool, resident: bool, csplit: int, xb_stages: int = 0
) -> int:
    """Round 5: exchange rounds of the small (non-aliased) cluster inbox next to the
    physical pipeline stages (host mirror of the Cake ``decode_cs_small_inbox_rounds``
    rule).  The aliased one-round exchange is selected only when this is > 1: with 4-8
    rows per rank the small inbox already holds the owner range in one round, and the
    all-rank ordering the aliased exchange needs costs ~1.6 us at C8."""
    if csplit < 2:
        return 1
    tok_rows = max(tok, 32)
    xb_bytes = tok * 512 if (fused and not resident) else 0
    xb_ring = bool(fused and not resident and xb_stages > 0)
    stage_x_bytes = 0 if resident else tok_rows * BLOCK_K
    stage_bytes = (
        DEC_W_BYTES + stage_x_bytes + DEC_SF_BYTES + (0 if xb_ring else xb_bytes)
    )
    xb_ring_bytes = xb_stages * xb_bytes if xb_ring else 0
    res_bytes = DEC_RES_SLOTS * (tok_rows * BLOCK_K + 1024) if resident else 0
    module_stages = decode_module_stages(tok, stages, fused, resident, xb_stages)
    need = 5 * (2 * csplit - 1) * 128 * 4
    while (
        module_stages > 1
        and DEC_SMEM_CAP - module_stages * stage_bytes - res_bytes - xb_ring_bytes
        < need
    ):
        module_stages -= 1
    budget = DEC_SMEM_CAP - module_stages * stage_bytes - res_bytes - xb_ring_bytes
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
    qlanes: int = (
        16  # lanes per quantization unit (16 = half-warp units, 8 / 4 = narrow units)
    )
    csplit: int = 1  # round 5: K split across the CTAs of one cluster (== split); the partials meet in SMEM (DSM)
    cs_alias: bool = False  # round 5: the DSM inbox aliases the dead pipeline stages (one exchange round); only when every CTA owns one work item
    epi_chunk: int = 32  # round 6: epilogue staging rows per flush (table key ``epi_chunk``; 16 frees SMEM for the 5-stage t32 ring)
    pf: int = 0  # round 6 (lever P): weight-tile L2 prefetch distance in stages (table key ``pf``; 0 = off)
    mc: int = 1  # round 6 (lever M): m tiles of one N tile per cluster sharing the W stage through TMA multicast (table key ``mc``; 1 = off)
    pfx: int = 0  # round 6 next loop (lever PX): BF16 token-tile L2 prefetch distance in stages (table key ``pfx``; fused, non-resident rows only; 0 = off)
    tstore: bool = False  # round 6 (lever E1): the split-1 epilogue stores BF16 through TMA (table key ``tstore``; the launch still needs a 16-byte-aligned output view)
    pfi: int = 0  # round 6 continuation 7 (lever PI-W): cross-item L2 prefetch of the next work item's first W/SFW stages by the load warp (table key ``pfi``; multi-item rows with pf > 0; 0 = off)
    qwarps: int = DEC_QUANT_WARPS  # round 6 continuation 8 (lever QW16): quantizing warps of the fused instance (table key ``qwarps``; 8 = the round-3 default, 16 on the fused 16384 rows)
    xbh: bool = False  # round 6 continuation 9 (lever XBH): half-slot BF16 ring -- the two 128-K blocks of a stage are loaded and released separately (table key ``xbh``; fused ring rows with narrow units)
    qer: bool = False  # round 6 continuation 10 (lever QER): early half-slot release -- each quantizing warp frees its BF16 half slot right after its register loads, before the conversion (table key ``qer``; xbh rows only)

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
            epi_chunk=self.epi_chunk,
            pf=self.pf,
            mc=self.mc,
            pfx=self.pfx,
            tstore=bool(tma_store)
            and self.tstore
            and self.split == 1
            and self.csplit == 1,
            pfi=self.pfi,
            qwarps=self.qwarps,
            xbh=self.xbh,
            qer=self.qer,
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
    # Table key ``stages`` (round 6, lever D): the row pins the ring depth past ``DEC_MAX_STAGES`` (5 x 43008 B + an 8 KB
    # epilogue chunk fit the pool at t32); the physical instance still applies the SMEM clamp (``decode_module_stages``).
    if entry.get("stages"):
        stages = int(entry["stages"])
    # Table key ``pf`` (round 6, lever P): the weight tiles (+ weight scales) are prefetched into L2 ``pf`` stages ahead.
    pf = int(entry.get("pf", 0))
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
    # Table key ``tstore`` (round 6, lever E1): the split-1 epilogue stores BF16 through TMA; the instance has no TMA path
    # for the split-K / cluster reductions.
    tstore = bool(entry.get("tstore", False))
    if tstore and (split != 1 or csplit > 1):
        raise ValueError(
            f"decode table entry tstore needs split 1 and no cluster split-K (got {entry})"
        )
    # Table key ``pfx`` (round 6 next loop, lever PX): the BF16 token tile of the fused variant is prefetched into L2
    # ``pfx`` stages ahead of its TMA load (its DRAM access then precedes the weight burst instead of queueing behind
    # it); fused, non-resident rows only (host mirror of the Cake ``decode_config`` rule).
    pfx = int(entry.get("pfx", 0)) if (fused and not resident) else 0
    # Table key ``pfi`` (round 6 continuation 7, lever PI-W): during the last ``pf`` stages of a work item the load warp
    # also prefetches the NEXT item's first W/SFW tiles into L2 (host mirror of the Cake ``decode_config`` rule: pf > 0).
    pfi = int(entry.get("pfi", 0)) if pf > 0 else 0
    cs_alias = (
        csplit > 1
        and grid == total_work
        and decode_cs_alias_fits(tok, stages, fused, resident, csplit, xb_stages)
        and decode_cs_small_inbox_rounds(
            tok, stages, fused, resident, csplit, xb_stages
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
        stages=stages,
        # The small-inbox cluster exchange gives up pipeline depth until the inbox fits (round 6: one t16 stage at C7..C16).
        module_stages=decode_cs_inbox_stages(
            tok, stages, fused, resident, csplit, xb_stages, epi_chunk
        )
        if csplit > 1 and not cs_alias
        else decode_module_stages(tok, stages, fused, resident, xb_stages, epi_chunk),
        m_tiles=m_tiles,
        tiles=tiles,
        total_work=total_work,
        grid=grid,
        tok_per_cta=-(-tok // split),
        xb_stages=xb_stages,
        qlanes=qlanes,
        csplit=csplit,
        epi_chunk=epi_chunk,
        pf=pf,
        mc=mc,
        pfx=pfx,
        pfi=pfi,
        qwarps=qwarps,
        xbh=xbh,
        qer=qer,
        cs_alias=cs_alias,
    )


def gemm_prefetch_distance(M: int, n_tiles128: int, num_k_iters: int, arch: str) -> int:
    """Round 6 (lever GP): weight-tile L2 prefetch distance of the GEMM route from the shape's table row (key
    ``gemm_pf``; 0 = off). Only tabulated GEMM-routed rows prefetch (e.g. the 48-pair ``tp1:q_proj:256`` whose 3-stage
    weight stream is HBM-latency-bound: 1.14x on both architectures)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return 0
    return int(entry.get("gemm_pf", 0))


def gemm_block_n(
    M: int, n_tiles128: int, num_k_iters: int, arch: str, n_valid: int, n_pad: int
) -> int:
    """Round 6 (lever L5): output columns per CTA pair of the GEMM route, from the shape's table row (key
    ``gemm_bn``: 192 on the tabulated N = 576 ``kv_a`` rows at M > 256, whose 256-wide tiles stream and multiply
    768 padded columns; 256 otherwise).  192 is only taken when the 192-padded N fits the stored 256-padded
    weight rows, so no weight box reads past the storage."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return BLOCK_N
    bn = int(entry.get("gemm_bn", BLOCK_N))
    if bn not in (GEMM_BLOCK_N_NARROW, BLOCK_N):
        raise ValueError(f"decode table entry gemm_bn must be 192 or 256 (got {entry})")
    if bn != BLOCK_N and -(-int(n_valid) // bn) * bn > int(n_pad):
        return BLOCK_N
    return bn


def gemm_n_tiles(prepared: "PreparedProjectionWeight", bn: int) -> int:
    """CTA-pair N tiles of the GEMM instance with ``bn`` output columns per pair over ``prepared``."""
    return prepared.n_tiles if int(bn) == BLOCK_N else -(-prepared.n_valid // int(bn))


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
    inst_slots: int = GEMM_SKF_MAX_SLOTS  # round 7 (CAKE-985): slot bound of the launched instance form
    sliced: bool = False  # round 7 (CAKE-985, lever SKS): sliced range mapping (``pairs`` = tiles, ``total`` = slices)

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
        # round 7 (CAKE-985, lever SKS): S slices per tile on tiles x S resident pairs (Cake host mirror)
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


def _sk_layout(
    prepared: "PreparedProjectionWeight", M: int, arch: str, sm_count: int
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


def _sk_launch_kwargs(
    plan: "ProjectionPlan", sf: torch.Tensor, device: torch.device
) -> dict[str, Any]:
    """The six stream-K launch arguments of a GEMM program (lever SKO): the hand-off area and the split scalars of
    ``plan.gemm_sk``, or zero dummies for the plain schedule (every GEMM program takes the parameters)."""
    sk = plan.gemm_sk
    skf = plan.gemm_skf
    if sk is None and skf is not None:
        # round 6 continuation 17/18 (lever SKF): the fix-up form's region length / slots / first tile travel in the
        # same three scalars (``sk_rem`` / ``sk_ksplit`` / ``sk_dp``) of the ``_skf`` program
        f_off = plan.counters_offset
        p_off = f_off + GEMM_SK_FLAG_BYTES
        p_bytes = skf.partial_bytes
        if sf.numel() < p_off + p_bytes:
            raise ValueError(
                f"workspace.sf must hold the stream-K partial area: >= {p_off + p_bytes} bytes "
                "(allocate_kimi_k3_fp8_projection_workspace)"
            )
        return dict(
            sk_partials=sf[p_off : p_off + p_bytes].view(torch.float32),
            sk_flags=sf[f_off : f_off + GEMM_SK_FLAG_BYTES].view(torch.uint32),
            sk_pairs=skf.pairs,
            sk_rem=skf.total,
            sk_ksplit=skf.slots,
            sk_dp=skf.first,
        )
    if sk is None:
        partials, flags = _sk_dummy(device)
        return dict(
            sk_partials=partials,
            sk_flags=flags,
            sk_pairs=0,
            sk_rem=0,
            sk_ksplit=0,
            sk_dp=0,
        )
    f_off = plan.counters_offset
    p_off = f_off + GEMM_SK_FLAG_BYTES
    p_bytes = sk.rem * GEMM_SK_TILE_BYTES
    if sf.numel() < p_off + p_bytes:
        raise ValueError(
            f"workspace.sf must hold the stream-K hand-off area: >= {p_off + p_bytes} bytes "
            "(allocate_kimi_k3_fp8_projection_workspace)"
        )
    return dict(
        sk_partials=sf[p_off : p_off + p_bytes].view(torch.float32),
        sk_flags=sf[f_off : f_off + GEMM_SK_FLAG_BYTES].view(torch.uint32),
        sk_pairs=sk.pairs,
        sk_rem=sk.rem,
        sk_ksplit=sk.ksplit,
        sk_dp=sk.dp,
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
        if entry.get("route") == "gemm":
            gpf, gbn = int(entry.get("gemm_pf", 0)), int(entry.get("gemm_bn", BLOCK_N))
            sfx = GEMM_N192_SUFFIX if gbn != BLOCK_N else ""
            gsk = bool(entry.get("gemm_sk", 0))
            gskf = bool(entry.get("gemm_skf", 0))
            gsks = bool(entry.get("gemm_sks", 0))
            gkeys: list[str] = []
            if gpf > 0:
                # round 6 (lever GP): the prefetching GEMM program of the tabulated row's production epilogue (16-byte
                # aligned views: TMA store); other views fall back to the non-prefetching program (``route_plan``)
                gkeys.append(gemm_kernel_key(GEMM_TSTORE_KERNEL_KEY + sfx, gpf))
            if sfx:
                # round 6 (lever L5): the 192-wide programs of the row's production epilogues (16-byte aligned views:
                # TMA store; 8-byte aligned rows: staged register epilogue); other views fall back to the 256-wide program
                gkeys += [GEMM_TSTORE_KERNEL_KEY + sfx, GEMM_RSTAGED_KERNEL_KEY + sfx]
            if gsk:
                # round 6 continuation 12 (lever SKO): the ordered stream-K instance of the row's production epilogue (16-byte
                # aligned views: TMA store, with the row's prefetch distance -- the only stream-K program the Cake export plan
                # ships); other views and launches outside the stream-K wave window fall back to the plain programs
                # (``route_plan``)
                gkeys.append(
                    gemm_kernel_key(GEMM_TSTORE_KERNEL_KEY + sfx + GEMM_SK_SUFFIX, gpf)
                )
            if gskf:
                # round 6 continuation 17/18 (lever SKF): the fix-up stream-K instance of the row's TMA-store program (the
                # only fix-up program the Cake export plan ships); other views fall back to the plain programs.  Round 7:
                # a slot bound of 4 / 8 in the cell names the wider form.
                skf_key = int(entry.get("gemm_skf", 0))
                skf_sfx = GEMM_SKF_SUFFIX + (
                    str(skf_key)
                    if skf_key in GEMM_SKF_SLOT_FORMS and skf_key != GEMM_SKF_MAX_SLOTS
                    else ""
                )
                gkeys.append(
                    gemm_kernel_key(GEMM_TSTORE_KERNEL_KEY + sfx + skf_sfx, gpf)
                )
            if gsks:
                # round 7 (CAKE-985, lever SKS): the sliced fix-up forms the row may launch (the slot form follows the
                # row count inside the bucket: every form is listed, unavailable ones fall back in ``route_plan``)
                for form in GEMM_SKF_SLOT_FORMS:
                    gkeys.append(
                        gemm_kernel_key(
                            GEMM_TSTORE_KERNEL_KEY + sfx + f"{GEMM_SKS_SUFFIX}{form}",
                            gpf,
                        )
                    )
            for gkey in gkeys:
                if gkey not in keys:
                    keys.append(gkey)
        for M in _bucket_rows(bucket):
            cfg = decode_config(M, n_tiles128, num_k_iters, arch, sm_count)
            if cfg is None:
                continue
            if cfg.kernel_key not in keys:
                keys.append(cfg.kernel_key)
            # round 6 (lever E1): tstore rows launch the ``_tso`` program on 16-byte-aligned output views and fall back to
            # the register-epilogue program otherwise, so both programs belong to the plan
            tso_key = cfg.kernel_key_for(True)
            if tso_key not in keys:
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
    gemm_n_tiles: int  # GEMM route: CTA-pair N tiles of the launched program (``n_pad / 256`` or ``ceil(n_valid / 192)``)
    gemm_sk: Optional[
        StreamKPlan
    ]  # GEMM route: the ordered stream-K split of the launched ``_sk`` program, None for the plain CLC schedule (round 6 continuation 12, lever SKO)
    gemm_skf: Optional[
        StreamKFixupPlan
    ]  # GEMM route: the fix-up stream-K split of the launched ``_skf`` program, None otherwise (round 6 continuation 17/18, lever SKF)
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
    c_off, c_bytes, p_off, p_bytes = reduction_layout(prepared, M, cfg)
    kernels: list[str] = []
    grids: list[int] = []
    units: Optional[int] = None
    if cfg is None or not cfg.fused:
        k_blocks = prepared.K // BLOCK
        units = quant_units(M, k_blocks, sm_count)
        units_per_row = k_blocks // units
        kernels.append(quant_kernel_key(units))
        grids.append(-(-(M * units_per_row) // (QUANT_WARPS * 2)))
    if cfg is None:
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
        # round 6 (lever GP): the prefetching program ships with the TMA-store epilogue only (Cake rule); use it when
        # registered.  Round 6 (lever L5): the 192-wide instance of the row's epilogue program when the table row asks
        # for it and the program is registered; every fallback keeps the row on a registered program.
        stems: list[tuple[str, int, int]] = []  # (program stem, prefetch distance, bn)
        if gbn != BLOCK_N and base != GEMM_KERNEL_KEY:
            # round 7 (CAKE-985): the 192-wide form ships for the TMA-store and staged-register epilogues only (Cake
            # launcher rule); the predicated register epilogue keeps the 256-wide program
            stems.append((base + GEMM_N192_SUFFIX, gpf, gbn))
            if gpf:
                stems.append((base + GEMM_N192_SUFFIX, 0, gbn))
        if gpf:
            stems.append((base, gpf, BLOCK_N))
        stems.append((base, 0, BLOCK_N))
        # round 6 continuation 12 (lever SKO): the ordered stream-K instance (``_sk`` before ``_pf``) of each candidate when
        # the table row asks for it and the launch sits in the stream-K wave window (one full wave plus at most half a
        # wave of tail tiles), falling back to the plain program of the same stem
        candidates: list[
            tuple[str, int, Optional[StreamKPlan], Optional[StreamKFixupPlan]]
        ] = []
        for stem, pf, bn in stems:
            # round 7 (CAKE-985): the ordered stream-K form ships for the TMA-store epilogue only, like the fix-up form
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
                candidates.append(
                    (gemm_kernel_key(stem + GEMM_SK_SUFFIX, pf), bn, sk, None)
                )
            else:
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
                    if stem.startswith(GEMM_TSTORE_KERNEL_KEY)
                    else None
                )
                if skf is not None:
                    candidates.append(
                        (gemm_kernel_key(stem + skf.program_suffix, pf), bn, None, skf)
                    )
            candidates.append((gemm_kernel_key(stem, pf), bn, None, None))
        key, gbn, sk_plan, skf_plan = next(
            ((k, b, s, f) for k, b, s, f in candidates if route_available(arch, (k,))),
            (base, BLOCK_N, None, None),
        )
        kernels.append(key)
        grids.append(
            sk_plan.grid
            if sk_plan is not None
            else skf_plan.grid
            if skf_plan is not None
            else _gemm_grid(_m_tiles(M), gemm_n_tiles(prepared, gbn))
        )
    else:
        gbn = BLOCK_N
        sk_plan = None
        skf_plan = None
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
        gemm_n_tiles=gemm_n_tiles(prepared, gbn),
        gemm_sk=sk_plan,
        gemm_skf=skf_plan,
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
                    SFB=prepared.scale_tiles_for(plan.gemm_bn).view(-1, 8, 128),
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
    """The four stream-K split scalars of a GEMM program (the two hand-off tensors are workspace bindings)."""
    sk = plan.gemm_sk
    skf = plan.gemm_skf
    if sk is None and skf is not None:
        return dict(
            sk_pairs=skf.pairs, sk_rem=skf.total, sk_ksplit=skf.slots, sk_dp=skf.first
        )
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
class _LaunchTemplate:
    """The launch sequence of one ``(M, output row stride, output address class)`` with the call-bound arguments
    (_CALL_ARGS) left as names: ``stages[i] = (entry, ((is_call_arg, value_or_name), ...))``."""

    plan: ProjectionPlan
    ldo: int
    stages: tuple[tuple[Callable[..., Any], tuple[tuple[bool, Any], ...]], ...]


@dataclass
class _WorkspaceState:
    workspace: ProjectionWorkspace
    bindings: dict[tuple[str, ...], dict[str, Any]]
    pinned: bool = False


class KimiK3Fp8ProjectionLauncher:
    """Allocation-light repeated launches of one prepared weight (an engine's linear layer).

    ``launcher(x, out)`` runs ``out[:, :n_valid] = bf16(x @ dequant(w2, s2).T)`` for any ``M = x.shape[0]``: the
    route plan and the per-stage argument templates are resolved once per ``(M, row stride, output address class)``,
    the workspace of each ``M`` is allocated once and reused (the programs leave it reusable), and a call only binds
    the call's tensors (``x``, the output view, the cached workspace views) before the launches -- no route
    resolution, validation of the prepared weight, workspace allocation or host synchronisation per call.

    ``max_workspaces`` bounds the per-``M`` workspace cache (least recently used ``M`` evicted first); an ``M`` used
    inside CUDA-graph capture (first seen there or resolved eagerly before) is pinned, because the captured graph
    keeps referencing its workspace.  One
    launcher serves one stream order: concurrent launches of the same ``M`` from several streams would share a
    workspace.  Resolve a new ``(M, out)`` class once eagerly before capturing a graph of it (the JIT modules of the
    route load on first use)."""

    def __init__(self, prepared: PreparedProjectionWeight, *, max_workspaces: int = 64):
        if int(max_workspaces) < 1:
            raise ValueError("max_workspaces must be positive")
        self.prepared = prepared
        self.device = prepared.device
        self._device_index = _device_index(prepared.device)
        self._facts = device_facts(prepared.device)
        self._max_workspaces = int(max_workspaces)
        self._templates: dict[tuple[int, int, int], _LaunchTemplate] = {}
        self._workspaces: dict[int, _WorkspaceState] = {}  # insertion order = recency

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
                    stages.append((getattr(module, record["ffi_entry"]), tuple(slots)))
            template = _LaunchTemplate(plan, ldo, tuple(stages))
            self._templates[key] = template
        return template

    # -- launch ---------------------------------------------------------------------------------------------------

    def __call__(
        self, x: torch.Tensor, out: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        prepared = self.prepared
        if (
            x.dtype != torch.bfloat16
            or x.dim() != 2
            or int(x.shape[1]) != prepared.K
            or not x.is_contiguous()
        ):
            raise ValueError(f"x must be a contiguous bf16 [M, {prepared.K}] tensor")
        M = int(x.shape[0])
        if M < 1:
            raise ValueError("M must be positive")
        if out is None:
            out = torch.empty(
                (M, prepared.n_valid), dtype=torch.bfloat16, device=self.device
            )
        elif (
            out.dtype != torch.bfloat16
            or out.dim() != 2
            or tuple(out.shape) != (M, prepared.n_valid)
            or out.stride(1) != 1
            or out.stride(0) % 2
            or out.stride(0) < prepared.n_valid
            or (out.storage_offset() * 2) % 4
        ):
            raise ValueError(
                f"out must be a 4-byte aligned bf16 [{M}, {prepared.n_valid}] view with unit column stride and an "
                f"even row stride >= {prepared.n_valid}"
            )
        if x.device != self.device or out.device != self.device:
            raise ValueError(
                "x, out and the prepared weight must be on one CUDA device"
            )
        template = self._template(M, out)
        state = self._workspace_state(M)
        plan = template.plan
        bound = state.bindings.get(plan.kernels)
        if bound is None:
            bound = _workspace_bindings(plan, prepared, M, state.workspace, self.device)
            state.bindings[plan.kernels] = bound
        call = dict(bound)
        call.update(
            _output_bindings(
                plan, prepared, M, template.ldo, x, out, self._device_index
            )
        )
        with tvm_ffi.use_torch_stream():
            for entry, slots in template.stages:
                entry(*[call[value] if dynamic else value for dynamic, value in slots])
        return out

    def plan(self, x: torch.Tensor, out: torch.Tensor) -> ProjectionPlan:
        """The route plan a call with these bindings launches (resolving / caching its template)."""
        return self._template(int(x.shape[0]), out).plan

    @property
    def cached_rows(self) -> tuple[int, ...]:
        """The ``M`` values with a cached workspace, least recently used first."""
        return tuple(self._workspaces)


def kimi_k3_fp8_projection_launcher(
    prepared: PreparedProjectionWeight, *, max_workspaces: int = 64
) -> KimiK3Fp8ProjectionLauncher:
    """A :class:`KimiK3Fp8ProjectionLauncher` for ``prepared`` (one per engine linear layer)."""
    return KimiK3Fp8ProjectionLauncher(prepared, max_workspaces=max_workspaces)
