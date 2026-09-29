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
  register epilogue for other strides, or
* ``decode:t<tok>_p<stages>[_fused][_res]`` -- the swap-AB split-K decode kernel
  (M <= 256, and the single-N-tile families the measured table routes here up
  to 16384 rows; the ``_fused`` instances quantize the token tile in-CTA, so
  the quantization launch is skipped).

Host work is split exactly like the Cake production launcher:

* :func:`prepare_kimi_k3_fp8_projection_weights` -- once per weight: UE8M0
  requantization, 256-row padding, the 128x128 E4M3 tile order the TMA streams
  (one contiguous 32 KB box per pipeline stage) and the swizzled weight scale
  tiles.
* :func:`allocate_kimi_k3_fp8_projection_workspace` -- once per ``M``: the E4M3
  activation ``[M, K]`` and the auxiliary byte workspace (activation scale
  tiles, split-K partial tiles and self-resetting per-tile counters; zero
  initialised once).
* :func:`prepare_kimi_k3_fp8_projection` -- selects the route from the measured
  per-architecture dispatch table (:mod:`.decode_table`) and binds the launch
  sequence to the generated argument plans.  The returned runner's ``launch()``
  performs no allocation and no host synchronisation and is CUDA-graph
  capturable.

The host dispatch reproduces the Cake dispatcher's measured table for the
representative ``(N, K)`` families; the generated-program export checks route,
configuration and bitwise output parity on every contract row.  See
``README.md`` in this package and flashinfer-ai/flashinfer#4568 (tracker #4254).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, NamedTuple, Optional, Sequence

import torch
import tvm_ffi

from .cake_jit import (
    GEMM_KERNEL_KEY,
    GEMM_RSTAGED_KERNEL_KEY,
    GEMM_TSTORE_KERNEL_KEY,
    MODULES,
    decode_kernel_key,
    gemm_kernel_key,
    kernel_module_name,
    load_cake_kimi_k3_fp8_projection_module,
    quant_kernel_key,
    route_available,
)
from .decode_table import DECODE_TABLE

# ---------------------------------------------------------------------------
# Fixed geometry of the generated programs
# ---------------------------------------------------------------------------

BLOCK = 128  # scale granularity along K (activations) and along N x K (weights)
BLOCK_M = 128  # activation rows per GEMM CTA (256 per CTA pair)
BLOCK_N = 256  # output columns per GEMM CTA pair
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
    4096,
    16384,
)  # M buckets of the measured dispatch table (the two large
#                                                     buckets only list families measured faster than the GEMM)
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
ARCHES = tuple(sorted(DECODE_TABLE))

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


def quant_units(M: int, k_blocks: int) -> int:
    """K blocks per half warp of the quantization launch: 1 for M <= 256 (maximal parallelism), 4 / 2 / 8
    for large M when they divide the K blocks and keep at least four CTAs per SM busy."""
    if M <= DECODE_MAX_M:
        return 1
    for units in (4, 2, 8):
        if (
            k_blocks % units == 0
            and (M * k_blocks) // units >= 148 * QUANT_WARPS * 2 * 4
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


# Co-resident cluster capacity of the cluster split-K decode instances per architecture and cluster size
# (``cuOccupancyMaxActiveClusters`` of the ``_cs<C>`` instance; one CTA per SM, so it depends only on the GPC topology).
# A persistent cluster grid larger than ``C x capacity`` serialises whole clusters into a second pass.  Measured on
# B200 / B300; mirrors the source repository's table.
DECODE_MAX_ACTIVE_CLUSTERS: dict[str, dict[int, int]] = {
    "sm_100a": {2: 74, 3: 45, 4: 33, 5: 26, 6: 22, 7: 15, 8: 15},  # B200, 148 SMs
    "sm_103a": {
        2: 74,
        3: 45,
        4: 33,
        5: 26,
        6: 22,
        7: 15,
        8: 15,
    },  # B300, 148 SMs (same GPC topology)
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

    @property
    def tok_rows(self) -> int:
        return max(self.tok, 32)

    @property
    def kernel_key(self) -> str:
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
        )


def decode_table_entry(
    M: int, n_tiles128: int, num_k_iters: int, arch: str
) -> Optional[dict[str, Any]]:
    """Measured table entry for the shape, or ``None`` when the architecture / family is not tabulated."""
    per_arch = DECODE_TABLE.get(arch)
    if not per_arch:
        return None
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
    is covered on both architectures).  Above ``DECODE_MAX_M`` rows only the tabulated narrow-N families (whose
    activation stream the 2-CTA GEMM cannot spread over the SMs) keep the decode route."""
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
        if csplit > 8 or csplit > tok or csplit > int(num_k_iters):
            raise ValueError(
                f"decode table entry csplit {csplit} needs 2 <= C <= min(8, tok {tok}, num_k_iters {num_k_iters})"
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
        # whole m-tile groups take the same route without the multicast cluster (host mirror of the Cake rule).
        mc = 1
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
    while qlanes < 16 and (2 * tok) % (DEC_QUANT_WARPS * (32 // qlanes)):
        qlanes *= 2
    return DecodeConfig(
        tok=tok,
        split=split,
        fused=fused,
        resident=resident,
        persist=persist,
        stages=stages,
        module_stages=decode_module_stages(
            tok, stages, fused, resident, xb_stages, epi_chunk
        ),
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
        cs_alias=csplit > 1
        and grid == total_work
        and decode_cs_alias_fits(tok, stages, fused, resident, csplit, xb_stages)
        and decode_cs_small_inbox_rounds(
            tok, stages, fused, resident, csplit, xb_stages
        )
        > 1,
    )


def gemm_prefetch_distance(
    M: int, n_tiles128: int, num_k_iters: int, arch: str
) -> int:
    """Round 6 (lever GP): weight-tile L2 prefetch distance of the GEMM route from the shape's table row (key
    ``gemm_pf``; 0 = off). Only tabulated GEMM-routed rows prefetch (e.g. the 48-pair ``tp1:q_proj:256`` whose 3-stage
    weight stream is HBM-latency-bound: 1.14x on both architectures)."""
    entry = decode_table_entry(M, n_tiles128, num_k_iters, arch)
    if entry is None or entry.get("route") != "gemm":
        return 0
    return int(entry.get("gemm_pf", 0))


def required_kernel_keys(arch: str, sm_count: int = 148) -> tuple[str, ...]:
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
    for key, entry in DECODE_TABLE.get(arch, {}).items():
        n_tiles128, num_k_iters, bucket = (int(v) for v in key.split(","))
        if entry.get("route") == "gemm" and int(entry.get("gemm_pf", 0)) > 0:
            # round 6 (lever GP): the prefetching GEMM program of the tabulated row's production epilogue (16-byte
            # aligned views: TMA store); other views fall back to the non-prefetching program (``route_plan``)
            gkey = gemm_kernel_key(GEMM_TSTORE_KERNEL_KEY, int(entry["gemm_pf"]))
            if gkey not in keys:
                keys.append(gkey)
        for M in _bucket_rows(bucket):
            cfg = decode_config(M, n_tiles128, num_k_iters, arch, sm_count)
            if cfg is not None and cfg.kernel_key not in keys:
                keys.append(cfg.kernel_key)
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
    splits: Optional[tuple[int, ...]] = None

    @property
    def device(self) -> torch.device:
        return self.weight_tiles.device

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
        splits=splits,
    )


# ---------------------------------------------------------------------------
# Workspaces
# ---------------------------------------------------------------------------


class ProjectionWorkspace(NamedTuple):
    """Caller-owned workspaces of one ``M``: the E4M3 activation ``q [M, K]`` and the auxiliary byte buffer
    ``sf`` (activation scale tiles + decode split-K partials + per-tile counters; zero initialised once)."""

    q: torch.Tensor
    sf: torch.Tensor


def _device_arch(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the Kimi-K3 FP8 projection requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    return arch


def _sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


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
    return p_off + p_bytes


def allocate_kimi_k3_fp8_projection_workspace(
    prepared: PreparedProjectionWeight, M: int
) -> ProjectionWorkspace:
    """Allocate the caller-owned workspaces for ``M`` activation rows on the weight's device (no launch)."""
    M = int(M)
    if M < 1:
        raise ValueError("M must be positive")
    device = prepared.device
    arch = _device_arch(device)
    q = torch.empty((M, prepared.K), dtype=torch.float8_e4m3fn, device=device)
    sf = torch.zeros(
        (workspace_sf_bytes(prepared, M, arch, _sm_count(device)),),
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
    """Staged row-coalesced register epilogue (round 5) for output views the TMA store cannot address but whose rows
    are at least 8-byte aligned (``store_vec >= 4``; e.g. ``n_valid = 6284``): the tile is staged in SMEM and copied
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
    quant_units: Optional[int]  # None when the decode instance quantizes in-CTA
    sf_rows: int
    kernels: tuple[str, ...]  # logical kernel key per launch, in launch order
    modules: tuple[str, ...]  # registered module per launch
    grids: tuple[int, ...]
    counters_offset: int
    partials_offset: int
    partials_bytes: int


def route_plan(
    prepared: PreparedProjectionWeight,
    M: int,
    arch: str,
    sm_count: int,
    gemm_tma_store: bool = True,
    gemm_reg_staged: bool = False,
) -> ProjectionPlan:
    """Resolve the launch sequence of ``M`` rows without touching device memory.

    ``gemm_tma_store`` / ``gemm_reg_staged`` select the GEMM epilogue program (``gemm_tma_store_eligible`` /
    ``gemm_reg_staged_eligible`` of the output view)."""
    M = int(M)
    cfg = decode_config(M, prepared.n_tiles128, prepared.num_k_iters, arch, sm_count)
    c_off, _c_bytes, p_off, p_bytes = reduction_layout(prepared, M, cfg)
    kernels: list[str] = []
    grids: list[int] = []
    units: Optional[int] = None
    if cfg is None or not cfg.fused:
        k_blocks = prepared.K // BLOCK
        units = quant_units(M, k_blocks)
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
        gpf = gemm_prefetch_distance(
            M, prepared.n_tiles128, prepared.num_k_iters, arch
        )
        # round 6 (lever GP): the prefetching program when it is registered for this epilogue, else the plain one
        key = gemm_kernel_key(base, gpf)
        kernels.append(key if gpf and route_available(arch, (key,)) else base)
        grids.append(_gemm_grid(_m_tiles(M), prepared.n_tiles))
    else:
        kernels.append(cfg.kernel_key)
        grids.append(cfg.grid)
    return ProjectionPlan(
        arch=arch,
        sm_count=int(sm_count),
        M=M,
        route="decode" if cfg is not None else "gemm",
        decode=cfg,
        gemm_tma_store=cfg is None and bool(gemm_tma_store),
        gemm_reg_staged=cfg is None and not gemm_tma_store and bool(gemm_reg_staged),
        quant_units=units,
        sf_rows=cfg.tok if cfg is not None else SF_TILE_ROWS,
        kernels=tuple(kernels),
        modules=tuple(kernel_module_name(arch, key) for key in kernels),
        grids=tuple(grids),
        counters_offset=c_off,
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
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return False
    if M is None and prepared is None:
        return bool(MODULES) and route_available(
            arch, required_kernel_keys(arch, _sm_count(device))
        )
    if M is None or prepared is None:
        raise ValueError("pass both M and prepared or neither")
    try:
        route_plan(prepared, int(M), arch, _sm_count(device))
    except NotImplementedError:
        return False
    return True


def _bind(module_name: str, kwargs: dict[str, Any]) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` and load its entry."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_kimi_k3_fp8_projection_module(module_name)
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


def validate_kimi_k3_fp8_projection_inputs(
    x: torch.Tensor,
    prepared: PreparedProjectionWeight,
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
    *,
    arch: str,
    sm_count: int,
) -> int:
    """Shape / dtype / alignment validation of one call; returns ``M``."""
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
    needed = workspace_sf_bytes(prepared, M, arch, sm_count)
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


def prepare_kimi_k3_fp8_projection(
    x: torch.Tensor,
    prepared: PreparedProjectionWeight,
    out: torch.Tensor,
    workspace: ProjectionWorkspace,
) -> KimiK3Fp8ProjectionRunner:
    """Validate the binding, select the route and bind the launch sequence.

    ``out`` is a ``[M, n_valid]`` BF16 view with unit column stride and an even row stride (any view into a
    wider buffer); ``workspace`` comes from :func:`allocate_kimi_k3_fp8_projection_workspace` for this ``M``
    (its ``sf`` buffer must have been zero initialised once; the programs keep the counters reset).  The JIT
    modules of the route are built and loaded here, so prepare outside CUDA Graph capture."""
    device = prepared.device
    arch = _device_arch(device)
    device_index = (
        device.index if device.index is not None else torch.cuda.current_device()
    )
    sm_count = _sm_count(device)
    M = validate_kimi_k3_fp8_projection_inputs(
        x, prepared, out, workspace, arch=arch, sm_count=sm_count
    )
    q, sf = workspace
    ldo = int(out.stride(0))
    tma_store = gemm_tma_store_eligible(out.data_ptr(), ldo, prepared.n_valid)
    plan = route_plan(
        prepared,
        M,
        arch,
        sm_count,
        tma_store,
        gemm_reg_staged_eligible(tma_store, _store_vec(out.data_ptr(), ldo)),
    )
    out_flat = torch.as_strided(
        out, (ldo * (M - 1) + prepared.n_valid,), (1,), out.storage_offset()
    )
    a_u8 = q.view(torch.uint8)
    b_u8 = prepared.weight_tiles.view(torch.uint8)
    sfa = sf[: activation_sf_workspace_bytes(M, prepared.K, plan.sf_rows)].view(
        -1, 4, 128
    )
    launches: list[tuple[Callable[..., Any], tuple]] = []
    stage = 0
    with torch.cuda.device(device_index):
        if plan.quant_units is not None:
            k_blocks = prepared.K // BLOCK
            launches.append(
                _bind(
                    plan.modules[stage],
                    dict(
                        x=x,
                        a_q=q,
                        a_sf=sf,
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
        cfg = plan.decode
        if cfg is None:
            # ``OUT`` is the TMA-store descriptor over the [M, n_valid] output view (clipped rows / columns are never
            # written); the register-epilogue program receives a placeholder map it never accesses.
            out_map = (
                out
                if plan.gemm_tma_store
                else torch.zeros(GEMM_TS_COLS, dtype=torch.bfloat16, device=device)
            )
            launches.append(
                _bind(
                    plan.modules[stage],
                    dict(
                        A=a_u8,
                        B=b_u8,
                        SFA=sfa,
                        SFB=prepared.scale_tiles.view(-1, 8, 128),
                        OUT=out_map,
                        x=x,
                        K=prepared.K,
                        out=out_flat,
                        M=M,
                        m_tiles=_m_tiles(M),
                        n_tiles=prepared.n_tiles,
                        n_valid=prepared.n_valid,
                        ldo=ldo,
                        store_vec=_store_vec(out.data_ptr(), ldo),
                        num_k_iters=prepared.num_k_iters,
                        sf_k_tiles=prepared.sf_k_sets,
                        grid=(plan.grids[stage], 1, 1),
                    ),
                )
            )
        else:
            c_off, c_bytes, p_off, p_bytes = reduction_layout(prepared, M, cfg)
            launches.append(
                _bind(
                    plan.modules[stage],
                    dict(
                        W=b_u8,
                        X=a_u8,
                        SFW=prepared.scale_tiles.view(-1, 4, 128),
                        SFX=sfa,
                        out=out_flat,
                        partials=sf[p_off : p_off + p_bytes].view(torch.float32),
                        counters=sf[c_off : c_off + c_bytes].view(torch.uint32),
                        M=M,
                        n_tiles=prepared.n_tiles128,
                        n_valid=prepared.n_valid,
                        ldo=ldo,
                        num_k_iters=prepared.num_k_iters,
                        sf_k_tiles=prepared.sf_k_sets,
                        split=cfg.split,
                        tok_per_cta=cfg.tok_per_cta,
                        total_work=cfg.total_work,
                        store_vec=_decode_store_vec(
                            out.data_ptr(), ldo, prepared.n_valid
                        ),
                        x=x,
                        K=prepared.K,
                        XB=x,
                        grid=(plan.grids[stage], 1, 1),
                    ),
                )
            )
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
