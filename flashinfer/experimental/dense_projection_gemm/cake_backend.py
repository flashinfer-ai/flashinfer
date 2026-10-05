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

Cake backend: the GLM-5.2 dense projection GEMMs for training and the FP32
router-gate GEMM on SM100 / SM103 / SM107 (flashinfer-ai/flashinfer#5677).

K1 -- dense projection GEMM (BF16 operands, FP32 accumulation)
--------------------------------------------------------------
``out[l, m, n] = sum_k A[l, m, k] * B[l, k, n]`` for strided *views*:

* ``A`` BF16 ``[L, M, K]`` (or ``[M, K]``) with unit stride on ``m`` or ``k``;
  ``B`` BF16 ``[L, K, N]`` (or ``[K, N]``) with unit stride on ``k`` or ``n``.
  The other matrix stride and the batch stride are multiples of 8 elements
  (16 bytes); a view starts at a 16-byte aligned address.
* ``out`` BF16 or FP32 ``[L, M, N]`` / ``[M, N]`` with unit stride on ``n``,
  16-byte aligned rows and batches, or -- with ``transposed_out`` --
  ``[L, N, M]`` / ``[N, M]`` with unit stride on ``m`` (element ``(m, n)`` is
  stored at ``out[l, n, m]``, so a swapped small-``N`` weight gradient lands
  directly in ``dW[N, K]``).
* ``M``, ``K`` >= 1 and ``N`` a positive multiple of 8 are runtime values (no
  padding); the batch dimension ``L`` and its strides are arbitrary, so the
  token-major head views of the MLA projections (``[T, H, D]`` permuted to
  ``[H, T, D]``) run without copies.

The training layouts map onto the operator without copies (``T`` tokens):
forward ``X[T, K] @ W[N, K].T`` (``A = X`` K-major, ``B = W.T`` K-major),
input gradient ``G[T, N] @ W[N, K]`` (``B = W`` MN-major), weight gradient
``G[T, N].T @ X[T, K]`` (both MN-major, ``K = T`` ragged); a weight gradient
with fewer than 256 output rows runs the swapped ``X.T @ G`` with the
transposed store (:func:`projection_wgrad`).

The host selects one traced kernel instance per (A layout, B layout, tile
width, tile height, output kind, epilogue path, raster group, TMA L2 eviction
hints) exactly as the Cake production launcher does (:func:`instance_symbol`; the raster group
and the hints follow the device's CTA-pair count and L2 size through
:func:`default_group_m` and the wave working-set gate of :func:`default_hints`),
plans the stream-K split from the device's SM count (:func:`stream_k_plan`
with ``sk="auto"``: a
two-part K-aligned split of every tile only when the whole problem is one
partial wave whose halves fit the CTA pairs and K has at least 16 steps;
multi-wave problems stay data-parallel) and binds the argument plan of the
registered module (``cake_jit.MODULES`` / ``cake_jit.KERNELS``).  The result
is bitwise identical to the Cake launcher's on the same device.

K2 -- FP32 router GEMM through split-BF16x3 emulation
----------------------------------------------------
``out[m, n] = sum_k A[m, k] * B[k, n]`` for FP32 2-D views (``A`` with unit
stride on ``m`` or ``k``, ``B`` with unit stride on ``k`` or ``n``, ``out``
FP32 ``[M, N]`` with unit inner stride and 16-byte rows, ``N`` a multiple of
8).  The streamed operand ``A`` is split into three BF16 parts inside the
kernel; the second operand ``B`` is split on the host into a retained
``[3, outer, inner]`` BF16 stack in its own layout every launch (exact:
``b = b1 + b2 + b3``); the products with ``i + j <= 3`` are accumulated with
chunked FP32 promotion and, for the forward and the weight gradient, split-K
partials are reduced on the host in a fixed order.  Splits, chunk boundaries
and the reduction order are fixed: results are bit-exact run to run.  The
operand layouts select the instance and the production split count
(``kk`` forward 4, ``kn`` input gradient 1, ``nn`` -> the swapped
transposed-store ``nn_t`` weight gradient 11).

Every allocation happens in :func:`prepare_dense_projection_gemm` /
:func:`prepare_router_fp32_gemm`; the returned prepared launch replays with no
CUDA allocation and no host synchronization for new values written into the
bound tensors.  Every ``launch()`` is launch-only: the generated binding encodes
the operand tensor maps by value from the bound views on each call (no descriptor
workspace), so a prepared launch can be captured into a CUDA Graph from its first
call.  The eager entry points (:func:`dense_projection_gemm`,
:func:`projection_forward` / :func:`projection_dgrad` / :func:`projection_wgrad`,
:func:`router_fp32_gemm`, :func:`router_forward` / :func:`router_dgrad` /
:func:`router_wgrad`) prepare and launch in one call.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple, Union

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    TRACKING_ISSUE,
    load_cake_dense_projection_gemm_module,
    select_module,
)

SUPPORTED_COMPUTE_CAPABILITIES = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}

# ---------------------------------------------------------------------------
# K1 tile configuration (mirror of the Cake kernel module's constants; the
# Cake kernel host is ``dense_projection_gemm.py`` of the Cake kernel library,
# cited by line below)
# ---------------------------------------------------------------------------

BLOCK_M = 128  # A rows per MMA instruction per CTA (256 per CTA pair)  [Cake L48]
BLOCK_K = 64  # K elements per stage (128-byte swizzle rows)  [Cake L58]
PANEL_BYTES = (
    BLOCK_K * 128
)  # one MN-major B panel: 64 K rows x 128 B = 8192  [Cake L60]
B_SWZ_CHOICES = (
    128,
    64,
    32,
)  # round 13 (Cake L35): MN-major B panel width in bytes; 128 = the 64-column panels of rounds 1-12
CTA_GROUP = 2  # CTAs per cluster / MMA pair  [Cake L59]
EPI_WARPS = 8  # [Cake L67]
WORK_STAGES = 4  # cluster-launch-control work-ring depth  [Cake L68]
GROUP_M = 16  # CTA row tiles per raster group (even: the two CTAs of a pair are the row halves of one 256-row tile)  [Cake L69]
SLOT_BYTES = 32 * 128  # one epilogue staging slot (TMA-store epilogue)  [Cake L95]
SK_MIN_ITERS = 8  # stream-K: fewest K steps per unit  [Cake L72]
K_TMA_BF16 = 1024  # bf16 rows with K at or below this: TMA-store epilogue  [Cake L103]
K_TWO_SLOTS = 0  # K at or below this: two staging slots per warp (off)  [Cake L104]
CH_T_COLS = 32  # output rows n per transposed TMA-store box / staging chunk (round 14)  [Cake L111]
# Stream-K slice counters: the real counters of the tail tiles use [0, tail_tiles * 16); lanes 1..31
# of a fixup warp fetch-add 0 to private dummy counters at SK_DUMMY_BASE + slice * 32 + lane (slice <
# 16), so the host asserts tail_tiles * 16 <= SK_DUMMY_BASE and allocates at least
# SK_DUMMY_BASE + 16 * 32 = 4608 counters (the Cake host allocates max(8192, tail_tiles * 16)).  [Cake L116, L122]
SK_DUMMY_BASE = 4096
SK_COUNTERS_MIN = (
    8192  # ``_sk_counters``: zeros of max(8192, needed) u32  [Cake L1395-L1402]
)
SMEM_OPT_IN = 232448  # 227 KiB dynamic shared memory opt-in per CTA (every architecture)  [Cake L117]
# CUDA 13.4 oversized shared-memory mode (launch attribute ALLOW_OVERSIZED; the Cake runtime's host shim enables
# it when a kernel exceeds the standard opt-in): per-architecture ceiling (R200: 334848 B) minus 512 B of headroom.
# Round 9: deeper pipelines on sm_107a.  [Cake L121]
SMEM_OVERSIZED = {"sm_107a": 334848 - 512}
L2_PROMOS = (
    "none",
    "l2_64b",
    "l2_128b",
    "l2_256b",
)  # TMA descriptor L2 promotion  [Cake L772]
L2_HINTS = (
    "none",
    "evict_normal",
    "evict_first",
    "evict_last",
)  # TMA load L2 eviction policy  [Cake L773]
EPI_MODES = ("reg", "tma")  # [Cake L785]
# Round 9 adds BLOCK_N = 160 / 224 (wave-quantization fits: BLOCK_N / 2 must be a multiple of 16 so every warp
# slice is whole 16-column store chunks) and the 64-row Layout-B tile family (tcgen05 M=128 cta_group::2 pair
# tiles, four TMEM buffers; symbol suffix ``_m64``).  [Cake L775-L776]
BLOCK_N_CHOICES = (128, 160, 192, 224, 256)
CTA_ROWS_CHOICES = (64, 128, 256)

# ---------------------------------------------------------------------------
# K2 tile configuration (mirror of the Cake router kernel module's constants)
# ---------------------------------------------------------------------------

ROUTER_BLOCK_M = 128
ROUTER_BLOCK_K = 32  # fp32 elements split per stage
ROUTER_BLOCK_N = 128  # output columns per CTA pair
# Production split-K counts per instance layout (router_forward / router_dgrad / router_wgrad).
ROUTER_SPLITS = {"kk": 4, "kn": 1, "nn_t": 11}


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))


def require_arch(device: torch.device) -> str:
    arch = arch_for(device)
    if arch is None:
        raise ValueError(
            "the Cake dense projection GEMMs require compute capability 10.0, 10.3 or 10.7"
        )
    return arch


def generated_program_available(
    device: Optional[torch.device] = None, template: Optional[str] = None
) -> bool:
    """True when this checkout registers a program for ``device`` (serving ``template`` when given)."""
    arch = arch_for(device)
    if arch is None:
        return False
    table = KERNELS.get(arch, {})
    if template is None:
        return bool(table)
    return template in table


def device_sm_count(device: torch.device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def sm_pairs(sm_count: int) -> int:
    """CTA pairs the persistent grid can hold at once (one pair per two SMs).  [Cake ``_sm_pairs`` L1365-L1371]"""
    return max(1, int(sm_count) // CTA_GROUP)


_L2_BYTES: dict[str, int] = {}


def device_l2_bytes(device: torch.device) -> int:
    """L2 size of ``device`` from the driver (no per-SKU table): resolves the hint working-set
    gate exactly as the Cake host's ``_l2_bytes`` does (``torch.cuda.get_device_properties``
    ``L2_cache_size``, falling back to ``cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)``).
    [Cake ``_l2_bytes`` L1377-L1392]"""
    key = str(device)
    if key not in _L2_BYTES:
        l2 = getattr(torch.cuda.get_device_properties(device), "L2_cache_size", None)
        if not l2:
            from cuda.bindings import driver as cuda

            dev = (
                torch.device(device).index
                if torch.device(device).index is not None
                else torch.cuda.current_device()
            )
            err, l2 = cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev
            )
            if err != cuda.CUresult.CUDA_SUCCESS or not l2:
                raise RuntimeError(
                    f"dense_projection_gemm: cannot resolve the L2 size of {device} ({err})"
                )
        _L2_BYTES[key] = int(l2)
    return _L2_BYTES[key]


# ---------------------------------------------------------------------------
# K1 instance selection (pure functions; mirror of the Cake launcher's)
# ---------------------------------------------------------------------------


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def epi_cols(block_n: int, cta_rows: int = 128) -> int:
    """Accumulator columns per epilogue warp slice: BLOCK_N / 2 (two warps per lane quadrant split
    the columns), or BLOCK_N / 4 for the 64-row Layout-B family (a quadrant holds one column half of
    32 rows; its two warps split that half).  [Cake ``epi_cols`` L779-L782]"""
    return block_n // 4 if cta_rows == 64 else block_n // 2


def epi_mode(
    out_f32: bool,
    out_t: bool,
    K: Optional[int] = None,
    epi: Optional[str] = None,
    block_n: int = 256,
    cta_rows: int = 128,
) -> str:
    """Epilogue of an instance: transposed output -> scalar register stores (``"reg"``);
    row-major fp32 -> per-warp TMA stores (``"tma"``) when the warp slice is whole 32-column
    chunks (the float4 register-store path is an opt-in ``epi="reg"``, measured 1-10 % slower
    on every fp32 row; a BLOCK_N = 160 / 224 or 64-row BLOCK_N = 192 slice has no TMA-store
    path); row-major bf16 -> TMA stores for short K (epilogue-bound tiles) when the slice is
    whole 64-column chunks, 32-byte register stores otherwise.  [Cake ``epi_mode`` L788-L805]"""
    if epi is not None:
        if epi not in EPI_MODES:
            raise ValueError(f"epi must be one of {EPI_MODES}, got {epi!r}")
        if out_t and epi != "reg" and epi_cols(block_n, cta_rows) % CH_T_COLS:
            # round 14: the transposed TMA-store epilogue stages CH_T_COLS-column chunks of the warp slice
            raise ValueError(
                "the transposed TMA-store epilogue needs warp slices of whole "
                f"{CH_T_COLS}-column chunks; BLOCK_N={block_n} CTA_ROWS={cta_rows} gives "
                f"{epi_cols(block_n, cta_rows)}"
            )
        return epi
    if out_t:
        return "reg"  # the transposed TMA store (round 14) is opt-in per row (ROW_RULES epi='tma')
    if out_f32:
        return "tma" if epi_cols(block_n, cta_rows) % 32 == 0 else "reg"
    # bf16 chunks are 64 columns: a 96-column (BLOCK_N = 192) or 32-column (Layout B, BLOCK_N = 128) warp slice has
    # no whole-chunk TMA-store path
    return (
        "tma"
        if (K is not None and K <= K_TMA_BF16 and epi_cols(block_n, cta_rows) % 64 == 0)
        else "reg"
    )


def epi_slots(
    epi: str,
    out_f32: bool,
    block_n: int,
    K: Optional[int] = None,
    slots: Optional[int] = None,
    cta_rows: int = 128,
    out_t: bool = False,
) -> int:
    """Staging slots per epilogue warp (0 without the TMA-store epilogue).  The chunk count
    per tile must be a multiple of the slot count so the round-robin rotation restarts at
    slot 0 on every tile.  Row-major chunks are 128-byte column groups (64 bf16 / 32 fp32);
    the transposed box (round 14, ``out_t``) stages ``CH_T_COLS`` columns per chunk.
    [Cake ``epi_slots`` L1151-L1164]"""
    if epi != "tma":
        return 0
    chunks = epi_cols(block_n, cta_rows) // (
        CH_T_COLS if out_t else (32 if out_f32 else 64)
    )
    if slots is None:
        slots = 2 if (K is None or K <= K_TWO_SLOTS) else 1
        if chunks % slots:
            slots = 1
    slots = int(slots)
    if slots not in (1, 2) or chunks % slots:
        raise ValueError(
            f"slots must be 1 or 2 and divide the {chunks} chunks per tile, got {slots}"
        )
    return slots


def staging_bytes(slots: int) -> int:
    # [Cake ``staging_bytes`` L824-L825]
    return EPI_WARPS * slots * SLOT_BYTES


def b_stage_bytes(b_mn: bool, block_n: int, b_swz: int = 128) -> int:
    """Bytes of one B stage per CTA: K-major = ``BLOCK_N / 2`` 128-byte rows; MN-major = whole panels of
    ``b_swz / 2`` columns (BLOCK_N = 192 with 128-byte panels loads two 64-column panels per stage and the MMA
    reads 1.5 of them; with 64-byte panels it loads exactly the three 32-column panels it reads - round 13).
    [Cake ``b_stage_bytes``]"""
    n_half = block_n // 2
    if not b_mn:
        return n_half * BLOCK_K * 2
    cols = int(b_swz) // 2
    return (-(-n_half // cols)) * BLOCK_K * int(b_swz)


def a_halves_of(cta_rows: int) -> int:
    """128-row A halves per CTA tile (1 for the 64 / 128-row families, 2 for the tall family).  [Cake ``A_HALVES_OF``]"""
    return max(1, int(cta_rows) // BLOCK_M)


def smem_limit_for(arch: Optional[str]) -> int:
    """Dynamic SMEM an instance may request on ``arch``: the 227 KiB opt-in, or the architecture's
    oversized ceiling (``SMEM_OVERSIZED``).  ``None`` = the largest limit of any architecture
    (instance keys rebuilt from an exported plan, which was validated with its own architecture's
    limit).  [Cake ``smem_limit_for`` L835-L841]"""
    if arch is None:
        return max(SMEM_OPT_IN, *SMEM_OVERSIZED.values())
    return SMEM_OVERSIZED.get(str(arch), SMEM_OPT_IN)


def default_stages(
    slots: int,
    cta_rows: int = 128,
    block_n: int = 256,
    b_mn: bool = False,
    b_swz: int = 128,
) -> int:
    """Mainloop stages that fit the 227 KiB opt-in with the epilogue staging: 32 KiB stages
    for 128-row tiles (24 KiB at BLOCK_N = 128, where the streaming-bound small-N rows are
    still latency-bound at 7 stages: 9 / 8 / 6 for 0 / 1 / 2 slots), 48 KiB stages for tall
    (256-row) tiles.  Layout-B tiles (64 rows: 24 KiB stages at BLOCK_N = 256, 16 KiB at 128)
    take the deepest pipeline that fits beside the staging, at most 12 stages (round 9;
    ``b_mn`` sizes the MN-major B panels).  Narrow-panel instances (``b_swz`` 64 / 32, round 13) have smaller
    stages than their 128-byte-panel siblings and likewise take the deepest pipeline that fits the opt-in (at most
    12).  [Cake ``default_stages``]"""
    if cta_rows == 64 or int(b_swz) != 128:
        stage = min(cta_rows, 128) * a_halves_of(
            cta_rows
        ) * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
        return max(
            2,
            min(12, (SMEM_OPT_IN - WORK_STAGES * 16 - staging_bytes(slots)) // stage),
        )
    if cta_rows != 128:
        return (4, 4, 3)[slots]
    if block_n == 128:
        return (9, 8, 6)[slots]
    return (7, 6, 5)[slots]


def box_rows_of(
    a_mn: bool,
    b_mn: bool,
    block_n: int,
    box_rows: Optional[int] = None,
    cta_rows: int = 128,
) -> tuple[int, int]:
    """TMA box heights of the A and B operands: full height unless capped by ``box_rows``.
    [Cake ``box_rows_of`` L860-L870]"""
    a_full = BLOCK_K if a_mn else cta_rows
    b_full = BLOCK_K if b_mn else block_n // 2
    if box_rows is None:
        return a_full, b_full
    box_rows = int(box_rows)
    a_box, b_box = min(a_full, box_rows), min(b_full, box_rows)
    if box_rows < 8 or a_full % a_box or b_full % b_box:
        raise ValueError(
            f"box_rows={box_rows} must divide the operand heights {a_full} / {b_full}"
        )
    return a_box, b_box


def instance_key(
    *,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool = False,
    out_t: bool = False,
    block_n: int = 256,
    stages: Optional[int] = None,
    diag: tuple = (),
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    box_rows: Optional[int] = None,
    cta_rows: int = 128,
    pf: int = 0,
    hints: tuple = ("none", "none"),
    f32_v8: bool = False,
    quad_store: bool = False,
    park: bool = False,
    ovl: bool = False,
    htail: bool = False,
    smem_limit: Optional[int] = None,
    b_swz: int = 128,
    sk_exact: bool = False,
    batch_group: int = 0,
    store_ef: bool = False,
) -> tuple:
    """The instance tuple the Cake kernel module traces one program per (validation included):
    ``(a_mn, b_mn, out_f32, out_t, block_n, stages, diag, epi, slots, box_rows, cta_rows, pf,
    hints, f32_v8, quad_store, park, ovl, htail, b_swz, sk_exact)`` (round 13: ``park`` = parked tall epilogue, ``ovl`` =
    overlapped single-TMEM-buffer tall epilogue, ``htail`` = deterministic half-height tail wave, ``b_swz`` = MN-major B
    panel width in bytes (128 / 64 / 32; K-major B instances always keep 128), ``sk_exact`` = exact p-way stream-K
    split, ``_skx`` symbols); the raster group width (``group_m``) and the TMA L2 promotion (``promo_code``) are launch parameters since round 11.  ``smem_limit`` (bytes; default = the largest
    architecture limit, ``smem_limit_for(None)``) only bounds the stage count - it is not part of
    the key, so an instance has one symbol on every architecture (the planner passes
    ``smem_limit_for(arch)`` like the Cake launcher).  Diagnostic (attribution) instances are not
    exported.  [Cake ``instance_key`` L873-L905]  ``batch_group`` (round 13, L38b) = the batch-entry raster of the
    batched MLA rows: 0 = off, 1 = every entry (``_bf``), G > 1 = G adjacent entries interleaved across the CTA pairs
    (``_bg{G}``); the planner keeps 0 for a batch of one.  ``store_ef`` (round 13, Cake W4) = st.global.L1::no_allocate.L2::evict_first on the fp32 v8
    register stores (``_ef``); only that store form carries the hint, so the field is False for every other epilogue."""
    a_mn, b_mn, out_f32, out_t = bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t)
    block_n, cta_rows, pf = int(block_n), int(cta_rows), int(pf)
    hints = (str(hints[0]), str(hints[1]))
    if any(h not in L2_HINTS for h in hints):
        raise ValueError(f"hints must be in {L2_HINTS}, got {hints!r}")
    if pf < 0 or pf > 16:
        raise ValueError(f"prefetch distance must be in 0..16 K steps, got {pf}")
    if cta_rows not in CTA_ROWS_CHOICES:
        raise ValueError(f"cta_rows must be one of {CTA_ROWS_CHOICES}, got {cta_rows}")
    box_rows = 0 if box_rows is None else int(box_rows)
    box_rows_of(a_mn, b_mn, block_n, box_rows or None, cta_rows)
    if block_n not in BLOCK_N_CHOICES:
        raise ValueError(f"BLOCK_N must be one of {BLOCK_N_CHOICES}, got {block_n}")
    cols = epi_cols(block_n, cta_rows)
    if cols % 16:
        raise ValueError(
            f"BLOCK_N={block_n} with CTA_ROWS={cta_rows} gives {cols}-column warp slices; "
            "slices must be whole 16-column chunks"
        )
    b_swz = int(b_swz) if b_mn else 128
    if b_swz not in B_SWZ_CHOICES:
        raise ValueError(f"b_swz must be one of {B_SWZ_CHOICES}, got {b_swz}")
    batch_group = int(batch_group)
    if batch_group < 0:
        raise ValueError(
            f"batch_group must be 0 (off), 1 (every batch entry) or a group size G > 1, got {batch_group}"
        )
    epi = epi_mode(out_f32, out_t, None, epi, block_n, cta_rows)
    if epi == "tma" and cols % (32 if out_f32 else 64):
        raise ValueError(
            f"the TMA-store epilogue needs whole 128-byte column chunks per warp; BLOCK_N={block_n} "
            f"CTA_ROWS={cta_rows} {'fp32' if out_f32 else 'bf16'} output needs epi='reg'"
        )
    slots = epi_slots(epi, out_f32, block_n, None, slots, cta_rows, out_t)
    stages = (
        default_stages(slots, cta_rows, block_n, b_mn, b_swz)
        if stages is None
        else int(stages)
    )
    diag = tuple(sorted(set(diag)))
    if diag:
        raise ValueError(f"diagnostic instances are not exported: {diag}")
    limit = smem_limit_for(None) if smem_limit is None else int(smem_limit)
    stage_bytes = cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
    if (
        stages < 2
        or stages * stage_bytes + staging_bytes(slots) + WORK_STAGES * 16 > limit
    ):
        raise ValueError(
            f"{stages} stages at BLOCK_N={block_n}, CTA_ROWS={cta_rows} exceed the {limit} B dynamic SMEM limit"
        )
    f32_v8 = (
        bool(f32_v8) and out_f32 and epi == "reg" and not out_t
    )  # only the row-major fp32 register epilogue has the knob
    quad_store = (
        bool(quad_store)
        and (not out_f32)
        and epi == "reg"
        and not out_t
        # bf16 row-major register epilogue with whole 64-column groups per warp slice: quad-transposed
        # 32-byte row segments (round 7)
        and cols % 64 == 0
    )
    # parked tall epilogue: bf16 row-major 256-row tiles only (Cake round 10 knob, measured per row)
    park = bool(park) and cta_rows == 256 and (not out_f32) and (not out_t)
    ovl = bool(ovl)
    if ovl and (cta_rows != 256 or block_n != 256 or park or pf or box_rows):
        # round 13: the overlapped epilogue's chunks are the two 128-column passes of the tall 256-column tile; it
        # owns the B load coordinates (no prefetch / box cap) and excludes the parked epilogue  [Cake instance_key]
        raise ValueError(
            f"ovl needs cta_rows=256, block_n=256 and no park / pf / box_rows (got cta_rows={cta_rows}, "
            f"block_n={block_n}, park={park}, pf={pf}, box_rows={box_rows})"
        )
    if ovl and b_swz != 128:
        raise ValueError(
            f"ovl owns the 256-column B load coordinates and needs the 128-byte B swizzle (got b_swz={b_swz})"
        )
    htail = bool(htail)
    if htail and (cta_rows != 256 or pf or box_rows):
        raise ValueError(
            f"htail (half-height tail wave) needs cta_rows=256 and no pf / box_rows (got cta_rows={cta_rows}, "
            f"pf={pf}, box_rows={box_rows})"
        )
    if htail and sk_exact:
        raise ValueError(
            "htail (half-height tail wave) and sk_exact (exact p-way tail split) are two tail policies - choose one per instance"
        )
    # L1::no_allocate.L2::evict_first on the fp32 v8 stores: only that store form carries the hint  [Cake instance_key]
    store_ef = bool(store_ef) and f32_v8
    return (
        a_mn,
        b_mn,
        out_f32,
        out_t,
        block_n,
        stages,
        diag,
        epi,
        slots,
        box_rows,
        cta_rows,
        pf,
        hints,
        f32_v8,
        quad_store,
        park,
        ovl,
        htail,
        b_swz,
        bool(sk_exact),
        batch_group,
        store_ef,
    )


def instance_symbol(key: tuple) -> str:
    """Kernel symbol / registry template of an instance key (``dense_proj_gemm_<a><b>_n<N>``
    followed by ``_m256`` for tall tiles / ``_m64`` for the 64-row Layout-B family, ``_pf<n>`` for a
    prefetch distance, ``_h<a><b>`` for non-default (A, B) eviction hints (first letters, e.g.
    ``_hen`` = A evict_first / B none), ``_f32``, ``_v8`` for the 256-bit fp32 register stores, ``_t``,
    ``_bz<bytes>`` right after the tile family for a narrow MN-major B panel (round 13), ``_pk`` / ``_ov`` / ``_ht`` for
    the parked / overlapped / half-height-tail tall epilogues (round 13), ``_<epi><slots>`` for the TMA-store
    epilogue, ``_s<stages>`` for a non-default stage count, ``_box<rows>`` and a trailing ``_skx`` for the exact
    p-way stream-K split (round 13)).  [Cake ``instance_symbol``]"""
    (
        a_mn,
        b_mn,
        out_f32,
        out_t,
        block_n,
        stages,
        diag,
        epi,
        slots,
        box_rows,
        cta_rows,
        pf,
        hints,
        f32_v8,
        quad_store,
        park,
        ovl,
        htail,
        b_swz,
        sk_exact,
        batch_group,
        store_ef,
    ) = key
    return (
        "dense_proj_gemm_"
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + f"_n{block_n}"
        + ("_m256" if cta_rows == 256 else "_m64" if cta_rows == 64 else "")
        + (f"_bz{b_swz}" if b_swz != 128 else "")
        + (f"_pf{pf}" if pf else "")
        + (f"_h{hints[0][0]}{hints[1][0]}" if hints != ("none", "none") else "")
        + ("_f32" if out_f32 else "")
        + ("_v8" if f32_v8 else "")
        + ("_ef" if store_ef else "")
        + ("_q" if quad_store else "")
        + ("_pk" if park else "")
        + ("_ov" if ovl else "")
        + ("_ht" if htail else "")
        + ("_t" if out_t else "")
        + (f"_{epi}{slots}" if epi != "reg" else "")
        + (
            f"_s{stages}"
            if stages != default_stages(slots, cta_rows, block_n, b_mn, b_swz)
            else ""
        )
        + (f"_box{box_rows}" if box_rows else "")
        + ("_skx" if sk_exact else "")
        + (
            "_bf"
            if batch_group == 1
            else f"_bg{batch_group}"
            if batch_group > 1
            else ""
        )
        + "".join(f"_{d}" for d in diag)
    )


def swap_small_m(L: int, M: int, N: int, transposed_out: bool) -> bool:
    """Batched rows with M <= 256 and N >= 2 M (the MLA weight gradients: heads = batch, M = head
    dim, N = latent dim) are computed as the transposed GEMM ``out^T[l] = B[l]^T A[l]^T`` with the
    transposed store.  [Cake ``swap_small_m`` L1164-L1167]"""
    return L > 1 and not transposed_out and M <= 256 and N >= 2 * M


# fmt: off
# Per-architecture, per-row knob overrides measured in Cake round 2, keyed by the static row identity
# (arch, A MN-major, B MN-major, fp32 output, transposed output, batched, N, K, M); the ragged token
# count never keys a rule (M = None for forward / input gradients, K = None for weight gradients).
# Byte-identical to Cake ``ROW_RULES``.
ROW_RULES: dict[tuple, dict] = {
    ('sm_100a', False, False, False, False, False, 576, 6144, None): {"hints": ('evict_first', 'evict_last'), "stages": 8},
    ('sm_100a', False, False, False, False, False, 2048, 6144, None): {"group_m": 8},
    ('sm_100a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 16384, 2048, None): {"group_m": 32},
    ('sm_100a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_100a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_100a', False, True, False, False, False, 2048, 16384, None): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True},
    ('sm_100a', False, True, False, False, False, 6144, 32, None): {"slots": 2},
    ('sm_100a', False, True, False, False, False, 6144, 128, None): {"slots": 2},  # round 14 (W2f): indexer_k bf16 input gradient on the 128-row two-slot TMA-store family
    ('sm_100a', False, True, False, False, False, 6144, 576, None): {"group_m": 8, "epi": 'reg', "stages": 6, "quad_store": True},
    ('sm_100a', False, True, False, False, False, 6144, 2048, None): {"group_m": 8},
    ('sm_100a', False, True, False, False, False, 6144, 12288, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_100a', False, True, False, False, True, 512, 256, None): {"cta_rows": 256, "group_m": 4, "batch_group": 4},
    ('sm_100a', False, True, True, False, False, 2048, 4096, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 32, None): {"block_n": 128, "slots": 2},
    ('sm_100a', False, True, True, False, False, 6144, 128, None): {"block_n": 128},
    ('sm_100a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b'},
    ('sm_100a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 12288, 6144, None): {"epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 16384, 6144, None): {"epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, False, False, False, 2048, None, 4096): {"cta_rows": 256},
    ('sm_100a', True, True, False, False, False, 2048, None, 6144): {"group_m": 4},
    ('sm_100a', True, True, False, False, False, 6144, None, 2048): {"block_n": 192, "sk_parts": 2},
    ('sm_100a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "group_m": 8},
    ('sm_100a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "group_m": 8},
    ('sm_100a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "group_m": 8, "htail": True},
    ('sm_100a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first')},
    ('sm_100a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, True, False, False, 2048, None, 4096): {"cta_rows": 256},
    ('sm_100a', True, True, True, False, False, 2048, None, 6144): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, True, False, False, 6144, None, 2048): {"group_m": 32, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256},
    ('sm_100a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256},
    ('sm_100a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256, "htail": True},
    ('sm_100a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first')},
    ('sm_107a', False, False, False, False, False, 32, 6144, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, False, 576, 6144, None): {"cta_rows": 256, "hints": ('evict_first', 'none'), "stages": 5},
    ('sm_107a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_107a', False, False, False, False, False, 16384, 2048, None): {"group_m": 32},
    ('sm_107a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 512, 256, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, False, 6144, 32, None): {"cta_rows": 128, "slots": 2},
    ('sm_107a', False, True, False, False, False, 6144, 128, None): {"block_n": 128, "cta_rows": 256},
    ('sm_107a', False, True, False, False, False, 6144, 576, None): {"group_m": 8, "epi": 'reg', "quad_store": True},
    ('sm_107a', False, True, False, False, False, 6144, 12288, None): {"sk_parts": 2},
    ('sm_107a', False, True, False, False, False, 12288, 6144, None): {"hints": ('none', 'evict_first')},
    ('sm_107a', False, True, False, False, False, 16384, 6144, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_107a', False, True, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 256, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 512, 192, None): {"batch_group": 8},
    ('sm_107a', False, True, False, False, True, 512, 256, None): {"promo": 'l2_256b', "batch_group": 8},
    ('sm_107a', False, True, True, False, False, 2048, 4096, None): {"epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 2048, 16384, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 6144, 32, None): {"block_n": 128},
    ('sm_107a', False, True, True, False, False, 6144, 128, None): {"block_n": 128},
    ('sm_107a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b'},
    ('sm_107a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', False, True, True, False, False, 12288, 6144, None): {"cta_rows": 256},
    ('sm_107a', False, True, True, False, False, 16384, 6144, None): {"cta_rows": 256},
    ('sm_107a', True, True, False, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "group_m": 4, "sk_parts": 2, "b_swz": 64},
    ('sm_107a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "sk_parts": 2},
    ('sm_107a', True, True, False, False, False, 12288, None, 6144): {"sk_parts": 3},
    ('sm_107a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "sk_exact": 3},
    ('sm_107a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2},
    ('sm_107a', True, True, False, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64},
    ('sm_107a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma', "b_swz": 64},
    ('sm_107a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma'},
    ('sm_107a', True, True, True, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "sk_parts": 3, "b_swz": 64},
    ('sm_107a', True, True, True, False, False, 2048, None, 6144): {"cta_rows": 256},
    ('sm_107a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', True, True, True, False, False, 6144, None, 2048): {"cta_rows": 256},
    ('sm_107a', True, True, True, False, False, 6144, None, 12288): {"epi": 'reg', "f32_v8": True},
    ('sm_107a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256},
    ('sm_107a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "sk_exact": 3},
    ('sm_107a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2},
    ('sm_107a', True, True, True, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64},
}
# fmt: on


def row_rule(
    arch: str,
    a_mn: bool,
    b_mn: bool,
    out_f32: bool,
    out_t: bool,
    batched: bool,
    N: int,
    K: int,
    M: int,
) -> dict:
    """The measured knob overrides of one row identity: the exact (N, K, M) rule, else the
    ragged-M (N, K, None) rule, else the ragged-K (N, None, M) rule, else empty.  [Cake ``row_rule`` L1153-L1161]"""
    ident = (arch, bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t), bool(batched))
    for tail in (
        (int(N), int(K), int(M)),
        (int(N), int(K), None),
        (int(N), None, int(M)),
    ):
        rule = ROW_RULES.get(ident + tail)
        if rule is not None:
            return dict(rule)
    return {}


def default_block_n(N: int, b_mn: bool) -> int:
    """256 columns per CTA pair unless the whole output fits a 128-column tile; 192 when N is a
    multiple of 192 but not of 256 (N = 576: three exact 192-column tiles instead of
    256 + 256 + 64; N = 192: one exact tile).  [Cake ``default_block_n`` L1170-L1176]"""
    if N <= 128:
        return 128
    return 192 if (N % 192 == 0 and N % 256 != 0) else 256


def default_pf(M: int, N: int, K: int) -> int:
    """Rolling L2 prefetch distance in K steps (0 = off; selected per row class by measurement).
    [Cake ``default_pf`` L1179-L1181]"""
    return 0


def default_promo(m_tiles: int, n_tiles: int) -> str:
    """TMA descriptor L2 promotion of the operands (``"none"``; selected by measurement).
    [Cake ``default_promo`` L1184-L1186]"""
    return "none"


def wave_working_set(
    m_tiles: int, n_tiles: int, k_len: int, group_m: int, pairs: int, elt_bytes: int = 2
) -> int:
    """Bytes of A and B panels touched by one wave of CTA pairs: with ``group_m`` row tiles per
    raster group a wave covers ``rows`` pair-row panels (256 x K) and ``cols`` column panels
    (K x 256) of one batch entry.  [Cake ``wave_working_set`` L1189-L1201]"""
    m_pairs = max(1, m_tiles // CTA_GROUP)
    g = max(1, group_m // CTA_GROUP)
    band = g * n_tiles  # pair tiles per raster group
    if band >= pairs:
        rows = min(m_pairs, g)
        cols = min(n_tiles, _ceil_div(pairs, rows))
    else:
        rows = min(m_pairs, g * _ceil_div(pairs, band))
        cols = n_tiles
    return (rows + cols) * 256 * k_len * elt_bytes


def default_hints(
    a_mn: bool,
    b_mn: bool,
    m_tiles: int,
    n_tiles: int,
    k_len: int,
    group_m: int,
    pairs: int,
    l2_bytes: int,
) -> tuple[str, str]:
    """TMA load L2 eviction policy of (A, B).  No hint while one wave's operand panels fit the
    L2 (a hint on a resident operand costs 3..23 percent).  Otherwise only a panel used by
    exactly one tile (A when there is one column tile) streams as ``evict_first``.  Reuse-ratio
    hints on the weight-gradient class measured 0..-4 percent in the production stream-K
    configuration and are not applied; ``a_mn`` / ``b_mn`` stay in the signature so a
    layout-class rule can be re-added without touching the call site.
    [Cake ``default_hints`` L1204-L1216]"""
    if wave_working_set(m_tiles, n_tiles, k_len, group_m, pairs) <= l2_bytes:
        return ("none", "none")
    if n_tiles == 1:
        return ("evict_first", "none")
    return ("none", "none")


def default_group_m(
    a_mn: bool, b_mn: bool, m_tiles: int, pair_tiles: int, pairs: int
) -> int:
    """CTA row tiles per raster group (even): 16 on every row (16 is fastest or within 1 percent
    on every projection row; small groups cost 3..30 percent on the many-wave rows).  A 4-row
    group for one-to-two-wave weight-gradient rows measured 0..+2 percent in the production
    configuration and is not applied.  The knob stays per instance.
    [Cake ``default_group_m`` L1219-L1224]"""
    return 16


def default_cta_rows(M: int, N: int, K: int) -> int:
    """Output rows per CTA: 128 (double-buffered TMEM tiles) by default; 256 (two MMAs per K step
    sharing the B stage) and 64 (round 9: 128 x BLOCK_N pair tiles through the M=128 cta_group::2
    MMA, four TMEM buffers, 24 KiB stages) are selected per row by measurement.
    [Cake ``default_cta_rows`` L1227-L1231]"""
    return 128


def sk_parts_plan(
    pair_tiles: int, k_blocks: int, pairs: int, parts: int
) -> Tuple[Union[bool, str], Optional[int]]:
    """``(sk, sk_max_units)`` for a measured ``sk_parts`` row rule: the tail tiles (the whole
    problem when it is one partial wave) are split ``parts`` ways when ``parts * tail`` units
    fit the CTA pairs and every unit keeps at least ``SK_MIN_ITERS`` K steps; otherwise the row
    keeps the ``auto`` policy.  [Cake ``sk_parts_plan`` L1405-L1412]"""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if (
        parts < 2
        or tail == 0
        or parts * tail > pairs
        or k_blocks < parts * SK_MIN_ITERS
    ):
        return "auto", None
    return True, parts * tail


def sk_exact_plan(
    pair_tiles: int, k_blocks: int, pairs: int, parts: int
) -> Optional[tuple[int, int, int, int]]:
    """(num_full, tail_tiles, sk_units, iters_per_unit) of the exact ``parts``-way K split of the tail tiles
    (Cake round 13, L36): the tiles that do not fill a whole wave of CTA pairs (all of them when there are fewer
    tiles than pairs) each become ``parts`` units of ``ceil(k_blocks / parts)`` contiguous K steps - identical K
    ranges on every tile, exactly ``parts`` partials per tile, no drift when ``parts`` does not divide
    ``k_blocks``.  The unit count may exceed the pairs.  ``None`` when the split is not admissible: fewer than two
    parts, no tail, a part under ``SK_MIN_ITERS`` steps, or an empty last part.  [Cake ``sk_exact_plan``]"""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    parts = int(parts)
    if parts < 2 or tail == 0:
        return None
    ipu = _ceil_div(k_blocks, parts)
    if ipu < SK_MIN_ITERS or (parts - 1) * ipu >= k_blocks:
        return None
    return pair_tiles - tail, tail, tail * parts, ipu


def stream_k_plan(
    pair_tiles: int,
    k_blocks: int,
    pairs: int,
    sk: bool | str = True,
    max_units: Optional[int] = None,
) -> tuple[int, int, int, int]:
    """``(num_full, tail_tiles, sk_units, iters_per_unit)``: the tiles that do not fill a
    whole wave of CTA pairs (all of them when there are fewer tiles than pairs) are the tail.

    * ``sk=False``: every work item is a whole tile.
    * ``sk="auto"`` (the launcher default): a K-aligned two-way split of a single partial
      wave -- unit ``u`` is tile ``u // 2``, K half ``u % 2`` -- so tiles sharing an A / B
      panel stay K-synchronised and every tile has exactly two parts (one slab read in the
      fixup).  Only when the whole problem is one partial wave whose halves fit the pairs
      (``pair_tiles <= pairs`` and ``2 * tail <= pairs``) and K has at least
      ``2 * SK_MIN_ITERS`` steps; multi-wave problems keep their tail data-parallel.
    * ``sk=True``: the tail tiles are linearised over their K steps and shared by up to
      ``pairs`` (``max_units``) stream-K units of at least ``SK_MIN_ITERS`` contiguous K
      steps; no stream-K when that cannot create more units than tail tiles.
    * ``sk="tiles"``: diagnostic -- the tail tiles go through the unit path as whole tiles.

    [Cake ``stream_k_plan`` L1415-L1444]"""
    if not sk or pair_tiles == 0:
        return pair_tiles, 0, 0, k_blocks
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if tail == 0:
        return pair_tiles, 0, 0, k_blocks
    if (
        sk == "tiles"
    ):  # diagnostic: the tail tiles go through the unit path as whole tiles (no partials, no fixup)
        return pair_tiles - tail, tail, tail, k_blocks
    if sk == "auto":
        if pair_tiles > pairs or 2 * tail > pairs or k_blocks < 2 * SK_MIN_ITERS:
            return pair_tiles, 0, 0, k_blocks
        half = _ceil_div(k_blocks, 2)
        return pair_tiles - tail, tail, 2 * tail, half
    total = tail * k_blocks
    units = min(pairs, total // SK_MIN_ITERS, max_units or pairs)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    iters_per_unit = _ceil_div(total, units)
    units = _ceil_div(total, iters_per_unit)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    return pair_tiles - tail, tail, units, iters_per_unit


# ---------------------------------------------------------------------------
# K1 view validation and planning
# ---------------------------------------------------------------------------


def as_batched(t: torch.Tensor, name: str) -> torch.Tensor:
    # [Cake ``_as_batched`` L1019-L1024]
    if t.dim() == 2:
        return t.unsqueeze(0)
    if t.dim() == 3:
        return t
    raise ValueError(f"{name} must be a 2-D or 3-D view, got {t.dim()}-D")


def operand_view(
    t: torch.Tensor, name: str, *, k_axis: int
) -> tuple[bool, torch.Tensor]:
    """Classify a ``[L, rows, cols]`` matrix view as K-major (contraction axis contiguous) or
    MN-major and return ``(mn_major, desc)`` with ``desc`` the ``[L, outer, inner]`` view
    (inner stride 1) the TMA descriptor spans.  [Cake ``_operand_view`` L1027-L1043]"""
    mn_axis = 3 - k_axis  # the other matrix axis (1 or 2)
    if t.stride(k_axis) == 1:
        mn_major = False
        desc = t if k_axis == 2 else t.transpose(1, 2)
    elif t.stride(mn_axis) == 1:
        mn_major = True
        desc = t if mn_axis == 2 else t.transpose(1, 2)
    else:
        raise ValueError(
            f"{name} needs unit stride on its contraction or its M/N axis; strides {tuple(t.stride())}"
        )
    if (desc.shape[1] > 1 and desc.stride(1) % 8 != 0) or (
        desc.shape[0] > 1 and desc.stride(0) % 8 != 0
    ):
        raise ValueError(
            f"{name}: row and batch strides must be multiples of 8 elements (16 bytes), got {tuple(desc.stride())}"
        )
    if t.data_ptr() % 16 != 0:
        raise ValueError(f"{name}: the view must start at a 16-byte aligned address")
    return mn_major, desc


@dataclass(frozen=True)
class GemmPlan:
    """The host plan of one dense projection GEMM launch (device-independent except ``sm_pairs``)."""

    L: int
    M: int
    N: int
    K: int
    a_mn: bool
    b_mn: bool
    out_f32: bool
    transposed_out: bool
    block_n: int
    cta_rows: int
    stages: int
    epi: str
    slots: int
    pf: int
    promo: str
    hints: tuple[str, str]
    group_m: int
    m_tiles: int
    n_tiles: int
    k_blocks: int
    pair_tiles: int
    sm_pairs: int
    l2_bytes: int
    num_full: int
    tail_tiles: int
    sk_units: int
    iters_per_unit: int
    template: str
    # FlashInfer-only (see ``plan_dense_projection_gemm``): the batched small-M swap was undone / the row's
    # measured rule was dropped because the planned instance is not a generated program of this architecture
    # (the Cake host JIT-compiles it)
    swap_fallback: bool = False
    rule_fallback: bool = False
    # ... and the plan landed on the nearest registered knob variant (tile height, raster group, epilogue path,
    # BLOCK_N) because no generated program serves the row's default tail-T instance either
    knob_fallback: bool = False
    # round-13 tall-family knobs of the planned instance (instance_key fields 15 / 16 / 17)
    park: bool = False
    ovl: bool = False
    htail: bool = False
    # round-13 W3 knobs: MN-major B panel width (instance_key field 18) and the exact p-way tail split (field 19 is
    # the flag; ``sk_exact`` here is the part count p, 0 = off)
    b_swz: int = 128
    sk_exact: int = 0
    # round 13 (Cake W3, L38b): batch-entry raster of the batched MLA rows (instance_key field 20; 0 = row-major)
    batch_group: int = 0
    # round 13 (Cake W4): fp32 v8 register stores carry L1::no_allocate.L2::evict_first (instance_key field 21)
    store_ef: bool = False

    @property
    def num_cluster_tiles(self) -> int:
        return self.num_full + self.sk_units

    @property
    def wave_working_set_bytes(self) -> int:
        """Operand bytes one wave of CTA pairs touches (the hint gate compares it with ``l2_bytes``)."""
        return wave_working_set(
            self.m_tiles, self.n_tiles, self.K, self.group_m, self.sm_pairs
        )

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.num_cluster_tiles * CTA_GROUP, 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        """fp32 elements of the stream-K partial slabs (unit + tail-tile slabs); 0 without stream-K and
        under the half-height tail wave (its items are whole-K tiles: no partial slabs, no fixup)."""
        if not self.sk_units or self.htail:
            return 0
        return (self.sk_units + self.tail_tiles) * 2 * self.cta_rows * self.block_n

    @property
    def counters_u32(self) -> int:
        """Stream-K slice counters in use: 16 per tail tile (one per epilogue warp of the pair)."""
        return self.tail_tiles * 16

    @property
    def counters_alloc_u32(self) -> int:
        """Slice counters allocated: ``max(8192, tail_tiles * 16)`` like the Cake host's
        ``_sk_counters`` -- the dummy counters at ``SK_DUMMY_BASE + slice * 32 + lane`` (up to
        index 4607) must be addressable even when few slice counters are in use."""
        return max(SK_COUNTERS_MIN, self.counters_u32)

    @property
    def sk_iters(self) -> int:
        """The launcher's linearised stream-K span; under ``htail`` every half item is one whole-K segment; the
        exact split (``_skx`` instances) passes the part count p in this slot."""
        if self.sk_exact:
            return int(self.sk_exact)
        return (self.sk_units if self.htail else self.tail_tiles) * self.k_blocks

    @property
    def tma_out(self) -> bool:
        return self.epi == "tma"


def plan_dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    sm_count: int,
    l2_bytes: int,
    transposed_out: bool = False,
    sk: bool | str = "auto",
    block_n: Optional[int] = None,
    stages: Optional[int] = None,
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    cta_rows: Optional[int] = None,
    pf: Optional[int] = None,
    sk_max_units: Optional[int] = None,
    sk_parts: Optional[int] = None,
    promo: Optional[str] = None,
    hints: Optional[tuple] = None,
    group_m: Optional[int] = None,
    f32_v8: Optional[bool] = None,
    quad_store: Optional[bool] = None,
    park: Optional[bool] = None,
    ovl: Optional[bool] = None,
    htail: Optional[bool] = None,
    b_swz: Optional[int] = None,
    sk_exact: Optional[int] = None,
    batch_group: Optional[int] = None,
    store_ef: Optional[bool] = None,
    arch: str = "sm_100a",
    _fallback: bool = True,
    _allow_swap: bool = True,
    _use_rules: bool = True,
) -> tuple[GemmPlan, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Validate the views exactly as the Cake launcher does and plan the launch for a device
    with ``sm_count`` SMs and an L2 of ``l2_bytes`` (the two device facts the plan depends on:
    stream-K split and raster group through the CTA-pair count, the hint working-set gate
    through the L2).  Returns ``(plan, a_desc, b_desc, out3)`` with the ``[L, outer, inner]``
    operand views the TMA descriptors span and the batched output view.  The knob defaults
    (``sk="auto"``, ``pf`` / ``promo`` / ``group_m`` / ``hints`` from ``default_*``, resolved in
    the launcher's order) are the launcher's; a caller-forced ``cta_rows`` outside the row's
    rule (or default) family drops the rule's ``block_n`` / ``stages`` / ``slots`` / ``epi``, ``sk_parts`` is a
    caller knob over the rule's, and the stage count is bounded by ``smem_limit_for(arch)``.
    [Cake ``dense_projection_gemm`` L1234-L1358]

    One FlashInfer-only deviation: the Cake host applies ``swap_small_m`` unconditionally because it compiles
    the swapped instance on demand, while this package can only launch the generated programs registered for
    ``arch``.  When the plan resolves to an unregistered instance (a batched forward / input gradient with a
    tiny T that engages the swap, or a per-row rule whose template was only generated for the contract's T),
    the plan is redone without the swap and / or without the rule (``plan.swap_fallback`` /
    ``plan.rule_fallback``) so the call still runs; the result is identical, only the tile walk differs.
    ``_fallback=False`` returns the pure mirror (unit tests of the planner)."""
    _caller_knobs: dict[str, Any] = dict(
        transposed_out=transposed_out,
        sk=sk,
        block_n=block_n,
        stages=stages,
        epi=epi,
        slots=slots,
        cta_rows=cta_rows,
        pf=pf,
        sk_max_units=sk_max_units,
        sk_parts=sk_parts,
        promo=promo,
        hints=hints,
        group_m=group_m,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        b_swz=b_swz,
        sk_exact=sk_exact,
        batch_group=batch_group,
        store_ef=store_ef,
        arch=arch,
    )
    if A.dtype != torch.bfloat16 or B.dtype != torch.bfloat16:
        raise ValueError("dense_projection_gemm: A and B must be bf16")
    if out.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("dense_projection_gemm: out must be bf16 or fp32")
    A3, B3, O3 = as_batched(A, "A"), as_batched(B, "B"), as_batched(out, "out")
    L, M, K = (int(s) for s in A3.shape)
    if tuple(B3.shape) != (L, K, int(B3.shape[2])):
        raise ValueError(
            f"dense_projection_gemm: B must be [L={L}, K={K}, N], got {tuple(B3.shape)}"
        )
    N = int(B3.shape[2])
    expect = (L, N, M) if transposed_out else (L, M, N)
    if tuple(O3.shape) != expect:
        raise ValueError(
            f"dense_projection_gemm: out must be {expect}, got {tuple(O3.shape)}"
        )
    if N % 8 != 0 or M < 1 or K < 1 or N < 8:
        raise ValueError(
            f"dense_projection_gemm: N must be a positive multiple of 8 and M, K >= 1 (M={M}, N={N}, K={K})"
        )
    if (
        O3.stride(2) != 1
        or (O3.shape[1] > 1 and O3.stride(1) % 8 != 0)
        or (L > 1 and O3.stride(0) % 8 != 0)
        or out.data_ptr() % 16 != 0
    ):
        raise ValueError(
            "dense_projection_gemm: out needs unit inner stride, 16-byte aligned rows / batches "
            f"and base; strides {tuple(O3.stride())}"
        )
    swap_eligible = swap_small_m(L, M, N, transposed_out)
    swapped = _allow_swap and swap_eligible
    if swapped:
        A3, B3 = B3.transpose(1, 2), A3.transpose(1, 2)
        M, N = N, M
        transposed_out = True
    a_mn, a_desc = operand_view(A3, "A", k_axis=2)
    b_mn, b_desc = operand_view(B3, "B", k_axis=1)
    rule_dropped = (
        {}
        if _use_rules
        else row_rule(
            arch, a_mn, b_mn, out.dtype == torch.float32, transposed_out, L > 1, N, K, M
        )
    )
    rule = (
        row_rule(
            arch, a_mn, b_mn, out.dtype == torch.float32, transposed_out, L > 1, N, K, M
        )
        if _use_rules
        else {}
    )
    if cta_rows is None:
        cta_rows = rule.get("cta_rows", default_cta_rows(M, N, K))
    elif int(cta_rows) != rule.get("cta_rows", default_cta_rows(M, N, K)):
        # A caller-forced tile family (sweeps, the registrations of the tall / 64-row families): the rule's knobs
        # that belong to its own family (tile width, stage count, staging slots, epilogue mode) do not carry over
        # (a 160-column rule of the 256-row family is not a valid 64-row tile).  [Cake L1271-L1279]
        rule = {
            k: v
            for k, v in rule.items()
            if k not in ("block_n", "stages", "slots", "epi", "ovl", "htail", "b_swz")
        }
    if block_n is None:
        block_n = rule.get("block_n", default_block_n(N, b_mn))
    if stages is None:
        stages = rule.get("stages")
    if epi is None:
        epi = rule.get("epi")
    if slots is None:
        slots = rule.get("slots")
    if sk == "auto" and "sk" in rule:
        sk = rule["sk"]
    if f32_v8 is None:
        f32_v8 = rule.get("f32_v8", False)
    if quad_store is None:
        quad_store = rule.get("quad_store", False)
    if park is None:
        park = rule.get("park", False)
    if ovl is None:
        ovl = rule.get("ovl", False)
    if htail is None:
        htail = rule.get("htail", False)
    if b_swz is None:
        b_swz = rule.get("b_swz", 128)
    if sk_exact is None:
        sk_exact = rule.get("sk_exact")
    if batch_group is None:
        batch_group = rule.get("batch_group", 0)
    # the raster knob only exists for batched rows (one batch entry: identical raster)  [Cake launcher]
    batch_group = int(batch_group) if L > 1 else 0
    if batch_group < 0 or (batch_group > 1 and L % batch_group):
        raise ValueError(
            f"dense_projection_gemm: batch_group must be 0, 1 or a divisor of the batch size {L}, got {batch_group}"
        )
    if store_ef is None:
        store_ef = rule.get("store_ef", False)
    if ovl and int(block_n) != 256:
        # the overlapped epilogue is built for the 256-column tall tile; a narrower tall tile keeps the serialized
        # tall epilogue  [Cake launcher]
        ovl = False
    m_tiles = _ceil_div(M, cta_rows)
    m_tiles += m_tiles % CTA_GROUP
    n_tiles = _ceil_div(N, block_n)
    k_blocks = _ceil_div(K, BLOCK_K)
    pair_tiles = L * (m_tiles // CTA_GROUP) * n_tiles
    pairs = sm_pairs(sm_count)
    # Round 13 (Cake W3): a ``sk_exact`` rule / kwarg is the row's tail policy (exact p-way K split of every tail
    # tile, one unit per part); it pre-empts ``sk_parts`` and ``htail`` exactly as the Cake launcher orders them.
    exact_plan = (
        sk_exact_plan(pair_tiles, k_blocks, pairs, int(sk_exact))
        if (sk_exact and sk == "auto" and sk_max_units is None)
        else None
    )
    parts = rule.get("sk_parts") if sk_parts is None else int(sk_parts)
    if sk == "auto" and sk_max_units is None and parts and exact_plan is None:
        # Measured per-row p-way split of the tail wave (Cake round 4, L20); a caller ``sk_parts`` wins over the
        # rule's.  [Cake L1296-L1301]
        sk, sk_max_units = sk_parts_plan(pair_tiles, k_blocks, pairs, int(parts))
    # Round 13 (Cake W1): deterministic half-height tail wave of the tall family - the tail tiles become 2 x tail
    # standard-geometry items of full K (no partial slabs, no fixup), only when there is a tail and its half items fit
    # the CTA pairs; otherwise the plain plan / stream-K policy of the row applies.  [Cake launcher]
    htail = bool(htail) and int(cta_rows) == 256 and exact_plan is None
    if htail:
        tail_h = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
        htail = bool(tail_h) and 2 * tail_h <= pairs
    if exact_plan is not None:
        num_full, tail_tiles, sk_units, iters_per_unit = exact_plan
    elif htail:
        num_full, tail_tiles, sk_units, iters_per_unit = (
            pair_tiles - tail_h,
            tail_h,
            2 * tail_h,
            k_blocks,
        )
    else:
        num_full, tail_tiles, sk_units, iters_per_unit = stream_k_plan(
            pair_tiles, k_blocks, pairs, sk, sk_max_units
        )
    if tail_tiles * 16 > SK_DUMMY_BASE:
        raise ValueError(
            f"dense_projection_gemm: {tail_tiles} stream-K tail tiles exceed the {SK_DUMMY_BASE // 16} slice-counter budget"
        )
    out_f32 = out.dtype == torch.float32
    mode = epi_mode(out_f32, transposed_out, K, epi, block_n, cta_rows)
    nslots = epi_slots(mode, out_f32, block_n, K, slots, cta_rows, transposed_out)
    if pf is None:
        pf = rule.get("pf", default_pf(M, N, K))
    if promo is None:
        promo = rule.get("promo", default_promo(m_tiles, n_tiles))
    promo = str(promo)
    if promo not in L2_PROMOS:
        raise ValueError(
            f"dense_projection_gemm: promo must be one of {L2_PROMOS}, got {promo!r}"
        )
    if group_m is None:
        group_m = rule.get(
            "group_m", default_group_m(a_mn, b_mn, m_tiles, pair_tiles, pairs)
        )
    group_m = int(group_m)
    if group_m < 2 or group_m % 2:
        raise ValueError(
            f"dense_projection_gemm: group_m must be an even number >= 2 (CTA pairs are adjacent row tiles), got {group_m}"
        )
    if hints is None:
        hints = rule.get(
            "hints",
            default_hints(
                a_mn, b_mn, m_tiles, n_tiles, K, group_m, pairs, int(l2_bytes)
            ),
        )
    key = instance_key(
        a_mn=a_mn,
        b_mn=b_mn,
        out_f32=out_f32,
        out_t=transposed_out,
        block_n=block_n,
        stages=stages,
        diag=(),
        epi=mode,
        slots=nslots,
        box_rows=None,
        cta_rows=cta_rows,
        pf=pf,
        hints=hints,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        smem_limit=smem_limit_for(arch),
        b_swz=int(b_swz),
        sk_exact=exact_plan is not None,
        batch_group=batch_group,
        store_ef=bool(store_ef),
    )
    plan = GemmPlan(
        L=L,
        M=M,
        N=N,
        K=K,
        a_mn=a_mn,
        b_mn=b_mn,
        out_f32=out_f32,
        transposed_out=bool(transposed_out),
        block_n=int(block_n),
        cta_rows=int(cta_rows),
        stages=int(key[5]),
        epi=mode,
        slots=int(nslots),
        pf=int(pf),
        promo=str(promo),
        hints=tuple(key[12]),
        group_m=int(group_m),
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        pair_tiles=pair_tiles,
        sm_pairs=pairs,
        l2_bytes=int(l2_bytes),
        num_full=num_full,
        tail_tiles=tail_tiles,
        sk_units=sk_units,
        iters_per_unit=iters_per_unit,
        template=instance_symbol(key),
        swap_fallback=swap_eligible and not _allow_swap,
        rule_fallback=bool(rule_dropped),
        park=bool(key[15]),
        ovl=bool(key[16]),
        htail=bool(key[17]),
        b_swz=int(key[18]),
        sk_exact=int(sk_exact) if exact_plan is not None else 0,
        batch_group=int(key[20]),
        store_ef=bool(key[21]),
    )
    if _fallback and plan.template not in KERNELS.get(arch, {}):
        # nearest registered plan: drop the swap first (keeps the measured rule), then the rule, then both
        for allow_swap, use_rules in ((False, True), (True, False), (False, False)):
            if (allow_swap, use_rules) == (_allow_swap, _use_rules):
                continue  # the plan just made
            if (not allow_swap and not swapped) or (not use_rules and not rule):
                continue  # dropping a swap not taken / an empty rule changes nothing
            candidate = plan_dense_projection_gemm(
                A,
                B,
                out,
                sm_count=sm_count,
                l2_bytes=l2_bytes,
                _fallback=False,
                _allow_swap=allow_swap,
                _use_rules=use_rules,
                **_caller_knobs,
            )
            if candidate[0].template in KERNELS.get(arch, {}):
                return candidate
        # last resort: the nearest registered knob variant of the same layout / output kind (every knob set
        # computes the same GEMM; only the tile walk and the store path differ)
        for allow_swap in (True, False) if swapped else (False,):
            for knobs in _FALLBACK_KNOB_VARIANTS:
                merged: dict[str, Any] = dict(_caller_knobs)
                for name, value in knobs.items():
                    if merged.get(name) is None:
                        merged[name] = value
                if merged == _caller_knobs:
                    continue
                try:
                    candidate = plan_dense_projection_gemm(
                        A,
                        B,
                        out,
                        sm_count=sm_count,
                        l2_bytes=l2_bytes,
                        _fallback=False,
                        _allow_swap=allow_swap,
                        _use_rules=False,
                        **merged,
                    )
                except ValueError:
                    continue  # a knob the row cannot take (e.g. epi="tma" with a transposed store)
                if candidate[0].template in KERNELS.get(arch, {}):
                    return (
                        dataclasses.replace(candidate[0], knob_fallback=True),
                        *candidate[1:],
                    )
    return plan, a_desc, b_desc, O3


# Knob variants the registry fallback tries, nearest first (see ``plan_dense_projection_gemm``): the tall tile,
# the register epilogue (the canonical-T templates), then BLOCK_N = 128.  The TMA L2 promotion and the raster group
# width are launch parameters since round 11: they select no program and are not variants.
_FALLBACK_KNOB_VARIANTS = (
    dict(cta_rows=256),
    dict(epi="reg"),
    dict(epi="reg", f32_v8=True),
    dict(epi="reg", cta_rows=256),
    dict(block_n=128),
    dict(block_n=128, epi="reg"),
)


# ---------------------------------------------------------------------------
# Launch binding (shared by K1 and K2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Launch:
    module: str
    entry: Callable[..., Any] = field(repr=False)
    arguments: tuple = field(repr=False)
    grid: tuple[int, int, int]

    def __call__(self) -> None:
        self.entry(*self.arguments)


def bind_launch(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``module_name`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide raises
    ``KeyError`` naming both sides.  The grid must be a multiple of the cluster shape baked
    into the module.
    """
    record = MODULES[module_name]
    cluster = record.get("launch", {}).get("cluster")
    if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"{module_name}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
        )
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid_values[name])
        elif name in values and values[name] is not None:
            arguments.append(values[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    module = load_cake_dense_projection_gemm_module(module_name, arch)
    return _Launch(
        module_name, getattr(module, record["ffi_entry"]), tuple(arguments), grid
    )


_FFI_DEVICES: dict[int, Any] = {}


def _ffi_stream_context(index: int):
    """tvm-ffi environment-stream context for torch's current stream on device ``index``
    (the caller's downstream ops are stream-ordered after the kernel)."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    getter = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    raw = (
        getter(index)
        if getter is not None
        else torch.cuda.current_stream(index).cuda_stream
    )
    return tvm_ffi.use_raw_stream(device, raw)


def _device_index(device: torch.device) -> int:
    return int(
        device.index if device.index is not None else torch.cuda.current_device()
    )


# Small tensors an instance never dereferences (the operands of the epilogue path it does not
# take), cached per device like the Cake launcher's: a 16-element buffer per pointer dtype and a
# ``[1, 32, 64]`` tensor covering both output boxes for the unused output descriptor.
_DUMMIES: dict[tuple, torch.Tensor] = {}


def _dummy(device: torch.device, dtype) -> torch.Tensor:
    key = (str(device), dtype)
    if key not in _DUMMIES:
        _DUMMIES[key] = torch.zeros(16, device=device, dtype=dtype)
    return _DUMMIES[key]


def _dummy_map(device: torch.device, dtype) -> torch.Tensor:
    key = (str(device), dtype, "map")
    if key not in _DUMMIES:
        _DUMMIES[key] = torch.zeros((1, 32, 64), device=device, dtype=dtype)
    return _DUMMIES[key]


def flat_alias(t: torch.Tensor) -> torch.Tensor:
    """Contiguous 1-D view over the storage span of ``t`` (same data pointer), for pointer
    parameters: the kernel addresses rows / batches through ``ldo`` / ``out_l``."""
    span = 1 + sum(
        (int(s) - 1) * int(st)
        for s, st in zip(t.shape, t.stride(), strict=True)
        if int(s) > 0
    )
    return torch.as_strided(t, (span,), (1,))


# ---------------------------------------------------------------------------
# K1 prepared launch
# ---------------------------------------------------------------------------


@dataclass
class PreparedGemm:
    """One prepared dense projection GEMM launch.

    ``launch()`` writes ``out`` and returns it; it allocates nothing and never synchronizes.
    The object owns its stream-K workspace and its slice counters; prepare a new one when a shape, dtype, stride or tensor binding changes
    (values may change freely).
    """

    module_name: str
    arch: str
    plan: GemmPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_Launch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        with _ffi_stream_context(self.device_index):
            self._launch()
        return self.out

    __call__ = launch


def prepare_dense_projection_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    sk: bool | str = "auto",
    block_n: Optional[int] = None,
    stages: Optional[int] = None,
    epi: Optional[str] = None,
    slots: Optional[int] = None,
    cta_rows: Optional[int] = None,
    pf: Optional[int] = None,
    sk_max_units: Optional[int] = None,
    sk_parts: Optional[int] = None,
    promo: Optional[str] = None,
    hints: Optional[tuple] = None,
    group_m: Optional[int] = None,
    f32_v8: Optional[bool] = None,
    quad_store: Optional[bool] = None,
    park: Optional[bool] = None,
    ovl: Optional[bool] = None,
    htail: Optional[bool] = None,
    b_swz: Optional[int] = None,
    sk_exact: Optional[int] = None,
    batch_group: Optional[int] = None,
    store_ef: Optional[bool] = None,
) -> PreparedGemm:
    """Validate one binding, plan it for the device and prepare its launch (the only
    allocations of the K1 backend: the stream-K partial slabs and the slice counters).  See the module docstring for the view contract; the keyword
    knobs default to the Cake launcher's choices."""
    device = out.device
    if (
        not (A.is_cuda and B.is_cuda and out.is_cuda)
        or A.device != device
        or B.device != device
    ):
        raise ValueError("Expected A, B and out on one CUDA device")
    arch = require_arch(device)
    plan, a_desc, b_desc, O3 = plan_dense_projection_gemm(
        A,
        B,
        out,
        sm_count=device_sm_count(device),
        l2_bytes=device_l2_bytes(device),
        transposed_out=transposed_out,
        sk=sk,
        block_n=block_n,
        stages=stages,
        epi=epi,
        slots=slots,
        cta_rows=cta_rows,
        pf=pf,
        sk_max_units=sk_max_units,
        sk_parts=sk_parts,
        promo=promo,
        hints=hints,
        group_m=group_m,
        f32_v8=f32_v8,
        quad_store=quad_store,
        park=park,
        ovl=ovl,
        htail=htail,
        b_swz=b_swz,
        sk_exact=sk_exact,
        batch_group=batch_group,
        store_ef=store_ef,
        arch=arch,
    )
    module_name = select_module(arch, plan.template)
    if plan.ws_f32_elems:
        ws = torch.empty(plan.ws_f32_elems, device=device, dtype=torch.float32)
    else:
        ws = _dummy(device, torch.float32)
    # Zero-initialised slice counters (the kernel resets every counter it uses); sized like the Cake
    # host's ``_sk_counters`` so the lane-spread dummy counters above SK_DUMMY_BASE are addressable.
    counters = torch.zeros(plan.counters_alloc_u32, device=device, dtype=torch.uint32)
    dummy16, dummy32 = _dummy(device, torch.bfloat16), _dummy(device, torch.float32)
    o_alias = flat_alias(O3)
    tma_out, out_f32 = plan.tma_out, plan.out_f32
    tensors = dict(
        A=a_desc, B=b_desc, out3=O3, ws=ws, counters=counters, o_alias=o_alias
    )
    values: dict[str, Any] = dict(
        A=a_desc,
        B=b_desc,
        OUT32=O3 if (tma_out and out_f32) else _dummy_map(device, torch.float32),
        OUT16=O3 if (tma_out and not out_f32) else _dummy_map(device, torch.bfloat16),
        out=o_alias if (not out_f32 and not tma_out) else dummy16,
        out32=o_alias if (out_f32 and not tma_out) else dummy32,
        ws=ws,
        counters=counters,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        group_m=plan.group_m,
        promo_code=L2_PROMOS.index(plan.promo),
        k_iters=plan.k_blocks,
        ldo=int(O3.stride(1)),
        out_l=int(O3.stride(0)) if O3.shape[0] > 1 else 0,
        num_cluster_tiles=plan.num_cluster_tiles,
        num_l=plan.L,
        num_full=plan.num_full,
        iters_per_unit=plan.iters_per_unit,
        sk_iters=plan.sk_iters,
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    return PreparedGemm(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=_device_index(device),
        _launch=launch,
    )


def dense_projection_gemm(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, transposed_out: bool = False
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (prepare + launch)."""
    return prepare_dense_projection_gemm(
        A, B, out, transposed_out=transposed_out
    ).launch()


# --- the training operations as thin wrappers (the Cake entry points) ---------------------


def projection_forward(
    X: torch.Tensor, W: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` (bf16 out by default)."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.bfloat16)
    return dense_projection_gemm(X, W.t(), out)


def projection_dgrad(
    G: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]``."""
    if out is None:
        out = torch.empty(
            G.shape[0], W.shape[1], device=G.device, dtype=out_dtype or torch.bfloat16
        )
    return dense_projection_gemm(G, W, out)


def wgrad_views(
    G: torch.Tensor, X: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """``(A, B, transposed_out)`` of the weight gradient ``G[T, N].T @ X[T, K] -> dW[N, K]``:
    when ``wgrad_swapped`` the swapped GEMM ``X.T @ G`` runs with the transposed store (the
    result lands directly in ``[N, K]``), otherwise ``G.T @ X``.
    [Cake ``projection_wgrad`` L1507-L1518 / ``wgrad_swapped`` L1496-L1504]"""
    N, K = int(G.shape[1]), int(X.shape[1])
    if wgrad_swapped(N, K):
        return X.t(), G, True
    return G.t(), X, False


def wgrad_swapped(N: int, K: int) -> bool:
    """Orientation of the weight gradient: swapped when its padded tile work is smaller than the
    direct route's (256-row tiles padded from N rows and ``default_block_n`` column tiles over K,
    against 256-row tiles over K and column tiles padded from N): N < 256 and N = 576 swap.
    [Cake ``wgrad_swapped`` L1496-L1504]"""
    bk, bn = default_block_n(K, True), default_block_n(N, True)
    direct = _ceil_div(N, 256) * 256 * _ceil_div(K, bk) * bk
    swapped = _ceil_div(K, 256) * 256 * _ceil_div(N, bn) * bn
    return swapped < direct


def prepare_projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: torch.Tensor,
    *,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
) -> PreparedGemm:
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]`` with the small-``N`` swap rule of the Cake launcher."""
    A, B, transposed = wgrad_views(G, X)
    return prepare_dense_projection_gemm(
        A, B, out, transposed_out=transposed, sk=sk, cta_rows=cta_rows
    )


def projection_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    out_dtype=None,
    sk: bool | str = "auto",
    cta_rows: Optional[int] = None,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` (ragged reduction length ``T``); see :func:`wgrad_views`."""
    N, K = int(G.shape[1]), int(X.shape[1])
    if out is None:
        out = torch.empty(N, K, device=G.device, dtype=out_dtype or torch.bfloat16)
    return prepare_projection_wgrad(G, X, out, sk=sk, cta_rows=cta_rows).launch()


# ---------------------------------------------------------------------------
# K2 planning and prepared launch
# ---------------------------------------------------------------------------


def router_symbol(a_mn: bool, b_mn: bool, out_t: bool) -> str:
    """Kernel symbol / registry template of a router instance (``router_fp32_gemm_<a><b>[_t]``)."""
    return (
        "router_fp32_gemm_"
        + ("n" if a_mn else "k")
        + ("n" if b_mn else "k")
        + ("_t" if out_t else "")
    )


def router_layout_class(a_mn: bool, b_mn: bool) -> str:
    """Instance layout class of the operand layouts: ``kk`` (forward), ``kn`` (input
    gradient) or ``nn_t`` (both MN-major: the swapped transposed-store weight gradient)."""
    if not a_mn and not b_mn:
        return "kk"
    if not a_mn and b_mn:
        return "kn"
    if a_mn and b_mn:
        return "nn_t"
    raise NotImplementedError(
        "router_fp32_gemm: an MN-major A with a K-major B has no registered instance "
        f"(see {TRACKING_ISSUE})"
    )


@dataclass(frozen=True)
class RouterPlan:
    M: int
    N: int
    K: int
    a_mn: bool
    b_mn: bool
    transposed_out: bool
    swapped: bool  # the caller's (A, B) were swapped into the transposed-store form
    layout: str
    splits: int
    m_tiles: int
    n_tiles: int
    k_blocks: int
    k_iters_split: int
    template: str

    @property
    def num_cluster_tiles(self) -> int:
        return self.splits * (self.m_tiles // CTA_GROUP) * self.n_tiles

    @property
    def grid(self) -> tuple[int, int, int]:
        return (self.num_cluster_tiles * CTA_GROUP, 1, 1)


def _router_a_layout(A: torch.Tensor) -> bool:
    if A.stride(1) == 1:
        return False
    if A.stride(0) == 1:
        return True
    raise ValueError("router_fp32_gemm: A needs a unit stride on m or k")


def _router_b_layout(B: torch.Tensor) -> bool:
    if B.stride(0) == 1:  # k contiguous -> K-major parts [3, N, K]
        return False
    if B.stride(1) == 1:  # n contiguous -> MN-major parts [3, K, N]
        return True
    raise ValueError("router_fp32_gemm: B needs a unit stride on k or n")


def plan_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
) -> tuple[RouterPlan, torch.Tensor, torch.Tensor]:
    """Classify the fp32 views, apply the swap rule of the weight gradient and plan the launch.

    Returns ``(plan, A_eff, B_eff)``: the operands of the launched (possibly swapped) GEMM;
    ``B_eff`` is the operand the host splits into BF16x3 parts.  ``splits`` overrides the
    production split-K count of the layout class (:data:`ROUTER_SPLITS`)."""
    if (
        A.dtype != torch.float32
        or B.dtype != torch.float32
        or out.dtype != torch.float32
    ):
        raise ValueError("router_fp32_gemm: A, B and out must be fp32")
    if A.dim() != 2 or B.dim() != 2 or out.dim() != 2:
        raise ValueError("router_fp32_gemm: A, B and out must be 2-D views")
    M, K = (int(s) for s in A.shape)
    Kb, N = (int(s) for s in B.shape)
    if Kb != K:
        raise ValueError(
            f"router_fp32_gemm: A is [M={M}, K={K}] but B is [{Kb}, {N}]; K must agree"
        )
    if tuple(out.shape) != (M, N):
        raise ValueError(
            f"router_fp32_gemm: out must be [{M}, {N}], got {tuple(out.shape)}"
        )
    if M < 1 or K < 1 or N < 1:
        # An empty problem (a ragged batch with T = 0 reaches here as M = 0 for the
        # forward / input gradient or K = 0 for the weight gradient) would otherwise
        # produce a zero-sized grid or a TMA descriptor with a zero extent.
        raise ValueError(
            f"router_fp32_gemm: M, K and N must be >= 1 (M={M}, N={N}, K={K})"
        )
    a_mn, b_mn = _router_a_layout(A), _router_b_layout(B)
    layout = router_layout_class(a_mn, b_mn)
    swapped = layout == "nn_t"
    if swapped:
        # out.T[n, m] = sum_k B.T[n, k] A.T[k, m]: A' = B.T (MN-major), B' = A.T (MN-major), transposed store
        A_eff, B_eff, transposed = B.t(), A.t(), True
        M_eff, N_eff = N, M
    else:
        A_eff, B_eff, transposed = A, B, False
        M_eff, N_eff = M, N
    a_eff_mn, b_eff_mn = _router_a_layout(A_eff), _router_b_layout(B_eff)
    lda = int(A_eff.stride(1)) if a_eff_mn else int(A_eff.stride(0))
    if lda % 4 != 0 or A_eff.data_ptr() % 16 != 0:
        raise ValueError("router_fp32_gemm: A rows must be 16-byte aligned")
    if B_eff.data_ptr() % 16 != 0:
        raise ValueError("router_fp32_gemm: B must start at a 16-byte aligned address")
    # the kernel's output view: [M_eff, N_eff] row-major, or [N_eff, M_eff] with the transposed store = out
    if out.stride(1) != 1 or out.stride(0) % 4 != 0 or N_eff % 8 != 0:
        raise ValueError(
            "router_fp32_gemm: out must have unit inner stride and 16-byte rows; N a multiple of 8"
        )
    if splits is None:
        splits = ROUTER_SPLITS[layout]
    splits = int(splits)
    if splits < 1:
        raise ValueError("router_fp32_gemm: splits must be >= 1")
    m_tiles = _ceil_div(M_eff, ROUTER_BLOCK_M)
    m_tiles += m_tiles % CTA_GROUP
    n_tiles = _ceil_div(N_eff, ROUTER_BLOCK_N)
    k_blocks = _ceil_div(K, ROUTER_BLOCK_K)
    plan = RouterPlan(
        M=M_eff,
        N=N_eff,
        K=K,
        a_mn=a_eff_mn,
        b_mn=b_eff_mn,
        transposed_out=transposed,
        swapped=swapped,
        layout=layout,
        splits=splits,
        m_tiles=m_tiles,
        n_tiles=n_tiles,
        k_blocks=k_blocks,
        k_iters_split=_ceil_div(k_blocks, splits),
        template=router_symbol(a_eff_mn, b_eff_mn, transposed),
    )
    return plan, A_eff, B_eff


def split_fp32_to_bf16x3_(
    src: torch.Tensor, parts: torch.Tensor, resid: torch.Tensor
) -> torch.Tensor:
    """Exact three-way BF16 split ``src = p1 + p2 + p3`` (``p1 = RN(src)``, ``p2 = RN(src - p1)``,
    ``p3 = RN(src - p1 - p2)``) into the retained ``parts [3, *src.shape]`` BF16 stack with the
    fp32 scratch ``resid`` -- no allocation; the same values as the Cake host's
    ``split_fp32_to_bf16x3``."""
    p1, p2, p3 = parts[0], parts[1], parts[2]
    p1.copy_(src)
    torch.sub(src, p1, out=resid)  # exact in fp32
    p2.copy_(resid)
    torch.sub(resid, p2, out=resid)  # exact in fp32
    p3.copy_(resid)
    return parts


@dataclass
class PreparedRouterGemm:
    """One prepared FP32 router GEMM: the retained BF16x3 stack of the second operand (re-split
    from the bound fp32 view on every ``launch()``), the split-K workspace and the fixed-order
    host reduction.  ``launch()`` writes ``out`` and returns it without allocating."""

    module_name: str
    arch: str
    plan: RouterPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_Launch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    @property
    def splits(self) -> int:
        return self.plan.splits

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        t = self.tensors
        split_fp32_to_bf16x3_(t["split_src"], t["parts"], t["resid"])
        with _ffi_stream_context(self.device_index):
            self._launch()
        if self.plan.splits > 1:
            torch.sum(
                t["workspace"], dim=0, out=self.out
            )  # fixed-order reduction of the partials
        return self.out

    __call__ = launch


def prepare_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
) -> PreparedRouterGemm:
    """Validate one fp32 binding, plan it and prepare its launch (the only allocations of the
    K2 backend: the BF16x3 stack and its fp32 residual scratch and the split-K workspace)."""
    device = out.device
    if (
        not (A.is_cuda and B.is_cuda and out.is_cuda)
        or A.device != device
        or B.device != device
    ):
        raise ValueError("Expected A, B and out on one CUDA device")
    arch = require_arch(device)
    plan, A_eff, B_eff = plan_router_fp32_gemm(A, B, out, splits=splits)
    module_name = select_module(arch, plan.template)
    # the split source in the operand's own layout: K-major parts [3, N, K] come from B.T, MN-major parts [3, K, N] from B
    split_src = B_eff if plan.b_mn else B_eff.t()
    parts = torch.empty((3, *split_src.shape), device=device, dtype=torch.bfloat16)
    resid = torch.empty(split_src.shape, device=device, dtype=torch.float32)
    a_view = (
        A_eff.unsqueeze(0) if not plan.a_mn else A_eff.t().unsqueeze(0)
    )  # [1, M, K] or [1, K, M]
    expect = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
    tensors = dict(A=a_view, split_src=split_src, parts=parts, resid=resid)
    if plan.splits > 1:
        workspace = torch.empty(
            (plan.splits, *expect), device=device, dtype=torch.float32
        )
        tensors["workspace"] = workspace
        target = workspace
        ldo, out_l = expect[1], expect[0] * expect[1]
    else:
        target = out
        ldo, out_l = int(out.stride(0)), 0
    out32 = target if target.is_contiguous() else flat_alias(target)
    values: dict[str, Any] = dict(
        A=a_view,
        B=parts,
        out32=out32,
        M=plan.M,
        N=plan.N,
        m_tiles=plan.m_tiles,
        n_tiles=plan.n_tiles,
        k_iters_split=plan.k_iters_split,
        ldo=ldo,
        out_l=out_l,
        num_cluster_tiles=plan.num_cluster_tiles,
    )
    launch = bind_launch(module_name, values, plan.grid, arch)
    return PreparedRouterGemm(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=_device_index(device),
        _launch=launch,
    )


def router_fp32_gemm(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, splits: Optional[int] = None
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the emulation (prepare + launch)."""
    return prepare_router_fp32_gemm(A, B, out, splits=splits).launch()


def router_forward(
    X: torch.Tensor,
    W: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: int = 4,
) -> torch.Tensor:
    """``out[T, N] = X[T, K] @ W[N, K].T`` in FP32 through the emulation."""
    if out is None:
        out = torch.empty(X.shape[0], W.shape[0], device=X.device, dtype=torch.float32)
    return router_fp32_gemm(X, W.t(), out, splits=splits)


def router_dgrad(
    G: torch.Tensor, W: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """``out[T, K] = G[T, N] @ W[N, K]`` in FP32 through the emulation."""
    if out is None:
        out = torch.empty(G.shape[0], W.shape[1], device=G.device, dtype=torch.float32)
    return router_fp32_gemm(G, W, out, splits=1)


def router_wgrad(
    G: torch.Tensor,
    X: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    splits: int = 11,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a transposed store."""
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    return router_fp32_gemm(G.t(), X, out, splits=splits)
