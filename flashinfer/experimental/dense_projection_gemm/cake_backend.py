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
transposed-store ``nn_t`` weight gradient 6).

Two host paths share one planner and one binding:

* the prepared path -- :func:`prepare_dense_projection_gemm` /
  :func:`prepare_router_fp32_gemm` plan once and own their workspace; the
  returned launch replays with no CUDA allocation and no host synchronization
  for new values written into the bound tensors.  Every ``launch()`` is
  launch-only: the generated binding encodes the operand tensor maps by value
  from the bound views on each call (no descriptor workspace), so a prepared
  launch can be captured into a CUDA Graph from its first call;
* the convenience path -- :func:`dense_projection_gemm`,
  :func:`projection_forward` / :func:`projection_dgrad` / :func:`projection_wgrad`,
  :func:`router_fp32_gemm`, :func:`router_forward` / :func:`router_dgrad` /
  :func:`router_wgrad` -- keeps a bounded per-process cache of planned launches
  keyed by the device, the stream and the geometry of the views (shapes,
  strides, dtypes, storage offsets, alignment, flags); a call that hits the
  cache rebinds the caller's views and launches without planning, device
  queries or allocation (``out`` is allocated only when the caller passes
  none).  The cached launches of one device and stream share one workspace
  arena (:class:`_Arena`: one growable buffer per workspace kind) instead of
  owning a workspace each, so the memory the convenience path holds is bounded
  by the largest geometry seen, not by the number of geometries (a ragged
  token count).  See ``CONVENIENCE_CACHE_SIZE`` and
  :func:`clear_convenience_caches`.

The device facts a plan depends on (architecture, SM count, L2 size) are read
once per device index (:func:`device_facts`).
"""

from __future__ import annotations

import dataclasses
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Tuple, Union

import torch
import tvm_ffi

from .cake_jit import (
    KERNELS,
    MODULES,
    TRACKING_ISSUE,
    arg_plan_of,
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
# CTAs per work item: 2 = the pair tile (cta_group::2 MMA over 2 x 128 rows, cluster (2, 1, 1)); 1 = the round-19
# (Cake W3) single-CTA form ``cta1`` (one 128 x BLOCK_N tile per CTA, cluster (1, 1, 1))  [Cake L59]
CTA_GROUP = 2
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
W8_FORMS = (
    "w8r",
    "w8rs",
)  # the eight-warp CTA form of the input-gradient GEMM (instance_key field 37; ``_w8r`` / ``_w8rs`` symbols)
PD_CHOICES = (
    0,
    1,
    2,
    3,
)  # round 15 (Cake W1): pipelined-drain forms of the 128-row single-pass epilogue (0 = the blocking load)
SK_SLAB_CHOICES = (
    0,
    1,
    2,
    3,
)  # round 19 (Cake W2): slab-path forms of the synchronised stream-K fixup (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32 output as the slab)
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
ROUTER_SPLITS = {"kk": 4, "kn": 1, "nn_t": 6}


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DeviceFacts:
    """The facts of one CUDA device the host path depends on, read once per device index: the architecture
    tag (``None`` on an unsupported device), the SM count (CTA pairs: stream-K split, raster group, wave working
    set) and the L2 size (the TMA eviction-hint working-set gate)."""

    index: int
    arch: Optional[str]
    sm_count: int
    l2_bytes: int


_DEVICE_FACTS: dict[int, DeviceFacts] = {}


def _cuda_index(device: Any = None) -> int:
    """Ordinal of a CUDA device (a ``torch.device``, an index or a ``"cuda[:n]"`` string; ``None`` or a bare
    ``cuda`` is the current device).  ``torch.device`` objects of one device are not interchangeable as dict
    keys (``cuda`` versus ``cuda:0``), so every per-device cache is keyed by this ordinal."""
    if device is None:
        return int(torch.cuda.current_device())
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    if device.index is None:
        return int(torch.cuda.current_device())
    return int(device.index)


def _driver_l2_bytes(index: int) -> int:
    from cuda.bindings import driver as cuda

    err, l2 = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, index
    )
    if err != cuda.CUresult.CUDA_SUCCESS or not l2:
        raise RuntimeError(
            f"dense_projection_gemm: cannot resolve the L2 size of cuda:{index} ({err})"
        )
    return int(l2)


def device_facts(device: Any = None) -> DeviceFacts:
    """The cached :class:`DeviceFacts` of ``device``: one ``torch.cuda.get_device_properties`` query per
    device index for the lifetime of the process.  The L2 size is ``L2_cache_size`` of the properties,
    falling back to ``cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)`` -- the same source as the
    Cake host's ``_l2_bytes`` (no per-SKU table).  [Cake ``_l2_bytes`` L1377-L1392]"""
    index = _cuda_index(device)
    facts = _DEVICE_FACTS.get(index)
    if facts is None:
        props = torch.cuda.get_device_properties(index)
        l2 = getattr(props, "L2_cache_size", None) or _driver_l2_bytes(index)
        facts = _DEVICE_FACTS[index] = DeviceFacts(
            index=index,
            arch=SUPPORTED_COMPUTE_CAPABILITIES.get(
                (int(props.major), int(props.minor))
            ),
            sm_count=int(props.multi_processor_count),
            l2_bytes=int(l2),
        )
    return facts


def arch_for(device: Optional[torch.device] = None) -> Optional[str]:
    """Architecture tag of ``device`` (``None`` when unsupported or without CUDA)."""
    if not torch.cuda.is_available():
        return None
    if device is not None and torch.device(device).type != "cuda":
        return None
    return device_facts(device).arch


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
    """SM count of ``device`` (cached per device index, see :func:`device_facts`)."""
    return device_facts(device).sm_count


def sm_pairs(sm_count: int) -> int:
    """CTA pairs the persistent grid can hold at once (one pair per two SMs).  [Cake ``_sm_pairs`` L1365-L1371]"""
    return max(1, int(sm_count) // CTA_GROUP)


EXACT_WAVE_BLOCK_N = 224  # the wave-quantisation tile width the exact-wave rules select (``exact_waves``)


def exact_waves(pair_tiles: int, units: int) -> bool:
    """Whether ``pair_tiles`` work items fill the device's ``units`` concurrent work items (CTA pairs, or SMs under
    ``cta1``) in whole waves - ``((m_tiles // 2) * n_tiles * L) % sm_pairs == 0`` in the pair form.  The BLOCK_N 224
    weight-gradient rules were chosen for this fit on the 76-pair measuring devices (152 SMs: the 4096 x 4096 row's
    152 tall tiles are two waves, the 4096 x 1024 row's 76 tiles one); the exact-wave gate of
    :func:`plan_dense_projection_gemm` reads it for the running device."""
    return pair_tiles > 0 and pair_tiles % units == 0


def sm_units(sm_count: int, cta_group: int = CTA_GROUP) -> int:
    """Concurrent work items of the launch: CTA pairs (``cta_group`` 2) or SMs (the single-CTA form ``cta1``,
    round 19).  [Cake ``_sm_units``]"""
    if int(cta_group) == CTA_GROUP:
        return sm_pairs(sm_count)
    return max(1, int(sm_count))


def device_l2_bytes(device: torch.device) -> int:
    """L2 size of ``device`` in bytes (cached per device index, see :func:`device_facts`): resolves the hint
    working-set gate exactly as the Cake host's ``_l2_bytes`` does."""
    return device_facts(device).l2_bytes


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


def stt_bytes(stt: bool) -> int:
    """Dedicated staging bytes of the transposed stmatrix / TMA store (round 27, Cake lane 7, knob ``stt``): 0 - the
    SMEM plan stages the per-warp EPI_COLS x 64 B slots in the DEAD mainloop stage ring of the CTA's last work item, so
    no mainloop stage is traded; a dedicated-slot plan would be entered here and flow into ``default_stages`` and the
    ``instance_key`` SMEM check together.  [Cake ``stt_bytes``]"""
    return 0


def b_stage_bytes(
    b_mn: bool, block_n: int, b_swz: int = 128, cta_group: int = CTA_GROUP
) -> int:
    """Bytes of one B stage per CTA: K-major = ``BLOCK_N / cta_group`` 128-byte rows; MN-major = whole panels of
    ``b_swz / 2`` columns (BLOCK_N = 192 with 128-byte panels loads two 64-column panels per stage and the MMA
    reads 1.5 of them; with 64-byte panels it loads exactly the three 32-column panels it reads - round 13).
    The single-CTA form (``cta1``, round 19) streams the whole BLOCK_N columns per CTA.  [Cake ``b_stage_bytes``]"""
    n_half = block_n // cta_group
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
    cta_group: int = CTA_GROUP,
    stt: bool = False,
) -> int:
    """Mainloop stages that fit the 227 KiB opt-in with the epilogue staging: 32 KiB stages
    for 128-row tiles (24 KiB at BLOCK_N = 128, where the streaming-bound small-N rows are
    still latency-bound at 7 stages: 9 / 8 / 6 for 0 / 1 / 2 slots), 48 KiB stages for tall
    (256-row) tiles.  Layout-B tiles (64 rows: 24 KiB stages at BLOCK_N = 256, 16 KiB at 128)
    take the deepest pipeline that fits beside the staging, at most 12 stages (round 9;
    ``b_mn`` sizes the MN-major B panels).  Narrow-panel instances (``b_swz`` 64 / 32, round 13) have smaller
    stages than their 128-byte-panel siblings and likewise take the deepest pipeline that fits the opt-in (at most
    12).  The single-CTA form (``cta1``, round 19: 48 KiB stages at BLOCK_N 256, 32 KiB at 128) takes the deepest
    fit the same way.  ``stt`` (round 27, Cake lane 7) subtracts the transposed stmatrix store's dedicated staging bytes
    (``stt_bytes``: 0 under the dead-ring plan) beside the slots'.  [Cake ``default_stages``]"""
    if cta_rows == 64 or int(b_swz) != 128 or int(cta_group) == 1:
        stage = min(cta_rows, 128) * a_halves_of(
            cta_rows
        ) * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cta_group)
        return max(
            2,
            min(
                12,
                (SMEM_OPT_IN - WORK_STAGES * 16 - staging_bytes(slots) - stt_bytes(stt))
                // stage,
            ),
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
    cta_group: int = CTA_GROUP,
) -> tuple[int, int]:
    """TMA box heights of the A and B operands: full height unless capped by ``box_rows``.
    [Cake ``box_rows_of`` L860-L870]"""
    a_full = BLOCK_K if a_mn else cta_rows
    b_full = BLOCK_K if b_mn else block_n // cta_group
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
    store_hint: str = "none",
    pd: int = 0,
    sh: int = 0,
    sk_sync: bool = False,
    sk_slab: int = 0,
    cta1: bool = False,
    cgrp: bool = False,
    ovr: bool = False,
    stt: bool = False,
    pro: int = 0,
    sk_first: bool = False,
    lw: bool = False,
    mg: bool = False,
    sw: int = 0,
    form: str = "",
) -> tuple:
    """The instance tuple the Cake kernel module traces one program per (validation included):
    ``(a_mn, b_mn, out_f32, out_t, block_n, stages, diag, epi, slots, box_rows, cta_rows, pf,
    hints, f32_v8, quad_store, park, ovl, htail, b_swz, sk_exact)`` (round 13: ``park`` = parked tall epilogue, ``ovl`` =
    overlapped single-TMEM-buffer tall epilogue, ``htail`` = deterministic half-height tail wave, ``b_swz`` = MN-major B
    panel width in bytes (128 / 64 / 32; K-major B instances always keep 128), ``sk_exact`` = exact p-way stream-K
    split, ``_skx`` symbols; round 18 (Cake W2): ``sk_sync`` = synchronised stream-K tail, field 26, ``_sks`` symbols;
    round 19: ``sk_slab`` (Cake W2, field 27, ``_sb{n}``) and ``cta1`` (Cake W3, field 28, ``_c1``); round 21: ``cgrp`` (Cake W3, field 29, ``_cg``); round 24: ``ovr`` (the round-20 Cake W3 knob ported in round 23 by W4, field 30 = LAST, ``_or``), see below); the raster group width (``group_m``) and the TMA L2 promotion (``promo_code``) are launch parameters since round 11 (under ``cgrp`` the group counts column tiles).  ``smem_limit`` (bytes; default = the largest
    architecture limit, ``smem_limit_for(None)``) only bounds the stage count - it is not part of
    the key, so an instance has one symbol on every architecture (the planner passes
    ``smem_limit_for(arch)`` like the Cake launcher).  Diagnostic (attribution) instances are not
    exported.  [Cake ``instance_key`` L873-L905]  ``batch_group`` (round 13, L38b) = the batch-entry raster of the
    batched MLA rows: 0 = off, 1 = every entry (``_bf``), G > 1 = G adjacent entries interleaved across the CTA pairs
    (``_bg{G}``); the planner keeps 0 for a batch of one.  ``store_ef`` (round 13, Cake W4) = st.global.L1::no_allocate.L2::evict_first on the fp32 v8
    register stores (``_ef``); only that store form carries the hint, so the field is False for every other epilogue.
    ``store_hint`` (round 15, Cake W3) = the L2 eviction policy operand of the TMA-store epilogue's bulk tensor stores
    (field 22, ``_so{f|l|n}``); only the TMA-store epilogue carries the operand, so the field is "none" for every other
    epilogue.  ``pd`` (round 15, Cake W1) = the software-pipelined chunked TMEM drain of the 128-row single-pass family
    (field 23, ``_pd{n}``: 1 = in-tile chunk pipelining, 2 = cross-tile prefetch, 3 = both chunks in flight) and ``sh``
    (field 24, ``_sh{n}``) = the PTX suspendTimeHint in ns of the free-running waits; both 0 when off.
    ``sk_slab`` (round 19, Cake W2, field 27, symbol ``_sb{n}``) selects the slab path of the synchronised stream-K fixup
    (1 early arrival; 2 + the bulk slab read through the dead mainloop stages; 3 + the fp32 output as the slab: fp32
    outputs through the transposed register path or the TMA-store epilogue); it is forced to 0 without ``sk_sync`` so
    every other instance keeps one key.  ``cta1`` (round 19, Cake W3, field 28, symbol ``_c1``) selects the
    single-CTA form of the 128-row single-pass family: one 128 x BLOCK_N tile per CTA through the cta_group::1 MMA into
    private TMEM, every barrier local, CLC per CTA (cluster dims 1) - no pair handoff; it excludes the pair-only tail
    policies / probes (``htail`` / ``sk_exact`` / ``sk_sync`` / ``a_mcast``), the tall and 64-row families (``ovl`` /
    ``park``) and the batched raster knob.  Its B stage streams the whole BLOCK_N columns per CTA, so the stage
    geometry (``b_stage_bytes`` / ``default_stages`` / ``box_rows_of``) takes the CTA group.
    ``cgrp`` (round 21, Cake W3, field 29, symbol ``_cg``) selects the column-grouped raster: the ``group_m``
    launch parameter counts COLUMN tiles per raster group (any count >= 1) and consecutive pairs sweep a group's
    columns of one pair row before the next pair row (a group's B panels stay L2-resident, A streams once per column
    group); every other field is unchanged and the OFF form renders byte-identical.
    ``ovr`` (round 24; the round-20 Cake W3 knob ported in round 23 by W4, field 30 = LAST, symbol ``_or`` after
    ``_cg``) selects the overlapped tall epilogue released by row half: the eight epilogue warps drain TMEM buffer 0
    (rows [0, 128), 32 rows x 128 columns per warp) and release it to the next tile's half-0 MMAs while buffer 1 is
    drained; the B stage keeps the plain tall layout (one N = 256 MMA per row half per K step - half the tcgen05.mma
    issues and 3/4 of the tensor-core SMEM operand traffic of ``ovl``).  Tall 256-column tiles only; it excludes
    ``ovl`` / ``park`` and the pf / box_rows / a_mcast probes, and (like ``ovl``) the synchronised stream-K plan and
    the single-CTA form; every other field is unchanged and the OFF form renders byte-identical.
    ``stt`` (round 27, Cake lane 7 / W-3 variant A1, field 31 = LAST after ``ovr`` = 30, symbol ``_stt`` directly after
    ``_sks``) selects the transposed stmatrix / TMA-store staging of the bf16 transposed REGISTER epilogue of the 128-row
    single-pass pair family: each epilogue warp writes its 32 x EPI_COLS bf16 slice with ``stmatrix.trans`` into a
    64 B-swizzled slot of the dead mainloop stage ring of the CTA's last work item and stores it with one bulk tensor
    store (``OUT16`` = the real [L, N, M] view, box (32, EPI_COLS, 1)); every other slice keeps the register stores
    through ``out``.  bf16 transposed outputs only (stmatrix is .b16), whole 32-column warp slices up to 256 columns,
    no ovl / ovr / park / htail / pd / cta1, ``sk_slab`` 0 / 1, and the pool must fit the 227 KiB opt-in; every other
    field is unchanged and the OFF form renders byte-identical.
    ``pro`` (round 33, Cake unit 2d / 2e, field 32 = LAST after ``stt`` = 31, symbol ``_pro{n}`` directly after ``_or`` =
    the last knob suffix) is the bitmask of the pair program's prologue / cleanup sync form: 1 = no peer-startup
    cluster rendezvous before the collective tcgen05.alloc, 2 = relaxed arrive on the post-init cluster sync, 4 = the
    MMA warp allocates TMEM, 8 = CTA-local exit (no cleanup barrier.cluster: role-drained rings, a leader -> follower
    tmem_free handoff and the collective dealloc on warp 1).  Pair programs only (the single-CTA form ``cta1`` refuses
    it), 0..15, and bit 8 admits no diag probe but ``trace`` (the planner never emits diag); 0 renders the round-27
    program byte-identical.  The production value is the merged row rule's (``row_rule`` lays ``ARCH_RULE_DEFAULTS``
    under every row rule: 7 on sm_107a and sm_100a; a row rule may pin another value), zeroed on the ``cta1`` rows.
    ``sk_first`` (the round-2 shapes tranche, field 33, symbol ``_skf`` directly after ``_skx``) is the tail-first
    persistent order of the exact p-way split: pair work items ``[0, sk_units)`` are the K parts of the tail tiles and
    item ``sk_units + w`` is whole tile ``w``; the plan arithmetic, the counters, the slabs and the launch contract are
    ``sk_exact``'s - only the program differs.  It needs ``sk_exact`` and keeps the plain item walk (no ``pd`` 2,
    ``sk_slab`` 2, ``stt``, ``sk_sync`` or ``htail``).  ``lw`` (the round-2 shapes tranche, field 34 = LAST, symbol
    ``_lw`` after ``_or`` and before ``_pro``) lets the load warp leave the pair prelude before the post-allocation CTA
    rendezvous (the first TMA loads issue once the cluster rendezvous and the barrier init are done); pair programs
    whose MMA warp allocates TMEM only (``pro`` bit 4, never ``cta1``).  ``mg`` (field 35 = LAST, symbol ``_mg``
    directly after ``_lw``) selects the whole-tile tile resolve: the load warp and the MMA warp take the whole-tile
    constant arm of the work-item walk (one segment per item, no division) and the load warp resolves the raster's
    three quotients through an fp32 reciprocal with an exact +-1 correction (four reciprocals set up once per CTA);
    the epilogue keeps the plain path.  Whole tiles only (the kernel requires a plan without stream-K units and
    fewer than 2**21 cluster tiles), so it excludes the batch raster, the column-grouped raster, the half-height
    tail and every stream-K form (``sk_exact`` / ``sk_first`` / ``sk_sync`` / ``sk_slab`` / ``stt``); every other
    field is unchanged and the OFF form renders byte-identical.  ``sw`` (field 36, symbol ``_sw<n>``
    directly after ``_mg`` and before ``_pro``; 0 = off) paces the output stores of the bf16 TMA-store pd-2 drain:
    each epilogue warp delays its bulk tensor store issue, after the staging write and its fences, by a per-warp
    number of steps of ``sw`` dependent u32 multiply-adds seeded by a clock sample (even chunks (9 - warp) steps,
    odd chunks (warp - 2) steps), so each per-item eight-store burst becomes a seven-step ramp; the stores and
    bytes are unchanged.  A step count >= 0 that needs that drain (``slots`` > 0, ``pd`` 2, no fp32 output);
    every other field is unchanged and the OFF form renders byte-identical.  ``form`` (field 37 = LAST, symbol
    ``_w8r`` / ``_w8rs`` directly after ``_pro<n>`` and before the diag tokens; "" = the ten-warp pipeline of every
    other program) selects the eight-warp CTA form of the same GEMM: ``w8r`` = four epilogue warps, a loader, a
    scheduler and the MMA warp on CTA-pair 256 x 256 tiles over six 32 KiB stages, whole tiles through
    cluster-launch-control items, the BF16 output through the TMA store and the raster quotients resolved by four
    fast divisors the host passes (``fd_tiles_l`` / ``fd_tiles_g`` / ``fd_group`` / ``fd_last``), with the
    accumulator released by every epilogue warp after its last read; ``w8rs`` = the same with the loader's elected
    lane issuing the cluster-launch-control queries (no scheduler warp).  It is refused with every field it cannot
    carry: a layout other than ``kn``, an fp32 or transposed output, the single-CTA form, a tile other than the
    128 x 256 pair tile, an epilogue other than the one-slot TMA store, a narrow B panel, a box cap, any tail or
    raster form (``sk_exact`` / ``sk_first`` / ``sk_sync`` / ``sk_slab`` / ``htail`` / ``cgrp`` / ``batch_group`` /
    ``stt`` / ``ovl`` / ``ovr`` / ``park``) and any pair-program knob off its plain value (``pd`` / ``pro`` / ``lw`` /
    ``mg`` / ``sw`` / a non-default stage count / ``hints`` / ``pf`` / ``sh`` / ``store_hint`` / ``store_ef`` /
    ``quad_store`` / ``f32_v8``); the OFF form renders byte-identical.
    [Cake ``instance_key``]"""
    a_mn, b_mn, out_f32, out_t = bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t)
    block_n, cta_rows, pf = int(block_n), int(cta_rows), int(pf)
    cta1 = bool(cta1)
    cgrp = bool(cgrp)
    ovr = bool(ovr)
    stt = bool(stt)
    cg = 1 if cta1 else CTA_GROUP
    hints = (str(hints[0]), str(hints[1]))
    if any(h not in L2_HINTS for h in hints):
        raise ValueError(f"hints must be in {L2_HINTS}, got {hints!r}")
    if pf < 0 or pf > 16:
        raise ValueError(f"prefetch distance must be in 0..16 K steps, got {pf}")
    if cta_rows not in CTA_ROWS_CHOICES:
        raise ValueError(f"cta_rows must be one of {CTA_ROWS_CHOICES}, got {cta_rows}")
    box_rows = 0 if box_rows is None else int(box_rows)
    box_rows_of(a_mn, b_mn, block_n, box_rows or None, cta_rows, cg)
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
    # a transposed box stages CH_T_COLS output rows per chunk (round 14); row-major chunks are 128 bytes  [Cake L1260]
    if epi == "tma" and cols % (CH_T_COLS if out_t else (32 if out_f32 else 64)):
        raise ValueError(
            f"the TMA-store epilogue needs whole 128-byte column chunks per warp; BLOCK_N={block_n} "
            f"CTA_ROWS={cta_rows} {'fp32' if out_f32 else 'bf16'} output needs epi='reg'"
        )
    slots = epi_slots(epi, out_f32, block_n, None, slots, cta_rows, out_t)
    stages = (
        default_stages(slots, cta_rows, block_n, b_mn, b_swz, cg, stt)
        if stages is None
        else int(stages)
    )
    diag = tuple(sorted(set(diag)))
    if diag:
        raise ValueError(f"diagnostic instances are not exported: {diag}")
    limit = smem_limit_for(None) if smem_limit is None else int(smem_limit)
    stage_bytes = cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
    if (
        stages < 2
        or stages * stage_bytes
        + staging_bytes(slots)
        + stt_bytes(stt)
        + WORK_STAGES * 16
        > limit
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
    if ovr and (
        cta_rows != 256
        or block_n != 256
        or ovl
        or park
        or pf
        or box_rows
        or "a_mcast" in diag
    ):
        # round 20 (Cake W3, ported in round 23 by W4): the row-half release is the overlapped epilogue of the tall
        # 256-column tile over the plain B stage (one N = 256 MMA per row half per K step); it excludes the column-chunk
        # release, the parked epilogue and the prefetch / box cap / A-multicast probe (the planner never emits diag, so
        # the a_mcast term is inert; kept for message parity)  [Cake instance_key]
        raise ValueError(
            f"ovr needs cta_rows=256, block_n=256 and no ovl / park / pf / box_rows / a_mcast (got cta_rows={cta_rows}, "
            f"block_n={block_n}, ovl={ovl}, park={park}, pf={pf}, box_rows={box_rows}, diag={diag})"
        )
    htail = bool(htail)
    if htail and (cta_rows not in (128, 256) or pf or box_rows or "a_mcast" in diag):
        raise ValueError(
            f"htail (half-height tail wave) needs cta_rows=256 or 128 and no pf / box_rows / a_mcast (got cta_rows={cta_rows}, "
            f"pf={pf}, box_rows={box_rows})"
        )
    if (
        htail
        and cta_rows == 128
        and (block_n != 256 or out_t or int(pd) or "no_tmem" in diag)
    ):
        # round 16 (Cake W2): the half-height tail wave on the standard family (M=128 cta_group::2 half items, TMEM Layout B)
        # needs the 256-column row-major tile with the plain drain  [Cake instance_key]
        raise ValueError(
            f"htail on the 128-row family needs BLOCK_N=256, a row-major output and pd=0 without the no_tmem probe "
            f"(got block_n={block_n}, out_t={out_t}, pd={pd}, diag={diag})"
        )
    if htail and sk_exact:
        raise ValueError(
            "htail (half-height tail wave) and sk_exact (exact p-way tail split) are two tail policies - choose one per instance"
        )
    sk_sync = bool(sk_sync)
    if sk_sync and (htail or sk_exact):
        raise ValueError(
            "sk_sync (synchronised stream-K) excludes htail and sk_exact - one tail policy per instance"
        )
    if sk_sync and (ovl or ovr or park or int(pd) or "a_mcast" in diag):
        # round 18 (Cake W2): the synchronised plan's two-partial fixup closes a unit's single-pass or serialised tall
        # epilogue; the overlapped (column-chunk ``ovl`` / row-half ``ovr``) / parked tall forms, the pipelined drain
        # and the A multicast are excluded  [Cake instance_key]
        raise ValueError(
            f"sk_sync needs the single-pass or the serialised tall epilogue (no ovl / ovr / park / pd / a_mcast; got ovl={ovl}, "
            f"ovr={ovr}, park={park}, pd={pd}, diag={diag})"
        )
    # L1::no_allocate.L2::evict_first on the fp32 v8 stores: only that store form carries the hint  [Cake instance_key]
    store_ef = bool(store_ef) and f32_v8
    store_hint = str(store_hint)
    if store_hint not in L2_HINTS:
        raise ValueError(f"store_hint must be in {L2_HINTS}, got {store_hint!r}")
    # round 15 (Cake W3): only the TMA-store epilogue carries the L2 eviction-policy operand  [Cake instance_key]
    store_hint = store_hint if epi == "tma" else "none"
    pd, sh = int(pd), int(sh)
    if pd not in PD_CHOICES:
        raise ValueError(
            f"pd must be one of {PD_CHOICES} (0 off, 1 chunk-pipelined drain, 2 cross-tile pipelined drain, 3 both chunks in flight), got {pd}"
        )
    if pd and (
        cta_rows != 128
        or out_t
        or cols not in (64, 128)
        or "no_tmem" in diag
        or box_rows
        or pf
        or "a_mcast" in diag
    ):
        # round 15 (Cake W1): the pipelined drain is built for the 128-row single-pass family's row-major epilogues
        # (64- / 128-column warp slices = one / two 64-column TMEM chunks)  [Cake instance_key]
        raise ValueError(
            f"pd needs cta_rows=128, a row-major output, BLOCK_N 128 / 256 and no no_tmem / a_mcast / pf / box_rows "
            f"(got cta_rows={cta_rows}, out_t={out_t}, block_n={block_n}, diag={diag}, pf={pf}, box_rows={box_rows})"
        )
    if pd in (1, 3) and cols != 128:
        raise ValueError(
            f"pd={pd} overlaps the two chunks of a 128-column warp slice; BLOCK_N={block_n} has one chunk per slice (use pd=2)"
        )
    if sh < 0 or sh > 4294967295:
        raise ValueError(f"sh (suspendTimeHint ns) must fit a u32, got {sh}")
    # round 19 (Cake W2): the slab-path forms exist only on the synchronised stream-K programs  [Cake instance_key]
    sk_slab = int(sk_slab) if sk_sync else 0
    if sk_slab not in SK_SLAB_CHOICES:
        raise ValueError(
            f"sk_slab must be one of {SK_SLAB_CHOICES} (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32 output as the slab), got {sk_slab}"
        )
    if sk_slab == 3 and (not out_f32 or (epi == "reg" and not out_t)):
        raise ValueError(
            f"sk_slab=3 (the fp32 output as the slab) needs an fp32 output through the transposed register path or the "
            f"TMA-store epilogue (got out_f32={out_f32}, out_t={out_t}, epi={epi!r})"
        )
    # the eight warp slabs of the bulk slab read stage in the mainloop stage ring (the pair form's stage: the slab
    # forms exist only on the synchronised programs, which the single-CTA form excludes)  [Cake instance_key]
    slab_bytes = EPI_WARPS * cols * 128
    if sk_slab == 2 and slab_bytes > stages * (
        cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
    ):
        raise ValueError(
            f"sk_slab=2 stages the eight warp slabs ({slab_bytes} B) in the mainloop stage ring ({stages} stages); too small"
        )
    if stt and (
        out_f32
        or not out_t
        or epi != "reg"
        or cta_rows != 128
        or cta1
        or ovl
        or ovr
        or park
        or htail
        or int(pd)
        or cols % 32
        or cols > 256
        or sk_slab not in (0, 1)
    ):
        # round 27 (Cake lane 7, W-3 A1): the transposed stmatrix / TMA-store staging is built for the bf16 transposed
        # REGISTER epilogue of the 128-row single-pass pair family (stmatrix is .b16: the fp32 rows are excluded; the
        # 16x256b drain covers whole 32-column blocks; the box outer extent is <= 256; the ovl / ovr / park / htail / pd
        # forms drain and store through their own chunk paths; the bulk slab read (sk_slab 2) copies a column-group
        # prefix that the 16x256b order no longer is; sk_slab 3 is fp32-only; the single-CTA form's handoff is
        # unmeasured)  [Cake instance_key]
        raise ValueError(
            f"stt (transposed stmatrix / TMA-store staging) needs a bf16 transposed output through the single-pass "
            f"register epilogue of the 128-row pair family (no ovl / ovr / park / htail / pd) with whole 32-column warp "
            f"slices of at most 256 columns and sk_slab 0 / 1 (got out_f32={out_f32}, out_t={out_t}, epi={epi!r}, "
            f"cta_rows={cta_rows}, cta1={cta1}, ovl={ovl}, ovr={ovr}, park={park}, htail={htail}, pd={pd}, "
            f"cols={cols}, sk_slab={sk_slab})"
        )
    stt_slots = EPI_WARPS * cols * 64
    if stt and stt_slots > stages * (
        cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
    ):
        raise ValueError(
            f"stt stages the eight warp slots ({stt_slots} B) in the dead mainloop stage ring ({stages} stages); too small"
        )
    stt_pool = (
        stages * (cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg))
        + staging_bytes(slots)
        + stt_bytes(stt)
        + WORK_STAGES * 16
    )
    if stt and stt_pool > SMEM_OPT_IN:
        # round 27 (Cake lane 7): an stt program must fit the 227 KiB opt-in - the oversized shared-memory mode is not
        # shippable  [Cake instance_key]
        raise ValueError(
            f"stt needs the dynamic SMEM pool inside the {SMEM_OPT_IN} B opt-in (got {stt_pool} B at {stages} stages)"
        )
    if cta1 and (
        cta_rows != 128
        or ovl
        or ovr
        or park
        or htail
        or sk_exact
        or sk_sync
        or batch_group
        or "a_mcast" in diag
    ):
        # round 19 (Cake W3): the single-CTA form is the 128-row single-pass family without the pair-level tail
        # policies / probes (and without the tall family's overlapped epilogues ``ovl`` / ``ovr``)  [Cake instance_key]
        raise ValueError(
            f"cta1 needs cta_rows=128 and no ovl / ovr / park / htail / sk_exact / sk_sync / batch_group / a_mcast (got "
            f"cta_rows={cta_rows}, ovl={ovl}, ovr={ovr}, park={park}, htail={htail}, sk_exact={sk_exact}, sk_sync={sk_sync}, "
            f"batch_group={batch_group}, diag={diag})"
        )
    pro = int(pro)
    if pro not in range(16):
        raise ValueError(
            f"pro must be a bitmask in 0..15 (1 = no startup rendezvous before the collective TMEM alloc, 2 = relaxed "
            f"arrive on the post-init cluster sync, 4 = the MMA warp allocates TMEM, 8 = CTA-local exit: no cleanup "
            f"barrier.cluster, role-drained rings + leader->follower tmem_free handoff + warp-1 collective dealloc); got {pro}"
        )
    if pro and cta1:
        # round 33 (Cake unit 2d): the pair prologue / cleanup sync form needs the cluster pair  [Cake instance_key]
        raise ValueError(
            "pro (pair prologue / cleanup sync form) needs the cluster pair (cta1=False)"
        )
    if pro & 8 and any(d != "trace" for d in diag):
        # round 33 (Cake unit 2e): the CTA-local exit's remote-op inventory covers the production protocol (+ the trace
        # probe's local stamps only); the planner never emits diag, so this term is inert - kept for message parity
        # [Cake instance_key]
        raise ValueError(
            f"pro bit 8 (CTA-local exit) admits no diag probe except 'trace' (got diag={diag})"
        )
    sk_first = bool(sk_first)
    if sk_first and not sk_exact:
        raise ValueError(
            "sk_first (tail-first persistent order) needs the exact p-way split (sk_exact)"
        )
    if sk_first and (pd == 2 or sk_slab == 2 or stt or sk_sync or htail):
        raise ValueError(
            f"sk_first keeps the plain item walk - no pd 2, sk_slab 2, stt, sk_sync or htail (got pd={pd}, "
            f"sk_slab={sk_slab}, stt={stt}, sk_sync={sk_sync}, htail={htail})"
        )
    lw = bool(lw)
    if lw and (cta1 or not (pro & 4)):
        raise ValueError(
            f"lw (the load warp out of the pair prelude) needs the cluster pair with the MMA-warp allocator (pro bit 4); "
            f"got cta1={cta1}, pro={pro}"
        )
    mg = bool(mg)
    if mg and (
        batch_group
        or cgrp
        or htail
        or sk_exact
        or sk_first
        or sk_sync
        or sk_slab
        or stt
    ):
        raise ValueError(
            "mg (whole-tile tile resolve) is written for the plain row-grouped raster and the whole-tile walk: no "
            f"batch_group, cgrp, htail or stream-K form (got batch_group={batch_group}, cgrp={cgrp}, htail={htail}, "
            f"sk_exact={bool(sk_exact)}, sk_first={sk_first}, sk_sync={sk_sync}, sk_slab={sk_slab}, stt={stt})"
        )
    sw = int(sw)
    if sw < 0 or (sw and (epi != "tma" or not slots or int(pd) != 2 or out_f32)):
        # the paced output stores are written for the bf16 TMA-store pd-2 drain (the kernel's refusal)
        raise ValueError(
            f"sw (paced output stores) is a step count >= 0 and needs the bf16 TMA-store pd 2 drain (slots > 0, pd 2, "
            f"no out_f32); got sw={sw}, epi={epi!r}, slots={slots}, pd={pd}, out_f32={out_f32}"
        )
    form = str(form or "")
    if form and (
        form not in W8_FORMS
        or a_mn
        or not b_mn
        or out_f32
        or out_t
        or cta1
        or block_n != 256
        or cta_rows != 128
        or epi != "tma"
        or slots != 1
        or b_swz != 128
        or box_rows
        or sk_exact
        or sk_first
        or sk_sync
        or sk_slab
        or htail
        or cgrp
        or batch_group
        or stt
        or ovl
        or ovr
        or park
        or pd
        or pro
        or lw
        or mg
        or sw
        or stages != default_stages(slots, cta_rows, block_n, b_mn, b_swz, cg, stt)
        or hints != ("none", "none")
        or pf
        or sh
        or store_hint != "none"
        or store_ef
        or quad_store
        or f32_v8
    ):
        # the eight-warp CTA form is written for the kn bf16 row-major 128 x 256 pair tile with the one-slot TMA-store
        # epilogue, whole tiles and none of the pair programs' pipeline knobs (the kernel's frozen configurations)
        raise ValueError(
            f"form (the eight-warp CTA form) must be one of {W8_FORMS} on the kn bf16 row-major 128 x 256 pair tile "
            f"with the one-slot TMA-store epilogue, the default stage count, plain hints and no tail, raster or "
            f"pipeline knob; got form={form!r}, a_mn={a_mn}, b_mn={b_mn}, out_f32={out_f32}, out_t={out_t}, "
            f"cta1={cta1}, block_n={block_n}, cta_rows={cta_rows}, epi={epi!r}, slots={slots}, b_swz={b_swz}, "
            f"box_rows={box_rows}, stages={stages}, hints={hints}, pd={pd}, pro={pro}, lw={lw}, mg={mg}, sw={sw}, "
            f"sk_exact={bool(sk_exact)}, sk_first={sk_first}, sk_sync={sk_sync}, sk_slab={sk_slab}, htail={htail}, "
            f"cgrp={cgrp}, batch_group={batch_group}, stt={stt}, ovl={ovl}, ovr={ovr}, park={park}, pf={pf}, "
            f"sh={sh}, store_hint={store_hint!r}, store_ef={store_ef}, quad_store={quad_store}, f32_v8={f32_v8}"
        )
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
        store_hint,
        pd,
        sh,
        sk_sync,
        sk_slab,
        cta1,
        cgrp,
        ovr,
        stt,
        pro,
        sk_first,
        lw,
        mg,
        sw,
        form,
    )


def instance_symbol(key: tuple) -> str:
    """Kernel symbol / registry template of an instance key (``dense_proj_gemm_<a><b>_n<N>``
    followed by ``_m256`` for tall tiles / ``_m64`` for the 64-row Layout-B family, ``_pf<n>`` for a
    prefetch distance, ``_h<a><b>`` for non-default (A, B) eviction hints (first letters, e.g.
    ``_hen`` = A evict_first / B none), ``_f32``, ``_v8`` for the 256-bit fp32 register stores, ``_t``,
    ``_bz<bytes>`` right after the tile family for a narrow MN-major B panel (round 13), ``_pk`` / ``_ov`` / ``_ht`` for
    the parked / overlapped / half-height-tail tall epilogues (round 13), ``_<epi><slots>`` for the TMA-store
    epilogue, ``_s<stages>`` for a non-default stage count, ``_box<rows>`` and a trailing ``_skx`` for the exact
    p-way stream-K split (round 13), ``_sks`` for the synchronised stream-K tail (round 18) followed by ``_sb<n>`` for
    its slab path (round 19), ``_so<f|l|n>`` after the batch-raster term for the TMA-store L2 eviction policy,
    then ``_pd<n>`` / ``_sh<n>`` for the pipelined TMEM drain / suspend-time hint (round 15), ``_c1`` for
    the single-CTA form (round 19; its default stage count is the single-CTA fit), ``_cg`` for the column-grouped
    raster (round 21) and ``_or`` for the tall epilogue released by row half (round 24); ``_stt`` directly
    after ``_sks`` for the transposed stmatrix / TMA-store staging (round 27); a trailing ``_pro<n>`` after ``_or`` for
    the pair prologue / cleanup sync form (round 33; absent when 0 - the single-CTA ``_c1`` programs never carry it);
    the round-2 shapes tranche adds ``_skf`` directly after ``_skx`` (``sk_first``) and ``_lw`` between ``_or`` and
    ``_pro`` (``lw``); ``_mg`` directly after ``_lw`` for the whole-tile tile resolve (``mg``); ``_sw<n>`` directly
    after ``_mg`` for the paced output stores (``sw``, absent when 0); ``_w8r`` / ``_w8rs`` directly after ``_pro<n>``
    and before the diag tokens for the eight-warp CTA form (``form``, absent when "")).
    [Cake ``instance_symbol``]"""
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
        store_hint,
        pd,
        sh,
        sk_sync,
        sk_slab,
        cta1,
        cgrp,
        ovr,
        stt,
        pro,
        sk_first,
        lw,
        mg,
        sw,
        form,
    ) = key
    cg = 1 if cta1 else CTA_GROUP
    so = {
        "evict_first": "f",
        "evict_last": "l",
        "evict_normal": "n",
        "default": "d",
    }.get(store_hint, "")
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
            if stages != default_stages(slots, cta_rows, block_n, b_mn, b_swz, cg, stt)
            else ""
        )
        + (f"_box{box_rows}" if box_rows else "")
        + ("_skx" if sk_exact else "")
        + ("_skf" if sk_first else "")
        + ("_sks" if sk_sync else "")
        + ("_stt" if stt else "")
        + (f"_sb{sk_slab}" if sk_slab else "")
        + (
            "_bf"
            if batch_group == 1
            else f"_bg{batch_group}"
            if batch_group > 1
            else ""
        )
        + (f"_so{so}" if so else "")
        + (f"_pd{pd}" if pd else "")
        + (f"_sh{sh}" if sh else "")
        + ("_c1" if cta1 else "")
        + ("_cg" if cgrp else "")
        + ("_or" if ovr else "")
        + ("_lw" if lw else "")
        + ("_mg" if mg else "")
        + (f"_sw{sw}" if sw else "")
        + (f"_pro{pro}" if pro else "")
        + (f"_{form}" if form else "")
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
# The round-2 to round-33 rules mirror the former launcher's row table; the shapes tranche below them (sm_100a and
# sm_103a) was measured with this package's planner and extends it per architecture (three round-33 sm_100a keys were
# re-measured and re-valued there).  An architecture-wide default (``ARCH_RULE_DEFAULTS``, round 33) sits under every rule
# of that architecture (``row_rule`` merges it; a row rule overrides).  Three sm_103a input-gradient rows (4096 -> 512,
# 5120 -> 512 and the kv_a 6144 -> 576) carry ``mg`` with ``sk: False``: the whole-tile tile resolve (``_mg`` programs)
# measured faster than the row's program above the pair spread at both T on GB300, and those programs run whole tiles
# only (the planner drops a rule's ``mg`` where the plan carries stream-K units and refuses a caller-forced one there).
# Two sm_103a input-gradient rows (5120 -> 512 and 5120 -> 1024) carry ``sw`` 25: the paced output stores (``_sw25``
# programs) measured faster than the row's program above the pair spread at both T on GB300; the knob needs the bf16
# TMA-store pd-2 drain (the planner drops a rule's ``sw`` where the planned program is not that drain or a caller forces
# the single-CTA form, and refuses a caller-forced one there).  Three sm_103a input-gradient rows (4096 -> 512, 5120 -> 512
# and the kv_a 6144 -> 576) carry ``form``: the eight-warp CTA form of the same GEMM (``_w8r`` / ``_w8rs`` programs)
# measured faster than the row's program above the pair spread at both T on GB300; the knobs after it are the row's
# program where the form yields (the planner drops a rule's form where the plan carries stream-K units, the single-CTA
# form, another tile or output class, the register epilogue or a caller-forced pipeline knob, and refuses a caller-forced
# form there).
ROW_RULES: dict[tuple, dict] = {
    ('sm_100a', False, False, False, False, False, 32, 6144, None): {"cta1": True},
    ('sm_100a', False, False, False, False, False, 128, 6144, None): {"cta1": True},
    ('sm_100a', False, False, False, False, False, 576, 6144, None): {"hints": ('evict_first', 'evict_last'), "stages": 8, "pro": 15},
    ('sm_100a', False, False, False, False, False, 2048, 6144, None): {"group_m": 8, "cgrp": True},
    ('sm_100a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_100a', False, False, False, False, False, 16384, 2048, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_100a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_100a', False, True, False, False, False, 2048, 6144, None): {"group_m": 8, "pro": 0},
    ('sm_100a', False, True, False, False, False, 2048, 16384, None): {"cta_rows": 256, "sk_parts": 2, "ovl": True, "htail": True},
    ('sm_100a', False, True, False, False, False, 6144, 32, None): {"slots": 2, "pd": 2},
    ('sm_100a', False, False, False, False, False, 4096, 2048, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 2048, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 6144, 128, None): {"slots": 2, "pd": 2, "pro": 15},
    ('sm_100a', False, True, False, False, False, 6144, 576, None): {"group_m": 8, "slots": 1, "epi": 'tma', "pd": 2, "pro": 15},
    ('sm_100a', False, True, False, False, False, 6144, 2048, None): {"group_m": 8},
    ('sm_100a', False, True, False, False, False, 6144, 12288, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True},
    ('sm_100a', False, True, False, False, True, 512, 256, None): {"cta_rows": 256, "group_m": 4, "batch_group": 4},
    ('sm_100a', False, True, True, False, False, 2048, 4096, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 2048, 16384, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 32, None): {"block_n": 128, "slots": 2, "stages": 4, "cta1": True},
    ('sm_100a', False, True, True, False, False, 6144, 128, None): {"block_n": 128, "group_m": 2},
    ('sm_100a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b'},
    ('sm_100a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 12288, 6144, None): {"epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_100a', False, True, True, False, False, 16384, 6144, None): {"epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, False, False, False, 2048, None, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 2048, None, 6144): {"group_m": 4, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 2048, None, 16384): {"cta_rows": 256, "ovl": True, "htail": True},
    ('sm_100a', True, True, False, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_100a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "group_m": 8, "htail": True},
    ('sm_100a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, False, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first'), "epi": 'tma'},
    ('sm_100a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "epi": 'tma'},
    ('sm_100a', True, True, True, False, False, 2048, None, 4096): {"cta_rows": 256, "sk_sync": True, "sk_sync_m": 62, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 2048, None, 6144): {"group_m": 8, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_100a', True, True, True, False, False, 6144, None, 2048): {"group_m": 32, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_100a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_100a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256, "htail": True},
    ('sm_100a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64},
    ('sm_100a', True, True, True, True, False, 576, None, 6144): {"hints": ('evict_first', 'evict_first'), "epi": 'tma'},
    ('sm_107a', False, False, False, False, False, 32, 6144, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, False, 576, 6144, None): {"cta_rows": 256, "hints": ('evict_first', 'none'), "stages": 5, "pro": 15},
    ('sm_107a', False, False, False, False, False, 6144, 2048, None): {"htail": True, "pro": 15},
    ('sm_107a', False, False, False, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, False, False, False, False, 12288, 6144, None): {"cta_rows": 256, "htail": True, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, False, False, False, False, 6144, 16384, None): {"cta_rows": 256, "htail": True, "ovr": True, "cgrp": True, "group_m": 8},
    ('sm_107a', False, False, False, False, False, 16384, 2048, None): {"group_m": 32, "pro": 15},
    ('sm_107a', False, False, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 256, 512, None): {"epi": 'reg', "quad_store": True, "promo": 'l2_256b'},
    ('sm_107a', False, False, False, False, True, 512, 256, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, False, 6144, 32, None): {"cta_rows": 128, "slots": 2, "pd": 3, "group_m": 2, "cta1": True},
    ('sm_107a', False, True, False, False, False, 6144, 128, None): {"cta_rows": 128, "slots": 2, "pd": 2, "pro": 15},
    ('sm_107a', False, True, False, False, False, 6144, 576, None): {"group_m": 8, "epi": 'reg', "quad_store": True, "pd": 1},
    ('sm_107a', False, True, False, False, False, 6144, 2048, None): {"htail": True, "pro": 15},
    ('sm_107a', False, True, False, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3, "ovr": True, "cgrp": True, "group_m": 12},
    ('sm_107a', False, True, False, False, False, 12288, 6144, None): {"hints": ('none', 'evict_first')},
    ('sm_107a', False, True, False, False, False, 16384, 6144, None): {"cta_rows": 256, "group_m": 8, "ovl": True, "htail": True, "pro": 15},
    ('sm_107a', False, True, False, False, True, 192, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 256, 512, None): {"promo": 'l2_256b'},
    ('sm_107a', False, True, False, False, True, 512, 192, None): {"batch_group": 8, "store_hint": "evict_last"},
    ('sm_107a', False, True, False, False, True, 512, 256, None): {"promo": 'l2_256b', "batch_group": 8},
    ('sm_107a', False, True, True, False, False, 2048, 4096, None): {"epi": 'reg', "f32_v8": True, "stages": 9, "store_ef": True, "pro": 15},
    ('sm_107a', False, True, True, False, False, 2048, 6144, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, True, False, False, 2048, 16384, None): {"group_m": 8, "epi": 'reg', "f32_v8": True},
    ('sm_107a', False, True, False, False, False, 2048, 16384, None): {"cgrp": True, "group_m": 8},
    ('sm_107a', False, True, True, False, False, 6144, 32, None): {"block_n": 128, "slots": 1, "group_m": 2, "cta1": True},
    ('sm_107a', False, True, True, False, False, 6144, 128, None): {"block_n": 128, "cta1": True, "group_m": 2, "pd": 2},
    ('sm_107a', False, True, True, False, False, 6144, 576, None): {"group_m": 8, "promo": 'l2_256b', "pro": 15},
    ('sm_107a', False, True, True, False, False, 6144, 2048, None): {"group_m": 8, "epi": 'reg', "f32_v8": True, "htail": True},
    ('sm_107a', False, True, True, False, False, 6144, 12288, None): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', False, True, True, False, False, 12288, 6144, None): {"cta_rows": 256},
    ('sm_107a', False, True, True, False, False, 16384, 6144, None): {"cta_rows": 256},
    ('sm_107a', True, True, False, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "group_m": 4, "sk_parts": 2, "b_swz": 64, "pro": 15},
    ('sm_107a', True, True, False, False, False, 2048, None, 6144): {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2},
    ('sm_107a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "sk_parts": 2},
    ('sm_107a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "htail": True},
    ('sm_107a', True, True, False, False, False, 16384, None, 6144): {"cta_rows": 256, "sk_parts": 3},
    ('sm_107a', True, True, False, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2},
    ('sm_107a', True, True, False, True, False, 32, None, 6144): {"block_n": 128, "sk_exact": 3},
    ('sm_107a', True, True, False, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', True, True, False, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64, "sk_sync": True, "sk_sync_m": 38, "stt": True, "pro": 15},
    ('sm_107a', True, True, False, True, True, 192, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma', "b_swz": 64, "stages": 6},
    ('sm_107a', True, True, False, True, True, 256, None, 512): {"cta_rows": 256, "hints": ('evict_first', 'evict_first'), "epi": 'tma', "stages": 6},
    ('sm_107a', True, True, True, False, False, 2048, None, 4096): {"block_n": 160, "cta_rows": 256, "sk_parts": 3, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 2048, None, 6144): {"sk_sync": True, "sk_sync_m": 38, "sk_slab": 2, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', True, True, True, False, False, 2048, None, 16384): {"group_m": 8, "epi": 'reg', "f32_v8": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 6144, None, 2048): {"sk_sync": True, "sk_sync_m": 26, "sk_slab": 2, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', False, False, False, False, False, 2048, 6144, None): {"group_m": 8, "cgrp": True, "pro": 15},
    ('sm_107a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "htail": True},
    ('sm_107a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "htail": True, "epi": 'reg', "f32_v8": True, "store_ef": True},
    ('sm_107a', True, True, True, False, False, 16384, None, 6144): {"cta_rows": 256},
    ('sm_107a', True, True, True, True, False, 32, None, 6144): {"block_n": 128, "sk_exact": 3},
    ('sm_107a', True, True, True, True, False, 128, None, 6144): {"block_n": 128, "cta_rows": 64, "sk_exact": 2, "pro": 15},
    ('sm_107a', False, False, False, False, False, 128, 6144, None): {"pro": 15},
    ('sm_107a', False, False, False, False, False, 4096, 2048, None): {"pro": 15},
    ('sm_107a', False, True, False, False, False, 2048, 4096, None): {"pro": 15},
    ('sm_107a', True, True, False, False, False, 2048, None, 16384): {"pro": 15},
    ('sm_107a', True, True, True, True, False, 576, None, 6144): {"group_m": 8, "hints": ('evict_first', 'evict_first'), "b_swz": 64, "sk_sync": True, "sk_sync_m": 38, "sk_slab": 2},
    # ---- sm_100a: the shapes tranche - forward / input gradient / weight gradient rows at K 4096 - 17408 and N 32 - 17408,
    #      measured with this package's planner against cuBLAS on GB200 at T 16172 - 16384 (see the test table
    #      MEASURED_SHAPE_ROWS); a row whose measured choice equals the planner's default carries no rule ----
    ('sm_100a', False, False, False, False, False, 32, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 32, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 48, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 48, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 64, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 64, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_100a', False, False, False, False, False, 128, 8192, None): {"pro": 15},
    ('sm_100a', False, False, False, False, False, 1024, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 1024, 5120, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, False, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_100a', False, False, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, False, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, True, False, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_100a', False, True, False, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_100a', False, True, False, False, False, 4096, 512, None): {"epi": 'tma', "slots": 1, "pro": 15, "pd": 2},
    ('sm_100a', False, True, False, False, False, 4096, 1024, None): {"pd": 2, "pro": 15},
    ('sm_100a', False, True, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_100a', False, True, False, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_100a', False, True, False, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True},
    ('sm_100a', False, True, False, False, False, 5120, 512, None): {"epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 8},
    ('sm_100a', False, True, False, False, False, 5120, 1024, None): {"pd": 2, "pro": 15},
    ('sm_100a', False, True, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_100a', False, True, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovr": True, "htail": True},
    ('sm_100a', False, True, False, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_100a', False, True, True, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', False, True, True, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_100a', True, True, False, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 32, None, 5120): {"pro": 15},
    ('sm_100a', True, True, False, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 48, None, 5120): {"pro": 15},
    ('sm_100a', True, True, False, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 64, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 128, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 32, None, 5120): {"pro": 15},
    ('sm_100a', True, True, True, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 48, None, 5120): {"pro": 15},
    ('sm_100a', True, True, True, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 64, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 5120): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, True, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_100a', True, True, False, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "block_n": 224},
    ('sm_100a', True, True, False, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True},
    ('sm_100a', True, True, False, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 39, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_100a', True, True, False, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True, "group_m": 8},
    ('sm_100a', True, True, False, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 32},
    ('sm_100a', True, True, False, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True},
    ('sm_100a', True, True, False, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_100a', True, True, True, False, False, 4096, None, 1024): {"slots": 1, "epi": 'tma', "pro": 15, "cgrp": True},
    ('sm_100a', True, True, True, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_100a', True, True, True, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_100a', True, True, True, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 26, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_100a', True, True, True, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True, "group_m": 8},
    ('sm_100a', True, True, True, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_100a', True, True, True, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_100a', True, True, True, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    # ---- sm_100a: the round-2 rules - the rows of the shapes tranche whose kernel-level round 2 selected another
    #      program or launch parameter, measured on GB200 at T 16231 / 16384 against cuBLAS (rows not listed keep the
    #      tranche's rule above) ----
    ('sm_100a', True, True, False, False, False, 4096, None, 1024): {"block_n": 224, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True},
    # ---- sm_103a: the shapes tranche - forward / input gradient / weight gradient rows at K 4096 - 17408 and N 32 - 17408,
    #      measured with this package's planner against cuBLAS on GB300 at T 16172 - 16384 (see the test table
    #      MEASURED_SHAPE_ROWS); a row whose measured choice equals the planner's default carries no rule ----
    ('sm_103a', False, False, False, False, False, 32, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 32, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 48, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 48, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 64, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 64, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 4096, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 5120, None): {"cta1": True, "hints": ('evict_first', 'none')},
    ('sm_103a', False, False, False, False, False, 128, 8192, None): {"pro": 15},
    ('sm_103a', False, False, False, False, False, 1024, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 1024, 5120, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 2048, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', False, False, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, False, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 8},
    ('sm_103a', False, False, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 16384, 2048, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, False, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, True, False, False, False, 2048, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 4096, 512, None): {"form": 'w8rs', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "lw": True, "promo": 'l2_256b', "mg": True, "sk": False},
    ('sm_103a', False, True, False, False, False, 4096, 1024, None): {"pd": 2, "pro": 15},
    ('sm_103a', False, True, False, False, False, 4096, 4096, None): {"htail": True, "cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 4096, 12288, None): {"cta_rows": 256, "ovl": True, "htail": True, "cgrp": True, "group_m": 16},
    ('sm_103a', False, True, False, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "htail": True, "lw": True},
    ('sm_103a', False, True, False, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "cta1": True, "pd": 3},
    ('sm_103a', False, True, False, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2, "lw": True, "group_m": 32},
    ('sm_103a', False, True, False, False, False, 5120, 512, None): {"form": 'w8r', "epi": 'tma', "slots": 1, "pro": 15, "pd": 2, "group_m": 12, "promo": 'l2_256b', "mg": True, "sk": False, "sw": 25},
    ('sm_103a', False, True, False, False, False, 5120, 1024, None): {"pd": 2, "pro": 15, "sw": 25},
    ('sm_103a', False, True, False, False, False, 5120, 5120, None): {"cgrp": True, "group_m": 12, "pro": 15},
    ('sm_103a', False, True, False, False, False, 5120, 17408, None): {"cta_rows": 256, "ovr": True, "htail": True},
    # the kv_a input gradient (6144 -> 576): the inherited sm_100a knobs restated so that this row keeps its program
    # where the eight-warp form yields and when it takes the whole-tile tile resolve (an sm_103a rule stops the parent
    # lookup); the raster group of 12 row tiles is the row's measured launch parameter
    ('sm_103a', False, True, False, False, False, 6144, 576, None): {"form": 'w8rs', "group_m": 12, "slots": 1, "epi": 'tma', "pd": 2, "pro": 15, "mg": True, "sk": False},
    ('sm_103a', False, True, False, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, False, False, False, 12288, 4096, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, False, False, False, 17408, 5120, None): {"cgrp": True, "group_m": 16, "pro": 15},
    ('sm_103a', False, True, True, False, False, 4096, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 4096, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 32, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 48, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 64, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 5120, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', False, True, True, False, False, 8192, 128, None): {"epi": 'tma', "slots": 2, "pro": 15, "pd": 2},
    ('sm_103a', True, True, False, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 32, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 48, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 64, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, True, False, 128, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, False, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 32, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 32, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 48, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 48, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 64, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 64, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 128, None, 4096): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, True, True, False, 128, None, 5120): {"pro": 7, "sk_sync": True, "sk_sync_m": 16},
    ('sm_103a', True, True, True, True, False, 128, None, 8192): {"cta_rows": 64, "pro": 15},
    ('sm_103a', True, True, False, False, False, 4096, None, 1024): {"block_n": 224, "pro": 15, "hints": ('none', 'evict_first'), "cgrp": True},
    ('sm_103a', True, True, False, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "block_n": 224},
    ('sm_103a', True, True, False, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True},
    ('sm_103a', True, True, False, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 39, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, False, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, False, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 32},
    ('sm_103a', True, True, False, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_103a', True, True, False, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_exact": 16, "sk_first": True},
    ('sm_103a', True, True, False, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 2},
    ('sm_103a', True, True, False, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
    ('sm_103a', True, True, True, False, False, 4096, None, 1024): {"block_n": 224, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15},
    ('sm_103a', True, True, True, False, False, 4096, None, 4096): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_103a', True, True, True, False, False, 4096, None, 12288): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_103a', True, True, True, False, False, 5120, None, 512): {"block_n": 192, "b_swz": 64, "epi": 'reg', "f32_v8": True, "store_ef": True, "sk_sync": True, "sk_slab": 2, "sk_sync_m": 26, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, True, False, False, 5120, None, 1024): {"block_n": 160, "b_swz": 64, "cta_rows": 256, "epi": 'reg', "f32_v8": True, "store_ef": True, "pro": 15, "hints": ('none', 'evict_first')},
    ('sm_103a', True, True, True, False, False, 5120, None, 5120): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3, "sk_sync_m": 32},
    ('sm_103a', True, True, True, False, False, 6144, None, 12288): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_103a', True, True, True, False, False, 12288, None, 4096): {"cta_rows": 256, "pro": 7, "sk_parts": 4},
    ('sm_103a', True, True, True, False, False, 12288, None, 6144): {"cta_rows": 256, "pro": 7, "sk_sync": True, "sk_slab": 3},
    ('sm_103a', True, True, True, True, False, 512, None, 4096): {"cta_rows": 64, "pro": 15, "hints": ('evict_first', 'none')},
}
# fmt: on

# Per-architecture defaults under every row rule of that architecture (round 33, Cake unit 2d): a knob measured as a
# win on every pair-program row of the architecture is set once here, and a row rule overrides it (``row_rule`` merges
# exactly as the Cake launcher does).  ``pro`` 7 = the pair prologue / cleanup sync form (no peer-startup rendezvous
# before the collective tcgen05.alloc, relaxed arrive on the post-init cluster sync, the MMA warp as allocator); the
# single-CTA programs (``cta1`` rows) take no ``pro`` (``instance_key`` refuses it): the planner zeroes it there like
# the launcher.  The sm_100a / sm_107a values are byte-identical to Cake ``ARCH_RULE_DEFAULTS``; sm_103a (compute
# capability 10.3) takes the same pair form: without it a pair template plans the round-27 form (``pro`` 0) there, which
# no measured sm_103a rule names.
ARCH_RULE_DEFAULTS: dict[str, dict] = {
    "sm_107a": {"pro": 7},
    "sm_100a": {"pro": 7},
    "sm_103a": {"pro": 7},
}


# Rule inheritance: an architecture whose own rule table covers only the rows its lanes measured plans every other
# row with the parent's rule - the same instance-key knobs, so the same program, which the registry fallback keeps
# only where the architecture registers it.  sm_103a (GB300) registers every sm_100a program and its lanes measured
# the shapes tranche alone; its Cake rows ran the architecture defaults (the generic programs) before, slower than
# the sm_100a programs the GB300 benchmark planned through the sm_100a table.  A rule the architecture measured itself
# always wins (either weight-gradient orientation); sm_100a and sm_107a inherit nothing.
RULE_PARENT: dict[str, str] = {"sm_103a": "sm_100a"}


def _measured_row_rule(
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
    """The row's own ``ROW_RULES`` entry without the architecture defaults: the exact (N, K, M) rule, else the
    ragged-M (N, K, None) rule, else the ragged-K (N, None, M) rule, else the parent architecture's entry
    (``RULE_PARENT``: sm_103a reads the sm_100a table for the rows its lanes did not measure), else empty (the lookup
    of Cake ``row_rule``; FlashInfer-only split so the registry fallback can drop the MEASURED rule alone, see
    ``plan_dense_projection_gemm``)."""
    ident = (arch, bool(a_mn), bool(b_mn), bool(out_f32), bool(out_t), bool(batched))
    for tail in (
        (int(N), int(K), int(M)),
        (int(N), int(K), None),
        (int(N), None, int(M)),
    ):
        rule = ROW_RULES.get(ident + tail)
        if rule is not None:
            return dict(rule)
    parent = RULE_PARENT.get(arch)
    if parent is not None:
        return _measured_row_rule(parent, a_mn, b_mn, out_f32, out_t, batched, N, K, M)
    return {}


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
    ragged-M (N, K, None) rule, else the ragged-K (N, None, M) rule, else the parent architecture's rule
    (``RULE_PARENT``), else empty (the class defaults apply) - each merged over the architecture's
    ``ARCH_RULE_DEFAULTS`` (round 33: a row rule overrides an architecture default).  [Cake ``row_rule``]"""
    merged = dict(ARCH_RULE_DEFAULTS.get(arch, {}))
    merged.update(
        _measured_row_rule(arch, a_mn, b_mn, out_f32, out_t, batched, N, K, M)
    )
    return merged


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


def wave_working_set_cgrp(
    m_tiles: int, n_tiles: int, k_len: int, group_n: int, pairs: int, elt_bytes: int = 2
) -> int:
    """Round 21 (Cake W3): bytes of A and B panels touched by one wave of CTA pairs under the column-grouped
    raster (``cgrp``): a group of ``group_n`` column panels (K x 256) is swept by consecutive pairs of one pair
    row, so a wave covers ``cols`` column panels and ``rows`` pair-row panels (256 x K) of one batch entry.
    [Cake ``wave_working_set_cgrp`` L2515-L2528]"""
    m_pairs = max(1, m_tiles // CTA_GROUP)
    g = max(1, min(group_n, n_tiles))
    band = g * m_pairs  # pair tiles per column group
    if band >= pairs:
        cols = g
        rows = min(m_pairs, _ceil_div(pairs, g))
    else:
        cols = min(n_tiles, g * _ceil_div(pairs, band))
        rows = m_pairs
    return (rows + cols) * 256 * k_len * elt_bytes


def default_hints_cgrp(
    a_mn: bool,
    b_mn: bool,
    m_tiles: int,
    n_tiles: int,
    k_len: int,
    group_n: int,
    pairs: int,
    l2_bytes: int,
) -> tuple[str, str]:
    """``default_hints`` for the column-grouped raster (round 21, Cake W3): the same policy over the column
    raster's wave working set.  [Cake ``default_hints_cgrp`` L2531-L2537]"""
    if wave_working_set_cgrp(m_tiles, n_tiles, k_len, group_n, pairs) <= l2_bytes:
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
    * ``sk="auto"`` (the launcher default): a two-way split of a single partial wave -- the
      tail's ``tail * k_blocks`` K steps, linearised, cut into units of ``ceil(k_blocks / 2)``
      steps.  On an even ``k_blocks`` unit ``u`` is tile ``u // 2``, K half ``u % 2``; on an odd
      one a unit may cross into the next tile.  The units are the ``ceil(tail * k_blocks /
      ceil(k_blocks / 2))`` that hold a K step (one fewer than ``2 * tail`` when ``k_blocks`` is
      odd and ``2 * tail > k_blocks``): a unit without a K step never finishes in the programs
      that hand off the work in an item's last segment (``pd`` 2, ``sk_slab`` 2, ``stt``).  Only
      when the whole problem is one partial wave whose halves fit the pairs (``pair_tiles <=
      pairs`` and ``2 * tail <= pairs``) and K has at least ``2 * SK_MIN_ITERS`` steps;
      multi-wave problems keep their tail data-parallel.
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
        return pair_tiles - tail, tail, _ceil_div(tail * k_blocks, half), half
    total = tail * k_blocks
    units = min(pairs, total // SK_MIN_ITERS, max_units or pairs)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    iters_per_unit = _ceil_div(total, units)
    units = _ceil_div(total, iters_per_unit)
    if units <= tail:
        return pair_tiles, 0, 0, k_blocks
    return pair_tiles - tail, tail, units, iters_per_unit


def check_stream_k_plan(plan: "GemmPlan") -> None:
    """Refuse a stream-K plan outside the program's launch contract (the Cake kernel's ``requires``) before it
    launches.  The programs that hand off the work inside an item's last segment (``pd`` 2, ``sk_slab`` 2,
    ``stt``) never finish an item that holds no K step: the linearised plan needs ``iters_per_unit >= 1`` and
    ``(sk_units - 1) * iters_per_unit < sk_iters``, the synchronised one at most ``tail_tiles`` collectors.  Every
    synchronised plan needs a collector for the K steps ``[s, k_blocks)`` of its tail tiles.  A plan without
    stream-K units holds every condition."""
    if not plan.sk_units:
        return
    handoff = plan.pd == 2 or plan.sk_slab == 2 or plan.stt
    if plan.sk_sync:
        collectors = plan.sk_collectors
        if collectors < 1 or (handoff and collectors > plan.tail_tiles):
            raise ValueError(
                f"dense_projection_gemm: {plan.template} needs 1 <= collectors"
                f"{' <= tail_tiles' if handoff else ''} (got {collectors} collectors, {plan.tail_tiles} tail tiles)"
            )
    elif handoff and not plan.sk_exact:
        if (
            plan.iters_per_unit < 1
            or (plan.sk_units - 1) * plan.iters_per_unit >= plan.sk_iters
        ):
            raise ValueError(
                f"dense_projection_gemm: {plan.template} needs every stream-K unit to hold a K step (got "
                f"{plan.sk_units} units of {plan.iters_per_unit} over {plan.sk_iters} K steps)"
            )


def sk_sync_plan(
    pair_tiles: int,
    k_blocks: int,
    pairs: int,
    s_override: Optional[int] = None,
    margin: int = 0,
) -> Optional[tuple[int, int, int, int, int]]:
    """``(num_full, tail_tiles, sk_units, s, collectors)`` of the synchronised stream-K plan (round 18, Cake W2) - pure
    integer arithmetic mirrored from the Cake launcher's ``sk_sync_plan``.  The tail tiles (all of them when there are
    fewer tiles than pairs) each get one MAIN unit over K steps [0, s) and ``collectors`` = min(pairs - tail, tail)
    COLLECTOR units own the leftovers [s, k_blocks) of tail tiles c, c + collectors, c + 2 collectors, ... (one segment
    per leftover, at most q = ceil(tail / collectors) per collector).  With one atomic segment per leftover the makespan
    is max(s, q (k_blocks - s)), balanced at s = ceil(q k_blocks / (q + 1)); ``margin`` m (K steps, rule key
    ``sk_sync_m``) lengthens the mains to s = ceil((q k_blocks + m) / (q + 1)) so the collectors finish m steps early;
    ``s_override`` replaces s (sweeps only).  ``sk_units`` = tail + collectors.  ``None`` when not admissible: no tail,
    no collector, or a part under ``SK_MIN_ITERS`` (the row keeps its other policy)."""
    tail = pair_tiles % pairs if pair_tiles > pairs else pair_tiles
    if tail == 0:
        return None
    collectors = min(pairs - tail, tail)
    if collectors < 1:
        return None
    q = _ceil_div(tail, collectors)
    s = (
        _ceil_div(q * k_blocks + int(margin), q + 1)
        if s_override is None
        else int(s_override)
    )
    if s < SK_MIN_ITERS or k_blocks - s < SK_MIN_ITERS:
        return None
    return pair_tiles - tail, tail, tail + collectors, s, collectors


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
    """The host plan of one dense projection GEMM launch (device-independent except ``sm_pairs``: the concurrent
    work items of the device - CTA pairs, or SMs under the single-CTA form ``cta1``)."""

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
    # ... and the plan landed on the nearest generated program of the row's layout / output class (another TMA L2
    # policy, the single-CTA form, another tile family, epilogue path or BLOCK_N, or the knobs a registered symbol
    # decodes to) because no generated program serves the row's default tail-T instance either
    knob_fallback: bool = False
    # round-13 tall-family knobs of the planned instance (instance_key fields 15 / 16 / 17)
    park: bool = False
    ovl: bool = False
    htail: bool = False
    # ... and whether the half-height tail PLAN is in effect: the ``_ht`` program's tail tiles run as 2 x tail whole-K
    # half items only when there is a tail and its half items fit the device's pairs (``0 < 2 * tail <= pairs``);
    # otherwise that program runs the row's plain / stream-K plan - the fit selects the plan, never the program a
    # rule or a caller named  [Cake launcher]
    htail_plan: bool = False
    # round-13 W3 knobs: MN-major B panel width (instance_key field 18) and the exact p-way tail split (field 19 is
    # the flag; ``sk_exact`` here is the part count p, 0 = off)
    b_swz: int = 128
    sk_exact: int = 0
    # round 13 (Cake W3, L38b): batch-entry raster of the batched MLA rows (instance_key field 20; 0 = row-major)
    batch_group: int = 0
    # round 13 (Cake W4): fp32 v8 register stores carry L1::no_allocate.L2::evict_first (instance_key field 21)
    store_ef: bool = False
    # round 15 (Cake W3): L2 eviction policy of the TMA-store epilogue's bulk tensor stores (instance_key field 22)
    store_hint: str = "none"
    # round 15 (Cake W1): pipelined TMEM drain form (field 23) and suspend-time hint in ns (field 24) of the planned instance
    pd: int = 0
    sh: int = 0
    # round 18 (Cake W2): synchronised stream-K tail of the planned instance (instance_key field 25 = the 26th field,
    # ``_sks`` symbols): ``tail_tiles`` main units over K steps [0, iters_per_unit) + ``sk_units - tail_tiles``
    # collector units over the leftovers, two partials and one slab per tail tile, deterministic two-addend fixup
    sk_sync: bool = False
    # round 19 (Cake W2): the slab path of that fixup (instance_key field 26 = the 27th field, ``_sb{n}`` symbols): 0 off,
    # 1 early arrival, 2 + the bulk slab read through the dead mainloop stages, 3 + the fp32 output as the slab; always 0
    # without ``sk_sync`` (the same synchronised plan arithmetic: ``ws_f32_elems`` keeps one slab per tail tile)
    sk_slab: int = 0
    # round 19 (Cake W3): the single-CTA form (instance_key field 27 = the 28th field, ``_c1`` symbols): one
    # 128 x BLOCK_N tile per CTA with cluster dims (1, 1, 1), so ``m_tiles`` is not rounded to pairs, ``pair_tiles``
    # counts CTA tiles, ``sm_pairs`` holds the SM count and ``grid`` = ``num_cluster_tiles`` CTAs
    cta1: bool = False
    # round 21 (Cake W3): the column-grouped raster (instance_key field 28 = the 29th field, ``_cg`` symbols):
    # ``group_m`` counts COLUMN tiles per raster group (any count >= 1; the pair invariant is unchanged), consecutive
    # pairs sweep a group's columns of one pair row before the next pair row, so a group's B panels stay L2-resident
    # and A streams once per column group; the plan arithmetic (tiles, units, tail policy, grid) is the pair form's
    cgrp: bool = False
    # round 24 (the round-20 Cake W3 knob ported in round 23 by W4): the overlapped tall epilogue released by ROW HALF
    # (instance_key field 29 = the 30th and LAST field, ``_or`` symbols, after ``_cg``): the eight epilogue warps drain
    # TMEM buffer 0 and release it to the next tile's half-0 MMAs while buffer 1 drains; the B stage keeps the plain
    # tall layout (one N = 256 MMA per row half per K step).  Tall 256-column tiles only; it replaces ``ovl`` on a row
    # (never both), and the plan arithmetic (tiles, units, tail policy, grid, hint gate) is the tall pair form's
    ovr: bool = False
    # round 27 (Cake lane 7, W-3 variant A1): the transposed stmatrix / TMA-store staging of the bf16 transposed register
    # epilogue (instance_key field 30 = the 31st and LAST field, ``_stt`` symbols, directly after ``_sks``): the last work
    # item's warp slices go out through one bulk tensor store per warp from the dead mainloop stage ring (``OUT16`` = the
    # real output view), every other slice through the register stores; the plan arithmetic is the pair form's
    stt: bool = False
    # round 33 (Cake unit 2d / 2e): the pair program's prologue / cleanup sync form (instance_key field 31 = the 32nd
    # and LAST field, ``_pro{n}`` symbols after ``_or``): a bitmask (1 no startup rendezvous before the collective TMEM
    # alloc, 2 relaxed arrive on the post-init cluster sync, 4 the MMA warp allocates TMEM, 8 CTA-local exit) from the
    # merged row rule (``ARCH_RULE_DEFAULTS`` 7 under every rule of sm_100a / sm_107a; a row rule may pin another
    # value), always 0 on the single-CTA form; it changes the program only - no launch argument, no plan arithmetic
    pro: int = 0
    # the round-2 shapes tranche: the tail-first order of the exact split (instance_key field 33, ``_skf``; the plan
    # arithmetic is ``sk_exact``'s) and the load warp leaving the pair prelude early (field 34, ``_lw``); programs only,
    # no launch argument
    sk_first: bool = False
    lw: bool = False
    # the whole-tile tile resolve (instance_key field 35, ``_mg``): the load warp and the MMA warp walk whole tiles with
    # the raster quotients resolved through corrected fp32 reciprocals; a program knob of plans without stream-K units
    # (``sk_units`` 0, the kernel's launch contract) - no launch argument, no plan arithmetic
    mg: bool = False
    # the paced output stores (instance_key field 36, ``_sw<n>``): the per-warp step count by which each epilogue warp of
    # the bf16 TMA-store pd-2 drain delays its bulk tensor store issue - a program knob, no launch argument, no plan
    # arithmetic (0 = off)
    sw: int = 0
    # the eight-warp CTA form (instance_key field 37, ``_w8r`` / ``_w8rs``; "" = the ten-warp pipeline): the same GEMM
    # through four epilogue warps, a loader, a scheduler (``w8r``) or the loader's elected lane in its place (``w8rs``)
    # and the MMA warp, whole kn bf16 TMA-store tiles only; a program knob that adds the raster's four fast divisors to
    # the launch (``_k1_values``) and changes no plan arithmetic
    form: str = ""
    # FlashInfer-only: the rule's exact-wave BLOCK_N (224) yielded to the default tile because the device's CTA pairs do
    # not fit its tiles in whole waves and the default tile needs fewer waves (``exact_waves``)
    wave_fallback: bool = False

    @property
    def num_cluster_tiles(self) -> int:
        return self.num_full + self.sk_units

    @property
    def wave_working_set_bytes(self) -> int:
        """Operand bytes one wave of CTA pairs touches (the hint gate compares it with ``l2_bytes``): the column
        raster's working set under ``cgrp`` (``wave_working_set_cgrp``), the row raster's otherwise."""
        return (wave_working_set_cgrp if self.cgrp else wave_working_set)(
            self.m_tiles, self.n_tiles, self.K, self.group_m, self.sm_pairs
        )

    @property
    def grid(self) -> tuple[int, int, int]:
        """CTAs of the launch: ``num_cluster_tiles`` pairs x 2, or one CTA per work item under ``cta1``."""
        return (self.num_cluster_tiles * (1 if self.cta1 else CTA_GROUP), 1, 1)

    @property
    def ws_f32_elems(self) -> int:
        """fp32 elements of the stream-K partial slabs (unit + tail-tile slabs); 0 without stream-K and
        under the half-height tail plan (its items are whole-K tiles: no partial slabs, no fixup)."""
        if not self.sk_units or self.htail_plan:
            return 0
        if self.sk_sync:
            # one slab per tail tile (the first arriver of a tile stores its partial, the second adds it)  [Cake launcher]
            return self.tail_tiles * 2 * self.cta_rows * self.block_n
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
        """The launcher's linearised stream-K span; under the half-tail plan every half item is one whole-K segment; the
        exact split (``_skx`` instances) passes the part count p in this slot."""
        if self.sk_exact:
            return int(self.sk_exact)
        if self.sk_sync:
            # the ``_sks`` instance reads the main-unit count n_t here (s in ``iters_per_unit``; collectors =
            # num_cluster_tiles - num_full - n_t)  [Cake launcher]
            return int(self.tail_tiles)
        return (self.sk_units if self.htail_plan else self.tail_tiles) * self.k_blocks

    @property
    def sk_collectors(self) -> int:
        """Collector units of the synchronised stream-K plan (0 for every other plan)."""
        return self.sk_units - self.tail_tiles if self.sk_sync else 0

    @property
    def tma_out(self) -> bool:
        # round 27: the stt program stores its last item's slices through OUT16 (the real view) and every other slice
        # through ``out``  [Cake launcher]
        return self.epi == "tma" or self.stt


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
    store_hint: Optional[str] = None,
    pd: Optional[int] = None,
    sh: Optional[int] = None,
    sk_sync: Optional[bool] = None,
    sk_sync_m: Optional[int] = None,
    sk_slab: Optional[int] = None,
    cta1: Optional[bool] = None,
    cgrp: Optional[bool] = None,
    ovr: Optional[bool] = None,
    stt: Optional[bool] = None,
    pro: Optional[int] = None,
    sk_first: Optional[bool] = None,
    lw: Optional[bool] = None,
    mg: Optional[bool] = None,
    sw: Optional[int] = None,
    form: Optional[str] = None,
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
    Round 19: ``cta1`` (Cake W3) selects the single-CTA form - the tile count, the concurrent work items
    (``sm_units``: SMs instead of CTA pairs) and the grid follow the CTA group, and the form carries no tail
    policy; ``sk_slab`` (Cake W2) selects the slab path of a synchronised stream-K plan (the same plan arithmetic),
    with the launcher's yield rules for a rule-derived value.  Round 21: ``cgrp`` (Cake W3) selects the
    column-grouped raster program (``_cg``): ``group_m`` counts column tiles per raster group (any count >= 1) and
    the hint gate takes the column raster's wave working set (``default_hints_cgrp``); the plan arithmetic is unchanged.
    Round 24: ``ovr`` (the round-20 Cake W3 knob ported in round 23 by W4) selects the tall epilogue released by row
    half (``_or``, instance-key field 30): a rule-derived ``ovr`` yields to a caller-forced ``ovl``, a caller-forced
    ``ovr`` replaces a rule-derived ``ovl`` (forcing both raises in ``instance_key``), it is dropped at BLOCK_N != 256,
    and it leaves the raster, the hint gate and the plan arithmetic unchanged.  Round 27: ``stt`` (Cake lane 7) selects the
    transposed stmatrix / TMA-store staging of the bf16 transposed register epilogue (``_stt``, instance-key field 31):
    the row's rule value is taken as is and ``instance_key`` admits or refuses it; the plan arithmetic is unchanged and
    the launch hands the real output view to ``OUT16`` next to the register-store alias.  Round 33: ``pro`` (Cake
    unit 2d / 2e) selects the pair program's prologue / cleanup sync form (``_pro{n}``, instance-key field 32 = LAST):
    read from the merged rule (``row_rule`` lays ``ARCH_RULE_DEFAULTS`` - 7 on sm_100a / sm_107a - under the row's
    rule) exactly as the launcher does and zeroed on the single-CTA programs (``instance_key`` refuses ``pro`` with
    ``cta1``); a caller-forced tile family drops the rule's ``pro`` with the rule's other family-bound knobs, a
    caller-forced ``pro`` is admitted or refused by ``instance_key``; no launch argument and no plan arithmetic change.
    The round-2 shapes tranche: ``sk_first`` (``_skf``) is the tail-first order of the row's exact split - taken from the
    rule with its ``sk_exact`` and dropped with it where the split is not admitted; ``lw`` (``_lw``) is the row's
    load-warp prelude form - taken from the rule and dropped where the planned program cannot carry it (the single-CTA
    form, a ``pro`` without bit 4, a caller-forced tile family); a caller-forced value of either is admitted or refused by
    ``instance_key``.  Neither changes a launch argument or the plan arithmetic.  ``mg`` (``_mg``) is the row's
    whole-tile tile resolve - taken from the rule (the three sm_103a input-gradient rules that carry it also pin
    ``sk: False``) and dropped where the planned program cannot carry it: a plan with stream-K units or over the
    kernel's cluster-tile bound (a caller ``sk`` / ``sk_max_units`` / ``sk_parts`` / exact or synchronised split), a
    caller-forced raster or tail form the program excludes (``batch_group`` / ``cgrp`` / ``htail`` / ``sk_first`` /
    ``stt``), the single-CTA form or a caller-forced tile family; a caller-forced ``mg`` is admitted or refused by
    ``instance_key`` and refused on a plan with stream-K units (the kernel's launch contract: ``sk_units`` is 0 on
    every ``_mg`` program).  It changes neither a launch argument nor the plan arithmetic.  ``sw`` (``_sw<n>``) is the
    row's paced output stores - taken from the rule (the two sm_103a input-gradient rules that carry it, 5120 -> 512
    and 5120 -> 1024, name bf16 TMA-store pd-2 rows) and dropped where the planned program is not the bf16 TMA-store
    pd-2 drain (a caller-forced register epilogue or ``slots`` 0, another ``pd``, the fp32 output) or is the
    single-CTA form; a caller-forced ``sw`` is admitted or refused by ``instance_key`` (the kernel's refusal: a step
    count >= 0 on that drain only).  It changes neither a launch argument nor the plan arithmetic.  ``form``
    (``_w8r`` / ``_w8rs``) is the row's eight-warp CTA form - taken from the rule (the three sm_103a input-gradient
    rules that carry it, 4096 -> 512, 5120 -> 512 and the kv_a 6144 -> 576, name whole-tile kn bf16 TMA-store rows)
    and dropped where the plan is not whole-tile or a caller forces a pipeline, tile, output, epilogue, tail or raster
    knob (the row's program above the form runs the plan asked for); where it survives, the rule's pair-program knobs
    (``pd`` / ``pro`` / ``lw`` / ``mg`` / ``sw`` / ``stages`` / ``hints`` / ``pf`` / ``sh`` / ``store_hint`` /
    ``store_ef`` / ``quad_store`` / ``f32_v8``) are cleared before the key: the eight-warp programs carry none of them.
    A caller-forced ``form`` is admitted or refused by ``instance_key`` and refused on a plan with stream-K units.  It
    changes no plan arithmetic; the launch adds the raster's four fast divisors (``fd_tiles_l`` / ``fd_tiles_g`` /
    ``fd_group`` / ``fd_last``), which the ten-warp programs' argument plans do not name.
    [Cake ``dense_projection_gemm`` L1234-L1358]

    One FlashInfer-only deviation: the Cake host applies ``swap_small_m`` unconditionally because it compiles
    the swapped instance on demand, while this package can only launch the generated programs registered for
    ``arch``.  When the plan resolves to an unregistered instance (a batched forward / input gradient with a
    tiny T that engages the swap, or a per-row rule whose template was only generated for the contract's T),
    the plan is redone without the swap and / or without the rule (``plan.swap_fallback`` /
    ``plan.rule_fallback``) so the call still runs; the result is identical, only the tile walk differs.  When
    that is still unregistered, the nearest generated program of the row's layout / output class is planned
    instead (``plan.knob_fallback``): the hand-ordered knob variants first (``_FALLBACK_KNOB_VARIANTS``: the plain
    TMA L2 policy, the single-CTA form, the 64-row pair family, the tall tile, the register epilogue, BLOCK_N 128,
    each in the three ``pro`` forms), then every generated program of the class, nearest symbol first, with the
    knobs its symbol decodes to (``_registry_variants``); a caller-forced knob wins over a variant's.  Only a class
    without any generated program on ``arch`` keeps an unregistered plan (the launch binding raises
    ``NotImplementedError`` naming it).  ``_fallback=False`` returns the pure mirror (unit tests of the planner).

    A second FlashInfer-only deviation, the exact-wave gate (``plan.wave_fallback``): a rule's BLOCK_N 224
    (``EXACT_WAVE_BLOCK_N``) was measured for the whole-wave fit of its tiles on the 76-pair devices
    (:func:`exact_waves`); on a device whose pairs do not fit them in whole waves and where the default tile needs fewer
    waves (74 pairs of a 148-SM part: the 4096 x 4096 weight gradient's 152 tiles are 2.05 waves against 1.73, the
    4096 x 1024 row's 76 tiles 1.03 against 0.86) the rule's BLOCK_N yields to the default tile; the rule's other knobs
    stay, a caller-forced ``block_n`` is never touched, and the plan is the launcher's wherever the fit is exact or both
    tiles need the same waves (152 and 160 SMs)."""
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
        store_hint=store_hint,
        pd=pd,
        sh=sh,
        sk_sync=sk_sync,
        sk_sync_m=sk_sync_m,
        sk_slab=sk_slab,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        sk_first=sk_first,
        lw=lw,
        mg=mg,
        sw=sw,
        form=form,
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
    # Round 33: ``row_rule`` lays the row's MEASURED rule over the architecture's ``ARCH_RULE_DEFAULTS`` (the
    # launcher's merge).  The FlashInfer-only rule fallback (``_use_rules=False``) drops the measured rule alone and
    # keeps the architecture default: it is what a row without a rule plans with, and the generated pair programs of
    # the architecture carry the default's ``_pro{n}`` form (the pro-0 pair programs are not registered since round 33).
    rule_dropped = (
        {}
        if _use_rules
        else _measured_row_rule(
            arch, a_mn, b_mn, out.dtype == torch.float32, transposed_out, L > 1, N, K, M
        )
    )
    rule = (
        row_rule(
            arch, a_mn, b_mn, out.dtype == torch.float32, transposed_out, L > 1, N, K, M
        )
        if _use_rules
        else dict(ARCH_RULE_DEFAULTS.get(arch, {}))
    )
    if cta_rows is None:
        cta_rows = rule.get("cta_rows", default_cta_rows(M, N, K))
    elif int(cta_rows) != rule.get("cta_rows", default_cta_rows(M, N, K)):
        # A caller-forced tile family (sweeps, the registrations of the tall / 64-row families): the rule's knobs
        # that belong to its own family (tile width, stage count, staging slots, epilogue mode) do not carry over
        # (a 160-column rule of the 256-row family is not a valid 64-row tile); the synchronised stream-K plan (round 18) is
        # admitted per family too (its unit arithmetic counts the family's pair tiles), so it does not carry over either.
        # [Cake launcher]
        rule = {
            k: v
            for k, v in rule.items()
            if k
            not in (
                "block_n",
                "stages",
                "slots",
                "epi",
                "ovl",
                "htail",
                "b_swz",
                "pd",
                "sk_sync",
                "sk_sync_m",
                "sk_slab",
                "cta1",
                "ovr",
                "stt",
                "pro",
                "lw",
                "mg",
                "sw",
            )
        }
    if cta1 is None:
        # round 19 (Cake W3): the single-CTA form of the 128-row family (no pair handoff)  [Cake launcher]
        cta1 = bool(rule.get("cta1", False))
    cta1 = bool(cta1)
    if cgrp is None:
        # round 21 (Cake W3): column-grouped raster (group_m = column tiles per group; B-panel L2 residency)  [Cake launcher]
        cgrp = rule.get("cgrp", False)
    cgrp = bool(cgrp)
    # CTAs per work item: the launch grid, the tile count and the scheduler units follow it  [Cake launcher]
    cg = 1 if cta1 else CTA_GROUP
    if cta1 and (
        sk is True
        or sk_exact
        or sk_sync
        or htail
        or sk_parts
        or sk_max_units
        or (batch_group and L > 1)
    ):
        # the single-CTA form carries no tail policy (its one-K-step rows never split) and no pair-only knob; a caller
        # forcing one with cta1 asked for two incompatible forms  [Cake launcher]
        raise ValueError(
            f"dense_projection_gemm: cta1 excludes sk / sk_exact / sk_sync / htail / sk_parts / sk_max_units / batch_group "
            f"(got sk={sk}, sk_exact={sk_exact}, sk_sync={sk_sync}, htail={htail}, sk_parts={sk_parts}, "
            f"sk_max_units={sk_max_units}, batch_group={batch_group})"
        )
    if cta1:
        # the row rule's tail policy / raster knob does not carry over to the single-CTA form  [Cake launcher]
        sk, sk_exact, sk_sync, htail, batch_group = False, None, False, False, 0
    if cta1 != bool(rule.get("cta1", False)):
        # a caller-forced switch between the pair and the single-CTA form: the rule's stage count belongs to its own
        # form (a 48 KiB cta1 stage at BLOCK_N 256 fits 3-4 deep where the 32 KiB pair stage fits 5-7); the instance
        # default of the forced form applies  [Cake launcher]
        rule = {k: v for k, v in rule.items() if k != "stages"}
    wave_fallback = False
    if block_n is None and rule.get("block_n") == EXACT_WAVE_BLOCK_N:
        # FlashInfer-only exact-wave gate: the rule's 224-column tile fills the measuring device's CTA pairs in whole
        # waves (``exact_waves``); where the running device's pairs do not, and the default tile needs fewer waves, the
        # narrower tile would cost a nearly empty extra wave, so the rule's BLOCK_N yields (its other knobs stay)
        m_tiles_g = _ceil_div(M, int(cta_rows))
        m_tiles_g += m_tiles_g % cg
        units_g = sm_units(sm_count, cg)
        tiles_rule = L * (m_tiles_g // cg) * _ceil_div(N, EXACT_WAVE_BLOCK_N)
        tiles_default = L * (m_tiles_g // cg) * _ceil_div(N, default_block_n(N, b_mn))
        if not exact_waves(tiles_rule, units_g) and _ceil_div(
            tiles_rule, units_g
        ) > _ceil_div(tiles_default, units_g):
            rule = {k: v for k, v in rule.items() if k != "block_n"}
            wave_fallback = True
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
    rule_ovl = ovl is None
    if ovl is None:
        ovl = rule.get("ovl", False)
    if ovr is None:
        # round 20 (Cake W3) / round 24: the overlapped tall epilogue released by row half, from the row's rule  [Cake launcher]
        ovr = rule.get("ovr", False)
        if ovr and ovl and not rule_ovl:
            # round 24: a caller-forced column-chunk release replaces the row's row-half release (the registrations of
            # the ``_ov`` programs at rows whose rule now carries ovr)  [Cake launcher]
            ovr = False
    elif ovr and rule_ovl:
        # a caller-forced row-half release replaces the row's column-chunk release (sweeps / registrations of the
        # ``_or`` programs); forcing both raises (instance_key)  [Cake launcher]
        ovl = False
    if ovr and int(block_n) != 256:
        # like ovl: built for the 256-column tall tile  [Cake launcher]
        ovr = False
    rule_stt = stt is None
    if stt is None:
        # round 27 (Cake lane 7, W-3 A1): the transposed stmatrix / TMA-store staging of the bf16 transposed register
        # epilogue, from the row's rule; a rule-derived value yields to a caller-forced form below, a caller-forced one is admitted or refused by instance_key  [Cake launcher]
        stt = rule.get("stt", False)
    stt = bool(stt)
    if pro is None:
        # round 33 (Cake unit 2d): pair prologue / cleanup sync form (bitmask; 0 = the round-27 form); the rule value is
        # the row's override of ARCH_RULE_DEFAULTS (7 on sm_107a / sm_100a); the single-CTA programs take none
        # (instance_key refuses pro with cta1)  [Cake launcher]
        pro = 0 if cta1 else rule.get("pro", 0)
    pro = int(pro)
    rule_lw = lw is None
    if lw is None:
        # the round-2 shapes tranche: the row's load-warp prelude form (the load warp leaves the pair prelude before the
        # post-allocation CTA rendezvous), from the rule
        lw = rule.get("lw", False)
    lw = bool(lw)
    if lw and rule_lw and (cta1 or not (pro & 4)):
        # the form belongs to the pair programs whose MMA warp allocates TMEM (``pro`` bit 4): a caller-forced single-CTA
        # form or another ``pro`` on the row keeps that form without it (a caller-forced ``lw`` is admitted or refused by
        # ``instance_key``)
        lw = False
    rule_mg = mg is None
    if mg is None:
        # the whole-tile tile resolve of the row's rule (the ``_mg`` programs): the load warp and the MMA warp walk
        # whole tiles with the raster quotients resolved through corrected fp32 reciprocals; the rules that carry it
        # pin ``sk: False`` (read below), and the whole-tile guard before ``instance_key`` drops it from a plan that
        # still carries stream-K units
        mg = rule.get("mg", False)
    mg = bool(mg)
    if mg and rule_mg and cta1:
        # the measured ``_mg`` programs are pair programs: a caller-forced single-CTA form keeps the row's single-CTA
        # program without it (a caller-forced ``mg`` is admitted or refused by ``instance_key``)
        mg = False
    rule_sw = sw is None
    if sw is None:
        # the paced output stores of the row's rule (the ``_sw<n>`` programs): each epilogue warp of the bf16
        # TMA-store pd-2 drain delays its bulk tensor store issue by a per-warp number of dependent multiply-add
        # steps; the drain guard before ``instance_key`` drops it from a plan that is not that drain
        sw = rule.get("sw", 0)
    sw = int(sw)
    if sw and rule_sw and cta1:
        # the measured ``_sw<n>`` programs are pair programs: a caller-forced single-CTA form keeps the row's
        # single-CTA program without it (a caller-forced ``sw`` is admitted or refused by ``instance_key``)
        sw = 0
    rule_form = form is None
    if form is None:
        # the eight-warp CTA form of the row's rule (the ``_w8r`` / ``_w8rs`` programs): the whole-tile guard before
        # ``instance_key`` drops it from a plan that is not whole-tile or that a caller shaped with a tile, output,
        # epilogue, tail, raster or pipeline knob
        form = rule.get("form", "")
    form = str(form or "")
    if form not in ("", *W8_FORMS):
        raise ValueError(
            f"dense_projection_gemm: form must be one of {W8_FORMS} (the eight-warp CTA form) or '', got {form!r}"
        )
    if htail is None:
        htail = rule.get("htail", False)
    if b_swz is None:
        b_swz = rule.get("b_swz", 128)
    if sk_exact is None:
        sk_exact = rule.get("sk_exact")
    rule_sk_first = sk_first is None
    if sk_first is None:
        # the round-2 shapes tranche: the tail-first order of the row's exact split, from the rule
        sk_first = rule.get("sk_first", False)
    sk_first = bool(sk_first)
    rule_sk_sync = sk_sync is None
    if sk_sync is None:
        # round 18 (Cake W2): synchronised stream-K tail (main units + collectors) of the row's rule
        sk_sync = rule.get("sk_sync", False)
    elif sk_sync:
        # a caller-forced synchronised plan replaces the row's other tail policies (the registrations of the ``_sks``
        # programs)  [Cake launcher]
        sk_exact, htail = None, False
    if sk_sync_m is None:
        # round 18 (Cake W2): main-unit margin in K steps (the collectors finish early; rule key ``sk_sync_m``)
        sk_sync_m = int(rule.get("sk_sync_m", 0))
    rule_sk_slab = sk_slab is None
    if sk_slab is None:
        # round 19 (Cake W2): slab path of the synchronised fixup (0 off, 1 early arrival, 2 + bulk slab read, 3 + fp32
        # output as the slab)  [Cake launcher]
        sk_slab = int(rule.get("sk_slab", 0))
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
    if store_hint is None:
        # round 15 (Cake W3): L2 eviction policy of the TMA-store epilogue's stores  [Cake launcher]
        store_hint = rule.get("store_hint", "none")
    if pd is None:
        pd = rule.get(
            "pd", 0
        )  # round 15 (Cake W1): pipelined TMEM drain of the 128-row family  [Cake launcher]
    if sh is None:
        sh = rule.get("sh", 0)
    if ovl and int(block_n) != 256:
        # the overlapped epilogue is built for the 256-column tall tile; a narrower tall tile keeps the serialized
        # tall epilogue  [Cake launcher]
        ovl = False
    if (
        sk_sync
        and rule_sk_sync
        and (ovl or ovr or park or int(pd) or htail or sk_exact)
    ):
        # The row's synchronised plan needs the single-pass or the serialised tall epilogue and is the row's only tail
        # policy (instance_key); a caller that forces another form on the row gets that form with the plain plan - the
        # rule's tail policy does not carry over (the Cake launcher also yields to its ``a_mcast`` attribution probe, which
        # the planner does not expose).  A caller-forced ``sk_sync`` keeps raising on such a conflict.  [Cake launcher]
        sk_sync = False
    m_tiles = _ceil_div(M, cta_rows)
    m_tiles += m_tiles % cg
    n_tiles = _ceil_div(N, block_n)
    k_blocks = _ceil_div(K, BLOCK_K)
    # work items = tiles per CTA group (pair tiles, or CTA tiles under cta1)  [Cake launcher]
    pair_tiles = L * (m_tiles // cg) * n_tiles
    # concurrent work items: CTA pairs, or SMs under cta1 (``_sm_units``)  [Cake launcher]
    units = sm_units(sm_count, cg)
    # Round 13 (Cake W3): a ``sk_exact`` rule / kwarg is the row's tail policy (exact p-way K split of every tail
    # tile, one unit per part); it pre-empts ``sk_parts`` and ``htail`` exactly as the Cake launcher orders them.
    exact_plan = (
        sk_exact_plan(pair_tiles, k_blocks, units, int(sk_exact))
        if (sk_exact and sk == "auto" and sk_max_units is None)
        else None
    )
    # Round 18 (Cake W2): synchronised stream-K - n_t main units (K steps [0, s) of every tail tile) + C collector units
    # (the leftovers [s, k_blocks) of tail tiles c, c + C, ...), two partials and one slab per tail tile; admitted when
    # the plan is (``sk_sync_plan``, host arithmetic); it disables ``sk_parts`` and ``htail`` and yields to ``sk_exact``.
    sync_plan = (
        sk_sync_plan(pair_tiles, k_blocks, units, None, int(sk_sync_m))
        if (sk_sync and sk == "auto" and sk_max_units is None and exact_plan is None)
        else None
    )
    if sk_first and rule_sk_first and exact_plan is None:
        # the row's tail-first order is its exact split's: where the split is not admitted (no tail, a part under
        # ``SK_MIN_ITERS`` steps, a caller ``sk`` / ``sk_max_units`` / single-CTA form) the plain plan runs without it;
        # a caller-forced ``sk_first`` is admitted or refused by ``instance_key``
        sk_first = False
    parts = rule.get("sk_parts") if sk_parts is None else int(sk_parts)
    if sync_plan is not None:
        parts = None
    if sk == "auto" and sk_max_units is None and parts and exact_plan is None:
        # Measured per-row p-way split of the tail wave (Cake round 4, L20); a caller ``sk_parts`` wins over the
        # rule's.  [Cake L1296-L1301]
        sk, sk_max_units = sk_parts_plan(pair_tiles, k_blocks, units, int(parts))
    # Round 13 (Cake W1): deterministic half-height tail wave of the tall family - the tail tiles become 2 x tail
    # standard-geometry items of full K (no partial slabs, no fixup).  The knob is the row's (its rule or the caller:
    # the ``_ht`` program, ``instance_key`` field 17); the half-tail PLAN applies only when there is a tail and its
    # half items fit the CTA pairs, otherwise the ``_ht`` program runs the plain plan / stream-K policy of the row (the
    # round-2 shapes tranche measured the K 5120 thin input gradients that way: 1280 tiles on 76 pairs, tail 64).  The
    # fit never changes the program a rule or a caller named.  [Cake launcher]
    htail = (
        bool(htail)
        and int(cta_rows) in (128, 256)
        and exact_plan is None
        and sync_plan is None
    )
    if htail and int(cta_rows) == 128:
        # round 16 (Cake W2): the half-height tail wave on the standard family needs the 256-column row-major tile without the
        # pipelined drain / prefetch; the planner never emits box_rows or diag probes, so the launcher's remaining conditions
        # hold by construction  [Cake launcher]
        pf_eff = pf if pf is not None else rule.get("pf", default_pf(M, N, K))
        htail = (
            int(block_n) == 256
            and not transposed_out
            and int(pd) == 0
            and not int(pf_eff)
        )
    htail_plan = False
    if htail:
        tail_h = pair_tiles % units if pair_tiles > units else pair_tiles
        htail_plan = bool(tail_h) and 2 * tail_h <= units
    if exact_plan is not None:
        num_full, tail_tiles, sk_units, iters_per_unit = exact_plan
    elif sync_plan is not None:
        # the ``_sks`` instance reads s in iters_per_unit and n_t in sk_iters; collectors = num_cluster_tiles - num_full - n_t
        num_full, tail_tiles, sk_units, iters_per_unit, _collectors = sync_plan
    elif htail_plan:
        num_full, tail_tiles, sk_units, iters_per_unit = (
            pair_tiles - tail_h,
            tail_h,
            2 * tail_h,
            k_blocks,
        )
    elif htail:
        # The ``_ht`` program reads every unit past ``num_full`` as a half-height item of full K (the half-tail
        # mapping above); it cannot run linearised stream-K units.  Where the half items do not fit, the program
        # runs the plain whole-tile plan, whatever ``sk`` the row or the caller asked for.
        num_full, tail_tiles, sk_units, iters_per_unit = pair_tiles, 0, 0, k_blocks
    else:
        num_full, tail_tiles, sk_units, iters_per_unit = stream_k_plan(
            pair_tiles, k_blocks, units, sk, sk_max_units
        )
    if tail_tiles * 16 > SK_DUMMY_BASE:
        raise ValueError(
            f"dense_projection_gemm: {tail_tiles} stream-K tail tiles exceed the {SK_DUMMY_BASE // 16} slice-counter budget"
        )
    out_f32 = out.dtype == torch.float32
    mode = epi_mode(out_f32, transposed_out, K, epi, block_n, cta_rows)
    nslots = epi_slots(mode, out_f32, block_n, K, slots, cta_rows, transposed_out)
    if (
        sync_plan is not None
        and rule_sk_slab
        and int(sk_slab) == 3
        and (not out_f32 or (mode == "reg" and not transposed_out))
    ):
        # Round 19 (Cake W2): the row's fp32-output-as-slab form needs the fp32 TMA-store or transposed register
        # epilogue; a caller that forces another epilogue on the row (the registrations / sweeps of the row-major f32_v8
        # programs) keeps the synchronised plan with the bulk slab read.  A caller-forced ``sk_slab`` keeps raising on
        # such a conflict (``instance_key``).  [Cake launcher]
        sk_slab = 2
    if sync_plan is not None and rule_sk_slab and int(sk_slab) == 2:
        stages_eff = (
            default_stages(nslots, cta_rows, block_n, b_mn, b_swz)
            if stages is None
            else int(stages)
        )
        if EPI_WARPS * epi_cols(block_n, cta_rows) * 128 > stages_eff * (
            cta_rows * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz)
        ):
            # the row's bulk slab read does not fit a caller-forced narrower pipeline: the plain synchronised program
            # [Cake launcher]
            sk_slab = 0
    if stt and rule_stt:
        # Round 27 (Cake lane 7, W-3 A1): the row's staging form belongs to the bf16 transposed single-pass register epilogue of
        # the 128-row pair family and must stage its eight warp slots in the dead stage ring inside the opt-in (instance_key's
        # admission).  A caller that forces another form on the row (the e2e registrations / sweeps of the _t_tma1, _sks_sb2, pd,
        # cta1 and tall programs at the kv_a weight gradient row, a forced narrower pipeline) keeps that form WITHOUT the
        # staging: the rule's stt does not carry over, like the rule's sk_sync / sk_slab above.  A caller-forced stt keeps
        # raising on such a conflict (instance_key).  [Cake launcher]
        cols_stt = epi_cols(block_n, cta_rows)
        stage_b = int(cta_rows) * BLOCK_K * 2 + b_stage_bytes(b_mn, block_n, b_swz, cg)
        stages_stt = (
            default_stages(nslots, cta_rows, block_n, b_mn, b_swz, cg, True)
            if stages is None
            else int(stages)
        )
        slab_eff = int(sk_slab) if sync_plan is not None else 0
        if (
            out_f32
            or not transposed_out
            or mode != "reg"
            or int(cta_rows) != 128
            or cta1
            or ovl
            or ovr
            or park
            or htail
            or int(pd)
            or cols_stt % 32
            or cols_stt > 256
            or slab_eff not in (0, 1)
            or EPI_WARPS * cols_stt * 64 > stages_stt * stage_b
            or stages_stt * stage_b
            + staging_bytes(nslots)
            + stt_bytes(True)
            + WORK_STAGES * 16
            > SMEM_OPT_IN
        ):
            stt = False
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
            "group_m", default_group_m(a_mn, b_mn, m_tiles, pair_tiles, units)
        )
    group_m = int(group_m)
    if group_m < 1 or (cg == 2 and not cgrp and (group_m < 2 or group_m % 2)):
        raise ValueError(
            f"dense_projection_gemm: group_m must be an even number >= 2 (CTA pairs are adjacent row tiles; any count >= 1 under cta1, or under cgrp where it counts column tiles), got {group_m}"
        )
    if hints is None:
        # round 21 (Cake W3): the hint gate over the column raster's wave working set under cgrp  [Cake launcher]
        hints = rule.get(
            "hints",
            (default_hints_cgrp if cgrp else default_hints)(
                a_mn, b_mn, m_tiles, n_tiles, K, group_m, units, int(l2_bytes)
            ),
        )
    # The whole-tile guard of ``mg``: the ``_mg`` programs require a plan without stream-K units and fewer than 2**21
    # cluster tiles (the kernel's ``requires``), and exclude the raster / tail forms ``instance_key`` refuses with it.
    # A rule's ``mg`` yields where the plan or a caller-forced form breaks that (a caller ``sk`` / ``sk_max_units`` /
    # ``sk_parts``, an exact or synchronised split, ``htail``, ``cgrp``, ``stt``, ``sk_first``, a batch raster): the
    # plain program runs the plan asked for.  A caller-forced ``mg`` is refused on such a plan (the forms by
    # ``instance_key``, the stream-K units and the tile bound here).
    whole_tiles = sk_units == 0 and (num_full + sk_units) * CTA_GROUP < 2**22
    if mg and rule_mg:
        if (
            not whole_tiles
            or htail
            or cgrp
            or stt
            or sk_first
            or exact_plan is not None
            or sync_plan is not None
            or batch_group
        ):
            mg = False
    elif mg and not whole_tiles:
        raise ValueError(
            f"dense_projection_gemm: mg (whole-tile tile resolve) needs a whole-tile plan under 2**21 cluster tiles "
            f"(got {sk_units} stream-K units over {num_full + sk_units} cluster tiles)"
        )
    # The drain guard of ``sw``: the ``_sw<n>`` programs pace the bf16 TMA-store pd-2 drain and the kernel refuses the
    # knob elsewhere.  A rule's ``sw`` yields where the planned program is not that drain (a caller-forced register
    # epilogue or ``slots`` 0, another ``pd``, the fp32 output): the row's unpaced program runs the plan asked for.  A
    # caller-forced ``sw`` is admitted or refused by ``instance_key``.
    if sw and rule_sw and (mode != "tma" or nslots == 0 or int(pd) != 2 or out_f32):
        sw = 0
    # The whole-tile guard of ``form``: the eight-warp programs run whole kn bf16 128 x 256 pair tiles through the
    # one-slot TMA-store epilogue under the kernel's cluster-tile bound, with none of the pair programs' pipeline knobs.
    # A rule's ``form`` yields where the plan or a caller-forced knob breaks that (stream-K units, the single-CTA form,
    # another tile, output class or epilogue, a tail or raster form, a forced pipeline knob - ``_W8_FIXED``): the row's
    # program above the form runs the plan asked for.  Where it survives, the rule-derived pair-program knobs are
    # cleared: the eight-warp programs carry none of them.  A caller-forced ``form`` reaches ``instance_key`` with the
    # caller's knobs and is refused there (the stream-K units here).
    if form and rule_form:
        if (
            not whole_tiles
            or cta1
            or out_f32
            or transposed_out
            or a_mn
            or not b_mn
            or int(block_n) != 256
            or int(cta_rows) != 128
            or mode != "tma"
            or nslots != 1
            or htail
            or cgrp
            or stt
            or sk_first
            or exact_plan is not None
            or sync_plan is not None
            or batch_group
            or any(_caller_knobs.get(k) is not None for k in _W8_FIXED)
        ):
            form = ""
        else:
            lw, mg, sw, pd, pro, stages = False, False, 0, 0, 0, None
            hints, pf, sh = _PLAIN_HINTS, 0, 0
            store_hint, store_ef, quad_store, f32_v8 = "none", False, False, False
    elif form and not whole_tiles:
        raise ValueError(
            f"dense_projection_gemm: form (the eight-warp CTA form) needs a whole-tile kn bf16 plan on 128 x 256 pair "
            f"tiles under 2**21 cluster tiles (got {sk_units} stream-K units over {num_full + sk_units} cluster tiles)"
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
        store_hint=str(store_hint),
        pd=int(pd),
        sh=int(sh),
        sk_sync=sync_plan is not None,
        sk_slab=int(sk_slab) if sync_plan is not None else 0,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        sk_first=sk_first,
        lw=lw,
        mg=mg,
        sw=sw,
        form=form,
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
        sm_pairs=units,
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
        htail_plan=htail_plan,
        b_swz=int(key[18]),
        sk_exact=int(sk_exact) if exact_plan is not None else 0,
        batch_group=int(key[20]),
        store_ef=bool(key[21]),
        store_hint=str(key[22]),
        pd=int(key[23]),
        sh=int(key[24]),
        sk_sync=bool(key[25]),
        sk_slab=int(key[26]),
        cta1=bool(key[27]),
        cgrp=bool(key[28]),
        ovr=bool(key[29]),
        stt=bool(key[30]),
        pro=int(key[31]),
        sk_first=bool(key[32]),
        lw=bool(key[33]),
        mg=bool(key[34]),
        sw=int(key[35]),
        form=str(key[36]),
        wave_fallback=wave_fallback,
    )
    check_stream_k_plan(plan)
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
        # computes the same GEMM; only the tile walk and the store path differ): the hand-ordered variants first
        # (``_FALLBACK_KNOB_VARIANTS``), then every generated program of the plan's layout / output class, nearest
        # by symbol first (``_registry_variants``), so a class the registry serves never fails to plan
        tried: set[tuple] = set()
        for allow_swap in (True, False) if swapped else (False,):
            try:
                base = plan_dense_projection_gemm(
                    A,
                    B,
                    out,
                    sm_count=sm_count,
                    l2_bytes=l2_bytes,
                    _fallback=False,
                    _allow_swap=allow_swap,
                    _use_rules=False,
                    **_caller_knobs,
                )
            except ValueError:
                continue
            variants = (
                *_FALLBACK_KNOB_VARIANTS,
                *_registry_variants(
                    base[0].template, arch, L > 1, sk_exact_parts=rule.get("sk_exact")
                ),
            )
            for knobs in variants:
                merged = _merge_fallback_knobs(_caller_knobs, knobs)
                frozen = _frozen_knobs(merged)
                if merged == _caller_knobs or frozen in tried:
                    continue
                tried.add(frozen)
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


# Knob variants the registry fallback tries, nearest first (see ``plan_dense_projection_gemm``): the plain TMA L2
# policy (the same tiles and plan; only the eviction hints of the loads differ), the single-CTA 128-row form, the
# 64-row pair family, the tall tile, the register epilogue (the canonical-T templates), then BLOCK_N = 128 with the same
# forms.  The TMA L2 promotion and the raster group width are launch parameters since round 11: they select no program
# and are not variants.
# Round 33: a row planned at a T outside the export contract takes the architecture default ``pro`` form (7), but
# the generated pair programs of a row may carry only the row's adopted form (15 = + the CTA-local exit) or the
# pinned round-27 form (0).  ``pro`` changes the pair prologue / cleanup sync only - never the GEMM, the tile walk or
# the store path - so a ``pro`` twin is the nearest registered program: the two other forms of the plan's own tile
# walk come first, then every tile variant in the three forms.  A single-CTA variant takes no pro (``instance_key``
# refuses it): ``_merge_fallback_knobs`` clears it with the tail knobs, so its three pro forms collapse into one try.
# A caller-forced tile family (``cta_rows`` away from the rule's) drops the rule's ``pro`` with the other family knobs
# [Cake launcher], so the forced-family variants must name the form: the architecture default 7 comes first (the form
# of most generated pair programs), then 15 and 0.
_FALLBACK_PRO_FORMS = (dict(pro=7), dict(pro=15), dict(pro=0))
_PLAIN_HINTS = ("none", "none")
_FALLBACK_TILE_VARIANTS = (
    dict(hints=_PLAIN_HINTS),
    dict(cta1=True),
    dict(cta1=True, hints=_PLAIN_HINTS),
    dict(cta_rows=64),
    dict(cta_rows=64, hints=_PLAIN_HINTS),
    dict(cta_rows=256),
    dict(cta_rows=256, hints=_PLAIN_HINTS),
    dict(epi="reg"),
    dict(epi="reg", f32_v8=True),
    dict(epi="reg", cta_rows=256),
    dict(block_n=128),
    dict(block_n=128, epi="reg"),
    dict(block_n=128, hints=_PLAIN_HINTS),
    dict(block_n=128, cta1=True),
    dict(block_n=128, cta_rows=64),
    dict(block_n=128, cta_rows=64, hints=_PLAIN_HINTS),
)
_FALLBACK_KNOB_VARIANTS = _FALLBACK_PRO_FORMS + tuple(
    dict(tile, **pro)
    for tile in _FALLBACK_TILE_VARIANTS
    for pro in ({}, *_FALLBACK_PRO_FORMS)
)
# The knobs the single-CTA form excludes (the planner raises on a forced one): a ``cta1`` variant clears them so the
# ``_c1`` programs stay reachable from a caller that forced a tail policy or a pair-only form on the row.
_CTA1_CLEARED = (
    "sk_exact",
    "sk_sync",
    "sk_sync_m",
    "sk_slab",
    "htail",
    "sk_parts",
    "sk_max_units",
    "batch_group",
    "pro",
    "sk_first",
    "lw",
)
# The caller knobs the eight-warp CTA form yields to (``plan_dense_projection_gemm``): a value a caller forces for any
# of them shapes the pipeline, tile, epilogue, tail or raster the form cannot carry, so the row's ten-warp program runs
# the plan asked for.  Every name is a planner keyword whose default is None (a default call never yields).
_W8_FIXED = (
    "stages",
    "epi",
    "slots",
    "pf",
    "hints",
    "f32_v8",
    "quad_store",
    "park",
    "ovl",
    "htail",
    "b_swz",
    "sk_exact",
    "batch_group",
    "store_ef",
    "store_hint",
    "pd",
    "sh",
    "sk_sync",
    "sk_sync_m",
    "sk_slab",
    "cgrp",
    "ovr",
    "stt",
    "pro",
    "sk_first",
    "lw",
    "mg",
    "sw",
    "sk_max_units",
    "sk_parts",
)
_TEMPLATE_PREFIX = "dense_proj_gemm_"
_HINT_OF_LETTER = {"e": "evict_first", "l": "evict_last", "n": "none"}
_STORE_HINT_OF_LETTER = {
    "f": "evict_first",
    "l": "evict_last",
    "n": "evict_normal",
    "d": "default",
}
_HINT_TOKEN = re.compile(r"h[eln][eln]")


def _merge_fallback_knobs(
    caller: dict[str, Any], knobs: dict[str, Any]
) -> dict[str, Any]:
    """The caller's knobs over a fallback variant's (a caller-forced knob wins, as the plan it replaces); a ``cta1``
    variant clears the knobs the single-CTA form excludes (``_CTA1_CLEARED``) and resets ``sk`` to the launcher's
    automatic policy, which the planner zeroes under ``cta1``."""
    merged = dict(caller)
    for name, value in knobs.items():
        if merged.get(name) is None:
            merged[name] = value
    if knobs.get("cta1"):
        for name in _CTA1_CLEARED:
            merged[name] = None
        merged["sk"] = "auto"
    return merged


def _frozen_knobs(knobs: dict[str, Any]) -> tuple:
    """A hashable identity of a knob set (sequence-valued knobs as tuples)."""
    return tuple(
        (name, tuple(value) if isinstance(value, (list, tuple)) else value)
        for name, value in sorted(knobs.items())
    )


def _template_tokens(template: str) -> Optional[tuple[str, ...]]:
    """The symbol tokens after the family prefix (``kk_n256_hen_pro7`` -> ``('kk', 'n256', 'hen', 'pro7')``);
    ``None`` outside the family."""
    if not template.startswith(_TEMPLATE_PREFIX):
        return None
    return tuple(template[len(_TEMPLATE_PREFIX) :].split("_"))


def _template_class(tokens: tuple[str, ...]) -> tuple[str, bool, bool]:
    """``(layout, fp32 output, transposed store)`` of a symbol: the tokens no knob can change."""
    return tokens[0], "f32" in tokens, "t" in tokens


def _template_form(tokens: tuple[str, ...]) -> str:
    """The eight-warp CTA form token of a symbol (``w8r`` / ``w8rs``), "" for the ten-warp programs: a class of its
    own for the registry fallback (a ten-warp row never falls onto an eight-warp program, nor the reverse)."""
    return next((token for token in tokens if token in W8_FORMS), "")


def _template_knobs(
    template: str, *, sk_exact_parts: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """The planner knobs that name ``template`` (the inverse of :func:`instance_symbol`).  Every knob the symbol can
    carry is set - an absent token to the value the symbol omits (plain hints, 128 rows, the register epilogue, no
    tail policy, no raster form, ``pro`` 0 on a pair program) - so a row of the template's class planned with them
    under ``_use_rules=False`` names the template unless the row cannot take a knob (the planner raises) or its
    arithmetic drops a tail policy (the plan names the template without it).  Launch parameters (promo, group_m) are
    not in the symbol and stay unset; ``_skx`` carries no part count, so ``sk_exact`` takes ``sk_exact_parts`` (the
    row's rule) or the kernel tests' 3 parts on 128-row / 2 on 64-row tiles (``_skf`` = its tail-first order,
    ``_lw`` = the load-warp prelude form, ``_mg`` = the whole-tile tile resolve, ``_sw<n>`` = the paced output stores,
    ``_w8r`` / ``_w8rs`` = the eight-warp CTA form); a single-CTA program leaves ``pro`` to
    the planner (``instance_key`` refuses a forced one).  ``None`` for a symbol outside the family or carrying a token
    the planner never emits (``_box<rows>``, diagnostics)."""
    tokens = _template_tokens(template)
    if tokens is None or len(tokens) < 2:
        return None
    knobs: dict[str, Any] = dict(
        block_n=None,
        cta_rows=128,
        b_swz=128,
        pf=0,
        hints=_PLAIN_HINTS,
        f32_v8=False,
        store_ef=False,
        quad_store=False,
        park=False,
        ovl=False,
        htail=False,
        epi="reg",
        slots=None,
        stages=None,
        sk_exact=None,
        sk_sync=False,
        stt=False,
        sk_slab=0,
        batch_group=0,
        store_hint="none",
        pd=0,
        sh=0,
        cta1=False,
        cgrp=False,
        ovr=False,
        pro=0,
        sk_first=False,
        lw=False,
        mg=False,
        sw=0,
        form="",
    )
    for token in tokens[1:]:
        if token in ("f32", "t"):
            continue  # the class
        if m := re.fullmatch(r"n(\d+)", token):
            knobs["block_n"] = int(m[1])
        elif token in ("m256", "m64"):
            knobs["cta_rows"] = int(token[1:])
        elif m := re.fullmatch(r"bz(\d+)", token):
            knobs["b_swz"] = int(m[1])
        elif m := re.fullmatch(r"pf(\d+)", token):
            knobs["pf"] = int(m[1])
        elif m := re.fullmatch(r"h([eln])([eln])", token):
            knobs["hints"] = (_HINT_OF_LETTER[m[1]], _HINT_OF_LETTER[m[2]])
        elif token == "v8":
            knobs["f32_v8"] = True
        elif token == "ef":
            knobs["store_ef"] = True
        elif token == "q":
            knobs["quad_store"] = True
        elif token == "pk":
            knobs["park"] = True
        elif token == "ov":
            knobs["ovl"] = True
        elif token == "ht":
            knobs["htail"] = True
        elif m := re.fullmatch(r"tma(\d+)", token):
            knobs["epi"], knobs["slots"] = "tma", int(m[1])
        elif m := re.fullmatch(r"s(\d+)", token):
            knobs["stages"] = int(m[1])
        elif token == "skx":
            knobs["sk_exact"] = int(
                sk_exact_parts or (2 if knobs["cta_rows"] == 64 else 3)
            )
        elif token == "skf":
            knobs["sk_first"] = True
        elif token == "sks":
            knobs["sk_sync"] = True
        elif token == "stt":
            knobs["stt"] = True
        elif m := re.fullmatch(r"sb(\d)", token):
            knobs["sk_slab"] = int(m[1])
        elif token == "bf":
            knobs["batch_group"] = 1
        elif m := re.fullmatch(r"bg(\d+)", token):
            knobs["batch_group"] = int(m[1])
        elif m := re.fullmatch(r"so([flnd])", token):
            knobs["store_hint"] = _STORE_HINT_OF_LETTER[m[1]]
        elif m := re.fullmatch(r"pd(\d+)", token):
            knobs["pd"] = int(m[1])
        elif m := re.fullmatch(r"sh(\d+)", token):
            knobs["sh"] = int(m[1])
        elif token == "c1":
            knobs["cta1"] = True
        elif token == "cg":
            knobs["cgrp"] = True
        elif token == "or":
            knobs["ovr"] = True
        elif token == "lw":
            knobs["lw"] = True
        elif token == "mg":
            knobs["mg"] = True
        elif m := re.fullmatch(r"sw(\d+)", token):
            knobs["sw"] = int(m[1])
        elif m := re.fullmatch(r"pro(\d+)", token):
            knobs["pro"] = int(m[1])
        elif token in W8_FORMS:
            knobs["form"] = token
        else:
            return None  # _box<rows> / a diagnostic: not a planner knob
    if knobs["block_n"] is None:
        return None
    if knobs["cta1"]:
        knobs["pro"] = None
    return knobs


def _template_distance(wanted: tuple[str, ...], candidate: tuple[str, ...]) -> float:
    """Tokens one symbol carries and the other does not; a hint token counts half (``_hen`` <-> plain is the
    nearest edit: the same tiles and plan)."""
    return sum(
        0.5 if _HINT_TOKEN.fullmatch(token) else 1.0
        for token in set(wanted) ^ set(candidate)
    )


def _registry_variants(
    template: str, arch: str, batched: bool, *, sk_exact_parts: Optional[int] = None
) -> tuple[dict[str, Any], ...]:
    """The knob sets of the generated programs of ``arch`` in the layout / output class and the CTA form of
    ``template``, nearest symbol first (:func:`_template_distance`, then the name); the batch-raster programs (``_bf``
    / ``_bg<n>``) only for a batched row (a batch of one plans no batch raster)."""
    wanted = _template_tokens(template)
    if wanted is None:
        return ()
    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for candidate in KERNELS.get(arch, {}):
        tokens = _template_tokens(candidate)
        if tokens is None or candidate == template:
            continue
        if _template_class(tokens) != _template_class(wanted):
            continue
        if bool(_template_form(tokens)) != bool(_template_form(wanted)):
            continue  # the eight-warp CTA form is a class of its own (its programs fall back onto each other only)
        knobs = _template_knobs(candidate, sk_exact_parts=sk_exact_parts)
        if knobs is None or (knobs["batch_group"] and not batched):
            continue
        ranked.append((_template_distance(wanted, tokens), candidate, knobs))
    ranked.sort(key=lambda item: item[:2])
    return tuple(knobs for _, _, knobs in ranked)


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


def _grid_fits_cluster(
    module_name: str, record: dict[str, Any], grid: tuple[int, int, int]
) -> None:
    cluster = record.get("launch", {}).get("cluster")
    if cluster and any(g % c for g, c in zip(grid, cluster, strict=True)):
        raise ValueError(
            f"{module_name}: grid {grid} is not a multiple of the cluster shape {tuple(cluster)} baked into the module"
        )


def _order_arguments(
    module_name: str, plan, values: dict[str, Any], grid: tuple[int, int, int]
) -> list:
    """``values`` in the order of the argument plan ``plan`` (``[kind, name]`` pairs); the ``grid`` entries come
    from ``grid``.  Fails closed: a keyword the kernel expects that the host does not provide raises ``KeyError``
    naming both sides."""
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = []
    for kind, name in plan:
        if kind == "grid":
            arguments.append(grid_values[name])
        elif name in values and values[name] is not None:
            arguments.append(values[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} ({kind}); "
                f"the host binding provides {sorted(k for k, v in values.items() if v is not None)}"
            )
    return arguments


def _bind(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> tuple[Callable[..., Any], tuple]:
    """``(entry, arguments)`` of ``module_name``: the loaded FFI entry and ``values`` ordered by the module's
    argument plan (``cake_jit.arg_plan_of``: a key into ``ARG_PLANS`` or an inline list)."""
    record = MODULES[module_name]
    _grid_fits_cluster(module_name, record, grid)
    arguments = _order_arguments(module_name, arg_plan_of(record), values, grid)
    module = load_cake_dense_projection_gemm_module(module_name, arch)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def bind_launch(
    module_name: str, values: dict[str, Any], grid: tuple[int, int, int], arch: str
) -> _Launch:
    """Order ``values`` by the generated argument plan of ``module_name`` and load its entry.

    Fails closed: a keyword the kernel expects that the host does not provide raises
    ``KeyError`` naming both sides.  The grid must be a multiple of the cluster shape baked
    into the module.
    """
    entry, arguments = _bind(module_name, values, grid, arch)
    return _Launch(module_name, entry, arguments, grid)


_FFI_DEVICES: dict[int, Any] = {}


_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _current_raw_stream(index: int) -> int:
    """torch's current stream on device ``index`` as a raw ``cudaStream_t`` value."""
    if _RAW_STREAM is not None:
        return int(_RAW_STREAM(index))
    return int(torch.cuda.current_stream(index).cuda_stream)


def _ffi_stream_context(index: int, raw: Optional[int] = None):
    """tvm-ffi environment-stream context for torch's current stream on device ``index`` (``raw`` when the
    caller already read it): the caller's downstream ops are stream-ordered after the kernel."""
    device = _FFI_DEVICES.get(index)
    if device is None:
        device = _FFI_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    if raw is None:
        raw = _current_raw_stream(index)
    return tvm_ffi.use_raw_stream(device, raw)


def _device_index(device: torch.device) -> int:
    return _cuda_index(device)


# Small tensors an instance never dereferences (the operands of the epilogue path it does not
# take), cached per device like the Cake launcher's: a 16-element buffer per pointer dtype and a
# ``[1, 32, 64]`` tensor covering both output boxes for the unused output descriptor.
_DUMMIES: dict[tuple, torch.Tensor] = {}


def _dummy_key(device: torch.device, dtype, tag: Optional[str] = None) -> tuple:
    """Per-device-ordinal key of a dummy (``cuda`` and ``cuda:0`` are one device; cheaper than ``str(device)``
    on the planning path)."""
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return (index, dtype, tag)


def _dummy(device: torch.device, dtype) -> torch.Tensor:
    key = _dummy_key(device, dtype)
    dummy = _DUMMIES.get(key)
    if dummy is None:
        dummy = _DUMMIES[key] = torch.zeros(16, device=device, dtype=dtype)
    return dummy


def _dummy_map(device: torch.device, dtype) -> torch.Tensor:
    """A ``[1, 256, 64]`` tensor for an output descriptor the instance never touches: it covers every output box (inner
    <= 64 elements, outer <= 256 rows - OUT_BOX_OUTER is EPI_COLS <= 256 on the stt programs (round 27), 32 / CH_T_COLS
    otherwise), so the dummy map encodes on every program.  [Cake ``_dummy_map``]"""
    key = _dummy_key(device, dtype, "map")
    dummy = _DUMMIES.get(key)
    if dummy is None:
        dummy = _DUMMIES[key] = torch.zeros((1, 256, 64), device=device, dtype=dtype)
    return dummy


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
# Convenience-path cache (shared by K1 and K2)
# ---------------------------------------------------------------------------

# The convenience entry points keep the planned launches of a process per kind (K1 / K2) in a least-recently-used
# cache of this many entries.  An entry holds its plan, the resolved module entry and an argument template whose
# per-call operands -- the caller's views and the workspace tensors -- are ``_Slot`` markers, never a caller's
# tensor: the operand views are rebound from the caller's tensors on every call.  The workspace of a plan (K1:
# the stream-K partial slabs and the slice counters; K2: the BF16x3 stack, its fp32 scratch and the split-K
# partials) is not owned by the entry but carved at launch time out of the workspace arena of the entry's
# (device index, raw stream) -- one growable buffer per kind shared by every entry of that device and stream
# (:class:`_Arena`) -- so the memory held is bounded by the largest requirement seen per kind, however many
# geometries (token counts) the cache has planned; an arena is released when its last entry is evicted and by
# :func:`clear_convenience_caches`.  The key holds the raw stream, so launches on different streams never share
# a workspace, and launches of one device and stream -- the only users of its arena -- are ordered by that
# stream.  Threads: planning and the arena bookkeeping (creation, growth, reference counts, insertion, eviction,
# ``clear_convenience_caches``) are serialised by the module lock ``_PLAN_LOCK`` (a miss re-checks the cache
# under it, so a key is planned once); the K2 hit path holds its arena's lock from the carve of the workspace
# views through the split-K reduction, so two threads issuing router launches of different geometries on one
# stream cannot interleave their split / kernel / reduction on the shared BF16x3 stack and partials; the K1 hit
# path and the cache lookup are lock-free (one self-contained kernel per call; a hit whose entry another thread
# evicts between the lookup and the recency refresh is a miss).  The caller still owns the stream: a stream is
# assumed to have one issuer at a time in the sense of CUDA stream semantics.  The bound is read on every
# insertion, so a process may set it after the import.  A CUDA Graph should capture a prepared launch, not the
# convenience path: a growth or an eviction after capture would replace or free the workspace the graph replays.
CONVENIENCE_CACHE_SIZE = 64


@dataclass(frozen=True)
class _Slot:
    """A per-call argument of a cached launch (a view of the caller's tensors), filled in at launch time."""

    name: str


def _view_key(t: torch.Tensor) -> tuple:
    """The geometry of a view that fixes its plan and its binding: shape, strides, dtype, storage offset and
    the 16-byte alignment residue of its first element (the planner's alignment checks); never its address.
    (``shape`` and ``stride()`` are tuples of ints already; this is on the hit path.)"""
    return (
        tuple(t.shape),
        t.stride(),
        t.dtype,
        t.storage_offset(),
        t.data_ptr() & 15,
    )


def convenience_key(
    device_index: int, stream: int, flags: tuple, *views: torch.Tensor
) -> tuple:
    """Cache key of a convenience-path launch: the device, the raw stream, the entry point's flags / knobs and
    the geometry of every view (:func:`_view_key`)."""
    return (
        int(device_index),
        int(stream),
        tuple(flags),
        *(_view_key(t) for t in views),
    )


def _cache_get(cache: "OrderedDict[tuple, Any]", key: tuple) -> Any:
    """The cached entry of ``key`` (refreshed as the most recent) or ``None``: a miss, or a hit whose entry
    another thread evicted between the lookup and the refresh (the caller plans the key again)."""
    entry = cache.get(key)
    if entry is not None:
        try:
            cache.move_to_end(key)
        except KeyError:
            return None
    return entry


def _cache_put(
    cache: "OrderedDict[tuple, Any]",
    key: tuple,
    entry: Any,
    bound: Optional[int] = None,
    on_evict: Optional[Callable[[Any], None]] = None,
) -> Any:
    """Insert ``entry`` as the most recent and evict the least recent entries beyond ``bound``
    (``CONVENIENCE_CACHE_SIZE`` at call time when ``None``); ``on_evict`` sees every entry that leaves the
    cache -- the evicted ones and an earlier entry of ``key`` that ``entry`` replaces."""
    if bound is None:
        bound = CONVENIENCE_CACHE_SIZE
    replaced = cache.get(key) if on_evict is not None else None
    cache[key] = entry
    cache.move_to_end(key)
    if replaced is not None and replaced is not entry:
        on_evict(replaced)
    while len(cache) > bound:
        _, evicted = cache.popitem(last=False)
        if on_evict is not None:
            on_evict(evicted)
    return entry


def _resolve_arguments(template: tuple, views: dict[str, torch.Tensor]) -> list:
    """The argument list of a cached launch: ``template`` with every ``_Slot`` replaced by the caller's view."""
    return [views[item.name] if type(item) is _Slot else item for item in template]


def _same_cuda_device(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor
) -> torch.device:
    device = out.device
    if (
        not (A.is_cuda and B.is_cuda and out.is_cuda)
        or A.device != device
        or B.device != device
    ):
        raise ValueError("Expected A, B and out on one CUDA device")
    return device


# ---------------------------------------------------------------------------
# Convenience-path workspace arena (one per device index and raw stream)
# ---------------------------------------------------------------------------

# The workspace kinds an arena carves: ``(dtype, zero-initialised)``.  The kinds are separate buffers -- a K1
# launch uses ``k1_ws`` + ``k1_counters``, a K2 launch ``k2_parts`` + ``k2_partials`` (+ ``k2_resid``, the fp32
# scratch of the torch-chain split, when the fused split program is not registered for the device) -- so no
# launch depends on the carving of another kind and the slice-counter zero contract has a buffer of its own.
_ARENA_KINDS: dict[str, tuple[torch.dtype, bool]] = {
    "k1_ws": (
        torch.float32,
        False,
    ),  # stream-K partial slabs (``GemmPlan.ws_f32_elems``)
    "k1_counters": (
        torch.uint32,
        True,
    ),  # stream-K slice counters (``GemmPlan.counters_alloc_u32``)
    "k2_parts": (
        torch.bfloat16,
        False,
    ),  # the BF16x3 stack of the split operand ``[3, *split_src.shape]``
    "k2_resid": (
        torch.float32,
        False,
    ),  # the fp32 scratch of the split ``split_src.shape``
    "k2_partials": (
        torch.float32,
        False,
    ),  # the split-K partials ``[splits, rows, cols]``
}
_ARENA_MIN_ELEMS = (
    16  # a pointer parameter the instance never dereferences still needs a buffer
)


@dataclass
class _Arena:
    """The workspace of every cached convenience-path launch of one (device index, raw stream): one buffer per
    kind of ``_ARENA_KINDS``, grown to the largest requirement a planned entry has asked for and never shrunk;
    an entry's workspace tensors are views of these buffers, resolved at launch time (:func:`_entry_views`).

    Why sharing one buffer per kind between every entry of the device and stream is safe:

    * every launch that reads or writes the arena is issued on this stream, and the hit path allocates nothing,
      so two uses of a buffer are always ordered by the stream;
    * growing replaces a buffer while an earlier launch on this stream may still be in flight on the old one.
      The old tensor stays referenced by the arguments of that launch until the FFI call has enqueued it, and
      once freed its block returns to PyTorch's caching allocator, which is stream-ordered: a block is handed
      out again only to a later allocation on the stream it was freed on (whose kernels run after the launch
      in flight), or after a ``cudaFree`` that synchronises the device.  The new buffer is allocated on this
      stream (the current stream of the calling thread is the arena's stream);
    * ``k1_counters``: the stream-K zero contract -- the kernel resets every counter it uses (the last arriver
      of a tail tile's slice writes zero after the fixup) and the lane-spread dummy counters above
      ``SK_DUMMY_BASE`` only ever receive a fetch-add of 0 -- holds for a shared buffer because the launches
      are serialised by the stream and each one leaves the whole buffer zero again (the buffer is
      zero-initialised when allocated, by a memset on the same stream).  This is the contract the Cake
      launcher's per-device counters rely on;
    * ``k2_parts`` / ``k2_resid``: the split rewrites the stack (and, under the torch chain, the scratch) in full
      before every launch (same stream);
    * ``k2_partials``: written by the kernel and consumed by ``torch.sum(..., out=out)`` right after the
      launch on the same stream; the next launch overwrites them only after that reduction has read them;
    * ``k1_ws``: the stream-K slabs carry no state between launches (the first arriver of a slice stores, the
      others add).

    ``version`` counts the growths: an entry memoises the views it carved against the version it read before
    carving and carves them again after a growth (:func:`_entry_views`); a growth also forgets every memo of
    the arena, so no cached entry keeps a replaced buffer alive.  ``refs`` counts the cached entries of the
    arena; the last eviction releases it (:func:`_release_entry`).

    Threads.  The stream-ordering argument above assumes the launches of one stream are issued in the order
    their host work happened, which one issuing thread per stream gives for free.  With several threads on one
    stream: ``require`` / ``refs`` / the registry are only touched under ``_PLAN_LOCK`` (planning is
    serialised; ``require`` is also monotone -- it keeps the larger of its own allocation and a concurrently
    stored buffer), and the K2 hit path holds ``lock`` from the carve of its workspace views through the
    split-K reduction (:func:`_launch_cached_router`), because a K2 call is three stream operations (the
    split, the kernel, the reduction) on the shared BF16x3 stack and partials, which two threads must not
    interleave.  A K1 call is one self-contained kernel (the slabs carry no state, the counters self-reset),
    so the K1 hit path takes no lock.  A memo stored by a thread that carved before a concurrent growth is
    stamped with the pre-growth version and is carved again on its next use.
    """

    device: torch.device
    key: tuple[int, int]  # (device index, raw stream) -- the key of ``_ARENAS``
    buffers: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    version: int = 0
    refs: int = 0
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def require(self, kind: str, numel: int) -> torch.Tensor:
        """The buffer of ``kind`` with at least ``numel`` elements, grown (replaced by a larger allocation on the
        current stream) when the present one is smaller; called when an entry is planned (under ``_PLAN_LOCK``),
        never on a hit.  Monotone: if a larger buffer was stored meanwhile, that one is kept."""
        buf = self.buffers.get(kind)
        numel = max(int(numel), _ARENA_MIN_ELEMS)
        if buf is None or buf.numel() < numel:
            dtype, zeroed = _ARENA_KINDS[kind]
            new = (torch.zeros if zeroed else torch.empty)(
                numel, device=self.device, dtype=dtype
            )
            buf = self.buffers.get(kind)  # re-read: keep the larger buffer
            if buf is None or buf.numel() < new.numel():
                buf = self.buffers[kind] = new
            self.version += 1
            _forget_arena_views(self)
        return buf

    def nbytes(self) -> int:
        """Bytes the arena holds (every kind)."""
        return sum(int(b.numel()) * b.element_size() for b in self.buffers.values())


class _CachedLaunch(Protocol):
    """What the arena bookkeeping needs of a cached convenience-path entry (K1 :class:`_CachedGemm`, K2
    :class:`_CachedRouter`): its arena, the memoised workspace views with the arena version they were carved
    at, and the carve itself."""

    arena: _Arena
    views: Optional[dict[str, torch.Tensor]]
    views_version: int

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]: ...


_ARENAS: dict[tuple[int, int], _Arena] = {}
# Serialises planning and every mutation of the arenas and the caches (``_plan_entry``, ``clear_convenience_caches``);
# the hit paths (``_cache_get``, ``_entry_views``) do not take it.
_PLAN_LOCK = threading.Lock()


def _arena_for(index: int, stream: int, device: torch.device) -> _Arena:
    """The arena of (device ``index``, raw ``stream``), created on first use (under ``_PLAN_LOCK``)."""
    key = (int(index), int(stream))
    arena = _ARENAS.get(key)
    if arena is None:
        arena = _ARENAS[key] = _Arena(device, key)
    return arena


def _release_entry(entry: _CachedLaunch) -> None:
    """``on_evict`` of the convenience caches: the arena of an evicted entry loses a reference and is released
    (its buffers freed, stream-ordered) when no cached entry uses it any more."""
    arena = entry.arena
    arena.refs -= 1
    if arena.refs <= 0 and _ARENAS.get(arena.key) is arena:
        del _ARENAS[arena.key]


def _plan_entry(
    cache: "OrderedDict[tuple, Any]",
    key: tuple,
    index: int,
    stream: int,
    device: torch.device,
    prepare: Callable[[_Arena], Any],
) -> Any:
    """Plan the entry of ``key`` with ``prepare(arena)`` -- which returns ``(entry, views)``: the entry and the
    per-call operand views its fail-closed check already built for this first call -- against the arena of
    (``index``, ``stream``), cache the entry (``_release_entry`` on eviction) and return both; all of it under
    ``_PLAN_LOCK``.  The cache is re-checked under the lock: when another thread planned ``key`` meanwhile its
    entry is returned with ``views = None`` (the caller rebuilds the views).  A failed plan leaves no arena
    behind that nothing references, and a shared arena it had grown is rolled back to its earlier buffers."""
    with _PLAN_LOCK:
        entry = _cache_get(cache, key)
        if entry is not None:
            return entry, None
        arena = _arena_for(index, stream, device)
        buffers, version = dict(arena.buffers), arena.version
        try:
            entry, views = prepare(arena)
        except BaseException:
            if arena.refs <= 0 and _ARENAS.get(arena.key) is arena:
                del _ARENAS[arena.key]
            elif arena.version != version:
                arena.buffers = buffers
                arena.version += 1
                _forget_arena_views(arena)
            raise
        _cache_put(cache, key, entry, on_evict=_release_entry)
        return entry, views


def _forget_arena_views(arena: _Arena) -> None:
    """Drop the memoised workspace views of every cached entry of ``arena`` (a buffer was replaced)."""
    entries: list[_CachedLaunch] = [*_K1_CACHE.values(), *_K2_CACHE.values()]
    for entry in entries:
        if entry.arena is arena:
            entry.views = None


def _numel(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def _carve(buf: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """The contiguous ``shape`` view over the first elements of the 1-D buffer ``buf`` (one strided view, no
    slice-then-reshape pair)."""
    strides = []
    stride = 1
    for s in reversed(shape):
        strides.append(stride)
        stride *= s
    return buf.as_strided(shape, tuple(reversed(strides)))


def _entry_views(entry: _CachedLaunch) -> dict[str, torch.Tensor]:
    """The workspace views of a cached entry, carved out of its arena's buffers by ``entry.carve(arena)``
    (K1: the slab and counter buffers themselves -- pointer parameters; K2: the ``[3, *split]`` parts, the
    ``split`` scratch of the torch-chain split and the ``[splits, rows, cols]`` partials) and memoised until the
    arena grows."""
    arena = entry.arena
    views = entry.views
    if views is None or entry.views_version != arena.version:
        version = (
            arena.version
        )  # read before carving: a growth that interleaves leaves the memo stale
        views = entry.carve(arena)
        entry.views_version = version
        entry.views = views
    return views


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


def _k1_plan(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    knobs: dict[str, Any],
) -> tuple[GemmPlan, torch.Tensor, torch.Tensor, torch.Tensor, str, str]:
    """Plan one K1 binding for its device: ``(plan, a_desc, b_desc, out3, arch, module_name)`` (the device facts
    come from the per-device cache, the module from the registry; see :func:`plan_dense_projection_gemm`)."""
    device = _same_cuda_device(A, B, out)
    facts = device_facts(device)
    arch = require_arch(device)
    plan, a_desc, b_desc, O3 = plan_dense_projection_gemm(
        A,
        B,
        out,
        sm_count=facts.sm_count,
        l2_bytes=facts.l2_bytes,
        transposed_out=transposed_out,
        arch=arch,
        **knobs,
    )
    return plan, a_desc, b_desc, O3, arch, select_module(arch, plan.template)


def _k1_workspace(
    plan: GemmPlan, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The stream-K partial slabs and the slice counters of a plan (the only allocations of the K1 backend).
    The counters are zero-initialised once: the kernel resets every counter it uses (the last arriver of a tail
    tile's slice writes zero after the fixup), so a buffer is zero again when its launch completes and is reused
    across launches without a memset, as the Cake launcher's per-device counters are.  Sized like the Cake host's
    ``_sk_counters`` so the lane-spread dummy counters above SK_DUMMY_BASE are addressable."""
    if plan.ws_f32_elems:
        ws = torch.empty(plan.ws_f32_elems, device=device, dtype=torch.float32)
    else:
        ws = _dummy(device, torch.float32)
    counters = torch.zeros(plan.counters_alloc_u32, device=device, dtype=torch.uint32)
    return ws, counters


def _out_strides(out3: torch.Tensor) -> tuple[int, int]:
    """``(ldo, out_l)`` of the batched output view: its row stride and its batch stride.  The kernel declares its
    tile grid against the output's extents, so it requires ``out_l >= rows * ldo`` for every batch size (rows = M, or
    N under the transposed store).  A batch of one never dereferences its batch stride: its ``out_l`` is the view's
    batch stride when that spans the matrix (a 2-D output unsqueezed to ``[1, rows, cols]``) and the matrix span
    ``rows * ldo`` otherwise (an expanded or ``as_strided`` batch axis).  A real batch passes its stride as is."""
    ldo = int(out3.stride(1))
    batch = int(out3.stride(0))
    span = int(out3.shape[1]) * ldo
    if int(out3.shape[0]) == 1 and batch < span:
        batch = span
    return ldo, batch


def _k1_values(
    plan: GemmPlan,
    device: torch.device,
    *,
    A: Any,
    B: Any,
    out3: Any,
    o_alias: Any,
    ws: Any,
    counters: Any,
    ldo: int,
    out_l: int,
) -> dict[str, Any]:
    """The keyword arguments of a K1 launch (:func:`bind_launch` orders them by the module's plan).  ``A`` /
    ``B`` / ``out3`` / ``o_alias`` are the operand views and ``ws`` / ``counters`` the workspace -- or
    :class:`_Slot` markers for a cached launch that rebinds the views on every call and resolves the workspace
    out of its arena; the plan decides which of the operands the instance dereferences."""
    dummy16, dummy32 = _dummy(device, torch.bfloat16), _dummy(device, torch.float32)
    tma_out, out_f32 = plan.tma_out, plan.out_f32
    return dict(
        A=A,
        B=B,
        OUT32=out3 if (tma_out and out_f32) else _dummy_map(device, torch.float32),
        OUT16=out3 if (tma_out and not out_f32) else _dummy_map(device, torch.bfloat16),
        # round 27: the stt program keeps the register-store alias next to the OUT16 descriptor  [Cake launcher]
        out=o_alias if (not out_f32 and (not tma_out or plan.stt)) else dummy16,
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
        ldo=ldo,
        out_l=out_l,
        num_cluster_tiles=plan.num_cluster_tiles,
        num_l=plan.L,
        num_full=plan.num_full,
        iters_per_unit=plan.iters_per_unit,
        sk_iters=plan.sk_iters,
        # the eight-warp programs' raster fast divisors: tiles per batch entry, tiles per raster group, the group's row
        # tiles and the last group's (the ten-warp programs' argument plans do not name them)
        fd_tiles_l=plan.m_tiles * plan.n_tiles,
        fd_tiles_g=plan.group_m * plan.n_tiles,
        fd_group=plan.group_m,
        fd_last=plan.m_tiles % plan.group_m or plan.group_m,
    )


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
    store_hint: Optional[str] = None,
    pd: Optional[int] = None,
    sh: Optional[int] = None,
    sk_sync: Optional[bool] = None,
    sk_sync_m: Optional[int] = None,
    sk_slab: Optional[int] = None,
    cta1: Optional[bool] = None,
    cgrp: Optional[bool] = None,
    ovr: Optional[bool] = None,
    stt: Optional[bool] = None,
    pro: Optional[int] = None,
    form: Optional[str] = None,
) -> PreparedGemm:
    """Validate one binding, plan it for the device and prepare its launch (the only
    allocations of the K1 backend: the stream-K partial slabs and the slice counters).  See the module docstring for the view contract; the keyword
    knobs default to the Cake launcher's choices."""
    knobs: dict[str, Any] = dict(
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
        store_hint=store_hint,
        pd=pd,
        sh=sh,
        sk_sync=sk_sync,
        sk_sync_m=sk_sync_m,
        sk_slab=sk_slab,
        cta1=cta1,
        cgrp=cgrp,
        ovr=ovr,
        stt=stt,
        pro=pro,
        form=form,
    )
    plan, a_desc, b_desc, O3, arch, module_name = _k1_plan(
        A, B, out, transposed_out=transposed_out, knobs=knobs
    )
    device = out.device
    ws, counters = _k1_workspace(plan, device)
    o_alias = flat_alias(O3)
    ldo, out_l = _out_strides(O3)
    tensors = dict(
        A=a_desc, B=b_desc, out3=O3, ws=ws, counters=counters, o_alias=o_alias
    )
    values = _k1_values(
        plan,
        device,
        A=a_desc,
        B=b_desc,
        out3=O3,
        o_alias=o_alias,
        ws=ws,
        counters=counters,
        ldo=ldo,
        out_l=out_l,
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


# ---------------------------------------------------------------------------
# K1 convenience path (cached planned launches)
# ---------------------------------------------------------------------------


def _k1_swapped(
    plan: GemmPlan, A: torch.Tensor, B: torch.Tensor, *, transposed_out: bool
) -> bool:
    """Did the planner run the swapped GEMM (``B.T @ A.T`` with the transposed store) for these views?  The
    launcher's small-M swap rule on the caller's dimensions, unless the registry fallback undid the swap
    (``plan.swap_fallback``)."""
    A3, B3 = as_batched(A, "A"), as_batched(B, "B")
    L, M, N = int(A3.shape[0]), int(A3.shape[1]), int(B3.shape[2])
    return swap_small_m(L, M, N, transposed_out) and not plan.swap_fallback


def _k1_views(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, swapped: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The operand views of a K1 launch for the caller's tensors -- ``(a_desc, b_desc, out3)`` exactly as
    :func:`plan_dense_projection_gemm` derives them (batched, the operands exchanged and transposed when the plan
    swapped, the ``[L, outer, inner]`` TMA spans of :func:`operand_view`) -- without planning.  ``operand_view``
    re-checks the stride and alignment contract of the views."""
    A3, B3, O3 = as_batched(A, "A"), as_batched(B, "B"), as_batched(out, "out")
    if swapped:
        A3, B3 = B3.transpose(1, 2), A3.transpose(1, 2)
    _, a_desc = operand_view(A3, "A", k_axis=2)
    _, b_desc = operand_view(B3, "B", k_axis=1)
    return a_desc, b_desc, O3


def _same_view(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.data_ptr() == b.data_ptr()
        and a.shape == b.shape
        and a.stride() == b.stride()
        and a.dtype == b.dtype
    )


@dataclass
class _CachedGemm:
    """A planned K1 launch of the convenience path: the plan, the resolved module entry, the argument template
    (constants and :class:`_Slot` markers in the module's plan order) and the arena its workspace is carved
    from.  The plan holds the workspace requirement (``ws_f32_elems`` slabs, ``counters_alloc_u32`` counters)
    the arena was grown to when the entry was planned."""

    module_name: str
    arch: str
    plan: GemmPlan
    swapped: bool
    entry: Callable[..., Any] = field(repr=False)
    template: tuple = field(repr=False)
    arena: _Arena = field(repr=False)
    views: Optional[dict[str, torch.Tensor]] = field(default=None, repr=False)
    views_version: int = -1

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]:
        """``ws`` and ``counters`` are pointer parameters (the binding checks dtype, device and contiguity
        only): the arena buffers themselves, no slicing."""
        return dict(ws=arena.buffers["k1_ws"], counters=arena.buffers["k1_counters"])


_K1_CACHE: "OrderedDict[tuple, _CachedGemm]" = OrderedDict()


def _prepare_cached_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool,
    knobs: dict[str, Any],
    arena: _Arena,
) -> tuple[_CachedGemm, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(entry, (a_desc, b_desc, out3))``: the planned entry and the per-call views of this first call (the
    rebuild the fail-closed check validated against the planner's views)."""
    plan, a_desc, b_desc, O3, arch, module_name = _k1_plan(
        A, B, out, transposed_out=transposed_out, knobs=knobs
    )
    device = out.device
    swapped = _k1_swapped(plan, A, B, transposed_out=transposed_out)
    rebuilt = _k1_views(A, B, out, swapped=swapped)
    if not all(
        _same_view(got, want)
        for got, want in zip(rebuilt, (a_desc, b_desc, O3), strict=True)
    ):
        raise RuntimeError(
            "dense_projection_gemm: the per-call operand views differ from the planner's; "
            "the convenience path cannot cache this binding (use prepare_dense_projection_gemm)"
        )
    ldo, out_l = _out_strides(O3)
    values = _k1_values(
        plan,
        device,
        A=_Slot("A"),
        B=_Slot("B"),
        out3=_Slot("out3"),
        o_alias=_Slot("o_alias"),
        ws=_Slot("ws"),
        counters=_Slot("counters"),
        ldo=ldo,
        out_l=out_l,
    )
    entry, template = _bind(module_name, values, plan.grid, arch)
    # the workspace is a slot: grow the arena only once the module and the grid are known to be good
    arena.require("k1_ws", plan.ws_f32_elems)
    arena.require("k1_counters", plan.counters_alloc_u32)
    arena.refs += 1
    return _CachedGemm(
        module_name, arch, plan, swapped, entry, template, arena
    ), rebuilt


def _launch_cached_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    transposed_out: bool = False,
    **knobs: Any,
) -> torch.Tensor:
    """The convenience path of K1: plan on the first call for a (device, stream, geometry, knobs) key; rebind
    the caller's views, resolve the workspace out of the arena of the device and stream and launch on every
    call (no planning, no device query, no allocation)."""
    device = _same_cuda_device(A, B, out)
    index = _cuda_index(device)
    stream = _current_raw_stream(index)
    key = convenience_key(
        index, stream, (bool(transposed_out), *sorted(knobs.items())), A, B, out
    )
    cached = _cache_get(_K1_CACHE, key)
    operands = None
    if cached is None:
        cached, operands = _plan_entry(
            _K1_CACHE,
            key,
            index,
            stream,
            device,
            lambda arena: _prepare_cached_gemm(
                A, B, out, transposed_out=transposed_out, knobs=knobs, arena=arena
            ),
        )
    if operands is None:
        operands = _k1_views(A, B, out, swapped=cached.swapped)
    a_desc, b_desc, O3 = operands
    views = dict(A=a_desc, B=b_desc, out3=O3, o_alias=flat_alias(O3))
    views.update(_entry_views(cached))
    arguments = _resolve_arguments(cached.template, views)
    with _ffi_stream_context(index, stream):
        cached.entry(*arguments)
    return out


def dense_projection_gemm(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, transposed_out: bool = False
) -> torch.Tensor:
    """``out[l, m, n] = sum_k A[l, m, k] B[l, k, n]`` for strided views (the cached convenience path:
    planned once per device, stream and view geometry, then rebound and launched)."""
    return _launch_cached_gemm(A, B, out, transposed_out=transposed_out)


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
    G: torch.Tensor,
    X: torch.Tensor,
    *,
    arch: Optional[str] = None,
    out_f32: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """``(A, B, transposed_out)`` of the weight gradient ``G[T, N].T @ X[T, K] -> dW[N, K]``:
    when ``wgrad_swapped`` the swapped GEMM ``X.T @ G`` runs with the transposed store (the
    result lands directly in ``[N, K]``), otherwise ``G.T @ X``.  ``arch`` / ``out_f32`` let a
    measured rule of that architecture and output dtype decide the orientation (see ``wgrad_swapped``).
    [Cake ``projection_wgrad`` L1507-L1518 / ``wgrad_swapped`` L1496-L1504]"""
    N, K = int(G.shape[1]), int(X.shape[1])
    if wgrad_swapped(N, K, arch=arch, out_f32=out_f32):
        return X.t(), G, True
    return G.t(), X, False


def _orientation_rules(arch: str, N: int, K: int, out_f32: bool) -> tuple[bool, bool]:
    """Whether ``ROW_RULES`` keys the weight gradient ``[N, K]`` of ``arch`` in the swapped and in the direct
    orientation; an architecture keyed in neither reads its parent's keys (``RULE_PARENT``)."""
    a: Optional[str] = arch
    while a is not None:
        swapped = (a, True, True, bool(out_f32), True, False, N, None, K) in ROW_RULES
        direct = (a, True, True, bool(out_f32), False, False, K, None, N) in ROW_RULES
        if swapped or direct:
            return swapped, direct
        a = RULE_PARENT.get(a)
    return False, False


def wgrad_swapped(
    N: int, K: int, *, arch: Optional[str] = None, out_f32: bool = False
) -> bool:
    """Orientation of the weight gradient.  A measured rule decides first: the row's ``ROW_RULES`` key
    names the orientation it was measured in - the swapped key ``(arch, nn, out_f32, True, False, N, None, K)``
    (the swapped GEMM ``X.T @ G`` has M = K and N = N with the transposed store) or the direct key
    ``(arch, nn, out_f32, False, False, K, None, N)`` (``G.T @ X``: M = N, N = K) - and a row keyed in exactly one
    orientation takes it (the 4096 -> 512 rows swap on sm_100a / sm_103a, the 4096 -> 1024 bf16 row swaps on
    sm_100a; the launcher's rule for the small-N rows below agrees).  Without a rule, or without ``arch``, the
    padded-tile-work heuristic of the Cake launcher: swapped when 256-row tiles over K and column tiles padded
    from N move less than 256-row tiles padded from N rows and ``default_block_n`` column tiles over K
    (N < 256 and N = 576 swap; a tie keeps the direct route).  [Cake ``wgrad_swapped`` L1496-L1504]"""
    if arch is not None:
        # the weight-gradient rules are keyed ragged in K (= T): the exact orientation keys, not ``_measured_row_rule``;
        # an architecture without a rule in either orientation reads its parent's keys (``RULE_PARENT``)
        swapped_rule, direct_rule = _orientation_rules(arch, N, K, out_f32)
        if swapped_rule != direct_rule:
            return swapped_rule
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
    """Prepared ``out[N, K] = G[T, N].T @ X[T, K]``; the orientation follows the row's measured rule, else the
    small-``N`` swap rule of the Cake launcher (``wgrad_swapped``)."""
    A, B, transposed = wgrad_views(
        G, X, arch=arch_for(out.device), out_f32=out.dtype == torch.float32
    )
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
    A, B, transposed = wgrad_views(
        G, X, arch=arch_for(out.device), out_f32=out.dtype == torch.float32
    )
    return _launch_cached_gemm(
        A, B, out, transposed_out=transposed, sk=sk, cta_rows=cta_rows
    )


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


# The fused split program: one launch reads the fp32 source once and writes the three BF16 parts (the residual
# stays in registers, so it needs no fp32 scratch).  The K2 paths take it when the registry holds it for the
# device's architecture and the source is a contiguous view, and split with the torch chain
# :func:`split_fp32_to_bf16x3_` otherwise -- the same bits either way for finite inputs (a NaN differs in its
# payload: the program writes the canonical BF16 NaN).  Its launch is ``exe(src, parts, n)`` over
# ``grid = ceil(n / SPLIT_BLOCK_ELEMS)`` blocks of 256 threads, 8 elements per thread.
SPLIT_TEMPLATE = "split_fp32_bf16x3"
SPLIT_BLOCK_ELEMS = 2048
_SPLIT_MAX_N = 2**31 - 1  # ``n`` is an int32 launch parameter


def fused_split_module(arch: str) -> Optional[str]:
    """The registered module serving the fused split program on ``arch``, or ``None`` (the torch chain splits)."""
    if SPLIT_TEMPLATE not in KERNELS.get(arch, {}):
        return None
    return select_module(arch, SPLIT_TEMPLATE)


def _fused_split_for(arch: str, split_src: torch.Tensor) -> Optional[str]:
    """The fused split's module when it is registered for ``arch`` and can split ``split_src``: a contiguous
    fp32 view (the program reads the elements in order; a storage offset is fine) of at most int32 elements.
    ``None`` selects the torch chain."""
    if not (split_src.is_contiguous() and split_src.numel() <= _SPLIT_MAX_N):
        return None
    return fused_split_module(arch)


def _split_values(src: Any, parts: Any, n: int) -> dict[str, Any]:
    """The keyword arguments of a fused split launch -- the fp32 source, the ``[3, *src.shape]`` BF16 stack and
    the element count -- ordered by the module's argument plan in :func:`_bind`.  The one place the program's
    binding names live."""
    return dict(src=src, parts=parts, n=int(n))


def _split_grid(n: int) -> tuple[int, int, int]:
    """The grid of a fused split launch over ``n`` elements."""
    return (_ceil_div(int(n), SPLIT_BLOCK_ELEMS), 1, 1)


@dataclass
class PreparedRouterGemm:
    """One prepared FP32 router GEMM: the retained BF16x3 stack of the second operand (re-split
    from the bound fp32 view on every ``launch()`` -- one launch of the fused split program when
    this checkout registers it for the device, else the torch chain through the fp32 scratch
    ``resid``), the split-K workspace and the fixed-order host reduction.  ``launch()`` writes
    ``out`` and returns it without allocating."""

    module_name: str
    arch: str
    plan: RouterPlan
    out: torch.Tensor
    tensors: dict[str, torch.Tensor] = field(repr=False)
    device_index: int = 0
    _launch: Optional[_Launch] = field(default=None, repr=False)
    _split: Optional[_Launch] = field(default=None, repr=False)

    @property
    def template(self) -> str:
        return self.plan.template

    @property
    def grid(self) -> tuple[int, int, int]:
        return self.plan.grid

    @property
    def splits(self) -> int:
        return self.plan.splits

    @property
    def fused_split(self) -> bool:
        """Whether ``launch()`` splits with the fused program (else the torch chain)."""
        return self._split is not None

    def launch(self) -> torch.Tensor:
        assert self._launch is not None
        t = self.tensors
        if self._split is None:
            split_fp32_to_bf16x3_(t["split_src"], t["parts"], t["resid"])
        with _ffi_stream_context(self.device_index):
            if self._split is not None:
                self._split()
            self._launch()
        if self.plan.splits > 1:
            torch.sum(
                t["workspace"], dim=0, out=self.out
            )  # fixed-order reduction of the partials
        return self.out

    __call__ = launch


def _k2_plan(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, splits: Optional[int]
) -> tuple[RouterPlan, torch.Tensor, torch.Tensor, str, str]:
    """Plan one K2 binding for its device: ``(plan, A_eff, B_eff, arch, module_name)``."""
    device = _same_cuda_device(A, B, out)
    arch = require_arch(device)
    plan, A_eff, B_eff = plan_router_fp32_gemm(A, B, out, splits=splits)
    return plan, A_eff, B_eff, arch, select_module(arch, plan.template)


def _k2_views(
    A: torch.Tensor, B: torch.Tensor, plan: RouterPlan
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(a_view, split_src)`` of a router launch for the caller's operands: the streamed operand as the
    ``[1, M, K]`` / ``[1, K, M]`` TMA span and the second operand in its own layout -- the source of the BF16x3
    split (K-major parts ``[3, N, K]`` come from ``B.T``, MN-major parts ``[3, K, N]`` from ``B``) -- after the
    weight gradient's swap when the plan swapped (``out.T = B.T A.T``)."""
    A_eff, B_eff = (B.t(), A.t()) if plan.swapped else (A, B)
    split_src = B_eff if plan.b_mn else B_eff.t()
    a_view = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
    return a_view, split_src


def _k2_checked_views(
    A: torch.Tensor,
    B: torch.Tensor,
    A_eff: torch.Tensor,
    B_eff: torch.Tensor,
    plan: RouterPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """:func:`_k2_views` of the caller's operands, checked against the planner's operands ``(A_eff, B_eff)``
    of :func:`plan_router_fp32_gemm` in the launch form (the streamed ``[1, M, K]`` / ``[1, K, M]`` span under
    ``plan.a_mn``, the split source under ``plan.b_mn``).  The convenience path rebuilds the views from the
    caller's tensors on every call, so a rebuild that is not the planner's view fails closed here (the K1 check
    of :func:`_prepare_cached_gemm`) instead of launching on a view the plan never validated."""
    a_view, split_src = _k2_views(A, B, plan)
    want_a = A_eff.t().unsqueeze(0) if plan.a_mn else A_eff.unsqueeze(0)
    want_split = B_eff if plan.b_mn else B_eff.t()
    if not (_same_view(a_view, want_a) and _same_view(split_src, want_split)):
        raise RuntimeError(
            "router_fp32_gemm: the per-call operand views differ from the planner's; "
            "the convenience path cannot cache this binding (use prepare_router_fp32_gemm)"
        )
    return a_view, split_src


def _k2_workspace(
    plan: RouterPlan,
    split_src: torch.Tensor,
    device: torch.device,
    *,
    scratch: bool = True,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """The retained BF16x3 stack of the second operand, the fp32 residual scratch of the torch-chain split
    (``None`` when the fused split program splits: ``scratch=False``) and the split-K workspace (``None`` for a
    single split: the kernel writes ``out`` directly)."""
    parts = torch.empty((3, *split_src.shape), device=device, dtype=torch.bfloat16)
    resid = None
    if scratch:
        resid = torch.empty(split_src.shape, device=device, dtype=torch.float32)
    workspace = None
    if plan.splits > 1:
        expect = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        workspace = torch.empty(
            (plan.splits, *expect), device=device, dtype=torch.float32
        )
    return parts, resid, workspace


def _k2_workspace_strides(workspace: torch.Tensor) -> tuple[int, int]:
    """``(ldo, out_l)`` of the ``[splits, rows, cols]`` split-K workspace: one row-major partial per split."""
    return int(workspace.shape[2]), int(workspace.shape[1] * workspace.shape[2])


def _k2_out32(out: torch.Tensor) -> torch.Tensor:
    """The pointer argument for a single-split launch that writes ``out`` directly."""
    return out if out.is_contiguous() else flat_alias(out)


def _k2_values(
    plan: RouterPlan,
    *,
    A: Any,
    parts: Any,
    out32: Any,
    ldo: int,
    out_l: int,
) -> dict[str, Any]:
    """The keyword arguments of a K2 launch; ``A`` / ``parts`` / ``out32`` are views or :class:`_Slot`
    markers."""
    return dict(
        A=A,
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


def prepare_router_fp32_gemm(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int] = None,
) -> PreparedRouterGemm:
    """Validate one fp32 binding, plan it and prepare its launch (the only allocations of the
    K2 backend: the BF16x3 stack, the fp32 residual scratch of the torch-chain split and the
    split-K workspace)."""
    plan, _, _, arch, module_name = _k2_plan(A, B, out, splits=splits)
    device = out.device
    a_view, split_src = _k2_views(A, B, plan)
    split_module = _fused_split_for(arch, split_src)
    parts, resid, workspace = _k2_workspace(
        plan, split_src, device, scratch=split_module is None
    )
    tensors = dict(A=a_view, split_src=split_src, parts=parts)
    if resid is not None:
        tensors["resid"] = resid
    if workspace is not None:
        tensors["workspace"] = workspace
        out32, ldo, out_l = workspace, *_k2_workspace_strides(workspace)
    else:
        out32, ldo, out_l = _k2_out32(out), int(out.stride(0)), 0
    values = _k2_values(plan, A=a_view, parts=parts, out32=out32, ldo=ldo, out_l=out_l)
    launch = bind_launch(module_name, values, plan.grid, arch)
    split_launch = None
    if split_module is not None:
        n = split_src.numel()
        split_launch = bind_launch(
            split_module, _split_values(split_src, parts, n), _split_grid(n), arch
        )
    return PreparedRouterGemm(
        module_name=module_name,
        arch=arch,
        plan=plan,
        out=out,
        tensors=tensors,
        device_index=_device_index(device),
        _launch=launch,
        _split=split_launch,
    )


@dataclass
class _CachedRouter:
    """A planned K2 launch of the convenience path: the plan, the resolved module entry, the argument template,
    the fused split's ``(entry, template)`` when the program is registered for the device (``None``: the torch
    chain splits) and the shapes of the workspace it carves out of its arena (the BF16x3 stack
    ``[3, *split_shape]``, under the torch chain the fp32 scratch ``split_shape`` and, for more than one split,
    the partials ``partials_shape``)."""

    module_name: str
    arch: str
    plan: RouterPlan
    entry: Callable[..., Any] = field(repr=False)
    template: tuple = field(repr=False)
    arena: _Arena = field(repr=False)
    split_shape: tuple[int, ...]
    partials_shape: Optional[tuple[int, ...]] = None
    views: Optional[dict[str, torch.Tensor]] = field(default=None, repr=False)
    views_version: int = -1
    split: Optional[tuple[Callable[..., Any], tuple]] = field(default=None, repr=False)

    def carve(self, arena: _Arena) -> dict[str, torch.Tensor]:
        """The ``parts`` stack (the TMA operand ``B``: exact shape), the ``resid`` scratch (torch chain only) and
        the split-K ``workspace`` as contiguous views over the first elements of the arena buffers."""
        bufs = arena.buffers
        views = dict(parts=_carve(bufs["k2_parts"], (3, *self.split_shape)))
        if self.split is None:
            views["resid"] = _carve(bufs["k2_resid"], self.split_shape)
        if self.partials_shape is not None:
            views["workspace"] = _carve(bufs["k2_partials"], self.partials_shape)
        return views


_K2_CACHE: "OrderedDict[tuple, _CachedRouter]" = OrderedDict()


def clear_convenience_caches() -> None:
    """Drop every cached convenience-path launch (K1 and K2) and release every workspace arena; a launch in
    flight keeps a reference to the buffers it was bound with until its FFI call has enqueued it, and the
    freed blocks are reused stream-ordered (see :class:`_Arena`)."""
    with _PLAN_LOCK:
        _K1_CACHE.clear()
        _K2_CACHE.clear()
        _ARENAS.clear()


def _prepare_cached_router(
    A: torch.Tensor,
    B: torch.Tensor,
    out: torch.Tensor,
    *,
    splits: Optional[int],
    arena: _Arena,
) -> tuple[_CachedRouter, tuple[torch.Tensor, torch.Tensor]]:
    """``(entry, (a_view, split_src))``: the planned entry and the per-call views of this first call (the
    rebuild the fail-closed check validated against the planner's operands)."""
    plan, A_eff, B_eff, arch, module_name = _k2_plan(A, B, out, splits=splits)
    a_view, split_src = _k2_checked_views(A, B, A_eff, B_eff, plan)
    split_shape = tuple(int(s) for s in split_src.shape)
    n = split_src.numel()
    split_module = _fused_split_for(arch, split_src)
    partials_shape = None
    if plan.splits > 1:
        rows, cols = (plan.N, plan.M) if plan.transposed_out else (plan.M, plan.N)
        partials_shape = (plan.splits, rows, cols)
        ldo, out_l = cols, rows * cols
    else:
        ldo, out_l = int(out.stride(0)), 0
    values = _k2_values(
        plan,
        A=_Slot("A"),
        parts=_Slot("parts"),
        out32=_Slot("out32"),
        ldo=ldo,
        out_l=out_l,
    )
    entry, template = _bind(module_name, values, plan.grid, arch)
    split = None
    if split_module is not None:
        split = _bind(
            split_module,
            _split_values(_Slot("src"), _Slot("parts"), n),
            _split_grid(n),
            arch,
        )
    # the workspace is a slot: grow the arena only once the modules and the grids are known to be good
    arena.require("k2_parts", 3 * n)
    if split is None:
        arena.require("k2_resid", n)
    if partials_shape is not None:
        arena.require("k2_partials", _numel(partials_shape))
    arena.refs += 1
    return _CachedRouter(
        module_name,
        arch,
        plan,
        entry,
        template,
        arena,
        split_shape,
        partials_shape,
        split=split,
    ), (a_view, split_src)


def _launch_cached_router(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, splits: Optional[int]
) -> torch.Tensor:
    """The convenience path of K2: plan on the first call for a (device, stream, geometry, splits) key; on every
    call re-split the second operand into the BF16x3 stack carved out of the arena of the device and stream (one
    launch of the fused split program when it is registered for the device, else the torch chain through the
    arena's fp32 scratch), launch and reduce the split-K partials into ``out`` in the fixed order."""
    device = _same_cuda_device(A, B, out)
    index = _cuda_index(device)
    stream = _current_raw_stream(index)
    key = convenience_key(
        index, stream, (None if splits is None else int(splits),), A, B, out
    )
    cached = _cache_get(_K2_CACHE, key)
    operands = None
    if cached is None:
        cached, operands = _plan_entry(
            _K2_CACHE,
            key,
            index,
            stream,
            device,
            lambda arena: _prepare_cached_router(A, B, out, splits=splits, arena=arena),
        )
    if operands is None:
        operands = _k2_views(A, B, cached.plan)
    a_view, split_src = operands
    # One K2 call is three stream operations on the shared workspace (the split into ``parts`` / ``resid``,
    # the kernel, the reduction of the partials): hold the arena's lock across them so another thread issuing
    # on this stream cannot interleave its own split or reduction (enqueue-only critical section).
    with cached.arena.lock:
        ws = _entry_views(cached)
        if cached.split is None:
            split_fp32_to_bf16x3_(split_src, ws["parts"], ws["resid"])
        workspace = ws.get("workspace")
        views = dict(
            A=a_view,
            parts=ws["parts"],
            out32=_k2_out32(out) if workspace is None else workspace,
            src=split_src,
        )
        with _ffi_stream_context(index, stream):
            if cached.split is not None:
                split_entry, split_template = cached.split
                split_entry(*_resolve_arguments(split_template, views))
            cached.entry(*_resolve_arguments(cached.template, views))
        if workspace is not None:
            torch.sum(
                workspace, dim=0, out=out
            )  # fixed-order reduction of the partials
    return out


def router_fp32_gemm(
    A: torch.Tensor, B: torch.Tensor, out: torch.Tensor, *, splits: Optional[int] = None
) -> torch.Tensor:
    """``out[m, n] = sum_k A[m, k] B[k, n]`` for fp32 views through the emulation (the cached convenience
    path: planned once per device, stream, view geometry and split count, then re-split and launched)."""
    return _launch_cached_router(A, B, out, splits=splits)


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
    splits: int = 6,
) -> torch.Tensor:
    """``out[N, K] = G[T, N].T @ X[T, K]`` in FP32: the swapped GEMM ``X.T @ G`` with a transposed store."""
    if out is None:
        out = torch.empty(G.shape[1], X.shape[1], device=G.device, dtype=torch.float32)
    return router_fp32_gemm(G.t(), X, out, splits=splits)
