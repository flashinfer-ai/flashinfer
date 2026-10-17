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

"""Quantized-base GEMM with a runtime BF16 LoRA correction (Cake generated programs).

``out = alpha_base * (dequant(a) @ dequant(b)^T) + lora_scale * (d @ lora_b^T) [+ bias]``

The base weights stay quantized (NVFP4 or FP8), the accumulation is fp32 and the
output is rounded once to bf16.  Two adapter paths exist; the call's API form
selects one (never a device read):

* **Raw path** (``lora_b`` + ``lora_scale``; both operators).  The LoRA factors
  are consumed as BF16 at their original scale, ``alpha_base`` and ``lora_scale``
  are independent fp32 device scalars and nothing is folded into the operands.
  Swapping the adapter is swapping pointers: no weight preparation
  (0-preparation switching).  ``lora_scale`` must be a normal float or ±0 (a
  one-element fp32 device tensor; ``None`` disables the adapter); denormal
  scales are unsupported.  This path is unchanged by the prepared path, bitwise.
* **Prepared path** (``lora_adapter=`` of :func:`mm_nvfp4_lora`; NVFP4 only).
  :func:`prepare_nvfp4_lora_adapter` folds the adapter ONCE at the adapter
  switch, ``b_folded = bf16(lora_b * lora_scale / alpha)``, and the call consumes
  ``b_folded`` with the base product accumulated first and ``alpha`` applied once
  in the fp32 epilogue -- SVDQuant-equivalent numerics (``lora_b`` is re-rounded
  once to bf16; the fold is bitwise the ``l1`` fold of
  :func:`~flashinfer.mm_nvfp4_svdquant`).  Snapshot semantics: the adapter pins
  its source tensors and a call raises :class:`StaleAdapterError` when one moved
  without a re-prepare; ``prepare_nvfp4_lora_adapter(..., out=adapter)`` refolds in
  place without a host read and is the adapter-switch protocol inside a captured
  CUDA graph.  The relaxation is opt-in (Cake user decisions of 2026-10-15:
  "允许放宽约束 1，分阶段做" and "A 认可通过，启动第二阶段").
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, Optional

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_quant_lora_gemm import (
    PROGRAMS,
    TARGET_ARCH,
    TARGET_OF_ARCH,
    CakeQuantLoraTarget,
    cake_quant_lora_launcher,
    cake_quant_lora_target,
)
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    get_device_sm_count,
    supported_compute_capability,
)

FAMILY_NVFP4 = "nvfp4"
FAMILY_FP8 = "fp8"
#: LoRA ranks the generated programs instantiate (``template <int RANK>`` on
#: Blackwell, runtime scalar on Hopper).  Rank 96 compiles in Cake but is not
#: delivered; any other rank raises ``ValueError``.
SUPPORTED_RANKS = (32, 64, 128)
#: Adapter states of one call, known from its API form alone (never a device read):
#: ``"none"`` (no adapter), ``"raw"`` (``lora_b`` + ``lora_scale``) and ``"prepared"``
#: (``lora_adapter=``).  Every ``PROGRAMS`` record lists the states it serves
#: (``adapter_states``): the raw-path programs serve ``"none"`` and ``"raw"`` (the
#: adapter-off call is the same program's ``has_lora == 0`` path), the prepared-path
#: program serves ``"prepared"`` only.
ADAPTER_STATES = ("none", "raw", "prepared")
#: The prepared-path raster rule this host ports (``dispatch[arch]["prepared"]["raster_rule"]``).
_PREPARED_RASTER_RULE = "k_band_then_n256_bytes_model"
_NO_ADAPTER_RANK = SUPPORTED_RANKS[0]
_SF_BLOCK = 16
_SF_SWIZZLE_ROWS = 128
_CARRIER_BUFFER = "cake_quant_lora_gemm_lora_carrier"


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _templates_for(
    family: str, target: CakeQuantLoraTarget, adapter_state: str = "raw"
) -> list[str]:
    arch = TARGET_ARCH[target]
    return sorted(
        name
        for name, record in PROGRAMS.items()
        if record["family"] == family
        and arch in record["arches"]
        and adapter_state in record["adapter_states"]
    )


def _adapter_state(lora_b, lora_scale, lora_adapter) -> str:
    """The call's adapter state from its API form alone (no device read): ``"prepared"`` with ``lora_adapter``,
    ``"raw"`` with ``lora_b`` / ``lora_scale``, ``"none"`` otherwise.  Both forms in one call are refused."""
    prepared = lora_adapter is not None
    raw = lora_b is not None or lora_scale is not None
    if prepared and raw:
        raise ValueError(
            "pass either lora_adapter (the prepared path) or lora_b + lora_scale (the raw path), not both"
        )
    return "prepared" if prepared else ("raw" if raw else "none")


def fp8_blackwell_tile_n(record: dict, m: int, n: int, sm_count: int) -> int:
    """Pair-tile width of the FP8 Blackwell programs (the Cake ``TILE_N_DISPATCH_RULE``).

    The wide tile (lowest operand traffic per FLOP) unless the wide grid is single-wave
    and the narrow grid still fits one wave of resident CTA pairs (``sm_count // 2``).
    The rule counts the 256-row (``dispatch["pair_m"]``) tile grid, whatever the record's own pair tile.
    """
    wide, narrow = max(record["tile_ns"]), min(record["tile_ns"])
    pairs = max(1, sm_count // 2)
    tok_tiles = _ceil_div(m, int(record["dispatch"]["pair_m"]))
    tiles_wide = tok_tiles * _ceil_div(n, wide)
    if tiles_wide < pairs and tok_tiles * _ceil_div(n, narrow) <= pairs:
        return narrow
    return wide


def _fp8_blackwell_block(
    constants: dict, m: int, n: int, tile_n: int, sm_count: int
) -> int:
    """The packed ``g_t + block_pack * g_w`` block of the FP8 Blackwell blocked raster (the Cake ``blocked_raster_group``):
    ``g_t = isqrt(pairs)``, ``g_w = pairs // g_t`` for ``sm_count // 2`` resident CTA pairs, each side clamped to the tile
    grid and to ``block_max`` (8 x 9 on 74 and 76 pairs)."""
    pairs = max(1, int(sm_count) // 2)
    tok_tiles = _ceil_div(int(m), int(constants["pair_m"]))
    w_tiles = _ceil_div(int(n), int(tile_n))
    block_max = int(constants["block_max"])
    g_t = max(1, min(math.isqrt(pairs), tok_tiles, block_max))
    g_w = max(1, min(pairs // g_t, w_tiles, block_max))
    return g_t + int(constants["block_pack"]) * g_w


def fp8_blackwell_raster_group(record: dict, m: int, n: int, sm_count: int) -> int:
    """Raster scalar of one FP8 Blackwell program: the packed block of the blocked raster for a ``raster_mode == "blocked"``
    record (``_fp8_blackwell_block``), else the grouped-raster size (the Cake ``RASTER_GROUP_RULE``)."""
    if record["raster_mode"] == "blocked":
        return _fp8_blackwell_block(
            record["dispatch"], m, n, int(record["tile_n"]), sm_count
        )
    tokens = int(record["raster_group_tokens"])
    tok_tiles = _ceil_div(
        m, int(record["dispatch"]["pair_m"])
    )  # the raster rules count the 256-row tile grid
    return tokens if tok_tiles > tokens else 0


def fp8_blackwell_l2_point(
    record: dict, arch: str, m: int, n: int, k: int, tile_n: int, sm_count: int
) -> tuple[str, str]:
    """``(sched, raster_mode)`` of the FP8 Blackwell production launch for a problem: the Cake rule-gated L2-locality point,
    evaluated from the delivered ``record["dispatch"]`` data (identical on every FP8 Blackwell record).

    On the architectures in ``dispatch["arches"]`` the gated point (``gated_sched`` / ``gated_raster_mode``: the
    clusterlaunchcontrol dynamic tile scheduler, one (2, 1, 1) cluster per ``pair_m x tile_n`` tile, with the 2-D blocked
    tile raster) wins when the launch has at least ``min_tiles_per_pair`` tiles per resident CTA pair (``sm_count // 2``
    pairs), ``K >= min_k`` and the ``g_t x g_w`` block is a proper sub-rectangle of the tile grid (``tok_tiles > g_t`` and
    ``w_tiles > g_w``); everywhere else the default point (the static persistent schedule with the grouped raster).
    """
    constants = record["dispatch"]
    default = (str(constants["default_sched"]), str(constants["default_raster_mode"]))
    if arch not in constants["arches"] or str(record["form"]) != str(constants["form"]):
        return default
    pairs = max(1, int(sm_count) // 2)
    tok_tiles = _ceil_div(int(m), int(constants["pair_m"]))
    w_tiles = _ceil_div(int(n), int(tile_n))
    block_pack = int(constants["block_pack"])
    packed = _fp8_blackwell_block(constants, m, n, tile_n, sm_count)
    g_t, g_w = packed % block_pack, packed // block_pack
    if (
        tok_tiles > g_t
        and w_tiles > g_w
        and tok_tiles * w_tiles / pairs >= int(constants["min_tiles_per_pair"])
        and int(k) >= int(constants["min_k"])
    ):
        return str(constants["gated_sched"]), str(constants["gated_raster_mode"])
    return default


def fp8_blackwell_tile_m(
    record: dict, arch: str, m: int, n: int, k: int, tile_n: int, sm_count: int
) -> int:
    """Token rows per CTA of the FP8 Blackwell production launch for a problem: the Cake production ``tile_m`` rule,
    evaluated from the delivered ``record["dispatch"]`` data (identical on every FP8 Blackwell record).

    On the architectures in ``dispatch["tile_m_arches"]`` (for the rule's form) the 512-row pair tile (``tile_m_wide``
    token rows per CTA) wins when the launch is on the gated L2-locality point (``fp8_blackwell_l2_point`` returns
    ``gated_sched`` / ``gated_raster_mode``) and has at least ``tile_m_min_waves`` tiles of ``2 * tile_m_wide x tile_n`` per
    resident CTA pair (``sm_count // 2`` pairs) and ``K >= tile_m_min_k``; everywhere else ``tile_m_default`` (the 256-row
    pair tile).  The 512-row tile is bound to the gated point (the Cake ``production_tile_m``), so the delivered point space
    has no static / grouped 512-row program; the L2-locality rules keep counting the 256-row tile grid.
    """
    constants = record["dispatch"]
    default = int(constants["tile_m_default"])
    if arch not in constants["tile_m_arches"] or str(record["form"]) != str(
        constants["form"]
    ):
        return default
    gated = (str(constants["gated_sched"]), str(constants["gated_raster_mode"]))
    if fp8_blackwell_l2_point(record, arch, m, n, k, tile_n, sm_count) != gated:
        return default
    wide = int(constants["tile_m_wide"])
    pairs = max(1, int(sm_count) // 2)
    tiles = _ceil_div(int(m), 2 * wide) * _ceil_div(int(n), int(tile_n))
    if tiles >= int(constants["tile_m_min_waves"]) * pairs and int(k) >= int(
        constants["tile_m_min_k"]
    ):
        return wide
    return default


def fp8_blackwell_program(
    candidates: list[str],
    record: dict,
    arch: str,
    m: int,
    n: int,
    k: int,
    sm_count: int,
) -> str:
    """The FP8 Blackwell program for a problem: the pair-tile width (``fp8_blackwell_tile_n``), the rule-gated
    L2-locality point (``fp8_blackwell_l2_point``) and the pair-tile height (``fp8_blackwell_tile_m``) select the candidate
    that carries that ``tile_n`` / ``tile_m`` / ``sched`` / ``raster_mode``; raises ``ValueError`` when no delivered program
    carries the point."""
    tile_n = fp8_blackwell_tile_n(record, m, n, sm_count)
    sched, raster_mode = fp8_blackwell_l2_point(record, arch, m, n, k, tile_n, sm_count)
    tile_m = fp8_blackwell_tile_m(record, arch, m, n, k, tile_n, sm_count)
    for name in candidates:
        program = PROGRAMS[name]
        if (
            int(program["tile_n"]) == tile_n
            and int(program["tile_m"]) == tile_m
            and program["sched"] == sched
            and program["raster_mode"] == raster_mode
        ):
            return name
    raise ValueError(
        f"no delivered FP8 program on {arch} carries TILE_N {tile_n} and tile_m {tile_m} with sched={sched!r} "
        f"raster_mode={raster_mode!r}; delivered: {sorted(candidates)}"
    )


def sm90_fp8_lora_program(
    candidates: list[str], record: dict, m: int, n: int, k: int, sm_count: int
) -> str:
    """The SM90 FP8 program for a problem: the Cake SM90 dispatch rule, evaluated from the delivered
    ``record["dispatch"]`` data (identical on both SM90 records).

    The persistent kernel rasters ``default_group_m`` m-pairs (``cluster_m x block_m`` rows each) across every N tile,
    so one raster group re-reads an A panel of ``default_group_m * cluster_m * block_m * K`` bytes per N tile.  Above
    ``panel_budget_bytes`` (and up to ``max_k``, the K range the gated program was measured on) on runs of at least
    ``waves_min`` cluster-tile waves over the launch's clusters -- ``(sm_count - sm_count % cluster_m) / cluster_m``
    clusters, ``ceil(M / block_m)`` m-tiles paired per cluster, ``ceil(N / block_n)`` n-tiles -- the gated program
    (TMA L2 policies: A evict_last, W evict_first; raster group ``gated_group_m``) wins; everywhere else the default
    program.  The selected record's ``group_m`` is the raster group the host passes.
    """
    constants = record["dispatch"]
    default_name, gated_name = (
        str(constants["default_template"]),
        str(constants["gated_template"]),
    )
    for name in (default_name, gated_name):
        if name not in candidates:
            raise ValueError(
                f"the SM90 dispatch data names program {name!r}, delivered: {sorted(candidates)}"
            )
    block_m, block_n, cluster_m = (
        int(constants["block_m"]),
        int(constants["block_n"]),
        int(constants["cluster_m"]),
    )
    k = int(k)
    panel_bytes = int(constants["default_group_m"]) * cluster_m * block_m * k
    if panel_bytes > int(constants["panel_budget_bytes"]) and k <= int(
        constants["max_k"]
    ):
        num_clusters = (int(sm_count) - int(sm_count) % cluster_m) // cluster_m
        if num_clusters < 1:
            raise ValueError(
                f"the SM90 persistent grid needs at least {cluster_m} SMs, got sm_count={sm_count}"
            )
        m_tiles = _ceil_div(int(m), block_m)
        n_tiles = _ceil_div(int(n), block_n)
        pairs, tail = divmod(m_tiles, cluster_m)
        n_pairs = _ceil_div(n_tiles, cluster_m)
        waves = (pairs * n_tiles + tail * n_pairs) / num_clusters
        if waves >= int(constants["waves_min"]):
            return gated_name
    return default_name


_NVFP4_ELEM_BITS = 4
_NVFP4_SCALE_BLOCK = 16  # one ue4m3 scale per 16 FP4 elements along K
_LORA_ELEM_BYTES = (
    2  # bf16 LoRA rows stream with their operand (d with a, lora_b with b)
)


def _nvfp4_operand_row_bytes(k: int, rank: int) -> int:
    """Bytes one operand row (an M row of ``a`` or an N row of ``b``) occupies: packed FP4 + block scales + LoRA row."""
    if k <= 0 or k % _NVFP4_SCALE_BLOCK:
        raise ValueError(
            f"K={k} must be a positive multiple of the scale block {_NVFP4_SCALE_BLOCK}"
        )
    return (
        (k * _NVFP4_ELEM_BITS) // 8
        + k // _NVFP4_SCALE_BLOCK
        + int(rank) * _LORA_ELEM_BYTES
    )


def _nvfp4_raster_dram_bytes(
    m: int,
    n: int,
    k: int,
    rank: int,
    *,
    tile_m: int,
    tile_n: int,
    raster_group: int,
    l2_budget_bytes: int,
    mainline_order: str,
) -> tuple[int, str, int, int]:
    """Modeled DRAM read bytes of one token-grouped raster order (the Cake rule's bytes model, ported verbatim).

    Grouped order ``G >= 1``: groups of ``G`` M tiles sweep every N tile; the group's A set must stay L2-resident next
    to one streamed W tile for A to be read once, W is then read once per group unless the whole of W fits as well.
    ``G = 0`` is the mainline order: along-M == grouped(tok_tiles), along-N == grouped(1).  Returns
    ``(dram_read_bytes, order, tok_tiles, w_tiles)``.
    """
    for name, value in (
        ("M", m),
        ("N", n),
        ("K", k),
        ("tile_m", tile_m),
        ("tile_n", tile_n),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if int(rank) < 0 or int(raster_group) < 0:
        raise ValueError(
            f"rank and raster_group must be >= 0, got rank={rank} raster_group={raster_group}"
        )
    if int(l2_budget_bytes) <= 0:
        raise ValueError(f"l2_budget_bytes must be positive, got {l2_budget_bytes}")
    tok_tiles = _ceil_div(int(m), int(tile_m))
    w_tiles = _ceil_div(int(n), int(tile_n))
    row_bytes = _nvfp4_operand_row_bytes(int(k), int(rank))
    a_tile = int(tile_m) * row_bytes
    w_tile = int(tile_n) * row_bytes
    a_bytes = tok_tiles * a_tile
    w_bytes = w_tiles * w_tile
    budget = int(l2_budget_bytes)
    if int(raster_group) == 0:
        if mainline_order == "longer":
            order = "along_m" if w_tiles > tok_tiles else "along_n"
        elif mainline_order in ("along_m", "along_n"):
            order = mainline_order
        else:
            raise ValueError(
                f"mainline_order must be one of ('longer', 'along_m', 'along_n'), got {mainline_order!r}"
            )
        g = tok_tiles if order == "along_m" else 1
    else:
        order = "grouped"
        g = min(int(raster_group), tok_tiles)
    groups = _ceil_div(tok_tiles, g)
    a_group = g * a_tile
    if a_group + w_tile <= budget:
        a_passes = 1
        w_passes = 1 if a_group + w_bytes <= budget else groups
    else:  # the group's A set thrashes: every W tile pass re-reads it
        a_passes = w_tiles
        w_passes = groups
    return a_passes * a_bytes + w_passes * w_bytes, order, tok_tiles, w_tiles


def _nvfp4_svdq_raster_bytes(
    m: int,
    n: int,
    k: int,
    rank: int,
    *,
    pair_m: int,
    tile_n: int,
    group: int,
    budget: int,
) -> int:
    """Modeled DRAM read bytes of one grouped raster ``group`` of the prepared-path program: the Cake 256-wide kernel's
    bytes model, ported verbatim.  Per token row A carries ``K/2`` NVFP4 bytes + ``K/16`` scale bytes + ``2 rank`` bf16
    LoRA bytes and per weight row W the same (``_nvfp4_operand_row_bytes``).  ``group == 0`` is the token-first order
    (consecutive tiles share the weight tile, so A streams once per weight tile unless it is L2-resident); ``group >= 1``
    walks ``group`` token tiles per weight tile: the group's A set stays resident and W streams ``ceil(tok_tiles /
    group)`` times unless it fits next to it; an A set above the budget thrashes (A streams per weight tile)."""
    tok_tiles = _ceil_div(int(m), int(pair_m))
    w_tiles = _ceil_div(int(n), int(tile_n))
    row_bytes = _nvfp4_operand_row_bytes(int(k), int(rank))
    a_bytes = int(m) * row_bytes
    w_bytes = int(n) * row_bytes
    budget = int(budget)
    if int(group) == 0:
        return w_bytes + (a_bytes if a_bytes <= budget else w_tiles * a_bytes)
    a_group = min(int(group), tok_tiles) * int(pair_m) * row_bytes
    passes = _ceil_div(tok_tiles, int(group))
    if a_group <= budget:
        return a_bytes + (w_bytes if w_bytes <= budget - a_group else passes * w_bytes)
    return w_tiles * a_bytes + passes * w_bytes


def nvfp4_svdq_raster_group(
    record: dict, arch: str, m: int, n: int, k: int, rank: int
) -> int:
    """Grouped-raster size of the prepared-path program: the Cake svdq kernel's own rule, evaluated from the delivered
    ``record["dispatch"][arch]["prepared"]`` data.

    The K band first: group ``k_band_group`` (1) for ``K <= k_band_max_k`` and ``M >= k_band_min_m``; otherwise the
    256-wide kernel's bytes model (``_nvfp4_svdq_raster_bytes``) over ``bytes_model_candidates`` at
    ``bytes_model_pair_m x bytes_model_tile_n`` tiles against ``bytes_model_budget_bytes`` -- fewest modeled DRAM read
    bytes, ties to the smallest group ``>= 1``, ``G = 0`` last.  Raises ``ValueError`` when ``record`` is not the
    prepared-path program or the delivered rule is not the one this host ports.
    """
    constants = record["dispatch"][arch]
    prepared = constants["prepared"]
    if (
        record["dispatch_kernel"] != prepared["kernel"]
        or "prepared" not in record["adapter_states"]
    ):
        raise ValueError(
            f"the prepared-path raster rule applies to the {prepared['kernel']!r} program only, got {record['dispatch_kernel']!r}"
        )
    if str(prepared["raster_rule"]) != _PREPARED_RASTER_RULE:
        raise ValueError(
            f"the delivered prepared-path raster rule {prepared['raster_rule']!r} is not {_PREPARED_RASTER_RULE!r}, "
            "which this host ports"
        )
    kernel = constants["kernels"][prepared["kernel"]]
    pair_m, tile_n = (
        int(prepared["bytes_model_pair_m"]),
        int(prepared["bytes_model_tile_n"]),
    )
    if (int(kernel["tile_m"]), int(kernel["tile_n"])) != (pair_m, tile_n):
        raise ValueError(
            f"the prepared-path bytes model counts {pair_m} x {tile_n} tiles, the {prepared['kernel']!r} program is "
            f"{kernel['tile_m']} x {kernel['tile_n']}"
        )
    if int(k) <= int(prepared["k_band_max_k"]) and int(m) >= int(
        prepared["k_band_min_m"]
    ):
        return int(prepared["k_band_group"])
    candidates = sorted({int(value) for value in prepared["bytes_model_candidates"]})
    if not candidates or candidates[0] < 0:
        raise ValueError(
            f"prepared-path raster candidates must be >= 0, got {prepared['bytes_model_candidates']!r}"
        )
    budget = int(prepared["bytes_model_budget_bytes"])
    if budget <= 0:
        raise ValueError(f"bytes_model_budget_bytes must be positive, got {budget}")
    last = (
        candidates[-1] + 1
    )  # G = 0 sorts after every grouped order on an exact modeled-byte tie
    return min(
        candidates,
        key=lambda group: (
            _nvfp4_svdq_raster_bytes(
                m, n, k, rank, pair_m=pair_m, tile_n=tile_n, group=group, budget=budget
            ),
            group if group >= 1 else last,
        ),
    )


def nvfp4_blackwell_raster_group(
    record: dict, arch: str, m: int, n: int, k: int, rank: int
) -> int:
    """Grouped-raster size of an NVFP4 pair-tile program: the Cake dispatch rule's raster selection, ported verbatim.

    A prepared-path record (``"prepared" in record["adapter_states"]``) follows the svdq kernel's own rule
    (:func:`nvfp4_svdq_raster_group`: the K band, then the 256-wide bytes model).  For the raw-path programs the
    candidates and the L2 resident budget come from the delivered ``record["dispatch"][arch]`` data (the Cake
    dispatch module's ``DISPATCH_CONSTANTS[arch]`` as committed: ``kernels[<kernel>]`` geometry + ``raster_candidates``
    and ``l2_budget_bytes``).  ``G >= 1`` is
    forced whenever the along-M mainline would walk more token tiles than weight tiles (``tok_tiles > w_tiles``);
    otherwise the candidate with the fewest modeled DRAM read bytes wins, ties go to the smallest grouped order
    ``G >= 1`` and ``G = 0`` only as the sole tie member.  Raises ``ValueError`` when no candidate is admissible.
    """
    if "prepared" in record["adapter_states"]:
        return nvfp4_svdq_raster_group(record, arch, m, n, k, rank)
    constants = record["dispatch"][arch]
    kernel = constants["kernels"][record["dispatch_kernel"]]
    tile_m, tile_n = int(kernel["tile_m"]), int(kernel["tile_n"])
    mainline_order = str(kernel["mainline_order"])
    budget = int(constants["l2_budget_bytes"])
    models = []
    for group in sorted({int(value) for value in kernel["raster_candidates"]}):
        if group < 0:
            raise ValueError(f"raster candidates must be >= 0, got {group}")
        dram_bytes, order, tok_tiles, w_tiles = _nvfp4_raster_dram_bytes(
            m,
            n,
            k,
            rank,
            tile_m=tile_m,
            tile_n=tile_n,
            raster_group=group,
            l2_budget_bytes=budget,
            mainline_order=mainline_order,
        )
        if group == 0 and order == "along_m" and tok_tiles > w_tiles:
            continue
        models.append((dram_bytes, 1 if group == 0 else 0, group))
    if not models:
        raise ValueError(
            f"no admissible raster group in {tuple(kernel['raster_candidates'])} for M={m} N={n} (tok_tiles > w_tiles needs a G >= 1 candidate)"
        )
    return min(models)[2]


_DISPATCH_BOUNDS = {
    "lt": lambda value, bound: value < bound,
    "le": lambda value, bound: value <= bound,
    "ge": lambda value, bound: value >= bound,
    "gt": lambda value, bound: value > bound,
    "eq": lambda value, bound: value == bound,
}


def nvfp4_blackwell_program(
    candidates: list[str], arch: str, m: int, n: int, k: int, rank: int
) -> str:
    """The NVFP4 pair-tile program for a problem: the Cake per-architecture dispatch rule (the pre-registered DISPATCH
    RULE of ``arch``), evaluated from the delivered ``PROGRAMS[...]["dispatch"][arch]["choice_rule"]`` data.

    The rule is an ordered clause list over ``(M, N, K, rank)``, first match wins: ``{"if": {"K_le": 4096, "M_le": 4096},
    "program": "pair"}`` clauses whose every condition (keys ``<M|N|K|rank>_<lt|le|ge|gt|eq>``, integer bounds; other
    clause keys are documentation) must hold, and a final ``{"else": "<program>"}`` default that must be the last entry.
    Every program a rule names must be delivered for ``arch`` (``candidates``; a rule naming an undelivered program fails
    closed instead of running another one), and the chosen kernel's applicability floor (``kernels[<kernel>]["min_k"]``)
    is asserted, so no rule can route a ``K < K_THRESHOLD`` shape to the 256-wide program.  This mirrors
    ``choose_kernel`` of the Cake dispatch module; the export's parity proof compares both on every contract row and a
    dense sweep per arch.
    """
    by_kernel = {PROGRAMS[name]["dispatch_kernel"]: name for name in candidates}
    constants = PROGRAMS[candidates[0]]["dispatch"][arch]
    rule = constants.get("choice_rule")
    if not isinstance(rule, list) or not rule:
        raise ValueError(
            f"the delivered NVFP4 dispatch constants for {arch} carry no ordered choice_rule"
        )
    problem = {"M": int(m), "N": int(n), "K": int(k), "rank": int(rank)}
    for index, clause in enumerate(rule):
        if "else" in clause:
            if index != len(rule) - 1 or "if" in clause:
                raise ValueError(
                    f"the NVFP4 dispatch rule's 'else' entry must be the last and only default (entry {index})"
                )
            kernel = str(clause["else"])
        else:
            holds = True
            for key, bound in (clause.get("if") or {}).items():
                axis, _, kind = str(key).rpartition("_")
                if axis not in problem or kind not in _DISPATCH_BOUNDS:
                    raise ValueError(
                        f"unsupported dispatch clause key {key!r} (expected <M|N|K|rank>_<lt|le|ge|gt|eq>)"
                    )
                holds &= bool(_DISPATCH_BOUNDS[kind](problem[axis], int(bound)))
            if not holds:
                continue
            kernel = str(clause["program"])
        if kernel not in by_kernel:
            raise ValueError(
                f"the dispatch rule names program {kernel!r}, delivered on {arch}: {sorted(by_kernel)}"
            )
        min_k = int(constants["kernels"][kernel].get("min_k", 0))
        if problem["K"] < min_k:
            raise ValueError(
                f"the NVFP4 dispatch rule routes K={k} to {kernel!r}, which needs K >= {min_k}"
            )
        return by_kernel[kernel]
    raise ValueError("the NVFP4 dispatch rule has no 'else' default entry")


def launch_grid(record: dict, m: int, n: int, sm_count: int) -> tuple[int, int, int]:
    """The launch grid of one generated program for an ``M x N`` output on ``sm_count`` SMs."""
    rule = record["grid"]
    if rule == "nvfp4_pairs":
        # CLC grid over the whole pair-tile domain: one (2, 1, 1) cluster per 256 x TILE_N tile; running pairs
        # cancel and claim the not-yet-launched clusters.
        num_tiles = _ceil_div(m, int(record["pair_m"])) * _ceil_div(
            n, int(record["tile_n"])
        )
        return (2 * num_tiles, 1, 1)
    if rule == "fp8_pairs":
        num_tiles = _ceil_div(m, int(record["pair_m"])) * _ceil_div(
            n, int(record["tile_n"])
        )
        if record["sched"] == "clc":
            # one (2, 1, 1) cluster per tile; running pairs cancel and claim the not-yet-launched clusters
            return (2 * num_tiles, 1, 1)
        return (2 * min(sm_count // 2, num_tiles), 1, 1)
    if rule == "sm90_persistent":
        cluster_m = int(record["cluster_m"])
        return (sm_count - sm_count % cluster_m, 1, 1)
    raise RuntimeError(f"unsupported generated grid rule {rule!r}")


def select_cake_quant_lora_template(
    *,
    family: str,
    target: CakeQuantLoraTarget,
    m: int,
    n: int,
    k: int,
    rank: int,
    sm_count: int,
    adapter_state: str = "raw",
) -> str:
    """Return the generated program for a problem, or raise ``ValueError`` naming the violated requirement.

    ``adapter_state`` is the call's API form (``ADAPTER_STATES``): ``"none"`` and ``"raw"`` run the family's per-problem
    rule over the raw-path programs (today's dispatch, unchanged); ``"prepared"`` (NVFP4 only) runs the prepared-path
    program -- the pre-switch of the Cake dispatch -- whenever the problem is at or above that program's ``min_k``
    floor.  The shape rules are shared by every program of a family on an architecture.
    """
    if adapter_state not in ADAPTER_STATES:
        raise ValueError(
            f"adapter_state must be one of {ADAPTER_STATES}, got {adapter_state!r}"
        )
    label = "NVFP4" if family == FAMILY_NVFP4 else "FP8"
    candidates = _templates_for(family, target, adapter_state)
    if not candidates:
        if adapter_state == "prepared":
            raise ValueError(
                f"the Cake {label} LoRA GEMM has no prepared-adapter program for {TARGET_ARCH[target]} (lora_adapter= "
                "is an NVFP4 Blackwell path; pass lora_b + lora_scale for the raw path)"
            )
        raise ValueError(
            f"the Cake {label} LoRA GEMM has no generated program for {TARGET_ARCH[target]}"
        )
    if rank not in SUPPORTED_RANKS:
        raise ValueError(
            f"the LoRA rank (d.shape[1] == lora_b.shape[1]) must be one of {SUPPORTED_RANKS}, got rank={rank}"
        )
    # Every program of one family on one architecture carries the same shape rules (the delivered test asserts it),
    # so the requirements are checked before the per-problem program choice (NVFP4: the per-row dispatch rule; FP8
    # Blackwell: the pair-tile width, then the rule-gated L2-locality point on SM100 / SM103 and the pair-tile height on
    # SM103; SM90: the rule-gated program on long runs of large-K panels).
    record = PROGRAMS[candidates[0]]
    rules = record["rules"]
    if m < int(rules["m_min"]):
        raise ValueError(
            f"the Cake {label} LoRA GEMM requires M >= {rules['m_min']}, got M={m}"
        )
    n_multiple = int(rules["n_multiple"])
    if n < 1 or n % n_multiple:
        raise ValueError(
            f"the Cake {label} LoRA GEMM on {TARGET_ARCH[target]} requires N to be a positive multiple of "
            f"{n_multiple}, got N={n}"
        )
    k_multiple = int(rules["k_multiple"])
    if k < 1 or k % k_multiple:
        raise ValueError(
            f"the Cake {label} LoRA GEMM on {TARGET_ARCH[target]} requires K to be a positive multiple of "
            f"{k_multiple}, got K={k}"
        )
    max_mn = rules["max_mn"]
    if max_mn is not None and m * n > int(max_mn):
        raise ValueError(
            f"the Cake {label} LoRA GEMM on {TARGET_ARCH[target]} requires M * N <= {max_mn} "
            f"(32-bit output indexing), got M*N={m * n}"
        )
    if adapter_state == "prepared":
        # The pre-switch: one prepared-path program per architecture, refused below its K floor (the Cake dispatch's
        # ``min_k`` of that kernel) so no prepared call reaches a problem the program was not admitted for.
        if len(candidates) != 1:
            raise ValueError(
                f"expected one prepared-adapter program for {TARGET_ARCH[target]}, delivered: {candidates}"
            )
        (template,) = candidates
        prepared = PROGRAMS[template]
        min_k = int(
            prepared["dispatch"][TARGET_ARCH[target]]["kernels"][
                prepared["dispatch_kernel"]
            ]["min_k"]
        )
        if k < min_k:
            raise ValueError(
                f"the prepared-adapter path ({template}) requires K >= {min_k}, got K={k}; pass lora_b + lora_scale "
                "for the raw path"
            )
        return template
    if len(candidates) == 1:
        return candidates[0]
    if record["grid"] == "sm90_persistent":
        return sm90_fp8_lora_program(candidates, record, m, n, k, sm_count)
    if family == FAMILY_NVFP4:
        return nvfp4_blackwell_program(candidates, TARGET_ARCH[target], m, n, k, rank)
    return fp8_blackwell_program(
        candidates, record, TARGET_ARCH[target], m, n, k, sm_count
    )


def _lora_carrier(record: dict, device: torch.device) -> torch.Tensor:
    """Valid bf16 tensor-map carrier for ``d`` / ``lora_b`` without an adapter (never dereferenced)."""
    rows, cols = (int(value) for value in record["lora_carrier"])
    nbytes = rows * cols * 2
    buffer = _get_cache_buf(_CARRIER_BUFFER, nbytes, device, zero_init=True)
    return buffer[:nbytes].view(torch.bfloat16).view(rows, cols)


class StaleAdapterError(ValueError):
    """A prepared adapter's source tensor (``lora_b`` / ``lora_scale`` / ``alpha``) moved since its prepare (snapshot
    semantics): re-prepare (``prepare_nvfp4_lora_adapter(..., out=adapter)``) or capture the refold inside the graph."""


def _pin(tensor: torch.Tensor) -> tuple:
    return (
        int(tensor.data_ptr()),
        int(tensor._version),
        tuple(tensor.shape),
        str(tensor.device),
    )


def _pins(lora_b: torch.Tensor, lora_scale: torch.Tensor, alpha: torch.Tensor) -> dict:
    return {
        "lora_b": _pin(lora_b),
        "lora_scale": _pin(lora_scale),
        "alpha": _pin(alpha),
    }


@dataclasses.dataclass(eq=False)
class PreparedNvFp4LoraAdapter:
    r"""A LoRA adapter prepared for the NVFP4 prepared path of :func:`mm_nvfp4_lora` (``lora_adapter=``).

    ``b_folded`` = ``bf16(lora_b * lora_scale / alpha)`` (``[N, rank]`` bf16: ONE bf16 rounding of the fp32 product of
    ``lora_b`` and the fp32 ratio ``div.rn.f32(fp32(lora_scale), fp32(alpha))`` -- bit-identical to the ``l1`` fold of
    :func:`~flashinfer.mm_nvfp4_svdquant`).  ``lora_scale`` is the adapter's own device scalar; the kernel reads it on
    the device (``lora_scale == 0`` takes the zero-adapter path, bitwise the adapter-off call).  Snapshot semantics: the
    adapter keeps references to its source tensors ``lora_b`` / ``lora_scale`` / ``alpha`` and ``pins`` = their
    ``(data_ptr, _version, shape, device)`` at prepare time; every call compares these host integers (no sync) and
    raises :class:`StaleAdapterError` naming the tensor that moved without a re-prepare.  Writes that bypass torch's
    version counter (NCCL into an existing buffer, a raw memcpy) need an explicit prepare.  ``alpha`` is fixed per
    prepared adapter: the call requires the SAME ``alpha`` tensor (identity) at its pinned version -- re-prepare when
    alpha changes (no value comparison).
    """

    b_folded: torch.Tensor
    lora_b: torch.Tensor
    lora_scale: torch.Tensor
    alpha: torch.Tensor
    pins: dict
    n: int
    rank: int

    def check_fresh(self) -> None:
        """Raise :class:`StaleAdapterError` when a source tensor moved since the prepare (host integers only, no sync)."""
        for name in ("lora_b", "lora_scale", "alpha"):
            now = _pin(getattr(self, name))
            if now != self.pins[name]:
                raise StaleAdapterError(
                    f"prepared adapter is stale: {name} moved since prepare (data_ptr, version, shape, device) "
                    f"{self.pins[name]} -> {now}; call prepare_nvfp4_lora_adapter(lora_b, lora_scale, alpha, out=adapter) "
                    "again (or capture that refold inside the graph); writes that bypass torch need an explicit prepare"
                )


def _require_prepared_tensor(
    tensor, *, name: str, shape: tuple, dtype: torch.dtype, device: torch.device
) -> None:
    if tensor is None:
        raise ValueError(f"{name} must not be None")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    if tensor.device != device:
        raise ValueError(
            f"{name} must be on the same device as lora_b ({device}), got {tensor.device}"
        )


@flashinfer_api
def prepare_nvfp4_lora_adapter(
    lora_b: torch.Tensor,
    lora_scale: torch.Tensor,
    alpha: torch.Tensor,
    *,
    out: Optional[PreparedNvFp4LoraAdapter] = None,
) -> PreparedNvFp4LoraAdapter:
    r"""Prepare (fold) one LoRA adapter for the NVFP4 prepared path: ``b_folded = bf16(lora_b * lora_scale / alpha)``.

    This is the adapter-switch cost of the prepared path (one small elementwise kernel, counted as the ``l1`` fold of
    :func:`~flashinfer.mm_nvfp4_svdquant` is counted); the per-call path then does no host read and no fold.  The fold
    kernel is PyTorch's own elementwise ``mul`` (bf16 x fp32 scalar in fp32 opmath, IEEE round-to-nearest-even to bf16;
    PyTorch is built without ``--use_fast_math``): ``ratio = div.rn.f32(fp32(lora_scale), fp32(alpha[0]))``,
    ``b_folded[i] = bf16_rne(fp32(lora_b[i]) * ratio)``.

    Parameters
    ----------
    lora_b : torch.Tensor
        LoRA-up weight, ``bf16 [N, r]`` row-major, contiguous (the raw path's ``lora_b``).
    lora_scale : torch.Tensor
        fp32 device tensor of shape ``(1,)`` -- ``s_lora`` as on the raw path (``0`` folds to the zero adapter).
    alpha : torch.Tensor
        fp32 device tensor of shape ``(1,)``: the base scale ``alpha_base / (a_global_scale * b_global_scale)`` the
        call passes as ``alpha`` -- the same tensor object (alpha is fixed per prepared adapter).  ``alpha == 0`` is
        refused (``s / alpha`` is undefined); the alpha == 0 LoRA-only call is the raw path.
    out : PreparedNvFp4LoraAdapter, optional
        Refold IN PLACE into ``out.b_folded`` with NO host read (graph-capturable; capturing this refold plus the call
        is the adapter-switch protocol inside a CUDA graph): the same fp32 ratio computed on the device, the same fp32
        product and the same single bf16 rounding -- bitwise the eager form.  ``alpha`` must be ``out``'s own alpha
        tensor; ``lora_scale`` may be a new tensor (the adapter then carries it); ``lora_b`` must keep ``out``'s
        ``[N, r]`` and device.  The source pins are refreshed.

    Returns
    -------
    PreparedNvFp4LoraAdapter
        A new adapter (``out=None``: the single documented host read of the adapter switch -- one device-to-host copy
        of ``(lora_scale, alpha)`` -- checks ``alpha != 0`` and a finite ratio, else ``ValueError``), or ``out``.
    """
    if lora_b is None or lora_scale is None or alpha is None:
        raise ValueError(
            "prepare_nvfp4_lora_adapter needs lora_b, lora_scale and alpha"
        )
    if (
        lora_b.dim() != 2
        or lora_b.dtype != torch.bfloat16
        or not lora_b.is_contiguous()
    ):
        raise ValueError(
            f"lora_b must be a contiguous bf16 [N, rank] tensor, got {tuple(lora_b.shape)} {lora_b.dtype}"
        )
    n, rank = (int(value) for value in lora_b.shape)
    device = lora_b.device
    _require_prepared_tensor(
        lora_scale, name="lora_scale", shape=(1,), dtype=torch.float32, device=device
    )
    _require_prepared_tensor(
        alpha, name="alpha", shape=(1,), dtype=torch.float32, device=device
    )
    if out is None:
        pair = torch.stack(
            (lora_scale.reshape(()), alpha.reshape(()))
        ).cpu()  # the ONE host read of the switch
        if float(pair[1]) == 0.0:
            raise ValueError(
                "prepare_nvfp4_lora_adapter: alpha == 0 has no folded operand (s / alpha is undefined); the alpha == 0 "
                "LoRA-only path is the raw lora_b + lora_scale call"
            )
        ratio = (
            pair[0] / pair[1]
        ).item()  # fp32 division on the host: div.rn.f32(fp32(s), fp32(alpha[0]))
        if not math.isfinite(ratio):
            raise ValueError(
                f"prepare_nvfp4_lora_adapter: s / alpha = {float(pair[0])} / {float(pair[1])} is not finite"
            )
        b_folded = torch.mul(
            lora_b, ratio
        )  # ONE kernel: bf16 x fp32 scalar in fp32 opmath, one RNE to bf16
        return PreparedNvFp4LoraAdapter(
            b_folded=b_folded.contiguous(),
            lora_b=lora_b,
            lora_scale=lora_scale,
            alpha=alpha,
            pins=_pins(lora_b, lora_scale, alpha),
            n=n,
            rank=rank,
        )
    if not isinstance(out, PreparedNvFp4LoraAdapter):
        raise TypeError(
            f"out must be a PreparedNvFp4LoraAdapter, got {type(out).__name__}"
        )
    if (out.n, out.rank) != (n, rank) or out.b_folded.device != device:
        raise ValueError(
            f"out was prepared for [N, rank] = {(out.n, out.rank)} on {out.b_folded.device}; refold needs the same "
            f"geometry and device, got {(n, rank)} on {device}"
        )
    if alpha is not out.alpha:
        raise ValueError(
            "alpha is fixed per prepared adapter: refold with out.alpha, or prepare a new adapter"
        )
    ratio_t = lora_scale.reshape(()) / alpha.reshape(
        ()
    )  # fp32 device division (IEEE), no host read
    out.b_folded.copy_(
        lora_b.float() * ratio_t
    )  # fp32 product, one RNE to bf16 -- bitwise the eager form
    out.lora_b = lora_b
    out.lora_scale = lora_scale
    out.pins = _pins(lora_b, lora_scale, alpha)
    return out


def _prepared_adapter_operands(
    adapter, *, d, alpha
) -> tuple[torch.Tensor, torch.Tensor]:
    """The LoRA operands ``(b_folded, lora_scale)`` of a prepared-path call after its host checks (no device read):
    the adapter's type, ``d`` present, the call's ``alpha`` is the adapter's own tensor, and the snapshot is fresh."""
    if not isinstance(adapter, PreparedNvFp4LoraAdapter):
        raise TypeError(
            f"lora_adapter must be a PreparedNvFp4LoraAdapter, got {type(adapter).__name__}"
        )
    if d is None:
        raise ValueError(
            "the prepared path needs d (the LoRA-down output) beside lora_adapter"
        )
    if adapter.alpha is not alpha:
        raise ValueError(
            "lora_adapter was prepared against a different alpha tensor: re-prepare when alpha changes"
        )
    adapter.check_fresh()
    return adapter.b_folded, adapter.lora_scale


def _check_adapter_and_output(
    *,
    m: int,
    n: int,
    anchor: torch.Tensor,
    d: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    lora_scale: Optional[torch.Tensor],
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    enable_pdl: Optional[bool],
) -> None:
    if (d is None) != (lora_b is None):
        raise ValueError(
            "d and lora_b must be given together (an adapter) or both be None (no adapter)"
        )
    if d is not None:
        if d.ndim != 2 or d.shape[0] != m:
            raise ValueError(
                f"d must have shape [M, r] = [{m}, r] (bf16 LoRA-down output), got {tuple(d.shape)}"
            )
        rank = int(d.shape[1])
        if rank not in SUPPORTED_RANKS:
            raise ValueError(
                f"the LoRA rank (d.shape[1] == lora_b.shape[1]) must be one of {SUPPORTED_RANKS}, got rank={rank}"
            )
        if lora_b.ndim != 2 or tuple(lora_b.shape) != (n, rank):
            raise ValueError(
                f"lora_b must have shape [N, r] = [{n}, {rank}] (bf16 LoRA-up weight), got {tuple(lora_b.shape)}"
            )
        if d.dtype != torch.bfloat16 or lora_b.dtype != torch.bfloat16:
            raise ValueError("d and lora_b must be bf16")
        if not d.is_contiguous() or not lora_b.is_contiguous():
            raise ValueError("d and lora_b must be contiguous row-major")
    if lora_scale is not None and (
        lora_scale.dtype != torch.float32 or lora_scale.numel() != 1
    ):
        raise ValueError(
            "lora_scale must be a one-element float32 device tensor (or None to disable the adapter)"
        )
    if bias is not None:
        if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (n,):
            raise ValueError(
                f"bias must have shape ({n},) and dtype bf16, got {tuple(bias.shape)} {bias.dtype}"
            )
        if not bias.is_contiguous():
            raise ValueError("bias must be contiguous")
    if out_dtype is not torch.bfloat16:
        raise ValueError(
            f"out_dtype must be torch.bfloat16 (the generated programs round once to bf16), got {out_dtype}"
        )
    if out is not None:
        if tuple(out.shape) != (m, n) or out.dtype != torch.bfloat16:
            raise ValueError(
                f"out must have shape ({m}, {n}) and dtype bf16, got {tuple(out.shape)} {out.dtype}"
            )
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
    if enable_pdl not in (None, True):
        raise ValueError(
            "the Cake programs are built with programmatic dependent launch; enable_pdl must be None or True"
        )
    if not anchor.is_cuda:
        raise ValueError("the Cake LoRA GEMM requires CUDA tensors")
    for name, tensor in (
        ("d", d),
        ("lora_b", lora_b),
        ("lora_scale", lora_scale),
        ("bias", bias),
        ("out", out),
    ):
        if tensor is not None and tensor.device != anchor.device:
            raise ValueError(
                f"{name} must be on the same device as a ({anchor.device}), got {tensor.device}"
            )


def _check_mm_nvfp4_lora_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    lora_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cake", "auto"] = "cake",
    enable_pdl: Optional[bool] = None,
    lora_adapter: Optional[PreparedNvFp4LoraAdapter] = None,
):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D packed-e2m1 (uint8) tensors")
    if a.dtype != torch.uint8 or b.dtype != torch.uint8:
        raise ValueError("a and b must be uint8 (two e2m1 values per byte)")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous row-major")
    m, k_packed = (int(value) for value in a.shape)
    n = int(b.shape[0])
    if int(b.shape[1]) != k_packed:
        raise ValueError(
            f"a and b inner dimensions mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}"
        )
    k = k_packed * 2
    if a_sf.dtype != torch.uint8 or b_sf.dtype != torch.uint8:
        raise ValueError(
            "a_sf and b_sf must be uint8 (ue4m3 block scales in the 128x4 swizzled layout)"
        )
    if not a_sf.is_contiguous() or not b_sf.is_contiguous():
        raise ValueError("a_sf and b_sf must be contiguous")
    sf_cols = k // _SF_BLOCK
    need_a = _ceil_div(m, _SF_SWIZZLE_ROWS) * _SF_SWIZZLE_ROWS * sf_cols
    need_b = _ceil_div(n, _SF_SWIZZLE_ROWS) * _SF_SWIZZLE_ROWS * sf_cols
    if a_sf.numel() < need_a or b_sf.numel() < need_b:
        raise ValueError(
            "128x4 swizzled scale buffers are too small: "
            f"a_sf has {a_sf.numel()} elements (need {need_a}), b_sf has {b_sf.numel()} elements (need {need_b})"
        )
    if alpha.dtype != torch.float32 or alpha.numel() < 1:
        raise ValueError(
            "alpha must be a non-empty float32 device tensor (its first element is alpha_base)"
        )
    for name, tensor in (("b", b), ("a_sf", a_sf), ("b_sf", b_sf), ("alpha", alpha)):
        if tensor.device != a.device:
            raise ValueError(
                f"{name} must be on the same device as a ({a.device}), got {tensor.device}"
            )
    # Routing by API form (no device read): the prepared path binds the adapter's folded operand and its own scale;
    # its host checks (type, d present, alpha identity, snapshot freshness) precede the shared geometry checks.
    if _adapter_state(lora_b, lora_scale, lora_adapter) == "prepared":
        lora_b, lora_scale = _prepared_adapter_operands(lora_adapter, d=d, alpha=alpha)
    _check_adapter_and_output(
        m=m,
        n=n,
        anchor=a,
        d=d,
        lora_b=lora_b,
        lora_scale=lora_scale,
        bias=bias,
        out=out,
        out_dtype=out_dtype,
        enable_pdl=enable_pdl,
    )
    return True


def _check_mm_fp8_lora_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    lora_scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cake", "auto"] = "cake",
    enable_pdl: Optional[bool] = None,
):
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2-D float8_e4m3fn tensors")
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError("a and b must be float8_e4m3fn")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a and b must be contiguous row-major")
    m, k = (int(value) for value in a.shape)
    n = int(b.shape[0])
    if int(b.shape[1]) != k:
        raise ValueError(
            f"a and b inner dimensions mismatch: a {tuple(a.shape)} vs b {tuple(b.shape)}"
        )
    for name, scale in (("a_scale", a_scale), ("b_scale", b_scale)):
        if scale.dtype != torch.float32 or scale.numel() != 1:
            raise ValueError(f"{name} must be a one-element float32 device tensor")
    for name, tensor in (("b", b), ("a_scale", a_scale), ("b_scale", b_scale)):
        if tensor.device != a.device:
            raise ValueError(
                f"{name} must be on the same device as a ({a.device}), got {tensor.device}"
            )
    _check_adapter_and_output(
        m=m,
        n=n,
        anchor=a,
        d=d,
        lora_b=lora_b,
        lora_scale=lora_scale,
        bias=bias,
        out=out,
        out_dtype=out_dtype,
        enable_pdl=enable_pdl,
    )
    return True


def _select_for_call(
    family: str,
    a: torch.Tensor,
    b: torch.Tensor,
    d: Optional[torch.Tensor],
    adapter_state: str,
) -> tuple[str, CakeQuantLoraTarget, int]:
    target = cake_quant_lora_target(a.device)
    m = int(a.shape[0])
    n = int(b.shape[0])
    k = int(a.shape[1]) * (2 if family == FAMILY_NVFP4 else 1)
    rank = int(d.shape[1]) if d is not None else _NO_ADAPTER_RANK
    sm_count = get_device_sm_count(a.device)
    template = select_cake_quant_lora_template(
        family=family,
        target=target,
        m=m,
        n=n,
        k=k,
        rank=rank,
        sm_count=sm_count,
        adapter_state=adapter_state,
    )
    return template, target, sm_count


@supported_compute_capability([100, 103])
def _cake_nvfp4_lora_requirement(
    a,
    b,
    a_sf,
    b_sf,
    alpha,
    d=None,
    lora_b=None,
    lora_scale=None,
    *,
    lora_adapter=None,
    **kwargs,
):
    _select_for_call(
        FAMILY_NVFP4, a, b, d, _adapter_state(lora_b, lora_scale, lora_adapter)
    )
    return True


@supported_compute_capability([90, 100, 103])
def _cake_fp8_lora_requirement(
    a, b, a_scale, b_scale, d=None, lora_b=None, lora_scale=None, **kwargs
):
    _select_for_call(FAMILY_FP8, a, b, d, _adapter_state(lora_b, lora_scale, None))
    return True


def _cake_only_heuristic(suitable_backends, *args, **kwargs):
    return list(suitable_backends)


def _run(
    family: str,
    *,
    a: torch.Tensor,
    b: torch.Tensor,
    base_scales: dict[str, torch.Tensor],
    d: Optional[torch.Tensor],
    lora_b: Optional[torch.Tensor],
    lora_scale: Optional[torch.Tensor],
    lora_adapter: Optional[PreparedNvFp4LoraAdapter],
    bias: Optional[torch.Tensor],
    out: Optional[torch.Tensor],
    enable_pdl: Optional[bool],
    scale_views: dict[str, torch.Tensor],
) -> torch.Tensor:
    device = a.device
    adapter_state = _adapter_state(lora_b, lora_scale, lora_adapter)
    template, target, sm_count = _select_for_call(family, a, b, d, adapter_state)
    record = PROGRAMS[template]
    if enable_pdl is not None and enable_pdl is not bool(record["pdl"]):
        raise ValueError(
            f"the selected Cake program requires enable_pdl={record['pdl']}, got {enable_pdl}"
        )
    m = int(a.shape[0])
    n = int(b.shape[0])
    k = int(a.shape[1]) * (2 if family == FAMILY_NVFP4 else 1)
    has_bias = bias is not None
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    if adapter_state == "prepared":
        # The prepared path: the adapter's folded operand and its own scale are the LoRA operands (the kernel reads
        # the scale on the device; lora_scale == 0 takes its zero-adapter path), no fold and no host read per call.
        has_lora = True
        rank = int(d.shape[1])
        d_arg, b_arg, scale_arg = (
            d,
            lora_adapter.b_folded,
            lora_adapter.lora_scale.reshape(1),
        )
    else:
        # lora_scale=None disables the adapter even when d / lora_b are passed (the zero-adapter fast path,
        # bitwise identical to lora_scale == 0 and to the all-None call).
        has_lora = d is not None and lora_b is not None and lora_scale is not None
        rank = int(d.shape[1]) if d is not None else _NO_ADAPTER_RANK
        if has_lora:
            d_arg, b_arg, scale_arg = d, lora_b, lora_scale.reshape(1)
        else:
            # The typed pointer / tensor-map ABI needs valid carriers; has_lora=0 keeps them unread.
            carrier = _lora_carrier(record, device)
            d_arg, b_arg, scale_arg = carrier, carrier, base_scales["carrier"]
    bias_arg = bias if has_bias else out
    grid = launch_grid(record, m, n, sm_count)
    block_k = int(record["block_k"])
    if record["operand_view"] == "uint8":
        a_view, b_view = a.view(torch.uint8), b.view(torch.uint8)
    else:
        a_view, b_view = a, b
    tensors: dict[str, torch.Tensor] = {
        "A": a_view,
        "B": b_view,
        "W": b_view,
        "D": d_arg,
        "LORA_B": b_arg,
        "LORA_B_PRE": b_arg,
        "LORA_B_TAIL": b_arg,
        "BL": b_arg,
        "lora_scale": scale_arg,
        "bias": bias_arg,
        "out": out,
        **scale_views,
    }
    params: dict[str, int] = {
        "M": m,
        "N": n,
        "K": k,
        "K_tiles": k // block_k,
        "rank": rank,
        "has_bias": 1 if has_bias else 0,
        "has_lora": 1 if has_lora else 0,
        "has_adapter": 1 if has_lora else 0,
    }
    rule = record["grid"]
    if rule == "nvfp4_pairs":
        tok_tiles = _ceil_div(m, int(record["pair_m"]))
        w_tiles = _ceil_div(n, int(record["tile_n"]))
        params["tok_tiles"] = tok_tiles
        params["w_tiles"] = w_tiles
        params["num_tiles"] = tok_tiles * w_tiles
        params["raster_group"] = nvfp4_blackwell_raster_group(
            record, TARGET_ARCH[target], m, n, k, rank
        )
    elif rule == "fp8_pairs":
        tok_tiles = _ceil_div(m, int(record["pair_m"]))
        w_tiles = _ceil_div(n, int(record["tile_n"]))
        params["tok_tiles"] = tok_tiles
        params["w_tiles"] = w_tiles
        params["num_tiles"] = tok_tiles * w_tiles
        params["raster_group"] = fp8_blackwell_raster_group(record, m, n, sm_count)
    elif rule == "sm90_persistent":
        params["group_m"] = int(record["group_m"])
    values = []
    for kind, key in record["arg_plan"]:
        if kind in ("buffer", "tma_buffer"):
            values.append(tensors[key])
        elif kind == "parameter":
            values.append(params[key])
        elif kind == "grid":
            values.append(grid[{"grid_x": 0, "grid_y": 1, "grid_z": 2}[key]])
        else:
            raise RuntimeError(f"unsupported generated argument {kind}:{key}")
    cake_quant_lora_launcher(template, target)(*values)
    return out


@backend_requirement(
    {"cake": _cake_nvfp4_lora_requirement},
    common_check=_check_mm_nvfp4_lora_problem,
    heuristic_func=_cake_only_heuristic,
)
@flashinfer_api
def mm_nvfp4_lora(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    lora_scale: Optional[torch.Tensor] = None,
    *,
    lora_adapter: Optional[PreparedNvFp4LoraAdapter] = None,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cake", "auto"] = "cake",
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""NVFP4 base GEMM with a runtime BF16 LoRA correction.

    ``out = alpha * (dequant(a) @ dequant(b)^T) + lora_scale * (d @ lora_b^T) [+ bias]``

    The NVFP4 operands are consumed exactly as :func:`~flashinfer.mm_fp4` does
    (E2M1 codes packed two per byte, UE4M3 block scales of 16 along K in the
    128x4 swizzled layout produced by :func:`~flashinfer.fp4_quantize`); the
    correction is computed in fp32 inside the same kernel and the result is
    rounded once to bf16.  The adapter arrives in one of two forms:

    * **Raw path** (``lora_b`` + ``lora_scale``): the BF16 factors are consumed as
      given -- ``d`` (the LoRA-down output ``x @ A^T``) and ``lora_b`` (the LoRA-up
      weight) are never re-rounded or pre-scaled -- and ``alpha`` / ``lora_scale``
      are independent fp32 device scalars (no folding of either into the weights).
      Swapping the adapter is swapping pointers (0-preparation switching).
    * **Prepared path** (``lora_adapter=``, from :func:`prepare_nvfp4_lora_adapter`):
      the adapter was folded once at the switch, ``b_folded = bf16(lora_b *
      lora_scale / alpha)``; the kernel accumulates the base product first and
      applies ``alpha`` once in the fp32 epilogue -- the numerics of
      :func:`~flashinfer.mm_nvfp4_svdquant` (``lora_b`` re-rounded once to bf16; on
      the measured contract rows SQNR 53.8 dB vs 55.6 dB on the raw path, max abs
      error vs fp64 0.038-0.042 vs 0.031).  Every prepared call runs the
      prepared-path program (the raw dispatch table is untouched) and keeps the
      snapshot checks of :class:`PreparedNvFp4LoraAdapter`.  Passing both forms
      raises ``ValueError``; neither form is the adapter-off call.

    Parameters
    ----------
    a : torch.Tensor
        Quantized activations, ``uint8 [M, K // 2]`` (two E2M1 codes per byte), row-major.
    b : torch.Tensor
        Quantized base weights, ``uint8 [N, K // 2]``, row-major (``a @ b^T``).
    a_sf, b_sf : torch.Tensor
        UE4M3 block scales of ``a`` / ``b`` as ``uint8`` in the 128x4 swizzled layout
        (rows padded to 128); larger contiguous backing buffers are accepted.
    alpha : torch.Tensor
        fp32 device tensor whose first element is the base scale
        ``alpha_base / (a_global_scale * b_global_scale)``.
    d : torch.Tensor or None
        LoRA-down output, ``bf16 [M, r]`` row-major, contiguous (16-byte aligned rows).
    lora_b : torch.Tensor or None
        LoRA-up weight, ``bf16 [N, r]`` row-major, contiguous.  A ``[r, N]`` factor
        must be transposed and made contiguous by the caller.
    lora_scale : torch.Tensor or None
        One-element fp32 device tensor ``s_lora`` (the LoRA ``alpha / r`` and any other
        multiplier), independent of ``alpha``.  ``0`` and ``None`` both take the zero-adapter
        fast path (no LoRA work, bitwise identical outputs): ``lora_scale=None`` disables the
        adapter even if ``d`` / ``lora_b`` are passed; pass a device scalar to apply it.
    lora_adapter : PreparedNvFp4LoraAdapter, optional
        The prepared path: an adapter from :func:`prepare_nvfp4_lora_adapter` (``lora_b`` and
        ``lora_scale`` must then be ``None``; ``d`` is still required).  Its ``alpha`` must be
        this call's ``alpha`` tensor (re-prepare when alpha changes) and its source tensors
        must not have moved since the prepare (:class:`StaleAdapterError` otherwise).  A
        prepared adapter with ``lora_scale == 0`` produces the adapter-off output, bitwise.
    bias : torch.Tensor, optional
        ``bf16 [N]``, added in fp32 before the single rounding.
    out : torch.Tensor, optional
        Caller-owned ``bf16 [M, N]`` output; allocated when ``None``.
    out_dtype : torch.dtype
        Only ``torch.bfloat16`` is supported.
    backend : {"cake", "auto"}
        The generated Cake programs are the only implementation.
    enable_pdl : bool, optional
        The programs are built with programmatic dependent launch; ``None`` or ``True``.

    Returns
    -------
    torch.Tensor
        ``out`` (``bf16 [M, N]``).

    Notes
    -----
    * Architectures: SM100 (B200) and SM103 (B300) only; SM120 / SM121 are not
      supported (the kernel uses tcgen05 / TMEM).
    * Shape rules: ``M >= 1``, ``N >= 1``, ``K`` a multiple of 256; the LoRA rank
      ``r = d.shape[1]`` must be one of ``(32, 64, 128)``.  Violations raise
      ``ValueError`` naming the violated requirement.
    * Tensor parallelism (host contract): column-parallel (``N`` sharded) ranks each
      hold their ``N`` shard of ``b`` / ``b_sf`` / ``lora_b`` / ``bias`` and the full
      ``d``, and each adds its own correction; row-parallel (``K`` sharded) ranks hold
      their ``K`` shard of ``a`` / ``b`` (codes and scales), the full ``d`` (reduced
      by the caller or computed from the full ``x``) and the full ``lora_b``, and only
      the owner rank passes ``lora_scale`` / ``bias`` (the others pass ``lora_scale = 0``
      and ``bias = None``) so the correction is counted exactly once before the
      all-reduce of the base partials.  On the prepared path every rank prepares its
      own adapter from its shard (the non-owner ranks of a row-parallel layout with
      ``lora_scale = 0``, which folds to the zero adapter).
    * CUDA graphs, raw path: a captured call keeps the adapter *pointers* captured.
      Swapping ``d`` / ``lora_b`` / ``lora_scale`` for new tensors after capture does not
      affect replays; write the new adapter into the captured tensors (``copy_``) instead.
      Prepared path: a captured call keeps the prepared adapter captured; the adapter
      switch is ``prepare_nvfp4_lora_adapter(lora_b, lora_scale, alpha, out=adapter)``
      captured right before the call (an in-place refold without a host read), after
      which a ``copy_`` into the captured ``lora_b`` / ``lora_scale`` is folded by every
      replay.  Outside a graph, a ``copy_`` into a source tensor makes the adapter stale
      (``StaleAdapterError``) until it is re-prepared.
    """
    if backend not in ("cake", "auto"):
        raise ValueError(
            f"unsupported backend {backend!r}; the Cake programs are the only implementation"
        )
    alpha_scalar = alpha.reshape(-1)[:1]
    m, k_packed = (int(value) for value in a.shape)
    n = int(b.shape[0])
    k = k_packed * 2
    sf_cols = k // _SF_BLOCK
    # Every NVFP4 program shares the scale-operand form: 128-row atoms of (K / MMA_K) 512-byte sets viewed as
    # uint32 words, one TMA box per 128-row x K=256 scale block (the delivered test asserts the shared form).
    record = PROGRAMS[_templates_for(FAMILY_NVFP4, cake_quant_lora_target(a.device))[0]]
    sf_rows = int(record["sf_row_pad"])
    sf_sets = k // int(record["sf_mma_k"])
    padded_m = _ceil_div(m, sf_rows) * sf_rows
    padded_n = _ceil_div(n, sf_rows) * sf_rows
    a_sf_view = (
        a_sf.reshape(-1)[: padded_m * sf_cols]
        .view(torch.uint32)
        .view(padded_m // sf_rows, sf_sets, 128)
    )
    b_sf_view = (
        b_sf.reshape(-1)[: padded_n * sf_cols]
        .view(torch.uint32)
        .view(padded_n // sf_rows, sf_sets, 128)
    )
    return _run(
        FAMILY_NVFP4,
        a=a,
        b=b,
        base_scales={"carrier": alpha_scalar},
        d=d,
        lora_b=lora_b,
        lora_scale=lora_scale,
        lora_adapter=lora_adapter,
        bias=bias,
        out=out,
        enable_pdl=enable_pdl,
        scale_views={"SFA": a_sf_view, "SFB": b_sf_view, "alpha": alpha_scalar},
    )


@backend_requirement(
    {"cake": _cake_fp8_lora_requirement},
    common_check=_check_mm_fp8_lora_problem,
    heuristic_func=_cake_only_heuristic,
)
@flashinfer_api
def mm_fp8_lora(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    d: Optional[torch.Tensor] = None,
    lora_b: Optional[torch.Tensor] = None,
    lora_scale: Optional[torch.Tensor] = None,
    *,
    bias: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    backend: Literal["cake", "auto"] = "cake",
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    r"""FP8 (E4M3, per-tensor scales) base GEMM with a runtime BF16 LoRA correction.

    ``out = a_scale * b_scale * (dequant(a) @ dequant(b)^T) + lora_scale * (d @ lora_b^T) [+ bias]``

    The base operands are consumed as :func:`~flashinfer.mm_fp8` consumes them
    (``float8_e4m3fn`` codes with one fp32 scale each; a global ``alpha_base`` is
    carried in ``b_scale``).  The correction is computed from the BF16 factors as
    given -- never re-rounded or pre-scaled -- in fp32 inside the same kernel, and
    the result is rounded once to bf16.  The base scales and ``lora_scale`` are
    independent fp32 device scalars.

    Parameters
    ----------
    a : torch.Tensor
        Quantized activations, ``float8_e4m3fn [M, K]``, row-major.
    b : torch.Tensor
        Quantized base weights, ``float8_e4m3fn [N, K]``, row-major (``a @ b^T``).
    a_scale, b_scale : torch.Tensor
        One-element fp32 device tensors (per-tensor dequantization scales).
    d : torch.Tensor or None
        LoRA-down output, ``bf16 [M, r]`` row-major, contiguous (16-byte aligned rows).
    lora_b : torch.Tensor or None
        LoRA-up weight, ``bf16 [N, r]`` row-major, contiguous.  A ``[r, N]`` factor
        must be transposed and made contiguous by the caller.
    lora_scale : torch.Tensor or None
        One-element fp32 device tensor ``s_lora``, independent of the base scales.
        ``0`` and ``None`` both take the zero-adapter fast path (no LoRA work, bitwise
        identical outputs): ``lora_scale=None`` disables the adapter even if ``d`` /
        ``lora_b`` are passed; pass a device scalar to apply it.
    bias : torch.Tensor, optional
        ``bf16 [N]``, added in fp32 before the single rounding.
    out : torch.Tensor, optional
        Caller-owned ``bf16 [M, N]`` output; allocated when ``None``.
    out_dtype : torch.dtype
        Only ``torch.bfloat16`` is supported.
    backend : {"cake", "auto"}
        The generated Cake programs are the only implementation.
    enable_pdl : bool, optional
        The programs are built with programmatic dependent launch; ``None`` or ``True``.

    Returns
    -------
    torch.Tensor
        ``out`` (``bf16 [M, N]``).

    Notes
    -----
    * Architectures: SM90 (H100 / H200), SM100 (B200) and SM103 (B300); SM120 / SM121
      are not supported.
    * Shape rules: ``M >= 1``, ``N`` a positive multiple of 8; ``K`` a multiple of 128
      on SM100 / SM103 and of 16 on SM90 (where additionally ``M * N <= 2**32``); the
      LoRA rank ``r = d.shape[1]`` must be one of ``(32, 64, 128)``.  Violations raise
      ``ValueError`` naming the violated requirement.  On SM100 / SM103 the pair-tile
      width is resolved from the shape and the device SM count; it is not a caller knob.  On
      SM100 / SM103 long large-K runs (at least 10 tiles per resident CTA pair, ``K >= 16384``, a
      proper block inside the tile grid) run the dynamic-tile-scheduler program with the blocked
      raster (``fp8_blackwell_l2_point``); on SM103 the longest of them (at least 20 waves of
      512 x 256 tiles over the resident pairs, ``K >= 32768``) run it on the 512-row pair tile
      (``fp8_blackwell_tile_m``); every other call runs the static persistent program.
    * Tensor parallelism (host contract): column-parallel (``N`` sharded) ranks each
      hold their ``N`` shard of ``b`` / ``lora_b`` / ``bias`` and the full ``d``, and each
      adds its own correction; row-parallel (``K`` sharded) ranks hold their ``K`` shard
      of ``a`` / ``b``, the full ``d`` and ``lora_b``, and only the owner rank passes
      ``lora_scale`` / ``bias`` (the others pass ``lora_scale = 0`` and ``bias = None``) so
      the correction is counted exactly once before the all-reduce of the base partials.
    * CUDA graphs: a captured call keeps the adapter *pointers* captured.  Swapping
      ``d`` / ``lora_b`` / ``lora_scale`` for new tensors after capture does not affect
      replays; write the new adapter into the captured tensors (``copy_``) instead.
    """
    if backend not in ("cake", "auto"):
        raise ValueError(
            f"unsupported backend {backend!r}; the Cake programs are the only implementation"
        )
    a_scale_view = a_scale.reshape(1)
    b_scale_view = b_scale.reshape(1)
    return _run(
        FAMILY_FP8,
        a=a,
        b=b,
        base_scales={"carrier": a_scale_view},
        d=d,
        lora_b=lora_b,
        lora_scale=lora_scale,
        lora_adapter=None,
        bias=bias,
        out=out,
        enable_pdl=enable_pdl,
        scale_views={"a_scale": a_scale_view, "b_scale": b_scale_view},
    )


__all__ = [
    "ADAPTER_STATES",
    "SUPPORTED_RANKS",
    "TARGET_OF_ARCH",
    "PreparedNvFp4LoraAdapter",
    "StaleAdapterError",
    "fp8_blackwell_l2_point",
    "fp8_blackwell_program",
    "fp8_blackwell_raster_group",
    "fp8_blackwell_tile_m",
    "fp8_blackwell_tile_n",
    "nvfp4_blackwell_program",
    "nvfp4_blackwell_raster_group",
    "nvfp4_svdq_raster_group",
    "launch_grid",
    "mm_fp8_lora",
    "mm_nvfp4_lora",
    "prepare_nvfp4_lora_adapter",
    "select_cake_quant_lora_template",
    "sm90_fp8_lora_program",
]
