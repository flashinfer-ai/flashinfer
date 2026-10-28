# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

"""FROST SDPA-forward capability declarations + spec table.

One engine per architecture x phase x geometry, named

    sdpa_fwd_<phase>_sm<arch>[_d<dqk>[x<dv>]]

(dtype is NOT part of the identity: a cell's engine serves every dtype its
kernel handles — fp16 and bf16 today — via ``Capabilities.dtypes``.)

The shared analyzer (``cudnn.sdpa.graph_analyzer.analyze``) parses the graph
once into :class:`SdpaGraphFacts`; each engine's probe is a cheap field-by-field
candidate match against its :class:`Capabilities` row below. Architecture
resource feasibility remains a lowering responsibility. Adding an engine is
one ``Capabilities``/spec row plus (usually) one kernel template.

An engine is a *lowering strategy*, not a kernel: its ``lower`` hook receives
the parsed facts and returns an executor, and is free to compile one kernel,
pick among several, or chain multiple launches (the THD path already launches
an O-descriptor builder kernel before the main one). Conversely several
engines may share one template. Neither direction is 1:1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Optional, Tuple

import flashinfer.experimental.qwen_3_5_block._cudnn_frost as cudnn

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.tile_dsl.constants import SCHED_LPT, SCHED_LPT_L2, SCHED_NATURAL
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import CUTEDSL_MIN_VERSION, cutedsl_arch_requirement_error, cutedsl_state, cutedsl_too_old
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa import graph_analyzer as ga
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import (
    d256_mxfp8_pair_leg,
    SM100_SCORE_STATS_SHAPES,
    SM100_THD_PACK_GQA_SHAPES,
    SM100_PAGED_FP8_THD_SHAPES,
    d256_wide_decode_launch,
    decode_d256_q_tile,
    pack_gqa_supported,
    supports_paged_prefill_cga1,
    supports_paged_d256_pack_gqa,
    supports_nonpaged_d128_thd_prefill,
    supports_paged_d128_partial_pack_gqa,
    supports_nonpaged_d128_thd_pack_gqa,
    supports_thd_split,
    supports_scalar_kv_tail_split,
    supports_quantized_split_sink,
    supports_thd_split_sink,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import (
    SM100_BIAS_D_SHAPES,
    SM100_DENSE_ONLY_D_SHAPES,
    SM100_NATIVE_D576_SHAPE,
    SM100_FP32_VECTOR_WITH_BIAS_TILE_N,
    SM100_SMALL_D_CGAS,
    SM100_TILE_NS_BY_D_SHAPE,
    sm100_bias_mode_pick,
    sm100_bias_resident_s_kv_max,
    sm100_tile_n_pick,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_d128_family import BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import config_sm107 as _config_sm107
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import (
    KV_MASK_ADDITIVE_GRANULE_ELEMS,
    SM107_BIAS_D_SHAPES,
    SM107_EPILOGUE_GATE_SHAPES,
    SM107_F16_SMALL_D_SHAPES,
    SM107_F16_THD_SHAPES,
    SM107_FP8_THD_SHAPES,
    SM107_FP8_SPLIT_SHAPES,
    supports_fp8_d256_padded_split,
    SM107_FP8_PACK_GQA_SHAPES,
    SM107_MXFP8_THD_SHAPES,
    SM107_SCORE_STATS_SHAPES,
    SM107_SMALL_D_CGAS,
    SM107_FP32_VECTOR_WITH_BIAS_TILE_N,
    SM107_TILE_NS_BY_D_SHAPE,
    supports_native_d128_narrow_tile,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import (
    D512_FLAVOR,
    GENERAL_HEAD_TILE_MAX as _SM120_GENERAL_HEAD_TILE_MAX,
    HEAD_TILE_GRANULE as _SM120_HEAD_TILE_GRANULE,
    SEQ_Q_TILES as _SM120_SEQ_Q_TILES,
    decode_kv_stage as _sm120_decode_kv_stage,
    head_tiles as _sm120_head_tiles,
    is_decode_tile as _sm120_is_decode_tile,
    tile_domain as _sm120_tile_domain,
)

# The DSL adapters (api_dsl) and cuda.bindings are LOWERING dependencies, not
# support-check ones: importing them here would drag the CuTe DSL (~1.0 s, 357
# modules) into every process that merely asks whether an engine could serve a
# graph. ENGINE_SPECS therefore names its adapter, and _adapter() resolves it
# at build time. The SM107 target gate lazily probes the DSL once; other
# capability checks stay import-free.
_SM100 = "SdpaFwdDslSm100"
_SM120 = "SdpaFwdDslSm120"
_SM80 = "SdpaFwdDslSm80"
_SM90 = "SdpaFwdDslSm90"


def _adapter(name: str):
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd import api_dsl

    return getattr(api_dsl, name)


def _cuda_driver():
    from cuda.bindings import driver

    return driver


_LOG = logging.getLogger(__name__)

# Arch ranges the spec rows below serve, inclusive, encoded major*10 + minor as
# in engines/manifest.py. RANGES, not the exact device families that exist
# today: an sm100 kernel runs on the whole sm100 line, so enumerating members
# silently declines the parts that ship later -- Rubin (sm107) and Thor (sm110)
# are meant to reuse these kernels and an exact {(10,0), (10,3)} excluded both.
# Within the range, cc10.3+ additionally enables the fused LDTM.STAT row-max for
# MXFP8 — handled in the lowering, from the device capability.
_BLACKWELL = (100, 119)
_BLACKWELL_GEFORCE = (120, 129)


@dataclass(frozen=True)
class SdpaFwdKnobs:
    """Per-plan tuning request for the SDPA-forward engines.

    Typed fields internally; ``None`` means "no preference". Travels as
    ``PlanConfig.knobs``; each engine's :class:`Capabilities` row advertises the
    domain it honors, and the probe rejects the engine for any request outside
    that domain (a knob is honored or the engine is ineligible — never silently
    degraded).

    At the public surface every tuning field is one ``cudnn.knob_type`` of the
    shared vocabulary (``to_public`` / ``from_public``), the same dict shape a
    backend plan reports: ``tile_m/tile_n`` and ``cga`` reuse the backend's
    ``TILE_M`` / ``TILE_N`` / ``TILE_CGA_M`` (a cga of 2 IS a 2-CTA M-cluster,
    the backend's own encoding of its 2-CTA SDPA variant); ``sched_policy``,
    ``pack_gqa`` and ``split_kv`` are frontend-only types (the backend's
    split-KV counterpart, ``STREAM_K``, is an on/off mode, not a chunk count,
    hence the distinct ``SPLIT_KV``).

    Knobs are performance-only: every value computes the same function, so an
    autotuner may pick any of them. The softmax accumulator precision (the
    cc 10.7 f16x2 exponent arm) changes numerics and is therefore NOT a knob: it
    is the ``sdpa(..., softmax_precision=)`` op attribute, read from the graph
    into ``SdpaGraphFacts.softmax_precision`` and gated by each row's
    ``Capabilities.softmax_precisions`` in :func:`mismatch`.
    """

    sched_policy: Optional[int] = None  # tile-scheduler policy (SCHED_NATURAL, ...)
    tile_m: Optional[int] = None  # Q sequence tile width
    tile_n: Optional[int] = None  # KV sequence tile width
    cga: Optional[int] = None  # CTA group width used by one cooperative MMA tile
    pack_gqa: Optional[bool] = None  # Head packing for GQA/MQA
    # KV-split count: each Q tile's KV range cut into this many chunks, each
    # run by its own CTA, recombined by the split_combine pass. 1 = off.
    split_kv: Optional[int] = None

    # field name -> cudnn.knob_type member name (resolved lazily: the compiled
    # module is not importable at class-definition time in every build).
    _PUBLIC_KNOBS = (
        ("sched_policy", "SCHED_POLICY"),
        ("tile_m", "TILE_M"),
        ("tile_n", "TILE_N"),
        ("cga", "TILE_CGA_M"),
        ("pack_gqa", "PACK_GQA"),
        ("split_kv", "SPLIT_KV"),
    )

    def to_public(self) -> dict:
        """``{cudnn.knob_type: int}`` for every tuning field that is set (bools as 0/1)."""
        kt = cudnn.knob_type
        out = {}
        for field, member in self._PUBLIC_KNOBS:
            value = getattr(self, field)
            if value is not None:
                out[getattr(kt, member)] = int(value)
        return out

    @classmethod
    def from_public(cls, public: dict) -> "SdpaFwdKnobs":
        """Inverse of :meth:`to_public`. Rejects knob types this operation has no field for."""
        kt = cudnn.knob_type
        by_type = {getattr(kt, member): field for field, member in cls._PUBLIC_KNOBS}
        kwargs = {}
        for knob, value in public.items():
            knob = kt(int(knob)) if not isinstance(knob, kt) else knob
            field = by_type.get(knob)
            if field is None:
                raise ValueError(f"knob {knob.name} is not a tuning axis of the SDPA-forward engines")
            if isinstance(value, bool):
                if field != "pack_gqa":
                    raise ValueError(f"knob {knob.name} value must be an int, got {value!r}")
                value = int(value)
            if not isinstance(value, int):
                # A persisted record carries ints; "0" or 1.0 would round-trip to
                # a different native knob than the one recorded.
                raise ValueError(f"knob {knob.name} value must be an int, got {value!r}")
            if field == "pack_gqa" and value not in (0, 1):
                raise ValueError(f"knob PACK_GQA must be 0 or 1, got {value}")
            kwargs[field] = bool(value) if field == "pack_gqa" else value
        return cls(**kwargs)


@dataclass(frozen=True)
class Capabilities:
    """What one ENGINE can serve — the envelope of graphs (and, later, requested
    tuning knobs) its lowering can honor. An engine spanning several kernels
    declares the union its lowering can actually deliver. Compared
    field-by-field against SdpaGraphFacts in the probe."""

    # Arch RANGE, inclusive, encoded major*10 + minor as in engines/manifest.py.
    # A range and not a set of exact device families: an sm100 kernel runs on
    # everything in the sm100 line, so enumerating the parts that exist today
    # silently declines the ones that ship tomorrow. Rubin (sm107) and Thor
    # (sm110) are meant to reuse these kernels and an exact set excluded both.
    sm_lo: int
    sm_hi: int
    phase: str
    # Head-dim DOMAIN. One engine per arch x dtype family: the head dim is a
    # LOWERING concern — the adapter picks the smallest kernel flavor whose
    # native shape covers the graph (api_dsl._pick_flavor) — not an engine
    # identity. ``d_shapes`` is the set of NATIVE flavor shapes (d_qk, d_v)
    # that lowering picks among.
    d_shapes: frozenset
    # Envelope + alignment rule. When > 0: any graph (d_qk, d_v) componentwise
    # <= some native shape AND a multiple of this is served via TMA
    # zero-padding — the kernel's descriptors carry the ACTUAL extents, so
    # padded contraction columns load as exact zeros (S/softmax unchanged) and
    # O stores past d_v are OOB-clipped. 8 = the TMA 16-byte global-stride
    # rule at 2 bytes/elem (f16/bf16), 16 = the same rule at 1 byte/elem
    # (per-tensor FP8), 1 = no constraint (the SM80 lowering pads host-side).
    # 0 = NO envelope: exact native shapes only (MXFP8, whose SF plumbing is
    # not audited for zero-padding).
    d_pad_multiple: int = 8
    # Shapes whose kernels carry the THD leg. None = THD (when the ``thd``
    # capability is set) serves the same head-dim domain as dense — the f16
    # THD compile key carries the head dims, so THD rides the envelope. A set
    # = THD graphs must match one of these NATIVE shapes exactly: the
    # quantized rows' packed THD compile key carries no head-dim entries
    # (native-tile contract), so their envelope is dense-only.
    thd_d_shapes: Optional[frozenset] = None
    # Per-native-shape ENVELOPE FLOOR: a tuple of ``((d_qk, d_v), min_dim)``
    # pairs (a tuple, not a dict, so the row stays hashable). A shape with a
    # floor serves graphs whose head dims are in ``(min_dim, shape]`` rather
    # than ``(0, shape]``, so a kernel whose geometry is tuned for a large head
    # dim does not silently swallow a much smaller graph at a multiple-x
    # zero-padding cost. The floor applies to BOTH head dims and only to
    # ENVELOPE matches — an exact ``d_shapes`` hit always passes. Example: the
    # d512 FP8 flavor floors at 256, so it serves (256, 512] on both dims
    # (at most 2x padding) and a d256 graph is declined rather than routed
    # onto it. ``()`` = no floors.
    d_envelope_floors: tuple = ()
    dtypes: frozenset = frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16})  # cudnn.data_type, see graph_analyzer
    is_mxfp8: bool = False  # block-scale MXFP8 engine (FP8 in + per-32-block E8M0 SF)
    is_fp8: bool = False  # per-tensor FP8 engine (FP8 in + scalar descales)
    # O dtype domain, declared only by the quantized rows: elsewhere O must
    # equal Q, which facts.uniform_dtype already enforces.
    out_dtypes: frozenset = frozenset()
    # Block-scaled O domain (facts.o_block_scale): 0 = plain O; 16 = FP4_E2M1 O
    # + E4M3 scale per 16 d in ``sf_o``; 32 = FP8_E4M3 O + UE8M0 scale per 32.
    o_block_scales: frozenset = frozenset({0})

    # optional features a graph may request
    bias: bool = False
    dropout: bool = False
    score_mod: bool = False
    paged_kv: bool = False
    alibi: bool = False
    block_mask: bool = False
    rng_dump: bool = False
    score_max: bool = False  # per-row/tile score-max side output
    score_sum_exp: bool = False  # per-row/tile sum-of-exp side output
    dynamic_scale: bool = False
    unfuse_fma: bool = False
    # Stats written as (max + ln(sum_exp)) * log2(e) (sdpa(stats_use_log2=True)): the
    # kernel epilogue (or the split-KV combine) scales the LSE by log2(e).
    stats_log2: bool = False
    seq_q_trim: bool = False
    right_band_widening: bool = False

    causal: bool = False
    bottom_right: bool = False
    swa: bool = False
    padded: bool = False
    sink: bool = False
    # Optional dtype subset for sink support. None means every dtype served by
    # the engine; a subset lets one exact-shape flavor decline an unsupported
    # low-precision sink path without affecting its non-sink coverage.
    sink_dtypes: Optional[frozenset] = None
    stats: bool = False
    # The adapter accepts lse_tensor=None, so a stats-less graph needs no
    # dummy-LSE workspace chunk. Every row sets True. The SM100/SM120 kernels
    # None-specialize the LSE store (compiled out entirely); the SM80 kernels
    # still compute LSE into a kernel-internal buffer when unbound — a
    # pre-existing allocation on that path, tracked with its other
    # TemplateParams-conversion follow-ups.
    lse_optional: bool = False
    # Most THD forward lowerings derive addresses from compact length prefixes.
    # The SM100/SM103 ungated, unsplit CGA2 half D128/D256/D576 graph lowering
    # reads independent Q/O/Stats origins and nonpaged K/V origins. See
    # SUPPORT_MATRIX_TRACKER.md for this scope and the remaining sibling gaps;
    # thd=True alone does not promise support for padding between sequences.
    thd: bool = False
    # cu_seq_len_q / cu_seq_len_kv (B+1,) prefix sums (cuDNN 9.24+). Serving
    # rows consume the form on THD host-side (lens = adjacent differences of
    # the inherent tolist); dense cu graphs stay declined until the kernels
    # grow a CU read mode (len = cu[b+1] - cu[b]) — see mismatch().
    cu_seq_len: bool = False
    # Dense padded + stats: the per-batch seq_len_q LSE trim (padded q-rows
    # write LSE=-inf / O=0, cuDNN >= 9.14). Every kernel carries the dense
    # padded-Q trim itself (a graph with per-batch Q lengths compiles the
    # SEQ_Q_LENS_PRESENT specialization), so there is no separate capability.
    padded_stats: bool = False
    # s_q == 1 (decode-shaped) graphs; the SM80 prefill kernels are gated off.
    decode: bool = True

    # Escape hatch for kernels whose persistent multi-wave rescheduling is
    # numerically invalid: eligible only when the whole grid fits in one wave
    # (B*H_q*ceil(S_q/512) <= SM_count / CTA_MMA). NO CURRENT ROW SETS THIS.
    # Historical user: the fp8 d128 row, whose multi-wave wrong-O bug (14-19%
    # mismatched elements, some tiles left unwritten NaN) was root-caused to
    # the classic-pipeline TMEM stats race — the next tile's prologue BMM1
    # overwrote the S_acc-head (total_max, total_sum) before the correction
    # epilogue read them — and fixed by the mb_stats_read barrier (same fix as
    # the f16 kernel's; see sm100/prefill_d128_fp8.py / _common_blackwell.Bars).
    single_wave_only: bool = False
    # Serve ragged S_kv on unmasked graphs through the kernel's padded mask
    # compiled against the scalar S_kv (TemplateParams.kv_tail_mask, no
    # per-batch lengths buffer; #1425). Only rows that opt in use it; the
    # KV-tail rule is waived.
    skv_tail_via_padding: bool = False

    # Dense layout envelope this engine accepts:
    #   "bshd"       — Q/K/V/O must be BSHD-physical (stride order 3,1,2,0).
    #   "dense_flex" — any B/H/S stride permutation, padded (oversized)
    #                  strides included, as long as the head dim is
    #                  innermost-contiguous (stride 1) and the strides are
    #                  non-broadcast / non-overlapping (facts.dense_layout;
    #                  see graph_analyzer.dense_layout_ok). The DSL executor
    #                  normalizes such tensors to the kernel's canonical
    #                  BSHD-compact buffers (zero-copy when already BSHD).
    # THD (ragged) graphs always require the BSHD-order packing regardless —
    # the ragged lowering rebuilds packed [1,T,H,D] views and only that
    # packing is defined for it.
    layouts: frozenset[str] = frozenset({"bshd"})
    skv_tile: int = 128  # KV tail rule (waived when padded / causal covers the tail)

    # Tuning-knob domains this engine's lowering honors (see SdpaFwdKnobs).
    sched_policies: frozenset[int] = frozenset({SCHED_NATURAL})
    tile_ms: frozenset[int] = frozenset()
    tile_ns: frozenset[int] = frozenset()
    cgas: frozenset[int] = frozenset()
    # Shape-specific CGA domains for rows that lower several native flavors.
    # ``cgas`` remains the default. A split-specific entry further narrows the
    # domain for split-KV plans without changing the unsplit public domain.
    # Per-d-shape scheduler domain, mirroring cgas_by_d_shape. Needed because
    # sched_policies is row-wide while LPT support is per-KERNEL: the SM107 port
    # dropped `lpt_q_tiles_in_cga_units=True` on most flavors, so a row-wide
    # claim would advertise LPT for kernels that write nothing under it.
    # An entry here OVERRIDES sched_policies for that flavor.
    sched_policies_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    cgas_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    split_cgas_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    pack_gqas: frozenset[bool] = frozenset({False})
    # Does this row's lowering wire the KV-split path (kernel SplitHelpers +
    # adapter carving the partial slabs + launching the combine)? A GATE, not a
    # domain: WHICH splits are worth trying is a device-derived search space
    # (heuristics.split_kv_candidates), not a per-row constant. Fail-closed —
    # a row that accepts split_kv > 1 without the plumbing leaves untouched
    # partial slots at lse_partial = 0, which corrupt the combine's
    # log-sum-exp rather than raising.
    split_kv_supported: bool = False
    # Shapes whose kernel flavors wire SplitHelpers. None = every flavor in
    # d_shapes does (f16/SM120). A set = split_kv > 1 is honored only when
    # the graph's dims are covered by a member (the quantized families wire
    # the split path in the d128 flavor only).
    split_d_shapes: Optional[frozenset] = None
    # Softmax-precision domain (cudnn.data_type values). Empty = unserved.
    # Arch-dependent membership (the f16x2 exponent arm exists only in the
    # cc 10.7 quantized kernels) is expressed by SPLITTING the row per arch line —
    # each row declares exactly what its own lowering carries — not by a
    # knob x arch notch here.
    softmax_precisions: frozenset[int] = frozenset()
    # PackGQA, like split-KV, can be wired per FLAVOR rather than per row: the
    # SM107 line ships it only in the d128 per-tensor FP8 kernel, while the
    # ported d256/d512 siblings skip it.  None = every shape the row serves.
    # APPENDED at the end deliberately: Capabilities evolves append-only, so a
    # positional construction of an older field never silently rebinds.
    pack_gqa_d_shapes: Optional[frozenset] = None
    # THD graphs whose Stats has NO ragged offsets (per-batch padded (b, s_max, h)
    # rows, FlashInfer's form): the kernel stores per batch and the adapter
    # fills the tail rows with -inf. Rows whose kernels lack the per-batch THD
    # store keep False and decline the form (the backend serves it).
    thd_padded_stats: bool = False
    # Fused epilogue gate: the graph tail ``sdpa(virtual O_v) -> sigmoid(G) ->
    # mul(O_v, s)`` (facts.has_epilogue_gate, cudnn._sdpa_tail) lowered as the
    # kernel's ``O := O * sigmoid(G)`` epilogue (TemplateParams.epilogue_gate;
    # the Rubin d256 f16/bf16 and per-tensor FP8 kernels).  A FEATURE, not a
    # knob -- the graph asks for it, so it follows the thd_d_shapes /
    # split_d_shapes / pack_gqa_d_shapes idiom: ``epilogue_gate_d_shapes`` is
    # the set of NATIVE shapes whose kernel carries the gate, matched on the
    # graph's EXACT (d_qk, d_v) -- never through the zero-padding envelope
    # (the gate tile is TILE_O wide and the padded columns would gate garbage).
    # None = every shape in d_shapes (no row says that today).  The standalone
    # adapter's check_support carries the twin of this claim (rule 8b) through
    # the same constant (config_sm107.SM107_EPILOGUE_GATE_SHAPES, rule 8b').
    epilogue_gate: bool = False
    epilogue_gate_d_shapes: Optional[frozenset] = None
    # dtype domain of G.  None = G must equal Q's dtype (the half kernels stage
    # the gate in Q's storage dtype); the quantized rows name the half dtype
    # their kernel stages it in (bf16 on the Rubin FP8 d256 kernel).
    epilogue_gate_dtypes: Optional[frozenset] = None
    # Flavors whose kernel packs a PROPER DIVISOR of the GQA group when the
    # group does not divide tile_m (partial PackGQA: 96/8 packs 4 of its 12
    # heads per token row-group, 48/8 packs 2; config_sm100.pack_gqa_group_size).
    # None = every flavor keeps the full-ratio contract (the group must divide
    # tile_m, or PACK_GQA=1 is declined; SM120 packs any group up to tile_m:
    # pack_gqa_requires_divisible).  Native on the SM100 d128 / d256 f16
    # kernels and the native SM107 dense d512 body; the other d192x128 /
    # d512 kernels pack HEADS_PER_TILE = G.
    # APPENDED after the epilogue-gate fields (append-only contract above):
    # test_capabilities_positional_prefix_is_append_only pins the tail.
    pack_gqa_partial_d_shapes: Optional[frozenset] = None
    # Shapes whose kernel flavors wire the PAGED_KV specialization (block-table
    # indirection on the K/V TMA loads). None = every flavor in d_shapes. A set
    # = a paged graph is served only when the flavor the lowering SELECTS
    # (_selected_d_shape, the smallest covering envelope) is a member — the
    # test is on the selection, not the raw dims, so mixed head dims that ride a
    # wired envelope ((256, 128) -> d256) are served and an unwired selection
    # ((512, 128) -> d512) is declined. Kernel-body gates mirror it
    # (config_sm100._PAGED_KV_FLAVORS is the backstop).  APPENDED after
    # pack_gqa_partial_d_shapes (append-only contract above; the same test
    # pins it).
    paged_d_shapes: Optional[frozenset] = None
    # Native THD prefill flavors whose worklist and Stats index packed query
    # heads. Empty is fail-closed; the separate ragged-Q decode leg is unchanged.
    # Appended to preserve positional construction of existing capabilities.
    thd_pack_gqa_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # attn_scale = 0. The SM100/SM107/SM120 kernels fold the scale into exp2 after an unscaled, -inf-masked
    # running max, which a zero scale turns into NaN (#1435); SM80 and SM90 specialize on the scale's sign. Appended last.
    zero_scale: bool = False
    # Shapes whose kernel flavors carry the pre-folded-scale arm (the op attribute
    # sdpa(attn_scale_prefolded=True): Q carries attn_scale * log2(e), the kernel
    # traces no per-score scale).  None = unserved (the default: a row opts in).
    # Matched on the flavor the lowering SELECTS (_selected_d_shape), like
    # paged_d_shapes.  The paged-KV bodies and the single-CTA half THD legs
    # (packed split / D192 single-Q) run bodies without the arm and are declined
    # by rule in mismatch().  APPENDED after zero_scale (append-only
    # contract above; the same test pins it).
    attn_scale_prefolded_d_shapes: Optional[frozenset] = None
    # Score side outputs -- the graph's optional ``score_max`` / ``score_sum_exp``
    # fp32 (B, H_q, S_q, 1) ports (``score_max`` / ``score_sum_exp`` above are the
    # row-level flags): per Q row the softmax max in NATURAL units (scale * max_j
    # s_j, the sink folded in as one more column) and the fp32 row-sum of
    # exp(s - max), the two operands the LSE is built from, so Stats ==
    # score_max + ln(score_sum_exp) to fp32 rounding; a keyless row reads -inf / 0
    # (with a sink sink_logit / 1), the same SELECT and store guard as the LSE;
    # ``stats_use_log2`` leaves them untouched (natural units always).  This set
    # follows the paged_d_shapes idiom: None = every shape in d_shapes when the
    # flags are set, a set = served only when the flavor the lowering SELECTS
    # (_selected_d_shape) is a member -- the kernel FILE owns the store.  Dense,
    # unsplit, unpaged plans on the 2-CTA prefill tile only (mismatch holds the
    # conditions); the standalone adapter's check_support reads the same constant
    # (config_sm100 / config_sm107.SM10x_SCORE_STATS_SHAPES, rule 8b').  APPENDED
    # after attn_scale_prefolded_d_shapes (append-only contract above).
    score_stats_d_shapes: Optional[frozenset] = None
    # Shapes that carry the row's DENSE leg only -- the Rubin 32- / 64-wide
    # tiles of the f16/bf16 d128 file (config_sm107.SM107_F16_SMALL_D_SHAPES).
    # ``_selected_d_shape`` takes such a shape only for a graph that asks for
    # nothing beyond that leg: no THD, no paged KV, no split_kv > 1, no
    # PackGQA, no score side outputs, and a cga (when requested) inside the
    # shape's cgas_by_d_shape domain.  Any other graph walks PAST it to the
    # next covering shape -- the d128 flavor it rode before the tile existed,
    # where those legs are wired and validated -- instead of a decline, so a
    # new native tile never withdraws a graph the row served through the
    # envelope, and a persisted autotune record keeps working (a cga2 request
    # at D <= 32 lands on the 64-wide tile, the smallest covering shape
    # validated at that width).  Empty on every other row: their native shapes
    # either carry a leg or decline it (the SM100 native d64's split / score
    # declines describe THAT kernel; the row never served d64 through an
    # envelope, so there is no route to fall back to).  The standalone adapter
    # walks the same predicate (config_sm107.sm107_small_tile_leg /
    # sm107_f16_tile_width(leg=, cga=), rule 8b').  APPENDED after
    # score_stats_d_shapes (append-only contract above).
    dense_only_d_shapes: frozenset[tuple[int, int]] = frozenset()
    # Additive bias on the RESIDENT-row-block arm (the Rubin f16/bf16 d128-file
    # tiles, config_sm107.SM107_BIAS_D_SHAPES) -- beside the row-level ``bias``
    # flag above: the native shapes whose kernel carries the block (None = every
    # shape of a ``bias`` row serves it, the SM80 row's semantics), and whether
    # the bias must be the Q/K/V dtype (the SM80 row also takes fp32; the block
    # arm loads the io dtype's 16-bit pairs).  The resident block holds the work item's
    # rows x its whole KV range (``config_sm107.sm107_bias_resident_s_kv_max``); above
    # that budget the kernel streams the bias as a 2- / 3-stage per-kv-tile ring
    # (``BIAS_MODE = 2``), so ``mismatch`` asks ``config_sm107.sm107_bias_mode_pick``
    # (one arithmetic with the adapter and the config factory, rule 8b') and declines
    # typed only where NEITHER form fits at any width of the flavor's domain.
    # Dense, unsplit, unpaged, non-PackGQA, ungated, without the score side
    # outputs; the bias is ``[1|B, 1|H_q, S_q, S_kv]`` with a contiguous key dim
    # and 16-byte strides (S_kv % 8 == 0 for a compact bias: the TMA rule).
    bias_d_shapes: Optional[frozenset] = None
    bias_io_dtype_only: bool = False
    # The per-key mask ``kv_mask`` (python-only input: ``[1|B, 1, 1, S_kv]``), with the
    # dtype set it is served in: BOOLEAN = a key with a False entry is masked for every
    # query (a query whose every key is masked reads O = 0 / Stats = -inf); the Q/K/V
    # dtype or FLOAT = an ADDITIVE per-key vector folded into the scores pre-scale like
    # the bias (the models' finite -1e9 fold, preserved as a value), S_kv % 8 == 0 with
    # 16-byte strides for its granule loads.  Served by the same arm as the bias (the
    # same shapes and S_kv-free legs); a 16-bit vector must be the Q/K/V dtype.
    kv_mask: bool = False
    kv_mask_dtypes: frozenset = frozenset()
    # Per-d-shape KV tile-width domain, mirroring sched_policies_by_d_shape: ``tile_ns``
    # stays the row-wide floor (and the heuristics' default), an entry here REPLACES it
    # for the flavor the lowering selects -- the Rubin f16/bf16 32- / 64-wide tiles
    # render TILE_N 64 as well as 128 (config_sm107.SM107_TILE_NS_BY_D_SHAPE, the
    # adapter's twin), the d128 rendering of the same file keeps 128.  APPENDED after
    # kv_mask_dtypes (append-only contract above).
    tile_ns_by_d_shape: tuple[tuple[tuple[int, int], frozenset[int]], ...] = ()
    # Whether PackGQA requires tile_m to be divisible by the GQA ratio G (or, on
    # a partial flavor, to share a factor with it).  False on the SM120 rows:
    # any G up to tile_m packs, tile_m % G != 0 included
    # (config_sm100.pack_gqa_group_size).  APPENDED after tile_ns_by_d_shape
    # (append-only contract above).
    pack_gqa_requires_divisible: bool = True
    # Native flavors with dense Q only. Unlike thd_d_shapes, this preserves
    # the THD envelope of every other flavor on the row. Append-only.
    thd_excluded_d_shapes: frozenset[tuple[int, int]] = frozenset()


def _band_covers_kv_tail(facts: "ga.SdpaGraphFacts") -> bool:
    """True when the causal band (plain or right-widened) provably masks every
    KV column >= S_kv, so a ragged KV tail cannot leak into the softmax.

    The last unmasked column is (S_q - 1) + R top-left or (S_kv - 1) + R
    bottom-right (R = the right-band bound, 0 for plain causal)."""
    if not (facts.causal or facts.right_band_widening):
        return False
    r = facts.right_bound or 0
    if facts.bottom_right:
        return r == 0
    return facts.s_q + r <= facts.s_kv


def _synth_kv_padding(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """True when this graph's ragged S_kv is served through the kernel's padded
    mask compiled against the scalar S_kv (the adapter's kv_tail_mask, #1425)
    — the one path split-KV cannot ride.

    Only a DENSE, mask-free graph whose S_kv is not a multiple of the KV tile
    takes it. A padded graph already carries real per-batch lengths; a paged
    graph is padded by construction (per-batch ``seq_len_kv`` is mandatory and
    bounds the walk on device), so a declared ``paged_attention_max_seq_len_kv``
    that is not a tile multiple — FlashInfer passes its true max verbatim,
    e.g. 4000 — never selects this path and must not cost the launch its
    split; THD carries its lengths via cu_seqlens. The split gates in
    :func:`mismatch` and ``heuristics._split_points`` share this predicate with
    the lowering so the three cannot drift apart again (they did: the gates
    lacked the padded/paged terms and declined every paged decode graph whose
    declared max was not a 128-multiple).
    """
    return (
        capabilities.skv_tail_via_padding
        and not facts.padded
        and not facts.has_paged_kv
        and not facts.thd
        and facts.s_kv % (capabilities.skv_tile or 128) != 0
        and not _band_covers_kv_tail(facts)
    )


_RAGGED_OFFSET_DTYPES = (cudnn.data_type.INT32, cudnn.data_type.INT64)


def _ragged_row_divisor(t) -> Optional[int]:
    """Offset-units-per-token divisor of a ragged tensor: its declared TOKEN
    stride (the ``S`` stride of the (B, H, S, D) declaration -- ``H * D`` for a
    packed buffer, larger with a token-stride gap) over the offset multiplier,
    when the tensor carries int32 / int64 ragged offsets and the multiplier
    divides that stride.  cuDNN's contract: ``offset[b] * multiplier`` is the
    ELEMENT offset of sequence ``b``'s first token, so the kernel reads
    ``offset[b] // divisor`` as its token row.  None when the tensor cannot
    be addressed that way."""
    off = getattr(t, "ragged_offset", None)
    if off is None or off.get_data_type() not in _RAGGED_OFFSET_DTYPES:
        return None
    stride = tuple(int(s) for s in t.get_stride())
    if len(stride) != 4:
        return None
    token_stride = stride[2]
    mult = int(t.get_ragged_offset_multiplier() or 1)
    if mult <= 0 or token_stride <= 0 or token_stride % mult != 0:
        return None
    return token_stride // mult


def _thd_decode_leg_int64(facts: "ga.SdpaGraphFacts") -> bool:
    """Whether a ``_thd_decode_leg`` graph's ragged offsets are int64 (one width
    for Q, O and Stats -- a mixed graph is declined by ``_thd_decode_leg``)."""
    return facts.q_t.ragged_offset.get_data_type() == cudnn.data_type.INT64


def _thd_decode_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """True when a THD (ragged Q/O/Stats) graph rides the SM100 d128 DECODE tile's
    ragged-Q leg (``sm100/decode_d128_f16.py``, ``TemplateParams.ragged_q``)
    instead of the prefill tile's THD leg: the FlashInfer prefill-style paged
    graph at one token per sequence.

    The leg keeps the dense grid over the declared batch, reads each batch's Q
    ragged offset on device as its row coordinate over the packed Q view, and
    always splits the KV walk so the combine pass -- not the kernel -- places
    the final O / Stats rows at their ragged offsets. Hence the shape of the
    predicate: the (128, 128) half flavor on the Blackwell line (the ragged-Q
    leg is not wired on cc 10.7, whose dense d128 graphs ride the same tile
    through TILE_CGA_M=1 since issue #1472), S_q(max) == 1, PAGED K/V (a ragged K/V needs the THD leg's
    clamped descriptors), ragged Stats when Stats are requested (a per-batch
    padded Stats has no ragged base to place rows at), int32 offsets whose
    multiplier divides the row, per-batch ``seq_len_kv`` (the dense kernel's
    SEQ_KV read; the cu form is not plumbed) and no fused epilogue gate
    (unsplittable); a sink rides it (the combine folds it once). Twin of
    ``SdpaFwdDslSm100.thd_decode_leg``; keep in lockstep.
    """
    if not (facts.thd and facts.has_paged_kv and facts.s_q == 1):
        return False
    if facts.is_fp8 or facts.is_mxfp8 or getattr(facts, "has_epilogue_gate", False):
        return False
    if capabilities.sm_lo != 100 or capabilities.sm_hi >= 107:
        return False
    if (facts.d_qk, facts.d_v) != (128, 128) or _selected_d_shape(capabilities, facts) != (128, 128):
        return False
    if facts.cu_seq_kv_t is not None:
        return False
    if _ragged_row_divisor(facts.q_t) is None or _ragged_row_divisor(facts.o_t) is None:
        return False
    if facts.stats_t is not None and _ragged_row_divisor(facts.stats_t) is None:
        return False
    # One offset width for every ragged operand (the kernels compile one read width).
    widths = {t.ragged_offset.get_data_type() for t in (facts.q_t, facts.o_t) + ((facts.stats_t,) if facts.stats_t is not None else ())}
    return len(widths) == 1


def decode_d256_q_tile_for_row(capabilities: Capabilities, s_q: int, pack_g: int, wide_launch: bool = False, paged: bool = False, masked: bool = True) -> int:
    """The d256 decode tile's N extent (the Q rows of one token unit) a ROW routes ``s_q``
    tokens packed ``pack_g`` heads per token onto, or 0 for the prefill tile: the Rubin
    row's rule (``config_sm107.decode_d256_q_tile``, table-driven: the 32- and 64-column
    tiles routed in one unit, then two token units of the widest admitted width for a packed
    MTP step) on cc 10.7, the Blackwell line's (``config_sm100.decode_d256_q_tile``: the
    16-column tile in one unit, and the 32-column one too for a ``wide_launch`` --
    :func:`config_sm100.d256_wide_decode_launch`) otherwise.  ``wide_launch`` (appended) is
    the Blackwell line's launch-size reading (:func:`d256_wide_decode_launch_for_facts`);
    ``paged`` / ``masked`` (appended) are the graph's cache kind and causal / window mask --
    the Rubin 64-column width's form predicate (``config_sm107.decode_d256_wide_tile_admits``)
    declines a mask-free PAGED step, which keeps the (16, 32) table.  Each line reads its own
    arguments only.  The facts-level twin of ``SdpaFwdDslSm100._decode_q_tile_for``'s arch
    dispatch; the heuristics' decode geometry reads it too -- keep the three in lockstep
    (:func:`_facts_decode_form` reads the two Rubin facts off ``SdpaGraphFacts``)."""
    if capabilities.sm_lo == 107:
        return _config_sm107.decode_d256_q_tile(s_q, pack_g, paged=bool(paged), masked=bool(masked))
    return decode_d256_q_tile(s_q, pack_g, wide_launch=wide_launch)


def d256_wide_decode_launch_for_facts(facts: "ga.SdpaGraphFacts", pack_g: int) -> bool:
    """:func:`config_sm100.d256_wide_decode_launch` on a graph's declared shape: batch x
    (heads / ``pack_g``) units over ``s_kv`` keys, never for a windowed graph.  The
    facts-level twin of ``SdpaFwdDslSm100._decode_q_tile_for``'s launch argument."""
    return d256_wide_decode_launch(facts.b, facts.h_q // max(1, pack_g), facts.s_kv, facts.window_left is not None)


def _facts_decode_form(facts: "ga.SdpaGraphFacts") -> Tuple[bool, bool]:
    """``(paged, masked)`` of a graph for :func:`decode_d256_q_tile_for_row`: the paged KV cache, and the causal /
    window mask exactly as the kernels' MASK_CAUSAL / MASK_SWA bits derive (``config_sm100._mask_flags_from``: a right
    band resolved -- ``right_bound`` 0 = causal, ``causal`` its effective reading -- or a left window)."""
    return bool(facts.has_paged_kv), bool(facts.causal) or facts.right_bound is not None or facts.window_left is not None


def _facts_decode_form_kw(facts: "ga.SdpaGraphFacts") -> dict:
    """:func:`_facts_decode_form` as the ``paged`` / ``masked`` keywords of :func:`decode_d256_q_tile_for_row`."""
    paged, masked = _facts_decode_form(facts)
    return dict(paged=paged, masked=masked)


def d256_decode_tile_selected(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", pack_g: int = 1, split_kv: Optional[int] = None) -> bool:
    """Whether the f16/bf16 row lowers this graph onto the d256 DECODE tile -- the
    swap-AB tile, ``sm100/decode_d256_f16.py`` on the Blackwell row and
    ``sm107/decode_d256_f16.py`` on the Rubin row -- the facts-level twin of
    ``SdpaFwdDslSm100._decode_q_tile`` (and of the heuristics' cost-model gate):
    half inputs, dense (not THD) queries, the (256, 256) flavor, no pre-folded scale
    (not wired on the decode tile), and ``S_q x pack_g`` packed Q rows within the
    row's routed envelope (``decode_d256_q_tile_for_row``: 16 rows in one unit on the
    Blackwell line, which also routes the 32-column tile for a launch big enough to be
    HBM-bound; on cc 10.7 the 32-column tile for every launch, in up to two token units
    for a packed MTP step; ``pack_g`` is the decode tile's WHOLE-group packing when the plan
    packs, else 1).

    The fused epilogue gate: the tile has no gate seams, so a GATED graph rides it only
    when the plan SPLITS (``split_kv`` >= 2) -- the gate then moves into the split
    combine (``sm100/split_combine`` ``gate=True``: ``O *= sigmoid(G)`` on the fp32
    merged value, the fused kernels' own arithmetic; ``SdpaFwdDslSm100._gate_in_combine``)
    on the Rubin row, which claims the gate; unsplit, the d256 prefill kernel's fused
    epilogue serves it.  ``split_kv=None`` asks whether SOME plan rides the tile (the
    facts-only question), so a gated graph answers True there.  A gated graph whose
    caller enabled execute-time shape overrides never rides it: the split combine
    binds G to the plan's declared (B, H_q, S_q, D_v) (``prepared.CombineGate``), so
    such a graph keeps the unsplit fused-gate kernel (``mismatch`` names the
    overrides).  Keep the three in lockstep."""
    gate_ok = not facts.has_epilogue_gate or (
        capabilities.sm_lo == 107 and capabilities.epilogue_gate and (split_kv is None or split_kv > 1) and not facts.shape_overrides
    )
    return (
        capabilities.sm_lo in (100, 107)
        and capabilities.sm_hi <= _BLACKWELL[1]
        and not (facts.is_fp8 or facts.is_mxfp8)
        and not facts.thd
        and gate_ok
        and not facts.attn_scale_prefolded
        and _selected_d_shape(capabilities, facts) == (256, 256)
        and decode_d256_q_tile_for_row(
            capabilities, facts.s_q, pack_g, wide_launch=d256_wide_decode_launch_for_facts(facts, pack_g), **_facts_decode_form_kw(facts)
        )
        > 0
    )


def _decode_tile_pack_g(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> int:
    """The decode tile's packing for a knob set: the whole GQA ratio when it packs, else 1."""
    return (facts.h_q // facts.h_kv) if (knobs is not None and knobs.pack_gqa and facts.h_kv) else 1


def _decode_tile_pack_g_choices(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> Tuple[int, ...]:
    """The packings a decode-tile question may assume: a knob set's own (:func:`_decode_tile_pack_g`); with NO knob set -- the
    facts-only question the heuristics' ``_eligible`` asks before any set is built -- every packing SOME plan may carry: unpacked,
    and the whole group where there is one.  A graph only the packed form serves (4 / 2 at ``S_q`` 33 over pages under the causal
    mask: 66 packed rows = two 64-column units, while 33 unpacked rows are past the 32-column tile) therefore keeps its row on the
    ranked list, where the heuristics propose the packed set (``_pack_gqa_eligible`` reads the same group) and ``mismatch`` declines
    the unpacked one.  Until 2026-10-15 the facts-only reading asked the unpacked question alone and dropped the row, so the ranked
    list offered nothing where the adapter and the facts twins routed the packed form -- the gate behind the graph-level pin of the
    Rubin decode suite (``test_decode_routing_boundary``), which the host twin test there now pins over a grid."""
    if knobs is not None:
        return (_decode_tile_pack_g(facts, knobs),)
    group = (facts.h_q // facts.h_kv) if facts.h_kv else 0
    return (1, group) if (group > 1 and facts.h_q % facts.h_kv == 0) else (1,)


def d256_decode_tile_served(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs], split_kv: Optional[int] = None) -> bool:
    """:func:`d256_decode_tile_selected` for a knob set's own packing, or -- with no knob set -- for SOME plan: any packing of
    :func:`_decode_tile_pack_g_choices` the row routes onto the tile."""
    return any(d256_decode_tile_selected(capabilities, facts, pack_g, split_kv) for pack_g in _decode_tile_pack_g_choices(facts, knobs))


def _decode_tile_split_kv(knobs: Optional[SdpaFwdKnobs]) -> Optional[int]:
    """The split a knob set lowers with, for the decode tile's gate question: an
    unrequested split (None) lowers unsplit, so it reads 1; no knob set at all
    (``knobs is None``, the facts-only question) reads None (= some plan)."""
    return None if knobs is None else (knobs.split_kv or 1)


def _thd_decode_leg_divisors(facts: "ga.SdpaGraphFacts") -> tuple:
    """The (Q, O, Stats) elements-per-token divisors of a ``_thd_decode_leg`` graph
    (Stats: 1 when no Stats output is declared)."""
    return (
        _ragged_row_divisor(facts.q_t),
        _ragged_row_divisor(facts.o_t),
        _ragged_row_divisor(facts.stats_t) if facts.stats_t is not None else 1,
    )


def rubin_dense_d128_shared_leg(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> bool:
    """Historical cc 10.7 dense half D128 decode/packing admission predicate (issue #1472).

    This includes the d64 envelope and excludes the pre-folded scale. The loader uses the shared decode tile
    at CGA1 and native SM107 prefill for packed CGA2. The historical predicate name and admission limits
    are retained for policy consumers. Twin of SdpaFwdDslSm100._rubin_shared_dense_leg; keep the gates in lockstep.

    ``knobs``: the request being validated, when there is one -- on the d64 envelope the flavor depends on it (the row's
    native 64-wide tile carries the dense leg only, so a PackGQA or split request walks past it onto (128, 128) and the
    shared bodies; a knob-less d64 graph selects the native tile and is not a shared leg)."""
    return (
        capabilities.sm_lo == 107
        and not (facts.is_fp8 or facts.is_mxfp8)
        and not facts.thd
        and not facts.has_paged_kv
        and _selected_d_shape(capabilities, facts, knobs) == (128, 128)
        and not facts.attn_scale_prefolded
    )


def paged_thd_split_domain(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """The paged subset of the bounded THD split contract."""
    return facts.has_paged_kv and thd_split_domain(capabilities, facts)


def thd_split_domain(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """Fixed packed-Q bounds whose partial workspace is caller-owned."""
    return (
        capabilities.sm_lo in (100, 107)
        and (
            not facts.shape_overrides
            or (facts.max_total_seq_len_q is not None and 0 < facts.max_total_seq_len_q <= facts.b * facts.s_q)
            # D576 partials retain the native binder's B*S_q capacity clamp
            # when the graph omits the optional tighter total-Q bound.
            or ((facts.d_qk, facts.d_v) == SM100_NATIVE_D576_SHAPE and facts.max_total_seq_len_q is None and facts.b * facts.s_q > 0)
        )
        and (
            not facts.has_sink
            or supports_thd_split_sink(
                (facts.d_qk, facts.d_v),
                device_cc=facts.device_cc,
                fp8=facts.is_fp8 or facts.is_mxfp8,
                thd=facts.thd,
                paged=facts.has_paged_kv,
                max_q=facts.s_q,
            )
        )
        and not facts.has_epilogue_gate
        and supports_thd_split(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            max_q=facts.s_q,
            padded_stats=facts.stats_t is not None and getattr(facts.stats_t, "ragged_offset", None) is None,
        )
    )


def _prepared_decline_reason(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", split_kv: Optional[int]) -> Optional[str]:
    """Pure admission for the graph's normalized VariantPack executor.

    Shared by planning and lowering; no adapter, tensor framework or compilation.
    ``split_kv=None`` asks whether the provider has a possible prepared plan.
    A complete assignment additionally checks the final O store's layout.
    Runtime geometry still has to fit the compiled binder's per-call contract.
    """
    if capabilities.sm_lo == 90 and facts.shape_overrides:
        return "SM90 prepared launch retains fixed graph geometry; shape/stride overrides are unsupported"
    if capabilities.sm_lo not in (90, 100, 107, 120):
        return "this engine has no prepared shape/stride override executor"
    if (facts.is_fp8 or facts.is_mxfp8) and capabilities.sm_lo == 107 and facts.device_cc != (10, 7):
        return "prepared SM107 FP8/MXFP8 requires device cc 10.7"
    if facts.o_block_scale and (facts.sf_o_t is None or facts.dtype_o != {16: cudnn.data_type.FP4_E2M1, 32: cudnn.data_type.FP8_E4M3}.get(facts.o_block_scale)):
        return "prepared block-scaled output requires the matching O and SF_O declarations"
    if facts.o_block_scale and (facts.thd or facts.has_paged_kv or (split_kv or 1) > 1 or facts.shape_overrides or facts.has_epilogue_gate):
        return "prepared block-scaled outputs require fixed dense unsplit plans"
    if facts.is_mxfp8 and (
        capabilities.sm_lo not in (100, 107, 120)
        or (capabilities.sm_lo == 107 and (split_kv or 1) > 1)
        or not capabilities.is_mxfp8
        or facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1)
        or (facts.has_epilogue_gate and capabilities.sm_lo != 107)
        or (facts.shape_overrides and not facts.thd)
    ):
        return "prepared MXFP8 serves SM100 fixed dense or bounded THD, SM107 fixed dense or bounded THD scalar outputs, and SM120 fixed dense or THD plans"
    if facts.is_fp8 and (
        capabilities.sm_lo not in (100, 107, 120)
        or facts.dtype_o not in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1)
        or (facts.has_epilogue_gate and capabilities.sm_lo != 107)
    ):
        return "prepared FP8 serves SM100, SM107 or SM120 scalar-scaled outputs"
    if capabilities.sm_lo == 120 and facts.has_paged_kv:
        return "prepared SM120 does not serve paged KV"
    if facts.has_bias and capabilities.bias_d_shapes is None:
        # The resident-block rows bind the bias through the prepared binder (bind_dense validates its runtime geometry);
        # the SM80 row's bias travels through its own adapter and stays off the prepared path.
        return "prepared overrides cannot use bias"
    if facts.thd:
        if facts.has_epilogue_gate:
            return "prepared THD overrides cannot use an epilogue gate"
        if (split_kv or 1) > 1:
            if facts.has_sink and not getattr(cudnn._pybind_module._SdpaThdBinder, "supports_paged_split_sink", False):
                return "packed split sinks require the matching native cuDNN Frontend extension"
            # Ragged-Q decode binds offsets through its dense launch. The
            # paged D128 THD leg instead owns bounded packed partial regions.
            return None if _thd_decode_leg(capabilities, facts) or thd_split_domain(capabilities, facts) else "prepared THD overrides cannot use split-KV"
        return None
    if facts.has_epilogue_gate and (split_kv or 1) > 1 and facts.shape_overrides:
        # The d256 decode tile's split applies the gate in its combine, whose gate binding is fixed to the declared
        # (B, H_q, S_q, D_v) (prepared.CombineGate); an override-enabled graph may change either at execute.
        return (
            "prepared dense overrides cannot ride the gate-in-combine split (its gate binding is fixed to the declared (B, H_q, S_q, D_v)); "
            "a graph with execute-time shape overrides keeps the unsplit fused-gate kernel"
        )
    if facts.cu_seq_q_t is not None or facts.cu_seq_kv_t is not None:
        return "prepared dense overrides require per-batch lengths, not prefix sums"
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import dense_bind_strides

    if capabilities.sm_lo == 90:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm90 import dense_bind_strides

    tensors = [facts.q_t] + ([] if facts.has_paged_kv else [facts.k_t, facts.v_t])
    if facts.has_epilogue_gate:
        tensors.append(facts.epilogue_gate_t)
    if split_kv is not None and split_kv <= 1:
        tensors.append(facts.o_t)
    for tensor in tensors:
        if tensor is None:
            return "prepared overrides require declared operands"
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry

        geometry = storage_geometry(tensor.get_dim(), tensor.get_stride(), tensor.get_data_type())
        width = 1 if tensor.get_data_type() in (cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1) else 2
        if geometry is None or dense_bind_strides(*geometry, width) is None:
            return "prepared overrides require input and unsplit-output layouts that bind without a copy"
    if facts.o_t is None or not ga.dense_layout_ok(tuple(facts.o_t.get_dim()), tuple(facts.o_t.get_stride())):
        return "prepared overrides require a non-overlapping dense output layout"
    return None


def _beyond_dense_leg(facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> bool:
    """Whether the graph asks for a leg beyond the dense one a ``dense_only_d_shapes`` tile carries: THD, paged KV, the
    score side outputs, the pre-folded attention scale (graph facts), split_kv > 1 or PackGQA (knobs).  The twin of
    ``config_sm107.sm107_small_tile_leg``."""
    if facts.thd or facts.has_paged_kv or facts.has_score_max or facts.has_score_sum_exp or facts.attn_scale_prefolded:
        return True
    return knobs is not None and ((knobs.split_kv or 1) > 1 or bool(knobs.pack_gqa))


def _selected_d_shape(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> Optional[tuple[int, int]]:
    """Smallest native flavor whose envelope covers this graph -- honouring the
    per-shape envelope floors, so the knob domains (cga, split) describe the
    flavor the lowering will actually pick (api_dsl._pick_flavor walks the same
    floors). An exact native shape is always its own selection -- except a
    ``dense_only_d_shapes`` tile, which is taken only for a graph on the dense
    leg (``_beyond_dense_leg`` False) at a cga inside its domain (``knobs.cga``
    unset or in ``cgas_by_d_shape``); otherwise the walk continues to the next
    covering shape, the route the graph took before the tile existed.  Callers
    that know the knobs pass them (``mismatch``, ``effective_cgas``); a
    knob-less call sees the graph-fact legs only."""

    floors = dict(capabilities.d_envelope_floors)
    covering = [
        shape
        for shape in capabilities.d_shapes
        if facts.d_qk <= shape[0] and facts.d_v <= shape[1] and ((facts.d_qk, facts.d_v) == shape or min(facts.d_qk, facts.d_v) > floors.get(shape, 0))
    ]
    if capabilities.dense_only_d_shapes:
        beyond = _beyond_dense_leg(facts, knobs)
        cga = knobs.cga if knobs is not None else None
        domains = dict(capabilities.cgas_by_d_shape)
        covering = [
            shape
            for shape in covering
            if shape not in capabilities.dense_only_d_shapes or (not beyond and (cga is None or cga in domains.get(shape, capabilities.cgas)))
        ]
    return min(covering, key=lambda shape: (shape[0], shape[1])) if covering else None


def sm120_decode_tile(capabilities: Capabilities, tile_m: Optional[int]) -> bool:
    """Whether ``tile_m`` runs the SM120 half DECODE tile (sm120/decode_f16.py):
    the f16/bf16 SM120 row with ``tile_m`` 16 or 32."""
    return capabilities.sm_lo == 120 and not (capabilities.is_fp8 or capabilities.is_mxfp8) and tile_m is not None and _sm120_is_decode_tile(tile_m)


def _sm120_tile_mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> Optional[str]:
    """Why the SM120 kernel table has no CTA tile for ``knobs`` at this graph's
    head dims (None when it has one, or on any other row). ``tile_m`` and
    ``tile_n`` each pass their own domain in ``mismatch``; the template serving
    the head dims takes only the pairs ``config_sm120.tile_domain`` lists, and
    the decode tile (tile_m 16 / 32) adds its own graph conditions. Twin of the
    tile lines of ``api_dsl.SdpaFwdDslSm120.check_support``."""
    if capabilities.sm_lo != 120 or (knobs.tile_m is None and knobs.tile_n is None):
        return None
    if sm120_decode_tile(capabilities, knobs.tile_m):
        if facts.thd:
            return "the SM120 decode tile (tile_m 16 / 32) is dense-only; THD graphs run the prefill templates"
        if max(facts.d_qk, facts.d_v) > _SM120_GENERAL_HEAD_TILE_MAX:
            return f"the SM120 decode tile serves head dims up to {_SM120_GENERAL_HEAD_TILE_MAX}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
        stage = _sm120_decode_kv_stage(*_sm120_head_tiles(facts.d_qk, facts.d_v, _SM120_HEAD_TILE_GRANULE))
        if knobs.tile_n is not None and knobs.tile_n != stage:
            return f"the SM120 decode tile streams {stage}-key K/V stages at D=({facts.d_qk}, {facts.d_v}); requested tile_n={knobs.tile_n}"
        return None
    domain = _sm120_tile_domain(facts.d_qk, facts.d_v, capabilities.is_fp8 or capabilities.is_mxfp8)
    # An unset tile_m runs the first sequence Q tile the table lists, else the
    # table's largest; an unset tile_n is picked from the table by SMEM fit.
    tile_m = knobs.tile_m
    if tile_m is None:
        tile_m = _SM120_SEQ_Q_TILES[0] if any(m == _SM120_SEQ_Q_TILES[0] for m, _ in domain) else max(m for m, _ in domain)
    served = (tile_m, knobs.tile_n) in domain if knobs.tile_n is not None else any(m == tile_m for m, _ in domain)
    if not served:
        return f"the SM120 kernel for D=({facts.d_qk}, {facts.d_v}) has no tile_m={knobs.tile_m}, tile_n={knobs.tile_n} tile; it takes {sorted(domain)}"
    return None


def pack_gqa_partial(capabilities: Capabilities, facts: "ga.SdpaGraphFacts") -> bool:
    """Whether the flavor this graph lowers onto packs a proper divisor of a
    GQA group that does not divide tile_m (``pack_gqa_partial_d_shapes``)."""
    return capabilities.pack_gqa_partial_d_shapes is not None and _selected_d_shape(capabilities, facts) in capabilities.pack_gqa_partial_d_shapes


def effective_sched_policies(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> frozenset[int]:
    """Scheduler domain of the native flavor the lowering will pick.

    Row-wide `sched_policies` is the floor; a `sched_policies_by_d_shape` entry
    for the selected flavor replaces it. That is what lets one Rubin row serve a
    d256 kernel that honours LPT next to flavors that do not.  ``knobs`` (when
    the caller has them) route a dense-only tile's non-dense legs to the flavor
    that carries them (``_selected_d_shape``), and name the SM120 decode tile,
    which walks its units in launch order (SCHED_NATURAL only).
    """
    if knobs is not None and sm120_decode_tile(capabilities, knobs.tile_m):
        return frozenset({SCHED_NATURAL})
    selected = _selected_d_shape(capabilities, facts, knobs)
    if selected is not None:
        for shape, shape_domain in capabilities.sched_policies_by_d_shape:
            if shape == selected:
                return shape_domain
    return capabilities.sched_policies


def effective_tile_ns(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> frozenset[int]:
    """KV tile-width domain of the native flavor the lowering will pick: row-wide ``tile_ns`` is the floor, a
    ``tile_ns_by_d_shape`` entry for the selected flavor replaces it (the Rubin small tiles render 64 and 128, the
    native D128 rendering additionally admits explicit 64 on its double-score pipeline)."""
    selected = _selected_d_shape(capabilities, facts, knobs)
    if capabilities.sm_lo == 107 and supports_native_d128_narrow_tile(
        selected,
        device_cc=facts.device_cc,
        half_precision=facts.dtype in (cudnn.data_type.HALF, cudnn.data_type.BFLOAT16),
        thd=facts.thd,
        paged=facts.has_paged_kv,
        cga=knobs.cga if knobs is not None else None,
        split_kv=(knobs.split_kv or 1) if knobs is not None else 1,
        has_bias=facts.has_bias,
        has_kv_mask=facts.has_kv_mask,
    ):
        return frozenset({64, 128})
    if selected is not None:
        for shape, shape_domain in capabilities.tile_ns_by_d_shape:
            if shape == selected:
                return shape_domain
    return capabilities.tile_ns


def packed_thd_split(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> bool:
    """Whether these knobs split a THD graph through the prepared packed split host (``_thd_split_host``)."""
    split_cga = knobs.cga if knobs.cga is not None else (1 if (facts.d_qk, facts.d_v) == (64, 64) else 2)
    return (
        split_cga == (2 if (facts.d_qk, facts.d_v) in ((256, 256), SM100_NATIVE_D576_SHAPE) else 1)
        and (knobs.split_kv or 1) > 1
        and thd_split_domain(capabilities, facts)
    )


# CUDA caps gridDim.y and gridDim.z at 65535 (#1573).
_GRID_YZ_MAX = 65535


def launch_grid_mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: SdpaFwdKnobs) -> Optional[str]:
    """Why these knobs' launch grid puts more than 65535 on gridDim.y or .z, else ``None``.

    The NATURAL scheduler launches heads and batch x split_kv on y/z; the LPT
    policies and the SM100+/SM120 ragged grids are flat. The d256 decode tile
    keeps batch on y, the SM80/SM90 ragged grids keep heads and batch on y/z, and
    the split-KV combine keeps heads and batch on y/z, under every policy; the
    packed THD split combine keeps only heads on y. Packed
    GQA launches fewer head rows; ``h_q`` bounds them from above because the
    packing factor is per kernel flavor.
    """
    batch, heads = facts.b * (knobs.split_kv or 1), facts.h_q
    if max(batch, heads) <= _GRID_YZ_MAX:
        return None
    if packed_thd_split(capabilities, facts, knobs):
        # The packed combine launches ((t_q+3)//4, heads, 1) and the split attention a flat grid.
        if heads <= _GRID_YZ_MAX:
            return None
        return f"the packed split-KV combine launches heads={heads} on gridDim.y, past the CUDA limit of {_GRID_YZ_MAX}"
    if (knobs.split_kv or 1) > 1 and max(facts.b, heads) > _GRID_YZ_MAX:
        return f"the split-KV combine launches heads={heads} and batch={facts.b} on gridDim.y/z, past the CUDA limit of {_GRID_YZ_MAX}"
    if d256_decode_tile_selected(capabilities, facts, _decode_tile_pack_g(facts, knobs), _decode_tile_split_kv(knobs)):
        if facts.b <= _GRID_YZ_MAX:
            return None
        return f"the d256 decode tile launches batch={facts.b} on gridDim.y, past the CUDA limit of {_GRID_YZ_MAX}"
    if facts.thd:
        if capabilities.sm_lo not in (80, 90):
            return None
        return f"the ragged grid launches heads={heads} and batch={facts.b} on gridDim.y/z, past the CUDA limit of {_GRID_YZ_MAX}"
    if knobs.sched_policy not in (None, SCHED_NATURAL):
        return None
    return (
        f"SCHED_POLICY=NATURAL launches heads={heads} and batch x split_kv={batch} on gridDim.y/z, "
        f"past the CUDA limit of {_GRID_YZ_MAX}; the LPT policies launch a flat grid"
    )


def effective_cgas(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", split_kv: Optional[int] = None, cga: Optional[int] = None) -> frozenset[int]:
    """CGA domain of the native flavor and split leg selected by the graph.

    ``cga`` is the REQUESTED width when the caller validates one (``mismatch``):
    on a row with ``dense_only_d_shapes`` the flavor a request lands on depends
    on it (a cga2 request at D <= 32 selects the 64-wide tile, not the
    cga1-only 32-wide one), so the domain returned is that flavor's.  Without
    it (the heuristics' auto pick) the domain is the dense selection's."""

    _knobs = SdpaFwdKnobs(split_kv=split_kv, cga=cga) if (split_kv is not None or cga is not None) else None
    selected = _selected_d_shape(capabilities, facts, _knobs)
    if (
        capabilities.sm_lo == 107
        and not facts.attn_scale_prefolded
        and supports_nonpaged_d128_thd_prefill(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            cga=cga,
            split_kv=split_kv or 1,
        )
    ):
        return frozenset({1, 2})
    if capabilities.sm_lo == 107 and supports_paged_prefill_cga1(
        (facts.d_qk, facts.d_v),
        device_cc=facts.device_cc,
        fp8=facts.is_fp8 or facts.is_mxfp8,
        thd=facts.thd,
        paged=facts.has_paged_kv,
        split_kv=split_kv or 1,
        max_q=facts.s_q,
    ):
        return frozenset({1, 2})
    if (split_kv or 1) > 1 and thd_split_domain(capabilities, facts):
        return frozenset({2 if (facts.d_qk, facts.d_v) in ((256, 256), SM100_NATIVE_D576_SHAPE) else 1})
    if capabilities.sm_lo == 107 and thd_split_domain(capabilities, facts) and not facts.has_paged_kv and selected == (192, 128):
        return frozenset({1, 2})
    if (
        capabilities.sm_lo == 107
        and selected == (128, 128)
        and not (capabilities.is_fp8 or capabilities.is_mxfp8)
        and not rubin_dense_d128_shared_leg(capabilities, facts, _knobs)
    ):
        # cc 10.7 half row: the (128, 128) cga1 entry of cgas_by_d_shape names the shared SM100 DECODE tile compiled for
        # cc 10.7 (issue #1472), a DENSE half leg (rubin_dense_d128_shared_leg): a ragged graph outside the single-CTA
        # legs above keeps the cga2 prefill pipeline (the tile has no THD_VARLEN leg), dense paged queries are not wired
        # on cc 10.7, the pre-folded scale lives in the Rubin sibling only, and quantized facts never ride this row.
        # Narrow the DOMAIN so neither a proposal nor a pin reaches the tile.
        return frozenset({2})
    if capabilities.sm_lo == 107 and capabilities.is_mxfp8 and selected == (256, 256):
        # d256 MXFP8 on cc 10.7: the dense unsplit legs run the 2-CTA pair, everything else one CTA
        # (config_sm100.d256_mxfp8_pair_leg; heuristics.select_d256_auto_knobs picks the same width).
        pair = d256_mxfp8_pair_leg(device_cc=facts.device_cc, pertensor=False, thd=facts.thd, paged=facts.has_paged_kv, split_kv=split_kv or 1)
        return frozenset({1, 2}) if pair else frozenset({1})
    domain = capabilities.cgas
    if selected is not None:
        for shape, shape_domain in capabilities.cgas_by_d_shape:
            if shape == selected:
                domain = shape_domain
                break
        if (split_kv or 1) > 1:
            for shape, split_domain in capabilities.split_cgas_by_d_shape:
                if shape == selected:
                    domain = split_domain
                    break
    if (facts.has_score_max or facts.has_score_sum_exp) and 2 in domain:
        # The score side outputs are stored by the 2-CTA prefill tile only (the cga1 decode tile of the
        # d128 / d64 f16 flavors carries no store): narrow the domain so the heuristics propose the prefill
        # tile for a decode-shaped graph instead of a candidate mismatch() would decline.
        return frozenset({2})
    return domain


_amax_s_warned = False


def _warn_amax_s_declared_once() -> None:
    """Reached only past the device-arch check, so it fires where a FROST FP8 row would otherwise consider the graph."""
    global _amax_s_warned
    if _amax_s_warned:
        return
    _amax_s_warned = True
    logging.getLogger(__name__).warning(
        "sdpa_fp8: the graph declares the deprecated Amax_S output, which the FROST engines do not produce, so it runs on "
        "cuDNN backend engines only (up to ~2x slower on some shapes). Leave Amax_S virtual and set scale_s from the bounds in "
        "docs/operations/Attention.md, 'Amax S' (448 for E4M3 without dropout)."
    )


def _resident_arm_layout_decline(facts: "ga.SdpaGraphFacts") -> Optional[str]:
    """The resident-block arms (bias / kv_mask) bind their operands through the PREPARED launch only: the adapter's staging-copy
    executor (``prepared_staged_forward``, the path a Q/K/V/O layout TMA cannot bind zero-copy takes) forwards neither the bias nor the
    kv_mask, so such a graph used to BUILD and then fail at EXECUTE time (``kv_mask_tensor is required by this compiled specialization``
    on a 72-byte head stride, 2026-10-08).  A layout the prepared launch cannot bind is therefore declined HERE, typed and early, with
    the lowering's own admission arithmetic (``_prepared_decline_reason`` -> ``config_sm100.dense_bind_strides``); a fact set without
    declared operands (a pure-fact probe) is not judged -- the adapter's ``check_support`` is the twin on real descriptors (rule 8b)."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import dense_bind_strides

    for name, t in (("Q", facts.q_t), ("K", facts.k_t), ("V", facts.v_t), ("O", facts.o_t)):
        if t is None or not (hasattr(t, "get_dim") and hasattr(t, "get_stride") and hasattr(t, "get_data_type")):
            continue
        geometry = storage_geometry(t.get_dim(), t.get_stride(), t.get_data_type())
        if geometry is None or dense_bind_strides(*geometry, 2) is None:
            return (
                f"the resident-block arms (bias / kv_mask) bind through the prepared launch only, and {name}'s layout (dim {tuple(t.get_dim())}, "
                f"strides {tuple(t.get_stride())}) does not bind without a copy: 16-byte-aligned BSHD-physical strides are required"
            )
    return None


def _bias_arm_decline(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs]) -> Optional[str]:
    """The resident-bias / kv_mask arm's gates (a row with ``bias_d_shapes`` or ``kv_mask``): the legs it does not carry,
    the flavor the graph selects, the operands' layouts and -- for a graph WITH a bias -- the S_kv budget of the resident
    block (the kv_mask ALONE stages nothing in SMEM and carries no budget) -- at the plan's cga
    when the knobs carry one, else at ANY width of the flavor's domain (the heuristics' row-level probe: the row serves
    the graph when some width holds the block; ``heuristics._auto_sched_cga`` then picks the narrowest such width through
    ``config_sm107.sm107_bias_auto_cga`` and every proposed plan is re-asked here with its own cga) -- every one a typed
    decline, the twin of the adapter's ``check_support`` (rule 8b)."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_softmax_wgs_pick
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import BIAS_MODE_NONE, BIAS_MODE_STREAMED_MULTICAST, sm107_bias_mode_pick, sm107_bias_resident_s_kv_max
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import sm107_pair_item_active, sm107_tile_n_pick

    what = "additive bias" if facts.has_bias else "per-key kv_mask"
    if facts.thd or facts.has_paged_kv or facts.has_score_max or facts.has_score_sum_exp or facts.has_epilogue_gate:
        return f"the {what} is served on dense, unpaged plans without the score side outputs or the epilogue gate (no THD)"
    if knobs is not None and ((knobs.split_kv or 1) > 1 or knobs.pack_gqa):
        return f"the {what} is served on unsplit, non-PackGQA plans only"
    layout_why = _resident_arm_layout_decline(facts)
    if layout_why is not None:
        return layout_why
    selected = _selected_d_shape(capabilities, facts, knobs)
    shapes = capabilities.bias_d_shapes if capabilities.bias_d_shapes is not None else capabilities.d_shapes
    if selected not in shapes:
        wired = ", ".join(f"d{sq}" if sq == sv else f"d{sq}x{sv}" for sq, sv in sorted(shapes))
        return f"the {what} is wired on the {wired} kernel flavors only (the f16/bf16 small tiles); head dims ({facts.d_qk}, {facts.d_v}) select {selected}"
    if facts.has_bias:
        dims, st = facts.bias_dim, facts.bias_stride
        elem = 2
        if len(st) == 4 and st[3] != 1:
            return f"the bias must have its key dim contiguous (stride 1); got strides {st}"
        if len(st) == 4 and any(n > 1 and (s * elem) % 16 for n, s in zip(dims[:3], st[:3])):
            return f"every live bias stride must be a 16-byte multiple for TMA (S_kv % 8 == 0 for a compact bias); got dims {dims} strides {st}"
    if facts.has_kv_mask:
        if facts.kv_mask_dtype not in capabilities.kv_mask_dtypes:
            return f"kv_mask dtype {facts.kv_mask_dtype} not in {sorted(str(d) for d in capabilities.kv_mask_dtypes)}"
        if len(facts.kv_mask_stride) == 4 and facts.kv_mask_stride[3] != 1:
            return f"kv_mask must have its key dim contiguous (stride 1); got strides {facts.kv_mask_stride}"
        if facts.kv_mask_dtype != cudnn.data_type.BOOLEAN:
            # The ADDITIVE vector: the io dtype's 16-bit pairs (never the other half dtype) or fp32, read as warp-uniform
            # 16-byte granules -- a whole-granule extent (S_kv % 8 == 0; the tail granule is clamped in-kernel) and a 16-byte
            # batch stride (config_sm107.KV_MASK_ADDITIVE_GRANULE_ELEMS; the adapter's check_support and the binder read the same rule).
            if facts.kv_mask_dtype != cudnn.data_type.FLOAT and facts.kv_mask_dtype != facts.dtype:
                return f"an additive 16-bit kv_mask must be the Q/K/V dtype ({facts.dtype}) or fp32; got {facts.kv_mask_dtype}"
            if facts.s_kv % KV_MASK_ADDITIVE_GRANULE_ELEMS:
                return f"an additive kv_mask vector needs S_kv % {KV_MASK_ADDITIVE_GRANULE_ELEMS} == 0 (16-byte granule loads); got S_kv={facts.s_kv}"
            elem = ga._ELEM_BYTES.get(facts.kv_mask_dtype, 2)
            if len(facts.kv_mask_stride) == 4 and len(facts.kv_mask_dim) == 4 and facts.kv_mask_dim[0] > 1 and (facts.kv_mask_stride[0] * elem) % 16:
                return f"an additive kv_mask vector's batch stride must be a 16-byte multiple; got strides {facts.kv_mask_stride}"
            if facts.has_bias and facts.kv_mask_dtype == cudnn.data_type.FLOAT and capabilities.sm_lo == 100:
                # The fp32 vector WITH a bias spills on the 128-wide rendering on this line too (STL / LDL 53 / 64): served at the 64-wide KV
                # tile only (config_sm100.sm100_tile_n_pick lands a knob-less plan there; an EXPLICIT 128 is a typed decline, never degraded).
                _tile_n = knobs.tile_n if knobs is not None and knobs.tile_n is not None else SM100_FP32_VECTOR_WITH_BIAS_TILE_N
                if _tile_n != SM100_FP32_VECTOR_WITH_BIAS_TILE_N:
                    return (
                        f"an fp32 additive kv_mask together with a bias is served at TILE_N {SM100_FP32_VECTOR_WITH_BIAS_TILE_N} (the heuristics' pick; the "
                        f"TILE_N knob) or as a vector in the Q/K/V dtype ({facts.dtype}; cast the fp32 mask on the host): the 128-wide rendering spills on "
                        f"the softmax role; got TILE_N {_tile_n}"
                    )
            if facts.has_bias and facts.kv_mask_dtype == cudnn.data_type.FLOAT:
                # The fp32 vector WITH a bias is served at TILE_N 64 only (config_sm107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N: the 128-wide
                # rendering spills on the softmax role -- the measured counts sit on the constant).  The heuristics propose 64 for this
                # form (heuristics._resident_arm_tile_n) and the adapter defaults to it, so a knob-less plan and a record without the knob
                # run the served width; an EXPLICIT 128 is a typed decline, never a degraded plan.  The adapter's check_support is the
                # twin (rule 8b); the rendering pins of test_sdpa_bias_tile_n_additive_sm107.py carry the counts.
                _tile_n = knobs.tile_n if knobs is not None and knobs.tile_n is not None else SM107_FP32_VECTOR_WITH_BIAS_TILE_N
                if _tile_n != SM107_FP32_VECTOR_WITH_BIAS_TILE_N:
                    return (
                        f"an fp32 additive kv_mask together with a bias is served at TILE_N {SM107_FP32_VECTOR_WITH_BIAS_TILE_N} (the heuristics' pick; "
                        f"the TILE_N knob) or as a vector in the Q/K/V dtype ({facts.dtype}; cast the fp32 mask on the host): the 128-wide rendering "
                        f"spills on the softmax role; got TILE_N {_tile_n}"
                    )
    if capabilities.sm_lo == 100:
        # The SM100 line (cc 10.0 / 10.3): the RESIDENT block of the one-batch-entry item at TILE_N 128 -- the first body seam of the
        # port; the streamed ring, the (n, n+1) pair item and the 64-wide KV tile are not on the SM100 file yet, so a graph the resident
        # block does not hold at ANY width of the flavor's domain is a typed decline here (the adapter's check_support is the twin, rule
        # 8b), through config_sm100.sm100_bias_resident_s_kv_max -- the budget arithmetic the heuristics' width pick reads too (rule 8b').
        _tile_ns = effective_tile_ns(capabilities, facts, knobs)
        if knobs is not None and knobs.tile_n is not None and knobs.tile_n not in _tile_ns:
            return f"the {what} is served at TILE_N {sorted(_tile_ns)} on this arch line; got TILE_N {knobs.tile_n}"
        if not facts.has_bias:
            # The kv_mask ALONE stages nothing in SMEM: no block, no S_kv budget (the same statement as the Rubin branch below).
            return None
        if knobs is not None and knobs.cga is not None:
            widths = (knobs.cga,)
        else:
            widths = tuple(sorted(effective_cgas(capabilities, facts, knobs.split_kv if knobs is not None else None))) or (1,)
        # the budget pick per width: the resident block where it holds S_kv, else the streamed ring where >= 2 stages fit (the
        # multicast form needs the pair item, not on the SM100 file yet); no form at any width = the typed decline
        # the plan's KV tile width: the knob when the record carries one, else the SM100 pick (the row's, the heuristics' and the adapter's one arithmetic)
        _tile_n = (
            knobs.tile_n
            if knobs is not None and knobs.tile_n is not None
            else sm100_tile_n_pick(
                selected[0],
                facts.s_kv,
                has_bias=facts.has_bias,
                has_kv_mask=facts.has_kv_mask,
                kv_mask_fp32=facts.kv_mask_dtype == cudnn.data_type.FLOAT,
                padded=facts.padded,
                domain=_tile_ns,
            )
        )
        forms = {cga: sm100_bias_mode_pick(selected[0], cga, facts.s_kv, tile_n=int(_tile_n)) for cga in widths}
        if not any(f in (BIAS_MODE_RESIDENT, BIAS_MODE_STREAMED) for f in forms.values()):
            held = ", ".join(f"cga {cga}: resident S_kv <= {sm100_bias_resident_s_kv_max(selected[0], cga, tile_n=int(_tile_n))}" for cga in widths)
            return (
                f"no bias form fits at head-dim tile {selected[0]} on this arch line for S_kv {facts.s_kv} ({held}; no 2-stage streamed ring of "
                f"its tiles fits beside the operands); the pair item is not on the SM100 file yet"
            )
        return None
    if capabilities.sm_lo != 107 or not facts.has_bias:
        # The kv_mask ALONE stages nothing in SMEM (4 byte loads + 4 warp ballots per kv tile, folded into the bit-word mask):
        # no resident block, hence no S_kv budget -- any S_kv is served and the flavor's own width rule applies (validated on
        # cc 10.7 across the ring wrap, n_kv = 4 / 5 / 8).  Only a graph WITH a bias carries the block's budget below.
        return None
    if knobs is not None and knobs.cga is not None:
        widths = (knobs.cga,)  # the plan's width: the block it gets
    else:
        # Knob-less (the heuristics' row-level probe, a record without the knob): the row serves the graph when ANY width
        # of the flavor's domain holds the block; the plan-level call above re-asks with the proposed cga.
        domain = effective_cgas(capabilities, facts, knobs.split_kv if knobs is not None else None)
        widths = tuple(sorted(domain)) or (1,)
    # The block of the FORM the plan would run (config_sm107.sm107_bias_pair_form): a batch-broadcast bias on >= 2 flattened
    # rows at cga1 runs the (n, n+1) pair item whose 128-row block holds twice the kv columns (768 / 384 at d32 / d64); a
    # per-batch bias, a single row or a cga2 plan keeps the one-batch-entry item's 256-row block (384 / 128 / 256).  Above the
    # resident budget the STREAMED ring serves the graph wherever a 2- / 3-stage ring of the form's tiles fits beside the
    # operands (config_sm107.sm107_bias_mode_pick, one arithmetic with the heuristics' width pick and the adapter): every width
    # of the small tiles except the one-batch-entry item at the 64-wide tile's cga1 (two 64 KiB stages exceed its spare -- the
    # heuristics take cga2 there); a width where NEITHER form fits is the typed decline below.
    _bias_b = facts.bias_dim[0] if len(facts.bias_dim) == 4 else 1
    # the item form at the plan's KV tile width (the knob when the record carries one, else the heuristics' pick -- the one-batch-entry
    # guard of config_sm107.sm107_n_chunk_pick holds at 128 only; config_sm107.sm107_tile_n_pick is the heuristics' and the adapter's pick)
    _tile_n = (
        knobs.tile_n
        if knobs is not None and knobs.tile_n is not None
        else sm107_tile_n_pick(
            selected[0],
            facts.s_kv,
            has_bias=facts.has_bias,
            has_kv_mask=facts.has_kv_mask,
            padded=facts.padded,
            domain=effective_tile_ns(capabilities, facts, knobs),
        )
    )
    _pair = {
        cga: sm107_pair_item_active(
            _bias_b, facts.b, facts.h_q, facts.s_q, facts.device_sm_count or 0, cga, s_kv=facts.s_kv, d_flavor=selected[0], tile_n=int(_tile_n)
        )
        for cga in widths
    }
    # the softmax warp-group count the plan would run at each width (3 on the bias + kv_mask pair form at TILE_N 64 above the measured S_kv;
    # config_sm107.sm107_softmax_wgs_pick -- the adapter's pick, rule 8b'): its third Q / O slab shrinks the resident bias budget
    _wgs = {
        cga: sm107_softmax_wgs_pick(
            d_flavor=selected[0],
            s_kv=facts.s_kv,
            tile_n=int(_tile_n),
            pair=_pair[cga],
            has_bias=facts.has_bias,
            has_kv_mask=facts.has_kv_mask,
            padded=facts.padded,
            cga=cga,
            b_flat=facts.b,
        )
        for cga in widths
    }
    # the pair ring's cluster-multicast default needs global lengths and the NATURAL scheduler (the pick's two facts; a knob-less graph
    # derives NATURAL on this device) -- with either fact absent the pick is the cga1 ring, exactly the adapter's arithmetic.  GLOBAL
    # lengths = what the adapter will SEE: a dense S_kv tail (S_kv % skv_tile != 0, no mask covering it) is served through the kernel's
    # padded path with SYNTHESIZED per-batch lengths (_synth_kv_padding -- the lowering passes seq_kv_lens_present=True to the adapter,
    # whose _resident_arm_bias_mode then picks the cga1 ring), so the row reads the lowering's predicate too (rule 8b: one form per plan).
    _natural = knobs is None or knobs.sched_policy is None or knobs.sched_policy == SCHED_NATURAL
    _global_lengths = not (facts.padded or facts.seq_q_trim or _synth_kv_padding(capabilities, facts))
    modes = {
        cga: sm107_bias_mode_pick(selected[0], cga, facts.s_kv, pair=_pair[cga], tiles_q=_wgs[cga], global_lengths=_global_lengths, natural=_natural)
        for cga in widths
    }
    if any(mode == BIAS_MODE_STREAMED_MULTICAST for mode in modes.values()):
        # The multicast bias ring (bias_mode 3: the pair ring's measured default at D 32, or the test seam's / the adapter's request):
        # ONE kv-tile sequence streamed to both CTAs of a cluster, so the two chunks' step bounds must be the same function of (q tile,
        # S_q, S_kv) -- per-batch lengths break that lockstep (the synthesized ones of a dense S_kv tail included: the adapter declines
        # them typed, so the row does too); the chunk pairing rides the NATURAL payload (the config validator pins both).
        if not _global_lengths:
            return (
                "the multicast bias ring (bias_mode 3) streams one kv-tile sequence to both CTAs of a cluster; per-batch lengths "
                f"({'synthesized for the dense S_kv tail ' + str(facts.s_kv) if not (facts.padded or facts.seq_q_trim) else 'real'}) are served by the cga1 ring only"
            )
        if knobs is not None and knobs.sched_policy is not None and knobs.sched_policy != SCHED_NATURAL:
            return f"the multicast bias ring (bias_mode 3) pairs chunks through the NATURAL scheduler's payload; got sched_policy={knobs.sched_policy}"
    if all(mode == BIAS_MODE_NONE for mode in modes.values()):
        budgets = {cga: sm107_bias_resident_s_kv_max(selected[0], cga, pair=_pair[cga], tiles_q=_wgs[cga]) for cga in widths}
        held = " / ".join(f"cga {cga}: S_kv <= {s_kv_max}" for cga, s_kv_max in budgets.items())
        return (
            f"no bias form fits at head-dim tile {selected[0]} for S_kv={facts.s_kv}: the resident bias block holds {held} (the SMEM left beside the Q/O "
            f"slabs and the K/V ring; {'the (n, n+1) pair item' if any(_pair.values()) else 'the one-batch-entry item'}) and no 2-stage streamed "
            f"ring of its tiles fits beside the operands at {'these widths' if len(widths) > 1 else 'this width'}"
        )
    return None


def mismatch(capabilities: Capabilities, facts: "ga.SdpaGraphFacts", knobs: Optional[SdpaFwdKnobs] = None) -> Optional[str]:
    """First reason this engine is not a candidate for these facts and tuning
    knobs, or ``None`` when lowering should perform the final feasibility check.

    Returns a human-readable reason string rather than a bool on purpose:
    with many engine rows, "why was my engine not eligible" is the first
    debugging question, and the strict-select error surfaces this string.
    Knob requests outside the engine's candidate domain are rejected here;
    architecture-specific resource checks remain in the adapter and must never
    silently degrade a requested value.
    """
    if facts.invalid:
        return facts.invalid
    if facts.is_backward:
        return "this engine serves sdpa() forward graphs only"
    if facts.softmax_precision is not None and facts.softmax_precision not in capabilities.softmax_precisions:
        # The op attribute sdpa(softmax_precision=HALF): numerics-changing, so
        # honored only by a row whose lowering carries that arm — never degraded.
        return f"requested softmax_precision={facts.softmax_precision} is outside this engine's domain {sorted(capabilities.softmax_precisions, key=int)}"
    if facts.attn_scale_prefolded:
        # The op attribute sdpa(attn_scale_prefolded=True): Q already carries
        # attn_scale * log2(e) and the kernel must trace no per-score scale --
        # served only by a row whose SELECTED flavor carries that arm, never
        # degraded to the scaled chain (that would scale twice).
        if capabilities.attn_scale_prefolded_d_shapes is None:
            return "attn_scale_prefolded (Q pre-multiplied by attn_scale * log2 e) is not wired in this engine's kernels"
        if facts.has_paged_kv:
            return "attn_scale_prefolded is not wired in the paged-KV kernel bodies"
        if _selected_d_shape(capabilities, facts) not in capabilities.attn_scale_prefolded_d_shapes:
            return (
                f"attn_scale_prefolded is wired only in the {sorted(capabilities.attn_scale_prefolded_d_shapes)} kernel flavors; "
                f"graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
    if knobs is not None:
        if not isinstance(knobs, SdpaFwdKnobs):
            return f"knob request is a {type(knobs).__name__}, not SdpaFwdKnobs — wrong operation's vocabulary"
        for value, domain, label in (
            (knobs.sched_policy, effective_sched_policies(capabilities, facts, knobs), "sched_policy"),
            (knobs.tile_m, capabilities.tile_ms, "tile_m"),
            (knobs.tile_n, effective_tile_ns(capabilities, facts, knobs), "tile_n"),
            (knobs.cga, effective_cgas(capabilities, facts, knobs.split_kv, knobs.cga), "cga"),
            (knobs.pack_gqa, capabilities.pack_gqas, "pack_gqa"),
        ):
            if value is not None and value not in domain:
                return f"requested {label}={value} is outside this engine's domain {sorted(domain, key=int)}"
        grid_reason = launch_grid_mismatch(capabilities, facts, knobs)
        if grid_reason is not None:
            return grid_reason
        sm120_tile_reason = _sm120_tile_mismatch(capabilities, facts, knobs)
        if sm120_tile_reason is not None:
            return sm120_tile_reason
        # D128 half cga1 uses the decode tile for ragged Q=1 and the split
        # host, or two-slab prefill for admitted paged graphs and native Rubin
        # nonpaged exact D128. effective_cgas gates the unsplit prefill domain.
        # api_dsl.check_support mirrors these lines (keep them in lockstep).
        ragged_decode = knobs.cga == 1 and facts.thd and _thd_decode_leg(capabilities, facts)
        packed_split = packed_thd_split(capabilities, facts, knobs)
        ragged_pack = knobs.pack_gqa and supports_nonpaged_d128_thd_pack_gqa(
            (facts.d_qk, facts.d_v),
            device_cc=facts.device_cc,
            fp8=facts.is_fp8 or facts.is_mxfp8,
            thd=facts.thd,
            paged=facts.has_paged_kv,
            cga=knobs.cga,
            split_kv=knobs.split_kv or 1,
        )
        # The d256 DECODE tile (sm107/decode_d256_f16.py) packs the WHOLE group into its 16-row Q tile
        # over a dense or paged cache (HEADS_PER_TILE = QH_PER_KH) and writes the SM100 tile's fp32 split
        # partials for sm100/split_combine, so a decode-shaped half d256 graph packs and splits whatever
        # its cache form; every other Rubin half packing / d256 split rides the paged / packed-split
        # pipelines.  The four Rubin gates below read this ONE predicate (api_dsl.check_support mirrors
        # each exemption through _decode_q_tile_for, rule 8b); past the tile the d256 prefill kernel
        # serves the graph, and it wires neither PackGQA nor a dense split.  A GATED graph rides the
        # tile only when this set splits (the gate moves into the split combine), so the set's split
        # is part of the question.
        decode_tile = d256_decode_tile_selected(capabilities, facts, _decode_tile_pack_g(facts, knobs), _decode_tile_split_kv(knobs))
        if (
            capabilities.sm_lo == 107
            and not (facts.is_fp8 or facts.is_mxfp8)
            and knobs.pack_gqa
            and not facts.has_paged_kv
            and not packed_split
            and not decode_tile
            and not rubin_dense_d128_shared_leg(capabilities, facts, knobs)
            and not ragged_pack
            and not _config_sm107.supports_native_d512_dense_pack_split(
                _selected_d_shape(capabilities, facts, knobs),
                device_cc=facts.device_cc,
                fp8=facts.is_fp8 or facts.is_mxfp8,
                thd=facts.thd,
                paged=facts.has_paged_kv,
                cga=knobs.cga,
            )
        ):
            return "Rubin half PackGQA requires paged KV, the D128 packed split, an unsplit dense or ragged D128 GQA graph without the pre-folded scale, dense D512, or the d256 decode tile (a decode-shaped graph: S_q x G packed rows within its routed envelope)"
        if (
            facts.attn_scale_prefolded
            and capabilities.sm_lo == 107
            and not (facts.is_fp8 or facts.is_mxfp8)
            and facts.thd
            and (packed_split or ragged_pack or (knobs.cga == 1 and _selected_d_shape(capabilities, facts) == (192, 128)))
        ):
            # Preserve the existing public packing / single-Q attribute domain.
            # Unsplit D128 packing now lowers to native SM107; this route migration
            # does not expand the pre-folded-scale combinations.
            return "attn_scale_prefolded is not admitted on the half THD packed split / D128 packing / D192 single-Q legs"
        if packed_split and not getattr(
            cudnn._pybind_module._SdpaThdBinder,
            (
                "supports_paged_d64_packed_split"
                if facts.d_v == 64
                else (
                    ("supports_paged_d256_packed_split" if facts.has_paged_kv else "supports_nonpaged_d256_packed_split")
                    if facts.d_v == 256
                    else (
                        "supports_paged_packed_split"
                        if facts.has_paged_kv
                        else ("supports_nonpaged_d128_packed_split" if facts.d_qk == 128 else "supports_nonpaged_packed_split")
                    )
                )
            ),
            False,
        ):
            return "packed split requires the matching native cuDNN Frontend extension"
        if (
            knobs.cga == 1
            and facts.thd
            and not (
                ragged_decode
                or packed_split
                or supports_paged_prefill_cga1(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    split_kv=knobs.split_kv or 1,
                    max_q=facts.s_q,
                )
            )
            and capabilities.sm_lo == 100
            and _selected_d_shape(capabilities, facts, knobs) == (128, 128)
        ):
            return (
                "cga=1 on the d128 flavor supports the decode tile for ragged Q over paged K/V with ragged Stats at S_q == 1, "
                "exact D128 with split_kv > 1, or the SM100 paged prefill body at S_q > 1; other THD graphs run the cga2 prefill tile"
            )
        if ragged_decode and (knobs.split_kv is None or knobs.split_kv < 2):
            # The ragged final rows exist only through the combine pass.
            return "the d128 decode tile's ragged-Q leg rides the split path (the combine places the ragged O / Stats rows); pin split_kv >= 2"
        if knobs.split_kv is not None and knobs.split_kv < 1:
            return f"requested split_kv={knobs.split_kv} is not a split count (1 = off)"
        if knobs.split_kv is not None and knobs.split_kv > 1:
            if not capabilities.split_kv_supported:
                return "split_kv > 1 is not wired in this engine's lowering"
            if facts.has_epilogue_gate and not decode_tile:
                if facts.shape_overrides and d256_decode_tile_selected(
                    capabilities, replace(facts, shape_overrides=False), _decode_tile_pack_g(facts, knobs), knobs.split_kv
                ):
                    # The decode tile's split WOULD carry this gate, but its combine binds G to the plan's declared
                    # (B, H_q, S_q, D_v) (prepared.CombineGate) while an override-enabled graph may change either at
                    # execute: declined here by name, never a bind failure at launch.
                    return (
                        "split_kv > 1 with the fused epilogue gate rides the d256 decode tile's split combine only, whose gate binding is fixed to the "
                        "declared (B, H_q, S_q, D_v): a graph with execute-time shape overrides cannot use it (its unsplit plan keeps the fused-gate kernel)"
                    )
                # The combine pass writes the recombined O from un-gated partials;
                # the gate lives in the unsplit kernel's epilogue -- except on the
                # d256 decode tile, whose split carries the gate IN the combine
                # (sm100/split_combine gate=True; api_dsl._gate_in_combine).
                return "split_kv > 1 cannot ride the fused epilogue gate (the combine would write the un-gated O); only the d256 decode tile's split applies the gate in its combine"
            native_fp8_padded_split = supports_fp8_d256_padded_split(
                _selected_d_shape(capabilities, facts, knobs),
                device_cc=facts.device_cc,
                pertensor=facts.is_fp8,
                thd=facts.thd,
                paged=facts.has_paged_kv,
            )
            # Facts x knobs: the split path is structurally dense-only (the
            # per-split LSE is the combine weight; the THD/sink/padded paths
            # do not produce per-split partials). Declined HERE so a split
            # request never reaches a kernel that cannot honor it.
            # Paged KV is padded by construction and its split composes with
            # the per-batch lengths (the decode lever when B*H_kv leaves the
            # machine underfilled), so it is exempt from the padded exclusion.
            if (
                (facts.thd and not (ragged_decode or packed_split))
                # The SM100, SM107 and SM120 half flavors write sink-free partials and fold
                # the sink once in the combine. Qualified per-tensor FP8 plans
                # compile sink-free partials too; a gated split carries no sink, and
                # the packed THD split carries one only where thd_split_domain admits
                # it (supports_thd_split_sink).
                or (
                    facts.has_sink
                    and not packed_split
                    and (
                        (
                            (facts.is_fp8 or facts.is_mxfp8)
                            and not supports_quantized_split_sink(device_cc=facts.device_cc, pertensor=facts.is_fp8, thd=facts.thd)
                        )
                        or facts.has_epilogue_gate
                        or capabilities.sm_lo not in (100, 107, 120)
                    )
                )
                or (
                    facts.padded
                    and not facts.has_paged_kv
                    and not packed_split
                    and not native_fp8_padded_split
                    and _selected_d_shape(capabilities, facts, knobs) != SM100_NATIVE_D576_SHAPE
                )
                or (
                    facts.seq_q_trim
                    and not (facts.padded and (native_fp8_padded_split or _selected_d_shape(capabilities, facts, knobs) == SM100_NATIVE_D576_SHAPE))
                )
                # A split sink over padded query lengths also needs the native dense binder that hands the combine the Q lengths (a padded
                # row must stay O := 0 / LSE := -inf): an older extension declines it here rather than at the spec build.
                or (
                    facts.has_sink
                    and not packed_split
                    and not facts.thd
                    and facts.padded
                    and facts.seq_q_t is not None
                    and not getattr(cudnn._pybind_module._SdpaDenseBinder, "supports_combine_q_lens", False)
                )
            ):
                return "split_kv > 1 serves sink-free dense graphs (sinks also split on SM100/SM103/SM107 per-tensor FP8 and SM100/SM107/SM120 half), the decode tile's ragged-Q leg, or supported native packed splits"
            if (
                facts.has_sink
                and (facts.is_fp8 or facts.is_mxfp8)
                and not getattr(cudnn._pybind_module._SdpaDenseBinder, "supports_quantized_split_sink", False)
            ):
                return "FP8 split attention with a sink needs the native quantized sink-combine binder; rebuild the extension"
            if (
                _synth_kv_padding(capabilities, facts)
                and not native_fp8_padded_split
                and not supports_scalar_kv_tail_split(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    pertensor=facts.is_fp8,
                )
            ):
                return "split_kv > 1 cannot ride the KV-tail mask this S_kv needs"
            # No gate on the O dtype: the partials are never narrower than it,
            # and the combine performs the only cast down to it.
            # _selected_d_shape, not an envelope walk over the raw dims: the
            # set names the flavors whose KERNELS wire SplitHelpers, and the
            # lowering picks the smallest covering one. An envelope test says
            # (64, 64) "fits" (128, 128) and admits a split the d64 kernel
            # cannot serve, so the plan would clear eligibility and then die in
            # the lowering (contract rule 8b'). Mirrors the pack_gqa gate below.
            # ... plus the d256 decode tile's dense / paged split (fp32 partials into the split-major
            # workspace the shared combine reduces, packed or not). Native per-tensor FP8
            # D256 prefill also has dense FP32 partials; half prefill keeps its THD split.
            if (
                capabilities.sm_lo == 107
                and not facts.is_fp8
                and _selected_d_shape(capabilities, facts, knobs) == (256, 256)
                and not packed_split
                and not decode_tile
            ):
                return "SM107 D256 split is qualified for half THD and the d256 decode tile (a decode-shaped graph: S_q x G packed rows within its routed envelope) only"
            # With the knobs: a dense-only tile (the Rubin d32 / d64) is walked
            # past, so a split graph at D <= 64 selects the d128 flavor it
            # rode before the tiles -- served, not declined.
            if capabilities.split_d_shapes is not None and _selected_d_shape(capabilities, facts, knobs) not in capabilities.split_d_shapes:
                return f"split_kv > 1 is wired only in the {sorted(capabilities.split_d_shapes)} kernel flavors; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
        if knobs.pack_gqa and capabilities.pack_gqa_d_shapes is not None:
            # _selected_d_shape, not the raw dims: the FP8 rows carry
            # d_pad_multiple=16, so an inexact graph (D=112) lowers onto the
            # smallest covering NATIVE flavor (d128) -- which is the kernel
            # whose PackGQA wiring the set is describing.
            if _selected_d_shape(capabilities, facts, knobs) not in capabilities.pack_gqa_d_shapes:
                return f"pack_gqa is wired only in the {sorted(capabilities.pack_gqa_d_shapes)} kernel flavors; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            # On the Rubin half row the (256, 256) entry is the decode tile and the paged THD d256 prefill
            # (CGA2, unsplit): the dense d256 prefill kernel runs unpacked, so a packed dense d256 graph is
            # honorable exactly when its whole group rides the decode tile.  The heuristics' _pack_gqa_eligible
            # proposes under the same predicate -- a proposal
            # declined here would leave the engine offering NOTHING whenever its base leg splits.
            if (
                capabilities.sm_lo == 107
                and not (facts.is_fp8 or facts.is_mxfp8)
                and _selected_d_shape(capabilities, facts) == (256, 256)
                and not decode_tile
                and not supports_paged_d256_pack_gqa(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    cga=knobs.cga,
                    split_kv=knobs.split_kv or 1,
                )
            ):
                return "Rubin half PackGQA at D256 is wired on the decode tile (a decode-shaped dense graph: S_q x G packed rows within its routed envelope) and on the paged half THD prefill (CGA2, unsplit) only; the dense d256 prefill kernel runs unpacked"
        if knobs.pack_gqa:
            if (
                capabilities.sm_lo == 107
                and not (capabilities.is_fp8 or capabilities.is_mxfp8)
                and _selected_d_shape(capabilities, facts) == (256, 256)
                and not supports_paged_d256_pack_gqa(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    cga=knobs.cga,
                    split_kv=knobs.split_kv or 1,
                )
                and not decode_tile
            ):
                return "SM107 D256 PackGQA requires exact paged half THD with CGA2 and no split, or the d256 decode tile (a decode-shaped dense graph: S_q x G packed rows within its routed envelope)"
            if (
                facts.thd
                and not ragged_decode
                and not (packed_split and (facts.d_qk, facts.d_v) == (64, 64))
                and not supports_paged_d256_pack_gqa(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    cga=knobs.cga,
                    split_kv=knobs.split_kv or 1,
                )
                and (facts.d_qk, facts.d_v) not in capabilities.thd_pack_gqa_d_shapes
            ):
                return "PackGQA is not supported for this THD/ragged flavor (except the decode tile's ragged-Q leg)"
            if capabilities.is_mxfp8 and facts.o_block_scale:
                return "PackGQA on the MXFP8 d128 flavor serves a plain (not block-scaled) O only"
            if facts.has_epilogue_gate and not decode_tile:
                # The gate tile is one TMA box per (head, Q tile); a packed
                # tile interleaves (token, head) rows the box cannot address.
                # Not on the d256 decode tile's split: its gate is applied by
                # the combine, element-wise on the real O.
                return "PackGQA cannot ride the fused epilogue gate (only the d256 decode tile's split applies the gate in its combine)"
            if knobs.tile_m is not None:
                _pg_tile_m = knobs.tile_m
            elif capabilities.sm_lo == _BLACKWELL_GEFORCE[0]:
                # The adapter's own Q tile: the flavor's largest (one 64-row tile on the D=512 templates).
                _pg_tile_m = max(m for m, _ in _sm120_tile_domain(facts.d_qk, facts.d_v, facts.is_fp8 or facts.is_mxfp8))
            else:
                _pg_tile_m = max(capabilities.tile_ms)
            _partial = (
                pack_gqa_partial(capabilities, facts)
                or ragged_pack
                or supports_paged_d128_partial_pack_gqa(
                    (facts.d_qk, facts.d_v),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    cga=knobs.cga,
                    split_kv=knobs.split_kv or 1,
                )
                or _config_sm107.supports_native_d128_dense_partial_pack(
                    _selected_d_shape(capabilities, facts, knobs),
                    device_cc=facts.device_cc,
                    fp8=facts.is_fp8 or facts.is_mxfp8,
                    thd=facts.thd,
                    paged=facts.has_paged_kv,
                    cga=knobs.cga,
                )
            )
            # The decode tile packs the WHOLE group into its 16-row Q tile whatever tile_m (24/2: 12 live
            # rows + 4 zero tail rows), so the prefill tiles' divisibility rule does not apply to a Rubin
            # graph that lowers onto it (api_dsl.check_support exempts the same graphs).  The SM100 line
            # keeps the rule as is: its partial PackGQA already admits such groups on the prefill tile.
            _decode_packs_whole_group = capabilities.sm_lo == 107 and decode_tile and facts.h_kv > 0 and facts.h_q % facts.h_kv == 0
            _requires_divisible = capabilities.pack_gqa_requires_divisible
            if (
                not pack_gqa_supported(facts.h_q, facts.h_kv, _pg_tile_m, partial=_partial, requires_divisible=_requires_divisible)
                and not _decode_packs_whole_group
            ):
                return (
                    f"PackGQA requires h_q/h_kv to {'not exceed' if not _requires_divisible else 'share a factor with' if _partial else 'divide'} tile_m: "
                    f"h_q/h_kv = {facts.h_q}/{facts.h_kv} does not pack at tile_m={_pg_tile_m}"
                )
        if facts.has_score_max or facts.has_score_sum_exp:
            # Facts x knobs for the score side outputs: the stores index the dense (B, H_q, S_q) row
            # from the PREFILL tile's correction epilogue.  A split's partials have no such slot (the
            # combine writes no score outputs) and the decode tile (cga=1 on the d128 / d64 f16
            # flavors) carries no store.  api_dsl.check_support mirrors both (keep them in lockstep).
            if knobs.split_kv is not None and knobs.split_kv > 1:
                return "score_max / score_sum_exp cannot be combined with split_kv > 1 (the combine writes no score outputs)"
            if knobs.cga == 1:
                return "score_max / score_sum_exp are stored by the 2-CTA prefill tile; cga=1 selects a tile without the store"
    cc = facts.device_cc
    sm = None if cc is None else cc[0] * 10 + cc[1]
    if sm is None or not (capabilities.sm_lo <= sm <= capabilities.sm_hi):
        return f"requires SM{capabilities.sm_lo}-{capabilities.sm_hi}; current device is {cc}"
    installed, version = cutedsl_state()
    if not installed:
        # Said HERE, not when lowering imports the adapter: a decline at build
        # is honest but late -- the plan is already in the ranked list.
        return "requires the cutedsl extra (nvidia-cutlass-dsl), which is not installed"
    if cutedsl_too_old(version):
        want = ".".join(str(v) for v in CUTEDSL_MIN_VERSION)
        return f"requires nvidia-cutlass-dsl >= {want}; found {version[1]}"
    arch_error = cutedsl_arch_requirement_error(cc)
    if arch_error is not None:
        return arch_error
    shapes = sorted(capabilities.d_shapes)
    if capabilities.d_pad_multiple and (facts.d_qk, facts.d_v) not in capabilities.d_shapes:
        # Envelope family: native flavor shapes are upper bounds (TMA
        # zero-padding semantics — see Capabilities.d_shapes/d_pad_multiple);
        # the lowering picks the smallest covering flavor. An EXACT native
        # shape is served above, so only inexact graphs walk the envelope —
        # where a per-shape floor (d_envelope_floors) may also apply.
        floors = dict(capabilities.d_envelope_floors)
        if not any(facts.d_qk <= sq and facts.d_v <= sv and min(facts.d_qk, facts.d_v) > floors.get((sq, sv), 0) for sq, sv in capabilities.d_shapes):
            return f"no kernel-flavor envelope covers (D_QK={facts.d_qk}, D_V={facts.d_v}); native shapes: {shapes}"
        m = capabilities.d_pad_multiple
        if m > 1 and (facts.d_qk % m != 0 or facts.d_v % m != 0):
            return (
                f"envelope zero-padding requires D_QK/D_V multiples of {m} (TMA 16-byte global-stride constraint); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
    elif not capabilities.d_pad_multiple and (facts.d_qk, facts.d_v) not in capabilities.d_shapes:
        return f"serves exact native shapes {shapes} (no envelope padding); graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
    if _selected_d_shape(capabilities, facts, knobs) == SM100_NATIVE_D576_SHAPE and not getattr(
        cudnn._pybind_module._SdpaDenseBinder, "supports_native_d576", False
    ):
        return "native D576x512 requires the matching cuDNN Frontend native extension"
    if (
        facts.thd
        and _selected_d_shape(capabilities, facts, knobs) == SM100_NATIVE_D576_SHAPE
        and not getattr(cudnn._pybind_module._SdpaThdBinder, "supports_native_d576", False)
    ):
        return "native D576x512 THD requires the matching cuDNN Frontend native extension"
    if (
        facts.thd
        and _selected_d_shape(capabilities, facts, knobs) == SM100_NATIVE_D576_SHAPE
        and knobs is not None
        and (knobs.split_kv or 1) > 1
        and not getattr(cudnn._pybind_module._SdpaThdBinder, "supports_native_d576_packed_split", False)
    ):
        return "native D576x512 THD split requires the matching cuDNN Frontend native extension"
    if facts.thd and _selected_d_shape(capabilities, facts, knobs) in capabilities.thd_excluded_d_shapes:
        return "this native head-dimension flavor requires dense Q; THD origins are not wired"
    if facts.thd and capabilities.thd_d_shapes is not None and (facts.d_qk, facts.d_v) not in capabilities.thd_d_shapes:
        return f"THD (ragged) rides the packed native-tile leg on this engine (shapes {sorted(capabilities.thd_d_shapes)}); the head-dim envelope is dense-only"
    if facts.thd and capabilities.sm_lo == capabilities.sm_hi == 80 and capabilities.phase == "prefill":
        if knobs is not None and knobs.sched_policy not in (None, SCHED_NATURAL):
            return "SM80 THD forward requires the natural scheduler (the varlen grid is 3-D)"
        for name, tensor in (("Q", facts.q_t), ("K", facts.k_t), ("V", facts.v_t), ("O", facts.o_t)):
            dim, stride = tuple(tensor.get_dim()), tuple(tensor.get_stride())
            if (stride[2], stride[1], stride[3]) != (dim[1] * dim[3], dim[3], 1):
                return f"SM80 THD {name} requires compact packed BSHD strides"
        if (facts.seq_q_t is None and facts.cu_seq_q_t is None) or (facts.seq_kv_t is None and facts.cu_seq_kv_t is None):
            return "SM80 THD forward requires Q and KV lengths (seq_len_* or cu_seq_len_*)"
        if facts.has_bias or facts.has_sink:
            return "SM80 THD forward does not support bias or sink fusion"
    if facts.s_q == 1 and not capabilities.decode:
        return "s_q == 1 (decode) is out of scope for the SM80 prefill kernels"
    if facts.dtype not in capabilities.dtypes:
        return f"dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.dtypes)}"
    if (facts.is_mxfp8, facts.is_fp8) != (capabilities.is_mxfp8, capabilities.is_fp8):
        quant = "block-scale MXFP8 (sdpa_mxfp8)" if capabilities.is_mxfp8 else "per-tensor FP8 (sdpa_fp8)" if capabilities.is_fp8 else "half (sdpa)"
        return f"this engine serves only {quant} graphs"
    if (capabilities.is_fp8 or capabilities.is_mxfp8) and facts.dtype_o not in capabilities.out_dtypes:
        return f"O dtype {facts.dtype_o} not in {sorted(str(d) for d in capabilities.out_dtypes)}"
    if facts.o_block_scale not in capabilities.o_block_scales:
        return f"block-scaled O (scale block {facts.o_block_scale} along d) is not served by this engine (domain {sorted(capabilities.o_block_scales)})"
    if facts.dtype_o == cudnn.data_type.FP4_E2M1 and facts.o_block_scale != 16:
        # FP4_E2M1 sits in out_dtypes for the block-scaled epilogue only (the
        # analyzer derives o_block_scale = 16 from sf_o); a bare FP4 O has no store.
        return "an FP4_E2M1 O is served only as a block-scaled O (sf_o with 16-element scale blocks)"
    if facts.o_block_scale:
        # The block-scaled epilogue writes SF_O per dense Q row of one
        # sequence; THD / per-batch Q trim / a KV split (fp32 partials) / a
        # packed GQA tile all break that row <-> scale-factor mapping.
        if facts.thd or facts.seq_q_trim:
            return "block-scaled O (sf_o) serves dense, untrimmed Q rows only"
        if knobs is not None and knobs.split_kv is not None and knobs.split_kv > 1:
            return "block-scaled O (sf_o) cannot be combined with split_kv > 1"
        if knobs is not None and knobs.pack_gqa:
            return "block-scaled O (sf_o) cannot be combined with pack_gqa"
        if facts.has_epilogue_gate:
            # Two different epilogues own the O store (quantize + SF_O vs. O *= sigmoid(G)).
            return "block-scaled O (sf_o) cannot be combined with the fused epilogue gate"
    if not facts.uniform_dtype:
        return "K/V dtypes must match Q" if (facts.is_mxfp8 or facts.is_fp8) else "K/V/O dtypes must match Q"
    if facts.thd:
        # The THD lowering rebuilds packed [1, T, H, D] views from the token,
        # head and element strides; the batch stride is never read (every
        # sequence base comes from the ragged offsets), so it is not gated --
        # FlashInfer declares it equal to the token stride.
        if not facts.packed_layout:
            return "THD (ragged) Q/K/V/O must be BSHD-physical over (H, S, D): head dim innermost, then heads, then tokens"
    elif "dense_flex" in capabilities.layouts:
        if not facts.dense_layout:
            return (
                "Q/K/V/O must have the head dim innermost-contiguous (stride 1) and "
                "non-broadcast, non-overlapping strides (any B/H/S order, padded strides allowed)"
            )
    elif "bshd" in capabilities.layouts and not facts.bshd_layout:
        return "Q/K/V/O must be BSHD-physical (stride order 3,1,2,0)"

    for fact, cap, label in (
        (facts.has_bias, capabilities.bias, "bias"),
        (facts.has_dropout, capabilities.dropout, "dropout"),
        (facts.has_score_mod, capabilities.score_mod, "score_mod"),
        (facts.has_paged_kv, capabilities.paged_kv, "paged attention"),
        (facts.has_alibi, capabilities.alibi, "ALiBi"),
        (facts.has_block_mask, capabilities.block_mask, "block_mask"),
        (facts.has_rng_dump, capabilities.rng_dump, "rng_dump"),
        (facts.has_kv_mask, capabilities.kv_mask, "kv_mask"),
        (facts.has_score_max, capabilities.score_max, "score_max output"),
        (facts.has_score_sum_exp, capabilities.score_sum_exp, "score_sum_exp output"),
        (facts.dynamic_scale, capabilities.dynamic_scale, "tensor attn_scale"),
        (facts.scale == 0.0, capabilities.zero_scale, "attn_scale = 0"),
        (facts.has_unfuse_fma, capabilities.unfuse_fma, "unfuse_fma"),
        (facts.has_stats_log2, capabilities.stats_log2, "stats_use_log2 (base-2 stats)"),
        (facts.seq_q_trim, capabilities.seq_q_trim, "seq_len_q without padding mask"),
        (facts.right_band_widening, capabilities.right_band_widening, "causal right-band widening"),
        (facts.causal, capabilities.causal, "causal mask"),
        (facts.window_left is not None, capabilities.swa, "sliding window"),
        (facts.padded, capabilities.padded, "padding mask"),
        (facts.has_sink, capabilities.sink, "sink token"),
        (facts.wants_stats, capabilities.stats, "stats output"),
        (facts.thd, capabilities.thd, "THD / ragged"),
    ):
        if fact and not cap:
            return f"graph uses {label}, which this engine does not support"

    if facts.has_score_max or facts.has_score_sum_exp:
        # The score side outputs are a dense-row store of the prefill epilogue (the row-level flags
        # above admitted the ports): no THD slot, the paged specialization is not validated with
        # them, and only the flavors in score_stats_d_shapes carry the store -- on the SELECTED
        # flavor, like paged_d_shapes, so an envelope-served graph lands on the kernel the set names.
        if facts.thd:
            return "score_max / score_sum_exp are served on dense graphs only (no THD / ragged)"
        if facts.has_paged_kv:
            return "score_max / score_sum_exp are not wired on the paged-KV flavors"
        if capabilities.score_stats_d_shapes is not None:
            # The ports are graph facts, so even a knob-less selection walks past a dense-only tile: a Rubin
            # d24 / d64 graph with the outputs selects the d128 flavor that stores them (its pre-tile route).
            selected = _selected_d_shape(capabilities, facts, knobs)
            if selected not in capabilities.score_stats_d_shapes:
                wired = ", ".join(f"d{sq}" if sq == sv else f"d{sq}x{sv}" for sq, sv in sorted(capabilities.score_stats_d_shapes))
                return f"score_max / score_sum_exp are wired on the {wired} kernel flavors only; head dims ({facts.d_qk}, {facts.d_v}) select {selected}"

    if facts.has_epilogue_gate:
        # The sdpa -> sigmoid(G) -> mul tail.  Every condition below is a
        # DECLINE with a reason, never facts.invalid: the graph is legal for the
        # backend (three ordinary nodes); only the fused lowering is particular.
        if not capabilities.epilogue_gate:
            return "graph fuses O * sigmoid(G) into the sdpa epilogue, which this engine does not support"
        if capabilities.epilogue_gate_d_shapes is not None and (facts.d_qk, facts.d_v) not in capabilities.epilogue_gate_d_shapes:
            # EXACT dims: a d=200 graph rides the (256, 256) envelope for plain
            # attention, but the gate tile would multiply the zero-padded
            # columns too -- the padded flavor is not claimed (plan §7 Q6).
            return (
                f"the fused epilogue gate is wired only at head dims {sorted(capabilities.epilogue_gate_d_shapes)}; graph has D_QK={facts.d_qk}/D_V={facts.d_v}"
            )
        if facts.thd:
            # The GRAPH lowering binds no gate operand on its THD launch (prepared.PreparedThdLaunch carries
            # no gate uid), so the row keeps declining the graph's THD x gate tail.  The STANDALONE adapter is
            # wider here by design: it serves THD x gate on the f16/bf16 d256 prefill (api_dsl.check_support,
            # the gated attention block's packed `fuse_gate`); a row must never admit what the adapter
            # declines (rule 8b), the reverse is a conservative row.
            return "the fused epilogue gate is dense-only (no THD gate descriptor)"
        # The paged PREFILL kernel has no gate; the d256 decode tile's SPLIT applies the
        # gate in its combine over any cache (knobs None = the facts-only question: some
        # plan -- a split -- rides the tile; a knob set answers for its own split).  A
        # SINK never splits (the shared no-split rule above and in the heuristics), so on a
        # sink graph the facts-only question is the UNSPLIT one: a paged gated sink graph
        # has NO plan on this row and is declined here, never admitted at the facts level
        # with an empty proposal list.
        _gate_split = _decode_tile_split_kv(knobs)
        if _gate_split is None and facts.has_sink:
            _gate_split = 1
        if facts.has_paged_kv and not d256_decode_tile_served(capabilities, facts, knobs, _gate_split):
            return "the fused epilogue gate is not wired on the paged-KV flavor (only the d256 decode tile's split applies the gate in its combine; a sink never splits)"
        if not facts.epilogue_gate_shape_ok:
            return "the gate G must have exactly O's shape (B, H_q, S_q, D_v); a broadcast G is not fused"
        dom = capabilities.epilogue_gate_dtypes if capabilities.epilogue_gate_dtypes is not None else frozenset({facts.dtype})
        if facts.epilogue_gate_dtype not in dom:
            return f"gate dtype {facts.epilogue_gate_dtype} not in {sorted(str(d) for d in dom)}"
        if not facts.epilogue_gate_layout_ok:
            # G is TMA-loaded zero-copy -- there is no normalisation copy for it
            # (Q/K/V/O have one).  The standalone adapter's check_support raises
            # on the same G; declining here keeps the row honest (rule 8b).
            return (
                "the gate G must be BSHD-physical (D innermost-contiguous, then H, then S, then B) with 16-byte-aligned, "
                "non-overlapping seq/head strides -- the fused kernel TMA-loads G zero-copy"
            )
        # The kernel never materialises O_v: it gates the fp32 pre-cast
        # accumulator and casts ONCE to O's dtype, so an O_v of AT LEAST O's
        # precision describes that math (the intermediate rounding it implies
        # is finer than or equal to the final one).  Half rows: FLOAT or Q's
        # dtype (== O's).  Quantized rows: FLOAT, a half dtype, or O's own
        # dtype -- never Q's FP8 dtype when O is half, which would spell
        # "quantize to fp8, THEN gate", a rounding the kernel does not do.
        if capabilities.is_fp8 or capabilities.is_mxfp8:
            _o_v_dom = (None, cudnn.data_type.FLOAT, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, facts.dtype_o)
            _o_v_msg = (
                "the sdpa node's virtual O must be FLOAT, a half dtype, or O's dtype (the fused kernel gates the fp32 pre-cast accumulator and quantizes once)"
            )
        else:
            _o_v_dom = (None, cudnn.data_type.FLOAT, facts.dtype)
            _o_v_msg = "the sdpa node's virtual O must be FLOAT or Q's dtype (the fused kernel gates the fp32 pre-cast accumulator)"
        if facts.sdpa_o_virtual_dtype not in _o_v_dom:
            return _o_v_msg
        if not facts.sdpa_o_virtual_declared:
            # Only user-assigned dim/stride are pushed to the C++ lowering, whose
            # pre-validation needs rank-4 dim + stride on the sdpa node's O.
            return "declare dim AND stride on the sdpa node's virtual O (set_dim/set_stride) -- the classic frontend requires it and FROST binds the mul output as O"

    if facts.has_paged_kv:
        # The paged PREFILL pipeline on cc 10.7 is qualified for THD queries (with or without an
        # attention sink -- the per-row sink fold); the d256 DECODE tile (sm107/decode_d256_f16.py)
        # walks the block table itself and folds the sink per Q row, so a decode-shaped half d256
        # graph is served paged whatever its THD-ness or sink.  Dense (non-THD) paged queries are
        # otherwise not wired on cc 10.7; the MXFP8 row answers through its own flags above.
        # api_dsl.check_support mirrors this (rule 8b).
        if capabilities.sm_lo == 107 and not capabilities.is_mxfp8 and not facts.thd and not d256_decode_tile_served(capabilities, facts, knobs):
            # knobs None = the facts-only question: SOME packing rides the tile (the packed form alone past 32 unpacked rows)
            return "Rubin paged KV requires THD queries or a decode-shaped half D256 graph (the decode tile); dense paged queries are not wired on cc 10.7 otherwise"
        # Served by the PAGED_KV specialization of the f16/bf16 kernels on the
        # flavors in paged_d_shapes and of the d128 per-tensor FP8 kernel (the
        # fp8 row's paged_d_shapes; config_sm100._validate_params mirrors these
        # as its backstop and each unwired kernel file backstops with a
        # module-scope guard on paged_kv). The attention sink composes with it
        # on the f16/bf16 kernels: the sink is a per-row epilogue fold and
        # PAGED_KV only changes the K/V TMA-LDG warp (validated together in
        # test_sdpa_fwd_paged_sm100, S_q 1..4, PackGQA on/off, HND/NHD, with a
        # left window, on the d128, d192x128 and d256 flavors). The per-tensor
        # FP8 d64/d128 epilogue also folds the sink once over paged KV, including
        # keyless rows and padded Q. FP8 sink + split-KV stays declined above:
        # those partials still include the sink. The block-scaled O epilogue
        # (sf_o) over pools remains unvalidated and declined.
        # On cc 10.7 the same composition rides the shared d128 / d256 bodies compiled for sm_107a (the Rubin
        # paged arm of api_dsl._load_sm100_kernel_module); validated there with packed THD queries (1 / 4 / 8
        # tokens per request), PackGQA on / off, cga1 / cga2, HND / NHD pools and keyless rows
        # (test_mhas_v2.py's "P2" block).  Dense (non-THD) paged queries stay declined on cc 10.7.
        # The cc 10.7 MXFP8 row serves dense queries over F8_128x4 pools on d128 / d256 through its own PAGED_KV
        # loader (sm107/prefill_d{128,256}_mxfp8.py); its THD leg and page_size % 128 are governed by the
        # is_mxfp8 clauses below.
        if facts.is_mxfp8:
            if facts.page_size % 128 != 0:
                # A page must hold whole 128-row F8_128x4 SF atoms.
                return f"paged MXFP8 KV needs page_size to be a multiple of 128; got {facts.page_size}"
            if facts.thd:
                return "paged MXFP8 KV with THD queries is not wired"
        if facts.is_fp8 and facts.thd and not (facts.device_cc in ((10, 0), (10, 3)) and (facts.d_qk, facts.d_v) in SM100_PAGED_FP8_THD_SHAPES):
            return "paged FP8 THD queries require native per-tensor D128/D256 on SM100/SM103"
        if (facts.is_fp8 or facts.is_mxfp8) and facts.o_block_scale:
            return "paged KV with a block-scaled O (sf_o) is served on dense K/V only (the block-scaled epilogue over pools is not validated)"
        if not facts.padded:
            return "paged KV requires use_padding_mask with seq_len_kv (the per-batch KV length bounds the block-table walk)"
        if capabilities.paged_d_shapes is not None:
            # The flavor the lowering picks (smallest covering envelope), not
            # the raw dims: (256, 128) and (64, 192) ride the wired d256
            # envelope, (192, 128) is the native d192x128 flavor, and a
            # d512-envelope selection is declined until that kernel wires it.
            # d64 is likewise excluded simply by not appearing in the set.
            selected = _selected_d_shape(capabilities, facts)
            if selected not in capabilities.paged_d_shapes:
                wired = ", ".join(f"d{sq}" if sq == sv else f"d{sq}x{sv}" for sq, sv in sorted(capabilities.paged_d_shapes))
                return f"paged KV is wired on the {wired} kernel flavors only; head dims ({facts.d_qk}, {facts.d_v}) select {selected}"
        p = facts.page_size
        if p % 8 != 0 or (p < 128 and 128 % p != 0) or (p > 128 and p % 128 != 0):
            return f"page_size {p} must be a multiple of 8 that divides the 128-row KV tile or is a multiple of it"

    if facts.has_sink and capabilities.sink_dtypes is not None and facts.dtype not in capabilities.sink_dtypes:
        return f"sink token with dtype {facts.dtype} not in {sorted(str(d) for d in capabilities.sink_dtypes)}"

    if facts.has_bias and capabilities.bias and facts.bias_t is not None:
        # uniform_dtype covers K/V/O only; the serving adapters compile the
        # bias load as fp32 or the io dtype, so anything else must decline
        # HERE, not ValueError at execute.
        bias_dt = facts.bias_t.get_data_type()
        if bias_dt not in (cudnn.data_type.FLOAT, facts.dtype):
            return f"bias dtype {bias_dt} must be fp32 or match the Q/K/V dtype ({facts.dtype})"
        if capabilities.bias_io_dtype_only and bias_dt != facts.dtype:
            return f"bias dtype {bias_dt} must match the Q/K/V dtype ({facts.dtype}) on this engine (the resident block loads the io dtype; cast an fp32 bias first)"
    if (facts.has_bias and capabilities.bias and capabilities.bias_d_shapes is not None) or (facts.has_kv_mask and capabilities.kv_mask):
        reason = _bias_arm_decline(capabilities, facts, knobs)
        if reason is not None:
            return reason

    if facts.right_band_widening and facts.right_bound is not None and facts.right_bound < 0:
        return f"negative diagonal_band_right_bound ({facts.right_bound}) is not supported"

    if facts.has_cu_seq_len:
        # cu_seq_len_* ((B+1,) prefix sums, cuDNN 9.24+). The THD lowering
        # consumes either length form host-side; the dense kernels' CU read
        # mode (len = cu[b+1] - cu[b]) is not plumbed yet, so dense cu graphs
        # stay declined even on serving rows.
        if not capabilities.cu_seq_len:
            return "graph uses cu_seq_len_q / cu_seq_len_kv, which this engine does not support"
        if not facts.thd:
            return "cu_seq_len_* on dense graphs is not supported yet (kernel CU read mode not plumbed)"
        if (facts.seq_q_t is not None and facts.cu_seq_q_t is not None) or (facts.seq_kv_t is not None and facts.cu_seq_kv_t is not None):
            return "seq_len_* and cu_seq_len_* on the same side is ambiguous (backend precedence is not replicated here)"

    if facts.amax_s_t is not None:
        # The FROST FP8 kernels no longer compute Amax_S (dropped: nothing
        # consumed it and the atomicMax serialized the epilogue); a graph that
        # DECLARES the output must go to an engine that writes it.
        _warn_amax_s_declared_once()
        return "graph requests the Amax_S output, which the FROST engines do not produce"

    if facts.bottom_right:
        if not (facts.causal or facts.right_band_widening):
            return "bottom-right alignment requires a causal upper bound (plain or right-widened)"
        if not capabilities.bottom_right:
            return "graph uses bottom-right causal, which this kernel does not support"
    if facts.padded and facts.wants_stats and not facts.thd and not capabilities.padded_stats:
        return "padding mask with generate_stats is not supported by this kernel"

    if (
        facts.thd
        and facts.wants_stats
        and facts.stats_t is not None
        and getattr(facts.stats_t, "ragged_offset", None) is None
        and not capabilities.thd_padded_stats
    ):
        # A Stats tensor with no ragged offsets is the per-batch padded form,
        # rows at b * s_max; a kernel without the per-batch THD store writes
        # packed (T, h) rows and has no per-sequence stats base to place them at.
        return "THD Stats without ragged offsets is addressed per batch ([b, h, s_max, 1]); this kernel writes packed (T, h) rows -- bind ragged stats offsets"
    if facts.stats_t is not None and not facts.thd:
        if facts.stats_t.get_data_type() != cudnn.data_type.FLOAT:
            return f"stats must be fp32; got {facts.stats_t.get_data_type()}"
        stats_dim = tuple(facts.stats_t.get_dim())
        stats_stride = tuple(facts.stats_t.get_stride())
        expected_dim = (facts.b, facts.h_q, facts.s_q, 1)
        if stats_dim != expected_dim:
            return f"stats must be (B, H_q, S_q, 1) = {expected_dim}; got {stats_dim}"
        if not ga.dense_layout_ok(stats_dim, stats_stride):
            return (
                f"stats must use a dense-compatible B/H/S permutation or padded layout with non-broadcast, non-overlapping-by-span strides; got {stats_stride}"
            )

    if capabilities.single_wave_only:
        # See Capabilities.single_wave_only. 512 = TILES_Q * TILE_M * CTA_MMA
        # rows per cluster; resident clusters = SM count / CTA_MMA (one CTA per
        # SM at this kernel's SMEM footprint). Unknown SM count -> stay gated.
        clusters = facts.b * facts.h_q * ((facts.s_q + 511) // 512)
        resident = (facts.device_sm_count or 0) // 2
        if clusters > resident:
            return (
                f"launch needs {clusters} Q-tile clusters but only {resident} fit in one wave; "
                "the fp8 kernel's persistent multi-wave rescheduling is numerically invalid (gated)"
            )

    if capabilities.skv_tile and facts.s_kv % capabilities.skv_tile != 0 and not capabilities.skv_tail_via_padding:
        if not (facts.padded or _band_covers_kv_tail(facts)):
            return f"S_kv ({facts.s_kv}) must be a multiple of {capabilities.skv_tile} unless a padding mask is given or the causal mask covers the KV tail"
    if facts.shape_overrides or _selected_d_shape(capabilities, facts, knobs) == SM100_NATIVE_D576_SHAPE:
        return _prepared_decline_reason(capabilities, facts, knobs.split_kv if knobs is not None else None)
    return None


@dataclass(frozen=True)
class EngineSpec:
    name: str
    capabilities: Capabilities
    # Lowering strategy: facts -> executor. A future engine may select between
    # kernels (e.g. decode vs prefill by S_q) or chain several launches under
    # one name.
    lower: "Callable[[EngineSpec, ga.SdpaGraphFacts, Optional[SdpaFwdKnobs]], Any]"


def _sm100_spec() -> EngineSpec:
    """f16/bf16 SM100-family engine: ONE row; the adapter picks the smallest
    kernel flavor (d128 / d192xd128 / d256 / d512) covering the graph's head
    dims (api_dsl._pick_flavor), and every flavor serves its envelope via TMA
    zero-padding. The d128 flavor has two tiles behind one knob: TILE_CGA_M=2
    is the prefill pipeline (512 Q rows per cluster), TILE_CGA_M=1 the decode
    tile (128 rows per CTA, sm100/decode_d128_f16.py) -- the "decode vs prefill
    by S_q" kernel choice EngineSpec.lower anticipates, driven by the
    heuristics' cga rule. sm_hi=106: no f16 lowering exists on the Rubin line —
    when one lands it gets its own row (the per-arch-line row doctrine)."""
    return EngineSpec(
        name="sdpa_fwd_prefill_sm100",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=106,
            phase="prefill",
            # (64, 64) is a NATIVE flavor, not an envelope: it compiles the d128
            # file at TILE_K = TILE_O = 64 (TemplateParams.d_flavor) instead of
            # zero-filling a 128-wide tile for gpt-oss-class head dims.  (32, 32)
            # is the same file's 32-wide small tile (d_flavor = 32; config_sm100.make_cfg_d32), the
            # SM100 twin of the Rubin small tile: it carries the row's DENSE leg only
            # (dense_only_d_shapes below) -- THD / paged / split-KV / PackGQA / the score
            # side outputs walk past it onto the native d64 flavor, exactly the route
            # a d <= 32 graph took before the tile existed.
            d_shapes=frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), SM100_NATIVE_D576_SHAPE}) | SM100_DENSE_ONLY_D_SHAPES,
            d_envelope_floors=((SM100_NATIVE_D576_SHAPE, 512),),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            # Paged KV caches (paged_attention_k/v_table + seq_len_kv) on the
            # d128 / d192x128 / d256 / d512 flavors: block-table indirection on
            # the K/V TMA loads, HND and NHD page layouts, K and V pools of
            # different row widths (d192x128, and the d512 envelope's
            # straddling pairs), KV split + combine (mismatch() holds the
            # padded / page-geometry conditions; paged_d_shapes below names the
            # wired flavors -- on d512 the role-split loader issues the K boxes
            # from the sub-group 0 CTAs and the V boxes from the sub-group 1
            # CTAs). Decode shapes on the d128 flavor ride the decode tile
            # (TILE_CGA_M=1, below); d192x128 and d512 have no decode tile
            # yet, so their paged decode runs the prefill geometry (one live
            # row per 128-row Q tile), measured behind the backend's paged
            # decode plan (B200, page 16, bf16: d192x128 b=32 S_q=1 32/32 MHA
            # 788.7 us vs 476.9 us, 32/8 GQA 275.8 vs 199.6 us; d512 b=8 S_q=1
            # 64/1 77.8 vs 65.1 us -- the tracker's gaps table;
            # sdpa/fwd/placement.py keeps the backend first for paged d512
            # S_q == 1 by default). Decode tiles for both are the follow-up, as
            # the d128 tile was: parity is a kernel's job, not an ordering rule's.
            paged_kv=True,
            paged_d_shapes=frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512), SM100_NATIVE_D576_SHAPE}),
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            thd=True,
            thd_padded_stats=True,
            cu_seq_len=True,
            padded_stats=True,
            # Ragged S_kv with an uncovered tail: the adapter compiles the
            # padded mask against the scalar S_kv (kv_tail_mask, #1425).
            skv_tail_via_padding=True,
            # The f16/bf16 lowering serves any dense B/H/S stride permutation
            # (padded strides included) with the head dim innermost; the
            # FP8/MXFP8 rows stay on the strict BSHD gate until their padded /
            # scale-factor paths are validated against relaxed layouts.
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # (128, 128): TILE_CGA_M=1 IS the d128 DECODE tile
            # (sm100/decode_d128_f16.py -- TILES_Q=1, one softmax warpgroup,
            # three KV stages; config_sm100.CfgD128Decode), which the lowering
            # selects for that knob value on dense graphs (THD keeps cga2, see
            # mismatch).  The heuristics propose it when one 128-row tile
            # covers a packed head's Q rows (S_q * pack_g <= 128, pack_g the
            # candidate's own packing: the packed group Cfg.PACK_G -- G, or its
            # largest divisor of 128 under partial PackGQA -- 1 unpacked):
            # decode and MTP.
            # A split rides either width (no split_cgas entry).
            # (64, 64): the native d64 prefill flavor builds at both widths.
            # (32, 32): the 32-wide small tile at its validated width (config_sm100.SM100_SMALL_D_CGAS: cga1).
            cgas_by_d_shape=(
                ((128, 128), frozenset({1, 2})),
                ((192, 128), frozenset({1, 2})),
                ((64, 64), frozenset({1, 2})),
                ((32, 32), SM100_SMALL_D_CGAS[(32, 32)]),
            ),
            split_cgas_by_d_shape=(((192, 128), frozenset({2})),),
            # The 32-wide small tile carries the dense leg only: every other leg (and a cga outside its
            # domain) is routed past it to the native d64 flavor -- see the field's doc.  ONE constant
            # with the adapter's pool (config_sm100, rule 8b').
            dense_only_d_shapes=SM100_DENSE_ONLY_D_SHAPES,
            # The additive bias / per-key kv_mask arms of the small tiles on this arch line (the port's first body seam): the RESIDENT
            # block of the 32-wide tile at cga1 (S_kv <= 256) and of the 64-wide tile at cga2 (S_kv <= 128), the kv_mask in every form at
            # any S_kv, TILE_N 128, the one-batch-entry item.  The streamed ring, the pair item and the 64-wide KV tile are typed declines
            # until they land (_bias_arm_decline's SM100 branch; config_sm100.SM100_BIAS_D_SHAPES / sm100_bias_resident_s_kv_max, rule 8b').
            bias=True,
            bias_d_shapes=SM100_BIAS_D_SHAPES,
            bias_io_dtype_only=True,
            kv_mask=True,
            kv_mask_dtypes=frozenset({cudnn.data_type.BOOLEAN, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT}),
            # The small tiles render the 64-wide KV tile too (the TILE_N knob; config_sm100.SM100_TILE_NS_BY_D_SHAPE): the SM100 default stays
            # 128 (config_sm100.sm100_tile_n_pick) until the GB200 sweep measures the width rule; the fp32 vector WITH a bias lands on 64.
            tile_ns_by_d_shape=SM100_TILE_NS_BY_D_SHAPE,
            # Every f16 flavor kernel wires SplitHelpers, and the adapter carves
            # the partial slabs + launches sm100/split_combine when split_kv > 1
            # (dense f16 only; see mismatch's facts x knobs gate). d64 included:
            # it compiles the same kernel body, and at a NATIVE d_v = TILE_O the
            # fp32 partial store has no surplus columns to clip.
            split_kv_supported=True,
            pack_gqas=frozenset({False, True}),
            # The d128 / d256 f16 kernels pack a GQA group that does not divide
            # the 128-row tile by its largest divisor that does (Cfg.PACK_G:
            # 96/8 -> 4 heads per token row-group); d192x128 / d512 pack the
            # whole group only.
            pack_gqa_partial_d_shapes=frozenset({(128, 128), (256, 256), SM100_NATIVE_D576_SHAPE}),
            thd_pack_gqa_d_shapes=SM100_THD_PACK_GQA_SHAPES,
            # Score side outputs (score_max / score_sum_exp): the d128 f16/bf16 prefill kernel stores
            # them from its correction epilogue (dense / unsplit / unpaged, the cga2 tile); the
            # other flavors decline until their files carry the store (the set is the claim).
            score_max=True,
            score_sum_exp=True,
            score_stats_d_shapes=SM100_SCORE_STATS_SHAPES,
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm107_spec() -> EngineSpec:
    """f16/bf16 Rubin (SM107) engine — the per-arch-line sibling of
    ``sdpa_fwd_prefill_sm100``, which stops at cc 10.6.

    The lowerings diverge enough to need their own row rather than a widened
    SM100 one: the Rubin kernels build **version-1 tcgen05 SMEM descriptors**
    (a version-0 descriptor addresses only 256 KiB, and Rubin's 327 KiB budget
    puts the d512 flavor's P ring at exactly that boundary).  The f16 line
    carries all four SM100 flavors, d192xd128 included
    (``sm107/prefill_d192_d128_f16.py`` — the d128 body with ``make_cfg_d192``);
    the QUANTIZED Rubin rows are the ones that ship a strict subset.

    Every capability below was MEASURED on a Rubin dev node against an
    explicit fp32 reference, not inherited from the SM100 row:

    - causal / bottom-right / right-band / SWA: validated on all three flavors,
      cos = 1.000000. ``window_right`` is load-bearing rather than decorative —
      ``bounds_for_tile`` already trims KV *tiles* by it, so a right-banded
      graph is silently wrong unless the element-level mask gets it too.
    - padded: per-batch ``seq_len_kv``, deliberately ragged and NOT tile
      aligned, so a tail tile is partly masked.
    - sink, and GQA 4:1.

    Deliberately NOT claimed, each because the kernels lack the machinery
    rather than because it went untested:

    - ``thd``: SERVED on EVERY f16 flavor as of 2026-09-09.  All four bodies were
      moved onto the FROST THD contract -- the 14-arg setup helper, the 4B+4
      metadata the SHARED decode already read (the two had disagreed, which is
      why ``compile()`` used to raise), the persistent claim-counter scheduler,
      the dead-unit O-store guard, and the packed-total-clamped runtime K/V
      descriptors that keep a NaN capacity tail out of BMM2.
    - ``split_kv_supported``: dense d128 and d192x128 use FP32 partials and
      the shared combine. Bounded D128 THD and nonpaged D192 THD
      also use the shared single-CTA packed partials. Half dense sink splits fold the
      sink once in the combine; nonpaged THD sink splits stay declined.
      On d256 the split is the decode tile's (dense or paged, packed or not:
      fp32 partials into the split-major workspace the shared combine reduces)
      and the paged THD packed split; the d256 prefill kernel has no dense split.
    - ``pack_gqas``: D128 dense CGA2 and unsplit nonpaged THD use native SM107 prefill. Dense CGA1 uses the
      shared decode tile; D128 paged/nonpaged split THD and D256 paged unsplit THD use the shared half pipeline.
      Existing packing admission still excludes the pre-folded scale;
      d256 also packs on the decode tile (``pack_gqa_d_shapes`` carries (256, 256) for both routes --
      on the tile the WHOLE group, dense or paged, whatever tile_m; the dense d256 prefill kernel
      runs unpacked, so ``mismatch`` declines a packed dense d256 graph the tile does not serve).
    - ``paged_kv``: D128/D256 half THD -- with or without an attention sink (the
      per-row epilogue fold; a keyless row stores O := 0 / LSE := sink) -- uses the
      shared Blackwell paged pipeline, compiled natively for SM107. The D128 paged THD
      Q>1 sink split uses its sink-aware packed combine; dense D256 decode sink
      splits use the shared dense combine. Other THD sink splits stay declined.
      Dense (non-THD) paged queries stay declined EXCEPT a DECODE-shaped half d256 graph
      (dense Q, ``S_q x G`` packed rows within the tile's routed envelope -- 16 rows, or the
      32-column tile in up to two token units for a packed MTP step -- sink or not), which
      rides the d256 decode tile ``sm107/decode_d256_f16.py`` instead
      (``d256_decode_tile_selected``), which walks the block table itself.
    - ``decode``: stated, not inherited -- ``S_q == 1`` is served on every
      flavor, and on d256 it is the decode tile above (the swap-AB body ported
      from ``sm100/decode_d256_f16.py``), with the SM100 tile's whole-group
      PackGQA (24/2: 12 live rows + 4 zero tail rows per unit) and dense / paged
      split-KV -- validated on cc 10.7 against the fp32 reference, paged == dense
      bitwise, packed == unpacked bitwise, split == unsplit within a derived
      budget (one output ulp plus twice the half-precision P quantization term,
      both paths quantizing P independently; test_sdpa_fwd_decode_d256_sm107.py).
      A GATED decode-shaped graph (the ``mul(O_v, sigmoid(G))`` tail) rides the
      tile when its plan SPLITS: the tile has no gate seams, so the split combine
      applies the gate to the fp32 merged value (``sm100/split_combine`` gate=True,
      the fused kernels' arithmetic, one rounding) -- dense or paged, packed or not;
      unsplit, the d256 prefill kernel's fused epilogue serves it as before
      (``d256_decode_tile_selected(..., split_kv)``).
    - ``softmax_precisions``: FLOAT only -- the half kernels run the f32 exponent
      (the f16x2 arm is a quantized-kernel specialization).
    - ``attn_scale_prefolded_d_shapes``: every half prefill body carries the
      pre-folded-scale arm (raw running max, plain subtract shift); the paged
      bodies, the single-CTA THD legs and the shared dense D128 legs (decode
      tile, dense PackGQA) apply the scale in-kernel and decline.

    Small head dims (``config_sm107.SM107_F16_SMALL_D_SHAPES``): the d128 file
    at ``TILE_K = TILE_O = 32`` and ``64`` (``make_cfg_d32`` / ``make_cfg_d64``)
    are native flavors of this row, so a head dim D <= 64 no longer rides the
    zero-filled 128-wide tile.  The width a graph lands on is the smallest one
    covering max(d_qk, d_v) -- 8 / 16 / 24 -> 32, 40 / 48 / 56 -> 64, 72..120 ->
    128, the natives on themselves (``sm107_f16_tile_width``; the same walk as
    ``_selected_d_shape`` below and ``api_dsl._pick_flavor``) -- and D % 8 != 0
    (the TMA 16-byte global-stride rule at 2 B/elem) is a typed decline, never
    a padded copy.  The small tiles serve the dense feature set (causal /
    bottom-right / right band / SWA / padded / sink / GQA / Stats) at the
    validated widths of ``SM107_SMALL_D_CGAS`` -- and ONLY that
    (``dense_only_d_shapes``): a graph that asks for THD, split-KV, paged KV,
    PackGQA, the score side outputs or a cga outside the tile's domain is
    selected PAST the small tiles onto the d128 flavor it rode before they
    existed (zero-filled to 128, where ``thd_d_shapes`` / ``split_d_shapes`` /
    ``paged_d_shapes`` / ``pack_gqa_d_shapes`` / ``score_stats_d_shapes`` and
    the cga2 domain apply exactly as before), so the tiles withdraw nothing the
    row served: split-KV and the score outputs at D <= 64 stay served on d128,
    a cga2 request at D <= 32 lands on the 64-wide tile, and THD / paged /
    PackGQA at D <= 64 keep their pre-tile typed declines (exact-shape legs).
    """
    return EngineSpec(
        name="sdpa_fwd_prefill_sm107",
        capabilities=Capabilities(
            sm_lo=107,
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            # Every f16 flavor has a Rubin module -- api_dsl._SM107_KERNEL_FILES
            # is the other half of this claim, and _pick_flavor lands a d=192
            # graph on the NATIVE kernel rather than the d256 envelope.  The two
            # small shapes are the d128 file's 32- and 64-wide tiles.
            d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}) | SM107_F16_SMALL_D_SHAPES,
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            # The LSE store is const_expr'd out on a None lse_tensor, and
            # compile(has_lse=False) binds no dummy buffer at any level -- so a
            # stats-less graph reports get_workspace_size() == 0.
            lse_optional=True,
            # Ragged S_kv with an uncovered tail: the adapter compiles the
            # padded mask against the scalar S_kv (kv_tail_mask, #1425).  It is
            # REQUIRED, not an optimization: at MASK_FLAGS == 0 the kernel's
            # kv_right is a floor division and would drop the tail tile.
            skv_tail_via_padding=True,
            # SM107_F16_THD_SHAPES is the single definition, shared with the
            # standalone adapter's gate so the two cannot drift (rule 8b').  It
            # now covers every f16 flavor; keeping it a named constant rather
            # than `True` is what makes a future partial arch line expressible.
            thd=True,
            thd_padded_stats=True,
            padded_stats=True,
            thd_d_shapes=SM107_F16_THD_SHAPES,
            cu_seq_len=True,
            paged_kv=True,
            paged_d_shapes=frozenset({(128, 128), (256, 256)}),
            decode=True,  # stated, not inherited: S_q == 1 is served; on d256 by the decode tile (see the docstring)
            # FLOAT only: the half kernels run the f32 exponent (a HALF request declines here, never
            # in the adapter); the pre-folded scale is a neutral arm of every half prefill body.
            softmax_precisions=frozenset({cudnn.data_type.FLOAT}),
            attn_scale_prefolded_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            pack_gqas=frozenset({False, True}),
            # (256, 256) = the d256 DECODE tile's whole-group packing (dense or paged, whatever tile_m) and the
            # paged THD d256 prefill's PackGQA (CGA2, unsplit); mismatch declines a packed dense d256 graph the
            # tile does not serve (the dense d256 prefill kernel runs unpacked).
            pack_gqa_d_shapes=frozenset({(128, 128), (256, 256), (512, 512)}),
            # Native D512 2x2 tiles pack gcd(G,64), including two subgroups
            # per KV head at G128. THD and paged D512 retain their declines.
            pack_gqa_partial_d_shapes=frozenset({(512, 512)}),
            thd_pack_gqa_d_shapes=frozenset({(128, 128), (256, 256)}),
            split_kv_supported=True,
            # (256, 256) = the paged THD packed split and the decode tile's dense / paged split.
            split_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            # NATURAL row-wide; LPT advertised PER D-SHAPE for what is validated.
            #
            # The old note here said the ported decode "does not honor
            # SCHED_LPT". The cause was narrower than that: every SM100 kernel
            # calls `make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)` and
            # the SM107 port omitted the argument on 9 of 11 flavors. Without
            # it the LPT linearization walks a row range CTA_MMA times too
            # large, no tile is ever claimed, and the kernel writes NOTHING --
            # cosine 0.0000, which reads like a dead kernel rather than a
            # scheduling bug. The two flavors that kept it (d128 FP8, d192 FP8)
            # are exactly the ones previously described as working.
            #
            # Restored on every 2-CTA SM107 flavor. It is a NO-OP under
            # SCHED_NATURAL -- that decode branch never reads q_tiles -- so the
            # shipped path is unchanged.
            #
            # Validated on d256 f16: cos 1.0000 at n_kv 2/3/4/8, dense AND
            # causal. Measured causal SOL on Rubin at S=4096/8192/32768:
            # 53.0/71.1/76.7% under NATURAL -> 62.6/79.3/77.5% under LPT
            # (+18.2/+11.6/+1.1%), recovering 40%/51%/29% of the causal-vs-dense
            # gap; dense itself is neutral. The decay with S is the signature of
            # scheduler imbalance, which is what LPT exists to fix.
            #
            # D128 is also qualified through dense and live-length THD
            # capture/replay, including the shared paged PackGQA pipeline.
            # D192 and D512 remain unqualified; D512 is cga4x1 role-split
            # with a different scheduler shape.
            # SCHED_LPT_L2 is claimed by NO f16 flavor -- its decode
            # needs `qh_per_kh` and `seqlen_kv` at every call site, which the
            # f16 kernels do not pass (the d128 / d192x128 FP8 and MXFP8
            # kernels do; see those rows), so it raises rather than
            # miscomputes. Both are follow-ups.
            sched_policies=frozenset({SCHED_NATURAL}),
            sched_policies_by_d_shape=(((128, 128), frozenset({SCHED_NATURAL, SCHED_LPT})), ((256, 256), frozenset({SCHED_NATURAL, SCHED_LPT}))),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # (128, 128) half carries two tiles behind TILE_CGA_M on cc 10.7 as on the SM100 row: cga2 = the Rubin prefill
            # pipeline (sm107/prefill_d128_f16.py; a packed set runs the shared SM100 prefill body), cga1 = the shared 128-row
            # decode tile (sm100/decode_d128_f16.py compiled for cc 10.7, issue #1472).  Dense graphs only: effective_cgas
            # keeps THD, dense paged and the pre-folded scale on cga2.  A split rides either width (no split_cgas entry).
            # The small tiles run at the widths their bodies are validated at (one constant with the config factory, which
            # refuses any other): cga1 for both -- the narrow slabs need no collective MMA to halve K/V, and a 512-row
            # cga2 cluster wastes 25-50 % of its Q rows at the S = 256-384 crops these tiles exist for; d64 also admits
            # cga2 (its per-CTA V operand is the 64-byte atom the SM100 d64 cga2 build runs).
            cgas_by_d_shape=(((128, 128), frozenset({1, 2})),) + tuple((shape, SM107_SMALL_D_CGAS[shape]) for shape in sorted(SM107_SMALL_D_CGAS)),
            # Fused epilogue gate (O := O * sigmoid(G)) on the d256 kernel,
            # f16 AND bf16 (G in Q's dtype).  EXACT (256, 256) only -- the
            # gate tile does not ride the head-dim envelope.  The standalone
            # adapter's twin reads the same constant (rule 8b').  On the d256
            # decode tile's split the gate is applied by the combine instead
            # (d256_decode_tile_selected's split_kv arm; the same claim).
            epilogue_gate=True,
            epilogue_gate_d_shapes=SM107_EPILOGUE_GATE_SHAPES,
            # Score side outputs (score_max / score_sum_exp): the Rubin d128 f16/bf16 prefill kernel
            # stores them from its correction epilogue (dense / unsplit / unpaged); the other flavors
            # decline until their files carry the store (the set is the claim, rule 8b').  A d <= 64
            # graph with the outputs selects the d128 flavor (dense_only_d_shapes below), as before
            # the small tiles.
            score_max=True,
            score_sum_exp=True,
            score_stats_d_shapes=SM107_SCORE_STATS_SHAPES,
            # The small tiles carry the dense leg only: every other leg (and a cga outside the tile's
            # domain) is routed past them to the d128 flavor -- see the field's doc and the row
            # docstring.  ONE constant with the adapter's pool (config_sm107, rule 8b').
            dense_only_d_shapes=SM107_F16_SMALL_D_SHAPES,
            # Additive bias + per-key mask on the small tiles as a RESIDENT SMEM row-block
            # (the d128 file at BIAS_MODE = 1 / KV_MASK_FORM = 1 / 2 / 3): bias [1|B, 1|H_q, S_q, S_kv] in
            # the io dtype, kv_mask [1|B, 1, 1, S_kv] BOOLEAN (the bit word) or an ADDITIVE vector in
            # the io dtype / fp32 (folded pre-scale like the bias; S_kv % 8 == 0, 16-byte strides; the fp32
            # vector WITH a bias at TILE_N 64 only -- the heuristics' pick for that form, a requested 128 a
            # typed decline: its rendering spills, config_sm107.SM107_FP32_VECTOR_WITH_BIAS_TILE_N);
            # dense / unsplit / unpaged / no score outputs; a BIAS as the RESIDENT block up to what the
            # carveout holds beside the operands (3 tiles = 384 at the 32-wide tile, 1 / 2 tiles = 128 / 256 at
            # the 64-wide one at cga 1 / 2; the (n, n+1) pair item's 128-row block 768 / 384 --
            # config_sm107.sm107_bias_resident_s_kv_max at the default TILE_N 128, the budget every plan is
            # asked at) and above it as the STREAMED ring (BIAS_MODE = 2: one bias tile per kv tile through a
            # 2- / 3-stage ring, any S_kv -- config_sm107.sm107_bias_mode_pick; the one-batch-entry item at the
            # 64-wide tile's cga1 has no ring and takes cga2); the shapes = SM107_BIAS_D_SHAPES, the d128 tile has
            # no spare for either form; the kv_mask alone at any S_kv.  The adapter's check_support is the twin
            # (rule 8b').
            bias=True,
            bias_d_shapes=SM107_BIAS_D_SHAPES,
            bias_io_dtype_only=True,
            kv_mask=True,
            kv_mask_dtypes=frozenset({cudnn.data_type.BOOLEAN, cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FLOAT}),
            # The small tiles render TILE_N 64 as well as 128 (the TILE_N knob; the heuristics' default stays
            # 128 = min(tile_ns)); the d128 rendering of the same file keeps 128.
            tile_ns_by_d_shape=SM107_TILE_NS_BY_D_SHAPE,
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_mxfp8_spec() -> EngineSpec:
    """Block-scale MXFP8 engine (E4M3/E5M2 + per-32-block E8M0 SF).

    THD/varlen on all four native shapes rides the shared packed lowering
    (write_thd_meta envelope design, issue #552; packed
    Q/K/V/O contract only). The SF tensors travel PACKED
    per-sequence-TILE-padded ([1, H, Σ_b ceil(S_b/128), SF_SMEM] tile sequences
    in cu_seqlens order — see prepared._bind_mxfp8_scales); the graph's declared
    SF dims stay the dense capacity, like the ragged Q/K/V storage.
    """

    return EngineSpec(
        name="sdpa_fwd_prefill_sm100_mxfp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL[0],
            sm_hi=106,  # no Rubin MXFP8 lowering
            phase="prefill",
            # Exact native shapes only (d_pad_multiple=0): the SF plumbing is
            # not audited for envelope zero-padding.
            # (64, 64): the native d64 leg of the d128 MXFP8 file (TemplateParams.d_flavor);
            # dense / unsplit / unpaged for now.
            d_shapes=frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)}),
            d_pad_multiple=0,
            thd_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            split_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset(
                {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
            ),
            # Block-scaled O epilogues (FP4_E2M1 + E4M3/16, FP8_E4M3 + UE8M0/32) on
            # the d128 MXFP8 kernel; the adapter declines the wider flavors.
            o_block_scales=frozenset({0, 16, 32}),
            is_mxfp8=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            # Ragged S_kv with an uncovered tail: the adapter compiles the
            # padded mask against the scalar S_kv (kv_tail_mask, #1425).  The
            # SF tile count is already ceil(S_kv / 128) and the kernels
            # replace the fully padded V scale blocks of that tail tile.
            skv_tail_via_padding=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            thd=True,
            thd_padded_stats=True,
            padded_stats=True,
            cu_seq_len=True,
            # The SF pools page with K/V. mismatch() gates page_size % 128.
            paged_kv=True,
            paged_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            cgas_by_d_shape=(
                ((64, 64), frozenset({1})),
                ((128, 128), frozenset({1, 2})),
                ((192, 128), frozenset({1, 2})),
                ((256, 256), frozenset({1})),
                ((512, 512), frozenset({1})),
            ),
            split_cgas_by_d_shape=(((192, 128), frozenset({2})),),
            # The split path also needs a half-precision O (mismatch's
            # facts x knobs gate).
            split_kv_supported=True,
            # PackGQA on the d128 flavor: the kernel gathers the packed tile's
            # per-row scale factors out of the group's F8_128x4 atoms (see
            # sm100/prefill_d128_mxfp8.py); plain O, dense only -- mismatch() declines
            # the rest.  The other flavors keep the one-atom TMA path.
            pack_gqas=frozenset({False, True}),
            pack_gqa_d_shapes=frozenset({(128, 128)}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm100_fp8_spec(*, arch: str = "sm100") -> EngineSpec:
    """Per-tensor FP8 engine with scalar descales.

    ONE row per ARCH LINE (``arch``: "sm100" = pre-Rubin Blackwell 100-106,
    "sm107" = Rubin line 107-119): the two lowerings genuinely diverge and a
    shared row could only describe their union with knob x arch notches.
    Within a row the head dim is a LOWERING concern — the adapter picks the
    kernel flavor covering the graph. Each row declares
    exactly what its own kernels carry:

    - d_shapes: both rows pick among d128, d192xd128, d256, and d512 (the
      SM107 line carries all four per-tensor FP8 siblings; the SM100 row adds
      the native d64 leg), so a graph outside a row's set is ineligible at
      probe time instead of failing during lowering.
    - The ENVELOPE (d_pad_multiple=16, the TMA 16-byte global-stride rule at
      1 byte/elem): smaller head dims ride TMA zero-padding — exact in FP8,
      and the descales are scalars so no per-column plumbing is affected.
      THD keeps native dims (thd_d_shapes: the packed THD compile key
      carries no head-dim entries).
    - softmax_precisions: the f16x2 exponent arm lives in the cc 10.7 sibling
      kernels (every per-tensor FP8 flavor), so only that row admits HALF.
      FLOAT is the pipeline every flavor already runs.  The pre-folded scale is
      not served on per-tensor FP8: the kernels fold descale_q * descale_k into
      the softmax scale in-kernel.
    - thd_d_shapes: all SM100 native flavors carry the
      write_thd_meta THD leg; the SM107 row carries all four of its per-tensor
      FP8 siblings (config_sm107.SM107_FP8_THD_SHAPES: d128, d192xd128, d256
      and d512, on the FROST THD contract since 2026-09-09).
    - split_kv_supported / split_d_shapes: SM100 d64/d128, d192x128, d256
      and d512 wire SplitHelpers. D512 graph plans share the standalone
      FP32-partial/combine contract, including its dense head envelope.
    - sched_policies: both rows serve the full {NATURAL, LPT, LPT_L2} domain
      (issue #653) — the SM107 sibling threads qh_per_kh/seqlen_kv through
      every decode call site, which is what the shared LPT_L2 decode requires,
      so its remap is in lockstep with the SM100 twin (place() hands the
      adapter an explicit policy from this domain).

    Padding mask (per-batch ``seq_len_kv`` → KV-side masking) is supported: KV-only
    padding leaves every query row real, so each row's total_sum > 0 and the
    per-row softmax normalization stays well-defined — no
    fully-masked row can poison the global amax. THD/varlen rides the shared
    packed lowering on all SM100 native shapes (write_thd_meta envelope
    design, issue #552; packed Q/K/V/O contract only) and on every Rubin
    per-tensor FP8 sibling (SM107_FP8_THD_SHAPES: d128, d192xd128, d256, d512).
    """

    rubin_row = arch == "sm107"
    return EngineSpec(
        name=f"sdpa_fwd_prefill_{arch}_fp8",
        capabilities=Capabilities(
            # Ranges, not the parts that exist today (see sm_lo above): the
            # split point is the Rubin line — 100-106 runs the SM100 modules,
            # 107-119 the SM107 sibling.
            sm_lo=107 if rubin_row else _BLACKWELL[0],
            sm_hi=_BLACKWELL[1] if rubin_row else 106,
            phase="prefill",
            # Both architectures carry native D64 through their d128 FP8 body.
            # The record selects 64-wide K/O geometry, including packed/split plans.
            d_shapes=frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)}),
            d_pad_multiple=16,
            # The d512 flavor serves the (256, 512] band on BOTH head dims —
            # the range no smaller FP8 flavor reaches, at most 2x zero-padding.
            # Below that floor it declines rather than swallowing e.g. a d256
            # graph onto a kernel whose cga4x1 role-split geometry is tuned for
            # d = 512. D192x128 serves its 16-aligned dense head envelope:
            # Q/K and V descriptors retain the actual logical widths. On SM100,
            # a partial Q/K head uses rank-4 K TMA so storage slack zero-fills;
            # exact D192 retains the rank-5 chunk load. SM107 already uses rank-4.
            # D256 also preserves actual K/V widths, including paged SM100 pools;
            # SM100/SM103 partial heads use bounded rank-4 loads and exact heads
            # keep rank-5. Native SM107 uses rank-4 for both exact and partial heads.
            # Both architecture rows mirror api_dsl._SM100_FP8_ENVELOPE_FLOORS
            # so graph admission and prepared lowering agree.
            d_envelope_floors=(((512, 512), 256),),
            # Both lines serve THD at every native flavor.  The Rubin arm keeps a
            # NAMED constant (shared with the standalone adapter's gate, rule 8b')
            # rather than collapsing to `d_shapes`, so a future partial arch line
            # stays expressible without reintroducing a boolean that cannot say it.
            thd_d_shapes=SM107_FP8_THD_SHAPES if rubin_row else frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset(
                {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
            ),
            # Block-scaled O epilogues (FP4_E2M1 + E4M3/16, FP8_E4M3 + UE8M0/32):
            # the d128 flavor on both arch lines; the adapter declines the
            # wider flavors (config_sm100 backstop).
            o_block_scales=frozenset({0, 16, 32}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            thd=True,
            thd_padded_stats=True,
            cu_seq_len=True,
            padded_stats=True,
            # Native per-tensor FP8 page pools on SM100/103: D64, D128 and
            # D256 use independent K/V block tables and pool strides. Dense Q,
            # scalar-scaled O, padding/causal/windows, sink and legal KV splits
            # compose with the loader. THD Q, block-scaled O and the Rubin
            # siblings retain their existing declines.
            paged_kv=not rubin_row,
            paged_d_shapes=None if rubin_row else frozenset({(64, 64), (128, 128), (256, 256)}),
            # Multi-wave launches are served: the former single_wave_only gate
            # (wrong O past one wave) was removed after the kernel's TMEM stats
            # race was fixed with the mb_stats_read barrier (verified on the
            # gated 132/192/200-cluster repros, 3x each).
            skv_tail_via_padding=True,
            # LPT/LPT_L2 remap is in lockstep with the SM100 sibling (issue
            # #653): every decode call site threads qh_per_kh/seqlen_kv, which
            # is what the shared _common_blackwell LPT_L2 decode demands.  The L2
            # figure the remap blocks against (CFG.L2_SIZE_MIB) is a tuning
            # hint for the grouping, not a chip capacity, so the SM100 number
            # carries to Rubin unchanged.
            # LPT_L2 needs qh_per_kh + seqlen_kv threaded through EVERY decode
            # call site (_common_blackwell._decode_initial raises otherwise).  On the
            # Rubin line only the shipped d128 kernel does that -- d256 and d512
            # have 0/10 sites threaded -- so the row cannot claim the policy for
            # the flavors it now serves.  Heuristics derives its proposals from
            # this domain (heuristics.py:329), so narrowing means those graphs
            # get LPT instead of going unserved: a scheduling trade, not a
            # correctness one.  Re-widen once the args are threaded everywhere.
            # Rubin: NATURAL row-wide, with LPT advertised per-d-shape for the
            # flavors that have been validated under it.
            #
            # The ROOT CAUSE of "the ported kernels do not honor LPT" was a
            # dropped argument, not an incorrect decode: every SM100 kernel calls
            # `make_sdpa_helpers(CFG, lpt_q_tiles_in_cga_units=True)` and the
            # SM107 port omitted it on 9 of 11 flavors. Without it the LPT
            # linearization walks a row range CTA_MMA times too large, no tile
            # is claimed, and the kernel writes NOTHING (cosine 0.0000). The two
            # flavors that kept it -- d128 FP8 and d192 FP8 -- are exactly the
            # ones this comment used to single out as working.
            #
            # Restored on every 2-CTA SM107 flavor; it is a no-op under
            # SCHED_NATURAL, so the shipped path is unchanged (SM107 suite 56
            # passed, block suite 89 passed). Validated under LPT on d256 f16:
            # cos 1.0000 at n_kv 2/3/4/8, dense AND causal. Measured causal SOL
            # on Rubin, S=4096/8192/32768: 53.0/71.1/76.7% NATURAL ->
            # 62.6/79.3/77.5% LPT (+18.2/+11.6/+1.1%); dense is neutral.
            #
            # d512 (cga4x1 role-split) is deliberately untouched: different
            # scheduler shape, and the old NaN report there is unexplained.
            #
            # SCHED_LPT_L2 is claimed PER FLAVOR too, and only where the kernel
            # threads `qh_per_kh` / `seqlen_kv` into every decode call site --
            # the LPT_L2 cost model's inputs, which the shared decode raises
            # without at trace time.  The d128 and d192x128 kernels do (through
            # make_split_helpers); the d256 and d512 kernels do not.
            #
            # The "KNOWN COST" this note used to carry -- that the d128 FP8
            # kernel honours LPT but loses the plan because sched_policies is
            # row-wide -- is what `sched_policies_by_d_shape` below now fixes.
            # d128 FP8 is not listed yet for the tolerance reason given there.
            sched_policies=(frozenset({SCHED_NATURAL}) if rubin_row else frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
            # Rubin: LPT is claimed PER FLAVOR, like the f16 row (`_sm107_spec`).
            # `lpt_q_tiles_in_cga_units=True` is restored on every 2-CTA SM107 FP8
            # kernel (#1001).  VALIDATED under LPT on Rubin (2026-09-11), standalone
            # adapter, E4M3 per-tensor scales, bf16 O, causal + dense + padded,
            # (B, S) in {(1,256), (2,1000), (1,4096)}, against the fp64
            # kernel-mirroring `fp8_ref.compute_ref`:
            #   (256, 256): max|O-ref| 0.0078 causal / <= 0.0019 dense (tol 0.075)
            #   (192, 128): max|O-ref| 0.0397 causal / <= 0.0060 dense (tol 0.04)
            # and on BOTH, O and LSE under LPT are BIT-IDENTICAL to NATURAL (the
            # scheduler reorders whole (batch, head, q-tile) work items; each
            # tile's KV loop is unchanged), sentinel 0, two-launch 0.  Perf node,
            # d256 causal H32/2, LPT vs NATURAL launch-interleaved: +5.2/+5.9/
            # +5.6/+2.0/+2.3 % at S=2K..32K (control pair within 1.9 %).
            # (192, 128) and (128, 128) serve SCHED_LPT_L2 as well (2026-09-14):
            # O and LSE under LPT_L2 are bit-identical to NATURAL on the same
            # inputs (dense and causal, sentinel 0).  (128, 128) joins the LPT
            # claim at the same time: bit-identical too, and the 0.041-0.048
            # its e5m2 causal path reads against the suite's 0.04 comes out of
            # the same bits under every policy, so it is not a scheduler
            # question.  Perf node, kernel-level d128 H64/8 causal vs NATURAL:
            # LPT_L2 +4.2/+7.6/+7.0/+5.9/+5.4 %, LPT +5.3/+5.8/+4.2/+2.0/+1.1 %
            # at S=2K..32K.  What heuristics PROPOSE depends on GQA
            # (heuristics._sched_points): with K/V heads shared across Q heads
            # LPT_L2 leads; with h_q == h_kv (the DSv3 layout) it has nothing to
            # group and measured -9.7 % at S=2K (d192x128 H128), so the Rubin
            # rule picks LPT at few waves (+16 % at S=4K) and NATURAL at many
            # (LPT -6.5 / -13 / -7.9 % at S=8K/16K/32K).  Every policy in the
            # domain stays an autotune runner.
            # NOT claimed: (512, 512) -- the cga4x1 role-split kernel
            # still calls make_sdpa_helpers(CFG) WITHOUT lpt_q_tiles_in_cga_units
            # (the #1001 bug, left on the d512 line), so under LPT it writes
            # NOTHING (sentinel on 100 % of cells; the old "NaN" report was that
            # unwritten output being read).
            sched_policies_by_d_shape=(
                (
                    ((256, 256), frozenset({SCHED_NATURAL, SCHED_LPT})),
                    ((192, 128), frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
                    ((128, 128), frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
                    ((64, 64), frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
                )
                if rubin_row
                else ()
            ),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # Native D64: cga1 on SM100; cga1/2 on Rubin, whose P.V descriptor
            # models the 32-byte per-CTA V swizzle (api_dsl.supported_cgas_for).
            # (128, 128): the per-tensor FP8 d128 kernel builds at both widths on
            # BOTH legs; the heuristic runs a dense graph at cga1 (heuristics.
            # _auto_sched_cga).  The split leg used to be held at cga2: on a
            # decode shape that doubles the MMA work per KV tile (each CTA of the
            # pair runs the full 256-row tile against the same keys) and the
            # kernel is MMA-bound there, so the split could never pay (B300,
            # llama 64/8 e4m3, 128k keys, b=1: 134 us at cga2 x split 8, flat to
            # split 32; b=8 stayed unsplit at 895 us).
            cgas_by_d_shape=(
                (((64, 64), frozenset({1, 2})), ((256, 256), frozenset({1})))
                if rubin_row
                else (((64, 64), frozenset({1})), ((128, 128), frozenset({1, 2})), ((192, 128), frozenset({1, 2})), ((256, 256), frozenset({1})))
            ),
            split_cgas_by_d_shape=(() if rubin_row else (((64, 64), frozenset({1})), ((192, 128), frozenset({2})))),
            # f16x2-softmax arm: only the cc 10.7 sibling kernels carry the
            # path, in every per-tensor FP8 flavor (MUFU EX2.F16x2 exists below
            # cc10.7 but no other file wires it). FLOAT is the f32 pipeline
            # every flavor already runs.
            softmax_precisions=(frozenset({cudnn.data_type.FLOAT, cudnn.data_type.HALF}) if rubin_row else frozenset({cudnn.data_type.FLOAT})),
            split_kv_supported=True,
            split_d_shapes=(SM107_FP8_SPLIT_SHAPES if rubin_row else frozenset({(64, 64), (128, 128), (192, 128), (256, 256), (512, 512)})),
            pack_gqas=frozenset({False, True}),
            # Native SM107 per-tensor FP8 D64/D128/D192x128/D256 pack whole
            # GQA groups. D512 retains its independent packing decline.
            pack_gqa_d_shapes=(SM107_FP8_PACK_GQA_SHAPES if rubin_row else None),
            # Fused epilogue gate on the Rubin d256 FP8 kernel (E4M3 / E5M2 in,
            # any out dtype): G is staged in BF16 (kernel GATE_STORAGE_DTYPE), so
            # the row names that one dtype rather than inheriting Q's.  The FP8
            # O quantizes the GATED value; Amax_O, when requested, is the amax of
            # the UNGATED normalised O (the sdpa node's output precedes the
            # sigmoid/mul tail, so G cannot move it).  Not on the SM100 line.
            epilogue_gate=rubin_row,
            epilogue_gate_d_shapes=(SM107_EPILOGUE_GATE_SHAPES if rubin_row else None),
            epilogue_gate_dtypes=(frozenset({cudnn.data_type.BFLOAT16}) if rubin_row else None),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm107_mxfp8_spec() -> EngineSpec:
    """Block-scale MXFP8 on Rubin (E4M3/E5M2 + per-32-block E8M0 SF).

    Rubin's own row rather than a widened SM100 one, for the same reason the
    f16 and FP8 rows split at the arch line: these are the SM107 sibling
    kernels (dense K=64 MMA, version-1 SMEM descriptors).  Rubin also carries a
    d512 MXFP8 flavor, which SM100 does NOT -- so this row is WIDER than its
    Blackwell counterpart at the top end.  d192xd128 now has a Rubin MXFP8
    sibling too, but at cga2 ONLY (SM100 serves it at both widths): the wider K
    pushes this flavor's scale-factor tiles past the 256 KiB version-0
    descriptor window at cga1.

    Declined deliberately, because the ported kernels lack the machinery (not
    because it went untested): split-KV and PackGQA (the dense padded-Q trim is
    carried since #1037).  THD/varlen is served at d256 ONLY
    (``thd_d_shapes = config_sm107.SM107_MXFP8_THD_SHAPES``, one constant with the
    standalone adapter's gate and its ``_can_prepare_mxfp8``, rule 8b'): that body
    rides the FROST THD contract at cga1 with the PACKED per-sequence-TILE-padded
    scale-factor layout the SM100 row and the SM107 MXFP8 backward already use
    (``[1, H, Σ_b ceil(S_b/128), SF_SMEM]`` tile sequences in cu_seqlens order;
    the native binder derives the packed tile extent from the bound buffer's byte
    size, so a producer may hand zero-filled slack tiles past the live total);
    the d128 / d192x128 / d512 MXFP8 bodies keep the pre-upstream THD arm and
    stay declined.  Paged KV IS served on d128 / d256 with dense queries
    (F8_128x4 descale POOLS paging with K/V, page_size % 128 == 0, sinks
    compose); THD queries over pools and the d192x128 / d512 pools are the SM107
    follow-ups in the tracker.  Optional stats IS served (``lse_optional=True`` below -- has_lse=False
    is a real specialization on every Rubin kernel, not an accepted-and-ignored
    flag).  See _sm107_spec for the same list on f16.

    Fused epilogue gate (PR-B, 2026-09-15): the d256 MXFP8 body carries the same
    ``O := O * sigmoid(G)`` hook as the f16 / per-tensor FP8 d256 kernels, so
    this row claims it at EXACTLY (256, 256) through the shared constant, with a
    bf16 G (the kernel's GATE_STORAGE_DTYPE).  A gated e4m3 O is written
    UNSCALED -- the MXFP8 kernel has no per-tensor scale_o (block scales
    dequantize in-MMA), which is the same O the ungated row writes.  Amax_O,
    when requested, is the amax of the UNGATED normalised O -- the sdpa node's
    output precedes the sigmoid/mul tail, so G cannot move it -- in the O's
    own units (the FP8 row's contract, minus the scale_o).
    """
    return EngineSpec(
        name="sdpa_fwd_prefill_sm107_mxfp8",
        capabilities=Capabilities(
            sm_lo=107,
            sm_hi=_BLACKWELL[1],
            phase="prefill",
            # Exact native shapes only -- the SF tensors are not zero-padded.
            # (192, 128) deliberately takes NO cgas_by_d_shape entry below, so it
            # stays on the row default cgas={2}: at cga1 its four SF tiles start
            # past the 256 KiB version-0 tcgen05 descriptor window and the UTCCP
            # would read Q data as scale factors.  See make_cfg_d192_mxfp8.
            d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            d_pad_multiple=0,
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset(
                {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
            ),
            # Block-scaled O epilogues (FP4_E2M1 + E4M3/16, FP8_E4M3 + UE8M0/32) on
            # the d128 MXFP8 kernel; the adapter declines the wider flavors.
            o_block_scales=frozenset({0, 16, 32}),
            is_mxfp8=True,
            # f16x2-exponent arm (softmax_precision=HALF) and the pre-folded scale (attn_scale_prefolded,
            # fused with the f16 arm on stats-less graphs): every MXFP8 flavor carries both.  FLOAT is the
            # f32 pipeline.
            softmax_precisions=frozenset({cudnn.data_type.FLOAT, cudnn.data_type.HALF}),
            attn_scale_prefolded_d_shapes=frozenset({(128, 128), (192, 128), (256, 256), (512, 512)}),
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            padded_stats=True,
            # See the f16 SM107 row: has_lse=False is a real specialization on
            # every Rubin kernel, not an accepted-and-ignored flag.
            lse_optional=True,
            skv_tail_via_padding=True,
            # THD/varlen at d256 ONLY (SM107_MXFP8_THD_SHAPES -- shared with the
            # standalone adapter's gate and _can_prepare_mxfp8, rule 8b').  The
            # d256 body rides the FROST THD contract at cga1: packed [1,T,H,D]
            # Q/K/V/O + cu_seqlens (both length forms -> cu_seq_len), PACKED
            # per-sequence-TILE-padded SF tensors, and the ragged Stats in all
            # three layouts -- token-major, head-major and per-batch padded
            # (thd_padded_stats).  The d128 / d192x128 / d512 MXFP8 bodies keep
            # the pre-upstream THD arm (7-arg setup call, 3B+2 metadata, static
            # K/V maps) and stay declined through thd_d_shapes.
            thd=True,
            thd_d_shapes=SM107_MXFP8_THD_SHAPES,
            thd_padded_stats=True,
            cu_seq_len=True,
            # Paged KV (issue #920 on cc 10.7; the SM100 MXFP8 row's pool contract verbatim): K/V page pools and the
            # F8_128x4 descale POOLS that page with them through the block tables (page id = TMA batch coordinate of
            # the K/V and SF descriptors, tile-in-page = SF tile coordinate, page -1 past a sequence's live pages =
            # TMA-OOB zero fill), page_size % 128 (mismatch() gates it), the sink composed (epilogue fold vs. loader;
            # validated on a Rubin dev node incl. keyless rows), softmax_precision=HALF composed, the pre-folded scale
            # declined over paged KV like every row.  Dense queries on the d128 / d256 siblings only: THD queries over
            # pools (the THD arm is a separate leg of the d256 body) and the d192x128 / d512 pools stay declined
            # until their loaders are ported.
            paged_kv=True,
            paged_d_shapes=frozenset({(128, 128), (256, 256)}),
            # NATURAL row-wide; LPT and LPT_L2 claimed PER FLAVOR, like the f16
            # and FP8 rows, where the kernel honours them.
            #
            # This row was NATURAL-only until 2026-09-14.  The report that kept
            # it there -- causal MXFP8 graphs under LPT turning 23 green tests
            # red with max|O-ref| ~ 1.9-4.1 while dense stayed correct -- has
            # the #1001 signature: the SM107 port had dropped
            # `lpt_q_tiles_in_cga_units=True`, so the LPT decode claimed no
            # tile and the kernel wrote nothing (dense graphs rank NATURAL and
            # never took that path).  With the argument restored the d128 and
            # d192x128 MXFP8 kernels honour LPT, and they now thread
            # `qh_per_kh` / `seqlen_kv` into every decode call site (the
            # LPT_L2 cost-model inputs the shared decode raises without), so
            # they honour LPT_L2 as well.  VALIDATED on Rubin through the
            # standalone adapter (E4M3 block scales, bf16 O, dense and causal):
            # O and LSE under LPT and under LPT_L2 are BIT-IDENTICAL to NATURAL
            # -- the scheduler reorders whole (batch, head, q-tile) work items,
            # each tile's KV loop is unchanged -- with a sentinel-filled O
            # (0 unwritten cells); and the MXFP8 suite, whose causal cases now
            # rank LPT_L2 first, stays green.  Pinned by the Rubin e2e
            # test_mxfp8_sched_policies_are_bit_identical_to_natural.
            # NOT claimed: (256, 256) and (512, 512) -- their kernels thread
            # neither argument, and d512 (cga4x1 role-split) still lacks the
            # #1001 argument and writes nothing under LPT.
            sched_policies=frozenset({SCHED_NATURAL}),
            sched_policies_by_d_shape=(
                ((128, 128), frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
                ((192, 128), frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2})),
            ),
            tile_ms=frozenset({128}),
            tile_ns=frozenset({128}),
            cgas=frozenset({2}),
            # d256 MXFP8 serves both widths: the dense unsplit legs run the 2-CTA pair on cc 10.7
            # (config_sm100.d256_mxfp8_pair_leg: -8..-13 % dense / -6..-7 % causal vs one CTA, O bitwise identical);
            # effective_cgas narrows THD, split-KV and paged graphs to cga1.
            cgas_by_d_shape=((((256, 256), frozenset({1, 2})),)),
            pack_gqas=frozenset({False}),
            # Fused epilogue gate: the d256 MXFP8 body carries the seams; ONE
            # constant with the f16 / FP8 rows and the adapter twin (rule 8b').
            # G is bf16 (the kernel stages a bf16 gate tile, never an e4m3 one).
            # The quantized O is the GATED value; Amax_O, when requested, is the
            # amax of the UNGATED normalised O (the sdpa node's output precedes
            # the sigmoid/mul tail, so G cannot move it) in the O's own units --
            # this path has no per-tensor scale_o.
            epilogue_gate=True,
            epilogue_gate_d_shapes=SM107_EPILOGUE_GATE_SHAPES,
            epilogue_gate_dtypes=frozenset({cudnn.data_type.BFLOAT16}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM100),
    )


def _sm80_spec() -> EngineSpec:
    """SM80 (A100) prefill row: lowers through ``lower_dsl_prefill`` onto the
    ``SdpaFwdDslSm80`` adapter (``fwd/api_dsl.py``), which owns kernel-flavor
    selection (gptoss/llama/dsv3/qwen), host-side head-dim padding (hence
    ``d_pad_multiple=1``), BHSD<->BSHD normalization, and per-shape kernel
    caching. Native-flavor packed THD graphs bind the varlen template directly;
    the dense head-dim envelope is not offered to THD. Dense graphs expose
    scheduler tuning; THD requires the natural 3-D grid."""
    return EngineSpec(
        name="sdpa_fwd_prefill_sm80",
        capabilities=Capabilities(
            zero_scale=True,  # score_sign specialization (#1435)
            sm_lo=80,
            sm_hi=80,  # A100 exactly: the kernels assume its 164 KiB opt-in SMEM
            phase="prefill",
            d_shapes=frozenset({(256, 256)}),  # flavor envelopes; host-side zero-padding
            d_pad_multiple=1,
            thd_d_shapes=frozenset({(64, 64), (128, 128), (192, 128), (256, 256)}),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            bias=True,
            right_band_widening=True,
            causal=True,
            bottom_right=True,
            swa=True,
            padded=True,
            thd=True,
            cu_seq_len=True,
            sink=True,
            stats=True,
            stats_log2=True,
            padded_stats=True,
            decode=False,
            # The kernels implement the dense padded-Q trim natively
            # (per-batch ``seq_len_q`` forward kwarg): rows >= seq_len_q[b]
            # are written explicitly by the kernel (O := 0, LSE := -inf).
            lse_optional=True,
            layouts=frozenset({"bshd", "dense_flex"}),
            skv_tile=0,  # the kernels' is_even_k path serves ragged S_kv
            # Dense graphs serve all three schedulers; the THD grid is 3-D
            # and the THD-only mismatch above accepts natural scheduling only.
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM80),
    )


def _sm120_spec() -> EngineSpec:
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import D512_FLAVOR, GENERAL_HEAD_TILE_MAX, GENERAL_HEAD_TILES

    return EngineSpec(
        name="sdpa_fwd_prefill_sm120",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            # The general template picks its Q/K and V head tiles independently,
            # so the native shapes are the cross product of the supported tiles;
            # the d512 flavor (sm120/prefill_d512_f16.py) adds its (512, 512).
            d_shapes=frozenset((tq, tv) for tq in GENERAL_HEAD_TILES for tv in GENERAL_HEAD_TILES) | {D512_FLAVOR},
            d_envelope_floors=((D512_FLAVOR, GENERAL_HEAD_TILE_MAX),),
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            padded_stats=True,
            thd=True,
            thd_padded_stats=True,
            # No KV-tail rule: the kernel walks KV tiles right-to-left and its
            # first (masked) step always covers the rightmost — and therefore
            # any partial — tile, comparing columns against seqlen_k regardless
            # of mask flags. Ragged S_kv is served natively with no synthesized
            # padding and no padded-path cost.
            skv_tile=0,
            cu_seq_len=True,
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            # The kernel's inline chunking + the shared split_combine pass
            # (the combine is one block per row — arch-agnostic). The config
            # backstop bars a split under the LPT remaps, so the heuristic's
            # split sets ride SCHED_NATURAL.
            split_kv_supported=True,
            # 16 / 32 are the half DECODE tile (sm120/decode_f16.py, packed rows
            # x the warps on the key axis), whose tile_n is its K/V stage width
            # (config_sm120.decode_kv_stage: 64 keys at head tiles up to 32, 32
            # at 48 and 64, 16 above; config_sm120.tile_domain).
            tile_ms=frozenset({16, 32, 64, 128}),
            # 32 is also the d512 flavor's KV tile (config_sm120.tile_domain).
            tile_ns=frozenset({16, 32, 64, 128}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
            pack_gqa_requires_divisible=False,
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


def analyze_for(spec: EngineSpec, graph, knobs: Optional[SdpaFwdKnobs] = None):
    """``(facts, reason)``: the parsed graph and the first reason ``spec``
    cannot serve it under ``knobs`` (``None`` when it can).

    The single eligibility entry point, shared by :func:`probe`, :func:`build`
    and ``engine.FrostSdpaFwdEngine.check_support``. ``knobs`` is the plan's
    tuning request (``PlanConfig.knobs``), ``None`` for no preference.
    """
    # The record validate() attached, not a fresh parse: one per graph, shared
    # with whatever ranked these plans before this engine was imported.
    facts = graph._facts_for(ga.analyze)
    if facts is None:
        return None, "graph is not a single sdpa() forward node (optionally followed by O * sigmoid(G))"
    return facts, mismatch(spec.capabilities, facts, knobs)


def build(spec: EngineSpec, graph, knobs: Optional[SdpaFwdKnobs] = None):
    """Lower ``spec`` for ``graph``, or raise the bare ineligibility reason (the
    caller — the engine — names itself in the message)."""
    facts, reason = analyze_for(spec, graph, knobs)
    if reason is not None:
        raise ValueError(reason)
    return spec.lower(spec, facts, knobs)


def _table_stride(t) -> tuple:
    """A paged block table's ``(batch, page)`` strides from its ``(B, 1, max_pages, 1)`` IR ref."""
    s = tuple(int(x) for x in t.get_stride())
    return (s[0], s[2])


def _table_view(buf, ir_t, b: int):
    """The kernel's ``(B, max_pages)`` view of a bound ``(B, 1, max_pages, 1)`` table — a
    view of the declared layout (size-1 axes dropped), never a copy."""
    return buf.as_strided((b, int(ir_t.get_dim()[2])), _table_stride(ir_t), buf.storage_offset())


def _epilogue_gate_ctor_kwargs(facts: "ga.SdpaGraphFacts", ctor_params: frozenset, exec_params: frozenset, engine_name: str) -> dict:
    """The standalone adapter's gate / Amax_O constructor kwargs for these facts.

    Contract (frozen, PR-A): ``SdpaFwdDsl.__init__(..., sample_gate=None,
    has_amax_o=True)`` and ``execute(..., gate=None)``.  All are
    FEATURE-DETECTED on the adapter class so a row whose adapter predates them
    keeps lowering unchanged:

    * ``sample_gate`` -- only when the graph carries the gate tail.  An adapter
      without the constructor keyword, or whose ``execute()`` cannot take the
      ``gate=`` buffer (never run the gated specialization without handing it
      G: that would silently write the un-gated O), cannot serve the tail, so
      that is a typed decline (``NotImplementedError`` advances the build walk)
      rather than a ``TypeError`` out of the constructor -- raised HERE, before
      the constructor and the kernel JIT.  Normally unreachable: mismatch()
      admitted the gate only on rows whose adapter carries both.
    * ``has_amax_o`` -- quantized graphs only: ``False`` when the graph did not
      request ``Amax_O`` (facts.amax_o_t is a REAL set_output(True) tensor or
      None), which lets the kernel fold its amax atomic out instead of writing
      a buffer nobody reads.  Half graphs never pass it (the legacy ``True``
      default is exactly their semantics).
    """
    out: dict = {}
    if facts.has_epilogue_gate:
        if "sample_gate" not in ctor_params:
            raise NotImplementedError(f"{engine_name}: this adapter does not carry the epilogue gate (no sample_gate)")
        if "gate" not in exec_params:
            raise NotImplementedError(f"{engine_name}: the adapter's execute() does not accept the epilogue gate (gate=)")
        out["sample_gate"] = ga.tensor_desc_from_ir(facts.epilogue_gate_t, name="gate")
    if (facts.is_fp8 or facts.is_mxfp8) and "has_amax_o" in ctor_params:
        out["has_amax_o"] = facts.amax_o_t is not None
    return out


def lower_dsl_prefill(
    spec: EngineSpec,
    facts: "ga.SdpaGraphFacts",
    knobs: Optional[SdpaFwdKnobs] = None,
    api_type: str = _SM100,
):
    """Lower one selected SDPA prefill engine through its DSL adapter.

    Every architecture adapter implements the same constructor and execution
    interface — the keyword contract is declared as ``SdpaFwdDsl.execute``.
    ``EngineSpec.lower`` may bind a different implementation through
    ``api_type``; descriptor conversion, adapter lifecycle, variant-pack binding,
    and launch construction remain shared here.
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.api_dsl import _parameter_names, _torch_stream_context

    seq_q_t = facts.seq_q_t if facts.padded else None
    seq_kv_t = facts.seq_kv_t if facts.padded else None
    # Mirrors the seq_q_lens_present constructor argument below: a dense padded
    # graph carrying per-batch Q lengths compiles the kernel's trim
    # specialization (every kernel has one), and execute forwards the buffer
    # (THD sources cu_seqlens from it instead).
    seq_q_lens_present = facts.padded and not facts.thd and facts.seq_q_t is not None
    api_cls = _adapter(api_type)
    _ctor_params = _parameter_names(api_cls.__init__)
    _exec_params = _parameter_names(api_cls.execute)
    api = api_cls(
        sample_q=ga.tensor_desc_from_ir(facts.q_t, name="q"),
        sample_k=ga.tensor_desc_from_ir(facts.k_t, name="k"),
        sample_v=ga.tensor_desc_from_ir(facts.v_t, name="v"),
        sample_o=ga.tensor_desc_from_ir(facts.o_t, name="o"),
        sample_lse=ga.tensor_desc_from_ir(facts.stats_t, "lse") if facts.stats_t is not None else None,
        # A right-widened band lowers as the causal mask with a BAND_RIGHT
        # diagonal offset (facts.causal is False when right_bound > 0).
        is_causal=facts.causal or facts.right_band_widening,
        causal_bottom_right=facts.bottom_right,
        window_size_left=facts.window_left,
        window_size_right=(facts.right_bound if facts.right_band_widening else None),
        scale_softmax=None if facts.attn_scale_prefolded else facts.scale,  # the fold: Q carries the scale
        seq_kv_lens_present=facts.padded,
        # Dense padded-Q trim (q rows >= seq_len_q[b] -> O := 0, LSE := -inf):
        # enabled whenever a dense padded graph carries per-batch Q lengths.
        # THD carries Q lengths via cu_seqlens; support is selected per native
        # flavor above because only kernels with a quantized Q-length ABI consume it.
        seq_q_lens_present=seq_q_lens_present,
        # cu_seq_len form (THD-only; the probe declined dense cu graphs): the
        # adapter's seq-lens execute arguments carry (B+1,) prefix sums.
        cu_seq_q_lens=facts.cu_seq_q_t is not None,
        cu_seq_kv_lens=facts.cu_seq_kv_t is not None,
        has_sink=facts.has_sink,
        stats_log2=facts.has_stats_log2,
        thd=facts.thd,
        # THD Stats without ragged offsets = FlashInfer's per-batch padded (b, s_max, h) buffer
        thd_stats_padded=(facts.thd and facts.stats_t is not None and getattr(facts.stats_t, "ragged_offset", None) is None),
        # The decode tile's ragged-Q leg reads the ragged offsets on device as
        # token rows: the (Q, O, Stats) elements-per-token divisors it divides by.
        **(
            {"ragged_divisors": _thd_decode_leg_divisors(facts), "ragged_offsets_int64": _thd_decode_leg_int64(facts)}
            if (_thd_decode_leg(spec.capabilities, facts) and "ragged_divisors" in _ctor_params)
            else {}
        ),
        # Caller-declared packed token totals (issue #624): when present the
        # adapter binds EXACT token extents instead of the buffer-derived
        # capacity, putting an over-allocated buffer's uninitialized tail out
        # of TMA reach. Only ever tightens (see _thd_declared_total).
        max_total_seq_len_q=facts.max_total_seq_len_q,
        max_total_seq_len_kv=facts.max_total_seq_len_kv,
        # Paged KV: K/V samples are the page pools; the adapter needs the page
        # geometry and the declared KV maximum the masks clamp against.
        paged_page_size=facts.page_size if facts.has_paged_kv else 0,
        paged_max_seq_len_kv=facts.s_kv if facts.has_paged_kv else None,
        # The tables' declared (batch, page) strides — the (B, 1, max_pages, 1)
        # IR tensor's axes 0 and 2 — so batch-innermost / padded tables bind
        # as views.
        paged_table_stride=_table_stride(facts.paged_k_table_t) if facts.has_paged_kv else None,
        paged_table_v_stride=_table_stride(facts.paged_v_table_t) if facts.has_paged_kv else None,
        dtype_o=facts.dtype_o if (facts.is_mxfp8 or facts.is_fp8) else None,
        pertensor_fp8=facts.is_fp8,
        # Block-scaled O: the sf_o output's declared geometry selects the
        # per-(b,h)-plane or token-major scale-factor layout (see
        # SdpaFwdDsl._sf_o_geometry).
        sample_sf_o=ga.tensor_desc_from_ir(facts.sf_o_t, name="sf_o") if facts.sf_o_t is not None else None,
        # MXFP8 input: scale_o is a python-only OPTIONAL input (the FP4 global
        # scale) and its presence is a compile form of the kernel (identity fold
        # otherwise). The per-tensor FP8 op's scale_o is a required execute operand.
        sample_scale_o=ga.tensor_desc_from_ir(facts.scale_o_t, name="scale_o") if (facts.is_mxfp8 and facts.scale_o_t is not None) else None,
        sched_policy=knobs.sched_policy if knobs is not None else None,
        tile_m=knobs.tile_m if knobs is not None else None,
        tile_n=knobs.tile_n if knobs is not None else None,
        cga=knobs.cga if knobs is not None else None,
        pack_gqa=knobs.pack_gqa if knobs is not None else None,
        split_kv=knobs.split_kv if knobs is not None else None,
        softmax_precision=facts.softmax_precision,  # op attribute (None = the f32 pipeline)
        # Op attribute attn_scale_prefolded -> the adapter's softmax_scale_prefolded (the row's
        # attn_scale_prefolded_d_shapes claim gated it; every forward adapter declares the kwarg).
        softmax_scale_prefolded=facts.attn_scale_prefolded,
        # Epilogue gate (sample_gate=) and the Amax_O fold-out (has_amax_o=):
        # feature-detected on the adapter's constructor, see the helper.
        **_epilogue_gate_ctor_kwargs(facts, _ctor_params, _exec_params, spec.name),
        # Score side outputs (score_max / score_sum_exp): plan-time state of the adapter like
        # sample_lse (None compiles the store out).  Forwarded only to a constructor that declares
        # them -- every other row's mismatch declined the ports already (the row-level flags).
        **(
            {
                "sample_score_max": ga.tensor_desc_from_ir(facts.score_max_t, name="score_max") if facts.score_max_t is not None else None,
                "sample_score_sum_exp": ga.tensor_desc_from_ir(facts.score_sum_exp_t, name="score_sum_exp") if facts.score_sum_exp_t is not None else None,
            }
            if "sample_score_max" in _ctor_params
            else {}
        ),
        # SM80-only PLAN-TIME axes (bias presence/dtype are compile-time
        # specializations of that template): forwarded only to adapters whose
        # constructor declares them — every other row's mismatch gated the
        # operands off already.
        **(
            {
                "bias_present": facts.bias_t is not None,
                "bias_fp32": facts.bias_t is not None and facts.bias_t.get_data_type() == cudnn.data_type.FLOAT,
            }
            if "bias_present" in _ctor_params
            else {}
        ),
        # The resident-block arm (the DSL adapters): the bias and the per-key mask are plan-time SAMPLES like
        # sample_lse (their presence is a compile-time arm of the kernel); forwarded only to a constructor that
        # declares them -- every other row's mismatch declined the operands already.
        **({"sample_bias": ga.tensor_desc_from_ir(facts.bias_t, name="bias")} if ("sample_bias" in _ctor_params and facts.has_bias) else {}),
        **({"sample_kv_mask": ga.tensor_desc_from_ir(facts.kv_mask_t, name="kv_mask")} if ("sample_kv_mask" in _ctor_params and facts.has_kv_mask) else {}),
    )
    api.check_support()  # raises ValueError / NotImplementedError if unsupported
    if (
        facts.thd
        and not getattr(api, "thd_decode_leg", False)
        and facts.epilogue_gate_t is None
        and (
            facts.device_cc in ((10, 0), (10, 3))
            or (facts.device_cc == (10, 7) and facts.has_paged_kv and facts.d_qk == 128 and api.split_kv == 1 and not (facts.is_fp8 or facts.is_mxfp8))
        )
        and (facts.d_qk, facts.d_v) in ((128, 128), (256, 256), SM100_NATIVE_D576_SHAPE)
        and (
            (
                not (facts.is_fp8 or facts.is_mxfp8)
                and (
                    ((api.split_kv == 1 or (facts.d_qk, facts.d_v) == SM100_NATIVE_D576_SHAPE) and api.cga in (None, 2))
                    or (facts.has_paged_kv and facts.d_qk == 128 and api.cga == 1)
                )
            )
            or (facts.is_fp8 and facts.has_paged_kv and api.split_kv == 1)
        )
    ):
        from .thd_origins import graph_origin_specs

        api._thd_origin_specs = graph_origin_specs(facts)
    if facts.shape_overrides:
        reason = _prepared_decline_reason(spec.capabilities, facts, getattr(api, "split_kv", 1))
        if reason is not None:
            raise NotImplementedError(reason)
    api.compile()
    # The template file that serves this plan (e.g. "prefill_d256_f16" vs the
    # decode-shaped "decode_d256_f16"), when the adapter records one.
    kernel_template = getattr(api, "kernel_template", None)
    # ... and the softmax arms that template traced (api_dsl.softmax_arms_of), when the adapter records them.
    softmax_arms = getattr(api, "softmax_arms", None)
    # ... and the loaded kernel MODULE itself (its PACK / split / paged constants are the geometry the plan
    # really runs: HEADS_PER_TILE, CFG.SPLIT_KV, PAGED_KV), when the adapter keeps one.
    kernel_module = getattr(api, "_k_mod", None)

    # Workspace requirement for the compiled geometry: every per-execute scratch
    # buffer is carved from the CALLER's workspace, so its size is fixed here at
    # build time and recorded on the executor as ``workspace_bytes`` — that
    # number is what the plan's CompiledPlan.get_workspace_size() reports.
    #   - api-level scratch (api.scratch_workspace_bytes()): the dense padded
    #     [seq_kv|seq_q] combine and the THD metadata/LSE buffers.
    # No dummy-LSE chunk: every lower_dsl_prefill row is lse_optional (the
    # kernels None-specialize the LSE argument and compile the store out), so
    # a stats-less graph binds no LSE buffer at any level.
    api_scratch_bytes = api.scratch_workspace_bytes()
    total_workspace_bytes = api_scratch_bytes

    # SM80-only feature operand (bias): the row's capability gate admitted it,
    # and the adapter's execute() declares the matching optional keyword —
    # forwarded below only when both hold. ALiBi / block_mask / score-stats
    # graphs never reach a FROST row (every capability row declines them, so
    # the backend serves them).
    # (A gated graph on an adapter whose execute() lacks ``gate=`` was declined
    # by _epilogue_gate_ctor_kwargs BEFORE the constructor and the JIT above.)
    _extra_exec_keys = {"bias_tensor", "gate", "kv_mask_tensor"} & _exec_params

    binding = ga.SdpaBinding(
        q=facts.q_t,
        k=facts.k_t,
        v=facts.v_t,
        o=facts.o_t,
        stats=facts.stats_t,
        sink_token=facts.sink_t,
        seq_len_kv=seq_kv_t,
        seq_len_q=seq_q_t,
        cu_seq_len_q=facts.cu_seq_q_t,
        cu_seq_len_kv=facts.cu_seq_kv_t,
        paged_k_table=facts.paged_k_table_t,
        paged_v_table=facts.paged_v_table_t,
        bias=facts.bias_t,
        # The per-key boolean mask: a bound operand of a row whose adapter folds it (the Rubin small tiles).
        kv_mask=facts.kv_mask_t if "kv_mask_tensor" in _extra_exec_keys else None,
        sf_q=facts.sf_q_t,
        sf_k=facts.sf_k_t,
        sf_v=facts.sf_v_t,
        amax_o=facts.amax_o_t,
        sf_o=facts.sf_o_t,
        descale_q=facts.descale_q_t,
        descale_k=facts.descale_k_t,
        descale_v=facts.descale_v_t,
        scale_o=facts.scale_o_t,
        # The gate tail's G is a REQUIRED bound operand; its virtual O_v and s
        # never are (facts.o_t already points at the mul output).
        gate=facts.epilogue_gate_t,
        # Score side outputs: bound operands of a row that serves them (the plan then requires their
        # buffers); a row whose adapter has no such operand never reaches here with the ports set.
        score_max=facts.score_max_t if "sample_score_max" in _ctor_params else None,
        score_sum_exp=facts.score_sum_exp_t if "sample_score_max" in _ctor_params else None,
        # THD ragged offsets: bound operands of the decode tile's ragged-Q leg
        # (read on device as the row bases). The SM100 half D128/D256 unsplit
        # CGA2 prefill path also binds independent origins for all five ports.
        # A stats-less graph has no Stats offsets.
        ragged_q=getattr(facts.q_t, "ragged_offset", None) if facts.thd else None,
        ragged_k=getattr(facts.k_t, "ragged_offset", None) if facts.thd else None,
        ragged_v=getattr(facts.v_t, "ragged_offset", None) if facts.thd else None,
        ragged_o=getattr(facts.o_t, "ragged_offset", None) if facts.thd else None,
        ragged_stats=getattr(facts.stats_t, "ragged_offset", None) if (facts.thd and facts.stats_t is not None) else None,
    )
    thd_decode_leg = bool(getattr(api, "thd_decode_leg", False))

    def _ir_view(buf, dim, stride):
        """Reinterpret a variant-pack buffer through the IR tensor's dim/stride.

        cuDNN's execute contract treats variant-pack entries as raw storage laid
        out per the IR tensor descriptor — callers may hand in a torch tensor
        whose *logical* shape is anything with the right bytes (e.g. a
        (B,S,H,D)-contiguous allocation for a (B,H,S,D) BSHD-strided IR tensor,
        as test_mhas_v2's fp8 harness does), so rebuild the IR-shaped view here
        instead of trusting the caller's metadata. No-op when the caller already
        passed an IR-shaped view. THD buffers are packed (fewer elements than
        dim x stride implies), so the caller's view is kept as-is there.
        """
        if tuple(buf.shape) == dim and tuple(buf.stride()) == stride:
            return buf
        return buf.as_strided(dim, stride)

    # What every execute needs that the graph fixed — operand ids, IR layouts,
    # which feature operands the facts demand, the static keywords — resolved
    # once here; an execute is then dict lookups and one adapter call.
    def _layout(t):
        dim, stride = tuple(t.get_dim()), tuple(t.get_stride())
        if t.get_data_type() == cudnn.data_type.FP4_E2M1:
            # Two E2M1 per byte: the buffer is the byte container of the logical
            # geometry (unit-stride extent and every other stride halved); the
            # adapter binds it as FP8-typed bytes (two E2M1 per byte).
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.graph_types import storage_geometry

            geom = storage_geometry(dim, stride, t.get_data_type())
            if geom is None:
                raise ValueError(f"FP4 O geometry dim={dim} stride={stride} has no byte-container spelling")
            dim, stride = geom
        return (dim, stride)

    id_q, id_k, id_v, id_o = id(binding.q), id(binding.k), id(binding.v), id(binding.o)
    lay_q, lay_k, lay_v, lay_o = _layout(binding.q), _layout(binding.k), _layout(binding.v), _layout(binding.o)
    id_stats = id(binding.stats) if binding.stats is not None else None
    sink_src = facts.sink_t if facts.has_sink else None
    # Either length form satisfies a side: per-batch seq_len_* or the (B+1,)
    # cu_seq_len_* prefix sums — the cu buffer travels through the same operand
    # slot (the adapter was constructed knowing the form).
    seq_kv_src = (facts.cu_seq_kv_t if facts.cu_seq_kv_t is not None else facts.seq_kv_t) if facts.padded else None
    seq_q_src = (facts.cu_seq_q_t if facts.cu_seq_q_t is not None else facts.seq_q_t) if facts.padded else None
    forward_seq_q = seq_q_lens_present or facts.thd
    bias_src = facts.bias_t if facts.has_bias else None
    forward_bias = bias_src is not None and "bias_tensor" in _extra_exec_keys
    # The DSL adapters bind the bias through its IR dims / strides (a [1|B, 1|H, S_q, S_kv] view over the caller's storage,
    # like the gate); the SM80 adapter takes the buffer as handed.
    lay_bias = _layout(binding.bias) if (forward_bias and "sample_bias" in _ctor_params) else None
    kv_mask_src = facts.kv_mask_t if (facts.has_kv_mask and "kv_mask_tensor" in _extra_exec_keys) else None
    lay_kv_mask = _layout(binding.kv_mask) if kv_mask_src is not None else None
    gate_src = facts.epilogue_gate_t if (getattr(facts, "has_epilogue_gate", False) and "gate" in _extra_exec_keys) else None
    lay_gate = _layout(binding.gate) if gate_src is not None else None
    # Score side outputs: the IR tensors are rank-4 (B, H_q, S_q, 1) like Stats; the adapter views the
    # bound buffer through those dims / strides and squeezes the unit axis itself.
    score_srcs = tuple(
        (name, t, _layout(t))
        for name, t in (("score_max_tensor", binding.score_max), ("score_sum_exp_tensor", binding.score_sum_exp))
        if t is not None and name in _exec_params
    )
    quant_ids = (
        {
            name: id(t)
            for name, t in (
                ("sf_q", binding.sf_q),
                ("sf_k", binding.sf_k),
                ("sf_v", binding.sf_v),
                ("amax_o", binding.amax_o),
                ("descale_q", binding.descale_q),
                ("descale_k", binding.descale_k),
                ("descale_v", binding.descale_v),
                ("scale_o", binding.scale_o),
                # Block-scaled O: the SF_O output travels with the quantized operands.
                ("sf_o", binding.sf_o),
            )
            if t is not None
        }
        if (facts.is_mxfp8 or facts.is_fp8)
        else {}
    )
    stream_handles: dict = {}  # raw CUstream int -> driver.CUstream, per stream this plan has seen

    def _need(resolved, t, label):
        # A feature the graph requests whose buffer is absent from the variant
        # pack is an error here — the lowering would otherwise fail later and
        # worse (a silently-dense mask, a null-deref in the kernel host code).
        buf = resolved.get(id(t))
        if buf is None:
            raise ValueError(f"cudnn.sdpa: {label} requested but no buffer was provided")
        return buf

    def _execute_resolved(resolved, workspace=None, stream=None):
        q_buf, k_buf, v_buf, o_buf = resolved[id_q], resolved[id_k], resolved[id_v], resolved[id_o]
        if not facts.thd:
            q_buf, k_buf, v_buf, o_buf = _ir_view(q_buf, *lay_q), _ir_view(k_buf, *lay_k), _ir_view(v_buf, *lay_v), _ir_view(o_buf, *lay_o)
        elif facts.has_paged_kv:
            # THD Q/O stay packed; the pools are ordinary dense tensors.
            k_buf, v_buf = _ir_view(k_buf, *lay_k), _ir_view(v_buf, *lay_v)
        # Scratch comes from the CALLER's workspace (never allocated here): the
        # adapter validates it against its scratch_workspace_bytes() and takes
        # fixed offsets into it.
        if total_workspace_bytes and workspace is None:
            raise ValueError(
                f"cudnn.sdpa: {spec.name} requires a {total_workspace_bytes}-byte workspace but execute() received none; "
                "allocate graph.get_workspace_size() bytes (uint8, on the graph's device) and pass the buffer to execute()"
            )
        api_workspace = workspace
        seq_kv_buf = _need(resolved, seq_kv_src, "padding mask (seq_len_kv / cu_seq_len_kv)") if seq_kv_src is not None else None
        seq_q_buf = _need(resolved, seq_q_src, "per-batch query lengths (seq_len_q / cu_seq_len_q)") if seq_q_src is not None else None
        bias_buf = _need(resolved, bias_src, "bias") if bias_src is not None else None
        if stream is None:
            cu_stream = None
        else:
            # Stream from the execute-time handle (raw CUstream int, the
            # ExecutionContext's stream); None keeps the default stream.
            cu_stream = stream_handles.get(stream)
            if cu_stream is None:
                cu_stream = stream_handles[stream] = _cuda_driver().CUstream(stream)
        execute_kwargs = dict(
            q_tensor=q_buf,
            k_tensor=k_buf,
            v_tensor=v_buf,
            o_tensor=o_buf,
            # Stats-less graphs bind lse_tensor=None: every adapter here is
            # lse_optional (the kernel compiles the LSE store out) — no dummy.
            lse_tensor=resolved.get(id_stats) if id_stats is not None else None,
            scale_softmax=None if facts.attn_scale_prefolded else facts.scale,  # the fold: Q carries the scale
            sinks=_need(resolved, sink_src, "sink_token") if sink_src is not None else None,
            seq_kv_lens=seq_kv_buf,
            seq_q_lens=seq_q_buf if forward_seq_q else None,
            current_stream=cu_stream,
        )
        if facts.has_paged_kv:
            # (B, 1, max_pages, 1) int32 tables bound as the kernel's flat
            # (B, max_pages) views — a view, never a copy (Rule 1); the same
            # buffer may back both.
            bt_k = resolved.get(id(binding.paged_k_table))
            bt_v = resolved.get(id(binding.paged_v_table))
            if bt_k is None or bt_v is None:
                raise ValueError("cudnn.sdpa: paged_attention_k_table / paged_attention_v_table requested but no buffer was provided")
            execute_kwargs.update(
                block_table=_table_view(bt_k, binding.paged_k_table, facts.b),
                block_table_v=_table_view(bt_v, binding.paged_v_table, facts.b),
            )
        if thd_decode_leg:
            # The ragged offsets are bound operands of this leg (device reads).
            execute_kwargs.update(
                ragged_q=_need(resolved, binding.ragged_q, "Q ragged offsets"),
                ragged_o=_need(resolved, binding.ragged_o, "O ragged offsets"),
                ragged_lse=_need(resolved, binding.ragged_stats, "Stats ragged offsets") if binding.ragged_stats is not None else None,
            )
        for name, tid in quant_ids.items():
            execute_kwargs[name] = resolved.get(tid)
        if facts.o_block_scale and execute_kwargs.get("sf_o") is None:
            raise ValueError("cudnn.sdpa: the graph requests the sf_o output but no buffer was provided for it")
        if forward_bias:
            # The bias operand (mismatch admitted it for this row): the SM80 adapter takes the buffer as handed, the
            # DSL adapters the IR-shaped view (dims / strides declared on the graph tensor).
            execute_kwargs["bias_tensor"] = _ir_view(bias_buf, *lay_bias) if lay_bias is not None else bias_buf
        if kv_mask_src is not None:
            execute_kwargs["kv_mask_tensor"] = _ir_view(_need(resolved, kv_mask_src, "kv_mask"), *lay_kv_mask)
        if gate_src is not None:
            # Epilogue gate G, reinterpreted through its IR dim/stride like Q/K/V/O.
            execute_kwargs["gate"] = _ir_view(_need(resolved, gate_src, "epilogue gate"), *lay_gate)
        for name, t, lay in score_srcs:
            execute_kwargs[name] = _ir_view(_need(resolved, t, f"{name[:-7]} output"), *lay)
        if api_scratch_bytes:
            execute_kwargs["workspace"] = api_workspace
        api.execute(**execute_kwargs)
        return None

    def _execute(variant_pack, workspace=None, stream=None):
        # Adapter copies, scratch initialization and allocator lifetime must
        # follow the same stream as the kernels launched through the handle.
        with _torch_stream_context(stream, api.q_desc.device):
            return _execute_resolved(ga.resolve_variant_pack(variant_pack, binding), workspace, stream)

    def _execute_by_tensor(resolved, workspace=None, stream=None):
        """The plan's lane: ``{id(ir_tensor): buffer}``, resolved by the plan from its uid table."""
        with _torch_stream_context(stream, api.q_desc.device):
            return _execute_resolved(resolved, workspace, stream)

    # Executor contract (engine._FrostSdpaFwdPlan): a non-zero workspace_bytes
    # means the plan calls _execute(variant_pack, workspace) with the caller's
    # buffer; 0 means _execute(variant_pack) and the buffer is never touched.
    # ``binding`` lets the plan key this executor's operands out of the graph's
    # variant pack (the pack covers every IO tensor of the graph).
    _execute.workspace_bytes = total_workspace_bytes
    _execute.binding = binding
    _execute.kernel_template = kernel_template
    _execute.softmax_arms = softmax_arms
    _execute.kernel_module = kernel_module
    _execute.execute_resolved = _execute_by_tensor
    _execute.prepared = None
    if getattr(api, "_sm80_spec", None) is not None:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared_sm80 import PreparedSm80Launch

        _execute.prepared = PreparedSm80Launch(api._sm80_spec, binding)
        _execute.default_stream = lambda: api._get_default_stream(None)
    if _prepared_decline_reason(spec.capabilities, facts, getattr(api, "split_kv", 1)) is None:
        # The prepared launch (cudnn.sdpa.fwd.prepared): the plan binds the normalized VariantPack itself.
        # THD: the f16 ragged plan without a gate. Dense: the f16 plan whose declared Q/K/V/O layouts TMA
        # binds zero-copy. Split plans bind their partial workspace and final strided O in the same prepared call.
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import PreparedDenseLaunch, PreparedThdLaunch

        if facts.thd and getattr(api, "_thd_spec", None) is not None:
            _execute.prepared = PreparedThdLaunch(api._thd_spec, binding, stats_stride_override=facts.shape_overrides)
        elif getattr(api, "_dense_spec", None) is not None:
            # Dense plans, and the decode tile's ragged-Q leg (a dense split
            # launch whose Q / O / Stats rows come from the bound ragged offsets).
            _execute.prepared = PreparedDenseLaunch(
                api._dense_spec, binding, seq_kv_src=seq_kv_src, seq_q_src=seq_q_src if seq_q_lens_present else None, gate_src=gate_src
            )
        if _execute.prepared is not None:
            _execute.default_stream = lambda: api._get_default_stream(None)
    if facts.shape_overrides and _execute.prepared is None:
        raise NotImplementedError("this plan has no prepared shape/stride override executor")
    return _execute


def engine_name(
    phase: str = "prefill",
    arch: str = "sm100",
    mxfp8: bool = False,
    fp8: bool = False,
) -> str:
    """The registered engine name for a coverage cell (test/user convenience).

    One engine per arch x dtype family — head dims are a lowering concern
    (kernel-flavor pick), not part of the engine identity."""

    suffix = "_mxfp8" if mxfp8 else "_fp8" if fp8 else ""
    return f"sdpa_fwd_{phase}_{arch}" + suffix


# ORDER MATTERS: this is the PREFERENCE order — ``engine.FrostSdpaFwdEngines()``
# wraps the specs in it, so the plans they propose reach graph.plans in this
# order and the build walk tries them top-down. One engine per arch x dtype
# family: kernel-FLAVOR choice (which head-dim tile) happens inside the
# lowering (api_dsl._pick_flavor, smallest covering flavor — the tightest
# tile, least padded work), so at most one row of a family is eligible per
# device and the order only breaks ties across families.
# Engine IDS do NOT follow this order — they are pinned per name in
# engines/manifest.py and never move.
def _sm120_fp8_spec() -> EngineSpec:
    """SM120 per-tensor FP8 engine (E4M3/E5M2 in + scalar descales, FP16/BF16/FP8 out).

    P quantization is stateless: a baked 2^4 cast bias preserves fp8 precision
    and is canceled in the epilogue. Graph Scale_S/Descale_S operands are
    accepted and ignored; they do not change the mathematical attention gain.
    Q/K/V descales and Scale_O remain runtime device scalars.

    Same mma.sync architecture as the f16 SM120 cell with the MMA lowered to
    m16n8k32 e4m3; ``descale_q*descale_k`` folds into the softmax scale and
    ``descale_v*scale_o`` into an epilogue scalar, so beyond those the kernel
    adds only the Amax_O atomic (no Amax_S — the shared mismatch rule declines
    graphs that declare one). E4M3/E5M2 QKV supported; O may
    be FP16, BF16, or FP8 (either flavor — a direct quantizing store applies
    ``scale_o`` before the cast, and Amax_O stays the pre-cast fp32 amax).
    General head TILES are multiples of 32 up to 256 with the QK^T and P@V sides
    independent; actual head dims may be any multiple of 16 up to the tile
    (TMA 16-byte global-stride rule at 1 byte/elem) via TMA zero-padding.
    The D512 flavor serves both head dimensions independently in (256, 512],
    in multiples of 16, with a 64x64 CTA tile. Both templates address
    declared BSHD strides natively (THD views of kv-interleaved records
    included, like the f16 row); the dense path normalizes non-BSHD orders
    to compact storage.
    Attention sinks fold into the softmax denominator: the sink is a virtual
    column with no V row — it rescales O and enters the LSE. THD (ragged) is
    served with token- or head-major Stats.
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import GENERAL_HEAD_TILE_MAX, FP8_GENERAL_HEAD_TILES

    return EngineSpec(
        name="sdpa_fwd_prefill_sm120_fp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            d_shapes=frozenset((tq, tv) for tq in FP8_GENERAL_HEAD_TILES for tv in FP8_GENERAL_HEAD_TILES) | {D512_FLAVOR},
            d_envelope_floors=((D512_FLAVOR, GENERAL_HEAD_TILE_MAX),),
            d_pad_multiple=16,  # TMA 16-byte global-stride rule at 1 byte/elem
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset(
                {cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2, cudnn.data_type.FP4_E2M1}
            ),
            # Block-scaled O epilogues (d_v = 128; the adapter declines other head dims).
            o_block_scales=frozenset({0, 16, 32}),
            is_fp8=True,
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            thd=True,
            padded_stats=True,
            skv_tile=0,
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            # The kernel's inline chunking (same shape as the f16 sibling) + the
            # shared split_combine pass. Under a split the kernel stands its amax
            # down and the combine reports the amax of the RECOMBINED O.
            split_kv_supported=True,
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
            pack_gqa_requires_divisible=False,
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


def _sm120_mxfp8_spec() -> EngineSpec:
    """SM120 block-scaled MXFP8 engine (E4M3/E5M2 + per-32-block E8M0 SF, FP16/BF16/FP8 out).

    General head TILES are multiples of 32 up to 256 with the QK^T and P@V sides
    independent; the D512 flavor serves both head dimensions independently in
    (256, 512], in multiples of 16, with a 64x64 CTA tile (the block-scaled O
    is d128 only).
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm120 import GENERAL_HEAD_TILE_MAX, FP8_GENERAL_HEAD_TILES

    return EngineSpec(
        name="sdpa_fwd_prefill_sm120_mxfp8",
        capabilities=Capabilities(
            sm_lo=_BLACKWELL_GEFORCE[0],
            sm_hi=_BLACKWELL_GEFORCE[1],
            phase="prefill",
            d_shapes=frozenset((tq, tv) for tq in FP8_GENERAL_HEAD_TILES for tv in FP8_GENERAL_HEAD_TILES) | {D512_FLAVOR},
            d_envelope_floors=((D512_FLAVOR, GENERAL_HEAD_TILE_MAX),),
            d_pad_multiple=16,  # TMA 16-byte global-stride rule at 1 byte/elem
            dtypes=frozenset({cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            out_dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16, cudnn.data_type.FP8_E4M3, cudnn.data_type.FP8_E5M2}),
            o_block_scales=frozenset({0, 32}),
            is_mxfp8=True,
            causal=True,
            bottom_right=True,
            swa=True,
            right_band_widening=True,
            padded=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            padded_stats=True,
            thd=True,
            skv_tile=0,
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            split_kv_supported=True,
            tile_ms=frozenset({64, 128}),
            tile_ns=frozenset({64, 128}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
            pack_gqa_requires_divisible=False,
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM120),
    )


def _sm90_spec() -> EngineSpec:
    """D512 tile with TMA padding; the adapter declines unserved native declarations at build."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm90 import D_ALIGN, D_TILE, TILE_M, TILE_N

    return EngineSpec(
        name="sdpa_fwd_prefill_sm90",
        capabilities=Capabilities(
            zero_scale=True,
            sm_lo=90,
            sm_hi=90,
            phase="prefill",
            d_shapes=frozenset({(D_TILE, D_TILE)}),
            # The envelope floor is a ROW decision, not a template one:
            # config_sm90.head_dims_mismatch still serves every multiple of 8 in
            # (0, 512] for a direct adapter caller. Unfloored, the sole cc-9.0 row
            # would take a d64 graph at 8x zero-padding -- and lead the preference
            # order while doing it. 256 = at most 2x padding, matching the SM120
            # d512 flavor. d <= 256 f16 Hopper graphs get no FROST plan and fall
            # back to the backend; an exact (512, 512) hit is unaffected.
            d_envelope_floors=(((D_TILE, D_TILE), 256),),
            d_pad_multiple=D_ALIGN,
            dtypes=frozenset({cudnn.data_type.HALF, cudnn.data_type.BFLOAT16}),
            causal=True,
            bottom_right=True,
            right_band_widening=True,
            swa=True,
            padded=True,
            padded_stats=True,
            sink=True,
            stats=True,
            stats_log2=True,
            lse_optional=True,
            decode=True,  # stated, not inherited: S_q == 1 is served (the sink decode tests pin it)
            thd=True,
            cu_seq_len=True,
            skv_tile=0,
            layouts=frozenset({"bshd", "dense_flex"}),
            sched_policies=frozenset({SCHED_NATURAL, SCHED_LPT, SCHED_LPT_L2}),
            tile_ms=frozenset({TILE_M}),
            tile_ns=frozenset({TILE_N}),
            cgas=frozenset({1}),
            pack_gqas=frozenset({False, True}),
            softmax_precisions=frozenset({cudnn.data_type.FLOAT}),
        ),
        lower=partial(lower_dsl_prefill, api_type=_SM90),
    )


ENGINE_SPECS = (
    _sm100_spec(),
    _sm100_mxfp8_spec(),
    _sm100_fp8_spec(),
    _sm100_fp8_spec(arch="sm107"),
    _sm107_spec(),
    _sm107_mxfp8_spec(),
    _sm120_spec(),
    _sm120_fp8_spec(),
    _sm120_mxfp8_spec(),
    _sm80_spec(),
    _sm90_spec(),
)

__all__ = ["Capabilities", "EngineSpec", "ENGINE_SPECS", "SdpaFwdKnobs", "analyze_for", "build", "engine_name", "mismatch"]
