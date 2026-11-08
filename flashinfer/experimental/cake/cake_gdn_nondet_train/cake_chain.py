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

Host plan and autograd chain of the non-deterministic GDN chunked training
(head dimension 128, chunk 64): the chunk and segment tables, the scan
configuration and segment planner, the workspace pool, the forward and backward
launch order, the adjoint-scan and finalize selection and the autograd wrapper
:func:`cake_gdn_nondet` with FLA's ``chunk_gated_delta_rule`` signature (no
context parallelism).

The forward runs the prep (``prep_tma``, the q / k tiles staged by TMA and one
CTA per (chunk, key head), unless its dispatch rule :func:`use_prep_tma` keeps
the plain prep; both give the same bits), the tcgen05 WY stage
(:func:`wy_stage`: ``wy_tc2``, five CTAs per SM, on the architectures of
``WY_TC2_ARCHS``, else ``wy_tc``; the same bits) and the fused tcgen05 scan
and output (``fwd_ho``; a split plan first runs the pre-scan and the stitch
for the segment start states; on the architectures of ``FWD_HO_VH2_ARCHS`` a
scan of one launch whose CTAs fill at most half of the SMs runs as
``fwd_ho_vh2``, two CTAs per segment and value head, one value half each,
:func:`fwd_ho_vh2_engaged`, where the record is the split program,
:func:`fwd_ho_vh2_splits`; the same bits).  The backward runs ``dv_local`` and
the register scan of the adjoint state (with the pre-scan and the stitch on a
split plan), or on a serial plan the tcgen05 adjoint scan with ``dv_local``
fused (``dhu_tc2``) when its selection rule (:func:`dhu_tc_pricing`) prices it
at most ``DHU_TC_GAIN`` of the register path; a split plan at ``SDT_CONFIG``
on a host-built table runs its pass 2 on ``dhu_tc2`` from the stitched segment
states when its rule (:func:`sdt_pricing`) prices that pass 2 below the
register one, and then launches no ``dv_local``: its pre-scan
(``dhu_pre_dvl_r64k1``) computes the ``dv_local`` tiles of the chunks it scans.
On the architectures of ``DVL32_ARCHS`` the tcgen05 adjoint scan adds the
FP32 ``dv_local`` values to ``dv2`` (``dhu_tc2_dvl32``: one rounding of ``dv2``
instead of two, :func:`dhu_tc2_stage`); on those of ``PRE_DVL32_ARCHS`` the
split pre-scan does too (``dhu_pre_dvl32_r64k1``, :func:`sdt_prescan`).  Then the tail :func:`tail_form`
dispatches: for a value-head group of two or three the cluster tail of
``CLUSTER_TAILS``, whose CTAs exchange the group's FP32 ``dq`` / ``dk``
partials over distributed shared memory and round ``dq`` / ``dk`` to BF16 once
(``tail_g2_f32x`` / ``tail_g3_f32x``: the deterministic package's cluster
kernels ``tail_g2`` / ``tail_g3`` with FP32 partials; they write ``dq``,
``dk`` and ``dv`` through token-row strides, so a call on the head views of one
packed row writes one packed gradient row, :class:`CakeGDNNondetPackedFunction`);
for other groups the package's tail, one tcgen05 kernel per (chunk, value head)
whose value heads write their FP32 ``dq`` / ``dk`` partials to a workspace slot
each, the last arriving CTA of a (chunk, key head) summing the slots in
value-head order, running the finalize and rounding ``dq`` / ``dk`` to BF16
once (:func:`tail_segments` bounds the workspace), and below the architecture's
``TAIL_MIN_CTAS`` (chunk, value head) CTAs the deterministic package's
``dqkwg``, WY pullback and finalize ("unfused").
Every launch is fixed-order.  A host-built table lists the sequences longest first
(:func:`lpt_sequence_order`; every chunk keeps its token offset, the state rows
are gathered into that order and the final state and its gradient scattered
back) and ``fwd_ho`` walks the forward segments longest first
(:func:`lpt_walk`): each CTA computes what it computes in the caller's order.
Where a serial scan leaves its last round partly idle, a host-built table runs
it as two contiguous sequence groups (:func:`sequence_groups`), one on a
high-priority side stream forked from the caller's stream and joined back
before the stage returns: the forward's ``wy_tc`` and ``fwd_ho`` of the partial
group overlap the full rounds of the other, and the adjoint ``dhu_tc2`` of the
partial group overlaps the fused tail of the other (:func:`adjoint_r_form`:
beside a value-head group of two's tail, and beside a group of three's cluster
tail on the architectures of ``TWO_STREAM_SERIAL_G3_ARCHS``).  On an
architecture of ``SPLIT_R_FORM`` the ``dhu_tc2`` pass 2 of a split adjoint plan
beside a group of three's cluster tail runs the same way over its segments
(:func:`split_r_form`).  On an architecture of ``ADJOINT_SPLIT_ARCHS`` the
adjoint walk of a host-built table is either the serial table or the exact
one-wave split (:func:`bwd_piece_table`, :func:`adjoint_split_pricing`, both
priced at their full cost): the last piece of every sequence walks beside the
affine maps of its middle pieces (``dhu_tc2_dvl32_modes`` / ``dhu_tc2_dvl32_gm``,
the maps on a side stream, or after them on the caller's stream where the
host's fork and join would cost more than the overlap buys, or with
``streams=1``), the stitch composes the other pieces' end adjoints and a
second wave walks them, so neither the plan nor its cost depends on the gates.

Every launch is a record of :mod:`.cake_jit` selected by its stage and the
device's architecture (:func:`.cake_backend.launch_stage`); the stages a plan
may use are the ones the registry delivers for that architecture.  The
planner's per-configuration facts (the register scans' shared-memory figure,
the chunks one pre-scan CTA covers) are the records' ``scan`` constants; the
``dhu_tc2`` selection constants (serial plans and the split pass 2) and the
``prep_tma`` dispatch window are those
records' ``select`` tables.

Every launch is fixed-order today, so two runs of one call on one device give
the same bits; the non-deterministic contract does not promise run-to-run bits.
The results agree with the deterministic package
(``flashinfer.experimental.cake.cake_gdn_det_train``) to its accuracy budgets, not
bitwise: the tails of a value-head group of two or more sum the group's FP32
``dq`` / ``dk`` partials before one BF16 rounding, the tcgen05 adjoint adds the
FP32 ``dv_local`` to ``dv2`` before its one rounding, and the adjoint table
priced for ``dhu_tc2`` splits the sequences at other segment boundaries.
The plan pins that tests and tuning use are explicit keyword arguments of
:func:`cake_gdn_nondet` (``scan_config``, ``segment_chunks``,
``finalize``, ``warmup_log2``, ``dhu_scan``, ``prep``, ``lpt``, ``streams``,
``tail``);
nothing here reads an environment variable.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, replace
from functools import cache, lru_cache
from typing import Any, cast

import torch

from . import cake_backend, cake_jit

__all__ = [
    "CHUNK",
    "FINALIZE_STAGES",
    "FULL",
    "HEAD_DIM",
    "PRESCAN_CONFIGS",
    "REG_SCAN_CONFIGS",
    "R_FORM_GROUPS",
    "TWO_STREAM_SERIAL_G3_ARCHS",
    "CLUSTER_TAILS",
    "TAIL_DEFAULT",
    "TAIL_FORMS",
    "TAIL_MIN_CTAS",
    "TAIL_TILE_FLOATS",
    "TAIL_WORKSPACE_TILES",
    "ADJOINT_SPLIT_ARCHS",
    "ADJOINT_SPLIT_COSTS",
    "ADJOINT_CHAIN_STAGES",
    "CHAIN_VSPLIT",
    "AdjointSplit",
    "adjoint_split_lists",
    "adjoint_split_pricing",
    "adjoint_split_tables",
    "bwd_piece_table",
    "CHAIN_PLAN",
    "FWD_HO_VH2_ARCHS",
    "FWD_HO_VH2_STEP",
    "SPLIT_R_FORM",
    "TC_PLAN",
    "TC_PLAN_GAIN",
    "TC_PLAN_MIN_SAVING",
    "TC_PLAN_SELECT",
    "TC_PLAN_STEP",
    "CakeGDNNondetFunction",
    "CakeGDNNondetPackedFunction",
    "DVL32_ARCHS",
    "F32X_TAILS",
    "GROUP_TAILS",
    "PACKED_ROW",
    "PRE_DVL32_ARCHS",
    "ChainOptions",
    "ChunkMeta",
    "ScanPlan",
    "SegmentTable",
    "StageSet",
    "StreamGroups",
    "adjoint_r_form",
    "build_chunk_meta",
    "build_chunk_meta_from_cu_seqlens",
    "cake_gdn_nondet",
    "cake_gdn_nondet_train_backward",
    "cake_gdn_nondet_train_forward",
    "choose_scan_plan",
    "chain_plan_pricing",
    "dhu_tc2_stage",
    "dhu_tc_pricing",
    "finalize_stage",
    "fwd_ho_vh2_engaged",
    "fwd_ho_vh2_splits",
    "host_plans",
    "lpt_sequence_order",
    "lpt_walk",
    "make_options",
    "scan_plans",
    "sdt_prescan",
    "sdt_pricing",
    "split_r_form",
    "tc_plan_pricing",
    "tc_plan_thresholds",
    "sequence_groups",
    "stage_set",
    "tail_form",
    "tail_min_ctas",
    "tail_segments",
    "two_stream_split",
    "use_prep_tma",
    "wy_stage",
]

CHUNK: int = int(cake_jit.CHUNK)
HEAD_DIM: int = int(cake_jit.HEAD_DIM)

# Register-scan configurations ``(value block, key split)`` in the planner's preference order, and the ones with a
# segment-parallel route (pre-scan + split scan for the forward, pre-scan + stitch for the adjoint).
REG_SCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (32, 1), (64, 1))
PRESCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (64, 1))
SCAN_KINDS = ("fwd_h", "dhu")
STITCH_VSPLIT = 4  # CTAs per (sequence, value head) of the stitch launch
CHAIN_VSPLIT = 8  # CTAs per (sequence, value head) of the piece chain launch

# The tag of this package's forward tape: the backward reads only a tape that carries it (the deterministic package's
# tape has no ``"mode"`` key, so it is refused).
_TAPE_MODE = "nondet"

# Kernel parameter names of every stage in call-site order; a delivered record whose parameters are these names in
# another order is launched with the values reordered.
_PREP_ARGS = ("q", "k", "g", "beta", "beta32_out", "qn", "kn", "rstd_q", "rstd_k", "g_cs", "chunk_start", "chunk_len",
              "num_heads", "num_v_heads", "normalize_qk")  # fmt: skip
# prep_tma takes q / k (dhu_tc2 qn / kn / w / do) as the tensors its binding encodes the tensor maps from
_PREP_TMA_ARGS = ("q_tma", "k_tma") + _PREP_ARGS[2:]
_WY_ARGS = ("kn", "v", "g_cs", "beta", "A_out", "w_out", "u_out", "chunk_start", "chunk_len", "num_heads",
            "num_v_heads", "recompute")  # fmt: skip
_FWD_H_ARGS = ("kn", "w", "u", "g_cs", "h0", "h_out", "v_new", "final_state", "seg_state", "chunk_start",
               "chunk_len", "seg_chunk_start", "seg_seq", "seq_chunk_start", "num_heads", "num_v_heads",
               "use_initial_state", "store_final_state")  # fmt: skip
_FWD_H_SPLIT_ARGS = _FWD_H_ARGS + ("seg_flag",)
_FWD_H_PRE_ARGS = _FWD_H_SPLIT_ARGS + ("warmup_thresh_log2",)
_STITCH_ARGS = ("seg_state", "init_state", "state_out", "seq_seg_start", "seg_flag", "num_v_heads", "use_init",
                "reverse")  # fmt: skip
# the piece chain: the stitch's arguments without ``reverse`` (the chain walks backward)
_CHAIN_ARGS = _STITCH_ARGS[:-1]
_FWD_HO_ARGS = ("qn_tma", "kn_tma", "w_tma", "u_tma", "g_cs", "h0", "seg_start", "h_tma", "v_new_tma", "v_new",
                "o_tma", "o_out", "final_state", "chunk_start", "chunk_len", "seg_chunk_start", "seg_seq",
                "seq_chunk_start", "walk", "num_heads", "num_v_heads", "use_initial_state", "store_final_state",
                "use_seg_start", "scale")  # fmt: skip
_DV_LOCAL_ARGS = ("qn", "kn", "do", "g_cs", "dv_local", "chunk_start", "chunk_len", "num_heads", "num_v_heads",
                  "scale")  # fmt: skip
_DHU_ARGS = ("qn", "kn", "w", "do", "dv_local", "g_cs", "dht", "dh_out", "dv2", "dh0", "seg_state", "chunk_start",
             "chunk_len", "seg_chunk_start", "seg_seq", "seq_chunk_start", "num_heads", "num_v_heads",
             "use_final_state_grad", "store_initial_state_grad", "scale")  # fmt: skip
_DHU_PRE_ARGS = _DHU_ARGS + ("seg_flag", "warmup_thresh_log2")
# the (64, 1) pre-scan that computes the dv_local tiles of its chunks in the CTA
_DHU_PRE_DVL_ARGS = tuple(a for a in _DHU_PRE_ARGS if a != "dv_local")
_DHU_TC2_ARGS = ("qn_tma", "kn_tma", "w_tma", "do_tma") + tuple(
    a for a in _DHU_ARGS[4:] if a != "dv_local"
)
# the dhu_tc2_dvl32 forms of the one-wave adjoint split: the scan's arguments, then the entry table (segment index and
# mode bits per CTA row)
_DHU_TC2_FORM_ARGS = (*_DHU_TC2_ARGS, "seg_index", "seg_mode")
_DQKWG_ARGS = ("qn", "kn", "v_new", "do", "dv2", "h", "dh", "g_cs", "dq_out", "dk_out", "dw_out", "dg_out",
               "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale")  # fmt: skip
_WY_BWD_ARGS = ("kn", "v", "A", "dw", "du", "g_cs", "beta", "dk2_out", "dv_out", "dbeta_out", "dg2_out",
                "chunk_start", "chunk_len", "num_heads", "num_v_heads")  # fmt: skip
_FINALIZE_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "dq_hv", "dk_hv", "dk2", "dg1", "dg2", "dq_out", "dk_out", "dg_out",
                  "chunk_start", "chunk_len", "num_heads", "num_v_heads", "normalize_qk")  # fmt: skip
_TAIL_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "v_new", "do", "dv2", "h", "dh", "g_cs", "v", "A", "beta", "dq_out",
              "dk_out", "dg_out", "dv_out", "dbeta_out", "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale",
              "normalize_qk")  # fmt: skip
# the tail of any value-head group: the operands of tail_g2, then the workspace, its arrival counters and its tiles
_TAIL_WS_ARGS = (*_TAIL_ARGS[:18], "ws", "ws_count", *_TAIL_ARGS[18:], "ws_tiles")
# the cluster tails with FP32 partials: the operands of tail_g2 / tail_g3, then the token-row strides (elements) of v,
# dq, dk and dv
_TAIL_F32X_ARGS = (
    *_TAIL_ARGS,
    "v_row_stride",
    "dq_row_stride",
    "dk_row_stride",
    "dv_row_stride",
)
_META_ARGS = ("cu_seqlens", "chunk_start", "chunk_len", "seq_chunk_start", "fwd_seg_chunk_start", "fwd_seg_seq",
              "fwd_seq_seg_start", "bwd_seg_chunk_start", "bwd_seg_seq", "bwd_seq_seg_start", "num_seqs",
              "num_chunks_max", "total_tokens", "fwd_segment_len", "bwd_segment_len", "fwd_num_segments_max",
              "bwd_num_segments_max")  # fmt: skip
LAUNCH_ARGS: dict[str, tuple[str, ...]] = {
    "meta": _META_ARGS,
    "prep": _PREP_ARGS,
    "prep_bf16beta": _PREP_ARGS,
    "prep_tma": _PREP_TMA_ARGS,
    "prep_tma_bf16beta": _PREP_TMA_ARGS,
    "wy_tc": _WY_ARGS,
    "wy_tc2": _WY_ARGS,
    "fwd_ho": _FWD_HO_ARGS,
    "fwd_ho_vh2": _FWD_HO_ARGS,
    "stitch": _STITCH_ARGS,
    "piece_chain": _CHAIN_ARGS,
    "dv_local": _DV_LOCAL_ARGS,
    "dhu_tc2": _DHU_TC2_ARGS,
    "dhu_tc2_dvl32": _DHU_TC2_ARGS,
    "dhu_tc2_dvl32_modes": _DHU_TC2_FORM_ARGS,
    "dhu_tc2_dvl32_gm": _DHU_TC2_FORM_ARGS,
    "dqkwg": _DQKWG_ARGS,
    "wy_bwd": _WY_BWD_ARGS,
    "wy_bwd_bf16": _WY_BWD_ARGS,
    "finalize": _FINALIZE_ARGS,
    "finalize_mlp": _FINALIZE_ARGS,
    "tail_g2": _TAIL_ARGS,
    "tail_g2_bf16": _TAIL_ARGS,
    "tail_g3": _TAIL_ARGS,
    "tail_g3_bf16": _TAIL_ARGS,
    "tail_g2_f32x": _TAIL_F32X_ARGS,
    "tail_g2_f32x_bf16": _TAIL_F32X_ARGS,
    "tail_g3_f32x": _TAIL_F32X_ARGS,
    "tail_g3_f32x_bf16": _TAIL_F32X_ARGS,
    "dhu_pre_dvl_r64k1": _DHU_PRE_DVL_ARGS,
    "dhu_pre_dvl32_r64k1": _DHU_PRE_DVL_ARGS,
    "tail": _TAIL_WS_ARGS,
    "tail_bf16": _TAIL_WS_ARGS,
    **{f"fwd_h_r{vb}k{ks}": _FWD_H_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"fwd_h_r{vb}k{ks}_split": _FWD_H_SPLIT_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"fwd_h_pre_r{vb}k{ks}": _FWD_H_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
    **{f"dhu_r{vb}k{ks}": _DHU_ARGS for vb, ks in REG_SCAN_CONFIGS},
    **{f"dhu_pre_r{vb}k{ks}": _DHU_PRE_ARGS for vb, ks in PRESCAN_CONFIGS},
}


# --------------------------------------------------------------------------- #
# the planner's per-configuration facts, read from the registry's scan records
# --------------------------------------------------------------------------- #
def _scan_facts() -> tuple[dict[tuple[str, int, int], int], int]:
    smem: dict[tuple[str, int, int], int] = {}
    chunks: set[int] = set()
    for name, found in cake_jit.MODULES.items():
        stage = found["stage"]
        scan = found.get("scan")
        for kind in SCAN_KINDS:
            for vb, ks in REG_SCAN_CONFIGS:
                if stage in (
                    f"{kind}_r{vb}k{ks}",
                    f"{kind}_pre_r{vb}k{ks}",
                    f"fwd_h_r{vb}k{ks}_split",
                ):
                    if scan is None or (scan.get("v_block"), scan.get("k_split")) != (
                        vb,
                        ks,
                    ):
                        raise ValueError(
                            f"record {name}: scan constants {scan} do not match stage {stage}"
                        )
        if scan is None:
            continue
        key = (stage.split("_r")[0], scan["v_block"], scan["k_split"])
        if "select_smem_bytes" in scan:
            if smem.setdefault(key, int(scan["select_smem_bytes"])) != int(
                scan["select_smem_bytes"]
            ):
                raise ValueError(f"records of {stage} disagree on select_smem_bytes")
        if "max_segment_chunks" in scan:
            chunks.add(int(scan["max_segment_chunks"]))
    missing = [
        (k, vb, ks)
        for k in SCAN_KINDS
        for vb, ks in REG_SCAN_CONFIGS
        if (k, vb, ks) not in smem
    ]
    if missing or len(chunks) != 1:
        raise ValueError(
            f"registry scan records incomplete: missing {missing}, pre-scan chunks {sorted(chunks)}"
        )
    return smem, chunks.pop()


REG_SCAN_SMEM, PRESCAN_BLOCK_THREADS = _scan_facts()


# --------------------------------------------------------------------------- #
# options (the plan pins)
# --------------------------------------------------------------------------- #
FINALIZE_MLP_GROUP = 3
FINALIZE_MLP_MIN_BLOCKS = 4
FINALIZE_STAGES = ("finalize", "finalize_mlp")
DHU_SCANS = ("reg", "tc")
PREP_FORMS = ("plain", "tma")
STREAM_COUNTS = (1, 2)
TAIL_FORMS = ("tail", "tail_g2", "tail_g3", "tail_g2_f32x", "tail_g3_f32x", "unfused")
# the cluster tails per value-head group (one launch per chunk table, clusters of the group's CTAs), and the ones that
# sum FP32 partials and take token-row strides
GROUP_TAILS: dict[str, int] = {
    "tail_g2": 2,
    "tail_g2_f32x": 2,
    "tail_g3": 3,
    "tail_g3_f32x": 3,
}
F32X_TAILS: tuple[str, ...] = ("tail_g2_f32x", "tail_g3_f32x")


@dataclass(frozen=True)
class ChainOptions:
    """The explicit plan pins of one call.

    ``scan_config`` forces the register-scan configuration ``(value block, key split)`` of both scans;
    ``segment_chunks`` forces the segment length in chunks (0: the serial scan; above the chunks one pre-scan
    CTA covers, 128, a plan that splits a sequence raises); ``finalize`` forces the finalize stage of the
    ``"unfused"`` tail; ``warmup_log2`` overrides the pre-scans' warm-up threshold; ``dhu_scan``
    forces the adjoint scan of a serial plan (``"reg"``: ``dv_local`` + the register scan, ``"tc"``: ``dhu_tc2``;
    ``None``: the selection rule) and the pass 2 of a split ``SDT_CONFIG`` plan on a host-built table (``"reg"``:
    ``dv_local`` + the register scan, ``"tc"``: ``dhu_tc2`` after the fused pre-scan; the same bits on ``dhu_tc2``, within
    the budgets on ``dhu_tc2_dvl32``); ``prep``
    forces the prep (``"plain"``: grid (chunks, HV); ``"tma"``: ``prep_tma``,
    grid (chunks, H), where q and k are 16-byte aligned; ``None``: the dispatch rule); ``save_state`` keeps the
    forward's ``w`` / ``u`` / ``h`` / ``v_new`` for the backward (else they are recomputed); ``host_cache`` keeps the
    chunk tables, the selection results and the per-stream workspaces between calls; ``lpt`` dispatches the
    sequences of a host-built table longest first (the same bits; ``False``: the caller's order); ``streams`` is
    ``2`` to let a host-built table's serial scans run as two sequence groups on the caller's stream and a side stream
    where their rule (:func:`two_stream_split`) fills the last round (the same bits), ``1`` to keep every launch on
    the caller's stream; ``tail`` forces the backward tail (``None``: the rule of :func:`tail_form`): the package's
    tail ``"tail"`` and the cluster tails ``"tail_g2_f32x"`` / ``"tail_g3_f32x"`` (a value-head group of two /
    three), which sum the FP32 ``dq`` / ``dk`` partials in value-head order (the same bits), or the deterministic
    package's fixed-order tails ``"tail_g2"`` / ``"tail_g3"`` and ``"unfused"`` (``dqkwg``, the WY pullback and the
    finalize), which round each value head's ``dq`` / ``dk`` partials to BF16 before their sum (``dv``, ``dg`` and
    ``dbeta`` are the same bits in every form).
    """

    scan_config: tuple[int, int] | None = None
    segment_chunks: int | None = None
    finalize: str | None = None
    warmup_log2: float | None = None
    save_state: bool = True
    host_cache: bool = True
    dhu_scan: str | None = None
    prep: str | None = None
    lpt: bool = True
    streams: int = 2
    tail: str | None = None

    @property
    def pins(self) -> tuple:
        return (self.segment_chunks, self.scan_config)


def make_options(
    *,
    scan_config: tuple[int, int] | None = None,
    segment_chunks: int | None = None,
    finalize: str | None = None,
    warmup_log2: float | None = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: str | None = None,
    prep: str | None = None,
    lpt: bool = True,
    streams: int = 2,
    tail: str | None = None,
) -> ChainOptions:
    """Validated :class:`ChainOptions`."""
    if scan_config is not None:
        if (
            not isinstance(scan_config, (tuple, list))
            or len(scan_config) != 2
            or not all(
                isinstance(x, int) and not isinstance(x, bool) and x > 0
                for x in scan_config
            )
        ):
            raise ValueError(
                f"scan_config must be a pair of positive ints; got {scan_config!r}"
            )
        scan_config = (int(scan_config[0]), int(scan_config[1]))
    if segment_chunks is not None:
        if (
            not isinstance(segment_chunks, int)
            or isinstance(segment_chunks, bool)
            or segment_chunks < 0
        ):
            raise ValueError(
                f"segment_chunks must be a non-negative int; got {segment_chunks!r}"
            )
    if (
        not isinstance(streams, int)
        or isinstance(streams, bool)
        or streams not in STREAM_COUNTS
    ):
        raise ValueError(f"streams must be one of {STREAM_COUNTS}; got {streams!r}")
    if finalize is not None and finalize not in FINALIZE_STAGES:
        raise ValueError(f"finalize must be one of {FINALIZE_STAGES}; got {finalize!r}")
    if warmup_log2 is not None:
        warmup_log2 = float(warmup_log2)
        if not math.isfinite(warmup_log2):
            raise ValueError(f"warmup_log2 must be finite; got {warmup_log2!r}")
    if dhu_scan is not None and dhu_scan not in DHU_SCANS:
        raise ValueError(
            f"dhu_scan must be one of {DHU_SCANS} or None; got {dhu_scan!r}"
        )
    if prep is not None and prep not in PREP_FORMS:
        raise ValueError(f"prep must be one of {PREP_FORMS} or None; got {prep!r}")
    if tail is not None and tail not in TAIL_FORMS:
        raise ValueError(f"tail must be one of {TAIL_FORMS} or None; got {tail!r}")
    return ChainOptions(
        scan_config,
        segment_chunks,
        finalize,
        warmup_log2,
        bool(save_state),
        bool(host_cache),
        dhu_scan,
        prep,
        bool(lpt),
        int(streams),
        tail,
    )


# the options of a call without plan pins (what ``make_options()`` returns), shared by such calls
_DEFAULT_OPTIONS = ChainOptions()


# --------------------------------------------------------------------------- #
# stage availability and launch
# --------------------------------------------------------------------------- #
@cache
def _available(stage: str, arch: str) -> bool:
    try:
        cake_backend.stage_record(stage, arch)
    except KeyError:
        return False
    return True


@cache
def _order(stage: str, arch: str) -> tuple[int, ...] | None:
    names = cake_backend.parameters(cake_backend.stage_record(stage, arch))
    wanted = LAUNCH_ARGS[stage]
    if names == wanted:
        return None
    if sorted(names) == sorted(wanted):
        return tuple(wanted.index(n) for n in names)
    raise ValueError(
        f"{stage}: record parameters {names} do not match the call site {wanted}"
    )


def _device_index(device) -> int:
    """The CUDA device index of ``device`` (the current device for a bare ``cuda``)."""
    index = torch.device(device).index
    return int(index) if index is not None else torch.cuda.current_device()


def _tensor_index(t) -> int:
    """``_device_index(t.device)`` from ``Tensor.get_device()``, without a ``torch.device`` object: a host or meta
    tensor (``-1``) maps to the current device too, so its launch reaches the binding's device checks."""
    index = t.get_device()
    return index if index >= 0 else torch.cuda.current_device()


def _launch(stage, gx, gy, gz, *values):
    """Launch ``stage`` on ``(gx, gy, gz)``; ``values`` in call-site order, the first a tensor on the device."""
    device_index = _tensor_index(values[0])
    order = _order(stage, cake_backend.device_arch(device_index))
    if order is not None:
        values = tuple(values[i] for i in order)
    cake_backend.launch_stage(stage, (gx, gy, gz), values, device_index=device_index)


def _bound_launch(stage, gx, gy, gz, *values):
    """:func:`_launch` with its device, argument order and grid resolved now: the returned call only launches (on the
    stream current when it runs).  Two launches meant to start together are bound first and fired back to back."""
    device_index = _tensor_index(values[0])
    order = _order(stage, cake_backend.device_arch(device_index))
    if order is not None:
        values = tuple(values[i] for i in order)
    grid = (gx, gy, gz)
    return lambda: cake_backend.launch_stage(
        stage, grid, values, device_index=device_index
    )


# --------------------------------------------------------------------------- #
# chunk table
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ScanPlan:
    config: tuple[int, int]
    segment_len: int


@dataclass(frozen=True)
class SegmentTable:
    seg_chunk_start: torch.Tensor
    seg_seq: torch.Tensor
    seq_seg_start: torch.Tensor
    num_segments: int
    segment_len: int
    scan_config: tuple[int, int]
    # fwd_ho's segment order (CTA group b runs walk[b]); None: the identity
    walk: torch.Tensor | None = None


@dataclass(frozen=True)
class ChunkMeta:
    """The chunk and segment tables of one layout.  ``order`` (host route): table sequence ``vs`` is the caller's
    sequence ``order[vs]``, its chunks at their original token offsets (``None``: the caller's order);
    ``order_index`` holds it on the device as ``[order, inverse]`` for the state rows.  ``fwd_groups`` /
    ``bwd_groups``: the two-stream sequence groups of the forward / adjoint scan (``None``: one stream);
    ``bwd_tc``: the adjoint table is priced for ``dhu_tc2`` (:func:`tc_plan_pricing`), which runs it; ``bwd_split``:
    the entry tables of the one-wave split of the adjoint walk over ``bwd_segments`` (:class:`AdjointSplit`), which
    the backward runs instead of a pre-scan (``None``: the other plans)."""

    chunk_start: torch.Tensor
    chunk_len: torch.Tensor
    seq_chunk_start: torch.Tensor
    num_chunks: int
    num_seqs: int
    seq_lens: tuple[int, ...] | None
    fwd_segments: SegmentTable
    bwd_segments: SegmentTable
    order: tuple[int, ...] | None = None
    order_index: torch.Tensor | None = None
    fwd_groups: StreamGroups | None = None
    bwd_groups: StreamGroups | None = None
    # the adjoint table was priced for dhu_tc2 (:func:`tc_plan_pricing`): the backward runs it on dhu_tc2
    bwd_tc: bool = False
    # the one-wave split of the adjoint walk over bwd_segments (:func:`adjoint_split_pricing`)
    bwd_split: AdjointSplit | None = None


@dataclass(frozen=True)
class StreamGroups:
    """Two sequence groups of a serial scan: group A is the table's segments ``[0, split)`` (its chunks
    ``[0, split_chunk)``), group B the rest; one is launched on a side stream (:func:`sequence_groups`)."""

    split: int
    split_chunk: int


def segment_bounds(num_chunks: int, segment_len: int) -> list[int]:
    if segment_len <= 0 or num_chunks <= segment_len:
        return [0, num_chunks]
    count = -(-num_chunks // segment_len)
    return [(s * num_chunks) // count for s in range(count + 1)]


def _segment_lists(seq_chunk_start: list[int], segment_len: int):
    seg_chunk_start = [0]
    seg_seq: list[int] = []
    seq_seg_start = [0]
    for seq in range(len(seq_chunk_start) - 1):
        c0 = seq_chunk_start[seq]
        bounds = segment_bounds(seq_chunk_start[seq + 1] - c0, segment_len)
        seg_chunk_start.extend(c0 + b for b in bounds[1:])
        seg_seq.extend([seq] * (len(bounds) - 1))
        seq_seg_start.append(len(seg_seq))
    return seg_chunk_start, seg_seq, seq_seg_start


def lpt_sequence_order(seq_lens) -> tuple[int, ...] | None:
    """The sequences longest first by chunk count, ties in the caller's order (``None`` when that is the caller's
    order)."""
    chunks = [-(-int(x) // CHUNK) for x in seq_lens]
    order = tuple(sorted(range(len(chunks)), key=lambda s: (-chunks[s], s)))
    return None if order == tuple(range(len(chunks))) else order


def lpt_walk(seg_chunk_start) -> list[int] | None:
    """The segments of a segment table longest first by chunk count, ties in table order (``None`` when that is the
    table order)."""
    lens = [
        b - a for a, b in zip(seg_chunk_start[:-1], seg_chunk_start[1:], strict=True)
    ]
    walk = sorted(range(len(lens)), key=lambda s: (-lens[s], s))
    return None if walk == list(range(len(lens))) else walk


def build_chunk_meta(
    seq_lens, device, *, scan_plans: tuple[ScanPlan, ScanPlan], lpt: bool = False
) -> ChunkMeta:
    """Host route: the chunk and segment tables of known sequence lengths, copied to ``device`` in one transfer.

    ``lpt`` dispatches longest first: the tables list the sequences in :func:`lpt_sequence_order` (each chunk keeps its
    token offset; the state rows follow ``order_index``) and ``fwd_ho`` walks the forward segments in
    :func:`lpt_walk` order.  Every CTA computes what it computes in the caller's order: the same bits."""
    lengths = [int(x) for x in seq_lens]
    offsets = [0]
    for n in lengths:
        offsets.append(offsets[-1] + n)
    order = lpt_sequence_order(lengths) if lpt else None
    starts: list[int] = []
    lens: list[int] = []
    seq_chunk_start = [0]
    for seq in order if order is not None else range(len(lengths)):
        seq_len, token = lengths[seq], offsets[seq]
        for chunk_start in range(0, seq_len, CHUNK):
            starts.append(token + chunk_start)
            lens.append(min(CHUNK, seq_len - chunk_start))
        seq_chunk_start.append(len(starts))
    parts: list[list[int]] = [starts, lens, seq_chunk_start]
    for plan in scan_plans:
        parts.extend(_segment_lists(seq_chunk_start, int(plan.segment_len)))
    walk = lpt_walk(parts[3]) if lpt else None
    extra: dict[str, int] = {}
    if walk is not None:
        extra["walk"] = len(parts)
        parts.append(walk)
    if order is not None:
        inverse = [0] * len(order)
        for vs, seq in enumerate(order):
            inverse[seq] = vs
        extra["order_index"] = len(parts)
        parts.append(list(order) + inverse)
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device(device)
    table = (
        table.pin_memory().to(device, non_blocking=True)
        if device.type == "cuda"
        else table.to(device)
    )
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    segments = [
        SegmentTable(
            seg_chunk_start=views[3 + 3 * i],
            seg_seq=views[4 + 3 * i],
            seq_seg_start=views[5 + 3 * i],
            num_segments=len(parts[4 + 3 * i]),
            segment_len=int(scan_plans[i].segment_len),
            scan_config=cast("tuple[int, int]", tuple(scan_plans[i].config)),
            walk=views[extra["walk"]] if i == 0 and "walk" in extra else None,
        )
        for i in range(2)
    ]
    order_index = views[extra["order_index"]].view(2, -1) if order is not None else None
    return ChunkMeta(views[0], views[1], views[2], len(starts), len(lengths), tuple(lengths), segments[0],
                     segments[1], order, order_index)  # fmt: skip


@lru_cache(maxsize=256)
def _host_chunk_meta(
    seq_lens: tuple[int, ...],
    plans: tuple[ScanPlan, ScanPlan],
    device_index: int,
    lpt: bool,
) -> ChunkMeta:
    return build_chunk_meta(
        seq_lens, torch.device("cuda", device_index), scan_plans=plans, lpt=lpt
    )


def segments_upper_bound(num_chunks_max: int, num_seqs: int, segment_len: int) -> int:
    return num_seqs + (
        num_chunks_max // int(segment_len) if int(segment_len) > 0 else 0
    )


def build_chunk_meta_from_cu_seqlens(
    cu_seqlens, total_tokens: int, *, scan_plans
) -> ChunkMeta:
    """Device route: the ``meta`` stage builds the tables on the GPU (padded launch extents)."""
    num_seqs = int(cu_seqlens.numel()) - 1
    num_chunks_max = total_tokens // CHUNK + num_seqs
    seg_max = [
        segments_upper_bound(num_chunks_max, num_seqs, plan.segment_len)
        for plan in scan_plans
    ]
    cu32 = (
        cu_seqlens if cu_seqlens.dtype == torch.int32 else cu_seqlens.to(torch.int32)
    ).contiguous()
    sizes = [num_chunks_max, num_chunks_max, num_seqs + 1]
    for n in seg_max:
        sizes.extend((n + 1, n, num_seqs + 1))
    table = torch.empty(sum(sizes), dtype=torch.int32, device=cu_seqlens.device)
    views = []
    offset = 0
    for n in sizes:
        views.append(table[offset : offset + n])
        offset += n
    _launch("meta", 1, 1, 1, cu32, *views, num_seqs, num_chunks_max, int(total_tokens),
            int(scan_plans[0].segment_len), int(scan_plans[1].segment_len), seg_max[0], seg_max[1])  # fmt: skip
    segments = [
        SegmentTable(
            views[3 + 3 * i],
            views[4 + 3 * i],
            views[5 + 3 * i],
            seg_max[i],
            int(scan_plans[i].segment_len),
            tuple(scan_plans[i].config),
        )  # fmt: skip
        for i in range(2)
    ]
    return ChunkMeta(
        views[0],
        views[1],
        views[2],
        num_chunks_max,
        num_seqs,
        None,
        segments[0],
        segments[1],
    )


# --------------------------------------------------------------------------- #
# scan configuration and segment planner
# --------------------------------------------------------------------------- #
@cache
def _device_props(device_index: int) -> tuple[int, int, int]:
    props = torch.cuda.get_device_properties(device_index)
    return (
        int(props.multi_processor_count),
        int(props.shared_memory_per_block_optin),
        int(props.major),
    )


def finalize_stage(
    num_chunks: int,
    num_heads: int,
    num_v_heads: int,
    sm_count: int,
    forced: str | None = None,
) -> str:
    """The finalize stage of one backward: ``forced`` when given, else the three-head MLP finalize while its grid
    stays within ``FINALIZE_MLP_MIN_BLOCKS`` CTAs per SM, else the generic finalize."""
    if num_heads <= 0 or num_v_heads % num_heads != 0:
        raise ValueError(
            f"num_v_heads={num_v_heads} must be a positive multiple of num_heads={num_heads}"
        )
    group_ok = num_v_heads == FINALIZE_MLP_GROUP * num_heads
    if forced:
        if forced not in FINALIZE_STAGES:
            raise ValueError(f"finalize={forced!r}")
        if forced == "finalize_mlp" and not group_ok:
            raise ValueError("finalize_mlp is unrolled for a value-head group of 3")
        return forced
    if group_ok and int(num_chunks) * int(num_heads) <= FINALIZE_MLP_MIN_BLOCKS * int(
        sm_count
    ):
        return "finalize_mlp"
    return "finalize"


# The package tail's workspace (a value-head group of two or more), per (device, stream of the launch): FP32 rows of
# 64 ``dq`` rows then 64 ``dk`` rows (65,536 bytes each), one slot per (chunk, value head) (written before it is
# read), and one u32 arrival counter per (chunk, key head) tile (zero when created, left zero by every launch).  A
# backward launches its chunk table in segments of at most ``TAIL_WORKSPACE_TILES`` slots (at least one chunk), so a
# workspace holds at most ``max(TAIL_WORKSPACE_TILES, num_v_heads)`` slots.
TAIL_TILE_FLOATS = 2 * CHUNK * HEAD_DIM
TAIL_WORKSPACE_TILES = 4096
# Below this many (chunk, value head) CTAs a call's backward takes the fixed-order unfused tail, per architecture and
# group class (1: a value-head group of one; 4: the larger groups without a cluster tail): where the package tail's
# fused per-CTA chain is latency-bound it measured slower in the backward.  Crossovers (package tail / unfused):
# B200 G = 1 between 2048 and 3072 CTAs (a tie at 2048), G = 4 about 2048 (the 1024-CTA rows within their spreads);
# GB300 G = 1 between 1024 and 2048 (1.02 at 1024), G = 4 between 512 and 1024 (1.02 at 512, 0.89 at 1024); R200
# (216 SMs) G = 1 between 512 and 1024 (0.87 at 1024), G = 4 at most 1024 (0.77-0.79 at 1024; 0.93 at 512, where
# forward + backward ties).
TAIL_MIN_CTAS: dict[str, dict[int, int]] = {
    "sm_100a": {1: 2048, 4: 2048},
    "sm_103a": {1: 2048, 4: 1024},
    "sm_107a": {1: 1024, 4: 1024},
}
# The cluster tail the rule takes per architecture and value-head group, and the package tail it takes elsewhere from
# its TAIL_MIN_CTAS on.  The cluster tails exchange the group's FP32 dq / dk partials over distributed shared memory,
# so dq / dk round once, as the package tail's (the same bits).  B200 kernel times (CUPTI, cold L2, four processes,
# both gate regimes): tail_g3_f32x is 1.024-1.034 x tail_g3 (which rounds each value head's partials to BF16) on the
# (2, 6) head pair at 241,817-262,144 tokens and the (16, 48) pair at 32,768 tokens, 1.034-1.051 x on the pairs
# (8, 24) .. (1, 3), where the package tail is 1.073-1.081 x tail_g3 at 3,782 or more (chunk, value head) CTAs, 1.05 x
# at the (4, 12) pair and 0.99-1.006 x at the (2, 6) and (1, 3) pairs (1,539 and 3,078 CTAs); tail_g2_f32x is
# 1.014-1.036 x tail_g2, which measured 16-19 % faster than the package tail at G = 2 (B200 and GB300).  sm_103a and
# sm_107a take the same forms until they are measured there.
CLUSTER_TAILS: dict[str, dict[int, str]] = {
    "sm_100a": {2: "tail_g2_f32x", 3: "tail_g3_f32x"},
    "sm_103a": {2: "tail_g2_f32x", 3: "tail_g3_f32x"},
    "sm_107a": {2: "tail_g2_f32x", 3: "tail_g3_f32x"},
}
TAIL_DEFAULT = "tail"
# The value-head groups whose adjoint dhu_tc2 scan runs as two sequence groups (two streams) feeding the fused tail
# (the R form) where :func:`sequence_groups` splits it.
R_FORM_GROUPS: tuple[int, ...] = (2,)
# The architectures where a serial dhu_tc2 adjoint scan feeding a cluster tail of a value-head group of three
# (``tail_g3_f32x`` or the pinned ``tail_g3``) also runs as
# the R form: group B (the partial round) on the side stream beside the tail of group A; elsewhere one launch (the
# deterministic package's rule).  Measured on the (2, 6) head pair over 64 x 4096 tokens (FP32 beta, one process per
# GPU, both arm orders), fwd+bwd against one launch: 3.3489 -> 3.3169 ms on B200 (49 + 15 sequences, sustained
# protocol) and 3.0008 -> 2.9691 ms on GB300 (25 + 39).
TWO_STREAM_SERIAL_G3_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a")


def adjoint_r_form(
    arch: str, group: int, form: str, *, serial_tc: bool, split_tc: bool
) -> bool:
    """Whether a call's adjoint scan runs as its two groups (the R form) where :func:`sequence_groups` splits it:
    the serial ``dhu_tc2`` scan (``serial_tc``) feeding a fused tail for a value-head group ``group`` of
    ``R_FORM_GROUPS``, or feeding a group of three's cluster tail on an architecture of
    ``TWO_STREAM_SERIAL_G3_ARCHS``; the ``dhu_tc2`` pass 2 of a split plan (``split_tc``) feeding a group of three's
    cluster tail (its segments are grouped on an architecture of ``SPLIT_R_FORM`` only).  ``form`` is the call's
    backward tail (:func:`tail_form`)."""
    g3 = GROUP_TAILS.get(form) == 3
    if serial_tc:
        return form != "unfused" and (
            group in R_FORM_GROUPS or (g3 and arch in TWO_STREAM_SERIAL_G3_ARCHS)
        )
    return split_tc and g3


def tail_min_ctas(arch: str, group: int) -> int:
    """The (chunk, value head) CTAs from which ``arch`` runs the package tail for a value-head group ``group`` without a
    cluster tail: ``TAIL_MIN_CTAS[arch]`` of its class (``1`` for a group of one, ``4`` for the larger groups)."""
    return TAIL_MIN_CTAS[arch][1 if group == 1 else 4]


def tail_form(
    num_heads: int,
    num_v_heads: int,
    forced: str | None = None,
    *,
    arch: str,
    num_chunks: int | None = None,
) -> str:
    """The backward tail of one call on ``arch`` over ``num_chunks`` chunks (``None``: a large call): ``forced`` when
    given (``"tail_g2"`` / ``"tail_g2_f32x"`` serve a value-head group of two, ``"tail_g3"`` / ``"tail_g3_f32x"`` a
    group of three), else the cluster tail of ``CLUSTER_TAILS[arch]`` for the call's value-head group
    ``G = num_v_heads / num_heads``; otherwise the package tail ``TAIL_DEFAULT`` from :func:`tail_min_ctas` (chunk,
    value head) CTAs on, and below them the fixed-order ``"tail_g3"`` for a group of three, ``"unfused"`` for the
    others (the fastest forms measured for the architecture, value-head group and size,
    ``benchmarks/bench_cake_gdn_nondet_train.py`` on B200 and GB300)."""
    if num_heads <= 0 or num_v_heads % num_heads != 0:
        raise ValueError(
            f"num_v_heads={num_v_heads} must be a positive multiple of num_heads={num_heads}"
        )
    group = num_v_heads // num_heads
    if forced is None:
        cluster = CLUSTER_TAILS[arch]
        if group in cluster:
            return cluster[group]
        floor = tail_min_ctas(arch, group)
        if num_chunks is not None and int(num_chunks) * num_v_heads < floor:
            return "tail_g3" if group == 3 else "unfused"
        return TAIL_DEFAULT
    if forced not in TAIL_FORMS:
        raise ValueError(f"tail={forced!r}")
    serves = GROUP_TAILS.get(forced)
    if serves is not None and group != serves:
        raise ValueError(
            f"the {forced} form serves a value-head group of {'two' if serves == 2 else 'three'}"
        )
    return forced


def tail_segments(num_chunks: int, num_heads: int, num_v_heads: int,
                  slots: int | None = None) -> tuple[tuple[tuple[int, int], ...], int]:  # fmt: skip
    """The chunk ranges ``[c0, c1)`` the package tail launches in stream order over a chunk table of ``num_chunks``
    chunks, and the (chunk, key head) tiles a launch may index (a tile is the group's ``num_v_heads // num_heads``
    workspace slots and one arrival counter): one launch and no tile at a value-head group of one, else ranges of
    ``max(1, slots // num_v_heads)`` chunks (``slots``: the workspace slots, default ``TAIL_WORKSPACE_TILES``)."""
    num_chunks, num_heads = int(num_chunks), int(num_heads)
    if num_chunks <= 0:
        return (), 0
    if int(num_v_heads) == num_heads:
        return ((0, num_chunks),), 0
    cap = TAIL_WORKSPACE_TILES if slots is None else int(slots)
    per = min(num_chunks, max(1, cap // int(num_v_heads)))
    return tuple(
        (c, min(c + per, num_chunks)) for c in range(0, num_chunks, per)
    ), per * num_heads


MIN_SEGMENT_CHUNKS = 4
SCAN_CO_RESIDENT_FACTOR = 1.60
PRESCAN_STEP_FACTOR = 1.16
PRESCAN_WARMUP_CHUNKS = 3.3
STITCH_FIXED_CHUNKS = 2.75
STITCH_COPY_CHUNKS = 2.3
SPLIT_GAIN_THRESHOLD = 0.95
MAX_SCAN_ROUNDS = 2
MAX_SCAN_WAVES_SERIAL = 4
SCAN_STEP_FACTOR = {(32, 2): 1.0, (32, 1): 1.18, (64, 1): 1.27}
WARMUP_THRESH_LOG2 = -10.0 / math.log(2.0)


def warmup_thresh_log2(options: ChainOptions) -> float:
    return (
        options.warmup_log2 if options.warmup_log2 is not None else WARMUP_THRESH_LOG2
    )


@dataclass(frozen=True)
class StageSet:
    """The scan stages a plan may use: register-scan configs per kind, and the configs whose split path (pre-scan +
    split scan for fwd_h, pre-scan + stitch for dhu) is complete.  ``FULL`` is every configuration."""

    reg: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    split: tuple[tuple[str, tuple[tuple[int, int], ...]], ...]
    label: str

    def reg_of(self, kind):
        return dict(self.reg)[kind]

    def split_of(self, kind):
        return dict(self.split)[kind]


FULL = StageSet(
    reg=(("fwd_h", REG_SCAN_CONFIGS), ("dhu", REG_SCAN_CONFIGS)),
    split=(("fwd_h", PRESCAN_CONFIGS), ("dhu", PRESCAN_CONFIGS)),
    label="all register scans, pre-scan, split, stitch",
)


@cache
def stage_set(arch: str) -> StageSet:
    """The scan stages the registry delivers for ``arch``."""
    reg = []
    split = []
    for kind in SCAN_KINDS:
        reg.append(
            (
                kind,
                tuple(
                    c
                    for c in REG_SCAN_CONFIGS
                    if _available(f"{kind}_r{c[0]}k{c[1]}", arch)
                ),
            )
        )
        ok = []
        for vb, ks in PRESCAN_CONFIGS:
            pre = _available(f"{kind}_pre_r{vb}k{ks}", arch)
            tail = (
                _available(f"fwd_h_r{vb}k{ks}_split", arch)
                if kind == "fwd_h"
                else _available("stitch", arch)
            )
            if pre and tail and _available(f"{kind}_r{vb}k{ks}", arch):
                ok.append((vb, ks))
        split.append((kind, tuple(ok)))
    return StageSet(tuple(reg), tuple(split), f"registry for {arch}")


def _residency(kind, vb, ks, smem_optin) -> int:
    threads = 32 * (vb // 16) * ks
    smem = REG_SCAN_SMEM[(kind, vb, ks)]
    return min(2048 // threads, 65536 // (256 * threads), smem_optin // smem, 4)


def choose_register_scan_config(kind, num_seqs, num_v_heads, device_index, sset: StageSet, pinned=True,
                                forced: tuple[int, int] | None = None):  # fmt: skip
    """The first configuration of ``sset`` in preference order whose grid is co-resident; else (64, 1) if available,
    else the last available configuration.  ``forced`` (honoured when ``pinned``) wins."""
    if pinned and forced is not None:
        return tuple(forced)
    sm_count, smem_optin, _major = _device_props(device_index)
    configs = sset.reg_of(kind)
    for vb, ks in REG_SCAN_CONFIGS:
        if (vb, ks) not in configs:
            continue
        ctas = num_seqs * num_v_heads * (HEAD_DIM // vb)
        resident = _residency(kind, vb, ks, smem_optin)
        if resident >= 1 and ctas <= sm_count * resident:
            return vb, ks
    if (64, 1) in configs:
        return 64, 1
    return configs[-1] if configs else (64, 1)


def _scan_fits(kind, config, device_index, sset: StageSet) -> str | None:
    vb, ks = config
    _sm, smem_optin, major = _device_props(device_index)
    if config not in sset.reg_of(kind):
        return f"no {kind}_r{vb}k{ks} stage in {sset.label}"
    if major < 9:
        return f"compute capability {major}.x has no register scan"
    if REG_SCAN_SMEM[(kind, vb, ks)] > smem_optin:
        return (
            f"{kind} {config} needs {REG_SCAN_SMEM[(kind, vb, ks)]} B of shared memory"
        )
    return None


def _prescan_fits(kind, config, device_index, sset: StageSet) -> str | None:
    if config not in PRESCAN_CONFIGS:
        return f"{config} has no pre-scan stage"
    if config not in sset.split_of(kind):
        return f"no complete split path ({kind}_pre_r{config[0]}k{config[1]} + split/stitch) in {sset.label}"
    return _scan_fits(kind, config, device_index, sset)


def _scan_residency(kind, vb, ks, device_index) -> int:
    _sm, smem_optin, _major = _device_props(device_index)
    return max(1, _residency(kind, vb, ks, smem_optin))


def _wave_factor(ctas: int, sm_count: int, residency: int) -> float:
    co_resident = residency >= 2
    per_wave = sm_count * (2 if co_resident else 1)
    full_factor = SCAN_CO_RESIDENT_FACTOR if co_resident else 1.0
    full, tail = divmod(ctas, per_wave)
    cost = full * full_factor
    if tail:
        cost += full_factor if (full or tail > sm_count) else 1.0
    return cost


def _scan_makespan(chunks, splits, per_segment, sm_count, residency) -> float:
    runs = sorted(
        (-(-n // k), k * per_segment) for n, k in zip(chunks, splits, strict=False)
    )
    slots = sm_count * (2 if residency >= 2 else 1)
    longest = runs[-1][0]
    releases = []
    capacity, waiting = slots, 0
    for steps, ctas in runs:
        resident = min(ctas, capacity)
        capacity -= resident
        if resident:
            releases.append((steps, resident))
        waiting += ctas - resident
    makespan = longest
    while waiting > 0 and releases:
        next_releases = []
        for at, count in releases:
            if waiting <= 0:
                break
            started = min(count, waiting)
            waiting -= started
            makespan = max(makespan, at + longest)
            next_releases.append((at + longest, started))
        releases = next_releases
    factor = (
        SCAN_CO_RESIDENT_FACTOR
        if (residency >= 2 and sum(c for _s, c in runs) > sm_count)
        else 1.0
    )
    return factor * makespan


def bound_layout_chunks(chunks_bound: int, num_seqs: int) -> tuple[int, ...]:
    return (chunks_bound - (num_seqs - 1),) + (1,) * (num_seqs - 1)


def _serial_cost(chunks, per_segment, sm_count, residency, step_factor) -> float:
    return step_factor * _scan_makespan(
        chunks, (1,) * len(chunks), per_segment, sm_count, residency
    )


def split_cta_limit(sm_count: int, residency: int) -> int:
    return (
        MAX_SCAN_ROUNDS * 2 * sm_count
        if residency >= 2
        else MAX_SCAN_WAVES_SERIAL * sm_count
    )


@cache
def _prescan_capacity(kind: str, config: tuple[int, int]) -> int:
    """Chunks one pre-scan CTA of ``kind`` at ``config`` covers (the registry's ``scan.max_segment_chunks``): the
    longest segment the split route can run."""
    vb, ks = config
    for found in cake_jit.MODULES.values():
        if found["stage"] == f"{kind}_pre_r{vb}k{ks}":
            return int(found["scan"]["max_segment_chunks"])
    return 0


def _best_split(chunks, per_segment, sm_count, residency, step_factor, max_segment_len):
    """The cheapest split ``(segment_len, cost)``; a candidate longer than ``max_segment_len`` chunks (the pre-scan
    capacity) is not admissible, so the search keeps the best of the rest (``(0, inf)`` when none is left)."""
    longest = max(chunks)
    limit = split_cta_limit(sm_count, residency)
    best_len, best_cost = 0, math.inf
    for segments in range(2, longest // MIN_SEGMENT_CHUNKS + 1):
        seg_len = -(-longest // segments)
        if seg_len < MIN_SEGMENT_CHUNKS or seg_len >= longest:
            continue
        if seg_len > max_segment_len:
            continue
        splits = tuple(max(1, -(-n // seg_len)) for n in chunks)
        if sum(splits) * per_segment > limit:
            continue
        scan = step_factor * _scan_makespan(
            chunks, splits, per_segment, sm_count, residency
        )
        pre = (
            PRESCAN_WARMUP_CHUNKS
            * PRESCAN_STEP_FACTOR
            * step_factor
            * _wave_factor(
                sum(k - 1 for k in splits) * per_segment, sm_count, residency
            )
        )
        cost = scan + pre + STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        if cost < best_cost:
            best_len, best_cost = seg_len, cost
    return best_len, best_cost


def choose_scan_plan(kind, total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset: StageSet,
                     pinned: bool = True, *, scan_config: tuple[int, int] | None = None,
                     segment_chunks: int | None = None) -> tuple[ScanPlan, dict]:  # fmt: skip
    """The scan plan of one scan kind over ``sset`` and its pricing record.  ``pinned`` honours the
    ``scan_config`` / ``segment_chunks`` pins."""
    forced_len = segment_chunks if pinned else None
    forced_config = scan_config if pinned else None
    baseline = choose_register_scan_config(
        kind, num_seqs, num_v_heads, device_index, sset, pinned, forced_config
    )
    rec: dict[str, Any] = {"baseline": list(baseline)}
    if forced_len is not None:
        unsplittable = _prescan_fits(kind, baseline, device_index, sset)
        if forced_len > 0 and unsplittable is not None:
            warnings.warn(
                f"forced segment length {forced_len} ignored: {unsplittable}",
                RuntimeWarning,
                stacklevel=2,
            )
            rec["forced_seg_ignored"] = unsplittable
            return ScanPlan(baseline, 0), rec
        rec["forced_seg"] = forced_len
        return ScanPlan(baseline, forced_len), rec
    unfit = _scan_fits(kind, baseline, device_index, sset)
    if unfit is not None:
        rec["baseline_unfit"] = unfit
        return ScanPlan(baseline, 0), rec
    is_pinned = forced_config is not None
    chunks_bound = total_tokens // CHUNK + num_seqs
    chunks = (
        tuple(-(-int(x) // CHUNK) for x in seq_lens)
        if seq_lens is not None
        else bound_layout_chunks(chunks_bound, num_seqs)
    )
    sm_count, _smem_optin, _major = _device_props(device_index)
    candidates = (
        [baseline]
        if is_pinned
        else [
            c
            for c in REG_SCAN_CONFIGS
            if _scan_fits(kind, c, device_index, sset) is None
        ]
    )
    baseline_cost = _serial_cost(
        chunks,
        num_v_heads * (HEAD_DIM // baseline[0]),
        sm_count,
        _scan_residency(kind, baseline[0], baseline[1], device_index),
        SCAN_STEP_FACTOR[baseline],
    )
    best = (baseline, 0, baseline_cost)
    priced = []
    for config in candidates:
        per_segment = num_v_heads * (HEAD_DIM // config[0])
        residency = _scan_residency(kind, config[0], config[1], device_index)
        options = [
            (
                0,
                _serial_cost(
                    chunks, per_segment, sm_count, residency, SCAN_STEP_FACTOR[config]
                ),
            )
        ]
        if _prescan_fits(kind, config, device_index, sset) is None:
            options.append(
                _best_split(
                    chunks,
                    per_segment,
                    sm_count,
                    residency,
                    SCAN_STEP_FACTOR[config],
                    _prescan_capacity(kind, tuple(config)),
                )
            )
        for seg_len, cost in options:
            priced.append(
                [list(config), seg_len, round(cost, 3) if math.isfinite(cost) else None]
            )
            if cost < best[2]:
                best = (config, seg_len, cost)
    rec["priced"] = priced
    rec["baseline_cost"] = round(baseline_cost, 3)
    config, seg_len, cost = best
    if (config, seg_len) != (
        baseline,
        0,
    ) and cost >= SPLIT_GAIN_THRESHOLD * baseline_cost:
        return ScanPlan(baseline, 0), rec
    return ScanPlan(config, seg_len), rec


@lru_cache(maxsize=4096)
def _scan_plans_memo(
    total_tokens,
    num_seqs,
    num_v_heads,
    device_index,
    seq_lens,
    sset,
    pinned,
    props,
    pins,
):
    segment_chunks, scan_config = pins
    f, _ = choose_scan_plan("fwd_h", total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset, pinned,
                            scan_config=scan_config, segment_chunks=segment_chunks)  # fmt: skip
    b, _ = choose_scan_plan("dhu", total_tokens, num_seqs, num_v_heads, device_index, seq_lens, sset, pinned,
                            scan_config=scan_config, segment_chunks=segment_chunks)  # fmt: skip
    return f, b


def scan_plans(total_tokens, num_seqs, num_v_heads, device_index, seq_lens=None, sset: StageSet | None = None,
               pinned: bool = True, *, scan_config: tuple[int, int] | None = None,
               segment_chunks: int | None = None) -> tuple[ScanPlan, ScanPlan]:  # fmt: skip
    """The forward and adjoint scan plans of one layout on one device (``seq_lens`` ``None``: the device route's
    bound layout)."""
    sset = sset or stage_set(cake_backend.device_arch(int(device_index)))
    seq_key = None if seq_lens is None else tuple(int(x) for x in seq_lens)
    return _scan_plans_memo(int(total_tokens), int(num_seqs), int(num_v_heads), int(device_index), seq_key, sset,
                            bool(pinned), _device_props(int(device_index)), (segment_chunks, scan_config))  # fmt: skip


# --------------------------------------------------------------------------- #
# workspace pool
# --------------------------------------------------------------------------- #
WORKSPACE_SHAPES_PER_ROLE = 4


# The handle of the current stream of a device without building a ``torch.cuda.Stream`` (that costs microseconds of
# host time per call); ``torch.cuda.current_stream`` where this PyTorch build lacks the getter.
_RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None)


def _stream_key(device_index: int) -> tuple[int, int]:
    """(device, handle of its current stream): the key of the per-stream workspaces."""
    if _RAW_STREAM is not None:
        return device_index, _RAW_STREAM(device_index)
    return device_index, torch.cuda.current_stream(device_index).cuda_stream


class Workspaces:
    def __init__(self):
        self.tables: dict = {}
        # self-cleaning buffers per (device, stream): zero when created, left zero by every launch that uses them
        self.zeroed: dict = {}

    def take_zeroed(
        self, device_index: int, role: str, rows: int, row_shape: tuple, dtype, device
    ) -> torch.Tensor:
        """The first ``rows`` rows of the ``role`` buffer of the current stream of ``device_index``: created (or grown)
        by ``torch.zeros`` on that stream, then reused without a memset, since every launch leaves it zero.  The
        stream is read at the launch, so launches that may run concurrently (other streams, a CUDA graph capture on
        its capture stream) take distinct buffers."""
        buffers = self.zeroed.setdefault(_stream_key(device_index), {})
        buffer = buffers.get(role)
        if (
            buffer is None
            or buffer.shape[0] < rows
            or tuple(buffer.shape[1:]) != tuple(row_shape)
        ):
            buffer = buffers[role] = _zeros((rows, *row_shape), dtype, device)
        return buffer[:rows]

    def table(self, device_index: int) -> dict:
        return self.tables.setdefault(_stream_key(device_index), {})

    def take(self, table, role, shape, dtype, device):
        key = (tuple(shape), dtype)
        buffers = table.setdefault(role, {})
        buffer = buffers.get(key)
        if buffer is None:
            buffer = buffers[key] = torch.empty(shape, dtype=dtype, device=device)
            if len(buffers) > WORKSPACE_SHAPES_PER_ROLE:
                del buffers[next(iter(buffers))]
        return buffer


class FreshWorkspaces(Workspaces):
    def table(self, device_index):
        return {}

    def take(self, table, role, shape, dtype, device):
        return torch.empty(shape, dtype=dtype, device=device)

    def take_zeroed(self, device_index, role, rows, row_shape, dtype, device):
        return _zeros((rows, *row_shape), dtype, device)


def _zeros(shape: tuple, dtype, device) -> torch.Tensor:
    """``torch.zeros`` on the current stream; an unsigned 32-bit buffer is zeroed as int32 and viewed."""
    if dtype == torch.uint32:
        return torch.zeros(shape, dtype=torch.int32, device=device).view(torch.uint32)
    return torch.zeros(shape, dtype=dtype, device=device)


_POOLED = Workspaces()
_FRESH = FreshWorkspaces()


def _workspaces(options: ChainOptions) -> Workspaces:
    return _POOLED if options.host_cache else _FRESH


@cache
def _scan_stages(kind: str, config: tuple[int, int], arch: str):
    """(scan stage, value block, pre-scan stage or None, chunks the pre-scan covers)."""
    vb, ks = config
    stage = f"{kind}_r{vb}k{ks}"
    if not _available(stage, arch):
        raise NotImplementedError(
            f"the {kind} register scan {config} has no record for {arch}"
        )
    pre = f"{kind}_pre_r{vb}k{ks}"
    if not _available(pre, arch):
        return stage, vb, None, 0
    return stage, vb, pre, PRESCAN_BLOCK_THREADS


# --------------------------------------------------------------------------- #
# adjoint-scan selection: dhu_tc2 (dv_local fused) or dv_local + the register scan
# --------------------------------------------------------------------------- #
# The rule: T_tc = makespan steps x t(n), t(n) = t_lat + (t_full - t_lat) x clamp((n - 48) / (SMs - 48), 0, 1) with n
# the resident CTAs (one per (sequence, value head)); T_reg = the planner's serial cost of the register scan at the
# bound-layout plan's configuration x t_unit + tiles x c_dvl / SMs.  dhu_tc2 is selected iff T_tc <= DHU_TC_GAIN T_reg.
DHU_TC_GAIN = 0.95
DHU_TC_LOW_RESIDENT = 48  # resident CTAs up to which a tcgen05 scan step takes t_lat
DHU_TC_CONSTANTS = ("t_lat", "t_full", "t_unit", "c_dvl")
_DHU_AUTO: dict = {}
_DHU_AUTO_LIMIT = 256


@cache
def _dhu_tc_constants(arch: str) -> tuple[tuple[str, float], ...] | None:
    """The ``dhu_tc2`` record's ``select`` constants for ``arch`` (``None`` when the record carries none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "dhu_tc2" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in DHU_TC_CONSTANTS):
                return None
            return tuple((key, float(const[key])) for key in DHU_TC_CONSTANTS)
    return None


def dhu_tc_pricing(chunks, num_v_heads, device_index, sset: StageSet | None = None, *,
                   scan_config: tuple[int, int] | None = None, segment_chunks: int | None = None) -> dict:  # fmt: skip
    """The selection rule's two sides for a serial layout of ``chunks`` (chunks per sequence); ``select`` is the
    answer.  The register path is priced at the configuration of the bound layout's adjoint plan, under the same
    pins as the call."""
    arch = cake_backend.device_arch(int(device_index))
    found = _dhu_tc_constants(arch)
    sm_count, _smem, _major = _device_props(device_index)
    n = min(len(chunks) * num_v_heads, sm_count)
    steps_tc = _scan_makespan(chunks, (1,) * len(chunks), num_v_heads, sm_count, 1)
    rec: dict[str, Any] = {"arch": arch, "steps_tc": steps_tc, "resident": n}
    if found is None:
        rec["missing_constants"] = True
        return rec
    const = dict(found)
    low = DHU_TC_LOW_RESIDENT
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    total = sum(chunks) * CHUNK
    plan, _pricing = choose_scan_plan("dhu", total, len(chunks), num_v_heads, device_index, None,
                                      sset or stage_set(arch), True, scan_config=scan_config,
                                      segment_chunks=segment_chunks)  # fmt: skip
    cfg = cast("tuple[int, int]", tuple(plan.config))
    cost_units = SCAN_STEP_FACTOR[cfg] * _scan_makespan(
        chunks,
        (1,) * len(chunks),
        num_v_heads * (HEAD_DIM // cfg[0]),
        sm_count,
        _scan_residency("dhu", *cfg, device_index),
    )
    tiles = sum(chunks) * num_v_heads
    rec.update(t_n=t_n, T_tc=steps_tc * t_n, reg_config=list(cfg), reg_cost_units=cost_units,
               T_reg=cost_units * const["t_unit"] + tiles * const["c_dvl"] / sm_count)  # fmt: skip
    rec["select"] = rec["T_tc"] <= DHU_TC_GAIN * rec["T_reg"]
    return rec


def _dhu_tc_selected(
    meta: ChunkMeta, num_v_heads: int, device_index: int, options: ChainOptions
) -> bool:
    """Whether a serial plan's adjoint scan is ``dhu_tc2``: the ``dhu_scan`` pin, else the rule on a host-built table
    (a device-built table has no layout on the host: the register path)."""
    if options.dhu_scan is not None:
        return options.dhu_scan == "tc"
    if meta.seq_lens is None:
        return False
    key = (meta.seq_lens, int(num_v_heads), int(device_index), options.pins)
    hit = _DHU_AUTO.get(key) if options.host_cache else None
    if hit is None:
        chunks = tuple(-(-int(x) // CHUNK) for x in meta.seq_lens)
        hit = bool(dhu_tc_pricing(chunks, num_v_heads, device_index, scan_config=options.scan_config,
                                  segment_chunks=options.segment_chunks).get("select", False))  # fmt: skip
        if options.host_cache:
            if len(_DHU_AUTO) >= _DHU_AUTO_LIMIT:
                del _DHU_AUTO[next(iter(_DHU_AUTO))]
            _DHU_AUTO[key] = hit
    return hit


# --------------------------------------------------------------------------- #
# pass 2 of a split adjoint plan: dhu_tc2 from the stitched segment states, or the register scan
# --------------------------------------------------------------------------- #
# A split plan at ``SDT_CONFIG`` runs the pre-scan and the stitch, then pass 2 either as ``dv_local`` + the register
# scan or on ``dhu_tc2`` (one CTA per (segment, value head), ``dv_local`` fused) from the stitched per-segment ``dht``;
# on the ``dhu_tc2`` form the pre-scan is ``dhu_pre_dvl_r64k1``, which computes the ``dv_local`` tiles of the chunks it
# scans, so no ``dv_local`` stage runs.  Both forms give the same bits at the same segment table.  The rule prices pass
# 2 the way :func:`dhu_tc_pricing` prices a serial scan: T_tc2 = steps x t(n), steps the makespan of the segment table
# at one CTA per (segment, value head) and one CTA per SM, n = min(segments x HV, SMs) and t(n) the ``dhu_tc2``
# interpolation above; T_reg2 = SCAN_STEP_FACTOR[SDT_CONFIG] x the planner makespan of the register pass 2 x t_unit +
# tiles x c_dvl / SMs (the ``dv_local`` stage that form needs, tiles = chunks x HV).  ``dhu_tc2`` runs pass 2 iff T_tc2
# <= sdt_gain x T_reg2 and T_reg2 - T_tc2 >= sdt_min_saving (microseconds), both per-architecture constants of the
# ``dhu_tc2`` record's ``select`` (an architecture without them keeps the register pass 2).  A device-built table has no
# layout on the host: the register pass 2.
SDT_CONFIG = (64, 1)
SDT_PRESCAN = "dhu_pre_dvl_r64k1"
SDT_CONSTANTS = ("sdt_gain", "sdt_min_saving")
_SDT_AUTO: dict = {}
_SDT_AUTO_LIMIT = 256
# The architectures whose tcgen05 adjoint scan adds the FP32 ``dv_local`` values to ``dv2``, so ``dv2`` rounds once
# (``dhu_tc2_dvl32`` for ``dhu_tc2``: the same arguments, tables and selection constants; :func:`dhu_tc2_stage`), and
# those whose split pre-scan does (``dhu_pre_dvl32_r64k1`` for ``SDT_PRESCAN``, :func:`sdt_prescan`).  B200 (CUPTI,
# cold L2, fresh processes): ``dhu_tc2_dvl32`` 0.990-0.995 x ``dhu_tc2``, -0.3 to -0.4 % of the backward;
# ``dhu_pre_dvl32_r64k1`` costs up to +1.2 % of the backward on the split plans of the long-sequence regime (its FP32
# tile adds 16 KiB of shared memory, still one CTA per SM), so no architecture routes it.
DVL32_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
PRE_DVL32_ARCHS: tuple[str, ...] = ()


def dhu_tc2_stage(arch: str) -> str:
    """The tcgen05 adjoint scan's stage on ``arch``: ``dhu_tc2_dvl32`` on an architecture of ``DVL32_ARCHS`` whose
    registry delivers it, else ``dhu_tc2``."""
    if arch in DVL32_ARCHS and _available("dhu_tc2_dvl32", arch):
        return "dhu_tc2_dvl32"
    return "dhu_tc2"


def sdt_prescan(arch: str) -> str:
    """The pre-scan of a split plan whose pass 2 runs on the tcgen05 scan, on ``arch``: ``dhu_pre_dvl32_r64k1`` on an
    architecture of ``PRE_DVL32_ARCHS`` whose registry delivers it, else ``SDT_PRESCAN``."""
    if arch in PRE_DVL32_ARCHS and _available("dhu_pre_dvl32_r64k1", arch):
        return "dhu_pre_dvl32_r64k1"
    return SDT_PRESCAN


@cache
def _sdt_constants(arch: str) -> tuple[float, float] | None:
    """``(sdt_gain, sdt_min_saving)`` of the ``dhu_tc2`` record's ``select`` for ``arch`` (``None``: none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "dhu_tc2" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in SDT_CONSTANTS):
                return None
            return float(const["sdt_gain"]), float(const["sdt_min_saving"])
    return None


def sdt_pricing(chunks, segment_len: int, num_v_heads: int, device_index: int) -> dict:
    """The rule's two sides for pass 2 of a split ``SDT_CONFIG`` plan of ``segment_len`` chunks over ``chunks`` (chunks
    per sequence); ``select`` is the answer."""
    arch = cake_backend.device_arch(int(device_index))
    sm_count, _smem, _major = _device_props(device_index)
    splits = tuple(max(1, -(-int(c) // int(segment_len))) for c in chunks)
    n_tc = sum(splits) * int(num_v_heads)
    rec: dict[str, Any] = {
        "arch": arch,
        "segment_len": int(segment_len),
        "segments": sum(splits),
        "tc2_ctas": n_tc,
    }
    found, sdt = _dhu_tc_constants(arch), _sdt_constants(arch)
    if found is None or sdt is None:
        rec["missing_constants"] = True
        rec["select"] = False
        return rec
    const = dict(found)
    gain, min_saving = sdt
    vb, ks = SDT_CONFIG
    reg_units = SCAN_STEP_FACTOR[SDT_CONFIG] * _scan_makespan(chunks, splits, int(num_v_heads) * (HEAD_DIM // vb),
                                                              sm_count, _scan_residency("dhu", vb, ks, device_index))  # fmt: skip
    steps_tc = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    n = min(n_tc, sm_count)
    low = DHU_TC_LOW_RESIDENT
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    tiles = sum(chunks) * int(num_v_heads)
    t_dvl = tiles * const["c_dvl"] / sm_count
    t_tc2, t_reg2 = steps_tc * t_n, reg_units * const["t_unit"] + t_dvl
    rec.update(steps_tc=steps_tc, t_n=t_n, T_tc2=t_tc2, reg2_units=reg_units, T_dvl=t_dvl, T_reg2=t_reg2,
               saving=t_reg2 - t_tc2, gain=gain, min_saving=min_saving)  # fmt: skip
    rec["select"] = t_tc2 <= gain * t_reg2 and t_reg2 - t_tc2 >= min_saving
    return rec


def _sdt_selected(meta: ChunkMeta, segs: SegmentTable, num_v_heads: int, device_index: int, options: ChainOptions,
                  arch: str) -> bool:  # fmt: skip
    """Whether a split adjoint plan runs pass 2 on ``dhu_tc2``: a split ``SDT_CONFIG`` plan of a host-built table,
    the ``dhu_scan`` pin, else the rule (see above)."""
    if options.dhu_scan == "reg" or meta.seq_lens is None:
        return False
    if not (segs.segment_len > 0 and segs.num_segments > meta.num_seqs):
        return False
    if tuple(segs.scan_config) != SDT_CONFIG or not (
        _available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)
    ):
        return False
    if options.dhu_scan == "tc":
        return True
    key = (meta.seq_lens, int(num_v_heads), int(device_index), int(segs.segment_len))
    hit = _SDT_AUTO.get(key) if options.host_cache else None
    if hit is None:
        chunks = tuple(-(-int(x) // CHUNK) for x in meta.seq_lens)
        hit = bool(
            sdt_pricing(chunks, segs.segment_len, num_v_heads, device_index)["select"]
        )
        if options.host_cache:
            if len(_SDT_AUTO) >= _SDT_AUTO_LIMIT:
                del _SDT_AUTO[next(iter(_SDT_AUTO))]
            _SDT_AUTO[key] = hit
    return hit


# --------------------------------------------------------------------------- #
# the adjoint table priced for dhu_tc2 (host route)
# --------------------------------------------------------------------------- #
# The register planner prices the adjoint table for the register scans (two CTAs per (segment, value head) at
# ``v_block`` 64, their step and co-residency factors); the pass-2 rules then run ``dhu_tc2`` on THAT table.  With
# ``TC_PLAN`` a host-built table without plan pins whose planner configuration is ``SDT_CONFIG`` may instead take the
# table priced for ``dhu_tc2`` itself.  Candidates: the serial table and the ``SDT_CONFIG`` split tables of ``L``
# chunks (``MIN_SEGMENT_CHUNKS`` to the pre-scan capacity), each priced (microseconds) T = makespan(segments x HV CTAs,
# one per SM) x t(n) (the ``dhu_tc2`` interpolation of :func:`sdt_pricing` over ``TC_PLAN_STEP``) + the register
# planner's pre-scan and stitch terms x ``t_unit`` + the ``dv_local`` tiles of the pre-scan's warm-up window.  The planner's table is priced
# in the form its rules run (tcgen05, or the register pass with ``dv_local``), and the cheapest candidate replaces it
# when it saves at least ``TC_PLAN_MIN_SAVING`` us and a fraction ``1 - TC_PLAN_GAIN``; the backward then runs the
# candidate on ``dhu_tc2``.  A serial planner table that the two-stream rule groups (:func:`two_stream_split`) keeps
# its form: its scan overlaps the tail (the R form), which the pricing does not credit.  Another table moves the warm-up truncation and the stitch points, so the bits follow the
# plan: plan pins keep the planner's table.
TC_PLAN = True
TC_PLAN_GAIN = 0.95
TC_PLAN_MIN_SAVING = 10.0
# (gain, min_saving) of an architecture in place of (TC_PLAN_GAIN, TC_PLAN_MIN_SAVING) (:func:`tc_plan_thresholds`).
# R200 measured the priced table against the planner's (backward, det-vs-nondet bench): faster at 4.2-5.6 us priced
# savings (r07 -3.2 %, (2, 6) -2.6 %, P1 -0.9 %), slower at 0.9 us ((8, 24) +0.35 %).
TC_PLAN_SELECT = {"sm_107a": (1.0, 2.0)}
# The dhu_tc2 step time (microseconds) at low and full residency that prices these tables, measured (CUPTI kernel
# durations of L = 64 tables, 3 to 288 CTAs: B200 2.32-2.44, GB300 2.15-2.26 per step); an architecture not listed takes
# the dhu_tc2 record's t_lat / t_full.
TC_PLAN_STEP = {"sm_100a": (2.35, 2.44), "sm_103a": (2.17, 2.25)}


def tc_plan_thresholds(arch: str) -> tuple[float, float]:
    """``(gain, min_saving)`` that select the tc-priced adjoint table on ``arch``."""
    return TC_PLAN_SELECT.get(arch, (TC_PLAN_GAIN, TC_PLAN_MIN_SAVING))


def _tc_table_price(
    chunks,
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    const: dict,
    pre_residency: int,
) -> dict:
    """The priced ``dhu_tc2`` form of one adjoint table (``segment_len`` 0 or not below the longest sequence: serial)."""
    if segment_len <= 0 or max(chunks) <= segment_len:
        splits = (1,) * len(chunks)
    else:
        splits = tuple(max(1, -(-int(n) // int(segment_len))) for n in chunks)
    ctas = sum(splits) * int(num_v_heads)
    steps = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    low = DHU_TC_LOW_RESIDENT
    n = min(ctas, sm_count)
    t_n = const["t_lat"] + (const["t_full"] - const["t_lat"]) * min(
        max((n - low) / (sm_count - low), 0.0), 1.0
    )
    extra = 0.0
    if max(splits) > 1:
        tails = sum(k - 1 for k in splits) * int(num_v_heads)
        units = (
            PRESCAN_WARMUP_CHUNKS * PRESCAN_STEP_FACTOR * SCAN_STEP_FACTOR[SDT_CONFIG]
            * _wave_factor(tails * (HEAD_DIM // SDT_CONFIG[0]), sm_count, pre_residency)
            + STITCH_FIXED_CHUNKS + STITCH_COPY_CHUNKS * (max(splits) - 1)
        )  # fmt: skip
        extra = (
            units * const["t_unit"]
            + tails * PRESCAN_WARMUP_CHUNKS * const["c_dvl"] / sm_count
        )
    return {"segment_len": int(segment_len) if max(splits) > 1 else 0, "segments": sum(splits), "ctas": ctas,
            "steps": steps, "T_tc2": steps * t_n, "T_extra": extra, "T": steps * t_n + extra}  # fmt: skip


@lru_cache(maxsize=1024)
def _tc_plan_memo(
    chunks: tuple[int, ...],
    num_v_heads: int,
    device_index: int,
    plan: ScanPlan,
    props,
    arch: str,
    _gain: float,
    _min_saving: float,
) -> dict:
    rec: dict[str, Any] = {
        "arch": arch,
        "select": False,
        "planner": [list(plan.config), plan.segment_len],
    }
    found = _dhu_tc_constants(arch)
    if found is None or tuple(plan.config) != SDT_CONFIG:
        return rec
    if not (_available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)):
        return rec
    const = dict(found)
    if arch in TC_PLAN_STEP:
        const["t_lat"], const["t_full"] = TC_PLAN_STEP[arch]
    sm_count = props[0]
    pre_res = _scan_residency("dhu", *SDT_CONFIG, device_index)
    serial = plan.segment_len <= 0 or max(chunks) <= plan.segment_len
    at = _tc_table_price(
        chunks, num_v_heads, 0 if serial else plan.segment_len, sm_count, const, pre_res
    )
    if serial:
        rule = dhu_tc_pricing(chunks, num_v_heads, device_index)
        plan_t = (
            at["T"] if rule.get("select") else min(at["T"], rule.get("T_reg", math.inf))
        )
    else:
        rule = sdt_pricing(chunks, plan.segment_len, num_v_heads, device_index)
        plan_t = (
            at["T"]
            if rule.get("select")
            else at["T_extra"] + rule.get("T_reg2", math.inf)
        )
    cap = _prescan_capacity("dhu", SDT_CONFIG)
    cands = [_tc_table_price(chunks, num_v_heads, 0, sm_count, const, pre_res)]
    for seg_len in range(MIN_SEGMENT_CHUNKS, min(cap, max(chunks) - 1) + 1):
        cands.append(
            _tc_table_price(chunks, num_v_heads, seg_len, sm_count, const, pre_res)
        )
    best = min(cands, key=lambda c: (c["T"], -c["segment_len"]))
    rec.update(plan_T=plan_t, best=best, saving=plan_t - best["T"])
    rec["select"] = bool(
        best["segment_len"] != (0 if serial else plan.segment_len)
        and plan_t - best["T"] >= _min_saving
        and best["T"] <= _gain * plan_t
    )
    rec["segment_len"] = best["segment_len"]
    return rec


def tc_plan_pricing(
    chunks, num_v_heads: int, device_index: int, plan: ScanPlan
) -> dict:
    """The adjoint table priced for ``dhu_tc2`` against the planner's ``plan`` for a host layout of ``chunks`` (chunks
    per sequence); ``select`` is the answer and ``segment_len`` the table (0: serial).  See above."""
    arch = cake_backend.device_arch(int(device_index))
    gain, min_saving = tc_plan_thresholds(arch)
    return dict(
        _tc_plan_memo(
            tuple(int(c) for c in chunks),
            int(num_v_heads),
            int(device_index),
            plan,
            _device_props(int(device_index)),
            arch,
            gain,
            min_saving,
        )  # fmt: skip
    )


# --------------------------------------------------------------------------- #
# deeper split tables priced for the tcgen05 chains (fwd_ho, dhu_tc2), per architecture
# --------------------------------------------------------------------------- #
# The register planner prices the forward table for the register scans (``HEAD_DIM / v_block`` CTAs per (segment, value
# head), two per SM), but ``fwd_ho`` runs it with one CTA per (segment, value head) and one per SM; the adjoint table
# that ``TC_PLAN`` keeps can likewise stay shallower than the ``dhu_tc2`` chain needs on a device with many SMs.  With
# ``TC_PLAN`` (False keeps the planner's tables in both directions), on an architecture of ``CHAIN_PLAN``, a host-built
# table without plan pins whose scan the two-stream rule does not group may take a DEEPER ``SDT_CONFIG`` split table
# (more segments than the chosen table) for each direction, priced with the architecture's measured constants
# (microseconds): T = makespan(segments x HV CTAs, one per SM) x t(n), t(n) = t_lat + (t_full - t_lat) x clamp((n - 48)
# / (SMs - 48), 0, 1) for n = min(CTAs, SMs) resident, plus, for a split table, ``fixed`` + ``per_segment`` x (most
# segments of a sequence - 1) + ``per_wave`` x the pre-scan's waves (segments x HV x 2 HEAD_DIM / v_block CTAs over 2 x
# SMs).  The cheapest deeper table replaces the chosen one when it saves at least ``min_saving`` and prices at most
# ``gain`` x the chosen table; the forward then runs the pre-scan, the stitch and ``fwd_ho`` on it, the backward the
# fused pre-scan, the stitch and ``dhu_tc2``.  The constants are R200 measurements (212 SMs, ``fwd_ho`` / ``dhu_tc2``
# step times of serial tables of 48 to 208 CTAs, split costs fitted to pinned split tables of 12 rows; the fit's error
# per table: forward median 1.8 %, up to 11.7 % on the 6-sequence (2, 6) pack, adjoint 0.9 %, up to 4.7 %, hence
# ``gain`` 0.95); an architecture not listed keeps its tables.  Another table moves the segment boundaries, so the bits
# follow the plan (plan pins keep the planner's tables).  On an architecture of ``FWD_HO_VH2_ARCHS`` the forward prices a
# table whose launch runs the value-half split (:func:`fwd_ho_vh2_engaged`: twice its segments x HV CTAs fit one round)
# with its chain term scaled by ``FWD_HO_VH2_STEP``, the split's chain time over ``fwd_ho``'s on the same table (R200,
# 216 SMs, the 96-CTA table of one 16384-token sequence over 16 value heads: 61.44 against 81.76 us, 0.7515); the
# deeper tables the split does not fit keep the ``fwd_ho`` price.
CHAIN_PLAN_CONSTANTS = ("t_lat", "t_full", "fixed", "per_segment", "per_wave")
CHAIN_PLAN: dict[str, dict[str, Any]] = {
    "sm_107a": {
        "fwd": (1.77, 2.31, 0.0, 1.7, 6.5),
        "bwd": (1.66, 1.88, 0.0, 0.6, 10.8),
        "gain": 0.95,
        "min_saving": 2.0,
    },
}
FWD_HO_VH2_STEP = 0.75


def _chain_table_price(
    chunks,
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    const: tuple,
    vh2: float = 1.0,
) -> float:
    """The priced tcgen05 chain of one table (``segment_len`` 0 or not below the longest sequence: serial); ``vh2``
    scales the chain term of a table whose launch runs the value-half split (:func:`fwd_ho_vh2_engaged`)."""
    t_lat, t_full, fixed, per_segment, per_wave = const
    if segment_len <= 0 or max(chunks) <= segment_len:
        splits = (1,) * len(chunks)
    else:
        splits = tuple(max(1, -(-int(n) // int(segment_len))) for n in chunks)
    ctas = sum(splits) * int(num_v_heads)
    steps = _scan_makespan(chunks, splits, int(num_v_heads), sm_count, 1)
    n = min(ctas, sm_count)
    low = DHU_TC_LOW_RESIDENT
    t_n = t_lat + (t_full - t_lat) * min(max((n - low) / (sm_count - low), 0.0), 1.0)
    extra = 0.0
    if max(splits) > 1:
        pre_ctas = ctas * (2 * HEAD_DIM // SDT_CONFIG[0])
        extra = (
            fixed
            + per_segment * (max(splits) - 1)
            + per_wave * -(-pre_ctas // (2 * sm_count))
        )
    chain = steps * t_n
    if vh2 != 1.0 and fwd_ho_vh2_engaged(sum(splits), num_v_heads, sm_count):
        chain *= vh2
    return chain + extra


@lru_cache(maxsize=1024)
def _chain_plan_memo(
    kind: str,
    chunks: tuple[int, ...],
    num_v_heads: int,
    segment_len: int,
    sm_count: int,
    arch: str,
    const: tuple,
    gain: float,
    min_saving: float,
    vh2: float,
) -> dict:
    current = 0 if segment_len <= 0 or max(chunks) <= segment_len else int(segment_len)
    segments = sum(max(1, -(-int(n) // current)) if current else 1 for n in chunks)
    plan_t = _chain_table_price(chunks, num_v_heads, current, sm_count, const, vh2)
    rec: dict[str, Any] = {
        "arch": arch,
        "kind": kind,
        "select": False,
        "segment_len": current,
        "plan_T": plan_t,
    }
    best = None
    for seg_len in range(MIN_SEGMENT_CHUNKS, min(_prescan_capacity("dhu" if kind == "bwd" else "fwd_h", SDT_CONFIG),
                                                    max(chunks) - 1) + 1):  # fmt: skip
        if sum(max(1, -(-int(n) // seg_len)) for n in chunks) <= segments:
            continue  # deeper tables only
        t = _chain_table_price(chunks, num_v_heads, seg_len, sm_count, const, vh2)
        if best is None or t < best[1]:
            best = (seg_len, t)
    if best is None:
        return rec
    rec.update(best={"segment_len": best[0], "T": best[1]}, saving=plan_t - best[1])
    if plan_t - best[1] >= min_saving and best[1] <= gain * plan_t:
        rec.update(select=True, segment_len=best[0])
    return rec


def chain_plan_pricing(
    kind: str, chunks, num_v_heads: int, device_index: int, plan: ScanPlan
) -> dict:
    """``kind`` ``"fwd"`` (the ``fwd_ho`` chain) or ``"bwd"`` (``dhu_tc2``): the deeper split table priced for that
    chain against ``plan`` for a host layout of ``chunks`` (chunks per sequence) on the device's architecture;
    ``select`` is the answer and ``segment_len`` the table; on an architecture of ``FWD_HO_VH2_ARCHS`` a forward table
    whose launch runs the value-half split is priced as that split (``FWD_HO_VH2_STEP``).  An architecture without
    ``CHAIN_PLAN`` constants never selects.  See above."""
    arch = cake_backend.device_arch(int(device_index))
    found = CHAIN_PLAN.get(arch)
    if found is None:
        return {
            "arch": arch,
            "kind": kind,
            "select": False,
            "segment_len": int(plan.segment_len),
        }
    return dict(
        _chain_plan_memo(
            kind,
            tuple(int(c) for c in chunks),
            int(num_v_heads),
            int(plan.segment_len),
            _device_props(int(device_index))[0],
            arch,
            tuple(float(x) for x in found[kind]),
            float(found["gain"]),
            float(found["min_saving"]),
            FWD_HO_VH2_STEP if kind == "fwd" and arch in FWD_HO_VH2_ARCHS else 1.0,
        )  # fmt: skip
    )


# --------------------------------------------------------------------------- #
# the split adjoint plan beside a group of three's cluster tail on two streams (the R form over segments), per
# architecture
# --------------------------------------------------------------------------- #
# On an architecture of ``SPLIT_R_FORM``, a call of a value-head group of three whose backward tail is a cluster tail
# (``tail_g3_f32x``, or the pinned ``tail_g3``) runs
# the ``dhu_tc2`` pass 2 of a split adjoint plan as two groups of segments where :func:`two_stream_split` groups them
# (:func:`sequence_groups` with ``split_adjoint``): the caller's stream runs group A (the full rounds), then the fused
# tail of its chunks while a side stream runs group B (the partial round), then the tail of the rest.  The
# deterministic package runs the same form on every split table beside ``tail_g3``.  On a host-built table without
# plan pins (and without a ``dhu_scan`` pin), a planner's split adjoint table that this form groups, with its pass 2 on
# ``dhu_tc2`` (the ``sdt_pricing`` rule), keeps its form: ``TC_PLAN`` and ``CHAIN_PLAN`` do not replace it (their
# pricing does not credit the overlap with the tail), as a serial table that the two-stream rule groups keeps its form.
# Measured on R200 (212 SMs, one process, sustained protocol): the backward of the (16, 48) head pair on this form of
# its planner table (94-chunk segments, 6 segments, groups of 4 + 2) took 1.3261 ms in the deterministic package,
# against 1.3454 ms on this package's tc-priced table (46-chunk segments, 13 segments, one stream).  On B200 and GB300
# (FP32 beta, one process per GPU, both arm orders) the (2, 6) head pair over six packed sequences (241,817 tokens)
# took 2.8153 ms fwd+bwd on GB300 on this form (110-chunk segments, 37 segments, groups of 25 + 12) against 2.8271 ms
# on the tc-priced table (49 segments, one stream), and 3.1311 against 3.1337 ms on B200 (groups of 24 + 13; sustained
# protocol, within its noise).  The architectures not listed run one stream there.
SPLIT_R_FORM: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")


def _split_r_groups(arch: str, group: int | None, options: ChainOptions) -> bool:
    """Whether a call on ``arch`` of value-head group ``group`` groups the segments of a split adjoint table (the R
    form beside a group of three's cluster tail; :func:`sequence_groups` with ``split_adjoint``)."""
    return (
        arch in SPLIT_R_FORM
        and group == 3
        and GROUP_TAILS.get(tail_form(1, 3, options.tail, arch=arch)) == 3
    )


def split_r_form(
    chunks,
    num_v_heads: int,
    group: int | None,
    device_index: int,
    plan: ScanPlan,
    options: ChainOptions,
) -> bool:
    """Whether a host layout of ``chunks`` (chunks per sequence) keeps the planner's adjoint table ``plan`` for the R
    form of a split plan beside a group of three's cluster tail (see above): the device's architecture in
    ``SPLIT_R_FORM``, a value-head group ``group`` of three with a cluster tail, no plan or ``dhu_scan`` pin, a split
    ``SDT_CONFIG`` table whose pass 2 the ``sdt_pricing`` rule runs on ``dhu_tc2`` and whose segments
    :func:`two_stream_split` groups.  Both stream counts keep the same table (``streams=1`` runs it on one stream: the
    same bits)."""
    index = int(device_index)
    arch = cake_backend.device_arch(index)
    if not _split_r_groups(arch, group, options):
        return False
    if options.pins != (None, None) or options.dhu_scan is not None:
        return False
    seg_len = int(plan.segment_len)
    if tuple(plan.config) != SDT_CONFIG or seg_len <= 0 or max(chunks) <= seg_len:
        return False
    if not (_available("dhu_tc2", arch) and _available(SDT_PRESCAN, arch)):
        return False
    segments = sum(max(1, -(-int(n) // seg_len)) for n in chunks)
    if two_stream_split(segments, int(num_v_heads), _device_props(index)[0]) is None:
        return False
    return bool(sdt_pricing(tuple(chunks), seg_len, int(num_v_heads), index)["select"])


# --------------------------------------------------------------------------- #
# the one-wave split of the adjoint walk (exact, independent of the gates)
# --------------------------------------------------------------------------- #
# On the architectures of ADJOINT_SPLIT_ARCHS a host-built table splits each long sequence's adjoint walk into pieces
# of at most ``L`` chunks (:func:`bwd_piece_table`, the bounds of :func:`segment_bounds`) and runs it in two waves of
# the ``dhu_tc2_dvl32`` forms.  Wave 1: ``dhu_tc2_dvl32_modes`` walks each sequence's last piece from ``dht`` with its
# tapes and stores the adjoint it reaches at the piece's start, while ``dhu_tc2_dvl32_gm`` (on a side stream) walks
# each middle piece's affine map ``[B | M]`` (``B`` from zero, ``M`` from the identity); the stitch composes the end
# adjoint of every other piece; wave 2: ``dhu_tc2_dvl32_modes`` walks those pieces from their end adjoints with their
# tapes, the first piece storing ``dh0``.  A sequence of one piece is walked whole in wave 1.  The end adjoints come
# from the stitch (3xBF16 products, about 16 mantissa bits per step); ``piece_chain`` (FP32 FFMA, one dot product per
# element and step, slower) is the FP32 composition a caller selects through ADJOINT_CHAIN_STAGES.  Every chunk's walk is the serial walk's from the same end
# adjoint, which the split computes exactly up to that composition's rounding (no warm-up bound, so the plan and its
# cost do not depend on the gates).  :func:`adjoint_split_pricing` selects it over the serial table, both at their
# full cost.
ADJOINT_SPLIT_ARCHS: frozenset[str] = frozenset({"sm_100a", "sm_103a", "sm_107a"})
ADJOINT_SPLIT_STAGES = ("dhu_tc2_dvl32_modes", "dhu_tc2_dvl32_gm")
# the stages that compose the end adjoints, in preference order
ADJOINT_CHAIN_STAGES = ("stitch", "piece_chain")
# the mode bits of an entry of the forms (``seg_mode``): seed the walk from ``dht`` (row of the segment, or of its
# sequence with SPLIT_SEED_SEQ), write the dh / dv2 tapes, store the adjoint at the segment's start into the B half of
# ``seg_state`` (the stitch's layout)
SPLIT_SEED, SPLIT_SEED_SEQ, SPLIT_TAPE, SPLIT_OUT = 1, 2, 4, 8
# Per architecture, microseconds: the step of the walk with its tapes in a full wave (``walk``) and of wave 1, the
# summaries beside the last pieces' walks (``summary``); the step of the serial walk at an empty and at a full wave
# (``serial_low`` / ``serial_full``, interpolated by the fill); the composition's fixed cost and its cost per
# composed piece (``stitch`` / ``stitch_piece``); the split's fixed cost (its extra launches and the stream fork and
# join); the host's cost of wave 1's fork and join (``fork``): where the overlap that the side stream buys, the shorter
# of wave 1's two launches, is at most ``fork``, wave 1 runs on the caller's stream, summaries first (the same kernels
# and bits; the host's fork and join, not the GPU, then decides wave 1's time: measured on GB200's Grace host at the
# group-of-three pair rows).  sm_100a measured on B200 (148 SMs), sm_103a on GB300 (152 SMs; its ``fork`` from the
# pair rows there: one stream at the (1, 3) and (2, 6) pairs, two from (4, 12) on; B300 not measured), sm_107a on R200
# (212 SMs; ``serial_full`` extrapolated from the 144-CTA rows, the fullest serial wave of the bench rows there).
ADJOINT_SPLIT_COSTS: dict[str, dict[str, float]] = {
    "sm_100a": {
        "walk": 2.69,
        "summary": 2.77,
        "serial_low": 2.29,
        "serial_full": 2.72,
        "stitch": 10.5,
        "stitch_piece": 3.0,
        "fixed": 15.0,
        "fork": 0.0,
    },
    "sm_103a": {
        "walk": 2.24,
        "summary": 2.64,
        "serial_low": 2.13,
        "serial_full": 2.19,
        "stitch": 8.8,
        "stitch_piece": 3.1,
        "fixed": 15.0,
        "fork": 65.0,
    },
    "sm_107a": {
        "walk": 2.0,
        "summary": 2.09,
        "serial_low": 1.67,
        "serial_full": 1.7,
        "stitch": 8.1,
        "stitch_piece": 3.4,
        "fixed": 15.0,
        "fork": 0.0,
    },
}


def bwd_piece_table(chunks, num_v_heads: int, sm_count: int) -> int | None:
    """The piece length ``L`` (chunks) of the one-wave split of the adjoint walk over sequences of ``chunks`` chunks:
    the smallest ``L`` whose first wave fits one CTA per SM, ``sum_d max(ceil(n_d / L) - 1, 1) * HV <= sm_count`` (a
    sequence of ``P`` pieces puts its last piece and its ``P - 2`` middle pieces in the first wave, a sequence of one
    piece its whole walk); the second wave, ``sum_d (P_d - 1) * HV`` CTAs, fits too.  ``None`` where no ``L`` splits a
    sequence (more sequences x value heads than SMs, or every sequence one piece at the smallest fitting ``L``).
    Depends on the shapes, ``HV`` and the SM count alone."""
    chunks = [int(n) for n in chunks]
    hv, sms = int(num_v_heads), int(sm_count)
    if not chunks or hv <= 0 or len(chunks) * hv > sms or max(chunks) < 2:
        return None

    def ctas(length: int) -> int:
        return sum(max(-(-n // length) - 1, 1) for n in chunks) * hv

    lo, hi = 1, max(chunks)  # ctas(max(chunks)) = N x HV fits
    while lo < hi:
        mid = (lo + hi) // 2
        if ctas(mid) <= sms:
            hi = mid
        else:
            lo = mid + 1
    return lo if max(chunks) > lo else None


def adjoint_split_pricing(chunks, num_v_heads: int, sm_count: int, arch: str) -> dict:
    """The one-wave split (:func:`bwd_piece_table`) against the serial table of ``chunks``, both at their full cost
    (``ADJOINT_SPLIT_COSTS``, microseconds): the split pays wave 1 (its longest last piece or one-piece sequence at the
    walk's step, its longest middle piece at the summary's), the stitch over the most pieces of a sequence, wave 2 (its
    longest other piece) and its fixed cost; the serial table pays its longest sequence at the step of its fill.
    ``select`` where the split is cheaper; ``segment_len`` is ``L`` (0 without a split); ``one_stream`` where wave 1
    runs on the caller's stream: it has summaries and the shorter of its two launches is at most the ``fork`` cost."""
    chunks = tuple(int(n) for n in chunks)
    hv, sms = int(num_v_heads), int(sm_count)
    out: dict[str, Any] = {
        "select": False,
        "segment_len": 0,
        "split_us": None,
        "serial_us": None,
        "one_stream": False,
    }
    costs = ADJOINT_SPLIT_COSTS.get(arch)
    if costs is None or not chunks:
        return out
    fill = min(1.0, len(chunks) * hv / sms)
    rounds = -(-len(chunks) * hv // sms)
    step = costs["serial_low"] + (costs["serial_full"] - costs["serial_low"]) * fill
    out["serial_us"] = rounds * max(chunks) * step
    length = bwd_piece_table(chunks, hv, sms)
    if length is None:
        return out
    last = middle = front = 0
    pieces = 1
    for n in chunks:
        bounds = segment_bounds(n, length)
        lens = [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)]
        pieces = max(pieces, len(lens))
        last = max(last, lens[-1])
        if len(lens) > 2:
            middle = max(middle, max(lens[1:-1]))
        if len(lens) > 1:
            front = max(front, max(lens[:-1]))
    split = (
        max(last * costs["walk"], middle * costs["summary"])
        + costs["stitch"]
        + costs["stitch_piece"] * (pieces - 1)
        + front * costs["walk"]
        + costs["fixed"]
    )
    one_stream = middle > 0 and min(
        last * costs["walk"], middle * costs["summary"]
    ) <= costs.get("fork", 0.0)
    out.update(
        select=split < out["serial_us"],
        segment_len=length,
        split_us=split,
        one_stream=one_stream,
    )
    return out


def _adjoint_chain_stage(arch: str) -> str | None:
    """The stage that composes the split's end adjoints on ``arch``: the first of
    ``ADJOINT_CHAIN_STAGES`` delivered."""
    delivered = [stage for stage in ADJOINT_CHAIN_STAGES if _available(stage, arch)]
    return delivered[0] if delivered else None


def _adjoint_split_length(
    seq_lens, num_v_heads: int, device_index: int, options: ChainOptions
) -> int | None:
    """``L`` of the one-wave split a host layout takes on device ``device_index``, ``None`` for the other plans: the
    device's architecture in ``ADJOINT_SPLIT_ARCHS`` with the forms and a chain stage delivered, no plan or ``dhu_scan``
    pin, and :func:`adjoint_split_pricing` selecting it."""
    index = int(device_index)
    arch = cake_backend.device_arch(index)
    if (
        arch not in ADJOINT_SPLIT_ARCHS
        or options.pins != (None, None)
        or options.dhu_scan is not None
    ):
        return None
    if not all(_available(stage, arch) for stage in ADJOINT_SPLIT_STAGES):
        return None
    if _adjoint_chain_stage(arch) is None:
        return None
    priced = _adjoint_split_priced(seq_lens, num_v_heads, index, arch)
    return int(priced["segment_len"]) if priced["select"] else None


def _adjoint_split_priced(
    seq_lens, num_v_heads: int, device_index: int, arch: str
) -> dict:
    """:func:`adjoint_split_pricing` of a host layout on device ``device_index`` of architecture ``arch``."""
    chunks = tuple(-(-int(n) // CHUNK) for n in seq_lens)
    return adjoint_split_pricing(
        chunks, int(num_v_heads), _device_props(int(device_index))[0], arch
    )


@dataclass(frozen=True)
class AdjointSplit:
    """The entry tables of the one-wave split of the adjoint walk over a table's adjoint segments (the pieces): rows
    ``[segment index, mode bits]`` of each launch of the ``dhu_tc2_dvl32`` forms (:func:`adjoint_split_tables`), and the
    stitch's ``seg_flag`` (``[segments, HV]``: 0 at each sequence's last piece, whose B half the stitch copies);
    ``one_stream``: wave 1 on the caller's stream, summaries first (:func:`adjoint_split_pricing`), instead of the
    summaries on a side stream beside the last pieces' walks (the same kernels and bits)."""

    walk: torch.Tensor
    summary: torch.Tensor
    front: torch.Tensor
    seg_flag: torch.Tensor
    num_walk: int
    num_summary: int
    num_front: int
    one_stream: bool = False


def adjoint_split_lists(
    seq_seg_start,
) -> tuple[list[int], list[int], list[int], list[int], list[int], list[int]]:
    """The host entry lists of the split over a segment table whose sequence ``vs`` owns the segments
    ``seq_seg_start[vs] .. seq_seg_start[vs + 1]``: wave 1's walks (each sequence's last piece, seeded from its
    sequence's ``dht`` row, tapes, start adjoint out; a one-piece sequence the same without the store), wave 1's
    summaries (the middle pieces, mode 0), wave 2's walks (every other piece, seeded from its own end adjoint, tapes),
    each as segment indices and modes, in table order."""
    walk_i: list[int] = []
    walk_m: list[int] = []
    summary_i: list[int] = []
    front_i: list[int] = []
    front_m: list[int] = []
    for s0, s1 in zip(seq_seg_start[:-1], seq_seg_start[1:], strict=True):
        if s1 - s0 == 1:
            walk_i.append(s0)
            walk_m.append(SPLIT_SEED | SPLIT_SEED_SEQ | SPLIT_TAPE)
            continue
        walk_i.append(s1 - 1)
        walk_m.append(SPLIT_SEED | SPLIT_SEED_SEQ | SPLIT_TAPE | SPLIT_OUT)
        summary_i.extend(range(s0 + 1, s1 - 1))
        front_i.extend(range(s0, s1 - 1))
        front_m.extend([SPLIT_SEED | SPLIT_TAPE] * (s1 - 1 - s0))
    return walk_i, walk_m, summary_i, [0] * len(summary_i), front_i, front_m


def adjoint_split_tables(
    seq_seg_start, num_v_heads: int, device, one_stream: bool = False
) -> AdjointSplit:
    """:class:`AdjointSplit` of a segment table (:func:`adjoint_split_lists`), copied to ``device`` in one transfer;
    ``one_stream`` as given."""
    seq_seg_start = [int(x) for x in seq_seg_start]
    walk_i, walk_m, summary_i, summary_m, front_i, front_m = adjoint_split_lists(
        seq_seg_start
    )
    flags: list[int] = []
    for s0, s1 in zip(seq_seg_start[:-1], seq_seg_start[1:], strict=True):
        flags.extend([1] * ((s1 - s0 - 1) * int(num_v_heads)) + [0] * int(num_v_heads))
    parts = [walk_i + walk_m, summary_i + summary_m, front_i + front_m, flags]
    table = torch.tensor([x for part in parts for x in part], dtype=torch.int32)
    device = torch.device(device)
    table = (
        table.pin_memory().to(device, non_blocking=True)
        if device.type == "cuda"
        else table.to(device)
    )
    views = []
    offset = 0
    for part in parts:
        views.append(table[offset : offset + len(part)])
        offset += len(part)
    return AdjointSplit(
        walk=views[0].view(2, -1),
        summary=views[1].view(2, -1),
        front=views[2].view(2, -1),
        seg_flag=views[3].view(-1, int(num_v_heads)),
        num_walk=len(walk_i),
        num_summary=len(summary_i),
        num_front=len(front_i),
        one_stream=bool(one_stream),
    )


@lru_cache(maxsize=256)
def _host_adjoint_split(
    seq_lens: tuple[int, ...],
    segment_len: int,
    num_v_heads: int,
    device_index: int,
    lpt: bool,
    one_stream: bool = False,
) -> AdjointSplit:
    """:func:`adjoint_split_tables` of the adjoint segments :func:`build_chunk_meta` builds for ``seq_lens`` (the
    sequences in :func:`lpt_sequence_order` with ``lpt``) at ``segment_len``, wave 1 on one stream with
    ``one_stream``."""
    order = lpt_sequence_order(seq_lens) if lpt else None
    seq_seg_start = [0]
    for seq in order if order is not None else range(len(seq_lens)):
        n = -(-int(seq_lens[seq]) // CHUNK)
        seq_seg_start.append(
            seq_seg_start[-1] + len(segment_bounds(n, int(segment_len))) - 1
        )
    return adjoint_split_tables(
        seq_seg_start, num_v_heads, torch.device("cuda", device_index), one_stream
    )


# --------------------------------------------------------------------------- #
# prep selection: prep_tma (TMA-staged q / k, one CTA per (chunk, key head)) or the plain prep
# --------------------------------------------------------------------------- #
# The rule: CTAs = chunks x H and slots = ctas_per_sm x SMs; the plain prep (grid (chunks, HV)) runs where prep_tma's
# grid needs a second, short wave, slots < CTAs <= window x slots, prep_tma elsewhere.  Both give the same bits.  The
# tensor maps need q and k 16-byte aligned; an unaligned q or k takes the plain prep.  The tensor maps read q and k
# through their strides, so prep_tma reads the head views of a packed row (token stride (2 H + HV) x 128) in place;
# the plain prep reads contiguous rows, so a view it runs on is copied first (:func:`_prep_inputs`).
PREP_TMA_CONSTANTS = ("ctas_per_sm", "window")
TMA_ALIGN_BYTES = 16
# q / k views that prep_tma reads in place run prep_tma inside the rule's plain band too (no copy of q / k); a
# ``prep`` pin still decides
VIEWS_TAKE_PREP_TMA = True
# a strided q / k / v is copied into contiguous rows as the widest of these words its strides allow (:func:`_dense`):
# 16 bytes (a pair of FP64 lanes, moved without arithmetic), else 8 bytes
_COPY_WORDS = (torch.complex128, torch.int64)


@cache
def _prep_tma_constants(arch: str) -> tuple[int, float] | None:
    """The ``prep_tma`` record's ``select`` window for ``arch`` (``None`` when the record carries none)."""
    for found in cake_jit.MODULES.values():
        if found["stage"] == "prep_tma" and arch in found["arches"]:
            const = (found.get("select") or {}).get(arch)
            if const is None or any(key not in const for key in PREP_TMA_CONSTANTS):
                return None
            return int(const["ctas_per_sm"]), float(const["window"])
    return None


def use_prep_tma(num_chunks: int, num_heads: int, device_index: int) -> bool:
    """The dispatch rule: ``prep_tma`` for ``num_chunks`` x ``num_heads`` CTAs unless slots < CTAs <= window x slots
    (``False`` when the architecture's record carries no window)."""
    found = _prep_tma_constants(cake_backend.device_arch(int(device_index)))
    if found is None:
        return False
    ctas_per_sm, window = found
    sm_count, _smem, _major = _device_props(device_index)
    ctas, slots = int(num_chunks) * int(num_heads), ctas_per_sm * int(sm_count)
    return not (slots < ctas <= window * slots)


def _prep_stage(
    q,
    k,
    num_chunks: int,
    num_heads: int,
    arch: str,
    options: ChainOptions,
    *,
    views: bool = False,
) -> str:
    """``prep_tma`` or ``prep`` for one call: the ``prep`` pin, else the rule (``views``: ``prep_tma`` without the
    rule); the plain prep when q or k is not 16-byte aligned or the architecture has no ``prep_tma`` record."""
    if options.prep == "plain" or not _available("prep_tma", arch):
        return "prep"
    if any(t.data_ptr() % TMA_ALIGN_BYTES for t in (q, k)):
        return "prep"
    if (
        options.prep == "tma"
        or views
        or use_prep_tma(num_chunks, num_heads, _tensor_index(q))
    ):
        return "prep_tma"
    return "prep"


def _tma_rows(t: torch.Tensor) -> bool:
    """Whether the ``prep_tma`` tensor map of ``t`` (``[..., T, H, 128]``) reads it in place: a unit channel stride,
    row-major non-overlapping head and token strides that are 16-byte multiples, a 16-byte aligned address, and
    leading axes that fold into the token axis (the binding's ``check_dense_leading_dims``)."""
    if t.dim() < 3 or t.stride(-1) != 1:
        return False
    head, token = t.stride(-2), t.stride(-3)
    if head < t.size(-1) or token < t.size(-2) * head:
        return False
    size = t.element_size()
    if (
        (head * size) % TMA_ALIGN_BYTES
        or (token * size) % TMA_ALIGN_BYTES
        or t.data_ptr() % TMA_ALIGN_BYTES
    ):
        return False
    expected = token
    for axis in range(t.dim() - 4, -1, -1):
        expected *= t.size(axis + 1)
        if t.size(axis) > 1 and t.stride(axis) != expected:
            return False
    return True


def _dense(t: torch.Tensor) -> torch.Tensor:
    """``t`` as contiguous rows with the same bits: ``t`` itself when contiguous, else one copy.  A view with a unit
    innermost stride whose row length, element offset and other strides are multiples of a word of
    :data:`_COPY_WORDS` (a head view of a packed row) is copied as the widest such word: the same bytes in an eighth
    (16-byte words) or a quarter (8-byte words) of the elements of a BF16 copy, which PyTorch's strided copy moves one
    element per thread step.  The word copy is a plain load and store of each word, so every bit pattern survives."""
    if t.is_contiguous():
        return t
    out = torch.empty(t.shape, dtype=t.dtype, device=t.device)
    size = t.element_size()
    if t.dim() > 0 and t.stride(-1) == 1:
        for word in _COPY_WORDS:
            ratio = word.itemsize // size
            if (
                ratio > 1
                and t.size(-1) % ratio == 0
                and t.storage_offset() % ratio == 0
                and all(stride % ratio == 0 for stride in t.stride()[:-1])
            ):
                out.view(word).copy_(t.view(word))
                return out
    out.copy_(t)
    return out


def _row_stride(t: torch.Tensor) -> int:
    """The token-row stride in elements of a ``[..., T, heads, 128]`` tensor that a kernel addresses through that
    stride (its pointer declared strided, its binding checks no layout): a unit channel stride, head stride 128, rows
    at least ``heads x 128`` apart, and leading axes that fold into the token axis (``ValueError`` otherwise).  Every
    backward of a cluster tail asks it four times, so it reads the shape and the strides once each."""
    shape = t.shape
    heads, dim = shape[-2], shape[-1]
    if t.is_contiguous():
        return heads * dim
    stride = t.stride()
    token = expected = stride[-3]
    for axis in range(len(shape) - 4, -1, -1):
        expected *= shape[axis + 1]
        if shape[axis] > 1 and stride[axis] != expected:
            raise ValueError(
                f"the leading axes of a {tuple(shape)} tensor of strides {stride} do not fold into its token axis"
            )
    if stride[-1] != 1 or stride[-2] != dim or token < heads * dim:
        raise ValueError(
            f"a {tuple(shape)} tensor of strides {stride} is not token rows of {heads} heads x {dim} "
            "contiguous elements"
        )
    return token


def _prep_inputs(
    q, k, num_chunks: int, num_heads: int, arch: str, options: ChainOptions
):
    """``(q, k, prep)``: the q / k the prep reads and its stage.  Strided views that ``prep_tma`` reads in place stay
    views when it runs (:data:`VIEWS_TAKE_PREP_TMA`: without the rule; a ``prep`` pin still decides); otherwise q / k
    become contiguous rows (:func:`_dense`) and :func:`_prep_stage` picks the stage, as for contiguous inputs."""
    if not (q.is_contiguous() and k.is_contiguous()) and _tma_rows(q) and _tma_rows(k):
        prep = _prep_stage(
            q, k, num_chunks, num_heads, arch, options, views=VIEWS_TAKE_PREP_TMA
        )
        if prep == "prep_tma":
            return q, k, prep
    q, k = _dense(q), _dense(k)
    return q, k, _prep_stage(q, k, num_chunks, num_heads, arch, options)


# --------------------------------------------------------------------------- #
# two-stream sequence groups
# --------------------------------------------------------------------------- #
# A serial scan launches one CTA per (segment, value head) and holds one CTA per SM, so its last round can leave SMs
# idle.  Split into two contiguous groups of segments, the group that fills whole rounds runs on one stream while the
# other group's scan overlaps the neighbouring stage (forward: wy_tc of the other group; adjoint: the tail of the
# first group) on a second stream.  On an architecture of ``SPLIT_R_FORM`` the adjoint ``dhu_tc2`` pass 2 of a split
# plan beside a group of three's cluster tail (one CTA per (segment, value head), every segment starting from its
# stitched state) is grouped
# the same way over its segments: a sequence's segments are contiguous in the table, so group A covers the chunks
# before the first segment of group B.  A group launch is the full launch restricted to its segments (fwd_ho through a
# slice of its walk, dhu_tc2 through views of the segment table and of dht) or chunks (wy_tc and the fused tails
# through views of the chunk table and of h / dh): every CTA does the work of the CTA of the full launch that covers
# the same (segment or chunk, value head), so the bits are the same.


def _idle_slots(ctas: int, sm_count: int) -> int:
    """The SM slots the last round of ``ctas`` one-per-SM CTAs leaves idle."""
    return -(-ctas // sm_count) * sm_count - ctas


@lru_cache(maxsize=1024)
def two_stream_split(num_segments: int, num_v_heads: int, sm_count: int) -> int | None:
    """The segments ``k`` of a serial scan's full-round group: the fewest idle slots of ``k * num_v_heads`` CTAs over
    ``k`` in ``[1, num_segments)``, ties to the larger ``k``, when that leaves fewer idle slots than the whole scan;
    ``None`` (one stream) otherwise, for fewer than two segments and for a scan of one round.  The adjoint scan runs
    the first ``k`` segments first; the forward scan runs the last ``k`` on the caller's stream."""
    n = int(num_segments)
    if n < 2 or n * num_v_heads <= sm_count:
        return None
    best = min(range(1, n), key=lambda k: (_idle_slots(k * num_v_heads, sm_count), -k))
    if _idle_slots(best * num_v_heads, sm_count) >= _idle_slots(
        n * num_v_heads, sm_count
    ):
        return None
    return best


def sequence_groups(
    meta: ChunkMeta, num_v_heads: int, sm_count: int, *, split_adjoint: bool = False
) -> tuple[StreamGroups | None, StreamGroups | None]:
    """The two-stream groups of a host-built table's forward and adjoint scans (``None``: one stream).  A scan is
    grouped on a serial plan (one segment per sequence, the table's walk) when :func:`two_stream_split` splits it: the
    adjoint scan's group A is its full-round group, the forward scan's group A the rest.  With ``split_adjoint`` (the
    R form of :func:`split_r_form`) the adjoint table of a split plan is grouped over its segments by the same rule:
    group A its first ``k`` segments, the chunks before segment ``k``."""
    if meta.seq_lens is None:
        return None, None
    order = meta.order if meta.order is not None else range(meta.num_seqs)
    chunks = [-(-meta.seq_lens[seq] // CHUNK) for seq in order]
    groups: list[StreamGroups | None] = []
    for segs, adjoint in ((meta.fwd_segments, False), (meta.bwd_segments, True)):
        k = None
        if segs.num_segments == meta.num_seqs and segs.walk is None:
            k = two_stream_split(meta.num_seqs, int(num_v_heads), int(sm_count))
        elif (
            adjoint
            and split_adjoint
            and segs.segment_len > 0
            and segs.num_segments > meta.num_seqs
        ):
            seq_chunk_start = [0]
            for n in chunks:
                seq_chunk_start.append(seq_chunk_start[-1] + n)
            seg_chunk_start = _segment_lists(seq_chunk_start, int(segs.segment_len))[0]
            if len(seg_chunk_start) - 1 == segs.num_segments:
                k = two_stream_split(segs.num_segments, int(num_v_heads), int(sm_count))
            split_chunk = 0 if k is None else seg_chunk_start[k]
            groups.append(
                StreamGroups(k, split_chunk)
                if k is not None and 0 < split_chunk < meta.num_chunks
                else None
            )
            continue
        split = None if k is None else (k if adjoint else meta.num_seqs - k)
        split_chunk = 0 if split is None else sum(chunks[:split])
        groups.append(
            StreamGroups(split, split_chunk)
            if split is not None and 0 < split_chunk < meta.num_chunks
            else None
        )
    return groups[0], groups[1]


_SIDE_STREAMS: dict[tuple[int, int], torch.cuda.Stream] = {}


def _side_stream(device_index: int, main: torch.cuda.Stream) -> torch.cuda.Stream:
    """The side stream of the two-stream groups forked from the caller's stream ``main``: one per (device, caller
    stream), a stream of PyTorch's pool at the greatest priority the device offers, created once.  Callers on
    different streams (another thread, a CUDA graph capture on its capture stream) fork different side streams.

    Every tensor a side stream reads or writes is allocated on the caller's stream before the fork and stays
    referenced by the stage until the join (``main.wait_stream(side)``, also when a launch raises), so the caching
    allocator reuses its memory only after the side stream's work: no ``record_stream``."""
    key = (device_index, main.cuda_stream)
    stream = _SIDE_STREAMS.get(key)
    if stream is None:
        _least, greatest = torch.cuda.Stream.priority_range()
        stream = torch.cuda.Stream(device=device_index, priority=greatest)
        _SIDE_STREAMS[key] = stream
    return stream


# --------------------------------------------------------------------------- #
# forward
# --------------------------------------------------------------------------- #
# The forward WY stage: on an architecture of ``WY_TC2_ARCHS`` every ``wy_tc`` launch (the forward, both sequence
# groups of the P form and the backward's recompute) runs ``wy_tc2`` instead, the same per-element chains, grid and
# outputs (so the same bits) with a 64-column TMEM accumulator and packed shared-memory operands, so five CTAs fit
# on an SM where ``wy_tc`` fits four.
WY_TC2_ARCHS = ("sm_100a", "sm_103a", "sm_107a")


def wy_stage(arch: str) -> str:
    """The stage of the forward WY launches on ``arch``: ``wy_tc2`` on ``WY_TC2_ARCHS``, else ``wy_tc``."""
    return "wy_tc2" if arch in WY_TC2_ARCHS else "wy_tc"


def _wy_outputs(kn, v, total_tokens, num_v_heads):
    w = torch.empty(
        (total_tokens, num_v_heads, HEAD_DIM), dtype=kn.dtype, device=kn.device
    )
    u = torch.empty(v.shape, dtype=v.dtype, device=v.device)
    return w, u


def _launch_wy(
    arch,
    kn,
    v,
    g_cs,
    beta32,
    A,
    meta,
    num_heads,
    num_v_heads,
    total_tokens,
    *,
    recompute,
):
    w, u = _wy_outputs(kn, v, total_tokens, num_v_heads)
    _launch(wy_stage(arch), meta.num_chunks, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start,
            meta.chunk_len, num_heads, num_v_heads, 1 if recompute else 0)  # fmt: skip
    return w, u


def _launch_fwd_h(arch, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, *, store_final_state, table,
                  workspaces, options):  # fmt: skip
    device = kn.device
    h = torch.empty(
        (meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM),
        dtype=kn.dtype,
        device=device,
    )
    v_new = torch.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:
        final_state = torch.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        final_state = workspaces.take(table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32,
                                      device)  # fmt: skip
    if initial_state is not None:
        h0 = initial_state.contiguous()
        if h0.dtype != torch.float32:
            h0 = h0.float()
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    stage, vb, pre, pre_threads = _scan_stages("fwd_h", segs.scan_config, arch)
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        split_stage = f"{stage}_split"
        if pre is None or not _available(split_stage, arch):
            raise NotImplementedError(f"the segmented fwd_h scan needs {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and "
                                      f"{split_stage} for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
        seg_state = workspaces.take(table, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
                seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                num_heads, num_v_heads, use_init, 0, seg_flag, warmup_thresh_log2(options))  # fmt: skip
        _launch(split_stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new,
                final_state, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_init, 1 if store_final_state else 0,
                seg_flag)  # fmt: skip
        return h, v_new, final_state
    _launch(stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
            final_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
            num_heads, num_v_heads, use_init, 1 if store_final_state else 0)  # fmt: skip
    return h, v_new, final_state


# The value-half split of the fused forward scan and output: on an architecture of ``FWD_HO_VH2_ARCHS`` a forward
# scan of one launch (no two-stream groups) whose CTAs fill at most half of the SMs runs ``fwd_ho_vh2``, two CTAs per
# (segment, value head) on twice the grid, each the scan and output of one value half.  Its per-element math is
# fwd_ho's, so the bits are the same; the split shortens each CTA's per-chunk chain where the SMs would sit idle.  The
# deeper split tables of ``CHAIN_PLAN`` price that form (``FWD_HO_VH2_STEP``).
FWD_HO_VH2_ARCHS = ("sm_107a",)


@cache
def fwd_ho_vh2_splits(arch: str) -> bool:
    """Whether the ``fwd_ho_vh2`` record of ``arch`` is the split program (two CTAs per segment and value head).  On
    ``sm_100a`` and ``sm_103a`` the record is the ``fwd_ho`` program itself, which runs on ``fwd_ho``'s grid."""
    split = cake_backend.record(cake_backend.stage_record("fwd_ho_vh2", arch))
    plain = cake_backend.record(cake_backend.stage_record("fwd_ho", arch))
    return split["library"] != plain["library"]


def fwd_ho_vh2_engaged(num_segments: int, num_v_heads: int, sm_count: int) -> bool:
    """Whether a one-launch forward scan of ``num_segments`` segments runs as ``fwd_ho_vh2``: its ``2 x num_segments
    x num_v_heads`` CTAs fit one round of ``sm_count`` SMs (one CTA per SM)."""
    return 2 * int(num_segments) * int(num_v_heads) <= int(sm_count)


_WALKS: dict = {}


def _walk(device, segs: SegmentTable, options: ChainOptions) -> torch.Tensor:
    """The walk of ``fwd_ho`` (CTA group ``b // HV`` runs segment ``walk[b]``): the table's own (longest first) when it
    has one, else the identity, a constant table per (device, segment count) kept between calls under
    ``host_cache``."""
    if segs.walk is not None:
        return segs.walk
    num_segments = segs.num_segments
    key = (_device_index(device), int(num_segments))
    walk = _WALKS.get(key) if options.host_cache else None
    if walk is None:
        walk = torch.arange(int(num_segments), dtype=torch.int32, device=device)
        if options.host_cache:
            _WALKS[key] = walk
    return walk


def _table_rows(state, meta: ChunkMeta):
    """Per-sequence state rows (``initial_state``, ``dht``) in the table's sequence order."""
    if state is None or meta.order_index is None:
        return state
    return state.index_select(0, meta.order_index[0])


def _caller_rows(state, meta: ChunkMeta):
    """Per-sequence state rows (``final_state``, ``dh0``) back in the caller's sequence order."""
    if meta.order_index is None:
        return state
    return state.index_select(0, meta.order_index[1])


def _launch_fwd_ho(arch, qn, kn, w, u, g_cs, initial_state, meta, num_heads, num_v_heads, *, store_final_state, table,
                   workspaces, options, scale, wy=None):  # fmt: skip
    """The fused forward scan and output on the plan's forward segment table: a split plan first runs the pre-scan
    and the stitch (``reverse = 0``) for the segments' start states.  ``wy`` (``(v, beta32, A)``, with
    ``meta.fwd_groups``): ``wy_tc`` has not run; it runs here with the scan as two sequence groups (P form)."""
    device = kn.device
    h = torch.empty(
        (meta.num_chunks, num_v_heads, HEAD_DIM, HEAD_DIM),
        dtype=kn.dtype,
        device=device,
    )
    v_new = torch.empty(u.shape, dtype=u.dtype, device=device)
    o = torch.empty(u.shape, dtype=u.dtype, device=device)
    if store_final_state:
        final_state = torch.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        final_state = workspaces.take(table, "f_zero_state", (0, num_v_heads, HEAD_DIM, HEAD_DIM), torch.float32,
                                      device)  # fmt: skip
    if initial_state is not None:
        h0 = initial_state.contiguous()
        if h0.dtype != torch.float32:
            h0 = h0.float()
        use_init = 1
    else:
        h0, use_init = final_state, 0
    segs = meta.fwd_segments
    seg_start, use_seg = final_state, 0
    if segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        stage, vb, pre, pre_threads = _scan_stages("fwd_h", segs.scan_config, arch)
        if pre is None or not _available("stitch", arch):
            raise NotImplementedError(f"the segmented fwd_ho path needs {stage.replace('fwd_h_r', 'fwd_h_pre_r')} and "
                                      f"stitch for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
        seg_state = workspaces.take(table, "f_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "f_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, kn, w, u, g_cs, h0, h, v_new, final_state,
                seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start,
                num_heads, num_v_heads, use_init, 0, seg_flag, warmup_thresh_log2(options))  # fmt: skip
        seg_start = workspaces.take(table, "f_seg_start", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        _launch("stitch", meta.num_seqs, num_v_heads, STITCH_VSPLIT, seg_state, h0, seg_start, segs.seq_seg_start,
                seg_flag, num_v_heads, use_init, 0)  # fmt: skip
        use_seg = 1
    walk = _walk(device, segs, options)
    if wy is None:
        stage, ctas = "fwd_ho", segs.num_segments * num_v_heads
        if arch in FWD_HO_VH2_ARCHS and fwd_ho_vh2_engaged(
            segs.num_segments, num_v_heads, _device_props(_tensor_index(kn))[0]
        ):
            stage = "fwd_ho_vh2"
            if fwd_ho_vh2_splits(arch):
                ctas = 2 * ctas
        _launch(stage, ctas, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h, v_new, v_new, o, o, final_state,
                meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, walk,
                num_heads, num_v_heads, use_init, 1 if store_final_state else 0, use_seg, float(scale))  # fmt: skip
        return h, v_new, final_state, o
    # P form: the side stream runs wy_tc and fwd_ho of group A (the partial round); the caller's stream runs those of
    # group B (the full rounds) once wy_tc(A) is done, then joins the side stream.  fwd_ho(A) can read w / u rows past
    # the length of a partial chunk (its 64-row tiles) while wy_tc(B) writes them: such rows enter fwd_ho only
    # through selects (v_new rows past the chunk's length are 0), so they do not change a bit.  A kernel that used them
    # arithmetically (a multiply by a zero mask) would make this form nondeterministic.
    v, beta32, A = wy
    groups = cast("StreamGroups", meta.fwd_groups)
    n, na, ca = segs.num_segments, groups.split, groups.split_chunk

    wy_name = wy_stage(arch)

    def wy_tc(c0, c1):
        _launch(wy_name, c1 - c0, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start[c0:c1],
                meta.chunk_len[c0:c1], num_heads, num_v_heads, 0)  # fmt: skip

    def fwd_ho(s0, s1):
        _launch("fwd_ho", (s1 - s0) * num_v_heads, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h, v_new, v_new, o, o,
                final_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, walk[s0:s1], num_heads, num_v_heads, use_init, 1 if store_final_state else 0,
                use_seg, float(scale))  # fmt: skip

    index = _tensor_index(kn)
    main = torch.cuda.current_stream(index)
    side = _side_stream(index, main)
    fork = torch.cuda.Event()
    fork.record(main)
    try:
        with torch.cuda.stream(side):
            side.wait_event(fork)
            wy_tc(0, ca)
            wy_a = torch.cuda.Event()
            wy_a.record(side)
            fwd_ho(0, na)
        main.wait_event(wy_a)
        wy_tc(ca, meta.num_chunks)
        fwd_ho(na, n)
    finally:
        main.wait_stream(side)  # the join
    return h, v_new, final_state, o


def cake_gdn_nondet_train_forward(q, k, v, g, beta, *, meta, arch, scale=None, initial_state=None,
                                  output_final_state=False, normalize_qk=True,
                                  options: ChainOptions | None = None) -> dict[str, Any]:  # fmt: skip
    """The forward chain over flattened ``[T, H, 128]`` / ``[T, HV, 128]`` inputs and one chunk table; returns the
    output and the tape the backward reads (the tape carries this package's tag, ``"mode": "nondet"``).  The forward
    runs fixed-order programs only today: it raises no determinism alert under PyTorch's deterministic algorithms."""
    options = options or ChainOptions()
    num_heads, key_dim = q.shape[-2], q.shape[-1]
    num_v_heads, value_dim = v.shape[-2], v.shape[-1]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("cake_gdn_nondet_train_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    total_tokens = q.numel() // (num_heads * key_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    # q / k stay views where prep_tma reads them in place (the head views of a packed row); v is read as contiguous
    # rows (wy_tc, the backward tail), copied once byte for byte when it is a view
    q, k, prep = _prep_inputs(q, k, meta.num_chunks, num_heads, arch, options)
    v = _dense(v)
    g32 = g.contiguous()
    if g32.dtype != torch.float32:
        g32 = g32.float()
    beta = beta.contiguous()
    beta_bf16 = beta.dtype == torch.bfloat16
    if beta_bf16:
        beta32 = torch.empty(
            (total_tokens, num_v_heads), dtype=torch.float32, device=device
        )
    else:
        beta32 = beta if beta.dtype == torch.float32 else beta.float()
    qn = torch.empty(q.shape, dtype=q.dtype, device=device)
    kn = torch.empty(k.shape, dtype=k.dtype, device=device)
    rstd_q = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    rstd_k = torch.empty((total_tokens, num_heads), dtype=torch.float32, device=device)
    g_cs = torch.empty((total_tokens, num_v_heads), dtype=torch.float32, device=device)
    _launch(prep + ("_bf16beta" if beta_bf16 else ""), meta.num_chunks, num_heads if prep == "prep_tma" else num_v_heads,
            1, q, k, g32, beta, beta32, qn, kn, rstd_q, rstd_k, g_cs, meta.chunk_start, meta.chunk_len, num_heads,
            num_v_heads, 1 if normalize_qk else 0)  # fmt: skip
    A = torch.empty((total_tokens, num_v_heads, CHUNK), dtype=q.dtype, device=device)
    if meta.fwd_groups is None:
        w, u = _launch_wy(
            arch,
            kn,
            v,
            g_cs,
            beta32,
            A,
            meta,
            num_heads,
            num_v_heads,
            total_tokens,
            recompute=False,
        )
        wy = None
    else:  # wy_tc runs with fwd_ho as two sequence groups
        w, u = _wy_outputs(kn, v, total_tokens, num_v_heads)
        wy = (v, beta32, A)
    workspaces = _workspaces(options)
    table = workspaces.table(_tensor_index(kn))
    h, v_new, final_state, o = _launch_fwd_ho(arch, qn, kn, w, u, g_cs, _table_rows(initial_state, meta), meta,
                                              num_heads, num_v_heads, store_final_state=output_final_state,
                                              table=table, workspaces=workspaces, options=options, scale=scale,
                                              wy=wy)  # fmt: skip
    if output_final_state:
        final_state = _caller_rows(final_state, meta)
    return {
        "arch": arch, "normalize_qk": normalize_qk, "num_heads": num_heads, "num_v_heads": num_v_heads,
        "total_tokens": total_tokens, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "g_cs": g_cs, "A": A,
        "w": w, "u": u, "v": v, "beta32": beta32, "meta": meta, "scale": float(scale), "output": o,
        "final_state": final_state if output_final_state else None, "initial_state": initial_state, "h": h,
        "v_new": v_new, "options": options, "mode": _TAPE_MODE,
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def _adjoint_split_walk(split1: AdjointSplit, segs: SegmentTable, meta: ChunkMeta, qn, kn, w, do, g_cs, dht32,
                        use_dht: int, dh, dv2, dh0, has_initial_state: bool, num_heads: int, num_v_heads: int,
                        scale: float, table, workspaces: Workspaces, index: int, device) -> None:  # fmt: skip
    """The adjoint walk of a table's segments as the one-wave split (:class:`AdjointSplit`): wave 1, the last and
    one-piece walks (``dhu_tc2_dvl32_modes``, caller's stream) beside the middle pieces' summaries
    (``dhu_tc2_dvl32_gm``, side stream, joined back; both launches bound before the fork and fired back to back), or
    with ``one_stream`` the summaries then the walks on the caller's stream; the composition of the pieces' end
    adjoints (the stitch with ``reverse``, or ``piece_chain``: each sequence's last piece ends at ``dht``, the piece
    before it at the B half its walk stored, every earlier one at ``M x + B`` of the piece after it); wave 2, the other
    pieces' walks from those end adjoints.  ``dh`` / ``dv2`` receive every chunk's tapes, ``dh0`` each sequence's
    start adjoint (with ``has_initial_state``)."""
    hv = num_v_heads
    seg_state = workspaces.take(table, "b_seg_state", (segs.num_segments, hv, HEAD_DIM, 2 * HEAD_DIM), torch.float32,
                                device)  # fmt: skip
    store0 = 1 if has_initial_state else 0
    common = (meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, num_heads,
              hv)  # fmt: skip

    def bound(stage, entries, count, dht, use_final, store_initial):
        return _bound_launch(stage, count, hv, 1, qn, kn, w, do, g_cs, dht, dh, dv2, dh0, seg_state, *common,
                             use_final, store_initial, scale, entries[0], entries[1])  # fmt: skip

    def walk(stage, entries, count, dht, use_final, store_initial):
        bound(stage, entries, count, dht, use_final, store_initial)()

    if split1.num_summary:
        summaries = bound(
            "dhu_tc2_dvl32_gm", split1.summary, split1.num_summary, dht32, 0, 0
        )
        walks = bound(
            "dhu_tc2_dvl32_modes", split1.walk, split1.num_walk, dht32, use_dht, store0
        )
        if split1.one_stream:
            summaries()
            walks()
        else:
            main = torch.cuda.current_stream(index)
            side = _side_stream(index, main)
            fork = torch.cuda.Event()
            fork.record(main)
            try:
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    summaries()
                walks()
            finally:
                main.wait_stream(side)  # the join
    else:
        walk(
            "dhu_tc2_dvl32_modes", split1.walk, split1.num_walk, dht32, use_dht, store0
        )
    if not split1.num_front:
        return
    dh_end = workspaces.take(
        table,
        "b_dh_end",
        (segs.num_segments, hv, HEAD_DIM, HEAD_DIM),
        torch.float32,
        device,
    )
    if _adjoint_chain_stage(cake_backend.device_arch(index)) == "piece_chain":
        _launch("piece_chain", meta.num_seqs, hv, CHAIN_VSPLIT, seg_state, dht32, dh_end, segs.seq_seg_start,
                split1.seg_flag, hv, use_dht)  # fmt: skip
    else:
        _launch("stitch", meta.num_seqs, hv, STITCH_VSPLIT, seg_state, dht32, dh_end, segs.seq_seg_start,
                split1.seg_flag, hv, use_dht, 1)  # fmt: skip
    walk("dhu_tc2_dvl32_modes", split1.front, split1.num_front, dh_end, 1, store0)


def cake_gdn_nondet_train_backward(
    forward, do, dht=None, *, initial_state=None, dbeta_dtype=None, grad_qkv=None
) -> dict[str, Any]:
    """The backward chain over one forward tape of :func:`cake_gdn_nondet_train_forward`; returns ``dq``, ``dk``,
    ``dv``, ``dg``, ``dbeta`` and ``dinitial_state``.  The tape must carry this package's tag (``"mode": "nondet"``):
    any other tape, such as one of the deterministic package, is refused (``ValueError``).  ``grad_qkv``, three
    tensors of the shapes of ``dq``, ``dk`` and ``dv`` with a unit channel stride, head stride 128 and any token-row
    stride (the head views of one packed gradient row), or a callable that returns them, receives ``dq`` / ``dk`` /
    ``dv`` in place when the call's tail is a cluster tail of ``F32X_TAILS`` (a callable is called then, right before
    the tail, and only then); the returned ``dq`` / ``dk`` / ``dv`` are then those tensors (other tails return tensors
    of their own).  With PyTorch's deterministic algorithms enabled it runs as usual: every launch is fixed-order
    today."""
    if forward.get("mode") != _TAPE_MODE:
        raise ValueError("cake_gdn_nondet_train_backward reads the tape of a non-deterministic forward "
                         "(cake_gdn_nondet_train_forward, tag 'mode': 'nondet'); this tape does not carry that tag "
                         "(a tape of the deterministic package, for example)")  # fmt: skip
    arch = forward["arch"]
    meta = forward["meta"]
    options = forward.get("options") or ChainOptions()
    workspaces = _workspaces(options)
    qn, kn, A, g_cs = forward["qn"], forward["kn"], forward["A"], forward["g_cs"]
    scale = float(forward["scale"])
    v, beta32 = forward["v"], forward["beta32"]
    num_heads, num_v_heads = qn.shape[-2], v.shape[-2]
    form = tail_form(
        num_heads, num_v_heads, options.tail, arch=arch, num_chunks=meta.num_chunks
    )
    total_tokens = qn.numel() // (num_heads * qn.shape[-1])
    device = qn.device
    index = _tensor_index(qn)
    table = workspaces.table(index)
    do = do.contiguous()
    if initial_state is None:
        initial_state = forward.get("initial_state")
    has_initial_state = initial_state is not None
    if forward.get("w") is None or forward.get("u") is None:
        w, u = _launch_wy(
            arch,
            kn,
            v,
            g_cs,
            beta32,
            A,
            meta,
            num_heads,
            num_v_heads,
            total_tokens,
            recompute=True,
        )
    else:
        w, u = forward["w"], forward["u"]
    if forward.get("h") is None or forward.get("v_new") is None:
        h, v_new, _ = _launch_fwd_h(arch, kn, w, u, g_cs, _table_rows(initial_state, meta), meta, num_heads,
                                    num_v_heads,
                                    store_final_state=False, table=table, workspaces=workspaces,
                                    options=options)  # fmt: skip
    else:
        h, v_new = forward["h"], forward["v_new"]
    token_hv = (total_tokens, num_v_heads, HEAD_DIM)
    segs = meta.bwd_segments
    # the one-wave split of the adjoint walk (:class:`AdjointSplit`): its own launches, then the tail as one launch
    split1 = meta.bwd_split
    serial = split1 is None and not (
        segs.segment_len > 0 and segs.num_segments > meta.num_seqs
    )
    use_tc = serial and (
        meta.bwd_tc or _dhu_tc_selected(meta, num_v_heads, index, options)
    )
    # a split plan's pass 2 on dhu_tc2: the fused pre-scan computes the dv_local tiles it reads, no dv_local stage
    split_tc = (
        split1 is None
        and not serial
        and (
            meta.bwd_tc or _sdt_selected(meta, segs, num_v_heads, index, options, arch)
        )
    )
    # two groups (R form, :func:`adjoint_r_form`): the tcgen05 serial scan feeding a fused tail (a group of
    # R_FORM_GROUPS; a group of three's cluster tail on TWO_STREAM_SERIAL_G3_ARCHS), and the dhu_tc2 pass 2 of a split
    # plan feeding that tail (its segments; grouped on an architecture of SPLIT_R_FORM only)
    two_groups = adjoint_r_form(
        arch, num_v_heads // num_heads, form, serial_tc=use_tc, split_tc=split_tc
    )
    groups = meta.bwd_groups if two_groups and split1 is None else None
    if not (use_tc or split_tc or split1 is not None):
        dv_local = workspaces.take(table, "b_dv_local", token_hv, v.dtype, device)
        _launch("dv_local", meta.num_chunks, num_v_heads, 1, qn, kn, do, g_cs, dv_local, meta.chunk_start,
                meta.chunk_len, num_heads, num_v_heads, scale)  # fmt: skip
    dh = workspaces.take(table, "b_dh", h.shape, h.dtype, device)
    dv2 = workspaces.take(table, "b_dv2", token_hv, v.dtype, device)
    if has_initial_state:
        dh0 = torch.empty(
            (meta.num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
    else:
        dh0 = workspaces.take(
            table,
            "b_zero_state",
            (0, num_v_heads, HEAD_DIM, HEAD_DIM),
            torch.float32,
            device,
        )
    if dht is not None:
        dht32 = _table_rows(dht, meta).contiguous()
        if dht32.dtype != torch.float32:
            dht32 = dht32.float()
        use_dht = 1
    else:
        dht32, use_dht = dh0, 0
    scan_stage, vb, pre, pre_threads = _scan_stages("dhu", segs.scan_config, arch)
    tc_stage = dhu_tc2_stage(arch)
    if split1 is not None:
        _adjoint_split_walk(split1, segs, meta, qn, kn, w, do, g_cs, dht32, use_dht, dh, dv2, dh0, has_initial_state,
                            num_heads, num_v_heads, scale, table, workspaces, index, device)  # fmt: skip
    elif use_tc and groups is None:
        # dv_local fused into the tcgen05 serial scan: the same dh / dv2 / dh0 (seg_state is not read)
        _launch(tc_stage, segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, dv2, dh0, dh0,
                meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, num_heads,
                num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    elif segs.segment_len > 0 and segs.num_segments > meta.num_seqs:
        if pre is None or not _available("stitch", arch):
            raise NotImplementedError(f"the segmented dhu scan needs {scan_stage.replace('dhu_r', 'dhu_pre_r')} and "
                                      f"stitch for {arch}")  # fmt: skip
        if segs.segment_len > pre_threads:
            raise ValueError(
                f"segment_len {segs.segment_len} exceeds the {pre_threads} chunks of {pre}"
            )
        seg_state = workspaces.take(table, "b_seg_state", (segs.num_segments, num_v_heads, HEAD_DIM, 2 * HEAD_DIM),
                                    torch.float32, device)  # fmt: skip
        seg_flag = workspaces.take(
            table, "b_seg_flag", (segs.num_segments, num_v_heads), torch.int32, device
        )
        if split_tc:
            # the pre-scan with the dv_local tiles of its chunks computed in the CTA (the same seg_state / seg_flag)
            _launch(sdt_prescan(arch), segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, qn, kn, w, do, g_cs, dht32, dh,
                    dv2, dh0, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                    meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 0, scale, seg_flag,
                    warmup_thresh_log2(options))  # fmt: skip
        else:
            _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32,
                    dh, dv2, dh0, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                    meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 0, scale, seg_flag,
                    warmup_thresh_log2(options))  # fmt: skip
        dh_end = workspaces.take(table, "b_dh_end", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM),
                                 torch.float32, device)  # fmt: skip
        _launch("stitch", meta.num_seqs, num_v_heads, STITCH_VSPLIT, seg_state, dht32, dh_end, segs.seq_seg_start,
                seg_flag, num_v_heads, use_dht, 1)  # fmt: skip
        dht32, use_dht = dh_end, 1
    if split_tc and groups is None:
        # pass 2 on the tcgen05 scan: segment ``seg`` starts from dht[seg], the stitched dh_end[seg]; the first segment
        # of a sequence stores its dh0 (the register pass 2's contract)
        _launch(tc_stage, segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, dv2, dh0, dh0,
                meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq, meta.seq_chunk_start, num_heads,
                num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    elif not (use_tc or split_tc or split1 is not None):
        _launch(scan_stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32, dh,
                dv2, dh0, dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    if form != "unfused":
        # one kernel per (chunk, value head) for dqkwg, the WY pullback and the finalize: a package tail for every
        # value-head group (its workspace bounded by segmented launches), or the cluster kernel of a group of two or
        # three (GROUP_TAILS); the FP32-partial cluster tails write dq / dk / dv through their token-row strides, into
        # the caller's views (``grad_qkv``) when given
        if grad_qkv is not None and form in F32X_TAILS:
            dq, dk, dv = grad_qkv() if callable(grad_qkv) else grad_qkv
        else:
            dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)
            dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
            dv = torch.empty(v.shape, dtype=v.dtype, device=device)
        dbeta = torch.empty(forward.get("beta_shape") or (total_tokens, num_v_heads),
                            dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)  # fmt: skip
        dg = torch.empty(
            forward.get("g_shape") or (total_tokens, num_v_heads),
            dtype=torch.float32,
            device=device,
        )
        normalize = 1 if forward.get("normalize_qk", True) else 0
        rstd_q, rstd_k = forward["rstd_q"], forward["rstd_k"]
        num_chunks = meta.num_chunks
        # the token-row strides of v, dq, dk and dv, which these tails take after their operands
        strides = (
            tuple(_row_stride(t) for t in (v, dq, dk, dv)) if form in F32X_TAILS else ()
        )
        if form in GROUP_TAILS and groups is None:
            # one launch over the whole table, issued directly: the host path of the deterministic package's tail
            _launch(form + ("_bf16" if dbeta_bf16 else ""), num_chunks * num_v_heads, 1, 1, qn, kn, rstd_q, rstd_k,
                    v_new, do, dv2, h, dh, g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, meta.chunk_start,
                    meta.chunk_len, num_heads, num_v_heads, scale, normalize, *strides)  # fmt: skip
            return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta,
                    "dinitial_state": _caller_rows(dh0, meta) if has_initial_state else None}  # fmt: skip

        def rows(a, b):
            # the chunk rows [a, b) of the per-chunk tensors: the tensors themselves for the whole table (no slicing
            # on the host path of a one-launch tail)
            if (a, b) == (0, num_chunks):
                return h, dh, meta.chunk_start, meta.chunk_len
            return h[a:b], dh[a:b], meta.chunk_start[a:b], meta.chunk_len[a:b]

        if form in GROUP_TAILS:
            stage = form + ("_bf16" if dbeta_bf16 else "")

            def tail(c0, c1):
                h_r, dh_r, cs_r, cl_r = rows(c0, c1)
                _launch(stage, (c1 - c0) * num_v_heads, 1, 1, qn, kn, rstd_q, rstd_k, v_new, do, dv2, h_r, dh_r, g_cs,
                        v, A, beta32, dq, dk, dg, dv, dbeta, cs_r, cl_r, num_heads, num_v_heads, scale,
                        normalize, *strides)  # fmt: skip

        else:
            stage = form + ("_bf16" if dbeta_bf16 else "")
            group = num_v_heads // num_heads

            def tail(c0, c1):
                # the workspace of the stream that launches (every tail runs on the caller's stream), and launches of
                # at most its rows in stream order
                ranges, tiles = tail_segments(c1 - c0, num_heads, num_v_heads)
                ws = workspaces.take_zeroed(
                    index,
                    "tail_ws",
                    max(tiles * group, 1),
                    (TAIL_TILE_FLOATS,),
                    torch.float32,
                    device,
                )
                ws_count = workspaces.take_zeroed(
                    index, "tail_count", max(tiles, 1), (), torch.uint32, device
                )
                for s0, s1 in ranges:
                    a, b = c0 + s0, c0 + s1
                    h_r, dh_r, cs_r, cl_r = rows(a, b)
                    _launch(stage, (b - a) * num_v_heads, 1, 1, qn, kn, rstd_q, rstd_k, v_new, do, dv2, h_r, dh_r,
                            g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, ws, ws_count, cs_r, cl_r, num_heads,
                            num_v_heads, scale, normalize, tiles)  # fmt: skip

        if groups is None:
            tail(0, meta.num_chunks)
        else:
            # R form: the caller's stream runs dhu_tc2 of group A (the full rounds; sequences, or the segments of a
            # split plan's pass 2), then the tail of A while the side stream runs dhu_tc2 of group B (the partial
            # round), then joins it and runs the tail of B
            na, ca = groups.split, groups.split_chunk

            def dhu_tc2(s0, s1):
                _launch(tc_stage, s1 - s0, num_v_heads, 1, qn, kn, w, do, g_cs, dht32[s0:s1] if use_dht else dht32,
                        dh, dv2, dh0, dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start[s0:s1 + 1],
                        segs.seg_seq[s0:s1], meta.seq_chunk_start, num_heads, num_v_heads, use_dht,
                        1 if has_initial_state else 0, scale)  # fmt: skip

            main = torch.cuda.current_stream(index)
            side = _side_stream(index, main)
            dhu_tc2(0, na)
            fork = torch.cuda.Event()
            fork.record(main)
            try:
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    dhu_tc2(na, segs.num_segments)
                tail(0, ca)
            finally:
                main.wait_stream(side)  # the join
            tail(ca, meta.num_chunks)
        return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta,
                "dinitial_state": _caller_rows(dh0, meta) if has_initial_state else None}  # fmt: skip
    dq_hv = workspaces.take(table, "b_dq_hv", token_hv, qn.dtype, device)
    dk_hv = workspaces.take(table, "b_dk_hv", token_hv, qn.dtype, device)
    dw = workspaces.take(table, "b_dw", w.shape, w.dtype, device)
    dg1 = workspaces.take(
        table, "b_dg1", (total_tokens, num_v_heads), torch.float32, device
    )
    _launch("dqkwg", meta.num_chunks, num_v_heads, 1, qn, kn, v_new, do, dv2, h, dh, g_cs, dq_hv, dk_hv, dw, dg1,
            meta.chunk_start, meta.chunk_len, num_heads, num_v_heads, scale)  # fmt: skip
    dk2 = workspaces.take(table, "b_dk2", token_hv, qn.dtype, device)
    dv = torch.empty(v.shape, dtype=v.dtype, device=device)
    dbeta = torch.empty(forward.get("beta_shape") or (total_tokens, num_v_heads),
                        dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)  # fmt: skip
    dg2 = workspaces.take(
        table, "b_dg2", (total_tokens, num_v_heads), torch.float32, device
    )
    _launch("wy_bwd_bf16" if dbeta_bf16 else "wy_bwd", meta.num_chunks, num_v_heads, 1, kn, v, A, dw, dv2, g_cs,
            beta32, dk2, dv, dbeta, dg2, meta.chunk_start, meta.chunk_len, num_heads, num_v_heads)  # fmt: skip
    dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)
    dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
    dg = torch.empty(
        forward.get("g_shape") or (total_tokens, num_v_heads),
        dtype=torch.float32,
        device=device,
    )
    sm_count, _smem, _major = _device_props(index)
    fin = forward.get("finalize_stage") or finalize_stage(meta.num_chunks, num_heads, num_v_heads, sm_count,
                                                          options.finalize)  # fmt: skip
    _launch(fin, meta.num_chunks, num_heads, 1, qn, kn,
            forward["rstd_q"], forward["rstd_k"], dq_hv, dk_hv, dk2, dg1, dg2, dq, dk, dg, meta.chunk_start,
            meta.chunk_len, num_heads, num_v_heads, 1 if forward.get("normalize_qk", True) else 0)  # fmt: skip
    return {"dq": dq, "dk": dk, "dv": dv, "dg": dg, "dbeta": dbeta,
            "dinitial_state": _caller_rows(dh0, meta) if has_initial_state else None}  # fmt: skip


# --------------------------------------------------------------------------- #
# autograd wrapper with FLA's signature (no context parallelism)
# --------------------------------------------------------------------------- #
_META_FRONT: dict = {}
_META_FRONT_LIMIT = 256
_OFFSET_DTYPES = (torch.int32, torch.int64)
# the checked tensors in check order; the last two are optional (None skips them)
_INPUT_NAMES = ("q", "k", "v", "g", "beta", "initial_state", "cu_seqlens")
_OPTIONAL_INPUT = 5


def _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu) -> None:
    """The API-boundary checks of :func:`cake_gdn_nondet`: devices, shapes and the offsets' form, from tensor
    metadata only (no device synchronization; the values of ``cu_seqlens_cpu`` are checked when its layout is first
    planned, those of the device ``cu_seqlens`` not at all).  FLA's ``chunk_gated_delta_rule`` checks the head
    counts, the batch of a packed call and the rows of ``initial_state``; the rest is this backend's contract.

    Every call pays these checks, so they read the cheapest metadata: ``is_cuda`` and ``get_device()`` instead of
    ``torch.device`` objects (made only for an error message), each shape once."""
    index = -1
    for i, t in enumerate((q, k, v, g, beta, initial_state, cu_seqlens)):
        if t is None and i >= _OPTIONAL_INPUT:
            continue
        if not isinstance(t, torch.Tensor) or not t.is_cuda:
            raise ValueError(f"{_INPUT_NAMES[i]} must be a CUDA tensor")
        if i == 0:
            index = t.get_device()
        elif t.get_device() != index:
            raise ValueError(f"{_INPUT_NAMES[i]} is on {t.device}, q on {q.device}")
    q_shape, v_shape = q.shape, v.shape
    if len(q_shape) != 4 or q_shape[3] != HEAD_DIM:
        raise ValueError(f"q must be [B, T, H, {HEAD_DIM}]; got {tuple(q_shape)}")
    if k.shape != q_shape:
        raise ValueError(
            f"k must have q's shape {tuple(q_shape)}; got {tuple(k.shape)}"
        )
    if (
        len(v_shape) != 4
        or v_shape[0] != q_shape[0]
        or v_shape[1] != q_shape[1]
        or v_shape[3] != HEAD_DIM
    ):
        raise ValueError(
            f"v must be [{q_shape[0]}, {q_shape[1]}, HV, {HEAD_DIM}]; got {tuple(v_shape)}"
        )
    num_heads, num_v_heads = q_shape[2], v_shape[2]
    if num_heads == 0 or num_v_heads == 0 or num_v_heads % num_heads != 0:
        raise ValueError(f"the value heads (HV={num_v_heads}) must be a positive multiple of the q / k heads "
                         f"(H={num_heads})")  # fmt: skip
    gate_shape = v_shape[:3]
    if g.shape != gate_shape:
        raise ValueError(f"g must be {tuple(gate_shape)}; got {tuple(g.shape)}")
    if beta.shape != gate_shape:
        raise ValueError(f"beta must be {tuple(gate_shape)}; got {tuple(beta.shape)}")
    if cu_seqlens is None:
        num_seqs = q_shape[0]
    else:
        offsets = cu_seqlens.numel()
        if (
            cu_seqlens.dim() != 1
            or offsets < 2
            or cu_seqlens.dtype not in _OFFSET_DTYPES
        ):
            raise ValueError(f"cu_seqlens must be a 1-D int32 / int64 tensor of at least two offsets; got "
                             f"{cu_seqlens.dtype} {tuple(cu_seqlens.shape)}")  # fmt: skip
        if q_shape[0] != 1:
            raise ValueError("cu_seqlens requires a leading batch dimension of 1")
        num_seqs = offsets - 1
        if cu_seqlens_cpu is not None:
            if isinstance(cu_seqlens_cpu, torch.Tensor):
                if (
                    cu_seqlens_cpu.dim() != 1
                    or cu_seqlens_cpu.dtype not in _OFFSET_DTYPES
                ):
                    raise ValueError(f"cu_seqlens_cpu must be a 1-D int32 / int64 tensor; got {cu_seqlens_cpu.dtype} "
                                     f"{tuple(cu_seqlens_cpu.shape)}")  # fmt: skip
                count = cu_seqlens_cpu.numel()
            else:
                count = len(cu_seqlens_cpu)
            if count != offsets:
                raise ValueError(
                    f"cu_seqlens_cpu has {count} offsets, cu_seqlens {offsets}"
                )
    if initial_state is not None:
        state_shape = (num_seqs, num_v_heads, HEAD_DIM, HEAD_DIM)
        if initial_state.shape != state_shape:
            raise ValueError(f"initial_state must be {state_shape} (one state per sequence); got "
                             f"{tuple(initial_state.shape)}")  # fmt: skip


def _host_offsets(cu_seqlens_cpu) -> tuple[int, ...]:
    """The host offsets as a tuple of ints (one ``tolist`` for a tensor, which may also live on the device)."""
    if isinstance(cu_seqlens_cpu, torch.Tensor):
        return tuple(cu_seqlens_cpu.tolist())
    return tuple(int(x) for x in cu_seqlens_cpu)


def _check_offsets(layout: tuple[int, ...], total_tokens: int) -> None:
    """The host offsets of a packed call start at 0, do not decrease and end at the tokens of ``q`` (an empty
    sequence is allowed)."""
    if (
        layout[0] != 0
        or layout[-1] != total_tokens
        or any(b < a for a, b in zip(layout[:-1], layout[1:], strict=True))
    ):
        raise ValueError(f"cu_seqlens_cpu must start at 0, not decrease and end at the {total_tokens} tokens of q; "
                         f"got {list(layout[:4])}{' ...' if len(layout) > 4 else ''} ending at {layout[-1]}")  # fmt: skip


def host_plans(
    seq_lens,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions | None = None,
    *,
    total_tokens: int | None = None,
    group: int | None = None,
) -> tuple[tuple[ScanPlan, ScanPlan], bool]:
    """The forward and adjoint tables of a host layout (``seq_lens``, tokens per sequence) on device ``device_index``
    and whether the adjoint table is priced for ``dhu_tc2`` (``ChunkMeta.bwd_tc``): the planner's tables, the adjoint
    table priced for ``dhu_tc2`` (``TC_PLAN``) and the deeper split tables of ``CHAIN_PLAN``; plan pins (and a
    ``dhu_scan`` pin) keep the planner's tables.  ``group`` is the call's value-head group ``HV / H`` (``None``: a
    group the R form of :func:`split_r_form` does not serve), the one input besides the layout that changes a table:
    on an architecture of ``SPLIT_R_FORM`` a group of three keeps a planner's split adjoint table that the R form
    groups.  Host work only."""
    return _host_plans(
        seq_lens,
        num_v_heads,
        device_index,
        options,
        total_tokens=total_tokens,
        group=group,
    )[:2]


def _host_plans(
    seq_lens,
    num_v_heads: int,
    device_index: int,
    options: ChainOptions | None = None,
    *,
    total_tokens: int | None = None,
    group: int | None = None,
) -> tuple[tuple[ScanPlan, ScanPlan], bool, bool]:
    """:func:`host_plans` and whether the adjoint table is the planner's table kept for the R form
    (:func:`split_r_form`)."""
    options = options or ChainOptions()
    seq_lens = tuple(int(n) for n in seq_lens)
    index = int(device_index)
    if total_tokens is None:
        total_tokens = sum(seq_lens)
    plans = scan_plans(int(total_tokens), len(seq_lens), int(num_v_heads), index, seq_lens,
                       scan_config=options.scan_config, segment_chunks=options.segment_chunks)  # fmt: skip
    bwd_tc = kept = False
    if TC_PLAN and options.pins == (None, None) and options.dhu_scan is None:
        chunks = tuple(-(-int(n) // CHUNK) for n in seq_lens)
        bwd = plans[1]
        # a serial planner table that the two-stream rule groups keeps its form (the R form overlaps its scan with the
        # tail, which the pricing does not credit), with one stream too: both stream counts plan the same tables; so
        # does a split planner table of the R form beside a group of three's cluster tail (SPLIT_R_FORM)
        kept = group is not None and split_r_form(
            chunks, int(num_v_heads), group, index, bwd, options
        )
        grouped = kept or (
            (bwd.segment_len <= 0 or max(chunks) <= bwd.segment_len)
            and two_stream_split(
                len(chunks), int(num_v_heads), _device_props(int(index))[0]
            )
            is not None
        )
        priced = (
            None
            if grouped
            else tc_plan_pricing(chunks, int(num_v_heads), int(index), bwd)
        )
        if priced is not None and priced["select"]:
            plans, bwd_tc = (
                (plans[0], ScanPlan(SDT_CONFIG, priced["segment_len"])),
                True,
            )
        if cake_backend.device_arch(index) in CHAIN_PLAN:
            # deeper split tables priced for fwd_ho / dhu_tc2 (CHAIN_PLAN); a serial table that the two-stream rule
            # groups keeps its form in each direction, as above
            sm_count = _device_props(index)[0]
            stream_split = two_stream_split(len(chunks), int(num_v_heads), sm_count)
            for kind in ("fwd", "bwd"):
                plan = plans[0] if kind == "fwd" else plans[1]
                serial = plan.segment_len <= 0 or max(chunks) <= plan.segment_len
                if (serial and stream_split is not None) or (kind == "bwd" and kept):
                    continue
                deeper = chain_plan_pricing(kind, chunks, int(num_v_heads), index, plan)
                if not deeper["select"]:
                    continue
                table = ScanPlan(SDT_CONFIG, deeper["segment_len"])
                if kind == "fwd":
                    plans = (table, plans[1])
                else:
                    plans, bwd_tc = (plans[0], table), True
    if (
        cake_backend.device_arch(index) in ADJOINT_SPLIT_ARCHS
        and options.pins == (None, None)
        and (options.dhu_scan is None)
    ):
        # the one-wave split of the adjoint walk or the serial table, priced at their full cost: never a pre-scan's
        # split table, whose cost follows the gates
        length = _adjoint_split_length(seq_lens, int(num_v_heads), index, options)
        bwd = plans[1]
        if length is not None:
            plans, bwd_tc, kept = (plans[0], ScanPlan(SDT_CONFIG, length)), False, False
        elif (
            bwd.segment_len > 0
            and max(-(-n // CHUNK) for n in seq_lens) > bwd.segment_len
        ):
            serial = (
                bwd.config
                if tuple(bwd.config) in REG_SCAN_CONFIGS
                else REG_SCAN_CONFIGS[0]
            )
            plans, bwd_tc, kept = (plans[0], ScanPlan(serial, 0)), False, False
    return plans, bwd_tc, kept


def _chunk_meta(
    cu_seqlens,
    cu_seqlens_cpu,
    batch,
    seq_len,
    index: int,
    num_v_heads,
    options: ChainOptions,
    group: int | None = None,
) -> ChunkMeta:
    """The chunk and segment tables of one call on device ``index`` (``group``: its value-head group ``HV / H``): a
    ``host_cache`` hit on the layout's key, else planned (the layout's sequence lengths are derived only then)."""
    total_tokens = int(batch) * int(seq_len)
    if cu_seqlens is None:
        layout: tuple[Any, ...] = ("equal", int(batch), int(seq_len))
    elif batch != 1:
        raise ValueError("cu_seqlens requires a leading batch dimension of 1")
    elif cu_seqlens_cpu is None:
        num_seqs = int(cu_seqlens.numel()) - 1
        return build_chunk_meta_from_cu_seqlens(
            cu_seqlens,
            total_tokens,
            scan_plans=scan_plans(total_tokens, num_seqs, int(num_v_heads), int(index),
                                  scan_config=options.scan_config, segment_chunks=options.segment_chunks),
        )  # fmt: skip
    else:
        layout = _host_offsets(cu_seqlens_cpu)
    key = (
        layout,
        int(num_v_heads),
        int(index),
        total_tokens,
        options.pins,
        options.dhu_scan,
        options.lpt,
        options.streams,
        TC_PLAN,
        tuple(sorted(CHAIN_PLAN)),
        FWD_HO_VH2_ARCHS,
        FWD_HO_VH2_STEP,
        group,
        options.tail,
        SPLIT_R_FORM,
        tuple(sorted(ADJOINT_SPLIT_ARCHS)),
    )
    if options.host_cache:
        meta = _META_FRONT.get(key)
        if meta is not None:
            return meta
    if cu_seqlens is None:
        seq_lens = (int(seq_len),) * int(batch)
    else:
        # a layout reaches the table cache only after this check, so a cache hit is a checked layout
        _check_offsets(layout, total_tokens)
        seq_lens = tuple(b - a for a, b in zip(layout[:-1], layout[1:], strict=True))
    plans, bwd_tc, _kept = _host_plans(seq_lens, int(num_v_heads), int(index), options, total_tokens=total_tokens,
                                       group=group)  # fmt: skip
    meta = _host_chunk_meta(seq_lens, plans, int(index), options.lpt)
    if bwd_tc:
        meta = replace(meta, bwd_tc=True)
    split_len = _adjoint_split_length(seq_lens, int(num_v_heads), int(index), options)
    if split_len is not None and int(plans[1].segment_len) == split_len:
        # wave 1 on one stream where the pricing says so or where the caller keeps every launch on its stream
        one_stream = options.streams == 1 or bool(
            _adjoint_split_priced(
                seq_lens, num_v_heads, index, cake_backend.device_arch(int(index))
            )["one_stream"]
        )
        meta = replace(
            meta,
            bwd_split=_host_adjoint_split(
                seq_lens,
                split_len,
                int(num_v_heads),
                int(index),
                options.lpt,
                one_stream,
            ),
        )
    if options.streams == 2:
        split_adjoint = _split_r_groups(
            cake_backend.device_arch(int(index)), group, options
        )
        fwd_groups, bwd_groups = sequence_groups(
            meta,
            int(num_v_heads),
            _device_props(int(index))[0],
            split_adjoint=split_adjoint,
        )
        if meta.bwd_split is not None:
            bwd_groups = None  # the split's two waves fill the device; the tail runs as one launch after them
        if fwd_groups is not None or bwd_groups is not None:
            meta = replace(meta, fwd_groups=fwd_groups, bwd_groups=bwd_groups)
    if options.host_cache:
        if len(_META_FRONT) >= _META_FRONT_LIMIT:
            del _META_FRONT[next(iter(_META_FRONT))]
        _META_FRONT[key] = meta
    return meta


def _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                      use_qk_l2norm_in_kernel, options):  # fmt: skip
    """The forward of :class:`CakeGDNNondetFunction` and :class:`CakeGDNNondetPackedFunction` over q / k / v (the
    inputs, or the head views of the packed row): the chain, the saved tape and ``ctx``'s fields."""
    batch, seq_len, num_heads, key_dim = q.shape
    num_v_heads = v.shape[2]
    index = _tensor_index(q)
    arch = cake_backend.device_arch(index)
    meta = _chunk_meta(cu_seqlens, cu_seqlens_cpu, batch, seq_len, index, num_v_heads, options,
                       num_v_heads // num_heads)  # fmt: skip
    fwd = cake_gdn_nondet_train_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale,
                                        initial_state=initial_state, output_final_state=output_final_state,
                                        normalize_qk=use_qk_l2norm_in_kernel, options=options)  # fmt: skip
    state_tensors = (
        (fwd["w"], fwd["u"], fwd["h"], fwd["v_new"]) if options.save_state else ()
    )
    ctx.save_for_backward(fwd["qn"], fwd["kn"], fwd["rstd_q"], fwd["rstd_k"], fwd["v"], fwd["g_cs"],
                          fwd["beta32"], fwd["A"], initial_state, *state_tensors)  # fmt: skip
    ctx.saved_state_tensors = bool(state_tensors)
    ctx.meta, ctx.arch, ctx.scale = meta, arch, float(scale)
    ctx.grad_shapes = (g.shape, beta.shape)
    ctx.normalize_qk = bool(use_qk_l2norm_in_kernel)
    ctx.has_initial_state_input = initial_state is not None
    ctx.dtypes = (g.dtype, beta.dtype)
    ctx.options = options
    # the forward tape's tag; the backward reads only a tape of its own
    ctx.mode = fwd["mode"]
    output = fwd["output"]
    if output.dtype != q.dtype:
        output = output.to(q.dtype)
    return output, (fwd["final_state"] if output_final_state else None)


def _function_backward(ctx, do, dht, grad_qkv=None):
    """The backward of both autograd functions: the gradients of q, k, v, g, beta and the initial state (``grad_qkv``:
    see :func:`cake_gdn_nondet_train_backward`)."""
    qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, *state_tensors = (
        ctx.saved_tensors
    )
    g_shape, beta_shape = ctx.grad_shapes
    g_dtype, beta_dtype = ctx.dtypes
    tape = {"arch": ctx.arch, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "v": v_flat, "g_cs": g_cs,
            "beta32": beta32, "A": A, "meta": ctx.meta, "scale": ctx.scale, "normalize_qk": ctx.normalize_qk,
            "g_shape": g_shape, "beta_shape": beta_shape, "options": ctx.options, "mode": ctx.mode}  # fmt: skip
    if ctx.saved_state_tensors:
        tape["w"], tape["u"], tape["h"], tape["v_new"] = state_tensors
    grads = cake_gdn_nondet_train_backward(
        tape,
        do.contiguous(),
        dht,
        initial_state=saved_state,
        dbeta_dtype=beta_dtype,
        grad_qkv=grad_qkv,
    )
    dg = grads["dg"]
    if dg.dtype != g_dtype:
        dg = dg.to(g_dtype)
    dbeta = grads["dbeta"]
    if dbeta.dtype != beta_dtype:
        dbeta = dbeta.to(beta_dtype)
    dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
    return grads["dq"], grads["dk"], grads["dv"], dg, dbeta, dh0


class CakeGDNNondetFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                use_qk_l2norm_in_kernel, options):  # fmt: skip
        return _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
                                 cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip

    @staticmethod
    def backward(ctx, do, dht):
        dq, dk, dv, dg, dbeta, dh0 = _function_backward(ctx, do, dht)
        return dq, dk, dv, dg, dbeta, None, dh0, None, None, None, None, None


# --------------------------------------------------------------------------- #
# q / k / v as the head views of one packed row: one autograd input, one packed gradient row
# --------------------------------------------------------------------------- #
# The customer layout (FlashInfer #7074): one BF16 row [..., T, (2 H + HV) x 128] per token, q / k / v its head views
# (``row.split((H * 128, H * 128, HV * 128), -1)``, each viewed as [B, T, heads, 128]).  Run on the views, autograd
# would return dq / dk / dv as three tensors and the split's backward would copy them into a new row gradient.  With
# the packed row as the one input instead, the backward allocates the row gradient and the cluster tails write dq /
# dk / dv straight into its head views through their token-row strides: no concatenation.  The views are recognized
# from their metadata alone (:func:`_packed_row`); any other layout takes :class:`CakeGDNNondetFunction`, as does a
# call whose tail is not a cluster tail of ``F32X_TAILS`` or that records no gradient for the row.
PACKED_ROW = True


def _packed_row(q, k, v) -> torch.Tensor | None:
    """The packed row whose head views ``q``, ``k`` and ``v`` are, or ``None``: their common base, a contiguous BF16
    tensor of ``B x T`` rows of ``(2 H + HV) x 128`` elements that requires grad, q / k / v at its row offsets 0,
    ``H x 128`` and ``2 H x 128`` with its row stride, head stride 128 and unit channel stride, each requiring grad
    through a view of it.  Every call with grad asks it, so it reads each tensor's shape, strides and offset once
    (tuples compared whole; the strides of size-one axes, which any value may take, only when the whole compare
    fails)."""
    base = q._base
    if base is None or k._base is not base or v._base is not base:
        return None
    if not (
        base.requires_grad and q.requires_grad and k.requires_grad and v.requires_grad
    ):
        return None
    if base.dtype != torch.bfloat16:
        return None
    shape, v_shape = q.shape, v.shape
    if len(shape) != 4 or len(v_shape) != 4:
        return None
    batch, seq_len, num_heads, dim = shape
    num_v_heads = v_shape[2]
    width = (2 * num_heads + num_v_heads) * dim
    base_shape = base.shape
    if (
        not base_shape
        or base_shape[-1] != width
        or k.shape != shape
        or v_shape != (batch, seq_len, num_v_heads, dim)
        or base.numel() != batch * seq_len * width
        or not base.is_contiguous()
    ):
        return None
    offset = base.storage_offset()
    if (
        q.storage_offset() != offset
        or k.storage_offset() != offset + num_heads * dim
        or v.storage_offset() != offset + 2 * num_heads * dim
    ):
        return None
    strides = (seq_len * width, width, dim, 1)
    for t in (q, k, v):
        stride = t.stride()
        if stride != strides and any(
            size > 1 and got != want
            for size, got, want in zip(t.shape, stride, strides, strict=True)
        ):
            return None
    return base


class CakeGDNNondetPackedFunction(torch.autograd.Function):
    """:class:`CakeGDNNondetFunction` with the packed row as its one q / k / v input (see :func:`_packed_row`): the
    same forward on the caller's head views of the row, and a backward that returns the row's gradient, written by
    the tail.  ``views`` is the tuple of those views: a container, so autograd does not take them as inputs, and the
    forward reads them as they are instead of viewing the row again."""

    @staticmethod
    def forward(ctx, row, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                use_qk_l2norm_in_kernel, options, views):  # fmt: skip
        q, k, v = views
        ctx.row_shape = row.shape
        ctx.view_shapes = (q.shape, v.shape)
        return _function_forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens,
                                 cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip

    @staticmethod
    def backward(ctx, do, dht):
        # the gradient row and its q / k / v head views (one strided view each, at the row offsets of the forward's
        # views), which the tail writes through their token-row strides; made when the chain reaches the tail, so the
        # host work runs while the adjoint scan's kernels do instead of before the first of them
        made = []

        def row_views():
            q_shape, v_shape = ctx.view_shapes
            seq_len, num_heads, dim = q_shape[1], q_shape[2], q_shape[3]
            width = ctx.row_shape[-1]
            strides = (seq_len * width, width, dim, 1)
            d_row = torch.empty(ctx.row_shape, dtype=torch.bfloat16, device=do.device)
            views = (
                d_row.as_strided(q_shape, strides, 0),
                d_row.as_strided(q_shape, strides, num_heads * dim),
                d_row.as_strided(v_shape, strides, 2 * num_heads * dim),
            )
            made.append((d_row, views))
            return views

        dq, dk, dv, dg, dbeta, dh0 = _function_backward(ctx, do, dht, row_views)
        if not made:
            # a tail that does not write the row's views (not one of F32X_TAILS)
            row_views()
        d_row, views = made[0]
        for view, grad in zip(views, (dq, dk, dv), strict=True):
            if grad is not view:
                view.copy_(grad)
        return d_row, dg, dbeta, None, dh0, None, None, None, None, None, None


def cake_gdn_nondet(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    use_qk_l2norm_in_kernel=False,
    *,
    scan_config: tuple[int, int] | None = None,
    segment_chunks: int | None = None,
    finalize: str | None = None,
    warmup_log2: float | None = None,
    save_state: bool = True,
    host_cache: bool = True,
    dhu_scan: str | None = None,
    prep: str | None = None,
    lpt: bool = True,
    streams: int = 2,
    tail: str | None = None,
):
    """Non-deterministic chunked gated delta rule with FLA's signature (head dimension 128, ``HV`` a multiple of ``H``).

    ``q`` / ``k`` are ``[B, T, H, 128]`` and ``v`` ``[B, T, HV, 128]`` bfloat16; ``g`` is ``[B, T, HV]`` (computed in
    float32); ``beta`` is ``[B, T, HV]`` float32 or bfloat16 (its gradient comes back in its dtype).  Packed
    variable-length batches pass ``B = 1`` and ``cu_seqlens`` on the device; ``cu_seqlens_cpu`` (the same offsets on
    the host) selects the host-table route, without it the ``meta`` kernel builds the tables on the device.  Returns
    ``(o, final_state)``; ``final_state`` is ``None`` unless ``output_final_state``.  The keyword-only arguments are
    the plan pins of :class:`ChainOptions`.

    The inputs are checked before the first launch (``ValueError``): every tensor on q's CUDA device, ``k`` of q's
    shape, ``v`` / ``g`` / ``beta`` of q's tokens, ``HV`` a positive multiple of ``H``, ``initial_state`` ``[N, HV,
    128, 128]``, ``cu_seqlens`` 1-D int32 / int64 with a batch of 1, ``cu_seqlens_cpu`` as many offsets, from 0, not
    decreasing, to the tokens of ``q``.  The device route trusts the values of ``cu_seqlens`` (reading them would
    synchronize): the same contract, unchecked.

    The package is the non-deterministic route: the outputs and gradients match the deterministic package within its
    accuracy budgets, not bitwise.  The mode is in the name; there is no ``deterministic`` keyword.  Every launch is
    fixed-order today, so PyTorch's deterministic algorithms (``torch.use_deterministic_algorithms(True)``, with or
    without ``warn_only``) change nothing: no alert, no warning, no fallback.  The contract does not promise run-to-run
    bits.  ``tail`` pins the backward tail (:class:`ChainOptions`).

    When q, k and v are the head views of one packed BF16 row that requires grad (``row.split((H * 128, H * 128,
    HV * 128), -1)``, each viewed as ``[B, T, heads, 128]``; :func:`_packed_row`) and the backward tail is a cluster
    tail of ``F32X_TAILS``, the row is the autograd input (:class:`CakeGDNNondetPackedFunction`): the backward writes
    dq / dk / dv into one packed gradient row, the same bits as on the views.
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise NotImplementedError("q/k/v must be bfloat16")
    _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
    if (
        scan_config is None and segment_chunks is None and finalize is None and warmup_log2 is None
        and save_state is True and host_cache is True and dhu_scan is None and prep is None and lpt is True
        and type(streams) is int and streams == 2 and tail is None
    ):  # fmt: skip
        options = _DEFAULT_OPTIONS  # no pins: the validation of make_options has nothing to check
    else:
        options = make_options(scan_config=scan_config, segment_chunks=segment_chunks, finalize=finalize,
                               warmup_log2=warmup_log2, save_state=save_state, host_cache=host_cache,
                               dhu_scan=dhu_scan, prep=prep, lpt=lpt, streams=streams, tail=tail)  # fmt: skip
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    if PACKED_ROW and torch.is_grad_enabled():
        row = _packed_row(q, k, v)
        if row is not None:
            arch = cake_backend.device_arch(_tensor_index(q))
            if tail_form(q.shape[2], v.shape[2], options.tail, arch=arch) in F32X_TAILS:
                return CakeGDNNondetPackedFunction.apply(row, g, beta, float(scale), initial_state, output_final_state,
                                                         cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, options,
                                                         (q, k, v))  # fmt: skip
    return CakeGDNNondetFunction.apply(q, k, v, g, beta, float(scale), initial_state, output_final_state, cu_seqlens,
                                       cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip
