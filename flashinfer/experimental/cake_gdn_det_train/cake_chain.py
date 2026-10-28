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

Host plan and autograd chain of the deterministic GDN chunked training
(head dimension 128, chunk 64): the chunk and segment tables, the scan
configuration and segment planner, the workspace pool, the forward and backward
launch order, the adjoint-scan and finalize selection and the autograd wrapper
with FLA's ``chunk_gated_delta_rule`` signature (no context parallelism).

The forward runs the prep (``prep_tma``, the q / k tiles staged by TMA and one
CTA per (chunk, key head), unless its dispatch rule :func:`use_prep_tma` keeps
the plain prep; both give the same bits), the tcgen05 WY stage and the fused
tcgen05 scan and output (``fwd_ho``; a split plan first runs the pre-scan and
the stitch for the segment start states).  The backward runs ``dv_local`` and
the register scan of the adjoint state (with the pre-scan and the stitch on a
split plan), or on a serial plan the tcgen05 adjoint scan with ``dv_local``
fused (``dhu_tc2``) when its selection rule (:func:`dhu_tc_pricing`) prices it
at most ``DHU_TC_GAIN`` of the register path; then, for a value-head group of
two, the fused tcgen05 tail (``tail_g2``), else ``dqkwg``, the WY pullback and
the finalize.  A host-built table lists the sequences longest first
(:func:`lpt_sequence_order`; every chunk keeps its token offset, the state rows
are gathered into that order and the final state and its gradient scattered
back) and ``fwd_ho`` walks the forward segments longest first
(:func:`lpt_walk`): each CTA computes what it computes in the caller's order.
Where a serial scan leaves its last round partly idle, a host-built table runs
it as two contiguous sequence groups (:func:`sequence_groups`), one on a
high-priority side stream forked from the caller's stream and joined back
before the stage returns: the forward's ``wy_tc`` and ``fwd_ho`` of the partial
group overlap the full rounds of the other, and the adjoint ``dhu_tc2`` of the
partial group overlaps the fused tail of the other.

Every launch is a record of :mod:`.cake_jit` selected by its stage and the
device's architecture (:func:`.cake_backend.launch_stage`); the stages a plan
may use are the ones the registry delivers for that architecture.  The
planner's per-configuration facts (the register scans' shared-memory figure,
the chunks one pre-scan CTA covers) are the records' ``scan`` constants; the
``dhu_tc2`` selection constants and the ``prep_tma`` dispatch window are those
records' ``select`` tables.

The chain is deterministic: no atomics, no data-dependent ordering, and the two
groups of a two-stream stage write disjoint outputs, so the outputs and
gradients are bitwise run to run for one layout and one plan, with one or two
streams.
The plan pins that tests and tuning use are explicit keyword arguments of
:func:`chunk_gated_delta_rule` (``scan_config``, ``segment_chunks``,
``finalize``, ``warmup_log2``, ``dhu_scan``, ``prep``, ``lpt``, ``streams``);
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
    "ChainOptions",
    "ChunkGatedDeltaRuleFunction",
    "ChunkMeta",
    "ScanPlan",
    "SegmentTable",
    "StageSet",
    "StreamGroups",
    "build_chunk_meta",
    "build_chunk_meta_from_cu_seqlens",
    "choose_scan_plan",
    "chunk_gated_delta_rule",
    "dhu_tc_pricing",
    "finalize_stage",
    "gdn_chunk_backward",
    "gdn_chunk_forward",
    "lpt_sequence_order",
    "lpt_walk",
    "make_options",
    "scan_plans",
    "sequence_groups",
    "stage_set",
    "two_stream_split",
    "use_prep_tma",
]

CHUNK: int = int(cake_jit.CHUNK)
HEAD_DIM: int = int(cake_jit.HEAD_DIM)

# Register-scan configurations ``(value block, key split)`` in the planner's preference order, and the ones with a
# segment-parallel route (pre-scan + split scan for the forward, pre-scan + stitch for the adjoint).
REG_SCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (32, 1), (64, 1))
PRESCAN_CONFIGS: tuple[tuple[int, int], ...] = ((32, 2), (64, 1))
SCAN_KINDS = ("fwd_h", "dhu")
STITCH_VSPLIT = 4  # CTAs per (sequence, value head) of the stitch launch

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
_DHU_TC2_ARGS = ("qn_tma", "kn_tma", "w_tma", "do_tma") + tuple(
    a for a in _DHU_ARGS[4:] if a != "dv_local"
)
_DQKWG_ARGS = ("qn", "kn", "v_new", "do", "dv2", "h", "dh", "g_cs", "dq_out", "dk_out", "dw_out", "dg_out",
               "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale")  # fmt: skip
_WY_BWD_ARGS = ("kn", "v", "A", "dw", "du", "g_cs", "beta", "dk2_out", "dv_out", "dbeta_out", "dg2_out",
                "chunk_start", "chunk_len", "num_heads", "num_v_heads")  # fmt: skip
_FINALIZE_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "dq_hv", "dk_hv", "dk2", "dg1", "dg2", "dq_out", "dk_out", "dg_out",
                  "chunk_start", "chunk_len", "num_heads", "num_v_heads", "normalize_qk")  # fmt: skip
_TAIL_ARGS = ("qn", "kn", "rstd_q", "rstd_k", "v_new", "do", "dv2", "h", "dh", "g_cs", "v", "A", "beta", "dq_out",
              "dk_out", "dg_out", "dv_out", "dbeta_out", "chunk_start", "chunk_len", "num_heads", "num_v_heads", "scale",
              "normalize_qk")  # fmt: skip
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
    "fwd_ho": _FWD_HO_ARGS,
    "stitch": _STITCH_ARGS,
    "dv_local": _DV_LOCAL_ARGS,
    "dhu_tc2": _DHU_TC2_ARGS,
    "dqkwg": _DQKWG_ARGS,
    "wy_bwd": _WY_BWD_ARGS,
    "wy_bwd_bf16": _WY_BWD_ARGS,
    "finalize": _FINALIZE_ARGS,
    "finalize_mlp": _FINALIZE_ARGS,
    "tail_g2": _TAIL_ARGS,
    "tail_g2_bf16": _TAIL_ARGS,
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


@dataclass(frozen=True)
class ChainOptions:
    """The explicit plan pins of one call.

    ``scan_config`` forces the register-scan configuration ``(value block, key split)`` of both scans;
    ``segment_chunks`` forces the segment length in chunks (0: the serial scan; above the chunks one pre-scan
    CTA covers, 128, a plan that splits a sequence raises); ``finalize`` forces the finalize stage of the unfused tail
    (value-head groups other than two); ``warmup_log2`` overrides the pre-scans' warm-up threshold; ``dhu_scan``
    forces the adjoint scan of a serial plan (``"reg"``: ``dv_local`` + the register scan, ``"tc"``: ``dhu_tc2``;
    ``None``: the selection rule); ``prep`` forces the prep (``"plain"``: grid (chunks, HV); ``"tma"``: ``prep_tma``,
    grid (chunks, H), where q and k are 16-byte aligned; ``None``: the dispatch rule); ``save_state`` keeps the
    forward's ``w`` / ``u`` / ``h`` / ``v_new`` for the backward (else they are recomputed); ``host_cache`` keeps the
    chunk tables, the selection results and the per-stream workspaces between calls; ``lpt`` dispatches the
    sequences of a host-built table longest first (the same bits; ``False``: the caller's order); ``streams`` is
    ``2`` to let a host-built table's serial scans run as two sequence groups on the caller's stream and a side stream
    where their rule (:func:`two_stream_split`) fills the last round (the same bits), ``1`` to keep every launch on
    the caller's stream.
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
    )


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
    ``bwd_groups``: the two-stream sequence groups of the forward / adjoint scan (``None``: one stream)."""

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


class Workspaces:
    def __init__(self):
        self.tables: dict = {}

    def table(self, device_index: int) -> dict:
        key = (device_index, torch.cuda.current_stream(device_index).cuda_stream)
        return self.tables.setdefault(key, {})

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
# prep selection: prep_tma (TMA-staged q / k, one CTA per (chunk, key head)) or the plain prep
# --------------------------------------------------------------------------- #
# The rule: CTAs = chunks x H and slots = ctas_per_sm x SMs; the plain prep (grid (chunks, HV)) runs where prep_tma's
# grid needs a second, short wave, slots < CTAs <= window x slots, prep_tma elsewhere.  Both give the same bits.  The
# tensor maps need q and k 16-byte aligned; an unaligned q or k takes the plain prep.
PREP_TMA_CONSTANTS = ("ctas_per_sm", "window")
TMA_ALIGN_BYTES = 16


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
    q, k, num_chunks: int, num_heads: int, arch: str, options: ChainOptions
) -> str:
    """``prep_tma`` or ``prep`` for one call: the ``prep`` pin, else the rule; the plain prep when q or k is not
    16-byte aligned or the architecture has no ``prep_tma`` record."""
    if options.prep == "plain" or not _available("prep_tma", arch):
        return "prep"
    if any(t.data_ptr() % TMA_ALIGN_BYTES for t in (q, k)):
        return "prep"
    if options.prep == "tma" or use_prep_tma(num_chunks, num_heads, _tensor_index(q)):
        return "prep_tma"
    return "prep"


# --------------------------------------------------------------------------- #
# two-stream sequence groups
# --------------------------------------------------------------------------- #
# A serial scan launches one CTA per (segment, value head) and holds one CTA per SM, so its last round can leave SMs
# idle.  Split into two contiguous groups of segments, the group that fills whole rounds runs on one stream while the
# other group's scan overlaps the neighbouring stage (forward: wy_tc of the other group; adjoint: the tail of the
# first group) on a second stream.  A group launch is the full launch restricted to its segments (fwd_ho through a
# slice of its walk, dhu_tc2 through views of the segment table and of dht) or chunks (wy_tc and tail_g2 through
# views of the chunk table and of h / dh): every CTA does the work of the CTA of the full launch that covers the same
# (segment or chunk, value head), so the bits are the same.


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
    meta: ChunkMeta, num_v_heads: int, sm_count: int
) -> tuple[StreamGroups | None, StreamGroups | None]:
    """The two-stream groups of a host-built table's forward and adjoint scans (``None``: one stream).  A scan is
    grouped on a serial plan (one segment per sequence, the table's walk) when :func:`two_stream_split` splits it: the
    adjoint scan's group A is its full-round group, the forward scan's group A the rest."""
    if meta.seq_lens is None:
        return None, None
    order = meta.order if meta.order is not None else range(meta.num_seqs)
    chunks = [-(-meta.seq_lens[seq] // CHUNK) for seq in order]
    groups: list[StreamGroups | None] = []
    for segs, adjoint in ((meta.fwd_segments, False), (meta.bwd_segments, True)):
        k = None
        if segs.num_segments == meta.num_seqs and segs.walk is None:
            k = two_stream_split(meta.num_seqs, int(num_v_heads), int(sm_count))
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
    _launch("wy_tc", meta.num_chunks, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start, meta.chunk_len,
            num_heads, num_v_heads, 1 if recompute else 0)  # fmt: skip
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
        _launch("fwd_ho", segs.num_segments * num_v_heads, 1, 1, qn, kn, w, u, g_cs, h0, seg_start, h, v_new, v_new, o,
                o, final_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, walk, num_heads, num_v_heads, use_init, 1 if store_final_state else 0, use_seg,
                float(scale))  # fmt: skip
        return h, v_new, final_state, o
    # P form: the side stream runs wy_tc and fwd_ho of group A (the partial round); the caller's stream runs those of
    # group B (the full rounds) once wy_tc(A) is done, then joins the side stream.  fwd_ho(A) can read w / u rows past
    # the length of a partial chunk (its 64-row tiles) while wy_tc(B) writes them: such rows enter fwd_ho only
    # through selects (v_new rows past the chunk's length are 0), so they do not change a bit.  A kernel that used them
    # arithmetically (a multiply by a zero mask) would make this form nondeterministic.
    v, beta32, A = wy
    groups = cast("StreamGroups", meta.fwd_groups)
    n, na, ca = segs.num_segments, groups.split, groups.split_chunk

    def wy_tc(c0, c1):
        _launch("wy_tc", c1 - c0, num_v_heads, 1, kn, v, g_cs, beta32, A, w, u, meta.chunk_start[c0:c1],
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


def gdn_chunk_forward(q, k, v, g, beta, *, meta, arch, scale=None, initial_state=None, output_final_state=False,
                      normalize_qk=True, options: ChainOptions | None = None) -> dict[str, Any]:  # fmt: skip
    """The forward chain over flattened ``[T, H, 128]`` / ``[T, HV, 128]`` inputs and one chunk table; returns the
    output and the tape the backward reads."""
    options = options or ChainOptions()
    num_heads, key_dim = q.shape[-2], q.shape[-1]
    num_v_heads, value_dim = v.shape[-2], v.shape[-1]
    if key_dim != HEAD_DIM or value_dim != HEAD_DIM:
        raise ValueError("gdn_chunk_forward requires K = V = 128")
    if num_v_heads % num_heads != 0:
        raise ValueError("num_v_heads must be a multiple of num_heads")
    total_tokens = q.numel() // (num_heads * key_dim)
    if scale is None:
        scale = 1.0 / math.sqrt(key_dim)
    device = q.device
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
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
    prep = _prep_stage(q, k, meta.num_chunks, num_heads, arch, options)
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
        "v_new": v_new, "options": options,
    }  # fmt: skip


# --------------------------------------------------------------------------- #
# backward
# --------------------------------------------------------------------------- #
def gdn_chunk_backward(
    forward, do, dht=None, *, initial_state=None, dbeta_dtype=None
) -> dict[str, Any]:
    """The backward chain over one forward tape; returns ``dq``, ``dk``, ``dv``, ``dg``, ``dbeta`` and
    ``dinitial_state``."""
    arch = forward["arch"]
    meta = forward["meta"]
    options = forward.get("options") or ChainOptions()
    workspaces = _workspaces(options)
    qn, kn, A, g_cs = forward["qn"], forward["kn"], forward["A"], forward["g_cs"]
    scale = float(forward["scale"])
    v, beta32 = forward["v"], forward["beta32"]
    num_heads, num_v_heads = qn.shape[-2], v.shape[-2]
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
    serial = not (segs.segment_len > 0 and segs.num_segments > meta.num_seqs)
    use_tc = serial and _dhu_tc_selected(meta, num_v_heads, index, options)
    # two sequence groups: the tcgen05 serial scan feeding the fused tail of a value-head group of two
    groups = meta.bwd_groups if use_tc and num_v_heads == 2 * num_heads else None
    if not use_tc:
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
    if use_tc and groups is None:
        # dv_local fused into the tcgen05 serial scan: the same dh / dv2 / dh0 (seg_state is not read)
        _launch("dhu_tc2", segs.num_segments, num_v_heads, 1, qn, kn, w, do, g_cs, dht32, dh, dv2, dh0, dh0,
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
        _launch(pre, segs.num_segments, num_v_heads, 2 * HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32, dh,
                dv2, dh0, seg_state, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 0, scale, seg_flag,
                warmup_thresh_log2(options))  # fmt: skip
        dh_end = workspaces.take(table, "b_dh_end", (segs.num_segments, num_v_heads, HEAD_DIM, HEAD_DIM),
                                 torch.float32, device)  # fmt: skip
        _launch("stitch", meta.num_seqs, num_v_heads, STITCH_VSPLIT, seg_state, dht32, dh_end, segs.seq_seg_start,
                seg_flag, num_v_heads, use_dht, 1)  # fmt: skip
        dht32, use_dht = dh_end, 1
    if not use_tc:
        _launch(scan_stage, segs.num_segments, num_v_heads, HEAD_DIM // vb, qn, kn, w, do, dv_local, g_cs, dht32, dh,
                dv2, dh0, dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start, segs.seg_seq,
                meta.seq_chunk_start, num_heads, num_v_heads, use_dht, 1 if has_initial_state else 0, scale)  # fmt: skip
    dbeta_bf16 = dbeta_dtype == torch.bfloat16
    if num_v_heads == 2 * num_heads:
        # a value-head group of two: dqkwg, the WY pullback and the finalize as one cluster kernel
        dv = torch.empty(v.shape, dtype=v.dtype, device=device)
        dbeta = torch.empty(forward.get("beta_shape") or (total_tokens, num_v_heads),
                            dtype=torch.bfloat16 if dbeta_bf16 else torch.float32, device=device)  # fmt: skip
        dq = torch.empty(qn.shape, dtype=qn.dtype, device=device)
        dk = torch.empty(kn.shape, dtype=kn.dtype, device=device)
        dg = torch.empty(
            forward.get("g_shape") or (total_tokens, num_v_heads),
            dtype=torch.float32,
            device=device,
        )
        tail = "tail_g2_bf16" if dbeta_bf16 else "tail_g2"
        normalize = 1 if forward.get("normalize_qk", True) else 0
        if groups is None:
            _launch(tail, meta.num_chunks * num_v_heads, 1, 1, qn, kn, forward["rstd_q"], forward["rstd_k"], v_new, do,
                    dv2, h, dh, g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, meta.chunk_start, meta.chunk_len, num_heads,
                    num_v_heads, scale, normalize)  # fmt: skip
        else:
            # R form: the caller's stream runs dhu_tc2 of group A (the full rounds), then the tail of A while the side
            # stream runs dhu_tc2 of group B (the partial round), then joins it and runs the tail of B
            na, ca = groups.split, groups.split_chunk

            def dhu_tc2(s0, s1):
                _launch("dhu_tc2", s1 - s0, num_v_heads, 1, qn, kn, w, do, g_cs, dht32[s0:s1] if use_dht else dht32,
                        dh, dv2, dh0, dh0, meta.chunk_start, meta.chunk_len, segs.seg_chunk_start[s0:s1 + 1],
                        segs.seg_seq[s0:s1], meta.seq_chunk_start, num_heads, num_v_heads, use_dht,
                        1 if has_initial_state else 0, scale)  # fmt: skip

            def tail_g2(c0, c1):
                _launch(tail, (c1 - c0) * num_v_heads, 1, 1, qn, kn, forward["rstd_q"], forward["rstd_k"], v_new, do,
                        dv2, h[c0:c1], dh[c0:c1], g_cs, v, A, beta32, dq, dk, dg, dv, dbeta, meta.chunk_start[c0:c1],
                        meta.chunk_len[c0:c1], num_heads, num_v_heads, scale, normalize)  # fmt: skip

            main = torch.cuda.current_stream(index)
            side = _side_stream(index, main)
            dhu_tc2(0, na)
            fork = torch.cuda.Event()
            fork.record(main)
            try:
                with torch.cuda.stream(side):
                    side.wait_event(fork)
                    dhu_tc2(na, segs.num_segments)
                tail_g2(0, ca)
            finally:
                main.wait_stream(side)  # the join
            tail_g2(ca, meta.num_chunks)
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
    """The API-boundary checks of :func:`chunk_gated_delta_rule`: devices, shapes and the offsets' form, from tensor
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


def _chunk_meta(
    cu_seqlens,
    cu_seqlens_cpu,
    batch,
    seq_len,
    index: int,
    num_v_heads,
    options: ChainOptions,
) -> ChunkMeta:
    """The chunk and segment tables of one call on device ``index``: a ``host_cache`` hit on the layout's key, else
    planned (the layout's sequence lengths are derived only then)."""
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
        options.lpt,
        options.streams,
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
    plans = scan_plans(total_tokens, len(seq_lens), int(num_v_heads), int(index), seq_lens,
                       scan_config=options.scan_config, segment_chunks=options.segment_chunks)  # fmt: skip
    meta = _host_chunk_meta(seq_lens, plans, int(index), options.lpt)
    if options.streams == 2:
        fwd_groups, bwd_groups = sequence_groups(
            meta, int(num_v_heads), _device_props(int(index))[0]
        )
        if fwd_groups is not None or bwd_groups is not None:
            meta = replace(meta, fwd_groups=fwd_groups, bwd_groups=bwd_groups)
    if options.host_cache:
        if len(_META_FRONT) >= _META_FRONT_LIMIT:
            del _META_FRONT[next(iter(_META_FRONT))]
        _META_FRONT[key] = meta
    return meta


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, g, beta, scale, initial_state, output_final_state, cu_seqlens, cu_seqlens_cpu,
                use_qk_l2norm_in_kernel, options):  # fmt: skip
        batch, seq_len, num_heads, key_dim = q.shape
        num_v_heads = v.shape[2]
        index = _tensor_index(q)
        arch = cake_backend.device_arch(index)
        meta = _chunk_meta(
            cu_seqlens, cu_seqlens_cpu, batch, seq_len, index, num_v_heads, options
        )
        fwd = gdn_chunk_forward(q, k, v, g, beta, meta=meta, arch=arch, scale=scale, initial_state=initial_state,
                                output_final_state=output_final_state, normalize_qk=use_qk_l2norm_in_kernel,
                                options=options)  # fmt: skip
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
        output = fwd["output"]
        if output.dtype != q.dtype:
            output = output.to(q.dtype)
        return output, (fwd["final_state"] if output_final_state else None)

    @staticmethod
    def backward(ctx, do, dht):
        qn, kn, rstd_q, rstd_k, v_flat, g_cs, beta32, A, saved_state, *state_tensors = (
            ctx.saved_tensors
        )
        g_shape, beta_shape = ctx.grad_shapes
        g_dtype, beta_dtype = ctx.dtypes
        tape = {"arch": ctx.arch, "qn": qn, "kn": kn, "rstd_q": rstd_q, "rstd_k": rstd_k, "v": v_flat, "g_cs": g_cs,
                "beta32": beta32, "A": A, "meta": ctx.meta, "scale": ctx.scale, "normalize_qk": ctx.normalize_qk,
                "g_shape": g_shape, "beta_shape": beta_shape, "options": ctx.options}  # fmt: skip
        if ctx.saved_state_tensors:
            tape["w"], tape["u"], tape["h"], tape["v_new"] = state_tensors
        grads = gdn_chunk_backward(
            tape,
            do.contiguous(),
            dht,
            initial_state=saved_state,
            dbeta_dtype=beta_dtype,
        )
        dg = grads["dg"]
        if dg.dtype != g_dtype:
            dg = dg.to(g_dtype)
        dbeta = grads["dbeta"]
        if dbeta.dtype != beta_dtype:
            dbeta = dbeta.to(beta_dtype)
        dh0 = grads["dinitial_state"] if ctx.has_initial_state_input else None
        return (
            grads["dq"],
            grads["dk"],
            grads["dv"],
            dg,
            dbeta,
            None,
            dh0,
            None,
            None,
            None,
            None,
            None,
        )


def chunk_gated_delta_rule(
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
):
    """Deterministic chunked gated delta rule with FLA's signature (head dimension 128, ``HV`` a multiple of ``H``).

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
    """
    if (
        q.dtype != torch.bfloat16
        or k.dtype != torch.bfloat16
        or v.dtype != torch.bfloat16
    ):
        raise NotImplementedError("q/k/v must be bfloat16")
    _check_inputs(q, k, v, g, beta, initial_state, cu_seqlens, cu_seqlens_cpu)
    options = make_options(scan_config=scan_config, segment_chunks=segment_chunks, finalize=finalize,
                           warmup_log2=warmup_log2, save_state=save_state, host_cache=host_cache,
                           dhu_scan=dhu_scan, prep=prep, lpt=lpt, streams=streams)  # fmt: skip
    if g.dtype != torch.float32:
        g = g.float()
    if scale is None:
        scale = 1.0 / math.sqrt(q.shape[-1])
    return ChunkGatedDeltaRuleFunction.apply(q, k, v, g, beta, float(scale), initial_state, output_final_state,
                                             cu_seqlens, cu_seqlens_cpu, use_qk_l2norm_in_kernel, options)  # fmt: skip
