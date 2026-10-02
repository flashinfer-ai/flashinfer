# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# See LICENSE.kda-for-kda.txt for the full license.

"""Persistent KDA scheduler imported from NVlabs/kda.

Whole sequence/head chains are packed across physical SMs. Long chains may be
split only with an exact FP32 state handoff (within one launch or across two
launches). Approximate state reconstruction and expected-norm substitution are
not supported by this integration.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
from itertools import pairwise

import torch

from .fallback import prepare_fwd
from .static_runtime import (
    compiled_stream_m64_fixed,
    compiled_stream_m64_fixed_final,
    compiled_stream_m128,
    compiled_stream_m128_cluster2,
    compiled_stream_m128_cluster8,
    compiled_stream_m128_fixed_h96_final,
)

K2_CHUNK = 32
HEAD_DIM = 128
PIECE_OVERHEAD_CHUNKS = 3
# Retained kernel ABI field; exact schedules never enable the legacy detector.
DETECT_THRESHOLD_LOG2 = -50.0
# An exact two-launch tail split must save at least four percent of the modeled
# persistent-CTA span to repay the extra pipeline fill and drain.
EXACT_TAIL_SPLIT_MARGIN = 0.96
# Measured fixed cost of the second kernel fill/drain plus FP32 state handoff,
# expressed in the same 32-token slot units as the LPT model.
EXACT_TAIL_PHASE_TAX = 8
# A single-launch handoff split must save at least two percent of the
# modeled span; no second launch means no fill/drain tax applies.
INLINE_SPLIT_MARGIN = 0.98
# Modeled extra slots per handoff piece (state store/reload traffic and
# the small-piece pipeline refill), which steers the search toward fewer
# and larger splits; measured on uniform/mixed screens in session 18.
HANDOFF_PIECE_TAX = 0.0
HANDOFF_SPLIT_SLOPE = 0.04
# A B300 sweep of dense random FP32 state remained finite through scale 2^12
# (observed max above 1.6e4) and became nonfinite at scale 2^16.  Keeping the
# absolute-value limit at 2^12 is conservative for dense inputs while retaining
# the required sparse striped-state split case whose maximum is exactly 2^12.
MAX_STREAM_INITIAL_STATE_ABS = float(2**12)

# Exact 173-slot packing for the canonical mixed-H96 final-state job multiset.
# Vector positions are head-5, tail-93, whole-A-97, B-65, C-42, D-32,
# E-19, and F-10. It retains the production 64-chain chunk-4 split while
# removing the final slot left by generic LPT packing.
_MIXED_H96_FINAL_PACKING = (
    (16, (0, 0, 0, 1, 0, 3, 0, 1)),
    (24, (0, 0, 0, 2, 0, 1, 0, 1)),
    (8, (0, 0, 1, 0, 0, 2, 0, 1)),
    (32, (0, 1, 0, 0, 1, 0, 2, 0)),
    (4, (0, 1, 0, 1, 0, 0, 0, 1)),
    (12, (1, 0, 0, 0, 4, 0, 0, 0)),
    (8, (1, 0, 1, 0, 0, 1, 2, 0)),
    (16, (1, 0, 1, 0, 1, 0, 1, 1)),
    (28, (1, 1, 0, 1, 0, 0, 0, 1)),
)


def _lpt_makespan(job_lens, machines):
    """Longest-processing-time greedy makespan.

    :param job_lens: Per-job costs in chunk slots.
    :type job_lens: list
    :param machines: Machine count.
    :type machines: int
    :return: Modeled makespan in chunk slots.
    :rtype: float
    """
    heap = [0.0] * machines
    heapq.heapify(heap)
    for cost in sorted(job_lens, reverse=True):
        earliest = heapq.heappop(heap)
        heapq.heappush(heap, earliest + cost)
    return max(heap)


def pack_bins(rows, num_sms, job_overhead=0):
    """LPT-pack piece rows onto at most one stream CTA per SM.

    :param rows: Piece-table rows from plan_pieces().
    :type rows: list
    :param num_sms: SM count bounding the CTA count.
    :type num_sms: int
    :param job_overhead: Extra scheduling cost for each row assigned to a CTA.
    :type job_overhead: int
    :return: (order, cta_vals) where order reindexes rows grouped by CTA and
        cta_vals is the flattened [row_start, row_count, total_chunks] table.
    :rtype: tuple

    .. code-block:: python

        order, cta_vals = pack_bins(rows, 148)
        rows = [rows[i] for i in order]
    """
    costs = [(r[1] - r[0] + K2_CHUNK - 1) // K2_CHUNK for r in rows]
    bins = min(num_sms, len(rows))
    heap = [(0, b) for b in range(bins)]
    heapq.heapify(heap)
    assign = [[] for _ in range(bins)]
    for i in sorted(range(len(rows)), key=lambda i: -costs[i]):
        load, b = heapq.heappop(heap)
        assign[b].append(i)
        heapq.heappush(heap, (load + costs[i] + job_overhead, b))
    order = []
    cta_vals = []
    for b in range(bins):
        cta_vals.extend([len(order), len(assign[b]), sum(costs[i] for i in assign[b])])
        order.extend(assign[b])
    return order, cta_vals


def _simulate_handoff_bins(jobs, num_sms):
    """LPT-pack handoff jobs and simulate the head-first tail-last timeline.

    :param jobs: (cost, kind, chain) tuples; kind 1=head, 0=whole, 2=tail.
    :type jobs: list
    :param num_sms: Machine count.
    :type num_sms: int
    :return: (makespan, spin, bins) with bins holding job indices in
        execution order.
    :rtype: tuple
    """
    heap = [(0, b) for b in range(num_sms)]
    heapq.heapify(heap)
    bins = [[] for _ in range(num_sms)]
    for j in sorted(range(len(jobs)), key=lambda i: (-jobs[i][0], jobs[i][1])):
        load, b = heapq.heappop(heap)
        bins[b].append(j)
        heapq.heappush(heap, (load + jobs[j][0], b))
    rank = {1: 0, 0: 1, 2: 2}
    finish_head = {}
    for pieces in bins:
        pieces.sort(key=lambda j: (rank[jobs[j][1]], -jobs[j][0]))
        t = 0
        for j in pieces:
            cost, kind, chain = jobs[j]
            if kind == 2:
                continue
            t += cost
            if kind == 1:
                finish_head[chain] = t
    makespan = 0
    spin = 0
    for pieces in bins:
        t = 0
        for j in pieces:
            cost, kind, chain = jobs[j]
            if kind == 2:
                ready = finish_head[chain]
                if ready > t:
                    spin += ready - t
                    t = ready
            t += cost
        makespan = max(makespan, t)
    return makespan, spin, bins


def _pack_mixed_h96_final_handoff(jobs, num_sms):
    """Pack the canonical exact handoff jobs to their 173-slot lower bound."""

    type_costs = (5, 93, 97, 65, 42, 32, 19, 10)
    whole_type = {cost: index for index, cost in enumerate(type_costs[2:], 2)}
    buckets = defaultdict(deque)
    for job_index, (cost, kind, _chain) in enumerate(jobs):
        type_index = 0 if kind == 1 else 1 if kind == 2 else whole_type[cost]
        buckets[type_index].append(job_index)
    if tuple(len(buckets[index]) for index in range(8)) != (
        64,
        64,
        32,
        96,
        96,
        96,
        96,
        96,
    ):
        raise RuntimeError("unexpected canonical mixed-H96 handoff jobs")

    bins = []
    for count, vector in _MIXED_H96_FINAL_PACKING:
        for _ in range(count):
            pieces = []
            for type_index, copies in enumerate(vector):
                for _copy in range(copies):
                    pieces.append(buckets[type_index].popleft())
            heads = [piece for piece in pieces if jobs[piece][1] == 1]
            tails = [piece for piece in pieces if jobs[piece][1] == 2]
            whole = sorted(
                (piece for piece in pieces if jobs[piece][1] == 0),
                key=lambda piece: -jobs[piece][0],
            )
            # All heads finish at slot 5. A tail-only bin executes one whole
            # chain first, so no tail spins on its release flag.
            bins.append(
                heads + tails + whole if heads else whole[:1] + tails + whole[1:]
            )
    if len(bins) != num_sms or any(buckets.values()):
        raise RuntimeError("invalid canonical mixed-H96 handoff packing")
    if max(sum(jobs[piece][0] for piece in pieces) for pieces in bins) != 173:
        raise RuntimeError("canonical mixed-H96 handoff span is not 173")
    return bins


def plan_exact_inline_split(rows, num_sms):
    """Split whole chains into head+tail pieces within a single launch.

    The head piece stores its final FP32 state into the handoff buffer
    (bound as ``final_state``) and publishes per-lane release flags; the
    tail piece spins on those flags before loading the state, so the
    recurrence stays exact and the stream output is bitwise-identical to
    the unsplit schedule.  Heads are ordered first and tails last on their
    CTAs and heads never wait, so the one-wave resident grid (one stream
    CTA per SM) cannot deadlock.

    :param rows: Exact whole-chain piece rows.
    :type rows: list
    :param num_sms: Available SM count.
    :type num_sms: int
    :return: (ordered_rows, cta_vals, n_split, prefix, base_span, span)
        or None when no split beats the modeled margin.
    :rtype: tuple or None
    """
    chunk_counts = [(row[1] - row[0] + K2_CHUNK - 1) // K2_CHUNK for row in rows]
    full_counts = [(row[1] - row[0]) // K2_CHUNK for row in rows]
    base_span = _lpt_makespan([c + 1 for c in chunk_counts], num_sms)
    area_floor = -(-(sum(chunk_counts) + len(rows)) // num_sms)
    longest = max(chunk_counts)
    if base_span * INLINE_SPLIT_MARGIN <= area_floor + 1:
        return None
    if longest + 1 >= base_span:
        return None
    order_desc = sorted(range(len(rows)), key=lambda i: -chunk_counts[i])
    canonical_mixed_h64_final = (
        num_sms == 148
        and len(rows) == 384
        and all(row[4] == 1 and row[5] == 1 for row in rows)
        and {row[6] for row in rows} == set(range(64))
        and {(row[0], row[1], row[3]) for row in rows}
        == {
            (0, 1300, 0),
            (1300, 1847, 1),
            (1847, 3895, 2),
            (3895, 4858, 3),
            (4858, 5129, 4),
            (5129, 8192, 5),
        }
    )
    canonical_mixed_h96_final = (
        num_sms == 148
        and len(rows) == 576
        and all(row[4] == 1 and row[5] == 1 for row in rows)
        and {row[6] for row in rows} == set(range(96))
        and {(row[0], row[1], row[3]) for row in rows}
        == {
            (0, 1300, 0),
            (1300, 1847, 1),
            (1847, 3895, 2),
            (3895, 4858, 3),
            (4858, 5129, 4),
            (5129, 8192, 5),
        }
    )
    canonical_uniform_h64_final = (
        num_sms == 148
        and len(rows) == 512
        and all(row[4] == 1 and row[5] == 1 for row in rows)
        and {row[6] for row in rows} == set(range(64))
        and {(row[0], row[1], row[3]) for row in rows}
        == {(start, start + 1024, start // 1024) for start in range(0, 8192, 1024)}
    )
    canonical_uniform_h96_final = (
        num_sms == 148
        and len(rows) == 768
        and all(row[4] == 1 and row[5] == 1 for row in rows)
        and {row[6] for row in rows} == set(range(96))
        and {(row[0], row[1], row[3]) for row in rows}
        == {(start, start + 1024, start // 1024) for start in range(0, 8192, 1024)}
    )

    def build_jobs(n_split, prefix):
        jobs = []
        for pos, idx in enumerate(order_desc):
            chunks = chunk_counts[idx]
            if pos < n_split:
                if prefix >= chunks or prefix > full_counts[idx]:
                    return None
                jobs.append((prefix + 1 + HANDOFF_PIECE_TAX, 1, idx))
                jobs.append((chunks - prefix + 1 + HANDOFF_PIECE_TAX, 2, idx))
            else:
                jobs.append((chunks + 1, 0, idx))
        return jobs

    best = None
    if canonical_mixed_h64_final:
        split_counts = (24,)
        prefixes = (12,)
    elif canonical_mixed_h96_final:
        split_counts = (64,)
        prefixes = (4,)
    elif canonical_uniform_h64_final:
        split_counts = (72,)
        prefixes = (16,)
    elif canonical_uniform_h96_final:
        split_counts = (56,)
        prefixes = (11,)
    else:
        split_counts = range(8, len(rows) + 1, 8)
        prefixes = range(1, longest)
    for n_split in split_counts:
        for prefix in prefixes:
            jobs = build_jobs(n_split, prefix)
            if jobs is None:
                continue
            span, _spin, _bins = _simulate_handoff_bins(jobs, num_sms)
            # Measured excess over the tax-free simulated span grows
            # linearly with the total split count (flag and state-traffic
            # contention), not per piece: charge it globally.
            key = (span + HANDOFF_SPLIT_SLOPE * n_split, n_split, prefix)
            if best is None or key < best[0]:
                best = (key, n_split, prefix)
    if best is None or best[0][0] > base_span - 1.0:
        return None
    n_split, prefix = best[1], best[2]
    jobs = build_jobs(n_split, prefix)
    span, _spin, bins = _simulate_handoff_bins(jobs, num_sms)
    if canonical_mixed_h96_final:
        bins = _pack_mixed_h96_final_handoff(jobs, num_sms)
        span = 173
    slot_of = {}
    for pos in range(n_split):
        slot_of[order_desc[pos]] = pos
    ordered_rows = []
    cta_vals = []
    for pieces in bins:
        if not pieces:
            continue
        start = len(ordered_rows)
        total = 0
        for j in pieces:
            _cost, kind, idx = jobs[j]
            row = rows[idx]
            if kind == 0:
                out_row = list(row)
            else:
                cut = row[0] + prefix * K2_CHUNK
                flag_base = slot_of[idx] * HEAD_DIM
                if kind == 1:
                    out_row = [
                        row[0],
                        cut,
                        row[0],
                        row[3],
                        row[4],
                        2,
                        row[6],
                        flag_base,
                    ]
                else:
                    out_row = [
                        cut,
                        row[1],
                        cut,
                        row[3],
                        2,
                        row[5],
                        row[6],
                        flag_base,
                    ]
            ordered_rows.append(out_row)
            total += (out_row[1] - out_row[0] + K2_CHUNK - 1) // K2_CHUNK
        cta_vals.extend([start, len(ordered_rows) - start, total])
    return ordered_rows, cta_vals, n_split, prefix, base_span, span


def plan_exact_tail_split(rows, offsets, num_heads, num_sms, store_final):
    """Split sequence tails across two ordered launches when it removes a wave.

    Both phases retain complete recurrent state through a native FP32 handoff.
    The planner is shape-only: it minimizes the sum of the two LPT makespans
    and declines when the modeled saving is below the launch-cost margin.

    :param rows: Exact whole-chain piece rows.
    :type rows: list
    :param offsets: Packed sequence boundaries.
    :type offsets: list
    :param num_heads: Head count.
    :type num_heads: int
    :param num_sms: Available SM count.
    :type num_sms: int
    :param store_final: Whether the public call exports final state.
    :type store_final: int
    :return: Phase rows and plan metadata, or None.
    :rtype: tuple or None
    """
    if (
        num_sms == 148
        and store_final
        and num_heads in (64, 96)
        and offsets
        in (
            [0, 1300, 1847, 3895, 4858, 5129, 8192],
            [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
        )
    ):
        return None

    row_chunks = [(row[1] - row[0] + K2_CHUNK - 1) // K2_CHUNK for row in rows]
    base_span = _lpt_makespan([cost + 1 for cost in row_chunks], num_sms)
    best = None

    def consider(seq_indices, prefix_chunks, other_jobs, seq_chunks):
        nonlocal best
        split_jobs = num_heads * len(seq_indices)
        phase_a_jobs = other_jobs + [
            p + 1 for p in prefix_chunks for _ in range(num_heads)
        ]
        phase_b_jobs = [
            seq_chunks - p + 1 for p in prefix_chunks for _ in range(num_heads)
        ]
        if len(phase_a_jobs) != len(rows) or len(phase_b_jobs) != split_jobs:
            raise RuntimeError("invalid exact tail-split model")
        span_a = _lpt_makespan(phase_a_jobs, num_sms)
        span_b = _lpt_makespan(phase_b_jobs, num_sms)
        key = (span_a + span_b, len(seq_indices), span_a, prefix_chunks, seq_indices)
        if best is None or key < best[0]:
            best = (key, seq_indices, prefix_chunks, span_a)

    length_groups = {}
    for seq_idx, (bos, eos) in enumerate(pairwise(offsets)):
        length_groups.setdefault(eos - bos, []).append(seq_idx)
    for seq_len, seq_group in length_groups.items():
        seq_chunks = (seq_len + K2_CHUNK - 1) // K2_CHUNK
        if seq_chunks < 2:
            continue
        for num_split_seqs in range(1, len(seq_group) + 1):
            seq_indices = tuple(seq_group[:num_split_seqs])
            seq_index_set = set(seq_indices)
            other_jobs = [
                cost + 1
                for row, cost in zip(rows, row_chunks, strict=False)
                if row[3] not in seq_index_set
            ]
            if len(rows) - len(other_jobs) != num_heads * num_split_seqs:
                continue
            for prefix_chunks in range(1, seq_chunks):
                consider(
                    seq_indices,
                    (prefix_chunks,) * num_split_seqs,
                    other_jobs,
                    seq_chunks,
                )

        # Pair-specific unequal cuts can reduce state handoff traffic while
        # retaining the best two-phase span. Bound the quadratic host search;
        # longer layouts keep the common-cut or single-launch plan.
        if len(seq_group) >= 2 and seq_chunks <= 64:
            seq_indices = tuple(seq_group[:2])
            seq_index_set = set(seq_indices)
            other_jobs = [
                cost + 1
                for row, cost in zip(rows, row_chunks, strict=False)
                if row[3] not in seq_index_set
            ]
            for prefix_a in range(1, seq_chunks):
                for prefix_b in range(1, seq_chunks):
                    consider(
                        seq_indices,
                        (prefix_a, prefix_b),
                        other_jobs,
                        seq_chunks,
                    )
    if (
        best is None
        or best[0][0] + EXACT_TAIL_PHASE_TAX >= base_span * EXACT_TAIL_SPLIT_MARGIN
    ):
        return None

    total_span = best[0][0]
    seq_indices, prefix_chunks, span_a = best[1:]
    seq_index_set = set(seq_indices)
    prefix_by_seq = dict(zip(seq_indices, prefix_chunks, strict=False))
    phase_a = []
    phase_b = []
    for row in rows:
        seq_idx = row[3]
        if seq_idx not in seq_index_set:
            phase_a.append(list(row))
            continue
        bos, eos = offsets[seq_idx], offsets[seq_idx + 1]
        cut = bos + prefix_by_seq[seq_idx] * K2_CHUNK
        prefix = list(row)
        prefix[1] = cut
        prefix[5] = 1
        phase_a.append(prefix)
        phase_b.append([cut, eos, cut, seq_idx, 1, store_final, row[6], row[7]])
    return (
        phase_a,
        phase_b,
        seq_indices,
        prefix_chunks,
        base_span,
        total_span,
        span_a,
    )


def plan_pieces(offsets, num_heads, num_sms, warm_tables, use_init, store_final):
    """Build unsplit recurrence rows; exact handoff planners balance them later."""
    total_chains = (len(offsets) - 1) * num_heads
    overhead = (
        1
        if total_chains <= num_sms // 2 or total_chains >= 5 * num_sms
        else PIECE_OVERHEAD_CHUNKS
    )
    rows = []
    for seq, (bos, eos) in enumerate(pairwise(offsets)):
        cost = (eos - bos + K2_CHUNK - 1) // K2_CHUNK + overhead
        for head in range(num_heads):
            rows.append((cost, [bos, eos, bos, seq, use_init, store_final, head, 0]))
    rows.sort(key=lambda item: -item[0])
    return [row for _, row in rows], False, []


class KDAPiecesLaunch:
    """Preallocated dynamic-g piece launch with an exact fallback."""

    def __init__(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
        allow_approximate_split=False,
        use_expected_norm=False,
    ):
        self.out = out
        self.final_state = final_state
        self._g_shape = tuple(g.shape)
        self._g_dtype = g.dtype
        self._g = g
        self._initial_state = initial_state
        if allow_approximate_split or use_expected_norm:
            raise ValueError(
                "PTX KDA supports exact state handoff and exact Q/K norms only"
            )
        self.expected_norm_enabled = False
        self.normalization_mode = "exact_norm"
        self._staging_in = (
            initial_state.to(torch.bfloat16) if initial_state is not None else None
        )
        self._staging_out = (
            torch.empty_like(final_state, dtype=torch.bfloat16)
            if final_state is not None
            else None
        )
        self._fallback = prepare_fwd(
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=self._staging_in,
            final_state=self._staging_out,
            cu_seqlens=cu_seqlens,
        )
        base_args = self._fallback.args
        if initial_state is not None:
            initial_state_abs_max = float(initial_state.detach().abs().max())
            if (
                not math.isfinite(initial_state_abs_max)
                or initial_state_abs_max > MAX_STREAM_INITIAL_STATE_ABS
            ):
                self._module = None
                self._detector = None
                self.expected_norm_enabled = False
                self.normalization_mode = "exact_norm_fallback"
                self.schedule = f"state_guard_{self._fallback.schedule}"
                self.state_input_mode = "timed_fp32_bf16_boundary_conversions"
                return
        offsets = [int(x) for x in base_args["cu_seqlens"].tolist()]
        num_heads = int(base_args["num_heads"])
        self._num_heads = num_heads
        if num_heads not in (64, 96):
            self._module = None
            self._detector = None
            self.expected_norm_enabled = False
            self.normalization_mode = "exact_norm_fallback"
            self.schedule = f"unsupported_heads_{self._fallback.schedule}"
            self.state_input_mode = "timed_fp32_bf16_boundary_conversions"
            return
        num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        warm_tables = [None] * (len(offsets) - 1)
        rows, has_split, self._windows = plan_pieces(
            offsets,
            num_heads,
            num_sms,
            warm_tables,
            int(initial_state is not None),
            int(final_state is not None),
        )
        full_chunks = all((row[1] - row[0]) % K2_CHUNK == 0 for row in rows)

        if (
            allow_approximate_split
            and not has_split
            and "m64" in self._fallback.schedule
        ):
            self._module = None
            self._detector = None
            self.expected_norm_enabled = False
            self.normalization_mode = "exact_norm_fallback"
            self.schedule = self._fallback.schedule
            self.state_input_mode = "timed_fp32_bf16_boundary_conversions"
            return

        n_split = sum(1 for r in rows if r[0] != r[2])
        dummy = torch.empty(1, dtype=torch.float32, device=q.device)
        args_template = dict(base_args)
        for name in (
            "cu_seqlens",
            "seq_order",
            "use_initial_state",
            "store_final_state",
        ):
            args_template.pop(name)
        beta_tma = args_template["beta_tma"]
        if num_heads % 8 != 0 and beta_tma.shape[1] == num_heads:
            padded_heads = ((num_heads + 7) // 8) * 8
            self._beta_tma = torch.zeros(
                (beta.numel() // num_heads, padded_heads),
                dtype=beta.dtype,
                device=beta.device,
            )
            self._beta_tma[:, :num_heads].copy_(beta.reshape(-1, num_heads))
            args_template["beta_tma"] = self._beta_tma

        def make_phase(phase_rows, state_in, state_out, job_overhead):
            order, cta_vals = pack_bins(
                phase_rows,
                num_sms,
                job_overhead=job_overhead,
            )
            ordered_rows = [phase_rows[i] for i in order]
            args = dict(args_template)
            args["piece_table"] = torch.tensor(
                ordered_rows, dtype=torch.int32, device=q.device
            ).contiguous()
            args["cta_table"] = torch.tensor(
                cta_vals, dtype=torch.int32, device=q.device
            ).contiguous()
            args["initial_state"] = state_in if state_in is not None else dummy
            args["final_state"] = state_out if state_out is not None else dummy
            phase_full = all((row[1] - row[0]) % K2_CHUNK == 0 for row in ordered_rows)
            return args, (len(cta_vals) // 3, 1, 1), ordered_rows, phase_full

        inline_plan = None
        if not has_split:
            inline_plan = plan_exact_inline_split(rows, num_sms)
        tail_plan = None
        if not has_split:
            tail_plan = plan_exact_tail_split(
                rows,
                offsets,
                num_heads,
                num_sms,
                int(final_state is not None),
            )
        if inline_plan is not None and tail_plan is not None:
            # Both exact rebalancing schedules qualified; keep the one
            # with the smaller modeled span (two-launch pays the phase
            # tax for its second pipeline fill/drain).
            if inline_plan[5] <= tail_plan[5] + EXACT_TAIL_PHASE_TAX:
                tail_plan = None
            else:
                inline_plan = None
        self._tail_module = None
        self._tail_args = None
        self._tail_grid = None
        self._handoff_flags = None
        use_mixed_cluster2 = False
        use_fixed_h64_m64 = (
            inline_plan is None
            and tail_plan is None
            and not allow_approximate_split
            and not use_expected_norm
            and initial_state is not None
            and lower_bound == -5.0
            and num_heads == 64
            and offsets == [0, 8192]
            and full_chunks
        )
        use_fixed_h96_cluster8 = (
            inline_plan is None
            and tail_plan is None
            and not allow_approximate_split
            and not use_expected_norm
            and initial_state is not None
            and final_state is None
            and lower_bound == -5.0
            and num_heads == 96
            and offsets == [0, 8192]
            and full_chunks
        )
        use_fixed_h96_static_final = (
            inline_plan is None
            and tail_plan is None
            and not allow_approximate_split
            and not use_expected_norm
            and initial_state is not None
            and final_state is not None
            and lower_bound == -5.0
            and num_heads == 96
            and offsets == [0, 8192]
            and full_chunks
        )
        if inline_plan is not None:
            (
                ordered_rows,
                cta_vals,
                n_split_chains,
                inline_prefix,
                inline_base_span,
                inline_span,
            ) = inline_plan
            self._handoff_state = (
                final_state.contiguous()
                if final_state is not None
                else torch.empty(
                    len(offsets) - 1,
                    num_heads,
                    HEAD_DIM,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            self._handoff_flags = torch.zeros(
                n_split_chains * HEAD_DIM, dtype=torch.int32, device=q.device
            )
            use_mixed_h96_gpc_swap = (
                num_sms == 148
                and not allow_approximate_split
                and not use_expected_norm
                and initial_state is not None
                and final_state is not None
                and lower_bound == -5.0
                and num_heads == 96
                and offsets == [0, 1300, 1847, 3895, 4858, 5129, 8192]
                and n_split_chains == 64
                and inline_prefix == 4
                and len(cta_vals) == 148 * 3
            )
            if use_mixed_h96_gpc_swap:
                cta_rows = [
                    cta_vals[index : index + 3] for index in range(0, len(cta_vals), 3)
                ]
                cta_rows[88:108], cta_rows[128:148] = (
                    cta_rows[128:148],
                    cta_rows[88:108],
                )
                cta_vals = [value for row in cta_rows for value in row]
            self._args = dict(args_template)
            self._args["piece_table"] = torch.tensor(
                ordered_rows, dtype=torch.int32, device=q.device
            ).contiguous()
            self._args["cta_table"] = torch.tensor(
                cta_vals, dtype=torch.int32, device=q.device
            ).contiguous()
            self._args["initial_state"] = (
                initial_state.contiguous() if initial_state is not None else dummy
            )
            self._args["final_state"] = self._handoff_state
            self._grid = (len(cta_vals) // 3, 1, 1)
            rows = ordered_rows
            full_chunks = all((row[1] - row[0]) % K2_CHUNK == 0 for row in ordered_rows)
            use_mixed_cluster2 = (
                not allow_approximate_split
                and not use_expected_norm
                and initial_state is not None
                and lower_bound == -5.0
                and num_heads in (64, 96)
                and offsets == [0, 1300, 1847, 3895, 4858, 5129, 8192]
                and self._grid == (148, 1, 1)
                and not full_chunks
            )
            use_mixed_static_final = (
                not allow_approximate_split
                and not use_expected_norm
                and initial_state is not None
                and final_state is not None
                and lower_bound == -5.0
                and num_heads in (64, 96)
                and offsets == [0, 1300, 1847, 3895, 4858, 5129, 8192]
                and self._grid == (148, 1, 1)
                and not full_chunks
            )
            inline_compiler = compiled_stream_m128
            if use_mixed_cluster2:
                inline_compiler = compiled_stream_m128_cluster2
            uniform_regs = (
                {"regs_epilogue": 64, "regs_service": 40}
                if inline_compiler is compiled_stream_m128 and full_chunks
                else {}
            )
            use_named_mma_signals = (
                not allow_approximate_split
                and not use_expected_norm
                and initial_state is not None
                and final_state is not None
                and lower_bound == -5.0
                and num_heads in (64, 96)
                and self._grid == (148, 1, 1)
                and offsets
                in (
                    [0, 1300, 1847, 3895, 4858, 5129, 8192],
                    [0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192],
                )
            )
            use_named_out_empty = (
                use_named_mma_signals
                and inline_compiler is compiled_stream_m128
                and full_chunks
            )
            self._module = inline_compiler(
                full_chunks=full_chunks,
                store_final_state=True,
                lower_bound_m5=lower_bound == -5.0 and full_chunks,
                expected_norm=use_expected_norm,
                handoff=True,
                idle_state_prefetch=not use_expected_norm,
                first_state_prefetch=not use_expected_norm,
                static_final_export=use_mixed_static_final,
                named_mma_signals=use_named_mma_signals,
                named_out_empty=use_named_out_empty,
                **uniform_regs,
            )
            full_suffix = "_full" if full_chunks else ""
            nofs_suffix = "_nofs" if final_state is None else ""
            inline_tile = (
                "m128c2"
                if use_mixed_cluster2
                else "m128sf"
                if use_mixed_static_final
                else "m128"
            )
            self.schedule = (
                f"exact_inline_handoff_{inline_tile}_n{n_split_chains}"
                f"_c{inline_prefix}"
                f"_j{len(ordered_rows)}_b{self._grid[0]}"
                f"_span{inline_base_span}-{inline_span}"
                + ("_gpc57" if use_mixed_h96_gpc_swap else "")
                + full_suffix
                + nofs_suffix
            )
            self.state_input_mode = "exact_single_launch_native_fp32_handoff"
        elif tail_plan is None:
            self._args, self._grid, rows, full_chunks = make_phase(
                rows,
                initial_state.contiguous() if initial_state is not None else None,
                final_state.contiguous() if final_state is not None else None,
                0 if has_split else 1,
            )
            if use_fixed_h64_m64:
                # The retained fixed_h64 program is laid out for a 138-CTA
                # split-major grid; it needs the grid resident, not exactly
                # 148 SMs (152-SM GB300 parts otherwise silently fall back).
                use_split_major = (
                    final_state is not None
                    and num_sms >= 138
                    and self._grid == (64, 1, 1)
                )
                self._module = (
                    compiled_stream_m64_fixed()
                    if final_state is None
                    else compiled_stream_m64_fixed_final(split_major=use_split_major)
                )
                if use_split_major:
                    # Keep the 64 heads in order in each value half while
                    # placing five empty physical slots at 8--12 and 77--81.
                    cta_rows = self._args["cta_table"].reshape(-1, 3)
                    empty_rows = torch.zeros_like(cta_rows[:5])
                    half_rows = torch.cat((cta_rows[:8], empty_rows, cta_rows[8:]))
                    self._args["cta_table"] = torch.cat(
                        (half_rows, half_rows)
                    ).flatten()
                    self._grid = (2 * half_rows.shape[0], 1, 1)
                else:
                    self._grid = (self._grid[0] * 2, 1, 1)
            else:
                stream_compiler = compiled_stream_m128
                if use_fixed_h96_static_final:
                    self._module = compiled_stream_m128_fixed_h96_final()
                    stream_compiler = None
                elif use_fixed_h96_cluster8:
                    stream_compiler = compiled_stream_m128_cluster8
                if stream_compiler is not None:
                    self._module = stream_compiler(
                        full_chunks=full_chunks,
                        store_final_state=final_state is not None,
                        lower_bound_m5=lower_bound == -5.0 and full_chunks,
                        expected_norm=use_expected_norm,
                        idle_state_prefetch=(
                            not has_split
                            and not full_chunks
                            and initial_state is not None
                            and final_state is None
                            and not use_expected_norm
                        ),
                        first_state_prefetch=(
                            not has_split
                            and full_chunks
                            and initial_state is not None
                            and final_state is None
                            and not use_expected_norm
                            and num_heads == 96
                            and offsets == [0, 8192]
                        ),
                    )
        else:
            (
                phase_a,
                phase_b,
                tail_seqs,
                prefix_chunks,
                base_span,
                total_span,
                span_a,
            ) = tail_plan
            self._handoff_state = (
                final_state.contiguous()
                if final_state is not None
                else torch.empty(
                    len(offsets) - 1,
                    num_heads,
                    HEAD_DIM,
                    HEAD_DIM,
                    dtype=torch.float32,
                    device=q.device,
                )
            )
            self._args, self._grid, rows, full_chunks = make_phase(
                phase_a,
                initial_state.contiguous() if initial_state is not None else None,
                self._handoff_state,
                1,
            )
            (
                self._tail_args,
                self._tail_grid,
                tail_rows,
                tail_full_chunks,
            ) = make_phase(
                phase_b,
                self._handoff_state,
                final_state.contiguous() if final_state is not None else None,
                1,
            )
            self._module = compiled_stream_m128(
                full_chunks=full_chunks,
                store_final_state=True,
                lower_bound_m5=lower_bound == -5.0 and full_chunks,
                expected_norm=use_expected_norm,
            )
            self._tail_module = compiled_stream_m128(
                full_chunks=tail_full_chunks,
                store_final_state=final_state is not None,
                lower_bound_m5=lower_bound == -5.0 and tail_full_chunks,
                expected_norm=use_expected_norm,
            )
            self.schedule = (
                f"exact_tail_stream_m128_q{tail_seqs[0]}x{len(tail_seqs)}"
                f"_c{'-'.join(str(p) for p in prefix_chunks)}"
                f"_j{len(rows)}+{len(tail_rows)}"
                f"_b{self._grid[0]}+{self._tail_grid[0]}"
                f"_span{base_span}-{span_a}-{total_span - span_a}"
                f"{'_nofs' if final_state is None else ''}"
            )
            self.state_input_mode = "exact_two_launch_native_fp32_handoff"
        if inline_plan is None and tail_plan is None:
            stream_mode = "dynamic" if has_split else "exact"
            stream_tile = "m128"
            if use_fixed_h64_m64:
                stream_tile = "m64"
            elif use_fixed_h96_static_final:
                stream_tile = "m128sf"
            elif use_fixed_h96_cluster8:
                stream_tile = "m128c8"
            self.schedule = (
                f"{stream_mode}_stream_{stream_tile}_j{len(rows)}"
                f"_b{self._grid[0]}_s{n_split}"
                f"{'_full' if full_chunks else ''}"
                f"{'_nofs' if final_state is None else ''}"
            )
            self.state_input_mode = (
                "dynamic_g_inkernel_checked_native_fp32"
                if has_split
                else "exact_unsplit_native_fp32"
            )
            if self._module is None:
                # No retained static program covers this route; launch()
                # runs the exact fallback, so report that instead of the
                # stream schedule that was not bound.
                self.schedule = f"no_static_program_{self._fallback.schedule}"
                self.state_input_mode = "timed_fp32_bf16_boundary_conversions"
        if self._windows and any(
            start - walk != K2_CHUNK for walk, start in self._windows
        ):
            raise RuntimeError("dynamic-g plan requires one-chunk warmups")
        self._generation = 0
        self._failure_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
        self._done_event = torch.cuda.Event()
        for stream_args in (self._args, self._tail_args):
            if stream_args is None:
                continue
            stream_args["failure_addr"] = self._failure_host.data_ptr()
            stream_args["generation"] = 0
            stream_args["detect_threshold_log2"] = DETECT_THRESHOLD_LOG2
            # Shared argument plan; programs without handoff flags ignore it.
            stream_args["handoff_flags_addr"] = (
                self._handoff_flags.data_ptr()
                if self._handoff_flags is not None
                else 0
            )
        # Upload the TMA descriptor tables while planning, outside any capture.
        for module, stream_args in (
            (self._module, self._args),
            (self._tail_module, self._tail_args),
        ):
            if module is not None and stream_args is not None:
                module.bind(stream_args)

    def update_g(self, g):
        """Bind a new contiguous BF16 gate tensor without rebuilding the plan."""

        if tuple(g.shape) != self._g_shape or g.dtype != self._g_dtype:
            raise ValueError(
                f"g must retain shape {self._g_shape} and dtype {self._g_dtype}"
            )
        if not g.is_cuda or not g.is_contiguous():
            raise ValueError("g must be a contiguous CUDA tensor")
        self._g = g
        g_flat = g.reshape(-1, self._num_heads, HEAD_DIM)
        self._fallback.args["g"] = g_flat
        self._fallback.args["g_tma"] = g_flat
        if self._module is not None:
            self._args["g"] = g_flat
            self._args["g_tma"] = g_flat
            if self._tail_args is not None:
                self._tail_args["g"] = g_flat
                self._tail_args["g_tma"] = g_flat

    def _launch_fallback(self):
        """Run the exact unsplit schedule with its BF16 state staging.

        :return: None.
        :rtype: NoneType
        """
        if self._staging_in is not None:
            self._staging_in.copy_(self._initial_state)
        self._fallback.launch()
        if self.final_state is not None:
            self.final_state.copy_(self._staging_out)

    def _stream_passes(self):
        """Launch the checked stream and read the in-kernel proof verdict.

        :return: True when no warmup window published a failure.
        :rtype: bool
        """
        if self._handoff_flags is not None:
            self._handoff_flags.zero_()
        self._generation += 1
        self._args["generation"] = self._generation
        self._module.launch(grid=self._grid, **self._args)
        if self._tail_module is not None:
            self._tail_args["generation"] = self._generation
            self._tail_module.launch(grid=self._tail_grid, **self._tail_args)
        if not self._windows:
            return True
        self._done_event.record()
        self._done_event.synchronize()
        return int(self._failure_host[0]) != self._generation

    def revalidate(self):
        """Recheck every proof window against the current gate tensor.

        The proof lives inside the stream kernel, so this replays one full
        checked launch (overwriting the bound output tensor).

        :return: True when all windows still satisfy the decay threshold.
        :rtype: bool
        """
        if self._module is None:
            return True
        return self._stream_passes()

    def launch(self, g=None):
        """Launch the exact stream or checked split; rerun on proof failure.

        :return: None.
        :rtype: NoneType
        """
        if g is not None:
            self.update_g(g)
        if self._module is None:
            self._launch_fallback()
            return
        if not self._stream_passes():
            self._launch_fallback()

    def close(self):
        """Release nothing; FFI modules are process-resident.

        :return: None.
        :rtype: NoneType
        """
        return


def prepare_fwd_pieces(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    allow_approximate_split=False,
    use_expected_norm=False,
):
    """Create a preallocated KDA forward launch.

    :param q: Packed BF16 query tensor.
    :type q: torch.Tensor
    :param k: Packed BF16 key tensor.
    :type k: torch.Tensor
    :param v: Packed BF16 value tensor.
    :type v: torch.Tensor
    :param g: Packed BF16 gate logit tensor.
    :type g: torch.Tensor
    :param beta: Packed BF16 beta logit tensor.
    :type beta: torch.Tensor
    :param scale: Attention scale.
    :type scale: float
    :param out: Preallocated BF16 output tensor.
    :type out: torch.Tensor
    :param A_log: FP32 per-head gate rate logits.
    :type A_log: torch.Tensor
    :param dt_bias: FP32 per-head gate biases.
    :type dt_bias: torch.Tensor
    :param lower_bound: Safe-gate lower bound.
    :type lower_bound: float
    :param initial_state: Optional FP32 initial state [N, H, 128, 128].
    :type initial_state: torch.Tensor
    :param final_state: Optional FP32 final-state output [N, H, 128, 128].
    :type final_state: torch.Tensor
    :param cu_seqlens: Optional packed sequence boundaries.
    :type cu_seqlens: torch.Tensor
    :param allow_approximate_split: Reserved; must be False.
    :param use_expected_norm: Reserved; must be False.
    :return: Launch object with a launch() method.
    :rtype: KDAPiecesLaunch

    .. code-block:: python

        launch = prepare_fwd_pieces(q, k, v, g, beta, scale, out, A_log,
                                    dt_bias, -5.0, initial_state=h0)
        launch.launch()
    """
    return KDAPiecesLaunch(
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        allow_approximate_split=allow_approximate_split,
        use_expected_norm=use_expected_norm,
    )


__all__ = ["KDAPiecesLaunch", "pack_bins", "plan_pieces", "prepare_fwd_pieces"]
