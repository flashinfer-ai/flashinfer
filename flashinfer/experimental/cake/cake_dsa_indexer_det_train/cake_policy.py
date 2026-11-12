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

Host dispatch policy of the Cake DSA indexer top-k programs.

The generated scan kernel exists in several physical forms (unit geometry,
key-range split, CTA pair, tile-loop unroll, snake unit order, sampled first
threshold on buffer-fitting units), the finalize in two families (the CUB
block radix sort programs and the prefix-popcount rank finalize programs, one
per bitmap window), and the host picks one per call from the call's geometry
alone: ``T`` queries, ``Tkv`` keys, ``S`` segments, the compression ``ratio``,
``top_k`` and the persistent grid (the device's SM count).  Every threshold of that choice is a plain number in the registry
record of the architecture (``cake_jit.POLICY[arch]``); this module evaluates
the record.  It depends on nothing but the standard library so that the
generated-program export can run the very same code against the kernel's own
dispatch functions before a record is frozen (a disagreement fails the export,
never the user).  A record may give the wide program precedence over the
three-warpgroup rule from ``wide_first_rule[0]`` mean unit tiles on, or
replace both wide rules by ``wide_wave_rule``, which compares the waves of the
persistent grid the wide unit needs with those of the unit the other rules
leave (``unit_block_q_for``), the
CTA pair serves the unit geometries of ``pair_units`` only and the key-range
split those of ``split_units`` only (the units whose split program the
architecture registers).  ``l6_solo_rule`` may prefer the three-warpgroup unit
without a CTA pair on calls that do not split, and ``l6_solo_unsampled_max_queries``
the unpaired form of a CTA-pair l6 plan on short calls with the sampled first threshold off (``select_program``, which takes
the registered scan keys so that it never picks an unregistered program;
without them the preference stays off).  ``snake_packed_rule`` bounds the
snake unit order of packed calls by their mean unit tiles, sampling mode and
waves of the persistent grid,
and ``snake_l6_fallback_waves`` lets a sampling-off call of few waves that the
l6 preference moves run the plain-order l6 program where the snake-order one
is not registered.  A record may also
replace the sampled rank shift from a ``top_k`` of ``sample_shift_rule[0]`` on
(``sample_shift_for``); an explicit ``sample_shift_permille`` knob still
overrides it.  The candidate-capacity floor may depend on ``top_k`` the same
way: ``cand_cap_floor_rule`` = ``[min top_k, floor]`` replaces
``cand_cap_floor`` from that ``top_k`` on (``cand_cap_floor_for``), so that a
small-``top_k`` call keeps the smaller buffers (half the workspace at
``top_k = 512``) and a large-``top_k`` call the larger floor.

Nothing here reads device memory: the segment boundaries stay on the device
and the dispatch sees only host-known integers.  The chosen program never
changes a result -- every form computes the same ids and score bits -- so a
wrong choice costs time, not correctness.

The opt-in FP8 input form (float8_e4m3fn ``q`` / ``k``) has its own scan
programs, keyed ``scan:<kind>:u<unroll>:s<snake>:f1:fp8[:lsf]``, for the narrow,
l6, split and wide kinds, in the plain unit order (``s0``) but for the snake copy
(``s1``) of the sm_107a l6 program.  An FP8 call runs the same decision procedure under the
architecture's FP8 record (``fp8_policy_record``: the ``POLICY`` record with
the fields of the registry's ``POLICY_FP8`` record replaced) and then the
registered FP8 program of the decided kind, or of its ``FP8_KIND_FALLBACK``
kind (``select_fp8_program``).  Every FP8 program of one architecture computes
the same ids and score bits.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any, Collection, NamedTuple, Optional, Tuple

# Record fields in canonical order (the export writes them in this order).
POLICY_FIELDS = (
    "tile_keys",  # keys per key tile (capacity and sampling granule)
    "block_q_narrow",  # queries per unit of the narrow program (two math warpgroups, one query pair each)
    "block_q_wide",  # queries per unit of the wide program (M128 x N256 MMA, staged weights)
    "block_q_l6",  # queries per unit of the three-warpgroup program
    "candidate_entry_bytes",  # packed (score bits, key id) entry size
    "candidate_multiplier",  # default capacity multiplier: cand_cap ~ multiplier * top_k
    "candidate_slack",  # minimum headroom above top_k before a buffer can be compacted
    "cand_mult_rule",  # [min mean unit tiles, multiplier] for long-unit calls, or None
    "cand_cap_floor",  # minimum candidate slots per (warpgroup, query), or None
    "cand_cap_floor_rule",  # [min top_k, floor] replacing cand_cap_floor from that top_k on (sampled calls), or None
    "candidate_multiplier_unsampled",  # capacity multiplier when the sampled first threshold is off
    "cand_mult_rule_unsampled",  # its [min mean unit tiles, multiplier] long-unit rule, or None
    "cand_cap_floor_unsampled",  # its minimum candidate slots, or None
    "cand_cap_ceiling",  # most candidate slots the default multiplier requests per (warpgroup, query), or None; lowers the multiplier, the floor and top_k + slack still hold
    "cand_cap_ceiling_unsampled",  # its ceiling when the sampled first threshold is off, or None
    "l6_rule",  # [max mean unit tiles, min top_k] of the three-warpgroup program, or None
    "wide_rule",  # [min wide rounds per CTA, min rounds x mean unit tiles] of the wide program
    "wide_first_rule",  # [min mean unit tiles] from which a wide-eligible call takes the wide program before the three-warpgroup rule, or None (the three-warpgroup rule first)
    "wide_wave_rule",  # [min narrow / wide wave ratio, min l6 / wide wave ratio, min mean unit tiles against l6(, the top_k up to which the tile minima hold, their exponent in top_k above it, the minimum of a packed call, the tiles a one-segment minimum rises per wide wave below the next element, the wide waves from which it holds, the l6 / wide wave ratio W below which the minimum is at least the next element / (W - 1) and on a packed call at least element 10 / (W - 1) ** 2, and from which the minimum of a one-document call of fewer wide waves than element 7 is at most element 11 / (W - 1) ** 2, on two or more wide waves at most the larger of that and element 12, from element 13 wide waves on of that and element 14)] replacing wide_rule and wide_first_rule, or None; see unit_block_q_for
    "desc_walk_kinds",  # [physical scan kinds whose descending-walk twin (key kind "<kind>_desc") a call with the sampled first threshold off runs, the same for a sampled call], or None; see select_program
    "split_max",  # most key-range splits per unit
    "split_min_range_tiles",  # a split's key range is at least this many tiles
    "split_wave_rule",  # [min mean unit tiles, min CTA-efficiency gain] for a two-way split, or None
    "split_wave_scope",  # [unit geometries, max segments, sampled calls only] the two-way split of split_wave_rule is limited to, or None (every call)
    "split_fit_rule",  # [merge chunk entries]: a call the split rule splits takes ceil(mean unit keys / top_k) splits, rounded up to whole merge chunks, where that is more and fits split_max and the grid; or None
    "split_resident_rule",  # [merge chunk entries, fit entries, merge-resident entries, max splits, min splits] replacing split_max and split_fit_rule on a call the split rule splits, or None; see split_rule_for
    "split_units",  # unit geometries ("l6", "narrow", "wide") whose split program the architecture registers; any other unit never splits
    "pair_rule",  # [min mean unit tiles, max mean unit tiles, min top_k] of the CTA-pair program, or None
    "pair_units",  # unit geometries ("l6", "narrow", "wide") whose CTA-pair program the architecture registers; the pair rule applies to these units only
    "l6_solo_rule",  # [max mean unit tiles, min segments] of the unpaired three-warpgroup preference on sampled calls, or None (no preference); see select_program
    "l6_solo_unsampled_max_queries",  # with the sampled first threshold off, a call of at most this many queries planned on the CTA-pair l6 unit runs it unpaired (other units keep their plan), or None (sampling off follows l6_solo_rule); see l6_solo_for
    "wide_pair_snake_only",  # a sampled call pairs the wide unit only when it walks the snake order (else the unpaired wide program)
    "snake_default",  # snake unit order when no rule applies
    "snake_rule",  # [program kinds, min mean tiles, min mean tiles (ratio > 1), min unit-cost spread], or None
    "snake_cap_rule",  # [max segments, max waves of the persistent grid] within which an FP8 call the decision marks snake runs the snake copy of its program, or None (no cap); read only by select_fp8_program
    "snake_packed_rule",  # [program kinds, min mean unit tiles sampled, min mean unit tiles with sampling off or None (never), max waves or None] a call of two or more segments needs for the snake order of snake_rule beyond that many waves of its persistent grid, or None (snake_rule alone); see snake_for
    "snake_l6_fallback_waves",  # max waves of the planned program's persistent grid within which a sampling-off call the l6 preference moves runs the plain-order l6 program where the snake-order one is not registered, or None; see select_program
    "tile_unroll_default",  # tile-loop unroll when no rule applies
    "tile_unroll_factor",  # tile-loop unroll when the rule applies
    "tile_unroll_rule",  # [program kinds, min top_k, yields to the snake order], or None
    "sample_fit_max_mean_tiles",  # sampled first threshold on buffer-fitting units up to this mean tile count
    "sample_fit_kinds",  # physical scan kinds whose sampled-fit program the architecture registers; every other kind runs its plain program
    "sample_tiles_max",  # sample-tile cap of long units (0 = no sampling)
    "sample_tiles_short_units",  # sample-tile cap of short units
    "sample_dispatch_mean_tiles_max",  # mean unit tiles up to which the short-unit cap applies
    "sample_tiles_tiny_units",  # sample-tile cap of tiny units (lever K3b)
    "sample_dispatch_tiny_tiles_max",  # mean unit tiles up to which the tiny-unit cap applies
    "sample_shift_permille",  # conservative shift of the sampled rank
    "sample_class_rule",  # [tiny-unit sample tiles, shift permille] for the wide-unit programs (shift only) and the tiny dispatch class (tiles and shift), or None; explicit knobs override it
    "sample_long_rule",  # [physical scan kinds, shift permille] of the long-unit calls of those kinds (more than sample_dispatch_mean_tiles_max mean unit tiles), or None; an explicit shift knob overrides it
    "sample_shift_rule",  # [min top_k, shift permille] replacing sample_shift_permille from that top_k on, or None (the default shift on every call); an explicit sample_shift_permille knob overrides both
    "lsf_programs",  # the architecture ships longest-first segment order copies (scan key suffix :lsf) of its scan programs; the host runs the decided program's copy on packed calls whose key segments differ in length (#7549), the program itself where the registry has no copy of it
    "lsf_single_rule",  # [min mean unit tiles] from which a one-segment FP8 call of the wide unit runs the :lsf copy of its program, or None (never); read only by select_fp8_program
    "static_scan_kinds",  # physical scan kinds whose programs keep the static unit enumeration on a ticket architecture: they never touch the ticket tail, so the host gives their runners none
    "far_sample",  # 1: the sampled first threshold of the ascending-tile-order programs ranks the far sample tiles only (the tile nearest the diagonal left out); 0: every sample tile
    "check_period_max",  # cap on the default selection-trigger check period (tiles)
    "check_period_knob_max",  # cap on an explicit check period (tiles)
    "check_period_cap_divisor",  # default period = cand_cap // divisor
    "check_period_kind_overrides",  # {program kind: period} overrides, clamped to the exactness bound
    "finalize_items",  # sort slots per finalize thread
    "finalize_threads_fit",  # right-sized finalize programs (threads) for small top_k
    "finalize_threads_small",  # finalize threads up to finalize_top_k_small
    "finalize_top_k_small",  # largest top_k of the small finalize program
    "finalize_threads",  # finalize threads above finalize_top_k_small
    "finalize_fit",  # small top_k takes the right-sized finalize program
    "finalize_exact_key_bits",  # sort (Tkv - 1).bit_length() key bits instead of Tkv.bit_length()
    "rank_finalize",  # the prefix-popcount rank finalize serves the calls rank_rule admits (False: every call sorts with CUB)
    "rank_window_variants",  # bitmap pool windows (bits) of the rank finalize program variants, ascending
    "rank_top_k_min",  # smallest top_k the rank finalize serves (below: the right-sized CUB programs)
    "rank_rule",  # [max pool window (Tkv), max mean segment keys (Tkv / S)] of the rank finalize, or None
    "rank_staged",  # the rank finalize runs its SMEM-staged, coalesced-I/O form where rank_staged_rule admits the window (lever FRS)
    "rank_staged_rule",  # [max pool window whose staged rank finalize form keeps two CTAs per SM], or None
    "rank_seg_window",  # the rank finalize's pool is sized from the call's max_seqlen_k when given (lever RW); False: the Tkv pool
    "rank_window_variants_seg_only",  # rank_window_variants entries the Tkv-pool rule skips (reached through max_seqlen_k only)
    "rank_two_level",  # the two-level (bucket + word) rank finalize serves the pools above rank_slab_window_max (lever FR2)
    "rank_two_level_window_variants",  # bitmap pool windows (bits) of the two-level rank finalize program variants, ascending
    "rank_slab_window_max",  # largest pool window of the slab (single-level) rank finalize form under lever FR2
    "rank_two_level_rule",  # [max pool window of the two-level rank finalize (no segment bound)], or None
    "rank_two_level_staged_rule",  # [max two-level pool window whose staged form keeps two CTAs per SM], or None
    "rank_bulk_io",  # the staged rank finalize moves its rows with cp.async.bulk where rank_bulk_align_bytes admits the call (lever FRB)
    "rank_bulk_align_bytes",  # cp.async.bulk address / size granule (bytes): top_k x 4 and the output addresses must be multiples of it
    "rank_t16",  # the staged rank finalize runs its 16-item thread form on the sort-slot counts of rank_t16_slots (lever FRT)
    "rank_t16_slots",  # sort-slot counts (threads x finalize_items) whose staged form runs as 16-item threads (half the threads), ascending
    "rank_persist_max_k",  # the persistent two-row pipelined rank finalize serves the staged bulk calls with top_k <= this on the architecture (lever FRP-K; 0 = the one-row program everywhere; program key tag :persist)
    "rank_persist_ctas_per_sm",  # persistent CTAs per SM: the :persist programs launch min(num_queries, this x (SM count - rank_persist_spare_sms)) CTAs, each walking rows bid, bid + grid, ...
    "rank_persist_spare_sms",  # SMs the persistent rank grid is sized without: the grid is a multiple of the other SMs, so a block of another stream on up to this many SMs adds no partial wave to its walk (#7548)
    "merge_rank",  # a split call whose finalize is the slab rank form (window <= rank_slab_window_max, top_k <= MERGE_RANK_MAX_TOP_K) runs the merge with the rank finalize fused in (key merge:rank:w<window>) and launches no finalize program
)

KINDS = ("narrow", "wide", "l6", "split", "pair_narrow", "pair_wide", "pair_l6")
UNITS = ("narrow", "wide", "l6")
# Names of the compiled scan program kinds (the split program is one per unit geometry).
PHYSICAL_KINDS = tuple(
    sorted(set(KINDS) - {"split"} | {f"split_{unit}" for unit in UNITS})
)
MERGE_KEY = "merge"
# The merge with the rank finalize fused in leaves each split row in its final ascending-ID layout (the slab rank finalize's
# bits); its programs hold up to this many kept entries per row in shared memory.
MERGE_RANK_MAX_TOP_K = 2048
# Input forms of a call, selected by its q / k dtype: ``bf16`` (the default) and ``fp8`` (float8_e4m3fn q / k with
# per-key FP32 scales, opt-in).  An FP8 scan program's key carries the form tag before the LSF suffix,
# ``scan:<kind>:u<unroll>:s<snake>:f<fit>:fp8[:lsf]``; the merge and finalize programs serve both forms.
BF16 = "bf16"
FP8 = "fp8"
FORMS = (BF16, FP8)
# The FP8 scan programs exist for the narrow, l6, split and wide kinds, in the plain unit order (sm_107a also registers the
# snake copy of its l6 program, which ``select_fp8_program`` runs for a snake-marked call within ``snake_cap_rule``) and in
# the sampled-fit form.
# An FP8 call runs the decision procedure of ``select_program`` under the FP8 record (``fp8_policy_record``); a decided
# kind without an FP8 program runs the FP8 program of its fallback kind below (unpaired; the same unit geometry where it
# has one, except that a CTA-pair wide unit runs l6).  An unpaired, unsplit wide unit (8 queries) runs the FP8 wide
# program where the architecture registers one (sm_100a, sm_103a and sm_107a); else it, and a CTA-pair wide unit always,
# runs the l6 unit (6 queries), the largest FP8 unit below the wide one: per key tile it reads K once for 6 queries
# rather than for the narrow unit's 4.  A split wide unit (sm_107a only) runs split_narrow: no architecture that splits
# wide units registers an FP8 split_l6 program.
FP8_KIND_FALLBACK = {
    "wide": "l6",
    "pair_l6": "l6",
    "pair_narrow": "narrow",
    "pair_wide": "l6",
    "split_wide": "split_narrow",
    "split_l6": "split_narrow",
}
# Admissible sampled rank shift (permille): the backend's ``sample_shift_permille`` knob bound and the bound of a record's
# ``sample_shift_rule`` shift.  Every value gives the same bits; the shift only sets how conservative the sampled first
# threshold is.
SAMPLE_SHIFT_PERMILLE_BOUNDS = (-1000, 10000)
# Global unit tickets (every scan program but the kinds of a record's
# ``static_scan_kinds``, on ``cake_jit.TICKET_TAIL_ARCHES``): the CTAs draw every
# work unit from the low u32 word of the first int64 behind the candidate buffers
# (and behind the split staging), the first of a tail every plan's workspace
# carries: 16 int64 for the ticket word, then the longest-first segment order the
# scan prologue writes (one int64 per segment, up to 1024 segments).
# Every completed scan launch leaves the word at 0; the host (``cake_backend``)
# zeroes the tail once at prepare on a backend-owned workspace and before every
# run on a caller-owned ``workspace_buffer``, which others may write between
# runs -- except where the call always takes the static fast path
# (``ticket_static_path``: the scan never touches the word) or runs a program of
# ``static_scan_kinds``.  A scan program whose registry record lists the
# architecture in ``ticket_arg_arches`` draws from its ``TicketWord`` block
# instead and never draws from this tail (the eager entry hands it the tail's
# first 16 bytes); its :lsf copies (sm_107a's l6 ticket programs) still keep the
# segment order in the tail, which every launch writes before it reads, so the
# host zeroes nothing for them.  Every plan carries the tail, so the size does
# not depend on the program.
TICKET_TAIL_BYTES = 8 * (16 + 1024)


def ticket_units_bound(choice: "ProgramChoice") -> int:
    """An upper bound, from the call's shape alone, on the work units ``n_total`` a ticket scan launch of ``choice``
    enumerates.  The kernel counts ``n_total`` from ``cu_seqlens_q`` on the device: ``ceil(l_s / block_q)`` blocks per
    segment, times ``n_split`` on a split plan, ``ceil(blocks / 2)`` per segment on a CTA pair.  For any segment
    lengths ``l_s >= 0`` with ``sum l_s <= T``, ``sum ceil(l_s / bq) <= floor((T + S (bq - 1)) / bq)`` and
    ``sum ceil(b_s / 2) <= floor((sum b_s + S) / 2)``; the bound reads no tensor, so a caller that rewrites the
    boundaries in place cannot move a call across it."""
    bq = int(choice.block_q)
    blocks = (int(choice.num_queries) + int(choice.num_segments) * (bq - 1)) // bq
    if choice.split:
        return blocks * int(choice.n_split)
    if choice.pair:
        return (blocks + int(choice.num_segments)) // 2
    return blocks


def ticket_static_path(choice: "ProgramChoice") -> bool:
    """True when every launch of ``choice`` takes the ticket scan's static fast path: ``n_total <= G`` (``G`` = the
    launched grid, half of it on a CTA pair), where the kernel gives every CTA / cluster at most one unit by its index
    and never reads or writes the ticket word (no draw, no publish).  Decided by ``ticket_units_bound``."""
    agents = int(choice.grid_ctas) // 2 if choice.pair else int(choice.grid_ctas)
    return ticket_units_bound(choice) <= agents


# sample_class_rule's tiny-class part applies to packed calls of short query segments only (#7549)
CLASS_RULE_MIN_SEGMENTS = 2
CLASS_RULE_MAX_QUERIES_PER_SEGMENT = 4096
# lever LSF: the longest-first segment order copy of a scan program runs on calls of at least this many segments
LSF_MIN_SEGMENTS = 2


def sample_class_for(
    policy: "DispatchPolicy",
    physical: str,
    num_queries: int,
    num_keys: int,
    num_segments: int,
    ratio: int,
    tiles: int,
    shift: int,
    *,
    tiles_knob: bool,
    shift_knob: bool,
) -> tuple[int, int]:
    """The sampled first threshold ``(tiles, shift)`` of a scan program of the physical kind ``physical`` under the
    record's ``sample_class_rule`` and ``sample_long_rule``, from the call's ``tiles`` / ``shift``: the wide-unit
    programs take the class rule's shift on packed calls (at least ``CLASS_RULE_MIN_SEGMENTS`` segments), the tiny
    dispatch class (packed calls of short query segments and few mean unit tiles) also its tile count -- where that
    count exceeds ``sample_tiles_tiny_units``, only on calls of more mean unit tiles than it; a long-unit call
    (more than ``sample_dispatch_mean_tiles_max`` mean unit tiles) of a kind the long rule names takes its shift.  A
    knob the caller set (``tiles_knob`` / ``shift_knob``) stays; without the rules or with sampling off nothing
    changes."""
    class_rule, long_rule = policy.sample_class_rule, policy.sample_long_rule
    T, S = int(num_queries), int(num_segments)
    if tiles <= 0:
        return tiles, shift
    # the long-unit calls of the long rule's kinds shift further (#7549 on sm_107a)
    if (
        long_rule is not None
        and physical in long_rule[0]
        and policy.mean_unit_tiles(T, num_keys, S, ratio)
        > policy.sample_dispatch_mean_tiles_max
    ):
        if not shift_knob:
            shift = int(long_rule[1])
        return tiles, shift
    if class_rule is None:
        return tiles, shift
    # the wide-unit programs shift only; the tiny dispatch class also samples more tiles (#7549)
    mean = policy.mean_unit_tiles(T, num_keys, S, ratio)
    if "wide" in physical:
        if not shift_knob and S >= CLASS_RULE_MIN_SEGMENTS:
            shift = int(class_rule[1])
    elif (
        not tiles_knob
        and mean <= policy.sample_dispatch_tiny_tiles_max
        and S >= CLASS_RULE_MIN_SEGMENTS
        and T <= CLASS_RULE_MAX_QUERIES_PER_SEGMENT * S
        # a rule that samples more tiles than the tiny default applies only where the mean unit holds more than that
        # many tiles: on shorter units the larger sample covers whole units (the ratio-4 pooled packed calls)
        and not (
            int(class_rule[0]) > policy.sample_tiles_tiny_units
            and mean <= int(class_rule[0])
        )
    ):
        tiles = int(class_rule[0])
        if not shift_knob:
            shift = int(class_rule[1])
    return tiles, shift


_INT = ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes", "candidate_multiplier",
        "candidate_multiplier_unsampled", "candidate_slack", "split_max", "split_min_range_tiles", "tile_unroll_default", "tile_unroll_factor",
        "sample_tiles_max", "sample_tiles_short_units", "sample_tiles_tiny_units", "sample_shift_permille", "far_sample",
        "check_period_max", "check_period_knob_max", "check_period_cap_divisor", "finalize_items",
        "finalize_threads_small", "finalize_top_k_small", "finalize_threads", "rank_top_k_min",
        "rank_slab_window_max", "rank_bulk_align_bytes", "rank_persist_max_k", "rank_persist_ctas_per_sm",
        "rank_persist_spare_sms")  # fmt: skip
_BOOL = (
    "wide_pair_snake_only",
    "snake_default",
    "lsf_programs",
    "finalize_fit",
    "finalize_exact_key_bits",
    "rank_finalize",
    "rank_staged",
    "rank_seg_window",
    "rank_two_level",
    "rank_bulk_io",
    "rank_t16",
    "merge_rank",
)
_NUMBER = (
    "sample_fit_max_mean_tiles",
    "sample_dispatch_mean_tiles_max",
    "sample_dispatch_tiny_tiles_max",
)
_OPTIONAL_INT = (
    "l6_solo_unsampled_max_queries",
    "snake_l6_fallback_waves",
    "cand_cap_floor",
    "cand_cap_floor_unsampled",
    "cand_cap_ceiling",
    "cand_cap_ceiling_unsampled",
)
_OPTIONAL_RULE = (
    "cand_mult_rule",
    "cand_cap_floor_rule",
    "cand_mult_rule_unsampled",
    "l6_rule",
    "wide_first_rule",
    "wide_wave_rule",
    "desc_walk_kinds",
    "split_wave_rule",
    "split_wave_scope",
    "split_fit_rule",
    "split_resident_rule",
    "pair_rule",
    "l6_solo_rule",
    "snake_rule",
    "snake_cap_rule",
    "lsf_single_rule",
    "snake_packed_rule",
    "tile_unroll_rule",
    "sample_shift_rule",
    "sample_class_rule",
    "sample_long_rule",
    "rank_rule",
    "rank_staged_rule",
    "rank_two_level_rule",
    "rank_two_level_staged_rule",
)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


@dataclass(frozen=True)
class DispatchPolicy:
    """The host dispatch rules of one architecture, evaluated from the registry record.

    Every method mirrors the kernel's own host function of the same name; the
    export verifies the mirror against the kernel before the record is frozen.
    """

    tile_keys: int
    block_q_narrow: int
    block_q_wide: int
    block_q_l6: int
    candidate_entry_bytes: int
    candidate_multiplier: int
    candidate_slack: int
    cand_mult_rule: Optional[list]
    cand_cap_floor: Optional[int]
    cand_cap_floor_rule: Optional[list]
    candidate_multiplier_unsampled: int
    cand_mult_rule_unsampled: Optional[list]
    cand_cap_floor_unsampled: Optional[int]
    cand_cap_ceiling: Optional[int]
    cand_cap_ceiling_unsampled: Optional[int]
    l6_rule: Optional[list]
    wide_rule: list
    wide_first_rule: Optional[list]
    wide_wave_rule: Optional[list]
    desc_walk_kinds: Optional[list]
    split_max: int
    split_min_range_tiles: int
    split_wave_rule: Optional[list]
    split_wave_scope: Optional[list]
    split_fit_rule: Optional[list]
    split_resident_rule: Optional[list]
    split_units: list
    pair_rule: Optional[list]
    pair_units: list
    l6_solo_rule: Optional[list]
    l6_solo_unsampled_max_queries: Optional[int]
    wide_pair_snake_only: bool
    snake_default: bool
    snake_rule: Optional[list]
    snake_cap_rule: Optional[list]
    snake_packed_rule: Optional[list]
    snake_l6_fallback_waves: Optional[int]
    tile_unroll_default: int
    tile_unroll_factor: int
    tile_unroll_rule: Optional[list]
    sample_fit_max_mean_tiles: float
    sample_fit_kinds: list
    sample_tiles_max: int
    sample_tiles_short_units: int
    sample_dispatch_mean_tiles_max: float
    sample_tiles_tiny_units: int
    sample_dispatch_tiny_tiles_max: float
    sample_shift_permille: int
    sample_shift_rule: Optional[list]
    sample_class_rule: Optional[list]
    sample_long_rule: Optional[list]
    lsf_programs: bool
    lsf_single_rule: Optional[list]
    static_scan_kinds: list
    far_sample: int
    check_period_max: int
    check_period_knob_max: int
    check_period_cap_divisor: int
    check_period_kind_overrides: dict
    finalize_items: int
    finalize_threads_fit: list
    finalize_threads_small: int
    finalize_top_k_small: int
    finalize_threads: int
    finalize_fit: bool
    finalize_exact_key_bits: bool
    rank_finalize: bool
    rank_window_variants: list
    rank_top_k_min: int
    rank_rule: Optional[list]
    rank_staged: bool
    rank_staged_rule: Optional[list]
    rank_seg_window: bool
    rank_window_variants_seg_only: list
    rank_two_level: bool
    rank_two_level_window_variants: list
    rank_slab_window_max: int
    rank_two_level_rule: Optional[list]
    rank_two_level_staged_rule: Optional[list]
    rank_bulk_io: bool
    rank_bulk_align_bytes: int
    rank_t16: bool
    rank_t16_slots: list
    rank_persist_max_k: int
    rank_persist_ctas_per_sm: int
    rank_persist_spare_sms: int
    merge_rank: bool

    @classmethod
    def from_record(cls, raw: Any) -> "DispatchPolicy":
        if not isinstance(raw, dict) or set(raw) != set(POLICY_FIELDS):
            raise ValueError(
                f"dispatch policy record must carry exactly the fields {POLICY_FIELDS}, got {raw!r}"
            )
        for name in _INT:
            if not _is_int(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be an integer, got {raw[name]!r}"
                )
        for name in _BOOL:
            if not isinstance(raw[name], bool):
                raise ValueError(f"policy[{name!r}] must be a bool, got {raw[name]!r}")
        for name in _NUMBER:
            if not _is_number(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be a number, got {raw[name]!r}"
                )
        for name in _OPTIONAL_INT:
            if raw[name] is not None and not _is_int(raw[name]):
                raise ValueError(
                    f"policy[{name!r}] must be an integer or None, got {raw[name]!r}"
                )
        for name in _OPTIONAL_RULE:
            if raw[name] is not None and not isinstance(raw[name], (list, tuple)):
                raise ValueError(
                    f"policy[{name!r}] must be a rule list or None, got {raw[name]!r}"
                )
        if (
            not isinstance(raw["wide_rule"], (list, tuple))
            or len(raw["wide_rule"]) != 2
        ):
            raise ValueError(
                f"policy['wide_rule'] must be [min rounds, min work], got {raw['wide_rule']!r}"
            )
        first = raw["wide_first_rule"]
        if first is not None and (
            len(first) != 1 or not _is_number(first[0]) or first[0] < 0
        ):
            raise ValueError(
                f"policy['wide_first_rule'] must be [min mean unit tiles] or None, got {first!r}"
            )
        waves = raw["wide_wave_rule"]
        if waves is not None and (
            len(waves) not in (3, 15)
            or not all(_is_number(v) and v >= 0 for v in waves)
            or (
                len(waves) == 15
                and not (
                    _is_int(waves[3])
                    and waves[3] >= 1
                    and waves[5] >= waves[2]
                    and _is_int(waves[7])
                    and waves[7] >= 1
                    and waves[8] >= waves[1]
                    and _is_int(waves[13])
                    and waves[13] >= 1
                )
            )
        ):
            raise ValueError(
                f"policy['wide_wave_rule'] must be [min narrow / wide wave ratio, min l6 / wide wave ratio, min mean unit tiles] or that with [the top_k (an int >= 1) up to which the tile minima hold, their exponent in top_k above it, the minimum of a packed call (>= the one-segment minimum), the tiles the one-segment minimum rises per wide wave below the next element, the wide waves (an int >= 1) from which it holds, the l6 / wide wave ratio W (>= element 1) below which the minimum is at least the next element / (W - 1) and on a packed call at least element 10 / (W - 1) ** 2, and from which the minimum of a one-document call of fewer wide waves than element 7 is at most element 11 / (W - 1) ** 2, on two or more wide waves at most the larger of that and element 12, from element 13 (an int >= 1) wide waves on of that and element 14], each >= 0, or None, got {waves!r}"
            )
        desc = raw["desc_walk_kinds"]
        if desc is not None and (
            len(desc) != 2
            or not all(
                isinstance(kinds, (list, tuple))
                and all(isinstance(k, str) and k in PHYSICAL_KINDS for k in kinds)
                and list(kinds) == sorted(set(kinds))
                for kinds in desc
            )
        ):
            raise ValueError(
                f"policy['desc_walk_kinds'] must be [unsampled kinds, sampled kinds] (ascending lists of physical scan kinds {PHYSICAL_KINDS}) or None, got {desc!r}"
            )
        pair_units = raw["pair_units"]
        if (
            not isinstance(pair_units, (list, tuple))
            or not all(isinstance(u, str) and u in UNITS for u in pair_units)
            or list(pair_units) != sorted(set(pair_units))
        ):
            raise ValueError(
                f"policy['pair_units'] must be an ascending list of unit geometries {UNITS}, got {pair_units!r}"
            )
        if raw["pair_rule"] is not None and not pair_units:
            raise ValueError(
                "policy['pair_units'] must name at least one unit geometry when policy['pair_rule'] is set"
            )
        split_units = raw["split_units"]
        if (
            not isinstance(split_units, (list, tuple))
            or not all(isinstance(u, str) and u in UNITS for u in split_units)
            or list(split_units) != sorted(set(split_units))
        ):
            raise ValueError(
                f"policy['split_units'] must be an ascending list of unit geometries {UNITS}, got {split_units!r}"
            )
        scope = raw["split_wave_scope"]
        if scope is not None and (
            len(scope) != 3
            or not isinstance(scope[0], (list, tuple))
            or not scope[0]
            or not all(isinstance(u, str) and u in UNITS for u in scope[0])
            or list(scope[0]) != sorted(set(scope[0]))
            or not _is_int(scope[1])
            or scope[1] < 1
            or not isinstance(scope[2], bool)
        ):
            raise ValueError(
                f"policy['split_wave_scope'] must be [ascending unit geometries {UNITS}, max segments >= 1, sampled only (bool)] or None, got {scope!r}"
            )
        fit_rule = raw["split_fit_rule"]
        if fit_rule is not None and (
            len(fit_rule) != 1 or not _is_int(fit_rule[0]) or fit_rule[0] < 1
        ):
            raise ValueError(
                f"policy['split_fit_rule'] must be [merge chunk entries >= 1] or None, got {fit_rule!r}"
            )
        resident = raw["split_resident_rule"]
        if resident is not None and (
            len(resident) != 5
            or not all(_is_int(v) and v >= 1 for v in resident)
            or not 2 <= resident[4] <= resident[3]
        ):
            raise ValueError(
                f"policy['split_resident_rule'] must be [merge chunk entries, fit entries, merge-resident entries, max splits, "
                f"min splits] (positive integers, 2 <= min splits <= max splits) or None, got {resident!r}"
            )
        solo = raw["l6_solo_rule"]
        if solo is not None and (
            len(solo) != 2
            or not _is_number(solo[0])
            or solo[0] < 0
            or not _is_int(solo[1])
            or solo[1] < 1
        ):
            raise ValueError(
                f"policy['l6_solo_rule'] must be [max mean unit tiles >= 0, min segments >= 1] or None, got {solo!r}"
            )
        solo_queries = raw["l6_solo_unsampled_max_queries"]
        if solo_queries is not None and solo_queries < 1:
            raise ValueError(
                f"policy['l6_solo_unsampled_max_queries'] must be an integer >= 1 or None, got {solo_queries!r}"
            )
        snake_cap = raw["snake_cap_rule"]
        if snake_cap is not None and (
            len(snake_cap) != 2 or not all(_is_int(v) and v >= 1 for v in snake_cap)
        ):
            raise ValueError(
                f"policy['snake_cap_rule'] must be [max segments >= 1, max waves >= 1] or None, got {snake_cap!r}"
            )
        lsf_single = raw["lsf_single_rule"]
        if lsf_single is not None and (
            len(lsf_single) != 1
            or not (_is_number(lsf_single[0]) and lsf_single[0] >= 0)
        ):
            raise ValueError(
                f"policy['lsf_single_rule'] must be [min mean unit tiles >= 0] or None, got {lsf_single!r}"
            )
        packed = raw["snake_packed_rule"]
        if packed is not None and (
            len(packed) != 4
            or not isinstance(packed[0], (list, tuple))
            or not packed[0]
            or not all(
                isinstance(k, str) and k in KINDS and k != "split" for k in packed[0]
            )
            or list(packed[0]) != sorted(set(packed[0]))
            or not _is_number(packed[1])
            or packed[1] < 0
            or not (packed[2] is None or (_is_number(packed[2]) and packed[2] >= 0))
            or not (packed[3] is None or (_is_int(packed[3]) and packed[3] >= 1))
        ):
            raise ValueError(
                f"policy['snake_packed_rule'] must be [ascending program kinds {KINDS} but split, min mean unit tiles sampled >= 0, "
                f"min mean unit tiles with sampling off >= 0 or None, max waves >= 1 or None] or None, got {packed!r}"
            )
        fallback = raw["snake_l6_fallback_waves"]
        if fallback is not None and fallback < 1:
            raise ValueError(
                f"policy['snake_l6_fallback_waves'] must be an integer >= 1 or None, got {fallback!r}"
            )
        shift_lo, shift_hi = SAMPLE_SHIFT_PERMILLE_BOUNDS
        shift_rule = raw["sample_shift_rule"]
        if shift_rule is not None and (
            len(shift_rule) != 2
            or not _is_int(shift_rule[0])
            or shift_rule[0] < 1
            or not _is_int(shift_rule[1])
            or not shift_lo <= shift_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_shift_rule'] must be [min top_k >= 1, shift permille in [{shift_lo}, {shift_hi}]] or None, got {shift_rule!r}"
            )
        class_rule = raw["sample_class_rule"]
        if class_rule is not None and (
            len(class_rule) != 2
            or not _is_int(class_rule[0])
            or class_rule[0] < 1
            or not _is_int(class_rule[1])
            or not shift_lo <= class_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_class_rule'] must be [tiny-unit sample tiles >= 1, shift permille in [{shift_lo}, {shift_hi}]] or None, got {class_rule!r}"
            )
        long_rule = raw["sample_long_rule"]
        if long_rule is not None and (
            len(long_rule) != 2
            or not isinstance(long_rule[0], (list, tuple))
            or not long_rule[0]
            or list(long_rule[0]) != sorted(set(long_rule[0]))
            or not set(long_rule[0]) <= set(PHYSICAL_KINDS)
            or not _is_int(long_rule[1])
            or not shift_lo <= long_rule[1] <= shift_hi
        ):
            raise ValueError(
                f"policy['sample_long_rule'] must be [ascending physical scan kinds {PHYSICAL_KINDS}, shift permille in "
                f"[{shift_lo}, {shift_hi}]] or None, got {long_rule!r}"
            )
        floor_rule = raw["cand_cap_floor_rule"]
        if floor_rule is not None and (
            len(floor_rule) != 2 or not all(_is_int(v) and v >= 1 for v in floor_rule)
        ):
            raise ValueError(
                f"policy['cand_cap_floor_rule'] must be [min top_k >= 1, floor slots >= 1] or None, got {floor_rule!r}"
            )
        if not isinstance(raw["check_period_kind_overrides"], dict):
            raise ValueError("policy['check_period_kind_overrides'] must be a mapping")
        if not isinstance(raw["finalize_threads_fit"], (list, tuple)) or not all(
            _is_int(v) for v in raw["finalize_threads_fit"]
        ):
            raise ValueError(
                "policy['finalize_threads_fit'] must be a list of thread counts"
            )
        variants = raw["rank_window_variants"]
        if (
            not isinstance(variants, (list, tuple))
            or not variants
            or not all(_is_int(v) and v > 0 for v in variants)
            or list(variants) != sorted(set(variants))
        ):
            raise ValueError(
                "policy['rank_window_variants'] must be an ascending list of positive bitmap windows"
            )
        if raw["rank_rule"] is not None and (
            len(raw["rank_rule"]) != 2
            or not all(_is_int(v) and v > 0 for v in raw["rank_rule"])
        ):
            raise ValueError(
                f"policy['rank_rule'] must be [max pool window, max mean segment keys] or None, got {raw['rank_rule']!r}"
            )
        staged = raw["rank_staged_rule"]
        if staged is not None and (
            len(staged) != 1 or not _is_int(staged[0]) or staged[0] <= 0
        ):
            raise ValueError(
                f"policy['rank_staged_rule'] must be [max pool window of the staged rank finalize form] or None, got {staged!r}"
            )
        seg_only = raw["rank_window_variants_seg_only"]
        if (
            not isinstance(seg_only, (list, tuple))
            or not all(_is_int(v) and v in variants for v in seg_only)
            or list(seg_only) != sorted(set(seg_only))
        ):
            raise ValueError(
                "policy['rank_window_variants_seg_only'] must be an ascending list of rank_window_variants entries"
            )
        two_variants = raw["rank_two_level_window_variants"]
        if (
            not isinstance(two_variants, (list, tuple))
            or not two_variants
            or not all(_is_int(v) and v > 0 for v in two_variants)
            or list(two_variants) != sorted(set(two_variants))
        ):
            raise ValueError(
                "policy['rank_two_level_window_variants'] must be an ascending list of positive bitmap windows"
            )
        if any(v <= raw["rank_slab_window_max"] for v in two_variants):
            raise ValueError(
                "policy['rank_two_level_window_variants'] must lie above policy['rank_slab_window_max']"
            )
        for name in ("rank_two_level_rule", "rank_two_level_staged_rule"):
            rule = raw[name]
            if rule is not None and (
                len(rule) != 1 or not _is_int(rule[0]) or rule[0] <= 0
            ):
                raise ValueError(
                    f"policy[{name!r}] must be [max two-level pool window] or None, got {rule!r}"
                )
        slots = raw["rank_t16_slots"]
        if (
            not isinstance(slots, (list, tuple))
            or not all(_is_int(v) and v > 0 and v % 16 == 0 for v in slots)
            or list(slots) != sorted(set(slots))
        ):
            raise ValueError(
                "policy['rank_t16_slots'] must be an ascending list of positive sort-slot counts divisible by 16"
            )
        for floor_name, ceiling_name, rule_name in (
            ("cand_cap_floor", "cand_cap_ceiling", "cand_cap_floor_rule"),
            ("cand_cap_floor_unsampled", "cand_cap_ceiling_unsampled", None),
        ):
            ceiling = raw[ceiling_name]
            floors = [raw[floor_name]]
            if rule_name is not None and raw[rule_name] is not None:
                floors.append(raw[rule_name][1])
            if ceiling is not None and (
                ceiling <= 0 or any(f is not None and ceiling < f for f in floors)
            ):
                raise ValueError(
                    f"policy[{ceiling_name!r}] must be a positive slot count at or above policy[{floor_name!r}]"
                    f"{' and the floor of policy[' + repr(rule_name) + ']' if rule_name else ''}, got {ceiling!r}"
                )
        static_kinds = raw["static_scan_kinds"]
        if (
            not isinstance(static_kinds, (list, tuple))
            or not all(isinstance(k, str) and k in PHYSICAL_KINDS for k in static_kinds)
            or list(static_kinds) != sorted(set(static_kinds))
        ):
            raise ValueError(
                f"policy['static_scan_kinds'] must be an ascending list of physical scan kinds {PHYSICAL_KINDS}, got {static_kinds!r}"
            )
        fit_kinds = raw["sample_fit_kinds"]
        if (
            not isinstance(fit_kinds, (list, tuple))
            or not all(isinstance(k, str) and k in PHYSICAL_KINDS for k in fit_kinds)
            or list(fit_kinds) != sorted(set(fit_kinds))
        ):
            raise ValueError(
                f"policy['sample_fit_kinds'] must be an ascending list of physical scan kinds {PHYSICAL_KINDS}, got {fit_kinds!r}"
            )
        if raw["rank_bulk_align_bytes"] % 4:
            raise ValueError(
                "policy['rank_bulk_align_bytes'] must be a multiple of the 4-byte output element"
            )
        if raw["rank_persist_max_k"] < 0:
            raise ValueError(
                "policy['rank_persist_max_k'] must be >= 0 (0 = no persistent rank finalize)"
            )
        if raw["rank_persist_ctas_per_sm"] <= 0:
            raise ValueError("policy['rank_persist_ctas_per_sm'] must be positive")
        if raw["rank_persist_spare_sms"] < 0:
            raise ValueError("policy['rank_persist_spare_sms'] must be >= 0")
        for name in ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes",
                     "candidate_multiplier", "candidate_multiplier_unsampled", "candidate_slack", "split_max",
                     "split_min_range_tiles",
                     "tile_unroll_default", "tile_unroll_factor", "sample_tiles_short_units", "sample_tiles_tiny_units",
                     "check_period_max", "check_period_knob_max", "check_period_cap_divisor", "finalize_items",
                     "finalize_threads_small", "finalize_top_k_small", "finalize_threads", "rank_top_k_min",
                     "rank_slab_window_max", "rank_bulk_align_bytes"):  # fmt: skip
            if raw[name] <= 0:
                raise ValueError(
                    f"policy[{name!r}] must be positive, got {raw[name]!r}"
                )
        if raw["sample_tiles_max"] < 0:
            raise ValueError("policy['sample_tiles_max'] must be >= 0")
        if raw["far_sample"] not in (0, 1):
            raise ValueError("policy['far_sample'] must be 0 or 1")
        values = dict(raw)
        for name in _OPTIONAL_RULE:
            if values[name] is not None:
                values[name] = [
                    list(v) if isinstance(v, (list, tuple)) else v for v in values[name]
                ]
        values["wide_rule"] = list(values["wide_rule"])
        values["finalize_threads_fit"] = [
            int(v) for v in values["finalize_threads_fit"]
        ]
        values["rank_window_variants"] = [
            int(v) for v in values["rank_window_variants"]
        ]
        values["rank_window_variants_seg_only"] = [
            int(v) for v in values["rank_window_variants_seg_only"]
        ]
        values["rank_two_level_window_variants"] = [
            int(v) for v in values["rank_two_level_window_variants"]
        ]
        values["rank_t16_slots"] = [int(v) for v in values["rank_t16_slots"]]
        values["sample_fit_kinds"] = [str(k) for k in values["sample_fit_kinds"]]
        values["static_scan_kinds"] = [str(k) for k in values["static_scan_kinds"]]
        values["pair_units"] = [str(u) for u in values["pair_units"]]
        values["split_units"] = [str(u) for u in values["split_units"]]
        values["check_period_kind_overrides"] = {
            str(k): int(v) for k, v in values["check_period_kind_overrides"].items()
        }
        return cls(**values)

    def record(self) -> dict:
        """The record in canonical field order."""
        values = asdict(self)
        return {name: values[name] for name in POLICY_FIELDS}

    # ------------------------------------------------------------------ geometry estimates

    def mean_unit_tiles(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> float:
        """Estimated mean key tiles per unit, ``(Tkv - T / (2 ratio)) / S / tile_keys`` (a causal segment's
        units see on average half of its query span less than its key count)."""
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = (int(num_keys) - int(num_queries) / (2 * r)) / segments
        return max(0.0, mean_keys) / self.tile_keys

    def _mean_unit_tiles_floor(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> int:
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = (int(num_keys) - int(num_queries) / (2 * r)) / segments
        return int(max(0.0, mean_keys) // self.tile_keys)

    def offset_tiles(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> float:
        """Estimated mean segment offset in key tiles, ``(Tkv - T / ratio) / S / tile_keys``."""
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        return (
            max(0.0, (int(num_keys) - int(num_queries) / r) / segments) / self.tile_keys
        )

    # ------------------------------------------------------------------ runtime scalars

    def sample_tiles_for(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> int:
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        if mean_tiles <= self.sample_dispatch_tiny_tiles_max:
            return self.sample_tiles_tiny_units
        return (
            self.sample_tiles_short_units
            if mean_tiles <= self.sample_dispatch_mean_tiles_max
            else self.sample_tiles_max
        )

    def sample_shift_for(self, top_k: int) -> int:
        """Conservative shift (permille) of the sampled rank for a call: ``sample_shift_rule[1]`` from the rule's
        ``top_k`` floor on, else ``sample_shift_permille``.  Every value gives the same bits -- the shift only sets
        how conservative the sampled first threshold is (a threshold that proves too aggressive is streamed again
        exactly); an unsampled call (``sample_tiles_max=0``) never reads it."""
        rule = self.sample_shift_rule
        if rule is not None and int(top_k) >= int(rule[0]):
            return int(rule[1])
        return self.sample_shift_permille

    def cand_cap_floor_for(self, top_k: int) -> Optional[int]:
        """Candidate-slot floor of a sampled call: ``cand_cap_floor_rule[1]`` from the rule's ``top_k`` floor on, else
        ``cand_cap_floor``.  The larger floor pays for itself on large-``top_k`` calls only (their units fill the buffer
        early; the floor spares compactions and keeps 64-127-tile units off the sampled path); below the rule's ``top_k``
        the smaller floor halves the candidate buffers -- 57 MiB instead of 114 MiB for a ``top_k = 512`` call (the
        ratio-4 pooled selection) on a 152-SM grid with six queries per unit -- with identical ids and score bits.
        Unsampled calls (``sample_tiles_max=0``) read ``cand_cap_floor_unsampled``.  The shipped ``sm_100a`` record sets
        the rule; ``sm_103a`` (whose pooled rows measured slower with the smaller buffers) and ``sm_107a`` set none."""
        rule = self.cand_cap_floor_rule
        if rule is not None and int(top_k) >= int(rule[0]):
            return int(rule[1])
        return self.cand_cap_floor

    def cand_cap_multiplier_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        top_k: int,
        sampled: bool = True,
    ) -> int:
        """Capacity multiplier of a call: the sampled values by default, the ``*_unsampled`` values
        when the sampled first threshold is off (``sample_tiles_max=0``), whose larger buffers would
        only delay the first selection.  The long-unit rule replaces the multiplier, the floor
        (``cand_cap_floor_for``: keyed by ``top_k`` where the record sets ``cand_cap_floor_rule``) raises it
        and the ceiling lowers it (``cand_cap_ceiling // top_k``, so the buffer stops growing with
        ``top_k`` once it holds the ceiling)."""
        if sampled:
            mult, rule, floor, ceiling = (
                self.candidate_multiplier,
                self.cand_mult_rule,
                self.cand_cap_floor_for(top_k),
                self.cand_cap_ceiling,
            )
        else:
            mult, rule, floor, ceiling = (
                self.candidate_multiplier_unsampled,
                self.cand_mult_rule_unsampled,
                self.cand_cap_floor_unsampled,
                self.cand_cap_ceiling_unsampled,
            )
        if rule is not None:
            if (
                self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
                >= rule[0]
            ):
                mult = int(rule[1])
        if floor is not None and int(top_k) * mult < int(floor):
            mult = -(-int(floor) // int(top_k))
        if ceiling is not None and int(top_k) * mult > int(ceiling):
            mult = max(1, int(ceiling) // int(top_k))
        return mult

    def candidate_capacity(self, top_k: int, multiplier: Optional[int] = None) -> int:
        """Per-(warpgroup, query) candidate slots: ``max(multiplier * top_k, top_k + slack)`` rounded up to whole tiles."""
        mult = self.candidate_multiplier if multiplier is None else int(multiplier)
        if mult < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        cap = max(int(top_k) * mult, int(top_k) + self.candidate_slack)
        return (cap + self.tile_keys - 1) // self.tile_keys * self.tile_keys

    def check_period_for(
        self, top_k: int, cand_cap: int, kind: Optional[str] = None
    ) -> int:
        tile = self.tile_keys
        period = max(
            1,
            min(
                self.check_period_max,
                (int(cand_cap) - int(top_k)) // tile,
                int(cand_cap) // self.check_period_cap_divisor,
            ),
        )
        override = (
            self.check_period_kind_overrides.get(kind) if kind is not None else None
        )
        if override is not None:
            period = max(
                1,
                min(
                    int(override),
                    self.check_period_knob_max,
                    (int(cand_cap) - int(top_k)) // tile,
                    int(cand_cap) // (2 * tile),
                ),
            )
        return period

    def check_period_limit(self, top_k: int, cand_cap: int) -> int:
        """Largest explicit check period that is still exact for the capacity (the kernel's headroom rule)."""
        tile = self.tile_keys
        return max(
            1,
            min(
                self.check_period_knob_max,
                (int(cand_cap) - int(top_k)) // tile,
                int(cand_cap) // (2 * tile),
            ),
        )

    # ------------------------------------------------------------------ program selection

    def unit_name(self, block_q: int) -> str:
        if int(block_q) == self.block_q_l6:
            return "l6"
        if int(block_q) == self.block_q_wide:
            return "wide"
        if int(block_q) == self.block_q_narrow:
            return "narrow"
        raise ValueError(
            f"block_q must be {self.block_q_narrow}, {self.block_q_wide} or {self.block_q_l6}, got {block_q!r}"
        )

    def program_kind(self, block_q: int, split: bool, pair: bool) -> str:
        if split:
            return "split"
        base = self.unit_name(block_q)
        return f"pair_{base}" if pair else base

    def unit_block_q_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        top_k: int,
    ) -> int:
        units_wide = -(-int(num_queries) // self.block_q_wide)
        rounds = units_wide / max(1, int(grid))
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        l6 = self.l6_rule
        three_wg = l6 is not None and mean_tiles <= l6[0] and int(top_k) >= int(l6[1])
        waves = self.wide_wave_rule
        if waves is not None:
            # the unit the other rules leave (three-warpgroup or narrow) against the wide unit, in waves of the
            # persistent grid: the wide unit when the other unit needs at least the record's ratio of the wide unit's
            # waves (against the three-warpgroup unit also from a mean unit tile count on).  A fifteen-element rule
            # raises that count by waves[6] tiles per wide wave below waves[7] (a short call's last waves cost the wide
            # unit more) but on one document to at most waves[11] / (W - 1) ** 2, from two wide waves on to at most the
            # larger of that and waves[12] (from waves[13] wide waves on, of that and waves[14]: there the wide unit
            # measured slower up to more tiles), while the wave ratio W of the three-warpgroup unit to the wide unit is
            # at least waves[8] (the more waves the wide unit saves, the shorter the units that pay for its last wave;
            # one wide wave has no last wave to pay for), to at least waves[5] on a packed call (two or more segments)
            # and to at least waves[9] / (W - 1) while W stays below waves[8] (the fewer waves the wide unit saves, the
            # more tiles its longer units need to pay for them), on a packed call then also to at least waves[10] /
            # (W - 1) ** 2, and multiplies it by (top_k / waves[3]) ** waves[4] above top_k = waves[3] (each unit
            # selects more candidates)
            other = self.block_q_l6 if three_wg else self.block_q_narrow
            ctas = max(1, int(grid))
            waves_wide = -(-units_wide // ctas)
            units_other = -(-int(num_queries) // other)
            waves_other = -(-units_other // ctas)
            if three_wg:
                floor = waves[2]
                if len(waves) > 3:
                    floor += waves[6] * max(0, int(waves[7]) - waves_wide)
                    if (
                        int(num_segments) < 2
                        and waves_wide < int(waves[7])
                        and waves_other > waves_wide
                        and waves_other >= waves[8] * waves_wide
                    ):
                        # 1 / (W - 1)
                        inv = waves_wide / (waves_other - waves_wide)
                        cap = waves[11] * inv * inv
                        if waves_wide > 1:
                            cap = max(
                                cap,
                                waves[14]
                                if waves_wide >= int(waves[13])
                                else waves[12],
                            )
                        floor = min(floor, cap)
                    if int(num_segments) >= 2:
                        floor = max(floor, waves[5])
                    if waves_other < waves[8] * waves_wide:
                        if waves_other > waves_wide:
                            # 1 / (W - 1)
                            inv = waves_wide / (waves_other - waves_wide)
                            floor = max(floor, waves[9] * inv)
                            if int(num_segments) >= 2:
                                floor = max(floor, waves[10] * inv * inv)
                        else:
                            floor = float("inf")
                    floor *= max(1.0, int(top_k) / waves[3]) ** waves[4]
                wide = waves_other >= waves[1] * waves_wide and mean_tiles >= floor
            else:
                wide = waves_other >= waves[0] * waves_wide
            return self.block_q_wide if wide else other
        min_rounds, min_work = self.wide_rule
        wide = rounds >= min_rounds and rounds * mean_tiles >= min_work
        first = self.wide_first_rule
        if wide and first is not None and mean_tiles >= first[0]:
            return self.block_q_wide
        if three_wg:
            return self.block_q_l6
        return self.block_q_wide if wide else self.block_q_narrow

    def n_split_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        block_q: int,
        sampled: bool = True,
        top_k: Optional[int] = None,
    ) -> int:
        """Key-range splits per unit of the unit ``block_q``: the split rule (``split_rule_for``) for a unit of
        ``split_units``; 1 for any other unit, whose split program the architecture does not register."""
        if self.unit_name(block_q) not in self.split_units:
            return 1
        return self.split_rule_for(
            num_queries, num_keys, num_segments, ratio, grid, block_q, sampled, top_k
        )

    def split_rule_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        block_q: int,
        sampled: bool = True,
        top_k: Optional[int] = None,
    ) -> int:
        """The split rule alone (no registration guard): ``min(grid // units, mean tiles // split_min_range_tiles,
        split_max)``, at least 1, and the two-way wave split of ``split_wave_rule`` where that gives 1 -- inside
        ``split_wave_scope`` only (its unit geometries, calls of at most its segment count, and with its third item the
        sampled calls only: ``sampled`` says whether the call's sampled first threshold is on).

        With ``top_k`` given, a call that rule splits takes the count of ``split_resident_rule`` [chunk, fit, resident,
        max, min] where the record sets it: ``min(grid // units, resident // top_k, max, max(min, n_fit if n_fit x top_k
        <= fit else max))``, at least 2, with ``n_fit = ceil(mean unit keys / top_k)`` rounded up to whole merge chunks
        of ``chunk`` entries.  The merge keeps its ``n x top_k`` staged entries in shared memory up to ``resident``
        entries, so the count stays there; below it more splits shorten each split's scan.  Otherwise
        ``split_fit_rule``."""
        units = max(1, -(-int(num_queries) // int(block_q)))
        mean_tiles = self._mean_unit_tiles_floor(
            num_queries, num_keys, num_segments, ratio
        )
        n = max(
            1,
            min(
                int(grid) // units,
                mean_tiles // self.split_min_range_tiles,
                self.split_max,
            ),
        )
        rule = self.split_wave_rule
        scope = self.split_wave_scope
        if scope is not None and (
            self.unit_name(block_q) not in scope[0]
            or int(num_segments) > int(scope[1])
            or (scope[2] and not sampled)
        ):
            rule = None
        if (
            n == 1
            and rule is not None
            and mean_tiles >= int(rule[0])
            and mean_tiles >= 2 * self.split_min_range_tiles
        ):

            def eff(
                k: int,
            ) -> (
                float
            ):  # share of the CTAs busy over the call's rounds: waves / ceil(waves)
                return (units * k) / (int(grid) * (-(-(units * k) // int(grid))))

            if eff(2) - eff(1) >= float(rule[1]):
                n = 2
        if n == 1 or top_k is None or int(top_k) <= 0:
            return n
        k = int(top_k)
        segments = max(1, int(num_segments))
        r = max(1, int(ratio))
        mean_keys = max(0.0, (int(num_keys) - int(num_queries) / (2 * r)) / segments)

        def fit_splits(chunk: int) -> int:
            # splits that hold at most top_k keys each, rounded up to whole merge chunks of ``chunk`` entries (more
            # splits in the last chunk cost the merge nothing)
            per_chunk = max(1, int(chunk) // k)
            return -(-max(1, -(-int(mean_keys) // k)) // per_chunk) * per_chunk

        resident = self.split_resident_rule
        if resident is not None:
            chunk, fit_entries, resident_entries, most, least = (
                int(v) for v in resident
            )
            n_fit = fit_splits(chunk)
            want = n_fit if n_fit * k <= fit_entries else most
            n = min(int(grid) // units, resident_entries // k, most, max(least, want))
            return max(2, n)
        fit = self.split_fit_rule
        if fit is not None:
            # split_fit_rule: a split call whose splits can each hold at most top_k keys needs no selection inside a
            # split; the merge reads its staging in chunks of fit[0] entries
            n_fit = fit_splits(int(fit[0]))
            if n < n_fit <= min(int(grid) // units, self.split_max):
                n = n_fit
        return n

    def pair_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        top_k: int,
        block_q: int,
    ) -> bool:
        """The two-CTA multicast program of the chosen unit ``block_q``: the pair serves the unit geometries of
        ``pair_units`` only (the units whose pair program the architecture registers), inside the ``pair_rule``
        mean-tile interval and from its ``top_k`` floor, on a grid of at least two CTAs."""
        rule = self.pair_rule
        if rule is None or int(grid) < 2:
            return False
        if self.unit_name(block_q) not in self.pair_units:
            return False
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        return rule[0] <= mean_tiles <= rule[1] and int(top_k) >= int(rule[2])

    def l6_solo_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        sampled: bool,
        pair_l6: bool = False,
    ) -> bool:
        """The preference for the three-warpgroup unit without a CTA pair.  A call with the sampled first threshold off
        (``sampled`` False): where the record sets ``l6_solo_unsampled_max_queries``, only a call planned on the CTA-pair
        l6 unit (``pair_l6``) of at most that many queries; else every such call where the record sets ``l6_solo_rule``.
        A sampled call: at least ``l6_solo_rule[1]`` segments or at most ``l6_solo_rule[0]`` mean unit tiles.  Never
        without either field.  ``select_program`` applies it only to a call whose plan does not split, whose l6 unit would
        not split either, and where the architecture registers the program it picks."""
        rule = self.l6_solo_rule
        if not sampled:
            cap = self.l6_solo_unsampled_max_queries
            if cap is not None:
                return pair_l6 and int(num_queries) <= int(cap)
            return rule is not None
        if rule is None:
            return False
        return int(num_segments) >= int(rule[1]) or self.mean_unit_tiles(
            num_queries, num_keys, num_segments, ratio
        ) <= float(rule[0])

    def snake_for(
        self,
        kind: str,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        sampled: bool = True,
        grid: Optional[int] = None,
    ) -> bool:
        """The snake unit order of a program kind: where ``snake_rule`` names the kind, from its mean unit tiles (the
        ratio-1 or the ratio > 1 floor) and unit-cost spread on.  On a call of two or more segments whose kind
        ``snake_packed_rule`` names, the order also needs the rule's mean unit tiles for the call's sampling mode
        (``sampled``: the sampled first threshold is on; with it off a None floor keeps the plain order), unless the call
        needs at most ``snake_packed_rule[3]`` waves of a persistent grid of ``grid`` CTAs (``ceil(ceil(T / block_q) /
        slots)``, slots = CTAs, or CTA pairs for a pair kind; never without ``grid``): the snake order balances the last
        waves of one causal document and of calls of few waves, and packed documents of few tiles over more waves run
        faster in the plain order."""
        rule = self.snake_rule
        if rule is None or kind not in rule[0]:
            return self.snake_default
        r = max(1, int(ratio))
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        spread = min(1.0, int(num_queries) / max(1.0, r * float(num_keys)))
        snake = mean_tiles >= (rule[1] if r == 1 else rule[2]) and spread >= rule[3]
        packed = self.snake_packed_rule
        if (
            snake
            and packed is not None
            and int(num_segments) >= 2
            and kind in packed[0]
        ):
            floor = packed[1] if sampled else packed[2]
            if floor is not None and mean_tiles >= float(floor):
                return True
            if packed[3] is None or grid is None:
                return False
            pair = kind.startswith("pair_")
            block_q = getattr(self, f"block_q_{kind[5:] if pair else kind}")
            ctas = max(1, int(grid))
            slots = max(1, (ctas - ctas % 2) // 2) if pair else ctas
            units = -(-int(num_queries) // int(block_q))
            return -(-units // slots) <= int(packed[3])
        return snake

    def snake_l6_fallback_for(
        self, num_queries: int, block_q: int, pair: bool, grid: int, sampled: bool
    ) -> bool:
        """Whether a sampling-off call (``sampled`` False) planned on the unit ``block_q`` (a CTA pair where ``pair``)
        over a persistent grid of ``grid`` CTAs needs at most ``snake_l6_fallback_waves`` waves of that grid,
        ``ceil(ceil(T / block_q) / slots)`` with ``slots`` the grid's CTAs (its CTA pairs for a pair program).  Such a
        call, which ``l6_solo_for`` moves to the unpaired l6 unit, runs that unit's plain-order program where its
        snake-order program is not registered (``select_program``); never where the record leaves the field None."""
        cap = self.snake_l6_fallback_waves
        if cap is None or sampled:
            return False
        ctas = max(1, int(grid))
        slots = max(1, (ctas - ctas % 2) // 2) if pair else ctas
        units = -(-int(num_queries) // max(1, int(block_q)))
        return -(-units // slots) <= int(cap)

    def snake_copy_fits(
        self, num_queries: int, num_segments: int, block_q: int, grid: int
    ) -> bool:
        """Whether a call of ``num_segments`` segments whose ``num_queries`` queries in units of ``block_q`` take a
        persistent grid of ``grid`` CTAs stays within ``snake_cap_rule`` (at most its segments and waves; always where
        the record leaves the rule None).  The snake unit order balances the last waves of one causal document; over
        packed documents, or over many waves, the plain order is faster (the FP8 sm_107a l6 program), so
        ``select_fp8_program`` runs a snake copy only within the rule."""
        rule = self.snake_cap_rule
        if rule is None:
            return True
        units = -(-int(num_queries) // max(1, int(block_q)))
        waves = -(-units // max(1, int(grid)))
        return int(num_segments) <= int(rule[0]) and waves <= int(rule[1])

    def lsf_single_fits(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> bool:
        """Whether a one-segment call reaches ``lsf_single_rule`` (at least its mean unit tiles; never where the record
        leaves the rule None).  On one segment the :lsf copy walks the units of the plain program in the same order and
        returns the same rows; on the FP8 sm_103a wide program it measures faster from 64 mean unit tiles on and slower
        below, so ``select_fp8_program`` runs it on a one-segment wide call only within the rule."""
        rule = self.lsf_single_rule
        return (
            rule is not None
            and int(num_segments) == 1
            and self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
            >= rule[0]
        )

    def tile_unroll_for(self, kind: str, top_k: int, snake: bool, block_q: int) -> int:
        rule = self.tile_unroll_rule
        if rule is None or (bool(rule[2]) and snake):
            return self.tile_unroll_default
        lookup = (
            "split_wide"
            if kind == "split" and int(block_q) == self.block_q_wide
            else kind
        )
        return (
            self.tile_unroll_factor
            if lookup in rule[0] and int(top_k) >= int(rule[1])
            else self.tile_unroll_default
        )

    def desc_walk_for(self, physical_kind: str, sampled: bool) -> bool:
        """The call runs the descending-walk twin of its scan program (key kind ``<physical_kind>_desc``, the same unit,
        split, pair, tile unroll and fit form): ``desc_walk_kinds`` names the physical kind for the call's sampling
        mode (``select_program`` takes the twin only where the registry has it)."""
        rule = self.desc_walk_kinds
        return rule is not None and physical_kind in rule[1 if sampled else 0]

    def sample_fit_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        cand_cap: int,
        physical_kind: Optional[str] = None,
    ) -> bool:
        """Sampled first threshold on buffer-fitting units: the mean unit is at most ``sample_fit_max_mean_tiles`` and
        the mean segment offset leaves units that fit the capacity.  A program kind outside ``sample_fit_kinds`` has no
        sampled-fit program registered and runs its plain program on every call (``physical_kind`` None: no kind gate)."""
        if physical_kind is not None and physical_kind not in self.sample_fit_kinds:
            return False
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        offset_tiles = self.offset_tiles(num_queries, num_keys, num_segments, ratio)
        fit_tiles = int(cand_cap) // self.tile_keys - 1
        return (
            mean_tiles <= self.sample_fit_max_mean_tiles
            and offset_tiles + 1 <= fit_tiles
        )

    def finalize_threads_for(self, top_k: int) -> int:
        k = int(top_k)
        if self.finalize_fit and k <= self.finalize_top_k_small // 2:
            slots = max(
                self.finalize_threads_fit[0] * self.finalize_items,
                1 << max(0, k - 1).bit_length(),
            )
            return slots // self.finalize_items
        return (
            self.finalize_threads_small
            if k <= self.finalize_top_k_small
            else self.finalize_threads
        )

    def finalize_threads_admissible(self, threads: int, top_k: int) -> bool:
        """Explicit finalize program: a right-sized program sorts ``threads x items`` slots, the small program sorts up
        to ``finalize_top_k_small``, the large program every ``top_k``."""
        t = int(threads)
        if t in self.finalize_threads_fit:
            return int(top_k) <= t * self.finalize_items
        if t == self.finalize_threads_small:
            return int(top_k) <= self.finalize_top_k_small
        return t == self.finalize_threads

    def finalize_key_bits_for(self, num_keys: int) -> int:
        n = int(num_keys)
        if self.finalize_exact_key_bits:
            return max(1, (n - 1).bit_length())
        return n.bit_length()

    def finalize_rank_for(
        self,
        top_k: int,
        num_keys: int,
        num_segments: int,
        threads: Optional[int] = None,
        *,
        max_seqlen_k: Optional[int] = None,
    ) -> Optional[int]:
        """Bitmap window of the prefix-popcount rank finalize program serving the call, or None (the CUB block radix sort).

        Mirrors the kernel's ``finalize_rank_for``.  The pool is ``Tkv``, or -- lever RW (``rank_seg_window``) with
        ``max_seqlen_k`` given -- the call's bound on every key segment (a bound above ``Tkv`` or below the mean segment
        ``Tkv / S`` raises whether or not the switch is on).  Lever FR2 (``rank_two_level`` with ``rank_two_level_rule``):
        a pool above ``rank_slab_window_max`` takes the smallest ``rank_two_level_window_variants`` entry holding it up to
        the two-level pool bound, with no segment bound, or None above it.  Otherwise ``rank_rule`` = [largest pool
        window, largest segment key count] bounds the pool and the segment (the mean ``Tkv / S``, or ``max_seqlen_k``
        itself under lever RW) and the variant is the smallest ``rank_window_variants`` entry holding the pool and one
        bitmap word (32 bits) per thread of the finalize program (``threads``: the explicit knob, else the thread count
        of ``top_k``), skipping the ``rank_window_variants_seg_only`` entries when the pool is ``Tkv``; ``top_k`` below
        ``rank_top_k_min`` and thread counts without a rank form keep CUB.
        """
        k, Tkv, S = int(top_k), int(num_keys), int(num_segments)
        seg = None
        if max_seqlen_k is not None:
            seg = int(max_seqlen_k)
            if seg > Tkv or seg * max(1, S) < Tkv:
                raise ValueError(
                    f"max_seqlen_k={seg} is not an upper bound of every key segment: {Tkv} keys in {S} segments"
                )
            if not self.rank_seg_window:
                seg = None
        rule = self.rank_rule
        if not self.rank_finalize or rule is None or k < self.rank_top_k_min:
            return None
        pool_max, mean_segment_max = int(rule[0]), int(rule[1])
        t = (
            (
                self.finalize_threads_small
                if k <= self.finalize_top_k_small
                else self.finalize_threads
            )
            if threads is None
            else int(threads)
        )
        if t not in (self.finalize_threads_small, self.finalize_threads) or (
            t == self.finalize_threads_small and k > self.finalize_top_k_small
        ):
            return None
        pool = Tkv if seg is None else seg
        need = max(pool, 32 * t)
        if (
            self.rank_two_level
            and self.rank_two_level_rule is not None
            and need > int(self.rank_slab_window_max)
        ):
            two_max = int(self.rank_two_level_rule[0])
            for variant in self.rank_two_level_window_variants:
                if int(variant) > two_max:
                    break
                if need <= int(variant):
                    return int(variant)
            return None
        if seg is None:
            if Tkv > mean_segment_max * max(1, S):
                return None
        elif seg > mean_segment_max:
            return None
        for variant in self.rank_window_variants:
            if int(variant) > pool_max:
                break
            if seg is None and int(variant) in self.rank_window_variants_seg_only:
                continue
            if need <= int(variant):
                return int(variant)
        return None

    def rank_two_level_form(self, rank_window: Optional[int]) -> bool:
        """True when the rank finalize program of bitmap window ``rank_window`` runs its two-level (bucket + word) form:
        ``rank_two_level`` on and the window above ``rank_slab_window_max`` (mirrors the kernel's ``rank_two_level_form``);
        False for the CUB call (None)."""
        return (
            rank_window is not None
            and bool(self.rank_two_level)
            and int(rank_window) > int(self.rank_slab_window_max)
        )

    def finalize_rank_staged_for(
        self, rank_window: Optional[int], threads: int, two_level: bool = False
    ) -> bool:
        """True when the rank finalize call of bitmap window ``rank_window`` at ``threads`` runs its SMEM-staged, coalesced-I/O
        form (lever FRS; program key ``finalize_rank:t<threads>:w<window>[:two]:staged``, same role and arguments).

        Mirrors the kernel's ``finalize_rank_staged_for``: ``rank_staged`` on and the window within ``rank_staged_rule`` = [largest
        pool window whose staged form keeps two CTAs per SM] -- within ``rank_two_level_staged_rule`` for a two-level window
        (lever FR2, ``two_level``); a call without a rank window (the CUB sort) is never staged.
        """
        if rank_window is None or not self.rank_staged:
            return False
        rule = self.rank_two_level_staged_rule if two_level else self.rank_staged_rule
        return rule is not None and int(rank_window) <= int(rule[0])

    def finalize_rank_t16_for(self, threads: int, staged: bool) -> bool:
        """True when the staged rank finalize call at ``threads`` (the 8-item thread count of ``finalize_threads_for``) runs its
        16-item thread form (lever FRT; program key tag ``:i16``): ``rank_t16`` on, a staged call and the call's sort-slot count
        ``threads x finalize_items`` in ``rank_t16_slots`` (mirrors the kernel's ``finalize_rank_t16_for``; the 4096-slot form runs
        256 threads x 16 slots for three rows in flight per SM instead of two).  Never on a direct-form or CUB call."""
        if not (bool(self.rank_t16) and bool(staged)):
            return False
        return int(threads) * int(self.finalize_items) in [
            int(v) for v in self.rank_t16_slots
        ]

    def rank_t16_form(self, threads: int) -> Tuple[int, int]:
        """The 16-item thread form of a staged rank finalize at ``threads`` 8-item threads: ``(threads / 2, 16)`` -- the same sort
        slots, half the threads (mirrors the kernel's ``rank_t16_form_for``)."""
        slots = int(threads) * int(self.finalize_items)
        if slots % 16:
            raise ValueError(
                f"the 16-item rank finalize form needs a sort-slot count divisible by 16, got {slots}"
            )
        return slots // 16, 16

    def finalize_rank_bulk_io_for(
        self, top_k: int, staged: bool, outputs_aligned: bool = True
    ) -> bool:
        """True when a staged rank finalize call moves its rows with ``cp.async.bulk`` (lever FRB; program key tag ``:bulk``):
        ``rank_bulk_io`` on, a staged call, ``top_k`` a multiple of ``rank_bulk_align_bytes / 4`` (every row base ``row x top_k x 4``
        and byte count ``top_k x 4`` are then multiples of the granule) and output tensors whose addresses are multiples of the
        granule (``outputs_aligned``: the backend's own allocations are, a caller-owned view may not be -- it then runs the plain
        twin program, same results).  Mirrors the kernel's ``finalize_rank_bulk_io_for``."""
        if not (bool(self.rank_bulk_io) and bool(staged)):
            return False
        if int(top_k) % (int(self.rank_bulk_align_bytes) // 4):
            return False
        return bool(outputs_aligned)

    def finalize_rank_persistent_for(
        self, top_k: int, staged: bool, two_level: bool, bulk: bool, t16: bool
    ) -> bool:
        """True when a staged rank finalize call with bulk row I/O runs the persistent two-row pipelined program (lever FRP-K; program
        key tag ``:persist``): ``rank_persist_max_k`` > 0 (the architecture's cap), a slab (not two-level) staged call with the bulk
        form and 8-item threads (``t16`` False: the 16-item thread form of lever FRT keeps the one-row program -- its doubled staging
        would halve the residency -- whether the call's top_k or the ``finalize_threads`` knob selected it) and ``top_k <=
        rank_persist_max_k``.  Mirrors the kernel's ``finalize_rank_persistent_for`` + ``rank_persist_max_k_for``; same rows, same
        bytes -- results are bitwise independent of the choice."""
        cap = int(self.rank_persist_max_k)
        if (
            cap <= 0
            or not (bool(staged) and bool(bulk))
            or bool(two_level)
            or bool(t16)
        ):
            return False
        return int(top_k) <= cap

    def merge_rank_for(
        self, split: bool, top_k: int, rank_window: Optional[int], two_level: bool
    ) -> bool:
        """True when a split call runs the merge with the rank finalize fused in (program key ``merge:rank:w<rank_window>``) and
        launches no finalize program: ``merge_rank`` on the architecture, a slab (not two-level) rank finalize window up to
        ``rank_slab_window_max`` and ``top_k <= MERGE_RANK_MAX_TOP_K``.  The fused program writes the slab rank finalize's rows
        (same kept set, same ascending-ID order, same padding), so results are bitwise independent of the choice."""
        return (
            bool(self.merge_rank)
            and bool(split)
            and rank_window is not None
            and not bool(two_level)
            and int(rank_window) <= int(self.rank_slab_window_max)
            and int(top_k) <= MERGE_RANK_MAX_TOP_K
        )


def merge_key(rank_window: Optional[int] = None) -> str:
    """Program key of a split call's merge: ``merge``, or ``merge:rank:w<rank_window>`` for the merge with the rank finalize
    fused in."""
    return MERGE_KEY if rank_window is None else f"{MERGE_KEY}:rank:w{int(rank_window)}"


def finalize_grid(choice: "ProgramChoice", num_sms: int) -> int:
    """CTAs of a call's finalize launch: ``min(num_queries, rank_persist_ctas x (SM count - rank_persist_spare))`` for the
    persistent rank program (lever FRP-K; every CTA walks the rows ``bid, bid + grid, ...``; at least one SM), one CTA per row
    otherwise (at least one CTA).  The grid is a multiple of the SMs that remain after the spare ones: where the CTAs one
    SM holds at a time divide ``rank_persist_ctas`` (sm_107a: 4 of the 256-thread program on a 1024-thread SM, 8 per
    SM), a block of another stream on up to ``rank_persist_spare`` SMs leaves the static row walk whole waves of the
    other SMs instead of a short extra wave (#7548)."""
    rows = max(1, int(choice.num_queries))
    if choice.rank_persistent:
        sms = max(1, int(num_sms) - int(choice.rank_persist_spare))
        return max(1, min(rows, int(choice.rank_persist_ctas) * sms))
    return rows


def physical_kind(kind: str, unit: str) -> str:
    """Name of the physical scan program kind: the split program is one per unit geometry."""
    return f"split_{unit}" if kind == "split" else kind


def scan_key(
    physical: str,
    tile_unroll: int,
    snake: bool,
    sample_fit: bool,
    lsf: bool = False,
    form: str = BF16,
) -> str:
    """``:lsf``: the longest-first (LSF) copy, which hands out the program's tickets longest segment first; ``form``
    ``"fp8"``: the FP8 program, tagged ``:fp8`` before the ``:lsf`` suffix."""
    key = f"scan:{physical}:u{int(tile_unroll)}:s{int(bool(snake))}:f{int(bool(sample_fit))}"
    if form == FP8:
        key += ":fp8"
    elif form != BF16:
        raise ValueError(f"form must be one of {FORMS}, got {form!r}")
    return key + ":lsf" if lsf else key


class ScanKeyFields(NamedTuple):
    """The fields of a scan program key (``scan_key``)."""

    physical: str
    tile_unroll: int
    snake: bool
    sample_fit: bool
    form: str
    lsf: bool


_SCAN_KEY = re.compile(r"scan:([a-z0-9_]+):u([0-9]+):s([01]):f([01])(:fp8)?(:lsf)?")


def scan_key_fields(key: str) -> Optional[ScanKeyFields]:
    """The fields of the scan program key ``key``; None for any other key."""
    match = _SCAN_KEY.fullmatch(key)
    if match is None:
        return None
    physical, unroll, snake, fit, fp8, lsf = match.groups()
    return ScanKeyFields(
        physical, int(unroll), snake == "1", fit == "1", FP8 if fp8 else BF16, bool(lsf)
    )


def scan_key_form(key: str) -> Optional[str]:
    """The input form (``FORMS``) of the scan program key ``key``; None for any other key."""
    fields = scan_key_fields(key)
    return None if fields is None else fields.form


def fp8_policy_record(record: dict, fp8_record: dict) -> dict:
    """The FP8 form's dispatch record of an architecture: its record ``record`` (``POLICY[arch]``) with the fields of
    its FP8 record ``fp8_record`` (``POLICY_FP8[arch]``) replaced.  ``fp8_record`` names its own ``static_scan_kinds``
    (the FP8 programs never keep the BF16 programs' static kinds) and no field outside ``POLICY_FIELDS``."""
    if (
        not isinstance(fp8_record, dict)
        or "static_scan_kinds" not in fp8_record
        or not set(fp8_record) <= set(POLICY_FIELDS)
    ):
        raise ValueError(
            f"an FP8 dispatch record must name static_scan_kinds and only fields of {POLICY_FIELDS}, got {fp8_record!r}"
        )
    return {**record, **fp8_record}


def fp8_scan_programs(scan_keys: Collection[str]) -> dict:
    """The plain FP8 scan programs of a registry's keys: ``{(physical kind, lsf): {tile unroll: key}}`` over the FP8
    keys in the plain unit order and the sampled-fit form (``s0`` / ``f1``).  A snake copy (``s1``) is not indexed
    here: ``select_fp8_program`` swaps it in for its plain program where the decision marks the call snake."""
    programs: dict = {}
    for key in scan_keys:
        fields = scan_key_fields(key)
        if (
            fields is None
            or fields.form != FP8
            or fields.snake
            or not fields.sample_fit
        ):
            continue
        programs.setdefault((fields.physical, fields.lsf), {})[fields.tile_unroll] = key
    return programs


FINALIZE_ROLE = "finalize"  # CUB block radix sort programs
FINALIZE_RANK_ROLE = "finalize_rank"  # prefix-popcount rank finalize programs (a bitmap window per program)


def finalize_role(rank_window: Optional[int]) -> str:
    """Stage role of the finalize program a call runs: the rank program when a bitmap window is named, else CUB."""
    return FINALIZE_RANK_ROLE if rank_window is not None else FINALIZE_ROLE


RANK_ITEMS_DEFAULT = 8  # sort slots per rank finalize thread of the production form (the kernel's FINALIZE_ITEMS); the 16-item form carries ``:i16``


def finalize_key(
    threads: int,
    rank_window: Optional[int] = None,
    staged: bool = False,
    two_level: bool = False,
    *,
    items: int = RANK_ITEMS_DEFAULT,
    bulk: bool = False,
    persistent: bool = False,
) -> str:
    """Program key of the finalize a call runs: ``finalize:t<threads>`` (CUB) or ``finalize_rank:t<threads>:w<window>`` (the rank
    scatter), with ``:two`` appended for its two-level (bucket + word) form (lever FR2), ``:staged`` for its SMEM-staged form
    (lever FRS), ``:i16`` for the staged form's 16-item threads (lever FRT; ``threads`` is then the launched 16-item thread count)
    and ``:bulk`` for the staged form's ``cp.async.bulk`` row I/O (lever FRB): ``finalize_rank:t256:w16384:staged:i16:bulk``; the
    persistent two-row pipelined form of a bulk program (lever FRP-K) appends ``:persist``: ``finalize_rank:t256:w8192:staged:bulk:persist``."""
    items = int(items)
    if persistent and not bulk:
        raise ValueError(
            "the persistent rank finalize is a form of the staged bulk-I/O program (staged=True, bulk=True)"
        )
    if rank_window is not None:
        key = f"{FINALIZE_RANK_ROLE}:t{int(threads)}:w{int(rank_window)}"
        if two_level:
            key += ":two"
        if staged:
            key += ":staged"
        elif items != RANK_ITEMS_DEFAULT or bulk:
            raise ValueError(
                "the 16-item thread form and the bulk row I/O are forms of the staged rank finalize (staged=True)"
            )
        if items == 16:
            key += ":i16"
        elif items != RANK_ITEMS_DEFAULT:
            raise ValueError(
                f"a rank finalize thread sorts {RANK_ITEMS_DEFAULT} or 16 slots, got items={items!r}"
            )
        if bulk:
            key += ":bulk"
        return f"{key}:persist" if persistent else key
    if staged or two_level or items != RANK_ITEMS_DEFAULT or bulk or persistent:
        raise ValueError(
            "the staged, two-level, 16-item, bulk-I/O and persistent forms belong to the rank finalize programs (a call without a bitmap window sorts with CUB)"
        )
    return f"{FINALIZE_ROLE}:t{int(threads)}"


def stage_slug(key: str) -> str:
    """Stage / template name of a program key (``scan:narrow:u1:s0:f0`` -> ``scan_narrow_u1_s0_f0``)."""
    return key.replace(":", "_")


@dataclass(frozen=True)
class ProgramChoice:
    """Everything the host decides for one call from host-known integers."""

    num_queries: int
    num_keys: int
    num_segments: int
    ratio: int
    top_k: int
    max_seqlen_k: Optional[
        int
    ]  # the call's bound on every key segment (None = not given); sizes the rank finalize's pool under lever RW
    grid: int  # requested persistent grid (SM count or the explicit knob)
    grid_ctas: int  # launched persistent grid (even for the CTA-pair program)
    sample_tiles_max: int
    sample_shift_permille: int
    far_sample: int
    cand_cap_multiplier: int
    cand_cap: int
    first_cap: int
    check_period: int
    block_q: int
    unit: str
    n_split: int
    split: bool
    pair: bool
    kind: str
    physical_kind: str
    snake: bool
    tile_unroll: int
    sample_fit: bool
    lsf: bool  # the scan program's longest-first segment order copy (lever LSF, scan key suffix :lsf)
    finalize_threads: int
    key_bits: int
    rank_window: Optional[
        int
    ]  # bitmap window of the rank finalize program, None = the CUB program
    rank_staged: (
        bool  # the rank finalize runs its SMEM-staged, coalesced-I/O form (lever FRS)
    )
    rank_two_level: (
        bool  # the rank finalize runs its two-level (bucket + word) form (lever FR2)
    )
    rank_threads: int  # launched threads of the rank finalize program (finalize_threads, or half of it for the 16-item form, lever FRT)
    rank_items: (
        int  # sort slots per rank finalize thread (8, or 16 for the 16-item form)
    )
    rank_bulk_io: (
        bool  # the rank finalize moves its rows with cp.async.bulk (lever FRB)
    )
    rank_persistent: bool  # the persistent two-row pipelined form of the staged bulk rank finalize (lever FRP-K; key tag :persist)
    rank_persist_ctas: int  # persistent CTAs per SM of that form (0 when the call runs a one-row program)
    rank_persist_spare: (
        int  # SMs that form's grid leaves out (0 when the call runs a one-row program)
    )
    scan_key: str
    merge_key: Optional[str]
    merge_rank: bool  # the merge program finalizes the split rows itself (key merge:rank:w*): no finalize launch
    finalize_key: str  # the finalize program of the call (recorded even when merge_rank fuses it into the merge)
    scan_entries: int  # int64 candidate entries of the persistent buffers
    staging_entries: int  # int64 split-staging entries behind them (0 without a split)
    workspace_bytes: int

    @property
    def keys(self) -> tuple:
        """Program keys in launch order."""
        return (
            (self.scan_key,)
            + ((self.merge_key,) if self.merge_key is not None else ())
            + (() if self.merge_rank else (self.finalize_key,))
        )

    @property
    def finalize_role(self) -> str:
        """Stage role of the finalize program (``finalize`` = CUB, ``finalize_rank`` = the rank scatter)."""
        return finalize_role(self.rank_window)

    def as_dict(self) -> dict:
        return asdict(self)


def select_program(
    policy: DispatchPolicy,
    *,
    num_queries: int,
    num_keys: int,
    num_segments: int,
    ratio: int,
    top_k: int,
    grid: int,
    candidate_multiplier: Optional[int] = None,
    check_period: Optional[int] = None,
    sample_tiles_max: Optional[int] = None,
    sample_shift_permille: Optional[int] = None,
    far_sample: Optional[int] = None,
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
    scan_keys: Optional[Collection[str]] = None,
) -> ProgramChoice:
    """The host dispatch of one call, in the kernel launcher's own order.

    The keyword knobs change the internal partition and scheduling only
    (results are bitwise identical for every admissible value); ``None`` takes
    the policy's default (``sample_shift_permille``: ``sample_shift_for`` of
    the call's ``top_k``).  ``far_sample`` (``0`` or ``1``) is the far-sample
    first threshold switch of the ascending-tile-order scan programs (the
    descending-order and wide programs ignore it).  ``max_seqlen_k`` (the call's bound on every key
    segment, optional) sizes the rank finalize's bitmap pool under lever RW
    (``rank_seg_window``); results are identical with and without it.  With the sampled first threshold off
    (``sample_tiles_max=0``) the candidate capacity follows the record's ``*_unsampled`` values.  The finalize program is the prefix-popcount rank
    scatter where ``finalize_rank_for`` names a bitmap window for the call
    (``finalize_rank:t<threads>:w<window>``; ``:staged`` appended where ``finalize_rank_staged_for``
    admits its SMEM-staged form, lever FRS), else the CUB block radix sort
    (``finalize:t<threads>``); both produce the same order and bits.  The staged form's 16-item threads (``:i16``, lever FRT) and
    ``cp.async.bulk`` row I/O (``:bulk``, lever FRB) follow ``finalize_rank_t16_for`` / ``finalize_rank_bulk_io_for``;
    ``outputs_aligned`` (default True: the backend's own allocations) tells the bulk rule whether the output addresses meet
    ``rank_bulk_align_bytes`` -- caller-owned unaligned views run the plain twin program, same results.
    ``scan_keys`` (the scan program keys the architecture registers, ``PROGRAM_KEYS[arch]``; None: no registry, the
    preference stays off) bounds the unpaired-l6 preference (``l6_solo_rule``, ``l6_solo_unsampled_max_queries``): it replaces the unit and pair choice of a call that does not
    split with the unpaired l6 unit only where that unit would not split and its program key is registered.  A record's
    ``wide_pair_snake_only`` (sampled calls) drops the CTA pair of a wide unit that would not walk the snake order where the
    unpaired wide program is in ``scan_keys`` (without a registry it stays off, as the l6 preference); ``split_wave_scope``
    limits the two-way wave split.  A record's ``desc_walk_kinds`` runs the descending-walk twin of the decided program
    (key kind ``<kind>_desc``) where it names the kind for the call's sampling mode and ``scan_keys`` has the twin.
    Where the record sets ``lsf_programs``, a packed call of unequal key segments runs the longest-first copy of the
    decided program (key suffix ``:lsf``) unless ``scan_keys`` has the program but not the copy; every rule that asks
    ``scan_keys`` for a program looks up its key without the suffix, so the copy never changes the plan.
    ``snake_packed_rule`` keeps packed calls of few mean unit tiles over more than a few
    waves in the plain unit order (``snake_for``), and ``snake_l6_fallback_waves`` lets a sampling-off call the l6 preference moves run the plain-order
    l6 program where the snake-order one is not registered, within that many waves of the planned program's persistent
    grid (``snake_l6_fallback_for``).
    """
    T, Tkv, S, r, K = (
        int(num_queries),
        int(num_keys),
        int(num_segments),
        int(ratio),
        int(top_k),
    )
    grid = int(grid)
    if grid < 1:
        raise ValueError("grid_ctas must be >= 1")
    tiles = (
        policy.sample_tiles_for(T, Tkv, S, r)
        if sample_tiles_max is None
        else int(sample_tiles_max)
    )
    sampled = tiles > 0
    shift = (
        policy.sample_shift_for(K)
        if sample_shift_permille is None
        else int(sample_shift_permille)
    )
    far = policy.far_sample if far_sample is None else int(far_sample)
    if far not in (0, 1):
        raise ValueError(f"far_sample must be 0 or 1, got {far_sample!r}")
    mult = (
        policy.cand_cap_multiplier_for(T, Tkv, S, r, K, sampled=tiles > 0)
        if candidate_multiplier is None
        else int(candidate_multiplier)
    )
    block_q = policy.unit_block_q_for(T, Tkv, S, r, grid, K)
    n_split = policy.n_split_for(
        T, Tkv, S, r, grid, block_q, sampled=tiles > 0, top_k=K
    )
    split = n_split > 1
    pair = (not split) and policy.pair_for(T, Tkv, S, r, grid, K, block_q)
    # lever LSF (#7549): the longest-first segment order pays off only where the segments differ, so the copy that
    # carries it runs on packed calls whose key segments are not all equal (one segment, or num_keys == S x the
    # call's max_seqlen_k, runs the program without it; without max_seqlen_k the host cannot tell and runs the copy).
    # The copy never changes the plan: every rule below that asks ``scan_keys`` for a program looks up its key without
    # the suffix, and the decided program takes the suffix last, where the registry has its copy
    lsf = (
        policy.lsf_programs
        and S >= LSF_MIN_SEGMENTS
        and not (max_seqlen_k is not None and Tkv == S * int(max_seqlen_k))
    )
    cap = policy.candidate_capacity(K, mult)
    # wide_pair_snake_only: a sampled call keeps the CTA pair of the wide unit only where the pair walks the snake order;
    # elsewhere it runs the unpaired wide program (same unit, no split) when that program is in ``scan_keys`` (no registry: off)
    if (
        pair
        and tiles > 0
        and policy.wide_pair_snake_only
        and block_q == policy.block_q_wide
        and not policy.snake_for(
            policy.program_kind(block_q, False, True), T, Tkv, S, r, sampled, grid
        )
    ):
        wide_snake = policy.snake_for("wide", T, Tkv, S, r, sampled, grid)
        wide_key = scan_key(
            physical_kind("wide", "wide"),
            policy.tile_unroll_for("wide", K, wide_snake, block_q),
            wide_snake,
            policy.sample_fit_for(T, Tkv, S, r, cap, physical_kind("wide", "wide")),
        )
        if scan_keys is not None and wide_key in scan_keys:
            pair = False
    # l6_solo_rule / l6_solo_unsampled_max_queries: a call that does not split takes the l6 unit without a CTA pair where
    # ``l6_solo_for`` prefers it (sampling off: every call, or with the query bound a CTA-pair l6 plan of at most that many
    # queries; sampled: several segments or short units), the l6 unit would not split either (the split rule
    # alone, before the registration guard of ``n_split_for``) and the program it picks is in ``scan_keys``; else the plan above.
    # snake_l6_fallback_waves: a sampling-off call whose snake-order l6 program is not registered takes the plain-order
    # one instead where the planned program needs at most that many waves of its persistent grid
    solo_plain = False
    if (
        not split
        and (block_q, pair) != (policy.block_q_l6, False)
        and policy.l6_solo_for(
            T,
            Tkv,
            S,
            r,
            sampled=tiles > 0,
            pair_l6=(block_q, pair) == (policy.block_q_l6, True),
        )
        and policy.split_rule_for(T, Tkv, S, r, grid, policy.block_q_l6, tiles > 0) == 1
    ):
        solo_snake = policy.snake_for("l6", T, Tkv, S, r, sampled, grid)
        solo_fit = policy.sample_fit_for(T, Tkv, S, r, cap, "l6")
        solo_key = scan_key(
            "l6",
            policy.tile_unroll_for("l6", K, solo_snake, policy.block_q_l6),
            solo_snake,
            solo_fit,
        )
        if (
            scan_keys is not None
            and solo_key not in scan_keys
            and solo_snake
            and policy.snake_l6_fallback_for(T, block_q, pair, grid, sampled)
        ):
            solo_key = scan_key(
                "l6",
                policy.tile_unroll_for("l6", K, False, policy.block_q_l6),
                False,
                solo_fit,
            )
            solo_plain = True
        if scan_keys is not None and solo_key in scan_keys:
            block_q, pair = policy.block_q_l6, False
        else:
            solo_plain = False
    kind = policy.program_kind(block_q, split, pair)
    unit = policy.unit_name(block_q)
    snake = False if solo_plain else policy.snake_for(kind, T, Tkv, S, r, sampled, grid)
    unroll = policy.tile_unroll_for(kind, K, snake, block_q)
    grid_ctas = grid - grid % 2 if pair else grid
    physical = physical_kind(kind, unit)
    tiles, shift = sample_class_for(
        policy,
        physical,
        T,
        Tkv,
        S,
        r,
        tiles,
        shift,
        tiles_knob=sample_tiles_max is not None,
        shift_knob=sample_shift_permille is not None,
    )
    fit = policy.sample_fit_for(T, Tkv, S, r, cap, physical)
    if check_period is None:
        period = policy.check_period_for(K, cap, kind)
    else:
        period = int(check_period)
        limit = policy.check_period_limit(K, cap)
        if not 1 <= period <= limit:
            raise ValueError(
                f"check_period must be in [1, {limit}] for top_k={K}, cand_cap={cap}, got {check_period!r}"
            )
    if finalize_threads is None:
        fin = policy.finalize_threads_for(K)
    else:
        fin = int(finalize_threads)
        if not policy.finalize_threads_admissible(fin, K):
            raise ValueError(
                f"finalize_threads must be one of {policy.finalize_threads_fit} (top_k <= threads x {policy.finalize_items}), "
                f"{policy.finalize_threads_small} (top_k <= {policy.finalize_top_k_small}) or {policy.finalize_threads}, got {finalize_threads!r}"
            )
    rank_window = policy.finalize_rank_for(K, Tkv, S, fin, max_seqlen_k=max_seqlen_k)
    rank_two_level = policy.rank_two_level_form(rank_window)
    rank_staged = policy.finalize_rank_staged_for(rank_window, fin, rank_two_level)
    # lever FRT: the staged form's 16-item threads on the production slot counts; lever FRB: its bulk row I/O where top_k and the
    # output addresses meet the granule (the launcher's order: the staged decision at the 8-item thread count, then the forms)
    rank_t16 = policy.finalize_rank_t16_for(fin, rank_staged)
    rank_threads, rank_items = (
        policy.rank_t16_form(fin) if rank_t16 else (fin, int(policy.finalize_items))
    )
    rank_bulk_io = policy.finalize_rank_bulk_io_for(K, rank_staged, outputs_aligned)
    # lever FRP-K: the persistent two-row pipelined program on the architecture's staged bulk slab calls with 8-item threads up to its
    # top_k cap (the 16-item form -- a K = 4096 row, or the 512-thread ``finalize_threads`` knob -- keeps the one-row program)
    rank_persistent = policy.finalize_rank_persistent_for(
        K, rank_staged, rank_two_level, rank_bulk_io, rank_t16
    )
    merge_rank = policy.merge_rank_for(split, K, rank_window, rank_two_level)
    scan_entries = grid_ctas * block_q * cap
    staging = T * n_split * K if split else 0
    key = scan_key(physical, unroll, snake, fit)
    # desc_walk_kinds: the descending-walk twin of the decided program, where the record names its kind for the call's
    # sampling mode and the registry has it (no registry: off); the plan, the knobs and the workspace stay
    if scan_keys is not None and policy.desc_walk_for(physical, tiles > 0):
        desc_key = scan_key(f"{physical}_desc", unroll, snake, fit)
        if desc_key in scan_keys:
            key = desc_key
    # lever LSF: the decided program runs its longest-first copy, but where the registry has the program and not its
    # copy it runs in the argument order (sm_107a registers the copies of its two l6 ticket programs only, sm_100a and
    # sm_103a that of every program); without a registry, or for a key it has in neither form (the BF16 key of an FP8
    # decision, which ``select_fp8_program`` maps to an FP8 program), the copy stays
    if (
        lsf
        and scan_keys is not None
        and key in scan_keys
        and key + ":lsf" not in scan_keys
    ):
        lsf = False
    if lsf:
        key += ":lsf"
    return ProgramChoice(
        num_queries=T,
        num_keys=Tkv,
        num_segments=S,
        ratio=r,
        top_k=K,
        max_seqlen_k=None if max_seqlen_k is None else int(max_seqlen_k),
        grid=grid,
        grid_ctas=grid_ctas,
        sample_tiles_max=tiles,
        sample_shift_permille=shift,
        far_sample=far,
        cand_cap_multiplier=mult,
        cand_cap=cap,
        first_cap=cap,
        check_period=period,
        block_q=block_q,
        unit=unit,
        n_split=n_split,
        split=split,
        pair=pair,
        kind=kind,
        physical_kind=physical,
        snake=snake,
        tile_unroll=unroll,
        sample_fit=fit,
        lsf=lsf,
        finalize_threads=fin,
        key_bits=policy.finalize_key_bits_for(Tkv),
        rank_window=rank_window,
        rank_staged=rank_staged,
        rank_two_level=rank_two_level,
        rank_threads=rank_threads,
        rank_items=rank_items,
        rank_bulk_io=rank_bulk_io,
        rank_persistent=rank_persistent,
        rank_persist_ctas=int(policy.rank_persist_ctas_per_sm)
        if rank_persistent
        else 0,
        rank_persist_spare=int(policy.rank_persist_spare_sms) if rank_persistent else 0,
        scan_key=key,
        merge_key=(merge_key(rank_window) if merge_rank else MERGE_KEY)
        if split
        else None,
        merge_rank=merge_rank,
        finalize_key=finalize_key(
            rank_threads,
            rank_window,
            rank_staged,
            rank_two_level,
            items=rank_items,
            bulk=rank_bulk_io,
            persistent=rank_persistent,
        ),
        scan_entries=scan_entries,
        staging_entries=staging,
        workspace_bytes=(scan_entries + staging) * policy.candidate_entry_bytes
        + TICKET_TAIL_BYTES,
    )


def fp8_program_choice(
    policy: DispatchPolicy,
    choice: ProgramChoice,
    key: str,
    n_split: Optional[int] = None,
) -> ProgramChoice:
    """``choice`` run by the FP8 scan program ``key``: the unit geometry, the split and the tile unroll follow the key,
    an FP8 program has no CTA pair (the launched grid is the requested one), and the capacity, check period, sampling
    knobs and finalize of ``choice`` stay.  ``n_split``: the key-range splits of a split key (default: the plan's, at
    least 2); the workspace bound follows the unit geometry and the split."""
    fields = scan_key_fields(key)
    if fields is None or fields.form != FP8:
        raise ValueError(f"{key!r} is not an FP8 scan program key")
    physical = fields.physical
    split = physical.startswith("split_")
    unit = physical[len("split_") :] if split else physical
    if unit not in UNITS:
        raise ValueError(f"no FP8 program runs the scan kind of {key!r}")
    block_q = {
        "narrow": policy.block_q_narrow,
        "wide": policy.block_q_wide,
        "l6": policy.block_q_l6,
    }[unit]
    if not split:
        n = 1
    elif n_split is None:
        n = max(2, int(choice.n_split))
    else:
        n = int(n_split)
        if n < 2:
            raise ValueError(f"a split program needs n_split >= 2, got {n_split!r}")
    grid_ctas = int(choice.grid)
    scan_entries = grid_ctas * block_q * int(choice.cand_cap)
    staging = int(choice.num_queries) * n * int(choice.top_k) if split else 0
    return replace(
        choice,
        grid_ctas=grid_ctas,
        block_q=block_q,
        unit=unit,
        n_split=n,
        split=split,
        pair=False,
        kind="split" if split else unit,
        physical_kind=physical,
        snake=fields.snake,
        tile_unroll=fields.tile_unroll,
        sample_fit=fields.sample_fit,
        lsf=fields.lsf,
        scan_key=key,
        merge_key=MERGE_KEY if split else None,
        merge_rank=False,  # the FP8 form keeps the merge + finalize programs of its records
        scan_entries=scan_entries,
        staging_entries=staging,
        workspace_bytes=(scan_entries + staging) * policy.candidate_entry_bytes
        + TICKET_TAIL_BYTES,
    )


def fallback_knobs(
    policy: DispatchPolicy, choice: ProgramChoice, physical: str, call: dict
) -> ProgramChoice:
    """``choice`` with the sampled first threshold and the check period of the scan kind ``physical`` that runs in place
    of the decided one (``FP8_KIND_FALLBACK``): the record's ``sample_class_rule`` and ``check_period_kind_overrides``
    key on the program kind, so a wide decision's wide-unit shift and check period do not carry over to the l6
    program.  The knobs ``call`` sets stay."""
    T, Tkv, S, r, K = (
        choice.num_queries,
        choice.num_keys,
        choice.num_segments,
        choice.ratio,
        choice.top_k,
    )
    tiles_knob = call.get("sample_tiles_max")
    shift_knob = call.get("sample_shift_permille")
    tiles, shift = sample_class_for(
        policy,
        physical,
        T,
        Tkv,
        S,
        r,
        policy.sample_tiles_for(T, Tkv, S, r)
        if tiles_knob is None
        else int(tiles_knob),
        policy.sample_shift_for(K) if shift_knob is None else int(shift_knob),
        tiles_knob=tiles_knob is not None,
        shift_knob=shift_knob is not None,
    )
    period = choice.check_period
    if call.get("check_period") is None:
        kind = "split" if physical.startswith("split_") else physical
        period = policy.check_period_for(K, choice.cand_cap, kind)
    return replace(
        choice, sample_tiles_max=tiles, sample_shift_permille=shift, check_period=period
    )


def select_fp8_program(
    policy: DispatchPolicy,
    *,
    scan_keys: Collection[str],
    fp8_programs: Optional[dict] = None,
    **call: Any,
) -> ProgramChoice:
    """The host dispatch of one FP8 call: the decision procedure of ``select_program`` (``call``: its keywords) under the
    architecture's FP8 record ``policy`` (``fp8_policy_record``) with the registry's ``scan_keys``, run by the FP8
    program (``fp8_program_choice``) of the decided physical kind -- of its ``FP8_KIND_FALLBACK`` kind, with that
    kind's sampled first threshold and check period (``fallback_knobs``), where no FP8 program of the decided kind is
    registered -- in the decided segment order (``:lsf`` where its FP8 copy is registered; also a one-segment call of
    the wide unit within the record's ``lsf_single_rule``, ``DispatchPolicy.lsf_single_fits``), with the decided tile
    unroll where an FP8 program of it is registered, else the smallest registered one, and in the snake unit order
    where the decision marks it, the registry has the snake copy (``s1``) of that program and the call stays within
    the record's ``snake_cap_rule`` (``DispatchPolicy.snake_copy_fits``).  ``fp8_programs``:
    ``fp8_scan_programs(scan_keys)``, made once by the caller.  Raises ``NotImplementedError`` when no FP8 program
    serves the kind."""
    choice = select_program(policy, scan_keys=scan_keys, **call)
    programs = fp8_scan_programs(scan_keys) if fp8_programs is None else fp8_programs
    physical = choice.physical_kind
    if not any(kind == physical for kind, _lsf in programs):
        physical = FP8_KIND_FALLBACK.get(physical, physical)
    # lsf_single_rule: a one-segment call of the wide unit runs the :lsf copy within the rule (the plan leaves it off)
    lsf = choice.lsf or (
        physical == "wide"
        and (physical, True) in programs
        and policy.lsf_single_fits(
            choice.num_queries, choice.num_keys, choice.num_segments, choice.ratio
        )
    )
    unrolls = programs.get((physical, lsf)) or programs.get((physical, False))
    if not unrolls:
        raise NotImplementedError(
            f"the FP8 dispatch selected the {physical!r} scan kind, which no registered FP8 program serves "
            f"(registered FP8 kinds: {sorted({kind for kind, _lsf in programs})})"
        )
    key = unrolls.get(choice.tile_unroll, unrolls[min(unrolls)])
    if physical != choice.physical_kind:
        choice = fallback_knobs(policy, choice, physical, call)
    plain = fp8_program_choice(policy, choice, key)
    twin = key.replace(":s0:", ":s1:", 1)
    if (
        choice.snake
        and twin in scan_keys
        and policy.snake_copy_fits(
            plain.num_queries, plain.num_segments, plain.block_q, plain.grid_ctas
        )
    ):
        return fp8_program_choice(policy, choice, twin)
    return plain


__all__ = [
    "BF16",
    "FINALIZE_RANK_ROLE",
    "FINALIZE_ROLE",
    "FORMS",
    "FP8",
    "FP8_KIND_FALLBACK",
    "KINDS",
    "MERGE_KEY",
    "MERGE_RANK_MAX_TOP_K",
    "POLICY_FIELDS",
    "RANK_ITEMS_DEFAULT",
    "SAMPLE_SHIFT_PERMILLE_BOUNDS",
    "UNITS",
    "DispatchPolicy",
    "ProgramChoice",
    "ScanKeyFields",
    "fallback_knobs",
    "finalize_key",
    "finalize_role",
    "fp8_policy_record",
    "fp8_program_choice",
    "fp8_scan_programs",
    "merge_key",
    "physical_kind",
    "sample_class_for",
    "scan_key",
    "scan_key_fields",
    "scan_key_form",
    "select_fp8_program",
    "select_program",
    "stage_slug",
    "ticket_static_path",
    "ticket_units_bound",
]
