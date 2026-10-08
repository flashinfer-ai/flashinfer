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
never the user).

Nothing here reads device memory: the segment boundaries stay on the device
and the dispatch sees only host-known integers.  The chosen program never
changes a result -- every form computes the same ids and score bits -- so a
wrong choice costs time, not correctness.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Tuple

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
    "l6_rule",  # [max mean unit tiles, min top_k] of the three-warpgroup program, or None
    "wide_rule",  # [min wide rounds per CTA, min rounds x mean unit tiles] of the wide program
    "split_max",  # most key-range splits per unit
    "split_min_range_tiles",  # a split's key range is at least this many tiles
    "split_wave_rule",  # [min mean unit tiles, min CTA-efficiency gain] for a two-way split, or None
    "pair_rule",  # [min mean unit tiles, max mean unit tiles, min top_k] of the CTA-pair program, or None
    "snake_default",  # snake unit order when no rule applies
    "snake_rule",  # [program kinds, min mean tiles, min mean tiles (ratio > 1), min unit-cost spread], or None
    "tile_unroll_default",  # tile-loop unroll when no rule applies
    "tile_unroll_factor",  # tile-loop unroll when the rule applies
    "tile_unroll_rule",  # [program kinds, min top_k, yields to the snake order], or None
    "sample_fit_max_mean_tiles",  # sampled first threshold on buffer-fitting units up to this mean tile count
    "sample_tiles_max",  # sample-tile cap of long units (0 = no sampling)
    "sample_tiles_short_units",  # sample-tile cap of short units
    "sample_dispatch_mean_tiles_max",  # mean unit tiles up to which the short-unit cap applies
    "sample_tiles_tiny_units",  # sample-tile cap of tiny units (lever K3b)
    "sample_dispatch_tiny_tiles_max",  # mean unit tiles up to which the tiny-unit cap applies
    "sample_shift_permille",  # conservative shift of the sampled rank
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
    "rank_persist_ctas_per_sm",  # persistent CTAs per SM: the :persist programs launch min(num_queries, this x SM count) CTAs, each walking rows bid, bid + grid, ...
)

KINDS = ("narrow", "wide", "l6", "split", "pair_narrow", "pair_wide", "pair_l6")
UNITS = ("narrow", "wide", "l6")
MERGE_KEY = "merge"

_INT = ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes", "candidate_multiplier",
        "candidate_slack", "split_max", "split_min_range_tiles", "tile_unroll_default", "tile_unroll_factor",
        "sample_tiles_max", "sample_tiles_short_units", "sample_tiles_tiny_units", "sample_shift_permille",
        "check_period_max", "check_period_knob_max", "check_period_cap_divisor", "finalize_items",
        "finalize_threads_small", "finalize_top_k_small", "finalize_threads", "rank_top_k_min",
        "rank_slab_window_max", "rank_bulk_align_bytes", "rank_persist_max_k", "rank_persist_ctas_per_sm")  # fmt: skip
_BOOL = (
    "snake_default",
    "finalize_fit",
    "finalize_exact_key_bits",
    "rank_finalize",
    "rank_staged",
    "rank_seg_window",
    "rank_two_level",
    "rank_bulk_io",
    "rank_t16",
)
_NUMBER = (
    "sample_fit_max_mean_tiles",
    "sample_dispatch_mean_tiles_max",
    "sample_dispatch_tiny_tiles_max",
)
_OPTIONAL_INT = ("cand_cap_floor",)
_OPTIONAL_RULE = (
    "cand_mult_rule",
    "l6_rule",
    "split_wave_rule",
    "pair_rule",
    "snake_rule",
    "tile_unroll_rule",
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
    l6_rule: Optional[list]
    wide_rule: list
    split_max: int
    split_min_range_tiles: int
    split_wave_rule: Optional[list]
    pair_rule: Optional[list]
    snake_default: bool
    snake_rule: Optional[list]
    tile_unroll_default: int
    tile_unroll_factor: int
    tile_unroll_rule: Optional[list]
    sample_fit_max_mean_tiles: float
    sample_tiles_max: int
    sample_tiles_short_units: int
    sample_dispatch_mean_tiles_max: float
    sample_tiles_tiny_units: int
    sample_dispatch_tiny_tiles_max: float
    sample_shift_permille: int
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
        for name in ("tile_keys", "block_q_narrow", "block_q_wide", "block_q_l6", "candidate_entry_bytes",
                     "candidate_multiplier", "candidate_slack", "split_max", "split_min_range_tiles",
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

    def cand_cap_multiplier_for(
        self, num_queries: int, num_keys: int, num_segments: int, ratio: int, top_k: int
    ) -> int:
        mult = self.candidate_multiplier
        rule = self.cand_mult_rule
        if rule is not None:
            if (
                self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
                >= rule[0]
            ):
                mult = int(rule[1])
        floor = self.cand_cap_floor
        if floor is not None and int(top_k) * mult < int(floor):
            mult = -(-int(floor) // int(top_k))
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
        if l6 is not None and mean_tiles <= l6[0] and int(top_k) >= int(l6[1]):
            return self.block_q_l6
        min_rounds, min_work = self.wide_rule
        return (
            self.block_q_wide
            if rounds >= min_rounds and rounds * mean_tiles >= min_work
            else self.block_q_narrow
        )

    def n_split_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        block_q: int,
    ) -> int:
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
        return n

    def pair_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        grid: int,
        top_k: int,
    ) -> bool:
        rule = self.pair_rule
        if rule is None or int(grid) < 2:
            return False
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        return rule[0] <= mean_tiles <= rule[1] and int(top_k) >= int(rule[2])

    def snake_for(
        self, kind: str, num_queries: int, num_keys: int, num_segments: int, ratio: int
    ) -> bool:
        rule = self.snake_rule
        if rule is None or kind not in rule[0]:
            return self.snake_default
        r = max(1, int(ratio))
        mean_tiles = self.mean_unit_tiles(num_queries, num_keys, num_segments, ratio)
        spread = min(1.0, int(num_queries) / max(1.0, r * float(num_keys)))
        return mean_tiles >= (rule[1] if r == 1 else rule[2]) and spread >= rule[3]

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

    def sample_fit_for(
        self,
        num_queries: int,
        num_keys: int,
        num_segments: int,
        ratio: int,
        cand_cap: int,
    ) -> bool:
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


def finalize_grid(choice: "ProgramChoice", num_sms: int) -> int:
    """CTAs of a call's finalize launch: ``min(num_queries, rank_persist_ctas x SM count)`` for the persistent rank program (lever
    FRP-K; every CTA walks the rows ``bid, bid + grid, ...``), one CTA per row otherwise (at least one CTA)."""
    rows = max(1, int(choice.num_queries))
    if choice.rank_persistent:
        return max(1, min(rows, int(choice.rank_persist_ctas) * int(num_sms)))
    return rows


def physical_kind(kind: str, unit: str) -> str:
    """Name of the physical scan program kind: the split program is one per unit geometry."""
    return f"split_{unit}" if kind == "split" else kind


def scan_key(physical: str, tile_unroll: int, snake: bool, sample_fit: bool) -> str:
    return f"scan:{physical}:u{int(tile_unroll)}:s{int(bool(snake))}:f{int(bool(sample_fit))}"


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
    scan_key: str
    merge_key: Optional[str]
    finalize_key: str
    scan_entries: int  # int64 candidate entries of the persistent buffers
    staging_entries: int  # int64 split-staging entries behind them (0 without a split)
    workspace_bytes: int

    @property
    def keys(self) -> tuple:
        """Program keys in launch order."""
        return (
            (self.scan_key,)
            + ((self.merge_key,) if self.merge_key is not None else ())
            + (self.finalize_key,)
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
    finalize_threads: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    outputs_aligned: bool = True,
) -> ProgramChoice:
    """The host dispatch of one call, in the kernel launcher's own order.

    The keyword knobs change the internal partition and scheduling only
    (results are bitwise identical for every admissible value); ``None`` takes
    the policy's default.  ``max_seqlen_k`` (the call's bound on every key
    segment, optional) sizes the rank finalize's bitmap pool under lever RW
    (``rank_seg_window``); results are identical with and without it.  The finalize program is the prefix-popcount rank
    scatter where ``finalize_rank_for`` names a bitmap window for the call
    (``finalize_rank:t<threads>:w<window>``; ``:staged`` appended where ``finalize_rank_staged_for``
    admits its SMEM-staged form, lever FRS), else the CUB block radix sort
    (``finalize:t<threads>``); both produce the same order and bits.  The staged form's 16-item threads (``:i16``, lever FRT) and
    ``cp.async.bulk`` row I/O (``:bulk``, lever FRB) follow ``finalize_rank_t16_for`` / ``finalize_rank_bulk_io_for``;
    ``outputs_aligned`` (default True: the backend's own allocations) tells the bulk rule whether the output addresses meet
    ``rank_bulk_align_bytes`` -- caller-owned unaligned views run the plain twin program, same results.
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
    shift = (
        policy.sample_shift_permille
        if sample_shift_permille is None
        else int(sample_shift_permille)
    )
    mult = (
        policy.cand_cap_multiplier_for(T, Tkv, S, r, K)
        if candidate_multiplier is None
        else int(candidate_multiplier)
    )
    block_q = policy.unit_block_q_for(T, Tkv, S, r, grid, K)
    n_split = policy.n_split_for(T, Tkv, S, r, grid, block_q)
    split = n_split > 1
    pair = (not split) and policy.pair_for(T, Tkv, S, r, grid, K)
    kind = policy.program_kind(block_q, split, pair)
    unit = policy.unit_name(block_q)
    snake = policy.snake_for(kind, T, Tkv, S, r)
    unroll = policy.tile_unroll_for(kind, K, snake, block_q)
    grid_ctas = grid - grid % 2 if pair else grid
    cap = policy.candidate_capacity(K, mult)
    fit = policy.sample_fit_for(T, Tkv, S, r, cap)
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
    scan_entries = grid_ctas * block_q * cap
    staging = T * n_split * K if split else 0
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
        physical_kind=physical_kind(kind, unit),
        snake=snake,
        tile_unroll=unroll,
        sample_fit=fit,
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
        scan_key=scan_key(physical_kind(kind, unit), unroll, snake, fit),
        merge_key=MERGE_KEY if split else None,
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
        workspace_bytes=(scan_entries + staging) * policy.candidate_entry_bytes,
    )


__all__ = [
    "FINALIZE_RANK_ROLE",
    "FINALIZE_ROLE",
    "KINDS",
    "MERGE_KEY",
    "POLICY_FIELDS",
    "RANK_ITEMS_DEFAULT",
    "UNITS",
    "DispatchPolicy",
    "ProgramChoice",
    "finalize_key",
    "finalize_role",
    "physical_kind",
    "scan_key",
    "select_program",
    "stage_slug",
]
