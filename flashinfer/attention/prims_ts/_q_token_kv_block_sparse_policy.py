# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only sparse policy shared by group recommendation and launch planning.

Reuse common FMHA geometry, split and scheduler selection. Explicit G remains
caller-owned; only the recommendation evaluates alternative groups.
"""

from dataclasses import dataclass, replace

# Groups beyond G8 use two (G16) or four (G32) page-major membership planes and
# the Keeps recipes; the recommendation considers powers of two up to G32.
MAX_GROUP_SIZE = 32
SUPPORTED_GROUP_SIZES = (*range(1, 9), 16, 32)
# Widest sparse block for G16/G32: their Keeps softmax reads each lane's
# memberships four blocks per word, so a lane must own at least four blocks.
MAX_WIDE_GROUP_BLOCK_SIZE = 16


def membership_planes(group_size: int) -> int:
    """Page-major membership byte planes per route row: ceil(G / 8)."""
    return (group_size + 7) // 8


MAX_TILE_SIZE_Q = 128
TILE_SIZE_KV = 128
# Standalone reducer capacity; work and the first SM wave bound actual fanout.
MAX_SPLITS_KV = 128


@dataclass(frozen=True)
class SparseLaunch:
    tile_size_q: int
    num_insts_kv: int
    splits_kv: int
    head_dim_per_stage_kv: int
    o_stages: int
    use_persistent_scheduler: bool

    @property
    def use_keeps_mma_ab(self) -> bool:
        return self.tile_size_q >= 64


# Widest SwapsMmaAb Q tile: more Q rows per KV head need a KeepsMmaAb recipe.
MAX_SWAPS_TILE_SIZE_Q = 32


# Widest group, in Q rows per KV head, for which the D256 policy takes a
# SwapsMmaAb recipe. Measured on GB300 (QToken sparse, real indexer top-k,
# 0-64K prefix, 1-4096 queries): Swaps Q8/Q16 tiles can beat the 64-row Keeps
# tile they would mostly pad, but Swaps Q32 trails Keeps Q64 by 10-110% at
# every dtype once there are more than a few queries of work.
SWAPS_MAX_ROWS_D256 = 16


def prefers_nvfp4_keeps(q_dtype_key: str, group_size: int, heads_q_per_kv: int) -> bool:
    """Whether NVFP4 K/V should take the KeepsMmaAb transform target.

    ``group_size * heads_q_per_kv`` Q rows share one KV head. Wider groups than
    the widest Swaps Q tile, and groups beyond G8, need Keeps for either Q
    dtype. BF16 Q always takes it: its full-D256 BF16 stages measured faster
    than the TMEM-transformed Swaps recipe on GB300. FP8 Q takes it from five
    grouped tokens, where its per-Q-row softmax measured faster than the Swaps
    recipe's per-K-row softmax over transformed NVFP4, and for groups of more
    than ``SWAPS_MAX_ROWS_D256`` rows. The choice cannot depend on the scale
    layout, which plan() does not see; below that row bound neither recipe
    wins for both layouts.
    """
    rows = group_size * heads_q_per_kv
    if rows > MAX_SWAPS_TILE_SIZE_Q or group_size > 8:
        # Swaps reads one membership byte plane (at most eight queries).
        return True
    if q_dtype_key == "float8_e4m3fn":
        return group_size >= 5 or rows > SWAPS_MAX_ROWS_D256
    return q_dtype_key == "bfloat16"


def select_sparse_launch(
    *,
    group_size: int,
    heads_q_per_kv: int,
    head_dim: int,
    q_dtype_key: str,
    num_routes: int,
    num_kv_heads: int,
    route_kv_tokens: int,
    multi_processor_count: int,
    split_kv: bool = True,
    prefer_keeps_mma_ab: bool = False,
    nvfp4_kv: bool = False,
    cap_swaps_rows: bool = True,
) -> SparseLaunch:
    """Choose the smallest fitting dense Q recipe, then useful one-wave splits.

    G is already fixed by the caller. This function never changes it, queries
    the device, reads tensors, or applies a batch/context tuning table.
    ``prefer_keeps_mma_ab`` takes the one-instance Keeps recipe for wide-head
    groups even when the Swaps recipe would split KV. ``nvfp4_kv`` marks the
    NVFP4 transform path, whose recipe ``prefers_nvfp4_keeps`` already chose;
    the raw-dtype D256 rule below then does not apply. ``cap_swaps_rows=False``
    skips that rule too (the coverage G rule scores groups without it).
    """
    from .kernels.fmha_decode.fmha_decode_config import (
        MIN_LOOP_ITERS_PER_SPLIT,
        _select_auto_launch_mode,
        enumerate_grouped_q_mma_candidates,
        select_splits_kv,
    )

    # Splits must also cover non-shared KV in the union. Retain the route's
    # capacity bound here; runtime masks the inactive split prefix. The G
    # recommendation below scores useful work with a shared-KV estimate.
    work_tokens = max(1, route_kv_tokens)
    # Wide-head FP8 G1 uses the qualified staged Keeps recipe. D64/D128 also have
    # qualified unstaged Swaps recipes. G16/G32 need Keeps for any head size:
    # Swaps reads one membership byte plane (at most eight queries).
    minimum_q = (
        64
        if group_size > 8
        or (
            head_dim in (192, 256)
            and q_dtype_key == "float8_e4m3fn"
            and group_size == 1
        )
        else 8
    )
    candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=heads_q_per_kv, seq_len_q=group_size
    )

    # Choose the first Q tile covering the route group and minimum MMA rows.
    def recipe(minimum: int):
        return next(
            c
            for c in candidates
            if c.tile_size_q >= minimum and c.q_tokens_per_cta >= group_size
        )

    # Choose split fanout using KV instances per loop and a useful-work floor.
    def fanout(instances: int) -> int:
        maximum = MAX_SPLITS_KV
        minimum_iterations = MIN_LOOP_ITERS_PER_SPLIT
        if group_size == 1:
            # Staged FP8 Keeps has one KV tile per iteration, whereas Swaps
            # has two. Preserve a two-tile work floor for FP8 in either recipe:
            # one-tile splits spend more on partial-output reduction than they
            # save in attention. Scale the cap with work, not a fixed split count.
            minimum_tiles = (
                max(2, instances) if q_dtype_key == "float8_e4m3fn" else instances
            )
            minimum_iterations = minimum_tiles // instances
            # A partial final iteration must not buy an otherwise empty split.
            maximum = min(
                maximum, max(1, work_tokens // (TILE_SIZE_KV * minimum_tiles))
            )
        return select_splits_kv(
            seq_len_kv=work_tokens,
            batch_size=num_routes,
            num_heads_kv=num_kv_heads,
            tile_size_kv=TILE_SIZE_KV,
            num_insts_kv=instances,
            service_capacity=multi_processor_count,
            max_splits_kv=maximum,
            min_loop_iters_per_split=minimum_iterations,
            split_kv=split_kv,
        )

    candidate = recipe(minimum_q)
    instances = 1 if head_dim > 128 and candidate.variant == "keeps_mma_ab" else 2
    splits = fanout(instances)
    # Other wide-head FP8 groups may use Swaps only when split-KV is active.
    # A caller preferring Keeps (the NVFP4 Keeps transform target) takes it
    # for either Q dtype.
    use_keeps = prefer_keeps_mma_ab or (q_dtype_key == "float8_e4m3fn" and splits == 1)
    if head_dim == 256 and not nvfp4_kv and cap_swaps_rows:
        # Measured D256 raw-dtype rule (GB300, see SWAPS_MAX_ROWS_D256): FP8
        # and BF16 Swaps stop at 16 rows. BF16 previously took Swaps for every
        # group of up to 32 rows, up to 1.9x slower.
        if group_size * heads_q_per_kv > SWAPS_MAX_ROWS_D256 and q_dtype_key in (
            "float8_e4m3fn",
            "bfloat16",
        ):
            use_keeps = True
    if head_dim in (192, 256) and candidate.tile_size_q < 64 and use_keeps:
        candidate = recipe(64)
        instances = 1
        splits = fanout(instances)
    return SparseLaunch(
        tile_size_q=candidate.tile_size_q,
        num_insts_kv=instances,
        splits_kv=splits,
        # Q128 amortizes a full D192 stage; narrower Q tiles retain D128
        # bands. Full FP8 stages preserve the 128-byte swizzled planes while issuing
        # PV at N256. BF16 and Swaps retain their existing D128 head bands.
        head_dim_per_stage_kv=(
            192
            if head_dim == 192
            and candidate.tile_size_q == 128
            and q_dtype_key == "bfloat16"
            else 256
            if head_dim == 256
            and q_dtype_key == "float8_e4m3fn"
            and candidate.variant == "keeps_mma_ab"
            else 128
            if head_dim > 128
            else 0
        ),
        o_stages=1 if head_dim > 128 and candidate.variant == "keeps_mma_ab" else 2,
        # A caller that disables split-KV (packed prefill) plans each step's own
        # route count. Taking the persistent scheduler for every count keeps one
        # program per group size instead of a static/persistent pair keyed by
        # whether a step happens to exceed one SM wave. Within one wave the two
        # schedulers measure within about 1 us of each other.
        use_persistent_scheduler=(
            splits == 1
            and (
                not split_kv
                or _select_auto_launch_mode(
                    batch_size=num_routes,
                    num_heads_kv=num_kv_heads,
                    seq_len_kv=work_tokens,
                    tile_size_kv=TILE_SIZE_KV,
                    split_kv=False,
                    service_capacity=multi_processor_count,
                )
                == "persistent"
            )
        ),
    )


def candidate_union_tokens(
    group_size: int, selected_kv_tokens: int, block_size: int = 4
) -> int:
    """Bound a union in semantic blocks; G1 retains its exact tail length."""
    if group_size == 1:
        return selected_kv_tokens
    return (
        group_size * ((selected_kv_tokens + block_size - 1) // block_size) * block_size
    )


# A group's K/V union grows sub-linearly in G: real sparse-indexer outputs
# (prefill, 32K-64K context, 12 sparse layers) revisit a
# shared pool of blocks, so union(G) ~ selected * G**0.48 (about 1.4x per
# doubling of G, despite only ~60% adjacent-query overlap). The G latency
# model uses a smaller fitted exponent, which also absorbs per-tile overlap
# effects of larger groups.
DEFAULT_UNION_EXPONENT = 0.35
# Relative launch-cost model, in us, for GB300 D256 QToken recipes: one KV128
# tile of a work item by Q tile (Q64, Q128), per-work-item prologue and
# epilogue, grouped metadata by group class (G2-4 byte map, G5-8, G16, G32),
# and the split reduction. Only the argmin matters: with
# DEFAULT_UNION_EXPONENT the constants minimize chosen/best-measured latency
# (geomean, then worst case) over NVFP4 sweeps on real indexer top-k of two
# corpora (4/1 and 16/2 Q/KV heads per GPU, one request, 0-64K
# prefix, 1-4096 queries, explicit G in {1, 2, 4, 8, 16, 32}; 152 SMs),
# cross-validated between the corpora: geomean 1.000-1.001x, worst
# 1.01-1.04x of the best G. NVFP4 K/V is transformed in the kernel (no
# per-call K/V conversion), and the expected union is capped at the longest
# request's KV length (``seq_len_kv``).
_TILE_US = {64: 1.215, 128: 2.062}
_ITEM_US = 1.041
_METADATA_US = {4: 3.742, 8: 6.36, 16: 7.89, 32: 10.423}
_REDUCE_US = 0.83
# Candidate groups: powers of two up to G32 (two to four membership planes
# beyond G8), plus the largest legal group (a short request then forms one
# route). Other odd groups measured no better and only add JIT variants.
_CANDIDATE_GROUPS = (1, 2, 4, 8, 16, 32)


def expected_union_tokens(
    group_size: int,
    selected_kv_tokens: int,
    union_exponent: float = DEFAULT_UNION_EXPONENT,
) -> int:
    """Expected union of ``group_size`` adjacent queries' selections, in tokens."""
    grown = selected_kv_tokens * group_size**union_exponent
    return max(1, min(group_size * selected_kv_tokens, int(round(grown))))


def suggest_sparse_group(
    *,
    batch_size: int,
    seq_len_q: int,
    selected_kv_tokens: int,
    heads_q_per_kv: int,
    num_kv_heads: int,
    multi_processor_count: int,
    head_dim: int,
    q_dtype_key: str,
    split_kv: bool,
    block_size: int = 4,
    union_exponent: float = DEFAULT_UNION_EXPONENT,
    nvfp4_kv: bool = False,
    kv_tokens: int = 0,
    packed_query: bool = False,
) -> int:
    """Choose G by an estimated-latency model of the whole sparse call.

    For each legal G (powers of two up to G32 with at most TileQ128 rows),
    the selected recipe's configured split fanout is applied to the group's
    *expected* union (``expected_union_tokens``, capped at ``kv_tokens``,
    the longest request's KV length, when positive) to obtain the active splits
    and KV tiles per work item; the estimate is
    ``waves * (tiles_per_item * tile_cost[TileQ] + item_cost)`` plus the
    grouped metadata launch and the split reduction (active splits > 1).
    ``nvfp4_kv`` plans the NVFP4 transform recipes as the resolver does
    (``prefers_nvfp4_keeps``, else one-instance Swaps). G16/G32 are
    candidates only for ``packed_query`` launches with blocks of at most
    ``MAX_WIDE_GROUP_BLOCK_SIZE`` tokens, the routes ``plan`` accepts for them.
    BF16/FP8 caches (``nvfp4_kv=False``) keep the coverage rule of
    ``_suggest_sparse_group_by_coverage`` (G1-G8). Bounds and dtype validation
    belong to the public caller. No SQ % G divisibility is required. Ties
    prefer the smaller G.
    """
    from .kernels.fmha_decode.fmha_decode_config import (
        compute_runtime_active_splits_kv,
    )

    if not 0.0 <= union_exponent <= 1.0:
        raise ValueError("union_exponent must be in [0, 1]")
    if not nvfp4_kv:
        # The latency constants were fitted on NVFP4 K/V only; on BF16/FP8
        # caches they picked up to 8% slower groups than the coverage rule.
        return _suggest_sparse_group_by_coverage(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            selected_kv_tokens=selected_kv_tokens,
            heads_q_per_kv=heads_q_per_kv,
            num_kv_heads=num_kv_heads,
            multi_processor_count=multi_processor_count,
            head_dim=head_dim,
            q_dtype_key=q_dtype_key,
            split_kv=split_kv,
            block_size=block_size,
        )
    largest = min(MAX_GROUP_SIZE, seq_len_q, MAX_TILE_SIZE_Q // heads_q_per_kv)
    best_group, best_cost = 1, None
    candidates = sorted({g for g in _CANDIDATE_GROUPS if g <= largest} | {largest})
    for group in candidates:
        if group > 8 and (
            not packed_query
            or group not in SUPPORTED_GROUP_SIZES
            or block_size > MAX_WIDE_GROUP_BLOCK_SIZE
        ):
            continue
        full_groups, tail = divmod(seq_len_q, group)
        groups_per_request = full_groups + int(tail > 0)
        routes = batch_size * groups_per_request
        capacity = candidate_union_tokens(group, selected_kv_tokens, block_size)
        keeps = nvfp4_kv and prefers_nvfp4_keeps(q_dtype_key, group, heads_q_per_kv)
        launch = select_sparse_launch(
            group_size=group,
            heads_q_per_kv=heads_q_per_kv,
            head_dim=head_dim,
            # NVFP4 Swaps transforms to BF16 for either Q dtype.
            q_dtype_key="bfloat16" if nvfp4_kv and not keeps else q_dtype_key,
            num_routes=routes,
            num_kv_heads=num_kv_heads,
            route_kv_tokens=capacity,
            multi_processor_count=multi_processor_count,
            split_kv=split_kv,
            prefer_keeps_mma_ab=keeps,
            nvfp4_kv=nvfp4_kv,
        )
        if nvfp4_kv and not keeps:
            # The NVFP4 Swaps transform runs one KV instance.
            launch = replace(launch, num_insts_kv=1)
        union = expected_union_tokens(group, selected_kv_tokens, union_exponent)
        if kv_tokens > 0:
            union = min(union, kv_tokens)
        tiles = (union + TILE_SIZE_KV - 1) // TILE_SIZE_KV
        active_splits = compute_runtime_active_splits_kv(
            valid_k=union,
            tile_size_kv=TILE_SIZE_KV,
            num_insts_kv=launch.num_insts_kv,
            configured_splits_kv=launch.splits_kv,
        )
        tiles_per_item = (tiles + active_splits - 1) // active_splits
        items = routes * num_kv_heads * active_splits
        waves = (items + multi_processor_count - 1) // multi_processor_count
        tile_us = _TILE_US[128 if launch.tile_size_q > 64 else 64]
        cost = waves * (tiles_per_item * tile_us + _ITEM_US)
        if group > 1:
            cost += _METADATA_US[
                4 if group <= 4 else 8 if group <= 8 else 16 if group <= 16 else 32
            ]
        cost += _REDUCE_US if active_splits > 1 else 0.0
        if best_cost is None or cost < best_cost - 1e-9:
            best_group, best_cost = group, cost
    return best_group


def _suggest_sparse_group_by_coverage(
    *,
    batch_size: int,
    seq_len_q: int,
    selected_kv_tokens: int,
    heads_q_per_kv: int,
    num_kv_heads: int,
    multi_processor_count: int,
    head_dim: int,
    q_dtype_key: str,
    split_kv: bool,
    block_size: int,
) -> int:
    """Prefer grouping, then splitting; shrink G only for better SM coverage.

    The G1-G8 rule for BF16/FP8 caches. A final short Q group contributes only
    its live rows. A fully-shared per-query KV proxy estimates useful active
    splits, rather than counting every capacity-based split as active. Padding
    breaks ties in wave coverage; it must not keep an otherwise under-filled
    GPU grouped.
    """
    from .kernels.fmha_decode.fmha_decode_config import (
        compute_runtime_active_splits_kv,
        enumerate_grouped_q_mma_candidates,
    )

    largest = min(8, seq_len_q, MAX_TILE_SIZE_Q // heads_q_per_kv)
    dense_candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=heads_q_per_kv, seq_len_q=largest
    )
    # Retain the largest group fitting each dense Q tile. Smaller groups
    # sharing that tile only add partially occupied CTAs. At the smallest
    # tile, still allow further ungrouping to expose otherwise idle SMs.
    groups = {
        min(largest, candidate.q_tokens_per_cta) for candidate in dense_candidates
    }
    smallest_capacity = min(
        candidate.q_tokens_per_cta for candidate in dense_candidates
    )
    groups.update(range(1, min(largest, smallest_capacity) + 1))
    best_group, best_coverage, best_density = largest, -1.0, -1.0
    for group in sorted(groups, reverse=True):
        full_groups, tail = divmod(seq_len_q, group)
        routes = batch_size * (full_groups + int(tail > 0))
        length = candidate_union_tokens(group, selected_kv_tokens, block_size)
        # Score with the uncapped Swaps recipes, as before the 16-row Swaps
        # bound: the bound only speeds up the group chosen here, while scoring
        # its one-instance Keeps fanout picked smaller, slower groups.
        launch = select_sparse_launch(
            group_size=group,
            heads_q_per_kv=heads_q_per_kv,
            head_dim=head_dim,
            q_dtype_key=q_dtype_key,
            num_routes=routes,
            num_kv_heads=num_kv_heads,
            route_kv_tokens=length,
            multi_processor_count=multi_processor_count,
            split_kv=split_kv,
            cap_swaps_rows=False,
        )

        # Score active work, not the configured union-capacity split grid.
        active_splits = compute_runtime_active_splits_kv(
            valid_k=length // group,
            tile_size_kv=TILE_SIZE_KV,
            num_insts_kv=launch.num_insts_kv,
            configured_splits_kv=launch.splits_kv,
        )
        per_request = (full_groups + int(tail > 0)) * active_splits
        grid = batch_size * num_kv_heads * per_request
        waves = (grid + multi_processor_count - 1) // multi_processor_count
        coverage = grid / waves
        # A short final group still pays for the full Q tile. Prefer less
        # padding only when active SM coverage is equal.
        live_rows = batch_size * num_kv_heads * seq_len_q * heads_q_per_kv
        density = live_rows / (routes * num_kv_heads * launch.tile_size_q)
        if coverage > best_coverage or (
            coverage == best_coverage and density > best_density
        ):
            best_group, best_coverage, best_density = group, coverage, density
        if group == largest and grid >= multi_processor_count:
            # The grouped Q grid already supplies a full first wave.
            break
        if coverage == multi_processor_count:
            break
    return best_group
