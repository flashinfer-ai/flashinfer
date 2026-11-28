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

"""Automatic decode launch selection and the ``make_decode_config`` factory.

Grouped-Q MMA tiles, split-KV fanout and reduction mode, and the launch mode
are chosen here from the problem shape and the device capacity; ``config``
holds the static configuration they fill in.
"""

import cutlass.utils as utils
from collections.abc import Sequence
from copy import deepcopy
from cutlass import BFloat16, Float16
from dataclasses import dataclass, replace
from ....common.hardware import is_sm107_compile_target
from ....common.split_kv_mode import select_split_kv_modes
from .config import (
    FmhaDecodeConfig,
    LAUNCH_SELECTION_FIELDS,
    MMA_SELECTION_FIELDS,
    apply_config_source,
    apply_default_q_grouping,
    apply_dtype_config,
    apply_feature_config,
    apply_layout_config,
    apply_mask_type_config,
    apply_swaps_tile_config,
    cluster_smem_reduction_unsupported_reason,
    effective_seq_len_for_sliding,
    finalize_static_decode_config,
    get_max_active_clusters_for_cluster_size,
    make_q_tile_geometry,
    make_static_decode_config,
    resolve_decode_dtype_aliases,
    resolve_explicit_split_controls,
    set_if_implicit,
    validate_causal_decode_lengths,
    validate_kv256_static_config,
    validate_mixed_kv_dtype_profile,
    validate_profile_support,
    validate_sliding_window_args,
    validate_split_kv_mode,
)
from .device.constants import (
    AUTO_LAUNCH_TILE_SIZE_KV,
    FALLBACK_SM_COUNT_B200,
    FULL_GRID_SPLIT_MIN_TOKENS_PER_CTA,
    MIN_LOOP_ITERS_PER_SPLIT,
    SPLIT_KV_MIN_TOKENS_PER_CTA,
)
from .warp_roles import finalize_warp_roles


GROUPED_Q_MMA_TILES = (
    ("swaps_mma_ab", 8),
    ("swaps_mma_ab", 16),
    ("swaps_mma_ab", 32),
    ("swaps_mma_ab", 48),
    ("keeps_mma_ab", 64),
    ("keeps_mma_ab", 128),
    ("swaps_mma_ab", 64),
    ("swaps_mma_ab", 128),
)

# Empirical GQA-generation factors derived from matched reference measurements.
# The selector reuses the measured relative costs but resolves its own legal
# profiles, Q geometry, split fanout, and cluster promotion instead of copying
# a final shape policy.
# Both tables are keyed by a K/V width (``grouped_q_cost_kv_width``), then by
# TileQ. Width 16 scores 16-bit K and V at head dim 128, the one head dim whose
# TileQ64 candidate is promoted to the Q64/KV256 CTA; every other profile,
# 16-bit D192 included, takes width 8. The widths share every factor but one:
# a 16-bit D128 Keeps Q128 CTA runs its MMAs at half the 8-bit rate, so
# TileQ128 is scored against the Q64/KV256 CTA that the Q64 factor stands for
# after the KV256 promotion rather than with the shared TileQ128 factor.
GROUPED_Q_MAINLOOP_COST = {
    8: {8: 1.0, 16: 1.2, 32: 1.48, 48: 1.4, 64: 1.68, 128: 2.2},
    16: {8: 1.0, 16: 1.2, 32: 1.48, 48: 1.4, 64: 1.68, 128: 3.2},
}
GROUPED_Q_REDUCTION_COST = {
    8: {8: 1.0, 16: 1.03, 32: 1.08, 48: 1.15, 64: 1.2, 128: 1.32},
    16: {8: 1.0, 16: 1.03, 32: 1.08, 48: 1.15, 64: 1.2, 128: 1.32},
}
GROUPED_Q_REDUCTION_SEQ_LEN_FACTOR = 128.0

# GB300 NVFP4 measurements at B128/KV128K/Q4,Q6,Q8,Q12. Conversion and
# softmax include the wide Swaps profiles; revalidate their costs after changes
# to staging or register grants.
# Keeps Q64 with SMEM-converted NVFP4 K/V measured 1.26-1.50x (D128) and
# 1.05-1.10x (D256) faster than the transformed Swaps Q64 profile at 64
# grouped rows on GB300.
GROUPED_Q_KEEPS_TRANSFORM_MAINLOOP_COST = {(128, 64): 1.65, (256, 64): 1.56}
GROUPED_Q_TRANSFORM_MAINLOOP_COST = {
    (128, 48): 1.4,
    (128, 64): 2.1,
    (128, 128): 5.3,
    (256, 128): 3.8,
}


AUTO_SPLIT_KV_REDUCTION_MODES = (
    "gmem_reduction",
    "gmem_reduction_with_separate_kernel",
    "cluster_smem_reduction",
)


@dataclass(frozen=True)
class GroupedQMmaCandidate:
    """One grouped-Q MMA geometry candidate."""

    variant: str
    tile_size_q: int
    q_tokens_per_cta: int
    q_tiles: int


@dataclass(frozen=True)
class GroupedQLaunchCandidate:
    """One production-valid grouped-Q MMA and KV-split launch recipe."""

    mma: GroupedQMmaCandidate
    split_kv_mode: str
    splits_kv: int
    base_ctas: int
    launched_ctas: int
    seq_len_per_cta_kv: int
    waves: int
    modeled_time: float


def apply_auto_grouped_q_mma_config(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    auto_tuner: bool,
    split_kv_mode: str,
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    splits_kv: int,
    max_splits_kv: int | None,
    split_kv: bool = True,
) -> GroupedQLaunchCandidate | None:
    """Select a legal multi-Q MMA and KV-split recipe within caller constraints.

    Qualified transformed SQ1 grids share this search. Packed sliding windows,
    explicit launch modes, and ungrouped layouts retain the existing path.
    Explicit MMA fields filter candidates; launch fields constrain their recipes.
    FP8 Q with NVFP4 K/V at D128/D256 may also compare SMEM-converted Keeps.

    Fanout arguments here are caller overrides, before automatic selection.
    Each Q candidate gets its own automatic split search; final KV/reduction
    choices are applied after the joint recipe has been selected.
    """
    # TODO: Refine short-KV setup/epilogue costs. Wider dense Q tiles can
    # regress even when their current mainloop/reduction proxy scores lower.
    # The SQ1 path never splits a grid that already fills a wave. Transformed
    # NVFP4 K/V instead scores multi-wave splits that trim a partial final
    # wave (D512 B128/Hkv2/KV128K: 3.83 -> 3.32 ms with four splits).
    # SM107 D256 also compares underfilled Q1 grids for long KV scans.
    # Below 32K tokens the converter startup can outweigh grouped-Q reuse;
    # retain the existing single-token launch policy for those short grids.
    single_token_transform = (
        seq_len_q == 1
        and cfg.use_nvfp4_kv
        and cfg.use_fp8_pv
        and (
            batch_size * num_heads_kv >= get_max_active_clusters_for_cluster_size(1)
            or (
                is_sm107_compile_target()
                and cfg.headdim == 256
                and cfg.out_dtype == BFloat16
                and seq_len_kv >= 32768
            )
        )
    )
    if (
        not auto_tuner
        or split_kv_mode != "disabled"
        or (seq_len_q <= 1 and not single_token_transform)
        or not cfg.groups_tokens_heads_q
        or not cfg.use_paged_kv
        or (cfg.use_variable_seqlens_q and cfg.use_sliding_window_causal)
    ):
        return None

    candidates = enumerate_grouped_q_mma_candidates(
        heads_q_per_kv=num_heads_q // num_heads_kv,
        seq_len_q=seq_len_q,
        # FP8-Q/NVFP4-KV D128/D256 also have Keeps Q64 recipes with SMEM
        # converters. D512 Keeps Q64 is qualified at eight Q heads per KV head:
        # it ties Swaps Q64 in this model and precedes it in candidate order
        # (B300 q_len 8: B32 525 -> 432 us, B16 278 -> 225 us, B8 152 -> 121 us).
        swaps_only=cfg.use_transform_kv
        and not (
            cfg.use_nvfp4_kv
            and cfg.use_fp8_q
            and (
                cfg.headdim in (128, 256)
                or (cfg.headdim == 512 and num_heads_q // num_heads_kv == 8)
            )
        ),
        include_nvfp4_tiles=cfg.use_nvfp4_kv and cfg.use_fp8_q,
    )
    if not (cfg.use_transform_kv and cfg.use_fp8_pv):
        # Preserve established homogeneous Keeps and BF16-compute policies;
        # Q48+ Swaps costs are qualified for transformed FP8-compute K/V.
        candidates = tuple(
            candidate
            for candidate in candidates
            if candidate.variant != "swaps_mma_ab" or candidate.tile_size_q <= 32
        )
    if not candidates:
        return None
    service_capacity = get_max_active_clusters_for_cluster_size(1)
    if cfg.use_sliding_window_causal:
        # Wider Q tiles reuse overlapping windows, but only win when fewer
        # Q-grid waves repay their per-tile cost. Score the visible union,
        # not the model's maximum context. Windowed split-KV recipes remain
        # outside this selector; qualified direct Keeps tiles can reuse a
        # query group with fewer KV loads than narrower Swaps tiles. Partial
        # final groups retain their ceil-divided CTA count in the cost model.
        seq_len_kv = min(seq_len_kv, cfg.attention_window_size + seq_len_q - 1)
        split_kv = False
    supported = tuple(
        recipe
        for candidate in candidates
        for recipe in resolve_grouped_q_launch_candidates(
            cfg,
            candidate,
            explicit_fields=explicit_fields,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            service_capacity=service_capacity,
            splits_kv=splits_kv,
            max_splits_kv=max_splits_kv,
            split_kv=split_kv,
        )
    )
    if not supported:
        return None

    has_underfilled_q_grid = any(
        batch_size * num_heads_kv * recipe.mma.q_tiles < service_capacity
        for recipe in supported
    )
    has_large_q_tile_transformed_recipe = (
        cfg.use_nvfp4_kv
        and cfg.use_fp8_pv
        and any(recipe.mma.tile_size_q >= 48 for recipe in supported)
    )
    if has_underfilled_q_grid or has_large_q_tile_transformed_recipe:
        selected = select_grouped_q_launch_candidate(supported)
    else:
        # Outside the qualified transformed recipes above, retain direct-only
        # selection when every legal Q grid fills a wave. Compare complete
        # direct recipes with the existing mainloop-aware score.
        direct = tuple(recipe for recipe in supported if recipe.splits_kv == 1)
        selected = (
            select_grouped_q_direct_wave_candidate(direct)
            if direct
            else select_grouped_q_launch_candidate(supported)
        )
    apply_grouped_q_mma_candidate(cfg, selected.mma)
    if (
        selected.splits_kv == 1
        and selected.base_ctas > service_capacity
        and not (LAUNCH_SELECTION_FIELDS & explicit_fields)
    ):
        # CLC persistence pays for work discovery by reusing one resident CTA
        # wave. Select it only when the direct launch has more than one wave;
        # a single-wave grid has no launch work for persistence to eliminate.
        persistent_probe = deepcopy(cfg)
        persistent_probe.use_persistent_scheduler = True
        try:
            finalize_static_decode_config(
                persistent_probe,
                explicit_fields | {"tile_size_q"},
            )
            validate_profile_support(
                cfg=persistent_probe,
                seq_len_q=seq_len_q,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                split_kv_mode="disabled",
            )
        except ValueError:
            # Some public profiles support the selected direct grouped tile but
            # not its CLC variant. Keep the valid static recipe rather than
            # turning a successful automatic selection into a late rejection.
            pass
        else:
            cfg.use_persistent_scheduler = True
    # Keeps static defaults otherwise canonicalize an implicit tile to 64.
    # Record only these local derived choices so finalization preserves them;
    # the pre-Q snapshot used by the downstream KV selector remains unpinned.
    explicit_fields.add("tile_size_q")
    return selected


def apply_auto_launch_mode(
    cfg: FmhaDecodeConfig,
    *,
    auto_tuner: bool,
    split_kv_mode: str,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    seq_len_kv: int,
    seq_len_q: int,
    split_kv: bool = True,
) -> str:
    """Apply the static/persistent/split-KV launch heuristic when it is allowed."""
    if not should_auto_select_launch_mode(
        cfg,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        seq_len_q=seq_len_q,
    ):
        return split_kv_mode

    mode = select_auto_launch_mode(
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        seq_len_kv=seq_len_kv,
        num_q_tiles=num_q_tiles_for_launch(cfg),
        tile_size_kv=cfg.tile_size_kv,
        split_kv=split_kv,
    )
    if cfg.use_sliding_window_causal and mode == ("gmem_reduction"):
        # Automatic sliding-window split recipes are not qualified. Query
        # storage layout alone does not restrict split selection.
        return split_kv_mode
    if mode not in ("gmem_reduction", "persistent"):
        return split_kv_mode

    # Profiles outside the joint selector can still leave launch mode
    # automatic. Probe the complete derived launch
    # before committing it, so an unsupported Keeps split/persistent recipe
    # falls back to the caller's valid direct profile.
    probe = deepcopy(cfg)
    if mode == "gmem_reduction":
        apply_split_kv_config(
            probe,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            batch_size=batch_size,
            num_heads_kv=num_heads_kv,
            split_kv_mode=mode,
            splits_kv=-1,
            max_splits_kv=None,
            sliding_window_causal=False,
            attention_window_size=0,
        )
    else:
        probe.use_persistent_scheduler = True
    try:
        validate_profile_support(
            cfg=probe,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            split_kv_mode=(mode if mode == "gmem_reduction" else "disabled"),
        )
    except ValueError:
        return split_kv_mode

    if mode == "persistent":
        cfg.use_persistent_scheduler = True
        return split_kv_mode
    if mode == "gmem_reduction":
        return mode
    return split_kv_mode


def apply_grouped_q_mma_candidate(
    cfg: FmhaDecodeConfig,
    candidate: GroupedQMmaCandidate,
) -> None:
    """Apply one internally selected grouped-Q MMA tile to ``cfg``."""
    cfg.use_keeps_mma_ab = candidate.variant == "keeps_mma_ab"
    cfg.tile_size_q = candidate.tile_size_q
    if cfg.use_keeps_mma_ab:
        return
    cfg.tmem_s_cols = candidate.tile_size_q
    cfg.tmem_o_cols = candidate.tile_size_q
    cfg.mma_tile_n_bmm1 = candidate.tile_size_q
    cfg.mma_tile_n_bmm2 = candidate.tile_size_q


def apply_split_kv_config(
    cfg: FmhaDecodeConfig,
    *,
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    split_kv_mode: str,
    splits_kv: int,
    max_splits_kv: int | None,
    sliding_window_causal: bool,
    attention_window_size: int,
    min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
) -> None:
    """Resolve split-KV fanout and config flags for the selected reduction mode."""
    if split_kv_mode == "disabled":
        return

    heuristic_seq_len_kv = effective_seq_len_for_sliding(
        seq_len_kv,
        sliding_window_causal,
        attention_window_size,
        cfg.tile_size_kv,
    )
    selected_splits_kv = select_splits_kv(
        seq_len_kv=heuristic_seq_len_kv,
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        tile_size_kv=cfg.tile_size_kv,
        num_insts_kv=cfg.num_insts_kv,
        num_q_tiles=num_q_tiles_for_launch(cfg),
        requested_splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
        min_loop_iters_per_split=min_loop_iters_per_split,
    )
    if selected_splits_kv <= 1:
        return

    cfg.use_split_kv = True
    cfg.splits_kv = selected_splits_kv
    cfg.max_splits_kv = selected_splits_kv
    cfg.use_persistent_scheduler = False
    if split_kv_mode == "gmem_reduction_with_separate_kernel":
        cfg.use_separate_reduction_kernel = True
    elif split_kv_mode == "cluster_smem_reduction":
        cfg.use_cluster_smem_reduction = True


def compute_runtime_active_splits_kv(
    *,
    valid_k: int,
    tile_size_kv: int,
    num_insts_kv: int,
    configured_splits_kv: int,
) -> int:
    """Host mirror of the device runtime split-prefix calculation."""
    if valid_k < 0:
        raise ValueError("valid_k must be non-negative")
    if tile_size_kv <= 0:
        raise ValueError("tile_size_kv must be positive")
    if num_insts_kv <= 0:
        raise ValueError("num_insts_kv must be positive")
    if configured_splits_kv <= 0:
        raise ValueError("configured_splits_kv must be positive")
    total_kv_tiles = (valid_k + tile_size_kv - 1) // tile_size_kv
    groups_per_split = (total_kv_tiles + configured_splits_kv * num_insts_kv - 1) // (
        configured_splits_kv * num_insts_kv
    )
    local_kv_tiles = max(groups_per_split * num_insts_kv, num_insts_kv)
    return (total_kv_tiles + local_kv_tiles - 1) // local_kv_tiles


def config_with_split_kv_mode(
    cfg: FmhaDecodeConfig, split_kv_mode: str
) -> FmhaDecodeConfig:
    """Return ``cfg`` with exactly one split-KV reduction mode selected."""

    return replace(
        cfg,
        use_cluster_smem_reduction=split_kv_mode == "cluster_smem_reduction",
        use_separate_reduction_kernel=(
            split_kv_mode == "gmem_reduction_with_separate_kernel"
        ),
    )


def enumerate_auto_splits_kv(
    *,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    num_q_tiles: int,
    service_capacity: int,
    split_kv: bool = True,
    requested_splits_kv: int = -1,
    max_splits_kv: int | None = None,
    service_waves: int = 1,
    min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
) -> tuple[int, ...]:
    """Enumerate direct and useful split fanouts for the requested wave budget.

    All one-wave fanouts and the first capacity-crossing fanout participate in
    the empirical score. The latter is important when a partially filled wave
    cannot be completed by any uniform integer fanout.
    A positive requested fanout is pinned, subject to the same KV-work cap.
    The default retains the one-wave policy. Q48+ transformed profiles can
    compare several waves to avoid a mostly empty final wave of a full grid.
    ``min_loop_iters_per_split`` caps the fanout by the KV work per split.
    """
    if not split_kv:
        return (1,)
    if num_q_tiles <= 0:
        raise ValueError("num_q_tiles must be positive")
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    max_by_work = max_splits_kv_by_work(
        seq_len_kv=seq_len_kv,
        tile_size_kv=tile_size_kv,
        num_insts_kv=num_insts_kv,
        max_splits_kv=max_splits_kv,
        min_loop_iters_per_split=min_loop_iters_per_split,
    )
    if requested_splits_kv > 0:
        return (max(1, min(requested_splits_kv, max_by_work)),)
    base_grid = max(1, batch_size * num_heads_kv * num_q_tiles)
    if service_waves <= 0:
        raise ValueError("service_waves must be positive")
    if (base_grid >= service_capacity and service_waves == 1) or max_by_work <= 1:
        return (1,)
    first_full_wave_fanout = (
        service_capacity * service_waves + base_grid - 1
    ) // base_grid
    max_considered = min(
        max_by_work,
        max(first_full_wave_fanout, 2),
    )
    return tuple(range(1, max_considered + 1))


def enumerate_grouped_q_mma_candidates(
    *,
    heads_q_per_kv: int,
    seq_len_q: int,
    swaps_only: bool = False,
    include_nvfp4_tiles: bool = False,
) -> tuple[GroupedQMmaCandidate, ...]:
    """Return grouped-Q candidates without applying launch policy.

    A CTA always owns an integral number of complete Q-head groups.  TileQ may
    leave structural padding rows and the final CTA may own fewer tokens than
    its capacity; the common Q geometry and row masks already represent both
    cases.  This helper deliberately does not inspect dtypes, reduction modes,
    scheduler state, or mutate ``FmhaDecodeConfig``. ``swaps_only`` excludes
    KeepsMmaAb tiles for profiles such as transformed mixed-precision K/V.
    """
    if heads_q_per_kv <= 0:
        raise ValueError("heads_q_per_kv must be positive")
    if seq_len_q <= 0:
        raise ValueError("seq_len_q must be positive")

    candidates = []
    for variant, tile_size_q in GROUPED_Q_MMA_TILES:
        if tile_size_q == 48 and not include_nvfp4_tiles:
            continue
        if swaps_only and variant != "swaps_mma_ab":
            continue
        if tile_size_q < heads_q_per_kv:
            continue
        q_tokens_per_cta = tile_size_q // heads_q_per_kv
        candidates.append(
            GroupedQMmaCandidate(
                variant=variant,
                tile_size_q=tile_size_q,
                q_tokens_per_cta=q_tokens_per_cta,
                q_tiles=(seq_len_q + q_tokens_per_cta - 1) // q_tokens_per_cta,
            )
        )
    return tuple(candidates)


def grouped_q_cost_kv_width(cfg: FmhaDecodeConfig) -> int:
    """Return the cost-table width that scores ``cfg``'s grouped-Q candidates.

    Only 16-bit K and V at head dim 128 take width 16, since only that head
    dim promotes its TileQ64 candidate to KV256. Every other profile takes
    width 8, under which D192 picks its faster Q128 tile.
    """
    if cfg.headdim == 128 and cfg.k_dtype.width == 16 and cfg.v_dtype.width == 16:
        return 16
    return 8


def make_decode_config(
    headdim: int = 128,
    args: object | None = None,
    *,
    seq_len_q: int = 1,
    seq_len_kv: int | None = None,
    batch_size: int | None = None,
    num_heads_q: int | None = None,
    num_heads_kv: int | None = None,
    q_dtype: type | None = None,
    k_dtype: type | None = None,
    v_dtype: type | None = None,
    qkv_dtype: type | None = None,
    kv_dtype: type | None = None,
    o_dtype: type = Float16,
    qkv_layout: str = "contiguousKv",
    num_tokens_per_page: int = 32,
    storage_tokens_per_page: int | None = None,
    split_kv_mode: str = "disabled",
    splits_kv: int = -1,
    max_splits_kv: int | None = None,
    min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
    sliding_window_causal: bool = False,
    attention_window_size: int = 0,
    mask_type: str | None = None,
    use_attention_sinks: bool = False,
    auto_tuner: bool = True,
    split_kv: bool = True,
) -> FmhaDecodeConfig:
    """Build the static decode kernel config and apply auto-selection policy.

    When no launch-shape inputs are supplied, this is the simple static config
    constructor: it reads only ``FmhaDecodeConfig`` fields from ``args`` and
    applies static profile defaults. When launch-shape inputs are supplied, it
    additionally runs the decode kernel-selection workflow below.

    ``q_dtype``, ``k_dtype``, and ``v_dtype`` select independent storage types.
    The legacy ``qkv_dtype`` sets their common default, while ``kv_dtype``
    sets a common K/V default. Mixed-profile scheduling derives from the
    effective dtypes before Q-tile or launch-mode selection.

    Workflow:
    1. Create a default ``FmhaDecodeConfig`` and apply caller-supplied config
       fields directly onto it, remembering which profile fields were explicit.
    2. Fill profile-derived defaults: dtype fields, paged-KV metadata, the
       dense/causal mask, sliding-window flags, attention-sink flags, default
       grouped-Q metadata for fixed or packed launches, and the SMEM-derived
       KV stage count. For a multi-Q paged launch, the Q selector first
       enumerates Swaps8/16/32 and Keeps64/128
       tiles and every useful direct/split recipe through the first
       capacity-crossing fanout.
       Production-valid recipes minimize the empirical TileQ
       mainloop-plus-reduction proxy using their actual Q-grid CTA waves.
       Only a selected Q64 Keeps recipe may then promote a qualified FP16/BF16
       D128 launch from KV128 to KV256; every other automatic Q tile retains
       KV128. The final KV width re-derives launch policy instead of reusing a
       fanout scored for KV128. Explicit policies remain caller-controlled.
       TileQ128 remains automatic over TileQ64 only for staged D256. SQ1 NVFP4
       searches full grids and qualified long-KV SM107 BF16-output grids;
       other SQ1 launches retain KV128, while ratios above 32 select
       the smallest qualified Q64/Q128 Keeps tile. Other profile families use
       exact-width ungrouped Keeps or ungrouped Swaps head bands as a fallback.
    3. Shapes outside that qualified Q/launch selector retain the general launch
       policy: under-filled long-sequence grids may use split-KV GMEM
       reduction, direct grids above one resident wave use persistent
       scheduling, and the rest stay static. Query packing does not disable
       splits; automatic sliding-window splits remain unqualified. An
       unsupported automatic mode falls back to direct.
    4. If split-KV is selected or requested, compute the split fanout from the
       effective KV length, requested split count, max split cap, SM count, and
       ``batch_size * num_heads_kv * q_tiles`` physical grid size.
    5. Choose the reduction mode for an automatic split-KV launch: prefer a
       legal one-wave cluster configuration, otherwise use the standalone GMEM
       reducer with inline reduction as a support fallback. A jointly scored
       recipe keeps its exact fanout; the legacy launch path may search
       downward for a one-wave cluster split.
    6. Validate the final profile combination before returning the config.

    Outside joint grouped-Q selection, attention-sink paths retain their
    existing launch mode. Explicit launch modes are still validated.

    ``split_kv`` is a common planning control for every FMHA kernel family.
    True permits supported automatic split recipes; False restricts selection
    to nonsplit recipes without disabling Q-tile or persistence selection.
    It is independent of fixed/packed Q. Pass False for prefill and True for
    decode. Explicit split modes/fanouts conflict with False. The resulting
    ``cfg.use_split_kv`` describes the selected launch, not this permission.
    """
    if not isinstance(split_kv, bool):
        raise TypeError("split_kv must be a bool")
    if not split_kv and (split_kv_mode != "disabled" or splits_kv > 1):
        raise ValueError("split_kv=False conflicts with an explicit split mode/fanout")
    dtype_keywords_supplied = any(
        dtype is not None for dtype in (q_dtype, k_dtype, v_dtype, qkv_dtype, kv_dtype)
    )
    q_dtype, k_dtype, v_dtype = resolve_decode_dtype_aliases(
        q_dtype=q_dtype,
        k_dtype=k_dtype,
        v_dtype=v_dtype,
        qkv_dtype=qkv_dtype,
        kv_dtype=kv_dtype,
    )
    shape_values = (seq_len_kv, batch_size, num_heads_q, num_heads_kv)
    if all(value is None for value in shape_values):
        if dtype_keywords_supplied:
            args = (
                {
                    "q_dtype": q_dtype,
                    "k_dtype": k_dtype,
                    "v_dtype": v_dtype,
                    "out_dtype": o_dtype,
                },
                args,
            )
        cfg = make_static_decode_config(
            headdim,
            args,
            mask_type=mask_type,
            sliding_window_causal=sliding_window_causal,
        )
        if not split_kv and cfg.use_split_kv:
            raise ValueError("split_kv=False conflicts with use_split_kv=True")
        return cfg
    if any(value is None for value in shape_values):
        raise ValueError(
            "seq_len_kv, batch_size, num_heads_q, and num_heads_kv are all "
            "required for decode kernel selection"
        )

    validate_sliding_window_args(sliding_window_causal, attention_window_size)
    cfg = FmhaDecodeConfig(headdim=headdim)
    explicit_fields = apply_config_source(cfg, args)
    # Policy selection uses the planned lengths; only the compiled program
    # replaces them with runtime operands.
    fixed_seqlen = cfg.fixed_seqlen
    cfg.fixed_seqlen = True
    if not split_kv and (cfg.use_split_kv or cfg.splits_kv > 1):
        raise ValueError("split_kv=False conflicts with an explicit split config")
    splits_kv, max_splits_kv = resolve_explicit_split_controls(
        cfg,
        explicit_fields=explicit_fields,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    apply_mask_type_config(
        cfg,
        source=args,
        mask_type=mask_type,
        sliding_window_causal=(sliding_window_causal or cfg.use_sliding_window_causal),
        explicit_fields=explicit_fields,
    )
    split_kv_mode = validate_split_kv_mode(split_kv_mode)
    # Whether the launch mode was left to the auto-tuner rather than explicitly
    # chosen by the caller. Only auto-derived modes are eligible for the cluster
    # promotion below.
    launch_mode_was_auto = split_kv_mode == "disabled"
    _fanout_was_explicit = splits_kv > 0
    if num_heads_q <= 0 or num_heads_kv <= 0:
        raise ValueError("fmha_decode head counts must be positive")
    if num_heads_q % num_heads_kv != 0:
        raise ValueError("fmha_decode requires num_heads_q divisible by num_heads_kv")
    apply_default_q_grouping(
        cfg,
        explicit_fields=explicit_fields,
        heads_q_per_kv=num_heads_q // num_heads_kv,
    )
    qkv_layout = apply_layout_config(
        cfg,
        qkv_layout=qkv_layout,
        num_tokens_per_page=num_tokens_per_page,
        storage_tokens_per_page=storage_tokens_per_page,
        seq_len_kv=seq_len_kv,
    )
    apply_dtype_config(
        cfg,
        explicit_fields=explicit_fields,
        q_dtype=q_dtype,
        k_dtype=k_dtype,
        v_dtype=v_dtype,
        o_dtype=o_dtype,
    )
    apply_feature_config(
        cfg,
        explicit_fields=explicit_fields,
        seq_len_q=seq_len_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        sliding_window_causal=sliding_window_causal,
        attention_window_size=attention_window_size,
        use_attention_sinks=use_attention_sinks,
    )
    if not cfg.use_variable_seqlens_q and cfg.max_seq_len_q != seq_len_q:
        raise ValueError(
            "fixed-Q max_seq_len_q must equal seq_len_q; use "
            "use_variable_seqlens_q for a runtime-varying Q length"
        )
    # Score Q geometry with candidate KV fanouts on the common KV128 basis.
    # Finalize the KV tile and reduction mode after selecting that joint recipe.
    auto_selection_explicit_fields = set(explicit_fields)
    selected_grouped_q_recipe = apply_auto_grouped_q_mma_config(
        cfg,
        explicit_fields=explicit_fields,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
        split_kv=split_kv,
    )
    if selected_grouped_q_recipe is None:
        selected_wide_keeps = try_apply_default_wide_keeps_config(
            cfg,
            explicit_fields=explicit_fields,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
        )
        if not selected_wide_keeps:
            heads_q_per_kv = num_heads_q // num_heads_kv
            if (
                heads_q_per_kv > 32
                and "groups_tokens_heads_q" not in explicit_fields
                and not (MMA_SELECTION_FIELDS & auto_selection_explicit_fields)
            ):
                # Profiles outside the grouped-Keeps matrix remain valid via
                # the established ungrouped Swaps head bands.
                cfg.groups_tokens_heads_q = False
            apply_swaps_tile_config(
                cfg,
                explicit_fields=explicit_fields,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
            )
    kv_tile_was_promoted = try_apply_auto_kv256_profile(
        cfg,
        q_candidate=(
            selected_grouped_q_recipe.mma
            if selected_grouped_q_recipe is not None
            else None
        ),
        explicit_fields=auto_selection_explicit_fields,
        auto_tuner=auto_tuner,
        split_kv_mode=split_kv_mode,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
    )
    if kv_tile_was_promoted:
        # The selected Q tile remains valid, but its KV128 launch fanout, score,
        # and possible persistent side effect do not. Re-derive launch policy
        # below from the final KV256 work granularity.
        cfg.use_persistent_scheduler = False
        selected_grouped_q_recipe = None

    finalize_static_decode_config(cfg, explicit_fields)
    if not cfg.use_variable_seqlens_q:
        validate_causal_decode_lengths(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            mask_type=cfg.mask_type,
        )

    # Auto launch-mode selection. Only kicks in if the caller has not already
    # opted into a specific mode. The caller's split permission is independent
    # of Q storage; unsupported automatic recipes retain their direct fallback.
    if selected_grouped_q_recipe is not None and not (
        LAUNCH_SELECTION_FIELDS & explicit_fields
    ):
        if selected_grouped_q_recipe.splits_kv > 1:
            split_kv_mode = selected_grouped_q_recipe.split_kv_mode
            splits_kv = selected_grouped_q_recipe.splits_kv
            max_splits_kv = selected_grouped_q_recipe.splits_kv
    elif not (LAUNCH_SELECTION_FIELDS & explicit_fields):
        if split_kv_mode == "disabled" and splits_kv > 1:
            split_kv_mode = "gmem_reduction"
        elif splits_kv <= 0:
            split_kv_mode = apply_auto_launch_mode(
                cfg,
                auto_tuner=auto_tuner,
                split_kv_mode=split_kv_mode,
                batch_size=batch_size,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                seq_len_kv=seq_len_kv,
                seq_len_q=seq_len_q,
                split_kv=split_kv,
            )

    reduction_mode_preconfigured = (
        cfg.use_cluster_smem_reduction or cfg.use_separate_reduction_kernel
    )
    auto_split_kv_selected = (
        auto_tuner
        and launch_mode_was_auto
        and split_kv_mode == "gmem_reduction"
        and not reduction_mode_preconfigured
        and not cfg.use_sliding_window_causal
        and not cfg.use_attention_sinks
    )

    apply_split_kv_config(
        cfg,
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        batch_size=batch_size,
        num_heads_kv=num_heads_kv,
        split_kv_mode=split_kv_mode,
        splits_kv=splits_kv,
        max_splits_kv=max_splits_kv,
        sliding_window_causal=sliding_window_causal,
        attention_window_size=attention_window_size,
        min_loop_iters_per_split=min_loop_iters_per_split,
    )

    if auto_split_kv_selected and cfg.use_split_kv:
        cfg, split_kv_mode = select_auto_split_kv_reduction_mode(
            cfg,
            seq_len_q=seq_len_q,
            batch_size=batch_size,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            preserve_exact_cluster_fanout=(
                _fanout_was_explicit or selected_grouped_q_recipe is not None
            ),
        )

    if (
        selected_grouped_q_recipe is not None
        and is_sm107_compile_target()
        and cfg.headdim == 256
        and cfg.out_dtype == BFloat16
        and cfg.uses_fast_smem_nvfp4_transform
        and (
            selected_grouped_q_recipe.base_ctas
            >= get_max_active_clusters_for_cluster_size(1)
            or selected_grouped_q_recipe.waves > 1
        )
    ):
        # Register-prefetched rows release raw slots before conversion, so
        # three complete K/V tiles can hide TMA latency behind both converters.
        # The existing stream keeps one raw tile for full launch grids.
        set_if_implicit(
            cfg,
            "kv_stages",
            12 if cfg.uses_prefetched_smem_nvfp4_transform else 4,
            explicit_fields,
        )

    validate_mixed_kv_dtype_profile(cfg)
    validate_kv256_static_config(cfg)
    finalize_warp_roles(cfg)

    validate_profile_support(
        cfg=cfg,
        seq_len_q=seq_len_q,
        num_heads_q=num_heads_q,
        num_heads_kv=num_heads_kv,
        split_kv_mode=split_kv_mode,
    )
    cfg.fixed_seqlen = fixed_seqlen
    return cfg


def make_grouped_q_launch_candidate(
    candidate: GroupedQMmaCandidate,
    *,
    splits_kv: int,
    seq_len_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    batch_size: int,
    num_heads_kv: int,
    service_capacity: int,
    cost_kv_width: int,
) -> GroupedQLaunchCandidate:
    """Build one empirical grouped-Q launch-cost record.

    The model follows the measured GQA-generation cost structure:

    ``(mainloop_factor[TileQ] * seq_len_per_cta_kv +``
    `` reduction_factor[TileQ] * 128 * splits_kv) * waves``.

    This is a small deterministic default heuristic. Both factors come from
    the tables of ``cost_kv_width`` (``grouped_q_cost_kv_width``); the
    production profile validator still determines which actual-dtype recipes
    are legal.
    """
    if splits_kv <= 0:
        raise ValueError("splits_kv must be positive")
    if seq_len_kv <= 0:
        raise ValueError("seq_len_kv must be positive")
    if tile_size_kv <= 0:
        raise ValueError("tile_size_kv must be positive")
    if num_insts_kv <= 0:
        raise ValueError("num_insts_kv must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if num_heads_kv <= 0:
        raise ValueError("num_heads_kv must be positive")
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    kv_tiles = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
    seq_len_per_cta_kv = ((kv_tiles + splits_kv - 1) // splits_kv) * tile_size_kv
    base_ctas = batch_size * num_heads_kv * candidate.q_tiles
    launched_ctas = base_ctas * splits_kv
    waves = (launched_ctas + service_capacity - 1) // service_capacity
    modeled_time = (
        GROUPED_Q_MAINLOOP_COST[cost_kv_width][candidate.tile_size_q]
        * seq_len_per_cta_kv
        + GROUPED_Q_REDUCTION_COST[cost_kv_width][candidate.tile_size_q]
        * GROUPED_Q_REDUCTION_SEQ_LEN_FACTOR
        * splits_kv
    ) * waves
    return GroupedQLaunchCandidate(
        mma=candidate,
        split_kv_mode=("gmem_reduction" if splits_kv > 1 else "disabled"),
        splits_kv=splits_kv,
        base_ctas=base_ctas,
        launched_ctas=launched_ctas,
        seq_len_per_cta_kv=seq_len_per_cta_kv,
        waves=waves,
        modeled_time=modeled_time,
    )


def max_splits_kv_by_work(
    *,
    seq_len_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    max_splits_kv: int | None = None,
    min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
) -> int:
    """Return the fanout cap that retains useful KV work per CTA."""
    if min_loop_iters_per_split <= 0:
        raise ValueError("min_loop_iters_per_split must be positive")
    tile_size_per_cta_kv = tile_size_kv * num_insts_kv * min_loop_iters_per_split
    max_by_seq = max(
        1,
        (seq_len_kv + tile_size_per_cta_kv - 1) // tile_size_per_cta_kv,
    )
    if max_splits_kv is not None and max_splits_kv > 0:
        max_by_seq = min(max_by_seq, max_splits_kv)
    return max_by_seq


def num_q_tiles_for_launch(cfg: FmhaDecodeConfig) -> int:
    """Return the physical Q-CTA multiplicity for launch occupancy."""
    q_geometry = make_q_tile_geometry(
        rows_per_cta=cfg.tile_size_q,
        heads_q_per_kv=cfg.heads_q_per_kv,
        groups_tokens_heads_q=cfg.groups_tokens_heads_q,
    )
    return max(q_geometry.num_q_ctas(cfg.max_seq_len_q), 1)


def resolve_grouped_q_launch_candidates(
    cfg: FmhaDecodeConfig,
    candidate: GroupedQMmaCandidate,
    *,
    explicit_fields: set[str],
    seq_len_q: int,
    seq_len_kv: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    service_capacity: int,
    splits_kv: int,
    max_splits_kv: int | None,
    split_kv: bool = True,
) -> tuple[GroupedQLaunchCandidate, ...]:
    """Resolve one TileQ with Q-grid-aware direct and split recipes.

    The under-filled Q grid always participates in split selection. If the
    GMEM split profile is unsupported, the legal direct recipe remains in the
    candidate set.
    """
    probe = deepcopy(cfg)
    # TileQ is always scored on the common KV128 baseline. This includes an
    # explicitly requested final KV width: explicit materialization remains a
    # downstream constraint and must not feed back into the Q winner.
    probe.tile_size_kv = 128
    apply_grouped_q_mma_candidate(probe, candidate)
    cost_tile_size_kv = 128
    if (
        candidate.variant == "keeps_mma_ab"
        and candidate.tile_size_q == 64
        and probe.headdim == 128
        and probe.q_dtype == BFloat16
        and probe.q_dtype == probe.k_dtype == probe.v_dtype == probe.out_dtype
    ):
        # TileQ selection uses a common KV128 cost basis and is independent of
        # the later KV-tile decision. BF16 Q64 has no final KV128 profile, but
        # its Q geometry and modeled cost are identical to the qualified FP16
        # logical candidate. The KV selector materializes and validates the
        # actual BF16 profile only after the Q winner is known.
        if cfg.tile_size_kv != 256 and not try_apply_auto_kv256_profile(
            probe,
            q_candidate=candidate,
            explicit_fields=explicit_fields,
            auto_tuner=True,
            split_kv_mode="disabled",
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            splits_kv=splits_kv,
            max_splits_kv=max_splits_kv,
        ):
            return ()
        probe.tile_size_kv = 128
        probe.q_dtype = Float16
        probe.k_dtype = Float16
        probe.v_dtype = Float16
        probe.out_dtype = Float16
    try:
        finalize_static_decode_config(
            probe,
            explicit_fields | {"tile_size_q"},
        )
        # Explicit MMA fields constrain candidates instead of disabling the
        # search over the remaining geometry and launch choices.
        if any(
            getattr(probe, field) != getattr(cfg, field)
            for field in MMA_SELECTION_FIELDS & explicit_fields
        ):
            return ()
        if probe.use_keeps_mma_ab and not probe.supports_auto_grouped_keeps:
            return ()
        cost_num_insts_kv = probe.num_insts_kv
        validate_profile_support(
            cfg=probe,
            seq_len_q=seq_len_q,
            num_heads_q=num_heads_q,
            num_heads_kv=num_heads_kv,
            split_kv_mode="disabled",
        )
    except ValueError:
        return ()
    cost_kv_width = grouped_q_cost_kv_width(probe)
    fills_wave = batch_size * num_heads_kv * candidate.q_tiles >= service_capacity
    # Qualified NVFP4 profiles may split to reduce a partial final wave.
    # Other storage precisions retain their established candidate scope.
    transformed_multi_wave = probe.use_nvfp4_kv and probe.use_fp8_pv
    sm107_smem_transform = (
        is_sm107_compile_target()
        and probe.headdim == 256
        and probe.out_dtype == BFloat16
        and probe.uses_fast_smem_nvfp4_transform
    )
    multi_wave = transformed_multi_wave and (candidate.tile_size_q >= 48 or fills_wave)
    min_loop_iters_per_split = MIN_LOOP_ITERS_PER_SPLIT
    if transformed_multi_wave and candidate.tile_size_q < 48 and fills_wave:
        # Narrow transformed tiles split a full grid only to trim its final
        # wave; short splits cannot repay the extra waves.
        min_loop_iters_per_split = -(
            -FULL_GRID_SPLIT_MIN_TOKENS_PER_CTA
            // (cost_tile_size_kv * cost_num_insts_kv)
        )
    if probe.use_keeps_mma_ab:
        # A lone resident CTA cannot hide pipeline fills behind another CTA.
        # Keep underfilled grids within one wave unless the native SM107
        # BF16-output converter qualifies for long extra waves. D512 NVFP4
        # Keeps Q64 scores the same extra waves as Swaps Q64, so a long
        # max_kv_len plan keeps Keeps instead of falling back to Swaps.
        multi_wave = multi_wave and (
            fills_wave or sm107_smem_transform or probe.headdim == 512
        )

    split_candidates = (
        (cfg.splits_kv if cfg.use_split_kv else 1,)
        if LAUNCH_SELECTION_FIELDS & explicit_fields
        else enumerate_auto_splits_kv(
            seq_len_kv=seq_len_kv,
            batch_size=batch_size,
            num_heads_kv=num_heads_kv,
            tile_size_kv=cost_tile_size_kv,
            num_insts_kv=cost_num_insts_kv,
            num_q_tiles=candidate.q_tiles,
            service_capacity=service_capacity,
            split_kv=split_kv,
            requested_splits_kv=splits_kv,
            max_splits_kv=max_splits_kv,
            service_waves=8 if multi_wave else 1,
            min_loop_iters_per_split=min_loop_iters_per_split,
        )
    )
    recipes = []
    for candidate_splits_kv in split_candidates:
        if (
            sm107_smem_transform
            and splits_kv <= 0
            and not (LAUNCH_SELECTION_FIELDS & explicit_fields)
            and not fills_wave
            and batch_size * num_heads_kv * candidate.q_tiles * candidate_splits_kv
            > service_capacity
            and (seq_len_kv + candidate_splits_kv - 1) // candidate_splits_kv
            < FULL_GRID_SPLIT_MIN_TOKENS_PER_CTA
        ):
            # Extra waves must retain enough KV work to amortize each
            # converter's startup. Keep ordinary one-wave fanouts eligible.
            continue
        candidate_reduction_mode = "gmem_reduction"
        if candidate_splits_kv > 1:
            split_probe = deepcopy(probe)
            split_probe.use_split_kv = True
            split_probe.splits_kv = candidate_splits_kv
            split_probe.max_splits_kv = candidate_splits_kv
            if (
                not probe.use_q_token_kv_block_sparse_route
                and batch_size * num_heads_kv * candidate.q_tiles > 2 * service_capacity
            ):
                # Sparse routes own their slice; shared-workspace fused
                # counters reserve two device waves.
                # Larger grids must qualify a counter-free reducer before
                # their split recipe can participate in cost selection.
                split_probe.use_separate_reduction_kernel = True
                candidate_reduction_mode = "gmem_reduction_with_separate_kernel"
            try:
                validate_profile_support(
                    cfg=split_probe,
                    seq_len_q=seq_len_q,
                    num_heads_q=num_heads_q,
                    num_heads_kv=num_heads_kv,
                    split_kv_mode=candidate_reduction_mode,
                )
            except ValueError:
                continue
        recipe = make_grouped_q_launch_candidate(
            candidate,
            splits_kv=candidate_splits_kv,
            seq_len_kv=seq_len_kv,
            tile_size_kv=cost_tile_size_kv,
            num_insts_kv=cost_num_insts_kv,
            batch_size=batch_size,
            num_heads_kv=num_heads_kv,
            service_capacity=service_capacity,
            cost_kv_width=cost_kv_width,
        )
        if candidate_splits_kv > 1:
            recipe = replace(recipe, split_kv_mode=candidate_reduction_mode)
        profile_cost = (
            GROUPED_Q_TRANSFORM_MAINLOOP_COST
            if candidate.variant == "swaps_mma_ab"
            else GROUPED_Q_KEEPS_TRANSFORM_MAINLOOP_COST
        ).get((probe.headdim, candidate.tile_size_q))
        if (
            is_sm107_compile_target()
            and probe.headdim == 256
            and probe.use_fp8_q
            and probe.use_nvfp4_kv
            and candidate.variant == "swaps_mma_ab"
        ):
            # SM107 scales unpacked FP4 through f16x2 rather than QMUL4.
            # The empirical D256 factor accounts for the TMEM converter's
            # repacking and scaling instructions. Keeps consumes dense pairs
            # with two converters, so it retains its own cost.
            # The refreshed factor is qualified for BF16 publication;
            # other output types retain their established policy calibration.
            profile_cost = (2.0 if probe.out_dtype == BFloat16 else 1.8) * (
                profile_cost
                if profile_cost is not None
                else GROUPED_Q_MAINLOOP_COST[cost_kv_width][candidate.tile_size_q]
            )
        if profile_cost is not None and probe.use_fp8_q and probe.use_nvfp4_kv:
            recipe = replace(
                recipe,
                modeled_time=recipe.modeled_time
                + (
                    profile_cost
                    - GROUPED_Q_MAINLOOP_COST[cost_kv_width][candidate.tile_size_q]
                )
                * recipe.seq_len_per_cta_kv
                * recipe.waves,
            )
        recipes.append(recipe)
    return tuple(recipes)


def select_auto_launch_mode(
    *,
    batch_size: int,
    num_heads_kv: int,
    seq_len_kv: int,
    num_q_tiles: int = 1,
    tile_size_kv: int = AUTO_LAUNCH_TILE_SIZE_KV,
    persistent_min_waves: int = 1,
    persistent_min_tiles_per_cta: int = 1,
    split_kv: bool = True,
    service_capacity: int | None = None,
) -> str:
    """Pick the launch mode that best matches the kernel's parallelism budget.

    The kernel can run in three launch modes, each suited to a different
    occupancy regime.

    Returns one of:

      ``"gmem_reduction"``
          Split the K/V sequence across several CTAs (Flash-Decoding GMEM
          reduction). Chosen when the static grid sits under one SM wave
          (``waves < 1``) *and* each CTA has at least 2,048 padded K/V tokens
          to dwarf the GMEM reduction overhead. Legal split fanout is still
          bounded independently by the minimum loop iterations. At ``b=1``
          there are only ``num_heads_kv`` CTAs in the static grid, which
          fills a few percent of an SM-rich device; splitting K/V across
          tens of CTAs unlocks the remaining bandwidth.

      ``"persistent"``
          Switch to the CLC dynamic persistent scheduler whenever the direct
          launch contains more than ``persistent_min_waves`` resident CTA
          waves and at least ``persistent_min_tiles_per_cta`` K/V tiles per
          task. Persistence has no launch work to eliminate within one wave;
          callers with heavier per-task scheduling may require more work to
          amortize that overhead.

      ``"static"``
          Everything else. The default static grid is a good fit when the
          launch already saturates the device with substantial per-CTA
          work.
    """
    if seq_len_kv <= 0 or batch_size <= 0 or num_heads_kv <= 0 or num_q_tiles <= 0:
        return "static"
    sm_count = service_capacity
    if sm_count is None:
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
    sm_count = FALLBACK_SM_COUNT_B200 if sm_count <= 0 else sm_count
    ctas = batch_size * num_heads_kv * num_q_tiles
    waves = ctas / sm_count
    tiles_per_cta = (seq_len_kv + tile_size_kv - 1) // tile_size_kv
    kv_tokens_per_cta = tiles_per_cta * tile_size_kv
    if split_kv and waves < 1 and kv_tokens_per_cta >= SPLIT_KV_MIN_TOKENS_PER_CTA:
        return "gmem_reduction"
    if (
        ctas > persistent_min_waves * sm_count
        and tiles_per_cta >= persistent_min_tiles_per_cta
    ):
        return "persistent"
    return "static"


def select_auto_split_kv_reduction_mode(
    cfg: FmhaDecodeConfig,
    *,
    seq_len_q: int,
    batch_size: int,
    num_heads_q: int,
    num_heads_kv: int,
    preserve_exact_cluster_fanout: bool,
) -> tuple[FmhaDecodeConfig, str]:
    """Select the first supported reduction mode from the shared policy.

    FlashInfer retains its general one-wave cluster search for legacy auto-derived
    fanouts. Jointly scored recipes and caller-provided fanouts remain exact,
    because changing their split count after selection would invalidate the
    TileQ/split comparison or the caller's request.
    """

    modes = select_split_kv_modes(
        family="fmha_decode",
        topology="1cta",
        tile_size_q=cfg.tile_size_q,
        head_dim=cfg.headdim,
        head_dim_per_cta_v=None,
        split_kv=cfg.splits_kv,
        available_modes=AUTO_SPLIT_KV_REDUCTION_MODES,
    )
    for split_kv_mode in modes:
        trial = config_with_split_kv_mode(cfg, split_kv_mode)
        if split_kv_mode == "cluster_smem_reduction":
            if not trial.can_use_cluster_smem_reduction:
                continue
            q_geometry = make_q_tile_geometry(
                rows_per_cta=trial.tile_size_q,
                heads_q_per_kv=num_heads_q // num_heads_kv,
                groups_tokens_heads_q=trial.groups_tokens_heads_q,
            )
            num_launched_clusters = (
                q_geometry.num_q_ctas(trial.max_seq_len_q) * num_heads_kv * batch_size
            )
            cluster_split = select_one_wave_cluster_split(
                initial_splits_kv=trial.splits_kv,
                minimum_splits_kv=(
                    trial.splits_kv if preserve_exact_cluster_fanout else 2
                ),
                num_launched_clusters=num_launched_clusters,
                tile_size_q=trial.tile_size_q,
                headdim=trial.headdim,
                correction_num_warps=trial.correction_num_warps,
            )
            if cluster_split is None:
                continue
            trial.splits_kv = cluster_split
            trial.max_splits_kv = cluster_split
        try:
            validate_profile_support(
                cfg=trial,
                seq_len_q=seq_len_q,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                split_kv_mode=split_kv_mode,
            )
        except ValueError:
            continue
        return trial, split_kv_mode

    # Inline GMEM was the prior automatic behavior. Final validation preserves
    # its canonical error if the base profile itself is unsupported.
    return config_with_split_kv_mode(cfg, "gmem_reduction"), "gmem_reduction"


def select_grouped_q_direct_wave_candidate(
    candidates: Sequence[GroupedQLaunchCandidate],
) -> GroupedQLaunchCandidate:
    """Select a direct recipe using the same mainloop-aware launch score."""
    direct = tuple(recipe for recipe in candidates if recipe.splits_kv == 1)
    if not direct:
        raise ValueError("at least one direct grouped-Q candidate is required")
    return select_grouped_q_launch_candidate(direct)


def select_grouped_q_launch_candidate(
    candidates: Sequence[GroupedQLaunchCandidate],
) -> GroupedQLaunchCandidate:
    """Select the lowest-cost legal grouped-Q launch recipe.

    The score compares actual Q-grid waves, so a wider tile can win only when
    it removes enough repeated KV mainloop work to repay its higher per-tile
    cost.  This is important for partial final tiles: TileQ128 and TileQ64 can
    have different Q-grid multiplicities even when both are legal.
    """
    if not candidates:
        raise ValueError("at least one grouped-Q launch candidate is required")

    return min(
        candidates,
        key=lambda recipe: (
            recipe.modeled_time,
            recipe.waves,
            recipe.splits_kv,
            -recipe.mma.tile_size_q,
        ),
    )


def select_one_wave_cluster_split(
    *,
    initial_splits_kv: int,
    minimum_splits_kv: int = 2,
    num_launched_clusters: int,
    tile_size_q: int,
    headdim: int,
    correction_num_warps: int = 4,
) -> int | None:
    """Return the largest legal cluster split count whose clusters fit one wave.

    The ordinary split heuristic maximizes mainloop parallelism without
    considering cluster residency.  If that cluster size does not fit the
    entire logical grid concurrently, search smaller cluster sizes instead of
    falling back immediately to GMEM reduction.  Every candidate goes through
    the same cluster-size and leader-DSMEM gates; no problem-shape whitelist is
    involved. ``minimum_splits_kv`` can pin an explicit fanout while retaining
    the same eligibility checks.
    """
    if initial_splits_kv < 2 or num_launched_clusters <= 0:
        return None
    lower_bound = max(minimum_splits_kv, 2)
    if lower_bound > initial_splits_kv:
        return None
    for candidate in range(initial_splits_kv, lower_bound - 1, -1):
        cluster_reason = cluster_smem_reduction_unsupported_reason(
            max_splits_kv=candidate,
            splits_kv=candidate,
            tile_size_q=tile_size_q,
            headdim=headdim,
            correction_num_warps=correction_num_warps,
        )
        if cluster_reason:
            continue
        max_active_clusters = get_max_active_clusters_for_cluster_size(candidate)
        if max_active_clusters > 0 and num_launched_clusters <= max_active_clusters:
            return candidate
    return None


def select_splits_kv(
    *,
    seq_len_kv: int,
    batch_size: int,
    num_heads_kv: int,
    tile_size_kv: int,
    num_insts_kv: int,
    num_q_tiles: int = 1,
    service_capacity: int | None = None,
    requested_splits_kv: int = -1,
    max_splits_kv: int | None = None,
    min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
    split_kv: bool = True,
) -> int:
    """
    Select a Q-grid-aware split-KV fanout.

    Every legal TileQ is considered with its actual number of Q CTAs.  The
    automatic fanout fills otherwise idle cluster-size-one service slots while
    retaining at least ``min_loop_iters_per_split`` KV iterations per CTA.
    Positive caller fanouts remain pinned subject only to the KV-work cap. The
    default preserves the dense-decode policy; specialized bounded-work routes
    may choose a lower positive iteration floor explicitly.
    """
    if not split_kv:
        return 1
    if num_q_tiles <= 0:
        raise ValueError("num_q_tiles must be positive")
    max_by_seq = max_splits_kv_by_work(
        seq_len_kv=seq_len_kv,
        tile_size_kv=tile_size_kv,
        num_insts_kv=num_insts_kv,
        max_splits_kv=max_splits_kv,
        min_loop_iters_per_split=min_loop_iters_per_split,
    )
    if requested_splits_kv > 0:
        return max(1, min(max_by_seq, requested_splits_kv))

    if service_capacity is None:
        hardware_info = utils.HardwareInfo()
        service_capacity = hardware_info.get_device_multiprocessor_count()
        # B200 fallback when the runtime SM query is unavailable.
        service_capacity = (
            FALLBACK_SM_COUNT_B200 if service_capacity <= 0 else service_capacity
        )
    if service_capacity <= 0:
        raise ValueError("service_capacity must be positive")
    base_grid = max(1, batch_size * num_heads_kv * num_q_tiles)
    return max(1, min(max_by_seq, max(service_capacity // base_grid, 1)))


def should_auto_select_launch_mode(
    cfg: FmhaDecodeConfig,
    *,
    auto_tuner: bool,
    split_kv_mode: str,
    seq_len_q: int,
) -> bool:
    """Return whether automatic launch-mode selection is allowed for this shape."""
    single_query = seq_len_q == 1
    grouped_query = cfg.groups_tokens_heads_q and seq_len_q > 1
    return (
        auto_tuner
        and split_kv_mode == "disabled"
        and not cfg.use_persistent_scheduler
        and not cfg.use_attention_sinks
        and (
            single_query
            or grouped_query
            or cfg.use_variable_seqlens_q
            or cfg.use_sliding_window_causal
        )
    )


def try_apply_auto_kv256_profile(
    cfg: FmhaDecodeConfig,
    *,
    q_candidate: GroupedQMmaCandidate | None,
    explicit_fields: set[str],
    auto_tuner: bool,
    split_kv_mode: str,
    num_heads_q: int,
    num_heads_kv: int,
    splits_kv: int,
    max_splits_kv: int | None,
) -> bool:
    """Promote KV128 after the Q cost model selects an exact Q64 Keeps tile.

    Every other automatic Q result retains the default KV128 tile. Explicit
    MMA, KV-tile, or launch policies remain caller-controlled. Device
    compatibility is enforced by the public PrimTS wrapper, alongside the
    other decode profiles, rather than duplicated in config selection.
    """
    selection_is_unpinned = (
        auto_tuner
        and split_kv_mode == "disabled"
        and splits_kv == -1
        and max_splits_kv is None
        and not (MMA_SELECTION_FIELDS & explicit_fields)
        and not (LAUNCH_SELECTION_FIELDS & explicit_fields)
        and "tile_size_kv" not in explicit_fields
    )
    profile_is_eligible = (
        selection_is_unpinned
        and q_candidate is not None
        and q_candidate.variant == "keeps_mma_ab"
        and q_candidate.tile_size_q == 64
        and cfg.use_paged_kv
        and cfg.groups_tokens_heads_q
        and cfg.headdim == 128
        and cfg.q_dtype in (Float16, BFloat16)
        and cfg.q_dtype == cfg.k_dtype == cfg.v_dtype == cfg.out_dtype
        and num_heads_q // num_heads_kv <= 64
    )
    if not profile_is_eligible:
        return False

    cfg.tile_size_kv = 256
    return True


def try_apply_default_wide_keeps_config(
    cfg: FmhaDecodeConfig,
    *,
    explicit_fields: set[str],
    seq_len_q: int,
    num_heads_q: int,
    num_heads_kv: int,
) -> bool:
    """Select Q64/Q128 Keeps for an unpinned head ratio above 32.

    Prefer a grouped tile so a partial GQA group can occupy the next supported
    MMA width. If that profile family is not qualified, an exact 64- or
    128-head group may still use the established ungrouped Keeps path. The
    caller falls back to ungrouped Q16 Swaps head bands for other profiles.
    """

    heads_q_per_kv = num_heads_q // num_heads_kv
    if (
        heads_q_per_kv <= 32
        or heads_q_per_kv > 128
        or bool(MMA_SELECTION_FIELDS & explicit_fields)
    ):
        return False

    tile_size_q = 64 if heads_q_per_kv <= 64 else 128
    grouping_was_explicit = "groups_tokens_heads_q" in explicit_fields
    grouping_candidates = (cfg.groups_tokens_heads_q,)
    if (
        not grouping_was_explicit
        and cfg.groups_tokens_heads_q
        and heads_q_per_kv == tile_size_q
    ):
        grouping_candidates += (False,)

    for groups_tokens_heads_q in grouping_candidates:
        probe = deepcopy(cfg)
        probe.groups_tokens_heads_q = groups_tokens_heads_q
        probe.use_keeps_mma_ab = True
        probe.tile_size_q = tile_size_q
        try:
            finalize_static_decode_config(
                probe,
                explicit_fields | {"tile_size_q"},
            )
            if probe.groups_tokens_heads_q and (
                not probe.supports_auto_grouped_keeps
                # D512 grouped Keeps is qualified only at eight Q heads per KV
                # head; wider ratios keep the ungrouped Q16 Swaps fallback.
                or probe.headdim == 512
            ):
                continue
            validate_profile_support(
                cfg=probe,
                seq_len_q=seq_len_q,
                num_heads_q=num_heads_q,
                num_heads_kv=num_heads_kv,
                split_kv_mode="disabled",
            )
        except ValueError:
            continue

        cfg.groups_tokens_heads_q = groups_tokens_heads_q
        cfg.use_keeps_mma_ab = True
        cfg.tile_size_q = tile_size_q
        # Keeps finalization otherwise canonicalizes an implicit tile to Q64.
        explicit_fields.add("tile_size_q")
        return True

    return False
