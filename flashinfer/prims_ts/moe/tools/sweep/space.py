# Copyright (c) 2026 by FlashInfer team.
#
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

"""Candidate generation and legality for the Prims-TS MoE sweep."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, fields, replace
from functools import cached_property
import hashlib
import json
import random
from typing import Any, Iterable, Mapping, Sequence

from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
    needs_extended_tma_multicast,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    compute_max_num_ctas_in_token_dim_for_moe,
    compute_warp_layout,
    make_config,
    validate_launch_grid,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.config_types import RouteImpl, SfLayout
from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _runtime_config
from flashinfer.prims_ts.moe.compile_cache import stable_config_hash
from flashinfer.prims_ts.moe.config_mapper import (
    PrimsTsConfigSpec,
    _JsonBatchedGemmConfig,
    _ensure_config_task_manager_buildable,
    _json_config_by_global_index,
    _json_config_kwargs,
    _json_route_value,
    _moe_json_passing_indices,
    _task_manager_smem_bytes,
)
from flashinfer.prims_ts.moe.tensor_adapter import _validate_swap_ab_cluster_span
from flashinfer.tllm_enums import WeightLayout

from .defaults import (
    DEFAULT_MAX_ACCUMULATOR_STAGES,
    DEFAULT_MAX_C_SMEM_STAGES,
    DEFAULT_MAX_INPUT_STAGES,
    DEFAULT_MAX_SF_RING_RATIO,
    DEFAULT_MAX_WORKID_STAGES,
)


_SMEM_STAGE_FIELDS = (
    "num_stages_a",
    "num_stages_b",
    "num_stages_smem_sfa",
    "num_stages_smem_sfb",
)
_TMEM_STAGE_FIELDS = (
    "num_stages_tmem_acc",
    "num_stages_tmem_sfa",
    "num_stages_tmem_sfb",
)
_C_SMEM_STAGE_FIELD = "num_stages_c_smem"
_WORKID_STAGE_FIELD = "num_stages_workid"
_MIN_TWO_CTA_MMA_N = 16
_ALL_BOUNDED_STAGE_FIELDS = (
    *_SMEM_STAGE_FIELDS,
    *_TMEM_STAGE_FIELDS,
    _C_SMEM_STAGE_FIELD,
    _WORKID_STAGE_FIELD,
)


@dataclass(frozen=True)
class StageLimits:
    """Loose search bounds for physically distinct pipeline resources."""

    inputs: int = DEFAULT_MAX_INPUT_STAGES
    accumulator: int = DEFAULT_MAX_ACCUMULATOR_STAGES
    c_smem: int = DEFAULT_MAX_C_SMEM_STAGES
    workid: int = DEFAULT_MAX_WORKID_STAGES

    def __post_init__(self) -> None:
        if min(self.inputs, self.accumulator, self.c_smem, self.workid) < 1:
            raise ValueError("stage limits must be positive")

    def limit_for(self, field: str) -> int:
        if field in {
            "num_stages_a",
            "num_stages_b",
            "num_stages_smem_sfa",
            "num_stages_smem_sfb",
            "num_stages_tmem_sfa",
            "num_stages_tmem_sfb",
        }:
            return self.inputs
        if field == "num_stages_tmem_acc":
            return self.accumulator
        if field == _C_SMEM_STAGE_FIELD:
            return self.c_smem
        if field == _WORKID_STAGE_FIELD:
            return self.workid
        raise ValueError(f"unknown bounded stage field {field!r}")


DEFAULT_STAGE_LIMITS = StageLimits()
# The validation TaskManager includes the kernel's standalone SMEM
# infrastructure, so its total matches the compiler's capacity check.
SWEEP_SMEM_HEADROOM_BYTES = 0
MOVE_KINDS = (
    "stages_all",
    "stages_one",
    "stages_trade",
    "tmem_stages",
    "tile_k",
    "tile_k_stages",
    "epilogue",
    "warps",
    "route_warps",
    "register_split",
    "flags",
    "skeleton",
    "cluster",
    "large_carveout",
    "crossover",
)
STAGE_MOVE_KINDS = frozenset(
    {"stages_all", "stages_one", "stages_trade", "tmem_stages"}
)
ROUTING_MOVE_KINDS = frozenset({"warps", "route_warps"})
STRUCTURAL_OPTION_FIELDS = frozenset(
    {
        "route_act",
        "route_sfs_act",
        "fuse_operand_sf_loads",
        "fuse_sf_copy_to_mma",
        "use_max_tmem_overlap",
        "umma_cta_count",
        "cluster_dim_x",
        "cluster_dim_y",
        "fallback_cluster_dim_x",
        "fallback_cluster_dim_y",
        "tile_scheduler",
        "cta_raster_order",
    }
)
CROSSOVER_DONOR_FIELD_GROUPS = (
    (
        "route_act",
        "route_sfs_act",
    ),
    (
        "fuse_operand_sf_loads",
        "fuse_sf_copy_to_mma",
        "use_max_tmem_overlap",
    ),
    (
        "umma_cta_count",
        "cluster_dim_x",
        "cluster_dim_y",
        "fallback_cluster_dim_x",
        "fallback_cluster_dim_y",
    ),
    (
        "tile_scheduler",
        "cta_raster_order",
    ),
)
CROSSOVER_DONOR_TUNING_FIELDS = (
    "tile_k",
    "epi_tile_n",
    "num_stages_c_smem",
    "use_tma_store",
    *_SMEM_STAGE_FIELDS,
    *_TMEM_STAGE_FIELDS,
    "tmem_ldst_max_num_regs",
    "epilogue_regs",
    "mma_regs",
    "load_regs",
)
CROSSOVER_OPTION_FIELDS = STRUCTURAL_OPTION_FIELDS | frozenset(
    CROSSOVER_DONOR_TUNING_FIELDS
)


def _tile256_overlap_tile_ks(cfg: Any) -> tuple[int, ...]:
    """Mirror the tile-K choices supported by max-TMEM-overlap kernels."""
    return (128, 256) if cfg.is_mx_mma else (256,)


def _unsupported_move_reason(cfg: Any, move_kind: str) -> str | None:
    """Return why a kernel-supported parent has no realization of a move."""
    if move_kind == "cluster" and cfg.mma_n < _MIN_TWO_CTA_MMA_N:
        return (
            "cluster requires 2-CTA MMA with "
            f"mma_n >= {_MIN_TWO_CTA_MMA_N}, got mma_n={cfg.mma_n}"
        )
    if (
        move_kind in {"tile_k", "tile_k_stages"}
        and cfg.use_tile256_tmem_overlap
        and _tile256_overlap_tile_ks(cfg) == (cfg.tile_k,)
    ):
        return (
            f"{move_kind} has no supported tile_k alternative for the "
            f"tile-256 TMEM-overlap kernel (tile_k={cfg.tile_k})"
        )
    if move_kind == "tmem_stages" and cfg.use_tile256_tmem_overlap:
        return (
            "tmem_stages cannot vary the tile-256 TMEM-overlap kernel's "
            "fixed accumulator and fused scale-factor staging"
        )
    return None


def applicable_move_kinds(
    context: ConfigContext,
    move_kinds: Iterable[str] = MOVE_KINDS,
    *,
    parent: Candidate | None = None,
) -> tuple[str, ...]:
    """Return moves that can change a candidate in this sweep context."""
    probe = (
        make_config(**candidate_kwargs(parent, context)) if parent is not None else None
    )
    return tuple(
        move
        for move in move_kinds
        if not (move in ROUTING_MOVE_KINDS and context.fc != "fc1")
        and not (move == "large_carveout" and context.sm_arch != 107)
        and not (
            probe is not None and _unsupported_move_reason(probe, move) is not None
        )
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ConfigContext:
    """Runtime matching information for one independently swept FC."""

    sm_arch: int
    fc: str
    activation_type: int
    dtype_a: int
    dtype_b: int
    dtype_c: int
    top_k: int
    num_local_experts: int
    has_bias: bool = False
    use_deepseek_fp8: bool = False
    use_per_token_sf_a: bool | None = None
    use_per_token_sf_b: bool | None = None
    per_token_sf_dtype: int | None = None
    has_gemm1_alpha: bool = False
    has_gemm1_beta: bool = False
    has_gemm1_clamp_limit: bool = False
    gemm1_alpha_value: float | None = None
    gemm1_beta_value: float | None = None
    gemm1_clamp_limit_value: float | None = None
    weight_layout: int = int(WeightLayout.MajorK)
    dtype_label: str = "Prims-TS"
    out_hiddens: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "out_hiddens",
            tuple(int(value) for value in self.out_hiddens),
        )
        if self.fc not in ("fc1", "fc2"):
            raise ValueError(f"fc must be fc1 or fc2, got {self.fc!r}")
        if self.sm_arch not in (100, 103, 107):
            raise ValueError(f"unsupported sm_arch={self.sm_arch}")
        if self.top_k <= 0 or self.num_local_experts < self.top_k:
            raise ValueError("num_local_experts must be at least top_k")


@dataclass(frozen=True)
class Candidate:
    """One mapper-format option row plus search lineage."""

    options: Mapping[str, object]
    tile_n: int
    source_global_index: int | None = None
    source_raw_index: int | None = None
    source_comment: str = ""
    scheduler_union_variant: bool = False
    parent_ids: tuple[str, ...] = ()
    move_kind: str = "incumbent"
    repair_kind: str | None = None

    @cached_property
    def candidate_id(self) -> str:
        return _digest(
            {
                "options": dict(self.options),
                "tile_n": self.tile_n,
            }
        )

    def as_json_row(self) -> dict[str, object]:
        return dict(self.options)


def structural_options_key(candidate: Candidate) -> str:
    """Identity of the full option subset copied by a crossover move."""

    return _digest(
        {
            field: candidate.options.get(field)
            for field in sorted(CROSSOVER_OPTION_FIELDS)
        }
    )


@dataclass(frozen=True)
class MaterializedCandidate:
    candidate: Candidate
    kwargs: Mapping[str, int]
    source_hash: str
    resolved_hashes: Mapping[int, str]
    compile_signatures: Mapping[int, Mapping[str, object]]
    smem_bytes: int | None
    smem_regime_kib: int
    tmem_columns: int
    multicast_operands: tuple[str, ...]


@dataclass(frozen=True)
class MoveLegality:
    """Legal move kinds plus stable rejection reasons for the rest."""

    legal: tuple[str, ...]
    rejected: Mapping[str, str]
    generated: Mapping[str, Candidate]


def enumerate_incumbents(
    context: ConfigContext,
    tile_ns: Iterable[int],
) -> list[Candidate]:
    """Return individual matching FC rows, without forming FC1×FC2 tactics."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for tile_n in sorted(set(int(value) for value in tile_ns)):
        indices = _moe_json_passing_indices(
            sm_arch=context.sm_arch,
            tile_n=tile_n,
            activation_type=context.activation_type,
            fc=context.fc,
            dtype_a=context.dtype_a,
            dtype_b=context.dtype_b,
            dtype_c=context.dtype_c,
            use_deepseek_fp8=context.use_deepseek_fp8,
            use_per_token_sf_a=context.use_per_token_sf_a,
            use_per_token_sf_b=context.use_per_token_sf_b,
            per_token_sf_dtype=context.per_token_sf_dtype,
            weight_layout=context.weight_layout,
        )
        for index in indices:
            cfg = _json_config_by_global_index(index, sm_arch=context.sm_arch)
            candidate = Candidate(
                options=dict(cfg.options),
                tile_n=tile_n,
                source_global_index=cfg.global_index,
                source_raw_index=cfg.raw_index,
                source_comment=cfg.comment,
                scheduler_union_variant=cfg.is_scheduler_union_variant,
            )
            candidate = _normalize_candidate_sf_routing(candidate, context)
            if candidate.candidate_id not in seen:
                seen.add(candidate.candidate_id)
                candidates.append(candidate)
    return candidates


def _json_cfg(candidate: Candidate) -> _JsonBatchedGemmConfig:
    return _JsonBatchedGemmConfig(
        global_index=(
            candidate.source_global_index
            if candidate.source_global_index is not None
            else -1
        ),
        raw_index=(
            candidate.source_raw_index if candidate.source_raw_index is not None else -1
        ),
        comment=candidate.source_comment or f"sweep:{candidate.move_kind}",
        combo_index=0,
        options=dict(candidate.options),
        is_scheduler_union_variant=candidate.scheduler_union_variant,
    )


def _mapped_candidate_kwargs(
    candidate: Candidate, context: ConfigContext
) -> dict[str, int]:
    return _json_config_kwargs(
        _json_cfg(candidate),
        sm_arch=context.sm_arch,
        fc=context.fc,
        activation_type=context.activation_type,
        has_bias=context.has_bias,
        default_dtype_a=context.dtype_a,
        default_dtype_b=context.dtype_b,
        default_dtype_c=context.dtype_c,
        weight_layout=context.weight_layout,
        use_per_token_sf_a=context.use_per_token_sf_a,
        use_per_token_sf_b=context.use_per_token_sf_b,
        per_token_sf_dtype=context.per_token_sf_dtype,
        has_gemm1_alpha=context.has_gemm1_alpha,
        has_gemm1_beta=context.has_gemm1_beta,
        has_gemm1_clamp_limit=context.has_gemm1_clamp_limit,
    )


def _explicitly_disables_sf_routing(options: Mapping[str, object]) -> bool:
    """Whether a candidate opts into pre-expanded activation scale factors."""
    if "route_sfs_act" not in options:
        return False
    value = options["route_sfs_act"]
    if value is None or value is False:
        return True
    if isinstance(value, str):
        return value.lower() in {"false", "none"}
    try:
        return int(value) == int(RouteImpl.NONE)
    except (TypeError, ValueError):
        return False


def _normalize_candidate_sf_routing(
    candidate: Candidate, context: ConfigContext
) -> Candidate:
    """Keep block-scaled activation SF routing enabled unless explicitly disabled."""
    if _explicitly_disables_sf_routing(candidate.options):
        return candidate
    cfg = make_config(**_mapped_candidate_kwargs(candidate, context))
    if not (
        cfg.has_input_block_scaling and cfg.has_routed_act and not cfg.has_routed_sfs
    ):
        return candidate
    options = dict(candidate.options)
    options["route_sfs_act"] = "ldgsts"
    return replace(candidate, options=options)


def candidate_kwargs(candidate: Candidate, context: ConfigContext) -> dict[str, int]:
    candidate = _normalize_candidate_sf_routing(candidate, context)
    kwargs = _mapped_candidate_kwargs(candidate, context)
    route_active = int(kwargs["route_act"]) != 0
    if route_active != (context.fc == "fc1"):
        raise ValueError(
            f"route_act is {'enabled' if route_active else 'disabled'} for "
            f"{context.fc}; this row would not match the production MoE mapper"
        )
    return kwargs


_MATERIALIZED_CACHE_MAX = 2048
_MATERIALIZED_CACHE: OrderedDict[
    tuple[str, ConfigContext, tuple[int, ...], bool],
    MaterializedCandidate | tuple[str, str],
] = OrderedDict()


def _materialize_candidate(
    candidate: Candidate,
    context: ConfigContext,
    problem_ks: Iterable[int],
    *,
    check_task_manager: bool = True,
) -> MaterializedCandidate:
    """Round-trip a row through the production mapper and resolve every K."""
    candidate = _normalize_candidate_sf_routing(candidate, context)
    kwargs = candidate_kwargs(candidate, context)
    cfg = make_config(**kwargs)
    compute_warp_layout(cfg)
    validate_config(cfg)
    register_conflicts = (
        warp_group_register_conflicts(cfg) if hasattr(cfg, "warp_layout") else []
    )
    if register_conflicts:
        raise ValueError("; ".join(register_conflicts))
    if check_task_manager:
        _ensure_config_task_manager_buildable(
            PrimsTsConfigSpec(kwargs=kwargs),
            dtype_label=context.dtype_label,
            fc=context.fc.upper(),
            tile_n=candidate.tile_n,
            moe_config_index=candidate.source_global_index or -1,
            top_k=context.top_k,
            num_experts=context.num_local_experts,
        )

    resolved_hashes: dict[int, str] = {}
    signatures: dict[int, Mapping[str, object]] = {}
    resolved_configs = []
    for problem_k in sorted(set(int(value) for value in problem_ks)):
        resolved = _runtime_config(cfg, problem_k)
        compute_warp_layout(resolved)
        validate_config(resolved)
        register_conflicts = (
            warp_group_register_conflicts(resolved)
            if hasattr(resolved, "warp_layout")
            else []
        )
        if register_conflicts:
            raise ValueError(f"K={problem_k}: {'; '.join(register_conflicts)}")
        resolved_configs.append((problem_k, resolved))
        resolved_hashes[problem_k] = stable_config_hash(resolved)
        signatures[problem_k] = resolved.compile_signature()

    multicast = tuple(
        operand
        for operand in ("a", "b", "sfa", "sfb")
        if needs_extended_tma_multicast(cfg, operand)
    )
    smem_bytes = None
    if check_task_manager:
        from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
            build_batched_gemm_task_manager,
        )

        for problem_k, resolved in resolved_configs:
            resolved_kwargs = {
                field.name: getattr(resolved, field.name)
                for field in fields(resolved)
                if field.init
            }
            manager = build_batched_gemm_task_manager(
                num_experts=context.num_local_experts,
                num_tokens=128,
                top_k=context.top_k,
                verbose=False,
                **resolved_kwargs,
            )
            resolved_smem_bytes = _task_manager_smem_bytes(manager)
            if resolved_smem_bytes is not None:
                smem_bytes = max(smem_bytes or 0, resolved_smem_bytes)
                limit = int(resolved.smem_capacity_bytes) - SWEEP_SMEM_HEADROOM_BYTES
                if candidate.move_kind != "incumbent" and resolved_smem_bytes > limit:
                    raise ValueError(
                        f"SMEM estimate {resolved_smem_bytes} B exceeds the sweep "
                        f"limit {limit} B ({SWEEP_SMEM_HEADROOM_BYTES} B below "
                        f"capacity) for K={problem_k}"
                    )

    regime = 327 if (smem_bytes or 0) > 227 * 1024 else 227
    return MaterializedCandidate(
        candidate=candidate,
        kwargs=kwargs,
        source_hash=_digest(dict(candidate.options)),
        resolved_hashes=resolved_hashes,
        compile_signatures=signatures,
        smem_bytes=smem_bytes,
        smem_regime_kib=regime,
        tmem_columns=int(cfg.tmem_required_cols),
        multicast_operands=multicast,
    )


def materialize_candidate(
    candidate: Candidate,
    context: ConfigContext,
    problem_ks: Iterable[int],
    *,
    check_task_manager: bool = True,
) -> MaterializedCandidate:
    """Materialize a candidate with a bounded cache of small validation results."""

    ks = tuple(sorted(set(int(value) for value in problem_ks)))
    key = (candidate.candidate_id, context, ks, check_task_manager)
    cached = _MATERIALIZED_CACHE.get(key)
    if cached is not None:
        _MATERIALIZED_CACHE.move_to_end(key)
        if isinstance(cached, tuple):
            error_type, message = cached
            raise ValueError(f"{error_type}: {message}")
        return replace(cached, candidate=candidate)
    try:
        result = _materialize_candidate(
            candidate,
            context,
            ks,
            check_task_manager=check_task_manager,
        )
    except Exception as exc:
        _MATERIALIZED_CACHE[key] = (type(exc).__name__, str(exc))
        _MATERIALIZED_CACHE.move_to_end(key)
        while len(_MATERIALIZED_CACHE) > _MATERIALIZED_CACHE_MAX:
            _MATERIALIZED_CACHE.popitem(last=False)
        raise
    _MATERIALIZED_CACHE[key] = result
    _MATERIALIZED_CACHE.move_to_end(key)
    while len(_MATERIALIZED_CACHE) > _MATERIALIZED_CACHE_MAX:
        _MATERIALIZED_CACHE.popitem(last=False)
    return result


def is_applicable(
    candidate: MaterializedCandidate,
    context: ConfigContext,
    shape: object,
) -> tuple[bool, str]:
    """Apply tile-N and per-shape cluster geometry checks."""
    selected = tuple(int(value) for value in shape.selected_tile_ns)
    if candidate.candidate.tile_n not in selected:
        return False, "tile_n_not_selected"
    fc = context.fc
    out_hidden = getattr(shape, "problem_n", None)
    if out_hidden is None:
        out_hidden = getattr(shape, f"{fc}_n")
    out_hidden = int(out_hidden)
    cfg = make_config(**dict(candidate.kwargs))
    try:
        _validate_swap_ab_cluster_span(cfg, out_hidden)
    except ValueError as exc:
        return False, f"cluster_span: {exc}"
    token_ctas = compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=int(shape.timing_num_tokens),
        num_experts=int(shape.local_num_experts),
        top_k=int(shape.top_k),
        token_tile_size=cfg.token_tile_size,
        cluster_dim_in_token=cfg.token_cluster_dim,
    )
    token_capacity = token_ctas * cfg.token_tile_size
    problem_m, problem_n = (
        (out_hidden, token_capacity) if cfg.is_swap_ab else (token_capacity, out_hidden)
    )
    try:
        validate_launch_grid(
            cfg,
            problem_m=problem_m,
            problem_n=problem_n,
            early_exit_max_token_ctas=token_ctas if cfg.use_early_exit else 0,
        )
    except ValueError as exc:
        return False, f"grid_limit: {exc}"
    return True, ""


def _supports_any_out_hidden(
    candidate: Candidate,
    context: ConfigContext | None,
) -> bool:
    """Return whether a structural candidate can run any sweep output width."""

    if context is None or not context.out_hiddens:
        return True
    try:
        normalized = _normalize_candidate_sf_routing(candidate, context)
        cfg = make_config(**candidate_kwargs(normalized, context))
        compute_warp_layout(cfg)
        validate_config(cfg)
    except ValueError:
        return False
    for out_hidden in context.out_hiddens:
        try:
            _validate_swap_ab_cluster_span(cfg, out_hidden)
        except ValueError:
            continue
        return True
    return False


def _register_roles(cfg) -> list[tuple[str, int, int, int, str | None]]:
    """Return active task register roles and their tunable option fields."""

    layout = cfg.warp_layout
    roles = [
        (
            "load_a",
            layout.load_a_warp_idx,
            layout.num_load_a_warps,
            cfg.load_a_task_regs,
            "load_a_regs",
        ),
        (
            "load_b",
            layout.load_b_warp_idx,
            layout.num_load_b_warps,
            cfg.load_b_task_regs,
            "load_b_regs",
        ),
        ("mma", layout.mma_warp_idx, layout.num_mma_warps, cfg.mma_regs, "mma_regs"),
        (
            "epilogue",
            layout.epilogue_warp_idx,
            layout.num_epilogue_warps,
            layout.epilogue_regs,
            "epilogue_regs",
        ),
    ]
    if cfg.has_input_block_scaling and not cfg.fuse_operand_sf_loads:
        roles.extend(
            (
                (
                    "load_sfa",
                    layout.load_sfa_warp_idx,
                    layout.num_load_sfa_warps,
                    cfg.load_sfa_task_regs,
                    "load_sfa_regs",
                ),
                (
                    "load_sfb",
                    layout.load_sfb_warp_idx,
                    layout.num_load_sfb_warps,
                    cfg.load_sfb_task_regs,
                    "load_sfb_regs",
                ),
            )
        )
    if cfg.has_input_block_scaling and not cfg.fuse_sf_copy_to_mma:
        roles.append(
            (
                "copy_sfa",
                layout.copy_sfa_warp_idx,
                layout.num_copy_sfa_warps,
                cfg.copy_sfa_task_regs,
                "copy_sf_regs",
            )
        )
        if not cfg.use_combined_sfab_copy:
            roles.append(
                (
                    "copy_sfb",
                    layout.copy_sfb_warp_idx,
                    layout.num_copy_sfb_warps,
                    cfg.copy_sfb_task_regs,
                    "copy_sf_regs",
                )
            )
    if cfg.has_cast_a:
        roles.extend(
            (
                (
                    "load_sfa",
                    layout.load_sfa_warp_idx,
                    layout.num_load_sfa_warps,
                    cfg.load_sfa_task_regs,
                    "load_sfa_regs",
                ),
                (
                    "cast_a",
                    layout.cast_a_warp_idx_first,
                    layout.num_cast_a_warps,
                    cfg.cast_a_regs,
                    "cast_a_regs",
                ),
            )
        )
    if cfg.has_gather and cfg.uses_two_cta_mma and layout.num_sync_warps > 0:
        roles.append(
            ("sync", layout.sync_warp_idx, layout.num_sync_warps, cfg.sync_regs, None)
        )
    if cfg.use_deepseek_fp8:
        roles.append(
            (
                "load_sfab",
                layout.load_sfab_warp_idx,
                layout.num_load_sfab_warps,
                cfg.load_sfab_regs,
                "load_sfab_regs",
            )
        )
    if layout.num_padding_warps > 0:
        roles.append(
            (
                "padding",
                layout.padding_warp_idx,
                layout.num_padding_warps,
                cfg.padding_regs,
                "padding_regs",
            )
        )
    if cfg.is_persistent:
        roles.append(
            (
                "workid",
                layout.workid_warp_idx,
                layout.num_workid_warps,
                cfg.workid_regs,
                "workid_regs",
            )
        )
    return roles


def warp_group_register_conflicts(cfg) -> list[str]:
    """Mirror TaskManager register invariants for sweep candidates."""

    layout = cfg.warp_layout
    active_roles = [role for role in _register_roles(cfg) if role[2] > 0]
    errors: list[str] = []
    num_warps = layout.threads_per_cta // 32
    for group_start in range(0, num_warps, 4):
        intersecting = [
            role
            for role in active_roles
            if role[1] < group_start + 4 and role[1] + role[2] > group_start
        ]
        register_values = {role[3] for role in intersecting}
        if len(register_values) > 1:
            details = ", ".join(
                f"{name}={regs}"
                for name, _start, _count, regs, _field in sorted(intersecting)
            )
            errors.append(
                f"warp group {group_start // 4} has mixed num_registers: {details}"
            )
        if register_values:
            covered = {
                warp
                for _name, start, count, _regs, _field in intersecting
                for warp in range(
                    max(start, group_start), min(start + count, group_start + 4)
                )
            }
            missing = set(range(group_start, group_start + 4)) - covered
            if missing:
                errors.append(
                    f"warp group {group_start // 4} has num_registers set but "
                    f"warps {sorted(missing)} are not covered"
                )
    per_warp_regs = [set() for _ in range(num_warps)]
    for _name, start, count, regs, _field in active_roles:
        for warp in range(start, start + count):
            per_warp_regs[warp].add(regs)
    total_registers = sum(max(registers, default=0) * 32 for registers in per_warp_regs)
    if total_registers > 65536:
        errors.append(
            f"CTA register allocation {total_registers} exceeds 65536 registers"
        )
    default_registers = (65536 // (num_warps * 32 * 8)) * 8
    register_budget = sum(
        default_registers - max(registers, default=default_registers)
        for registers in per_warp_regs
    )
    if register_budget < 0:
        errors.append(f"negative register budget: {register_budget}")
    return errors


def _harmonize_register_options(
    candidate: Candidate,
    context: ConfigContext,
) -> dict[str, object]:
    """Align tunable task register budgets within each physical warp group."""

    options = dict(candidate.options)
    try:
        cfg = make_config(
            **candidate_kwargs(replace(candidate, options=options), context)
        )
    except (KeyError, ValueError):
        # Repair tests deliberately use partial option dictionaries. Complete
        # production candidates continue through the register validation path.
        return options
    if not hasattr(cfg, "warp_layout"):
        return options
    layout = cfg.warp_layout
    roles = _register_roles(cfg)

    priority = ("sync", "mma", "epilogue", "cast_a", "load_sfab")
    num_warps = layout.threads_per_cta // 32
    for group_start in range(0, num_warps, 4):
        group_roles = [
            role
            for role in roles
            if role[2] > 0
            and role[1] < group_start + 4
            and role[1] + role[2] > group_start
        ]
        if len({role[3] for role in group_roles}) <= 1:
            continue
        target = next(
            (
                role[3]
                for role_name in priority
                for role in group_roles
                if role[0] == role_name
            ),
            max(role[3] for role in group_roles),
        )
        for _name, _start, _count, _regs, field in group_roles:
            if field is not None:
                options[field] = target
    return options


def _candidate_from_options(
    parent: Candidate,
    options: Mapping[str, object],
    *,
    move_kind: str,
    repair_kind: str | None = None,
    other_parent: Candidate | None = None,
) -> Candidate:
    parent_ids = (parent.candidate_id,)
    if other_parent is not None:
        parent_ids += (other_parent.candidate_id,)
    return Candidate(
        options=dict(options),
        tile_n=int(options.get("tile_n", parent.tile_n)),
        parent_ids=parent_ids,
        move_kind=move_kind,
        repair_kind=repair_kind,
    )


def _stage_fields(
    options: Mapping[str, object],
    *,
    tmem: bool = False,
    parent: Candidate | None = None,
    context: ConfigContext | None = None,
) -> list[str]:
    fields = _TMEM_STAGE_FIELDS if tmem else _SMEM_STAGE_FIELDS
    present = [name for name in fields if name in options and int(options[name]) > 0]
    if parent is None or context is None:
        return present
    signature = _effective_signature(options, parent, context)
    if signature is None:
        return present
    return [name for name in present if int(signature.get(name, 0)) > 0]


def _stage_groups(
    options: Mapping[str, object],
    *,
    tmem: bool = False,
    parent: Candidate | None = None,
    context: ConfigContext | None = None,
) -> list[tuple[str, ...]]:
    fields = _stage_fields(
        options,
        tmem=tmem,
        parent=parent,
        context=context,
    )
    groups = [(field,) for field in fields]
    linked: list[tuple[str, ...]] = []
    if (
        not tmem
        and int(options.get("umma_cta_count", 1)) == 2
        and {"num_stages_a", "num_stages_b"}.issubset(fields)
    ):
        linked.append(("num_stages_a", "num_stages_b"))
    if (
        tmem
        and bool(options.get("fuse_operand_sf_loads", False))
        and {
            "num_stages_tmem_sfa",
            "num_stages_tmem_sfb",
        }.issubset(fields)
    ):
        linked.append(("num_stages_tmem_sfa", "num_stages_tmem_sfb"))
    linked_fields = {field for group in linked for field in group}
    return linked + [group for group in groups if not linked_fields.intersection(group)]


def _smem_trade_groups(
    options: Mapping[str, object],
    *,
    parent: Candidate | None = None,
    context: ConfigContext | None = None,
) -> list[tuple[str, ...]]:
    """Include operand+scale linked variants in addition to ordinary rings."""

    groups = _stage_groups(options, parent=parent, context=context)
    fields = {field for group in groups for field in group}
    operand_groups = [
        group for group in groups if "num_stages_a" in group or "num_stages_b" in group
    ]
    linked = []
    for operand_group in operand_groups:
        for operand, sf in (
            ("num_stages_a", "num_stages_smem_sfa"),
            ("num_stages_b", "num_stages_smem_sfb"),
        ):
            if operand in operand_group and sf in fields:
                linked.append((*operand_group, sf))
    return list(dict.fromkeys([*groups, *linked]))


def _normalize_scheduler_dependents(
    options: dict[str, object],
    *,
    prefer_fast_drain: bool = False,
) -> None:
    """Keep flags derived from the scheduler legal after structural moves."""

    scheduler = options.get("tile_scheduler", "static")
    persistent = (
        str(scheduler).lower() == "persistent"
        if isinstance(scheduler, str)
        else int(scheduler) == 1
    )
    if prefer_fast_drain and bool(options.get("use_clc_fast_drain", False)):
        options["tile_scheduler"] = "persistent"
        options["use_early_exit"] = True
    elif not persistent:
        if "use_clc_fast_drain" in options:
            options["use_clc_fast_drain"] = False
        if "cta_raster_order" in options:
            options["cta_raster_order"] = 0
    elif bool(options.get("use_clc_fast_drain", False)):
        options["tile_scheduler"] = "persistent"


def _finalize_options(
    options: Mapping[str, object],
    *,
    stage_limits: StageLimits,
    max_sf_ring_ratio: int,
    prefer_fast_drain: bool = False,
) -> dict[str, object]:
    """Normalize dependent fields before comparing or materializing a move."""

    finalized = dict(options)
    # validate_config() pins CTA_2 to mma_m=256; derive mma_m so moves that
    # toggle the atom width never spend a validation slot on that mismatch.
    if "mma_m" in finalized:
        finalized["mma_m"] = (
            256
            if int(finalized.get("umma_cta_count", 1)) == 2
            else min(int(finalized["mma_m"]), 128)
        )
    _normalize_scheduler_dependents(
        finalized,
        prefer_fast_drain=prefer_fast_drain,
    )
    for field in _ALL_BOUNDED_STAGE_FIELDS:
        if field in finalized:
            finalized[field] = min(
                int(finalized[field]),
                stage_limits.limit_for(field),
            )
    # Keep SF rings within the deliberately loose search bound.
    for sf_field, operand_field in (
        ("num_stages_smem_sfa", "num_stages_a"),
        ("num_stages_smem_sfb", "num_stages_b"),
    ):
        if sf_field in finalized and operand_field in finalized:
            finalized[sf_field] = min(
                int(finalized[sf_field]),
                max_sf_ring_ratio * int(finalized[operand_field]),
            )
    return finalized


def _effective_signature_result(
    options: Mapping[str, object],
    parent: Candidate,
    context: ConfigContext,
) -> tuple[Mapping[str, object] | None, Exception | None]:
    trial = replace(
        parent,
        options=dict(options),
        tile_n=int(options.get("tile_n", parent.tile_n)),
    )
    try:
        kwargs = candidate_kwargs(trial, context)
        cfg = make_config(**kwargs)
        compute_warp_layout(cfg)
        validate_config(cfg)
    except Exception as exc:
        return None, exc
    return cfg.compile_signature(), None


def _effective_signature(
    options: Mapping[str, object],
    parent: Candidate,
    context: ConfigContext,
) -> Mapping[str, object] | None:
    signature, _ = _effective_signature_result(options, parent, context)
    return signature


def _effective_key(
    options: Mapping[str, object],
    parent: Candidate,
    context: ConfigContext | None,
) -> str | None:
    """Return the effective compile-time identity, or None when illegal."""

    if context is None:
        return _digest(dict(options))
    signature = _effective_signature(options, parent, context)
    return None if signature is None else _digest(signature)


def _pick_effective(
    parent: Candidate,
    trials: Iterable[Mapping[str, object]],
    context: ConfigContext | None,
    rng: random.Random,
    *,
    move_kind: str,
    stage_limits: StageLimits,
    max_sf_ring_ratio: int,
    check_task_manager: bool = False,
) -> dict[str, object]:
    """Pick a legal trial that changes the effective kernel configuration."""

    parent_options = _finalize_options(
        parent.options,
        stage_limits=stage_limits,
        max_sf_ring_ratio=max_sf_ring_ratio,
    )
    parent_key = _effective_key(parent_options, parent, context)
    shuffled = [
        _finalize_options(
            trial,
            stage_limits=stage_limits,
            max_sf_ring_ratio=max_sf_ring_ratio,
        )
        for trial in trials
    ]
    rng.shuffle(shuffled)
    first_error: Exception | None = None
    saw_legal_noop = False
    for options in shuffled:
        if context is None:
            key = _digest(dict(options))
            error = None
        else:
            signature, error = _effective_signature_result(options, parent, context)
            key = None if signature is None else _digest(signature)
        if key is None:
            if error is not None and first_error is None:
                first_error = error
            continue
        if key == parent_key:
            saw_legal_noop = True
            continue
        if check_task_manager and context is not None:
            trial = replace(
                parent,
                options=options,
                tile_n=int(options.get("tile_n", parent.tile_n)),
            )
            try:
                kwargs = candidate_kwargs(trial, context)
                _ensure_config_task_manager_buildable(
                    PrimsTsConfigSpec(kwargs=kwargs),
                    dtype_label=context.dtype_label,
                    fc=context.fc.upper(),
                    tile_n=trial.tile_n,
                    moe_config_index=trial.source_global_index or -1,
                    top_k=context.top_k,
                    num_experts=context.num_local_experts,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                continue
        return options
    if first_error is not None and not saw_legal_noop:
        raise ValueError(
            f"no effective {move_kind} mutation "
            f"(every trial illegal: {type(first_error).__name__}: {first_error})"
        )
    raise ValueError(f"no effective {move_kind} mutation")


def enumerate_stage_neighbors(
    parent: Candidate,
    move_kind: str,
    *,
    stage_limits: StageLimits = DEFAULT_STAGE_LIMITS,
    max_sf_ring_ratio: int = DEFAULT_MAX_SF_RING_RATIO,
    context: ConfigContext | None = None,
) -> tuple[Candidate, ...]:
    """Enumerate every one-step, effectively distinct stage neighbor.

    Unlike the symbolic ``fill`` repairs, these are concrete local neighbors.
    Search drivers can therefore mark a ``(parent, move)`` pair exhausted once
    this finite set has been consumed instead of repeatedly drawing duplicates.
    """

    if move_kind not in STAGE_MOVE_KINDS:
        raise ValueError(f"{move_kind!r} is not a stage move")
    options = dict(parent.options)
    trials: list[tuple[dict[str, object], str]] = []
    tmem = move_kind == "tmem_stages"
    groups = (
        _smem_trade_groups(options, parent=parent, context=context)
        if move_kind == "stages_trade"
        else _stage_groups(
            options,
            tmem=tmem,
            parent=parent,
            context=context,
        )
    )
    if move_kind == "stages_all":
        fields = _stage_fields(options, parent=parent, context=context)
        for delta in (-1, 1):
            if fields and all(
                1 <= int(options[field]) + delta <= stage_limits.limit_for(field)
                for field in fields
            ):
                trial = dict(options)
                for field in fields:
                    trial[field] = int(trial[field]) + delta
                trials.append((trial, f"stages_all:{delta}"))
    elif move_kind in {"stages_one", "tmem_stages"}:
        prefix = "tmem_one" if tmem else "stages_one"
        for group in groups:
            for delta in (-1, 1):
                if all(
                    1 <= int(options[field]) + delta <= stage_limits.limit_for(field)
                    for field in group
                ):
                    trial = dict(options)
                    for field in group:
                        trial[field] = int(trial[field]) + delta
                    trials.append((trial, f"{prefix}:{','.join(group)}:{delta}"))
        if tmem:
            for donor in groups:
                for receiver in groups:
                    if set(donor).intersection(receiver):
                        continue
                    for amount in (1, 2):
                        if not (
                            all(int(options[field]) - amount >= 1 for field in donor)
                            and all(
                                int(options[field]) + amount
                                <= stage_limits.limit_for(field)
                                for field in receiver
                            )
                        ):
                            continue
                        trial = dict(options)
                        for field in donor:
                            trial[field] = int(trial[field]) - amount
                        for field in receiver:
                            trial[field] = int(trial[field]) + amount
                        trials.append(
                            (
                                trial,
                                "tmem_neighbor_trade:"
                                f"{','.join(donor)}:{','.join(receiver)}:{amount}",
                            )
                        )
    else:
        for donor in groups:
            for receiver in groups:
                if set(donor).intersection(receiver):
                    continue
                for amount in (1, 2):
                    if not (
                        all(int(options[field]) - amount >= 1 for field in donor)
                        and all(
                            int(options[field]) + amount
                            <= stage_limits.limit_for(field)
                            for field in receiver
                        )
                    ):
                        continue
                    trial = dict(options)
                    for field in donor:
                        trial[field] = int(trial[field]) - amount
                    for field in receiver:
                        trial[field] = int(trial[field]) + amount
                    trials.append(
                        (
                            trial,
                            "neighbor_stages_trade:"
                            f"{','.join(donor)}:{','.join(receiver)}:{amount}",
                        )
                    )

    parent_key = _effective_key(
        _finalize_options(
            parent.options,
            stage_limits=stage_limits,
            max_sf_ring_ratio=max_sf_ring_ratio,
        ),
        parent,
        context,
    )
    result: list[Candidate] = []
    seen: set[str] = set()
    for trial, repair_kind in trials:
        finalized = _finalize_options(
            trial,
            stage_limits=stage_limits,
            max_sf_ring_ratio=max_sf_ring_ratio,
        )
        key = _effective_key(finalized, parent, context)
        if key is None or key == parent_key or key in seen:
            continue
        seen.add(key)
        child = _candidate_from_options(
            parent,
            finalized,
            move_kind=move_kind,
            repair_kind=repair_kind,
        )
        result.append(
            _normalize_candidate_sf_routing(child, context)
            if context is not None
            else child
        )
    return tuple(result)


def _move_rejection_reason(exc: Exception) -> str:
    message = str(exc).lower()
    if "every trial illegal" in message:
        return "illegal"
    if "did not change" in message or "no effective" in message:
        return "no_effective_mutation"
    if (
        "no live" in message
        or "no legal neighbor" in message
        or "saturated" in message
        or "requires other_parent" in message
    ):
        return "no_legal_neighbor"
    return "illegal"


def filter_legal_move_kinds(
    parent: Candidate,
    move_kinds: Sequence[str],
    *,
    other_parent: Candidate | None = None,
    stage_limits: StageLimits = DEFAULT_STAGE_LIMITS,
    max_sf_ring_ratio: int = DEFAULT_MAX_SF_RING_RATIO,
    context: ConfigContext | None = None,
    seed: int = 0,
    enable_cluster_y: bool = False,
) -> MoveLegality:
    """Filter moves that have no legal local realization for ``parent``."""

    legal: list[str] = []
    rejected: dict[str, str] = {}
    generated: dict[str, Candidate] = {}
    for offset, move_kind in enumerate(move_kinds):
        try:
            if move_kind in STAGE_MOVE_KINDS:
                if not enumerate_stage_neighbors(
                    parent,
                    move_kind,
                    stage_limits=stage_limits,
                    max_sf_ring_ratio=max_sf_ring_ratio,
                    context=context,
                ):
                    raise ValueError("no legal neighbor")
            else:
                errors: list[Exception] = []
                trial_count = 1 if move_kind == "warps" else 6
                for trial_index in range(trial_count):
                    try:
                        child = generate_move(
                            parent,
                            move_kind,
                            random.Random(seed + 17 * offset + trial_index),
                            other_parent=other_parent,
                            stage_limits=stage_limits,
                            max_sf_ring_ratio=max_sf_ring_ratio,
                            repair_index=trial_index % 3,
                            context=context,
                            enable_cluster_y=enable_cluster_y,
                        )
                    except Exception as exc:
                        errors.append(exc)
                    else:
                        generated[move_kind] = child
                        break
                else:
                    raise errors[-1]
        except Exception as exc:
            rejected[move_kind] = _move_rejection_reason(exc)
        else:
            legal.append(move_kind)
    return MoveLegality(tuple(legal), rejected, generated)


def generate_move(
    parent: Candidate,
    move_kind: str,
    rng: random.Random,
    *,
    other_parent: Candidate | None = None,
    stage_limits: StageLimits = DEFAULT_STAGE_LIMITS,
    max_sf_ring_ratio: int = DEFAULT_MAX_SF_RING_RATIO,
    repair_index: int = 0,
    context: ConfigContext | None = None,
    enable_cluster_y: bool = False,
) -> Candidate:
    """Generate one child. Legality and resources are checked after generation."""
    if move_kind not in MOVE_KINDS:
        raise ValueError(f"unknown move kind {move_kind!r}")
    if move_kind in STAGE_MOVE_KINDS:
        raise ValueError(f"{move_kind} must be selected from enumerate_stage_neighbors")
    if context is not None and move_kind not in applicable_move_kinds(
        context, (move_kind,)
    ):
        if move_kind in ROUTING_MOVE_KINDS:
            raise ValueError(
                f"{move_kind} only varies routed-operand loading; "
                f"{context.fc} has no routing"
            )
        raise ValueError(f"{move_kind} is not applicable to sm_{context.sm_arch}")
    if context is not None:
        probe = make_config(**candidate_kwargs(parent, context))
        unsupported_reason = _unsupported_move_reason(probe, move_kind)
        if unsupported_reason is not None:
            raise ValueError(unsupported_reason)
    options = dict(parent.options)
    repair_kind = f"repair_{repair_index}"
    parent_signature = (
        _effective_signature(options, parent, context) if context is not None else None
    )

    def bump(field: str, delta: int) -> None:
        options[field] = max(
            1,
            min(
                stage_limits.limit_for(field),
                int(options.get(field, 1)) + delta,
            ),
        )

    def current(field: str, default: object) -> object:
        if parent_signature is not None and field in parent_signature:
            return parent_signature[field]
        return options.get(field, default)

    def current_route(field: str) -> int:
        value = current(field, "ldgsts")
        if isinstance(value, int) and not isinstance(value, bool):
            return int(RouteImpl(value))
        return _json_route_value(value)

    def pick(
        trials: Iterable[Mapping[str, object]],
        *,
        check_task_manager: bool = False,
    ) -> dict[str, object]:
        return _pick_effective(
            parent,
            trials,
            context,
            rng,
            move_kind=move_kind,
            stage_limits=stage_limits,
            max_sf_ring_ratio=max_sf_ring_ratio,
            check_task_manager=check_task_manager,
        )

    def rescale_tmem_sf_stages(
        trial: Mapping[str, object],
        next_tile_k: int,
        *,
        round_up: bool,
        clamp: bool,
    ) -> dict[str, object] | None:
        scaled = dict(trial)
        current_tile_k = int(options["tile_k"])
        for field in ("num_stages_tmem_sfa", "num_stages_tmem_sfb"):
            current_stages = int(current(field, 0))
            if current_stages <= 0:
                continue
            if next_tile_k > current_tile_k:
                next_stages = (
                    (current_stages + 1) // 2 if round_up else current_stages // 2
                )
            else:
                next_stages = current_stages * 2
            if clamp:
                next_stages = min(
                    max(1, next_stages),
                    stage_limits.limit_for(field),
                )
            elif not 1 <= next_stages <= stage_limits.limit_for(field):
                return None
            scaled[field] = next_stages
        return scaled

    if move_kind == "route_warps":
        route_trials = []
        routed_operand = "a"
        if context is not None:
            probe = make_config(**candidate_kwargs(parent, context))
            routed_operand = "b" if probe.is_swap_ab else "a"
        for field in ("route_act", "route_sfs_act"):
            route = current_route(field)
            next_route = "ldgsts" if route == int(RouteImpl.TMA) else "tma"
            operand = (
                f"sf{routed_operand}" if field == "route_sfs_act" else routed_operand
            )
            warp_field = f"num_load_{operand}_warps"
            current_warps = int(current(warp_field, 1))
            for next_warps in (1, 2, 4):
                if next_warps == current_warps:
                    continue
                trial = {
                    **options,
                    field: next_route,
                    warp_field: next_warps,
                }
                if bool(current("fuse_operand_sf_loads", False)):
                    paired_operand = (
                        operand.removeprefix("sf")
                        if operand.startswith("sf")
                        else f"sf{operand}"
                    )
                    trial[f"num_load_{paired_operand}_warps"] = next_warps
                route_trials.append(trial)
        options = pick(route_trials, check_task_manager=True)
        repair_kind = "route_warps"
    elif move_kind == "tile_k_stages":
        current_tile_k = int(options["tile_k"])
        stage_fields = _stage_fields(
            options,
            parent=parent,
            context=context,
        )
        trials = []
        for next_tile_k in (current_tile_k // 2, current_tile_k * 2):
            if next_tile_k <= 0 or next_tile_k == current_tile_k:
                continue
            for round_up in (False, True):
                trial = {**options, "tile_k": next_tile_k}
                valid = bool(stage_fields)
                for field in stage_fields:
                    current_stages = int(current(field, options[field]))
                    if next_tile_k < current_tile_k:
                        next_stages = current_stages * 2
                    else:
                        next_stages = (
                            (current_stages + 1) // 2
                            if round_up
                            else current_stages // 2
                        )
                    if not 1 <= next_stages <= stage_limits.limit_for(field):
                        valid = False
                        break
                    trial[field] = next_stages
                if not valid:
                    continue
                trial = rescale_tmem_sf_stages(
                    trial,
                    next_tile_k,
                    round_up=round_up,
                    clamp=False,
                )
                if trial is not None:
                    trials.append(trial)
        options = pick(trials, check_task_manager=True)
        repair_kind = "tile_k_stages:inverse"
    elif move_kind == "tile_k":
        tile_k = int(options["tile_k"])
        trials = []
        for next_tile_k in (tile_k // 2, tile_k * 2):
            plain = {**options, "tile_k": next_tile_k}
            trials.append(plain)
            rescaled = rescale_tmem_sf_stages(
                plain,
                next_tile_k,
                round_up=True,
                clamp=True,
            )
            if rescaled is not None:
                trials.append(rescaled)
        options = pick(trials, check_task_manager=True)
    elif move_kind == "epilogue":
        tile_n = int(options.get("tile_n", parent.tile_n))
        current_epi = int(current("epi_tile_n", min(8, tile_n)))
        current_stages = int(current("num_stages_c_smem", 1))
        current_tma_store = bool(current("use_tma_store", True))
        trials = [
            {**options, "epi_tile_n": value}
            for value in (8, 16, 32, 64, 128)
            if tile_n % value == 0 and value != current_epi
        ]
        trials.extend(
            {**options, "num_stages_c_smem": value}
            for value in (current_stages - 1, current_stages + 1)
            if 1 <= value <= stage_limits.c_smem
        )
        store_trial = {**options, "use_tma_store": not current_tma_store}
        if not current_tma_store:
            store_trial["num_stages_c_smem"] = max(
                1,
                int(options.get("num_stages_c_smem", current_stages)),
            )
        trials.append(store_trial)
        options = pick(trials, check_task_manager=True)
    elif move_kind == "warps":
        fields = (
            "num_load_a_warps",
            "num_load_b_warps",
            "num_load_sfa_warps",
            "num_load_sfb_warps",
        )
        probe = None
        if context is not None:
            probe = make_config(
                **candidate_kwargs(
                    replace(parent, options=options),
                    context,
                )
            )
        routed_operand = "b" if probe is not None and probe.is_swap_ab else "a"

        def is_routed(value: str) -> bool:
            if probe is None:
                return True
            return (
                probe.has_routed_sfs and value == f"sf{routed_operand}"
                if value.startswith("sf")
                else probe.has_routed_act and value == routed_operand
            )

        choices_by_field = {}
        for field in fields:
            operand = field.removeprefix("num_load_").removesuffix("_warps")
            paired_operand = (
                operand.removeprefix("sf")
                if operand.startswith("sf")
                else f"sf{operand}"
            )
            choices = (1, 2, 4)
            if not is_routed(operand) or (
                probe is not None
                and probe.fuse_operand_sf_loads
                and not is_routed(paired_operand)
            ):
                choices = (1,)
            current_warps = int(current(field, 0))
            alternatives = tuple(value for value in choices if value != current_warps)
            if alternatives:
                choices_by_field[field] = alternatives
        if not choices_by_field:
            raise ValueError("no load-warp field has an effective alternative")
        trials = []
        alternatives = [
            (field, selected_warps)
            for field, choices in choices_by_field.items()
            for selected_warps in choices
        ]
        for field, selected_warps in alternatives:
            trial_options = {**options, field: selected_warps}
            if probe is not None and probe.fuse_operand_sf_loads:
                if field in ("num_load_a_warps", "num_load_b_warps"):
                    trial_options[field.replace("num_load_", "num_load_sf")] = (
                        selected_warps
                    )
                elif field in ("num_load_sfa_warps", "num_load_sfb_warps"):
                    trial_options[field.replace("num_load_sf", "num_load_")] = (
                        selected_warps
                    )
            trials.append(trial_options)
        rng.shuffle(trials)
        options = pick(trials, check_task_manager=True)
    elif move_kind == "register_split":
        current_load = int(current("load_regs", 48))
        current_mma = int(current("mma_regs", 48))
        current_epilogue = int(current("epilogue_regs", 160))
        trials = []
        for delta in (-16, -8, 8, 16):
            trial = {
                **options,
                "load_regs": current_load + delta,
                "mma_regs": current_mma + delta,
                "epilogue_regs": current_epilogue - 2 * delta,
            }
            if all(
                24 <= int(trial[field]) <= 232 and int(trial[field]) % 8 == 0
                for field in ("load_regs", "mma_regs", "epilogue_regs")
            ):
                trials.append(trial)
        if context is not None:
            trials = [
                _harmonize_register_options(
                    _candidate_from_options(parent, trial, move_kind=move_kind),
                    context,
                )
                for trial in trials
            ]
        options = pick(
            trials,
            check_task_manager=True,
        )
    elif move_kind == "flags":
        trials = []
        for field in (
            "use_unroll_loop_2x_for_mma",
            "use_tma_oob_opt",
            "use_clc_fast_drain",
            "use_work_throttle",
            "tile_scheduler",
            "cta_raster_order",
        ):
            trial = dict(options)
            if field == "cta_raster_order":
                trial[field] = 1 - int(current(field, 0))
                if trial[field] == 1:
                    trial["tile_scheduler"] = "persistent"
            elif field == "tile_scheduler":
                scheduler = current(field, "static")
                is_persistent = (
                    str(scheduler).lower() == "persistent"
                    if isinstance(scheduler, str)
                    else int(scheduler) == 1
                )
                trial[field] = "static" if is_persistent else "persistent"
            else:
                trial[field] = not bool(current(field, False))
                if field == "use_clc_fast_drain" and trial[field]:
                    trial = _finalize_options(
                        trial,
                        stage_limits=stage_limits,
                        max_sf_ring_ratio=max_sf_ring_ratio,
                        prefer_fast_drain=True,
                    )
            trials.append(trial)
        options = pick(trials, check_task_manager=True)
    elif move_kind == "skeleton":
        probe = (
            make_config(**candidate_kwargs(parent, context))
            if context is not None
            else None
        )
        trials = []
        for field in (
            "route_act",
            "route_sfs_act",
            "fuse_operand_sf_loads",
            "fuse_sf_copy_to_mma",
            "use_max_tmem_overlap",
            "umma_cta_count",
        ):
            trial = dict(options)
            if field.startswith("route_"):
                if context is not None and context.fc != "fc1":
                    continue
                route = current_route(field)
                trial[field] = "ldgsts" if route == int(RouteImpl.TMA) else "tma"
            elif field == "umma_cta_count":
                trial[field] = 2 if int(current(field, 1)) == 1 else 1
                if (
                    trial[field] == 2
                    and probe is not None
                    and probe.mma_n < _MIN_TWO_CTA_MMA_N
                ):
                    continue
                if trial[field] == 2:
                    trial["cluster_dim_x"] = max(2, int(trial.get("cluster_dim_x", 0)))
                else:
                    trial["cluster_dim_x"] = 0
                    trial["fallback_cluster_dim_x"] = 0
            else:
                trial[field] = not bool(current(field, False))
                if (
                    field == "fuse_operand_sf_loads"
                    and context is not None
                    and context.fc != "fc1"
                ):
                    continue
                if (
                    field == "fuse_sf_copy_to_mma"
                    and trial[field]
                    and probe is not None
                    and (
                        probe.smem_sfa_layout != int(SfLayout.R128c4)
                        or probe.smem_sfb_layout != int(SfLayout.R128c4)
                    )
                ):
                    continue
                if (
                    field == "use_max_tmem_overlap"
                    and trial[field]
                    and probe is not None
                    and (probe.tile_n != 256 or not probe.has_input_block_scaling)
                ):
                    continue
                if field == "fuse_sf_copy_to_mma" and not trial[field]:
                    trial["fuse_operand_sf_loads"] = False
                    trial["use_max_tmem_overlap"] = False
                elif field == "fuse_operand_sf_loads" and trial[field]:
                    trial["fuse_sf_copy_to_mma"] = True
            trials.append(trial)
        if not trials:
            raise ValueError("skeleton is unsupported for this parent")
        options = pick(trials, check_task_manager=True)
    elif move_kind == "cluster":
        cluster_shapes = [
            (2, 1, 0, 1),
            (4, 1, 0, 1),
            (4, 1, 2, 1),
        ]
        if enable_cluster_y:
            cluster_shapes.extend(
                (
                    (2, 2, 0, 1),
                    (2, 4, 0, 1),
                    (4, 2, 0, 1),
                )
            )
        cluster_trials = [
            {
                **options,
                "umma_cta_count": 2,
                "cluster_dim_x": cluster_dim_x,
                "cluster_dim_y": cluster_dim_y,
                "fallback_cluster_dim_x": fallback,
                "fallback_cluster_dim_y": fallback_y,
            }
            for cluster_dim_x, cluster_dim_y, fallback, fallback_y in cluster_shapes
        ]
        cluster_trials = [
            trial
            for trial in cluster_trials
            if _supports_any_out_hidden(
                _candidate_from_options(parent, trial, move_kind=move_kind),
                context,
            )
        ]
        if context is not None:
            cluster_trials = [
                _harmonize_register_options(
                    _candidate_from_options(parent, trial, move_kind=move_kind),
                    context,
                )
                for trial in cluster_trials
            ]
        if not cluster_trials:
            raise ValueError("no cluster shape supports any sweep out_hidden")
        options = pick(cluster_trials, check_task_manager=True)
    elif move_kind == "large_carveout":
        delta = 1 if repair_index % 2 == 0 else -1
        fields = _stage_fields(
            options,
            parent=parent,
            context=context,
        )
        if not fields or not any(
            1 <= int(options[field]) + delta <= stage_limits.limit_for(field)
            for field in fields
        ):
            raise ValueError("large carveout has no movable live SMEM stage")
        for field in fields:
            bump(field, delta)
    elif move_kind == "crossover":
        if other_parent is None:
            raise ValueError("crossover requires other_parent")
        fallback_donor_trials = []
        for field_group in CROSSOVER_DONOR_FIELD_GROUPS:
            trial = dict(options)
            for key in field_group:
                if not enable_cluster_y and key in {
                    "cluster_dim_y",
                    "fallback_cluster_dim_y",
                }:
                    continue
                if key in other_parent.options:
                    trial[key] = other_parent.options[key]
            fallback_donor_trials.append(trial)
        fallback_donor_trials.append(
            {
                **options,
                **{
                    key: other_parent.options[key]
                    for key in STRUCTURAL_OPTION_FIELDS
                    if enable_cluster_y
                    or key not in {"cluster_dim_y", "fallback_cluster_dim_y"}
                    if key in other_parent.options
                },
            }
        )
        tuned_donor_trials = []
        for trial in fallback_donor_trials:
            tuned_trial = dict(trial)
            for key in CROSSOVER_DONOR_TUNING_FIELDS:
                if key in other_parent.options:
                    tuned_trial[key] = other_parent.options[key]
            tuned_donor_trials.append(tuned_trial)
        try:
            options = pick(tuned_donor_trials, check_task_manager=True)
            repair_kind = "crossover:donor_tuned"
        except ValueError:
            options = pick(fallback_donor_trials, check_task_manager=True)
            repair_kind = "crossover:donor_repaired"
    options = _finalize_options(
        options,
        stage_limits=stage_limits,
        max_sf_ring_ratio=max_sf_ring_ratio,
    )

    guarded_move = move_kind in {
        "tile_k",
        "tile_k_stages",
        "epilogue",
        "warps",
        "route_warps",
        "register_split",
        "flags",
        "skeleton",
        "cluster",
        "crossover",
    }
    if guarded_move:
        child_key = _effective_key(options, parent, context)
        parent_key = _effective_key(
            _finalize_options(
                parent.options,
                stage_limits=stage_limits,
                max_sf_ring_ratio=max_sf_ring_ratio,
            ),
            parent,
            context,
        )
        if child_key is None or child_key == parent_key:
            reason = (
                "is illegal"
                if child_key is None
                else "did not change the effective config"
            )
            raise ValueError(f"{move_kind} move {reason}")
    child = _candidate_from_options(
        parent,
        options,
        move_kind=move_kind,
        repair_kind=repair_kind,
        other_parent=other_parent,
    )
    return (
        _normalize_candidate_sf_routing(child, context)
        if context is not None
        else child
    )


def fixed_seed_variants(
    incumbents: Sequence[Candidate],
    *,
    limit: int,
    sm_arch: int | None = None,
) -> list[Candidate]:
    """Generate bounded max-stage, direct-store, and cluster seeds."""
    result: list[Candidate] = []
    for parent in incumbents[: max(0, limit)]:
        options = dict(parent.options)
        result.append(
            _candidate_from_options(
                parent,
                options,
                move_kind="seed_stages",
                repair_kind="stages_all:max",
            )
        )
        result.append(
            _candidate_from_options(
                parent,
                {**options, "use_tma_store": False},
                move_kind="seed_stages",
                repair_kind="stages_all:max",
            )
        )
        if sm_arch == 107:
            result.append(
                _candidate_from_options(
                    parent,
                    options,
                    move_kind="large_carveout",
                    repair_kind="stages_all:max",
                )
            )
        for group in _stage_groups(options):
            result.append(
                _candidate_from_options(
                    parent,
                    options,
                    move_kind="seed_stages",
                    repair_kind=f"stages_all_then_one:{','.join(group)}:fill",
                )
            )
            if sm_arch == 107:
                result.append(
                    _candidate_from_options(
                        parent,
                        options,
                        move_kind="large_carveout",
                        repair_kind=(f"stages_all_then_one:{','.join(group)}:fill"),
                    )
                )
        if int(parent.options.get("umma_cta_count", 1)) == 2:
            for fallback in (0, 2):
                clustered = {
                    **parent.options,
                    "cluster_dim_x": 4,
                    "fallback_cluster_dim_x": fallback,
                }
                result.append(
                    _candidate_from_options(
                        parent,
                        clustered,
                        move_kind="seed_cluster",
                        repair_kind=f"fallback_{fallback}",
                    )
                )
    if sm_arch == 107:
        result.extend(
            replace(
                candidate,
                repair_kind=f"{candidate.repair_kind}:tmem_fill",
            )
            for candidate in list(result)
        )
    # Symbolic max/fill seeds intentionally have the parent's options until
    # repair spends the resource budget.  Do not deduplicate them before that
    # repair gives each seed its concrete configuration.
    return result


def repair_structural_candidate(
    candidate: Candidate,
    context: ConfigContext,
    problem_ks: Iterable[int],
    *,
    repair_index: int,
    stage_limits: StageLimits = DEFAULT_STAGE_LIMITS,
    max_sf_ring_ratio: int = DEFAULT_MAX_SF_RING_RATIO,
    target_smem_regime_kib: int | None = None,
    fill_tmem: bool | None = None,
) -> Candidate:
    """Move a structural child to a nearby legal SMEM boundary.

    Repair zero preserves ring ratios with uniform shifts. Repairs one and two
    spend remaining capacity on one ring group. Every accepted step is
    confirmed by the production TaskManager rather than trusting a duplicate
    resource model.
    """
    if candidate.move_kind not in {
        "stages_all",
        "stages_one",
        "stages_trade",
        "tmem_stages",
        "tile_k",
        "tile_k_stages",
        "skeleton",
        "cluster",
        "large_carveout",
        "crossover",
        "seed_stages",
        "seed_cluster",
    }:
        return candidate
    options = _finalize_options(
        candidate.options,
        stage_limits=stage_limits,
        max_sf_ring_ratio=max_sf_ring_ratio,
    )
    options = _harmonize_register_options(
        replace(candidate, options=options),
        context,
    )
    repair = candidate.repair_kind or ""
    seed_tmem_fill = repair.endswith(":tmem_fill")
    if seed_tmem_fill:
        repair = repair.removesuffix(":tmem_fill")
    is_tmem_repair = repair.startswith("tmem_")
    if fill_tmem is None:
        fill_tmem = seed_tmem_fill or (
            repair.startswith("repair_")
            and candidate.move_kind
            in {
                "skeleton",
                "cluster",
                "large_carveout",
                "crossover",
            }
        )
    fields = _stage_fields(options, tmem=is_tmem_repair)
    if not fields:
        if not _supports_any_out_hidden(candidate, context):
            raise ValueError("candidate supports no sweep out_hidden cluster span")
        return candidate
    problem_ks = tuple(problem_ks)

    def with_options(value: Mapping[str, object], repair: str) -> Candidate:
        if fill_tmem:
            repair = f"{repair}:tmem_fill"
        return replace(
            candidate,
            options=dict(value),
            tile_n=int(value.get("tile_n", candidate.tile_n)),
            repair_kind=repair,
        )

    # Compound and donor-tuned moves encode deliberate coupled ratios. Reject
    # an illegal combination instead of independently clamping one component.
    preserve_coupling = candidate.move_kind in {
        "tile_k_stages",
        "crossover",
    }
    if preserve_coupling:
        current = with_options(options, f"repair_{repair_index}")
        current_materialized = materialize_candidate(
            current, context, problem_ks, check_task_manager=True
        )
    else:
        # Clamp ordinary structural children down until they are legal.
        while True:
            current = with_options(options, f"repair_{repair_index}")
            try:
                current_materialized = materialize_candidate(
                    current, context, problem_ks, check_task_manager=True
                )
                break
            except Exception as exc:
                message = str(exc).lower()
                if "smem" not in message and "shared memory" not in message:
                    raise
                reducible = [field for field in fields if int(options[field]) > 1]
                if not reducible:
                    raise
                for field in reducible:
                    options[field] = int(options[field]) - 1

    if candidate.move_kind == "large_carveout":
        target_kib = 327 if context.sm_arch == 107 and repair_index % 2 == 0 else 227
    elif target_smem_regime_kib is not None:
        target_kib = target_smem_regime_kib
    else:
        target_kib = current_materialized.smem_regime_kib
    if (
        not is_tmem_repair
        and candidate.move_kind != "large_carveout"
        and target_kib == 327
        and (current_materialized.smem_bytes or 0) <= 227 * 1024
    ):
        raise ValueError("stage move would cross from the 327 KiB SMEM regime")
    prefill_fields: Sequence[str] | None = None
    if repair == "stages_all:max":
        selected_fields = fields
    elif repair.startswith("stages_all_then_one:") and repair.endswith(":fill"):
        prefill_fields = fields
        selected_fields = repair.split(":")[1].split(",")
    elif (
        repair.startswith("stages_one:")
        or repair.startswith("stages_trade:")
        or repair.startswith("tmem_one:")
        or repair.startswith("tmem_trade:")
    ) and (repair.endswith(":fill") or "trade:" in repair):
        parts = repair.split(":")
        selected_fields = parts[2 if "trade:" in repair else 1].split(",")
    elif (
        candidate.move_kind
        in {
            "stages_all",
            "stages_one",
            "stages_trade",
            "tmem_stages",
        }
        or (candidate.move_kind == "tile_k_stages" and repair.startswith("stages_one:"))
        or candidate.move_kind
        in {
            "tile_k_stages",
            "crossover",
        }
    ):
        if not _supports_any_out_hidden(current, context):
            raise ValueError("candidate supports no sweep out_hidden cluster span")
        return current
    elif repair_index % 3 == 0:
        selected_fields = fields
    else:
        selected_fields = [fields[(repair_index - 1) % len(fields)]]

    def fill(selected: Sequence[str]) -> None:
        nonlocal options
        while all(
            int(options[field]) < stage_limits.limit_for(field) for field in selected
        ):
            trial_options = dict(options)
            for field in selected:
                trial_options[field] = int(trial_options[field]) + 1
            if not is_tmem_repair and any(
                sf_field in trial_options
                and operand_field in trial_options
                and int(trial_options[sf_field])
                > max_sf_ring_ratio * int(trial_options[operand_field])
                for sf_field, operand_field in (
                    ("num_stages_smem_sfa", "num_stages_a"),
                    ("num_stages_smem_sfb", "num_stages_b"),
                )
            ):
                break
            trial = with_options(trial_options, f"repair_{repair_index}")
            try:
                materialized = materialize_candidate(
                    trial, context, problem_ks, check_task_manager=True
                )
            except Exception:
                break
            if (
                not is_tmem_repair
                and materialized.smem_bytes is not None
                and materialized.smem_bytes > target_kib * 1024
            ):
                break
            options = trial_options

    if prefill_fields is not None:
        fill(prefill_fields)
    fill(selected_fields)
    if fill_tmem and not is_tmem_repair:
        tmem_fields = _stage_fields(options, tmem=True)
        if repair_index % 3 == 0:
            selected_tmem_fields = [
                field for field in tmem_fields if field == "num_stages_tmem_acc"
            ]
        elif repair_index % 3 == 1:
            selected_tmem_fields = [
                field for field in tmem_fields if field != "num_stages_tmem_acc"
            ]
        else:
            selected_tmem_fields = tmem_fields

        if selected_tmem_fields:
            current = with_options(options, f"repair_{repair_index}")
            current_materialized = materialize_candidate(
                current, context, problem_ks, check_task_manager=True
            )
            while all(
                int(options[field]) < stage_limits.limit_for(field)
                for field in selected_tmem_fields
            ):
                trial_options = dict(options)
                for field in selected_tmem_fields:
                    trial_options[field] = int(trial_options[field]) + 1
                trial = with_options(trial_options, f"repair_{repair_index}")
                try:
                    materialized = materialize_candidate(
                        trial, context, problem_ks, check_task_manager=True
                    )
                except Exception:
                    break
                # Some fused paths retain dormant stage fields.  Do not run
                # those to max when they consume no physical TMEM.
                if materialized.tmem_columns <= current_materialized.tmem_columns:
                    break
                options = trial_options
                current_materialized = materialized
    repaired = with_options(options, f"repair_{repair_index}")
    if not _supports_any_out_hidden(repaired, context):
        raise ValueError("candidate supports no sweep out_hidden cluster span")
    return repaired
