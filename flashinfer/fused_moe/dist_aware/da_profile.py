"""Selector publication helpers for distribution-aware MoE profiling."""

import os
from dataclasses import dataclass

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch

from . import da_state
from .da_config import DAConfig
from .da_utils import generate_da_distribution_assignments
from ..utils import compute_local_expert_counts_from_plain_ids
from ...autotuner import AutoTuner
from ...tllm_enums import (
    ActivationType,
    DtypeTrtllmGen,
    Fp8QuantizationType,
    WeightLayout,
)

# Keep this value in sync with da_heuristic::kMaxExemplars in
# include/flashinfer/trtllm/fused_moe/da_heuristic_constants.cuh.
MAX_EXEMPLARS = 8
DEFAULT_PROFILE_CONTRACT = "moe_runner_original_profile_v1"
_bundle_has_tactics = False
_bundle_loaded = False


def _dtype_from_context(value: int) -> DtypeTrtllmGen:
    """Recover the enum member stored as an integer in an immutable DA context."""
    return cast(DtypeTrtllmGen, DtypeTrtllmGen._value2member_map_[int(value)])


def _resolve_config(config: Optional[DAConfig]) -> DAConfig:
    return DAConfig() if config is None else config


def _profiling_cache_key_parts(cache_key: Any) -> tuple[str, str, int, Any]:
    """Read both upstream typed cache keys and legacy tuple cache keys."""
    if hasattr(cache_key, "custom_op"):
        return (
            cache_key.custom_op,
            cache_key.runner_class_name,
            cache_key.runner_hash,
            cache_key.nearest_profile,
        )
    op_name, runner_name, runner_hash, profile_key, *_extras = cache_key
    return op_name, runner_name, runner_hash, profile_key


def active_auto_distributions(config: Optional[DAConfig] = None) -> tuple:
    return _resolve_config(config).distributions


def profile_signature(config: Optional[DAConfig] = None) -> tuple[str, ...]:
    return _resolve_config(config).profile_signature


def auto_distribution_sample_count(config: Optional[DAConfig] = None) -> int:
    return _resolve_config(config).distribution_sample_count


def _bundle_path(config: Optional[DAConfig] = None) -> str:
    return _resolve_config(config).bundle_path


def bundle_path(config: Optional[DAConfig] = None) -> str:
    """Return the configured optional DA profile bundle path."""
    return _bundle_path(config)


@dataclass(frozen=True)
class DAProfileBackend:
    """FFI operations required to publish validated DA profile state."""

    get_ffi_moe_op: Callable[[], Any]
    supported_tile_sizes: Callable[..., Sequence[int]]
    capture_backend: Optional[Callable[[], Any]] = None


def merge_same_tactic_exemplars(rows: list, best_idxs: list) -> tuple:
    """Merge same-tactic exemplar rows using normalized centroids."""
    import numpy as np

    by_tactic: dict = {}
    for index, tactic_index in enumerate(best_idxs):
        by_tactic.setdefault(tactic_index, []).append(index)
    if len(by_tactic) == len(best_idxs):
        return rows, best_idxs

    merged_rows, merged_idxs = [], []
    for tactic_index in sorted(by_tactic):
        members = by_tactic[tactic_index]
        if len(members) == 1:
            merged_rows.append(rows[members[0]])
        else:
            centroid = np.mean([rows[index] for index in members], axis=0)
            norm = float(np.linalg.norm(centroid))
            merged_rows.append(centroid / norm if norm > 1e-12 else centroid)
        merged_idxs.append(tactic_index)
    return merged_rows, merged_idxs


def limit_exemplars(
    rows: list, best_idxs: list, max_exemplars: int = MAX_EXEMPLARS
) -> tuple:
    """Bound selector rows while retaining every represented tactic."""
    import numpy as np

    if len(rows) <= max_exemplars:
        return rows, best_idxs

    by_tactic: dict = {}
    for index, tactic_index in enumerate(best_idxs):
        by_tactic.setdefault(tactic_index, []).append(index)

    selected = [positions[len(positions) // 2] for positions in by_tactic.values()]
    if len(selected) > max_exemplars:
        selected = selected[:max_exemplars]
    else:
        remaining = [index for index in range(len(rows)) if index not in set(selected)]
        slots = max_exemplars - len(selected)
        if slots >= len(remaining):
            selected.extend(remaining)
        elif slots > 0:
            picked = np.linspace(0, len(remaining) - 1, slots, dtype=np.int64)
            selected.extend(remaining[int(index)] for index in picked)

    selected = sorted(set(selected))
    return [rows[index] for index in selected], [best_idxs[index] for index in selected]


def deduplicate_body_tactics(tactics: list) -> tuple:
    """Map exemplar tactics onto the smallest equivalent SWITCH body list."""
    body_by_tactic: dict[tuple[int, int], int] = {}
    per_body: list[tuple[int, int]] = []
    body_indices = []
    for tactic in tactics:
        tactic = (int(tactic[0]), int(tactic[1]))
        body_index = body_by_tactic.get(tactic)
        if body_index is None:
            body_index = len(per_body)
            body_by_tactic[tactic] = body_index
            per_body.append(tactic)
        body_indices.append(body_index)
    return per_body, body_indices


def baseline_guard_signature(config: DAConfig) -> Dict[str, Any]:
    """Return the bundle identity for policy inputs that affect final plans."""
    return {
        "enabled": bool(config.baseline_guard),
        "control_overhead_us": float(config.control_overhead_us),
        "margin": float(config.baseline_guard_margin),
    }


def bundle_guard_signature_matches(meta: Dict[str, Any], config: DAConfig) -> bool:
    """Return whether a persisted final plan used compatible guard inputs."""
    return meta.get("baseline_guard_signature") == baseline_guard_signature(config)


def bundle_default_profile_contract_matches(meta: Dict[str, Any]) -> bool:
    """Whether persisted eager tactics came from the marked default profile."""

    return meta.get("default_profile_contract") == DEFAULT_PROFILE_CONTRACT


def unconstrained_candidate_plan(
    switch_tiles: Sequence[int],
    best_idxs: Sequence[int],
    tile_to_tactic: Dict[int, Tuple[int, int]],
) -> Tuple[str, List[Tuple[int, int]]]:
    """Classify the fixed pre-guard assignments after tactic deduplication."""
    assigned = [
        tuple(int(v) for v in tile_to_tactic[int(switch_tiles[int(best_idx)])])
        for best_idx in best_idxs
    ]
    tactics, _body_indices = deduplicate_body_tactics(assigned)
    return ("da_singleton" if len(tactics) == 1 else "da_switch"), tactics


@dataclass(frozen=True)
class DABaselineGuardDecision:
    """Publication policy chosen by the NoDA baseline guard."""

    policy: str
    baseline_tactic: Optional[Tuple[int, int]]
    baseline_ms: Optional[float]
    baseline_worst_ms: Optional[float]
    overhead_ms: float
    margin: float
    dynamic_worst_ms: Optional[float]
    singleton_tactic: Optional[Tuple[int, int]]
    singleton_worst_ms: Optional[float]
    singleton_source: Optional[str]
    admission_applied: bool
    limitation: Optional[str]

    def asdict(self) -> Dict[str, Any]:
        """Return a stable dictionary shape for diagnostics and benchmark CSVs."""

        return {
            "policy": self.policy,
            "baseline_tactic": self.baseline_tactic,
            "baseline_ms": self.baseline_ms,
            "baseline_worst_ms": self.baseline_worst_ms,
            "overhead_ms": self.overhead_ms,
            "margin": self.margin,
            "dynamic_worst_ms": self.dynamic_worst_ms,
            "singleton_tactic": self.singleton_tactic,
            "singleton_worst_ms": self.singleton_worst_ms,
            "singleton_source": self.singleton_source,
            "admission_applied": self.admission_applied,
            "limitation": self.limitation,
        }


def choose_baseline_guard_policy(
    *,
    baseline: Optional[Tuple[Tuple[int, int], float]],
    switch_tiles: Sequence[int],
    best_idxs: Sequence[int],
    per_distribution_latencies: Sequence[Dict[int, float]],
    per_distribution_tactic_latencies: Optional[
        Sequence[Dict[Tuple[int, int], float]]
    ] = None,
    tile_to_tactic: Dict[int, Tuple[int, int]],
    config: DAConfig,
) -> Tuple[str, Optional[Tuple[int, int]], DABaselineGuardDecision]:
    """Choose a post-deduplication DA switch or singleton plan for one bucket.

    Latencies are recorded ordinary-autotuner measurements in milliseconds.
    Admission compares the fixed candidate and fixed NoDA tactic measured on
    the same distribution samples. Switches must win every assignment after
    control overhead. A natural singleton uses the robust worst-case score
    across those samples, matching singleton selection rather than allowing
    one noisy pairwise crossing to force a different execution topology.
    """

    overhead_ms = float(config.control_overhead_us) / 1000.0
    margin = float(config.baseline_guard_margin)
    candidate_tactics: list[Tuple[int, int]] = []
    for best_idx in best_idxs:
        try:
            tile = int(switch_tiles[int(best_idx)])
            raw_tactic = tile_to_tactic[tile]
            candidate_tactics.append((int(raw_tactic[0]), int(raw_tactic[1])))
        except (IndexError, KeyError, TypeError, ValueError):
            continue
    unique_candidate_tactics = list(dict.fromkeys(candidate_tactics))
    natural_singleton = len(unique_candidate_tactics) == 1
    unguarded_policy = "da_singleton" if natural_singleton else "da_switch"

    exact_latencies = per_distribution_tactic_latencies or ()
    assigned: list[tuple[Tuple[int, int], float]] = []
    for dist_idx, best_idx in enumerate(best_idxs):
        if dist_idx >= len(exact_latencies):
            continue
        try:
            tile = int(switch_tiles[int(best_idx)])
            raw_tactic = tile_to_tactic[tile]
            tactic = (int(raw_tactic[0]), int(raw_tactic[1]))
            body_ms = float(exact_latencies[dist_idx][tactic])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        assigned.append((tactic, body_ms))

    unique_assigned = list(dict.fromkeys(tactic for tactic, _ in assigned))
    assignments_complete = (
        len(candidate_tactics) == len(best_idxs)
        and len(assigned) == len(best_idxs)
        and len(per_distribution_latencies) == len(best_idxs)
        and len(exact_latencies) == len(best_idxs)
    )
    natural_singleton_worst = (
        max(body_ms for _tactic, body_ms in assigned)
        if natural_singleton and assigned
        else None
    )

    baseline_costs: list[float] = []
    if baseline is not None:
        baseline_tactic = (int(baseline[0][0]), int(baseline[0][1]))
        for dist_idx in range(len(best_idxs)):
            if dist_idx >= len(exact_latencies):
                continue
            try:
                baseline_costs.append(float(exact_latencies[dist_idx][baseline_tactic]))
            except (KeyError, TypeError, ValueError):
                continue
    baseline_timings_complete = len(baseline_costs) == len(best_idxs)
    baseline_worst_ms = max(baseline_costs) if baseline_costs else None

    limitation = None
    if config.baseline_guard:
        if baseline is None:
            limitation = "missing_noda_baseline"
        elif not exact_latencies or not assignments_complete:
            limitation = "missing_fixed_candidate_timing"
        elif not baseline_timings_complete:
            limitation = "missing_matched_noda_timing"

    if not config.baseline_guard or limitation is not None:
        natural_tactic = unique_candidate_tactics[0] if natural_singleton else None
        return (
            unguarded_policy,
            None,
            DABaselineGuardDecision(
                policy=unguarded_policy,
                baseline_tactic=None if baseline is None else baseline[0],
                baseline_ms=None if baseline is None else float(baseline[1]),
                baseline_worst_ms=baseline_worst_ms,
                overhead_ms=overhead_ms,
                margin=margin,
                dynamic_worst_ms=None,
                singleton_tactic=natural_tactic,
                singleton_worst_ms=natural_singleton_worst,
                singleton_source="profiled" if natural_tactic is not None else None,
                admission_applied=False,
                limitation=limitation,
            ),
        )

    baseline_tactic, baseline_ms = baseline
    baseline_ms = float(baseline_ms)

    switch_costs = [body_ms + overhead_ms for _tactic, body_ms in assigned]
    dynamic_worst = max(switch_costs) if switch_costs else None
    if (
        len(unique_assigned) > 1
        and assignments_complete
        and switch_costs
        and all(
            cost <= baseline_cost * max(0.0, 1.0 - margin)
            for cost, baseline_cost in zip(switch_costs, baseline_costs, strict=True)
        )
    ):
        return (
            "da_switch",
            None,
            DABaselineGuardDecision(
                policy="da_switch",
                baseline_tactic=(int(baseline_tactic[0]), int(baseline_tactic[1])),
                baseline_ms=baseline_ms,
                baseline_worst_ms=baseline_worst_ms,
                overhead_ms=overhead_ms,
                margin=margin,
                dynamic_worst_ms=dynamic_worst,
                singleton_tactic=None,
                singleton_worst_ms=None,
                singleton_source=None,
                admission_applied=True,
                limitation=None,
            ),
        )

    if (
        natural_singleton
        and assignments_complete
        and natural_singleton_worst is not None
        and baseline_worst_ms is not None
        and natural_singleton_worst <= baseline_worst_ms * (1.0 + margin)
    ):
        natural_tactic = unique_assigned[0]
        return (
            "da_singleton",
            None,
            DABaselineGuardDecision(
                policy="da_singleton",
                baseline_tactic=(int(baseline_tactic[0]), int(baseline_tactic[1])),
                baseline_ms=baseline_ms,
                baseline_worst_ms=baseline_worst_ms,
                overhead_ms=overhead_ms,
                margin=margin,
                dynamic_worst_ms=None,
                singleton_tactic=natural_tactic,
                singleton_worst_ms=float(natural_singleton_worst),
                singleton_source="profiled",
                admission_applied=True,
                limitation=None,
            ),
        )

    # Guard rejection is an execution-topology decision, not another DA body
    # search. The measured baseline is the monolithic public-wrapper path, so
    # running its tactic inside a DA metadata graph is not a valid fallback.
    forced_baseline = (int(baseline_tactic[0]), int(baseline_tactic[1]))
    return (
        "noda_baseline_guard",
        forced_baseline,
        DABaselineGuardDecision(
            policy="noda_baseline_guard",
            baseline_tactic=forced_baseline,
            baseline_ms=baseline_ms,
            baseline_worst_ms=baseline_worst_ms,
            overhead_ms=overhead_ms,
            margin=margin,
            dynamic_worst_ms=dynamic_worst,
            singleton_tactic=forced_baseline,
            singleton_worst_ms=baseline_worst_ms,
            singleton_source="noda_baseline",
            admission_applied=True,
            limitation=None,
        ),
    )


def load_knn_v2_bundle(
    bundle: dict,
    bundle_path: str,
    *,
    config: Optional[DAConfig] = None,
    backend: Optional[DAProfileBackend] = None,
    da_context: Optional[da_state.DAMoeContext] = None,
    expected_top_k: Optional[int] = None,
    expected_num_local_experts: Optional[int] = None,
    expected_hidden_size: Optional[int] = None,
    expected_intermediate_size: Optional[int] = None,
    expected_activation_type: Optional[int] = None,
) -> int:
    """Validate a DAKNNv2 bundle before selector publication.

    Upload is deliberately a separate step: callers provide the FFI operation
    only after this function has accepted the bundle's immutable context.
    """
    del bundle_path
    config = _resolve_config(config)
    meta = bundle.get("meta", {})
    verbose = config.verbose
    required_identity = {"schema_version", "device_type", "device_index"}
    if not required_identity.issubset(meta):
        if verbose:
            missing = sorted(required_identity.difference(meta))
            print(f"[DA k-NN v2] skip: missing identity fields {missing}")
        return 0
    if int(meta["schema_version"]) != da_state.CONTEXT_SCHEMA_VERSION:
        if verbose:
            print("[DA k-NN v2] skip: schema_version mismatch")
        return 0

    bundle_signature = tuple(str(name) for name in meta.get("profile_signature", ()))
    if bundle_signature != config.profile_signature:
        if verbose:
            print("[DA k-NN v2] skip: profile_signature mismatch")
        return 0
    if "profile_sample_count" in meta:
        try:
            bundle_sample_count = int(meta["profile_sample_count"])
        except (TypeError, ValueError):
            if verbose:
                print("[DA k-NN v2] skip: invalid profile_sample_count")
            return 0
        if bundle_sample_count != config.distribution_sample_count:
            if verbose:
                print("[DA k-NN v2] skip: profile_sample_count mismatch")
            return 0

    expected_guard_signature = baseline_guard_signature(config)
    if not bundle_guard_signature_matches(meta, config):
        if verbose:
            print(
                "[DA k-NN v2] skip: baseline_guard_signature mismatch "
                f"bundle={meta.get('baseline_guard_signature')!r} "
                f"expected={expected_guard_signature!r}"
            )
        return 0

    num_local = int(meta.get("num_local", 256))
    top_k = int(meta.get("top_k", 8))
    if expected_top_k is not None and int(expected_top_k) != top_k:
        return 0
    if (
        expected_num_local_experts is not None
        and int(expected_num_local_experts) != num_local
    ):
        return 0
    for key, expected in (
        ("hidden_size", expected_hidden_size),
        ("intermediate_size", expected_intermediate_size),
        ("activation_type", expected_activation_type),
    ):
        if expected is not None and key in meta and int(meta[key]) != int(expected):
            return 0

    if da_context is not None:
        expected_meta = {
            "schema_version": int(da_context.schema_version),
            "device_type": da_context.device_type,
            "device_index": int(da_context.device_index),
            "op_name": da_context.op_name,
            "dtype_act": int(da_context.dtype_act),
            "dtype_weights": int(da_context.dtype_weights),
            "quantization_type": int(da_context.quantization_type),
            "weight_layout": int(da_context.weight_layout),
            "use_shuffled_weight": bool(da_context.use_shuffled_weight),
            "use_per_token_scaling": bool(da_context.use_per_token_scaling),
            "has_gemm1_lora_delta": bool(da_context.has_gemm1_lora_delta),
        }
        for key, expected_value in expected_meta.items():
            if key in required_identity and key not in meta:
                return 0
            if key in meta and meta[key] != expected_value:
                return 0

    num_global = int(meta.get("num_global_experts", num_local))
    local_offset = int(meta.get("local_offset", 0))
    if da_context is None:
        da_context = da_state.make_context(
            "flashinfer::trtllm_fp4_block_scale_moe",
            device=f"{meta['device_type']}:{int(meta['device_index'])}",
            dtype_act=DtypeTrtllmGen.E2m1,
            dtype_weights=DtypeTrtllmGen.E2m1,
            quantization_type=Fp8QuantizationType.NoneFp8,
            top_k=top_k,
            num_experts=num_global,
            num_local_experts=num_local,
            local_expert_offset=local_offset,
            hidden_size=int(meta.get("hidden_size", expected_hidden_size or 0)),
            intermediate_size=int(
                meta.get("intermediate_size", expected_intermediate_size or 0)
            ),
            activation_type=int(
                meta.get(
                    "activation_type", expected_activation_type or ActivationType.Swiglu
                )
            ),
            weight_layout=WeightLayout.MajorK,
            use_shuffled_weight=True,
        )
    if backend is None:
        return 0

    import numpy as np

    exemplars_by_bucket = bundle.get("exemplars", {})
    plans_by_bucket = bundle.get("plans", {}) or {}
    global _bundle_has_tactics
    tactic_table = bundle.get("tactic_table", {}) or {}
    if tactic_table:
        _bundle_has_tactics = True
        da_state.BUNDLE_TACTIC_CONTEXTS.add(da_context)

    offsets = list(range(0, max(num_global, num_local), max(1, num_local)))
    if local_offset not in offsets:
        offsets.append(local_offset)
    if config.exemplar_offsets:
        offsets = list(config.exemplar_offsets)

    offset_contexts = {
        da_state.context_with_offset(da_context, int(offset))
        for offset in sorted(set(offsets))
    }
    for eager_key in tuple(da_state.BUNDLE_EAGER_TACTICS):
        if eager_key[1] in offset_contexts:
            da_state.BUNDLE_EAGER_TACTICS.pop(eager_key, None)

    eager_tactics = bundle.get("eager_tactics", {}) or {}
    if eager_tactics and bundle_default_profile_contract_matches(meta):
        for bucket_key, record in eager_tactics.items():
            try:
                tactic = tuple(int(value) for value in record["tactic"])
                if len(tactic) != 2:
                    raise ValueError
                time_ms = float(record["time_ms"])
                bucket = int(bucket_key)
            except (KeyError, TypeError, ValueError):
                continue
            for offset_context in offset_contexts:
                da_state.BUNDLE_EAGER_TACTICS[
                    da_state.cache_key(offset_context, bucket)
                ] = (tactic, time_ms)

    uploaded = 0
    for bucket_key, raw_plan in sorted(
        plans_by_bucket.items(), key=lambda item: int(item[0])
    ):
        plan = dict(raw_plan)
        policy = plan.get("final_policy", plan.get("policy"))
        if policy != "noda_baseline_guard":
            continue
        try:
            bucket = int(bucket_key)
            baseline_tactic = tuple(int(v) for v in plan["baseline_tactic"])
            if len(baseline_tactic) != 2:
                raise ValueError
        except (KeyError, TypeError, ValueError):
            if verbose:
                print(
                    f"[DA k-NN v2] skip: invalid guarded NoDA plan "
                    f"for bucket {bucket_key}",
                    flush=True,
                )
            return 0
        plan["policy"] = "noda_baseline_guard"
        plan["final_policy"] = "noda_baseline_guard"
        plan["final_tactics"] = [baseline_tactic]
        for offset in sorted(set(offsets)):
            offset_context = da_state.context_with_offset(da_context, int(offset))
            offset_key = da_state.cache_key(offset_context, bucket)
            da_state.BASELINE_GUARD_DECISIONS[offset_key] = dict(plan)
            da_state.PER_TILE_TACTICS.pop(offset_key, None)
            da_state.PER_BODY_TACTICS.pop(offset_key, None)
        uploaded += 1

    for bucket_key, exemplars in sorted(exemplars_by_bucket.items()):
        if not exemplars:
            continue
        rows = []
        for exemplar in exemplars:
            row = np.asarray(exemplar["norm_vec"], dtype=np.float32)
            if row.shape[0] < num_local:
                row = np.pad(row, (0, num_local - row.shape[0]))
            rows.append(row[:num_local])
        tactics = [
            (int(exemplar["tile_shape"]), int(exemplar["kernel_id"]))
            for exemplar in exemplars
        ]
        if config.merge_same_tactic_exemplars:
            rows, tactics = merge_same_tactic_exemplars(rows, tactics)
        rows, tactics = limit_exemplars(rows, tactics)
        per_body, body_indices = deduplicate_body_tactics(tactics)
        plan = dict(plans_by_bucket.get(str(bucket_key), {}))
        if not plan:
            plan = {
                "policy": "da_singleton" if len(per_body) == 1 else "da_switch",
                "candidate_policy": (
                    "da_singleton" if len(per_body) == 1 else "da_switch"
                ),
                "candidate_tactics": list(per_body),
                "final_policy": "da_singleton" if len(per_body) == 1 else "da_switch",
                "final_tactics": list(per_body),
                "singleton_tactic": per_body[0] if len(per_body) == 1 else None,
                "singleton_source": "profiled" if len(per_body) == 1 else None,
            }
        if plan.get("final_policy", plan.get("policy")) == "noda_baseline_guard":
            continue
        if plan.get("policy") not in {
            "da_switch",
            "da_singleton",
        }:
            if verbose:
                print(
                    f"[DA k-NN v2] skip: invalid plan policy "
                    f"{plan.get('policy')!r} for bucket {bucket_key}"
                )
            return 0
        flat = np.stack(rows).astype(np.float32).reshape(-1).copy()
        flat_tensor = torch.from_numpy(flat).to(
            device=torch.device(da_context.device_type, da_context.device_index)
        )
        tile_shapes = [tile for tile, _ in tactics]
        kernel_ids = [kernel for _, kernel in tactics]
        tile_sizes = sorted(set(tile_shapes))
        tile_map: Dict[int, Tuple[int, int]] = {}
        for tactic in per_body:
            tile_map.setdefault(tactic[0], tactic)
        for offset in sorted(set(offsets)):
            offset_context = da_state.context_with_offset(da_context, int(offset))
            upload_and_publish_selector_tactics(
                backend.get_ffi_moe_op(),
                offset_context,
                flat_tensor,
                body_indices,
                tile_shapes,
                kernel_ids,
                tile_sizes,
                num_local,
                int(offset),
                top_k,
                int(bucket_key),
                per_tile_tactics=tile_map,
                per_body_tactics=per_body,
            )
            da_state.BASELINE_GUARD_DECISIONS[
                da_state.cache_key(offset_context, int(bucket_key))
            ] = dict(plan)
        uploaded += 1
    return uploaded


def bundle_has_tactics() -> bool:
    """Whether the last compatible bundle supplied an explicit tactic table."""
    return _bundle_has_tactics


def maybe_load_existing_bundle(
    tuner: Any,
    backend: DAProfileBackend,
    *,
    da_context: da_state.DAMoeContext,
    hidden_states: torch.Tensor,
    top_k: int,
    intermediate_size: int,
    num_local_experts: int,
    activation_type: int,
    debug_log: Callable[[str], None],
    config: Optional[DAConfig] = None,
) -> bool:
    """Load a compatible persisted bundle before the DA profile sweep."""
    global _bundle_loaded

    config = _resolve_config(config)
    path = config.bundle_path
    if (
        da_context in da_state.BUNDLE_LOADED_CONTEXTS
        or not config.enabled
        or not path
        or not os.path.isfile(path)
        or not tuner.is_tuning_mode
        or torch.cuda.is_current_stream_capturing()
    ):
        return False

    _bundle_loaded = True
    da_state.BUNDLE_LOADED_CONTEXTS.add(da_context)
    try:
        hidden_size = int(hidden_states.shape[-1])
        if hidden_states.dtype == torch.uint8:
            hidden_size *= 2
        import pickle

        with open(path, "rb") as bundle_file:
            bundle = pickle.load(bundle_file)
        count = load_knn_v2_bundle(
            bundle,
            path,
            config=config,
            backend=backend,
            da_context=da_context,
            expected_top_k=top_k,
            expected_num_local_experts=num_local_experts,
            expected_hidden_size=hidden_size,
            expected_intermediate_size=intermediate_size,
            expected_activation_type=activation_type,
        )
        if count <= 0:
            raise ValueError("bundle contains no compatible exemplar cells")
        debug_log(f"kNN bundle early-loaded cells={count} path={path}")
        if config.verbose:
            suffix = " (skipping DA autotune sweep)" if bundle_has_tactics() else ""
            print(
                f"[DA k-NN] early-loaded {count} exemplar cells from {path}{suffix}",
                flush=True,
            )
        return True
    except Exception as error:
        da_state.BUNDLE_LOADED_CONTEXTS.discard(da_context)
        _bundle_loaded = bool(da_state.BUNDLE_LOADED_CONTEXTS)
        debug_log(f"kNN bundle early load failed path={path} error={error}")
        print(f"[DA k-NN] early load failed: {error}", flush=True)
        return False


def maybe_prepare_bundle(
    tuner: Any,
    backend: DAProfileBackend,
    *,
    da_context: da_state.DAMoeContext,
    hidden_states: torch.Tensor,
    top_k: int,
    intermediate_size: int,
    num_local_experts: int,
    activation_type: int,
    register_live_profile: Callable[[], None],
    debug_log: Callable[[str], None],
    config: Optional[DAConfig] = None,
) -> None:
    """Load a compatible selector bundle or register live profiling once."""
    global _bundle_loaded
    config = _resolve_config(config)
    if not _bundle_loaded:
        da_state.BUNDLE_LOADED_CONTEXTS.clear()
        da_state.BUNDLE_TACTIC_CONTEXTS.clear()
    if (
        da_context in da_state.BUNDLE_LOADED_CONTEXTS
        or not config.enabled
        or torch.cuda.is_current_stream_capturing()
    ):
        return

    path = config.bundle_path
    if path and os.path.isfile(path):
        _bundle_loaded = True
        da_state.BUNDLE_LOADED_CONTEXTS.add(da_context)
        try:
            hidden_size = int(hidden_states.shape[-1])
            if hidden_states.dtype == torch.uint8:
                hidden_size *= 2
            import pickle

            with open(path, "rb") as bundle_file:
                bundle = pickle.load(bundle_file)
            count = load_knn_v2_bundle(
                bundle,
                path,
                config=config,
                backend=backend,
                da_context=da_context,
                expected_top_k=top_k,
                expected_num_local_experts=num_local_experts,
                expected_hidden_size=hidden_size,
                expected_intermediate_size=intermediate_size,
                expected_activation_type=activation_type,
            )
            if count <= 0:
                raise ValueError("bundle contains no compatible exemplar cells")
            debug_log(f"kNN bundle loaded cells={count} path={path}")
            return
        except Exception as error:
            da_state.BUNDLE_LOADED_CONTEXTS.discard(da_context)
            _bundle_loaded = bool(da_state.BUNDLE_LOADED_CONTEXTS)
            debug_log(f"kNN bundle load failed path={path} error={error}")

    if tuner.is_tuning_mode:
        _bundle_loaded = True
        da_state.BUNDLE_LOADED_CONTEXTS.add(da_context)
        register_live_profile()


def cached_profile_tactics(
    da_context: da_state.DAMoeContext,
    top_k: int,
    config: Optional[DAConfig] = None,
) -> Dict[Tuple[int, int], Dict[int, float]]:
    """Return finite autotuner latencies keyed by token bucket and distribution."""
    import numpy as np

    latencies: Dict[Tuple[int, int], Dict[int, float]] = {}
    tuner = AutoTuner.get()
    for cache_key, (tactic, _profile) in tuner.profiling_cache.items():
        op_name, runner_name, _runner_hash, profile_key = _profiling_cache_key_parts(
            cache_key
        )
        if op_name != da_context.op_name or runner_name != "MoERunner":
            continue
        if not isinstance(profile_key, tuple) or len(profile_key) != 2:
            continue
        shapes, value_buckets = profile_key
        if not value_buckets or len(value_buckets) < 2:
            continue
        try:
            bucket = int(shapes[0][0])
            if len(shapes) >= 3 and int(shapes[2][1]) != int(top_k):
                continue
            distribution = int(value_buckets[1])
            tile = int(tactic[0])
        except (IndexError, TypeError, ValueError):
            continue
        if not 0 <= distribution < len(active_auto_distributions(config)):
            continue
        latency = float(tuner.profiling_time_cache.get(cache_key, float("inf")))
        if np.isfinite(latency):
            latencies.setdefault((bucket, distribution), {})[tile] = latency
    return latencies


def cached_profile_tactic_latencies(
    da_context: da_state.DAMoeContext,
    top_k: int,
    *,
    runner_hash: Optional[int] = None,
    config: Optional[DAConfig] = None,
) -> Dict[Tuple[int, int], Dict[Tuple[int, int], float]]:
    """Return every finite fixed-tactic timing by bucket and distribution."""
    import numpy as np

    latencies: Dict[Tuple[int, int], Dict[Tuple[int, int], float]] = {}
    tuner = AutoTuner.get()
    for (
        profile_cache_key,
        raw_tactic,
    ), latency in tuner.profiling_tactic_time_cache.items():
        op_name, runner_name, cached_hash, profile_key = _profiling_cache_key_parts(
            profile_cache_key
        )
        if op_name != da_context.op_name or runner_name != "MoERunner":
            continue
        if runner_hash is not None and int(cached_hash) != int(runner_hash):
            continue
        if not isinstance(profile_key, tuple) or len(profile_key) != 2:
            continue
        shapes, value_buckets = profile_key
        if not value_buckets or len(value_buckets) < 2:
            continue
        try:
            bucket = int(shapes[0][0])
            if len(shapes) >= 3 and int(shapes[2][1]) != int(top_k):
                continue
            profile_tile = int(value_buckets[0])
            distribution = int(value_buckets[1])
            tactic = (int(raw_tactic[0]), int(raw_tactic[1]))
        except (IndexError, TypeError, ValueError):
            continue
        if profile_tile != tactic[0]:
            continue
        if not 0 <= distribution < len(active_auto_distributions(config)):
            continue
        latency = float(latency)
        if not np.isfinite(latency):
            continue
        per_distribution = latencies.setdefault((bucket, distribution), {})
        previous = per_distribution.get(tactic)
        if previous is None or latency < previous:
            per_distribution[tactic] = latency
    return latencies


def populate_per_tile_tactics_from_autotune(
    backend: DAProfileBackend,
    tuner: AutoTuner,
    custom_op: str,
    *,
    da_context: da_state.DAMoeContext,
    runner_hash: int,
    config: Optional[DAConfig] = None,
) -> None:
    """Extract DAKNNv2 per-(bucket, tile_n) body tactics from AutoTuner."""
    try:
        _ = backend.get_ffi_moe_op()
    except Exception:
        return

    verbose = _resolve_config(config).verbose
    stats = {
        "total": 0,
        "reject_op_or_runner": 0,
        "reject_runner_hash": 0,
        "reject_profile_key_shape": 0,
        "reject_no_value_buckets": 0,
        "reject_tactic_shape": 0,
        "added": 0,
    }

    per_bucket_tile_best: Dict[int, Dict[int, Tuple[int, float]]] = {}
    for cache_key, (tactic, _profile) in tuner.profiling_cache.items():
        stats["total"] += 1
        op_name, runner_name, cached_hash, profile_key = _profiling_cache_key_parts(
            cache_key
        )
        if op_name != custom_op or runner_name != "MoERunner":
            stats["reject_op_or_runner"] += 1
            continue
        if cached_hash != runner_hash:
            stats["reject_runner_hash"] += 1
            continue
        if not isinstance(profile_key, tuple) or len(profile_key) != 2:
            stats["reject_profile_key_shape"] += 1
            continue
        shapes, value_buckets = profile_key
        if not value_buckets:
            stats["reject_no_value_buckets"] += 1
            continue
        if tuple(value_buckets) == da_state.DEFAULT_PROFILE_VALUE_BUCKETS:
            continue
        tactic_value = cast(Any, tactic)
        try:
            tile = int(tactic_value[0])
            kernel_config = int(tactic_value[1])
            if len(tactic_value) != 2:
                raise ValueError
        except (TypeError, IndexError, ValueError):
            stats["reject_tactic_shape"] += 1
            continue
        num_tokens_bucket = int(shapes[0][0])
        valid = backend.supported_tile_sizes(
            num_tokens_bucket,
            top_k=da_context.top_k,
            num_local_experts=da_context.num_local_experts,
            local_expert_offset=da_context.local_expert_offset,
            dtype_act=_dtype_from_context(da_context.dtype_act),
            dtype_weights=_dtype_from_context(da_context.dtype_weights),
            quantization_type=Fp8QuantizationType(da_context.quantization_type),
            da_context=None,
        )
        if tile not in valid:
            stats["reject_tactic_shape"] += 1
            continue
        ms = float(tuner.profiling_time_cache.get(cache_key, float("inf")))
        per_tile = per_bucket_tile_best.setdefault(num_tokens_bucket, {})
        previous = per_tile.get(tile)
        if previous is None or ms < previous[1]:
            per_tile[tile] = (kernel_config, ms)
        stats["added"] += 1

    per_bucket_tiles: Dict[int, Dict[int, Tuple[int, int]]] = {
        bucket: {tile: (tile, cfg_ms[0]) for tile, cfg_ms in per_tile.items()}
        for bucket, per_tile in per_bucket_tile_best.items()
    }

    if verbose:
        sample = {
            bucket: sorted(per_bucket_tiles[bucket].keys())
            for bucket in list(per_bucket_tiles.keys())[:5]
        }
        print(
            f"[DA populate] stats={stats} "
            f"buckets={sorted(per_bucket_tiles.keys())[:8]} "
            f"sample_tiles={sample}",
            flush=True,
        )

    for bucket, tile_map in per_bucket_tiles.items():
        config_key = da_state.cache_key(da_context, int(bucket))
        existing = da_state.PER_TILE_TACTICS.get(config_key, {})
        existing.update(tile_map)
        da_state.PER_TILE_TACTICS[config_key] = existing


def best_static_tactic_from_profiles(
    tuner: AutoTuner,
    custom_op: str,
    runner_name: str,
    runner_hash: int,
    num_tokens_bucket: int,
    *,
    da_context: da_state.DAMoeContext,
) -> Optional[Tuple[Tuple[int, int], float]]:
    """Return the fastest measured NoDA tactic for a shape bucket."""
    cache_key = (custom_op, int(runner_hash), int(num_tokens_bucket), da_context)
    if cache_key in da_state.STATIC_FALLBACK_TACTICS:
        return da_state.STATIC_FALLBACK_TACTICS[cache_key]
    bundle_fallback = da_state.BUNDLE_EAGER_TACTICS.get(
        da_state.cache_key(da_context, int(num_tokens_bucket))
    )
    if bundle_fallback is not None:
        da_state.STATIC_FALLBACK_TACTICS[cache_key] = bundle_fallback
        return bundle_fallback

    best_exact: Optional[Tuple[Tuple[int, int], float]] = None
    best_compatible: Optional[Tuple[Tuple[int, int], float]] = None
    for profile_cache_key, (tactic, _profile) in tuner.profiling_cache.items():
        op_name, cached_runner_name, cached_runner_hash, profile_key = (
            _profiling_cache_key_parts(profile_cache_key)
        )
        if op_name != custom_op or cached_runner_name != runner_name:
            continue
        if not isinstance(profile_key, tuple):
            continue
        # Shape-only cache keys store ``shapes`` directly. Value-aware keys
        # wrap them as ``(shapes, value_buckets)``.
        shapes: Any
        value_buckets: Any
        if (
            len(profile_key) == 2
            and isinstance(profile_key[0], tuple)
            and profile_key[0]
            and isinstance(profile_key[0][0], tuple)
        ):
            shapes, value_buckets = profile_key
        else:
            shapes, value_buckets = profile_key, ()
        if value_buckets:
            continue
        try:
            if int(shapes[0][0]) != int(num_tokens_bucket):
                continue
            if int(shapes[0][1]) != int(da_context.hidden_size):
                continue
            if len(shapes) > 2 and len(shapes[2]) > 1:
                if int(shapes[2][1]) != int(da_context.top_k):
                    continue
            tactic_value = cast(Any, tactic)
            if len(tactic_value) != 2:
                continue
            candidate_tactic = (int(tactic_value[0]), int(tactic_value[1]))
        except (TypeError, IndexError, ValueError):
            continue
        time_ms = float(tuner.profiling_time_cache.get(profile_cache_key, float("inf")))
        if not np.isfinite(time_ms):
            continue
        candidate = (candidate_tactic, time_ms)
        if cached_runner_hash == runner_hash:
            if best_exact is None or time_ms < best_exact[1]:
                best_exact = candidate
        elif best_compatible is None or time_ms < best_compatible[1]:
            # The NoDA and DA runners can differ only in DA policy fields and
            # therefore hash differently. The shape/context checks above keep
            # this fallback scoped to the same MoE problem.
            best_compatible = candidate

    best = best_exact or best_compatible

    da_state.STATIC_FALLBACK_TACTICS[cache_key] = best
    return best


def validate_selector_upload(
    exemplar_norm_flat: torch.Tensor,
    exemplar_body_idx: Sequence[int],
    exemplar_tile_shapes: Sequence[int],
    exemplar_kernel_ids: Sequence[int],
    tile_sizes: Sequence[int],
    num_local_experts: int,
) -> List[Tuple[int, int]]:
    num_exemplars = len(exemplar_body_idx)
    if not (num_exemplars == len(exemplar_tile_shapes) == len(exemplar_kernel_ids)):
        raise ValueError("DA selector exemplar arrays must have the same length")
    if not 1 <= num_exemplars <= MAX_EXEMPLARS:
        raise ValueError(
            "DA selector exemplar count must be in "
            f"[1, {MAX_EXEMPLARS}], got {num_exemplars}"
        )
    if int(num_local_experts) <= 0:
        raise ValueError("DA selector num_local_experts must be positive")
    if exemplar_norm_flat.dtype != torch.float32:
        raise ValueError("DA selector exemplar tensor must have dtype float32")
    expected_numel = num_exemplars * int(num_local_experts)
    if int(exemplar_norm_flat.numel()) != expected_numel:
        raise ValueError(
            "DA selector exemplar tensor size mismatch: "
            f"got {exemplar_norm_flat.numel()}, expected {expected_numel}"
        )

    candidates = tuple(int(tile) for tile in tile_sizes)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("DA selector candidate tiles must be non-empty and unique")
    if any(tile <= 0 for tile in candidates):
        raise ValueError("DA selector candidate tiles must be positive")
    candidate_set = set(candidates)
    body_to_tactic: Dict[int, Tuple[int, int]] = {}
    tactic_to_body: Dict[Tuple[int, int], int] = {}
    for index, (body, tile, kernel) in enumerate(
        zip(
            exemplar_body_idx,
            exemplar_tile_shapes,
            exemplar_kernel_ids,
            strict=True,
        )
    ):
        body = int(body)
        tactic = (int(tile), int(kernel))
        if body < 0:
            raise ValueError(f"DA selector body index {index} must be non-negative")
        if tactic[0] not in candidate_set:
            raise ValueError(
                f"DA selector exemplar tile {tactic[0]} is not a candidate tile"
            )
        if tactic[1] < 0:
            raise ValueError("DA selector kernel IDs must be non-negative")
        previous_tactic = body_to_tactic.setdefault(body, tactic)
        if previous_tactic != tactic:
            raise ValueError("DA selector body index must map to one tactic")
        previous_body = tactic_to_body.setdefault(tactic, body)
        if previous_body != body:
            raise ValueError("DA selector tactic must map to one body")

    expected_bodies = set(range(len(body_to_tactic)))
    if set(body_to_tactic) != expected_bodies:
        raise ValueError("DA selector body indices must be contiguous from zero")
    return [body_to_tactic[index] for index in range(len(body_to_tactic))]


def upload_exemplars_for_context(
    moe_op: Any,
    da_context: da_state.DAMoeContext,
    exemplar_norm_flat: torch.Tensor,
    exemplar_body_idx: Sequence[int],
    exemplar_tile_shapes: Sequence[int],
    exemplar_kernel_ids: Sequence[int],
    tile_sizes: Sequence[int],
    num_local_experts: int,
    local_expert_offset: int,
    top_k: int,
    num_tokens_bucket: int,
) -> None:
    validate_selector_upload(
        exemplar_norm_flat,
        exemplar_body_idx,
        exemplar_tile_shapes,
        exemplar_kernel_ids,
        tile_sizes,
        num_local_experts,
    )
    expected_device = torch.device(da_context.device_type, da_context.device_index)
    if exemplar_norm_flat.device != expected_device:
        raise ValueError(
            f"DA exemplar tensor device {exemplar_norm_flat.device} does not "
            f"match selector context {expected_device}"
        )
    with torch.cuda.device(expected_device):
        moe_op.da_upload_knn_exemplars_with_handle(
            da_state.selector_handle(da_context),
            exemplar_norm_flat,
            exemplar_body_idx,
            exemplar_tile_shapes,
            exemplar_kernel_ids,
            tile_sizes,
            int(num_local_experts),
            int(local_expert_offset),
            int(top_k),
            int(num_tokens_bucket),
        )


def upload_and_publish_selector_tactics(
    moe_op: Any,
    da_context: da_state.DAMoeContext,
    exemplar_norm_flat: torch.Tensor,
    exemplar_body_idx: Sequence[int],
    exemplar_tile_shapes: Sequence[int],
    exemplar_kernel_ids: Sequence[int],
    tile_sizes: Sequence[int],
    num_local_experts: int,
    local_expert_offset: int,
    top_k: int,
    num_tokens_bucket: int,
    *,
    per_tile_tactics: Optional[Dict[int, Tuple[int, int]]],
    per_body_tactics: Sequence[Tuple[int, int]],
) -> None:
    validated_bodies = validate_selector_upload(
        exemplar_norm_flat,
        exemplar_body_idx,
        exemplar_tile_shapes,
        exemplar_kernel_ids,
        tile_sizes,
        num_local_experts,
    )
    normalized_bodies = [
        (int(tactic[0]), int(tactic[1])) for tactic in per_body_tactics
    ]
    if normalized_bodies != validated_bodies:
        raise ValueError(
            "DA selector published body tactics do not match exemplar mappings"
        )
    normalized_tiles = (
        None
        if per_tile_tactics is None
        else {
            int(tile): (int(tactic[0]), int(tactic[1]))
            for tile, tactic in per_tile_tactics.items()
        }
    )
    if normalized_tiles is not None and any(
        tile != tactic[0] or tactic not in validated_bodies
        for tile, tactic in normalized_tiles.items()
    ):
        raise ValueError(
            "DA selector published tile tactics do not match exemplar mappings"
        )

    upload_exemplars_for_context(
        moe_op,
        da_context,
        exemplar_norm_flat,
        exemplar_body_idx,
        exemplar_tile_shapes,
        exemplar_kernel_ids,
        tile_sizes,
        num_local_experts,
        local_expert_offset,
        top_k,
        num_tokens_bucket,
    )
    config_key = da_state.cache_key(da_context, int(num_tokens_bucket))
    if normalized_tiles is not None:
        da_state.PER_TILE_TACTICS[config_key] = normalized_tiles
    da_state.PER_BODY_TACTICS[config_key] = normalized_bodies


def auto_profile_knn_exemplars(
    backend: DAProfileBackend,
    moe_op,
    *,
    da_context: da_state.DAMoeContext,
    hidden_states: torch.Tensor,
    hidden_states_scale: Optional[torch.Tensor],
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    per_token_scale: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    num_local_experts: int,
    routed_scaling_factor: Optional[float],
    use_routing_scales_on_input: bool,
    routing_method_type: int,
    activation_type: int,
    tune_max_num_tokens: int,
    enable_pdl: bool,
    runner: Any,
    runner_hash: Optional[int] = None,
    config: Optional[DAConfig] = None,
) -> int:
    """Build kNN exemplars from DA autotune results.

    Builds a mini Stage-B from the autotuner's value-bucket profiling cache:
    for each (bucket, distribution, tile), pick the best tile per
    (bucket, dist) and upload the resulting exemplars for the capture-aware
    fast path.

    Returns the number of (bucket) cells uploaded, or 0 on failure.
    """
    config = _resolve_config(config)
    _verbose = config.verbose
    if _verbose:
        print("[DA k-NN auto-profile] starting inline profiling", flush=True)

    hidden_size = hidden_states.shape[-1]
    if hidden_states.dtype == torch.uint8:
        hidden_size *= 2
    device = hidden_states.device
    dtype_act = _dtype_from_context(da_context.dtype_act)
    quant_type = Fp8QuantizationType(da_context.quantization_type)

    buckets_with_tiles: Dict[int, Dict[int, Tuple[int, int]]] = {}
    for config_key, tile_map in da_state.PER_TILE_TACTICS.items():
        bucket, ck_context = config_key
        if ck_context == da_context:
            buckets_with_tiles[bucket] = tile_map

    if not buckets_with_tiles:
        if _verbose:
            print(
                "[DA k-NN auto-profile] no per-tile tactics available, skipping",
                flush=True,
            )
        return 0

    tie_eps = config.tie_epsilon
    warmup = config.auto_warmup
    iters = config.auto_iters
    n_uploaded = 0
    bundle_exemplars: Dict[str, List[Dict[str, Any]]] = {}
    bundle_plans: Dict[str, Dict[str, Any]] = {}

    num_global = max(num_experts, num_local_experts)
    auto_offsets = sorted(set(range(0, num_global, max(1, num_local_experts))))
    if local_expert_offset not in auto_offsets:
        auto_offsets.append(local_expert_offset)
        auto_offsets = sorted(set(auto_offsets))

    def _finish_bucket(
        bucket: int,
        switch_tiles: List[int],
        rows: List[np.ndarray],
        best_idxs: List[int],
        per_distribution_latencies: List[Dict[int, float]],
        per_distribution_tactic_latencies: List[Dict[Tuple[int, int], float]],
    ) -> int:
        nonlocal n_uploaded

        if not rows:
            return 0
        config_key = da_state.cache_key(da_context, int(bucket))
        tile_map = buckets_with_tiles[int(bucket)]
        candidate_policy, candidate_tactics = unconstrained_candidate_plan(
            switch_tiles,
            best_idxs,
            tile_map,
        )
        static_baseline = (
            None
            if runner_hash is None
            else best_static_tactic_from_profiles(
                tuner,
                da_context.op_name,
                "MoERunner",
                int(runner_hash),
                int(bucket),
                da_context=da_context,
            )
        )
        policy, forced_singleton_tactic, decision = choose_baseline_guard_policy(
            baseline=static_baseline,
            switch_tiles=switch_tiles,
            best_idxs=best_idxs,
            per_distribution_latencies=per_distribution_latencies,
            per_distribution_tactic_latencies=per_distribution_tactic_latencies,
            tile_to_tactic=tile_map,
            config=config,
        )
        decision_dict = decision.asdict()
        decision_dict.update(
            {
                "candidate_policy": candidate_policy,
                "candidate_tactics": list(candidate_tactics),
                "final_policy": policy,
                "control_overhead_source": "pre_recorded_calibration",
                "baseline_guard_signature": baseline_guard_signature(config),
            }
        )

        if policy == "noda_baseline_guard":
            if forced_singleton_tactic is None:
                raise RuntimeError("guarded NoDA policy requires a baseline tactic")
            decision_dict["final_tactics"] = [
                (
                    int(forced_singleton_tactic[0]),
                    int(forced_singleton_tactic[1]),
                )
            ]
            bundle_plans[str(int(bucket))] = dict(decision_dict)
            for off in auto_offsets:
                offset_context = da_state.context_with_offset(da_context, int(off))
                offset_key = da_state.cache_key(offset_context, int(bucket))
                da_state.BASELINE_GUARD_DECISIONS[offset_key] = dict(decision_dict)
                da_state.PER_TILE_TACTICS.pop(offset_key, None)
                da_state.PER_BODY_TACTICS.pop(offset_key, None)
            n_uploaded += 1
            if _verbose:
                print(
                    f"[DA k-NN auto-profile] bucket={bucket} "
                    f"policy={policy} tactic={forced_singleton_tactic} "
                    "DA metadata upload skipped",
                    flush=True,
                )
            return 1

        if forced_singleton_tactic is not None:
            raise RuntimeError(
                f"DA policy {policy!r} unexpectedly forced tactic "
                f"{forced_singleton_tactic!r}"
            )
        if len(set(best_idxs)) == 1:
            rows = rows[:1]
            best_idxs = best_idxs[:1]
        elif config.merge_same_tactic_exemplars:
            rows, best_idxs = merge_same_tactic_exemplars(rows, best_idxs)
        original_count = len(rows)
        rows, best_idxs = limit_exemplars(rows, best_idxs)
        if _verbose and len(rows) != original_count:
            print(
                f"[DA k-NN auto-profile] bucket={bucket} limited exemplars "
                f"from {original_count} to {len(rows)}",
                flush=True,
            )
        tile_shapes = [int(switch_tiles[bi]) for bi in best_idxs]
        kernel_ids = [int(tile_map[int(tile_shape)][1]) for tile_shape in tile_shapes]

        flat = np.stack(rows).astype(np.float32).reshape(-1).copy()
        upload_device = torch.device(da_context.device_type, da_context.device_index)
        flat_t = torch.from_numpy(flat).contiguous().to(device=upload_device)
        n_ex = len(rows)
        per_body, body_idxs = deduplicate_body_tactics(
            list(zip(tile_shapes, kernel_ids, strict=True))
        )
        expected_policy = "da_singleton" if len(per_body) == 1 else "da_switch"
        if policy != expected_policy:
            raise RuntimeError(
                f"DA plan classification {policy!r} disagrees with "
                f"post-deduplication bodies {per_body!r}"
            )
        decision_dict["final_tactics"] = list(per_body)
        da_state.BASELINE_GUARD_DECISIONS[config_key] = dict(decision_dict)
        bundle_plans[str(int(bucket))] = dict(decision_dict)
        published_tiles = sorted({int(tactic[0]) for tactic in per_body})
        published_tile_map = {int(tactic[0]): tactic for tactic in per_body}
        bundle_exemplars[str(int(bucket))] = [
            {
                "norm_vec": np.asarray(rows[i], dtype=np.float32),
                "tile_shape": int(tile_shapes[i]),
                "kernel_id": int(kernel_ids[i]),
            }
            for i in range(n_ex)
        ]
        for off in auto_offsets:
            offset_context = da_state.context_with_offset(da_context, int(off))
            da_state.BASELINE_GUARD_DECISIONS[
                da_state.cache_key(offset_context, int(bucket))
            ] = dict(decision_dict)
            upload_and_publish_selector_tactics(
                moe_op,
                offset_context,
                flat_t,
                body_idxs,
                tile_shapes,
                kernel_ids,
                published_tiles,
                int(num_local_experts),
                int(off),
                int(top_k),
                int(bucket),
                per_tile_tactics=published_tile_map,
                per_body_tactics=per_body,
            )
        n_uploaded += 1
        if _verbose:
            print(
                f"[DA k-NN auto-profile] bucket={bucket} "
                f"policy={policy} "
                f"tiles={published_tiles} "
                f"exemplars={len(rows)} "
                f"best_idxs={best_idxs}",
                flush=True,
            )
        return 1

    cached_latencies: Dict[Tuple[int, int], Dict[int, float]] = {}
    tuner = AutoTuner.get()
    for cache_key, (tactic, _profile) in tuner.profiling_cache.items():
        op_name, runner_name, cached_hash, profile_key = _profiling_cache_key_parts(
            cache_key
        )
        if op_name != da_context.op_name or runner_name != "MoERunner":
            continue
        if runner_hash is not None and int(cached_hash) != int(runner_hash):
            continue
        if not isinstance(profile_key, tuple) or len(profile_key) != 2:
            continue
        shapes, value_buckets = profile_key
        if not value_buckets or len(value_buckets) < 2:
            continue
        try:
            bucket = int(shapes[0][0])
            if bucket not in buckets_with_tiles:
                continue
            if (
                len(shapes) >= 3
                and len(shapes[2]) >= 2
                and int(shapes[2][1]) != int(top_k)
            ):
                continue
            tile = int(tactic[0])
            dist_idx = int(value_buckets[1])
        except (TypeError, IndexError, ValueError):
            continue
        if dist_idx < 0 or dist_idx >= len(active_auto_distributions(config)):
            continue
        ms = float(tuner.profiling_time_cache.get(cache_key, float("inf")))
        if not np.isfinite(ms):
            continue
        cached_latencies.setdefault((bucket, dist_idx), {})[tile] = ms

    cached_tactic_latencies = cached_profile_tactic_latencies(
        da_context,
        top_k,
        runner_hash=runner_hash,
        config=config,
    )

    # The ordinary autotuner cache is the only source of candidate assignments
    # and body costs. Guard policy is evaluated after those assignments are
    # fixed and never triggers a second profiling discipline.
    if cached_latencies:
        for bucket in sorted(buckets_with_tiles):
            switch_tiles = sorted(
                t
                for t in backend.supported_tile_sizes(
                    bucket,
                    top_k=top_k,
                    num_local_experts=num_local_experts,
                    local_expert_offset=local_expert_offset,
                    dtype_act=dtype_act,
                    dtype_weights=_dtype_from_context(da_context.dtype_weights),
                    quantization_type=quant_type,
                    da_context=da_context,
                )
            )
            if not switch_tiles:
                continue

            rows = []
            best_idxs = []
            bucket_latencies = []
            bucket_tactic_latencies = []
            for dist_idx, dist in enumerate(active_auto_distributions(config)):
                tile_latencies = {
                    int(t): float(cached_latencies[(bucket, dist_idx)][int(t)])
                    for t in switch_tiles
                    if int(t) in cached_latencies.get((bucket, dist_idx), {})
                }
                if not tile_latencies:
                    continue

                ti = generate_da_distribution_assignments(
                    dist,
                    torch.zeros(bucket, top_k, dtype=torch.int32, device=device),
                    num_local_experts,
                    num_experts,
                    top_k,
                    local_expert_offset,
                )
                counts = compute_local_expert_counts_from_plain_ids(
                    ti,
                    num_local_experts,
                    local_expert_offset,
                ).astype(np.float64)

                min_lat = min(tile_latencies.values())
                if tie_eps > 0.0 and min_lat > 0.0:
                    near_best = {
                        t: l
                        for t, l in tile_latencies.items()
                        if (l - min_lat) / min_lat <= tie_eps
                    }
                else:
                    near_best = tile_latencies
                best_tile = max(near_best.keys())
                best_idx = switch_tiles.index(int(best_tile))

                sorted_desc = np.sort(counts)[::-1].astype(np.float32)
                n = float(np.linalg.norm(sorted_desc))
                if n < 1e-12:
                    continue
                sorted_desc = sorted_desc / n
                if sorted_desc.shape[0] < num_local_experts:
                    pad = np.zeros(
                        num_local_experts - sorted_desc.shape[0],
                        dtype=np.float32,
                    )
                    sorted_desc = np.concatenate([sorted_desc, pad])
                elif sorted_desc.shape[0] > num_local_experts:
                    sorted_desc = sorted_desc[:num_local_experts]
                rows.append(sorted_desc)
                best_idxs.append(best_idx)
                bucket_latencies.append(tile_latencies)
                bucket_tactic_latencies.append(
                    dict(cached_tactic_latencies.get((bucket, dist_idx), {}))
                )

            _finish_bucket(
                bucket,
                switch_tiles,
                rows,
                best_idxs,
                bucket_latencies,
                bucket_tactic_latencies,
            )

    if not cached_latencies:
        unavailable_decision = {
            "policy": "noda",
            "candidate_policy": "unavailable",
            "candidate_tactics": [],
            "final_policy": "noda",
            "final_tactics": [],
            "control_overhead_source": "pre_recorded_calibration",
            "overhead_ms": float(config.control_overhead_us) / 1000.0,
            "margin": float(config.baseline_guard_margin),
            "admission_applied": False,
            "limitation": "missing_fixed_candidate_timing",
            "baseline_guard_signature": baseline_guard_signature(config),
        }
        for bucket in sorted(buckets_with_tiles):
            bundle_plans[str(int(bucket))] = dict(unavailable_decision)
            for offset in auto_offsets:
                offset_context = da_state.context_with_offset(da_context, int(offset))
                da_state.BASELINE_GUARD_DECISIONS[
                    da_state.cache_key(offset_context, int(bucket))
                ] = dict(unavailable_decision)
        if _verbose:
            print(
                "[DA k-NN auto-profile] no recorded autotuner timing; "
                "baseline guard admission unavailable, skipping",
                flush=True,
            )
    eager_tactics: Dict[str, Dict[str, Any]] = {}
    effective_runner_hash = hash(runner) if runner_hash is None else runner_hash
    for bucket in sorted(buckets_with_tiles):
        eager = best_static_tactic_from_profiles(
            AutoTuner.get(),
            da_context.op_name,
            runner.__class__.__name__,
            int(effective_runner_hash),
            int(bucket),
            da_context=da_context,
        )
        if eager is None:
            continue
        eager_fields = {
            "eager_tactic": tuple(int(value) for value in eager[0]),
            "eager_time_ms": float(eager[1]),
            "eager_profile_contract": DEFAULT_PROFILE_CONTRACT,
        }
        eager_tactics[str(int(bucket))] = {
            "tactic": [int(eager[0][0]), int(eager[0][1])],
            "time_ms": float(eager[1]),
        }
        if str(int(bucket)) in bundle_plans:
            bundle_plans[str(int(bucket))].update(eager_fields)
        for offset in auto_offsets:
            decision = da_state.BASELINE_GUARD_DECISIONS.get(
                da_state.cache_key(
                    da_state.context_with_offset(da_context, int(offset)), int(bucket)
                )
            )
            if decision is not None:
                decision.update(eager_fields)

    if _verbose:
        print(
            f"[DA k-NN auto-profile] done: {n_uploaded} buckets uploaded, "
            f"{sum(len(v) for v in bundle_exemplars.values())} total exemplars",
            flush=True,
        )

    # Save bundle to disk if a path was specified but didn't exist.
    if config.bundle_path and not os.path.isfile(config.bundle_path):
        import pickle

        bundle = {
            "tactic_table": {},
            "eager_tactics": eager_tactics,
            "exemplars": bundle_exemplars,
            "plans": bundle_plans,
            "meta": {
                "schema_version": int(da_context.schema_version),
                "baseline_guard_signature": baseline_guard_signature(config),
                "default_profile_contract": DEFAULT_PROFILE_CONTRACT,
                "device_type": da_context.device_type,
                "device_index": int(da_context.device_index),
                "ep": max(1, num_experts // max(1, num_local_experts)),
                "num_local": num_local_experts,
                "local_offset": local_expert_offset,
                "num_global_experts": num_experts,
                "top_k": top_k,
                "hidden_size": hidden_size,
                "intermediate_size": intermediate_size,
                "activation_type": int(da_context.activation_type),
                "op_name": da_context.op_name,
                "dtype_act": int(da_context.dtype_act),
                "dtype_weights": int(da_context.dtype_weights),
                "quantization_type": int(da_context.quantization_type),
                "weight_layout": int(da_context.weight_layout),
                "use_shuffled_weight": bool(da_context.use_shuffled_weight),
                "use_per_token_scaling": bool(da_context.use_per_token_scaling),
                "has_gemm1_lora_delta": bool(da_context.has_gemm1_lora_delta),
                "num_tokens": sorted(buckets_with_tiles.keys()),
                "distributions": list(config.profile_signature),
                "profile_signature": list(config.profile_signature),
                "profile_sample_count": config.distribution_sample_count,
                "warmup": warmup,
                "iters": iters,
                "auto_profiled": True,
            },
        }
        os.makedirs(
            os.path.dirname(os.path.abspath(config.bundle_path)) or ".",
            exist_ok=True,
        )
        with open(config.bundle_path, "wb") as f:
            pickle.dump(bundle, f)
        print(
            f"[DA k-NN auto-profile] saved bundle to {config.bundle_path}",
            flush=True,
        )

    return n_uploaded


def register_auto_profile_callback(
    backend: DAProfileBackend,
    tuner,
    moe_op,
    *,
    da_context: da_state.DAMoeContext,
    hidden_states,
    hidden_states_scale,
    routing_logits,
    routing_bias,
    gemm1_weights,
    gemm1_weights_scale,
    gemm1_bias,
    gemm1_alpha,
    gemm1_beta,
    gemm1_clamp_limit,
    gemm2_weights,
    gemm2_weights_scale,
    gemm2_bias,
    output1_scale_scalar,
    output1_scale_gate_scalar,
    output2_scale_scalar,
    per_token_scale,
    num_experts,
    top_k,
    n_group,
    topk_group,
    intermediate_size,
    local_expert_offset,
    num_local_experts,
    routed_scaling_factor,
    use_routing_scales_on_input,
    routing_method_type,
    activation_type,
    tune_max_num_tokens,
    enable_pdl,
    runner,
    runner_hash: Optional[int] = None,
    config: Optional[DAConfig] = None,
):
    """Register a post-autotuning callback that runs kNN auto-profiling.

    The callback fires when the ``with autotune():`` context exits, at
    which point ``da_state.PER_TILE_TACTICS`` is fully populated for
    all profiled buckets.  Running here (instead of mid-autotune) ensures
    the kNN exemplar sweep covers every bucket the autotuner discovered.
    """

    def _cb():
        if not da_state.PER_TILE_TACTICS:
            return
        try:
            n = auto_profile_knn_exemplars(
                backend,
                moe_op,
                da_context=da_context,
                hidden_states=hidden_states,
                hidden_states_scale=hidden_states_scale,
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                gemm1_weights=gemm1_weights,
                gemm1_weights_scale=gemm1_weights_scale,
                gemm1_bias=gemm1_bias,
                gemm1_alpha=gemm1_alpha,
                gemm1_beta=gemm1_beta,
                gemm1_clamp_limit=gemm1_clamp_limit,
                gemm2_weights=gemm2_weights,
                gemm2_weights_scale=gemm2_weights_scale,
                gemm2_bias=gemm2_bias,
                output1_scale_scalar=output1_scale_scalar,
                output1_scale_gate_scalar=output1_scale_gate_scalar,
                output2_scale_scalar=output2_scale_scalar,
                per_token_scale=per_token_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=n_group,
                topk_group=topk_group,
                intermediate_size=intermediate_size,
                local_expert_offset=local_expert_offset,
                num_local_experts=num_local_experts,
                routed_scaling_factor=routed_scaling_factor,
                use_routing_scales_on_input=use_routing_scales_on_input,
                routing_method_type=routing_method_type,
                activation_type=activation_type,
                tune_max_num_tokens=tune_max_num_tokens,
                enable_pdl=enable_pdl,
                runner=runner,
                runner_hash=runner_hash,
                config=config,
            )
            label = (
                "auto-profiled (in-memory)"
                if not _resolve_config(config).bundle_path
                else f"auto-profiled → {_resolve_config(config).bundle_path}"
            )
            print(f"[DA k-NN] {label}: {n} bucket cells", flush=True)
        except Exception as e:
            print(f"[DA k-NN] auto-profile failed: {e}", flush=True)

    tuner._post_autotune_callbacks.append(_cb)
