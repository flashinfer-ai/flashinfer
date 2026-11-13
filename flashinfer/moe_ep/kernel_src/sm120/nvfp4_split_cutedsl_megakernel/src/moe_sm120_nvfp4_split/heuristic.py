"""Pure candidate-based SM120 selector. No CUDA calls, routing reads or env state.

Legality, numerical eligibility and performance estimates are separate layers.
Only implemented local EP4 variants are costed. Other topologies use the frozen
compatibility selector until independently qualified. Estimates are not counters.
"""

from __future__ import annotations
from dataclasses import dataclass, replace, asdict
from functools import lru_cache
from copy import deepcopy
import math
from typing import Optional, Tuple
from .candidate_catalog import ALL_PROFILES, CATALOG_VERSION
from .legacy_heuristic import (
    MegaMoEHeuristicInput,
    MegaMoEHeuristicOverrides as _LegacyOverrides,
    MegaMoEKernelConfig as _LegacyConfig,
    _scale_sm_partition,
    select_megamoe_config as _legacy,
)


@dataclass(frozen=True)
class MegaMoEHeuristicOverrides(_LegacyOverrides):
    k1_compact_tokens: Optional[bool] = None
    k2_compact_tokens: Optional[bool] = None
    k1_stages: Optional[int] = None


@dataclass(frozen=True)
class MegaMoEKernelConfig(_LegacyConfig):
    k1_compact_tokens: bool = False
    k2_compact_tokens: bool = False
    # None only on compatibility fallback: preserve its previous kernel resolver.
    k1_stages: Optional[int] = None

    def with_overrides(self, overrides):
        # Applying an idempotent backend override must preserve this config.
        # Backend changes need shape/topology and go through the selector.
        return _apply_overrides(self, overrides)


MODEL_VERSION = "candidate-v15-abi27-large-joint-unaccepted"
NUMERICAL_CONTRACTS = ("ordinary_topk", "owner_bf16")


@dataclass(frozen=True)
class CandidateFeatures:
    """Per-rank work estimates, not measured traffic or hardware counters.

    Useful work uses unpadded routes; padded work and weight requests follow
    tile counts. Wave counts divide tile tasks by resident worker CTAs. Byte
    estimates include NVFP4 payload and scales and are expressed in bytes.
    """

    useful_fc1_flops: float
    useful_fc2_flops: float
    padded_fc1_flops: float
    padded_fc2_flops: float
    fc1_weight_request_bytes: float
    fc2_weight_request_bytes: float
    unique_weight_bytes: float
    fc1_tiles: float
    fc2_tiles: float
    k1_ctas: int
    k2_ctas: int
    k1_waves: float
    k2_waves: float
    ready_token_n: int
    ready_blocks: float
    descriptors: float
    ready_fanout: float
    descriptor_k_steps: int
    grain_relative_to_worker_work: float
    dispatch_bytes: float
    cache_saved_bytes: float
    token_back_bytes: float
    metadata_bytes: float
    owner_probability: float
    k1_smem_estimate: int
    k2_smem_estimate: int
    k1_threads: int
    k2_threads: int
    reducer: str


@dataclass(frozen=True)
class Cost:
    """Predicted full-call costs in microseconds (including K3).

    Components feed the calibrated overlap model; they are not disjoint
    hardware phase timings and should not be summed again by callers.
    """

    total_us: float
    producer_us: float
    consumer_us: float
    dispatch_us: float
    tail_us: float
    k3_us: float
    queue_us: float


@dataclass(frozen=True)
class Candidate:
    config: MegaMoEKernelConfig
    features: CandidateFeatures
    cost: Cost
    resource_evidence: str = (
        "analytic SMEM budget; constructor/codegen required for qualification"
    )


@lru_cache(maxsize=512)
def _expected_tiles(mu: float, n: int) -> float:
    """Poisson expected ceil(X/n), with bounded normal approximation at large mu.

    This models unknown routing only; it is explicitly not observed expert counts.
    """
    if mu < 128:
        probability = math.exp(-mu)
        out = 0.0
        limit = int(mu + 14 * math.sqrt(mu + 1) + 64)
        for x in range(1, limit + 1):
            probability *= mu / x
            out += ((x + n - 1) // n) * probability
        return out
    sigma = math.sqrt(mu)
    lo = max(0, int((mu - 10 * sigma) // n))
    hi = int((mu + 10 * sigma) // n) + 2
    return float(lo) + sum(
        0.5 * math.erfc((j * n + 0.5 - mu) / (math.sqrt(2) * sigma))
        for j in range(lo, hi)
    )


def _validate_counts(shape, counts):
    if counts is None:
        return
    if not isinstance(counts, tuple) or len(counts) != shape.num_total_experts:
        raise ValueError(
            "host_expert_counts must be an already-host tuple with one count per global expert"
        )
    if any(type(x) is not int or x < 0 for x in counts):
        raise ValueError("expert counts must be nonnegative integers")
    if (
        sum(counts)
        != shape.tokens_per_rank * shape.expert_parallel_size * shape.num_topk
    ):
        raise ValueError("expert counts do not match EP route volume")


def _tiles(shape, n, counts=None):
    if counts is not None:
        # Worst owner workload, not a GPU->CPU query and not an average hiding skew.
        e = shape.num_total_experts // shape.expert_parallel_size
        return float(
            max(
                sum((x + n - 1) // n for x in counts[j : j + e])
                for j in range(0, len(counts), e)
            )
        )
    return (
        shape.num_total_experts
        / shape.expert_parallel_size
        * _expected_tiles(shape.expected_rows_per_expert, n)
    )


def derive_features(shape, c, host_expert_counts=None):
    n1, n2 = c.k1_tile[1], c.k2_tile[1]
    t1, t2 = _tiles(shape, n1, host_expert_counts), _tiles(
        shape, n2, host_expert_counts
    )
    ready = max(n1, n2, 128 if c.k2_warps == 12 else 0)
    blocks = _tiles(shape, ready, host_expert_counts)
    htiles = (shape.hidden + 63) // 64
    ksteps = (shape.intermediate // 2 + 127) // 128
    descriptors = t2 * ((htiles + c.ready_queue_bundle - 1) // c.ready_queue_bundle)
    fanout = descriptors / max(blocks, 1e-30)
    c1 = c.k1_sms
    c2 = c.k2_sms * (2 if c.k2_warps == 12 else 1)
    rows = shape.tokens_per_rank * shape.num_topk
    h, i = shape.hidden, shape.intermediate
    owner_p = 1 - shape.probability_one_owner_is_untouched
    row_bytes = (h + 1) // 2 + (h + 15) // 16 + 4
    remote_rows = shape.expected_remote_route_rows
    cache_saved = (
        shape.expected_dispatch_rank_cache_saved_bytes if c.dispatch_rank_cache else 0.0
    )
    dispatch = max(0.0, remote_rows * row_bytes - cache_saved)
    return_rows = (
        shape.tokens_per_rank * (shape.expert_parallel_size - 1) * owner_p
        if c.rank_local_combine
        else remote_rows
    )
    metadata = shape.tokens_per_rank * shape.num_topk * 8

    # SM120 TMA layouts have128-row scale padding and128-byte data granularity.
    # These are conservative screening estimates; actual layouts remain authoritative.
    def smem(n, stage, dispatch):
        stage_bytes = 64 * 128 // 2 + n * 128 // 2 + 2 * 128 * 128 // 16
        # Match kernel budget: base1024, four packed activation pull rows,
        # aligned barriers32+128, and FC1 reduction/packed-store epilogue.
        fixed = 1024 + (
            32
            + ((4 * ((shape.hidden + 1) // 2) + 127) // 128) * 128
            + 128
            + 2048
            + n * 16
            if dispatch
            else 0
        )
        return stage * stage_bytes + fixed

    work = t2 * htiles * ksteps
    return CandidateFeatures(
        2 * rows * h * i,
        rows * h * i,
        2 * t1 * n1 * h * i,
        t2 * n2 * h * i,
        t1 * h * i * 9 / 16,
        t2 * h * i * 9 / 32,
        _tiles(
            shape,
            shape.tokens_per_rank * shape.expert_parallel_size * shape.num_topk + 1,
            host_expert_counts,
        )
        * h
        * i
        * 27
        / 32,
        t1,
        t2,
        c1,
        c2,
        t1 * ((i + 63) // 64) / c1,
        t2 * htiles / c2,
        ready,
        blocks,
        descriptors,
        fanout,
        ksteps,
        min(c.ready_queue_bundle, htiles) * ksteps / max(work / c2, 1.0),
        dispatch,
        cache_saved,
        return_rows * h * 2,
        metadata,
        owner_p,
        smem(n1, c.k1_stages or 2, True),
        smem(n2, c.k2_stages, False),
        384,
        352 if c.k2_warps == 12 else 256,
        (
            "owner_vector"
            if c.rank_local_combine and h % 2048 == 0 and shape.num_topk <= 8
            else "owner_scalar" if c.rank_local_combine else "ordinary_topk"
        ),
    )


def legality_errors(shape, c):
    """Kernel/layout/progress constraints only; no performance or BF16 policy."""
    errors = []
    for label in ("k1", "k2"):
        flag = getattr(c, label + "_compact_tokens")
        if type(flag) is not bool:
            errors.append(label + " compact token flag must be bool")
        if flag and getattr(c, label + "_tile")[1] != 128:
            errors.append(label + " compact token path requires an N128 task")
    if c.k2_compact_tokens and (c.k2_warps != 8 or c.k2_tail_reclaim):
        errors.append("compact K2 requires qualified 8-warp non-tail path")
    if shape.hidden % 32 or shape.intermediate % 64:
        errors.append("NVFP4 alignment requires H%32=0,I%64=0")
    for name, tile in (("K1", c.k1_tile), ("K2", c.k2_tile)):
        if tile[0] != 64 or tile[1] not in (16, 32, 64, 128) or tile[2] != 128:
            errors.append(name + " tile outside implemented candidate catalog")
    if c.k1_stages not in (None, 1, 2, 3, 4, 5):
        errors.append("K1 stage count exceeds implemented1..5 bound")
    if c.k2_stages not in (1, 2, 3, 4, 5):
        errors.append("K2 stage count exceeds implemented1..5 bound")
    if c.k2_warps not in (8, 12):
        errors.append("K2 warp mode unsupported")
    if c.k2_warps == 12 and c.k2_tile[1] != 16:
        errors.append("dual group requires K2N16")
    if c.k2_tail_reclaim and c.k2_warps != 12:
        errors.append("tail catalog requires dual group")
    if c.total_sms != shape.num_sms:
        errors.append("SM coverage mismatch")
    counts = (
        (c.k1_sms, c.k2_sms)
        if c.comm_backend == "p2p_direct"
        else (c.k1_sms, c.tx_sms, c.rx_sms, c.k2_sms)
    )
    if any(x < shape.sm_min_partition for x in counts):
        errors.append("SM partition below minimum")
    if any(x % shape.sm_partition_alignment for x in counts[:-1]):
        errors.append("SM alignment violation")
    if c.dispatch_warps not in (1, 2, 4):
        errors.append("dispatch warp count")
    if c.ready_queue_bundle not in (1, 2, 4, 8, 16):
        errors.append("unsupported Q catalog")
    if c.comm_backend == "p2p_direct":
        if c.tx_sms or c.rx_sms or c.token_back_mode != "epi_warps":
            errors.append("P2P protocol mismatch")
    elif c.comm_backend in ("gin_hybrid", "gin"):
        if c.dispatch_warps != 4 or c.token_back_mode != "reuse_dispatch_warps":
            errors.append("transport dispatcher reuse/progress")
        if (
            shape.ep_cross_numa_peer_count
            and shape.tokens_per_rank <= 128
            and c.combine_slots < 4
        ):
            errors.append("small cross-NUMA combine progress requires full window")
    else:
        errors.append("unsupported backend")
    if c.rank_local_combine:
        if (
            c.comm_backend != "p2p_direct"
            or c.token_back_mode != "epi_warps"
            or shape.tensor_parallel_size != 1
            or shape.expert_parallel_size > shape.num_topk
            or c.k2_warps == 12
            or c.k2_tail_reclaim
        ):
            errors.append("RLC layout/protocol incompatibility")
    if c.dispatch_rank_cache and c.comm_backend != "p2p_direct":
        errors.append("rank cache requires P2P token-strided dispatch")
    if any(
        getattr(c, n) <= 0
        for n in (
            "dispatch_chunk_tokens",
            "combine_chunk_rows",
            "dispatch_channels",
            "combine_channels",
            "dispatch_slots",
            "combine_slots",
            "dispatch_local_handoff_windows",
            "dispatch_remote_handoff_windows",
            "ibgda_rc_per_pe",
            "tp_k3_chunks",
        )
    ):
        errors.append("transport capacities must be positive")
    if shape.tensor_parallel_size == 1 and c.tp_k3_chunks != 1:
        errors.append("TP1 K3 chunks must equal1")
    return tuple(errors)


def semantic_errors(c, numerical_contract):
    """Caller permission for BF16 association; never a speed threshold."""
    if numerical_contract not in NUMERICAL_CONTRACTS:
        return ("unknown numerical contract",)
    if c.rank_local_combine and numerical_contract != "owner_bf16":
        return ("RLC requires explicit owner_bf16 numerical permission",)
    return ()


def qualification_limits(shape):
    """Performance evidence scope, deliberately separate from hardware legality."""
    limits = []
    if shape.num_sms != 110:
        limits.append("uncalibrated SM count")
    # These independent ranges cover the preregistered training/holdout matrix.
    # They are evidence bounds, not layout illegality or model identifiers.
    if not 1 <= shape.tokens_per_rank <= 24576:
        limits.append("uncalibrated token range")
    if not 2048 <= shape.hidden <= 7168:
        limits.append("uncalibrated hidden range")
    if not 4096 <= shape.intermediate <= 8192:
        limits.append("uncalibrated intermediate range")
    if not 4 <= shape.num_topk <= 8:
        limits.append("uncalibrated top-k range")
    if not 128 <= shape.num_total_experts <= 512:
        limits.append("uncalibrated expert-count range")
    if (
        shape.expert_parallel_size != 4
        or shape.ep_same_numa_peer_count != 3
        or shape.ep_cross_numa_peer_count
    ):
        limits.append("uncalibrated EP topology")
    if shape.tensor_parallel_size != 1 or shape.data_parallel_size != 1:
        limits.append("uncalibrated parallel decomposition")
    return tuple(limits)


def _apply_overrides(c, o):
    if o is None:
        return c
    updates = {k: v for k, v in vars(o).items() if v is not None}
    sms = [k for k in ("k1_sms", "k2_sms", "tx_sms", "rx_sms") if k in updates]
    if sms and len(sms) != 4:
        raise ValueError("SM overrides must specify K1/K2/TX/RX together")
    if "k2_tile" in updates or "k2_warps" in updates:
        # A constraint on N does not also choose a warp mode or disable tail.
        # Keep each complete proposal's unspecified fields, then filter illegal
        # combinations in enumerate_candidates. Resource flags are derived from
        # the resulting geometry even for a warp-only override.
        n = updates.get("k2_tile", c.k2_tile)[1]
        dual = updates.get("k2_warps", c.k2_warps) == 12
        updates.update(
            k2_natural_regs=n == 32 or dual,
            k2_min_blocks_per_sm=2 if n == 32 or dual else 1,
        )
    # A redundant backend override must not reset the already-selected SM partition.
    if updates.get("comm_backend", c.comm_backend) != c.comm_backend:
        raise ValueError("backend override requires transport-specific fallback")
    return replace(c, **updates)


def _base(
    shape, n1, n2, warps, stage, sms, q, dispatch, overlap, cache, rlc, tail, k1_stage
):
    return MegaMoEKernelConfig(
        comm_backend="p2p_direct",
        kernel_comm_backend="p2p_direct",
        token_back_mode="epi_warps",
        k1_tile=(64, n1, 128),
        k2_tile=(64, n2, 128),
        k2_stages=stage,
        k2_warps=warps,
        k1_sms=sms,
        k2_sms=shape.num_sms - sms,
        tx_sms=0,
        rx_sms=0,
        sm_min_partition=shape.sm_min_partition,
        sm_partition_alignment=shape.sm_partition_alignment,
        dispatch_chunk_tokens=64,
        combine_chunk_rows=16,
        dispatch_channels=2,
        combine_channels=2,
        dispatch_slots=2,
        combine_slots=2,
        combine_blocking_put=False,
        dispatch_local_handoff_windows=1,
        dispatch_remote_handoff_windows=1,
        ibgda_rc_per_pe=1,
        ibgda_rc_mapping="warp",
        tp_k3_chunks=1,
        dispatch_warps=dispatch,
        dispatch_compute_overlap=overlap,
        dispatch_rank_cache=cache,
        rank_local_combine=rlc,
        k1_ready_queue=True,
        k1_ready_queue_m_rotation=0,
        k2_ready_queue=True,
        ready_queue_bundle=q,
        k2_natural_regs=n2 == 32 or warps == 12,
        k2_min_blocks_per_sm=2 if n2 == 32 or warps == 12 else 1,
        k2_tail_reclaim=tail,
        expected_rows_per_expert=shape.expected_rows_per_expert,
        k1_stages=k1_stage,
    )


def estimate_cost(shape, c, f):
    from .fitted_cost import COEFFICIENTS, evaluate

    return Cost(*evaluate(shape, c, f, COEFFICIENTS))


def _in_cost_domain(shape, overrides):
    # Uncalibrated explicit stage overrides remain legal but use compatibility
    # selection for the other fields. Catalog limits are not kernel limits.
    if overrides and (
        overrides.k1_stages not in (None, 2, 3)
        or overrides.k2_stages not in (None, 2, 3)
    ):
        return False
    return not qualification_limits(shape) and (
        overrides is None or overrides.comm_backend in (None, "p2p_direct")
    )


def _full_profiles(shape, numerical_contract):
    for n1, n2, warps in (
        (16, 16, 12),
        (32, 16, 12),
        (32, 16, 8),
        (32, 32, 8),
        (64, 64, 8),
        (128, 128, 8),
    ):
        for k1_stage in (2, 3):
            for stage in ((2,) if warps == 12 else (2, 3)):
                for sms in (72, 80, 88):
                    for q in ((2, 4) if n2 <= 32 else (4, 8, 16)):
                        for dispatch, overlap, cache in (
                            (1, False, False),
                            (1, False, True),
                            (1, True, True),
                            (4, False, False),
                            (4, True, False),
                            (4, True, True),
                        ):
                            for rlc in (
                                (False, True)
                                if numerical_contract == "owner_bf16"
                                and warps != 12
                                and shape.expert_parallel_size <= shape.num_topk
                                else (False,)
                            ):
                                for tail in (
                                    (False, True) if warps == 12 else (False,)
                                ):
                                    yield n1, n2, warps, k1_stage, stage, sms, q, dispatch, overlap, cache, rlc, tail


def correctness_qualification_errors(shape, c):
    # Shared-buffer repair qualified on same-NUMA EP4. Preserve the historical
    # exclusion on the unqualified cross-NUMA combination until separately tested.
    if (
        shape.expert_parallel_size == 4
        and shape.tensor_parallel_size == 1
        and shape.data_parallel_size == 1
        and shape.num_sms == 110
        and shape.ep_cross_numa_peer_count
        and c.comm_backend == "p2p_direct"
        and c.dispatch_warps == 4
        and not c.dispatch_rank_cache
    ):
        return (
            "correctness qualification: cross-NUMA dispatch4/cache-off remains unqualified",
        )
    return ()


def enumerate_candidates(
    shape,
    overrides=None,
    *,
    numerical_contract="ordinary_topk",
    host_expert_counts=None,
    full_catalog=False,
):
    shape.validate()
    _validate_counts(shape, host_expert_counts)
    if numerical_contract not in NUMERICAL_CONTRACTS:
        raise ValueError("unknown numerical contract")
    if not _in_cost_domain(shape, overrides):
        return ()
    result = []
    seen = set()
    # Keep the complete original proposal set for ABI27. The old81-profile
    # subset was pruned by ABI26 predictions and is not used here.
    profiles = (
        _full_profiles(shape, numerical_contract) if full_catalog else ALL_PROFILES
    )
    for (
        n1,
        n2,
        warps,
        k1_stage,
        stage,
        sms,
        q,
        dispatch,
        overlap,
        cache,
        rlc,
        tail,
    ) in profiles:
        if rlc and numerical_contract != "owner_bf16":
            continue
        c = _base(
            shape,
            n1,
            n2,
            warps,
            stage,
            sms,
            q,
            dispatch,
            overlap,
            cache,
            rlc,
            tail,
            k1_stage,
        )
        c = _apply_overrides(c, overrides)
        if c in seen:
            continue
        seen.add(c)
        if (
            legality_errors(shape, c)
            or semantic_errors(c, numerical_contract)
            or correctness_qualification_errors(shape, c)
        ):
            continue
        f = derive_features(shape, c, host_expert_counts)
        if (
            f.k1_smem_estimate > 99 * 1024
            or f.k2_smem_estimate > 99 * 1024 / c.k2_min_blocks_per_sm
        ):
            continue
        result.append(Candidate(c, f, estimate_cost(shape, c, f)))
    if not result:
        raise ValueError(
            "no candidate satisfies overrides, numerical/legality constraints and correctness qualification"
        )
    return tuple(sorted(result, key=lambda x: x.cost.total_us))


def _policy_context():
    """Explicit policy dependencies; no module lookup or import side effects."""
    from .policy_context import PolicyContext

    return PolicyContext(
        derive_features=derive_features,
        legality_errors=legality_errors,
        semantic_errors=semantic_errors,
        correctness_qualification_errors=correctness_qualification_errors,
    )


@lru_cache(maxsize=256)
def _select_cached(shape, overrides, numerical_contract, host_expert_counts):
    choices = enumerate_candidates(
        shape,
        overrides,
        numerical_contract=numerical_contract,
        host_expert_counts=host_expert_counts,
    )
    if choices:
        from .robust_policy import select

        return select(
            _policy_context(),
            shape,
            choices[0].config,
            overrides,
            numerical_contract,
            host_expert_counts,
        )[0]
    return select_compatibility_config(
        shape, overrides, numerical_contract=numerical_contract
    )


def select_compatibility_config(
    shape, overrides=None, *, numerical_contract="ordinary_topk"
):
    # Frozen compatibility path; never represent this as a generalized cost result.
    shape.validate()
    if numerical_contract not in NUMERICAL_CONTRACTS:
        raise ValueError("unknown numerical contract")
    if numerical_contract == "ordinary_topk":
        if overrides and overrides.rank_local_combine:
            raise ValueError("RLC requires explicit owner_bf16 numerical permission")
    legacy_o = (
        _LegacyOverrides(
            **{k: v for k, v in vars(overrides).items() if k not in ("k1_stages", "k1_compact_tokens", "k2_compact_tokens")}
        )
        if overrides
        else None
    )
    base = _legacy(shape, legacy_o)
    if numerical_contract == "ordinary_topk" and base.rank_local_combine:
        base = _legacy(
            shape, replace(legacy_o or _LegacyOverrides(), rank_local_combine=False)
        )
    c = MegaMoEKernelConfig(
        **vars(base), k1_stages=overrides.k1_stages if overrides else None,
        k1_compact_tokens=bool(overrides.k1_compact_tokens) if overrides else False,
        k2_compact_tokens=bool(overrides.k2_compact_tokens) if overrides else False
    )
    errors = (
        legality_errors(shape, c)
        + semantic_errors(c, numerical_contract)
        + correctness_qualification_errors(shape, c)
    )
    if errors:
        raise ValueError("; ".join(errors))
    return c


def select_megamoe_config(
    shape,
    overrides=None,
    *,
    numerical_contract="ordinary_topk",
    host_expert_counts=None,
    tuning_config=None,
    hardware=None,
):
    """Return a complete config. Callers must explicitly permit owner BF16 order."""
    _validate_counts(shape, host_expert_counts)
    return _select_with_profiles(shape, overrides, numerical_contract, host_expert_counts,
                                 tuning_config, hardware)[0]


@lru_cache(maxsize=256)
def _select_with_profiles(shape, overrides, contract, counts, tuning_config, hardware):
    # Cache the FINAL immutable config, including policy and offline validation.
    # Diagnostics are private cache state; explain_selection returns a deep copy.
    # The existing selector remains the anchor and authoritative override validator.
    base = _select_cached(shape, overrides, contract, counts)
    from .model_policy import select as select_model
    selected, builtin = select_model(shape, base, overrides, contract, counts)
    info = dict(builtin=builtin, offline=None, source=(
        'builtin_model' if builtin['matched'] else 'general'))
    if tuning_config is not None:
        from .tuning_config import resolve_tuning
        tuned, detail = resolve_tuning(tuning_config, shape, overrides, contract, hardware)
        info['offline'] = detail
        if tuned is not None:
            return tuned, deepcopy(dict(info, source='offline_tuning'))
    from .compact_policy import select as select_compact
    selected, info["compact"] = select_compact(shape, selected, overrides, contract, counts)
    return selected, deepcopy(info)


def explain_selection(
    shape,
    overrides=None,
    *,
    numerical_contract="ordinary_topk",
    host_expert_counts=None,
    tuning_config=None,
    hardware=None,
):
    _validate_counts(shape, host_expert_counts)
    selected, profile_info = _select_with_profiles(
        shape, overrides, numerical_contract, host_expert_counts, tuning_config, hardware
    )
    choices = enumerate_candidates(
        shape,
        overrides,
        numerical_contract=numerical_contract,
        host_expert_counts=host_expert_counts,
    )
    from .fitted_cost import describe_rlc_path, describe_service
    from .robust_policy import select as select_robust

    return dict(
        model_version=MODEL_VERSION,
        profiles=deepcopy(profile_info),
        catalog_version=CATALOG_VERSION,
        numerical_contract=numerical_contract,
        rlc_path_features=describe_rlc_path(
            shape, selected, derive_features(shape, selected, host_expert_counts)
        ),
        service_features=(
            describe_service(
                shape, selected, derive_features(shape, selected, host_expert_counts)
            )
            if choices
            else dict(
                applicable=False,
                reason="compatibility fallback outside calibrated cost domain",
            )
        ),
        route_statistics=(
            "caller_host_counts"
            if host_expert_counts is not None
            else "uniform_poisson_estimate"
        ),
        communication_statistics="estimated_owner_probability_and_traffic",
        mode="route_robust_policy" if choices else "legacy_compatibility_fallback",
        candidate_count=len(choices),
        robust_policy=(
            select_robust(
                _policy_context(),
                shape,
                choices[0].config,
                overrides,
                numerical_contract,
                host_expert_counts,
            )[1]
            if choices
            else None
        ),
        qualification_limits=qualification_limits(shape),
        selected=asdict(selected),
        alternatives=[asdict(x) for x in choices[:8]],
        fixed_build=dict(
            k1_stages=selected.k1_stages or "legacy kernel default",
            scheduler_stages=2,
            fc2_ready_bundle_k_tiles=12,
            k2_token_stripe=2,
            flag_batch=4,
            epi_flag_batch=(1, 1),
            cluster=(1, 1, 1),
        ),
        qualification="experimental; performance/compiled resources require offline validation",
    )


__all__ = [
    "MegaMoEHeuristicInput",
    "MegaMoEHeuristicOverrides",
    "MegaMoEKernelConfig",
    "select_megamoe_config",
    "enumerate_candidates",
    "derive_features",
    "legality_errors",
    "semantic_errors",
    "qualification_limits",
    "explain_selection",
    "select_compatibility_config",
]
