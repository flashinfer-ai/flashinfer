"""Production construction API for SM120 Split-MegaMoE.

This module intentionally does not import :mod:`mega_runner`.  Frameworks own
input conversion, symmetric-buffer allocation, Green Context streams and graph
capture; this API owns deterministic heuristic selection, CuTe kernel
construction and the opaque workspace contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Any, Dict, Literal, Optional, Tuple

from .heuristic import (
    MegaMoEHeuristicInput,
    MegaMoEHeuristicOverrides,
    MegaMoEKernelConfig,
    select_megamoe_config,
)
from .jit_config import Sm120JitConfig


# Bump when a generated-kernel ABI or opaque workspace layout changes.
KERNEL_CACHE_ABI = 2


@dataclass(frozen=True)
class MegaMoEProblemSpec:
    """Static problem dimensions consumed by one EP-rank kernel instance.

    ``intermediate`` is the gate+up width stored by this rank.  For TP it is
    the already-sharded local width, matching the local weight tensors.
    """

    tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    expert_parallel_size: int
    expert_parallel_rank: int
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    gate_up_clamp: Optional[float] = None

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.expert_parallel_size

    def validate(self) -> None:
        positive = {
            "tokens_per_rank": self.tokens_per_rank,
            "num_topk": self.num_topk,
            "num_total_experts": self.num_total_experts,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "expert_parallel_size": self.expert_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        invalid = {name: value for name, value in positive.items() if value <= 0}
        if invalid:
            raise ValueError(f"problem dimensions must be positive: {invalid}")
        if self.num_total_experts % self.expert_parallel_size:
            raise ValueError(
                "num_total_experts must be divisible by expert_parallel_size"
            )
        if not 0 <= self.expert_parallel_rank < self.expert_parallel_size:
            raise ValueError("expert_parallel_rank is outside the EP group")
        if self.num_topk > self.num_total_experts:
            raise ValueError("num_topk cannot exceed num_total_experts")
        if self.hidden % 32 or self.intermediate % 64:
            raise ValueError(
                "SM120 MXFP4 x MXFP8 requires hidden % 32 == 0 and "
                "intermediate % 64 == 0"
            )
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0:
            raise ValueError("gate_up_clamp must be non-negative")


@dataclass(frozen=True)
class SplitKernelBuildOptions:
    """Production host-side options shared by K1 and K2 construction."""

    cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1)
    force_static_sched: bool = True
    clc_bundle_size: Optional[int] = None
    num_sched_stages: Optional[int] = None
    load_balance_mode: Literal["static", "atomic_counter"] = "static"
    group_hint: Optional[int] = None
    non_ubulk_fc2_store: bool = True
    in_kernel_fc2_reduce: bool = False
    flag_batch: int = 4
    epi_flag_batch: Optional[Tuple[int, int]] = (1, 1)
    concurrent_k1_k2: bool = True
    k1_active_clusters: Optional[int] = None
    k2_active_clusters: Optional[int] = None

    def validate(self) -> None:
        cm, cn, ck = self.cluster_shape_mnk
        if (cm, cn, ck) != (1, 1, 1):
            raise NotImplementedError(
                "the production split API currently supports cluster 1x1x1"
            )
        if not self.force_static_sched:
            raise NotImplementedError("dynamic CLC is not a production path")
        if self.flag_batch <= 0:
            raise ValueError("flag_batch must be positive")
        for name, value in (
            ("k1_active_clusters", self.k1_active_clusters),
            ("k2_active_clusters", self.k2_active_clusters),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class MegaMoECompileSpec:
    """Complete immutable specialization record for one compiled kernel set."""

    problem: MegaMoEProblemSpec
    kernel: MegaMoEKernelConfig
    jit: Sm120JitConfig = Sm120JitConfig()
    build: SplitKernelBuildOptions = SplitKernelBuildOptions()

    def __post_init__(self) -> None:
        self.problem.validate()
        self.build.validate()
        if self.kernel.total_sms <= 0:
            raise ValueError("kernel SM partition must be non-empty")

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "cache_abi": KERNEL_CACHE_ABI,
            "architecture": "sm_120a",
            "weight_dtype": "mxfp4_e2m1_block32_e8m0",
            "activation_dtype": "mxfp8_e4m3_block32_e8m0",
            "weight_layout": "packed_k_major_x2",
            "problem": asdict(self.problem),
            "kernel": asdict(self.kernel),
            "jit": self.jit.canonical_dict(),
            "build": asdict(self.build),
        }

    @property
    def cache_key(self) -> str:
        payload = json.dumps(
            self.canonical_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return (
            f"sm120-mxfp4mxfp8-split-v{KERNEL_CACHE_ABI}-"
            f"{sha256(payload).hexdigest()}"
        )


@dataclass(frozen=True)
class SplitKernelBundle:
    """Constructed K1/K2 objects and their shared opaque-workspace contract."""

    k1: Any
    k2: Any
    k2_drain: Optional[Any]
    k2_finalizer: Optional[Any]
    local_workspace_bytes: int
    shared_workspace_bytes: int
    cache_key: str


def select_compile_spec(
    *,
    problem: MegaMoEProblemSpec,
    ep_same_numa_peer_count: int,
    ep_cross_numa_peer_count: int,
    num_sms: int,
    sm_min_partition: int,
    sm_partition_alignment: int,
    overrides: Optional[MegaMoEHeuristicOverrides] = None,
    jit: Optional[Sm120JitConfig] = None,
    build: Optional[SplitKernelBuildOptions] = None,
) -> MegaMoECompileSpec:
    """Resolve the production heuristic into a cacheable compile spec."""

    problem.validate()
    heuristic_input = MegaMoEHeuristicInput(
        tokens_per_rank=problem.tokens_per_rank,
        hidden=problem.hidden,
        intermediate=problem.intermediate,
        num_topk=problem.num_topk,
        num_total_experts=problem.num_total_experts,
        data_parallel_size=problem.data_parallel_size,
        tensor_parallel_size=problem.tensor_parallel_size,
        expert_parallel_size=problem.expert_parallel_size,
        ep_same_numa_peer_count=ep_same_numa_peer_count,
        ep_cross_numa_peer_count=ep_cross_numa_peer_count,
        num_sms=num_sms,
        sm_min_partition=sm_min_partition,
        sm_partition_alignment=sm_partition_alignment,
    )
    return MegaMoECompileSpec(
        problem=problem,
        kernel=select_megamoe_config(heuristic_input, overrides),
        jit=jit or Sm120JitConfig(),
        build=build or SplitKernelBuildOptions(),
    )


def build_split_kernels(spec: MegaMoECompileSpec) -> SplitKernelBundle:
    """Construct K1/K2/K2-tail objects without benchmark-runner dependencies."""

    import cutlass

    from common.megamoe_constants import SfPaddingBlock
    from .kernel_dispatch_fc1 import build_sm120_dispatch_fc1_kernel
    from .kernel_fc2_combine import Sm120Fc2CombineKernel
    from .sm120_mma import CTA_TOKEN_TILE

    problem = spec.problem
    config = spec.kernel
    options = spec.build
    cluster_size = options.cluster_shape_mnk[0] * options.cluster_shape_mnk[1]
    k1_clusters = options.k1_active_clusters or config.k1_sms // cluster_size
    k2_clusters = options.k2_active_clusters or config.k2_sms // cluster_size
    k1_group_hint = options.group_hint or k1_clusters
    k2_group_hint = options.group_hint or k2_clusters

    k1_token_n = config.k1_tile[1]
    k2_token_n = config.k2_tile[1]
    ready_token_n = max(k1_token_n, k2_token_n)
    token_padding_block = min(CTA_TOKEN_TILE, k1_token_n)
    if (
        ready_token_n % k1_token_n
        or ready_token_n % k2_token_n
        or k1_token_n % token_padding_block
    ):
        raise ValueError("K1/K2 token tiles do not share a valid ready block")

    common_kwargs = dict(
        cluster_shape_mnk=options.cluster_shape_mnk,
        use_2cta_instrs=False,
        token_padding_block=token_padding_block,
        sf_padding_block=SfPaddingBlock,
        fc1_ready_tile_tokens=k1_token_n,
        fc1_producer_tile_tokens=k1_token_n,
        k2_token_tile_tokens=k2_token_n,
        static_expert_shape=(
            problem.num_experts_per_rank,
            problem.intermediate,
            problem.hidden,
        ),
        force_static_sched=options.force_static_sched,
        clc_bundle_size=options.clc_bundle_size,
        num_sched_stages=options.num_sched_stages,
        world_size=problem.expert_parallel_size,
        local_rank=problem.expert_parallel_rank,
        num_topk=problem.num_topk,
        max_tokens_per_rank=problem.tokens_per_rank,
        hidden=problem.hidden,
        comm_backend=config.kernel_comm_backend,
        ibgda_dispatch_chunk_tokens=config.dispatch_chunk_tokens,
        fc2_output_dtype=cutlass.BFloat16,
        non_ubulk_fc2_store=options.non_ubulk_fc2_store,
        in_kernel_fc2_reduce=options.in_kernel_fc2_reduce,
        token_back_mode=config.token_back_mode,
        apply_topk_in_fc1=False,
        gate_up_clamp=problem.gate_up_clamp,
        flag_batch=options.flag_batch,
        epi_flag_batch=options.epi_flag_batch,
        dispatch_pull_mode=config.dispatch_pull_mode,
        dispatch_warps_per_tile=config.dispatch_warps_per_tile,
        dispatch_compute_overlap=config.dispatch_compute_overlap,
        k1_ready_queue_workspace=config.k1_ready_queue,
        k2_ready_queue=config.k2_ready_queue,
        k2_ready_queue_bundle=config.ready_queue_bundle,
        k2_natural_regs=config.k2_natural_regs,
        k2_min_blocks_per_sm=config.k2_min_blocks_per_sm,
        k1_ready_queue_m_rotation=config.k1_ready_queue_m_rotation,
        jit_config=spec.jit,
    )

    k1 = build_sm120_dispatch_fc1_kernel(
        group_hint=k1_group_hint,
        mma_tiler_mnk=config.k1_tile,
        load_balance_mode=options.load_balance_mode,
        k2_tail_reclaim=config.k2_tail_reclaim,
        green_trace_role=0,
        **common_kwargs,
    )
    k2_load_balance = (
        "atomic_counter" if config.k2_tail_reclaim
        else options.load_balance_mode
    )
    k2_common = dict(
        group_hint=k2_group_hint,
        mma_tiler_mnk=config.k2_tile,
        num_ab_stages_override=config.k2_stages,
        compact_k2=(config.k2_warps == 8 or config.k2_tile[1] == 256),
        load_balance_mode=k2_load_balance,
        k2_tail_reclaim=config.k2_tail_reclaim,
        skip_global_tail=config.k2_tail_reclaim,
        producer_sm_count=(
            k1_clusters * cluster_size if options.concurrent_k1_k2 else 0
        ),
        green_trace_role=1,
    )
    k2 = Sm120Fc2CombineKernel(**k2_common, **common_kwargs)

    k2_drain = None
    k2_finalizer = None
    if config.k2_tail_reclaim:
        tail_common = dict(k2_common)
        tail_common.update(
            load_balance_mode="atomic_counter",
            k2_tail_reclaim=True,
            producer_sm_count=0,
        )
        tail_common.update(group_hint=k1_group_hint, green_trace_role=2)
        k2_drain = Sm120Fc2CombineKernel(**tail_common, **common_kwargs)
        tail_common.update(
            group_hint=1,
            skip_global_tail=False,
            green_trace_role=3,
        )
        k2_finalizer = Sm120Fc2CombineKernel(**tail_common, **common_kwargs)

    expected_threads = 32 * config.k2_warps
    if k2.threads_per_cta != expected_threads:
        raise RuntimeError(
            f"K2 requested {expected_threads} threads, got {k2.threads_per_cta}"
        )
    workspace_sizes = k1.get_workspace_sizes()
    peers = [k2]
    if k2_drain is not None:
        peers.extend((k2_drain, k2_finalizer))
    if any(kernel.get_workspace_sizes() != workspace_sizes for kernel in peers):
        raise RuntimeError("K1/K2 workspace layouts are not byte-identical")

    return SplitKernelBundle(
        k1=k1,
        k2=k2,
        k2_drain=k2_drain,
        k2_finalizer=k2_finalizer,
        local_workspace_bytes=workspace_sizes[0],
        shared_workspace_bytes=workspace_sizes[1],
        cache_key=spec.cache_key,
    )


def compile_combine_reduce(*args, **kwargs):
    """Compile K3 lazily so importing the host API does not initialize CuTe."""

    from .kernel_combine_reduce import compile_topk_reduce

    return compile_topk_reduce(*args, **kwargs)


__all__ = [
    "KERNEL_CACHE_ABI",
    "MegaMoECompileSpec",
    "MegaMoEProblemSpec",
    "SplitKernelBuildOptions",
    "SplitKernelBundle",
    "Sm120JitConfig",
    "build_split_kernels",
    "compile_combine_reduce",
    "select_compile_spec",
]
