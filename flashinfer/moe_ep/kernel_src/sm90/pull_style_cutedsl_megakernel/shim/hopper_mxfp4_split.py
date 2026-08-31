# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Production Green-Context split runtime for Hopper Humming MXFP4 MegaMoE.

The session owns one fixed-address K0 -> {K1 || K2} -> K3 pipeline. K1 and K2
are compiled against role-specific narrow ABIs and run on disjoint CUDA Green
Context streams. There is intentionally no sequential or fused fallback.
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Optional, Tuple

import torch

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    normalize_sm90_routing_profile,
)

from .comm import (
    _compute_peer_offsets,
    ensure_not_capturing,
    free_sym_tensor,
    resolve_gate_up_clamp,
    sym_zeros,
)
from .hopper_fp8 import _sym_zeros_byte_view_1b
from .hopper_mxfp4 import (
    MegaMoEHopperMxfp4Config,
    MegaMoEHopperMxfp4Frontend,
    MegaMoEHopperMxfp4Inputs,
    TransformedMxfp4Weights,
    _build_mxfp4_inputs,
)


class Mxfp4SplitError(RuntimeError):
    """Base error for the production MXFP4 Green split runtime."""


class Mxfp4SplitUnavailableError(Mxfp4SplitError):
    """Raised when the driver cannot provide the required split capability."""


class Mxfp4SplitSessionPoisonedError(Mxfp4SplitError):
    """Raised after a launch, synchronization, or capture failure."""


class Mxfp4SplitLifecycleError(Mxfp4SplitError):
    """Raised for invalid capture pointers, stream use, or teardown order."""


_SPLIT_GENERATIONS = itertools.count(1)
_SPLIT_TUNING_IDENTITY = (
    "sm90_w_mxfp4_e2m1_k32_a_fp8_e4m3_per_token_full_hidden_"
    "humming_v1_fold_m64_k128_gateup8_packedk2_residual64_"
    "swapab_green_split_v1"
)


@dataclass(frozen=True)
class MegaMoEHopperMxfp4SplitConfig:
    """Complete compile/session identity for one Green split pipeline."""

    rank: int
    world_size: int
    num_tokens_per_rank: int
    num_topk: int
    num_total_experts: int
    hidden: int
    intermediate: int
    k1_mma_tiler_mnk: Tuple[int, int, int]
    k2_mma_tiler_mnk: Tuple[int, int, int]
    k1_cluster_shape_mnk: Tuple[int, int, int]
    k2_cluster_shape_mnk: Tuple[int, int, int]
    k1_sm_count: int
    k2_sm_count: int
    k1_group_hint: Optional[int] = None
    k2_group_hint: Optional[int] = None
    k1_num_sched_stages: Optional[int] = None
    k2_num_sched_stages: Optional[int] = None
    counter_epoch_banks: Literal[1, 2] = 1
    graph_variant: Literal["cold_k0", "steady_k3_reset"] = "steady_k3_reset"
    clc_bundle_size: Optional[int] = None
    flag_batch: int = 1
    epi_flag_batch: Tuple[int, int] = (2, 4)
    gate_up_clamp: Optional[float] = None
    enable_iket: bool = False
    apply_topk_in_fc1: Literal[True] = True
    routing_profile: str = field(
        default=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        kw_only=True,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "routing_profile",
            normalize_sm90_routing_profile(self.routing_profile),
        )
        if self.world_size < 1 or not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"invalid split rank/world_size {self.rank}/{self.world_size}"
            )
        if self.num_total_experts % self.world_size:
            raise ValueError("num_total_experts must be divisible by world_size")
        if self.num_tokens_per_rank <= 0 or self.num_topk <= 0:
            raise ValueError("split token capacity and top-k must be positive")
        if not self.apply_topk_in_fc1:
            raise ValueError("production split requires apply_topk_in_fc1=True")
        if isinstance(
            self.counter_epoch_banks, bool
        ) or self.counter_epoch_banks not in (
            1,
            2,
        ):
            raise ValueError("counter_epoch_banks must be 1 or 2")
        if self.graph_variant not in ("cold_k0", "steady_k3_reset"):
            raise ValueError("graph_variant must be 'cold_k0' or 'steady_k3_reset'")
        if self.counter_epoch_banks == 2 and self.graph_variant != "steady_k3_reset":
            raise ValueError("two counter banks require steady_k3_reset")
        if self.k1_cluster_shape_mnk != self.k2_cluster_shape_mnk:
            raise ValueError("K1/K2 cluster shapes must match")
        for role, tiler, cluster, sm_count, stages, group_hint in (
            (
                "K1",
                self.k1_mma_tiler_mnk,
                self.k1_cluster_shape_mnk,
                self.k1_sm_count,
                self.k1_num_sched_stages,
                self.k1_group_hint,
            ),
            (
                "K2",
                self.k2_mma_tiler_mnk,
                self.k2_cluster_shape_mnk,
                self.k2_sm_count,
                self.k2_num_sched_stages,
                self.k2_group_hint,
            ),
        ):
            if (
                len(tiler) != 3
                or tiler[0] not in (128, 256)
                or tiler[1] not in (16, 32, 64, 128)
                or tiler[2] not in (128, 256)
            ):
                raise ValueError(f"{role} has unsupported split MMA tiler {tiler}")
            if len(cluster) != 3 or cluster[2] != 1:
                raise ValueError(f"{role} cluster must be an M/N/1 triple")
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in cluster
            ):
                raise ValueError(f"{role} cluster entries must be positive integers")
            cluster_size = cluster[0] * cluster[1]
            if (
                isinstance(sm_count, bool)
                or not isinstance(sm_count, int)
                or sm_count <= 0
                or sm_count % cluster_size
            ):
                raise ValueError(
                    f"{role} SM count must be positive and cluster-aligned"
                )
            for name, value in (
                ("scheduler stages", stages),
                ("group hint", group_hint),
            ):
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                ):
                    raise ValueError(f"{role} {name} must be a positive integer")
        if self.k1_mma_tiler_mnk[1] != self.k2_mma_tiler_mnk[
            1
        ] and self.k1_cluster_shape_mnk != (1, 1, 1):
            raise ValueError(
                "independent K1/K2 token-N tiles require cluster shape (1,1,1)"
            )
        if self.hidden % self.k1_mma_tiler_mnk[2]:
            raise ValueError("hidden must be divisible by the K1 MMA K tile")
        if self.intermediate % self.k2_mma_tiler_mnk[2]:
            raise ValueError("intermediate must be divisible by the K2 MMA K tile")

        # Validate the common Humming tensor/communication ABI with a
        # tactic-neutral K128 proxy.  A fused config incorrectly couples one
        # tile-K to both H and I, while split intentionally permits K1/H and
        # K2/I to select K128/K256 independently.
        _make_shape_validation_config(self)

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def handoff_token_n(self) -> int:
        return max(self.k1_mma_tiler_mnk[1], self.k2_mma_tiler_mnk[1])

    @property
    def workspace_counter_tile_tokens(self) -> int:
        return (
            min(self.k1_mma_tiler_mnk[1], self.k2_mma_tiler_mnk[1])
            * (self.k1_cluster_shape_mnk[1])
        )

    @property
    def max_active_clusters(self) -> Tuple[int, int]:
        def count(sm_count: int, cluster: Tuple[int, int, int]) -> int:
            return sm_count // (cluster[0] * cluster[1])

        return (
            count(self.k1_sm_count, self.k1_cluster_shape_mnk),
            count(self.k2_sm_count, self.k2_cluster_shape_mnk),
        )

    @property
    def tuning_identity(self) -> str:
        return _SPLIT_TUNING_IDENTITY


def _resolve_mxfp4_split_tactic(
    knobs: Optional[Any],
    *,
    world_size: int,
    hidden: int,
    intermediate: int,
    num_total_experts: int,
    num_topk: int,
    num_max_tokens: int,
    gate_up_clamp: Optional[float] = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> dict[str, Any]:
    """Resolve only complete Green-split tactics under the split identity."""

    routing_profile = normalize_sm90_routing_profile(routing_profile)
    from .mxfp4_tuner import (
        hopper_mxfp4_ordered_candidates,
        is_hopper_mxfp4_tactic_shape_compatible,
        require_hopper_mxfp4_tuning_device,
        validate_hopper_mxfp4_tactic,
    )

    if isinstance(knobs, dict):
        # The embedded candidates remain certified for the 132-SM H200, but
        # an explicit benchmark tactic may target another SM90 device. Only
        # the non-H200 partition needs a live-device check, keeping CPU
        # manifest validation and the H200 production path unchanged.
        requested_sms = knobs.get("k1_sm_count", 0) + knobs.get("k2_sm_count", 0)
        total_sms = 132
        if requested_sms != total_sms:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "non-H200 split tactics require a live CUDA device for "
                    "SM-partition validation"
                )
            device = torch.cuda.current_device()
            total_sms = int(
                torch.cuda.get_device_properties(device).multi_processor_count
            )
        return validate_hopper_mxfp4_tactic(
            knobs,
            execution_mode="split",
            total_sms=total_sms,
        )
    if knobs == "auto":
        raise ValueError(
            "direct split tactic resolution cannot execute collective autotune; "
            "pass knobs='auto' through Sm90PullMxfp4MegaKernelBackend"
        )
    if knobs is not None:
        raise ValueError("split knobs must be None or a complete tactic dict")

    from .knob_cache import lookup_knobs

    require_hopper_mxfp4_tuning_device()
    cached = lookup_knobs(
        dtype=_SPLIT_TUNING_IDENTITY,
        fp8_scale_mode="mxfp4_hybrid",
        world_size=world_size,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_total_experts,
        topk=num_topk,
        max_tokens=num_max_tokens,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
    )
    if cached is not None:
        tactic = validate_hopper_mxfp4_tactic(cached, execution_mode="split")
        if not is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode="split",
            hidden=hidden,
            intermediate=intermediate,
        ):
            raise ValueError(
                "MXFP4 split tuning-cache tactic is incompatible with "
                f"hidden={hidden}, intermediate={intermediate}"
            )
        return tactic
    return hopper_mxfp4_ordered_candidates(
        num_max_tokens,
        execution_mode="split",
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=routing_profile,
    )[0]


def _make_shape_validation_config(
    config: MegaMoEHopperMxfp4SplitConfig,
) -> MegaMoEHopperMxfp4Config:
    """Build a tactic-neutral Phase-A proxy for the shared tensor ABI."""

    return MegaMoEHopperMxfp4Config(
        rank=config.rank,
        world_size=config.world_size,
        num_tokens_per_rank=config.num_tokens_per_rank,
        num_topk=config.num_topk,
        num_total_experts=config.num_total_experts,
        hidden=config.hidden,
        intermediate=config.intermediate,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(1, 1, 1),
        load_balance_mode="static",
        in_kernel_fc2_reduce=False,
        token_back_mode="epi_warps",
        apply_topk_in_fc1=True,
        gate_up_clamp=config.gate_up_clamp,
        routing_profile=config.routing_profile,
    )


@dataclass(frozen=True)
class _SplitProblem:
    num_experts_per_rank: int
    intermediate: int
    hidden: int
    world_size: int
    num_topk: int
    num_tokens_per_rank: int
    gate_up_clamp: Optional[float]


@dataclass
class _PreparedBank:
    pair: Any
    k1_request: Any
    k2_request: Any
    compiled_k1: Any
    compiled_k2: Any
    compiled_join: Any
    reset_memsets: tuple[tuple[str, int, int], ...]
    k1_runtime_kwargs: Mapping[str, Any]
    k2_runtime_kwargs: Mapping[str, Any]
    join_runtime_kwargs: Optional[Mapping[str, Any]]

    @property
    def k1_stream(self) -> Any:
        return self.k1_request.kwargs["stream"]

    @property
    def k2_stream(self) -> Any:
        return self.k2_request.kwargs["stream"]


@dataclass
class _PreparedPipeline:
    pair: Any
    green_contexts: Any
    parent_stream: Any
    k3_stream: Any
    compiled_k1: Any
    compiled_k2: Any
    compiled_reset_barrier: Any
    compiled_join: Any
    compiled_k3: Any
    reset_barrier_runtime_kwargs: Mapping[str, Any]
    reset_memsets: tuple[tuple[str, int, int], ...]
    driver: Any
    k1_stream: Any
    k2_stream: Any
    k1_runtime_kwargs: Mapping[str, Any]
    k2_runtime_kwargs: Mapping[str, Any]
    join_runtime_kwargs: Optional[Mapping[str, Any]]
    k3_runtime_kwargs: Mapping[str, Any]
    steady_state_tail_reset: bool
    counter_banks: tuple[_PreparedBank, ...] = ()

    def _launch_memsets(
        self, memsets: tuple[tuple[str, int, int], ...], stream: Any
    ) -> None:
        for label, address, byte_count in memsets:
            if byte_count:
                _checked_driver_result(
                    self.driver,
                    f"cuMemsetD8Async({label})",
                    self.driver.cuMemsetD8Async(address, 0, byte_count, stream),
                )

    def _launch_reset(self, stream: Any) -> None:
        self._launch_memsets(self.reset_memsets, stream)
        kwargs = dict(self.reset_barrier_runtime_kwargs)
        kwargs["stream"] = stream
        self.compiled_reset_barrier(**kwargs)

    def launch_k0(self, stream: Any) -> None:
        self._launch_reset(stream)

    def launch_k1(self, stream: Any) -> None:
        kwargs = dict(self.k1_runtime_kwargs)
        kwargs["stream"] = stream
        self.compiled_k1(**kwargs)

    def launch_k2(self, stream: Any) -> None:
        kwargs = dict(self.k2_runtime_kwargs)
        kwargs["stream"] = stream
        self.compiled_k2(**kwargs)

    def launch_k3(self, stream: Any) -> None:
        if self.compiled_join is not None:
            kwargs = dict(self.join_runtime_kwargs or {})
            kwargs["stream"] = stream
            self.compiled_join(**kwargs)
        kwargs = dict(self.k3_runtime_kwargs)
        kwargs["stream"] = stream
        self.compiled_k3(**kwargs)
        if self.steady_state_tail_reset:
            self._launch_reset(stream)

    def launch_k1_bank(self, bank_index: int, stream: Any) -> None:
        bank = self.counter_banks[bank_index]
        kwargs = dict(bank.k1_runtime_kwargs)
        kwargs["stream"] = stream
        bank.compiled_k1(**kwargs)

    def launch_k2_bank(self, bank_index: int, stream: Any) -> None:
        bank = self.counter_banks[bank_index]
        kwargs = dict(bank.k2_runtime_kwargs)
        kwargs["stream"] = stream
        bank.compiled_k2(**kwargs)

    def launch_k3_bank(self, bank_index: int, stream: Any) -> None:
        bank = self.counter_banks[bank_index]
        if bank.compiled_join is not None:
            kwargs = dict(bank.join_runtime_kwargs or {})
            kwargs["stream"] = stream
            bank.compiled_join(**kwargs)
        kwargs = dict(self.k3_runtime_kwargs)
        kwargs["stream"] = stream
        self.compiled_k3(**kwargs)
        self._launch_memsets(bank.reset_memsets, stream)


class _CapturedGreenExecutor:
    def __init__(self, prepared: _PreparedPipeline, graph_type: Any) -> None:
        self._prepared = prepared
        self._graph_type = graph_type
        self._graph = None
        self._bank_graphs: tuple[Any, ...] = ()
        self._next_bank = 0
        self._closed = False
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def _ensure_usable(self) -> None:
        if self._closed:
            raise Mxfp4SplitLifecycleError("split graph executor is closed")
        if self._poisoned:
            raise Mxfp4SplitSessionPoisonedError("split graph executor is poisoned")

    def capture(self) -> None:
        self._ensure_usable()
        if self._graph is not None or self._bank_graphs:
            return
        p = self._prepared
        completed_graphs: list[Any] = []
        try:
            if p.counter_banks:
                for index, bank in enumerate(p.counter_banks):
                    graph = self._graph_type.capture_steady(
                        k1_stream=bank.k1_stream,
                        k1_launch=lambda stream, i=index: p.launch_k1_bank(i, stream),
                        k2_stream=bank.k2_stream,
                        k2_launch=lambda stream, i=index: p.launch_k2_bank(i, stream),
                        k3_stream=p.k3_stream,
                        k3_launch=lambda stream, i=index: p.launch_k3_bank(i, stream),
                    )
                    completed_graphs.append(graph)
                self._bank_graphs = tuple(completed_graphs)
            elif p.steady_state_tail_reset:
                self._graph = self._graph_type.capture_steady(
                    k1_stream=p.k1_stream,
                    k1_launch=p.launch_k1,
                    k2_stream=p.k2_stream,
                    k2_launch=p.launch_k2,
                    k3_stream=p.k3_stream,
                    k3_launch=p.launch_k3,
                )
            else:
                self._graph = self._graph_type.capture(
                    k0_stream=p.k3_stream,
                    k0_launch=p.launch_k0,
                    k1_stream=p.k1_stream,
                    k1_launch=p.launch_k1,
                    k2_stream=p.k2_stream,
                    k2_launch=p.launch_k2,
                    k3_stream=p.k3_stream,
                    k3_launch=p.launch_k3,
                )
        except BaseException as exc:
            self._poisoned = True
            cleanup_errors = []
            for graph in reversed(completed_graphs):
                try:
                    graph.close(synchronize=True)
                except BaseException as cleanup_exc:
                    cleanup_errors.append(cleanup_exc)
            if cleanup_errors and hasattr(exc, "add_note"):
                exc.add_note(
                    "partial split graph cleanup also failed: "
                    + "; ".join(map(str, cleanup_errors))
                )
            raise

    def launch(self) -> None:
        self._ensure_usable()
        try:
            if self._bank_graphs:
                index = self._next_bank
                self._bank_graphs[index].launch(self._prepared.parent_stream)
                # Parent launches are stream ordered, so alternating immediately
                # is safe without a host synchronization between forwards.
                self._next_bank = 1 - index
            elif self._graph is not None:
                self._graph.launch(self._prepared.parent_stream)
            else:
                raise Mxfp4SplitLifecycleError("split graph has not been captured")
        except BaseException:
            self._poisoned = True
            raise

    def synchronize(self) -> None:
        self._ensure_usable()
        try:
            _checked_driver_result(
                self._prepared.driver,
                "cuStreamSynchronize(parent)",
                self._prepared.driver.cuStreamSynchronize(self._prepared.parent_stream),
            )
        except BaseException:
            self._poisoned = True
            raise

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        graphs = list(self._bank_graphs)
        if self._graph is not None:
            graphs.append(self._graph)
        self._bank_graphs = ()
        self._graph = None
        errors = []
        for graph in reversed(graphs):
            try:
                graph.close(synchronize=True)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise Mxfp4SplitLifecycleError(
                "failed to close split Green graph(s): " + "; ".join(map(str, errors))
            ) from errors[0]


def _checked_driver_result(driver: Any, name: str, result: Any) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or not result:
        raise Mxfp4SplitLifecycleError(
            f"CUDA API {name} returned invalid result {result!r}"
        )
    try:
        status = int(result[0])
    except (TypeError, ValueError) as exc:
        raise Mxfp4SplitLifecycleError(
            f"CUDA API {name} returned invalid CUresult"
        ) from exc
    if status:
        status_name = getattr(result[0], "name", str(result[0]))
        raise Mxfp4SplitLifecycleError(
            f"CUDA API {name} failed with {status_name} ({status})"
        )
    return tuple(result[1:])


def _create_nonblocking_stream(driver: Any) -> Any:
    flags = getattr(driver, "CUstream_flags", None)
    flag = getattr(flags, "CU_STREAM_NON_BLOCKING", None)
    if flag is None or not callable(getattr(driver, "cuStreamCreate", None)):
        raise Mxfp4SplitUnavailableError(
            "CUDA driver binding lacks nonblocking primary-context streams"
        )
    (stream,) = _checked_driver_result(
        driver, "cuStreamCreate", driver.cuStreamCreate(flag)
    )
    return stream


def _to_cute(tensor: torch.Tensor, assumed_align: int = 16) -> Any:
    import cutlass.torch as cutlass_torch

    result = cutlass_torch.from_dlpack(tensor, assumed_align=assumed_align)
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return result.mark_layout_dynamic(leading_dim=leading_dim)


def _to_cute_ptr(tensor: torch.Tensor) -> Any:
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.typing import AddressSpace

    return cute.runtime.make_ptr(
        cutlass.Uint8,
        tensor.data_ptr(),
        AddressSpace.gmem,
        assumed_align=16,
    )


def _launch_kwargs(request: Any) -> dict[str, Any]:
    return {
        name: value
        for name, value in request.kwargs.items()
        if name not in ("max_active_clusters", "options")
    }


class MegaMoEHopperMxfp4SplitSession:
    """Fixed-pointer owner for compiled roles, Green streams, and CUDA graphs."""

    def __init__(
        self,
        config: MegaMoEHopperMxfp4SplitConfig,
        *,
        process_group: Any = None,
    ) -> None:
        self.config = config
        self._process_group = process_group
        self._generation = next(_SPLIT_GENERATIONS)
        self._input_validator = MegaMoEHopperMxfp4Frontend(
            _make_shape_validation_config(config)
        )
        self._inputs: Optional[MegaMoEHopperMxfp4Inputs] = None
        self._fixed_pointer_key: Optional[tuple[Any, ...]] = None
        self._pair: Any = None
        self._pairs: tuple[Any, ...] = ()
        self._workspace_contract: Any = None
        self._local_workspace: Optional[torch.Tensor] = None
        self._shared_workspace: Optional[torch.Tensor] = None
        self._green_contexts: Any = None
        self._k3_stream: Any = None
        self._driver: Any = None
        self._prepared: Optional[_PreparedPipeline] = None
        self._executor: Optional[_CapturedGreenExecutor] = None
        self._captured = False
        self._destroyed = False
        self._poison_reason: Optional[str] = None

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def workspace_contract(self) -> Any:
        return self._workspace_contract

    @property
    def fixed_pointer_key(self) -> Optional[tuple[Any, ...]]:
        return self._fixed_pointer_key

    @property
    def poisoned(self) -> bool:
        return self._poison_reason is not None

    @property
    def destroyed(self) -> bool:
        return self._destroyed

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def graph_variant(self) -> str:
        return self.config.graph_variant

    @property
    def green_sm_counts(self) -> Tuple[int, int]:
        if self._green_contexts is None:
            return self.config.k1_sm_count, self.config.k2_sm_count
        return tuple(self._green_contexts.sm_counts)

    @property
    def max_active_clusters(self) -> Tuple[int, int]:
        return self.config.max_active_clusters

    @property
    def local_workspace(self) -> Optional[torch.Tensor]:
        return self._local_workspace

    @property
    def shared_workspace(self) -> Optional[torch.Tensor]:
        return self._shared_workspace

    def _ensure_usable(self) -> None:
        if self._destroyed:
            raise Mxfp4SplitLifecycleError("split session is destroyed")
        if self._poison_reason is not None:
            raise Mxfp4SplitSessionPoisonedError(
                "split session is poisoned: " + self._poison_reason
            )

    def poison(self, reason: Any) -> None:
        if self._poison_reason is None:
            self._poison_reason = str(reason)

    @staticmethod
    def _pointer_key(inputs: MegaMoEHopperMxfp4Inputs) -> tuple[Any, ...]:
        tensors = (
            inputs.activation,
            inputs.activation_sf,
            inputs.topk_idx,
            inputs.topk_weights,
            inputs.fc1_weight,
            inputs.fc1_weight_sf,
            inputs.fc1_activation_dequant_scale,
            inputs.fc1_weight_dequant_scale,
            inputs.fc2_weight,
            inputs.fc2_weight_sf,
            inputs.fc2_activation_dequant_scale,
            inputs.fc2_weight_dequant_scale,
            inputs.output_activation,
        )
        return (
            *(tensor.data_ptr() for tensor in tensors),
            *(tuple(tensor.shape) for tensor in tensors),
            torch.cuda.current_device(),
            int(torch.cuda.current_stream().cuda_stream),
        )

    def bind_inputs(self, inputs: MegaMoEHopperMxfp4Inputs) -> None:
        self._ensure_usable()
        self._input_validator._validate_inputs(
            inputs, num_tokens=inputs.activation.shape[0]
        )
        key = self._pointer_key(inputs)
        if self._fixed_pointer_key is None:
            self._fixed_pointer_key = key
            self._inputs = inputs
            return
        if key != self._fixed_pointer_key:
            raise Mxfp4SplitLifecycleError(
                "split graph inputs/weights/output/stream changed after fixed "
                "pointer binding; create a new unpooled split session"
            )
        self._inputs = inputs

    def _make_plan(self) -> Any:
        from moe_hopper_fp8.split_mega_runner import (
            SplitMegaPlan,
            SplitMegaTactic,
        )

        def impl(
            tiler: Tuple[int, int, int],
            cluster: Tuple[int, int, int],
            group_hint: Optional[int],
            stages: Optional[int],
        ) -> Any:
            return SplitMegaTactic(
                mma_tiler_mnk=tiler,
                cluster_shape_mnk=cluster,
                use_2cta_instrs=False,
                force_static_sched=True,
                clc_bundle_size=self.config.clc_bundle_size,
                num_sched_stages=stages,
                load_balance_mode="static",
                group_hint=group_hint,
                in_kernel_fc2_reduce=False,
                token_back_mode="epi_warps",
                flag_batch=self.config.flag_batch,
                epi_flag_batch=self.config.epi_flag_batch,
            )

        return SplitMegaPlan(
            fc1_impl=impl(
                self.config.k1_mma_tiler_mnk,
                self.config.k1_cluster_shape_mnk,
                self.config.k1_group_hint,
                self.config.k1_num_sched_stages,
            ),
            fc2_impl=impl(
                self.config.k2_mma_tiler_mnk,
                self.config.k2_cluster_shape_mnk,
                self.config.k2_group_hint,
                self.config.k2_num_sched_stages,
            ),
            k1_sm_count=self.config.k1_sm_count,
            k2_sm_count=self.config.k2_sm_count,
        )

    def _dist_barrier(self) -> None:
        if self.config.world_size == 1:
            torch.cuda.synchronize()
            return
        import torch.distributed as dist

        if not dist.is_available() or not dist.is_initialized():
            raise Mxfp4SplitLifecycleError(
                "multi-rank split reset requires initialized torch.distributed"
            )
        torch.cuda.synchronize()
        dist.barrier(group=self._process_group)
        torch.cuda.synchronize()

    def _reset_host_state(self) -> None:
        if (
            self._workspace_contract is None
            or self._local_workspace is None
            or self._shared_workspace is None
            or self._inputs is None
        ):
            raise Mxfp4SplitLifecycleError("split workspaces are not prepared")
        contract = self._workspace_contract
        for bank in range(contract.counter_epoch_banks):
            lo, ln = contract.counter_bank_span("local", bank)
            so, sn = contract.counter_bank_span("shared", bank)
            if ln:
                self._local_workspace.narrow(0, lo, ln).zero_()
            if sn:
                self._shared_workspace.narrow(0, so, sn).zero_()
        self._pairs[0].combine_quant_view(self._shared_workspace).zero_()
        self._inputs.output_activation.zero_()

    def _initialize_epoch_state(self) -> None:
        p = self._prepared
        if p is None:
            raise Mxfp4SplitLifecycleError("split pipeline is not prepared")
        self._reset_host_state()
        # Host-side zero_ calls are issued on the parent stream. Finish them
        # before launching reset work on the independent K3 stream.
        torch.cuda.synchronize()
        if p.counter_banks:
            for bank in p.counter_banks:
                p._launch_memsets(bank.reset_memsets, p.k3_stream)
        else:
            p.launch_k0(p.k3_stream)
        _checked_driver_result(
            p.driver,
            "cuStreamSynchronize(K0 initialization)",
            p.driver.cuStreamSynchronize(p.k3_stream),
        )

    def _reset_memsets(
        self, pair: Any, combine_quant: torch.Tensor
    ) -> tuple[tuple[str, int, int], ...]:
        assert self._local_workspace is not None
        assert self._shared_workspace is not None
        if pair.workspace.counter_epoch_banks == 2:
            local_offset, local_bytes = pair.selected_counter_bank_span("local")
            shared_offset, shared_bytes = pair.selected_counter_bank_span("shared")
        else:
            local_offset = shared_offset = 0
            local_bytes = int(pair.workspace.local_zero_i32_count) * 4
            shared_bytes = int(pair.workspace.shared_zero_i32_count) * 4
        return (
            (
                f"local_counter_bank{pair.counter_epoch_bank}",
                self._local_workspace.data_ptr() + local_offset,
                local_bytes,
            ),
            (
                f"shared_counter_bank{pair.counter_epoch_bank}",
                self._shared_workspace.data_ptr() + shared_offset,
                shared_bytes,
            ),
            (
                "combine_quant",
                combine_quant.data_ptr(),
                combine_quant.numel() * combine_quant.element_size(),
            ),
        )

    def _prepare(self) -> None:
        self._ensure_usable()
        if self._prepared is not None:
            return
        ensure_not_capturing("MXFP4 split JIT/Green Context/workspace preparation")
        if self._inputs is None:
            raise Mxfp4SplitLifecycleError("bind_inputs() is required before warmup")

        try:
            import cuda.bindings.driver as cuda
            import cutlass.cute as cute
            import cutlass.torch as cutlass_torch

            from moe_hopper_fp8.green_context import (
                GreenContextSplit,
                check_green_context_support,
            )
            from moe_hopper_fp8.green_graph import GreenGraph
            from moe_hopper_fp8.split_mega_runner import (
                build_mxfp4_split_kernel_pair,
            )
            from src.sym_buffer import SymBufferHost

            device = torch.cuda.current_device()
            support = check_green_context_support(device)
            if not support.supported:
                raise Mxfp4SplitUnavailableError(
                    "CUDA Green Context support is required: " + str(support.reason)
                )
            if support.total_sms != self.config.k1_sm_count + self.config.k2_sm_count:
                raise Mxfp4SplitUnavailableError(
                    "requested split SM budget does not cover the device: "
                    f"{self.config.k1_sm_count}+{self.config.k2_sm_count} "
                    f"!= {support.total_sms}"
                )
            green = GreenContextSplit.create(
                self.config.k1_sm_count, device_ordinal=device
            )
            self._green_contexts = green
            if tuple(green.sm_counts) != (
                self.config.k1_sm_count,
                self.config.k2_sm_count,
            ):
                raise Mxfp4SplitUnavailableError(
                    "driver Green partition differs from requested tactic plan"
                )

            plan = self._make_plan()
            problem = _SplitProblem(
                num_experts_per_rank=self.config.num_experts_per_rank,
                intermediate=2 * self.config.intermediate,
                hidden=self.config.hidden,
                world_size=self.config.world_size,
                num_topk=self.config.num_topk,
                num_tokens_per_rank=self.config.num_tokens_per_rank,
                gate_up_clamp=self.config.gate_up_clamp,
            )
            pair_kwargs = dict(
                rank=self.config.rank,
                kind="fp8_e4m3",
                fp8_scale_mode="mxfp4_hybrid",
                fp8_accum_mode="1xacc",
                apply_topk_in_fc1=True,
            )
            if self.config.counter_epoch_banks == 2:
                pairs = tuple(
                    build_mxfp4_split_kernel_pair(
                        problem,
                        plan,
                        counter_epoch_banks=2,
                        counter_epoch_bank=bank,
                        **pair_kwargs,
                    )
                    for bank in (0, 1)
                )
                if pairs[0].workspace != pairs[1].workspace:
                    raise Mxfp4SplitLifecycleError(
                        "counter-bank kernel pairs disagree on workspace ABI"
                    )
            else:
                pairs = (build_mxfp4_split_kernel_pair(problem, plan, **pair_kwargs),)
            self._pairs = pairs
            self._pair = pairs[0]
            self._workspace_contract = pairs[0].workspace

            local_bytes, shared_bytes = pairs[0].get_workspace_sizes()
            self._local_workspace = torch.zeros(
                (local_bytes,), dtype=torch.uint8, device="cuda"
            )
            self._shared_workspace = sym_zeros((shared_bytes,), torch.uint8)
            symmetric_base, peer_offsets = _compute_peer_offsets(
                self._shared_workspace, self.config.world_size
            )
            peer_mapper = None
            if self.config.world_size > 1:
                peer_mapper = SymBufferHost(
                    base_addr=symmetric_base,
                    offsets=tuple(peer_offsets),
                    rank_idx=self.config.rank,
                    num_max_ranks=self.config.world_size,
                )

            i = self._inputs
            runtime = dict(
                activation=_to_cute(i.activation),
                activation_sf=_to_cute(i.activation_sf),
                topk_idx=_to_cute(i.topk_idx),
                topk_weights=_to_cute(i.topk_weights),
                fc1_weight=_to_cute(i.fc1_weight),
                fc1_weight_sf=_to_cute(i.fc1_weight_sf),
                fc1_weight_dequant_scale=_to_cute(
                    i.fc1_weight_dequant_scale, assumed_align=4
                ),
                fc2_weight=_to_cute(i.fc2_weight),
                fc2_weight_sf=_to_cute(i.fc2_weight_sf),
                fc2_weight_dequant_scale=_to_cute(
                    i.fc2_weight_dequant_scale, assumed_align=4
                ),
                local_workspace=_to_cute_ptr(self._local_workspace),
                shared_workspace=_to_cute_ptr(self._shared_workspace),
                peer_rank_ptr_mapper_host=peer_mapper,
            )
            k1_stream = cuda.CUstream(int(green.k1.raw_stream))
            k2_stream = cuda.CUstream(int(green.k2.raw_stream))
            options = "iket" if self.config.enable_iket else None
            entries = []
            for pair in pairs:
                k1_request, k2_request = pair.compile_requests(
                    runtime,
                    k1_stream=k1_stream,
                    k2_stream=k2_stream,
                    options=options,
                )
                entries.append(
                    (
                        pair,
                        k1_request,
                        k2_request,
                        k1_request.compile(cute),
                        k2_request.compile(cute),
                    )
                )

            combine_quant = pairs[0].combine_quant_view(self._shared_workspace)
            combine_cute = cutlass_torch.from_dlpack(combine_quant, assumed_align=16)
            output_cute = cutlass_torch.from_dlpack(
                i.output_activation, assumed_align=16
            )
            parent_stream = cuda.CUstream(
                int(torch.cuda.current_stream(device).cuda_stream)
            )
            k3_stream = _create_nonblocking_stream(cuda)
            self._driver = cuda
            self._k3_stream = k3_stream

            compiled_reset = None
            reset_runtime: dict[str, Any] = {}
            if self.config.counter_epoch_banks == 1:
                from moe_hopper_fp8.split_epoch_reset import SplitEpochResetBarrier

                reset_runtime = dict(
                    barrier_signal=_to_cute(
                        pairs[0].reset_barrier_signal_view(self._shared_workspace),
                        assumed_align=4,
                    ),
                    phase_counter=_to_cute(
                        pairs[0].reset_barrier_phase_view(self._local_workspace),
                        assumed_align=4,
                    ),
                    peer_rank_ptr_mapper_host=peer_mapper,
                    stream=k3_stream,
                )
                compile_kwargs = dict(reset_runtime)
                if options is not None:
                    compile_kwargs["options"] = options
                compiled_reset = cute.compile(
                    SplitEpochResetBarrier(self.config.world_size, self.config.rank),
                    **compile_kwargs,
                )

            join_entries = []
            for pair in pairs:
                compiled_join = None
                join_runtime = None
                if self.config.world_size > 1:
                    from moe_hopper_fp8.split_k3_join import SplitK2GlobalJoin

                    join_runtime = dict(
                        join_counter=_to_cute(
                            pair.join_counter_view(self._shared_workspace),
                            assumed_align=4,
                        ),
                        peer_rank_ptr_mapper_host=peer_mapper,
                        stream=k3_stream,
                    )
                    compile_kwargs = dict(join_runtime)
                    if options is not None:
                        compile_kwargs["options"] = options
                    compiled_join = cute.compile(
                        SplitK2GlobalJoin(self.config.world_size, self.config.rank),
                        **compile_kwargs,
                    )
                join_entries.append((compiled_join, join_runtime))

            k3_runtime = dict(
                combine_quant=combine_cute,
                combine_sf=None,
                reduced_output=output_cute,
                topk_idx=_to_cute(i.topk_idx),
                topk_score=None,
                stream=k3_stream,
            )
            k3_compile = dict(k3_runtime)
            if options is not None:
                k3_compile["options"] = options
            compiled_k3 = cute.compile(pairs[0].make_k3(sm_arch="sm_90"), **k3_compile)

            _, k1_req, k2_req, compiled_k1, compiled_k2 = entries[0]
            compiled_join, join_runtime = join_entries[0]
            bank_entries: tuple[_PreparedBank, ...] = ()
            if self.config.counter_epoch_banks == 2:
                bank_entries = tuple(
                    _PreparedBank(
                        pair=pair,
                        k1_request=req1,
                        k2_request=req2,
                        compiled_k1=comp1,
                        compiled_k2=comp2,
                        compiled_join=join[0],
                        reset_memsets=self._reset_memsets(pair, combine_quant),
                        k1_runtime_kwargs=_launch_kwargs(req1),
                        k2_runtime_kwargs=_launch_kwargs(req2),
                        join_runtime_kwargs=join[1],
                    )
                    for (pair, req1, req2, comp1, comp2), join in zip(
                        entries, join_entries, strict=True
                    )
                )
            prepared = _PreparedPipeline(
                pair=pairs[0],
                green_contexts=green,
                parent_stream=parent_stream,
                k3_stream=k3_stream,
                compiled_k1=compiled_k1,
                compiled_k2=compiled_k2,
                compiled_reset_barrier=compiled_reset,
                compiled_join=compiled_join,
                compiled_k3=compiled_k3,
                reset_barrier_runtime_kwargs=reset_runtime,
                reset_memsets=self._reset_memsets(pairs[0], combine_quant),
                driver=cuda,
                k1_stream=k1_stream,
                k2_stream=k2_stream,
                k1_runtime_kwargs=_launch_kwargs(k1_req),
                k2_runtime_kwargs=_launch_kwargs(k2_req),
                join_runtime_kwargs=join_runtime,
                k3_runtime_kwargs=k3_runtime,
                steady_state_tail_reset=(
                    self.config.graph_variant == "steady_k3_reset"
                    and self.config.counter_epoch_banks == 1
                ),
                counter_banks=bank_entries,
            )
            self._prepared = prepared
            self._executor = _CapturedGreenExecutor(prepared, GreenGraph)
        except BaseException as exc:
            self.poison(f"prepare failed: {exc}")
            cleanup_errors = self._release_owned_resources()
            if cleanup_errors and hasattr(exc, "add_note"):
                exc.add_note(
                    "partial split session cleanup also failed: "
                    + "; ".join(map(str, cleanup_errors))
                )
            raise

    def _eager_concurrent_warmup(self) -> None:
        p = self._prepared
        if p is None:
            raise Mxfp4SplitLifecycleError("split pipeline is not prepared")
        try:
            self._dist_barrier()
            self._initialize_epoch_state()
            self._dist_barrier()
            if p.counter_banks:
                p.launch_k1_bank(0, p.counter_banks[0].k1_stream)
                p.launch_k2_bank(0, p.counter_banks[0].k2_stream)
            else:
                p.launch_k1(p.k1_stream)
                p.launch_k2(p.k2_stream)
            _checked_driver_result(
                p.driver,
                "cuStreamSynchronize(K1 warmup)",
                p.driver.cuStreamSynchronize(p.k1_stream),
            )
            _checked_driver_result(
                p.driver,
                "cuStreamSynchronize(K2 warmup)",
                p.driver.cuStreamSynchronize(p.k2_stream),
            )
            if p.counter_banks:
                p.launch_k3_bank(0, p.k3_stream)
            else:
                p.launch_k3(p.k3_stream)
            _checked_driver_result(
                p.driver,
                "cuStreamSynchronize(K3 warmup)",
                p.driver.cuStreamSynchronize(p.k3_stream),
            )
            self._dist_barrier()
        except BaseException as exc:
            self.poison(f"eager warmup failed: {exc}")
            raise Mxfp4SplitSessionPoisonedError("split eager warmup failed") from exc

    def warmup(self, inputs: MegaMoEHopperMxfp4Inputs) -> None:
        self.bind_inputs(inputs)
        if self._prepared is not None:
            return
        self._prepare()
        self._eager_concurrent_warmup()

    def capture(
        self,
        inputs: MegaMoEHopperMxfp4Inputs,
        *,
        graph_variant: Optional[str] = None,
    ) -> None:
        self._ensure_usable()
        if graph_variant is not None and graph_variant != self.config.graph_variant:
            raise Mxfp4SplitLifecycleError(
                "graph_variant is compile/session identity; create a new session"
            )
        self.warmup(inputs)
        if self._captured:
            return
        assert self._executor is not None
        try:
            self._dist_barrier()
            self._initialize_epoch_state()
            self._dist_barrier()
            self._executor.capture()
            self._captured = True
            self._dist_barrier()
        except BaseException as exc:
            self.poison(f"capture failed: {exc}")
            raise Mxfp4SplitSessionPoisonedError(
                "split CUDA graph capture failed"
            ) from exc

    def replay(
        self,
        *,
        sync: bool = False,
        inputs: Optional[MegaMoEHopperMxfp4Inputs] = None,
    ) -> torch.Tensor:
        self._ensure_usable()
        if inputs is not None:
            self.bind_inputs(inputs)
        if not self._captured or self._executor is None or self._inputs is None:
            raise Mxfp4SplitLifecycleError(
                "capture() is required before split graph replay"
            )
        if int(torch.cuda.current_stream().cuda_stream) != int(
            self._prepared.parent_stream
        ):
            raise Mxfp4SplitLifecycleError(
                "split graph replay must use its fixed primary-context stream"
            )
        try:
            self._executor.launch()
            if sync:
                self._executor.synchronize()
        except BaseException as exc:
            self.poison(f"replay failed: {exc}")
            raise Mxfp4SplitSessionPoisonedError(
                "split graph replay failed and poisoned the session"
            ) from exc
        return self._inputs.output_activation

    def synchronize(self) -> None:
        self._ensure_usable()
        if self._executor is None:
            return
        try:
            self._executor.synchronize()
        except BaseException as exc:
            self.poison(f"synchronize failed: {exc}")
            raise Mxfp4SplitSessionPoisonedError(
                "split graph synchronization failed and poisoned the session"
            ) from exc

    def _workspace_region_view(self, name: str, dtype: torch.dtype) -> torch.Tensor:
        contract = self._workspace_contract
        workspace = self._local_workspace
        if contract is None or workspace is None:
            raise Mxfp4SplitLifecycleError("split workspace is not prepared")
        if (
            workspace.dtype is not torch.uint8
            or workspace.ndim != 1
            or workspace.numel() < int(contract.local_total_bytes)
        ):
            raise Mxfp4SplitLifecycleError(
                "local split workspace is smaller than its byte-exact ABI"
            )

        allowed_dtypes = {
            torch.float8_e4m3fn: ("Float8E4M3FN", "cutlass.Float8E4M3FN"),
            torch.float32: ("Float32", "cutlass.Float32"),
            torch.uint8: ("Uint8", "cutlass.Uint8"),
        }
        expected_contract_dtypes = allowed_dtypes.get(dtype)
        if expected_contract_dtypes is None:
            raise Mxfp4SplitLifecycleError(
                f"unsupported split workspace view dtype {dtype}"
            )

        region = contract.region("local", name)
        if region.dtype not in expected_contract_dtypes:
            raise Mxfp4SplitLifecycleError(
                f"local.{name} contract dtype {region.dtype!r} is not one of "
                f"{expected_contract_dtypes}"
            )
        shape = tuple(int(dim) for dim in region.shape)
        expected_stride = [1]
        for dim in reversed(shape[1:]):
            expected_stride.append(expected_stride[-1] * dim)
        expected_stride.reverse()
        if tuple(int(value) for value in region.stride) != tuple(expected_stride):
            raise Mxfp4SplitLifecycleError(
                f"local.{name} is not row-major: stride={region.stride}, "
                f"expected={tuple(expected_stride)}"
            )

        numel = 1
        for dim in shape:
            if dim < 0:
                raise Mxfp4SplitLifecycleError(
                    f"local.{name} has negative shape dimension {shape}"
                )
            numel *= dim
        expected_bytes = numel * torch.empty((), dtype=dtype).element_size()
        offset = int(region.byte_offset)
        byte_size = int(region.byte_size)
        end = offset + byte_size
        if (
            expected_bytes != byte_size
            or offset < 0
            or byte_size < 0
            or end > int(contract.local_total_bytes)
            or end > workspace.numel()
        ):
            raise Mxfp4SplitLifecycleError(
                f"local.{name} byte contract mismatch: expected={expected_bytes}, "
                f"offset={offset}, bytes={byte_size}, total={contract.local_total_bytes}"
            )
        alignment = int(region.alignment)
        if alignment <= 0 or (workspace.data_ptr() + offset) % alignment:
            raise Mxfp4SplitLifecycleError(
                f"local.{name} violates contract alignment {alignment}"
            )
        return workspace.narrow(0, offset, byte_size).view(dtype).reshape(shape)

    def handoff_payload_view(self) -> torch.Tensor:
        """Read-only-by-contract E4M3 FC1 handoff view for verification."""

        return self._workspace_region_view("fc1_output", torch.float8_e4m3fn)

    def handoff_scale_view(self) -> torch.Tensor:
        """Read-only-by-contract FP32 per-token/K64 FC1 scale view."""

        return self._workspace_region_view("fc1_output_sf", torch.float32)

    def handoff_metadata_view(self) -> torch.Tensor:
        """Read-only token source metadata for byte-exact handoff oracles."""

        return self._workspace_region_view("token_src_metadata", torch.uint8)

    def close(self) -> None:
        self.destroy()

    def _release_owned_resources(self) -> list[BaseException]:
        errors: list[BaseException] = []
        executor, self._executor = self._executor, None
        if executor is not None:
            try:
                executor.close()
            except BaseException as exc:
                errors.append(exc)

        driver, k3_stream = self._driver, self._k3_stream
        self._driver = None
        self._k3_stream = None
        if driver is not None and k3_stream is not None:
            try:
                _checked_driver_result(
                    driver,
                    "cuStreamDestroy(K3)",
                    driver.cuStreamDestroy(k3_stream),
                )
            except BaseException as exc:
                errors.append(exc)

        green, self._green_contexts = self._green_contexts, None
        if green is not None:
            try:
                green.close()
            except BaseException as exc:
                errors.append(exc)

        shared, self._shared_workspace = self._shared_workspace, None
        if shared is not None:
            try:
                free_sym_tensor(shared)
            except BaseException as exc:
                errors.append(exc)

        self._local_workspace = None
        self._prepared = None
        self._pairs = ()
        self._pair = None
        self._workspace_contract = None
        self._captured = False
        self._inputs = None
        self._fixed_pointer_key = None
        return errors

    def destroy(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        errors = self._release_owned_resources()
        if errors:
            raise Mxfp4SplitLifecycleError(
                "split session teardown failed: " + "; ".join(map(str, errors))
            ) from errors[0]

    def __enter__(self) -> "MegaMoEHopperMxfp4SplitSession":
        self._ensure_usable()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Literal[False]:
        self.destroy()
        return False


@dataclass
class MegaMoEHopperMxfp4SplitSymmBuffer:
    """Unpooled symmetric staging owner for one split graph session."""

    num_total_experts: int
    num_max_tokens: int
    num_topk: int
    hidden: int
    intermediate: int
    rank: int
    world_size: int
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor
    _session: MegaMoEHopperMxfp4SplitSession
    _sym_roots: list[torch.Tensor] = field(default_factory=list)
    _destroyed: bool = False

    @property
    def kind(self) -> str:
        return "fp8_e4m3"

    @property
    def fp8_scale_mode(self) -> str:
        return "mxfp4_hybrid"

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    @property
    def session(self) -> MegaMoEHopperMxfp4SplitSession:
        return self._session

    def destroy(self) -> None:
        if self._destroyed:
            return
        errors = []
        try:
            self._session.destroy()
        except BaseException as exc:
            errors.append(exc)
        for root in reversed(self._sym_roots):
            try:
                free_sym_tensor(root)
            except BaseException as exc:
                errors.append(exc)
        self._sym_roots.clear()
        self._destroyed = True
        if errors:
            raise Mxfp4SplitLifecycleError(
                "split buffer teardown failed: " + "; ".join(map(str, errors))
            ) from errors[0]


def get_symm_buffer_for_hopper_mxfp4_split_mega_moe(
    num_total_experts: int,
    num_max_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate: int,
    rank: int,
    world_size: int,
    *,
    split_k1_mma_tiler_mnk: Tuple[int, int, int] = (128, 32, 128),
    split_k2_mma_tiler_mnk: Tuple[int, int, int] = (128, 32, 128),
    split_k1_cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1),
    split_k2_cluster_shape_mnk: Tuple[int, int, int] = (1, 1, 1),
    split_k1_group_hint: Optional[int] = None,
    split_k2_group_hint: Optional[int] = None,
    split_k1_num_sched_stages: Optional[int] = None,
    split_k2_num_sched_stages: Optional[int] = None,
    split_k1_sm_count: int,
    split_k2_sm_count: int,
    split_counter_epoch_banks: Literal[1, 2] = 1,
    split_graph_variant: Literal["cold_k0", "steady_k3_reset"] = ("steady_k3_reset"),
    clc_bundle_size: Optional[int] = None,
    flag_batch: int = 1,
    epi_flag_batch: Tuple[int, int] = (2, 4),
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    split_enable_iket: bool = False,
    process_group: Any = None,
    routing_profile: str = SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
) -> MegaMoEHopperMxfp4SplitSymmBuffer:
    """Allocate fixed symmetric inputs and an unpooled split session."""

    routing_profile = normalize_sm90_routing_profile(routing_profile)
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    config = MegaMoEHopperMxfp4SplitConfig(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=num_max_tokens,
        num_topk=num_topk,
        num_total_experts=num_total_experts,
        hidden=hidden,
        intermediate=intermediate,
        k1_mma_tiler_mnk=split_k1_mma_tiler_mnk,
        k2_mma_tiler_mnk=split_k2_mma_tiler_mnk,
        k1_cluster_shape_mnk=split_k1_cluster_shape_mnk,
        k2_cluster_shape_mnk=split_k2_cluster_shape_mnk,
        k1_sm_count=split_k1_sm_count,
        k2_sm_count=split_k2_sm_count,
        k1_group_hint=split_k1_group_hint,
        k2_group_hint=split_k2_group_hint,
        k1_num_sched_stages=split_k1_num_sched_stages,
        k2_num_sched_stages=split_k2_num_sched_stages,
        counter_epoch_banks=split_counter_epoch_banks,
        graph_variant=split_graph_variant,
        clc_bundle_size=clc_bundle_size,
        flag_batch=flag_batch,
        epi_flag_batch=epi_flag_batch,
        gate_up_clamp=clamp,
        enable_iket=split_enable_iket,
        routing_profile=routing_profile,
    )
    session = MegaMoEHopperMxfp4SplitSession(config, process_group=process_group)
    roots: list[torch.Tensor] = []
    try:
        x, x_root = _sym_zeros_byte_view_1b(
            (num_max_tokens, hidden), torch.float8_e4m3fn
        )
        roots.append(x_root)
        x_sf = sym_zeros((num_max_tokens, 4), torch.float32)
        roots.append(x_sf)
        topk_idx = sym_zeros((num_max_tokens, num_topk), torch.int64)
        topk_idx.fill_(-1)
        roots.append(topk_idx)
        topk_weights = sym_zeros((num_max_tokens, num_topk), torch.float32)
        roots.append(topk_weights)
        output = sym_zeros((num_max_tokens, hidden), torch.bfloat16)
        roots.append(output)
        return MegaMoEHopperMxfp4SplitSymmBuffer(
            num_total_experts=num_total_experts,
            num_max_tokens=num_max_tokens,
            num_topk=num_topk,
            hidden=hidden,
            intermediate=intermediate,
            rank=rank,
            world_size=world_size,
            x=x,
            x_sf=x_sf,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            output_activation=output,
            _session=session,
            _sym_roots=roots,
        )
    except BaseException as exc:
        cleanup_errors: list[BaseException] = []
        try:
            session.destroy()
        except BaseException as cleanup_exc:
            cleanup_errors.append(cleanup_exc)
        for root in reversed(roots):
            try:
                free_sym_tensor(root)
            except BaseException as cleanup_exc:
                cleanup_errors.append(cleanup_exc)
        if cleanup_errors and hasattr(exc, "add_note"):
            exc.add_note(
                "partial split input allocation cleanup also failed: "
                + "; ".join(map(str, cleanup_errors))
            )
        raise


@dataclass
class _Mxfp4InputView:
    x: torch.Tensor
    x_sf: torch.Tensor
    topk_idx: torch.Tensor
    topk_weights: torch.Tensor
    output_activation: torch.Tensor


def _split_inputs(
    buffer: MegaMoEHopperMxfp4SplitSymmBuffer,
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
) -> MegaMoEHopperMxfp4Inputs:
    # Reuse the strict four-slot Phase-A adapter through a structural view.
    view = _Mxfp4InputView(
        x=buffer.x,
        x_sf=buffer.x_sf,
        topk_idx=buffer.topk_idx,
        topk_weights=buffer.topk_weights,
        output_activation=buffer.output_activation,
    )
    return _build_mxfp4_inputs(view, transformed_l1, transformed_l2)


def hopper_mxfp4_split_mega_moe(
    y: Optional[torch.Tensor],
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
    symm_buffer: MegaMoEHopperMxfp4SplitSymmBuffer,
    *,
    num_tokens: Optional[int] = None,
    gate_up_clamp: Optional[float] = None,
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    sync: bool = False,
) -> Optional[torch.Tensor]:
    """Compile/capture once, then replay the concurrent split graph."""

    if not fast_math:
        warnings.warn(
            "fast_math=False has no effect in the CuTeDSL SM90 MXFP4 split path.",
            UserWarning,
            stacklevel=2,
        )
    if symm_buffer._destroyed:
        raise Mxfp4SplitLifecycleError("split symmetric buffer is destroyed")
    n = symm_buffer.num_max_tokens if num_tokens is None else num_tokens
    if not 0 <= n <= symm_buffer.num_max_tokens:
        raise ValueError("num_tokens is outside split workspace capacity")
    clamp = resolve_gate_up_clamp(
        gate_up_clamp=gate_up_clamp,
        activation_clamp=activation_clamp,
    )
    if clamp is not None and clamp != symm_buffer.session.config.gate_up_clamp:
        raise Mxfp4SplitLifecycleError(
            "gate_up_clamp is split compile identity; create a new session"
        )
    if y is not None:
        if y.shape != (n, symm_buffer.hidden) or y.dtype != torch.bfloat16:
            raise ValueError(
                f"split output must be bfloat16 ({n}, {symm_buffer.hidden})"
            )
    inputs = _split_inputs(symm_buffer, transformed_l1, transformed_l2)
    if not symm_buffer.session.captured:
        symm_buffer.session.capture(inputs)
    else:
        symm_buffer.session.bind_inputs(inputs)
    output = symm_buffer.session.replay(sync=False)
    result: Optional[torch.Tensor]
    if y is None:
        result = output[:n]
    else:
        y.copy_(output[:n])
        result = None
    if sync:
        symm_buffer.session.synchronize()
    return result


def hopper_mxfp4_split_mega_launch_thunk(
    transformed_l1: TransformedMxfp4Weights,
    transformed_l2: TransformedMxfp4Weights,
    symm_buffer: MegaMoEHopperMxfp4SplitSymmBuffer,
) -> Callable[[], None]:
    """Return a zero-argument graph replay thunk with fixed pointers."""

    inputs = _split_inputs(symm_buffer, transformed_l1, transformed_l2)
    if not symm_buffer.session.captured:
        symm_buffer.session.capture(inputs)
    else:
        symm_buffer.session.bind_inputs(inputs)

    def launch() -> None:
        symm_buffer.session.replay(sync=False)

    return launch


__all__ = [
    "MegaMoEHopperMxfp4SplitConfig",
    "MegaMoEHopperMxfp4SplitSession",
    "MegaMoEHopperMxfp4SplitSymmBuffer",
    "Mxfp4SplitError",
    "Mxfp4SplitLifecycleError",
    "Mxfp4SplitSessionPoisonedError",
    "Mxfp4SplitUnavailableError",
    "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
    "hopper_mxfp4_split_mega_launch_thunk",
    "hopper_mxfp4_split_mega_moe",
]
