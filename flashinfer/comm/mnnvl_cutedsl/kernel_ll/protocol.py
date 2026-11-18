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

"""Low-latency MNNVL protocol and its two operation paths."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float8E4M3FN, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import QUAD_BF16
from ..dynamic_fp8_epilogue import fake_output_scale, output_scale_arg
from ..runtime import (
    current_cu_stream,
    make_fake_dynamic_compact_tensor,
    to_cute,
    to_cute_dynamic,
)
from ..symmetric_buffer import SymmetricBuffer
from .device_kernels import (
    LAMPORT_GENERATIONS,
    _LamportResidualRMSNormDeviceKernel,
    _LamportTwoShotDeviceKernel,
    _QuadFinalizePublishDeviceKernel,
    _ScalarFinalizePublishDeviceKernel,
    _SharedOnlyPublishDeviceKernel,
)


@dataclass(frozen=True, slots=True)
class LLCollectiveTuning:
    cluster_size: int = 8
    rank_lanes: int = 1
    threads: int = 128
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class LLFinalizeTuning:
    elements_per_thread: int = 4
    threads: int = 128
    prefetch_group: int = 10
    load_shared_expert_before_pdl: bool = False
    collective: LLCollectiveTuning = LLCollectiveTuning()


@dataclass(frozen=True, slots=True)
class LLAllReduceTuning:
    publish_elements_per_thread: int = 8
    publish_threads: int = 128
    publish_release_before_store: bool = False
    collective: LLCollectiveTuning = LLCollectiveTuning()
    # Publish through each rank's unicast mapping instead of one multicast
    # store (requires publish_elements_per_thread=8). The first
    # publish_multicast_fragments 8-element fragments of every token keep the
    # multicast store, so the publish draws on the switch's multicast
    # replication and on the unicast paths at once. With
    # publish_bulk_peer_stores each CTA stages its unicast fragments in shared
    # memory and sends every rank one bulk copy, which moves fewer, larger
    # NVLink writes than one 16-byte store per thread; publish_multicast_fragments
    # must then be a multiple of publish_threads.
    peer_stores: bool = False
    publish_multicast_fragments: int = 0
    publish_bulk_peer_stores: bool = False
    # Replace the one-shot protocol with a one-launch Lamport reduce-scatter
    # and multicast all-gather (see _LamportTwoShotDeviceKernel). Uses the
    # collective's cluster_size and threads; the publish knobs do not apply.
    two_shot: bool = False


LL_FINALIZE_GB300_TP8_H8192_K10 = LLFinalizeTuning()
LL_FINALIZE_GB300_TP8_H8192_K10_M_GE_20 = LLFinalizeTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_FINALIZE_GB300_TP16_H8192_K10 = LLFinalizeTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_ALL_REDUCE_GB300_TP8_H8192_M_LE_4 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, threads=64)
)
LL_ALL_REDUCE_GB300_TP8_H8192_M_GE_5 = LLAllReduceTuning()
LL_ALL_REDUCE_GB300_TP16_H8192_M_LE_10 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, threads=64)
)
LL_ALL_REDUCE_GB300_TP16_H8192_M_11_TO_17 = LLAllReduceTuning()
LL_ALL_REDUCE_GB300_TP16_H8192_M_GE_18 = LLAllReduceTuning(
    collective=LLCollectiveTuning(cluster_size=16, rank_lanes=2)
)
LL_ALL_REDUCE_GB300_TP8_H8192 = LL_ALL_REDUCE_GB300_TP8_H8192_M_GE_5
LL_ALL_REDUCE_GB300_TP16_H8192 = LL_ALL_REDUCE_GB300_TP16_H8192_M_LE_10

# TP4, hidden 8192, measured separately on GB300 and Rubin. The multicast
# one-shot default is fastest for the fewest tokens. Once its publish is byte
# bound, the BULK variant keeps the multicast store for the first quarter of
# each token and sends the rest as per-CTA bulk copies to each rank. Without
# the norm, the TWO_SHOT variants move fewer bytes per GPU than the one-shot
# and amortize their second hop; fewer tokens favour more CTAs per token
# (a larger cluster) to spread the per-token work.
_LL_ALL_REDUCE_H8192_BULK = LLAllReduceTuning(
    publish_threads=256,
    peer_stores=True,
    publish_multicast_fragments=256,
    publish_bulk_peer_stores=True,
)


def _two_shot(cluster_size: int, threads: int) -> LLAllReduceTuning:
    return LLAllReduceTuning(
        two_shot=True,
        collective=LLCollectiveTuning(cluster_size=cluster_size, threads=threads),
    )


LL_ALL_REDUCE_GB300_TP4_H8192 = LLAllReduceTuning()
LL_ALL_REDUCE_GB300_TP4_H8192_BULK = _LL_ALL_REDUCE_H8192_BULK
LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_25_TO_64 = _two_shot(16, 64)
LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_65_TO_192 = _two_shot(8, 128)
LL_ALL_REDUCE_GB300_TP4_H8192_TWO_SHOT_M_193_TO_320 = _two_shot(8, 64)
LL_FINALIZE_GB300_TP4_H8192_K10 = LLFinalizeTuning()
LL_ALL_REDUCE_RUBIN_TP4_H8192 = LLAllReduceTuning()
LL_ALL_REDUCE_RUBIN_TP4_H8192_BULK = _LL_ALL_REDUCE_H8192_BULK
LL_ALL_REDUCE_RUBIN_TP4_H8192_TWO_SHOT = _two_shot(8, 64)
LL_FINALIZE_RUBIN_TP4_H8192_K10 = LLFinalizeTuning()

# hidden_size=5120, bf16. Tuning presets carry launch
# knobs only -- hidden_size/top_k reach the kernels through StaticProfile -- so
# these differ from the H8192 presets purely in knobs picked for a 640-vector
# token. cluster_size 5 makes fragment_stride (cluster_size * threads / rank
# lanes) exactly 640, so the Lamport RMSNorm covers one token in a single fully
# utilised trip; the H8192 cluster_size of 8 leaves 37% of the lanes idle here.
# prefetch_group is sized to top_k so the staged rmem tensor holds no unused
# slots (the gather loop is const_expr-predicated on k < top_k, so a larger
# group is correct but burns registers).
_LL_COLLECTIVE_H5120 = LLCollectiveTuning(cluster_size=5)

LL_FINALIZE_GB300_TP4_H5120_K6 = LLFinalizeTuning(
    prefetch_group=6, collective=_LL_COLLECTIVE_H5120
)
LL_FINALIZE_GB300_TP4_H5120_K3 = LLFinalizeTuning(
    prefetch_group=3, collective=_LL_COLLECTIVE_H5120
)
LL_FINALIZE_GB300_TP8_H5120_K6 = LLFinalizeTuning(
    prefetch_group=6, collective=_LL_COLLECTIVE_H5120
)
LL_FINALIZE_GB300_TP8_H5120_K3 = LLFinalizeTuning(
    prefetch_group=3, collective=_LL_COLLECTIVE_H5120
)
LL_ALL_REDUCE_GB300_TP4_H5120 = LLAllReduceTuning(collective=_LL_COLLECTIVE_H5120)
LL_ALL_REDUCE_GB300_TP8_H5120 = LLAllReduceTuning(collective=_LL_COLLECTIVE_H5120)

# hidden_size=7168, bf16, top_k=16 (Kimi K3), TP4 and TP8. A token is
# 896 bf16x8 fragments, so cluster_size=7 with 128 threads gives a fragment
# stride of exactly 896. The collective therefore covers one token without a
# partially utilised trip. TP8 finalize uses a separately measured decode
# preset: two rank lanes and 256 collective threads reduce the inter-node
# latency, while an 8-route prefetch group keeps its register footprint down.
_LL_COLLECTIVE_H7168 = LLCollectiveTuning(cluster_size=7)

LL_FINALIZE_GB300_TP4_H7168_K16 = LLFinalizeTuning(
    prefetch_group=16, collective=_LL_COLLECTIVE_H7168
)
LL_FINALIZE_GB300_TP8_H7168_K16 = LLFinalizeTuning(
    elements_per_thread=1,
    threads=256,
    prefetch_group=8,
    collective=LLCollectiveTuning(cluster_size=8, rank_lanes=2, threads=256),
)
LL_ALL_REDUCE_GB300_TP4_H7168 = LLAllReduceTuning(collective=_LL_COLLECTIVE_H7168)
LL_ALL_REDUCE_GB300_TP8_H7168 = LLAllReduceTuning(collective=_LL_COLLECTIVE_H7168)


@dataclass(slots=True)
class LLProtocolState:
    contribution_mailbox: SymmetricBuffer
    stage_state: torch.Tensor


def create_ll_state(
    group: dist.ProcessGroup,
    tp_size: int,
    rank: int,
    capacity_m: int,
    hidden: int,
    *,
    materialize_peer_addresses: bool = False,
) -> LLProtocolState:
    """Collectively allocate a sentinel-filled ``[3, tp, capacity_m, hidden]`` mailbox."""
    if dist.get_world_size(group) != tp_size:
        raise ValueError("ProcessGroup size does not match tp_size")
    if dist.get_rank(group) != rank:
        raise ValueError("ProcessGroup rank does not match rank")
    device = torch.device("cuda", torch.cuda.current_device())
    mailbox = SymmetricBuffer.allocate(
        (LAMPORT_GENERATIONS, tp_size, capacity_m, hidden),
        torch.bfloat16,
        device,
        group,
        require_multicast=True,
        materialize_peer_addresses=materialize_peer_addresses,
    )
    mailbox.tensor.view(torch.int16).fill_(-32768)
    return LLProtocolState(
        contribution_mailbox=mailbox,
        stage_state=torch.zeros((2,), dtype=torch.int32, device=device),
    )


def compile_shared_only_publish(
    *,
    hidden: int,
    tp: int,
    rank: int,
    capacity_m: int,
    elements_per_thread: int,
    threads: int,
    release_before_store: bool,
    enable_pdl: bool,
    peer_stores: bool = False,
    multicast_fragments: int = 0,
    bulk_peer_stores: bool = False,
):
    """Compile the publisher that multicasts a local ``[m, hidden]`` BF16 tensor.

    The compiled function takes ``(local, stage_state, multicast_address,
    peer_mailboxes, m, stream)``; ``peer_mailboxes`` is the ``[tp]`` int64
    table of unicast mailbox addresses with ``peer_stores`` and ``None``
    otherwise.
    """
    device_kernel = _SharedOnlyPublishDeviceKernel(
        hidden=hidden,
        tp=tp,
        rank=rank,
        capacity_m=capacity_m,
        elements_per_thread=elements_per_thread,
        threads=threads,
        release_before_store=release_before_store,
        enable_pdl=enable_pdl,
        peer_stores=peer_stores,
        multicast_fragments=multicast_fragments,
        bulk_peer_stores=bulk_peer_stores,
    )
    return cute.compile(
        device_kernel,
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=hidden),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        make_fake_compact_tensor(Int64, (tp,), assumed_align=8)
        if peer_stores
        else None,
        Int32(capacity_m),
        current_cu_stream(),
    )


@dataclass(slots=True)
class _TwoShotState:
    scatter_mailbox: SymmetricBuffer
    gather_mailbox: SymmetricBuffer
    generations: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompiledFinalize:
    publish: Any
    collective: Any


@dataclass(frozen=True, slots=True)
class _CompiledAllReduce:
    publish: Any
    collective: Any


class _PathKwargs(TypedDict):
    hidden_size: int
    top_k: int
    capacity_m: int
    write_residual_output: bool
    output_dtype: torch.dtype


class _LLPath:
    def __init__(
        self,
        *,
        hidden_size: int,
        top_k: int,
        capacity_m: int,
        write_residual_output: bool,
        output_dtype: torch.dtype,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.capacity_m = capacity_m
        self.write_residual_output = write_residual_output
        self.output_dtype = output_dtype

    def _outputs(
        self,
        m: int,
        norm_output: torch.Tensor | None,
        residual_output: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        shape = (m, self.hidden_size)
        device = torch.device("cuda", torch.cuda.current_device())
        if norm_output is None:
            norm_output = torch.empty(shape, dtype=self.output_dtype, device=device)
        if self.write_residual_output and residual_output is None:
            residual_output = torch.empty(shape, dtype=torch.bfloat16, device=device)
        return norm_output, residual_output

    def _validate_state(self, state: LLProtocolState, m: int) -> None:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")
        address = state.contribution_mailbox.multicast_address
        if address is None or address % 16:
            raise ValueError(
                "LL contribution mailbox requires a 16-byte-aligned multicast address"
            )

    def _launch_collective(
        self,
        collective,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        state: LLProtocolState,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
        norm_output_bf16: torch.Tensor | None,
        output_scale: torch.Tensor | None,
    ) -> None:
        residual_arg = residual_source if residual_source is not None else gamma
        residual_output_arg = residual_output if residual_output is not None else gamma
        collective(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(
                norm_output_bf16.flatten(), 16, divisibility=self.hidden_size
            )
            if norm_output_bf16 is not None
            else None,
            output_scale_arg(output_scale),
            to_cute(state.stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )


class FinalizeAllReduceRMSNormLLKernel(_LLPath):
    def __init__(self, *, compiled: _CompiledFinalize, **kwargs) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled

    def __call__(
        self,
        routed_output: torch.Tensor,
        expert_weights: torch.Tensor,
        permuted_indices: torch.Tensor,
        shared_output: torch.Tensor | None,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: LLProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        norm_output_bf16: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        shared_arg = shared_output if shared_output is not None else gamma
        self._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=self.top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=self.top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.contribution_mailbox.multicast_address)),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_collective(
            self._compiled.collective,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
            norm_output_bf16,
            output_scale,
        )
        return norm_output, residual_output


class AllReduceRMSNormLLKernel(_LLPath):
    def __init__(
        self, *, compiled: _CompiledAllReduce, peer_stores: bool = False, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled
        self._peer_stores = peer_stores

    def __call__(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: LLProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        norm_output_bf16: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        self._compiled.publish(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.contribution_mailbox.multicast_address)),
            to_cute(cast(torch.Tensor, state.contribution_mailbox.peer_addresses), 8)
            if self._peer_stores
            else None,
            Int32(m),
            current_cu_stream(),
        )
        self._launch_collective(
            self._compiled.collective,
            residual_source,
            gamma,
            state,
            norm_output,
            residual_output,
            m,
            norm_output_bf16,
            output_scale,
        )
        return norm_output, residual_output


class TwoShotAllReduceRMSNormLLKernel(_LLPath):
    def __init__(
        self, *, compiled: Any, two_shot_state: _TwoShotState, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled
        self._state = two_shot_state

    def __call__(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: LLProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        norm_output_bf16: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        scatter = self._state.scatter_mailbox
        gather = self._state.gather_mailbox
        residual_arg = residual_source if residual_source is not None else gamma
        residual_output_arg = residual_output if residual_output is not None else gamma
        self._compiled(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute(scatter.tensor.flatten(), 16),
            to_cute(cast(torch.Tensor, scatter.peer_addresses), 8),
            to_cute(gather.tensor.flatten(), 16),
            Int64(cast(int, gather.multicast_address)),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(gamma, 16),
            to_cute_dynamic(
                residual_output_arg.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(
                norm_output_bf16.flatten(), 16, divisibility=self.hidden_size
            )
            if norm_output_bf16 is not None
            else None,
            output_scale_arg(output_scale),
            to_cute(self._state.generations, 4),
            Int32(m),
            current_cu_stream(),
        )
        return norm_output, residual_output


class LLProtocol:
    """Own LL State and protocol-local compiled variants for both paths."""

    def __init__(
        self,
        hidden_size: int,
        top_k: int,
        tp_size: int,
        rank: int,
        capacity_m: int,
        rms_epsilon: float,
        routed_scaling_factor: float,
        weight_bias: float,
        *,
        include_shared_expert: bool,
        add_residual: bool,
        write_residual_output: bool,
        apply_rms_norm: bool = True,
        finalize_tunings: tuple[LLFinalizeTuning, ...],
        all_reduce_tunings: tuple[LLAllReduceTuning, ...],
        group: dist.ProcessGroup,
        output_dtype: torch.dtype = torch.bfloat16,
        write_norm_output: bool = False,
        dynamic_fp8: bool = False,
        all_reduce_capacities: Mapping[LLAllReduceTuning, int] | None = None,
    ) -> None:
        """``all_reduce_capacities`` maps an all-reduce tuning to the largest
        ``m`` it serves (at most ``capacity_m``, the default); a two-shot
        tuning sizes its own mailboxes to it."""
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.rms_epsilon = rms_epsilon
        self.routed_scaling_factor = routed_scaling_factor
        self.weight_bias = weight_bias
        self.include_shared_expert = include_shared_expert
        self.add_residual = add_residual
        self.write_residual_output = write_residual_output
        self.output_dtype = output_dtype
        self.write_norm_output = write_norm_output
        self.dynamic_fp8 = dynamic_fp8
        self.apply_rms_norm = apply_rms_norm

        for tuning in all_reduce_tunings:
            if tuning.peer_stores and tuning.publish_elements_per_thread != 8:
                raise ValueError("peer_stores requires publish_elements_per_thread=8")
            if tuning.publish_bulk_peer_stores and (
                not tuning.peer_stores
                or tuning.publish_multicast_fragments % tuning.publish_threads
            ):
                raise ValueError(
                    "publish_bulk_peer_stores requires peer_stores and a "
                    "publish_multicast_fragments multiple of publish_threads"
                )
            if tuning.two_shot and tuning.collective.rank_lanes != 1:
                raise ValueError("two_shot requires rank_lanes == 1")
            if tuning.two_shot and self.dynamic_fp8:
                raise ValueError("two_shot does not support dynamic FP8 output")
        one_shot_tunings = tuple(
            item for item in all_reduce_tunings if not item.two_shot
        )
        collective_cache = {
            tuning: self._compile_collective(tuning)
            for tuning in {
                *(item.collective for item in finalize_tunings),
                *(item.collective for item in one_shot_tunings),
            }
        }
        self.finalize_kernels = {
            tuning: FinalizeAllReduceRMSNormLLKernel(
                compiled=_CompiledFinalize(
                    publish=self._compile_finalize(tuning),
                    collective=collective_cache[tuning.collective],
                ),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(finalize_tunings)
        }
        self.all_reduce_kernels: dict[
            LLAllReduceTuning,
            AllReduceRMSNormLLKernel | TwoShotAllReduceRMSNormLLKernel,
        ] = {
            tuning: AllReduceRMSNormLLKernel(
                compiled=_CompiledAllReduce(
                    publish=self._compile_all_reduce_publish(tuning),
                    collective=collective_cache[tuning.collective],
                ),
                peer_stores=tuning.peer_stores,
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(one_shot_tunings)
        }
        self.state = create_ll_state(
            group,
            tp_size,
            rank,
            capacity_m,
            hidden_size,
            materialize_peer_addresses=any(
                item.peer_stores for item in one_shot_tunings
            ),
        )
        # A two-shot variant derives its mailbox slots and per-CTA Lamport
        # generations from its own CTA-to-fragment mapping, so each collective
        # shape owns its mailboxes; variants never share a slot. The
        # mailboxes only cover the token counts routed to that shape.
        capacities = all_reduce_capacities or {}
        two_shot_capacities: dict[LLCollectiveTuning, int] = {}
        for tuning in all_reduce_tunings:
            if tuning.two_shot:
                route_capacity = min(capacities.get(tuning, capacity_m), capacity_m)
                if route_capacity <= 0:
                    raise ValueError("all-reduce capacities must be positive")
                two_shot_capacities[tuning.collective] = max(
                    two_shot_capacities.get(tuning.collective, 0), route_capacity
                )
        two_shot_cache = {
            collective: (
                self._compile_two_shot(collective, two_shot_capacity),
                self._create_two_shot_state(group, collective, two_shot_capacity),
            )
            for collective, two_shot_capacity in two_shot_capacities.items()
        }
        for tuning in dict.fromkeys(all_reduce_tunings):
            if not tuning.two_shot:
                continue
            compiled, two_shot_state = two_shot_cache[tuning.collective]
            self.all_reduce_kernels[tuning] = TwoShotAllReduceRMSNormLLKernel(
                compiled=compiled,
                two_shot_state=two_shot_state,
                **{
                    **self._path_kwargs(),
                    "capacity_m": two_shot_capacities[tuning.collective],
                },
            )

    def _path_kwargs(self) -> _PathKwargs:
        return {
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "capacity_m": self.capacity_m,
            "write_residual_output": self.write_residual_output,
            "output_dtype": self.output_dtype,
        }

    def _compile_finalize(self, tuning: LLFinalizeTuning):
        if tuning.elements_per_thread not in (1, QUAD_BF16):
            raise ValueError("LL finalize elements_per_thread must be 1 or 4")
        kwargs: dict[str, Any] = {
            "hidden": self.hidden_size,
            "top_k": self.top_k,
            "tp": self.tp_size,
            "rank": self.rank,
            "capacity_m": self.capacity_m,
            "threads": tuning.threads,
            "routed_scaling_factor": self.routed_scaling_factor,
            "include_shared_expert": self.include_shared_expert,
            "load_shared_expert_before_pdl": tuning.load_shared_expert_before_pdl,
            "enable_pdl": tuning.collective.enable_pdl,
            "prefetch_group": tuning.prefetch_group,
        }
        device_kernel = (
            _ScalarFinalizePublishDeviceKernel(**kwargs)
            if tuning.elements_per_thread == 1
            else _QuadFinalizePublishDeviceKernel(**kwargs)
        )
        args = (
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=2, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _compile_all_reduce_publish(self, tuning: LLAllReduceTuning):
        return compile_shared_only_publish(
            hidden=self.hidden_size,
            tp=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            elements_per_thread=tuning.publish_elements_per_thread,
            threads=tuning.publish_threads,
            release_before_store=tuning.publish_release_before_store,
            enable_pdl=tuning.collective.enable_pdl,
            peer_stores=tuning.peer_stores,
            multicast_fragments=tuning.publish_multicast_fragments,
            bulk_peer_stores=tuning.publish_bulk_peer_stores,
        )

    def _compile_collective(self, tuning: LLCollectiveTuning):
        device_kernel = _LamportResidualRMSNormDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            capacity_m=self.capacity_m,
            cluster_size=tuning.cluster_size,
            rank_lanes=tuning.rank_lanes,
            threads=tuning.threads,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
            apply_rms_norm=self.apply_rms_norm,
            dynamic_fp8=self.dynamic_fp8,
        )
        activation = self.capacity_m * self.hidden_size
        args = (
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * self.tp_size * activation,),
                assumed_align=16,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                Float8E4M3FN if self.output_dtype == torch.float8_e4m3fn else BFloat16,
                alignment=16,
                divisibility=self.hidden_size,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            )
            if self.write_norm_output
            else None,
            fake_output_scale(self.output_dtype),
            make_fake_compact_tensor(Int32, (2,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _compile_two_shot(self, collective: LLCollectiveTuning, capacity_m: int):
        device_kernel = _LamportTwoShotDeviceKernel(
            hidden=self.hidden_size,
            tp=self.tp_size,
            rank=self.rank,
            capacity_m=capacity_m,
            cluster_size=collective.cluster_size,
            threads=collective.threads,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            apply_rms_norm=self.apply_rms_norm,
            enable_pdl=collective.enable_pdl,
        )
        activation = capacity_m * self.hidden_size
        args = (
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * self.tp_size * activation,),
                assumed_align=16,
            ),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * activation,),
                assumed_align=16,
            ),
            Int64(0),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_dynamic_compact_tensor(
                Float8E4M3FN if self.output_dtype == torch.float8_e4m3fn else BFloat16,
                alignment=16,
                divisibility=self.hidden_size,
            ),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            )
            if self.write_norm_output
            else None,
            fake_output_scale(self.output_dtype),
            make_fake_compact_tensor(
                Int32,
                (capacity_m * collective.cluster_size,),
                assumed_align=4,
            ),
            Int32(capacity_m),
            current_cu_stream(),
        )
        return cute.compile(device_kernel, *args)

    def _create_two_shot_state(
        self,
        group: dist.ProcessGroup,
        collective: LLCollectiveTuning,
        capacity_m: int,
    ) -> _TwoShotState:
        device = torch.device("cuda", torch.cuda.current_device())
        scatter = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.tp_size,
                capacity_m,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            materialize_peer_addresses=True,
        )
        gather = SymmetricBuffer.allocate(
            (LAMPORT_GENERATIONS, capacity_m, self.hidden_size),
            torch.bfloat16,
            device,
            group,
            require_multicast=True,
        )
        scatter.tensor.view(torch.int16).fill_(-32768)
        gather.tensor.view(torch.int16).fill_(-32768)
        return _TwoShotState(
            scatter_mailbox=scatter,
            gather_mailbox=gather,
            generations=torch.zeros(
                (capacity_m * collective.cluster_size,),
                dtype=torch.int32,
                device=device,
            ),
        )
