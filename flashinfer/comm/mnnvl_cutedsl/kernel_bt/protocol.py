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

"""Balanced MNNVL protocol and its two operation paths."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypedDict, cast

import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float8E4M3FN, Int32, Int64
from cutlass.cute.runtime import make_fake_compact_tensor

from ..cute_dsl_primitives import VEC_BF16
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
    STAGE_STATE_WORDS,
    _FusedAllReduceDeviceKernel,
    _MaterializeRMSNormDeviceKernel,
    _NarrowVectorFinalizeUnicastDeviceKernel,
    _OwnerReduceMulticastDeviceKernel,
    _ScalarFinalizeUnicastDeviceKernel,
    _SharedOnlyPublishDeviceKernel,
    _VectorFinalizeUnicastDeviceKernel,
)


@dataclass(frozen=True, slots=True)
class BTCollectiveTuning:
    reduction_threads: int = 128
    rms_threads: int = 1024
    enable_pdl: bool = True


@dataclass(frozen=True, slots=True)
class BTFinalizeTuning:
    elements_per_thread: int = VEC_BF16
    threads: int = 128
    prefetch_group: int = 1
    load_shared_expert_before_pdl: bool = False
    collective: BTCollectiveTuning = BTCollectiveTuning()


# Threads per CTA of the fused all-reduce; a 1024-thread CTA covers a
# hidden-8192 token in one 16-byte vector per thread.
_FUSED_THREADS = 1024


def _fused_grid_ctas(device: torch.device) -> int:
    """Grid size of the fused all-reduce: one CTA per SM.

    The schedule only needs one resident CTA per rank to make progress; one
    per SM lets every SM move data.
    """
    return torch.cuda.get_device_properties(device).multi_processor_count


@dataclass(frozen=True, slots=True)
class BTAllReduceTuning:
    publish_threads: int = 128
    publish_vectors_per_thread: int = 1
    collective: BTCollectiveTuning = BTCollectiveTuning(reduction_threads=32)
    # fused=True runs publish, owner reduce and materialize as one persistent
    # launch of one 1024-thread CTA per SM (see _FusedAllReduceDeviceKernel)
    # that needs at least one of its CTAs resident on every rank;
    # publish_* and the collective thread counts are then unused, while
    # collective.enable_pdl still applies. fused_lag is the distance in rounds
    # between consecutive stages of one token, so most data has landed before
    # a stage waits on it.
    fused: bool = False
    fused_lag: int = 3


BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256
)
BT_FINALIZE_GB300_TP8_H8192_K10_PRESET_1 = BTFinalizeTuning()
BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256
)
BT_FINALIZE_GB300_TP16_H8192_K10_PRESET_1 = BTFinalizeTuning()
BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP8_H8192_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)
BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP16_H8192_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)

# hidden_size=5120, bf16. A 5120-wide token is 640 bf16x8
# vectors, so the PRESET_1 shape (elements_per_thread=8, threads=128) tiles it
# in exactly 5 CTAs with no predicated tail, while PRESET_0 keeps the narrower
# 2-element/256-thread tiling for the small-M unicast regime. prefetch_group
# tracks top_k; every other knob matches the H8192 presets because the BT
# kernels size their grids with ceil() and predicate the tail, so they are
# already shape-tolerant.
BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256, prefetch_group=6
)
BT_FINALIZE_GB300_TP4_H5120_K6_PRESET_1 = BTFinalizeTuning(prefetch_group=6)
BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256, prefetch_group=3
)
BT_FINALIZE_GB300_TP4_H5120_K3_PRESET_1 = BTFinalizeTuning(prefetch_group=3)
BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256, prefetch_group=6
)
BT_FINALIZE_GB300_TP8_H5120_K6_PRESET_1 = BTFinalizeTuning(prefetch_group=6)
BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=2, threads=256, prefetch_group=3
)
BT_FINALIZE_GB300_TP8_H5120_K3_PRESET_1 = BTFinalizeTuning(prefetch_group=3)
BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP4_H5120_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)
BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP8_H5120_PRESET_1 = BTAllReduceTuning(
    collective=BTCollectiveTuning(reduction_threads=320)
)

# hidden_size=7168, bf16, top_k=16 (Kimi K3). These use only the existing BT
# launch knobs. PRESET_0 covers the latency-sensitive end of the BT range;
# PRESET_1 increases publication/reduction parallelism for larger M.
BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=8,
    threads=128,
    prefetch_group=16,
    collective=BTCollectiveTuning(reduction_threads=160, rms_threads=896),
)
BT_FINALIZE_GB300_TP4_H7168_K16_PRESET_1 = BTFinalizeTuning(
    elements_per_thread=4,
    threads=128,
    prefetch_group=4,
    collective=BTCollectiveTuning(reduction_threads=448, rms_threads=512),
)
BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_0 = BTFinalizeTuning(
    elements_per_thread=4,
    threads=128,
    prefetch_group=16,
    collective=BTCollectiveTuning(reduction_threads=192, rms_threads=896),
)
BT_FINALIZE_GB300_TP8_H7168_K16_PRESET_1 = BTFinalizeTuning(
    elements_per_thread=4,
    threads=64,
    prefetch_group=16,
    collective=BTCollectiveTuning(reduction_threads=256, rms_threads=512),
)
BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_0 = BTAllReduceTuning(
    publish_threads=512,
    publish_vectors_per_thread=4,
    collective=BTCollectiveTuning(reduction_threads=32, rms_threads=1024),
)
BT_ALL_REDUCE_GB300_TP4_H7168_PRESET_1 = BTAllReduceTuning(
    publish_threads=768,
    publish_vectors_per_thread=8,
    collective=BTCollectiveTuning(reduction_threads=320, rms_threads=512),
)
BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_0 = BTAllReduceTuning(
    publish_threads=64,
    publish_vectors_per_thread=8,
    collective=BTCollectiveTuning(reduction_threads=128, rms_threads=1024),
)
BT_ALL_REDUCE_GB300_TP8_H7168_PRESET_1 = BTAllReduceTuning(
    publish_threads=768,
    publish_vectors_per_thread=8,
    collective=BTCollectiveTuning(reduction_threads=192, rms_threads=448),
)

# hidden_size=8192, bf16, TP4, measured separately on GB300 and Rubin.
# PRESET_0 keeps the three-launch path, which has the lower fixed latency up to
# a few hundred tokens; FUSED is the one-launch path, ahead from there up.
BT_ALL_REDUCE_GB300_TP4_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_GB300_TP4_H8192_FUSED = BTAllReduceTuning(fused=True)
BT_FINALIZE_GB300_TP4_H8192_K10 = BTFinalizeTuning()
BT_ALL_REDUCE_RUBIN_TP4_H8192_PRESET_0 = BTAllReduceTuning()
BT_ALL_REDUCE_RUBIN_TP4_H8192_FUSED = BTAllReduceTuning(fused=True)
BT_FINALIZE_RUBIN_TP4_H8192_K10 = BTFinalizeTuning()


@dataclass(slots=True)
class BTProtocolState:
    contribution_mailbox: SymmetricBuffer
    prenorm_mailbox: SymmetricBuffer
    stage_state: torch.Tensor


@dataclass(frozen=True, slots=True)
class _CompiledTail:
    reduce: Any
    rms_norm: Any


@dataclass(frozen=True, slots=True)
class _CompiledFinalize:
    publish: Any
    tail: _CompiledTail


@dataclass(frozen=True, slots=True)
class _CompiledAllReduce:
    publish: Any
    tail: _CompiledTail | None


class _PathKwargs(TypedDict):
    hidden_size: int
    top_k: int
    capacity_m: int
    write_residual_output: bool
    output_dtype: torch.dtype


class _BTPath:
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

    def _validate_state(self, state: BTProtocolState, m: int) -> None:
        if not 1 <= m <= self.capacity_m:
            raise ValueError(f"m must be in [1, {self.capacity_m}]")
        if (
            state.contribution_mailbox.peer_addresses is None
            or state.prenorm_mailbox.peer_addresses is None
        ):
            raise ValueError("BT contribution mailbox requires peer addresses")
        address = state.prenorm_mailbox.multicast_address
        if address is None or address % 16:
            raise ValueError(
                "BT prenorm mailbox requires a 16-byte-aligned multicast address"
            )

    def _launch_tail(
        self,
        tail: _CompiledTail,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        state: BTProtocolState,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
        norm_output_bf16: torch.Tensor | None,
        output_scale: torch.Tensor | None,
    ) -> None:
        residual_arg = residual_source if residual_source is not None else gamma
        residual_output_arg = residual_output if residual_output is not None else gamma
        stream = current_cu_stream()
        tail.reduce(
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            Int64(cast(int, state.prenorm_mailbox.multicast_address)),
            Int32(m),
            stream,
        )
        tail.rms_norm(
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
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
            to_cute(gamma, 16),
            to_cute(state.stage_state, 4),
            Int32(m),
            stream,
        )


class FinalizeAllReduceRMSNormBTKernel(_BTPath):
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
        state: BTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        norm_output_bf16: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        shared_arg = shared_output if shared_output is not None else gamma
        peers = cast(torch.Tensor, state.contribution_mailbox.peer_addresses)
        self._compiled.publish(
            to_cute_dynamic(routed_output.flatten(), 16, divisibility=self.hidden_size),
            to_cute_dynamic(expert_weights.flatten(), 2, divisibility=self.top_k),
            to_cute_dynamic(permuted_indices.flatten(), 4, divisibility=self.top_k),
            to_cute_dynamic(shared_arg.flatten(), 16, divisibility=self.hidden_size),
            to_cute(state.stage_state, 4),
            to_cute(peers, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_tail(
            self._compiled.tail,
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


class AllReduceRMSNormBTKernel(_BTPath):
    def __init__(self, *, compiled: _CompiledAllReduce, **kwargs) -> None:
        super().__init__(**kwargs)
        self._compiled = compiled

    def __call__(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        m: int,
        *,
        state: BTProtocolState,
        norm_output: torch.Tensor | None = None,
        residual_output: torch.Tensor | None = None,
        norm_output_bf16: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        self._validate_state(state, m)
        norm_output, residual_output = self._outputs(m, norm_output, residual_output)
        peers = cast(torch.Tensor, state.contribution_mailbox.peer_addresses)
        if self._compiled.tail is None:
            self._launch_fused(
                local_contribution,
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
        self._compiled.publish(
            to_cute_dynamic(
                local_contribution.flatten(),
                16,
                divisibility=self.hidden_size,
            ),
            to_cute(state.stage_state, 4),
            to_cute(peers, 8),
            Int32(m),
            current_cu_stream(),
        )
        self._launch_tail(
            self._compiled.tail,
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

    def _launch_fused(
        self,
        local_contribution: torch.Tensor,
        residual_source: torch.Tensor | None,
        gamma: torch.Tensor,
        state: BTProtocolState,
        norm_output: torch.Tensor,
        residual_output: torch.Tensor | None,
        m: int,
        norm_output_bf16: torch.Tensor | None,
        output_scale: torch.Tensor | None,
    ) -> None:
        residual_arg = residual_source if residual_source is not None else gamma
        residual_output_arg = residual_output if residual_output is not None else gamma
        h = self.hidden_size
        self._compiled.publish(
            to_cute_dynamic(local_contribution.flatten(), 16, divisibility=h),
            to_cute_dynamic(residual_arg.flatten(), 16, divisibility=h),
            to_cute(gamma, 16),
            to_cute_dynamic(residual_output_arg.flatten(), 16, divisibility=h),
            to_cute_dynamic(norm_output.flatten(), 16, divisibility=h),
            to_cute_dynamic(norm_output_bf16.flatten(), 16, divisibility=h)
            if norm_output_bf16 is not None
            else None,
            output_scale_arg(output_scale),
            to_cute(state.contribution_mailbox.tensor.flatten(), 16),
            to_cute(cast(torch.Tensor, state.contribution_mailbox.peer_addresses), 8),
            to_cute(state.prenorm_mailbox.tensor.flatten(), 16),
            to_cute(cast(torch.Tensor, state.prenorm_mailbox.peer_addresses), 8),
            to_cute(state.stage_state, 4),
            Int32(m),
            current_cu_stream(),
        )


class BTProtocol:
    """Own BT State and protocol-local compiled variants for both paths."""

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
        finalize_tunings: tuple[BTFinalizeTuning, ...],
        all_reduce_tunings: tuple[BTAllReduceTuning, ...],
        group: dist.ProcessGroup,
        output_dtype: torch.dtype = torch.bfloat16,
        write_norm_output: bool = False,
        dynamic_fp8: bool = False,
    ) -> None:
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.tp_size = tp_size
        self.rank = rank
        self.capacity_m = capacity_m
        self.local_capacity = math.ceil(capacity_m / tp_size)
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

        tail_cache = {
            tuning: self._compile_tail(tuning)
            for tuning in {
                *(item.collective for item in finalize_tunings),
                *(item.collective for item in all_reduce_tunings if not item.fused),
            }
        }
        self.finalize_kernels = {
            tuning: FinalizeAllReduceRMSNormBTKernel(
                compiled=_CompiledFinalize(
                    publish=self._compile_finalize(tuning),
                    tail=tail_cache[tuning.collective],
                ),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(finalize_tunings)
        }
        self.all_reduce_kernels = {
            tuning: AllReduceRMSNormBTKernel(
                compiled=_CompiledAllReduce(
                    publish=self._compile_fused_all_reduce(tuning),
                    tail=None,
                )
                if tuning.fused
                else _CompiledAllReduce(
                    publish=self._compile_all_reduce_publish(tuning),
                    tail=tail_cache[tuning.collective],
                ),
                **self._path_kwargs(),
            )
            for tuning in dict.fromkeys(all_reduce_tunings)
        }
        self.state = self._create_state(group)

    def _path_kwargs(self) -> _PathKwargs:
        return {
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "capacity_m": self.capacity_m,
            "write_residual_output": self.write_residual_output,
            "output_dtype": self.output_dtype,
        }

    def _compile_finalize(self, tuning: BTFinalizeTuning):
        if tuning.elements_per_thread not in (1, 2, 4, VEC_BF16):
            raise ValueError("BT finalize elements_per_thread must be 1, 2, 4, or 8")
        kwargs: dict[str, Any] = {
            "hidden_size": self.hidden_size,
            "top_k": self.top_k,
            "tp_size": self.tp_size,
            "rank": self.rank,
            "local_capacity": self.local_capacity,
            "threads": tuning.threads,
            "routed_scaling_factor": self.routed_scaling_factor,
            "include_shared_expert": self.include_shared_expert,
            "load_shared_expert_before_pdl": tuning.load_shared_expert_before_pdl,
            "enable_pdl": tuning.collective.enable_pdl,
            "prefetch_group": tuning.prefetch_group,
        }
        device_kernel: Any
        if tuning.elements_per_thread == 1:
            device_kernel = _ScalarFinalizeUnicastDeviceKernel(**kwargs)
        elif tuning.elements_per_thread == VEC_BF16:
            device_kernel = _VectorFinalizeUnicastDeviceKernel(**kwargs)
        else:
            device_kernel = _NarrowVectorFinalizeUnicastDeviceKernel(
                **kwargs, elements_per_thread=tuning.elements_per_thread
            )
        return cute.compile(
            device_kernel,
            *self._publish_compile_args(include_routed=True),
        )

    def _compile_all_reduce_publish(self, tuning: BTAllReduceTuning):
        device_kernel = _SharedOnlyPublishDeviceKernel(
            hidden_size=self.hidden_size,
            tp_size=self.tp_size,
            rank=self.rank,
            local_capacity=self.local_capacity,
            threads=tuning.publish_threads,
            vectors_per_thread=tuning.publish_vectors_per_thread,
            enable_pdl=tuning.collective.enable_pdl,
        )
        return cute.compile(
            device_kernel,
            *self._publish_compile_args(include_routed=False),
        )

    def _compile_fused_all_reduce(self, tuning: BTAllReduceTuning):
        if tuning.fused_lag < 1:
            raise ValueError("fused_lag must be positive")
        device = torch.device("cuda", torch.cuda.current_device())
        device_kernel = _FusedAllReduceDeviceKernel(
            hidden_size=self.hidden_size,
            tp_size=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            local_capacity=self.local_capacity,
            threads=_FUSED_THREADS,
            grid_ctas=_fused_grid_ctas(device),
            lag=tuning.fused_lag,
            add_residual=self.add_residual,
            write_residual_output=self.write_residual_output,
            apply_rms_norm=self.apply_rms_norm,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            enable_pdl=tuning.collective.enable_pdl,
            dynamic_fp8=self.dynamic_fp8,
        )
        # The kernel spins on peers, so one CTA of it must fit on an SM. The
        # block size is static, so the compiler limits the registers to what a
        # 1024-thread CTA can get (reqntid); shared memory is the remaining
        # per-CTA limit.
        shared_memory_limit = torch.cuda.get_device_properties(
            device
        ).shared_memory_per_block_optin
        if device_kernel.smem_bytes > shared_memory_limit:
            raise ValueError(
                f"the fused BT all-reduce needs {device_kernel.smem_bytes} bytes "
                f"of shared memory per CTA for hidden_size={self.hidden_size}, "
                f"but the device allows {shared_memory_limit}; use a "
                "three-launch BT preset (fused=False)"
            )
        activation = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=self.hidden_size
        )
        return cute.compile(
            device_kernel,
            activation,
            activation,
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            activation,
            make_fake_dynamic_compact_tensor(
                Float8E4M3FN if self.output_dtype == torch.float8_e4m3fn else BFloat16,
                alignment=16,
                divisibility=self.hidden_size,
            ),
            activation if self.write_norm_output else None,
            fake_output_scale(self.output_dtype),
            make_fake_compact_tensor(
                BFloat16,
                (
                    LAMPORT_GENERATIONS
                    * self.tp_size
                    * self.local_capacity
                    * self.hidden_size,
                ),
                assumed_align=16,
            ),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            make_fake_compact_tensor(
                BFloat16,
                (LAMPORT_GENERATIONS * self.capacity_m * self.hidden_size,),
                assumed_align=16,
            ),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            make_fake_compact_tensor(Int32, (STAGE_STATE_WORDS,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )

    def _publish_compile_args(self, *, include_routed: bool) -> tuple:
        activation = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=self.hidden_size
        )
        common = (
            make_fake_compact_tensor(Int32, (STAGE_STATE_WORDS,), assumed_align=4),
            make_fake_compact_tensor(Int64, (self.tp_size,), assumed_align=8),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        if not include_routed:
            return (activation, *common)
        return (
            activation,
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=2, divisibility=self.top_k
            ),
            make_fake_dynamic_compact_tensor(
                Int32, alignment=4, divisibility=self.top_k
            ),
            activation,
            *common,
        )

    def _compile_tail(self, tuning: BTCollectiveTuning) -> _CompiledTail:
        reduce_kernel = _OwnerReduceMulticastDeviceKernel(
            hidden_size=self.hidden_size,
            tp_size=self.tp_size,
            rank=self.rank,
            capacity_m=self.capacity_m,
            local_capacity=self.local_capacity,
            threads=tuning.reduction_threads,
            add_residual=self.add_residual,
            enable_pdl=tuning.enable_pdl,
        )
        reduce_elements = (
            LAMPORT_GENERATIONS * self.tp_size * self.local_capacity * self.hidden_size
        )
        reduce = cute.compile(
            reduce_kernel,
            make_fake_compact_tensor(BFloat16, (reduce_elements,), assumed_align=16),
            make_fake_dynamic_compact_tensor(
                BFloat16, alignment=16, divisibility=self.hidden_size
            ),
            make_fake_compact_tensor(Int32, (STAGE_STATE_WORDS,), assumed_align=4),
            Int64(0),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        rms_kernel = _MaterializeRMSNormDeviceKernel(
            hidden_size=self.hidden_size,
            capacity_m=self.capacity_m,
            threads=tuning.rms_threads,
            rms_epsilon=self.rms_epsilon,
            weight_bias=self.weight_bias,
            write_residual_output=self.write_residual_output,
            enable_pdl=tuning.enable_pdl,
            apply_rms_norm=self.apply_rms_norm,
            dynamic_fp8=self.dynamic_fp8,
        )
        prenorm_elements = LAMPORT_GENERATIONS * self.capacity_m * self.hidden_size
        rms_norm = cute.compile(
            rms_kernel,
            make_fake_compact_tensor(BFloat16, (prenorm_elements,), assumed_align=16),
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
            make_fake_compact_tensor(BFloat16, (self.hidden_size,), assumed_align=16),
            make_fake_compact_tensor(Int32, (STAGE_STATE_WORDS,), assumed_align=4),
            Int32(self.capacity_m),
            current_cu_stream(),
        )
        return _CompiledTail(reduce=reduce, rms_norm=rms_norm)

    def _create_state(self, group: dist.ProcessGroup) -> BTProtocolState:
        if dist.get_world_size(group) != self.tp_size:
            raise ValueError("ProcessGroup size does not match tp_size")
        if dist.get_rank(group) != self.rank:
            raise ValueError("ProcessGroup rank does not match rank")
        device = torch.device("cuda", torch.cuda.current_device())
        contribution = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.tp_size,
                self.local_capacity,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            materialize_peer_addresses=True,
        )
        contribution.tensor.view(torch.int16).fill_(-32768)
        prenorm = SymmetricBuffer.allocate(
            (
                LAMPORT_GENERATIONS,
                self.capacity_m,
                self.hidden_size,
            ),
            torch.bfloat16,
            device,
            group,
            require_multicast=True,
            materialize_peer_addresses=True,
        )
        prenorm.tensor.view(torch.int16).fill_(-32768)
        return BTProtocolState(
            contribution_mailbox=contribution,
            prenorm_mailbox=prenorm,
            stage_state=torch.zeros(
                (STAGE_STATE_WORDS,), dtype=torch.int32, device=device
            ),
        )
