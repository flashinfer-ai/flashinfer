# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replay-stable device buffers for balanced PrimsTS MLA scheduling."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import functools
from types import MappingProxyType

import torch

from flashinfer.jit.prims_balanced_mla import gen_prims_balanced_mla_plan_module
from .kernels.mla_decode.helpers.constants import (
    balanced_partial_capacity,
    balanced_reducer_capacity,
    balanced_work_descriptor_capacity,
)

from ._balanced_scheduler import (
    B200_BF16_2CTA_COST,
    BalancedCostModelCalibration,
    BalancedCostModel,
    require_balanced_cost_model_calibration,
    select_calibrated_balanced_cost_model,
)

_INT32_MAX = 2**31 - 1


@functools.cache
def _get_native_planner():
    return gen_prims_balanced_mla_plan_module().build_and_load()


def _balanced_device_identity(
    device: torch.device,
) -> tuple[str, tuple[int, int], int, int]:
    """Return the exact hardware identity used by the calibration registry."""

    device = torch.device(device)
    device_index = (
        int(device.index)
        if device.index is not None
        else int(torch.cuda.current_device())
    )
    properties = torch.cuda.get_device_properties(device_index)
    return (
        str(properties.name),
        (int(properties.major), int(properties.minor)),
        int(properties.multi_processor_count),
        device_index,
    )


def require_balanced_mla_calibration(
    device: torch.device,
) -> BalancedCostModelCalibration:
    """Require a checked-in model for the exact CUDA device configuration."""

    device_name, compute_capability, sm_count, device_index = _balanced_device_identity(
        device
    )
    return require_balanced_cost_model_calibration(
        device_name=device_name,
        compute_capability=compute_capability,
        multi_processor_count=sm_count,
        device_index=device_index,
    )


class BalancedMLADecodePlan:
    """Own fixed-address work tensors and refill them between graph replays."""

    def __init__(
        self,
        *,
        batch_size: int,
        num_partitions: int,
        device: torch.device,
        k_tile_tokens: int = 128,
        cost: BalancedCostModel | None = None,
        kernel_family: str = "2cta",
        dtype_name: str = "bf16",
        calibration: BalancedCostModelCalibration | None = None,
        cost_source: str = "explicit",
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("balanced MLA plan buffers must be allocated on CUDA")

        device_name, compute_capability, sm_count, device_index = (
            _balanced_device_identity(device)
        )
        if cost is None:
            calibration = calibration or require_balanced_cost_model_calibration(
                device_name=device_name,
                compute_capability=compute_capability,
                multi_processor_count=sm_count,
                device_index=device_index,
            )
            if (
                calibration.device_name,
                calibration.compute_capability,
                calibration.multi_processor_count,
            ) != (device_name, compute_capability, sm_count):
                raise ValueError(
                    "balanced cost-model calibration does not match device"
                )

        self.batch_size = batch_size
        self.num_partitions = num_partitions
        self.descriptor_capacity = balanced_work_descriptor_capacity(
            batch_size, num_partitions
        )
        self.partial_capacity = balanced_partial_capacity(batch_size, num_partitions)
        self.reducer_capacity = balanced_reducer_capacity(batch_size, num_partitions)
        self.k_tile_tokens = k_tile_tokens
        self.device = device
        self.kernel_family = kernel_family
        self.dtype_name = dtype_name
        self.device_name = device_name
        self.compute_capability = compute_capability
        self.multi_processor_count = sm_count
        self._calibration = calibration
        self._auto_cost = cost is None
        self._cost = cost or B200_BF16_2CTA_COST
        self._cost_model_id = (
            calibration.model_id if calibration is not None else "explicit-unregistered"
        )
        self._cost_source = "calibrated-registry" if cost is None else cost_source
        self.last_cost_bucket: str | None = None

        # These are the canonical packed scheduler ABI structs, expressed as
        # contiguous int32 rows so their addresses remain stable across replay.
        self.work_descriptors = torch.empty(
            (self.descriptor_capacity, 4),
            dtype=torch.int32,
            device=device,
        )
        self.partition_offsets = torch.empty(
            (num_partitions + 1,),
            dtype=torch.int32,
            device=device,
        )
        self.combine_descriptors = torch.empty(
            (self.reducer_capacity, 4),
            dtype=torch.int32,
            device=device,
        )
        self.num_combine_descriptors = torch.empty(
            (1,),
            dtype=torch.int32,
            device=device,
        )
        self._host_seq_lens = torch.empty(
            (batch_size,),
            dtype=torch.int32,
            pin_memory=True,
        )
        self._host_plan_metadata = torch.empty(
            (3,),
            dtype=torch.int32,
            pin_memory=True,
        )
        self._host_seq_lens_np = self._host_seq_lens.numpy()
        self._host_plan_metadata_np = self._host_plan_metadata.numpy()
        self.last_descriptor_count = 0
        self.last_combine_request_count = 0
        self.last_target_piece_tiles = 0

    @property
    def cost(self) -> BalancedCostModel:
        return self._cost

    @cost.setter
    def cost(self, value: BalancedCostModel) -> None:
        """Install an explicit model, disabling automatic bucket selection."""

        self._cost = value
        self._auto_cost = False
        self._calibration = None
        self._cost_model_id = "explicit-unregistered"
        self._cost_source = "explicit"
        self.last_cost_bucket = None

    @property
    def cost_model_info(self) -> Mapping[str, object]:
        """Return an immutable snapshot of the selected model provenance."""

        return MappingProxyType(
            {
                "model_id": self._cost_model_id,
                "source": self._cost_source,
                "device_name": self.device_name,
                "compute_capability": self.compute_capability,
                "multi_processor_count": self.multi_processor_count,
                "kernel_family": self.kernel_family,
                "dtype": self.dtype_name,
                "bucket": self.last_cost_bucket,
                "parameters": self._cost,
            }
        )

    @staticmethod
    def _normalize_seq_lens(seq_lens: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(seq_lens, torch.Tensor):
            if seq_lens.device.type != "cpu":
                raise ValueError(
                    "replan() requires CPU sequence lengths; implicit D2H reads "
                    "would serialize the decode stream"
                )
            if seq_lens.ndim != 1:
                raise ValueError("seq_lens must be rank 1")
            normalized = tuple(int(value) for value in seq_lens.tolist())
        else:
            normalized = tuple(int(value) for value in seq_lens)
        if any(value < 0 or value > _INT32_MAX for value in normalized):
            raise ValueError("seq_lens must contain non-negative int32 values")
        return normalized

    def replan(
        self,
        seq_lens: Sequence[int] | torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Refill existing device tensors without changing any address."""

        normalized = self._normalize_seq_lens(seq_lens)
        self._replan_normalized(normalized, stream=stream)

    def _replan_normalized(
        self,
        normalized: tuple[int, ...],
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Refill from lengths already normalized by the public wrapper."""

        self._replan_native(normalized, forced_target_piece_tiles=0, stream=stream)

    def _replan_forced_target(
        self,
        seq_lens: Sequence[int] | torch.Tensor,
        target_piece_tiles: int,
        *,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Stage an offline-calibration schedule through the native emitter."""

        normalized = self._normalize_seq_lens(seq_lens)
        if target_piece_tiles <= 0 or target_piece_tiles > _INT32_MAX:
            raise ValueError("target_piece_tiles must be a positive int32 value")
        self._replan_native(
            normalized,
            forced_target_piece_tiles=target_piece_tiles,
            stream=stream,
        )

    def _replan_native(
        self,
        normalized: tuple[int, ...],
        *,
        forced_target_piece_tiles: int,
        stream: torch.cuda.Stream | None,
    ) -> None:
        """Refill descriptors through the sole native placement and emitter."""

        if len(normalized) != self.batch_size:
            raise ValueError(
                f"seq_lens must contain {self.batch_size} values, got {len(normalized)}"
            )
        if self._auto_cost:
            assert self._calibration is not None
            self.last_cost_bucket, self._cost = select_calibrated_balanced_cost_model(
                self._calibration,
                normalized,
                kernel_family=self.kernel_family,
                dtype_name=self.dtype_name,
                num_partitions=self.num_partitions,
                k_tile_tokens=self.k_tile_tokens,
            )
        self._host_seq_lens_np[:] = normalized
        stream_context = torch.cuda.stream(stream) if stream is not None else None
        planner_args = (
            self._host_seq_lens,
            self.work_descriptors,
            self.partition_offsets,
            self.combine_descriptors,
            self.num_combine_descriptors,
            self._host_plan_metadata,
            self.num_partitions,
            self.k_tile_tokens,
            self.cost.cost_per_k_tile,
            self.cost.fixed_piece_cost,
            self.cost.split_fixed_cost,
            self.cost.split_piece_cost,
            self.cost.reducer_fixed_cost,
            self.cost.reducer_piece_cost,
            forced_target_piece_tiles,
        )
        with torch.cuda.device(self.device):
            if stream_context is None:
                _get_native_planner().build_prims_balanced_mla_plan(*planner_args)
            else:
                with stream_context:
                    _get_native_planner().build_prims_balanced_mla_plan(*planner_args)
        self.last_descriptor_count = int(self._host_plan_metadata_np[0])
        self.last_target_piece_tiles = int(self._host_plan_metadata_np[1])
        self.last_combine_request_count = int(self._host_plan_metadata_np[2])


__all__ = ["BalancedMLADecodePlan"]
