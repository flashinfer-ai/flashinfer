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
from typing import Literal

import torch

from ..kernels.mla_decode.helpers.constants import (
    balanced_partial_capacity,
    balanced_reducer_capacity,
    balanced_work_descriptor_capacity,
)
from .jit import gen_prims_balanced_mla_plan_module

from .cost_model import (
    B200_BALANCED_COST_MODEL_ID,
    B200_BALANCED_COST_MODELS,
    B200_BF16_2CTA_COST,
    BalancedCostModelCalibration,
    BalancedCostModel,
    require_balanced_cost_model_calibration,
)

_INT32_MAX = 2**31 - 1
_DEVICE_COST_BUCKETS = (
    "single",
    "sparse_small",
    "sparse_uniform_small",
    "sparse",
    "dense_small",
    "dense_large",
    "legacy",
)
_DEVICE_PLAN_STATUS = {
    1: "sequence length is negative or exceeds the planned maximum",
    2: "derived target piece size exceeds int32 capacity",
    3: "work descriptor capacity was exceeded",
    4: "split partial capacity was exceeded",
    5: "combine descriptor capacity was exceeded",
    6: "the packed split-info 4096-piece capacity was exceeded",
}


def balanced_device_scheduler_workspace_size(
    batch_size: int, num_partitions: int
) -> int:
    """Return the graph-stable scratch size used by the device scheduler."""

    if batch_size <= 0 or num_partitions <= 0:
        raise ValueError("batch_size and num_partitions must be positive")
    return 80 * (int(batch_size) + int(num_partitions)) + 4096


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
        num_insts_kv: int = 1,
        cost: BalancedCostModel | None = None,
        evaluation_cost: BalancedCostModel | None = None,
        kernel_family: str = "2cta",
        dtype_name: str = "bf16",
        calibration: BalancedCostModelCalibration | None = None,
        cost_source: str = "explicit",
        max_seq_len: int = _INT32_MAX,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if num_partitions <= 0:
            raise ValueError("num_partitions must be positive")
        if k_tile_tokens <= 0 or k_tile_tokens > _INT32_MAX:
            raise ValueError("k_tile_tokens must be a positive int32 value")
        if num_insts_kv <= 0 or num_insts_kv > _INT32_MAX:
            raise ValueError("num_insts_kv must be a positive int32 value")
        if max_seq_len < 0 or max_seq_len > _INT32_MAX:
            raise ValueError("max_seq_len must be a non-negative int32 value")
        max_base_tiles = (max_seq_len + k_tile_tokens - 1) // k_tile_tokens
        if max_base_tiles + num_insts_kv - 1 > _INT32_MAX:
            raise ValueError(
                "the maximum aligned KV work-unit extent exceeds int32 capacity"
            )
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
        self.num_insts_kv = num_insts_kv
        self.max_seq_len = max_seq_len
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
        self._evaluation_cost = evaluation_cost
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
        self.device_plan_metadata = torch.empty(
            (5,),
            dtype=torch.int32,
            device=device,
        )
        self.device_predicted_cost = torch.empty(
            (1,),
            dtype=torch.int64,
            device=device,
        )
        self.device_scheduler_workspace = torch.empty(
            balanced_device_scheduler_workspace_size(batch_size, num_partitions),
            dtype=torch.uint8,
            device=device,
        )
        self._device_cost_models = self._make_device_cost_models()
        self._device_evaluation_cost_model = torch.tensor(
            self._cost_row(evaluation_cost or BalancedCostModel(0, 0)),
            dtype=torch.int32,
            device=self.device,
        )
        self.last_descriptor_count = 0
        self.last_combine_request_count = 0
        self.last_target_piece_tiles = 0
        self.last_predicted_cost: int | None = None
        self._predicted_cost_pending = False

    @staticmethod
    def _cost_row(cost: BalancedCostModel) -> tuple[int, ...]:
        row = (
            cost.cost_per_k_tile,
            cost.fixed_piece_cost,
            cost.split_fixed_cost,
            cost.split_piece_cost,
            cost.reducer_fixed_cost,
            cost.reducer_piece_cost,
        )
        if any(value < 0 or value > _INT32_MAX for value in row):
            raise ValueError(
                "balanced cost-model coefficients must be non-negative int32 values"
            )
        return row

    def _make_device_cost_models(self) -> torch.Tensor:
        if not self._auto_cost:
            rows = [self._cost_row(self._cost)] * len(_DEVICE_COST_BUCKETS)
        else:
            assert self._calibration is not None
            if self._calibration.model_id != B200_BALANCED_COST_MODEL_ID:
                raise RuntimeError(
                    f"balanced cost-model calibration {self._calibration.model_id!r} "
                    "has no device selector"
                )
            family_aliases = {
                "1cta": "1cta",
                "2cta": "2cta",
                "throughput_latency_1cta": "1cta",
                "throughput_2cta": "2cta",
            }
            dtype_aliases = {
                "bf16": "bf16",
                "bfloat16": "bf16",
                "fp8": "fp8",
                "e4m3": "fp8",
                "float8_e4m3fn": "fp8",
            }
            try:
                family = family_aliases[self.kernel_family]
            except KeyError as error:
                raise ValueError(
                    f"unsupported balanced kernel family {self.kernel_family!r}"
                ) from error
            try:
                dtype = dtype_aliases[self.dtype_name]
            except KeyError as error:
                raise ValueError(
                    f"unsupported balanced input dtype {self.dtype_name!r}"
                ) from error
            rows = []
            for bucket in _DEVICE_COST_BUCKETS:
                cost = (
                    B200_BF16_2CTA_COST
                    if bucket == "legacy"
                    else B200_BALANCED_COST_MODELS[(family, dtype, bucket)]
                )
                rows.append(self._cost_row(cost))
        return torch.tensor(rows, dtype=torch.int32, device=self.device)

    def _cost_for_device_bucket(self, bucket: str) -> BalancedCostModel:
        if bucket == "legacy":
            return B200_BF16_2CTA_COST
        # The table was normalized and validated when it was materialized.
        family = (
            "1cta"
            if self.kernel_family in {"1cta", "throughput_latency_1cta"}
            else "2cta"
        )
        dtype = "fp8" if self.dtype_name in {"fp8", "e4m3", "float8_e4m3fn"} else "bf16"
        return B200_BALANCED_COST_MODELS[(family, dtype, bucket)]

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
        self._device_cost_models = self._make_device_cost_models()

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
                "evaluation_parameters": self._evaluation_cost,
                "last_predicted_cost": self.last_predicted_cost,
            }
        )

    @staticmethod
    def _normalize_seq_lens(seq_lens: Sequence[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(seq_lens, torch.Tensor):
            if seq_lens.ndim != 1:
                raise ValueError("seq_lens must be rank 1")
            normalized = tuple(int(value) for value in seq_lens.tolist())
        else:
            normalized = tuple(int(value) for value in seq_lens)
        if any(value < 0 or value > _INT32_MAX for value in normalized):
            raise ValueError("seq_lens must contain non-negative int32 values")
        return normalized

    def schedule_device(
        self,
        seq_lens: torch.Tensor,
        *,
        stream: torch.cuda.Stream | None = None,
        validate: bool = True,
        scheduler: Literal["exact", "optimized"] = "optimized",
        estimate_cost: bool = False,
        forced_target_piece_tiles: int = 0,
    ) -> None:
        """Build the schedule from live CUDA lengths without synchronizing.

        With ``validate=False`` the call is graph-capturable and all dynamic
        metadata remains device-resident. ``validate=True`` synchronizes after
        launch, raises a descriptive scheduling error, and refreshes the host
        diagnostic properties. ``scheduler="optimized"`` uses the production
        two-kernel rank-and-fold policy; ``scheduler="exact"`` uses the
        deterministic four-kernel reference policy. ``estimate_cost=True``
        additionally publishes the final emitted placement's modeled
        makespan. A constructor-supplied ``evaluation_cost`` is used for that
        estimate without changing target selection or placement.
        """

        if not isinstance(seq_lens, torch.Tensor):
            raise TypeError("seq_lens must be a torch.Tensor")
        planned_device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        runtime_device_index = (
            seq_lens.device.index
            if seq_lens.device.index is not None
            else torch.cuda.current_device()
        )
        if (
            seq_lens.device.type != "cuda"
            or runtime_device_index != planned_device_index
        ):
            raise ValueError(f"seq_lens must be on {self.device}")
        if seq_lens.dtype != torch.int32:
            raise ValueError("seq_lens must have dtype int32")
        if seq_lens.ndim != 1 or seq_lens.numel() != self.batch_size:
            raise ValueError(
                f"seq_lens must have shape [{self.batch_size}], got {tuple(seq_lens.shape)}"
            )
        if not seq_lens.is_contiguous():
            raise ValueError("seq_lens must be contiguous")
        if not isinstance(validate, bool):
            raise TypeError("validate must be a bool")
        if not isinstance(scheduler, str):
            raise TypeError("scheduler must be a string")
        if scheduler not in {"exact", "optimized"}:
            raise ValueError("scheduler must be 'exact' or 'optimized'")
        if not isinstance(estimate_cost, bool):
            raise TypeError("estimate_cost must be a bool")
        if forced_target_piece_tiles < 0 or forced_target_piece_tiles > _INT32_MAX:
            raise ValueError(
                "forced_target_piece_tiles must be a non-negative int32 value"
            )
        if forced_target_piece_tiles + self.num_insts_kv - 1 > _INT32_MAX:
            raise ValueError("the aligned forced target exceeds int32 capacity")

        planner_args = (
            seq_lens,
            self.work_descriptors,
            self.partition_offsets,
            self.combine_descriptors,
            self.num_combine_descriptors,
            self.device_plan_metadata,
            self.device_predicted_cost,
            self._device_evaluation_cost_model,
            self.device_scheduler_workspace,
            self._device_cost_models,
            self.num_partitions,
            self.k_tile_tokens,
            self.num_insts_kv,
            self.max_seq_len,
            self._auto_cost,
            scheduler == "optimized",
            estimate_cost,
            self._evaluation_cost is not None,
            forced_target_piece_tiles,
        )
        self.last_predicted_cost = None
        self._predicted_cost_pending = estimate_cost
        stream_context = torch.cuda.stream(stream) if stream is not None else None
        with torch.cuda.device(self.device):
            if stream_context is None:
                _get_native_planner().build_prims_balanced_mla_plan_device(
                    *planner_args
                )
            else:
                with stream_context:
                    _get_native_planner().build_prims_balanced_mla_plan_device(
                        *planner_args
                    )
        if validate:
            self.synchronize_device_metadata(stream=stream)

    def synchronize_device_metadata(
        self, *, stream: torch.cuda.Stream | None = None
    ) -> tuple[int, int, int, int, int]:
        """Synchronize and return device-planner diagnostics for tests/debugging."""

        active_stream = stream or torch.cuda.current_stream(self.device)
        active_stream.synchronize()
        metadata_values = [int(value) for value in self.device_plan_metadata.cpu()]
        metadata = (
            metadata_values[0],
            metadata_values[1],
            metadata_values[2],
            metadata_values[3],
            metadata_values[4],
        )
        status = metadata[3]
        if status:
            detail = _DEVICE_PLAN_STATUS.get(status, f"unknown status {status}")
            raise RuntimeError(f"balanced device scheduler failed: {detail}")
        self.last_descriptor_count = metadata[0]
        self.last_target_piece_tiles = metadata[1]
        self.last_combine_request_count = metadata[2]
        if self._auto_cost:
            bucket_index = metadata[4]
            if bucket_index < 0 or bucket_index >= len(_DEVICE_COST_BUCKETS):
                raise RuntimeError(
                    f"balanced device scheduler returned invalid cost bucket {bucket_index}"
                )
            self.last_cost_bucket = _DEVICE_COST_BUCKETS[bucket_index]
            self._cost = self._cost_for_device_bucket(self.last_cost_bucket)
        if self._predicted_cost_pending:
            self.last_predicted_cost = int(self.device_predicted_cost.item())
            self._predicted_cost_pending = False
        return metadata


__all__ = ["BalancedMLADecodePlan", "balanced_device_scheduler_workspace_size"]
