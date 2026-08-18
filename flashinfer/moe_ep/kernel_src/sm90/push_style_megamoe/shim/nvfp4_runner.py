"""SM90 push runners for NVFP4 W4A8 and W4A16-RS weights."""

from __future__ import annotations

import os
import socket
import weakref
from typing import Any, Protocol, TypeAlias, cast

import torch

from ......fused_moe.sm90_nvfp4_repack import (
    NVFP4RSWeightView,
    NVFP4SM90WeightViewV3,
    NVFP4SM90WeightViewV4,
)
from .nvfp4_rs_gemm import create_sm90_push_nvfp4_rs_gemm_runner
from .nvfp4_w4a8_gemm import (
    _W4A8ScheduleWorkspace,
    create_sm90_push_nvfp4_w4a8_gemm,
)
from .gemm import create_sm90_push_fp8_moe_gemm_runner
from .nvfp4_weights import (
    Sm90PushNvFp4DualWeights,
    Sm90PushNvFp4HotFoldedWeights,
    Sm90PushNvFp4Weights,
)
from .protocol import Sm90PushCombine, Sm90PushPipe, _run_guarded_phase
from .runner import Sm90PushMoERunner

W4A8WeightView: TypeAlias = NVFP4SM90WeightViewV3 | NVFP4SM90WeightViewV4


def _align(value: int, alignment: int = 128) -> int:
    return (value + alignment - 1) // alignment * alignment


def _padded_scale_stride(max_rows: int, num_experts: int) -> int:
    return max((max_rows + num_experts * 31) // 32 * 32, 1)


class _W4A8LayerRunner:
    def __init__(
        self,
        max_rows: int,
        total_experts: int,
        padded_scale_stride: int,
        weights: W4A8WeightView,
        *,
        device: torch.device,
        tma_cache_capacity: int = 128,
        prefer_n64_main: bool = False,
        allow_legacy_layout: bool = False,
        shared_schedule_workspace: _W4A8ScheduleWorkspace | None = None,
        counter_bank: int = 0,
    ) -> None:
        if not isinstance(weights, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
            raise TypeError("W4A8 weights must be an SM90 W4A8 view")
        self.max_rows = int(max_rows)
        self.total_experts = int(total_experts)
        self.padded_scale_stride = int(padded_scale_stride)
        if self.max_rows <= 0:
            raise ValueError("W4A8 max_rows must be positive")
        if self.padded_scale_stride != _padded_scale_stride(
            self.max_rows, self.total_experts
        ):
            raise ValueError("W4A8 activation scale stride is inconsistent")
        if weights.packed_e2m1.device != device:
            raise ValueError("W4A8 weights must be on the pipe device")
        logical_shape = weights.manifest.logical_shape
        experts = int(logical_shape[0])
        self.n = int(logical_shape[1])
        self.k = int(logical_shape[2])
        mapping = tuple(weights.manifest.expert_mapping)
        if experts <= 0 or experts > self.total_experts:
            raise ValueError("W4A8 weight expert count is outside the pipe")
        if any(expert < 0 or expert >= total_experts for expert in mapping):
            raise ValueError("W4A8 expert mapping is outside the pipe")
        if self.k % 128:
            raise ValueError("W4A8 activation K must be divisible by 128")
        self.runner = create_sm90_push_nvfp4_w4a8_gemm(
            self.max_rows,
            weights,
            total_experts=self.total_experts,
            tma_cache_capacity=tma_cache_capacity,
            prefer_n64_main=prefer_n64_main,
            payload_layout=weights.manifest.layout_version,
            allow_legacy_layout=allow_legacy_layout,
            shared_schedule_workspace=shared_schedule_workspace,
            counter_bank=counter_bank,
        )

    def _validate_weight_view(self, weights: W4A8WeightView) -> None:
        if not isinstance(weights, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
            raise TypeError("W4A8 weights must be an SM90 W4A8 view")
        weights.verify_checksums()
        current = self.runner.weight_view
        if weights.packed_e2m1.device != current.packed_e2m1.device:
            raise ValueError("W4A8 rebound weights must remain on the runner device")
        if weights.manifest.logical_shape != current.manifest.logical_shape:
            raise ValueError("W4A8 rebound weights must preserve the logical shape")
        if weights.manifest.padded_shape != current.manifest.padded_shape:
            raise ValueError("W4A8 rebound weights must preserve the padded shape")
        if weights.manifest.group_size != current.manifest.group_size:
            raise ValueError("W4A8 rebound weights must preserve group_size")
        if weights.manifest.residual_scheme != current.manifest.residual_scheme:
            raise ValueError("W4A8 rebound weights must preserve residual_scheme")
        if weights.manifest.layout_version != current.manifest.layout_version:
            raise ValueError("W4A8 rebound weights must preserve payload layout")
        if weights.manifest.expert_mapping != current.manifest.expert_mapping:
            raise ValueError("W4A8 rebound weights must preserve expert_mapping")

    def _bind_weight_view(self, weights: W4A8WeightView) -> None:
        self.runner.weight_view = weights

    def run(
        self,
        output: torch.Tensor,
        activation: torch.Tensor,
        activation_scales: torch.Tensor,
        offsets: torch.Tensor,
        *,
        prepare_schedule: bool = True,
    ) -> None:
        scale_count = (self.k // 128) * self.padded_scale_stride
        scales = activation_scales[:scale_count].view(
            self.k // 128, self.padded_scale_stride
        )
        self.runner.run(
            activation[: self.max_rows, : self.k].view(torch.float8_e4m3fn),
            scales,
            offsets,
            out=output[: self.max_rows, : self.n],
            trusted_offsets=True,
            prepare_schedule=prepare_schedule,
        )


class _W4A8PairEngine(Protocol):
    I: int
    execution_identity: tuple[object, ...]

    def validate_weights(self, weights: object) -> None: ...

    def bind_validated_weights(self, weights: object) -> None: ...

    def warm_legacy(self, owner: "Sm90PushNvFp4MoERunner") -> None: ...

    def prepare_fp8_collective(self, owner: "Sm90PushNvFp4MoERunner") -> None: ...

    def run_fc1(self, owner: "Sm90PushNvFp4MoERunner") -> None: ...

    def run_fc2(self, owner: "Sm90PushNvFp4MoERunner") -> None: ...

    def destroy(self) -> None: ...


class _LegacyW4A8PairEngine:
    fc1: Any
    fc2: Any

    def __init__(
        self,
        max_rows: int,
        total_experts: int,
        padded_scale_stride: int,
        weights: Sm90PushNvFp4Weights,
        *,
        device: torch.device,
        hidden_size: int,
        expected_m: int,
        n64_expected_m_per_sm: float,
        tma_cache_capacity: int,
        allow_legacy_layout: bool,
    ) -> None:
        self.total_experts = total_experts
        self.validate_weights(weights)
        w13 = cast(W4A8WeightView, weights.w13)
        w2 = cast(W4A8WeightView, weights.w2)
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        threshold = float(n64_expected_m_per_sm) * sm_count
        prefer_n64_main = expected_m <= threshold
        self.selection_provenance = {
            "sm_count": sm_count,
            "expected_m": int(expected_m),
            "threshold_k": float(n64_expected_m_per_sm),
            "threshold_rows": threshold,
            "prefer_n64_main": prefer_n64_main,
        }
        self.fc1 = _W4A8LayerRunner(
            max_rows,
            total_experts,
            padded_scale_stride,
            w13,
            device=device,
            tma_cache_capacity=tma_cache_capacity,
            prefer_n64_main=prefer_n64_main,
            allow_legacy_layout=allow_legacy_layout,
        )
        if self.fc1.n <= 0 or self.fc1.n % 2 or self.fc1.k != hidden_size:
            raise ValueError("W4A8 w13 logical shape must be (E, 2I, H)")
        self.I = self.fc1.n // 2
        self.fc2 = _W4A8LayerRunner(
            max_rows,
            total_experts,
            padded_scale_stride,
            w2,
            device=device,
            tma_cache_capacity=tma_cache_capacity,
            prefer_n64_main=prefer_n64_main,
            allow_legacy_layout=allow_legacy_layout,
            shared_schedule_workspace=self.fc1.runner.schedule_workspace,
            counter_bank=1,
        )
        if (self.fc2.n, self.fc2.k) != (hidden_size, self.I):
            raise ValueError("W4A8 w2 logical shape must be (E, H, I)")
        self.execution_identity: tuple[object, ...] = (
            "legacy-v1",
            total_experts,
        )

    def validate_weights(self, weights: object) -> None:
        if not isinstance(weights, Sm90PushNvFp4Weights):
            raise TypeError("legacy W4A8 engine requires Sm90PushNvFp4Weights")
        if weights.nvfp4_mode != "w4a8":
            raise ValueError("legacy W4A8 engine requires w4a8 weights")
        for view in (weights.w13, weights.w2):
            if not isinstance(view, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
                raise TypeError("legacy W4A8 engine requires two W4A8 views")
            if view.manifest.expert_mapping != tuple(range(self.total_experts)):
                raise ValueError("legacy W4A8 weights must map every expert in order")
        if hasattr(self, "fc1"):
            self.fc1._validate_weight_view(cast(W4A8WeightView, weights.w13))
            self.fc2._validate_weight_view(cast(W4A8WeightView, weights.w2))

    def bind_validated_weights(self, weights: object) -> None:
        typed = cast(Sm90PushNvFp4Weights, weights)
        self.fc1._bind_weight_view(cast(W4A8WeightView, typed.w13))
        self.fc2._bind_weight_view(cast(W4A8WeightView, typed.w2))

    def warm_legacy(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        offsets = torch.zeros(
            self.total_experts + 1, dtype=torch.int64, device=owner.pipe.device
        )
        self.fc1.run(owner.h, owner.a1, owner.sfa1, offsets, prepare_schedule=True)
        self.fc2.run(owner.y, owner.a2, owner.sfa2, offsets, prepare_schedule=False)
        torch.cuda.synchronize()

    def prepare_fp8_collective(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        del owner

    def run_fc1(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self.fc1.run(
            owner.h,
            owner.a1,
            owner.sfa1,
            owner.pipe._offsets,
            prepare_schedule=True,
        )

    def run_fc2(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self.fc2.run(
            owner.y,
            owner.a2,
            owner.sfa2,
            owner.pipe._offsets,
            prepare_schedule=False,
        )

    def destroy(self) -> None:
        self.fc1 = None
        self.fc2 = None


class _HotFoldedW4A8PairEngine:
    def __init__(
        self,
        max_rows: int,
        total_experts: int,
        padded_scale_stride: int,
        weights: Sm90PushNvFp4HotFoldedWeights,
        *,
        device: torch.device,
        expected_m: int,
        hidden_size: int,
        n64_expected_m_per_sm: float,
        tma_cache_capacity: int,
        allow_legacy_layout: bool,
    ) -> None:
        self.max_rows = max_rows
        self.total_experts = total_experts
        self.padded_scale_stride = padded_scale_stride
        self.expected_m = expected_m
        self.device = device
        self.hidden_size = hidden_size
        self.hot_experts = weights.hot_experts
        sm_count = int(torch.cuda.get_device_properties(device).multi_processor_count)
        threshold = float(n64_expected_m_per_sm) * sm_count
        self.prefer_n64_main = expected_m <= threshold
        self.selection_provenance = {
            "sm_count": sm_count,
            "expected_m": int(expected_m),
            "threshold_k": float(n64_expected_m_per_sm),
            "threshold_rows": threshold,
            "prefer_n64_main": self.prefer_n64_main,
        }
        self.validate_weights(weights)
        self.hot_runner = None
        self.hot_workspace = None
        self.cold_fc1 = None
        self.cold_fc2 = None

        if weights.hot_fp8 is not None:
            hot = weights.hot_fp8
            self.I = int(hot.w13_fp8.shape[1]) // 2
            if int(hot.w13_fp8.shape[2]) != hidden_size:
                raise ValueError("hot FP8 FC1 hidden size does not match the pipe")
            self.hot_runner = create_sm90_push_fp8_moe_gemm_runner()
            size = int(
                self.hot_runner.get_moe_workspace_size_with_scale_problems(
                    expected_m,
                    max_rows,
                    max(2 * self.I, int(hot.w2_fp8.shape[1])),
                    max(int(hot.w13_fp8.shape[2]), int(hot.w2_fp8.shape[2])),
                    self.hot_experts,
                    total_experts,
                    True,
                    True,
                )
            )
            self.hot_workspace = torch.empty(
                max(size, 1), dtype=torch.uint8, device=device
            )
            self.hot_runner.configure_workspace(self.hot_workspace)
        if weights.cold_nvfp4 is not None:
            cold = weights.cold_nvfp4
            w13 = cast(W4A8WeightView, cold.w13)
            w2 = cast(W4A8WeightView, cold.w2)
            self.cold_fc1 = _W4A8LayerRunner(
                max_rows,
                total_experts,
                padded_scale_stride,
                w13,
                device=device,
                tma_cache_capacity=tma_cache_capacity,
                prefer_n64_main=self.prefer_n64_main,
                allow_legacy_layout=allow_legacy_layout,
            )
            cold_i = self.cold_fc1.n // 2
            if hasattr(self, "I") and cold_i != self.I:
                raise ValueError(
                    "hot and cold FC1 weights disagree on intermediate size"
                )
            self.I = cold_i
            self.cold_fc2 = _W4A8LayerRunner(
                max_rows,
                total_experts,
                padded_scale_stride,
                w2,
                device=device,
                tma_cache_capacity=tma_cache_capacity,
                prefer_n64_main=self.prefer_n64_main,
                allow_legacy_layout=allow_legacy_layout,
                shared_schedule_workspace=self.cold_fc1.runner.schedule_workspace,
                counter_bank=1,
            )
            if self.cold_fc1.k != hidden_size or (
                self.cold_fc2.n,
                self.cold_fc2.k,
            ) != (hidden_size, self.I):
                raise ValueError("cold W4A8 weight shapes do not match the pipe")
        self.execution_identity: tuple[object, ...] = weights.execution_identity
        self.weights = weights

    @staticmethod
    def _validate_fp8_tensor(
        name: str,
        tensor: torch.Tensor,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        if tuple(tensor.shape) != shape:
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device != device:
            raise ValueError(f"{name} must be on {device}, got {tensor.device}")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    def validate_weights(self, weights: object) -> None:
        if not isinstance(weights, Sm90PushNvFp4HotFoldedWeights):
            raise TypeError("hot-folded engine requires Sm90PushNvFp4HotFoldedWeights")
        weights.__post_init__()
        if weights.execution_identity != (
            "hot-prefix-v1",
            self.hot_experts,
            self.total_experts,
            (
                0
                if weights.cold_nvfp4 is None
                else cast(
                    W4A8WeightView,
                    weights.cold_nvfp4.w13,
                ).manifest.layout_version
            ),
        ):
            raise ValueError("hot-folded rebound weights must preserve the hot prefix")
        device = None
        intermediate = None
        if weights.hot_fp8 is not None:
            hot = weights.hot_fp8
            device = hot.w13_fp8.device
            if device != self.device:
                raise ValueError("hot-folded weights must remain on the pipe device")
            intermediate = int(hot.w13_fp8.shape[1]) // 2
            hidden = self.hidden_size
            expected = (
                (
                    "hot.w13_fp8",
                    hot.w13_fp8,
                    (self.hot_experts, 2 * intermediate, hidden),
                    torch.float8_e4m3fn,
                ),
                (
                    "hot.w13_sf",
                    hot.w13_sf,
                    (
                        self.hot_experts,
                        2 * intermediate // 128,
                        hidden // 128,
                    ),
                    torch.float32,
                ),
                (
                    "hot.w2_fp8",
                    hot.w2_fp8,
                    (self.hot_experts, hidden, intermediate),
                    torch.float8_e4m3fn,
                ),
                (
                    "hot.w2_sf",
                    hot.w2_sf,
                    (
                        self.hot_experts,
                        hidden // 128,
                        intermediate // 128,
                    ),
                    torch.float32,
                ),
            )
            for name, tensor, shape, dtype in expected:
                self._validate_fp8_tensor(name, tensor, shape, dtype, device)
        if weights.cold_nvfp4 is not None:
            cold = weights.cold_nvfp4
            w13 = cast(W4A8WeightView, cold.w13)
            w2 = cast(W4A8WeightView, cold.w2)
            if device is not None and w13.packed_e2m1.device != device:
                raise ValueError("hot and cold weights must share a device")
            if w13.packed_e2m1.device != self.device:
                raise ValueError("hot-folded weights must remain on the pipe device")
            if hasattr(self, "cold_fc1") and self.cold_fc1 is not None:
                self.cold_fc1._validate_weight_view(w13)
                cast(_W4A8LayerRunner, self.cold_fc2)._validate_weight_view(w2)
            cold_i = int(w13.manifest.logical_shape[1]) // 2
            if intermediate is not None and intermediate != cold_i:
                raise ValueError("hot and cold weights disagree on intermediate size")

    def bind_validated_weights(self, weights: object) -> None:
        typed = cast(Sm90PushNvFp4HotFoldedWeights, weights)
        hot = typed.hot_fp8
        cold = typed.cold_nvfp4
        if hot is not None:
            self.hot_w13_fp8 = hot.w13_fp8
            self.hot_w13_sf = hot.w13_sf
            self.hot_w2_fp8 = hot.w2_fp8
            self.hot_w2_sf = hot.w2_sf
        if cold is not None:
            cast(_W4A8LayerRunner, self.cold_fc1)._bind_weight_view(
                cast(W4A8WeightView, cold.w13)
            )
            cast(_W4A8LayerRunner, self.cold_fc2)._bind_weight_view(
                cast(W4A8WeightView, cold.w2)
            )
        self.weights = typed

    def warm_legacy(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self.bind_validated_weights(self.weights)
        if self.cold_fc1 is None:
            return
        offsets = torch.zeros(
            self.total_experts + 1, dtype=torch.int64, device=owner.pipe.device
        )
        self.cold_fc1.run(owner.h, owner.a1, owner.sfa1, offsets, prepare_schedule=True)
        cast(_W4A8LayerRunner, self.cold_fc2).run(
            owner.y, owner.a2, owner.sfa2, offsets, prepare_schedule=False
        )
        torch.cuda.synchronize()

    def _prepare_fp8(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        if self.hot_runner is None:
            return
        offsets = torch.zeros(
            self.hot_experts + 1, dtype=torch.int64, device=owner.pipe.device
        )
        self.hot_runner.moe_gemm(
            owner.h,
            owner.a1.view(torch.float8_e4m3fn),
            self.hot_w13_fp8,
            offsets,
            2 * self.I,
            owner.pipe.H,
            owner.sfa1,
            self.hot_w13_sf,
            True,
        )
        self.hot_runner.moe_gemm(
            owner.y,
            owner.a2.view(torch.float8_e4m3fn),
            self.hot_w2_fp8,
            offsets,
            owner.pipe.H,
            self.I,
            owner.sfa2,
            self.hot_w2_sf,
            True,
        )
        torch.cuda.synchronize()

    def prepare_fp8_collective(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        if self.hot_runner is None:
            return
        pipe = owner.pipe
        rank = getattr(pipe, "rank", 0)
        props = torch.cuda.get_device_properties(pipe.device)
        cache_dir = os.path.normcase(
            os.path.realpath(str(self.hot_runner.get_deepgemm_cache_dir()))
        )
        compiler = str(self.hot_runner.get_deepgemm_nvcc_compiler())
        cache_identities = _run_guarded_phase(
            pipe._comm,
            rank,
            "hot-fp8-cache-layout",
            lambda: (
                socket.gethostname(),
                cache_dir,
                int(props.major),
                int(props.minor),
                int(props.multi_processor_count),
                bool(self.hot_runner.is_deepgemm_jit_enabled()),
                self.expected_m,
                self.hot_experts,
                pipe.H,
                self.I,
                owner._nvcc_available(compiler),
            ),
        )
        local_group = cache_identities[rank][:-1]
        peers = [
            peer_rank
            for peer_rank, identity in enumerate(cache_identities)
            if identity[:-1] == local_group
        ]
        leader_rank = next(
            (peer_rank for peer_rank in peers if cache_identities[peer_rank][-1]),
            peers[0],
        )
        _run_guarded_phase(
            pipe._comm,
            rank,
            "hot-fp8-cache-warm",
            lambda: self._prepare_fp8(owner) if rank == leader_rank else None,
        )
        _run_guarded_phase(
            pipe._comm,
            rank,
            "hot-fp8-cache-load",
            lambda: self._prepare_fp8(owner) if rank != leader_rank else None,
        )

    def _hot_offsets(self, owner: "Sm90PushNvFp4MoERunner") -> torch.Tensor:
        return owner.pipe._offsets[: self.hot_experts + 1]

    def run_fc1(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        if self.hot_runner is not None:
            self.hot_runner.moe_gemm(
                owner.h,
                owner.a1.view(torch.float8_e4m3fn),
                self.hot_w13_fp8,
                self._hot_offsets(owner),
                2 * self.I,
                owner.pipe.H,
                owner.sfa1,
                self.hot_w13_sf,
                True,
            )
        if self.cold_fc1 is not None:
            self.cold_fc1.run(
                owner.h,
                owner.a1,
                owner.sfa1,
                owner.pipe._offsets,
                prepare_schedule=True,
            )

    def run_fc2(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        if self.hot_runner is not None:
            self.hot_runner.moe_gemm(
                owner.y,
                owner.a2.view(torch.float8_e4m3fn),
                self.hot_w2_fp8,
                self._hot_offsets(owner),
                owner.pipe.H,
                self.I,
                owner.sfa2,
                self.hot_w2_sf,
                True,
            )
        if self.cold_fc2 is not None:
            self.cold_fc2.run(
                owner.y,
                owner.a2,
                owner.sfa2,
                owner.pipe._offsets,
                prepare_schedule=False,
            )

    def destroy(self) -> None:
        self.hot_runner = None
        self.hot_workspace = None
        self.cold_fc1 = None
        self.cold_fc2 = None
        self.weights = None


class _DualW4A8PairEngine:
    """Execute folded FP8 while retaining and validating full packed weights."""

    def __init__(
        self,
        max_rows: int,
        total_experts: int,
        padded_scale_stride: int,
        weights: Sm90PushNvFp4DualWeights,
        *,
        device: torch.device,
        expected_m: int,
        hidden_size: int,
        n64_expected_m_per_sm: float,
        tma_cache_capacity: int,
        allow_legacy_layout: bool,
    ) -> None:
        weights.__post_init__()
        self.total_experts = total_experts
        self.execution_identity: tuple[object, ...] = weights.execution_identity
        if weights.total_experts != total_experts:
            raise ValueError("dual weights must cover every local expert")
        self._validate_packed(weights.packed_nvfp4)
        projected = self._project(weights)
        self._folded_engine = _HotFoldedW4A8PairEngine(
            max_rows,
            total_experts,
            padded_scale_stride,
            projected,
            device=device,
            expected_m=expected_m,
            hidden_size=hidden_size,
            n64_expected_m_per_sm=n64_expected_m_per_sm,
            tma_cache_capacity=tma_cache_capacity,
            allow_legacy_layout=allow_legacy_layout,
        )
        self.I = self._folded_engine.I
        self.selection_provenance = self._folded_engine.selection_provenance
        self.weights = weights

    @staticmethod
    def _project(weights: Sm90PushNvFp4DualWeights) -> Sm90PushNvFp4HotFoldedWeights:
        return Sm90PushNvFp4HotFoldedWeights(
            hot_experts=weights.total_experts,
            total_experts=weights.total_experts,
            hot_fp8=weights.folded_fp8,
            cold_nvfp4=None,
        )

    def _validate_packed(self, packed: Sm90PushNvFp4Weights) -> None:
        if packed.nvfp4_mode != "w4a8":
            raise ValueError("dual packed weights must use W4A8")
        for view in (packed.w13, packed.w2):
            if not isinstance(view, (NVFP4SM90WeightViewV3, NVFP4SM90WeightViewV4)):
                raise TypeError("dual packed weights must contain W4A8 views")
            if view.manifest.expert_mapping != tuple(range(self.total_experts)):
                raise ValueError("dual packed weights must map every local expert")
            view.verify_checksums()

    def validate_weights(self, weights: object) -> None:
        if not isinstance(weights, Sm90PushNvFp4DualWeights):
            raise TypeError("dual W4A8 engine requires Sm90PushNvFp4DualWeights")
        weights.__post_init__()
        if weights.execution_identity != self.execution_identity:
            raise ValueError(
                "dual rebound weights must preserve layout and expert count"
            )
        self._validate_packed(weights.packed_nvfp4)
        self._folded_engine.validate_weights(self._project(weights))

    def bind_validated_weights(self, weights: object) -> None:
        typed = cast(Sm90PushNvFp4DualWeights, weights)
        self._folded_engine.bind_validated_weights(self._project(typed))
        self.weights = typed

    def warm_legacy(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self._folded_engine.warm_legacy(owner)

    def prepare_fp8_collective(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self._folded_engine.prepare_fp8_collective(owner)

    def run_fc1(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self._folded_engine.run_fc1(owner)

    def run_fc2(self, owner: "Sm90PushNvFp4MoERunner") -> None:
        self._folded_engine.run_fc2(owner)

    def destroy(self) -> None:
        self._folded_engine.destroy()
        self.weights = None


class Sm90PushNvFp4MoERunner(Sm90PushMoERunner):
    """Two-phase SM90 push runner selected by a typed NVFP4 weight bundle."""

    # W4A8 binds a pair engine; W4A16-RS binds the FFI runner objects.
    fc1: Any
    fc2: Any

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: (
            Sm90PushNvFp4Weights
            | Sm90PushNvFp4HotFoldedWeights
            | Sm90PushNvFp4DualWeights
        ),
        *,
        rs_n_tactic: int = 64,
        rs_stages: int = 3,
        rs_stage_k: int = 64,
        tma_cache_capacity: int = 128,
        n64_expected_m_per_sm: float = 4.0,
        payload_layout: int = 4,
        allow_legacy_layout: bool = False,
    ) -> None:
        if not isinstance(
            weights,
            (
                Sm90PushNvFp4Weights,
                Sm90PushNvFp4HotFoldedWeights,
                Sm90PushNvFp4DualWeights,
            ),
        ):
            raise TypeError("weights must be a typed SM90 push NVFP4 bundle")
        self._init_round_state(pipe)
        self.weights = weights
        self.nvfp4_mode = (
            weights.nvfp4_mode if isinstance(weights, Sm90PushNvFp4Weights) else "w4a8"
        )
        self._rs_n_tactic = int(rs_n_tactic)
        self._rs_stages = int(rs_stages)
        self._rs_stage_k = int(rs_stage_k)
        self._tma_cache_capacity = int(tma_cache_capacity)
        self._n64_expected_m_per_sm = float(n64_expected_m_per_sm)
        self._payload_layout = int(payload_layout)
        self._allow_legacy_layout = bool(allow_legacy_layout)

        def _local_init() -> tuple[object, ...]:
            if pipe.config.fuse_fc1_epilogue:
                raise ValueError("SM90 push NVFP4 does not support fused FC1")
            if self.nvfp4_mode == "w4a8":
                packed = None
                if isinstance(weights, Sm90PushNvFp4Weights):
                    packed = weights
                elif isinstance(weights, Sm90PushNvFp4HotFoldedWeights):
                    packed = weights.cold_nvfp4
                elif isinstance(weights, Sm90PushNvFp4DualWeights):
                    packed = weights.packed_nvfp4
                if packed is not None:
                    layout = cast(W4A8WeightView, packed.w13).manifest.layout_version
                    if layout != self._payload_layout:
                        raise ValueError(
                            "NVFP4 payload layout does not match the runner config"
                        )
                    if layout == 3 and not self._allow_legacy_layout:
                        raise ValueError(
                            "NVFP4 payload layout 3 requires allow_legacy_layout=True"
                        )
                self._init_w4a8(weights)
            else:
                if not isinstance(weights, Sm90PushNvFp4Weights):
                    raise TypeError("W4A16-RS requires Sm90PushNvFp4Weights")
                if pipe.config.combine_dtype is not Sm90PushCombine.BF16:
                    raise ValueError("W4A16-RS supports only BF16 combine wire")
                if pipe.config.fuse_act:
                    raise ValueError("W4A16-RS requires fuse_act=False")
                if (
                    self._rs_n_tactic,
                    self._rs_stages,
                    self._rs_stage_k,
                ) != (64, 3, 64):
                    raise ValueError("W4A16-RS supports only the N64/S3/K64 tactic")
                self._init_rs(
                    cast(NVFP4RSWeightView, weights.w13),
                    cast(NVFP4RSWeightView, weights.w2),
                )

            return self.execution_identity

        identities = _run_guarded_phase(
            pipe._comm,
            getattr(pipe, "rank", 0),
            "nvfp4-weights+gemm-resources",
            _local_init,
        )
        if any(identity != identities[0] for identity in identities[1:]):
            raise RuntimeError("NVFP4 execution identity must match on every EP rank")
        if self.nvfp4_mode == "w4a8":
            self._w4a8_engine.prepare_fp8_collective(self)
        self._bound_weights = weights
        self._validated_weights[id(weights)] = weakref.ref(weights)

    def _validate_rs_weights(
        self,
        w13: NVFP4RSWeightView,
        w2: NVFP4RSWeightView,
    ) -> None:
        if not isinstance(w13, NVFP4RSWeightView) or not isinstance(
            w2, NVFP4RSWeightView
        ):
            raise TypeError("W4A16-RS weights must contain two RS views")
        w13.__post_init__()
        w2.__post_init__()
        pipe = self.pipe
        device = pipe.device
        if w13.payload.device != device or w2.payload.device != device:
            raise ValueError("W4A16-RS weights must be on the pipe device")
        if tuple(w13.payload.shape[:3]) != (
            pipe.E,
            (2 * self.I) // 64,
            pipe.H // 16,
        ):
            raise ValueError("W4A16-RS w13 logical shape must be (E, 2I, H)")
        if tuple(w2.payload.shape[:3]) != (
            pipe.E,
            pipe.H // 64,
            self.I // 16,
        ):
            raise ValueError("W4A16-RS w2 logical shape must be (E, H, I)")

    def bind_weights(self, weights: object) -> None:
        """Bind a same-mode, same-geometry NVFP4 bundle while idle."""
        self._require_weight_bindable()
        if weights is self._bound_weights:
            return
        if not isinstance(
            weights,
            (
                Sm90PushNvFp4Weights,
                Sm90PushNvFp4HotFoldedWeights,
                Sm90PushNvFp4DualWeights,
            ),
        ):
            raise TypeError(
                "sm90_push NVFP4 weights must be a typed bundle, "
                f"got {type(weights).__name__}"
            )
        mode = (
            weights.nvfp4_mode if isinstance(weights, Sm90PushNvFp4Weights) else "w4a8"
        )
        if mode != self.nvfp4_mode:
            raise ValueError("sm90_push NVFP4 rebound weights must preserve nvfp4_mode")

        cached = self._validated_weights.get(id(weights))
        if cached is None or cached() is not weights:
            if self.nvfp4_mode == "w4a8":
                self._w4a8_engine.validate_weights(weights)
            else:
                rs_weights = cast(Sm90PushNvFp4Weights, weights)
                w13_rs = cast(NVFP4RSWeightView, rs_weights.w13)
                w2_rs = cast(NVFP4RSWeightView, rs_weights.w2)
                self._validate_rs_weights(w13_rs, w2_rs)
            self._validated_weights[id(weights)] = weakref.ref(weights)

        if self.nvfp4_mode == "w4a8":
            self._w4a8_engine.bind_validated_weights(weights)
        else:
            rs_weights = cast(Sm90PushNvFp4Weights, weights)
            self.w13_rs = cast(NVFP4RSWeightView, rs_weights.w13)
            self.w2_rs = cast(NVFP4RSWeightView, rs_weights.w2)
        self.weights = weights
        self._bound_weights = weights

    def _init_w4a8(
        self,
        weights: (
            Sm90PushNvFp4Weights
            | Sm90PushNvFp4HotFoldedWeights
            | Sm90PushNvFp4DualWeights
        ),
    ) -> None:
        pipe = self.pipe
        max_rows = pipe.m_cap
        buffer_rows = _align(max_rows)
        scale_stride = _padded_scale_stride(max_rows, pipe.E)
        device = pipe.device
        self.a1 = torch.empty(buffer_rows, pipe.H, dtype=torch.uint8, device=device)
        self.sfa1 = torch.empty(
            (pipe.H // 128) * scale_stride + 128,
            dtype=torch.float32,
            device=device,
        )
        self.meta = torch.empty(buffer_rows, 4, dtype=torch.int32, device=device)
        self.row_expert = torch.empty(buffer_rows, dtype=torch.int32, device=device)
        engine: _W4A8PairEngine
        if isinstance(weights, Sm90PushNvFp4DualWeights):
            engine = _DualW4A8PairEngine(
                max_rows,
                pipe.E,
                scale_stride,
                weights,
                device=device,
                expected_m=pipe.token_capacity * pipe.K,
                hidden_size=pipe.H,
                n64_expected_m_per_sm=self._n64_expected_m_per_sm,
                tma_cache_capacity=self._tma_cache_capacity,
                allow_legacy_layout=self._allow_legacy_layout,
            )
        elif isinstance(weights, Sm90PushNvFp4HotFoldedWeights):
            engine = _HotFoldedW4A8PairEngine(
                max_rows,
                pipe.E,
                scale_stride,
                weights,
                device=device,
                expected_m=pipe.token_capacity * pipe.K,
                hidden_size=pipe.H,
                n64_expected_m_per_sm=self._n64_expected_m_per_sm,
                tma_cache_capacity=self._tma_cache_capacity,
                allow_legacy_layout=self._allow_legacy_layout,
            )
        else:
            engine = _LegacyW4A8PairEngine(
                max_rows,
                pipe.E,
                scale_stride,
                weights,
                device=device,
                hidden_size=pipe.H,
                expected_m=pipe.token_capacity * pipe.K,
                n64_expected_m_per_sm=self._n64_expected_m_per_sm,
                tma_cache_capacity=self._tma_cache_capacity,
                allow_legacy_layout=self._allow_legacy_layout,
            )
        self._w4a8_engine = engine
        self.execution_identity = engine.execution_identity
        self.I = engine.I
        self.h = torch.empty(
            buffer_rows, 2 * self.I, dtype=torch.bfloat16, device=device
        )
        self.a2 = torch.empty(buffer_rows, self.I, dtype=torch.uint8, device=device)
        self.sfa2 = torch.empty(
            (self.I // 128) * scale_stride + 128,
            dtype=torch.float32,
            device=device,
        )
        self.y = torch.empty(buffer_rows, pipe.H, dtype=torch.bfloat16, device=device)
        self._g = (
            None
            if pipe.config.fuse_act
            else torch.empty(
                buffer_rows, self.I, dtype=torch.bfloat16, device=pipe.device
            )
        )
        engine.warm_legacy(self)

    def _new_rs_runner(self, n: int, k: int):
        runner = create_sm90_push_nvfp4_rs_gemm_runner(
            "rs_wgmma",
            self._rs_n_tactic,
            self._rs_stages,
            self._rs_stage_k,
            use_environment=False,
        )
        size = int(
            runner.get_workspace_size(
                self._padded_max_rows,
                self.pipe.E,
                n,
                k,
            )
        )
        workspace = torch.empty(
            max(size, 1), dtype=torch.uint8, device=self.pipe.device
        )
        runner.configure_workspace(workspace)
        return runner, workspace

    def _init_rs(
        self,
        w13: NVFP4RSWeightView,
        w2: NVFP4RSWeightView,
    ) -> None:
        if not isinstance(w13, NVFP4RSWeightView) or not isinstance(
            w2, NVFP4RSWeightView
        ):
            raise TypeError("W4A16-RS weights must contain two RS views")
        pipe = self.pipe
        self._padded_max_rows = _align(pipe.m_cap + 7 * pipe.E)
        actual_rows = _align(pipe.m_cap)
        device = pipe.device
        if w13.payload.device != device or w2.payload.device != device:
            raise ValueError("W4A16-RS weights must be on the pipe device")
        if int(w13.payload.shape[0]) != pipe.E:
            raise ValueError("W4A16-RS w13 expert count does not match the pipe")
        two_i = int(w13.payload.shape[1]) * 64
        if two_i <= 0 or two_i % 256 or int(w13.payload.shape[2]) * 16 != pipe.H:
            raise ValueError("W4A16-RS w13 logical shape must be (E, 2I, H)")
        self.I = two_i // 2
        if tuple(w2.payload.shape[:3]) != (
            pipe.E,
            pipe.H // 64,
            self.I // 16,
        ):
            raise ValueError("W4A16-RS w2 logical shape must be (E, H, I)")
        self.w13_rs = w13
        self.w2_rs = w2
        self.execution_identity = ("w4a16-rs-v1", pipe.E)
        self.a1 = torch.empty(
            self._padded_max_rows, pipe.H, dtype=torch.bfloat16, device=device
        )
        self.meta = torch.empty(actual_rows, 4, dtype=torch.int32, device=device)
        self.h = torch.empty(
            self._padded_max_rows,
            2 * self.I,
            dtype=torch.bfloat16,
            device=device,
        )
        self.a2 = torch.empty(
            self._padded_max_rows,
            self.I,
            dtype=torch.bfloat16,
            device=device,
        )
        self.y = torch.empty(
            self._padded_max_rows,
            pipe.H,
            dtype=torch.bfloat16,
            device=device,
        )
        self.real_to_padded = torch.empty(actual_rows, dtype=torch.int32, device=device)
        self.rs_offsets = torch.empty(pipe.E + 1, dtype=torch.int64, device=device)
        self.rs_tile_prefix = torch.empty(pipe.E + 1, dtype=torch.int64, device=device)
        self.rs_m_dev = torch.zeros(1, dtype=torch.int32, device=device)
        self.fc1, self.fc1_workspace = self._new_rs_runner(2 * self.I, pipe.H)
        self.fc2, self.fc2_workspace = self._new_rs_runner(pipe.H, self.I)
        offsets = torch.zeros(pipe.E + 1, dtype=torch.int64, device=device)
        prefix = torch.zeros_like(offsets)
        self.fc1.grouped_run_padded(
            self.h,
            self.a1,
            w13.payload,
            w13.scales,
            w13.alpha,
            offsets,
            prefix,
            True,
        )
        self.fc2.grouped_run_padded(
            self.y,
            self.a2,
            w2.payload,
            w2.scales,
            w2.alpha,
            offsets,
            prefix,
            True,
        )
        torch.cuda.synchronize()

    def _round_compact(self) -> None:
        if self.nvfp4_mode == "w4a8":
            self.pipe.proto_compact(self.a1, self.sfa1, self.meta, self.row_expert)
            return
        self.pipe.proto_compact_bf16_padded(
            self.a1,
            self.meta,
            self.real_to_padded,
            self.rs_offsets,
            self.rs_tile_prefix,
            self.rs_m_dev,
            self._rs_n_tactic,
        )

    def _round_fc1(self) -> None:
        if self.nvfp4_mode == "w4a8":
            self._w4a8_engine.run_fc1(self)
            return
        self.fc1.grouped_run_padded(
            self.h,
            self.a1,
            self.w13_rs.payload,
            self.w13_rs.scales,
            self.w13_rs.alpha,
            self.rs_offsets,
            self.rs_tile_prefix,
            True,
        )

    def _round_activation(self) -> None:
        pipe = self.pipe
        if self.nvfp4_mode == "w4a8":
            if pipe.config.fuse_act:
                pipe.proto_silu_mul_quant(self.h, self.a2, self.sfa2, self.row_expert)
                return
            if self._g is None:
                raise RuntimeError("W4A8 unfused activation buffer is unavailable")
            pipe.module.sm90_silu_mul_gated(
                self._g, self.h, pipe._m_dev, self._g.shape[0]
            )
            pipe.module.sm90_quant_grouped(
                self.a2,
                self.sfa2,
                self._g,
                pipe._offsets,
                pipe._pad_base,
                pipe._m_dev,
                pipe._p_dev,
                self.row_expert,
                self._g.shape[0],
            )
            return
        pipe.module.sm90_silu_mul_gated(
            self.a2,
            self.h,
            self.rs_m_dev,
            self.a2.shape[0],
        )

    def _round_activation_stage(self) -> str | None:
        return "act_quant" if self.nvfp4_mode == "w4a8" else "activation"

    def _round_fc2(self) -> None:
        if self.nvfp4_mode == "w4a8":
            self._w4a8_engine.run_fc2(self)
            return
        self.fc2.grouped_run_padded(
            self.y,
            self.a2,
            self.w2_rs.payload,
            self.w2_rs.scales,
            self.w2_rs.alpha,
            self.rs_offsets,
            self.rs_tile_prefix,
            True,
        )

    def _round_combine(self) -> None:
        if self.nvfp4_mode == "w4a8":
            self.pipe.proto_combine(self.y, self.meta)
            return
        self.pipe.proto_combine_mapped(self.y, self.meta, self.real_to_padded)

    def _release_resources(self) -> None:
        super()._release_resources()
        engine = getattr(self, "_w4a8_engine", None)
        if engine is not None:
            engine.destroy()
        self._w4a8_engine = None
        self.fc1 = None
        self.fc2 = None
        self.fc1_workspace = None
        self.fc2_workspace = None
        self.w13_rs = None
        self.w2_rs = None
        self.weights = None
        self.a1 = None
        self.sfa1 = None
        self.meta = None
        self.row_expert = None
        self.h = None
        self.a2 = None
        self.sfa2 = None
        self.y = None
        self._g = None
        self.real_to_padded = None
        self.rs_offsets = None
        self.rs_tile_prefix = None
        self.rs_m_dev = None


__all__ = ["Sm90PushNvFp4MoERunner"]
