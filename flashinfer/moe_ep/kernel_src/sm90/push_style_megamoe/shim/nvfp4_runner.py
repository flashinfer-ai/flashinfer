"""SM90 push runners for NVFP4 W4A8 and W4A16-RS weights."""

from __future__ import annotations

import weakref
from typing import Any, cast

import torch

from ......fused_moe.sm90_nvfp4_repack import (
    NVFP4RSWeightView,
    NVFP4SM90WeightViewV3,
)
from .nvfp4_rs_gemm import create_sm90_push_nvfp4_rs_gemm_runner
from .nvfp4_w4a8_gemm import (
    _W4A8ScheduleWorkspace,
    create_sm90_push_nvfp4_w4a8_gemm,
)
from .nvfp4_weights import Sm90PushNvFp4Weights
from .protocol import Sm90PushCombine, Sm90PushPipe, _run_guarded_phase
from .runner import Sm90PushMoERunner


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
        weights: NVFP4SM90WeightViewV3,
        *,
        device: torch.device,
        shared_schedule_workspace: _W4A8ScheduleWorkspace | None = None,
        counter_bank: int = 0,
    ) -> None:
        if not isinstance(weights, NVFP4SM90WeightViewV3):
            raise TypeError("W4A8 weights must be NVFP4SM90WeightViewV3")
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
        experts, self.n, self.k = weights.manifest.logical_shape
        if experts != self.total_experts:
            raise ValueError("W4A8 weight expert count does not match the pipe")
        if tuple(weights.manifest.expert_mapping) != tuple(range(total_experts)):
            raise ValueError("W4A8 weights must map every local expert in order")
        if self.k % 128:
            raise ValueError("W4A8 activation K must be divisible by 128")
        self.runner = create_sm90_push_nvfp4_w4a8_gemm(
            self.max_rows,
            weights,
            total_experts=self.total_experts,
            shared_schedule_workspace=shared_schedule_workspace,
            counter_bank=counter_bank,
        )

    def _validate_weight_view(self, weights: NVFP4SM90WeightViewV3) -> None:
        if not isinstance(weights, NVFP4SM90WeightViewV3):
            raise TypeError("W4A8 weights must be NVFP4SM90WeightViewV3")
        weights.verify_checksums()
        current = self.runner.weight_view
        if weights.packed_e2m1.device != current.packed_e2m1.device:
            raise ValueError("W4A8 rebound weights must remain on the runner device")
        if tuple(weights.manifest.logical_shape) != (
            self.total_experts,
            self.n,
            self.k,
        ):
            raise ValueError("W4A8 rebound weights must preserve the logical shape")
        if weights.manifest.padded_shape != current.manifest.padded_shape:
            raise ValueError("W4A8 rebound weights must preserve the padded shape")
        if weights.manifest.group_size != current.manifest.group_size:
            raise ValueError("W4A8 rebound weights must preserve group_size")
        if weights.manifest.residual_scheme != current.manifest.residual_scheme:
            raise ValueError("W4A8 rebound weights must preserve residual_scheme")
        if weights.manifest.expert_mapping != current.manifest.expert_mapping:
            raise ValueError("W4A8 rebound weights must preserve expert_mapping")

    def _bind_weight_view(self, weights: NVFP4SM90WeightViewV3) -> None:
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


class Sm90PushNvFp4MoERunner(Sm90PushMoERunner):
    """Two-phase SM90 push runner selected by a typed NVFP4 weight bundle."""

    # w4a8 mode binds _W4A8LayerRunner; w4a16_rs binds the FFI RS runner object.
    fc1: Any
    fc2: Any

    def __init__(
        self,
        pipe: Sm90PushPipe,
        weights: Sm90PushNvFp4Weights,
        *,
        rs_n_tactic: int = 64,
        rs_stages: int = 3,
        rs_stage_k: int = 64,
    ) -> None:
        if not isinstance(weights, Sm90PushNvFp4Weights):
            raise TypeError("weights must be Sm90PushNvFp4Weights")
        self._init_round_state(pipe)
        self.weights = weights
        self.nvfp4_mode = weights.nvfp4_mode
        self._rs_n_tactic = int(rs_n_tactic)
        self._rs_stages = int(rs_stages)
        self._rs_stage_k = int(rs_stage_k)

        def _local_init() -> None:
            if pipe.config.fuse_fc1_epilogue:
                raise ValueError("SM90 push NVFP4 does not support fused FC1")
            if self.nvfp4_mode == "w4a8":
                self._init_w4a8(
                    cast(NVFP4SM90WeightViewV3, weights.w13),
                    cast(NVFP4SM90WeightViewV3, weights.w2),
                )
            else:
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

        _run_guarded_phase(
            pipe._comm,
            getattr(pipe, "rank", 0),
            "nvfp4-weights+gemm-resources",
            _local_init,
        )
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

    def bind_weights(self, weights: Sm90PushNvFp4Weights) -> None:
        """Bind a same-mode, same-geometry NVFP4 bundle while idle."""
        self._require_weight_bindable()
        if weights is self._bound_weights:
            return
        if not isinstance(weights, Sm90PushNvFp4Weights):
            raise TypeError(
                "sm90_push NVFP4 weights must be an Sm90PushNvFp4Weights bundle, "
                f"got {type(weights).__name__}"
            )
        if weights.nvfp4_mode != self.nvfp4_mode:
            raise ValueError("sm90_push NVFP4 rebound weights must preserve nvfp4_mode")

        cached = self._validated_weights.get(id(weights))
        if cached is None or cached() is not weights:
            if self.nvfp4_mode == "w4a8":
                w13_w4a8 = cast(NVFP4SM90WeightViewV3, weights.w13)
                w2_w4a8 = cast(NVFP4SM90WeightViewV3, weights.w2)
                self.fc1._validate_weight_view(w13_w4a8)
                self.fc2._validate_weight_view(w2_w4a8)
            else:
                w13_rs = cast(NVFP4RSWeightView, weights.w13)
                w2_rs = cast(NVFP4RSWeightView, weights.w2)
                self._validate_rs_weights(w13_rs, w2_rs)
            self._validated_weights[id(weights)] = weakref.ref(weights)

        if self.nvfp4_mode == "w4a8":
            self.fc1._bind_weight_view(cast(NVFP4SM90WeightViewV3, weights.w13))
            self.fc2._bind_weight_view(cast(NVFP4SM90WeightViewV3, weights.w2))
        else:
            self.w13_rs = cast(NVFP4RSWeightView, weights.w13)
            self.w2_rs = cast(NVFP4RSWeightView, weights.w2)
        self.weights = weights
        self._bound_weights = weights

    def _init_w4a8(
        self,
        w13: NVFP4SM90WeightViewV3,
        w2: NVFP4SM90WeightViewV3,
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
        self.fc1 = _W4A8LayerRunner(
            max_rows,
            pipe.E,
            scale_stride,
            w13,
            device=device,
        )
        if self.fc1.n <= 0 or self.fc1.n % 2 or self.fc1.k != pipe.H:
            raise ValueError("W4A8 w13 logical shape must be (E, 2I, H)")
        self.I = self.fc1.n // 2
        self.fc2 = _W4A8LayerRunner(
            max_rows,
            pipe.E,
            scale_stride,
            w2,
            device=device,
            shared_schedule_workspace=self.fc1.runner.schedule_workspace,
            counter_bank=1,
        )
        if (self.fc2.n, self.fc2.k) != (pipe.H, self.I):
            raise ValueError("W4A8 w2 logical shape must be (E, H, I)")
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
        offsets = torch.zeros(pipe.E + 1, dtype=torch.int64, device=device)
        self.fc1.run(
            self.h,
            self.a1,
            self.sfa1,
            offsets,
            prepare_schedule=True,
        )
        self.fc2.run(
            self.y,
            self.a2,
            self.sfa2,
            offsets,
            prepare_schedule=False,
        )
        torch.cuda.synchronize()

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
            self.fc1.run(
                self.h,
                self.a1,
                self.sfa1,
                self.pipe._offsets,
                prepare_schedule=True,
            )
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
            self.fc2.run(
                self.y,
                self.a2,
                self.sfa2,
                self.pipe._offsets,
                prepare_schedule=False,
            )
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
