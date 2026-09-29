"""Local CuTe-DSL MegaMOE FC12 launch adapters.

The adapters deliberately own only FC1/SwiGLU/FC2.  Routing and token movement
remain in the unified runner so they can be shared by direct and EP callers.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch

from .api import QuantConfig, QuantFormat, SwiGLU


def prepare_megamoe_fc12_weights(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    *,
    quant: QuantConfig,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    activation=None,
    device=None,
) -> dict[str, torch.Tensor]:
    """Prepare canonical weights for a local MegaMOE FC12 kernel."""
    del device
    if not isinstance(activation or SwiGLU(), SwiGLU):
        raise NotImplementedError("MegaMOE FC12 currently implements SwiGLU only.")
    expected_w1 = (num_local_experts, 2 * intermediate_size, hidden_size)
    expected_w2 = (num_local_experts, hidden_size, intermediate_size)
    if tuple(w1_bf16.shape) != expected_w1 or tuple(w2_bf16.shape) != expected_w2:
        raise ValueError(
            "MegaMOE FC12 canonical weights must have shapes "
            f"{expected_w1} and {expected_w2}; got {tuple(w1_bf16.shape)} and "
            f"{tuple(w2_bf16.shape)}."
        )
    pair = quant.pair
    if pair == (QuantFormat.MXFP8, QuantFormat.BF16):
        # Keep the architecture-specific quantizer at the kernel boundary.
        # Its output already has the FC12 K-major data and scale layouts.
        from ..moe_ep.backends.mega.kernel.sm100.bf16_mxfp8_bf16_cutedsl.weights import (  # noqa: E501
            preprocess_mega_weights,
        )

        fc1, fc2 = preprocess_mega_weights(
            SimpleNamespace(w13=w1_bf16, w2=w2_bf16),
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            kind="bf16_mxfp8_e4m3",
        )
        return {
            "fc1_weight": fc1[0],
            "fc1_weight_sf": fc1[1],
            "fc2_weight": fc2[0],
            "fc2_weight_sf": fc2[1],
        }
    if pair != (QuantFormat.BF16, QuantFormat.BF16):
        raise NotImplementedError(
            "MegaMOE FC12 weight preparation for "
            f"weight={pair[0].name}, activation={pair[1].name} "
            "is not available in this build."
        )
    if w1_bf16.dtype is not torch.bfloat16 or w2_bf16.dtype is not torch.bfloat16:
        raise TypeError("BF16 MegaMOE FC12 requires bfloat16 canonical weights.")
    if intermediate_size % 32:
        raise ValueError(
            "BF16 MegaMOE FC12 requires intermediate_size divisible by 32."
        )

    # FC12's WGMMA layout alternates gate/up groups of 32 along N.
    fc1 = torch.empty_like(w1_bf16)
    fc1_interleaved = fc1.view(
        num_local_experts, intermediate_size // 32, 2, 32, hidden_size
    )
    fc1_interleaved[:, :, 0].copy_(
        w1_bf16[:, :intermediate_size].view(
            num_local_experts, intermediate_size // 32, 32, hidden_size
        )
    )
    fc1_interleaved[:, :, 1].copy_(
        w1_bf16[:, intermediate_size:].view(
            num_local_experts, intermediate_size // 32, 32, hidden_size
        )
    )
    return {
        "fc1_weight": fc1.transpose(1, 2),
        "fc2_weight": w2_bf16.transpose(1, 2),
    }


# TODO: Should this implementation live under moe_ep directly?
def _enable_fc12_sources() -> None:
    """Make the verbatim CuTe-DSL FC12 source package importable lazily."""
    source_root = (
        Path(__file__).resolve().parents[1]
        / "moe_ep"
        / "kernel_src"
        / "cutedsl_megamoe"
        / "src"
    )
    for path in (source_root, source_root / "moe_nvfp4_swapab"):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


@dataclass
class Bf16Fc12Inputs:
    activation: torch.Tensor
    fc1_weight: torch.Tensor
    fc2_weight: torch.Tensor
    output: torch.Tensor
    expert_token_sizes: torch.Tensor
    fc1_weight_sf: Optional[torch.Tensor] = None
    fc2_weight_sf: Optional[torch.Tensor] = None


class Bf16Fc12Launcher:
    """Compile and launch the SM100 BF16 standalone FC12 kernel."""

    def __init__(
        self, num_experts: int, max_rows: int, hidden: int, intermediate: int
    ) -> None:
        if hidden % 32 or intermediate % 64:
            raise ValueError(
                "BF16 MegaMOE FC12 requires hidden divisible by 32 and "
                "intermediate_size divisible by 64."
            )
        self.num_experts = num_experts
        self.max_rows = max_rows
        self.hidden = hidden
        self.intermediate = intermediate
        self._compiled = None
        self._workspace: Optional[torch.Tensor] = None
        self._fc1_output: Optional[torch.Tensor] = None
        self._fc1_done_counter: Optional[torch.Tensor] = None
        self._unit_scores: Optional[torch.Tensor] = None

    @staticmethod
    def _to_cute(tensor: torch.Tensor, *, assumed_align: int = 16):
        import cutlass.torch as cutlass_torch

        return cutlass_torch.from_dlpack(
            tensor, assumed_align=assumed_align
        ).mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(tensor))

    def _runtime_kwargs(
        self,
        inputs: Bf16Fc12Inputs,
        fc1_output: torch.Tensor,
        fc1_done_counter: torch.Tensor,
        unit_scores: torch.Tensor,
    ) -> dict:
        import cuda.bindings.driver as cuda

        return {
            "activation": self._to_cute(inputs.activation),
            "fc1_weight": self._to_cute(inputs.fc1_weight),
            "fc1_output": self._to_cute(fc1_output),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc2_output": self._to_cute(inputs.output),
            "topk_scores": self._to_cute(unit_scores),
            "fc1_done_counter": self._to_cute(fc1_done_counter, assumed_align=4),
            "expert_token_sizes": self._to_cute(
                inputs.expert_token_sizes, assumed_align=4
            ),
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def _compile(self, inputs: Bf16Fc12Inputs) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMOE FC12 must be warmed up before CUDA Graph capture."
            )
        _enable_fc12_sources()
        from cutlass import BFloat16
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils
        from moe_bf16_glu.kernel_bf16_glu_fc12 import Sm100SwigluBf16Fc12Kernel

        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)
        kernel = Sm100SwigluBf16Fc12Kernel(
            mma_tiler_mnk=(256, 256, 64),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            group_hint=max_active_clusters,
            token_padding_block=64,
            load_balance_mode="static",
            static_expert_shape=(self.num_experts, 2 * self.intermediate, self.hidden),
            force_static_sched=True,
            ab_dtype=BFloat16,
            apply_topk_in_fc1=False,
        )
        workspace_bytes = kernel.get_workspace_size_in_bytes(
            inputs.activation, inputs.fc1_weight
        )
        workspace = torch.zeros(
            workspace_bytes, dtype=torch.uint8, device=inputs.activation.device
        )
        fc1_bytes = self.max_rows * self.intermediate * torch.bfloat16.itemsize
        fc1_output = (
            workspace[:fc1_bytes]
            .view(torch.bfloat16)
            .reshape(self.max_rows, self.intermediate)
        )
        counter_slots = (self.max_rows + 255) // 256 + self.num_experts
        fc1_done_counter = workspace[
            fc1_bytes : fc1_bytes + counter_slots * torch.int32.itemsize
        ].view(torch.int32)
        unit_scores = torch.ones(
            self.max_rows, dtype=torch.float32, device=inputs.activation.device
        )
        self._compiled = cute.compile(
            kernel,
            **self._runtime_kwargs(inputs, fc1_output, fc1_done_counter, unit_scores),
            max_active_clusters=max_active_clusters,
        )
        self._workspace = workspace
        self._fc1_output = fc1_output
        self._fc1_done_counter = fc1_done_counter
        self._unit_scores = unit_scores

    def run(self, inputs: Bf16Fc12Inputs) -> None:
        expected = (self.max_rows, self.hidden)
        if (
            tuple(inputs.activation.shape) != expected
            or tuple(inputs.output.shape) != expected
            or inputs.activation.dtype is not torch.bfloat16
            or inputs.output.dtype is not torch.bfloat16
        ):
            raise ValueError(
                "BF16 FC12 activation/output must be bfloat16 "
                f"[{self.max_rows}, {self.hidden}]."
            )
        if self._compiled is None:
            self._compile(inputs)
        assert self._workspace is not None
        assert self._fc1_output is not None
        assert self._fc1_done_counter is not None
        assert self._unit_scores is not None
        compiled = self._compiled
        assert compiled is not None
        self._workspace.zero_()
        compiled(
            **self._runtime_kwargs(
                inputs,
                self._fc1_output,
                self._fc1_done_counter,
                self._unit_scores,
            )
        )


class Bf16Mxfp8Fc12Launcher(Bf16Fc12Launcher):
    """SM100 FC12 launcher for BF16 activations and E4M3 MXFP8 weights."""

    def _runtime_kwargs(
        self,
        inputs: Bf16Fc12Inputs,
        fc1_output: torch.Tensor,
        fc1_done_counter: torch.Tensor,
        unit_scores: torch.Tensor,
    ) -> dict:
        import cuda.bindings.driver as cuda

        return {
            "activation": self._to_cute(inputs.activation),
            "fc1_weight": self._to_cute(inputs.fc1_weight),
            "fc1_weight_sf": self._to_cute(inputs.fc1_weight_sf),
            "fc1_output": self._to_cute(fc1_output),
            "fc2_weight": self._to_cute(inputs.fc2_weight),
            "fc2_weight_sf": self._to_cute(inputs.fc2_weight_sf),
            "fc2_output": self._to_cute(inputs.output.reshape(-1, 1, self.hidden)),
            "topk_scores": self._to_cute(unit_scores),
            "fc1_done_counter": self._to_cute(fc1_done_counter, assumed_align=4),
            "expert_token_sizes": self._to_cute(
                inputs.expert_token_sizes, assumed_align=4
            ),
            "stream": cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        }

    def _compile(self, inputs: Bf16Fc12Inputs) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "MegaMOE FC12 must be warmed up before CUDA Graph capture."
            )
        _enable_fc12_sources()
        import cutlass.cute as cute
        import cutlass.utils as cutlass_utils
        from moe_mxfp8_bf16_glu.kernel_mxfp8_bf16_glu_fc12 import (
            Sm100SwapABMxfp8Bf16Fc12Kernel,
        )

        max_active_clusters = cutlass_utils.HardwareInfo().get_max_active_clusters(2)
        kernel = Sm100SwapABMxfp8Bf16Fc12Kernel(
            mma_tiler_mnk=(256, 128, 128),
            cluster_shape_mnk=(2, 1, 1),
            use_2cta_instrs=True,
            group_hint=max_active_clusters,
            token_padding_block=64,
            load_balance_mode="static",
            static_expert_shape=(self.num_experts, 2 * self.intermediate, self.hidden),
            force_static_sched=True,
            transform_buffer="tmem",
            accumulator_overlap=False,
            transform_k_tile=128,
            epi_flag_batch=(1, 1),
            apply_topk_in_fc1=False,
        )
        workspace_bytes = kernel.get_workspace_size_in_bytes(
            inputs.activation, inputs.fc1_weight
        )
        workspace = torch.zeros(
            workspace_bytes, dtype=torch.uint8, device=inputs.activation.device
        )
        fc1_bytes = self.max_rows * self.intermediate * torch.bfloat16.itemsize
        fc1_output = (
            workspace[:fc1_bytes]
            .view(torch.bfloat16)
            .reshape(self.max_rows, self.intermediate)
        )
        counter_slots = (self.max_rows + 127) // 128 + self.num_experts
        fc1_done_counter = workspace[
            fc1_bytes : fc1_bytes + counter_slots * torch.int32.itemsize
        ].view(torch.int32)
        unit_scores = torch.ones(
            self.max_rows, dtype=torch.float32, device=inputs.activation.device
        )
        self._compiled = cute.compile(
            kernel,
            **self._runtime_kwargs(inputs, fc1_output, fc1_done_counter, unit_scores),
            max_active_clusters=max_active_clusters,
        )
        self._workspace = workspace
        self._fc1_output = fc1_output
        self._fc1_done_counter = fc1_done_counter
        self._unit_scores = unit_scores
