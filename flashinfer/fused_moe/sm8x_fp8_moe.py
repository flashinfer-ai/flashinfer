# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The CUTLASS FP8-family fused-MoE runners' semantics over the SM8x grouped GEMMs, on the
architectures where the CUTLASS fused-MoE module has no kernel for the quantization pair.

``_CutlassRunnerBase`` delegates to these: the cuDNN grouped-GEMM pipeline (``moe_sort`` ->
``moe_permute`` -> GEMM1 -> activation -> GEMM2 -> ``moe_unpermute``) with the
``sm8x_fp8_dense`` grouped GEMMs, CUTLASS's activation set and its scale conventions.
"""

import math
from typing import ClassVar, Dict, List, Optional, Union

import torch
import torch.nn.functional as F

from ..gemm.sm8x_fp4_dense import _unswizzle
from ..gemm.sm8x_fp8_dense import (
    sm8x_grouped_mm_fp4,
    sm8x_grouped_mm_fp8,
    sm8x_grouped_mm_mxfp8,
)
from .api import ActivationType, MoEConfig
from .runners import (
    _check_moe_sort_limits,
    _CudnnGroupedGemmBlockScaleRunnerBase,
    _CudnnGroupedGemmRunnerBase,
    _GroupedGemmLayout,
    _GroupedGemmWorkspace,
)

_Param = Union[torch.Tensor, float]


def _per_row(
    params: Dict[str, Optional[torch.Tensor]],
    name: str,
    row_expert: torch.Tensor,
    default: float,
) -> _Param:
    """``params[name]`` (``[num_experts]`` float32) of every row's expert, or ``default``."""
    value = params.get(name)
    if value is None:
        return default
    return value.index_select(0, row_expert)[:, None]


def cutlass_activation(
    x: torch.Tensor,
    activation_type: ActivationType,
    params: Dict[str, Optional[torch.Tensor]],
    row_expert: torch.Tensor,
) -> torch.Tensor:
    """The CUTLASS fused-MoE activation of float32 GEMM1 rows ``x``: ``[rows, I]``, or
    ``[rows, 2I]`` as ``[up | gate]`` for the gated ones. ``params`` are the runner's
    optional per-expert tensors (``_cutlass_activation_params``) and ``row_expert``
    every row's expert; an absent tensor takes the kernel's default."""
    t = activation_type
    if t == ActivationType.Identity:
        return x
    if t == ActivationType.Gelu:
        return F.gelu(x)
    if t == ActivationType.Relu:
        return F.relu(x)
    if t == ActivationType.Silu:
        return F.silu(x)
    if t == ActivationType.Relu2:
        return F.relu(x).square()
    inter = x.shape[1] // 2
    up, gate = x[:, :inter], x[:, inter:]
    if t == ActivationType.Geglu:
        return F.gelu(gate) * up
    if t == ActivationType.GegluTanh:
        return F.gelu(gate, approximate="tanh") * up
    if t == ActivationType.Swiglu:
        if all(
            params.get(name) is None
            for name in ("swiglu_alpha", "swiglu_beta", "swiglu_limit")
        ):
            return F.silu(gate) * up
        # Any of the three present selects the SwigluBias adaptor.
        alpha = _per_row(params, "swiglu_alpha", row_expert, 1.0)
        beta = _per_row(params, "swiglu_beta", row_expert, 0.0)
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        gate = gate.clamp(max=limit)
        return gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + beta)
    if t == ActivationType.SwigluStep:
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        return F.silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
    if t == ActivationType.Situ:
        beta = _per_row(params, "situ_beta", row_expert, 4.0)
        linear_beta = _per_row(params, "situ_linear_beta", row_expert, 25.0)
        act = torch.tanh(gate / beta) * torch.sigmoid(gate) * beta
        return act * (torch.tanh(up / linear_beta) * linear_beta)
    if t == ActivationType.PowLU:
        # ``m`` rides in the swiglu_alpha slot, the clamp in swiglu_limit.
        m = _per_row(params, "swiglu_alpha", row_expert, 2.5)
        limit = _per_row(params, "swiglu_limit", row_expert, math.inf)
        pos = gate.clamp(min=0)
        act = torch.where(gate > 0, pos.pow(m / (pos.sqrt() + 1)), gate)
        act = act * torch.sigmoid(gate)
        return act.clamp(max=limit) * up.clamp(-limit, limit)
    raise NotImplementedError(f"CUTLASS activation {t!r} has no SM8x implementation.")


class _Sm8xCutlassMoEBase(_CudnnGroupedGemmRunnerBase):
    """CUTLASS semantics on the cuDNN pipeline: GEMM1's per-expert alpha and the
    activation in float32, then the GEMM2 operand. Built and driven by the CUTLASS
    runner, which checked the configuration and packed the inputs; ``run`` maps
    its packed inputs onto this pipeline's."""

    # Packed-input index of GEMM1's [E] float32 per-expert alpha, if any.
    _gemm1_alpha_input: ClassVar[Optional[int]] = None
    _intermediate_alignment: ClassVar[int]

    def __init__(self, config: MoEConfig, device: torch.device) -> None:
        super().__init__(config, device)
        self._activation_params: Dict[str, Optional[torch.Tensor]] = {}

    def _check_support(self) -> None:
        if self._device_arch not in self._supported_archs:
            raise RuntimeError(
                f"{type(self).__name__} does not support SM{self._device_arch}; "
                f"supported architectures are {self._supported_archs}."
            )
        _check_moe_sort_limits(type(self).__name__, self.config)
        intermediate_size = self.config.experts.intermediate_size
        if intermediate_size % self._intermediate_alignment:
            raise NotImplementedError(
                f"{type(self).__name__} requires intermediate_size divisible by "
                f"{self._intermediate_alignment}, got {intermediate_size}."
            )

    def _build(self) -> None:
        from .cute_dsl.moe_utils import _get_moe_utils_module

        _get_moe_utils_module()

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """This pipeline's packed inputs from the CUTLASS runner's (output first)."""
        raise NotImplementedError

    def run(
        self,
        inputs: List[torch.Tensor],
        activation_params: Dict[str, Optional[torch.Tensor]],
    ) -> torch.Tensor:
        """Run the CUTLASS runner's packed ``inputs`` into their output ``inputs[0]``."""
        out, hidden_states = inputs[0], inputs[1]
        if hidden_states.shape[0] == 0:
            return out
        self._validate_hidden_states(hidden_states)
        self._activation_params = activation_params
        self.forward(self._body_inputs(inputs))
        return out

    def _activate(
        self,
        gemm1_out: torch.Tensor,
        layout: _GroupedGemmLayout,
        workspace: _GroupedGemmWorkspace,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        row_expert = layout.row_expert(gemm1_out.shape[0])
        x = gemm1_out.float()
        if self._gemm1_alpha_input is not None:
            x.mul_(inputs[self._gemm1_alpha_input].index_select(0, row_expert)[:, None])
        y = cutlass_activation(
            x, self.config.activation.type, self._activation_params, row_expert
        )
        return self._gemm2_operand(y, workspace, inputs)

    def _gemm2_operand(
        self,
        y: torch.Tensor,
        workspace: _GroupedGemmWorkspace,
        inputs: List[torch.Tensor],
    ) -> torch.Tensor:
        """GEMM2's operand from the float32 activation ``y``."""
        return workspace.intermediate.copy_(y)


class Sm8xCutlassFp8PerTensorMoE(_Sm8xCutlassMoEBase):
    """``CutlassFp8PerTensorRunner`` on SM80 / SM86 / SM87. Inputs:
    ``[hidden, ids, weights, w1, w2, fc1_dequant, fc2_act_quant, fc2_dequant, out]``."""

    backend_key = "cutlass_fp8_per_tensor"
    _supported_archs = (80, 86, 87)
    _x_dtype = torch.float8_e4m3fn
    _weight_dtype = torch.float8_e4m3fn
    _expected_num_inputs = 8
    _gemm1_alpha_input = 5
    _fc2_act_quant_input = 6
    _fc2_dequant_input = 7
    _intermediate_dtype = torch.float8_e4m3fn
    _intermediate_alignment = 16  # GEMM2's K

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return [*inputs[1:9], inputs[0]]

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        return sm8x_grouped_mm_fp8(
            workspace.permuted_hidden_states,
            inputs[3],
            layout.m_indptr,
            None,
            torch.bfloat16,
            workspace.gemm1_out,
        )

    def _gemm2_operand(self, y, workspace, inputs) -> torch.Tensor:
        fp8_max = torch.finfo(torch.float8_e4m3fn).max
        return (
            y.mul_(inputs[self._fc2_act_quant_input])
            .clamp_(-fp8_max, fp8_max)
            .to(torch.float8_e4m3fn)
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        return sm8x_grouped_mm_fp8(
            intermediate, inputs[4], layout.m_indptr, None, torch.bfloat16, None
        )


class _Sm8xCutlassMxfp8MoEBase(
    _Sm8xCutlassMoEBase, _CudnnGroupedGemmBlockScaleRunnerBase
):
    """MXFP8 activations: E4M3 rows with UE8M0 32-element block scales, the
    intermediate MXFP8-quantized for GEMM2. Inputs: ``[hidden, ids, weights, w1, w2,
    w1_scale, w2_scale, fc1_input_scale, fc2_input_scale, hidden_states_scale, out]``
    with token-major ``hidden_states_scale``."""

    _supported_archs = (80, 86, 87, 89)
    _x_dtype = torch.float8_e4m3fn
    _expected_num_inputs = 10
    _block_size = 32
    _block_scale_dtype = torch.uint8
    _intermediate_alignment = 128  # whole 128x4 scale tiles

    def _body_inputs(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        act_sf = inputs[10]
        if act_sf.ndim == 1:  # QuantConfig(swizzled_scale_factors=True)
            num_tokens, hidden_size = inputs[1].shape
            act_sf = _unswizzle(act_sf, num_tokens, hidden_size // 32)
        return [*inputs[1:10], act_sf.view(torch.uint8), inputs[0]]


class Sm8xCutlassMxfp8MoE(_Sm8xCutlassMxfp8MoEBase):
    """``CutlassMxfp8Runner`` on SM8x: E4M3 weights with int32-packed per-expert
    128x4-swizzled UE8M0 scales; ``fc1_input_scale`` and ``fc2_input_scale`` are the
    two GEMMs' per-expert alphas."""

    backend_key = "cutlass_mxfp8"
    _weight_dtype = torch.float8_e4m3fn
    _gemm1_alpha_input = 7
    _fc2_dequant_input = 8

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_mxfp8(
            workspace.permuted_hidden_states,
            inputs[3],
            workspace.permuted_hidden_states_scale,
            inputs[5].view(torch.uint8),
            layout.m_indptr,
            torch.bfloat16,
            workspace.gemm1_out,
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _quantize_mxfp8_rows

        quantized, scale = _quantize_mxfp8_rows(intermediate)
        return sm8x_grouped_mm_mxfp8(
            quantized,
            inputs[4],
            scale,
            inputs[6].view(torch.uint8),
            layout.m_indptr,
            torch.bfloat16,
            None,
        )


class Sm8xCutlassMxfp8Mxfp4MoE(_Sm8xCutlassMxfp8MoEBase):
    """``CutlassMxfp8Mxfp4Runner`` on SM8x: packed E2M1 weights with per-expert
    128x4-swizzled UE8M0 scales. CUTLASS reads neither input scale."""

    backend_key = "cutlass_mxfp8_mxfp4"
    _weight_dtype = torch.int64

    def _gemm1(self, inputs, layout, workspace, tactic) -> torch.Tensor:
        assert workspace.permuted_hidden_states_scale is not None
        return sm8x_grouped_mm_fp4(
            workspace.permuted_hidden_states,
            inputs[3].view(torch.uint8),
            workspace.permuted_hidden_states_scale,
            inputs[5],
            layout.m_indptr,
            None,
            torch.bfloat16,
            workspace.gemm1_out,
            32,
        )

    def _gemm2(self, inputs, layout, workspace, intermediate, tactic) -> torch.Tensor:
        from .prepare import _quantize_mxfp8_rows

        quantized, scale = _quantize_mxfp8_rows(intermediate)
        return sm8x_grouped_mm_fp4(
            quantized,
            inputs[4].view(torch.uint8),
            scale,
            inputs[6],
            layout.m_indptr,
            None,
            torch.bfloat16,
            None,
            32,
        )
