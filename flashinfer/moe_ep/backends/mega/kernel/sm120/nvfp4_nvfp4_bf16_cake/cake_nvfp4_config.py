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

"""Configuration for the Cake SM120 NVFP4 x NVFP4 (W4A4) persistent MegaMoE backend."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from ..mxfp4_mxfp8_bf16_cake.cake_config import (
    RETURN_CODECS,
    Sm120CakeMegaMoeConfigBase,
)

if TYPE_CHECKING:
    import torch

_SCALE_FIELDS = ("fc1_alpha", "fc2_alpha", "fc1_norm_const")


@dataclass
class Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(Sm120CakeMegaMoeConfigBase):
    """Static model parameters of the Cake persistent MegaMoE kernel with NVFP4 activations.

    The device code, transport, shared-expert, transport-table and tuning options are those of
    :class:`~flashinfer.moe_ep.Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig`; the numerics and the
    parameter names follow FlashInfer's other NVFP4 W4A4 MegaMoE backends
    (``sm100_nvfp4_nvfp4_bf16_cutedsl``, ``sm107_nvfp4_nvfp4_bf16_cutedsl``):

    * activations and routed weights are NVFP4: E2M1 values with one E4M3 scale per 16 values;
      both GEMMs run at the FP4 rate (``mma.sync kind::mxf4nvf4``);
    * ``fc1 = (x_q @ w13_q) * fc1_alpha[e]``, then the clamp and the gated activation, times the
      routing weight; that intermediate is requantized to NVFP4 per 16 values with
      ``scale = e4m3((amax / 6) * fc1_norm_const[e])`` and codes
      ``e2m1(v * fc1_norm_const[e] / scale)``; ``fc2 = (h_q @ w2_q) * fc2_alpha[e]``;
    * ``fc1_alpha``, ``fc2_alpha`` and ``fc1_norm_const`` are optional 1-D float32 CUDA tensors
      ``[num_local_experts]`` (``None`` means 1.0; the length is checked against
      ``num_experts // world_size`` when the layer is built); per-call values in
      :class:`~flashinfer.moe_ep.MoEEpTensors` override them.  With ``MegaConfig.quantize_input``
      the BF16 input is quantized as ``x * input_norm_const``, so ``fc1_alpha`` carries the
      ``1 / input_norm_const`` (and any weight global scale) and ``fc2_alpha`` the
      ``1 / fc1_norm_const``.
    * ``gate_up_clamp`` (alias ``activation_clamp``): clamp ``gate <= c``, ``-c <= up <= c``
      before the activation (``None``: unclamped).  The kernel compiles the clamp, so it must be a
      whole number.  ``swiglu_limit`` (the MXFP8 backend's name, 0 = unclamped) is accepted as a
      further alias; aliases that are set must agree, otherwise the config raises.
    * ``apply_routing_weights_before_fc2`` must be ``True`` (the routing weight multiplies the
      intermediate before its requantization, as the sibling backends' default).
    * ``combine_dtype``: ``"bf16"`` (default, lossless BF16 return codec) or ``"int8"`` (int8 with
      one BF16 scale per 64 values, see ``return_codec`` of the MXFP8 backend).  ``return_codec``
      is accepted as an alias and must agree when both are set.

    After construction the aliases hold the resolved values (``swiglu_limit`` the clamp or 0,
    ``return_codec`` = ``combine_dtype``), so ``dataclasses.replace`` sees a consistent set.

    The in-kernel FP8 shared expert (``shared_expert="fp8_in_kernel"``) keeps its FP8 e4m3
    weights and FP8 math; its input rows are the MXFP8 requantization of the NVFP4 input and its
    output carries no global scale.  NVFP4 needs ``hidden_size % 1024 == 0`` (and
    ``intermediate_size % 256 == 0`` with the shared expert).
    """

    # aliases of gate_up_clamp / combine_dtype (fields of the shared base); None = not set
    swiglu_limit: int | None = None  # type: ignore[assignment]
    kernel_name: str = "sm120_nvfp4_nvfp4_bf16_cake"
    return_codec: str | None = None  # type: ignore[assignment]
    gate_up_clamp: float | None = None
    activation_clamp: float | None = None
    apply_routing_weights_before_fc2: bool = True
    combine_dtype: str | None = None  # "bf16" | "int8"
    input_norm_const: float = 1.0
    fc1_alpha: Optional["torch.Tensor"] = field(default=None, repr=False)
    fc2_alpha: Optional["torch.Tensor"] = field(default=None, repr=False)
    fc1_norm_const: Optional["torch.Tensor"] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        clamps = {}
        for name in ("gate_up_clamp", "activation_clamp"):
            value = getattr(self, name)
            if value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                or float(value) != int(value)
            ):
                raise ValueError(
                    f"{name} must be a positive whole number (compiled into the kernel), got {value!r}"
                )
            clamps[name] = int(value)
        if self.swiglu_limit is not None:
            if (
                not isinstance(self.swiglu_limit, int)
                or isinstance(self.swiglu_limit, bool)
                or self.swiglu_limit < 0
            ):
                raise ValueError(
                    f"swiglu_limit must be a non-negative int (0 = unclamped), got {self.swiglu_limit!r}"
                )
            clamps["swiglu_limit"] = self.swiglu_limit
        if len(set(clamps.values())) > 1:
            raise ValueError(f"conflicting activation clamps: {clamps}")
        clamp = next(iter(clamps.values()), 0)
        self.swiglu_limit = clamp
        self.gate_up_clamp = float(clamp) if clamp else None
        if self.activation_clamp is not None:
            self.activation_clamp = self.gate_up_clamp

        codecs = {
            name: getattr(self, name)
            for name in ("combine_dtype", "return_codec")
            if getattr(self, name) is not None
        }
        for name, value in codecs.items():
            if value not in RETURN_CODECS:
                raise ValueError(
                    f"{name} must be one of {sorted(RETURN_CODECS)}, got {value!r}"
                )
        if len(set(codecs.values())) > 1:
            raise ValueError(f"conflicting return codecs: {codecs}")
        self.combine_dtype = self.return_codec = next(iter(codecs.values()), "bf16")

        if not self.apply_routing_weights_before_fc2:
            raise ValueError(
                "the Cake kernel applies the routing weights before fc2 (apply_routing_weights_before_fc2=True)"
            )
        if (
            isinstance(self.input_norm_const, bool)
            or not isinstance(self.input_norm_const, (int, float))
            or not math.isfinite(self.input_norm_const)
            or self.input_norm_const <= 0
        ):
            raise ValueError(
                f"input_norm_const must be a positive finite number, got {self.input_norm_const!r}"
            )
        for name in _SCALE_FIELDS:
            check_expert_scale(name, getattr(self, name))
        super().__post_init__()


def check_expert_scale(
    name: str, value: object, local_experts: int | None = None
) -> None:
    """A per-expert scale must be ``None`` or a 1-D float32 CUDA tensor (of ``local_experts`` entries).

    CUDA because every launch copies it into the kernel's scale tail on the stream (a host tensor
    would be a pageable host-to-device copy, which CUDA graph capture rejects).
    """

    if value is None:
        return
    import torch

    if (
        not isinstance(value, torch.Tensor)
        or value.dtype != torch.float32
        or value.dim() != 1
        or not value.is_cuda
    ):
        raise ValueError(
            f"{name} must be a 1-D float32 CUDA tensor [num_local_experts], got "
            + (
                f"{value.dtype} {tuple(value.shape)} on {value.device}"
                if isinstance(value, torch.Tensor)
                else repr(value)
            )
        )
    if local_experts is not None and value.numel() != local_experts:
        raise ValueError(
            f"{name} has {value.numel()} entries, this rank holds {local_experts} local experts"
        )


__all__ = ["Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig", "check_expert_scale"]
