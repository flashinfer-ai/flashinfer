"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Configuration of the Cake W4A16 MegaMoE backend (SM103 / SM100).

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from flashinfer.moe_ep._compat import deprecated_aliases, legacy_finalize
from flashinfer.moe_ep.config import FinalizeMode, validate_finalize_mode

from .cake_plan import KERNEL_NAME, TOP_K, cell_from_knobs

if TYPE_CHECKING:
    import torch


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@legacy_finalize
@deprecated_aliases(apply_topk_in_fc1="apply_routing_weights_before_fc2")
@dataclass
class Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig:
    """BF16 activations and NVFP4 expert weights, decoded in-kernel to BF16.

    The field names mirror :class:`Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig`
    so a caller swaps ``kernel_name`` only. The numerics contract is the one
    of that kernel: FC1 = BF16(x) x deq(w13) with FP32 accumulation, SwiGLU in
    FP32 (``silu(gate) * up``, optional ``gate_up_clamp``), a BF16 handoff,
    FC2 = BF16(h) x deq(w2) with FP32 accumulation, ``fc1_alpha`` /
    ``fc2_alpha`` as optional per-expert FP32 epilogue scales, routing scores
    applied after FC2, and a combine that sums the BF16 route terms in FP32 in
    fixed slot order (deterministic; a ``-1`` slot contributes zero).

    Out-of-scope forms of the sibling config are rejected here:
    ``apply_routing_weights_before_fc2``, ``enable_in_kernel_fc2_reduce``,
    ``fc1_norm_const``, ``swiglu_alpha`` / ``swiglu_beta``, ``activation="situ"``
    (``situ_beta`` / ``situ_linear_beta``), ``activation_clamp`` (the clamp is
    ``gate_up_clamp``) and ``finalize_mode=NONE``. ``knobs`` is ``None`` or ``"auto"`` (the host
    selects the route tile from the token bound; no tuning runs, and
    ``get_tuned_knobs()`` reports the selected cell after the first forward)
    or a dict pinning a shipped cell, ``{"n": 32 | 64 | 128, "cta_group": 1 | 2}``.
    """

    intermediate_size: int
    top_k: int = TOP_K
    kernel_name: str = KERNEL_NAME
    gate_up_clamp: float | None = None
    enable_in_kernel_fc2_reduce: bool = False
    fc1_alpha: torch.Tensor | None = None
    fc2_alpha: torch.Tensor | None = None
    knobs: dict | None = None
    apply_routing_weights_before_fc2: bool = False
    fc1_norm_const: torch.Tensor | None = None
    swiglu_alpha: float | None = None
    swiglu_beta: float | None = None
    activation: Literal["swiglu"] = "swiglu"
    situ_beta: float | None = None
    situ_linear_beta: float | None = None
    activation_clamp: float | None = None
    finalize_mode: FinalizeMode = FinalizeMode.AUTO

    def __post_init__(self) -> None:
        if self.kernel_name != KERNEL_NAME:
            raise ValueError(
                f"kernel_name must be {KERNEL_NAME!r}, got {self.kernel_name!r}"
            )
        if not _is_int(self.intermediate_size) or self.intermediate_size <= 0:
            raise ValueError(
                f"{KERNEL_NAME}: intermediate_size must be a positive int, "
                f"got {self.intermediate_size!r}"
            )
        if not _is_int(self.top_k) or self.top_k != TOP_K:
            raise ValueError(
                f"{KERNEL_NAME}: top_k must be {TOP_K}, got {self.top_k!r}"
            )
        validate_finalize_mode(
            self.finalize_mode, (FinalizeMode.AUTO, FinalizeMode.DEFAULT)
        )
        if self.apply_routing_weights_before_fc2:
            raise ValueError(
                f"{KERNEL_NAME} applies routing weights after FC2 only; "
                "apply_routing_weights_before_fc2=True is not supported"
            )
        if self.enable_in_kernel_fc2_reduce:
            raise ValueError(
                f"{KERNEL_NAME} combines BF16 route terms in FP32 in fixed slot "
                "order; enable_in_kernel_fc2_reduce=True is not supported"
            )
        if self.fc1_norm_const is not None:
            raise ValueError(f"{KERNEL_NAME} does not support fc1_norm_const")
        if self.swiglu_alpha is not None or self.swiglu_beta is not None:
            raise ValueError(
                f"{KERNEL_NAME} implements the default SwiGLU (silu(gate) * up); "
                "swiglu_alpha / swiglu_beta are not supported"
            )
        if self.activation != "swiglu":
            raise ValueError(
                f"{KERNEL_NAME} supports activation='swiglu' only, got {self.activation!r}"
            )
        if self.situ_beta is not None or self.situ_linear_beta is not None:
            raise ValueError(f"{KERNEL_NAME} does not support the SiTU activation")
        if self.activation_clamp is not None:
            raise ValueError(
                f"{KERNEL_NAME} takes the SwiGLU input clamp as gate_up_clamp only; "
                "activation_clamp is not supported"
            )
        value = self.gate_up_clamp
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"{KERNEL_NAME}: gate_up_clamp must be a positive finite number or "
                f"None, got {value!r}"
            )
        if self.knobs is not None and self.knobs != "auto":
            if not isinstance(self.knobs, dict):
                raise ValueError(
                    f"{KERNEL_NAME}: knobs must be None or a dict, got {self.knobs!r}"
                )
            unknown = set(self.knobs) - {"n", "cta_group"}
            if unknown:
                raise ValueError(
                    f"{KERNEL_NAME}: unknown knobs {sorted(unknown)}; the route "
                    "knows n and cta_group"
                )
            try:
                cell_from_knobs(self.knobs)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{KERNEL_NAME}: knobs {self.knobs!r}: {error}"
                ) from error

    @property
    def resolved_gate_up_clamp(self) -> float | None:
        """The clamp of the SwiGLU inputs (``gate_up_clamp``; the sibling's
        ``activation_clamp`` spelling is rejected above)."""

        return self.gate_up_clamp


__all__ = ["KERNEL_NAME", "Sm103_Bf16_Nvfp4_Bf16_Cake_MegaMoeConfig", "TOP_K"]
