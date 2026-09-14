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

"""Shared MoE input validation helpers."""

from __future__ import annotations

import warnings
from typing import Optional, Union

import torch

from ...tllm_enums import ActivationType, SfLayout
from ...utils import check_shape_dtype_device


def validate_bf16_gemm1_activation_params(
    activation_type: int,
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    local_num_experts: int,
    device: torch.device,
) -> None:
    if gemm1_alpha is None and gemm1_beta is None and gemm1_clamp_limit is None:
        return
    if int(activation_type) != int(ActivationType.Swiglu):
        raise ValueError(
            "gemm1_alpha, gemm1_beta, and gemm1_clamp_limit are only supported "
            "for ActivationType.Swiglu."
        )
    for name, tensor in (
        ("gemm1_alpha", gemm1_alpha),
        ("gemm1_beta", gemm1_beta),
        ("gemm1_clamp_limit", gemm1_clamp_limit),
    ):
        if tensor is not None:
            check_shape_dtype_device(
                tensor,
                (local_num_experts,),
                torch.float32,
                device,
                name,
            )


# The trtllm-gen batched GEMM that consumes the caller's activation scale is
# built with routeAct=true (csrc/trtllm_fused_moe_runner.cu:119), and its
# kernel-option validation hard-requires a routed operand's scale factors to be
# in the LINEAR layout:
#
#   TLLM_CHECK_ERROR(options.mSfLayoutB == tg::SfLayout::Linear || ...,
#                    "Tokens need use SF linear layout when being routed");
#     -- trtllmGen_bmm_export/BatchedGemmOptions.h
#
# So the layout is a contract, not a dispatch knob: declaring it lets a mismatch
# fail loudly instead of being read as linear bytes and silently producing wrong
# numbers.  (The runtime SfLayout switch in trtllm_fused_moe_runner.cu belongs to
# GEMM2, whose scale factors FlashInfer produces itself and never come from the
# caller.)
SUPPORTED_MOE_ACT_SF_LAYOUT = SfLayout.layout_linear

# Double deprecation: both the implicit layout *and* the optionality of the
# parameter are announced at once, so a later release can make the parameter
# required and delete the inference in a single step without ever flipping a
# default under an existing caller.
IMPLICIT_ACT_SF_LAYOUT_DEPRECATION = (
    "Calling a trtllm-gen block-scale MoE op with a `hidden_states_scale` but "
    "without `hidden_states_scale_layout` is deprecated; the layout is being "
    "inferred as SfLayout.layout_linear. The layout cannot actually be "
    "recovered from the tensor -- a 128x4-swizzled buffer has exactly the "
    "linear numel whenever num_tokens % 128 == 0 (see #3455) -- so pass "
    "`hidden_states_scale_layout=flashinfer.tllm_enums.SfLayout.layout_linear` "
    "explicitly. Both the inference and the parameter's optionality are "
    "deprecated: the parameter is scheduled to become required and the "
    "inference to be removed in a future release."
)


def resolve_moe_act_sf_layout(
    hidden_states_scale_layout: Optional[Union[int, SfLayout]],
) -> SfLayout:
    """Resolve the activation scale-factor layout for a trtllm-gen MoE call.

    ``None`` reproduces the historical behavior exactly -- assume the linear
    layout -- and emits a :class:`DeprecationWarning`.  Any explicitly declared
    layout other than :attr:`SfLayout.layout_linear` is rejected, because the
    routed GEMM cannot consume it (see ``SUPPORTED_MOE_ACT_SF_LAYOUT``).
    """
    if hidden_states_scale_layout is None:
        # Dedup is per distinct *caller* line: with stacklevel != 1 ``warnings``
        # resolves ``__warningregistry__`` against the caller's frame, so this
        # bounds the warning to one per user call site rather than one per
        # process.  That is enough for a hot path -- ``_make_tuning_config``
        # runs on every op call and is not memoized.
        #
        # stacklevel 6 walks this helper -> MoERunner._make_tuning_config ->
        # <op impl> -> _auto_dump_wrapper -> <public wrapper> -> user code.
        # Call it from _make_tuning_config itself, not from deeper inside
        # make_moe_tuning_config, or this count is off by one.  Note
        # _auto_dump_wrapper is present *unconditionally* whenever the API is
        # declared with trace= (it does not depend on FLASHINFER_LOGLEVEL), so
        # omitting it attributes the warning to flashinfer/api_logging.py and
        # the default filters then hide it entirely.  Depth still varies for the
        # unified-API runners, so the message stands on its own.
        warnings.warn(
            IMPLICIT_ACT_SF_LAYOUT_DEPRECATION, DeprecationWarning, stacklevel=6
        )
        return SUPPORTED_MOE_ACT_SF_LAYOUT

    # Annotated wide because the fallback keeps the caller's raw value: an int
    # outside the enum has no SfLayout member, and it is that unrecognized value
    # the error message must echo back.
    layout: Union[int, SfLayout]
    try:
        layout = SfLayout(hidden_states_scale_layout)
    except ValueError:
        layout = hidden_states_scale_layout  # reported verbatim below
    if layout != SUPPORTED_MOE_ACT_SF_LAYOUT:
        raise NotImplementedError(
            f"hidden_states_scale_layout={layout!r} is not supported by the "
            "trtllm-gen MoE path: the routed batched GEMM requires the "
            "activation scale factors in "
            f"{SUPPORTED_MOE_ACT_SF_LAYOUT!r} "
            '("Tokens need use SF linear layout when being routed"). Quantize '
            "the activations with is_sf_swizzled_layout=False (equivalently "
            "sf_swizzle_layout=SfLayout.layout_linear)."
        )
    # Equal to the sole supported layout, so hand back the canonical enum member
    # rather than a bare int that merely compares equal to it.
    return SUPPORTED_MOE_ACT_SF_LAYOUT
