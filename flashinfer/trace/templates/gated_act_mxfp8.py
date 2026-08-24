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

"""Trace templates for fused gated activation and MXFP8 quantization."""

from typing import Any, Callable, cast

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


def _quantize_reference(logical, rowwise, colwise):
    from flashinfer import SfLayout, mxfp8_quantize

    empty_q = logical.new_empty(0, dtype=torch.float8_e4m3fn)
    empty_s = logical.new_empty(0, dtype=torch.float8_e8m0fnu)
    if rowwise:
        row_q, row_s = mxfp8_quantize(
            logical, sf_swizzle_layout=SfLayout.layout_128x4
        )
        row_s = row_s.reshape(logical.shape[0], logical.shape[1] // 32).view(
            torch.float8_e8m0fnu
        )
    else:
        row_q, row_s = empty_q, empty_s
    if colwise:
        col_q_t, col_s = mxfp8_quantize(
            logical.T.contiguous(), sf_swizzle_layout=SfLayout.layout_128x4
        )
        col_q = col_q_t.T
        col_s = col_s.view(torch.float8_e8m0fnu)
    else:
        col_q, col_s = empty_q, empty_s
    return row_q, col_q, row_s, col_s


def _forward_reference(gated_input, rowwise=True, colwise=False):
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    logical = (torch.nn.functional.silu(gate) * up).bfloat16()
    return _quantize_reference(logical, rowwise, colwise)


def _backward_reference(
    gated_input, grad_output, rowwise=True, colwise=False
):
    k = gated_input.shape[1] // 2
    gate = gated_input[:, :k].float()
    up = gated_input[:, k:].float()
    grad = grad_output.float()
    sigmoid_gate = torch.sigmoid(gate)
    silu_gate = gate * sigmoid_gate
    dact = silu_gate * (1.0 - sigmoid_gate) + sigmoid_gate
    logical = torch.cat(
        (((dact * grad) * up).bfloat16(), (silu_gate * grad).bfloat16()), dim=1
    )
    return _quantize_reference(logical, rowwise, colwise)


cast(Any, _forward_reference)._trace_reference_dependencies = (_quantize_reference,)
cast(Any, _backward_reference)._trace_reference_dependencies = (_quantize_reference,)


def _init_inputs(
    *,
    M: int,
    K_doubled: int = 4096,
    backward: bool,
    rowwise: bool,
    colwise: bool,
    device: str = "cuda",
    seed: int = 0,
    **_unused,
):
    generator = torch.Generator(device=device).manual_seed(seed)
    gated_input = torch.randn(
        M,
        K_doubled,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    result = {
        "gated_input": gated_input,
        "rowwise": rowwise,
        "colwise": colwise,
    }
    if backward:
        result["grad_output"] = torch.randn(
            M,
            K_doubled // 2,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
    return result


def _forward_row_init(**kwargs):
    return _init_inputs(backward=False, rowwise=True, colwise=False, **kwargs)


def _forward_col_init(**kwargs):
    return _init_inputs(backward=False, rowwise=False, colwise=True, **kwargs)


def _forward_both_init(**kwargs):
    return _init_inputs(backward=False, rowwise=True, colwise=True, **kwargs)


def _backward_row_init(**kwargs):
    return _init_inputs(backward=True, rowwise=True, colwise=False, **kwargs)


def _backward_col_init(**kwargs):
    return _init_inputs(backward=True, rowwise=False, colwise=True, **kwargs)


def _backward_both_init(**kwargs):
    return _init_inputs(backward=True, rowwise=True, colwise=True, **kwargs)


for _init in (
    _forward_row_init,
    _forward_col_init,
    _forward_both_init,
    _backward_row_init,
    _backward_col_init,
    _backward_both_init,
):
    cast(Any, _init)._trace_init_dependencies = (_init_inputs,)


def _make_trace(
    direction: str,
    rowwise: bool,
    colwise: bool,
    init: Callable,
) -> TraceTemplate:
    mode = "both" if rowwise and colwise else "row" if rowwise else "col"
    inputs: dict[str, Tensor | Scalar] = {
        "gated_input": Tensor(
            ["M", "K_doubled"],
            description="Contiguous BF16 gate and up values.",
        ),
        "rowwise": Scalar("bool"),
        "colwise": Scalar("bool"),
    }
    if direction == "backward":
        inputs["grad_output"] = Tensor(
            ["M", "K"], description="Contiguous BF16 output gradient."
        )

    outputs: dict[str, Tensor | Scalar] = {
        "row_output": Tensor(
            ["M", "O"] if rowwise else ["zero"], dtype="float8_e4m3fn"
        ),
        "col_output": Tensor(
            ["M", "O"] if colwise else ["zero"], dtype="float8_e4m3fn"
        ),
        "row_scales": Tensor(
            ["M", "O_div_32"] if rowwise else ["zero"],
            dtype="float8_e8m0fnu",
        ),
        "col_scales": Tensor(
            ["SF"] if colwise else ["zero"], dtype="float8_e8m0fnu"
        ),
    }
    constraints = [
        "K_doubled == 2 * K",
        "M % 128 == 0",
        "K % 128 == 0",
        "O_div_32 == O // 32",
        "SF == M * O // 32",
        "zero == 0",
        f"O == {'K' if direction == 'forward' else 'K_doubled'}",
        f"rowwise == {int(rowwise)}",
        f"colwise == {int(colwise)}",
    ]
    return TraceTemplate(
        op_type="activation_quantize",
        name_prefix=f"silu_and_mul_mxfp8_{direction}_{mode}",
        description=(
            f"Fused {'SwiGLU' if direction == 'forward' else 'SwiGLU backward'} "
            f"with RCEIL MXFP8 {mode} quantization."
        ),
        axes={
            "M": Var(description="Number of rows."),
            "K_doubled": Const(description="Gate plus up width."),
            "K": Var(description="Single gated hidden width."),
            "O": Var(description="Logical output width."),
            "O_div_32": Var(description="Scale columns."),
            "SF": Var(description="Columnwise scale buffer length."),
            "zero": Var(description="Disabled output length."),
        },
        inputs=inputs,
        outputs=outputs,
        constraints=constraints,
        tags=["status:verified", "fused", "quantization:mxfp8"],
        reference=(
            _forward_reference if direction == "forward" else _backward_reference
        ),
        init=init,
    )


silu_and_mul_mxfp8_forward_row_trace = _make_trace(
    "forward", True, False, _forward_row_init
)
silu_and_mul_mxfp8_forward_col_trace = _make_trace(
    "forward", False, True, _forward_col_init
)
silu_and_mul_mxfp8_forward_both_trace = _make_trace(
    "forward", True, True, _forward_both_init
)
silu_and_mul_mxfp8_backward_row_trace = _make_trace(
    "backward", True, False, _backward_row_init
)
silu_and_mul_mxfp8_backward_col_trace = _make_trace(
    "backward", False, True, _backward_col_init
)
silu_and_mul_mxfp8_backward_both_trace = _make_trace(
    "backward", True, True, _backward_both_init
)


def silu_and_mul_mxfp8_forward_trace_dispatch(**kwargs):
    rowwise = bool(kwargs.get("rowwise", True))
    colwise = bool(kwargs.get("colwise", False))
    if rowwise and colwise:
        return silu_and_mul_mxfp8_forward_both_trace
    if rowwise:
        return silu_and_mul_mxfp8_forward_row_trace
    if colwise:
        return silu_and_mul_mxfp8_forward_col_trace
    raise ValueError("at least one quantization route must be enabled")


silu_and_mul_mxfp8_forward_trace_dispatch.templates = [  # type: ignore[attr-defined]
    silu_and_mul_mxfp8_forward_row_trace,
    silu_and_mul_mxfp8_forward_col_trace,
    silu_and_mul_mxfp8_forward_both_trace,
]


def silu_and_mul_mxfp8_backward_trace_dispatch(**kwargs):
    rowwise = bool(kwargs.get("rowwise", True))
    colwise = bool(kwargs.get("colwise", False))
    if rowwise and colwise:
        return silu_and_mul_mxfp8_backward_both_trace
    if rowwise:
        return silu_and_mul_mxfp8_backward_row_trace
    if colwise:
        return silu_and_mul_mxfp8_backward_col_trace
    raise ValueError("at least one quantization route must be enabled")


silu_and_mul_mxfp8_backward_trace_dispatch.templates = [  # type: ignore[attr-defined]
    silu_and_mul_mxfp8_backward_row_trace,
    silu_and_mul_mxfp8_backward_col_trace,
    silu_and_mul_mxfp8_backward_both_trace,
]


__all__ = [
    "silu_and_mul_mxfp8_backward_trace_dispatch",
    "silu_and_mul_mxfp8_forward_trace_dispatch",
]
