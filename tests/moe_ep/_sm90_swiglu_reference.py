# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Test-owned instruction-level SM90 SwiGLU reference.

The production Hopper epilogue evaluates sigmoid with ``ex2.approx.f32`` and
``rcp.approx.ftz.f32``.  Ordinary Torch ``exp2``/``reciprocal`` can land on
the other side of an E4M3 midpoint, so it is not a byte-exact handoff oracle.
This tiny CuTeDSL kernel mirrors only those documented PTX instructions and
does not import the vendored MegaMoE implementation.
"""

from __future__ import annotations

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack


_BLOCK = 256
_compiled_swiglu = None


def _f32_asm1(asm: str, value):
    """Emit one unary FP32 PTX instruction for the reference kernel."""
    return cutlass.Float32(
        llvm.inline_asm(
            cutlass.Float32.mlir_type,
            [value.ir_value()],
            asm,
            "=f,f",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.kernel
def _swiglu_kernel(
    gate: cute.Tensor,
    up: cute.Tensor,
    output: cute.Tensor,
    elements: Int32,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    index = block_idx * _BLOCK + thread_idx
    if index < elements:
        gate_value = gate[index]
        up_times_gate = up[index] * gate_value
        neg_gate_log2e = gate_value * cutlass.Float32(-1.4426950408889634)
        exp_neg = _f32_asm1("ex2.approx.f32 $0, $1;", neg_gate_log2e)
        sigmoid = _f32_asm1(
            "rcp.approx.ftz.f32 $0, $1;",
            exp_neg + cutlass.Float32(1.0),
        )
        output[index] = up_times_gate * sigmoid


@cute.jit
def _swiglu_host(
    gate: cute.Tensor,
    up: cute.Tensor,
    output: cute.Tensor,
):
    elements = cute.size(gate)
    _swiglu_kernel(gate, up, output, Int32(elements)).launch(
        grid=((elements + _BLOCK - 1) // _BLOCK, 1, 1),
        block=(_BLOCK, 1, 1),
    )


def _flat_dynamic(tensor: torch.Tensor):
    return from_dlpack(tensor.view(-1)).mark_layout_dynamic()


def swiglu_sm90_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Evaluate SM90's FP32 SwiGLU instruction sequence bit-exactly.

    Fail closed outside CUDA FP32 to prevent an instruction-inexact fallback.
    """
    if gate.shape != up.shape:
        raise ValueError(
            f"gate shape {tuple(gate.shape)} != up shape {tuple(up.shape)}"
        )
    if gate.device != up.device:
        raise ValueError(f"gate device {gate.device} != up device {up.device}")
    if not gate.is_cuda or not up.is_cuda:
        raise ValueError("SM90 SwiGLU reference requires CUDA tensors")
    if gate.dtype != torch.float32 or up.dtype != torch.float32:
        raise ValueError(
            "SM90 SwiGLU reference requires matching torch.float32 tensors"
        )
    if gate.numel() == 0:
        return torch.empty_like(gate)

    gate_contiguous = gate.contiguous()
    up_contiguous = up.contiguous()
    output = torch.empty_like(gate_contiguous)
    global _compiled_swiglu
    if _compiled_swiglu is None:
        _compiled_swiglu = cute.compile(
            _swiglu_host,
            _flat_dynamic(gate_contiguous),
            _flat_dynamic(up_contiguous),
            _flat_dynamic(output),
        )
    _compiled_swiglu(
        _flat_dynamic(gate_contiguous),
        _flat_dynamic(up_contiguous),
        _flat_dynamic(output),
    )
    torch.cuda.synchronize()
    return output.view_as(gate)


__all__ = ["swiglu_sm90_reference"]
