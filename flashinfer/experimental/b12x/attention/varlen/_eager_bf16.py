"""Prepared two-pass attention with BF16 score and probability boundaries."""

from __future__ import annotations

from dataclasses import dataclass
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32

from b12x._lib.compile_plan import attach_programs
from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec, scratch_tensor
from b12x._lib.utils import current_cuda_stream, make_ptr
from b12x.attention._shared.contiguous.forward import ContiguousAttentionForwardKernel


def _align(value):
    return (value + 1023) // 1024 * 1024


def _layout(invocation):
    rows, heads, _ = invocation["q_shape"]
    width = invocation["v_shape"][-1]
    output_bytes = rows * heads * width * 2
    row_bytes = rows * heads * 4
    lse = _align(output_bytes)
    maximum = _align(lse + row_bytes)
    inverse = _align(maximum + row_bytes)
    return (0, lse, maximum, inverse), max(1, inverse + row_bytes)


def scratch_spec(invocation, ordinal):
    return scratch_buffer_spec(
        "varlen_eager_bf16.scratch", nbytes=_layout(invocation)[1],
        device=torch.device("cuda", ordinal),
    )


def validate_extents(q_shape, k_shape, v_shape, cu_q_shape, cu_k_shape):
    if min(q_shape[0], k_shape[0], v_shape[0]) <= 0:
        raise ValueError("eager_bf16 requires positive tensor row capacities; represent empty segments in cumulative lengths")
    if k_shape[0] != v_shape[0]:
        raise ValueError("eager_bf16 K/V row counts must match")
    if min(cu_q_shape[0], cu_k_shape[0]) < 2:
        raise ValueError("eager_bf16 requires at least one segment in cumulative-length storage")


class _Launch:
    def __init__(self, invocation, config, phase):
        self.q_shape = tuple(invocation["q_shape"])
        self.k_shape = tuple(invocation["k_shape"])
        self.v_shape = tuple(invocation["v_shape"])
        self.max_q = invocation["max_seqlen_q"]
        self.max_k = invocation["max_seqlen_k"]
        if not ContiguousAttentionForwardKernel.can_implement(
            BFloat16, self.q_shape[-1], self.v_shape[-1],
            config.tile_m, config.tile_n, 1, 160, False,
        ):
            raise ValueError("eager_bf16 head geometry or tile does not fit the SM120 attention kernel")
        self.kernel = ContiguousAttentionForwardKernel(
            BFloat16, self.q_shape[-1], head_dim_v=self.v_shape[-1],
            tile_m=config.tile_m, tile_n=config.tile_n, math_phase=phase, pack_gqa=False,
        )

    @cute.jit
    def __call__(
        self, q_ptr: cute.Pointer, k_ptr: cute.Pointer, v_ptr: cute.Pointer,
        out_ptr: cute.Pointer, lse_ptr: cute.Pointer,
        maximum_ptr: cute.Pointer, inverse_ptr: cute.Pointer,
        cu_q_ptr: cute.Pointer, cu_k_ptr: cute.Pointer,
        scale: Float32, q_rows: Int32, k_rows: Int32, segments: Int32,
        stream: cuda.CUstream,
    ):
        heads, qdim, vdim = self.q_shape[1], self.q_shape[2], self.v_shape[2]
        q = cute.make_tensor(q_ptr, cute.make_layout((q_rows, heads, qdim), stride=(heads * qdim, qdim, 1)))
        k = cute.make_tensor(k_ptr, cute.make_layout((k_rows, heads, qdim), stride=(heads * qdim, qdim, 1)))
        v = cute.make_tensor(v_ptr, cute.make_layout((k_rows, heads, vdim), stride=(heads * vdim, vdim, 1)))
        out = cute.make_tensor(out_ptr, cute.make_layout((self.q_shape[0], heads, vdim), stride=(heads * vdim, vdim, 1)))
        stats_layout = cute.make_layout((heads, self.q_shape[0]), stride=(self.q_shape[0], 1))
        lse = cute.make_tensor(lse_ptr, stats_layout)
        maximum = cute.make_tensor(maximum_ptr, stats_layout)
        inverse = cute.make_tensor(inverse_ptr, stats_layout)
        cu_q = cute.make_tensor(cu_q_ptr, cute.make_layout((segments + 1,), stride=(1,)))
        cu_k = cute.make_tensor(cu_k_ptr, cute.make_layout((segments + 1,), stride=(1,)))
        self.kernel(
            q, k, v, out, lse, scale, mCuSeqlensQ=cu_q, mCuSeqlensK=cu_k,
            aux_tensors=(maximum, inverse), logical_num_batch_static=1,
            logical_seqlen_q_static=self.max_q, logical_seqlen_k_static=self.max_k,
            learnable_sink=lse[None, 0],
            stream=stream,
        )


@program_cache(scope="preparation")
def compile_programs(invocation, config, ordinal):
    programs = []
    with torch.cuda.device(ordinal):
        for phase in ("statistics", "normalized"):
            launch = _Launch(invocation, config, phase)
            dtypes = (BFloat16,) * 4 + (Float32,) * 3 + (Int32,) * 2
            pointers = tuple(make_ptr(dtype, 16, cute.AddressSpace.gmem,
                                      assumed_align=16 if dtype == BFloat16 else 4) for dtype in dtypes)
            key = ("eager_bf16", phase, tuple(invocation["q_shape"]),
                   tuple(invocation["k_shape"]), tuple(invocation["v_shape"]),
                   int(invocation["max_seqlen_q"]), int(invocation["max_seqlen_k"]),
                   config.tile_m, config.tile_n)
            programs.append(b12x_compile(
                launch, *pointers, Float32(1), Int32(1), Int32(1), Int32(1),
                current_cuda_stream(),
                compile_spec=KernelCompileSpec.from_key("attention.varlen.eager_bf16", 1, key),
            ))
    return tuple(programs)


@dataclass(frozen=True)
class _Plan:
    invocation: object
    compiled: tuple
    device: torch.device


@dataclass(frozen=True)
class _Binding:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    output: torch.Tensor
    lse: torch.Tensor
    maximum: torch.Tensor
    inverse: torch.Tensor
    cu_q: torch.Tensor
    cu_k: torch.Tensor
    scale: float
    plan: _Plan


@dataclass(frozen=True)
class _ScratchPlan:
    plan: _Plan

    def scratch_specs(self):
        return (scratch_spec(self.plan.invocation, self.plan.device.index),)

    def shapes_and_dtypes(self):
        return tuple((spec.shape, spec.dtype) for spec in self.scratch_specs())


@dataclass(frozen=True)
class _State:
    plan: _Plan
    scratch_plan: _ScratchPlan

    def bind(
        self, *, scratch, q, k, v, cu_seqlens_q, cu_seqlens_k=None,
        plan=None, max_seqlen_q=None, max_seqlen_k=None, softmax_scale=None,
        causal=None, window_size=None, attention_sink_bias=None,
        block_indices=None, block_offsets=None,
    ):
        from ._preparation import VarlenBinding, _explicit_max_seqlen
        from b12x.attention._shared.contiguous import api as contiguous

        if causal not in (None, False) or any(value is not None for value in (
            window_size, attention_sink_bias, block_indices, block_offsets,
        )):
            raise ValueError("eager_bf16 binding requires dense noncausal attention without windows or sinks")
        cu_seqlens_k = cu_seqlens_q if cu_seqlens_k is None else cu_seqlens_k
        qshape, kshape, vshape, cqshape, ckshape, device, dtype = contiguous._validate_varlen_inputs(
            q, k, v, cu_seqlens_q, cu_seqlens_k,
        )
        validate_extents(qshape, kshape, vshape, cqshape, ckshape)
        invocation = self.plan.invocation
        if device != self.plan.device or dtype != torch.bfloat16:
            raise ValueError("eager_bf16 binding device or dtype differs from prepared plan")
        for name, shape in (("q", qshape), ("k", kshape), ("v", vshape),
                            ("cu_seqlens_q", cqshape), ("cu_seqlens_k", ckshape)):
            capacity = tuple(invocation[name + "_shape"])
            if shape[1:] != capacity[1:] or shape[0] > capacity[0]:
                raise ValueError(f"{name} shape exceeds or differs from eager_bf16 prepared capacity")
        for name, length in (("max_seqlen_q", max_seqlen_q), ("max_seqlen_k", max_seqlen_k)):
            if length is not None and _explicit_max_seqlen(length, name=name) > invocation[name]:
                raise ValueError(f"{name} exceeds eager_bf16 prepared capacity")
        scale = qshape[-1] ** -.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        arena = scratch_tensor(scratch, self.scratch_plan.scratch_specs(), owner="eager_bf16 attention")
        if any(value.data_ptr() % 16 for value in (arena, q, k, v)):
            raise ValueError("eager_bf16 scratch and Q/K/V must be 16-byte aligned")
        for source in (q, k, v, cu_seqlens_q, cu_seqlens_k):
            if torch._C._overlaps(arena, source):
                raise ValueError("eager_bf16 scratch must not overlap attention inputs")
        offsets, _ = _layout(invocation)
        rows, heads, _ = invocation["q_shape"]
        shapes = ((rows, heads, invocation["v_shape"][-1]),) + ((heads, rows),) * 3
        dtypes = (torch.bfloat16,) + (torch.float32,) * 3
        output, lse, maximum, inverse = (
            arena.narrow(0, offset, math.prod(shape) * dtype.itemsize).view(dtype).reshape(shape)
            for offset, shape, dtype in zip(offsets, shapes, dtypes)
        )
        binding = _Binding(q, k, v, output[:qshape[0]], lse[:, :qshape[0]], maximum,
                           inverse, cu_seqlens_q, cu_seqlens_k, scale, self.plan)
        return VarlenBinding(plan, binding)

    def run(self, binding):
        active = binding.binding
        if active.plan is not self.plan:
            raise ValueError("eager_bf16 binding belongs to another prepared plan")
        tensors = (active.q, active.k, active.v, active.output, active.lse,
                   active.maximum, active.inverse, active.cu_q, active.cu_k)
        dtypes = (BFloat16,) * 4 + (Float32,) * 3 + (Int32,) * 2
        pointers = tuple(make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                                  assumed_align=16 if dtype == BFloat16 else 4)
                         for tensor, dtype in zip(tensors, dtypes))
        for program in self.plan.compiled:
            program(*pointers, active.scale, active.q.shape[0], active.k.shape[0],
                    active.cu_q.shape[0] - 1, current_cuda_stream())
        return active.output, active.lse


def materialize(invocation, config, ordinal):
    plan = _Plan(invocation, compile_programs(invocation, config, ordinal), torch.device("cuda", ordinal))
    attach_programs(plan, *plan.compiled)
    return _State(plan, _ScratchPlan(plan))
