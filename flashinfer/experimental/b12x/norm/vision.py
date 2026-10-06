"""CuTe support operations for the replicated V4.1 vision tower.

Only static channel/head geometry enters compilation. Image dimensions and live
rows are launch scalars; all outputs, including attention metadata, are owned by
the caller. Spatial merge follows channel-major torch.unfold, not pixel shuffle.
"""
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64

from .._lib.compiler import KernelCompileSpec, compile as compile_cute, run_compiled
from .._lib.runtime_control import raise_if_kernel_resolution_frozen
from .._lib.utils import current_cuda_stream, make_ptr


class _Vision:
    def __init__(self, operation, width, heads, ratio):
        self.operation, self.width = operation, width
        self.heads, self.ratio = heads, ratio

    @cute.jit
    def __call__(self, x: cute.Pointer, out: cute.Pointer, k: cute.Pointer,
                 v: cute.Pointer, inv: cute.Pointer, cu: cute.Pointer,
                 rows: Int32, height: Int32, width: Int32,
                 stream: cuda.CUstream):
        self.kernel(x, out, k, v, inv, cu, rows, height, width).launch(
            grid=(rows, 1, 1), block=(128, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, x: cute.Pointer, out: cute.Pointer, k: cute.Pointer,
               v: cute.Pointer, inv: cute.Pointer, cu: cute.Pointer,
               rows: Int32, height: Int32, width: Int32):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        if cutlass.const_expr(self.operation == "rope"):
            if row == 0 and tid == 0:
                cu[0], cu[1] = Int32(0), rows
            for item in cutlass.range_constexpr((self.width + 127) // 128):
                col = tid + item * 128
                if col < self.width:
                    dim = self.width // self.heads
                    half = dim // 2
                    local = col % dim
                    freq_col = local % half
                    position = row // width
                    if freq_col >= half // 2:
                        position = row % width
                    angle = Float32(position) * Float32(inv[freq_col % (half // 2)])
                    cosine = cute.math.cos(angle)
                    sine = cute.math.sin(angle)
                    partner = col + half
                    sign = Float32(-1.0)
                    if local >= half:
                        partner = col - half
                        sign = Float32(1.0)
                    base = Int64(row) * Int64(3 * self.width)
                    dst = Int64(row) * Int64(self.width) + Int64(col)
                    q0, q1 = Float32(x[base + Int64(col)]), Float32(x[base + Int64(partner)])
                    k0 = Float32(x[base + Int64(self.width + col)])
                    k1 = Float32(x[base + Int64(self.width + partner)])
                    out[dst] = BFloat16(q0 * cosine + sign * q1 * sine)
                    k[dst] = BFloat16(k0 * cosine + sign * k1 * sine)
                    v[dst] = x[base + Int64(2 * self.width + col)]
        elif cutlass.const_expr(self.operation == "merge"):
            r = self.ratio
            merged_width = (width + r - 1) // r
            for item in cutlass.range_constexpr((self.width * r * r + 127) // 128):
                col = tid + item * 128
                if col < self.width * r * r:
                    channel = col // (r * r)
                    ih = (row // merged_width) * r + (col // r) % r
                    iw = (row % merged_width) * r + col % r
                    value = BFloat16(0.0)
                    if ih < height and iw < width:
                        src = (Int64(ih) * Int64(width) + Int64(iw)) * Int64(self.width) + Int64(channel)
                        value = x[src]
                    out[Int64(row) * Int64(self.width * r * r) + Int64(col)] = value
        else:
            for item in cutlass.range_constexpr((self.width + 127) // 128):
                col = tid + item * 128
                if col < self.width:
                    index = Int64(row) * Int64(self.width) + Int64(col)
                    value = Float32(x[index])
                    out[index] = BFloat16(Float32(0.5) * value * (
                        Float32(1.0) + cute.math.erf(value * Float32(0.7071067811865476))))


@cache
def _compile(operation, width, heads, ratio, device):
    key = (operation, width, heads, ratio, device)
    entry = _Vision(operation, width, heads, ratio)
    raise_if_kernel_resolution_frozen("cute.compile", target=entry, cache_key=key)
    types = (BFloat16, BFloat16, BFloat16, BFloat16, Float32, Int32)
    pointers = tuple(make_ptr(t, 16, cute.AddressSpace.gmem, assumed_align=t.width // 8) for t in types)
    with torch.cuda.device(device):
        compiled = compile_cute(entry, *pointers, Int32(1), Int32(1), Int32(1), current_cuda_stream(), compile_spec=KernelCompileSpec.from_key("norm.vision." + operation, 1, key))
    return compiled, types


def _check(x, out):
    if not x.is_cuda or x.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise TypeError("vision kernels require CUDA BF16 input/output")
    if x.device != out.device or not x.is_contiguous() or not out.is_contiguous():
        raise ValueError("vision input/output must be contiguous on the same device")


def _run_state(state, x, out, rows, height=1, width=1, k=None, v=None, inv=None, cu=None):
    """Launch an already-resolved vision specialization."""
    if rows == 0:
        return out
    query = state.query
    channels = x.shape[-1] // 3 if query.operation == "rope" else x.shape[-1]
    if query.channels != channels:
        raise ValueError("prepared vision plan has incompatible channel geometry")
    tensors = (x, out, k, v, inv, cu)
    pointers = tuple(make_ptr(t, (tensor if tensor is not None else x).data_ptr(), cute.AddressSpace.gmem, assumed_align=t.width // 8) for t, tensor in zip(state.types, tensors, strict=True))
    with torch.cuda.device(x.device):
        run_compiled(state.compiled, (*pointers, Int32(rows), Int32(height), Int32(width), current_cuda_stream()))
    return out


def run_gelu(x, *, out, plan):
    from b12x.preparation.types import require_prepared
    _check(x, out)
    if x.ndim != 2 or out.shape != x.shape:
        raise ValueError("GELU requires matching [rows, channels] tensors")
    state = require_prepared(plan, "norm.vision", x.device)
    if state.query.operation != "gelu":
        raise ValueError("prepared plan is not a GELU operation")
    return _run_state(state, x, out, x.shape[0])


def run_spatial_merge(x, height, width, *, ratio=3, out, plan):
    from b12x.preparation.types import require_prepared
    _check(x, out)
    if height <= 0 or width <= 0 or ratio <= 0 or x.ndim != 2:
        raise ValueError("spatial merge requires positive image dimensions and ratio")
    rows = ((height + ratio - 1) // ratio) * ((width + ratio - 1) // ratio)
    if x.shape[0] != height * width or out.shape != (rows, x.shape[1] * ratio * ratio):
        raise ValueError("spatial merge input/output shape mismatch")
    if x.data_ptr() == out.data_ptr():
        raise ValueError("spatial merge must not be in-place")
    state = require_prepared(plan, "norm.vision", x.device)
    if state.query.operation != "merge" or state.query.ratio != ratio:
        raise ValueError("prepared spatial merge has incompatible geometry")
    return _run_state(state, x, out, rows, height, width)


def run_rope_qkv(qkv, height, width, inv_freq, *, q, k, v, cu_seqlens, plan):
    from b12x.preparation.types import require_prepared
    _check(qkv, q)
    for tensor in (k, v):
        _check(qkv, tensor)
    if height <= 0 or width <= 0 or q.ndim != 3 or q.shape != k.shape or q.shape != v.shape:
        raise ValueError("invalid image dimensions or Q/K/V capacities")
    heads, dim = q.shape[1:]
    rows = height * width
    if dim % 4 or q.shape[0] < rows or qkv.shape != (rows, 3 * heads * dim):
        raise ValueError("QKV shape or rotary dimension mismatch")
    if inv_freq.shape != (dim // 4,) or inv_freq.dtype != torch.float32 or not inv_freq.is_contiguous():
        raise ValueError("inv_freq must be contiguous FP32[head_dim/4]")
    if cu_seqlens.shape != (2,) or cu_seqlens.dtype != torch.int32 or not cu_seqlens.is_contiguous():
        raise ValueError("cu_seqlens must be contiguous int32[2]")
    if inv_freq.device != q.device or cu_seqlens.device != q.device:
        raise ValueError("vision metadata must share the QKV device")
    if len({t.data_ptr() for t in (qkv, q, k, v)}) != 4:
        raise ValueError("QKV input and split outputs must not alias")
    state = require_prepared(plan, "norm.vision", qkv.device)
    if state.query.operation != "rope" or state.query.heads != heads:
        raise ValueError("prepared rotary plan has incompatible geometry")
    return _run_state(state, qkv, q, rows, height, width, k=k, v=v, inv=inv_freq, cu=cu_seqlens)


# Declarations stay separate from this kernel implementation so constructing a
# plan cannot allocate, load, or compile a native executable.
from ._vision_preparation import TUNING, VisionQuery, VisionState, plan

__all__ = [
    "VisionQuery", "VisionState", "TUNING", "plan", "run_gelu",
    "run_spatial_merge", "run_rope_qkv",
]
