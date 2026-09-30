"""Caller-owned preparation for lagged four-stream MHC prefill projection.

The residual is already post-mixed. Copy its BF16 bits and compute the squared
norm; the incoming lagged mix lets the finalizer omit pairwise Gram statistics.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass import Float32, Int32, Int64, Uint32

from b12x._lib.compiler import DimKey, KernelCompileSpec, launch, tensor_key
from b12x._lib.intrinsics import bfloat2_to_float2_scaled
from b12x._lib.utils import current_cuda_stream
from ._kernels import _to_kernel_tensor, _warp_allreduce_sum, _compile_mhc_entry
from b12x._lib.compiler import run_compiled


class _PrepareLaggedPrefill:
    def __init__(self, hidden: int):
        self.hidden = hidden

    @cute.jit
    def __call__(
        self,
        residual: cute.Tensor,
        output: cute.Tensor,
        partials: cute.Tensor,
        tokens: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(residual, output, partials).launch(
            grid=(tokens, 1, 1), block=(512, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, residual: cute.Tensor, output: cute.Tensor, partials: cute.Tensor):
        row, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane, warp = Int32(tid) % Int32(32), Int32(tid) // Int32(32)
        source = cute.recast_tensor(residual, Uint32)
        target = cute.recast_tensor(output, Uint32)
        smem = utils.SmemAllocator()
        sums = smem.allocate_tensor(Float32, cute.make_layout((16,)), 16)
        total = Float32(0)
        for offset in cutlass.range_constexpr((self.hidden * 2 + 511) // 512):
            pair = Int32(offset * 512) + Int32(tid)
            if pair < Int32(self.hidden * 2):
                stream_id = pair // Int32(self.hidden // 2)
                column = pair % Int32(self.hidden // 2)
                value = source[Int64(row), stream_id, column]
                target[Int64(row), stream_id, column] = value
                lo, hi = bfloat2_to_float2_scaled(value, Float32(1))
                total += lo * lo
                total += hi * hi
        total = _warp_allreduce_sum(total)
        if lane == Int32(0):
            sums[warp] = total
        cute.arch.sync_threads()
        if tid == Int32(0):
            square_sum = Float32(0)
            for index in cutlass.range_constexpr(16):
                square_sum += sums[index]
            partials[Int64(row), 0, 0] = square_sum


@cache
def _kernel(hidden):
    return _PrepareLaggedPrefill(hidden)


def prepare_lagged_prefill(
    residual: torch.Tensor, output: torch.Tensor, partials: torch.Tensor, *, _prepared=None,
):
    if (
        residual.ndim != 3
        or residual.shape[1] != 4
        or residual.shape[2] <= 0
        or residual.shape[2] % 2
        or residual.dtype != torch.bfloat16
        or not residual.is_cuda
    ):
        raise ValueError("lagged MHC prefill requires BF16 [tokens,4,hidden] residual")
    if not residual.is_contiguous() or not output.is_contiguous():
        raise ValueError("lagged MHC prefill requires contiguous residual and output")
    if output.shape != residual.shape or output.dtype != residual.dtype:
        raise ValueError("lagged MHC output must match the residual")
    if (
        partials.dtype != torch.float32
        or partials.ndim != 3
        or partials.shape[0] != residual.shape[0]
        or partials.shape[1] < 1
        or partials.shape[2] < 1
        or not partials.is_contiguous()
    ):
        raise ValueError("lagged MHC partials require FP32 [tokens,splits,statistics]")
    if output.device != residual.device or partials.device != residual.device:
        raise ValueError("lagged MHC inputs and outputs must share a CUDA device")
    if (
        torch._C._overlaps(residual, output)
        or torch._C._overlaps(partials, output)
        or torch._C._overlaps(residual, partials)
    ):
        raise ValueError(
            "lagged MHC preparation outputs must not alias inputs or scratch"
        )
    tokens, _, hidden = residual.shape
    if not tokens:
        return
    args = (
        _to_kernel_tensor(residual, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(output, cutlass.BFloat16, dynamic_layout=True),
        _to_kernel_tensor(partials, cutlass.Float32, dynamic_layout=True),
        Int32(tokens),
        current_cuda_stream(),
    )
    if _prepared is not None:
        return run_compiled(_prepared, args)
    key = tuple(
        tensor_key(
            name,
            tensor,
            dims=(DimKey.dynamic(),) + tuple(DimKey.exact(v) for v in tensor.shape[1:]),
        )
        for name, tensor in (
            ("residual", residual),
            ("output", output),
            ("partials", partials),
        )
    )
    return _compile_mhc_entry(
        _kernel(hidden),
        compile_spec=KernelCompileSpec.from_key(
            "norm.mhc.lagged_prefill_prepare", 1, key
        ),
        compile_args=args,
        runtime_args=args,
    )
