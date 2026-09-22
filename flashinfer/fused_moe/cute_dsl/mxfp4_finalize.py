# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Route-weighted reduction of permuted expert rows into the combined MoE
output (the second stage of the swap-AB path's two-stage finalize).

``out[t, :] = sum_k route_weights[t, k] * rows[expanded_idx_to_permuted_idx[t, k], :]``
over the slots whose permuted index is ``>= 0`` (rank-local); tokens without a
local slot are written as zeros, so the output needs no zero-fill. FP32
accumulation with one BF16 rounding, the same numerics as the in-epilogue
``red.global.add.bf16x2`` finalize up to summation order. Each thread owns 8
hidden elements (16 bytes) of one token; the kernel is a memory-bound gather.
Minimum architecture SM100 (B300: SM103) only because the surrounding path is.
"""

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
import cuda.bindings.driver as cuda
import torch

from ...cute_dsl.utils import make_ptr

_finalize_kernel_cache: dict = {}

# BF16 elements per thread task: four 32-bit words.
FINALIZE_VEC = 8


@dsl_user_op
def _fma_bf16_lo(word, weight, acc, *, loc=None, ip=None):
    """``acc + weight * bf16(word & 0xFFFF)`` in FP32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
                cutlass.Float32(weight).ir_value(loc=loc, ip=ip),
                cutlass.Float32(acc).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 t;\n.reg .f32 f;\n"
            "shl.b32 t, $1, 16;\nmov.b32 f, t;\nfma.rn.f32 $0, f, $2, $3;\n}",
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _fma_bf16_hi(word, weight, acc, *, loc=None, ip=None):
    """``acc + weight * bf16(word >> 16)`` in FP32."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Uint32(word).ir_value(loc=loc, ip=ip),
                cutlass.Float32(weight).ir_value(loc=loc, ip=ip),
                cutlass.Float32(acc).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .b32 t;\n.reg .f32 f;\n"
            "and.b32 t, $1, 0xffff0000;\nmov.b32 f, t;\nfma.rn.f32 $0, f, $2, $3;\n}",
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_bf16x2(lo, hi, *, loc=None, ip=None):
    """Round two FP32 values to one BF16x2 word (``lo`` in the low half)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                cutlass.Float32(lo).ir_value(loc=loc, ip=ip),
                cutlass.Float32(hi).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class _FinalizeRows:
    """``expanded_rows=False``: row ``expanded_idx_to_permuted_idx[t, k]`` of
    ``rows`` holds slot ``k`` of token ``t`` (swap-AB ``partial`` order);
    ``True``: row ``t * top_k + k`` does (the dense kernel's expanded-row
    order) and the map only marks rank-local slots (``>= 0``)."""

    def __init__(self, top_k, threads, expanded_rows):
        self.top_k = top_k
        self.threads = threads
        self.expanded_rows = expanded_rows

    @cute.jit
    def __call__(
        self,
        rows_ptr: cute.Pointer,
        perm_ptr: cute.Pointer,
        weights_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tokens: cutlass.Int32,
        hidden_words: cutlass.Int32,
        rows_words: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        # 32-bit word views: two BF16 per word, four words per thread task.
        rows = cute.make_tensor(rows_ptr, cute.make_layout((rows_words,)))
        # Row-major [T, top_k] maps and weights (make_layout defaults to
        # column-major, so the strides are explicit).
        perm = cute.make_tensor(
            perm_ptr, cute.make_layout((tokens, self.top_k), stride=(self.top_k, 1))
        )
        weights = cute.make_tensor(
            weights_ptr,
            cute.make_layout((tokens, self.top_k), stride=(self.top_k, 1)),
        )
        out = cute.make_tensor(out_ptr, cute.make_layout((tokens * hidden_words,)))
        chunks = hidden_words // (FINALIZE_VEC // 2)
        tasks = tokens * chunks
        self.kernel(rows, perm, weights, out, chunks, tasks).launch(
            grid=(cute.ceil_div(tasks, self.threads), 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        rows: cute.Tensor,
        perm: cute.Tensor,
        weights: cute.Tensor,
        out: cute.Tensor,
        chunks: cutlass.Int32,
        tasks: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        task = bidx * self.threads + tidx
        words_per_chunk = FINALIZE_VEC // 2
        if task < tasks:
            token = task // chunks
            chunk = task - token * chunks
            row_words = chunks * words_per_chunk
            chunk_words = chunk * words_per_chunk
            # Gather all local slots first (independent 16-byte loads), then
            # accumulate; slots that are not rank-local contribute zero and
            # issue no load.
            src = cute.make_rmem_tensor((self.top_k, words_per_chunk), cutlass.Uint32)
            wgt = cute.make_rmem_tensor((self.top_k,), cutlass.Float32)
            for k in cutlass.range_constexpr(self.top_k):
                for j in cutlass.range_constexpr(words_per_chunk):
                    src[(k, j)] = cutlass.Uint32(0)
                wgt[k] = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(self.top_k):
                permuted = perm[(token, k)]
                if permuted >= 0:
                    if cutlass.const_expr(self.expanded_rows):
                        row = token * self.top_k + k
                    else:
                        row = permuted
                    wgt[k] = weights[(token, k)]
                    base = cute.assume(row * row_words + chunk_words, divby=4)
                    g = cute.make_tensor(
                        rows.iterator + base, cute.make_layout((words_per_chunk,))
                    )
                    cute.autovec_copy(g, src[(k, None)])
            acc = cute.make_rmem_tensor((FINALIZE_VEC,), cutlass.Float32)
            for e in cutlass.range_constexpr(FINALIZE_VEC):
                acc[e] = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(self.top_k):
                for j in cutlass.range_constexpr(words_per_chunk):
                    word = src[(k, j)]
                    acc[2 * j] = _fma_bf16_lo(word, wgt[k], acc[2 * j])
                    acc[2 * j + 1] = _fma_bf16_hi(word, wgt[k], acc[2 * j + 1])
            packed = cute.make_rmem_tensor((words_per_chunk,), cutlass.Uint32)
            for j in cutlass.range_constexpr(words_per_chunk):
                packed[j] = _pack_bf16x2(acc[2 * j], acc[2 * j + 1])
            out_base = cute.assume(token * row_words + chunk_words, divby=4)
            g_out = cute.make_tensor(
                out.iterator + out_base, cute.make_layout((words_per_chunk,))
            )
            cute.autovec_copy(packed, g_out)


class _FinalizeRowsPlan:
    """Fixed-address launcher for one bound (rows, map, weights, out) set."""

    def __init__(self, compiled, arguments, owners, output):
        self._compiled = compiled
        self._arguments = arguments
        self._owners = owners
        self.output = output

    def run(self, stream):
        """Enqueue one kernel; no allocation, JIT or host synchronization."""
        self._compiled(*self._arguments, stream=stream)
        return self.output


def plan_finalize_rows(
    rows: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    route_weights: torch.Tensor,
    out: torch.Tensor,
    *,
    threads: int = 256,
    expanded_rows: bool = False,
) -> _FinalizeRowsPlan:
    """Compile, bind and enqueue one warmup, outside CUDA Graph capture.

    ``rows`` is the ``[R, H]`` BF16 buffer of expert outputs in permuted row
    order (the swap-AB GEMM2 ``partial`` epilogue's output, already scaled by
    the expert alpha), ``expanded_idx_to_permuted_idx`` the int32 ``[T, top_k]``
    row of each (token, slot) with ``-1`` for slots that are not rank-local,
    ``route_weights`` the FP32 ``[T, top_k]`` weights and ``out`` the ``[T, H]``
    BF16 combined output, which is fully overwritten. With
    ``expanded_rows=True`` the rows are in expanded ``(token, slot)`` order
    (``rows[t * top_k + k]``, the dense kernel's non-fused output) and the map
    only marks the rank-local slots. All tensors must be contiguous and on one
    device; ``H`` must be a multiple of 8.
    """
    tokens, top_k = expanded_idx_to_permuted_idx.shape
    hidden = out.shape[1]
    device = out.device
    if rows.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise ValueError("finalize rows and output must be bfloat16")
    if expanded_idx_to_permuted_idx.dtype != torch.int32:
        raise ValueError("expanded_idx_to_permuted_idx must be int32")
    if route_weights.dtype != torch.float32:
        raise ValueError("route_weights must be float32")
    if rows.ndim != 2 or rows.shape[1] != hidden or hidden % FINALIZE_VEC != 0:
        raise ValueError("rows must be [R, H] with H a multiple of 8")
    if expanded_rows and rows.shape[0] < tokens * top_k:
        raise ValueError("expanded rows need at least T * top_k rows")
    if out.shape != (tokens, hidden) or route_weights.shape != (tokens, top_k):
        raise ValueError("out must be [T, H] and route_weights [T, top_k]")
    for name, tensor in (
        ("rows", rows),
        ("expanded_idx_to_permuted_idx", expanded_idx_to_permuted_idx),
        ("route_weights", route_weights),
        ("out", out),
    ):
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
        if tensor.device != device:
            raise ValueError("all finalize tensors must be on one device")
    if not (1 <= top_k <= 64):
        raise ValueError("top_k must be in [1, 64]")
    if rows.shape[0] * (hidden // 2) >= 2**31 or tokens * (hidden // 2) >= 2**31:
        # 32-bit word offsets inside the kernel.
        raise ValueError("finalize rows/output exceed 2^31 32-bit words")
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability(device)
        arguments = (
            make_ptr(
                cutlass.Uint32,
                rows.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            make_ptr(
                cutlass.Int32,
                expanded_idx_to_permuted_idx.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Float32,
                route_weights.data_ptr(),
                cute.AddressSpace.gmem,
                assumed_align=4,
            ),
            make_ptr(
                cutlass.Uint32, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            cutlass.Int32(tokens),
            cutlass.Int32(hidden // 2),
            cutlass.Int32(rows.shape[0] * (hidden // 2)),
        )
        cache_key = (top_k, threads, bool(expanded_rows), device.index, (major, minor))
        stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
        compiled = _finalize_kernel_cache.get(cache_key)
        if compiled is None:
            compiled = cute.compile(
                _FinalizeRows(top_k, threads, bool(expanded_rows)),
                *arguments,
                stream=stream,
            )
            _finalize_kernel_cache[cache_key] = compiled
        bound = _FinalizeRowsPlan(
            compiled,
            arguments,
            (rows, expanded_idx_to_permuted_idx, route_weights, out),
            out,
        )
        bound.run(stream)
        return bound
