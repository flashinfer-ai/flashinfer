# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""BF16 top-k reduction used by the SM120 MegaMoE form-A path."""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Float32, Int32


BF16_VECTOR_THREADS = 512
BF16_HIDDEN_PER_THREAD = 8

@cute.kernel
def topk_reduce_bf16_vec_kernel(
    combine_output: cute.Tensor,
    topk_score: Optional[cute.Tensor],
    reduced_output: cute.Tensor,
    num_topk: cutlass.Constexpr[int],
    hidden: cutlass.Constexpr[int],
    store_dtype: cutlass.Constexpr[str],
):
    """BF16 reduce with one thread handling one 8-hidden vector."""

    hidden_vec_block_idx, token_idx, _ = cute.arch.block_idx()
    tid = cute.arch.thread_idx()[0]
    block_dim = cute.arch.block_dim()[0]
    vec_idx = hidden_vec_block_idx * block_dim + tid
    base_h = vec_idx * Int32(BF16_HIDDEN_PER_THREAD)

    if base_h < Int32(hidden):
        acc = cute.make_rmem_tensor((BF16_HIDDEN_PER_THREAD,), cutlass.Float32)
        for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
            acc[i] = Float32(0.0)

        copy_atom_bf16_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.BFloat16,
            num_bits_per_copy=128,
        )

        for k in cutlass.range_constexpr(0, num_topk, 1):
            score_value = Float32(1.0)
            if cutlass.const_expr(topk_score is not None):
                score_value = Float32(topk_score[token_idx, Int32(k)])
            score_pair = (score_value, score_value)

            in_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD,),
                cutlass.BFloat16,
            )
            in_row = combine_output[token_idx, Int32(k), None]
            in_tile = cute.local_tile(
                in_row,
                (BF16_HIDDEN_PER_THREAD,),
                (base_h // Int32(BF16_HIDDEN_PER_THREAD),),
            )
            in_aligned_iter = cute.make_ptr(
                in_tile.element_type,
                in_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            in_tile = cute.make_tensor(in_aligned_iter, in_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(in_tile),
                cute.coalesce(in_regs),
            )

            for pair_i in cutlass.range_constexpr(
                0,
                BF16_HIDDEN_PER_THREAD // 2,
                1,
            ):
                val_pair = (
                    Float32(in_regs[2 * pair_i]),
                    Float32(in_regs[2 * pair_i + 1]),
                )
                old_acc_pair = (acc[2 * pair_i], acc[2 * pair_i + 1])
                if cutlass.const_expr(topk_score is not None):
                    acc_pair = cute.arch.fma_packed_f32x2(
                        val_pair,
                        score_pair,
                        old_acc_pair,
                    )
                else:
                    acc_pair = cute.arch.add_packed_f32x2(
                        old_acc_pair,
                        val_pair,
                    )
                acc[2 * pair_i] = acc_pair[0]
                acc[2 * pair_i + 1] = acc_pair[1]

        out_row = reduced_output[token_idx, None]
        out_tile = cute.local_tile(
            out_row,
            (BF16_HIDDEN_PER_THREAD,),
            (base_h // Int32(BF16_HIDDEN_PER_THREAD),),
        )
        if cutlass.const_expr(store_dtype == "bf16"):
            out_regs = cute.make_rmem_tensor(
                (BF16_HIDDEN_PER_THREAD,),
                cutlass.BFloat16,
            )
            out_regs.store(acc.load().to(cutlass.BFloat16))
            out_aligned_iter = cute.make_ptr(
                out_tile.element_type,
                out_tile.iterator.toint(),
                AddressSpace.gmem,
                assumed_align=16,
            )
            out_tile = cute.make_tensor(out_aligned_iter, out_tile.layout)
            cute.copy(
                copy_atom_bf16_vec,
                cute.coalesce(out_regs),
                cute.coalesce(out_tile),
            )
        else:
            for i in cutlass.range_constexpr(0, BF16_HIDDEN_PER_THREAD, 1):
                out_tile[i] = acc[i]

def _validate_tensors(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
) -> Tuple[int, int, int]:
    if combine_output.dim() != 3:
        raise ValueError(
            f"combine_output must have shape (T, K, H), got "
            f"{tuple(combine_output.shape)}."
        )
    if reduced_output.dim() != 2:
        raise ValueError(
            f"reduced_output must have shape (T, H), got "
            f"{tuple(reduced_output.shape)}."
        )
    if combine_output.dtype != torch.bfloat16:
        raise TypeError(f"combine_output must be BF16, got {combine_output.dtype}.")
    if reduced_output.dtype != torch.bfloat16:
        raise TypeError(f"reduced_output must be BF16, got {reduced_output.dtype}.")
    if not combine_output.is_cuda or not reduced_output.is_cuda:
        raise ValueError("combine_output and reduced_output must be CUDA tensors.")
    if combine_output.device != reduced_output.device:
        raise ValueError("combine_output and reduced_output must share a device.")

    tokens, num_topk, hidden = map(int, combine_output.shape)
    if tokens <= 0 or num_topk <= 0 or hidden <= 0:
        raise ValueError(f"combine_output dimensions must be positive: {combine_output.shape}.")
    if tuple(reduced_output.shape) != (tokens, hidden):
        raise ValueError(
            f"reduced_output shape must be {(tokens, hidden)}, got "
            f"{tuple(reduced_output.shape)}."
        )
    if hidden % BF16_HIDDEN_PER_THREAD != 0:
        raise ValueError(
            f"hidden ({hidden}) must be divisible by {BF16_HIDDEN_PER_THREAD}."
        )
    if combine_output.stride(-1) != 1 or reduced_output.stride(-1) != 1:
        raise ValueError("top-k reduction requires contiguous hidden dimensions.")
    if combine_output.stride(-2) % BF16_HIDDEN_PER_THREAD != 0:
        raise ValueError("combine_output top-k rows must preserve 16-byte alignment.")
    if reduced_output.stride(0) % BF16_HIDDEN_PER_THREAD != 0:
        raise ValueError("reduced_output rows must preserve 16-byte alignment.")

    if topk_score is not None:
        if topk_score.shape != (tokens, num_topk):
            raise ValueError(
                f"topk_score shape must be {(tokens, num_topk)}, got "
                f"{tuple(topk_score.shape)}."
            )
        if topk_score.dtype != torch.float32 or not topk_score.is_cuda:
            raise TypeError("topk_score must be a CUDA FP32 tensor.")
        if topk_score.device != combine_output.device:
            raise ValueError("topk_score must share combine_output's device.")
    return tokens, num_topk, hidden


def _infer_assumed_align(tensor: torch.Tensor, max_align: int = 16) -> int:
    ptr = int(tensor.data_ptr())
    for align in (16, 8, 4, 2, 1):
        if align <= max_align and ptr % align == 0:
            return align
    return 1


def _to_cute_tensor(tensor: torch.Tensor) -> cute.Tensor:
    cute_tensor = cutlass_torch.from_dlpack(
        tensor, assumed_align=_infer_assumed_align(tensor)
    )
    leading_dim = cutlass_torch.get_leading_dim(tensor)
    return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)


def compile_topk_reduce(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    threads: int = BF16_VECTOR_THREADS,
    stream: Optional[cuda.CUstream] = None,
):
    """Compile the shape-specialized BF16 top-k reduction launcher."""
    tokens, num_topk, hidden = _validate_tensors(
        combine_output, reduced_output, topk_score
    )
    if threads <= 0:
        raise ValueError(f"threads must be positive, got {threads}.")
    if stream is None:
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

    combine_cute = _to_cute_tensor(combine_output)
    reduced_cute = _to_cute_tensor(reduced_output)
    topk_score_cute = _to_cute_tensor(topk_score) if topk_score is not None else None
    hidden_blocks = (
        hidden + threads * BF16_HIDDEN_PER_THREAD - 1
    ) // (threads * BF16_HIDDEN_PER_THREAD)
    launch_grid = [hidden_blocks, tokens, 1]

    @cute.jit
    def _launcher(
        combine_cute: cute.Tensor,
        reduced_cute: cute.Tensor,
        topk_score_cute: Optional[cute.Tensor],
        stream: cuda.CUstream,
    ):
        topk_reduce_bf16_vec_kernel(
            combine_cute,
            topk_score_cute,
            reduced_cute,
            num_topk=num_topk,
            hidden=hidden,
            store_dtype="bf16",
        ).launch(
            grid=launch_grid,
            block=[threads, 1, 1],
            stream=stream,
        )

    compiled = cute.compile(
        _launcher,
        combine_cute,
        reduced_cute,
        topk_score_cute,
        stream,
    )
    return compiled, combine_cute, reduced_cute, topk_score_cute, stream


def launch_compiled_topk_reduce(
    compiled,
    combine_cute: cute.Tensor,
    reduced_cute: cute.Tensor,
    topk_score_cute: Optional[cute.Tensor],
    stream: cuda.CUstream,
    *,
    synchronize: bool = False,
) -> None:
    compiled(
        combine_cute=combine_cute,
        reduced_cute=reduced_cute,
        topk_score_cute=topk_score_cute,
        stream=stream,
    )
    if synchronize:
        torch.cuda.synchronize()


def run_topk_reduce(
    combine_output: torch.Tensor,
    reduced_output: torch.Tensor,
    topk_score: Optional[torch.Tensor] = None,
    *,
    threads: int = BF16_VECTOR_THREADS,
    stream: Optional[cuda.CUstream] = None,
    synchronize: bool = False,
) -> None:
    plan = compile_topk_reduce(
        combine_output,
        reduced_output,
        topk_score,
        threads=threads,
        stream=stream,
    )
    launch_compiled_topk_reduce(*plan, synchronize=synchronize)
