# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Private T512 weighted BF16 contribution combination; minimum arch SM103.

Each route already includes its FP32 routing/expert scale before BF16 storage.
Sum only local, written expanded rows in FP32; round the caller output once.
"""
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from ...cute_dsl.utils import make_ptr


class _ExpandedWeightedCombine:
    def __init__(self, paired_io: bool):
        self.paired_io = paired_io

    @cute.jit
    def __call__(self, contributions_ptr: cute.Pointer, mapping_ptr: cute.Pointer,
                 output_ptr: cute.Pointer, stream: cuda.CUstream):
        contributions = cute.make_tensor(contributions_ptr,
            cute.make_layout((8192, 7168), stride=(7168, 1)))
        mapping = cute.make_tensor(mapping_ptr,
            cute.make_layout((512, 16), stride=(16, 1)))
        output = cute.make_tensor(output_ptr,
            cute.make_layout((512, 7168), stride=(7168, 1)))
        self.kernel(contributions, mapping, output).launch(
            grid=(512, 1, 1), block=(512, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, contributions: cute.Tensor, mapping: cute.Tensor,
               output: cute.Tensor):
        token, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        lane = tid % 32
        written = cutlass.Boolean(False)
        if lane < 16:
            written = mapping[token, lane] >= 0
        written_mask = cute.arch.vote_ballot_sync(written)
        totals = cute.make_rmem_tensor((14,), cutlass.Float32)
        totals.fill(cutlass.Float32(0.0))
        if cutlass.const_expr(self.paired_io):
            contribution_words = cute.recast_tensor(contributions, cutlass.Uint32)
            output_words = cute.recast_tensor(output, cutlass.Uint32)
            pair = cute.make_rmem_tensor((2,), cutlass.BFloat16)
            pair_word = cute.recast_tensor(pair, cutlass.Uint32)
            if written_mask != 0:
                for slot in cutlass.range_constexpr(16):
                    if (written_mask & (cutlass.Int32(1) << slot)) != 0:
                        for j in cutlass.range_constexpr(7):
                            channel_pair = tid + 512 * j
                            pair_word[0] = contribution_words[token * 16 + slot, channel_pair]
                            totals[2 * j] = totals[2 * j] + pair[0].to(cutlass.Float32)
                            totals[2 * j + 1] = totals[2 * j + 1] + pair[1].to(cutlass.Float32)
            for j in cutlass.range_constexpr(7):
                channel_pair = tid + 512 * j
                pair[0] = totals[2 * j].to(cutlass.BFloat16)
                pair[1] = totals[2 * j + 1].to(cutlass.BFloat16)
                output_words[token, channel_pair] = pair_word[0]
        else:
            if written_mask != 0:
                for slot in cutlass.range_constexpr(16):
                    # Ballot bit preserves the original local/written slot predicate.
                    # Remote/unwritten contribution rows remain unread.
                    if (written_mask & (cutlass.Int32(1) << slot)) != 0:
                        for j in cutlass.range_constexpr(14):
                            channel = tid + 512 * j
                            totals[j] = totals[j] + contributions[token * 16 + slot, channel].to(cutlass.Float32)
            for j in cutlass.range_constexpr(14):
                channel = tid + 512 * j
                output[token, channel] = totals[j].to(cutlass.BFloat16)


class _ExpandedWeightedCombinePlan:
    def __init__(self, compiled, arguments, owners):
        self._compiled, self._arguments, self._owners = compiled, arguments, owners

    def run(self, stream):
        self._compiled(*self._arguments, stream=stream)


def _plan_expanded_weighted_combine(contributions, mapping, output):
    if (contributions.shape != (8192, 7168) or mapping.shape != (512, 16)
            or output.shape != (512, 7168)):
        raise ValueError("private weighted combine requires T512/H7168/top16")
    if (contributions.dtype != torch.bfloat16 or mapping.dtype != torch.int32
            or output.dtype != torch.bfloat16):
        raise TypeError("weighted combine requires BF16 contributions/output and INT32 mapping")
    owners = (contributions, mapping, output)
    if any(not t.is_contiguous() or t.device != output.device for t in owners):
        raise ValueError("weighted combine tensors must be contiguous on the output device")
    with torch.cuda.device(output.device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        paired_io = output.data_ptr() % 4 == 0
        arguments = tuple(make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem,
                                   assumed_align=alignment)
                          for tensor, dtype, alignment in zip(owners,
                              (cutlass.BFloat16, cutlass.Int32, cutlass.BFloat16),
                              (16, 16, 4 if paired_io else 2)))
        compiled = cute.compile(_ExpandedWeightedCombine(paired_io), *arguments, stream=stream)
    return _ExpandedWeightedCombinePlan(compiled, arguments, owners)
