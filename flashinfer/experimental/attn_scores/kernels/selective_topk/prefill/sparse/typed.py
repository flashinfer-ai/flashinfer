# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Typed candidate scoring and selection using CuTe DSL."""

import torch
import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass import testing
from cutlass.cute.runtime import from_dlpack
from ..dtype_selection import SparseSelector
from ...api import DTYPES
from ...mqa_logits_util import KernelLaunch
from .compile import compile_call
from .scoring import SparseScore


class SampleThreshold:
    @cute.jit
    def __call__(self, samples, lengths, thresholds, counts, stream: cuda.CUstream):
        self.kernel(samples, lengths, thresholds, counts).launch(
            grid=(samples.shape[0], 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.kernel
    def kernel(self, samples, lengths, thresholds, counts):
        row = cute.arch.block_idx()[0]
        tid = cute.arch.thread_idx()[0]
        smem = cutlass.utils.SmemAllocator()
        values = smem.allocate_tensor(cutlass.Float32, cute.make_layout(512))
        values[tid] = cutlass.Float32(samples[row, tid])
        values[tid + 256] = cutlass.Float32(samples[row, tid + 256])
        cute.arch.barrier()
        for level in cutlass.range_constexpr(1, 10):
            for step in cutlass.range_constexpr(level):
                distance = 1 << (level - step - 1)
                left = (tid // distance) * (2 * distance) + tid % distance
                right = left + distance
                a, b = values[left], values[right]
                descending = (left & (1 << level)) == 0
                values[left] = max(a, b) if descending else min(a, b)
                values[right] = min(a, b) if descending else max(a, b)
                cute.arch.barrier()
        if tid == 0:
            expected = cutlass.Float32(512 * 512) / cutlass.Float32(
                max(1, lengths[row])
            )
            estimate = expected + cutlass.Float32(6) * cute.math.sqrt(expected)
            rank = cutlass.Int32(estimate)
            rank = min(511, rank + cutlass.Int32(cutlass.Float32(rank) < estimate))
            thresholds[row] = values[rank]
            counts[row] = 0


def prepare_typed_stages(self, x, math_mode, logits_dtype):
    m = len(x["q"])
    device = x["q"].device
    self.samples = torch.empty((m, 512), device=device, dtype=logits_dtype)
    self.thresholds = torch.empty(m, device=device, dtype=torch.float32)
    self.candidates = torch.empty((m, 4096), device=device, dtype=torch.int64)
    self.scores = torch.empty((m, 16384), device=device, dtype=logits_dtype)
    self.args = (
        x["q"],
        x["q_scale"].view(torch.int32).reshape(-1, 32),
        x["k"],
        x["k_scale"].view(torch.int32).flatten(),
        x["weights"],
        x["candidate_blocks"],
        x["candidate_valid_lens"],
        self.thresholds,
        self.candidates,
        self.samples,
        self.counts,
        self.scores,
        x["ends"],
        x["error"],
    )
    self.select_args = (
        self.candidates,
        self.counts,
        x["candidate_valid_lens"],
        x["candidate_blocks"],
        self.output,
    )
    self.recovery_args = (self.scores, *self.select_args[1:])
    self.selector = compile_call(SparseSelector(DTYPES[logits_dtype]), self.select_args)
    self.recovery = compile_call(
        SparseSelector(DTYPES[logits_dtype], recovery=True), self.recovery_args
    )
    threshold_args = (
        self.samples,
        x["candidate_valid_lens"],
        self.thresholds,
        self.counts,
    )
    threshold = compile_call(SampleThreshold(), threshold_args)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    tensors = tuple(from_dlpack(t, assumed_align=16) for t in self.args)
    self.compiled = {
        stage: cute.compile(
            SparseScore(
                sms * (3 if stage == 1 else 1),
                pdl=self.pdl,
                stage=stage,
                math_mode=math_mode,
                weight_bytes=x["weights"].element_size(),
            ),
            *tensors,
            stream,
            options="--enable-tvm-ffi",
        )
        for stage in (1, 2, 4)
    }

    def score_launch(stage):
        fn = self.compiled[stage]

        def launch(*args):
            fn(*args, cuda.CUstream(torch.cuda.current_stream(device).cuda_stream))

        return launch

    self.stages = [
        (name, KernelLaunch(fn, testing.JitArguments(*args), output, block_arg))
        for name, fn, args, output, block_arg in (
            ("K1", score_launch(1), self.args, self.samples, 5),
            ("K1.threshold", threshold, threshold_args, self.thresholds, None),
            ("K2", score_launch(2), self.args, self.candidates, 5),
            ("K3", self.selector, self.select_args, self.output, 3),
            ("K4.score", score_launch(4), self.args, self.scores, 5),
            ("K4", self.recovery, self.recovery_args, self.output, 3),
        )
    ]
