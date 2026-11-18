# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Prepare candidate-domain stages for ThrowawayTopK."""

import torch
import cutlass
from cutlass import testing
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import cuda.bindings.driver as cuda
from ...mqa_logits_util import KernelLaunch
from .scoring import SparseScore
from .compile import compile_call
from .selection import compile_selector
from .packed_recovery import GvrPackedClampWindowKernel


def prepare_stages(self, x, math_mode, logits_dtype):
    self.candidate_error = x["error"]
    self._candidate_inputs = x
    self.sample_bounds = None
    self.sample_fuses_metadata = False
    self.direct_candidates = True
    self.dynamic_metadata = True
    self.materializes_scores = False
    self.logical_indices = True
    self._packed_consumer = (
        math_mode == "native_bf16" and x["weights"].dtype == torch.bfloat16
    )
    if not self._packed_consumer:
        from .typed import prepare_typed_stages

        prepare_typed_stages(self, x, math_mode, logits_dtype)
        return
    m = len(x["q"])
    device = x["q"].device
    self.candidates = torch.empty((m, 8224), device=device, dtype=torch.int32)
    self.thresholds = torch.empty(m, device=device, dtype=torch.float32)
    self.samples = torch.empty((m, 288), device=device, dtype=torch.bfloat16)
    self.queue = torch.zeros(m + 2, device=device, dtype=torch.int32)
    self.scores = torch.empty((m, 16384), device=device, dtype=torch.bfloat16)
    candidate_starts = torch.zeros(m, device=device, dtype=torch.int32)
    self.select_args = (
        self.candidates,
        self.counts,
        candidate_starts,
        x["candidate_valid_lens"],
        self.output,
        self.thresholds,
        self.samples,
        self.queue,
        x["candidate_blocks"],
        self.output,
    )
    self.selector = compile_selector(
        tuple((t.dtype, t.ndim) for t in self.select_args),
        logical_only=True,
        pdl=self.pdl,
        cache_entries=128,
        banks=16,
    )
    self.recovery_args = (
        self.scores,
        x["candidate_valid_lens"],
        self.counts,
        x["candidate_blocks"],
        self.output,
        self.output,
        self.thresholds,
        self.queue,
    )
    recovery = GvrPackedClampWindowKernel(pdl=self.pdl)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    recovery.sms = sms
    self.recovery = compile_call(recovery, self.recovery_args)
    self.score_args = (
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
        self.queue,
        self.scores,
        x["ends"],
        x["error"],
    )
    fake = tuple(from_dlpack(t, assumed_align=16) for t in self.score_args)
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    self.compiled = {
        stage: cute.compile(
            SparseScore(sms * (3 if stage == 1 else 1), pdl=self.pdl, stage=stage),
            *fake,
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
            ("K1", score_launch(1), self.score_args, self.samples, 5),
            ("K2", score_launch(2), self.score_args, self.candidates, 5),
            ("K3", self.selector, self.select_args, self.output, 8),
            ("K4.score", score_launch(4), self.score_args, self.scores, 5),
            ("K4", self.recovery, self.recovery_args, self.output, 3),
        )
    ]


def prepare_live_bounds(self, row_ends, active, starts, ends, score_ends):
    if not self._packed_consumer:
        return
    x = self._candidate_inputs
    bounds = (row_ends, active, starts, ends, score_ends, x["ends"])
    fake = tuple(from_dlpack(t, assumed_align=16) for t in self.score_args)
    fake_bounds = tuple(from_dlpack(t, assumed_align=4) for t in bounds)
    device = x["q"].device
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    stream = cuda.CUstream(torch.cuda.current_stream(device).cuda_stream)
    fn = cute.compile(
        SparseScore(3 * sms, pdl=self.pdl, stage=1),
        *fake,
        stream,
        fake_bounds,
        cutlass.Int32(1),
        options="--enable-tvm-ffi",
    )

    def launch(enabled):
        def call(*args):
            fn(
                *args,
                cuda.CUstream(torch.cuda.current_stream(device).cuda_stream),
                bounds,
                enabled,
            )

        return KernelLaunch(
            call, testing.JitArguments(*self.score_args), self.samples, 5
        )

    self.sample_bounds = tuple(launch(enabled) for enabled in (0, 1))
    self.sample_fuses_metadata = True
