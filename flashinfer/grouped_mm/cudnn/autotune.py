# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-shape autotuning for plain cuDNN grouped GEMM.

Block-scale inputs need offsets and per-expert scale-factor storage to be
regenerated together; they deliberately do not use the plain pre-hook.
"""

import functools

import torch

from ...autotuner import AutoTuner, TunableRunner, TuningConfig
from . import core


def _uniform_offsets(inputs):
    a, b, offsets, out, alpha = inputs
    quotient, remainder = divmod(a.shape[0], b.shape[0])
    counts = torch.full((b.shape[0],), quotient, dtype=torch.int32, device=a.device)
    counts[:remainder] += 1
    offsets = torch.empty_like(offsets)
    offsets[0] = 0
    torch.cumsum(counts, 0, out=offsets[1:])
    return [a, b, offsets, out, alpha]


_TUNING_CONFIG = TuningConfig(
    use_cuda_graph=True, use_cold_l2_cache=True, inputs_pre_hook=_uniform_offsets
)


class CudnnGroupedMmRunner(TunableRunner):
    def __hash__(self):
        return hash(type(self))

    def get_cache_key_extras(self, inputs):
        a = inputs[0]
        props = torch.cuda.get_device_properties(a.device)
        return (
            "cudnn-grouped-mm-v1",
            core._runtime_key(a.device),
            props.name,
            props.major,
            props.minor,
            tuple(
                None if t is None else (str(t.dtype), tuple(t.stride())) for t in inputs
            ),
        )

    def get_valid_tactics(self, inputs, profile):
        a, b, offsets, out, alpha = inputs
        graph = core._execute_cudnn_moe_grouped_gemm(
            a,
            b,
            offsets,
            alpha=alpha,
            out=out,
            out_dtype=out.dtype,
            tactic=0,
            _prepare_only=True,
        )
        # Populate the exact graph -> index map outside the measured path.
        return list(core._plan_indices(graph))

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        a, b, offsets, out, alpha = inputs
        return core._execute_cudnn_moe_grouped_gemm(
            a,
            b,
            offsets,
            alpha=alpha,
            out=out,
            out_dtype=out.dtype,
            tactic=tactic,
        )


@functools.cache
def _runner():
    return CudnnGroupedMmRunner()


def run(a, b, offsets, alpha, out_dtype, out):
    if out is None:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    inputs = [a, b, offsets, out, alpha]
    runner, tactic = AutoTuner.get().choose_one(
        "cudnn_grouped_mm",
        [_runner()],
        _TUNING_CONFIG,
        inputs,
    )
    return runner(inputs, tactic=tactic)
