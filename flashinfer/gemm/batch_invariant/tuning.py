# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer entry of the bf16 GEMM class: what :func:`flashinfer.mm_bf16` runs under ``batch_invariant=True``.

The flag requires ``backend="auto"``: the class replaces the backends rather than joining them (a backend's kernel
is batch-variant, so the autotuner must never be free to pick one). Within the class the autotuner is free: the
runner's tactics are the cost model's top configurations for the shape, every one of which computes the same bits
(R3), so a tuned and an untuned process agree bit for bit and differ in speed only. Tactic ``-1`` is the dispatcher's
default configuration.
"""

from __future__ import annotations

import functools
from typing import Any, List, Optional

import torch

from ...autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ...fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ...batch_invariant.support import unsupported
from .api import CLASS, mm_bf16_batch_invariant, part_of, prepare_operands
from .generator import Knobs
from .schedule import chunks_for, default_config, rank_topk

#: The autotuner op name of the class; ``skip_ops={"bf16_gemm_batch_invariant"}`` skips its profiling.
CUSTOM_OP = "bf16_gemm_batch_invariant"
#: How many of the cost model's configurations the autotuner profiles per shape bucket.
TOP_K = 10

__all__ = [
    "CUSTOM_OP",
    "TOP_K",
    "BatchInvariantBf16GemmRunner",
    "knobs_from_tactic",
    "knobs_to_tactic",
    "mm_bf16_batch_invariant_auto",
]


def knobs_to_tactic(k: Knobs) -> tuple:
    """A configuration as an autotuner tactic: a flat tuple of scalars (JSON round-trips through the tuner's cache
    files turn lists into tuples and keep ints, bools and strings)."""
    return (
        k.tile_m,
        k.tile_n,
        k.cluster_m,
        k.cluster_n,
        k.use_2cta_instrs,
        k.use_tma_store,
        k.raster_along,
        k.swizzle_size,
        k.grid_clusters,
        k.cta_chunks,
    )


def knobs_from_tactic(t: Any) -> Knobs:
    tm, tn, cm, cn, two, tma, raster, sw, grid, cc = t
    return Knobs(
        (int(tm), int(tn)),
        (int(cm), int(cn)),
        bool(two),
        bool(tma),
        str(raster),
        int(sw),
        int(grid),
        bool(cc),
    )


# inputs = [a, b, out]: a's row count is the tuned dimension (the hybrid token buckets of the bf16 GEMM backends),
# out follows it; b is static and lives in the profiling arena so that its layout is the caller's.
TUNING_CONFIG = TuningConfig(
    use_cuda_graph=True,
    use_cold_l2_graph_replay=True,
    use_cold_l2_cache=True,
    cold_chain_launches=8,
    profile_arena_input_indices=(1,),
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),
            (-2,),
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(ConstraintSpec(2, -2, lambda shapes: shapes[0][-2]),),
)


class BatchInvariantBf16GemmRunner(TunableRunner):
    """The class as one autotuner runner: fixed weight layout, chunk count (a function of the weight shape) and part (SM
    count and architecture family); the tactics are the top configurations of the cost model for the profiled
    shape. Any tactic computes the class, so a cached tactic from another bucket or another process is as correct as
    the default; one the kernel cannot implement for the real shape falls back to the default configuration."""

    def __init__(self, b_major: str, chunks: int, sm_count: int, family: str) -> None:
        self.b_major = b_major
        self.chunks = chunks
        self.sm_count = sm_count
        self.family = family

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (self.b_major, self.chunks, self.sm_count, self.family)

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile
    ) -> List[Any]:
        a, b, _ = inputs
        M, K = a.shape
        N = b.shape[1]
        return [
            knobs_to_tactic(c)
            for c in rank_topk(
                M, N, K, self.chunks, self.b_major, self.sm_count, TOP_K, self.family
            )
        ]

    def config_for(self, a: torch.Tensor, b: torch.Tensor, tactic: Any) -> Knobs:
        from .blocks.operands import can_implement

        M, K = a.shape
        N = b.shape[1]
        if tactic != -1:
            cfg = knobs_from_tactic(tactic)
            if can_implement(cfg, M, N, K, self.b_major, self.chunks):
                return cfg
        return default_config(
            M, N, K, self.chunks, self.b_major, self.sm_count, self.family
        )

    def precompile_tactics(
        self,
        inputs: List[torch.Tensor],
        tactics: List[Any],
        profile: OptimizationProfile,
        **kwargs,
    ) -> bool:
        """Compile and run every candidate (and the default) once, eagerly: the autotuner profiles under CUDA-graph
        capture, where a first call's JIT compile and module load are not allowed, so an uncompiled tactic would be
        dropped as "unsupported" instead of being timed."""
        for tactic in [-1, *tactics]:
            self.forward(inputs, tactic=tactic)
        torch.cuda.synchronize()
        return True

    def forward(
        self,
        inputs: List[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        a, b, out = inputs
        return mm_bf16_batch_invariant(
            a, b, out, config=self.config_for(a, b, tactic), chunks=self.chunks
        )


@functools.cache
def _runner(
    b_major: str, chunks: int, sm_count: int, family: str
) -> BatchInvariantBf16GemmRunner:
    return BatchInvariantBf16GemmRunner(b_major, chunks, sm_count, family)


def mm_bf16_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: Optional[torch.Tensor],
    pdl: Optional[bool],
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    backend: str,
) -> torch.Tensor:
    """``flashinfer.mm_bf16(a, b, ..., batch_invariant=True)``: the class behind the FlashInfer API's autotuner.
    ``pdl`` is accepted and launches nothing differently: the class's kernels run in plain stream order."""
    if backend != "auto":
        raise unsupported(
            CLASS,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    if bias is not None:
        raise unsupported(CLASS, "a bias", "bias=None (no fused epilogue)")
    if out_dtype != torch.bfloat16:
        raise unsupported(CLASS, f"out_dtype={out_dtype}", "bf16 output")
    # every typed check runs before the autotuner sees the shape: it only ever profiles shapes the class accepts
    a, out, b_major, M, N, K = prepare_operands(a, b, out)
    if M == 0:
        return out
    runner = _runner(b_major, chunks_for(N, K), *part_of(a.device))
    inputs = [a, b, out]
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP, [runner], TUNING_CONFIG, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
    return out
