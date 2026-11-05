# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer entries of the GEMM classes: what :func:`flashinfer.mm_bf16` and :func:`flashinfer.bmm_fp8` run under
``batch_invariant=True``.

The flag requires ``backend="auto"``: the class replaces the backends rather than joining them (a backend's kernel
is batch-variant, so the autotuner must never be free to pick one). Within the class the autotuner is free: the
runner's tactics are the dispatcher's top configurations for the shape, every one of which computes the same bits
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
from .api import (
    CLASS,
    CLASS_FP8,
    bmm_fp8_batch_invariant,
    mm_bf16_batch_invariant,
    part_of,
    prepare_bmm_fp8_operands,
    prepare_operands,
)
from .generator import Knobs
from .schedule import chunks_for, default_config, rank_topk

#: The autotuner op names of the classes per API; ``skip_ops={"bf16_gemm_batch_invariant"}`` skips an API's profiling.
CUSTOM_OP = "bf16_gemm_batch_invariant"
CUSTOM_OP_BMM_FP8 = "fp8_bmm_batch_invariant"
#: The APIs the runner serves and the position of the output in their input lists.
APIS = ("mm_bf16", "bmm_fp8")
OUT_INDEX = {"mm_bf16": 2, "bmm_fp8": 4}
#: How many of the dispatcher's configurations the autotuner profiles per shape bucket.
TOP_K = 10

__all__ = [
    "CUSTOM_OP",
    "CUSTOM_OP_BMM_FP8",
    "TOP_K",
    "BatchInvariantGemmRunner",
    "BatchInvariantBf16GemmRunner",
    "bmm_fp8_batch_invariant_auto",
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


def _tuning_config(out_index: int) -> TuningConfig:
    """inputs = [a, b, (the scales,) out]: a's row count (its second-to-last mode, also for the batched operands) is the
    tuned dimension (the hybrid token buckets of the GEMM backends), out follows it; b is static and lives in the profiling
    arena so that its layout is the caller's."""
    return TuningConfig(
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
        constraint_specs=(ConstraintSpec(out_index, -2, lambda shapes: shapes[0][-2]),),
    )


TUNING_CONFIG = _tuning_config(OUT_INDEX["mm_bf16"])  # inputs [a, b, out]
TUNING_CONFIG_BMM_FP8 = _tuning_config(
    OUT_INDEX["bmm_fp8"]
)  # inputs [A, B, A_scale, B_scale, out]


class BatchInvariantGemmRunner(TunableRunner):
    """A class as one autotuner runner: fixed API, operand format, weight layout, chunk count (a function of the weight
    shape) and part (SM count and architecture family); the tactics are the top configurations of the dispatcher's heuristic for the
    profiled shape. Any tactic computes the class, so a cached tactic from another bucket or another process is as
    correct as the default; one the kernel cannot implement for the real shape falls back to the default configuration.
    Inputs: ``mm_bf16`` ``[a (M, K), b (K, N), out]``; ``bmm_fp8`` ``[A (b, M, K), B (b, K, N), A_scale, B_scale, out]``."""

    def __init__(
        self,
        api: str,
        in_fmt: str,
        b_major: str,
        chunks: int,
        sm_count: int,
        family: str,
    ) -> None:
        assert api in APIS, api
        self.api = api
        self.in_fmt = in_fmt
        self.b_major = b_major
        self.chunks = chunks
        self.sm_count = sm_count
        self.family = family

    def _shape(self, inputs: List[torch.Tensor]) -> tuple:
        a, b = inputs[0], inputs[1]
        return a.shape[-2], b.shape[-1], a.shape[-1]

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (
            self.api,
            self.in_fmt,
            self.b_major,
            self.chunks,
            self.sm_count,
            self.family,
        )

    def get_valid_tactics(
        self, inputs: List[torch.Tensor], profile: OptimizationProfile
    ) -> List[Any]:
        M, N, K = self._shape(inputs)
        return [
            knobs_to_tactic(c)
            for c in rank_topk(
                M,
                N,
                K,
                self.chunks,
                self.b_major,
                self.sm_count,
                TOP_K,
                self.family,
                self.in_fmt,
            )
        ]

    def config_for(self, inputs: List[torch.Tensor], tactic: Any) -> Knobs:
        from .blocks.operands import can_implement

        M, N, K = self._shape(inputs)
        if tactic != -1:
            cfg = knobs_from_tactic(tactic)
            if can_implement(cfg, M, N, K, self.b_major, self.chunks, self.in_fmt):
                return cfg
        return default_config(
            M, N, K, self.chunks, self.b_major, self.sm_count, self.family, self.in_fmt
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
        cfg = self.config_for(inputs, tactic)
        if self.api == "bmm_fp8":
            A, B, sa, sb, out = inputs
            return bmm_fp8_batch_invariant(
                A, B, sa, sb, out.dtype, out, config=cfg, chunks=self.chunks
            )
        a, b, out = inputs
        return mm_bf16_batch_invariant(a, b, out, config=cfg, chunks=self.chunks)


BatchInvariantBf16GemmRunner = (
    BatchInvariantGemmRunner  # the bf16 class's historical name
)


@functools.cache
def _runner(
    api: str, in_fmt: str, b_major: str, chunks: int, sm_count: int, family: str
) -> BatchInvariantGemmRunner:
    return BatchInvariantGemmRunner(api, in_fmt, b_major, chunks, sm_count, family)


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
    runner = _runner("mm_bf16", "bf16", b_major, chunks_for(N, K), *part_of(a.device))
    inputs = [a, b, out]
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP, [runner], TUNING_CONFIG, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
    return out


def bmm_fp8_batch_invariant_auto(
    A: torch.Tensor,
    B: torch.Tensor,
    A_scale: torch.Tensor,
    B_scale: torch.Tensor,
    dtype: torch.dtype,
    out: Optional[torch.Tensor],
    backend: str,
) -> torch.Tensor:
    """``flashinfer.bmm_fp8(A, B, A_scale, B_scale, dtype, ..., batch_invariant=True)``: the fp8 classes behind the FlashInfer
    API's autotuner (``alpha = mul.rn(A_scale, B_scale)`` on the device; bf16 or fp16 output)."""
    if backend != "auto":
        raise unsupported(
            CLASS_FP8,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    A, A_scale, B_scale, out, in_fmt, b_major, b, M, N, K = prepare_bmm_fp8_operands(
        A, B, A_scale, B_scale, dtype, out
    )
    if b == 0 or M == 0:
        return out
    runner = _runner(
        "bmm_fp8", in_fmt, b_major, chunks_for(N, K, in_fmt), *part_of(A.device)
    )
    inputs = [A, B, A_scale, B_scale, out]
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP_BMM_FP8, [runner], TUNING_CONFIG_BMM_FP8, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
    return out
