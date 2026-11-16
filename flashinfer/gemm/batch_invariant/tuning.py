# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The FlashInfer entries of the GEMM classes: what :func:`flashinfer.mm_bf16`, :func:`flashinfer.bmm_fp8` and
:func:`flashinfer.mm_mxfp8` run under ``batch_invariant=True``.

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
    CLASS_FP4,
    CLASS_FP8,
    CLASS_FP8BS,
    CLASS_MXFP8,
    GEMM_FP8BS_API_OUT_DTYPES,
    MM_FP4_API_OUT_DTYPES,
    MM_MXFP8_API_OUT_DTYPES,
    bmm_fp8_batch_invariant,
    fp4_format_for,
    gemm_fp8_nt_groupwise_batch_invariant,
    native_k_staging,
    mm_bf16_batch_invariant,
    mm_fp4_batch_invariant,
    mm_mxfp8_batch_invariant,
    mx_format_for,
    part_of,
    prepare_bias,
    prepare_bmm_fp8_operands,
    prepare_fp4_operands,
    prepare_fp8_blockscale_operands,
    prepare_mxfp8_operands,
    prepare_operands,
)
from .generator import Knobs
from .schedule import chunks_for, default_config, rank_topk
from .spec import ELEM_BITS_OF, NVFP4_FMTS, sf_bytes

#: The autotuner op names of the classes per API; ``skip_ops={"bf16_gemm_batch_invariant"}`` skips an API's profiling.
CUSTOM_OP = "bf16_gemm_batch_invariant"
CUSTOM_OP_FP8BS = "fp8_blockscale_gemm_batch_invariant"
CUSTOM_OP_BMM_FP8 = "fp8_bmm_batch_invariant"
CUSTOM_OP_MM_MXFP8 = "mxfp8_gemm_batch_invariant"
CUSTOM_OP_MM_FP4 = "fp4_gemm_batch_invariant"
#: The APIs the runner serves and the position of the output in their input lists.
APIS = ("mm_bf16", "bmm_fp8", "mm_mxfp8", "mm_fp4", "gemm_fp8_nt_groupwise")
OUT_INDEX = {
    "mm_bf16": 2,
    "bmm_fp8": 4,
    "mm_mxfp8": 4,
    "mm_fp4": 5,
    "gemm_fp8_nt_groupwise": 4,
}
#: How many of the dispatcher's configurations the autotuner profiles per shape bucket.
TOP_K = 10

__all__ = [
    "CUSTOM_OP",
    "CUSTOM_OP_BMM_FP8",
    "CUSTOM_OP_MM_MXFP8",
    "CUSTOM_OP_MM_FP4",
    "TOP_K",
    "BatchInvariantGemmRunner",
    "BatchInvariantBf16GemmRunner",
    "bmm_fp8_batch_invariant_auto",
    "knobs_from_tactic",
    "knobs_to_tactic",
    "mm_bf16_batch_invariant_auto",
    "mm_mxfp8_batch_invariant_auto",
    "mm_fp4_batch_invariant_auto",
]


def knobs_to_tactic(k: Knobs) -> tuple:
    """A configuration as an autotuner tactic: a flat tuple of scalars (JSON round-trips through the tuner's cache
    files turn lists into tuples and keep ints, bools and strings); the block-scaled knobs (the in-cluster split factor,
    the swapped decode record) last."""
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
        k.cluster_k,
        k.swap_ab,
    )


def knobs_from_tactic(t: Any) -> Knobs:
    """A tactic back to its configuration; a tactic without the two block-scaled knobs (a cache file written before they
    existed) is a dense record."""
    tm, tn, cm, cn, two, tma, raster, sw, grid, cc, *bs = t
    ck, swap = bs if bs else (1, False)
    return Knobs(
        (int(tm), int(tn)),
        (int(cm), int(cn)),
        bool(two),
        bool(tma),
        str(raster),
        int(sw),
        int(grid),
        bool(cc),
        int(ck),
        bool(swap),
    )


def _tuning_config(
    out_index: int, sf_index: Optional[int] = None, in_fmt: str = "mxfp8"
) -> TuningConfig:
    """inputs = [a, b, out, bias] / [A, B, A_scale, B_scale, out] / [a, b, a_descale, b_descale, out] /
    [a, b, a_descale, b_descale, alpha, out]: a's row count (its second-to-last mode, also for the batched operands) is
    the tuned dimension (the hybrid token buckets of the GEMM backends), out follows it, and so does the byte count of A's
    block scales (``sf_index``: the 128x4 swizzled layout of the profiled row count for the format's block size, so the
    kernel never reads past the caller's buffer while a larger bucket is profiled; a's last mode is in the operand's own
    units, bytes of two elements for the 4-bit formats); b is static and lives in the profiling arena so that its layout
    is the caller's."""
    constraints = [ConstraintSpec(out_index, -2, lambda shapes: shapes[0][-2])]
    if sf_index is not None:
        per_unit = (
            2 if ELEM_BITS_OF[in_fmt] == 4 else 1
        )  # elements per unit of a's last mode (byte-packed 4-bit formats)

        def _sf(shapes, fmt: str = in_fmt, per_unit: int = per_unit) -> int:
            return sf_bytes(shapes[0][-2], per_unit * shapes[0][-1], fmt)

        constraints.append(ConstraintSpec(sf_index, 0, _sf))
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
        constraint_specs=tuple(constraints),
    )


TUNING_CONFIG = _tuning_config(OUT_INDEX["mm_bf16"])  # inputs [a, b, out]
#: The block-scale fp8 class: inputs [a (M, K), b (N, K), a_scale (K / 128, M), b_scale (K / 128, ceil(N / 128)), out]; a's
#: row count is the tuned dimension, the output and the MN-layout A scales follow it (their last mode).
TUNING_CONFIG_FP8BS = TuningConfig(
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
    constraint_specs=(
        ConstraintSpec(
            OUT_INDEX["gemm_fp8_nt_groupwise"], -2, lambda shapes: shapes[0][-2]
        ),
        ConstraintSpec(2, -1, lambda shapes: shapes[0][-2]),
    ),
)
#: The same tuning for the K-layout runner (the K-native staging at a few rows): the A scales are (M, K / 128) there, so the
#: row count is their first mode.
TUNING_CONFIG_FP8BS_K = TuningConfig(
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
    constraint_specs=(
        ConstraintSpec(
            OUT_INDEX["gemm_fp8_nt_groupwise"], -2, lambda shapes: shapes[0][-2]
        ),
        ConstraintSpec(2, -2, lambda shapes: shapes[0][-2]),
    ),
)
TUNING_CONFIG_BMM_FP8 = _tuning_config(
    OUT_INDEX["bmm_fp8"]
)  # inputs [A, B, A_scale, B_scale, out]
TUNING_CONFIG_MM_MXFP8 = _tuning_config(
    OUT_INDEX["mm_mxfp8"], sf_index=2
)  # inputs [a, b, a_descale, b_descale, out]
TUNING_CONFIG_MM_FP4 = {  # inputs [a, b, a_descale, b_descale, alpha, out]; the scale bytes per block size
    fmt: _tuning_config(OUT_INDEX["mm_fp4"], sf_index=2, in_fmt=fmt)
    for fmt in ("nvfp4", "mxfp4")
}


class BatchInvariantGemmRunner(TunableRunner):
    """A class as one autotuner runner: fixed API, operand format, weight layout, chunk count (a function of the weight
    shape) and part (SM count and architecture family); the tactics are the top configurations of the dispatcher's heuristic for the
    profiled shape. Any tactic computes the class, so a cached tactic from another bucket or another process is as
    correct as the default; one the kernel cannot implement for the real shape falls back to the default configuration.
    Inputs: ``mm_bf16`` ``[a (M, K), b (K, N), out]``; ``bmm_fp8`` ``[A (b, M, K), B (b, K, N), A_scale, B_scale, out]``;
    ``mm_mxfp8`` ``[a (M, K), b (K, N), a_descale, b_descale, out]`` (the scales as flat uint8 buffers); ``mm_fp4``
    ``[a (M, K / 2), b (K / 2, N), a_descale, b_descale, alpha, out]`` (packed-fp4 bytes, the scales flat, alpha a
    1-element fp32 tensor)."""

    def __init__(
        self,
        api: str,
        in_fmt: str,
        b_major: str,
        chunks: int,
        sm_count: int,
        family: str,
        scale_mode: str = "MN",
    ) -> None:
        assert api in APIS, api
        self.api = api
        self.in_fmt = in_fmt
        self.b_major = b_major
        self.chunks = chunks
        self.sm_count = sm_count
        self.family = family
        # the block-scale class's scale layout as the runner receives it: "MN" (the K layout's copy), or "K" with the
        # K-native staging (a few rows: `api.native_k_staging`); a placement, part of the tuner's cache key
        self.scale_mode = scale_mode

    def _shape(self, inputs: List[torch.Tensor]) -> tuple:
        a, b = inputs[0], inputs[1]
        if self.api == "gemm_fp8_nt_groupwise":
            return (
                a.shape[0],
                b.shape[0],
                a.shape[1],
            )  # the NT operator's weight is (N, K)
        per_unit = (
            2 if ELEM_BITS_OF[self.in_fmt] == 4 else 1
        )  # the 4-bit formats arrive byte-packed: two elements per unit of a's K mode; every other format, one
        return a.shape[-2], b.shape[-1], per_unit * a.shape[-1]

    def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
        return (
            self.api,
            self.in_fmt,
            self.b_major,
            self.chunks,
            self.sm_count,
            self.family,
            self.scale_mode,
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
        if self.api == "mm_mxfp8":
            a, b, sfa, sfb, out = inputs
            return mm_mxfp8_batch_invariant(
                a,
                b,
                sfa,
                sfb,
                out,
                out_dtype=out.dtype,
                config=cfg,
                chunks=self.chunks,
                in_fmt=self.in_fmt,
            )
        if self.api == "mm_fp4":
            a, b, sfa, sfb, alpha, out = inputs
            nv = self.in_fmt in NVFP4_FMTS
            return mm_fp4_batch_invariant(
                a,
                b,
                sfa,
                sfb,
                alpha,
                out.dtype,
                out,
                block_size=16 if nv else 32,
                use_nvfp4=nv,
                config=cfg,
                chunks=self.chunks,
                in_fmt=self.in_fmt,
            )
        if self.api == "gemm_fp8_nt_groupwise":
            # the scales in the runner's layout: MN (K / 128, M) and (K / 128, ceil(N / 128)), or K (M, K / 128) and
            # (ceil(N / 128), K / 128) with the K-native staging
            a, b, sfa, sfb, out = inputs
            return gemm_fp8_nt_groupwise_batch_invariant(
                a,
                b,
                sfa,
                sfb,
                self.scale_mode,
                out,
                out_dtype=out.dtype,
                config=cfg,
                native_k=self.scale_mode == "K",
            )
        a, b, out, bias = (
            inputs  # bias: the (N,) vector or None (the autotuner keeps a None input as is)
        )
        return mm_bf16_batch_invariant(
            a, b, out, bias=bias, config=cfg, chunks=self.chunks
        )


BatchInvariantBf16GemmRunner = (
    BatchInvariantGemmRunner  # the bf16 class's historical name
)


@functools.cache
def _runner(
    api: str,
    in_fmt: str,
    b_major: str,
    chunks: int,
    sm_count: int,
    family: str,
    scale_mode: str = "MN",
) -> BatchInvariantGemmRunner:
    return BatchInvariantGemmRunner(
        api, in_fmt, b_major, chunks, sm_count, family, scale_mode
    )


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
    ``bias`` (bf16, (N,)) selects the biased chain (the bias added once in fp32 to the reduced accumulator before the
    single rounding); ``pdl`` is accepted and launches nothing differently: the class's kernels run in plain stream
    order."""
    if backend != "auto":
        raise unsupported(
            CLASS,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    if out_dtype != torch.bfloat16:
        raise unsupported(CLASS, f"out_dtype={out_dtype}", "bf16 output")
    # every typed check runs before the autotuner sees the shape: it only ever profiles shapes the class accepts
    a, out, b_major, M, N, K = prepare_operands(a, b, out)
    bias = prepare_bias(bias, N, a)
    if M == 0:
        return out
    runner = _runner("mm_bf16", "bf16", b_major, chunks_for(N, K), *part_of(a.device))
    inputs = [a, b, out, bias]
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


def mm_mxfp8_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    use_8x4_sf_layout: bool,
    backend: str,
) -> torch.Tensor:
    """``flashinfer.mm_mxfp8(a, b, a_descale, b_descale, ..., batch_invariant=True)``: the MXFP8 class of the device's
    architecture behind the FlashInfer API's autotuner (the block scales applied inside the instruction; bf16 or fp16
    output; the 128x4 swizzled scale-factor layout only)."""
    if backend != "auto":
        raise unsupported(
            CLASS_MXFP8,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    if use_8x4_sf_layout:
        raise unsupported(
            CLASS_MXFP8,
            "the 8x4 scale-factor layout (use_8x4_sf_layout=True)",
            "the 128x4 swizzled layout (mxfp8_quantize(..., is_sf_swizzled_layout=True))",
        )
    a, sfa, sfb, out, M, N, K = prepare_mxfp8_operands(
        a, b, a_descale, b_descale, out, out_dtype, MM_MXFP8_API_OUT_DTYPES
    )
    if M == 0:
        return out
    in_fmt = mx_format_for(a.device)
    runner = _runner(
        "mm_mxfp8", in_fmt, "k", chunks_for(N, K, in_fmt), *part_of(a.device)
    )
    inputs = [a, b, sfa, sfb, out]
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP_MM_MXFP8, [runner], TUNING_CONFIG_MM_MXFP8, inputs
    )
    chosen(inputs=inputs, tactic=tactic)
    return out


def mm_fp4_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    use_8x4_sf_layout: bool,
    backend: str,
    use_nvfp4: bool,
) -> torch.Tensor:
    """``flashinfer.mm_fp4(a, b, a_descale, b_descale, alpha, ..., batch_invariant=True)``: the NVFP4 (``use_nvfp4=True``)
    or MXFP4 class of the device's architecture and the weight shape behind the FlashInfer API's autotuner (the block
    scales applied inside the instruction, the caller's scalar alpha applied once after the whole fold; bf16 or fp16
    output; the 128x4 swizzled scale-factor layout only; a per-token alpha is not a class of the mode)."""
    if backend != "auto":
        raise unsupported(
            CLASS_FP4,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    if use_8x4_sf_layout:
        raise unsupported(
            CLASS_FP4,
            "the 8x4 scale-factor layout (use_8x4_sf_layout=True)",
            "the 128x4 swizzled layout (nvfp4_quantize(..., sfLayout=SfLayout.layout_128x4, do_shuffle=False) / mxfp4_quantize)",
        )
    if block_size != (16 if use_nvfp4 else 32):
        raise unsupported(
            CLASS_FP4,
            f"block_size={block_size} with use_nvfp4={use_nvfp4}",
            "block_size 16 for NVFP4 and 32 for MXFP4 (the formats' block scales)",
        )
    a, b, sfa, sfb, alpha, out, M, N, K = prepare_fp4_operands(
        a,
        b,
        a_descale,
        b_descale,
        alpha,
        out,
        out_dtype,
        use_nvfp4,
        MM_FP4_API_OUT_DTYPES,
    )
    if M == 0:
        return out
    in_fmt = fp4_format_for(a.device, use_nvfp4, N, K)
    runner = _runner(
        "mm_fp4", in_fmt, "k", chunks_for(N, K, in_fmt), *part_of(a.device)
    )
    inputs = [a, b, sfa, sfb, alpha, out]
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP_MM_FP4,
        [runner],
        TUNING_CONFIG_MM_FP4["nvfp4" if use_nvfp4 else "mxfp4"],
        inputs,
    )
    chosen(inputs=inputs, tactic=tactic)
    return out


def gemm_fp8_nt_groupwise_batch_invariant_auto(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[str],
    mma_sm: int,
    scale_granularity_mnk: tuple,
    out: Optional[torch.Tensor],
    out_dtype: Optional[torch.dtype],
    backend: str,
) -> torch.Tensor:
    """``flashinfer.gemm_fp8_nt_groupwise(a, b, a_scale, b_scale, scale_major_mode, ..., batch_invariant=True)``: the
    block-scale class behind the FlashInfer API's autotuner. The scale tensors reach the runner in the MN layout (a
    K-major tensor is copied once: a layout copy, no arithmetic), except at a few rows in the K layout, where the runner
    takes them as they are for the K-native staging (:func:`native_k_staging`; a placement, the tuner's candidates run
    the production staging); the tuned row count follows ``a``. ``mma_sm`` is a placement of the backends and is ignored;
    the scale granularity must be the class's (1, 128, 128)."""
    if backend != "auto":
        raise unsupported(
            CLASS_FP8BS,
            f"backend={backend!r} together with batch_invariant=True",
            'backend="auto" (the flag selects the batch-invariant class in place of the backends)',
        )
    if tuple(scale_granularity_mnk) != (1, 128, 128):
        raise unsupported(
            CLASS_FP8BS,
            f"scale_granularity_mnk={tuple(scale_granularity_mnk)}",
            "scale_granularity_mnk = (1, 128, 128) (the class's block sizes)",
        )
    if out is not None:
        out_dtype = out.dtype
    out_dtype = out_dtype or torch.bfloat16
    if out_dtype not in GEMM_FP8BS_API_OUT_DTYPES:
        raise unsupported(CLASS_FP8BS, f"out_dtype={out_dtype}", "bf16 or fp16 output")
    a, b, sfa, sfb, out, M, N, K = prepare_fp8_blockscale_operands(
        a, b, a_scale, b_scale, scale_major_mode, out, out_dtype
    )
    if M == 0:
        return out
    native_k = (
        scale_major_mode == "K"
        and native_k_staging(M)
        and sfa.data_ptr() % 16 == 0
        and sfb.data_ptr() % 16 == 0
    )
    if native_k:
        inputs = [
            a,
            b,
            sfa,
            sfb,
            out,
        ]  # the K layout as it is: (M, K / 128) and (ceil(N / 128), K / 128)
        mode = "K"
    else:
        if scale_major_mode == "K":
            sfa = sfa.t().contiguous().t()  # the layout copy, no arithmetic
        sfa_mn = sfa.t()  # (K / 128, M), contiguous: the MN layout
        sfb_mn = sfb.t().contiguous()  # (K / 128, ceil(N / 128))
        inputs = [a, b, sfa_mn, sfb_mn, out]
        mode = "MN"
    runner = _runner(
        "gemm_fp8_nt_groupwise", "e4m3_bs", "k", 1, *part_of(a.device), mode
    )
    chosen, tactic = AutoTuner.get().choose_one(
        CUSTOM_OP_FP8BS,
        [runner],
        TUNING_CONFIG_FP8BS_K if native_k else TUNING_CONFIG_FP8BS,
        inputs,
    )
    chosen(inputs=inputs, tactic=tactic)
    return out
