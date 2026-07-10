#!/usr/bin/env python3
"""Benchmark TRT-LLM DA-versus-NoDA MoE execution across precisions.

The legacy bundle-driven command line remains supported for existing
experiments. Each precision records its full launcher context in the bundle,
so a saved artifact is only reused when it matches the runtime configuration.
The DA-enabled graph must increment the capture dispatch counter; otherwise
the row fails instead of silently measuring the original path.
"""

from __future__ import annotations

import argparse
import gc
import pickle
import csv
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch

REPO_ROOT = str(Path(__file__).resolve().parents[1])
BENCH_DIR = str(Path(REPO_ROOT) / "benchmarks")
for path in (REPO_ROOT, BENCH_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

from bench_moe_deepseek import CFG
from flashinfer import (
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fp4_quantization import block_scale_interleave, fp4_quantize
from flashinfer.fused_moe import (
    ActivationType,
    Fp8QuantizationType,
    RoutingMethodType,
    WeightLayout,
    convert_to_block_layout,
    trtllm_bf16_moe,
    trtllm_bf16_routed_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_moe,
    trtllm_fp8_block_scale_routed_moe,
    trtllm_fp8_per_tensor_scale_moe,
    trtllm_mxint4_block_scale_moe,
    trtllm_mxint4_block_scale_routed_moe,
)
from flashinfer.fused_moe import core as moe_core
from flashinfer.fused_moe.dist_aware import da_capture, da_core, da_profile, da_state
from flashinfer.fused_moe.core import (
    _maybe_get_cached_w3_w1_permute_indices,
    get_w2_permute_indices_with_cache,
)
from flashinfer.fused_moe.utils import get_hybrid_num_tokens_buckets
from flashinfer.fused_moe.dist_aware.da_utils import (
    generate_da_distribution_assignments,
    get_da_distribution_specs,
)
from flashinfer.tllm_enums import DtypeTrtllmGen


FLOAT8_E4M3_MAX = 448.0
BENCHMARK_MODES = {
    "bf16": 0.925,
    "fp8_per_tensor": 0.92,
    "fp8_block": 0.79,
    "mxfp8": 0.79,
    "nvfp4": 0.92,
    "mxfp4_mxfp8": 0.92,
    "mxfp4_bf16": 0.92,
    "mxint4": 0.925,
}
# Keep these capability sets synchronized with the routed wrapper contracts in
# flashinfer/fused_moe/core.py. FP8 per-tensor has no public routed wrapper;
# only the FP4 wrapper accepts separate raw IDs and BF16 weights.
ROUTED_API_PRECISION_MODES = frozenset(BENCHMARK_MODES) - {"fp8_per_tensor"}
UNPACKED_PRECOMPUTED_PRECISION_MODES = frozenset({"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"})


# ---------------------------------------------------------------------------
# NVTX helpers
# ---------------------------------------------------------------------------

_nvtx_enabled = False
_cuda_profiler_da_phase_enabled = False


@contextmanager
def nvtx_range(msg: str):
    """Annotate a benchmark phase when NVTX output was requested."""
    if _nvtx_enabled:
        torch.cuda.nvtx.range_push(msg)
    try:
        yield
    finally:
        if _nvtx_enabled:
            torch.cuda.nvtx.range_pop()


@contextmanager
def cuda_profiler_da_phase():
    """Bracket DA tuning for an opt-in Nsight CUDA-profiler-API capture."""
    if _cuda_profiler_da_phase_enabled:
        print("[nsys] cudaProfilerStart phase=da-autotune", flush=True)
        torch.cuda.profiler.start()
    try:
        yield
    finally:
        if _cuda_profiler_da_phase_enabled:
            torch.cuda.profiler.stop()
            print("[nsys] cudaProfilerStop phase=da-autotune", flush=True)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BenchConfig:
    num_tokens: int
    num_experts: int
    local_num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    n_group: int
    topk_group: int
    routed_scaling_factor: float
    tune_max_num_tokens: int


@dataclass
class PrecisionCase:
    name: str
    op_name: str
    dtype_act: DtypeTrtllmGen
    dtype_weights: DtypeTrtllmGen
    quantization_type: Fp8QuantizationType
    weight_layout: WeightLayout
    use_shuffled_weight: bool
    use_per_token_scaling: bool
    match_ratio: float
    call: Callable[[Any], torch.Tensor]


# ---------------------------------------------------------------------------
# Weight / state preparation
# ---------------------------------------------------------------------------


def _as_tensor(result):
    if isinstance(result, (list, tuple)):
        return result[0]
    return result


def _fp8_quantize_per_expert(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = x.float().reshape(x.shape[0], -1)
    amax = flat.abs().amax(dim=1).clamp(min=1.0e-6)
    scale = FLOAT8_E4M3_MAX / amax
    view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    q = (x.float() * scale.reshape(view_shape)).to(torch.float8_e4m3fn)
    return q, scale.reciprocal().to(torch.float32)


def _fp8_quantize_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    amax = x.float().abs().amax().clamp(min=1.0e-6)
    scale = FLOAT8_E4M3_MAX / amax
    return (x.float() * scale).to(torch.float8_e4m3fn), scale.reciprocal()


def _fp8_block_quantize(
    x: torch.Tensor,
    row_block: int,
    col_block: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 2:
        rows, cols = x.shape
        view = x.float().reshape(
            rows // row_block, row_block, cols // col_block, col_block
        )
        amax = view.abs().amax(dim=(1, 3)).clamp(min=1.0e-6)
        scale = FLOAT8_E4M3_MAX / amax
        q = (view * scale[:, None, :, None]).reshape(rows, cols).to(torch.float8_e4m3fn)
        return q, scale.reciprocal().to(torch.float32)

    experts, rows, cols = x.shape
    view = x.float().reshape(
        experts, rows // row_block, row_block, cols // col_block, col_block
    )
    amax = view.abs().amax(dim=(2, 4)).clamp(min=1.0e-6)
    scale = FLOAT8_E4M3_MAX / amax
    q = (
        (view * scale[:, :, None, :, None])
        .reshape(experts, rows, cols)
        .to(torch.float8_e4m3fn)
    )
    return q, scale.reciprocal().to(torch.float32)


def _mxint4_quantize(
    x: torch.Tensor, sf_vec_size: int = 32
) -> tuple[torch.Tensor, torch.Tensor]:
    x_reshaped = x.float().reshape(-1, sf_vec_size)
    x_max = x_reshaped.max(dim=-1, keepdim=True)[0] * 8.0 / 7.0
    x_min = x_reshaped.min(dim=-1, keepdim=True)[0]
    amax = torch.where(x_max > -x_min, x_max, -x_min).clamp(min=1.0e-6)
    scales = amax / 8.0
    x_scaled = x_reshaped * scales.reciprocal()
    x_int8 = (
        x_scaled.round().clamp(-8, 7).to(torch.int8).reshape(-1, sf_vec_size // 2, 2)
    )
    x_int4 = (x_int8[..., 0] & 0x0F) | ((x_int8[..., 1] & 0x0F) << 4)
    return x_int4.reshape(*x.shape[:-1], x.shape[-1] // 2).view(
        torch.uint8
    ), scales.reshape(-1, sf_vec_size)


def _prepare_bf16_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w1_bf16 = w1.to(torch.bfloat16)
    w2_bf16 = w2.to(torch.bfloat16)
    gemm1 = []
    gemm2 = []
    for expert in range(w1.shape[0]):
        tmp1 = reorder_rows_for_gated_act_gemm(w1_bf16[expert]).view(torch.uint8)
        tmp1 = shuffle_matrix_a(tmp1, 128)
        tmp1 = convert_to_block_layout(tmp1, 128).view(torch.bfloat16)

        tmp2 = shuffle_matrix_a(w2_bf16[expert].view(torch.uint8), 128)
        tmp2 = convert_to_block_layout(tmp2, 128).view(torch.bfloat16)
        gemm1.append(tmp1)
        gemm2.append(tmp2)
    return torch.stack(gemm1).contiguous(), torch.stack(gemm2).contiguous()


def _prepare_fp8_per_tensor_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_fp8, w1_deq = _fp8_quantize_per_expert(w1)
    w2_fp8, w2_deq = _fp8_quantize_per_expert(w2)
    gemm1 = []
    gemm2 = []
    for expert in range(w1.shape[0]):
        tmp1 = reorder_rows_for_gated_act_gemm(w1_fp8[expert]).view(torch.uint8)
        tmp1 = shuffle_matrix_a(tmp1, 128).view(torch.float8_e4m3fn)
        tmp2 = shuffle_matrix_a(w2_fp8[expert].view(torch.uint8), 128).view(
            torch.float8_e4m3fn
        )
        gemm1.append(tmp1)
        gemm2.append(tmp2)
    return (
        torch.stack(gemm1).contiguous(),
        w1_deq,
        w1_deq,
        torch.stack(gemm2).contiguous(),
        w2_deq,
    )


def _prepare_mxint4_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    cfg: BenchConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_int4, w1_scales = _mxint4_quantize(w1, 32)
    w2_int4, w2_scales = _mxint4_quantize(w2, 32)
    w1_scales = w1_scales.to(torch.bfloat16).reshape(
        cfg.local_num_experts, 2 * cfg.intermediate_size, cfg.hidden_size // 32
    )
    w2_scales = w2_scales.to(torch.bfloat16).reshape(
        cfg.local_num_experts, cfg.hidden_size, cfg.intermediate_size // 32
    )

    cache = {}
    gemm1_w, gemm1_s, gemm2_w, gemm2_s = [], [], [], []
    for expert in range(cfg.local_num_experts):
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, w1_int4[expert].view(torch.uint8), 128
        )
        tmp_w1 = w1_int4[expert].view(torch.uint8)[perm].contiguous()

        perm_s = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            w1_scales[expert].view(torch.bfloat16),
            128,
            num_elts_per_sf=32,
        )
        tmp_s1 = block_scale_interleave(
            w1_scales[expert].view(torch.bfloat16)[perm_s].contiguous()
        )

        perm = get_w2_permute_indices_with_cache(
            cache, w2_int4[expert].view(torch.uint8), 128
        )
        tmp_w2 = w2_int4[expert].view(torch.uint8)[perm].contiguous()

        perm_s = get_w2_permute_indices_with_cache(
            cache,
            w2_scales[expert].view(torch.bfloat16),
            128,
            num_elts_per_sf=16,
        )
        tmp_s2 = block_scale_interleave(
            w2_scales[expert].view(torch.bfloat16)[perm_s].contiguous()
        )

        gemm1_w.append(convert_to_block_layout(tmp_w1, 128).contiguous())
        gemm2_w.append(convert_to_block_layout(tmp_w2, 128).contiguous())
        gemm1_s.append(tmp_s1)
        gemm2_s.append(tmp_s2)

    return (
        torch.stack(gemm1_w).contiguous(),
        torch.stack(gemm1_s).view(torch.bfloat16).contiguous(),
        torch.stack(gemm2_w).contiguous(),
        torch.stack(gemm2_s).view(torch.bfloat16).contiguous(),
    )


def _mxfp8_quantize_batches(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_batches = []
    s_batches = []
    for expert in range(x.shape[0]):
        q, s = mxfp8_quantize(x[expert].to(torch.bfloat16), False)
        q_batches.append(q)
        s_batches.append(s.view(torch.uint8).reshape(x.shape[1], -1))
    return torch.stack(q_batches).contiguous(), torch.stack(s_batches).contiguous()


def _prepare_fp8_mxfp8_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    cfg: BenchConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    w1_fp8, w1_scale = _mxfp8_quantize_batches(w1)
    w2_fp8, w2_scale = _mxfp8_quantize_batches(w2)

    epilogue_tile_m = 128
    gemm1, gemm1_scale, gemm2, gemm2_scale = [], [], [], []
    for expert in range(cfg.local_num_experts):
        tmp_w1 = reorder_rows_for_gated_act_gemm(w1_fp8[expert]).view(torch.uint8)
        tmp_s1 = reorder_rows_for_gated_act_gemm(w1_scale[expert]).view(torch.uint8)
        gemm1.append(
            shuffle_matrix_a(tmp_w1, epilogue_tile_m)
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        gemm1_scale.append(
            shuffle_matrix_sf_a(tmp_s1, epilogue_tile_m).contiguous().view(torch.uint8)
        )

        gemm2.append(
            shuffle_matrix_a(w2_fp8[expert].view(torch.uint8), epilogue_tile_m)
            .contiguous()
            .view(torch.float8_e4m3fn)
        )
        gemm2_scale.append(
            shuffle_matrix_sf_a(
                w2_scale[expert].view(torch.uint8).reshape(cfg.hidden_size, -1),
                epilogue_tile_m,
            )
            .contiguous()
            .view(torch.uint8)
        )

    return (
        torch.stack(gemm1).contiguous(),
        torch.stack(gemm1_scale)
        .reshape(
            cfg.local_num_experts,
            2 * cfg.intermediate_size,
            cfg.hidden_size // 32,
        )
        .contiguous(),
        torch.stack(gemm2).contiguous(),
        torch.stack(gemm2_scale)
        .reshape(
            cfg.local_num_experts,
            cfg.hidden_size,
            cfg.intermediate_size // 32,
        )
        .contiguous(),
    )


def _prepare_fp4_weights(
    w1: torch.Tensor,
    w2: torch.Tensor,
    cfg: BenchConfig,
    *,
    sf_vec_size: int,
    sf_use_ue8m0: bool,
    global_sf: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cache = {}
    device = w1.device
    global_scale = torch.tensor([1.0 / global_sf], dtype=torch.float32, device=device)

    def _quant_weight(t: torch.Tensor, rows: int, cols: int):
        q, s = fp4_quantize(
            t.to(torch.bfloat16),
            global_scale,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            is_sf_swizzled_layout=False,
        )
        q = q.view(torch.uint8).reshape(rows, cols // 2)
        s = s.view(torch.float8_e4m3fn).reshape(rows, cols // sf_vec_size)
        return q, s

    gemm1_w, gemm1_s, gemm2_w, gemm2_s = [], [], [], []
    for expert in range(cfg.local_num_experts):
        q1, s1 = _quant_weight(w1[expert], 2 * cfg.intermediate_size, cfg.hidden_size)
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, q1, 128, is_gated_act_gemm=True
        )
        gemm1_w.append(q1[perm.to(device)].contiguous())
        perm_s = _maybe_get_cached_w3_w1_permute_indices(
            cache,
            s1.view(torch.uint8),
            128,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        gemm1_s.append(
            block_scale_interleave(s1.view(torch.uint8)[perm_s.to(device)].contiguous())
        )

        q2, s2 = _quant_weight(w2[expert], cfg.hidden_size, cfg.intermediate_size)
        perm = get_w2_permute_indices_with_cache(cache, q2, 128)
        gemm2_w.append(q2[perm.to(device)].contiguous())
        perm_s = get_w2_permute_indices_with_cache(
            cache,
            s2.view(torch.uint8),
            128,
            num_elts_per_sf=16,
        )
        gemm2_s.append(
            block_scale_interleave(s2.view(torch.uint8)[perm_s.to(device)].contiguous())
        )

    output_scale = torch.full(
        (cfg.local_num_experts,),
        global_sf * global_sf,
        dtype=torch.float32,
        device=device,
    )
    return (
        torch.stack(gemm1_w).contiguous(),
        torch.stack(gemm1_s)
        .view(torch.float8_e4m3fn)
        .reshape(
            cfg.local_num_experts,
            2 * cfg.intermediate_size,
            cfg.hidden_size // sf_vec_size,
        )
        .contiguous(),
        torch.stack(gemm2_w).contiguous(),
        torch.stack(gemm2_s)
        .view(torch.float8_e4m3fn)
        .reshape(
            cfg.local_num_experts,
            cfg.hidden_size,
            cfg.intermediate_size // sf_vec_size,
        )
        .contiguous(),
        output_scale,
    )


def _make_base_tensors(cfg: BenchConfig, device: torch.device, seed: int):
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    hidden = (
        torch.randn(
            cfg.num_tokens,
            cfg.hidden_size,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.1
    )
    w1 = (
        torch.randn(
            cfg.local_num_experts,
            2 * cfg.intermediate_size,
            cfg.hidden_size,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.02
    )
    w2 = (
        torch.randn(
            cfg.local_num_experts,
            cfg.hidden_size,
            cfg.intermediate_size,
            device=device,
            dtype=torch.float32,
            generator=gen,
        )
        * 0.02
    )
    return hidden.to(torch.bfloat16), w1, w2


def _make_precision_case(
    name: str,
    cfg: BenchConfig,
    hidden_bf16: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    routing_input_mode: str,
) -> PrecisionCase:
    routing_method = RoutingMethodType(
        int(
            os.environ.get(
                "FLASHINFER_TEST_ROUTING_METHOD", RoutingMethodType.DeepSeekV3
            )
        )
    )
    common = dict(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        n_group=cfg.n_group if routing_method == RoutingMethodType.DeepSeekV3 else None,
        topk_group=(
            cfg.topk_group if routing_method == RoutingMethodType.DeepSeekV3 else None
        ),
        intermediate_size=cfg.intermediate_size,
        local_expert_offset=0,
        local_num_experts=cfg.local_num_experts,
        routed_scaling_factor=cfg.routed_scaling_factor,
        routing_method_type=routing_method,
        tune_max_num_tokens=cfg.tune_max_num_tokens,
        activation_type=ActivationType.Swiglu.value,
        norm_topk_prob=True,
    )

    if name == "bf16":
        gemm1, gemm2 = _prepare_bf16_weights(w1, w2)

        if routing_input_mode == "routed":
            routed_common = dict(common)
            routed_common.pop("norm_topk_prob")

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_bf16_routed_moe(
                            routing_input,
                            hidden_bf16,
                            gemm1,
                            gemm2,
                            use_shuffled_weight=True,
                            weight_layout=WeightLayout.BlockMajorK,
                            **routed_common,
                        )
                    )

        else:

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_bf16_moe(
                            routing_input,
                            None,
                            hidden_bf16,
                            gemm1,
                            gemm2,
                            use_shuffled_weight=True,
                            weight_layout=WeightLayout.BlockMajorK,
                            **common,
                        )
                    )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_bf16_moe",
            DtypeTrtllmGen.Bfloat16,
            DtypeTrtllmGen.Bfloat16,
            Fp8QuantizationType.NoneFp8,
            WeightLayout.BlockMajorK,
            True,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    if name == "fp8_per_tensor":
        hidden_fp8, hidden_deq = _fp8_quantize_tensor(hidden_bf16)
        gemm1, scale1, scale1_gate, gemm2, scale2 = _prepare_fp8_per_tensor_weights(
            w1, w2
        )
        scale1 = (scale1 * hidden_deq).contiguous()
        scale1_gate = (scale1_gate * hidden_deq).contiguous()
        scale2 = scale2.contiguous()

        def call(routing_logits):
            with autotune(False):
                return _as_tensor(
                    trtllm_fp8_per_tensor_scale_moe(
                        routing_logits,
                        None,
                        hidden_fp8,
                        gemm1,
                        scale1,
                        scale1_gate,
                        gemm2,
                        scale2,
                        use_routing_scales_on_input=False,
                        **common,
                    )
                )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_fp8_per_tensor_scale_moe",
            DtypeTrtllmGen.E4m3,
            DtypeTrtllmGen.E4m3,
            Fp8QuantizationType.NoneFp8,
            WeightLayout.MajorK,
            True,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    if name == "fp8_block":
        hidden_fp8, hidden_scale = _fp8_block_quantize(hidden_bf16, 1, 128)
        hidden_scale = hidden_scale.transpose(0, 1).contiguous()
        gemm1, gemm1_scale = _fp8_block_quantize(w1, 128, 128)
        gemm2, gemm2_scale = _fp8_block_quantize(w2, 128, 128)

        if routing_input_mode == "routed":
            routed_common = dict(common)
            routed_common.pop("norm_topk_prob")

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp8_block_scale_routed_moe(
                            routing_input,
                            None,
                            hidden_fp8,
                            hidden_scale,
                            gemm1,
                            gemm1_scale,
                            gemm2,
                            gemm2_scale,
                            use_shuffled_weight=False,
                            weight_layout=WeightLayout.MajorK,
                            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
                            **routed_common,
                        )
                    )

        else:

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp8_block_scale_moe(
                            routing_input,
                            None,
                            hidden_fp8,
                            hidden_scale,
                            gemm1,
                            gemm1_scale,
                            gemm2,
                            gemm2_scale,
                            use_shuffled_weight=False,
                            weight_layout=WeightLayout.MajorK,
                            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
                            **common,
                        )
                    )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_fp8_block_scale_moe",
            DtypeTrtllmGen.E4m3,
            DtypeTrtllmGen.E4m3,
            Fp8QuantizationType.DeepSeekFp8,
            WeightLayout.MajorK,
            False,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    if name == "mxfp8":
        hidden_fp8, hidden_scale = mxfp8_quantize(hidden_bf16, False)
        hidden_scale = hidden_scale.view(torch.uint8).reshape(cfg.num_tokens, -1)
        gemm1, gemm1_scale, gemm2, gemm2_scale = _prepare_fp8_mxfp8_weights(w1, w2, cfg)

        if routing_input_mode == "routed":
            routed_common = dict(common)
            routed_common.pop("norm_topk_prob")

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp8_block_scale_routed_moe(
                            routing_input,
                            None,
                            hidden_fp8,
                            hidden_scale,
                            gemm1,
                            gemm1_scale,
                            gemm2,
                            gemm2_scale,
                            use_shuffled_weight=True,
                            weight_layout=WeightLayout.MajorK,
                            fp8_quantization_type=Fp8QuantizationType.MxFp8,
                            **routed_common,
                        )
                    )

        else:

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp8_block_scale_moe(
                            routing_input,
                            None,
                            hidden_fp8,
                            hidden_scale,
                            gemm1,
                            gemm1_scale,
                            gemm2,
                            gemm2_scale,
                            use_shuffled_weight=True,
                            weight_layout=WeightLayout.MajorK,
                            fp8_quantization_type=Fp8QuantizationType.MxFp8,
                            **common,
                        )
                    )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_fp8_block_scale_moe",
            DtypeTrtllmGen.MxE4m3,
            DtypeTrtllmGen.MxE4m3,
            Fp8QuantizationType.MxFp8,
            WeightLayout.MajorK,
            True,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    if name in {"nvfp4", "mxfp4_mxfp8", "mxfp4_bf16"}:
        if name == "nvfp4":
            sf_vec_size = 16
            sf_use_ue8m0 = False
            global_sf = 1.0 / 448.0 / 6.0
            hidden_fp4, hidden_scale = fp4_quantize(
                hidden_bf16,
                torch.tensor([1.0 / global_sf], device=hidden_bf16.device),
                sf_vec_size=sf_vec_size,
                sf_use_ue8m0=sf_use_ue8m0,
                is_sf_swizzled_layout=False,
            )
            hidden_states = hidden_fp4.view(torch.uint8).reshape(
                cfg.num_tokens, cfg.hidden_size // 2
            )
            hidden_states_scale = hidden_scale.view(torch.float8_e4m3fn).reshape(
                cfg.num_tokens, cfg.hidden_size // sf_vec_size
            )
            dtype_act = DtypeTrtllmGen.E2m1
            dtype_weights = DtypeTrtllmGen.E2m1
        elif name == "mxfp4_mxfp8":
            sf_vec_size = 32
            sf_use_ue8m0 = True
            global_sf = 1.0
            hidden_fp8, hidden_scale = mxfp8_quantize(hidden_bf16, False)
            hidden_states = hidden_fp8
            hidden_states_scale = hidden_scale.view(torch.float8_e4m3fn).reshape(
                cfg.num_tokens, cfg.hidden_size // 32
            )
            dtype_act = DtypeTrtllmGen.MxE4m3
            dtype_weights = DtypeTrtllmGen.MxE2m1
        else:
            sf_vec_size = 32
            sf_use_ue8m0 = True
            global_sf = 1.0
            hidden_states = hidden_bf16
            hidden_states_scale = None
            dtype_act = DtypeTrtllmGen.Bfloat16
            dtype_weights = DtypeTrtllmGen.MxE2m1

        gemm1, gemm1_scale, gemm2, gemm2_scale, output_scale = _prepare_fp4_weights(
            w1,
            w2,
            cfg,
            sf_vec_size=sf_vec_size,
            sf_use_ue8m0=sf_use_ue8m0,
            global_sf=global_sf,
        )

        if routing_input_mode == "routed":
            routed_common = dict(common)
            routed_common.pop("norm_topk_prob")

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp4_block_scale_routed_moe(
                            routing_input,
                            None,
                            hidden_states,
                            hidden_states_scale,
                            gemm1,
                            gemm1_scale,
                            None,
                            None,
                            None,
                            None,
                            gemm2,
                            gemm2_scale,
                            None,
                            output_scale,
                            output_scale,
                            output_scale,
                            **routed_common,
                        )
                    )

        else:

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_fp4_block_scale_moe(
                            routing_input,
                            None,
                            hidden_states,
                            hidden_states_scale,
                            gemm1,
                            gemm1_scale,
                            None,
                            None,
                            None,
                            None,
                            gemm2,
                            gemm2_scale,
                            None,
                            output_scale,
                            output_scale,
                            output_scale,
                            **common,
                        )
                    )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_fp4_block_scale_moe",
            dtype_act,
            dtype_weights,
            Fp8QuantizationType.NoneFp8,
            WeightLayout.MajorK,
            True,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    if name == "mxint4":
        gemm1, gemm1_scale, gemm2, gemm2_scale = _prepare_mxint4_weights(w1, w2, cfg)
        mx_common = dict(common)
        mx_common.pop("activation_type")
        if routing_input_mode == "routed":
            mx_common.pop("norm_topk_prob")

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_mxint4_block_scale_routed_moe(
                            routing_input,
                            hidden_bf16,
                            gemm1,
                            gemm1_scale,
                            None,
                            None,
                            None,
                            gemm2,
                            gemm2_scale,
                            **mx_common,
                        )
                    )

        else:

            def call(routing_input):
                with autotune(False):
                    return _as_tensor(
                        trtllm_mxint4_block_scale_moe(
                            routing_input,
                            None,
                            hidden_bf16,
                            gemm1,
                            gemm1_scale,
                            None,
                            None,
                            None,
                            gemm2,
                            gemm2_scale,
                            **mx_common,
                        )
                    )

        return PrecisionCase(
            name,
            "flashinfer::trtllm_mxint4_block_scale_moe",
            DtypeTrtllmGen.Bfloat16,
            DtypeTrtllmGen.MxInt4,
            Fp8QuantizationType.NoneFp8,
            WeightLayout.BlockMajorK,
            True,
            False,
            BENCHMARK_MODES[name],
            call,
        )

    raise ValueError(f"unknown precision: {name}")


# ---------------------------------------------------------------------------
# Autotuning and bundle compatibility
# ---------------------------------------------------------------------------


def _make_routing_input(
    routing_input_mode: str,
    precision: str,
    distribution: str,
    cfg: BenchConfig,
    device: torch.device,
) -> Any:
    specs = get_da_distribution_specs(distribution)
    ids = generate_da_distribution_assignments(
        specs[0],
        torch.zeros(cfg.num_tokens, cfg.top_k, dtype=torch.int32, device=device),
        cfg.local_num_experts,
        cfg.num_experts,
        cfg.top_k,
        0,
    ).to(torch.int32)
    if routing_input_mode == "routed":
        weights = torch.linspace(
            1.0,
            0.5,
            steps=max(cfg.top_k, 1),
            dtype=torch.float32,
            device=device,
        )
        weights = weights / weights.sum()
        weights = (weights * cfg.routed_scaling_factor).to(torch.bfloat16)
        weights = weights.expand(cfg.num_tokens, -1).contiguous()
        if _supports_unpacked_precomputed(precision):
            return ids.contiguous(), weights
        return (ids << 16) | (weights.view(torch.int16).to(torch.int32) & 0xFFFF)

    logits = torch.full(
        (cfg.num_tokens, cfg.num_experts), -30.0, dtype=torch.float32, device=device
    )
    vals = torch.linspace(
        30.0, 29.0, steps=max(cfg.top_k, 1), dtype=torch.float32, device=device
    ).reshape(1, cfg.top_k)
    logits.scatter_(1, ids.to(torch.long), vals.expand(cfg.num_tokens, -1))
    return logits


def _make_da_context(
    case: PrecisionCase,
    cfg: BenchConfig,
    device: torch.device,
):
    return da_state.make_context(
        case.op_name,
        device=device,
        dtype_act=case.dtype_act,
        dtype_weights=case.dtype_weights,
        quantization_type=case.quantization_type,
        top_k=cfg.top_k,
        num_experts=cfg.num_experts,
        num_local_experts=cfg.local_num_experts,
        local_expert_offset=0,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        activation_type=ActivationType.Swiglu.value,
        weight_layout=case.weight_layout,
        use_shuffled_weight=case.use_shuffled_weight,
        use_per_token_scaling=case.use_per_token_scaling,
        has_gemm1_lora_delta=False,
    )


def _tune_result_from_loaded_bundle(
    case: PrecisionCase,
    cfg: BenchConfig,
    device: torch.device,
    bundle_path: str,
) -> dict:
    da_context = _make_da_context(case, cfg, device)
    with open(bundle_path, "rb") as f:
        bundle = pickle.load(f)

    def valid_tile_sizes(bucket: int, **_kwargs) -> tuple[int, ...]:
        tactics = moe_core.get_trtllm_moe_sm100_module().ffi_moe_op.trtllm_get_valid_moe_configs(
            case.dtype_act,
            case.dtype_weights,
            case.quantization_type,
            cfg.top_k,
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.local_num_experts,
            ActivationType.Swiglu,
            case.use_shuffled_weight,
            case.weight_layout,
            case.use_per_token_scaling,
            int(bucket),
            False,
        )
        return tuple(sorted({int(tactic[0]) for tactic in tactics if len(tactic)}))

    da_profile.load_knn_v2_bundle(
        bundle,
        bundle_path,
        backend=da_profile.DAProfileBackend(
            get_ffi_moe_op=lambda: moe_core.get_trtllm_moe_sm100_module().ffi_moe_op,
            supported_tile_sizes=valid_tile_sizes,
        ),
        da_context=da_context,
        expected_top_k=cfg.top_k,
        expected_num_local_experts=cfg.local_num_experts,
        expected_hidden_size=cfg.hidden_size,
        expected_intermediate_size=cfg.intermediate_size,
        expected_activation_type=ActivationType.Swiglu.value,
    )
    key = da_state.cache_key(
        da_context,
        da_core.upload_bucket(cfg.num_tokens, cfg.tune_max_num_tokens),
    )
    tactics = da_state.PER_BODY_TACTICS.get(key)
    if not tactics:
        raise RuntimeError(f"bundle has no runtime bodies for {key}")
    return {
        "tactics": [(int(tile), int(config)) for tile, config in tactics],
        "static_profiles": 0,
        "total_profiles": 0,
        "static_tune_seconds": 0.0,
        "da_tune_seconds": 0.0,
        "tune_seconds": 0.0,
    }


def _prime_single_body_da_cache(
    case: PrecisionCase, cfg: BenchConfig
) -> tuple[int, int]:
    da_state.PER_TILE_TACTICS.clear()
    da_state.PER_BODY_TACTICS.clear()

    device = torch.device("cuda", torch.cuda.current_device())
    da_context = _make_da_context(case, cfg, device)
    bucket = da_core.upload_bucket(cfg.num_tokens, cfg.tune_max_num_tokens)
    moe_op = moe_core.get_trtllm_moe_sm100_module().ffi_moe_op
    raw = moe_op.trtllm_get_valid_moe_configs(
        case.dtype_act,
        case.dtype_weights,
        case.quantization_type,
        cfg.top_k,
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.local_num_experts,
        ActivationType.Swiglu,
        case.use_shuffled_weight,
        case.weight_layout,
        case.use_per_token_scaling,
        bucket,
        False,
    )
    candidate_tiles = {int(item[0]) for item in raw if len(item) > 0}
    tactic = None
    for item in raw:
        if len(item) < 2:
            continue
        tile, config = int(item[0]), int(item[1])
        if tile not in candidate_tiles:
            continue
        tactic = (tile, config)
        break
    if tactic is None:
        raise RuntimeError(
            f"no valid DA tactic for {case.name}; candidate_tiles={sorted(candidate_tiles)} "
            f"raw={list(raw)[:8]}"
        )

    exemplar = torch.full(
        (cfg.local_num_experts,),
        float(cfg.local_num_experts) ** -0.5,
        dtype=torch.float32,
        device=device,
    )
    tile, config = tactic
    da_profile.upload_and_publish_selector_tactics(
        moe_op,
        da_context,
        exemplar,
        [0],
        [tile],
        [config],
        [tile],
        cfg.local_num_experts,
        0,
        cfg.top_k,
        bucket,
        per_tile_tactics={tile: tactic},
        per_body_tactics=[tactic],
    )
    return tactic


def _is_value_profile_cache_key(cache_key: Any) -> bool:
    """Return whether an autotuner cache key includes value-profile buckets."""
    if hasattr(cache_key, "nearest_profile"):
        profile_key = cache_key.nearest_profile
    else:
        profile_key = cache_key[3]
    return bool(
        isinstance(profile_key, tuple)
        and len(profile_key) == 2
        and isinstance(profile_key[0], tuple)
        and profile_key[0]
        and isinstance(profile_key[0][0], tuple)
    )


def _run_real_autotune(
    case: PrecisionCase,
    cfg: BenchConfig,
    device: torch.device,
    routing_input_mode: str,
) -> dict:
    tuner = AutoTuner.get()
    tuner.clear_cache()
    tuner.reset_statistics()
    da_state.PER_TILE_TACTICS.clear()
    da_state.PER_BODY_TACTICS.clear()
    da_state.STATIC_FALLBACK_TACTICS.clear()
    da_state.BASELINE_GUARD_DECISIONS.clear()

    tuning_input = _make_routing_input(
        routing_input_mode, case.name, "uniform", cfg, device
    )
    # Tune every runtime bucket through the declared maximum, not only the
    # shape that happened to trigger this benchmark invocation.
    tuning_buckets = get_hybrid_num_tokens_buckets(cfg.tune_max_num_tokens, 1)

    # Establish the normal shape-only autotuned baseline first.
    tune_start = time.perf_counter()
    os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "0"
    with (
        nvtx_range(f"phase=static-autotune, precision={case.name}"),
        autotune(True, tuning_buckets=tuning_buckets, round_up=True),
    ):
        case.call(tuning_input)
    torch.cuda.synchronize()
    static_tune_seconds = time.perf_counter() - tune_start
    static_profiles = len(tuner.profiling_cache)

    # Then populate per-distribution profiles, body tactics, and kNN exemplars.
    da_tune_start = time.perf_counter()
    os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "1"
    with (
        cuda_profiler_da_phase(),
        nvtx_range(f"phase=da-autotune, precision={case.name}"),
        autotune(True, tuning_buckets=tuning_buckets, round_up=True),
    ):
        case.call(tuning_input)
    torch.cuda.synchronize()
    da_tune_seconds = time.perf_counter() - da_tune_start
    tune_seconds = time.perf_counter() - tune_start

    da_context = _make_da_context(case, cfg, device)
    key = da_state.cache_key(
        da_context,
        da_core.upload_bucket(cfg.num_tokens, cfg.tune_max_num_tokens),
    )
    tactics = da_state.PER_BODY_TACTICS.get(key)
    guard_decision = da_state.BASELINE_GUARD_DECISIONS.get(key, {})
    final_policy = guard_decision.get("final_policy", guard_decision.get("policy"))
    if not tactics and final_policy != "noda_baseline_guard":
        raise RuntimeError(f"DA autotune produced no runtime bodies for {key}")
    return {
        "tactics": [(int(tile), int(config)) for tile, config in tactics or ()],
        "guard_decision": guard_decision,
        "static_profiles": static_profiles,
        "total_profiles": len(tuner.profiling_cache),
        "static_tune_seconds": static_tune_seconds,
        "da_tune_seconds": da_tune_seconds,
        "tune_seconds": tune_seconds,
    }


def _runtime_tactics(
    case: PrecisionCase,
    cfg: BenchConfig,
    device: torch.device,
) -> list[tuple[int, int]]:
    """Return the retained selector bodies for one runtime token bucket."""
    da_context = _make_da_context(case, cfg, device)
    key = da_state.cache_key(
        da_context,
        da_core.upload_bucket(cfg.num_tokens, cfg.tune_max_num_tokens),
    )
    tactics = da_state.PER_BODY_TACTICS.get(key)
    decision = da_state.BASELINE_GUARD_DECISIONS.get(key, {})
    final_policy = decision.get("final_policy", decision.get("policy"))
    if not tactics and final_policy == "noda_baseline_guard":
        return []
    if not tactics:
        raise RuntimeError(f"tuning result has no runtime bodies for {key}")
    return [(int(tile), int(config)) for tile, config in tactics or ()]


def _runtime_guard_decision(
    case: PrecisionCase,
    cfg: BenchConfig,
    device: torch.device,
) -> dict:
    """Return the DA baseline-guard decision for one runtime token bucket."""

    da_context = _make_da_context(case, cfg, device)
    key = da_state.cache_key(
        da_context,
        da_core.upload_bucket(cfg.num_tokens, cfg.tune_max_num_tokens),
    )
    decision = dict(da_state.BASELINE_GUARD_DECISIONS.get(key, {}))
    if decision:
        return decision
    tactics = da_state.PER_BODY_TACTICS.get(key, ())
    if len(tactics) == 1:
        return {
            "policy": "da_singleton",
            "candidate_policy": "da_singleton",
            "candidate_tactics": [tuple(int(v) for v in tactics[0])],
            "final_policy": "da_singleton",
            "final_tactics": [tuple(int(v) for v in tactics[0])],
            "singleton_tactic": tuple(int(v) for v in tactics[0]),
            "singleton_source": "profiled",
        }
    if len(tactics) > 1:
        normalized = [tuple(int(v) for v in tactic) for tactic in tactics]
        return {
            "policy": "da_switch",
            "candidate_policy": "da_switch",
            "candidate_tactics": normalized,
            "final_policy": "da_switch",
            "final_tactics": normalized,
        }
    return {
        "policy": "noda",
        "candidate_policy": "noda",
        "candidate_tactics": [],
        "final_policy": "noda",
        "final_tactics": [],
    }


def _guard_row_fields(decision: dict) -> dict:
    """Convert a guard decision into stable benchmark CSV columns."""

    return {
        "da_policy": decision.get("policy", "noda"),
        "da_candidate_policy": decision.get(
            "candidate_policy", decision.get("policy", "noda")
        ),
        "da_candidate_tactics": repr(decision.get("candidate_tactics", [])),
        "da_final_policy": decision.get("final_policy", decision.get("policy", "noda")),
        "da_final_tactics": repr(decision.get("final_tactics", [])),
        "da_baseline_tactic": repr(decision.get("baseline_tactic")),
        "da_baseline_ms": decision.get("baseline_ms"),
        "da_baseline_worst_ms": decision.get("baseline_worst_ms"),
        "da_overhead_ms": decision.get("overhead_ms"),
        "da_control_overhead_source": decision.get(
            "control_overhead_source", "unavailable"
        ),
        "da_guard_margin": decision.get("margin"),
        "da_guard_admission_applied": decision.get("admission_applied"),
        "da_guard_limitation": decision.get("limitation"),
        "da_dynamic_worst_ms": decision.get("dynamic_worst_ms"),
        "da_singleton_tactic": repr(decision.get("singleton_tactic")),
        "da_singleton_worst_ms": decision.get("singleton_worst_ms"),
        "da_singleton_source": decision.get("singleton_source"),
    }


def _last_runtime_tactic(custom_op: str) -> tuple[Any, str]:
    """Return the last inference tactic and whether it was tuned or fallback."""
    selection = AutoTuner.get().last_selection
    if (
        not selection
        or selection.get("custom_op") != custom_op
        or selection.get("is_tuning_mode")
        or "tactic" not in selection
    ):
        return None, "unavailable"

    tactic = selection["tactic"]
    is_fallback = tactic == -1 or tactic == [-1] or tactic == (-1,)
    if is_fallback or not selection.get("is_cache_hit", False):
        return tactic, "fallback"
    return tactic, "tuned"


# ---------------------------------------------------------------------------
# Benchmark routines
# ---------------------------------------------------------------------------


def _capture_and_time(call: Callable[[], torch.Tensor], warmup: int, iters: int):
    for _ in range(warmup):
        _ = call()
    torch.cuda.synchronize()

    moe_core.reset_da_fast_path_stats()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    torch.cuda.synchronize()
    stats = moe_core.get_da_fast_path_stats()

    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    latency_ms = start.elapsed_time(end) / max(iters, 1)
    torch.cuda.synchronize()
    return out.detach().clone(), latency_ms, int(stats["capture_dispatch_count"])


def _eager_and_time(call: Callable[[], torch.Tensor], warmup: int, iters: int):
    for _ in range(warmup):
        _ = call()
    torch.cuda.synchronize()

    moe_core.reset_da_fast_path_stats()
    out = call()
    torch.cuda.synchronize()
    stats = moe_core.get_da_fast_path_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _ = call()
    end.record()
    end.synchronize()
    latency_ms = start.elapsed_time(end) / max(iters, 1)
    return out.detach().clone(), latency_ms, int(stats["capture_dispatch_count"])


def _accuracy(ref: torch.Tensor, out: torch.Tensor) -> dict[str, float]:
    diff = (ref.float() - out.float()).abs()
    denom = ref.float().abs().clamp(min=1.0e-6)
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
        "pct_abs_le_1e_1": float((diff <= 1.0e-1).float().mean().item()),
        "pct_abs_le_5e_2": float((diff <= 5.0e-2).float().mean().item()),
        "match_ratio": float(
            torch.isclose(ref.float(), out.float(), atol=0.1, rtol=0.85)
            .float()
            .mean()
            .item()
        ),
        "finite": float(torch.isfinite(out).all().item()),
    }


def _parse_distributions(value: str) -> list[str]:
    value = value.strip()
    if value == "full":
        vals = [round(1.1 + 0.1 * i, 1) for i in range(70)]
        return ["uniform"] + [f"ddist:{v:.1f}" for v in vals]
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "uniform" or item.startswith("ddist:"):
            result.append(item)
        else:
            result.append(f"ddist:{float(item):.1f}")
    return result


def _parse_precision_modes(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(BENCHMARK_MODES)
    modes = [item.strip() for item in value.split(",") if item.strip()]
    unsupported = sorted(set(modes) - set(BENCHMARK_MODES))
    if unsupported:
        raise ValueError("unsupported precision mode(s): " + ", ".join(unsupported))
    if not modes:
        raise ValueError("at least one precision mode is required")
    return modes


def _validate_routing_input_mode(mode: str, precisions: list[str]) -> list[str]:
    if mode == "routed":
        unsupported = sorted(set(precisions) - ROUTED_API_PRECISION_MODES)
        if unsupported:
            raise ValueError(
                "--routing-input-mode routed requires a public routed MoE API; "
                "unsupported precision mode(s): " + ", ".join(unsupported)
            )
    return precisions


def _supports_unpacked_precomputed(precision: str) -> bool:
    """Mirror which routed wrappers accept ``UnpackedPrecomputed`` in core.py."""
    return precision in UNPACKED_PRECOMPUTED_PRECISION_MODES


def _reported_internal_routing_mode(mode: str) -> str:
    return "routed" if mode == "routed" else "packed"


def _resolve_local_num_experts(num_experts: int, local_num_experts: int | None) -> int:
    """Default to no expert parallelism unless local experts are explicit."""
    return num_experts if local_num_experts is None else local_num_experts


def _row_status(row: dict) -> str:
    if row.get("noda_capture_dispatch_count", 0) != 0:
        return "FAIL_NODA_USED_DA"
    if row.get("da_policy") in {"noda", "noda_baseline_guard"}:
        if row.get("da_capture_dispatch_count", 0) != 0:
            return "FAIL_NODA_USED_DA_CAPTURE"
    elif (
        row.get("execution_mode") == "graph"
        and row.get("da_capture_dispatch_count", 0) <= 0
    ):
        return "FAIL_NO_DA_CAPTURE"
    threshold = row.get("match_ratio_threshold", 1.0)
    if not row.get("noda_finite", 0.0) or not row.get("da_finite", 0.0):
        return "FAIL_NONFINITE"
    if row.get("noda_match_ratio", 0.0) < threshold:
        return "FAIL_NODA_ACCURACY"
    if row.get("da_match_ratio", 0.0) < threshold:
        return "FAIL_ACCURACY"
    return "PASS"


def _release_benchmark_capture_memory() -> None:
    """Drop graph-lifetime tensors after a completed benchmark configuration."""
    da_capture.CAPTURE_RESOURCES.clear()
    da_state.CAPTURE_KEEPALIVE.clear()
    gc.collect()
    torch.cuda.empty_cache()


def _run_precision_sweep(
    args,
    configs: list[BenchConfig],
    device: torch.device,
    precisions: list[str],
) -> list[dict]:
    distributions = _parse_distributions(args.distributions)
    tuning_cfg = configs[0]

    rows = []
    for precision in precisions:
        print(f"[precision] {precision}", flush=True)
        try:
            hidden, w1, w2 = _make_base_tensors(tuning_cfg, device, args.seed)
            tuning_case = _make_precision_case(
                precision,
                tuning_cfg,
                hidden,
                w1,
                w2,
                args.routing_input_mode,
            )
            bundle_path = _bundle_path_for_precision(
                args.bundle_output, precision, len(precisions) > 1
            )
            os.environ["FLASHINFER_DA_KNN_BUNDLE"] = bundle_path
            if args.skip_autotune:
                tune_result = _tune_result_from_loaded_bundle(
                    tuning_case, tuning_cfg, device, bundle_path
                )
            else:
                tune_result = _run_real_autotune(
                    tuning_case, tuning_cfg, device, args.routing_input_mode
                )
            if not args.skip_autotune:
                print(
                    "  [autotune] "
                    f"static={tune_result['static_tune_seconds']:.2f}s "
                    f"da={tune_result['da_tune_seconds']:.2f}s "
                    f"total={tune_result['tune_seconds']:.2f}s",
                    flush=True,
                )
        except Exception as exc:
            print(f"  setup failed: {exc}", flush=True)
            rows.append(
                {
                    "precision": precision,
                    "distribution": "setup",
                    "status": "FAIL_SETUP",
                    "error": repr(exc),
                }
            )
            continue

        if args.build_only:
            tactics = _runtime_tactics(tuning_case, tuning_cfg, device)
            guard_decision = _runtime_guard_decision(tuning_case, tuning_cfg, device)
            rows.append(
                {
                    "precision": precision,
                    "distribution": "setup",
                    "status": "PASS_BUILD_ONLY",
                    "bundle_output": bundle_path,
                    "da_num_bodies": len(tactics),
                    "da_tactics": repr(tactics),
                    "static_profiles": tune_result["static_profiles"],
                    "total_profiles": tune_result["total_profiles"],
                    "static_tune_seconds": tune_result["static_tune_seconds"],
                    "da_tune_seconds": tune_result["da_tune_seconds"],
                    "tune_seconds": tune_result["tune_seconds"],
                    **_guard_row_fields(guard_decision),
                }
            )
            continue

        tuning_case = hidden = w1 = w2 = None
        _release_benchmark_capture_memory()

        for cfg in configs:
            try:
                hidden, w1, w2 = _make_base_tensors(cfg, device, args.seed)
                case = _make_precision_case(
                    precision, cfg, hidden, w1, w2, args.routing_input_mode
                )
                tactics = _runtime_tactics(case, cfg, device)
                guard_decision = _runtime_guard_decision(case, cfg, device)
                print(
                    f"  [tokens={cfg.num_tokens}] "
                    f"[DA policy] {guard_decision.get('policy', 'noda')} "
                    f"[DA bodies] {tactics}",
                    flush=True,
                )
            except Exception as exc:
                print(f"  tokens={cfg.num_tokens} setup failed: {exc}", flush=True)
                rows.append(
                    {
                        "precision": precision,
                        "distribution": "setup",
                        "num_tokens": cfg.num_tokens,
                        "status": "FAIL_SETUP",
                        "error": repr(exc),
                    }
                )
                continue

            for distribution in distributions:
                print(f"  [dist] {distribution}", flush=True)
                row = {
                    "precision": precision,
                    "distribution": distribution,
                    "execution_mode": args.execution_mode,
                    "input_routing_mode": args.routing_input_mode,
                    "internal_routing_mode": _reported_internal_routing_mode(
                        args.routing_input_mode
                    ),
                    "num_tokens": cfg.num_tokens,
                    "num_experts": cfg.num_experts,
                    "local_num_experts": cfg.local_num_experts,
                    "top_k": cfg.top_k,
                    "hidden_size": cfg.hidden_size,
                    "intermediate_size": cfg.intermediate_size,
                    "da_num_bodies": len(tactics),
                    "da_tactics": repr(tactics),
                    "static_profiles": tune_result["static_profiles"],
                    "total_profiles": tune_result["total_profiles"],
                    "static_tune_seconds": tune_result["static_tune_seconds"],
                    "da_tune_seconds": tune_result["da_tune_seconds"],
                    "tune_seconds": tune_result["tune_seconds"],
                    "match_ratio_threshold": case.match_ratio,
                    "bundle_output": bundle_path,
                    **_guard_row_fields(guard_decision),
                }
                try:
                    routing_input = _make_routing_input(
                        args.routing_input_mode,
                        precision,
                        distribution,
                        cfg,
                        device,
                    )

                    os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "0"
                    ref = case.call(routing_input).detach().clone()
                    torch.cuda.synchronize()
                    time_call = (
                        _capture_and_time
                        if args.execution_mode == "graph"
                        else _eager_and_time
                    )
                    with nvtx_range(
                        f"tokens={cfg.num_tokens}, precision={precision}, "
                        f"distribution={distribution}, method=NoDA"
                    ):
                        noda_out, noda_ms, noda_count = time_call(
                            lambda: case.call(routing_input),
                            args.warmup,
                            args.iters,
                        )
                    noda_tactic, noda_tactic_source = _last_runtime_tactic(case.op_name)

                    os.environ["FLASHINFER_DIST_AWARE_AUTOTUNE"] = "1"
                    with nvtx_range(
                        f"tokens={cfg.num_tokens}, precision={precision}, "
                        f"distribution={distribution}, method=DA"
                    ):
                        da_out, da_ms, da_count = time_call(
                            lambda: case.call(routing_input),
                            args.warmup,
                            args.iters,
                        )

                    noda_acc = _accuracy(ref, noda_out)
                    da_acc = _accuracy(ref, da_out)
                    row.update(
                        {
                            "noda_latency_ms": noda_ms,
                            "da_latency_ms": da_ms,
                            "noda_tokens_per_s": cfg.num_tokens / (noda_ms / 1000.0),
                            "da_tokens_per_s": cfg.num_tokens / (da_ms / 1000.0),
                            "speedup_da_over_noda": (
                                noda_ms / da_ms if da_ms > 0 else 0.0
                            ),
                            "noda_capture_dispatch_count": noda_count,
                            "noda_tactic": repr(noda_tactic),
                            "noda_tactic_source": noda_tactic_source,
                            "da_capture_dispatch_count": da_count,
                            "noda_max_abs": noda_acc["max_abs"],
                            "noda_mean_abs": noda_acc["mean_abs"],
                            "noda_max_rel": noda_acc["max_rel"],
                            "noda_pct_abs_le_1e_1": noda_acc["pct_abs_le_1e_1"],
                            "noda_match_ratio": noda_acc["match_ratio"],
                            "noda_finite": noda_acc["finite"],
                            "da_max_abs": da_acc["max_abs"],
                            "da_mean_abs": da_acc["mean_abs"],
                            "da_max_rel": da_acc["max_rel"],
                            "da_pct_abs_le_1e_1": da_acc["pct_abs_le_1e_1"],
                            "da_pct_abs_le_5e_2": da_acc["pct_abs_le_5e_2"],
                            "da_match_ratio": da_acc["match_ratio"],
                            "da_finite": da_acc["finite"],
                        }
                    )
                    row["status"] = _row_status(row)
                    print(
                        "    "
                        f"{row['status']} "
                        f"noda={noda_ms:.4f}ms da={da_ms:.4f}ms "
                        f"speedup={row['speedup_da_over_noda']:.3f} "
                        f"noda_tactic={noda_tactic!r} "
                        f"noda_source={noda_tactic_source} "
                        f"da_count={da_count} max_abs={da_acc['max_abs']:.4g}",
                        flush=True,
                    )
                except Exception as exc:
                    row["status"] = "FAIL_RUN"
                    row["error"] = repr(exc)
                    print(f"    FAIL_RUN {exc}", flush=True)
                rows.append(row)
            case = hidden = w1 = w2 = None
            _release_benchmark_capture_memory()
    return rows


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def _bundle_path_for_precision(
    bundle_path: str,
    precision: str,
    multiple_precisions: bool,
) -> str:
    """Keep the legacy NVFP4 artifact name while isolating multi-mode bundles."""
    if not multiple_precisions:
        return bundle_path
    path = Path(bundle_path)
    return str(path.with_name(f"{path.stem}_{precision}{path.suffix}"))


def build_parser() -> argparse.ArgumentParser:
    """Build the legacy-compatible DA benchmark command line."""
    parser = argparse.ArgumentParser(description="TRT-LLM MoE DA benchmark")
    parser.add_argument(
        "--precision",
        "--precisions",
        dest="precision",
        default="nvfp4",
        help=(
            "Comma-separated precision labels, or 'all'. Labels: bf16, "
            "fp8_per_tensor, fp8_block, mxfp8, nvfp4, "
            "mxfp4_mxfp8, mxfp4_bf16, mxint4"
        ),
    )
    parser.add_argument(
        "--routing-input-mode",
        choices=("routed", "logits"),
        default="routed",
        help=(
            "Public API routing input: the precision's established routed format "
            "(default), or routing logits. FP4 routed APIs receive raw IDs plus "
            "BF16 weights; other routed APIs receive packed int32 routing."
        ),
    )
    parser.add_argument(
        "--distributions",
        default="uniform,1.1,2.0,4.0,8.0",
        help="Comma-separated distributions or 'full' for uniform + ddist:1.1..8.0",
    )
    parser.add_argument("--num-tokens", default="128,512,2048")
    parser.add_argument("--num-experts", type=int, default=CFG.num_experts)
    parser.add_argument(
        "--local-num-experts",
        type=int,
        help="Experts resident on this rank (defaults to --num-experts)",
    )
    parser.add_argument("--top-k", type=int, default=CFG.top_k)
    parser.add_argument("--hidden-size", type=int, default=CFG.hidden_size)
    parser.add_argument("--intermediate-size", type=int, default=CFG.intermediate_size)
    parser.add_argument("--n-group", type=int, default=CFG.n_group)
    parser.add_argument("--topk-group", type=int, default=CFG.topk_group)
    parser.add_argument(
        "--routed-scaling-factor", type=float, default=CFG.routed_scaling_factor
    )
    parser.add_argument("--tune-max-num-tokens", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bundle-output", default="exp/trtllm_moe_da_bundle.pkl")
    parser.add_argument("--csv", default="")
    parser.add_argument(
        "--skip-autotune",
        action="store_true",
        help="Reuse the DA bundle instead of profiling it again",
    )
    parser.add_argument(
        "--skip-noda-autotune",
        action="store_true",
        help="Retained for legacy invocations; the static cache is in-memory",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Stop after loading or profiling DA bodies",
    )
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges")
    parser.add_argument(
        "--run-autotune",
        action="store_true",
        help=(
            "Legacy no-op retained for command compatibility; tuning runs by default "
            "unless --skip-autotune is specified"
        ),
    )
    parser.add_argument(
        "--execution-mode",
        choices=("graph", "eager"),
        default="graph",
    )
    parser.add_argument("--out", default="tmp/da_precision_runtime_results.csv")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        token_counts = [int(value) for value in args.num_tokens.split(",") if value]
        precisions = _validate_routing_input_mode(
            args.routing_input_mode, _parse_precision_modes(args.precision)
        )
    except ValueError as error:
        parser.error(str(error))
    if not token_counts:
        parser.error("at least one token count is required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    global _cuda_profiler_da_phase_enabled, _nvtx_enabled
    _nvtx_enabled = args.nvtx
    if _nvtx_enabled:
        # The benchmark phase ranges alone cannot identify the candidate
        # session that owns staging, warmup, capture, and replay work.
        os.environ["FLASHINFER_AUTOTUNE_NVTX"] = "1"
    _cuda_profiler_da_phase_enabled = (
        os.environ.get("FLASHINFER_DA_AUTOTUNE_CUDA_PROFILER") == "1"
    )
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    torch.manual_seed(args.seed)
    if args.skip_noda_autotune:
        print(
            "[legacy] --skip-noda-autotune uses the current AutoTuner cache", flush=True
        )

    configs = []
    for num_tokens in token_counts:
        cfg = BenchConfig(
            num_tokens=num_tokens,
            num_experts=args.num_experts,
            local_num_experts=_resolve_local_num_experts(
                args.num_experts, args.local_num_experts
            ),
            top_k=args.top_k,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            n_group=args.n_group,
            topk_group=args.topk_group,
            routed_scaling_factor=args.routed_scaling_factor,
            tune_max_num_tokens=args.tune_max_num_tokens or max(token_counts),
        )
        if cfg.num_experts % cfg.local_num_experts != 0:
            parser.error("num_experts must be divisible by local_num_experts")
        configs.append(cfg)

    all_rows = _run_precision_sweep(args, configs, device, precisions)

    out_path = args.csv or args.out
    fieldnames = sorted({key for row in all_rows for key in row})
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[done] wrote {out_path}", flush=True)
    return (
        0
        if all(str(row.get("status", "")).startswith("PASS") for row in all_rows)
        else 1
    )


if __name__ == "__main__":
    sys.exit(main())
