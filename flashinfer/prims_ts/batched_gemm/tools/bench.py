# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unified BatchedGemm benchmark driver: TS vs trtllm-gen.

Configurable MoE parameters: --num-experts, --top-k, --tokens,
--hidden-size, --intermediate-size.  Problem dimensions are derived:
  FC1: N = 2 * intermediate (SwiGLU gate+up), K = hidden
  FC2: N = hidden, K = intermediate

Presets:
  DeepSeek-R1:   --hidden-size 7168 --intermediate-size 2048 --num-experts 256 --top-k 8
  GPT-OSS 120B:  --hidden-size 3072 --intermediate-size 3072 --num-experts 128 --top-k 4
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
import time

from ..batched_gemm_config import (
    ActKind,
    BiasType,
    BatchMode,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    uniform_pipeline_stage_overrides,
)
from ..batched_gemm_run import benchmark

DEFAULT_TOKENS = (1, 32, 256, 1024, 8192)
DEFAULT_NUM_EXPERTS = 256
DEFAULT_TOP_K = 8
DEFAULT_HIDDEN_SIZE = 7168
DEFAULT_INTERMEDIATE_SIZE = 2048
DS_R1_NUM_EXPERTS = DEFAULT_NUM_EXPERTS
DS_R1_TOP_K = DEFAULT_TOP_K

# Module-level shape state, set by configure_shapes() before variant construction.
_FC1_N = 2 * DEFAULT_INTERMEDIATE_SIZE
_FC1_K = DEFAULT_HIDDEN_SIZE
_FC2_N = DEFAULT_HIDDEN_SIZE
_FC2_K = DEFAULT_INTERMEDIATE_SIZE
_NUM_EXPERTS = DEFAULT_NUM_EXPERTS
_TOP_K = DEFAULT_TOP_K


def configure_shapes(
    hidden_size: int, intermediate_size: int, num_experts: int, top_k: int
) -> None:
    """Recompute FC1/FC2 problem dimensions from model-level parameters."""
    global _FC1_N, _FC1_K, _FC2_N, _FC2_K, _NUM_EXPERTS, _TOP_K
    _FC1_N = 2 * intermediate_size  # gate+up for SwiGLU
    _FC1_K = hidden_size
    _FC2_N = hidden_size
    _FC2_K = intermediate_size
    _NUM_EXPERTS = num_experts
    _TOP_K = top_k


TRTLLM_GEN_ROOT_ENV = "TRTLLM_GEN_ROOT"
TRTLLM_GEN_BUILD_DIR_ENV = "TRTLLM_GEN_BUILD_DIR"
TRTLLM_GEN_CONFIG_NAMES = {
    "fp4": "batched_gemm_tllm_config_fi_mx.json",
    "bf16": "batched_gemm_tllm_config_fi_mx_fp16.json",
    "mx": "batched_gemm_tllm_config_fi_mxfp4_mxfp8.json",
    "mxfp4_bf16": "batched_gemm_tllm_config_fi_mxfp4_bf16.json",
    "fp8": "batched_gemm_tllm_config_fi_fp8.json",
}
TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY = "transposeMmaOutput"
TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_FLAG = "-transposeMmaOutput"
ELAPSED_TIME_RE = re.compile(r"Elapsed time\s*:\s*([0-9.]+)\s*ms")


def _trtllm_gen_root_from_env() -> Path | None:
    root = os.environ.get(TRTLLM_GEN_ROOT_ENV)
    return Path(root) if root else None


def _trtllm_gen_build_dir_from_env() -> Path | None:
    build_dir = os.environ.get(TRTLLM_GEN_BUILD_DIR_ENV)
    return Path(build_dir) if build_dir else None


@dataclass(frozen=True)
class BenchVariant:
    """A curated TS variant matched to a generated trtllm-gen config row."""

    name: str
    problem_n: int
    problem_k: int
    kwargs: dict
    source: str = "curated"
    config_index: int | None = None
    config_comment: str = ""
    combo_index: int | None = None
    generated_options: dict | None = None
    ts_skip_reason: str | None = None
    ts_notes: str = ""
    trtllm_gen_sf_layout_a: str | None = None
    trtllm_gen_sf_layout_b: str | None = None
    trtllm_gen_sf_layout_c: str | None = None


def _ts_kwargs(kwargs: dict) -> dict:
    """Translate generated benchmark knobs to the TS-supported option set."""
    result = dict(kwargs)
    if int(result.get("use_clc_fast_drain", 0)) != 0:
        result["use_clc_fast_drain"] = 0
    if int(result.get("use_unroll_loop_2x_for_mma", 0)) != 0:
        result["use_unroll_loop_2x_for_mma"] = 0
    return result


def _ts_unsupported_reason(kwargs: dict) -> str | None:
    reasons = []
    if int(kwargs.get("route_act", int(RouteImpl.NONE))) == int(RouteImpl.LDG_PLUS_STS):
        reasons.append("Kernel does not implement routeAct=LDG_PLUS_STS")
    if int(kwargs.get("route_sfs_act", int(RouteImpl.NONE))) == int(
        RouteImpl.LDG_PLUS_STS
    ):
        reasons.append("Kernel does not implement routeSfsAct=LDG_PLUS_STS")
    use_per_token_sf_b = int(kwargs.get("use_per_token_sf_b", 0))
    if use_per_token_sf_b and int(kwargs.get("transpose_mma_output", 1)) != 1:
        reasons.append("Kernel per-token sfB requires transpose_mma_output=1")
    return "; ".join(reasons) if reasons else None


def _with_ts_skip_reason(variant: BenchVariant) -> BenchVariant:
    reason = _ts_unsupported_reason(variant.kwargs)
    if reason is None:
        return variant
    if variant.ts_skip_reason:
        reason = f"{variant.ts_skip_reason}; {reason}"
    return replace(variant, ts_skip_reason=reason)


def _bool_arg(value: bool | int) -> str:
    return "true" if bool(value) else "false"


def _sf_layout_arg(layout: int) -> str:
    if layout == int(SfLayout.R8c4):
        return "8x4"
    if layout == int(SfLayout.LINEAR):
        return "linear"
    if layout == int(SfLayout.R128c4):
        return "128x4"
    raise ValueError(f"Unsupported generated sf layout: {layout}")


def _route_arg(route: int) -> str:
    if route == int(RouteImpl.NONE):
        return "false"
    if route == int(RouteImpl.TMA):
        return "tma"
    if route == int(RouteImpl.LDGSTS):
        return "ldgsts"
    if route == int(RouteImpl.LDG_PLUS_STS):
        return "ldgPlusSts"
    raise ValueError(f"Unsupported generated route: {route}")


def _scheduler_arg(scheduler: int) -> str:
    if scheduler == int(TileScheduler.STATIC):
        return "static"
    if scheduler == int(TileScheduler.PERSISTENT):
        return "persistent"
    raise ValueError(f"Unsupported generated scheduler: {scheduler}")


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    return bool(value)


def _dtype_value(
    value: object | None, *, default: str = "bf16", plain_fp8: bool = False
) -> int:
    dtype = default if value is None else str(value).lower()
    if dtype == "fp32":
        return int(DType.FP32)
    if dtype == "fp16":
        return int(DType.FP16)
    if dtype == "bf16":
        return int(DType.BF16)
    if dtype == "e2m1":
        return int(DType.E2M1)
    if dtype == "mxe2m1":
        return int(DType.MXE2M1)
    if dtype == "mxe4m3":
        return int(DType.MXE4M3)
    if dtype == "e4m3":
        return int(DType.E4M3) if plain_fp8 else int(DType.MXE4M3)
    raise ValueError(f"Unsupported dtype value: {value}")


def _dtype_arg(dtype: int) -> str:
    if dtype == int(DType.FP32):
        return "fp32"
    if dtype == int(DType.FP16):
        return "fp16"
    if dtype == int(DType.BF16):
        return "bf16"
    if dtype == int(DType.E2M1):
        return "e2m1"
    if dtype == int(DType.MXE2M1):
        return "mxe2m1"
    if dtype == int(DType.MXE4M3):
        return "mxe4m3"
    if dtype == int(DType.E4M3):
        return "e4m3"
    raise ValueError(f"Unsupported dtype: {dtype}")


def _uses_input_block_scaling(kwargs: dict) -> bool:
    dtype_a = int(kwargs["dtype_a"])
    dtype_b = int(kwargs["dtype_b"])
    block_scaled_dtypes = (int(DType.MXE2M1), int(DType.MXE4M3), int(DType.E2M1))
    return dtype_a in block_scaled_dtypes or dtype_b in block_scaled_dtypes


def _route_value(value: object) -> int:
    if value is False or value is None:
        return int(RouteImpl.NONE)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in ("false", "none"):
            return int(RouteImpl.NONE)
        if lowered == "tma":
            return int(RouteImpl.TMA)
        if lowered == "ldgsts":
            return int(RouteImpl.LDGSTS)
        if lowered == "ldgplussts":
            return int(RouteImpl.LDG_PLUS_STS)
    raise ValueError(f"Unsupported route value: {value}")


def _scheduler_value(value: object) -> int:
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "static":
            return int(TileScheduler.STATIC)
        if lowered == "persistent":
            return int(TileScheduler.PERSISTENT)
    raise ValueError(f"Unsupported scheduler value: {value}")


def _sf_layout_value(value: object | None, *, default: str = "128x4") -> int:
    layout = default if value is None else str(value)
    lowered = layout.lower()
    if lowered == "8x4":
        return int(SfLayout.R8c4)
    if lowered == "linear":
        return int(SfLayout.LINEAR)
    if lowered == "128x4":
        return int(SfLayout.R128c4)
    raise ValueError(f"Unsupported scale-factor layout: {value}")


def _act_value(
    *, generated_fused_act: bool, act: object | None, eltwise_act: object | None
) -> int:
    if generated_fused_act:
        if act == "swiglu":
            return int(ActKind.SWIGLU)
        if act == "geglu":
            return int(ActKind.GEGLU)
        if act == "silu":
            return int(ActKind.SILU)
    if eltwise_act == "relu2":
        return int(ActKind.RELU2)
    return int(ActKind.NONE)


def _per_token_sf_dtype(
    value: object | None,
    *,
    use_per_token_sf_a: bool = False,
    use_per_token_sf_b: bool = False,
) -> int:
    if value is None:
        if use_per_token_sf_a and use_per_token_sf_b:
            return int(DType.FP32)
        return int(DType.BF16)
    dtype = str(value).lower()
    if dtype == "fp16":
        return int(DType.FP16)
    if dtype == "bf16":
        return int(DType.BF16)
    if dtype == "fp32":
        return int(DType.FP32)
    raise ValueError(f"Unsupported per-token scale dtype: {value}")


def _has_activation_semantics(kwargs: dict) -> bool:
    return (
        int(kwargs["route_act"]) != int(RouteImpl.NONE)
        or int(kwargs["act_kind"]) != int(ActKind.NONE)
        or int(kwargs["dtype_c"]) not in (int(DType.BF16), int(DType.FP16))
    )


def _oa_clamp_kwargs(options: dict, kwargs: dict) -> dict:
    if int(kwargs.get("act_kind", int(ActKind.NONE))) != int(ActKind.SWIGLU):
        return {}
    clamp_limit = float(options.get("clampLimit", 0.0))
    if clamp_limit <= 0.0:
        return {}
    return {
        "has_gemm1_clamp_limit": 1,
        "gemm1_clamp_limit_value": clamp_limit,
    }


def _generated_fused_act_default(act_kind: int) -> bool:
    return act_kind in (int(ActKind.SWIGLU), int(ActKind.GEGLU), int(ActKind.SILU))


def _generated_act_arg(act_kind: int) -> str | None:
    if act_kind == int(ActKind.SWIGLU):
        return "swiglu"
    if act_kind == int(ActKind.GEGLU):
        return "geglu"
    if act_kind == int(ActKind.SILU):
        return "silu"
    return None


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return slug or "config"


def _num_activated_experts(num_experts: int, num_tokens: int, top_k: int) -> int:
    """Match the generated DeepSeek sweep's activated-expert formula."""
    return min(num_experts, num_tokens * top_k)


def _append_option(cmd: list[str], name: str, value: object) -> None:
    cmd.extend([name, str(value)])


def _append_bool_option(cmd: list[str], name: str, value: bool | int) -> None:
    _append_option(cmd, name, _bool_arg(value))


def _append_optional_option(cmd: list[str], name: str, options: dict, key: str) -> None:
    if key in options and options[key] is not None:
        _append_option(cmd, name, options[key])


def _append_optional_bool_option(
    cmd: list[str], name: str, options: dict, key: str
) -> None:
    if key in options and options[key] is not None:
        _append_bool_option(cmd, name, _bool_value(options[key]))


def _trtllm_gen_command(
    *,
    binary: Path,
    variant: BenchVariant,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    warmup_iters: int,
    bench_iters: int,
    num_rotated_buffers: int,
    use_ccache: bool,
    use_cuda_graph: bool,
) -> list[str]:
    kwargs = variant.kwargs
    generated_options = variant.generated_options or {}
    has_activation_semantics = _has_activation_semantics(kwargs)
    dtype_a_kind = int(kwargs["dtype_a"])
    dtype_b_kind = int(kwargs["dtype_b"])
    dtype_c_kind = int(kwargs["dtype_c"])
    act_kind = int(kwargs["act_kind"])
    uses_input_block_scaling = _uses_input_block_scaling(kwargs)
    route_act = int(kwargs["route_act"])
    route_sfs_act = int(kwargs["route_sfs_act"])
    scheduler = int(kwargs["tile_scheduler"])
    dtype_a = str(generated_options.get("dtypeA", _dtype_arg(dtype_a_kind)))
    dtype_b = str(generated_options.get("dtypeB", _dtype_arg(dtype_b_kind)))
    default_dtype_c = _dtype_arg(dtype_c_kind)
    dtype_c = str(generated_options.get("dtypeC", default_dtype_c))
    num_stages = int(kwargs["num_stages_a"])
    stage_fields = {
        "num_stages_b": kwargs["num_stages_b"],
        "num_stages_smem_sfa": kwargs["num_stages_smem_sfa"],
        "num_stages_smem_sfb": kwargs["num_stages_smem_sfb"],
        "num_stages_tmem_sfa": kwargs["num_stages_tmem_sfa"],
        "num_stages_tmem_sfb": kwargs["num_stages_tmem_sfb"],
    }
    mismatched_stages = {
        field: int(value)
        for field, value in stage_fields.items()
        if int(value) != num_stages
    }
    if mismatched_stages:
        raise ValueError(
            "Generated benchmark command only supports equal load/SF stage counts; "
            f"got A={num_stages}, mismatches={mismatched_stages} for {variant.name}"
        )
    generated_mma_stages = int(
        generated_options.get("numStagesMma", kwargs["num_stages_tmem_acc"])
    )

    cmd = [str(binary)]
    _append_option(
        cmd,
        "-testName",
        "DeepSeekR1_TP1_EP1_MoE_"
        f"{'activation' if has_activation_semantics else 'output'}_tokens{num_tokens}",
    )
    _append_option(cmd, "-numExperts", num_experts)
    _append_option(
        cmd,
        "-numActivatedExperts",
        _num_activated_experts(num_experts, num_tokens, top_k),
    )
    _append_option(cmd, "-batch", "N")
    _append_option(cmd, "-topK", top_k)
    _append_option(cmd, "-numTokens", num_tokens)
    _append_option(cmd, "-m", variant.problem_n)
    _append_option(cmd, "-k", variant.problem_k)
    _append_bool_option(
        cmd,
        TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_FLAG,
        _bool_value(generated_options.get("transpose_mma_output", True)),
    )
    _append_bool_option(
        cmd,
        "-useShuffledMatrix",
        _bool_value(generated_options.get("useShuffledMatrix", True)),
    )
    _append_bool_option(
        cmd,
        "-fusedAct",
        _bool_value(
            generated_options.get("fusedAct", _generated_fused_act_default(act_kind))
        ),
    )
    if "act" in generated_options:
        _append_option(cmd, "-act", generated_options["act"])
    elif (act_arg := _generated_act_arg(act_kind)) is not None:
        _append_option(cmd, "-act", act_arg)
    _append_option(cmd, "-routeAct", _route_arg(route_act))
    if route_sfs_act != int(RouteImpl.NONE):
        _append_option(cmd, "-routeSfsAct", _route_arg(route_sfs_act))
    if "biasType" in generated_options:
        _append_option(cmd, "-biasType", generated_options["biasType"])
    elif int(kwargs.get("bias_type", int(BiasType.NONE))) == int(BiasType.M):
        _append_option(cmd, "-biasType", "m")
    _append_optional_option(
        cmd, "-fusedBiasShuffleMode", generated_options, "fusedBiasShuffleMode"
    )
    if "eltwiseActType" in generated_options:
        _append_option(cmd, "-eltwiseActType", generated_options["eltwiseActType"])
    elif has_activation_semantics:
        _append_option(cmd, "-eltwiseActType", "none")
    _append_option(cmd, "-verbLvl", 30)
    _append_option(cmd, "-smVersion", generated_options.get("smVersion", "100f"))
    _append_bool_option(cmd, "-usesCcache", use_ccache)
    _append_bool_option(cmd, "-generatesLineInfo", True)
    _append_option(cmd, "-numWarmUpSteps", warmup_iters)
    _append_option(cmd, "-numBenchmarkSteps", bench_iters)
    _append_option(cmd, "-checkResults", "none")
    _append_bool_option(cmd, "-rotateBuffers", num_rotated_buffers > 0)
    _append_bool_option(cmd, "-usePdl", False)
    _append_bool_option(cmd, "-useCudaGraph", use_cuda_graph)
    _append_bool_option(cmd, "-skipsBadKernelConfig", False)
    _append_option(cmd, "-dtypeA", dtype_a)
    _append_option(cmd, "-dtypeB", dtype_b)
    _append_option(cmd, "-dtypeC", dtype_c)
    _append_bool_option(
        cmd,
        "-useDeepSeekFp8",
        _bool_value(generated_options.get("useDeepSeekFp8", False)),
    )
    _append_optional_bool_option(
        cmd, "-usePerTokenSfA", generated_options, "usePerTokenSfA"
    )
    _append_optional_bool_option(
        cmd, "-usePerTokenSfB", generated_options, "usePerTokenSfB"
    )
    _append_optional_option(
        cmd, "-perTokenSfDtype", generated_options, "perTokenSfDtype"
    )
    if dtype_a_kind == int(DType.E4M3) and dtype_b_kind == int(DType.E4M3):
        _append_option(cmd, "-quantScaleC", generated_options.get("quantScaleC", 1))
    _append_option(cmd, "-tileM", kwargs["tile_m"])
    _append_option(cmd, "-tileN", kwargs["tile_n"])
    _append_option(cmd, "-tileK", kwargs["tile_k"])
    _append_option(cmd, "-mmaM", kwargs["mma_m"])
    _append_option(cmd, "-mmaN", kwargs["mma_n"])
    _append_option(cmd, "-epilogueTileM", kwargs["epi_tile_m"])
    _append_option(cmd, "-epilogueTileN", kwargs["epi_tile_n"])
    _append_bool_option(cmd, "-useCustomMmaSchedule", True)
    _append_bool_option(
        cmd, "-sliceK", _bool_value(generated_options.get("sliceK", False))
    )
    _append_option(cmd, "-numStages", num_stages)
    _append_option(cmd, "-numStagesMma", generated_mma_stages)
    _append_option(
        cmd, "-numSlicesForSplitK", generated_options.get("numSlicesForSplitK", 1)
    )
    _append_option(cmd, "-clusterDimZ", generated_options.get("clusterDimZ", 1))
    _append_bool_option(
        cmd,
        "-useTwoMmaWarps",
        _bool_value(generated_options.get("useTwoMmaWarps", False)),
    )
    _append_bool_option(
        cmd,
        "-useTwoTmaLoadWarps",
        _bool_value(generated_options.get("useTwoTmaLoadWarps", False))
        or route_act != int(RouteImpl.NONE)
        or (scheduler == int(TileScheduler.PERSISTENT) and int(kwargs["tile_n"]) >= 64),
    )
    _append_bool_option(
        cmd,
        "-useHoistTryWaitForCustomMmaSchedule",
        _bool_value(
            generated_options.get("useHoistTryWaitForCustomMmaSchedule", False)
        ),
    )
    _append_option(cmd, "-tileScheduler", _scheduler_arg(scheduler))
    _append_bool_option(
        cmd,
        "-hoistMmaTaskTryWaits",
        _bool_value(generated_options.get("hoistMmaTaskTryWaits", False)),
    )

    if uses_input_block_scaling:
        epilogue_regs = generated_options.get(
            "numRegsPerThreadEpilogueWarp", kwargs.get("epilogue_regs")
        )
        non_epilogue_regs = generated_options.get(
            "numRegsPerThreadNonEpilogueWarp", kwargs.get("mma_regs")
        )
        if epilogue_regs is not None:
            _append_option(
                cmd,
                "-numRegsPerThreadEpilogueWarp",
                epilogue_regs,
            )
        if non_epilogue_regs is not None:
            _append_option(
                cmd,
                "-numRegsPerThreadNonEpilogueWarp",
                non_epilogue_regs,
            )
        sf_layout_a = variant.trtllm_gen_sf_layout_a
        sf_layout_b = variant.trtllm_gen_sf_layout_b
        sf_layout_c = variant.trtllm_gen_sf_layout_c
        _append_option(
            cmd,
            "-sfLayoutA",
            sf_layout_a if sf_layout_a else _sf_layout_arg(int(kwargs["sf_layout_a"])),
        )
        _append_option(
            cmd,
            "-sfLayoutB",
            sf_layout_b if sf_layout_b else _sf_layout_arg(int(kwargs["sf_layout_b"])),
        )
        _append_option(
            cmd,
            "-sfLayoutC",
            sf_layout_c if sf_layout_c else _sf_layout_arg(int(kwargs["sf_layout_c"])),
        )
    else:
        _append_option(cmd, "-numRegsPerThreadEpilogueWarp", kwargs["epilogue_regs"])
        _append_option(cmd, "-numRegsPerThreadNonEpilogueWarp", kwargs["mma_regs"])
        if "layoutA" in generated_options:
            _append_option(cmd, "-layoutA", generated_options["layoutA"])
        else:
            _append_option(cmd, "-layoutA", "B")

    _append_bool_option(cmd, "-useTmaOobOpt", bool(kwargs.get("use_tma_oob_opt", 0)))
    _append_option(cmd, "-clusterDimX", kwargs["cluster_m"])
    _append_option(cmd, "-clusterDimY", generated_options.get("clusterDimY", 1))
    _append_optional_option(cmd, "-ctaSwizzleType", generated_options, "ctaSwizzleType")
    _append_optional_option(
        cmd, "-numRegsPerThreadLoadA", generated_options, "numRegsPerThreadLoadA"
    )
    _append_optional_option(
        cmd, "-numRegsPerThreadLoadB", generated_options, "numRegsPerThreadLoadB"
    )
    _append_optional_option(
        cmd, "-numRegsPerThreadLoadSfA", generated_options, "numRegsPerThreadLoadSfA"
    )
    _append_optional_option(
        cmd, "-numRegsPerThreadLoadSfB", generated_options, "numRegsPerThreadLoadSfB"
    )
    _append_optional_option(cmd, "-numWarpsLoadA", generated_options, "numWarpsLoadA")
    _append_optional_option(cmd, "-numWarpsLoadB", generated_options, "numWarpsLoadB")
    _append_optional_option(
        cmd, "-numWarpsLoadSfA", generated_options, "numWarpsLoadSfA"
    )
    _append_optional_option(
        cmd, "-numWarpsLoadSfB", generated_options, "numWarpsLoadSfB"
    )
    _append_optional_option(
        cmd, "-numEpilogueWarps", generated_options, "numEpilogueWarps"
    )
    _append_optional_option(cmd, "-dtypeMmaA", generated_options, "dtypeMmaA")
    _append_optional_option(cmd, "-dtypeMmaB", generated_options, "dtypeMmaB")
    _append_optional_bool_option(
        cmd, "-fuseUtccpWithUtcmma", generated_options, "fuseUtccpWithUtcmma"
    )
    _append_optional_bool_option(
        cmd, "-useMaxTmemOverlap", generated_options, "useMaxTmemOverlap"
    )
    _append_optional_bool_option(cmd, "-patchF2fp", generated_options, "patchF2fp")
    _append_optional_option(cmd, "-sfBlockSizeA", generated_options, "sfBlockSizeA")
    _append_optional_option(cmd, "-sfBlockSizeB", generated_options, "sfBlockSizeB")
    _append_optional_option(cmd, "-sfBlockSizeC", generated_options, "sfBlockSizeC")
    _append_bool_option(
        cmd, "-useUnrollLoop2xForMma", kwargs["use_unroll_loop_2x_for_mma"]
    )
    _append_bool_option(
        cmd, "-useTmaStore", _bool_value(generated_options.get("useTmaStore", True))
    )
    _append_option(cmd, "-clampLimit", generated_options.get("clampLimit", 2))
    _append_option(cmd, "-mmaK", kwargs["mma_k"])
    return cmd


def _run_trtllm_gen_benchmark(
    *,
    cmd: list[str],
    cwd: Path,
    timeout_s: int,
) -> float:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "trtllm-gen benchmark failed with return code "
            f"{result.returncode}\ncommand: {shlex.join(cmd)}\n{result.stdout}"
        )
    match = ELAPSED_TIME_RE.search(result.stdout)
    if match is None:
        raise RuntimeError(
            "Could not parse trtllm-gen elapsed time\n"
            f"command: {shlex.join(cmd)}\n{result.stdout}"
        )
    return float(match.group(1)) * 1000.0


def _resolve_template(
    name: str,
    templates: dict[str, dict],
    *,
    stack: tuple[str, ...] = (),
) -> dict:
    if name in stack:
        raise ValueError(f"Template inheritance cycle: {' -> '.join(stack + (name,))}")
    raw = templates[name]
    result = {}
    parent = raw.get("_template")
    if parent is not None:
        result.update(_resolve_template(parent, templates, stack=stack + (name,)))
    result.update({key: value for key, value in raw.items() if key != "_template"})
    return result


def _expanded_json_options(path: Path) -> list[tuple[int, str, int, dict]]:
    with path.open() as handle:
        data = json.load(handle)
    templates = data["templates"]
    expanded = []
    for config_index, raw_config in enumerate(data["configs"]):
        config_comment = raw_config.get("_comment", f"config_{config_index}")
        merged = {}
        parent = raw_config.get("_template")
        if parent is not None:
            merged.update(_resolve_template(parent, templates))
        merged.update(
            {
                key: value
                for key, value in raw_config.items()
                if key not in ("_template", "_comment")
            }
        )

        combos = [({}, 0)]
        for key, value in merged.items():
            if "," in key:
                keys = tuple(item.strip() for item in key.split(",") if item.strip())
                rows = value if isinstance(value, list) else [value]
                choices = []
                for row in rows:
                    if len(row) != len(keys):
                        raise ValueError(
                            f"Config {config_comment}: key group {key} expects "
                            f"{len(keys)} values, got {row}"
                        )
                    choices.append(dict(zip(keys, row, strict=True)))
            elif isinstance(value, list):
                choices = [{key: item} for item in value]
            else:
                choices = [{key: value}]
            combos = [
                ({**base, **choice}, combo_index)
                for combo_index, (base, _) in enumerate(combos)
                for choice in choices
            ]
        for combo_index, (combo, _) in enumerate(combos):
            if TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY in combo:
                if "transpose_mma_output" in combo:
                    raise ValueError(
                        f"Config {config_comment} defines both the TRT-LLM Gen "
                        "transpose option and transpose_mma_output"
                    )
                combo["transpose_mma_output"] = combo.pop(
                    TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY
                )
            expanded.append((config_index, config_comment, combo_index, combo))
    return expanded


def _fp4_base() -> dict:
    return {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": int(DType.E2M1),
        "dtype_b": int(DType.E2M1),
        "dtype_c": int(DType.BF16),
        "sf_bits": 8,
        "mma_k": 64,
        "use_tma_oob_opt": 1,
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        "use_global_scales": 1,
        "bias_type": int(BiasType.M),
    }


def _bf16_base() -> dict:
    return {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": int(DType.BF16),
        "dtype_b": int(DType.BF16),
        "dtype_c": int(DType.BF16),
        "mma_k": 16,
        "use_tma_oob_opt": 1,
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        **uniform_pipeline_stage_overrides(6, tmem_acc_stages=2),
        "tile_scheduler": int(TileScheduler.PERSISTENT),
        "cluster_m": 2,
        "tile_m": 128,
        "tile_n": 128,
        "tile_k": 64,
        "epi_tile_m": 128,
        "mma_m": 256,
        "mma_n": 128,
        "use_unroll_loop_2x_for_mma": 1,
        "use_max_tmem_overlap": 0,
        "epilogue_regs": 168,
        "mma_regs": 96,
        "load_regs": 96,
        "padding_regs": 96,
        "workid_regs": 96,
    }


def _comment_has_activation_semantics(comment: str) -> bool:
    if "FC1" in comment:
        return True
    if "FC2" in comment:
        return False
    raise ValueError(f"Cannot infer FC kind from config comment: {comment}")


def _fp4_json_variant(
    *,
    config_index: int,
    config_comment: str,
    combo_index: int,
    options: dict,
) -> BenchVariant:
    has_activation_semantics = _comment_has_activation_semantics(config_comment)
    tile_n = int(options["tileN"])
    tile_k = int(options["tileK"])
    scheduler = _scheduler_value(options["tileScheduler"])
    generated_mma_stages = int(options["numStagesMma"])
    use_max_tmem_overlap = int(_bool_value(options.get("useMaxTmemOverlap", False)))
    tmem_acc_stages = (
        1 if scheduler == int(TileScheduler.STATIC) else generated_mma_stages
    )
    if use_max_tmem_overlap:
        tmem_acc_stages = 1
    generated_fused_act = _bool_value(options.get("fusedAct", False))
    sf_layout_c = options.get("sfLayoutC")
    bias_type = options.get("biasType")
    use_per_token_sf_a = _bool_value(options.get("usePerTokenSfA", False))
    use_per_token_sf_b = _bool_value(options.get("usePerTokenSfB", False))
    kwargs = {
        **_fp4_base(),
        "dtype_c": _dtype_value(options.get("dtypeC"), default="bf16"),
        "route_act": _route_value(options.get("routeAct", False)),
        "route_sfs_act": _route_value(options.get("routeSfsAct", False)),
        "tile_scheduler": scheduler,
        "act_kind": _act_value(
            generated_fused_act=generated_fused_act,
            act=options.get("act"),
            eltwise_act=options.get("eltwiseActType"),
        ),
        "sf_layout_a": _sf_layout_value(options.get("sfLayoutA")),
        "sf_layout_b": _sf_layout_value(options.get("sfLayoutB")),
        "sf_layout_c": _sf_layout_value(sf_layout_c),
        "cluster_m": int(options.get("clusterDimX", 1)),
        "tile_m": int(options.get("tileM", 128)),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "epi_tile_m": int(options.get("epilogueTileM", 128)),
        "epi_tile_n": int(options["epilogueTileN"]),
        "mma_m": int(options["mmaM"]),
        "mma_n": int(options["mmaN"]),
        "mma_k": int(options.get("mmaK", 64)),
        **uniform_pipeline_stage_overrides(
            int(options["numStages"]),
            tmem_acc_stages=tmem_acc_stages,
        ),
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("useUnrollLoop2xForMma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_max_tmem_overlap": use_max_tmem_overlap,
        "use_tma_oob_opt": int(_bool_value(options.get("useTmaOobOpt", True))),
        "use_tma_store": int(_bool_value(options.get("useTmaStore", False))),
        "use_per_token_sf_a": int(use_per_token_sf_a),
        "use_per_token_sf_b": int(use_per_token_sf_b),
        "per_token_sf_dtype": _per_token_sf_dtype(
            options.get("perTokenSfDtype"),
            use_per_token_sf_a=use_per_token_sf_a,
            use_per_token_sf_b=use_per_token_sf_b,
        ),
    }
    kwargs.update(_oa_clamp_kwargs(options, kwargs))
    if bias_type is not None:
        kwargs["bias_type"] = (
            int(BiasType.M) if bias_type == "m" else int(BiasType.NONE)
        )
    non_epi_regs = int(options.get("numRegsPerThreadNonEpilogueWarp", 48))
    kwargs.update(
        {
            "epilogue_regs": int(options.get("numRegsPerThreadEpilogueWarp", 160)),
            "mma_regs": non_epi_regs,
            "load_regs": non_epi_regs,
            "load_sf_regs": int(options.get("numRegsPerThreadLoadSfB", non_epi_regs)),
            "copy_sf_regs": non_epi_regs,
            "workid_regs": non_epi_regs,
            "padding_regs": non_epi_regs,
        }
    )
    ts_notes = []
    if (
        not has_activation_semantics
        and kwargs["use_unroll_loop_2x_for_mma"]
        and kwargs["cluster_m"] > 1
        and kwargs["tile_n"] >= 128
        and kwargs["tile_scheduler"] == int(TileScheduler.PERSISTENT)
    ):
        ts_notes.append("TS maps output-side clustered tile128+ unroll2x to unroll0")
    return BenchVariant(
        name=(
            f"json_fp4_i{config_index:02d}_v{combo_index:03d}_"
            f"{_slug(config_comment)}_t{tile_n}_k{tile_k}_"
            f"{_scheduler_arg(scheduler)}_u{kwargs['use_unroll_loop_2x_for_mma']}"
        ),
        problem_n=_FC1_N if has_activation_semantics else _FC2_N,
        problem_k=_FC1_K if has_activation_semantics else _FC2_K,
        kwargs=kwargs,
        source="json_fp4",
        config_index=config_index,
        config_comment=config_comment,
        combo_index=combo_index,
        generated_options=options,
        ts_notes="; ".join(ts_notes),
        trtllm_gen_sf_layout_a=str(options.get("sfLayoutA", "128x4")),
        trtllm_gen_sf_layout_b=str(options.get("sfLayoutB", "128x4")),
        trtllm_gen_sf_layout_c=str(options.get("sfLayoutC", "128x4")),
    )


def _mx_json_variant(
    *,
    config_index: int,
    config_comment: str,
    combo_index: int,
    options: dict,
) -> BenchVariant:
    has_activation_semantics = _comment_has_activation_semantics(config_comment)
    dtype_a = _dtype_value(options.get("dtypeA"), default="mxe2m1")
    dtype_b = _dtype_value(options.get("dtypeB"), default="mxe4m3")
    dtype_c = _dtype_value(
        options.get("dtypeC"), default="mxe4m3" if has_activation_semantics else "bf16"
    )
    tile_n = int(options["tileN"])
    tile_k = int(options["tileK"])
    scheduler = _scheduler_value(options["tileScheduler"])
    generated_mma_stages = int(options["numStagesMma"])
    use_max_tmem_overlap = int(_bool_value(options.get("useMaxTmemOverlap", False)))
    tmem_acc_stages = 1 if use_max_tmem_overlap else generated_mma_stages
    generated_fused_act = _bool_value(options.get("fusedAct", False))
    bias_type = options.get("biasType")
    non_epi_regs = int(options.get("numRegsPerThreadNonEpilogueWarp", 48))
    epilogue_regs = int(options.get("numRegsPerThreadEpilogueWarp", 160))
    load_b_regs = int(options.get("numRegsPerThreadLoadB", non_epi_regs))
    load_sfb_regs = int(options.get("numRegsPerThreadLoadSfB", non_epi_regs))
    kwargs = {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "sf_bits": 8,
        "sf_block_size_a": int(options.get("sfBlockSizeA", 32)),
        "sf_block_size_b": int(options.get("sfBlockSizeB", 32)),
        "sf_block_size_c": int(options.get("sfBlockSizeC", 32)),
        "mma_k": int(options.get("mmaK", 32)),
        "route_act": _route_value(options.get("routeAct", False)),
        "route_sfs_act": _route_value(options.get("routeSfsAct", False)),
        "tile_scheduler": scheduler,
        "act_kind": _act_value(
            generated_fused_act=generated_fused_act,
            act=options.get("act"),
            eltwise_act=options.get("eltwiseActType"),
        ),
        "bias_type": int(BiasType.M) if bias_type == "m" else int(BiasType.NONE),
        "sf_layout_a": _sf_layout_value(options.get("sfLayoutA")),
        "sf_layout_b": _sf_layout_value(options.get("sfLayoutB")),
        "sf_layout_c": _sf_layout_value(options.get("sfLayoutC"), default="8x4"),
        "cluster_m": int(options.get("clusterDimX", 1)),
        "tile_m": int(options.get("tileM", 128)),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "epi_tile_m": int(options.get("epilogueTileM", 128)),
        "epi_tile_n": int(options["epilogueTileN"]),
        "mma_m": int(options["mmaM"]),
        "mma_n": int(options["mmaN"]),
        **uniform_pipeline_stage_overrides(
            int(options["numStages"]),
            tmem_acc_stages=tmem_acc_stages,
        ),
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("useUnrollLoop2xForMma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_max_tmem_overlap": use_max_tmem_overlap,
        "use_tma_oob_opt": int(_bool_value(options.get("useTmaOobOpt", True))),
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        "use_tma_store": int(_bool_value(options.get("useTmaStore", True))),
        "epilogue_regs": epilogue_regs,
        "mma_regs": non_epi_regs,
        "load_regs": non_epi_regs,
        "load_sf_regs": non_epi_regs,
        "load_b_regs": load_b_regs,
        "load_sfb_regs": load_sfb_regs,
        "copy_sf_regs": non_epi_regs,
        "workid_regs": non_epi_regs,
        "padding_regs": non_epi_regs,
        "gather_regs": non_epi_regs,
    }
    kwargs.update(_oa_clamp_kwargs(options, kwargs))
    if "numWarpsLoadB" in options:
        kwargs["num_load_b_warps"] = int(options["numWarpsLoadB"])
    if "numWarpsLoadSfB" in options:
        kwargs["num_load_sfb_warps"] = int(options["numWarpsLoadSfB"])
    return BenchVariant(
        name=(
            f"json_mx_i{config_index:02d}_v{combo_index:03d}_"
            f"{_slug(config_comment)}_t{tile_n}_k{tile_k}_"
            f"{_scheduler_arg(scheduler)}_u{kwargs['use_unroll_loop_2x_for_mma']}_"
            f"{_slug(_dtype_arg(dtype_a))}x{_slug(_dtype_arg(dtype_b))}"
        ),
        problem_n=_FC1_N if has_activation_semantics else _FC2_N,
        problem_k=_FC1_K if has_activation_semantics else _FC2_K,
        kwargs=kwargs,
        source="json_mx",
        config_index=config_index,
        config_comment=config_comment,
        combo_index=combo_index,
        generated_options=options,
        trtllm_gen_sf_layout_a=str(options.get("sfLayoutA", "128x4")),
        trtllm_gen_sf_layout_b=str(options.get("sfLayoutB", "128x4")),
        trtllm_gen_sf_layout_c=str(options.get("sfLayoutC", "8x4")),
    )


def _mxfp4_bf16_json_variant(
    *,
    config_index: int,
    config_comment: str,
    combo_index: int,
    options: dict,
) -> BenchVariant:
    has_activation_semantics = _comment_has_activation_semantics(config_comment)
    dtype_a = _dtype_value(options.get("dtypeA"), default="mxe2m1")
    dtype_b = _dtype_value(options.get("dtypeB"), default="bf16")
    dtype_c = _dtype_value(options.get("dtypeC"), default="bf16")
    tile_n = int(options["tileN"])
    tile_k = int(options["tileK"])
    scheduler = _scheduler_value(options["tileScheduler"])
    generated_mma_stages = int(options["numStagesMma"])
    tmem_acc_stages = (
        1 if scheduler == int(TileScheduler.STATIC) else generated_mma_stages
    )
    generated_fused_act = _bool_value(options.get("fusedAct", False))
    bias_type = options.get("biasType")
    non_epi_regs = int(options.get("numRegsPerThreadNonEpilogueWarp", 48))
    kwargs = {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "sf_bits": 8,
        "sf_block_size_a": int(options.get("sfBlockSizeA", 32)),
        "mma_k": int(options.get("mmaK", 16)),
        "route_act": _route_value(options.get("routeAct", False)),
        "route_sfs_act": int(RouteImpl.NONE),
        "tile_scheduler": scheduler,
        "act_kind": _act_value(
            generated_fused_act=generated_fused_act,
            act=options.get("act"),
            eltwise_act=options.get("eltwiseActType"),
        ),
        "bias_type": int(BiasType.M) if bias_type == "m" else int(BiasType.NONE),
        "sf_layout_a": _sf_layout_value(options.get("sfLayoutA")),
        "sf_layout_b": int(SfLayout.R8c4),
        "sf_layout_c": _sf_layout_value(options.get("sfLayoutC"), default="8x4"),
        "cluster_m": int(options.get("clusterDimX", 1)),
        "tile_m": int(options.get("tileM", 128)),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "epi_tile_m": int(options.get("epilogueTileM", 128)),
        "epi_tile_n": int(options["epilogueTileN"]),
        "mma_m": int(options["mmaM"]),
        "mma_n": int(options["mmaN"]),
        **uniform_pipeline_stage_overrides(
            int(options["numStages"]),
            tmem_acc_stages=tmem_acc_stages,
        ),
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("useUnrollLoop2xForMma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_tma_oob_opt": int(_bool_value(options.get("useTmaOobOpt", True))),
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        "use_tma_store": int(_bool_value(options.get("useTmaStore", True))),
        "use_max_tmem_overlap": 0,
        "epilogue_regs": int(options.get("numRegsPerThreadEpilogueWarp", 128)),
        "mma_regs": non_epi_regs,
        "load_regs": non_epi_regs,
        "load_sf_regs": int(options.get("numRegsPerThreadLoadSfA", non_epi_regs)),
        "cast_a_regs": int(options.get("numRegsPerThreadCastA", 160)),
        "copy_sf_regs": non_epi_regs,
        "workid_regs": non_epi_regs,
        "padding_regs": non_epi_regs,
        "gather_regs": non_epi_regs,
    }
    kwargs.update(_oa_clamp_kwargs(options, kwargs))
    return BenchVariant(
        name=(
            f"json_mxfp4_bf16_i{config_index:02d}_v{combo_index:03d}_"
            f"{_slug(config_comment)}_t{tile_n}_k{tile_k}_"
            f"{_scheduler_arg(scheduler)}_u{kwargs['use_unroll_loop_2x_for_mma']}"
        ),
        problem_n=_FC1_N if has_activation_semantics else _FC2_N,
        problem_k=_FC1_K if has_activation_semantics else _FC2_K,
        kwargs=kwargs,
        source="json_mxfp4_bf16",
        config_index=config_index,
        config_comment=config_comment,
        combo_index=combo_index,
        generated_options=options,
        trtllm_gen_sf_layout_a=str(options.get("sfLayoutA", "128x4")),
        trtllm_gen_sf_layout_b=str(options.get("sfLayoutB", "linear")),
        trtllm_gen_sf_layout_c=str(options.get("sfLayoutC", "8x4")),
    )


def _bf16_json_variant(
    *,
    config_index: int,
    config_comment: str,
    combo_index: int,
    options: dict,
) -> BenchVariant:
    has_activation_semantics = _comment_has_activation_semantics(config_comment)
    tile_n = int(options["tileN"])
    tile_k = int(options["tileK"])
    scheduler = _scheduler_value(options["tileScheduler"])
    generated_fused_act = _bool_value(options.get("fusedAct", False))
    bias_type = options.get("biasType")
    ts_skip_reason = None
    if bias_type == "mn":
        ts_skip_reason = "TS biasType=MN is not implemented"
    kwargs = {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": int(DType.BF16),
        "dtype_b": int(DType.BF16),
        "dtype_c": _dtype_value(options.get("dtypeC"), default="bf16"),
        "mma_k": int(options.get("mmaK", 16)),
        "route_act": _route_value(options.get("routeAct", False)),
        "route_sfs_act": int(RouteImpl.NONE),
        "tile_scheduler": scheduler,
        "act_kind": _act_value(
            generated_fused_act=generated_fused_act,
            act=options.get("act"),
            eltwise_act=options.get("eltwiseActType"),
        ),
        "bias_type": int(BiasType.M) if bias_type == "m" else int(BiasType.NONE),
        "cluster_m": int(options.get("clusterDimX", 1)),
        "tile_m": int(options.get("tileM", 128)),
        "tile_n": tile_n,
        "tile_k": tile_k,
        "epi_tile_m": int(options.get("epilogueTileM", 128)),
        "epi_tile_n": int(options["epilogueTileN"]),
        "mma_m": int(options["mmaM"]),
        "mma_n": int(options["mmaN"]),
        **uniform_pipeline_stage_overrides(
            int(options["numStages"]),
            tmem_acc_stages=(
                1
                if scheduler == int(TileScheduler.STATIC)
                else int(options["numStagesMma"])
            ),
        ),
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("useUnrollLoop2xForMma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_tma_oob_opt": int(_bool_value(options.get("useTmaOobOpt", True))),
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        "use_tma_store": int(_bool_value(options.get("useTmaStore", False))),
        "use_max_tmem_overlap": 0,
    }
    kwargs.update(_oa_clamp_kwargs(options, kwargs))
    non_epi_regs = int(options.get("numRegsPerThreadNonEpilogueWarp", 96))
    kwargs.update(
        {
            "epilogue_regs": int(options.get("numRegsPerThreadEpilogueWarp", 128)),
            "mma_regs": non_epi_regs,
            "load_regs": non_epi_regs,
            "padding_regs": non_epi_regs,
            "workid_regs": non_epi_regs,
            "gather_regs": non_epi_regs,
        }
    )

    return BenchVariant(
        name=(
            f"json_bf16_i{config_index:02d}_v{combo_index:03d}_"
            f"{_slug(config_comment)}_t{tile_n}_k{tile_k}_"
            f"{_scheduler_arg(scheduler)}_u{kwargs['use_unroll_loop_2x_for_mma']}"
        ),
        problem_n=_FC1_N if has_activation_semantics else _FC2_N,
        problem_k=_FC1_K if has_activation_semantics else _FC2_K,
        kwargs=kwargs,
        source="json_bf16",
        config_index=config_index,
        config_comment=config_comment,
        combo_index=combo_index,
        generated_options=options,
        ts_skip_reason=ts_skip_reason,
    )


def _fp8_json_variant(
    *,
    config_index: int,
    config_comment: str,
    combo_index: int,
    options: dict,
) -> BenchVariant:
    has_activation_semantics = _comment_has_activation_semantics(config_comment)
    dtype_a = _dtype_value(options.get("dtypeA"), default="e4m3", plain_fp8=True)
    dtype_b = _dtype_value(
        options.get("dtypeB", options.get("dtypeA")),
        default="e4m3",
        plain_fp8=True,
    )
    dtype_c = _dtype_value(
        options.get("dtypeC"),
        default="e4m3" if has_activation_semantics else "bf16",
        plain_fp8=True,
    )
    tile_n = int(options["tileN"])
    scheduler = _scheduler_value(options["tileScheduler"])
    generated_mma_stages = int(options["numStagesMma"])
    tmem_acc_stages = (
        1 if scheduler == int(TileScheduler.STATIC) else generated_mma_stages
    )
    generated_fused_act = _bool_value(options.get("fusedAct", False))
    non_epi_regs = int(options.get("numRegsPerThreadNonEpilogueWarp", 48))
    use_per_token_sf_a = _bool_value(options.get("usePerTokenSfA", False))
    use_per_token_sf_b = _bool_value(options.get("usePerTokenSfB", False))
    kwargs = {
        "batch_mode": int(BatchMode.BATCH_N),
        "dtype_a": dtype_a,
        "dtype_b": dtype_b,
        "dtype_c": dtype_c,
        "mma_k": int(options.get("mmaK", 32)),
        "route_act": _route_value(options.get("routeAct", False)),
        "route_sfs_act": int(RouteImpl.NONE),
        "tile_scheduler": scheduler,
        "act_kind": _act_value(
            generated_fused_act=generated_fused_act,
            act=options.get("act"),
            eltwise_act=options.get("eltwiseActType"),
        ),
        "cluster_m": int(options.get("clusterDimX", 1)),
        "tile_m": int(options.get("tileM", 128)),
        "tile_n": tile_n,
        "tile_k": int(options["tileK"]),
        "epi_tile_m": int(options.get("epilogueTileM", 128)),
        "epi_tile_n": int(options["epilogueTileN"]),
        "mma_m": int(options["mmaM"]),
        "mma_n": int(options["mmaN"]),
        **uniform_pipeline_stage_overrides(
            int(options["numStages"]),
            tmem_acc_stages=tmem_acc_stages,
        ),
        "use_unroll_loop_2x_for_mma": int(
            _bool_value(options.get("useUnrollLoop2xForMma", False))
        ),
        "transpose_mma_output": int(
            _bool_value(options.get("transpose_mma_output", True))
        ),
        "use_global_scales": int(_bool_value(options.get("useGlobalScales", True))),
        "use_per_token_sf_a": int(use_per_token_sf_a),
        "use_per_token_sf_b": int(use_per_token_sf_b),
        "per_token_sf_dtype": _per_token_sf_dtype(
            options.get("perTokenSfDtype"),
            use_per_token_sf_a=use_per_token_sf_a,
            use_per_token_sf_b=use_per_token_sf_b,
        ),
        "use_tma_oob_opt": int(_bool_value(options.get("useTmaOobOpt", True))),
        "use_early_exit": 1,
        "use_clc_fast_drain": 0,
        "use_tma_store": int(_bool_value(options.get("useTmaStore", True))),
        "use_max_tmem_overlap": 0,
        "epilogue_regs": int(options.get("numRegsPerThreadEpilogueWarp", 160)),
        "mma_regs": non_epi_regs,
        "load_regs": non_epi_regs,
        "padding_regs": non_epi_regs,
        "workid_regs": non_epi_regs,
        "gather_regs": non_epi_regs,
    }
    kwargs.update(_oa_clamp_kwargs(options, kwargs))
    if "numEpilogueWarps" in options:
        kwargs["num_epilogue_warps"] = int(options["numEpilogueWarps"])
    if "numWarpsLoadB" in options:
        kwargs["num_load_b_warps"] = int(options["numWarpsLoadB"])

    return BenchVariant(
        name=(
            f"json_fp8_i{config_index:02d}_v{combo_index:03d}_"
            f"{_slug(config_comment)}_t{tile_n}_k{kwargs['tile_k']}_"
            f"{_scheduler_arg(scheduler)}_u{kwargs['use_unroll_loop_2x_for_mma']}"
        ),
        problem_n=_FC1_N if has_activation_semantics else _FC2_N,
        problem_k=_FC1_K if has_activation_semantics else _FC2_K,
        kwargs=kwargs,
        source="json_fp8",
        config_index=config_index,
        config_comment=config_comment,
        combo_index=combo_index,
        generated_options=options,
    )


def _json_variants(
    selection: str, trtllm_gen_root: Path | None
) -> dict[str, BenchVariant]:
    if trtllm_gen_root is None:
        raise ValueError(
            "--variant-source json/all requires --trtllm-gen-root "
            f"or {TRTLLM_GEN_ROOT_ENV}"
        )
    config_paths = {
        name: trtllm_gen_root / filename
        for name, filename in TRTLLM_GEN_CONFIG_NAMES.items()
    }
    result = {}
    if selection in ("fp4", "both", "all"):
        for config_index, comment, combo_index, options in _expanded_json_options(
            config_paths["fp4"]
        ):
            variant = _with_ts_skip_reason(
                _fp4_json_variant(
                    config_index=config_index,
                    config_comment=comment,
                    combo_index=combo_index,
                    options=options,
                )
            )
            result[variant.name] = variant
    if selection in ("bf16", "both", "all"):
        for config_index, comment, combo_index, options in _expanded_json_options(
            config_paths["bf16"]
        ):
            if options.get("biasType") == "mn":
                continue
            variant = _with_ts_skip_reason(
                _bf16_json_variant(
                    config_index=config_index,
                    config_comment=comment,
                    combo_index=combo_index,
                    options=options,
                )
            )
            result[variant.name] = variant
    if selection in ("mx", "all"):
        for config_index, comment, combo_index, options in _expanded_json_options(
            config_paths["mx"]
        ):
            variant = _with_ts_skip_reason(
                _mx_json_variant(
                    config_index=config_index,
                    config_comment=comment,
                    combo_index=combo_index,
                    options=options,
                )
            )
            result[variant.name] = variant
    if selection in ("mxfp4_bf16", "all"):
        for config_index, comment, combo_index, options in _expanded_json_options(
            config_paths["mxfp4_bf16"]
        ):
            variant = _with_ts_skip_reason(
                _mxfp4_bf16_json_variant(
                    config_index=config_index,
                    config_comment=comment,
                    combo_index=combo_index,
                    options=options,
                )
            )
            result[variant.name] = variant
    if selection in ("fp8", "all"):
        for config_index, comment, combo_index, options in _expanded_json_options(
            config_paths["fp8"]
        ):
            variant = _with_ts_skip_reason(
                _fp8_json_variant(
                    config_index=config_index,
                    config_comment=comment,
                    combo_index=combo_index,
                    options=options,
                )
            )
            result[variant.name] = variant
    return result


def _variants() -> dict[str, BenchVariant]:
    fp4_base = _fp4_base()
    bf16_base = _bf16_base()
    variants = {
        "fp4_fc2_ll_t8_k512": BenchVariant(
            name="fp4_fc2_ll_t8_k512",
            problem_n=_FC2_N,
            problem_k=_FC2_K,
            trtllm_gen_sf_layout_a="128x4",
            trtllm_gen_sf_layout_b="8x4",
            trtllm_gen_sf_layout_c="8x4",
            kwargs={
                **fp4_base,
                "route_act": int(RouteImpl.NONE),
                "route_sfs_act": int(RouteImpl.NONE),
                "tile_scheduler": int(TileScheduler.STATIC),
                "act_kind": int(ActKind.NONE),
                "sf_layout_a": int(SfLayout.R128c4),
                "sf_layout_b": int(SfLayout.R8c4),
                "sf_layout_c": int(SfLayout.R8c4),
                "cluster_m": 1,
                "tile_m": 128,
                "tile_n": 8,
                "tile_k": 512,
                "epi_tile_m": 128,
                "epi_tile_n": 8,
                "mma_m": 128,
                "mma_n": 8,
                **uniform_pipeline_stage_overrides(5, tmem_acc_stages=1),
                "use_unroll_loop_2x_for_mma": 1,
            },
        ),
        "fp4_fc2_ht_t64_k512": BenchVariant(
            name="fp4_fc2_ht_t64_k512",
            problem_n=_FC2_N,
            problem_k=_FC2_K,
            trtllm_gen_sf_layout_a="128x4",
            trtllm_gen_sf_layout_b="8x4",
            trtllm_gen_sf_layout_c="128x4",
            kwargs={
                **fp4_base,
                "route_act": int(RouteImpl.NONE),
                "route_sfs_act": int(RouteImpl.NONE),
                "tile_scheduler": int(TileScheduler.PERSISTENT),
                "act_kind": int(ActKind.NONE),
                "sf_layout_a": int(SfLayout.R128c4),
                "sf_layout_b": int(SfLayout.R8c4),
                "sf_layout_c": int(SfLayout.R128c4),
                "cluster_m": 2,
                "tile_m": 128,
                "tile_n": 64,
                "tile_k": 512,
                "epi_tile_m": 128,
                "epi_tile_n": 64,
                "mma_m": 256,
                "mma_n": 64,
                **uniform_pipeline_stage_overrides(4, tmem_acc_stages=2),
                "use_unroll_loop_2x_for_mma": 0,
            },
        ),
        "fp4_fc1_ll_t8_k512": BenchVariant(
            name="fp4_fc1_ll_t8_k512",
            problem_n=_FC1_N,
            problem_k=_FC1_K,
            trtllm_gen_sf_layout_a="128x4",
            trtllm_gen_sf_layout_b="linear",
            trtllm_gen_sf_layout_c="8x4",
            kwargs={
                **fp4_base,
                "dtype_c": int(DType.E2M1),
                "route_act": int(RouteImpl.TMA),
                "route_sfs_act": int(RouteImpl.TMA),
                "tile_scheduler": int(TileScheduler.STATIC),
                "act_kind": int(ActKind.SWIGLU),
                "sf_layout_a": int(SfLayout.R128c4),
                "sf_layout_b": int(SfLayout.LINEAR),
                "sf_layout_c": int(SfLayout.R8c4),
                "cluster_m": 1,
                "tile_m": 128,
                "tile_n": 8,
                "tile_k": 512,
                "epi_tile_m": 128,
                "epi_tile_n": 8,
                "mma_m": 128,
                "mma_n": 8,
                **uniform_pipeline_stage_overrides(5, tmem_acc_stages=1),
                "use_unroll_loop_2x_for_mma": 1,
            },
        ),
        "fp4_fc1_ht_t128_k256_ldgsts": BenchVariant(
            name="fp4_fc1_ht_t128_k256_ldgsts",
            problem_n=_FC1_N,
            problem_k=_FC1_K,
            trtllm_gen_sf_layout_a="128x4",
            trtllm_gen_sf_layout_b="linear",
            trtllm_gen_sf_layout_c="128x4",
            kwargs={
                **fp4_base,
                "dtype_c": int(DType.E2M1),
                "route_act": int(RouteImpl.TMA),
                "route_sfs_act": int(RouteImpl.LDGSTS),
                "tile_scheduler": int(TileScheduler.PERSISTENT),
                "act_kind": int(ActKind.SWIGLU),
                "sf_layout_a": int(SfLayout.R128c4),
                "sf_layout_b": int(SfLayout.LINEAR),
                "sf_layout_c": int(SfLayout.R128c4),
                "cluster_m": 2,
                "tile_m": 128,
                "tile_n": 128,
                "tile_k": 256,
                "epi_tile_m": 128,
                "epi_tile_n": 32,
                "mma_m": 256,
                "mma_n": 128,
                **uniform_pipeline_stage_overrides(6, tmem_acc_stages=2),
                "use_unroll_loop_2x_for_mma": 1,
                "epilogue_regs": 128,
                "mma_regs": 48,
                "load_regs": 48,
                "load_sf_regs": 48,
                "copy_sf_regs": 48,
                "workid_regs": 48,
                "padding_regs": 48,
                "gather_regs": 48,
            },
        ),
        "bf16_fc1_ht_t128_k64": BenchVariant(
            name="bf16_fc1_ht_t128_k64",
            problem_n=_FC1_N,
            problem_k=_FC1_K,
            kwargs={
                **bf16_base,
                "route_act": int(RouteImpl.TMA),
                "route_sfs_act": int(RouteImpl.NONE),
                "act_kind": int(ActKind.SWIGLU),
                "epi_tile_n": 128,
                "gather_regs": 96,
            },
        ),
        "bf16_fc2_ht_t128_k64": BenchVariant(
            name="bf16_fc2_ht_t128_k64",
            problem_n=_FC2_N,
            problem_k=_FC2_K,
            kwargs={
                **bf16_base,
                "route_act": int(RouteImpl.NONE),
                "route_sfs_act": int(RouteImpl.NONE),
                "act_kind": int(ActKind.NONE),
                "epi_tile_n": 64,
                **uniform_pipeline_stage_overrides(8, tmem_acc_stages=2),
            },
        ),
    }
    return {name: _with_ts_skip_reason(variant) for name, variant in variants.items()}


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item)


def _write_row(writer: csv.DictWriter, row: dict) -> None:
    writer.writerow(row)
    status = row.get("status", "ok")
    if status == "ok":
        print(
            f"{row['runner']},{row['variant']},tokens={row['num_tokens']},"
            f"time_us={float(row['time_us']):.3f}"
        )
    else:
        print(
            f"{row['runner']},{row['variant']},tokens={row['num_tokens']},"
            f"status={status},error={row.get('error', '')}"
        )


def _base_row(
    *,
    runner: str,
    variant: BenchVariant,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    num_activated_experts: int,
    warmup_iterations: int,
    iterations: int,
) -> dict:
    return {
        "runner": runner,
        "variant": variant.name,
        "source": variant.source,
        "config_index": "" if variant.config_index is None else variant.config_index,
        "config_comment": variant.config_comment,
        "combo_index": "" if variant.combo_index is None else variant.combo_index,
        "num_tokens": num_tokens,
        "num_experts": num_experts,
        "top_k": top_k,
        "num_activated_experts": num_activated_experts,
        "problem_n": variant.problem_n,
        "problem_k": variant.problem_k,
        "time_us": "",
        "status": "ok",
        "error": "",
        "warmup_iterations": warmup_iterations,
        "iterations": iterations,
        "ts_notes": variant.ts_notes,
        "command": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default=",".join(str(v) for v in DEFAULT_TOKENS))
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Model hidden dimension. FC1_K=hidden, FC2_N=hidden.",
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=DEFAULT_INTERMEDIATE_SIZE,
        help="MoE intermediate dimension. FC1_N=2*intermediate (SwiGLU), FC2_K=intermediate.",
    )
    parser.add_argument(
        "--variant-source",
        choices=("curated", "json", "all"),
        default="curated",
        help="Use curated smoke variants, expanded trtllm-gen JSON variants, or both.",
    )
    parser.add_argument(
        "--json-configs",
        choices=("fp4", "bf16", "mx", "mxfp4_bf16", "fp8", "both", "all"),
        default="both",
        help=(
            "Which trtllm-gen config JSONs to expand when --variant-source includes "
            "json. 'both' preserves the FP4+BF16 legacy selection; 'all' adds MX, "
            "MXFP4/BF16, and FP8."
        ),
    )
    parser.add_argument(
        "--variants",
        default="fp4_fc2_ll_t8_k512",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument("--warmup-iterations", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--num-rotated-buffers",
        type=int,
        default=0,
        help=(
            "TS workspace rotation count. 0 disables rotation and passes "
            "-rotateBuffers false to trtllm-gen for L2-cache parity."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument(
        "--runner",
        choices=("ts", "trtllm-gen", "both"),
        default="ts",
        help="Benchmark TS, the generated trtllm-gen binary, or both.",
    )
    parser.add_argument(
        "--trtllm-gen-root",
        type=Path,
        default=_trtllm_gen_root_from_env(),
        help=f"Path to trtllm-gen checkout. Defaults to ${TRTLLM_GEN_ROOT_ENV}.",
    )
    parser.add_argument(
        "--trtllm-gen-build-dir",
        type=Path,
        default=_trtllm_gen_build_dir_from_env(),
        help=(
            "Build directory relative to --trtllm-gen-root unless absolute. "
            f"Defaults to ${TRTLLM_GEN_BUILD_DIR_ENV}."
        ),
    )
    parser.add_argument("--trtllm-gen-timeout-s", type=int, default=180)
    parser.add_argument("--trtllm-gen-use-ccache", action="store_true")
    parser.add_argument("--trtllm-gen-no-cuda-graph", action="store_true")
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Bracket only the TS benchmark call with cudaProfilerStart/Stop.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Abort on the first benchmark failure instead of recording an error row.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    configure_shapes(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
    )

    variants = {}
    if args.variant_source in ("curated", "all"):
        variants.update(_variants())
    if args.variant_source in ("json", "all"):
        variants.update(_json_variants(args.json_configs, args.trtllm_gen_root))
    variant_names = (
        tuple(variants.keys())
        if args.variants == "all"
        else tuple(args.variants.split(","))
    )
    missing = [name for name in variant_names if name not in variants]
    if missing:
        raise ValueError(f"Unknown variants: {missing}. Available: {sorted(variants)}")

    output_file = args.out_csv
    if output_file is None:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output_file = Path("/tmp") / f"ts_ds_r1_batched_gemm_{stamp}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "runner",
        "variant",
        "source",
        "config_index",
        "config_comment",
        "combo_index",
        "num_tokens",
        "num_experts",
        "top_k",
        "num_activated_experts",
        "problem_n",
        "problem_k",
        "time_us",
        "status",
        "error",
        "warmup_iterations",
        "iterations",
        "ts_notes",
        "command",
    ]
    trtllm_gen_binary = None
    if args.runner in ("trtllm-gen", "both"):
        if args.trtllm_gen_root is None:
            raise ValueError(
                "--runner trtllm-gen/both requires --trtllm-gen-root "
                f"or {TRTLLM_GEN_ROOT_ENV}"
            )
        if args.trtllm_gen_build_dir is None:
            raise ValueError(
                "--runner trtllm-gen/both requires --trtllm-gen-build-dir "
                f"or {TRTLLM_GEN_BUILD_DIR_ENV}"
            )
        trtllm_gen_build_dir = args.trtllm_gen_build_dir
        if not trtllm_gen_build_dir.is_absolute():
            trtllm_gen_build_dir = args.trtllm_gen_root / trtllm_gen_build_dir
        trtllm_gen_binary = trtllm_gen_build_dir / "kernels/BatchedGemm/BatchedGemm"
    if trtllm_gen_binary is not None and not trtllm_gen_binary.exists():
        raise FileNotFoundError(
            f"trtllm-gen binary does not exist: {trtllm_gen_binary}\n"
            "Build trtllm-gen with `. setup_repo.sh` or the BatchedGemm target first."
        )

    with output_file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for variant_name in variant_names:
            variant = variants[variant_name]
            for num_tokens in _parse_csv_ints(args.tokens):
                num_activated_experts = _num_activated_experts(
                    args.num_experts, num_tokens, args.top_k
                )
                if args.dry_run:
                    if args.runner in ("ts", "both"):
                        action = "skip" if variant.ts_skip_reason else "run"
                        suffix = (
                            f", reason={variant.ts_skip_reason}"
                            if variant.ts_skip_reason
                            else ""
                        )
                        print(
                            f"would {action} ts {variant.name}: tokens={num_tokens}, "
                            f"experts={args.num_experts}, top_k={args.top_k}, "
                            f"activated_experts={num_activated_experts}, "
                            f"n={variant.problem_n}, k={variant.problem_k}{suffix}"
                        )
                    if args.runner in ("trtllm-gen", "both"):
                        cmd = _trtllm_gen_command(
                            binary=trtllm_gen_binary,
                            variant=variant,
                            num_tokens=num_tokens,
                            num_experts=args.num_experts,
                            top_k=args.top_k,
                            warmup_iters=args.warmup_iterations,
                            bench_iters=args.iterations,
                            num_rotated_buffers=args.num_rotated_buffers,
                            use_ccache=args.trtllm_gen_use_ccache,
                            use_cuda_graph=not args.trtllm_gen_no_cuda_graph,
                        )
                        print(f"would run trtllm-gen: {shlex.join(cmd)}")
                    continue
                if args.runner in ("ts", "both"):
                    row = _base_row(
                        runner="ts",
                        variant=variant,
                        num_tokens=num_tokens,
                        num_experts=args.num_experts,
                        top_k=args.top_k,
                        num_activated_experts=num_activated_experts,
                        warmup_iterations=args.warmup_iterations,
                        iterations=args.iterations,
                    )
                    if variant.ts_skip_reason:
                        row["status"] = "skipped"
                        row["error"] = variant.ts_skip_reason
                    else:
                        try:
                            row["time_us"] = benchmark(
                                num_experts=args.num_experts,
                                num_tokens=num_tokens,
                                top_k=args.top_k,
                                problem_n=variant.problem_n,
                                problem_k=variant.problem_k,
                                seed=args.seed,
                                warmup_iters=args.warmup_iterations,
                                bench_iters=args.iterations,
                                num_rotated_buffers=args.num_rotated_buffers,
                                cuda_profiler_range=args.cuda_profiler_range,
                                **_ts_kwargs(variant.kwargs),
                            )
                        except Exception as exc:
                            if args.fail_fast:
                                raise
                            row["status"] = "error"
                            row["error"] = str(exc)[:4000]
                    _write_row(writer, row)
                    handle.flush()
                if args.runner in ("trtllm-gen", "both"):
                    cmd = _trtllm_gen_command(
                        binary=trtllm_gen_binary,
                        variant=variant,
                        num_tokens=num_tokens,
                        num_experts=args.num_experts,
                        top_k=args.top_k,
                        warmup_iters=args.warmup_iterations,
                        bench_iters=args.iterations,
                        num_rotated_buffers=args.num_rotated_buffers,
                        use_ccache=args.trtllm_gen_use_ccache,
                        use_cuda_graph=not args.trtllm_gen_no_cuda_graph,
                    )
                    row = _base_row(
                        runner="trtllm-gen",
                        variant=variant,
                        num_tokens=num_tokens,
                        num_experts=args.num_experts,
                        top_k=args.top_k,
                        num_activated_experts=num_activated_experts,
                        warmup_iterations=args.warmup_iterations,
                        iterations=args.iterations,
                    )
                    row["command"] = shlex.join(cmd)
                    try:
                        row["time_us"] = _run_trtllm_gen_benchmark(
                            cmd=cmd,
                            cwd=args.trtllm_gen_root,
                            timeout_s=args.trtllm_gen_timeout_s,
                        )
                    except Exception as exc:
                        if args.fail_fast:
                            raise
                        row["status"] = "error"
                        row["error"] = str(exc)[:4000]
                    _write_row(writer, row)
                    handle.flush()
    print(f"wrote {output_file}")


if __name__ == "__main__":
    main()
