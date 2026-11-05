# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
AllReduce Communication Benchmark Routine

This module provides benchmarking for AllReduce fusion operations using
FlashInfer's unified AllReduce API (create_allreduce_fusion_workspace +
allreduce_fusion). Designed to run with mpirun for multi-GPU benchmarking.

Supports backends: trtllm, mnnvl, mnnvl-cutedsl, auto (several may be given)
Supports patterns: every AllReduceFusionPattern (see PATTERN_NAME_TO_CODE);
    BACKEND_PATTERNS lists which backend implements which pattern.

Launch examples:
    # Basic allreduce with auto backend
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096

    # With specific backend
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --ar_backend mnnvl

    # AllReduce + Residual + RMSNorm fusion
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --pattern ar_residual_rmsnorm

    # With validation (recommended for first run)
    mpirun -np 8 python benchmarks/flashinfer_benchmark.py \
        --routine allreduce_fusion \
        --num_tokens 64 --hidden_size 4096 \
        --validate

Options:
    --ar_backend auto|trtllm|mnnvl|mnnvl-cutedsl : One or more backends (default: auto)
    --pattern <name>               : Fusion pattern (default: allreduce)
    --layout_code                  : NVFP4 scale-factor layout for the fp4 patterns
    --block_quant_group_size       : Group size for the per-token-group FP8 patterns
    --weight_bias                  : 1.0 selects the Gemma / Qwen3.5 RMSNorm
    --top_k, --shared_expert       : MoE reduction / finalize geometry
    --validate                     : Run correctness validation before benchmarking

Note: oneshot and twoshot are both benchmarked and reported, except for the MoE
patterns and the mnnvl-cutedsl backend, which choose their protocol internally.
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from mpi4py import MPI

from flashinfer.comm import (
    AllReduceFusionPattern,
    AllReduceFusionWorkspace,
    QuantizationSFLayout,
    allreduce_fusion,
    compute_fp4_swizzled_layout_sf_size,
    create_allreduce_fusion_workspace,
    create_shared_buffer,
    free_shared_buffer,
    vllm_all_reduce,
    vllm_dispose,
    vllm_init_custom_ar,
    vllm_meta_size,
    vllm_register_buffer,
)
from flashinfer.comm.mnnvl import MnnvlMemory, TorchDistBackend
from flashinfer.norm import rmsnorm
from flashinfer.testing.utils import bench_gpu_time

try:
    from .flashinfer_benchmark_utils import (
        dtype_str_to_torch_dtype,
        print_perf_metrics,
        warn_if_pdl_unsupported,
    )
except ImportError:
    from flashinfer_benchmark_utils import (
        dtype_str_to_torch_dtype,
        print_perf_metrics,
        warn_if_pdl_unsupported,
    )

_P = AllReduceFusionPattern
PATTERN_NAME_TO_CODE = {
    "allreduce": _P.kAllReduce,
    "ar_residual_rmsnorm": _P.kARResidualRMSNorm,
    "ar_residual_rmsnorm_fp8": _P.kARResidualRMSNormFP8Quant,
    "ar_residual_rmsnorm_fp4": _P.kARResidualRMSNormFP4Quant,
    "ar_residual_rmsnorm_out_fp8": _P.kARResidualRMSNormOutFP8Quant,
    "ar_residual_rmsnorm_out_fp4": _P.kARResidualRMSNormOutFP4Quant,
    "moe_reduction": _P.kMoEReductionARResidualRMSNorm,
    "moe_finalize": _P.kMoEFinalizeARResidualRMSNorm,
    "ar_residual_rmsnorm_group_fp8": _P.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
    "ar_residual_rmsnorm_out_group_fp8": (
        _P.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant
    ),
    "ar_residual_rmsnorm_dynamic_fp8": _P.kARResidualRMSNormDynamicFP8Quant,
    "ar_residual_rmsnorm_out_dynamic_fp8": _P.kARResidualRMSNormOutDynamicFP8Quant,
    "moe_finalize_fp8": _P.kMoEFinalizeARResidualRMSNormFP8Quant,
    "ar_residual_rmsnorm_mxfp8": _P.kARResidualRMSNormMXFP8Quant,
    "ar_residual_rmsnorm_out_mxfp8": _P.kARResidualRMSNormOutMXFP8Quant,
}

PATTERN_CODE_TO_NAME = {v: k for k, v in PATTERN_NAME_TO_CODE.items()}

# Patterns each workspace type implements (flashinfer/comm/allreduce.py dispatch;
# mnnvl-cutedsl: flashinfer/comm/mnnvl_cutedsl_ar.py).
BACKEND_PATTERNS = {
    # allreduce.py raises for pattern 12 on a TRT-LLM workspace.
    "trtllm": set(PATTERN_NAME_TO_CODE.values())
    - {_P.kMoEFinalizeARResidualRMSNormFP8Quant},
    # allreduce.py mnnvl_quant_patterns
    "mnnvl": {
        _P.kAllReduce,
        _P.kARResidualRMSNorm,
        _P.kARResidualRMSNormFP8Quant,
        _P.kARResidualRMSNormFP4Quant,
        _P.kARResidualRMSNormOutFP8Quant,
        _P.kARResidualRMSNormOutFP4Quant,
        _P.kARResidualRMSNormDynamicFP8Quant,
        _P.kARResidualRMSNormOutDynamicFP8Quant,
        _P.kARResidualRMSNormMXFP8Quant,
        _P.kARResidualRMSNormOutMXFP8Quant,
    },
    # mnnvl_cutedsl_ar.py _BF16_PATTERNS / _FP8_PATTERNS / _PLAIN_ALL_REDUCE_PATTERN;
    # the workspace is compiled for one output mode (see _cutedsl_workspace_kwargs).
    "mnnvl-cutedsl": {
        _P.kAllReduce,
        _P.kARResidualRMSNorm,
        _P.kARResidualRMSNormFP8Quant,
        _P.kARResidualRMSNormOutFP8Quant,
        _P.kMoEFinalizeARResidualRMSNorm,
        _P.kARResidualRMSNormDynamicFP8Quant,
        _P.kARResidualRMSNormOutDynamicFP8Quant,
        _P.kMoEFinalizeARResidualRMSNormFP8Quant,
    },
    # Unfused all-reduce APIs (PlainAllReduceWorkspace).
    "vllm": {_P.kAllReduce},
    "quantized_fp8": {_P.kAllReduce},
}
PLAIN_BACKENDS = ("vllm", "quantized_fp8")

STATIC_FP8_PATTERNS = {
    _P.kARResidualRMSNormFP8Quant,
    _P.kARResidualRMSNormOutFP8Quant,
    _P.kMoEFinalizeARResidualRMSNormFP8Quant,
}
MXFP8_PATTERNS = {_P.kARResidualRMSNormMXFP8Quant, _P.kARResidualRMSNormOutMXFP8Quant}
FP4_PATTERNS = {_P.kARResidualRMSNormFP4Quant, _P.kARResidualRMSNormOutFP4Quant}
GROUP_FP8_PATTERNS = {
    _P.kARResidualRMSNormPerTokenGroupFP8PackedQuant,
    _P.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
}
DYNAMIC_FP8_PATTERNS = {
    _P.kARResidualRMSNormDynamicFP8Quant,
    _P.kARResidualRMSNormOutDynamicFP8Quant,
}
MOE_PATTERNS = {
    _P.kMoEReductionARResidualRMSNorm,
    _P.kMoEFinalizeARResidualRMSNorm,
    _P.kMoEFinalizeARResidualRMSNormFP8Quant,
}
MOE_FINALIZE_PATTERNS = {
    _P.kMoEFinalizeARResidualRMSNorm,
    _P.kMoEFinalizeARResidualRMSNormFP8Quant,
}
# Quant patterns that also materialize the normalized (pre-quant) output.
OUT_QUANT_PATTERNS = {
    _P.kARResidualRMSNormOutFP8Quant,
    _P.kARResidualRMSNormOutFP4Quant,
    _P.kARResidualRMSNormOutPerTokenGroupFP8PackedQuant,
    _P.kARResidualRMSNormOutDynamicFP8Quant,
    _P.kARResidualRMSNormOutMXFP8Quant,
}

LAYOUT_NAME_TO_CODE = {
    "swizzled_128x4": QuantizationSFLayout.SWIZZLED_128x4,
    "swizzled_8x4": QuantizationSFLayout.SWIZZLED_8x4,
    "linear": QuantizationSFLayout.LINEAR,
}

RMS_EPS = 1e-5


class PlainAllReduceWorkspace:
    """Drive an unfused all-reduce API like a fusion workspace (pattern allreduce).

    vllm:          flashinfer.comm.vllm_all_reduce over CUDA-IPC buffers (ws <= 8,
                   even). Out-of-place: the input is copied into the registered
                   buffer first. Timed eagerly: under stream capture the kernel defers
                   to graph-registered inputs (register_graph_buffers after capture),
                   which bench_gpu_time cannot do between its capture and replay.
    quantized_fp8: flashinfer.comm.quantized_all_reduce (FP8 E4M3 two-shot with a
                   scale per 256 elements, bf16 in/out, cc >= 9.0, ws a power of 2).
    """

    VLLM_NUM_CTAS = 36  # vLLM's block limit (kMaxBlocks).

    def __init__(self, backend, rank, max_token_num, hidden_dim, dtype):
        import torch.distributed as dist

        self.backend = backend
        self.group = dist.group.WORLD
        self.max_bytes = (
            max_token_num * hidden_dim * torch.tensor([], dtype=dtype).element_size()
        )
        self.graph_capturable = backend != "vllm"
        self._fa = None
        if backend == "vllm":
            self._meta_ptrs = create_shared_buffer(
                vllm_meta_size() + self.max_bytes, group=self.group
            )
            self._rank_data = torch.empty(
                8 * 1024 * 1024, dtype=torch.uint8, device="cuda"
            )
            self._buffer_ptrs = create_shared_buffer(self.max_bytes, group=self.group)
            self._fa = vllm_init_custom_ar(self._meta_ptrs, self._rank_data, rank, True)
            vllm_register_buffer(self._fa, self._buffer_ptrs)
            self._reg_buffer = self._buffer_ptrs[rank]
        else:
            # Lazy: flashinfer.comm exposes it lazily too (it pulls in Triton).
            from flashinfer.comm import quantized_all_reduce

            self._quantized_all_reduce = quantized_all_reduce
            # The first call allocates the symmetric buffer collectively; do it
            # here, at the largest size, so it never happens inside a capture.
            warm = torch.zeros((max_token_num, hidden_dim), dtype=dtype, device="cuda")
            quantized_all_reduce(warm, self.group)
            torch.cuda.synchronize()

    def is_buffer_size_sufficient(self, tp_size, num_tokens, hidden_dim, dtype, *_):
        size = num_tokens * hidden_dim * torch.tensor([], dtype=dtype).element_size()
        return size <= self.max_bytes

    def all_reduce(self, inp, out):
        if self.backend == "vllm":
            vllm_all_reduce(
                self._fa,
                inp,
                out,
                self._reg_buffer,
                self.max_bytes,
                self.VLLM_NUM_CTAS,
            )
        else:
            self._quantized_all_reduce(inp, self.group, output=out)

    def destroy(self):
        if self._fa is not None:
            torch.cuda.synchronize()
            vllm_dispose(self._fa)
            free_shared_buffer(self._buffer_ptrs, self.group)
            free_shared_buffer(self._meta_ptrs, self.group)
            self._fa = None


DEFAULT_TOP_K = 8
DEFAULT_BLOCK_QUANT_GROUP_SIZE = 128


def _top_k(args) -> int:
    """top_k for the MoE patterns / mnnvl-cutedsl profile ("" means unset)."""
    return args.top_k or DEFAULT_TOP_K


def _group_size(args) -> int:
    """Group size for the per-token-group FP8 patterns ("" means unset)."""
    return args.block_quant_group_size or DEFAULT_BLOCK_QUANT_GROUP_SIZE


def _layout_name(pattern_code: int, args) -> str:
    """NVFP4/MXFP8 scale layout name used by a pattern ("" for other patterns)."""
    if pattern_code not in FP4_PATTERNS | MXFP8_PATTERNS:
        return ""
    return args.layout_code or "swizzled_128x4"


def _cutedsl_workspace_kwargs(pattern_code: int) -> dict:
    """Compile-time output mode of an MNNVL CuTe DSL workspace serving a pattern."""
    if pattern_code == _P.kAllReduce:
        return {"apply_rms_norm": False, "add_residual": False}
    if pattern_code in (_P.kARResidualRMSNorm, _P.kMoEFinalizeARResidualRMSNorm):
        return {}
    kw = {"output_dtype": torch.float8_e4m3fn}
    if pattern_code in DYNAMIC_FP8_PATTERNS:
        kw["dynamic_fp8"] = True
    if pattern_code in OUT_QUANT_PATTERNS:
        kw["write_norm_output"] = True
    return kw


def _build_pattern_operands(
    pattern_code: int,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    args,
    device,
    residual: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
):
    """Allocate the input and the allreduce_fusion keyword operands for a pattern.

    Returns (input, kwargs, out_key): kwargs excludes input/workspace/pattern/
    launch_with_pdl/use_oneshot; out_key names the kwarg holding the primary output.
    """
    x = torch.ones((num_tokens, hidden_size), dtype=input_dtype, device=device)
    if pattern_code == _P.kAllReduce:
        return x, {"output": torch.empty_like(x)}, "output"

    if residual is None:
        residual = torch.randn_like(x)
    if norm_weight is None:
        norm_weight = torch.randn((hidden_size,), dtype=input_dtype, device=device)
    kwargs = {
        "residual_in": residual,
        "residual_out": torch.empty_like(x),
        "rms_gamma": norm_weight,
        "rms_eps": RMS_EPS,
    }
    if args.weight_bias != 0.0:
        kwargs["weight_bias"] = args.weight_bias
    if args.fp32_acc:
        kwargs["fp32_acc"] = True

    top_k = _top_k(args)
    if pattern_code == _P.kMoEReductionARResidualRMSNorm:
        # Shapes follow tests/comm/test_allreduce_fusion_moe_unified_api.py.
        kwargs["norm_out"] = torch.empty_like(x)
        kwargs["moe_reduction_device_num_experts"] = top_k
        kwargs["moe_reduction_scale_input"] = torch.randn(
            (num_tokens, top_k), dtype=torch.float32, device=device
        )
        kwargs["moe_reduction_active_experts_token_input"] = torch.randn(
            (num_tokens * top_k, hidden_size), dtype=input_dtype, device=device
        )
        kwargs["moe_reduction_token_input"] = x
        return x, kwargs, "norm_out"

    if pattern_code in MOE_FINALIZE_PATTERNS:
        # Permuted expert output [tokens * top_k, hidden]; each (token, k) slot maps
        # to a distinct permuted row.
        inp = torch.randn(
            (num_tokens * top_k, hidden_size), dtype=input_dtype, device=device
        )
        if pattern_code == _P.kMoEFinalizeARResidualRMSNormFP8Quant:
            # Static per-tensor FP8 output; the BF16 norm is not written.
            kwargs["quant_out"] = torch.empty_like(x, dtype=torch.float8_e4m3fn)
            kwargs["scale_factor"] = torch.ones(1, dtype=torch.float32, device=device)
        else:
            kwargs["norm_out"] = torch.empty_like(x)
        kwargs["expanded_idx_to_permuted_idx"] = (
            torch.randperm(num_tokens * top_k, device=device)
            .to(torch.int32)
            .view(num_tokens, top_k)
        )
        kwargs["expert_scale_factor"] = torch.rand(
            (num_tokens, top_k), dtype=input_dtype, device=device
        )
        if args.shared_expert:
            kwargs["shared_expert_output"] = torch.randn_like(x)
        return inp, kwargs, ("quant_out" if "quant_out" in kwargs else "norm_out")

    if pattern_code == _P.kARResidualRMSNorm or pattern_code in OUT_QUANT_PATTERNS:
        kwargs["norm_out"] = torch.empty_like(x)
    if pattern_code == _P.kARResidualRMSNorm:
        return x, kwargs, "norm_out"

    if pattern_code in STATIC_FP8_PATTERNS:
        kwargs["quant_out"] = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        kwargs["scale_factor"] = torch.ones(1, dtype=torch.float32, device=device)
    elif pattern_code in FP4_PATTERNS:
        layout_code = LAYOUT_NAME_TO_CODE[_layout_name(pattern_code, args)]
        kwargs["quant_out"] = torch.empty(
            (num_tokens, hidden_size // 2), dtype=torch.uint8, device=device
        )
        if layout_code == QuantizationSFLayout.LINEAR:
            sf_shape = (num_tokens, hidden_size // 16)
        else:
            # The 128x4 padded size also bounds the 8x4 layout.
            sf_shape = (
                compute_fp4_swizzled_layout_sf_size(num_tokens, hidden_size // 16),
            )
        kwargs["scale_out"] = torch.empty(
            sf_shape, dtype=torch.float8_e4m3fn, device=device
        )
        kwargs["scale_factor"] = torch.ones(1, dtype=torch.float32, device=device)
        kwargs["layout_code"] = layout_code
    elif pattern_code in GROUP_FP8_PATTERNS:
        group_size = _group_size(args)
        k_num_packed = (hidden_size // group_size + 3) // 4
        tma_aligned_mn = (num_tokens + 3) // 4 * 4
        kwargs["quant_out"] = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        # Packed UE8M0 scales, MN-major: shape (tokens, k_packed), stride (1, mn).
        kwargs["scale_out"] = torch.empty(
            (k_num_packed, tma_aligned_mn), dtype=torch.int32, device=device
        ).T[:num_tokens]
        kwargs["block_quant_group_size"] = group_size
    elif pattern_code in DYNAMIC_FP8_PATTERNS:
        kwargs["quant_out"] = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        kwargs["scale_out"] = torch.empty(
            num_tokens, 1, dtype=torch.float32, device=device
        )
    elif pattern_code in MXFP8_PATTERNS:
        # UE8M0 scale per 32 elements, 1-D, laid out like mxfp8_quantize returns
        # (allreduce.py _check_mxfp8_fusion_args).
        layout_code = LAYOUT_NAME_TO_CODE[_layout_name(pattern_code, args)]
        num_sf_cols = hidden_size // 32
        if layout_code == QuantizationSFLayout.LINEAR:
            sf_numel = num_tokens * num_sf_cols
        else:
            sf_numel = ((num_tokens + 127) // 128 * 128) * ((num_sf_cols + 3) // 4 * 4)
        kwargs["quant_out"] = torch.empty_like(x, dtype=torch.float8_e4m3fn)
        kwargs["scale_out"] = torch.empty(sf_numel, dtype=torch.uint8, device=device)
        kwargs["layout_code"] = layout_code
    else:
        raise ValueError(f"Unsupported pattern code {pattern_code}")
    return x, kwargs, "quant_out"


def _setup_mpi_and_device() -> Tuple[MPI.Comm, int, int, int]:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED)
    local_rank = node_comm.Get_rank()
    torch.cuda.set_device(local_rank)
    return comm, rank, world_size, local_rank


def _init_torch_distributed(rank: int, world_size: int):
    """Initialize torch.distributed for TRTLLM backend (uses NCCL for IPC)."""
    import os

    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _cleanup_torch_distributed():
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()


def _calculate_allreduce_busbw(
    message_size_bytes: int,
    world_size: int,
    time_ms: float,
) -> float:
    """Calculate bus bandwidth for allreduce in TB/sec.

    Uses the standard NCCL bus bandwidth formula:
    busbw = message_size * 2 * (N-1) / N / time
    The 2*(N-1)/N factor accounts for the ring allreduce algorithm
    (reduce-scatter + allgather phases).
    """
    if time_ms <= 0:
        return 0.0
    correction_factor = 2.0 * (world_size - 1) / world_size
    return message_size_bytes * correction_factor / (time_ms * 1e-3) / 1e12


def _calculate_allreduce_algbw(
    message_size_bytes: int,
    time_ms: float,
) -> float:
    """Calculate algorithmic bandwidth (message_size / time) in TB/sec."""
    if time_ms <= 0:
        return 0.0
    return message_size_bytes / (time_ms * 1e-3) / 1e12


def _validate_allreduce(
    workspace: AllReduceFusionWorkspace,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    pattern_code: int,
    use_oneshot: Optional[bool],
    world_size: int,
    rank: int,
    comm: MPI.Comm,
    args,
    verbose: int = 0,
) -> bool:
    """Validate allreduce correctness via comparison with torch sum reference."""
    torch.manual_seed(42)

    # Create identical data across ranks for reference computation
    x_full = torch.randn(
        (world_size, num_tokens, hidden_size), dtype=input_dtype, device="cuda"
    )
    x_local = x_full[rank].clone()

    allreduce_ref = x_full.sum(dim=0)

    if pattern_code == AllReduceFusionPattern.kAllReduce:
        output = torch.empty_like(x_local)
        if isinstance(workspace, PlainAllReduceWorkspace):
            workspace.all_reduce(x_local, output)
        else:
            allreduce_fusion(
                input=x_local,
                workspace=workspace,
                pattern=pattern_code,
                launch_with_pdl=False,
                output=output,
                use_oneshot=use_oneshot,
            )
        torch.cuda.synchronize()

        # quantized_fp8 moves e4m3 (~2^-4 relative step) with a scale per 256 elements.
        atol, rtol = (
            (0.5, 0.125) if workspace.backend == "quantized_fp8" else (0.05, 0.05)
        )
        try:
            torch.testing.assert_close(output, allreduce_ref, atol=atol, rtol=rtol)
            passed = True
        except AssertionError as e:
            passed = False
            if rank == 0:
                print(f"[VALIDATE] AllReduce mismatch: {e}")

    else:
        residual = torch.randn(
            (num_tokens, hidden_size), dtype=input_dtype, device="cuda"
        )
        norm_weight = torch.randn((hidden_size,), dtype=input_dtype, device="cuda")
        eps = RMS_EPS

        # Broadcast residual and norm_weight so all ranks have the same data
        residual_cpu = comm.bcast(residual.cpu().float().numpy(), root=0)
        norm_weight_cpu = comm.bcast(norm_weight.cpu().float().numpy(), root=0)
        residual = torch.from_numpy(residual_cpu).to(device="cuda", dtype=input_dtype)
        norm_weight = torch.from_numpy(norm_weight_cpu).to(
            device="cuda", dtype=input_dtype
        )

        # Same seed on every rank -> identical MoE operands, so the allreduce of the
        # local expert reduction is world_size times the local value.
        torch.manual_seed(43)
        inp, kwargs, _ = _build_pattern_operands(
            pattern_code,
            num_tokens,
            hidden_size,
            input_dtype,
            args,
            "cuda",
            residual=residual,
            norm_weight=norm_weight,
        )
        if pattern_code not in MOE_PATTERNS:
            inp = x_local

        allreduce_fusion(
            input=inp,
            workspace=workspace,
            pattern=pattern_code,
            launch_with_pdl=False,
            use_oneshot=use_oneshot,
            **kwargs,
        )
        torch.cuda.synchronize()

        # Reference: allreduce + residual + rmsnorm
        if pattern_code in MOE_FINALIZE_PATTERNS:
            idx = kwargs["expanded_idx_to_permuted_idx"].long()
            local = (
                inp.float()[idx] * kwargs["expert_scale_factor"].float().unsqueeze(-1)
            ).sum(dim=1)
            if "shared_expert_output" in kwargs:
                local = local + kwargs["shared_expert_output"].float()
            allreduce_ref = (local * world_size).to(input_dtype)
        ref_residual_out = allreduce_ref + residual
        if args.weight_bias == 0.0:
            ref_norm_out = rmsnorm(ref_residual_out, norm_weight, eps, enable_pdl=False)
        else:
            r = ref_residual_out.float()
            ref_norm_out = (
                r
                * torch.rsqrt(r.pow(2).mean(dim=-1, keepdim=True) + eps)
                * (norm_weight.float() + args.weight_bias)
            ).to(input_dtype)

        try:
            if pattern_code == _P.kMoEReductionARResidualRMSNorm:
                # No closed-form reference here (as in the library test): finite only.
                assert torch.isfinite(kwargs["norm_out"]).all(), "non-finite norm_out"
                assert torch.isfinite(kwargs["residual_out"]).all(), (
                    "non-finite residual_out"
                )
            else:
                atol = 0.5 if pattern_code in MOE_PATTERNS else 0.05
                torch.testing.assert_close(
                    kwargs["residual_out"], ref_residual_out, atol=atol, rtol=0.05
                )
                if "norm_out" in kwargs:
                    torch.testing.assert_close(
                        kwargs["norm_out"], ref_norm_out, atol=0.15, rtol=0.05
                    )
            if pattern_code in DYNAMIC_FP8_PATTERNS:
                quant_out, scale_out = kwargs["quant_out"], kwargs["scale_out"]
                fp8_max = float(torch.finfo(torch.float8_e4m3fn).max)
                min_scale = 1.0 / (fp8_max * 512.0)
                ref_scale = torch.clamp(
                    ref_norm_out.float().abs().amax(dim=-1, keepdim=True) / fp8_max,
                    min=min_scale,
                )
                ref_quant = (
                    (ref_norm_out.float() / ref_scale)
                    .clamp(
                        min=float(torch.finfo(torch.float8_e4m3fn).min),
                        max=float(torch.finfo(torch.float8_e4m3fn).max),
                    )
                    .to(torch.float8_e4m3fn)
                )
                torch.testing.assert_close(scale_out, ref_scale, atol=1e-3, rtol=1e-3)
                torch.testing.assert_close(
                    quant_out.float() * scale_out,
                    ref_quant.float() * ref_scale,
                    atol=0.5,
                    rtol=0.5,
                )
            elif "quant_out" in kwargs and rank == 0 and verbose >= 1:
                print(
                    "[VALIDATE] static/group/fp4 quantized values are not checked; "
                    "residual_out and norm_out (when produced) are"
                )
            passed = True
        except AssertionError as e:
            passed = False
            if rank == 0:
                print(f"[VALIDATE] Fused allreduce mismatch: {e}")

    all_passed = comm.allreduce(passed, op=MPI.LAND)
    if rank == 0:
        if all_passed:
            print("[VALIDATE] PASSED: All ranks validated successfully")
        else:
            print("[VALIDATE] FAILED: Validation errors detected")
    return all_passed


def _benchmark_single_config(
    workspace: AllReduceFusionWorkspace,
    num_tokens: int,
    hidden_size: int,
    input_dtype: torch.dtype,
    pattern_code: int,
    use_oneshot: Optional[bool],
    args,
    comm: MPI.Comm,
    rank: int,
    world_size: int,
    requested_backend: str,
) -> Optional[dict]:
    """Benchmark a single (shape, pattern, backend) configuration.

    Returns result dict on rank 0, None on other ranks.
    """
    device = torch.device("cuda")

    # Check workspace capacity
    if not workspace.is_buffer_size_sufficient(
        world_size, num_tokens, hidden_size, input_dtype, use_oneshot
    ):
        if rank == 0 and args.verbose >= 1:
            print(
                f"[SKIP] Workspace insufficient for num_tokens={num_tokens}, "
                f"hidden_size={hidden_size}, use_oneshot={use_oneshot}"
            )
        return None

    # Create input tensors
    x, fusion_kwargs, out_key = _build_pattern_operands(
        pattern_code, num_tokens, hidden_size, input_dtype, args, device
    )

    if isinstance(workspace, PlainAllReduceWorkspace):

        def run_allreduce(inp):
            workspace.all_reduce(inp, fusion_kwargs["output"])
            return fusion_kwargs["output"]

    else:

        def run_allreduce(inp):
            allreduce_fusion(
                input=inp,
                workspace=workspace,
                pattern=pattern_code,
                launch_with_pdl=True,
                use_oneshot=use_oneshot,
                **fusion_kwargs,
            )
            return fusion_kwargs[out_key]

    use_cuda_graph = not args.no_cuda_graph and getattr(
        workspace, "graph_capturable", True
    )

    # Synchronize before benchmarking
    comm.Barrier()
    torch.cuda.synchronize()

    try:
        times = bench_gpu_time(
            fn=run_allreduce,
            input_args=(x,),
            dry_run_iters=args.dry_run_iters,
            repeat_iters=args.num_iters,
            sleep_after_run=False,
            enable_cupti=args.use_cupti,
            use_cuda_graph=use_cuda_graph,
            cold_l2_cache=True,
        )
    except RuntimeError as e:
        # Kernel may fail for very large message sizes or unsupported configs
        if rank == 0:
            elem_size = torch.tensor([], dtype=input_dtype).element_size()
            msg_size_mb = num_tokens * hidden_size * elem_size / (1024 * 1024)
            print(
                f"[ERROR] Kernel failed for shape=({num_tokens}, {hidden_size}) "
                f"msg_size={msg_size_mb:.1f} MiB: {e}"
            )
        return None

    num_measure_iters = len(times)

    # Gather times from all ranks and use max (communication is synchronous)
    all_times = comm.allgather(times)

    if rank == 0:
        per_iter_max = [max(t[i] for t in all_times) for i in range(num_measure_iters)]
        median_time = float(np.median(per_iter_max))
        std_time = float(np.std(per_iter_max))

        elem_size = torch.tensor([], dtype=input_dtype).element_size()
        message_size_bytes = num_tokens * hidden_size * elem_size

        busbw = _calculate_allreduce_busbw(message_size_bytes, world_size, median_time)
        algbw = _calculate_allreduce_algbw(message_size_bytes, median_time)

        backend_name = workspace.backend
        pattern_name = PATTERN_CODE_TO_NAME.get(pattern_code, str(pattern_code))
        oneshot_str = ""
        if use_oneshot is True:
            oneshot_str = "_oneshot"
        elif use_oneshot is False:
            oneshot_str = "_twoshot"

        label = f"{backend_name}_{pattern_name}{oneshot_str}"

        print_perf_metrics(label, median_time, std_time, torch.nan, busbw)

        if args.verbose >= 1:
            print(
                f"  algbw={algbw:.4f} TB/s, "
                f"msg_size={message_size_bytes / 1024:.1f} KiB, "
                f"time={median_time * 1000:.1f} us"
            )

        return {
            "routine": args.routine,
            "median_time": median_time,
            "std_time": std_time,
            "tflops": "N/A",
            "tb_per_sec": busbw,
            "backend": backend_name,
            "resolved_backend": label,
            "ar_backend": requested_backend,
            "pattern": pattern_name,
            "layout_code": _layout_name(pattern_code, args),
            "no_cuda_graph": not use_cuda_graph,
            # Resolved values where the pattern/backend consumes the knob; otherwise
            # the writer backfills the (possibly "") flag value.
            "top_k": (
                _top_k(args)
                if pattern_code in MOE_PATTERNS or workspace.backend == "mnnvl-cutedsl"
                else args.top_k
            ),
            "block_quant_group_size": (
                _group_size(args)
                if pattern_code in GROUP_FP8_PATTERNS
                else args.block_quant_group_size
            ),
            "num_tokens": num_tokens,
            "hidden_size": hidden_size,
            "input_dtype": str(input_dtype),
        }

    return None


def run_allreduce_comm_test(args):
    """Entry point from flashinfer_benchmark.py."""
    if args.routine == "allreduce_fusion":
        return test_allreduce_fusion(args)
    else:
        raise ValueError(f"Unsupported routine: {args.routine}")


def parse_allreduce_comm_args(line, parser):
    """Parse allreduce-specific command line arguments."""
    parser.add_argument(
        "--num_tokens",
        type=int,
        default=64,
        help="Number of tokens (rows) in the input tensor.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=4096,
        help="Hidden dimension size.",
    )
    parser.add_argument(
        "--input_dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16"],
        help="Data type for input tensors.",
    )
    parser.add_argument(
        "--ar_backend",
        type=str,
        nargs="+",
        default=["auto"],
        choices=["auto", "trtllm", "mnnvl", "mnnvl-cutedsl", *PLAIN_BACKENDS],
        help="AllReduce backend(s), each benchmarked with its own workspace. "
        "'vllm' (vllm_all_reduce, timed without CUDA graphs) and 'quantized_fp8' "
        "(quantized_all_reduce, bf16 only) run the allreduce pattern only. "
        "'auto' uses the library heuristic (mnnvl wherever NVLink multicast works). "
        "'mnnvl-cutedsl' supports ar_residual_rmsnorm and moe_finalize in bf16 only, "
        "for the (world_size, hidden_size, top_k) profiles in "
        "flashinfer/comm/mnnvl_cutedsl/presets.py.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="allreduce",
        choices=list(PATTERN_NAME_TO_CODE.keys()),
        help="Fusion pattern. Backends skip patterns they do not implement "
        "(see BACKEND_PATTERNS).",
    )
    parser.add_argument(
        "--layout_code",
        type=str,
        # "" (not None): the CSV writer backfills empty columns from args, and
        # non-fp4 rows must keep an empty layout_code.
        default="",
        choices=list(LAYOUT_NAME_TO_CODE.keys()),
        help="Scale-factor layout for the *_fp4 / *_mxfp8 patterns "
        "(default swizzled_128x4; mnnvl rejects swizzled_8x4, MXFP8 takes "
        "swizzled_128x4 or linear).",
    )
    parser.add_argument(
        "--block_quant_group_size",
        type=int,
        default=None,
        help="Group size for the *_group_fp8 patterns (trtllm only; default 128).",
    )
    parser.add_argument(
        "--weight_bias",
        type=float,
        default=0.0,
        help="Added to the RMSNorm weight: 0.0 standard, 1.0 Gemma / Qwen3.5 RMSNorm.",
    )
    parser.add_argument(
        "--fp32_acc",
        action="store_true",
        default=False,
        help="FP32 accumulation (trtllm only).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Experts per token for moe_reduction / moe_finalize (default 8); also "
        "selects the mnnvl-cutedsl preset profile.",
    )
    parser.add_argument(
        "--shared_expert",
        action="store_true",
        default=False,
        help="moe_finalize: add a shared-expert output (DeepSeek-style).",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run correctness validation before benchmarking.",
    )

    args = parser.parse_args(line)
    # The CSV writer backfills unset columns from args. Unset knobs become "" so rows
    # that do not use them write an empty column; in particular the shared top_k
    # column stays "" for every pattern that existed before --top_k was added.
    if args.top_k is None:
        args.top_k = ""
    if args.block_quant_group_size is None:
        args.block_quant_group_size = ""
    return args


def test_allreduce_fusion(args):
    """Benchmark allreduce fusion across shapes, backends, and patterns."""
    warn_if_pdl_unsupported(args, args.routine)
    comm, rank, world_size, local_rank = _setup_mpi_and_device()
    gpus_per_node = torch.cuda.device_count()

    if world_size < 2:
        if rank == 0:
            print("[ERROR] AllReduce benchmark requires at least 2 ranks")
        return []

    if rank == 0:
        print(
            f"[INFO] AllReduce benchmark: world_size={world_size}, "
            f"gpus_per_node={gpus_per_node}"
        )

    input_dtype = dtype_str_to_torch_dtype(args.input_dtype)
    res = []

    num_tokens = args.num_tokens
    hidden_size = args.hidden_size
    backend_list = list(args.ar_backend)
    pattern_list = [args.pattern]

    # Initialize backends
    torch_dist_initialized = False

    needs_mnnvl = any(b in ("mnnvl", "auto") for b in backend_list)
    # mnnvl-cutedsl and the plain backends rendezvous through torch.distributed.
    needs_torch_dist = any(
        b in ("trtllm", "auto", "mnnvl-cutedsl", *PLAIN_BACKENDS) for b in backend_list
    )

    if needs_mnnvl:
        try:
            MnnvlMemory.initialize()
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] MNNVL initialization failed: {e}")
            backend_list = [b for b in backend_list if b != "mnnvl"]
            if not backend_list:
                if rank == 0:
                    print("[ERROR] No backends available after MNNVL init failure")
                return []

    if needs_torch_dist:
        try:
            _init_torch_distributed(rank, world_size)
            torch_dist_initialized = True
        except Exception as e:
            if rank == 0:
                print(f"[WARNING] torch.distributed initialization failed: {e}")
            backend_list = [
                b
                for b in backend_list
                if b not in ("trtllm", "mnnvl-cutedsl", *PLAIN_BACKENDS)
            ]
            if not backend_list:
                if rank == 0:
                    print(
                        "[ERROR] No backends available after torch.distributed init failure"
                    )
                return []

    try:
        for backend in backend_list:
            # Create workspace sized for the largest shape
            comm_backend = TorchDistBackend() if torch_dist_initialized else None

            try:
                if backend in PLAIN_BACKENDS:
                    workspace = PlainAllReduceWorkspace(
                        backend, rank, num_tokens, hidden_size, input_dtype
                    )
                elif backend == "mnnvl-cutedsl":
                    # Not reachable through create_allreduce_fusion_workspace: the
                    # kernels are compiled for one static (tp, hidden, top_k) profile.
                    from flashinfer.comm.mnnvl_cutedsl_ar import (
                        MNNVLCuteDSLAllReduceFusionWorkspace,
                    )

                    workspace = MNNVLCuteDSLAllReduceFusionWorkspace(
                        tp_size=world_size,
                        tp_rank=rank,
                        max_token_num=num_tokens,
                        hidden_dim=hidden_size,
                        dtype=input_dtype,
                        top_k=_top_k(args),
                        rms_eps=RMS_EPS,
                        weight_bias=args.weight_bias,
                        include_shared_expert=args.shared_expert,
                        **_cutedsl_workspace_kwargs(PATTERN_NAME_TO_CODE[args.pattern]),
                    )
                else:
                    workspace = create_allreduce_fusion_workspace(
                        backend=backend,
                        world_size=world_size,
                        rank=rank,
                        max_token_num=num_tokens,
                        hidden_dim=hidden_size,
                        dtype=input_dtype,
                        gpus_per_node=gpus_per_node,
                        comm_backend=comm_backend,
                        force_oneshot_support=True,
                    )
            except Exception as e:
                if rank == 0:
                    print(
                        f"[ERROR] Failed to create workspace for backend={backend}: {e}"
                    )
                continue

            if rank == 0:
                print(
                    f"[INFO] Created workspace: backend={workspace.backend}, "
                    f"shape=({num_tokens}, {hidden_size})"
                )

            try:
                for pattern_name in pattern_list:
                    pattern_code = PATTERN_NAME_TO_CODE[pattern_name]

                    if pattern_code not in BACKEND_PATTERNS[workspace.backend]:
                        if rank == 0:
                            print(
                                f"[SKIP] {workspace.backend} does not support "
                                f"pattern {pattern_name}"
                            )
                        continue

                    # MoE patterns, the CuTe DSL backend and the plain backends pick
                    # their protocol internally (no use_oneshot knob).
                    oneshot_list: List[Optional[bool]] = (
                        [None]
                        if pattern_code in MOE_PATTERNS
                        or workspace.backend in ("mnnvl-cutedsl", *PLAIN_BACKENDS)
                        else [True, False]
                    )
                    for use_oneshot in oneshot_list:
                        if use_oneshot is False and num_tokens <= world_size:
                            if rank == 0 and args.verbose >= 1:
                                print(
                                    f"[SKIP] twoshot requires num_tokens > world_size; "
                                    f"num_tokens={num_tokens}, world_size={world_size}"
                                )
                            continue

                        if rank == 0 and args.verbose >= 1:
                            print(
                                f"[INFO] Benchmarking: backend={workspace.backend}, "
                                f"pattern={pattern_name}, "
                                f"shape=({num_tokens}, {hidden_size}), "
                                f"use_oneshot={use_oneshot}"
                            )

                        # Validate if requested
                        if args.validate:
                            valid = _validate_allreduce(
                                workspace=workspace,
                                num_tokens=num_tokens,
                                hidden_size=hidden_size,
                                input_dtype=input_dtype,
                                pattern_code=pattern_code,
                                use_oneshot=use_oneshot,
                                world_size=world_size,
                                rank=rank,
                                comm=comm,
                                args=args,
                                verbose=args.verbose,
                            )
                            if not valid:
                                if rank == 0:
                                    print(
                                        "[ERROR] Validation failed, skipping benchmark"
                                    )
                                continue

                        result = _benchmark_single_config(
                            workspace=workspace,
                            num_tokens=num_tokens,
                            hidden_size=hidden_size,
                            input_dtype=input_dtype,
                            pattern_code=pattern_code,
                            use_oneshot=use_oneshot,
                            args=args,
                            comm=comm,
                            rank=rank,
                            world_size=world_size,
                            requested_backend=backend,
                        )

                        if result is not None:
                            res.append(result)

            finally:
                workspace.destroy()

    finally:
        if torch_dist_initialized:
            _cleanup_torch_distributed()

    return res
