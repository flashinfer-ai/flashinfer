"""Low-token B12x Direct FP4-weight/FP4-activation fused MoE."""

from __future__ import annotations

import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_direct_micro_kernel import (
    MoEDirectMicroKernel,
    build_direct_micro_kernel,
    compile_direct_micro_kernel,
)
from flashinfer.jit.b12x_direct_nvfp4_fused_moe import (
    gen_b12x_direct_nvfp4_fused_moe_module,
)
from flashinfer.quantization import fp4_quantize
from flashinfer.utils import _get_cache_buf, register_custom_op


_TUNED_LAUNCHES = {
    (2048, 512): {
        1: (1, 512),
        2: (1, 512),
        3: (1, 512),
        4: (1, 512),
        5: (2, 256),
        6: (2, 256),
        7: (2, 256),
        8: (4, 128),
    },
    (2048, 768): {
        1: (1, 512),
        2: (1, 512),
        3: (1, 512),
        4: (1, 512),
        5: (2, 256),
        6: (2, 256),
        7: (2, 256),
        8: (1, 512),
    },
}

_HYBRID_GATE_LAUNCHES = {
    (2048, 512): {m: (1, 512) for m in range(2, 9)},
    (2048, 768): {
        2: (1, 512),
        3: (1, 512),
        4: (2, 256),
        5: (2, 256),
        6: (2, 256),
        7: (2, 256),
        8: (2, 256),
    },
}


@dataclass(frozen=True)
class B12xDirectNVFP4Workspace:
    """Stable-address scratch tensors used by the three-stage NVFP4 path."""

    intermediate_quantized: torch.Tensor
    intermediate_scales: torch.Tensor
    barrier_count: torch.Tensor
    barrier_epoch: torch.Tensor


def b12x_direct_nvfp4_fused_moe_workspace(
    num_tokens: int,
    topk: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    device: torch.device | str,
) -> B12xDirectNVFP4Workspace:
    """Allocate all scratch buffers required by Direct NVFP4."""
    if not 1 <= num_tokens <= 8:
        raise ValueError(f"num_tokens must be in [1, 8], got {num_tokens}")
    if not 1 <= topk <= 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")
    if (
        hidden_size < 16
        or intermediate_size < 16
        or hidden_size % 16
        or intermediate_size % 16
    ):
        raise ValueError(
            "hidden_size and intermediate_size must be positive multiples of 16"
        )
    routed_rows = num_tokens * topk
    return B12xDirectNVFP4Workspace(
        intermediate_quantized=torch.empty(
            (routed_rows, intermediate_size // 2), dtype=torch.uint8, device=device
        ),
        intermediate_scales=torch.empty(
            (routed_rows, intermediate_size // 16),
            dtype=torch.bfloat16,
            device=device,
        ),
        barrier_count=torch.zeros(1, dtype=torch.int32, device=device),
        barrier_epoch=torch.zeros(1, dtype=torch.int32, device=device),
    )


def _recommended_launch(
    num_tokens: int, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    return _TUNED_LAUNCHES.get((hidden_size, intermediate_size), {}).get(
        num_tokens, (2, 256)
    )


@functools.cache
def _require_cuda_129() -> None:
    """Fail before JIT compilation when SM12x normalization is unavailable."""
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "b12x_direct_nvfp4_fused_moe requires CUDA 12.9 or newer on SM120"
        )


def _resolve_hybrid_launch(
    hidden_states: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    intermediate_size: int,
    outputs_per_warp: Optional[int],
    num_threads: Optional[int],
) -> tuple[int, int] | None:
    """Resolve the launch pair used by the hybrid gate before validation."""
    if hidden_states.ndim != 2:
        return None
    num_tokens, hidden_size = hidden_states.shape
    if (
        num_tokens < 2
        or (expert_map is not None and expert_map.numel() != 0)
        or num_tokens
        not in _HYBRID_GATE_LAUNCHES.get((int(hidden_size), int(intermediate_size)), {})
    ):
        return None
    gate_outputs, gate_threads = _HYBRID_GATE_LAUNCHES[
        (int(hidden_size), int(intermediate_size))
    ][int(num_tokens)]
    if outputs_per_warp is not None:
        gate_outputs = int(outputs_per_warp)
    if num_threads is not None:
        gate_threads = int(num_threads)
    return int(gate_outputs), int(gate_threads)


@functools.cache
def _get_global_scale(device_index: int, value: float) -> torch.Tensor:
    """Materialize one stable scalar per device/value outside graph capture."""
    return torch.tensor(
        [value], dtype=torch.float32, device=torch.device("cuda", device_index)
    )


@functools.cache
def _get_expert_ones(device_index: int, num_experts: int) -> torch.Tensor:
    return torch.ones(
        num_experts,
        dtype=torch.float32,
        device=torch.device("cuda", device_index),
    )


@functools.cache
def _get_hybrid_fc2(
    device_index: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    topk: int,
    num_experts: int,
):
    device = torch.device("cuda", device_index)
    kernel = build_direct_micro_kernel(
        num_experts,
        num_tokens,
        hidden_size,
        intermediate_size,
        topk,
        activation="silu",
        fast_math=True,
        compile_time_phase=2,
        enable_pdl=True,
        use_f16_dot=True,
        rowmajor_fp4_intermediate=True,
        scale_format="bf16_k16",
        device=device,
    )
    compiled = compile_direct_micro_kernel(
        kernel,
        topk_ids_dtype=torch.int32,
        options="--opt-level 3",
    )
    return kernel, compiled


@functools.cache
def _get_module():
    module = gen_b12x_direct_nvfp4_fused_moe_module().build_and_load()

    @register_custom_op(
        "flashinfer::b12x_direct_nvfp4_fused_moe",
        mutates_args=[
            "hidden_quantized",
            "hidden_scales",
            "intermediate_quantized",
            "intermediate_scales",
            "output",
        ],
    )
    def run(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_scales: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_scales: torch.Tensor,
        expert_map: torch.Tensor,
        hidden_quantized: torch.Tensor,
        hidden_scales: torch.Tensor,
        intermediate_quantized: torch.Tensor,
        intermediate_scales: torch.Tensor,
        output: torch.Tensor,
        outputs_per_warp: int,
        num_threads: int,
        hidden_global_encode_scale: float,
        intermediate_global_encode_scale: float,
    ) -> None:
        module.b12x_direct_nvfp4_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            hidden_quantized,
            hidden_scales,
            intermediate_quantized,
            intermediate_scales,
            output,
            outputs_per_warp,
            num_threads,
            hidden_global_encode_scale,
            intermediate_global_encode_scale,
            1,
        )

    @register_custom_op(
        "flashinfer::b12x_direct_nvfp4_gate",
        mutates_args=["intermediate_quantized", "intermediate_scales"],
    )
    def run_gate(
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_scales: torch.Tensor,
        gemm2_weights: torch.Tensor,
        gemm2_scales: torch.Tensor,
        expert_map: torch.Tensor,
        hidden_quantized: torch.Tensor,
        hidden_scales: torch.Tensor,
        intermediate_quantized: torch.Tensor,
        intermediate_scales: torch.Tensor,
        output: torch.Tensor,
        outputs_per_warp: int,
        num_threads: int,
        hidden_global_encode_scale: float,
        intermediate_global_encode_scale: float,
    ) -> None:
        module.b12x_direct_nvfp4_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            hidden_quantized,
            hidden_scales,
            intermediate_quantized,
            intermediate_scales,
            output,
            outputs_per_warp,
            num_threads,
            hidden_global_encode_scale,
            intermediate_global_encode_scale,
            0,
        )

    return SimpleNamespace(run=run, run_gate=run_gate)


def _validate(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    expert_map: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    workspace: Optional[B12xDirectNVFP4Workspace],
    outputs_per_warp: Optional[int],
    num_threads: Optional[int],
    launch_override: tuple[int, int] | None = None,
) -> None:
    if not hidden_states.is_cuda or torch.cuda.get_device_capability(
        hidden_states.device
    ) != (
        12,
        0,
    ):
        raise ValueError("b12x_direct_nvfp4_fused_moe requires an SM120 CUDA tensor")
    if hidden_states.dtype != torch.bfloat16 or hidden_states.ndim != 2:
        raise ValueError("hidden_states must be a 2D BF16 tensor")
    num_tokens, hidden_size = hidden_states.shape
    if not 1 <= num_tokens <= 8 or hidden_size % 16:
        raise ValueError("num_tokens must be in [1, 8] and hidden_size divisible by 16")
    if topk_ids.dtype != torch.int32 or topk_weights.dtype != torch.float32:
        raise ValueError("routes must use int32 ids and float32 weights")
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != num_tokens:
        raise ValueError("route tensors must have matching [num_tokens, topk] shapes")
    topk = int(topk_ids.shape[1])
    if not 1 <= topk <= 8:
        raise ValueError("topk must be in [1, 8]")
    if gemm1_weights.dtype != torch.uint8 or gemm2_weights.dtype != torch.uint8:
        raise ValueError("packed weights must use uint8 E2M1 pairs")
    if gemm1_scales.dtype != torch.bfloat16 or gemm2_scales.dtype != torch.bfloat16:
        raise ValueError("Direct dequant scales must be BF16")
    experts = int(gemm1_weights.shape[0])
    intermediate_size = (
        int(gemm2_weights.shape[2]) * 2 if gemm2_weights.ndim >= 3 else 0
    )
    if intermediate_size < 16 or intermediate_size % 16:
        raise ValueError(
            "gemm2_weights must provide an intermediate_size that is a positive "
            "multiple of 16"
        )
    expected_shapes = (
        (experts, 2 * intermediate_size, hidden_size // 2),
        (experts, 2 * intermediate_size, hidden_size // 16),
        (experts, hidden_size, intermediate_size // 2),
        (experts, hidden_size, intermediate_size // 16),
    )
    actual_shapes = tuple(
        tuple(tensor.shape)
        for tensor in (gemm1_weights, gemm1_scales, gemm2_weights, gemm2_scales)
    )
    if actual_shapes != expected_shapes:
        raise ValueError(
            f"weight/scale shapes {actual_shapes} do not match {expected_shapes}"
        )
    tensors = [
        hidden_states,
        topk_ids,
        topk_weights,
        gemm1_weights,
        gemm1_scales,
        gemm2_weights,
        gemm2_scales,
    ]
    if expert_map is not None:
        tensors.append(expert_map)
    if output is not None:
        tensors.append(output)
    if any(tensor.device != hidden_states.device for tensor in tensors):
        raise ValueError("all tensors must be on the hidden_states device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all input/output tensors must be contiguous")
    if output is not None and (
        output.dtype != torch.bfloat16 or output.shape != hidden_states.shape
    ):
        raise ValueError("output must be BF16 with the hidden_states shape")
    if workspace is not None:
        routed_rows = num_tokens * topk
        expected_workspace = {
            "intermediate_quantized": (
                (routed_rows, intermediate_size // 2),
                torch.uint8,
            ),
            "intermediate_scales": (
                (routed_rows, intermediate_size // 16),
                torch.bfloat16,
            ),
        }
        for name, (shape, dtype) in expected_workspace.items():
            actual = getattr(workspace, name)
            if actual.shape != shape or actual.dtype != dtype:
                raise ValueError(f"workspace.{name} has an incompatible shape or dtype")
            if actual.device != hidden_states.device or not actual.is_contiguous():
                raise ValueError(
                    f"workspace.{name} must be contiguous and on the input device"
                )
    if launch_override is None:
        default_outputs, default_threads = _recommended_launch(
            num_tokens, hidden_size, intermediate_size
        )
    else:
        default_outputs, default_threads = launch_override
    launch_outputs = default_outputs if outputs_per_warp is None else outputs_per_warp
    launch_threads = default_threads if num_threads is None else num_threads
    if launch_outputs not in (1, 2, 4, 8):
        raise ValueError("outputs_per_warp must be one of 1, 2, 4, or 8")
    if launch_threads < 64 or launch_threads > 1024 or launch_threads % 32:
        raise ValueError("num_threads must be a warp multiple in [64, 1024]")
    outputs_per_block = launch_outputs * (launch_threads // 32)
    if (
        outputs_per_block < 16
        or outputs_per_block % 16
        or intermediate_size % outputs_per_block
    ):
        raise ValueError(
            "NVFP4 gate fusion requires outputs_per_warp * num_warps to be "
            "a multiple of 16 that divides intermediate_size"
        )


def b12x_direct_nvfp4_fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[B12xDirectNVFP4Workspace] = None,
    outputs_per_warp: Optional[int] = None,
    num_threads: Optional[int] = None,
    hidden_global_encode_scale: float = 448.0,
    intermediate_global_encode_scale: float = 448.0,
    *,
    skip_check: bool = False,
) -> torch.Tensor:
    r"""Run a three-stage Direct NVFP4/W4A4 fused MoE for SM120 decode.

    Hidden and SwiGLU activations are quantized per K/16 block to E2M1 with
    E4M3-rounded scales. Weights use the same packed representation and
    model-load-time folded BF16 scales as the Direct W4A16 path. The global
    encode scales shift E4M3 block scales away from underflow and are divided
    back out before GEMM; callers may replace the conservative defaults with
    calibrated model scales.
    """
    if hidden_states.is_cuda:
        _require_cuda_129()
    intermediate_size = (
        int(gemm2_weights.shape[2]) * 2 if gemm2_weights.ndim >= 3 else 0
    )
    if intermediate_size < 16 or intermediate_size % 16:
        raise ValueError(
            "gemm2_weights must provide an intermediate_size that is a positive "
            "multiple of 16"
        )
    hybrid_launch = _resolve_hybrid_launch(
        hidden_states,
        expert_map,
        intermediate_size,
        outputs_per_warp,
        num_threads,
    )
    if not skip_check:
        _validate(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            output,
            workspace,
            outputs_per_warp,
            num_threads,
            launch_override=hybrid_launch,
        )
    num_tokens, hidden_size = hidden_states.shape
    topk = int(topk_ids.shape[1])
    if output is None:
        output = torch.empty_like(hidden_states)
    if hidden_global_encode_scale <= 0 or intermediate_global_encode_scale <= 0:
        raise ValueError("activation global encode scales must be positive")
    if workspace is None:
        workspace = b12x_direct_nvfp4_fused_moe_workspace(
            num_tokens,
            topk,
            hidden_size,
            intermediate_size,
            device=hidden_states.device,
        )
    if expert_map is None:
        raw = _get_cache_buf(
            "b12x_direct_nvfp4_empty_expert_map", 4, hidden_states.device
        )
        expert_map = raw[:0].view(torch.int32)
    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    global_scale = _get_global_scale(
        device_index,
        float(hidden_global_encode_scale),
    )
    hidden_quantized, hidden_scales = fp4_quantize(
        hidden_states,
        global_scale,
        sf_vec_size=16,
        is_sf_swizzled_layout=False,
    )
    use_hybrid = hybrid_launch is not None and expert_map.numel() == 0
    if use_hybrid:
        gate_outputs, gate_threads = hybrid_launch
        _get_module().run_gate(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            hidden_quantized,
            hidden_scales,
            workspace.intermediate_quantized,
            workspace.intermediate_scales,
            output,
            gate_outputs,
            gate_threads,
            hidden_global_encode_scale,
            intermediate_global_encode_scale,
        )
        num_experts = int(gemm1_weights.shape[0])
        kernel, compiled = _get_hybrid_fc2(
            device_index,
            num_tokens,
            hidden_size,
            intermediate_size,
            topk,
            num_experts,
        )
        expert_ones = _get_expert_ones(device_index, num_experts)
        MoEDirectMicroKernel.launch(
            compiled,
            x=hidden_states,
            w1_fp4=gemm1_weights,
            w1_blockscale=gemm1_scales,
            w1_alphas=expert_ones,
            a1_gscale=expert_ones,
            a2_gscale=workspace.intermediate_scales,
            inter_fp32=workspace.intermediate_quantized,
            w2_fp4=gemm2_weights,
            w2_blockscale=gemm2_scales,
            w2_alphas=expert_ones,
            topk_ids=topk_ids.view(-1),
            topk_weights=topk_weights.view(-1),
            out=output,
            barrier_count=workspace.barrier_count,
            barrier_epoch=workspace.barrier_epoch,
            m=num_tokens,
            grid_x=kernel.grid_x,
        )
    else:
        default_outputs, default_threads = _recommended_launch(
            num_tokens, hidden_size, intermediate_size
        )
        _get_module().run(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            hidden_quantized,
            hidden_scales,
            workspace.intermediate_quantized,
            workspace.intermediate_scales,
            output,
            default_outputs if outputs_per_warp is None else outputs_per_warp,
            default_threads if num_threads is None else num_threads,
            hidden_global_encode_scale,
            intermediate_global_encode_scale,
        )
    return output
