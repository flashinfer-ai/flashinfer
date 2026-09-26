"""Low-token B12x Direct FP4-weight/BF16-activation fused MoE."""

from __future__ import annotations

import functools
import weakref
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Optional

import torch

from flashinfer.cute_dsl.utils import current_cuda_stream
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
    _allocate_sm120_w4a16_workspace,
    _get_w4a16_packed_weights,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
    select_route_block_size_m,
)
from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_kernel import (
    compile_w4a16_fused_moe,
    run_w4a16_moe,
    run_w4a16_route_pack,
)
from flashinfer.jit.b12x_direct_w4a16_fused_moe import (
    gen_b12x_direct_w4a16_fused_moe_module,
)
from flashinfer.utils import _get_cache_buf, register_custom_op


_TUNED_LAUNCHES = {
    (2048, 512): {
        1: (1, 768),
        2: (1, 768),
        3: (1, 576),
        4: (1, 768),
        5: (2, 448),
        6: (2, 384),
        7: (2, 640),
        8: (2, 768),
    },
    (2048, 768): {
        1: (1, 640),
        2: (1, 768),
        3: (1, 576),
        4: (1, 704),
        5: (2, 896),
        6: (2, 384),
        7: (2, 640),
        8: (2, 704),
    },
}


@dataclass(frozen=True)
class _DirectScaleSource:
    block_scales: torch.Tensor
    global_scales: torch.Tensor
    rows: int
    cols: int
    source_format: str


# ``prepare_b12x_direct_w4a16_scales`` is the model-load boundary where the
# original B12x scale tensor is still available.  Remember that provenance so
# the public Direct call can select the tensor-core path without changing its
# tensor-only signature.  Weak finalizers remove entries when the returned
# Direct scale tensor dies, preventing stale data-pointer reuse.
_DIRECT_SCALE_SOURCES: dict[tuple[int, int], _DirectScaleSource] = {}


def _direct_scale_key(tensor: torch.Tensor) -> tuple[int, int]:
    device_index = tensor.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return int(device_index), int(tensor.data_ptr())


def _forget_direct_scale_source(key: tuple[int, int]) -> None:
    _DIRECT_SCALE_SOURCES.pop(key, None)


def _remember_direct_scale_source(
    direct_scales: torch.Tensor,
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    source_format: str,
) -> None:
    key = _direct_scale_key(direct_scales)
    _DIRECT_SCALE_SOURCES[key] = _DirectScaleSource(
        block_scales=block_scales,
        global_scales=global_scales,
        rows=int(rows),
        cols=int(cols),
        source_format=str(source_format),
    )
    weakref.finalize(direct_scales, _forget_direct_scale_source, key)


# (use direct top-k routes, FC1 K/N, FC2 K/N).  M1--M4 keep the launch chain
# to one cooperative kernel.  The larger cases use compact expert routes and
# the non-cooperative FC1->FC2 readiness pipeline.
_TC_LAUNCHES = {
    (2048, 512): {
        1: (True, (128, 64, 64, 128)),
        2: (True, (128, 64, 64, 128)),
        3: (False, (64, 128, 64, 128)),
        4: (True, (64, 256, 32, 512)),
        5: (False, (64, 256, 64, 256)),
        6: (False, (64, 256, 64, 256)),
        7: (False, (64, 256, 64, 256)),
        8: (False, (64, 256, 64, 256)),
    },
    (2048, 768): {
        1: (True, (128, 64, 128, 64)),
        2: (True, (128, 64, 128, 64)),
        3: (False, (64, 128, 64, 128)),
        4: (False, (64, 256, 64, 256)),
        5: (False, (64, 256, 64, 256)),
        6: (False, (64, 256, 64, 256)),
        7: (False, (64, 256, 64, 256)),
        8: (False, (64, 256, 64, 256)),
    },
}


def prepare_b12x_direct_w4a16_scales(
    block_scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    source_format: str = "modelopt",
) -> torch.Tensor:
    r"""Convert B12x block scales to Direct row-major BF16 dequant scales.

    This is a model-load-time operation. It removes the B12x MMA swizzle and
    folds each expert's inverse global scale into its E4M3 K/16 block scales,
    leaving the hot kernel with one contiguous BF16 scale load per 16 weights.
    """
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_w4a16_host import (
        normalize_expert_block_scales,
        unswizzle_expert_scales,
    )

    source_global_scales = global_scales
    source_format = source_format.lower()
    if cols % 16:
        raise ValueError(f"cols must be divisible by 16, got {cols}")
    num_experts = int(global_scales.numel())
    if num_experts < 1:
        raise ValueError("global_scales must contain at least one expert")
    normalized = normalize_expert_block_scales(
        block_scales,
        num_experts=num_experts,
        rows=rows,
        cols=cols,
    )
    scales = unswizzle_expert_scales(normalized, rows=rows, cols=cols).float()
    global_scales = global_scales.to(device=scales.device, dtype=torch.float32)
    if source_format == "compressed_tensors":
        global_scales = global_scales.reciprocal()
    elif source_format != "modelopt":
        raise ValueError("source_format must be 'modelopt' or 'compressed_tensors'")
    direct_scales = (
        (scales * global_scales.reshape(-1, 1, 1)).to(torch.bfloat16).contiguous()
    )
    _remember_direct_scale_source(
        direct_scales,
        block_scales,
        source_global_scales,
        rows=rows,
        cols=cols,
        source_format=source_format,
    )
    return direct_scales


def b12x_direct_w4a16_fused_moe_workspace(
    num_tokens: int,
    topk: int,
    intermediate_size: int,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Allocate the stable BF16 intermediate required for CUDA Graph capture."""
    if not 1 <= num_tokens <= 8:
        raise ValueError(f"num_tokens must be in [1, 8], got {num_tokens}")
    if not 1 <= topk <= 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")
    if intermediate_size < 16 or intermediate_size > 1024 or intermediate_size % 16:
        raise ValueError(
            "intermediate_size must be a multiple of 16 in [16, 1024], "
            f"got {intermediate_size}"
        )
    return torch.empty(
        (num_tokens * topk, intermediate_size),
        dtype=torch.bfloat16,
        device=device,
    )


def _recommended_launch(
    num_tokens: int, hidden_size: int, intermediate_size: int
) -> tuple[int, int]:
    return _TUNED_LAUNCHES.get((hidden_size, intermediate_size), {}).get(
        num_tokens, (1, 256)
    )


@functools.cache
def _require_cuda_129() -> None:
    """Fail before JIT compilation when SM12x normalization is unavailable."""
    from flashinfer.jit.cpp_ext import is_cuda_version_at_least

    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "b12x_direct_w4a16_fused_moe requires CUDA 12.9 or newer on SM12x"
        )


def _get_tc_workspace(
    workspace: torch.Tensor,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
):
    """Return TC scratch owned by the caller's intermediate workspace.

    The public Direct API is functional and must not retain model-sized GPU
    allocations in a process-global cache.  Views are common in CUDA Graph
    callers, so attach the per-shape CuTe workspace to the ultimate base
    tensor; all views of one caller-owned workspace then share its lifetime.
    """
    owner = workspace
    while isinstance(getattr(owner, "_base", None), torch.Tensor):
        owner = owner._base
    cache = getattr(owner, "_flashinfer_b12x_tc_workspace_cache", None)
    if cache is None:
        cache = {}
        owner._flashinfer_b12x_tc_workspace_cache = cache  # type: ignore[attr-defined]
    device = workspace.device
    key = (
        int(num_tokens),
        int(hidden_size),
        int(intermediate_size),
        int(num_experts),
        int(topk),
    )
    prepared = cache.get(key)
    if prepared is None:
        prepared = _allocate_sm120_w4a16_workspace(
            state_E=int(num_experts),
            weight_E=int(num_experts),
            routed_rows=int(num_tokens) * int(topk),
            k=int(hidden_size),
            n=int(intermediate_size),
            num_topk=int(topk),
            device=device,
            activation="silu",
        )
        # The public route-pack helper requires capacity for the expert table
        # even when a tiny decode batch activates only a few blocks.  The
        # generic workspace allocator intentionally sizes this table to the
        # active route count, so grow this Direct TC-owned view to the fixed
        # 64-expert contract before passing it to the helper.
        if prepared.block_expert_ids.numel() < int(num_experts):
            prepared.block_expert_ids = torch.empty(
                (int(num_experts),), dtype=torch.int32, device=device
            )
        cache[key] = prepared
    # ``workspace`` is the public intermediate buffer ([M * topk, N]).  Reuse
    # it for the TC pipeline's activated FC1 output instead of silently
    # allocating a second tensor with the same logical contract.
    prepared.intermediate_cache2 = workspace
    return prepared


@functools.cache
def _get_tc_launch(
    device_index: int,
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    topk: int,
    weight_layout: str,
    scale_format: str,
    w13_layout: str,
    direct_topk_routes: bool,
    max_m_blocks: int,
    tile_config: tuple[int, int, int, int],
):
    with torch.cuda.device(int(device_index)):
        props = torch.cuda.get_device_properties(int(device_index))
        sms = int(props.multi_processor_count)
        max_shared_mem = int(props.shared_memory_per_block_optin)
        return compile_w4a16_fused_moe(
            size_m=int(num_tokens),
            hidden_size=int(hidden_size),
            intermediate_size=int(intermediate_size),
            num_experts=int(num_experts),
            top_k=int(topk),
            activation="silu",
            apply_router_weight_on_input=False,
            zero_fc2_output=False,
            moe_block_size=select_route_block_size_m(
                int(num_tokens), int(topk), int(num_experts)
            ),
            max_m_blocks=int(max_m_blocks),
            element_dtype="bf16",
            fast_math=True,
            sms=sms,
            max_shared_mem=max_shared_mem,
            weight_layout=weight_layout,
            scale_format=scale_format,
            w13_layout=w13_layout,
            direct_topk_routes=bool(direct_topk_routes),
            tc_decode_fused_sum=True,
            tc_pair_activation=True,
            tc_zero_output=bool(direct_topk_routes),
            force_tile_config=tile_config,
        )


def _try_run_tc_w4a16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    workspace: torch.Tensor,
    output: torch.Tensor,
) -> bool:
    """Run the caller-owned CuTe tensor-core decode path when provenance is known."""
    num_tokens, hidden_size = hidden_states.shape
    intermediate_size = int(gemm2_weights.shape[2]) * 2
    launch_spec = _TC_LAUNCHES.get((hidden_size, intermediate_size), {}).get(num_tokens)
    if launch_spec is None:
        return False
    topk = int(topk_ids.shape[1])
    num_experts = int(gemm1_weights.shape[0])
    if topk != 8 or num_experts != 64:
        return False
    source1 = _DIRECT_SCALE_SOURCES.get(_direct_scale_key(gemm1_scales))
    source2 = _DIRECT_SCALE_SOURCES.get(_direct_scale_key(gemm2_scales))
    if source1 is None or source2 is None:
        return False
    if (
        source1.source_format != source2.source_format
        or (source1.rows, source1.cols) != (2 * intermediate_size, hidden_size)
        or (source2.rows, source2.cols) != (hidden_size, intermediate_size)
    ):
        return False

    prepared = _get_w4a16_packed_weights(
        w1_weight=gemm1_weights,
        w1_weight_sf=source1.block_scales,
        w1_alpha=source1.global_scales,
        w2_weight=gemm2_weights,
        w2_weight_sf=source2.block_scales,
        w2_alpha=source2.global_scales,
        activation="silu",
        params_dtype=torch.bfloat16,
        source_format=source1.source_format,
    )
    tc_workspace = _get_tc_workspace(
        workspace,
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
    )
    direct_topk_routes, tile_config = launch_spec
    max_m_blocks = (
        num_tokens * topk
        if direct_topk_routes
        else int(tc_workspace.block_expert_ids.numel())
    )
    device_index = hidden_states.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    fused = _get_tc_launch(
        int(device_index),
        num_tokens,
        hidden_size,
        intermediate_size,
        num_experts,
        topk,
        prepared.weight_layout,
        prepared.scale_format,
        prepared.w13_layout,
        bool(direct_topk_routes),
        max_m_blocks,
        tile_config,
    )
    if not direct_topk_routes:
        run_w4a16_route_pack(
            topk_ids,
            tc_workspace.packed_route_indices,
            tc_workspace.block_expert_ids,
            tc_workspace.packed_route_count,
            output,
            prepared.workspace,
            num_experts=num_experts,
            block_size=int(fused.moe_block_size),
            clear_output=True,
            clear_ctas=8,
            clear_lock_words=int(prepared.workspace.numel()),
        )
    run_w4a16_moe(
        hidden_states,
        prepared,
        topk_weights,
        topk_ids,
        activation="silu",
        intermediate_cache13=tc_workspace.intermediate_cache13,
        intermediate_cache2=tc_workspace.intermediate_cache2,
        output=output,
        fc1_c_tmp=tc_workspace.fc1_c_tmp,
        fc2_c_tmp=tc_workspace.fc2_c_tmp,
        packed_route_indices=tc_workspace.packed_route_indices,
        block_expert_ids=tc_workspace.block_expert_ids,
        packed_route_count=tc_workspace.packed_route_count,
        expert_offsets=tc_workspace.expert_offsets,
        fused_launch=fused,
        routes_prepacked=not direct_topk_routes,
    )
    return True


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape={tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


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
    workspace: Optional[torch.Tensor],
    outputs_per_warp: Optional[int],
    num_threads: Optional[int],
) -> None:
    if (
        not hidden_states.is_cuda
        or torch.cuda.get_device_capability(hidden_states.device)[0] != 12
    ):
        raise ValueError("b12x_direct_w4a16_fused_moe requires an SM12x CUDA tensor")
    device = hidden_states.device
    _check_tensor(
        "hidden_states", hidden_states, ndim=2, dtype=torch.bfloat16, device=device
    )
    num_tokens, hidden_size = hidden_states.shape
    if not 1 <= num_tokens <= 8:
        raise ValueError(f"num_tokens must be in [1, 8], got {num_tokens}")
    if hidden_size < 16 or hidden_size > 8192 or hidden_size % 16:
        raise ValueError("hidden_size must be a multiple of 16 in [16, 8192]")
    _check_tensor("topk_ids", topk_ids, ndim=2, dtype=torch.int32, device=device)
    _check_tensor(
        "topk_weights", topk_weights, ndim=2, dtype=torch.float32, device=device
    )
    _check_tensor(
        "gemm1_weights", gemm1_weights, ndim=3, dtype=torch.uint8, device=device
    )
    _check_tensor(
        "gemm1_scales", gemm1_scales, ndim=3, dtype=torch.bfloat16, device=device
    )
    _check_tensor(
        "gemm2_weights", gemm2_weights, ndim=3, dtype=torch.uint8, device=device
    )
    _check_tensor(
        "gemm2_scales", gemm2_scales, ndim=3, dtype=torch.bfloat16, device=device
    )
    if topk_ids.shape != topk_weights.shape or topk_ids.shape[0] != num_tokens:
        raise ValueError(
            "topk ids/weights must have matching [num_tokens, topk] shapes"
        )
    topk = int(topk_ids.shape[1])
    if not 1 <= topk <= 8:
        raise ValueError(f"topk must be in [1, 8], got {topk}")
    num_experts = int(gemm1_weights.shape[0])
    intermediate_size = int(gemm2_weights.shape[2]) * 2
    if intermediate_size < 16 or intermediate_size > 1024 or intermediate_size % 16:
        raise ValueError("intermediate_size must be a multiple of 16 in [16, 1024]")
    expected = (num_experts, 2 * intermediate_size, hidden_size // 2)
    if tuple(gemm1_weights.shape) != expected:
        raise ValueError(f"gemm1_weights must have shape {expected}")
    if tuple(gemm1_scales.shape) != (
        num_experts,
        2 * intermediate_size,
        hidden_size // 16,
    ):
        raise ValueError("gemm1_scales has an incompatible shape")
    if tuple(gemm2_weights.shape) != (
        num_experts,
        hidden_size,
        intermediate_size // 2,
    ):
        raise ValueError("gemm2_weights has an incompatible shape")
    if tuple(gemm2_scales.shape) != (
        num_experts,
        hidden_size,
        intermediate_size // 16,
    ):
        raise ValueError("gemm2_scales has an incompatible shape")
    if expert_map is not None:
        _check_tensor(
            "expert_map", expert_map, ndim=1, dtype=torch.int32, device=device
        )
    if output is not None:
        _check_tensor("output", output, ndim=2, dtype=torch.bfloat16, device=device)
        if output.shape != hidden_states.shape:
            raise ValueError("output must have the same shape as hidden_states")
    if workspace is not None:
        _check_tensor(
            "workspace", workspace, ndim=2, dtype=torch.bfloat16, device=device
        )
        if tuple(workspace.shape) != (num_tokens * topk, intermediate_size):
            raise ValueError("workspace has an incompatible shape")
    default_outputs, default_threads = _recommended_launch(
        num_tokens, hidden_size, intermediate_size
    )
    launch_outputs = default_outputs if outputs_per_warp is None else outputs_per_warp
    launch_threads = default_threads if num_threads is None else num_threads
    if launch_outputs not in (1, 2, 4, 8):
        raise ValueError("outputs_per_warp must be one of 1, 2, 4, or 8")
    if launch_threads < 64 or launch_threads > 1024 or launch_threads % 32:
        raise ValueError("num_threads must be a warp multiple in [64, 1024]")


@functools.cache
def _get_module():
    module = gen_b12x_direct_w4a16_fused_moe_module().build_and_load()

    @register_custom_op(
        "flashinfer::b12x_direct_w4a16_fused_moe",
        mutates_args=["intermediate", "output"],
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
        intermediate: torch.Tensor,
        output: torch.Tensor,
        outputs_per_warp: int,
        num_threads: int,
    ) -> None:
        module.b12x_direct_w4a16_fused_moe(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            expert_map,
            intermediate,
            output,
            outputs_per_warp,
            num_threads,
        )

    @register_custom_op(
        "flashinfer::b12x_direct_pack_routes",
        mutates_args=[
            "packed_route_indices",
            "block_expert_ids",
            "packed_route_count",
        ],
    )
    def pack_routes(
        topk_ids: torch.Tensor,
        packed_route_indices: torch.Tensor,
        block_expert_ids: torch.Tensor,
        packed_route_count: torch.Tensor,
        num_experts: int,
        block_size: int,
        stream_int: int,
    ) -> None:
        module.b12x_direct_pack_routes(
            topk_ids,
            packed_route_indices,
            block_expert_ids,
            packed_route_count,
            num_experts,
            block_size,
            stream_int,
        )

    return SimpleNamespace(run=run, pack_routes=pack_routes)


def _b12x_direct_pack_routes(
    topk_ids: torch.Tensor,
    packed_route_indices: torch.Tensor,
    block_expert_ids: torch.Tensor,
    packed_route_count: torch.Tensor,
    *,
    num_experts: int,
    block_size: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _get_module().pack_routes(
        topk_ids,
        packed_route_indices,
        block_expert_ids,
        packed_route_count,
        num_experts,
        block_size,
        int(current_cuda_stream()),
    )
    return packed_route_indices, block_expert_ids, packed_route_count


def b12x_direct_w4a16_fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_scales: torch.Tensor,
    gemm2_weights: torch.Tensor,
    gemm2_scales: torch.Tensor,
    expert_map: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    outputs_per_warp: Optional[int] = None,
    num_threads: Optional[int] = None,
    *,
    skip_check: bool = False,
) -> torch.Tensor:
    r"""Run the experimental low-token B12x Direct W4A16 fused MoE.

    FP4 weights are B12x-compatible row-major E2M1 pairs. Scale tensors are
    model-load-time prepared row-major BF16 dequant multipliers, one per K/16
    block. Public activations, intermediate values, and output remain BF16.
    Pass a caller-owned ``workspace`` to select the tensor-core decode path and
    keep its scratch allocations stable for CUDA Graph capture; omitting it
    uses the scalar fallback.
    """
    if hidden_states.is_cuda:
        _require_cuda_129()
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
        )
    num_tokens, hidden_size = hidden_states.shape
    topk = int(topk_ids.shape[1])
    intermediate_size = int(gemm2_weights.shape[2]) * 2
    if workspace is not None:
        # The tensor-core path aliases this caller-owned buffer as the
        # activated FC1 output.  Keep this contract enforced even when the
        # fast ``skip_check`` mode is used; otherwise a smaller view could
        # turn the subsequent kernel launch into an out-of-bounds write.
        _check_tensor(
            "workspace",
            workspace,
            ndim=2,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        if tuple(workspace.shape) != (num_tokens * topk, intermediate_size):
            raise ValueError("workspace has an incompatible shape")
    if output is None:
        output = torch.empty_like(hidden_states)
    if (
        workspace is not None
        and outputs_per_warp is None
        and num_threads is None
        and (expert_map is None or expert_map.numel() == 0)
        and _try_run_tc_w4a16(
            hidden_states,
            topk_ids,
            topk_weights,
            gemm1_weights,
            gemm1_scales,
            gemm2_weights,
            gemm2_scales,
            workspace,
            output,
        )
    ):
        return output
    if workspace is None:
        items = num_tokens * topk * intermediate_size
        raw = _get_cache_buf(
            "b12x_direct_w4a16_fused_moe_workspace",
            items * torch.bfloat16.itemsize,
            hidden_states.device,
        )
        workspace = raw[: items * torch.bfloat16.itemsize].view(torch.bfloat16)
        workspace = workspace.reshape(num_tokens * topk, intermediate_size)
    if expert_map is None:
        raw = _get_cache_buf(
            "b12x_direct_w4a16_empty_expert_map", 4, hidden_states.device
        )
        expert_map = raw[:0].view(torch.int32)
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
        workspace,
        output,
        default_outputs if outputs_per_warp is None else outputs_per_warp,
        default_threads if num_threads is None else num_threads,
    )
    return output
