# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN backend for the bf16 x fp4 GEMM (graph build / execute / runner)."""

import functools
import warnings
from typing import List, Optional, Tuple

import torch

from ..autotuner import (
    AutoTuner,
    ConstraintSpec,
    DynamicTensorSpec,
    OptimizationProfile,
    TunableRunner,
    TuningConfig,
)
from ..fused_moe.utils import (
    get_hybrid_num_tokens_buckets,
    map_to_hybrid_bucket_uncapped,
)
from ..utils import _get_cache_buf, get_native_fp4_dtype

from .gemm_base import (
    CUDNN_AVAILABLE,
    DEFAULT_WORKSPACE_SIZE,
    UIDs,
    _check_cudnn_fp4_availability,
    _get_cudnn_handle,
    _get_cudnn_override_shape_workspace_size,
    _get_cudnn_plan_index_for_tactic,
    _get_cudnn_workspace_size,
    _cudnn_graph_engine_knob_tactics,
    _finalize_cudnn_graph_for_tactic,
    _tactic_for_graph_cache,
    _torch_data_type_to_cudnn_data_type,
    _is_cudnn_override_shape_available,
    _check_cudnn_override_shape_availability,
)
from .gemm_bf16_fp4 import _unswizzle_sf_128x4

if CUDNN_AVAILABLE:
    import cudnn

# Sentinel "cache M" for override-shape graphs (any value works; this one
# covers typical LLM inference shapes).  Kept local to avoid importing a
# private constant from gemm_base.
_OVERRIDE_SHAPE_CACHE_M = 8192


def _bf16_fp4_b_descale_layout(batch, n, k, block_size):
    """Return ``(dim, stride, reordering_type)`` for the B scale-factor tensor."""
    k_sf = k // block_size
    dim = (batch, k_sf, n)
    stride = (k_sf * n, 1, k_sf)
    return dim, stride, cudnn.tensor_reordering.NONE


def _build_bf16_fp4_graph_common(
    graph,
    a_cudnn_tensor,
    b_cudnn_tensor,
    block_descale_b_cudnn_tensor,
    a_type,
    o_type,
    block_size,
    alpha_is_not_none,
):
    """Shared graph body: dequant(B) -> A @ dequant(B) -> optional alpha."""
    dequant_b_tensor = graph.block_scale_dequantize(
        b_cudnn_tensor,
        block_descale_b_cudnn_tensor,
        block_size=[block_size, 1],
        name="dequant_b",
    )
    dequant_b_tensor.set_data_type(a_type)

    c_tensor = graph.matmul(
        a_cudnn_tensor,
        dequant_b_tensor,
        compute_data_type=cudnn.data_type.FLOAT,
        name="gemm",
    )
    c_tensor.set_data_type(cudnn.data_type.FLOAT)

    c_final_cudnn_tensor = c_tensor
    if alpha_is_not_none:
        global_scale_cudnn_tensor = graph.tensor(
            name="global_scale",
            dim=(1, 1, 1),
            stride=(1, 1, 1),
            data_type=cudnn.data_type.FLOAT,
        )
        c_final_cudnn_tensor = graph.mul(
            name="scale_mul",
            a=c_tensor,
            b=global_scale_cudnn_tensor,
            compute_data_type=cudnn.data_type.FLOAT,
        )
        global_scale_cudnn_tensor.set_uid(UIDs.ALPHA_UID.value)

    c_final_cudnn_tensor.set_name("c_final").set_output(True).set_data_type(o_type)

    a_cudnn_tensor.set_uid(UIDs.A_UID.value)
    b_cudnn_tensor.set_uid(UIDs.B_UID.value)
    block_descale_b_cudnn_tensor.set_uid(UIDs.BLOCK_DESCALE_B_UID.value)
    c_final_cudnn_tensor.set_uid(UIDs.O_UID.value)
    return c_final_cudnn_tensor


@functools.lru_cache(maxsize=2048)
def build_cudnn_bf16_fp4_graph(
    batch,
    m,
    n,
    k,
    a_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    tactic=-1,
):
    """Build a fixed-shape cuDNN bf16 x fp4 GEMM graph (no override-shape)."""
    _check_cudnn_fp4_availability()

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    # b weight bytes are row-major (N, K); present as column-major (K, N).
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, b_descale_reordering = (
        _bf16_fp4_b_descale_layout(batch, n, k, block_size)
    )

    stream = torch.cuda.current_stream(device)
    with cudnn.graph(_get_cudnn_handle(device, stream)) as (graph, _):
        a_cudnn_tensor = graph.tensor(
            name="a", dim=a_shape, stride=a_stride, data_type=a_type
        )
        b_cudnn_tensor = graph.tensor(
            name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP4_E2M1
        )
        block_descale_b_cudnn_tensor = graph.tensor(
            name="block_descale_b",
            dim=b_descale_shape,
            stride=b_descale_stride,
            data_type=scale_type,
            reordering_type=b_descale_reordering,
        )

        _build_bf16_fp4_graph_common(
            graph,
            a_cudnn_tensor,
            b_cudnn_tensor,
            block_descale_b_cudnn_tensor,
            a_type,
            o_type,
            block_size,
            alpha_is_not_none,
        )

        _finalize_cudnn_graph_for_tactic(
            graph, tactic, [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
        )
        return graph


@functools.lru_cache(maxsize=2048)
def build_cudnn_bf16_fp4_graph_override_shape(
    batch,
    n,
    k,
    a_type,
    o_type,
    block_size,
    device,
    alpha_is_not_none,
    use_nvfp4,
    cache_m: int = _OVERRIDE_SHAPE_CACHE_M,
    tactic=-1,
):
    """Build a cuDNN bf16 x fp4 GEMM graph with override-shape support."""
    _check_cudnn_fp4_availability()

    _check_cudnn_override_shape_availability()

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]
    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]
    b_descale_shape, b_descale_stride, b_descale_reordering = (
        _bf16_fp4_b_descale_layout(batch, n, k, block_size)
    )

    stream = torch.cuda.current_stream(device)
    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
        handle=_get_cudnn_handle(device, stream),
        is_override_shape_enabled=True,
    )

    a_cudnn_tensor = graph.tensor(
        name="a", dim=a_shape, stride=a_stride, data_type=a_type
    )
    b_cudnn_tensor = graph.tensor(
        name="b", dim=b_shape, stride=b_stride, data_type=cudnn.data_type.FP4_E2M1
    )
    block_descale_b_cudnn_tensor = graph.tensor(
        name="block_descale_b",
        dim=b_descale_shape,
        stride=b_descale_stride,
        data_type=scale_type,
        reordering_type=b_descale_reordering,
    )

    _build_bf16_fp4_graph_common(
        graph,
        a_cudnn_tensor,
        b_cudnn_tensor,
        block_descale_b_cudnn_tensor,
        a_type,
        o_type,
        block_size,
        alpha_is_not_none,
    )

    _finalize_cudnn_graph_for_tactic(
        graph, tactic, [cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK]
    )
    return graph


def _bf16_fp4_variant_pack(a, b, b_descale, alpha, out):
    """Build the {uid: tensor} variant pack shared by both execute paths."""
    variant_pack = {
        UIDs.A_UID.value: a,
        UIDs.B_UID.value: b.view(get_native_fp4_dtype()),
        UIDs.BLOCK_DESCALE_B_UID.value: b_descale,
        UIDs.O_UID.value: out,
    }
    if alpha is not None:
        variant_pack[UIDs.ALPHA_UID.value] = alpha.view(torch.float)
    return variant_pack


def execute_cudnn_bf16_fp4_graph(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    tactic=-1,
):
    variant_pack = _bf16_fp4_variant_pack(a, b, b_descale, alpha, out)

    plan_index = _get_cudnn_plan_index_for_tactic(graph, tactic)

    workspace_size = _get_cudnn_workspace_size(graph, plan_index)
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    stream = torch.cuda.current_stream(a.device)
    handle = _get_cudnn_handle(a.device, stream)
    if plan_index < 0:
        graph.execute(variant_pack, workspace_buffer, handle=handle)
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace_buffer, plan_index, handle=handle
        )


def execute_cudnn_bf16_fp4_graph_override_shape(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    block_size: int = 16,
    tactic=-1,
):
    """Execute the bf16 x fp4 graph, overriding A / output to the real M."""
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[0])
    batch = 1

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, _ = _bf16_fp4_b_descale_layout(
        batch, n, k, block_size
    )
    out_shape = (batch, m, n)
    out_stride = (m * n, n, 1)

    variant_pack = _bf16_fp4_variant_pack(a, b, b_descale, alpha, out)

    override_uids = [
        UIDs.A_UID.value,
        UIDs.B_UID.value,
        UIDs.BLOCK_DESCALE_B_UID.value,
        UIDs.O_UID.value,
    ]
    override_shapes = [a_shape, b_shape, b_descale_shape, out_shape]
    override_strides = [a_stride, b_stride, b_descale_stride, out_stride]

    stream = torch.cuda.current_stream(a.device)
    cudnn_handle = _get_cudnn_handle(a.device, stream)

    plan_index = _get_cudnn_plan_index_for_tactic(graph, tactic)

    workspace_size = _get_cudnn_override_shape_workspace_size(
        graph,
        plan_index,
        cudnn_handle,
        override_uids,
        override_shapes,
        override_strides,
    )
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    if plan_index < 0:
        graph.execute(
            variant_pack,
            workspace_buffer,
            handle=cudnn_handle,
            override_uids=override_uids,
            override_shapes=override_shapes,
            override_strides=override_strides,
        )
    else:
        graph.execute_plan_at_index(
            variant_pack,
            workspace_buffer,
            plan_index,
            handle=cudnn_handle,
            override_uids=override_uids,
            override_shapes=override_shapes,
            override_strides=override_strides,
        )


# Autotuner sweeps M (token count) of the bf16 activation ``a``
_BF16_FP4_TUNING_CONFIG = TuningConfig(
    dynamic_tensor_specs=(
        DynamicTensorSpec(
            (0,),  # a_tensor_index
            (0,),  # M dimension
            get_hybrid_num_tokens_buckets,
            map_to_hybrid_bucket_uncapped,
        ),
    ),
    constraint_specs=(
        ConstraintSpec(
            5,  # out_tensor_index follows M
            0,
            lambda shapes: shapes[0][0],
        ),
    ),
)


def _cudnn_bf16_fp4_runner(tuning_config):
    """Build a ``CudnnBf16Fp4Runner`` bound to the active tuning config."""
    m_bucket_mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(
        tuning_config, spec_idx=0
    )

    class CudnnBf16Fp4Runner(TunableRunner):
        def __init__(self):
            super().__init__()
            self._m_bucket_mapper = m_bucket_mapper
            self._use_override_shape = _is_cudnn_override_shape_available()

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            _, _, _, alpha, out_dtype, _, block_size, use_nvfp4, _ = inputs
            return (out_dtype, block_size, use_nvfp4, alpha is not None)

        def _get_override_graph(
            self, a, b, alpha, out_dtype, block_size, use_nvfp4, tactic=-1
        ):
            actual_m, k = int(a.shape[0]), int(a.shape[1])
            n = int(b.shape[0])
            cache_m = self._m_bucket_mapper(actual_m)
            return build_cudnn_bf16_fp4_graph_override_shape(
                batch=1,
                n=n,
                k=k,
                a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                block_size=block_size,
                device=a.device,
                alpha_is_not_none=alpha is not None,
                use_nvfp4=use_nvfp4,
                cache_m=cache_m,
                tactic=_tactic_for_graph_cache(tactic),
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[tuple]:
            (
                a,
                b,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs
            if self._use_override_shape:
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4, tactic=0
                )
            else:
                graph = build_cudnn_bf16_fp4_graph(
                    batch=1,
                    m=int(a.shape[0]),
                    n=int(b.shape[0]),
                    k=int(a.shape[1]),
                    a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                    o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                    block_size=block_size,
                    device=a.device,
                    alpha_is_not_none=alpha is not None,
                    use_nvfp4=use_nvfp4,
                    tactic=0,
                )
            return _cudnn_graph_engine_knob_tactics(graph)

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=-1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            (
                a,
                b,
                b_descale,
                alpha,
                out_dtype,
                out,
                block_size,
                use_nvfp4,
                workspace_buffer,
            ) = inputs
            try:
                if self._use_override_shape:
                    graph = self._get_override_graph(
                        a, b, alpha, out_dtype, block_size, use_nvfp4, tactic=tactic
                    )
                    execute_cudnn_bf16_fp4_graph_override_shape(
                        graph,
                        a,
                        b,
                        b_descale,
                        alpha,
                        out,
                        workspace_buffer,
                        block_size=block_size,
                        tactic=tactic,
                    )
                else:
                    graph = build_cudnn_bf16_fp4_graph(
                        batch=1,
                        m=int(a.shape[0]),
                        n=int(b.shape[0]),
                        k=int(a.shape[1]),
                        a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                        o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                        block_size=block_size,
                        device=a.device,
                        alpha_is_not_none=alpha is not None,
                        use_nvfp4=use_nvfp4,
                        tactic=tactic,
                    )
                    execute_cudnn_bf16_fp4_graph(
                        graph,
                        a,
                        b,
                        b_descale,
                        alpha,
                        out,
                        workspace_buffer,
                        tactic=tactic,
                    )
            except Exception as exc:
                warnings.warn(
                    "cuDNN bf16-fp4 GEMM tactic failed; falling back to default "
                    f"tactic=-1. ({exc})",
                    stacklevel=2,
                )
                graph = build_cudnn_bf16_fp4_graph(
                    batch=1,
                    m=int(a.shape[0]),
                    n=int(b.shape[0]),
                    k=int(a.shape[1]),
                    a_type=_torch_data_type_to_cudnn_data_type(a.dtype),
                    o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
                    block_size=block_size,
                    device=a.device,
                    alpha_is_not_none=alpha is not None,
                    use_nvfp4=use_nvfp4,
                    tactic=-1,
                )
                execute_cudnn_bf16_fp4_graph(
                    graph,
                    a,
                    b,
                    b_descale,
                    alpha,
                    out,
                    workspace_buffer,
                    tactic=-1,
                )
            return out

    return CudnnBf16Fp4Runner()


def _prepare_cudnn(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cuDNN-backend prep.

    The weight bytes ``(N, K//2)`` are already in the layout the cuDNN graph
    consumes (see the module banner).  The scale factor, however, must be
    *non-swizzled* for the cuDNN bf16 x fp4 path (cuDNN does not support the
    128x4-swizzled SF layout), so we unswizzle the canonical 128x4 SF into a
    linear ``(N, K // block_size)`` FP8-E4M3 tensor.
    """
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size
    # (N, K_sf) uint8 bytes, each byte an FP8-E4M3 per-block scale.
    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf).contiguous()
    return b, linear_sf.view(torch.float8_e4m3fn), alpha


def _compute_cudnn(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
) -> torch.Tensor:
    """cuDNN-backend compute with autotuning over the M (token) dimension."""
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from b.shape={tuple(b.shape)} "
            f"is {k}"
        )

    if out is None:
        out = torch.empty((a.shape[0], n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (a.shape[0], n):
            raise ValueError(
                f"out shape {tuple(out.shape)} != expected {(a.shape[0], n)}"
            )
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")

    workspace_buffer = _get_cache_buf(
        "mm_bf16_fp4_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    tuning_config = _BF16_FP4_TUNING_CONFIG
    tuner = AutoTuner.get()
    runner = _cudnn_bf16_fp4_runner(tuning_config)

    use_nvfp4 = True
    inputs = [
        a,
        b,
        b_descale,
        alpha,
        out_dtype,
        out,
        block_size,
        use_nvfp4,
        workspace_buffer,
    ]
    chosen_runner, tactic = tuner.choose_one(
        "bf16_fp4_gemm",
        [runner],
        tuning_config,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out
