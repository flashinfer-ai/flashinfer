# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import functools
from typing import List, Literal, Optional, Tuple, cast

import torch

from ..api_logging import flashinfer_api
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
from ..utils import (
    _get_cache_buf,
    backend_requirement,
    get_native_fp4_dtype,
    supported_compute_capability,
)

from .gemm_base import (
    CUDNN_AVAILABLE,
    DEFAULT_WORKSPACE_SIZE,
    UIDs,
    _TORCH_TO_CUTLASS_DTYPE_ATTR,
    _check_cudnn_fp4_availability,
    _check_cute_dsl_availability,
    _get_cudnn_handle,
    _get_cudnn_override_shape_workspace_size,
    _get_cudnn_workspace_size,
    _torch_data_type_to_cudnn_data_type,
    is_cudnn_override_shape_available,
)

if CUDNN_AVAILABLE:
    import cudnn


# Earliest cuDNN backend version supporting W4A16 GEMM
_CUDNN_W4A16_MIN_BACKEND_VERSION = 92301


def _check_mm_w4a16_fp4_problem_size(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    if a.dim() != 2:
        raise ValueError(f"a must be 2-D (M, K); got shape {tuple(a.shape)}")
    if a.dtype != torch.bfloat16:
        raise TypeError(
            f"a must be bfloat16; got {a.dtype}.  fp16 support is not implemented yet."
        )
    if out_dtype is not None and out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"out_dtype must be bfloat16 or float16; got {out_dtype}")
    if block_size != 16:
        raise ValueError(f"block_size must be 16 for FP4; got {block_size}")
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cudnn_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    """cuDNN backend: requires a cuDNN build with FP4 block-scale support."""
    _check_cudnn_fp4_availability()

    backend_version = cudnn.backend_version()
    if backend_version < _CUDNN_W4A16_MIN_BACKEND_VERSION:
        raise RuntimeError(
            f"cuDNN W4A16 FP4 GEMM requires backend version >= "
            f"{_CUDNN_W4A16_MIN_BACKEND_VERSION} (9.23.1), found {backend_version}. "
        )
    return True


@supported_compute_capability([100, 103, 110, 120, 121])
def _cute_dsl_w4a16_fp4_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
):
    _check_cute_dsl_availability()
    return True


# =============================================================================
# Public dispatchers
# =============================================================================


@flashinfer_api
def prepare_w4a16_fp4_weights(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    block_size: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Prepare FP4 W4A16 weights for a specific backend.

    The caller is expected to start with weights in the canonical format
    that :func:`flashinfer.nvfp4_quantize` produces with
    ``sfLayout=layout_128x4``:

    * ``b`` is ``(N, K // 2)`` ``uint8`` with two FP4 codes packed per
      byte (low nibble = K=2i, high nibble = K=2i+1).
    * ``b_descale`` is the 128x4-swizzled FP8-E4M3 per-block scales,
      either as a 1-D byte buffer or a 2-D tensor.

    Each backend transforms these into whatever layout its compute kernel
    expects.  The returned ``(b, b_descale, alpha)`` tuple must be passed
    back to :func:`mm_w4a16_fp4` with the *same* ``backend`` -- the
    shapes / dtypes may not match other backends' expectations.

    Args:
        b: ``(N, K // 2)`` ``uint8`` packed FP4 weight.
        b_descale: 128x4-swizzled FP8-E4M3 scale factors from
            ``nvfp4_quantize``.  Either 1-D byte buffer or 2-D tensor.
        alpha: Optional ``(1,) float32`` global scalar.  Pass ``None``
            (default) for implicit ``alpha=1.0``.  Returned unchanged;
            forward the returned tuple to :func:`mm_w4a16_fp4`.
        backend: Identifier of a supported backend (``"cudnn"`` or
            ``"cute-dsl"``).
        block_size: SF block size.  Always 16 for FP4.

    Returns:
        ``(b_prepared, b_descale_prepared, alpha_prepared)`` -- pass all
        three to :func:`mm_w4a16_fp4` with the same ``backend``.

    Raises:
        ValueError: ``backend`` is unknown, or an input has an invalid
            shape (``b`` not 2-D, ``K`` not a multiple of ``block_size``,
            or ``alpha`` not shape ``(1,)``).
        TypeError: ``b`` is not ``uint8`` or ``alpha`` is not ``float32``.
    """
    if b.dim() != 2:
        raise ValueError(f"b must be 2-D (N, K/2); got shape {tuple(b.shape)}")
    if b.dtype != torch.uint8:
        raise TypeError(f"b must be uint8; got {b.dtype}")
    k = int(b.shape[1]) * 2
    if k % block_size != 0:
        raise ValueError(f"K={k} must be a multiple of block_size={block_size}")
    if alpha is not None:
        if alpha.dim() != 1 or alpha.shape[0] != 1:
            raise ValueError(f"alpha must be shape (1,); got {tuple(alpha.shape)}")
        if alpha.dtype != torch.float32:
            raise TypeError(f"alpha must be float32; got {alpha.dtype}")
    if backend == "cudnn":
        return _prepare_cudnn(b, b_descale, alpha, block_size)
    if backend == "cute-dsl":
        return _prepare_cute_dsl(b, b_descale, alpha, block_size)
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


@backend_requirement(
    {
        "cudnn": _cudnn_w4a16_fp4_requirement,
        "cute-dsl": _cute_dsl_w4a16_fp4_requirement,
    },
    common_check=_check_mm_w4a16_fp4_problem_size,
)
@flashinfer_api
def mm_w4a16_fp4(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor] = None,
    *,
    backend: Literal["cudnn", "cute-dsl"],
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
    block_size: int = 16,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """W4A16 FP4 GEMM: ``out = (a @ dequant(b).T) * alpha``.

    ``b``, ``b_descale``, and ``alpha`` must be the tensors returned by
    :func:`prepare_w4a16_fp4_weights` with the same ``backend``.  Mixing
    backends (preparing with one, computing with another) is undefined.

    Example:
        .. code-block:: python

            # 1) Prepare weights for a backend (once, at model load).
            b_p, sf_p, alpha_p = flashinfer.prepare_w4a16_fp4_weights(
                b, b_descale, alpha, backend="cute-dsl",
            )
            # 2) Run the GEMM with the *same* backend tag.
            out = flashinfer.mm_w4a16_fp4(
                a, b_p, sf_p, alpha_p, backend="cute-dsl",
            )

    Args:
        a: ``(M, K)`` activation matrix in ``torch.bfloat16``.  This is
            the only currently supported activation dtype; fp16 support
            can be added when needed.
        b: Prepared weight tensor (backend-specific layout).
        b_descale: Prepared scale-factor tensor (backend-specific layout).
        alpha: Optional ``(1,) float32`` global scalar.  Pass through
            whatever ``prepare_w4a16_fp4_weights`` returned -- it may be
            ``None`` if the backend folded it into ``b_descale``.
        backend: Same identifier passed to ``prepare_w4a16_fp4_weights``.
        out_dtype: Output dtype.  Defaults to ``a.dtype`` (``bfloat16``).
        out: Optional preallocated ``(M, N)`` output tensor.
        block_size: SF block size.  Always 16 for FP4.
        enable_pdl: Enable Programmatic Dependent Launch
    Returns:
        ``(M, N)`` tensor of ``out_dtype``.

    """
    out_dtype = out_dtype or a.dtype
    if backend == "cudnn":
        return _compute_cudnn(a, b, b_descale, alpha, out_dtype, out, block_size)
    if backend == "cute-dsl":
        return _compute_cute_dsl(
            a, b, b_descale, alpha, out_dtype, out, block_size, enable_pdl=enable_pdl
        )
    raise ValueError(f"Unknown backend {backend!r}.  Supported: 'cudnn', 'cute-dsl'.")


# =============================================================================
# Shared SF utility
# =============================================================================
#
# Used by both backends' prepare paths (cuDNN and cute-dsl) to turn the
# canonical 128x4-swizzled SF into a linear ``(N, K_sf)`` layout.


def _unswizzle_sf_128x4(sf_swizzled: torch.Tensor, n: int, k_sf: int) -> torch.Tensor:
    """Reverse the 128x4 SF swizzle into a flat ``(N, K_sf)`` byte tensor.

    The swizzle stores SF in 512-byte blocks each holding 128 N-rows x 4
    K_sf-cols.  The byte address of logical ``(n, k_sf)`` is::

        offset = ((n // 128) * sf_pad_blocks + k_sf // 4) * 512
               + (n % 32) * 16
               + ((n % 128) // 32) * 4
               + (k_sf % 4)

    where ``sf_pad_blocks = ceil(k_sf, 4) // 4`` accounts for K_sf
    padding inside each 128-row N block.
    """
    device = sf_swizzled.device
    sf_flat = sf_swizzled.contiguous().view(torch.uint8).view(-1)
    sf_pad_blocks = (k_sf + 3) // 4  # ceil_div(k_sf, 4)
    n_idx = torch.arange(n, device=device, dtype=torch.int64)
    k_idx = torch.arange(k_sf, device=device, dtype=torch.int64)
    n_grid, k_grid = torch.meshgrid(n_idx, k_idx, indexing="ij")
    offsets = (
        ((n_grid // 128) * sf_pad_blocks + (k_grid // 4)) * 512
        + (n_grid % 32) * 16
        + ((n_grid % 128) // 32) * 4
        + (k_grid % 4)
    )
    return sf_flat[offsets]


# =============================================================================
# cuDNN backend
# =============================================================================

# Sentinel "cache M" for override-shape graphs (any value works; this one
# covers typical LLM inference shapes).  Kept local to avoid importing a
# private constant from gemm_base.
_OVERRIDE_SHAPE_CACHE_M = 8192


def _w4a16_b_descale_layout(batch, n, k, block_size):
    """Return ``(dim, stride, reordering_type)`` for the B scale-factor tensor."""
    k_sf = k // block_size
    dim = (batch, k_sf, n)
    stride = (k_sf * n, 1, k_sf)
    return dim, stride, cudnn.tensor_reordering.NONE


def _build_w4a16_fp4_graph_common(
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


@functools.lru_cache(maxsize=1024)
def build_cudnn_w4a16_fp4_graph(
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
    policy=None,
):
    """Build a fixed-shape cuDNN W4A16 FP4 GEMM graph (no override-shape)."""
    _check_cudnn_fp4_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    # b weight bytes are row-major (N, K); present as column-major (K, N).
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, b_descale_reordering = _w4a16_b_descale_layout(
        batch, n, k, block_size
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

        _build_w4a16_fp4_graph_common(
            graph,
            a_cudnn_tensor,
            b_cudnn_tensor,
            block_descale_b_cudnn_tensor,
            a_type,
            o_type,
            block_size,
            alpha_is_not_none,
        )

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
        graph.check_support()
        graph.build_plans(policy)
        return graph


@functools.lru_cache(maxsize=1024)
def build_cudnn_w4a16_fp4_graph_override_shape(
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
    policy=None,
):
    """Build a cuDNN W4A16 FP4 GEMM graph with override-shape support."""
    _check_cudnn_fp4_availability()
    if policy is None:
        policy = cudnn.build_plan_policy.HEURISTICS_CHOICE

    scale_type = cudnn.data_type.FP8_E4M3 if use_nvfp4 else cudnn.data_type.FP8_E8M0

    a_shape = [batch, cache_m, k]
    a_stride = [cache_m * k, k, 1]
    b_shape = [batch, k, n]
    b_stride = [k * n, 1, k]
    b_descale_shape, b_descale_stride, b_descale_reordering = _w4a16_b_descale_layout(
        batch, n, k, block_size
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

    _build_w4a16_fp4_graph_common(
        graph,
        a_cudnn_tensor,
        b_cudnn_tensor,
        block_descale_b_cudnn_tensor,
        a_type,
        o_type,
        block_size,
        alpha_is_not_none,
    )

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans(policy)
    return graph


def _w4a16_variant_pack(a, b, b_descale, alpha, out):
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


def execute_cudnn_w4a16_fp4_graph(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    tactic: int = -1,
):
    variant_pack = _w4a16_variant_pack(a, b, b_descale, alpha, out)

    workspace_size = _get_cudnn_workspace_size(graph, tactic)
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    stream = torch.cuda.current_stream(a.device)
    handle = _get_cudnn_handle(a.device, stream)
    if tactic == -1:
        graph.execute(variant_pack, workspace_buffer, handle=handle)
    else:
        graph.execute_plan_at_index(
            variant_pack, workspace_buffer, tactic, handle=handle
        )


def execute_cudnn_w4a16_fp4_graph_override_shape(
    graph,
    a,
    b,
    b_descale,
    alpha,
    out,
    workspace_buffer,
    block_size: int = 16,
    tactic: int = 0,
):
    """Execute the W4A16 graph, overriding A / output to the real M."""
    m, k = int(a.shape[0]), int(a.shape[1])
    n = int(b.shape[0])
    batch = 1

    a_shape = (batch, m, k)
    a_stride = (m * k, k, 1)
    b_shape = (batch, k, n)
    b_stride = (k * n, 1, k)
    b_descale_shape, b_descale_stride, _ = _w4a16_b_descale_layout(
        batch, n, k, block_size
    )
    out_shape = (batch, m, n)
    out_stride = (m * n, n, 1)

    variant_pack = _w4a16_variant_pack(a, b, b_descale, alpha, out)

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

    workspace_size = _get_cudnn_override_shape_workspace_size(
        graph, tactic, cudnn_handle, override_uids, override_shapes, override_strides
    )
    if workspace_buffer.numel() < workspace_size:
        workspace_buffer.resize_(workspace_size)

    graph.execute_plan_at_index(
        variant_pack,
        workspace_buffer,
        tactic,
        handle=cudnn_handle,
        override_uids=override_uids,
        override_shapes=override_shapes,
        override_strides=override_strides,
    )


# Autotuner sweeps M (token count) of the bf16 activation ``a`` and keeps
# the output ``out`` in lockstep.  The FP4 weight ``b`` and its scale are
# M-independent and stay fixed during profiling.
_W4A16_FP4_TUNING_CONFIG = TuningConfig(
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


def _cudnn_w4a16_fp4_runner(tuning_config):
    """Build a ``CudnnW4a16Fp4Runner`` bound to the active tuning config."""
    m_bucket_mapper = AutoTuner.get().get_effective_map_to_tuning_buckets(
        tuning_config, spec_idx=0
    )

    class CudnnW4a16Fp4Runner(TunableRunner):
        def __init__(self):
            super().__init__()
            self._m_bucket_mapper = m_bucket_mapper
            self._use_override_shape = is_cudnn_override_shape_available()

        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            _, _, _, alpha, out_dtype, _, block_size, use_nvfp4, _ = inputs
            return (out_dtype, block_size, use_nvfp4, alpha is not None)

        def _get_override_graph(self, a, b, alpha, out_dtype, block_size, use_nvfp4):
            actual_m, k = int(a.shape[0]), int(a.shape[1])
            n = int(b.shape[0])
            cache_m = self._m_bucket_mapper(actual_m)
            return build_cudnn_w4a16_fp4_graph_override_shape(
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
                policy=cudnn.build_plan_policy.ALL,
            )

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
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
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )
            else:
                graph = build_cudnn_w4a16_fp4_graph(
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
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
            return list(range(graph.get_execution_plan_count()))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
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
            if self._use_override_shape:
                graph = self._get_override_graph(
                    a, b, alpha, out_dtype, block_size, use_nvfp4
                )
                execute_cudnn_w4a16_fp4_graph_override_shape(
                    graph,
                    a,
                    b,
                    b_descale,
                    alpha,
                    out,
                    workspace_buffer,
                    block_size=block_size,
                    tactic=max(tactic, 0),
                )
            else:
                graph = build_cudnn_w4a16_fp4_graph(
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
                    policy=cudnn.build_plan_policy.HEURISTICS_CHOICE,
                )
                execute_cudnn_w4a16_fp4_graph(
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

    return CudnnW4a16Fp4Runner()


def _prepare_cudnn(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cuDNN-backend prep.

    The weight bytes ``(N, K//2)`` are already in the layout the cuDNN graph
    consumes (see the module banner).  The scale factor, however, must be
    *non-swizzled* for the cuDNN W4A16 path (cuDNN does not support the
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
        "mm_w4a16_fp4_workspace", DEFAULT_WORKSPACE_SIZE, a.device
    )

    tuning_config = _W4A16_FP4_TUNING_CONFIG
    tuner = AutoTuner.get()
    runner = _cudnn_w4a16_fp4_runner(tuning_config)

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
        "w4a16_fp4_gemm",
        [runner],
        tuning_config,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out


# =============================================================================
# cute-DSL backend
# =============================================================================

_W4A16_ALPHA_ONE_CACHE: dict = {}


def _prepare_w4a16_alpha(
    alpha: Optional[torch.Tensor], device: torch.device
) -> torch.Tensor:
    """Normalize ``alpha`` to a ``(1,) float32`` tensor for the kernel.

    ``alpha=None`` means implicit ``1.0``; we cache the per-device unit
    scalar so the hot path doesn't allocate.
    """
    if alpha is None:
        cached = _W4A16_ALPHA_ONE_CACHE.get(device)
        if cached is None:
            cached = torch.tensor([1.0], dtype=torch.float32, device=device)
            _W4A16_ALPHA_ONE_CACHE[device] = cached
        return cached
    if alpha.dim() == 0:
        return alpha.to(torch.float32).unsqueeze(0)
    return alpha.to(torch.float32).reshape(1)


def _select_w4a16_tile_shape(
    m: int, n: int, k: int
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Pick a CTA tile shape AND MMA atom_layout for the cute-DSL W4A16 kernel.

    Returns ``(tile_shape_mnk, atom_layout)``.

    Tile shape selection:
      tile_M choice
        * M <= 16 (and tile_K=128 path): use tile_M=16 with atom_layout
          (1,2,1).  Halves wasted M-rows vs tile_M=32, and a 1-M-warp
          layout removes the duplicate dequant that (2,2,1) suffers from.
        * 16 < M <= 32: use tile_M=32 with atom_layout (2,2,1).  Smaller
          MMA + epilogue waste than tile_M=64.
        * M > 32: use tile_M=64 with atom_layout (2,2,1) -- standard tile,
          more rows to amortize across.

      tile_K choice
        * K % 128 == 0: tile_K=128 (halves K-tile count and barrier
          overhead).
        * Otherwise: tile_K=64.

    Why atom_layout differs:
      * (2,2,1) (default for tile_M >= 32): 4 MMA warps as 2 M x 2 N --
        well-tested cute layout, but the 2 M-warps redundantly dequant
        the same B values into their own register files (~50% waste in
        dequant compute).
      * (1,2,1) (used for tile_M=16): 2 MMA warps as 1 M x 2 N -- no
        M-warp duplication.  Permutation_m = 16, so tile_M must be 16.
    """
    tile_k = 128 if k % 128 == 0 else 64
    if m <= 16 and tile_k == 128:
        return ((16, 64, 128), (1, 2, 1))
    if m <= 32:
        return ((32, 64, tile_k), (2, 2, 1))
    return ((64, 64, tile_k), (2, 2, 1))


_CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE: dict = {}


def _get_cute_dsl_w4a16_gemm(
    tile_shape_mnk: Tuple[int, int, int],
    a_dtype: torch.dtype,
    c_dtype: torch.dtype,
    atom_layout: Tuple[int, int, int] = (2, 2, 1),
    pipeline_depth: int = 1,
    use_fp16_mma: int = 1,
    enable_pdl: bool = True,
    tile_swizzle: int = 1,
):
    # Normalize to a tuple (callers may pass a list) so the cache key is hashable.
    atom_layout = cast(Tuple[int, int, int], tuple(atom_layout))
    pipeline_depth = int(pipeline_depth)
    use_fp16_mma = int(use_fp16_mma)
    enable_pdl = bool(enable_pdl)
    tile_swizzle = int(tile_swizzle)
    cache_key = (
        tile_shape_mnk,
        a_dtype,
        c_dtype,
        atom_layout,
        pipeline_depth,
        use_fp16_mma,
        enable_pdl,
        tile_swizzle,
    )
    cached = _CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    _check_cute_dsl_availability()

    import cutlass
    import cutlass.cute as cute
    from flashinfer.cute_dsl.utils import get_max_active_clusters

    from .kernels.dense_gemm_w4a16_blackwell import (
        BlackwellDenseGemmW4A16Kernel,
    )

    a_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[a_dtype])
    c_cutlass_dtype = getattr(cutlass, _TORCH_TO_CUTLASS_DTYPE_ATTR[c_dtype])

    sym_m = cute.sym_int()
    sym_k = cute.sym_int()
    sym_n = cute.sym_int()
    sym_k_tiles = cute.sym_int()
    sym_n_packed = cute.sym_int()

    a_fake = cute.runtime.make_fake_compact_tensor(
        a_cutlass_dtype, (sym_m, sym_k), stride_order=(1, 0), assumed_align=16
    )
    b_packed_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_k_tiles, sym_n_packed),
        stride_order=(1, 0),
        assumed_align=16,
    )
    b_sf_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_k_tiles, sym_n), stride_order=(1, 0), assumed_align=16
    )
    c_fake = cute.runtime.make_fake_compact_tensor(
        c_cutlass_dtype, (sym_m, sym_n), stride_order=(1, 0), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    gemm = BlackwellDenseGemmW4A16Kernel(
        acc_dtype=cutlass.Float32,
        tile_shape_mnk=tile_shape_mnk,
        atom_layout=atom_layout,
        pipeline_depth=pipeline_depth,
        use_fp16_mma=use_fp16_mma,
        enable_pdl=enable_pdl,
        tile_swizzle=tile_swizzle,
    )
    max_active_clusters = get_max_active_clusters(1)

    compiled = cute.compile(
        gemm.wrapper,
        a_fake,
        b_packed_fake,
        b_sf_fake,
        c_fake,
        alpha_fake,
        1,  # l (batch)
        max_active_clusters,
        stream_fake,
        options="--opt-level 2 --enable-tvm-ffi",
    )

    _CUTE_DSL_MM_FP4_W4A16_KERNEL_CACHE[cache_key] = compiled
    return compiled


def _e4m3_to_s0e5m3(sf_u8: torch.Tensor) -> torch.Tensor:
    """Reformat a uint8 tensor of E4M3 scale bytes to S0E5M3 bytes.
    Used in cute-dsl backend for faster in-kernel scale decode.
    """
    f16 = sf_u8.contiguous().view(torch.float8_e4m3fn).to(torch.float16)
    bits = f16.view(torch.int16).to(torch.int32) & 0xFFFF
    return ((bits >> 7) & 0xFF).to(torch.uint8)


# -----------------------------------------------------------------------------
# Host-side FP4 weight repack for the cute-DSL kernel.
# -----------------------------------------------------------------------------

_CUTE_DSL_PACK_TILE_K: int = 16  # K-tile size = MMA K-block size
_CUTE_DSL_PACK_TILE_N: int = 64  # N-tile size = kernel tile_N
_CUTE_DSL_PACK_INTS_PER_TILE: int = 128  # int32s per (16K x 64N) repack block


def _cute_dsl_pack_fp4_weight(b: torch.Tensor) -> torch.Tensor:
    """Repack a packed FP4 weight for the W4A16 cute-DSL kernel."""
    if b.dtype != torch.uint8:
        b = b.view(torch.uint8)

    k_half, n = b.shape
    k = k_half * 2
    if k % _CUTE_DSL_PACK_TILE_K != 0:
        raise ValueError(f"K must be a multiple of {_CUTE_DSL_PACK_TILE_K} (got K={k})")
    if n % _CUTE_DSL_PACK_TILE_N != 0:
        raise ValueError(f"N must be a multiple of {_CUTE_DSL_PACK_TILE_N} (got N={n})")

    device = b.device
    k_tiles = k // _CUTE_DSL_PACK_TILE_K
    n_tiles = n // _CUTE_DSL_PACK_TILE_N
    k_half_per_tile = _CUTE_DSL_PACK_TILE_K // 2  # 8 packed K-rows per tile

    # The repack is a fixed permutation of each (8 K-half x 64 N) source tile
    # into 128 int32 (4 bytes each).  The within-tile byte mapping depends only
    # on the layout, so build the 512-entry permutation once and apply it with a
    # single small-index gather -- far cheaper than materializing the full
    # (K_tiles, N_tiles, 128, 4) int64 index tensors of the per-element gather.
    u32_pos = torch.arange(
        _CUTE_DSL_PACK_INTS_PER_TILE, device=device, dtype=torch.long
    )
    u32_idx_local = u32_pos % 2
    lane = (u32_pos // 2) % 32
    n_warp_idx = u32_pos // 64

    tc_col = lane // 4  # in [0, 8)
    tc_row_half = lane % 4  # tc_row = tc_row_half * 2 in {0, 2, 4, 6}
    base_n = n_warp_idx * 8 + tc_col  # in [0, 16)

    byte_k_half_offset = torch.tensor([0, 4, 0, 4], device=device, dtype=torch.long)
    n_offset_stack = torch.tensor(
        [[0, 0, 16, 16], [32, 32, 48, 48]], device=device, dtype=torch.long
    )
    byte_n_offset = n_offset_stack[u32_idx_local]  # (128, 4)

    # Source byte within the (8, 64) tile for each (u32_pos, byte_idx).
    k_half_in_tile = tc_row_half[:, None] + byte_k_half_offset[None, :]  # (128, 4)
    n_in_tile = base_n[:, None] + byte_n_offset  # (128, 4)
    within_idx = (k_half_in_tile * _CUTE_DSL_PACK_TILE_N + n_in_tile).reshape(
        -1
    )  # (512,) flat index into a row-major (8, 64) tile

    # (K/2, N) -> (K_tiles, 8, N_tiles, 64) -> (K_tiles, N_tiles, 8*64) so the
    # 512 source bytes of each tile are contiguous, then gather them in
    # (u32_pos, byte_idx) order.
    tile_bytes = (
        b.reshape(k_tiles, k_half_per_tile, n_tiles, _CUTE_DSL_PACK_TILE_N)
        .permute(0, 2, 1, 3)
        .reshape(k_tiles, n_tiles, k_half_per_tile * _CUTE_DSL_PACK_TILE_N)
    )
    gathered = tile_bytes[:, :, within_idx].reshape(
        k_tiles, n_tiles, _CUTE_DSL_PACK_INTS_PER_TILE, 4
    )

    # Pack 4 bytes (little-endian: byte_idx 0 in bits 0-7) into one int32.
    out64 = torch.zeros(
        (k_tiles, n_tiles, _CUTE_DSL_PACK_INTS_PER_TILE),
        dtype=torch.int64,
        device=device,
    )
    for byte_idx in range(4):
        out64 |= gathered[..., byte_idx].to(torch.int64) << (byte_idx * 8)

    return out64.to(torch.int32).reshape(
        k_tiles, n_tiles * _CUTE_DSL_PACK_INTS_PER_TILE
    )


def _prepare_cute_dsl(
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """cute-DSL-backend prep: repack the weight + unswizzle the SF.

    Produces the bespoke layout the cute-DSL kernel consumes:
      * weight: ``(K // 16, N * 2)`` int32 (see :func:`_cute_dsl_pack_fp4_weight`).
      * SF:     ``(K // block_size, N)`` uint8 -- per-block scales reformatted to
        S0E5M3, the format the cute-DSL kernel decodes.
    ``alpha`` is passed through unchanged (the compute step normalizes it
    to a ``(1,) float32`` scalar).  Pair the returned tensors with
    ``mm_w4a16_fp4(..., backend='cute-dsl')``.
    """
    n = int(b.shape[0])
    k = int(b.shape[1]) * 2
    k_sf = k // block_size

    b_kn = b.t().contiguous()
    b_packed = _cute_dsl_pack_fp4_weight(b_kn)  # (K//16, N*2) int32

    linear_sf = _unswizzle_sf_128x4(b_descale, n, k_sf)  # (N, K_sf) uint8
    sf_ksf_n = linear_sf.t().contiguous()  # (K_sf, N) uint8 (E4M3)
    sf_ksf_n = _e4m3_to_s0e5m3(sf_ksf_n)  # -> S0E5M3
    return b_packed, sf_ksf_n, alpha


# =============================================================================
# cute-DSL autotuner integration
# =============================================================================


def _w4a16_cute_dsl_tactic_configs(
    n: int, k: int
) -> List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]]:
    """Enumerate cute-DSL tactic configs for a given ``(N, K)``.

    Returns a list of ``(tile_shape_mnk, atom_layout, pipeline_depth,
    use_fp16_mma)`` tuples.
    """
    tile_k = 128 if k % 128 == 0 else 64

    # (tile_M, atom_layout) shapes the kernel is designed/validated for, at the
    # default tile_N=64; a tile_N=128 variant is added below for very large N.
    tile_m_atoms: List[Tuple[int, Tuple[int, int, int]]] = []
    if tile_k == 128:
        tile_m_atoms.append((16, (1, 2, 1)))
    tile_m_atoms.append((32, (2, 2, 1)))
    tile_m_atoms.append((64, (2, 2, 1)))

    configs: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], int, int, int]] = []
    seen = set()

    def add(tile_m, atom, pdepth, fp16, tile_n=64, tk=None, swz=1):
        cfg = (
            (tile_m, tile_n, tile_k if tk is None else tk),
            atom,
            pdepth,
            fp16,
            swz,
        )
        key = (cfg[0], cfg[1], pdepth, fp16, swz)
        if key not in seen:
            seen.add(key)
            configs.append(cfg)

    base_tile_m, base_atom = tile_m_atoms[0]
    add(base_tile_m, base_atom, 1, 1)  # 0: baseline
    add(base_tile_m, base_atom, 0, 1)  # no dequant prefetch (helps short-K)
    for tile_m, atom in tile_m_atoms[1:]:
        add(tile_m, atom, 1, 1)

    # tile_N=128 halves the (m,n)-tile count but needs large wave count.
    if tile_k == 128 and n >= 12288 and n % 128 == 0:
        add(base_tile_m, base_atom, 1, 1, tile_n=128)

    # tile_K=64 has more ab stages, but requires larger problem size.
    if tile_k == 128 and n >= 8192:
        add(base_tile_m, base_atom, 1, 1, tile_n=64, tk=64)

    # tile_M=128 (taller M tile, atom (2,2,1)) -- the large-M *prefill* lever.
    if tile_k == 128:
        add(128, (2, 2, 1), 1, 1)

    # Threadblock swizzle (tile_swizzle=8) -- for large-M prefill.
    if tile_k == 128 and n * k >= 16 * 1024 * 1024:
        add(64, (2, 2, 1), 1, 1, swz=8)
    if tile_k == 128:
        add(128, (2, 2, 1), 1, 1, swz=8)

    # tile_N=128 (with tile_M=64, atom (2,2,1)) -- large shapes.
    if tile_k == 128 and n % 128 == 0 and n >= 4096:
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=8)
        add(64, (2, 2, 1), 1, 1, tile_n=128, swz=1)

    return configs


# Autotuner sweeps M (token count) of the bf16 activation ``a`` and keeps the
# output ``out`` in lockstep.  The packed weight ``b``, its scale, and ``alpha``
# are M-independent and stay fixed during profiling.
#
# inputs layout (see _compute_cute_dsl):
#   [0]=a  [1]=b (packed int32)  [2]=b_sf (uint8)  [3]=alpha
#   [4]=out_dtype  [5]=out  [6]=block_size
_W4A16_CUTE_DSL_TUNING_CONFIG = TuningConfig(
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


def _cute_dsl_w4a16_fp4_runner(enable_pdl: bool = True) -> TunableRunner:
    """Build a ``CuteDslW4a16Fp4Runner`` for the cute-DSL W4A16 GEMM."""

    class CuteDslW4a16Fp4Runner(TunableRunner):
        def get_cache_key_extras(self, inputs: List[torch.Tensor]) -> tuple:
            a, b, _, _, out_dtype, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return (out_dtype, n, k)

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> List[int]:
            _, b, _, _, _, _, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            return list(range(len(_w4a16_cute_dsl_tactic_configs(n, k))))

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic: int = -1,
            do_preparation: bool = False,
            **kwargs,
        ) -> torch.Tensor:
            a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size = inputs
            n = int(b.shape[1]) // 2
            k = int(b.shape[0]) * int(block_size)
            m = int(a.shape[0])
            if tactic < 0:
                # Fallback == pre-autotuner heuristic (M-aware), default knobs.
                tile_shape_mnk, atom_layout = _select_w4a16_tile_shape(m, n, k)
                pipeline_depth, use_fp16_mma, tile_swizzle = 1, 1, 1
            else:
                (
                    tile_shape_mnk,
                    atom_layout,
                    pipeline_depth,
                    use_fp16_mma,
                    tile_swizzle,
                ) = _w4a16_cute_dsl_tactic_configs(n, k)[tactic]
            compiled = _get_cute_dsl_w4a16_gemm(
                tile_shape_mnk,
                a.dtype,
                out_dtype,
                atom_layout,
                pipeline_depth,
                use_fp16_mma,
                enable_pdl=enable_pdl,
                tile_swizzle=tile_swizzle,
            )
            compiled(a, b, b_sf_u8, out, alpha_for_launch)
            return out

    return CuteDslW4a16Fp4Runner()


def _compute_cute_dsl(
    a: torch.Tensor,
    b: torch.Tensor,
    b_descale: torch.Tensor,
    alpha: Optional[torch.Tensor],
    out_dtype: torch.dtype,
    out: Optional[torch.Tensor],
    block_size: int,
    enable_pdl: bool = True,
) -> torch.Tensor:
    """cute-DSL-backend compute: dispatch to the compiled Blackwell kernel.

    ``b`` is the packed ``(K // 16, N * 2)`` int32 weight and
    ``b_descale`` the ``(K // 16, N)`` FP8-E4M3 SF returned by
    :func:`_prepare_cute_dsl`.
    """
    if b.dtype != torch.int32:
        raise TypeError(
            f"cute-dsl backend expects the packed int32 weight from "
            f"prepare_w4a16_fp4_weights(..., backend='cute-dsl'); got {b.dtype}."
        )
    if out_dtype != a.dtype:
        raise NotImplementedError(
            f"cute-dsl backend requires out_dtype == a.dtype (got "
            f"out_dtype={out_dtype}, a.dtype={a.dtype}).  Use the cudnn "
            f"backend for a mismatched output dtype."
        )
    k_tiles = int(b.shape[0])
    n = int(b.shape[1]) // 2
    k = k_tiles * block_size
    m = int(a.shape[0])
    if a.shape[1] != k:
        raise ValueError(
            f"a.shape[1]={a.shape[1]} but k inferred from prepared b.shape="
            f"{tuple(b.shape)} is {k}"
        )

    if out is None:
        out = torch.empty((m, n), device=a.device, dtype=out_dtype)
    else:
        if tuple(out.shape) != (m, n):
            raise ValueError(f"out shape {tuple(out.shape)} != expected {(m, n)}")
        if out.dtype != out_dtype:
            raise TypeError(f"out dtype {out.dtype} != requested out_dtype {out_dtype}")

    b_sf_u8 = b_descale.view(torch.uint8).contiguous()
    alpha_for_launch = _prepare_w4a16_alpha(alpha, a.device)

    tuner = AutoTuner.get()
    runner = _cute_dsl_w4a16_fp4_runner(enable_pdl=enable_pdl)
    inputs = [a, b, b_sf_u8, alpha_for_launch, out_dtype, out, block_size]
    chosen_runner, tactic = tuner.choose_one(
        "w4a16_fp4_cute_dsl_gemm",
        [runner],
        _W4A16_CUTE_DSL_TUNING_CONFIG,
        inputs,
    )
    chosen_runner(inputs=inputs, tactic=tactic)
    return out


__all__ = [
    "prepare_w4a16_fp4_weights",
    "mm_w4a16_fp4",
]
