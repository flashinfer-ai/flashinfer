# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchDecode CuTe DSL wrappers — PyTorch-facing APIs for GQA decode attention.

Two wrappers are provided, mirroring the structure of
``flashinfer/cute_dsl/attention/wrappers/batch_prefill.py``:

* :class:`BatchDecodeCuteDSLWrapper`        — ragged contiguous KV cache,
                                              uses ``gqa_decode.GroupedQueryAttentionDecode``.
* :class:`BatchDecodePagedCuteDSLWrapper`   — paged KV cache,
                                              uses ``gqa_decode_paged.GroupedQueryAttentionDecodePaged``.

Both compile their kernels with symbolic dimensions (``cute.sym_int``) so the
same compiled module can be reused across batches of varying size, and memoize
compilation via :func:`functools.cache` keyed on the static configuration.
"""

import functools
import warnings
from typing import Any, Literal, Optional, Tuple, cast

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from flashinfer.api_logging import flashinfer_api
from flashinfer.cute_dsl.utils import get_num_sm

from ..fusion.mask import AttentionMask, DenseMask, CausalMask, SlidingWindowMask
from ..gqa_decode import GroupedQueryAttentionDecode
from ..gqa_decode_paged import GroupedQueryAttentionDecodePaged


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e5m2: cutlass.Float8E5M2,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}

_RAGGED_DECODE_PLAN_LEGACY_POS_ARGS = (
    "q_data_type",
    "kv_data_type",
    "o_data_type",
    "sm_scale",
    "kv_splits",
    "reduction",
    "q_len_per_req",
    "is_causal",
)

_RAGGED_DECODE_RUN_LEGACY_POS_ARGS = (
    "out",
    "sm_scale",
    "o_scale",
    "lse",
    "enable_pdl",
)

_PAGED_DECODE_PLAN_LEGACY_POS_ARGS = (
    "q_data_type",
    "kv_data_type",
    "o_data_type",
    "sm_scale",
    "kv_splits",
    "reduction",
    "q_len_per_req",
    "is_causal",
    "max_kv_len",
    "non_blocking",
    "precompile_skip_softmax_kernel",
)

_PAGED_DECODE_RUN_LEGACY_POS_ARGS = (
    "out",
    "sm_scale",
    "o_scale",
    "skip_softmax_threshold_scale_factor",
    "lse",
    "enable_pdl",
)


def _merge_deprecated_kwargs(
    api_name: str,
    method_name: str,
    deprecated_positional_args: Tuple[Any, ...],
    legacy_positional_names: Tuple[str, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    if len(deprecated_positional_args) > len(legacy_positional_names):
        raise TypeError(
            f"{api_name}.{method_name}() accepts at most "
            f"{len(legacy_positional_names)} deprecated optional positional "
            "arguments after the required arguments; got "
            f"{len(deprecated_positional_args)}"
        )

    merged_kwargs = dict(kwargs)
    for name, value in zip(
        legacy_positional_names, deprecated_positional_args, strict=False
    ):
        if name in merged_kwargs:
            raise TypeError(
                f"{api_name}.{method_name}() got multiple values for argument {name!r}"
            )
        merged_kwargs[name] = value
    return merged_kwargs


def _merge_deprecated_plan_kwargs(
    api_name: str,
    deprecated_positional_args: Tuple[Any, ...],
    legacy_positional_names: Tuple[str, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return _merge_deprecated_kwargs(
        api_name,
        "plan",
        deprecated_positional_args,
        legacy_positional_names,
        kwargs,
    )


def _merge_deprecated_run_kwargs(
    api_name: str,
    deprecated_positional_args: Tuple[Any, ...],
    legacy_positional_names: Tuple[str, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    return _merge_deprecated_kwargs(
        api_name,
        "run",
        deprecated_positional_args,
        legacy_positional_names,
        kwargs,
    )


def _warn_deprecated_positional_args(api_name: str, method_name: str) -> None:
    warnings.warn(
        f"Passing optional arguments to {api_name}.{method_name}() positionally is "
        "deprecated; pass them as keyword arguments instead. Scheduled for "
        "removal in a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def _warn_deprecated_plan_positional_args(api_name: str) -> None:
    _warn_deprecated_positional_args(api_name, "plan")


def _warn_deprecated_run_positional_args(api_name: str) -> None:
    _warn_deprecated_positional_args(api_name, "run")


def _warn_deprecated_is_causal(api_name: str) -> None:
    warnings.warn(
        f"Passing `is_causal` to {api_name}.plan() is deprecated; use "
        "`window_left` and `window_right` instead. Scheduled for removal in "
        "a future release.",
        DeprecationWarning,
        stacklevel=3,
    )


def _torch_to_cutlass(dtype: torch.dtype) -> "cutlass.dtype":
    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise TypeError(
            f"cute-dsl decode does not support {dtype}; "
            f"supported: {list(_TORCH_TO_CUTLASS_DTYPE)}"
        )
    return _TORCH_TO_CUTLASS_DTYPE[dtype]


def _pick_tile_shape(
    num_qo_heads: int,
    num_kv_heads: int,
    prediction: int,
    qkv_dtype_width: int,
) -> Tuple[int, int]:
    """Pick (grouped_head_tile, prediction_tile) matching ``gqa_decode.run`` logic."""
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be a multiple of num_kv_heads "
            f"({num_kv_heads})"
        )
    npo2 = lambda x: 1 if x <= 1 else 1 << (x - 1).bit_length()
    grouped_heads = num_qo_heads // num_kv_heads
    grouped_head_tile = min(32, npo2(grouped_heads))
    prediction_tile = min(32 // grouped_head_tile, npo2(prediction))
    if grouped_head_tile == prediction_tile == 1 and qkv_dtype_width == 8:
        # Reproduce the special-case bump applied for fp8 with 1×1 tiles.
        grouped_head_tile = 2
    return grouped_head_tile, prediction_tile


def _compute_kv_splits(
    batch_size: int,
    num_kv_heads: int,
    grouped_head_tile: int,
    prediction_tile: int,
    grouped_heads: int,
    prediction: int,
    max_kv_len: int,
    sm_count: int,
    reduction: str,
) -> int:
    """Replicate ``gqa_decode.run`` kv_splits auto-computation."""
    if reduction == "none":
        return 1
    ceil_div = lambda a, b: (a + b - 1) // b
    grouped_head_tiles = ceil_div(grouped_heads, grouped_head_tile)
    prediction_tiles = ceil_div(prediction, prediction_tile)
    grid_z = batch_size * num_kv_heads
    grid_y = grouped_head_tiles * prediction_tiles
    grid_yz = grid_y * grid_z
    sm_count = sm_count if sm_count > 0 else 148
    kv_splits = max(1, sm_count // grid_yz)
    if sm_count == 148 and grid_yz == 32:
        kv_splits = 9  # 2 waves
    kv_splits = min(kv_splits, max(1, ceil_div(max_kv_len, 256)))
    if reduction == "atomic" and kv_splits not in (1, 2, 4):
        kv_splits = 8
    return kv_splits


def _resolve_reduction(reduction: str, kv_splits: int, o_dtype: "cutlass.dtype") -> str:
    if reduction != "auto":
        return reduction
    if kv_splits == 1:
        # No split-K — skip flash-decoding entirely (no reduction kernel,
        # no cluster atomics).
        return "none"
    if o_dtype in (
        cutlass.Float32,
        cutlass.Float16,
        cutlass.BFloat16,
    ) and kv_splits in (
        2,
        4,
        8,
    ):
        return "atomic"
    return "kernel"


def _get_mask_config(
    window_left: Optional[int],
    window_right: Optional[int],
) -> AttentionMask:
    if window_left is None and window_right is None:
        return DenseMask()
    elif window_left is None and window_right == 0:
        return CausalMask()
    else:
        # Runtime sliding window for int types
        window_left_ = Int32(window_left) if window_left is not None else None
        window_right_ = Int32(window_right) if window_right is not None else None
        return SlidingWindowMask(window_left_, window_right_)


def _slice_workspace(
    workspace: torch.Tensor,
    kv_splits: int,
    batch_size: int,
    prediction: int,
    num_qo_heads: int,
    head_dim: int,
):
    """Slice pre-allocated workspace into partial result workspace"""
    o_partial_shape = torch.Size(
        (kv_splits, batch_size, prediction, num_qo_heads, head_dim)
    )
    m_partial_shape = l_partial_shape = o_partial_shape[:-1]
    m_shape = o_partial_shape[1:-1]

    workspace = workspace.view(dtype=torch.float32)

    required_elts = (
        o_partial_shape.numel()
        + m_partial_shape.numel()
        + l_partial_shape.numel()
        + m_shape.numel()
    )
    if workspace.numel() < required_elts:
        raise RuntimeError(
            f"kernel reduction with kv_splits={kv_splits} "
            f"batch_size={batch_size} q_len_per_req={prediction} "
            f"num_qo_heads={num_qo_heads} head_dim={head_dim} "
            f"requires {required_elts * 4} byte workspace "
            f"which exceeds provided workspace of {workspace.nbytes} bytes"
        )

    start, end = 0, o_partial_shape.numel()
    o_partial = workspace[start:end].view(o_partial_shape)

    start, end = end, end + l_partial_shape.numel()
    l_partial = workspace[start:end].view(l_partial_shape)

    start, end = end, end + m_partial_shape.numel()
    m_partial = workspace[start:end].view(m_partial_shape)

    start, end = end, end + m_shape.numel()
    m = workspace[start:end].view(m_shape)

    return m, o_partial, l_partial, m_partial


# ---------------------------------------------------------------------------
# Compile helpers (memoized)
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_decode_kernel(
    in_dtype: "cutlass.dtype",
    out_dtype: "cutlass.dtype",
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prediction: int,
    reduction: str,
    window_left: Optional[int],
    window_right: Optional[int],
    use_lse: bool,
    use_sink: bool,
):
    """Compile and cache the ragged (contiguous-KV) GQA decode kernel."""
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads, num_kv_heads, prediction, in_dtype.width
    )
    blk_tile_n = grouped_head_tile * prediction_tile

    sequence_tile = 256
    if prediction_tile > 1 and blk_tile_n > 8:
        sequence_tile = 128  # Prevent regspills

    fmha = GroupedQueryAttentionDecode(
        head_dim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=sequence_tile,
        reduction_mode=cast(Literal["kernel", "atomic", "none"], reduction),
    )
    mask_config = _get_mask_config(window_left, window_right)
    has_workspace = reduction == "kernel"
    acc_dtype = cutlass.Float32

    sym_kv_splits = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_seq_k = cute.sym_int()
    # Symbolic prediction so a single compiled kernel handles varying runtime
    # q_len_per_req. `prediction` (the Python int) still drives the compile-
    # time prediction_tile / softmax_warpgroups choices via the kernel ctor.
    sym_prediction = cute.sym_int()

    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_batch, sym_prediction, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    # Fully-symbolic outer strides so the same kernel handles both NHD layout
    # (kernel-native) and HND layout (presented to the kernel via transpose to
    # a logical NHD view, which has the same shape but different strides).
    k_fake = cute.runtime.make_fake_tensor(
        in_dtype,
        (sym_batch, sym_seq_k, num_kv_heads, head_dim),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        in_dtype,
        (sym_batch, sym_seq_k, num_kv_heads, head_dim),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        (sym_batch, sym_prediction, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )

    l_fake = None
    if use_lse:
        l_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_batch, sym_prediction, num_qo_heads),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )

    if not has_workspace:
        # atomic / none reduction: no per-split workspace tensors.
        m_fake = o_partial_fake = m_partial_fake = l_partial_fake = None
    else:
        m_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_batch, sym_prediction, num_qo_heads),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        o_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    sink_fake = None
    if use_sink:
        sink_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype, (num_qo_heads,), assumed_align=16
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        Int32(1),  # kv_splits placeholder
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        l_fake,
        m_fake,
        o_partial_fake,
        l_partial_fake,
        m_partial_fake,
        sink_fake,
        mask_config,
        Float32(1.0),  # scale_s placeholder
        Float32(1.0),  # scale_o placeholder
        stream_fake,
        True,  # enable_pdl placeholder (runtime-dynamic)
        options="--enable-tvm-ffi --opt-level 3",
    )


@functools.cache
def _get_compiled_paged_decode_kernel(
    in_dtype: "cutlass.dtype",
    out_dtype: "cutlass.dtype",
    page_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    prediction: int,
    reduction: str,
    window_left: Optional[int],
    window_right: Optional[int],
    use_threshold: bool,
    use_lse: bool,
    use_sink: bool,
):
    """Compile and cache the paged GQA decode kernel."""
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads, num_kv_heads, prediction, in_dtype.width
    )
    blk_tile_n = grouped_head_tile * prediction_tile

    sequence_tile = 256
    if use_threshold:
        sequence_tile = 128  # Promote skipping
    elif prediction_tile > 1 and blk_tile_n > 8:
        sequence_tile = 128  # Prevent regspills

    softmax_warpgroups = (
        2
        if (not use_threshold)
        and (
            (in_dtype.width <= 8 and blk_tile_n > 8)
            or (in_dtype.width == 16 and blk_tile_n > 16)
        )
        else 1
    )
    fmha = GroupedQueryAttentionDecodePaged(
        page_size,
        head_dim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=sequence_tile,
        reduction_mode=cast(Literal["kernel", "atomic", "none"], reduction),
        softmax_warpgroups=softmax_warpgroups,
    )
    mask_config = _get_mask_config(window_left, window_right)
    has_workspace = reduction == "kernel"
    acc_dtype = cutlass.Float32

    sym_kv_splits = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_virtual_pages = cute.sym_int()
    sym_total_logical_pages = cute.sym_int()
    # Symbolic prediction so one compiled kernel handles any runtime
    # q_len_per_req. `prediction` (Python int) still drives the compile-time
    # prediction_tile / softmax_warpgroups choices via the kernel ctor.
    sym_prediction = cute.sym_int()

    seqlens_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=16
    )
    table_offsets_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=16
    )
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_total_logical_pages,), assumed_align=16
    )
    # Fully-symbolic outer strides so the same kernel handles:
    #   * NHD-contiguous   [num_pages,    page_size,    num_kv_heads, head_dim]
    #   * NHD post-unbind  (leading-dim stride 2× from the combined kv tensor)
    #   * HND-transposed   (kernel sees logical NHD, gmem strides reflect the
    #                       underlying HND layout via .transpose(-3, -2))
    # Only the innermost (head_dim) stride is fixed at 1, since the kernel's
    # TMA expects head_dim to be contiguous.
    k_fake = cute.runtime.make_fake_tensor(
        in_dtype,
        (sym_virtual_pages, page_size, num_kv_heads, head_dim),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        in_dtype,
        (sym_virtual_pages, page_size, num_kv_heads, head_dim),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        in_dtype,
        (sym_batch, sym_prediction, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        (sym_batch, sym_prediction, num_qo_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )

    l_fake = None
    if use_lse:
        l_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_batch, sym_prediction, num_qo_heads),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )

    if not has_workspace:
        # atomic / none reduction: no per-split workspace tensors.
        m_fake = o_partial_fake = m_partial_fake = l_partial_fake = None
    else:
        m_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_batch, sym_prediction, num_qo_heads),
            stride_order=(2, 1, 0),
            assumed_align=16,
        )
        o_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (sym_kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    sink_fake = None
    if use_sink:
        sink_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype, (num_qo_heads,), assumed_align=16
        )

    threshold_p_fake = Float32(0.0) if use_threshold else None
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        Int32(1),  # kv_splits placeholder
        seqlens_fake,
        table_offsets_fake,
        page_table_fake,
        k_fake,
        v_fake,
        q_fake,
        o_fake,
        l_fake,
        m_fake,
        o_partial_fake,
        l_partial_fake,
        m_partial_fake,
        sink_fake,
        mask_config,
        Float32(1.0),  # scale_s placeholder
        Float32(1.0),  # scale_o placeholder
        threshold_p_fake,
        stream_fake,
        True,  # enable_pdl placeholder (runtime-dynamic)
        options="--enable-tvm-ffi --opt-level 3",
    )


# ---------------------------------------------------------------------------
# Ragged-KV decode wrapper
# ---------------------------------------------------------------------------


class BatchDecodeCuteDSLWrapper:
    """PyTorch-facing wrapper for the ragged-KV CuTe DSL GQA decode kernel.

    Assumes a contiguous (non-paged) KV cache where all batches have the same
    KV sequence length.  For paged KV with varying sequence lengths use
    :class:`BatchDecodePagedCuteDSLWrapper` instead.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        r"""Construct a ragged-KV CuTe DSL decode wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            Pre-allocated float32 workspace buffer used by the underlying
            CuTe DSL kernel for split-K partial reductions. The wrapper
            does not resize this buffer; the caller is responsible for
            sizing it for the largest expected batch (see :meth:`plan`).
            The buffer's device determines the device of subsequent
            kernel launches.
        use_cuda_graph : bool
            If ``True``, prepare the wrapper for capture in a CUDA graph
            so that subsequent :meth:`run` calls are graph-safe (no host
            sync, stable workspace pointers). Defaults to ``False``.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._use_cuda_graph = use_cuda_graph
        self._compiled_fmha_std = None
        self._compile_args: Optional[tuple] = None
        self._planned = False

    @flashinfer_api
    def plan(
        self,
        batch_size: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *deprecated_positional_args: Any,
        **kwargs: Any,
    ) -> None:
        """Compile the ragged-KV decode kernel for the planned configuration.

        Parameters
        ----------
        batch_size : int
            Representative batch size used to auto-tune ``kv_splits``.  Runtime
            batches may differ.
        max_kv_len : int
            Representative KV sequence length used for ``kv_splits`` tuning.
        num_qo_heads, num_kv_heads, head_dim : int
            GQA configuration.  ``num_qo_heads`` must be a multiple of
            ``num_kv_heads`` and ``head_dim`` must be a multiple of 64.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            Q/K/V/O dtypes. Q and KV must match. ``o_data_type`` defaults to
            ``q_data_type`` (or float16 for fp8 inputs).
        sm_scale : Optional[float]
            Softmax scale; defaults to ``1 / sqrt(head_dim)``.
        kv_splits : Optional[int]
            Threadblocks per sequence (flash decoding).  ``None`` auto-tunes
            from the planned shape and the device SM count.
        reduction : str
            ``"kernel"``, ``"atomic"``, ``"none"``, or ``"auto"`` (default).
            ``"none"`` skips flash-decoding entirely (no reduction kernel,
            no cluster atomics) and requires ``kv_splits == 1``. Atomic
            reduction is faster than kernel reduction but requires kv_splits
            in {1, 2, 4, 8, 16} and an output dtype in {float32, float16,
            bfloat16}. ``"auto"`` picks ``"none"`` when kv_splits == 1,
            ``"atomic"`` for compatible dtypes and small kv_splits, else
            ``"kernel"``.
        q_len_per_req : int
            Predicted tokens per request (1 for plain decode, >1 for
            speculative decode).
        window_left : int
            Sliding-window left bound. ``None`` disables left bound.
        window_right : int
            Sliding-window right bound. ``None`` disables right bound.

        Note
        ----
        Optional arguments after ``head_dim`` are accepted positionally for
        backward compatibility, but that calling convention is deprecated and
        scheduled for removal in a future release. Pass them by keyword instead.
        ``window_left`` and ``window_right`` are keyword-only. The legacy
        ``is_causal`` argument is deprecated; use ``window_left`` and
        ``window_right`` instead.
        """
        if not deprecated_positional_args and "is_causal" not in kwargs:
            return self._plan_impl(
                batch_size,
                max_kv_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                **kwargs,
            )

        plan_kwargs = _merge_deprecated_plan_kwargs(
            "BatchDecodeCuteDSLWrapper",
            deprecated_positional_args,
            _RAGGED_DECODE_PLAN_LEGACY_POS_ARGS,
            kwargs,
        )
        is_causal_was_provided = "is_causal" in plan_kwargs
        if is_causal_was_provided:
            window_args = sorted(
                {"window_left", "window_right"}.intersection(plan_kwargs)
            )
            if window_args:
                raise TypeError(
                    "BatchDecodeCuteDSLWrapper.plan() got deprecated `is_causal` "
                    "together with explicit window arguments "
                    f"{window_args!r}; use `window_left` and `window_right` only"
                )
            is_causal = plan_kwargs.pop("is_causal")
            plan_kwargs["window_left"] = None
            plan_kwargs["window_right"] = 0 if is_causal else None

        if deprecated_positional_args:
            _warn_deprecated_plan_positional_args("BatchDecodeCuteDSLWrapper")
        if is_causal_was_provided:
            _warn_deprecated_is_causal("BatchDecodeCuteDSLWrapper")

        return self._plan_impl(
            batch_size,
            max_kv_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            **plan_kwargs,
        )

    def _plan_impl(
        self,
        batch_size: int,
        max_kv_len: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        *,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
        q_len_per_req: int = 1,
        window_left: Optional[int] = None,
        window_right: Optional[int] = 0,
    ) -> None:
        if kv_data_type is not None and kv_data_type != q_data_type:
            raise NotImplementedError(
                "cute-dsl decode requires kv_data_type == q_data_type"
            )
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        if head_dim <= 0 or head_dim % 64 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be a positive multiple of 64")

        in_dtype = _torch_to_cutlass(q_data_type)
        if o_data_type is None:
            o_data_type = q_data_type
        out_dtype = _torch_to_cutlass(o_data_type)

        grouped_head_tile, prediction_tile = _pick_tile_shape(
            num_qo_heads, num_kv_heads, q_len_per_req, in_dtype.width
        )
        if kv_splits is None or kv_splits <= 0:
            kv_splits = _compute_kv_splits(
                batch_size,
                num_kv_heads,
                grouped_head_tile,
                prediction_tile,
                num_qo_heads // num_kv_heads,
                q_len_per_req,
                max_kv_len,
                get_num_sm(self.device),
                reduction,
            )
        reduction = _resolve_reduction(reduction, kv_splits, out_dtype)
        if reduction == "atomic":
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits ∈ {{1,2,4,8,16}}, got {kv_splits}"
                )
            if out_dtype not in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
                raise ValueError(
                    f"atomic reduction requires output dtype in (float32, float16, "
                    f"bfloat16); got {o_data_type}"
                )
        elif reduction == "none" and kv_splits != 1:
            raise ValueError(
                f'reduction="none" requires kv_splits == 1, got {kv_splits}'
            )

        self._q_data_type = q_data_type
        self._o_data_type = o_data_type
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._batch_size = batch_size
        self._q_len_per_req = q_len_per_req
        self._kv_splits = kv_splits
        self._reduction = reduction
        # Only kernel-reduction uses workspace tensors; atomic and none
        # write to o_bshd directly (atomic via atomic_add, none via direct
        # store).
        self._has_workspace = reduction == "kernel"
        self._mask_config = _get_mask_config(window_left, window_right)
        if sm_scale is None:
            sm_scale = head_dim**-0.5
        self._sm_scale = sm_scale

        # Carve out workspace tensors
        if self._has_workspace:
            m, o_partial, l_partial, m_partial = _slice_workspace(
                self._float_workspace_buffer,
                kv_splits,
                batch_size,
                q_len_per_req,
                num_qo_heads,
                head_dim,
            )
            self._m = m
            self._o_partial = o_partial
            self._l_partial = l_partial
            self._m_partial = m_partial

        # Always compile the no-LSE/no-sink variant at plan() time; optional
        # feature variants are lazily fetched at run() time when needed.
        self._compile_args = (
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_len_per_req,
            reduction,
            window_left,
            window_right,
        )
        self._compiled_fmha_std = _get_compiled_decode_kernel(
            *self._compile_args, False, False
        )
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *deprecated_positional_args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run ragged-KV GQA decode.

        Parameters
        ----------
        q : torch.Tensor
            Shape ``[batch_size, q_len_per_req, num_qo_heads, head_dim]``.
            ``q_len_per_req`` is read from ``q.shape[1]`` at run time; it
            does not have to match the value passed to :meth:`plan` (which
            is only a compile-time tile-size hint).
        k, v : torch.Tensor
            Shape ``[batch_size, seq_len, num_kv_heads, head_dim]``.  Both
            must have the same seq_len.
        out : Optional[torch.Tensor]
            Pre-allocated output buffer.  For atomic reduction it must be
            zero-initialized before being passed in.
        sm_scale : Optional[float]
            Per-call override of the softmax scale set at plan() time.
        o_scale : Optional[float]
            Output scale applied to the final O before it is written. The
            cute-dsl kernel folds this in for free in the reduction
            epilogue (no separate post-kernel multiply). Defaults to 1.0.
        sinks : Optional[torch.Tensor]
            Contiguous float32 per-head attention sink logits on the query
            device, shape ``(num_qo_heads,)``. When provided, the sink logit
            is included in the softmax denominator and receives no output
            value contribution.
        lse : Optional[torch.Tensor]
            Pre-allocated float32 buffer of shape
            ``(batch_size, q_len_per_req, num_qo_heads)`` to receive the
            log-sum-exp (log2 base, matching flashinfer convention). When
            ``None`` (default) the kernel skips the LSE write entirely;
            otherwise a log2-base LSE variant is lazily compiled on first
            use (cache hit afterwards).
        enable_pdl : bool
            Whether to launch with Programmatic Dependent Launch (PDL).
            Default ``True``. Set to ``False`` to disable PDL when the
            target device does not support it. See
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization

        Note
        ----
        Optional arguments after ``v`` are accepted positionally for backward
        compatibility, but that calling convention is deprecated and scheduled
        for removal in a future release. Pass them by keyword instead.
        ``sinks`` is keyword-only.
        """
        if not deprecated_positional_args:
            return self._run_impl(q, k, v, **kwargs)

        run_kwargs = _merge_deprecated_run_kwargs(
            "BatchDecodeCuteDSLWrapper",
            deprecated_positional_args,
            _RAGGED_DECODE_RUN_LEGACY_POS_ARGS,
            kwargs,
        )
        _warn_deprecated_run_positional_args("BatchDecodeCuteDSLWrapper")
        return self._run_impl(q, k, v, **run_kwargs)

    def _run_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
    ) -> torch.Tensor:
        if not self._planned:
            raise RuntimeError("Call plan() before run().")
        if (
            q.dtype != self._q_data_type
            or k.dtype != self._q_data_type
            or v.dtype != self._q_data_type
        ):
            raise ValueError(
                f"q/k/v dtype mismatch: expected {self._q_data_type}, got "
                f"q={q.dtype}, k={k.dtype}, v={v.dtype}"
            )
        if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
            raise ValueError(
                f"q/k/v must be 4D [b, s, h, d]; got shapes {tuple(q.shape)}, "
                f"{tuple(k.shape)}, {tuple(v.shape)}"
            )
        b, s_q, h_q, d = q.shape
        b_k, s_k, h_k, d_k = k.shape
        if (
            b != b_k
            or h_q != self._num_qo_heads
            or h_k != self._num_kv_heads
            or d != d_k
            or d != self._head_dim
        ):
            raise ValueError(
                f"q/k/v shape mismatch with plan: q={tuple(q.shape)}, k={tuple(k.shape)}, "
                f"v={tuple(v.shape)}; expected q=[b,*,{self._num_qo_heads},"
                f"{self._head_dim}], k=[b,*,{self._num_kv_heads},"
                f"{self._head_dim}]"
            )
        if v.shape != k.shape:
            raise ValueError(
                f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}"
            )

        device = q.device
        # Atomic reduction accumulates into out via atomic_add and requires
        # zero init; kernel/none modes overwrite via direct store.
        atomic = self._reduction == "atomic"
        if out is None:
            torch_alloc = torch.zeros if atomic else torch.empty
            out = torch_alloc((b, s_q, h_q, d), dtype=self._o_data_type, device=device)

        m = o_partial = l_partial = m_partial = None
        if self._has_workspace:
            # Kernel expects partial + batch stride to be coalescible
            # So we must reslice workspace if runtime batch doesn't match
            if b == self._batch_size and s_q <= self._q_len_per_req:
                # Runtime q_len doesnt exceed plantime q_len, return slice
                m = self._m[:, :s_q, :]
                o_partial = self._o_partial[:, :, :s_q, :, :]
                l_partial = self._l_partial[:, :, :s_q, :]
                m_partial = self._m_partial[:, :, :s_q, :]
            else:
                # Re-validate workspace size and slice
                m, o_partial, l_partial, m_partial = _slice_workspace(
                    self._float_workspace_buffer,
                    self._kv_splits,
                    b,
                    s_q,
                    self._num_qo_heads,
                    self._head_dim,
                )
            m.fill_(float("-inf"))

        scale_s = self._sm_scale if sm_scale is None else sm_scale
        use_lse = lse is not None
        use_sink = sinks is not None
        if use_lse:
            if lse.dtype != torch.float32:
                raise ValueError(
                    f"lse must be float32 (LSE is log2-base); got {lse.dtype}"
                )
            expected_lse_shape = (b, s_q, h_q)
            if lse.shape != expected_lse_shape:
                raise ValueError(
                    f"lse shape {tuple(lse.shape)} must equal {expected_lse_shape}"
                )

        if use_sink:
            if sinks.ndim != 1 or sinks.shape[0] != self._num_qo_heads:
                raise ValueError(
                    f"sinks tensor must have shape (num_qo_heads,) = "
                    f"({self._num_qo_heads},), got shape {tuple(sinks.shape)}"
                )
            if sinks.dtype != torch.float32:
                raise ValueError(f"sinks must be float32, got {sinks.dtype}")
            if sinks.device != device:
                raise ValueError(
                    f"sinks must be on device {device}, got {sinks.device}"
                )
            if not sinks.is_contiguous():
                raise ValueError(
                    f"sinks tensor must be contiguous, got strides {sinks.stride()} "
                    f"for shape {sinks.shape}"
                )

        if use_lse or use_sink:
            fmha = _get_compiled_decode_kernel(*self._compile_args, use_lse, use_sink)
        else:
            fmha = self._compiled_fmha_std

        scale_o = 1.0 if o_scale is None else o_scale
        # Stream is bound from the TVM-FFI env stream at runtime
        # (use_tvm_ffi_env_stream=True at compile), so the stream arg is omitted.
        fmha(
            self._kv_splits,
            q,
            k,
            v,
            out,
            lse,
            m,
            o_partial,
            l_partial,
            m_partial,
            sinks,
            self._mask_config,
            Float32(scale_s),
            Float32(scale_o),
            enable_pdl,
        )
        return out


# ---------------------------------------------------------------------------
# Paged-KV decode wrapper
# ---------------------------------------------------------------------------


class BatchDecodePagedCuteDSLWrapper:
    """PyTorch-facing wrapper for the paged CuTe DSL GQA decode kernel."""

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
        r"""Construct a paged-KV CuTe DSL decode wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            Pre-allocated float32 workspace buffer used by the underlying
            CuTe DSL kernel for split-K partial reductions. The wrapper
            does not resize this buffer; the caller is responsible for
            sizing it for the largest expected batch and page table (see
            :meth:`plan`). The buffer's device determines the device of
            subsequent kernel launches.
        use_cuda_graph : bool
            If ``True``, prepare the wrapper for capture in a CUDA graph
            so that subsequent :meth:`run` calls are graph-safe (no host
            sync, stable workspace pointers). Defaults to ``False``.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._use_cuda_graph = use_cuda_graph
        self._compiled_fmha_std = None
        self._compile_args: Optional[tuple] = None
        self._planned = False

    @flashinfer_api
    def plan(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *deprecated_positional_args: Any,
        **kwargs: Any,
    ) -> None:
        """Plan paged GQA decode for the given problem.

        Parameters
        ----------
        indptr : torch.Tensor (int32, [batch_size + 1])
            Prefix-sum offsets into ``indices``.
        indices : torch.Tensor (int32, [num_pages_total])
            Flat per-sequence virtual page indices.
        seq_lens : torch.Tensor (int32, [batch_size])
            Per-sequence KV length in tokens. The kernel reads this
            directly; callers that have ``last_page_len`` instead should
            use :func:`flashinfer.page.get_seq_lens` to convert.
        num_qo_heads, num_kv_heads, head_dim, page_size : int
            GQA + paging configuration.  ``page_size`` must be in
            ``{8, 16, 32, 64}`` and ``head_dim`` a positive multiple of 64.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            Q/K/V/O dtypes; ``kv_data_type`` must equal ``q_data_type``.
        sm_scale : Optional[float]
            Softmax scale; defaults to ``1 / sqrt(head_dim)``.
        kv_splits : Optional[int]
            Threadblocks per sequence (flash decoding).  ``None`` auto-tunes
            from the planned shapes and SM count.
        reduction : str
            ``"kernel"`` (deterministic with workspace), ``"atomic"``
            (cluster reduction, faster but lower precision), ``"none"``
            (no flash-decoding split-K; requires kv_splits == 1), or
            ``"auto"`` (picks ``"none"`` when kv_splits == 1, else atomic
            for compatible dtypes, else kernel).
        q_len_per_req : int
            Predicted tokens per request (1 for plain decode).
        window_left : int
            Sliding-window left bound. ``None`` disables left bound.
        window_right : int
            Sliding-window right bound. ``None`` disables the right bound.
            Defaults to ``0``.
        max_kv_len : Optional[int]
            Maximum KV sequence length across the batch.  Used to auto-tune
            ``kv_splits``; pass it explicitly to avoid a GPU->CPU sync.
        non_blocking : bool
            Async device copies for the plan-time integer buffers.
        precompile_skip_softmax_kernel : bool
            If True, also compile the BLASST skip-softmax variant of the
            kernel at plan() time, so the first :meth:`run` call that
            passes ``skip_softmax_threshold_scale_factor`` is fast.

        Note
        ----
        Optional arguments after ``page_size`` are accepted positionally for
        backward compatibility, but that calling convention is deprecated and
        scheduled for removal in a future release. Pass them by keyword instead.
        ``window_left`` and ``window_right`` are keyword-only. The legacy
        ``is_causal`` argument is deprecated; use ``window_left`` and
        ``window_right`` instead.
        """
        if not deprecated_positional_args and "is_causal" not in kwargs:
            return self._plan_impl(
                indptr,
                indices,
                seq_lens,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                page_size,
                **kwargs,
            )

        plan_kwargs = _merge_deprecated_plan_kwargs(
            "BatchDecodePagedCuteDSLWrapper",
            deprecated_positional_args,
            _PAGED_DECODE_PLAN_LEGACY_POS_ARGS,
            kwargs,
        )
        is_causal_was_provided = "is_causal" in plan_kwargs
        if is_causal_was_provided:
            window_args = sorted(
                {"window_left", "window_right"}.intersection(plan_kwargs)
            )
            if window_args:
                raise TypeError(
                    "BatchDecodePagedCuteDSLWrapper.plan() got deprecated "
                    "`is_causal` together with explicit window arguments "
                    f"{window_args!r}; use `window_left` and `window_right` only"
                )
            is_causal = plan_kwargs.pop("is_causal")
            plan_kwargs["window_left"] = None
            plan_kwargs["window_right"] = 0 if is_causal else None

        if deprecated_positional_args:
            _warn_deprecated_plan_positional_args("BatchDecodePagedCuteDSLWrapper")
        if is_causal_was_provided:
            _warn_deprecated_is_causal("BatchDecodePagedCuteDSLWrapper")

        return self._plan_impl(
            indptr,
            indices,
            seq_lens,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            **plan_kwargs,
        )

    def _plan_impl(
        self,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        *,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
        q_len_per_req: int = 1,
        window_left: Optional[int] = None,
        window_right: Optional[int] = 0,
        max_kv_len: Optional[int] = None,
        non_blocking: bool = True,
        precompile_skip_softmax_kernel: bool = False,
    ) -> None:
        if page_size not in (8, 16, 32, 64):
            raise ValueError(
                f"cute-dsl paged decode supports page_size ∈ {{8,16,32,64}}, got {page_size}"
            )
        if num_qo_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_qo_heads ({num_qo_heads}) must be a multiple of "
                f"num_kv_heads ({num_kv_heads})"
            )
        if head_dim <= 0 or head_dim % 64 != 0:
            raise ValueError(f"head_dim ({head_dim}) must be a positive multiple of 64")
        if kv_data_type is not None and kv_data_type != q_data_type:
            raise NotImplementedError(
                "cute-dsl paged decode requires kv_data_type == q_data_type"
            )

        in_dtype = _torch_to_cutlass(q_data_type)
        if o_data_type is None:
            o_data_type = q_data_type
        out_dtype = _torch_to_cutlass(o_data_type)

        batch_size = seq_lens.numel()
        if indptr.numel() < batch_size:
            raise ValueError(
                f"indptr length ({indptr.numel()}) must be at least batch_size "
                f"({batch_size})"
            )

        if max_kv_len is None:
            if not seq_lens.is_cuda:
                max_kv_len = int(seq_lens.max().item())
            else:
                # Avoid a device sync, just assume largest possible sequence
                max_kv_len = (2**31) - 1

        def to_int32_device(tensor: torch.Tensor):
            return tensor.to(
                dtype=torch.int32,
                device=self.device,
                non_blocking=non_blocking,
            )

        seq_lens = to_int32_device(seq_lens)
        indices = to_int32_device(indices)
        indptr = to_int32_device(indptr)

        grouped_head_tile, prediction_tile = _pick_tile_shape(
            num_qo_heads, num_kv_heads, q_len_per_req, in_dtype.width
        )
        if kv_splits is None or kv_splits <= 0:
            kv_splits = _compute_kv_splits(
                batch_size,
                num_kv_heads,
                grouped_head_tile,
                prediction_tile,
                num_qo_heads // num_kv_heads,
                q_len_per_req,
                max_kv_len,
                get_num_sm(self.device),
                reduction,
            )
        reduction = _resolve_reduction(reduction, kv_splits, out_dtype)
        if reduction == "atomic":
            if kv_splits not in (1, 2, 4, 8, 16):
                raise ValueError(
                    f"atomic reduction requires kv_splits ∈ {{1,2,4,8,16}}, got {kv_splits}"
                )
            if out_dtype not in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16):
                raise ValueError(
                    f"atomic reduction requires output dtype in (float32, float16, "
                    f"bfloat16); got {o_data_type}"
                )
        elif reduction == "none" and kv_splits != 1:
            raise ValueError(
                f'reduction="none" requires kv_splits == 1, got {kv_splits}'
            )
        self._q_data_type = q_data_type
        self._o_data_type = o_data_type
        self._num_qo_heads = num_qo_heads
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._page_size = page_size
        self._q_len_per_req = q_len_per_req
        self._kv_splits = kv_splits
        self._reduction = reduction
        # Only kernel-reduction uses workspace tensors; atomic and none
        # write to o_bshd directly.
        self._has_workspace = reduction == "kernel"
        self._mask_config = _get_mask_config(window_left, window_right)
        if sm_scale is None:
            sm_scale = head_dim**-0.5
        self._sm_scale = sm_scale

        self._batch_size = batch_size
        self._seqlens = seq_lens
        self._table_offsets = indptr[:batch_size]
        self._page_table = indices

        # Carve out workspace tensors
        if self._has_workspace:
            m, o_partial, l_partial, m_partial = _slice_workspace(
                self._float_workspace_buffer,
                kv_splits,
                batch_size,
                q_len_per_req,
                num_qo_heads,
                head_dim,
            )
            self._m = m
            self._o_partial = o_partial
            self._l_partial = l_partial
            self._m_partial = m_partial

        # Compile args shared by std and BLASST variants. Stashed so run()
        # can lazily fetch the BLASST variant via the @functools.cache.
        self._compile_args = (
            in_dtype,
            out_dtype,
            page_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_len_per_req,
            reduction,
            window_left,
            window_right,
        )
        # Standard (no-BLASST, no-LSE) variant — always compiled at plan().
        # BLASST and LSE variants compile lazily on first use via the cache.
        self._compiled_fmha_std = _get_compiled_paged_decode_kernel(
            *self._compile_args, False, False, False
        )
        if precompile_skip_softmax_kernel:
            # Warm the cache so the BLASST variant is ready for the first
            # run() that uses skip_softmax_threshold.
            _get_compiled_paged_decode_kernel(*self._compile_args, True, False, False)
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *deprecated_positional_args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Run paged GQA decode.

        Parameters
        ----------
        q : torch.Tensor
            ``[batch_size * q_len_per_req, num_qo_heads, head_dim]`` or
            ``[batch_size, q_len_per_req, num_qo_heads, head_dim]``.
            ``q_len_per_req`` is read from ``q.shape`` at run time; it does
            not have to match the value passed to :meth:`plan` (which is
            only a compile-time tile-size hint).
        k_cache, v_cache : torch.Tensor
            Logical shape ``[num_pages, page_size, num_kv_heads, head_dim]``.
            Both NHD-contiguous layouts and HND layouts (presented as a
            transposed view) are accepted; the kernel handles arbitrary
            strides as long as ``head_dim`` is innermost.
        out : Optional[torch.Tensor]
            Pre-allocated output buffer.  For atomic reduction it must be
            zero-initialized before being passed in.
        sm_scale : Optional[float]
            Per-call override of the softmax scale set at plan() time.
        o_scale : Optional[float]
            Output scale applied to the final O before it is written. The
            cute-dsl kernel folds this in for free in the reduction
            epilogue (no separate post-kernel multiply). Defaults to 1.0.
        sinks : Optional[torch.Tensor]
            Contiguous float32 per-head attention sink logits on the query
            device, shape ``(num_qo_heads,)``. When provided, the sink logit
            is included in the softmax denominator and receives no output
            value contribution.
        skip_softmax_threshold_scale_factor : Optional[float]
            BLASST skip-softmax scale factor. The kernel divides this by
            each batch's KV seqlen to obtain the per-request effective
            threshold. Must be > 0 when set. ``None`` (default) dispatches
            to the standard kernel; a value triggers lazy compile of the
            BLASST variant on first use, or hits the precompiled cache if
            ``plan(precompile_skip_softmax_kernel=True)`` was used.
        lse : Optional[torch.Tensor]
            Pre-allocated float32 buffer of shape
            ``(batch_size, q_len_per_req, num_qo_heads)`` (or the flat
            equivalent ``(batch_size * q_len_per_req, num_qo_heads)``) to
            receive the log-sum-exp (log2 base, matching flashinfer
            convention). When ``None`` (default) the kernel skips the LSE
            write; otherwise an LSE variant is lazily compiled on first
            use.
        enable_pdl : bool
            Whether to launch with Programmatic Dependent Launch (PDL).
            Default ``True``. Set to ``False`` to disable PDL when the
            target device does not support it. See
            https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization

        Note
        ----
        Optional arguments after ``v_cache`` are accepted positionally for
        backward compatibility, but that calling convention is deprecated and
        scheduled for removal in a future release. Pass them by keyword instead.
        ``sinks`` is keyword-only.
        """
        if not deprecated_positional_args:
            return self._run_impl(q, k_cache, v_cache, **kwargs)

        run_kwargs = _merge_deprecated_run_kwargs(
            "BatchDecodePagedCuteDSLWrapper",
            deprecated_positional_args,
            _PAGED_DECODE_RUN_LEGACY_POS_ARGS,
            kwargs,
        )
        _warn_deprecated_run_positional_args("BatchDecodePagedCuteDSLWrapper")
        return self._run_impl(q, k_cache, v_cache, **run_kwargs)

    def _run_impl(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        lse: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
    ) -> torch.Tensor:
        if not self._planned:
            raise RuntimeError("Call plan() before run().")
        if q.dtype != self._q_data_type:
            raise ValueError(
                f"q.dtype={q.dtype} mismatches planned {self._q_data_type}"
            )
        if k_cache.dtype != self._q_data_type or v_cache.dtype != self._q_data_type:
            raise ValueError(
                f"k_cache/v_cache dtype must equal q.dtype={self._q_data_type}"
            )
        if k_cache.shape != v_cache.shape:
            raise ValueError(
                f"k_cache.shape={tuple(k_cache.shape)} must equal "
                f"v_cache.shape={tuple(v_cache.shape)}"
            )
        if k_cache.dim() != 4:
            raise ValueError(
                f"k_cache must be 4D [num_pages, page_size, num_kv_heads, head_dim]; "
                f"got {tuple(k_cache.shape)}"
            )
        if (
            k_cache.shape[1] != self._page_size
            or k_cache.shape[2] != self._num_kv_heads
            or k_cache.shape[3] != self._head_dim
        ):
            raise ValueError(
                f"k_cache shape {tuple(k_cache.shape)} mismatches plan: "
                f"expected [*, {self._page_size}, {self._num_kv_heads}, {self._head_dim}]"
            )

        # Normalize q to [batch_size, q_len_per_req, num_qo_heads, head_dim].
        # q_len_per_req is derived from q.shape at run time, not from plan.
        q_in = q
        if q_in.dim() == 3:
            total_q, h_q, d = q_in.shape
            if total_q % self._batch_size != 0:
                raise ValueError(
                    f"q.shape[0]={total_q} not divisible by planned "
                    f"batch_size={self._batch_size}"
                )
            q_len_per_req = total_q // self._batch_size
            q_view = q_in.view(self._batch_size, q_len_per_req, h_q, d)
            out_was_3d = True
        elif q_in.dim() == 4:
            q_view = q_in
            q_len_per_req = q_view.shape[1]
            if q_view.shape[0] != self._batch_size:
                raise ValueError(
                    f"q.shape[0]={q_view.shape[0]} mismatches planned "
                    f"batch_size={self._batch_size}"
                )
            out_was_3d = False
        else:
            raise ValueError(f"q must be 3D or 4D; got ndim={q_in.dim()}")

        if q_view.shape[2] != self._num_qo_heads or q_view.shape[3] != self._head_dim:
            raise ValueError(
                f"q shape {tuple(q_view.shape)} mismatches plan "
                f"(num_qo_heads={self._num_qo_heads}, head_dim={self._head_dim})"
            )

        device = q.device
        # Atomic reduction accumulates into out via atomic_add and requires
        # zero init; kernel/none modes overwrite via direct store.
        atomic = self._reduction == "atomic"
        if out is None:
            torch_alloc = torch.zeros if atomic else torch.empty
            out_4d = torch_alloc(
                (self._batch_size, q_len_per_req, self._num_qo_heads, self._head_dim),
                dtype=self._o_data_type,
                device=device,
            )
            user_out = None
        else:
            # user out should be zero-init
            user_out = out
            if out.dim() == 3:
                out_4d = out.view(
                    self._batch_size,
                    q_len_per_req,
                    self._num_qo_heads,
                    self._head_dim,
                )
            elif out.dim() == 4:
                out_4d = out
            else:
                raise ValueError(f"out must be 3D or 4D; got ndim={out.dim()}")

        m = o_partial = l_partial = m_partial = None
        if self._has_workspace:
            if q_len_per_req <= self._q_len_per_req:
                # Runtime q_len doesnt exceed plantime q_len, return slice
                m = self._m[:, :q_len_per_req, :]
                o_partial = self._o_partial[:, :, :q_len_per_req, :, :]
                l_partial = self._l_partial[:, :, :q_len_per_req, :]
                m_partial = self._m_partial[:, :, :q_len_per_req, :]
            else:
                # Re-validate workspace size and slice
                m, o_partial, l_partial, m_partial = _slice_workspace(
                    self._float_workspace_buffer,
                    self._kv_splits,
                    self._batch_size,
                    q_len_per_req,
                    self._num_qo_heads,
                    self._head_dim,
                )
            m.fill_(float("-inf"))

        scale_s = self._sm_scale if sm_scale is None else sm_scale
        use_threshold = skip_softmax_threshold_scale_factor is not None
        use_lse = lse is not None
        use_sink = sinks is not None
        if use_threshold:
            if not skip_softmax_threshold_scale_factor > 0:
                raise ValueError(
                    f"skip_softmax_threshold_scale_factor must be > 0; "
                    f"got {skip_softmax_threshold_scale_factor}"
                )
            threshold_arg = Float32(skip_softmax_threshold_scale_factor)
        else:
            threshold_arg = None
        if use_lse:
            if lse.dtype != torch.float32:
                raise ValueError(
                    f"lse must be float32 (LSE is log2-base); got {lse.dtype}"
                )
            expected_lse_shape = (
                self._batch_size,
                q_len_per_req,
                self._num_qo_heads,
            )
            if lse.shape == expected_lse_shape:
                l_view = lse
            elif lse.shape == (self._batch_size * q_len_per_req, self._num_qo_heads):
                l_view = lse.view(*expected_lse_shape)
            else:
                raise ValueError(
                    f"lse shape {tuple(lse.shape)} doesn't match expected "
                    f"{expected_lse_shape} or its 3D-flat form"
                )
        else:
            l_view = None

        if use_sink:
            if sinks.ndim != 1 or sinks.shape[0] != self._num_qo_heads:
                raise ValueError(
                    f"sinks tensor must have shape (num_qo_heads,) = "
                    f"({self._num_qo_heads},), got shape {tuple(sinks.shape)}"
                )
            if sinks.dtype != torch.float32:
                raise ValueError(f"sinks must be float32, got {sinks.dtype}")
            if sinks.device != device:
                raise ValueError(
                    f"sinks must be on device {device}, got {sinks.device}"
                )
            if not sinks.is_contiguous():
                raise ValueError(
                    f"sinks tensor must be contiguous, got strides {sinks.stride()} "
                    f"for shape {sinks.shape}"
                )

        # Pick the variant matching this optional-feature set. The standard
        # no-threshold/no-LSE/no-sink path is always already compiled; the
        # other combinations are lazy.
        if use_threshold or use_lse or use_sink:
            fmha = _get_compiled_paged_decode_kernel(
                *self._compile_args, use_threshold, use_lse, use_sink
            )
        else:
            fmha = self._compiled_fmha_std

        o_scale_val = 1.0 if o_scale is None else o_scale
        # Stream is bound from the TVM-FFI env stream at runtime
        # (use_tvm_ffi_env_stream=True at compile), so the stream arg is omitted.
        fmha(
            self._kv_splits,
            self._seqlens,
            self._table_offsets,
            self._page_table,
            k_cache,
            v_cache,
            q_view,
            out_4d,
            l_view,
            m,
            o_partial,
            l_partial,
            m_partial,
            sinks,
            self._mask_config,
            Float32(scale_s),
            Float32(o_scale_val),
            threshold_arg,
            enable_pdl,
        )

        if user_out is not None:
            # Honour the in-place contract: return exactly what the user passed.
            return user_out
        # Caller didn't supply out; return shape matching the input rank.
        if out_was_3d:
            return out_4d.view(-1, self._num_qo_heads, self._head_dim)
        return out_4d
