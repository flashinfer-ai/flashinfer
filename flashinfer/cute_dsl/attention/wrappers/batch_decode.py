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
import math
from typing import Optional, Tuple

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32

from flashinfer.api_logging import flashinfer_api
from flashinfer.cute_dsl.utils import get_num_sm

from ..gqa_decode import GroupedQueryAttentionDecode
from ..gqa_decode_paged import GroupedQueryAttentionDecodePaged


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


_TORCH_TO_CUTLASS_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
}


def _torch_to_cutlass(dtype: torch.dtype) -> "cutlass.dtype":
    if dtype not in _TORCH_TO_CUTLASS_DTYPE:
        raise TypeError(
            f"cute-dsl decode does not support {dtype}; "
            f"supported: {list(_TORCH_TO_CUTLASS_DTYPE)}"
        )
    return _TORCH_TO_CUTLASS_DTYPE[dtype]


def _npo2(x: int) -> int:
    return 1 if x <= 1 else 2 ** math.ceil(math.log2(x))


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
    grouped_heads = num_qo_heads // num_kv_heads
    grouped_head_tile = min(32, _npo2(grouped_heads))
    prediction_tile = min(32 // grouped_head_tile, _npo2(prediction))
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
    grouped_head_tiles = math.ceil(grouped_heads / grouped_head_tile)
    prediction_tiles = math.ceil(prediction / prediction_tile)
    grid_yz = batch_size * num_kv_heads * grouped_head_tiles * prediction_tiles
    sm_count = sm_count if sm_count > 0 else 148
    kv_splits = max(1, sm_count // grid_yz)
    if sm_count == 148 and grid_yz == 32:
        kv_splits = 9  # 2 waves
    kv_splits = min(kv_splits, max(1, math.ceil(max_kv_len / 256)))
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
    if o_dtype in (cutlass.Float32, cutlass.Float16, cutlass.BFloat16) and kv_splits in (
        2,
        4,
        8,
    ):
        return "atomic"
    return "kernel"


# Float32 workspace tensors are 4-byte elements; the kernel's TMA fakes use
# assumed_align=16 so successive offsets need to be 16-byte-aligned. Each
# tensor's element count is always a multiple of `num_qo_heads * head_dim`,
# both of which are >= 8 — so byte sizes are naturally >= 16-aligned.
_FP32_BYTES = 4


def _workspace_layout(
    kv_splits: int,
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    head_dim: int,
):
    """Compute (offset, nbytes) for each kernel-reduction workspace tensor.

    Layout order matches the kernel call order: m_bsh, o_partial, m_partial,
    l_partial. Returns (offsets, sizes, total_bytes).
    """
    m_bsh_n = batch_size * q_len_per_req * num_qo_heads
    o_partial_n = kv_splits * batch_size * q_len_per_req * num_qo_heads * head_dim
    m_partial_n = kv_splits * batch_size * q_len_per_req * num_qo_heads
    l_partial_n = m_partial_n
    sizes = (
        m_bsh_n * _FP32_BYTES,
        o_partial_n * _FP32_BYTES,
        m_partial_n * _FP32_BYTES,
        l_partial_n * _FP32_BYTES,
    )
    offsets = (
        0,
        sizes[0],
        sizes[0] + sizes[1],
        sizes[0] + sizes[1] + sizes[2],
    )
    total = sum(sizes)
    return offsets, sizes, total


def _slice_fp32(buf_uint8: torch.Tensor, offset: int, shape):
    """Carve a contiguous float32 tensor of `shape` from a uint8 buffer at
    `offset` bytes. The resulting view aliases the buffer's storage."""
    nelem = 1
    for d in shape:
        nelem *= d
    nbytes = nelem * _FP32_BYTES
    return (
        buf_uint8[offset : offset + nbytes]
        .view(torch.float32)
        .view(*shape)
    )


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
    kv_splits: int,
    reduction: str,
    tma_mask: bool,
    use_lse: bool,
):
    """Compile and cache the ragged (contiguous-KV) GQA decode kernel."""
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads, num_kv_heads, prediction, in_dtype.width
    )
    fmha = GroupedQueryAttentionDecode(
        head_dim,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        reduction_mode=reduction,
        tma_mask=tma_mask,
    )
    has_workspace = reduction == "kernel"
    acc_dtype = cutlass.Float32

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
            (kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        kv_splits,
        q_fake,
        k_fake,
        v_fake,
        o_fake,
        l_fake,
        m_fake,
        o_partial_fake,
        l_partial_fake,
        m_partial_fake,
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
    kv_splits: int,
    reduction: str,
    tma_mask: bool,
    use_threshold: bool,
    use_lse: bool,
):
    """Compile and cache the paged GQA decode kernel."""
    grouped_head_tile, prediction_tile = _pick_tile_shape(
        num_qo_heads, num_kv_heads, prediction, in_dtype.width
    )
    blk_tile_n = grouped_head_tile * prediction_tile
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
        sequence_tile=128 if use_threshold else 256,
        reduction_mode=reduction,
        softmax_warpgroups=softmax_warpgroups,
        tma_mask=tma_mask,
    )
    has_workspace = reduction == "kernel"
    acc_dtype = cutlass.Float32

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
            (kv_splits, sym_batch, sym_prediction, num_qo_heads, head_dim),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        l_partial_fake = cute.runtime.make_fake_compact_tensor(
            acc_dtype,
            (kv_splits, sym_batch, sym_prediction, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )

    threshold_p_fake = Float32(0.0) if use_threshold else None
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    return cute.compile(
        fmha,
        kv_splits,
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
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        q_len_per_req: int = 1,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
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
        q_len_per_req : int
            Predicted tokens per request (1 for plain decode, >1 for
            speculative decode).
        is_causal : bool
            Causal masking for speculative decode.
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
            ∈ {1, 2, 4, 8, 16} and an output dtype in {float32, float16,
            bfloat16}. ``"auto"`` picks ``"none"`` when kv_splits == 1,
            ``"atomic"`` for compatible dtypes and small kv_splits, else
            ``"kernel"``.
        """
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
            o_data_type = torch.float16 if q_data_type == torch.float8_e4m3fn else q_data_type
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
        self._q_len_per_req = q_len_per_req
        self._kv_splits = kv_splits
        self._reduction = reduction
        # Only kernel-reduction uses workspace tensors; atomic and none
        # write to o_bshd directly (atomic via atomic_add, none via direct
        # store).
        self._has_workspace = reduction == "kernel"
        self._tma_mask = not is_causal
        if sm_scale is None:
            sm_scale = head_dim ** -0.5
        self._sm_scale = sm_scale

        # Validate float_workspace_buffer at plan() time. The size is
        # computed against the planned batch_size (max-batch hint) and the
        # planned q_len_per_req. Runtime values <= these hints reuse the
        # same allocation; larger values are re-validated in run().
        if self._has_workspace:
            _, _, total_bytes = _workspace_layout(
                kv_splits, batch_size, q_len_per_req, num_qo_heads, head_dim
            )
            if self._float_workspace_buffer.numel() < total_bytes:
                raise ValueError(
                    f"float_workspace_buffer too small for kernel-reduction: "
                    f"{self._float_workspace_buffer.numel()} bytes, need "
                    f"{total_bytes} bytes (kv_splits={kv_splits}, "
                    f"batch_size={batch_size}, q_len_per_req={q_len_per_req}, "
                    f"num_qo_heads={num_qo_heads}, head_dim={head_dim})"
                )

        # Always compile the no-LSE variant at plan() time; the LSE variant
        # is lazily fetched at run() time when needed (compile cache hit).
        self._compile_args = (
            in_dtype,
            out_dtype,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            q_len_per_req,
            kv_splits,
            reduction,
            self._tma_mask,
        )
        self._compiled_fmha_std = _get_compiled_decode_kernel(
            *self._compile_args, False
        )
        self._planned = True

    def _allocate_workspace(
        self, batch_size: int, q_len_per_req: int, device: torch.device
    ):
        """Carve kernel-reduction workspace tensors out of float_workspace_buffer."""
        if not self._has_workspace:
            return None, None, None, None
        shape_m = (batch_size, q_len_per_req, self._num_qo_heads)
        shape_p_o = (
            self._kv_splits,
            batch_size,
            q_len_per_req,
            self._num_qo_heads,
            self._head_dim,
        )
        shape_p_ml = (
            self._kv_splits,
            batch_size,
            q_len_per_req,
            self._num_qo_heads,
        )
        offsets, _, total_bytes = _workspace_layout(
            self._kv_splits, batch_size, q_len_per_req,
            self._num_qo_heads, self._head_dim,
        )
        if self._float_workspace_buffer.numel() < total_bytes:
            raise ValueError(
                f"float_workspace_buffer too small at run(): "
                f"{self._float_workspace_buffer.numel()} bytes, need "
                f"{total_bytes} bytes (batch_size={batch_size}, "
                f"q_len_per_req={q_len_per_req} — exceeded the plan-time hint)"
            )
        buf = self._float_workspace_buffer
        m_bsh = _slice_fp32(buf, offsets[0], shape_m)
        o_partial = _slice_fp32(buf, offsets[1], shape_p_o)
        m_partial = _slice_fp32(buf, offsets[2], shape_p_ml)
        l_partial = _slice_fp32(buf, offsets[3], shape_p_ml)
        # m_bsh is read by the kernel as a running colmax accumulator and
        # must start at -inf. The other three are written from scratch.
        m_bsh.fill_(float("-inf"))
        return m_bsh, o_partial, m_partial, l_partial

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        lse: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
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
        lse : Optional[torch.Tensor]
            Pre-allocated float32 buffer of shape
            ``(batch_size, q_len_per_req, num_qo_heads)`` to receive the
            log-sum-exp (log2 base, matching flashinfer convention). When
            ``None`` (default) the kernel skips the LSE write entirely;
            otherwise a log2-base LSE variant is lazily compiled on first
            use (cache hit afterwards).
        """
        if not self._planned:
            raise RuntimeError("Call plan() before run().")
        if q.dtype != self._q_data_type or k.dtype != self._q_data_type or v.dtype != self._q_data_type:
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
            raise ValueError(f"k.shape={tuple(k.shape)} must equal v.shape={tuple(v.shape)}")

        device = q.device
        # Atomic reduction accumulates into out via atomic_add and requires
        # zero init; kernel/none modes overwrite via direct store.
        atomic = self._reduction == "atomic"
        if out is None:
            if atomic:
                out = torch.zeros(
                    (b, s_q, h_q, d), dtype=self._o_data_type, device=device
                )
            else:
                out = torch.empty(
                    (b, s_q, h_q, d), dtype=self._o_data_type, device=device
                )
        elif atomic:
            out.zero_()

        m_bsh, o_partial, m_partial, l_partial = self._allocate_workspace(
            b, s_q, device
        )

        scale_s = self._sm_scale if sm_scale is None else sm_scale
        use_lse = lse is not None
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

        fmha = (
            self._compiled_fmha_std
            if not use_lse
            else _get_compiled_decode_kernel(*self._compile_args, True)
        )

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
            m_bsh,
            o_partial,
            l_partial,
            m_partial,
            Float32(scale_s),
            Float32(scale_o),
            enable_pdl,
        )
        return out


# ---------------------------------------------------------------------------
# Paged-KV decode wrapper
# ---------------------------------------------------------------------------


class BatchDecodePagedCuteDSLWrapper:
    """PyTorch-facing wrapper for the paged CuTe DSL GQA decode kernel.

    Designed to back the ``cute-dsl`` backend of
    :class:`flashinfer.BatchDecodeWithPagedKVCacheWrapper`.  Accepts a
    flashinfer-style paged KV cache (indptr/indices/last_page_len) and adapts
    it to the kernel's ``(seqlens, table_offsets, page_table)`` triplet.
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
    ) -> None:
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
        last_page_len: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        q_data_type: torch.dtype = torch.bfloat16,
        kv_data_type: Optional[torch.dtype] = None,
        o_data_type: Optional[torch.dtype] = None,
        q_len_per_req: int = 1,
        is_causal: bool = True,
        sm_scale: Optional[float] = None,
        kv_splits: Optional[int] = None,
        reduction: str = "auto",
        precompile_skip_softmax_kernel: bool = False,
        max_kv_len: Optional[int] = None,
        non_blocking: bool = True,
    ) -> None:
        """Plan paged GQA decode for the given problem.

        Parameters
        ----------
        indptr : torch.Tensor (int32, [batch_size + 1])
            Prefix-sum offsets into ``indices``.
        indices : torch.Tensor (int32, [num_pages_total])
            Flat per-sequence virtual page indices.
        last_page_len : torch.Tensor (int32, [batch_size])
            Number of valid tokens on the last page of each sequence.
        num_qo_heads, num_kv_heads, head_dim, page_size : int
            GQA + paging configuration.  ``page_size`` must be in
            ``{8, 16, 32, 64}`` and ``head_dim`` a positive multiple of 64.
        q_data_type, kv_data_type, o_data_type : torch.dtype
            Q/K/V/O dtypes; ``kv_data_type`` must equal ``q_data_type``.
        q_len_per_req : int
            Predicted tokens per request (1 for plain decode).
        is_causal : bool
            Causal masking for speculative decode.
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
        precompile_skip_softmax_kernel : bool
            If True, also compile the BLASST skip-softmax variant of the
            kernel at plan() time, so the first :meth:`run` call that
            passes ``skip_softmax_threshold_scale_factor`` is fast.  When False (default)
            the BLASST variant is lazily compiled on first use, which keeps
            plan() fast at the cost of a one-time compile latency spike on
            that first call.  The standard (no-BLASST) variant is always
            compiled at plan() time because the BLASST predicate has a
            real per-tile cost on the standard path.
        max_kv_len : Optional[int]
            Maximum KV sequence length across the batch.  Used to auto-tune
            ``kv_splits``; pass it explicitly to avoid a GPU→CPU sync.
        non_blocking : bool
            Async device copies for the plan-time integer buffers.
        """
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
            o_data_type = torch.float16 if q_data_type == torch.float8_e4m3fn else q_data_type
        out_dtype = _torch_to_cutlass(o_data_type)

        batch_size = last_page_len.numel()
        if indptr.numel() != batch_size + 1:
            raise ValueError(
                f"indptr length ({indptr.numel()}) must equal batch_size+1 "
                f"({batch_size + 1})"
            )

        # Derive table_offsets and seqlens directly on-device.
        indptr_i32 = indptr.to(torch.int32) if indptr.dtype != torch.int32 else indptr
        last_page_len_i32 = (
            last_page_len.to(torch.int32)
            if last_page_len.dtype != torch.int32
            else last_page_len
        )
        if indptr_i32.device != self.device:
            indptr_i32 = indptr_i32.to(self.device, non_blocking=non_blocking)
        if last_page_len_i32.device != self.device:
            last_page_len_i32 = last_page_len_i32.to(
                self.device, non_blocking=non_blocking
            )
        if indices.device != self.device:
            indices = indices.to(self.device, non_blocking=non_blocking)
        indices_i32 = indices.to(torch.int32) if indices.dtype != torch.int32 else indices

        table_offsets = indptr_i32[:-1].contiguous()
        # Equivalent to flashinfer.page.get_seq_lens (clamps empty sequences to 0).
        seqlens = (
            torch.clamp(indptr_i32[1:] - indptr_i32[:-1] - 1, min=0) * page_size
            + last_page_len_i32
        ).contiguous()
        if max_kv_len is None:
            # Forces a GPU→CPU sync; callers that already know max_kv_len on
            # the host (e.g. BatchDecodeWithPagedKVCacheWrapper) should pass it
            # in to avoid this.
            max_kv_len = int(seqlens.max().item()) if seqlens.numel() > 0 else 0

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
        self._tma_mask = not is_causal
        if sm_scale is None:
            sm_scale = head_dim ** -0.5
        self._sm_scale = sm_scale

        self._batch_size = batch_size
        self._seqlens = seqlens
        self._table_offsets = table_offsets
        self._page_table = indices_i32

        # Validate float_workspace_buffer at plan() time. Sized against the
        # plan-time-fixed batch_size + planned q_len_per_req hint; runtime
        # q_len_per_req <= hint reuses the same allocation.
        if self._has_workspace:
            _, _, total_bytes = _workspace_layout(
                kv_splits, batch_size, q_len_per_req, num_qo_heads, head_dim
            )
            if self._float_workspace_buffer.numel() < total_bytes:
                raise ValueError(
                    f"float_workspace_buffer too small for kernel-reduction: "
                    f"{self._float_workspace_buffer.numel()} bytes, need "
                    f"{total_bytes} bytes (kv_splits={kv_splits}, "
                    f"batch_size={batch_size}, q_len_per_req={q_len_per_req}, "
                    f"num_qo_heads={num_qo_heads}, head_dim={head_dim})"
                )

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
            kv_splits,
            reduction,
            self._tma_mask,
        )
        # Standard (no-BLASST, no-LSE) variant — always compiled at plan().
        # BLASST and LSE variants compile lazily on first use via the cache.
        self._compiled_fmha_std = _get_compiled_paged_decode_kernel(
            *self._compile_args, False, False
        )
        if precompile_skip_softmax_kernel:
            # Warm the cache so the BLASST variant is ready for the first
            # run() that uses skip_softmax_threshold.
            _get_compiled_paged_decode_kernel(*self._compile_args, True, False)
        self._planned = True

    def _allocate_workspace(
        self, batch_size: int, q_len_per_req: int, device: torch.device
    ):
        """Carve kernel-reduction workspace tensors out of float_workspace_buffer."""
        if not self._has_workspace:
            return None, None, None, None
        shape_m = (batch_size, q_len_per_req, self._num_qo_heads)
        shape_p_o = (
            self._kv_splits,
            batch_size,
            q_len_per_req,
            self._num_qo_heads,
            self._head_dim,
        )
        shape_p_ml = (
            self._kv_splits,
            batch_size,
            q_len_per_req,
            self._num_qo_heads,
        )
        offsets, _, total_bytes = _workspace_layout(
            self._kv_splits, batch_size, q_len_per_req,
            self._num_qo_heads, self._head_dim,
        )
        if self._float_workspace_buffer.numel() < total_bytes:
            raise ValueError(
                f"float_workspace_buffer too small at run(): "
                f"{self._float_workspace_buffer.numel()} bytes, need "
                f"{total_bytes} bytes (q_len_per_req={q_len_per_req} — "
                f"exceeded the plan-time hint)"
            )
        buf = self._float_workspace_buffer
        m_bsh = _slice_fp32(buf, offsets[0], shape_m)
        o_partial = _slice_fp32(buf, offsets[1], shape_p_o)
        m_partial = _slice_fp32(buf, offsets[2], shape_p_ml)
        l_partial = _slice_fp32(buf, offsets[3], shape_p_ml)
        # m_bsh is read by the kernel as a running colmax accumulator and
        # must start at -inf. The other three are written from scratch.
        m_bsh.fill_(float("-inf"))
        return m_bsh, o_partial, m_partial, l_partial

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        sm_scale: Optional[float] = None,
        o_scale: Optional[float] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        lse: Optional[torch.Tensor] = None,
        enable_pdl: bool = True,
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
            Pre-allocated output of the same logical shape as ``q``.
            For atomic reduction this is zero-filled on entry.
        sm_scale : Optional[float]
            Per-call override of the softmax scale set at plan() time.
        o_scale : Optional[float]
            Output scale applied to the final O before it is written. The
            cute-dsl kernel folds this in for free in the reduction
            epilogue (no separate post-kernel multiply). Defaults to 1.0.
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
        """
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
        q_in = q if q.is_contiguous() else q.contiguous()
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
            alloc = torch.zeros if atomic else torch.empty
            out_4d = alloc(
                (self._batch_size, q_len_per_req, self._num_qo_heads, self._head_dim),
                dtype=self._o_data_type,
                device=device,
            )
            user_out = None
        else:
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
            if atomic:
                out_4d.zero_()

        m_bsh, o_partial, m_partial, l_partial = self._allocate_workspace(
            self._batch_size, q_len_per_req, device
        )

        scale_s = self._sm_scale if sm_scale is None else sm_scale
        use_threshold = skip_softmax_threshold_scale_factor is not None
        use_lse = lse is not None
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

        # Pick the variant matching this (use_threshold, use_lse) — std is
        # always already compiled; the other three are lazy.
        if use_threshold or use_lse:
            fmha = _get_compiled_paged_decode_kernel(
                *self._compile_args, use_threshold, use_lse
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
            m_bsh,
            o_partial,
            l_partial,
            m_partial,
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
