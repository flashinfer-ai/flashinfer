# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Internal native-row experiment for the Rubin MTP pipeline.

Row packing follows the native-layout strategy discussed in LightSeek's MLA 2
write-up. This adapter reuses FlashInfer's preferred/fallback pipeline and shared
reducer; it is not registered with public dispatch. SM107 runtime qualification
is required before admission.
"""

import functools
import math

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

from ..monolithic.mla_reducer import MLAReducer
from .warp_reducer import WarpMLAReducer
from .kernel import (
    RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo,
    _MixedFallbackPrep,
)


def native_geometry(heads, q_len, batch=1, splits=1):
    if any(type(x) is not int for x in (heads, q_len, batch, splits)):
        raise ValueError("native geometry requires integer dimensions and splits")
    if heads not in (96, 128) or q_len not in (2, 4, 8):
        raise ValueError("native experiment requires H96/H128 and Q2/Q4/Q8")
    if batch <= 0 or not 1 <= splits <= 32:
        raise ValueError("batch must be positive and splits must be in [1,32]")
    rows = heads * q_len
    if batch * rows * 576 >= 1 << 31:
        raise ValueError("native query exceeds signed Int32 indexing")
    tiles = (rows + 255) // 256
    elements = batch * tiles * 256 * splits * 513
    if splits > 1 and elements >= 1 << 31:
        raise ValueError("native workspace exceeds signed Int32 element indexing")
    return rows, tiles, 0 if splits == 1 else elements * 4


@cute.jit
def _query_rows(t):
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (t.shape[1] * t.shape[2], t.shape[3], 1, t.shape[0]),
            stride=(t.stride[2], t.stride[3], t.stride[0], t.stride[0]),
        ),
    )


@cute.jit
def _output_tiles(t):
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (256, t.shape[3], cute.ceil_div(t.shape[1] * t.shape[2], 256), t.shape[0]),
            stride=(t.stride[2], t.stride[3], 256 * t.stride[2], t.stride[0]),
        ),
    )


@cute.jit
def _lse_tiles(t):
    return cute.make_tensor(
        t.iterator,
        cute.make_layout(
            (256, cute.ceil_div(t.shape[1] * t.shape[2], 256), t.shape[0]),
            stride=(t.stride[2], 256 * t.stride[2], t.stride[0]),
        ),
    )


class NativeMtpKernel:
    def __init__(
        self,
        *,
        heads,
        q_len,
        page_size,
        splits,
        clusters,
        branch="auto",
        correction=0.0,
        fp16=False,
        prefetch=8,
        reducer_kind="shared",
        fallback_arch="sm_107",
    ):
        if reducer_kind not in ("shared", "warp"):
            raise ValueError("native reducer must be shared or warp")
        rows, _, _ = native_geometry(heads, q_len, splits=splits)
        capacity = max(4, 1 << (splits - 1).bit_length())
        self.inner = RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo(
            Float32,
            Float32,
            (256, 128),
            (256, 256),
            clusters,
            page_size,
            correction,
            False,
            True,
            False,
            use_fp16_softmax=fp16,
            force_branch=branch,
            reducer_max_splits=capacity,
            packed_rows=rows,
        )
        self.inner.causal_num_heads = heads
        self.inner.causal_seq_len_q = q_len
        self.inner.causal_fold_ratio = 2
        if reducer_kind == "warp":
            self.inner.reducer = WarpMLAReducer(capacity)
        else:
            self.inner.reducer = MLAReducer(
                acc_dtype=Float32,
                lse_dtype=Float32,
                qk_tile_shape=(256, 128),
                num_heads=heads,
                seq_len_q=q_len,
                max_splits=capacity,
                empty_lse_negative_inf=True,
                max_prefetch_splits=prefetch,
            )
        self.inner._fb = _MixedFallbackPrep(
            Float32,
            Float32,
            (128, 128),
            (128, 256),
            clusters,
            page_size,
            correction,
            False,
            True,
            False,
            fold_sq=False,
            num_heads=128,
            seq_len_q=q_len,
            use_fp16_softmax=fp16,
        )
        self.inner._fb.causal_num_heads = heads
        self.inner._fb.packed_rows = rows
        self.inner._fb.early_compute_clc = branch == "auto"
        self.inner._fb.merge_softmax_loops = False
        self.inner._fb.arch_str = fallback_arch

    @cute.jit
    def __call__(
        self,
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        table,
        out,
        lse,
        workspace,
        splits,
        lengths,
        scale,
        output_scale,
        lse_scale,
        stream,
    ):
        # Input row bounds remain logical; the output/workspace uses M256 tiles.
        q_latent, q_rope = _query_rows(q_latent), _query_rows(q_rope)
        out, lse = _output_tiles(out), _lse_tiles(lse)
        c_latent = cute.make_tensor(
            c_latent.iterator, cute.select(c_latent.layout, [1, 2, 0])
        )
        c_rope = cute.make_tensor(
            c_rope.iterator, cute.select(c_rope.layout, [1, 2, 0])
        )
        table = cute.make_tensor(table.iterator, cute.select(table.layout, [1, 0]))
        self.inner(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            table,
            out,
            lse,
            workspace,
            splits,
            lengths,
            None,
            scale,
            output_scale,
            lse_scale,
            stream,
        )


@functools.cache
def compile_native(
    *,
    heads,
    q_len,
    page_size=64,
    splits=1,
    clusters=52,
    branch="auto",
    correction=0.0,
    fp16=False,
    prefetch=8,
    reducer_kind="shared",
    arch="sm_107a",
):
    """Compile a private raw launcher; the caller must validate tensor metadata.

    Only SM107 is admitted here. A separate test launcher can qualify the
    fallback body on SM100 without constructing this preferred pipeline.
    """
    if type(clusters) is not int or clusters <= 0:
        raise ValueError("cluster capacity must be a positive integer")
    if arch != "sm_107a":
        raise ValueError("native experiment targets SM107")
    if page_size not in (64, 128) or branch not in ("auto", "preferred", "fallback"):
        raise ValueError("unsupported native page size or branch")
    kernel = NativeMtpKernel(
        heads=heads,
        q_len=q_len,
        page_size=page_size,
        splits=splits,
        clusters=clusters,
        branch=branch,
        correction=correction,
        fp16=fp16,
        prefetch=prefetch,
        reducer_kind=reducer_kind,
        fallback_arch="sm_107",
    )
    b, q, h, pages, ps, table_pages = (cute.sym_int() for _ in range(6))
    d, r = (cute.sym_int(divisibility=16) for _ in range(2))

    def strided(shape):
        return cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN,
            shape,
            stride=tuple(cute.sym_int() for _ in shape[:-1]) + (1,),
            assumed_align=16,
        )

    def compact(dtype, shape, order, align=16):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=order, assumed_align=align
        )

    return cute.compile(
        kernel,
        strided((b, q, h, d)),
        strided((b, q, h, r)),
        strided((pages, ps, d)),
        strided((pages, ps, r)),
        compact(Int32, (b, table_pages), (1, 0), 4),
        compact(cutlass.Float8E4M3FN, (b, q, h, d), (3, 2, 1, 0)),
        compact(Float32, (b, q, h), (2, 1, 0), 4),
        None if splits == 1 else compact(cutlass.Int8, (cute.sym_int(64),), (0,), 32),
        Int32(splits),
        compact(Int32, (b,), (0,), 4),
        Float32(1),
        Float32(1),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch={arch}",
    )


def prepare_native_mla_decode(**configuration):
    """Prepare the internal prototype with metadata-only checks on every call.

    The caller supplies valid page IDs and live lengths in [0, table capacity].
    Input, output and scratch storage must not overlap. A prepared callable can
    be reused with fresh buffers, but concurrent launches need separate scratch.
    """
    import torch

    from flashinfer.cute_dsl._mla_validation import _validate_mtp_scales
    from .mla_decode import _check_tensor_indexing, _check_kv_tensor_indexing

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("prepare native MTP before CUDA graph capture")
    heads = configuration["heads"]
    q_len = configuration["q_len"]
    splits = configuration.get("splits", 1)
    page_size = configuration.get("page_size", 64)
    compiled = compile_native(**configuration)

    def launch(
        q_latent,
        q_rope,
        c_latent,
        c_rope,
        table,
        out,
        lse,
        workspace,
        runtime_splits,
        lengths,
        scale,
        output_scale,
        lse_scale,
    ):
        if not isinstance(runtime_splits, (int, Int32)) or isinstance(
            runtime_splits, bool
        ):
            raise ValueError("runtime splits must be a host integer")
        if int(runtime_splits) != splits:
            raise ValueError("runtime splits differ from the native prepared budget")
        if q_latent.ndim != 4:
            raise ValueError("native query must be [B,Q,H,D]")
        b = q_latent.shape[0]
        _, _, workspace_bytes = native_geometry(heads, q_len, b, splits)
        device = q_latent.device
        if device.type != "cuda" or torch.cuda.get_device_capability(device) != (10, 7):
            raise ValueError("native MTP execution requires SM107")
        for tensor, shape in (
            (q_latent, (b, q_len, heads, 512)),
            (q_rope, (b, q_len, heads, 64)),
            (out, (b, q_len, heads, 512)),
        ):
            if tuple(tensor.shape) != shape or tensor.dtype != torch.float8_e4m3fn:
                raise ValueError(
                    "native query/output must match the prepared FP8 shape"
                )
            if tensor.stride(-1) != 1 or tensor.stride(1) != heads * tensor.stride(2):
                raise ValueError(
                    "native query rows must have affine token/head strides"
                )
        if not out.is_contiguous() or not lse.is_contiguous():
            raise ValueError("native output and LSE must be contiguous")
        if tuple(lse.shape) != (b, q_len, heads) or lse.dtype != torch.float32:
            raise ValueError("native LSE must be float32 [B,Q,H]")
        for tensor, dim in ((c_latent, 512), (c_rope, 64)):
            if tensor.ndim != 3 or tuple(tensor.shape[1:]) != (page_size, dim):
                raise ValueError("native KV views must match the prepared page size")
            if tensor.dtype != torch.float8_e4m3fn or tensor.stride(-1) != 1:
                raise ValueError("native KV must be FP8 with contiguous columns")
            if tensor.shape[0] <= 0:
                raise ValueError("native KV pool must contain at least one page")
            _check_kv_tensor_indexing(tensor, "native KV")
        if c_latent.shape[0] != c_rope.shape[0]:
            raise ValueError("native KV views must have matching page counts")
        if (
            table.ndim != 2
            or table.shape[0] != b
            or not 0 < table.shape[1] * page_size < (1 << 31)
            or table.dtype != torch.int32
            or not table.is_contiguous()
        ):
            raise ValueError("native page table must be contiguous int32 [B,max_pages]")
        if (
            tuple(lengths.shape) != (b,)
            or lengths.dtype != torch.int32
            or not lengths.is_contiguous()
        ):
            raise ValueError("native lengths must be contiguous int32 [B]")
        tensors = (q_latent, q_rope, c_latent, c_rope, table, out, lse, lengths)
        if any(t.device != device for t in tensors):
            raise ValueError("native tensors must be on one device")
        for tensor in (q_latent, q_rope, c_latent, c_rope):
            strides = tensor.stride()
            if tensor.data_ptr() % 16 or any(s % 16 for s in strides[:-1]):
                raise ValueError(
                    "native TMA pointers and leading strides must be 16-byte aligned"
                )
            if any(
                strides[i] < strides[i + 1] * tensor.shape[i + 1]
                for i in range(tensor.ndim - 1)
            ):
                raise ValueError("native TMA rows and batches must not overlap")
        if out.data_ptr() % 16 or any(t.data_ptr() % 4 for t in (lse, table, lengths)):
            raise ValueError("native output or metadata pointer is misaligned")
        for tensor in (q_latent, q_rope, table, out, lse, lengths):
            _check_tensor_indexing(tensor, "native tensor")
        if workspace_bytes:
            if (
                workspace is None
                or workspace.ndim != 1
                or workspace.dtype != torch.int8
                or workspace.device != device
                or not workspace.is_contiguous()
                or workspace.numel() < workspace_bytes
                or workspace.data_ptr() % 32
            ):
                raise ValueError(
                    "native workspace is undersized, misaligned or incompatible"
                )
        elif workspace is not None:
            raise ValueError("unsplit native execution requires workspace=None")
        # Avoid implicit device-to-host conversion of tensor scalar overrides.
        scalars = (scale, output_scale, lse_scale)
        if any(not isinstance(x, (float, int, Float32)) for x in scalars):
            raise ValueError("native scales must be host floating-point scalars")
        lse_factor = float(getattr(lse_scale, "value", lse_scale))
        if not math.isfinite(lse_factor) or lse_factor <= 0:
            raise ValueError("native LSE scale must be finite and positive")
        _validate_mtp_scales(
            float(getattr(scale, "value", scale)),
            float(getattr(output_scale, "value", output_scale)),
        )
        compiled(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            table,
            out,
            lse,
            workspace,
            runtime_splits,
            lengths,
            scale,
            output_scale,
            lse_scale,
        )

    return launch
