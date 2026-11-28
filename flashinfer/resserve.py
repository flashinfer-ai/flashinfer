# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Experimental ResServe quantization, folded GEMM and paged decode APIs.

Import public entry points from this module. Each call explicitly opts into
an experimental API and lazily imports its backend. Implementation modules
under ``flashinfer.experimental.resserve_*`` are internal. These APIs remain
experimental and may change without compatibility guarantees.
"""

import torch

from .api_logging import flashinfer_experimental_api
from .utils import get_compute_capability

__all__ = [
    "nvfp4_quantize_residue",
    "nvfp4_quantize_kext",
    "mm_nvfp4_residue",
    "paged_decode_residue",
]


def _check_tensor(tensor, name, device, dtype, alignment=16, *, contiguous=True):
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype not in dtype:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if (contiguous and not tensor.is_contiguous()) or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


def _swizzle_shape(rows: int, k: int) -> tuple[int, int]:
    """128-row, 4-scale E4M3 swizzle viewed as ``(padded_rows, k_tiles * 4)``."""
    return ((rows + 127) // 128) * 128, ((k + 63) // 64) * 4


def _swizzle_view(scales: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    padded_rows, cols = _swizzle_shape(rows, k)
    return scales.view(torch.float8_e4m3fn).view(padded_rows, cols)


def _check_swizzle(scale, rows: int, k: int, name: str, device) -> None:
    padded_rows, cols = _swizzle_shape(rows, k)
    expected = padded_rows * cols
    if scale.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError(f"{name} must be uint8 or float8_e4m3fn")
    if not scale.is_cuda or scale.device != device or not scale.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor on {device}")
    if scale.data_ptr() % 16:
        raise ValueError(f"{name} must be 16-byte aligned")
    if scale.ndim == 2 and tuple(scale.shape) != (padded_rows, cols):
        raise ValueError(f"{name} must have shape {(padded_rows, cols)}")
    if scale.ndim not in (1, 2) or scale.numel() != expected:
        raise ValueError(f"{name} must contain {expected} swizzle bytes")


def _check_sm100(tensor):
    if not tensor.is_cuda:
        raise ValueError("ResServe requires CUDA tensors")
    if get_compute_capability(tensor.device) != (10, 0):
        raise ValueError("ResServe currently supports SM100 only")


@flashinfer_experimental_api
def nvfp4_quantize_kext(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    residue_per_8: int,
    *,
    elts_per_thread: int = 0,
):
    """Append selected-channel NVFP4 residues along K (ResServe KEXT).

    SM100; contiguous FP16/BF16 ``input[M, K]``, positive M and K, K divisible
    by 128. ``global_scale`` is one positive finite FP32 quantization multiplier
    on the input device. Inputs must be finite.

    ``channel_masks`` is contiguous uint8 ``[K/8]`` on the same device. Every
    byte must have exactly ``residue_per_8`` set bits, selecting that group's
    channels in ascending bit order. ``residue_per_8`` is 1, 2 or 4 (12.5%,
    25% or 50% extension). Mask contents and scale values are caller
    preconditions, not checked with a synchronizing device-to-host copy.

    Returns packed uint8 E2M1 ``[M, K_ext/2]`` (base then selected residue)
    and E4M3 scales in the 128-row, 4-scale swizzle, viewed as
    ``(padded_M, K_ext_tiles * 4)``. Padding entries are zero.
    Extended weights must append the same selected columns in
    ascending channel order *before* weight quantization. Use ordinary
    ``flashinfer.mm_fp4`` with reconstruction alpha equal to the reciprocal
    product of activation and weight global multipliers. Its backend-specific
    K alignment/padding requirements still apply.

    ``elts_per_thread=0`` preserves ResServe's SM100 pack8/pack16 shape policy;
    8 or 16 explicitly selects one producer. Pack8 requires K divisible by
    256; widths divisible by 128 but not 256 use pack16 automatically.
    This is not GEMM autotuning.
    Warm up before CUDA graph capture. Swap and ratio-1 MEXT are separate.
    """
    _check_sm100(input)
    _check_tensor(input, "input", input.device, (torch.float16, torch.bfloat16), 32)
    _check_tensor(global_scale, "global_scale", input.device, (torch.float32,), 4)
    _check_tensor(channel_masks, "channel_masks", input.device, (torch.uint8,), 1)
    if input.ndim != 2 or min(input.shape) <= 0 or input.shape[1] % 128:
        raise ValueError("input must be [M, K], M > 0, K > 0 and K divisible by 128")
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain one element")
    if channel_masks.ndim != 1 or channel_masks.numel() != input.shape[1] // 8:
        raise ValueError("channel_masks must have shape [K/8]")
    if residue_per_8 not in (1, 2, 4) or isinstance(residue_per_8, bool):
        raise ValueError("residue_per_8 must be 1, 2 or 4")
    if elts_per_thread not in (0, 8, 16):
        raise ValueError("elts_per_thread must be 0, 8 or 16")
    from .experimental.resserve_kext.quantization import quantize

    with torch.cuda.device(input.device):
        packed, scales = quantize(
            input, global_scale, channel_masks, residue_per_8, elts_per_thread
        )
        ext = input.shape[1] + input.shape[1] // 8 * residue_per_8
        return packed, _swizzle_view(scales, input.shape[0], ext)


@flashinfer_experimental_api
def nvfp4_quantize_residue(input: torch.Tensor, global_scale: torch.Tensor):
    """Quantize each activation row and its full-channel residue on SM100.

    ``input`` is contiguous FP16/BF16 ``[M, K]``, with positive M and K,
    K divisible by 16. ``global_scale`` contains one positive finite
    FP32 quantization multiplier shared by the base and residue on the same device.
    Scale values and finite input values are caller preconditions (not checked
    with a synchronizing device-to-host copy).

    Returns uint8 packed E2M1 ``[2*M, K/2]`` in adjacent base/residue row order
    and E4M3 scales in the 128-row, 4-scale swizzle, viewed as
    ``(padded_rows, K_tiles * 4)``. Padding entries are zero.
    The residue is computed from the quantized
    and reconstructed base, following ResServe's production pack16 algorithm.
    """
    _check_sm100(input)
    _check_tensor(input, "input", input.device, (torch.float16, torch.bfloat16), 32)
    _check_tensor(global_scale, "global_scale", input.device, (torch.float32,), 4)
    if input.ndim != 2 or min(input.shape) <= 0 or input.shape[1] % 16:
        raise ValueError("input must be [M, K], M > 0, K > 0 and K divisible by 16")
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain one element")
    from .experimental.resserve_mext.quantization import quantize

    with torch.cuda.device(input.device):
        packed, scales = quantize(input, global_scale)
        return packed, _swizzle_view(scales, 2 * input.shape[0], input.shape[1])


@flashinfer_experimental_api
def mm_nvfp4_residue(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Multiply ResServe row-pair activations by shared weights and fold in FP32.

    ``act`` is packed uint8 ``[2*M, K/2]`` from ``nvfp4_quantize_residue``;
    ``weight`` is packed uint8 ``[N, K/2]`` with K-contiguous, nonoverlapping
    rows and a 32-byte-aligned row stride. A base-K view of extended weights
    is accepted; its scales must be separately packed for the base K.
    Both scale buffers are the 128-row, 4-scale E4M3 swizzle, as a
    ``(padded_rows, K_tiles * 4)`` view (float8 or uint8) or the same bytes
    flattened. K is divisible by 64, N by 8.
    ``alpha`` is one FP32 reconstruction multiplier: the reciprocal of
    the product of activation and weight quantization global multipliers.
    It scales the sum of the base and residue accumulators.
    A preceding kernel on the same stream may produce ``alpha``.
    Keep the value unchanged after production until the operation completes.
    The output is ``[M, N]`` in FP16 or BF16. Weight quantization is external.
    The Fold tile is a static row-pair choice: ``(128, 24)`` through M 8,
    ``(128, 32)`` through M 16, ``(128, 64)`` below M 256, and ``(256, 128)``
    on a 2-SM cluster from M 256.
    Warm up outside CUDA graph capture to compile the selected kernel.
    """
    _check_sm100(act)
    for tensor, name, alignment in (
        (act, "act", 32),
        (weight, "weight", 32),
        (act_scale, "act_scale", 16),
        (weight_scale, "weight_scale", 16),
    ):
        if name in ("act_scale", "weight_scale"):
            continue
        _check_tensor(
            tensor,
            name,
            act.device,
            (torch.uint8,),
            alignment,
            contiguous=name != "weight",
        )
    _check_tensor(alpha, "alpha", act.device, (torch.float32,), 4)
    if act.ndim != 2 or weight.ndim != 2:
        raise ValueError("act and weight must be matrices")
    m2, kp = act.shape
    n, wk = weight.shape
    if min(m2, kp, n) <= 0 or m2 % 2 or kp != wk or kp % 32 or n % 8:
        raise ValueError(
            "require positive dimensions, paired rows, matching K divisible by 64, N divisible by 8"
        )
    if weight.stride(1) != 1 or weight.stride(0) < kp or weight.stride(0) % 32:
        raise ValueError(
            "weight requires nonoverlapping K-contiguous rows with 32B-aligned stride"
        )
    _check_swizzle(act_scale, m2, 2 * kp, "act_scale", act.device)
    _check_swizzle(weight_scale, n, 2 * kp, "weight_scale", act.device)
    if alpha.numel() != 1 or out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("alpha must have one element; output must be FP16/BF16")
    from .experimental.resserve_mext.host import (
        mext_fold_gemm,
        select_row_pair_tactic,
    )

    with torch.cuda.device(act.device):
        tiler, cluster = select_row_pair_tactic(m2 // 2)
        return mext_fold_gemm(
            weight,
            act,
            weight_scale,
            act_scale,
            alpha,
            out_dtype=out_dtype,
            mma_tiler_mn=tiler,
            cluster_shape_mn=cluster,
        )


@flashinfer_experimental_api
def paged_decode_residue(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float | None = None,
    key_scale: float = 1.0,
    value_scale: float = 1.0,
    max_seq_len: int | None = None,
    num_splits: int | None = None,
    out: torch.Tensor | None = None,
):
    """Single-token page32 NVFP4 attention with Q residue and BF16 output.

    SM100; original BF16 ``q[B, H_q, D]``, D128/D256 and GQA4/6/8.
    K/V are packed uint8 ``[pages, H_kv, 32, D/2]``. K scales are E4M3
    (or uint8 bytes) ``[pages, H_kv, 32, D/16]``. V scales are four-token
    interleaved ``[pages, H_kv, 8, D/16, 4]``; prepare this layout when
    writing KV. K/V data may use HND or NHD storage, exposed as the HND-shaped
    views above. Scales have contiguous inner page contents; page strides may
    include other cache buffers. No KV data or scale repacking is performed.

    ``block_tables`` is int32 ``[B, max_pages]`` with unit column stride;
    padded row strides are accepted. ``seq_lens`` is contiguous int32 ``[B]``.
    Lengths must be nonnegative and at most 32*max_pages; used page IDs must
    be valid. Zero-length rows are Graph padding and their outputs are ignored.
    These device values are caller preconditions, not synchronized to the CPU.
    Q must be finite; key/value scales are positive finite reconstruction
    multipliers prepared before launch.

    ``max_seq_len`` is an optional host upper bound for static split selection;
    the default is the table capacity. ``num_splits=None`` uses ResServe's
    existing static policy; a power of two through 64 requests deterministic
    FP32-partial reduction (one split needs no reducer). No timing/autotuning
    is performed. ``out`` may supply a contiguous BF16 tensor shaped like Q.
    Scratch is invocation-local, including under CUDA Graph capture.

    FP4 QK folds base/residue before softmax. QMUL4 converts V to E4M3;
    unnormalized probabilities are rounded to E4M3 for PV with an FP32
    denominator. This is not full BF16 attention. Requires CUTLASS DSL 4.8
    with CUDA 13.4 PTX support; warm up outside graph capture.
    """
    import math

    _check_sm100(q)
    device, q_shape, q_stride = q.device, q.shape, q.stride()
    _check_tensor(q, "q", device, (torch.bfloat16,), 32, contiguous=False)
    if len(q_shape) != 3 or min(q_shape) <= 0 or q_shape[-1] not in (128, 256):
        raise ValueError("q must be positive [batch, query_heads, 128 or 256]")
    b, hq, d = q_shape
    if q_stride[2] != 1 or q_stride[1] != d or q_stride[0] < hq * d:
        raise ValueError("q requires nonoverlapping rows with contiguous heads")
    k_shape = k.shape
    if len(k_shape) != 4 or min(k_shape) <= 0:
        raise ValueError("k must be positive [pages, kv_heads, 32, D/2]")
    pages, h = k_shape[:2]
    if hq % h or hq // h not in (4, 6, 8):
        raise ValueError("ResAttn requires GQA4, GQA6 or GQA8")
    data_dim, scale_dim = d // 2, d // 16
    data_strides = ((32 * data_dim, data_dim, 1), (data_dim, h * data_dim, 1))
    for tensor, name, shape, layouts, dtype in (
        (k, "k", (pages, h, 32, data_dim), data_strides, (torch.uint8,)),
        (v, "v", (pages, h, 32, data_dim), data_strides, (torch.uint8,)),
        (
            k_scale,
            "k_scale",
            (pages, h, 32, scale_dim),
            ((32 * scale_dim, scale_dim, 1),),
            (torch.uint8, torch.float8_e4m3fn),
        ),
        (
            v_scale,
            "v_scale",
            (pages, h, 8, scale_dim, 4),
            ((32 * scale_dim, 4 * scale_dim, 4, 1),),
            (torch.uint8, torch.float8_e4m3fn),
        ),
    ):
        _check_tensor(tensor, name, device, dtype, contiguous=False)
        if tensor.shape != shape:
            raise ValueError(
                f"{name} must have shape {shape}; V scales are four-token interleaved"
            )
        strides = tensor.stride()
        # Singleton head strides are immaterial in both HND and NHD views.
        inner = strides[1:] if h > 1 else strides[2:]
        if inner not in (layouts if h > 1 else tuple(layout[1:] for layout in layouts)):
            raise ValueError(f"{name} requires contiguous inner page contents")
        page_bytes = h * 32 * (data_dim if name in ("k", "v") else scale_dim)
        if strides[0] < page_bytes or strides[0] % 16:
            raise ValueError(f"{name} requires nonoverlapping, 16-byte-aligned pages")
    _check_tensor(
        block_tables, "block_tables", device, (torch.int32,), 4, contiguous=False
    )
    _check_tensor(seq_lens, "seq_lens", device, (torch.int32,), 4)
    table_shape, table_stride = block_tables.shape, block_tables.stride()
    if (
        len(table_shape) != 2
        or table_shape[0] != b
        or table_shape[1] <= 0
        or table_stride[1] != 1
        or table_stride[0] < table_shape[1]
    ):
        raise ValueError(
            "block_tables requires nonoverlapping rows and unit column stride"
        )
    if tuple(seq_lens.shape) != (b,):
        raise ValueError("seq_lens must be [batch]")
    if num_splits is not None and (
        type(num_splits) is not int
        or num_splits < 1
        or num_splits > 64
        or num_splits & (num_splits - 1)
    ):
        raise ValueError("num_splits must be a power of two from 1 through 64")
    capacity = 32 * table_shape[1]
    max_seq_len = capacity if max_seq_len is None else max_seq_len
    if type(max_seq_len) is not int or not 0 < max_seq_len <= capacity:
        raise ValueError(
            "max_seq_len must be a positive host bound within table capacity"
        )
    sm_scale = d**-0.5 if sm_scale is None else float(sm_scale)
    if not all(math.isfinite(x) and x > 0 for x in (sm_scale, key_scale, value_scale)):
        raise ValueError(
            "attention and reconstruction scales must be positive finite scalars"
        )
    # The kernel uses FP32 score_scale * log2(e) and FP32 value_scale.
    scales = torch.tensor(
        [sm_scale * key_scale, value_scale], dtype=torch.float32, device="cpu"
    )
    scales[0] *= math.log2(math.e)
    if not (torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError(
            "sm_scale * key_scale and value_scale must be positive finite FP32 values"
        )
    if out is not None:
        _check_tensor(out, "out", device, (torch.bfloat16,))
        if out.shape != q_shape:
            raise ValueError("out must have Q's shape")
    from .experimental.resserve_resattn.host import decode

    with torch.cuda.device(device.index):
        return decode(
            q,
            k,
            v,
            k_scale,
            v_scale,
            block_tables,
            seq_lens,
            sm_scale,
            key_scale,
            value_scale,
            num_splits,
            max_seq_len,
            out,
        )
