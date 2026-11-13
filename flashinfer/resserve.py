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
    Prepare this global multiplier before launch and keep it unchanged until
    the operation completes; asynchronous alpha production via PDL is unsupported.
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
    num_splits: int = 1,
):
    """ResServe single-token paged NVFP4 attention with BF16 output (SM100).

    Q is contiguous BF16 ``[batch, query_heads, D]``. K/V are packed uint8
    ``[pages, kv_heads, 128, D/2]``. K scales are contiguous E4M3 (or uint8
    bytes) ``[pages, kv_heads, 128, D/16]``. V scales use the same dtype but
    four-token interleaving: ``[pages, kv_heads, 32, D/16, 4]``, indexed by
    page, head, token//4, channel//16, token%4. Prepare that layout when
    writing the cache; this call does not reorder scale contents. D is 128/256,
    and the query/KV head ratio is 4, 6 or 8. ``key_scale`` and ``value_scale``
    are positive finite reconstruction multipliers for the prequantized cache.

    ``block_tables`` is contiguous int32 ``[batch, max_pages]``; ``seq_lens``
    is contiguous int32 ``[batch]``. The caller must supply positive lengths
    no greater than ``128 * max_pages`` and valid page indices for every
    used page. Values are not copied to the CPU for validation. Q must be finite.

    Q is quantized into base/residue groups and the two QK contributions are
    folded before softmax. V is converted to E4M3 by QMUL4; unnormalized
    probabilities are rounded to E4M3 for PV, with an FP32 denominator. Split count 1
    uses no reducer; 2 uses a deterministic FP32-partial kernel reducer.
    Requires CUTLASS DSL 4.8 with its CUDA 13.4 PTX compiler. Warm up before
    graph capture. No implicit auto-backend selection is performed.
    """
    import math

    _check_sm100(q)
    _check_tensor(q, "q", q.device, (torch.bfloat16,), 32)
    if q.ndim != 3 or min(q.shape) <= 0 or q.shape[-1] not in (128, 256):
        raise ValueError("q must be positive [batch, query_heads, 128 or 256]")
    b, hq, d = q.shape
    if k.ndim != 4 or min(k.shape) <= 0:
        raise ValueError("k must be positive [pages, kv_heads, 128, D/2]")
    pages, h = k.shape[:2]
    if hq % h or hq // h not in (4, 6, 8):
        raise ValueError("ResAttn requires GQA4, GQA6 or GQA8")
    for tensor, name in ((k, "k"), (v, "v")):
        _check_tensor(tensor, name, q.device, (torch.uint8,), 16)
        if tuple(tensor.shape) != (pages, h, 128, d // 2):
            raise ValueError(f"{name} must be [pages, kv_heads, 128, D/2]")
    for tensor, name in ((k_scale, "k_scale"), (v_scale, "v_scale")):
        _check_tensor(tensor, name, q.device, (torch.uint8, torch.float8_e4m3fn))
    if tuple(k_scale.shape) != (pages, h, 128, d // 16):
        raise ValueError("k_scale must be [pages, kv_heads, 128, D/16]")
    if tuple(v_scale.shape) != (pages, h, 32, d // 16, 4):
        raise ValueError(
            "v_scale must be four-token interleaved [pages, kv_heads, 32, D/16, 4]"
        )
    for tensor, name in ((block_tables, "block_tables"), (seq_lens, "seq_lens")):
        _check_tensor(tensor, name, q.device, (torch.int32,), 4)
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != b
        or block_tables.shape[1] <= 0
    ):
        raise ValueError("block_tables must be [batch, max_pages > 0]")
    if tuple(seq_lens.shape) != (b,):
        raise ValueError("seq_lens must be [batch]")
    if type(num_splits) is not int or num_splits not in (1, 2):
        raise ValueError("this experimental API exposes num_splits=1 or 2")
    sm_scale = d**-0.5 if sm_scale is None else float(sm_scale)
    if not all(math.isfinite(x) and x > 0 for x in (sm_scale, key_scale, value_scale)):
        raise ValueError(
            "attention and cache reconstruction scales must be positive finite scalars"
        )
    from .experimental.resserve_resattn.host import decode

    with torch.cuda.device(q.device):
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
        )
