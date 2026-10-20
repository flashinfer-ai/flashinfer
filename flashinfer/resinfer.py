# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
"""Experimental ResInfer quantization, folded GEMM and paged decode APIs.

Ported from https://github.com/ShawRong/resinfer at
51b906836a69fa43915040314df57bc9f23030ad. See the backend README for lineage.

Import public entry points from this module. Each call explicitly opts into
an experimental API and lazily imports its backend. Implementation modules
under ``flashinfer.experimental.resinfer_*`` are internal. These APIs remain
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


def _check_tensor(tensor, name, device, dtype, alignment=16):
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tensor.dtype not in dtype:
        raise ValueError(f"{name} has unsupported dtype {tensor.dtype}")
    if not tensor.is_contiguous() or tensor.data_ptr() % alignment:
        raise ValueError(f"{name} must be contiguous and {alignment}-byte aligned")


def _check_sm100(tensor):
    if not tensor.is_cuda:
        raise ValueError("ResInfer requires CUDA tensors")
    if get_compute_capability(tensor.device) != (10, 0):
        raise ValueError("ResInfer currently supports SM100 only")


@flashinfer_experimental_api
def nvfp4_quantize_kext(
    input: torch.Tensor,
    global_scale: torch.Tensor,
    channel_masks: torch.Tensor,
    residue_per_8: int,
    *,
    elts_per_thread: int = 0,
):
    """Append selected-channel NVFP4 residues along K (ResInfer KEXT).

    SM100; contiguous FP16/BF16 ``input[M, K]``, positive M and K, K divisible
    by 256. ``global_scale`` is one positive finite FP32 quantization multiplier
    on the input device. Inputs must be finite.

    ``channel_masks`` is contiguous uint8 ``[K/8]`` on the same device. Every
    byte must have exactly ``residue_per_8`` set bits, selecting that group's
    channels in ascending bit order. ``residue_per_8`` is 1, 2 or 4 (12.5%,
    25% or 50% extension). Mask contents and scale values are caller
    preconditions, not checked with a synchronizing device-to-host copy.

    Returns packed uint8 E2M1 ``[M, K_ext/2]`` (base then selected residue)
    and flat uint8 E4M3 scales in the 128-row, 4-scale swizzle, with zero
    padding. Extended weights must append the same selected columns in
    ascending channel order *before* weight quantization. Use ordinary
    ``flashinfer.mm_fp4`` with reconstruction alpha equal to the reciprocal
    product of activation and weight global multipliers. Its backend-specific
    K alignment/padding requirements still apply.

    ``elts_per_thread=0`` preserves ResInfer's SM100 pack8/pack16 shape policy;
    8 or 16 explicitly selects one producer. This is not GEMM autotuning.
    Warm up before CUDA graph capture. Swap and ratio-1 MEXT are separate.
    """
    _check_sm100(input)
    _check_tensor(input, "input", input.device, (torch.float16, torch.bfloat16), 32)
    _check_tensor(global_scale, "global_scale", input.device, (torch.float32,), 4)
    _check_tensor(channel_masks, "channel_masks", input.device, (torch.uint8,), 1)
    if input.ndim != 2 or min(input.shape) <= 0 or input.shape[1] % 256:
        raise ValueError("input must be [M, K], M > 0, K > 0 and K divisible by 256")
    if global_scale.numel() != 1:
        raise ValueError("global_scale must contain one element")
    if channel_masks.ndim != 1 or channel_masks.numel() != input.shape[1] // 8:
        raise ValueError("channel_masks must have shape [K/8]")
    if residue_per_8 not in (1, 2, 4) or isinstance(residue_per_8, bool):
        raise ValueError("residue_per_8 must be 1, 2 or 4")
    if elts_per_thread not in (0, 8, 16):
        raise ValueError("elts_per_thread must be 0, 8 or 16")
    from .experimental.resinfer_kext.quantization import quantize

    with torch.cuda.device(input.device):
        return quantize(
            input, global_scale, channel_masks, residue_per_8, elts_per_thread
        )


@flashinfer_experimental_api
def nvfp4_quantize_residue(input: torch.Tensor, global_scale: torch.Tensor):
    """Quantize each activation row and its full-channel residue on SM100.

    ``input`` is contiguous FP16/BF16 ``[M, K]``, with positive M and K,
    K divisible by 16. ``global_scale`` contains one or two positive finite
    FP32 quantization multipliers on the same device: base, optionally residue.
    Scale values and finite input values are caller preconditions (not checked
    with a synchronizing device-to-host copy).

    Returns uint8 packed E2M1 ``[2*M, K/2]`` in adjacent base/residue row order
    and a flat uint8 E4M3 scale buffer in the 128-row, 4-scale swizzle layout.
    Padding scale entries are zero. The residue is computed from the quantized
    and reconstructed base, following ResInfer's production pack16 algorithm.
    """
    _check_sm100(input)
    _check_tensor(input, "input", input.device, (torch.float16, torch.bfloat16), 32)
    _check_tensor(global_scale, "global_scale", input.device, (torch.float32,), 4)
    if input.ndim != 2 or min(input.shape) <= 0 or input.shape[1] % 16:
        raise ValueError("input must be [M, K], M > 0, K > 0 and K divisible by 16")
    if global_scale.numel() not in (1, 2):
        raise ValueError("global_scale must contain one or two elements")
    from .experimental.resinfer_mext.quantization import quantize

    with torch.cuda.device(input.device):
        return quantize(input, global_scale)


@flashinfer_experimental_api
def mm_nvfp4_residue(
    act: torch.Tensor,
    weight: torch.Tensor,
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
):
    """Multiply ResInfer row-pair activations by shared weights and fold in FP32.

    ``act`` is packed uint8 ``[2*M, K/2]`` from ``nvfp4_quantize_residue``;
    ``weight`` is contiguous packed uint8 ``[N, K/2]``. Both scale buffers use
    the 128-row, 4-scale swizzle (E4M3 bytes). K is divisible by 64, N by 8.
    ``alpha`` is one or two FP32 reconstruction multipliers: reciprocals of
    the product of activation and weight quantization global multipliers.
    Two values apply independently to the base and residue accumulators.
    The output is ``[M, N]`` in FP16 or BF16. Weight quantization is external.
    Warm up outside CUDA graph capture to compile the selected kernel.
    """
    _check_sm100(act)
    for tensor, name, alignment in (
        (act, "act", 32),
        (weight, "weight", 32),
        (act_scale, "act_scale", 16),
        (weight_scale, "weight_scale", 16),
    ):
        _check_tensor(tensor, name, act.device, (torch.uint8,), alignment)
    _check_tensor(alpha, "alpha", act.device, (torch.float32,), 4)
    if act.ndim != 2 or weight.ndim != 2:
        raise ValueError("act and weight must be matrices")
    m2, kp = act.shape
    n, wk = weight.shape
    if min(m2, kp, n) <= 0 or m2 % 2 or kp != wk or kp % 32 or n % 8:
        raise ValueError(
            "require positive dimensions, paired rows, matching K divisible by 64, N divisible by 8"
        )
    for scale, rows, name in (
        (act_scale, m2, "act_scale"),
        (weight_scale, n, "weight_scale"),
    ):
        expected = ((rows + 127) // 128) * ((2 * kp + 63) // 64) * 512
        if scale.numel() != expected:
            raise ValueError(f"{name} must contain {expected} bytes")
    if alpha.numel() not in (1, 2) or out_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("alpha must have one/two elements; output must be FP16/BF16")
    from .experimental.resinfer_mext.host import mext_fold_gemm

    with torch.cuda.device(act.device):
        return mext_fold_gemm(
            weight, act, weight_scale, act_scale, alpha, out_dtype=out_dtype
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
    """ResInfer single-token paged NVFP4 attention with BF16 output (SM100).

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
    from .experimental.resinfer_resattn.backend import decode

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
