# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-Torch MXFP8-weight to BF16 transform reference.

This module defines the numerical contract used before the dense BF16 MMA:

.. code-block:: text

    BF16(decode_fp8(weight) * decode_e8m0(weight_scale))

One E8M0 scale covers exactly 32 consecutive elements along the K dimension.
The conversion to BF16 is deliberately the final operation: callers must feed
the returned BF16 tensor to the BF16 GEMM reference instead of multiplying the
decoded operands directly in FP32.

The implementation depends only on PyTorch and runs on either CPU or CUDA.
"""

from __future__ import annotations

from typing import Any, Callable, Final, Literal, Optional

import torch


MXFP8_BLOCK_SIZE: Final[int] = 32
# Phase-2 v1 deliberately retains the swap-AB epilogue's 16-column
# gate/up pairing.  This is independent of MXFP8's K32 scale granularity.
MXFP8_GATE_UP_INTERLEAVE: Final[int] = 16
_SF_ATOM_ROWS: Final[int] = 128
_SF_ATOM_COLS: Final[int] = 4

_E4M3_DTYPE = getattr(torch, "float8_e4m3fn", None)
_E5M2_DTYPE = getattr(torch, "float8_e5m2", None)
_E8M0_DTYPE = getattr(torch, "float8_e8m0fnu", None)
_SUPPORTED_WEIGHT_DTYPES = tuple(
    dtype for dtype in (_E4M3_DTYPE, _E5M2_DTYPE) if dtype is not None
)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _swizzled_scale_numel(*, output_size: int, reduction_size: int) -> int:
    """Return the byte count of one public 32x4x4 SF plane."""

    raw_scale_cols = reduction_size // MXFP8_BLOCK_SIZE
    padded_rows = _ceil_div(output_size, _SF_ATOM_ROWS) * _SF_ATOM_ROWS
    padded_cols = _ceil_div(raw_scale_cols, _SF_ATOM_COLS) * _SF_ATOM_COLS
    return padded_rows * padded_cols


def _from_blocked_scale(
    flat: torch.Tensor,
    raw_rows: int,
    raw_cols: int,
) -> torch.Tensor:
    """Pure-Torch inverse of the repository's public 32x4x4 SF swizzle.

    Keeping this small inverse local is intentional: importing the runner
    helper also imports the CuTeDSL host stack, which makes this otherwise
    CPU-only numerical module initialize CUDA-facing dependencies.
    """

    if flat.ndim != 1:
        raise ValueError(f"expected a flat SF plane, got {flat.ndim}D.")
    if raw_rows <= 0 or raw_cols <= 0:
        raise ValueError(
            f"raw SF extents must be positive, got ({raw_rows}, {raw_cols})."
        )

    row_blocks = _ceil_div(raw_rows, _SF_ATOM_ROWS)
    col_blocks = _ceil_div(raw_cols, _SF_ATOM_COLS)
    padded_rows = row_blocks * _SF_ATOM_ROWS
    padded_cols = col_blocks * _SF_ATOM_COLS
    expected = padded_rows * padded_cols
    if flat.numel() != expected:
        raise ValueError(
            f"swizzled SF plane has {flat.numel()} elements; expected "
            f"{expected} for raw shape ({raw_rows}, {raw_cols}) padded to "
            f"({padded_rows}, {padded_cols})."
        )

    rearranged = flat.reshape(-1, 32, 16).reshape(-1, 32, 4, 4)
    blocks = rearranged.transpose(1, 2).reshape(-1, _SF_ATOM_ROWS, 4)
    blocks = blocks.reshape(
        row_blocks,
        col_blocks,
        _SF_ATOM_ROWS,
        4,
    )
    padded = blocks.permute(0, 2, 1, 3).reshape(
        padded_rows,
        padded_cols,
    )
    return padded[:raw_rows, :raw_cols].contiguous()


def decode_e8m0(scale: torch.Tensor) -> torch.Tensor:
    """Decode an E8M0 tensor to FP32 without changing its device or shape.

    Finite byte ``b`` represents ``2 ** (b - 127)``. Byte ``0xFF`` is the
    format's sole NaN encoding and is returned as FP32 NaN. In particular,
    ``0x00`` is the finite value ``2**-127`` rather than zero.
    """

    if not isinstance(scale, torch.Tensor):
        raise TypeError(
            f"scale must be a torch.Tensor, got {type(scale).__name__}."
        )
    if _E8M0_DTYPE is None:
        raise RuntimeError(
            "this PyTorch build does not provide torch.float8_e8m0fnu."
        )
    if scale.dtype is not _E8M0_DTYPE:
        raise TypeError(
            "scale must have dtype torch.float8_e8m0fnu, "
            f"got {scale.dtype}."
        )

    # Reinterpret the native one-byte storage so 0xFF can be handled
    # explicitly.  ldexp is exact for every finite E8M0 power of two.
    raw = scale.view(torch.uint8)
    exponent = raw.to(torch.int16) - 127
    decoded = torch.ldexp(
        torch.ones(raw.shape, dtype=torch.float32, device=raw.device),
        exponent.to(torch.int32),
    )
    return decoded.masked_fill(raw == 0xFF, float("nan"))


def mxfp8_weight_to_bf16(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    k_dim: int = -1,
    block_size: int = MXFP8_BLOCK_SIZE,
) -> torch.Tensor:
    """Transform an MXFP8 weight tensor to the BF16 Tensor-Core operand.

    Args:
        weight:
            E4M3FN or E5M2 tensor. It may have arbitrary leading dimensions.
        weight_scale:
            E8M0 tensor with the same rank as ``weight``. Its shape must equal
            ``weight.shape`` with the K extent replaced by ``K // 32``.
        k_dim:
            Dimension of ``weight`` covered by the per-32 scales.
        block_size:
            Exposed to make the ABI check explicit; MXFP8 requires exactly 32.

    Returns:
        A BF16 tensor on the input device with the same shape as ``weight``.

    NaN and infinity follow IEEE propagation. Thus an E8M0 ``0xFF`` scale
    produces NaN for every element in its 32-value block, including zero
    weights.
    """

    if not isinstance(weight, torch.Tensor):
        raise TypeError(
            f"weight must be a torch.Tensor, got {type(weight).__name__}."
        )
    if not isinstance(weight_scale, torch.Tensor):
        raise TypeError(
            "weight_scale must be a torch.Tensor, "
            f"got {type(weight_scale).__name__}."
        )
    if weight.dtype not in _SUPPORTED_WEIGHT_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_WEIGHT_DTYPES)
        raise TypeError(
            f"weight must have MXFP8 dtype ({supported}), got {weight.dtype}."
        )
    if _E8M0_DTYPE is None:
        raise RuntimeError(
            "this PyTorch build does not provide torch.float8_e8m0fnu."
        )
    if weight_scale.dtype is not _E8M0_DTYPE:
        raise TypeError(
            "weight_scale must have dtype torch.float8_e8m0fnu, "
            f"got {weight_scale.dtype}."
        )
    if weight.device != weight_scale.device:
        raise ValueError(
            "weight and weight_scale must be on the same device, got "
            f"{weight.device} and {weight_scale.device}."
        )
    if block_size != MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"MXFP8 block_size must be {MXFP8_BLOCK_SIZE}, got {block_size}."
        )
    if weight.ndim == 0:
        raise ValueError("weight must have at least one dimension.")
    if weight_scale.ndim != weight.ndim:
        raise ValueError(
            "weight_scale must have the same rank as weight, got "
            f"{weight_scale.ndim} and {weight.ndim}."
        )
    if isinstance(k_dim, bool) or not isinstance(k_dim, int):
        raise TypeError(f"k_dim must be an int, got {type(k_dim).__name__}.")
    if not -weight.ndim <= k_dim < weight.ndim:
        raise ValueError(
            f"k_dim={k_dim} is out of range for a rank-{weight.ndim} tensor."
        )

    normalized_k_dim = k_dim % weight.ndim
    k = weight.shape[normalized_k_dim]
    if k <= 0 or k % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"weight K extent must be a positive multiple of "
            f"{MXFP8_BLOCK_SIZE}, got {k}."
        )

    expected_scale_shape = list(weight.shape)
    expected_scale_shape[normalized_k_dim] = k // MXFP8_BLOCK_SIZE
    if tuple(weight_scale.shape) != tuple(expected_scale_shape):
        raise ValueError(
            "weight_scale shape must equal weight.shape with K replaced by "
            f"K/{MXFP8_BLOCK_SIZE}; expected {tuple(expected_scale_shape)}, "
            f"got {tuple(weight_scale.shape)}."
        )

    decoded_weight = weight.to(torch.float32)
    decoded_scale = decode_e8m0(weight_scale)
    expanded_scale = decoded_scale.repeat_interleave(
        MXFP8_BLOCK_SIZE, dim=normalized_k_dim
    )

    # This cast is a required numerical boundary, not merely an output-format
    # convenience. The dense reference must consume this rounded BF16 tensor.
    return (decoded_weight * expanded_scale).to(torch.bfloat16)


def mxfp8_weight_from_swizzled_to_bf16(
    weight_kn: torch.Tensor,
    weight_scale_swizzled: torch.Tensor,
) -> torch.Tensor:
    """Transform one lean-runner weight from its public host ABI to BF16.

    The lean fused-FC12 host contract stores a weight as a K-major ``(K, N)``
    view, while the scale-factor bytes use the repository's existing 32x4x4
    atom-swizzled ABI.  This helper reverses both host-side views and returns
    the logical dense-MMA operand ``(N, K)``:

    .. code-block:: text

        weight_kn (K, N) --transpose--> weight_nk (N, K)
        swizzled E8M0   --from_blocked--> raw scale (N, K/32)
        result = BF16(FP8(weight_nk) * E8M0(scale))

    ``weight_scale_swizzled`` may be either the flat per-expert buffer exposed
    by the runner or any contiguous view containing exactly the same bytes.
    Padding implied by the 32x4x4 atom is removed before the numerical
    transform, so the returned tensor contains no padded rows or columns.
    """

    if not isinstance(weight_kn, torch.Tensor):
        raise TypeError(
            f"weight_kn must be a torch.Tensor, got {type(weight_kn).__name__}."
        )
    if not isinstance(weight_scale_swizzled, torch.Tensor):
        raise TypeError(
            "weight_scale_swizzled must be a torch.Tensor, "
            f"got {type(weight_scale_swizzled).__name__}."
        )
    if weight_kn.ndim != 2:
        raise ValueError(
            f"weight_kn must be rank 2 with shape (K, N), got {weight_kn.ndim}D."
        )
    if weight_scale_swizzled.device != weight_kn.device:
        raise ValueError(
            "weight_kn and weight_scale_swizzled must be on the same device, "
            f"got {weight_kn.device} and {weight_scale_swizzled.device}."
        )

    reduction_size, output_size = weight_kn.shape
    if reduction_size <= 0 or reduction_size % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"weight_kn K extent must be a positive multiple of "
            f"{MXFP8_BLOCK_SIZE}, got {reduction_size}."
        )
    if output_size <= 0:
        raise ValueError(
            f"weight_kn N extent must be positive, got {output_size}."
        )

    scale_flat = weight_scale_swizzled.contiguous().view(-1)
    raw_scale = _from_blocked_scale(
        scale_flat,
        output_size,
        reduction_size // MXFP8_BLOCK_SIZE,
    )
    return mxfp8_weight_to_bf16(
        weight_kn.transpose(0, 1),
        raw_scale,
    )


def _load_bf16_megamoe_reference() -> Callable[..., Any]:
    """Load the CUDA-backed BF16 MegaMoE reference only when it is needed."""

    # ``mega_reference_bf16`` owns a CuTeDSL dense-GEMM launcher and imports
    # CUDA bindings at module scope.  A local import keeps transform/ABI tests
    # CPU-only; those tests monkeypatch this loader with a recording callable.
    from moe_bf16_glu.mega_reference_bf16 import compute_megamoe_reference

    return compute_megamoe_reference


def _validate_megamoe_public_abi(
    *,
    input_activation: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
) -> tuple[int, int, int, int]:
    """Validate the mixed public tensor ABI.

    Weight tensors retain the kernel-facing K-major views:

    * FC1 weight ``(R, E, H, I)`` with the ``H`` axis stride one;
    * FC2 weight ``(R, E, I/2, H)`` with the ``I/2`` axis stride one.

    Each scale tensor is one atom-swizzled, byte-flat plane per rank/expert:
    ``(R, E, S)``.  Its logical unswizzled shape is ``(I, H/32)`` for FC1
    and ``(H, (I/2)/32)`` for FC2.  The flat extent includes the public
    32x4x4 atom's effective 128-row by 4-column padding.
    """

    tensors = {
        "input_activation": input_activation,
        "input_topk_idx": input_topk_idx,
        "input_topk_weights": input_topk_weights,
        "fc1_weight": fc1_weight,
        "fc1_weight_sf": fc1_weight_sf,
        "fc2_weight": fc2_weight,
        "fc2_weight_sf": fc2_weight_sf,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"{name} must be a torch.Tensor, got "
                f"{type(tensor).__name__}."
            )

    if input_activation.ndim != 3:
        raise ValueError(
            "input_activation must have shape (num_ranks, tokens, hidden), "
            f"got {tuple(input_activation.shape)}."
        )
    if input_activation.dtype is not torch.bfloat16:
        raise TypeError(
            "input_activation must have dtype torch.bfloat16, got "
            f"{input_activation.dtype}."
        )
    num_ranks, num_tokens_per_rank, hidden = input_activation.shape
    if min(num_ranks, num_tokens_per_rank, hidden) <= 0:
        raise ValueError(
            "input_activation extents must be positive, got "
            f"{tuple(input_activation.shape)}."
        )

    if input_topk_idx.ndim != 3:
        raise ValueError(
            "input_topk_idx must have shape (num_ranks, tokens, topk), got "
            f"{tuple(input_topk_idx.shape)}."
        )
    if input_topk_idx.dtype is not torch.int64:
        raise TypeError(
            "input_topk_idx must have dtype torch.int64, got "
            f"{input_topk_idx.dtype}."
        )
    if tuple(input_topk_idx.shape[:2]) != (
        num_ranks,
        num_tokens_per_rank,
    ):
        raise ValueError(
            "input_topk_idx leading shape must match input_activation; "
            f"expected ({num_ranks}, {num_tokens_per_rank}, topk), got "
            f"{tuple(input_topk_idx.shape)}."
        )
    if input_topk_idx.shape[2] <= 0:
        raise ValueError("input_topk_idx topk extent must be positive.")

    if tuple(input_topk_weights.shape) != tuple(input_topk_idx.shape):
        raise ValueError(
            "input_topk_weights must have the same shape as input_topk_idx; "
            f"expected {tuple(input_topk_idx.shape)}, got "
            f"{tuple(input_topk_weights.shape)}."
        )
    if input_topk_weights.dtype is not torch.float32:
        raise TypeError(
            "input_topk_weights must have dtype torch.float32, got "
            f"{input_topk_weights.dtype}."
        )

    if fc1_weight.ndim != 4:
        raise ValueError(
            "fc1_weight must have public shape (R, E, H, I), got "
            f"{tuple(fc1_weight.shape)}."
        )
    if fc1_weight.shape[0] != num_ranks:
        raise ValueError(
            f"fc1_weight rank extent must be {num_ranks}, got "
            f"{fc1_weight.shape[0]}."
        )
    num_experts_per_rank = fc1_weight.shape[1]
    intermediate = fc1_weight.shape[3]
    if num_experts_per_rank <= 0 or intermediate <= 0:
        raise ValueError(
            "fc1_weight expert and intermediate extents must be positive, got "
            f"{tuple(fc1_weight.shape)}."
        )
    expected_fc1_shape = (
        num_ranks,
        num_experts_per_rank,
        hidden,
        intermediate,
    )
    if tuple(fc1_weight.shape) != expected_fc1_shape:
        raise ValueError(
            f"fc1_weight must have public shape {expected_fc1_shape}, got "
            f"{tuple(fc1_weight.shape)}."
        )
    if fc1_weight.stride(2) != 1:
        raise ValueError(
            "fc1_weight public ABI requires the H/K axis (dimension 2) to "
            f"have stride 1, got strides {fc1_weight.stride()}."
        )

    if intermediate % 2 != 0:
        raise ValueError(
            f"fc1 intermediate extent must be even, got {intermediate}."
        )
    intermediate_downproj = intermediate // 2
    expected_fc2_shape = (
        num_ranks,
        num_experts_per_rank,
        intermediate_downproj,
        hidden,
    )
    if fc2_weight.ndim != 4 or tuple(fc2_weight.shape) != expected_fc2_shape:
        raise ValueError(
            f"fc2_weight must have public shape {expected_fc2_shape}, got "
            f"{tuple(fc2_weight.shape)}."
        )
    if fc2_weight.stride(2) != 1:
        raise ValueError(
            "fc2_weight public ABI requires the I/2/K axis (dimension 2) to "
            f"have stride 1, got strides {fc2_weight.stride()}."
        )

    if fc1_weight.dtype not in _SUPPORTED_WEIGHT_DTYPES:
        supported = ", ".join(str(dtype) for dtype in _SUPPORTED_WEIGHT_DTYPES)
        raise TypeError(
            f"fc1_weight must have MXFP8 dtype ({supported}), got "
            f"{fc1_weight.dtype}."
        )
    if fc2_weight.dtype is not fc1_weight.dtype:
        raise TypeError(
            "fc2_weight must have the same MXFP8 dtype as fc1_weight, got "
            f"{fc2_weight.dtype} and {fc1_weight.dtype}."
        )

    if hidden % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"hidden must be a multiple of {MXFP8_BLOCK_SIZE}, got {hidden}."
        )
    if intermediate_downproj % MXFP8_BLOCK_SIZE != 0:
        raise ValueError(
            f"intermediate/2 must be a multiple of {MXFP8_BLOCK_SIZE}, got "
            f"{intermediate_downproj}."
        )

    expected_fc1_sf_size = _swizzled_scale_numel(
        output_size=intermediate,
        reduction_size=hidden,
    )
    expected_fc2_sf_size = _swizzled_scale_numel(
        output_size=hidden,
        reduction_size=intermediate_downproj,
    )
    expected_fc1_sf_shape = (
        num_ranks,
        num_experts_per_rank,
        expected_fc1_sf_size,
    )
    expected_fc2_sf_shape = (
        num_ranks,
        num_experts_per_rank,
        expected_fc2_sf_size,
    )
    if fc1_weight_sf.ndim != 3 or tuple(fc1_weight_sf.shape) != (
        expected_fc1_sf_shape
    ):
        raise ValueError(
            "fc1_weight_sf must contain one flat atom-swizzled plane per "
            f"rank/expert with shape {expected_fc1_sf_shape}, got "
            f"{tuple(fc1_weight_sf.shape)}."
        )
    if fc2_weight_sf.ndim != 3 or tuple(fc2_weight_sf.shape) != (
        expected_fc2_sf_shape
    ):
        raise ValueError(
            "fc2_weight_sf must contain one flat atom-swizzled plane per "
            f"rank/expert with shape {expected_fc2_sf_shape}, got "
            f"{tuple(fc2_weight_sf.shape)}."
        )
    if _E8M0_DTYPE is None:
        raise RuntimeError(
            "this PyTorch build does not provide torch.float8_e8m0fnu."
        )
    if fc1_weight_sf.dtype is not _E8M0_DTYPE:
        raise TypeError(
            "fc1_weight_sf must have dtype torch.float8_e8m0fnu, got "
            f"{fc1_weight_sf.dtype}."
        )
    if fc2_weight_sf.dtype is not _E8M0_DTYPE:
        raise TypeError(
            "fc2_weight_sf must have dtype torch.float8_e8m0fnu, got "
            f"{fc2_weight_sf.dtype}."
        )
    if fc1_weight_sf.stride(-1) != 1 or fc2_weight_sf.stride(-1) != 1:
        raise ValueError(
            "public weight SF planes must be contiguous along their flat "
            "last dimension."
        )

    expected_device = input_activation.device
    for name, tensor in tensors.items():
        if tensor.device != expected_device:
            raise ValueError(
                f"{name} must be on {expected_device}, got {tensor.device}."
            )

    num_total_experts = num_ranks * num_experts_per_rank
    if torch.any(input_topk_idx < 0).item() or torch.any(
        input_topk_idx >= num_total_experts
    ).item():
        raise ValueError(
            "input_topk_idx values must be in [0, "
            f"{num_total_experts}), got min={input_topk_idx.min().item()} "
            f"max={input_topk_idx.max().item()}."
        )

    return num_ranks, num_experts_per_rank, hidden, intermediate


def _transform_megamoe_weights_to_bf16(
    *,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    num_ranks: int,
    num_experts_per_rank: int,
    hidden: int,
    intermediate: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Transform every public rank/expert plane and retain K-major views."""

    fc1_nk_planes = []
    fc2_nk_planes = []
    for rank in range(num_ranks):
        for expert in range(num_experts_per_rank):
            fc1_nk_planes.append(
                mxfp8_weight_from_swizzled_to_bf16(
                    fc1_weight[rank, expert],
                    fc1_weight_sf[rank, expert],
                )
            )
            fc2_nk_planes.append(
                mxfp8_weight_from_swizzled_to_bf16(
                    fc2_weight[rank, expert],
                    fc2_weight_sf[rank, expert],
                )
            )

    # The transform helper returns launcher-ready (N, K) planes.  Stack in
    # that contiguous form, then expose the BF16 MegaMoE reference's public
    # K-major (K, N) view without repacking its stride-one K axis.
    fc1_weight_bf16 = torch.stack(fc1_nk_planes).reshape(
        num_ranks,
        num_experts_per_rank,
        intermediate,
        hidden,
    ).permute(0, 1, 3, 2)
    fc2_weight_bf16 = torch.stack(fc2_nk_planes).reshape(
        num_ranks,
        num_experts_per_rank,
        hidden,
        intermediate // 2,
    ).permute(0, 1, 3, 2)
    return fc1_weight_bf16, fc2_weight_bf16


def compute_megamoe_reference_mxfp8_bf16(
    input_activation: torch.Tensor,
    input_topk_idx: torch.Tensor,
    input_topk_weights: torch.Tensor,
    fc1_weight: torch.Tensor,
    fc1_weight_sf: torch.Tensor,
    fc2_weight: torch.Tensor,
    fc2_weight_sf: torch.Tensor,
    ref_compute_graph: Literal["transformers", "deepgemm"],
    fc2_output_dtype: torch.dtype = torch.bfloat16,
    gate_up_clamp: Optional[float] = None,
    apply_topk_in_fc1: bool = False,
    return_fc1_gateup: bool = False,
):
    """Adapt public multi-rank MXFP8 weights to the BF16 MegaMoE reference.

    Exact call signature::

        compute_megamoe_reference_mxfp8_bf16(
            input_activation, input_topk_idx, input_topk_weights,
            fc1_weight, fc1_weight_sf, fc2_weight, fc2_weight_sf,
            ref_compute_graph, fc2_output_dtype=torch.bfloat16,
            gate_up_clamp=None, apply_topk_in_fc1=False,
            return_fc1_gateup=False,
        )

    Public ABI:

    * activation: BF16 ``(R, T, H)``;
    * FC1 weight: E4M3FN/E5M2 ``(R, E, H, I)``, dimension 2 (K) stride one;
    * FC1 SF: E8M0 ``(R, E, S1)`` atom-swizzled flat planes whose logical
      shape is ``(I, H/32)``;
    * FC2 weight: the same FP8 dtype, ``(R, E, I/2, H)``, dimension 2 (K)
      stride one;
    * FC2 SF: E8M0 ``(R, E, S2)`` atom-swizzled flat planes whose logical
      shape is ``(H, (I/2)/32)``.

    ``S1`` and ``S2`` include the repository's 32x4x4 atom padding (an
    effective 128 scale rows by 4 scale columns).  The wrapper reverses that
    swizzle independently for every rank/expert, evaluates
    ``BF16(FP8 * E8M0)``, then delegates routing, SwiGLU, top-k placement,
    BF16 dense MMA, and optional ``generate_c`` data to
    :func:`moe_bf16_glu.mega_reference_bf16.compute_megamoe_reference`.
    The delegate is imported lazily so importing and validating this adapter
    does not initialize the CUDA/CuTeDSL reference stack.
    """

    if ref_compute_graph not in ("transformers", "deepgemm"):
        raise ValueError(
            "ref_compute_graph must be 'transformers' or 'deepgemm', got "
            f"{ref_compute_graph!r}."
        )
    if fc2_output_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            "fc2_output_dtype must be torch.bfloat16 or torch.float16, got "
            f"{fc2_output_dtype}."
        )
    if not isinstance(apply_topk_in_fc1, bool):
        raise TypeError(
            "apply_topk_in_fc1 must be a bool, got "
            f"{type(apply_topk_in_fc1).__name__}."
        )
    if not isinstance(return_fc1_gateup, bool):
        raise TypeError(
            "return_fc1_gateup must be a bool, got "
            f"{type(return_fc1_gateup).__name__}."
        )

    (
        num_ranks,
        num_experts_per_rank,
        hidden,
        intermediate,
    ) = _validate_megamoe_public_abi(
        input_activation=input_activation,
        input_topk_idx=input_topk_idx,
        input_topk_weights=input_topk_weights,
        fc1_weight=fc1_weight,
        fc1_weight_sf=fc1_weight_sf,
        fc2_weight=fc2_weight,
        fc2_weight_sf=fc2_weight_sf,
    )
    fc1_weight_bf16, fc2_weight_bf16 = (
        _transform_megamoe_weights_to_bf16(
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            num_ranks=num_ranks,
            num_experts_per_rank=num_experts_per_rank,
            hidden=hidden,
            intermediate=intermediate,
        )
    )

    compute_bf16_reference = _load_bf16_megamoe_reference()
    return compute_bf16_reference(
        input_activation=input_activation,
        input_topk_idx=input_topk_idx,
        input_topk_weights=input_topk_weights,
        fc1_weight=fc1_weight_bf16,
        fc2_weight=fc2_weight_bf16,
        ref_compute_graph=ref_compute_graph,
        fc2_output_dtype=fc2_output_dtype,
        gate_up_clamp=gate_up_clamp,
        apply_topk_in_fc1=apply_topk_in_fc1,
        return_fc1_gateup=return_fc1_gateup,
        gate_up_interleave=MXFP8_GATE_UP_INTERLEAVE,
    )


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "MXFP8_GATE_UP_INTERLEAVE",
    "compute_megamoe_reference_mxfp8_bf16",
    "decode_e8m0",
    "mxfp8_weight_to_bf16",
    "mxfp8_weight_from_swizzled_to_bf16",
]
