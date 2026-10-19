"""Offline FP4 weight preparation for SM90 MegaMoE.

Input weights are ``(packed, scales)`` in row-major ``[experts, N, K/2]``
and ``[experts, N, K/group_size]``. Packed bytes follow Marlin order: byte b in
each four-byte word stores K[b] high and K[b+4] low.
NVFP4 scales are raw unsigned E4M3 bytes per 16 elements; MXFP4 scales are
unsigned E8M0 bytes per 32 elements. L1 inputs have gate rows followed by up.
"""

from typing import Tuple

import torch

MXFP4_TILE_ROWS = 128
MXFP4_TILE_BYTES = MXFP4_TILE_ROWS * 68


def _interleave_gate_up(t: torch.Tensor) -> torch.Tensor:
    e, n, *rest = t.shape
    gate, up = t[:, : n // 2], t[:, n // 2 :]
    return torch.stack(
        (gate.reshape(e, n // 16, 8, *rest), up.reshape(e, n // 16, 8, *rest)),
        dim=2,
    ).reshape(e, n, *rest)


def _braid_signs(rows: torch.Tensor) -> torch.Tensor:
    # Magnitude and sign positions are independent in the mode2 decoder.
    # Braid signs before arranging the format-specific row or tile layout.
    packed = rows[..., :64].reshape(*rows.shape[:-1], 16, 4)
    codes = torch.cat(((packed >> 4) & 15, packed & 15), dim=-1)
    signs = (codes >> 3)[..., [4, 0, 5, 1, 6, 2, 7, 3]]
    braided = (codes & 7) | (signs << 3)
    rows[..., :64] = (braided[..., 0::2] | (braided[..., 1::2] << 4)).reshape(
        *rows.shape[:-1], 64
    )
    return rows


def _transform_one(weights, group_size, interleave, decode_tiles):
    packed, scales = weights
    if packed.dtype != torch.uint8 or scales.dtype != torch.uint8:
        raise ValueError("packed weights and raw scale bytes must have dtype uint8")
    if packed.ndim != 3 or scales.ndim != 3:
        raise ValueError("packed weights and scales must be 3-D")
    e, n, half_k = packed.shape
    k = half_k * 2
    if n % 128 or k % 128 or tuple(scales.shape) != (e, n, k // group_size):
        raise ValueError(
            "weights must contain complete N128/K128 tiles and matching scales"
        )
    if packed.device != scales.device:
        raise ValueError("weights and scales must be on the same device")
    if interleave:
        packed, scales = _interleave_gate_up(packed), _interleave_gate_up(scales)
    rows = torch.zeros((e, n, k // 128, 80), dtype=torch.uint8, device=packed.device)
    rows[..., :64] = packed.reshape(e, n, k // 128, 64)
    rows[..., 64 : 64 + 128 // group_size] = scales.reshape(
        e, n, k // 128, 128 // group_size
    )
    rows = _braid_signs(rows)
    if not decode_tiles:
        return rows.reshape(e, n, k // 128 * 80).contiguous()
    # Chunk-major N128 tiles omit 12 padding bytes per row and keep each
    # decoder's 16-byte loads aligned. Tile index = n_tile * k_blocks + k.
    nt, kb = n // MXFP4_TILE_ROWS, k // 128
    tiled = rows.reshape(e, nt, MXFP4_TILE_ROWS, kb, 80).permute(0, 1, 3, 2, 4)
    fp4 = tiled[..., :64].reshape(e, nt, kb, MXFP4_TILE_ROWS, 4, 16)
    fp4 = fp4.permute(0, 1, 2, 4, 3, 5).reshape(e, nt * kb, MXFP4_TILE_ROWS * 64)
    sf = tiled[..., 64:68].reshape(e, nt * kb, MXFP4_TILE_ROWS * 4)
    return torch.cat((fp4, sf), dim=-1).contiguous()


def _transform(l1_weights, l2_weights, group_size, decode_tiles):
    p1, _ = l1_weights
    p2, _ = l2_weights
    if p1.ndim != 3 or p2.ndim != 3 or p1.shape[0] != p2.shape[0]:
        raise ValueError("L1 and L2 must contain the same expert count")
    if p1.shape[1] != p2.shape[2] * 4 or p1.shape[2] * 2 != p2.shape[1]:
        raise ValueError(
            "expected L1 [E, 2*intermediate, hidden/2] and L2 [E, hidden, intermediate/2]"
        )
    return (
        _transform_one(l1_weights, group_size, True, decode_tiles),
        _transform_one(l2_weights, group_size, False, decode_tiles),
    )


def transform_nvfp4_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return braided NVFP4 weights for both fused and split SM90 kernels.

    Parameters
    ----------
    l1_weights : tuple[torch.Tensor, torch.Tensor]
        Packed UINT8 weights of shape ``[E, 2 * intermediate, hidden // 2]``
        and raw UINT8 E4M3 scale bytes per 16 values. Gate rows precede up rows.
    l2_weights : tuple[torch.Tensor, torch.Tensor]
        Packed UINT8 weights of shape ``[E, hidden, intermediate // 2]``
        and raw UINT8 E4M3 scale bytes per 16 values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Contiguous L1 and L2 weight tensors in the kernel's packed layout,
        on the input device. Weight preparation runs before graph capture.
    """
    return _transform(l1_weights, l2_weights, 16, False)


def transform_mxfp4_weights_for_mega_moe(
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return chunk-major MXFP4 weights for the SM90 fused kernel.

    Parameters
    ----------
    l1_weights : tuple[torch.Tensor, torch.Tensor]
        Packed UINT8 weights of shape ``[E, 2 * intermediate, hidden // 2]``
        and raw UINT8 E8M0 scale bytes per 32 values. Gate rows precede up rows.
    l2_weights : tuple[torch.Tensor, torch.Tensor]
        Packed UINT8 weights of shape ``[E, hidden, intermediate // 2]``
        and raw UINT8 E8M0 scale bytes per 32 values.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Contiguous L1 and L2 weight tensors in the kernel's packed layout,
        on the input device. Weight preparation runs before graph capture.
    """
    return _transform(l1_weights, l2_weights, 32, True)
