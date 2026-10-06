"""Lossless preparation of safetensors IQ2 expert blocks."""

from __future__ import annotations

import torch

from b12x._lib.quant.block_codec import IQ2_CODECS, block_codec
from b12x._lib.quant.iq2_xs import iq2_xs_execution_lut
from .prepare import PreparedW4A16MoeWeights, _make_workspace


def pack_iq2_xs_matrix(
    blocks: torch.Tensor,
    *,
    codec: str = "iq2_xs",
    swap_halves: bool = False,
    tile_descriptors: bool = False,
    tile_scales: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack raw IQ2 blocks into encoded payload and compact metadata planes.

    Descriptors are uint16[E,K/16,N/16,8,2,2], exposed as int32 words.
    Each row pair joins output channels r and r+8 within an N16 tile.
    With tile_descriptors, words use [E,N/64,K/128,8,4,8,2] int32 tiles
    so each K128/N64 descriptor block is contiguous for bulk DMA.
    Metadata holds contiguous FP16[E,K/256,N/16,8,2] bases followed by
    uint8[E,K/256,N/16,8,8,2] scale pairs. With tile_scales, scale pairs use
    uint8[E,N/64,K/128,4,4,8,2] so each stage reads a contiguous 256 bytes.
    XXS uses the same payload permutation for its grid/sign-scale words and
    omits the separate scale-pair plane: 66 bytes per 256 weights versus 74
    for XS. Temporary copies contain at most 256 output rows of one expert.
    """
    spec = block_codec(codec)
    if blocks.dtype != torch.uint8:
        raise TypeError("IQ2_XS blocks must be uint8")
    if blocks.ndim != 4 or blocks.shape[-1] != spec.block_bytes:
        raise ValueError(f"{codec.upper()} blocks must have shape [E,N,K/{spec.block_weights},{spec.block_bytes}]")
    e, n, kb, _ = blocks.shape
    k16 = kb * spec.block_weights // 16
    words_per_row = 4 // spec.pack_factor
    if min(e, n, kb) <= 0 or n % 16:
        raise ValueError("IQ2_XS requires positive geometry and N divisible by 16")
    if swap_halves and n % 32:
        raise ValueError("IQ2_XS projection halves must be divisible by 16")
    if (tile_descriptors or tile_scales) and (n % 64 or (swap_halves and n % 128)):
        raise ValueError("tiled IQ2_XS requires N and projection halves divisible by 64")
    words = torch.empty(
        (e, n // 64, k16 // 8, 8, 4, 8, 2, words_per_row)
        if tile_descriptors else (e, k16, n // 16, 8, 2, words_per_row),
        dtype=torch.int32,
        device=blocks.device,
    )
    metadata = torch.empty(e * kb * n * spec.metadata_bytes, dtype=torch.uint8, device=blocks.device)
    base_bytes = e * kb * n * 2
    bases = metadata[:base_bytes].view(torch.float16).reshape(e, kb, n // 16, 8, 2)
    if spec.subscale_bytes:
        scales = metadata[base_bytes:].reshape(
            (e, n // 64, kb, 2, 4, 4, 8, 2)
            if tile_scales else (e, kb, n // 16, 8, 8, 2)
        )
    for expert in range(e):
        boundaries = (0, n // 2, n) if swap_halves else (0, n)
        for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            for row in range(begin, end, 256):
                stop = min(row + 256, end)
                source_row = (row + n // 2) % n if swap_halves else row
                chunk = blocks[expert, source_row : source_row + stop - row]
                d = (
                    chunk[..., :2]
                    .contiguous()
                    .view(torch.float16)
                    .reshape(stop - row, kb)
                )
                if not bool(torch.isfinite(d).all()):
                    raise ValueError("IQ2_XS block bases must be finite")
                q = chunk[..., 2:2 + spec.payload_bytes].contiguous().view(torch.int32)
                descriptors = (q.reshape(stop - row, k16, words_per_row)
                               .permute(1, 0, 2).contiguous()
                               .reshape(k16, (stop - row) // 16, 2, 8, words_per_row)
                               .transpose(2, 3))
                if tile_descriptors:
                    words[expert, row // 64 : stop // 64].copy_(
                        descriptors.reshape(k16 // 8, 8, (stop - row) // 64, 4, 8, 2, words_per_row)
                        .permute(2, 0, 1, 3, 4, 5, 6)
                    )
                else:
                    words[expert, :, row // 16 : stop // 16].copy_(descriptors)
                bases[expert, :, row // 16 : stop // 16].copy_(
                    d.T.reshape(kb, (stop - row) // 16, 2, 8).transpose(-2, -1)
                )
                if not spec.subscale_bytes:
                    continue
                if tile_scales:
                    scales[expert, row // 64 : stop // 64].copy_(
                        chunk[..., 66:]
                        .reshape((stop - row) // 64, 4, 2, 8, kb, 2, 4)
                        .permute(0, 4, 5, 6, 1, 3, 2)
                    )
                else:
                    scales[expert, :, row // 16 : stop // 16].copy_(
                        chunk[..., 66:]
                        .reshape((stop - row) // 16, 2, 8, kb, 8)
                        .permute(3, 0, 4, 2, 1)
                    )
    return words.reshape(-1), metadata


def prepare_iq2_xs_moe_weights(
    w13: torch.Tensor,
    w2: torch.Tensor,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    activation: str,
    w13_layout: str,
    codec: str = "iq2_xs",
) -> PreparedW4A16MoeWeights:
    """Prepare compact IQ2_XS storage and device tables before execution."""
    spec = block_codec(codec)
    if activation not in {"silu", "relu2"}:
        raise ValueError("IQ2_XS supports SiLU and ReLU²")
    if w13_layout not in {"w13", "w31"}:
        raise ValueError("IQ2_XS requires w13 or w31 projection order")
    if (
        min(hidden_size, intermediate_size, num_experts) <= 0
        or hidden_size % max(128, spec.block_weights)
        or intermediate_size % max(128, spec.block_weights)
    ):
        raise ValueError("IQ2_XS requires positive geometry and H/I divisible by 256")
    if not w13.is_cuda or w2.device != w13.device:
        raise ValueError("IQ2_XS preparation requires tensors on one CUDA device")
    gated = activation == "silu"
    expected13 = (
        num_experts,
        intermediate_size * (2 if gated else 1),
        hidden_size // spec.block_weights,
        spec.block_bytes,
    )
    expected2 = (num_experts, hidden_size, intermediate_size // spec.block_weights, spec.block_bytes)
    if tuple(w13.shape) != expected13 or tuple(w2.shape) != expected2:
        raise ValueError(f"IQ2_XS blocks must have shapes {expected13} and {expected2}")
    q13, s13 = pack_iq2_xs_matrix(
        w13, codec=codec, swap_halves=gated and w13_layout == "w13",
        tile_descriptors=True, tile_scales=True,
    )
    q2, s2 = pack_iq2_xs_matrix(w2, codec=codec, tile_descriptors=True, tile_scales=True)
    if codec in IQ2_CODECS:
        iq2_xs_execution_lut(w13.device, prepare=True, selectors=True, codec=codec)
    unit = torch.ones(num_experts, dtype=torch.float32, device=w13.device)
    return PreparedW4A16MoeWeights(
        w13=q13,
        w13_scale=s13,
        w13_global_scale=unit,
        w2=q2,
        w2_scale=s2,
        w2_global_scale=unit,
        workspace=_make_workspace(w13.device),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=gated,
        params_dtype=torch.bfloat16,
        fc1_tile_n=64,
        fc2_tile_n=64,
        source_format=codec,
        w13_layout="packed",
        weight_layout=codec,
        scale_format=codec,
    )
