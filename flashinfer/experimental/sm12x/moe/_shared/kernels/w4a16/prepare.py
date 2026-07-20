# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/moe/fused/w4a16/prepare.py @ 3663c726 (2026-07-03) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Local NVFP4/BF16 weight preparation for the CuTeDSL W4A16 path."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from flashinfer.experimental.sm12x.moe._shared.kernels.w4a16.host import (
    W4A16PackedBuffers,
    make_w4a16_packed_buffers as _make_w4a16_packed_buffers,
    unswizzle_expert_scales,
    validate_nf3_moe_inputs,
    validate_w4a16_packed_inputs,
)


_PACKED_TILE_SIZE = 16
_PACKED_TILE_N_SIZE = 64
_PACK_FACTOR_4BIT = 8
_MODEL_OPT_W13_LAYOUTS = {"w13", "w31"}
_SOURCE_FORMATS = {
    "modelopt_nvfp4": "modelopt_nvfp4",
    "fp4_e8m0_k32": "fp4_e8m0_k32",
    "compressed_tensors": "compressed_tensors",
}
_E8M0_K32_BF16_MAX_SCALE_BYTE = 247
_E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT = 64
# Canonical W13 layout names are "w13"/"w31"; accept the physical FC1-half
# spellings as aliases. Logical checkpoint order "w13" arrives up/gate and
# needs a swap before the kernel's SwiGLU; "w31" is already kernel-native
# gate/up order.
_W13_LAYOUTS = {
    "w13": "w13",
    "w31": "w31",
    "up_gate": "w13",
    "gate_up": "w31",
}
_MODEL_OPT_NVFP4_FORMATS = {"modelopt_nvfp4"}
# NF3 ("nf3_2p1") 3-bit codebook. Must match kernel._NF3_CODEBOOK. The scale
# convention (see NF3_KERNEL_MODSPEC.md): weights decode to the FULL-precision
# bf16 codebook value, the K/32 scale byte encodes t_s * 2**-4 (e4m3-style,
# E4=0 arm), and the per-tensor global_scale carries the matching 2**116.
_NF3_CODEBOOK = (-1.0, -0.6047, -0.3563, -0.1275, 0.1275, 0.3563, 0.6047, 1.0)
_NF3_SCALE_GLOBAL = 2.0**116
_NF3_SCALE_FLOOR = (
    2.0**-10
)  # f16(t_s*2^-4) must stay f16-NORMAL (>=2^-14) -> t_s >= 2^-10; real early-layer scales dip below the old 2^-7


@dataclass(frozen=True)
class W4A16PackedWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt_nvfp4"
    w13_layout: str = "w13"
    weight_layout: str = "packed"
    scale_format: str = "e4m3_k16"


@dataclass(frozen=True)
class W4A16ModelOptWeights:
    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    source_format: str = "modelopt_nvfp4"
    weight_layout: str = "modelopt"
    scale_format: str = "e4m3_k16"
    micro_w13_scale: torch.Tensor | None = None
    micro_w13_global_scale: torch.Tensor | None = None
    micro_w2_scale: torch.Tensor | None = None
    micro_w2_global_scale: torch.Tensor | None = None
    # Physical order of the two fused FC1 halves in source W13. "w13" (logical,
    # == "up_gate" physical) needs a row rotation before W4A16 SwiGLU; "w31"
    # (== "gate_up") is already in the kernel-native order.
    w13_layout: str = "w13"


@dataclass(frozen=True)
class PreparedNF3MoeWeights:
    """Runtime NF3 ("nf3_2p1") hybrid-expert W4A16 weights.

    Mirrors W4A16PackedWeights so run_w4a16_moe consumes it identically. The
    packed planes are int32 (3 words per 32-code unit); the K/32 scales are
    e4m3-style bytes viewed uint8; the global scales carry the 2**116 the NF3
    scale convention defers out of the codebook. fc1_tile_n / fc2_tile_n record
    the CTA N-tile the flat-span packing was built for -- the kernel MUST be
    compiled/launched with the SAME tile_n (read them back from the compiled
    W4A16FusedMoeCompileResult).
    """

    w13: torch.Tensor
    w13_scale: torch.Tensor
    w13_global_scale: torch.Tensor
    w2: torch.Tensor
    w2_scale: torch.Tensor
    w2_global_scale: torch.Tensor
    workspace: torch.Tensor
    hidden_size: int
    intermediate_size: int
    num_experts: int
    is_gated: bool
    params_dtype: torch.dtype
    fc1_tile_n: int
    fc2_tile_n: int
    source_format: str = "nf3_2p1"
    w13_layout: str = "packed"
    weight_layout: str = "nf3_2p1"
    scale_format: str = "e4m3_k32"


def _make_workspace(
    device: torch.device, *, max_blocks_per_sm: int = 4
) -> torch.Tensor:
    props = torch.cuda.get_device_properties(device)
    sms = int(props.multi_processor_count)
    return torch.zeros(
        (sms * int(max_blocks_per_sm) + 2,),
        dtype=torch.int32,
        device=device,
    )


def _scale_perms() -> tuple[list[int], list[int]]:
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single


def _e8m0_logical_tail_scale_n(size_n: int) -> int:
    return (
        (int(size_n) + _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT - 1)
        // _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT
    ) * _E8M0_LOGICAL_TAIL_SCALE_N_ALIGNMENT


def _permute_packed_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    group_size: int,
    output_size_n: int | None = None,
) -> torch.Tensor:
    scale_perm, scale_perm_single = _scale_perms()
    if group_size < size_k and group_size != -1:
        block = len(scale_perm)
        if output_size_n is not None or int(size_n) % block != 0:
            padded_n = (
                ((int(size_n) + block - 1) // block) * block
                if output_size_n is None
                else int(output_size_n)
            )
            if padded_n < int(size_n) or padded_n % block != 0:
                raise ValueError(
                    f"output_size_n must be a multiple of {block} and >= size_n"
                )
            padded = scales.new_zeros((int(scales.shape[0]), padded_n))
            padded[:, : int(size_n)] = scales
            scales = padded.reshape((int(scales.shape[0]), -1, block))
            scales = scales[:, :, scale_perm].reshape((int(scales.shape[0]), padded_n))
            if output_size_n is None:
                scales = scales[:, : int(size_n)]
            return scales.contiguous()
        scales = scales.reshape((-1, block))[:, scale_perm]
    else:
        block = len(scale_perm_single)
        if output_size_n is not None or int(size_n) % block != 0:
            padded_n = (
                ((int(size_n) + block - 1) // block) * block
                if output_size_n is None
                else int(output_size_n)
            )
            if padded_n < int(size_n) or padded_n % block != 0:
                raise ValueError(
                    f"output_size_n must be a multiple of {block} and >= size_n"
                )
            padded = scales.new_zeros((int(scales.shape[0]), padded_n))
            padded[:, : int(size_n)] = scales
            scales = padded.reshape((int(scales.shape[0]), -1, block))
            scales = scales[:, :, scale_perm_single].reshape(
                (int(scales.shape[0]), padded_n)
            )
            if output_size_n is None:
                scales = scales[:, : int(size_n)]
            return scales.contiguous()
        scales = scales.reshape((-1, block))[:, scale_perm_single]
    return scales.reshape((-1, size_n)).contiguous()


def _nvfp4_compute_scale_factor(
    packed_scales: torch.Tensor,
    a_dtype: torch.dtype,
) -> float:
    if a_dtype == torch.float16:
        return 1.0
    max_scalar = 0.0
    for expert in range(int(packed_scales.shape[0])):
        ws_float = packed_scales[expert].float() * (2**7)
        nonzero_mask = ws_float > 0
        if bool(nonzero_mask.any().item()):
            max_scalar = max(max_scalar, float(ws_float[nonzero_mask].max().item()))
    if max_scalar > 0.0 and max_scalar < 448 * (2**7):
        return float(2 ** math.floor(math.log2((448 * (2**7)) / max_scalar)))
    return 1.0


def _process_nvfp4_packed_scales(
    packed_scales: torch.Tensor,
    *,
    scale_factor: float,
) -> torch.Tensor:
    packed_scales = packed_scales.to(torch.float16)
    packed_scales = packed_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        packed_scales.size(0),
        -1,
    )
    if scale_factor > 1.0:
        packed_scales = (packed_scales.float() * scale_factor).to(torch.float16)
    packed_scales = packed_scales * (2**7)
    packed_scales[packed_scales < 2] = 0
    packed_scales = packed_scales.view(torch.int16) << 1
    packed_scales = packed_scales.view(torch.float8_e4m3fn)
    return packed_scales[:, 1::2].contiguous()


def _process_nvfp4_packed_global_scale(
    global_scale: torch.Tensor,
    *,
    a_dtype: torch.dtype,
) -> torch.Tensor:
    if a_dtype == torch.float16:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise TypeError(f"unsupported W4A16 activation dtype {a_dtype}")
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return global_scale * (2.0 ** (exponent_bias - 7))


def _process_nvfp4_micro_global_scale_from_packed(
    packed_global_scale: torch.Tensor,
    *,
    a_dtype: torch.dtype,
) -> torch.Tensor:
    if a_dtype == torch.float16:
        target_exponent = 5
    elif a_dtype == torch.bfloat16:
        target_exponent = 8
    else:
        raise TypeError(f"unsupported W4A16 activation dtype {a_dtype}")
    fp4_exponent = 2
    exponent_bias = 2 ** (target_exponent - 1) - 2 ** (fp4_exponent - 1)
    return (
        (packed_global_scale * (2.0 ** (-exponent_bias))).to(torch.float32).contiguous()
    )


def _normalize_source_format(source_format: str) -> str:
    if source_format.lower() == "mxfp4_native":
        raise ValueError(
            "source_format='mxfp4_native' has been removed; use "
            "source_format='fp4_e8m0_k32' for E8M0 K/32 scales, or add "
            "a real MXFP4 source contract"
        )
    try:
        return _SOURCE_FORMATS[source_format.lower()]
    except KeyError as exc:
        raise ValueError(
            "source_format must be one of 'modelopt_nvfp4', "
            "'fp4_e8m0_k32', or 'compressed_tensors', "
            f"got {source_format!r}"
        ) from exc


def _normalize_w13_layout(w13_layout: str) -> str:
    try:
        return _W13_LAYOUTS[w13_layout.lower()]
    except KeyError as exc:
        raise ValueError(
            "w13_layout must be one of 'w13'/'w31' (or the 'up_gate'/'gate_up' "
            f"aliases), got {w13_layout!r}"
        ) from exc


def _source_global_scale(
    global_scale: torch.Tensor, *, source_format: str
) -> torch.Tensor:
    if source_format == "compressed_tensors":
        return (1.0 / global_scale).to(torch.float32).contiguous()
    return global_scale.contiguous()


def _validate_e8m0_k32_scales(
    scales: torch.Tensor,
    *,
    rows: int,
    cols: int,
    name: str,
    allow_k_tail: bool = False,
) -> torch.Tensor:
    """Validate source E8M0 K/32 scale tensor shape and dtype."""
    if scales.ndim != 3:
        raise ValueError(
            f"{name} must be [E, N, ceil(K/32)], got {tuple(scales.shape)}"
        )
    if allow_k_tail:
        if int(cols) % 8 != 0:
            raise ValueError(
                f"{name} compact E8M0 K-tail requires K divisible by 8, got {int(cols)}"
            )
        expected_cols = (int(cols) + 31) // 32
    elif int(cols) % 32 != 0:
        raise ValueError(f"{name} requires K divisible by 32, got {int(cols)}")
    else:
        expected_cols = int(cols) // 32
    if tuple(scales.shape[1:]) != (int(rows), expected_cols):
        raise ValueError(
            f"{name} must have shape [E, {int(rows)}, {expected_cols}] for "
            f"E8M0 K/32 scales, got {tuple(scales.shape)}"
        )
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    if scales.dtype == torch.uint8:
        return scales.view(e8m0_dtype) if e8m0_dtype is not None else scales
    if e8m0_dtype is not None and scales.dtype == e8m0_dtype:
        return scales
    raise TypeError(f"{name} must be torch.uint8 or torch.float8_e8m0fnu")


def _pack_e8m0_k32_scales(
    scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
    allow_k_tail: bool = False,
    padded_size_n: int | None = None,
) -> torch.Tensor:
    if allow_k_tail:
        if int(size_k) % 8 != 0:
            raise ValueError(
                f"compact E8M0 K-tail requires K divisible by 8, got {size_k}"
            )
        scale_cols = (int(size_k) + 31) // 32
    elif int(size_k) % 32 != 0:
        raise ValueError(f"E8M0 K/32 scales require K divisible by 32, got {size_k}")
    else:
        scale_cols = int(size_k) // 32
    if tuple(scales.shape[1:]) != (int(size_n), scale_cols):
        raise ValueError(
            f"expected E8M0 scale shape [E, {int(size_n)}, {scale_cols}], "
            f"got {tuple(scales.shape)}"
        )
    output_size_n = int(size_n) if padded_size_n is None else int(padded_size_n)
    if output_size_n < int(size_n):
        raise ValueError(
            f"padded_size_n must be >= size_n, got {output_size_n} < {int(size_n)}"
        )
    source = scales.view(torch.uint8)
    if reuse_input_storage:
        if output_size_n != int(size_n):
            raise ValueError("reuse_input_storage requires unpadded E8M0 scales")
        if allow_k_tail:
            raise ValueError(
                "reuse_input_storage is not supported for compact E8M0 K-tail scales"
            )
        if not source.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous E8M0 scales")
        source.clamp_(max=_E8M0_K32_BF16_MAX_SCALE_BYTE)
        packed = source.reshape(
            int(source.shape[0]),
            scale_cols,
            output_size_n,
        )
    else:
        source = source.clamp(max=_E8M0_K32_BF16_MAX_SCALE_BYTE)
        packed = torch.empty(
            (int(source.shape[0]), scale_cols, output_size_n),
            dtype=torch.uint8,
            device=scales.device,
        )
    for expert in range(int(source.shape[0])):
        expert_source = source[expert]
        if row_rotation is not None:
            expert_source = torch.cat(
                [expert_source[row_rotation:], expert_source[:row_rotation]],
                dim=0,
            )
        expert_packed = _permute_packed_scales(
            expert_source.T.contiguous(),
            size_k=size_k,
            size_n=size_n,
            group_size=32,
            output_size_n=output_size_n,
        )
        expert_packed = (
            expert_packed.view(-1, 4)[:, [0, 2, 1, 3]]
            .reshape_as(expert_packed)
            .contiguous()
        )
        packed[expert].copy_(expert_packed)
    return packed.view(scales.dtype) if scales.dtype != torch.uint8 else packed


def _repack_4bit_no_perm(
    qweight_i32: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    out: torch.Tensor | None = None,
    flat_scratch: torch.Tensor | None = None,
    gather_scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pack 4-bit weights into the W4A16 A16 kernel layout."""
    if qweight_i32.dtype != torch.int32:
        raise TypeError("qweight_i32 must be torch.int32")
    if tuple(qweight_i32.shape) != (size_k // _PACK_FACTOR_4BIT, size_n):
        raise ValueError(
            f"expected qweight shape {(size_k // _PACK_FACTOR_4BIT, size_n)}, "
            f"got {tuple(qweight_i32.shape)}"
        )
    if size_k % _PACKED_TILE_SIZE != 0 or size_n % _PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )

    k_tiles = size_k // _PACKED_TILE_SIZE
    n_tiles = size_n // _PACKED_TILE_N_SIZE
    packed_shape = (k_tiles, n_tiles, 128)
    if out is not None and (
        out.dtype != torch.int32 or tuple(out.shape) != packed_shape
    ):
        raise ValueError(
            f"out must be int32 with shape {packed_shape}, got "
            f"{out.dtype} {tuple(out.shape)}"
        )
    if flat_scratch is not None and (
        flat_scratch.dtype != torch.int32 or tuple(flat_scratch.shape) != packed_shape
    ):
        raise ValueError(
            f"flat_scratch must be int32 with shape {packed_shape}, got "
            f"{flat_scratch.dtype} {tuple(flat_scratch.shape)}"
        )
    if gather_scratch is not None and (
        gather_scratch.dtype != torch.int32
        or tuple(gather_scratch.shape) != packed_shape
    ):
        raise ValueError(
            f"gather_scratch must be int32 with shape {packed_shape}, got "
            f"{gather_scratch.dtype} {tuple(gather_scratch.shape)}"
        )

    tiles = qweight_i32.view(
        k_tiles,
        2,
        n_tiles,
        _PACKED_TILE_N_SIZE,
    )
    if flat_scratch is None:
        flat = tiles.permute(0, 2, 1, 3).reshape(
            k_tiles,
            n_tiles,
            2 * _PACKED_TILE_N_SIZE,
        )
    else:
        flat_scratch.view(k_tiles, n_tiles, 2, _PACKED_TILE_N_SIZE).copy_(
            tiles.permute(0, 2, 1, 3)
        )
        flat = flat_scratch

    device = qweight_i32.device
    out_pos = torch.arange(128, device=device, dtype=torch.long)
    th_id = out_pos // 4
    warp_id = out_pos % 4
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    offsets = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    pack_idx = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7], device=device, dtype=torch.long)

    elem = tc_row[:, None] + offsets[None, :]
    row = elem // _PACK_FACTOR_4BIT
    pos = elem % _PACK_FACTOR_4BIT
    col1 = (warp_id * 16 + tc_col)[:, None].expand(-1, 4)
    col2 = col1 + 8
    source_index = torch.cat(
        [row * _PACKED_TILE_N_SIZE + col1, row * _PACKED_TILE_N_SIZE + col2],
        dim=1,
    )[:, pack_idx]
    source_shift = torch.cat([pos, pos], dim=1)[:, pack_idx] * 4

    result = (
        torch.empty(packed_shape, device=device, dtype=torch.int32)
        if out is None
        else out
    )
    result.zero_()
    for slot in range(8):
        gather_index = (
            source_index[:, slot]
            .view(1, 1, 128)
            .expand(
                k_tiles,
                n_tiles,
                128,
            )
        )
        shift = source_shift[:, slot].view(1, 1, 128)
        if gather_scratch is None:
            gathered = flat.gather(2, gather_index)
            nibble = (gathered >> shift) & 0xF
            result |= nibble << (slot * 4)
        else:
            torch.gather(flat, 2, gather_index, out=gather_scratch)
            torch.bitwise_right_shift(gather_scratch, shift, out=gather_scratch)
            torch.bitwise_and(gather_scratch, 0xF, out=gather_scratch)
            if slot:
                torch.bitwise_left_shift(
                    gather_scratch,
                    slot * 4,
                    out=gather_scratch,
                )
            torch.bitwise_or(result, gather_scratch, out=result)

    return result.reshape(k_tiles, n_tiles * 128).contiguous()


def _repack_weight(
    weight: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    row_rotation: int | None = None,
    reuse_input_storage: bool = False,
) -> torch.Tensor:
    num_experts = int(weight.shape[0])
    if tuple(weight.shape[1:]) != (size_n, size_k // 2):
        raise ValueError(
            f"expected packed weight shape {(num_experts, size_n, size_k // 2)}, "
            f"got {tuple(weight.shape)}"
        )
    if size_k % _PACKED_TILE_SIZE != 0 or size_n % _PACKED_TILE_N_SIZE != 0:
        raise ValueError(
            f"W4A16 repack requires K,N multiples of 16,64; got {size_k},{size_n}"
        )

    packed_shape = (
        num_experts,
        size_k // _PACKED_TILE_SIZE,
        (size_n // _PACKED_TILE_N_SIZE) * 128,
    )
    if reuse_input_storage:
        if not weight.is_contiguous():
            raise ValueError("reuse_input_storage requires contiguous packed weights")
        packed = weight.view(torch.int32).reshape(packed_shape)
    else:
        packed = torch.empty(packed_shape, device=weight.device, dtype=torch.int32)

    k_tiles = size_k // _PACKED_TILE_SIZE
    n_tiles = size_n // _PACKED_TILE_N_SIZE
    qweight_scratch = torch.empty(
        (size_k // _PACK_FACTOR_4BIT, size_n),
        device=weight.device,
        dtype=torch.int32,
    )
    flat_scratch = torch.empty(
        (k_tiles, n_tiles, 128),
        device=weight.device,
        dtype=torch.int32,
    )
    gather_scratch = torch.empty_like(flat_scratch)

    for expert in range(num_experts):
        expert_weight = weight[expert].view(torch.int32)
        if row_rotation is not None:
            rotated_rows = int(size_n) - int(row_rotation)
            qweight_scratch[:, :rotated_rows].copy_(expert_weight[row_rotation:].T)
            qweight_scratch[:, rotated_rows:].copy_(expert_weight[:row_rotation].T)
        else:
            qweight_scratch.copy_(expert_weight.T)
        _repack_4bit_no_perm(
            qweight_scratch,
            size_k=size_k,
            size_n=size_n,
            out=packed[expert].view(k_tiles, n_tiles, 128),
            flat_scratch=flat_scratch,
            gather_scratch=gather_scratch,
        )
    return packed


def _permute_nvfp4_scales(
    scales: torch.Tensor,
    global_scales: torch.Tensor,
    *,
    size_k: int,
    size_n: int,
    a_dtype: torch.dtype,
    row_rotation: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_scale_factor = _nvfp4_compute_scale_factor(scales, a_dtype)
    packed_scales: torch.Tensor | None = None
    for expert in range(scales.shape[0]):
        expert_source = scales[expert].to(a_dtype)
        if row_rotation is not None:
            expert_source = torch.cat(
                [expert_source[row_rotation:], expert_source[:row_rotation]],
                dim=0,
            )
        expert_scales = _permute_packed_scales(
            expert_source.T,
            size_k=size_k,
            size_n=size_n,
            group_size=16,
        )
        expert_packed = _process_nvfp4_packed_scales(
            expert_scales,
            scale_factor=combined_scale_factor,
        )
        if packed_scales is None:
            packed_scales = torch.empty(
                (int(scales.shape[0]), *expert_packed.shape),
                dtype=expert_packed.dtype,
                device=expert_packed.device,
            )
        packed_scales[expert].copy_(expert_packed)
    if packed_scales is None:
        packed_scales = torch.empty(
            (0, size_k // _PACKED_TILE_SIZE, size_n // 2),
            dtype=torch.float8_e4m3fn,
            device=scales.device,
        )
    packed_global = _process_nvfp4_packed_global_scale(
        global_scales,
        a_dtype=a_dtype,
    ).to(torch.float32)
    packed_global = packed_global / combined_scale_factor
    return packed_scales, packed_global.contiguous()


# ---------------------------------------------------------------------------
# NF3 ("nf3_2p1") packing.
#
# The kernel's B operand is a stream of 32-code "units" (2 N-columns x 16 K),
# each a 12-byte (lo0, lo1, hi) triple. The GEMM reads unit index
#   unit = cur_group_id * (tile_n // 2) + (tid % (tile_n // 2))
# from the pipe's B SMEM region (see kernel._load_b_scale_registers,
# NF3-MAPPING-V1) which -- verified algebraically -- equals the stock staged
# unit index; cur_group_id is the K16 row within the CTA-K tile, (tid %
# (tile_n // 2)) is the N-pair. The flat-span staging copies the packer's
# global bytes verbatim, so the packer must place, at global unit
#   I = nt*(K/16)*(tile_n/2) + R*(tile_n/2) + p          (n_tile-major)
# the codes for CTA N-tile nt, K16-row-global R, N-pair p. The 4 jj words in
# the triple correspond to the stock 4-u32 group; per jj the 8 codes decode
# (packed_dequant_nf3x8_to_bfloat2x4) to fragments
#   frag[0,0]=(cb[c0],cb[c1]) frag[0,1]=(cb[c2],cb[c3])
#   frag[1,0]=(cb[c4],cb[c5]) frag[1,1]=(cb[c6],cb[c7])
# which must equal the stock fragment identity for this thread/jj:
#   frag[0,0]=(C1@K0,   C1@K0+1)  frag[0,1]=(C1@K0+8, C1@K0+9)
#   frag[1,0]=(C2@K0,   C2@K0+1)  frag[1,1]=(C2@K0+8, C2@K0+9)
# where, from _repack_4bit_no_perm composed with the read mapping,
#   wru = p + (tile_n/2)*nt ; n_tile_64 = wru // 32 ; th_id = wru % 32
#   tc_col = th_id // 4 ; tc_row = (th_id % 4) * 2
#   C1 = n_tile_64*64 + jj*16 + tc_col ; C2 = C1 + 8 ; K0 = R*16 + tc_row
# => code order [c0..c7] =
#   [ (C1,K0),(C1,K0+1),(C1,K0+8),(C1,K0+9),
#     (C2,K0),(C2,K0+1),(C2,K0+8),(C2,K0+9) ].
# This whole chain is asserted by _nf3_pack_selftest against an independent
# torch simulation of the kernel read+dequant.
# ---------------------------------------------------------------------------


def _nf3_check_shapes(size_k: int, size_n: int, tile_n: int) -> None:
    if int(size_k) % 32 != 0:
        raise ValueError(f"NF3 requires K % 32 == 0, got {size_k}")
    if int(size_n) % 64 != 0:
        raise ValueError(f"NF3 requires N % 64 == 0, got {size_n}")
    if int(tile_n) % 16 != 0 or int(tile_n) < 64:
        raise ValueError(f"NF3 tile_n must be a multiple of 16 and >= 64, got {tile_n}")
    if int(size_n) % int(tile_n) != 0:
        raise ValueError(f"NF3 requires N ({size_n}) divisible by tile_n ({tile_n})")


def _nf3_code_gather_index(
    size_k: int, size_n: int, tile_n: int, device: torch.device
) -> torch.Tensor:
    """[units, 4, 8] flat indices into a contiguous [N, K] code tensor.

    Entry [I, jj, v] is the (C * size_k + K) index of the v-th code (NF3 order)
    of triple word jj at global unit I. Built from the exact _repack /
    read-mapping math documented above so the packer and the kernel agree.
    """
    _nf3_check_shapes(size_k, size_n, tile_n)
    npairs = int(tile_n) // 2
    k16 = int(size_k) // 16
    units = k16 * (int(size_n) // 2)
    ntile_units = k16 * npairs
    idx_i = torch.arange(units, device=device, dtype=torch.long)
    nt = idx_i // ntile_units
    within = idx_i % ntile_units
    R = within // npairs
    p = within % npairs
    wru = p + npairs * nt
    n_tile_64 = wru // 32
    th_id = wru % 32
    tc_col = th_id // 4
    tc_row = (th_id % 4) * 2
    jj = torch.arange(4, device=device, dtype=torch.long)
    col1 = jj[None, :] * 16 + tc_col[:, None]  # [units, 4]
    col2 = col1 + 8
    C1 = n_tile_64[:, None] * 64 + col1  # [units, 4]
    C2 = n_tile_64[:, None] * 64 + col2
    K0 = (R * 16 + tc_row)[:, None]  # [units, 1]
    k_off = torch.tensor([0, 1, 8, 9], device=device, dtype=torch.long)
    idx = torch.empty((units, 4, 8), device=device, dtype=torch.long)
    for t in range(4):
        idx[:, :, t] = C1 * int(size_k) + (K0 + k_off[t])
        idx[:, :, 4 + t] = C2 * int(size_k) + (K0 + k_off[t])
    return idx


def _nf3_pack_codes(
    codes_nk: torch.Tensor, *, size_k: int, size_n: int, tile_n: int
) -> torch.Tensor:
    """Pack one expert's [N, K] codes (0..7) into int32 [units*3] NF3 planes."""
    if tuple(codes_nk.shape) != (int(size_n), int(size_k)):
        raise ValueError(
            f"expected codes shape {(int(size_n), int(size_k))}, "
            f"got {tuple(codes_nk.shape)}"
        )
    device = codes_nk.device
    flat = codes_nk.reshape(-1).to(torch.int64)
    if bool((flat < 0).any()) or bool((flat > 7).any()):
        raise ValueError("NF3 codes must be integers in [0, 7]")
    idx = _nf3_code_gather_index(size_k, size_n, tile_n, device)  # [units,4,8]
    g = flat[idx.reshape(-1)].reshape(idx.shape)  # [units,4,8]
    sh_lo = (torch.arange(4, device=device, dtype=torch.int64) * 2).view(1, 1, 4)
    sh_hi = torch.arange(4, device=device, dtype=torch.int64).view(1, 1, 4)
    la = ((g[:, :, 0:4] & 3) << sh_lo).sum(-1)  # [units,4]
    lb = ((g[:, :, 4:8] & 3) << sh_lo).sum(-1)
    lo16 = la | (lb << 8)
    ha = (((g[:, :, 0:4] >> 2) & 1) << sh_hi).sum(-1)
    hb = (((g[:, :, 4:8] >> 2) & 1) << sh_hi).sum(-1)
    hi8 = ha | (hb << 4)
    lo0 = lo16[:, 0] | (lo16[:, 1] << 16)
    lo1 = lo16[:, 2] | (lo16[:, 3] << 16)
    hi = hi8[:, 0] | (hi8[:, 1] << 8) | (hi8[:, 2] << 16) | (hi8[:, 3] << 24)
    words = torch.stack([lo0, lo1, hi], dim=1).reshape(-1)  # int64 in [0, 2**32)
    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).contiguous()


def _process_nf3_packed_scales(packed_scales: torch.Tensor) -> torch.Tensor:
    """[rows, N] float t_s -> [rows, N] uint8 e4m3-style byte = highbyte(f16(t_s*2**-4)<<1)."""
    packed_scales = packed_scales.to(torch.float16)
    packed_scales = packed_scales.view(-1, 4)[:, [0, 2, 1, 3]].view(
        packed_scales.size(0),
        -1,
    )
    packed_scales = (packed_scales.float() * (2.0**-4)).to(torch.float16)
    packed_scales = packed_scales.view(torch.int16) << 1
    packed_scales = packed_scales.view(torch.uint8)
    return packed_scales[:, 1::2].contiguous()


def _nf3_pack_scales(t_s: torch.Tensor, *, size_k: int, size_n: int) -> torch.Tensor:
    """Pack one expert's [N, K//32] scales into uint8 [K//32, N] (permuted)."""
    if tuple(t_s.shape) != (int(size_n), int(size_k) // 32):
        raise ValueError(
            f"expected scale shape {(int(size_n), int(size_k) // 32)}, "
            f"got {tuple(t_s.shape)}"
        )
    t_s = t_s.to(torch.float32)
    # zero scales (whole group exactly zero in the checkpoint) MUST stay zero:
    # stored byte 0 decodes to exact +0.0. Flooring them to _NF3_SCALE_FLOOR
    # injects noise into all-zero groups (real early-layer experts have many)
    # -> compounding early-layer error -> prompt-dependent degeneration.
    zero_mask = t_s == 0
    t_s = t_s.clamp(min=_NF3_SCALE_FLOOR)
    t_s[zero_mask] = 0.0
    if bool((t_s >= 32.0).any()):
        raise ValueError("NF3 scales must be < 32 for the e4m3 K/32 encoding")
    permuted = _permute_packed_scales(
        t_s.T.contiguous(),
        size_k=size_k,
        size_n=size_n,
        group_size=32,
    )  # [K//32, N] float
    return _process_nf3_packed_scales(permuted)


def _nf3_pack_code_experts(
    codes: torch.Tensor, *, size_k: int, size_n: int, tile_n: int
) -> torch.Tensor:
    packed = [
        _nf3_pack_codes(codes[e], size_k=size_k, size_n=size_n, tile_n=tile_n)
        for e in range(int(codes.shape[0]))
    ]
    return torch.stack(packed, dim=0).contiguous()


def _nf3_pack_scale_experts(
    t_s: torch.Tensor, *, size_k: int, size_n: int
) -> torch.Tensor:
    packed = [
        _nf3_pack_scales(t_s[e], size_k=size_k, size_n=size_n)
        for e in range(int(t_s.shape[0]))
    ]
    return torch.stack(packed, dim=0).contiguous()


def _prepare_w4a16_packed_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    source_format: str,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    source_format = _normalize_source_format(source_format)
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13 = w13_fp4
    w13_scale = unswizzle_expert_scales(
        w13_blockscale,
        rows=w13_rows,
        cols=hidden_size,
    )
    w13_row_rotation = None
    if is_gated and w13_layout == "w13":
        # In-place: the half-swap is folded into the repack via row_rotation;
        # never materialize a second copy of the weights/scales.
        w13_row_rotation = intermediate_size

    w2_scale = unswizzle_expert_scales(
        w2_blockscale,
        rows=hidden_size,
        cols=intermediate_size,
    )

    packed_w13 = _repack_weight(
        w13 if reuse_input_storage else w13.contiguous(),
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2 = _repack_weight(
        w2_fp4 if reuse_input_storage else w2_fp4.contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )
    native_w13_global_scale = _source_global_scale(
        w13_global_scale,
        source_format=source_format,
    )
    native_w2_global_scale = _source_global_scale(
        w2_global_scale,
        source_format=source_format,
    )
    packed_w13_scale, packed_w13_global_scale = _permute_nvfp4_scales(
        w13_scale,
        native_w13_global_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        a_dtype=params_dtype,
        row_rotation=w13_row_rotation,
    )
    packed_w2_scale, packed_w2_global_scale = _permute_nvfp4_scales(
        w2_scale,
        native_w2_global_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        a_dtype=params_dtype,
    )

    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=packed_w13_global_scale,
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=packed_w2_global_scale,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format=source_format,
        w13_layout=w13_layout,
        scale_format="e4m3_k16",
    )


def prepare_w4a16_modelopt_nvfp4_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare ModelOpt NVFP4 tensors into the W4A16 packed runtime layout.

    The per-block scales are the normal NVFP4 K/16 scale grid in sm12x swizzled
    storage. The global scales are raw ModelOpt weight global scales; activation
    input scales are not folded into W4A16 weight preparation. For gated
    activations, ``w13_layout`` describes whether fused W13 rows arrive in
    checkpoint/logical W13 order or already swapped W31 order.
    """
    return _prepare_w4a16_packed_weights(
        w13_fp4,
        w13_blockscale,
        w13_global_scale,
        w2_fp4,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="modelopt_nvfp4",
        w13_layout=w13_layout,
        reuse_input_storage=reuse_input_storage,
    )


def prepare_w4a16_modelopt_native_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
) -> W4A16ModelOptWeights:
    """Prepare W4A16 metadata while keeping ModelOpt FP4 weights native.

    This is the memory-safe path for GLM serving that needs A4 prefill and A16
    decode in the same process. It keeps the checkpoint FP4 tensors resident
    instead of materializing a second full W4A16 packed copy.
    """
    source_format = _normalize_source_format(source_format)
    if source_format not in _MODEL_OPT_NVFP4_FORMATS:
        raise ValueError(
            "native W4A16 ModelOpt weights require source_format 'modelopt_nvfp4'"
        )
    w13_layout = _normalize_w13_layout(w13_layout)

    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13_scale = unswizzle_expert_scales(
        w13_blockscale,
        rows=w13_rows,
        cols=hidden_size,
    )
    w2_scale = unswizzle_expert_scales(
        w2_blockscale,
        rows=hidden_size,
        cols=intermediate_size,
    )
    native_w13_global_scale = _source_global_scale(
        w13_global_scale,
        source_format=source_format,
    )
    native_w2_global_scale = _source_global_scale(
        w2_global_scale,
        source_format=source_format,
    )

    # The W4A16 activation consumes FC1 output in gate/up logical order.
    # Checkpoint-native ModelOpt GLM tensors are up/gate, while vLLM/FI can
    # hand over gate/up tensors after its own W13 reorder. Keep that physical
    # order explicit so source_format never implies a layout transformation.
    w13_row_rotation = intermediate_size if is_gated and w13_layout == "w13" else None
    packed_w13_scale, packed_w13_global_scale = _permute_nvfp4_scales(
        w13_scale,
        native_w13_global_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        a_dtype=params_dtype,
        row_rotation=w13_row_rotation,
    )
    packed_w2_scale, packed_w2_global_scale = _permute_nvfp4_scales(
        w2_scale,
        native_w2_global_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        a_dtype=params_dtype,
    )

    return W4A16ModelOptWeights(
        w13=w13_fp4,
        w13_scale=packed_w13_scale,
        w13_global_scale=packed_w13_global_scale,
        w2=w2_fp4,
        w2_scale=packed_w2_scale,
        w2_global_scale=packed_w2_global_scale,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format=source_format,
        micro_w13_scale=packed_w13_scale,
        micro_w13_global_scale=_process_nvfp4_micro_global_scale_from_packed(
            packed_w13_global_scale,
            a_dtype=params_dtype,
        ),
        micro_w2_scale=packed_w2_scale,
        micro_w2_global_scale=_process_nvfp4_micro_global_scale_from_packed(
            packed_w2_global_scale,
            a_dtype=params_dtype,
        ),
        w13_layout=w13_layout,
    )


def prepare_w4a16_e8m0_native_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
) -> W4A16ModelOptWeights:
    """Prepare native MXFP4 (E8M0 K/32) weights for the W4A16 path.

    Keeps the FP4 weights resident as a single copy (``weight_layout="modelopt"``)
    and carries two small scale forms so one object serves both kernels:
    ``w13_scale``/``w2_scale`` are the packed E8M0 grid the main W4A16 GEMM reads
    at med/large M, and ``micro_*`` are packed E8M0 grids in the native row order
    that the small-M micro decode kernel reads. ``run_w4a16_moe`` routes small M
    to micro and the rest to the main W4A16 kernel automatically.
    """
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated
    allow_w2_k_tail = intermediate_size % 32 != 0
    padded_w13_scale_n = (
        _e8m0_logical_tail_scale_n(w13_rows) if allow_w2_k_tail else w13_rows
    )

    w13_scale = _validate_e8m0_k32_scales(
        w13_e8m0_scale,
        rows=w13_rows,
        cols=hidden_size,
        name="w13_e8m0_scale",
    )
    w2_scale = _validate_e8m0_k32_scales(
        w2_e8m0_scale,
        rows=hidden_size,
        cols=intermediate_size,
        name="w2_e8m0_scale",
        allow_k_tail=allow_w2_k_tail,
    )
    # Main-GEMM (med/large M) packed E8M0 scales. The "w13" (up_gate) layout
    # needs the FC1 half-swap folded into the scale grid; the kernel applies the
    # matching source_n_rotation to the native weights. micro reads the un-rotated
    # grid and handles the layout itself (w13_gate_first).
    w13_row_rotation = intermediate_size if (is_gated and w13_layout == "w13") else None
    packed_w13_scale = _pack_e8m0_k32_scales(
        w13_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        padded_size_n=padded_w13_scale_n,
    )
    micro_w13_scale = (
        packed_w13_scale
        if w13_row_rotation is None
        else _pack_e8m0_k32_scales(
            w13_scale,
            size_k=hidden_size,
            size_n=w13_rows,
            padded_size_n=padded_w13_scale_n,
        )
    )
    packed_w2_scale = _pack_e8m0_k32_scales(
        w2_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        allow_k_tail=allow_w2_k_tail,
    )
    # Storage-compatible single grid: micro reads the SAME packed _pack_e8m0_k32
    # scales the main GEMM reads (no separate K/16 micro grid).
    w13_global = w13_global_scale.contiguous()
    w2_global = w2_global_scale.contiguous()
    return W4A16ModelOptWeights(
        w13=w13_fp4,
        w13_scale=packed_w13_scale,
        w13_global_scale=w13_global,
        w2=w2_fp4,
        w2_scale=packed_w2_scale,
        w2_global_scale=w2_global,
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
        scale_format="e8m0_k32",
        micro_w13_scale=micro_w13_scale,
        micro_w13_global_scale=w13_global,
        micro_w2_scale=packed_w2_scale,
        micro_w2_global_scale=w2_global,
        w13_layout=w13_layout,
    )


def prepare_w4a16_compressed_tensors_weights(
    w13_fp4: torch.Tensor,
    w13_blockscale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_blockscale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare CompressedTensors NVFP4 tensors into the W4A16 packed runtime layout.

    The per-block scales are the normal NVFP4 K/16 scale grid in sm12x swizzled
    storage. The CT global scales are stored inverted relative to the ModelOpt
    weight global scale convention, so they are inverted before packing.
    """
    return _prepare_w4a16_packed_weights(
        w13_fp4,
        w13_blockscale,
        w13_global_scale,
        w2_fp4,
        w2_blockscale,
        w2_global_scale,
        activation=activation,
        params_dtype=params_dtype,
        source_format="compressed_tensors",
        w13_layout=w13_layout,
        reuse_input_storage=reuse_input_storage,
    )


def prepare_w4a16_fp4_e8m0_k32_weights(
    w13_fp4: torch.Tensor,
    w13_e8m0_scale: torch.Tensor,
    w13_global_scale: torch.Tensor,
    w2_fp4: torch.Tensor,
    w2_e8m0_scale: torch.Tensor,
    w2_global_scale: torch.Tensor,
    *,
    activation: str,
    params_dtype: torch.dtype = torch.bfloat16,
    w13_layout: str = "w13",
    reuse_input_storage: bool = False,
) -> W4A16PackedWeights:
    """Prepare FP4 weights with E8M0 K/32 scales for W4A16.

    The per-block source scales are [E, N, K/32] E8M0 bytes. They are only
    saturated to the BF16 kernel's supported byte range and rearranged for
    kernel access; they are not expanded to K/16 or folded into global scales.
    """
    w13_layout = _normalize_w13_layout(w13_layout)
    shape = validate_w4a16_packed_inputs(
        w13_fp4,
        w13_global_scale,
        w2_fp4,
        w2_global_scale,
        activation=activation,
    )
    num_experts = shape.num_experts
    hidden_size = shape.hidden_size
    intermediate_size = shape.intermediate_size
    w13_rows = shape.w13_rows
    is_gated = shape.is_gated

    w13 = w13_fp4
    w13_scale = _validate_e8m0_k32_scales(
        w13_e8m0_scale,
        rows=w13_rows,
        cols=hidden_size,
        name="w13_e8m0_scale",
    )
    w13_row_rotation = None
    if is_gated and w13_layout != "w31":
        w13_row_rotation = intermediate_size

    w2_scale = _validate_e8m0_k32_scales(
        w2_e8m0_scale,
        rows=hidden_size,
        cols=intermediate_size,
        name="w2_e8m0_scale",
    )

    packed_w13 = _repack_weight(
        w13 if reuse_input_storage else w13.contiguous(),
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2 = _repack_weight(
        w2_fp4 if reuse_input_storage else w2_fp4.contiguous(),
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w13_scale = _pack_e8m0_k32_scales(
        w13_scale,
        size_k=hidden_size,
        size_n=w13_rows,
        row_rotation=w13_row_rotation,
        reuse_input_storage=reuse_input_storage,
    )
    packed_w2_scale = _pack_e8m0_k32_scales(
        w2_scale,
        size_k=intermediate_size,
        size_n=hidden_size,
        reuse_input_storage=reuse_input_storage,
    )

    return W4A16PackedWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=w13_global_scale.contiguous(),
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=w2_global_scale.contiguous(),
        workspace=_make_workspace(w13_fp4.device, max_blocks_per_sm=4),
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        is_gated=is_gated,
        params_dtype=params_dtype,
        source_format="fp4_e8m0_k32",
        w13_layout=w13_layout,
        scale_format="e8m0_k32",
    )


def prepare_w4a16_packed_weights(
    *args,
    source_format: str = "modelopt_nvfp4",
    w13_layout: str = "w13",
    **kwargs,
) -> W4A16PackedWeights:
    source_format = _normalize_source_format(source_format)
    w13_layout = _normalize_w13_layout(w13_layout)
    if source_format == "modelopt_nvfp4":
        return prepare_w4a16_modelopt_nvfp4_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    if source_format == "compressed_tensors":
        return prepare_w4a16_compressed_tensors_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    if source_format == "fp4_e8m0_k32":
        return prepare_w4a16_fp4_e8m0_k32_weights(
            *args, w13_layout=w13_layout, **kwargs
        )
    raise AssertionError(f"unhandled W4A16 source_format {source_format!r}")


def make_w4a16_packed_buffers(
    prepared: W4A16PackedWeights | W4A16ModelOptWeights,
    *,
    m: int,
    topk: int,
    dtype: torch.dtype,
    device: torch.device,
    route_num_experts: int | None = None,
) -> W4A16PackedBuffers:
    return _make_w4a16_packed_buffers(
        prepared,
        m=m,
        topk=topk,
        dtype=dtype,
        device=device,
        route_num_experts=route_num_experts,
    )


def prepare_nf3_moe_weights(
    w13_codes: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_codes: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    activation: str,
    fc1_tile_n: int,
    fc2_tile_n: int,
    params_dtype: torch.dtype = torch.bfloat16,
) -> PreparedNF3MoeWeights:
    """Pack NF3 codes/scales into the runtime "nf3_2p1" W4A16 layout.

    Inputs are per-expert integer codes (0..7) in kernel-native output order
    (gate/up already resolved for FC1) and per-group float scales ``t_s``:
        w13_codes [E, 2*I, hidden]   w13_scale [E, 2*I, hidden//32]
        w2_codes  [E, hidden, I]     w2_scale  [E, hidden, I//32]
    ``fc1_tile_n`` / ``fc2_tile_n`` MUST equal the CTA N-tiles the kernel will
    use (read them from the compiled W4A16FusedMoeCompileResult) -- the
    flat-span layout is tile_n specific.
    """
    if params_dtype != torch.bfloat16:
        raise ValueError("nf3_2p1 W4A16 weights are bf16-only for v1")
    shape = validate_nf3_moe_inputs(
        w13_codes,
        w13_scale,
        w2_codes,
        w2_scale,
        activation=activation,
    )
    device = w13_codes.device
    packed_w13 = _nf3_pack_code_experts(
        w13_codes, size_k=shape.hidden_size, size_n=shape.w13_rows, tile_n=fc1_tile_n
    )
    packed_w2 = _nf3_pack_code_experts(
        w2_codes,
        size_k=shape.intermediate_size,
        size_n=shape.hidden_size,
        tile_n=fc2_tile_n,
    )
    packed_w13_scale = _nf3_pack_scale_experts(
        w13_scale, size_k=shape.hidden_size, size_n=shape.w13_rows
    )
    packed_w2_scale = _nf3_pack_scale_experts(
        w2_scale, size_k=shape.intermediate_size, size_n=shape.hidden_size
    )
    global_scale = torch.full(
        (shape.num_experts,), _NF3_SCALE_GLOBAL, dtype=torch.float32, device=device
    )
    return PreparedNF3MoeWeights(
        w13=packed_w13,
        w13_scale=packed_w13_scale,
        w13_global_scale=global_scale,
        w2=packed_w2,
        w2_scale=packed_w2_scale,
        w2_global_scale=global_scale.clone(),
        workspace=_make_workspace(device, max_blocks_per_sm=4),
        hidden_size=shape.hidden_size,
        intermediate_size=shape.intermediate_size,
        num_experts=shape.num_experts,
        is_gated=shape.is_gated,
        params_dtype=params_dtype,
        fc1_tile_n=int(fc1_tile_n),
        fc2_tile_n=int(fc2_tile_n),
    )


def _nf3_pack_selftest() -> None:
    """Prove the packer matches the kernel's read+dequant chain (torch/CPU).

    Packs random codes, then INDEPENDENTLY simulates the kernel's flat-span
    staging + register read (unit = cur_group_id*(tile_n//2) + tid%(tile_n//2))
    + packed_dequant_nf3x8 fragment placement in pure torch, and asserts the
    reconstructed dequantized matrix equals codebook[codes] at every (N, K).
    """
    codebook = _NF3_CODEBOOK

    def simulate(packed, size_k, size_n, tile_n, tile_k):
        cta_threads = tile_n * tile_k // 64
        b_sh_stride = tile_n // 2
        cta_k_blocks = tile_k // 16
        b_sh_stage = b_sh_stride * cta_k_blocks
        b_sh_wr_iters = b_sh_stage // cta_threads
        tb_n_warps = tile_n // 64
        b_chunks = b_sh_stage * 12 // 16
        wr_iters_nf3 = (b_chunks + cta_threads - 1) // cta_threads
        ntile_stride = (size_k // 16) * (tile_n // 2)
        n_tiles = size_n // tile_n
        k_tiles = size_k // tile_k
        packed = packed.tolist()
        w = [[None] * size_n for _ in range(size_k)]

        def u32(x):
            return x & 0xFFFFFFFF

        for nt in range(n_tiles):
            for tile_idx in range(k_tiles):
                span_base_unit = (
                    nt * ntile_stride + tile_idx * cta_k_blocks * b_sh_stride
                )
                smem = {}
                for i in range(wr_iters_nf3):
                    for tid in range(cta_threads):
                        c = i * cta_threads + tid
                        if c >= b_chunks:
                            continue
                        for word in range(4):
                            smem[c * 4 + word] = u32(
                                packed[span_base_unit * 3 + c * 4 + word]
                            )
                for tid in range(cta_threads):
                    warp_row = (tid // 32) // tb_n_warps
                    for kk in range(b_sh_wr_iters):
                        cur = b_sh_wr_iters * warp_row + kk
                        unit = cur * b_sh_stride + (tid % b_sh_stride)
                        lo0 = smem[unit * 3 + 0]
                        lo1 = smem[unit * 3 + 1]
                        hi = smem[unit * 3 + 2]
                        R = tile_idx * cta_k_blocks + cur
                        p = tid % b_sh_stride
                        for jj in range(4):
                            lo_w = lo0 if (jj // 2) == 0 else lo1
                            lo16 = (lo_w >> (16 * (jj % 2))) & 0xFFFF
                            hi8 = (hi >> (8 * jj)) & 0xFF
                            la = lo16 & 0xFF
                            lb = (lo16 >> 8) & 0xFF
                            ha = hi8 & 0xF
                            hb = (hi8 >> 4) & 0xF

                            def code(byte2, nib1, j):
                                return ((byte2 >> (2 * j)) & 3) | (
                                    ((nib1 >> j) & 1) << 2
                                )

                            cds = [code(la, ha, j) for j in range(4)] + [
                                code(lb, hb, j) for j in range(4)
                            ]
                            wru = p + (tile_n // 2) * nt
                            n64 = wru // 32
                            th = wru % 32
                            tc_col = th // 4
                            tc_row = (th % 4) * 2
                            col1 = jj * 16 + tc_col
                            c1 = n64 * 64 + col1
                            c2 = c1 + 8
                            k0 = R * 16 + tc_row
                            w[k0 + 0][c1] = codebook[cds[0]]
                            w[k0 + 1][c1] = codebook[cds[1]]
                            w[k0 + 8][c1] = codebook[cds[2]]
                            w[k0 + 9][c1] = codebook[cds[3]]
                            w[k0 + 0][c2] = codebook[cds[4]]
                            w[k0 + 1][c2] = codebook[cds[5]]
                            w[k0 + 8][c2] = codebook[cds[6]]
                            w[k0 + 9][c2] = codebook[cds[7]]
        return w

    torch.manual_seed(0)
    cases = [
        (256, 128, 128, 128),
        (256, 128, 128, 64),
        (64, 256, 256, 64),
        (512, 256, 256, 64),
    ]
    for size_k, size_n, tile_n, tile_k in cases:
        codes = torch.randint(0, 8, (size_n, size_k), dtype=torch.int64)
        packed = _nf3_pack_codes(codes, size_k=size_k, size_n=size_n, tile_n=tile_n)
        w = simulate(packed, size_k, size_n, tile_n, tile_k)
        for k in range(size_k):
            for n in range(size_n):
                expect = _NF3_CODEBOOK[int(codes[n, k])]
                got = w[k][n]
                assert got is not None, (
                    f"unfilled ({k},{n}) for {(size_k, size_n, tile_n, tile_k)}"
                )
                assert abs(got - expect) < 1e-9, (
                    f"mismatch at ({k},{n}) for {(size_k, size_n, tile_n, tile_k)}: "
                    f"{got} != {expect}"
                )
    print("nf3 pack self-test PASSED")


__all__ = [
    "PreparedNF3MoeWeights",
    "W4A16PackedBuffers",
    "W4A16ModelOptWeights",
    "W4A16PackedWeights",
    "make_w4a16_packed_buffers",
    "prepare_nf3_moe_weights",
    "prepare_w4a16_compressed_tensors_weights",
    "prepare_w4a16_e8m0_native_weights",
    "prepare_w4a16_fp4_e8m0_k32_weights",
    "prepare_w4a16_modelopt_native_weights",
    "prepare_w4a16_modelopt_nvfp4_weights",
    "prepare_w4a16_packed_weights",
]
