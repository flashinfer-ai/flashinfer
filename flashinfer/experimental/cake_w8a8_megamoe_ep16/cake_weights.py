"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# Weight preprocessing for the Cake W8A8 MXFP8 MegaMoE EP16 route.
#
# Canonical input: :class:`~flashinfer.moe_ep.weights.UnquantizedMoEWeights`
# with bf16 ``w13 [32, 10240, 3072]`` (gate rows then up rows) and
# ``w2 [32, 3072, 5120]`` for the rank's 32 experts. The route's own
# quantization program converts them to block-32 MXFP8 (e4m3 values, one
# UE8M0 scale byte per 32-wide K block) in the two kernel-native scale
# layouts: mode 1 for ``w13`` (one 1024-byte scale tile per 256-row x
# 128-column weight tile, gate||up interleaved as up128|gate128 per 256-row
# chunk) and mode 2 for ``w2`` (one 512-byte scale tile per 128-row x
# 128-column weight tile, consumed by the SwapAB FC2). Running the same
# program FlashInfer ships keeps the weights bit-identical to the measured
# route; :func:`pack_scales_cp` is the pure-Torch statement of both layouts
# (tests). Pre-quantized weights are not accepted.
#
# FP8 payload placement: the program's trailing ``ktile_major_rows`` scalar
# stores the e4m3 bytes of a ``[experts, rows, K]`` weight k-tile-major, as
# ``[experts][rows / R][K / 128][R][128 B]`` with ``R = 256`` for ``w13``
# (10240 packed rows, K = 3072) and ``R = 128`` for ``w2`` (3072 rows,
# K = 5120); the host tensors keep their contiguous ``[experts, rows, K]``
# shape and the FC1/FC2 A tensor maps load one contiguous ``R x 128``-byte
# block per box. The exported FC1/FC2 programs address only this placement,
# so :class:`TransformedCakeW8a8MegaWeights` carries the layout tags the
# manifest names (``w13_fp8_layout`` / ``w2_fp8_layout``) and the validator
# rejects anything else. :func:`reference_ktile_major_payload` is the
# pure-Torch statement of the placement (tests).

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ...moe_ep.core.validation.common import MoEEpConfigError
from ...moe_ep.weights import (
    MoEWeightPack,
    PrequantizedMoEWeights,
    UnquantizedMoEWeights,
)
from .cake_config import KERNEL_NAME
from .cake_plan import (
    FC1_ROWS,
    HIDDEN,
    INTERMEDIATE,
    LOCAL_EXPERTS,
    W2_SCALE_BYTES,
    W13_SCALE_BYTES,
    quantize_grid,
)

FP8_MAX = 448.0
SCALE_BLOCK = 32
W13_SHAPE = (LOCAL_EXPERTS, FC1_ROWS, HIDDEN)
W2_SHAPE = (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE)
# Scale layout modes of the quantization program's ``scale_tile_n256_k128``.
SCALE_MODE_W13 = 1
SCALE_MODE_W2_SWAP = 2
SCALE_TILE_BYTES = {SCALE_MODE_W13: 1024, SCALE_MODE_W2_SWAP: 512}
SCALE_TILE_ROWS = {SCALE_MODE_W13: 256, SCALE_MODE_W2_SWAP: 128}
SCALE_TILE_BLOCKS = 4  # 32-wide blocks per 128-column scale tile
# FP8 payload placement (the program's ``ktile_major_rows`` scalar, the box
# height of the FC1/FC2 A tensor maps) and the layout tags of the manifest's
# ``route_contract.w13_fp8_layout`` / ``w2_fp8_layout``.
W13_KTILE_ROWS = 256
W2_KTILE_ROWS = 128
W13_FP8_LAYOUT = "ktile_major_r256_k128"
W2_FP8_LAYOUT = "ktile_major_r128_k128"
KTILE_COLUMNS = 128  # bytes of K per k-tile-major block row
# The quantization program's amax floor and scale-code clamp (UE8M0 codes).
AMAX_FLOOR = 1e-7
SCALE_CODE_MIN = 1
SCALE_CODE_MAX = 254
# tvm-ffi entry name of the weight-quantization program binding. The contract
# is the binding file name and its export:
# ``cake_w8a8_megamoe_ep16_quantize_weights_binding.cu`` exports exactly one
# function named ``quantize_weights``.
QUANTIZE_ENTRY = "quantize_weights"


@dataclass(frozen=True)
class TransformedCakeW8a8MegaWeights:
    """Kernel-layout rank-local weights (TMA sources owned by this object).

    ``w13_layout`` / ``w2_layout`` name the FP8 payload placement of ``w13``
    and ``w2`` (``W13_FP8_LAYOUT`` = k-tile-major in 256-row blocks,
    ``W2_FP8_LAYOUT`` = k-tile-major in 128-row blocks). The route's FC1/FC2
    programs address only these placements; :func:`preprocess_mega_weights`
    sets the tags and :func:`validate_transformed_mega_weights` requires them.
    """

    w13: torch.Tensor  # float8_e4m3fn [32, 10240, 3072], up128|gate128 per 256 rows
    w13_scales: torch.Tensor  # uint8 [31,457,280] mode-1 scale tiles (1024 B each)
    w2: torch.Tensor  # float8_e4m3fn [32, 3072, 5120]
    w2_scales_swap: torch.Tensor  # uint8 [15,728,640] mode-2 scale tiles (512 B each)
    w13_layout: str  # W13_FP8_LAYOUT: payload of w13 as [32][40][24][256][128 B]
    w2_layout: str  # W2_FP8_LAYOUT: payload of w2 as [32][24][40][128][128 B]


def _require(
    tensor: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device | None,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise MoEEpConfigError(f"{KERNEL_NAME}: {name} must be a tensor")
    if tuple(tensor.shape) != shape:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: {name} must have shape {shape}, got {tuple(tensor.shape)}"
        )
    if tensor.dtype != dtype:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: {name} must have dtype {dtype}, got {tensor.dtype}"
        )
    if not tensor.is_cuda:
        raise MoEEpConfigError(f"{KERNEL_NAME}: {name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise MoEEpConfigError(f"{KERNEL_NAME}: {name} must be contiguous")
    if device is not None and tensor.device != device:
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: {name} must be on {device}, got {tensor.device}"
        )


def validate_canonical_weights(weights: MoEWeightPack) -> torch.device:
    """Check the bf16 pack this route accepts; return its device."""

    if isinstance(weights, PrequantizedMoEWeights):
        raise NotImplementedError(
            f"{KERNEL_NAME} quantizes bf16 weights with its own program; "
            "PrequantizedMoEWeights are not supported"
        )
    if not isinstance(weights, UnquantizedMoEWeights):
        raise TypeError(
            f"{KERNEL_NAME}: weights must be UnquantizedMoEWeights, got {type(weights).__name__}"
        )
    _require(
        weights.w13, name="w13", shape=W13_SHAPE, dtype=torch.bfloat16, device=None
    )
    _require(
        weights.w2,
        name="w2",
        shape=W2_SHAPE,
        dtype=torch.bfloat16,
        device=weights.w13.device,
    )
    return weights.w13.device


# ---------------------------------------------------------------------------
# Setup-program call interface.
# ---------------------------------------------------------------------------
def launch_quantize_weights(
    module: Any,
    *,
    source: torch.Tensor,
    quantized: torch.Tensor,
    scales_cp: torch.Tensor,
    rows: int,
    columns: int,
    gate_width: int,
    interleave_gate_up: int,
    scale_tile_n256_k128: int,
    ktile_major_rows: int,
    grid_x: int,
) -> None:
    """One launch of the weight-quantization program through its binding.

    Argument order of ``QUANTIZE_ENTRY``: the program's device arguments in
    the manifest's physical order (tensors, then int64 scalars), then the
    launch grid::

        source               bf16   [experts, rows, columns]
        quantized            e4m3   [experts, rows, columns]
        scales_cp            uint8  [scale_bytes(rows, columns)]
        experts              int64  = 32
        rows                 int64
        columns              int64
        gate_width           int64  (5120 for w13, 0 for w2)
        interleave_gate_up   int64  (1 for w13, 0 for w2)
        scale_tile_n256_k128 int64  (1 = mode 1, 2 = mode 2)
        ktile_major_rows     int64  (256 for w13, 128 for w2; 0 = row-major payload)
        grid_x               int64  = quantize_grid(device)  (block = 256 threads)

    ``quantized`` is the e4m3 output viewed by the binding as bytes; with
    ``ktile_major_rows = R > 0`` its bytes are placed k-tile-major
    (:func:`reference_ktile_major_payload`), the scale bytes are unaffected.
    """

    getattr(module, QUANTIZE_ENTRY)(
        source,
        quantized,
        scales_cp,
        LOCAL_EXPERTS,
        rows,
        columns,
        gate_width,
        interleave_gate_up,
        scale_tile_n256_k128,
        ktile_major_rows,
        grid_x,
    )


def preprocess_mega_weights(
    weights: MoEWeightPack,
    *,
    module: Any,
) -> TransformedCakeW8a8MegaWeights:
    """bf16 ``w13``/``w2`` -> kernel-layout MXFP8 weights, via the route's program.

    Two launches (w13 in mode 1 with 256-row k-tile-major blocks, w2 in mode
    2 with 128-row blocks) on the current stream of the weights' device, then
    a device synchronization: this is setup work, not the submission path.
    ``module`` is the loaded JIT module.
    """

    import tvm_ffi

    device = validate_canonical_weights(weights)
    if torch.cuda.is_current_stream_capturing():
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: weight preprocessing cannot run during CUDA graph capture"
        )
    w13 = torch.empty(W13_SHAPE, dtype=torch.float8_e4m3fn, device=device)
    w13_scales = torch.empty((W13_SCALE_BYTES,), dtype=torch.uint8, device=device)
    w2 = torch.empty(W2_SHAPE, dtype=torch.float8_e4m3fn, device=device)
    w2_scales_swap = torch.empty((W2_SCALE_BYTES,), dtype=torch.uint8, device=device)
    grid_x = quantize_grid(device.index)
    with torch.cuda.device(device), tvm_ffi.use_torch_stream():
        launch_quantize_weights(
            module,
            source=weights.w13,
            quantized=w13,
            scales_cp=w13_scales,
            rows=FC1_ROWS,
            columns=HIDDEN,
            gate_width=INTERMEDIATE,
            interleave_gate_up=1,
            scale_tile_n256_k128=SCALE_MODE_W13,
            ktile_major_rows=W13_KTILE_ROWS,
            grid_x=grid_x,
        )
        launch_quantize_weights(
            module,
            source=weights.w2,
            quantized=w2,
            scales_cp=w2_scales_swap,
            rows=HIDDEN,
            columns=INTERMEDIATE,
            gate_width=0,
            interleave_gate_up=0,
            scale_tile_n256_k128=SCALE_MODE_W2_SWAP,
            ktile_major_rows=W2_KTILE_ROWS,
            grid_x=grid_x,
        )
    torch.cuda.synchronize(device)
    return TransformedCakeW8a8MegaWeights(
        w13=w13,
        w13_scales=w13_scales,
        w2=w2,
        w2_scales_swap=w2_scales_swap,
        w13_layout=W13_FP8_LAYOUT,
        w2_layout=W2_FP8_LAYOUT,
    )


def validate_transformed_mega_weights(
    transformed: Any, *, device: torch.device | None = None
) -> None:
    """One-time check of kernel-ready weights (``preprocess_weights=False``).

    The payload layout tags are checked first: the route's FC1/FC2 programs
    address only the k-tile-major placements ``W13_FP8_LAYOUT`` and
    ``W2_FP8_LAYOUT``; a row-major or otherwise tagged payload is rejected.
    """

    if not isinstance(transformed, TransformedCakeW8a8MegaWeights):
        raise MoEEpConfigError(
            f"{KERNEL_NAME}: transformed weights must be TransformedCakeW8a8MegaWeights, "
            f"got {type(transformed).__name__}"
        )
    for name, layout, expected in (
        ("w13_layout", transformed.w13_layout, W13_FP8_LAYOUT),
        ("w2_layout", transformed.w2_layout, W2_FP8_LAYOUT),
    ):
        if layout != expected:
            raise MoEEpConfigError(
                f"{KERNEL_NAME}: {name} must be {expected!r} (the k-tile-major FP8 "
                f"payload the route's FC1/FC2 programs address), got {layout!r}; "
                "run preprocess_weights on bf16 weights to produce it"
            )
    device = transformed.w13.device if device is None else device
    _require(
        transformed.w13,
        name="w13",
        shape=W13_SHAPE,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _require(
        transformed.w13_scales,
        name="w13_scales",
        shape=(W13_SCALE_BYTES,),
        dtype=torch.uint8,
        device=device,
    )
    _require(
        transformed.w2,
        name="w2",
        shape=W2_SHAPE,
        dtype=torch.float8_e4m3fn,
        device=device,
    )
    _require(
        transformed.w2_scales_swap,
        name="w2_scales_swap",
        shape=(W2_SCALE_BYTES,),
        dtype=torch.uint8,
        device=device,
    )


# ---------------------------------------------------------------------------
# Pure-Torch reference of the quantization rule (tests only; importable on CPU).
# ---------------------------------------------------------------------------
def reference_quantize_mxfp8_block32(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Block-32 MXFP8 along the last dim: (e4m3 values, UE8M0 scale codes).

    The rule of the quantization program: the block amax is floored at
    ``AMAX_FLOOR`` (1e-7), the scale is the power of two ``>= amax / 448``
    (``ceil(log2(amax / 448)) + 127`` as a UE8M0 code, computed from the
    float32 bits) clamped to ``[1, 254]``, so an all-zero block gets code 95
    and the value 0. Values are divided by the scale and rounded to e4m3. The
    returned codes have shape ``[..., columns // 32]`` in plain row-major
    order; :func:`pack_scales_cp` applies the kernel layout. The program
    forms ``amax / 448`` with an approximate reciprocal, so a block whose
    amax is exactly ``448 * 2**k`` may differ by one code.
    """

    if weight.shape[-1] % SCALE_BLOCK:
        raise ValueError(
            f"MXFP8 reduction dimension must be divisible by {SCALE_BLOCK}, "
            f"got {weight.shape[-1]}"
        )
    lead = tuple(weight.shape[:-1])
    columns = int(weight.shape[-1])
    grouped = weight.float().reshape(*lead, columns // SCALE_BLOCK, SCALE_BLOCK)
    amax = grouped.abs().amax(dim=-1).clamp_min(AMAX_FLOOR)
    bits = (amax / FP8_MAX).contiguous().view(torch.int32)
    exponent = (((bits >> 23) & 255) + (((bits & 0x7FFFFF) + 0x7FFFFF) >> 23)).clamp(
        min=SCALE_CODE_MIN, max=SCALE_CODE_MAX
    )
    inverse = ((254 - exponent) << 23).view(torch.float32)
    values = (grouped * inverse.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return values.reshape(*lead, columns), exponent.to(torch.uint8)


def reference_interleave_gate_up_128(w13: torch.Tensor) -> torch.Tensor:
    """Canonical gate||up rows -> up128|gate128 per 256-row chunk.

    The row rule of the quantization program in mode 1 with
    ``interleave_gate_up=1`` and ``gate_width=5120``: output row ``r`` of a
    256-row chunk ``c`` reads canonical row ``gate_width + 128c + r`` (an up
    row) when ``r < 128``, else canonical row ``128c + (r - 128)`` (a gate
    row). Each 256-row chunk is therefore ``up[128c : 128c + 128]`` followed
    by ``gate[128c : 128c + 128]``; the SwiGLU epilogue applies SiLU to the
    second half.
    """

    experts, rows, hidden = w13.shape
    if rows % 256:
        raise ValueError("gate/up rows must be divisible by 256")
    half = rows // 2
    gate = w13[:, :half].reshape(experts, half // 128, 128, hidden)
    up = w13[:, half:].reshape(experts, half // 128, 128, hidden)
    return torch.stack((up, gate), dim=2).reshape(experts, rows, hidden)


def _check_ktile_major_shape(q: torch.Tensor, box_rows: int) -> tuple[int, int, int]:
    if q.dim() != 3:
        raise ValueError(
            f"weight payload must be [experts, rows, K], got {tuple(q.shape)}"
        )
    experts, rows, k = (int(n) for n in q.shape)
    if box_rows <= 0 or rows % box_rows or k % KTILE_COLUMNS:
        raise ValueError(
            f"k-tile-major placement needs rows % {box_rows} == 0 and K % "
            f"{KTILE_COLUMNS} == 0, got ({rows}, {k})"
        )
    return experts, rows, k


def reference_ktile_major_payload(q: torch.Tensor, *, box_rows: int) -> torch.Tensor:
    """Row-major quantized bytes ``[E, rows, K]`` -> the k-tile-major payload.

    The placement rule of the quantization program with ``ktile_major_rows =
    R`` (``box_rows``): byte ``(e, r, k)`` of the row-major weight is stored
    at ``(((e * (rows // R) + r // R) * (K // 128) + k // 128) * R + r % R) *
    128 + k % 128``, i.e. ``[E][rows // R][K // 128][R][128 B]``, one
    contiguous ``R x 128``-byte block per FC1/FC2 TMA box. The result keeps
    the ``[E, rows, K]`` shape and dtype of ``q``; ``R = 256`` for ``w13``
    and ``R = 128`` for ``w2``.
    """

    experts, rows, k = _check_ktile_major_shape(q, box_rows)
    return (
        q.reshape(
            experts, rows // box_rows, box_rows, k // KTILE_COLUMNS, KTILE_COLUMNS
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, rows, k)
    )


def reference_row_major_payload(
    payload: torch.Tensor, *, box_rows: int
) -> torch.Tensor:
    """Inverse of :func:`reference_ktile_major_payload`."""

    experts, rows, k = _check_ktile_major_shape(payload, box_rows)
    return (
        payload.reshape(
            experts, rows // box_rows, k // KTILE_COLUMNS, box_rows, KTILE_COLUMNS
        )
        .permute(0, 1, 3, 2, 4)
        .contiguous()
        .view(experts, rows, k)
    )


def pack_scales_cp(codes: torch.Tensor, *, mode: int) -> torch.Tensor:
    """Kernel layout of plain scale codes ``[experts, rows, columns // 32]``.

    Both modes tile the scale plane by 128 columns (four 32-wide blocks,
    index ``u``) and store one contiguous tile per ``(expert, tile_m,
    tile_k)`` in row-major tile order. Within a tile the byte offset is

    * mode 1 (``w13``; 256 rows per tile, 1024 bytes):
      ``d * 512 + row32 * 16 + a * 4 + u`` with ``d`` the 128-row half,
      ``a`` the 32-row group inside it and ``row32`` the row inside the group;
    * mode 2 (``w2`` SwapAB; 128 rows per tile, 512 bytes):
      ``row32 * 16 + a * 4 + u``.

    The result is a flat uint8 tensor of ``rows * columns // 32`` bytes per
    expert (no padding), a permutation of ``codes``; ``cake_plan.scale_bytes``
    gives the same size.
    """

    if mode not in SCALE_TILE_BYTES:
        raise ValueError(f"scale layout mode must be 1 or 2, got {mode!r}")
    if codes.dim() != 3:
        raise ValueError(
            f"scale codes must be [experts, rows, blocks], got {tuple(codes.shape)}"
        )
    experts, rows, blocks = (int(n) for n in codes.shape)
    tile_rows = SCALE_TILE_ROWS[mode]
    if rows % tile_rows or blocks % SCALE_TILE_BLOCKS:
        raise ValueError(
            f"mode {mode} needs rows % {tile_rows} == 0 and blocks % "
            f"{SCALE_TILE_BLOCKS} == 0, got ({rows}, {blocks})"
        )
    tiles_m, tiles_k = rows // tile_rows, blocks // SCALE_TILE_BLOCKS
    if mode == SCALE_MODE_W13:
        # rows = tile_m, d, a, row32 ; blocks = tile_k, u
        packed = codes.reshape(experts, tiles_m, 2, 4, 32, tiles_k, 4).permute(
            0, 1, 5, 2, 4, 3, 6
        )
    else:
        # rows = tile_m, a, row32 ; blocks = tile_k, u
        packed = codes.reshape(experts, tiles_m, 4, 32, tiles_k, 4).permute(
            0, 1, 4, 3, 2, 5
        )
    return packed.reshape(-1).contiguous()


def unpack_scales_cp(
    scales_cp: torch.Tensor, *, mode: int, rows: int, columns: int
) -> torch.Tensor:
    """Inverse of :func:`pack_scales_cp`: kernel bytes -> ``[experts, rows, columns // 32]``."""

    if mode not in SCALE_TILE_BYTES:
        raise ValueError(f"scale layout mode must be 1 or 2, got {mode!r}")
    blocks = columns // SCALE_BLOCK
    tile_rows = SCALE_TILE_ROWS[mode]
    if columns % SCALE_BLOCK or rows % tile_rows or blocks % SCALE_TILE_BLOCKS:
        raise ValueError(
            f"mode {mode} needs rows % {tile_rows} == 0 and columns % 128 == 0, "
            f"got ({rows}, {columns})"
        )
    tiles_m, tiles_k = rows // tile_rows, blocks // SCALE_TILE_BLOCKS
    per_expert = rows * blocks
    if scales_cp.dim() != 1 or scales_cp.numel() % per_expert:
        raise ValueError(
            f"scale bytes must be flat with a multiple of {per_expert} entries, "
            f"got {tuple(scales_cp.shape)}"
        )
    experts = scales_cp.numel() // per_expert
    if mode == SCALE_MODE_W13:
        tiles = scales_cp.reshape(experts, tiles_m, tiles_k, 2, 32, 4, 4)
        codes = tiles.permute(0, 1, 3, 5, 4, 2, 6)
    else:
        tiles = scales_cp.reshape(experts, tiles_m, tiles_k, 32, 4, 4)
        codes = tiles.permute(0, 1, 4, 3, 2, 5)
    return codes.reshape(experts, rows, blocks).contiguous()


__all__ = [
    "AMAX_FLOOR",
    "KTILE_COLUMNS",
    "QUANTIZE_ENTRY",
    "SCALE_CODE_MAX",
    "SCALE_CODE_MIN",
    "SCALE_MODE_W13",
    "SCALE_MODE_W2_SWAP",
    "SCALE_TILE_BYTES",
    "TransformedCakeW8a8MegaWeights",
    "W13_FP8_LAYOUT",
    "W13_KTILE_ROWS",
    "W13_SHAPE",
    "W2_FP8_LAYOUT",
    "W2_KTILE_ROWS",
    "W2_SHAPE",
    "launch_quantize_weights",
    "pack_scales_cp",
    "preprocess_mega_weights",
    "reference_interleave_gate_up_128",
    "reference_ktile_major_payload",
    "reference_quantize_mxfp8_block32",
    "reference_row_major_payload",
    "unpack_scales_cp",
    "validate_canonical_weights",
    "validate_transformed_mega_weights",
]
