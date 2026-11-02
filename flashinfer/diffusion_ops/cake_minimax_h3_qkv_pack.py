# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 one-pass QKV quantize-and-pack operation.

BF16 ``Q``, ``K``, ``V`` ``[M, 56, 128]`` (three contiguous tensors or the
kind slices of one ``[M, 56, 3, 128]`` projection output) are routed into the
destination-major Ulysses send buffer ``[P, M, 56 / P, 3, 128]`` and quantized
per destination in the same HBM pass:

* ``"nvfp4"``: E2M1 nibble pairs ``uint8 [P, M, 56 / P, 3, 64]`` plus block-16
  E4M3 scales ``uint8 [P, round_up(R, 128) * 8]`` in the swizzled 128x4 layout
  (``R = M * (56 / P) * 3``), with a caller-supplied static global encode
  scale.  Byte for byte the send-buffer ABI of
  :class:`flashinfer.MiniMaxH3Nvfp4PreAttention`.
* ``"mxfp8"``: ``float8_e4m3fn [P, M, 56 / P, 3, 128]`` plus block-32 UE8M0
  scales ``uint8 [P, round_up(R, 128) * 4]`` in the same per-destination
  swizzled 128x4 layout.  Byte for byte the send-buffer ABI of
  :class:`flashinfer.MiniMaxH3Mxfp8PreAttention`.

Both recipes reproduce ``flashinfer.fp4_quantize`` (static global scale) and
``flashinfer.mxfp8_quantize`` bit for bit, including the zero scale-tile
padding rows, which the generated program writes itself (no host-side clear).
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import tvm_ffi

from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_qkv_quantize_pack import (
    MXFP8_P1,
    MXFP8_P2,
    MXFP8_P4,
    MXFP8_P8,
    NVFP4_P1,
    NVFP4_P2,
    NVFP4_P4,
    NVFP4_P8,
)

_HEADS = 56
_KINDS = 3
_HEAD_DIM = 128
_FP4_BLOCK = 16
_MXFP8_BLOCK = 32
_NVFP4_PACKED_COLS = _HEAD_DIM // 2  # 64 E2M1 nibble-pair bytes per head row
_NVFP4_SCALE_COLS = _HEAD_DIM // _FP4_BLOCK  # 8 E4M3 scales per head row
_MXFP8_PACKED_COLS = _HEAD_DIM  # 128 E4M3 bytes per head row
_MXFP8_SCALE_COLS = _HEAD_DIM // _MXFP8_BLOCK  # 4 UE8M0 scales per head row
_ELEMS_PER_LANE = 16  # 32 BF16 bytes per lane: the row alignment the loads need
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)
_SUPPORTED_FORMATS = ("nvfp4", "mxfp8")

# The generated program of every supported (format, P): the format decides the
# quantization recipe and the argument list (NVFP4 takes ``out_global_scale``),
# P is the destination partition count compiled into the program.
_PROGRAMS: dict[tuple[str, int], cake_program.CakeProgram] = {
    ("nvfp4", 1): NVFP4_P1,
    ("nvfp4", 2): NVFP4_P2,
    ("nvfp4", 4): NVFP4_P4,
    ("nvfp4", 8): NVFP4_P8,
    ("mxfp8", 1): MXFP8_P1,
    ("mxfp8", 2): MXFP8_P2,
    ("mxfp8", 4): MXFP8_P4,
    ("mxfp8", 8): MXFP8_P8,
}
# Launch geometry shared by every program: 8 warps per CTA (block 256), each
# warp packing 16 tokens of one (local head, kind) of one destination.
_TOKENS_PER_WARP = 16
_WARPS_PER_CTA = 8


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def minimax_h3_qkv_pack_rows_per_destination(M: int, P: int) -> int:
    """Rows of 128 elements per destination: every (token, local head, kind)."""
    return int(M) * (_HEADS // int(P)) * _KINDS


def minimax_h3_qkv_pack_scale_stride(M: int, P: int, format: str) -> int:
    """Bytes of one destination's swizzled 128x4 scale tile (rows padded to 128)."""
    cols = _NVFP4_SCALE_COLS if format == "nvfp4" else _MXFP8_SCALE_COLS
    return _round_up(minimax_h3_qkv_pack_rows_per_destination(M, P), 128) * cols


def minimax_h3_qkv_pack_output_shapes(
    M: int, P: int, format: str
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """``{"out_q": (shape, dtype), "out_sf": (shape, dtype)}`` of the packed outputs."""
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"format must be one of {_SUPPORTED_FORMATS}, got {format!r}")
    heads_per_destination = _HEADS // int(P)
    scale_stride = minimax_h3_qkv_pack_scale_stride(M, P, format)
    if format == "nvfp4":
        out_q = (
            (int(P), int(M), heads_per_destination, _KINDS, _NVFP4_PACKED_COLS),
            torch.uint8,
        )
    else:
        out_q = (
            (int(P), int(M), heads_per_destination, _KINDS, _MXFP8_PACKED_COLS),
            torch.float8_e4m3fn,
        )
    return {"out_q": out_q, "out_sf": ((int(P), scale_stride), torch.uint8)}


def _source_strides(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, M: int, device: torch.device
) -> tuple[int, int]:
    """Validate the three BF16 sources and return ``(token_stride, head_stride)`` in elements.

    Accepts contiguous ``[M, 56, 128]`` tensors, or three equally strided
    ``[M, 56, 128]`` views whose last dimension is contiguous and whose rows are
    32-byte aligned (for example the kind slices of a ``[M, 56, 3, 128]``
    projection output).
    """
    strides: Optional[tuple[int, int]] = None
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if not tensor.is_cuda or tensor.device != device:
            raise ValueError(f"{name} must be a CUDA tensor on {device}")
        if tensor.dtype != torch.bfloat16:
            raise TypeError(
                f"{name} must have dtype torch.bfloat16, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != (M, _HEADS, _HEAD_DIM):
            raise ValueError(
                f"{name} must have shape {(M, _HEADS, _HEAD_DIM)}, got {tuple(tensor.shape)}"
            )
        if tensor.stride(2) != 1:
            raise ValueError(f"{name} head_dim must be contiguous")
        current = (int(tensor.stride(0)), int(tensor.stride(1)))
        if strides is None:
            strides = current
        elif current != strides:
            raise ValueError("q, k and v must share the same token and head strides")
        if (
            current[1] % _ELEMS_PER_LANE
            or current[0] % _ELEMS_PER_LANE
            or int(tensor.data_ptr()) % 32
        ):
            raise ValueError(f"{name} rows must be 32-byte aligned")
        if current[1] < _HEAD_DIM or current[0] < _HEADS * current[1]:
            raise ValueError(f"{name} strides overlap")
    assert strides is not None
    if strides[0] > 2**31 - 1:
        raise ValueError("token stride must fit int32")
    return strides


def _require_output(
    value: torch.Tensor,
    *,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
    if value.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {value.dtype}")
    if not value.is_cuda or value.device != device:
        raise ValueError(f"{name} must be a CUDA tensor on {device}")
    if not value.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return value


def minimax_h3_qkv_pack_target(device: torch.device) -> str:
    """Exact target (``"sm_100a"`` or ``"sm_103a"``) every program ships for ``device``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    target = cake_program.CAPABILITY_TARGET.get((capability[0], capability[1]))
    if target is None or any(
        target not in program.targets for program in _PROGRAMS.values()
    ):
        raise RuntimeError(
            "MiniMax-H3 QKV quantize-and-pack requires exact compute capability "
            f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
        )
    return target


def minimax_h3_qkv_pack_launch_grid(M: int, P: int) -> tuple[int, int, int]:
    """Launch grid of the pack program for the runtime token count ``M``.

    grid.x: one warp per (token group, local head, kind) of one destination
    plus one CTA that zeroes the scale-tile padding rows; grid.y: the
    destination.  The token count is not part of the program identity, so the
    grid is computed here for every preparation.
    """
    token_groups = -(-M // _TOKENS_PER_WARP)
    warps = token_groups * (_HEADS // P) * _KINDS
    return (-(-warps // _WARPS_PER_CTA) + 1, P, 1)


def _flat_source_view(
    tensor: torch.Tensor, M: int, token_stride: int, head_stride: int
) -> torch.Tensor:
    """Stride-1 view over the address span of one ``[M, 56, 128]`` source.

    The generated host shim binds tensor arguments only when they are
    contiguous; the program addresses the source through ``token_stride`` /
    ``head_stride`` from its base pointer, so the kind slice of a fused
    projection output is exposed as the flat span it covers instead of being
    copied.
    """
    span = (M - 1) * token_stride + (_HEADS - 1) * head_stride + _HEAD_DIM
    if tensor.is_contiguous() and tensor.numel() == span:
        return tensor.reshape(-1)
    return tensor.as_strided((span,), (1,), tensor.storage_offset())


def _call_args(
    program: cake_program.CakeProgram,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
) -> tuple[Any, ...]:
    """Positional ``run`` arguments of ``program``: its host parameters in binding order, then the grid."""
    bound = {**values, **dict(zip(cake_program.GRID_ARGS, grid, strict=True))}
    missing = [name for name in program.args if name not in bound]
    if missing:
        raise RuntimeError(
            f"program {program.name} requires arguments {missing} the host did not bind"
        )
    return tuple(bound[name] for name in program.args)


class PreparedMiniMaxH3QkvQuantizePack:
    """Exact-shape prepared operation with caller-owned storage.

    :meth:`__call__` launches exactly one generated kernel on the current
    PyTorch stream, allocates nothing, and can be captured by a CUDA Graph.
    Unlike the fused pre-attention siblings it does not clear ``out_sf``: the
    generated program zeroes the scale-tile padding rows itself.
    """

    def __init__(
        self,
        *,
        M: int,
        P: int,
        format: str,
        module,
        args: tuple,
        out_q: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        self.M = M
        self.P = P
        self.format = format
        self._module = module
        self._args = args
        self.out_q = out_q
        self.out_sf = out_sf

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        with tvm_ffi.use_torch_stream():
            self._module.run(*self._args)
        return self.out_q, self.out_sf


def prepare_minimax_h3_qkv_quantize_pack(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
) -> PreparedMiniMaxH3QkvQuantizePack:
    """Bind caller-owned tensors to the ``(format, P)`` program of the device's exact target.

    ``out_global_scale`` (float32 ``[1]``, FlashInfer convention
    ``(448 * 6) / amax``) is required for ``"nvfp4"`` and must be ``None`` for
    ``"mxfp8"``.  Tensor contents may change between launches; rebinding a
    tensor requires a new preparation.
    """
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    if format not in _SUPPORTED_FORMATS:
        raise ValueError(f"format must be one of {_SUPPORTED_FORMATS}, got {format!r}")
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise TypeError("q must be a three-dimensional torch.Tensor")
    M = int(q.shape[0])
    if M <= 0:
        raise ValueError("M must be positive")
    device = q.device
    token_stride, head_stride = _source_strides(q, k, v, M, device)
    shapes = minimax_h3_qkv_pack_output_shapes(M, P, format)
    _require_output(
        out_q,
        name="out_q",
        shape=shapes["out_q"][0],
        dtype=shapes["out_q"][1],
        device=device,
    )
    _require_output(
        out_sf,
        name="out_sf",
        shape=shapes["out_sf"][0],
        dtype=shapes["out_sf"][1],
        device=device,
    )
    if format == "nvfp4":
        if out_global_scale is None:
            raise ValueError("out_global_scale is required for the nvfp4 format")
        _require_output(
            out_global_scale,
            name="out_global_scale",
            shape=(1,),
            dtype=torch.float32,
            device=device,
        )
    elif out_global_scale is not None:
        raise ValueError("out_global_scale must be None for the mxfp8 format")

    program = _PROGRAMS[(format, P)]
    module = program.load(minimax_h3_qkv_pack_target(device))

    values: dict[str, Any] = {
        "q": _flat_source_view(q, M, token_stride, head_stride),
        "k": _flat_source_view(k, M, token_stride, head_stride),
        "v": _flat_source_view(v, M, token_stride, head_stride),
        "out_q": out_q,
        "out_sf": out_sf,
        # Runtime shape parameters of the generated program (M is not part of
        # the program identity; the derived strides follow the tensor layout
        # validated above).
        "M": M,
        "token_stride": token_stride,
        "head_stride": head_stride,
        "ROWS_PER_DESTINATION": minimax_h3_qkv_pack_rows_per_destination(M, P),
        "SCALE_STRIDE": minimax_h3_qkv_pack_scale_stride(M, P, format),
    }
    if out_global_scale is not None:
        values["out_global_scale"] = out_global_scale
    args = _call_args(program, values, grid=minimax_h3_qkv_pack_launch_grid(M, P))
    return PreparedMiniMaxH3QkvQuantizePack(
        M=M,
        P=P,
        format=format,
        module=module,
        args=args,
        out_q=out_q,
        out_sf=out_sf,
    )


def minimax_h3_qkv_quantize_pack(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    P: int,
    format: str,
    out_global_scale: Optional[torch.Tensor] = None,
    out_q: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One-shot convenience wrapper around :func:`prepare_minimax_h3_qkv_quantize_pack`.

    Allocates ``out_q`` / ``out_sf`` when they are not supplied; that
    allocation (and the per-call preparation) makes this wrapper
    unsuitable for CUDA Graph capture.  Prepare once and reuse the returned
    operation for graph-safe, allocation-free launches.
    """
    if not isinstance(q, torch.Tensor) or q.ndim != 3:
        raise TypeError("q must be a three-dimensional torch.Tensor")
    shapes = minimax_h3_qkv_pack_output_shapes(int(q.shape[0]), P, format)
    if out_q is None:
        out_q = torch.empty(
            shapes["out_q"][0], dtype=shapes["out_q"][1], device=q.device
        )
    if out_sf is None:
        out_sf = torch.empty(
            shapes["out_sf"][0], dtype=shapes["out_sf"][1], device=q.device
        )
    prepared = prepare_minimax_h3_qkv_quantize_pack(
        q=q,
        k=k,
        v=v,
        out_q=out_q,
        out_sf=out_sf,
        P=P,
        format=format,
        out_global_scale=out_global_scale,
    )
    return prepared()


__all__ = [
    "PreparedMiniMaxH3QkvQuantizePack",
    "minimax_h3_qkv_pack_launch_grid",
    "minimax_h3_qkv_pack_output_shapes",
    "minimax_h3_qkv_pack_rows_per_destination",
    "minimax_h3_qkv_pack_scale_stride",
    "minimax_h3_qkv_pack_target",
    "minimax_h3_qkv_quantize_pack",
    "prepare_minimax_h3_qkv_quantize_pack",
]
