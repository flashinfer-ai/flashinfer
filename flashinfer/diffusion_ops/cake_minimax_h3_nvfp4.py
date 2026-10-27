# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 NVFP4 (W4A4) pre-attention operation.

Two generated programs per exact target (``sm_100a``, ``sm_103a``): the RMSNorm
+ AdaLN + NVFP4 quantize stage and the fused NVFP4 QKV GEMM whose epilogue
applies the Q/K RMSNorm, the split-half NeoX RoPE and the destination-major
output, one GEMM schedule per target and one program per output format:

* ``out_format="nvfp4"``: E2M1 nibble pairs ``uint8 [P, M, 56 / P, 3, 64]``
  plus swizzled-128x4 E4M3 block-16 scales
  ``uint8 [P, round_up(M * (56 / P) * 3, 128) * 8]`` under one static global
  encode scale (``out_global_scale``).
* ``out_format="e4m3"``: ``float8_e4m3fn [P, M, 56 / P, 3, 128]`` with one
  per-tensor scale per kind, ``out_q[p, m, h, kind, d] = e4m3(value *
  f32(1 / qkv_scales[kind]))`` (round to nearest even, saturating at +-448);
  no scale tile is written.

The token count ``M`` and the destination geometry are launch parameters, so
every destination partition count ``P`` of one format routes to the same GEMM
program of its target.  The norm stage is one program for both targets, except
that on ``sm_103a`` four destinations (``P == 4``), and eight destinations
(``P == 8``) of ``out_format="nvfp4"`` from ``M = 6096`` tokens on, take a
second norm program that stages each row's activation in shared memory; both
write the same bytes.  Tensor maps are passed by value; no descriptor workspace
exists.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import tvm_ffi

from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention import (
    FUSED_E4M3_EPI8,
    FUSED_E4M3_K96,
    FUSED_PACK_EPI8,
    FUSED_PACK_K96,
    NORM,
    NORM_V5,
)

_HIDDEN = 5376
_HEADS = 56
_KINDS = 3
_HEAD_DIM = 128
_QKV_WIDTH = _HEADS * _KINDS * _HEAD_DIM
_ADALN_ROWS = 9
_ROPE_DIM = 96
_EPS = 1.0e-5
_FP4_BLOCK = 16
_ACTIVATION_SCALE_COLS = _HIDDEN // _FP4_BLOCK  # 336 E4M3 scales per row
_ACTIVATION_PACKED_COLS = _HIDDEN // 2  # 2688 E2M1 nibble-pair bytes per row
_OUTPUT_SCALE_COLS = _HEAD_DIM // _FP4_BLOCK  # 8 E4M3 scales per head row
_NVFP4_PACKED_COLS = _HEAD_DIM // 2  # 64 E2M1 nibble-pair bytes per head row
_E4M3_PACKED_COLS = _HEAD_DIM  # 128 E4M3 bytes per head row
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)
_SUPPORTED_FORMATS = ("nvfp4", "e4m3")
# Fused GEMM operand geometry: the swizzled-128x4 scale layout stores one
# 128-row x 4-K-block tile in 512 bytes; the fused kernel consumes the weight
# scales of one 256-column tile pair per CTA pair, half per CTA.
_SCALE_TILE_BYTES = 512
_SCALE_K_TILES = _HIDDEN // (4 * _FP4_BLOCK)  # 84 K-sets per 128-row block
_GEMM_COLUMN_TILES = _QKV_WIDTH // 256  # 84 column tile pairs
_GEMM_CTA_GROUP = 2
_GEMM_BLOCK_M = 128
_ACTIVATION_SCALE_ROWS_PER_TILE = 4
_WEIGHT_SCALE_ROWS_PER_TILE = 8
# Norm stage geometry: one CTA per two rows of the 128-row-aligned activation
# scale tile (the padding rows are zeroed by the stage).
_NORM_ROWS_PER_CTA = 2
_NORM_ROW_ALIGNMENT = 128

# The fused GEMM program of every (output format, exact target); the norm stage
# is one program for both targets.
_GEMM_PROGRAMS: dict[tuple[str, str], cake_program.CakeProgram] = {
    ("nvfp4", "sm_100a"): FUSED_PACK_EPI8,
    ("nvfp4", "sm_103a"): FUSED_PACK_K96,
    ("e4m3", "sm_100a"): FUSED_E4M3_EPI8,
    ("e4m3", "sm_103a"): FUSED_E4M3_K96,
}


# The norm program of an (exact target, P) pair that does not take ``NORM``.
_NORM_PROGRAMS: dict[tuple[str, int], cake_program.CakeProgram] = {
    ("sm_103a", 4): NORM_V5,
}

# The norm program of an (output format, exact target, P) triple that does not
# take ``NORM`` from a token count ``M`` on: ``(minimum M, program)``.
_NORM_PROGRAMS_FROM_M: dict[
    tuple[str, str, int], tuple[int, cake_program.CakeProgram]
] = {
    ("nvfp4", "sm_103a", 8): (6096, NORM_V5),
}


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _require_tensor(
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
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if not value.is_cuda or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
    return value


def _stage_workspace(value: Optional[torch.Tensor], *, name: str) -> None:
    """The generated stages pass their tensor maps by value: no descriptor workspace exists.

    The keyword arguments are kept for API stability and must be ``None``.
    """
    if value is not None:
        raise ValueError(
            f"{name} must be None: the generated stages take their tensor maps by value"
        )


def minimax_h3_nvfp4_target(device: torch.device) -> str:
    """Exact target (``"sm_100a"`` or ``"sm_103a"``) the programs ship for ``device``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    target = cake_program.CAPABILITY_TARGET.get((capability[0], capability[1]))
    if (
        target is None
        or target not in NORM.targets
        or any((fmt, target) not in _GEMM_PROGRAMS for fmt in _SUPPORTED_FORMATS)
    ):
        raise RuntimeError(
            "MiniMax-H3 NVFP4 pre-attention requires exact compute capability "
            f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
        )
    return target


def minimax_h3_nvfp4_programs(
    device: torch.device,
    out_format: str = "nvfp4",
    P: Optional[int] = None,
    M: Optional[int] = None,
) -> tuple[cake_program.CakeProgram, cake_program.CakeProgram]:
    """``(norm, gemm)``: the two generated programs of ``out_format``, ``P``
    destinations and ``M`` tokens on the device's exact target (``P=None``: the
    norm program of every ``P`` but the ones with a program of their own;
    ``M=None``: the norm program of every ``M`` but the ones with a program of
    their own)."""
    if out_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"out_format must be one of {_SUPPORTED_FORMATS}, got {out_format!r}"
        )
    target = minimax_h3_nvfp4_target(device)
    norm = _NORM_PROGRAMS.get((target, P), NORM) if P is not None else NORM
    from_m = _NORM_PROGRAMS_FROM_M.get((out_format, target, P))
    if from_m is not None and M is not None and from_m[0] <= M:
        norm = from_m[1]
    return norm, _GEMM_PROGRAMS[(out_format, target)]


def _gemm_row_tiles(M: int) -> int:
    """128-row output tiles of the fused GEMM, padded to whole CTA pairs."""
    tiles = -(-M // _GEMM_BLOCK_M)
    return tiles + tiles % _GEMM_CTA_GROUP


def minimax_h3_nvfp4_launch_grid(stage: str, M: int) -> tuple[int, int, int]:
    """Launch grid of one generated stage for the runtime token count ``M``.

    ``"norm"``: one CTA per two rows of the 128-row-aligned activation scale
    tile.  ``"gemm"``: one CTA pair (cluster) per pair of 128-row tiles x
    256-column tile pair; the persistent clusters claim the remaining tiles
    through the cluster launch-control scheduler, so the grid is the whole
    tile space.  The token count is not part of the program identity, so the
    grid is computed here for every preparation.
    """
    if stage == "norm":
        rows = _round_up(int(M), _NORM_ROW_ALIGNMENT)
        return (rows // _NORM_ROWS_PER_CTA, 1, 1)
    if stage == "gemm":
        row_tiles = _gemm_row_tiles(int(M))
        return (
            (row_tiles // _GEMM_CTA_GROUP) * _GEMM_COLUMN_TILES * _GEMM_CTA_GROUP,
            1,
            1,
        )
    raise ValueError(f"unknown stage {stage!r}; the stages are ('norm', 'gemm')")


def minimax_h3_nvfp4_rows_per_destination(M: int, P: int) -> int:
    """Rows of 128 elements per destination: every (token, local head, kind)."""
    return int(M) * (_HEADS // int(P)) * _KINDS


def minimax_h3_nvfp4_output_shapes(
    M: int, P: int, out_format: str = "nvfp4"
) -> dict[str, Optional[tuple[tuple[int, ...], torch.dtype]]]:
    """``{"out_q": (shape, dtype), "out_sf": (shape, dtype) | None}`` of the caller-owned outputs.

    ``out_sf`` is ``None`` for ``"e4m3"``: the per-kind scales are the
    caller's ``qkv_scales`` and the program writes no scale tile.
    """
    if out_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"out_format must be one of {_SUPPORTED_FORMATS}, got {out_format!r}"
        )
    heads_per_destination = _HEADS // int(P)
    if out_format == "nvfp4":
        rows = minimax_h3_nvfp4_rows_per_destination(M, P)
        return {
            "out_q": (
                (int(P), int(M), heads_per_destination, _KINDS, _NVFP4_PACKED_COLS),
                torch.uint8,
            ),
            "out_sf": (
                (int(P), _round_up(rows, 128) * _OUTPUT_SCALE_COLS),
                torch.uint8,
            ),
        }
    return {
        "out_q": (
            (int(P), int(M), heads_per_destination, _KINDS, _E4M3_PACKED_COLS),
            torch.float8_e4m3fn,
        ),
        "out_sf": None,
    }


def _call_args(
    program: cake_program.CakeProgram,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
) -> tuple[Any, ...]:
    """Positional ``run`` arguments of ``program``: its host parameters in binding order, then the grid.

    Each torch tensor is converted to a ``tvm_ffi.Tensor`` here, once, so a
    launch passes the converted handles instead of converting every tensor
    again.
    """
    bound = {**values, **dict(zip(cake_program.GRID_ARGS, grid, strict=True))}
    missing = [name for name in program.args if name not in bound]
    if missing:
        raise RuntimeError(
            f"program {program.name} requires arguments {missing} the host did not bind"
        )
    return tuple(
        tvm_ffi.from_dlpack(bound[name])
        if isinstance(bound[name], torch.Tensor)
        else bound[name]
        for name in program.args
    )


def repack_minimax_h3_qkv_weight_scales_for_fused_gemm(
    qkv_weight_sf: torch.Tensor,
) -> torch.Tensor:
    """Reorder swizzled-128x4 QKV weight scales for the fused GEMM's CTA pairs.

    The prepacked weight scales are ``[168 (head, kind) row blocks][84 K-sets]
    [512 B]``. The fused GEMM assigns one 256-column tile pair to a CTA pair,
    each CTA loading the scales of its own 128-column half, so the tiles are
    reordered to ``[84 pairs][84 K-sets][2 halves][512 B]``. Called once at
    preparation; the result is bound to the prepared operation.
    """
    expected = _QKV_WIDTH * _ACTIVATION_SCALE_COLS
    if qkv_weight_sf.numel() != expected:
        raise ValueError(
            f"qkv_weight_sf must hold {expected} swizzled scale bytes, "
            f"got {qkv_weight_sf.numel()}"
        )
    tiles = qkv_weight_sf.reshape(
        _GEMM_COLUMN_TILES, _GEMM_CTA_GROUP, _SCALE_K_TILES, _SCALE_TILE_BYTES
    )
    return tiles.permute(0, 2, 1, 3).contiguous().reshape(-1)


class PreparedMiniMaxH3Nvfp4PreAttention:
    """Exact-shape prepared operation with caller-owned storage.

    The generated stages take their TMA tensor maps by value, which each
    launch encodes from the bound tensors; there is no descriptor workspace.
    The ``norm_descriptor_workspace`` / ``gemm_descriptor_workspace`` keyword
    arguments of :func:`prepare_minimax_h3_nvfp4_pre_attention` are kept for
    API stability and must be ``None``.
    """

    def __init__(
        self,
        *,
        M: int,
        P: int,
        out_format: str,
        eps: float,
        norm_run,
        gemm_run,
        norm_args: tuple,
        gemm_args: tuple,
        derived: tuple[torch.Tensor, ...],
        out_q: torch.Tensor,
        out_sf: Optional[torch.Tensor],
    ) -> None:
        self.M = M
        self.P = P
        self.out_format = out_format
        self.eps = eps
        self._norm_run = norm_run
        self._gemm_run = gemm_run
        self._norm_args = norm_args
        self._gemm_args = gemm_args
        # Keeps the device operands derived at preparation (alpha, pair-ordered
        # weight scales, the e4m3 out scales, unbound placeholders) alive for
        # the bound args.
        self._derived = derived
        self.out_q = out_q
        self.out_sf = out_sf
        self._device = out_q.device
        self._ffi_device = tvm_ffi.device(str(out_q.device))

    def __call__(self) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # The NVFP4 output scale tile pads rows to 128 per destination: the
        # norm stage zeroes each destination's last 128-row tile and the fused
        # GEMM then writes every live scale, so the padding bytes are
        # deterministic without a separate clear. The norm stage zero-pads the
        # activation scale tile as well. The e4m3 output has no scale tile.
        # The converted arguments carry no stream, so both launches use the
        # current torch stream of the tensors' device, which need not be the
        # current device.
        stream = torch.cuda.current_stream(self._device).cuda_stream
        with tvm_ffi.use_raw_stream(self._ffi_device, stream):
            self._norm_run(*self._norm_args)
            self._gemm_run(*self._gemm_args)
        return self.out_q, self.out_sf


def prepare_minimax_h3_nvfp4_pre_attention(
    *,
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    x_global_scale: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    w_global_scale: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out_q: torch.Tensor,
    activation_q: torch.Tensor,
    activation_sf: torch.Tensor,
    P: int,
    out_format: str = "nvfp4",
    out_global_scale: Optional[torch.Tensor] = None,
    out_sf: Optional[torch.Tensor] = None,
    qkv_scales: Optional[torch.Tensor] = None,
    norm_descriptor_workspace: Optional[torch.Tensor] = None,
    gemm_descriptor_workspace: Optional[torch.Tensor] = None,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    debug_adaln_bf16: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> PreparedMiniMaxH3Nvfp4PreAttention:
    """Validate the caller-owned tensors once and bind both stage launches.

    ``out_format="nvfp4"`` takes ``out_global_scale`` (float32 ``[1]``) and
    ``out_sf`` (uint8 ``[P, round_up(M * (56 / P) * 3, 128) * 8]``, 8-byte
    aligned) and refuses
    ``qkv_scales``; ``out_format="e4m3"`` takes ``qkv_scales`` (float32 ``[3]``,
    the per-tensor scales of Q, K and V: ``out_q = e4m3(value / qkv_scales
    [kind])``, every entry positive and finite) and refuses ``out_global_scale``
    and ``out_sf``.  The out scales ``f32(1 / qkv_scales[kind])`` are derived at
    preparation, like the GEMM ``alpha``; changing ``qkv_scales`` afterwards
    requires a new prepared operation.

    ``norm_descriptor_workspace`` and ``gemm_descriptor_workspace`` must be
    ``None``: the generated stages pass their tensor maps by value, so no
    caller-provided descriptor workspace exists.  The keyword arguments are
    kept for API stability; a non-``None`` value raises :class:`ValueError`.
    The ``debug_*`` tensors are optional BF16 intermediates (AdaLN output,
    post-norm Q / K) written only when all three are supplied.
    """
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    if out_format not in _SUPPORTED_FORMATS:
        raise ValueError(
            f"out_format must be one of {_SUPPORTED_FORMATS}, got {out_format!r}"
        )
    if float(eps) != _EPS:
        raise ValueError(f"eps must be exactly {_EPS}")
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise TypeError("x must be a two-dimensional torch.Tensor")
    M = int(x.shape[0])
    if M <= 0:
        raise ValueError("M must be positive")
    device = x.device
    heads_per_destination = _HEADS // P
    rows_per_destination = minimax_h3_nvfp4_rows_per_destination(M, P)
    activation_sf_len = _round_up(M, 128) * _ACTIVATION_SCALE_COLS
    shapes = minimax_h3_nvfp4_output_shapes(M, P, out_format)
    tensors = {
        "x": (x, (M, _HIDDEN), torch.bfloat16),
        "x_norm_weight": (x_norm_weight, (_HIDDEN,), torch.bfloat16),
        "adaln_scale": (adaln_scale, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_shift": (adaln_shift, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_index": (adaln_index, (M,), torch.int32),
        "x_global_scale": (x_global_scale, (1,), torch.float32),
        "qkv_weight_q": (
            qkv_weight_q,
            (_QKV_WIDTH, _ACTIVATION_PACKED_COLS),
            torch.uint8,
        ),
        "qkv_weight_sf": (
            qkv_weight_sf,
            (_QKV_WIDTH * _ACTIVATION_SCALE_COLS,),
            torch.uint8,
        ),
        "w_global_scale": (w_global_scale, (1,), torch.float32),
        "q_norm_weight": (q_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "k_norm_weight": (k_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "rope_cos_sin": (rope_cos_sin, (M, _ROPE_DIM), torch.bfloat16),
        "out_q": (out_q, *shapes["out_q"]),
        "activation_q": (activation_q, (M, _ACTIVATION_PACKED_COLS), torch.uint8),
        "activation_sf": (activation_sf, (activation_sf_len,), torch.uint8),
    }
    if out_format == "nvfp4":
        if out_global_scale is None or out_sf is None:
            raise ValueError(
                "out_global_scale and out_sf are required for out_format 'nvfp4'"
            )
        if qkv_scales is not None:
            raise ValueError("qkv_scales must be None for out_format 'nvfp4'")
        assert shapes["out_sf"] is not None
        tensors["out_global_scale"] = (out_global_scale, (1,), torch.float32)
        tensors["out_sf"] = (out_sf, *shapes["out_sf"])
    else:
        if qkv_scales is None:
            raise ValueError("qkv_scales is required for out_format 'e4m3'")
        if out_global_scale is not None or out_sf is not None:
            raise ValueError(
                "out_global_scale and out_sf must be None for out_format 'e4m3': "
                "the per-kind scales are qkv_scales and no scale tile is written"
            )
        tensors["qkv_scales"] = (qkv_scales, (_KINDS,), torch.float32)
    for name, (value, shape, dtype) in tensors.items():
        _require_tensor(value, name=name, shape=shape, dtype=dtype, device=device)
    debug_present = (
        debug_q_bf16 is not None,
        debug_k_bf16 is not None,
        debug_adaln_bf16 is not None,
    )
    if any(debug_present) and not all(debug_present):
        raise ValueError(
            "debug_q_bf16, debug_k_bf16, and debug_adaln_bf16 must be supplied together"
        )
    write_debug = int(all(debug_present))
    derived: list[torch.Tensor] = []
    if write_debug:
        assert (
            debug_q_bf16 is not None
            and debug_k_bf16 is not None
            and debug_adaln_bf16 is not None
        )
        _require_tensor(
            debug_q_bf16,
            name="debug_q_bf16",
            shape=(M, _HEADS, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        _require_tensor(
            debug_k_bf16,
            name="debug_k_bf16",
            shape=(M, _HEADS, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )
        _require_tensor(
            debug_adaln_bf16,
            name="debug_adaln_bf16",
            shape=(M, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
        debug_q_words = debug_q_bf16.view(torch.uint32).reshape(-1)
        debug_k_words = debug_k_bf16.view(torch.uint32).reshape(-1)
    else:
        # The generated programs bind the debug pointers unconditionally and
        # never dereference them when ``write_debug`` is zero; one-element
        # device placeholders keep the launch free of caller scratch.
        debug_adaln_bf16 = torch.zeros((1,), dtype=torch.bfloat16, device=device)
        debug_q_words = torch.zeros((1,), dtype=torch.uint32, device=device)
        debug_k_words = debug_q_words
        derived += [debug_adaln_bf16, debug_q_words]
    _stage_workspace(norm_descriptor_workspace, name="norm_descriptor_workspace")
    _stage_workspace(gemm_descriptor_workspace, name="gemm_descriptor_workspace")
    target = minimax_h3_nvfp4_target(device)
    norm_program, gemm_program = minimax_h3_nvfp4_programs(device, out_format, P, M)
    norm_run = norm_program.entry(target)
    gemm_run = gemm_program.entry(target)
    # Operands derived once at preparation: the GEMM output scale
    # alpha = 1 / (x_global_scale * w_global_scale), the CTA-pair ordering of
    # the prepacked weight scales and, for e4m3, the per-kind out scales
    # 1 / qkv_scales. Changing any of their sources afterwards requires a new
    # prepared operation; the E2M1 weight bytes themselves are read in place
    # on every launch.
    alpha = (
        (1.0 / (x_global_scale.float() * w_global_scale.float()))
        .reshape(1)
        .contiguous()
    )
    weight_scales = repack_minimax_h3_qkv_weight_scales_for_fused_gemm(qkv_weight_sf)
    derived += [alpha, weight_scales]
    if out_format == "nvfp4":
        assert out_global_scale is not None and out_sf is not None
        out_scale_arg = out_global_scale
        # The norm stage zeroes each destination's padding rows in 8-byte words.
        if out_sf.data_ptr() % 8 != 0:
            raise ValueError("out_sf must be 8-byte aligned")
        out_sf_arg = out_sf
        out_q_bytes = out_q
        scale_stride = int(out_sf.shape[1])
        row_bytes = _NVFP4_PACKED_COLS
    else:
        assert qkv_scales is not None
        # f32(1 / qkv_scales[kind]) in the out_global_scale slot: the kernel
        # multiplies in f32 and rounds to nearest even, saturating at +-448.
        out_scale_arg = (1.0 / qkv_scales.double()).float().reshape(_KINDS).contiguous()
        # out_sf is bound and never read or written (scale stride 0).
        out_sf_arg = torch.zeros((1,), dtype=torch.uint8, device=device)
        derived += [out_scale_arg, out_sf_arg]
        out_q_bytes = out_q.view(torch.uint8)
        scale_stride = 0
        row_bytes = _E4M3_PACKED_COLS
    values = {
        "x": x,
        "x_norm_weight": x_norm_weight,
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "adaln_index": adaln_index,
        "x_global_scale": x_global_scale,
        "debug_adaln_bf16": debug_adaln_bf16,
        "activation_q": activation_q,
        "activation_sf": activation_sf,
        # Fused GEMM operands: TMA views of the quantized activation, the
        # prepacked weight and both swizzled scale tiles, plus the
        # destination-major output view the epilogue stores through TMA.
        "A": activation_q,
        "B": qkv_weight_q,
        "SFA": activation_sf.view(-1, _ACTIVATION_SCALE_ROWS_PER_TILE, 128),
        "SFB": weight_scales.view(-1, _WEIGHT_SCALE_ROWS_PER_TILE, 128),
        "alpha": alpha,
        "OUTQ": out_q_bytes.reshape(P, M, heads_per_destination * _KINDS, row_bytes),
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_cos_sin": rope_cos_sin,
        "out_global_scale": out_scale_arg,
        "out_q": out_q_bytes,
        "out_sf": out_sf_arg,
        # The fused program keeps the BF16 QKV pointer of its GEMM-only
        # sibling; the pack program never writes it.
        "qkv_words": debug_q_words,
        "debug_q_words": debug_q_words,
        "debug_k_words": debug_k_words,
        "write_debug": write_debug,
        "eps": eps,
        # Runtime shape parameters of the generated stages (M is not part of
        # the program identity; the derived strides follow the tensor layout
        # validated above).
        "M": M,
        "m_tiles": _gemm_row_tiles(M),
        "HEADS_PER_DESTINATION": heads_per_destination,
        "ROWS_PER_DESTINATION": rows_per_destination,
        "SCALE_STRIDE": scale_stride,
    }
    norm_args = _call_args(
        norm_program, values, grid=minimax_h3_nvfp4_launch_grid("norm", M)
    )
    gemm_args = _call_args(
        gemm_program, values, grid=minimax_h3_nvfp4_launch_grid("gemm", M)
    )
    return PreparedMiniMaxH3Nvfp4PreAttention(
        M=M,
        P=P,
        out_format=out_format,
        eps=eps,
        norm_run=norm_run,
        gemm_run=gemm_run,
        norm_args=norm_args,
        gemm_args=gemm_args,
        derived=tuple(derived),
        out_q=out_q,
        out_sf=out_sf,
    )


__all__ = [
    "PreparedMiniMaxH3Nvfp4PreAttention",
    "minimax_h3_nvfp4_launch_grid",
    "minimax_h3_nvfp4_output_shapes",
    "minimax_h3_nvfp4_programs",
    "minimax_h3_nvfp4_rows_per_destination",
    "minimax_h3_nvfp4_target",
    "prepare_minimax_h3_nvfp4_pre_attention",
    "repack_minimax_h3_qkv_weight_scales_for_fused_gemm",
]
