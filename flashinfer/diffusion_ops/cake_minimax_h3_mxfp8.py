# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 MXFP8 pre-attention operation.

Two generated programs, each shipped for the exact targets ``sm_100a`` and
``sm_103a``:

* ``NORM``: RMSNorm + indexed AdaLN + MXFP8 quantize of the hidden states, one
  CTA per token row, writing ``activation_q`` (E4M3) and ``activation_sf``
  (UE8M0 block-32 scales in the swizzled 128x4 layout).
* ``FUSED``: the persistent 2-CTA MXFP8 QKV GEMM whose epilogue applies the
  Q/K RMSNorm, the split-half NeoX RoPE over the first 96 dims and the
  destination-major MXFP8 pack.  The FP32 accumulation runs in the K order of
  a block-scaled MXFP8 GEMM and is rounded to BF16 before the epilogue, so the
  epilogue reads the BF16 QKV projection such a GEMM writes; the projection
  never leaves the SM.

The token count ``M`` and the destination geometry are launch parameters: any
``M >= 1`` and every destination partition count ``P`` in ``(1, 2, 4, 8)`` run
the same two programs.  Tensor maps are passed by value: the fused stage's
binding encodes its five maps on every launch, and no descriptor workspace
exists.

The first stage reads the hidden states, the norm weight and the AdaLN tables
with 16-byte vector loads, then writes ``activation_q`` with 8-byte stores and
``activation_sf`` one byte at a time; the fused stage reads both of them, the
weight and its scales through TMA, the Q/K norm weights and ``rope_cos_sin``
with 16-byte vector loads, writes ``out_q`` through TMA and the debug outputs
with 16-byte stores, and stores the four UE8M0 codes of an output row as one
32-bit word.  Preparation therefore requires 16-byte aligned data pointers for
every accessed tensor but ``adaln_index`` and ``out_sf``, which take 4-byte
words, and raises ``ValueError`` otherwise.  Fresh PyTorch allocations meet
this; an offset view into a larger buffer may not.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import tvm_ffi

from flashinfer.gemm import gemm_base
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_mxfp8_pre_attention import FUSED, NORM

_HIDDEN = 5376
_HEADS = 56
_KINDS = 3
_HEAD_DIM = 128
_QKV_WIDTH = _HEADS * _KINDS * _HEAD_DIM
_ADALN_ROWS = 9
_ROPE_DIM = 96
_EPS = 1.0e-5
_MX_BLOCK = 32
_ACTIVATION_SCALE_COLS = _HIDDEN // _MX_BLOCK  # 168 UE8M0 scales per row
_OUTPUT_SCALE_COLS = _HEAD_DIM // _MX_BLOCK  # 4 UE8M0 scales per head row
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)
# Fused GEMM geometry: the swizzled 128x4 scale layout stores one 128-row x
# 4-K-block tile in 512 bytes, which the GEMM's TMA reads as [tiles, 4, 128];
# one CTA pair owns a pair of 128-row tiles x one 256-column tile pair.
_SCALE_ROWS_PER_TILE = 4
_SCALE_TILE_COLS = 128
_GEMM_COLUMN_TILES = _QKV_WIDTH // 256  # 84 column tile pairs
_GEMM_CTA_GROUP = 2
_GEMM_BLOCK_M = 128
# Data-pointer alignment in bytes of the tensors the stages access: the fused
# stage's TMA operands (activation_q, qkv_weight_q, both scale tiles, out_q)
# and the 16-byte vector loads and stores (x, the norm weights, the AdaLN
# tables, rope_cos_sin, the debug outputs) need 16; the first stage's 8-byte
# activation_q stores and byte activation_sf stores need less than the TMA
# reads of the same tensors.  adaln_index is read as one 32-bit word per row,
# and out_sf takes one 32-bit store per output row.  qkv_bf16 and
# gemm_workspace are never accessed.
_ALIGNMENT = {
    "x": 16,
    "x_norm_weight": 16,
    "adaln_scale": 16,
    "adaln_shift": 16,
    "adaln_index": 4,
    "qkv_weight_q": 16,
    "qkv_weight_sf": 16,
    "q_norm_weight": 16,
    "k_norm_weight": 16,
    "rope_cos_sin": 16,
    "out_q": 16,
    "out_sf": 4,
    "activation_q": 16,
    "activation_sf": 16,
    "debug_q_bf16": 16,
    "debug_k_bf16": 16,
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
    alignment = _ALIGNMENT.get(name, 1)
    if value.data_ptr() % alignment:
        raise ValueError(
            f"{name} must have a {alignment}-byte aligned data pointer, "
            f"got an offset of {value.data_ptr() % alignment} bytes"
        )
    return value


def minimax_h3_mxfp8_target(device: torch.device) -> str:
    """Exact target (``"sm_100a"`` or ``"sm_103a"``) the programs ship for ``device``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    target = cake_program.CAPABILITY_TARGET.get((capability[0], capability[1]))
    if target is None or target not in NORM.targets or target not in FUSED.targets:
        raise RuntimeError(
            "MiniMax-H3 MXFP8 pre-attention requires exact compute capability "
            f"10.0 or 10.3, got {capability[0]}.{capability[1]}"
        )
    return target


def _gemm_row_tiles(M: int) -> int:
    """128-row output tiles of the fused GEMM, padded to whole CTA pairs."""
    tiles = -(-M // _GEMM_BLOCK_M)
    return tiles + tiles % _GEMM_CTA_GROUP


def minimax_h3_mxfp8_launch_grid(stage: str, M: int) -> tuple[int, int, int]:
    """Launch grid of one generated stage for the runtime token count ``M``.

    ``"norm"``: one CTA per token row.  ``"gemm"``: one CTA pair (cluster) per
    pair of 128-row tiles x 256-column tile pair; the persistent clusters
    claim the remaining tiles through the cluster launch-control scheduler, so
    the grid is the whole tile space.  The token count is not part of the
    program identity, so the grid is computed here for every preparation.
    """
    if stage == "norm":
        return (int(M), 1, 1)
    if stage == "gemm":
        return (_gemm_row_tiles(int(M)) * _GEMM_COLUMN_TILES, 1, 1)
    raise ValueError(f"unknown stage {stage!r}; the stages are ('norm', 'gemm')")


def minimax_h3_mxfp8_rows_per_destination(M: int, P: int) -> int:
    """Rows of 128 elements per destination: every (token, local head, kind)."""
    return int(M) * (_HEADS // int(P)) * _KINDS


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


class PreparedMiniMaxH3Mxfp8PreAttention:
    """Exact-shape prepared operation with caller-owned storage.

    The fused stage takes its TMA tensor maps by value; its binding encodes
    them from the bound tensors on every call, so there is no descriptor
    workspace.  Preparation checks what those encoders check (shape,
    contiguity and 16-byte aligned addresses), so tensors that pass
    preparation also pass the encoders.  The ``norm_descriptor_workspace`` /
    ``post_descriptor_workspace`` keyword arguments of
    :func:`prepare_minimax_h3_mxfp8_pre_attention` are kept for API stability
    and must be ``None``.
    """

    def __init__(
        self,
        *,
        M: int,
        P: int,
        eps: float,
        norm_run,
        gemm_run,
        norm_args: tuple,
        gemm_args: tuple,
        activation_sf: torch.Tensor,
        out_q: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        self.M = M
        self.P = P
        self.eps = eps
        self._norm_run = norm_run
        self._gemm_run = gemm_run
        self._norm_args = norm_args
        self._gemm_args = gemm_args
        self._activation_sf = activation_sf
        self.out_q = out_q
        self.out_sf = out_sf

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Both swizzled 128x4 scale tiles pad their rows to 128: the norm
        # stage writes every activation scale of the M token rows and the
        # fused GEMM every live output scale, and these asynchronous clears
        # make the padding bytes deterministic.
        self.out_sf.zero_()
        self._activation_sf.zero_()
        with tvm_ffi.use_torch_stream():
            self._norm_run(*self._norm_args)
            self._gemm_run(*self._gemm_args)
        return self.out_q, self.out_sf


def prepare_minimax_h3_mxfp8_pre_attention(
    *,
    x: torch.Tensor,
    x_norm_weight: torch.Tensor,
    adaln_scale: torch.Tensor,
    adaln_shift: torch.Tensor,
    adaln_index: torch.Tensor,
    qkv_weight_q: torch.Tensor,
    qkv_weight_sf: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rope_cos_sin: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    activation_q: torch.Tensor,
    activation_sf: torch.Tensor,
    qkv_bf16: torch.Tensor,
    gemm_workspace: torch.Tensor,
    P: int,
    norm_descriptor_workspace: Optional[torch.Tensor] = None,
    post_descriptor_workspace: Optional[torch.Tensor] = None,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> PreparedMiniMaxH3Mxfp8PreAttention:
    """Validate the caller-owned tensors once and bind both stage launches.

    Every tensor must be contiguous, on the device of ``x``, with the
    documented shape and dtype.  The data pointers of the tensors the stages
    access must be 16-byte aligned (``adaln_index`` and ``out_sf``: 4-byte
    aligned); a violation raises ``ValueError`` here, before any launch.

    ``qkv_bf16`` and ``gemm_workspace`` are validated and kept for API
    stability; the fused GEMM never writes the BF16 projection to memory and
    needs no GEMM workspace.  ``norm_descriptor_workspace`` and
    ``post_descriptor_workspace`` must be ``None``: the fused stage's binding
    encodes its tensor maps on each launch.  ``debug_q_bf16`` and
    ``debug_k_bf16`` are optional BF16 ``[M, 56, 128]`` outputs of the rotated
    Q and K rows, written only when both are supplied.
    """
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    if float(eps) != _EPS:
        raise ValueError(f"eps must be exactly {_EPS}")
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise TypeError("x must be a two-dimensional torch.Tensor")
    M = int(x.shape[0])
    if M <= 0:
        raise ValueError("M must be positive")
    if norm_descriptor_workspace is not None or post_descriptor_workspace is not None:
        raise ValueError(
            "norm_descriptor_workspace and post_descriptor_workspace must be None: "
            "the generated stages take their tensor maps by value"
        )
    device = x.device
    heads_per_destination = _HEADS // P
    rows_per_destination = minimax_h3_mxfp8_rows_per_destination(M, P)
    activation_sf_len = _round_up(M, 128) * _ACTIVATION_SCALE_COLS
    out_sf_stride = _round_up(rows_per_destination, 128) * _OUTPUT_SCALE_COLS
    tensors = {
        "x": (x, (M, _HIDDEN), torch.bfloat16),
        "x_norm_weight": (x_norm_weight, (_HIDDEN,), torch.bfloat16),
        "adaln_scale": (adaln_scale, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_shift": (adaln_shift, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_index": (adaln_index, (M,), torch.int32),
        "qkv_weight_q": (qkv_weight_q, (_QKV_WIDTH, _HIDDEN), torch.float8_e4m3fn),
        "qkv_weight_sf": (
            qkv_weight_sf,
            (_QKV_WIDTH * _ACTIVATION_SCALE_COLS,),
            torch.uint8,
        ),
        "q_norm_weight": (q_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "k_norm_weight": (k_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "rope_cos_sin": (rope_cos_sin, (M, _ROPE_DIM), torch.bfloat16),
        "out_q": (
            out_q,
            (P, M, heads_per_destination, _KINDS, _HEAD_DIM),
            torch.float8_e4m3fn,
        ),
        "out_sf": (out_sf, (P, out_sf_stride), torch.uint8),
        "activation_q": (activation_q, (M, _HIDDEN), torch.float8_e4m3fn),
        "activation_sf": (activation_sf, (activation_sf_len,), torch.uint8),
        "qkv_bf16": (qkv_bf16, (M, _QKV_WIDTH), torch.bfloat16),
    }
    for name, (value, shape, dtype) in tensors.items():
        _require_tensor(value, name=name, shape=shape, dtype=dtype, device=device)
    _require_tensor(
        gemm_workspace,
        name="gemm_workspace",
        shape=(int(gemm_workspace.numel()),),
        dtype=torch.uint8,
        device=device,
    )
    if gemm_workspace.numel() < int(gemm_base.DEFAULT_WORKSPACE_SIZE):
        raise ValueError(
            "gemm_workspace is smaller than FlashInfer DEFAULT_WORKSPACE_SIZE"
        )
    if (debug_q_bf16 is None) != (debug_k_bf16 is None):
        raise ValueError("debug_q_bf16 and debug_k_bf16 must be supplied together")
    # The fused program keeps the BF16 projection pointer of its GEMM-only
    # sibling and never writes it; qkv_bf16 is bound there and, without the
    # debug outputs, to the debug pointers, which the program then never
    # dereferences.
    qkv_words = qkv_bf16.view(torch.uint32).reshape(-1)
    write_debug = int(debug_q_bf16 is not None)
    if debug_q_bf16 is None or debug_k_bf16 is None:
        debug_q_words = qkv_words
        debug_k_words = qkv_words
    else:
        for name, value in (
            ("debug_q_bf16", debug_q_bf16),
            ("debug_k_bf16", debug_k_bf16),
        ):
            _require_tensor(
                value,
                name=name,
                shape=(M, _HEADS, _HEAD_DIM),
                dtype=torch.bfloat16,
                device=device,
            )
        debug_q_words = debug_q_bf16.view(torch.uint32).reshape(-1)
        debug_k_words = debug_k_bf16.view(torch.uint32).reshape(-1)
    target = minimax_h3_mxfp8_target(device)
    norm_run = NORM.entry(target)
    gemm_run = FUSED.entry(target)
    values = {
        "x": x,
        "x_norm_weight": x_norm_weight,
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "adaln_index": adaln_index,
        "activation_q": activation_q,
        "activation_sf": activation_sf,
        # Fused GEMM operands: byte views of the E4M3 activation and weight and
        # [tiles, 4, 128] views of both swizzled scale tiles, which the GEMM
        # loads through TMA as they are, plus the destination-major byte view
        # of out_q its epilogue stores 128-byte rows through.
        "A": activation_q.view(torch.uint8),
        "B": qkv_weight_q.view(torch.uint8),
        "SFA": activation_sf.view(-1, _SCALE_ROWS_PER_TILE, _SCALE_TILE_COLS),
        "SFB": qkv_weight_sf.view(-1, _SCALE_ROWS_PER_TILE, _SCALE_TILE_COLS),
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_cos_sin": rope_cos_sin,
        "out_sf": out_sf,
        "OUTQ": out_q.view(torch.uint8).reshape(
            P, M, heads_per_destination * _KINDS, _HEAD_DIM
        ),
        "qkv_words": qkv_words,
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
        "SCALE_STRIDE": out_sf_stride,
    }
    norm_args = _call_args(NORM, values, grid=minimax_h3_mxfp8_launch_grid("norm", M))
    gemm_args = _call_args(FUSED, values, grid=minimax_h3_mxfp8_launch_grid("gemm", M))
    return PreparedMiniMaxH3Mxfp8PreAttention(
        M=M,
        P=P,
        eps=eps,
        norm_run=norm_run,
        gemm_run=gemm_run,
        norm_args=norm_args,
        gemm_args=gemm_args,
        activation_sf=activation_sf,
        out_q=out_q,
        out_sf=out_sf,
    )


__all__ = [
    "PreparedMiniMaxH3Mxfp8PreAttention",
    "minimax_h3_mxfp8_launch_grid",
    "minimax_h3_mxfp8_rows_per_destination",
    "minimax_h3_mxfp8_target",
    "prepare_minimax_h3_mxfp8_pre_attention",
]
