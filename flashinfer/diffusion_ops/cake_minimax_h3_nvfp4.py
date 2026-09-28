# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 NVFP4 (W4A4) pre-attention operation."""

from __future__ import annotations

from typing import Optional

import torch
import tvm_ffi

from flashinfer.jit.cake_minimax_h3_nvfp4_pre_attention import (
    load_minimax_h3_nvfp4_route,
    minimax_h3_nvfp4_route_record,
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
_OUTPUT_PACKED_COLS = _HEAD_DIM // 2  # 64 E2M1 nibble-pair bytes per head row
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)
# Fused GEMM operand geometry: the swizzled-128x4 scale layout stores one
# 128-row x 4-K-block tile in 512 bytes; the fused kernel consumes the weight
# scales of one 256-column tile pair per CTA pair, half per CTA.
_SCALE_TILE_BYTES = 512
_SCALE_K_TILES = _HIDDEN // (4 * _FP4_BLOCK)  # 84 K-sets per 128-row block
_GEMM_COLUMN_TILES = _QKV_WIDTH // 256  # 84 column tile pairs
_GEMM_CTA_GROUP = 2
_ACTIVATION_SCALE_ROWS_PER_TILE = 4
_WEIGHT_SCALE_ROWS_PER_TILE = 8


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


def _stage_workspace(
    value: Optional[torch.Tensor],
    *,
    name: str,
    record: dict,
    device: torch.device,
) -> Optional[torch.Tensor]:
    required = int(record["tma_workspace_bytes"])
    if required == 0:
        if value is not None:
            raise ValueError(f"{name} must be None for a by-value descriptor route")
        return None
    if value is None:
        raise ValueError(f"{name} must provide at least {required} caller-owned bytes")
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise TypeError(f"{name} must be a one-dimensional torch.Tensor")
    if value.dtype != torch.uint8 or value.device != device:
        raise ValueError(f"{name} must be a CUDA uint8 tensor on {device}")
    if not value.is_cuda or not value.is_contiguous() or value.numel() < required:
        raise ValueError(
            f"{name} must be contiguous and contain at least {required} bytes"
        )
    if int(value.data_ptr()) % 128:
        raise ValueError(f"{name} must be 128-byte aligned")
    return value


def _gemm_row_tiles(rule: dict, *, M: int) -> int:
    """128-row output tiles of the fused GEMM, padded to whole CTA pairs."""
    tiles = -(-M // int(rule["block_m"]))
    cta_group = int(rule["cta_group"])
    return tiles + tiles % cta_group


def _stage_launch_grid(record: dict, *, M: int, P: int) -> tuple[int, int, int]:
    """Launch grid of one generated stage for the runtime token count ``M``.

    The route record carries the rule with the constants of the generated
    program (rows per CTA, tile geometry); the token count is not part of the
    program identity, so the grid is computed here for every call.
    """
    rule = record["launch_grid_rule"]
    kind = str(rule["kind"])
    if kind == "norm_rows":
        # One CTA per ``rows_per_cta`` rows of the ``row_alignment``-padded
        # activation scale tile (the padding rows are zeroed by the stage).
        rows = _round_up(M, int(rule["row_alignment"]))
        return (rows // int(rule["rows_per_cta"]), 1, 1)
    if kind == "gemm_cluster_tiles":
        # One CTA pair (cluster) per pair of row tiles x column tile pair; the
        # persistent clusters claim the remaining tiles through the cluster
        # launch-control scheduler, so the grid is the whole tile space.
        row_tiles = _gemm_row_tiles(rule, M=M)
        cta_group = int(rule["cta_group"])
        return ((row_tiles // cta_group) * int(rule["n_tiles"]) * cta_group, 1, 1)
    raise RuntimeError(f"generated stage has unsupported launch grid rule {kind!r}")


def _stage_call_args(
    record: dict,
    values: dict,
    *,
    workspace: Optional[torch.Tensor],
    grid: tuple[int, ...],
) -> tuple:
    grid_values = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    args = []
    for raw_kind, raw_name in record["arg_plan"]:
        kind = str(raw_kind)
        name = str(raw_name)
        if kind in {"buffer", "tma_buffer", "parameter"}:
            if name not in values:
                raise RuntimeError(
                    f"generated stage requires unknown argument {name!r}"
                )
            args.append(values[name])
        elif kind == "workspace" and name == "tma_descriptor_workspace":
            if workspace is None:
                raise RuntimeError("generated stage requires descriptor workspace")
            args.append(workspace)
        elif kind == "grid" and name in grid_values:
            args.append(grid_values[name])
        else:
            raise RuntimeError(
                f"generated stage has unsupported argument {(kind, name)!r}"
            )
    return tuple(args)


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
    """Exact-shape prepared operation with caller-owned storage."""

    def __init__(
        self,
        *,
        M: int,
        P: int,
        eps: float,
        norm_module,
        gemm_module,
        norm_args: tuple,
        gemm_args: tuple,
        gemm_alpha: torch.Tensor,
        gemm_weight_scales: torch.Tensor,
        placeholders: tuple[torch.Tensor, ...],
        out_q: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        self.M = M
        self.P = P
        self.eps = eps
        self._norm_module = norm_module
        self._gemm_module = gemm_module
        self._norm_args = norm_args
        self._gemm_args = gemm_args
        # Keeps the device operands derived at preparation (alpha, pair-ordered
        # weight scales, unbound-debug placeholders) alive for the bound args.
        self._gemm_alpha = gemm_alpha
        self._gemm_weight_scales = gemm_weight_scales
        self._placeholders = placeholders
        self.out_q = out_q
        self.out_sf = out_sf

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The output scale tile pads rows to 128 per destination; the fused
        # GEMM overwrites every live scale and this asynchronous clear makes
        # the padding bytes deterministic. The activation scale tile is
        # zero-padded inside the norm stage, so it needs no separate clear.
        self.out_sf.zero_()
        with tvm_ffi.use_torch_stream():
            self._norm_module.run(*self._norm_args)
            self._gemm_module.run(*self._gemm_args)
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
    out_global_scale: torch.Tensor,
    out_q: torch.Tensor,
    out_sf: torch.Tensor,
    activation_q: torch.Tensor,
    activation_sf: torch.Tensor,
    P: int,
    norm_descriptor_workspace: Optional[torch.Tensor] = None,
    gemm_descriptor_workspace: Optional[torch.Tensor] = None,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    debug_adaln_bf16: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> PreparedMiniMaxH3Nvfp4PreAttention:
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    if float(eps) != _EPS:
        raise ValueError(f"eps must be exactly {_EPS}")
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise TypeError("x must be a two-dimensional torch.Tensor")
    M = int(x.shape[0])
    if M <= 0:
        raise ValueError("M must be positive")
    device = x.device
    heads_per_destination = _HEADS // P
    rows_per_destination = M * heads_per_destination * _KINDS
    activation_sf_len = _round_up(M, 128) * _ACTIVATION_SCALE_COLS
    out_sf_stride = _round_up(rows_per_destination, 128) * _OUTPUT_SCALE_COLS
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
        "out_global_scale": (out_global_scale, (1,), torch.float32),
        "out_q": (
            out_q,
            (P, M, heads_per_destination, _KINDS, _OUTPUT_PACKED_COLS),
            torch.uint8,
        ),
        "out_sf": (out_sf, (P, out_sf_stride), torch.uint8),
        "activation_q": (activation_q, (M, _ACTIVATION_PACKED_COLS), torch.uint8),
        "activation_sf": (activation_sf, (activation_sf_len,), torch.uint8),
    }
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
    placeholders: tuple[torch.Tensor, ...] = ()
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
        placeholders = (debug_adaln_bf16, debug_q_words)
    route = minimax_h3_nvfp4_route_record(device, P)
    norm_record = route["stages"]["norm_adaln_nvfp4_quantize"]
    gemm_record = route["stages"]["qkv_nvfp4_gemm_fused_pack"]
    norm_descriptor_workspace = _stage_workspace(
        norm_descriptor_workspace,
        name="norm_descriptor_workspace",
        record=norm_record,
        device=device,
    )
    gemm_descriptor_workspace = _stage_workspace(
        gemm_descriptor_workspace,
        name="gemm_descriptor_workspace",
        record=gemm_record,
        device=device,
    )
    norm_module, gemm_module = load_minimax_h3_nvfp4_route(device, P)
    # Operands derived once at preparation: the GEMM output scale
    # alpha = 1 / (x_global_scale * w_global_scale) and the CTA-pair ordering
    # of the prepacked weight scales. Changing either global scale or the
    # weight scale bytes afterwards requires a new prepared operation; the
    # E2M1 weight bytes themselves are read in place on every launch.
    alpha = (
        (1.0 / (x_global_scale.float() * w_global_scale.float()))
        .reshape(1)
        .contiguous()
    )
    weight_scales = repack_minimax_h3_qkv_weight_scales_for_fused_gemm(qkv_weight_sf)
    gemm_rule = gemm_record["launch_grid_rule"]
    if str(gemm_rule["kind"]) != "gemm_cluster_tiles":
        raise RuntimeError(
            f"fused GEMM route has unsupported launch grid rule {gemm_rule['kind']!r}"
        )
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
        "OUTQ": out_q.reshape(
            P, M, heads_per_destination * _KINDS, _OUTPUT_PACKED_COLS
        ),
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_cos_sin": rope_cos_sin,
        "out_global_scale": out_global_scale,
        "out_q": out_q,
        "out_sf": out_sf,
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
        "m_tiles": _gemm_row_tiles(gemm_rule, M=M),
        "HEADS_PER_DESTINATION": heads_per_destination,
        "ROWS_PER_DESTINATION": rows_per_destination,
        "SCALE_STRIDE": out_sf_stride,
    }
    norm_args = _stage_call_args(
        norm_record,
        values,
        workspace=norm_descriptor_workspace,
        grid=_stage_launch_grid(norm_record, M=M, P=P),
    )
    gemm_args = _stage_call_args(
        gemm_record,
        values,
        workspace=gemm_descriptor_workspace,
        grid=_stage_launch_grid(gemm_record, M=M, P=P),
    )
    return PreparedMiniMaxH3Nvfp4PreAttention(
        M=M,
        P=P,
        eps=eps,
        norm_module=norm_module,
        gemm_module=gemm_module,
        norm_args=norm_args,
        gemm_args=gemm_args,
        gemm_alpha=alpha,
        gemm_weight_scales=weight_scales,
        placeholders=placeholders,
        out_q=out_q,
        out_sf=out_sf,
    )


__all__ = [
    "PreparedMiniMaxH3Nvfp4PreAttention",
    "prepare_minimax_h3_nvfp4_pre_attention",
    "repack_minimax_h3_qkv_weight_scales_for_fused_gemm",
]
