# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 MXFP8 pre-attention operation."""

from __future__ import annotations

from typing import Optional

import torch
import tvm_ffi

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm import gemm_base
from flashinfer.jit.cake_minimax_h3_mxfp8_pre_attention import (
    load_minimax_h3_mxfp8_route,
    minimax_h3_mxfp8_route_record,
)

_HIDDEN = 5376
_HEADS = 56
_KINDS = 3
_HEAD_DIM = 128
_QKV_WIDTH = _HEADS * _KINDS * _HEAD_DIM
_ADALN_ROWS = 9
_EPS = 1.0e-5
_SUPPORTED_PARTITIONS = (1, 2, 4, 8)


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


class PreparedMiniMaxH3Mxfp8PreAttention:
    """Exact-shape prepared operation with caller-owned storage."""

    def __init__(
        self,
        *,
        M: int,
        P: int,
        eps: float,
        norm_module,
        post_module,
        norm_args: tuple,
        post_args: tuple,
        gemm_runner,
        gemm_tactic: int,
        gemm_inputs: list,
        activation_sf: torch.Tensor,
        out_q: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        self.M = M
        self.P = P
        self.eps = eps
        self._norm_module = norm_module
        self._post_module = post_module
        self._norm_args = norm_args
        self._post_args = post_args
        self._gemm_runner = gemm_runner
        self._gemm_tactic = gemm_tactic
        self._gemm_inputs = gemm_inputs
        self._activation_sf = activation_sf
        self.out_q = out_q
        self.out_sf = out_sf

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # F8_128x4 padding bytes must be deterministic. Both asynchronous
        # clears are part of the prepared operation's measured launch chain.
        self.out_sf.zero_()
        self._activation_sf.zero_()
        with tvm_ffi.use_torch_stream():
            self._norm_module.run(*self._norm_args)
        self._gemm_runner(
            inputs=self._gemm_inputs,
            tactic=self._gemm_tactic,
        )
        with tvm_ffi.use_torch_stream():
            self._post_module.run(*self._post_args)
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
    activation_sf_len = _round_up(M, 128) * (_HIDDEN // 32)
    out_sf_stride = _round_up(rows_per_destination, 128) * (_HEAD_DIM // 32)

    tensors = {
        "x": (x, (M, _HIDDEN), torch.bfloat16),
        "x_norm_weight": (x_norm_weight, (_HIDDEN,), torch.bfloat16),
        "adaln_scale": (adaln_scale, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_shift": (adaln_shift, (_ADALN_ROWS, _HIDDEN), torch.bfloat16),
        "adaln_index": (adaln_index, (M,), torch.int32),
        "qkv_weight_q": (qkv_weight_q, (_QKV_WIDTH, _HIDDEN), torch.float8_e4m3fn),
        "qkv_weight_sf": (
            qkv_weight_sf,
            (_QKV_WIDTH * (_HIDDEN // 32),),
            torch.uint8,
        ),
        "q_norm_weight": (q_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "k_norm_weight": (k_norm_weight, (_HEAD_DIM,), torch.bfloat16),
        "rope_cos_sin": (rope_cos_sin, (M, 96), torch.bfloat16),
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
    write_debug = int(debug_q_bf16 is not None)
    if debug_q_bf16 is None:
        debug_q_bf16 = qkv_bf16
        debug_k_bf16 = qkv_bf16
    else:
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
    assert debug_q_bf16 is not None and debug_k_bf16 is not None

    route = minimax_h3_mxfp8_route_record(device, M, P)
    norm_record = route["stages"]["norm_adaln_mxfp8_quantize"]
    post_record = route["stages"]["qk_rope_destination_mxfp8_pack"]
    norm_descriptor_workspace = _stage_workspace(
        norm_descriptor_workspace,
        name="norm_descriptor_workspace",
        record=norm_record,
        device=device,
    )
    post_descriptor_workspace = _stage_workspace(
        post_descriptor_workspace,
        name="post_descriptor_workspace",
        record=post_record,
        device=device,
    )
    norm_module, post_module = load_minimax_h3_mxfp8_route(device, M, P)
    cutlass_module = gemm_base.get_cutlass_mxfp8_gemm_module(10)
    runner_factory = getattr(cutlass_module, "cutlass_mxfp8_gemm_runner", None)
    if not callable(runner_factory):
        raise RuntimeError("FlashInfer CUTLASS MXFP8 runner is unavailable")
    runner = runner_factory()
    weight_t = qkv_weight_q.T
    gemm_inputs = [
        activation_q,
        weight_t,
        activation_sf,
        qkv_weight_sf,
        torch.bfloat16,
        qkv_bf16,
        gemm_workspace,
    ]
    values = {
        "x": x,
        "x_norm_weight": x_norm_weight,
        "adaln_scale": adaln_scale,
        "adaln_shift": adaln_shift,
        "adaln_index": adaln_index,
        "activation_q": activation_q,
        "activation_sf": activation_sf,
        "qkv_bf16": qkv_bf16,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_cos_sin": rope_cos_sin,
        "out_q": out_q,
        "out_sf": out_sf,
        "debug_q_bf16": debug_q_bf16,
        "debug_k_bf16": debug_k_bf16,
        "write_debug": write_debug,
        "eps": eps,
    }
    norm_args = _stage_call_args(
        norm_record,
        values,
        workspace=norm_descriptor_workspace,
        grid=tuple(int(value) for value in norm_record["launch_grid"]),
    )
    post_args = _stage_call_args(
        post_record,
        values,
        workspace=post_descriptor_workspace,
        grid=tuple(int(value) for value in post_record["launch_grid"]),
    )

    activation_sf.zero_()
    with tvm_ffi.use_torch_stream():
        norm_module.run(*norm_args)
    with autotune(tune_mode=True, tuning_buckets=(M,), round_up=False):
        selected_runner, tactic = AutoTuner.get().choose_one(
            custom_op="mxfp8_gemm",
            runners=[runner],
            tuning_config=gemm_base._MM_MXFP8_TUNING_CONFIG,
            inputs=gemm_inputs,
        )
    if type(tactic) is not int or tactic < 0:
        raise RuntimeError(
            f"CUTLASS MXFP8 preparation returned invalid tactic {tactic!r}"
        )
    return PreparedMiniMaxH3Mxfp8PreAttention(
        M=M,
        P=P,
        eps=eps,
        norm_module=norm_module,
        post_module=post_module,
        norm_args=norm_args,
        post_args=post_args,
        gemm_runner=selected_runner,
        gemm_tactic=tactic,
        gemm_inputs=gemm_inputs,
        activation_sf=activation_sf,
        out_q=out_q,
        out_sf=out_sf,
    )


__all__ = [
    "PreparedMiniMaxH3Mxfp8PreAttention",
    "prepare_minimax_h3_mxfp8_pre_attention",
]
