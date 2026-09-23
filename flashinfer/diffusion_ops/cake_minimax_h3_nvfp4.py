# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared MiniMax-H3 NVFP4 (W4A4) pre-attention operation."""

from __future__ import annotations

from typing import Optional

import torch
import tvm_ffi

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.gemm import gemm_base
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
_SUPPORTED_GEMM_BACKENDS = ("cutlass", "cudnn")


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


class PreparedMiniMaxH3Nvfp4PreAttention:
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
        gemm_backend: str,
        gemm_runner,
        gemm_tactic,
        gemm_inputs: list,
        gemm_alpha: torch.Tensor,
        out_q: torch.Tensor,
        out_sf: torch.Tensor,
    ) -> None:
        self.M = M
        self.P = P
        self.eps = eps
        self.gemm_backend = gemm_backend
        self._norm_module = norm_module
        self._post_module = post_module
        self._norm_args = norm_args
        self._post_args = post_args
        self._gemm_runner = gemm_runner
        self._gemm_tactic = gemm_tactic
        self._gemm_inputs = gemm_inputs
        # Keeps the device alpha operand alive for the life of the GEMM inputs.
        self._gemm_alpha = gemm_alpha
        self.out_q = out_q
        self.out_sf = out_sf

    def __call__(self) -> tuple[torch.Tensor, torch.Tensor]:
        # The output scale tile pads rows to 128 per destination; the pack
        # stage overwrites every live scale and this asynchronous clear makes
        # the padding bytes deterministic. The activation scale tile is
        # zero-padded inside the norm stage, so it needs no separate clear.
        self.out_sf.zero_()
        with tvm_ffi.use_torch_stream():
            self._norm_module.run(*self._norm_args)
        self._gemm_runner(
            inputs=self._gemm_inputs,
            tactic=self._gemm_tactic,
        )
        with tvm_ffi.use_torch_stream():
            self._post_module.run(*self._post_args)
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
    qkv_bf16: torch.Tensor,
    gemm_workspace: torch.Tensor,
    P: int,
    gemm_backends: tuple[str, ...] = ("cutlass",),
    norm_descriptor_workspace: Optional[torch.Tensor] = None,
    post_descriptor_workspace: Optional[torch.Tensor] = None,
    debug_q_bf16: Optional[torch.Tensor] = None,
    debug_k_bf16: Optional[torch.Tensor] = None,
    debug_adaln_bf16: Optional[torch.Tensor] = None,
    eps: float = _EPS,
) -> PreparedMiniMaxH3Nvfp4PreAttention:
    if not isinstance(P, int) or isinstance(P, bool) or P not in _SUPPORTED_PARTITIONS:
        raise ValueError(f"P must be one of {_SUPPORTED_PARTITIONS}")
    if float(eps) != _EPS:
        raise ValueError(f"eps must be exactly {_EPS}")
    gemm_backends = tuple(gemm_backends)
    if not gemm_backends or any(
        backend not in _SUPPORTED_GEMM_BACKENDS for backend in gemm_backends
    ):
        raise ValueError(
            f"gemm_backends must be a non-empty subset of {_SUPPORTED_GEMM_BACKENDS}"
        )
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
    debug_present = (
        debug_q_bf16 is not None,
        debug_k_bf16 is not None,
        debug_adaln_bf16 is not None,
    )
    if any(debug_present) and not all(debug_present):
        raise ValueError(
            "debug_q_bf16, debug_k_bf16, and debug_adaln_bf16 must be supplied "
            "together"
        )
    write_debug = int(all(debug_present))
    if debug_q_bf16 is None:
        debug_q_bf16 = qkv_bf16
        debug_k_bf16 = qkv_bf16
        debug_adaln_bf16 = qkv_bf16
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
        _require_tensor(
            debug_adaln_bf16,
            name="debug_adaln_bf16",
            shape=(M, _HIDDEN),
            dtype=torch.bfloat16,
            device=device,
        )
    assert (
        debug_q_bf16 is not None
        and debug_k_bf16 is not None
        and debug_adaln_bf16 is not None
    )

    route = minimax_h3_nvfp4_route_record(device, M, P)
    norm_record = route["stages"]["norm_adaln_nvfp4_quantize"]
    post_record = route["stages"]["qk_rope_destination_nvfp4_pack"]
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
    norm_module, post_module = load_minimax_h3_nvfp4_route(device, M, P)

    # FlashInfer ``mm_fp4`` runner order (gemm_base.mm_fp4): a, b, a_descale,
    # b_descale, alpha, out_dtype, out, block_size, use_nvfp4, workspace.
    # ``a`` is the [M, K/2] packed activation, ``b`` the column-major [K/2, N]
    # view of the [N, K/2] prepacked weight; both swizzled-128x4 scale tiles are
    # passed as 2-D views because the cuDNN runner rejects flat 1-D scales.
    tuning_config = gemm_base._MM_FP4_TUNING_CONFIG_128x4
    major, minor = (int(v) for v in torch.cuda.get_device_capability(device))
    runners = []
    runner_backend: dict[int, str] = {}
    for backend in gemm_backends:
        if backend == "cutlass":
            cutlass_module = gemm_base.get_cutlass_fp4_gemm_module(major, minor)
            runner_factory = getattr(cutlass_module, "cutlass_fp4_gemm_runner", None)
            if not callable(runner_factory):
                raise RuntimeError("FlashInfer CUTLASS NVFP4 runner is unavailable")
            runner = runner_factory()
        else:
            gemm_base._cudnn_available_or_raise_for_backend("cudnn")
            runner = gemm_base._cudnn_gemm_fp4_runner(tuning_config)
        runners.append(runner)
        runner_backend[id(runner)] = backend
    alpha = (
        (1.0 / (x_global_scale.float() * w_global_scale.float()))
        .reshape(1)
        .contiguous()
    )
    gemm_inputs = [
        activation_q,
        qkv_weight_q.T,
        activation_sf.view(-1, _ACTIVATION_SCALE_COLS),
        qkv_weight_sf.view(_QKV_WIDTH, _ACTIVATION_SCALE_COLS).T,
        alpha,
        torch.bfloat16,
        qkv_bf16,
        _FP4_BLOCK,
        True,
        gemm_workspace,
    ]
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
        "qkv_bf16": qkv_bf16,
        "q_norm_weight": q_norm_weight,
        "k_norm_weight": k_norm_weight,
        "rope_cos_sin": rope_cos_sin,
        "out_global_scale": out_global_scale,
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

    # Realistic activation operands before profiling the exact M.
    with tvm_ffi.use_torch_stream():
        norm_module.run(*norm_args)
    with autotune(tune_mode=True, tuning_buckets=(M,), round_up=False):
        selected_runner, tactic = AutoTuner.get().choose_one(
            custom_op="fp4_gemm",
            runners=runners,
            tuning_config=tuning_config,
            inputs=gemm_inputs,
        )
    selected_backend = runner_backend[id(selected_runner)]
    if selected_backend == "cutlass" and (type(tactic) is not int or tactic < 0):
        raise RuntimeError(
            f"CUTLASS NVFP4 preparation returned invalid tactic {tactic!r}"
        )
    if tactic is None:
        raise RuntimeError(f"{selected_backend} NVFP4 preparation returned no tactic")
    return PreparedMiniMaxH3Nvfp4PreAttention(
        M=M,
        P=P,
        eps=eps,
        norm_module=norm_module,
        post_module=post_module,
        norm_args=norm_args,
        post_args=post_args,
        gemm_backend=selected_backend,
        gemm_runner=selected_runner,
        gemm_tactic=tactic,
        gemm_inputs=gemm_inputs,
        gemm_alpha=alpha,
        out_q=out_q,
        out_sf=out_sf,
    )


__all__ = [
    "PreparedMiniMaxH3Nvfp4PreAttention",
    "prepare_minimax_h3_nvfp4_pre_attention",
]
