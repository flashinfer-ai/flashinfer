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

Compute-capability 10.0/10.3 backend for MiniMax Sparse Attention.

The Cake-generated programs behind this module are registered in
``flashinfer.jit.blackwell_msa``: ``ROUTES`` maps a logical route
(``<route key>:<stage>``) to the program that serves it on each target and
``MODULES`` carries the program's physical argument order.  This module owns
the public semantics only -- argument validation, route selection from the
inputs, the host-side plan of the long-prefill route and the CUDA-graph
workspace contract.  There are no environment switches and no shape-exact
routes: every TopK16 call on a supported device is served by the route its
inputs select.  Compute capability 10.7 is admitted for the packed-NVFP4
paged-KV routes only.
"""

from __future__ import annotations

import functools
import math
import threading
from contextlib import nullcontext
from typing import Any, Optional, Tuple

import torch
import tvm_ffi

from ..jit.blackwell_msa import (
    MODULES,
    BlackwellMSATarget,
    load_blackwell_msa_module,
    route_program,
)
from ..utils import get_compute_capability, get_device_sm_count

_BLOCK_SIZE = 128
_HEAD_DIM = 128
_TOPK = 16
# Compute capability 10.7 (Rubin) is admitted for the packed-NVFP4 paged-KV
# routes only; the dense SM100/SM103 programs are not qualified there and
# ``_program`` rejects them explicitly.
_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0), (10, 3), (10, 7)}
_M128_Q_TILE = 256
_GQA8_Q_TILE = 32
_GQA16_Q_TILE = 16
_UNIFORM_FP8_EVEN_WAVE_GRID = 128

_LONG_PARTIAL_SEGMENT_COUNT = 4
_LONG_WORK_GROUP_BUCKETS = (128, 64, 32, 16, 8, 4, 2, 1)
_LONG_GROUP_BOUNDARIES = _LONG_WORK_GROUP_BUCKETS[:-1]
_LONG_QSPLIT_Q_MASK = 0x00FF_FFFF
_LONG_QSPLIT_SLOT_SHIFT = 24
_LONG_QSPLIT_SINGLE_SHIFT = 28

_MODE_DECODE_ONLY = 1
_SPLIT_ADAPTIVE = 0

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.float8_e4m3fn: "float8_e4m3fn",
}
_GRID_AXES = {"grid_x": 0, "grid_y": 1, "grid_z": 2}


class MSASparseAttentionWorkspace:
    """Caller-owned storage for SM100/SM103 MSA CUDA graph capture.

    Construct one workspace per captured sparse-attention invocation. Warm it
    by calling the operation eagerly with the exact tensors, options, and CUDA
    stream that will be captured, then synchronize that stream before capture.
    The workspace owns output and temporary tensors whose addresses must stay
    stable for graph replay.

    A workspace binds to its first stream. Once it participates in capture it
    cannot be passed through Python again; graph replay remains valid for the
    lifetime of the workspace.
    """

    def __init__(self, device: torch.device | str) -> None:
        normalized_device = torch.device(device)
        if normalized_device.type != "cuda":
            raise ValueError("MSASparseAttentionWorkspace requires a CUDA device")
        if normalized_device.index is None:
            normalized_device = torch.device("cuda", torch.cuda.current_device())
        self.device = normalized_device
        self._lock = threading.Lock()
        self._buffers: dict[str, torch.Tensor] = {}
        self._long_prefill_state: dict = {}
        self._warmed_launches: set[tuple] = set()
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


_topk_warmed_devices: set[tuple[int, str]] = set()
_topk_warmed_devices_lock = threading.Lock()
_eager_dummies: dict[tuple, torch.Tensor] = {}
_eager_dummies_lock = threading.Lock()
_implicit_long_prefill_states: dict[tuple, dict] = {}
_implicit_long_prefill_states_lock = threading.Lock()


def is_blackwell_msa_device(device: torch.device | str) -> bool:
    """Return whether ``device`` is a supported SM100/SM103 MSA target."""

    normalized_device = torch.device(device)
    return (
        normalized_device.type == "cuda"
        and get_compute_capability(normalized_device) in _SUPPORTED_COMPUTE_CAPABILITIES
    )


# ---------------------------------------------------------------------------
# Device facts and program launch
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _device_facts(device_index: int) -> tuple[BlackwellMSATarget, int]:
    """``(target, multiprocessor count)`` of one device, resolved once."""

    from ..jit.cpp_ext import is_cuda_version_at_least

    device = torch.device("cuda", device_index)
    compute_capability = get_compute_capability(device)
    if compute_capability not in _SUPPORTED_COMPUTE_CAPABILITIES:
        raise RuntimeError(
            "the SM100/SM103 MSA backend requires compute capability 10.0 or 10.3; "
            f"got {compute_capability[0]}.{compute_capability[1]}"
        )
    if compute_capability == (10, 7):
        target: BlackwellMSATarget = "sm107a"  # type: ignore[assignment]
    elif compute_capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError(
                "MSA on compute capability 10.3 requires CUDA 12.9 or newer"
            )
        target = "sm103a"
    else:
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError(
                "MSA on compute capability 10.0 requires CUDA 12.8 or newer"
            )
        target = "sm100a"
    num_sms = int(get_device_sm_count(device))
    return target, num_sms


def _select_target(device: torch.device) -> BlackwellMSATarget:
    return _device_facts(_device_index(device))[0]


def _num_sms(device: torch.device) -> int:
    return _device_facts(_device_index(device))[1]


class _Program:
    """One loaded program: its FFI entry and physical argument order."""

    __slots__ = ("entry", "plan", "name")

    def __init__(
        self, name: str, entry: Any, plan: tuple[tuple[str, str], ...]
    ) -> None:
        self.name = name
        self.entry = entry
        self.plan = plan

    def launch(self, grid: tuple[int, int, int], **arguments: Any) -> None:
        """Launch on the current torch stream with the generated argument order."""

        values = []
        for kind, name in self.plan:
            if kind == "grid":
                values.append(int(grid[_GRID_AXES[name]]))
            else:
                values.append(arguments[name])
        with tvm_ffi.use_torch_stream():
            self.entry(*values)


@functools.cache
def _program(route: str, target: BlackwellMSATarget) -> _Program:
    if target == "sm107a":
        raise RuntimeError(
            "the dense SM100/SM103 MSA backend is not qualified on compute "
            "capability 10.7 (Rubin); only the packed-NVFP4 paged-KV routes are "
            "enabled there"
        )
    name = route_program(route, target)
    record = MODULES[name]
    module = load_blackwell_msa_module(name, target)
    plan = tuple((str(kind), str(argument)) for kind, argument in record["arg_plan"])
    return _Program(name, getattr(module, record["ffi_entry"]), plan)


# ---------------------------------------------------------------------------
# Workspace and launch contract
# ---------------------------------------------------------------------------


def _normalize_device(device: torch.device) -> torch.device:
    if device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _stream_ptr(device: torch.device) -> int:
    return int(torch.cuda.current_stream(device).cuda_stream)


def _bind_workspace(
    workspace: MSASparseAttentionWorkspace,
    *,
    device: torch.device,
    stream_ptr: int,
    capturing: bool,
) -> None:
    device = _normalize_device(device)
    if workspace.device != device:
        raise ValueError(
            f"MSASparseAttentionWorkspace is bound to {workspace.device}, "
            f"but MSA inputs are on {device}"
        )
    if workspace._bound_stream_ptr is None:
        workspace._bound_stream_ptr = stream_ptr
    elif workspace._bound_stream_ptr != stream_ptr:
        raise RuntimeError(
            "MSASparseAttentionWorkspace is bound to a different CUDA stream; "
            "warm and capture it on one stream"
        )
    if workspace._captured:
        reuse_kind = "captured again" if capturing else "reused eagerly"
        raise RuntimeError(
            "MSASparseAttentionWorkspace has already participated in CUDA graph "
            f"capture and cannot be {reuse_kind}"
        )


def _workspace_buffer(
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if workspace is None:
        return torch.empty(shape, dtype=dtype, device=device)
    tensor = workspace._buffers.get(name)
    valid = (
        tensor is not None
        and tensor.device == device
        and tensor.dtype == dtype
        and tuple(tensor.shape) == shape
        and tensor.is_contiguous()
    )
    if not valid:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"MSASparseAttentionWorkspace buffer {name!r} is not warmed "
                f"for shape {shape} and dtype {dtype}"
            )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        workspace._buffers[name] = tensor
    return tensor


def _eager_dummy(
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """A never-dereferenced descriptor carrier, allocated once per device."""

    if workspace is not None:
        return _workspace_buffer(workspace, name, shape, dtype=dtype, device=device)
    key = (_device_index(device), name, shape, dtype)
    with _eager_dummies_lock:
        tensor = _eager_dummies.get(key)
        if tensor is None:
            tensor = torch.empty(shape, dtype=dtype, device=device)
            _eager_dummies[key] = tensor
    return tensor


def _tensor_signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
    )


def _launch_signature(
    *,
    route: str,
    target: str,
    tensors: tuple[torch.Tensor, ...],
    scalars: tuple,
    grid: tuple[int, int, int],
) -> tuple:
    return (
        route,
        target,
        tuple(_tensor_signature(tensor) for tensor in tensors),
        scalars,
        grid,
    )


def _check_warmed_launch(
    workspace: Optional[MSASparseAttentionWorkspace],
    signature: tuple,
    *,
    capturing: bool,
) -> None:
    if capturing and (workspace is None or signature not in workspace._warmed_launches):
        raise RuntimeError(
            "MSA CUDA graph capture requires an explicit "
            "MSASparseAttentionWorkspace warmed by an eager call with the "
            "exact tensors, options, and capture stream"
        )


def _record_successful_launch(
    workspace: Optional[MSASparseAttentionWorkspace],
    signature: Optional[tuple],
    *,
    capturing: bool,
) -> None:
    if workspace is None:
        return
    if capturing:
        workspace._captured = True
    elif signature is not None:
        workspace._warmed_launches.add(signature)


def _signature_tensors(arguments: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(
        value for value in arguments.values() if isinstance(value, torch.Tensor)
    )


def _signature_scalars(arguments: dict[str, Any]) -> tuple:
    return tuple(
        value for value in arguments.values() if not isinstance(value, torch.Tensor)
    )


def _enter_workspace(
    workspace: Optional[MSASparseAttentionWorkspace],
    *,
    device: torch.device,
    capturing: bool,
):
    if capturing and workspace is None:
        raise RuntimeError(
            "CUDA graph capture of MSA on compute capability 10.0/10.3/10.7 "
            "requires an explicit MSASparseAttentionWorkspace warmed with the "
            "exact tensors and capture stream"
        )
    if workspace is not None and not isinstance(workspace, MSASparseAttentionWorkspace):
        raise TypeError("workspace must be an MSASparseAttentionWorkspace")
    if workspace is None:
        return nullcontext()
    _bind_workspace(
        workspace, device=device, stream_ptr=_stream_ptr(device), capturing=capturing
    )
    return workspace._lock


# ---------------------------------------------------------------------------
# Argument validation and layout
# ---------------------------------------------------------------------------


def _require_cuda_i32(
    value,
    *,
    device: torch.device,
    name: str,
    length: Optional[int] = None,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} must be a preallocated CUDA int32 tensor during graph capture"
            )
        value = torch.as_tensor(value, dtype=torch.int32, device=device)
    elif (
        value.device != device
        or value.dtype != torch.int32
        or not value.is_contiguous()
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"{name} must already be contiguous CUDA int32 on {device} during graph capture"
            )
        value = value.to(
            device=device,
            dtype=torch.int32,
            non_blocking=True,
            memory_format=torch.contiguous_format,
        )
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and value.numel() != length:
        raise ValueError(f"{name} must contain {length} entries")
    return value


def _explicit_q_offsets(
    q_offset,
    *,
    batch_size: int,
    device: torch.device,
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
) -> torch.Tensor:
    if isinstance(q_offset, int):
        offsets = _workspace_buffer(
            workspace, name, (batch_size,), dtype=torch.int32, device=device
        )
        offsets.fill_(q_offset)
        return offsets
    return _require_cuda_i32(
        q_offset, device=device, name="q_offset", length=batch_size
    )


def _cumulative_kv_lengths(
    kv_lens: torch.Tensor,
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
) -> torch.Tensor:
    cu_k = _workspace_buffer(
        workspace,
        name,
        (kv_lens.numel() + 1,),
        dtype=torch.int32,
        device=kv_lens.device,
    )
    cu_k[0].zero_()
    torch.cumsum(kv_lens, dim=0, dtype=torch.int32, out=cu_k[1:])
    return cu_k


def _validate_scale_arguments(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    allow_uniform_fp8: bool,
    nvfp4_decline_reason: Optional[str] = None,
) -> tuple[float, float]:
    if k.dtype == torch.uint8:
        if nvfp4_decline_reason:
            raise NotImplementedError(
                "NVFP4 K/V MSA on compute capability 10.0/10.3/10.7 declined this "
                f"call: {nvfp4_decline_reason}. NVFP4 K/V IS supported on this "
                "architecture -- this shape is not, and there is no other "
                "implementation of this operation over an NVFP4 cache, so the "
                "call cannot be served at any speed."
            )
        raise NotImplementedError(
            "NVFP4 K/V is not supported by MSA on compute capability 10.0/10.3/10.7"
        )
    if k_scale is not None or v_scale is not None:
        raise NotImplementedError(
            "tensor K/V scales are not supported by MSA on compute capability 10.0/10.3/10.7"
        )
    uniform_fp8 = q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
    if (k_global_scale is not None or v_global_scale is not None) and not (
        allow_uniform_fp8 and uniform_fp8
    ):
        raise NotImplementedError("global K/V scales require uniform FP8 Q/K/V decode")
    return (
        1.0 if k_global_scale is None else float(k_global_scale),
        1.0 if v_global_scale is None else float(v_global_scale),
    )


def _validate_attention_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
) -> tuple[int, int, int, int]:
    if not isinstance(q, torch.Tensor) or not q.is_cuda:
        raise ValueError("q must be a CUDA tensor")
    if q.dtype not in _DTYPE_NAMES:
        raise ValueError(f"q must be bf16/fp16/fp8, got {q.dtype}")
    if q.ndim != 3 or q.shape[2] != _HEAD_DIM:
        raise ValueError(f"q must have shape (total_q, num_q_heads, {_HEAD_DIM})")
    if not q.is_contiguous():
        raise ValueError("q must be contiguous")
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    if total_q <= 0 or num_q_heads <= 0:
        raise ValueError("q must contain at least one query and one head")

    if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        raise ValueError("k and v must be CUDA tensors")
    if k.ndim not in (3, 4):
        raise ValueError("k/v must use a flat 3D or paged 4D layout")
    if k.device != q.device or v.device != q.device:
        raise ValueError("k/v must be on the same device as q")
    if k.shape != v.shape or k.dtype != v.dtype:
        raise ValueError("k/v must have the same shape and dtype")
    if not k.is_contiguous() or not v.is_contiguous():
        if k.ndim == 4:
            raise ValueError(
                "MSA on compute capability 10.0/10.3/10.7 does not directly support "
                "K/V views split from a packed paged cache; pass separate "
                "contiguous K and V tensors (implicit copies are not performed)"
            )
        raise ValueError("k/v must be contiguous")
    if k.dtype == torch.float8_e4m3fn:
        if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
            raise NotImplementedError(
                "FP8 K/V with FP16 Q is not supported on compute capability 10.0/10.3/10.7"
            )
    elif k.dtype != q.dtype:
        raise ValueError("dense k/v dtype must match q; FP8 K/V requires BF16 Q")
    num_kv_heads = int(k.shape[1])
    if num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        raise ValueError("num_q_heads must be a multiple of num_kv_heads")
    group_size = num_q_heads // num_kv_heads
    if not 0 < group_size <= 16:
        raise ValueError("the GQA group size must be in [1, 16]")

    if (
        not isinstance(q2k_indices, torch.Tensor)
        or q2k_indices.device != q.device
        or q2k_indices.dtype != torch.int32
        or q2k_indices.ndim != 3
        or tuple(q2k_indices.shape[:2]) != (num_kv_heads, total_q)
        or not q2k_indices.is_contiguous()
    ):
        raise ValueError(
            "q2k_indices must be contiguous CUDA int32 with shape (num_kv_heads, total_q, topk)"
        )
    if int(q2k_indices.shape[2]) != _TOPK:
        raise ValueError("Blackwell MSA sparse attention requires topk=16")
    return total_q, num_q_heads, num_kv_heads, group_size


def _prepare_layout(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    batch_size: int,
    prefill: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """``(paged, cu_k, kv_lens, page_table, max_pages)`` of one call."""

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged K/V requires seqused_k")
        if k.ndim != 4 or k.shape[2] != _BLOCK_SIZE or k.shape[3] != _HEAD_DIM:
            raise ValueError(
                "paged k/v must have shape (num_pages, num_kv_heads, 128, 128)"
            )
        if (
            not isinstance(page_table, torch.Tensor)
            or page_table.device != q.device
            or page_table.dtype != torch.int32
            or page_table.ndim != 2
            or page_table.shape[0] != batch_size
            or not page_table.is_contiguous()
        ):
            raise ValueError(
                "page_table must be contiguous CUDA int32 with shape (batch_size, max_pages)"
            )
        kv_lens = _require_cuda_i32(
            seqused_k, device=q.device, name="seqused_k", length=batch_size
        )
        if cu_seqlens_k is None:
            cu_k = (
                _cumulative_kv_lengths(
                    kv_lens, workspace=workspace, name="prefill_cu_seqlens_k"
                )
                if prefill
                else kv_lens
            )
        else:
            cu_k = _require_cuda_i32(
                cu_seqlens_k,
                device=q.device,
                name="cu_seqlens_k",
                length=batch_size + 1,
            )
        return True, cu_k, kv_lens, page_table, int(page_table.shape[1])

    if k.ndim != 3 or k.shape[2] != _HEAD_DIM:
        raise ValueError("flat k/v must have shape (total_k, num_kv_heads, 128)")
    if cu_seqlens_k is None:
        raise ValueError("flat K/V requires cu_seqlens_k")
    cu_k = _require_cuda_i32(
        cu_seqlens_k, device=q.device, name="cu_seqlens_k", length=batch_size + 1
    )
    return False, cu_k, cu_k, q.reshape(-1).view(torch.int32), 0


# ---------------------------------------------------------------------------
# Route selection (mirrors the Cake production dispatcher)
# ---------------------------------------------------------------------------


def _prefill_route(
    *,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    paged: bool,
    folded_gqa_group: int,
    causal: bool,
    max_pages: int,
) -> str:
    layout = "paged" if paged else "flat"
    causal_retrace = paged and folded_gqa_group == 16 and causal
    variant = (
        "causal_mask64"
        if causal_retrace and max_pages <= 64
        else "causal_large"
        if causal_retrace
        else "any"
    )
    return (
        f"prefill_union:{_DTYPE_NAMES[q_dtype]}:{_DTYPE_NAMES[k_dtype]}:{layout}:"
        f"gqa{folded_gqa_group}:{variant}"
    )


def _decode_route(*, q_dtype: torch.dtype, k_dtype: torch.dtype, paged: bool) -> str:
    layout = "paged" if paged else "flat"
    return f"decode_m16:{_DTYPE_NAMES[q_dtype]}:{_DTYPE_NAMES[k_dtype]}:{layout}"


def _long_prefill_route(*, paged: bool, group_size: int, direct_group: bool) -> str:
    layout = "paged" if paged else "flat"
    return f"long_bf16_reverse:{layout}:gqa{group_size}" + (
        ":direct_group" if direct_group else ""
    )


def _fp8_q1_schedule(
    *,
    capturing: bool,
    paged: bool,
    force_fused: Optional[bool],
    causal: bool,
    q_offset_is_none: bool,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    batch_size: int,
    total_q: int,
    seqlen_q: int,
    num_q_heads: int,
    num_kv_heads: int,
    k_outer_dim: int,
    max_pages: int,
) -> str:
    """The two FP8-KV Q1 serving specializations, selected from the inputs."""

    common = (
        not capturing
        and force_fused is True
        and causal
        and q_offset_is_none
        and q_dtype == torch.bfloat16
        and k_dtype == torch.float8_e4m3fn
        and batch_size == total_q
        and seqlen_q == 1
        and num_q_heads == 64
        and num_kv_heads == 4
    )
    if not common:
        return ""
    if paged and batch_size == 128 and k_outer_dim == 4096 and max_pages == 32:
        return "q1_paged_xform2"
    if not paged and batch_size == 32 and k_outer_dim == 262144 and max_pages == 0:
        return "q1_flat_xform2"
    return ""


def _uniform_fp8_decode_grid(
    *, total_work_items: int, num_sms: int, seqlen_q: int
) -> int:
    physical_grid = min(total_work_items, num_sms)
    default_waves = (total_work_items + physical_grid - 1) // physical_grid
    even_waves = (
        total_work_items + _UNIFORM_FP8_EVEN_WAVE_GRID - 1
    ) // _UNIFORM_FP8_EVEN_WAVE_GRID
    if (
        seqlen_q >= 4
        and num_sms >= _UNIFORM_FP8_EVEN_WAVE_GRID
        and total_work_items % _UNIFORM_FP8_EVEN_WAVE_GRID == 0
        and even_waves == default_waves
    ):
        return _UNIFORM_FP8_EVEN_WAVE_GRID
    return physical_grid


def _use_long_prefill(
    *,
    batch_size: int,
    total_q: int,
    paged: bool,
    group_size: int,
    max_pages: int,
    k_outer_dim: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    causal: bool,
    q_offset_is_none: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
) -> bool:
    return bool(
        batch_size == 1
        and total_q >= 8192
        and (
            (
                paged
                and (
                    (group_size == 8 and max_pages >= 64)
                    or (group_size == 16 and max_pages > 64)
                )
            )
            or (not paged and group_size == 16 and k_outer_dim >= 8192)
        )
        and q_dtype == k_dtype == v_dtype == torch.bfloat16
        and causal
        and q_offset_is_none
        and (not return_temperature_lse or lse_temperature_scale == 1.0)
    )


# ---------------------------------------------------------------------------
# Long prefill: host-side work plan over the selected blocks
# ---------------------------------------------------------------------------


def _long_plan_signature(
    q2k_indices: torch.Tensor,
    *,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> tuple:
    return (
        q2k_indices.device.type,
        q2k_indices.device.index,
        int(q2k_indices.data_ptr()),
        int(q2k_indices._version),
        tuple(q2k_indices.shape),
        total_k,
        num_sms,
        group_size,
        paged,
    )


def _long_state_tensor(
    state: dict,
    name: str,
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    tensor = state.get(name)
    valid = (
        isinstance(tensor, torch.Tensor)
        and tuple(tensor.shape) == shape
        and tensor.dtype == dtype
        and tensor.device == device
        and tensor.is_contiguous()
    )
    if not valid:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                f"long-prefill workspace entry {name!r} must be warmed before capture"
            )
        tensor = torch.empty(shape, dtype=dtype, device=device)
        state[name] = tensor
    return tensor


def _build_long_prefill_plan(
    q2k_indices: torch.Tensor,
    *,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> dict:
    """One exact single-request CSR of selected blocks plus the CTA work list."""

    num_kv_heads, total_q, topk = (int(value) for value in q2k_indices.shape)
    if topk != _TOPK:
        raise ValueError("long prefill requires topk=16")
    if group_size not in {8, 16} or (not paged and group_size != 16):
        raise ValueError("long prefill requires paged GQA8/GQA16 or flat GQA16")
    if total_q > _LONG_QSPLIT_Q_MASK:
        raise ValueError("long prefill exceeds the packed query-index range")
    total_rows = (total_k + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    nnz_per_head = total_q * topk
    q_tokens_per_group = 128 // group_size
    work_group_cap = _LONG_WORK_GROUP_BUCKETS[0]
    if paged:
        total_groups_upper = (
            num_kv_heads * total_q * topk + q_tokens_per_group - 1
        ) // q_tokens_per_group
        target = max(1, (total_groups_upper + 2 * num_sms - 1) // (2 * num_sms))
        work_group_cap = min(128, 1 << (target - 1).bit_length())
        if group_size == 8 and total_rows == 64:
            work_group_cap = max(work_group_cap, 64)
    buckets = tuple(
        value for value in _LONG_WORK_GROUP_BUCKETS if value <= work_group_cap
    )

    row_ptr = torch.zeros(
        (num_kv_heads, total_rows + 1), dtype=torch.int32, device=q2k_indices.device
    )
    qsplit = torch.full(
        (num_kv_heads, nnz_per_head), -1, dtype=torch.int32, device=q2k_indices.device
    )
    # [total_q, num_kv_heads] (the reducer's layout) straight from the reduction: no transpose copy.
    by_query = q2k_indices.transpose(0, 1)
    split_counts = ((by_query >= 0) & (by_query < total_rows)).sum(
        dim=2, dtype=torch.int32
    )
    counts_by_head: list[torch.Tensor] = []
    for head in range(num_kv_heads):
        flat = q2k_indices[head].reshape(-1)
        valid = (flat >= 0) & (flat < total_rows)
        positions = torch.nonzero(valid, as_tuple=False).flatten()
        blocks = flat.index_select(0, positions)
        counts = torch.bincount(blocks.to(torch.int64), minlength=total_rows).to(
            torch.int32
        )
        row_ptr[head, 1:] = torch.cumsum(counts, dim=0, dtype=torch.int32)
        counts_by_head.append(counts)
        order = torch.argsort(blocks, stable=True)
        sorted_positions = positions.index_select(0, order)
        q_indices = torch.div(sorted_positions, topk, rounding_mode="floor")
        slots = sorted_positions - q_indices * topk
        packed = q_indices.to(torch.int32) | (
            slots.to(torch.int32) << _LONG_QSPLIT_SLOT_SHIFT
        )
        packed |= (split_counts[:, head].index_select(0, q_indices) == 1).to(
            torch.int32
        ) << _LONG_QSPLIT_SINGLE_SHIFT
        qsplit[head, : packed.numel()] = packed

    # The host decomposes the per-row counts into the CTA work list: one device-to-host
    # transfer for all heads (the plan is rebuilt only when the selection changes).
    counts_host = torch.stack(counts_by_head).cpu().tolist()
    work: list[tuple[int, tuple[int, int, int, int, int, int]]] = []
    for head, counts in enumerate(counts_host):
        for kv_block, row_count in enumerate(counts):
            q_begin = 0
            remaining = row_count
            for group_count in buckets:
                capacity = group_count * q_tokens_per_group
                while (
                    remaining + q_tokens_per_group - 1
                ) // q_tokens_per_group >= group_count:
                    q_count = min(capacity, remaining)
                    work.append(
                        (group_count, (head, kv_block, q_begin, q_count, 0, kv_block))
                    )
                    q_begin += q_count
                    remaining -= q_count
            if remaining:
                raise AssertionError("long-prefill work decomposition failed")
    if not work:
        raise ValueError("long prefill requires at least one selected edge")
    work.sort(key=lambda item: item[0], reverse=True)
    metadata = torch.tensor(
        [entry for _group, entry in work], dtype=torch.int32, device=q2k_indices.device
    )
    counts_by_group = [0] * 129
    for group_count, _entry in work:
        counts_by_group[group_count] += 1
    running = 0
    end_by_group: dict[int, int] = {}
    for group_count in range(128, 1, -1):
        running += counts_by_group[group_count]
        end_by_group[group_count] = running
    return {
        "scheduler_metadata": metadata,
        "k2q_row_ptr": row_ptr,
        "k2q_qsplit_indices": qsplit,
        "split_counts": split_counts,
        "group_segment_ends": tuple(
            end_by_group[value] for value in _LONG_GROUP_BOUNDARIES
        ),
        "work_count": len(work),
        "total_rows": total_rows,
        "nnz_per_head": nnz_per_head,
    }


def _long_prefill_state(
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
    q2k_indices: torch.Tensor,
    total_k: int,
    num_sms: int,
    group_size: int,
    paged: bool,
) -> dict:
    signature = _long_plan_signature(
        q2k_indices,
        total_k=total_k,
        num_sms=num_sms,
        group_size=group_size,
        paged=paged,
    )
    if workspace is not None:
        state = workspace._long_prefill_state
    else:
        with _implicit_long_prefill_states_lock:
            state = _implicit_long_prefill_states.pop(signature, {})
            _implicit_long_prefill_states.clear()
            _implicit_long_prefill_states[signature] = state
    if state.get("signature") != signature:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "long-prefill plan must be warmed before CUDA graph capture"
            )
        state.clear()
        state["signature"] = signature
        state["plan"] = _build_long_prefill_plan(
            q2k_indices,
            total_k=total_k,
            num_sms=num_sms,
            group_size=group_size,
            paged=paged,
        )
        state["q2k_owner"] = q2k_indices
    return state


def _run_long_prefill(
    *,
    target: BlackwellMSATarget,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    temperature_lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    group_size: int,
    max_pages: int,
    softmax_scale_log2: float,
    lse_temperature_scale: float,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    total_k = max_pages * _BLOCK_SIZE if paged else int(k.shape[0])
    num_sms = _num_sms(q.device)
    state = _long_prefill_state(
        workspace=workspace,
        q2k_indices=q2k_indices,
        total_k=total_k,
        num_sms=num_sms,
        group_size=group_size,
        paged=paged,
    )
    plan = state["plan"]
    partial_o = _long_state_tensor(
        state,
        "partial_o",
        (_TOPK, total_q, num_q_heads, _HEAD_DIM),
        dtype=torch.uint8,
        device=q.device,
    )
    if paged:
        scale_shape: tuple[int, ...] = (
            _TOPK,
            total_q,
            num_q_heads,
            _LONG_PARTIAL_SEGMENT_COUNT,
        )
        scale_dtype = torch.bfloat16
    else:
        scale_shape = (2, _TOPK, total_q, num_q_heads)
        scale_dtype = torch.float32
    partial_scale = _long_state_tensor(
        state, "partial_scale", scale_shape, dtype=scale_dtype, device=q.device
    )
    partial_lse = _long_state_tensor(
        state,
        "partial_lse",
        (_TOPK, total_q, num_q_heads),
        dtype=torch.float32,
        device=q.device,
    )
    partial_temperature_lse = partial_lse
    if return_temperature_lse:
        partial_temperature_lse = _long_state_tensor(
            state,
            "partial_temperature_lse",
            (_TOPK, total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
    direct_group = (
        target == "sm100a" and paged and group_size == 16 and max_pages == 8192
    )
    route = _long_prefill_route(
        paged=paged, group_size=group_size, direct_group=direct_group
    )
    forward_grid = (int(plan["work_count"]), 1, 1)
    forward = {
        "q": q,
        "k": k,
        "v": v,
        "scheduler_metadata": plan["scheduler_metadata"],
        "k2q_row_ptr": plan["k2q_row_ptr"],
        "k2q_qsplit_indices": plan["k2q_qsplit_indices"],
        "partial_o": partial_o,
        "partial_scale": partial_scale,
        "partial_lse": partial_lse,
        "partial_temperature_lse": partial_temperature_lse,
        "out": out,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "q_offsets": q_offsets,
        "kv_lens": kv_lens,
        "page_table": page_table if paged else q2k_indices.reshape(-1),
        "total_q": total_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "total_rows": int(plan["total_rows"]),
        "nnz_per_head": int(plan["nnz_per_head"]),
        "work_capacity": int(plan["work_count"]),
        "num_work_items": int(plan["work_count"]),
        "topk": _TOPK,
        "max_pages": max_pages if paged else 0,
        "causal": 1,
        "derive_q_offset": 1,
        "softmax_scale_log2": softmax_scale_log2,
        "lse_temperature_scale": lse_temperature_scale,
        "return_temperature_lse": int(return_temperature_lse),
    }
    forward.update(
        {
            f"q_group_segment_end_{group_count}": int(value)
            for group_count, value in zip(
                _LONG_GROUP_BOUNDARIES, plan["group_segment_ends"], strict=True
            )
        }
    )
    combine_grid = ((total_q * num_q_heads + 31) // 32, 1, 1)
    combine = {
        "partial_o": partial_o,
        "partial_scale": partial_scale,
        "partial_lse": partial_lse,
        "partial_temperature_lse": partial_temperature_lse,
        "split_counts": plan["split_counts"],
        "out": out,
        "lse": lse,
        "temperature_lse": temperature_lse,
        "total_q": total_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "qhead_per_kv": group_size,
        "topk": _TOPK,
        "return_softmax_lse": int(return_softmax_lse or return_temperature_lse),
        "return_temperature_lse": int(return_temperature_lse),
    }
    signature = _launch_signature(
        route=route,
        target=target,
        tensors=(*_signature_tensors(forward), *_signature_tensors(combine)),
        scalars=(
            *_signature_scalars(forward),
            *_signature_scalars(combine),
            combine_grid,
        ),
        grid=forward_grid,
    )
    _check_warmed_launch(workspace, signature, capturing=capturing)
    _program(f"{route}:main", target).launch(forward_grid, **forward)
    _program(f"{route}:reduce", target).launch(combine_grid, **combine)
    _record_successful_launch(workspace, signature, capturing=capturing)


# ---------------------------------------------------------------------------
# Single-launch routes
# ---------------------------------------------------------------------------


def _launch_route(
    route: str,
    *,
    target: BlackwellMSATarget,
    grid: tuple[int, int, int],
    arguments: dict[str, Any],
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    signature = _launch_signature(
        route=route,
        target=target,
        tensors=_signature_tensors(arguments),
        scalars=_signature_scalars(arguments),
        grid=grid,
    )
    _check_warmed_launch(workspace, signature, capturing=capturing)
    _program(f"{route}:main", target).launch(grid, **arguments)
    _record_successful_launch(workspace, signature, capturing=capturing)


def _run_prefill(
    *,
    target: BlackwellMSATarget,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    temperature_lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    folded_gqa_group: int,
    batch_size: int,
    max_pages: int,
    causal: bool,
    derive_q_offset: bool,
    softmax_scale_log2: float,
    lse_temperature_scale: float,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    fp8_kv = k.dtype == torch.float8_e4m3fn
    # The union program also serves packed-NVFP4 K/V in its own route; the
    # dense routes bind a never-read scale carrier.
    scale_dummy = _eager_dummy(
        workspace,
        "prefill_scale_dummy",
        (1, 1, _BLOCK_SIZE, _HEAD_DIM // 16),
        dtype=torch.uint8,
        device=q.device,
    )
    q_tile = (
        _GQA8_Q_TILE
        if folded_gqa_group == 8
        else _GQA16_Q_TILE
        if folded_gqa_group == 16
        else _M128_Q_TILE
    )
    grid = (
        (total_q + q_tile - 1) // q_tile + batch_size - 1,
        num_kv_heads if folded_gqa_group else num_q_heads,
        1,
    )
    arguments = {
        "q": q,
        "k": k.view(torch.uint8) if fp8_kv else k,
        "k_scale": scale_dummy,
        "v": v.view(torch.uint8) if fp8_kv else v,
        "v_scale": scale_dummy,
        "out": out,
        "lse": lse,
        "temperature_lse": temperature_lse,
        "q2k_indices": q2k_indices,
        "cu_seqlens_q": cu_q,
        "cu_seqlens_k": cu_k,
        "q_offsets": q_offsets,
        "kv_lens": kv_lens,
        "page_table": page_table,
        "total_q": total_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "topk": _TOPK,
        "batch_size": batch_size,
        "uniform_q_len": 0,
        "max_pages": max_pages,
        "causal": int(causal),
        "derive_q_offset": int(derive_q_offset),
        "softmax_scale_log2": softmax_scale_log2,
        "k_global_scale": 1.0,
        "v_global_scale": 1.0,
        "lse_temperature_scale": lse_temperature_scale,
        "return_softmax_lse": int(return_softmax_lse or return_temperature_lse),
        "return_temperature_lse": int(return_temperature_lse),
    }
    route = _prefill_route(
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        paged=paged,
        folded_gqa_group=folded_gqa_group,
        causal=causal,
        max_pages=max_pages,
    )
    _launch_route(
        route,
        target=target,
        grid=grid,
        arguments=arguments,
        workspace=workspace,
        capturing=capturing,
    )


def _run_decode_m16(
    *,
    target: BlackwellMSATarget,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    max_pages: int,
    seqlen_q: int,
    softmax_scale_log2: float,
    causal: bool,
    derive_q_offset: bool,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    """The direct persistent M16 decode: one wave strides over (query, KV head) tickets."""

    fp8 = k.dtype == torch.float8_e4m3fn
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    i32_dummy = q2k_indices.reshape(-1)
    f32_dummy = lse.reshape(-1)
    if fp8:
        q_prefill_dummy = _eager_dummy(
            workspace,
            "decode_q_prefill_dummy",
            (128, 2, _HEAD_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        k_pair_dummy = q_prefill_dummy.reshape(2, 1, 128, _HEAD_DIM)
        v_pair_dummy = k_pair_dummy
        k_launch = k.view(torch.uint8)
        v_launch = v.view(torch.uint8)
    else:
        # Pure decode never dereferences the prefill-pair descriptors; bind
        # the largest 64-token-aligned prefix of K/V as their carrier.
        q_prefill_dummy = k.reshape(-1, 1, _HEAD_DIM)
        pair_tokens = int(q_prefill_dummy.shape[0]) // 64 * 64
        k_pair_dummy = q_prefill_dummy[:pair_tokens].reshape(-1, 1, 64, _HEAD_DIM)
        v_pair_dummy = v.reshape(-1, 1, _HEAD_DIM)[:pair_tokens].reshape(
            -1, 1, 64, _HEAD_DIM
        )
        k_launch = k
        v_launch = v
    scale_dummy = _eager_dummy(
        workspace,
        "decode_scale_dummy",
        (1, 1, 128, 8),
        dtype=torch.uint8,
        device=q.device,
    )
    status = _workspace_buffer(
        workspace, "decode_status", (2,), dtype=torch.int32, device=q.device
    )
    total_tasks = total_q * num_kv_heads
    physical_ctas = min(total_tasks, _num_sms(q.device))
    grid = (physical_ctas, 1, 1)
    arguments = {
        "Q": q,
        "Q_prefill": q_prefill_dummy,
        "Q_prefill_raw": q_prefill_dummy,
        "K": k_launch,
        "K_scale": scale_dummy,
        "K_prefill_pair": k_pair_dummy,
        "V": v_launch,
        "V_scale": scale_dummy,
        "V_prefill_pair": v_pair_dummy,
        "KV": q.reshape(-1, _HEAD_DIM),
        "O": out,
        "partial_O": f32_dummy,
        "partial_M": f32_dummy,
        "partial_D": f32_dummy,
        "split_completion": i32_dummy,
        "msa_lse": lse,
        "kv_indices": page_table if paged else i32_dummy,
        "qo_indptr": i32_dummy,
        "kv_indptr": cu_k,
        "kv_len_arr": kv_lens,
        "task_kind": q2k_indices,
        "task_request": q_offsets,
        "task_kv_head": kv_lens,
        "task_q_tile": i32_dummy,
        "task_split": i32_dummy,
        "task_kv_tile_begin": i32_dummy,
        "task_kv_tile_end": i32_dummy,
        "task_qo_begin": i32_dummy,
        "task_qo_end": i32_dummy,
        "task_page_begin": i32_dummy,
        "task_page_end": i32_dummy,
        "status": status,
        "num_requests": total_q,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "max_kv_tiles": _TOPK,
        "max_splits": 1,
        "max_task_claims": (total_tasks + physical_ctas - 1) // physical_ctas - 1,
        "softmax_scale_log2": softmax_scale_log2,
        "k_global_scale": 1.0,
        "v_global_scale": 1.0,
        "attention_mode": _MODE_DECODE_ONLY,
        "is_causal": int(causal),
        "derive_q_offset": int(derive_q_offset),
        "record_tasks": seqlen_q,
        "msa_max_pages": max_pages,
        "msa_split_policy": _SPLIT_ADAPTIVE,
    }
    route = _decode_route(q_dtype=q.dtype, k_dtype=k.dtype, paged=paged)
    _launch_route(
        route,
        target=target,
        grid=grid,
        arguments=arguments,
        workspace=workspace,
        capturing=capturing,
    )


def _run_fp8_direct(
    *,
    route: str,
    target: BlackwellMSATarget,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_k: torch.Tensor,
    q_offsets: torch.Tensor,
    kv_lens: torch.Tensor,
    page_table: torch.Tensor,
    paged: bool,
    max_pages: int,
    seqlen_q: int,
    softmax_scale_log2: float,
    output_scale: float,
    workspace: Optional[MSASparseAttentionWorkspace],
    capturing: bool,
) -> None:
    """One direct FP8-MMA decode program (BF16-Q xform2 or uniform FP8 Q/K/V)."""

    uniform = route == "decode_uniform_fp8:paged"
    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    arguments: dict[str, Any] = {
        "Q": q.view(torch.uint8) if uniform else q,
        "K": k.view(torch.uint8),
        "V": v.view(torch.uint8),
        "O": out,
        "msa_lse": lse,
        "kv_indices": page_table if paged else q2k_indices.reshape(-1),
        "kv_indptr": cu_k,
        "task_kind": q2k_indices,
        "task_request": q_offsets,
        "task_kv_head": kv_lens,
        "softmax_scale_log2": softmax_scale_log2,
        "msa_max_pages": max_pages,
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
    }
    if uniform:
        total_work_items = total_q * num_kv_heads
        arguments.update(
            total_q=total_q,
            seqlen_q=seqlen_q,
            output_scale=output_scale,
            K_scale=k.view(torch.uint8),
            V_scale=v.view(torch.uint8),
            partial_O=lse.reshape(-1),
            partial_M=lse.reshape(-1),
            partial_D=lse.reshape(-1),
            split_completion=q2k_indices.reshape(-1),
        )
        grid = (
            _uniform_fp8_decode_grid(
                total_work_items=total_work_items,
                num_sms=_num_sms(q.device),
                seqlen_q=seqlen_q,
            ),
            1,
            1,
        )
    else:
        arguments["num_requests"] = total_q
        grid = (total_q, num_kv_heads, 1)
    _launch_route(
        route,
        target=target,
        grid=grid,
        arguments=arguments,
        workspace=workspace,
        capturing=capturing,
    )


# ---------------------------------------------------------------------------
# Packed-NVFP4 paged-KV hand-offs
# ---------------------------------------------------------------------------


def _try_nvfp4_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    cu_seqlens_k,
    causal: bool,
    softmax_scale,
    page_table,
    seqused_k,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    q_offset,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Serve NVFP4 paged K/V prefill, or decline with a reason.

    Returns ``(output, None)`` when the route served the call and
    ``(None, reason)`` when it did not; the caller falls through on the
    latter and carries ``reason`` into the error it raises if nothing else
    can serve the call either.
    """

    from . import _nvfp4_prefill_sm100 as nvfp4

    reason = nvfp4.check_surface(
        q=q,
        k=k,
        v=v,
        q2k_indices=q2k_indices,
        cu_seqlens_q=cu_seqlens_q,
        page_table=page_table,
        seqused_k=seqused_k,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        return_softmax_lse=return_softmax_lse,
        return_temperature_lse=return_temperature_lse,
        lse_temperature_scale=lse_temperature_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        q_offset=q_offset,
    )
    if reason is not None:
        return None, reason
    k_scale = nvfp4.as_scale_bytes(k_scale)
    v_scale = nvfp4.as_scale_bytes(v_scale)

    total_q = int(q.shape[0])
    batch_size = int(cu_seqlens_q.shape[0]) - 1
    capturing = torch.cuda.is_current_stream_capturing()
    if not capturing:
        # A serving engine's profile run precedes graph capture; this keeps
        # every build out of a capture region.
        nvfp4.warm(q.device)
    with _enter_workspace(workspace, device=q.device, capturing=capturing):
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        out = _workspace_buffer(
            workspace,
            "prefill_nvfp4_out",
            tuple(q.shape),
            dtype=torch.bfloat16,
            device=q.device,
        )
        tiles = -(-total_q // 8) + batch_size
        signature = _launch_signature(
            route="prefill_nvfp4_kv_paged",
            target=_select_target(q.device),
            tensors=(
                q,
                k,
                v,
                k_scale,
                v_scale,
                q2k_indices,
                cu_seqlens_q,
                page_table,
                seqused_k,
                out,
            ),
            scalars=(
                scale,
                float(k_global_scale),
                float(v_global_scale),
                total_q,
                batch_size,
                int(page_table.shape[1]),
            ),
            grid=(tiles, int(k.shape[1]), 1),
        )
        _check_warmed_launch(workspace, signature, capturing=capturing)
        nvfp4.run(
            q=q,
            k=k,
            v=v,
            k_scale=k_scale,
            v_scale=v_scale,
            q2k_indices=q2k_indices,
            cu_seqlens_q=cu_seqlens_q,
            page_table=page_table,
            seqused_k=seqused_k,
            out=out,
            softmax_scale=scale,
            k_global_scale=float(k_global_scale),
            v_global_scale=float(v_global_scale),
        )
        _record_successful_launch(workspace, signature, capturing=capturing)
    return out, None


def _try_nvfp4_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table,
    seqused_k,
    cu_seqlens_k,
    seqlen_q: int,
    causal: bool,
    softmax_scale,
    return_softmax_lse: bool,
    k_scale,
    v_scale,
    k_global_scale,
    v_global_scale,
    q_offset,
    force_fused,
    workspace: Optional[MSASparseAttentionWorkspace],
    out: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], Optional[str]]:
    """Serve NVFP4 paged K/V decode, or decline with a reason.

    ``out``, when the caller supplies one, is the kernel's destination.
    """

    from . import _nvfp4_decode_sm100 as nvfp4

    reason = nvfp4.check_surface(
        q=q,
        k=k,
        v=v,
        q2k_indices=q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        cu_seqlens_k=cu_seqlens_k,
        seqlen_q=seqlen_q,
        causal=causal,
        return_softmax_lse=return_softmax_lse,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        q_offset=q_offset,
        force_fused=force_fused,
    )
    if reason is not None:
        return None, reason
    k_scale = nvfp4.as_scale_bytes(k_scale)
    v_scale = nvfp4.as_scale_bytes(v_scale)

    total_q = int(q.shape[0])
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and workspace is None and nvfp4.capture_requires_workspace():
        raise RuntimeError(
            "CUDA graph capture of MSA on compute capability 10.0/10.3/10.7 "
            "requires an explicit MSASparseAttentionWorkspace warmed with the "
            "exact tensors and capture stream"
        )
    if workspace is not None and not isinstance(workspace, MSASparseAttentionWorkspace):
        raise TypeError("workspace must be an MSASparseAttentionWorkspace")
    if not capturing:
        nvfp4.warm(q.device)
    context = workspace._lock if workspace is not None else nullcontext()
    with context:
        if workspace is not None:
            _bind_workspace(
                workspace,
                device=q.device,
                stream_ptr=_stream_ptr(q.device),
                capturing=capturing,
            )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        out = _decode_output(out, q=q, dtype=torch.bfloat16, workspace=workspace)
        num_kv_heads = int(k.shape[1])
        signature = None
        if workspace is not None:
            signature = _launch_signature(
                route="decode_nvfp4_kv_paged",
                target=_select_target(q.device),
                tensors=(
                    q,
                    k,
                    v,
                    k_scale,
                    v_scale,
                    q2k_indices,
                    page_table,
                    seqused_k,
                    out,
                ),
                scalars=(
                    scale,
                    float(k_global_scale),
                    float(v_global_scale),
                    total_q,
                    int(seqlen_q),
                    bool(causal),
                    int(page_table.shape[1]),
                ),
                grid=(total_q, num_kv_heads, 1),
            )
            _check_warmed_launch(workspace, signature, capturing=capturing)
        nvfp4.run(
            q=q,
            k=k,
            v=v,
            k_scale=k_scale,
            v_scale=v_scale,
            q2k_indices=q2k_indices,
            page_table=page_table,
            seqused_k=seqused_k,
            out=out,
            seqlen_q=int(seqlen_q),
            causal=bool(causal),
            softmax_scale=scale,
            k_global_scale=float(k_global_scale),
            v_global_scale=float(v_global_scale),
        )
        _record_successful_launch(workspace, signature, capturing=capturing)
    return out, None


def _decode_output(
    out: Optional[torch.Tensor],
    *,
    q: torch.Tensor,
    dtype: torch.dtype,
    workspace: Optional[MSASparseAttentionWorkspace],
) -> torch.Tensor:
    """The caller's ``out`` (validated) or the route's own output buffer."""

    if out is None:
        return _workspace_buffer(
            workspace, "decode_out", tuple(q.shape), dtype=dtype, device=q.device
        )
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if out.device != q.device:
        raise ValueError("out must be on the same device as q")
    if out.dtype != dtype:
        raise ValueError(f"out must be {dtype}, got {out.dtype}")
    if tuple(out.shape) != tuple(q.shape):
        raise ValueError(
            f"out must have q's shape {tuple(q.shape)}, got {tuple(out.shape)}"
        )
    if not out.is_contiguous():
        raise ValueError("out must be contiguous")
    return out


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def blackwell_msa_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    workspace: Optional[MSASparseAttentionWorkspace] = None,
):
    """Run sparse prefill on compute capability 10.0 or 10.3."""

    specialized, nvfp4_decline_reason = _try_nvfp4_prefill(
        q,
        k,
        v,
        q2k_indices,
        cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        causal=causal,
        softmax_scale=softmax_scale,
        page_table=page_table,
        seqused_k=seqused_k,
        return_softmax_lse=return_softmax_lse,
        return_temperature_lse=return_temperature_lse,
        lse_temperature_scale=lse_temperature_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        q_offset=q_offset,
        workspace=workspace,
    )
    if specialized is not None:
        return specialized
    _validate_scale_arguments(
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        allow_uniform_fp8=False,
        nvfp4_decline_reason=nvfp4_decline_reason,
    )
    total_q, num_q_heads, num_kv_heads, group_size = _validate_attention_tensors(
        q, k, v, q2k_indices
    )
    if q.dtype == torch.float8_e4m3fn:
        raise NotImplementedError(
            "uniform FP8 Q/K/V is supported only by sparse decode"
        )
    capturing = torch.cuda.is_current_stream_capturing()
    with _enter_workspace(workspace, device=q.device, capturing=capturing):
        cu_q = _require_cuda_i32(cu_seqlens_q, device=q.device, name="cu_seqlens_q")
        batch_size = cu_q.numel() - 1
        if batch_size <= 0:
            raise ValueError("cu_seqlens_q must contain at least two entries")
        paged, cu_k, kv_lens, page_table_arg, max_pages = _prepare_layout(
            q=q,
            k=k,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_k=cu_seqlens_k,
            batch_size=batch_size,
            prefill=True,
            workspace=workspace,
        )
        derive_q_offset = q_offset is None
        q_offsets = (
            cu_k
            if derive_q_offset
            else _explicit_q_offsets(
                q_offset,
                batch_size=batch_size,
                device=q.device,
                workspace=workspace,
                name="prefill_q_offsets",
            )
        )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        temperature_scale = float(lse_temperature_scale)
        if not math.isfinite(temperature_scale) or temperature_scale <= 0:
            raise ValueError("lse_temperature_scale must be positive and finite")
        target = _select_target(q.device)
        out = _workspace_buffer(
            workspace, "prefill_out", tuple(q.shape), dtype=q.dtype, device=q.device
        )
        lse = _workspace_buffer(
            workspace,
            "prefill_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        temperature_lse = _workspace_buffer(
            workspace,
            "prefill_temperature_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        common = dict(
            target=target,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            temperature_lse=temperature_lse,
            q2k_indices=q2k_indices,
            cu_q=cu_q,
            cu_k=cu_k,
            q_offsets=q_offsets,
            kv_lens=kv_lens,
            page_table=page_table_arg,
            paged=paged,
            max_pages=max_pages,
            softmax_scale_log2=scale / math.log(2.0),
            lse_temperature_scale=temperature_scale,
            return_softmax_lse=return_softmax_lse,
            return_temperature_lse=return_temperature_lse,
            workspace=workspace,
            capturing=capturing,
        )
        if _use_long_prefill(
            batch_size=batch_size,
            total_q=total_q,
            paged=paged,
            group_size=group_size,
            max_pages=max_pages,
            k_outer_dim=int(k.shape[0]),
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            causal=causal,
            q_offset_is_none=derive_q_offset,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=temperature_scale,
        ):
            _run_long_prefill(group_size=group_size, **common)
        else:
            folded_gqa_group = (
                group_size
                if group_size in {8, 16}
                and q.dtype == torch.bfloat16
                and k.dtype == torch.bfloat16
                else 0
            )
            _run_prefill(
                folded_gqa_group=folded_gqa_group,
                batch_size=batch_size,
                causal=causal,
                derive_q_offset=derive_q_offset,
                **common,
            )
    if return_temperature_lse:
        return out, lse, temperature_lse
    if return_softmax_lse:
        return out, lse
    return out


def blackwell_msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    partial_dtype: Optional[torch.dtype] = None,
    force_fused: Optional[bool] = None,
    workspace: Optional[MSASparseAttentionWorkspace] = None,
    out: Optional[torch.Tensor] = None,
):
    """Run sparse decode on compute capability 10.0 or 10.3.

    ``out``, when given, is the destination of every route and the returned
    tensor; the routes never copy into it.
    """

    del partial_dtype
    specialized, nvfp4_decline_reason = _try_nvfp4_decode(
        q,
        k,
        v,
        q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        cu_seqlens_k=cu_seqlens_k,
        seqlen_q=seqlen_q,
        causal=causal,
        softmax_scale=softmax_scale,
        return_softmax_lse=return_softmax_lse,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        q_offset=q_offset,
        force_fused=force_fused,
        workspace=workspace,
        out=out,
    )
    if specialized is not None:
        return specialized
    k_global_multiplier, output_scale = _validate_scale_arguments(
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
        allow_uniform_fp8=True,
        nvfp4_decline_reason=nvfp4_decline_reason,
    )
    total_q, num_q_heads, num_kv_heads, _ = _validate_attention_tensors(
        q, k, v, q2k_indices
    )
    if seqlen_q <= 0 or total_q % seqlen_q:
        raise ValueError("q rows must equal batch_size * positive seqlen_q")
    if force_fused not in (None, True, False):
        raise ValueError("force_fused must be True, False, or None")
    batch_size = total_q // seqlen_q
    capturing = torch.cuda.is_current_stream_capturing()
    with _enter_workspace(workspace, device=q.device, capturing=capturing):
        paged, cu_k, kv_lens, page_table_arg, max_pages = _prepare_layout(
            q=q,
            k=k,
            page_table=page_table,
            seqused_k=seqused_k,
            cu_seqlens_k=cu_seqlens_k,
            batch_size=batch_size,
            prefill=False,
            workspace=workspace,
        )
        derive_q_offset = q_offset is None
        q_offsets = (
            cu_k
            if derive_q_offset
            else _explicit_q_offsets(
                q_offset,
                batch_size=batch_size,
                device=q.device,
                workspace=workspace,
                name="decode_explicit_q_offsets",
            )
        )
        scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
        scale *= k_global_multiplier
        if not math.isfinite(scale):
            raise ValueError("softmax_scale must be finite")
        if not math.isfinite(output_scale):
            raise ValueError("v_global_scale must be finite")
        target = _select_target(q.device)
        uniform_fp8 = q.dtype == k.dtype == v.dtype == torch.float8_e4m3fn
        out = _decode_output(
            out,
            q=q,
            dtype=torch.bfloat16 if uniform_fp8 else q.dtype,
            workspace=workspace,
        )
        lse = _workspace_buffer(
            workspace,
            "decode_lse",
            (total_q, num_q_heads),
            dtype=torch.float32,
            device=q.device,
        )
        common = dict(
            target=target,
            q=q,
            k=k,
            v=v,
            out=out,
            lse=lse,
            q2k_indices=q2k_indices,
            cu_k=cu_k,
            q_offsets=q_offsets,
            kv_lens=kv_lens,
            page_table=page_table_arg,
            paged=paged,
            max_pages=max_pages,
            seqlen_q=int(seqlen_q),
            softmax_scale_log2=scale / math.log(2.0),
            workspace=workspace,
            capturing=capturing,
        )
        if uniform_fp8:
            if not (
                paged
                and force_fused is True
                and causal
                and derive_q_offset
                and 1 <= seqlen_q <= 32
            ):
                raise ValueError(
                    "uniform FP8 Q/K/V requires paged causal Q1-Q32/topk16, "
                    "force_fused=True, and no explicit q_offset"
                )
            _run_fp8_direct(
                route="decode_uniform_fp8:paged", output_scale=output_scale, **common
            )
        else:
            schedule = _fp8_q1_schedule(
                capturing=capturing,
                paged=paged,
                force_fused=force_fused,
                causal=causal,
                q_offset_is_none=derive_q_offset,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                batch_size=batch_size,
                total_q=total_q,
                seqlen_q=int(seqlen_q),
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                k_outer_dim=int(k.shape[0]),
                max_pages=max_pages,
            )
            if schedule:
                _run_fp8_direct(
                    route=f"decode_fp8_q1:{schedule}",
                    output_scale=output_scale,
                    **common,
                )
            else:
                _run_decode_m16(
                    causal=causal, derive_q_offset=derive_q_offset, **common
                )
    return (out, lse) if return_softmax_lse else out


def blackwell_msa_topk_select(
    max_score: torch.Tensor,
    topk: int,
    num_valid_pages: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    force_begin_blocks: int = 0,
    force_end_blocks: int = 0,
) -> torch.Tensor:
    """Select exact top-16 block indices on compute capability 10.0/10.3/10.7."""

    if not isinstance(max_score, torch.Tensor) or not max_score.is_cuda:
        raise ValueError("max_score must be a CUDA tensor")
    if max_score.dtype != torch.float32:
        raise ValueError(f"max_score must be float32, got {max_score.dtype}")
    if max_score.ndim != 3 or not max_score.is_contiguous():
        raise ValueError(
            "max_score must be contiguous with shape (num_q_heads, max_k_tiles, total_q)"
        )
    if topk != _TOPK:
        raise ValueError(f"topk must be {_TOPK}, got {topk}")
    num_heads, max_k_tiles, total_q = (int(value) for value in max_score.shape)
    if min(num_heads, max_k_tiles, total_q) <= 0:
        raise ValueError("max_score dimensions must be positive")
    valid = max_k_tiles if num_valid_pages is None else int(num_valid_pages)
    if not 0 < valid <= max_k_tiles:
        raise ValueError(f"num_valid_pages must be in (0, {max_k_tiles}], got {valid}")
    forced = force_begin_blocks + force_end_blocks
    if force_begin_blocks < 0 or force_end_blocks < 0:
        raise ValueError("force_begin_blocks and force_end_blocks must be non-negative")
    if forced > topk or forced > valid:
        raise ValueError(
            "force_begin_blocks + force_end_blocks must not exceed topk or num_valid_pages"
        )
    expected_shape = (total_q, num_heads, topk)
    if output is None:
        output = torch.empty(expected_shape, dtype=torch.int32, device=max_score.device)
    elif (
        output.device != max_score.device
        or output.dtype != torch.int32
        or tuple(output.shape) != expected_shape
        or not output.is_contiguous()
    ):
        raise ValueError(
            f"output must be contiguous CUDA int32 with shape {expected_shape}"
        )
    target = _select_target(max_score.device)
    warm_key = (_device_index(max_score.device), target)
    capturing = torch.cuda.is_current_stream_capturing()
    with _topk_warmed_devices_lock:
        warmed = warm_key in _topk_warmed_devices
    if capturing and not warmed:
        raise RuntimeError(
            "msa_topk_select must be invoked eagerly on this device before CUDA graph capture"
        )
    _program("topk_select:main", target).launch(
        (total_q * num_heads, 1, 1),
        max_score=max_score,
        output=output,
        num_heads=num_heads,
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        num_valid_pages=valid,
        force_begin_blocks=int(force_begin_blocks),
        force_end_blocks=int(force_end_blocks),
    )
    if not capturing:
        with _topk_warmed_devices_lock:
            _topk_warmed_devices.add(warm_key)
    return output


__all__ = [
    "MSASparseAttentionWorkspace",
    "blackwell_msa_sparse_attention",
    "blackwell_msa_sparse_decode_attention",
    "blackwell_msa_topk_select",
    "is_blackwell_msa_device",
]
