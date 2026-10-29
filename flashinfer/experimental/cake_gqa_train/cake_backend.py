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

Loader, launches and autograd of the Cake packed-varlen GQA training
attention (head dimension 256, SM100 / SM103 / SM107).

The registry (:mod:`.cake_jit`) is data plus the host plan: ``MODULES`` holds
one record per program (its ``library`` stem; the ``build.json`` entries are
``<library>_<arch>``), :func:`.cake_jit.grid` the launch grids and
:func:`.cake_jit.scratch_layout` the backward scratch.  :mod:`.cake_build`
serves a library from a prebuilt ``aot/`` file when a builder compiled it and
otherwise through FlashInfer's JIT from the sources and flags of
``build.json``; the prebuilt route imports nothing from ``flashinfer``.

Every launch is a plain stream-ordered launch on the current torch stream:
nothing cooperative, no PDL, no co-residency requirement.  Nothing here reads
an environment variable, reads device memory or synchronizes; grids and
scratch sizes come from tensor shapes, ``len(cu_seqlens) - 1`` and the SM
count.  The segment boundaries ``cu_seqlens_q`` / ``cu_seqlens_k`` (int32 on
the device, ``cu[0] == 0``, non-decreasing, ``cu_q[-1] == T_q``,
``cu_k[-1] == T_k``) are read by the kernels only.
"""

from __future__ import annotations

import functools
import math
import weakref
from collections.abc import Callable, Mapping
from typing import Any, Optional, Union

import torch
import tvm_ffi

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "DkvAccSource",
    "VarlenGQAAttentionFunction",
    "Workspace",
    "arch_for_capability",
    "default_workspace",
    "device_arch",
    "gqa_attention_bwd",
    "gqa_attention_fwd",
    "is_prebuilt",
    "library_name",
    "load",
    "parameters",
    "record",
    "run_record",
    "sm_count",
    "specs_for_arch",
    "stage_record",
    "varlen_gqa_attention",
    "workspace_bytes",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 /
# GB300 (``sm_103a``) and R200 (``sm_107a``), each compiled with its exact
# ``-gencode`` flag set.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
HEAD_DIM: int = cake_jit.HEAD_DIM
LOG2_E = math.log2(math.e)

# The three dK / dV outputs of the backward main kernel, by stage.
_BWD_STAGE = {"bf16": "bwd_bf16", "fp32": "bwd_fp32", "rowmap": "bwd_rowmap"}


# ----------------------------------------------------------------------------------------------------------------
# Devices


def arch_for_capability(capability: tuple[int, int]) -> Optional[str]:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device is not an exported target); cached
    per device index."""
    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake GQA training attention "
            f"target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


# ----------------------------------------------------------------------------------------------------------------
# Records and libraries (``build.json``)


def record(name: str) -> dict[str, Any]:
    """The registry record ``name`` (a key of ``MODULES``)."""
    found = cake_jit.MODULES.get(name)
    if found is None:
        raise KeyError(
            f"{name!r} is not a record of the Cake GQA training attention registry"
        )
    return found


@functools.cache
def stage_record(stage: str, arch: str) -> str:
    """The record serving ``stage`` on ``arch`` (``KeyError`` when none does)."""
    return cake_jit.stage_record(stage, arch)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    found = record(name)
    if arch not in found["arches"]:
        raise RuntimeError(
            f"Cake GQA training attention record {name} is not delivered for {arch}; its architectures: "
            f"{found['arches']}"
        )
    return f"{found['library']}_{arch}"


def is_prebuilt(name: str, arch: str) -> bool:
    """Whether the library of record ``name`` for ``arch`` is served from ``aot/`` (a builder compiled it)."""
    return cake_build.prebuilt_path(library_name(name, arch)).is_file()


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the prebuilt ``aot/`` library when a builder
    compiled it, else FlashInfer's JIT build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of ``build.json`` without a prebuilt file (for ahead-of-time
    warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


@functools.cache
def parameters(name: str) -> tuple[str, ...]:
    """The kernel parameter names of record ``name`` in ``run`` order (the argument plan without the grid)."""
    names: list[str] = []
    axes: list[str] = []
    for kind, key in record(name)["arg_plan"]:
        if kind == "grid":
            axes.append(key)
        else:
            if axes:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} follows the grid arguments"
                )
            names.append(key)
    if axes != ["grid_x", "grid_y", "grid_z"]:
        raise ValueError(
            f"record {name}: the argument plan must end with grid_x, grid_y, grid_z; got {axes}"
        )
    return tuple(names)


def run_record(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with its kernel parameters named in ``values`` on ``grid``, on the current torch
    stream of device ``device_index`` (the operands' device, made current for the launch)."""
    args = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    entry = getattr(load(name, device_arch(device_index)), record(name)["ffi_entry"])
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        entry(*args, int(grid[0]), int(grid[1]), int(grid[2]))


def _launch(
    stage: str, values: Mapping[str, Any], dims: Mapping[str, int], device_index: int
) -> None:
    grid = cake_jit.grid(stage, sm_count=sm_count(device_index), **dims)
    run_record(
        stage_record(stage, device_arch(device_index)),
        values,
        grid=grid,
        device_index=device_index,
    )


# ----------------------------------------------------------------------------------------------------------------
# Workspace


def workspace_bytes(total_q: int, num_heads_q: int, num_segments: int) -> int:
    """Bytes of the backward scratch one call of these sizes needs (FP32 ``dq_accum [H_q / 2, rows_pad, 256]``,
    ``lse_log2`` and ``dpsum [H_q / 2, rows_pad]``, ``rows_pad = round_up(2 T_q + 128 B, 128)``)."""
    return cake_jit.scratch_bytes(int(total_q), int(num_heads_q), int(num_segments))


class Workspace:
    """Device scratch of the GQA training attention, reused across calls and shapes.

    It holds the forward's tile counter (one ``uint32``, zeroed once; every launch leaves it zero) and the backward
    scratch, grown to the largest shape seen and never shrunk.  One workspace serves one stream at a time: two
    launches running concurrently on different streams need two workspaces.  Autograd keeps only a weak reference
    to it, so a checkpointed forward retains nothing through it.
    """

    def __init__(self, device: Union[torch.device, str, int, None] = None) -> None:
        if device is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        elif isinstance(device, int):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device(device)
        if resolved.type != "cuda":
            raise ValueError(f"a workspace lives on a CUDA device, not {resolved}")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        self.device = resolved
        self._counter: Optional[torch.Tensor] = None
        self._scratch: Optional[torch.Tensor] = None
        self._views: Optional[tuple[tuple[int, int, int], tuple[torch.Tensor, ...]]] = (
            None
        )

    @property
    def nbytes(self) -> int:
        """Device bytes the workspace holds now."""
        held = 0 if self._scratch is None else self._scratch.numel()
        return held + (
            0 if self._counter is None else self._counter.untyped_storage().nbytes()
        )

    def tile_counter(self) -> torch.Tensor:
        """The forward's ``uint32`` tile counter (zero between launches)."""
        if self._counter is None:
            self._counter = torch.zeros(1, dtype=torch.uint32, device=self.device)
        return self._counter

    def scratch(
        self, total_q: int, num_heads_q: int, num_segments: int
    ) -> tuple[torch.Tensor, ...]:
        """``(dq_accum, lse_log2, dpsum)`` views for one backward call, growing the scratch when it is too small."""
        key = (total_q, num_heads_q, num_segments)
        if self._views is not None and self._views[0] == key:
            return self._views[1]
        need = cake_jit.scratch_bytes(total_q, num_heads_q, num_segments)
        if self._scratch is None or self._scratch.numel() < need:
            self._views = None
            self._scratch = None
            self._scratch = torch.empty(need, dtype=torch.uint8, device=self.device)
        views = tuple(
            self._scratch[offset : offset + 4 * math.prod(shape)]
            .view(torch.float32)
            .view(shape)
            for offset, shape in cake_jit.scratch_layout(
                total_q, num_heads_q, num_segments
            ).values()
        )
        # One entry: the views of the last shape (a forward, its replay and its backward share one shape).
        self._views = (key, views)
        return views

    def release(self) -> None:
        """Drop the backward scratch (the tile counter stays)."""
        self._views = None
        self._scratch = None


_DEFAULT_WORKSPACES: dict[tuple[int, int], Workspace] = {}


def default_workspace(device_index: int) -> Workspace:
    """The workspace of the device's current stream, made on first use and kept for the process."""
    stream = torch.cuda.current_stream(device_index).stream_id
    found = _DEFAULT_WORKSPACES.get((device_index, stream))
    if found is None:
        found = _DEFAULT_WORKSPACES[(device_index, stream)] = Workspace(device_index)
    return found


# ----------------------------------------------------------------------------------------------------------------
# Checks (shapes, dtypes, strides and devices only: nothing reads tensor contents)


def _check_operands(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
) -> dict[str, int]:
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype != torch.bfloat16 or t.dim() != 3 or t.shape[2] != HEAD_DIM:
            raise ValueError(
                f"{name} must be bfloat16 [T, H, {HEAD_DIM}], got {t.dtype} {tuple(t.shape)}"
            )
        if not t.is_cuda or t.device != q.device:
            raise ValueError(
                f"{name} must be on q's CUDA device {q.device}, got {t.device}"
            )
        if t.stride(2) != 1 or t.stride(0) % 8 or t.stride(1) % 8 or t.data_ptr() % 16:
            raise ValueError(
                f"{name} must have a unit head-dim stride and 16-byte aligned token / head strides and base, got "
                f"strides {t.stride()}"
            )
    if v.shape != k.shape:
        raise ValueError(
            f"v must have k's shape {tuple(k.shape)}, got {tuple(v.shape)}"
        )
    total_q, heads_q = int(q.shape[0]), int(q.shape[1])
    total_k, heads_kv = int(k.shape[0]), int(k.shape[1])
    if heads_kv == 0 or heads_q % heads_kv or (heads_q // heads_kv) % 2:
        raise ValueError(
            f"the query heads ({heads_q}) must be an even multiple of the key heads ({heads_kv}): the backward packs "
            "two query heads of one key head per tile"
        )
    for name, cu in (("cu_seqlens_q", cu_seqlens_q), ("cu_seqlens_k", cu_seqlens_k)):
        if (
            cu.dtype != torch.int32
            or cu.dim() != 1
            or cu.device != q.device
            or not cu.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous int32 vector on {q.device}, got {cu.dtype} {cu.device}"
            )
    if cu_seqlens_q.numel() != cu_seqlens_k.numel() or cu_seqlens_q.numel() < 2:
        raise ValueError(
            "cu_seqlens_q and cu_seqlens_k must both hold num_segments + 1 >= 2 entries, got "
            f"{cu_seqlens_q.numel()} and {cu_seqlens_k.numel()}"
        )
    return {
        "total_q": total_q,
        "total_k": total_k,
        "num_heads_q": heads_q,
        "num_heads_kv": heads_kv,
        "num_segments": int(cu_seqlens_q.numel()) - 1,
    }


def _check_out(
    name: str,
    t: torch.Tensor,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if (
        tuple(t.shape) != shape
        or t.dtype != dtype
        or t.device != device
        or not t.is_contiguous()
    ):
        raise ValueError(
            f"{name} must be a contiguous {dtype} tensor of shape {shape} on {device}, got {t.dtype} "
            f"{tuple(t.shape)} on {t.device}"
        )


def _scale(softmax_scale: Optional[float]) -> float:
    return HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)


# ----------------------------------------------------------------------------------------------------------------
# Forward and backward


def gqa_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    workspace: Optional[Workspace] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward: ``(out [T_q, H_q, 256] bf16, lse [H_q, T_q] fp32)``, causal with bottom-right alignment per segment;
    a row that sees no key gets ``out = 0`` and ``lse = -inf``."""
    dims = _check_operands(q, k, v, cu_seqlens_q, cu_seqlens_k)
    total_q, heads_q = dims["total_q"], dims["num_heads_q"]
    if out is None:
        out = torch.empty(
            total_q, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=q.device
        )
    else:
        _check_out("out", out, (total_q, heads_q, HEAD_DIM), torch.bfloat16, q.device)
    if lse is None:
        lse = torch.empty(heads_q, total_q, dtype=torch.float32, device=q.device)
    else:
        _check_out("lse", lse, (heads_q, total_q), torch.float32, q.device)
    if total_q == 0:
        return out, lse
    if dims["total_k"] == 0:
        out.zero_()
        lse.fill_(-math.inf)
        return out, lse
    device_index = q.device.index
    ws = workspace if workspace is not None else default_workspace(device_index)
    scale = _scale(softmax_scale)
    values = {
        "Q": q,
        "K": k,
        "V": v,
        "O": out,
        "LSE": lse,
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "tile_counter": ws.tile_counter(),
        "num_segments": dims["num_segments"],
        "num_heads_q": heads_q,
        "group": heads_q // dims["num_heads_kv"],
        "total_q": total_q,
        "softmax_scale_log2": scale * LOG2_E,
    }
    _launch("fwd", values, dims, device_index)
    return out, lse


def gqa_attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    dout: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    dq: Optional[torch.Tensor] = None,
    dk: Optional[torch.Tensor] = None,
    dv: Optional[torch.Tensor] = None,
    dkv_dtype: torch.dtype = torch.bfloat16,
    dk_acc: Optional[torch.Tensor] = None,
    dv_acc: Optional[torch.Tensor] = None,
    row_map: Optional[torch.Tensor] = None,
    max_seqlen_k: Optional[int] = None,
    workspace: Optional[Workspace] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward: ``(dq, dk, dv)``.  ``dq`` is BF16 ``[T_q, H_q, 256]``.  dK / dV are summed over the query heads of
    each key head in FP32 and leave the kernel as:

    - ``dkv_dtype=torch.bfloat16`` (default): BF16 ``[T_k, H_kv, 256]``, rounded once;
    - ``dkv_dtype=torch.float32``: FP32 window gradients ``[T_k, H_kv, 256]``, stored (``dk`` / ``dv`` overwritten
      when given);
    - ``row_map`` given: FP32 adds into the caller's ``dk_acc`` / ``dv_acc`` ``[N_dst, H_kv, 256]``: window row ``j``
      adds into destination row ``row_map[j]`` (int32 ``[T_k]``, rows may repeat; out-of-range rows are not
      checked), and the destinations are returned as ``dk`` / ``dv``.

    ``max_seqlen_k`` (host int, optional) must be at least every segment's key count (the caller's contract, as
    FlashAttention's ``max_seqlen_k``: a smaller value leaves key blocks out); it sizes the backward grid, one CTA pair
    per 128 keys of the longest segment, per key head and segment.  ``None`` bounds it by ``T_k``: always correct,
    with more CTAs that retire at once.  It is never read from the device.
    """
    dims = _check_operands(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if max_seqlen_k is not None and int(max_seqlen_k) < 0:
        raise ValueError(f"max_seqlen_k must be >= 0, got {max_seqlen_k}")
    total_q, total_k = dims["total_q"], dims["total_k"]
    heads_q, heads_kv = dims["num_heads_q"], dims["num_heads_kv"]
    device = q.device
    _check_out("out", out, (total_q, heads_q, HEAD_DIM), torch.bfloat16, device)
    _check_out("dout", dout, (total_q, heads_q, HEAD_DIM), torch.bfloat16, device)
    _check_out("lse", lse, (heads_q, total_q), torch.float32, device)
    kv_shape = (total_k, heads_kv, HEAD_DIM)
    if row_map is not None:
        if dk_acc is None or dv_acc is None:
            raise ValueError("row_map needs both dk_acc and dv_acc")
        if (
            row_map.dtype != torch.int32
            or tuple(row_map.shape) != (total_k,)
            or row_map.device != device
        ):
            raise ValueError(
                f"row_map must be int32 [{total_k}] on {device}, got {row_map.dtype} {tuple(row_map.shape)}"
            )
        if not row_map.is_contiguous():
            raise ValueError("row_map must be contiguous")
        dst_shape = (int(dk_acc.shape[0]),) + kv_shape[1:] if dk_acc.dim() == 3 else ()
        _check_out("dk_acc", dk_acc, dst_shape, torch.float32, device)
        _check_out("dv_acc", dv_acc, dst_shape, torch.float32, device)
        mode, dk_out, dv_out = "rowmap", dk_acc, dv_acc
    else:
        if dk_acc is not None or dv_acc is not None:
            raise ValueError(
                "dk_acc / dv_acc need a row_map (use dk / dv with dkv_dtype=torch.float32 to store)"
            )
        if dkv_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(
                f"dkv_dtype must be torch.bfloat16 or torch.float32, got {dkv_dtype}"
            )
        mode = "bf16" if dkv_dtype == torch.bfloat16 else "fp32"
        if dk is None:
            dk = torch.empty(kv_shape, dtype=dkv_dtype, device=device)
        else:
            _check_out("dk", dk, kv_shape, dkv_dtype, device)
        if dv is None:
            dv = torch.empty(kv_shape, dtype=dkv_dtype, device=device)
        else:
            _check_out("dv", dv, kv_shape, dkv_dtype, device)
        dk_out, dv_out = dk, dv
    if dq is None:
        dq = torch.empty(
            total_q, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
    else:
        _check_out("dq", dq, (total_q, heads_q, HEAD_DIM), torch.bfloat16, device)
    if total_q == 0 or total_k == 0:
        # No (query, key) pair: every gradient is zero; a row-map destination keeps its contents.
        dq.zero_()
        if mode != "rowmap":
            dk_out.zero_()
            dv_out.zero_()
        return dq, dk_out, dv_out
    device_index = device.index
    ws = workspace if workspace is not None else default_workspace(device_index)
    dq_accum, lse_log2, dpsum = ws.scratch(total_q, heads_q, dims["num_segments"])
    scale = _scale(softmax_scale)
    rows = cake_jit.rows_pad(total_q, dims["num_segments"])
    common = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "num_segments": dims["num_segments"],
        "rows_pad": rows,
        "T_q": total_q,
        "H_q": heads_q,
        "softmax_scale": scale,
    }
    _launch(
        "bwd_pre",
        {
            **common,
            "O": out,
            "dO": dout,
            "LSE": lse,
            "LSE_log2": lse_log2,
            "dPsum": dpsum,
            "dQ_accum": dq_accum,
        },
        dims,
        device_index,
    )
    _launch(
        _BWD_STAGE[mode],
        {
            **common,
            "Q": q,
            "dO": dout,
            "K": k,
            "V": v,
            "lse_log2": lse_log2,
            "dpsum": dpsum,
            "dq_accum": dq_accum,
            "dk_out": dk_out,
            "dv_out": dv_out,
            "h_kv": heads_kv,
            "group": heads_q // heads_kv,
            "softmax_scale_log2": scale * LOG2_E,
            "row_map": row_map,
        },
        {
            **dims,
            "max_seqlen_k": None if max_seqlen_k is None else int(max_seqlen_k),
        },
        device_index,
    )
    _launch("bwd_post", {**common, "dQ_accum": dq_accum, "dQ": dq}, dims, device_index)
    return dq, dk_out, dv_out


# ----------------------------------------------------------------------------------------------------------------
# Autograd

# ``dkv_acc``: a zero-argument callable, called in backward, returning ``(dk_acc, dv_acc, row_map)`` (FP32 adds into
# the caller's destinations through the row map) or ``(dk_win, dv_win)`` (FP32 window gradients stored into them).
# Under activation checkpointing a destination passed at forward time would live for every layer at once; one the
# callable produces lives only while its backward runs.
DkvAccSource = Callable[
    [],
    Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
]


def _resolve_dkv_acc(
    dkv_acc: DkvAccSource,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    produced = dkv_acc()
    if not isinstance(produced, tuple) or len(produced) not in (2, 3):
        raise ValueError(
            "a dkv_acc callable returns (dk_acc, dv_acc, row_map) or (dk_win, dv_win)"
        )
    if not all(isinstance(t, torch.Tensor) for t in produced):
        raise ValueError("a dkv_acc callable returns tensors")
    if len(produced) == 2:
        return produced[0], produced[1], None
    return produced[0], produced[1], produced[2]


class VarlenGQAAttentionFunction(torch.autograd.Function):
    """``(out, lse) = attention(q, k, v)``; the saved set is ``(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)``
    through ``ctx.save_for_backward`` alone, with scalars, the ``dkv_acc`` callable and a weak reference to the
    workspace on ``ctx`` (a checkpointed forward keeps no tensor alive through ``ctx``)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        dkv_acc: Optional[DkvAccSource],
        max_seqlen_k: Optional[int],
        workspace: Optional[Workspace],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = gqa_attention_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale=softmax_scale,
            workspace=workspace,
        )
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.softmax_scale = softmax_scale
        ctx.max_seqlen_k = max_seqlen_k
        ctx.dkv_acc = dkv_acc
        ctx.workspace_ref = None if workspace is None else weakref.ref(workspace)
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(
        ctx: Any, dout: Optional[torch.Tensor], _dlse: Any
    ) -> tuple[Optional[torch.Tensor], ...]:
        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        workspace = None if ctx.workspace_ref is None else ctx.workspace_ref()
        if dout is None:
            dout = torch.zeros_like(out)
        elif not dout.is_contiguous():
            dout = dout.contiguous()
        kwargs: dict[str, Any] = {
            "softmax_scale": ctx.softmax_scale,
            "max_seqlen_k": ctx.max_seqlen_k,
            "workspace": workspace,
        }
        if ctx.dkv_acc is None:
            dq, dk, dv = gqa_attention_bwd(
                q, k, v, out, lse, dout, cu_seqlens_q, cu_seqlens_k, **kwargs
            )
            return dq, dk, dv, None, None, None, None, None, None
        dk_t, dv_t, row_map = _resolve_dkv_acc(ctx.dkv_acc)
        if row_map is None:
            kwargs.update(dk=dk_t, dv=dv_t, dkv_dtype=torch.float32)
        else:
            kwargs.update(dk_acc=dk_t, dv_acc=dv_t, row_map=row_map)
        dq = gqa_attention_bwd(
            q, k, v, out, lse, dout, cu_seqlens_q, cu_seqlens_k, **kwargs
        )[0]
        return dq, None, None, None, None, None, None, None, None


def varlen_gqa_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dkv_acc: Optional[DkvAccSource] = None,
    max_seqlen_k: Optional[int] = None,
    workspace: Optional[Workspace] = None,
    return_lse: bool = False,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Packed-varlen causal GQA attention with autograd (see :func:`gqa_attention_fwd` / :func:`gqa_attention_bwd`).

    ``dkv_acc=None``: autograd returns BF16 gradients for ``k`` / ``v``.  A ``dkv_acc`` callable is called in
    backward; it returns ``(dk_acc, dv_acc, row_map)`` (FP32 adds into the destinations) or ``(dk_win, dv_win)`` (FP32
    window gradients stored into them), and ``k`` / ``v`` then get no gradient.  ``max_seqlen_k``: see
    :func:`gqa_attention_bwd`.
    """
    if not causal:
        raise NotImplementedError(
            "only causal attention (bottom-right aligned per segment) is implemented"
        )
    if dkv_acc is not None and not callable(dkv_acc):
        raise TypeError(
            "dkv_acc must be None or a zero-argument callable (see DkvAccSource)"
        )
    out, lse = VarlenGQAAttentionFunction.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        _scale(softmax_scale),
        dkv_acc,
        None if max_seqlen_k is None else int(max_seqlen_k),
        workspace,
    )
    return (out, lse) if return_lse else out
