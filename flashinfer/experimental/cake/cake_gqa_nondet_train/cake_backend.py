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
nothing cooperative, no PDL, no co-residency requirement.  The backward
builds its tile tables (``bwd_plan``) and the row-map fold order on a
high-priority side stream of the device, forked from the current stream and
joined back by CUDA events before their first read, so they overlap
``bwd_pre`` and the main kernels.  Nothing here reads
an environment variable, reads device memory or synchronizes; grids and
scratch sizes come from tensor shapes, ``len(cu_seqlens) - 1`` and the SM
count.  The segment boundaries ``cu_seqlens_q`` / ``cu_seqlens_k`` (int32 on
the device, ``cu[0] == 0``, non-decreasing, ``cu_q[-1] == T_q``,
``cu_k[-1] == T_k``) are read by the kernels only.

The package is the non-deterministic one of its family (``nondet``; the name
keeps the mode word), yet every sum of the forward and the backward has one
fixed order: ``out``, ``lse``, dQ and every dK / dV form are bitwise
reproducible run to run for one shape, one ``max_seqlen_k``, one workspace
budget and the same host boundaries (the budget and the optional host copies
of the boundaries set the chunks and with them the dQ rounding).
The backward main kernel stores dS^T into a workspace and computes dK and dV
in tensor memory; dQ is a GEMM over the workspace.  Nothing adds with
atomics, so the package raises no PyTorch determinism alert.  The row-map
form adds the FP32 window rows into their destination rows in ascending
window order through a stable sort of the row map.
"""

from __future__ import annotations

import functools
import math
import weakref
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional, Union

import torch
import tvm_ffi

from . import cake_build, cake_jit

__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CakeGQANondetDkvAccSource",
    "CakeGQANondetFunction",
    "CakeGQANondetWorkspace",
    "HEAD_DIM",
    "arch_for_capability",
    "cake_gqa_nondet",
    "cake_gqa_nondet_train_backward",
    "cake_gqa_nondet_train_forward",
    "cake_gqa_nondet_train_workspace_bytes",
    "default_workspace",
    "device_arch",
    "is_prebuilt",
    "library_name",
    "load",
    "parameters",
    "record",
    "run_record",
    "sm_count",
    "specs_for_arch",
    "stage_record",
]

# One verified build per architecture: B200 / GB200 (``sm_100a``), B300 /
# GB300 (``sm_103a``) and R200 (``sm_107a``), each compiled with its exact
# ``-gencode`` flag set.
ARCHES: tuple[str, ...] = tuple(cake_jit.ARCHES)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = dict(cake_jit.ARCH_BY_CAPABILITY)
HEAD_DIM: int = cake_jit.HEAD_DIM
LOG2_E = math.log2(math.e)

# The main kernel of each dK / dV output (the row-map form runs the FP32 window, then the fold).
_MAIN_STAGE = {
    "bf16": "bwd_main_bf16",
    "fp32": "bwd_main_fp32",
    "rowmap": "bwd_main_fp32",
}


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


@functools.cache
def _entry(name: str, arch: str) -> Any:
    """The FFI entry of record ``name`` in its ``arch`` library (looked up once per process)."""
    return getattr(load(name, arch), record(name)["ffi_entry"])


def _run(
    name: str, values: Mapping[str, Any], grid: tuple[int, int, int], arch: str
) -> None:
    """Launch record ``name`` on the current device and torch stream: the caller holds
    ``torch.cuda.device(device_index)`` and ``tvm_ffi.use_torch_stream()`` (once per forward or backward call)."""
    args = []
    for key in parameters(name):
        try:
            args.append(values[key])
        except KeyError:
            raise ValueError(f"record {name} needs argument {key!r}") from None
    _entry(name, arch)(*args, int(grid[0]), int(grid[1]), int(grid[2]))


def run_record(
    name: str,
    values: Mapping[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with its kernel parameters named in ``values`` on ``grid``, on the current torch
    stream of device ``device_index`` (the operands' device, made current for the launch)."""
    arch = device_arch(device_index)
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        _run(name, values, grid, arch)


def _launch(
    stage: str,
    values: Mapping[str, Any],
    dims: Mapping[str, int],
    device_index: int,
    **chunk: int,
) -> None:
    """Launch ``stage`` inside the caller's device and stream context (see :func:`_run`)."""
    arch = device_arch(device_index)
    grid = cake_jit.grid(stage, sm_count=sm_count(device_index), **dims, **chunk)
    _run(stage_record(stage, arch), values, grid, arch)


# ----------------------------------------------------------------------------------------------------------------
# Workspace


def _budget(budget_bytes: Optional[int]) -> int:
    """The dS^T workspace budget of one backward call (``None``: :data:`.cake_jit.DEFAULT_WORKSPACE_BUDGET`)."""
    if budget_bytes is None:
        return cake_jit.DEFAULT_WORKSPACE_BUDGET
    if int(budget_bytes) <= 0:
        raise ValueError(f"budget_bytes must be positive, got {budget_bytes}")
    return int(budget_bytes)


HostBoundaries = Union[Sequence[int], torch.Tensor]


def _host_lists(
    cu_seqlens_q_host: Optional[HostBoundaries],
    cu_seqlens_k_host: Optional[HostBoundaries],
) -> Optional[tuple[list[int], list[int]]]:
    """The host boundaries as two lists of ints, or ``None`` when neither is given.  A tensor must be on the CPU:
    nothing here reads device memory."""
    if cu_seqlens_q_host is None and cu_seqlens_k_host is None:
        return None
    if cu_seqlens_q_host is None or cu_seqlens_k_host is None:
        raise ValueError("cu_seqlens_q_host and cu_seqlens_k_host are given together")
    lists = []
    for name, cu in (
        ("cu_seqlens_q_host", cu_seqlens_q_host),
        ("cu_seqlens_k_host", cu_seqlens_k_host),
    ):
        if isinstance(cu, torch.Tensor):
            if cu.device.type != "cpu" or cu.dim() != 1:
                raise ValueError(
                    f"{name} must be a 1-D host tensor or a sequence of ints, got {tuple(cu.shape)} on {cu.device}"
                )
            cu = cu.tolist()
        lists.append([int(x) for x in cu])
    return lists[0], lists[1]


def _host_seqlens(
    cu_seqlens_q_host: Optional[HostBoundaries],
    cu_seqlens_k_host: Optional[HostBoundaries],
    total_q: int,
    total_k: int,
    num_segments: int,
) -> Optional[tuple[list[int], list[int]]]:
    """``(q_lens, k_lens)`` of the host boundaries (:func:`.cake_jit.host_seqlens`), or ``None`` when neither is
    given."""
    lists = _host_lists(cu_seqlens_q_host, cu_seqlens_k_host)
    if lists is None:
        return None
    return cake_jit.host_seqlens(lists[0], lists[1], total_q, total_k, num_segments)


def cake_gqa_nondet_train_workspace_bytes(
    total_q: int,
    num_heads_q: int,
    num_segments: int,
    *,
    total_k: int = 0,
    num_heads_kv: int = 0,
    max_seqlen_k: Optional[int] = None,
    row_map: bool = False,
    budget_bytes: Optional[int] = None,
    cu_seqlens_q_host: Optional[HostBoundaries] = None,
    cu_seqlens_k_host: Optional[HostBoundaries] = None,
) -> int:
    """Bytes of device memory one backward call of these sizes needs beside its operands and outputs: the scratch a
    workspace keeps (FP32 ``dq_accum [H_q / 2, rows_pad, 256]``, ``lse_log2`` and ``dpsum [H_q / 2, rows_pad]``,
    ``rows_pad = round_up(2 T_q + 128 B, 128)``), with ``total_k`` / ``num_heads_kv`` given plus the dS^T chunk of
    ``budget_bytes`` (:func:`.cake_jit.chunk_plan`; default 16 GiB; exact with the host boundaries
    ``cu_seqlens_q_host`` / ``cu_seqlens_k_host``, else a bound) and with ``row_map`` plus the two FP32 window
    gradients ``[T_k, H_kv, 256]``; the call allocates those two from PyTorch's caching allocator and frees them when
    it returns."""
    # The host boundaries come both or neither, at every size.
    host = _host_lists(cu_seqlens_q_host, cu_seqlens_k_host)
    need = cake_jit.scratch_bytes(int(total_q), int(num_heads_q), int(num_segments))
    if total_q and total_k and num_heads_kv:
        plan = cake_jit.chunk_plan(
            total_q=int(total_q),
            total_k=int(total_k),
            num_heads_q=int(num_heads_q),
            num_heads_kv=int(num_heads_kv),
            num_segments=int(num_segments),
            max_seqlen_k=max_seqlen_k,
            budget_bytes=_budget(budget_bytes),
            seqlens=None
            if host is None
            else cake_jit.host_seqlens(
                *host, int(total_q), int(total_k), int(num_segments)
            ),
        )
        need += plan["ws_tiles"] * cake_jit.DS_TILE_BYTES
        if row_map:
            need += 2 * 4 * int(total_k) * int(num_heads_kv) * HEAD_DIM
    return need


class CakeGQANondetWorkspace:
    """Device scratch of the GQA training attention, reused across calls and shapes.

    It holds the forward's tile counter (one ``uint32``, zeroed once; every launch leaves it zero) and the backward
    scratch (the FP32 dQ accumulator, ``LSE * log2(e)`` and ``dPsum``), grown to the largest shape seen and never
    shrunk.  ``budget_bytes`` bounds the dS^T workspace of one backward call (default
    :data:`.cake_jit.DEFAULT_WORKSPACE_BUDGET`, 16 GiB): a call whose dS^T exceeds it runs in chunks (KV heads first,
    then key-block ranges).  The dS^T workspace and the row-map window gradients are allocated per call from
    PyTorch's caching allocator on the current stream and freed when the call returns; the workspace does not hold
    them.  The chunks depend on the shape, ``max_seqlen_k``, the budget and the host boundaries a call passes (given
    or not) only, so one budget gives the same bits on every run; a different budget may round dQ differently.  One workspace serves one stream at a time: two
    launches running concurrently on different streams need two workspaces.  Autograd keeps only a weak reference
    to it, so a checkpointed forward retains nothing through it.
    """

    def __init__(
        self,
        device: Union[torch.device, str, int, None] = None,
        *,
        budget_bytes: Optional[int] = None,
    ) -> None:
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
        self.budget_bytes = _budget(budget_bytes)
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


def _ds_tiles(tiles: int, device: torch.device) -> torch.Tensor:
    """The dS^T workspace of one call as BF16 ``[tiles * 256, 64]`` (``tiles`` 32 KiB tiles; the last one, the
    all-zero tile the GEMMs read past a segment's tiles, zeroed here), from the caching allocator on the current
    stream."""
    lines = tiles * cake_jit.DS_TILE_LINES
    ds = torch.empty(lines, 64, dtype=torch.bfloat16, device=device)
    ds[lines - cake_jit.DS_TILE_LINES :].zero_()
    return ds


_DEFAULT_WORKSPACES: dict[tuple[int, int], CakeGQANondetWorkspace] = {}


def default_workspace(device_index: int) -> CakeGQANondetWorkspace:
    """The workspace of the device's current stream, made on first use and kept for the process."""
    stream = torch.cuda.current_stream(device_index).stream_id
    found = _DEFAULT_WORKSPACES.get((device_index, stream))
    if found is None:
        found = _DEFAULT_WORKSPACES[(device_index, stream)] = CakeGQANondetWorkspace(
            device_index
        )
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


def cake_gqa_nondet_train_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    workspace: Optional[CakeGQANondetWorkspace] = None,
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
    # The kernel stores ``out`` in 32-byte pieces: a caller's ``out`` off a 32-byte boundary gets an aligned buffer,
    # copied in after the launch.
    dst = out if out.data_ptr() % 32 == 0 else torch.empty_like(out)
    values = {
        "Q": q,
        "K": k,
        "V": v,
        "O": dst,
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
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        _launch("fwd", values, dims, device_index)
    if dst is not out:
        out.copy_(dst)
    return out, lse


def cake_gqa_nondet_train_backward(
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
    workspace: Optional[CakeGQANondetWorkspace] = None,
    cu_seqlens_q_host: Optional[HostBoundaries] = None,
    cu_seqlens_k_host: Optional[HostBoundaries] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Backward: ``(dq, dk, dv)``, bitwise reproducible run to run.  ``dq`` is BF16 ``[T_q, H_q, 256]``.  dK / dV
    are summed over the query heads of each key head in FP32 and leave the kernels as:

    - ``dkv_dtype=torch.bfloat16`` (default): BF16 ``[T_k, H_kv, 256]``, rounded once;
    - ``dkv_dtype=torch.float32``: FP32 window gradients ``[T_k, H_kv, 256]``, stored (``dk`` / ``dv`` overwritten
      when given);
    - ``row_map`` given: FP32 adds into the caller's ``dk_acc`` / ``dv_acc`` ``[N_dst, H_kv, 256]``: window row ``j``
      adds into destination row ``row_map[j]`` (int32 ``[T_k]``, rows may repeat; out-of-range rows are not
      checked); a destination row becomes ``((dst + w_j1) + w_j2) + ...`` over its window rows in ascending ``j``,
      and the destinations are returned as ``dk`` / ``dv``.

    ``max_seqlen_k`` (host int, optional) must be at least every segment's key count (the caller's contract, as
    FlashAttention's ``max_seqlen_k``: a smaller value leaves key blocks out); it sizes the backward grids and the
    workspace chunks.  ``None`` bounds it by ``T_k``: always correct, with more (mostly empty) chunks.  It is never
    read from the device.

    ``cu_seqlens_q_host`` / ``cu_seqlens_k_host`` (optional, together): host copies of the boundaries (sequences of
    ints or 1-D CPU tensors; never copied to the device).  With them the workspace chunks follow each key block's
    exact dS^T tile count (:func:`.cake_jit.exact_tiles`): fewer, fuller chunks and a dS^T workspace of the exact
    size.  The key blocks come from the longest host key segment.  The kernels still read the device boundaries:
    host boundaries that disagree only move the chunks (and the dQ rounding) as long as every device key segment fits
    the key blocks and every chunk holds its device tiles; otherwise a device-side assert fires before the main kernel
    runs (reported at the next synchronization; the CUDA context is unusable after it).

    Every sum has one fixed order: dS^T goes to the workspace, dK and dV are summed in tensor memory by the CTA pair
    that owns their key block, dQ is a GEMM over the workspace with each output tile summed in one order by one CTA
    pair, and the row-map destinations are folded in window order.  The bits depend on the shape, ``max_seqlen_k``,
    the workspace's ``budget_bytes`` and the host boundaries (given or not, and their values) only; no PyTorch
    determinism alert is raised.
    """
    dims = _check_operands(q, k, v, cu_seqlens_q, cu_seqlens_k)
    if max_seqlen_k is not None and int(max_seqlen_k) < 0:
        raise ValueError(f"max_seqlen_k must be >= 0, got {max_seqlen_k}")
    total_q, total_k = dims["total_q"], dims["total_k"]
    heads_q, heads_kv = dims["num_heads_q"], dims["num_heads_kv"]
    seqlens = _host_seqlens(
        cu_seqlens_q_host, cu_seqlens_k_host, total_q, total_k, dims["num_segments"]
    )
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
        if dk is not None:
            _check_out("dk", dk, kv_shape, dkv_dtype, device)
        if dv is not None:
            _check_out("dv", dv, kv_shape, dkv_dtype, device)
    if dq is not None:
        _check_out("dq", dq, (total_q, heads_q, HEAD_DIM), torch.bfloat16, device)
    if mode != "rowmap":
        if dk is None:
            dk = torch.empty(kv_shape, dtype=dkv_dtype, device=device)
        if dv is None:
            dv = torch.empty(kv_shape, dtype=dkv_dtype, device=device)
        dk_out, dv_out = dk, dv
    if dq is None:
        dq = torch.empty(
            total_q, heads_q, HEAD_DIM, dtype=torch.bfloat16, device=device
        )
    if total_q == 0 or total_k == 0:
        # No (query, key) pair: every gradient is zero; a row-map destination keeps its contents.
        dq.zero_()
        if mode != "rowmap":
            dk_out.zero_()
            dv_out.zero_()
        return dq, dk_out, dv_out
    # The dQ GEMM stores ``dq`` in 32-byte pieces: a caller's ``dq`` off a 32-byte boundary gets an aligned buffer,
    # copied in after the launches.
    dst = dq if dq.data_ptr() % 32 == 0 else torch.empty_like(dq)
    device_index = device.index
    ws = workspace if workspace is not None else default_workspace(device_index)
    segments = dims["num_segments"]
    dq_accum, lse_log2, dpsum = ws.scratch(total_q, heads_q, segments)
    scale = _scale(softmax_scale)
    rows = cake_jit.rows_pad(total_q, segments)
    group = heads_q // heads_kv
    plan = cake_jit.chunk_plan(
        total_q=total_q,
        total_k=total_k,
        num_heads_q=heads_q,
        num_heads_kv=heads_kv,
        num_segments=segments,
        max_seqlen_k=None if max_seqlen_k is None else int(max_seqlen_k),
        budget_bytes=ws.budget_bytes,
        seqlens=seqlens,
    )
    common = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "num_segments": segments,
        "rows_pad": rows,
        "T_q": total_q,
        "H_q": heads_q,
        "softmax_scale": scale,
    }
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        # The tile tables, the host-boundary checks and the fold order read cu_seqlens and row_map only.  Their buffers
        # come from the current stream; side streams fill them while the current stream runs bwd_pre, and the current
        # stream waits for them before their first read: the same kernels write the same integers, off the critical
        # path.  One side stream runs the plan kernel and then the fold order, which only bwd_fold reads.  With host
        # boundaries the checks join: the range check follows the plan kernel, and a second side stream runs the key
        # check and then the fold order.  Both checks are short chains of small kernels that run while bwd_pre does,
        # and the main kernels start after both.  The host queues the work in the order the current stream needs it
        # (its time before the first main kernel is on the critical path): the plan kernel, bwd_pre, the checks, the
        # main kernels, and last the fold order.  Past PLAN_MAX_PAIRS there is no plan kernel: the PyTorch ops of
        # _tile_plan and then the checks run on the current stream after bwd_pre, and the fold order alone on the
        # first side stream.
        stream = torch.cuda.current_stream(device_index)
        tables = _plan_buffer(plan["key_blocks"], total_q, segments, device)
        fold = _fold_buffers(total_k, device) if mode == "rowmap" else None
        checked = tables is not None and seqlens is not None
        planned = keyed = side = side_keys = None
        try:
            if tables is not None or fold is not None:
                side = _side_stream(device_index, stream.stream_id, 0)
                side.wait_stream(stream)
            if tables is not None:
                with torch.cuda.stream(side), tvm_ffi.use_torch_stream():
                    _launch_plan(
                        cu_seqlens_q,
                        cu_seqlens_k,
                        plan["key_blocks"],
                        total_q,
                        tables,
                        dims,
                        device_index,
                    )
                    if seqlens is None:
                        planned = side.record_event()
            if checked:
                side_keys = _side_stream(device_index, stream.stream_id, 1)
                side_keys.wait_stream(stream)
            _launch(
                "bwd_pre",
                {
                    **common,
                    "O": out,
                    "dO": dout,
                    "LSE": lse,
                    "LSE_log2": lse_log2,
                    "dPsum": dpsum,
                },
                dims,
                device_index,
            )
            if checked:
                with torch.cuda.stream(side):
                    _check_ranges(tables[0], plan, segments, total_q)
                    planned = side.record_event()
                with torch.cuda.stream(side_keys):
                    _check_keys(cu_seqlens_k, plan)
                    keyed = side_keys.record_event()
            if tables is None:
                tables = _tile_plan(
                    cu_seqlens_q, cu_seqlens_k, plan["key_blocks"], total_q
                )
                if seqlens is not None:
                    _check_ranges(tables[0], plan, segments, total_q)
                    _check_keys(cu_seqlens_k, plan)
            tile_base, pair_base, pair_order = tables
            # Per call, from the caching allocator on the current stream: the dS^T chunk and (row-map form) the FP32 window
            # gradients, which the main kernel stores in full (zero rows included) before the fold reads them.
            ds = _ds_tiles(plan["ws_tiles"], device)
            if mode == "rowmap":
                win_k = torch.empty(kv_shape, dtype=torch.float32, device=device)
                win_v = torch.empty(kv_shape, dtype=torch.float32, device=device)
            else:
                win_k, win_v = dk_out, dv_out
            values = {
                **common,
                "Q": q,
                "dO": dout,
                "K": k,
                "V": v,
                "lse_log2": lse_log2,
                "dpsum": dpsum,
                "ds_ws": ds.view(-1),
                "dsw": ds,
                "tile_base": tile_base,
                "pair_base": pair_base,
                "pair_order": pair_order,
                "dq_accum": dq_accum,
                "dQ": dst,
                "dk_out": win_k,
                "dv_out": win_v,
                "row_map": None,
                "h_kv": heads_kv,
                "group": group,
                "softmax_scale_log2": scale * LOG2_E,
                "tile_stride": plan["tile_stride"],
                "zero_tile": plan["zero_tile"],
            }
            h_chunk = plan["h_chunk"]
            for event in (planned, keyed):
                if event is not None:
                    stream.wait_event(event)
            for h0 in range(0, heads_kv, h_chunk):
                hc = min(h_chunk, heads_kv - h0)
                for index, (n0, n1) in enumerate(plan["ranges"]):
                    chunk = {"chunk_blocks": n1 - n0, "chunk_heads": hc}
                    at = {
                        **values,
                        "n0": n0,
                        "n1": n1,
                        "h0": h0,
                        "hc": hc,
                        "accumulate": 1 if index else 0,
                        # The last range finishes every Q tile left: its dQ GEMM writes BF16 dQ for all of them; an
                        # earlier range writes it for the tiles whose key range it ends.
                        "finalize": 2 if index == len(plan["ranges"]) - 1 else 1,
                    }
                    _launch(_MAIN_STAGE[mode], at, dims, device_index, **chunk)
                    _launch("bwd_dq", at, dims, device_index, **chunk)
            if fold is not None:
                with torch.cuda.stream(side if side_keys is None else side_keys):
                    _fold_index(row_map, *fold)
        finally:
            # Joined on every path (an exception included): the side streams read and write buffers of this call.
            for other in (side, side_keys):
                if other is not None:
                    stream.wait_stream(other)
        if fold is not None:
            dst_rows, _order, perm, run_end = fold
            _launch(
                "bwd_fold",
                {
                    "dk_win": win_k,
                    "dv_win": win_v,
                    "perm": perm,
                    "dst_rows": dst_rows,
                    "run_end": run_end,
                    "dk_out": dk_out,
                    "dv_out": dv_out,
                    "T_k": total_k,
                    "h_kv": heads_kv,
                },
                dims,
                device_index,
            )
        if dst is not dq:
            dq.copy_(dst)
    return dq, dk_out, dv_out


def _plan_buffer(
    key_blocks: int, total_q: int, segments: int, device: torch.device
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """``(tile_base, pair_base, pair_order)`` views of one int32 buffer for :func:`_launch_plan`, from the caching
    allocator on the current stream; ``None`` past :data:`.cake_jit.PLAN_MAX_PAIRS` pair positions (the PyTorch ops
    of :func:`_tile_plan` build those tables)."""
    bound = cake_jit.pair_bound(total_q, segments)
    if bound > cake_jit.PLAN_MAX_PAIRS:
        return None
    tiles = (key_blocks + 1) * segments
    tables = torch.empty(tiles + segments + 1 + bound, dtype=torch.int32, device=device)
    return (
        tables[:tiles],
        tables[tiles : tiles + segments + 1],
        tables[tiles + segments + 1 :],
    )


def _launch_plan(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    key_blocks: int,
    total_q: int,
    tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    dims: Mapping[str, int],
    device_index: int,
) -> None:
    """One ``bwd_plan`` launch (one CTA) writing the integers of :func:`_tile_plan` into ``tables``."""
    segments = dims["num_segments"]
    tile_base, pair_base, pair_order = tables
    _launch(
        "bwd_plan",
        {
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "tile_base": tile_base,
            "pair_base": pair_base,
            "pair_order": pair_order,
            "num_segments": segments,
            "key_blocks": key_blocks,
            "pair_bound": cake_jit.pair_bound(total_q, segments),
        },
        dims,
        device_index,
    )


def _check_ranges(
    tile_base: torch.Tensor,
    plan: Mapping[str, Any],
    segments: int,
    total_q: int,
) -> None:
    """Queue a device-side assert, on the current stream, that every key-block range of a host-boundary plan holds at
    most ``tile_stride`` dS^T tiles of one head pair in the device tile table (``tile_base[n * B]`` is the count before
    key block ``n``).  Past it the main kernel would write beyond its chunk: the assert fires first, and the process's
    CUDA context is unusable after it (PyTorch reports it at the next synchronization).  The counts come from
    :func:`_range_marks`, exact past 2^31 tiles: a range past ``tile_stride`` fails the check instead of wrapping
    into it.  With :func:`_check_keys`, host boundaries within both bounds give the device boundaries' gradients (only
    the chunks, and with them the dQ rounding, may differ)."""
    ranges = plan["ranges"]
    marks = _range_marks(tile_base, segments, total_q, plan["key_blocks"])
    if len(ranges) == 1:  # one range: one subtraction (fewer kernels in the chain)
        fits = marks[ranges[0][1]] - marks[ranges[0][0]] <= plan["tile_stride"]
    else:
        edges = torch.stack([marks[n0] for n0, _ in ranges] + [marks[ranges[-1][1]]])
        fits = torch.diff(edges).max() <= plan["tile_stride"]
    torch._assert_async(fits, _DISAGREE)


def _check_keys(cu_seqlens_k: torch.Tensor, plan: Mapping[str, Any]) -> None:
    """Queue a device-side assert, on the current stream, that every device segment's keys lie within a host-boundary
    plan's ``key_blocks``.  Past them the main kernels would leave key blocks out and the dQ GEMM would read tile
    counts past the table; the assert fires first.  It reads ``cu_seqlens_k`` only, so with the plan kernel's tables
    it runs on its own side stream, beside the plan kernel and :func:`_check_ranges`."""
    keys = torch.diff(cu_seqlens_k).max() <= cake_jit.BWD_TILE_KEYS * plan["key_blocks"]
    torch._assert_async(keys, _DISAGREE)


_DISAGREE = "cake_gqa_nondet_train: cu_seqlens_q_host / cu_seqlens_k_host disagree with the device boundaries"


def _range_marks(
    tile_base: torch.Tensor, segments: int, total_q: int, key_blocks: int
) -> torch.Tensor:
    """The tile count before each key block ``n <= key_blocks`` (``tile_base[n * B]``), exact.  ``tile_base`` is int32
    and wraps modulo 2^32 past 2^31 tiles (both table builders).  One key block holds at most
    :func:`.cake_jit.tiles_per_key_block` tiles (below 2^32), so when ``key_blocks + 1`` such blocks can reach 2^31
    the marks are rebuilt in int64 from their differences modulo 2^32; below that they are the int32 entries."""
    marks = tile_base[::segments]
    if (key_blocks + 1) * cake_jit.tiles_per_key_block(total_q, segments) < 1 << 31:
        return marks
    counts = torch.diff(marks.to(torch.int64)).remainder_(1 << 32)
    return torch.cat([counts.new_zeros(1), counts.cumsum(0)])


def _plan_tables(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    key_blocks: int,
    total_q: int,
    dims: Mapping[str, int],
    device_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The tables of :func:`_tile_plan` on the current stream of device ``device_index``: from one ``bwd_plan``
    launch, or from the PyTorch ops of :func:`_tile_plan` past :data:`.cake_jit.PLAN_MAX_PAIRS` pair positions.
    Both write the same integers."""
    tables = _plan_buffer(
        key_blocks, total_q, dims["num_segments"], cu_seqlens_q.device
    )
    if tables is None:
        return _tile_plan(cu_seqlens_q, cu_seqlens_k, key_blocks, total_q)
    with torch.cuda.device(device_index), tvm_ffi.use_torch_stream():
        _launch_plan(
            cu_seqlens_q, cu_seqlens_k, key_blocks, total_q, tables, dims, device_index
        )
    return tables


def _fold_buffers(total_k: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    """``(dst_rows, order, perm, run_end)`` for :func:`_fold_index`, from the caching allocator on the current
    stream."""
    return (
        torch.empty(total_k, dtype=torch.int32, device=device),
        torch.empty(total_k, dtype=torch.int64, device=device),
        torch.empty(total_k, dtype=torch.int32, device=device),
        torch.empty(total_k, dtype=torch.int32, device=device),
    )


def _fold_index(
    row_map: torch.Tensor,
    dst_rows: torch.Tensor,
    order: torch.Tensor,
    perm: torch.Tensor,
    run_end: torch.Tensor,
) -> None:
    """The fold order: the window rows sorted by destination row with ties in window order (``dst_rows``, ``perm``
    as int32), and ``run_end[j]`` = one past the last sorted position of ``dst_rows[j]``; each destination row adds
    its run in order."""
    torch.sort(row_map, stable=True, out=(dst_rows, order))
    torch.searchsorted(dst_rows, dst_rows, right=True, out_int32=True, out=run_end)
    perm.copy_(order)


@functools.cache
def _side_stream(device_index: int, stream_id: int, slot: int) -> torch.cuda.Stream:
    """High-priority side stream ``slot`` of the caller stream ``stream_id`` of device ``device_index``: slot 0 runs
    the backward's tile tables (the one-CTA ``bwd_plan``) and, with host boundaries, the range check; slot 1 the
    host-boundary key check.  The row-map fold order runs on slot 1 when it exists (host boundaries with the plan
    kernel's tables), else on slot 0.  One cached stream per caller stream and slot, from PyTorch's high-priority
    stream pool (a pool that cycles, so two caller streams may share one stream past its size: their work then queues
    on it in order); high priority: their small kernels take the first SMs bwd_pre and the main kernels free."""
    return torch.cuda.Stream(device_index, priority=-1)


def _tile_plan(
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    key_blocks: int,
    total_q: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The workspace tile numbering on the device, from the segment boundaries (no host read):

    - ``tile_base`` int32 ``[(key_blocks + 1) * B]``: the exclusive prefix sum, in (key block ``n``, segment ``b``)
      order, of ``cnt(b, n) = max(0, ceil(q_b / 64) - m_min(b, n))`` for ``n < ceil(k_b / 128)`` (else 0), ``m_min(b,
      n) = max(128 n - (k_b - q_b), 0) // 64`` the first Q tile key block ``n`` sees;
    - ``pair_base`` int32 ``[B + 1]``: the exclusive prefix sum of the segments' Q-tile pairs ``ceil(ceil(q_b / 64) /
      2)``, the total last;
    - ``pair_order`` int32 ``[ceil(total_q / 128) + B]`` (a bound of the pairs): the dQ GEMM's work order, the pairs by
      the key blocks their later Q tile sees, most first (stable), the positions past the total last.  The order only
      balances the work; it does not change any result bit.
    """
    cq = cu_seqlens_q.to(torch.int64)
    ck = cu_seqlens_k.to(torch.int64)
    q_len = cq[1:] - cq[:-1]
    k_len = ck[1:] - ck[:-1]
    n = torch.arange(key_blocks + 1, device=cq.device, dtype=torch.int64)[:, None]
    m_min = (
        torch.clamp(n * cake_jit.BWD_TILE_KEYS - (k_len - q_len)[None, :], min=0)
        // cake_jit.BWD_TILE_ROWS
    )
    tiles = (q_len + cake_jit.BWD_TILE_ROWS - 1) // cake_jit.BWD_TILE_ROWS
    blocks = (k_len + cake_jit.BWD_TILE_KEYS - 1) // cake_jit.BWD_TILE_KEYS
    cnt = torch.where(
        n < blocks[None, :], torch.clamp(tiles[None, :] - m_min, min=0), 0
    ).reshape(-1)
    tile_base = (torch.cumsum(cnt, 0) - cnt).to(torch.int32)
    pair_base = torch.zeros(q_len.numel() + 1, dtype=torch.int64, device=cq.device)
    torch.cumsum((tiles + 1) // 2, 0, out=pair_base[1:])
    pair = torch.arange(
        (total_q + cake_jit.BWD_TILE_KEYS - 1) // cake_jit.BWD_TILE_KEYS
        + q_len.numel(),
        device=cq.device,
    )
    seg = torch.searchsorted(pair_base[1:].contiguous(), pair, right=True).clamp(
        max=q_len.numel() - 1
    )
    last = torch.minimum(2 * (pair - pair_base[seg]) + 2, tiles[seg])
    seen = (
        cake_jit.BWD_TILE_ROWS * last
        + (k_len - q_len)[seg]
        + cake_jit.BWD_TILE_KEYS
        - 1
    ) // cake_jit.BWD_TILE_KEYS
    seen = torch.minimum(seen.clamp(min=0), blocks[seg])
    seen = torch.where(pair < pair_base[-1], seen, -1)
    pair_order = torch.sort(-seen, stable=True).indices.to(torch.int32)
    return tile_base, pair_base.to(torch.int32), pair_order


# ----------------------------------------------------------------------------------------------------------------
# Autograd

# ``dkv_acc``: a zero-argument callable, called in backward, returning ``(dk_acc, dv_acc, row_map)`` (FP32 adds into
# the caller's destinations through the row map) or ``(dk_win, dv_win)`` (FP32 window gradients stored into them).
# Under activation checkpointing a destination passed at forward time would live for every layer at once; one the
# callable produces lives only while its backward runs.
CakeGQANondetDkvAccSource = Callable[
    [],
    Union[
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ],
]


def _resolve_dkv_acc(
    dkv_acc: CakeGQANondetDkvAccSource,
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


class CakeGQANondetFunction(torch.autograd.Function):
    """``(out, lse) = attention(q, k, v)``; the saved set is ``(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)``
    through ``ctx.save_for_backward`` alone, with scalars, the host boundaries as lists, the ``dkv_acc`` callable
    and a weak reference to the workspace on ``ctx`` (a checkpointed forward keeps no tensor alive through ``ctx``)."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        softmax_scale: float,
        dkv_acc: Optional[CakeGQANondetDkvAccSource],
        max_seqlen_k: Optional[int],
        workspace: Optional[CakeGQANondetWorkspace],
        host_boundaries: Optional[tuple[list[int], list[int]]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out, lse = cake_gqa_nondet_train_forward(
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
        ctx.host_boundaries = host_boundaries
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
        if ctx.host_boundaries is not None:
            kwargs["cu_seqlens_q_host"], kwargs["cu_seqlens_k_host"] = (
                ctx.host_boundaries
            )
        # One gradient per forward argument: nine or ten (``host_boundaries`` may be left out).
        rest = (None,) * (len(ctx.needs_input_grad) - 3)
        if ctx.dkv_acc is None:
            dq, dk, dv = cake_gqa_nondet_train_backward(
                q, k, v, out, lse, dout, cu_seqlens_q, cu_seqlens_k, **kwargs
            )
            return (dq, dk, dv, *rest)
        dk_t, dv_t, row_map = _resolve_dkv_acc(ctx.dkv_acc)
        if row_map is None:
            kwargs.update(dk=dk_t, dv=dv_t, dkv_dtype=torch.float32)
        else:
            kwargs.update(dk_acc=dk_t, dv_acc=dv_t, row_map=row_map)
        dq = cake_gqa_nondet_train_backward(
            q, k, v, out, lse, dout, cu_seqlens_q, cu_seqlens_k, **kwargs
        )[0]
        return (dq, None, None, *rest)


def cake_gqa_nondet(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    causal: bool = True,
    dkv_acc: Optional[CakeGQANondetDkvAccSource] = None,
    max_seqlen_k: Optional[int] = None,
    workspace: Optional[CakeGQANondetWorkspace] = None,
    return_lse: bool = False,
    cu_seqlens_q_host: Optional[HostBoundaries] = None,
    cu_seqlens_k_host: Optional[HostBoundaries] = None,
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Packed-varlen causal GQA attention with autograd (see :func:`cake_gqa_nondet_train_forward` /
    :func:`cake_gqa_nondet_train_backward`).

    ``dkv_acc=None``: autograd returns BF16 gradients for ``k`` / ``v``.  A ``dkv_acc`` callable is called in
    backward; it returns ``(dk_acc, dv_acc, row_map)`` (FP32 adds into the destinations) or ``(dk_win, dv_win)`` (FP32
    window gradients stored into them), and ``k`` / ``v`` then get no gradient.  ``max_seqlen_k`` and the host
    boundaries ``cu_seqlens_q_host`` / ``cu_seqlens_k_host`` (checked here, kept as lists for the backward): see
    :func:`cake_gqa_nondet_train_backward`.  Both halves are bitwise reproducible run to run and raise no PyTorch
    determinism alert.
    """
    if not causal:
        raise NotImplementedError(
            "only causal attention (bottom-right aligned per segment) is implemented"
        )
    if dkv_acc is not None and not callable(dkv_acc):
        raise TypeError(
            "dkv_acc must be None or a zero-argument callable (see CakeGQANondetDkvAccSource)"
        )
    host = _host_lists(cu_seqlens_q_host, cu_seqlens_k_host)
    if host is not None:
        cake_jit.host_seqlens(
            host[0],
            host[1],
            int(q.shape[0]),
            int(k.shape[0]),
            int(cu_seqlens_q.shape[0]) - 1,
        )
    out, lse = CakeGQANondetFunction.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        _scale(softmax_scale),
        dkv_acc,
        None if max_seqlen_k is None else int(max_seqlen_k),
        workspace,
        host,
    )
    return (out, lse) if return_lse else out
