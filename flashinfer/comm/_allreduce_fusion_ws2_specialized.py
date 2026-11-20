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

Specialized two-rank all-reduce + residual add + RMSNorm on compute capability 10.3 (B300 /
GB300), hidden size 6144, bf16: a kernel of the ``"specialized"`` backend of
:func:`flashinfer.comm.allreduce_fusion`.

Private implementation module, not public API. Its callers are
:func:`flashinfer.comm.create_allreduce_fusion_workspace` (``backend="specialized"``, and
``backend="auto"``, which tries it first), :class:`flashinfer.comm.SpecializedAllReduceFusionWorkspace`
(checkpoint, destroy) and :func:`flashinfer.comm.allreduce_fusion`, which calls :func:`try_dispatch`
for ``pattern=kARResidualRMSNorm`` on a specialized workspace and runs the workspace's fallback
backend whenever that returns ``None``.

State (workspace creation). :func:`creation_envelope_reason` is the static part of the envelope:
world size 2, hidden size 6144, bf16, symmetric device memory (no Confidential Computing) and a
``max_token_num`` that covers the smallest allowlisted token count. It reads only arguments every
rank passes identically, so no collective runs for any other configuration. For the envelope,
:func:`create_state` adds one peer-mapped region (``SymmDeviceMemory``, unicast, no signal pads), a
rank-local device call counter and a launch grid. Before allocating anything the ranks vote with
one ``allgather``: every rank must be on compute capability 10.3, on one node, have the JIT module
built, a non-empty allowlist, enough free memory and ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE``
unset. One rank's veto leaves the state absent on every rank, and the reason is returned on every
rank alike. Creation never raises.

Dispatch (call time). :func:`allreduce_fusion` first asks :func:`outside_token_ranges`, a few
cheap checks of the token count against the state's ranges, so that calls the kernel never serves
(e.g. decode sizes) reach the fallback backend without the rest of the guard; then
:func:`try_dispatch` checks, in this order (any failing check runs the fallback backend):

- inside ``torch.compile`` tracing nothing of this module is traced (fallback);
- ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` (read on every call);
- the workspace has the state above;
- ``input`` is 2-D and ``num_tokens`` lies in an allowlisted range and within the state's
  capacity (first, so that calls outside the token ranges, e.g. decode, return after a few
  checks);
- ``fp32_acc=True`` and an allowlisted ``weight_bias``;
- ``residual_in``, ``residual_out``, ``norm_out`` and ``rms_gamma`` given, all bf16;
- ``input`` of shape ``[num_tokens, 6144]``, residual/norm tensors of the same shape, ``rms_gamma``
  of shape ``[6144]``, all contiguous;
- all tensors on the workspace's device.

``use_oneshot``, ``trigger_completion_at_end``, ``output``, ``layout_code`` and the quantization
arguments do not change the result of this pattern and are ignored. ``residual_out`` and/or
``norm_out`` may be the same tensor as ``input`` or ``residual_in``.

Rank symmetry. The call is collective: if one rank dispatched and its peer did not, the dispatched
rank would wait forever for the peer's data. Every check above depends only on values that are
identical on both ranks of a tensor-parallel group (shapes, dtypes, layouts, arguments, the state
the ranks agreed on, the environment), so the environment variable must be set identically on
every rank. Nothing per-rank (pointer values, tensor contents) decides the path; a launch error
raises instead of falling back.

The kernel waits for its predecessor before reading anything when launched with programmatic
dependent launch (``launch_with_pdl=True``) and never triggers its dependents early. It launches on
the current stream with no host synchronisation or allocation, so it is CUDA-graph capturable; the
module is built at workspace creation, never at call time.

Markers, once per process each, written to ``sys.stderr`` (serving frameworks may route or drop
library loggers) and to the ``flashinfer.jit`` logger: :data:`STATE_MARKER` when a state is created,
:data:`DISPATCH_MARKER` on the first dispatch, :data:`KILL_SWITCH_MARKER` on the first
kill-switched call of a workspace with a state, ``<FALLBACK_MARKER>: <reason>`` on the first
fallback for each reason of a workspace with a state, and ``<NO_STATE_MARKER>: <reason>`` when the
vote rejects the state on a compute capability 10.3 device. Everything else is logged at DEBUG
level only. :func:`_allreduce_fusion_ws2_specialized_stats` returns the counters.
"""

import contextlib
import functools
import json
import os
import socket
import sys
import threading
from dataclasses import dataclass
from importlib import resources
from typing import Any, Dict, Optional, Set, Tuple

import torch

from ..jit.allreduce_fusion_ws2_specialized import (
    ALLREDUCE_FUSION_WS2_SPECIALIZED_MODULE_NAME,
    gen_allreduce_fusion_ws2_specialized_module,
)
from ..jit.core import logger
from ..utils import get_compute_capability
from .comm_backend import MPIBackend, TorchDistBackend
from .mnnvl import SymmDeviceMemory

# Kill switch shared by FlashInfer's specialized kernels: read at CALL time (and at workspace
# creation, where it skips the allocation). Must be set identically on every rank.
_DISABLE_ENV = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Dispatch surface (package data; see the file's note).
_WORKLOAD_FILE = "allreduce_fusion_ws2_specialized_workloads.json"
_WORLD_SIZE = 2
_HIDDEN = 6144
_DTYPE = torch.bfloat16
_DTYPE_NAME = "bfloat16"
_PATTERN = 1  # AllReduceFusionPattern.kARResidualRMSNorm
_COMPUTE_CAPABILITY = (10, 3)
# Free device memory required beyond the region before allocating it.
_FREE_MEMORY_MARGIN = 256 << 20

DISPATCH_MARKER = "allreduce_fusion: specialized SM103 two-rank AR + residual + RMSNorm kernel dispatched"
KILL_SWITCH_MARKER = (
    "allreduce_fusion: specialized SM103 two-rank AR + residual + RMSNorm kernel disabled by "
    "FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1, using the fallback backend"
)
FALLBACK_MARKER = "allreduce_fusion: specialized SM103 two-rank path fallback"
STATE_MARKER = "allreduce_fusion: specialized SM103 two-rank state created"
NO_STATE_MARKER = "allreduce_fusion: specialized SM103 two-rank state not created"


@dataclass(frozen=True)
class _Allowlist:
    weight_bias: Tuple[float, ...]
    num_tokens: Tuple[Tuple[int, int], ...]  # inclusive ranges

    @property
    def t_min(self) -> int:
        return min(lo for lo, _ in self.num_tokens)

    @property
    def t_max(self) -> int:
        return max(hi for _, hi in self.num_tokens)

    def allows_num_tokens(self, num_tokens: int) -> bool:
        return any(lo <= num_tokens <= hi for lo, hi in self.num_tokens)


@dataclass
class _State:
    """Per-workspace state; identical in shape on both ranks (same capacity and grid)."""

    handle: Any  # SymmDeviceMemory holding both ranks' regions
    own_ptr: int
    peer_ptr: int
    epoch: torch.Tensor  # rank-local int32 call counter on the workspace device
    capacity: int  # tokens the region holds per call
    grid: int  # CTAs per launch (minimum over the ranks)
    device_idx: int
    rank: int
    region_bytes: int
    # Allowlisted token-count ranges clipped to the capacity (derived when left empty).
    token_ranges: Tuple[Tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        if not self.token_ranges:
            self.token_ranges = tuple(
                (lo, min(hi, self.capacity))
                for lo, hi in _load_allowlist().num_tokens
                if lo <= self.capacity
            )


_module: Optional[Any] = None
_module_error: Optional[str] = None
_module_lock = threading.Lock()
_stats: Dict[str, Any] = {
    "calls": 0,
    "dispatch_count": 0,
    "fallbacks": 0,
    "fallback_reasons": {},
    "guard_error": None,
    "states_created": 0,
    "states_released": 0,
    "state_rejections": {},
    "last_state_rejection": None,
    "module_loaded": False,
    "module_error": None,
}
_markers_emitted: Set[str] = set()
_markers_logged_debug: Set[str] = set()
_allowlist_error: Optional[str] = None


def _emit_once(key: str, line: str) -> None:
    """Write ``line`` once per process to stderr and to the ``flashinfer.jit`` logger (INFO)."""
    if key in _markers_emitted:
        return
    _markers_emitted.add(key)
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 - diagnostics must never break the call
        pass
    logger.info(line)


def _debug_once(key: str, line: str) -> None:
    if key in _markers_logged_debug:
        return
    _markers_logged_debug.add(key)
    logger.debug(line)


@functools.cache
def _load_allowlist() -> _Allowlist:
    """The dispatch surface; an empty token range list (never dispatch) when the file is missing
    or inconsistent with this module (the error is kept in ``_allowlist_error``)."""
    global _allowlist_error
    try:
        payload = json.loads(
            resources.files("flashinfer").joinpath(_WORKLOAD_FILE).read_text()
        )
        expected = {
            "world_size": _WORLD_SIZE,
            "hidden_dim": _HIDDEN,
            "dtype": _DTYPE_NAME,
            "pattern": _PATTERN,
            "compute_capability": list(_COMPUTE_CAPABILITY),
            "fp32_acc": True,
        }
        for key, value in expected.items():
            if payload[key] != value:
                raise ValueError(f"unexpected {key}: {payload[key]!r}")
        weight_bias = tuple(float(v) for v in payload["weight_bias"])
        ranges = tuple((int(lo), int(hi)) for lo, hi in payload["num_tokens"])
        if not ranges or any(lo < 1 or hi < lo for lo, hi in ranges):
            raise ValueError(f"invalid num_tokens ranges: {ranges}")
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        _allowlist_error = f"{type(exc).__name__}: {exc}"[:300]
        return _Allowlist(weight_bias=(), num_tokens=())
    _allowlist_error = None
    return _Allowlist(weight_bias=weight_bias, num_tokens=ranges)


def _get_module() -> Optional[Any]:
    """Build/load the JIT module once per process (only at workspace creation)."""
    global _module, _module_error
    if _module is not None:
        return _module
    with _module_lock:
        if _module is None and _module_error is None:
            try:
                _module = gen_allreduce_fusion_ws2_specialized_module().build_and_load()
                _stats["module_loaded"] = True
            except Exception as exc:  # noqa: BLE001 - a failed build leaves no state (fallback only)
                _module_error = f"{type(exc).__name__}: {exc}"[:500]
                _stats["module_error"] = _module_error
                logger.warning(
                    "allreduce_fusion: JIT build of the specialized two-rank kernel failed, "
                    "so no workspace gets its state: %s",
                    _module_error,
                )
    return _module


def _node_id() -> str:
    """Identifies the host: the kernel boot id (not namespaced by containers) and host name."""
    try:
        with open("/proc/sys/kernel/random/boot_id") as f:
            boot_id = f.read().strip()
    except OSError:
        boot_id = ""
    return f"{boot_id}@{socket.gethostname()}"


def _record_rejection(reason: str, detail: str, target_device: bool) -> None:
    _stats["state_rejections"][reason] = _stats["state_rejections"].get(reason, 0) + 1
    _stats["last_state_rejection"] = f"{reason}: {detail}"[:600]
    line = f"{NO_STATE_MARKER}: {reason} ({detail})"[:800]
    if target_device:
        _emit_once(f"no_state:{reason}", line)
    else:
        _debug_once(f"no_state:{reason}", line)


def _local_probe(
    device: torch.device, max_token_num: int, backend: Any, tp_rank: int
) -> Tuple[Optional[str], str, int, int, int, bool]:
    """This rank's vote: ``(reason or None, detail, capacity, grid, region_bytes,
    target_device)``. Never raises; the JIT module is only built when every other local check
    passes."""
    capacity = grid = region_bytes = 0
    target = False
    try:
        if os.environ.get(_DISABLE_ENV) == "1":
            return "kill_switch", f"{_DISABLE_ENV}=1", 0, 0, 0, False
        capability = get_compute_capability(device)
        target = capability == _COMPUTE_CAPABILITY
        if not target:
            return "compute_capability", f"capability={capability}", 0, 0, 0, False
        if backend.Get_size() != _WORLD_SIZE or backend.Get_rank() != tp_rank:
            detail = f"backend rank/size {backend.Get_rank()}/{backend.Get_size()}"
            return "group", detail, 0, 0, 0, target
        allowlist = _load_allowlist()
        if not allowlist.num_tokens:
            return "allowlist_unavailable", str(_allowlist_error), 0, 0, 0, target
        capacity = min(int(max_token_num), allowlist.t_max)
        if capacity < allowlist.t_min:
            detail = f"max_token_num={max_token_num} < {allowlist.t_min}"
            return "capacity", detail, 0, 0, 0, target
        module = _get_module()
        if module is None:
            return "jit_unavailable", str(_module_error), 0, 0, 0, target
        grid = int(module.allreduce_fusion_ws2_max_grid(device.index))
        if grid <= 0:
            return "occupancy", f"max_grid={grid}", 0, 0, 0, target
        region_bytes = int(module.allreduce_fusion_ws2_region_bytes(capacity))
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < region_bytes + _FREE_MEMORY_MARGIN:
            detail = (
                f"free={free_bytes} < region {region_bytes} + {_FREE_MEMORY_MARGIN}"
            )
            return "memory", detail, 0, 0, 0, target
    except Exception as exc:  # noqa: BLE001 - a probe failure is a veto, never an exception
        return "probe_error", f"{type(exc).__name__}: {exc}"[:300], 0, 0, 0, target
    return None, "", capacity, grid, region_bytes, target


def creation_envelope_reason(
    tp_size: int,
    hidden_dim: int,
    dtype: torch.dtype,
    max_token_num: Optional[int],
    use_symm_dev_mem: bool,
) -> Optional[str]:
    """Why no workspace of this configuration can hold the state, or ``None``: the static part
    of the creation-time envelope. Reads only arguments every rank passes identically and runs no
    collective, so a workspace outside it never reaches :func:`create_state`."""
    if tp_size != _WORLD_SIZE:
        return f"world_size={tp_size} (the kernel serves world size {_WORLD_SIZE})"
    if hidden_dim != _HIDDEN:
        return f"hidden_dim={hidden_dim} (the kernel serves {_HIDDEN})"
    if dtype != _DTYPE:
        return f"dtype={dtype} (the kernel serves {_DTYPE})"
    if not use_symm_dev_mem:
        # Confidential Computing (a host-wide mode) uses torch symmetric memory instead.
        return "no symmetric device memory (Confidential Computing)"
    allowlist = _load_allowlist()
    if not allowlist.num_tokens:
        return f"allowlist unavailable ({_allowlist_error})"
    if max_token_num is None or int(max_token_num) < allowlist.t_min:
        return (
            f"max_token_num={max_token_num} is below {allowlist.t_min}, the smallest token "
            "count the kernel serves"
        )
    return None


# A rejected state: (reason, detail), the same on every rank of the workspace.
Rejection = Tuple[str, str]


def create_state(
    tp_rank: int,
    max_token_num: int,
    comm_backend: Optional[Any],
    group: Optional[Any],
) -> Tuple[Optional[_State], Optional[Rejection]]:
    """The state of a workspace inside the static envelope (:func:`creation_envelope_reason`
    returned ``None`` on every rank): ``(state, None)``, or ``(None, (reason, detail))`` alike on
    every rank (``reason == "kill_switch"`` when a rank has
    ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1``). Collective over the workspace's ranks; never
    raises."""
    try:
        return _create_state(tp_rank, max_token_num, comm_backend, group)
    except Exception as exc:  # noqa: BLE001 - creation must never raise
        rejection = ("error", f"{type(exc).__name__}: {exc}"[:300])
        _record_rejection(*rejection, False)
        return None, rejection


def _vote_backend(comm_backend: Optional[Any], group: Optional[Any]) -> Any:
    """The communicator of the vote and the handle exchange: the workspace's own, else the one
    ``all_ranks_support_mnnvl`` uses (torch.distributed when initialized, else MPI)."""
    if comm_backend is not None:
        return comm_backend
    import torch.distributed as dist

    return TorchDistBackend(group=group) if dist.is_initialized() else MPIBackend()


def _create_state(
    tp_rank: int,
    max_token_num: int,
    comm_backend: Optional[Any],
    group: Optional[Any],
) -> Tuple[Optional[_State], Optional[Rejection]]:
    device_idx = torch.cuda.current_device()
    device = torch.device("cuda", device_idx)
    backend = _vote_backend(comm_backend, group)

    # 1. Local probes, then one vote: any rank's veto leaves the state absent on every rank.
    reason, detail, capacity, grid, region_bytes, target = _local_probe(
        device, max_token_num, backend, tp_rank
    )
    votes = backend.allgather((reason, detail, capacity, grid, _node_id()))
    vetoes = [(r, v[0], v[1]) for r, v in enumerate(votes) if v[0] is not None]
    rejection: Optional[Rejection] = None
    if vetoes:
        rank, why, what = vetoes[0]
        rejection = (why, f"rank {rank}: {what}")
    elif len({v[4] for v in votes}) != 1:
        rejection = ("multi_node", f"nodes={[v[4] for v in votes]}")
    if rejection is not None:
        _record_rejection(*rejection, target)
        return None, rejection
    # Same layout and grid on both ranks (they normally agree already).
    capacity = min(int(v[2]) for v in votes)
    grid = min(int(v[3]) for v in votes)
    module = _module
    assert module is not None
    region_bytes = int(module.allreduce_fusion_ws2_region_bytes(capacity))

    # 2. Allocation. The handle exchange gathers every rank's allocation error, so a failure
    #    raises on every rank alike.
    try:
        handle = SymmDeviceMemory(
            buf_size=region_bytes,
            group_size=_WORLD_SIZE,
            group_rank=tp_rank,
            device_idx=device_idx,
            comm_backend_for_handle_transfer=backend,
            enable_multicast=False,
            allocate_signal_pads=False,
        )
    except Exception as exc:  # noqa: BLE001
        rejection = ("allocation", f"{type(exc).__name__}: {exc}"[:300])
        _record_rejection(*rejection, target)
        return None, rejection

    # 3. Idle state on this rank, then a second vote that doubles as the barrier: both ranks'
    #    regions hold the sentinel before either rank can push into its peer's.
    state: Optional[_State] = None
    error: Optional[str] = None
    try:
        ptrs = handle.get_buffer_ptrs_host()
        own_ptr, peer_ptr = int(ptrs[tp_rank]), int(ptrs[1 - tp_rank])
        words = int(module.allreduce_fusion_ws2_epoch_words(grid))
        epoch = torch.zeros(words, dtype=torch.int32, device=device)
        module.allreduce_fusion_ws2_init(epoch, own_ptr, capacity)
        torch.cuda.synchronize(device)
        state = _State(
            handle=handle,
            own_ptr=own_ptr,
            peer_ptr=peer_ptr,
            epoch=epoch,
            capacity=capacity,
            grid=grid,
            device_idx=device_idx,
            rank=tp_rank,
            region_bytes=region_bytes,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"[:300]
    errors = backend.allgather(error)
    failed = [(r, e) for r, e in enumerate(errors) if e is not None]
    if failed:
        state = None
        del handle
        rejection = ("init", f"rank {failed[0][0]}: {failed[0][1]}")
        _record_rejection(*rejection, target)
        return None, rejection
    assert state is not None
    _stats["states_created"] += 1
    _emit_once(
        "state",
        f"{STATE_MARKER} (rank={tp_rank}, device=cuda:{device_idx}, capacity={capacity}, "
        f"grid={grid}, region_bytes={region_bytes})",
    )
    return state, None


def checkpoint_prepare_state(workspace: Any) -> bool:
    """Unmap the region's physical backing (stable virtual addresses); ``True`` when it was
    mapped. Collective (every rank holds the same state); the caller barriers afterwards so that no
    rank returns before every rank has released it."""
    state = getattr(workspace, "_ar_ws2", None)
    if state is None or not state.handle.mapped:
        return False
    state.handle._unmap_and_release_handles()
    return True


def checkpoint_restore_state(workspace: Any, comm_backend: Any) -> bool:
    """Remap the region at the same addresses and put it and the call counter back into the idle
    state (the new backing has undefined contents); ``True`` when it was unmapped. Collective; the
    caller synchronises and barriers before any rank can launch."""
    state = getattr(workspace, "_ar_ws2", None)
    if state is None or state.handle.mapped:
        return False
    state.handle._create_and_map_handles(comm_backend)
    ptrs = state.handle.get_buffer_ptrs_host()
    if (
        int(ptrs[state.rank]) != state.own_ptr
        or int(ptrs[1 - state.rank]) != state.peer_ptr
    ):
        raise RuntimeError(
            "specialized all-reduce region was remapped at other addresses"
        )
    assert _module is not None
    _module.allreduce_fusion_ws2_init(state.epoch, state.own_ptr, state.capacity)
    return True


def release_state(workspace: Any) -> None:
    """Drop the state (workspace destroy); the region is freed with its last reference."""
    state = getattr(workspace, "_ar_ws2", None)
    if state is None:
        return
    workspace._ar_ws2 = None
    with contextlib.suppress(Exception):  # teardown must not raise
        torch.cuda.synchronize(state.device_idx)
    state.handle = None
    _stats["states_released"] += 1


def _reject_reason(
    state: Optional[_State],
    input: torch.Tensor,
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    weight_bias: Any,
    fp32_acc: Any,
) -> Optional[str]:
    """``None`` when the specialized kernel serves the call, else the fallback reason. Every
    input is identical on both ranks of a tensor-parallel group."""
    if os.environ.get(_DISABLE_ENV) == "1":
        return "kill_switch"
    if state is None:
        return "no_state"
    allowlist = _load_allowlist()
    # ---- token count first: calls outside the ranges (e.g. decode) leave after a few checks ----
    shape = input.shape
    if len(shape) != 2:
        return "shape"
    num_tokens = int(shape[0])
    if not any(lo <= num_tokens <= hi for lo, hi in state.token_ranges):
        return "num_tokens"
    # ---- semantics ----
    if fp32_acc is not True:
        return "fp32_acc"
    if weight_bias is None or float(weight_bias) not in allowlist.weight_bias:
        return "weight_bias"
    if (
        residual_in is None
        or residual_out is None
        or norm_out is None
        or rms_gamma is None
    ):
        return "missing_tensor"
    tensors = (input, residual_in, residual_out, norm_out, rms_gamma)
    if any(t.dtype != _DTYPE for t in tensors):
        return "dtype"
    # ---- shape / layout ----
    if (
        shape[1] != _HIDDEN
        or residual_in.shape != shape
        or residual_out.shape != shape
        or norm_out.shape != shape
        or rms_gamma.dim() != 1
        or rms_gamma.shape[0] != _HIDDEN
    ):
        return "shape"
    if not all(t.is_contiguous() for t in tensors):
        return "non_contiguous"
    # ---- device (last) ----
    if any(t.get_device() != state.device_idx for t in tensors):
        return "device"
    return None


def _bump_fallback(reason: str, state: Optional[_State], input: Any) -> None:
    _stats["fallbacks"] += 1
    reasons = _stats["fallback_reasons"]
    reasons[reason] = reasons.get(reason, 0) + 1
    if state is None:
        if reason == "kill_switch":
            _debug_once("kill_switch", KILL_SWITCH_MARKER)
        return
    if reason == "kill_switch":
        _emit_once("kill_switch", KILL_SWITCH_MARKER)
        return
    if f"fallback:{reason}" in _markers_emitted:
        return
    _bump_fallback_marker(reason, input)


def _bump_fallback_marker(reason: str, input: Any) -> None:
    """The one-time marker of a fallback reason on a workspace with a state."""
    try:
        context = (
            f"shape={tuple(input.shape)}, dtype={input.dtype}, device={input.device}"
        )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break the call
        context = f"context unavailable: {type(exc).__name__}"
    if reason == "guard_error":
        context += f"; {_stats['guard_error']}"
    _emit_once(f"fallback:{reason}", f"{FALLBACK_MARKER}: {reason} ({context})"[:800])


def outside_token_ranges(workspace: Any, input: torch.Tensor) -> bool:
    """``True`` when the kernel cannot serve the call because of its token count alone (a 2-D
    ``input`` outside every allowlisted range of the workspace's state): the cheap first check
    of :func:`allreduce_fusion`, so calls such as decode sizes reach the fallback backend
    without the full guard. Counted as a ``num_tokens`` fallback. Rank-symmetric (the token
    count and the agreed state)."""
    if torch.compiler.is_compiling():
        return True  # nothing of this module is traced; try_dispatch would decline too
    state = workspace._ar_ws2
    if state is None:
        return False  # try_dispatch records the reason
    shape = input.shape
    if len(shape) != 2:
        return False
    num_tokens = shape[0]
    for lo, hi in state.token_ranges:
        if lo <= num_tokens <= hi:
            return False
    _stats["calls"] += 1
    _stats["fallbacks"] += 1
    reasons = _stats["fallback_reasons"]
    reasons["num_tokens"] = reasons.get("num_tokens", 0) + 1
    if "fallback:num_tokens" not in _markers_emitted:
        _bump_fallback_marker("num_tokens", input)
    return True


def try_dispatch(
    workspace: Any,
    input: torch.Tensor,
    residual_in: Optional[torch.Tensor],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    rms_gamma: Optional[torch.Tensor],
    rms_eps: float,
    weight_bias: Any,
    fp32_acc: Any,
    launch_with_pdl: bool,
) -> Optional[torch.Tensor]:
    """``norm_out`` after the specialized kernel ran, or ``None`` when the workspace's fallback
    backend must serve the call. Raises when the region is detached (between
    ``checkpoint_prepare`` and ``checkpoint_restore``) or the launch itself fails: the call is
    collective, so a rank never falls back after its guard passed."""
    # Under torch.compile (dynamo) tracing use the fallback: this check folds to a constant,
    # so none of the host-side dispatch below is traced.
    if torch.compiler.is_compiling():
        return None
    _stats["calls"] += 1
    state = getattr(workspace, "_ar_ws2", None)
    try:
        reason = _reject_reason(
            state,
            input,
            residual_in,
            residual_out,
            norm_out,
            rms_gamma,
            weight_bias,
            fp32_acc,
        )
    except Exception as exc:  # noqa: BLE001 - a guard exception must never break the fallback
        reason = "guard_error"
        _stats["guard_error"] = f"{type(exc).__name__}: {exc}"[:500]
    if reason is not None:
        _bump_fallback(reason, state, input)
        return None

    assert state is not None and _module is not None
    if not state.handle.mapped:
        raise RuntimeError("specialized all-reduce fusion handles are not attached")
    _module.allreduce_fusion_ws2_ar_rmsnorm(
        input,
        residual_in,
        rms_gamma,
        residual_out,
        norm_out,
        state.own_ptr,
        state.peer_ptr,
        state.epoch,
        state.capacity,
        state.grid,
        float(rms_eps),
        float(weight_bias),
        bool(launch_with_pdl),
    )
    _stats["dispatch_count"] += 1
    if "dispatch" not in _markers_emitted:
        _emit_once(
            "dispatch",
            f"{DISPATCH_MARKER} (num_tokens={int(input.shape[0])}, device={input.device}, "
            f"grid={state.grid}, capacity={state.capacity})",
        )
    return norm_out


def _allreduce_fusion_ws2_specialized_stats() -> Dict[str, Any]:
    """Diagnostics: dispatch counters, state/JIT status and the compile footprint.

    One JIT module (``compiled_variants`` is 0 or 1) with one kernel instantiation; the token count
    is a runtime launch argument, so no call shape triggers a compile and the compile-cache key is
    the module name alone. The module is built at workspace creation (``precompiled``), never at
    call time or during a CUDA-graph capture.
    """
    allowlist = _load_allowlist()
    return {
        **_stats,
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "state_rejections": dict(_stats["state_rejections"]),
        "module_name": ALLREDUCE_FUSION_WS2_SPECIALIZED_MODULE_NAME,
        "compile_cache_key": ALLREDUCE_FUSION_WS2_SPECIALIZED_MODULE_NAME
        + " (no shape key: the token count is a runtime argument)",
        "compiled_variants": int(_module is not None),
        "distinct_kernels_for_allowlist": 1,
        "precompiled": _module is not None,
        "allowlist_weight_bias": list(allowlist.weight_bias),
        "allowlist_num_tokens": [list(r) for r in allowlist.num_tokens],
        "allowlist_error": _allowlist_error,
        "world_size": _WORLD_SIZE,
        "hidden_dim": _HIDDEN,
        "dtype": _DTYPE_NAME,
        "pattern": _PATTERN,
        "compute_capability": list(_COMPUTE_CAPABILITY),
        "dispatch_marker": DISPATCH_MARKER,
        "kill_switch_marker": KILL_SWITCH_MARKER,
        "fallback_marker": FALLBACK_MARKER,
        "state_marker": STATE_MARKER,
        "no_state_marker": NO_STATE_MARKER,
        "markers_emitted": sorted(_markers_emitted),
    }
