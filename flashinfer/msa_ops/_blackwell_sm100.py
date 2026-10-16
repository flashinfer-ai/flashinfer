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

Long-prefill route.  A prefill call whose requests each have at least 8192
queries (and, on flat K/V, at least 8192 keys) with BF16 Q/K/V, causal
masking, no explicit ``q_offset`` and unit temperature scale is served request
by request on the selected-block long route (``_run_long_prefill``; one
request is the one-element case of the same path): the host slices
``q``/``k``/``v``/``out``/``lse`` on their first dimension, binds zero-based
2-entry ``cu_seqlens`` and launches the producer and combine once per request
with its own work plan, carving the partials (u8x32 payload, BF16x4 dequant
scales, LSE) from pools sized by the longest request.  One request takes its
lengths from the tensor extents.  Several requests need their boundaries on
the host: without a workspace they
are read from ``cu_seqlens_q`` and the KV length carrier on every call (one
device-to-host copy); an ``MSASparseAttentionWorkspace`` keeps the boundaries
of its last binding -- pointer, version, extent *and identity* of the two
tensors -- so the eager warm call reads once and CUDA graph capture never
synchronizes.  ``cu_seqlens_q[0]`` / ``cu_seqlens_k[0]`` must be 0 on this
route.  A batch with one request below the thresholds takes the union route
as a whole.
"""

from __future__ import annotations

import functools
import math
import threading
import weakref
from contextlib import nullcontext
from typing import Any, NamedTuple, Optional, Tuple

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

_LONG_WORK_GROUP_BUCKETS = (128, 64, 32, 16, 8, 4, 2, 1)
_LONG_GROUP_BOUNDARIES = _LONG_WORK_GROUP_BUCKETS[:-1]
# Tail-wave split of the long-prefill work list (mirror of the Weave module's WORK_TAIL_SPLIT* constants and of
# decompose_long_prefill_work / simulate_long_prefill_makespan / tail_split_long_prefill_work; the Cake test
# loom/tests/test_minimax_sparse_long_prefill_work_plan.py pins the parity).
_LONG_WORK_TAIL_SPLIT = True
_LONG_WORK_TAIL_SPLIT_FIXED_GROUPS = 2
_LONG_WORK_TAIL_SPLIT_MIN_GROUPS = 32
_LONG_WORK_TAIL_SPLIT_MAX_ROUNDS = 4
# Q-resident launch order of the long-prefill work list (lever P1b; mirror of the Weave module's WORK_ORDER_*
# constants and of first_query_tokens_of_long_prefill_work / order_long_prefill_work_q_resident, pinned by the
# same Cake test).  Requires the long_prefill_* kernels of this export, which derive the CTA's group count from
# the item's q_count (the q_group_segment_end_* scalars are bucket counts, not launch-index boundaries).
_LONG_WORK_ORDER_Q_RESIDENT = True
_LONG_WORK_ORDER_Q_WINDOW = 1024
_LONG_WORK_ORDER_FILL_GROUPS = 128
_LONG_QSPLIT_Q_MASK = 0x00FF_FFFF
_LONG_QSPLIT_SLOT_SHIFT = 24
_LONG_QSPLIT_SINGLE_SHIFT = 28
# BF16 dequant scales per u8x32 partial row (one per 32-column segment).
_LONG_PARTIAL_SEGMENT_COUNT = 4

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

    The multi-request long-prefill route reads the request boundaries from
    ``cu_seqlens_q`` and ``cu_seqlens_k`` / ``seqused_k`` to the host in the
    eager warm call and keeps them for that binding: pass the same tensor
    objects to every later call (a rebuilt ``cu_seqlens`` tensor -- a new
    object, even at the old address -- is read again at the cost of one
    synchronization) and rewrite their contents only through torch operations,
    which advance the tensor version the binding is keyed on.
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
        # Long-prefill route: one plan per request, the zero-based 2-entry
        # ``cu_seqlens`` views and the partial pools, keyed by the q2k binding
        # and the request lengths (``_long_prefill_state``).
        self._long_prefill: dict = {}
        # Host copy of the request boundaries of one (cu_seqlens_q, cu_seqlens_k
        # | seqused_k) binding with weak references to the two tensor objects;
        # filled by the eager warm call, read during capture (``_request_lengths``).
        self._request_lengths: dict[tuple, tuple] = {}
        # Union census verdict of one long-admitted paged GQA-16 binding
        # (lever P10; ``_union_census_route``): signature, counters, verdict
        # and strong references to the two tensors it was read from.
        self._union_census: dict = {}
        self._warmed_launches: set[tuple] = set()
        self._bound_stream_ptr: Optional[int] = None
        self._captured = False


_topk_warmed_devices: set[tuple[int, str]] = set()
_topk_warmed_devices_lock = threading.Lock()
# Top-K16 v2 launch plan (duplicated verbatim from
# loom/examples/weave/minimax_sparse_topk_select_v2_sm100.py::topk_grid_plan because this module
# cannot import loom).  loom/tests/test_blackwell_msa_export_topk_plan.py (CPU pytest) exec's this copy
# from its source and asserts the plan and every _TOPK_* constant agree with the module over a shape
# grid; tools/msa_rebench/dev/topk_v2_check.py repeats the check on the GPU steps.
_TOPK_QT = 16
_TOPK_SLICES = (1, 2, 4, 8, 16)
_TOPK_WAVE_CTAS = 444
_TOPK_CTA_FIXED_BLOCKS = 1536
_TOPK_CTA_STAGE_ONLY_BLOCKS = 512
_TOPK_CTA_SLICE_BLOCKS = 112
_TOPK_MIN_SLICE = 32
_TOPK_MAX_INDEX_BITS = 22
_TOPK_CAND_CAP = 64
_TOPK_BMAX_CHUNK_ITERS = 8
_TOPK_BMAX_WORDS_PER_CHUNK = 64
_TOPK_STREAM_MAX_ROWS = 512
_TOPK_STREAM_MAX_NVP = 1024
# Lever T11b (MSA ceiling Goal round 7): grids of at most _TOPK_SMALL_GRID_MAX_ROWS rows over at most
# _TOPK_STREAM_MAX_NVP valid pages whose cost-minimal slice streams plan the stage-only S = _TOPK_SMALL_GRID_SLICES
# split instead (one single-pass CTA per slice; the streaming rule then does not fire).
_TOPK_SMALL_GRID_MAX_ROWS = 32
_TOPK_SMALL_GRID_SLICES = 16
_TOPK_ROUTE_RADIX = "topk_select"
_TOPK_ROUTE_STREAMING = "topk_select_streaming"
# 32-query instance (MSA ceiling Goal lever T4): two queries per lane, u16-packed histogram, 2 CTAs/SM.
_TOPK_Q32_WAVE_CTAS = 296
_TOPK_Q32_MIN_Q = 32
_TOPK_Q32_MIN_SLICE = 1024
_TOPK_Q32_MAX_SLICE = 65535
_TOPK_ROUTE_Q32 = "topk_select_q32"
_TOPK_KEQ_MAX_SLEN = 512
_TOPK_ROUTE_KEQ = "topk_select_keq"
_topk_workspaces: dict[int, dict[str, torch.Tensor]] = {}
_topk_workspaces_lock = threading.Lock()


def _topk_grid_plan(
    total_q: int, num_heads: int, num_valid_pages: int, max_k_tiles: int, qpack: int = 1
) -> dict:
    """Shape-only plan: over ``s in (1,2,4,8,16)`` with ``ceil(nvp / s) >= 32`` (else ``s = 1``), ``S`` minimises
    ``ceil(groups * s / W) * (fixed(s) + ceil(nvp / s) + 112 * s)`` with ``fixed(s) = 1536`` for a slice above the
    candidate cap (multi-pass CTA) and ``512`` for a slice within it (stage-only CTA); ``W`` = 444 for the 16-query
    program, 296 for the 32-query instance ``qpack = 2``; ties: the larger ``s``.  Lever T11b (16-query plan only):
    when ``total_q * num_heads <= 32``, ``nvp <= 1024``, the minimiser's slice exceeds the cap and the ``S = 16`` grid
    fits one wave, ``S = 16`` (stage-only) replaces the minimiser.  ``bid = (h * S + s) * num_qtiles + qtile``."""
    total_q = int(total_q)
    num_heads = int(num_heads)
    num_valid_pages = int(num_valid_pages)
    max_k_tiles = int(max_k_tiles)
    qpack = int(qpack)
    if min(total_q, num_heads, num_valid_pages, max_k_tiles) <= 0:
        raise ValueError(
            "topk grid plan requires positive total_q, num_heads, num_valid_pages, max_k_tiles"
        )
    if qpack not in (1, 2):
        raise ValueError("qpack must be 1 (16-query program) or 2 (32-query program)")
    if num_heads * max_k_tiles >= 2**31:
        raise ValueError("num_heads * max_k_tiles must fit in int32")
    if total_q * num_heads * _TOPK >= 2**31:
        raise ValueError("total_q * num_heads * 16 must fit in int32")
    index_bits = max(1, (max_k_tiles - 1).bit_length())
    if index_bits > _TOPK_MAX_INDEX_BITS:
        raise ValueError(
            f"max_k_tiles must be <= 2**{_TOPK_MAX_INDEX_BITS}, got {max_k_tiles}"
        )
    qt = _TOPK_QT * qpack
    wave_ctas = _TOPK_WAVE_CTAS if qpack == 1 else _TOPK_Q32_WAVE_CTAS
    num_qtiles = (total_q + qt - 1) // qt
    groups = num_heads * num_qtiles
    fitting = [
        c for c in _TOPK_SLICES if -(-num_valid_pages // c) >= _TOPK_MIN_SLICE
    ] or [1]
    num_slices = fitting[0]
    best_cost = None
    for candidate in fitting:
        waves = -(-(groups * candidate) // wave_ctas)
        cand_slice = -(-num_valid_pages // candidate)
        fixed = (
            _TOPK_CTA_FIXED_BLOCKS
            if cand_slice > _TOPK_CAND_CAP
            else _TOPK_CTA_STAGE_ONLY_BLOCKS
        )
        cost = waves * (fixed + cand_slice + _TOPK_CTA_SLICE_BLOCKS * candidate)
        if best_cost is None or cost <= best_cost:
            best_cost = cost
            num_slices = candidate
    # Lever T11b (round 7): a small grid whose cost-minimal slice streams (> cap) over the streaming program's nvp
    # range takes the stage-only S = 16 plan (duplicated verbatim from the module; see its T11b paragraph).
    if (
        qpack == 1
        and total_q * num_heads <= _TOPK_SMALL_GRID_MAX_ROWS
        and num_valid_pages <= _TOPK_STREAM_MAX_NVP
        and -(-num_valid_pages // num_slices) > _TOPK_CAND_CAP
        and groups * _TOPK_SMALL_GRID_SLICES <= wave_ctas
    ):
        # here 513 <= nvp <= 1024, so the slice 33..64 is admitted and stage-only
        assert (
            _TOPK_SMALL_GRID_SLICES in fitting
            and -(-num_valid_pages // _TOPK_SMALL_GRID_SLICES) <= _TOPK_CAND_CAP
        )
        num_slices = _TOPK_SMALL_GRID_SLICES
    slice_len = -(-num_valid_pages // num_slices)
    grid = groups * num_slices
    slice_k_tiles = -(-slice_len // 16)
    chunks = -(-slice_k_tiles // _TOPK_BMAX_CHUNK_ITERS)
    return {
        "num_qtiles": num_qtiles,
        "groups": groups,
        "num_slices": num_slices,
        "slice_len": slice_len,
        "grid": grid,
        "index_bits": index_bits,
        "bmax_words": grid * chunks * _TOPK_BMAX_WORDS_PER_CHUNK,
        "qpack": qpack,
    }


def _topk_radix_route(slice_len: int) -> str:
    """The radix select's production instance for a planned slice (duplicated verbatim from
    loom/examples/weave/minimax_sparse_topk_select_v2_sm100.py::topk_radix_route): ``topk_select_keq`` (the
    lever T9 pass-0 all-equal fold) for a streamed slice above the candidate cap ``_TOPK_CAND_CAP`` and up to
    ``_TOPK_KEQ_MAX_SLEN`` blocks, ``topk_select`` for a stage-only slice (the fold cannot fire) and beyond the
    bound -- a cost choice, both instances are bit-identical on every input."""
    if _TOPK_CAND_CAP < int(slice_len) <= _TOPK_KEQ_MAX_SLEN:
        return _TOPK_ROUTE_KEQ
    return _TOPK_ROUTE_RADIX


def _topk_route(
    total_q: int, num_heads: int, num_valid_pages: int, max_k_tiles: int
) -> str:
    """Shape-only program choice (duplicated verbatim from
    loom/examples/weave/minimax_sparse_topk_select_v2_sm100.py::topk_route): ``topk_select_streaming`` (the
    one-CTA-per-row program) for small latency-bound multi-pass grids -- at most ``_TOPK_STREAM_MAX_ROWS`` rows,
    at most ``_TOPK_STREAM_MAX_NVP`` valid pages and a planned slice above the candidate cap (lever T11b plans the
    grids of at most ``_TOPK_SMALL_GRID_MAX_ROWS`` rows of that regime stage-only at ``S = 16``, so they never
    reach the streaming program); ``topk_select_q32``
    (the 32-query instance) for an even ``total_q >= _TOPK_Q32_MIN_Q`` whose 32-query plan has a slice in
    ``[_TOPK_Q32_MIN_SLICE, _TOPK_Q32_MAX_SLICE]``; otherwise one of the query-tiled radix select's two 16-query
    production instances by the planned slice (``_topk_radix_route``): ``topk_select_keq`` for a streamed
    slice in (``_TOPK_CAND_CAP``, ``_TOPK_KEQ_MAX_SLEN``], ``topk_select`` otherwise."""
    plan = _topk_grid_plan(total_q, num_heads, num_valid_pages, max_k_tiles)
    if (
        int(total_q) * int(num_heads) <= _TOPK_STREAM_MAX_ROWS
        and int(num_valid_pages) <= _TOPK_STREAM_MAX_NVP
        and plan["slice_len"] > _TOPK_CAND_CAP
    ):
        return _TOPK_ROUTE_STREAMING
    if int(total_q) % 2 == 0 and int(total_q) >= _TOPK_Q32_MIN_Q:
        plan32 = _topk_grid_plan(total_q, num_heads, num_valid_pages, max_k_tiles, 2)
        if _TOPK_Q32_MIN_SLICE <= plan32["slice_len"] <= _TOPK_Q32_MAX_SLICE:
            return _TOPK_ROUTE_Q32
    return _topk_radix_route(plan["slice_len"])


def _topk_workspace(
    device: torch.device,
    groups: int,
    num_slices: int,
    bmax_words: int,
    capturing: bool,
    qt: int = _TOPK_QT,
):
    """Per-device ``(partials u64, counters u32, block_max u32)`` grown eagerly to the largest plan seen.

    ``counters`` is zeroed once on (re)allocation and self-resets through the kernel's ``atom.inc``
    semaphore, so eager launches and graph replays share one buffer; ``block_max`` (the per-block tile
    maxima of the histogram pass) is fully rewritten by every launch before it is read.  Growth under
    capture is refused.  ``qt`` is the program's queries per CTA (16; 32 for the lever-T4 instance).
    """
    index = _device_index(device)
    need_partials = groups * num_slices * _TOPK * qt if num_slices > 1 else 1
    need_counters = groups if num_slices > 1 else 1
    need_bmax = max(int(bmax_words), 1)
    with _topk_workspaces_lock:
        entry = _topk_workspaces.get(index)
        if (
            entry is None
            or entry["partials"].numel() < need_partials
            or entry["counters"].numel() < need_counters
            or entry["block_max"].numel() < need_bmax
        ):
            if capturing:
                raise RuntimeError(
                    "msa_topk_select must be invoked eagerly with this (or a larger) shape on this "
                    "device before CUDA graph capture: the K-split workspace cannot grow under capture"
                )
            cap_partials = max(
                need_partials, 0 if entry is None else entry["partials"].numel()
            )
            cap_counters = max(
                need_counters, 0 if entry is None else entry["counters"].numel()
            )
            cap_bmax = max(
                need_bmax, 0 if entry is None else entry["block_max"].numel()
            )
            with torch.cuda.device(index):
                entry = {
                    "partials": torch.empty(
                        cap_partials,
                        dtype=torch.uint64,
                        device=torch.device("cuda", index),
                    ),
                    "counters": torch.zeros(
                        cap_counters,
                        dtype=torch.uint32,
                        device=torch.device("cuda", index),
                    ),
                    "block_max": torch.empty(
                        cap_bmax, dtype=torch.uint32, device=torch.device("cuda", index)
                    ),
                }
            _topk_workspaces[index] = entry
        return entry["partials"], entry["counters"], entry["block_max"]


_eager_dummies: dict[tuple, torch.Tensor] = {}
_eager_dummies_lock = threading.Lock()
# Workspace-free long-prefill calls keep the state of their latest binding only.
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


def _paged_prefill_cu_seqlens_k(
    kv_lens: torch.Tensor,
    *,
    workspace: Optional[MSASparseAttentionWorkspace],
    name: str,
) -> torch.Tensor:
    """The ``(batch + 1,)`` int32 ``cu_seqlens_k`` carrier of paged prefill: all zero.

    The paged prefill programs take their request KV lengths from
    ``kv_lens`` (``seqused_k``) and their K/V rows from ``page_table``; the
    offsets are read only for a token base the paged loaders overwrite with
    0, so no scan is issued (lever P-H1d, mirrors the Cake dispatcher).  The
    buffer is zeroed when it is created -- once per warmed workspace, every
    call without one -- and only re-bound afterwards: the attention kernel is
    the first GPU activity of the call.  The flat-ABI form ``max_pages == 0``
    (kv_len from adjacent offsets) is handled by the caller.
    """

    previous = None if workspace is None else workspace._buffers.get(name)
    cu_k = _workspace_buffer(
        workspace,
        name,
        (kv_lens.numel() + 1,),
        dtype=torch.int32,
        device=kv_lens.device,
    )
    if cu_k is not previous:
        cu_k.zero_()
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
) -> tuple[bool, Optional[torch.Tensor], torch.Tensor, torch.Tensor, int]:
    """``(paged, cu_k, kv_lens, page_table, max_pages)`` of one call.

    ``cu_k`` is ``None`` for paged prefill without caller offsets: the prefill
    runners bind the all-zero carrier of :func:`_paged_prefill_cu_seqlens_k`
    (levers P-H1 / P-H1d; the paged programs never read the offsets).
    """

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
            # Paged decode reads seqused_k and binds it as the ABI-only
            # cu_seqlens_k carrier; paged prefill binds an all-zero one at
            # its launch (P-H1d).
            cu_k = None if prefill else kv_lens
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


_LONG_MIN_REQUEST_LEN = 8192


def _long_prefill_candidate(
    *,
    paged: bool,
    group_size: int,
    max_pages: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    causal: bool,
    q_offset_is_none: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
) -> bool:
    """The length-independent half of the long-prefill predicate."""

    return bool(
        (
            (
                paged
                and (
                    (group_size == 8 and max_pages >= 64)
                    or (group_size == 16 and max_pages > 64)
                )
            )
            or (not paged and group_size == 16)
        )
        and q_dtype == k_dtype == v_dtype == torch.bfloat16
        and causal
        and q_offset_is_none
        and (not return_temperature_lse or lse_temperature_scale == 1.0)
    )


def _use_long_prefill(
    *,
    q_lens: tuple[int, ...],
    k_lens: tuple[int, ...],
    paged: bool,
    group_size: int,
    max_pages: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    causal: bool,
    q_offset_is_none: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
) -> bool:
    """Whether every request of the call is served by the selected-block long route.

    ``q_lens`` / ``k_lens`` are the per-request query and flat-KV lengths (one
    entry per request; for one request the tensor extents).  A request
    qualifies when its query length is at least 8192 and, on flat K/V, its KV
    length is at least 8192; the remaining conditions (BF16 Q/K/V, causal, no
    explicit q_offset, unit temperature scale, flat GQA-16 or paged GQA-8/16
    with the page-count bounds) are shared by all requests.  A batch with one
    non-qualifying request takes the union route as a whole.
    """

    if len(q_lens) < 1 or len(q_lens) != len(k_lens):
        raise ValueError("q_lens and k_lens must carry one entry per request")
    return bool(
        _long_prefill_candidate(
            paged=paged,
            group_size=group_size,
            max_pages=max_pages,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            causal=causal,
            q_offset_is_none=q_offset_is_none,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=lse_temperature_scale,
        )
        and min(q_lens) >= _LONG_MIN_REQUEST_LEN
        and (paged or min(k_lens) >= _LONG_MIN_REQUEST_LEN)
    )


# Lever P10 (mirror of minimax_sparse_attention_long_prefill_sm100.prefill_union_census /
# union_census_prefers_union / union_census_route; the Cake test
# loom/tests/test_minimax_sparse_prefill_union_census_route.py pins the parity): a paged GQA-16
# request the long route admits takes the folded union program when its tokens select (nearly)
# the same KV blocks -- the exact per-tile union census from ``q2k_indices`` (torch ops, one
# three-counter device-to-host read at the long route's own plan sync point, cached per binding)
# reads a waste ratio union_visits / ideal_visits <= 5/4 and no tile union above 64 blocks.
_UNION_CENSUS_TILE_TOKENS = 16
_UNION_CENSUS_MAX_PAGES = 1024
_UNION_CENSUS_MAX_TILE_UNION = 64
_UNION_CENSUS_WASTE_NUM = 5
_UNION_CENSUS_WASTE_DEN = 4
_implicit_union_census: dict[tuple, dict] = {}
_implicit_union_census_lock = threading.Lock()


class _UnionCensus(NamedTuple):
    """Per-request block-visit counts of the folded GQA-16 union program."""

    union_visits: int
    ideal_visits: int
    max_tile_union: int
    tiles: int


def _prefill_union_census(
    q2k_indices: torch.Tensor,
    *,
    q_len: int,
    kv_len,
    tile_tokens: int = _UNION_CENSUS_TILE_TOKENS,
) -> _UnionCensus:
    """Exact per-tile union census of one request (``q2k_indices[:, :q_len]``)."""

    heads, total_q, topk = (int(value) for value in q2k_indices.shape)
    q_len = int(q_len)
    if q_len <= 0 or q_len > total_q:
        raise ValueError("union census needs 0 < q_len <= total_q")
    tile_tokens = int(tile_tokens)
    tiles = (q_len + tile_tokens - 1) // tile_tokens
    device = q2k_indices.device
    selection = q2k_indices[:, :q_len]
    pad = tiles * tile_tokens - q_len
    if pad:
        selection = torch.cat(
            (
                selection,
                torch.full(
                    (heads, pad, topk), -1, dtype=q2k_indices.dtype, device=device
                ),
            ),
            dim=1,
        )
    selection = selection.reshape(heads, tiles, tile_tokens * topk)
    tile_index = torch.arange(tiles, dtype=torch.int64, device=device)
    q_valid = torch.clamp(q_len - tile_index * tile_tokens, max=tile_tokens)
    if isinstance(kv_len, int):
        kv_length = torch.tensor([kv_len], dtype=torch.int64, device=device)
    else:
        kv_length = kv_len.reshape(-1)[:1].to(device=device, dtype=torch.int64)
    pages = (kv_length + _BLOCK_SIZE - 1) // _BLOCK_SIZE
    visible_tokens = kv_length - q_len + tile_index * tile_tokens + q_valid
    visible = torch.minimum(
        (visible_tokens + _BLOCK_SIZE - 1) // _BLOCK_SIZE, pages
    ).clamp_(min=1)
    valid = (selection >= 0) & (
        selection < visible.view(1, tiles, 1).to(selection.dtype)
    )
    masked = torch.where(valid, selection, torch.full_like(selection, -1))
    ordered = masked.sort(dim=-1).values
    distinct = ((ordered[..., 1:] != ordered[..., :-1]) & (ordered[..., 1:] >= 0)).sum(
        dim=-1, dtype=torch.int64
    ) + (ordered[..., 0] >= 0).to(torch.int64)
    row_counts = valid.view(heads, tiles, tile_tokens, topk).sum(
        dim=-1, dtype=torch.int64
    )
    counters = torch.stack(
        (distinct.sum(), row_counts.amax(dim=-1).sum(), distinct.amax())
    ).cpu()
    union_visits, ideal_visits, max_tile_union = (
        int(value) for value in counters.tolist()
    )
    return _UnionCensus(union_visits, ideal_visits, max_tile_union, heads * tiles)


def _union_census_prefers_union(census: _UnionCensus) -> bool:
    return bool(
        census.ideal_visits > 0
        and census.max_tile_union <= _UNION_CENSUS_MAX_TILE_UNION
        and census.union_visits * _UNION_CENSUS_WASTE_DEN
        <= census.ideal_visits * _UNION_CENSUS_WASTE_NUM
    )


def _union_census_route(
    workspace: Optional[MSASparseAttentionWorkspace],
    *,
    q2k_indices: torch.Tensor,
    q_len: int,
    kv_lens: torch.Tensor,
    max_pages: int,
    capturing: bool,
) -> bool:
    """Whether a long-admitted paged GQA-16 request takes the union program (one request only).

    Cached per binding (``q2k_indices`` pointer / version / shape, query length, ``seqused_k``
    binding, page-table width) in the workspace, or in a module slot keeping the latest binding;
    capture replays a warmed verdict and never synchronizes (an unwarmed capture keeps the long
    route, whose own capture rule applies).
    """

    if int(max_pages) > _UNION_CENSUS_MAX_PAGES:
        return False
    signature = (
        _device_index(q2k_indices.device),
        int(q2k_indices.data_ptr()),
        int(q2k_indices._version),
        tuple(int(value) for value in q2k_indices.shape),
        int(q_len),
        _host_lengths_signature(kv_lens),
        int(max_pages),
    )
    if workspace is not None:
        entry = workspace._union_census
    else:
        with _implicit_union_census_lock:
            entry = _implicit_union_census.get(signature)
    if entry and entry.get("signature") == signature:
        return bool(entry["prefers_union"])
    if capturing or torch.cuda.is_current_stream_capturing():
        return False
    census = _prefill_union_census(q2k_indices, q_len=q_len, kv_len=kv_lens)
    entry = {
        "signature": signature,
        "census": census,
        "prefers_union": _union_census_prefers_union(census),
        "q2k_owner": q2k_indices,
        "kv_lens_owner": kv_lens,
    }
    if workspace is not None:
        workspace._union_census = entry
    else:
        with _implicit_union_census_lock:
            _implicit_union_census.clear()
            _implicit_union_census[signature] = entry
    return bool(entry["prefers_union"])


class _RequestLengths(NamedTuple):
    """Host view of one prefill call's request boundaries."""

    q_lens: tuple[int, ...]
    k_lens: tuple[int, ...]
    q_first: int
    k_first: int


def _host_lengths_signature(tensor: torch.Tensor) -> tuple:
    return (int(tensor.data_ptr()), int(tensor._version), int(tensor.numel()))


def _read_request_lengths(
    *,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    kv_lens: torch.Tensor,
    paged: bool,
) -> _RequestLengths:
    """One device-to-host copy of the request boundaries."""

    q_bounds = [int(value) for value in cu_q.tolist()]
    q_lens = tuple(b - a for a, b in zip(q_bounds, q_bounds[1:], strict=False))
    if paged:
        k_lens = tuple(int(value) for value in kv_lens.tolist())
        k_first = 0
    else:
        k_bounds = [int(value) for value in cu_k.tolist()]
        k_lens = tuple(b - a for a, b in zip(k_bounds, k_bounds[1:], strict=False))
        k_first = k_bounds[0]
    if len(q_lens) != len(k_lens):
        raise ValueError(
            "cu_seqlens_q and the KV lengths must describe the same requests"
        )
    if min(q_lens) <= 0 or min(k_lens) <= 0:
        raise ValueError("every request must have positive query and KV lengths")
    return _RequestLengths(q_lens, k_lens, q_bounds[0], k_first)


def _request_lengths(
    workspace: Optional[MSASparseAttentionWorkspace],
    *,
    cu_q: torch.Tensor,
    cu_k: Optional[torch.Tensor],
    kv_lens: torch.Tensor,
    paged: bool,
    total_q: int,
    k_outer_dim: int,
    capturing: bool,
) -> _RequestLengths:
    """Request boundaries of a multi-request call.

    Without a workspace every call reads them from the device (one
    synchronization).  A workspace keeps the boundaries of its last binding:
    the ``(pointer, version, extent)`` of ``cu_seqlens_q`` and of the KV
    length carrier (``cu_seqlens_k`` on flat K/V, ``seqused_k`` on paged
    K/V), the ``q``/``k`` extents, and the identity of the two tensor objects.
    An eager call with the same tensor objects reads nothing; a rebuilt
    ``cu_seqlens`` tensor -- including one the allocator placed at the old
    address with a fresh version counter -- is read again, so a cached split
    is never applied to new boundaries.  CUDA graph capture never
    synchronizes: it requires the warmed binding and trusts it, as the graph
    bakes the same pointers.
    """

    k_carrier = kv_lens if paged else cu_k
    if workspace is None:
        if capturing:
            raise RuntimeError(
                "MSA long-prefill request lengths cannot be read during CUDA graph "
                "capture; warm an MSASparseAttentionWorkspace eagerly first"
            )
        return _read_request_lengths(cu_q=cu_q, cu_k=cu_k, kv_lens=kv_lens, paged=paged)
    signature = (
        _device_index(cu_q.device),
        _host_lengths_signature(cu_q),
        _host_lengths_signature(k_carrier),
        paged,
        total_q,
        k_outer_dim,
    )
    entry = workspace._request_lengths.get(signature)
    if entry is not None:
        owners, lengths = entry
        if capturing or (owners[0]() is cu_q and owners[1]() is k_carrier):
            return lengths
    if capturing:
        raise RuntimeError(
            "MSA long-prefill request lengths must be warmed by an eager call "
            "with the exact cu_seqlens / seqused_k tensors before CUDA graph capture"
        )
    lengths = _read_request_lengths(cu_q=cu_q, cu_k=cu_k, kv_lens=kv_lens, paged=paged)
    workspace._request_lengths.clear()
    workspace._request_lengths[signature] = (
        (weakref.ref(cu_q), weakref.ref(k_carrier)),
        lengths,
    )
    return lengths


# ---------------------------------------------------------------------------
# Long prefill: host-side work plan over the selected blocks
# ---------------------------------------------------------------------------


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


def _long_prefill_decompose_work(counts_host, *, buckets, q_tokens_per_group):
    """Binary decomposition of every (kv head, kv block) row into (group_count, (head, kv_block, q_begin,
    q_count, 0, kv_block)) items, largest bucket first, each item as full as the bucket allows."""
    work = []
    for head, head_counts in enumerate(counts_host):
        for kv_block, row_count in enumerate(head_counts):
            q_begin = 0
            remaining = int(row_count)
            for group_count in buckets:
                group_capacity = group_count * q_tokens_per_group
                while (
                    remaining + q_tokens_per_group - 1
                ) // q_tokens_per_group >= group_count:
                    q_count = min(group_capacity, remaining)
                    derived_group_count = (
                        q_count + q_tokens_per_group - 1
                    ) // q_tokens_per_group
                    if derived_group_count != group_count:
                        raise AssertionError(
                            "long-prefill q-count group derivation drifted"
                        )
                    work.append(
                        (group_count, (head, kv_block, q_begin, q_count, 0, kv_block))
                    )
                    q_begin += q_count
                    remaining -= q_count
            if remaining != 0 or q_begin != row_count:
                raise AssertionError("long-prefill binary work decomposition failed")
    return work


def _long_prefill_simulate_makespan(work, *, sm_count, fixed_groups):
    """Longest-processing-time simulation of the launch order (group count descending, stable) on sm_count
    machines: the finish time of every item in that order and the makespan (cost = groups + fixed)."""
    import heapq

    machines = [0] * max(1, int(sm_count))
    heapq.heapify(machines)
    finishes = []
    for group_count, _entry in sorted(work, key=lambda item: item[0], reverse=True):
        start = heapq.heappop(machines)
        finish = start + int(group_count) + int(fixed_groups)
        heapq.heappush(machines, finish)
        finishes.append(finish)
    return finishes, (max(finishes) if finishes else 0)


def _long_prefill_tail_split_work(
    work,
    *,
    sm_count,
    q_tokens_per_group,
    fixed_groups=_LONG_WORK_TAIL_SPLIT_FIXED_GROUPS,
    min_groups=_LONG_WORK_TAIL_SPLIT_MIN_GROUPS,
    max_rounds=_LONG_WORK_TAIL_SPLIT_MAX_ROUNDS,
):
    """Halve the items that finish inside the simulated tail wave while the simulated makespan drops (mirror
    of the Weave module's tail_split_long_prefill_work; bitwise-neutral host reordering)."""
    current = list(work)
    for _round in range(max(0, int(max_rounds))):
        ordered = sorted(current, key=lambda item: item[0], reverse=True)
        finishes, makespan = _long_prefill_simulate_makespan(
            ordered, sm_count=sm_count, fixed_groups=fixed_groups
        )
        total = sum(group_count + int(fixed_groups) for group_count, _entry in ordered)
        ideal = -(-total // max(1, int(sm_count)))
        if makespan <= ideal:
            break
        candidate = []
        split_any = False
        for (group_count, entry), finish in zip(ordered, finishes, strict=False):
            if (
                finish > ideal
                and group_count >= int(min_groups)
                and group_count % 2 == 0
            ):
                head, kv_block, q_begin, q_count, zero, kv_block_again = entry
                half_groups = group_count // 2
                q_first = half_groups * q_tokens_per_group
                q_second = q_count - q_first
                if (
                    q_second <= 0
                    or (q_second + q_tokens_per_group - 1) // q_tokens_per_group
                    != half_groups
                ):
                    raise AssertionError(
                        "long-prefill tail split drifted from the kernel's group derivation"
                    )
                candidate.append(
                    (
                        half_groups,
                        (head, kv_block, q_begin, q_first, zero, kv_block_again),
                    )
                )
                candidate.append(
                    (
                        half_groups,
                        (
                            head,
                            kv_block,
                            q_begin + q_first,
                            q_second,
                            zero,
                            kv_block_again,
                        ),
                    )
                )
                split_any = True
            else:
                candidate.append((group_count, entry))
        if not split_any:
            break
        _finishes, new_makespan = _long_prefill_simulate_makespan(
            candidate, sm_count=sm_count, fixed_groups=fixed_groups
        )
        if new_makespan >= makespan:
            break
        current = candidate
    return current


def _long_prefill_first_query_tokens(work, *, row_starts_host, q_indices_by_head):
    """The first (lowest) query token of every item, q_indices_by_head[head][row_start + q_begin] (mirror of
    the Weave module's first_query_tokens_of_long_prefill_work; one gather + one copy per head)."""
    positions = [[] for _head in q_indices_by_head]
    slots = [[] for _head in q_indices_by_head]
    for index, (
        _group_count,
        (head, kv_block, q_begin, _q_count, _zero, _again),
    ) in enumerate(work):
        positions[head].append(int(row_starts_host[head][kv_block]) + int(q_begin))
        slots[head].append(index)
    first_tokens = [0] * len(work)
    for head, q_indices in enumerate(q_indices_by_head):
        if not positions[head]:
            continue
        gathered = q_indices.index_select(
            0, torch.tensor(positions[head], dtype=torch.int64, device=q_indices.device)
        )
        for index, token in zip(slots[head], gathered.cpu().tolist(), strict=False):
            first_tokens[index] = int(token)
    return first_tokens


def _long_prefill_order_work_q_resident(work, first_tokens, *, window, fill_groups):
    """Launch order of lever P1b (mirror of the Weave module's order_long_prefill_work_q_resident): the items
    with at least fill_groups groups sorted by (head, first_token // window, group_count desc, kv_block,
    q_begin), then the smaller items in bucket-descending (stable) order.  A permutation of work."""
    if len(first_tokens) != len(work):
        raise ValueError("one first query token per work item is required")
    if int(window) <= 0 or int(fill_groups) <= 0:
        raise ValueError("window and fill_groups must be positive")
    bulk = []
    fill = []
    for item, token in zip(work, first_tokens, strict=False):
        group_count, (head, kv_block, q_begin, _q_count, _zero, _again) = item
        if int(group_count) >= int(fill_groups):
            bulk.append(
                (
                    (
                        int(head),
                        int(token) // int(window),
                        -int(group_count),
                        int(kv_block),
                        int(q_begin),
                    ),
                    item,
                )
            )
        else:
            fill.append(item)
    bulk.sort(key=lambda pair: pair[0])
    fill.sort(key=lambda item: item[0], reverse=True)
    return [item for _key, item in bulk] + fill


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
    q_indices_by_head: list[torch.Tensor] = []
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
        q_indices_by_head.append(q_indices)

    # The host decomposes the per-row counts into the CTA work list: one device-to-host
    # transfer for all heads (the plan is rebuilt only when the selection changes).
    counts_host = torch.stack(counts_by_head).cpu().tolist()
    work = _long_prefill_decompose_work(
        counts_host, buckets=buckets, q_tokens_per_group=q_tokens_per_group
    )
    if not work:
        raise ValueError("long prefill requires at least one selected edge")
    if _LONG_WORK_TAIL_SPLIT:
        work = _long_prefill_tail_split_work(
            work, sm_count=int(num_sms), q_tokens_per_group=q_tokens_per_group
        )
    if _LONG_WORK_ORDER_Q_RESIDENT:
        row_starts_host = []
        for head_counts in counts_host:
            starts = [0]
            for count in head_counts:
                starts.append(starts[-1] + int(count))
            row_starts_host.append(starts)
        first_tokens = _long_prefill_first_query_tokens(
            work, row_starts_host=row_starts_host, q_indices_by_head=q_indices_by_head
        )
        work = _long_prefill_order_work_q_resident(
            work,
            first_tokens,
            window=_LONG_WORK_ORDER_Q_WINDOW,
            fill_groups=min(_LONG_WORK_ORDER_FILL_GROUPS, work_group_cap),
        )
    else:
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
    lengths: _RequestLengths,
    paged: bool,
    max_pages: int,
    num_sms: int,
    group_size: int,
) -> dict:
    """Per-request plans and zero-based 2-entry ``cu_seqlens`` of one long-prefill binding.

    Every plan is built from the request's slice of the caller's ``q2k_indices``
    (``[:, q_begin:q_end]``, a view: each head's rows stay contiguous, and the
    view shares the owner's version counter, so an in-place rewrite of the
    selection invalidates every request).  The state is keyed by the q2k
    binding and the request lengths; a mismatch rebuilds every plan in one
    eager call -- the only place this route synchronizes -- and forgets the
    workspace's warmed launches, which named the old plans.  The partial pools
    (``state["pools"]``) outlive a plan rebuild and are resized only when the
    longest request grows.
    """

    signature = (
        q2k_indices.device.type,
        q2k_indices.device.index,
        int(q2k_indices.data_ptr()),
        int(q2k_indices._version),
        tuple(q2k_indices.shape),
        tuple(lengths.q_lens),
        tuple(lengths.k_lens),
        paged,
        max_pages,
        num_sms,
        group_size,
    )
    if workspace is not None:
        state = workspace._long_prefill
    else:
        with _implicit_long_prefill_states_lock:
            state = _implicit_long_prefill_states.pop(signature, {})
            _implicit_long_prefill_states.clear()
            _implicit_long_prefill_states[signature] = state
    if state.get("signature") != signature:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "long-prefill plans must be warmed before CUDA graph capture"
            )
        pools = state.get("pools", {})
        state.clear()
        state["pools"] = pools
        if workspace is not None:
            workspace._warmed_launches.clear()
        plans = []
        cu_q_views = []
        cu_k_views = []
        several = len(lengths.q_lens) > 1
        q_begin = 0
        for len_q, len_k in zip(lengths.q_lens, lengths.k_lens, strict=True):
            plans.append(
                _build_long_prefill_plan(
                    q2k_indices[:, q_begin : q_begin + len_q],
                    total_k=max_pages * _BLOCK_SIZE if paged else len_k,
                    num_sms=num_sms,
                    group_size=group_size,
                    paged=paged,
                )
            )
            if several:
                # Zero-based 2-entry bounds of the request's slices; one request
                # binds the caller's tensors instead.
                cu_q_views.append(
                    torch.tensor(
                        [0, len_q], dtype=torch.int32, device=q2k_indices.device
                    )
                )
                cu_k_views.append(
                    torch.tensor(
                        [0, len_k], dtype=torch.int32, device=q2k_indices.device
                    )
                )
            q_begin += len_q
        state["signature"] = signature
        state["plans"] = plans
        state["cu_q"] = cu_q_views
        state["cu_k"] = cu_k_views
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
    cu_k: Optional[torch.Tensor],
    q_offsets: Optional[torch.Tensor],
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
    lengths: _RequestLengths,
) -> None:
    """Serve every request of the call on the selected-block long route.

    The producer and combine address every per-request quantity through the
    metadata ``batch`` field the host pins to 0 (``cu_seqlens_q[0]`` /
    ``cu_seqlens_k[0]`` offsets, ``total_q`` rows of ``partial_o``), so each
    request -- one request as much as several -- runs on dim-0 slices of
    ``q``/``k``/``v``/``out``/``lse`` (16 B aligned rows, contiguous views),
    zero-based 2-entry ``cu_seqlens``, its own plan, and ``partial_o`` (u8x32
    payload) / ``partial_scale`` (BF16x4 dequant scales) / ``partial_lse``
    views carved from pools sized by the longest request.  The
    sequence forward_0, combine_0, forward_1, ... on one stream makes the
    shared pools safe: the PDL-launched combine waits for its producer and the
    next forward is an ordinary launch behind the combine.  Every (forward,
    combine) pair records its own warmed-launch signature, so the eager warm
    call covers all requests and capture checks each in turn.  One request
    binds the caller's ``cu_seqlens`` / ``q_offsets`` / ``kv_lens`` tensors
    unchanged (its boundaries are never read; zero-based by contract); a
    paged ``cu_k`` of ``None`` (levers P-H1 / P-H1d) binds the all-zero
    carrier and a ``None`` ``q_offsets`` is bound to it.
    """

    total_q, num_q_heads, _ = (int(value) for value in q.shape)
    num_kv_heads = int(k.shape[1])
    q_lens, k_lens = lengths.q_lens, lengths.k_lens
    if lengths.q_first != 0 or lengths.k_first != 0:
        raise ValueError(
            "the long-prefill route addresses requests from row 0: cu_seqlens_q[0] "
            "and cu_seqlens_k[0] must be 0"
        )
    # Like the union route, rows beyond the last request are left untouched;
    # only requests that overrun their buffers are rejected.
    if sum(q_lens) > total_q:
        raise ValueError("cu_seqlens_q addresses more query rows than q holds")
    if not paged and sum(k_lens) > int(k.shape[0]):
        raise ValueError("cu_seqlens_k addresses more K/V rows than k holds")
    num_sms = _num_sms(q.device)
    state = _long_prefill_state(
        workspace=workspace,
        q2k_indices=q2k_indices,
        lengths=lengths,
        paged=paged,
        max_pages=max_pages,
        num_sms=num_sms,
        group_size=group_size,
    )
    pools = state["pools"]
    max_len_q = max(q_lens)
    # Block-scaled u8x32 partials (CAKE-1183 partial-byte round): one aligned
    # 128-byte u8 payload line per (split, token, head) row plus four BF16
    # dequant scales per row, written by the producer as two aligned pairs and
    # read back by the combine.  Both pools are sized by the longest request
    # and carved per request below.
    partial_o_pool = _long_state_tensor(
        pools,
        "partial_o_u8",
        (_TOPK * max_len_q * num_q_heads * _HEAD_DIM,),
        dtype=torch.uint8,
        device=q.device,
    )
    partial_scale_pool = _long_state_tensor(
        pools,
        "partial_scale_bf16x4",
        (_TOPK * max_len_q * num_q_heads * _LONG_PARTIAL_SEGMENT_COUNT,),
        dtype=torch.bfloat16,
        device=q.device,
    )
    partial_lse_pool = _long_state_tensor(
        pools,
        "partial_lse",
        (_TOPK * max_len_q * num_q_heads,),
        dtype=torch.float32,
        device=q.device,
    )
    partial_temperature_lse_pool = partial_lse_pool
    if return_temperature_lse:
        partial_temperature_lse_pool = _long_state_tensor(
            pools,
            "partial_temperature_lse",
            (_TOPK * max_len_q * num_q_heads,),
            dtype=torch.float32,
            device=q.device,
        )
    direct_group = (
        target == "sm100a" and paged and group_size == 16 and max_pages == 8192
    )
    route = _long_prefill_route(
        paged=paged, group_size=group_size, direct_group=direct_group
    )
    forward_program = _program(f"{route}:main", target)
    combine_program = _program(f"{route}:reduce", target)
    # Flat K/V: a never-dereferenced carrier, the caller's whole contiguous q2k
    # flattened (a view; a per-request slice is not flattenable without a copy).
    q2k_flat = q2k_indices.reshape(-1)
    single = len(q_lens) == 1
    if single and cu_k is None:
        # Lever P-H1d: the paged request binds the all-zero cu_seqlens_k
        # carrier; the producer reads kv_lens / page_table (the long route
        # admits paged K/V at 64 pages or more, never the flat-ABI form).
        cu_k = _paged_prefill_cu_seqlens_k(
            kv_lens, workspace=workspace, name="prefill_cu_seqlens_k"
        )
        if q_offsets is None:
            q_offsets = cu_k
    q_begin = 0
    k_begin = 0
    for index, (len_q, len_k) in enumerate(zip(q_lens, k_lens, strict=True)):
        plan = state["plans"][index]
        q_end = q_begin + len_q
        if single:
            cu_q_view, cu_k_view, q_offsets_view, kv_lens_view = (
                cu_q,
                cu_k,
                q_offsets,
                kv_lens,
            )
        else:
            cu_q_view = state["cu_q"][index]
            cu_k_view = q_offsets_view = state["cu_k"][index]
            kv_lens_view = kv_lens[index : index + 1] if paged else cu_k_view
        if paged:
            k_view, v_view = k, v
            page_table_view = page_table[index : index + 1]
        else:
            k_view = k[k_begin : k_begin + len_k]
            v_view = v[k_begin : k_begin + len_k]
            page_table_view = q2k_flat
        partial_o = partial_o_pool[: _TOPK * len_q * num_q_heads * _HEAD_DIM].view(
            _TOPK, len_q, num_q_heads, _HEAD_DIM
        )
        partial_scale = partial_scale_pool[
            : _TOPK * len_q * num_q_heads * _LONG_PARTIAL_SEGMENT_COUNT
        ].view(_TOPK, len_q, num_q_heads, _LONG_PARTIAL_SEGMENT_COUNT)
        partial_lse = partial_lse_pool[: _TOPK * len_q * num_q_heads].view(
            _TOPK, len_q, num_q_heads
        )
        partial_temperature_lse = partial_temperature_lse_pool[
            : _TOPK * len_q * num_q_heads
        ].view(_TOPK, len_q, num_q_heads)
        out_view = out[q_begin:q_end]
        forward_grid = (int(plan["work_count"]), 1, 1)
        forward = {
            "q": q[q_begin:q_end],
            "k": k_view,
            "v": v_view,
            "scheduler_metadata": plan["scheduler_metadata"],
            "k2q_row_ptr": plan["k2q_row_ptr"],
            "k2q_qsplit_indices": plan["k2q_qsplit_indices"],
            "partial_o": partial_o,
            "partial_scale": partial_scale,
            "partial_lse": partial_lse,
            "partial_temperature_lse": partial_temperature_lse,
            "out": out_view,
            "cu_seqlens_q": cu_q_view,
            "cu_seqlens_k": cu_k_view,
            "q_offsets": q_offsets_view,
            "kv_lens": kv_lens_view,
            "page_table": page_table_view,
            "total_q": len_q,
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
        combine_grid = ((len_q * num_q_heads + 31) // 32, 1, 1)
        combine = {
            "partial_o": partial_o,
            "partial_scale": partial_scale,
            "partial_lse": partial_lse,
            "partial_temperature_lse": partial_temperature_lse,
            "split_counts": plan["split_counts"],
            "out": out_view,
            "lse": lse[q_begin:q_end],
            "temperature_lse": temperature_lse[q_begin:q_end],
            "total_q": len_q,
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
                index,
            ),
            grid=forward_grid,
        )
        _check_warmed_launch(workspace, signature, capturing=capturing)
        forward_program.launch(forward_grid, **forward)
        combine_program.launch(combine_grid, **combine)
        _record_successful_launch(workspace, signature, capturing=capturing)
        q_begin = q_end
        k_begin += len_k


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
    program = _program(f"{route}:main", target)
    program.launch(grid, **arguments)
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
    cu_k: Optional[torch.Tensor],
    q_offsets: Optional[torch.Tensor],
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
    if cu_k is None:
        # Lever P-H1d: paged prefill without caller offsets binds the all-zero
        # cu_seqlens_k carrier (the program reads kv_lens / page_table); no
        # scan kernel precedes the launch.
        cu_k = _paged_prefill_cu_seqlens_k(
            kv_lens, workspace=workspace, name="prefill_cu_seqlens_k"
        )
        if max_pages == 0:
            # Flat-ABI form: the kernel derives kv_len from adjacent
            # cu_seqlens_k entries.  A page table without pages carries no
            # K/V, so the offsets of this degenerate call are all zero; write
            # them rather than trust the buffer's creation-time fill.
            cu_k.zero_()
        if q_offsets is None:
            q_offsets = cu_k
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
    """Run sparse prefill on compute capability 10.0 or 10.3.

    Route selection is input-derived (module docstring).  A call whose
    requests all qualify for the selected-block long route is served request
    by request.  With several requests the host needs their boundaries: they
    are read from ``cu_seqlens_q`` and ``cu_seqlens_k`` / ``seqused_k`` on
    every call without a workspace, and once per binding (same tensor
    objects, pointers, versions and ``q``/``k`` extents) with an
    ``MSASparseAttentionWorkspace``; ``cu_seqlens_q[0]`` and
    ``cu_seqlens_k[0]`` must be 0 on that route.  The union route reads no
    host lengths.
    """

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
        predicate = dict(
            paged=paged,
            group_size=group_size,
            max_pages=max_pages,
            q_dtype=q.dtype,
            k_dtype=k.dtype,
            v_dtype=v.dtype,
            causal=causal,
            q_offset_is_none=derive_q_offset,
            return_temperature_lse=return_temperature_lse,
            lse_temperature_scale=temperature_scale,
        )
        k_outer_dim = int(k.shape[0])
        lengths: Optional[_RequestLengths] = None
        if batch_size == 1:
            # One request: the lengths are the tensor extents; no device read.
            lengths = _RequestLengths((total_q,), (k_outer_dim,), 0, 0)
        elif (
            _long_prefill_candidate(**predicate)
            and total_q >= _LONG_MIN_REQUEST_LEN * batch_size
            and (paged or k_outer_dim >= _LONG_MIN_REQUEST_LEN * batch_size)
        ):
            # Several requests that may all qualify: read their boundaries
            # (once per workspace binding, never during capture) and decide
            # per request.
            lengths = _request_lengths(
                workspace,
                cu_q=cu_q,
                cu_k=cu_k,
                kv_lens=kv_lens,
                paged=paged,
                total_q=total_q,
                k_outer_dim=k_outer_dim,
                capturing=capturing,
            )
        use_long = lengths is not None and _use_long_prefill(
            q_lens=lengths.q_lens, k_lens=lengths.k_lens, **predicate
        )
        if (
            use_long
            and paged
            and group_size == 16
            and batch_size == 1
            and _union_census_route(
                workspace,
                q2k_indices=q2k_indices,
                q_len=total_q,
                kv_lens=kv_lens,
                max_pages=max_pages,
                capturing=capturing,
            )
        ):
            # Lever P10: same-block selections take the folded union program
            # (``prefill_union:...:paged:gqa16:causal_large``).
            use_long = False
        if use_long:
            _run_long_prefill(group_size=group_size, lengths=lengths, **common)
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
    """Select exact top-16 block indices on compute capability 10.0/10.3/10.7.

    One launch: the query-tiled radix-select program -- in its ``topk_select_keq`` instance (lever T9
    pass-0 all-equal fold) for a planned slice above the candidate cap (64) and at most 512 blocks
    (``_topk_radix_route``), its ``topk_select`` instance for a stage-only slice and beyond --
    or -- for small latency-bound multi-pass grids (``_topk_route``: at most 512 rows, at most 1024 valid
    pages, planned slice above the candidate cap) -- the one-CTA-per-row streaming program; all implement
    the same order (IEEE numeric, ties by ascending block index) and are bit-identical on every input.  Shapes whose plan splits the valid pages over ``S > 1`` K-slices
    use a per-device workspace (``partials`` u64, self-resetting ``counters``, ``block_max`` tile
    maxima) that is allocated by the first eager call for the largest shape seen; under CUDA-graph
    capture a missing warm-up or an insufficient workspace raises ``RuntimeError``.  The workspace is
    shared by every caller on the device: Top-K selects must be issued stream-ordered (the FlashInfer
    contract); two ``S > 1`` launches overlapping on different streams of one device would share
    ``counters`` and ``partials``.
    """

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
    route = _topk_route(total_q, num_heads, valid, max_k_tiles)
    if route == _TOPK_ROUTE_STREAMING:
        _program(f"{_TOPK_ROUTE_STREAMING}:main", target).launch(
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
    # The 32-query instance (lever T4) loads a lane's query pair with one 8-byte load: a score tensor whose
    # storage offset is only 4-byte aligned takes the 16-query instance (same selection, bit for bit).
    qpack = 2 if route == _TOPK_ROUTE_Q32 and max_score.data_ptr() % 8 == 0 else 1
    # (a 16-query launch runs the instance of the route key; the alignment fallback of a ``topk_select_q32`` shape
    # runs ``topk_select``, as the source dispatcher does)
    program_key = (
        _TOPK_ROUTE_Q32
        if qpack == 2
        else (_TOPK_ROUTE_KEQ if route == _TOPK_ROUTE_KEQ else _TOPK_ROUTE_RADIX)
    )
    plan = _topk_grid_plan(total_q, num_heads, valid, max_k_tiles, qpack)
    partials, counters, block_max = _topk_workspace(
        max_score.device,
        plan["groups"],
        plan["num_slices"],
        plan["bmax_words"],
        capturing,
        _TOPK_QT * qpack,
    )
    _program(f"{program_key}:main", target).launch(
        (plan["grid"], 1, 1),
        max_score=max_score,
        output=output,
        partials=partials,
        counters=counters,
        block_max=block_max,
        num_heads=num_heads,
        max_k_tiles=max_k_tiles,
        total_q=total_q,
        num_valid_pages=valid,
        force_begin_blocks=int(force_begin_blocks),
        force_end_blocks=int(force_end_blocks),
        num_qtiles=plan["num_qtiles"],
        num_slices=plan["num_slices"],
        slice_len=plan["slice_len"],
        index_bits=plan["index_bits"],
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
