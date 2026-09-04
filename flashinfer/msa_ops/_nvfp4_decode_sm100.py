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

NVFP4 paged-KV sparse decode for compute capability 10.0/10.3.

Compute capability 10.0/10.3 has no NVFP4 MSA route at all: the backend rejects
packed uint8 K/V and rejects tensor K/V scales, so a caller holding vLLM's
NVFP4 MSA cache cannot use the FlashInfer MSA path on those parts. This module
adds that capability.

Because there is nothing to fall back to, the guard is a *capability* guard,
not a benchmarked-shape allowlist: it constrains the model geometry the kernel
body is built from (64 query heads, 4 KV heads, head_dim 128, page_size 128,
topk 16) and the KV layout, and is parametric in batch size, per-request KV
length, block-table width, query length and causality. Narrowing any of those
would not route a call somewhere slower, it would make the capability
unreachable.

Anything outside the capability surface falls through untouched: when
:func:`check_surface` reports a reason, the caller reaches the unchanged
backend, which validates and rejects the call exactly as it did before.

TWO KERNELS, ONE ROUTE
----------------------

Inside the capability surface the route dispatches one of two implementations,
and the choice is INTERNAL: it changes which kernel runs, never what the API
accepts. :func:`check_surface` is unchanged and is still the only thing that
decides whether this route serves a call at all.

* ``flashinfer.msa_ops.cute_dsl.sparse_decode_nvfp4_sm100`` -- a CuTe-DSL
  implementation whose instantiations are specialised for one geometry and a
  small set of split counts. Where it applies it is materially faster.
* ``csrc/msa_decode_nvfp4_specialized.cu`` -- the warp-specialised ping-pong
  kernel, which has no cliff: it serves every batch size on its specialised
  path and is FASTER at batch 1 than at batch 8.

The specialised implementation is entered only where its own dispatch
arithmetic selects one of its specialised instantiations. That predicate is
NOT restated here -- :func:`~flashinfer.msa_ops.cute_dsl.sparse_decode_nvfp4_sm100.specialised_reason`
is the single copy, and this module asks it. Two copies of a routing predicate
is how a route stops matching the kernel it routes to in silence.

The predicate is not a threshold in batch size. At the deployment geometry
(64 query heads, 4 KV heads, top-k 16, page 128, one decode token, causal) the
number of CTAs a call is cut into is chosen to fill the machine, capped at the
top-k, capped again where the grid stops fitting one resident wave of its own
clusters, and then reduced to the largest value that has a specialised
instantiation. That last reduction is what makes the covered set complete: the
value the fill target asks for takes EVERY integer in its range, and only a few
of them have a binary, so before it batches 1-7 and 10-15 -- three capture
rungs among them -- ran the ping-pong kernel. Nothing about that was a
capability statement: both kernels compute the same function on all of them.

The covered set is ``[1, 256]``. It is enumerated from ``plan()`` rather than
written down -- :func:`msa_decode_nvfp4_specialized_stats` reports
``batch_spans_at_geometry`` by executing that arithmetic, so a change to the
thresholds moves the reported spans with it.

``FLASHINFER_MSA_DECODE_NVFP4_ROUTE`` overrides the choice for A/B measurement
and for tests -- ``auto`` (default), ``pingpong``, or ``specialised`` (which
RAISES rather than falling back, so a test can prove the guard).

THE WIN IS A CAPTURED-GRAPH WIN
-------------------------------

Measured on GB300, same process, same tensors, medians of three alternated
passes. Device time and captured-replay wall time both favour the CuTe-DSL
implementation on every shape it serves. EAGER wall time does not, and by a
wide margin:

===========  ==========  =============  =============  ==============
shape        device us   eager wall us  graph wall us  ping-pong eager
===========  ==========  =============  =============  ==============
batch 8          7.44         182.4          15.90          55.0
batch 16         9.60         180.2          17.15          59.4
batch 24        11.62         171.8          18.94          53.2
batch 32        14.08         179.9          22.27          56.3
batch 128       44.54         194.2          52.99          64.5
===========  ==========  =============  =============  ==============

The CuTe-DSL launch path costs 88-100 us of host time per call before the
kernel starts -- nearly flat in batch, so it is argument marshalling rather
than work -- against 55-66 us for the C++ binding, and the layering here adds
its second validation on top. CUDA-graph replay erases
all of it, and a serving engine captures its decode graphs, so the operative
number is the graph column. But an eager decode step -- capture disabled, or a
batch size outside the captured rungs -- pays that difference ONCE PER LAYER,
and it is three to fifteen times the kernel's own device time.

Nothing here acts on that. Routing on ``is_current_stream_capturing()`` would
put a different kernel in the captured graph than in the eager warm-up that
precedes it, which is a behaviour change with its own numerics to measure, and
the two implementations already differ by more than either differs from the
FP32 reference. It is stated so the next serving measurement starts from it
rather than rediscovering it.
"""

from __future__ import annotations

import functools
import json
import os
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch

from ..jit.core import logger

_WORKLOAD_PACKAGE = "flashinfer.msa_ops"
_WORKLOAD_FILE = "msa_decode_nvfp4_specialized_workloads.json"
# `topk` is deliberately NOT one of these any more. It was a compile-time
# constant of the kernel body until the selection row's strides were threaded
# through the signature; the parametric family now reads it at runtime, so it
# belongs in `parametric_axes` with its executed bound, not in a whitelist of
# geometries.
_WORKLOAD_FIELDS = (
    "num_qo_heads",
    "num_kv_heads",
    "head_dim",
    "page_size",
)
# Route override. Read at call time, never at import, so one process can
# measure both implementations on the same tensors.
_ROUTE_ENV = "FLASHINFER_MSA_DECODE_NVFP4_ROUTE"
_ROUTE_CHOICES = ("auto", "pingpong", "specialised")

_HEAD_DIM = 128
_PAGE_SIZE = 128
_NUM_QO_HEADS = 64
_NUM_KV_HEADS = 4
# The top-k the PINNED family and the CuTe-DSL `scored_geom` fast path bake in.
# Not a capability bound: any other value in [1, _MAX_TOPK] is served by the
# parametric family instead.
_TOPK = 16
# Structural ceiling of the parametric family: every selection slot is one lane
# of warp 0's ballot in `csrc/msa_decode_nvfp4_specialized.cu`, and the two
# compaction arrays are sized by it. Mirrored from `general::kSelectedCapacity`
# and cross-checked against the binding by a device test, so the two copies
# cannot drift.
_MAX_TOPK = 32
_SCALE_VEC = 16
_DATA_DIM = _HEAD_DIM // 2
_SCALE_DIM = _HEAD_DIM // _SCALE_VEC

_DATA_HEAD_STRIDE = _PAGE_SIZE * _DATA_DIM
_SCALE_HEAD_STRIDE = _PAGE_SIZE * _SCALE_DIM
_K_SCALE_BYTE_OFFSET = _NUM_KV_HEADS * _DATA_HEAD_STRIDE
_V_DATA_BYTE_OFFSET = _K_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE
_V_SCALE_BYTE_OFFSET = _V_DATA_BYTE_OFFSET + _K_SCALE_BYTE_OFFSET
_PAGE_BYTES = _V_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE

# Block-table row width of the deployment the kernel's PINNED instantiation
# family is compiled for: ceil(max_model_len 16384 / page_size 128). It is NOT
# a capability bound -- the block-table width is a runtime argument and any
# other width is served, just by the parametric family instead.
_PINNED_MAX_BLOCKS = 128

_SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm100a", (10, 3): "sm103a"}

# e2m1 value table indexed by the 4-bit code; even elements occupy the low
# nibble of each packed byte.
_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)

# Peak-memory budget for one chunk of the composable reference gather.
_REFERENCE_CHUNK_BYTES = 96 << 20

try:
    from ..jit.msa_decode_nvfp4_specialized import (
        load_msa_decode_nvfp4_specialized_module,
    )

    _SPECIALIZED_AVAILABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - unbuildable environment
    _SPECIALIZED_AVAILABLE = False
    load_msa_decode_nvfp4_specialized_module = None

_dispatch_count = 0
# Dispatches that took the CuTe-DSL specialised implementation, and the reasons
# calls did not. A route that silently stops taking its fast path is a
# performance regression with no failing test attached to it, so the split is
# reported rather than inferred (mirrors the pinned/general split below).
_specialised_dispatch_count = 0
_specialised_decline_counts: Dict[str, int] = {}
_specialised_warm_devices: Set[Tuple[str, Optional[int]]] = set()
# Devices whose specialised warm FAILED. Latched, because the decode hook calls
# warm() on every eager dispatch and re-attempting a failed compile there would
# turn a missing optional accelerator into minutes of ptxas per call.
_specialised_warm_failed: Set[Tuple[str, Optional[int]]] = set()
_specialised_import_error: Optional[str] = None
# Split by instantiation family. The pinned family is the one the kernel-level
# figure measures; the general one is correct but untimed. A deployment whose
# geometry drifts out of the pinned envelope keeps working and gets slower with
# nothing failing, so the split is reported rather than inferred.
_pinned_dispatch_count = 0
_general_dispatch_count = 0
# Launches this module issues for itself in :func:`warm`. Kept apart from
# ``_dispatch_count`` so that "the route served this call" stays a statement
# about callers: the kernel-level A/B harness attests dispatch by differencing
# ``dispatch_count`` around a call, and a self-issued warm launch is not one.
_warm_dispatch_count = 0
_reject_counts: Dict[int, int] = {}
_warmed_devices: Set[Tuple[str, Optional[int]]] = set()


def _reject(reason: str) -> str:
    """Record and return a guard rejection, logging each site once.

    A bare ``return reason`` at any of the ~30 predicates below makes a silent
    non-dispatch indistinguishable from a bug, so every site is counted and the
    first occurrence is logged with its source line.
    """
    line = sys._getframe(1).f_lineno
    seen = _reject_counts.get(line, 0)
    _reject_counts[line] = seen + 1
    if seen == 0:
        logger.info_once(
            "msa_decode_nvfp4_specialized: guard rejected at %s:%d (%s)",
            __name__,
            line,
            reason,
        )
    return reason


@functools.cache
def _read_workload_file() -> Dict[str, Any]:
    """Read the allowlist from the installed package or the source checkout.

    Mirrors ``_get_blackwell_msa_csrc_dir``: an install that does not expose
    the package through ``importlib.resources`` still has the file next to this
    module, and a silently empty allowlist would disable the route.
    """
    try:
        text = resources.files(_WORKLOAD_PACKAGE).joinpath(_WORKLOAD_FILE).read_text()
    except (
        AttributeError,
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        TypeError,
        ValueError,
    ):
        text = (Path(__file__).resolve().parent / _WORKLOAD_FILE).read_text()
    return json.loads(text)


@functools.cache
def _load_allowlist() -> frozenset:
    try:
        payload = _read_workload_file()
        fields = tuple(payload["fields"])
        if fields != _WORKLOAD_FIELDS:
            raise ValueError(f"unexpected workload fields: {fields}")
        return frozenset(
            tuple(int(value) for value in row) for row in payload["workloads"]
        )
    except (
        AttributeError,
        FileNotFoundError,
        ModuleNotFoundError,
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning_once(
            "Unable to load the specialized NVFP4 MSA decode allowlist: %s",
            type(exc).__name__,
        )
        return frozenset()


def _module_is_loaded() -> bool:
    if not _SPECIALIZED_AVAILABLE:
        return False
    return load_msa_decode_nvfp4_specialized_module.cache_info().currsize > 0


def _is_capturing() -> bool:
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:  # pragma: no cover - no CUDA context
        return False


def _target_for(device: torch.device) -> Optional[str]:
    try:
        capability = torch.cuda.get_device_capability(device)
    except RuntimeError:  # pragma: no cover - no CUDA context
        return None
    return _SUPPORTED_COMPUTE_CAPABILITIES.get(capability)


# vLLM's ``nvfp4_split_data_scale`` hands the block-scale regions back as
# float8_e4m3fn views and the data regions as uint8; both are the same bytes,
# so either spelling of the scale tensors is accepted and reinterpreted as
# bytes before the kernel sees it.
_SCALE_DTYPES = (torch.uint8, torch.float8_e4m3fn)


def as_scale_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Reinterpret an e4m3 block-scale view as the raw bytes, no copy."""
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _page_region_reason(
    tensor: torch.Tensor,
    name: str,
    num_pages: int,
    inner: int,
    head_stride: int,
    dtypes: Tuple[torch.dtype, ...] = (torch.uint8,),
) -> Optional[str]:
    if tensor.dtype not in dtypes:
        expected = " or ".join(str(dtype) for dtype in dtypes)
        return f"{name} must be {expected}, got {tensor.dtype}"
    if tensor.ndim != 4:
        return f"{name} must be 4-D, got {tensor.ndim}-D"
    shape = tuple(int(value) for value in tensor.shape)
    if shape != (num_pages, _NUM_KV_HEADS, _PAGE_SIZE, inner):
        return f"{name} shape {shape} is not (num_pages, 4, 128, {inner})"
    # The kernel derives every byte address from these strides rather than
    # reading them, so they are asserted, not propagated.
    stride = tuple(int(value) for value in tensor.stride())
    if stride != (_PAGE_BYTES, head_stride, inner, 1):
        return (
            f"{name} stride {stride} is not ({_PAGE_BYTES}, {head_stride}, {inner}, 1)"
        )
    return None


def check_surface(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    seqlen_q: int,
    causal: bool,
    return_softmax_lse: bool,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    k_global_scale: Optional[float],
    v_global_scale: Optional[float],
    q_offset,
    force_fused: Optional[bool],
) -> Optional[str]:
    """Return ``None`` when this call belongs to the NVFP4 decode route.

    Ordering is deliberate: semantics, then layout, then device and
    architecture LAST, so the whole semantic and layout surface of the guard is
    exercisable on a host with no GPU.
    """

    # ---- semantics ------------------------------------------------------
    if not isinstance(k, torch.Tensor) or k.dtype != torch.uint8:
        return _reject("k is not a packed NVFP4 (uint8) tensor")
    if not isinstance(v, torch.Tensor) or v.dtype != torch.uint8:
        return _reject("v is not a packed NVFP4 (uint8) tensor")
    if k_scale is None or v_scale is None:
        return _reject("NVFP4 KV requires k_scale and v_scale")
    if k_global_scale is None or v_global_scale is None:
        return _reject("NVFP4 KV requires k_global_scale and v_global_scale")
    if q.dtype != torch.bfloat16:
        return _reject(f"q must be bfloat16 on this route, got {q.dtype}")
    if int(seqlen_q) < 1:
        return _reject(f"seqlen_q must be positive, got {seqlen_q}")
    if return_softmax_lse:
        return _reject("this route does not produce a softmax LSE")
    if q_offset is not None:
        return _reject("this route does not accept an explicit q_offset")
    if force_fused not in (None, True, False):
        return _reject("force_fused must be True, False, or None")
    # force_fused is accepted for API compatibility and ignored, as the
    # documented compute-capability 10.0/10.3 contract says: this route writes
    # the final output directly and never splits into GMEM partials.
    if page_table is None or seqused_k is None:
        return _reject("this route serves the paged layout and needs seqused_k")
    if cu_seqlens_k is not None:
        return _reject("this route does not accept cu_seqlens_k")

    # ---- shapes ---------------------------------------------------------
    if q.ndim != 3 or not q.is_contiguous():
        return _reject("q must be a contiguous 3-D (total_q, num_qo_heads, head_dim)")
    total_q, num_qo_heads, head_dim = (int(value) for value in q.shape)
    if (num_qo_heads, head_dim) != (_NUM_QO_HEADS, _HEAD_DIM):
        return _reject(
            f"this route serves 64 query heads at head_dim 128, "
            f"got {num_qo_heads}x{head_dim}"
        )
    if total_q < 1 or total_q % int(seqlen_q):
        return _reject(f"q rows ({total_q}) must be batch_size * seqlen_q ({seqlen_q})")
    batch_size = total_q // int(seqlen_q)

    num_pages = int(k.shape[0]) if k.ndim == 4 else -1
    for tensor, name, inner, head_stride, dtypes in (
        (k, "k", _DATA_DIM, _DATA_HEAD_STRIDE, (torch.uint8,)),
        (v, "v", _DATA_DIM, _DATA_HEAD_STRIDE, (torch.uint8,)),
        (k_scale, "k_scale", _SCALE_DIM, _SCALE_HEAD_STRIDE, _SCALE_DTYPES),
        (v_scale, "v_scale", _SCALE_DIM, _SCALE_HEAD_STRIDE, _SCALE_DTYPES),
    ):
        reason = _page_region_reason(
            tensor, name, num_pages, inner, head_stride, dtypes
        )
        if reason is not None:
            return _reject(reason)

    if (
        q2k_indices.dtype != torch.int32
        or q2k_indices.ndim != 3
        or tuple(int(value) for value in q2k_indices.shape[:2])
        != (_NUM_KV_HEADS, total_q)
    ):
        return _reject(
            f"q2k_indices must be int32 (4, total_q, topk) with total_q={total_q}"
        )
    topk = int(q2k_indices.shape[2])
    if not 1 <= topk <= _MAX_TOPK:
        return _reject(
            f"top-k must be in [1, {_MAX_TOPK}] -- every selection slot is one "
            f"lane of the compaction ballot -- got {topk}"
        )
    # ONLY the innermost dimension has to be dense. The two outer strides are
    # passed to the kernel, so the transposed view of a token-major
    # (total_q, num_kv_heads, topk) selection buffer -- what the MSA indexer
    # actually writes -- is read in place instead of being copied contiguous.
    if int(q2k_indices.stride(2)) != 1:
        return _reject(
            "q2k_indices must be dense in its innermost (top-k) dimension, "
            f"got stride {int(q2k_indices.stride(2))}"
        )
    if int(q2k_indices.stride(0)) < 0 or int(q2k_indices.stride(1)) < 0:
        return _reject("q2k_indices must not be negatively strided")
    if (
        (_NUM_KV_HEADS - 1) * int(q2k_indices.stride(0))
        + (total_q - 1) * int(q2k_indices.stride(1))
        + topk
    ) > 0x7FFFFFFF:
        return _reject("the q2k_indices view is too large for 32-bit addressing")
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or not page_table.is_contiguous()
        or int(page_table.shape[0]) != batch_size
        or int(page_table.shape[1]) < 1
    ):
        return _reject(
            "page_table must be contiguous int32 (batch_size, max_blocks) with a "
            "positive row width"
        )
    if (
        seqused_k.dtype != torch.int32
        or seqused_k.ndim != 1
        or not seqused_k.is_contiguous()
        or int(seqused_k.shape[0]) != batch_size
    ):
        return _reject("seqused_k must be contiguous int32 (batch_size,)")

    # ---- layout proof ---------------------------------------------------
    # Shape, dtype and stride cannot distinguish four views of one planar page
    # from four unrelated allocations strided the same way, and the (4, 4)
    # V-scale swizzle is invisible to all three. The byte offsets between the
    # four base pointers are the property that actually pins the inputs to the
    # page map the cache writer used, so they are checked here and asserted
    # again in the kernel binding.
    base = k.data_ptr()
    for tensor, name, offset in (
        (k_scale, "k_scale", _K_SCALE_BYTE_OFFSET),
        (v, "v", _V_DATA_BYTE_OFFSET),
        (v_scale, "v_scale", _V_SCALE_BYTE_OFFSET),
    ):
        if tensor.data_ptr() - base != offset:
            return _reject(
                f"{name} is not the +{offset} B region of the same packed page as k "
                f"(delta {tensor.data_ptr() - base})"
            )

    # ---- allowlist ------------------------------------------------------
    signature = (num_qo_heads, _NUM_KV_HEADS, head_dim, _PAGE_SIZE)
    if signature not in _load_allowlist():
        return _reject(f"geometry {signature} is not in the capability allowlist")

    # ---- device / architecture (LAST) -----------------------------------
    if not q.is_cuda:
        return _reject("q must be a CUDA tensor")
    for tensor, name in (
        (k, "k"),
        (v, "v"),
        (k_scale, "k_scale"),
        (v_scale, "v_scale"),
        (q2k_indices, "q2k_indices"),
        (page_table, "page_table"),
        (seqused_k, "seqused_k"),
    ):
        if tensor.device != q.device:
            return _reject(f"{name} must be on the same device as q")
    if _target_for(q.device) is None:
        return _reject("this route requires compute capability 10.0 or 10.3")
    return None


def capture_requires_workspace() -> bool:
    """Whether CUDA graph capture of this route needs a caller-owned workspace.

    ``False``, and that is a property of the route rather than a relaxation.
    Everything between entering the hook and the launch is host-side arithmetic
    over shapes, strides and dtypes: no device value is read back, no memory is
    allocated below Python, no persistent state carries between calls, and the
    kernel is a single ``cudaLaunchKernelEx`` whose only non-launch neighbours
    are two ``cudaFuncSetAttribute`` calls that are not stream-ordered. A graph
    that captures that launch replays it verbatim, so pointer stability is
    whatever the capturing engine's own memory pool already guarantees.

    That holds for BOTH implementations the route dispatches. The specialised
    one adds route selection -- integer arithmetic over the same shapes -- and
    then one launch; its call path neither allocates nor compiles (see
    ``run`` in ``cute_dsl/sparse_decode_nvfp4_sm100.py``), and capture is
    exercised on both sides of the route by
    ``test_a_captured_graph_replays_both_sides_of_the_route``. It holds no
    persistent device memory at all, so there is nothing for a replay to find
    freed.

    The general SM100/SM103 MSA paths do need
    :class:`MSASparseAttentionWorkspace` -- they carry split state and
    temporaries across calls -- and it stays available here for callers that
    want the warm/capture identity check. But *requiring* it would make the
    route unusable by a serving engine: the workspace admits exactly one
    capture (it latches ``_captured``) while vLLM captures one graph per decode
    shape, and its warm-vs-capture identity includes ``data_ptr()`` for nine
    tensors while vLLM captures activations out of a separate graph pool, so
    the warm and capture pointers can never agree. Neither is fixable by the
    consumer.

    What capture *does* still require is that the kernel has been launched once
    eagerly on the device -- see :func:`warm`.
    """

    return False


def check_specialized(device: torch.device) -> Optional[str]:
    """Return ``None`` when the kernel may serve a call on ``device``."""

    if not _SPECIALIZED_AVAILABLE:
        return "the NVFP4 MSA decode module is unavailable"
    if _is_capturing():
        # Building a module or touching the driver for the first time inside a
        # capture region is illegal.
        if not _module_is_loaded():
            return "CUDA graph capture before the module was built"
        if (device.type, device.index) not in _warmed_devices:
            return "CUDA graph capture before the first eager dispatch on this device"
    return None


# ---------------------------------------------------------------------------
# the second implementation, and the predicate that selects it
# ---------------------------------------------------------------------------
def _route_choice() -> str:
    """``auto`` unless overridden. Read at call time, never at import."""

    value = (os.environ.get(_ROUTE_ENV) or "auto").strip().lower()
    if value not in _ROUTE_CHOICES:
        raise ValueError(f"{_ROUTE_ENV}={value!r} is not one of {list(_ROUTE_CHOICES)}")
    return value


@functools.cache
def _specialised_module():
    """Import the CuTe-DSL implementation, or ``None`` if it is unavailable.

    Lazy and cached: importing it costs a ``cutlass`` import, and this route is
    reached on architectures and in builds where that package need not exist.
    An import failure is recorded and degrades to the ping-pong kernel, which
    serves every shape this route accepts -- so a missing optional dependency
    costs speed on some batch sizes and nothing else.
    """

    global _specialised_import_error
    try:
        from .cute_dsl import sparse_decode_nvfp4_sm100 as module

        return module
    except Exception as exc:  # noqa: BLE001 - optional dependency probe
        _specialised_import_error = f"{type(exc).__name__}: {str(exc)[:300]}"
        logger.warning_once(
            "The specialised NVFP4 MSA decode implementation is unavailable "
            "(%s); the route continues on the ping-pong kernel, which serves "
            "every shape this route accepts.",
            _specialised_import_error,
        )
        return None


def specialised_route_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    q2k_indices: torch.Tensor,
    seqlen_q: int,
    causal: bool,
    softmax_scale: float,
    k_global_scale: float,
    device_warm: bool = True,
) -> Optional[str]:
    """``None`` when the CuTe-DSL implementation serves this call.

    Every condition is answered by the implementation itself
    (``specialised_reason``) except the two this module owns: whether the
    module could be imported, and whether it has been warmed on this device.
    Nothing here re-derives a geometry the other file already decides.
    """

    module = _specialised_module()
    if module is None:
        return f"the specialised implementation is unavailable: {_specialised_import_error}"
    num_qo_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[1])
    reason = module.specialised_reason(
        total_q=int(q.shape[0]),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        grp=num_qo_heads // num_kv_heads if num_kv_heads else 0,
        topk=int(q2k_indices.shape[2]),
        page_size=int(k.shape[2]),
        seqlen_q=int(seqlen_q),
        causal=int(bool(causal)),
        softmax_scale=float(softmax_scale),
        k_global_scale=float(k_global_scale),
    )
    if reason is not None:
        return reason
    if device_warm and not module.is_warm(q.device):
        # The call path of that implementation never compiles, by design, so a
        # cold device is a route-away rather than a raise: the ping-pong kernel
        # is already warm and already correct here.
        return f"not warmed on {q.device}"
    return None


def _specialised_warm_shapes() -> Tuple[int, ...]:
    """One batch size per compiled instantiation, smallest of each.

    Derived from the implementation's own plan(), not tabulated: the first
    launch of an instantiation is what makes the driver resolve it, and a
    CUDA-graph capture may not be the first. A hard-coded list here would go
    stale the moment the split thresholds move.
    """

    module = _specialised_module()
    if module is None:
        return ()
    seen: Dict[int, int] = {}
    for batch in range(1, 257):
        plan = module.plan(
            total_q=batch,
            num_qo_heads=_NUM_QO_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            grp=_NUM_QO_HEADS // _NUM_KV_HEADS,
            topk=_TOPK,
            page_size=_PAGE_SIZE,
            seqlen_q=1,
            causal=1,
        )
        if plan["specialised"]:
            seen.setdefault(plan["kernel_idx"], batch)
    return tuple(sorted(seen.values()))


def _normalize_cuda_device(device: torch.device) -> torch.device:
    """Give ``device`` the concrete index a CUDA tensor's ``.device`` carries.

    ``check_specialized`` keys on ``(type, index)`` and every CUDA tensor
    reports a concrete index, so a ``cuda``-without-index warm request has to
    resolve to the same key or it would record a device nothing ever matches.
    """

    if device.type == "cuda" and device.index is None:
        return torch.device("cuda", torch.cuda.current_device())
    return device


def _warm_inputs(device: torch.device) -> Dict[str, Any]:
    """The smallest call that satisfies :func:`check_surface` exactly.

    One page (73,728 B) and one decode token. Zero bytes are a valid NVFP4
    encoding -- e2m1 code 0 and an e4m3 scale of 0 -- so the launch is defined
    without quantizing anything; the output is discarded. The four K/V views
    are cut out of one allocation at the documented byte offsets because the
    guard proves that relationship and the kernel asserts it again.
    """

    pool = torch.zeros(_PAGE_BYTES, dtype=torch.uint8, device=device)
    data_shape = (1, _NUM_KV_HEADS, _PAGE_SIZE, _DATA_DIM)
    scale_shape = (1, _NUM_KV_HEADS, _PAGE_SIZE, _SCALE_DIM)
    data_stride = (_PAGE_BYTES, _DATA_HEAD_STRIDE, _DATA_DIM, 1)
    scale_stride = (_PAGE_BYTES, _SCALE_HEAD_STRIDE, _SCALE_DIM, 1)
    q = torch.zeros((1, _NUM_QO_HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device)
    # Block 0 selected once, the rest of the top-k tail-padded with -1, exactly
    # as msa_topk_select documents its output.
    q2k_indices = torch.full(
        (_NUM_KV_HEADS, 1, _TOPK), -1, dtype=torch.int32, device=device
    )
    q2k_indices[:, :, 0] = 0
    return dict(
        q=q,
        k=torch.as_strided(pool, data_shape, data_stride, 0),
        v=torch.as_strided(pool, data_shape, data_stride, _V_DATA_BYTE_OFFSET),
        k_scale=torch.as_strided(pool, scale_shape, scale_stride, _K_SCALE_BYTE_OFFSET),
        v_scale=torch.as_strided(pool, scale_shape, scale_stride, _V_SCALE_BYTE_OFFSET),
        q2k_indices=q2k_indices,
        page_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        seqused_k=torch.full((1,), _PAGE_SIZE, dtype=torch.int32, device=device),
        out=torch.empty_like(q),
        seqlen_q=1,
        causal=True,
        softmax_scale=float(_HEAD_DIM**-0.5),
        k_global_scale=1.0,
        v_global_scale=1.0,
    )


def _specialised_warm_inputs(device: torch.device, batch: int) -> Dict[str, Any]:
    """:func:`_warm_inputs` widened to ``batch`` requests over the same page.

    Every request selects block 0 of a one-page pool, so the launch is defined
    on zero bytes exactly as the single-request form is; only the batch size,
    which is what selects the instantiation, changes.
    """

    base = _warm_inputs(device)
    q = torch.zeros(
        (batch, _NUM_QO_HEADS, _HEAD_DIM), dtype=torch.bfloat16, device=device
    )
    q2k = torch.full(
        (_NUM_KV_HEADS, batch, _TOPK), -1, dtype=torch.int32, device=device
    )
    q2k[:, :, 0] = 0
    inputs = dict(base)
    inputs.update(
        q=q,
        q2k_indices=q2k,
        page_table=torch.zeros((batch, 1), dtype=torch.int32, device=device),
        seqused_k=torch.full((batch,), _PAGE_SIZE, dtype=torch.int32, device=device),
        out=torch.empty_like(q),
    )
    return inputs


def _specialised_warm(device: torch.device) -> None:
    """Compile and first-launch every instantiation the route can reach.

    Two separate preconditions, and both are this function's job:

    * the implementation's call path never compiles, so its variant table has
      to be built here or ``run`` on that side declines;
    * the first LAUNCH of an instantiation is what makes the driver resolve it
      (``CUDA_MODULE_LOADING=LAZY``), and that may not happen inside a
      CUDA-graph capture -- so one eager launch per compiled instantiation is
      taken here, on the smallest batch that selects it.

    Compilation is minutes of ptxas on a cold cache. It happens once per
    device, in the eager phase a serving engine runs before it captures, and
    ``FLASHINFER_MSA_DECODE_NVFP4_ROUTE=pingpong`` skips it entirely.
    Failure is not fatal: the ping-pong kernel serves every shape this route
    accepts, so a failed warm costs speed on some batch sizes and nothing else.
    """

    key = (device.type, device.index)
    if key in _specialised_warm_devices or key in _specialised_warm_failed:
        return
    module = _specialised_module()
    if module is None:
        return
    try:
        module.warmup(device)
        for batch in _specialised_warm_shapes():
            inputs = _specialised_warm_inputs(device, batch)
            module.run(
                inputs["q"],
                inputs["k"],
                inputs["v"],
                inputs["k_scale"],
                inputs["v_scale"],
                inputs["q2k_indices"],
                inputs["page_table"],
                inputs["seqused_k"],
                1,
                True,
                float(_HEAD_DIM**-0.5),
                1.0,
                1.0,
                inputs["out"],
            )
    except Exception as exc:  # noqa: BLE001 - optional accelerator
        logger.warning_once(
            "Unable to warm the specialised NVFP4 MSA decode implementation: "
            "%s: %s -- the route continues on the ping-pong kernel, which "
            "serves every shape it accepts.",
            type(exc).__name__,
            str(exc)[:400],
        )
        _specialised_warm_failed.add(key)
        return
    _specialised_warm_devices.add(key)


def warm(device: torch.device | str) -> None:
    """Make ``device`` ready for CUDA graph capture of this route.

    Two things have to have happened before a capture region can contain this
    kernel: the module must be built, and the kernel must have been *launched*
    at least once on the device, because the first launch is what makes the
    driver resolve a lazily loaded module (``CUDA_MODULE_LOADING=LAZY`` is the
    default) and set the two function attributes. So this does both -- it
    builds and then dispatches one 73,728-byte single-token call through EACH
    instantiation family, parametric and pinned -- and records the device,
    which is the precondition :func:`check_specialized` names.

    Building alone would not do it: that is precisely the bug this replaces.
    ``check_specialized`` refused capture until ``_warmed_devices`` contained
    the device, ``_warmed_devices`` was only ever written by a real dispatch,
    and the RuntimeError told the caller to run this function -- so the remedy
    the error named could not clear the error, and the route worked only where
    something else happened to dispatch eagerly first.

    Idempotent: after the first success this is a set lookup, which matters
    because the decode hook calls it on every eager dispatch.
    """

    if not _SPECIALIZED_AVAILABLE:
        return
    device = torch.device(device)
    if device.type != "cuda":
        return
    if _is_capturing():
        # Do not record the device: retry on the next eager call.
        return
    device = _normalize_cuda_device(device)
    if (device.type, device.index) in _warmed_devices:
        # The route is warm; the optional half may not be, because it is warmed
        # after the route and may have been skipped by an override that has
        # since changed. Idempotent and latched, so this is a set lookup once
        # it has settled either way.
        if _route_choice() != "pingpong":
            _specialised_warm(device)
        return
    target = _target_for(device)
    if target is None:
        return
    global _warm_dispatch_count
    try:
        module = load_msa_decode_nvfp4_specialized_module(target)
        # Both instantiation families, because a captured decode takes the
        # pinned one and the very first launch of a family is what makes the
        # driver resolve it. `_warm_inputs` is called with exactly one argument
        # so a GPU-free gate can redirect it; the pinned shape is derived from
        # its result by widening the block table, which is the only axis that
        # separates the two here.
        general_inputs = _warm_inputs(device)
        _dispatch(module, **general_inputs)
        pinned_inputs = dict(general_inputs)
        pinned_inputs["page_table"] = torch.zeros(
            (general_inputs["page_table"].shape[0], _PINNED_MAX_BLOCKS),
            dtype=torch.int32,
            device=general_inputs["q"].device,
        )
        pinned_inputs["out"] = torch.empty_like(general_inputs["q"])
        _dispatch(module, **pinned_inputs)
    except (ImportError, OSError, RuntimeError) as exc:
        # torch.cuda.OutOfMemoryError is a RuntimeError, so the 106 KB the dummy
        # needs is covered here too.
        # Stale text here is not cosmetic: this line is what a failed build
        # reports, and it must say what actually happens next. Nothing serves
        # the call -- `run` raises, by design, rather than substituting
        # anything slower.
        logger.warning_once(
            "Unable to warm the NVFP4 MSA decode kernel: %s -- every call on "
            "this route will now RAISE, because this kernel is the only "
            "implementation of it. Full error: %s",
            type(exc).__name__,
            str(exc)[:512],
        )
        return
    _warm_dispatch_count += 2
    _warmed_devices.add((device.type, device.index))
    # After the route itself is warm, because this one is optional: if it
    # cannot be warmed the route still works, and `_warmed_devices` must
    # already say so or capture would be refused for the wrong reason.
    if _route_choice() != "pingpong":
        _specialised_warm(device)


# ---------------------------------------------------------------------------
# composable reference
# ---------------------------------------------------------------------------
@functools.cache
def _e2m1_lut(device_key: str) -> torch.Tensor:
    return torch.tensor(
        _E2M1_VALUES, dtype=torch.float32, device=torch.device(device_key)
    )


@functools.cache
def _v_scale_unswizzle_index(device_key: str) -> torch.Tensor:
    """Flat position of the logical (token, group) V block scale.

    The cache writer stores the scale of logical ``(t, s)`` at
    ``((t // 4) * 4 + s // 2, (s % 2) * 4 + t % 4)``; a reader of logical
    ``(t, s)`` therefore has to look there.
    """
    device = torch.device(device_key)
    t = torch.arange(_PAGE_SIZE, device=device).unsqueeze(1)
    s = torch.arange(_SCALE_DIM, device=device).unsqueeze(0)
    groups = _SCALE_DIM // 4
    swizzled_t = (t // 4) * 4 + s // groups
    swizzled_s = (s % groups) * 4 + t % 4
    return (swizzled_t * _SCALE_DIM + swizzled_s).reshape(-1)


def _unpack_e2m1(packed: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    low = lut[(packed & 0x0F).long()]
    high = lut[(packed >> 4).long()]
    return torch.stack((low, high), dim=-1).reshape(*packed.shape[:-1], _HEAD_DIM)


def reference(
    *,
    q: torch.Tensor,
    k_data: torch.Tensor,
    v_data: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
    out: torch.Tensor,
    seqlen_q: int = 1,
    causal: bool = True,
) -> torch.Tensor:
    """FP32 composable oracle for the same problem surface.

    Never selected automatically: it is two orders of magnitude slower than the
    kernel and exists so that correctness has a denominator that does not come
    from the kernel under test. It is a peer of the kernel, not an
    approximation of it -- the dequant, the block-scale layouts, the mask and
    the accumulation order are the documented ones, evaluated in FP32.

    ``q2k_indices`` is consumed as :func:`msa_topk_select` documents it -
    ascending, ``-1`` tail-padded and distinct - so, like the kernel, no
    de-duplication pass is applied.
    """
    total_q = int(q.shape[0])
    seqlen_q = int(seqlen_q)
    # Read from the selection tensor, not from a module constant: the oracle has
    # to follow the kernel onto every top-k the route now admits.
    topk = int(q2k_indices.shape[2])
    max_blocks = int(page_table.shape[1])
    device_key = str(q.device)
    lut = _e2m1_lut(device_key)
    unswizzle = _v_scale_unswizzle_index(device_key)
    seq = seqused_k.long()
    columns = torch.arange(_PAGE_SIZE, device=q.device)

    per_request = topk * _PAGE_SIZE * _HEAD_DIM * 4 * 2
    chunk = max(1, min(total_q, _REFERENCE_CHUNK_BYTES // per_request))
    # Decode tokens are right-aligned: row i of a request sits at KV position
    # seq - seqlen_q + (i % seqlen_q).
    rows_index = torch.arange(total_q, device=q.device)
    request_of_row = rows_index // seqlen_q
    position_of_row = rows_index - request_of_row * seqlen_q
    seq = seq[request_of_row]
    causal_limit = position_of_row + seq - seqlen_q

    for head in range(_NUM_KV_HEADS):
        for low in range(0, total_q, chunk):
            high = min(total_q, low + chunk)
            rows = high - low
            selected = q2k_indices[head, low:high].long()  # (rows, topk)
            safe_blocks = selected.clamp(min=0, max=max_blocks - 1)
            table = page_table[request_of_row[low:high]].long()
            pages = torch.gather(table, 1, safe_blocks)
            valid = (
                (selected >= 0)
                & (selected < max_blocks)
                & (pages >= 0)
                & (selected * _PAGE_SIZE < seq[low:high].unsqueeze(1))
            )
            safe_pages = torch.where(valid, pages, torch.zeros_like(pages)).reshape(-1)

            k_bytes = k_data[safe_pages, head]  # (rows*topk, 128, 64)
            v_bytes = v_data[safe_pages, head]
            k_sf = k_scale[safe_pages, head].view(torch.float8_e4m3fn).float()
            v_sf = v_scale[safe_pages, head].view(torch.float8_e4m3fn).float()
            v_sf = v_sf.reshape(-1, _PAGE_SIZE * _SCALE_DIM)[:, unswizzle].reshape(
                v_sf.shape
            )

            keys = _unpack_e2m1(k_bytes, lut) * k_sf.repeat_interleave(
                _SCALE_VEC, dim=-1
            )
            keys = keys * float(k_global_scale)
            values = _unpack_e2m1(v_bytes, lut) * v_sf.repeat_interleave(
                _SCALE_VEC, dim=-1
            )
            values = values * float(v_global_scale)
            keys = keys.reshape(rows, topk * _PAGE_SIZE, _HEAD_DIM)
            values = values.reshape(rows, topk * _PAGE_SIZE, _HEAD_DIM)

            # At seqlen_q == 1 the right-aligned causal limit is seq - 1, so the
            # causal predicate coincides with the KV-length predicate.
            absolute = safe_blocks.unsqueeze(-1) * _PAGE_SIZE + columns
            mask = valid.unsqueeze(-1) & (absolute < seq[low:high].view(rows, 1, 1))
            if causal and seqlen_q != 1:
                mask = mask & (absolute <= causal_limit[low:high].view(rows, 1, 1))
            mask = mask.reshape(rows, topk * _PAGE_SIZE)

            group = q[low:high, head * 16 : (head + 1) * 16].float()
            scores = torch.einsum("bgd,bnd->bgn", group, keys) * float(softmax_scale)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
            weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
            out[low:high, head * 16 : (head + 1) * 16] = torch.einsum(
                "bgn,bnd->bgd", weights, values
            ).to(out.dtype)
    return out


# ---------------------------------------------------------------------------
# instantiation family
# ---------------------------------------------------------------------------
def pinned_path_reason(
    *,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    topk: int,
    max_blocks: int,
    seqlen_q: int,
    total_q: int,
    num_pages: int,
) -> Optional[str]:
    """Why this call does NOT take the pinned instantiation family, or ``None``.

    This is the Python mirror of ``selects_pinned_path`` in
    ``csrc/msa_decode_nvfp4_specialized.cu``, and the mirroring is deliberate,
    not accidental duplication:

    * the C++ copy is the one that dispatches, and it re-derives its answer from
      the tensors, so it cannot be lied to;
    * this copy is the one a GPU-free serviceability preflight can EXECUTE over
      every coordinate a serving run reaches, which is the only way to find out
      that the pin has stopped matching the deployment before a benchmark does;
    * ``run`` passes this copy's answer to the binding, which refuses the call
      if the two disagree -- so the copies cannot drift apart in silence.

    Both families compute the same function. The pinned one resolves the
    deployment's geometry at compile time and carries a deeper page pipeline;
    the general one reads every dimension at runtime. A miss here is a
    performance statement, never a correctness or a capability one.
    """

    if num_qo_heads != _NUM_QO_HEADS:
        return f"num_qo_heads {num_qo_heads} != {_NUM_QO_HEADS}"
    if num_kv_heads != _NUM_KV_HEADS:
        return f"num_kv_heads {num_kv_heads} != {_NUM_KV_HEADS}"
    if head_dim != _HEAD_DIM:
        return f"head_dim {head_dim} != {_HEAD_DIM}"
    if page_size != _PAGE_SIZE:
        return f"page_size {page_size} != {_PAGE_SIZE}"
    if topk != _TOPK:
        return f"topk {topk} != {_TOPK}"
    if max_blocks != _PINNED_MAX_BLOCKS:
        return f"block-table width {max_blocks} != {_PINNED_MAX_BLOCKS}"
    if seqlen_q != 1:
        return f"seqlen_q {seqlen_q} != 1"
    # A batch of 32 over a page pool smaller than 32 rows per request is not a
    # serving shape (a serving pool is millions of pages); the pinned family's
    # short-cache instantiation is not tuned for it, so it is routed to the
    # general family rather than guessed at.
    if total_q == 32 and num_pages < 32 * total_q:
        return f"batch 32 over a {num_pages}-page pool is outside the pinned envelope"
    return None


def selects_pinned_path(**kwargs: int) -> bool:
    """``True`` when :func:`pinned_path_reason` admits the pinned family."""

    return pinned_path_reason(**kwargs) is None


def _pinned_kwargs_for(
    q: torch.Tensor,
    k: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqlen_q: int,
) -> Dict[str, int]:
    """The nine quantities the binding derives the same decision from."""

    return dict(
        num_qo_heads=int(q.shape[1]),
        num_kv_heads=int(k.shape[1]),
        head_dim=int(q.shape[2]),
        page_size=int(k.shape[2]),
        topk=int(q2k_indices.shape[2]),
        max_blocks=int(page_table.shape[1]),
        seqlen_q=int(seqlen_q),
        total_q=int(q.shape[0]),
        num_pages=int(k.shape[0]),
    )


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def _dispatch(
    module,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    seqlen_q: int,
    causal: bool,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
) -> None:
    """The launch itself, and nothing else.

    One entry point for both the caller's dispatch and :func:`warm`'s, so that
    warming exercises the same launch a captured call will replay rather than
    something adjacent to it.
    """

    global _pinned_dispatch_count, _general_dispatch_count

    miss = pinned_path_reason(
        **_pinned_kwargs_for(q, k, q2k_indices, page_table, seqlen_q)
    )
    if miss is not None:
        # Loud once per process, and only for calls that are otherwise exactly
        # the deployment's: everything matched but one axis is the shape of "the
        # pin stopped matching the deployment", which is a silent slowdown with
        # no failing test attached to it unless something says so.
        logger.info_once(
            "NVFP4 MSA decode taking the parametric instantiation family: %s. "
            "Correct, but not the instantiation the pinned figures were "
            "measured on.",
            miss,
        )
    module.msa_decode_nvfp4_specialized(
        q,
        k,
        v,
        k_scale,
        v_scale,
        q2k_indices,
        page_table,
        seqused_k,
        out,
        int(seqlen_q),
        int(bool(causal)),
        float(softmax_scale),
        float(k_global_scale),
        float(v_global_scale),
        0 if miss is not None else 1,
    )
    if miss is None:
        _pinned_dispatch_count += 1
    else:
        _general_dispatch_count += 1


def _dispatch_specialised(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    seqlen_q: int,
    causal: bool,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
) -> None:
    """Launch the CuTe-DSL implementation, and nothing else.

    Positional, because that implementation's entry point is positional. It
    re-validates every tensor it was handed; this route's guard is a superset
    of what it requires, so the second validation is redundant and kept -- a
    launch argument nobody re-derives is how a layout contract rots.
    """

    global _specialised_dispatch_count

    _specialised_module().run(
        q,
        k,
        v,
        k_scale,
        v_scale,
        q2k_indices,
        page_table,
        seqused_k,
        int(seqlen_q),
        bool(causal),
        float(softmax_scale),
        float(k_global_scale),
        float(v_global_scale),
        out,
    )
    _specialised_dispatch_count += 1


def run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    seqlen_q: int,
    causal: bool,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
) -> torch.Tensor:
    """Dispatch the kernel into ``out``.

    Raises rather than substituting a slower implementation: this kernel is the
    only NVFP4 MSA decode implementation on this architecture, so a silent
    fallback would turn a build or warm-up problem into a two-orders-of-
    magnitude regression that nothing downstream would notice.
    """

    global _dispatch_count

    reason = check_specialized(q.device)
    if reason is not None:
        raise RuntimeError(
            "the NVFP4 MSA decode kernel cannot serve this call: "
            f"{reason}. Warm it before CUDA graph capture with "
            "flashinfer.msa_ops.msa_decode_nvfp4_specialized_warmup(device)."
        )
    try:
        module = load_msa_decode_nvfp4_specialized_module(_target_for(q.device))
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "failed to build the NVFP4 MSA decode kernel: "
            f"{type(exc).__name__}: {str(exc)[:512]}"
        ) from exc

    capturing = _is_capturing()
    if not capturing:
        logger.info_once(
            "FlashInfer NVFP4 MSA decode kernel active on compute capability 10.0/10.3"
        )

    # ---- internal route selection -----------------------------------------
    # Not a capability decision. Both implementations compute the same
    # function over everything check_surface admitted; this picks the faster
    # one for the call at hand.
    choice = _route_choice()
    if choice == "pingpong":
        reason = f"{_ROUTE_ENV}=pingpong"
    else:
        reason = specialised_route_reason(
            q=q,
            k=k,
            q2k_indices=q2k_indices,
            seqlen_q=seqlen_q,
            causal=causal,
            softmax_scale=softmax_scale,
            k_global_scale=k_global_scale,
        )
    if reason is not None and choice == "specialised":
        # Forced, and it cannot serve this call. RAISE: the point of this
        # setting is to make the guard provable, so it must not silently do
        # the other thing.
        raise RuntimeError(
            f"{_ROUTE_ENV}=specialised but the specialised NVFP4 MSA decode "
            f"implementation cannot serve this call: {reason}"
        )
    if reason is None:
        _dispatch_specialised(
            q=q,
            k=k,
            v=v,
            k_scale=k_scale,
            v_scale=v_scale,
            q2k_indices=q2k_indices,
            page_table=page_table,
            seqused_k=seqused_k,
            out=out,
            seqlen_q=seqlen_q,
            causal=causal,
            softmax_scale=softmax_scale,
            k_global_scale=k_global_scale,
            v_global_scale=v_global_scale,
        )
        if not capturing:
            _warmed_devices.add((q.device.type, q.device.index))
        _dispatch_count += 1
        return out
    _specialised_decline_counts[reason] = _specialised_decline_counts.get(reason, 0) + 1
    _dispatch(
        module,
        q=q,
        k=k,
        v=v,
        k_scale=k_scale,
        v_scale=v_scale,
        q2k_indices=q2k_indices,
        page_table=page_table,
        seqused_k=seqused_k,
        out=out,
        seqlen_q=seqlen_q,
        causal=causal,
        softmax_scale=softmax_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    if not capturing:
        _warmed_devices.add((q.device.type, q.device.index))
    _dispatch_count += 1
    return out


def _specialised_route_stats() -> Dict[str, Any]:
    """What the CuTe-DSL implementation is, costs, and refuses.

    Reported rather than documented because every number here is something a
    consumer, a serviceability preflight or an A/B harness has to check against
    its own deployment: which batch sizes take it, what it holds on the device,
    and how many calls it declined and for what.
    """

    module = _specialised_module()
    stats: Dict[str, Any] = {
        "available": module is not None,
        "import_error": _specialised_import_error,
        "override_env": _ROUTE_ENV,
        "override_choices": list(_ROUTE_CHOICES),
        "dispatch_count": _specialised_dispatch_count,
        "declines": dict(sorted(_specialised_decline_counts.items())),
        "warmed_devices": sorted(str(entry) for entry in _specialised_warm_devices),
        "warm_failed_devices": sorted(str(entry) for entry in _specialised_warm_failed),
        "fallback": (
            "the warp-specialised ping-pong kernel in "
            "csrc/msa_decode_nvfp4_specialized.cu, which serves every shape "
            "this route accepts"
        ),
    }
    # A misspelled override is a hard error where it changes what runs, and a
    # REPORTED one here. This function is a consumer's capability probe --
    # vLLM's `has_flashinfer_msa_nvfp4_kv()` requires it to be callable and
    # takes its answer -- so a typo in an environment variable must not turn a
    # capability question into a startup crash.
    try:
        stats["override"] = _route_choice()
    except ValueError as exc:
        stats["override"] = os.environ.get(_ROUTE_ENV)
        stats["override_error"] = str(exc)
    if module is None:
        return stats
    # Enumerated from the implementation's own plan(), not tabulated: the
    # supported batch sizes are ceil(256 / (batch * kv_heads)) landing in the
    # set of split counts that have an instantiation, which is not an interval.
    covered: list = []
    idx_for: Dict[int, int] = {}
    for batch in range(1, 257):
        plan = module.plan(
            total_q=batch,
            num_qo_heads=_NUM_QO_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            grp=_NUM_QO_HEADS // _NUM_KV_HEADS,
            topk=_TOPK,
            page_size=_PAGE_SIZE,
            seqlen_q=1,
            causal=1,
        )
        if plan["specialised"]:
            covered.append(batch)
            idx_for.setdefault(plan["kernel_idx"], batch)
    spans, start = [], None
    for batch in range(1, 258):
        hit = batch in set(covered)
        if hit and start is None:
            start = batch
        elif not hit and start is not None:
            spans.append([start, batch - 1])
            start = None
    stats.update(
        implementation="flashinfer.msa_ops.cute_dsl.sparse_decode_nvfp4_sm100",
        geometry={
            "num_qo_heads": _NUM_QO_HEADS,
            "num_kv_heads": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
            "page_size": _PAGE_SIZE,
            "topk": _TOPK,
            "seqlen_q": 1,
            "causal": True,
        },
        batch_spans_at_geometry=spans,
        batch_span_note=(
            "batch sizes 1-256 at the geometry above; outside these spans the "
            "route serves the call with the ping-pong kernel. Not an interval: "
            "only split counts with a compiled instantiation are covered"
        ),
        compiled_instantiations=sorted(module.SPECIALISED_KERNEL_IDS),
        instantiation_first_batch={str(k): v for k, v in sorted(idx_for.items())},
        # The generalized instantiation is not built. It is the only one that
        # publishes split-K partials through global memory, and the only
        # consumer of the split-K arena; both facts are consequences of the
        # same decision.
        uncompiled_instantiations=sorted(
            set(range(len(module._VARIANTS))) - set(module.SPECIALISED_KERNEL_IDS)
        ),
        persistent_device_bytes=module.PERSISTENT_DEVICE_BYTES,
        persistent_device_bytes_if_generalized_were_reachable=(
            module.ARENA_BYTES_IF_GENERALIZED_WERE_REACHABLE
        ),
        requires_eager_warm=True,
        warm_entry_point="flashinfer.msa_ops.msa_decode_nvfp4_specialized_warmup",
        concurrent_stream_limit=None,
        concurrent_stream_limit_note=(
            "unbounded: the compiled instantiations keep every split partial "
            "in distributed shared memory, so no per-stream device scratch "
            "exists to contend for"
        ),
    )
    return stats


def msa_decode_nvfp4_specialized_stats() -> Dict[str, Any]:
    """Introspection for benchmarks, tests and e2e dispatch pinning."""

    allowlist = sorted(_load_allowlist())
    return {
        "available": bool(_SPECIALIZED_AVAILABLE),
        # One translation unit holds every reachable instantiation: the cluster
        # width is a pure function of batch_size * num_kv_heads and the query
        # length only selects between two bodies, so nothing about a call shape
        # can trigger a compile.
        "compiled_variants": (
            load_msa_decode_nvfp4_specialized_module.cache_info().currsize
            if _SPECIALIZED_AVAILABLE
            else 0
        ),
        # 30 = 24 parametric (4 cluster widths x {q1, general q} x {q1 with a
        # short page pool} x {full-page tile, partial-page tile}) + 6 pinned.
        # Every one is compiled into the single translation unit, so nothing
        # about a call shape can trigger a compile.
        "distinct_kernels_for_allowlist": 30,
        "kernel_instantiations": [
            f"general_cluster_width_{width}_{variant}_page_tile_{tile}"
            for tile in ("full", "partial")
            for variant in ("q1", "q1_short_pool", "general_q")
            for width in (1, 2, 4, 8)
        ]
        + [
            f"pinned_cluster_width_{width}_chunk_{chunk}{tail}"
            for width, chunk, tail in (
                (8, 2, ""),
                (4, 4, ""),
                (4, 4, "_short_pool"),
                (2, 4, ""),
                (2, 2, "_short_pool"),
                (1, 4, ""),
            )
        ],
        # The pinned family's envelope, stated so a consumer or a preflight can
        # check it against its own deployment instead of reading it off a
        # benchmark table.
        "pinned_path_envelope": {
            "num_qo_heads": _NUM_QO_HEADS,
            "num_kv_heads": _NUM_KV_HEADS,
            "head_dim": _HEAD_DIM,
            "page_size": _PAGE_SIZE,
            "topk": _TOPK,
            "max_blocks": _PINNED_MAX_BLOCKS,
            "seqlen_q": 1,
        },
        "pinned_path_semantics": (
            "a faster instantiation of the same function, not a narrower "
            "capability: a call outside the envelope is served by the "
            "parametric family, never refused"
        ),
        "compile_cache_key": "(compute capability target,)",
        "precompiled": True,
        "allowlist_rows": len(allowlist),
        "allowlist_fields": list(_WORKLOAD_FIELDS),
        "allowlist": [list(row) for row in allowlist],
        "parametric_axes": [
            "batch_size",
            "seqlen_q",
            "seqused_k",
            "max_blocks",
            "causal",
            "topk",
            "q2k_indices_outer_strides",
        ],
        # The two axes that stopped being compile-time constants, with the
        # bound that is actually enforced rather than the one that was assumed.
        "topk_range": [1, _MAX_TOPK],
        "topk_range_reason": (
            "every selection slot is one lane of warp 0's compaction ballot in "
            "the parametric family (general::kSelectedCapacity); topk 16 also "
            "selects the faster pinned and CuTe-DSL instantiations"
        ),
        "q2k_indices_layout": (
            "int32 (num_kv_heads, total_q, topk); only the innermost dimension "
            "must be dense. The two outer strides are passed to the kernel, so "
            "a transposed view of a token-major (total_q, num_kv_heads, topk) "
            "buffer is read in place -- no contiguous copy on the serving path"
        ),
        "supported_compute_capability": sorted(_SUPPORTED_COMPUTE_CAPABILITIES),
        # The CUDA graph contract, stated where a consumer can read it instead
        # of inferring it from a docstring or a raised exception. A serving
        # engine can check this before it decides how to capture; a GPU-free
        # serviceability gate can check it before it spends an allocation.
        "cuda_graph": {
            "requires_workspace": capture_requires_workspace(),
            "requires_eager_warm": True,
            "warm_entry_point": (
                "flashinfer.msa_ops.msa_decode_nvfp4_specialized_warmup"
            ),
        },
        # The internal route. `dispatch_count` counts calls this route served
        # by either implementation; these say which one, and why not the other.
        "specialised_route": _specialised_route_stats(),
        "dispatch_count": _dispatch_count,
        "pinned_dispatch_count": _pinned_dispatch_count,
        "general_dispatch_count": _general_dispatch_count,
        "warm_dispatch_count": _warm_dispatch_count,
        "guard_rejections": dict(sorted(_reject_counts.items())),
        "warmed_devices": sorted(str(entry) for entry in _warmed_devices),
    }


__all__ = [
    "as_scale_bytes",
    "specialised_route_reason",
    "capture_requires_workspace",
    "check_specialized",
    "check_surface",
    "msa_decode_nvfp4_specialized_stats",
    "pinned_path_reason",
    "reference",
    "run",
    "selects_pinned_path",
    "warm",
]
