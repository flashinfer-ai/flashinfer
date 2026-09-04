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

NVFP4 paged-KV sparse prefill for compute capability 10.0/10.3.

Compute capability 10.0/10.3 has no NVFP4 MSA route at all: the backend rejects
packed uint8 K/V and rejects tensor K/V scales, so a caller holding an NVFP4 MSA
cache cannot use the FlashInfer MSA path on those parts. This module adds that
capability for the prefill attend, and nothing else: the decode attend over the
same cache is a separate route with its own guard. A cache one of the two cannot
read is worse than no NVFP4 cache, so a consumer that turns NVFP4 KV on must
find BOTH capabilities present before doing so -- see
:func:`msa_prefill_nvfp4_specialized_stats` for the marker to probe.

Because there is nothing to fall back to, the guard is a *capability* guard, not
a benchmarked-shape allowlist: it constrains the model geometry the kernel body
is built from (64 query heads, 4 KV heads, head_dim 128, page_size 128, topk
16), the KV page layout, and the two semantics the kernel body cannot express
(causal, no LSE); it is parametric in batch size, per-request query length,
per-request KV length and total query count.

BLOCK-TABLE WIDTH IS A FREE PARAMETER. The per-tile block union is a
128-slot open-addressed hash table keyed on the block id, not a bitmap indexed
by it, so its size is set by how many blocks eight queries can select
(``8 * topk``) and not by how many blocks exist. The only ceiling left is the
packing of a block id into the low 24 bits of a union entry --
``MAX_SELECTABLE_BLOCKS``, i.e. a context of ``MAX_CONTEXT_TOKENS`` tokens at
page_size 128. It is checked against the block-table width rather than clamped,
because truncating an id would silently attend to fewer blocks than the caller
selected. In practice the width axis is unconstrained: a 2048-wide block table,
a 262,144-token context, is admitted and served.

SOFTMAX RANGE GUARD. The kernel exponentiates against a FIXED origin on its
fast path and pairs that with a range CHECK on the row denominator and an exact
REPLAY of any tile that fails it, against a data-derived running maximum. The
three parts are one mechanism; see the header of
``csrc/msa_prefill_nvfp4_specialized.cu``. The replay is exercised by real
inputs, not dead code, and its cost is inside every number this route quotes.

RUN-TO-RUN REPRODUCIBILITY IS WIDTH-DEPENDENT, AND THIS IS THE ONLY PLACE IT IS
WRITTEN DOWN. The per-tile union is consumed in ascending hash-slot order, so
the order in which selected blocks are accumulated -- and therefore the last
bits of the output -- is a function of which slot each block lands in. While a
request's block count is at most ``UNION_TABLE_SLOTS`` (i.e. a context of
``DETERMINISTIC_CONTEXT_TOKENS`` tokens or less) the slot is
``block * odd_multiplier mod UNION_TABLE_SLOTS``, a permutation of a set smaller
than the table: distinct blocks cannot collide, the insert is a commutative
``atomicOr``, and repeated runs of the same call are BIT-IDENTICAL. Above that
context the insert falls back to linear probing under ``atomicCAS``, colliding
ids land in whatever order the atomics resolve in, and two runs of the same call
may differ in the low bits of the output. Every result is correct either way --
this is a determinism property, not a correctness one -- but a caller that needs
bitwise reproducibility above ``DETERMINISTIC_CONTEXT_TOKENS`` tokens does not
get it here today. Sorting the compacted union by block id would remove the
dependence; it is not done because it would change the kernel that was
benchmarked.

MEMORY COST: none. Each CTA dequantizes the K and V sides of one selected page
straight into its own 98,848 B of dynamic shared memory, in the swizzled layout
the MMA reads, and never materializes a BF16 copy of the cache in global memory.
This route allocates no scratch of any kind -- no workspace buffer, no device
global, no caching-allocator temporary -- so its peak-HBM contribution beyond the
output tensor is zero bytes, at every block-table width. That is also what makes
the width free: nothing in the kernel is proportional to it except one runtime
row stride of the block table itself.
"""

from __future__ import annotations

import functools
import json
import sys
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import torch

from ..jit.core import logger

_WORKLOAD_PACKAGE = "flashinfer.msa_ops"
_WORKLOAD_FILE = "msa_prefill_nvfp4_specialized_workloads.json"
_WORKLOAD_FIELDS = (
    "num_qo_heads",
    "num_kv_heads",
    "head_dim",
    "page_size",
    "topk",
)

_HEAD_DIM = 128
_PAGE_SIZE = 128
_NUM_QO_HEADS = 64
_NUM_KV_HEADS = 4
_TOPK = 16
_SCALE_VEC = 16
_DATA_DIM = _HEAD_DIM // 2
_SCALE_DIM = _HEAD_DIM // _SCALE_VEC

# Page map. Stated here as the single Python-side source of truth and asserted
# again in the kernel binding, so a divergence is a launch-time error rather
# than a silent misread. Any sibling route over the same cache must restate it
# rather than import it -- the routes ship as independent capabilities and
# neither should fail to import because the other did.
_DATA_HEAD_STRIDE = _PAGE_SIZE * _DATA_DIM
_SCALE_HEAD_STRIDE = _PAGE_SIZE * _SCALE_DIM
_K_SCALE_BYTE_OFFSET = _NUM_KV_HEADS * _DATA_HEAD_STRIDE
_V_DATA_BYTE_OFFSET = _K_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE
_V_SCALE_BYTE_OFFSET = _V_DATA_BYTE_OFFSET + _K_SCALE_BYTE_OFFSET
_PAGE_BYTES = _V_SCALE_BYTE_OFFSET + _NUM_KV_HEADS * _SCALE_HEAD_STRIDE

# A block id is carried in the low 24 bits of a union-table entry (bits 24..31
# hold the per-query membership mask), and 0 marks an empty slot, so the largest
# representable id is 0x00FFFFFF - 1 and a block-table width of 0x00FFFFFF is
# admissible -- 2,147,483,520 context tokens at page_size 128. This is the
# route's ONLY width ceiling, and it is five orders of magnitude above the
# 128-wide block table a typical deployment uses today.
MAX_SELECTABLE_BLOCKS = 0x00FFFFFF
MAX_CONTEXT_TOKENS = MAX_SELECTABLE_BLOCKS * _PAGE_SIZE
# Slots in the per-tile block union.  Sized by how many blocks one tile of
# queries can select (queries_per_tile * topk), not by how many blocks exist,
# which is what makes the block-table width a free parameter.  It is also the
# width below which the union is built collision-free and the route is
# bit-reproducible run to run -- see the module docstring.
UNION_TABLE_SLOTS = 128
DETERMINISTIC_CONTEXT_TOKENS = UNION_TABLE_SLOTS * _PAGE_SIZE
# Dynamic shared memory per CTA. Recorded here because it is the whole of this
# route's memory cost and because it caps occupancy at 2 CTAs/SM.
_DYNAMIC_SMEM_BYTES = 98848

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
    from ..jit.msa_prefill_nvfp4_specialized import (
        load_msa_prefill_nvfp4_specialized_module,
    )

    _SPECIALIZED_AVAILABLE = True
except (ImportError, RuntimeError):  # pragma: no cover - unbuildable environment
    _SPECIALIZED_AVAILABLE = False
    load_msa_prefill_nvfp4_specialized_module = None

_dispatch_count = 0
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
            "msa_prefill_nvfp4_specialized: guard rejected at %s:%d (%s)",
            __name__,
            line,
            reason,
        )
    return reason


@functools.cache
def _read_workload_file() -> Dict[str, Any]:
    """Read the allowlist from the installed package or the source checkout.

    Mirrors ``_get_blackwell_msa_csrc_dir``: an install that does not expose the
    package through ``importlib.resources`` still has the file next to this
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
            "Unable to load the specialized NVFP4 MSA prefill allowlist: %s",
            type(exc).__name__,
        )
        return frozenset()


def _module_is_loaded() -> bool:
    if not _SPECIALIZED_AVAILABLE:
        return False
    return load_msa_prefill_nvfp4_specialized_module.cache_info().currsize > 0


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
# float8_e4m3fn views and the data regions as uint8; both are the same bytes, so
# either spelling of the scale tensors is accepted and reinterpreted as bytes
# before the kernel sees it.
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
    cu_seqlens_q: Optional[torch.Tensor],
    page_table: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
    causal: bool,
    return_softmax_lse: bool,
    return_temperature_lse: bool,
    lse_temperature_scale: float,
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    k_global_scale: Optional[float],
    v_global_scale: Optional[float],
    q_offset,
) -> Optional[str]:
    """Return ``None`` when this call belongs to the NVFP4 prefill route.

    Ordering is deliberate: semantics, then layout, then device and architecture
    LAST, so the whole semantic and layout surface of the guard is exercisable
    on a host with no GPU.
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
    if not causal:
        # Not a preference: the kernel masks every key past the query's own
        # right-aligned position unconditionally, so a non-causal call would get
        # a causal answer.
        return _reject("this route is causal-only")
    if q_offset is not None:
        # The kernel derives the query offset as seqused_k - query_length, i.e.
        # the queries are the tail of their request's KV.
        return _reject("this route does not accept an explicit q_offset")
    if return_softmax_lse or return_temperature_lse:
        return _reject("this route does not produce a softmax LSE")
    if float(lse_temperature_scale) != 1.0:
        return _reject("this route does not apply an LSE temperature")
    if page_table is None or seqused_k is None:
        return _reject("this route serves the paged layout and needs seqused_k")
    if cu_seqlens_k is not None:
        # ``seqused_k`` is the sole authority for per-request KV length on this
        # route, and the kernel derives the causal offset from it as
        # ``seqused_k - query_length``. A ``cu_seqlens_k`` would be a second,
        # independent source for the same quantity, and reconciling the two
        # would need a device-to-host copy of both. Refuse rather than silently
        # prefer one: a caller that disagrees with itself gets told so.
        return _reject(
            "this route does not accept cu_seqlens_k; seqused_k is the sole "
            "source of per-request KV length"
        )
    if cu_seqlens_q is None:
        return _reject("this route needs cu_seqlens_q")

    # ---- shapes ---------------------------------------------------------
    if q.ndim != 3 or not q.is_contiguous():
        return _reject("q must be a contiguous 3-D (total_q, num_qo_heads, head_dim)")
    total_q, num_qo_heads, head_dim = (int(value) for value in q.shape)
    if (num_qo_heads, head_dim) != (_NUM_QO_HEADS, _HEAD_DIM):
        return _reject(
            f"this route serves 64 query heads at head_dim 128, "
            f"got {num_qo_heads}x{head_dim}"
        )
    if total_q < 1:
        return _reject("q must contain at least one query token")
    if (
        cu_seqlens_q.dtype != torch.int32
        or cu_seqlens_q.ndim != 1
        or not cu_seqlens_q.is_contiguous()
        or int(cu_seqlens_q.shape[0]) < 2
    ):
        return _reject("cu_seqlens_q must be contiguous int32 (batch_size + 1,)")
    batch_size = int(cu_seqlens_q.shape[0]) - 1

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
        or tuple(int(value) for value in q2k_indices.shape)
        != (_NUM_KV_HEADS, total_q, _TOPK)
    ):
        # topk is load-bearing here in a way it is not on the decode route: it
        # is the capacity the union table is sized from, and
        # `kQueriesPerTile * kTopK <= kHashSize` holds with EXACT equality at
        # 16, so raising it is a shared-memory and ballot-width change rather
        # than an argument. The kernel binding asserts it again for the same
        # reason.
        return _reject(f"q2k_indices must be int32 ({_NUM_KV_HEADS}, total_q, {_TOPK})")
    # The LAYOUT, unlike the extent, is now free: the two outer strides are
    # kernel arguments, so a transposed view of a token-major selection buffer
    # is read in place instead of copied contiguous once per prefill call.
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
        + _TOPK
    ) > 0x7FFFFFFF:
        return _reject("the q2k_indices view is too large for 32-bit addressing")
    # Entries below ``ceil(seqused_k[request] / page_size)`` must be real page
    # ids: the kernel reads them without re-validating, exactly as the BF16 MSA
    # paged path does. Everything at or above that index is never read, so the
    # usual -1/0 tail padding is fine. This cannot be checked here without a
    # device-to-host copy of seqused_k, so it is stated, not asserted.
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
    if int(page_table.shape[1]) > MAX_SELECTABLE_BLOCKS:
        # A kernel capability, not a benchmarked shape: a selected block id is
        # packed into the low 24 bits of a union-table entry. Truncating instead
        # would drop selected blocks from the union with no diagnostic. This is
        # the only bound on the width axis, and it is 65,536x production's.
        return _reject(
            f"a selected block id is carried in 24 bits, so the block-table "
            f"width may not exceed {MAX_SELECTABLE_BLOCKS} (page_size "
            f"{_PAGE_SIZE} => context {MAX_CONTEXT_TOKENS} tokens), got "
            f"{int(page_table.shape[1])}"
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
    # page map the cache writer used -- and they are also what lets the kernel
    # address a whole page from k's base pointer -- so they are checked here and
    # asserted again in the kernel binding.
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
    signature = (num_qo_heads, _NUM_KV_HEADS, head_dim, _PAGE_SIZE, _TOPK)
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
        (cu_seqlens_q, "cu_seqlens_q"),
        (page_table, "page_table"),
        (seqused_k, "seqused_k"),
    ):
        if tensor.device != q.device:
            return _reject(f"{name} must be on the same device as q")
    if _target_for(q.device) is None:
        return _reject("this route requires compute capability 10.0 or 10.3")
    return None


def check_specialized(device: torch.device) -> Optional[str]:
    """Return ``None`` when the kernel may serve a call on ``device``."""

    if not _SPECIALIZED_AVAILABLE:
        return "the NVFP4 MSA prefill module is unavailable"
    if _is_capturing():
        # Building a module or touching the driver for the first time inside a
        # capture region is illegal.
        if not _module_is_loaded():
            return "CUDA graph capture before the module was built"
        if (device.type, device.index) not in _warmed_devices:
            return "CUDA graph capture before the first eager dispatch on this device"
    return None


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

    One page (73,728 B), one request, one query token, a one-wide block table.
    Zero bytes are a valid NVFP4 encoding -- e2m1 code 0 and an e4m3 scale of
    0 -- so the launch is defined without quantizing anything; every logit is
    zero, the softmax is uniform over the page, and the output is discarded.
    The four K/V views are cut out of one allocation at the documented byte
    offsets because the guard proves that relationship and the kernel asserts
    it again.

    The launch allocates nothing beyond the tensors listed here: this route has
    no scratch, so warming costs one page, one query and one output row.
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
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        page_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        # The query is the tail of its request's KV, which is what the kernel
        # derives the causal offset from: seqused_k - query_length = 127.
        seqused_k=torch.full((1,), _PAGE_SIZE, dtype=torch.int32, device=device),
        out=torch.empty_like(q),
        softmax_scale=float(_HEAD_DIM**-0.5),
        k_global_scale=1.0,
        v_global_scale=1.0,
    )


def warm(device: torch.device | str) -> None:
    """Make ``device`` ready for CUDA graph capture of this route.

    Two things have to have happened before a capture region can contain this
    kernel: the module must be built, and it must have been *launched* at least
    once on the device, because the first launch is what makes the driver
    resolve a lazily loaded module (``CUDA_MODULE_LOADING=LAZY`` is the
    default). So this does both -- it builds and then dispatches one
    single-token call over one 73,728-byte page -- and records the device,
    which is the precondition :func:`check_specialized` names.

    Building alone would not do it, and that is the bug this replaces.
    ``check_specialized`` would otherwise refuse capture until
    ``_warmed_devices`` contained the device, ``_warmed_devices`` is only ever
    written by a real dispatch, and an error that told the caller to run
    ``msa_prefill_nvfp4_specialized_warmup(device)`` could not then clear
    itself. On this route that gate is also shadowed by a stricter one (capture
    without an ``MSASparseAttentionWorkspace`` is refused by the hook before
    ``run`` is reached, and a warmed workspace implies a prior eager dispatch),
    and prefill is not captured by serving engines today. Neither of those is a
    reason to ship a remedy that does not work:
    both are configuration, and this is the mechanism.

    Idempotent: after the first success this is a set lookup, which matters
    because the prefill hook calls it on every eager dispatch.
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
        return
    target = _target_for(device)
    if target is None:
        return
    global _warm_dispatch_count
    try:
        module = load_msa_prefill_nvfp4_specialized_module(target)
        _dispatch(module, **_warm_inputs(device))
    except (ImportError, OSError, RuntimeError) as exc:
        # torch.cuda.OutOfMemoryError is a RuntimeError, so the ~370 KB the
        # dummy needs is covered here too.
        # Stale text here is not cosmetic: this line is what a failed build
        # reports, and it must say what actually happens next. Nothing serves
        # the call -- `run` raises, by design, rather than substituting
        # anything slower.
        logger.warning_once(
            "Unable to warm the NVFP4 MSA prefill kernel: %s -- every call on "
            "this route will now RAISE, because this kernel is the only "
            "implementation of it. Full error: %s",
            type(exc).__name__,
            str(exc)[:512],
        )
        return
    _warm_dispatch_count += 1
    _warmed_devices.add((device.type, device.index))


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
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
    out: torch.Tensor,
) -> torch.Tensor:
    """FP32 composable oracle for the same problem surface.

    Never selected automatically: it is two orders of magnitude slower than the
    kernel and exists so that correctness has a denominator that does not come
    from the kernel under test. It is a peer of the kernel, not an approximation
    of it -- the dequant, the block-scale layouts, the right-aligned causal mask
    and the accumulation order are the documented ones, evaluated in FP32.

    ``q2k_indices`` is consumed as :func:`msa_topk_select` documents it --
    ascending, ``-1`` tail-padded and distinct -- so, like the kernel, no
    de-duplication pass is applied, and nothing is inferred about entries the
    selector did not produce.
    """
    max_blocks = int(page_table.shape[1])
    device_key = str(q.device)
    lut = _e2m1_lut(device_key)
    unswizzle = _v_scale_unswizzle_index(device_key)
    columns = torch.arange(_PAGE_SIZE, device=q.device)
    boundaries = cu_seqlens_q.to(torch.long).tolist()
    lengths = seqused_k.to(torch.long)

    per_row = _TOPK * _PAGE_SIZE * _HEAD_DIM * 4 * 2
    chunk = max(1, _REFERENCE_CHUNK_BYTES // per_row)

    for request in range(len(boundaries) - 1):
        q_lo, q_hi = int(boundaries[request]), int(boundaries[request + 1])
        if q_hi <= q_lo:
            continue
        kv_len = int(lengths[request])
        num_blocks = min(max_blocks, (kv_len + _PAGE_SIZE - 1) // _PAGE_SIZE)
        table = page_table[request].long()
        # Queries are the tail of their request's KV: row i of the request sits
        # at KV position i + (kv_len - query_length).
        offset = kv_len - (q_hi - q_lo)

        for head in range(_NUM_KV_HEADS):
            for low in range(q_lo, q_hi, chunk):
                high = min(q_hi, low + chunk)
                rows = high - low
                selected = q2k_indices[head, low:high].long()  # (rows, topk)
                safe_blocks = selected.clamp(min=0, max=max(num_blocks - 1, 0))
                pages = table[safe_blocks]
                valid = (
                    (selected >= 0)
                    & (selected < num_blocks)
                    & (pages >= 0)
                    & (selected * _PAGE_SIZE < kv_len)
                )
                safe_pages = torch.where(valid, pages, torch.zeros_like(pages)).reshape(
                    -1
                )

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
                keys = keys.reshape(rows, _TOPK * _PAGE_SIZE, _HEAD_DIM)
                values = values.reshape(rows, _TOPK * _PAGE_SIZE, _HEAD_DIM)

                absolute = safe_blocks.unsqueeze(-1) * _PAGE_SIZE + columns
                positions = (
                    torch.arange(low, high, device=q.device) - q_lo + offset
                ).view(rows, 1, 1)
                mask = (
                    valid.unsqueeze(-1) & (absolute < kv_len) & (absolute <= positions)
                )
                mask = mask.reshape(rows, _TOPK * _PAGE_SIZE)

                group = q[low:high, head * 16 : (head + 1) * 16].float()
                scores = torch.einsum("bgd,bnd->bgn", group, keys) * float(
                    softmax_scale
                )
                scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
                weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
                out[low:high, head * 16 : (head + 1) * 16] = torch.einsum(
                    "bgn,bnd->bgd", weights, values
                ).to(out.dtype)
    return out


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
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
) -> None:
    """The launch itself, and nothing else.

    One entry point for both the caller's dispatch and :func:`warm`'s, so that
    warming exercises the same launch a captured call will replay rather than
    something adjacent to it.
    """

    module.msa_prefill_nvfp4_specialized(
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
        float(softmax_scale),
        float(k_global_scale),
        float(v_global_scale),
    )


def run(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    page_table: torch.Tensor,
    seqused_k: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
    k_global_scale: float,
    v_global_scale: float,
) -> torch.Tensor:
    """Dispatch the kernel into ``out``.

    Raises rather than substituting a slower implementation: this kernel is the
    only NVFP4 MSA prefill implementation on this architecture, so a silent
    fallback would turn a build or warm-up problem into a two-orders-of-
    magnitude regression that nothing downstream would notice.
    """

    global _dispatch_count

    reason = check_specialized(q.device)
    if reason is not None:
        raise RuntimeError(
            "the NVFP4 MSA prefill kernel cannot serve this call: "
            f"{reason}. Warm it before CUDA graph capture with "
            "flashinfer.msa_ops.msa_prefill_nvfp4_specialized_warmup(device)."
        )
    try:
        module = load_msa_prefill_nvfp4_specialized_module(_target_for(q.device))
    except (ImportError, OSError, RuntimeError) as exc:
        raise RuntimeError(
            "failed to build the NVFP4 MSA prefill kernel: "
            f"{type(exc).__name__}: {str(exc)[:512]}"
        ) from exc

    capturing = _is_capturing()
    if not capturing:
        logger.info_once(
            "FlashInfer NVFP4 MSA prefill kernel active on compute capability 10.0/10.3"
        )
    _dispatch(
        module,
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
        softmax_scale=softmax_scale,
        k_global_scale=k_global_scale,
        v_global_scale=v_global_scale,
    )
    if not capturing:
        _warmed_devices.add((q.device.type, q.device.index))
    _dispatch_count += 1
    return out


def msa_prefill_nvfp4_specialized_stats() -> Dict[str, Any]:
    """Introspection for benchmarks, tests and e2e dispatch pinning.

    ``supported_compute_capability`` is also the capability marker a consumer
    probes: an NVFP4 MSA cache is only usable when the decode *and* the prefill
    attend both exist, so a framework enabling NVFP4 KV should require this
    function *and* the corresponding decode-side marker, and intersect their
    supported-capability sets. Enabling on the strength of this one alone would
    build a cache that decode cannot read.
    """

    allowlist = sorted(_load_allowlist())
    return {
        "available": bool(_SPECIALIZED_AVAILABLE),
        # One translation unit, one kernel, and the only compile-cache key is
        # the compute-capability target: every axis the guard leaves free is a
        # runtime argument, so nothing about a call shape can trigger a compile
        # or select a different instantiation.
        "compiled_variants": (
            load_msa_prefill_nvfp4_specialized_module.cache_info().currsize
            if _SPECIALIZED_AVAILABLE
            else 0
        ),
        "distinct_kernels_for_allowlist": 1,
        "kernel_instantiations": ["attend"],
        "compile_cache_key": "(compute capability target,)",
        "precompiled": True,
        "allowlist_rows": len(allowlist),
        "allowlist_fields": list(_WORKLOAD_FIELDS),
        "allowlist": [list(row) for row in allowlist],
        "parametric_axes": [
            "batch_size",
            "per_request_query_length",
            "seqused_k",
            "total_q",
            "max_blocks",
        ],
        "max_selectable_blocks": MAX_SELECTABLE_BLOCKS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "max_selectable_blocks_reason": (
            "a selected block id is packed into the low 24 bits of a union-table "
            "entry; the union itself is a 128-slot hash table sized by "
            "queries_per_tile * topk, so it does not scale with the block-table "
            "width"
        ),
        # Zero. Not "small": this route makes no allocation of any kind, at any
        # block-table width, so a caller sizing a KV pool need not budget for it.
        "scratch_bytes_formula": "0 (the dequant target is shared memory)",
        "scratch_bytes_per_request_at_128_blocks": 0,
        "dynamic_shared_memory_bytes": _DYNAMIC_SMEM_BYTES,
        "max_ctas_per_sm": 2,
        "causal_only": True,
        "kv_length_authority": "seqused_k",
        # Reproducibility, published so a consumer can decide rather than
        # discover.  See the module docstring for the mechanism.
        "union_table_slots": UNION_TABLE_SLOTS,
        "run_to_run_bitwise_reproducible_up_to_context": DETERMINISTIC_CONTEXT_TOKENS,
        "supported_compute_capability": sorted(_SUPPORTED_COMPUTE_CAPABILITIES),
        # The CUDA graph contract, stated where a consumer can read it.
        "cuda_graph": {
            # Kept required even though this route allocates nothing: the
            # signature check is what makes a captured replay provably the same
            # launch as the eager one it was warmed with, and dropping that on
            # the strength of one engine's current capture configuration would
            # be a guess rather than a property.
            "requires_workspace": True,
            "requires_eager_warm": True,
            "warm_entry_point": (
                "flashinfer.msa_ops.msa_prefill_nvfp4_specialized_warmup"
            ),
        },
        "dispatch_count": _dispatch_count,
        "warm_dispatch_count": _warm_dispatch_count,
        "guard_rejections": dict(sorted(_reject_counts.items())),
        "warmed_devices": sorted(str(entry) for entry in _warmed_devices),
    }


__all__ = [
    "DETERMINISTIC_CONTEXT_TOKENS",
    "MAX_CONTEXT_TOKENS",
    "MAX_SELECTABLE_BLOCKS",
    "UNION_TABLE_SLOTS",
    "as_scale_bytes",
    "check_specialized",
    "check_surface",
    "msa_prefill_nvfp4_specialized_stats",
    "reference",
    "run",
    "warm",
]
