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

Host side of the Cake QSA sparse paged GQA decode over an NVFP4 KV cache
(SM80 through SM90): the argument checks of ``qsa_sparse_paged_attention``,
the split rule, the form rule, the program rule, the workspace rule and the
launch sequence.

A call attends ``rows`` decode rows (one query token each, ``num_qo_heads``
heads of 256 dims) to the ``selection_width`` tokens each row selects from a
paged E2M1 cache with E4M3 block scales, and writes ``bf16(attention) *
sigmoid(output_gate)``.  The selection is cut into tiles of 64 tokens and the
tiles into ``splits`` splits; one decode CTA serves one (row, KV head, split).

Split rule (:func:`num_splits`): ``tiles = ceil(selection_width / 64)``;
``splits = max(1, min(tiles, 2 * sm_count // (rows * num_kv_heads)))`` (two
resident CTAs per SM, the decode's launch bound); then the measured table
:data:`SPLIT_OVERRIDES` of the device's SM count replaces the formula where it
names ``(sm_count, rows)``; then ``splits <= min(tiles, MAX_SPLITS)``.  The
rule reads the SM count alone, cached per device.

Form rule (:func:`select_form`): with one split the plain decode writes the
output itself.  Above one split, on compute capability 9.0 the call takes the
*fused* form -- the cooperative build of the decode, whose CTAs merge the
splits in the same launch through a counter per (row, KV head) -- when its
grid ``rows * num_kv_heads * splits`` does not exceed the co-residency the
decode's launch bound guarantees on the device (:func:`fused_capacity`: two
CTAs per SM times the SM count, 264 on an H100; a cooperative launch above the
device's co-residency is refused by the driver); else,
and always on compute capability 8.x, the *split* form: the plain decode, then
the separate merge.

Program rule (:func:`select_wide`, then :func:`cake_jit.select_records`): the
plain decode has two programs.  On ``sm_80`` a split-form call is *wide* when
both hold: its decode grid ``rows * num_kv_heads * splits`` exceeds one
resident wave (``CTAS_PER_SM`` CTAs per SM times the SM count: 216 CTAs on an
A100's 108 SMs) and each of its CTAs walks a long tile stream, ``ceil(tiles /
splits) >= LONG_STREAM_TILES`` (16 tiles = 1024 selected tokens).  A wide call
launches the whole-tile program (``decode_whole_tile``: whole-tile gather, no
L2 cache hint, shift selector -- the previous release's ``sm_80`` decode);
every other call launches the K-first program (``decode``), and ``sm_90a``
always does.  On 108 SMs at the owner's 24 x 2 heads and 2051 selections (33
tiles): rows 1-32 run at most one wave -> K-first; rows 64 (4 splits, 512
CTAs, 9 tiles per CTA) -> K-first; rows 128 (2 splits, 512 CTAs, 17 tiles per
CTA) -> whole-tile; the 2048-row single-split call (33 tiles per CTA) ->
whole-tile.  Measured paired on two A100s (80 GB and 40 GB PCIe) under the
memset flush, the K-first program is 1-7 % faster at rows 64 and the
whole-tile program 6-8 % faster at rows 128, on both units.  The fused form
is never wide: its grid is within the co-residency by the form rule.  The
rule reads the architecture, the SM count and the call's tile and split
counts alone.

Workspace rule: the partials (``partial_out`` FP32 ``[splits, rows,
num_qo_heads, 256]``, ``partial_lse`` FP32 ``[splits, rows, num_qo_heads]``)
and the fused form's counters (uint32 ``[rows * num_kv_heads]``) come from the
caller's ``workspace_buffer`` when one is given (a contiguous uint8 tensor on
the device of ``q``, 256-byte aligned, at least :func:`qsa_workspace_bytes`
bytes; checked on every call that passes one, one split included; carved, never
copied; the counters are zeroed once, the first time a buffer's *storage* is
seen for a ``rows * num_kv_heads`` -- a new buffer the allocator places at a
freed buffer's address is a new storage and is zeroed again -- and reset
themselves after every launch), so a call with a workspace allocates nothing
and is CUDA-graph capturable with a replay-stable layout.  A call with no rows
returns before the workspace is read.  Without a workspace the partials come
from a module scratch per (device, stream) -- one float32 buffer, grown when a
call needs more than it holds, its ``(partial_out, partial_lse)`` views kept
per call shape -- and the fused form's counters from a module buffer per
(device, stream): zeroed when it is allocated, left at its idle value by every
launch, grown when a call needs more counters than it holds.  Calls on one
stream are ordered, so no two launches share live partials or counters; two
Python threads enqueuing on one stream are ordered by a module lock from the
scratch's carve through the call's last launch (the split form's merge reads
the partials its own decode wrote).  A capturing stream allocates its partials
and counters per call (the graph's private pool, the counters replayed with
the zero fill) and leaves both caches untouched, so neither holds a graph
pool's memory and no graph replays into a buffer the host may replace.  That
path is outside the zero-allocation guarantee.

Nothing here reads tensor contents: every dispatch is on host scalars and
tensor metadata.  The launch plan of a call shape -- split count, form,
program, loaded FFI entries, argument order and grids -- is resolved once per
shape and device and reused; a call then checks its arguments, binds them in
one flat sequence and issues its launches under one environment-stream
context.  The kernels run on the current torch stream of the device.
"""

from __future__ import annotations

import functools
import math
import threading
import weakref
from typing import Any, Callable, Optional

import torch

from . import cake_jit

HEAD_DIM = 256
TILE_TOKENS = cake_jit.TILE_TOKENS
MAX_GROUP_SIZE = 16
MAX_SPLITS = cake_jit.MAX_SPLITS
# Resident decode CTAs per SM the split rule and the fused form's capacity count on (the kernel's launch bound).
CTAS_PER_SM = cake_jit.DECODE_CTAS_PER_SM
# The program rule's long-stream bound: tiles per CTA from which an sm_80 multi-wave grid runs the whole-tile program.
LONG_STREAM_TILES = cake_jit.DECODE_LONG_STREAM_TILES
# Every workspace slot starts on a 256-byte boundary.
ALIGN = 256
_LOG2E = math.log2(math.e)
# The architecture whose decode may take the fused form (the H100 measured it faster at every multi-split row of
# the owner's shape list; the A100 measured it slower at rows 1-4).
_FUSED_ARCH = "sm_90a"

# The measured split table: ``(sm_count, rows) -> splits`` where the fastest split count of the acceptance fixture
# (2 KV heads, a selection of 2051 tokens = 33 tiles) differs from the formula.  Keyed by the SM count, never by a
# product name.  H100 (132 SMs): rows 4 take 16 splits and rows 8 take 8; A100 (108 SMs): rows 64 take 4 and rows
# 128 take 2 (both above the formula's capacity; a plain launch runs in more than one wave).
SPLIT_OVERRIDES: dict[tuple[int, int], int] = {
    (132, 4): 16,
    (132, 8): 8,
    (108, 64): 4,
    (108, 128): 2,
}


def num_tiles(selection_width: int) -> int:
    """``ceil(selection_width / 64)``: the tiles of one row's selection."""
    return -(-int(selection_width) // TILE_TOKENS)


def num_splits(
    num_rows: int, num_kv_heads: int, selection_width: int, sm_count: int
) -> int:
    """The split count of a call on a device of ``sm_count`` SMs (module docstring, "Split rule")."""
    if num_rows < 1 or num_kv_heads < 1 or sm_count < 1:
        raise ValueError(
            f"num_rows={num_rows}, num_kv_heads={num_kv_heads}, sm_count={sm_count} must all be at least 1"
        )
    tiles = num_tiles(selection_width)
    formula = max(
        1,
        min(
            tiles, (CTAS_PER_SM * int(sm_count)) // (int(num_rows) * int(num_kv_heads))
        ),
    )
    chosen = SPLIT_OVERRIDES.get((int(sm_count), int(num_rows)), formula)
    return max(1, min(tiles, MAX_SPLITS, chosen))


def is_supported(device: torch.device | int | None = None) -> bool:
    """Whether ``device`` (the current CUDA device by default) is an exported target."""
    if not torch.cuda.is_available():
        return False
    index = _device_index(device)
    return cake_jit.arch_for_capability(cake_jit.device_capability(index)) is not None


@functools.cache
def fused_capacity(device_index: int) -> int:
    """The largest decode grid (``rows * num_kv_heads * splits`` CTAs) the fused form serves on the device: the
    co-residency the decode's launch bound guarantees, ``CTAS_PER_SM`` CTAs per SM times the SM count (264 on an
    H100); 0 on an architecture whose calls always take the split form."""
    if cake_jit.device_arch(device_index) != _FUSED_ARCH:
        return 0
    return CTAS_PER_SM * cake_jit.sm_count(device_index)


def select_form(
    num_rows: int, num_kv_heads: int, splits: int, device_index: int
) -> str:
    """``"fused"`` or ``"split"`` (module docstring, "Form rule")."""
    if int(splits) > 1 and int(num_rows) * int(num_kv_heads) * int(
        splits
    ) <= fused_capacity(device_index):
        return "fused"
    return "split"


def select_wide(
    num_rows: int,
    num_kv_heads: int,
    splits: int,
    num_tiles: int,
    form: str,
    sm_count: int,
) -> bool:
    """Whether a call of ``form`` is *wide* on a device of ``sm_count`` SMs (module docstring, "Program rule"):
    the split form with a decode grid ``num_rows * num_kv_heads * splits`` above one resident wave,
    ``CTAS_PER_SM * sm_count`` CTAs, whose CTAs each walk at least ``LONG_STREAM_TILES`` of the call's
    ``num_tiles`` tiles (``ceil(num_tiles / splits)``).  The fused form is never wide.  Pure: the SM count is
    the caller's."""
    if (
        int(num_rows) < 1
        or int(num_kv_heads) < 1
        or int(splits) < 1
        or int(num_tiles) < 1
        or int(sm_count) < 1
    ):
        raise ValueError(
            f"num_rows={num_rows}, num_kv_heads={num_kv_heads}, splits={splits}, num_tiles={num_tiles}, "
            f"sm_count={sm_count} must all be at least 1"
        )
    if form not in cake_jit.FORMS:
        raise ValueError(f"unknown form {form!r}; the forms are {cake_jit.FORMS}")
    multi_wave = int(num_rows) * int(num_kv_heads) * int(splits) > CTAS_PER_SM * int(
        sm_count
    )
    long_stream = -(-int(num_tiles) // int(splits)) >= LONG_STREAM_TILES
    return form == "split" and multi_wave and long_stream


def _round_up(n: int, m: int) -> int:
    return -(-int(n) // m) * m


def _layout(
    num_rows: int, num_qo_heads: int, num_kv_heads: int, splits: int
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    """``(offset, bytes)`` of the counters, the partial outputs and the partial LSEs, in that order, each on a
    256-byte boundary; the counters come first so that their slot is the same for every split count."""
    counters = (0, 4 * int(num_rows) * int(num_kv_heads))
    out_offset = _round_up(counters[0] + counters[1], ALIGN)
    partial_out = (
        out_offset,
        4 * int(splits) * int(num_rows) * int(num_qo_heads) * HEAD_DIM,
    )
    lse_offset = _round_up(partial_out[0] + partial_out[1], ALIGN)
    partial_lse = (lse_offset, 4 * int(splits) * int(num_rows) * int(num_qo_heads))
    return counters, partial_out, partial_lse


def qsa_workspace_bytes(
    num_rows: int, num_qo_heads: int, num_kv_heads: int, splits: int
) -> int:
    """Bytes a ``workspace_buffer`` needs for a call of these sizes (``splits`` from :func:`num_splits`; a buffer
    sized for the largest split count a caller launches, ``MAX_SPLITS`` at most, serves every smaller one)."""
    _counters, _partial_out, partial_lse = _layout(
        num_rows, num_qo_heads, num_kv_heads, splits
    )
    return _round_up(partial_lse[0] + partial_lse[1], ALIGN)


# The ``(storage_offset, rows * num_kv_heads)`` slots whose group counters were zeroed once, per workspace storage
# (module docstring, "Workspace rule").  The storage is held weakly: a storage that dies takes its entry with it,
# so a new buffer the allocator places at a recycled address is zeroed again.  Keyed by the storage, never by the
# address: an address is reused by the next allocation of its size and may hold anything in between.
_CounterSlots = set[tuple[int, int]]
_zeroed_counters: weakref.WeakKeyDictionary[torch.UntypedStorage, _CounterSlots] = (
    weakref.WeakKeyDictionary()
)


def _check_workspace(
    workspace: torch.Tensor,
    device: torch.device,
    num_rows: int,
    num_qo_heads: int,
    num_kv_heads: int,
    splits: int,
) -> torch.Tensor:
    """The caller's ``workspace`` as a flat uint8 view after its checks (device of ``q``, uint8, contiguous, at
    least ``qsa_workspace_bytes`` bytes for this call, 256-byte aligned); run on every call that passes a buffer,
    whatever the split count, so that a buffer's verdict does not depend on the device's SM count."""
    _check(
        workspace.device == device
        and workspace.dtype == torch.uint8
        and workspace.is_contiguous(),
        "workspace_buffer must be a contiguous uint8 CUDA tensor on the device of q",
    )
    raw = workspace.reshape(-1)
    need = qsa_workspace_bytes(num_rows, num_qo_heads, num_kv_heads, splits)
    _check(
        raw.numel() >= need,
        f"workspace_buffer needs at least {need} bytes for this call (qsa_workspace_bytes), got {raw.numel()}",
    )
    _check(
        raw.data_ptr() % ALIGN == 0, f"workspace_buffer must be {ALIGN}-byte aligned"
    )
    return raw


def _carve(
    raw: torch.Tensor,
    num_rows: int,
    num_qo_heads: int,
    num_kv_heads: int,
    splits: int,
    *,
    counters: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """The partial outputs, the partial LSEs and (when ``counters``) the zeroed group counters as views of the
    checked flat workspace ``raw`` (:func:`_check_workspace`); nothing is allocated or copied."""
    slot_counters, slot_out, slot_lse = _layout(
        num_rows, num_qo_heads, num_kv_heads, splits
    )
    partial_out = (
        raw[slot_out[0] : slot_out[0] + slot_out[1]]
        .view(torch.float32)
        .view(splits, num_rows, num_qo_heads, HEAD_DIM)
    )
    partial_lse = (
        raw[slot_lse[0] : slot_lse[0] + slot_lse[1]]
        .view(torch.float32)
        .view(splits, num_rows, num_qo_heads)
    )
    if not counters:
        return partial_out, partial_lse, None
    group_counters = (
        raw[slot_counters[0] : slot_counters[0] + slot_counters[1]]
        .view(torch.uint32)
        .view(num_rows * num_kv_heads)
    )
    seen = _zeroed_counters.setdefault(raw.untyped_storage(), set())
    key = (raw.storage_offset(), num_rows * num_kv_heads)
    if key not in seen:
        group_counters.zero_()
        seen.add(key)
    return partial_out, partial_lse, group_counters


def reset_workspace(workspace: torch.Tensor) -> None:
    """Forget that ``workspace`` (its storage) was seen: the next call carving it zeroes its counters again (for a
    buffer whose bytes another use overwrote)."""
    _zeroed_counters.pop(workspace.untyped_storage(), None)


# The three CUDA state queries of the per-call path, bound to their C entries
# (``torch.cuda.current_stream`` constructs a ``Stream`` object per call), with
# the ``torch.cuda`` forms on a torch build without them.
_current_raw_stream = cake_jit.current_raw_stream
try:
    _current_stream_capturing = torch._C._cuda_isCurrentStreamCapturing
except AttributeError:  # pragma: no cover - older torch
    _current_stream_capturing = torch.cuda.is_current_stream_capturing
try:
    _current_device = torch._C._cuda_getDevice
except AttributeError:  # pragma: no cover - older torch
    _current_device = torch.cuda.current_device

# The fused form's group counters for calls without a workspace, one buffer per (device index, CUDA stream
# handle) (module docstring, "Workspace rule"): zeroed when allocated, left at an idle value by every launch,
# replaced by a larger zeroed buffer when a call needs more counters.  Launches on one stream are ordered, so a
# buffer per stream is never live in two launches at once.  The ``[:count]`` view of a buffer is kept per
# (stream, count) beside the buffer it views, so a repeated call slices nothing; a view of a replaced buffer is
# dropped.  Both tables are bounded.
_STREAM_COUNTERS_MIN = 1024
_STREAM_VIEWS_MAX = 4096
_stream_counters: dict[tuple[int, int], torch.Tensor] = {}
_stream_counter_views: dict[
    tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]
] = {}


def _stream_counters_buffer(
    device: torch.device, key: tuple[int, int], count: int
) -> torch.Tensor:
    """``count`` idle group counters (uint32, contiguous) of the buffer of stream ``key`` -- ``(device index,
    raw stream handle)`` -- on ``device``."""
    buffer = _stream_counters.get(key)
    if buffer is None or buffer.numel() < count:
        buffer = torch.zeros(
            max(count, _STREAM_COUNTERS_MIN), dtype=torch.uint32, device=device
        )
        _stream_counters[key] = buffer
        for stale in [k for k in _stream_counter_views if k[:2] == key]:
            del _stream_counter_views[stale]
    view_key = (key[0], key[1], count)
    cached = _stream_counter_views.get(view_key)
    if cached is not None and cached[0] is buffer:
        return cached[1]
    if len(_stream_counter_views) >= _STREAM_VIEWS_MAX:
        _stream_counter_views.clear()
    view = buffer[:count]
    _stream_counter_views[view_key] = (buffer, view)
    return view


# The partials of calls without a workspace (module docstring, "Workspace rule"): one float32 scratch per
# (device index, CUDA stream handle), grown when a call needs more, with the ``(partial_out, partial_lse)``
# views kept per call shape.  Launches on one stream are ordered, so a stream's partials are never live in two
# launches at once; ``_SCRATCH_LOCK`` orders two Python threads enqueuing on one stream from the carve through
# the call's last launch (the split form's merge reads the partials its own decode wrote).  A capturing stream
# allocates its partials per call and never touches the scratch: a graph must not replay into a buffer the host
# may replace.  Bounded: one buffer per stream (the largest call seen), at most ``_SCRATCH_VIEWS_MAX`` view
# pairs each.
class _Scratch:
    __slots__ = ("buffer", "views")

    def __init__(self, buffer: torch.Tensor) -> None:
        self.buffer = buffer
        self.views: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}


_SCRATCH_VIEWS_MAX = 64
_SCRATCH_LOCK = threading.Lock()
_stream_partials: dict[tuple[int, int], _Scratch] = {}


def _stream_partials_views(
    device: torch.device,
    key: tuple[int, int],
    splits: int,
    num_rows: int,
    num_qo_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(partial_out, partial_lse)`` -- float32 ``[splits, rows, qo_heads, 256]`` and ``[splits, rows,
    qo_heads]``, each on a 256-byte boundary -- as views of the scratch of stream ``key`` on ``device``.
    Called under ``_SCRATCH_LOCK``."""
    shape = (splits, num_rows, num_qo_heads)
    scratch = _stream_partials.get(key)
    if scratch is not None:
        views = scratch.views.get(shape)
        if views is not None:
            return views
    count = splits * num_rows * num_qo_heads
    lse_offset = _round_up(count * HEAD_DIM, ALIGN // 4)
    need = lse_offset + count
    if scratch is None or scratch.buffer.numel() < need:
        scratch = _Scratch(torch.empty(need, dtype=torch.float32, device=device))
        _stream_partials[key] = scratch
    elif len(scratch.views) >= _SCRATCH_VIEWS_MAX:
        scratch.views.clear()
    buffer = scratch.buffer
    views = (
        buffer[: count * HEAD_DIM].view(splits, num_rows, num_qo_heads, HEAD_DIM),
        buffer[lse_offset : lse_offset + count].view(splits, num_rows, num_qo_heads),
    )
    scratch.views[shape] = views
    return views


def _device_index(device: torch.device | int | None) -> int:
    if device is None:
        return torch.cuda.current_device()
    if isinstance(device, int):
        return device
    return device.index if device.index is not None else torch.cuda.current_device()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@functools.cache
def _device_facts(device_index: int) -> tuple[str, int]:
    """``(arch, sm_count)`` of a supported CUDA device, once per index; ``RuntimeError`` for a capability
    outside SM80 through SM90 (the refusal is not cached, the facts are)."""
    capability = cake_jit.device_capability(device_index)
    arch = cake_jit.arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"QSA NVFP4 attention supports SM80 through SM90, got SM{capability[0]}{capability[1]}"
        )
    return arch, cake_jit.sm_count(device_index)


@functools.lru_cache(maxsize=4096)
def _resolve(
    num_rows: int, num_kv_heads: int, num_qo_heads: int, tiles: int, device_index: int
) -> tuple[int, bool, tuple[Any, ...]]:
    """The plan of one call shape on one device, resolved once per key: ``(splits, fused, launches)`` from
    the split rule, the form rule, the program rule and :func:`cake_jit.resolved_launches` (the loaded FFI
    entries, their argument getters and grids, in launch order).  Pure in its host scalars."""
    arch, sm_count = _device_facts(device_index)
    splits = num_splits(num_rows, num_kv_heads, tiles * TILE_TOKENS, sm_count)
    form = select_form(num_rows, num_kv_heads, splits, device_index)
    wide = select_wide(num_rows, num_kv_heads, splits, tiles, form, sm_count)
    launches = cake_jit.resolved_launches(
        form=form,
        splits=splits,
        arch=arch,
        wide=wide,
        rows=num_rows,
        kv_heads=num_kv_heads,
        qo_heads=num_qo_heads,
    )
    return splits, form == "fused", launches


# Slots of the three scratch tensors in the flat argument sequence (``cake_jit.LOGICAL_INDEX``).
_SLOT_PARTIAL_OUT = cake_jit.LOGICAL_INDEX["partial_out"]
_SLOT_PARTIAL_LSE = cake_jit.LOGICAL_INDEX["partial_lse"]
_SLOT_COUNTERS = cake_jit.LOGICAL_INDEX["merge_counters"]


def _launch(
    launches: tuple[
        tuple[Any, Callable[[Any], tuple[Any, ...]], tuple[int, int, int]], ...
    ],
    arguments: list[Any],
    device_index: int,
    stream: int,
) -> None:
    """Issue the resolved ``launches`` over the flat ``arguments`` under one environment-stream context on
    ``stream`` (torch's current stream of ``device_index``)."""
    with cake_jit.ffi_stream_context(device_index, stream):
        for entry, get, grid in launches:
            entry(*get(arguments), *grid)


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: Optional[torch.Tensor] = None,
    k_scale: Optional[float] = None,
    v_scale: Optional[float] = None,
    *,
    output_gate: torch.Tensor,
    k_sf: torch.Tensor,
    v_sf: torch.Tensor,
    workspace_buffer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Gated sparse paged GQA attention over an NVFP4 KV cache (the contract of
    ``flashinfer.sparse.qsa_sparse_paged_attention``), through the Cake kernels.

    The arguments are the public entry's; ``workspace_buffer`` is the caller-owned scratch of the module docstring
    ("Workspace rule"), ``None`` for the module's per-stream scratch.
    """
    del (
        use_prefill_config
    )  # kept for the signature; the split rule does not depend on it
    # Every check raises before any launch; each predicate reads tensor metadata once.
    q_device = q.device
    if q_device.type != "cuda":
        raise ValueError("device must be a cuda device")
    device_index = q_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _device_facts(device_index)
    q_shape = q.shape
    k_shape = k_cache.shape
    if len(q_shape) != 3 or len(k_shape) != 4:
        raise ValueError("q must be 3D and k_cache 4D")
    num_rows, num_qo_heads, head_dim = q_shape
    num_blocks, page_size, num_kv_heads, packed_dim = k_shape
    if q.dtype != torch.bfloat16:
        raise ValueError("q must be bfloat16")
    if head_dim != HEAD_DIM:
        raise ValueError(f"head_dim must be {HEAD_DIM}")
    if packed_dim * 2 != head_dim:
        raise ValueError("k_cache must hold head_dim / 2 packed bytes")
    k_stride = k_cache.stride()
    if not (
        k_cache.dtype == torch.uint8
        and v_cache.shape == k_shape
        and v_cache.dtype == k_cache.dtype
        and v_cache.stride() == k_stride
    ):
        raise ValueError(
            "k_cache and v_cache must be uint8 views with the same shape and strides"
        )
    sf_shape = (num_blocks, page_size, num_kv_heads, head_dim // 16)
    sf_stride = k_sf.stride()
    if not (
        k_sf.dtype == torch.float8_e4m3fn
        and k_sf.shape == sf_shape
        and v_sf.shape == k_sf.shape
        and v_sf.dtype == k_sf.dtype
        and v_sf.stride() == sf_stride
    ):
        raise ValueError(
            f"k_sf and v_sf must be float8_e4m3fn views of shape {sf_shape} with the same strides"
        )
    if not (num_blocks > 0 and page_size > 0):
        raise ValueError("k_cache must not be empty")
    if not (
        num_qo_heads % num_kv_heads == 0
        and num_qo_heads // num_kv_heads <= MAX_GROUP_SIZE
    ):
        raise ValueError(
            f"num_qo_heads must be a multiple of num_kv_heads, at most {MAX_GROUP_SIZE}x"
        )
    indices_shape = logical_indices.shape
    indices_stride = logical_indices.stride()
    if not (
        len(indices_shape) == 2
        and indices_shape[0] == num_rows
        and indices_shape[1] >= 2
        and logical_indices.dtype == torch.int32
        and indices_stride[1] == 1
    ):
        raise ValueError(
            "logical_indices must be int32 [rows, selection_width + 1] with a contiguous last dim"
        )
    table_shape = block_table.shape
    table_stride = block_table.stride()
    if not (
        len(table_shape) == 2
        and table_shape[0] > 0
        and table_shape[1] > 0
        and block_table.dtype == torch.int32
        and table_stride[1] == 1
    ):
        raise ValueError(
            "block_table must be a non-empty int32 2D tensor with a contiguous last dim"
        )
    if not (
        token_to_req.shape == (num_rows,)
        and token_to_req.dtype == torch.int32
        and token_to_req.is_contiguous()
    ):
        raise ValueError("token_to_req must be contiguous int32 [rows]")
    if k_scale is None or v_scale is None:
        raise ValueError("k_scale and v_scale are required")
    if not (
        output_gate.shape == q_shape
        and output_gate.dtype == torch.bfloat16
        and output_gate.is_contiguous()
    ):
        raise ValueError("output_gate must be contiguous bfloat16 with the shape of q")
    if out is None:
        out = torch.empty_like(q)
    if not (out.shape == q_shape and out.dtype == q.dtype):
        raise ValueError("out must match q in shape and dtype")
    if not (
        k_cache.device == q_device
        and v_cache.device == q_device
        and k_sf.device == q_device
        and v_sf.device == q_device
        and logical_indices.device == q_device
        and block_table.device == q_device
        and token_to_req.device == q_device
        and output_gate.device == q_device
        and out.device == q_device
    ):
        raise ValueError("all tensors must be on the same device")
    # The kernels move rows in 8- or 16-byte vectors: base and every outer stride must be aligned (the BF16
    # tensors' strides in elements of two bytes).  ``v_cache`` and ``v_sf`` carry the dtype and strides of
    # ``k_cache`` and ``k_sf`` (checked above), so their base alone remains to check.
    q_stride = q.stride()
    if not (
        q_stride[2] == 1
        and q.data_ptr() % 8 == 0
        and q_stride[0] * 2 % 8 == 0
        and q_stride[1] * 2 % 8 == 0
    ):
        raise ValueError(
            "q needs a contiguous last dim and 8-byte aligned base and strides"
        )
    if not (
        k_stride[3] == 1
        and k_cache.data_ptr() % 16 == 0
        and k_stride[0] % 16 == 0
        and k_stride[1] % 16 == 0
        and k_stride[2] % 16 == 0
    ):
        raise ValueError(
            "k_cache needs a contiguous last dim and 16-byte aligned base and strides"
        )
    if v_cache.data_ptr() % 16 != 0:
        raise ValueError(
            "v_cache needs a contiguous last dim and 16-byte aligned base and strides"
        )
    if not (
        sf_stride[3] == 1
        and k_sf.data_ptr() % 16 == 0
        and sf_stride[0] % 16 == 0
        and sf_stride[1] % 16 == 0
        and sf_stride[2] % 16 == 0
    ):
        raise ValueError(
            "k_sf needs a contiguous last dim and 16-byte aligned base and strides"
        )
    if v_sf.data_ptr() % 16 != 0:
        raise ValueError(
            "v_sf needs a contiguous last dim and 16-byte aligned base and strides"
        )
    gate_stride = output_gate.stride()
    if not (
        gate_stride[2] == 1
        and output_gate.data_ptr() % 16 == 0
        and gate_stride[0] * 2 % 16 == 0
        and gate_stride[1] * 2 % 16 == 0
    ):
        raise ValueError(
            "output_gate needs a contiguous last dim and 16-byte aligned base and strides"
        )
    out_stride = out.stride()
    if not (
        out_stride[2] == 1
        and out.data_ptr() % 16 == 0
        and out_stride[0] * 2 % 16 == 0
        and out_stride[1] * 2 % 16 == 0
    ):
        raise ValueError(
            "out needs a contiguous last dim and 16-byte aligned base and strides"
        )
    if num_rows == 0:
        return out

    selection_width = indices_shape[1] - 1
    tiles = -(-selection_width // TILE_TOKENS)
    splits, fused, launches = _resolve(
        num_rows, num_kv_heads, num_qo_heads, tiles, device_index
    )
    raw = (
        _check_workspace(
            workspace_buffer, q_device, num_rows, num_qo_heads, num_kv_heads, splits
        )
        if workspace_buffer is not None
        else None
    )
    # The flat argument sequence, ``cake_jit.LOGICAL_TENSORS`` then ``cake_jit.LOGICAL_SCALARS``; the three
    # scratch slots are filled below.
    arguments = [
        q,
        k_cache,
        v_cache,
        k_sf.view(torch.uint8),
        v_sf.view(torch.uint8),
        logical_indices,
        block_table,
        token_to_req,
        output_gate,
        out,
        None,  # partial_out
        None,  # partial_lse
        None,  # merge_counters
        q_stride[0],
        q_stride[1],
        k_stride[0],
        k_stride[1],
        k_stride[2],
        sf_stride[0],
        sf_stride[1],
        sf_stride[2],
        indices_stride[0],
        table_stride[0],
        out_stride[0],
        out_stride[1],
        float(k_scale) * _LOG2E / math.sqrt(head_dim),  # score_scale
        float(v_scale),  # output_scale
        num_rows,
        table_shape[0],  # num_requests
        num_blocks,
        page_size,  # page_div
        table_shape[1],  # table_width
        selection_width,  # topk
        num_qo_heads // num_kv_heads,  # group_size
        num_qo_heads,
        tiles,
        splits,
    ]
    stream = _current_raw_stream(device_index)
    if splits == 1:
        _launch(launches, arguments, device_index, stream)
    elif raw is not None:
        (
            arguments[_SLOT_PARTIAL_OUT],
            arguments[_SLOT_PARTIAL_LSE],
            arguments[_SLOT_COUNTERS],
        ) = _carve(raw, num_rows, num_qo_heads, num_kv_heads, splits, counters=fused)
        _launch(launches, arguments, device_index, stream)
    elif _current_stream_capturing() or device_index != _current_device():
        # The graph's private pool: a capture records no buffer the host may later replace.  The
        # capture state is readable for the current device's stream only, so a call on another
        # device takes this per-call path as well.
        arguments[_SLOT_PARTIAL_OUT] = q.new_empty(
            (splits, num_rows, num_qo_heads, head_dim), dtype=torch.float32
        )
        arguments[_SLOT_PARTIAL_LSE] = q.new_empty(
            (splits, num_rows, num_qo_heads), dtype=torch.float32
        )
        if fused:
            arguments[_SLOT_COUNTERS] = torch.zeros(
                num_rows * num_kv_heads, dtype=torch.uint32, device=q_device
            )
        _launch(launches, arguments, device_index, stream)
    else:
        key = (device_index, stream)
        with _SCRATCH_LOCK:
            arguments[_SLOT_PARTIAL_OUT], arguments[_SLOT_PARTIAL_LSE] = (
                _stream_partials_views(q_device, key, splits, num_rows, num_qo_heads)
            )
            if fused:
                arguments[_SLOT_COUNTERS] = _stream_counters_buffer(
                    q_device, key, num_rows * num_kv_heads
                )
            _launch(launches, arguments, device_index, stream)
    return out


__all__ = [
    "ALIGN",
    "CTAS_PER_SM",
    "HEAD_DIM",
    "LONG_STREAM_TILES",
    "MAX_GROUP_SIZE",
    "MAX_SPLITS",
    "SPLIT_OVERRIDES",
    "TILE_TOKENS",
    "fused_capacity",
    "is_supported",
    "num_splits",
    "num_tiles",
    "qsa_sparse_paged_attention",
    "qsa_workspace_bytes",
    "reset_workspace",
    "select_form",
    "select_wide",
]
