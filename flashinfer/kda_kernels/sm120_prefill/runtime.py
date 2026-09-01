# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Host-side runtime shared by the ``decomp`` and ``fused`` prefill variants.

Both variants implement the same operation and neither imports the other, so
what they both consume on the host lives here:

* the shape constants and numeric floors of the contract, and the error type
  both raise;
* the canonical launch description and the shape/dtype/alias validation that
  produces it;
* the exact ``sm_120a`` target check, expressed through
  :mod:`flashinfer.cute_dsl.utils` and never by writing ``CUTE_DSL_ARCH``;
* the bounded per-device cache, the capture probe, tensor identity and pinned
  descriptor staging -- the *containers*, not the cache instances, which stay
  with the variant that keys them;
* canonical INT32 ``cu_seqlens``, the workspace resource slot, the graph
  stream/signature binding and the resource lifetime that goes with it;
* the naming convention for :func:`~flashinfer.jit.build_and_load_cute_dsl_kernel`.

Importing this module loads no device code, compiles nothing and does not
import ``cutlass``: the facade and the public dispatcher must be able to
answer "not eligible" on any host.  Device-side helpers the two variants share
-- PTX wrappers, TMA loads and stores, shared-memory layout constants -- live
in ``device_common.py``, which does import the CuTe DSL.  Layouts, swizzles,
TMA descriptor construction and the kernels themselves stay with their
variant: helpers with matching names in ``decomp`` and ``fused`` index
different shapes, and hoisting one would be a silent numerical change.
"""

from __future__ import annotations

import importlib.util
import math
import ctypes
import os
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

# ---------------------------------------------------------------------------
# Shape constants and scalars both variants agree on.
#
# These are properties of the operation, not of either schedule: KDA prefill on
# this backend is equal-head, 128-wide in both K and V.  The chunk size, the
# SMEM arena and every swizzle are schedule properties and live with their
# variant.
# ---------------------------------------------------------------------------

#: Key dimension.  Fixed; the eligibility predicate rejects anything else.
DK = 128

#: Value dimension.  Equal to :data:`DK` for every shape this backend supports.
DV = 128

#: log2(e).  The gate is evaluated in the log2 domain by both variants.
LOG2_E = 1.4426950408889634

#: Clamp on the gate's chunk prefix, in the log2 domain.  Here rather than in
#: each variant because it is a numeric boundary of the contract, not a guard:
#: both variants must apply the same floor, and one definition cannot drift.
PREFIX_FLOOR = -126.0

#: Floor on the Q/K L2 sum of squares before its reciprocal square root.
#: Shared for the same reason as :data:`PREFIX_FLOOR`.
NORM_FLOOR = 1.0e-24

#: The safe gate's worst-case chunk prefix is ``16 * lower_bound * log2e``,
#: which reaches the ``rcp.approx.ftz`` cliff at ``lower_bound == -5.4585``.
#: The supported range keeps a real margin below that.  This direct backend ABI
#: also accepts the degenerate zero gate; the public ``recurrent_kda`` selector
#: deliberately requires a strictly negative bound.
LOWER_BOUND_RANGE = (-5.0, 0.0)

#: Base alignment every tensor a TensorMap describes must satisfy.  The driver
#: accepts a misaligned base and the corruption surfaces as wrong numbers in
#: one head, far from the call that caused it.
GLOBAL_BASE_ALIGN = 16

INT32_MAX = 2**31 - 1

#: Inputs that are only ever read, so overlaps among them are legal.
READ_ONLY_ROLES = ("q", "k", "v", "g", "beta", "A_log", "dt_bias")


class KDAPrefillValidationError(ValueError):
    """Raised for any violation of the SM120 backend ABI.

    A ``ValueError`` subclass rather than a bare one so a caller can tell a
    contract violation from an unrelated failure, and so the public adapter in
    ``flashinfer/kda_prefill.py`` can let it propagate unchanged.

    The line this draws, since both variants also raise plain ``ValueError``
    and the split is not obvious from a count: this type is for what the
    *caller* got wrong -- everything :func:`validate_inputs` checks, plus the
    two cross-cutting checks a variant owns (``cu_seqlens`` disagreeing with
    the state shapes, and ``safe_gate=False`` on ``decomp``).  A plain
    ``ValueError`` is for an invariant of this backend's own construction: a
    TMA spec it built itself, a workspace geometry, a derived grid extent, a
    config object.  A well-formed call cannot reach one, so raising the
    contract type there would tell a caller their inputs were bad when they
    were not.
    """


class UnsupportedArchitectureError(RuntimeError):
    """Raised when the SM120 backend is asked to run on another target."""


# ---------------------------------------------------------------------------
# Architecture and compile target.
#
# Two separate questions, and both must hold:
#
#   1. is the *device* compute capability 12.0?
#   2. can the installed CuTe DSL and CUDA toolkit compile and load ``sm_120a``
#      natively -- not a family-conditional ``sm_120f`` fallback?
#
# Neither is answered by mutating ``CUTE_DSL_ARCH``: the DSL captures its
# default target when ``cutlass`` is first imported, so a write here would not
# retarget an already-imported DSL, and a write before import would change
# every other CuTe-DSL kernel in the process.
# ---------------------------------------------------------------------------

SM120_CAPABILITY = (12, 0)

#: The code target both variants compile for.  Spelled once; the persistent
#: cache's own arch resolution is checked against it before any build.
SM120_CODE_TARGET = "sm_120a"


def _capability(device: torch.device | None = None) -> tuple[int, int]:
    from ...utils import get_compute_capability

    if device is None:
        device = torch.device("cuda", torch.cuda.current_device())
    return get_compute_capability(device)


def sm120a_available(device: torch.device | None = None) -> bool:
    """Can this process compile and run ``sm_120a`` on ``device``?

    A fail-closed predicate: it allocates nothing, launches nothing and
    synchronizes with nothing, so the eligibility path can call it.  Anything
    unexpected -- no driver, no DSL, an ``Arch`` enum that does not know
    ``sm_120a`` -- reads as "no".
    """
    if not torch.cuda.is_available():
        return False
    try:
        if _capability(device) != SM120_CAPABILITY:
            return False
        from ...cute_dsl.availability import is_cute_dsl_arch_supported
        from ...utils import is_sm120a_supported

        if not is_sm120a_supported(
            device or torch.device("cuda", torch.cuda.current_device())
        ):
            return False
        # native_only: a family-conditional ``sm_120f`` target is not what this
        # backend compiles for, and accepting it would produce an artifact
        # named sm_120a that is not.
        return is_cute_dsl_arch_supported(*SM120_CAPABILITY, native_only=True)
    except Exception:  # noqa: BLE001 -- an unavailable backend is not an error
        return False


def require_sm120a(device: torch.device | str | int | None = None) -> None:
    """Raise unless ``device`` is CC 12.0 and ``sm_120a`` is buildable."""
    if not torch.cuda.is_available():
        raise UnsupportedArchitectureError(
            "the SM120 KDA prefill backend requires a CUDA device"
        )
    normalized = (
        None
        if device is None
        else torch.device(device)
        if not isinstance(device, int)
        else torch.device("cuda", device)
    )
    capability = _capability(normalized)
    if capability != SM120_CAPABILITY:
        major, minor = capability
        raise UnsupportedArchitectureError(
            f"the SM120 KDA prefill backend requires compute capability 12.0, "
            f"got sm_{major}{minor}"
        )
    if not sm120a_available(normalized):
        raise UnsupportedArchitectureError(
            f"the installed CuTe DSL and CUDA toolkit cannot natively target "
            f"{SM120_CODE_TARGET} on this device"
        )


def sm120a_compile_options(enable_tvm_ffi: bool = True) -> tuple:
    """``cute.compile`` options pinning the code target to ``sm_120a``.

    An explicit :class:`cute.GPUArch` rather than an environment variable, for
    the reason above.  ``EnableTVMFFI`` is not optional either: it selects the
    argument-marshalling ABI, and the ctypes fallback is several times slower
    to invoke.

    **These must be passed with ``cute.compile[options](...)``, not
    ``cute.compile(..., options=options)``.**  The keyword form accepts the
    tuple and silently ignores ``EnableTVMFFI``: it yields a
    ``CudaDialectJitCompiledFunction``, which marshals every argument through
    ``ctypes.addressof``, where the subscript form yields a
    ``TVMFFIJitCompiledFunctionWithKwargs``.  The mistake produces no error,
    only a slower call, so :func:`assert_tvm_ffi_dispatched` checks the result.
    """
    import cutlass.cute as cute

    options: tuple = (cute.GPUArch(SM120_CODE_TARGET),)
    if enable_tvm_ffi:
        options = (cute.EnableTVMFFI(True),) + options
    return options


def assert_tvm_ffi_dispatched(compiled, kernel_name: str):
    """Refuse a compiled entry that fell back to the ctypes argument path.

    The fallback is not an error the DSL reports -- it produces a working
    callable that is simply slower to invoke.  Checking the type is cheap and
    happens once per specialization.
    """
    compiled_type = type(compiled)
    known_tvm_ffi_types = {
        "TVMFFIJitCompiledFunction",
        "TVMFFIJitCompiledFunctionWithKwargs",
    }
    if (
        compiled_type.__module__.endswith("tvm_ffi_provider")
        or compiled_type.__name__ in known_tvm_ffi_types
    ):
        return compiled
    raise RuntimeError(
        f"the compiled entry for {kernel_name!r} is a "
        f"{compiled_type.__name__}, not a TVM-FFI callable: the compile "
        f"options did not take. Pass them as cute.compile[options](...) -- the "
        f"options= keyword accepts EnableTVMFFI and ignores it."
    )


# ---------------------------------------------------------------------------
# Persistent JIT.
#
# The op namespace is fixed and distinct from every other KDA entry point, so
# this backend can never load another implementation's cached artifact.
# ---------------------------------------------------------------------------

#: Op-family name of the persistent cache module.  An identifier, like the
#: other adopters' names, because it also prefixes the exported C symbol.
JIT_MODULE_NAME = "kda_prefill_sm120"


def _module_key_files() -> tuple:
    """Every source file whose content should invalidate this module.

    Every package file, for both variants, and deliberately the *same* tuple
    for every kernel in the namespace.  ``JitSpecCuteDsl`` writes one
    ``meta.json`` per module directory and wipes the directory whenever a
    kernel arrives with a different source hash, so kernels in one namespace
    keyed on different files would invalidate each other.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return tuple(
        os.path.join(here, name)
        for name in (
            "runtime.py",
            "device_common.py",
            "decomp.py",
            "fused.py",
            "__init__.py",
        )
    )


def build_kernel(kernel_name: str, compile_fn, *, device, key_files=()):
    """Compile through FlashInfer's persistent CuTe-DSL cache.

    ``kernel_name`` is the specialization -- variant, dtype presence flags and
    every other compile-time parameter, and nothing that varies per call.  A
    tensor address or a runtime shape in this string is a cache that never
    hits.  ``key_files`` is accepted and ignored: invalidation is a property of
    the module namespace, not of one kernel in it, so it always uses
    :func:`_module_key_files`.

    The cache infrastructure already degrades export and load failures to a
    warning.  What can still escape it is a filesystem or lock problem with the
    cache directory itself, and only that is caught here: the kernel is then
    compiled in-process without caching, with a warning.  A compile error
    propagates unchanged rather than being retried.
    """
    # The object module is loaded and a cold compile runs against the current
    # CUDA context.  The public API accepts a tensor on a non-current device,
    # so both must happen under the input tensor's device guard rather than
    # whichever device the caller happened to leave current.
    with torch.cuda.device(device):
        from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        del key_files
        try:
            return build_and_load_cute_dsl_kernel(
                JIT_MODULE_NAME,
                kernel_name,
                compile_fn,
                extra_key_files=_module_key_files(),
                arch=SM120_CODE_TARGET,
            )
        except OSError as exc:
            from ...jit.core import logger

            logger.warning(
                "CuTe-DSL persistent cache unavailable for %s (%s: %s); "
                "compiling in-process without caching",
                kernel_name,
                type(exc).__name__,
                exc,
            )
            return compile_fn()


# ---------------------------------------------------------------------------
# Capture probe and stream-correct cache lifetime.
#
# Cached device payloads -- TMA descriptor sets, chunk metadata, canonical
# INT32 offsets -- are read asynchronously by a kernel, so they need three
# properties, and the same three for both variants.
#
# **Bounded.**  Per device, with an entry ceiling and optionally a payload
# ceiling.  Whichever binds first evicts from the LRU tail.  An unbounded cache
# keyed on tensor addresses is a leak with a slow fuse.
#
# **Stream-correct on the way in.**  The upload happens on whichever stream
# created the entry, so a later hit on another stream issues ``wait_event``
# before reading it.  That is a device-side ordering edge, not a host
# synchronization.
#
# **Stream-correct on the way out.**  Eviction never synchronizes; every hit
# calls ``record_stream``, so the allocator waits for the streams that used the
# block before reclaiming it.
# ---------------------------------------------------------------------------


def capturing() -> bool:
    """Is the current stream inside a CUDA graph capture?

    The cache paths below skip their ``wait_event`` / ``record_stream`` pair
    under capture; ``record_stream`` in particular is a caching-allocator
    operation with no graph representation.  Skipping is correct, not a
    workaround: a capture replays only work it recorded, so an entry it reads
    was already live and reachable when the capture began -- there is no other
    stream to order against and no lifetime for the allocator to extend.
    """
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


def record_stream_once(tensors, stream) -> None:
    """``record_stream`` each distinct CUDA storage exactly once.

    Exact aliases -- ``out`` with ``v``, ``initial_state`` with
    ``final_state`` -- are recorded once rather than once per name.
    """
    seen: set[int] = set()
    for tensor in tensors:
        if tensor is None or not tensor.is_cuda:
            continue
        key = tensor.untyped_storage().data_ptr()
        if key in seen:
            continue
        seen.add(key)
        tensor.record_stream(stream)


@dataclass
class _Entry:
    value: Any
    storages: tuple
    #: ``None`` for a CPU payload: there is no upload to order against, and the
    #: host-only paths must not require a driver.
    event: Any
    nbytes: int


@dataclass
class CacheStats:
    entries: int = 0
    bytes: int = 0
    hits: int = 0
    misses: int = 0
    evictions: int = 0


#: Default entry ceiling for a per-device cache.
#:
#: Descriptor entries keep their source tensors addressable, so this bounds a
#: retention as well as a size.  See :data:`FLAT_VIEW_MAX_ENTRIES`: the caches
#: are redundant retainers and only bind together.
MAX_ENTRIES = 64


class BoundedDeviceCache:
    """LRU cache of device payloads, one bucket per CUDA device.

    Every mutating path holds :attr:`_lock`.  The individual dict operations are
    atomic under the GIL, but the pairs are not -- a ``get`` plus its
    ``move_to_end``, or an insert racing the eviction loop -- and this runs on
    the pre-launch host path, which ``flashinfer/kda_prefill.py`` reaches with
    no lock of its own whenever the caller passes no workspace.  Reentrant
    because evicting an entry drops its storages, and a weakref callback on the
    releasing thread can come back through this class.
    """

    #: Bucket index standing for "not a CUDA device".
    CPU_BUCKET = -1

    def __init__(
        self,
        name: str,
        *,
        max_entries: int = MAX_ENTRIES,
        max_bytes: Optional[int] = None,
    ):
        if max_entries < 1:
            raise ValueError(f"{name}: max_entries must be positive")
        self.name = name
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._buckets: dict = {}
        self._stats: dict = {}
        self._lock = threading.RLock()

    @classmethod
    def _index(cls, device) -> int:
        if isinstance(device, int):
            return device
        if device.type != "cuda":
            return cls.CPU_BUCKET
        if device.index is None:
            return torch.cuda.current_device()
        return device.index

    def _bucket(self, index: int):
        return self._buckets.setdefault(index, OrderedDict())

    def stats(self, device) -> CacheStats:
        return self._stats.setdefault(self._index(device), CacheStats())

    def _evict(self, index: int) -> None:
        bucket = self._bucket(index)
        stats = self.stats(index)
        while bucket and (
            len(bucket) > self.max_entries
            or (self.max_bytes is not None and stats.bytes > self.max_bytes)
        ):
            _, entry = bucket.popitem(last=False)
            stats.bytes -= entry.nbytes
            stats.evictions += 1
            # No synchronization: the recorded streams are what make the
            # allocator wait for any launch still reading the block.
        stats.entries = len(bucket)

    def get(self, device, key) -> Any:
        index = self._index(device)
        with self._lock:
            bucket = self._bucket(index)
            stats = self.stats(index)
            entry = bucket.get(key)
            if entry is None:
                stats.misses += 1
                return None
            bucket.move_to_end(key)
            stats.hits += 1
        if entry.event is not None and not capturing():
            stream = torch.cuda.current_stream(index)
            # Cheap and correct on the creating stream too: an event recorded
            # on the same stream is already satisfied.
            stream.wait_event(entry.event)
            record_stream_once(entry.storages, stream)
        return entry.value

    def put(self, device, key, value: Any, storages: tuple = ()) -> Any:
        index = self._index(device)
        event = None
        if index != self.CPU_BUCKET and torch.cuda.is_available() and not capturing():
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(index))
        nbytes = sum(t.numel() * t.element_size() for t in storages)
        with self._lock:
            bucket = self._bucket(index)
            stats = self.stats(index)
            if key in bucket:
                stats.bytes -= bucket.pop(key).nbytes
            bucket[key] = _Entry(
                value=value, storages=tuple(storages), event=event, nbytes=nbytes
            )
            stats.bytes += nbytes
            self._evict(index)
        return value

    def contains(self, device, key) -> bool:
        """Membership without touching LRU order, hit counts or streams.

        The graph-capture warmth check needs to ask "is this already warm?"
        without the side effects a real hit has.
        """
        with self._lock:
            return key in self._bucket(self._index(device))

    def clear(self, device=None) -> None:
        """Drop entries without synchronizing."""
        with self._lock:
            if device is None:
                self._buckets.clear()
                self._stats.clear()
                return
            index = self._index(device)
            self._buckets.pop(index, None)
            self._stats.pop(index, None)


class GraphResourcePins:
    """Strong references to anything a CUDA graph captured.

    From the start of capture until the graph is destroyed, the compiled
    artifact, the descriptor storage, the canonical offsets and the validation
    record must stay alive at their captured addresses.  Replay never re-enters
    Python, so no hook could renew an LRU position: the only safe policy is to
    leave the LRU.

    Ownership is the caller's workspace, not this process-wide table -- see
    :class:`SM120PrefillResources`.  This exists for the pins that outlive an
    individual workspace, and it deliberately never shrinks.
    """

    def __init__(self) -> None:
        self._pinned: dict = {}

    def pin(self, key, *objects) -> None:
        existing = self._pinned.get(key, ())
        self._pinned[key] = existing + tuple(o for o in objects if o is not None)

    def is_pinned(self, key) -> bool:
        return key in self._pinned

    def __len__(self) -> int:
        return len(self._pinned)

    def clear(self) -> None:
        """Drop every pin.  For tests only -- a live graph makes this unsafe."""
        self._pinned.clear()


#: The single process-wide pin table.
GRAPH_PINS = GraphResourcePins()


@dataclass
class IdentityCache:
    """Weak-reference lookup keyed on ``(id, version)`` of a source tensor.

    Validating ``cu_seqlens`` needs a device-to-host copy, which synchronizes.
    A caller passing the same tensor object every step must not pay that again,
    and must still see a rebuild if the tensor is mutated in place -- hence the
    version check alongside the weak reference.  Values must be lightweight
    secondary-cache keys: retaining the source tensor or a device payload here
    would defeat the weak reference and bypass the bounded cache's stream
    ordering and eviction policy.
    """

    _entries: dict = field(default_factory=dict)

    def get(self, tensor: torch.Tensor):
        cached = self._entries.get(id(tensor))
        if cached is None:
            return None
        ref, version, value = cached
        if ref() is tensor and version == tensor_version(tensor):
            return value
        self._entries.pop(id(tensor), None)
        return None

    def put(self, tensor: torch.Tensor, value) -> None:
        key = id(tensor)

        def _purge(_ref, _key=key):
            self._entries.pop(_key, None)

        self._entries[key] = (
            weakref.ref(tensor, _purge),
            tensor_version(tensor),
            value,
        )

    def drop(self, predicate) -> None:
        for key in [k for k, (_, _, v) in self._entries.items() if predicate(v)]:
            self._entries.pop(key, None)

    def clear(self) -> None:
        self._entries.clear()


# ---------------------------------------------------------------------------
# Flat CuTe views of torch tensors, cached on the address they describe.
#
# Every launch converts its tensors with ``from_dlpack(t.reshape(-1))``, and
# repeating that on tensors whose addresses have not moved is avoidable host
# time.  The conversion is a pure function of (pointer, element count, dtype,
# alignment), so keying on exactly those four is sound.  With
# ``enable_tvm_ffi=True``, the CuTe view owns a TVM-FFI DLPack consumer object;
# that object keeps the reshaped tensor's storage alive until the view is
# evicted.  The allocator therefore cannot recycle a live entry's address, and
# the LRU below bounds that retention.
# ---------------------------------------------------------------------------

#: Bounded so a workload cycling through many buffers cannot grow it without
#: limit.  A forward touches ~25 tensors, so this holds several shapes' worth.
#:
#: One of three caches that can hold a buffer alive -- the others are the plan
#: memo and :data:`MAX_ENTRIES` -- and any one of them is enough, so lowering
#: this alone does not shorten the retention; the three only bind together.
FLAT_VIEW_MAX_ENTRIES = 256

_FLAT_VIEWS: "OrderedDict[tuple, Any]" = OrderedDict()

#: Held on both paths.  ``get`` and ``move_to_end`` are each atomic under the
#: GIL but not jointly: an eviction or ``clear_flat_views`` between them raises
#: ``KeyError`` from ``move_to_end`` on an entry that was just hit.
_FLAT_VIEWS_LOCK = threading.RLock()


def _require_tvm_ffi() -> None:
    """``apache-tvm-ffi`` is not optional for this backend.

    The persistent cache reloads artifacts with
    ``load_module(..., enable_tvm_ffi=True)``, and the compiled entry rejects
    a view built without it (``'_Tensor' object has no attribute
    '_tvm_ffi_tensor'``), so a missing package is reported here instead.
    """
    if importlib.util.find_spec("tvm_ffi") is None:
        raise RuntimeError(
            "the SM120 KDA prefill backend requires `apache-tvm-ffi`, which "
            "FlashInfer already depends on; reinstall the package"
        )


def flat_view(tensor: torch.Tensor, *, align: int = 16):
    """``from_dlpack(tensor.reshape(-1))``, reused when the address repeats.

    The returned TVM-FFI view retains the reshape's storage under DLPack's
    consumer-ownership contract, so each cached entry pins one allocation.
    :data:`FLAT_VIEW_MAX_ENTRIES` bounds that retention.

    Safe inside ``torch.cuda.graph``: a dict lookup and, on a miss, the same
    conversion the caller would have done anyway.  It issues no CUDA work and
    records no events, unlike :class:`BoundedDeviceCache`.
    """
    if not tensor.is_contiguous():
        raise KDAPrefillValidationError(
            "flat_view requires a contiguous tensor; reshaping a strided "
            "tensor would create a copy at a different address"
        )

    from cutlass.cute.runtime import from_dlpack

    key = (tensor.data_ptr(), tensor.numel(), tensor.dtype, align)
    with _FLAT_VIEWS_LOCK:
        hit = _FLAT_VIEWS.get(key)
        if hit is not None:
            _FLAT_VIEWS.move_to_end(key)
            return hit
    _require_tvm_ffi()
    view = from_dlpack(tensor.reshape(-1), assumed_align=align, enable_tvm_ffi=True)
    # Under tvm-ffi the extent is part of the compiled entry's signature, so a
    # plan compiled for one sequence length would reject the next.  Keying the
    # compile cache on shape would fix the error and reintroduce one compile
    # per length; marking the single flat dimension dynamic keeps one entry.
    view = view.mark_layout_dynamic()
    with _FLAT_VIEWS_LOCK:
        _FLAT_VIEWS[key] = view
        while len(_FLAT_VIEWS) > FLAT_VIEW_MAX_ENTRIES:
            _FLAT_VIEWS.popitem(last=False)
    return view


def clear_flat_views() -> None:
    with _FLAT_VIEWS_LOCK:
        _FLAT_VIEWS.clear()


# ---------------------------------------------------------------------------
# Capture-safe descriptor upload.
#
# A descriptor build copies from pageable host memory, which is harmless
# outside a capture and fatal inside one:
#
#     RuntimeError: Cannot copy between CPU and CUDA tensors during CUDA graph
#     capture unless the CPU tensor is pinned.
#
# A descriptor build is supposed to be a cache miss only, but a capture runs on
# its own stream, so anything keyed by stream misses exactly when it matters.
# Staging through pinned memory makes the upload legal either way.
# ---------------------------------------------------------------------------

#: Pinned staging buffers, one per size, each paired with the event that says
#: its last upload has been read.  Descriptor blobs are a few hundred bytes and
#: come in a handful of sizes, so this stays tiny.
#:
#: The event is what makes the reuse safe.  A ``non_blocking`` copy out of
#: pinned memory returns before the transfer runs -- it is queued behind
#: whatever else is on the stream -- so refilling the buffer for the next
#: descriptor build would overwrite bytes the DMA has not read yet, and the
#: descriptor that reached the device would be a mix of two.  Two builds of one
#: size is the common case rather than a corner: the sizes are a function of
#: the descriptor count, so any two cold calls of the same shape collide.
#: Waiting here costs nothing a steady state pays, because a build is a cache
#: miss only.
_PINNED_STAGING: dict = {}
_PINNED_STAGING_LOCK = threading.RLock()

#: Staging buffers whose upload was captured into a CUDA graph.  A captured H2D
#: node reads its source at *replay*, so such a buffer can never be refilled --
#: no host wait exists to place before a replay.  It leaves the pool instead and
#: is held for the process's lifetime; the alternative is a graph that uploads
#: whichever descriptor was built last.
_CAPTURED_STAGING: list = []


def upload_bytes(payload, device: torch.device) -> torch.Tensor:
    """Copy ``payload`` to ``device`` in a way a graph capture accepts."""
    size = len(payload)
    if device.type != "cuda" or not torch.cuda.is_available():
        # No device to pin against and no capture to be safe for.
        return torch.frombuffer(bytearray(payload), dtype=torch.uint8).clone()
    # Keep ownership from checkout through publication.  Two cold builds of
    # the same descriptor size otherwise both see an empty slot and the later
    # publication orphans the earlier buffer while its H2D copy is in flight.
    with _PINNED_STAGING_LOCK:
        entry = _PINNED_STAGING.pop(size, None)
        if entry is None:
            staging = torch.empty(size, dtype=torch.uint8, pin_memory=True)
        else:
            staging, pending = entry
            if pending is not None:
                pending.synchronize()
        # frombuffer needs a writable buffer; bytes is not one.
        staging.copy_(torch.frombuffer(bytearray(payload), dtype=torch.uint8))
        out = torch.empty(size, dtype=torch.uint8, device=device)
        out.copy_(staging, non_blocking=True)
        if capturing():
            _CAPTURED_STAGING.append(staging)
            return out
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(device))
        _PINNED_STAGING[size] = (staging, event)
        return out


def clear_pinned_staging() -> None:
    """Drain and drop the pool; captured buffers remain pinned for replay."""
    with _PINNED_STAGING_LOCK:
        for _staging, pending in _PINNED_STAGING.values():
            if pending is not None:
                pending.synchronize()
        _PINNED_STAGING.clear()


# ---------------------------------------------------------------------------
# Storage-range aliasing.
#
# Aliasing is decided on byte ranges, not on storage identity: two tensors can
# share a storage object and never overlap, and two tensors from different
# allocations can be views of one block.
# ---------------------------------------------------------------------------


def storage_interval(tensor: Optional[torch.Tensor]):
    """``[data_ptr, data_ptr + numel * element_size)``; ``None`` when empty.

    Every tensor reaching this point is contiguous, so the interval is exactly
    the bytes the tensor owns.  A zero-element tensor owns nothing and cannot
    alias anything.
    """
    if tensor is None or tensor.numel() == 0:
        return None
    start = tensor.data_ptr()
    return (start, start + tensor.numel() * tensor.element_size())


def intervals_overlap(a, b) -> bool:
    if a is None or b is None:
        return False
    return a[0] < b[1] and b[0] < a[1]


def is_exact_alias(x: Optional[torch.Tensor], y: Optional[torch.Tensor]) -> bool:
    """Same bytes, dtype, shape and stride -- the only reuse either variant allows."""
    if x is None or y is None:
        return False
    return (
        storage_interval(x) == storage_interval(y)
        and x.dtype == y.dtype
        and x.shape == y.shape
        and x.stride() == y.stride()
    )


# Both variants check their grid against the device limits, so the driver query
# lives here with the other shared device queries.
_GRID_LIMITS: dict[int, tuple[int, int]] = {}


def max_grid_dims(device: torch.device) -> tuple[int, int]:
    """``(maxGridSize[0], maxGridSize[1])`` for ``device``.

    ``torch.cuda.get_device_properties`` does not expose the grid limits, so
    this goes to the driver.  The result is cached because the check runs on
    every launch, before anything is allocated.
    """
    index = torch.cuda.current_device() if device.index is None else device.index
    cached = _GRID_LIMITS.get(index)
    if cached is not None:
        return cached

    import cuda.bindings.driver as drv

    attributes = (
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X,
        drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y,
    )
    values = []
    for attribute in attributes:
        err, value = drv.cuDeviceGetAttribute(attribute, index)
        if int(err) != 0:
            raise RuntimeError(f"cuDeviceGetAttribute failed: {err}")
        values.append(int(value))
    limits = (values[0], values[1])
    _GRID_LIMITS[index] = limits
    return limits


def check_tma_base_alignment(named: dict) -> None:
    """Every tensor a TensorMap describes must have a 16-byte-aligned base."""
    for name in ("q", "k", "v", "g", "out", "initial_state", "final_state"):
        tensor = named.get(name)
        if tensor is None or tensor.numel() == 0:
            continue
        if tensor.data_ptr() % GLOBAL_BASE_ALIGN:
            raise KDAPrefillValidationError(
                f"{name} must be {GLOBAL_BASE_ALIGN}-byte aligned for TMA, got "
                f"{tensor.data_ptr():#x}"
            )


# ---------------------------------------------------------------------------
# The canonical launch description.
#
# Both variants accept the same public arguments and reduce them to the same
# facts before doing anything variant-specific.  Producing that reduction once
# is what keeps the two from disagreeing about, say, whether a zero-token call
# has zero sequences or one.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalInputs:
    """What the validated arguments say about the launch."""

    input_mode: str
    batch: int
    tokens_per_sequence: int
    sequences: int
    heads: int
    total_tokens: int
    g_fp32: bool
    state_dtype: Optional[torch.dtype]
    has_initial_state: bool
    has_final_state: bool
    out_aliases_v: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KDAPrefillValidationError(message)


def _check_device_and_contiguity(named: dict) -> torch.device:
    present = {n: t for n, t in named.items() if t is not None}
    _require(bool(present), "no tensors supplied")
    devices = {n: t.device for n, t in present.items()}
    first_name, device = next(iter(devices.items()))
    _require(device.type == "cuda", f"{first_name} must be on a CUDA device")
    for name, dev in devices.items():
        _require(
            dev == device,
            f"all tensors must share one CUDA device: {first_name} is on "
            f"{device}, {name} is on {dev}",
        )
    for name, tensor in present.items():
        _require(tensor.is_contiguous(), f"{name} must be contiguous")
    return device


def _check_dtypes(named: dict) -> None:
    for name in ("q", "k", "v", "out"):
        tensor = named[name]
        _require(
            tensor.dtype is torch.bfloat16, f"{name} must be BF16, got {tensor.dtype}"
        )
    _require(
        named["g"].dtype in (torch.bfloat16, torch.float32),
        f"g must be BF16 or FP32, got {named['g'].dtype}",
    )
    _require(
        named["beta"].dtype is torch.bfloat16,
        f"beta must be BF16, got {named['beta'].dtype}",
    )
    for name in ("A_log", "dt_bias"):
        _require(
            named[name].dtype is torch.float32,
            f"{name} must be FP32, got {named[name].dtype}",
        )


def _check_aliasing(named: dict, out_aliases_v: bool) -> None:
    """Refuse every overlap except the two the kernels' schedules prove safe.

    ``out`` may alias ``v``, since a chunk's output is published only after
    that chunk's V has been loaded and consumed; ``initial_state`` may alias
    ``final_state``, since all initial loads finish before the chunk loop and
    the final store happens after it drains.  Both are *exact* aliases only: a
    partial overlap is refused, because the overwrite proof covers exactly the
    exact-alias case and nothing else.

    Read-only inputs may overlap each other freely -- a caller broadcasting one
    buffer into several of them is fine.  The moment ``out`` aliases ``v``,
    though, ``out`` still has to be disjoint from all the others.
    """
    out = named["out"]
    out_range = storage_interval(out)

    for name in READ_ONLY_ROLES + ("cu_seqlens", "initial_state", "final_state"):
        other = named.get(name)
        if other is None:
            continue
        if name == "v" and out_aliases_v:
            continue
        _require(
            not intervals_overlap(out_range, storage_interval(other)),
            f"out must not overlap {name}; only an exact alias with v is allowed",
        )

    initial = named.get("initial_state")
    final = named.get("final_state")
    if initial is not None and final is not None:
        _require(
            initial.dtype == final.dtype,
            "initial_state and final_state must have the same dtype, got "
            f"{initial.dtype} and {final.dtype}",
        )
        if not is_exact_alias(initial, final):
            _require(
                not intervals_overlap(
                    storage_interval(initial), storage_interval(final)
                ),
                "initial_state and final_state may only alias exactly; a "
                "partial overlap is rejected",
            )

    for state_name in ("initial_state", "final_state"):
        state = named.get(state_name)
        if state is None:
            continue
        state_range = storage_interval(state)
        for other_name in READ_ONLY_ROLES + ("cu_seqlens",):
            other = named.get(other_name)
            if other is None:
                continue
            _require(
                not intervals_overlap(state_range, storage_interval(other)),
                f"{state_name} must not overlap {other_name}",
            )


def check_flat_output_range(total_tokens: int, heads: int) -> None:
    """Refuse a shape whose flat output does not fit in an INT32 extent.

    Two things need this bound and they disagree by exactly one element, so the
    tighter of the two is what is checked:

    * The tail store writes a partial chunk element-wise through ``(token * H +
      head) * DV + d``, built and consumed as INT32 on the device.  That needs
      the largest *index*, ``T_total * H * DV - 1``, to fit.  Full chunks go
      out through TMA, which addresses the same elements through a descriptor.
    * The CuTe DSL packs a memref descriptor's extents as INT32 when the flat
      view of the output crosses into the compiled entry, so it needs the
      *count* to fit -- one more than the largest index.

    The second is the binding one: at 2**31 elements the DSL raises
    ``OverflowError`` out of ``build_memref_desc``, naming neither the tensor
    nor the shape that caused it.  So the count is the bound, and both failures
    are refused here where the shape is still in hand: the wrapped index would
    write far below the buffer without saying anything, and the DSL error
    arrives at compile time with nothing a caller can act on.
    """
    elements = total_tokens * heads * DV
    if elements > INT32_MAX:
        raise KDAPrefillValidationError(
            f"the flat output would hold {elements} elements, which does not "
            f"fit in a non-negative INT32 (T_total={total_tokens}, H={heads}, "
            f"DV={DV}); the largest T_total at this head count is "
            f"{INT32_MAX // (heads * DV)}"
        )


def validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    out: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    initial_state: Optional[torch.Tensor] = None,
    final_state: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> CanonicalInputs:
    """Validate the backend ABI and describe the canonical launch.

    Everything here is checked before a single byte moves, and none of it reads
    element values, so the result is a pure function of shapes, dtypes,
    devices, addresses and two floats -- which is what makes it cacheable.
    """
    named = {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "out": out,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "cu_seqlens": cu_seqlens,
        "initial_state": initial_state,
        "final_state": final_state,
    }
    _check_device_and_contiguity(named)
    _check_dtypes(named)

    for name in ("q", "k", "v", "g"):
        tensor = named[name]
        _require(tensor.dim() == 4, f"{name} must be rank 4, got {tensor.dim()}")
    _require(
        tuple(q.shape) == tuple(k.shape) == tuple(g.shape),
        "q, k and g must have identical shapes, got "
        f"{tuple(q.shape)}, {tuple(k.shape)}, {tuple(g.shape)}",
    )
    _require(
        tuple(v.shape) == tuple(q.shape),
        f"v must have q's shape, got {tuple(v.shape)} against {tuple(q.shape)}",
    )
    _require(q.shape[3] == DK, f"the key dimension is fixed at {DK}, got {q.shape[3]}")
    _require(
        v.shape[3] == DV, f"the value dimension is fixed at {DV}, got {v.shape[3]}"
    )
    _require(
        tuple(beta.shape) == tuple(q.shape[:3]),
        f"beta must be q without its last dimension, got {tuple(beta.shape)} "
        f"against {tuple(q.shape[:3])}",
    )
    _require(
        tuple(out.shape) == tuple(v.shape) and out.dtype == v.dtype,
        "out must have v's shape and dtype, got "
        f"{tuple(out.shape)}/{out.dtype} against {tuple(v.shape)}/{v.dtype}",
    )

    batch, tokens, heads = q.shape[0], q.shape[1], q.shape[2]
    _require(heads > 0, f"H must be positive, got {heads}")
    _require(
        tuple(A_log.shape) == (heads,),
        f"A_log must be [H] = [{heads}], got {tuple(A_log.shape)}",
    )
    _require(
        tuple(dt_bias.shape) == (heads, DK),
        f"dt_bias must be [H, {DK}] = [{heads}, {DK}], got {tuple(dt_bias.shape)}",
    )

    if cu_seqlens is None:
        input_mode = "fixed"
        _require(batch >= 0, f"fixed mode needs B >= 0, got {batch}")
        _require(tokens >= 0, f"fixed mode needs T >= 0, got {tokens}")
        sequences = batch
        total_tokens = batch * tokens
    else:
        input_mode = "packed"
        _require(
            batch == 1,
            f"packed mode needs the leading dimension to be exactly 1, got {batch}",
        )
        _require(tokens >= 0, f"packed mode needs T_total >= 0, got {tokens}")
        _require(
            cu_seqlens.dtype in (torch.int32, torch.int64),
            f"cu_seqlens must be INT32 or INT64, got {cu_seqlens.dtype}",
        )
        _require(cu_seqlens.dim() == 1, "cu_seqlens must be 1-D")
        _require(
            cu_seqlens.numel() >= 2,
            f"cu_seqlens must have N + 1 >= 2 entries, got {cu_seqlens.numel()}",
        )
        sequences = cu_seqlens.numel() - 1
        total_tokens = tokens

    state_dtype: Optional[torch.dtype] = None
    for name in ("initial_state", "final_state"):
        state = named[name]
        if state is None:
            continue
        _require(
            state.dtype in (torch.bfloat16, torch.float32),
            f"{name} must be BF16 or FP32, got {state.dtype}",
        )
        _require(
            tuple(state.shape) == (sequences, heads, DV, DK),
            f"{name} must be [N, H, {DV}, {DK}] = "
            f"[{sequences}, {heads}, {DV}, {DK}], got {tuple(state.shape)}",
        )
    if initial_state is not None:
        state_dtype = initial_state.dtype
    elif final_state is not None:
        state_dtype = final_state.dtype

    _require(math.isfinite(float(scale)), f"scale must be finite, got {scale}")
    _require(
        math.isfinite(float(lower_bound)),
        f"lower_bound must be finite, got {lower_bound}",
    )
    low, high = LOWER_BOUND_RANGE
    _require(
        low <= float(lower_bound) <= high,
        f"lower_bound must be in [{low}, {high}], got {lower_bound}",
    )

    check_tma_base_alignment(named)
    out_aliases_v = is_exact_alias(out, v)
    if not out_aliases_v:
        _require(
            not intervals_overlap(storage_interval(out), storage_interval(v)),
            "out may alias v only exactly (same base, dtype, shape and stride)",
        )
    _check_aliasing(named, out_aliases_v)

    return CanonicalInputs(
        input_mode=input_mode,
        batch=batch,
        tokens_per_sequence=tokens,
        sequences=sequences,
        heads=heads,
        total_tokens=total_tokens,
        g_fp32=g.dtype is torch.float32,
        state_dtype=state_dtype,
        has_initial_state=initial_state is not None,
        has_final_state=final_state is not None,
        out_aliases_v=out_aliases_v,
    )


# ---------------------------------------------------------------------------
# Canonical INT32 offsets.
#
# The device kernels take exactly one metadata tensor: a device INT32
# ``cu_seqlens`` of length ``N + 1``.  Everything the two public input modes
# differ by is resolved before launch:
#
# * **fixed** ``[B, T, H, 128]`` becomes ``arange(0, (B + 1) * T, T)``;
# * **packed varlen** validates the caller's ``cu_seqlens`` and, for INT64,
#   converts it -- *after* the range check, never before.  Narrowing first and
#   checking the narrowed value cannot detect the overflow it just caused.
#
# Validation reads the offsets on the host, which synchronizes.  That is why
# the result is cached on the source tensor's identity *and* version: a caller
# passing the same tensor every step pays once, and an in-place mutation still
# invalidates.  Inside a capture the read is illegal, so a miss there is an
# error rather than something to work around.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CanonicalOffsets:
    """A validated canonical INT32 ``cu_seqlens`` and what produced it.

    ``source`` and ``canonical`` are both strong references.  Keeping the
    source alive is not redundant: the cache key contains its ``data_ptr``, and
    an allocator that freed it and handed the same address to an unrelated
    tensor would otherwise produce a false hit.
    """

    key: tuple
    source: torch.Tensor
    canonical: torch.Tensor
    sequences: int
    total_tokens: int
    lengths: tuple
    #: Host copy of the offsets themselves, so a variant can derive its own
    #: metadata (chunk counts, task bins) without a second synchronization.
    host: tuple

    @property
    def longest_sequence(self) -> int:
        return max(self.lengths, default=0)


_PACKED_OFFSETS = BoundedDeviceCache("kda-sm120-packed-offsets")
_FIXED_OFFSETS = BoundedDeviceCache("kda-sm120-fixed-offsets")


def _packed_key(cu_seqlens: torch.Tensor, total_tokens: int) -> tuple:
    device = cu_seqlens.device
    return (
        device.type,
        device.index,
        cu_seqlens.data_ptr(),
        cu_seqlens.dtype,
        tuple(cu_seqlens.shape),
        tensor_version(cu_seqlens),
        total_tokens,
    )


def validate_packed_offsets(
    cu_seqlens: torch.Tensor, total_tokens: int
) -> CanonicalOffsets:
    """Validate and canonicalize a packed ``cu_seqlens``."""
    if cu_seqlens.device.type != "cuda":
        raise KDAPrefillValidationError("cu_seqlens must live on a CUDA device")
    if not cu_seqlens.is_contiguous():
        raise KDAPrefillValidationError("cu_seqlens must be contiguous")
    if cu_seqlens.dim() != 1:
        raise KDAPrefillValidationError(
            f"cu_seqlens must be 1-D, got shape {tuple(cu_seqlens.shape)}"
        )
    if cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise KDAPrefillValidationError(
            f"cu_seqlens must be INT32 or INT64, got {cu_seqlens.dtype}"
        )
    if cu_seqlens.numel() < 2:
        raise KDAPrefillValidationError(
            f"cu_seqlens must have N + 1 >= 2 entries, got {cu_seqlens.numel()}"
        )

    key = _packed_key(cu_seqlens, total_tokens)
    hit = _PACKED_OFFSETS.get(cu_seqlens.device, key)
    if hit is not None:
        return hit
    if capturing():
        raise RuntimeError(
            "CUDA graph capture cannot validate cu_seqlens: it needs a "
            "device-to-host copy.  Warm the workspace with one eager call "
            "using the same offsets tensor before capturing"
        )

    # One synchronizing read, on the caller's dtype.  INT64 is checked here and
    # narrowed only afterwards.
    host = cu_seqlens.detach().cpu().tolist()
    if host[0] != 0:
        raise KDAPrefillValidationError(f"cu_seqlens must start at 0, got {host[0]}")
    if host[-1] != total_tokens:
        raise KDAPrefillValidationError(
            f"cu_seqlens must end at T_total={total_tokens}, got {host[-1]}"
        )
    lengths = []
    for i in range(len(host) - 1):
        length = host[i + 1] - host[i]
        if length < 0:
            raise KDAPrefillValidationError(
                f"cu_seqlens must be non-decreasing; entry {i + 1} "
                f"({host[i + 1]}) is below entry {i} ({host[i]})"
            )
        lengths.append(length)
    if not 0 <= host[-1] <= INT32_MAX:
        raise KDAPrefillValidationError(
            f"T_total={host[-1]} does not fit in a non-negative INT32"
        )

    canonical = (
        cu_seqlens if cu_seqlens.dtype is torch.int32 else cu_seqlens.to(torch.int32)
    )
    record = CanonicalOffsets(
        key=key,
        source=cu_seqlens,
        canonical=canonical,
        sequences=len(host) - 1,
        total_tokens=total_tokens,
        lengths=tuple(lengths),
        host=tuple(host),
    )
    _PACKED_OFFSETS.put(cu_seqlens.device, key, record, (canonical,))
    return record


def fixed_offsets(batch: int, tokens: int, device) -> CanonicalOffsets:
    """Canonical offsets for fixed mode: ``arange(0, (B + 1) * T, T)``.

    ``T == 0`` builds an explicit zero tensor of length ``B + 1`` rather than
    calling ``arange`` with a zero step, which raises.
    """
    if batch < 0:
        raise KDAPrefillValidationError(f"fixed mode needs B >= 0, got {batch}")
    if tokens < 0:
        raise KDAPrefillValidationError(f"fixed mode needs T >= 0, got {tokens}")
    total_tokens = batch * tokens
    if total_tokens > INT32_MAX:
        raise KDAPrefillValidationError(
            f"B * T = {total_tokens} does not fit in a non-negative INT32"
        )

    key = (device.type, device.index, batch, tokens)
    hit = _FIXED_OFFSETS.get(device, key)
    if hit is not None:
        return hit
    if capturing():
        raise RuntimeError(
            "CUDA graph capture cannot allocate canonical offsets; warm the "
            "workspace with one eager call at the same (B, T) before capturing"
        )

    if tokens == 0:
        canonical = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    else:
        canonical = torch.arange(
            0, (batch + 1) * tokens, tokens, dtype=torch.int32, device=device
        )
    record = CanonicalOffsets(
        key=key,
        source=canonical,
        canonical=canonical,
        sequences=batch,
        total_tokens=total_tokens,
        lengths=(tokens,) * batch,
        host=tuple(range(0, (batch + 1) * tokens, tokens))
        if tokens
        else (0,) * (batch + 1),
    )
    return _FIXED_OFFSETS.put(device, key, record, (canonical,))


def canonical_offsets(
    cu_seqlens: Optional[torch.Tensor],
    *,
    batch: int,
    tokens: int,
    total_tokens: int,
    device,
) -> CanonicalOffsets:
    """The one entry point both variants use to reach canonical INT32 offsets."""
    if cu_seqlens is None:
        return fixed_offsets(batch, tokens, device)
    return validate_packed_offsets(cu_seqlens, total_tokens)


def clear_offsets_caches() -> None:
    _PACKED_OFFSETS.clear()
    _FIXED_OFFSETS.clear()


# ---------------------------------------------------------------------------
# The workspace resource slot.
#
# ``RecurrentKDAPrefillWorkspace`` is the only public workspace, and it is
# shared with the SM100-family Cake backend.  This is what a workspace holds
# *for* this backend, created lazily on the first eager warmup so a Cake-only
# caller pays one ``None`` field and never imports CuTe DSL.
#
# Once a workspace enters capture, everything about it freezes: the backend,
# the variant, the stream, the tensor addresses, the shapes, the strides, the
# dtypes, the capacity and the packed metadata signature.  A bound workspace is
# not reusable by another backend, another graph, another stream or a Python
# eager call -- a caller needing another signature creates another workspace.
# ---------------------------------------------------------------------------


@dataclass
class SM120PrefillResources:
    """Per-workspace SM120 state, composed into the public workspace.

    Deliberately a plain container: the pieces it holds are built by whichever
    variant is bound, and it exists so the *lifetime* of those pieces is the
    workspace's rather than a process-global cache's.  A CUDA graph replays
    without re-entering Python, so anything it captured has to stay alive at
    its captured address for as long as the graph does; that is what this owns.
    """

    device: torch.device
    #: Stable identity for process-wide plan caches.  Using ``id(self)`` would
    #: let Python reuse a destroyed workspace's address while its cache entry
    #: still exists, and omitting the workspace would let two live workspaces
    #: share captured buffers and descriptors.
    cache_token: object = field(
        default_factory=object, init=False, repr=False, compare=False
    )
    #: Serializes the whole launch sequence.  The decomp variant enqueues two
    #: kernels that share a scratch arena, so a second host thread must not
    #: interleave its own prepare between this call's prepare and recurrence.
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    #: The canonical INT32 offsets buffer replay copies into.  Allocated at
    #: warmup and grown only there.
    cu_seqlens_i32: Optional[torch.Tensor] = None
    #: Decomp's chunk tables, frozen at warmup.  ``None`` for fused.
    cu_chunks_i32: Optional[torch.Tensor] = None
    chunk_to_seq_i32: Optional[torch.Tensor] = None
    #: Decomp's factor arena.
    scratch: Any = None
    #: Stable final-state scratch for ``initial_state=None,
    #: output_final_state=True``, which cannot allocate during capture.
    state_scratch: Optional[torch.Tensor] = None
    #: Descriptor storage and any other object a captured graph reads.
    pins: tuple = ()

    #: What this workspace is bound to.  Set on first use and then immutable.
    variant: Optional[str] = None
    stream_ptr: Optional[int] = None
    signature: Optional[tuple] = None
    captured: bool = False

    def bind(self, *, variant: str, stream_ptr: int, signature: tuple) -> None:
        """Pin this workspace to one variant, stream and call signature."""
        if self.captured:
            raise RuntimeError(
                "this RecurrentKDAPrefillWorkspace has already participated in "
                "a CUDA graph capture and cannot be reused; create another one"
            )
        for name, current, incoming in (
            ("variant", self.variant, variant),
            ("stream", self.stream_ptr, stream_ptr),
            ("call signature", self.signature, signature),
        ):
            if current is not None and current != incoming:
                raise RuntimeError(
                    f"RecurrentKDAPrefillWorkspace is bound to a different "
                    f"{name} ({current!r} against {incoming!r}); create a "
                    f"separate workspace for it"
                )
        self.variant = variant
        self.stream_ptr = stream_ptr
        self.signature = signature

    def pin(self, *objects) -> None:
        """Hold strong references for the lifetime of this workspace."""
        self.pins = self.pins + tuple(o for o in objects if o is not None)

    def ensure_capacity(
        self, name: str, elements: int, dtype: torch.dtype
    ) -> torch.Tensor:
        """A buffer of at least ``elements``, grown only outside capture.

        Monotonic on purpose: shrinking would move an address a captured graph
        already recorded.
        """
        current = getattr(self, name)
        if current is not None and current.numel() >= elements:
            return current[:elements]
        if capturing():
            raise RuntimeError(
                f"CUDA graph capture cannot grow the workspace's {name} buffer; "
                f"warm it eagerly at this size first"
            )
        grown = torch.empty(elements, dtype=dtype, device=self.device)
        setattr(self, name, grown)
        return grown

    # -- the decomposed variant's frozen tables ---------------------------- #
    #
    # Held here rather than in a module-level cache for the reason every other
    # graph resource is: replay reads them at the addresses capture recorded,
    # and an LRU that evicted one would leave a live graph reading freed
    # memory.  The signature is the offsets themselves, so a caller that
    # changes its sequence lengths gets a rebuild -- eagerly, or an error if it
    # tries during capture.

    _chunk_signature: Optional[tuple] = None
    _chunk_tables: Any = None
    _arena_shape: Optional[tuple] = None
    _arena: Any = None

    def chunk_signature_matches(self, signature: tuple):
        """The frozen chunk tables iff they were built for ``signature``."""
        if self._chunk_signature == signature:
            return self._chunk_tables
        return None

    def freeze_chunk_tables(self, signature: tuple, tables: Any) -> None:
        self._chunk_signature = signature
        self._chunk_tables = tables

    def scratch_arena(self, shape: tuple, factory):
        """A variant's scratch, allocated once per shape and held here.

        ``factory`` builds it; this module never learns what it is.  That
        inversion is the point -- the container owns the *lifetime*, which is
        what a captured graph needs, and the variant owns the *contents*, which
        is what neither variant can share with the other.  A ``runtime`` that
        imported a variant to build this would be the reverse dependency the
        package layout exists to prevent.
        """
        if self._arena_shape == shape and self._arena is not None:
            return self._arena
        if capturing():
            raise RuntimeError(
                "CUDA graph capture cannot allocate workspace scratch; warm "
                "the workspace with one eager call at this shape first"
            )
        self._arena = factory()
        self._arena_shape = shape
        return self._arena


def current_stream_ptr(device: Optional[torch.device] = None) -> int:
    """The current CUDA stream's raw handle.

    Only ever compared, never dereferenced.  The raw accessor is used where
    this torch provides it because ``torch.cuda.current_stream()`` constructs
    a ``Stream`` object, which is avoidable overhead on a memoized host path.
    """
    if not torch.cuda.is_available():
        return 0
    if _raw_stream is not None:
        index = torch.cuda.current_device() if device is None else device.index
        return _raw_stream(torch.cuda.current_device() if index is None else index)
    return torch.cuda.current_stream(device).cuda_stream


try:  # pragma: no cover - exercised by whichever branch this torch provides
    from torch._C import _cuda_getCurrentRawStream as _raw_stream
except ImportError:  # pragma: no cover
    _raw_stream = None


#: Stands for "this tensor does not track a version counter", which is the
#: case for every tensor created under ``torch.inference_mode()``.
NO_VERSION = object()


def tensor_version(tensor: torch.Tensor):
    """``tensor._version``, or :data:`NO_VERSION` where it does not exist.

    Reading ``_version`` on a tensor created under ``torch.inference_mode()``
    raises ``RuntimeError: Inference tensors do not track version counter``,
    so the read is guarded.  Without a version an entry cannot be invalidated
    by an in-place write at an unchanged address.  That is harmless for the
    activations, whose contents are expected to change between calls, but
    ``cu_seqlens`` is read on the host: under ``inference_mode`` a caller who
    refills an offsets buffer in place must use a different buffer or call
    ``clear_kda_prefill_sm120_caches()``, as for the SM100-family backend.
    """
    try:
        return tensor._version
    except RuntimeError:
        return NO_VERSION


def tensor_identity(tensor: Optional[torch.Tensor]):
    """What a call-plan key can distinguish about ``tensor``.

    The version is included so an in-place write invalidates the entry where it
    can be observed at all; see :func:`tensor_version` for what happens when it
    cannot.
    """
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tensor.shape,
        tensor.dtype,
        tensor.device,
        tensor.is_contiguous(),
        tensor_version(tensor),
    )


def tensor_layout_identity(tensor: Optional[torch.Tensor]):
    """The address and layout a bound workspace freezes.

    Tensor contents may change between graph replays, so the version counter
    is deliberately excluded.  A different tensor allocation or layout is a
    different capture signature even when its logical shape is unchanged.
    """
    if tensor is None:
        return None
    return (
        tensor.data_ptr(),
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.device,
    )


def resource_cache_token(resources: Optional[SM120PrefillResources]):
    """Stable plan-cache identity for a caller-owned workspace."""
    return None if resources is None else resources.cache_token


def clear_shared_caches() -> None:
    """Drop everything this module holds.  Tests, and callers freeing buffers.

    Deliberately does not touch :data:`GRAPH_PINS`: a live graph makes that
    unsafe, and the caller that knows its graphs are gone can clear it itself.
    """
    clear_offsets_caches()
    clear_flat_views()
    clear_pinned_staging()


# ---------------------------------------------------------------------------
# Per-variant call memo and plan execution.
# ---------------------------------------------------------------------------

#: Marks a call with no tokens (or no chunks), so the zero-work path is reached
#: on a memo hit without re-deriving the metadata that proves it.
STATE_ONLY_PLAN = object()

#: One entry per distinct set of buffers a caller uses; a serving loop that
#: reuses its activations needs exactly one.  A ceiling on how many rotating
#: buffer sets stay fast, not a memory budget: below it the retained memory is
#: the same whatever the value, and above it a hit becomes a plan rebuild.
CALL_PLAN_MAX_ENTRIES = 16


def state_only(initial_state, final_state) -> None:
    """The zero-work path: no launch, only the state ABI is left to honour."""
    if final_state is None:
        return
    if initial_state is None:
        final_state.zero_()
        return
    if is_exact_alias(initial_state, final_state):
        return
    final_state.copy_(initial_state)


def execute(plan, initial_state, final_state) -> None:
    """Run an already-resolved plan.

    Split from the lookup so the facade, which keeps its own memo on the same
    tensor identities, does not repeat the comparison to find the plan again.
    """
    if plan is STATE_ONLY_PLAN:
        state_only(initial_state, final_state)
        return
    plan.run()


class PlanMemo:
    """Two-level memo from a call's tensors to its launch plan, one per variant.

    :meth:`fast_path` recognises the previous call by object identity and
    version without building a key; :meth:`get` and :meth:`remember_plan` are
    the bounded LRU behind it, keyed on :func:`tensor_identity`.  Neither level
    holds a caller's tensors: the fast path keeps weak references and the LRU
    keys on identity tuples.  What an entry does retain is its plan, and a plan
    retains the buffers its descriptors and views address.  ``build_lock``
    serializes the miss path: plan construction, descriptor encoding and the
    compile behind it are not re-entrant, and the two memo levels in front of
    it mean a warm caller never takes it.  The LRU methods take the same lock,
    so a ``clear`` cannot land between a lookup and its ``move_to_end``.
    """

    def __init__(self, max_entries: int = CALL_PLAN_MAX_ENTRIES) -> None:
        self.max_entries = max_entries
        self.plans: "OrderedDict[tuple, Any]" = OrderedDict()
        #: ``(weakrefs, versions, scalars, resource token, stream, plan)`` of
        #: the previous call, or ``None``.
        self.last: Optional[tuple] = None
        self.build_lock = threading.RLock()

    def identity(self, device, tensors, scalars: tuple, resources) -> tuple:
        # The stream of the *inputs'* device: a plan bakes that device's current
        # stream into its argument tuple, so keying on the current device's
        # stream would let two streams on the inputs' device share one entry.
        return (
            tuple(tensor_identity(t) for t in tensors),
            scalars,
            resource_cache_token(resources),
            current_stream_ptr(device),
        )

    def fast_path(self, device, tensors, scalars: tuple, resources):
        """The previous call's plan if this call is identical to it, else ``None``."""
        last = self.last
        if last is None:
            return None
        last_tensors, last_versions, last_scalars, last_resources, last_stream, plan = (
            last
        )
        if last_scalars != scalars:
            return None
        if last_resources is not resource_cache_token(resources):
            return None
        if last_stream != current_stream_ptr(device):
            return None
        if len(last_tensors) != len(tensors):
            return None
        for ref, tensor in zip(last_tensors, tensors, strict=True):
            if ref is None:
                if tensor is not None:
                    return None
            elif ref() is not tensor:
                return None
        for tensor, version in zip(tensors, last_versions, strict=True):
            if tensor is not None and tensor_version(tensor) != version:
                return None
        return plan

    def remember(self, device, tensors, scalars: tuple, resources, plan) -> None:
        self.last = (
            tuple(None if t is None else weakref.ref(t) for t in tensors),
            tuple(None if t is None else tensor_version(t) for t in tensors),
            scalars,
            resource_cache_token(resources),
            current_stream_ptr(device),
            plan,
        )

    def get(self, key):
        with self.build_lock:
            plan = self.plans.get(key)
            if plan is not None:
                self.plans.move_to_end(key)
            return plan

    def remember_plan(self, key, plan) -> None:
        with self.build_lock:
            self.plans[key] = plan
            while len(self.plans) > self.max_entries:
                self.plans.popitem(last=False)

    def clear(self) -> None:
        with self.build_lock:
            self.plans.clear()
            self.last = None


# ---------------------------------------------------------------------------
# TMA descriptors.
# ---------------------------------------------------------------------------

#: Bytes of one ``CUtensorMap``.
DESCRIPTOR_BYTES = 128

_TMA_DTYPES = {
    torch.bfloat16: "CU_TENSOR_MAP_DATA_TYPE_BFLOAT16",
    torch.float32: "CU_TENSOR_MAP_DATA_TYPE_FLOAT32",
}
_TMA_SWIZZLES = {
    "128B": "CU_TENSOR_MAP_SWIZZLE_128B",
    "NONE": "CU_TENSOR_MAP_SWIZZLE_NONE",
}
_TMA_INTERLEAVES = {"NONE": "CU_TENSOR_MAP_INTERLEAVE_NONE"}
_TMA_L2_PROMOTIONS = {"128B": "CU_TENSOR_MAP_L2_PROMOTION_L2_128B"}
_TMA_OOB_FILLS = {"NONE": "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE"}


@dataclass(frozen=True)
class TensorMapSpec:
    """One descriptor's complete ``cuTensorMapEncodeTiled`` configuration.

    Two roles that produce equal specs are the same descriptor, so both
    variants deduplicate on equality.  Only the fields below vary between
    roles; the encoder's remaining arguments are fixed by the contract and are
    carried so the cache key and the equality test cover them.
    """

    dtype: torch.dtype
    base: int
    global_dims: tuple[int, ...]
    global_stride_bytes: tuple[int, ...]
    box_dims: tuple[int, ...]
    element_strides: tuple[int, ...] = (1, 1, 1)
    interleave: str = "NONE"
    swizzle: str = "128B"
    l2_promotion: str = "128B"
    oob_fill: str = "NONE"

    @property
    def rank(self) -> int:
        return len(self.global_dims)

    @property
    def element_size(self) -> int:
        return self.dtype.itemsize

    @property
    def box_bytes(self) -> int:
        n = 1
        for b in self.box_dims:
            n *= b
        return n * self.element_size

    def key(self, device) -> tuple:
        """Full cache key: device plus every encoder field."""
        index = device.index if isinstance(device, torch.device) else device
        return (
            index,
            self.base,
            self.dtype,
            self.rank,
            self.global_dims,
            self.global_stride_bytes,
            self.box_dims,
            self.element_strides,
            self.interleave,
            self.swizzle,
            self.l2_promotion,
            self.oob_fill,
        )

    def validate(self) -> None:
        """Check what the driver will not.

        The driver rejects some of this itself, but with an error that names a
        parameter index rather than a role, and it accepts a misaligned global
        base outright; the corruption from that shows up as wrong numbers in
        one head, far from here.
        """
        if self.rank != 3:
            raise ValueError(f"TMA rank must be 3, got {self.rank}")
        if self.base % GLOBAL_BASE_ALIGN:
            raise ValueError(
                f"TMA global base must be {GLOBAL_BASE_ALIGN}-byte aligned, "
                f"got {self.base:#x}"
            )
        if any(d <= 0 for d in self.global_dims):
            raise ValueError(
                f"TMA global dims must be positive, got {self.global_dims}"
            )
        if len(self.global_stride_bytes) != self.rank - 1:
            raise ValueError(
                "TMA global strides cover dimensions 1..rank-1 only, got "
                f"{len(self.global_stride_bytes)} for rank {self.rank}"
            )
        for s in self.global_stride_bytes:
            if s <= 0 or s % GLOBAL_BASE_ALIGN:
                raise ValueError(
                    f"TMA global strides must be positive and "
                    f"{GLOBAL_BASE_ALIGN}-byte aligned, got {self.global_stride_bytes}"
                )
        if len(self.box_dims) != self.rank:
            raise ValueError(f"TMA box must have rank {self.rank}")
        if any(not 1 <= b <= 256 for b in self.box_dims):
            raise ValueError(
                f"TMA box extents must be in [1, 256], got {self.box_dims}"
            )
        if self.dtype not in _TMA_DTYPES:
            raise ValueError(f"unsupported TMA element type {self.dtype}")
        if self.swizzle not in _TMA_SWIZZLES:
            raise ValueError(
                f"swizzle must be one of {tuple(_TMA_SWIZZLES)}, got {self.swizzle!r}"
            )
        inner_bytes = self.box_dims[0] * self.element_size
        if self.swizzle == "128B" and inner_bytes != 128:
            raise ValueError(
                f"a 128B-swizzled inner box must be exactly 128 bytes, got {inner_bytes}"
            )
        if self.swizzle == "NONE" and (inner_bytes % 16 or inner_bytes > 512):
            raise ValueError(
                f"an unswizzled inner box must be a multiple of 16 bytes and at most "
                f"512, got {inner_bytes}"
            )

    def encode(self) -> bytes:
        """Encode this spec as its :data:`DESCRIPTOR_BYTES` raw bytes."""
        import cuda.bindings.driver as drv

        self.validate()
        err, tmap = drv.cuTensorMapEncodeTiled(
            getattr(drv.CUtensorMapDataType, _TMA_DTYPES[self.dtype]),
            self.rank,
            self.base,
            [drv.cuuint64_t(d) for d in self.global_dims],
            [drv.cuuint64_t(s) for s in self.global_stride_bytes],
            [drv.cuuint32_t(b) for b in self.box_dims],
            [drv.cuuint32_t(e) for e in self.element_strides],
            getattr(drv.CUtensorMapInterleave, _TMA_INTERLEAVES[self.interleave]),
            getattr(drv.CUtensorMapSwizzle, _TMA_SWIZZLES[self.swizzle]),
            getattr(drv.CUtensorMapL2promotion, _TMA_L2_PROMOTIONS[self.l2_promotion]),
            getattr(drv.CUtensorMapFloatOOBfill, _TMA_OOB_FILLS[self.oob_fill]),
        )
        if int(err) != 0:
            raise RuntimeError(f"cuTensorMapEncodeTiled failed: {err}")
        # cuda-python wraps the descriptor, so take its address via getPtr().
        return bytes(ctypes.string_at(tmap.getPtr(), DESCRIPTOR_BYTES))


def descriptor_cache_key(specs: dict, device, roles) -> tuple:
    """Cache key of a role -> spec map: every encoder field of every role."""
    return tuple((role, specs[role].key(device)) for role in roles if role in specs)


__all__ = [
    "assert_tvm_ffi_dispatched",
    "BoundedDeviceCache",
    "build_kernel",
    "canonical_offsets",
    "capturing",
    "check_flat_output_range",
    "clear_offsets_caches",
    "clear_pinned_staging",
    "clear_shared_caches",
    "current_stream_ptr",
    "DESCRIPTOR_BYTES",
    "descriptor_cache_key",
    "DK",
    "DV",
    "execute",
    "flat_view",
    "GLOBAL_BASE_ALIGN",
    "GRAPH_PINS",
    "IdentityCache",
    "INT32_MAX",
    "intervals_overlap",
    "is_exact_alias",
    "KDAPrefillValidationError",
    "LOG2_E",
    "max_grid_dims",
    "NO_VERSION",
    "NORM_FLOOR",
    "PlanMemo",
    "PREFIX_FLOOR",
    "record_stream_once",
    "require_sm120a",
    "resource_cache_token",
    "SM120_CODE_TARGET",
    "sm120a_available",
    "sm120a_compile_options",
    "SM120PrefillResources",
    "STATE_ONLY_PLAN",
    "storage_interval",
    "tensor_identity",
    "tensor_layout_identity",
    "tensor_version",
    "TensorMapSpec",
    "UnsupportedArchitectureError",
    "upload_bytes",
    "validate_inputs",
    "validate_packed_offsets",
]
