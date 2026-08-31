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

"""The one thing ``decomp`` and ``fused`` share.

Both variants implement the same operation and neither can see the other, so
anything they both consume has to live somewhere neither owns.  That is this
module, and the bar for entering it is deliberately high: a helper belongs here
only when both variants call it, it means the same thing to both, its lifetime
is the same for both, and a unit test can pin it without a device kernel.

What that admits:

* the canonical launch description and the shape/dtype/alias validation that
  produces it, plus the error type both raise;
* the exact ``sm_120a`` target check, expressed through
  :mod:`flashinfer.cute_dsl.utils` and never by writing ``CUTE_DSL_ARCH``;
* the bounded per-device cache, the capture probe, tensor identity, pinned
  descriptor staging and the cache statistics -- the *containers*, not the
  cache instances, which stay with the variant that keys them;
* canonical INT32 ``cu_seqlens``, the workspace resource slot, the graph
  stream/signature binding and the resource lifetime that goes with it;
* the naming convention for :func:`~flashinfer.jit.build_and_load_cute_dsl_kernel`.

What it excludes, and why the exclusion is not cosmetic: every layout, swizzle,
TMA descriptor, PTX helper and device kernel.  The two variants have helpers
with matching names -- both have a ``raw_bf16_s128``, both have a pairwise
image -- and the images are not the same, because the shapes they index are
not the same.  Hoisting one and letting the other call it would be a silent
numerical change, so neither is hoisted.

Nothing here imports ``decomp`` or ``fused``, and importing this module loads
no device code and compiles nothing.
"""

from __future__ import annotations

import importlib.util
import math
import os
import threading
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
    """


class UnsupportedArchitectureError(RuntimeError):
    """Raised when the SM120 backend is asked to run on another target."""


# ---------------------------------------------------------------------------
# Architecture and compile target.
#
# Two separate questions, and conflating them is how a report ends up naming a
# target the artifact does not have:
#
#   1. is the *device* compute capability 12.0?
#   2. can the installed CuTe DSL and CUDA toolkit compile and load ``sm_120a``
#      natively -- not a family-conditional ``sm_120f`` fallback?
#
# Both must hold.  Neither is answered by mutating ``CUTE_DSL_ARCH``: the DSL
# captures its default target when ``cutlass`` is first imported, so a write
# here would not retarget an already-imported DSL, and a write before import
# would change every other CuTe-DSL kernel in the process.
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
        from ...cute_dsl.utils import is_cute_dsl_arch_supported
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
    argument-marshalling ABI, and the slow one costs about 4x of the host path.

    **These must be passed with ``cute.compile[options](...)``, not
    ``cute.compile(..., options=options)``.** The keyword form accepts the
    tuple and silently ignores ``EnableTVMFFI``: it yields a
    ``CudaDialectJitCompiledFunction``, which marshals every argument through
    ``ctypes.addressof``, where the subscript form yields a
    ``TVMFFIJitCompiledFunctionWithKwargs``. Measured on this backend, that
    mistake cost 82 us of host time per call against 17.5 us -- invisible
    against a 1 ms kernel and 4x the entire call at B=1 T=16 H=4.
    :func:`assert_tvm_ffi_dispatched` exists so it cannot happen quietly again.
    """
    import cutlass.cute as cute

    options: tuple = (cute.GPUArch(SM120_CODE_TARGET),)
    if enable_tvm_ffi:
        options = (cute.EnableTVMFFI(True),) + options
    return options


def assert_tvm_ffi_dispatched(compiled, kernel_name: str):
    """Refuse a compiled entry that fell back to the ctypes argument path.

    The fallback is not an error the DSL reports -- it produces a working
    callable that is simply several times slower to invoke, which is the kind
    of regression that gets attributed to the kernel months later. Checking the
    type is cheap and happens once per specialization.
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
# a cross-implementation A/B cannot reuse another implementation's cached
# artifact and misreport reuse as agreement.
# ---------------------------------------------------------------------------

#: Logical compile namespace.  Bumping the suffix invalidates every artifact.
JIT_MODULE_NAME = "flashinfer-kda-prefill-sm120-v1"


def _assert_cache_target_matches() -> None:
    """The persistent cache's arch must be the one we asked ``cute`` for.

    ``JitSpecCuteDsl`` derives its module directory and its ``meta.json`` arch
    from ``CUTE_DSL_ARCH`` or the current device's capability.  We pass an
    explicit ``GPUArch(sm_120a)``, so if those two ever disagree the artifact
    on disk is named for one target and built for another -- the exact failure
    the plan forbids.  Checking is cheap; guessing is not recoverable.
    """
    from ...jit.cute_dsl_core import _get_compile_arch

    resolved = _get_compile_arch()
    expected = SM120_CODE_TARGET.replace("_", "")
    if resolved != expected:
        raise UnsupportedArchitectureError(
            f"the CuTe-DSL persistent cache resolves its target to "
            f"{resolved!r} while this backend compiles for {expected!r}; "
            f"refusing to write an artifact whose name does not describe it "
            f"(unset CUTE_DSL_ARCH to let it follow the device)"
        )


def _module_key_files() -> tuple:
    """Every source file whose content should invalidate this module.

    All four package files, for both variants, and deliberately the *same*
    tuple for every kernel in the namespace.  ``JitSpecCuteDsl`` writes one
    ``meta.json`` per module directory and wipes the directory whenever a
    kernel arrives with a different source hash -- so passing each variant only
    its own file made ``decomp`` and ``fused`` invalidate each other on every
    alternation, which a benchmark sees as a recompile per switch and a
    correctness A/B sees as a cold build every time.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    return tuple(
        os.path.join(here, name)
        for name in ("runtime.py", "decomp.py", "fused.py", "__init__.py")
    )


def build_kernel(kernel_name: str, compile_fn, *, device, key_files=()):
    """Compile through FlashInfer's persistent CuTe-DSL cache.

    ``kernel_name`` is the specialization -- variant, dtype presence flags and
    every other compile-time parameter, and nothing that varies per call.  A
    tensor address or a runtime shape in this string is a cache that never
    hits.  ``key_files`` is accepted and ignored: invalidation is a property of
    the module namespace, not of one kernel in it, so it always uses
    :func:`_module_key_files`.

    Falls back to a plain in-process ``compile_fn()`` when the persistent path
    raises.  That is a real possibility rather than defensive coding: whether
    ``export_to_c`` can round-trip a given compiled object is a property of the
    pinned CuTe DSL, not of this code.  When it cannot, the kernel still
    compiles and still runs -- only the cross-process cache is lost, and
    :func:`persistent_cache_status` says so rather than leaving a warning in a
    log for someone to notice.
    """
    # JitSpecCuteDsl derives its cache arch, loads the object module and runs a
    # cold compile against the current CUDA context.  The public API accepts a
    # tensor on a non-current device, so all three operations must happen under
    # the input tensor's device guard rather than whichever device the caller
    # happened to leave current.
    with torch.cuda.device(device):
        _assert_cache_target_matches()
        from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel

        del key_files
        try:
            return build_and_load_cute_dsl_kernel(
                JIT_MODULE_NAME,
                kernel_name,
                compile_fn,
                extra_key_files=_module_key_files(),
            )
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            _record_persistent_cache_failure(kernel_name, exc)
            return compile_fn()


def persistent_cache_status() -> dict:
    """Whether a compiled kernel can actually survive this process.

    Reported rather than assumed, because on some pinned CuTe DSL releases it
    cannot.  ``JitSpecCuteDsl`` exports with
    ``export_to_c(path, function_name=...)`` and reloads the result as a
    TVM-FFI callable; a DSL whose ``export_to_c`` takes
    ``(file_path, file_name, function_prefix)`` and emits a plain C entry
    satisfies neither half.  FlashInfer degrades gracefully when that happens
    -- it keeps the in-process kernel and logs a warning -- so nothing breaks,
    but every process pays a cold compile and no amount of reading the cache
    directory reveals why.

    ``supported`` is False exactly when the export signature does not match.
    The check is static: it inspects the signature and compiles nothing.
    """
    status = {
        "module": JIT_MODULE_NAME,
        "supported": False,
        "reason": "",
        "failures": dict(_PERSISTENT_CACHE_FAILURES),
    }
    try:
        import inspect

        from cutlass.base_dsl.dsl import JitCompiledFunction

        parameters = inspect.signature(JitCompiledFunction.export_to_c).parameters
        if "function_name" in parameters:
            status["supported"] = True
        else:
            status["reason"] = (
                "the installed CuTe DSL exports with "
                f"export_to_c{inspect.signature(JitCompiledFunction.export_to_c)}, "
                "which FlashInfer's persistent CuTe-DSL cache does not call; "
                "kernels compile in-process every run"
            )
    except Exception as exc:  # noqa: BLE001 -- reporting only
        status["reason"] = f"could not inspect the DSL export API: {exc}"
    return status


_PERSISTENT_CACHE_FAILURES: "OrderedDict[str, str]" = OrderedDict()


def _record_persistent_cache_failure(kernel_name: str, exc: BaseException) -> None:
    _PERSISTENT_CACHE_FAILURES[kernel_name] = f"{type(exc).__name__}: {exc}"
    while len(_PERSISTENT_CACHE_FAILURES) > 32:
        _PERSISTENT_CACHE_FAILURES.popitem(last=False)


def persistent_cache_unavailable_reason(kernel_name: str) -> Optional[str]:
    """Why ``kernel_name`` fell back to an in-process compile, if it did."""
    return _PERSISTENT_CACHE_FAILURES.get(kernel_name)


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

    Both cross-stream primitives below are illegal there: ``cudaStreamWaitEvent``
    on an event recorded outside the capture, and ``record_stream``, which is a
    caching-allocator operation with no graph representation.  Either
    invalidates the capture, and the failure surfaces later as "operation
    failed due to a previous error during capture" rather than at the call.

    Skipping them is correct, not a workaround.  A capture replays only work it
    recorded, so an entry it reads was already live and reachable when the
    capture began: there is no other stream to order against and no lifetime
    for the allocator to extend.
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
        import weakref

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
# Every launch converts its tensors with ``from_dlpack(t.reshape(-1))``, and at
# ~5 us each that is tens of microseconds per launch on tensors whose addresses
# have not moved.  The conversion is a pure function of (pointer, element
# count, dtype, alignment), so keying on exactly those four is sound. With
# ``enable_tvm_ffi=True``, the CuTe view owns a TVM-FFI DLPack consumer object;
# that object keeps the reshaped tensor's storage alive until the view is
# evicted. The allocator therefore cannot recycle a live entry's address, and
# the LRU below bounds that retention.
# ---------------------------------------------------------------------------

#: Bounded so a workload cycling through many buffers cannot grow it without
#: limit.  A forward touches ~25 tensors, so this holds several shapes' worth.
#:
#: One of three caches that can hold a buffer alive -- the others are the plan
#: memo and :data:`MAX_ENTRIES` -- and any one of them is enough.  Measured:
#: lowering this alone changes the retention by nothing at all, because the
#: plan memo still has the tensors; the three only bind together.
FLAT_VIEW_MAX_ENTRIES = 256

_FLAT_VIEWS: "OrderedDict[tuple, Any]" = OrderedDict()
_FLAT_STATS = {"hits": 0, "misses": 0}

#: Held on the miss path only.  ``flat_view`` runs once per tensor per launch --
#: five times before a kernel that can be nine microseconds long -- so a lock on
#: the hit path is measurable where one on the miss path is not.  The hit path
#: is two C-level dict operations, each atomic under the GIL: the worst a race
#: can do there is evict an entry that was just touched, which costs one rebuild
#: and no correctness.  ``_FLAT_STATS`` is advisory for the same reason; its
#: increments are not atomic and are not read by anything that decides.
_FLAT_VIEWS_LOCK = threading.RLock()


def _require_tvm_ffi() -> None:
    """``apache-tvm-ffi`` is not optional for this backend.

    The persistent cache reloads artifacts with
    ``load_module(..., enable_tvm_ffi=True)``, and the compiled entry rejects
    a view built without it (``'_Tensor' object has no attribute
    '_tvm_ffi_tensor'``).  Falling back to the ctypes argument path would cost
    ~3.3x of the host path silently, which is the kind of regression that gets
    attributed to the kernel months later.
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
    hit = _FLAT_VIEWS.get(key)
    if hit is not None:
        _FLAT_VIEWS.move_to_end(key)
        _FLAT_STATS["hits"] += 1
        return hit

    _FLAT_STATS["misses"] += 1
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


def flat_view_stats() -> dict:
    return dict(_FLAT_STATS, entries=len(_FLAT_VIEWS))


def clear_flat_views() -> None:
    with _FLAT_VIEWS_LOCK:
        _FLAT_VIEWS.clear()
        _FLAT_STATS.update(hits=0, misses=0)


# ---------------------------------------------------------------------------
# Capture-safe descriptor upload.
#
# A descriptor build copies from pageable host memory, which is a 0.0 us memcpy
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
#: whatever else is on the stream, which in a busy loop is milliseconds -- so
#: refilling the buffer for the next descriptor build would overwrite bytes the
#: DMA has not read yet, and the descriptor that reached the device would be a
#: mix of two.  Two builds of one size is the common case rather than a corner:
#: the sizes are a function of the descriptor count, so any two cold calls of
#: the same shape collide.  Waiting here costs nothing a steady state pays,
#: because a build is a cache miss only.
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


# Moved here from the decomposed variant: both variants have to check their
# grid against the device, so the helper that asks the driver belongs with
# the other shared device queries rather than inside one of them.
_GRID_LIMITS: dict[int, tuple[int, int]] = {}


def max_grid_dims(device: torch.device) -> tuple[int, int]:
    """``(maxGridSize[0], maxGridSize[1])`` for ``device``.

    ``torch.cuda.get_device_properties`` does not expose the grid limits, so
    this goes to the driver, and caches: plan Section 9.2 runs the check on
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

    The second is the binding one, and it was measured rather than assumed: at
    exactly 2**31 elements the DSL raises ``OverflowError: Value overflow:
    2147483648 exceeds range of l`` out of ``build_memref_desc``, which names
    neither the tensor nor the shape that caused it.  So the count is the
    bound, and both failures are refused here where the shape is still in hand:
    the wrapped index would write far below the buffer without saying anything,
    and the DSL error arrives at compile time with nothing a caller can act on.
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


def offsets_cache_stats(device) -> dict:
    return {
        "packed": _PACKED_OFFSETS.stats(device),
        "fixed": _FIXED_OFFSETS.stats(device),
    }


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

    Only ever compared, never dereferenced.  ``torch.cuda.current_stream()``
    builds a Stream object through five Python frames, which is measurable once
    the rest of the host path is memoized, so the raw accessor is used where
    this torch provides it.
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

    Reading ``_version`` on an inference tensor raises ``RuntimeError:
    Inference tensors do not track version counter``, and ``inference_mode`` is
    how serving actually calls this backend -- so a cache that reads it
    unguarded works in a benchmark and fails in production.

    The consequence is worth stating rather than hiding. The version guards
    against an in-place edit at an unchanged address; without it, a cached
    entry for such a tensor cannot be invalidated by content. For the
    activations that does not matter: a call plan is a function of addresses,
    shapes and dtypes, and their contents are *expected* to change between
    calls. It matters for ``cu_seqlens``, whose values are read on the host --
    so under ``inference_mode`` those values are a caller contract, exactly as
    they already are for the SM100-family backend. A caller who refills an
    offsets buffer in place must either use a different buffer or call
    ``clear_kda_prefill_sm120_caches()``.
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


__all__ = [
    "DK",
    "DV",
    "GLOBAL_BASE_ALIGN",
    "GRAPH_PINS",
    "INT32_MAX",
    "JIT_MODULE_NAME",
    "LOG2_E",
    "LOWER_BOUND_RANGE",
    "MAX_ENTRIES",
    "READ_ONLY_ROLES",
    "SM120_CAPABILITY",
    "SM120_CODE_TARGET",
    "BoundedDeviceCache",
    "CacheStats",
    "CanonicalInputs",
    "CanonicalOffsets",
    "GraphResourcePins",
    "IdentityCache",
    "KDAPrefillValidationError",
    "SM120PrefillResources",
    "UnsupportedArchitectureError",
    "build_kernel",
    "canonical_offsets",
    "capturing",
    "check_tma_base_alignment",
    "clear_flat_views",
    "clear_offsets_caches",
    "clear_pinned_staging",
    "clear_shared_caches",
    "check_flat_output_range",
    "current_stream_ptr",
    "fixed_offsets",
    "flat_view",
    "flat_view_stats",
    "intervals_overlap",
    "is_exact_alias",
    "offsets_cache_stats",
    "persistent_cache_status",
    "persistent_cache_unavailable_reason",
    "record_stream_once",
    "resource_cache_token",
    "require_sm120a",
    "sm120a_available",
    "assert_tvm_ffi_dispatched",
    "sm120a_compile_options",
    "storage_interval",
    "NO_VERSION",
    "max_grid_dims",
    "tensor_identity",
    "tensor_layout_identity",
    "tensor_version",
    "upload_bytes",
    "validate_inputs",
    "validate_packed_offsets",
]
