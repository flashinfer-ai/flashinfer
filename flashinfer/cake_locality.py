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

Locality-domain runtime of the Cake kernels on a dual-die GPU (sm_107a).

A Cake locality-domain kernel is one launch over both dies of the GPU. It reads
the die of its SM from the SM table (:func:`sm_table`), takes the operands that
each die reads most from memory homed on that die (:func:`split_empty`), and
claims its work through one small counter workspace per die (:func:`counters`).
The SM table and the homing only order the claims: such a kernel is bitwise
equal to the whole-GPU build of the same source for any cut and any SM table.
It is equal within the family's tolerance to the whole-GPU programs FlashInfer
ships for the same problem, which are an older build, so a call that falls back
to the whole-GPU route (no SM table, an unhomed operand, an unlisted cell) may
change the low bits of the result.

:func:`sm_table` probes the SMs of a device outside CUDA graph capture through a
FlashInfer JIT build of :data:`PROBE_SOURCE`, and caches only the outcomes that a
later probe would repeat. Every condition it cannot meet gives ``None``, and a
caller then takes its whole-GPU route.

:func:`split_empty` reserves one virtual address range and backs ``[0, cut)``
with memory of die 0 and ``[cut, size)`` with memory of die 1. The allocation is
in a registry keyed by its base pointer from allocation until its last tensor is
freed; :func:`homing` reads it. Freeing it unmaps the memory, so a CUDA graph
that still reads a freed allocation faults on its next replay: keep the tensor
alive as long as any graph that reads it.

Imports: the standard library at module level; ``torch`` and the JIT module
inside the functions that need them.
"""

from __future__ import annotations

import dataclasses
import fractions
import functools
import threading
from collections.abc import Sequence
from typing import Any

__all__ = [
    "ENTRY_BITS",
    "ENTRY_COUNT_SHIFT",
    "ENTRY_DOMAIN_BITS",
    "ENTRY_DOMAIN_SHIFT",
    "ENTRY_RANK_SHIFT",
    "ENTRY_SUPPLEMENTAL_SHIFT",
    "EVEN_GAP",
    "LOCALITY_ARCH",
    "LOCALITY_CUDA",
    "PROBE_ATTEMPTS",
    "PROBE_SOURCE",
    "SM_TABLE_LENGTH",
    "Homing",
    "SmTable",
    "counters",
    "die_class",
    "granularity",
    "homing",
    "pack_rows",
    "pointer_domain",
    "sm_table",
    "split_empty",
    "split_units",
]

# The SM table a twin takes by value: one int32 word per SM id.
SM_TABLE_LENGTH = 256
# One word: the SM's rank among its die's SMs (by SM id) and the die's SM count,
# ENTRY_BITS wide each, the die ENTRY_DOMAIN_BITS wide, the supplemental flag.
ENTRY_RANK_SHIFT = 0
ENTRY_COUNT_SHIFT = 10
ENTRY_DOMAIN_SHIFT = 20
ENTRY_SUPPLEMENTAL_SHIFT = 24
ENTRY_BITS = 10
ENTRY_DOMAIN_BITS = 4
# The architecture whose twins route, and the CUDA release (toolkit and driver)
# whose PTX reads the locality registers.
LOCALITY_ARCH = "sm_107a"
LOCALITY_CUDA = (13, 5)
# A probe that missed an SM (another context held it) is taken again, at most
# this many times in all.
PROBE_ATTEMPTS = 4
# Two dies whose SM counts differ by at most this many SMs are "even".
EVEN_GAP = 4

# One 1-thread CTA per SM (each asks for an SM's whole opt-in shared memory, so no
# two share an SM) stores its SM's id, locality domain, supplemental flag and the
# SM-id bound at its CTA's index.
PROBE_SOURCE = r"""
#include <cstdint>

#include <cuda_runtime.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/error.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <tvm/ffi/function.h>

__global__ void __launch_bounds__(1, 1) cake_locality_probe_kernel(uint4* out) {
  unsigned smid, domain, supplemental, nsmid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  asm volatile("mov.u32 %0, %%locality_domain_id;" : "=r"(domain));
  asm volatile("mov.u32 %0, %%locality_domain_supplemental_sm;" : "=r"(supplemental));
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsmid));
  out[blockIdx.x] = make_uint4(smid, domain, supplemental, nsmid);
}

void cake_locality_probe(tvm::ffi::TensorView out, int64_t ctas, int64_t smem) {
  TVM_FFI_ICHECK(out.ndim() == 2 && out.size(0) >= ctas && out.size(1) == 4);
  DLDevice device = out.device();
  TVM_FFI_ICHECK(cudaSetDevice(device.device_id) == cudaSuccess);
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));
  TVM_FFI_ICHECK(cudaFuncSetAttribute(cake_locality_probe_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      static_cast<int>(smem)) == cudaSuccess);
  uint4* rows = reinterpret_cast<uint4*>(static_cast<char*>(out.data_ptr()) +
                                         out.byte_offset());
  cake_locality_probe_kernel<<<static_cast<unsigned>(ctas), 1,
                               static_cast<size_t>(smem), stream>>>(rows);
  TVM_FFI_ICHECK(cudaGetLastError() == cudaSuccess);
}

int64_t cake_driver_version() {
  int version = 0;
  TVM_FFI_ICHECK(cudaDriverGetVersion(&version) == cudaSuccess);
  return version;
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(probe, cake_locality_probe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(driver_version, cake_driver_version);
"""


@dataclasses.dataclass(frozen=True)
class SmTable:
    """The probed SM table of one device: ``counts`` are the SMs of die 0 and die 1;
    ``entries`` is the CPU int32 tensor of :data:`SM_TABLE_LENGTH` words a twin takes
    as its ``sm_table`` (contiguous; never written)."""

    device_index: int
    counts: tuple[int, int]
    entries: Any


def die_class(counts: Sequence[int]) -> str:
    """``"even"`` when the two dies' SM counts differ by at most :data:`EVEN_GAP`,
    else ``"uneven"``."""
    return "even" if abs(int(counts[0]) - int(counts[1])) <= EVEN_GAP else "uneven"


def split_units(units: int, counts: Sequence[int]) -> tuple[int, int]:
    """``units`` whole units shared by the two dies in proportion to their SM counts:
    each die gets the floor of its exact share, and the units left over go one each to
    the larger remainders; a remainder tie goes to the die with more SMs, then to die
    0. 7 over (100, 112) is (3, 4), over (106, 106) (4, 3); 1 over (110, 102) is
    (1, 0), over (102, 110) (0, 1)."""
    weights = [int(count) for count in counts]
    if int(units) < 0 or len(weights) != 2 or min(weights) < 0 or sum(weights) <= 0:
        raise ValueError(f"{units} units are not shared by the SM counts {counts}")
    total = sum(weights)
    shares = [fractions.Fraction(int(units) * weight, total) for weight in weights]
    parts = [share.numerator // share.denominator for share in shares]
    order = sorted(range(2), key=lambda i: (parts[i] - shares[i], -weights[i], i))
    for index in order[: int(units) - sum(parts)]:
        parts[index] += 1
    return (parts[0], parts[1])


def pack_rows(
    rows: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], list[int]] | None:
    """``(counts, words)`` of the probe rows ``(smid, domain, supplemental, nsmid)``,
    one row per CTA: ``words`` holds :data:`SM_TABLE_LENGTH` entries, 0 at an SM id no
    row names. ``None`` when an SM id is not below :data:`SM_TABLE_LENGTH`, an
    ``nsmid`` exceeds it, two rows name one SM id, the domains are not exactly 0 and 1,
    or a die's count does not fit :data:`ENTRY_BITS`."""
    by_sm: dict[int, tuple[int, bool]] = {}
    for smid, domain, supplemental, nsmid in rows:
        if not 0 <= int(smid) < SM_TABLE_LENGTH or int(nsmid) > SM_TABLE_LENGTH:
            return None
        if int(smid) in by_sm:
            return None
        by_sm[int(smid)] = (int(domain), bool(int(supplemental)))
    if {domain for domain, _ in by_sm.values()} != {0, 1}:
        return None
    counts = [0, 0]
    ranks: dict[int, int] = {}
    for smid in sorted(by_sm):
        domain = by_sm[smid][0]
        ranks[smid] = counts[domain]
        counts[domain] += 1
    if max(counts) >= 1 << ENTRY_BITS:
        return None
    words = [0] * SM_TABLE_LENGTH
    for smid, (domain, supplemental) in by_sm.items():
        words[smid] = (
            ranks[smid] << ENTRY_RANK_SHIFT
            | counts[domain] << ENTRY_COUNT_SHIFT
            | domain << ENTRY_DOMAIN_SHIFT
            | int(supplemental) << ENTRY_SUPPLEMENTAL_SHIFT
        )
    return (counts[0], counts[1]), words


@dataclasses.dataclass(frozen=True)
class Homing:
    """One live :func:`split_empty` allocation of device ``device_index``: ``nbytes``
    bytes at ``base`` in a reserved range of ``size`` bytes (``nbytes`` rounded up to
    the granularity); bytes ``[0, cut)`` from ``base`` are on die 0, the rest on
    die 1."""

    device_index: int
    base: int
    nbytes: int
    size: int
    cut: int


# --------------------------------------------------------------------------- module

_LOCK = threading.Lock()
_TABLES: dict[int, SmTable | None] = {}
_PROBE_MODULE: list[Any] = []  # the loaded module, or None for an old toolkit
_PROBE_LOCKS: dict[int, threading.Lock] = {}  # one per device: one probe at a time


def _device_index(device: Any) -> int:
    import torch

    if device is None:
        return int(torch.cuda.current_device())
    if isinstance(device, int):
        return device
    resolved = torch.device(device)
    if resolved.type != "cuda":
        raise ValueError(f"a CUDA device is required, got {resolved}")
    return int(
        torch.cuda.current_device() if resolved.index is None else resolved.index
    )


def _is_locality_arch(device_index: int) -> bool:
    import torch

    major, minor = torch.cuda.get_device_capability(device_index)
    return f"sm_{major}{minor}a" == LOCALITY_ARCH


@functools.cache
def _module() -> Any:
    """The loaded locality module (the probe and the split allocator), built once
    through FlashInfer's JIT; raises ``LocalityDomainError`` when the toolkit
    predates :data:`LOCALITY_CUDA` or the build fails."""
    from .jit.cpp_ext import get_cuda_version
    from .jit.cake_locality import gen_cake_locality_module
    from .locality_domain import LocalityDomainError

    version = get_cuda_version()
    if (int(version.major), int(version.minor)) < LOCALITY_CUDA:
        raise LocalityDomainError(
            "Cake locality domains need a CUDA "
            f"{LOCALITY_CUDA[0]}.{LOCALITY_CUDA[1]} toolkit, found CUDA {version}"
        )
    try:
        return gen_cake_locality_module().build_and_load()
    except Exception as exc:
        raise LocalityDomainError(
            f"failed to build the Cake locality module: {exc}"
        ) from exc


def _toolkit_supported() -> bool:
    from .jit.cpp_ext import get_cuda_version

    version = get_cuda_version()
    return (int(version.major), int(version.minor)) >= LOCALITY_CUDA


def _probe_module() -> tuple[Any, bool]:
    """``(module, cacheable)``: the loaded locality module, or ``None`` when it
    cannot load; a toolkit older than :data:`LOCALITY_CUDA` is remembered, a build
    or load failure is not (the next call tries again)."""
    with _LOCK:
        if _PROBE_MODULE:
            return _PROBE_MODULE[0], True
    try:
        if not _toolkit_supported():
            with _LOCK:
                if not _PROBE_MODULE:
                    _PROBE_MODULE.append(None)
                return _PROBE_MODULE[0], True
        return _module(), True
    except Exception:  # noqa: BLE001 - no probe library: this call stays whole
        return None, False


# --------------------------------------------------------------------------- probe


def sm_table(device: Any = None) -> SmTable | None:
    """The SM table of ``device`` (default: the current CUDA device), probed once per
    process and device, or ``None``; a caller routes nothing on ``None``.

    ======================================================  =================
    condition                                               result (cached)
    ======================================================  =================
    the device is not :data:`LOCALITY_ARCH`                 None (yes)
    toolkit before :data:`LOCALITY_CUDA`                    None (yes)
    driver before :data:`LOCALITY_CUDA`                     None (yes)
    the dies are not exactly 0 and 1, or a row is invalid  None (yes)
    the call is under CUDA graph capture, or torch cannot
    tell whether it is                                      None (no)
    the probe library does not build or load, the probe
    raises, or it misses an SM :data:`PROBE_ATTEMPTS`
    times (another context held it)                         None (no)
    ======================================================  =================

    An outcome that is not cached is probed again by the next call. One probe of a
    device runs at a time. The probe synchronizes the device and runs on a private
    stream; call this eagerly before a graph capture, so a captured call and an
    eager call see the same table."""
    import torch

    try:
        index = _device_index(device)
    except Exception:  # noqa: BLE001 - no CUDA device to probe
        return None
    with _LOCK:
        if index in _TABLES:
            return _TABLES[index]
        lock = _PROBE_LOCKS.setdefault(index, threading.Lock())
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:  # noqa: BLE001 - no CUDA runtime to ask: nothing to probe yet
        return None
    if capturing:
        return None
    with lock:
        with _LOCK:
            if index in _TABLES:
                return _TABLES[index]
        if not _is_locality_arch(index):
            return _remember(index, None, True)
        compiler = getattr(torch, "compiler", None)
        disable = getattr(compiler, "disable", None)
        probe = disable(_probe) if disable is not None else _probe
        table, cacheable = probe(index)
        return _remember(index, table, cacheable)


def _remember(index: int, table: SmTable | None, cacheable: bool) -> SmTable | None:
    if not cacheable:
        return table
    with _LOCK:
        return _TABLES.setdefault(index, table)


def _probe(device_index: int) -> tuple[SmTable | None, bool]:
    """Run the probe on device ``device_index``: ``(table, cacheable)`` (see
    :func:`sm_table`)."""
    import torch

    module, cacheable = _probe_module()
    if module is None:
        return None, cacheable
    try:
        driver = int(module.driver_version())
        if (driver // 1000, driver % 1000 // 10) < LOCALITY_CUDA:
            return None, True
        import tvm_ffi

        properties = torch.cuda.get_device_properties(device_index)
        ctas = int(properties.multi_processor_count)
        smem = int(properties.shared_memory_per_block_optin)
        out = torch.zeros((ctas, 4), dtype=torch.int32, device=f"cuda:{device_index}")
        torch.cuda.synchronize(device_index)
        stream = torch.cuda.Stream(device=device_index)
        rows: list[list[int]] = []
        for _ in range(PROBE_ATTEMPTS):
            with torch.cuda.stream(stream), tvm_ffi.use_torch_stream():
                module.probe(out, ctas, smem)
            stream.synchronize()
            rows = out.cpu().tolist()
            if len({row[0] for row in rows}) == ctas:
                break
        else:
            return None, False
    except Exception:  # noqa: BLE001 - a probe that cannot run routes nothing now
        return None, False
    packed = pack_rows(rows)
    if packed is None:
        return None, True
    counts, words = packed
    entries = torch.tensor(words, dtype=torch.int32)
    return SmTable(device_index=device_index, counts=counts, entries=entries), True


def _force_table(counts: Sequence[int], device_index: int = 0) -> SmTable:
    """Test only: cache a table of two dies holding ``counts`` SMs (SM ids
    ``0 .. counts[0] - 1`` on die 0, the rest on die 1) for ``device_index``, as the
    probe would."""
    import torch

    n0, n1 = int(counts[0]), int(counts[1])
    rows = [(sm, 0 if sm < n0 else 1, 0, n0 + n1) for sm in range(n0 + n1)]
    packed = pack_rows(rows)
    if packed is None:
        raise ValueError(f"SM counts {tuple(counts)} make no SM table")
    words = torch.tensor(packed[1], dtype=torch.int32)
    table = SmTable(device_index=int(device_index), counts=packed[0], entries=words)
    with _LOCK:
        _TABLES[int(device_index)] = table
    return table


def _forget() -> None:
    """Test only: drop every cached table and the probe module state."""
    with _LOCK:
        _TABLES.clear()
        _PROBE_MODULE.clear()


# --------------------------------------------------------------------------- memory


def _require_locality_device(device_index: int) -> None:
    from .locality_domain import LocalityDomainError

    if not _is_locality_arch(device_index):
        raise LocalityDomainError(
            f"Cake locality domains need {LOCALITY_ARCH}; device {device_index} is not"
        )


def granularity(device: Any = None) -> int:
    """The allocation granularity in bytes of locality-domain memory on ``device``;
    a :func:`split_empty` cut is a multiple of it."""
    index = _device_index(device)
    _require_locality_device(index)
    return int(_module().cake_locality_granularity(index))


def split_empty(
    shape: Sequence[int],
    dtype: Any,
    cut_bytes: int,
    device: Any = None,
) -> Any:
    """An uninitialized contiguous tensor of ``shape`` and ``dtype`` on ``device``
    in one virtual address range whose bytes ``[0, cut_bytes)`` are memory of die 0
    and the rest memory of die 1.

    ``cut_bytes`` is ``0`` (all on die 1), the tensor's byte size (all on die 0), or
    a multiple of :func:`granularity` between them. The memory is freed when the
    tensor and all its views are freed: the device is synchronized, both parts are
    unmapped and released, and the allocation leaves the :func:`homing` registry.
    Allocate and free outside CUDA graph capture; a graph that still reads the
    tensor after it is freed faults on replay."""
    import math

    import torch

    index = _device_index(device)
    _require_locality_device(index)
    shape = tuple(int(extent) for extent in shape)
    if any(extent < 0 for extent in shape):
        raise ValueError(f"invalid shape {shape}")
    itemsize = torch.empty((), dtype=dtype).element_size()
    nbytes = math.prod(shape) * itemsize
    if nbytes == 0:
        raise ValueError(f"a split allocation needs at least one byte, got {shape}")
    cut = int(cut_bytes)
    if not 0 <= cut <= nbytes:
        raise ValueError(f"cut {cut} is outside the allocation of {nbytes} bytes")
    unit = granularity(index)
    if cut != nbytes and cut % unit != 0:
        raise ValueError(
            f"cut {cut} is not a multiple of the allocation granularity {unit}"
        )
    raw = _module().cake_locality_split_empty(nbytes, cut, index)
    if not isinstance(raw, torch.Tensor):
        raw = torch.from_dlpack(raw)
    return raw.view(dtype).view(shape)


def homing(tensor_or_ptr: Any) -> Homing | None:
    """The :class:`Homing` of the live :func:`split_empty` allocation based at
    ``tensor_or_ptr`` (a tensor: its storage's base pointer; an int: that address),
    or ``None``."""
    import torch

    if isinstance(tensor_or_ptr, torch.Tensor):
        if tensor_or_ptr.device.type != "cuda":
            return None
        base = int(tensor_or_ptr.untyped_storage().data_ptr())
    else:
        base = int(tensor_or_ptr)
    if _module.cache_info().currsize == 0:
        return None  # the module never loaded, so nothing was split-allocated
    record = [int(value) for value in _module().cake_locality_split_info(base)]
    if not record:
        return None
    owner, nbytes, size, cut = record
    return Homing(device_index=owner, base=base, nbytes=nbytes, size=size, cut=cut)


def pointer_domain(ptr: int, device: Any = None) -> int:
    """The locality domain whose memory backs device address ``ptr``."""
    index = _device_index(device)
    _require_locality_device(index)
    return int(_module().cake_locality_pointer_domain(int(ptr), index))


def counters(words: int, device: Any = None) -> tuple[Any, Any]:
    """Each die's claim counters of one launch: two zeroed int32 tensors of ``words``
    words, homed on die 0 and die 1 through FlashInfer's locality-domain memory
    pools. When those pools are unavailable (another PyTorch allocator backend, no
    ``MemPool`` support, or no locality domains), the pair is two ordinary zeroed
    tensors: the claims are not homed, and the results are the same bits. A kernel
    leaves them zero after every launch; two launches in flight never share one
    pair. Call outside CUDA graph capture."""
    import torch

    from . import locality_domain

    index = _device_index(device)
    target = torch.device("cuda", index)
    shape = (int(words),)
    try:
        pools = locality_domain._domain_pools(index).pools
        use_mem_pool = locality_domain._use_mem_pool()
        pair = []
        for die in range(2):
            with use_mem_pool(pools[die], device=target):
                pair.append(torch.empty(shape, dtype=torch.int32, device=target))
    except (locality_domain.LocalityDomainError, torch.OutOfMemoryError):
        pair = [torch.empty(shape, dtype=torch.int32, device=target) for _ in range(2)]
    for tensor in pair:
        tensor.zero_()
    return pair[0], pair[1]
