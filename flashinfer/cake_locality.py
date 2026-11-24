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
claims its work through one pair of small per-die counter workspaces: a prepared
launch owns its pair (:func:`counters`), and a call that launches at once reuses
the pair of its stream, or under CUDA graph capture takes a pre-zeroed pair of
its own (:func:`launch_counters`).
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

Twins gain mostly from die-local memory reads, not from their schedule. Timed
as CUPTI kernel time of CUDA graph replays after a read flush of twice the L2
size: at 4752 MHz, the batch DeepGEMM FP8 twin (B=64 weights of (4096, 7168))
on ordinary interleaved memory took 0.95-0.97x the time of the whole-GPU
program on the full grid at expected_m 1-16 and 1.05x at 36 and 64, and homed
memory cut its time 1.34-1.41x and 1.09x; the GDN decode twin (H=16, HV=32,
B 48-512) on interleaved memory took 1.03-1.06x the whole-GPU time and homed
memory cut it 1.08-1.19x. With every die's SMs labelled with the other die
(an inverted SM table) both twins took 1.8-2.6x their homed time at 4752 MHz
and 1.6-1.8x at 3186 MHz, with the same bits. Some boards keep their memory
at a lower clock under load (driver 620.43: 3186 of 4752 MHz). There the batch DeepGEMM FP8 twin took the same time on
homed as on interleaved memory at expected_m 1-16 (0.99-1.00x), and homed
weights were slower than plain weights at expected_m 1-32 (1.04-1.07x) and
equal at 36-64; at 4752 MHz homed weights were faster up to expected_m 64
(1.03-1.47x) and slower from 72 to 512 (see
``flashinfer.gemm.cake_batch_deepgemm_fp8.localize_weights``). Neither the CUDA
driver API version (cuDriverGetVersion) nor the CUDA memory-clock attribute tells the two apart,
while NVML's current memory clock does: :func:`memory_at_full_clock` reads it
once per device, and the batch DeepGEMM FP8 path homes and routes no twin
below :data:`MEMORY_CLOCK_FRACTION` of the maximum, or when NVML cannot read
it. The GDN decode twin (H=16, HV=32, B 48-512) stays faster than its
whole-GPU route at 3186 MHz (0.93-0.96x its time; CUPTI kernel time of CUDA
graph replays after a read flush of twice the L2 size, so the 128 MiB-1 GiB
state is read from memory), so :func:`sm_table` does not depend on the memory
clock.

:func:`split_empty` reserves one virtual address range and backs ``[0, cut)``
with memory of die 0 and ``[cut, size)`` with memory of die 1. The allocation is
in a registry keyed by its base pointer from allocation until its last tensor is
freed; :func:`homing` reads it. The storage of the tensor :func:`split_empty`
returns also carries its :class:`Homing` record, which :func:`tensor_homing` reads
without a registry query for every tensor over that storage that starts at the
allocation's base (the tensor itself, an ``nn.Parameter`` over it, ``detach()``,
``.data``, a view); a copy has its own storage and no record. Freeing it unmaps the memory, so a CUDA graph
that still reads a freed allocation faults on its next replay: keep the tensor
alive as long as any graph that reads it.

Imports: the standard library at module level; ``torch`` and the JIT module
inside the functions that need them.
"""

from __future__ import annotations

import collections
import contextlib
import dataclasses
import fractions
import functools
import itertools
import logging
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
    "CAPTURE_PAIR_LIMIT",
    "CAPTURE_RESERVE",
    "EVEN_GAP",
    "LOCALITY_ARCH",
    "LOCALITY_CUDA",
    "MEMORY_CLOCK_FRACTION",
    "PROBE_ATTEMPTS",
    "PROBE_SOURCE",
    "SM_TABLE_LENGTH",
    "STREAM_PAIR_LIMIT",
    "Homing",
    "SmTable",
    "counters",
    "die_class",
    "granularity",
    "homing",
    "launch_counters",
    "memory_at_full_clock",
    "pack_rows",
    "pointer_domain",
    "reserve_launch_counters",
    "sm_table",
    "split_empty",
    "split_units",
    "tensor_homing",
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
# memory_at_full_clock: NVML reads the memory clock at no less than this fraction
# of the maximum memory clock (see the module docstring).
MEMORY_CLOCK_FRACTION = 0.95

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
# The (current, max) memory clock in MHz NVML read for a device by
# memory_at_full_clock, or None when it could not read them.
_MEMORY_CLOCKS: dict[int, tuple[int, int] | None] = {}
# Devices whose unreadable memory clock was logged: one warning per device and
# process (_forget does not clear it).
_CLOCK_WARNED: set[int] = set()
_LOGGER = logging.getLogger(__name__)


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
    no such CUDA device, the call is under CUDA graph
    capture, or torch cannot tell whether it is             None (no)
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
    if index in _TABLES:  # a cached outcome needs no lock
        return _TABLES.get(index)
    with _LOCK:
        if index in _TABLES:
            return _TABLES[index]
        lock = _PROBE_LOCKS.setdefault(index, threading.Lock())
    try:
        capturing = _stream_capturing(index)
    except Exception:  # noqa: BLE001 - no CUDA runtime to ask: nothing to probe yet
        return None
    if capturing:
        return None
    with lock:
        with _LOCK:
            if index in _TABLES:
                return _TABLES[index]
        try:
            locality_arch = _is_locality_arch(index)
        except Exception:  # noqa: BLE001 - not a device index: nothing to probe
            return None
        if not locality_arch:
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


def memory_at_full_clock(device: Any = None) -> bool:
    """Whether NVML reads the memory clock of ``device`` (default: the current
    CUDA device) at no less than :data:`MEMORY_CLOCK_FRACTION` of its maximum;
    ``False`` below it or when NVML cannot read it. The clock is read once per
    process and device, through NVML by the device's UUID, and the outcome is
    cached: a cached outcome costs a dictionary lookup. The batch DeepGEMM FP8
    path routes no twin where this is ``False`` (see the module docstring);
    :func:`sm_table` does not depend on it."""
    try:
        index = _device_index(device)
    except Exception:  # noqa: BLE001 - no CUDA device to read
        return False
    if index not in _MEMORY_CLOCKS:
        read = _memory_clocks(index)
        with _LOCK:
            _MEMORY_CLOCKS.setdefault(index, read)
    clocks = _MEMORY_CLOCKS.get(index)
    if clocks is None:
        return False
    current, peak = clocks
    return peak > 0 and current >= MEMORY_CLOCK_FRACTION * peak


def _nvml_uuid(device_index: int) -> str:
    """The NVML UUID (``GPU-...``) of CUDA device ``device_index``."""
    import torch

    uuid = str(torch.cuda.get_device_properties(device_index).uuid)
    return uuid if uuid.startswith(("GPU-", "MIG-")) else f"GPU-{uuid}"


def _memory_clocks(device_index: int) -> tuple[int, int] | None:
    """The current and maximum memory clock in MHz of CUDA device
    ``device_index`` read through NVML (the device is found by its UUID, so a
    ``CUDA_VISIBLE_DEVICES`` order does not matter), or ``None`` when NVML cannot
    read them. A ``None`` is logged once per device and process, since the
    batch DeepGEMM FP8 path then routes no twin on the device."""
    try:
        import pynvml

        uuid = _nvml_uuid(device_index)
        pynvml.nvmlInit()
        try:
            handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            current = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            peak = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        finally:
            pynvml.nvmlShutdown()
        return int(current), int(peak)
    except Exception as exc:  # noqa: BLE001 - an unreadable clock routes nothing
        _warn_unreadable_clock(device_index, exc)
        return None


def _warn_unreadable_clock(device_index: int, exc: Exception) -> None:
    """Log, once per device and process, why the memory clock of CUDA device
    ``device_index`` could not be read."""
    with _LOCK:
        if device_index in _CLOCK_WARNED:
            return
        _CLOCK_WARNED.add(device_index)
    reason = (
        "pynvml is not installed"
        if isinstance(exc, ImportError)
        else f"NVML could not read it ({type(exc).__name__}: {exc})"
    )
    _LOGGER.warning(
        "the memory clock of CUDA device %d is unknown (%s), so batch DeepGEMM "
        "FP8 calls on it route no locality-domain twin and take the whole-GPU "
        "route",
        device_index,
        reason,
    )


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
    # Explicitly on the CPU: a twin binding takes the table by value from host
    # memory, whatever torch.set_default_device or a torch.device context says.
    entries = torch.tensor(words, dtype=torch.int32, device="cpu")
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
    words = torch.tensor(packed[1], dtype=torch.int32, device="cpu")
    table = SmTable(device_index=int(device_index), counts=packed[0], entries=words)
    with _LOCK:
        _TABLES[int(device_index)] = table
    return table


def _forget() -> None:
    """Test only: drop every cached table and the probe module state."""
    with _LOCK:
        _TABLES.clear()
        _PROBE_MODULE.clear()
        _MEMORY_CLOCKS.clear()


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
    tensor and all its views are freed: the context that made the allocation and,
    when it is on the allocation's device, the context that frees it are
    synchronized, both parts are unmapped and released, and the allocation leaves
    the :func:`homing` registry. Work of any other context that reads or writes
    the tensor must be complete before then (only contexts of the allocation's
    device can access it).
    Allocate and free outside CUDA graph capture; a graph that still reads the
    tensor after it is freed faults on replay.

    The returned tensor's storage carries its :class:`Homing` record (see
    :func:`tensor_homing`), so every alias of the tensor sees it; a copy does
    not."""
    import math

    import torch

    index = _device_index(device)
    _require_locality_device(index)
    shape = tuple(int(extent) for extent in shape)
    if any(extent < 0 for extent in shape):
        raise ValueError(f"invalid shape {shape}")
    itemsize = torch.empty((), dtype=dtype, device="cpu").element_size()
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
    tensor = raw.view(dtype).view(shape)
    record = homing(tensor)
    if record is not None:
        _mark_homed(tensor, record)
    return tensor


# The attribute that holds the Homing record of a split_empty allocation, set on
# the storage of the returned tensor only: every alias of it shares that storage
# (PyTorch keeps a storage's Python object, and so its attributes, alive as long
# as the storage), and a tensor whose ``.data`` or storage is replaced later sees
# the record of its new storage, not a stale one of its own.
_HOMING_ATTR = "_flashinfer_cake_homing"
# Whether this process ever marked an allocation: until then no tensor is homed.
_MARKED: list[bool] = []


def _mark_homed(tensor: Any, record: Homing) -> None:
    """Attach ``record`` to the storage of ``tensor``."""
    setattr(tensor.untyped_storage(), _HOMING_ATTR, record)
    if not _MARKED:
        _MARKED.append(True)


def tensor_homing(tensor: Any) -> Homing | None:
    """The :class:`Homing` of the :func:`split_empty` allocation whose storage
    ``tensor`` uses, when ``tensor`` starts at the allocation's base, else
    ``None``. Every alias over that storage sees the record: the returned tensor,
    an ``nn.Parameter`` over it, its ``detach()``, its ``.data`` and its views
    (a view of part of it starting at the base too: callers that need the whole
    allocation also compare the byte size with ``Homing.nbytes``). A copy has
    its own storage and gives ``None``. The record is read from the storage the
    tensor uses now: after ``w.data = other`` (or ``set_``) ``w`` is homed
    exactly when ``other``'s storage is, even when ``other`` reuses the address
    of an allocation ``w`` used before. Reads an attribute, not the registry,
    so it costs no call into the locality module."""
    if not _MARKED:
        return None
    record = getattr(tensor.untyped_storage(), _HOMING_ATTR, None)
    if record is None or int(tensor.data_ptr()) != record.base:
        return None
    return record


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


# Devices whose locality-domain pools refused (a LocalityDomainError, which a
# later call would repeat): their counters are ordinary tensors from then on. A
# pool that ran out of memory (see _pool_out_of_memory), also at PyTorch's own
# memory limit, is not a refusal: a later call tries the pools again.
_POOLS_UNAVAILABLE: set[int] = set()


def _pool_out_of_memory(exc: BaseException) -> bool:
    """Whether a locality-domain pool error is exhaustion, which a later call may
    not meet: a ``torch.OutOfMemoryError``, or a ``LocalityDomainError`` from the
    pools' first allocation whose allocator error is ``CUDA_ERROR_OUT_OF_MEMORY``
    or that carries no allocator error at all (PyTorch refused the request, for
    example at its per-process memory limit, before the pool's allocator ran).
    That first allocation reports every allocator failure as an out-of-memory
    cause, so its other driver errors (``CUDA_ERROR_NOT_SUPPORTED``,
    ``CUDA_ERROR_INVALID_VALUE``, ...) are refusals."""
    import torch

    if isinstance(exc, torch.OutOfMemoryError):
        return True
    if not isinstance(exc.__cause__, torch.OutOfMemoryError):
        return False
    text = str(exc)
    return "CUDA_ERROR_OUT_OF_MEMORY" in text or text.endswith(
        "allocation smoke test failed"
    )


def _allocate_pair(words: int, device_index: int) -> tuple[tuple[Any, Any], bool]:
    """``(pair, lasting)``: two uninitialized int32 tensors of ``words`` words on the
    current stream, homed on die 0 and die 1 when FlashInfer's locality-domain pools
    exist. ``lasting`` is False only for an unhomed pair made because a pool was
    out of memory, which a later call may not be."""
    import torch

    from . import locality_domain

    target = torch.device("cuda", device_index)
    shape = (int(words),)
    if device_index not in _POOLS_UNAVAILABLE:
        try:
            pools = locality_domain._domain_pools(device_index).pools
            use_mem_pool = locality_domain._use_mem_pool()
            pair = []
            for die in range(2):
                with use_mem_pool(pools[die], device=target):
                    pair.append(torch.empty(shape, dtype=torch.int32, device=target))
            return (pair[0], pair[1]), True
        except (locality_domain.LocalityDomainError, torch.OutOfMemoryError) as exc:
            if _pool_out_of_memory(exc):
                # Exhaustion, also when the pools' first allocation wrapped it in
                # a LocalityDomainError: not remembered, a later call retries.
                unhomed = tuple(
                    torch.empty(shape, dtype=torch.int32, device=target)
                    for _ in range(2)
                )
                return (unhomed[0], unhomed[1]), False
            with _LOCK:
                _POOLS_UNAVAILABLE.add(device_index)
    plain = tuple(
        torch.empty(shape, dtype=torch.int32, device=target) for _ in range(2)
    )
    return (plain[0], plain[1]), True


def counters(
    words: int, device: Any = None, *, settle: bool = False
) -> tuple[Any, Any]:
    """Each die's claim counters of one launch: two zeroed int32 tensors of ``words``
    words, homed on die 0 and die 1 through FlashInfer's locality-domain memory
    pools. When those pools are unavailable (another PyTorch allocator backend, no
    ``MemPool`` support, or no locality domains), the pair is two ordinary zeroed
    tensors: the claims are not homed, and the results are the same bits. A kernel
    leaves them zero after every launch; two launches in flight never share one
    pair. Call outside CUDA graph capture.

    The pair is allocated and zeroed on the current stream of ``device``. With
    ``settle=True`` it is allocated and zeroed on a private stream of the device
    instead, and the call waits for that zeroing (not for the caller's stream), so
    the pair may be used at once on any stream. The pair is marked used on the
    caller's current stream (``Tensor.record_stream``): freed, its memory is not
    reused before the work enqueued on that stream by then is done. A launch on
    another stream must mark that stream too (a prepared launch does), or keep
    the pair alive until its launches are done."""
    index = _device_index(device)
    if settle:
        import torch

        pair, _ = _settled_pair(int(words), index)
        # Allocated on the private stream: freeing the pair must also wait for
        # the work the caller enqueues on its stream, or the next allocation on
        # the private stream could take and zero it under a running launch.
        stream = torch.cuda.current_stream(index)
        for tensor in pair:
            tensor.record_stream(stream)
        return pair
    pair, _ = _allocate_pair(int(words), index)
    for tensor in pair:
        tensor.zero_()
    return pair


def _capture_id(device_index: int) -> int | None:
    """CUDA's id of the capture the current stream of ``device_index`` is in, 0
    when it is not capturing, ``None`` when it cannot be read (no CUDA Python
    runtime bindings: neither ``cuda.bindings.runtime`` nor ``cuda.cudart``)."""
    import torch

    try:
        from cuda.bindings import runtime as cudart
    except ImportError:
        try:
            from cuda import cudart  # cuda-python before cuda.bindings
        except ImportError:
            return None
    try:
        handle = int(torch._C._cuda_getCurrentRawStream(device_index))
        info = cudart.cudaStreamGetCaptureInfo(handle)
    except Exception:  # noqa: BLE001 - the bindings cannot read it: unknown
        return None
    if info[0] != cudart.cudaError_t.cudaSuccess:
        return None
    if info[1] != cudart.cudaStreamCaptureStatus.cudaStreamCaptureStatusActive:
        return 0
    return int(info[2])


# ----------------------------------------------------------------- launch counters

# Eager launches: one cached pair per (stream handle, words) of a device, at most
# this many pairs per device; past it the least recently used pair is dropped.
STREAM_PAIR_LIMIT = 64
# Captured launches: pre-zeroed pairs made outside capture. A refill keeps at least
# CAPTURE_RESERVE pairs ready, and twice the most pairs one capture has taken,
# and makes at most CAPTURE_PAIR_LIMIT pairs per (device, words) in the process's
# lifetime.
CAPTURE_RESERVE = 64
CAPTURE_PAIR_LIMIT = 1 << 16
# Pairs taken are counted per capture id for this many of the most recently
# active captures of one (device, words); an older capture's count is dropped
# (CUDA capture ids only grow, so a retired id does not come back).
_CAPTURE_IDS_TRACKED = 64
# Pairs cut from one reserve allocation start 128 bytes apart: no two pairs share
# an L2 line.
_PAIR_STRIDE_BYTES = 128
# Raw stream handles that do not name one stream for every thread: the default
# stream (0), cudaStreamLegacy (1) and cudaStreamPerThread (2). Their pairs are
# cached per thread.
_THREAD_KEYED_HANDLES = frozenset({0, 1, 2})


@dataclasses.dataclass
class _CaptureReserve:
    """The pre-zeroed pairs of one (device, words): ``free`` pairs are not handed out
    yet; ``issued`` were handed to captures; ``made`` counts every pair cut;
    ``peak`` is the most pairs one capture took; ``taken`` counts the pairs each
    capture id took so far, for the :data:`_CAPTURE_IDS_TRACKED` most recently
    active captures (least recent first), so overlapping captures keep separate
    counts; ``blocks`` holds every allocation the pairs were cut from, for the
    life of the process (a graph may read a handed-out pair at any later
    replay)."""

    free: list[tuple[Any, Any]] = dataclasses.field(default_factory=list)
    issued: int = 0
    made: int = 0
    peak: int = 0
    taken: collections.OrderedDict[int, int] = dataclasses.field(
        default_factory=collections.OrderedDict
    )
    refilling: bool = False
    blocks: list[tuple[Any, Any]] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class _DeviceCounters:
    """The launch counters of one device, written under ``lock``: the eager pairs
    by ``(stream handle, words)`` (read without the lock) with their last-use
    stamps, the capture reserves by ``words``, and the private stream that zeroes
    the reserves."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    stream_pairs: dict[tuple[int, int], tuple[Any, Any]] = dataclasses.field(
        default_factory=dict
    )
    reserves: dict[int, _CaptureReserve] = dataclasses.field(default_factory=dict)
    side_stream: Any = None
    # The last use of each stream_pairs key, a stamp of ``clock``: written by
    # cache hits without the lock; the least recently used pair is evicted.
    used: dict[tuple[int, int], int] = dataclasses.field(default_factory=dict)
    clock: Any = dataclasses.field(default_factory=itertools.count)


_COUNTER_STATES: dict[int, _DeviceCounters] = {}
_THREAD_PAIRS = threading.local()


def _counter_state(device_index: int) -> _DeviceCounters:
    state = _COUNTER_STATES.get(device_index)
    if state is None:
        with _LOCK:
            state = _COUNTER_STATES.setdefault(device_index, _DeviceCounters())
    return state


def _thread_pairs() -> dict[tuple[int, int, int], tuple[Any, Any]]:
    pairs = getattr(_THREAD_PAIRS, "pairs", None)
    if pairs is None:
        pairs = _THREAD_PAIRS.pairs = {}
    return pairs


def _stream_capturing(device_index: int) -> bool:
    """Whether the current stream of device ``device_index`` (not of the current
    device) is capturing a CUDA graph."""
    import torch

    if torch.cuda.device_count() <= 1 or device_index == torch.cuda.current_device():
        return bool(torch.cuda.is_current_stream_capturing())
    with torch.cuda.device(device_index):
        return bool(torch.cuda.is_current_stream_capturing())


def _settled_pair(words: int, device_index: int) -> tuple[tuple[Any, Any], bool]:
    """``(pair, lasting)`` as :func:`_allocate_pair`, allocated and zeroed on the
    device's private stream; returns once that zeroing is done. Outside capture;
    holds no lock while it waits."""
    import torch

    state = _counter_state(device_index)
    with state.lock:
        if state.side_stream is None:
            state.side_stream = torch.cuda.Stream(device=device_index)
        side = state.side_stream
    with torch.cuda.stream(side):
        pair, lasting = _allocate_pair(words, device_index)
        for tensor in pair:
            tensor.zero_()
    side.synchronize()
    return pair, lasting


def _refill_capture_reserve(device_index: int, words: int) -> None:
    """Top up the capture reserve of ``(device_index, words)`` outside capture,
    once captures took about half of it, to ``max(CAPTURE_RESERVE, 2 x peak)``
    pairs (``peak``: the most pairs one capture took), within
    :data:`CAPTURE_PAIR_LIMIT`. The pairs are zeroed on the device's private stream
    and published after that zeroing is done; the caller's stream is not waited
    for, and no lock is held while waiting. An allocation that could not be homed
    for lack of memory is not published (a later refill tries again)."""
    state = _counter_state(device_index)
    reserve = state.reserves.get(words)
    if reserve is not None and (
        reserve.refilling
        or 2 * len(reserve.free) > max(CAPTURE_RESERVE, 2 * reserve.peak)
    ):
        # Full enough (read without the lock; the check below under the lock
        # decides): the steady state of an eager call takes no lock here.
        return
    with state.lock:
        reserve = state.reserves.setdefault(words, _CaptureReserve())
        target = max(CAPTURE_RESERVE, 2 * reserve.peak)
        if reserve.refilling or 2 * len(reserve.free) > target:
            return
        count = min(target - len(reserve.free), CAPTURE_PAIR_LIMIT - reserve.made)
        if count <= 0:
            return
        reserve.refilling = True
    try:
        stride = -(-words * 4 // _PAIR_STRIDE_BYTES) * _PAIR_STRIDE_BYTES // 4
        block, lasting = _settled_pair(count * stride, device_index)
        with state.lock:
            if lasting:
                reserve.blocks.append(block)
                reserve.free.extend(
                    (
                        block[0][i * stride : i * stride + words],
                        block[1][i * stride : i * stride + words],
                    )
                    for i in reversed(range(count))
                )
                reserve.made += count
    finally:
        with state.lock:
            reserve.refilling = False


def reserve_launch_counters(words: int, device: Any = None) -> None:
    """Fill the capture reserve of :func:`launch_counters` for ``words`` on
    ``device`` now (see there), so that captures made before any eager call take
    pre-zeroed pairs. Call outside CUDA graph capture; a call under capture does
    nothing."""
    index = _device_index(device)
    if _stream_capturing(index):
        return
    _refill_capture_reserve(index, int(words))


def _captured_pair(device_index: int, words: int) -> tuple[Any, Any]:
    import torch

    state = _counter_state(device_index)
    # An unreadable capture id counts every such capture as one capture: the
    # reserve then grows with all of them (within CAPTURE_PAIR_LIMIT).
    capture = _capture_id(device_index)
    if capture is None:
        capture = -1
    with state.lock:
        reserve = state.reserves.setdefault(words, _CaptureReserve())
        taken = reserve.taken.pop(capture, 0) + 1
        reserve.taken[capture] = taken  # most recently active last
        while len(reserve.taken) > _CAPTURE_IDS_TRACKED:
            reserve.taken.popitem(last=False)
        reserve.peak = max(reserve.peak, taken)
        if reserve.free:
            reserve.issued += 1
            return reserve.free.pop()
    # No reserved pair: zero one inside the capture (memset nodes in the graph).
    target = torch.device("cuda", device_index)
    return (
        torch.zeros((words,), dtype=torch.int32, device=target),
        torch.zeros((words,), dtype=torch.int32, device=target),
    )


def launch_counters(words: int, device: Any = None) -> tuple[Any, Any]:
    """Each die's claim counters (two zeroed int32 tensors of ``words`` words, homed
    as by :func:`counters`) for one launch enqueued right away on the current
    stream of ``device``, by a kernel that leaves them zero. Allocates nothing in
    steady state.

    - Outside capture, every launch on one stream reuses that stream's pair,
      which is safe because launches on one stream are ordered. The pair is
      allocated and zeroed on that stream at its first use, and cached by the raw
      ``cudaStream_t`` handle: PyTorch hands out stream handles from a per-device
      pool, and a recycled handle names the same CUDA stream, so launches with
      one handle are ordered whatever stream object made them. A stream wrapped
      by ``torch.cuda.ExternalStream`` must therefore not be destroyed while
      launches on it are in flight (a new stream could get its handle and its
      pair), the same rule as for any FlashInfer workspace. The handles 0
      (default stream), 1 (``cudaStreamLegacy``) and 2 (``cudaStreamPerThread``)
      can name a different stream in each thread, so their pairs are cached per
      thread. At most :data:`STREAM_PAIR_LIMIT` pairs are cached per device; the
      least recently used one is dropped past it (its memory goes back to the
      caching allocator, which reuses it only in its stream's order). A pair that
      could not be homed for lack of memory is used once and not cached.
    - Under CUDA graph capture, the pair is a pre-zeroed pair of a reserve made
      outside capture, handed to that one capture and to no eager launch or other
      capture, and kept alive for the life of the process (a graph may replay it
      at any time). The graph then has no memset node. The reserve is filled by
      :func:`reserve_launch_counters` (``localize_weights`` calls it) or by the
      first eager call, and topped up by an eager call once captures took about
      half of it, to :data:`CAPTURE_RESERVE` pairs or twice the most pairs one
      capture took, whichever is more. A refill that fails does not fail the
      call. When no reserved pair is available (no reserve yet, a capture took
      more pairs than the reserve held, or :data:`CAPTURE_PAIR_LIMIT` pairs were
      made for this device and ``words``), the pair is zeroed inside the capture
      from the capture's own memory pool (unhomed; two memset nodes in the graph;
      the graph keeps it alive).

    A captured pair is used by every replay of its graph: replaying one graph
    concurrently on two streams is not supported, as for any FlashInfer workspace
    held by a graph."""
    import torch

    index = _device_index(device)
    words = int(words)
    if _stream_capturing(index):
        return _captured_pair(index, words)
    handle = int(torch._C._cuda_getCurrentRawStream(index))
    pair: tuple[Any, Any] | None
    if handle in _THREAD_KEYED_HANDLES:
        local = _thread_pairs()
        key = (index, handle, words)
        pair = local.get(key)
        if pair is None:
            pair, lasting = _allocate_pair(words, index)
            for tensor in pair:
                tensor.zero_()
            if lasting:
                local[key] = pair
    else:
        state = _counter_state(index)
        stream_key = (handle, words)
        # A hit takes no lock: one dict read and one dict store of its use stamp
        # (each atomic in CPython); a miss evicts by the stamps under the lock.
        pair = state.stream_pairs.get(stream_key)
        if pair is not None:
            state.used[stream_key] = next(state.clock)
        else:
            # Allocated and zeroed on this stream, ahead of the launch. Another
            # thread may have cached a pair of this stream meanwhile: take that
            # one (same stream, so ordered) and drop this one.
            pair, lasting = _allocate_pair(words, index)
            for tensor in pair:
                tensor.zero_()
            if lasting:
                with state.lock:
                    pair = state.stream_pairs.setdefault(stream_key, pair)
                    state.used[stream_key] = next(state.clock)
                    while len(state.stream_pairs) > STREAM_PAIR_LIMIT:
                        oldest = min(
                            state.stream_pairs, key=lambda k: state.used.get(k, -1)
                        )
                        del state.stream_pairs[oldest]
                        state.used.pop(oldest, None)
                    if len(state.used) > 2 * STREAM_PAIR_LIMIT:
                        # Stamps a hit wrote for a pair evicted meanwhile.
                        for stale in set(state.used) - set(state.stream_pairs):
                            state.used.pop(stale, None)
    # A refill that fails does not fail the call: captures then zero their own
    # pair until a later refill works.  A full reserve needs no refill (checked
    # here first, without the lock, as in _refill_capture_reserve).
    reserve = _counter_state(index).reserves.get(words)
    if reserve is not None and (
        reserve.refilling
        or 2 * len(reserve.free) > max(CAPTURE_RESERVE, 2 * reserve.peak)
    ):
        return pair
    with contextlib.suppress(Exception):
        _refill_capture_reserve(index, words)
    return pair


def _forget_counters() -> None:
    """Test only: drop the cached stream pairs, the capture reserves, this thread's
    pairs and the remembered pool refusals."""
    with _LOCK:
        _COUNTER_STATES.clear()
        _POOLS_UNAVAILABLE.clear()
    _THREAD_PAIRS.__dict__.clear()
