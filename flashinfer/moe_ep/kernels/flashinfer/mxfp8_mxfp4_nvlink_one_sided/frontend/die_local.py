# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Die-local weights on GPUs whose memory forms two locality domains (B300: two dies).

Each die reads the HBM attached to it faster than the other die's: on B300 a
DRAM miss homed in the reading SM's own die takes about 320 ns, one homed in the
other die about 520 ns, and streaming die-matched halves of a buffer runs 4-5%
faster than the driver's default allocation, which interleaves both dies in
4 KB runs. Die-local weights store the first and the second half of every
weight matrix's M tiles in the two domains, and the main kernel's clusters claim
the M tiles whose half their die holds (``die_local_weights``).

- ``sm_die_table``: which die each SM is on, measured once per device by a
  pointer chase from every SM into memory homed in each domain. The split
  differs between GPUs of one machine (70 / 78 and 74 / 74 SMs on B300), so it is
  measured rather than assumed.
- ``die_local_weight_legs``: a prepared (weight, scale) pair rewritten half-major,
  ``[2, E, rows / 2, K]`` viewed as ``[2E, rows / 2, K]``, half ``h`` homed in
  locality domain ``h``. Both the block-major weights and the swizzled scale
  factors keep each expert's 128-row blocks outermost, so a half is one
  contiguous run per expert and the main kernel addresses it as expert
  ``h * E + e`` of a tensor with half the rows.

The allocation uses the driver's virtual memory API with
``CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN`` (CUDA 13.5 driver API): one
address range per tensor, its pages mapped from one handle per domain. A page
that straddles the two halves is homed in the domain that holds more of it.
"""

from __future__ import annotations

import ctypes
import dataclasses
import functools
import sys
from typing import Optional, Tuple

import torch

CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT = 149
CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL = 22
CU_MEM_LOCATION_TYPE_DEVICE = 1
CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN = 6
CU_MEM_ALLOCATION_TYPE_PINNED = 1
CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3
CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8

# Entries of the per-SM tables in the main kernel's workspace (``sm_die_table``):
# SM ids (``%smid``) must stay below this.
SM_TABLE_SLOTS = 256
# Far over near DRAM latency below which an SM's die counts as not measured.
MIN_FAR_NEAR_RATIO = 1.25


class _CUmemLocation(ctypes.Structure):
    # A union of ``int id`` and ``{unsigned char deviceId, localityDomainId}``.
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class _CUmemAllocationProp(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("requestedHandleTypes", ctypes.c_int),
        ("location", _CUmemLocation),
        ("win32HandleMetaData", ctypes.c_void_p),
        ("allocFlags", ctypes.c_uint64),
    ]


class _CUmemAccessDesc(ctypes.Structure):
    _fields_ = [("location", _CUmemLocation), ("flags", ctypes.c_int)]


@functools.cache
def _driver() -> ctypes.CDLL:
    lib = ctypes.CDLL("libcuda.so.1")
    u64, size_t, c_int, ptr = (
        ctypes.c_uint64,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_void_p,
    )
    signatures = {
        "cuDeviceGetAttribute": (ctypes.POINTER(c_int), c_int, c_int),
        "cuMemGetAllocationGranularity": (
            ctypes.POINTER(size_t),
            ctypes.POINTER(_CUmemAllocationProp),
            c_int,
        ),
        "cuMemCreate": (
            ctypes.POINTER(u64),
            size_t,
            ctypes.POINTER(_CUmemAllocationProp),
            u64,
        ),
        "cuMemAddressReserve": (ctypes.POINTER(u64), size_t, size_t, u64, u64),
        "cuMemMap": (u64, size_t, size_t, u64, u64),
        "cuMemSetAccess": (u64, size_t, ctypes.POINTER(_CUmemAccessDesc), size_t),
        "cuPointerGetAttribute": (ptr, c_int, u64),
        "cuMemUnmap": (u64, size_t),
        "cuMemAddressFree": (u64, size_t),
        "cuMemRelease": (u64,),
        "cuModuleLoadData": (ctypes.POINTER(ptr), ctypes.c_char_p),
        "cuModuleGetFunction": (ctypes.POINTER(ptr), ptr, ctypes.c_char_p),
        "cuFuncSetAttribute": (ptr, c_int, c_int),
        "cuLaunchKernel": (
            ptr,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ptr,
            ctypes.POINTER(ptr),
            ctypes.POINTER(ptr),
        ),
        "cuModuleUnload": (ptr,),
    }
    for name, argtypes in signatures.items():
        function = getattr(lib, name)
        function.argtypes = argtypes
        function.restype = c_int
    return lib


def _check(result: int, what: str) -> None:
    if result != 0:
        raise RuntimeError(f"{what} failed with CUresult {result}.")


def _current_device() -> int:
    # Driver and runtime device ordinals agree.
    return torch.cuda.current_device()


_LOCALITY_DOMAIN_COUNTS: dict = {}


def locality_domain_count() -> int:
    """Locality domains of the current device (1 where the driver reports none), read once per device."""
    torch.cuda.init()
    device = _current_device()
    count = _LOCALITY_DOMAIN_COUNTS.get(device)
    if count is None:
        value = ctypes.c_int(0)
        result = _driver().cuDeviceGetAttribute(
            ctypes.byref(value), CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, device
        )
        count = max(value.value, 1) if result == 0 else 1
        _LOCALITY_DOMAIN_COUNTS[device] = count
    return count


class LocalityDomainBuffer:
    """Device memory whose first ``split_bytes`` are homed in domain 0, the rest in domain 1.

    The split falls on the page boundary nearest ``split_bytes``; ``split_bytes``
    of ``nbytes`` or more homes every page in domain 0.

    Exposes ``__cuda_array_interface__`` as ``nbytes`` bytes, so ``torch.as_tensor``
    wraps it without a copy and the tensor keeps the buffer alive.
    """

    def __init__(self, nbytes: int, split_bytes: int) -> None:
        driver = _driver()
        self.device = _current_device()
        self.nbytes = int(nbytes)
        self._handles: list = []
        self._ptr = 0
        props = [self._prop(domain) for domain in range(2)]
        granularity = ctypes.c_size_t()
        _check(
            driver.cuMemGetAllocationGranularity(
                ctypes.byref(granularity),
                ctypes.byref(props[0]),
                CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            ),
            "cuMemGetAllocationGranularity",
        )
        page = granularity.value
        pages = max(-(-self.nbytes // page), 1)
        split_page = (
            pages
            if split_bytes >= self.nbytes
            else min(max(round(split_bytes / page), 0), pages)
        )
        self.size = pages * page
        self.split_offset = split_page * page
        address = ctypes.c_uint64()
        _check(
            driver.cuMemAddressReserve(ctypes.byref(address), self.size, page, 0, 0),
            "cuMemAddressReserve",
        )
        self._ptr = address.value
        for domain, (begin, end) in enumerate(((0, split_page), (split_page, pages))):
            if end == begin:
                continue
            handle = ctypes.c_uint64()
            length = (end - begin) * page
            _check(
                driver.cuMemCreate(
                    ctypes.byref(handle), length, ctypes.byref(props[domain]), 0
                ),
                "cuMemCreate",
            )
            self._handles.append((handle.value, begin * page, length))
            _check(
                driver.cuMemMap(self._ptr + begin * page, length, 0, handle.value, 0),
                "cuMemMap",
            )
        access = _CUmemAccessDesc(
            _CUmemLocation(CU_MEM_LOCATION_TYPE_DEVICE, self.device),
            CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        )
        _check(
            driver.cuMemSetAccess(self._ptr, self.size, ctypes.byref(access), 1),
            "cuMemSetAccess",
        )
        for _, offset, _ in self._handles:
            expected = 0 if offset < self.split_offset else 1
            if self.domain_of(offset) != expected:
                raise RuntimeError(
                    f"locality domain {expected} allocation is not homed in domain {expected}."
                )

    def _prop(self, domain: int) -> _CUmemAllocationProp:
        location = _CUmemLocation(
            CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN, self.device | (domain << 8)
        )
        return _CUmemAllocationProp(CU_MEM_ALLOCATION_TYPE_PINNED, 0, location, None, 0)

    def domain_of(self, offset: int) -> int:
        """The locality domain the driver reports for the page at ``offset``."""
        ordinal = ctypes.c_int(-1)
        _check(
            _driver().cuPointerGetAttribute(
                ctypes.byref(ordinal),
                CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL,
                self._ptr + offset,
            ),
            "cuPointerGetAttribute",
        )
        return ordinal.value

    @property
    def data_ptr(self) -> int:
        return self._ptr

    @property
    def __cuda_array_interface__(self) -> dict:
        return {
            "shape": (self.nbytes,),
            "typestr": "|u1",
            "data": (self._ptr, False),
            "version": 3,
        }

    def __del__(self) -> None:
        if not self._ptr or sys.is_finalizing():
            return
        driver = _driver()
        driver.cuMemUnmap(self._ptr, self.size)
        driver.cuMemAddressFree(self._ptr, self.size)
        for handle, _, _ in self._handles:
            driver.cuMemRelease(handle)
        self._ptr = 0


def _domain_tensor(nbytes: int, split_bytes: int, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(LocalityDomainBuffer(nbytes, split_bytes), device=device)


# ----------------------------------------------------------------------------
# SM-to-die map
# ----------------------------------------------------------------------------

_PROBE_SOURCE = r"""
extern "C" __global__ void sm_die_probe(const unsigned* domain0, const unsigned* domain1, unsigned chunk_words,
                                        int hops, int* sm_out, long long* cycles_out) {
  extern __shared__ unsigned smem[];
  if (threadIdx.x != 0) return;
  unsigned sm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  sm_out[blockIdx.x] = (int)sm;
  unsigned idx;
  for (int domain = 0; domain < 2; ++domain) {
    const unsigned* buf = domain == 0 ? domain0 : domain1;
    idx = blockIdx.x * chunk_words;
    // The first hop takes the chunk's TLB miss and stays out of the timing.
    asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(buf + idx) : "memory");
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "r"(idx)
                 : "memory");
    long long start = clock64();
    for (int hop = 0; hop < hops; ++hop) {
      asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(idx) : "l"(buf + idx) : "memory");
    }
    asm volatile("st.volatile.shared.u32 [%0], %1;" ::"r"((unsigned)__cvta_generic_to_shared(smem)), "r"(idx)
                 : "memory");
    cycles_out[blockIdx.x * 2 + domain] = clock64() - start;
  }
}
"""

# Pointer-chase geometry: each probe CTA walks its own chunk of 128-byte lines,
# visiting line (l + stride) % lines after line l, so every hop misses L2 once
# the buffers have been flushed out of it.
_PROBE_CHUNK_LINES = 256
_PROBE_LINE_STRIDE = 97
_PROBE_HOPS = 64
_PROBE_CTAS_PER_SM = 4
# Dynamic SMEM that keeps a second probe CTA off an SM.
_PROBE_SMEM_BYTES = 120 * 1024
_PROBE_ATTEMPTS = 5


@functools.cache
def _probe_function(device: int):
    from cuda.bindings import nvrtc

    major, minor = torch.cuda.get_device_capability(device)
    result, program = nvrtc.nvrtcCreateProgram(
        _PROBE_SOURCE.encode(), b"sm_die_probe.cu", 0, [], []
    )
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcCreateProgram failed: {result}.")
    options = [f"--gpu-architecture=compute_{major}{minor}".encode()]
    (result,) = nvrtc.nvrtcCompileProgram(program, len(options), options)
    if result != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        _, log_size = nvrtc.nvrtcGetProgramLogSize(program)
        log = b" " * log_size
        nvrtc.nvrtcGetProgramLog(program, log)
        raise RuntimeError(
            f"SM-to-die probe failed to compile: {log.decode(errors='replace')}"
        )
    _, ptx_size = nvrtc.nvrtcGetPTXSize(program)
    ptx = b" " * ptx_size
    nvrtc.nvrtcGetPTX(program, ptx)
    nvrtc.nvrtcDestroyProgram(program)
    driver = _driver()
    module, function = ctypes.c_void_p(), ctypes.c_void_p()
    _check(driver.cuModuleLoadData(ctypes.byref(module), ptx), "cuModuleLoadData")
    _check(
        driver.cuModuleGetFunction(ctypes.byref(function), module, b"sm_die_probe"),
        "cuModuleGetFunction",
    )
    _check(
        driver.cuFuncSetAttribute(
            function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _PROBE_SMEM_BYTES
        ),
        "cuFuncSetAttribute",
    )
    return module, function


def _probe_once(
    device: torch.device, sm_count: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(SM id, cycles of the chase into domain 0 / domain 1) of every probe CTA."""
    ctas = sm_count * _PROBE_CTAS_PER_SM
    chunk_words = _PROBE_CHUNK_LINES * 32
    nbytes = ctas * chunk_words * 4
    buffers = []
    for domain in range(2):
        words = _domain_tensor(nbytes, nbytes if domain == 0 else 0, device).view(
            torch.int32
        )
        lines = torch.arange(
            ctas * _PROBE_CHUNK_LINES, device=device, dtype=torch.int64
        )
        chunk, line = lines // _PROBE_CHUNK_LINES, lines % _PROBE_CHUNK_LINES
        successor = (
            chunk * _PROBE_CHUNK_LINES
            + (line + _PROBE_LINE_STRIDE) % _PROBE_CHUNK_LINES
        )
        words.view(-1, 32)[:, 0] = (successor * 32).to(torch.int32)
        buffers.append(words)
    sm_out = torch.full((ctas,), -1, dtype=torch.int32, device=device)
    cycles = torch.zeros((ctas, 2), dtype=torch.int64, device=device)
    # Twice the L2 of streamed writes leave none of the chase lines in L2.
    torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device=device).zero_()
    args = [
        ctypes.c_uint64(buffers[0].data_ptr()),
        ctypes.c_uint64(buffers[1].data_ptr()),
        ctypes.c_uint32(chunk_words),
        ctypes.c_int32(_PROBE_HOPS),
        ctypes.c_uint64(sm_out.data_ptr()),
        ctypes.c_uint64(cycles.data_ptr()),
    ]
    params = (ctypes.c_void_p * len(args))(*[ctypes.addressof(arg) for arg in args])
    _, function = _probe_function(device.index)
    stream = ctypes.c_void_p(torch.cuda.current_stream(device).cuda_stream)
    _check(
        _driver().cuLaunchKernel(
            function, ctas, 1, 1, 32, 1, 1, _PROBE_SMEM_BYTES, stream, params, None
        ),
        "cuLaunchKernel",
    )
    torch.cuda.synchronize(device)
    return sm_out.cpu(), cycles.cpu()


@dataclasses.dataclass(frozen=True)
class SmDieMap:
    """Die of every SM id (``%smid``) of one device, with the latencies that decided it."""

    die: Tuple[int, ...]
    near_cycles: float
    far_cycles: float

    @property
    def sm_count_per_die(self) -> Tuple[int, int]:
        return (self.die.count(0), self.die.count(1))

    def table(self) -> torch.Tensor:
        """The main kernel's ``sm_die_table``: die, rank within the die, then SMs per die.

        Words ``[0, SM_TABLE_SLOTS)`` hold each SM id's die and words
        ``[SM_TABLE_SLOTS, 2 * SM_TABLE_SLOTS)`` its rank among its die's SMs in SM id
        order; the last two words count each die's SMs.
        """
        words = [0] * (2 * SM_TABLE_SLOTS + 2)
        ranks = [0, 0]
        for sm, die in enumerate(self.die):
            words[sm] = die
            words[SM_TABLE_SLOTS + sm] = ranks[die]
            ranks[die] += 1
        words[2 * SM_TABLE_SLOTS : 2 * SM_TABLE_SLOTS + 2] = ranks
        return torch.tensor(words, dtype=torch.int32)


def _decide(
    sm_ids: torch.Tensor, cycles: torch.Tensor, sm_count: int
) -> Optional[SmDieMap]:
    """The map if every SM was measured with a clear near and far domain and each TPC sits on one die."""
    if int(sm_ids.max()) >= SM_TABLE_SLOTS or int(sm_ids.min()) < 0:
        return None
    die, near, far = [], [], []
    for sm in range(sm_count):
        mine = cycles[sm_ids == sm]
        if mine.numel() == 0:
            return None
        median = mine.double().median(dim=0).values
        near_domain = int(median[1] < median[0])
        ratio = float(median[1 - near_domain] / median[near_domain])
        if ratio < MIN_FAR_NEAR_RATIO:
            return None
        die.append(near_domain)
        near.append(float(median[near_domain]))
        far.append(float(median[1 - near_domain]))
    # The two SMs of a TPC share their memory path, so they share a die.
    if any(die[sm] != die[sm + 1] for sm in range(0, sm_count - 1, 2)):
        return None
    if 0 not in die or 1 not in die:
        return None
    hops = float(_PROBE_HOPS)
    return SmDieMap(
        tuple(die),
        sorted(near)[len(near) // 2] / hops,
        sorted(far)[len(far) // 2] / hops,
    )


_SM_DIE_MAPS: dict = {}


def sm_die_map(device: Optional[torch.device] = None) -> SmDieMap:
    """The SM-to-die map of ``device`` (default: the current device), probed once per process.

    Raises RuntimeError where the device has fewer than two locality domains or
    the probe cannot tell the dies apart.
    """
    device = torch.device(
        "cuda", torch.cuda.current_device() if device is None else device.index
    )
    cached = _SM_DIE_MAPS.get(device.index)
    if cached is not None:
        return cached
    with torch.cuda.device(device):
        if locality_domain_count() < 2:
            raise RuntimeError(
                "die-local weights need a device with two locality domains."
            )
        sm_count = torch.cuda.get_device_properties(device).multi_processor_count
        for _ in range(_PROBE_ATTEMPTS):
            sm_ids, cycles = _probe_once(device, sm_count)
            decided = _decide(sm_ids, cycles, sm_count)
            if decided is not None:
                _SM_DIE_MAPS[device.index] = decided
                return decided
    raise RuntimeError("the SM-to-die probe could not assign every SM to a die.")


# ----------------------------------------------------------------------------
# Weights
# ----------------------------------------------------------------------------


def _split_halves(planes: torch.Tensor) -> torch.Tensor:
    """``[E, n]`` bytes as ``[2, E, n / 2]`` in a new buffer, half ``h`` homed in locality domain ``h``."""
    experts, per_expert = planes.shape
    if per_expert % 2:
        raise ValueError("die-local weights need an even number of bytes per expert.")
    split = _domain_tensor(planes.numel(), planes.numel() // 2, planes.device).view(
        2, experts, per_expert // 2
    )
    split.copy_(planes.view(experts, 2, per_expert // 2).permute(1, 0, 2))
    return split


def die_local_weight_leg(
    leg: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One prepared block-major (weight, scale) leg with each half of every expert's rows in its die's HBM.

    ``weight`` is the ``(E, K, rows)`` K-major view the preprocess functions
    return; the result is ``(2E, K, rows / 2)``, entry ``h * E + e`` holding rows
    ``[h * rows / 2, (h + 1) * rows / 2)`` of expert ``e``, homed in locality
    domain ``h``. ``scale`` becomes ``(2E, numel / 2)`` the same way. Each half
    must be whole 128-row blocks.
    """
    weight, scale = leg
    if getattr(weight, "_mega_weight_layout", "kmajor") != "blockmajor":
        raise ValueError(
            "die-local weights are qualified for block-major weights only."
        )
    experts, k_elements, rows = weight.shape
    if rows % 256 or scale.shape[0] != experts or scale.shape[1] % 2:
        raise ValueError("die-local weights need halves of whole 128-row blocks.")
    data = weight.permute(0, 2, 1).view(torch.uint8).reshape(experts, -1)
    split = _split_halves(data)
    split_weight = (
        split.view(2 * experts, rows // 2, -1).view(weight.dtype).permute(0, 2, 1)
    )
    split_weight._mega_weight_layout = weight._mega_weight_layout
    split_weight._mega_die_local = True
    split_scale = (
        _split_halves(scale.view(torch.uint8)).view(2 * experts, -1).view(scale.dtype)
    )
    return split_weight, split_scale


def die_local_weight_legs(transformed_l1, transformed_l2):
    """``die_local_weight_leg`` of both legs, made once and kept with the source weights."""
    legs = []
    for leg in (transformed_l1, transformed_l2):
        if getattr(leg[0], "_mega_die_local", False):
            legs.append(leg)
            continue
        cached = getattr(leg[0], "_mega_die_local_leg", None)
        if cached is None or cached[1] is not leg[1]:
            cached = (die_local_weight_leg(leg), leg[1])
            leg[0]._mega_die_local_leg = cached
        legs.append(cached[0])
    return tuple(legs)


__all__ = [
    "LocalityDomainBuffer",
    "SM_TABLE_SLOTS",
    "SmDieMap",
    "die_local_weight_leg",
    "die_local_weight_legs",
    "locality_domain_count",
    "sm_die_map",
]
