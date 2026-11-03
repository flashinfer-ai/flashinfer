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

Host side of the Cake Ulysses all-to-all package: the per-communicator handle
that turns the IPC allocations a communicator already owns into the kernel's
arguments and launches the exported programs.

The package allocates nothing and opens no IPC handle itself.  The caller
(``flashinfer.comm.ulysses.UlyssesCommunicator``) provisions, per rank, one
symmetric staging allocation of ``max_bytes`` bytes and one symmetric signal
allocation of at least :data:`cake_jit.SIGNAL_BYTES` bytes, exchanges their
device addresses over CUDA IPC, and hands every rank's two addresses to
:class:`CakeUlyssesA2A`.  The handle builds the two CPU ``int64`` peer tables
once, views this rank's own signal allocation as a ``uint32`` tensor (zeroed
once by :meth:`CakeUlyssesA2A.zero_signals`), binds the records' argument plans
once, and launches with the caller's input and output: no allocation, no copy
and no host synchronization per call, so a call is CUDA-graph capturable.

An output every rank allocated together and exchanged over CUDA IPC (the
caller's registered output) takes the direct route instead:
:meth:`CakeUlyssesA2A.bind_direct` loads and binds its programs once,
:func:`peer_table` turns the ranks' output addresses into the output's peer
table, :func:`device_view` views this rank's allocation as the output tensor,
and :meth:`CakeUlyssesA2A.launch_direct` writes every block straight into its
destination rank's output.

The route decision is :func:`route_arch`: the device's capability names an
exported architecture and ``build.json`` delivers the scatter and the gather
library of ``(dtype, world_size)`` for it.  A communicator whose device or
element type has no delivered route keeps FlashInfer's portable NVLink kernel.
"""

from __future__ import annotations

import ctypes
import functools
from typing import Any, Optional, Sequence

import torch
import tvm_ffi

from . import cake_jit

_DTYPE_NAMES: dict[torch.dtype, str] = {
    torch.bfloat16: "bf16",
    torch.float16: "f16",
    torch.float32: "f32",
}
# The logical arguments a launch supplies per call, in the order ``launch`` and ``launch_direct`` take them.
_PER_CALL = ("inp", "out", "B", "S_local", "H_local", "D")
_PER_CALL_DIRECT = ("inp", "out", "out_peers", "B", "S_local", "H_local", "D")


def dtype_name(dtype: torch.dtype) -> str:
    """The registry's name of a torch element type (``bf16`` / ``f16`` / ``f32``)."""

    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise ValueError(f"dtype must be one of {tuple(_DTYPE_NAMES)}, got {dtype!r}")
    return name


@functools.cache
def device_arch(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device, or ``None`` when the package ships none for it."""

    return cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device_index))
    )


def route_arch(
    device_index: int, dtype: torch.dtype, world_size: int, *, direct: bool = False
) -> Optional[str]:
    """The architecture whose delivered libraries own both collectives of a communicator, else ``None``.

    ``None`` when the device is not an exported target, the element type or
    world size is outside the package's portfolio, or ``build.json`` does not
    deliver both the scatter and the gather library of the route (staged, or
    direct with ``direct=True``) for the architecture.
    """

    arch = device_arch(device_index)
    if arch is None or dtype not in _DTYPE_NAMES:
        return None
    if cake_jit.route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device_index)),
        dtype=_DTYPE_NAMES[dtype],
        world_size=world_size,
        direct=direct,
    ):
        return arch
    return None


def warm(
    arch: str, dtype: torch.dtype, world_size: int, *, direct: bool = False
) -> None:
    """Load (JIT-building when no prebuilt file exists) the scatter and gather libraries of one communicator
    on the route (staged, or direct with ``direct=True``)."""

    name = dtype_name(dtype)
    for mode in cake_jit.MODES:
        cake_jit.load(cake_jit.record_name(name, world_size, mode, direct=direct), arch)


def peer_table(addresses: Sequence[int]) -> torch.Tensor:
    """A by-value peer table as the bindings take it: a contiguous CPU ``int64`` tensor of exactly
    :data:`cake_jit.PEER_TABLE_CAPACITY` device addresses, entry ``r`` rank ``r``'s and the rest zero."""

    values = [int(address) for address in addresses]
    if not 1 <= len(values) <= cake_jit.PEER_TABLE_CAPACITY:
        raise ValueError(
            f"a peer table holds 1..{cake_jit.PEER_TABLE_CAPACITY} addresses, got {len(values)}"
        )
    if any(address <= 0 for address in values):
        raise ValueError("every peer address must be a nonzero device pointer")
    values += [0] * (cake_jit.PEER_TABLE_CAPACITY - len(values))
    return torch.tensor(values, dtype=torch.int64, device="cpu")


# ---- a torch view of a raw device allocation ----------------------------------
#
# The signal allocation and a registered output are plain ``cudaMalloc``s of the
# caller's IPC transactions, not torch tensors; the bindings check
# ``signals_self`` and ``out`` as contiguous CUDA tensors on the kernel's device.
# A DLPack capsule over the raw pointer gives that view without a copy.  The
# capsule's structures live in one C allocation whose deleter is the C
# library's ``free``: torch calls it when the view's storage is released, so
# the structures outlive every tensor sharing that storage (a narrowed view
# made under ``torch.inference_mode`` records no base, and ``.data`` drops
# it), and no Python code runs at release.  The device memory itself belongs
# to the caller's IPC transaction, which frees it.


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_size_t),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLDeleter),
]


class _ViewBlock(ctypes.Structure):
    """One C allocation: the managed tensor first (the deleter frees it by that address), then its 1-D shape and
    strides."""

    _fields_ = [
        ("managed", _DLManagedTensor),
        ("shape", ctypes.c_int64),
        ("strides", ctypes.c_int64),
    ]


_LIBC = ctypes.CDLL(None)
_LIBC.malloc.restype = ctypes.c_void_p
_LIBC.malloc.argtypes = [ctypes.c_size_t]
_LIBC.free.restype = None
_LIBC.free.argtypes = [ctypes.c_void_p]
_FREE_ADDRESS = ctypes.cast(_LIBC.free, ctypes.c_void_p).value


_K_DL_CUDA = 2
_K_DL_UINT = 1
_K_DL_FLOAT = 2
_K_DL_BFLOAT = 4
# DLPack (type code, bits) of every element type a view takes.
_DL_TYPES: dict[torch.dtype, tuple[int, int]] = {
    torch.uint32: (_K_DL_UINT, 32),
    torch.bfloat16: (_K_DL_BFLOAT, 16),
    torch.float16: (_K_DL_FLOAT, 16),
    torch.float32: (_K_DL_FLOAT, 32),
}


def device_view(
    pointer: int, device_index: int, dtype: torch.dtype, numel: int
) -> torch.Tensor:
    """A contiguous CUDA ``dtype[numel]`` tensor over the device allocation at ``pointer`` (no copy).

    The tensor does not own the memory: the allocation's owner frees it, after
    which the tensor must not be used.
    """

    if int(pointer) <= 0:
        raise ValueError("the view's pointer must be a nonzero device pointer")
    if dtype not in _DL_TYPES:
        raise ValueError(f"dtype must be one of {tuple(_DL_TYPES)}, got {dtype!r}")
    if int(numel) < 1:
        raise ValueError(f"a view holds at least one element, got {numel!r}")
    code, bits = _DL_TYPES[dtype]
    address = _LIBC.malloc(ctypes.sizeof(_ViewBlock))
    if not address:
        raise MemoryError("cannot allocate a DLPack view's structures")
    block = _ViewBlock.from_address(address)
    block.shape = int(numel)
    block.strides = 1
    managed = block.managed
    managed.dl_tensor.data = ctypes.c_void_p(int(pointer))
    managed.dl_tensor.device = _DLDevice(_K_DL_CUDA, int(device_index))
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(code, bits, 1)
    managed.dl_tensor.shape = ctypes.cast(
        address + _ViewBlock.shape.offset, ctypes.POINTER(ctypes.c_int64)
    )
    managed.dl_tensor.strides = ctypes.cast(
        address + _ViewBlock.strides.offset, ctypes.POINTER(ctypes.c_int64)
    )
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _DLDeleter(_FREE_ADDRESS)
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(address, b"dltensor", None)
    try:
        return torch.utils.dlpack.from_dlpack(capsule)
    except BaseException:
        # torch renames a capsule it consumed (and then owns the structures);
        # one still named "dltensor" was not taken, so its block is ours.
        is_valid = ctypes.pythonapi.PyCapsule_IsValid
        is_valid.restype = ctypes.c_int
        is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
        if is_valid(capsule, b"dltensor"):
            _LIBC.free(address)
        raise


def signal_view(pointer: int, device_index: int) -> torch.Tensor:
    """This rank's signal allocation as a contiguous CUDA ``uint32[SIGNAL_WORDS]`` tensor (no copy)."""

    return device_view(pointer, device_index, torch.uint32, cake_jit.SIGNAL_WORDS)


# ---- the per-communicator handle --------------------------------------------------


class CakeUlyssesA2A:
    """One communicator's bound launches of the exported scatter and gather programs.

    Parameters
    ----------
    staging_ptrs, signal_ptrs : sequence of int
        Rank ``r``'s symmetric staging and signal device addresses at entry
        ``r`` (own allocation for this rank, IPC-opened for the peers), exactly
        ``world_size`` entries each.
    rank, world_size : int
        This rank and the group size (2, 4, 6 or 8).
    dtype : torch.dtype
        Element type of every collective (float16 / bfloat16 / float32).
    device_index : int
        The CUDA device of this rank; its capability must be an exported target
        whose libraries ``build.json`` delivers (:func:`route_arch`).
    """

    def __init__(
        self,
        *,
        staging_ptrs: Sequence[int],
        signal_ptrs: Sequence[int],
        rank: int,
        world_size: int,
        dtype: torch.dtype,
        device_index: int,
    ) -> None:
        if world_size not in cake_jit.WORLD_SIZES:
            raise ValueError(
                f"world_size must be one of {cake_jit.WORLD_SIZES}, got {world_size!r}"
            )
        if type(rank) is not int or not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank!r}")
        if len(staging_ptrs) != world_size or len(signal_ptrs) != world_size:
            raise ValueError(
                "staging_ptrs and signal_ptrs must hold exactly world_size addresses"
            )
        arch = route_arch(device_index, dtype, world_size)
        if arch is None:
            raise RuntimeError(
                f"device {device_index} has no delivered Cake Ulysses all-to-all route for "
                f"{dtype} with world size {world_size}"
            )
        self.rank = rank
        self.world_size = world_size
        self.dtype = dtype
        self.device_index = int(device_index)
        self.arch = arch
        self.staging_peers = peer_table(staging_ptrs)
        self.signals_peers = peer_table(signal_ptrs)
        self.signals_self = signal_view(signal_ptrs[rank], self.device_index)
        self._launches = self._bind(direct=False)
        self._direct_launches: Optional[
            dict[int, tuple[Any, list[Any], tuple[int, ...]]]
        ] = None
        self._closed = False

    def _bind(
        self, *, direct: bool
    ) -> dict[int, tuple[Any, list[Any], tuple[int, ...]]]:
        """Each mode's plan on a route bound once: the fixed arguments in place, one slot index per per-call
        value (found by identity through a sentinel per value, so no argument is compared by value)."""

        fixed: dict[str, Any] = {
            "signals_peers": self.signals_peers,
            "signals_self": self.signals_self,
            "rank": self.rank,
        }
        if not direct:
            fixed["staging_peers"] = self.staging_peers
        per_call = _PER_CALL_DIRECT if direct else _PER_CALL
        name = dtype_name(self.dtype)
        launches: dict[int, tuple[Any, list[Any], tuple[int, ...]]] = {}
        for mode in cake_jit.MODES:
            record_name = cake_jit.record_name(
                name, self.world_size, mode, direct=direct
            )
            record = cake_jit.MODULES[record_name]
            sentinels = {key: object() for key in per_call}
            template = cake_jit.plan_arguments(record, {**fixed, **sentinels})
            slots = tuple(
                next(
                    index
                    for index, value in enumerate(template)
                    if value is sentinels[key]
                )
                for key in per_call
            )
            module = cake_jit.load(record_name, self.arch)
            entry = getattr(module, record["ffi_entry"])
            launches[mode] = (entry, template, slots)
        return launches

    @property
    def direct_route(self) -> bool:
        """Whether ``build.json`` delivers the direct route's scatter and gather libraries for this handle."""

        return (
            route_arch(self.device_index, self.dtype, self.world_size, direct=True)
            == self.arch
        )

    def bind_direct(self) -> None:
        """Load (JIT-building when needed) and bind the direct route's programs once; idempotent.

        ``RuntimeError`` when the package does not deliver them for this
        handle's architecture, element type and world size.
        """

        self._check_open()
        if self._direct_launches is not None:
            return
        if not self.direct_route:
            raise RuntimeError(
                f"the package delivers no direct Cake Ulysses all-to-all route for "
                f"{self.dtype} with world size {self.world_size} on {self.arch}"
            )
        self._direct_launches = self._bind(direct=True)

    def output_view(self, pointer: int, numel: int) -> torch.Tensor:
        """This rank's registered output allocation at ``pointer`` as a contiguous ``numel``-element tensor of
        the handle's element type (no copy; the caller owns and frees the allocation)."""

        return device_view(pointer, self.device_index, self.dtype, numel)

    def zero_signals(self) -> None:
        """Zero this rank's signal words on the current stream (once, before the first collective)."""

        self._check_open()
        self.signals_self.zero_()

    def launch(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        B: int,
        S_local: int,
        H: int,
        D: int,
        mode: int,
    ) -> None:
        """One collective on the current torch stream: ``mode`` 0 scatters heads, ``mode`` 1 gathers them.

        ``H`` is the global head count; the kernel takes ``H_local = H // world_size``.
        The caller validates the operands (contiguous CUDA tensors of the
        communicator's dtype on its device, shapes consistent with the mode);
        the binding re-checks the tensors and the scalar ranges.
        """

        self._check_open()
        if mode not in self._launches:
            raise ValueError(f"mode must be one of {cake_jit.MODES}, got {mode!r}")
        if H % self.world_size:
            raise ValueError(
                f"the global head count {H} is not divisible by world size {self.world_size}"
            )
        entry, template, slots = self._launches[mode]
        args = list(template)
        values = (inp, out, B, S_local, H // self.world_size, D)
        for slot, value in zip(slots, values, strict=True):
            args[slot] = value
        with tvm_ffi.use_torch_stream():
            entry(*args)

    def launch_direct(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        out_peers: torch.Tensor,
        B: int,
        S_local: int,
        H: int,
        D: int,
        mode: int,
    ) -> None:
        """One collective on the direct route, on the current torch stream: every block goes straight into its
        destination rank's output.

        ``out`` is this rank's registered output and ``out_peers`` its peer
        table (:func:`peer_table` of every rank's address of the same
        registration); every rank passes its own view of the same registration.
        Otherwise as :meth:`launch`; :meth:`bind_direct` must have run.
        """

        self._check_open()
        if self._direct_launches is None:
            raise RuntimeError("bind_direct() must run before the first direct launch")
        if mode not in self._direct_launches:
            raise ValueError(f"mode must be one of {cake_jit.MODES}, got {mode!r}")
        if H % self.world_size:
            raise ValueError(
                f"the global head count {H} is not divisible by world size {self.world_size}"
            )
        entry, template, slots = self._direct_launches[mode]
        args = list(template)
        values = (inp, out, out_peers, B, S_local, H // self.world_size, D)
        for slot, value in zip(slots, values, strict=True):
            args[slot] = value
        with tvm_ffi.use_torch_stream():
            entry(*args)

    def dispose(self) -> None:
        """Drop the bound arguments; the caller frees the allocations. Idempotent."""

        self._closed = True
        self._launches = {}
        self._direct_launches = None
        self.signals_self = None  # type: ignore[assignment]
        self.staging_peers = None  # type: ignore[assignment]
        self.signals_peers = None  # type: ignore[assignment]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this Cake Ulysses all-to-all handle has been disposed")


__all__ = [
    "CakeUlyssesA2A",
    "device_arch",
    "device_view",
    "dtype_name",
    "peer_table",
    "route_arch",
    "signal_view",
    "warm",
]
