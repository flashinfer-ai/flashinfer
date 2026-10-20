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
symmetric staging allocation of ``max_elems`` elements and one symmetric signal
allocation of at least :data:`cake_jit.SIGNAL_BYTES` bytes, exchanges their
device addresses over CUDA IPC, and hands every rank's two addresses to
:class:`CakeUlyssesA2A`.  The handle builds the two CPU ``int64`` peer tables
once, views this rank's own signal allocation as a ``uint32`` tensor (zeroed
once by :meth:`CakeUlyssesA2A.zero_signals`), binds the records' argument plans
once, and launches with the caller's input and output: no allocation, no copy
and no host synchronization per call, so a call is CUDA-graph capturable.

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
# The logical arguments a launch supplies per call, in the order ``launch`` takes them.
_PER_CALL = ("inp", "out", "B", "S_local", "H_local", "D")


def dtype_name(dtype: torch.dtype) -> str:
    """The registry's name of a torch element type (``bf16`` / ``f16`` / ``f32``)."""

    name = _DTYPE_NAMES.get(dtype)
    if name is None:
        raise ValueError(
            f"dtype must be one of {tuple(_DTYPE_NAMES)}, got {dtype!r}"
        )
    return name


@functools.cache
def device_arch(device_index: int) -> Optional[str]:
    """The exported architecture of one CUDA device, or ``None`` when the package ships none for it."""

    return cake_jit.arch_for_capability(
        tuple(torch.cuda.get_device_capability(device_index))
    )


def route_arch(device_index: int, dtype: torch.dtype, world_size: int) -> Optional[str]:
    """The architecture whose delivered libraries own both collectives of a communicator, else ``None``.

    ``None`` when the device is not an exported target, the element type or
    world size is outside the package's portfolio, or ``build.json`` does not
    deliver both the scatter and the gather library for the architecture.
    """

    arch = device_arch(device_index)
    if arch is None or dtype not in _DTYPE_NAMES:
        return None
    if cake_jit.route_applies(
        device_capability=tuple(torch.cuda.get_device_capability(device_index)),
        dtype=_DTYPE_NAMES[dtype],
        world_size=world_size,
    ):
        return arch
    return None


def warm(arch: str, dtype: torch.dtype, world_size: int) -> None:
    """Load (JIT-building when no prebuilt file exists) the scatter and gather libraries of one communicator."""

    name = dtype_name(dtype)
    for mode in cake_jit.MODES:
        cake_jit.load(cake_jit.record_name(name, world_size, mode), arch)


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
# The signal allocation is a plain ``cudaMalloc`` of the caller's IPC transaction,
# not a torch tensor; the binding checks ``signals_self`` as a contiguous CUDA
# ``uint32`` tensor on the kernel's device.  A DLPack capsule over the raw pointer
# with a no-op deleter gives that view without a copy; the capsule's backing
# structures stay alive on the returned tensor.


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


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


_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))),
]


@ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
def _no_op_deleter(_managed: Any) -> None:
    # The memory belongs to the caller's IPC transaction, which frees it.
    pass


_K_DL_CUDA = 2
_K_DL_UINT = 1


class _ViewKeepAlive:
    """The ctypes objects a DLPack view points into, kept alive on the view."""

    def __init__(self, shape: Any, strides: Any, managed: Any) -> None:
        self.shape = shape
        self.strides = strides
        self.managed = managed


def signal_view(pointer: int, device_index: int) -> torch.Tensor:
    """This rank's signal allocation as a contiguous CUDA ``uint32[SIGNAL_WORDS]`` tensor (no copy)."""

    if int(pointer) <= 0:
        raise ValueError("the signal pointer must be a nonzero device pointer")
    words = cake_jit.SIGNAL_WORDS
    shape = (ctypes.c_int64 * 1)(words)
    strides = (ctypes.c_int64 * 1)(1)
    managed = _DLManagedTensor()
    managed.dl_tensor.data = ctypes.c_void_p(int(pointer))
    managed.dl_tensor.device = _DLDevice(_K_DL_CUDA, int(device_index))
    managed.dl_tensor.ndim = 1
    managed.dl_tensor.dtype = _DLDataType(_K_DL_UINT, 32, 1)
    managed.dl_tensor.shape = ctypes.cast(shape, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.strides = ctypes.cast(strides, ctypes.POINTER(ctypes.c_int64))
    managed.dl_tensor.byte_offset = 0
    managed.manager_ctx = None
    managed.deleter = _no_op_deleter
    new_capsule = ctypes.pythonapi.PyCapsule_New
    new_capsule.restype = ctypes.py_object
    new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = new_capsule(ctypes.addressof(managed), b"dltensor", None)
    view = torch.utils.dlpack.from_dlpack(capsule)
    view._cake_keep_alive = _ViewKeepAlive(shape, strides, managed)  # type: ignore[attr-defined]
    return view


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
        name = dtype_name(dtype)
        fixed = {
            "staging_peers": self.staging_peers,
            "signals_peers": self.signals_peers,
            "signals_self": self.signals_self,
            "rank": rank,
        }
        # Bind each record's plan once: the fixed arguments in place, one slot index per per-call value (found
        # by identity through a sentinel per value, so no argument is compared by value).
        self._launches: dict[int, tuple[Any, list[Any], tuple[int, ...]]] = {}
        for mode in cake_jit.MODES:
            record_name = cake_jit.record_name(name, world_size, mode)
            record = cake_jit.MODULES[record_name]
            sentinels = {key: object() for key in _PER_CALL}
            template = cake_jit.plan_arguments(record, {**fixed, **sentinels})
            slots = tuple(
                next(index for index, value in enumerate(template) if value is sentinels[key])
                for key in _PER_CALL
            )
            module = cake_jit.load(record_name, arch)
            entry = getattr(module, record["ffi_entry"])
            self._launches[mode] = (entry, template, slots)
        self._closed = False

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

    def dispose(self) -> None:
        """Drop the bound arguments; the caller frees the allocations. Idempotent."""

        self._closed = True
        self._launches = {}
        self.signals_self = None  # type: ignore[assignment]
        self.staging_peers = None  # type: ignore[assignment]
        self.signals_peers = None  # type: ignore[assignment]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("this Cake Ulysses all-to-all handle has been disposed")


__all__ = [
    "CakeUlyssesA2A",
    "device_arch",
    "dtype_name",
    "peer_table",
    "route_arch",
    "signal_view",
    "warm",
]
