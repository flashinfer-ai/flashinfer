"""NDTensor — thin wrapper around ``ncclNDTensor_t``.

`nccl_ep.NCCLLibrary` exposes ``ncclEpTensorCreate/Destroy/GetData/GetSizes``
in its `exported_functions` table but does NOT wrap them as Python methods.
We use the loaded function pointers via ``lib._funcs[...]`` (the same path
``NCCLLibrary.ncclEpCreateGroup`` itself uses).

API:

* :py:meth:`NDTensor.from_torch(group, tensor, tag)` — register a torch
  tensor's storage as an ``ncclNDTensor_t``. The NDTensor holds a non-owning
  reference; the underlying memory remains owned by the torch tensor.
* :py:meth:`NDTensor.allocate(group, dtype, shape, tag)` — register a
  freshly-allocated tensor with NCCL EP's library allocator (data=nullptr
  triggers cudaMalloc inside the library). The NDTensor owns the allocation
  and frees it in ``__del__`` via ``ncclEpTensorDestroy``.
* :py:meth:`NDTensor.as_torch()` — round-trip back to a torch view via
  ``ncclEpTensorGetData`` + ``ncclEpTensorGetSizes``.
"""

from __future__ import annotations

import contextlib
import ctypes
from typing import TYPE_CHECKING, Optional, Sequence

from .....errors import MoEEpNotBuiltError

if TYPE_CHECKING:
    import torch


_NCCL_LIB = None  # singleton NCCLLibrary instance


def _torch_dtype_to_nccl(dtype: "torch.dtype") -> int:
    """Map torch dtype → ncclDataType_t enum, lazily imported.

    We can't import nccl_ep at module-import time because that triggers the
    libnccl_ep.so load before _load_libnccl_ep() has run.
    """
    from nccl_ep import ncclDataTypeEnum  # type: ignore[import-not-found]

    return ncclDataTypeEnum.from_torch(dtype)


def _torch_dtype_from_nccl(enum_value: int) -> "torch.dtype":
    """Inverse of :func:`_torch_dtype_to_nccl`."""
    import torch

    from nccl_ep import ncclDataTypeEnum  # type: ignore[import-not-found]

    inv = {
        ncclDataTypeEnum.ncclInt8: torch.int8,
        ncclDataTypeEnum.ncclUint8: torch.uint8,
        ncclDataTypeEnum.ncclInt32: torch.int32,
        ncclDataTypeEnum.ncclInt64: torch.int64,
        ncclDataTypeEnum.ncclFloat16: torch.float16,
        ncclDataTypeEnum.ncclFloat32: torch.float32,
        ncclDataTypeEnum.ncclFloat64: torch.float64,
        ncclDataTypeEnum.ncclBfloat16: torch.bfloat16,
    }
    if enum_value not in inv:
        raise ValueError(f"unsupported NCCL dtype enum {enum_value}")
    return inv[enum_value]


def get_nccl_lib():
    """Return a cached :class:`nccl_ep.NCCLLibrary` instance bound to our
    staged ``libnccl_ep.so``. Raises :class:`MoEEpNotBuiltError` if the EP
    plugin wasn't built or the python module isn't importable."""
    global _NCCL_LIB
    if _NCCL_LIB is not None:
        return _NCCL_LIB

    # Force libnccl + libnccl_ep into the process before NCCLLibrary tries
    # to find them on its own search path.
    from . import _load_libnccl_ep, _libs_dir  # noqa: F401  (side effect)

    _load_libnccl_ep()

    try:
        from nccl_ep import HAVE_NCCL_EP, NCCLLibrary  # type: ignore[import-not-found]
    except ImportError as e:
        raise MoEEpNotBuiltError(
            "nccl_ep python bindings not importable. Install with: "
            "pip install -e 3rdparty/nccl/contrib/nccl_ep/python"
        ) from e

    if not HAVE_NCCL_EP:
        raise MoEEpNotBuiltError(
            "nccl_ep module imported but ncclEp* symbols not exposed; "
            "the loaded libnccl_ep.so may be from an older NCCL build"
        )

    so_path = _libs_dir / "libnccl_ep.so"
    _NCCL_LIB = NCCLLibrary(so_file=str(so_path))
    return _NCCL_LIB


class NDTensor:
    """Owning or borrowing wrapper around an ``ncclNDTensor_t``."""

    __slots__ = (
        "_group",
        "_handle",
        "_owns",
        "_dtype",
        "_shape",
        "_tag",
        "_keepalive",
        "_lib",
    )

    def __init__(
        self,
        group: ctypes.c_void_p,
        handle: ctypes.c_void_p,
        dtype: "torch.dtype",
        shape: Sequence[int],
        tag: int,
        owns: bool = False,
        keepalive: Optional["torch.Tensor"] = None,
    ) -> None:
        self._group = group
        self._handle = handle
        self._owns = owns
        self._dtype = dtype
        self._shape = tuple(shape)
        self._tag = tag
        # Hold a reference to the source torch tensor (if borrowing) to keep
        # its storage alive while the NDTensor handle exists.
        self._keepalive = keepalive
        # Cache the NCCL library handle so __del__ doesn't re-resolve it during
        # interpreter shutdown (when module globals may already be cleared).
        self._lib = get_nccl_lib()

    @classmethod
    def from_torch(
        cls,
        group: ctypes.c_void_p,
        tensor: "torch.Tensor",
        tag: int,
    ) -> "NDTensor":
        """Wrap an existing torch tensor as an ncclNDTensor_t (non-owning).

        The NDTensor keeps a reference to the source tensor in ``_keepalive``
        so its storage isn't freed under us.
        """
        if not tensor.is_contiguous():
            raise ValueError("NDTensor.from_torch: tensor must be contiguous")
        if not tensor.is_cuda:
            raise ValueError("NDTensor.from_torch: tensor must be on a CUDA device")

        lib = get_nccl_lib()
        from nccl_ep import ncclNDTensor_t  # type: ignore[import-not-found]

        handle = ncclNDTensor_t()
        sizes = list(tensor.shape) + [0] * (5 - len(tensor.shape))
        lib.NCCL_CHECK(
            lib._funcs["ncclEpTensorCreate"](
                group,
                ctypes.byref(handle),
                len(tensor.shape),
                _torch_dtype_to_nccl(tensor.dtype),
                tag,
                ctypes.c_void_p(tensor.data_ptr()),
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
            )
        )
        return cls(
            group,
            handle,
            tensor.dtype,
            tensor.shape,
            tag,
            owns=False,
            keepalive=tensor,
        )

    @classmethod
    def allocate(
        cls,
        group: ctypes.c_void_p,
        dtype: "torch.dtype",
        shape: Sequence[int],
        tag: int,
    ) -> "NDTensor":
        """Allocate a tensor via NCCL EP's library allocator (data=nullptr)."""
        lib = get_nccl_lib()
        from nccl_ep import ncclNDTensor_t  # type: ignore[import-not-found]

        handle = ncclNDTensor_t()
        sizes = list(shape) + [0] * (5 - len(shape))
        lib.NCCL_CHECK(
            lib._funcs["ncclEpTensorCreate"](
                group,
                ctypes.byref(handle),
                len(shape),
                _torch_dtype_to_nccl(dtype),
                tag,
                ctypes.c_void_p(0),  # nullptr → library allocates
                sizes[0],
                sizes[1],
                sizes[2],
                sizes[3],
                sizes[4],
            )
        )
        return cls(group, handle, dtype, shape, tag, owns=True, keepalive=None)

    @property
    def handle(self) -> ctypes.c_void_p:
        return self._handle

    @property
    def tag(self) -> int:
        return self._tag

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def dtype(self) -> "torch.dtype":
        return self._dtype

    def as_torch(self) -> "torch.Tensor":
        """Return a torch view over the underlying storage.

        For borrowing NDTensors, this is the same storage as the source
        ``_keepalive`` tensor; we just re-build the view from sizes returned
        by the library (which may differ from the original shape if the
        library reshaped it).
        """
        import torch

        if self._keepalive is not None:
            # Borrowing case — re-view the source storage with our recorded shape.
            return self._keepalive.view(*self._shape)

        # Owning case — get the device pointer + sizes from the library and
        # build a torch tensor over it. The library owns the storage; we
        # capture it via __cuda_array_interface__-style construction.
        lib = self._lib
        data_p = ctypes.c_void_p()
        lib.NCCL_CHECK(
            lib._funcs["ncclEpTensorGetData"](self._handle, ctypes.byref(data_p))
        )
        # Reconstruct shape from ncclEpTensorGetSizes (output is uint*).
        sizes_ptr = ctypes.POINTER(ctypes.c_uint)()
        ndim_out = ctypes.c_uint()
        lib.NCCL_CHECK(
            lib._funcs["ncclEpTensorGetSizes"](
                self._handle, ctypes.byref(sizes_ptr), ctypes.byref(ndim_out)
            )
        )
        shape = tuple(sizes_ptr[i] for i in range(ndim_out.value))
        # Wrap the storage via torch.from_blob-equivalent (no copy).
        # torch doesn't ship from_blob in Python; use cuda_array_interface.
        numel = 1
        for s in shape:
            numel *= s
        iface = {
            "version": 2,
            "shape": shape,
            "typestr": _torch_typestr(self._dtype),
            # data_p.value is None for a NULL pointer; coerce to 0 so the
            # interface dict stays a valid (int, bool) tuple.
            "data": (data_p.value or 0, False),
            "strides": None,
        }

        class _CudaArrayProxy:
            __cuda_array_interface__ = iface

        view = torch.as_tensor(_CudaArrayProxy(), device="cuda")
        # bfloat16 has no __cuda_array_interface__ typestr, so we represent it
        # as a 2-byte int above and reinterpret here (no copy, same storage).
        if self._dtype == torch.bfloat16:
            view = view.view(torch.bfloat16)
        # `view` borrows; the NDTensor owns the actual storage and frees it
        # in __del__.
        assert view.numel() == numel
        return view

    def __del__(self) -> None:
        if self._owns and self._handle is not None:
            # Interpreter shutdown can pull the library out from under us;
            # nothing useful we can do if the destroy call fails.
            with contextlib.suppress(Exception):
                self._lib._funcs["ncclEpTensorDestroy"](self._group, self._handle)


def _torch_typestr(dtype: "torch.dtype") -> str:
    """Return a __cuda_array_interface__ typestr string for the dtype."""
    import torch

    mapping = {
        torch.int8: "|i1",
        torch.uint8: "|u1",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.float16: "<f2",
        torch.float32: "<f4",
        torch.float64: "<f8",
        # bfloat16 has no canonical __cuda_array_interface__ typestr; represent
        # it as a 2-byte int and reinterpret to bfloat16 in as_torch (the
        # opaque "<V2" form is rejected by torch.as_tensor).
        torch.bfloat16: "<i2",
    }
    if dtype not in mapping:
        raise ValueError(f"no __cuda_array_interface__ typestr for {dtype}")
    return mapping[dtype]
