# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import ctypes
import sys
from ctypes import (
    CFUNCTYPE,
    POINTER,
    c_int,
    c_int64,
    c_size_t,
    c_uint8,
    c_uint16,
    c_void_p,
    pointer,
)

import torch


# Define data structures required for DLPack
class DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", c_uint8),  # Data type code, e.g., 2 for float
        ("bits", c_uint8),  # Number of bits per element, e.g., 32
        ("lanes", c_uint16),  # Number of lanes, usually 1
    ]


class DLDevice(ctypes.Structure):
    _fields_ = [
        ("device_type", c_int),  # Device type, typically 2 for GPU
        ("device_id", c_int),  # Device ID, usually 0 for default GPU
    ]


class DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),  # Data pointer
        ("device", DLDevice),  # Device information
        ("ndim", c_int),  # Number of dimensions
        ("dtype", DLDataType),  # Data type
        ("shape", POINTER(c_int64)),  # Pointer to array of dimension sizes
        (
            "strides",
            POINTER(c_int64),
        ),  # Pointer to strides array (can be NULL for default contiguous layout)
        ("byte_offset", c_size_t),  # Byte offset (usually 0)
    ]


# Define DLManagedTensor structure, with deleter prototype void(*deleter)(DLManagedTensor*)
class DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorPointer = POINTER(DLManagedTensor)


DLManagedTensor._fields_ = [
    ("dl_tensor", DLTensor),
    ("manager_ctx", c_void_p),
    ("deleter", CFUNCTYPE(None, POINTER(DLManagedTensor))),
]


_Py_IncRef = ctypes.PYFUNCTYPE(None, c_void_p)(("Py_IncRef", ctypes.pythonapi))
_Py_DecRef = ctypes.PYFUNCTYPE(None, c_void_p)(("Py_DecRef", ctypes.pythonapi))
_PyCapsule_New = ctypes.PYFUNCTYPE(
    ctypes.py_object, c_void_p, ctypes.c_char_p, c_void_p
)(("PyCapsule_New", ctypes.pythonapi))
_PyCapsule_IsValid = ctypes.PYFUNCTYPE(c_int, c_void_p, ctypes.c_char_p)(
    ("PyCapsule_IsValid", ctypes.pythonapi)
)
_PyCapsule_GetPointer = ctypes.PYFUNCTYPE(c_void_p, c_void_p, ctypes.c_char_p)(
    ("PyCapsule_GetPointer", ctypes.pythonapi)
)
_DLPACK_CAPSULE_NAME = b"dltensor"


class _DLPackContext:
    def __init__(self, managed_tensor, shape, strides, owner):
        self.managed_tensor = managed_tensor
        self.shape = shape
        self.strides = strides
        self.owner = owner
        # PyCapsule stores only a function pointer, not a Python reference.
        self.capsule_deleter = _delete_capsule


@CFUNCTYPE(None, POINTER(DLManagedTensor))
def _delete_managed_tensor(ptr, _decref=_Py_DecRef, _is_finalizing=sys.is_finalizing):
    # ctypes acquires the GIL for this callback. Keep late shutdown consumers
    # alive rather than releasing Python objects during interpreter teardown.
    if _is_finalizing():
        return
    context = ptr.contents.manager_ctx
    ptr.contents.manager_ctx = None
    if context:
        _decref(context)


@CFUNCTYPE(None, c_void_p)
def _delete_capsule(
    capsule,
    _is_valid=_PyCapsule_IsValid,
    _get_pointer=_PyCapsule_GetPointer,
    _delete=_delete_managed_tensor,
    _cast=ctypes.cast,
    _pointer_type=_DLManagedTensorPointer,
    _name=_DLPACK_CAPSULE_NAME,
):
    # Consuming a capsule renames it to used_dltensor and transfers cleanup
    # to the consumer. An unconsumed capsule still owns its descriptor.
    if _is_valid(capsule, _name):
        _delete(_cast(_get_pointer(capsule, _name), _pointer_type))


class CapsuleWrapper:
    """A DLPack capsule and its metadata, retained for callers inspecting the view."""

    def __init__(self, capsule, shape_array, managed_tensor):
        """
        Initialize the CapsuleWrapper with the necessary objects.

        Parameters:
            capsule: The PyCapsule object that follows the DLPack protocol
            shape_array: The array containing tensor shape information
            managed_tensor: The DLManagedTensor instance that the capsule points to
        """
        self.capsule = (
            capsule  # The main PyCapsule object that can be passed to other libraries
        )
        self._shape_array = shape_array  # Keep reference to prevent garbage collection
        self._managed_tensor = (
            managed_tensor  # Keep reference to prevent garbage collection
        )


def create_dlpack_capsule(
    ptr, segment_size, segment_stride, num_segments, torch_dtype, dev_id, *, owner=None
):
    """
    Parameters:
      ptr: GPU memory address obtained from cudaMalloc (Python int)
      segment_size: Memory size of each segments in bytes
      segment_stride: Memory stride size between segments in bytes
      num_segments: Number of segments
      torch_dtype: torch dtype
      dev_id: device id.
      owner: optional allocation owner to retain until the consumer releases storage.
    Returns:
      A CapsuleWrapper whose capsule can be consumed by torch.utils.dlpack.from_dlpack.
    """
    bits_per_elements = 0
    dldata_type_code = 0
    # refer to https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h#L160
    if torch_dtype in [
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.bfloat16,
        torch.float16,
        torch.float32,
        torch.float64,
    ]:
        bits_per_elements = torch.finfo(torch_dtype).bits
        dldata_type_code = 2
    elif torch_dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 0
    elif torch_dtype in [torch.uint8, torch.uint16, torch.uint32, torch.uint64]:
        bits_per_elements = torch.iinfo(torch_dtype).bits
        dldata_type_code = 1
    else:
        raise NotImplementedError(torch_dtype)
    bytes_per_element = bits_per_elements // 8
    # Allocate space for shape (constructing a one-dimensional tensor here)
    ShapeArrayType = c_int64 * 2  # 1 dimension
    shape_array = ShapeArrayType(num_segments, segment_size // bytes_per_element)
    stride_array = ShapeArrayType(segment_stride // bytes_per_element, 1)
    # Set device information: GPU (device_type=2) and device_id=dev_id (modify as needed)
    device = DLDevice(device_type=2, device_id=dev_id)
    # Set data type
    dtype = DLDataType(code=dldata_type_code, bits=bits_per_elements, lanes=1)
    # Construct DLTensor
    dltensor = DLTensor()
    dltensor.data = c_void_p(ptr)
    dltensor.device = device
    dltensor.ndim = 2
    dltensor.dtype = dtype
    dltensor.shape = ctypes.cast(shape_array, POINTER(c_int64))
    dltensor.strides = ctypes.cast(stride_array, POINTER(c_int64))
    dltensor.byte_offset = 0
    managed_tensor = DLManagedTensor()
    managed_tensor.dl_tensor = dltensor
    managed_tensor.deleter = _delete_managed_tensor
    context = _DLPackContext(managed_tensor, shape_array, stride_array, owner)
    managed_tensor.manager_ctx = id(context)
    # This reference belongs to DLPack, not to a Python tensor. Dtype views
    # can outlive that tensor while sharing its underlying storage.
    _Py_IncRef(id(context))
    try:
        capsule = _PyCapsule_New(
            pointer(managed_tensor), _DLPACK_CAPSULE_NAME, _delete_capsule
        )
    except BaseException:
        _delete_managed_tensor(pointer(managed_tensor))
        raise
    return CapsuleWrapper(capsule, shape_array, managed_tensor)


def pack_strided_memory(
    ptr: int,
    segment_size: int,
    segment_stride: int,
    num_segments: int,
    dtype: torch.dtype,
    dev_id: int,
    *,
    owner=None,
) -> torch.Tensor:
    r"""Pack a strided device allocation as a PyTorch tensor view.

    A fresh DLPack capsule is created on every call, so each tensor returned
    by this function consumes its own capsule.

    Parameters
    ----------
    ptr : int
        Device pointer (e.g. one returned by ``cudaMalloc``).
    segment_size : int
        Size of each segment in bytes.
    segment_stride : int
        Stride between consecutive segments in bytes.
    num_segments : int
        Number of segments to expose.
    dtype : torch.dtype
        Element dtype of the resulting tensor.
    dev_id : int
        CUDA device ID hosting ``ptr``.
    owner : object, optional
        Allocation owner retained until the last tensor or storage view is
        released. Without it, the caller must keep the allocation alive.

    Returns
    -------
    torch.Tensor
        A tensor that views the provided device memory.
    """
    # Create a new capsule each time
    capsule_wrapper = create_dlpack_capsule(
        ptr, segment_size, segment_stride, num_segments, dtype, dev_id, owner=owner
    )
    torch_tensor = torch.utils.dlpack.from_dlpack(capsule_wrapper.capsule)
    torch_tensor._capsule_wrapper = capsule_wrapper
    return torch_tensor
