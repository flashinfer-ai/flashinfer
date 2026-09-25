# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepared eager submission of a CuTe-compiled device image.

The native extension stores a CUDA launch configuration only. All device code
comes from CuTe DSL. Descriptor preparation reads storage geometry, never GPU
input contents; each forward reads the current contents of the bound tensors.
"""

import ctypes
import functools
import hashlib
import json
from contextlib import suppress
from pathlib import Path

import torch
from cuda.bindings import driver

from .abi import PARAMETERS, TENSOR_MAPS


def checked(result):
    status, *values = result
    if status != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA driver call failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


@functools.cache
def native_module():
    from tvm_ffi import cpp
    from tvm_ffi.cpp.extension import _find_cuda_home

    source = Path(__file__).with_name("submission.cpp").read_text()
    header = Path(_find_cuda_home()) / "include/cuda.h"
    identity = dict(
        source=hashlib.sha256(source.encode()).hexdigest(),
        cuda_header=hashlib.sha256(header.read_bytes()).hexdigest(),
        flags=["-O3", "-lcuda"],
    )
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return cpp.load_inline(
        "w4a8_cutedsl_submission_" + key,
        cpp_sources=source,
        extra_include_paths=[str(header.parent)],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
    )


def tensor_map(tensor, spec):
    dtype, box, swizzle, promotion = spec
    if tensor.ndim != 2 or not tensor.is_contiguous() or not tensor.is_cuda:
        raise ValueError("tensor maps require contiguous two-dimensional CUDA storage")
    expected = {
        "16U4_ALIGN16B": torch.uint8,
        "UINT8": torch.uint8,
        "UINT32": torch.uint32,
    }[dtype]
    if tensor.dtype != expected:
        raise TypeError(f"tensor map expects {expected}, got {tensor.dtype}")
    row_bytes = tensor.stride(0) * tensor.element_size()
    if tensor.data_ptr() % 16 or row_bytes % 16:
        raise ValueError("tensor map addresses and row strides must be 16-byte aligned")
    factor = 2 if dtype == "16U4_ALIGN16B" else 1
    return checked(
        driver.cuTensorMapEncodeTiled(
            getattr(driver.CUtensorMapDataType, "CU_TENSOR_MAP_DATA_TYPE_" + dtype),
            2,
            tensor.data_ptr(),
            [
                driver.cuuint64_t(tensor.shape[1] * factor),
                driver.cuuint64_t(tensor.shape[0]),
            ],
            [driver.cuuint64_t(row_bytes)],
            [driver.cuuint32_t(x) for x in box],
            [driver.cuuint32_t(1)] * 2,
            driver.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE,
            getattr(driver.CUtensorMapSwizzle, "CU_TENSOR_MAP_SWIZZLE_" + swizzle),
            getattr(
                driver.CUtensorMapL2promotion, "CU_TENSOR_MAP_L2_PROMOTION_" + promotion
            ),
            driver.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
        )
    )


class DeviceImage:
    def __init__(self, image):
        self.image = image
        self.module = checked(driver.cuModuleLoadData(image))
        count = checked(driver.cuModuleGetFunctionCount(self.module))
        if count != 1:
            raise RuntimeError(f"expected one unified device entry, got {count}")
        functions = checked(driver.cuModuleEnumerateFunctions(count, self.module))
        self.function = functions[0]
        self.symbol = checked(driver.cuFuncGetName(self.function)).decode()
        checked(
            driver.cuFuncSetAttribute(
                self.function,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                226432,
            )
        )
        self.parameters = []
        offset = 0
        for index, (name, category, size, alignment) in enumerate(PARAMETERS):
            offset = (offset + alignment - 1) // alignment * alignment
            actual = checked(driver.cuFuncGetParamInfo(self.function, index))
            if actual != (offset, size):
                raise RuntimeError(f"CuTe argument ABI mismatch for {name}: {actual}")
            self.parameters.append(
                dict(name=name, category=category, size=size, offset=offset)
            )
            offset += size

    def __del__(self):
        module = getattr(self, "module", None)
        if module is not None:
            with suppress(Exception):
                driver.cuModuleUnload(module)


class PreparedLaunch:
    """Fixed bindings and stream; storage contents remain live on each call."""

    def __init__(self, image, arguments, stream, owner):
        if set(arguments) != {p[0] for p in PARAMETERS}:
            raise ValueError("arguments do not match the unified CuTe ABI")
        self.image, self.owner, self.stream = image, owner, stream
        self.tensors = tuple(
            dict(
                (id(v), v) for v in arguments.values() if isinstance(v, torch.Tensor)
            ).values()
        )
        self.recorders = tuple(tensor.record_stream for tensor in self.tensors)
        self.storage = []
        pointers = []
        for name, category, size, alignment in PARAMETERS:
            value = arguments[name]
            if category == "tma":
                encoded = tensor_map(value, TENSOR_MAPS[name])
                # CUDA's grid-constant tensor-map ABI requires 64-byte alignment.
                raw = ctypes.create_string_buffer(size + alignment - 1)
                address = (
                    (ctypes.addressof(raw) + alignment - 1) // alignment * alignment
                )
                ctypes.memmove(address, int(encoded.getPtr()), size)
                self.storage.append(raw)
            else:
                if category == "pointer":
                    if not isinstance(value, torch.Tensor) or not value.is_cuda:
                        raise TypeError(f"{name} must be CUDA tensor storage")
                    raw = ctypes.c_uint64(value.data_ptr())
                else:
                    if type(value) is not int or not -(1 << 31) <= value < (1 << 32):
                        raise ValueError(f"{name} must fit the 32-bit scalar ABI")
                    raw = ctypes.c_uint32(value)
                self.storage.append(raw)
                address = ctypes.addressof(raw)
            pointers.append(address)
        self.packed = (ctypes.c_void_p * len(pointers))(*pointers)
        self.native = native_module()
        values = [
            int(image.function),
            ctypes.addressof(self.packed),
            int(stream.cuda_stream),
            152,
            1,
            1,
            512,
            1,
            1,
            226432,
            2,
            1,
            1,
            1,
        ]
        self.state = int(self.native["create"](*values))
        cluster = int(driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION)
        policy = int(
            driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE
        )
        spread = int(
            driver.CUclusterSchedulingPolicy.CU_CLUSTER_SCHEDULING_POLICY_SPREAD
        )
        observed = [
            int(self.native["inspect"](self.state, index)) for index in range(17)
        ]
        if observed != [*values, cluster, policy, spread]:
            self.close()
            raise RuntimeError(
                "native launch configuration did not preserve the prepared bindings"
            )
        self.run = self.native["run"]

    def __call__(self):
        if not self.state:
            raise RuntimeError("prepared launch is closed")
        try:
            self.run(self.state)
        finally:
            for record_stream in self.recorders:
                record_stream(self.stream)

    def close(self):
        state = getattr(self, "state", 0)
        if state:
            self.state = 0
            self.native["destroy"](state)

    def __del__(self):
        with suppress(Exception):
            self.close()
