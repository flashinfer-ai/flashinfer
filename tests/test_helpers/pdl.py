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
"""

import ctypes
import functools

import torch

from flashinfer.cuda_utils import checkCudaErrors, driver, nvrtc

# Triggers the dependent launch first and writes `value` only after spinning, so a
# PDL consumer that reads `dst` before `griddepcontrol.wait` sees the old value.
_LATE_STORE_SOURCE = r"""
extern "C" __global__ void late_store(float* dst, float value, long long spin_cycles) {
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  long long start = clock64();
  while (clock64() - start < spin_cycles) {
  }
  asm volatile("" ::: "memory");
  *dst = value;
}
"""
# Long enough for the consumer kernel to get past its prologue before the store.
_LATE_STORE_SPIN_CYCLES = 50_000_000


@functools.cache
def _late_store_kernel(device_index: int):
    major, minor = torch.cuda.get_device_capability(device_index)
    program = checkCudaErrors(
        nvrtc.nvrtcCreateProgram(
            _LATE_STORE_SOURCE.encode(), b"late_store.cu", 0, [], []
        )
    )
    arch = f"--gpu-architecture=sm_{major}{minor}".encode()
    checkCudaErrors(nvrtc.nvrtcCompileProgram(program, 1, [arch]))
    cubin = bytes(checkCudaErrors(nvrtc.nvrtcGetCUBINSize(program)))
    checkCudaErrors(nvrtc.nvrtcGetCUBIN(program, cubin))
    module = checkCudaErrors(driver.cuModuleLoadData(cubin))
    return checkCudaErrors(driver.cuModuleGetFunction(module, b"late_store"))


def late_store(dst: torch.Tensor, value: float) -> None:
    kernel = _late_store_kernel(dst.device.index)
    args = (
        (dst.data_ptr(), value, _LATE_STORE_SPIN_CYCLES),
        (ctypes.c_void_p, ctypes.c_float, ctypes.c_longlong),
    )
    stream = torch.cuda.current_stream(dst.device).cuda_stream
    checkCudaErrors(driver.cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, stream, args, 0))
