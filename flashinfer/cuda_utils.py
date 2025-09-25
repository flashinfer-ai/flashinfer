"""
Copyright (c) 2025 by FlashInfer team.

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

from flashinfer.utils import has_cuda_cudart

try:
    # Check if cuda.cudart module is available and import accordingly
    if has_cuda_cudart():
        # cuda-python < 12.9 (has cuda.cudart)
        import cuda.bindings.driver as driver
        import cuda.bindings.runtime as runtime
        import cuda.cudart as cudart
        import cuda.nvrtc as nvrtc
    else:
        # cuda-python >= 13.0 (no cuda.cudart, use runtime as cudart)
        from cuda.bindings import driver, nvrtc, runtime

        cudart = runtime  # Alias runtime as cudart for compatibility
except ImportError as e:
    raise ImportError(
        "Could not import the 'cuda' module. "
        "Please install cuda-python that matches your CUDA version."
    ) from e


def _cudaGetErrorEnum(error):
    if isinstance(error, driver.CUresult):
        err, name = driver.cuGetErrorName(error)
        return name if err == driver.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, runtime.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError(f"Unknown error type: {error}")


def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError(
            f"CUDA error code={result[0].value}({_cudaGetErrorEnum(result[0])})"
        )
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
