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

CuTe-DSL Quantization Kernels (EXPERIMENTAL)
============================================

.. warning::
    This subpackage is **experimental** and not part of the stable FlashInfer API.
    The interfaces, implementations, and behaviors may change or be removed
    in future versions without notice. Use at your own risk for production workloads.

This subpackage contains high-performance CuTe-DSL implementations of
quantization kernels for MXFP4 and MXFP8 formats. These kernels require
SM100+ (Blackwell) GPUs and the nvidia-cutlass-dsl package.
"""

from .mxfp4_quantize import (
    MXFP4QuantizeSwizzledKernel,
    mxfp4_quantize_cute_dsl,
)
from .mxfp8_quantize import (
    MXFP8QuantizeLinearKernel,
    MXFP8QuantizeSwizzledKernel,
    mxfp8_quantize_cute_dsl,
)

__all__ = [
    "MXFP4QuantizeSwizzledKernel",
    "mxfp4_quantize_cute_dsl",
    "MXFP8QuantizeLinearKernel",
    "MXFP8QuantizeSwizzledKernel",
    "mxfp8_quantize_cute_dsl",
]
