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

# SM103 (Blackwell Ultra) mega-kernel backends.
#
# ``bf16_nvfp4_bf16_cake`` registers ``kernel_name="sm103_bf16_nvfp4_bf16_cake"``
# (BF16 activations x NVFP4 expert weights over the Cake fused program). The
# experimental W8A8 backend lives under
# ``flashinfer.experimental.cake_w8a8_megamoe_ep16``; importing its
# ``cake_backend`` module registers ``kernel_name="sm103_mxfp8_mxfp8_bf16_cake"``.
# Both are lightweight at import time (their JIT loaders and tvm-ffi are
# imported inside the functions that use them).
#
# ``mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl`` registers
# ``kernel_name="sm103_mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl"`` (the
# NVLinkOneSided MegaMoE); it also runs on SM100 devices. Importing it imports
# neither its torch-only host frontend nor the CuTe DSL. The backend imports the
# frontend when it checks a ``knobs`` dictionary at creation or resolves an
# engine configuration (``validate_init`` at layer construction), and the CuTe
# DSL engine when ``validate_init`` builds that configuration on an SM100 or
# SM103 device.

from flashinfer.experimental.cake_w8a8_megamoe_ep16 import (
    cake_backend as mxfp8_mxfp8_bf16_cake,
)

from . import bf16_nvfp4_bf16_cake, mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl

__all__ = [
    "bf16_nvfp4_bf16_cake",
    "mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl",
    "mxfp8_mxfp8_bf16_cake",
]
