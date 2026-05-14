# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FlashInfer flashinfer solution for mxfp8_quantize."""

from flashinfer.quantization.fp8_quantization import mxfp8_quantize as _api

definition = "mxfp8_quantize"
api = "flashinfer.quantization.fp8_quantization.mxfp8_quantize"
backend = "flashinfer"
inputs = ("input",)
outputs = ("quantized", "scales")
api_kwargs = {"input": "input"}
constants = {"K": 4096}


def run(input):
    result = _api(
        input=input,
    )
    if result is not None:
        return result
    raise RuntimeError(
        "mxfp8_quantize" + " returned None without mutating declared outputs"
    )
