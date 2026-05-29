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

"""FlashInfer flashinfer solution for fp4_quantize."""

from flashinfer.quantization.fp4_quantization import fp4_quantize as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "fp4_quantize"
api = "flashinfer.quantization.fp4_quantization.fp4_quantize"
backend = "flashinfer"
inputs = ("input", "global_scale", "sf_vec_size")
outputs = ("quantized", "scales")
api_kwargs = {
    "input": "input",
    "global_scale": "global_scale",
    "sf_vec_size": "sf_vec_size",
}
constants = {"K": 4096}


def run(input, global_scale, sf_vec_size):
    with solution_autotune(
        definition,
        backend,
        input,
        global_scale,
        sf_vec_size,
    ):
        result = _api(
            input=input,
            global_scale=global_scale,
            sf_vec_size=sf_vec_size,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "fp4_quantize" + " returned None without mutating declared outputs"
        )
