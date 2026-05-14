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

"""FlashInfer cutlass solution for gemm_fp4."""

from flashinfer.gemm.gemm_base import mm_fp4 as _api

definition = "gemm_fp4"
api = "flashinfer.gemm.gemm_base.mm_fp4"
backend = "cutlass"
inputs = ("A", "B", "a_descale", "b_descale", "block_size")
outputs = ("C",)
api_kwargs = {
    "a": "A",
    "b": "B",
    "a_descale": "a_descale",
    "b_descale": "b_descale",
    "block_size": "block_size",
}
constants = {"N": 2048, "K": 7168, "block_size": 16}


def run(A, B, a_descale, b_descale, block_size):
    result = _api(
        a=A,
        b=B,
        a_descale=a_descale,
        b_descale=b_descale,
        block_size=block_size,
        backend=backend,
    )
    if result is not None:
        return result
    raise RuntimeError("gemm_fp4" + " returned None without mutating declared outputs")
