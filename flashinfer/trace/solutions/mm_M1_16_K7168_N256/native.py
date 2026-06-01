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

"""FlashInfer flashinfer solution for mm_M1_16_K7168_N256."""

from flashinfer.gemm.routergemm import mm_M1_16_K7168_N256 as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "mm_M1_16_K7168_N256"
api = "flashinfer.gemm.routergemm.mm_M1_16_K7168_N256"
backend = "flashinfer"
inputs = ("mat_a", "mat_b", "out")
outputs = ("out",)
api_kwargs = {"mat_a": "mat_a", "mat_b": "mat_b", "out": "out"}


def run(mat_a, mat_b, out):
    with solution_autotune(
        definition,
        backend,
        mat_a,
        mat_b,
        out,
    ):
        result = _api(
            mat_a=mat_a,
            mat_b=mat_b,
            out=out,
        )
        if result is not None:
            return result
        return out
