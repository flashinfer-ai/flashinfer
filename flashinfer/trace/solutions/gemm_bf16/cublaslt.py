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

"""FlashInfer cublaslt solution for gemm_bf16."""

from flashinfer.gemm.gemm_base import mm_bf16 as _api
from flashinfer.trace.solutions._helpers import solution_autotune

definition = "gemm_bf16"
api = "flashinfer.gemm.gemm_base.mm_bf16"
backend = "cublaslt"
inputs = ("A", "B")
outputs = ("C",)
api_kwargs = {"a": "A", "b": "B"}
constants = {"N": 256, "K": 7168}


def run(A, B):
    with solution_autotune(
        definition,
        backend,
        A,
        B,
    ):
        result = _api(
            a=A,
            b=B,
            backend=backend,
        )
        if result is not None:
            return result
        raise RuntimeError(
            "gemm_bf16" + " returned None without mutating declared outputs"
        )
