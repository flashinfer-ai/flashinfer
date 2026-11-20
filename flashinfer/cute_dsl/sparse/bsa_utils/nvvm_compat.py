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

import inspect

from cutlass._mlir.dialects import nvvm

# Old nvidia-cutlass-dsl builds (CUDA 12.9, cutlass < 4.6) required the MLIR
# result type as an explicit first positional arg: nvvm.fmax(T.f32(), a, b)
# — 3 positional params (res, a, b).
# Newer builds (cutlass >= 4.6) infer it automatically: nvvm.fmax(a, b)
# — 2 positional params (a, b).
# We detect via inspect.signature rather than version numbers to handle all
# wheel/binding combinations correctly. The threshold of 2 distinguishes the
# two known ABIs: old API has 3 positional params, new API has 2.
NVVM_FMAX_REQUIRES_RESULT_TYPE: bool = (
    sum(
        1
        for p in inspect.signature(nvvm.fmax).parameters.values()
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
    )
    > 2
)
