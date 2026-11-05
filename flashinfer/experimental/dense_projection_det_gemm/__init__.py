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

# Experimental Cake backend: the GLM-5.2 dense projection GEMMs for training
# (BF16 operands, FP32 accumulation, BF16 or FP32 output, strided views, ragged
# token counts, batched head projections) and the FP32 router-gate GEMM through
# split-BF16x3 tensor-core emulation, on SM100 / SM103 / SM107
# (flashinfer-ai/flashinfer#5677).  Host planning, argument-plan binding, the
# prepared allocation-free launches, the eager entry points, JIT registration
# and the generated sources live in this package; there is no core entry point
# yet (importing this package is the explicit opt-in).
