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

# Experimental Cake backend: the Kimi-K3 serialized FP8_PB_WO projection GEMMs
# (BF16 activations -> per-token 1x128 E4M3 / UE8M0 quantization -> block-scaled
# tcgen05 GEMM against the serialized ModelOpt weight -> BF16) for SM100 / SM103
# (flashinfer-ai/flashinfer#4568, tracker #4254).  The public entry points are
# ``flashinfer.gemm.prepare_kimi_k3_fp8_projection_weights``,
# ``flashinfer.gemm.prepare_kimi_k3_fp8_projection`` and
# ``flashinfer.gemm.kimi_k3_fp8_projection``; the weight preparation, the
# measured dispatch, the launch binding, the JIT registration and the
# generated sources live in this package.
