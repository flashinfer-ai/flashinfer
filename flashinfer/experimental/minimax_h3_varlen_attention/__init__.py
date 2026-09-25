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

# Experimental Cake backend: MiniMax-H3 packed-varlen noncausal attention over
# THD ``[T, H, 128]`` BF16 tensors for SM100 / SM103
# (flashinfer-ai/flashinfer#4532, tracker #4254).  Two generated program
# families live here: the BF16 kernel (``minimax_h3_varlen_attention``) and
# the NVFP4-QK kernels with NVFP4 or FP8 PV plus their in-pipeline quantizers
# (``minimax_h3_varlen_nvfp4_attention``).  The public entry points are
# ``flashinfer.prefill.minimax_h3_varlen_attention`` and
# ``flashinfer.prefill.minimax_h3_varlen_nvfp4_attention``; host segment
# planning, JIT registration and the generated sources live in this package.
