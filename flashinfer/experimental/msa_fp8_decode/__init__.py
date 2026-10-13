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

# Experimental backend: packed-FP8 MiniMax Sparse Attention (MSA) speculative
# decode for SM100 / SM103, served by the existing TRT-LLM block-sparse
# attention kernel.  The public entry point is
# ``flashinfer.msa_ops.msa_packed_fp8_sparse_decode``; the decode-metadata
# preparation kernel and its JIT registration live in this package.
