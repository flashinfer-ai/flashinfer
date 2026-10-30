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

# Experimental Cake backend of FlashInfer's per-token NVFP4 path for SM100 /
# SM103: the per-token activation quantizer behind
# ``nvfp4_quantize(..., per_token_activation=True, backend="cake")`` and the
# per-token-alpha block-scaled GEMM behind ``mm_fp4(..., backend="cake")``.
# The host dispatch, the launch binding, the JIT registration and the
# generated sources live in this package; the stable entry points in
# ``flashinfer.quantization`` and ``flashinfer.gemm`` hand off here.
#
# This module stays import-light on purpose (no kernel or JIT imports): core
# imports ``.support`` for the ``@experimental_backend`` checker at module
# level and reaches ``.cake_backend`` through deferred imports only.
