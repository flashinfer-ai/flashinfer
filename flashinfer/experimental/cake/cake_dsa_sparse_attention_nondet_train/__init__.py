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

# Backend package of ``flashinfer.cake_dsa_sparse_attention_nondet_train`` (issue #8928): native 64-query-head
# DeepSeek Sparse Attention (top-k sparse MLA, absorbed queries) training forward and backward on SM100 / SM103 /
# SM107 (flashinfer-ai/flashinfer#5657, tracker #4254), the non-deterministic package: the backward adds the dK / dV
# gradients with unordered FP32 atomics and raises PyTorch's alert under
# ``torch.use_deterministic_algorithms(True)`` (a ``UserWarning`` with ``warn_only=True``) before its first backward kernel launch.  The
# stable entry points are ``cake_dsa_sparse_attention_nondet``, ``cake_dsa_sparse_attention_nondet_varlen``,
# ``cake_dsa_sparse_attention_nondet_nope`` and ``cake_dsa_sparse_attention_nondet_nope_varlen`` of
# ``flashinfer.cake_dsa_sparse_attention_nondet_train``; the autograd Functions, host planning, JIT registration and
# the generated sources live in this package.
