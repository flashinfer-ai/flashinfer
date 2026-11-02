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

# Experimental Cake backend, non-deterministic form:
# chunked large-vocabulary LM-head projection + log-sum-exp / cross-entropy /
# policy-gradient loss with a memory-bounded backward for training on
# SM100 / SM103, whose dX GEMM combines its K-slice partials in an unordered
# FP32 reduction.  The public entry points are
# ``flashinfer.chunked_lm_head.chunked_lm_head_loss``, ``chunked_lm_head_logprob``,
# ``chunked_lm_head_preload`` and the candidate-support entries
# ``chunked_lm_head_support_logprob`` / ``chunked_lm_head_support_kl`` with
# ``deterministic=False`` (their default ``deterministic=True`` runs the
# deterministic package ``flashinfer.experimental.cake_lm_head_det_loss``); the
# autograd wrappers, host planning, JIT registration and the generated sources
# live in this package.
