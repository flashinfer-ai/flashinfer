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

# Experimental Cake backend of ``flashinfer.cake_lm_head_nondet_train``:
# chunked large-vocabulary LM-head projection + log-sum-exp / cross-entropy /
# policy-gradient loss with a memory-bounded backward for training on
# SM100 / SM103 / SM107, whose dX GEMM stores its K-slice partials into an FP32 slab
# that a slab sum adds in slice order.  The public entry points are the module's
# ``cake_lm_head_nondet_train_loss``, ``cake_lm_head_nondet_train_logprob``,
# ``cake_lm_head_nondet_train_preload`` and the candidate-support entries
# ``cake_lm_head_nondet_train_support_logprob`` /
# ``cake_lm_head_nondet_train_support_kl``; none takes a ``deterministic``
# keyword (the deterministic counterpart is ``flashinfer.cake_lm_head_det_train``).
# The autograd wrappers, host planning, JIT registration and the generated
# sources live in this package.
