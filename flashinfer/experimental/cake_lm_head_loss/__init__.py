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

# Experimental Cake backend: chunked large-vocabulary LM-head projection +
# log-sum-exp / cross-entropy / policy-gradient loss with a memory-bounded
# backward (every vocabulary-sized intermediate spans at most one token chunk)
# for training on SM100 / SM103 (flashinfer-ai/flashinfer#5680, tracker #4642).
# The public entry points are ``flashinfer.chunked_lm_head.chunked_lm_head_loss``
# and ``chunked_lm_head_logprob``; the autograd wrappers, host planning, JIT
# registration and the generated sources live in this package.
