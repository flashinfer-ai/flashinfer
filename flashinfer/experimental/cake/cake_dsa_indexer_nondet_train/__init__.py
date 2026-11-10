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

# Cake backend of flashinfer.cake_dsa_indexer_nondet_train: fused DSA indexer scoring + exact top-k selection for
# training whose selected set repeats per row while its slot order may vary (flashinfer-ai/flashinfer#8946, point 4;
# the operator of flashinfer-ai/flashinfer#5676; target geometry: 32 indexer heads, head dimension 128, top-k 2048) on
# SM100 / SM103 / SM107.  It runs the deterministic package's BF16 scan programs and its plain split merge and no
# ordering pass, and raises PyTorch's alert under ``torch.use_deterministic_algorithms(True)`` (a ``UserWarning``
# with ``warn_only=True``) before an eager call's first allocation and before the first launch of every prepared
# run.  The public entry points are
# ``flashinfer.cake_dsa_indexer_nondet_train.cake_dsa_indexer_nondet_train_topk`` and
# ``cake_dsa_indexer_nondet_train_topk_workspace_size``; the host binding, workspace policy, JIT registration and
# the generated kernel sources live in this package.
