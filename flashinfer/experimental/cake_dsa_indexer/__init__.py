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

# Experimental Cake backend: fused DSA indexer scoring + deterministic exact
# top-k selection for training (flashinfer-ai/flashinfer#5676; GLM-5.2
# geometry: 32 indexer heads, head dimension 128, top-k 2048) on SM100 / SM103 /
# SM107.  The public entry point is ``flashinfer.dsa_indexer.dsa_indexer_topk``;
# the host binding, workspace policy, JIT registration and the generated kernel
# sources live in this package.
