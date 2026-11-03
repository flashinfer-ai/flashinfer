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

# Experimental Cake backend: the Kimi-K3 Stable LatentMoE front (router
# logits + latent down-projection + shared-expert SiTU) and tail (partial
# sum + KimiRMSNorm + latent up-projection + shared down-projection)
# projections for SM100 / SM103 (flashinfer-ai/flashinfer#4568, tracker
# #4254).  The public entry points are ``flashinfer.kimi_k3_latent_moe``
# (``kimi_k3_latent_moe_front`` / ``kimi_k3_latent_moe_tail`` and their
# ``prepare_*`` forms); the host planner, JIT registration and the generated
# sources live in this package.
