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

# Experimental generated-program backend: the Kimi-K3 fused MoE router
# (896 experts, top-16, sigmoid gate with selection bias) that writes the
# expert-aligned route plan in the same launch, for SM100 / SM103.  The public
# entry points are ``flashinfer.fused_moe.kimi_k3_fused_router`` and
# ``flashinfer.fused_moe.prepare_kimi_k3_fused_router``; the per-shape
# dispatch, launch geometry and JIT registration live in this package.
