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

# Experimental Cake backend: the Kimi-K3 TP12 fused LatentMoE communication
# tail (Lamport all-reduce + KimiRMSNorm, per-rank up-projection slice, column
# reduce-scatter + add + all-gather) for twelve-rank SM100 / SM103 multi-node
# NVLink domains (flashinfer-ai/flashinfer#4542, tracker #4254).  The public
# entry points are ``flashinfer.kimi_k3_tp12_tail``
# (``create_kimi_k3_tp12_tail_workspace``, ``prepare_kimi_k3_tp12_tail``,
# ``kimi_k3_tp12_tail``); the host runtime, JIT registration and the generated
# sources live in this package.
