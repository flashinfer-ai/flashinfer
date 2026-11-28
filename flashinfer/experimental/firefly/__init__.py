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

# Experimental backend "firefly": the per-token NVFP4 routed MoE for SM100 /
# SM103 (datacenter Blackwell) written in CuTe DSL. The stable entry point is
# ``MoELayer`` with ``FireflyNvfp4Config`` (flashinfer.fused_moe), whose runner
# imports this package on first use.
#
# - ``moe.CuteMoE(E, topk, H, I, max_tokens)``: the MoE for every batch size up
#   to max_tokens (one workspace); called with the per-token NVFP4 input
#   (q [m, H / 2], sf [m, H / 16], gs [m]), the weights and the pre-routed
#   topk_ids / topk_weights, it returns the BF16 output [m, H].
# - ``moe.select_schedule``: the per-batch-size kernel choice.
# - ``weights.prepare_view``: canonical BF16 experts -> the weight view.
#
# Hand-written PTX and the reasons for it: HANDWRITTEN_PTX.md.
#
# Importing this package imports none of its modules.
