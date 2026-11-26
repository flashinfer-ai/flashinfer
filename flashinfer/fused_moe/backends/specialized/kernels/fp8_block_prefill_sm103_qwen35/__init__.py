# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Specialized MoE kernel ``fp8_block_prefill_sm103_qwen35``: FP8 block-scale
MoE prefill of Qwen3.5-397B-A17B-FP8 on SM103 (512 experts, top-10
RenormalizeNaive from bf16 logits, hidden 4096, intermediate 1024, SwiGLU).

``kernel.py`` holds the guards, the per-device build, the launch and the
diagnostics (``stats()``); ``workloads.json`` is the token-count allowlist. The
CuTe DSL programs live in
``flashinfer/fused_moe/cute_dsl/blackwell/fp8_block_scale_moe_prefill.py`` and
are imported on the first call admitted; device imports stay deferred.
"""

from .kernel import Fp8BlockPrefillSm103Qwen35Kernel as KERNEL
from .kernel import stats

__all__ = ["KERNEL", "stats"]
