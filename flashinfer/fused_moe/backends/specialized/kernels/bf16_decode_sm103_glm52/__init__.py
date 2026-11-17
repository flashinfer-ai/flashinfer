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

"""Specialized MoE kernel ``bf16_decode_sm103_glm52``: BF16 MoE decode of the
GLM-5.2 MTP layer on SM103 (256 experts, DeepSeekV3 top-8 with one group from
fp32 logits and an fp32 bias, hidden 6144, intermediate 256, SwiGLU).

``kernel.py`` holds the guards, the per-device build, the launch and the
diagnostics (``stats()``); ``workloads.json`` is the token-count allowlist. The
CUDA kernels live in ``csrc/fused_moe/bf16_moe_decode_sm103.cu`` (JIT module
``flashinfer/jit/bf16_moe_decode_sm103.py``) and are loaded on the first call
admitted.
"""

from .kernel import Bf16DecodeSm103Glm52Kernel as KERNEL
from .kernel import stats

__all__ = ["KERNEL", "stats"]
