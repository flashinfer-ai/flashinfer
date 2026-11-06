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

"""SM120 MXFP4 x MXFP8 nosync MegaMoE backend: one persistent kernel per layer
(routing exchange, dispatch, FC1 + SwiGLU, FC2, combine) for up to 128 tokens per
rank over PCIe peer-to-peer, or NCCL GIN for the pairs the transport assigns to
the NIC. Deployments add it to their configured backend with
``MoEEpLayer(..., nosync=NosyncOptions())``, which runs it for small calls only.

"nosync": the kernel has no cross-rank barrier. Each producer writes an
epoch-tagged flag into the consumer's memory after its data (release), and the
consumer polls its own memory (acquire); over GIN the consumer counts signals
per source instead. No remote atomics and no host-side synchronization."""

from .backend import Sm120Mxfp4Mxfp8NosyncMegaKernelBackend
from .config import Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig
from .weights import NosyncMegaWeights, preprocess_mega_weights

__all__ = [
    "Sm120Mxfp4Mxfp8NosyncMegaKernelBackend",
    "Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig",
    "NosyncMegaWeights",
    "preprocess_mega_weights",
]
