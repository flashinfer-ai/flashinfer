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

"""SM120 backend of :func:`flashinfer.dsa_indexer.dsa_indexer_topk`: FP8 or
MXFP4 DSA indexer scores, bit-identical to DeepGEMM's MQA logits, and their
exact top-k. See README.md; tracking issue flashinfer-ai/flashinfer#7084."""

from ._api import (  # noqa: F401
    dsa_indexer_topk,
    dsa_indexer_topk_paged,
    ragged_scratch_bytes,
)
