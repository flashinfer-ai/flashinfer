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

# Experimental backend of
# flashinfer.attn_scores.fp8_ue8m0_paged_mqa_logits_decode: Cake-generated
# CuTe DSL programs (csrc/) and their launch adapter.

from .backend import (
    NEXT_NS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    check_inputs,
    paged_mqa_logits,
)

__all__ = [
    "NEXT_NS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "check_inputs",
    "paged_mqa_logits",
]
