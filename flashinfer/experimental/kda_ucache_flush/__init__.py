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

# Experimental CuTe DSL backend behind
# ``flashinfer.kda_decode.kda_delta_rule_mtp_ucache_flush``: the KDA (Kimi K3)
# ReplaySSM verify + ring append + per-request state flush kernel. Contract
# validation, compilation, caching and the kernel all live in
# ``kda_decode_bf16_wy_ucache_flush``.

from .kda_decode_bf16_wy_ucache_flush import (
    MAX_LOG_DECAY_SPAN,
    RING_SLOTS,
    W_RING,
    kda_delta_rule_mtp_ucache_flush,
)

__all__ = [
    "MAX_LOG_DECAY_SPAN",
    "RING_SLOTS",
    "W_RING",
    "kda_delta_rule_mtp_ucache_flush",
]
