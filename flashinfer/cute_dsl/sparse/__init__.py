# Copyright (c) 2025 by FlashInfer team.
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

__all__ = [
    "bsa_attn_fwd",
    "bsa_attn_blk64_fwd",
]


def __getattr__(name):
    if name == "bsa_attn_fwd":
        from .bsa_attn_sm100_blk128 import bsa_attn_fwd

        return bsa_attn_fwd
    if name == "bsa_attn_blk64_fwd":
        from .bsa_attn_sm100_blk64 import bsa_attn_blk64_fwd

        return bsa_attn_blk64_fwd
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
