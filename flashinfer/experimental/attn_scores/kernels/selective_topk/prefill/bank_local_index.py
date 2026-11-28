# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Decode compact indices from eight interleaved 32-key candidate banks."""

import cutlass
import cutlass.cute as cute
from .selection import candidate_index


@cute.jit
def bank_index(
    entry, packed: cutlass.Constexpr, bank, local: cutlass.Constexpr = False
):
    index = candidate_index(entry, packed)
    if cutlass.const_expr(local):
        index = ((index >> 5) << 8) | (bank << 5) | (index & 31)
    return index
