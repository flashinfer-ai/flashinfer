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

"""Transformed NVFP4 K/V resource backed by tensor memory."""

import cutlass
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL

from ......cute_dsl.availability import is_rubin_cute_dsl_available


def _has_packed_e4m3_e2m1_mul() -> bool:
    """Whether the compile target has the packed E4M3 x E2M1 multiply.

    PTX 9.4 (CUDA 13.4) adds it for SM100 and SM103; ptxas rejects it for
    SM107, which takes the f16x2 path instead.
    """
    if not cutlass.target_version(min_version="13.4"):
        return False
    if not is_rubin_cute_dsl_available():
        return True
    arch = BaseDSL._get_dsl().get_arch_enum()
    return not (Arch.sm_107 <= arch <= Arch.sm_107f)


# The shared-memory helpers below take 32-bit shared-window byte addresses.
# Transform loops compute per-token row addresses arithmetically (fragment
# slots, token rows, swizzled segments), so raw vector loads and stores avoid
# building a CuTe tensor view per row inside the rolled loops.
