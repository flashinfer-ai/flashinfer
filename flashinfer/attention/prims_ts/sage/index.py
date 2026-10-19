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

"""Device-side index math and loads over the Sage scale buffers.

The helpers turn coordinates into words of the flat scale tensors
(``config.flat_scale_slot``) and of the prepared buffer
(``layout._SageScaleLayout``), and load single FP32 scales. They hold no
resource or pipeline state; each attention kernel moves the words it reads
with its own resources.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64

from .config import flat_scale_slot


@cute.jit
def _load_scale(scale_addr: Int64, index: Int32) -> Float32:
    """Return one FP32 scale loaded from a global memory base address."""
    value_ptr = cutlass.inttoptr(
        scale_addr + Int64(index) * Int64(4),
        mem_space=1,
        dtype=Float32,
    )
    return Float32(value_ptr.load(count=1, alignment=4)[0])


@cute.jit
def load_flat_scale(
    scale_addr: Int64,
    head_stride: Int32,
    *,
    head_idx: Int32,
    batch_idx: Int32,
    seq_len: Int32,
    token_idx: Int32,
    log2_block: Int32,
) -> Float32:
    """Return one head's scale of one token from a flat-layout scale array.

    A token past ``seq_len`` takes the last valid slot.
    """
    token_idx = cute.math.min(token_idx, seq_len - Int32(1))
    slot = flat_scale_slot(batch_idx, token_idx, seq_len, log2_block)
    return _load_scale(scale_addr, head_idx * head_stride + slot)
