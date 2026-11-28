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

"""Softmax row-max helpers shared by the decode and MLA kernels.

The running maximum starts at ``neg_max_f32()``. Warps that share a row publish
their maxima to shared memory with an unsigned atomic max over an order-preserving
encoding of float32.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass.experimental import primitives as prims

# Masked scores start at negative Float32 max.
NEG_FLT_MAX = -3.4028235e38


@cute.jit
def float_to_u32_for_atomic_max(val: Float32) -> Uint32:
    """Encode a float so unsigned atomic max preserves float ordering."""
    # Encode signed floats so unsigned atomic max has the same ordering.
    bits = prims.mov_b32(val, target_type=Int32)
    mask = (bits >> Int32(31)) | Int32(0x80000000)
    encoded = bits ^ mask
    return prims.mov_b32(encoded, target_type=Uint32)


@cute.jit
def init_softmax_scratch_u32(
    scratch: cutlass.Array, warp_grp_thread_idx: Int32, num_entries: int
) -> None:
    """Initialize encoded softmax scratch maxima across the warp group."""
    encoded = float_to_u32_for_atomic_max(neg_max_f32())
    # Initialize scratch with a warp-group-thread-index-stepped loop instead
    # of a simple `thread < 8` predicate. This avoids the widened store pattern
    # that the TS helper currently lowers into.
    for scratch_idx in cutlass.range(
        warp_grp_thread_idx, Int32(num_entries), Int32(128), unroll=1
    ):
        scratch[scratch_idx] = encoded


def neg_max_f32() -> Float32:
    """Return the negative sentinel used for running softmax maxima."""
    return Float32(NEG_FLT_MAX)


@cute.jit
def smem_atomic_max_u32(ptr: cute.Pointer, val: Uint32) -> None:
    """Atomically update a CTA-scope SMEM max encoded as unsigned int."""
    # Softmax row max is reduced through CTA SMEM using unsigned atomics over
    # the encoded float representation. CTA scope is sufficient because all
    # participating softmax warps are inside one CTA.
    prims.atomicrmw(
        prims.AtomicOp.MAX,
        ptr,
        val,
        syncscope=prims.MemScope.CTA,
        space=prims.SharedSpace.shared_cta,
    )


@cute.jit
def u32_to_float_for_atomic_max(val: Uint32) -> Float32:
    """Decode the unsigned atomic-max representation back to float."""
    # Decode the monotonic unsigned representation back to float.
    encoded = prims.mov_b32(val, target_type=Int32)
    mask = (~(encoded >> Int32(31))) | Int32(0x80000000)
    bits = encoded ^ mask
    return prims.mov_b32(bits, target_type=Float32)
