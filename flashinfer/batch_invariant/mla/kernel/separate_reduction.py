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

"""Common arithmetic for standalone split-KV GMEM reducers.

Every producer using this interface publishes the same mathematical state:

* one FP32 log2-LSE value per split and output row; and
* one normalized 16-bit partial-O vector for that split and row.

Reducer policies still own row addressing, active-split discovery, cluster
topology, PDL ordering, and final-output stores. Those details differ between
FMHA, MLA 1CTA, and MLA 2CTA and are deliberately kept out of this module.
"""

import cutlass.cute as cute
from cutlass import Float32

from .mla_decode.helpers.constants import LOG2_E


@cute.jit
def finalize_log2_sum_exp(frame: Float32, exp_sum: Float32) -> Float32:
    """Return ``frame + log2(exp_sum)`` or the neutral ``-inf`` LSE."""

    has_mass = exp_sum == exp_sum and exp_sum != Float32(0.0)
    return (
        frame + cute.math.log2(exp_sum, fastmath=True)
        if has_mass
        else Float32(-Float32.inf)
    )


@cute.jit
def normalized_lse_weight(partial_lse: Float32, global_lse: Float32) -> Float32:
    """Return the weight of one normalized partial in the global output."""

    has_mass = global_lse != Float32(-Float32.inf)
    return (
        Float32(
            cute.math.exp2(
                partial_lse - global_lse,
                fastmath=True,
            )
        )
        if has_mass
        else Float32(0.0)
    )


@cute.jit
def attention_sink_scale(log2_lse, sink):
    """Denominator-only sink mass; keep the KV LSE independent of the sink."""
    scale = Float32(0)
    if log2_lse != Float32(-Float32.inf) and sink != Float32(Float32.inf):
        delta = sink * Float32(LOG2_E) - log2_lse
        z = cute.math.exp2(-cute.math.abs(delta), fastmath=True)
        inverse = cute.math.rcp(Float32(1) + z, approx=True)
        scale = z * inverse if delta > Float32(0) else inverse
    return scale
