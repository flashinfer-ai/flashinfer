# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Saturating softplus shared by the GDN decode kernels."""

import cutlass
import cutlass.cute as cute


@cute.jit
def softplus_fp32(
    x: cutlass.Float32,
    softplus_beta: cutlass.Constexpr[float],
    softplus_threshold: cutlass.Constexpr[float],
) -> cutlass.Float32:
    """Branch before exp: a masked 0 * exp(large x) still produces NaN."""
    beta_x = softplus_beta * x
    y = x
    if beta_x <= softplus_threshold:
        y = (cutlass.Float32(1.0) / softplus_beta) * cute.log(
            cutlass.Float32(1.0) + cute.exp(beta_x, fastmath=True), fastmath=True
        )
    return y
