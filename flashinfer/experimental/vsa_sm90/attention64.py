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

import cutlass
import cutlass.cute as cute
from cutlass import Float32


@cute.jit
def softmax_update(
    s: cute.Tensor,
    mx: cute.Tensor,
    den: cute.Tensor,
    logscale: Float32,
    first: cutlass.Constexpr,
):
    factors = cute.make_rmem_tensor((2,), Float32)
    for r in cutlass.range_constexpr(2):
        prev = mx[r]
        maximum = prev
        for col in cutlass.range_constexpr(16):
            maximum = cute.arch.fmax(maximum, s[r, col], nan=True, ftz=False)
        maximum = cute.arch.warp_reduction_max(maximum, threads_in_group=4)
        alpha = cute.math.exp2((prev - maximum) * logscale, fastmath=True)
        if cutlass.const_expr(first):
            alpha = Float32(0.0)
        factors[r] = alpha
        mx[r] = maximum
        scaled_max = maximum * logscale
        for col in cutlass.range_constexpr(16):
            prob = cute.math.exp2(
                cute.math.fma(s[r, col], logscale, -scaled_max), fastmath=True
            )
            s[r, col] = prob
        den[r] = den[r] * alpha
        for col in cutlass.range_constexpr(16):
            den[r] = den[r] + s[r, col]
    return factors
