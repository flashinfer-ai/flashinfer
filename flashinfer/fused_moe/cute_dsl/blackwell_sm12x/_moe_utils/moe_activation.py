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
"""Gated activation, the epilogue step every fc1_act arm runs on its two accumulators."""

import cutlass
import cutlass.cute as cute

from .....tllm_enums import DEFAULT_SITU_BETA, DEFAULT_SITU_LINEAR_BETA, ActivationType


class SwiGLUActivation:
    def __init__(self, fastmath=False, swiglu_limit=None):
        self.fastmath = fastmath
        self.swiglu_limit = swiglu_limit

    @cute.jit
    def __call__(self, acc_u, acc_g):
        for i in cutlass.range_constexpr(cute.size(acc_u)):
            g, u = acc_g[i], acc_u[i]
            if cutlass.const_expr(self.swiglu_limit is not None):
                g = cute.math.min(g, self.swiglu_limit)
                u = cute.math.max(
                    -self.swiglu_limit, cute.math.min(u, self.swiglu_limit)
                )
            sigmoid = cute.arch.rcp_approx(
                1.0 + cute.math.exp(-g, fastmath=self.fastmath)
            )
            acc_u[i] = g * sigmoid * u
        return acc_u


class SiTUActivation:
    def __init__(
        self,
        fastmath=False,
        beta=DEFAULT_SITU_BETA,
        linear_beta=DEFAULT_SITU_LINEAR_BETA,
    ):
        self.fastmath = fastmath
        self.beta = beta
        self.linear_beta = linear_beta

    @cute.jit
    def __call__(self, acc_u, acc_g):
        inv_beta = 1.0 / self.beta
        inv_linear_beta = 1.0 / self.linear_beta
        for i in cutlass.range_constexpr(cute.size(acc_u)):
            g, u = acc_g[i], acc_u[i]
            soft_g = self.beta * cute.math.tanh(g * inv_beta, approx=False)
            soft_u = self.linear_beta * cute.math.tanh(
                u * inv_linear_beta, approx=False
            )
            sigmoid = cute.arch.rcp_approx(
                1.0 + cute.math.exp(-g, fastmath=self.fastmath)
            )
            acc_u[i] = soft_g * sigmoid * soft_u
        return acc_u


def make_gated_activation(
    activation,
    fastmath=False,
    swiglu_limit=None,
    situ_beta=DEFAULT_SITU_BETA,
    situ_linear_beta=DEFAULT_SITU_LINEAR_BETA,
):
    if activation is ActivationType.Swiglu:
        return SwiGLUActivation(fastmath, swiglu_limit)
    if activation is ActivationType.Situ:
        return SiTUActivation(fastmath, situ_beta, situ_linear_beta)
    raise ValueError(f"unsupported gated activation {activation!r}")
