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

import functools

import cutlass
import cutlass.cute as cute

from .....tllm_enums import ActivationType


@cute.jit
def _sigmoid(x, fastmath: cutlass.Constexpr):
    return cute.arch.rcp_approx(1.0 + cute.math.exp(-x, fastmath=fastmath))


@cute.jit
def apply_silu(acc_u, acc_g, fastmath: cutlass.Constexpr):
    for i in cutlass.range_constexpr(cute.size(acc_u)):
        g = acc_g[i]
        acc_u[i] = g * _sigmoid(g, fastmath) * acc_u[i]
    return acc_u


@cute.jit
def apply_situ(
    acc_u,
    acc_g,
    fastmath: cutlass.Constexpr,
    situ_beta: cutlass.Constexpr,
    situ_linear_beta: cutlass.Constexpr,
):
    inv_beta = 1.0 / situ_beta
    inv_linear_beta = 1.0 / situ_linear_beta
    for i in cutlass.range_constexpr(cute.size(acc_u)):
        g, u = acc_g[i], acc_u[i]
        soft_g = situ_beta * cute.math.tanh(g * inv_beta, approx=False)
        soft_u = situ_linear_beta * cute.math.tanh(u * inv_linear_beta, approx=False)
        acc_u[i] = soft_g * _sigmoid(g, fastmath) * soft_u
    return acc_u


ACTIVATION_FNS = {ActivationType.Swiglu: apply_silu, ActivationType.Situ: apply_situ}


def resolve_activation_fn(activation, situ_beta, situ_linear_beta):
    fn = ACTIVATION_FNS[activation]
    if activation is ActivationType.Situ:
        fn = functools.partial(
            fn, situ_beta=situ_beta, situ_linear_beta=situ_linear_beta
        )
    return fn
