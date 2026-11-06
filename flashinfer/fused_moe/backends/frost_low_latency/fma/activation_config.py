# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import-light activation lowering; scalar semantics are not tactics."""

from ....api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    PowLU,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)

ACTIVATIONS = {
    SwiGLU: "swiglu",
    SiTU: "situ",
    GeGLU: "geglu",
    GeGLUTanh: "geglu_tanh",
    SwiGLUStep: "swiglu_step",
    PowLU: "powlu",
    GELU: "gelu",
    ReLU: "relu",
    ReLU2: "relu2",
    SiLU: "silu",
    Identity: "identity",
}


def lower_activation(activation):
    try:
        name = ACTIVATIONS[type(activation)]
    except KeyError:
        raise NotImplementedError(f"Frost does not support {activation!r}") from None
    if isinstance(activation, PowLU):
        return name, (activation.m, activation.limit)
    if activation == type(activation)():
        # Keep the exact legacy device specialization for default semantics.
        return name, None
    if isinstance(activation, SwiGLU):
        return name, (activation.alpha, activation.beta, activation.limit)
    if isinstance(activation, SwiGLUStep):
        return name, (activation.limit,)
    raise NotImplementedError(
        f"Frost requires default parameters for {type(activation).__name__}"
    )
