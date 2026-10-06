"""Activation contracts for the selected BF16 cuDNN Frost sources."""

from ...fused_moe.api import (
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    ReLU,
    ReLU2,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)

ACTIVATIONS = {
    "swiglu": SwiGLU,
    "geglu": GeGLU,
    "geglu_tanh": GeGLUTanh,
    "relu2": ReLU2,
    "situ": SiTU,
    "swiglu_step": SwiGLUStep,
    "gelu": GELU,
    "relu": ReLU,
    "silu": SiLU,
    "identity": Identity,
}


def activation_name(activation):
    for name, cls in ACTIVATIONS.items():
        if type(activation) is cls and activation == cls():
            return name
    raise NotImplementedError(
        "cuDNN Frost selected sources require default activation parameters; "
        f"got {activation!r}"
    )


def contract_activation(contract):
    # Original schema-2 SwiGLU sources predate the typed activation key.
    value = contract.get("activation")
    return "swiglu" if value == "silu(gate) * up" else value


def is_gated(name):
    return ACTIVATIONS[name]().is_gated
