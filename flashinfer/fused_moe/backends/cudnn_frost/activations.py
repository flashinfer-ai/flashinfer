"""Activation contracts for the selected BF16 cuDNN Frost sources."""

from ....fused_moe.api import (
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

# Activations whose selected sources bake in the activation's default parameters.
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
# Activations whose sources read their parameters as launch-time scalars, so one
# kernel serves every value. ``swiglu_param`` is SwiGLU with non-default
# ``alpha``, ``beta`` or ``limit``: ``g = min(gate, limit)``,
# ``u = clamp(up, -limit, limit)``, ``g * sigmoid(alpha * g) * (u + beta)``.
PARAMETERIZED_ACTIVATIONS = {"swiglu_param": SwiGLU}
# Every activation name a catalog record or exported op may carry.
KNOWN_ACTIVATIONS = {**ACTIVATIONS, **PARAMETERIZED_ACTIVATIONS}
# Launch-tail names of the swiglu_param scalars; the lower bound is -limit.
SWIGLU_PARAM_SCALARS = ("swiglu_alpha", "swiglu_beta", "swiglu_limit", "swiglu_lower")
_DEFAULT_ACTIVATIONS = {name: cls() for name, cls in KNOWN_ACTIVATIONS.items()}


def activation_name(activation):
    for name, cls in ACTIVATIONS.items():
        if type(activation) is cls and activation == _DEFAULT_ACTIVATIONS[name]:
            return name
    if type(activation) is SwiGLU:
        # Non-default alpha/beta/limit (e.g. DeepSeek-V4's limit 10, gpt-oss's
        # alpha 1.702 / beta 1 / limit 7) run the scalar-parameterized sources.
        return "swiglu_param"
    raise NotImplementedError(
        "cuDNN Frost selected sources require default activation parameters; "
        f"got {activation!r}"
    )


def activation_scalars(activation) -> dict[str, float]:
    """Launch-time scalar values (launch-tail name -> value) of an activation.

    Empty for activations whose parameters are compiled into their sources.
    """
    if activation is None or activation_name(activation) != "swiglu_param":
        return {}
    limit = float(activation.limit)
    return dict(
        swiglu_alpha=float(activation.alpha),
        swiglu_beta=float(activation.beta),
        swiglu_limit=limit,
        swiglu_lower=-limit,
    )


_FLOAT32_MAX = 3.4028234663852886e38
_SCALAR_DEFAULTS = (1.0, 0.0, _FLOAT32_MAX, -_FLOAT32_MAX)


def launch_scalars(activation) -> tuple[float, float, float, float]:
    """``(alpha, beta, limit, -limit)`` for the native MoE executors' scalar slots.

    Kernels without launch-time parameters never read them; they then hold
    SwiGLU's defaults.
    """
    values = activation_scalars(activation)
    alpha, beta, limit, lower = (
        values.get(name, default)
        for name, default in zip(SWIGLU_PARAM_SCALARS, _SCALAR_DEFAULTS, strict=True)
    )
    return alpha, beta, limit, lower


# Scalar slots of executors whose kernels take no launch-time parameters.
DEFAULT_LAUNCH_SCALARS = launch_scalars(None)


def contract_activation(contract):
    # Original schema-2 SwiGLU sources predate the typed activation key.
    value = contract.get("activation")
    return "swiglu" if value == "silu(gate) * up" else value


def is_gated(name):
    return _DEFAULT_ACTIVATIONS[name].is_gated
