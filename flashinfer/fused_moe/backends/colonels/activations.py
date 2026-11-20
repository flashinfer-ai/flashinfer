from ...api import ActivationConfig, ClampedReLU2, ReLU2, SwiGLU


def activation_kind(activation: ActivationConfig) -> int:
    """Lower a shared activation config to its native CUDA specialization."""
    if type(activation) is SwiGLU:
        if activation != SwiGLU():
            raise NotImplementedError(
                "Colonels SwiGLU requires the default alpha, beta and limit"
            )
        return 0
    if type(activation) is ReLU2:
        return 1
    if type(activation) is ClampedReLU2:
        return 2
    raise NotImplementedError(f"Colonels does not support {activation!r}")
