"""Host-side FC1 geometry for supported SM107 activations."""


def fc1_rows_per_intermediate(activation: str) -> int:
    """Return FC1 rows per activated intermediate element; reject unknown layouts."""
    if activation in ("swiglu", "situ"):
        return 2
    if activation == "clamped_relu2":
        return 1
    raise ValueError(f"unsupported SM107 activation {activation!r}")
