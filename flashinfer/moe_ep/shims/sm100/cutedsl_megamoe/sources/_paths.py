"""Load the provider package without claiming a global sources namespace."""

from importlib import import_module


def bootstrap_paths() -> None:
    import_module("flashinfer.moe_ep.kernels.cutedsl_megamoe")
