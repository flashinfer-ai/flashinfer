"""Load the Rubin export without changing the Python search path."""

from importlib import import_module


def bootstrap_paths() -> None:
    import_module("flashinfer.moe_ep.kernels.cutedsl_megamoe")
