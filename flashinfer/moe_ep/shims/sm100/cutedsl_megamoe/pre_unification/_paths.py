"""Load the legacy provider package without modifying Python's search path."""

from importlib import import_module


def bootstrap_paths() -> None:
    import_module("flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification")
