"""Load the Rubin export without changing the Python search path."""

from importlib import import_module


def bootstrap_paths() -> None:
    import_module("flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.src")
