"""Load this export without claiming a process-wide ``sources`` package."""

from importlib import import_module


def bootstrap_paths() -> None:
    """Initialize the lazy source package under its FlashInfer namespace."""
    import_module("..src.sources", __package__)
