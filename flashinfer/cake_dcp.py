"""Compatibility imports for the renamed :mod:`flashinfer.dcp` module."""

from .dcp import (
    get_dcp_spec_counter_bytes,
    get_dcp_spec_workspace_size_bytes,
)

__all__ = [
    "get_dcp_spec_counter_bytes",
    "get_dcp_spec_workspace_size_bytes",
]
