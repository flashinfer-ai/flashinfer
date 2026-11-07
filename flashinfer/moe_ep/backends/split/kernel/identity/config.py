"""Identity split kernel config — dispatch/combine roundtrip stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IdentityConfig:
    """Inner compute stub — returns a copy of the dispatched tokens."""

    kernel_name: str = "identity"
