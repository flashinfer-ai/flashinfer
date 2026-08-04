"""Identity split kernel config — dispatch/combine roundtrip stub."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IdentityConfig:
    """Inner compute stub — passes dispatched tokens through unchanged."""

    kernel_name: str = "identity"
