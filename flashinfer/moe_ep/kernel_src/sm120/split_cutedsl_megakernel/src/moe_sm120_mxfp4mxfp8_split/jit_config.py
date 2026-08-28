"""Explicit code-generation options for SM120 Split-MegaMoE.

The benchmark CLI historically selected a few compile-time experiments via
environment variables.  Production callers should construct this immutable
object instead: it makes the CuTe DSL specialization inputs visible and gives
frameworks a deterministic value to include in their compiled-kernel cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from typing import Any, Dict, Optional


def _read_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "on"):
        return True
    if normalized in ("0", "false", "off"):
        return False
    raise ValueError(f"{name} must be a boolean value, got {value!r}")


def _read_optional_int(name: str) -> Optional[int]:
    value = os.environ.get(name)
    return None if value is None else int(value)


@dataclass(frozen=True)
class Sm120JitConfig:
    """All non-shape options that specialize generated SM120 device code."""

    fc12_stream_tiles: int = 0
    k1_ready_major: bool = True
    k2_token_stripe: int = 2
    fc2_ready_bundle_k_tiles: int = 12
    fc2_packed_store: Optional[bool] = None
    num_ab_stages_override: Optional[int] = None
    enable_globaltimer: bool = False
    enable_k2_tile_trace: bool = False

    def __post_init__(self) -> None:
        if self.fc12_stream_tiles < 0:
            raise ValueError("fc12_stream_tiles must be non-negative")
        if self.k2_token_stripe <= 0:
            raise ValueError("k2_token_stripe must be positive")
        if self.fc2_ready_bundle_k_tiles <= 0:
            raise ValueError("fc2_ready_bundle_k_tiles must be positive")
        if (
            self.num_ab_stages_override is not None
            and self.num_ab_stages_override <= 0
        ):
            raise ValueError("num_ab_stages_override must be positive")

    @classmethod
    def from_environment(cls) -> "Sm120JitConfig":
        """Parse legacy benchmark environment variables exactly once."""

        stream_tiles = _read_optional_int("MEGA_FC12_STREAM_TILES")
        if stream_tiles is None:
            stream_tiles = 1 if _read_bool("MEGA_FC12_STREAMING", False) else 0
        enable_k2_tile_trace = _read_bool("MEGA_SPLIT_K2_TILE_TRACE", False)
        return cls(
            fc12_stream_tiles=stream_tiles,
            k1_ready_major=_read_bool("MEGA_K1_READY_MAJOR", True),
            k2_token_stripe=int(os.environ.get("MEGA_K2_TOKEN_STRIPE", "2")),
            fc2_ready_bundle_k_tiles=int(
                os.environ.get("MEGA_FC2_READY_BUNDLE_K_TILES", "12")
            ),
            fc2_packed_store=(
                _read_bool("MEGA_FC2_PACKED_STORE", True)
                if "MEGA_FC2_PACKED_STORE" in os.environ
                else None
            ),
            num_ab_stages_override=_read_optional_int("MEGA_NUM_AB_STAGES"),
            enable_globaltimer=(
                _read_bool("MEGA_SPLIT_GLOBALTIMER", False)
                or enable_k2_tile_trace
            ),
            enable_k2_tile_trace=enable_k2_tile_trace,
        )

    def canonical_dict(self) -> Dict[str, Any]:
        """Return the JSON-safe representation used by the cache key."""

        return asdict(self)


__all__ = ["Sm120JitConfig"]
