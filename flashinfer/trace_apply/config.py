from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TraceApplyConfig:
    # Whitelist of submission authors. None = accept all.
    allowed_authors: list[str] | None = None
    # Library-version compatibility window: "patch" | "minor" | "any".
    library_version_window: str = "minor"
