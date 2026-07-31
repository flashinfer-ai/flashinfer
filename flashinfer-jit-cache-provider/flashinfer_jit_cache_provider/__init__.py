"""Runtime entry point used by an installed jit-cache provider wheel."""

import json
from pathlib import Path
from typing import Any, Dict


_PACKAGE_DIR = Path(__file__).resolve().parent


def get_provider() -> Dict[str, Any]:
    """Return this provider's generated manifest to the jit-cache shim."""
    manifest_path = _PACKAGE_DIR / "manifest.json"
    with manifest_path.open() as manifest_file:
        manifest = json.load(manifest_file)
    manifest["jit_cache_dir"] = str(_PACKAGE_DIR / "jit_cache")
    return manifest


try:
    from ._build_meta import __git_version__, __version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
    __git_version__ = "unknown"


__all__ = ["get_provider"]
