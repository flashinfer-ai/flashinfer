"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
from pathlib import Path

# Get the path to the cubins directory within this package
CUBIN_DIR = Path(__file__).parent / "cubins"


def get_cubin_dir():
    """Get the directory containing the cubins."""
    return str(CUBIN_DIR)


def list_cubins():
    """List all available cubin files."""
    if not CUBIN_DIR.exists():
        return []

    cubins = []
    for root, _, files in os.walk(CUBIN_DIR):
        for file in files:
            if file.endswith(".cubin"):
                rel_path = os.path.relpath(os.path.join(root, file), CUBIN_DIR)
                cubins.append(rel_path)
    return sorted(cubins)


def get_cubin_path(relative_path):
    """Get the absolute path to a specific cubin file."""
    return str(CUBIN_DIR / relative_path)


# Read version from build metadata or fallback to main flashinfer version.txt
def _get_version():
    # First try to read from build metadata (for wheel distributions)
    try:
        from . import _build_meta

        return _build_meta.__version__
    except ImportError:
        pass

    # Fallback to reading from the main flashinfer version.txt (for development)
    version_file = Path(__file__).parent.parent.parent / "version.txt"
    if version_file.exists():
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0"


def _get_git_version():
    # First try to read from build metadata (for wheel distributions)
    try:
        from . import _build_meta

        return _build_meta.__git_version__
    except (ImportError, AttributeError):
        pass

    return "unknown"


__version__ = _get_version()
__git_version__ = _get_git_version()
__all__ = ["get_cubin_dir", "list_cubins", "get_cubin_path", "CUBIN_DIR"]
