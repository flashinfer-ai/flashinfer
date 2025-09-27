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
from typing import List, Optional

# Get the path to the AOT modules directory within this package
jit_cache_DIR = Path(__file__).parent / "jit_cache"


def get_jit_cache_dir() -> str:
    """Get the directory containing the AOT compiled modules."""
    return str(jit_cache_DIR)


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


__version__ = _get_version()
__all__ = [
    "get_jit_cache_dir",
]
