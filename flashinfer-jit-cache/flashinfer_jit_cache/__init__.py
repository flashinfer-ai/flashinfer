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
jit_cache_dir = Path(__file__).parent / "jit_cache"


def get_jit_cache_dir() -> str:
    """Get the directory containing the AOT compiled modules."""
    return str(jit_cache_dir)


try:
    from ._build_meta import __version__ as __version__
    from ._build_meta import __git_version__ as __git_version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
    __git_version__ = "unknown"


__all__ = [
    "get_jit_cache_dir",
]
