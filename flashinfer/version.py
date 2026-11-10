"""
Copyright (c) 2023 by FlashInfer team.

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

# Centralized version information to avoid circular imports
try:
    from . import _build_meta  # type: ignore[attr-defined]

    __version__: str = _build_meta.__version__
    # Prefer __git_commit__ (new name); fall back to __git_version__ for stale
    # _build_meta.py files generated before the rename (editable installs that
    # haven't re-run pip install since the rename).
    __git_commit__: str = getattr(_build_meta, "__git_commit__", None) or getattr(
        _build_meta, "__git_version__", "unknown"
    )
except ImportError:
    __version__ = "0.0.0+unknown"
    __git_commit__ = "unknown"

# Backward-compat alias: __git_version__ was the original name before it was
# renamed to the clearer __git_commit__.
__git_version__: str = __git_commit__
