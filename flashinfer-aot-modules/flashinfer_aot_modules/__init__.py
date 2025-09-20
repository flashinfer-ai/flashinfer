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
AOT_MODULES_DIR = Path(__file__).parent / "aot_modules"


def get_aot_modules_dir():
    """Get the directory containing the AOT compiled modules."""
    return str(AOT_MODULES_DIR)


def list_aot_modules():
    """List all available AOT compiled modules."""
    if not AOT_MODULES_DIR.exists():
        return []

    modules = []
    for root, _, files in os.walk(AOT_MODULES_DIR):
        for file in files:
            if file.endswith(".so"):
                rel_path = os.path.relpath(os.path.join(root, file), AOT_MODULES_DIR)
                modules.append(rel_path)
    return sorted(modules)


def get_module_path(module_name: str) -> Optional[str]:
    """Get the absolute path to a specific AOT module.

    Args:
        module_name: Name of the module directory or relative path to .so file

    Returns:
        Absolute path to the module's .so file, or None if not found
    """
    # If module_name doesn't end with .so, assume it's a directory name
    if not module_name.endswith(".so"):
        module_path = AOT_MODULES_DIR / module_name / f"{module_name}.so"
    else:
        module_path = AOT_MODULES_DIR / module_name

    if module_path.exists():
        return str(module_path)
    return None


def list_module_categories():
    """List the categories of available modules (the subdirectories)."""
    if not AOT_MODULES_DIR.exists():
        return []

    categories = []
    for item in AOT_MODULES_DIR.iterdir():
        if item.is_dir():
            # Check if this directory contains a .so file
            so_file = item / f"{item.name}.so"
            if so_file.exists():
                categories.append(item.name)
    return sorted(categories)


def get_module_info():
    """Get detailed information about available modules."""
    info = {
        "aot_modules_dir": str(AOT_MODULES_DIR),
        "modules": [],
        "categories": [],
        "total_count": 0,
    }

    if AOT_MODULES_DIR.exists():
        modules = list_aot_modules()
        info["modules"] = modules
        info["categories"] = list_module_categories()
        info["total_count"] = len(modules)

    return info


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
    "get_aot_modules_dir",
    "list_aot_modules",
    "get_module_path",
    "list_module_categories",
    "get_module_info",
    "AOT_MODULES_DIR",
]
