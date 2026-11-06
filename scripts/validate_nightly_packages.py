#!/usr/bin/env python3
"""Redirect: this script moved to scripts/infra/validate_nightly_packages.py.

Kept until the GitLab CI setup uses the new location; then delete.
"""

import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent / "infra" / "validate_nightly_packages.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
