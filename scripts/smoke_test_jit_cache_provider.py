#!/usr/bin/env python3
"""Redirect: this script moved to scripts/infra/smoke_test_jit_cache_provider.py.

Kept until the GitLab CI setup uses the new location; then delete.
"""

import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent / "infra" / "smoke_test_jit_cache_provider.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
