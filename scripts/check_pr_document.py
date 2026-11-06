#!/usr/bin/env python3
"""Redirect: this script moved to scripts/infra/check_pr_document.py.

Kept until the GitLab CI setup uses the new location; then delete.
"""

import runpy
import sys
from pathlib import Path

_target = Path(__file__).resolve().parent / "infra" / "check_pr_document.py"
sys.argv[0] = str(_target)
runpy.run_path(str(_target), run_name="__main__")
