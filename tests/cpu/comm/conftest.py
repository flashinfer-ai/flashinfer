"""Keep the original comm tests' import setup for spawned workers."""

from pathlib import Path
import sys


_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
