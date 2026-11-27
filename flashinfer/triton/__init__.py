import os
import re
import shutil
import subprocess


def _patch_triton_ptxas_blackwell():
    """Point Triton at a newer system ptxas for Blackwell (CUDA >= 13)."""
    if os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"):
        return
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return
    try:
        out = subprocess.check_output(
            [ptxas, "--version"], stderr=subprocess.STDOUT
        ).decode()
    except (OSError, subprocess.CalledProcessError):
        return
    m = re.search(r"release (\d+)\.(\d+)", out)
    if not m or int(m.group(1)) < 13:
        return
    try:
        from triton.backends.nvidia.compiler import get_ptxas

        bundled_ver = tuple(map(int, get_ptxas(100).version.split(".")[:2]))
    except Exception:
        bundled_ver = (0, 0)
    if (int(m.group(1)), int(m.group(2))) > bundled_ver:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas


_patch_triton_ptxas_blackwell()

from . import cascade  # noqa: F401
from . import sm_constraint_gemm  # noqa: F401
