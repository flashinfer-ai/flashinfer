import os
import shutil
import subprocess
import re


def _patch_triton_ptxas_blackwell():
    """Point Triton at the system ptxas for Blackwell (CUDA >= 13)."""
    if os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"):
        return
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return
    try:
        out = subprocess.check_output(
            [ptxas, "--version"], stderr=subprocess.STDOUT
        ).decode()
    except subprocess.CalledProcessError:
        return
    m = re.search(r"release (\d+)\.", out)
    if m and int(m.group(1)) >= 13:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas


_patch_triton_ptxas_blackwell()

from . import cascade  # noqa: F401
from . import sm_constraint_gemm  # noqa: F401
