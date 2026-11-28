import os
import shutil

from packaging.version import Version


def _patch_triton_ptxas_blackwell():
    """Use the system ptxas for Blackwell if it is CUDA >= 13 and newer than Triton's bundled one."""
    if os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"):
        return
    ptxas = shutil.which("ptxas")
    if not ptxas:
        return
    try:
        from triton.knobs import NvidiaTool, nvidia
    except ImportError:
        return
    # Check the class dict: hasattr() on the class would run the knob descriptor
    # with obj=None, which raises AttributeError and always yields False.
    if "ptxas_blackwell" not in vars(type(nvidia)):
        return  # Triton without a separate Blackwell ptxas (<= 3.5, main)
    try:
        system = NvidiaTool.from_path(ptxas)
        if system is None or Version(system.version) < Version("13.0"):
            return
        try:
            bundled = Version(nvidia.ptxas_blackwell.version)
        except RuntimeError:  # bundled ptxas-blackwell missing
            bundled = None
    except OSError:  # e.g. not executable; keep Triton's default
        return
    if bundled is None or Version(system.version) > bundled:
        os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = ptxas


_patch_triton_ptxas_blackwell()

from . import cascade  # noqa: F401
from . import sm_constraint_gemm  # noqa: F401
