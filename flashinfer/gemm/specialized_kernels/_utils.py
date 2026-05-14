from __future__ import annotations

from packaging.version import Version


def is_cuda_13_or_newer() -> bool:
    try:
        from ...jit.cpp_ext import get_cuda_version

        return get_cuda_version() >= Version("13.0")
    except Exception:
        return False
