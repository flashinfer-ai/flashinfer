from __future__ import annotations


def is_cuda_13_or_newer() -> bool:
    try:
        from ...jit.cpp_ext import get_cuda_version

        return get_cuda_version().major >= 13
    except Exception:
        return False
