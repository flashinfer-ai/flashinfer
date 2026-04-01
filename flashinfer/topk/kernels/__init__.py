"""
FlashInfer Top-K CuTe-DSL Kernels (Experimental)
=================================================

This subpackage contains CuTe-DSL top-k kernel implementations vendored from
TRT-LLM's Blackwell top-k kernel path. These kernels use the single-pass
multi-CTA radix top-k algorithm optimized for large-k selection (k <= 2048).

.. warning::

    These kernels are **experimental** and require:
    - Blackwell architecture (SM100+)
    - nvidia-cutlass-dsl package

    The API may change or be removed in future versions without notice.
"""

__all__ = []

_cute_dsl_available = False
try:
    from ...cute_dsl import is_cute_dsl_available

    if is_cute_dsl_available():
        from .top_k_cute_dsl import top_k_cute_dsl  # noqa: F401

        _cute_dsl_available = True
except ImportError:
    pass

if _cute_dsl_available:
    __all__ += [
        "top_k_cute_dsl",
    ]
