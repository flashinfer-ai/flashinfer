"""Runtime for kernels generated offline by cuDNN Frost.

cuDNN Frost is a build-time dependency only.  The deployed FlashInfer process reads
a source manifest and JIT-compiles its standalone Python kernels with CuTe DSL.
Compiled code is cached in FlashInfer's writable JIT directory.
"""

__all__ = [
    "CudnnFrostGroupedGemm1SwiGLURunner",
    "matching_kernels",
    "workspace_size",
]


def __getattr__(name):
    # A deferred support-check import must not load the execution implementation.
    if name in __all__:
        from . import runtime

        return getattr(runtime, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
