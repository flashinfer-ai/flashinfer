"""On-disk cache for the SM90 MSA CuTe-DSL kernels.

``cute.compile`` has no persistent cache, so without this every process (and
every tensor-parallel rank) recompiles each kernel at its first use, 1-2 s a
kernel and often mid-serving.  Each kernel module keeps its in-process dict in
front of this; a miss there loads the exported object from FlashInfer's CuTe-DSL
cache, compiling and exporting it only on the node's first use.
"""

import hashlib
from typing import Any, Callable

from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel


def load_or_compile(
    module: str, key: tuple, compile_fn: Callable[[], Any], *source_files: str
) -> Any:
    """The kernel of ``key`` in ``module``: loaded from disk, or compiled by
    ``compile_fn`` (a ``cute.compile(..., options="--enable-tvm-ffi")`` call)
    and exported.  ``key`` must name everything the compile specializes on;
    a change to ``source_files`` (the kernel's module and any it imports
    kernel code from) invalidates the cache."""
    name = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return build_and_load_cute_dsl_kernel(
        module, f"k_{name}", compile_fn, extra_key_files=source_files
    )
