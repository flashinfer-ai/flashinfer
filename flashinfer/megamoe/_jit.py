"""Process-wide cache of the built MegaMoE JIT modules."""

import functools


@functools.cache
def _get_jit_module(factory, *args):
    """Build (once per argument tuple) and load the module ``factory(*args)`` describes."""
    return factory(*args).build_and_load()
