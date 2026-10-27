"""Process-wide cache of the built MegaMoE JIT modules."""

import functools


@functools.cache
def _get_jit_module(factory, *args):
    return factory(*args).build_and_load()
