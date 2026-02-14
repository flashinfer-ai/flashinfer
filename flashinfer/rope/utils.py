"""
Copyright (c) 2024-2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

RoPE Utility Functions
======================

This module contains utility functions for the RoPE operations:
- _is_cute_dsl_available(): Check if CuTe-DSL backend is available
- get_rope_module(): Get the JIT-compiled CUDA rope module
"""

import functools


def _is_cute_dsl_available() -> bool:
    """Check if CuTe-DSL backend is available.

    Returns
    -------
    bool
        True if CuTe-DSL backend is available and can be imported.
    """
    try:
        from ..cute_dsl import is_cute_dsl_available

        return is_cute_dsl_available()
    except ImportError:
        return False


@functools.cache
def get_rope_module():
    """Get the JIT-compiled CUDA RoPE module.

    Returns
    -------
    module
        The compiled and loaded CUDA module containing RoPE kernels.
    """
    from ..jit.rope import gen_rope_module

    return gen_rope_module().build_and_load()
