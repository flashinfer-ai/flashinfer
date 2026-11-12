"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import hashlib
import inspect
import re
from typing import Callable, Sequence

import torch

# The specialization name is the sole per-kernel on-disk cache key (see
# docs/design_docs/cute_dsl_kernel_cache.md) and becomes both a filename and
# part of the exported TVM-FFI symbol, so it must stay within [A-Za-z0-9_].
_SANITIZE = re.compile(r"[^A-Za-z0-9_]")

# ext4 caps filenames at 255 bytes; the module dir adds "<module>_" to the
# exported symbol, so leave generous headroom before falling back to a digest.
_MAX_NAME_LEN = 180


def format_name_part(value) -> str:
    """Format one cache-key component as a symbol-safe name fragment."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return str(value).replace(".", "_").replace("-", "m").replace("+", "p")
    if isinstance(value, int):
        return str(value).replace("-", "m")
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if isinstance(value, tuple):
        return "t" + "x".join(format_name_part(v) for v in value)
    if isinstance(value, str):
        return _SANITIZE.sub("_", value.removeprefix("torch."))
    if isinstance(value, type) and value.__module__.startswith("cutlass"):
        return _SANITIZE.sub("_", value.__name__)
    raise TypeError(
        f"Unsupported cache-key component type {type(value).__name__}: {value!r}"
    )


def make_kernel_name(*parts) -> str:
    """Join cache-key components into a specialization name.

    Every codegen parameter must be passed; a component the name ignores makes
    two different kernels collide on one on-disk artifact.
    """
    name = "_".join(format_name_part(p) for p in parts)
    if len(name) > _MAX_NAME_LEN:
        digest = hashlib.sha256(name.encode()).hexdigest()[:16]
        name = f"{name[:_MAX_NAME_LEN]}_h{digest}"
    return name


def persist_cute_dsl_compile(
    module_name: str, key_files: Sequence[str]
) -> Callable[[Callable], Callable]:
    """Persist a TVM-FFI ``cute.compile`` factory in the CuTe-DSL disk cache.

    The kernel name is the function name plus every argument, so each argument
    must be a codegen parameter that :func:`format_name_part` accepts.
    """

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel

            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            name = make_kernel_name(fn.__name__.lstrip("_"), *bound.arguments.values())
            return build_and_load_cute_dsl_kernel(
                module_name, name, lambda: fn(*args, **kwargs), tuple(key_files)
            )

        return wrapper

    return decorator
