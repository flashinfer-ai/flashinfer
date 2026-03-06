"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Ahead-of-time compilation support for CuTe DSL kernels.

CuTe DSL kernels are normally JIT-compiled via ``cute.compile()`` with
``--enable-tvm-ffi``.  This module provides utilities to:

1. **Export** a compiled CuTe DSL kernel to an object file via TVM-FFI ABI
   (``compiled_fn.export_to_c()``)
2. **Link** the object file into a shared library (``.so``)
3. **Load** the pre-compiled ``.so`` at runtime via
   ``cute.runtime.load_module()``

The workflow mirrors how FlashInfer's JitSpec-based AOT system works:
build-time compilation produces ``.so`` files that are discovered and
loaded at runtime, skipping the JIT step entirely.
"""

import gc
import subprocess
from pathlib import Path
from typing import Any, Callable, Optional

from . import env as jit_env
from .core import logger


# Directory for CuTe DSL AOT artifacts (alongside the JitSpec AOT dir)
CUTE_DSL_AOT_SUBDIR = "cute_dsl"


def get_cute_dsl_aot_dir() -> Path:
    """Get the directory for pre-compiled CuTe DSL shared libraries."""
    return jit_env.FLASHINFER_AOT_DIR / CUTE_DSL_AOT_SUBDIR


def get_cute_dsl_jit_dir() -> Path:
    """Get the directory for JIT-compiled CuTe DSL shared libraries."""
    return jit_env.FLASHINFER_JIT_DIR / CUTE_DSL_AOT_SUBDIR


def export_cute_dsl_kernel(
    compiled_fn: Any,
    func_name: str,
    output_dir: Path,
) -> Path:
    """Export a compiled CuTe DSL kernel to a shared library.

    Args:
        compiled_fn: The result of ``cute.compile(..., options="--enable-tvm-ffi")``.
            Must be a ``TVMFFIJitCompiledFunction`` with ``export_to_c`` method.
        func_name: Unique function name for the exported symbol.
        output_dir: Directory to write the ``.o`` and ``.so`` files.

    Returns:
        Path to the linked ``.so`` file.
    """
    import cutlass.cute.runtime as rt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    obj_file = output_dir / f"{func_name}.o"
    so_file = output_dir / f"{func_name}.so"

    # Export to object file using TVM-FFI ABI
    compiled_fn.export_to_c(str(obj_file), function_name=func_name)

    # Link into shared library
    runtime_libs = rt.find_runtime_libraries(enable_tvm_ffi=True)
    link_cmd = ["gcc", "-shared", "-o", str(so_file), str(obj_file)] + runtime_libs
    subprocess.run(link_cmd, check=True)

    logger.info(f"CuTe DSL AOT: exported {func_name} -> {so_file}")
    return so_file


def load_cute_dsl_module(so_path: Path) -> Any:
    """Load a pre-compiled CuTe DSL shared library via TVM-FFI.

    Args:
        so_path: Path to the ``.so`` file.

    Returns:
        A TVM-FFI module with the exported function(s) as attributes.
    """
    import cutlass.cute.runtime as rt

    return rt.load_module(str(so_path), enable_tvm_ffi=True)


def try_load_aot_kernel(func_name: str) -> Optional[Callable]:
    """Try to load a pre-compiled CuTe DSL kernel from the AOT directory.

    Checks the AOT directory first, then the JIT cache directory.

    Args:
        func_name: The function name used during export.

    Returns:
        The loaded TVM-FFI function, or None if not found.
    """
    for base_dir in [get_cute_dsl_aot_dir(), get_cute_dsl_jit_dir()]:
        so_path = base_dir / f"{func_name}.so"
        if so_path.exists():
            try:
                mod = load_cute_dsl_module(so_path)
                func = getattr(mod, func_name, None)
                if func is not None:
                    logger.info(f"CuTe DSL AOT: loaded {func_name} from {so_path}")
                    return func
            except Exception as e:
                logger.warning(
                    f"CuTe DSL AOT: failed to load {func_name} from {so_path}: {e}"
                )
    return None


def compile_and_cache_cute_dsl_kernel(
    compile_fn: Callable[[], Any],
    func_name: str,
) -> Callable:
    """Compile a CuTe DSL kernel with AOT caching.

    First checks if a pre-compiled version exists.  If not, calls
    ``compile_fn()`` to JIT-compile, then exports to the JIT cache
    directory so subsequent runs skip compilation.

    Args:
        compile_fn: A zero-argument callable that returns the result of
            ``cute.compile(..., options="--enable-tvm-ffi")``.
        func_name: Unique function name for caching.

    Returns:
        The loaded TVM-FFI function (either from cache or freshly compiled).
    """
    # Try AOT/cached version first
    cached = try_load_aot_kernel(func_name)
    if cached is not None:
        return cached

    # JIT compile
    # Disable GC during compilation to prevent ir_module from being collected
    # before export_to_c can use it.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        compiled_fn = compile_fn()

        # Cache the compiled kernel for future runs
        cache_dir = get_cute_dsl_jit_dir()
        try:
            so_path = export_cute_dsl_kernel(compiled_fn, func_name, cache_dir)
            mod = load_cute_dsl_module(so_path)
            func = getattr(mod, func_name)
            logger.info(f"CuTe DSL: compiled and cached {func_name}")
            return func
        except Exception as e:
            # If export/load fails, fall back to using the JIT-compiled function directly
            logger.warning(
                f"CuTe DSL AOT: cache export failed for {func_name}: {e}. "
                f"Using JIT-compiled function directly."
            )
            return compiled_fn
    finally:
        if gc_was_enabled:
            gc.enable()
