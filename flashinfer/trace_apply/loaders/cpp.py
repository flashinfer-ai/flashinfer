# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""C++/CUDA-family solution loader (cpp / cuda / cutlass / cutile).

These require compilation, linking, caching, and ABI binding, so they go through
``flashinfer.jit``. The exported entry-point symbol is registered via
``TVM_FFI_DLL_EXPORT_TYPED_FUNC`` and is called positionally.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from flashinfer.trace.solution import Solution
from flashinfer.trace_apply.loaders.python import materialize

# Compiled extensions; the rest (.cuh/.h/.hpp/.inc) are headers seen via the
# include path.
_COMPILE_EXTS = {".cu", ".cpp", ".cc", ".cxx", ".c"}


def load(solution: Solution) -> Callable:
    """Build the solution via ``flashinfer.jit`` and return the entry callable.

    Sources are materialized to a hashed cache dir; the compilable files are
    passed to ``gen_jit_spec`` and the dir is added as an include path so the
    solution's own headers resolve. Cached by solution hash so it never collides
    with built-in flashinfer modules and is built at most once per process/disk.
    """
    from flashinfer.jit import gen_jit_spec  # noqa: PLC0415

    sol_dir = materialize(solution)

    sources = [
        str(sol_dir / src.path)
        for src in solution.sources
        if Path(src.path).suffix.lower() in _COMPILE_EXTS
    ]
    if not sources:
        raise ValueError(
            f"Solution {solution.name!r} (language={solution.spec.language.value}) has "
            f"no compilable sources ({sorted(_COMPILE_EXTS)})."
        )

    name = f"trace_apply_{solution.hash()[:16]}"
    include_paths: list[str | Path] = sorted(
        {str((sol_dir / src.path).parent) for src in solution.sources} | {str(sol_dir)}
    )

    spec = gen_jit_spec(name, sources, extra_include_paths=include_paths)
    module = spec.build_and_load()

    symbol = solution.entry_symbol()
    fn = getattr(module, symbol, None)
    if fn is None:
        raise AttributeError(
            f"Solution {solution.name!r}: entry-point symbol {symbol!r} not found in the "
            f"built module. Ensure it is exported via "
            f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({symbol}, ...)."
        )
    return fn


__all__ = ["load"]
