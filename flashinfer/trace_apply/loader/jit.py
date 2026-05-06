from __future__ import annotations

from typing import Callable

from flashinfer.trace_apply.schema import Solution


def load(solution: Solution) -> Callable:
    raise NotImplementedError(
        "JIT loader (CUDA / CUTLASS / CuTile) is not yet wired up in this POC. "
        "It will materialize Solution.sources to disk and build via "
        "flashinfer.jit.gen_jit_spec, mirroring how built-in kernels are compiled."
    )
