# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compile worker subprocesses for native W4A16 tactics.

The native kernels specialize on M, so autotuning a small-M bucket compiles
every candidate of that bucket before profiling it; see
``flashinfer.jit.cute_dsl_compile_pool``. Job:
``[m, n, k, f16, has_alpha, enable_pdl, tactic]``.
"""

import os

from ....jit.cute_dsl_compile_pool import CompilePool, serve, workers_from_env


def num_workers():
    """``FLASHINFER_NATIVE_W4A16_COMPILE_WORKERS``, else half the usable CPUs
    up to 8. 1 compiles in this process."""
    return workers_from_env(os.environ.get("FLASHINFER_NATIVE_W4A16_COMPILE_WORKERS"))


POOL = CompilePool(__name__, num_workers, "Native W4A16")


def _kernel_spec(m, n, k, f16, has_alpha, enable_pdl, tactic):
    import torch

    from .runner import _kernel_spec

    dtype = torch.float16 if f16 else torch.bfloat16
    return _kernel_spec(m, n, k, dtype, has_alpha, enable_pdl, tuple(tactic))


if __name__ == "__main__":
    serve(_kernel_spec)
