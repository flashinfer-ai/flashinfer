# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Shared lightweight helpers for cuTile-backed kernels.

This module intentionally has *no* ``cuda.tile`` imports so it stays
importable on environments where the cuTile compile chain isn't present
(e.g. ``flashinfer-ci-cu126/cu128/cu129`` docker images, which target CUDA
12.x ecosystems and don't ship the ``nvidia-cuda-tileiras`` cu13 toolchain).
Callers — including pytest skip-guards — can use
:func:`is_cuda_tile_available` to gate ``backend="cutile"`` paths without
triggering a hard ``ImportError`` at module import time.
"""

import importlib.metadata
import importlib.util
import os
import shutil


def is_cuda_tile_available() -> bool:
    """Return True iff cuTile kernels can actually JIT-compile in this env.

    A working cuTile setup requires *both*:

    1. The ``cuda.tile.tune`` Python submodule (added in ``cuda-tile>=1.4.0``;
       older 1.2.0 wheels shipped with the cu12 CI base images do not have it).
    2. The ``tileiras`` compiler binary, which ``cuda.tile`` searches for in
       three locations (mirrored here in the same order):

       a. As the Python wheel ``nvidia-cuda-tileiras`` (cu13-only, ships
          with the ``cuda-tile[tileiras]`` extras and our build-backend hook).
       b. As ``tileiras`` on ``PATH``.
       c. As ``${CUDA_HOME:-/usr/local/cuda}/bin/tileiras`` (system CTK 13.1+).

    Mirrors :func:`flashinfer.cute_dsl.utils.is_cute_dsl_available` (which only
    probes Python modules) but extends the check to the native compiler — cuTile
    fails *mid-autotune* with a confusing ``ValueError: No valid config found``
    + ``FileNotFoundError: 'tileiras' compiler not found`` cascade when the
    compiler is absent, which Python-module-only probes would miss.

    This is needed because in CI's ``pip install -e . -v`` path (PEP 517
    isolation), our build-backend's ``_install_cuda_tile_compile_deps()`` hook
    cannot install ``nvidia-cuda-tileiras`` into the target env (the build venv
    has no ``pip`` module to call). On cu12x CI images that don't pre-install
    tileiras, this skip-guard prevents cuTile tests from crashing mid-JIT.
    """
    if importlib.util.find_spec("cuda.tile.tune") is None:
        return False
    try:
        importlib.metadata.version("nvidia-cuda-tileiras")
        return True
    except importlib.metadata.PackageNotFoundError:
        pass
    if shutil.which("tileiras") is not None:
        return True
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    return os.path.exists(os.path.join(cuda_home, "bin", "tileiras"))
