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
import subprocess
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Hinted-kernel cache (ported from tilegym.ops.cutile.utils.cached_replace_hints)
# ---------------------------------------------------------------------------
# cuTile's per-shape JIT compile cache lives *on the hinted-kernel object*
# returned by ``kernel.replace_hints(...)``. Calling ``replace_hints`` afresh on
# every launch (as a naive migration does) builds a new object each time and
# discards that compile cache -> a full recompile per launch on the non-autotune
# / single-config paths. Memoizing the hinted kernel keeps the compile cache
# alive across launches, matching the TileGym source behavior.
#
# Deliberately no ``cuda.tile`` import here so this module stays importable on
# environments without the cuTile compile chain (see module docstring).
_HINTED_KERNEL_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_HINTED_KERNEL_CACHE_MAX = 256


def cached_replace_hints(kernel, **hints):
    """Return a memoized ``kernel.replace_hints(**hints)``.

    Keyed on ``(id(kernel), sorted(hints))``. The source kernel is stored
    alongside the hinted kernel so its ``id()`` cannot be recycled while the
    entry is live. Bounded LRU (cap ``_HINTED_KERNEL_CACHE_MAX``).
    """
    key = (id(kernel), tuple(sorted(hints.items())))
    entry = _HINTED_KERNEL_CACHE.get(key)
    if entry is not None:
        _HINTED_KERNEL_CACHE.move_to_end(key)
        return entry[0]
    hinted = kernel.replace_hints(**hints)
    _HINTED_KERNEL_CACHE[key] = (hinted, kernel)  # keep owner alive
    if len(_HINTED_KERNEL_CACHE) > _HINTED_KERNEL_CACHE_MAX:
        _HINTED_KERNEL_CACHE.popitem(last=False)
    return hinted


def _find_tileiras_binary() -> str | None:
    """Return the path to the tileiras binary, or None if not found."""
    try:
        importlib.metadata.version("nvidia-cuda-tileiras")
        from cuda.tile._compile import _find_compiler_bin

        binary = _find_compiler_bin()
        return str(binary.path)
    except Exception:
        pass
    p = shutil.which("tileiras")
    if p is not None:
        return p
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    candidate = os.path.join(cuda_home, "bin", "tileiras")
    return candidate if os.path.exists(candidate) else None


def _tileiras_supports_arch(tileiras_path: str, sm_arch: str) -> bool:
    """Return True iff the given tileiras binary supports ``sm_arch`` (e.g. ``sm_90``)."""
    try:
        result = subprocess.run(
            [tileiras_path, "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return f"={sm_arch}" in (result.stdout + result.stderr)
    except Exception:
        return True  # assume supported if we can't check


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

    Also verifies that the installed tileiras binary supports the current GPU's
    SM architecture (e.g. ``sm_90`` for Hopper).  Some cuda-tile wheel builds
    (e.g. the cu13 toolchain shipped with ``9.9.99.dev*``) do not include SM90
    support even though the Python API lists it as a valid target; calling
    ``compile_cubin`` with ``--gpu-name sm_90`` would crash mid-autotune with a
    confusing ``ValueError: No valid config found`` otherwise.

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

    tileiras_path = _find_tileiras_binary()
    if tileiras_path is None:
        return False

    # Check that the installed tileiras supports the current GPU's SM arch.
    # Some toolchain builds omit certain architectures (e.g. cu13 drops sm_90).
    try:
        import torch

        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            sm_arch = f"sm_{major}{minor}"
            if not _tileiras_supports_arch(tileiras_path, sm_arch):
                return False
    except Exception:
        pass  # no torch or no GPU — skip arch check, let the caller decide

    return True
