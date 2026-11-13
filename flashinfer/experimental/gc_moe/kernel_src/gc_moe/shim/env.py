"""Runtime environment adaptation for the vendored GC MoE sources.

The vendored sources resolve their PTX assembler once, at import time
(``src/runtime/jit.py`` reads ``GC_MOE_PTXAS`` with a hard-coded
``/usr/local/cuda/bin/ptxas`` fallback). This module resolves the assembler
first, exports it, and validates it, so the failure is an actionable message
instead of a raw ``RuntimeError`` from deep inside the JIT.

The vendored Green Context runtime also reads ``GC_MOE_PRIMARY_CTX`` when it
builds its first context pair. The integration pins it on: every role kernel
runs in the primary context and the SM split only sizes grids.
"""

from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess

PTXAS_ENV = "GC_MOE_PTXAS"
PRIMARY_CTX_ENV = "GC_MOE_PRIMARY_CTX"
_MIN_PTXAS = (13, 2)


def _parse_release(text: str) -> tuple[int, int] | None:
    match = re.search(r"release[ \t]+([0-9]+)\.([0-9]+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def ptxas_version(path: str) -> tuple[int, int]:
    if not Path(path).is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"GC MoE needs an executable CUDA PTX assembler at {path!r}; "
            "install CUDA >= 13.2 or point GC_MOE_PTXAS (or the ptxas= argument) "
            "at one"
        )
    result = subprocess.run(
        [path, "--version"], check=False, capture_output=True, text=True
    )
    release = _parse_release(result.stdout + result.stderr)
    if result.returncode != 0 or release is None:
        raise RuntimeError(
            f"cannot query the CUDA PTX assembler at {path!r}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    if release < _MIN_PTXAS:
        raise RuntimeError(
            f"GC MoE assembles PTX ISA >= 9.1 and needs ptxas >= 13.2; {path!r} "
            f"reports CUDA {release[0]}.{release[1]}"
        )
    return release


def _require_executable(path: str, origin: str) -> str:
    if not Path(path).is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(
            f"GC MoE needs an executable CUDA PTX assembler; {path!r} (from {origin}) "
            "is not one. Point it at ptxas from CUDA >= 13.2"
        )
    return path


def resolve_ptxas(explicit: str | None = None) -> str:
    """Explicit argument, then ``$GC_MOE_PTXAS``, then the CUDA toolkit.

    A named assembler is used or reported, never silently replaced by a
    different toolkit's; only the discovery step skips absent candidates.
    """
    if explicit:
        return _require_executable(explicit, "the ptxas= argument")
    requested = os.environ.get(PTXAS_ENV)
    if requested:
        return _require_executable(requested, f"${PTXAS_ENV}")
    candidates = []
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if cuda_home:
        candidates.append(str(Path(cuda_home) / "bin" / "ptxas"))
    candidates.append(shutil.which("ptxas"))
    candidates.append("/usr/local/cuda/bin/ptxas")
    checked = []
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.append(candidate)
        if Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "GC MoE could not find ptxas (needs CUDA >= 13.2); searched "
        + ", ".join(repr(path) for path in checked)
        + ". Install the CUDA toolkit or set GC_MOE_PTXAS."
    )


def prepare(explicit: str | None = None) -> str:
    """Pin ``GC_MOE_PRIMARY_CTX`` and export ``GC_MOE_PTXAS`` before import."""
    path = resolve_ptxas(explicit)
    ptxas_version(path)
    current = os.environ.get(PTXAS_ENV)
    if current is not None and os.path.realpath(current) != os.path.realpath(path):
        raise RuntimeError(
            f"GC MoE already compiled against {current!r}; a process cannot "
            f"switch the PTX assembler to {path!r} after the first launch"
        )
    os.environ[PTXAS_ENV] = path
    os.environ[PRIMARY_CTX_ENV] = "1"
    return path
