"""Build the host-tier S3-FIFO row-cache extension without a PyTorch link.

The C code (``_row_cache.c``) is pure CPython C-API: it touches only the
numpy/mapped-host buffers DiskRowCache binds into ``row_cache_create``, never
the torch C++ ABI, so the build needs nothing beyond ``Python.h``. Follows the
``_native.py`` pattern exactly: hash-keyed XDG cache directory,
``functools.cache`` + lock, compiler from ``CC`` (default ``cc``), and a
fail-loud ``RuntimeError`` -- the row cache never silently falls back.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
from pathlib import Path

_LOCK = threading.Lock()


def _build() -> Path:
    source = Path(__file__).with_name("_row_cache.c")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("b12x row cache needs a C99 compiler (CC or cc)")
    compiler[0] = shutil.which(compiler[0])
    includes = [Path(sysconfig.get_path("include"))]
    headers = [includes[0] / "Python.h", source]
    for path in headers:
        if not path.is_file():
            raise RuntimeError(f"b12x row cache build dependency is missing: {path}")
    flags = [
        "-O2",
        "-std=c99",
        "-shared",
        "-fPIC",
        "-D_FILE_OFFSET_BITS=64",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    version = subprocess.check_output([*compiler, "--version"], text=True)
    identity = {
        "abi": 1,
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "compiler": compiler,
        "version": version,
        "flags": flags,
        "headers": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in headers
        },
    }
    digest = hashlib.sha256(
        source.read_bytes() + json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = root / "b12x" / "loader" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("_b12x_row_cache" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            command = [
                *compiler,
                *flags,
                *(f"-I{p}" for p in includes),
                str(source),
                "-o",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError(
                    "b12x row cache native build failed:\n" + result.stderr
                )
            os.replace(output, target)
    return target


@functools.cache
def load():
    with _LOCK:
        spec = importlib.util.spec_from_file_location("_b12x_row_cache", _build())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.ABI_VERSION != 1:
            raise RuntimeError("b12x row cache native ABI mismatch")
        return module
