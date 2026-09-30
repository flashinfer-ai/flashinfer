"""Build the C99 I/O helper without linking the PyTorch C++ ABI."""

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
    import torch

    source = Path(__file__).with_name("_storage.c")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("b12x.loader needs a C99 compiler (CC or cc)")
    compiler[0] = shutil.which(compiler[0])
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")).resolve()
    includes = [
        Path(sysconfig.get_path("include")),
        cuda / "include",
    ]
    headers = [
        includes[0] / "Python.h",
        includes[1] / "cuda_runtime_api.h",
        includes[1] / "driver_types.h",
    ]
    for path in headers:
        if not path.is_file():
            raise RuntimeError(f"b12x.loader build dependency is missing: {path}")
    flags = [
        "-O2",
        "-std=c99",
        "-shared",
        "-fPIC",
        "-pthread",
        "-D_FILE_OFFSET_BITS=64",
        "-Wall",
        "-Wextra",
        "-Werror",
    ]
    # liburing is optional for generic checkpoint transport and resident PLE.
    uring_cflags: list[str] = []
    uring_ldflags: list[str] = []
    uring_version = None
    pkg_config = shutil.which("pkg-config")
    if pkg_config:
        available = subprocess.run(
            [pkg_config, "--exists", "liburing"], capture_output=True
        )
        if available.returncode == 0:
            uring_cflags = shlex.split(
                subprocess.check_output([pkg_config, "--cflags", "liburing"], text=True)
            )
            uring_ldflags = shlex.split(
                subprocess.check_output([pkg_config, "--libs", "liburing"], text=True)
            )
            uring_version = subprocess.check_output(
                [pkg_config, "--modversion", "liburing"], text=True
            ).strip()
            flags.extend(["-DB12X_HAVE_LIBURING=1", *uring_cflags])
    version = subprocess.check_output([*compiler, "--version"], text=True)
    identity = {
        "abi": 1,
        "python": sys.version,
        "soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "compiler": compiler,
        "version": version,
        "flags": flags,
        "uring_ldflags": uring_ldflags,
        "uring_version": uring_version,
        "cuda": str(cuda),
        "torch": torch.__version__,
        "headers": {
            str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in headers
        },
    }
    digest = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((*source.parent.glob("*.c"), *source.parent.glob("*.h"))))
        + json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    cache = root / "b12x" / "loader" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("_b12x_loader_storage" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            command = [
                *compiler,
                *flags,
                *(f"-I{p}" for p in includes),
                str(source),
                f"-L{cuda / 'lib64'}",
                f"-Wl,-rpath,{cuda / 'lib64'}",
                "-lcudart",
                "-lcuda",
                *uring_ldflags,
                "-o",
                str(output),
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("b12x.loader native build failed:\n" + result.stderr)
            os.replace(output, target)
    return target


@functools.cache
def load():
    with _LOCK:
        spec = importlib.util.spec_from_file_location("_b12x_loader_storage", _build())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if module.ABI_VERSION != 1:
            raise RuntimeError("b12x.loader native ABI mismatch")
        return module
