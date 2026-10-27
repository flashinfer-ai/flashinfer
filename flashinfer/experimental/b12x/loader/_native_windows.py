"""Build the Windows checkpoint reader against CPython and CUDA, without Torch headers."""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile


@functools.cache
def dll_directories():
    from flashinfer.jit.cpp_ext import get_cuda_path

    cuda = Path(get_cuda_path())
    return tuple(
        os.add_dll_directory(str(path))
        for path in (cuda / "bin", cuda / "bin/x64", cuda / "lib/x64")
        if path.is_dir()
    )


def build() -> Path:
    from flashinfer.jit.cpp_ext import get_cuda_path

    compiler = shutil.which("cl")
    if compiler is None:
        raise RuntimeError(
            "b12x.loader requires an MSVC developer shell matching Python's architecture"
        )
    cuda = Path(get_cuda_path()).resolve()
    source = Path(__file__).with_name("_storage_windows.cpp")
    python = Path(sys.base_prefix)
    flags = ["/nologo", "/LD", "/O2", "/MD", "/std:c++17", "/EHsc", "/W4", "/WX"]
    version = subprocess.run([compiler], capture_output=True, text=True)
    identity = json.dumps(
        {
            "python": sys.version,
            "platform": sysconfig.get_platform(),
            "compiler": compiler,
            "version": version.stdout + version.stderr,
            "cuda": str(cuda),
            "flags": flags,
        },
        sort_keys=True,
    ).encode()
    digest = hashlib.sha256(
        source.read_bytes() + source.with_name("_cuda_range.h").read_bytes() + identity
    ).hexdigest()[:24]
    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "b12x/loader"
        / digest
    )
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("_b12x_loader_storage" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as temporary:
            output = Path(temporary) / target.name
            command = [
                compiler,
                *flags,
                f"/I{python / 'include'}",
                f"/I{cuda / 'include'}",
                str(source),
                f"/Fe:{output}",
                f"/Fo:{Path(temporary) / 'reader.obj'}",
                "/link",
                f"/LIBPATH:{python / 'libs'}",
                f"/LIBPATH:{cuda / 'lib/x64'}",
                "cudart.lib",
                "cuda.lib",
                "ntdll.lib",
            ]
            result = subprocess.run(
                command, cwd=temporary, capture_output=True, text=True
            )
            if result.returncode:
                raise RuntimeError(
                    "b12x.loader Windows build failed:\n"
                    + result.stdout
                    + result.stderr
                )
            os.replace(output, target)
    return target
