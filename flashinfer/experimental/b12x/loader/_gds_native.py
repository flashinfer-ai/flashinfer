"""Lazy native cuFile transport; importing the ordinary loader never loads GDS."""
from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading

_LOCK = threading.Lock()


def _strip_json_comments(text):
    """Drop // and /* */ comments outside strings; NVIDIA ships cufile.json with them."""
    out, i, n, in_string = [], 0, len(text), False
    while i < n:
        char = text[i]
        if in_string:
            out.append(char)
            if char == "\\" and i + 1 < n:
                out.append(text[i + 1])
                i += 1
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
            out.append(char)
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = n if end < 0 else end
            continue
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end < 0 else end + 2
            continue
        else:
            out.append(char)
        i += 1
    return "".join(out)


def _configure_cufile():
    configured = os.environ.get("CUFILE_ENV_PATH_JSON")
    source = Path(configured) if configured else Path("/etc/cufile.json")
    settings = (
        json.loads(_strip_json_comments(source.read_text()))
        if configured or source.exists()
        else {}
    )
    properties = settings.setdefault("properties", {})
    if "allow_compat_mode" in properties:
        return
    properties["allow_compat_mode"] = True
    payload = json.dumps(settings, sort_keys=True) + "\n"
    digest = hashlib.sha256(payload.encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "b12x" / "gds"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"cufile-{digest}.json"
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            staging = Path(tmp) / target.name
            staging.write_text(payload)
            os.replace(staging, target)
    os.environ["CUFILE_ENV_PATH_JSON"] = str(target)


@functools.cache
def load():
    with _LOCK:
        _configure_cufile()
        return _load()


def _load():
    source = Path(__file__).with_name("_gds_reader.c")
    shared = source.with_name("_row_plan.h")
    compiler = shlex.split(os.environ.get("CC", "cc"))
    if not compiler or not shutil.which(compiler[0]):
        raise RuntimeError("GDS disk backend needs a C compiler")
    compiler[0] = shutil.which(compiler[0])
    if "CUDA_HOME" in os.environ:
        cuda = Path(os.environ["CUDA_HOME"]).resolve()
        cflags, ldflags = [f"-I{cuda / 'include'}"], [f"-L{cuda / 'lib64'}", f"-Wl,-rpath,{cuda / 'lib64'}", "-lcufile"]
    elif shutil.which("pkg-config") and subprocess.run(["pkg-config", "--exists", "cufile"], capture_output=True).returncode == 0:
        include = subprocess.check_output(["pkg-config", "--variable=includedir", "cufile"], text=True).strip()
        cuda = Path(include).parent.resolve()
        cflags = shlex.split(subprocess.check_output(["pkg-config", "--cflags", "cufile"], text=True))
        ldflags = shlex.split(subprocess.check_output(["pkg-config", "--libs", "cufile"], text=True))
        libdir = subprocess.check_output(["pkg-config", "--variable=libdir", "cufile"], text=True).strip()
        ldflags.append(f"-Wl,-rpath,{libdir}")
    else:
        cuda = Path("/usr/local/cuda").resolve()
        cflags, ldflags = [f"-I{cuda / 'include'}"], [f"-L{cuda / 'lib64'}", f"-Wl,-rpath,{cuda / 'lib64'}", "-lcufile"]
    ldflags += ["-lcudart", "-lcuda"]
    headers = [cuda / "include" / name for name in ("cufile.h", "cuda.h", "cuda_runtime_api.h", "driver_types.h")]
    missing = [str(path) for path in headers if not path.is_file()]
    if missing:
        raise RuntimeError("GDS disk backend requires cuFile >= 1.14 development files; missing: " + ", ".join(missing))
    if b"cuFileSetParameterBool" not in headers[0].read_bytes():
        raise RuntimeError("GDS disk backend requires cuFile >= 1.14 parameter APIs")
    flags = ["-O2", "-std=c99", "-shared", "-fPIC", "-pthread", "-D_FILE_OFFSET_BITS=64", "-Wall", "-Wextra", "-Werror"]
    if b"batch_nvfs_submit_ops" in headers[0].read_bytes():
        flags.append("-DB12X_GDS_STATS=1")
    libraries = sorted((cuda / "lib64").glob("libcufile.so*")) or sorted((cuda / "lib").glob("libcufile.so*"))
    identity = dict(python=sys.version, soabi=sysconfig.get_config_var("SOABI"),
                    compiler=compiler, version=subprocess.check_output([*compiler, "--version"], text=True),
                    flags=flags, cflags=cflags, ldflags=ldflags,
                    libraries={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries},
                    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                        source, shared, source.with_name("_gds_checkpoint.c"),
                        source.with_name("_gds_owner.c"), source.with_name("_cuda_range.h"), *headers]})
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    cache = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "b12x" / "gds" / digest
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / ("_b12x_gds_reader" + sysconfig.get_config_var("EXT_SUFFIX"))
    if not target.exists():
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            output = Path(tmp) / target.name
            command = [*compiler, *flags, f"-I{sysconfig.get_path('include')}", *cflags,
                       str(source), *ldflags, "-o", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise RuntimeError("GDS native build failed:\n" + result.stderr)
            os.replace(output, target)
    spec = importlib.util.spec_from_file_location("_b12x_gds_reader", target)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
