"""CuTe compilation identity, device target and system PTX assembly."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import inspect
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading

import cutlass.cute as cute
from cutlass.base_dsl.dsl import BaseDSL
from cutlass._mlir._mlir_libs import _cutlass_ir
from cutlass.base_dsl.compiler import (
    CompileOptions,
    DeviceTarget,
    ExtraCompilerOpts,
    GPUArch,
    LinkLibraries,
    OptLevel,
    PtxasOptions,
    RDC,
)


SYSTEM_PTXAS = os.environ.get('GC_MOE_PTXAS', '/usr/local/cuda/bin/ptxas')
PTXAS_OPTIONS = ('-O3', '--register-usage-level=10', '--fmad=true')
COMPILE_LOCK = threading.RLock()


@dataclass(frozen=True)
class CompilerIdentity:
    target: str
    compile_options: str
    ptxas_options: tuple[str, ...]
    tool_realpath: str
    tool_stamp: tuple[int, ...]
    tool_version: str
    tool_sha256: str
    environment: str
    effective_options: str
    frontend_sha256: str
    source_sha256: str
    native_library: tuple
    bridge_revision: int = 3


@lru_cache(maxsize=16)
def _tool_identity(realpath: str, stamp: tuple[int, ...]) -> tuple[str, str]:
    result = subprocess.run([SYSTEM_PTXAS, '--version'], check=False,
                            capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('cannot query the system PTX assembler')
    return ((result.stdout + result.stderr).strip(),
            hashlib.sha256(Path(realpath).read_bytes()).hexdigest())


@lru_cache(maxsize=16)
def _binary_identity(realpath: str, stamp: tuple[int, ...]) -> str:
    return hashlib.sha256(Path(realpath).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def _source_identity(source_root: str, stamps: tuple) -> str:
    digest = hashlib.sha256()
    for name, *_ in stamps:
        path = Path(name)
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def compiler_identity(target: str, entrypoint, source_root: Path,
                      kernel_name: str) -> CompilerIdentity:
    tool = Path(SYSTEM_PTXAS)
    if not tool.is_file() or not os.access(tool, os.X_OK):
        raise RuntimeError(f"an executable system PTX assembler is required at {tool}")
    info = tool.stat()
    stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    realpath = str(tool.resolve())
    version, digest = _tool_identity(realpath, stamp)
    assembly_options = ' '.join(PTXAS_OPTIONS)
    options = (f"--gpu-arch {target} --opt-level 3 "
               f"--ptxas-options {shlex.quote(assembly_options)}")
    function = inspect.unwrap(entrypoint)
    with COMPILE_LOCK:
        BaseDSL._lazy_initialize_dsl(function)
        settings = function._dsl_object.envar
    effective = CompileOptions((GPUArch(target), OptLevel(3),
                                PtxasOptions(assembly_options)))
    effective.apply_envar_settings(settings, kernel_name)
    if (effective.options[GPUArch].value != target or
            effective.options[OptLevel].value != 3):
        raise RuntimeError('compiler environment overrides the required target or optimization level')
    if (effective.options[RDC].value or effective.options[DeviceTarget].value or
            effective.options[LinkLibraries].value):
        raise RuntimeError('kernels require standalone kernel compilation without device linking')
    extra = effective.options[ExtraCompilerOpts].value
    if re.search(r'cubin-chip|gpu-arch|opt-level|ptx-options|fast.?math|unsafe.?fp|'
                 r'ftz|fmad|contract|prec.?div|prec.?sqrt|device-debug', extra, re.IGNORECASE):
        raise RuntimeError('raw compiler overrides conflict with the kernel compilation policy')
    normalized = effective.to_str()
    assembly = tuple(shlex.split(effective.options[PtxasOptions].value.strip("'")))
    allowed = {*PTXAS_OPTIONS, '-v', '--verbose', '--warn-on-spills',
               '--warn-on-local-memory-usage', '--generate-line-info'}
    if (any(assembly.count(option) != 1 for option in PTXAS_OPTIONS) or
            any(option not in allowed for option in assembly)):
        raise RuntimeError('assembler options conflict with the kernel compilation policy')
    if effective.generate_line_info and '--generate-line-info' not in assembly:
        assembly += ('--generate-line-info',)
    source_root = source_root.resolve()
    stamps = []
    for path in sorted(source_root.rglob('*.py')):
        stat = path.stat()
        stamps.append((str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    native = Path(_cutlass_ir.__file__).resolve()
    native_stat = native.stat()
    native_stamp = (native_stat.st_dev, native_stat.st_ino, native_stat.st_size,
                    native_stat.st_mtime_ns)
    native_identity = (str(native), native_stamp, _binary_identity(str(native), native_stamp))
    return CompilerIdentity(
        target, options, assembly, realpath, stamp, version, digest,
        settings.cache_key_str(), normalized,
        function._dsl_object.get_version().hexdigest(), _source_identity(str(source_root), tuple(stamps)),
        native_identity)


def device_compile_target(capability: tuple[int, int]) -> str:
    major, minor = capability
    suffix = 'f' if major > 9 else 'a' if major == 9 else ''
    return f"sm_{major * 10 + minor}{suffix}"


def ptx_target(ptx: str) -> str:
    targets = re.findall(r"(?m)^\.target[ \t]+(sm_[0-9]+[af]?)(?:[ \t]*,.*)?[ \t]*$", ptx)
    if len(targets) != 1:
        raise RuntimeError('PTX target header is missing or ambiguous')
    return targets[0]


def assemble_ptx(ptx: str, dump_dir: str,
                 basename: str, *, target: str,
                 ptxas_options: tuple[str, ...] = PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    if ptx_target(ptx) != target:
        raise RuntimeError('PTX target differs from the selected device target')
    ptx_version = re.search(r"(?m)^\.version[ \t]+([0-9]+)\.([0-9]+)[ \t]*$",
                            ptx)
    if ptx_version is None:
        raise RuntimeError('PTX version header is missing')
    minimum_release = ((13, 2) if
                       (int(ptx_version.group(1)), int(ptx_version.group(2)))
                       >= (9, 1) else (13, 0))
    # Assemble with the system CUDA toolchain.
    tool = SYSTEM_PTXAS
    required = (f"CUDA {minimum_release[0]}.{minimum_release[1]} or newer "
                f"ptxas is required at {tool} for this PTX version")
    if not Path(tool).is_file() or not os.access(tool, os.X_OK):
        raise RuntimeError(required)
    result = subprocess.run([tool, '--version'], check=False,
                            capture_output=True, text=True)
    version = (result.stdout + result.stderr).strip()
    release = re.search(r"release[ \t]+([0-9]+)\.([0-9]+)", version)
    if (result.returncode != 0 or release is None or
            (int(release.group(1)), int(release.group(2))) < minimum_release):
        raise RuntimeError(required)
    source = Path(dump_dir) / f"{basename}.ptx"
    output = Path(dump_dir) / f"{basename}.cubin"
    source.write_text(ptx)
    assembled = subprocess.run([tool, f'-arch={target}', *ptxas_options,
                                '-o', str(output), str(source)],
                               capture_output=True, text=True)
    if assembled.returncode:
        raise RuntimeError(f"{basename} assembly failed: " +
                           assembled.stderr)
    return ptx, output.read_bytes(), version


def compile_kernel(kernel, arguments: tuple, identity: CompilerIdentity,
                   assemble) -> bytes:
    """Isolate frontend artifacts and return the assembled launch binary."""
    with tempfile.TemporaryDirectory(prefix='gc_moe_cute_role_') as dump_dir:
        compiled = cute.compile(
            kernel, *arguments,
            options=identity.compile_options + f" --keep-ptx --dump-dir {dump_dir}")
        _, cubin, _ = assemble(compiled.artifacts.PTX, dump_dir)
    return cubin
