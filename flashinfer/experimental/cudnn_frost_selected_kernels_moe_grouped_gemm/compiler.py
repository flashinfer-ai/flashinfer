# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Optional external PTX assembly shared by all Frost operand dtypes.

The generated PTX and host launch wrapper are preserved. Only the GPU binary
initializer in standard LLVM-dialect IR is replaced before exporting a new object.
"""

from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import threading
import warnings

_ENV = "FLASHINFER_CUDNN_FROST_PTXAS"
_FATBIN_MAGIC = b"\x50\xed\x55\xba"
_ELF_MAGIC = b"\x7fELF"
_EXPORT_LOCK = threading.RLock()


@functools.lru_cache(maxsize=16)
def _executable_identity(path: str, stat_key: tuple[int, ...]) -> tuple[str, str]:
    # The stat key invalidates this expensive inspection after in-place upgrades.
    del stat_key
    hasher = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    result = subprocess.run(
        [path, "--version"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip(), digest


def compiler_identity() -> dict[str, str]:
    """Return the selected assembler's CPU-only, JSON-serializable identity."""
    configured = os.environ.get(_ENV)
    if not configured:
        return {"backend": "bundled"}
    path = Path(configured).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"{_ENV} must name an executable PTXAS file: {path}")
    info = path.stat()
    version, digest = _executable_identity(
        str(path),
        (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns),
    )
    return dict(
        backend="external_ptxas", path=str(path), version=version, sha256=digest
    )


def identity_key() -> str:
    """A stable cache component; the default compiler keeps its existing key."""
    identity = compiler_identity()
    if identity["backend"] == "bundled":
        return ""
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


@contextmanager
def _binary_initializer(compiled, cubin: bytes):
    from cutlass._mlir import ir

    if not cubin.startswith(_ELF_MAGIC):
        raise RuntimeError("External PTXAS did not produce an ELF CUBIN")
    module = compiled.ir_module
    candidates = []

    def inspect(op):
        if op.name == "llvm.mlir.global" and "value" in op.attributes:
            value = op.attributes["value"]
            if isinstance(value, ir.StringAttr):
                data = ir.StringAttr(value).value_bytes
                if data.startswith((_FATBIN_MAGIC, _ELF_MAGIC)):
                    candidates.append(op)
        return ir.WalkResult.ADVANCE

    with _EXPORT_LOCK, module.context:
        module.operation.walk(inspect)
        if len(candidates) != 1:
            raise RuntimeError(
                "External Frost assembly requires exactly one LLVM GPU binary "
                f"initializer, found {len(candidates)}"
            )
        op = candidates[0]
        previous_value = op.attributes["value"]
        previous_type = op.attributes["global_type"]
        try:
            op.attributes["value"] = ir.StringAttr.get(cubin)
            op.attributes["global_type"] = ir.TypeAttr.get(
                ir.Type.parse(f"!llvm.array<{len(cubin)} x i8>")
            )
            module.operation.verify()
            yield
        finally:
            op.attributes["value"] = previous_value
            op.attributes["global_type"] = previous_type
            module.operation.verify()


class _ExternalCompiledKernel:
    """Exportable adapter whose fallback callable also uses the new CUBIN."""

    def __init__(self, compiled, cubin: bytes):
        self._compiled = compiled
        self._cubin = cubin
        self._function = None
        self._lock = threading.RLock()

    def export_to_c(self, object_file_path, function_name=None, **kwargs):
        with _binary_initializer(self._compiled, self._cubin):
            return self._compiled.export_to_c(
                object_file_path, function_name=function_name, **kwargs
            )

    def __tvm_ffi_object__(self):
        # A persistent cache miss normally exports and reloads through JitSpec.
        # Disabled caching and failed persistence must also load the new binary,
        # never fall back to the original in-process CuTe executable.
        with self._lock:
            if self._function is None:
                from cutlass.runtime import load_module

                with tempfile.TemporaryDirectory(
                    prefix="flashinfer_frost_ptxas_"
                ) as work:
                    path = Path(work) / "kernel.o"
                    symbol = "flashinfer_frost_external_kernel"
                    self.export_to_c(str(path), function_name=symbol)
                    module = load_module(str(path), enable_tvm_ffi=True)
                    self._function = getattr(module, symbol)
            return self._function

    def __call__(self, *args):
        return self.__tvm_ffi_object__()(*args)


def compile_module(module, arch: str, compiler_key: str):
    """Compile a frozen module, optionally assembling its unchanged PTX externally."""
    if not compiler_key:
        return module.compile()
    identity = json.loads(compiler_key)
    if identity != compiler_identity():
        raise RuntimeError("External Frost assembler changed before compilation")
    options = getattr(module, "frost_compile_options", None)
    if not isinstance(options, str):
        raise RuntimeError(
            "Frost source must expose its compile options for external PTXAS"
        )
    with tempfile.TemporaryDirectory(prefix="flashinfer_frost_ptxas_") as work:
        directory = Path(work)
        module.frost_compile_options = (
            options + " --keep-ptx --dump-dir " + shlex.quote(str(directory))
        )
        try:
            compiled = module.compile()
        finally:
            module.frost_compile_options = options
        ptx = compiled.__ptx__
        if not isinstance(ptx, str):
            raise RuntimeError("CuTe did not return PTX through --keep-ptx")
        target = re.search(r"^\s*\.target\s+(\w+)", ptx, re.MULTILINE)
        if target is None or target[1] != arch:
            raise RuntimeError(
                "Frost PTX target differs from the requested architecture"
            )
        ptx_path, cubin_path = directory / "kernel.ptx", directory / "kernel.cubin"
        ptx_path.write_text(ptx)
        result = subprocess.run(
            [
                identity["path"],
                "--gpu-name",
                arch,
                str(ptx_path),
                "--output-file",
                str(cubin_path),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(
                f"External Frost PTXAS failed ({result.returncode}): "
                f"{result.stdout}{result.stderr}"
            )
        if result.stderr.strip():
            warnings.warn(
                f"External Frost PTXAS: {result.stderr.strip()}", stacklevel=2
            )
        cubin = cubin_path.read_bytes()
        if identity != compiler_identity():
            raise RuntimeError("External Frost assembler changed during compilation")
        # Verify the IR contract now even when persistent export is unavailable.
        with _binary_initializer(compiled, cubin):
            pass
        return _ExternalCompiledKernel(compiled, cubin)
