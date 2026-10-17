# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Optional external PTX assembly shared by all Frost operand dtypes.

The generated PTX and host launch wrapper are preserved. Only the GPU binary
initializer in standard LLVM-dialect IR is replaced before exporting a new object.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
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
from typing import NamedTuple
import warnings

_ENV = "FLASHINFER_CUDNN_FROST_PTXAS"
_FATBIN_MAGIC = b"\x50\xed\x55\xba"
_ELF_MAGIC = b"\x7fELF"
_EXPORT_LOCK = threading.RLock()


class _CompilerSnapshot(NamedTuple):
    identity: tuple[tuple[str, str], ...]
    key: str


_COMPILER_SCOPE: ContextVar[_CompilerSnapshot | None] = ContextVar(
    "frost_compiler_scope", default=None
)


def capture_compiler_snapshot() -> _CompilerSnapshot:
    """Freeze the selected assembler for a built engine until explicit refresh.

    Source-runtime entry points keep checking their own compiler selection.
    Only engine-owned preparation uses this snapshot instead of filesystem
    inspection; cold compilation independently verifies the actual assembler.
    """
    identity = tuple(sorted(_read_compiler_identity().items()))
    return _CompilerSnapshot(identity, _serialized_identity(identity))


@contextmanager
def use_compiler_snapshot(snapshot: _CompilerSnapshot):
    """Run a cold engine preparation with its immutable compiler selection."""
    token = _COMPILER_SCOPE.set(snapshot)
    try:
        yield
    finally:
        _COMPILER_SCOPE.reset(token)


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
    snapshot = _COMPILER_SCOPE.get()
    if snapshot is not None:
        return dict(snapshot.identity)
    return _read_compiler_identity()


def _read_compiler_identity() -> dict[str, str]:
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
    snapshot = _COMPILER_SCOPE.get()
    if snapshot is not None:
        return snapshot.key
    identity = compiler_identity()
    return _serialized_identity(tuple(sorted(identity.items())))


@functools.lru_cache(maxsize=16)
def _serialized_identity(identity: tuple[tuple[str, str], ...]) -> str:
    values = dict(identity)
    if values["backend"] == "bundled":
        return ""
    return json.dumps(values, sort_keys=True, separators=(",", ":"))


def _fresh_compiler_identity() -> dict[str, str]:
    # Compilation is a cold path that must detect changes during compilation,
    # even when its caller is using an engine's fixed snapshot.
    token = _COMPILER_SCOPE.set(None)
    try:
        return compiler_identity()
    finally:
        _COMPILER_SCOPE.reset(token)


def require_moe_assembler() -> None:
    """Require CUDA 13.4 or newer before admitting a MoELayer candidate.

    External and bundled assemblers use the same release minimum. CUDA_HOME
    does not select the assembler; FLASHINFER_CUDNN_FROST_PTXAS does.
    Source runtimes inspect the selected stack directly. MoE engines validate
    their fixed compiler snapshot when built or explicitly refreshed.
    """
    try:
        identity = compiler_identity()
        if identity["backend"] == "external_ptxas":
            match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", identity["version"])
            if match is None:
                raise ValueError("cannot identify the selected PTXAS release")
            version = tuple(map(int, match.groups()))
        else:
            from cutlass import CUDA_VERSION

            version = (int(CUDA_VERSION.major), int(CUDA_VERSION.minor))
        if version >= (13, 4):
            return
        reason = f"selected {identity['backend']} assembler is CUDA {version[0]}.{version[1]}"
    except (
        ImportError,
        AttributeError,
        OSError,
        RuntimeError,
        ValueError,
        subprocess.SubprocessError,
    ) as exc:
        reason = str(exc)
    raise NotImplementedError(
        "cuDNN Frost MoELayer requires PTXAS from CUDA 13.4 or newer; "
        f"{reason}. Set FLASHINFER_CUDNN_FROST_PTXAS to a compatible ptxas "
        "executable, or install a CuTe DSL stack bundling CUDA 13.4 or newer. "
        "Setting CUDA_HOME alone does not select the assembler."
    )


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
        if _fresh_compiler_identity() != {"backend": "bundled"}:
            raise RuntimeError(
                "Frost assembler changed before compilation; refresh or rebuild "
                "the MoE runner before preparing a new shape"
            )
        compiled = module.compile()
        if _fresh_compiler_identity() != {"backend": "bundled"}:
            raise RuntimeError(
                "Frost assembler changed during compilation; refresh or rebuild "
                "the MoE runner before preparing a new shape"
            )
        return compiled
    identity = json.loads(compiler_key)
    if identity != _fresh_compiler_identity():
        raise RuntimeError(
            "External Frost assembler changed before compilation; refresh or "
            "rebuild the MoE runner before preparing a new shape"
        )
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
        if identity != _fresh_compiler_identity():
            raise RuntimeError(
                "External Frost assembler changed during compilation; refresh or "
                "rebuild the MoE runner before preparing a new shape"
            )
        # Verify the IR contract now even when persistent export is unavailable.
        with _binary_initializer(compiled, cubin):
            pass
        return _ExternalCompiledKernel(compiled, cubin)
