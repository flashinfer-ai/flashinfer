"""Build exported PTX and its generated TVM FFI binding from source.

Requires an installed CUDA toolkit and Apache TVM FFI with ``embed_cubin``
support. Build products live in the supplied private workspace. This module
accepts no precompiled GPU binary and imports no source-generation framework.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_from_ptx(
    *,
    ptx_path: str | os.PathLike[str],
    binding_path: str | os.PathLike[str],
    module_ident: str,
    arch: str,
    ptxas_options: Sequence[str],
    workdir: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    include_paths: Sequence[str | os.PathLike[str]] = (),
    module_name: str | None = None,
) -> Any:
    """Freshly assemble exported PTX, compile its binding, and return its module.

    ``ptxas_options`` is the exact option list from the source manifest, e.g.
    ``["--register-usage-level=10"]``. Target/output flags belong to this
    builder. The generated binding owns the complete tensor and launch ABI.
    """
    started = time.monotonic()
    ptx_path = Path(ptx_path).resolve(strict=True)
    binding_path = Path(binding_path).resolve(strict=True)
    if ptx_path.suffix != ".ptx":
        raise ValueError("ptx_path must identify exported .ptx source")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", module_ident):
        raise ValueError("module_ident must be the generated binding's C++ identifier")
    ptx_bytes = ptx_path.read_bytes()
    binding_bytes = binding_path.read_bytes()
    ptx_source = ptx_bytes.decode("utf-8")
    binding_source = binding_bytes.decode("utf-8")
    target = re.search(r"(?m)^\s*\.target\s+([^\s,]+)", ptx_source)
    if target is None or target.group(1) != arch:
        raise ValueError(f"exported PTX .target must equal requested architecture {arch!r}")
    if isinstance(ptxas_options, str):
        raise TypeError("ptxas_options must be a sequence of individual arguments")
    options = tuple(ptxas_options)
    for option in options:
        if not isinstance(option, str) or not option.startswith("-"):
            raise ValueError("assembler options must be individual flag arguments")
        if option.startswith(("-arch", "--gpu-name", "-o", "--output-file", "--options-file")):
            raise ValueError(f"builder owns source, target and output selection: {option!r}")
    assembler = shutil.which("ptxas")
    if assembler is None:
        raise FileNotFoundError("installed CUDA toolkit ptxas is required")
    assembler = str(Path(assembler).resolve())
    include_dir = Path(assembler).parent.parent / "include"
    if not (include_dir / "cuda.h").is_file():
        raise FileNotFoundError(f"CUDA toolkit headers are absent from {include_dir}")
    target_includes = [str(Path(path).resolve(strict=True)) for path in include_paths]
    if any(not Path(path).is_dir() for path in target_includes):
        raise NotADirectoryError("include_paths must name existing target header directories")
    root = Path(workdir).resolve()
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    private = Path(tempfile.mkdtemp(prefix="ptx-build-", dir=root))
    receipt_path = Path(receipt_path)
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = {
        "ptx_sha256": _sha256(ptx_bytes),
        "binding_sha256": _sha256(binding_bytes),
        "arch": arch,
        "ptxas_options": list(options),
        "module_ident": module_ident,
        "target_include_paths": target_includes,
    }
    if module_name is None:
        module_name = "ptx_export_" + _sha256(json.dumps(identity, sort_keys=True).encode())[:24]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", module_name):
        raise ValueError("module_name must be a C++ identifier")
    record: dict[str, Any] = {
        "status": "started", "phase": "assembly", "source_identity": identity,
        "ptx_path": str(ptx_path), "binding_path": str(binding_path),
        "private_build_directory": str(private), "module_name": module_name,
        "assembler": assembler, "fresh_source_assembly": True,
    }

    def save() -> None:
        temporary = receipt_path.with_name(receipt_path.name + ".tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(receipt_path)

    try:
        source = private / ptx_path.name
        source.write_bytes(ptx_bytes)
        (private / "binding.cpp").write_bytes(binding_bytes)
        cubin_path = private / "kernel.cubin"
        version = subprocess.run([assembler, "--version"], check=True, text=True,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=20)
        record["assembler_version"] = version.stdout.strip()
        command = [assembler, "-v", *options, f"--gpu-name={arch}", str(source), "-o", str(cubin_path)]
        record["assembly_command"] = command
        save()
        t = time.monotonic()
        with (private / "ptxas.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        record["assembly_seconds"] = time.monotonic() - t
        record["assembly_exit_code"] = result.returncode
        record["assembly_log"] = str(private / "ptxas.log")
        result.check_returncode()
        cubin = cubin_path.read_bytes()
        if not cubin.startswith(b"\x7fELF"):
            raise RuntimeError("ptxas did not produce an ELF GPU image")
        record["assembled_cubin_sha256"] = _sha256(cubin)
        record["assembled_cubin_path"] = str(cubin_path)
        record["phase"] = "binding-build"
        save()
        import tvm_ffi
        from tvm_ffi import cpp

        record["tvm_ffi_version"] = tvm_ffi.__version__
        t = time.monotonic()
        artifact = cpp.build_inline(
            module_name,
            cpp_sources=binding_source,
            embed_cubin={module_ident: cubin},
            extra_include_paths=[str(include_dir), *target_includes],
            extra_ldflags=["-lcuda"],
            build_directory=str(private / "ffi"),
            backend="cuda",
        )
        record["binding_build_seconds"] = time.monotonic() - t
        record["artifact_path"] = str(artifact)
        record["phase"] = "module-load"
        save()
        t = time.monotonic()
        module = tvm_ffi.load_module(artifact)
        record["module_load_seconds"] = time.monotonic() - t
        record["status"] = "passed"
        record["phase"] = "complete"
        record["physical_payload_seconds"] = time.monotonic() - started
        save()
        return module
    except BaseException as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        record["physical_payload_seconds"] = time.monotonic() - started
        save()
        raise
