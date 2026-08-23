# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""JIT loader for the source-only Cake GDN CP-prefill backend."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from filelock import FileLock
from tvm_ffi import cpp

from . import env as jit_env
from .cpp_ext import get_cuda_path, get_nvcc_parallelism_flags

CakeGDNCPArch = Literal["sm_100a", "sm_103a"]

_EXPORT_SCHEMA = "flashinfer-pr4078-cake-only-standalone-export-v3"
_MANIFEST_SHA256 = "888ef5f5c2bab79606439273deae22a229a642de9ea8e3ebc15c860c129409c3"
_BASELINE_REVISION = "6cb2e70995d92edbc443b1bfc317ecacac907640"
_FOCUS_CONTRACT = (
    150,
    "d4f3fad233af91b8afac35271d6848df8f0f090b08f17807b9e2830139dd37ab",
)
_FULL_CONTRACT = (
    822,
    "0dff83c89b9a17f67e0a2db9bb9c20ed77506fa3b38cc55d7772864021553592",
)


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "gdn" / "cake"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "gdn" / "cake"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "frozen CAKE GDN CP-prefill sources were not found; checked "
        f"{installed} and {checkout}"
    )


@functools.cache
def cake_gdn_cp_nvcc_version() -> tuple[int, int]:
    """Return the version of the nvcc executable used by the Cake JIT."""

    nvcc = Path(get_cuda_path()) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"nvcc was not found at {nvcc}")
    result = subprocess.run(
        [str(nvcc), "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"failed to query the Cake GDN CP-prefill nvcc at {nvcc}:\n{result.stdout}"
        )
    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", result.stdout)
    if match is None:
        raise RuntimeError(
            f"could not parse the Cake GDN CP-prefill nvcc version at {nvcc}:\n"
            f"{result.stdout}"
        )
    return int(match.group(1)), int(match.group(2))


@functools.cache
def _manifest() -> dict[str, Any]:
    path = _source_dir() / "manifest.json"
    observed_digest = _sha256(path)
    if observed_digest != _MANIFEST_SHA256:
        raise RuntimeError(
            f"Cake GDN CP-prefill manifest drift at {path}: "
            f"expected {_MANIFEST_SHA256}, got {observed_digest}"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    support = manifest.get("support_contract", {})
    focus = support.get("focus_contract", {})
    full = support.get("full_regression_contract", {})
    observed = (
        manifest.get("schema"),
        manifest.get("baseline_revision"),
        support.get("external_fallbacks_allowed"),
        (focus.get("row_count"), focus.get("canonical_stream_sha256")),
        (full.get("row_count"), full.get("canonical_stream_sha256")),
    )
    expected = (
        _EXPORT_SCHEMA,
        _BASELINE_REVISION,
        0,
        _FOCUS_CONTRACT,
        _FULL_CONTRACT,
    )
    if observed != expected:
        raise RuntimeError(
            "Cake GDN CP-prefill manifest does not match the ratified v3 "
            f"support contract: expected {expected!r}, got {observed!r}"
        )
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kernel_record(name: str) -> dict[str, Any]:
    records = [record for record in _manifest()["kernels"] if record["name"] == name]
    if len(records) != 1:
        raise ValueError(f"unknown CAKE GDN CP-prefill kernel: {name!r}")
    return records[0]


def _cuda_record(record: dict[str, Any], arch: CakeGDNCPArch) -> dict[str, Any]:
    outputs = [
        output for output in record["outputs"] if arch in output["architectures"]
    ]
    if len(outputs) != 1:
        raise ValueError(f"kernel {record['name']!r} does not support {arch}")
    return outputs[0]


def _compile_cubin(source: Path, *, arch: CakeGDNCPArch, digest: str) -> bytes:
    cache_dir = jit_env.FLASHINFER_JIT_DIR / "cake_gdn_cp_prefill" / arch
    cache_dir.mkdir(parents=True, exist_ok=True)
    cubin = cache_dir / f"{source.stem}-{digest[:16]}.cubin"
    lock = FileLock(f"{cubin}.lock", thread_local=False)
    with lock:
        if not cubin.exists():
            nvcc = Path(get_cuda_path()) / "bin" / "nvcc"
            if not nvcc.is_file():
                raise RuntimeError(f"nvcc was not found at {nvcc}")
            with tempfile.NamedTemporaryFile(
                dir=cache_dir,
                prefix=f".{cubin.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
            try:
                command = [
                    str(nvcc),
                    "--cubin",
                    "--std=c++17",
                    "-O3",
                    f"--gpu-architecture={arch}",
                    *get_nvcc_parallelism_flags(),
                    str(source),
                    "-o",
                    str(temporary),
                ]
                result = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"failed to compile {source.name} for {arch}:\n{result.stdout}"
                    )
                os.replace(temporary, cubin)
            finally:
                temporary.unlink(missing_ok=True)
    return cubin.read_bytes()


@functools.cache
def load_cake_gdn_cp_kernel(name: str, arch: CakeGDNCPArch):
    """Compile and load one checksum-verified Cake kernel binding."""

    if arch not in ("sm_100a", "sm_103a"):
        raise ValueError(f"unsupported CAKE GDN CP-prefill architecture: {arch!r}")
    record = _kernel_record(name)
    cuda = _cuda_record(record, arch)
    host = record["host_binding"]
    root = _source_dir()
    cuda_path = root / cuda["path"]
    host_path = root / host["path"]
    headers = _manifest().get("cuda_headers", [])
    sources = [
        (cuda_path, cuda["sha256"]),
        (host_path, host["sha256"]),
        *((root / header["path"], header["sha256"]) for header in headers),
    ]
    for path, expected in sources:
        observed = _sha256(path)
        if observed != expected:
            raise RuntimeError(
                f"Cake GDN CP-prefill source drift at {path}: expected {expected}, got {observed}"
            )
    compile_digest = hashlib.sha256(
        "\0".join([cuda["sha256"], *(header["sha256"] for header in headers)]).encode()
    ).hexdigest()
    cubin = _compile_cubin(cuda_path, arch=arch, digest=compile_digest)
    module = cpp.load_inline(
        f"flashinfer_cake_gdn_cp_{name}_{arch}_{compile_digest[:12]}",
        cpp_sources=host_path.read_text(encoding="utf-8"),
        embed_cubin={host["module_ident"]: cubin},
        extra_include_paths=[
            str(Path(get_cuda_path()) / "include"),
            str(root.parents[1]),
            str(root.parents[2] / "include"),
        ],
        extra_ldflags=["-lcuda"],
    )
    return module[host["entry"]]


__all__ = [
    "CakeGDNCPArch",
    "cake_gdn_cp_nvcc_version",
    "load_cake_gdn_cp_kernel",
]
