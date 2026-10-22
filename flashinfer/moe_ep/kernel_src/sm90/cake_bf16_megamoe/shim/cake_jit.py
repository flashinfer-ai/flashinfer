"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from ......jit import env as jit_env
from ......jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags

__all__ = [
    "GROUPED_GEMM_STAGES",
    "gen_sm90_cake_bf16_combine_prereduced_module",
    "gen_sm90_cake_bf16_combine_tail_module",
    "gen_sm90_cake_bf16_combine_tail_prereduced_module",
    "gen_sm90_cake_bf16_compact_module",
    "gen_sm90_cake_bf16_dispatch_module",
    "gen_sm90_cake_bf16_grouped_gemm_module",
    "grouped_gemm_manifest",
    "grouped_gemm_record",
    "sm90_cake_bf16_combine_prereduced_uri",
    "sm90_cake_bf16_combine_tail_prereduced_uri",
    "sm90_cake_bf16_combine_tail_uri",
    "sm90_cake_bf16_compact_uri",
    "sm90_cake_bf16_dispatch_uri",
]

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
# The protocol header is read from the vendored push package and snapshotted
# next to this package's sources so each JIT translation unit is self-contained.
_PROTOCOL_HEADER = (
    Path(__file__).resolve().parents[2]
    / "push_style_megamoe"
    / "src"
    / "a2a"
    / "sm90_push_a2a.cuh"
)
_Sources = tuple[tuple[str, Path], ...]
# One JIT module per protocol kernel of this package.
_COMPACT_SOURCES: _Sources = (
    ("cake_compact_bf16.cu", _SOURCE_DIR / "cake_compact_bf16.cu"),
    ("sm90_push_a2a.cuh", _PROTOCOL_HEADER),
)
_COMBINE_TAIL_SOURCES: _Sources = (
    ("cake_combine_tail_bf16.cu", _SOURCE_DIR / "cake_combine_tail_bf16.cu"),
    ("sm90_push_a2a.cuh", _PROTOCOL_HEADER),
)
_DISPATCH_SOURCES: _Sources = (
    ("cake_dispatch_fused_bf16.cu", _SOURCE_DIR / "cake_dispatch_fused_bf16.cu"),
    ("sm90_push_a2a.cuh", _PROTOCOL_HEADER),
)
_COMBINE_PREREDUCED_SOURCES: _Sources = (
    (
        "cake_combine_prereduced_bf16.cu",
        _SOURCE_DIR / "cake_combine_prereduced_bf16.cu",
    ),
    ("sm90_push_a2a.cuh", _PROTOCOL_HEADER),
)
_COMBINE_TAIL_PREREDUCED_SOURCES: _Sources = (
    (
        "cake_combine_tail_prereduced_bf16.cu",
        _SOURCE_DIR / "cake_combine_tail_prereduced_bf16.cu",
    ),
    ("sm90_push_a2a.cuh", _PROTOCOL_HEADER),
)


def _canonical_source(source: bytes) -> bytes:
    return source.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _source_blobs(sources: _Sources) -> dict[str, bytes]:
    return {name: _canonical_source(path.read_bytes()) for name, path in sources}


def _snapshot_matches(path: Path, content: str) -> bool:
    try:
        with path.open("r", encoding="utf-8", newline="") as source:
            return source.read() == content
    except (FileNotFoundError, IsADirectoryError, PermissionError):
        return False


def _write_snapshot_atomic(path: Path, content: str) -> None:
    if _snapshot_matches(path, content):
        return
    temp_path: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as snapshot:
            snapshot.write(content)
        os.replace(temp_path, path)
        temp_path = None
    except PermissionError:
        if not _snapshot_matches(path, content):
            raise
    finally:
        if temp_path is not None:
            with contextlib.suppress(FileNotFoundError):
                temp_path.unlink()


def _module_uri(
    prefix: str, sources: _Sources, cuda_flags: tuple[str, ...] | None
) -> str:
    """Content-addressed module name over the sources and the CUDA flags."""
    flags = tuple(sm90a_nvcc_flags) if cuda_flags is None else cuda_flags
    digest = hashlib.sha256()
    for name, source in _source_blobs(sources).items():
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(source)
        digest.update(b"\0")
    digest.update(json.dumps(flags, separators=(",", ":")).encode())
    return f"{prefix}_{digest.hexdigest()[:20]}"


def _gen_module(prefix: str, sources: _Sources, translation_unit: str) -> JitSpec:
    """Snapshot the sources into the writable JIT cache and return the spec."""
    flags = tuple(sm90a_nvcc_flags)
    uri = _module_uri(prefix, sources, flags)
    output_dir = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    for name, source in _source_blobs(sources).items():
        _write_snapshot_atomic(output_dir / name, source.decode())
    return gen_jit_spec(
        uri,
        [output_dir / translation_unit],
        extra_cuda_cflags=list(flags),
        extra_include_paths=[output_dir],
    )


def sm90_cake_bf16_compact_uri(cuda_flags: tuple[str, ...] | None = None) -> str:
    """Content-addressed name of the BF16 inbox compaction module."""
    return _module_uri("sm90_cake_bf16_compact", _COMPACT_SOURCES, cuda_flags)


def gen_sm90_cake_bf16_compact_module() -> JitSpec:
    """JIT spec of ``sm90_cake_compact_bf16`` (inbox rows -> expert-major bf16 matrix)."""
    return _gen_module(
        "sm90_cake_bf16_compact", _COMPACT_SOURCES, "cake_compact_bf16.cu"
    )


def sm90_cake_bf16_combine_tail_uri(
    cuda_flags: tuple[str, ...] | None = None,
) -> str:
    """Content-addressed name of the fused combine-tail module."""
    return _module_uri("sm90_cake_bf16_combine_tail", _COMBINE_TAIL_SOURCES, cuda_flags)


def gen_sm90_cake_bf16_combine_tail_module() -> JitSpec:
    """JIT spec of ``sm90_cake_combine_tail_bf16`` (wait + top-k reduce + ack in one kernel)."""
    return _gen_module(
        "sm90_cake_bf16_combine_tail",
        _COMBINE_TAIL_SOURCES,
        "cake_combine_tail_bf16.cu",
    )


def sm90_cake_bf16_dispatch_uri(cuda_flags: tuple[str, ...] | None = None) -> str:
    """Content-addressed name of the fused small-T dispatch module."""
    return _module_uri("sm90_cake_bf16_dispatch", _DISPATCH_SOURCES, cuda_flags)


def gen_sm90_cake_bf16_dispatch_module() -> JitSpec:
    """JIT spec of ``sm90_cake_dispatch_fused_bf16`` (count + reserve + store_publish, one cooperative launch)."""
    return _gen_module(
        "sm90_cake_bf16_dispatch", _DISPATCH_SOURCES, "cake_dispatch_fused_bf16.cu"
    )


def sm90_cake_bf16_combine_prereduced_uri(
    cuda_flags: tuple[str, ...] | None = None,
) -> str:
    """Content-addressed name of the pre-reduced combine publish module."""
    return _module_uri(
        "sm90_cake_bf16_combine_prereduced", _COMBINE_PREREDUCED_SOURCES, cuda_flags
    )


def gen_sm90_cake_bf16_combine_prereduced_module() -> JitSpec:
    """JIT spec of ``sm90_cake_combine_prereduced_bf16`` (group build + fp32 pre-reduce + one bf16 row per (token, source rank))."""
    return _gen_module(
        "sm90_cake_bf16_combine_prereduced",
        _COMBINE_PREREDUCED_SOURCES,
        "cake_combine_prereduced_bf16.cu",
    )


def sm90_cake_bf16_combine_tail_prereduced_uri(
    cuda_flags: tuple[str, ...] | None = None,
) -> str:
    """Content-addressed name of the pre-reduced combine-tail module."""
    return _module_uri(
        "sm90_cake_bf16_combine_tail_prereduced",
        _COMBINE_TAIL_PREREDUCED_SOURCES,
        cuda_flags,
    )


def gen_sm90_cake_bf16_combine_tail_prereduced_module() -> JitSpec:
    """JIT spec of ``sm90_cake_combine_tail_prereduced_bf16`` (wait + source-rank-ordered reduce + ack)."""
    return _gen_module(
        "sm90_cake_bf16_combine_tail_prereduced",
        _COMBINE_TAIL_PREREDUCED_SOURCES,
        "cake_combine_tail_prereduced_bf16.cu",
    )


# ---------------------------------------------------------------------------
# Generated expert-grouped GEMMs (FC1 fused SwiGLU / FC1 + clamp / FC2)
# ---------------------------------------------------------------------------
# The device sources and their tvm-ffi bindings are generated by Cake and sealed
# by ``cake_sm90_bf16_megamoe_manifest.json`` (schema ``cake.library_export.v5``):
# one record per stage with the translation units, the launch geometry and a
# sha256 per file.  Every file is re-hashed before it is handed to the JIT so a
# hand edit of a generated source fails closed instead of silently shipping.

_GROUPED_GEMM_PACKAGE = "cake_sm90_bf16_megamoe"
_GROUPED_GEMM_ARCH = "sm_90a"
_GROUPED_GEMM_MANIFEST = _SOURCE_DIR / f"{_GROUPED_GEMM_PACKAGE}_manifest.json"
GROUPED_GEMM_STAGES = ("fc1_gated", "fc1_gated_clamp", "fc2")


def _manifest_path() -> Path:
    return _GROUPED_GEMM_MANIFEST


def _source_dir() -> Path:
    return _SOURCE_DIR


@functools.cache
def grouped_gemm_manifest() -> dict[str, Any]:
    """The sealed manifest of the generated grouped GEMM modules (validated)."""
    path = _manifest_path()
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"generated grouped GEMM manifest is missing: {path}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema") != "cake.library_export.v5"
        or value.get("library") != "flashinfer"
        or value.get("name") != _GROUPED_GEMM_PACKAGE
    ):
        raise RuntimeError("invalid SM90 BF16 grouped GEMM manifest")
    modules = value.get("modules")
    files = value.get("files")
    if (
        not isinstance(modules, list)
        or not modules
        or not isinstance(files, list)
        or not files
    ):
        raise RuntimeError("empty SM90 BF16 grouped GEMM module inventory")
    return value


def grouped_gemm_record(stage: str) -> dict[str, Any]:
    """Manifest record of one generated stage (``fc1_gated`` / ``fc1_gated_clamp`` / ``fc2``)."""
    if stage not in GROUPED_GEMM_STAGES:
        raise ValueError(
            f"unknown grouped GEMM stage {stage!r}; expected one of {GROUPED_GEMM_STAGES}"
        )
    matches = [
        item
        for item in grouped_gemm_manifest()["modules"]
        if item.get("arch") == _GROUPED_GEMM_ARCH
        and dict(item.get("route", {})).get("stage") == stage
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one generated grouped GEMM module for ({_GROUPED_GEMM_ARCH}, {stage}), got {len(matches)}"
        )
    return matches[0]


def _sealed_source(relative_path: str) -> Path:
    """Resolve a manifest path to the on-disk file and verify its sha256."""
    inventory = {item["path"]: item for item in grouped_gemm_manifest()["files"]}
    receipt = inventory.get(relative_path)
    path = _source_dir() / Path(relative_path).name
    if not isinstance(receipt, dict) or not path.is_file():
        raise FileNotFoundError(
            f"generated source is absent from the manifest: {relative_path}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != receipt.get("sha256"):
        raise RuntimeError(f"generated source hash mismatch: {relative_path}")
    return path


@functools.cache
def gen_sm90_cake_bf16_grouped_gemm_module(stage: str) -> JitSpec:
    """JIT spec of one generated grouped GEMM module (device TU + tvm-ffi binding).

    The production build of these kernels uses no fast-math flags, so the JIT
    build does not either (``use_fast_math=False``): the fused SwiGLU epilogue
    is evaluated in fp32 and must match the generator's own build bit for bit.
    """
    record = grouped_gemm_record(stage)
    units = record["translation_units"]
    return gen_jit_spec(
        f"{record['name']}_{record['closure_sha256'][:20]}",
        [_sealed_source(units["device"]), _sealed_source(units["binding"])],
        extra_cuda_cflags=[*sm90a_nvcc_flags, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
        use_fast_math=False,
    )
