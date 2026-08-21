# Copyright (c) 2026 FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Experimental: the RS (W4A16) track is pending dedicated performance
validation and is not wired into any default policy.
"""

import functools
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ......jit import env as jit_env
from ......jit.core import JitSpec, gen_jit_spec, sm90a_nvcc_flags
from ......jit.cpp_ext import is_cuda_version_at_least

KERNEL_VERSION = 5
_IMPLEMENTATIONS = ("scalar", "rs_wgmma")
_N_TACTICS = (16, 32, 64, 96, 128)


def _binary_env(name: str) -> int:
    value = os.environ.get(name, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{name} must be 0 or 1, got {value!r}")
    return int(value)


@dataclass(frozen=True)
class _ExperimentKnobs:
    wgmma_group: int
    static_sched: int
    no_union: int


_PRODUCTION_KNOBS = _ExperimentKnobs(
    wgmma_group=1,
    static_sched=0,
    no_union=0,
)


def _experiment_knobs(*, use_environment: bool = True) -> _ExperimentKnobs:
    if not use_environment:
        return _PRODUCTION_KNOBS
    group = int(os.environ.get("FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP", "1"))
    if group not in (1, 2, 4):
        raise ValueError(
            f"FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP must be 1, 2 or 4, got {group}"
        )
    return _ExperimentKnobs(
        wgmma_group=group,
        static_sched=_binary_env("FLASHINFER_SM90_PUSH_NVFP4_RS_STATIC_SCHED"),
        no_union=_binary_env("FLASHINFER_SM90_PUSH_NVFP4_RS_NO_UNION"),
    )


_STAGE_COUNTS = (3,)
_STAGE_K = (64, 128)
_SOURCE_NAMES = (
    "decode.cuh",
    "scheduler.cuh",
    "sm90_nvfp4_rs_kernel.cuh",
    "sm90_nvfp4_rs_binding.cu",
)


@dataclass(frozen=True)
class _SourceSnapshot:
    sources: tuple[tuple[str, bytes], ...]
    tvm_ffi_utils: bytes
    layout_cuh: bytes
    generator: bytes


def _source_directory() -> Path:
    return Path(__file__).resolve().parents[1] / "src" / "nvfp4_rs_gemm"


def _csrc_directory() -> Path:
    if jit_env.FLASHINFER_CSRC_DIR.is_dir():
        return jit_env.FLASHINFER_CSRC_DIR
    return Path(__file__).resolve().parents[6] / "csrc"


def _canonical_source(path: Path) -> bytes:
    return path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _capture_source_snapshot() -> _SourceSnapshot:
    directory = _source_directory()
    return _SourceSnapshot(
        sources=tuple(
            (name, _canonical_source(directory / name)) for name in _SOURCE_NAMES
        ),
        tvm_ffi_utils=_canonical_source(_csrc_directory() / "tvm_ffi_utils.h"),
        layout_cuh=_canonical_source(
            _csrc_directory().parent / "include" / "flashinfer" / "layout.cuh"
        ),
        generator=_canonical_source(Path(__file__).resolve()),
    )


def _write_snapshot_file(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == content:
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        try:
            os.replace(temporary_path, path)
        except OSError:
            try:
                destination_matches = path.read_bytes() == content
            except OSError:
                destination_matches = False
            if not destination_matches:
                raise
        else:
            temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _materialize_source_snapshot(
    uri: str, snapshot: _SourceSnapshot
) -> tuple[Path, Path]:
    source_root = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    directory = source_root / "nvfp4_rs_gemm"
    for name, content in snapshot.sources:
        _write_snapshot_file(directory / name, content)
    _write_snapshot_file(source_root / "tvm_ffi_utils.h", snapshot.tvm_ffi_utils)
    _write_snapshot_file(source_root / "flashinfer" / "layout.cuh", snapshot.layout_cuh)
    return directory, source_root


def _normalize_implementation(implementation: str) -> str:
    normalized = str(implementation).lower()
    if normalized not in _IMPLEMENTATIONS:
        raise ValueError(
            f"implementation must be one of {_IMPLEMENTATIONS}, got {implementation!r}"
        )
    return normalized


def _normalize_n_tactic(n_tactic: int) -> int:
    normalized = int(n_tactic)
    if normalized not in _N_TACTICS:
        raise ValueError(f"n_tactic must be one of {_N_TACTICS}, got {n_tactic!r}")
    return normalized


def _normalize_stages(stages: int) -> int:
    normalized = int(stages)
    if normalized not in _STAGE_COUNTS:
        raise ValueError(f"stages must be one of {_STAGE_COUNTS}, got {stages!r}")
    return normalized


def _normalize_stage_k(stage_k: int) -> int:
    normalized = int(stage_k)
    if normalized not in _STAGE_K:
        raise ValueError(f"stage_k must be one of {_STAGE_K}, got {stage_k!r}")
    return normalized


def _validate_wgmma_group(stage_k: int, knobs: _ExperimentKnobs) -> None:
    stage_subtiles = stage_k // 32
    if stage_subtiles % knobs.wgmma_group:
        raise ValueError(
            f"FLASHINFER_SM90_PUSH_NVFP4_RS_WGMMA_GROUP={knobs.wgmma_group} does not divide "
            f"stage_k/32={stage_subtiles}"
        )


def _cuda_flags(
    implementation: str,
    n_tactic: int,
    stages: int,
    stage_k: int,
    knobs: _ExperimentKnobs,
) -> tuple[str, ...]:
    return (
        *sm90a_nvcc_flags,
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        f"-DSM90_NVFP4_RS_GEMM_VERSION={KERNEL_VERSION}",
        f"-DSM90_NVFP4_RS_USE_WGMMA={int(implementation == 'rs_wgmma')}",
        f"-DSM90_NVFP4_RS_N_TACTIC={n_tactic}",
        f"-DSM90_NVFP4_RS_STAGES={stages}",
        f"-DSM90_NVFP4_RS_STAGE_K={stage_k}",
        f"-DSM90_NVFP4_RS_WGMMA_GROUP={knobs.wgmma_group}",
        f"-DSM90_NVFP4_RS_STATIC_SCHED={knobs.static_sched}",
        f"-DSM90_NVFP4_RS_NO_UNION={knobs.no_union}",
    )


def _source_digest(
    implementation: str,
    n_tactic: int,
    stages: int,
    stage_k: int,
    knobs: _ExperimentKnobs,
    snapshot: _SourceSnapshot | None = None,
) -> str:
    if snapshot is None:
        snapshot = _capture_source_snapshot()
    digest = hashlib.sha256(
        f"v{KERNEL_VERSION}:{implementation}:n{n_tactic}:s{stages}:k{stage_k}"
        f":g{knobs.wgmma_group}:d{knobs.static_sched}:u{knobs.no_union}".encode()
    )
    for name, content in (
        *snapshot.sources,
        ("tvm_ffi_utils.h", snapshot.tvm_ffi_utils),
        ("flashinfer/layout.cuh", snapshot.layout_cuh),
        (Path(__file__).name, snapshot.generator),
    ):
        digest.update(name.encode())
        digest.update(content)
    digest.update(
        json.dumps(
            _cuda_flags(implementation, n_tactic, stages, stage_k, knobs),
            separators=(",", ":"),
        ).encode()
    )
    return digest.hexdigest()[:16]


def _uri(
    implementation: str,
    n_tactic: int,
    stages: int,
    stage_k: int,
    knobs: _ExperimentKnobs,
    source_digest: str | None = None,
) -> str:
    if source_digest is None:
        source_digest = _source_digest(implementation, n_tactic, stages, stage_k, knobs)
    return (
        f"sm90_push_nvfp4_rs_gemm_v{KERNEL_VERSION}_{implementation}_"
        f"n{n_tactic}_s{stages}_k{stage_k}_"
        f"{source_digest}"
    )


def get_sm90_push_nvfp4_rs_gemm_uri(
    implementation: str = "rs_wgmma",
    n_tactic: int = 64,
    stages: int = 3,
    stage_k: int = 64,
) -> str:
    knobs = _experiment_knobs()
    implementation = _normalize_implementation(implementation)
    n_tactic = _normalize_n_tactic(n_tactic)
    stages = _normalize_stages(stages)
    stage_k = _normalize_stage_k(stage_k)
    _validate_wgmma_group(stage_k, knobs)
    return _uri(implementation, n_tactic, stages, stage_k, knobs)


def _gen_sm90_push_nvfp4_rs_gemm_module(
    implementation: str,
    n_tactic: int,
    stages: int,
    stage_k: int,
    knobs: _ExperimentKnobs,
    source_digest: str | None = None,
    source_snapshot: _SourceSnapshot | None = None,
) -> JitSpec:
    _validate_wgmma_group(stage_k, knobs)
    if not is_cuda_version_at_least("12.0"):
        raise RuntimeError("SM90 push NVFP4 RS GEMM requires CUDA 12.0 or newer")
    if source_snapshot is None:
        source_snapshot = _capture_source_snapshot()
    snapshot_digest = _source_digest(
        implementation,
        n_tactic,
        stages,
        stage_k,
        knobs,
        snapshot=source_snapshot,
    )
    if source_digest is None:
        source_digest = snapshot_digest
    elif source_digest != snapshot_digest:
        raise ValueError("source_digest does not match source_snapshot")
    uri = _uri(
        implementation,
        n_tactic,
        stages,
        stage_k,
        knobs,
        source_digest=source_digest,
    )
    directory, source_root = _materialize_source_snapshot(uri, source_snapshot)
    return gen_jit_spec(
        uri,
        [directory / "sm90_nvfp4_rs_binding.cu"],
        extra_cuda_cflags=list(
            _cuda_flags(implementation, n_tactic, stages, stage_k, knobs)
        ),
        extra_include_paths=[source_root, directory],
    )


def gen_sm90_push_nvfp4_rs_gemm_module(
    implementation: str = "rs_wgmma",
    n_tactic: int = 64,
    stages: int = 3,
    stage_k: int = 64,
    *,
    use_environment: bool = True,
) -> JitSpec:
    knobs = _experiment_knobs(use_environment=use_environment)
    implementation = _normalize_implementation(implementation)
    n_tactic = _normalize_n_tactic(n_tactic)
    stages = _normalize_stages(stages)
    stage_k = _normalize_stage_k(stage_k)
    _validate_wgmma_group(stage_k, knobs)
    return _gen_sm90_push_nvfp4_rs_gemm_module(
        implementation, n_tactic, stages, stage_k, knobs
    )


@functools.cache
def _load_sm90_push_nvfp4_rs_gemm_module_cached(
    implementation: str,
    n_tactic: int,
    stages: int,
    stage_k: int,
    knobs: _ExperimentKnobs,
    source_digest: str,
    source_snapshot: _SourceSnapshot,
):
    return _gen_sm90_push_nvfp4_rs_gemm_module(
        implementation,
        n_tactic,
        stages,
        stage_k,
        knobs,
        source_digest,
        source_snapshot,
    ).build_and_load()


def load_sm90_push_nvfp4_rs_gemm_module(
    implementation: str = "rs_wgmma",
    n_tactic: int = 64,
    stages: int = 3,
    stage_k: int = 64,
    *,
    use_environment: bool = True,
):
    knobs = _experiment_knobs(use_environment=use_environment)
    implementation = _normalize_implementation(implementation)
    n_tactic = _normalize_n_tactic(n_tactic)
    stages = _normalize_stages(stages)
    stage_k = _normalize_stage_k(stage_k)
    _validate_wgmma_group(stage_k, knobs)
    source_snapshot = _capture_source_snapshot()
    source_digest = _source_digest(
        implementation,
        n_tactic,
        stages,
        stage_k,
        knobs,
        snapshot=source_snapshot,
    )
    return _load_sm90_push_nvfp4_rs_gemm_module_cached(
        implementation,
        n_tactic,
        stages,
        stage_k,
        knobs,
        source_digest,
        source_snapshot,
    )


def create_sm90_push_nvfp4_rs_gemm_runner(
    implementation: str = "rs_wgmma",
    n_tactic: int = 64,
    stages: int = 3,
    stage_k: int = 64,
    *,
    use_environment: bool = True,
):
    return load_sm90_push_nvfp4_rs_gemm_module(
        implementation,
        n_tactic,
        stages,
        stage_k,
        use_environment=use_environment,
    ).init()
