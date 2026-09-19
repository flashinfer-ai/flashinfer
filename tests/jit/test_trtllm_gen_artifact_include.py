"""Tests for versioned artifact-header include roots (upstream issue #2987).

The trtllm-gen BMM/MoE/GEMM JIT modules compile against headers that ship in a
downloaded artifact.  The export headers are staged behind a symlink and the
arch-filtered ``flashinferMetaInfo.h`` is written next to it, so the ``-I`` root
for a module used to be the same string for every artifact version and the
on-disk build did not depend on the artifact at all: two versions sharing one
cache directory could shadow each other's headers, and a different version
reused the same objects and ``.so``.

These are CPU-only tests.  They drive the real module generators, the real
``ensure_symlink``/``verify_symlinked_headers`` and the real checksum
verification against a synthetic artifact laid out under ``tmp_path``; only the
artifact-fetch boundary (``get_artifact``/``download_file``) is stubbed, and the
offline/checksum tests use the real loader with the network blocked.  No GPU, no
nvcc, no network.
"""

import dataclasses
import hashlib
import multiprocessing
import os
import pathlib
import shlex
import types
from concurrent.futures import ProcessPoolExecutor
from typing import NamedTuple, Tuple

import pytest

from flashinfer.artifacts import ArtifactPath, CheckSumHash
from flashinfer.jit import core as jit_core
from flashinfer.jit import cpp_ext
from flashinfer.jit import cubin_loader
from flashinfer.jit import env as jit_env
from flashinfer.jit import fused_moe as fused_moe_jit
from flashinfer.jit import moe_utils as moe_utils_jit
from flashinfer.jit.fused_moe import BMM_EXPORT_HEADERS
from flashinfer.jit.gemm import core as gemm_core
from flashinfer.jit.gemm.core import GEMM_EXPORT_HEADERS
from flashinfer.jit.trtllm_gen_metainfo import (
    BLACKWELL_CUBIN_ARCHS,
    RUBIN_CUBIN_ARCHS,
    filter_metainfo,
)

# Captured before any test monkeypatches it, so the offline/checksum tests can
# restore the real artifact loader.
_REAL_GET_ARTIFACT = cubin_loader.get_artifact

# Two synthetic publishes of the BMM artifact and one of the GEMM artifact.  The
# first BMM pin is the real one; the second stands for another version of the
# same artifact, which is what a repin (or a downgrade) looks like on disk.
_BMM_PIN_A = ArtifactPath.TRTLLM_GEN_BMM
_BMM_PIN_B = (
    "8b1f4c07d9e2536a4c7b0e18f2a95d364c0e71ba9f3d8265a1c4e07b6d938f21/"
    "batched_gemm-31ee4e5-4a17c9d/"
)
_GEMM_PIN_A = ArtifactPath.TRTLLM_GEN_GEMM

# exitcode of a worker killed right before the artifact's atomic rename.
_KILLED_EXITCODE = 137


@dataclasses.dataclass(frozen=True)
class _Family:
    """One trtllm-gen export-header family, as its consumers see it."""

    path_attr: str
    checksum_attr: str
    export_dir: str
    link_parts: Tuple[str, ...]
    headers: Tuple[str, ...]
    gemm: bool = False


BMM = _Family(
    path_attr="TRTLLM_GEN_BMM",
    checksum_attr="TRTLLM_GEN_BMM",
    export_dir="trtllmGen_bmm_export",
    link_parts=("flashinfer", "trtllm", "batched_gemm"),
    headers=tuple(BMM_EXPORT_HEADERS),
)
GEMM = _Family(
    path_attr="TRTLLM_GEN_GEMM",
    checksum_attr="TRTLLM_GEN_GEMM",
    export_dir="trtllmGen_gemm_export",
    link_parts=("flashinfer", "trtllm", "gemm"),
    headers=tuple(GEMM_EXPORT_HEADERS),
    gemm=True,
)


class _Resolution(NamedTuple):
    """What one generation pass resolved; enough to spot a cross-redirect."""

    module: str
    version: str
    include_root: str
    manifest_sha256: str
    link_target: str
    headers: Tuple[str, ...]


class _StubCompilationContext:
    """No-GPU stand-in so module generation needs no CUDA device."""

    TARGET_CUDA_ARCHS = {(10, "0f")}

    def get_nvcc_flags_list(
        self, supported_major_versions=None, map_sm107_to_100f=False
    ):
        return ["-gencode=arch=compute_100f,code=sm_100f"]


def _manifest(tag: str, gemm: bool = False) -> str:
    """A ``flashinferMetaInfo.h`` manifest with Blackwell and Rubin entries."""
    list_name = "tllmGenGemmList" if gemm else "tllmGenBatchedGemmList"
    config = "gemm::GemmConfig" if gemm else "batchedGemm::BatchedGemmConfig"
    entries = "\n".join(
        f'{{nullptr, 0, {index}, "kernel_{tag}_{arch}", 512, "hash_{tag}_{arch}", "",\n'
        f" nullptr, nullptr, nullptr, 0, {{ /* mA */ {index}\n"
        f" }}, gemm::SmVersion::{arch}}},"
        for index, arch in enumerate(("Sm100f", "Sm103a", "Sm107a"), start=1)
    )
    return (
        "#pragma once\n"
        f"static constexpr size_t {list_name}Len = 3;\n\n"
        f"static const {config} {list_name}[] = {{\n"
        f"{entries}\n"
        "};\n"
    )


def _publish(workspace, family: _Family, pin: str, tag: str) -> str:
    """Lay out one synthetic artifact publish and return its version tag.

    The version tag is the checksum of ``checksums.txt``, which is what the
    generators use as ``CheckSumHash``: it identifies the artifact's content, so
    a republish under the same pin still gets an include root of its own.
    """
    root = workspace.cubin / pin
    (root / "include" / family.export_dir).mkdir(parents=True)
    files = [("include/flashinferMetaInfo.h", _manifest(tag, gemm=family.gemm))]
    files += [
        (f"include/{family.export_dir}/{name}", f"// {name} of artifact {tag}\n")
        for name in family.headers
    ]
    checksums = []
    for name, content in files:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        checksums.append(f"{hashlib.sha256(content.encode()).hexdigest()} {name}")
    manifest = "\n".join(checksums) + "\n"
    (root / "checksums.txt").write_text(manifest)
    return hashlib.sha256(manifest.encode()).hexdigest()


def _cached_artifact_reader(file_name, sha256, session=None):
    """Stand in for ``get_artifact``: read the local cache, never the network."""
    path = jit_env.FLASHINFER_CUBIN_DIR / file_name
    return path.read_bytes() if path.is_file() else b""


def _offline(*args, **kwargs):
    """Block the network: any download attempt fails the test."""
    raise AssertionError("test attempted to download an artifact")


def _download_fails(*args, **kwargs):
    """The artifact repository is unreachable."""
    return False


def _point_at(monkeypatch, family: _Family, pin: str, version: str) -> None:
    """Point *family*'s generators at one synthetic artifact version."""
    monkeypatch.setattr(ArtifactPath, family.path_attr, pin)
    monkeypatch.setattr(CheckSumHash, family.checksum_attr, version)
    for module in (fused_moe_jit, gemm_core, cubin_loader):
        monkeypatch.setattr(module, "get_artifact", _cached_artifact_reader)


def _workspace(root, monkeypatch):
    """Redirect every cache the generators touch into *root*."""
    paths = types.SimpleNamespace(
        cubin=root / "cubins",
        gen=root / "generated",
        jit=root / "cached_ops",
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_CUBIN_DIR", paths.cubin)
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", paths.gen)
    monkeypatch.setattr(jit_env, "FLASHINFER_JIT_DIR", paths.jit)
    monkeypatch.setattr(cubin_loader, "FLASHINFER_CUBIN_DIR", paths.cubin)
    monkeypatch.setattr(cpp_ext, "get_cuda_path", lambda: "/usr/local/cuda")
    context = _StubCompilationContext()
    for module in (jit_core, fused_moe_jit, moe_utils_jit, gemm_core):
        monkeypatch.setattr(module, "current_compilation_context", context)
    return paths


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A synthetic artifact cache plus a writable JIT workspace."""
    return _workspace(tmp_path, monkeypatch)


@pytest.fixture
def spaced_workspace(tmp_path, monkeypatch):
    """The same harness, with a space in every cache path."""
    return _workspace(tmp_path / "flash infer", monkeypatch)


def _gen_bmm(**kwargs):
    return fused_moe_jit.gen_trtllm_gen_fused_moe_sm100_module(**kwargs)


def _versioned_root(spec, version: str) -> pathlib.Path:
    """The include root the spec resolves *version*'s headers through."""
    roots = [path for path in spec.extra_include_dirs if version in path.parts]
    assert len(roots) == 1, f"expected one versioned include root, got {roots}"
    assert spec.extra_include_dirs[0] == roots[0], "export root must be searched first"
    return roots[0]


def _export_link(root: pathlib.Path, family: _Family) -> pathlib.Path:
    """The staged export-header symlink inside a versioned include root."""
    return root.joinpath(*family.link_parts, family.export_dir)


def _resolve(include_dirs, header: str, includer_dir) -> pathlib.Path:
    """Resolve a quoted ``#include`` the way the compiler does.

    The directory of the including file is searched first, then every ``-I``
    root in the order ``build_common_cflags`` emits them.
    """
    for root in (pathlib.Path(includer_dir), *include_dirs):
        candidate = root / header
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{header} is on no include path of {include_dirs}")


def _resolution(spec, family: _Family) -> _Resolution:
    """The observable resolution result of one generation pass."""
    root = spec.extra_include_dirs[0]
    link = _export_link(root, family)
    return _Resolution(
        module=spec.build_dir.name,
        version=spec.artifact_version,
        include_root=str(root),
        manifest_sha256=hashlib.sha256(
            (root / "flashinferMetaInfo.h").read_text().encode()
        ).hexdigest(),
        link_target=str(link.resolve()),
        # The download path keeps <name>.<sha>.tmp (atomic-rename staging) and
        # <name>.lock (per-file mutex) next to the committed header; neither is a
        # resolved header, and a killed writer can leave the staging file behind.
        headers=tuple(
            sorted(
                str(path.relative_to(link))
                for path in link.rglob("*")
                if path.is_file()
                and not path.name.endswith((".tmp", ".lock"))
            )
        ),
    )


def _ninja_variable(ninja: str, name: str) -> list:
    """The shell-split tokens of one ninja variable assignment."""
    lines = ninja.split("\n")
    index = next(i for i, line in enumerate(lines) if line.startswith(f"{name} = "))
    parts = []
    line = lines[index].split(" = ", 1)[1]
    while True:
        parts.append(line[: -len(" $")] if line.endswith(" $") else line)
        if not line.endswith(" $"):
            break
        index += 1
        line = lines[index].strip()
    return shlex.split(" ".join(parts))


def _worker_generate_bmm(_ignored):
    """Run one BMM generation pass in a forked worker process."""
    return _resolution(_gen_bmm(), BMM)


def _worker_generate_version(job):
    """Resolve one artifact version in a forked worker process."""
    path_attr, checksum_attr, pin, version, tag = job
    setattr(ArtifactPath, path_attr, pin)
    setattr(CheckSumHash, checksum_attr, version)
    return tag, _resolution(_gen_bmm(), BMM)


def _worker_killed_mid_download(source: str, destination: str) -> None:
    """Copy an artifact, then die right before the atomic rename."""
    cubin_loader.os.replace = lambda *args: os._exit(_KILLED_EXITCODE)
    cubin_loader.download_file(source, destination)


def test_same_version_regenerates_identical_identity(workspace, monkeypatch):
    """A1-01: regenerating one artifact version changes nothing."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    first = _gen_bmm()
    filtered = _versioned_root(first, version) / "flashinferMetaInfo.h"
    stamp = filtered.stat().st_mtime_ns

    second = _gen_bmm()

    assert second.name == first.name
    assert second.build_dir == first.build_dir
    assert second.ninja_path == first.ninja_path
    assert second.jit_library_path == first.jit_library_path
    assert second.lock_path == first.lock_path
    assert second.extra_include_dirs == first.extra_include_dirs
    assert second.artifact_version == version
    assert filtered.stat().st_mtime_ns == stamp


def test_different_versions_change_the_build_identity(workspace, monkeypatch):
    """A1-02: the artifact version is part of the build identity."""
    version_a = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version_a)
    spec_a = _gen_bmm()

    version_b = _publish(workspace, BMM, _BMM_PIN_B, "b")
    _point_at(monkeypatch, BMM, _BMM_PIN_B, version_b)
    spec_b = _gen_bmm()

    assert version_a != version_b
    # The module keeps its name, so an installed AOT cache still matches; only
    # the on-disk build is versioned.
    assert spec_a.name == spec_b.name
    assert spec_a.build_dir != spec_b.build_dir
    assert spec_a.ninja_path != spec_b.ninja_path
    assert spec_a.jit_library_path != spec_b.jit_library_path
    assert spec_a.lock_path != spec_b.lock_path
    assert spec_a.artifact_version == version_a
    assert spec_b.artifact_version == version_b
    assert _versioned_root(spec_a, version_a) != _versioned_root(spec_b, version_b)

    # A module with no exported artifact headers keeps its historical layout.
    plain = jit_core.gen_jit_spec("plain_module", [])
    assert plain.artifact_version is None
    assert plain.build_dir == jit_env.FLASHINFER_JIT_DIR / "plain_module"
    assert plain.jit_library_path == plain.build_dir / "plain_module.so"


def test_generating_b_does_not_rewrite_a(workspace, monkeypatch):
    """A1-03: generating B after A leaves A's root and build alone."""
    version_a = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version_a)
    spec_a = _gen_bmm()
    spec_a.write_ninja()

    version_b = _publish(workspace, BMM, _BMM_PIN_B, "b")
    _point_at(monkeypatch, BMM, _BMM_PIN_B, version_b)
    spec_b = _gen_bmm()
    spec_b.write_ninja()

    root_a = _versioned_root(spec_a, version_a)
    root_b = _versioned_root(spec_b, version_b)
    manifest_a = (root_a / "flashinferMetaInfo.h").read_text()
    assert "kernel_a_" in manifest_a
    assert "kernel_b_" in (root_b / "flashinferMetaInfo.h").read_text()
    assert "kernel_b_" not in manifest_a
    # A's export symlink still points at A's publish, not at B's.
    assert _export_link(root_a, BMM).resolve() == (
        workspace.cubin / _BMM_PIN_A / "include" / BMM.export_dir
    ).resolve()
    assert _export_link(root_b, BMM).resolve() == (
        workspace.cubin / _BMM_PIN_B / "include" / BMM.export_dir
    ).resolve()
    # ... and A's build file only ever references A's include root.
    ninja_a = spec_a.ninja_path.read_text()
    assert f"-I{root_a.resolve()}" in ninja_a
    assert f"-I{root_b.resolve()}" not in ninja_a
    assert spec_a.ninja_path != spec_b.ninja_path


def test_filtered_metainfo_beats_the_raw_artifact_header(workspace, monkeypatch):
    """A1-04: only the arch-filtered manifest is reachable by name."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    spec = _gen_bmm()

    root = _versioned_root(spec, version)
    link = _export_link(root, BMM)
    raw = workspace.cubin / _BMM_PIN_A / "include" / "flashinferMetaInfo.h"
    assert "Sm107a" in raw.read_text(), "the raw artifact manifest lists every arch"

    filtered = _resolve(spec.extra_include_dirs, "flashinferMetaInfo.h", link)
    assert filtered == root / "flashinferMetaInfo.h"
    assert "Sm107a" not in filtered.read_text()
    assert "Sm100f" in filtered.read_text()

    # The artifact's own include/ directory is on no include path, so the
    # unfiltered manifest is unreachable even through a qualified include.
    artifact_include = (workspace.cubin / _BMM_PIN_A / "include").resolve()
    resolved_dirs = [path.resolve() for path in spec.extra_include_dirs]
    assert artifact_include not in resolved_dirs


def test_module_scoped_roots_stay_isolated(workspace, monkeypatch):
    """Criterion 3: BMM, routing, MoE and GEMM roots never collapse together."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    gemm_version = _publish(workspace, GEMM, _GEMM_PIN_A, "g")
    _point_at(monkeypatch, GEMM, _GEMM_PIN_A, gemm_version)

    bmm_roots = {
        "fused_moe_trtllm_sm100": _versioned_root(_gen_bmm(), version),
        "fused_moe_trtllm_sm107": _versioned_root(_gen_bmm(enable_rubin=True), version),
        "trtllm_gen_routing": _versioned_root(
            fused_moe_jit.gen_trtllm_gen_routing_module(), version
        ),
        "moe_utils": _versioned_root(moe_utils_jit.gen_moe_utils_module(), version),
    }
    gemm_specs = {
        "trtllm_gemm": gemm_core.gen_trtllm_gen_gemm_module(),
        "trtllm_low_latency_gemm": gemm_core.gen_trtllm_low_latency_gemm_module(),
    }
    gemm_roots = {
        name: _versioned_root(spec, gemm_version) for name, spec in gemm_specs.items()
    }

    roots = {**bmm_roots, **gemm_roots}
    assert len(set(roots.values())) == len(roots)
    assert {root.parent.name for root in roots.values()} == set(roots)

    for name in ("trtllm_gen_routing", "moe_utils"):
        assert _export_link(bmm_roots[name], BMM).is_symlink()
    for name, spec in gemm_specs.items():
        assert spec.artifact_version == gemm_version
        assert _export_link(gemm_roots[name], GEMM).is_symlink()

    # The Blackwell and Rubin variants share one artifact but not one manifest.
    rubin_manifest = (
        bmm_roots["fused_moe_trtllm_sm107"] / "flashinferMetaInfo.h"
    ).read_text()
    blackwell_manifest = (
        bmm_roots["fused_moe_trtllm_sm100"] / "flashinferMetaInfo.h"
    ).read_text()
    assert "Sm107a" in rubin_manifest
    assert "Sm107a" not in blackwell_manifest


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
@pytest.mark.parametrize("workers", sorted({2, 4, min(os.cpu_count() or 4, 8)}))
def test_concurrent_same_version_workers_agree(workspace, monkeypatch, workers):
    """A1-05: concurrent workers of one version resolve one complete result."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    with ProcessPoolExecutor(
        workers, mp_context=multiprocessing.get_context("fork")
    ) as pool:
        observed = set(pool.map(_worker_generate_bmm, range(workers)))

    assert len(observed) == 1
    resolution = observed.pop()
    assert resolution.module.startswith("fused_moe_trtllm_sm100")
    assert version in resolution.module
    assert resolution.headers == tuple(sorted(BMM.headers))
    expected, kept, _ = filter_metainfo(_manifest("a"), BLACKWELL_CUBIN_ARCHS)
    assert kept == 2
    assert resolution.manifest_sha256 == hashlib.sha256(expected.encode()).hexdigest()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
def test_concurrent_versions_share_a_cache_without_cross_redirect(
    workspace, monkeypatch
):
    """A1-06: two versions in one cache directory never redirect each other."""
    version_a = _publish(workspace, BMM, _BMM_PIN_A, "a")
    version_b = _publish(workspace, BMM, _BMM_PIN_B, "b")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version_a)

    versions = [("a", _BMM_PIN_A, version_a), ("b", _BMM_PIN_B, version_b)]
    jobs = [
        (BMM.path_attr, BMM.checksum_attr, pin, version, tag)
        for tag, pin, version in versions * 2
    ]
    with ProcessPoolExecutor(4, mp_context=multiprocessing.get_context("fork")) as pool:
        results = list(pool.map(_worker_generate_version, jobs))

    for tag, pin, version in versions:
        resolutions = {value for worker_tag, value in results if worker_tag == tag}
        assert len(resolutions) == 1, f"version {tag} resolved inconsistently"
        resolution = resolutions.pop()
        assert resolution.version == version
        assert version in resolution.include_root
        assert resolution.link_target == str(
            (workspace.cubin / pin / "include" / BMM.export_dir).resolve()
        )
        assert resolution.headers == tuple(sorted(BMM.headers))
        expected, kept, _ = filter_metainfo(_manifest(tag), BLACKWELL_CUBIN_ARCHS)
        assert kept == 2
        assert resolution.manifest_sha256 == hashlib.sha256(
            expected.encode()
        ).hexdigest()


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs the fork start method")
def test_killed_worker_leaves_no_torn_artifact(workspace, monkeypatch):
    """A1-07: a worker killed mid-write leaves the cache consumable."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    destination = (
        workspace.cubin / _BMM_PIN_A / "include" / BMM.export_dir / BMM.headers[0]
    )
    good = destination.read_bytes()
    torn = workspace.cubin / "torn-write"
    torn.mkdir()
    (torn / BMM.headers[0]).write_bytes(b"torn write from a killed worker\n")

    process = multiprocessing.get_context("fork").Process(
        target=_worker_killed_mid_download,
        args=(str(torn / BMM.headers[0]), str(destination)),
    )
    process.start()
    process.join()

    assert process.exitcode == _KILLED_EXITCODE
    # The killed writer left its temporary file behind, but the destination is
    # still the previous complete artifact rather than a half-written one.
    assert [path.name for path in destination.parent.glob("*.tmp")]
    assert destination.read_bytes() == good

    assert _resolution(_gen_bmm(), BMM).headers == tuple(sorted(BMM.headers))


def test_missing_artifact_fails_loudly(workspace, monkeypatch):
    """A1-08: a missing artifact is an explicit failure, not a stale header."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")
    for module in (fused_moe_jit, gemm_core, cubin_loader):
        monkeypatch.setattr(module, "get_artifact", _REAL_GET_ARTIFACT)

    (workspace.cubin / _BMM_PIN_A / "checksums.txt").unlink()

    with pytest.raises(RuntimeError, match="Artifact not found locally"):
        _gen_bmm()


def test_corrupt_cached_artifact_fails_loudly(workspace, monkeypatch):
    """A1-08: a checksum mismatch is reported, never papered over."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    for module in (fused_moe_jit, gemm_core, cubin_loader):
        monkeypatch.setattr(module, "get_artifact", _REAL_GET_ARTIFACT)
    monkeypatch.setattr(cubin_loader, "download_file", _download_fails)

    cached = workspace.cubin / _BMM_PIN_A / "include" / BMM.export_dir / BMM.headers[0]
    cached.write_bytes(b"header left behind by a different artifact version\n")

    with pytest.raises(AssertionError, match="not found"):
        _gen_bmm()


def test_tampered_header_through_the_link_is_rejected(workspace, monkeypatch):
    """Criterion 4: the staged link keeps its checksum verification."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    spec = _gen_bmm()
    link = _export_link(_versioned_root(spec, version), BMM)
    checksums = (workspace.cubin / _BMM_PIN_A / "checksums.txt").read_bytes()

    (
        workspace.cubin / _BMM_PIN_A / "include" / BMM.export_dir / BMM.headers[0]
    ).write_bytes(b"tampered\n")

    with pytest.raises(RuntimeError, match="wrong checksum"):
        cubin_loader.verify_symlinked_headers(link, BMM.headers, checksums)


def test_include_flags_survive_paths_with_spaces(spaced_workspace, monkeypatch):
    """A1-09: the versioned include flag survives a space in the path."""
    version = _publish(spaced_workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    spec = _gen_bmm()
    spec.write_ninja()

    root = _versioned_root(spec, version)
    assert " " in str(root)
    flags = _ninja_variable(spec.ninja_path.read_text(), "common_cflags")
    assert f"-I{root.resolve()}" in flags


def test_read_only_artifact_cache_is_never_written(workspace, monkeypatch):
    """A1-10: staging happens in the writable cache, not the artifact cache."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    before = {path for path in workspace.cubin.rglob("*")}
    directories = [path for path in before if path.is_dir()]
    for path in directories:
        path.chmod(0o555)
    try:
        spec = _gen_bmm()
    finally:
        for path in directories:
            path.chmod(0o755)

    assert {path for path in workspace.cubin.rglob("*")} == before
    root = _versioned_root(spec, version)
    assert workspace.gen in root.parents
    assert workspace.jit in spec.build_dir.parents
    assert _export_link(root, BMM).is_symlink()


def test_offline_preset_artifact_resolves_and_verifies(workspace, monkeypatch):
    """A1-11: a preset artifact with the network blocked still verifies."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)
    monkeypatch.setenv("FLASHINFER_NO_DOWNLOAD", "1")
    for module in (fused_moe_jit, gemm_core, cubin_loader):
        monkeypatch.setattr(module, "get_artifact", _REAL_GET_ARTIFACT)
    monkeypatch.setattr(cubin_loader, "download_file", _offline)

    spec = _gen_bmm()

    root = _versioned_root(spec, version)
    link = _export_link(root, BMM)
    checksums = (workspace.cubin / _BMM_PIN_A / "checksums.txt").read_bytes()
    cubin_loader.verify_symlinked_headers(link, BMM.headers, checksums)
    assert _resolve(spec.extra_include_dirs, "flashinferMetaInfo.h", link) == (
        root / "flashinferMetaInfo.h"
    )
    assert _resolution(spec, BMM).version == version


def test_rubin_and_blackwell_keep_distinct_manifests(workspace, monkeypatch):
    """The module variant, not the visible device, picks the manifest."""
    version = _publish(workspace, BMM, _BMM_PIN_A, "a")
    _point_at(monkeypatch, BMM, _BMM_PIN_A, version)

    for spec, archs in (
        (_gen_bmm(), BLACKWELL_CUBIN_ARCHS),
        (_gen_bmm(enable_rubin=True), RUBIN_CUBIN_ARCHS),
    ):
        filtered = (_versioned_root(spec, version) / "flashinferMetaInfo.h").read_text()
        assert filtered == filter_metainfo(_manifest("a"), archs)[0]
