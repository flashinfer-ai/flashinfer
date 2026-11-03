"""flashinfer.experimental._sources: how experimental loaders find their kernel
sources when a release wheel left them out (flashinfer-ai/flashinfer#7977), and
the `flashinfer download-experimental-sources` install flow against a tarball
that build_backend._slim_wheel wrote.

The module under test is loaded from its file so no CUDA, torch or flashinfer
import is needed; its three environment indirections are replaced by fixtures.

    python -m pytest --confcutdir=tests/cpu tests/cpu/test_experimental_sources.py
"""

import base64
import csv
import hashlib
import importlib.util
import io
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[2]
VERSION = "0.0.0+test"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve postponed annotations through sys.modules[module.__name__]
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def sources():
    return _load(
        "flashinfer_experimental_sources_under_test",
        _ROOT / "flashinfer" / "experimental" / "_sources.py",
    )


@pytest.fixture(scope="module")
def build_backend():
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    return _load("flashinfer_build_backend_under_test", _ROOT / "build_backend.py")


@pytest.fixture
def env(tmp_path, monkeypatch, sources):
    """A fake installed package tree, cache and version; nothing under the real cache."""
    tmp_path = (
        tmp_path.resolve()
    )  # the helper resolves loader paths; keep both sides resolved
    cache = tmp_path / "cache"
    package = tmp_path / "site" / "flashinfer"
    experimental = package / "experimental"
    experimental.mkdir(parents=True)
    installed_csrc = package / "data" / "csrc"
    installed_csrc.mkdir(parents=True)
    (installed_csrc / "tvm_ffi_utils.h").write_text("// binding header\n")
    monkeypatch.setattr(sources, "_cache_dir", lambda: cache)
    monkeypatch.setattr(sources, "_version", lambda: VERSION)
    monkeypatch.setattr(sources, "_installed_csrc_dir", lambda: installed_csrc)
    monkeypatch.setattr(sources, "_EXPERIMENTAL_DIR", experimental)
    monkeypatch.delenv(sources.SOURCES_DIR_ENV, raising=False)
    return SimpleNamespace(
        tmp=tmp_path,
        cache=cache,
        experimental=experimental,
        installed_csrc=installed_csrc,
        root=cache / "experimental-sources" / VERSION,
    )


def _loader(env, *parts):
    """A fake loader module file of the experimental package ``parts`` (no csrc)."""
    package = env.experimental.joinpath(*parts)
    package.mkdir(parents=True, exist_ok=True)
    loader = package / "jit.py"
    loader.write_text("")
    return loader


# --- resolution -----------------------------------------------------------------------------


def test_sources_root_is_version_keyed_in_the_cache_unless_overridden(
    env, sources, monkeypatch
):
    assert sources.sources_root() == env.root
    monkeypatch.setenv(sources.SOURCES_DIR_ENV, str(env.tmp / "shared"))
    assert sources.sources_root() == env.tmp / "shared"


def test_package_csrc_dir_prefers_the_package_copy(env, sources):
    loader = _loader(env, "foo")
    in_package = loader.parent / "csrc"
    in_package.mkdir()
    fetched = env.root / "flashinfer" / "experimental" / "foo" / "csrc"
    fetched.mkdir(parents=True)
    assert sources.package_csrc_dir(loader) == in_package


def test_package_csrc_dir_falls_back_to_the_fetched_tree(env, sources):
    loader = _loader(env, "foo")
    fetched = env.root / "flashinfer" / "experimental" / "foo" / "csrc"
    fetched.mkdir(parents=True)
    assert sources.package_csrc_dir(loader) == fetched
    # nested packages keep their relative path (flashinfer/experimental/cake/<pkg>)
    nested = _loader(env, "cake", "bar")
    nested_fetched = env.root / "flashinfer" / "experimental" / "cake" / "bar" / "csrc"
    nested_fetched.mkdir(parents=True)
    assert sources.package_csrc_dir(nested) == nested_fetched


def test_package_csrc_dir_missing_names_marker_command_and_directory(env, sources):
    loader = _loader(env, "foo")
    with pytest.raises(sources.ExperimentalSourcesNotFound) as info:
        sources.package_csrc_dir(loader)
    assert isinstance(
        info.value, FileNotFoundError
    )  # callers' available() checks still work
    message = str(info.value)
    for expected in (
        "flashinfer.experimental.foo",
        sources.MARKER_NAME,
        sources.COMMAND,
        str(env.root),
        sources.SOURCES_DIR_ENV,
        str(loader.parent / "csrc"),
    ):
        assert expected in message, expected


def test_package_csrc_dir_outside_experimental_is_a_broken_install(env, sources):
    # the Cake loader template is shared with core packages, whose csrc every wheel ships
    core = env.tmp / "site" / "flashinfer" / "cake_core_backend"
    core.mkdir(parents=True)
    (core / "cake_build.py").write_text("")
    (env.root / "flashinfer" / "experimental" / "cake_core_backend" / "csrc").mkdir(
        parents=True
    )
    with pytest.raises(sources.ExperimentalSourcesNotFound, match="is not under"):
        sources.package_csrc_dir(core / "cake_build.py")


def test_shared_csrc_dir_prefers_installed_then_fetched(env, sources):
    with pytest.raises(sources.ExperimentalSourcesNotFound) as info:
        sources.shared_csrc_dir()
    assert sources.COMMAND in str(info.value)
    fetched = env.root / "flashinfer" / "data" / "csrc"
    (fetched / "experimental").mkdir(parents=True)
    assert sources.shared_csrc_dir() == fetched
    # the fetched root first (experimental/ headers), then the installed csrc (binding headers)
    assert sources.shared_include_dirs() == [fetched, env.installed_csrc]
    (env.installed_csrc / "experimental").mkdir()
    assert sources.shared_csrc_dir() == env.installed_csrc
    assert sources.shared_include_dirs() == [env.installed_csrc]


# --- the marker -----------------------------------------------------------------------------


def test_read_marker(env, sources):
    assert sources.read_marker() is None
    marker = env.experimental / sources.MARKER_NAME
    digest = "ab" * 32
    marker.write_text(
        f"prose: not a field\n\ntarball: x-0.0.0-experimental-sources.tar.xz\nsha256: {digest}\n"
    )
    parsed = sources.read_marker()
    assert (parsed.tarball, parsed.sha256) == (
        "x-0.0.0-experimental-sources.tar.xz",
        digest,
    )
    assert sources.tarball_url(parsed) == (
        f"{sources.RELEASE_DOWNLOAD_URL}/v{VERSION}/x-0.0.0-experimental-sources.tar.xz"
    )
    marker.write_text("an older marker without the tarball lines\n")
    older = sources.read_marker()
    assert older.tarball is None and older.sha256 is None
    with pytest.raises(sources.SourcesError, match="predates"):
        sources.tarball_url(older)
    marker.write_text("sha256: nope\n")
    with pytest.raises(sources.SourcesError, match="malformed"):
        sources.read_marker()


def test_status(env, sources):
    assert sources.status().state == "in_package"
    (env.experimental / sources.MARKER_NAME).write_text(
        "tarball: t\nsha256: " + "0" * 64 + "\n"
    )
    state = sources.status()
    assert (state.state, state.root) == ("missing", env.root)
    env.root.mkdir(parents=True)
    (env.root / sources.MANIFEST_NAME).write_text(
        "# header\n" + "1" * 64 + "  a/b\n" + "2" * 64 + "  c\n"
    )
    state = sources.status()
    assert (state.state, state.files) == ("installed", 2)


# --- the install flow against a real _slim_wheel tarball --------------------------------------

DIST_INFO = "flashinfer_python-0.0.0.dist-info"
FILES = {
    "flashinfer/__init__.py": b"print('hi')\n",
    "flashinfer/experimental/__init__.py": b"",
    "flashinfer/experimental/foo/__init__.py": b"",
    "flashinfer/experimental/foo/build.json": b'{"programs": []}\n',
    "flashinfer/experimental/foo/csrc/kernel.cu": b"__global__ void k() {}\n" * 200,
    "flashinfer/experimental/foo/csrc/sub/kernel2.cu": b"// generated\n" * 50,
    "flashinfer/data/csrc/experimental/bar/generated/gemm.cu": b"// deepgemm\n" * 50,
    "flashinfer/data/csrc/experimental/cake_host_common.cuh": b"#pragma once\n",
    "flashinfer/data/csrc/norm/rmsnorm.cu": b"// stable kernel\n" * 50,
    "flashinfer/data/csrc/tvm_ffi_utils.h": b"#pragma once\n",
    f"{DIST_INFO}/METADATA": b"Metadata-Version: 2.1\nName: flashinfer-python\nVersion: 0.0.0\n",
    f"{DIST_INFO}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
}
DROPPED = sorted(
    n
    for n in FILES
    if n.startswith(
        ("flashinfer/experimental/foo/csrc/", "flashinfer/data/csrc/experimental/")
    )
)


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        rows = []
        for name, data in FILES.items():
            zf.writestr(zipfile.ZipInfo(name, date_time=(2026, 1, 2, 3, 4, 6)), data)
            digest = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            rows.append((name, f"sha256={digest}", str(len(data))))
        rows.append((f"{DIST_INFO}/RECORD", "", ""))
        text = io.StringIO()
        csv.writer(text, lineterminator="\n").writerows(rows)
        zf.writestr(f"{DIST_INFO}/RECORD", text.getvalue())


@pytest.fixture
def release(env, sources, build_backend, monkeypatch):
    """A slimmed wheel, its tarball, and the wheel's marker installed into the fake package."""
    monkeypatch.setenv(build_backend._EXPERIMENTAL_SOURCES_FLAG, "1")
    dist = env.tmp / "dist"
    dist.mkdir()
    wheel = dist / "flashinfer_python-0.0.0-py3-none-any.whl"
    _write_wheel(wheel)
    build_backend._slim_wheel(wheel)
    tarball = dist / "flashinfer_python-0.0.0-experimental-sources.tar.xz"
    assert tarball.is_file()
    with zipfile.ZipFile(wheel) as zf:
        marker = zf.read(build_backend._EXPERIMENTAL_SOURCES_MARKER)
    (env.experimental / sources.MARKER_NAME).write_bytes(marker)
    return SimpleNamespace(wheel=wheel, tarball=tarball, marker=marker.decode())


def _tree(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_install_from_local_tarball_lays_out_the_wheel_paths(env, sources, release):
    logs = []
    result = sources.install_sources(tarball=release.tarball, log=logs.append)
    assert (result.root, result.files, result.already_installed) == (
        env.root,
        len(DROPPED),
        False,
    )
    assert result.source == str(release.tarball)
    tree = _tree(env.root)
    assert tree.pop(sources.MANIFEST_NAME).startswith(b"# flashinfer-python 0.0.0:")
    assert tree == {name: FILES[name] for name in DROPPED}
    assert sources.verify_manifest(env.root) == (True, len(DROPPED), [])
    assert any("Verified sha256" in line for line in logs)
    # the loaders now resolve to the fetched tree
    loader = _loader(env, "foo")
    assert (
        sources.package_csrc_dir(loader)
        == env.root / "flashinfer" / "experimental" / "foo" / "csrc"
    )
    assert sources.shared_csrc_dir() == env.root / "flashinfer" / "data" / "csrc"
    assert sources.shared_include_dirs() == [
        env.root / "flashinfer" / "data" / "csrc",
        env.installed_csrc,
    ]
    state = sources.status()
    assert (state.state, state.files) == ("installed", len(DROPPED))
    # the tarball passed by the caller is left alone; nothing else is left beside the root
    assert release.tarball.is_file()
    assert [p.name for p in env.root.parent.iterdir()] == [VERSION]


def test_install_is_idempotent_and_replaces_a_modified_tree(
    env, sources, release, monkeypatch
):
    sources.install_sources(tarball=release.tarball)

    def no_network(*args, **kwargs):
        raise AssertionError("a verified tree must not trigger a download")

    monkeypatch.setattr(sources, "_fetch_url", no_network)
    again = sources.install_sources()
    assert again.already_installed and again.files == len(DROPPED)
    # a modified file fails verification and the next install replaces the tree
    victim = env.root / "flashinfer" / "experimental" / "foo" / "csrc" / "kernel.cu"
    victim.write_bytes(b"// edited\n")
    (env.root / "stray.txt").write_text("left over\n")
    assert sources.verify_manifest(env.root)[0] is False
    logs = []
    result = sources.install_sources(tarball=release.tarball, log=logs.append)
    assert not result.already_installed
    assert any("Replacing" in line and "modified" in line for line in logs)
    assert victim.read_bytes() == FILES["flashinfer/experimental/foo/csrc/kernel.cu"]
    assert not (env.root / "stray.txt").exists()
    # --force re-installs a healthy tree too
    forced = sources.install_sources(tarball=release.tarball, force=True)
    assert not forced.already_installed


def test_install_downloads_the_release_asset_of_the_installed_version(
    env, sources, release, monkeypatch
):
    fetched = []

    def fake_fetch(url, destination, log):
        fetched.append(url)
        shutil.copyfile(release.tarball, destination)

    monkeypatch.setattr(sources, "_fetch_url", fake_fetch)
    result = sources.install_sources()
    assert fetched == [
        f"{sources.RELEASE_DOWNLOAD_URL}/v{VERSION}/flashinfer_python-0.0.0-experimental-sources.tar.xz"
    ]
    assert result.source == fetched[0] and result.files == len(DROPPED)
    assert (
        _tree(env.root)["flashinfer/data/csrc/experimental/bar/generated/gemm.cu"]
        == FILES["flashinfer/data/csrc/experimental/bar/generated/gemm.cu"]
    )
    # the downloaded file is removed after unpacking; --url goes to the mirror as given
    assert [p.name for p in env.root.parent.iterdir()] == [VERSION]
    sources.install_sources(url="https://mirror.invalid/x.tar.xz", force=True)
    assert fetched[-1] == "https://mirror.invalid/x.tar.xz"


def test_install_refuses_a_tarball_that_does_not_match_the_marker(
    env, sources, release, monkeypatch
):
    corrupt = env.tmp / "corrupt.tar.xz"
    data = bytearray(release.tarball.read_bytes())
    data[-1] ^= 0xFF
    corrupt.write_bytes(data)
    with pytest.raises(sources.SourcesError, match="sha256"):
        sources.install_sources(tarball=corrupt)
    assert not env.root.exists()
    # a downloaded file that does not match is rejected and removed as well
    monkeypatch.setattr(
        sources, "_fetch_url", lambda url, dest, log: shutil.copyfile(corrupt, dest)
    )
    with pytest.raises(sources.SourcesError, match="sha256"):
        sources.install_sources()
    assert not env.root.exists()
    assert not list(env.root.parent.glob(".download.*"))
    # without a marker there is nothing to install; without a digest nothing is trusted
    marker = env.experimental / sources.MARKER_NAME
    marker.write_text("tarball: x.tar.xz\n")
    with pytest.raises(sources.SourcesError, match="no sha256"):
        sources.install_sources(tarball=release.tarball)
    marker.unlink()
    with pytest.raises(
        sources.SourcesError, match="ships its experimental kernel sources"
    ):
        sources.install_sources(tarball=release.tarball)


def _tarball(path: Path, members: list[tuple[str, bytes]], extra=None) -> Path:
    with tarfile.open(path, "w:xz") as tar:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            tar.addfile(extra)
    return path


@pytest.mark.parametrize(
    "member",
    [
        "flashinfer/experimental/../../etc/passwd",
        "/flashinfer/experimental/foo/csrc/k.cu",
        "flashinfer/experimental/foo/jit.py",  # a Python module, not a kernel source
        "flashinfer/data/csrc/norm/rmsnorm.cu",  # a stable kernel outside csrc/experimental
        "flashinfer/experimental/foo/csrc",  # a file where a directory is expected: too short
    ],
)
def test_unpack_rejects_members_outside_the_source_trees(env, sources, member):
    manifest = (sources.MANIFEST_NAME, b"# x\n")
    tarball = _tarball(env.tmp / "bad.tar.xz", [manifest, (member, b"x")])
    with pytest.raises(sources.SourcesError, match="unsafe|outside"):
        sources.unpack(tarball, env.root)
    assert not env.root.exists()
    assert list(env.root.parent.iterdir()) == []  # the staging directory is gone


def test_unpack_rejects_links_and_requires_the_manifest(env, sources):
    link = tarfile.TarInfo("flashinfer/experimental/foo/csrc/k.cu")
    link.type = tarfile.SYMTYPE
    link.linkname = "/etc/passwd"
    tarball = _tarball(
        env.tmp / "link.tar.xz", [(sources.MANIFEST_NAME, b"")], extra=link
    )
    with pytest.raises(sources.SourcesError, match="non-regular"):
        sources.unpack(tarball, env.root)
    tarball = _tarball(
        env.tmp / "nomanifest.tar.xz", [("flashinfer/experimental/foo/csrc/k.cu", b"x")]
    )
    with pytest.raises(sources.SourcesError, match=sources.MANIFEST_NAME):
        sources.unpack(tarball, env.root)
    (env.tmp / "notatar.tar.xz").write_bytes(b"not an archive")
    with pytest.raises(sources.SourcesError, match="not a readable tar archive"):
        sources.unpack(env.tmp / "notatar.tar.xz", env.root)
    assert not env.root.exists()


def test_unpack_accepts_nested_packages_and_directory_members(env, sources):
    members = [
        (sources.MANIFEST_NAME, b"# x\n"),
        ("flashinfer/experimental/cake/bar/csrc/k.cu", b"a"),
        ("flashinfer/data/csrc/experimental/cake_host_common.cuh", b"b"),
    ]
    directory = tarfile.TarInfo("flashinfer/experimental/cake/bar/csrc")
    directory.type = tarfile.DIRTYPE
    tarball = _tarball(env.tmp / "ok.tar.xz", members, extra=directory)
    assert sources.unpack(tarball, env.root) == 3
    assert _tree(env.root) == {name: data for name, data in members}
    assert (
        oct(
            os.stat(env.root / "flashinfer/experimental/cake/bar/csrc/k.cu").st_mode
            & 0o777
        )
        == "0o644"
    )
