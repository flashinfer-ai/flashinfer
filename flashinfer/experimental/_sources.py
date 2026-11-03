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

# Where the kernel sources of an experimental package are found.
#
# Release wheels of flashinfer-python leave the experimental packages' generated
# kernel sources out (flashinfer/experimental/<pkg>/csrc/** and
# flashinfer/data/csrc/experimental/**; build_backend._slim_wheel, #7977) and
# write the dropped entries to a tarball published as an asset of the GitHub
# release. The wheel carries KERNEL_SOURCES_EXCLUDED.txt next to this module,
# naming that tarball and its SHA-256. `flashinfer download-experimental-sources`
# fetches and unpacks it under the FlashInfer cache (or
# FLASHINFER_EXPERIMENTAL_SOURCES_DIR), and the loaders below resolve their
# sources through this module: the in-package copy when it is present (source
# checkouts, nightly wheels), else the fetched copy, else an error that says
# how to get them. Nothing here writes into the installed package directory.

from __future__ import annotations

import hashlib
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

# Written into the wheel by build_backend._slim_wheel, next to this module.
MARKER_NAME = "KERNEL_SOURCES_EXCLUDED.txt"
# First member of the tarball: `<sha256>  <path>` per source (sha256sum -c format).
MANIFEST_NAME = "EXPERIMENTAL_SOURCES_MANIFEST.txt"
# The one override: the directory the tarball is unpacked into. The default
# lives under FLASHINFER_WORKSPACE_BASE with the rest of the cache; this is for
# hosts whose cache must stay node-local (JIT outputs) while the sources come
# from a shared, possibly read-only, mount that an administrator populated.
SOURCES_DIR_ENV = "FLASHINFER_EXPERIMENTAL_SOURCES_DIR"
RELEASE_DOWNLOAD_URL = "https://github.com/flashinfer-ai/flashinfer/releases/download"
COMMAND = "flashinfer download-experimental-sources"

# The wheel paths the tarball holds, the same prefixes build_backend drops.
PACKAGE_PREFIX = "flashinfer/experimental/"
SHARED_PREFIX = "flashinfer/data/csrc/experimental/"

_EXPERIMENTAL_DIR = Path(__file__).resolve().parent
_MANIFEST_LINE = re.compile(r"^([0-9a-f]{64})  (\S.*)$")


class ExperimentalSourcesNotFound(FileNotFoundError):
    """An experimental package's kernel sources are neither in the package nor fetched."""


class SourcesError(RuntimeError):
    """`flashinfer download-experimental-sources` cannot proceed (reported as-is)."""


# --- indirections the CPU tests replace (the real ones import torch through flashinfer.jit)


def _version() -> str:
    from ..version import __version__

    return __version__


def _cache_dir() -> Path:
    from ..jit import env as jit_env

    return Path(jit_env.FLASHINFER_CACHE_DIR)


def _installed_csrc_dir() -> Path:
    from ..jit import env as jit_env

    return Path(jit_env.FLASHINFER_CSRC_DIR)


# --- resolution


def sources_root() -> Path:
    """The directory the release tarball is (to be) unpacked into.

    ``$FLASHINFER_EXPERIMENTAL_SOURCES_DIR`` when set, else
    ``<FLASHINFER_CACHE_DIR>/experimental-sources/<flashinfer version>``: one
    tree per version, so an upgrade never compiles another version's sources.
    """
    override = os.environ.get(SOURCES_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return _cache_dir() / "experimental-sources" / _version()


def marker_path() -> Path:
    return _EXPERIMENTAL_DIR / MARKER_NAME


@dataclass(frozen=True)
class Marker:
    """The wheel's KERNEL_SOURCES_EXCLUDED.txt: the tarball it was built without."""

    tarball: Optional[str]
    sha256: Optional[str]
    text: str


def read_marker(path: Optional[Path] = None) -> Optional[Marker]:
    """The marker of this installation, or ``None`` when the wheel ships its sources."""
    path = marker_path() if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    fields: dict[str, str] = {}
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in ("tarball", "sha256") and key not in fields:
            fields[key] = value.strip()
    sha256 = fields.get("sha256")
    if sha256 is not None and not re.fullmatch(r"[0-9a-f]{64}", sha256):
        raise SourcesError(f"{path} carries a malformed sha256 line: {sha256!r}")
    return Marker(fields.get("tarball") or None, sha256 or None, text)


def _missing(what: str, looked: Iterable[Path]) -> ExperimentalSourcesNotFound:
    looked_at = ", ".join(str(p) for p in looked)
    return ExperimentalSourcesNotFound(
        f"The kernel sources of {what} are not installed (looked at {looked_at}). "
        "Release wheels of flashinfer-python leave the experimental kernel sources "
        f"out (flashinfer/experimental/{MARKER_NAME} in the installed package says "
        f"so). Run `{COMMAND}` to fetch the matching tarball from the GitHub release "
        f"into {sources_root()}, or unpack that tarball there by hand (set "
        f"{SOURCES_DIR_ENV} to use another directory); see "
        "flashinfer/experimental/README.md."
    )


def package_csrc_dir(package_file: str | os.PathLike[str]) -> Path:
    """The ``csrc`` directory of the experimental package containing ``package_file``.

    Loaders pass their ``__file__``. The package's own ``csrc/`` wins when it is
    present; otherwise the fetched copy under :func:`sources_root`, which keeps
    the wheel layout (``flashinfer/experimental/<pkg>/csrc``). Raises
    :class:`ExperimentalSourcesNotFound` (a ``FileNotFoundError``) otherwise.
    """
    package_dir = Path(package_file).resolve().parent
    in_package = package_dir / "csrc"
    if in_package.is_dir():
        return in_package
    try:
        relative = package_dir.relative_to(_EXPERIMENTAL_DIR)
    except ValueError:
        # Every wheel ships the kernel sources of packages outside
        # flashinfer/experimental (nothing is excluded there), so there is no
        # fetched copy to fall back to: this is a broken installation.
        raise ExperimentalSourcesNotFound(
            f"{package_dir} is not under {_EXPERIMENTAL_DIR} and its kernel sources "
            f"({in_package}) are missing from the installation"
        ) from None
    fetched = sources_root() / PACKAGE_PREFIX / relative / "csrc"
    if fetched.is_dir():
        return fetched
    raise _missing(
        f"flashinfer.experimental.{'.'.join(relative.parts)}", (in_package, fetched)
    )


def shared_csrc_dir() -> Path:
    """The ``csrc`` root holding ``experimental/`` (the generated DeepGEMM-derived programs).

    ``FLASHINFER_CSRC_DIR`` when its ``experimental/`` is present, else the
    fetched copy (``<root>/flashinfer/data/csrc``). Loaders join the catalog's
    ``experimental/...`` paths onto it where they used ``FLASHINFER_CSRC_DIR``.
    """
    installed = _installed_csrc_dir()
    if (installed / "experimental").is_dir():
        return installed
    fetched = sources_root() / "flashinfer" / "data" / "csrc"
    if (fetched / "experimental").is_dir():
        return fetched
    raise _missing(
        "the generated programs under csrc/experimental",
        (installed / "experimental", fetched / "experimental"),
    )


def shared_include_dirs() -> list[Path]:
    """What replaces ``[FLASHINFER_CSRC_DIR]`` in those loaders' include paths.

    The root that holds ``experimental/`` (its headers are included as
    ``"experimental/cake_host_common.cuh"``) first, then the installed ``csrc``
    for the binding headers (``tvm_ffi_utils.h``) when the two differ.
    """
    root = shared_csrc_dir()
    installed = _installed_csrc_dir()
    return [root] if root == installed else [root, installed]


# --- the fetch path (`flashinfer download-experimental-sources`)


@dataclass(frozen=True)
class SourcesStatus:
    #: "in_package" (the wheel ships its sources), "installed" (fetched tree
    #: present) or "missing".
    state: str
    root: Optional[Path]
    marker: Optional[Marker]
    files: Optional[int] = None


def status(root: Optional[Path] = None) -> SourcesStatus:
    """What `flashinfer show-config` reports; never raises on a damaged marker."""
    try:
        marker = read_marker()
    except SourcesError:
        marker = Marker(None, None, "")
    if marker is None:
        return SourcesStatus("in_package", None, None)
    root = sources_root() if root is None else root
    try:
        entries = _read_manifest(root / MANIFEST_NAME)
    except (FileNotFoundError, SourcesError):
        return SourcesStatus("missing", root, marker)
    return SourcesStatus("installed", root, marker, len(entries))


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        match = _MANIFEST_LINE.match(line)
        if match is None:
            raise SourcesError(f"{path} has a malformed line: {line!r}")
        entries.append((match.group(1), match.group(2)))
    return entries


def verify_manifest(root: Path) -> tuple[bool, int, list[str]]:
    """Whether every file the manifest under ``root`` lists is present with its digest."""
    try:
        entries = _read_manifest(root / MANIFEST_NAME)
    except FileNotFoundError:
        return False, 0, [f"{root / MANIFEST_NAME} is missing"]
    problems = []
    for digest, name in entries:
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif _sha256_file(path) != digest:
            problems.append(f"modified: {name}")
    return not problems, len(entries), problems


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tarball_url(marker: Marker, version: Optional[str] = None) -> str:
    """The release asset of the installed version: ``<releases>/v<version>/<tarball>``."""
    if not marker.tarball:
        raise SourcesError(
            f"{marker_path()} names no sources tarball: this wheel predates "
            f"`{COMMAND}`. Install a nightly wheel from https://flashinfer.ai/whl "
            "or install from source to use experimental features."
        )
    version = _version() if version is None else version
    return f"{RELEASE_DOWNLOAD_URL}/v{version}/{marker.tarball}"


def _fetch_url(url: str, destination: Path, log: Callable[[str], Any]) -> None:
    """Stream ``url`` into ``destination`` (``requests`` follows GitHub's redirect)."""
    import requests  # type: ignore[import-untyped]

    log(f"Downloading {url}")
    with requests.get(url, stream=True, timeout=(10, 120)) as response:
        response.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in response.iter_content(1 << 20):
                f.write(chunk)


def _check_member(member: tarfile.TarInfo) -> str:
    """The member's path under the unpack root, or raise. Regular files only, no
    links or devices, relative paths without ``..`` inside the two source trees."""
    name = member.name
    if member.isdir():
        return ""
    if not member.isfile():
        raise SourcesError(f"the tarball holds a non-regular member: {name!r}")
    parts = Path(name).parts
    if (
        not name
        or name.startswith(("/", "\\"))
        or ".." in parts
        or "\\" in name
        or any(part in ("", ".") for part in parts)
    ):
        raise SourcesError(f"the tarball holds an unsafe member path: {name!r}")
    if name == MANIFEST_NAME:
        return name
    # flashinfer/experimental/<pkg>/csrc/... (also nested: .../cake/<pkg>/csrc/...)
    if name.startswith(PACKAGE_PREFIX) and "csrc" in parts[3:-1]:
        return name
    if name.startswith(SHARED_PREFIX) and len(parts) >= 5:
        return name
    raise SourcesError(
        f"the tarball holds a member outside the experimental source trees: {name!r}"
    )


def unpack(tarball: Path, root: Path) -> int:
    """Unpack ``tarball`` into ``root`` (replacing it atomically); the member count."""
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.", suffix=".tmp", dir=root.parent)
    )
    count = 0
    try:
        try:
            with tarfile.open(tarball, "r:*") as tar:
                members = tar.getmembers()
                names = [_check_member(member) for member in members]
                if MANIFEST_NAME not in names:
                    raise SourcesError(f"{tarball} carries no {MANIFEST_NAME}")
                for member, name in zip(members, names, strict=True):
                    if not name:
                        continue
                    target = staging / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    assert source is not None
                    with source, open(target, "wb") as f:
                        shutil.copyfileobj(source, f)
                    os.chmod(target, 0o644)
                    count += 1
        except tarfile.TarError as error:
            raise SourcesError(
                f"{tarball} is not a readable tar archive: {error}"
            ) from error
        if root.is_symlink() or root.is_file():
            root.unlink()
        elif root.is_dir():
            shutil.rmtree(root)
        staging.rename(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return count


@dataclass(frozen=True)
class InstallResult:
    root: Path
    files: int
    source: str
    #: True when the tree was already present and verified; nothing was fetched.
    already_installed: bool = False


def install_sources(
    *,
    tarball: Optional[Path] = None,
    url: Optional[str] = None,
    root: Optional[Path] = None,
    force: bool = False,
    log: Callable[[str], Any] = lambda message: None,
) -> InstallResult:
    """Fetch (or take ``tarball``), verify against the marker and unpack into ``root``.

    A present tree that verifies against its manifest is kept unless ``force``;
    one that does not verify is replaced. The downloaded file is deleted after
    unpacking; a caller-supplied ``tarball`` is left alone.
    """
    marker = read_marker()
    if marker is None:
        raise SourcesError(
            "this FlashInfer installation ships its experimental kernel sources "
            f"(no flashinfer/experimental/{MARKER_NAME}); nothing to download."
        )
    root = sources_root() if root is None else root
    if not force and root.exists():
        ok, files, problems = verify_manifest(root)
        if ok:
            return InstallResult(root, files, str(root), already_installed=True)
        log(
            f"Replacing {root}: {problems[0]}"
            + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
        )
    if marker.sha256 is None:
        raise SourcesError(
            f"{marker_path()} records no sha256 for the sources tarball; refusing "
            "to install unverified sources."
        )
    downloaded: Optional[Path] = None
    try:
        if tarball is None:
            url = tarball_url(marker) if url is None else url
            root.parent.mkdir(parents=True, exist_ok=True)
            fd, name = tempfile.mkstemp(
                prefix=".download.", suffix=".tar.xz", dir=root.parent
            )
            os.close(fd)
            downloaded = Path(name)
            _fetch_url(url, downloaded, log)
            tarball, source = downloaded, url
        else:
            tarball, source = Path(tarball), str(tarball)
        digest = _sha256_file(tarball)
        if digest != marker.sha256:
            raise SourcesError(
                f"{source} does not match the tarball this wheel was built without: "
                f"sha256 {digest} != {marker.sha256} (from {marker_path()})."
            )
        log(f"Verified sha256 {digest}")
        files = unpack(tarball, root) - 1  # the manifest itself is not a source
    finally:
        if downloaded is not None and downloaded.exists():
            downloaded.unlink()
    return InstallResult(root, files, source)


__all__ = [
    "COMMAND",
    "MANIFEST_NAME",
    "MARKER_NAME",
    "SOURCES_DIR_ENV",
    "ExperimentalSourcesNotFound",
    "InstallResult",
    "Marker",
    "SourcesError",
    "SourcesStatus",
    "install_sources",
    "marker_path",
    "package_csrc_dir",
    "read_marker",
    "shared_csrc_dir",
    "shared_include_dirs",
    "sources_root",
    "status",
    "tarball_url",
    "unpack",
    "verify_manifest",
]
