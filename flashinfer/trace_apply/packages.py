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

Sealed kernel packages: content-addressed kernel loading for trace_apply.

A sealed package is a directory whose name is the sha256 of its own
``manifest.json``. The manifest lists every bundle file with its sha256 and
seals the exact file set with a sorted-listing digest, so the digest that
names a package pins every byte of it before anything is imported::

    <root>/<sha256(manifest.json)>/
        manifest.json        the directory name is the sha256 of these bytes
        <sidecar files>      verified bytes, opaque to FlashInfer (e.g. a
                             qualification receipt)
        bundle/
            solution.json    a FlashInfer Solution (trace/solution.py)
            ...              exactly the regular files the manifest lists

Two environment variables drive loading at ``import flashinfer`` time:
``FLASHINFER_KERNEL_PACKAGES`` (comma-separated package digests) and
``FLASHINFER_KERNEL_PACKAGE_ROOT`` (the root directory holding them,
defaulting to ``/opt/flashinfer/kernel_packages``). ``FLASHINFER_KERNEL_PLUGINS_DISABLE``
is a kill switch. The unverified curated-folder path
(``FLASHINFER_TRACE_APPLY_PATH``) remains available for development; sealed
packages are the production path because every byte they load is pinned by
the digest chain first.

Nothing here imports torch or the rest of FlashInfer at module scope: package
verification must run on a CPU-only build host. Importing a verified bundle
executes its code — verification is content authentication, not a sandbox.
Signatures are not checked (no signing infrastructure exists in FlashInfer
today); the manifest reserves ``signature`` and ``schema_version`` fields for
future use, and the trust anchor is the party that names the digest.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "API_VERSION",
    "DEFAULT_PACKAGE_ROOT",
    "DISABLE_ENV",
    "PACKAGES_ENV",
    "ROOT_ENV",
    "LoadResult",
    "SealedPackage",
    "SealedPackageError",
    "check_target_hardware",
    "import_entry",
    "load_packages",
    "verify_package",
]

#: Comma-separated sha256 digests of sealed packages to load and enable at
#: ``import flashinfer``. Unset or empty means inert.
PACKAGES_ENV = "FLASHINFER_KERNEL_PACKAGES"
#: Root directory holding the digest-named package directories. When unset but
#: digests are named, the default root below is used.
ROOT_ENV = "FLASHINFER_KERNEL_PACKAGE_ROOT"
#: Kill switch: when set to a truthy value, sealed loading refuses with a
#: warning even if the other variables are set.
DISABLE_ENV = "FLASHINFER_KERNEL_PLUGINS_DISABLE"
#: The default package root used when ``ROOT_ENV`` is unset but digests are named.
DEFAULT_PACKAGE_ROOT = Path("/opt/flashinfer/kernel_packages")
#: Version of this module's public surface, so consumers can feature-detect
#: (an import that resolves but lacks the expected API_VERSION never activates).
API_VERSION = 1

MANIFEST_NAME = "manifest.json"
BUNDLE_DIR = "bundle"
SOLUTION_NAME = "solution.json"
DEFINITION_NAME = "definition.json"

SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
# A Definition name: letters, digits, underscores, dots and hyphens, starting
# with a letter or digit (the same shape trace_apply's definition_name emits).
_DEFINITION_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
# ``sm100``, ``sm_100`` and ``sm100a`` all name compute capability 10.0.
_SM = re.compile(r"^sm_?(\d+)[a-z]?$")
# Languages the trace loaders import as Python modules: a mirror of
# ``_PYTHON_FAMILY`` in flashinfer/trace/solution.py (python, triton, cutedsl,
# cutile, tilelang), kept here so verification needs no FlashInfer import.
# The other families (cpp, cuda, cutlass) compile with nvcc at load, which a
# serving start must not depend on.
_PYTHON_FAMILY = frozenset({"python", "triton", "cutedsl", "cutile", "tilelang"})
_EXPECTED_FIELDS = frozenset({"operation", "target", "family"})


class SealedPackageError(RuntimeError):
    """A sealed package could not be authenticated or loaded.

    Deliberately a RuntimeError, never ImportError: engines that lazily catch
    ImportError around optional integrations must not read a sealed refusal
    as "integration unavailable" and silently fall back to stock kernels.
    """


@dataclass(frozen=True)
class SealedPackage:
    """A verified package: every byte accounted for by the named digest."""

    digest: str
    directory: Path
    manifest: dict[str, Any]
    solution: dict[str, Any]
    #: Verified sidecar files, by package-relative name (e.g. a qualification
    #: receipt). FlashInfer authenticates their bytes and never interprets
    #: them; consumers with domain metadata read them through this mapping.
    sealed_files: dict[str, Path] = field(default_factory=dict)
    definition: dict[str, Any] | None = None
    operation: str = ""
    target: str | None = None
    family: str | None = None
    #: (module path, symbol) of the declared entry point, parsed and validated.
    entry_point: tuple[str, str] = ("", "")

    @property
    def bundle_sha256(self) -> str:
        return str(self.manifest["bundle_sha256"])


@dataclass
class LoadResult:
    """What ``load_packages`` verified, imported and installed."""

    packages: list[SealedPackage] = field(default_factory=list)
    #: {definition name: flashinfer.trace Solution} passed to enable_apply.
    solutions: dict[str, Any] = field(default_factory=dict)
    #: {(definition, solution name): sealed-imported callable}, also registered
    #: as preloaded so the trace_apply build path never touches sealed sources.
    callables: dict[tuple[str, str], Callable] = field(default_factory=dict)
    #: {digest: reason} for every refused package (strict mode raises instead).
    errors: dict[str, str] = field(default_factory=dict)
    #: The enable_apply() return: number of wrapped attributes, 0 when disabled.
    applied: int = 0


# ---------------------------------------------------------------------------
# Digest-chain verification (standard library only)
# ---------------------------------------------------------------------------


def _required_text(value: Mapping[str, Any], key: str, path: Path) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise SealedPackageError(f"{path}: {key} must be a non-empty string")
    return item


def _sm_number(value: str) -> str | None:
    match = _SM.fullmatch(value.strip().lower())
    return match.group(1) if match else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SealedPackageError(f"{path} is not readable JSON: {error}") from error


def _refuse_link(path: Path) -> None:
    """Refuse a symbolic link where the package wants a real file or directory.

    Every other check follows ``path`` as the filesystem presents it, so a
    link would let the bytes under a sealed name live anywhere; an image
    build copies a package in, and a copy holds none.
    """
    if path.is_symlink():
        raise SealedPackageError(
            f"{path} is a symbolic link; a sealed package holds no links"
        )


def _canonical_relative(name: str, where: Path) -> None:
    relative = Path(name)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or str(relative) != name
        or "\0" in name
    ):
        raise SealedPackageError(f"{where} lists a non-canonical path {name!r}")


def _bundle_files(bundle: Path) -> set[str]:
    """Every regular file under ``bundle``, by bundle-relative name.

    Walked without following links: a link or a special file (a FIFO, a
    device) anywhere in the bundle is refused before anything is hashed —
    hashing a FIFO never returns.
    """

    def onerror(error: OSError) -> None:
        raise SealedPackageError(
            f"cannot inspect kernel bundle {bundle}: {error}"
        ) from error

    present: set[str] = set()
    for directory, dirnames, filenames in os.walk(
        bundle, followlinks=False, onerror=onerror
    ):
        for name in dirnames:
            _refuse_link(Path(directory, name))
        for name in filenames:
            path = Path(directory, name)
            _refuse_link(path)
            if not path.is_file():
                raise SealedPackageError(
                    f"{path} is not a regular file; a sealed package holds files only"
                )
            present.add(str(path.relative_to(bundle)))
    return present


def _metadata_constraint(
    name: str, documents: Sequence[tuple[Path, Mapping[str, Any]]]
) -> str | None:
    """Read one identity field from authenticated documents, requiring agreement."""
    values = [str(doc[name]) for _, doc in documents if name in doc]
    if len(set(values)) > 1:
        raise SealedPackageError(f"sealed metadata has conflicting {name}: {values}")
    return values[0] if values else None


def _sealed_file_digests(
    manifest: Mapping[str, Any], manifest_path: Path
) -> dict[str, str]:
    """Sidecar digests from ``sealed_files``, with the legacy qualification alias.

    An earlier package format sealed a single qualification receipt
    via the top-level ``qualification_sha256`` key; that spelling is accepted
    as ``sealed_files = {"qualification.json": ...}`` so existing packages
    verify byte-for-byte.
    """
    sealed: dict[str, str] = {}
    legacy = manifest.get("qualification_sha256")
    if legacy is not None:
        sealed["qualification.json"] = str(legacy)
    declared = manifest.get("sealed_files", {})
    if not isinstance(declared, dict) or not all(
        isinstance(name, str) and isinstance(digest, str)
        for name, digest in declared.items()
    ):
        raise SealedPackageError(
            f"{manifest_path}: sealed_files must map names to sha256 digests"
        )
    for name, digest in declared.items():
        if name in sealed and sealed[name] != digest:
            raise SealedPackageError(
                f"{manifest_path}: {name!r} is sealed twice with conflicting digests"
            )
        sealed[name] = digest
    for name, digest in sealed.items():
        _canonical_relative(name, manifest_path)
        if not SHA256_HEX.fullmatch(digest):
            raise SealedPackageError(
                f"{manifest_path}: {name} is not sealed with a sha256 digest"
            )
    return sealed


def _entry_point(
    solution: Mapping[str, Any], files: Mapping[str, Any]
) -> tuple[str, str]:
    """Parse and validate ``spec.entry_point`` into a (module path, symbol) pair."""
    spec = solution.get("spec")
    entry = spec.get("entry_point") if isinstance(spec, dict) else None
    if not isinstance(entry, str):
        raise SealedPackageError("sealed spec must declare an entry_point")
    if "::" in entry:
        filename, symbol = entry.split("::", 1)
        path = Path(filename)
        if path.suffix != ".py" or path.is_absolute() or str(path) != filename:
            raise SealedPackageError(f"invalid Python entry_point {entry!r}")
        parts = path.with_suffix("").parts
        module_parts = parts[:-1] if parts[-1:] == ("__init__",) else parts
        if module_parts == parts and "/".join(parts) + "/__init__.py" in files:
            raise SealedPackageError(
                f"entry_point file {filename!r} is shadowed by a package"
            )
    else:
        module, separator, symbol = entry.partition(":")
        if not separator:
            raise SealedPackageError(f"invalid Python entry_point {entry!r}")
        parts = module_parts = tuple(module.split("."))
        package_file = "/".join(parts) + "/__init__.py"
        filename = package_file if package_file in files else "/".join(parts) + ".py"
    if (
        not parts
        or not all(part.isidentifier() for part in parts)
        or not symbol.isidentifier()
    ):
        raise SealedPackageError(f"invalid Python entry_point {entry!r}")
    if filename not in files:
        raise SealedPackageError(
            f"entry_point file {filename!r} is not sealed in the bundle"
        )
    return ".".join(module_parts), symbol


def verify_package(
    root: Path, digest: str, *, expected: Mapping[str, str] | None = None
) -> SealedPackage:
    """Authenticate every package byte before resolving identity, without importing code.

    The package directory name must equal the sha256 of its manifest; the
    manifest lists the exact bundle file set with per-file digests and a
    sorted-listing ``bundle_sha256``; sidecar files outside the bundle are
    authenticated by ``sealed_files`` (or the legacy ``qualification_sha256``)
    and returned as opaque verified paths. ``expected`` optionally constrains
    the identity fields ``operation``, ``target`` and ``family`` as resolved
    from the sealed documents.
    """
    if not isinstance(digest, str) or not SHA256_HEX.fullmatch(digest):
        raise SealedPackageError(
            f"package digest must be 64 lowercase hex characters, got {digest!r}"
        )
    unknown = sorted(set(expected or {}) - _EXPECTED_FIELDS)
    if unknown:
        raise SealedPackageError(f"unknown identity constraint(s) {unknown}")

    directory = root / digest
    manifest_path = directory / MANIFEST_NAME
    _refuse_link(directory)
    _refuse_link(manifest_path)
    if not manifest_path.is_file():
        raise SealedPackageError(
            f"no sealed package {digest} under {root}; the image does not carry the package named"
        )
    if (actual := _sha256_file(manifest_path)) != digest:
        raise SealedPackageError(
            f"{manifest_path} hashes to {actual}, not the named digest {digest}"
        )
    for child in directory.iterdir():
        _refuse_link(child)
        if not child.is_file() and not child.is_dir():
            raise SealedPackageError(f"{child} is not a regular file or directory")

    manifest = _read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SealedPackageError(f"{manifest_path} must hold a JSON object")
    schema_version = manifest.get("schema_version", 1)
    if schema_version != 1:
        raise SealedPackageError(
            f"{manifest_path}: unsupported schema_version {schema_version!r}"
        )

    files = manifest.get("files")
    listed_digest = manifest.get("bundle_sha256")
    if (
        not isinstance(files, dict)
        or not files
        or not all(
            isinstance(name, str)
            and name
            and isinstance(digest_of_file, str)
            and SHA256_HEX.fullmatch(digest_of_file)
            for name, digest_of_file in files.items()
        )
        or not isinstance(listed_digest, str)
        or not SHA256_HEX.fullmatch(listed_digest)
    ):
        raise SealedPackageError(
            f"{manifest_path} must list bundle files with their sha256 digests"
        )
    listing = "".join(
        f"{name}\0{digest_of_file}\n" for name, digest_of_file in sorted(files.items())
    )
    if hashlib.sha256(listing.encode("utf-8")).hexdigest() != listed_digest:
        raise SealedPackageError(
            f"{manifest_path}: bundle_sha256 does not match its own file listing"
        )
    declared_dir = manifest.get("bundle_dir", BUNDLE_DIR)
    if declared_dir != BUNDLE_DIR:
        raise SealedPackageError(f"{manifest_path}: bundle_dir must be {BUNDLE_DIR!r}")

    bundle = directory / BUNDLE_DIR
    _refuse_link(bundle)
    if not bundle.is_dir():
        raise SealedPackageError(f"{bundle} is not a bundle directory")
    present = _bundle_files(bundle)
    for name, digest_of_file in sorted(files.items()):
        _canonical_relative(name, manifest_path)
        path = bundle / Path(name)
        if not path.is_file():
            raise SealedPackageError(f"bundle file {name!r} is missing from {bundle}")
        if (actual := _sha256_file(path)) != digest_of_file:
            raise SealedPackageError(
                f"bundle file {name!r} hashes to {actual}, not the {digest_of_file} its manifest sealed"
            )
    if extra := sorted(present - set(files)):
        raise SealedPackageError(
            f"bundle holds files its manifest never sealed: {', '.join(extra)}"
        )

    sealed_files: dict[str, Path] = {}
    for name, digest_of_file in sorted(
        _sealed_file_digests(manifest, manifest_path).items()
    ):
        path = directory / Path(name)
        _refuse_link(path)
        if not path.is_file():
            raise SealedPackageError(
                f"sealed file {name!r} is missing beside {manifest_path}"
            )
        if (actual := _sha256_file(path)) != digest_of_file:
            raise SealedPackageError(
                f"sealed file {name!r} hashes to {actual}, not the {digest_of_file} its manifest sealed"
            )
        sealed_files[name] = path

    documents: dict[str, dict[str, Any]] = {}
    for name in (DEFINITION_NAME, SOLUTION_NAME):
        if name in files:
            content = _read_json(bundle / name)
            if not isinstance(content, dict):
                raise SealedPackageError(f"{name} must hold a JSON object")
            documents[name] = content
    metadata: list[tuple[Path, Mapping[str, Any]]] = [
        (manifest_path, manifest),
        *((bundle / name, doc) for name, doc in documents.items()),
    ]
    solution = documents.get(SOLUTION_NAME)
    if solution is None:
        raise SealedPackageError(f"bundle lists no {SOLUTION_NAME}")

    operation = solution.get("definition")
    if not isinstance(operation, str) or not _DEFINITION_NAME.fullmatch(operation):
        raise SealedPackageError(
            f"{SOLUTION_NAME}: definition must be a Definition name"
        )
    # The solution's definition is the operation; sealed documents must agree on it.
    _metadata_constraint(
        "operation", [*metadata, (bundle / SOLUTION_NAME, {"operation": operation})]
    )
    target = _metadata_constraint("target", metadata)
    family = _metadata_constraint("family", metadata)
    for key, wanted in (expected or {}).items():
        actual = {"operation": operation, "target": target, "family": family}[key]
        if wanted != actual:
            raise SealedPackageError(
                f"sealed package has {key} {actual!r}, not the {key} {wanted!r}"
            )

    spec = solution.get("spec")
    if not isinstance(spec, dict):
        raise SealedPackageError(f"{SOLUTION_NAME}: spec must be a JSON object")
    if "languages" in spec and "language" not in spec:
        raise SealedPackageError(
            f"{SOLUTION_NAME} is a SOLBench/Kernel Factory solution (spec.languages "
            f"{spec['languages']!r}); the package must carry a FlashInfer Solution, "
            "with spec.language in FlashInfer's spelling such as cutedsl and sm "
            "targets such as sm100, and nothing here translates one into the other"
        )
    language = spec.get("language")
    if not isinstance(language, str) or language not in _PYTHON_FAMILY:
        raise SealedPackageError(
            f"{SOLUTION_NAME} is a {language!r} solution; only Python-family "
            "solutions activate, a compiled family would need nvcc at serving start"
        )
    entry_point = _entry_point(solution, files)
    return SealedPackage(
        digest=digest,
        directory=directory,
        manifest=manifest,
        solution=solution,
        sealed_files=sealed_files,
        definition=documents.get(DEFINITION_NAME),
        operation=operation,
        target=target,
        family=family,
        entry_point=entry_point,
    )


# ---------------------------------------------------------------------------
# Sealed import: execute exactly the verified bytes, once, under a private namespace
# ---------------------------------------------------------------------------


class _SourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname: str, filename: str, digest: str) -> None:
        super().__init__(fullname, filename)
        self.digest = digest

    def get_code(self, fullname: str) -> Any:
        # Deferred imports must authenticate the bytes they compile, too.
        path = Path(self.path)
        _refuse_link(path)
        if not path.is_file():
            raise SealedPackageError(f"{path} is not a regular sealed source file")
        try:
            source = self.get_data(self.path)
        except OSError as error:
            raise SealedPackageError(
                f"cannot read sealed source {path}: {error}"
            ) from error
        if hashlib.sha256(source).hexdigest() != self.digest:
            raise SealedPackageError(f"{path} changed after package verification")
        # Neither read nor create __pycache__ inside an exact-file-set bundle.
        return self.source_to_code(source, self.path)


class _BundleFinder(importlib.abc.MetaPathFinder):
    def __init__(self, namespace: str, bundle: Path, files: Mapping[str, str]) -> None:
        self.namespace, self.bundle, self.files = namespace, bundle, dict(files)

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname != self.namespace and not fullname.startswith(self.namespace + "."):
            return None
        relative = fullname[len(self.namespace) :].lstrip(".").replace(".", "/")
        directory = self.bundle / relative
        for name, package in (
            (f"{relative}/__init__.py".lstrip("/"), True),
            (f"{relative}.py", False),
        ):
            if name in self.files:
                for parent in (
                    self.bundle.parent,
                    *(self.bundle / part for part in Path(name).parents),
                ):
                    _refuse_link(parent)
                filename = str(self.bundle / name)
                return importlib.util.spec_from_file_location(
                    fullname,
                    filename,
                    loader=_SourceLoader(fullname, filename, self.files[name]),
                    submodule_search_locations=[str(directory)] if package else None,
                )
        if directory.is_dir():
            spec = importlib.machinery.ModuleSpec(
                fullname, loader=None, is_package=True
            )
            spec.submodule_search_locations = [str(directory)]
            return spec
        # Do not let another finder discover an unsealed file in our namespace.
        # Never ImportError: a deferred refusal must not be readable as an
        # unavailable optional dependency by the caller of the sealed code.
        raise SealedPackageError(f"no sealed module {fullname!r}")


def import_entry(package: SealedPackage) -> Callable[..., Any]:
    """Import the declared entry point, then re-authenticate the whole package.

    The bundle is imported under a digest-and-PID-private namespace so
    relative imports stay inside the package and no two packages (or a
    parent and its forked child) share module state. Every deferred import
    re-hashes the bytes it compiles; importing selected code executes it —
    verification is content authentication, not a sandbox.
    """
    bundle = (package.directory / BUNDLE_DIR).resolve()
    for source in package.solution.get("sources", ()):
        if (
            not isinstance(source, dict)
            or source.get("path") not in package.manifest["files"]
        ):
            raise SealedPackageError("solution source is not a sealed bundle file")
        if (bundle / source["path"]).read_text(encoding="utf-8") != source.get(
            "content"
        ):
            raise SealedPackageError(
                f"embedded solution source differs from {source['path']!r}"
            )
    try:
        module, symbol = package.entry_point or _entry_point(
            package.solution, package.manifest["files"]
        )
        namespace = f"_flashinfer_sealed_{package.digest[:16]}_{os.getpid()}"
        finder = next(
            (
                item
                for item in sys.meta_path
                if isinstance(item, _BundleFinder) and item.namespace == namespace
            ),
            None,
        )
        if finder is not None and finder.bundle != bundle:
            raise SealedPackageError(
                "this package digest was already imported from another directory"
            )
        if finder is None:
            sys.meta_path.insert(
                0, _BundleFinder(namespace, bundle, package.manifest["files"])
            )
        imported = importlib.import_module(
            f"{namespace}.{module}" if module else namespace
        )
        function = getattr(imported, symbol, None)
        if not callable(function):
            raise SealedPackageError("declared entry_point is not callable")
    except (ImportError, AttributeError, OSError, SyntaxError) as error:
        raise SealedPackageError(f"cannot load sealed entry_point: {error}") from error
    finally:
        verify_package(package.directory.parent, package.digest)
    return function


# ---------------------------------------------------------------------------
# Hardware gate and installation
# ---------------------------------------------------------------------------


def check_target_hardware(solution: Mapping[str, Any], *, device: Any = None) -> None:
    """Refuse a Solution whose ``spec.target_hardware`` excludes this GPU.

    ``apply.current_sm`` answers None on a machine with no visible GPU (a
    build host, a CPU-only container): the check is skipped then, so callers
    that must not skip pass an explicit device.
    """
    from .apply import current_sm

    sm = current_sm(device)
    if sm is None:
        return
    hardware = (solution.get("spec") or {}).get("target_hardware") or []
    if all(
        (number := _sm_number(str(item))) is not None and number != sm
        for item in hardware
    ):
        raise SealedPackageError(
            f"sealed solution targets {list(hardware)}, not this GPU (sm{sm})"
        )


def load_packages(
    root: Path,
    digests: Sequence[str],
    *,
    strict: bool = True,
    enable: bool = True,
    sm_check: bool = True,
) -> LoadResult:
    """Verify, import and install sealed packages; install only after all verify.

    With ``strict`` (default) the first refusal raises; otherwise each refusal
    is collected in ``LoadResult.errors`` and the surviving packages install.
    Installation hands one ``{definition: Solution}`` mapping to
    ``trace_apply.enable_apply`` and registers every imported callable as
    preloaded, so the trace build path never materializes sealed sources.
    """
    result = LoadResult()
    solutions: dict[str, Any] = {}
    for digest in digests:
        try:
            package = verify_package(Path(root), digest)
            if sm_check:
                check_target_hardware(package.solution)
            entry = import_entry(package)
            from flashinfer.trace import (
                Solution,
            )  # after verification: the runtime is trusted now

            if package.operation in solutions:
                raise SealedPackageError(
                    f"two sealed packages claim the same definition {package.operation!r}"
                )
            solutions[package.operation] = Solution.from_dict(package.solution)
            result.packages.append(package)
            result.callables[
                (package.operation, str(solutions[package.operation].name))
            ] = entry
        except SealedPackageError as error:
            if strict:
                raise
            result.errors[digest] = f"{type(error).__name__}: {error}"
    result.solutions = solutions
    if enable and solutions:
        from .apply import enable_apply, register_preloaded

        for (definition, name), fn in result.callables.items():
            register_preloaded(definition, name, fn)
        result.applied = enable_apply(solutions)
    return result


def _env_digests(environ: Mapping[str, str]) -> list[str]:
    raw = environ.get(PACKAGES_ENV, "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()] if raw else []


def _load_packages_from_env() -> None:
    """Import-time hook: load sealed packages named by the environment.

    Internal — called only by the ``flashinfer`` package import. Warning-only:
    a bad package never makes ``import flashinfer`` fail. The kill switch and
    the coexistence warnings are handled here.
    """
    import logging
    import warnings

    logger = logging.getLogger("flashinfer.trace_apply")
    environ = os.environ
    digests = _env_digests(environ)
    if not digests:
        if environ.get(ROOT_ENV):
            logger.warning(
                "%s is set but %s names no package digests; nothing loaded",
                ROOT_ENV,
                PACKAGES_ENV,
            )
        return
    if environ.get(DISABLE_ENV, "") not in ("", "0", "false", "False"):
        warnings.warn(
            f"{DISABLE_ENV} is set; sealed kernel packages not loaded.", stacklevel=2
        )
        return
    if environ.get("FLASHINFER_CUBIN_CHECKSUM_DISABLED"):
        logger.warning(
            "%s is set beside sealed kernel packages; package bytes are still digest-verified, "
            "but cubin checksums are globally disabled",
            "FLASHINFER_CUBIN_CHECKSUM_DISABLED",
        )
    if environ.get("FLASHINFER_TRACE_APPLY_PATH"):
        logger.warning(
            "FLASHINFER_TRACE_APPLY_PATH curated solutions are unverified and intended for "
            "development; sealed kernel packages are the verified production path"
        )
    root = environ.get(ROOT_ENV) or str(DEFAULT_PACKAGE_ROOT)
    result = load_packages(Path(root), digests, strict=False)
    for digest, reason in result.errors.items():
        warnings.warn(f"sealed kernel package {digest} refused: {reason}", stacklevel=2)
    if result.applied == 0 and not result.errors:
        warnings.warn(
            f"{PACKAGES_ENV} named {len(digests)} package(s) but none installed; "
            f"check the root {root}",
            stacklevel=2,
        )
    elif result.packages:
        logger.info(
            "sealed kernel packages: %d loaded, %d refused, %d attribute(s) wrapped",
            len(result.packages),
            len(result.errors),
            result.applied,
        )
