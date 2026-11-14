# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flashinfer.trace_apply.packages — sealed kernel packages (no GPU).

Verification is content authentication, not a sandbox: importing a verified
bundle executes its code. Every test runs on the CPU against packages sealed
on the fly by ``seal_helpers``, plus the preserved golden package."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import flashinfer.trace_apply as ta
from flashinfer.trace_apply import packages as pk
from flashinfer.trace_apply.apply import (
    _enable_apply_from_env,
    _get_loaded_solution,
    clear_preloaded,
    preloaded_entry,
    register_preloaded,
)
from tests.cpu.trace_apply.seal_helpers import (
    GOLDEN_DIGEST,
    GOLDEN_ROOT,
    OPERATION,
    QUALIFICATION,
    SOLUTION,
    SOURCE,
    json_bytes,
    listing_sha256,
    seal_flashinfer_fixture,
    seal_package,
)


@pytest.fixture(autouse=True)
def clean_install():
    """Release sealed packages and disable Trace Apply after every test."""
    yield
    pk.release_packages()
    if ta.is_enabled():
        ta.disable_apply()


@pytest.fixture(autouse=True)
def private_imports(monkeypatch):
    """Restore ``sys.meta_path`` and drop sealed modules imported during a test."""
    before = set(sys.modules)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    yield
    for name in set(sys.modules) - before:
        if name.startswith("_flashinfer_sealed_"):
            del sys.modules[name]


def package(root, files, entry="kernel.py::run"):
    # Embedded sources mirror the sealed bytes exactly, so the packages these
    # tests build are loadable end to end (Solution.from_dict requires a
    # sources list; the importer requires the embedded copy to equal the bytes).
    """Seal ``files`` with a solution.json embedding them as sources, and verify it."""
    solution = {
        **SOLUTION,
        "spec": {**SOLUTION["spec"], "entry_point": entry},
        "sources": [
            {"path": name, "content": data.decode()} for name, data in files.items()
        ],
    }
    digest = seal_package(
        root, {"solution.json": json_bytes(solution), **files}, sidecar=QUALIFICATION
    )
    return pk.verify_package(root, digest)


@pytest.fixture
def sm_gate_off(monkeypatch):
    """The test fixtures target sm100; load_packages tests must not depend on
    the host's GPU (the gate itself has its own dedicated test below)."""
    monkeypatch.setattr(
        pk, "check_target_hardware", lambda solution, *, device=None: None
    )


# ---------------------------------------------------------------------------
# Verification: identity, digest chain, exact file set
# ---------------------------------------------------------------------------


def test_omitted_identity_resolves_from_sealed_metadata(tmp_path):
    """An unconstrained verify reads the identity from the sealed metadata."""
    digest = seal_flashinfer_fixture(tmp_path)
    pkg = pk.verify_package(tmp_path, digest)
    assert pkg.operation == OPERATION
    assert pkg.target is None and pkg.family is None
    assert pkg.directory == tmp_path / digest
    assert pkg.sealed_files["qualification.json"].is_file()


def test_expected_constraints_cannot_relabel_a_package(tmp_path):
    """``expected`` constraints check the sealed identity and never relabel it."""
    digest = seal_flashinfer_fixture(tmp_path)
    for field, value in (("operation", "another_operation"), ("family", "activation")):
        with pytest.raises(pk.SealedPackageError, match=f"not the {field}"):
            pk.verify_package(tmp_path, digest, expected={field: value})
    with pytest.raises(pk.SealedPackageError, match="unknown identity constraint"):
        pk.verify_package(tmp_path, digest, expected={"bogus": "x"})


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("operation", "another_operation", "conflicting operation"),
        ("family", "activation", "conflicting family"),
    ],
)
def test_sealed_metadata_must_agree(tmp_path, field, value, message):
    """A manifest identity field that contradicts solution.json refuses."""
    solution = {**SOLUTION, "family": "rope"} if field == "family" else SOLUTION
    digest = seal_package(
        tmp_path,
        {"solution.json": json_bytes(solution), "kernel.py": SOURCE},
        sidecar=QUALIFICATION,
        manifest_fields={field: value},
    )
    with pytest.raises(pk.SealedPackageError, match=message):
        pk.verify_package(tmp_path, digest)


def test_qualification_alias_and_sealed_files_spellings_agree(tmp_path):
    """Both sidecar spellings seal alike; conflicting spellings refuse."""

    def seal(sidecar_key: str) -> str:
        """Seal SOURCE and SOLUTION with the sidecar under ``sidecar_key``."""
        digests = {
            "kernel.py": hashlib.sha256(SOURCE).hexdigest(),
            "solution.json": hashlib.sha256(json_bytes(SOLUTION)).hexdigest(),
        }
        sidecar_digest = hashlib.sha256(json_bytes(QUALIFICATION)).hexdigest()
        manifest = {
            "files": digests,
            "bundle_sha256": listing_sha256(digests),
            sidecar_key: {"qualification.json": sidecar_digest}
            if sidecar_key == "sealed_files"
            else sidecar_digest,
        }
        manifest_bytes = json_bytes(manifest)
        digest = hashlib.sha256(manifest_bytes).hexdigest()
        directory = tmp_path / digest
        (directory / "bundle").mkdir(parents=True)
        (directory / "manifest.json").write_bytes(manifest_bytes)
        (directory / "qualification.json").write_bytes(json_bytes(QUALIFICATION))
        (directory / "bundle" / "kernel.py").write_bytes(SOURCE)
        (directory / "bundle" / "solution.json").write_bytes(json_bytes(SOLUTION))
        return digest

    legacy = pk.verify_package(tmp_path, seal("qualification_sha256"))
    generic = pk.verify_package(tmp_path, seal("sealed_files"))
    assert legacy.sealed_files.keys() == generic.sealed_files.keys()
    assert legacy.operation == generic.operation
    assert legacy.entry_point == generic.entry_point
    # Conflicting spellings for the same sidecar refuse.
    with pytest.raises(pk.SealedPackageError, match="sealed twice with conflicting"):
        pk.verify_package(
            tmp_path,
            seal_package(
                tmp_path,
                {"solution.json": json_bytes(SOLUTION), "kernel.py": SOURCE},
                sidecar=QUALIFICATION,
                manifest_fields={"sealed_files": {"qualification.json": "0" * 64}},
            ),
        )


def test_unknown_schema_version_refuses(tmp_path):
    """A manifest with a schema_version this reader does not know refuses."""
    digest = seal_package(
        tmp_path,
        {"solution.json": json_bytes(SOLUTION), "kernel.py": SOURCE},
        sidecar=QUALIFICATION,
        manifest_fields={"schema_version": 99},
    )
    with pytest.raises(pk.SealedPackageError, match="unsupported schema_version"):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize("name", ["definition.json", "solution.json", "metadata.json"])
def test_regular_unsealed_sidecars_cannot_change_identity(tmp_path, name):
    """Unsealed sidecars beside the manifest never change what verify returns."""
    digest = seal_flashinfer_fixture(tmp_path)
    original = pk.verify_package(tmp_path, digest)
    sidecar = tmp_path / digest / name
    for content in (b"not JSON", json_bytes({"target": "vllm", "name": "other"})):
        sidecar.write_bytes(content)
        assert pk.verify_package(tmp_path, digest) == original
    sidecar.unlink()
    assert pk.verify_package(tmp_path, digest) == original


@pytest.mark.parametrize(
    "change, message",
    [
        ("manifest", "not the named digest"),
        ("source", "its manifest sealed"),
        ("missing", "missing from"),
        ("extra", "never sealed"),
        ("qualification", "its manifest sealed"),
    ],
)
def test_every_sealed_byte_and_the_exact_bundle_file_set_are_checked(
    tmp_path, change, message
):
    """Any changed, missing or extra byte under the digest chain refuses the package."""
    digest = seal_flashinfer_fixture(tmp_path)
    directory = tmp_path / digest
    if change == "manifest":
        path = directory / "manifest.json"
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "source":
        (directory / "bundle" / "kernel.py").write_bytes(b"changed")
    elif change == "missing":
        (directory / "bundle" / "kernel.py").unlink()
    elif change == "extra":
        (directory / "bundle" / "unsealed.py").write_bytes(b"extra")
    else:
        path = directory / "qualification.json"
        path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(pk.SealedPackageError, match=message):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize("scan_error", [PermissionError, FileNotFoundError, OSError])
def test_bundle_traversal_errors_cannot_hide_unsealed_files(
    tmp_path, monkeypatch, scan_error
):
    """A bundle directory that cannot be listed refuses; unseen files never pass."""
    digest = seal_flashinfer_fixture(tmp_path)
    hidden = tmp_path / digest / "bundle" / "unreadable"
    hidden.mkdir()
    (hidden / "unsealed.py").write_bytes(b"unsealed")
    scandir = os.scandir

    def cannot_scan(path):
        """Fail the scan of the hidden directory only."""
        if Path(path) == hidden:
            raise scan_error("cannot list bundle directory")
        return scandir(path)

    monkeypatch.setattr(os, "scandir", cannot_scan)
    with pytest.raises(pk.SealedPackageError, match="cannot inspect kernel bundle"):
        pk.verify_package(tmp_path, digest)


def test_manifest_listing_digest_is_checked_independently_of_the_manifest_digest(
    tmp_path,
):
    """The listing digest is checked even when the manifest digest matches."""
    digest = seal_package(
        tmp_path,
        {"solution.json": json_bytes(SOLUTION), "kernel.py": SOURCE},
        sidecar=QUALIFICATION,
        manifest_fields={"bundle_sha256": "f" * 64},
    )
    with pytest.raises(
        pk.SealedPackageError, match="does not match its own file listing"
    ):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "./kernel.py", "a//kernel.py", "bad\0name"]
)
def test_manifest_paths_must_be_canonical_bundle_relative_names(tmp_path, name):
    """A manifest path that escapes, is absolute or is not normalized refuses."""
    files = {name: hashlib.sha256(SOURCE).hexdigest()}
    digest = seal_package(
        tmp_path,
        {"kernel.py": SOURCE},
        manifest_fields={"files": files, "bundle_sha256": listing_sha256(files)},
    )
    with pytest.raises(pk.SealedPackageError, match="non-canonical path"):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize(
    "name", ["", "manifest.json", "qualification.json", "bundle", "bundle/kernel.py"]
)
def test_sealed_files_and_directories_must_not_be_symbolic_links(tmp_path, name):
    """A symbolic link anywhere in the sealed tree refuses the package."""
    digest = seal_flashinfer_fixture(tmp_path)
    path = tmp_path / digest / name if name else tmp_path / digest
    moved = tmp_path / "moved"
    path.rename(moved)
    path.symlink_to(moved, target_is_directory=moved.is_dir())
    with pytest.raises(pk.SealedPackageError, match="symbolic link"):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize("name", ["unsealed-sidecar", "bundle/unsealed-file"])
def test_root_and_bundle_special_file_safety_checks_remain(tmp_path, name):
    """A special file (a FIFO) in the package root or bundle refuses."""
    digest = seal_flashinfer_fixture(tmp_path)
    os.mkfifo(tmp_path / digest / name)
    with pytest.raises(pk.SealedPackageError, match="not a regular file"):
        pk.verify_package(tmp_path, digest)


@pytest.mark.parametrize(
    "language", ["python", "triton", "cutedsl", "cutile", "cpp", "cuda", "cute", None]
)
def test_only_python_family_solutions_load(tmp_path, language):
    """Verification accepts Python-family languages and refuses every other."""
    solution = {**SOLUTION, "spec": {**SOLUTION["spec"], "language": language}}
    digest = seal_flashinfer_fixture(tmp_path, solution=solution)
    if language in {"python", "triton", "cutedsl", "cutile"}:
        assert pk.verify_package(tmp_path, digest).solution == solution
    else:
        with pytest.raises(pk.SealedPackageError, match="only Python-family"):
            pk.verify_package(tmp_path, digest)


def test_kernel_factory_schema_is_not_silently_translated(tmp_path):
    """A SOLBench/Kernel Factory ``languages`` spec is refused, not translated."""
    spec = {key: value for key, value in SOLUTION["spec"].items() if key != "language"}
    solution = {**SOLUTION, "spec": {**spec, "languages": ["cute"]}}
    digest = seal_flashinfer_fixture(tmp_path, solution=solution)
    with pytest.raises(pk.SealedPackageError, match="SOLBench/Kernel Factory"):
        pk.verify_package(tmp_path, digest)


def test_verification_does_not_import_a_sealed_executable(tmp_path):
    """verify_package never executes a sealed source file."""
    digest = seal_package(
        tmp_path,
        {
            "solution.json": json_bytes(SOLUTION),
            "kernel.py": b"raise AssertionError('imported')\n",
        },
        sidecar=QUALIFICATION,
    )
    assert pk.verify_package(tmp_path, digest).operation == OPERATION


def test_golden_package_verifies_unchanged():
    """The checked-in golden package verifies; its sources match the embedded copy."""
    pkg = pk.verify_package(GOLDEN_ROOT, GOLDEN_DIGEST)
    manifest_bytes = (pkg.directory / "manifest.json").read_bytes()
    assert hashlib.sha256(manifest_bytes).hexdigest() == GOLDEN_DIGEST
    assert pkg.sealed_files["qualification.json"].is_file()
    for source in pkg.solution["sources"]:
        assert (pkg.directory / "bundle" / source["path"]).read_bytes() == source[
            "content"
        ].encode()


# ---------------------------------------------------------------------------
# Sealed import: private namespace, deferred re-authentication
# ---------------------------------------------------------------------------


def test_relative_imports_are_private_and_leave_exact_file_set(tmp_path):
    """Relative imports resolve inside a per-digest namespace and write no bytecode."""
    for source in (
        b"from .helpers.value import VALUE\ndef run(): return VALUE\n",
        b"def run():\n    from .helpers.value import VALUE\n    return VALUE\n",
    ):
        loaded = []
        for value in (17, 42):
            sealed = package(
                tmp_path,
                {
                    "kernel.py": source,
                    "helpers/__init__.py": b"",
                    "helpers/value.py": f"VALUE = {value}\n".encode(),
                },
            )
            function = pk.import_entry(sealed)
            loaded.append(function)
            assert function() == value
            assert function is pk.import_entry(sealed)
            assert pk.verify_package(tmp_path, sealed.digest) == sealed
            assert not list(sealed.directory.rglob("*.pyc"))
            assert not list(sealed.directory.rglob("__pycache__"))
        assert loaded[0].__module__ != loaded[1].__module__
        assert [function() for function in loaded] == [17, 42]


@pytest.mark.parametrize("mutate_manifest", [False, True])
def test_deferred_import_uses_the_original_sealed_digest(tmp_path, mutate_manifest):
    """A lazy import is checked against the digest sealed at verification time."""
    sealed = package(
        tmp_path,
        {
            "kernel.py": b"def run():\n    from .helper import VALUE\n    return VALUE\n",
            "helper.py": b"VALUE = 17\n",
        },
    )
    function = pk.import_entry(sealed)
    pk.verify_package(tmp_path, sealed.digest)
    replacement = b"raise AssertionError('unsealed helper executed')\n"
    (sealed.directory / "bundle/helper.py").write_bytes(replacement)
    if mutate_manifest:
        sealed.manifest["files"]["helper.py"] = hashlib.sha256(replacement).hexdigest()
    with pytest.raises(
        pk.SealedPackageError, match="changed after package verification"
    ):
        function()


def test_deferred_missing_module_refuses_as_sealed_error(tmp_path):
    """A lazy import of an unsealed name raises SealedPackageError, not ImportError."""
    sealed = package(
        tmp_path,
        {
            "kernel.py": b"def run():\n    from .missing import VALUE\n    return VALUE\n"
        },
    )
    function = pk.import_entry(sealed)  # the module level imports nothing
    with pytest.raises(pk.SealedPackageError, match="no sealed module"):
        function()


def test_manifest_changed_on_disk_after_import_refuses_the_next_import(tmp_path):
    """import_entry re-verifies the manifest; an edited one refuses the next import."""
    sealed = package(tmp_path, {"kernel.py": b"def run(): return 1\n"})
    assert pk.import_entry(sealed)() == 1
    manifest = sealed.directory / "manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")
    with pytest.raises(pk.SealedPackageError, match="not the named digest"):
        pk.import_entry(sealed)


@pytest.mark.parametrize(
    "replacement",
    [
        "file-link",
        "parent-link",
        "bundle-link",
        "package-link",
        "directory",
        "fifo",
        "missing",
    ],
)
def test_deferred_import_refuses_replaced_paths(tmp_path, replacement):
    """A sealed source replaced by a link, directory, FIFO or nothing refuses lazily."""
    sealed = package(
        tmp_path,
        {
            "kernel.py": b"def run():\n    from .helpers.value import VALUE\n    return VALUE\n",
            "helpers/value.py": b"VALUE = 17\n",
        },
    )
    function = pk.import_entry(sealed)
    source = sealed.directory / "bundle/helpers/value.py"
    if replacement.endswith("-link"):
        target = {
            "file-link": source,
            "parent-link": source.parent,
            "bundle-link": source.parent.parent,
            "package-link": sealed.directory,
        }[replacement]
        saved = tmp_path / "original"
        target.rename(saved)
        target.symlink_to(saved, target_is_directory=saved.is_dir())
    else:
        source.unlink()
        if replacement == "directory":
            source.mkdir()
        elif replacement == "fifo":
            os.mkfifo(source)
    with pytest.raises(
        pk.SealedPackageError, match="symbolic link|not a regular sealed source file"
    ):
        function()


@pytest.mark.parametrize("entry", ["nested:run", "nested.kernel:run"])
def test_module_colon_symbol_entry_points_are_refused_at_verification(tmp_path, entry):
    # BuildSpec takes only '<file>.py::<symbol>'; verification refuses the
    # module:symbol spelling before any sealed code runs, instead of letting
    # load_packages discover it after the import already executed.
    """A ``module:symbol`` entry point refuses at verification, before any import."""
    with pytest.raises(pk.SealedPackageError, match="must be '<file>.py::<symbol>'"):
        package(
            tmp_path,
            {
                "nested/__init__.py": b"def run(): return 1\n",
                "nested/kernel.py": b"def run(): return 1\n",
            },
            entry,
        )


@pytest.mark.parametrize(
    "entry,filename,helper",
    [
        ("__init__.py::run", "__init__.py", "helper.py"),
        ("nested/__init__.py::run", "nested/__init__.py", "nested/helper.py"),
    ],
)
def test_explicit_module_or_package_entry_point(tmp_path, entry, filename, helper):
    """An ``__init__.py`` entry file imports as a package, at the root or nested."""
    sealed = package(
        tmp_path,
        {
            filename: (
                b"assert not __name__.endswith('.__init__')\n"
                b"from .helper import VALUE\ndef run(): return VALUE\n"
            ),
            helper: b"VALUE = 19\n",
        },
        entry,
    )
    assert pk.import_entry(sealed)() == 19
    pk.verify_package(tmp_path, sealed.digest)


@pytest.mark.parametrize(
    "entry,files,message",
    [
        ("kernel.py::missing", {"kernel.py": b"def run(): pass\n"}, "not callable"),
        ("kernel.py::run", {"kernel.py": b"run = 7\n"}, "not callable"),
        ("missing.py::run", {"kernel.py": b"def run(): pass\n"}, "not sealed"),
        (
            "../kernel.py::run",
            {"kernel.py": b"def run(): pass\n"},
            "invalid Python entry_point",
        ),
        (
            "/kernel.py::run",
            {"kernel.py": b"def run(): pass\n"},
            "invalid Python entry_point",
        ),
        (
            "kernel.py::run",
            {
                "kernel.py": b"def run(): pass\n",
                "kernel/__init__.py": b"def run(): pass\n",
            },
            "shadowed",
        ),
        (
            "kernel.py::run",
            {"kernel.py": b"from .missing import run\n"},
            "no sealed module",
        ),
        (
            "kernel.py::run",
            {"kernel.py": b"raise ImportError('dependency absent')\n"},
            "dependency absent",
        ),
    ],
)
def test_no_entry_point_guessing_or_optional_dependency_fallback(
    tmp_path, entry, files, message
):
    # The entry point is parsed and validated at verification, so invalid spellings
    # refuse before any import; sealed-but-unloadable entries refuse at import.
    """Invalid entry points refuse at verification; unloadable ones refuse at import."""
    with pytest.raises(pk.SealedPackageError, match=message):
        pk.import_entry(package(tmp_path, files, entry))


def test_reauthenticate_after_executable_import(tmp_path):
    """A module that rewrites its own file on import fails the post-import re-hash."""
    sealed = package(
        tmp_path,
        {
            "kernel.py": (
                b"from pathlib import Path\n"
                b"Path(__file__).write_text('# modified')\ndef run(): pass\n"
            )
        },
    )
    with pytest.raises(pk.SealedPackageError, match="hashes to"):
        pk.import_entry(sealed)


def test_embedded_sources_must_equal_sealed_bytes(tmp_path):
    """An embedded source that differs from the sealed file refuses before import."""
    digest = seal_package(
        tmp_path,
        {
            "solution.json": json_bytes(SOLUTION),
            "kernel.py": b"raise AssertionError('must not execute')\n",
        },
        sidecar=QUALIFICATION,
    )
    sealed = pk.verify_package(tmp_path, digest)
    with pytest.raises(pk.SealedPackageError, match="embedded solution source differs"):
        pk.import_entry(sealed)


def test_same_digest_from_another_directory_is_not_a_module_cache_hit(tmp_path):
    """One digest binds to one directory per process; a copy elsewhere is refused."""
    files = {"kernel.py": b"def run(): return __file__\n"}
    first = package(tmp_path / "first", files)
    second = package(tmp_path / "second", files)
    assert first.digest == second.digest
    assert str(first.directory) in pk.import_entry(first)()
    with pytest.raises(pk.SealedPackageError, match="another directory"):
        pk.import_entry(second)


# ---------------------------------------------------------------------------
# Installation: preloaded seam, load_packages, env loading, gates
# ---------------------------------------------------------------------------


def test_register_preloaded_short_circuits_the_build_path(tmp_path):
    """A preloaded callable is served instead of a build, until it is cleared."""
    sealed = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    sentinel = lambda x: x  # noqa: E731
    register_preloaded(sealed.operation, "test_solution", sentinel)
    from flashinfer.trace.solution import Solution

    solution = Solution.from_dict(sealed.solution)
    assert _get_loaded_solution(solution) is sentinel
    clear_preloaded()
    assert _get_loaded_solution(solution) is not sentinel


def test_load_packages_strict_refuses_and_loose_collects(tmp_path, sm_gate_off):
    """Strict loading raises on a bad digest; best effort records it in ``errors``."""
    good = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    with pytest.raises(pk.SealedPackageError):
        pk.load_packages(tmp_path, [good.digest, "0" * 64], enable=False)
    result = pk.load_packages(
        tmp_path, [good.digest, "0" * 64], strict=False, enable=False
    )
    assert list(result.errors) == ["0" * 64]
    assert result.packages[0].digest == good.digest


def test_duplicate_definitions_refuse(tmp_path, sm_gate_off):
    """Two packages for one Definition refuse before the second's module runs."""
    first = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    # Refused before its module runs: a duplicate never executes sealed code.
    second = package(
        tmp_path / "second", {"kernel.py": b"raise AssertionError('imported')\n"}
    )
    both = tmp_path / "both"
    both.mkdir()
    shutil.copytree(tmp_path / first.digest, both / first.digest)
    shutil.copytree(tmp_path / "second" / second.digest, both / second.digest)
    with pytest.raises(pk.SealedPackageError, match="same definition"):
        pk.load_packages(both, [first.digest, second.digest], enable=False)


def test_loose_loading_isolates_a_package_whose_module_raises(tmp_path, sm_gate_off):
    """A module that raises on import fails strict loading and is isolated in loose."""
    good = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    broken = package(
        tmp_path / "broken", {"kernel.py": b"raise RuntimeError('boom')\n"}
    )
    both = tmp_path / "both"
    both.mkdir()
    shutil.copytree(tmp_path / good.digest, both / good.digest)
    shutil.copytree(tmp_path / "broken" / broken.digest, both / broken.digest)
    with pytest.raises(RuntimeError, match="boom"):  # strict: the error is the caller's
        pk.load_packages(both, [broken.digest, good.digest], enable=False)
    result = pk.load_packages(
        both, [broken.digest, good.digest], strict=False, enable=False
    )
    assert [p.digest for p in result.packages] == [good.digest]
    assert result.errors[broken.digest].startswith("RuntimeError: boom")


def test_preloaded_callable_is_bound_to_its_solution(tmp_path, monkeypatch):
    """A hash-bound preloaded callable serves only its Solution and beats a build."""
    import flashinfer.trace_apply.apply as apply_module
    from flashinfer.trace.solution import Solution

    def sentinel(x):
        return x

    apply_module.reset_loaded()
    sealed = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    solution = Solution.from_dict(sealed.solution)
    other = Solution.from_dict(
        {
            **sealed.solution,
            "spec": {**sealed.solution["spec"], "target_hardware": ["sm100", "sm120"]},
        }
    )
    assert (other.definition, other.name) == (solution.definition, solution.name)
    register_preloaded(
        sealed.operation, "test_solution", sentinel, solution_hash=solution.hash()
    )
    assert _get_loaded_solution(solution) is sentinel
    with pytest.raises(RuntimeError, match="built for another Solution"):
        _get_loaded_solution(other)  # same name, another Solution: never served

    # A callable preloaded while a build is in flight wins over the build.
    clear_preloaded()

    def build(sol):
        register_preloaded(sol.definition, sol.name, sentinel)
        return lambda x: x - 1

    monkeypatch.setattr(apply_module, "load_solution", build)
    assert _get_loaded_solution(solution) is sentinel


def test_release_drops_every_sealed_identity_a_definition_accumulated(
    tmp_path, monkeypatch, sm_gate_off
):
    """release_packages drops routes before every sealed identity's preloaded entry."""
    import flashinfer.trace_apply.apply as apply_module

    def plain(root, name):
        source = b"def run(x): return x + 1\n"
        solution = {
            **SOLUTION,
            "name": name,
            "sources": [{"path": "kernel.py", "content": source.decode()}],
        }
        return seal_package(
            root,
            {"solution.json": json_bytes(solution), "kernel.py": source},
            sidecar=QUALIFICATION,
        )

    # Two sealed Solutions for one Definition across a disable: both identities
    # are kept until the release, which drops both preloaded entries.
    pk.load_packages(tmp_path / "first", [plain(tmp_path / "first", "first")])
    ta.disable_apply()
    pk.load_packages(tmp_path / "second", [plain(tmp_path / "second", "second")])
    assert preloaded_entry(OPERATION, "first") and preloaded_entry(OPERATION, "second")
    # Routes go before entries: at every unregister the Definition is already
    # out of the mapping, so no call can find it enabled without its entry.
    still_enabled = []
    unregister = apply_module.unregister_preloaded

    def spy(definition, name):
        still_enabled.append(definition in apply_module._solutions)
        unregister(definition, name)

    monkeypatch.setattr(apply_module, "unregister_preloaded", spy)
    pk.release_packages()
    assert still_enabled == [False, False]
    assert preloaded_entry(OPERATION, "first") is None
    assert preloaded_entry(OPERATION, "second") is None
    assert not pk._preloaded_solutions and not apply_module._solutions


def test_a_registration_during_another_callers_build_is_seen_under_the_build_lock(
    tmp_path, monkeypatch
):
    """A registration made during another caller's build wins under the build lock."""
    import threading

    import flashinfer.trace_apply.apply as apply_module
    from flashinfer.trace.solution import Solution

    def sentinel(x):
        return x

    def built(x):
        return x - 1

    apply_module.reset_loaded()
    clear_preloaded()
    sealed = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    solution = Solution.from_dict(sealed.solution)
    key = (solution.definition, solution.name)

    class Gate:  # the per-solution build lock, held by a caller whose build is in flight
        def __init__(self):
            self.lock, self.waited = threading.Lock(), threading.Event()

        def __enter__(self):
            self.waited.set()
            self.lock.acquire()

        def __exit__(self, *exc):
            self.lock.release()

    gate = Gate()
    monkeypatch.setitem(apply_module._loading_locks, key, gate)
    rebuilt = []
    monkeypatch.setattr(
        apply_module, "load_solution", lambda sol: rebuilt.append(sol) or built
    )
    gate.lock.acquire()  # caller A is building
    served = []
    caller_b = threading.Thread(
        target=lambda: served.append(_get_loaded_solution(solution))
    )
    caller_b.start()
    assert gate.waited.wait(5)  # B missed the registry and waits for A's lock
    with apply_module._loaded_lock:
        apply_module._loaded[key] = built  # A's build completes and is cached
    register_preloaded(
        *key, sentinel, solution_hash=solution.hash()
    )  # then a loader registers
    gate.lock.release()  # A lets go; B takes the lock
    caller_b.join(5)
    assert (
        served == [sentinel] and rebuilt == []
    )  # B serves the registration, builds nothing


def test_target_hardware_gate(tmp_path, monkeypatch):
    """The SM gate compares numbers: sm120 refuses an sm100 package; no GPU skips it."""
    import flashinfer.trace_apply.apply as apply_mod

    sealed = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    # current_sm answers the prefixed form ("sm120"); the fixture targets sm100.
    monkeypatch.setattr(apply_mod, "current_sm", lambda device=None: "sm120")
    with pytest.raises(pk.SealedPackageError, match=r"not this GPU \(sm120\)"):
        pk.load_packages(tmp_path, [sealed.digest], enable=False)
    pk.check_target_hardware({"spec": {"target_hardware": []}})  # every GPU
    pk.check_target_hardware({"spec": {"target_hardware": ["sm100", "sm120a"]}})
    monkeypatch.setattr(apply_mod, "current_sm", lambda device=None: "sm100")
    pk.load_packages(tmp_path, [sealed.digest], enable=False)  # fixture targets sm100
    monkeypatch.setattr(apply_mod, "current_sm", lambda device=None: None)
    pk.load_packages(tmp_path, [sealed.digest], enable=False)  # no GPU: skip


def test_env_loading_is_warning_only(tmp_path, monkeypatch, sm_gate_off):
    """The env hook warns about a refused digest and still loads the good ones."""
    good = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    monkeypatch.setenv(pk.PACKAGES_ENV, f"{good.digest},{'0' * 64}")
    monkeypatch.setenv(pk.ROOT_ENV, str(tmp_path))
    with pytest.warns(UserWarning, match="refused"):
        pk._load_packages_from_env()
    # The good package verified, imported and registered despite the refusal.
    from flashinfer.trace.solution import Solution

    sealed = pk.verify_package(tmp_path, good.digest)
    fn = _get_loaded_solution(Solution.from_dict(sealed.solution))
    assert fn.__module__.startswith("_flashinfer_sealed_")
    assert fn(1) == 2


def test_env_kill_switch(tmp_path, monkeypatch):
    """FLASHINFER_KERNEL_PLUGINS_DISABLE=1 warns and loads nothing."""
    good = package(tmp_path, {"kernel.py": b"def run(x): return x + 1\n"})
    monkeypatch.setenv(pk.PACKAGES_ENV, good.digest)
    monkeypatch.setenv(pk.ROOT_ENV, str(tmp_path))
    monkeypatch.setenv(pk.DISABLE_ENV, "1")
    with pytest.warns(UserWarning, match="not loaded"):
        pk._load_packages_from_env()
    assert not ta.is_enabled()


def test_env_gate_asymmetry_regression(tmp_path, monkeypatch):
    # Only exactly "1" reaches the env hook; "true" warns and never enables.
    """FLASHINFER_TRACE_APPLY=true warns and never enables Trace Apply."""
    monkeypatch.setenv("FLASHINFER_TRACE_APPLY", "true")
    with pytest.warns(UserWarning, match="must be '0' or '1'"):
        _enable_apply_from_env()
    assert not ta.is_enabled()


def test_gate_asymmetry_in_package_import():
    """FLASHINFER_TRACE_APPLY=true must not enable (nor even attempt) trace apply.

    Runs a fresh interpreter so the import-time hook is exercised exactly as
    the engine would see it.
    """
    code = (
        "import os, warnings\n"
        "os.environ['FLASHINFER_TRACE_APPLY'] = 'true'\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        "    import flashinfer.trace_apply as ta\n"
        "assert not any('TRACE_APPLY' in str(w.message) for w in caught), caught\n"
        "assert not ta.is_enabled()\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
