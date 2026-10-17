#!/usr/bin/env python3
"""Check the build manifests of the Cake-exported packages (flashinfer-ai/flashinfer#6553).

Eight packages ship generated CUDA sources under ``csrc/`` together with a build
manifest ``build.json`` (schema ``cake-build/1``) -- the only record of what to
compile -- and a byte-identical ``cake_build.py`` loader that loads the prebuilt
library under ``aot/`` when present and otherwise JIT-builds the manifest entry
through FlashInfer.  The package registries (``cake_jit.py`` / ``jit.py`` /
``sources.json``) reference the manifest entries by an architecture-free
``library`` stem: entry ``<library>[<specialization tag>]_<arch>``.

The manifest's ``package`` field is the dotted import path of the package
directory (``flashinfer.experimental.cake_mok_bf16``,
``flashinfer.cake_rmsnorm_train_backend``): the one spelling every exporter
writes and this checker compares (``Package.name``); a directory path is not
accepted.  ``name`` is the unique key of ``libraries``; ``closure_sha256`` is
not unique (two entries of one program and architecture, for example two tile
specialisations of a stage, share the bundle's closure) and is not checked for
uniqueness.  ``common.requires.cuda_toolkit.min`` is the lowest
``libraries[].min_cuda`` -- the floor below which nothing in the package
builds -- not the toolkit that builds every entry: each entry's own
``min_cuda`` gates it (a warm-up or the vendored gate skips an entry above the
toolkit, "skip min_cuda > toolkit").

The checker is deliberately dependency-free: it parses the registry literals
with ``ast`` and never imports ``flashinfer`` (no CUDA probe, no compiler).

In-tree checks (``--in-tree``, the default):

* manifest schema -- required fields, ``package`` == the dotted import path,
  ``path == aot/<name>.so``, ``arch`` in ``arch_flags``, ``name`` ends in
  ``_<arch>``, ``sources`` package-relative under ``csrc/`` and present,
  ``sha256`` keyed by the sources and equal to the file digests, 64-hex
  ``closure_sha256``, ``min_cuda`` as ``X.Y``, ``entry == "run"``;
* entry names unique and equal to the set the registry references, so every
  registry stem has its per-architecture entries and no entry is orphaned;
* ``common.requires.cuda_toolkit.min`` == the lowest ``libraries[].min_cuda``;
* package-data coverage in ``pyproject.toml`` for ``build.json``,
  ``csrc/cake_ffi_checks.h`` and every manifest source (wheels ship them);
* the ``cake_build.py`` files of all checked packages are byte-identical;
* ``common.requires.tvm_ffi.python`` equals line 1 of ``requirements.txt``.

Not implemented yet (TODO, exit code 2 when requested):

* ``--vendored-gate``: per package, copy the package directory to a temporary
  location, compile every manifest entry with plain ``nvcc`` from the manifest
  flags only (skip ``min_cuda`` above the toolkit) into ``<copy>/aot/``, write
  ``aot/build_info.json`` with ``arches`` = every architecture compiled into
  the copy (the sidecar rule ``cake_build.record_cuda_version`` applies: an
  arch absent from the sidecar takes the JIT path), then in a fresh
  ``python -I`` subprocess with
  ``sys.modules["flashinfer"] = None`` import the package, load every library
  through ``cake_build`` and run one smoke launch, asserting the module came
  from ``aot/`` and that every entry reports one ``build_cuda_version``.
* ``--built-wheel``: build the wheel and import every package from it,
  asserting ``build.json``, ``csrc/cake_ffi_checks.h`` and the sources are
  present (model: scripts/task_test_jit_cache_package_build_import.sh).
* the JIT-path flag check (``get_compile_commands()`` flags are a superset of
  the manifest flags for one entry per package; needs ``FLASHINFER_CUDA_ARCH_LIST``
  set, ``FLASHINFER_JIT_DEBUG`` / ``FLASHINFER_JIT_VERBOSE`` unset, an
  ``isinstance(spec, JitSpecNvcc)`` check and FlashInfer's flag spelling
  ``--compiler-options=-fPIC``) and the "prebuilt path never calls
  ``gen_jit_spec``" check belong to the same compute step.

Exit codes: 0 all checks passed, 1 findings, 2 the check could not run or is
not implemented.  Run on allocated compute (the vendored gate needs nvcc).
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

SCHEMA = "cake-build/1"
HEADER = "csrc/cake_ffi_checks.h"
ARCH_RE = re.compile(r"^sm_\d+[a-z]?$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_KEYS = {"schema", "package", "generator", "common", "arch_flags", "libraries"}
GENERATOR_KEYS = {"cake_revision", "rendered_at", "cuda"}
COMMON_KEYS = {
    "nvcc_flags",
    "nvcc_flags_if",
    "include_dirs",
    "link_flags",
    "link",
    "requires",
}
ENTRY_KEYS = {
    "name",
    "path",
    "arch",
    "gencode",
    "sources",
    "nvcc_flags",
    "nvcc_flags_if",
    "min_cuda",
    "closure_sha256",
    "sha256",
    "entry",
}


@dataclass(frozen=True)
class Finding:
    package: str
    message: str

    def __str__(self) -> str:
        return f"{self.package}: {self.message}"


# ---------------------------------------------------------------------------
# Registry readers: the manifest entry names a package's registry references.
# ---------------------------------------------------------------------------


def module_literals(path: Path) -> dict[str, Any]:
    """Top-level literal assignments of a Python module (``NAME = <literal>`` / ``NAME: T = <literal>``)."""
    tree = ast.parse(path.read_text(), filename=str(path))
    values: dict[str, Any] = {}
    for node in tree.body:
        target = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
        elif isinstance(node, ast.AnnAssign):
            target = node.target
        if not isinstance(target, ast.Name) or node.value is None:
            continue
        try:
            values[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return values


def _tag(specializations: dict[str, Any]) -> str:
    return "".join(
        f"_{key.lower()}{value}" for key, value in sorted(specializations.items())
    )


def generic_library_refs(value: Any, arches: tuple[str, ...] = ()) -> set[str]:
    """Every ``<library>_<arch>`` a literal tree references: a mapping carrying ``library`` names one build
    per architecture of the nearest enclosing ``arches`` list (or ``arch`` string)."""
    refs: set[str] = set()
    if isinstance(value, dict):
        if isinstance(value.get("arches"), list):
            arches = tuple(str(a) for a in value["arches"])
        elif isinstance(value.get("arch"), str):
            arches = (value["arch"],)
        if isinstance(value.get("library"), str):
            if not arches:
                raise ValueError(
                    f"library {value['library']!r} has no enclosing arches"
                )
            refs.update(f"{value['library']}_{arch}" for arch in arches)
        for item in value.values():
            refs |= generic_library_refs(item, arches)
    elif isinstance(value, list):
        for item in value:
            refs |= generic_library_refs(item, arches)
    return refs


def refs_generic(package_dir: Path, loader: str) -> set[str]:
    return generic_library_refs(module_literals(package_dir / loader))


def refs_lm_head_loss(package_dir: Path, loader: str) -> set[str]:
    """Compact rows ``[unit, arg_plan, launch, grid_x, grid_y, grid_z]`` per stage of a per-architecture record;
    the stage's library stem is ``cake_lm_head_loss_<unit>``."""
    registry = module_literals(package_dir / loader)["_REGISTRY"]
    return {
        f"cake_lm_head_loss_{row[0]}_{compact['arch']}"
        for compact in registry.values()
        for row in compact["stages"].values()
    }


def refs_moe_grouped_gemm(package_dir: Path, loader: str) -> set[str]:
    """Route stages bind a program and its compile-line specializations; the build is
    ``<library><tag>_<arch>`` for every architecture the route and the program share."""
    literals = module_literals(package_dir / loader)
    programs, routes = literals["PROGRAMS"], literals["ROUTES"]
    refs: set[str] = set()
    used: set[str] = set()
    for route in routes.values():
        for stage in route["stages"]:
            program = programs[stage["program"]]
            used.add(stage["program"])
            tag = _tag(stage.get("specializations") or {})
            for arch in route["arches"]:
                if arch in program["arches"]:
                    refs.add(f"{program['library']}{tag}_{arch}")
    unused = set(programs) - used
    if unused:
        raise ValueError(f"PROGRAMS not referenced by any route: {sorted(unused)}")
    return refs


def refs_mok_bf16(package_dir: Path, loader: str) -> set[str]:
    """``sources.json``: role -> {library, arches, arguments}."""
    registry = json.loads((package_dir / loader).read_text())
    return {
        f"{record['library']}_{arch}"
        for record in registry.values()
        for arch in record["arches"]
    }


@dataclass(frozen=True)
class Package:
    name: str  # dotted import path of ``path``; build.json ``package`` must equal it
    path: str  # repository-relative directory
    loader: str  # registry file inside the package
    refs: Callable[[Path, str], set[str]]


# dsa_train, dsa_indexer and the rmsnorm backend are read with the generic walker (a mapping
# carrying ``library`` under an ``arches`` / ``arch`` scope).  Their exporters' registry shapes
# fit it: a record carries ``arches`` and either ``library`` itself (dsa_indexer PROGRAMS, rmsnorm
# MODULES) or per-stage sub-records carrying ``library`` (dsa_train).
PACKAGES: tuple[Package, ...] = (
    Package(
        "flashinfer.cake_rmsnorm_train_backend",
        "flashinfer/cake_rmsnorm_train_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_dsa_train",
        "flashinfer/experimental/cake_dsa_train",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_dsa_indexer",
        "flashinfer/experimental/cake_dsa_indexer",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_moe_grouped_gemm",
        "flashinfer/experimental/cake_moe_grouped_gemm",
        "cake_jit.py",
        refs_moe_grouped_gemm,
    ),
    Package(
        "flashinfer.experimental.cake_lm_head_loss",
        "flashinfer/experimental/cake_lm_head_loss",
        "cake_jit.py",
        refs_lm_head_loss,
    ),
    Package(
        "flashinfer.experimental.cake_mok_bf16",
        "flashinfer/experimental/cake_mok_bf16",
        "sources.json",
        refs_mok_bf16,
    ),
    Package(
        "flashinfer.experimental.dense_projection_gemm",
        "flashinfer/experimental/dense_projection_gemm",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_stepfun_moe",
        "flashinfer/experimental/cake_stepfun_moe",
        "cake_jit.py",
        refs_moe_grouped_gemm,
    ),
)


# ---------------------------------------------------------------------------
# Manifest checks
# ---------------------------------------------------------------------------


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def check_manifest(
    package: Package, package_dir: Path, manifest: dict[str, Any]
) -> list[Finding]:
    found: list[Finding] = []

    def note(message: str) -> None:
        found.append(Finding(package.name, message))

    missing = MANIFEST_KEYS - set(manifest)
    if missing:
        note(f"build.json lacks {sorted(missing)}")
        return found
    if manifest["schema"] != SCHEMA:
        note(f"schema {manifest['schema']!r} != {SCHEMA!r}")
    if manifest["package"] != package.name:
        note(
            f"package {manifest['package']!r} != {package.name!r} "
            f"(the dotted import path of {package.path})"
        )
    if not isinstance(manifest["generator"], dict) or GENERATOR_KEYS - set(
        manifest["generator"]
    ):
        note(
            f"generator lacks {sorted(GENERATOR_KEYS - set(manifest.get('generator') or {}))}"
        )
    common = manifest["common"]
    if not isinstance(common, dict) or COMMON_KEYS - set(common):
        note(
            f"common lacks {sorted(COMMON_KEYS - set(common if isinstance(common, dict) else {}))}"
        )
    else:
        for key in ("nvcc_flags", "include_dirs", "link_flags"):
            if not _is_str_list(common[key]):
                note(f"common.{key} is not a list of strings")
        if "csrc" not in common["include_dirs"]:
            note("common.include_dirs lacks 'csrc'")
        found.extend(_check_flags_if(package, "common", common["nvcc_flags_if"]))
    arch_flags = manifest["arch_flags"]
    if not isinstance(arch_flags, dict) or not arch_flags:
        note("arch_flags is not a non-empty mapping")
        arch_flags = {}
    for arch, flags in arch_flags.items():
        if not ARCH_RE.match(arch):
            note(f"arch_flags key {arch!r} is not an sm_XXX[a] architecture")
        if not _is_str_list(flags) or not any(f.startswith("-gencode") for f in flags):
            note(f"arch_flags[{arch!r}] carries no -gencode flag")
    libraries = manifest["libraries"]
    if not isinstance(libraries, list) or not libraries:
        note("libraries is not a non-empty list")
        return found
    # ``name`` is the unique key.  ``closure_sha256`` is deliberately not checked
    # for uniqueness: two entries of one program and architecture (for example
    # two tile specialisations of a stage) share the bundle's closure.
    names = [entry.get("name") for entry in libraries]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        note(f"duplicate entry names {duplicates}")
    for entry in libraries:
        found.extend(_check_entry(package, package_dir, entry, arch_flags))
    return found


def _check_flags_if(package: Package, where: str, items: Any) -> list[Finding]:
    found: list[Finding] = []
    if not isinstance(items, list):
        return [Finding(package.name, f"{where}.nvcc_flags_if is not a list")]
    for item in items:
        if (
            not isinstance(item, dict)
            or set(item) != {"cuda", "flags"}
            or not re.match(r"^(==|>=|<=|>|<)\d+\.\d+$", str(item.get("cuda")))
            or not _is_str_list(item.get("flags"))
        ):
            found.append(
                Finding(
                    package.name,
                    f"{where}.nvcc_flags_if item {item!r} is not {{cuda: '==X.Y'|'>=X.Y', flags: [...]}}",
                )
            )
    return found


def _check_entry(
    package: Package, package_dir: Path, entry: Any, arch_flags: dict[str, Any]
) -> list[Finding]:
    if not isinstance(entry, dict) or "name" not in entry:
        return [Finding(package.name, f"library entry without a name: {entry!r}")]
    name = entry["name"]
    found: list[Finding] = []

    def note(message: str) -> None:
        found.append(Finding(package.name, f"{name}: {message}"))

    missing = ENTRY_KEYS - set(entry)
    if missing:
        note(f"lacks {sorted(missing)}")
        return found
    arch = entry["arch"]
    if arch not in arch_flags:
        note(f"arch {arch!r} not in arch_flags")
    if not name.endswith(f"_{arch}"):
        note(f"name does not end in _{arch}")
    if entry["path"] != f"aot/{name}.so":
        note(f"path {entry['path']!r} != 'aot/{name}.so'")
    if entry["entry"] != "run":
        note(f"entry {entry['entry']!r} != 'run'")
    if not HEX64_RE.match(str(entry["closure_sha256"])):
        note("closure_sha256 is not 64 hex digits")
    if not re.match(r"^\d+\.\d+$", str(entry["min_cuda"])):
        note(f"min_cuda {entry['min_cuda']!r} is not 'X.Y'")
    if not _is_str_list(entry["nvcc_flags"]):
        note("nvcc_flags is not a list of strings")
    found.extend(_check_flags_if(package, name, entry["nvcc_flags_if"]))
    sources = entry["sources"]
    if not _is_str_list(sources) or not sources:
        note("sources is not a non-empty list of strings")
        return found
    for source in sources:
        if not source.startswith("csrc/") or ".." in source.split("/"):
            note(f"source {source!r} is not package-relative under csrc/")
        elif not (package_dir / source).is_file():
            note(f"source {source!r} is missing")
    digests = entry["sha256"]
    if not isinstance(digests, dict) or set(digests) != set(sources):
        note("sha256 is not keyed by exactly the sources")
    else:
        for source, digest in digests.items():
            path = package_dir / source
            if (
                path.is_file()
                and hashlib.sha256(path.read_bytes()).hexdigest() != digest
            ):
                note(f"sha256 of {source!r} does not match the file")
    return found


def _version_key(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(text).split("."))


def check_cuda_toolkit_floor(
    package: Package, manifest: dict[str, Any]
) -> list[Finding]:
    """``common.requires.cuda_toolkit.min`` is the lowest ``libraries[].min_cuda``: the floor below which
    no entry of the package builds, not the toolkit that builds every entry (each entry's ``min_cuda``
    gates it; a warm-up skips an entry above the local toolkit)."""
    floors = [
        entry["min_cuda"]
        for entry in manifest.get("libraries", [])
        if isinstance(entry, dict)
        and re.match(r"^\d+\.\d+$", str(entry.get("min_cuda")))
    ]
    if not floors:
        return []  # entries without a well-formed min_cuda are reported by check_manifest
    lowest = min(floors, key=_version_key)
    common = manifest.get("common")
    requires = common.get("requires") if isinstance(common, dict) else None
    declared = requires.get("cuda_toolkit") if isinstance(requires, dict) else None
    if declared != {"min": lowest}:
        return [
            Finding(
                package.name,
                f"common.requires.cuda_toolkit {declared!r} != {{'min': {lowest!r}}} "
                "(the lowest libraries[].min_cuda)",
            )
        ]
    return []


def check_references(
    package: Package, package_dir: Path, manifest: dict[str, Any]
) -> list[Finding]:
    """Entry names == the builds the registry references (every stem has its entries, none is orphaned)."""
    try:
        expected = package.refs(package_dir, package.loader)
    except (KeyError, ValueError, TypeError, SyntaxError, OSError) as error:
        return [
            Finding(package.name, f"cannot read registry {package.loader}: {error}")
        ]
    names = {
        entry.get("name")
        for entry in manifest.get("libraries", [])
        if isinstance(entry, dict)
    }
    found: list[Finding] = []
    if not expected:
        found.append(
            Finding(
                package.name,
                f"registry {package.loader} references no library (no 'library' stems)",
            )
        )
    for name in sorted(expected - names):
        found.append(
            Finding(
                package.name,
                f"registry references {name!r}, which build.json does not list",
            )
        )
    for name in sorted(names - expected):
        found.append(
            Finding(
                package.name,
                f"build.json entry {name!r} is referenced by no registry record",
            )
        )
    return found


# ---------------------------------------------------------------------------
# Packaging checks
# ---------------------------------------------------------------------------


def package_data(pyproject: Path) -> dict[str, list[str]]:
    """``[tool.setuptools.package-data]`` of pyproject.toml (tomllib when available, else the section's lines)."""
    text = pyproject.read_text()
    try:
        import tomllib  # Python >= 3.11
    except ModuleNotFoundError:
        tomllib = None
    if tomllib is not None:
        return dict(tomllib.loads(text)["tool"]["setuptools"]["package-data"])
    section = re.search(
        r"^\[tool\.setuptools\.package-data\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M
    )
    if section is None:
        return {}
    table: dict[str, list[str]] = {}
    for match in re.finditer(r'^"([^"]+)"\s*=\s*\[([^\]]*)\]', section.group(1), re.M):
        table[match.group(1)] = re.findall(r'"([^"]*)"', match.group(2))
    return table


def glob_matches(pattern: str, relative: str) -> bool:
    """setuptools package-data glob: ``**`` spans directories, ``*`` / ``?`` stay within one segment."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.fullmatch("".join(out), relative) is not None


def check_package_data(
    package: Package, manifest: dict[str, Any], table: dict[str, list[str]]
) -> list[Finding]:
    globs = table.get(package.name)
    if globs is None:
        return [Finding(package.name, "pyproject.toml has no package-data entry")]
    required = {"build.json", HEADER}
    for entry in manifest.get("libraries", []):
        if isinstance(entry, dict) and _is_str_list(entry.get("sources")):
            required.update(entry["sources"])
    return [
        Finding(package.name, f"package-data {globs} does not ship {relative!r}")
        for relative in sorted(required)
        if not any(glob_matches(glob, relative) for glob in globs)
    ]


def check_loader_identity(found_dirs: dict[str, Path]) -> list[Finding]:
    """Every checked package carries the same cake_build.py bytes (one template, copied verbatim)."""
    contents = {name: (path / "cake_build.py") for name, path in found_dirs.items()}
    missing = [name for name, path in contents.items() if not path.is_file()]
    found = [Finding(name, "cake_build.py is missing") for name in missing]
    present = {
        name: path.read_bytes() for name, path in contents.items() if path.is_file()
    }
    if len(set(present.values())) > 1:
        reference = next(iter(present))
        found.extend(
            Finding(name, f"cake_build.py differs from {reference}'s")
            for name, data in present.items()
            if data != present[reference]
        )
    return found


def check_tvm_ffi_pin(
    package: Package, manifest: dict[str, Any], requirements: Path
) -> list[Finding]:
    pin = requirements.read_text().splitlines()[0].strip()
    declared = (
        ((manifest.get("common") or {}).get("requires") or {})
        .get("tvm_ffi", {})
        .get("python")
    )
    if declared != pin:
        return [
            Finding(
                package.name,
                f"common.requires.tvm_ffi.python {declared!r} != requirements.txt line 1 {pin!r}",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def select(names: Iterable[str] | None) -> list[Package]:
    if not names:
        return list(PACKAGES)
    wanted = set(names)
    chosen = [
        p
        for p in PACKAGES
        if p.name in wanted or p.path in wanted or p.path.rsplit("/", 1)[-1] in wanted
    ]
    unknown = (
        wanted
        - {p.name for p in chosen}
        - {p.path for p in chosen}
        - {p.path.rsplit("/", 1)[-1] for p in chosen}
    )
    if unknown:
        raise SystemExit(
            f"unknown package(s) {sorted(unknown)}; known: {[p.path.rsplit('/', 1)[-1] for p in PACKAGES]}"
        )
    return chosen


def run_in_tree(root: Path, packages: list[Package]) -> list[Finding]:
    findings: list[Finding] = []
    table = package_data(root / "pyproject.toml")
    checked: dict[str, Path] = {}
    for package in packages:
        package_dir = root / package.path
        if not package_dir.is_dir():
            findings.append(
                Finding(package.name, f"package directory {package.path} is missing")
            )
            continue
        manifest_path = package_dir / "build.json"
        if not manifest_path.is_file():
            findings.append(Finding(package.name, "build.json is missing"))
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as error:
            findings.append(
                Finding(package.name, f"build.json is not valid JSON: {error}")
            )
            continue
        checked[package.name] = package_dir
        findings.extend(check_manifest(package, package_dir, manifest))
        findings.extend(check_cuda_toolkit_floor(package, manifest))
        findings.extend(check_references(package, package_dir, manifest))
        findings.extend(check_package_data(package, manifest, table))
        findings.extend(check_tvm_ffi_pin(package, manifest, root / "requirements.txt"))
    findings.extend(check_loader_identity(checked))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="FlashInfer checkout (default: this script's repository)",
    )
    parser.add_argument(
        "--package",
        action="append",
        help="Check only this package (dotted name, path or directory name); repeatable",
    )
    parser.add_argument(
        "--in-tree",
        action="store_true",
        help="Run the in-tree checks (default when no other mode is given)",
    )
    parser.add_argument(
        "--vendored-gate",
        action="store_true",
        help="TODO: plain-nvcc vendored build + flashinfer-free import gate (not implemented; exit 2)",
    )
    parser.add_argument(
        "--built-wheel",
        action="store_true",
        help="TODO: built-wheel import check (not implemented; exit 2)",
    )
    parser.add_argument("--report-json", type=Path, help="Write findings as JSON")
    args = parser.parse_args()

    if args.vendored_gate or args.built_wheel:
        mode = "--vendored-gate" if args.vendored_gate else "--built-wheel"
        print(
            f"{mode}: not implemented yet (see the module docstring TODO); run on allocated compute once it lands",
            file=sys.stderr,
        )
        return 2

    root = args.repo_root.resolve()
    if (
        not (root / "pyproject.toml").is_file()
        or not (root / "requirements.txt").is_file()
    ):
        print(
            f"{root} is not a FlashInfer checkout (pyproject.toml / requirements.txt missing)",
            file=sys.stderr,
        )
        return 2
    packages = select(args.package)
    findings = run_in_tree(root, packages)
    for finding in findings:
        print(finding)
    if args.report_json:
        args.report_json.write_text(
            json.dumps(
                [{"package": f.package, "message": f.message} for f in findings],
                indent=2,
            )
            + "\n"
        )
    print(f"checked {len(packages)} package(s): {len(findings)} finding(s)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
