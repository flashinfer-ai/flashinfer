#!/usr/bin/env python3
"""Check the build manifests of the Cake-exported packages (flashinfer-ai/flashinfer#6553).

The Cake packages below ship generated CUDA sources under ``csrc/`` together with a build
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

The checker never imports ``flashinfer``.  The in-tree checks are
dependency-free (they parse the registry literals with ``ast``: no CUDA probe,
no compiler); the vendored gate drives ``nvcc``, ``nvidia-smi`` and child
interpreters through ``subprocess``.

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

Vendored gate (``--vendored-gate``; combinable with ``--package``): the form
of a consumer that vendors a package and builds it from ``build.json`` with its
own rules, without FlashInfer.  Needs ``nvcc``, ``tvm_ffi`` (its include
directories) and, for the default architecture list, a visible GPU -- run it
on allocated compute.  Per selected package, under the work directory
(``--work-dir``, default a fresh ``$TMPDIR/cake-vendored-gate-*``; kept with
``--keep`` and whenever there are findings, its path printed; otherwise a
directory the gate created is removed and a pre-existing one loses only the
entries the gate wrote into it):

1. copy the package directory to ``<work>/vendored_<directory name>/``
   without ``aot/`` and ``__pycache__/`` (``shutil.copytree``);
2. compile every manifest entry of a selected architecture (``--arch``,
   repeatable; default: the ``compute_cap`` of the visible GPUs from
   ``nvidia-smi`` as ``sm_<major><minor>a``, intersected with ``arch_flags``;
   a package with none of them is reported as skipped, not a finding, and a
   ``--suite`` requested for it as not run, no finding either) with
   plain ``nvcc`` (``--nvcc``; default ``$CUDA_HOME/bin/nvcc``, else ``PATH``)
   and the manifest flags only -- ``common.nvcc_flags`` + the matching
   ``common.nvcc_flags_if`` + ``arch_flags[arch]`` + the entry's
   ``nvcc_flags`` + its matching ``nvcc_flags_if`` (``==X.Y`` / ``>=X.Y``
   against the toolkit of ``nvcc --version``, the rule of
   ``cake_build._cuda_constraint_matches`` re-implemented here), ``-I`` for
   ``common.include_dirs`` and for the tvm-ffi and dlpack include directories
   (the two ``flashinfer/jit/cpp_ext.py`` adds), the entry's copy-relative
   sources, ``common.link_flags`` and ``-o <copy>/<entry.path>`` -- run from
   the copy, ``--jobs`` at a time, skipping an entry whose ``min_cuda``
   exceeds the toolkit ("skip min_cuda > toolkit", recorded); a non-zero
   exit or a missing output file is a finding (the first 40 stderr lines are
   kept);
3. write the sidecar ``<copy>/aot/build_info.json`` (schema
   ``cake-build-info/1``: ``cuda`` "X.Y", ``cudart_version`` 1000 * X +
   10 * Y, ``arches`` = every architecture with a compiled entry, plus the
   extra keys ``builder`` and ``nvcc``), as ``cake_build.load_library`` /
   ``record_cuda_version`` read it (an architecture absent from the sidecar
   takes the JIT path);
4. load check -- a fresh ``<python> -I`` child (``--python``, default this
   interpreter) running the generated ``<work>/vendored_gate_child.py``: it
   puts the work directory and the interpreter's dependency paths (every
   ``site-packages`` / ``dist-packages`` entry and the parents of ``torch``
   and ``tvm_ffi``, reported by a probe run of the same script) on
   ``sys.path``, installs a meta-path finder that raises ``ImportError`` for
   ``flashinfer`` and ``flashinfer.*`` plus ``sys.modules["flashinfer"] =
   None``, imports ``torch`` (the libraries link ``-lcudart``; the runtime
   resolves through it, as in every consumer), the copy and its
   ``cake_build``, then checks for every compiled entry
   ``cake_build.load_library(name)`` with ``library_path(name)`` == the
   copy's ``aot/<name>.so`` (prebuilt, not JIT) and ``build_cuda_version()``
   == ``cudart_version``, for every compiled architecture
   ``record_cuda_version(arch)`` == (X, Y), and last that no ``flashinfer``
   module is loaded; one JSON line per check, every failed check a finding
   (the libraries also link ``-lcuda``, so the load needs the CUDA driver
   library);
5. suite (``--suite PACKAGE=TESTFILE``, repeatable across packages, one test
   file per package -- a different second file for one package is a usage
   error, further files go through ``--pytest-args``, whose extra pytest
   arguments are split like a shell) -- a second ``<python> -I`` child with the
   same blocker, the repository root on ``sys.path`` and
   ``tests.test_helpers.cake_package.PACKAGES[<key>]`` pointed at the copy
   (the key whose import path is the package's, read from that table), runs
   ``pytest TESTFILE --noconftest -p no:cacheprovider -q -rfEs --junitxml
   <work>/<key>-suite.xml`` from the repository root (``tests/conftest.py``
   imports flashinfer, hence ``--noconftest``; the backend suites import
   nothing else of it); a failed test or a non-zero pytest exit is a finding
   with the first ten failing ids.  A test that needs a program the gate did
   not compile fails with "... flashinfer's JIT is not available": a real
   gap of the manifest or of ``--arch``, reported as such.  The suite child
   also exports its ``sys.path`` entries as ``PYTHONPATH``, and the gate
   writes ``<work>/sitecustomize.py`` with the same blocker and table patch,
   so an interpreter a test spawns (the fresh-process test of the chunked
   LM-head suite) blocks flashinfer and imports the copy as well.

The report: one line per package (``compiled N of M (skipped K by min_cuda),
loaded N of N, build_cuda_version ok, suite: P passed F failed S skipped``),
the findings, and with ``--report-json`` ``{"mode": "vendored-gate",
"toolkit", "nvcc", "arches", "packages": [...], "findings": [...]}``.  The
gate itself imports neither ``flashinfer`` nor ``torch`` / ``tvm_ffi``: the
probe child does, in the interpreter the children use.

Not implemented yet (TODO, exit code 2 when requested):

* ``--built-wheel``: build the wheel and import every package from it,
  asserting ``build.json``, ``csrc/cake_ffi_checks.h`` and the sources are
  present (model: scripts/task_test_jit_cache_package_build_import.sh).
* the JIT-path flag check (``get_compile_commands()`` flags are a superset of
  the manifest flags for one entry per package; needs ``FLASHINFER_CUDA_ARCH_LIST``
  set, ``FLASHINFER_JIT_DEBUG`` / ``FLASHINFER_JIT_VERBOSE`` unset, an
  ``isinstance(spec, JitSpecNvcc)`` check and FlashInfer's flag spelling
  ``--compiler-options=-fPIC``) and the "prebuilt path never calls
  ``gen_jit_spec``" check belong to the same compute step.

Exit codes: 0 all checks passed, 1 findings, 2 the check could not run (no
nvcc, no tvm_ffi, no architecture, every selected package skipped) or is not
implemented.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from xml.etree import ElementTree

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


def refs_stepfun(package_dir: Path, loader: str) -> set[str]:
    """Compact rows keyed by the 20-hex program digest (``_PROGRAMS``; the library stem is
    ``_PREFIX + digest``), every program built for every architecture of ``ARCHES``."""
    literals = module_literals(package_dir / loader)
    prefix, arches = literals["_PREFIX"], literals["ARCHES"]
    return {
        f"{prefix}{digest}_{arch}"
        for digest in literals["_PROGRAMS"]
        for arch in arches
    }


def refs_shipped_modules(package_dir: Path, loader: str) -> set[str]:
    """A registry whose ``MODULES`` literal holds the shipped records and whose other tables (``TEST_MODULES``)
    hold programs of a side manifest ``build.json`` never lists: the generic walk over ``MODULES`` alone."""
    return generic_library_refs(module_literals(package_dir / loader)["MODULES"])


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


# dsa_train, kda_train, dsa_indexer, the rmsnorm, the mhc training, the causal conv1d, the ulysses, the NVFP4 SVDQuant GEMM, the Sage
# block-sparse attention, the fused QK RoPE append (BF16 / FP8), the experimental NVFP4 attention, the SM90 VSA,
# the Kimi-K3 AttnRes, the experimental FP8 1D1D GEMM (deepgemm_fp8_gemm) and the experimental batched FP8 projection (deepgemm_batched_gemm) backends are read with the generic walker (a mapping carrying ``library`` under an ``arches`` /
# ``arch`` scope).  Their exporters' registry shapes fit it: a record carries ``arches`` and either ``library`` itself
# (dsa_indexer PROGRAMS, rmsnorm, mhc, causal conv1d, ulysses, svdquant, Sage block-sparse attention, fused QK RoPE append,
# nvfp4_attention, vsa_sm90, Kimi-K3 AttnRes, deepgemm_fp8_gemm and deepgemm_batched_gemm MODULES) or per-stage sub-records carrying ``library`` (dsa_train).
PACKAGES: tuple[Package, ...] = (
    Package(
        "flashinfer.cake_rmsnorm_train_backend",
        "flashinfer/cake_rmsnorm_train_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_causal_conv1d_train_backend",
        "flashinfer/cake_causal_conv1d_train_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_gated_act_mxfp8_backend",
        "flashinfer/cake_gated_act_mxfp8_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_rmsnorm_silu_gate_train_backend",
        "flashinfer/cake_rmsnorm_silu_gate_train_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_ulysses_backend",
        "flashinfer/cake_ulysses_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_fused_qk_rope_append_backend",
        "flashinfer/cake_fused_qk_rope_append_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_sage_block_sparse_attention_backend",
        "flashinfer/cake_sage_block_sparse_attention_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_vsa_sm90_backend",
        "flashinfer/cake_vsa_sm90_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_mhc_train_backend",
        "flashinfer/cake_mhc_train_backend",
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
        "flashinfer.experimental.cake_kda_train",
        "flashinfer/experimental/cake_kda_train",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_gdn_train",
        "flashinfer/experimental/cake_gdn_train",
        "cake_jit.py",
        refs_shipped_modules,
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
        refs_stepfun,
    ),
    Package(
        "flashinfer.cake_nvfp4_svdquant_gemm_backend",
        "flashinfer/cake_nvfp4_svdquant_gemm_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.deepgemm_fp8_gemm",
        "flashinfer/experimental/deepgemm_fp8_gemm",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.cake_fused_qk_rope_fp8_append_backend",
        "flashinfer/cake_fused_qk_rope_fp8_append_backend",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.nvfp4_attention",
        "flashinfer/experimental/nvfp4_attention",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_kimi_k3_attn_res",
        "flashinfer/experimental/cake_kimi_k3_attn_res",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.deepgemm_batched_gemm",
        "flashinfer/experimental/deepgemm_batched_gemm",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.sm110_xqa",
        "flashinfer/experimental/sm110_xqa",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_gdn_det_train",
        "flashinfer/experimental/cake_gdn_det_train",
        "cake_jit.py",
        refs_generic,
    ),
    Package(
        "flashinfer.experimental.cake_gqa_train",
        "flashinfer/experimental/cake_gqa_train",
        "cake_jit.py",
        refs_generic,
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


# ---------------------------------------------------------------------------
# Vendored gate: plain-nvcc build of the manifest, flashinfer-free import
# ---------------------------------------------------------------------------

BUILD_INFO_SCHEMA = "cake-build-info/1"
GATE_BUILDER = "scripts/check_cake_build_manifests.py --vendored-gate"
GATE_CHILD_NAME = "vendored_gate_child.py"
SITECUSTOMIZE_NAME = "sitecustomize.py"
PACKAGE_TABLE = "tests/test_helpers/cake_package.py"
CUDA_CONSTRAINT_RE = re.compile(r"(==|>=)([0-9]+)\.([0-9]+)")
NVCC_RELEASE_RE = re.compile(r"release (\d+)\.(\d+)")
COMPUTE_CAP_RE = re.compile(r"(\d+)\.(\d+)")
NVCC_STDERR_LINES = 40  # kept per failed compile
SUITE_FAILING_IDS = 10  # kept per failed suite
LOAD_CHILD_TIMEOUT = 600  # seconds
JIT_UNAVAILABLE = "flashinfer's JIT is not available"  # cake_build's wording


class GateError(Exception):
    """The vendored gate cannot run (exit code 2)."""


# The gate's one child script, written to ``<work>/vendored_gate_child.py`` and
# run as ``<python> [-I] vendored_gate_child.py <mode> <json>``: ``probe``
# (without ``-I``) reports where the interpreter's dependencies live; ``load``
# and ``suite`` (with ``-I``) block ``flashinfer`` and check the vendored copy.
GATE_CHILD = """\
# Generated by scripts/check_cake_build_manifests.py --vendored-gate.
import importlib
import importlib.abc
import importlib.util
import json
import os
import sys

MODE, CONFIG = sys.argv[1], json.loads(sys.argv[2])


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError(f"flashinfer is blocked by the vendored gate: {name}")
        return None


def emit(**record):
    print(json.dumps(record), flush=True)
    return record["ok"]


def probe():
    import tvm_ffi.libinfo

    paths = [p for p in sys.path if p.endswith(("site-packages", "dist-packages"))]
    for module in ("torch", "tvm_ffi"):
        spec = importlib.util.find_spec(module)
        if spec is None or not spec.origin:
            raise SystemExit(f"{module} is not importable by {sys.executable}")
        paths.append(os.path.dirname(os.path.dirname(os.path.abspath(spec.origin))))
    print(
        json.dumps(
            {
                "paths": list(dict.fromkeys(paths)),
                "tvm_ffi_include": [
                    tvm_ffi.libinfo.find_include_path(),
                    tvm_ffi.libinfo.find_dlpack_include_path(),
                ],
            }
        )
    )


def isolate():
    for p in reversed(CONFIG["paths"]):
        if p not in sys.path:
            sys.path.insert(0, p)
    sys.meta_path.insert(0, Block())
    sys.modules["flashinfer"] = None


def leaked():
    return sorted(
        k
        for k in sys.modules
        if (k == "flashinfer" and sys.modules[k] is not None)
        or k.startswith("flashinfer.")
    )


def load():
    isolate()
    neutral = CONFIG["neutral"]
    copy = os.path.realpath(CONFIG["copy"])
    expected = int(CONFIG["cudart_version"])
    cuda = tuple(CONFIG["cuda"])
    try:
        import torch  # the libraries' CUDA runtime resolves through it

        pkg = importlib.import_module(neutral)
        cake_build = importlib.import_module(neutral + ".cake_build")
    except Exception as exc:
        emit(check="import", ok=False, module=neutral, error=repr(exc))
        return 1
    del torch
    inside = os.path.realpath(pkg.__file__).startswith(copy + os.sep)
    ok = emit(check="import", ok=inside, module=neutral, file=pkg.__file__)
    for entry in CONFIG["entries"]:
        name = entry["name"]
        want = os.path.join(copy, entry["path"])
        try:
            module = cake_build.load_library(name)
            got = os.path.realpath(str(cake_build.library_path(name)))
            built = int(module.build_cuda_version())
        except Exception as exc:
            emit(check="load", ok=False, name=name, error=repr(exc))
            ok = False
            continue
        ok = (
            emit(
                check="load",
                ok=got == want and built == expected,
                name=name,
                path=got,
                build_cuda_version=built,
            )
            and ok
        )
    for arch in CONFIG["arches"]:
        try:
            got = tuple(cake_build.record_cuda_version(arch))
        except Exception as exc:
            emit(check="record_cuda_version", ok=False, arch=arch, error=repr(exc))
            ok = False
            continue
        ok = (
            emit(
                check="record_cuda_version",
                ok=got == cuda,
                arch=arch,
                value=list(got),
            )
            and ok
        )
    modules = leaked()
    ok = emit(check="no_flashinfer", ok=not modules, leaked=modules) and ok
    return 0 if ok else 1


def suite():
    isolate()
    # interpreters a test spawns pick up <work>/sitecustomize.py from here
    os.environ["PYTHONPATH"] = os.pathsep.join(CONFIG["paths"])
    table = importlib.import_module("tests.test_helpers.cake_package")
    table.PACKAGES[CONFIG["key"]] = CONFIG["neutral"]
    import pytest

    return int(pytest.main(CONFIG["pytest_args"]))


if MODE == "probe":
    probe()
    sys.exit(0)
sys.exit(load() if MODE == "load" else suite())
"""

# ``<work>/sitecustomize.py``, written for a ``--suite`` run: the suite child puts
# its ``sys.path`` entries on ``PYTHONPATH``, so an interpreter a test spawns
# imports this module at startup and sees the copy the way the suite child does.
SITECUSTOMIZE = """\
# Generated by scripts/check_cake_build_manifests.py --vendored-gate for one
# --suite run: block flashinfer and point the cake_package table at the copy in
# every interpreter started with this directory on PYTHONPATH.
import importlib
import importlib.abc
import sys

KEY = {key}
NEUTRAL = {neutral}


class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "flashinfer" or name.startswith("flashinfer."):
            raise ImportError("flashinfer is blocked by the vendored gate: " + name)
        return None


sys.meta_path.insert(0, Block())
sys.modules["flashinfer"] = None
importlib.import_module("tests.test_helpers.cake_package").PACKAGES[KEY] = NEUTRAL
"""


@dataclass(frozen=True)
class Gate:
    """What one ``--vendored-gate`` run shares across its packages."""

    root: Path
    work: Path
    child: Path
    python: str
    nvcc: str
    cuda: tuple[int, int]
    nvcc_line: str
    arches: tuple[str, ...]
    includes: tuple[str, ...]
    paths: tuple[str, ...]
    jobs: int
    suites: dict[str, Path]
    pytest_args: tuple[str, ...]


def cuda_constraint_matches(constraint: str, cuda: tuple[int, int]) -> bool:
    """Does toolkit ``cuda`` satisfy ``"==X.Y"`` / ``">=X.Y"``?

    The rule of ``cake_build._cuda_constraint_matches``, re-implemented so that
    nothing of the package is imported here.
    """
    match = CUDA_CONSTRAINT_RE.fullmatch(str(constraint))
    if match is None:
        raise ValueError(
            f"unsupported CUDA constraint {constraint!r}; expected '==X.Y' or '>=X.Y'"
        )
    operator, major, minor = match.groups()
    version = (int(major), int(minor))
    return cuda == version if operator == "==" else cuda >= version


def conditional_flags(items: list[Any], cuda: tuple[int, int]) -> list[str]:
    """The flags of every ``nvcc_flags_if`` item whose ``cuda`` constraint matches."""
    flags: list[str] = []
    for item in items:
        if cuda_constraint_matches(item["cuda"], cuda):
            flags.extend(item["flags"])
    return flags


def version_pair(text: str) -> tuple[int, int]:
    """``(major, minor)`` of ``"12.8"`` (an entry's ``min_cuda``)."""
    parts = str(text).split(".")
    return (int(parts[0]), int(parts[1]))


def cudart_version(cuda: tuple[int, int]) -> int:
    """``CUDART_VERSION`` of toolkit ``cuda``: ``1000 * major + 10 * minor``."""
    return 1000 * cuda[0] + 10 * cuda[1]


def executable(spec: str) -> str:
    """``spec`` as the children can run it from another directory: a path with a
    separator made absolute (the compiles run from the copy, the children from
    the work directory or the repository root); a bare name stays a PATH lookup."""
    return os.path.abspath(spec) if os.sep in spec else spec


def find_nvcc(explicit: str | None) -> str:
    """``--nvcc``, else ``$CUDA_HOME/bin/nvcc``, else ``nvcc`` on ``PATH``; a path
    is made absolute (:func:`executable`)."""
    if explicit:
        return executable(explicit)
    cuda_home = os.environ.get("CUDA_HOME")
    if cuda_home:
        return executable(str(Path(cuda_home) / "bin" / "nvcc"))
    found = shutil.which("nvcc")
    if found is None:
        raise GateError("no nvcc: pass --nvcc, set CUDA_HOME or put nvcc on PATH")
    return executable(found)


def nvcc_toolkit(nvcc: str) -> tuple[tuple[int, int], str]:
    """``((major, minor), last line)`` of ``nvcc --version`` (``release X.Y``)."""
    try:
        proc = subprocess.run(
            [nvcc, "--version"], capture_output=True, text=True, check=False
        )
    except OSError as exc:
        raise GateError(f"cannot run {nvcc} --version: {exc}") from exc
    match = NVCC_RELEASE_RE.search(proc.stdout)
    if proc.returncode or match is None:
        raise GateError(
            f"{nvcc} --version (exit {proc.returncode}) reports no 'release X.Y':\n"
            f"{proc.stdout}{proc.stderr}"
        )
    last = proc.stdout.strip().splitlines()[-1].strip()
    return (int(match.group(1)), int(match.group(2))), last


def visible_gpu_arches() -> list[str]:
    """``sm_<major><minor>a`` of every visible GPU (``nvidia-smi``'s ``compute_cap``,
    ``10.0`` -> ``sm_100a``); empty without nvidia-smi or without a GPU."""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    if proc.returncode:
        return []
    arches: list[str] = []
    for line in proc.stdout.splitlines():
        match = COMPUTE_CAP_RE.fullmatch(line.strip())
        if match is not None:
            arches.append(f"sm_{int(match.group(1))}{int(match.group(2))}a")
    return list(dict.fromkeys(arches))


def probe_environment(python: str, child: Path) -> dict[str, list[str]]:
    """``{"paths": [...], "tvm_ffi_include": [...]}`` of ``python``: the dependency
    paths its ``-I`` children need and the include directories nvcc needs."""
    try:
        proc = subprocess.run(
            [python, str(child), "probe", "{}"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise GateError(f"cannot run {python}: {exc}") from exc
    lines = proc.stdout.strip().splitlines()
    if proc.returncode or not lines:
        raise GateError(
            f"{python} cannot report its torch / tvm_ffi paths "
            f"(exit {proc.returncode}):\n{proc.stderr.strip()}"
        )
    return json.loads(lines[-1])


def parse_suites(
    root: Path, specs: list[str] | None, packages: list[Package]
) -> dict[str, Path]:
    """``--suite PACKAGE=TESTFILE`` -> ``{package.name: test file}``.

    The package must be among the selected ones; the file must exist; one test
    file per package (an identical repeat is accepted, a different file is an
    error: further files go through ``--pytest-args``).
    """
    suites: dict[str, Path] = {}
    for spec in specs or ():
        name, sep, testfile = spec.partition("=")
        if not sep or not name or not testfile:
            raise GateError(f"--suite expects PACKAGE=TESTFILE, got {spec!r}")
        try:
            (package,) = select([name])
        except (SystemExit, ValueError) as exc:
            raise GateError(f"--suite {spec!r}: {exc}") from exc
        if package not in packages:
            raise GateError(
                f"--suite {spec!r}: {package.path} is not among the selected packages"
            )
        path = Path(testfile)
        if not path.is_absolute():
            path = root / path
        if not path.is_file():
            raise GateError(f"--suite {spec!r}: {path} is not a file")
        resolved = path.resolve()
        previous = suites.get(package.name)
        if previous is not None and previous != resolved:
            raise GateError(
                f"--suite {spec!r}: {package.path} already has the suite "
                f"{previous}; one test file per package (pass further files "
                "through --pytest-args)"
            )
        suites[package.name] = resolved
    return suites


def gate_entries(child: Path, reports: list[dict[str, Any]]) -> list[Path]:
    """The work-directory entries the gate wrote: the child script and, per
    report, the copy, the load log and the suite's junit file, log and
    sitecustomize module (every path read from the report, so a run that
    wrote no suite lists none of its files)."""
    entries = [child]
    for report in reports:
        if "copy" in report:
            entries.append(Path(report["copy"]))
        load = report.get("load") or {}
        suite = report.get("suite") or {}
        paths = (
            load.get("log"),
            suite.get("junit"),
            suite.get("log"),
            suite.get("sitecustomize"),
        )
        entries.extend(Path(path) for path in paths if path)
    return entries


def discard_work(work: Path, created: bool, entries: list[Path]) -> None:
    """Remove the work directory when the gate created it; a pre-existing
    ``--work-dir`` is kept and only ``entries`` (what the gate wrote) go."""
    if created:
        shutil.rmtree(work, ignore_errors=True)
        return
    for entry in entries:
        if entry.is_dir():
            shutil.rmtree(entry, ignore_errors=True)
        else:
            entry.unlink(missing_ok=True)
    print(f"work dir kept (pre-existing): {work}")


def copy_package(package_dir: Path, work: Path) -> tuple[str, Path]:
    """Step 1: the package under ``<work>/vendored_<directory name>/`` without
    ``aot/`` and ``__pycache__/``, with an empty ``aot/`` for the builds."""
    neutral = f"vendored_{package_dir.name}"
    copy = work / neutral
    if copy.exists():
        shutil.rmtree(copy)
    shutil.copytree(
        package_dir, copy, ignore=shutil.ignore_patterns("aot", "__pycache__")
    )
    (copy / "aot").mkdir()
    return neutral, copy


def entry_flags(
    manifest: dict[str, Any], entry: dict[str, Any], cuda: tuple[int, int]
) -> list[str]:
    """The nvcc flags of ``entry`` from the manifest alone: ``common.nvcc_flags`` +
    the matching ``common.nvcc_flags_if`` + ``arch_flags[arch]`` + the entry's
    ``nvcc_flags`` + its matching ``nvcc_flags_if``."""
    common = manifest["common"]
    return [
        *common["nvcc_flags"],
        *conditional_flags(common["nvcc_flags_if"], cuda),
        *manifest["arch_flags"][entry["arch"]],
        *entry["nvcc_flags"],
        *conditional_flags(entry["nvcc_flags_if"], cuda),
    ]


def compile_command(
    nvcc: str,
    copy: Path,
    manifest: dict[str, Any],
    entry: dict[str, Any],
    cuda: tuple[int, int],
    includes: Iterable[str],
) -> list[str]:
    """The plain-nvcc command of ``entry``, run from ``copy``: the manifest flags,
    ``-I`` for ``common.include_dirs`` and ``includes`` (tvm-ffi, dlpack), the
    copy-relative sources, ``common.link_flags`` and ``-o <copy>/<entry.path>``."""
    common = manifest["common"]
    include_dirs = [str(copy / d) for d in common["include_dirs"]] + list(includes)
    return [
        nvcc,
        *entry_flags(manifest, entry, cuda),
        *(f"-I{d}" for d in include_dirs),
        *entry["sources"],
        *common["link_flags"],
        "-o",
        str(copy / entry["path"]),
    ]


def run_compile(command: list[str], copy: Path) -> dict[str, Any]:
    """One nvcc run from ``copy``: ``rc``, ``seconds`` and, on failure, ``stderr``
    (``rc`` None and ``error`` when nvcc cannot be started)."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            command, cwd=copy, capture_output=True, text=True, check=False
        )
    except OSError as exc:
        return {
            "rc": None,
            "seconds": round(time.monotonic() - start, 1),
            "error": f"cannot run {command[0]}: {exc}",
        }
    record: dict[str, Any] = {
        "rc": proc.returncode,
        "seconds": round(time.monotonic() - start, 1),
    }
    if proc.returncode:
        record["stderr"] = proc.stderr.splitlines()[:NVCC_STDERR_LINES]
    return record


def compile_entries(
    gate: Gate, copy: Path, manifest: dict[str, Any], entries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Step 2: compile ``entries`` ``gate.jobs`` at a time; one record per entry in
    manifest order (``name``, ``arch``, ``path``, ``command``, ``rc``, ``seconds``,
    ``output``; ``stderr`` on a failed run, ``error`` on an unformable command or
    an nvcc that cannot start)."""
    pending: list[tuple[dict[str, Any], Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, gate.jobs)) as pool:
        for entry in entries:
            record: dict[str, Any] = {
                "name": entry["name"],
                "arch": entry["arch"],
                "path": entry["path"],
            }
            try:
                command = compile_command(
                    gate.nvcc, copy, manifest, entry, gate.cuda, gate.includes
                )
            except (KeyError, TypeError, ValueError) as exc:
                record.update(rc=None, error=f"cannot form the nvcc command: {exc}")
                pending.append((record, None))
                continue
            record["command"] = command
            pending.append((record, pool.submit(run_compile, command, copy)))
        records: list[dict[str, Any]] = []
        for record, future in pending:
            if future is not None:
                record.update(future.result())
            record["output"] = (copy / record["path"]).is_file()
            records.append(record)
    return records


def compiled_ok(record: dict[str, Any]) -> bool:
    """Did ``record``'s nvcc run exit 0 and leave its output file?"""
    return record["rc"] == 0 and bool(record["output"])


def compile_findings(
    package: Package, copy: Path, records: list[dict[str, Any]]
) -> list[Finding]:
    """One finding per failed compile record: a command that could not be formed
    or run, a non-zero nvcc exit (with the kept stderr lines) or a missing output
    file."""
    found: list[Finding] = []
    for record in records:
        name = record["name"]
        if record["rc"] is None:
            found.append(Finding(package.name, f"{name}: {record['error']}"))
        elif record["rc"]:
            tail = "\n    ".join(record.get("stderr") or ["(no stderr)"])
            found.append(
                Finding(package.name, f"{name}: nvcc exit {record['rc']}\n    {tail}")
            )
        elif not record["output"]:
            found.append(
                Finding(
                    package.name,
                    f"{name}: nvcc exit 0 but {copy / record['path']} is missing",
                )
            )
    return found


def write_build_info(
    copy: Path, cuda: tuple[int, int], arches: Iterable[str], nvcc_line: str
) -> dict[str, Any]:
    """Step 3: ``<copy>/aot/build_info.json``, the ``cake-build-info/1`` sidecar
    ``cake_build`` reads (``schema``, ``cuda``, ``cudart_version``, ``arches``;
    ``builder`` and ``nvcc`` are extra keys it ignores)."""
    info = {
        "schema": BUILD_INFO_SCHEMA,
        "cuda": f"{cuda[0]}.{cuda[1]}",
        "cudart_version": cudart_version(cuda),
        "arches": list(arches),
        "builder": GATE_BUILDER,
        "nvcc": nvcc_line,
    }
    (copy / "aot" / "build_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def run_child(
    gate: Gate, mode: str, config: dict[str, Any], cwd: Path, timeout: int | None
) -> subprocess.CompletedProcess[str]:
    """``<python> -I <work>/vendored_gate_child.py <mode> <json>`` from ``cwd``."""
    return subprocess.run(
        [gate.python, "-I", str(gate.child), mode, json.dumps(config)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
    )


def json_lines(text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """The ``{"check": ...}`` records among ``text``'s lines, and the other lines."""
    records: list[dict[str, Any]] = []
    other: list[str] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except ValueError:
            record = None
        if isinstance(record, dict) and "check" in record:
            records.append(record)
        elif line.strip():
            other.append(line)
    return records, other


def captured(data: bytes | str | None) -> str:
    """The text of a child's captured stream: ``subprocess.TimeoutExpired`` carries
    bytes even under ``text=True``."""
    if isinstance(data, bytes):
        return data.decode(errors="replace")
    return data or ""


def load_check(
    gate: Gate,
    package: Package,
    neutral: str,
    copy: Path,
    compiled: list[dict[str, Any]],
    arches: list[str],
) -> tuple[dict[str, Any], list[Finding]]:
    """Step 4: the flashinfer-blocked child imports the copy and loads every
    compiled entry through ``cake_build``; every failed check is a finding."""
    config = {
        "paths": [str(gate.work), *gate.paths],
        "neutral": neutral,
        "copy": str(copy),
        "entries": [{"name": r["name"], "path": r["path"]} for r in compiled],
        "arches": arches,
        "cuda": list(gate.cuda),
        "cudart_version": cudart_version(gate.cuda),
    }
    log = gate.work / f"{neutral}-load.log"
    try:
        proc = run_child(gate, "load", config, gate.work, LOAD_CHILD_TIMEOUT)
    except subprocess.TimeoutExpired as exc:
        log.write_text(captured(exc.stdout) + captured(exc.stderr))
        message = f"load child timed out after {LOAD_CHILD_TIMEOUT} s (log {log})"
        return {"rc": None, "checks": [], "log": str(log)}, [
            Finding(package.name, message)
        ]
    log.write_text(proc.stdout + proc.stderr)
    checks, other = json_lines(proc.stdout)
    found: list[Finding] = []
    for check in checks:
        if check.get("ok"):
            continue
        what = check.get("name") or check.get("arch") or check.get("module") or ""
        detail = check.get("error") or json.dumps(
            {k: v for k, v in check.items() if k not in ("check", "ok")}
        )
        found.append(Finding(package.name, f"{check['check']} {what}: {detail}"))
    if proc.returncode and not found:
        tail = "\n    ".join(
            (proc.stderr or proc.stdout).strip().splitlines()[-NVCC_STDERR_LINES:]
        )
        found.append(
            Finding(package.name, f"load child exit {proc.returncode}\n    {tail}")
        )
    report = {
        "rc": proc.returncode,
        "checks": checks,
        "other_output": other,
        "log": str(log),
    }
    return report, found


def junit_summary(path: Path) -> dict[str, Any]:
    """``passed`` / ``failed`` / ``skipped`` and the ``failing`` ids of a junit file."""
    root = ElementTree.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    tests = failures = errors = skipped = 0
    failing: list[str] = []
    for suite in suites:
        tests += int(suite.get("tests", 0))
        failures += int(suite.get("failures", 0))
        errors += int(suite.get("errors", 0))
        skipped += int(suite.get("skipped", 0))
        for case in suite.iter("testcase"):
            if case.find("failure") is not None or case.find("error") is not None:
                failing.append(f"{case.get('classname')}::{case.get('name')}")
    return {
        "passed": tests - failures - errors - skipped,
        "failed": failures + errors,
        "skipped": skipped,
        "failing": failing,
    }


def suite_key(root: Path, package: Package) -> str | None:
    """The ``tests.test_helpers.cake_package.PACKAGES`` key importing ``package``."""
    table = module_literals(root / PACKAGE_TABLE).get("PACKAGES") or {}
    for key, path in table.items():
        if path == package.name:
            return str(key)
    return None


def run_suite(
    gate: Gate, package: Package, neutral: str, testfile: Path
) -> tuple[dict[str, Any] | None, list[Finding]]:
    """Step 5: the backend suite ``testfile`` against the copy, flashinfer blocked
    in the suite child and in every interpreter its tests spawn."""
    key = suite_key(gate.root, package)
    if key is None:
        message = (
            f"{PACKAGE_TABLE} has no entry importing {package.name}; "
            f"the suite {testfile} cannot be pointed at the copy"
        )
        return None, [Finding(package.name, message)]
    junit = gate.work / f"{key}-suite.xml"
    log = gate.work / f"{key}-suite.log"
    sitecustomize = gate.work / SITECUSTOMIZE_NAME
    pytest_args = [
        str(testfile),
        "--noconftest",
        "-p",
        "no:cacheprovider",
        "-q",
        "-rfEs",
        "--junitxml",
        str(junit),
        *gate.pytest_args,
    ]
    config = {
        "paths": [str(gate.root), str(gate.work), *gate.paths],
        "key": key,
        "neutral": neutral,
        "pytest_args": pytest_args,
    }
    sitecustomize.write_text(
        SITECUSTOMIZE.format(key=json.dumps(key), neutral=json.dumps(neutral))
    )
    # a reused work directory may hold the junit file of an earlier run; pytest
    # writes its own only at session finish, so a child that dies before that
    # must not have the old counts attributed to it
    junit.unlink(missing_ok=True)
    proc = run_child(gate, "suite", config, gate.root, None)
    log.write_text(proc.stdout + proc.stderr)
    if junit.is_file():
        summary = junit_summary(junit)
    else:
        summary = {"passed": 0, "failed": 0, "skipped": 0, "failing": []}
    summary.update(
        rc=proc.returncode,
        key=key,
        testfile=str(testfile),
        junit=str(junit),
        log=str(log),
        sitecustomize=str(sitecustomize),
    )
    found: list[Finding] = []
    if summary["failed"] or proc.returncode:
        message = (
            f"suite {testfile}: {summary['passed']} passed {summary['failed']} failed "
            f"{summary['skipped']} skipped, pytest exit {proc.returncode} (log {log})"
        )
        if summary["failing"]:
            ids = ", ".join(summary["failing"][:SUITE_FAILING_IDS])
            message += f"; failing: {ids}"
        if JIT_UNAVAILABLE in proc.stdout + proc.stderr:
            message += (
                "; a test needs a library the gate did not compile "
                "(a manifest / --arch gap)"
            )
        found.append(Finding(package.name, message))
    return summary, found


def gate_package(gate: Gate, package: Package) -> tuple[dict[str, Any], list[Finding]]:
    """Steps 1-5 for one package: its report record and its findings."""
    report: dict[str, Any] = {"package": package.name, "skipped": False}
    package_dir = gate.root / package.path
    manifest_path = package_dir / "build.json"
    if not manifest_path.is_file():
        report["error"] = "build.json is missing"
        return report, [Finding(package.name, report["error"])]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as error:
        report["error"] = f"build.json is not valid JSON: {error}"
        return report, [Finding(package.name, report["error"])]
    arch_flags = manifest["arch_flags"]
    arches = [arch for arch in gate.arches if arch in arch_flags]
    testfile = gate.suites.get(package.name)
    if not arches:
        report.update(
            skipped=True,
            reason=f"arch_flags {sorted(arch_flags)} has none of {list(gate.arches)}",
        )
        if testfile is not None:
            # the request stays visible in the report line and the JSON record;
            # like the skip itself, it is not a finding
            report["suite"] = {
                "testfile": str(testfile),
                "not_run": "package skipped (no selected architecture)",
            }
        return report, []
    entries = [e for e in manifest["libraries"] if e["arch"] in arches]
    skipped = [e["name"] for e in entries if version_pair(e["min_cuda"]) > gate.cuda]
    to_build = [e for e in entries if e["name"] not in set(skipped)]
    neutral, copy = copy_package(package_dir, gate.work)
    records = compile_entries(gate, copy, manifest, to_build)
    findings = compile_findings(package, copy, records)
    compiled = [r for r in records if compiled_ok(r)]
    built_arches = [a for a in arches if any(r["arch"] == a for r in compiled)]
    report.update(
        neutral=neutral,
        copy=str(copy),
        arches=arches,
        skipped_min_cuda=skipped,
        compiled=records,
        build_info=write_build_info(copy, gate.cuda, built_arches, gate.nvcc_line),
        load=None,
        suite=None,
    )
    if compiled:
        report["load"], found = load_check(
            gate, package, neutral, copy, compiled, built_arches
        )
        findings.extend(found)
        if testfile is not None:
            report["suite"], found = run_suite(gate, package, neutral, testfile)
            findings.extend(found)
    elif testfile is not None:
        findings.append(
            Finding(package.name, f"suite {testfile} not run: no entry compiled")
        )
    return report, findings


def suite_text(suite: dict[str, Any] | None) -> str:
    """The suite part of a report line: the counts, or why a requested suite did
    not run; empty without a suite."""
    if suite is None:
        return ""
    if "not_run" in suite:
        return f", suite {suite['testfile']} not run: {suite['not_run']}"
    return (
        f", suite: {suite['passed']} passed {suite['failed']} failed "
        f"{suite['skipped']} skipped"
    )


def summary_line(report: dict[str, Any]) -> str:
    """The one report line of a package."""
    name = report["package"]
    if "error" in report:
        return f"{name}: {report['error']}"
    if report["skipped"]:
        text = f"{name}: skipped ({report['reason']})"
        return text + suite_text(report.get("suite"))
    records = report["compiled"]
    ok = sum(1 for r in records if compiled_ok(r))
    text = (
        f"{name}: compiled {ok} of {len(records)} "
        f"(skipped {len(report['skipped_min_cuda'])} by min_cuda)"
    )
    load = report["load"]
    if load is None:
        text += ", load check not run"
    else:
        checks = load["checks"]
        loaded = sum(1 for c in checks if c["check"] == "load" and c["ok"])
        passed = bool(checks) and load["rc"] == 0 and all(c["ok"] for c in checks)
        version = "ok" if passed else "FAILED"
        text += f", loaded {loaded} of {ok}, build_cuda_version {version}"
    return text + suite_text(report["suite"])


def run_vendored_gate(
    args: argparse.Namespace, root: Path, packages: list[Package]
) -> int:
    """``--vendored-gate`` over ``packages`` (the module docstring's steps).

    Returns the exit code; raises :class:`GateError` when the gate cannot run.
    """
    nvcc = find_nvcc(args.nvcc)
    cuda, nvcc_line = nvcc_toolkit(nvcc)
    python = executable(args.python)
    arches = list(dict.fromkeys(args.arch or visible_gpu_arches()))
    if not arches:
        raise GateError("no architecture: no visible GPU (nvidia-smi) and no --arch")
    bad = [arch for arch in arches if not ARCH_RE.match(arch)]
    if bad:
        raise GateError(f"--arch {bad}: not sm_XXX[a] architectures")
    suites = parse_suites(root, args.suite, packages)
    if args.work_dir is not None:
        work = args.work_dir.resolve()
        created = not work.exists()
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="cake-vendored-gate-"))
        created = True
    child = work / GATE_CHILD_NAME
    child.write_text(GATE_CHILD)
    print(
        f"vendored gate: {nvcc} (CUDA {cuda[0]}.{cuda[1]}; {nvcc_line}), "
        f"arches {arches}, {args.jobs} job(s), work dir {work}"
    )
    try:
        probe = probe_environment(python, child)
    except GateError:
        if not args.keep:
            discard_work(work, created, gate_entries(child, []))
        raise
    gate = Gate(
        root=root,
        work=work,
        child=child,
        python=python,
        nvcc=nvcc,
        cuda=cuda,
        nvcc_line=nvcc_line,
        arches=tuple(arches),
        includes=tuple(probe["tvm_ffi_include"]),
        paths=tuple(probe["paths"]),
        jobs=args.jobs,
        suites=suites,
        pytest_args=tuple(shlex.split(args.pytest_args)),
    )
    reports: list[dict[str, Any]] = []
    findings: list[Finding] = []
    for package in packages:
        report, found = gate_package(gate, package)
        reports.append(report)
        findings.extend(found)
        print(summary_line(report))
    if all(report["skipped"] for report in reports):
        if not args.keep:
            discard_work(work, created, gate_entries(child, reports))
        raise GateError(f"every selected package is skipped: none builds {arches}")
    for finding in findings:
        print(finding)
    if args.report_json:
        document = {
            "mode": "vendored-gate",
            "toolkit": f"{cuda[0]}.{cuda[1]}",
            "nvcc": nvcc_line,
            "arches": arches,
            "packages": reports,
            "findings": [
                {"package": f.package, "message": f.message} for f in findings
            ],
        }
        args.report_json.write_text(json.dumps(document, indent=2) + "\n")
    print(f"vendored gate: {len(packages)} package(s): {len(findings)} finding(s)")
    if args.keep or findings:
        print(f"work dir kept: {work}")
    else:
        discard_work(work, created, gate_entries(child, reports))
    return 1 if findings else 0


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
        help=(
            "Copy each package, build its manifest with plain nvcc and import the "
            "copy with flashinfer blocked (the module docstring's steps; needs "
            "nvcc, tvm_ffi and a GPU or --arch)"
        ),
    )
    parser.add_argument(
        "--built-wheel",
        action="store_true",
        help="TODO: built-wheel import check (not implemented; exit 2)",
    )
    parser.add_argument("--report-json", type=Path, help="Write findings as JSON")
    gate = parser.add_argument_group("vendored gate options")
    gate.add_argument(
        "--arch",
        action="append",
        help=(
            "Architecture to compile (sm_100a, ...); repeatable; default: the "
            "visible GPUs' compute capabilities (nvidia-smi)"
        ),
    )
    gate.add_argument(
        "--nvcc",
        help=(
            "nvcc to run (default: $CUDA_HOME/bin/nvcc, else nvcc on PATH); a "
            "relative path is resolved against the current directory"
        ),
    )
    gate.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel nvcc invocations (default: %(default)s)",
    )
    gate.add_argument(
        "--work-dir",
        type=Path,
        help=(
            "Work directory (default: a fresh cake-vendored-gate-* under $TMPDIR); "
            "without --keep a directory the gate created is removed after a clean "
            "run, a pre-existing one only loses the entries the gate wrote"
        ),
    )
    gate.add_argument(
        "--keep",
        action="store_true",
        help="Keep the work directory (always kept when there are findings)",
    )
    gate.add_argument(
        "--suite",
        action="append",
        metavar="PACKAGE=TESTFILE",
        help=(
            "After the load check, run this backend pytest file against the copy "
            "with flashinfer blocked; repeatable, one test file per package "
            "(further files: --pytest-args)"
        ),
    )
    gate.add_argument(
        "--pytest-args",
        default="",
        help="Extra pytest arguments for --suite (one string, split like a shell)",
    )
    gate.add_argument(
        "--python",
        default=sys.executable,
        help=(
            "Interpreter of the gate's child processes (default: this one); a "
            "relative path is resolved against the current directory"
        ),
    )
    args = parser.parse_args()

    if args.built_wheel:
        print(
            "--built-wheel: not implemented yet (see the module docstring TODO); "
            "run on allocated compute once it lands",
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
    if args.vendored_gate:
        try:
            return run_vendored_gate(args, root, packages)
        except GateError as exc:
            print(f"--vendored-gate: {exc}", file=sys.stderr)
            return 2
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
