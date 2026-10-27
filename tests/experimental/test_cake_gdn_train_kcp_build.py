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

The registry (``cake_jit``), the loader glue (``cake_backend``) and the manifests of the Cake packed GDN KCP training
package: the package contract that needs no device -- ``build.json`` (schema ``cake-build/1``, the shipped programs,
never a test-only one, digests over the listed sources), ``build.test.json`` (the test-only programs), the program
names against the identity rule, the records against the frozen kernel ABIs, the loader's independence from
``flashinfer.jit`` (the prebuilt route) and the package's independence from the KDA training package.  Nothing is
built or launched.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.test_helpers.cake_package import PACKAGES, cake_package

PACKAGE = "cake_gdn_train"
cake_build = cake_package(PACKAGE, "cake_build")
PACKAGE_DIR = Path(cake_build.__file__).parent
PACKAGE_IMPORT = PACKAGES[PACKAGE]
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]
PREFIX = "cake_gdn_train"
_W = "sm100_sm103_sm107"

# The nine programs of a step (the shipped records by program name: sixteen libraries) and the two test-only records.
SHIPPED_PROGRAMS = (
    "fwd_prep",
    "kcp_fwdh_offset",
    "kcp_fwdh_tape",
    "kcp_transfer_walk",
    "kcp_affine_merge",
    "fwd_o",
    "bwd_prep",
    "kcp_dhu_offset",
    "kcp_dhu_tape",
    "bwd_dqkg",
    "gate_epilogue",
)
TEST_RECORDS = ("kcp_transfer_walk_three_mma", "kcp_transfer_walk_plain_tf32")
# The identity rule: ``cake_gdn_train_<20 hex of sha256("<kernel module>\\n<sorted JSON of the build keywords>")>``.
# The build keywords are the contract of each record; a keyword spelled at its default names the same program as an
# omitted one, so these pairs are frozen together with the kernels' build signatures.
IDENTITIES = {
    "fwd_prep": (f"tart.gdn_chunk_train_fwd_prep.{_W}.gdn_chunk_train_fwd_prep_v1", {}),
    "kcp_fwdh_offset": (
        f"tart.gdn_chunk_kcp_fwd_h.{_W}.gdn_chunk_kcp_fwdh_v1",
        {"program": "offset"},
    ),
    "kcp_fwdh_tape": (
        f"tart.gdn_chunk_kcp_fwd_h.{_W}.gdn_chunk_kcp_fwdh_v1",
        {"program": "tape"},
    ),
    "kcp_fwdh_offset_vh64": (
        f"tart.gdn_chunk_kcp_fwd_h.{_W}.gdn_chunk_kcp_fwdh_v1",
        {"program": "offset", "vh": 64},
    ),
    "kcp_fwdh_tape_vh64": (
        f"tart.gdn_chunk_kcp_fwd_h.{_W}.gdn_chunk_kcp_fwdh_v1",
        {"program": "tape", "vh": 64},
    ),
    "kcp_transfer_walk": (
        f"tart.gdn_chunk_kcp_transfer_walk.{_W}.gdn_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 0, "compute_warps": 8},
    ),
    "kcp_transfer_walk_three_mma": (
        f"tart.gdn_chunk_kcp_transfer_walk.{_W}.gdn_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 1},
    ),
    "kcp_transfer_walk_plain_tf32": (
        f"tart.gdn_chunk_kcp_transfer_walk.{_W}.gdn_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 2},
    ),
    "kcp_affine_merge_c8": (
        f"tart.kda_chunk_kcp_affine_merge.{_W}.kda_chunk_kcp_affine_merge_fp32_v1",
        {"cols_per_cta": 8},
    ),
    "kcp_affine_merge_c16": (
        f"tart.kda_chunk_kcp_affine_merge.{_W}.kda_chunk_kcp_affine_merge_fp32_v1",
        {"cols_per_cta": 16},
    ),
    "fwd_o": (f"tart.gdn_chunk_train_fwd_o.{_W}.gdn_chunk_train_fwd_o_v1", {}),
    "bwd_prep": (f"tart.gdn_chunk_train_bwd_prep.{_W}.gdn_chunk_train_bwd_prep_v1", {}),
    "kcp_dhu_offset": (
        f"tart.gdn_chunk_kcp_dhu.{_W}.gdn_chunk_kcp_dhu_v1",
        {"program": "offset"},
    ),
    "kcp_dhu_tape": (
        f"tart.gdn_chunk_kcp_dhu.{_W}.gdn_chunk_kcp_dhu_v1",
        {"program": "tape"},
    ),
    "kcp_dhu_offset_vh64": (
        f"tart.gdn_chunk_kcp_dhu.{_W}.gdn_chunk_kcp_dhu_v1",
        {"program": "offset", "vh": 64},
    ),
    "kcp_dhu_tape_vh64": (
        f"tart.gdn_chunk_kcp_dhu.{_W}.gdn_chunk_kcp_dhu_v1",
        {"program": "tape", "vh": 64},
    ),
    "bwd_dqkg": (
        f"tart.gdn_chunk_train_bwd_dqkg_fused.{_W}.gdn_chunk_train_bwd_dqkg_v1",
        {},
    ),
    "gate_epilogue": (
        f"tart.gdn_chunk_train_gate_epilogue.{_W}.gdn_chunk_train_gate_epilogue_v1",
        {},
    ),
}
# The grid rules the host evaluates: the walks launch one CTA per listed segment and value head, the affine merge one
# CTA per chain and column block, the chunk-shaped programs one CTA per chunk and (Q/K or value) head.
GRID_RULES = (
    "kcp_segments_heads",
    "kcp_chains",
    "chunks_by_qk_heads",
    "chunks_heads_persistent",
    "chunks_qk_heads_persistent",
    "chunk_head_items",
)
SELECT_RULES = ("always", "columns", "ctas_per_head")


def _registry():
    """The registry module, or a skip while the seed registry is not in the tree yet."""
    return pytest.importorskip(f"{PACKAGE_IMPORT}.cake_jit")


def _backend():
    """The backend module; it imports the registry, so it skips with it while the seed registry is not in the tree."""
    _registry()
    return importlib.import_module(f"{PACKAGE_IMPORT}.cake_backend")


def _records(cake_jit) -> dict[str, dict]:
    return {**cake_jit.MODULES, **cake_jit.TEST_MODULES}


def _manifest_rendered() -> bool:
    return bool(cake_build.manifest()["libraries"])


# ----------------------------------------------------------------------------------------------------------------
# The package's independence: no KDA training import, no environment read, importable without flashinfer.jit


def _python_files() -> list[Path]:
    return sorted(p for p in PACKAGE_DIR.glob("*.py"))


def test_the_package_imports_nothing_from_the_kda_training_package() -> None:
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [alias.name for alias in node.names]
            else:
                continue
            if any("cake_kda_train" in name for name in names):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_the_package_reads_no_environment_variable() -> None:
    offenders = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr in ("environ", "getenv")
                and isinstance(node.value, ast.Name)
                and node.value.id == "os"
            ):
                offenders.append(f"{path.name}:{node.lineno}")
    assert not offenders, offenders


def test_the_loader_imports_standalone_without_flashinfer() -> None:
    """The vendorable form: ``cake_build.py`` loaded by file path from the package directory, in a process where
    ``flashinfer`` cannot be imported, reads the manifest (the prebuilt route imports nothing from FlashInfer)."""
    code = (
        "import importlib.util, sys\n"
        "sys.modules['flashinfer'] = None\n"
        f"spec = importlib.util.spec_from_file_location('vendored_cake_build', {str(PACKAGE_DIR / 'cake_build.py')!r})\n"
        "b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)\n"
        "assert sys.modules['flashinfer'] is None\n"
        "m = b.manifest(); assert m['schema'] == 'cake-build/1'\n"
        "print('ok', m['package'])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"ok {PACKAGE_IMPORT}"


def test_the_loader_and_header_are_the_pinned_bytes_of_every_cake_package() -> None:
    """``cake_build.py`` is byte-identical across the Cake packages (the manifest checker's rule); the binding checks
    header too."""
    kda_dir = Path(
        importlib.import_module(
            "flashinfer.experimental.cake_kda_train.cake_build"
        ).__file__
    ).parent
    for relative in ("cake_build.py", "csrc/cake_ffi_checks.h"):
        ours = (PACKAGE_DIR / relative).read_bytes()
        theirs = (kda_dir / relative).read_bytes()
        assert hashlib.sha256(ours).hexdigest() == hashlib.sha256(theirs).hexdigest(), (
            relative
        )


# ----------------------------------------------------------------------------------------------------------------
# The registry


def test_the_tables_are_disjoint_and_complete() -> None:
    cake_jit = _registry()
    shipped_programs = {record["program"] for record in cake_jit.MODULES.values()}
    assert shipped_programs == set(SHIPPED_PROGRAMS)
    assert set(cake_jit.TEST_MODULES) == set(TEST_RECORDS)
    assert not set(cake_jit.MODULES) & set(cake_jit.TEST_MODULES)
    assert tuple(cake_jit.ARCHES) == ARCHES
    for name, record in cake_jit.TEST_MODULES.items():
        assert record["program"] in shipped_programs, name


@pytest.mark.parametrize("name", sorted(IDENTITIES))
def test_record_matches_the_frozen_abi(name: str) -> None:
    cake_jit = _registry()
    record = _records(cake_jit)[name]
    assert record["arches"] == list(ARCHES)
    assert record["ffi_entry"] == "run"
    assert record["kernel_symbol"] == f"kernel_{record['library']}"
    assert (
        record["launch"]["cooperative"] is False
        and record["launch"]["use_pdl"] is False
    )
    assert tuple(record["launch"]["cluster"]) == (1, 1, 1)
    assert isinstance(record["launch"]["dynamic_smem_bytes"], int)
    assert record["select_rule"]["kind"] in SELECT_RULES
    assert record["grid_rule"]["kind"] in GRID_RULES
    expected_name = (
        record["program"]
        if record["variant"] is None
        else f"{record['program']}_{record['variant']}"
    )
    assert name == expected_name
    assert record["arg_plan"][-3:] == GRID
    kept = {parameter for kind, parameter in record["arg_plan"] if kind != "grid"}
    assert set(record["logical"]) == kept
    for kind, parameter in record["arg_plan"]:
        if kind == "grid":
            continue
        vocabulary = (
            cake_jit.LOGICAL_TENSORS if kind == "buffer" else cake_jit.LOGICAL_SCALARS
        )
        assert record["logical"][parameter] in vocabulary, (name, parameter)


def test_program_names_follow_the_identity_rule() -> None:
    cake_jit = _registry()
    records = _records(cake_jit)
    for name, (module, kwargs) in IDENTITIES.items():
        digest = hashlib.sha256(
            f"{module}\n{json.dumps(kwargs, sort_keys=True)}".encode()
        ).hexdigest()
        assert records[name]["library"] == f"{PREFIX}_{digest[:20]}", name
    libraries = [record["library"] for record in records.values()]
    assert len(set(libraries)) == len(libraries), "every record is its own program"
    assert all(library.startswith(f"{PREFIX}_") for library in libraries)


def test_selection_and_grids() -> None:
    cake_jit = _registry()
    assert (
        cake_jit.select_record("kcp_affine_merge", ncols=128) == "kcp_affine_merge_c8"
    )
    assert (
        cake_jit.select_record("kcp_affine_merge", ncols=256) == "kcp_affine_merge_c16"
    )
    with pytest.raises(ValueError):
        cake_jit.select_record("kcp_affine_merge", ncols=64)
    assert cake_jit.select_record("kcp_fwdh_offset") == "kcp_fwdh_offset"
    assert cake_jit.launch_grid("kcp_fwdh_offset", n_walk=3, num_heads=48) == (
        144,
        1,
        1,
    )
    # The vh = 64 form of the forward state walk: two CTAs per (segment, head).
    assert (
        cake_jit.select_record("kcp_fwdh_offset", ctas_per_head=2)
        == "kcp_fwdh_offset_vh64"
    )
    assert (
        cake_jit.select_record("kcp_fwdh_tape", ctas_per_head=2) == "kcp_fwdh_tape_vh64"
    )
    assert cake_jit.launch_grid("kcp_fwdh_offset_vh64", n_walk=3, num_heads=48) == (
        288,
        1,
        1,
    )
    assert cake_jit.launch_grid(
        "kcp_fwdh_tape_vh64", n_walk=3, num_heads=48, ctas_per_head=2
    ) == (288, 1, 1)
    with pytest.raises(ValueError):
        cake_jit.launch_grid(
            "kcp_fwdh_offset_vh64", n_walk=3, num_heads=48, ctas_per_head=1
        )
    with pytest.raises(ValueError):
        cake_jit.select_record("kcp_fwdh_offset", ctas_per_head=3)
    # The adjoint walk has the same two forms: one CTA per (segment, head) or two half-width CTAs.
    assert cake_jit.select_record("kcp_dhu_offset") == "kcp_dhu_offset"
    assert cake_jit.select_record("kcp_dhu_tape") == "kcp_dhu_tape"
    assert (
        cake_jit.select_record("kcp_dhu_offset", ctas_per_head=2)
        == "kcp_dhu_offset_vh64"
    )
    assert (
        cake_jit.select_record("kcp_dhu_tape", ctas_per_head=2) == "kcp_dhu_tape_vh64"
    )
    assert cake_jit.launch_grid("kcp_dhu_tape", n_walk=3, num_heads=48) == (144, 1, 1)
    assert cake_jit.launch_grid("kcp_dhu_tape_vh64", n_walk=3, num_heads=48) == (
        288,
        1,
        1,
    )
    with pytest.raises(ValueError):
        cake_jit.launch_grid("kcp_dhu_offset", n_walk=3, num_heads=48, ctas_per_head=2)
    # The three persistent programs take the ticket word as a buffer argument.
    for name in ("fwd_o", "bwd_prep", "bwd_dqkg"):
        assert ["buffer", "ticket"] in cake_jit.MODULES[name]["arg_plan"], name
        assert cake_jit.MODULES[name]["logical"]["ticket"] == "ticket"
    assert "ticket" in cake_jit.LOGICAL_TENSORS
    # The chunk-shaped programs: a 2-D grid, persistent grids capped by the SM count, one thread per item.
    assert cake_jit.launch_grid("fwd_prep", num_chunks=7, num_qk_heads=16) == (7, 16, 1)
    assert cake_jit.launch_grid("fwd_o", num_chunks=7, num_heads=48, sm_count=148) == (
        148,
        1,
        1,
    )
    assert cake_jit.launch_grid(
        "bwd_prep", num_chunks=2, num_heads=48, sm_count=148
    ) == (96, 1, 1)
    # #6's record carries a per-architecture CTA factor: two persistent CTAs per SM on
    # sm_100a, sm_103a and sm_107a, one when the launch names no architecture.
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        assert cake_jit.launch_grid(
            "bwd_prep", num_chunks=7, num_heads=48, sm_count=148, arch=arch
        ) == (296, 1, 1)
    assert cake_jit.launch_grid(
        "bwd_prep", num_chunks=7, num_heads=48, sm_count=148, arch=None
    ) == (148, 1, 1)
    assert cake_jit.launch_grid(
        "bwd_prep", num_chunks=2, num_heads=48, sm_count=148, arch="sm_103a"
    ) == (96, 1, 1)
    assert cake_jit.launch_grid("bwd_dqkg", num_chunks=7, num_qk_heads=16) == (
        112,
        1,
        1,
    )
    assert cake_jit.launch_grid("gate_epilogue", num_chunks=7, num_heads=48) == (
        3,
        1,
        1,
    )
    with pytest.raises(ValueError):
        cake_jit.launch_grid("fwd_prep", num_chunks=0, num_qk_heads=16)
    assert cake_jit.launch_grid("kcp_affine_merge_c16", n_chains=5, ncols=256) == (
        80,
        1,
        1,
    )
    assert cake_jit.launch_grid("kcp_affine_merge_c8", n_chains=2, ncols=128) == (
        32,
        1,
        1,
    )
    with pytest.raises(ValueError):
        cake_jit.launch_grid("kcp_fwdh_offset", n_walk=0, num_heads=48)


# ----------------------------------------------------------------------------------------------------------------
# The manifests


def test_manifest_skeleton_is_well_formed() -> None:
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == PACKAGE_IMPORT
    assert set(manifest) == {
        "schema",
        "package",
        "generator",
        "common",
        "arch_flags",
        "libraries",
    }
    assert set(manifest["arch_flags"]) == set(ARCHES)
    assert manifest["common"]["include_dirs"] == ["csrc"]
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()


def test_shipped_manifest_lists_the_programs_and_no_test_only_one() -> None:
    if not _manifest_rendered():
        pytest.skip("build.json carries no libraries yet (the package is not rendered)")
    cake_jit = _registry()
    backend = _backend()
    manifest = cake_build.manifest()
    libraries = cake_build.libraries_by_name()
    for name, record in cake_jit.MODULES.items():
        for arch in ARCHES:
            library = backend.library_name(name, arch)
            assert library == f"{record['library']}_{arch}"
            entry = libraries[library]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            for source in entry["sources"]:
                path = PACKAGE_DIR / source
                assert path.is_file(), source
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == entry["sha256"][source]
                ), source
    names = {entry["name"] for entry in manifest["libraries"]}
    test_only = {
        f"{record['library']}_{arch}"
        for record in cake_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    assert not names & test_only
    assert names == {
        f"{record['library']}_{arch}"
        for record in cake_jit.MODULES.values()
        for arch in ARCHES
    }


def test_test_manifest_is_the_test_only_programs() -> None:
    backend = _backend()
    cake_jit = _registry()
    path = PACKAGE_DIR / backend.TEST_MANIFEST
    if not path.is_file():
        pytest.skip(f"{backend.TEST_MANIFEST} is not shipped here")
    backend.test_manifest.cache_clear()
    document = backend.test_manifest()
    shipped = cake_build.manifest()
    assert document["schema"] == shipped["schema"] == "cake-build/1"
    assert document["package"] == shipped["package"] == PACKAGE_IMPORT
    assert document["generator"] == shipped["generator"]
    names = {entry["name"] for entry in document["libraries"]}
    assert names == {
        f"{record['library']}_{arch}"
        for record in cake_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    for entry in document["libraries"]:
        for source in entry["sources"]:
            source_path = PACKAGE_DIR / source
            assert source_path.is_file(), source
            assert (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
                == entry["sha256"][source]
            ), source
    for name in cake_jit.TEST_MODULES:
        for arch in ARCHES:
            assert backend.test_library_name(name, arch) in names
    with pytest.raises(RuntimeError):
        backend.test_library_name(TEST_RECORDS[0], "sm_90a")
    backend.test_manifest.cache_clear()


def test_aot_load_prefers_the_prebuilt_library() -> None:
    """A builder's ``aot/<name>.so`` is served before the JIT path (``cake_build.load_library``); without one the
    JIT spec is derived from the manifest alone.  Checked on the shipped manifest once it carries libraries."""
    if not _manifest_rendered():
        pytest.skip("build.json carries no libraries yet (the package is not rendered)")
    backend = _backend()
    cake_jit = _registry()
    name = next(iter(cake_jit.MODULES))
    for arch in ARCHES:
        library = backend.library_name(name, arch)
        assert (
            cake_build.prebuilt_path(library) == PACKAGE_DIR / "aot" / f"{library}.so"
        )
        assert (
            backend.is_prebuilt(name, arch)
            == cake_build.prebuilt_path(library).is_file()
        )
