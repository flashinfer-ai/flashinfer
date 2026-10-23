# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CPU tests of the Cake single-GPU norm package (``flashinfer.cake_muse_spark_norm_backend``): the registry
records and routes of its two operators as the exporter wrote them, the selection and launch rules over the literals,
the variant / route helpers of the host module, and ``build.json`` under FlashInfer's manifest checker."""

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_build
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(pkg.__file__).parent
OP = pkg.GATED_RESIDUAL_NEXT_RMSNORM
OP_A = "gated_residual_rmsnorm"
BLACKWELL = ("sm_100a", "sm_103a")
# The kernel parameters of each operator's programs, in binding order (the grid triple follows).
PARAMETERS = {
    OP: (
        "branch",
        "residual",
        "alpha",
        "beta",
        "next_weight",
        "post_weight",
        "residual_out",
        "ffn_input",
        "M",
        "post_eps",
        "next_eps",
        "post_gain_center",
        "next_gain_center",
        "alpha_value",
    ),
    OP_A: (
        "branch",
        "residual",
        "weight",
        "alpha",
        "beta",
        "out",
        "M",
        "eps",
        "gain_center",
    ),
}
# The variants with exported programs: three of op B's four forms (the fourth runs through the post_weight
# programs), both modes of op A in the plain and the PDL build.
SHIPPED_VARIANTS = {
    OP: ("alpha_tensor_pw", "alpha_scalar_nopw", "alpha_scalar_pw"),
    OP_A: ("mode0", "mode0_pdl", "mode1", "mode1_pdl"),
}
FAMILY_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")
NAME = re.compile(r"cake_muse_spark_norm_[0-9a-f]{20}")


def _variant(op: str, build: dict) -> str:
    if op == OP:
        return pkg.gated_residual_next_rmsnorm_variant(
            build["tensor_alpha"], build["post_weight"]
        )
    return f"mode{build['mode']}" + ("_pdl" if build["pdl"] else "")


def test_variant_and_route_helpers():
    assert pkg.gated_residual_next_rmsnorm_variant(True, True) == "alpha_tensor_pw"
    assert pkg.gated_residual_next_rmsnorm_variant(False, False) == "alpha_scalar_nopw"
    assert pkg.gated_residual_next_rmsnorm_variant(False, True) == "alpha_scalar_pw"
    assert pkg.gated_residual_next_rmsnorm_variant(True, False) == "alpha_tensor_nopw"
    for tensor_alpha, has_post_weight in ((True, True), (False, False), (False, True)):
        route = pkg.gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
        assert (
            route
            == f"{OP}_{pkg.gated_residual_next_rmsnorm_variant(tensor_alpha, has_post_weight)}"
        )
    assert pkg.gated_residual_next_rmsnorm_route(True, False) == f"{OP}_alpha_tensor_pw"
    assert loader.route_key(OP, "alpha_tensor_nopw") == f"{OP}_alpha_tensor_nopw"
    with pytest.raises(ValueError, match="unknown Cake norm operator"):
        loader.route_key("rmsnorm", "alpha_tensor_pw")
    with pytest.raises(ValueError, match="unknown variant"):
        loader.route_key(OP, "alpha_vector")
    assert loader.arch_for_capability((10, 0)) == "sm_100a"
    assert loader.arch_for_capability([10, 3]) == "sm_103a"
    assert loader.arch_for_capability((10, 7)) == "sm_107a"
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.arch_for_capability((10,)) is None
    assert tuple(loader.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")
    assert set(loader.ARCH_BY_CAPABILITY.values()) == set(loader.ARCHES)
    assert tuple(loader.OPS) == (OP, OP_A)
    assert loader.route_key(OP_A, "mode1_pdl") == f"{OP_A}_mode1_pdl"
    with pytest.raises(ValueError, match="unknown variant"):
        loader.route_key(OP_A, "mode2")
    for op in loader.OPS:
        assert set(SHIPPED_VARIANTS[op]) <= set(loader.VARIANTS[op])


def test_support_follows_the_device_architecture(monkeypatch):
    device = torch.device("cuda", 0)
    for capability, arch in (((10, 0), "sm_100a"), ((10, 3), "sm_103a")):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *_, c=capability: c
        )
        assert pkg._arch(device) == arch
        for tensor_alpha in (True, False):
            for has_post_weight in (True, False):
                assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
                    device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight
                )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    assert pkg._arch(device) == "sm_107a"
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(device)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert pkg._arch(device) is None
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(device)
    assert pkg._arch(torch.device("cpu")) is None
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(torch.device("cpu"))


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` libraries and nothing to compile: one entry per
    (library, architecture) with the kernel, binding and occupancy sources under ``csrc/``, their digests and the
    family's numerics flags; the registry itself records the host plan only."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) >= set(BLACKWELL)
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    referenced = set()
    logical_values = set()
    for name, record in loader.MODULES.items():
        assert NAME.fullmatch(name), name
        assert record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert record["occupancy_entry"] == "max_active_clusters"
        op = record["op"]
        assert op in loader.OPS, name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        arches = tuple(record["arches"])
        assert arches and set(arches) <= set(BLACKWELL), name
        build = record["build"]
        assert build["H"] == pkg.HIDDEN_SIZE
        variant = _variant(op, build)
        assert variant in SHIPPED_VARIANTS[op], name
        assert record["route"] == loader.route_key(op, variant)
        launch = record["launch"]
        assert tuple(launch["block"]) == (build["threads"], 1, 1)
        assert tuple(launch["cluster"]) == (build["cluster"], 1, 1)
        assert launch["cooperative"] is False
        assert launch["use_pdl"] is build["pdl"]
        if op == OP:
            assert build["pdl"] is False
        else:
            assert build["pdl"] is variant.endswith("_pdl")
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        rule = record["grid_rule"]
        assert rule["cluster"] == build["cluster"]
        assert rule["kind"] == ("persistent_rows" if build["persistent"] else "rows")
        assert rule["kind"] in loader.GRID_KINDS
        select = record["select_rule"]
        assert select["kind"] == loader.SELECT_KIND
        assert set(select["windows"]) == set(arches)
        parameters = [key for kind, key in record["arg_plan"] if kind != "grid"]
        assert tuple(parameters) == PARAMETERS[op]
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        assert record["logical"] == {
            key: ("rows" if key == "M" else key) for key in PARAMETERS[op]
        }
        logical_values |= set(record["logical"].values())
        for arch in arches:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            sources = {
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
                f"csrc/{name}_occupancy.cu",
            }
            assert set(entry["sources"]) == sources, library
            assert set(entry["sha256"]) == sources
            for source, digest in entry["sha256"].items():
                assert (
                    hashlib.sha256((PACKAGE_DIR / source).read_bytes()).hexdigest()
                    == digest
                ), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            flags = tuple(entry["nvcc_flags"])
            assert flags[: len(FAMILY_FLAGS)] == FAMILY_FLAGS, library
            # nvcc's default --fmad=true is left off the entries; nothing opts into fast math.
            assert flags == FAMILY_FLAGS, library
            assert entry["entry"] == "run" and entry["arch"] == arch
            assert entry["min_cuda"]
    assert referenced == set(libraries)
    assert logical_values == set(loader.LOGICAL_TENSORS) | set(loader.LOGICAL_SCALARS)
    assert {record["op"] for record in loader.MODULES.values()} == set(loader.OPS)


def test_routes_partition_the_row_counts():
    for arch, routes in loader.ROUTES.items():
        assert arch in BLACKWELL
        assert set(routes) == {
            loader.route_key(op, variant)
            for op in loader.OPS
            for variant in SHIPPED_VARIANTS[op]
        }
        for route, names in routes.items():
            assert len(names) == len(set(names))
            assert tuple(names) == loader.route_modules(arch, route)
            owners = {
                name
                for name, record in loader.MODULES.items()
                if record["route"] == route and arch in record["arches"]
            }
            assert set(names) == owners
            windows = sorted(
                (int(above), at_most, name)
                for name in names
                for above, at_most in loader.MODULES[name]["select_rule"]["windows"][
                    arch
                ]
            )
            assert windows[0][0] == 0
            assert [w[1] for w in windows[:-1]] == [w[0] for w in windows[1:]]
            assert windows[-1][1] is None and all(
                w[1] is not None for w in windows[:-1]
            )
            top = windows[-1][0] + 2
            for rows in range(1, top + 1):
                (owner,) = [
                    name
                    for above, at_most, name in windows
                    if above < rows <= (at_most or rows)
                ]
                assert loader.selected_module(arch, route, rows) == owner
            assert loader.selected_module(arch, route, 10**6) == windows[-1][2]
    assert not loader.route_available(
        "sm_107a", loader.route_key(OP, "alpha_tensor_pw")
    )
    assert (
        loader.route_modules("sm_107a", loader.route_key(OP, "alpha_tensor_pw")) == ()
    )
    with pytest.raises(NotImplementedError, match="no exported program"):
        loader.selected_module("sm_107a", loader.route_key(OP, "alpha_tensor_pw"), 1)
    with pytest.raises(ValueError, match="positive row count"):
        loader.selected_module("sm_100a", loader.route_key(OP, "alpha_tensor_pw"), 0)
    assert loader.admits(None, 1, "sm_100a")
    rule = {"kind": "rows_between", "windows": {"sm_100a": [[0, 16], [128, 512]]}}
    assert [
        r for r in (1, 16, 17, 128, 129, 512, 513) if loader.admits(rule, r, "sm_100a")
    ] == [1, 16, 129, 512]
    assert not loader.admits(rule, 1, "sm_103a")
    with pytest.raises(ValueError, match="unknown Cake norm select rule"):
        loader.admits({"kind": "always"}, 1, "sm_100a")


def test_gated_residual_rmsnorm_routes_window_the_row_count():
    """Op A's eight routes (two architectures x four variants) hold four programs each over the windows 1-16 /
    17-64 / 65-128 / 129-: the 65-128 window is the whole-row (one CTA per row, 512 threads) program and the
    programs of a variant differ only by their window."""
    for arch in BLACKWELL:
        for variant in SHIPPED_VARIANTS[OP_A]:
            route = loader.route_key(OP_A, variant)
            names = loader.route_modules(arch, route)
            assert len(names) == 4, (arch, route)
            edges = [
                loader.MODULES[name]["select_rule"]["windows"][arch] for name in names
            ]
            assert [tuple(window) for (window,) in edges] == [
                (0, 16),
                (16, 64),
                (64, 128),
                (128, None),
            ], (arch, route)
            for rows, position in (
                (1, 0),
                (16, 0),
                (17, 1),
                (64, 1),
                (65, 2),
                (96, 2),
                (128, 2),
                (129, 3),
                (4096, 3),
            ):
                assert loader.selected_module(arch, route, rows) == names[position]
            window = loader.MODULES[names[2]]
            assert window["grid_rule"] == {"cluster": 1, "kind": "rows", "row_ctas": 1}
            assert window["build"]["threads"] == 512 and window["build"]["split"] == 1
            assert tuple(window["launch"]["block"]) == (512, 1, 1)
            for name in names:
                build = loader.MODULES[name]["build"]
                assert build["mode"] == int(variant[4]) and build[
                    "pdl"
                ] is variant.endswith("_pdl")


def test_launch_grid_and_args_follow_the_record(monkeypatch):
    record = next(
        r
        for r in loader.MODULES.values()
        if r["grid_rule"]["kind"] == "persistent_rows"
    )
    cluster = record["grid_rule"]["cluster"]
    monkeypatch.setattr(loader, "device_arch", lambda index: record["arches"][0])
    for capacity, rows, expected in (
        (148, 7, 7),
        (148, 4096, 148),
        (0, 5, 1),
        (3, 3, 3),
    ):
        monkeypatch.setattr(loader, "cluster_capacity", lambda *a, c=capacity: c)
        assert loader.launch_grid(record, rows, device_index=0) == (
            expected * cluster,
            1,
            1,
        )
    plain = {"library": "x", "grid_rule": {"kind": "rows", "cluster": 4}}
    assert loader.launch_grid(plain, 10, device_index=0) == (40, 1, 1)
    with pytest.raises(ValueError, match="positive row count"):
        loader.launch_grid(plain, 0, device_index=0)
    with pytest.raises(ValueError, match="unknown Cake norm grid rule"):
        loader.launch_grid(
            {"library": "x", "grid_rule": {"kind": "tokens", "cluster": 1}},
            1,
            device_index=0,
        )
    values = {
        logical: object()
        for logical in (*loader.LOGICAL_TENSORS, *loader.LOGICAL_SCALARS)
    }
    args = loader.launch_args(record, values, (7, 1, 1))
    assert args[:-3] == tuple(
        values[record["logical"][key]] for key in PARAMETERS[record["op"]]
    )
    assert args[-3:] == (7, 1, 1)
    with pytest.raises(ValueError, match="needs logical argument 'rows'"):
        loader.launch_args(
            record, {k: v for k, v in values.items() if k != "rows"}, (1, 1, 1)
        )
    name = record["library"]
    assert (
        loader.library_name(name, record["arches"][0])
        == f"{name}_{record['arches'][0]}"
    )
    with pytest.raises(RuntimeError, match="not delivered for sm_107a"):
        loader.library_name(name, "sm_107a")


def _checker():
    path = ROOT / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


def _package(checker):
    return checker.Package(
        name=pkg.__name__,
        path=str(PACKAGE_DIR.relative_to(ROOT)),
        loader="cake_jit.py",
        refs=checker.refs_generic,
    )


def test_manifest_passes_the_flashinfer_checker():
    checker = _checker()
    package = _package(checker)
    manifest = cake_build.manifest()
    findings = [
        *checker.check_manifest(package, PACKAGE_DIR, manifest),
        *checker.check_cuda_toolkit_floor(package, manifest),
        *checker.check_references(package, PACKAGE_DIR, manifest),
    ]
    requirements = ROOT / "requirements.txt"
    if requirements.is_file():
        findings.extend(checker.check_tvm_ffi_pin(package, manifest, requirements))
    assert findings == []


@pytest.mark.xfail(
    reason="pyproject.toml package-data entry for the package is a shared-file edit outside this branch",
    strict=False,
)
def test_package_data_lists_the_package():
    checker = _checker()
    table = checker.package_data(ROOT / "pyproject.toml")
    assert (
        checker.check_package_data(_package(checker), cake_build.manifest(), table)
        == []
    )
