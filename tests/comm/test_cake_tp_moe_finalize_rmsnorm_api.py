# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CPU tests of the Cake TP MoE finalize package (``flashinfer.cake_tp_moe_finalize_rmsnorm_backend``): the registry
records and routes as the exporter wrote them, the selection, slot and launch rules over the literals, the workspace
layout and binding of the host, the ``backend`` keyword of the public entry points, and ``build.json`` under
FlashInfer's manifest checker."""

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.cake_tp_moe_finalize_rmsnorm_backend as pkg
from flashinfer.cake_tp_moe_finalize_rmsnorm_backend import cake_backend, cake_build
from flashinfer.cake_tp_moe_finalize_rmsnorm_backend import cake_jit as loader

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(pkg.__file__).parent
BLACKWELL = ("sm_100a", "sm_103a")
R200 = ("sm_107a",)
EVERY = BLACKWELL + R200
KINDS = ("onehop", "rsag", "tokown", "token1", "eo")
DATA = (
    "gemm2",
    "row_stride",
    "rows",
    "idx",
    "expert_weights",
    "weights_fp32",
    "norm_weight",
    "output",
    "n_tok",
)
TAIL = ("eps", "gain_center", "mode", "rank", "peer_table")
KIND_PARAMETERS = {
    "onehop": (*DATA, "slots", *TAIL, "gen"),
    "rsag": (*DATA, "slots", *TAIL, "gen"),
    "tokown": (*DATA, "max_ctas", *TAIL, "c_mcast", "ctl"),
    "token1": (*DATA, "slots", *TAIL, "gen", "part", "ctr", "r_mcast"),
    "eo": (*DATA, "slots", *TAIL, "gen", "b_mcast"),
}
RENAMES = {
    "gemm2": "gemm2_permuted",
    "idx": "expanded_idx_to_permuted_idx",
    "expert_weights": "expert_weights_words",
    "mode": "weight_before_norm",
    "b_mcast": "partial_multicast",
    "r_mcast": "rows_multicast",
    "c_mcast": "bcast_multicast",
}
FAMILY_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")
NAME = re.compile(r"cake_tp_moe_finalize_rmsnorm_[0-9a-f]{20}")
#: Slot counts per program kind and route: the largest bounded window edge of the kind's records.
KIND_SLOTS = {
    ("sm_100a", "tp2"): {"eo": 32, "rsag": 33, "token1": 148},
    ("sm_100a", "tp8"): {"eo": 7, "token1": 144},
    ("sm_103a", "tp2"): {"onehop": 17, "token1": 148, "rsag": 33, "eo": 64},
    ("sm_103a", "tp4"): {"onehop": 8, "token1": 152},
    ("sm_107a", "tp4"): {"onehop": 8, "eo": 24, "token1": 212},
}


def test_route_and_architecture_helpers():
    assert tuple(loader.WORLDS) == (2, 4, 8) and tuple(loader.ARCHES) == EVERY
    assert [loader.route_key(world) for world in loader.WORLDS] == ["tp2", "tp4", "tp8"]
    with pytest.raises(ValueError, match="world size 3"):
        loader.route_key(3)
    assert loader.arch_for_capability((10, 0)) == "sm_100a"
    assert loader.arch_for_capability([10, 3]) == "sm_103a"
    assert loader.arch_for_capability((10, 7)) == "sm_107a"
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.arch_for_capability((10,)) is None
    assert set(loader.ARCH_BY_CAPABILITY.values()) == set(loader.ARCHES)
    assert pkg.route_key is loader.route_key and pkg.WORLDS is loader.WORLDS
    assert (pkg.HIDDEN, pkg.TOPK, pkg.TOKOWN_MAX_CTAS) == (4096, 8, 160)


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` libraries and nothing to compile: one entry per
    (library, architecture) with the kernel, binding and occupancy sources under ``csrc/``, their digests and the
    family's numerics flags; the registry itself records the host plan only."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) >= set(EVERY)
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    referenced = set()
    for name, record in loader.MODULES.items():
        assert NAME.fullmatch(name), name
        assert record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert record["occupancy_entry"] == "max_active_clusters"
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        kind, world = record["program"], record["world"]
        assert kind in KINDS and world in loader.WORLDS, name
        assert record["route"] == loader.route_key(world)
        arches = tuple(record["arches"])
        assert arches and set(arches) <= set(EVERY), name
        build = record["build"]
        split = int(build.get("split", 1))
        launch = record["launch"]
        block = tuple(launch["block"])
        assert block[0] > 0 and block[1:] == (1, 1), name
        # the token-owner cluster-pair build: one slot is a cluster of two CTAs
        pair = kind == "tokown" and build.get("cpair", False) is True
        expected_cluster = (
            (split, 1, 1)
            if kind in ("eo", "onehop", "token1")
            else ((2, 1, 1) if pair else (1, 1, 1))
        )
        assert tuple(launch["cluster"]) == expected_cluster, name
        if kind == "tokown":
            assert block[0] * expected_cluster[0] == 512, name
        assert launch["cooperative"] is (kind == "tokown")
        assert launch["use_pdl"] is False and build.get("pdl", False) is False
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        rule = record["grid_rule"]
        assert rule["kind"] == kind and rule["world"] == world
        if kind in ("eo", "onehop", "token1"):
            assert rule["split"] == split
        select = record["select_rule"]
        assert select["kind"] == loader.SELECT_KIND
        assert set(select["windows"]) == set(arches)
        workspace = record["workspace_rule"]
        assert (
            workspace["kind"] == "lamport"
            and workspace["program"] == kind
            and workspace["world"] == world
        )
        assert workspace["slot_parameter"] == (
            "max_ctas" if kind == "tokown" else "slots"
        )
        assert {buffer["placement"] for buffer in workspace["buffers"]} <= {
            "symmetric",
            "local",
        }
        for table in workspace["tables"]:
            assert table["parameter"] == "peer_table"
            assert set(table["entries"]) <= {
                buffer["name"] for buffer in workspace["buffers"]
            }
        parameters = [
            key for plan_kind, key in record["arg_plan"] if plan_kind != "grid"
        ]
        assert set(parameters) == set(KIND_PARAMETERS[kind]), name
        assert [
            key for plan_kind, key in record["arg_plan"] if plan_kind == "grid"
        ] == ["grid_x", "grid_y", "grid_z"]
        assert record["arg_plan"][-3][0] == "grid"
        assert record["logical"] == {
            key: RENAMES.get(key, key) for key in KIND_PARAMETERS[kind]
        }
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
            assert tuple(entry["nvcc_flags"]) == FAMILY_FLAGS, library
            assert entry["entry"] == "run" and entry["arch"] == arch
            assert entry["min_cuda"]
    assert referenced == set(libraries)


def test_routes_partition_the_token_counts():
    seen = set()
    for arch, routes in loader.ROUTES.items():
        assert arch in EVERY
        for route, names in routes.items():
            seen.add((arch, route))
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
            assert loader.MODULES[windows[-1][2]]["program"] == "tokown"
            top = windows[-1][0] + 2
            for n_tok in range(1, top + 1):
                (owner,) = [
                    name
                    for above, at_most, name in windows
                    if above < n_tok <= (at_most or n_tok)
                ]
                assert loader.selected_module(arch, route, n_tok) == owner
            assert loader.selected_module(arch, route, 10**6) == windows[-1][2]
    assert seen == {
        ("sm_100a", "tp2"),
        ("sm_100a", "tp8"),
        ("sm_103a", "tp2"),
        ("sm_103a", "tp4"),
        ("sm_107a", "tp4"),
    }
    assert not loader.route_available("sm_100a", "tp4") and not loader.route_available(
        "sm_103a", "tp8"
    )
    assert not loader.route_available("sm_107a", "tp2") and not loader.route_available(
        "sm_107a", "tp8"
    )
    assert loader.route_modules("sm_100a", "tp4") == ()
    with pytest.raises(NotImplementedError, match="no exported program"):
        loader.selected_module("sm_100a", "tp4", 1)
    with pytest.raises(ValueError, match="positive token count"):
        loader.selected_module("sm_100a", "tp2", 0)
    assert loader.admits(None, 1, "sm_100a")
    rule = {"kind": "rows_between", "windows": {"sm_100a": [[0, 16], [128, 512]]}}
    assert [
        r for r in (1, 16, 17, 128, 129, 512, 513) if loader.admits(rule, r, "sm_100a")
    ] == [1, 16, 129, 512]
    assert not loader.admits(rule, 1, "sm_103a")
    with pytest.raises(ValueError, match="unknown Cake TP MoE finalize select rule"):
        loader.admits({"kind": "always"}, 1, "sm_100a")


def test_kind_slots_are_the_largest_window_edges():
    for (arch, route), expected in KIND_SLOTS.items():
        assert pkg.kind_slots(arch, route) == expected, (arch, route)
    assert pkg.kind_slots("sm_100a", "tp4") == {}
    assert pkg.kind_slots("sm_107a", "tp2") == {}


def test_launch_grid_follows_the_record_and_the_agreed_capacity(monkeypatch):
    def unasked(*args):
        raise AssertionError(
            "the device's own co-residency is not queried under an agreed capacity"
        )

    monkeypatch.setattr(loader, "cluster_capacity", unasked)
    monkeypatch.setattr(loader, "device_arch", lambda index: "sm_100a")
    by_kind = {
        record["program"]: record
        for record in loader.MODULES.values()
        if record["world"] == 2
    }
    eo = by_kind["eo"]
    split = int(eo["grid_rule"]["split"])
    assert loader.launch_grid(
        eo, 10, device_index=0, values={"slots": 32}, capacity=20
    ) == (split * 10, 1, 1)
    with pytest.raises(ValueError, match="exceed the 32 slots"):
        loader.launch_grid(eo, 33, device_index=0, values={"slots": 32}, capacity=200)
    with pytest.raises(RuntimeError, match="exceed the device's co-residency of 5"):
        loader.launch_grid(eo, 10, device_index=0, values={"slots": 32}, capacity=5)
    assert loader.launch_grid(
        by_kind["rsag"], 7, device_index=0, values={"slots": 33}, capacity=7
    ) == (7, 1, 1)
    token1 = by_kind["token1"]
    assert loader.launch_grid(
        token1, 9, device_index=0, values={"slots": 148}, capacity=10
    ) == (
        int(token1["grid_rule"]["split"]),
        5,
        2,
    )
    with pytest.raises(RuntimeError, match="exceed the device's co-residency of 9"):
        loader.launch_grid(token1, 9, device_index=0, values={"slots": 148}, capacity=9)
    tokown = by_kind["tokown"]
    for n_tok, capacity in ((1, 148), (149, 148), (1025, 152), (8193, 160)):
        assert loader.launch_grid(
            tokown, n_tok, device_index=0, values={"max_ctas": 160}, capacity=capacity
        ) == (
            loader.tokown_grid(n_tok, min(capacity, 160), 2),
            1,
            1,
        )
    # a cluster-pair record's grid counts slots, held as for one CTA per slot (its module launches two CTAs per slot)
    pairs = [
        record
        for record in loader.MODULES.values()
        if record["program"] == "tokown" and record["launch"]["cluster"][0] == 2
    ]
    assert sorted((record["world"], tuple(record["arches"])) for record in pairs) == [
        (4, ("sm_103a",)),
        (8, ("sm_100a",)),
    ]
    for record in pairs:
        world = int(record["world"])
        for n_tok, capacity in ((145, 148), (153, 74), (288, 148), (224, 200)):
            assert loader.launch_grid(
                record,
                n_tok,
                device_index=0,
                values={"max_ctas": 160},
                capacity=capacity,
            ) == (loader.tokown_grid(n_tok, min(capacity, 160), world), 1, 1)
    with pytest.raises(ValueError, match="positive token count"):
        loader.launch_grid(eo, 0, device_index=0, capacity=1)
    monkeypatch.setattr(loader, "cluster_capacity", lambda *args: 12)
    assert loader.launch_grid(eo, 12, device_index=0) == (split * 12, 1, 1)
    values = {logical: object() for logical in eo["logical"].values()}
    args = loader.launch_args(eo, values, (7, 1, 1))
    assert args[:-3] == tuple(
        values[eo["logical"][key]] for kind, key in eo["arg_plan"] if kind != "grid"
    )
    assert args[-3:] == (7, 1, 1)
    with pytest.raises(ValueError, match="needs logical argument 'rows'"):
        loader.launch_args(
            eo, {k: v for k, v in values.items() if k != "rows"}, (1, 1, 1)
        )
    with pytest.raises(RuntimeError, match="not delivered for sm_107a"):
        loader.library_name(eo["library"], "sm_107a")


def test_load_launches_a_cluster_record_at_its_cluster_per_slot(monkeypatch):
    """``load`` gives a cluster-pair record's module behind its slot grid: the entry launches ``launch.cluster[0]``
    CTAs per slot of ``launch_grid`` and every other attribute is the library's; every other record's module is the
    library itself."""
    calls = []

    class Library:
        def __getattr__(self, attr):
            return lambda *args: calls.append((attr, args))

    library = Library()
    monkeypatch.setattr(cake_build, "load_library", lambda name: library)
    pairs = 0
    for name, record in loader.MODULES.items():
        module = loader.load.__wrapped__(name, record["arches"][0])
        ctas = record["launch"]["cluster"][0]
        if record["program"] != "tokown" or ctas == 1:
            assert module is library, name
            continue
        pairs += 1
        assert ctas == 2, name
        grid = (loader.tokown_grid(145, 148, int(record["world"])), 1, 1)
        axes = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
        args = tuple(
            axes[key] if kind == "grid" else f"{kind}:{key}"
            for kind, key in record["arg_plan"]
        )
        getattr(module, record["ffi_entry"])(*args)
        launched = tuple(
            ctas * value if (kind, key) == ("grid", "grid_x") else value
            for (kind, key), value in zip(record["arg_plan"], args, strict=True)
        )
        assert calls[-1] == (record["ffi_entry"], launched), name
        assert launched.count(2 * grid[0]) == 1, name
        getattr(module, record["occupancy_entry"])(0, 256, 1, 1, 2, 1, 1, 1024)
        assert calls[-1] == (record["occupancy_entry"], (0, 256, 1, 1, 2, 1, 1, 1024))
    assert pairs == 2


def test_selection_table_picks_the_record_of_the_route_scan():
    """``selected_module`` reads the route's selection table; for every token count it picks the first record of
    the route whose window admits it (the record the scan over the route picks)."""
    for arch, routes in loader.ROUTES.items():
        for route, names in routes.items():
            assert loader._SELECTION[(arch, route)] is not None, (arch, route)
            edges = {
                int(edge)
                for name in names
                for window in loader.MODULES[name]["select_rule"]["windows"].get(
                    arch, ()
                )
                for edge in window
                if edge is not None
            }
            counts = set(range(1, 2049)) | {
                edge + step for edge in edges for step in (-1, 0, 1, 2)
            }
            for n_tok in sorted(n for n in counts | {65536, 10**6} if n > 0):
                scan = next(
                    name
                    for name in names
                    if loader.admits(loader.MODULES[name]["select_rule"], n_tok, arch)
                )
                assert loader.selected_module(arch, route, n_tok) == scan, (
                    arch,
                    route,
                    n_tok,
                )


def test_tokown_slots_are_agreed_over_the_one_cta_records():
    """The slots of the persistent program: min(TOKOWN_MAX_CTAS, capacities of its one-CTA-per-slot records); a
    cluster-pair record is held to the slots and to its own cluster capacity and never lowers the others'."""
    for arch, routes in loader.ROUTES.items():
        for route, names in routes.items():
            tokown = [n for n in names if loader.MODULES[n]["program"] == "tokown"]
            single = [
                n for n in tokown if loader.MODULES[n]["launch"]["cluster"][0] == 1
            ]
            pair = [n for n in tokown if n not in single]
            assert single, (arch, route)
            assert len(pair) == (
                1 if (arch, route) in (("sm_100a", "tp8"), ("sm_103a", "tp4")) else 0
            )
            for one, two in (
                (148, 148),
                (148, 150),
                (148, 74),
                (152, 76),
                (200, 300),
                (200, 90),
            ):
                capacity = {n: (two if n in pair else one) for n in tokown}
                slots, agreed = cake_backend.tokown_slots(capacity)
                assert slots == min(160, one)
                assert {n: agreed[n] for n in single} == dict.fromkeys(single, slots)
                assert {n: agreed[n] for n in pair} == dict.fromkeys(
                    pair, min(slots, two)
                )
                if not pair or two >= one:
                    # the rule before the cluster records: every record at min(160, every capacity)
                    assert agreed == dict.fromkeys(tokown, min(160, *capacity.values()))
            # one-CTA records of unequal capacities: the smallest
            capacity = {n: 150 + i for i, n in enumerate(tokown)}
            slots, agreed = cake_backend.tokown_slots(capacity)
            assert slots == min(160, *(capacity[n] for n in single))


def test_workspace_layout_fill_and_binding_follow_the_rules():
    arch, route = "sm_100a", "tp2"
    names = list(loader.route_modules(arch, route))
    slots = {**pkg.kind_slots(arch, route), "tokown": 148}
    kinds, total = cake_backend.CakeTpMoeFinalize._layout(names, slots)
    assert set(kinds) == {"eo", "rsag", "token1", "tokown"}
    assert total % cake_backend.ALIGNMENT == 0
    cpu = torch.device("cpu")
    buffer = torch.zeros(total, dtype=torch.uint8)
    cake_backend.CakeTpMoeFinalize._fill(buffer, kinds, cpu)
    peers = [1 << 40, 2 << 40]
    cake_backend.CakeTpMoeFinalize._bind(
        kinds, peers=peers, multicast=3 << 40, rank=1, device=cpu
    )
    end = 0
    for space in kinds.values():
        rule = space.rule
        assert space.slots == slots[space.kind]
        logical = space.logical
        values = space.values
        assert (
            values[logical["rank"]] == 1
            and values[logical[rule["slot_parameter"]]] == space.slots
        )
        expected_table: list[int] = []
        for declared in rule["buffers"]:
            name, nbytes = (
                declared["name"],
                int(declared["bytes_per_slot"]) * space.slots,
            )
            if declared["placement"] == "symmetric":
                offset, size = space.regions[name]
                assert (
                    size == nbytes
                    and offset % cake_backend.ALIGNMENT == 0
                    and offset >= end
                )
                end = offset + size
                words = buffer[offset : offset + size].view(torch.int32)
                assert torch.all(
                    words == cake_backend._signed32(int(declared["fill"] or 0))
                ), name
                if declared["multicast"] is not None:
                    assert values[logical[declared["multicast"]]] == (3 << 40) + offset
            else:
                local = space.locals[name]
                assert (
                    local.numel() * local.element_size() == nbytes
                    and local.device == cpu
                )
                assert local.dtype == cake_backend._VIEW_DTYPES[declared["view"]]
                assert values[logical[declared["view"]]] is local
                assert (
                    not local.view(torch.int32).any() if not declared["fill"] else True
                )
        for table in rule["tables"]:
            for entry in table["entries"]:
                if entry in space.regions:
                    expected_table.extend(
                        peer + space.regions[entry][0] for peer in peers
                    )
                else:
                    expected_table.append(space.locals[entry].data_ptr())
            assert values[logical[table["parameter"]]].tolist() == expected_table
            assert values[logical[table["parameter"]]].dtype == torch.int64
        assert set(values) == {
            logical[key]
            for key in KIND_PARAMETERS[space.kind]
            if key not in DATA and key not in ("eps", "gain_center", "mode")
        }
    assert end <= total
    assert (
        cake_backend._signed32(0x80008000) == -2147450880
        and cake_backend._signed32(7) == 7
    )


def test_backend_keyword_is_validated_before_any_device_work():
    pytest.importorskip("cutlass")
    module = pytest.importorskip("flashinfer.comm.tp_moe_finalize_rmsnorm")
    from flashinfer.comm import (
        init_tp_moe_finalize_rmsnorm,
        supports_tp_moe_finalize_rmsnorm,
        tp_moe_finalize_rmsnorm,
    )

    rows = torch.zeros(1, 4096, dtype=torch.bfloat16)
    weights = torch.zeros(1, 8, dtype=torch.bfloat16)
    indices = torch.zeros(1, 8, dtype=torch.int32)
    norm = torch.zeros(4096, dtype=torch.bfloat16)
    out = torch.zeros(1, 4096, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="backend must be one of"):
        init_tp_moe_finalize_rmsnorm(backend="x")
    with pytest.raises(ValueError, match="backend must be one of"):
        supports_tp_moe_finalize_rmsnorm(rows, weights, indices, norm, out, backend="x")
    with pytest.raises(ValueError, match="backend must be one of"):
        tp_moe_finalize_rmsnorm(
            rows,
            weights,
            indices,
            norm,
            out,
            eps=1e-8,
            gain_center=0.0,
            weight_before_norm=False,
            backend="x",
        )
    if module._WS.initialized:
        pytest.skip("the TP MoE finalize workspace is initialized in this process")
    for backend in ("auto", "cake", "cute"):
        assert not supports_tp_moe_finalize_rmsnorm(
            rows, weights, indices, norm, out, backend=backend
        )
    with pytest.raises(
        RuntimeError, match="Cake TP MoE finalize backend is not prepared"
    ):
        tp_moe_finalize_rmsnorm(
            rows,
            weights,
            indices,
            norm,
            out,
            eps=1e-8,
            gain_center=0.0,
            weight_before_norm=False,
            backend="cake",
        )


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


def test_package_data_lists_the_package():
    checker = _checker()
    table = checker.package_data(ROOT / "pyproject.toml")
    assert (
        checker.check_package_data(_package(checker), cake_build.manifest(), table)
        == []
    )
