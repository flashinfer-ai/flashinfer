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

from __future__ import annotations

import hashlib
import json

import pytest

from flashinfer.jit.generated_program_pack import (
    PromotionPackError,
    pack_public_fragment_promotions,
    pack_public_promotions,
)
from flashinfer.jit.generated_program_promotion import (
    import_promotion,
    load_manifest,
)


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _write_public_input(root, architecture, mode="cubin"):
    root.mkdir()
    artifacts = []
    for artifact_id, relative, payload in (
        ("dispatcher", "runtime/dispatcher.py", b"def select(): pass\n"),
        ("module-a", "runtime/module-a.cubin", architecture.encode()),
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        artifacts.append(
            {
                "executable": False,
                "id": artifact_id,
                "kind": "runtime",
                "path": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        )
    inventory = {
        "architecture": architecture,
        "mode": mode,
        "modules": [{"id": "module-a"}],
        "routes": [{"id": "route-a", "module_ids": ["module-a"]}],
        "seeds": [{"id": "seed-a"}],
    }
    denominator = hashlib.sha256(b"contract").hexdigest()
    receipt = {
        "architecture": architecture,
        "artifacts": artifacts,
        "contracts": {
            "correctness": {"denominator_sha256": denominator},
            "performance": {"denominator_sha256": denominator},
        },
        "kind": "generated_program_public_promotion_receipt",
        "mode": mode,
        "name": "example-program",
        "route_count": 1,
        "route_denominator_sha256": hashlib.sha256(b"routes").hexdigest(),
        "runtime_inventory": inventory,
        "runtime_inventory_identity": "sha256:"
        + hashlib.sha256(_canonical(inventory)).hexdigest(),
        "schema_version": 1,
    }
    (root / "promotion-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return receipt


_RUNTIME_CONTRACT = {
    "operation": "generated-prefill",
    "targets": ["sm100a", "sm103a"],
}


def _write_artifact(
    root,
    artifact_id,
    kind,
    relative,
    payload,
    *,
    executable=False,
):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.chmod(0o755 if executable else 0o644)
    digest = hashlib.sha256(payload).hexdigest()
    artifact = {
        "executable": executable,
        "kind": kind,
        "path": relative,
        "sha256": digest,
        "size_bytes": len(payload),
    }
    reference = {
        "artifact_id": artifact_id,
        "kind": kind,
        "path": relative,
        "sha256": digest,
        "size_bytes": len(payload),
    }
    return artifact, reference


def _evidence_reference(artifact_id, kind, relative, payload):
    return {
        "artifact_id": artifact_id,
        "kind": kind,
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _seal_fragment_identities(fragment):
    fragment["route_denominator_sha256"] = hashlib.sha256(
        _canonical(fragment["routes"])
    ).hexdigest()
    fragment["dispatcher_seed_identity"] = (
        "sha256:"
        + hashlib.sha256(
            _canonical(
                {
                    "dispatcher": fragment["dispatcher"],
                    "routes": fragment["routes"],
                    "seeds": fragment["seeds"],
                }
            )
        ).hexdigest()
    )


def _write_fragment_input(root, target_name, mode):
    root.mkdir()
    architecture = "sm_" + target_name.removeprefix("sm")
    mode_root = f"generated/{target_name}/{mode}"
    artifacts = []

    dispatcher_artifact, dispatcher = _write_artifact(
        root,
        f"dispatcher-{target_name}-{mode}",
        "python_source",
        f"{mode_root}/dispatcher.py",
        b"def bind_loaded_modules(modules): return modules\n",
    )
    artifacts.append(dispatcher_artifact)
    package_artifact, package_library = _write_artifact(
        root,
        f"package-{target_name}-{mode}",
        "shared_library",
        f"{mode_root}/package/family.so",
        b"shared package " + architecture.encode(),
    )
    artifacts.append(package_artifact)
    recipe = None
    if mode == "cuda":
        recipe_artifact, recipe = _write_artifact(
            root,
            f"recipe-{target_name}-{mode}",
            "build_recipe",
            f"{mode_root}/build.py",
            b"#!/usr/bin/env python3\n",
            executable=True,
        )
        artifacts.append(recipe_artifact)

    modules = []
    build_outputs = []
    for module_index in range(2):
        module_root = f"{mode_root}/modules/module-{module_index:03d}"
        host_payload = f"// generated host {module_index}\n".encode()
        host_artifact, host = _write_artifact(
            root,
            f"host-{target_name}-{mode}-{module_index}",
            "host_source",
            f"{module_root}/host.cpp",
            host_payload,
        )
        artifacts.append(host_artifact)
        shared_payload = f"shared module {architecture} {module_index}".encode()
        shared_artifact, shared = _write_artifact(
            root,
            f"shared-{target_name}-{mode}-{module_index}",
            "shared_library",
            f"{module_root}/module.so",
            shared_payload,
        )
        artifacts.append(shared_artifact)
        source_payload = (
            f'extern "C" __global__ void generated_kernel_{module_index}() {{}}\n'
        ).encode()
        cubin_payload = f"cubin {architecture} {module_index}".encode()
        if mode == "cuda":
            source_artifact, source = _write_artifact(
                root,
                f"source-{target_name}-{mode}-{module_index}",
                "cuda_source",
                f"{module_root}/kernel.cu",
                source_payload,
            )
            artifacts.append(source_artifact)
            build_output = _evidence_reference(
                f"output-{target_name}-{mode}-{module_index}",
                "cubin",
                f"{module_root}/kernel.cubin",
                cubin_payload,
            )
            cubin = dict(build_output)
            build_outputs.append(build_output)
        else:
            source = None
            cubin_artifact, cubin = _write_artifact(
                root,
                f"cubin-{target_name}-{mode}-{module_index}",
                "cubin",
                f"{module_root}/kernel.cubin",
                cubin_payload,
            )
            artifacts.append(cubin_artifact)
            build_output = dict(cubin)

        modules.append(
            {
                "build_output": build_output,
                "build_receipt": {
                    "compile_options": ["--std=c++17"],
                    "cooperative": False,
                    "cubin_sha256": hashlib.sha256(cubin_payload).hexdigest(),
                    "cubin_size_bytes": len(cubin_payload),
                    "cuda_source_sha256": hashlib.sha256(source_payload).hexdigest(),
                    "cuda_source_size_bytes": len(source_payload),
                    "host_source_sha256": hashlib.sha256(host_payload).hexdigest(),
                    "host_source_size_bytes": len(host_payload),
                    "tma_abi": "pointer",
                    "use_pdl": False,
                },
                "build_recipe": recipe,
                "cubin": cubin,
                "cuda_source": source,
                "entry_point": "run",
                "host_source": host,
                "id": f"module-{target_name}-{mode}-{module_index}",
                "kernel_name": f"generated_kernel_{module_index}",
                "module_ident": f"generated_module_{module_index}",
                "shared_library": shared,
            }
        )

    build = (
        {"kind": "nvrtc", "outputs": build_outputs, "recipe": recipe}
        if mode == "cuda"
        else {"kind": "prebuilt", "outputs": [], "recipe": None}
    )

    contract_artifact, _contract_reference = _write_artifact(
        root,
        f"contract-{target_name}-{mode}",
        "data",
        "contracts/correctness.json",
        b'{"rows":48}\n',
    )
    artifacts.append(contract_artifact)
    module_ids = [module["id"] for module in modules]
    seeds = [{"id": "seed-shared-policy"}]
    selector_arguments = [
        "fixed_layout",
        "gpu_arch",
        "num_heads",
        "sequence_lengths",
        "sm_count",
        "store_final_state",
        "use_initial_state",
    ]
    routes = []
    for index in range(48):
        if index < 32:
            route_module_ids = [module_ids[0]]
        elif index < 41:
            route_module_ids = module_ids
        else:
            route_module_ids = [
                module_ids[0],
                module_ids[1],
                module_ids[0],
                module_ids[1],
            ]
        host_roles = []
        markers = []
        host_count = 0
        segments = [
            {"activity_count": 0, "fixed_markers": []}
            for _ in range(len(route_module_ids) + 1)
        ]
        if index < 9:
            host_roles = ["beta_tma_refresh"]
            markers = ["direct_copy_kernel_cuda"]
            host_count = 1
            segments[0] = {
                "activity_count": 1,
                "fixed_markers": ["direct_copy_kernel_cuda"],
            }
        elif 41 <= index < 45:
            host_roles = [
                "beta_tma_refresh",
                "beta_tma_refresh",
                "beta_tma_refresh",
                "affine_torch_epilogue",
            ]
            markers = ["direct_copy_kernel_cuda"] * 3
            host_count = 6
            for segment_index in (0, 1, 3):
                segments[segment_index] = {
                    "activity_count": 1,
                    "fixed_markers": ["direct_copy_kernel_cuda"],
                }
            segments[4] = {"activity_count": 3, "fixed_markers": []}
        elif index >= 45:
            host_roles = ["affine_torch_epilogue"]
            host_count = 3
            segments[4] = {"activity_count": 3, "fixed_markers": []}
        kernel_names = [
            modules[module_ids.index(module_id)]["kernel_name"]
            for module_id in route_module_ids
        ]
        routes.append(
            {
                "id": f"route-{index:03d}",
                "module_ids": route_module_ids,
                "public_activity_contract": {
                    "device_kernel_names": kernel_names,
                    "expected_activity_segments": segments,
                    "expected_fixed_host_activity_markers": markers,
                    "expected_host_activity_count": host_count,
                    "host_roles": host_roles,
                },
                "route": f"variant-{index:03d}",
                "route_index": index,
                "seed_id": seeds[0]["id"],
                "selector_facts": {
                    "fixed_layout": True,
                    "gpu_arch": architecture,
                    "num_heads": 8,
                    "sequence_lengths": [index + 1],
                    "sm_count": 100 if target_name == "sm100a" else 103,
                    "store_final_state": True,
                    "use_initial_state": True,
                },
            }
        )
    fragment = {
        "architecture": architecture,
        "build": build,
        "contract": dict(_RUNTIME_CONTRACT),
        "dispatcher": dispatcher,
        "dispatcher_seed_identity": "",
        "kind": "flashinfer.generated_program_pack.fragment",
        "mode": mode,
        "modules": modules,
        "pack_kind": "flashinfer.generated_program_pack",
        "package_shared_library": package_library,
        "route_denominator_sha256": "",
        "routes": routes,
        "schema_version": 1,
        "seeds": seeds,
        "selector": {
            "arguments": selector_arguments,
            "kind": "exact_selector_facts",
            "route_count": 48,
        },
        "target": target_name,
    }
    _seal_fragment_identities(fragment)
    fragment_artifact, _fragment_reference = _write_artifact(
        root,
        f"fragment-{target_name}-{mode}",
        "data",
        f"{mode_root}/fragment.json",
        _canonical(fragment) + b"\n",
    )
    artifacts.append(fragment_artifact)
    denominator = hashlib.sha256(b"fixed contract denominator").hexdigest()
    receipt = {
        "architecture": architecture,
        "artifacts": artifacts,
        "contracts": {
            "correctness": {"denominator_sha256": denominator, "passed": True},
            "performance": {"denominator_sha256": denominator, "passed": True},
        },
        "kind": "generated_program_public_promotion_receipt",
        "mode": mode,
        "name": "example-program",
        "route_count": 48,
        "route_denominator_sha256": hashlib.sha256(_canonical(routes)).hexdigest(),
        "schema_version": 1,
    }
    (root / "promotion-receipt.json").write_bytes(_canonical(receipt) + b"\n")
    return receipt


def _rewrite_fragment(root, mutation):
    receipt_path = root / "promotion-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    artifact = next(
        item for item in receipt["artifacts"] if item["path"].endswith("fragment.json")
    )
    fragment_path = root / artifact["path"]
    fragment = json.loads(fragment_path.read_text())
    mutation(fragment)
    payload = _canonical(fragment) + b"\n"
    fragment_path.write_bytes(payload)
    artifact["sha256"] = hashlib.sha256(payload).hexdigest()
    artifact["size_bytes"] = len(payload)
    receipt_path.write_bytes(_canonical(receipt) + b"\n")


def test_pack_merges_targets_and_imports_exact_inventory(tmp_path):
    sm100a = tmp_path / "sm100a"
    sm103a = tmp_path / "sm103a"
    _write_public_input(sm100a, "sm_100a")
    _write_public_input(sm103a, "sm_103a")
    packed = tmp_path / "packed"

    runtime = pack_public_promotions(
        {"sm100a": sm100a, "sm103a": sm103a},
        mode="cubin",
        name="example-program",
        target=packed,
        runtime_manifest_destination="csrc/example/runtime.json",
    )

    assert [entry["target"] for entry in runtime["entries"]] == [
        "sm100a",
        "sm103a",
    ]
    manifest = load_manifest(packed / "promotion-manifest.json")
    checkout = tmp_path / "checkout"
    import_promotion(
        manifest,
        payload_root=packed / "payload",
        output_root=checkout,
        mode="cubin",
    )
    assert json.loads((checkout / "csrc/example/runtime.json").read_text()) == runtime
    assert (
        checkout
        / "csrc/generated_programs/example-program/sm100a/runtime/module-a.cubin"
    ).read_bytes() == b"sm_100a"


def test_pack_rejects_inventory_drift_and_wrong_selected_mode(tmp_path):
    source = tmp_path / "source"
    receipt = _write_public_input(source, "sm_100a")
    receipt["runtime_inventory"]["routes"].append({"id": "route-b"})
    (source / "promotion-receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(PromotionPackError, match="inventory identity"):
        pack_public_promotions(
            {"sm100a": source},
            mode="cubin",
            name="example-program",
            target=tmp_path / "pack-a",
            runtime_manifest_destination="csrc/example/runtime.json",
        )

    source = tmp_path / "source-cuda"
    _write_public_input(source, "sm_100a", mode="cuda")
    with pytest.raises(PromotionPackError, match="selected mode"):
        pack_public_promotions(
            {"sm100a": source},
            mode="cubin",
            name="example-program",
            target=tmp_path / "pack-b",
            runtime_manifest_destination="csrc/example/runtime.json",
        )


@pytest.mark.parametrize("mode", ["cubin", "cuda"])
def test_fragment_pack_builds_exact_two_target_runtime_closure(tmp_path, mode):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, mode)
        inputs[target_name] = source
    packed = tmp_path / f"packed-{mode}"

    runtime = pack_public_fragment_promotions(
        inputs,
        mode=mode,
        name="example-program",
        target=packed,
        runtime_manifest_destination="csrc/example/runtime.json",
        runtime_contract=_RUNTIME_CONTRACT,
        dispatcher_run_entrypoint="prepare_fwd",
        dispatcher_select_entrypoint="select_route",
    )

    assert [entry["target"] for entry in runtime["entries"]] == [
        "sm100a",
        "sm103a",
    ]
    for entry in runtime["entries"]:
        target_name = entry["target"]
        inventory = entry["runtime_inventory"]
        assert inventory["contract"] == _RUNTIME_CONTRACT
        assert entry["route_count"] == len(inventory["routes"]) == 48
        assert inventory["seeds"] == [{"id": "seed-shared-policy"}]
        assert {tuple(route["module_ids"]) for route in inventory["routes"]} == {
            (f"module-{target_name}-{mode}-0",),
            (
                f"module-{target_name}-{mode}-0",
                f"module-{target_name}-{mode}-1",
            ),
            (
                f"module-{target_name}-{mode}-0",
                f"module-{target_name}-{mode}-1",
                f"module-{target_name}-{mode}-0",
                f"module-{target_name}-{mode}-1",
            ),
        }
        assert {route["seed_id"] for route in inventory["routes"]} == {
            "seed-shared-policy"
        }
        assert (
            entry["route_denominator_sha256"]
            == hashlib.sha256(_canonical(inventory["routes"])).hexdigest()
        )
        dispatcher_seed = {
            "contract": _RUNTIME_CONTRACT,
            "dispatcher": inventory["dispatcher"],
            "routes": inventory["routes"],
            "seeds": inventory["seeds"],
        }
        assert (
            inventory["dispatcher_seed_identity"]
            == "sha256:" + hashlib.sha256(_canonical(dispatcher_seed)).hexdigest()
        )
        assert (
            entry["runtime_inventory_identity"]
            == "sha256:" + hashlib.sha256(_canonical(inventory)).hexdigest()
        )
        installed_ids = {artifact["id"] for artifact in entry["artifacts"]}
        if mode == "cubin":
            expected_ids = {f"dispatcher-{target_name}-{mode}"}
            for module_index in range(2):
                expected_ids.update(
                    {
                        f"cubin-{target_name}-{mode}-{module_index}",
                        f"shared-{target_name}-{mode}-{module_index}",
                    }
                )
            assert installed_ids == expected_ids
            for module in inventory["modules"]:
                assert module["build_output"] is None
                assert module["host"]["artifact_id"] not in installed_ids
        else:
            expected_ids = {
                f"dispatcher-{target_name}-{mode}",
                f"recipe-{target_name}-{mode}",
            }
            for module_index in range(2):
                expected_ids.update(
                    {
                        f"host-{target_name}-{mode}-{module_index}",
                        f"source-{target_name}-{mode}-{module_index}",
                    }
                )
            assert installed_ids == expected_ids
            output_paths = [
                module["build_output"]["path"] for module in inventory["modules"]
            ]
            assert len(set(output_paths)) == 2
            assert all(path.endswith("/kernel.cubin") for path in output_paths)
            assert {path.rsplit("/", 1)[-1] for path in output_paths} == {
                "kernel.cubin"
            }
            for module in inventory["modules"]:
                assert module["cubin"]["artifact_id"] not in installed_ids
                assert module["shared_library"]["artifact_id"] not in installed_ids

    manifest = load_manifest(packed / "promotion-manifest.json")
    checkout = tmp_path / f"checkout-{mode}"
    import_promotion(
        manifest,
        payload_root=packed / "payload",
        output_root=checkout,
        mode=mode,
    )
    assert json.loads((checkout / "csrc/example/runtime.json").read_text()) == runtime


@pytest.mark.parametrize("mode", ["cubin", "cuda"])
def test_fragment_pack_allows_target_specific_route_implementation(tmp_path, mode):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, mode)
        inputs[target_name] = source

    def mutate(fragment):
        target_seed_id = "seed-sm103a-policy"
        fragment["seeds"] = [{"id": target_seed_id}]
        modules = fragment["modules"]
        modules[0]["kernel_name"] = "sm103a_generated_kernel_0"
        module_names = {module["id"]: module["kernel_name"] for module in modules}
        for route in fragment["routes"]:
            route["route"] = f"sm103a_{route['route']}"
            route["seed_id"] = target_seed_id
            if len(route["module_ids"]) == 1:
                route["module_ids"] = [modules[1]["id"]]
            route["public_activity_contract"]["device_kernel_names"] = [
                module_names[module_id] for module_id in route["module_ids"]
            ]
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm103a"], mutate)
    runtime = pack_public_fragment_promotions(
        inputs,
        mode=mode,
        name="example-program",
        target=tmp_path / f"packed-target-specific-{mode}",
        runtime_manifest_destination="csrc/example/runtime.json",
        runtime_contract=_RUNTIME_CONTRACT,
        dispatcher_run_entrypoint="prepare_fwd",
        dispatcher_select_entrypoint="select_route",
    )

    sm100a, sm103a = [entry["runtime_inventory"] for entry in runtime["entries"]]
    assert sm100a["seeds"] == [{"id": "seed-shared-policy"}]
    assert sm103a["seeds"] == [{"id": "seed-sm103a-policy"}]
    assert {route["module_ids"][0] for route in sm100a["routes"][:32]} == {
        f"module-sm100a-{mode}-0"
    }
    assert {route["module_ids"][0] for route in sm103a["routes"][:32]} == {
        f"module-sm103a-{mode}-1"
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("kind", "fragment kind"),
        ("pack_kind", "fragment kind"),
        ("contract", "runtime contract"),
        ("target", "target identity"),
        ("selector", "selector denominator"),
        ("selector_arguments", "selector argument set"),
        ("artifact", "differs from its publicized artifact"),
        ("identity", "dispatcher/seed identity"),
        ("tma_abi", "tensor-map ABI"),
        ("closure", "cubin mode has CUDA build fields"),
    ],
)
def test_fragment_pack_rejects_mutated_identity_and_closure(
    tmp_path, mutation, message
):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        if mutation == "kind":
            fragment["kind"] = "unknown.fragment"
        elif mutation == "pack_kind":
            fragment["pack_kind"] = "unknown.pack"
        elif mutation == "contract":
            fragment["contract"]["operation"] = "different-operation"
        elif mutation == "target":
            fragment["target"] = "sm999a"
        elif mutation == "selector":
            fragment["selector"]["route_count"] = 47
        elif mutation == "selector_arguments":
            fragment["selector"]["arguments"].remove("fixed_layout")
        elif mutation == "artifact":
            fragment["dispatcher"]["sha256"] = "f" * 64
        elif mutation == "identity":
            fragment["dispatcher_seed_identity"] = "sha256:" + "f" * 64
        elif mutation == "tma_abi":
            fragment["modules"][0]["build_receipt"]["tma_abi"] = "value"
        else:
            build_receipt = fragment["modules"][0]["build_receipt"]
            fragment["modules"][0]["cuda_source"] = {
                "artifact_id": "unexpected-source",
                "kind": "cuda_source",
                "path": "evidence/unexpected.cu",
                "sha256": build_receipt["cuda_source_sha256"],
                "size_bytes": build_receipt["cuda_source_size_bytes"],
            }

    _rewrite_fragment(inputs["sm100a"], mutate)
    with pytest.raises(PromotionPackError, match=message):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


def test_fragment_pack_rejects_route_topology_denominator_drift(tmp_path):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        module_ids = [module["id"] for module in fragment["modules"]]
        route = fragment["routes"][0]
        route["module_ids"] = module_ids
        route["public_activity_contract"]["device_kernel_names"] = [
            module["kernel_name"] for module in fragment["modules"]
        ]
        route["public_activity_contract"].update(
            {
                "expected_activity_segments": [
                    {"activity_count": 0, "fixed_markers": []},
                    {"activity_count": 0, "fixed_markers": []},
                    {"activity_count": 0, "fixed_markers": []},
                ],
                "expected_fixed_host_activity_markers": [],
                "expected_host_activity_count": 0,
                "host_roles": [],
            }
        )
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm100a"], mutate)
    with pytest.raises(PromotionPackError, match="topology denominator"):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / "rejected-route-denominator",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unknown_seed", "references an unknown seed"),
        ("unknown_module", "references an unknown module"),
        ("unused_seed", "seed denominator differs from route references"),
        ("seed_module_bundle", "seed 0 envelope is invalid"),
    ],
)
def test_fragment_pack_rejects_invalid_seed_and_route_closure(
    tmp_path, mutation, message
):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        if mutation == "unknown_seed":
            fragment["routes"][0]["seed_id"] = "seed-unknown"
        elif mutation == "unknown_module":
            fragment["routes"][0]["module_ids"] = ["module-unknown"]
        elif mutation == "seed_module_bundle":
            fragment["seeds"][0]["module_ids"] = fragment["routes"][0]["module_ids"]
        else:
            fragment["seeds"].append({"id": "seed-unused"})
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm100a"], mutate)
    with pytest.raises(PromotionPackError, match=message):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("seed", "dispatcher/seed identity"),
        ("route_modules", "route denominator identity"),
    ],
)
def test_fragment_pack_rejects_unsealed_seed_or_route_identity_drift(
    tmp_path, mutation, message
):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate_fragment(fragment):
        if mutation == "seed":
            fragment["seeds"][0]["id"] = "seed-drifted"
        else:
            fragment["routes"][32]["module_ids"].reverse()

    _rewrite_fragment(inputs["sm100a"], mutate_fragment)
    with pytest.raises(PromotionPackError, match=message):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-unsealed-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


@pytest.mark.parametrize("mutation", ["role", "marker", "count", "order"])
def test_fragment_pack_rejects_activity_outside_fixed_contract(tmp_path, mutation):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        route_index = 0 if mutation == "order" else 9
        activity = fragment["routes"][route_index]["public_activity_contract"]
        if mutation == "role":
            activity["host_roles"] = ["unknown_host_role"]
        elif mutation == "marker":
            activity["expected_fixed_host_activity_markers"] = [
                "direct_copy_kernel_cuda"
            ]
        elif mutation == "count":
            activity["expected_host_activity_count"] = 1
        else:
            activity["host_roles"] = [
                "affine_torch_epilogue",
                "beta_tma_refresh",
            ]
            activity["expected_host_activity_count"] = 4
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm100a"], mutate)
    with pytest.raises(PromotionPackError, match="activity identity"):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-activity-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "unknown_key",
        "moved_marker",
        "wrong_count",
        "early_epilogue",
        "role_order",
    ],
)
def test_fragment_pack_rejects_activity_segment_drift(tmp_path, mutation):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        activity = fragment["routes"][41]["public_activity_contract"]
        segments = activity["expected_activity_segments"]
        if mutation == "missing":
            segments.pop()
        elif mutation == "extra":
            segments.append({"activity_count": 0, "fixed_markers": []})
        elif mutation == "unknown_key":
            segments[0]["unknown"] = True
        elif mutation == "moved_marker":
            segments[0]["fixed_markers"] = []
            segments[2]["fixed_markers"] = ["direct_copy_kernel_cuda"]
        elif mutation == "wrong_count":
            segments[0]["activity_count"] = 2
        elif mutation == "early_epilogue":
            segments[3]["activity_count"] = 4
            segments[4]["activity_count"] = 0
        else:
            activity["host_roles"] = [
                "beta_tma_refresh",
                "beta_tma_refresh",
                "affine_torch_epilogue",
                "beta_tma_refresh",
            ]
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm100a"], mutate)
    with pytest.raises(PromotionPackError, match="activity (segments|identity)"):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-activity-segment-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


@pytest.mark.parametrize("mutation", ["combined", "bt16", "affine", "denominator"])
def test_fragment_pack_rejects_activity_topology_or_denominator_drift(
    tmp_path, mutation
):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cubin")
        inputs[target_name] = source

    def mutate(fragment):
        route_index = {
            "combined": 0,
            "bt16": 32,
            "affine": 41,
            "denominator": 9,
        }[mutation]
        activity = fragment["routes"][route_index]["public_activity_contract"]
        if mutation == "combined":
            activity["host_roles"] = [
                "beta_tma_refresh",
                "affine_torch_epilogue",
            ]
            activity["expected_fixed_host_activity_markers"] = [
                "direct_copy_kernel_cuda"
            ]
            activity["expected_host_activity_count"] = 4
        elif mutation in ("bt16", "denominator"):
            activity["host_roles"] = ["beta_tma_refresh"]
            activity["expected_fixed_host_activity_markers"] = [
                "direct_copy_kernel_cuda"
            ]
            activity["expected_host_activity_count"] = 1
            if mutation == "denominator":
                activity["expected_activity_segments"] = [
                    {
                        "activity_count": 1,
                        "fixed_markers": ["direct_copy_kernel_cuda"],
                    },
                    {"activity_count": 0, "fixed_markers": []},
                ]
        else:
            activity["host_roles"] = []
            activity["expected_fixed_host_activity_markers"] = []
            activity["expected_host_activity_count"] = 0
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm100a"], mutate)
    expected_error = (
        "activity denominator" if mutation == "denominator" else "activity identity"
    )
    with pytest.raises(PromotionPackError, match=expected_error):
        pack_public_fragment_promotions(
            inputs,
            mode="cubin",
            name="example-program",
            target=tmp_path / f"rejected-activity-topology-{mutation}",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


def test_fragment_pack_rejects_cross_target_route_identity_drift(tmp_path):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cuda")
        inputs[target_name] = source

    def mutate(fragment):
        fragment["routes"][0]["id"] = "different-route-id"
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm103a"], mutate)
    with pytest.raises(PromotionPackError, match="logical route topology"):
        pack_public_fragment_promotions(
            inputs,
            mode="cuda",
            name="example-program",
            target=tmp_path / "rejected-topology",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


def test_fragment_pack_rejects_cross_target_activity_segment_topology_drift(tmp_path):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cuda")
        inputs[target_name] = source

    def mutate(fragment):
        routes = fragment["routes"]
        routes[0]["public_activity_contract"], routes[9]["public_activity_contract"] = (
            routes[9]["public_activity_contract"],
            routes[0]["public_activity_contract"],
        )
        _seal_fragment_identities(fragment)

    _rewrite_fragment(inputs["sm103a"], mutate)
    with pytest.raises(PromotionPackError, match="logical route topology"):
        pack_public_fragment_promotions(
            inputs,
            mode="cuda",
            name="example-program",
            target=tmp_path / "rejected-activity-topology",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )


def test_fragment_pack_rejects_nonexecutable_cuda_recipe(tmp_path):
    inputs = {}
    for target_name in ("sm100a", "sm103a"):
        source = tmp_path / f"source-{target_name}"
        _write_fragment_input(source, target_name, "cuda")
        inputs[target_name] = source
    receipt_path = inputs["sm100a"] / "promotion-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    recipe = next(
        item for item in receipt["artifacts"] if item["kind"] == "build_recipe"
    )
    recipe["executable"] = False
    (inputs["sm100a"] / recipe["path"]).chmod(0o644)
    receipt_path.write_bytes(_canonical(receipt) + b"\n")

    with pytest.raises(PromotionPackError, match="executable flag"):
        pack_public_fragment_promotions(
            inputs,
            mode="cuda",
            name="example-program",
            target=tmp_path / "rejected-recipe-mode",
            runtime_manifest_destination="csrc/example/runtime.json",
            runtime_contract=_RUNTIME_CONTRACT,
            dispatcher_run_entrypoint="prepare_fwd",
            dispatcher_select_entrypoint="select_route",
        )
