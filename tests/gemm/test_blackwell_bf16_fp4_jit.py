# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.gemm import gemm_bf16_fp4_blackwell as blackwell_backend
from flashinfer.jit import blackwell_bf16_fp4 as blackwell_jit


def _artifact_paths(target: str) -> tuple[Path, Path]:
    source_dir = blackwell_jit._source_dir()
    return (
        source_dir / blackwell_jit._SOURCE_NAMES[target],
        source_dir / blackwell_jit._MANIFEST_NAMES[target],
    )


def _integration_manifest(target: str) -> dict:
    legacy = _legacy_variant_manifest(target)
    integration_grid_modes = {
        "generic_2d": "two_dimensional",
        "generic_flat": "flat_overflow",
        "group_m128_2d": "group_m128",
        "split_k2_partial": "split_k2_partial",
        "split_k2_reduce": "split_k2_reduce",
        "persistent_sm_count": "persistent",
    }
    kernels = []
    ir_symbols = []
    for index, variant in enumerate(legacy["variants"]):
        component = variant["component"]
        ir_symbol = f"synthetic_ir_{component}"
        if ir_symbol not in ir_symbols:
            ir_symbols.append(ir_symbol)
        arg_plan_kind, prepared_abi, stage = (
            blackwell_jit._INTEGRATION_COMPONENT_METADATA[component]
        )
        threads, smem_bytes = blackwell_jit._INTEGRATION_LAUNCH_RESOURCES[
            component
        ]
        grid_mode = integration_grid_modes[variant["grid_kind"]]
        arg_plan = blackwell_jit._expected_integration_arg_plan(component)
        kernel = {
            "arg_plan": arg_plan,
            "arg_plan_kind": arg_plan_kind,
            "cluster_dims": copy.deepcopy(variant["cluster_dims"]),
            "component": component,
            "enable_pdl": variant["enable_pdl"],
            "flat_grid": variant["flat_grid"],
            "grid_mode": grid_mode,
            "logical_grid_mode": grid_mode,
            "has_alpha": variant["has_alpha"],
            "ir_symbol": ir_symbol,
            "kernel_symbol": f"kernel_flashinfer_bf16_fp4_synthetic_{index}",
            "launch_grid": {},
            "module_ident": f"flashinfer_bf16_fp4_synthetic_{index:010x}",
            "output_dtype": (
                "float32_workspace"
                if component == "cudnn_split_k2_partial_f32"
                else variant["output_dtype"]
            ),
            "prepared_abi": prepared_abi,
            "route": variant["route"],
            "schedule_symbol": f"flashinfer_bf16_fp4_synthetic_{index}",
            "smem_bytes": smem_bytes,
            "smem_data_offset_bytes": variant["smem_data_offset_bytes"],
            "smem_pool_bytes": variant["smem_pool_bytes"],
            "stage": stage,
            "threads": threads,
            "tma_descriptors": [
                {
                    "descriptor_name": resource,
                    "host_argument_index": index,
                    "kernel_argument_index": index,
                    "resource": resource,
                }
                for index, (kind, resource) in enumerate(arg_plan)
                if kind == "tma_buffer"
            ],
            "use_pdl": variant["use_pdl"],
        }
        if "tile_m" in variant:
            kernel["tile_m"] = variant["tile_m"]
        kernels.append(kernel)

    generic_row7 = next(
        kernel
        for kernel in kernels
        if kernel["component"] == "cute_warp_mma_m32_bf16"
        and kernel["has_alpha"] is True
        and kernel["enable_pdl"] is True
    )
    exact_row7 = copy.deepcopy(generic_row7)
    exact_row7.update(
        {
            "arg_plan": blackwell_jit._raw_pointer_integration_arg_plan(),
            "arg_plan_kind": "raw_pointer",
            "exact_shape": copy.deepcopy(
                blackwell_jit._INTEGRATION_ROW7_EXACT_SHAPE
            ),
            "grid_mode": "persistent_m16_2sm",
            "ir_symbol": "synthetic_ir_cute_warp_m16_2sm",
            "kernel_symbol": "kernel_flashinfer_bf16_fp4_synthetic_74",
            "module_ident": "flashinfer_bf16_fp4_synthetic_000000004a",
            "schedule_symbol": "flashinfer_bf16_fp4_synthetic_74",
            "smem_bytes": 34816,
            "smem_data_offset_bytes": 0,
            "smem_pool_bytes": 34816,
            "threads": 128,
            "tile_m": 16,
            "tma_descriptors": [],
        }
    )
    kernels.append(exact_row7)
    ir_symbols.append(exact_row7["ir_symbol"])

    route_names = list(
        dict.fromkeys(
            selection["route"]
            for selection in blackwell_jit._DISPATCH_SELECTION
        )
    )
    routes = [{"route": route, "specializations": []} for route in route_names]
    m32_route = next(
        route
        for route in routes
        if route["route"]
        == blackwell_jit._COMPONENT_SPECS["cute_warp_mma_m32_bf16"][0]
    )
    m32_kernels = [
        kernel
        for kernel in kernels
        if kernel["component"] == "cute_warp_mma_m32_bf16"
    ]
    m32_kernels.sort(key=lambda kernel: "exact_shape" not in kernel)
    m32_route["specializations"] = [
        {
            "match": {
                "out_dtype": kernel["output_dtype"],
                "has_alpha": kernel["has_alpha"],
                "enable_pdl": kernel["enable_pdl"],
                **copy.deepcopy(kernel.get("exact_shape", {})),
            }
        }
        for kernel in m32_kernels
    ]

    return {
        "schema_version": 3,
        "bundle": "flashinfer_blackwell_bf16_fp4_gemm",
        "arch": blackwell_jit._NVCC_ARCH[target],
        "tma_abi": "pointer",
        "tensor_map_abi": copy.deepcopy(blackwell_jit._TENSOR_MAP_ABI),
        "adapter_boundary": "separate_translation_unit",
        "prepared_abis": copy.deepcopy(blackwell_jit._PREPARED_ABIS),
        "ir_symbols": ir_symbols,
        "kernels": kernels,
        "dispatch": {
            "selection": "ordered_first_match_after_input_validation",
            "inputs": list(blackwell_jit._DISPATCH_INPUTS),
            "routes": routes,
        },
    }


def _legacy_variant_manifest(target: str) -> dict:
    variants = []
    for component, (
        route,
        output_dtype,
        grids,
        tile_m,
    ) in blackwell_jit._COMPONENT_SPECS.items():
        if component == "cudnn_split_k2_reduce_bf16":
            alpha_values = (None,)
        else:
            alpha_values = (False, True)
        for grid_kind, flat_grid in grids:
            for has_alpha in alpha_values:
                for enable_pdl in (False, True):
                    variant = {
                        "arg_plan": [
                            ["tma_buffer", "A"],
                            ["grid", "grid_x"],
                            ["grid", "grid_y"],
                            ["grid", "grid_z"],
                        ],
                        "cluster_dims": [1, 1, 1],
                        "component": component,
                        "enable_pdl": enable_pdl,
                        "flat_grid": flat_grid,
                        "grid_kind": grid_kind,
                        "has_alpha": has_alpha,
                        "launch_grid": {
                            axis: {
                                "host_argument_index": index,
                                "expression": {"op": "constant", "value": 1},
                            }
                            for index, axis in enumerate(("x", "y", "z"), 1)
                        },
                        "output_dtype": output_dtype,
                        "route": route,
                        "smem_bytes": 0,
                        "smem_data_offset_bytes": 0,
                        "smem_pool_bytes": 0,
                        "threads": 1,
                        "tma_descriptors": [
                            {
                                "descriptor_name": "A",
                                "host_argument_index": 0,
                                "kernel_argument_index": 0,
                                "resource": "A",
                            }
                        ],
                        "use_pdl": enable_pdl,
                    }
                    if tile_m is not None:
                        variant["tile_m"] = tile_m
                    if component == "cudnn_split_k2_reduce_bf16":
                        variant["reuses_alpha_specializations"] = [False, True]
                    symbol_stem = blackwell_jit._variant_symbol_stem(variant)
                    variant["kernel_symbol"] = f"kernel_{symbol_stem}"
                    variant["module_ident"] = (
                        f"{symbol_stem}_{len(variants):010x}"
                    )
                    variant["schedule_symbol"] = symbol_stem
                    variants.append(variant)

    return {
        "schema_version": 3,
        "bundle": "flashinfer_blackwell_bf16_fp4_gemm",
        "arch": blackwell_jit._NVCC_ARCH[target],
        "tma_abi": "pointer",
        "tensor_map_abi": copy.deepcopy(blackwell_jit._TENSOR_MAP_ABI),
        "adapter_boundary": "separate_translation_unit",
        "variants": variants,
        "dispatcher": copy.deepcopy(blackwell_jit._DISPATCHER),
        "composite_routes": copy.deepcopy(blackwell_jit._COMPOSITE_ROUTES),
        "workspaces": copy.deepcopy(blackwell_jit._WORKSPACES),
    }


def _bind_legacy_variant_fixture(
    monkeypatch: pytest.MonkeyPatch, manifest: dict
) -> None:
    monkeypatch.setattr(
        blackwell_jit,
        "_VARIANT_ABI_SHA256",
        blackwell_jit._variant_abi_sha256(manifest["variants"]),
    )


def _write_manifest(path: Path, manifest: dict) -> bytes:
    raw = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    return raw


def _integration_source(target: str, manifest_raw: bytes) -> bytes:
    definitions = "\n".join(
        f'extern "C" __global__ void '
        f"kernel_flashinfer_bf16_fp4_synthetic_{index}() {{}}"
        for index in range(75)
    )
    return (
        "#define FLASHINFER_BLACKWELL_BF16_FP4_SOURCE_READY 1\n"
        "#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION 3\n"
        f"#define FLASHINFER_BLACKWELL_BF16_FP4_TARGET_SM "
        f"{blackwell_jit._TARGET_SM[target]}\n"
        "#define FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256 "
        f'"{hashlib.sha256(b"synthetic generated source").hexdigest()}"\n'
        "#define FLASHINFER_BLACKWELL_BF16_FP4_ABI_MANIFEST_SHA256 "
        f'"{hashlib.sha256(manifest_raw).hexdigest()}"\n'
        f"{definitions}\n"
    ).encode()


@pytest.mark.parametrize("target", ["sm100", "sm103"])
def test_checked_in_schema_3_artifact_pair_is_accepted(target: str) -> None:
    source_path, manifest_path = _artifact_paths(target)

    parsed, manifest_raw = blackwell_jit._load_abi_manifest(manifest_path, target)
    source_raw = source_path.read_bytes()

    assert manifest_raw == manifest_path.read_bytes()
    assert parsed["schema_version"] == 3
    assert parsed["arch"] == blackwell_jit._NVCC_ARCH[target]
    assert len(blackwell_jit._manifest_kernel_symbols(parsed)) == 75
    assert len(parsed["ir_symbols"]) == 15
    blackwell_jit._validate_source_header(
        source_raw, parsed, manifest_raw, target
    )


@pytest.mark.parametrize("target", ["sm100", "sm103"])
def test_integration_schema_3_artifact_pair_is_accepted(
    tmp_path: Path, target: str
) -> None:
    manifest = _integration_manifest(target)
    manifest_path = tmp_path / blackwell_jit._MANIFEST_NAMES[target]
    manifest_raw = _write_manifest(manifest_path, manifest)

    parsed, parsed_raw = blackwell_jit._load_abi_manifest(manifest_path, target)

    assert parsed_raw == manifest_raw
    assert parsed["arch"] == blackwell_jit._NVCC_ARCH[target]
    assert len(parsed["kernels"]) == 75
    assert len(parsed["ir_symbols"]) == 15
    assert len(parsed["dispatch"]["routes"]) == 11
    blackwell_jit._validate_source_header(
        _integration_source(target, manifest_raw), parsed, manifest_raw, target
    )


@pytest.mark.parametrize("family", ["integration", "variant"])
def test_binding_kernel_specs_are_rendered_from_selected_manifest(
    family: str,
) -> None:
    manifest = (
        _integration_manifest("sm100")
        if family == "integration"
        else _legacy_variant_manifest("sm100")
    )
    binding_path = blackwell_jit._source_dir() / blackwell_jit._BINDING_NAME
    rendered = blackwell_jit._render_binding_source(
        binding_path.read_bytes(),
        manifest,
        "flashinfer_blackwell_bf16_fp4_test_module",
    )
    specs = blackwell_jit._manifest_kernel_specs(manifest)
    expected_count = 75 if family == "integration" else 74

    assert len(specs) == expected_count
    assert rendered.count("KernelSpec{") == expected_count
    assert blackwell_jit._KERNEL_SPECS_MARKER not in rendered
    assert blackwell_jit._KERNEL_COUNT_MARKER not in rendered
    assert f"constexpr size_t kKernelSpecCount = {expected_count};" in rendered
    assert "FLASHINFER_BLACKWELL_BF16_FP4_MODULE_IDENT" not in rendered


def test_integration_manifest_preserves_exact_row7_then_generic_m32() -> None:
    manifest = _integration_manifest("sm100")
    m32_kernels = [
        kernel
        for kernel in manifest["kernels"]
        if kernel["component"] == "cute_warp_mma_m32_bf16"
    ]
    raw_pointer = [
        kernel for kernel in m32_kernels if kernel["arg_plan_kind"] == "raw_pointer"
    ]
    canonical = [
        kernel for kernel in m32_kernels if kernel["arg_plan_kind"] == "cute_warp"
    ]

    assert len(raw_pointer) == 1
    assert len(canonical) == 4
    assert raw_pointer[0]["exact_shape"] == {
        "M": 17,
        "N": 3072,
        "K": 3072,
    }
    assert raw_pointer[0]["has_alpha"] is True
    assert raw_pointer[0]["enable_pdl"] is True
    assert raw_pointer[0]["grid_mode"] == "persistent_m16_2sm"
    assert raw_pointer[0]["logical_grid_mode"] == "persistent"
    assert raw_pointer[0]["tile_m"] == 16
    assert (raw_pointer[0]["threads"], raw_pointer[0]["smem_bytes"]) == (
        128,
        34816,
    )
    assert raw_pointer[0]["tma_descriptors"] == []

    m32_route = next(
        route
        for route in manifest["dispatch"]["routes"]
        if route["route"]
        == blackwell_jit._COMPONENT_SPECS["cute_warp_mma_m32_bf16"][0]
    )
    assert len(m32_route["specializations"]) == 5
    assert m32_route["specializations"][0]["match"] == {
        "out_dtype": "bfloat16",
        "has_alpha": True,
        "enable_pdl": True,
        "M": 17,
        "N": 3072,
        "K": 3072,
    }
    assert any(
        specialization["match"]
        == {
            "out_dtype": "bfloat16",
            "has_alpha": True,
            "enable_pdl": True,
        }
        for specialization in m32_route["specializations"][1:]
    )
    blackwell_jit._validate_integration_manifest(manifest)


@pytest.mark.parametrize(
    "value",
    [
        {"M": 17, "N": 3072, "K": 4096},
        {"M": 17.0, "N": 3072.0, "K": 3072.0},
    ],
)
def test_integration_manifest_rejects_row7_exact_shape_drift(
    value: dict[str, object],
) -> None:
    manifest = _integration_manifest("sm100")
    exact = next(
        kernel for kernel in manifest["kernels"] if "exact_shape" in kernel
    )
    exact["exact_shape"] = value

    with pytest.raises(ValueError, match="row7 exact shape"):
        blackwell_jit._validate_integration_manifest(manifest)


@pytest.mark.parametrize("target", ["sm100", "sm103"])
def test_legacy_variant_schema_3_manifest_is_accepted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, target: str
) -> None:
    manifest = _legacy_variant_manifest(target)
    _bind_legacy_variant_fixture(monkeypatch, manifest)
    manifest_path = tmp_path / blackwell_jit._MANIFEST_NAMES[target]
    manifest_raw = _write_manifest(manifest_path, manifest)

    parsed, parsed_raw = blackwell_jit._load_abi_manifest(manifest_path, target)

    assert parsed_raw == manifest_raw
    assert parsed["arch"] == blackwell_jit._NVCC_ARCH[target]
    assert len(blackwell_jit._manifest_kernel_symbols(parsed)) == 74
    assert len(parsed["dispatcher"]["selection_order"]) == 11


def test_integration_manifest_rejects_incompatible_prepared_layouts(
    tmp_path: Path,
) -> None:
    manifest = _integration_manifest("sm100")
    manifest["prepared_abis"] = {}
    manifest_path = tmp_path / "generated.abi.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="incompatible prepared layouts"):
        blackwell_jit._load_abi_manifest(manifest_path, "sm100")


def test_manifest_rejects_mixed_abi_families(tmp_path: Path) -> None:
    manifest = _integration_manifest("sm100")
    manifest["variants"] = []
    manifest_path = tmp_path / "generated.abi.json"
    _write_manifest(manifest_path, manifest)

    with pytest.raises(ValueError, match="keys do not match schema 3"):
        blackwell_jit._load_abi_manifest(manifest_path, "sm100")


def test_checked_in_sm100_and_sm103_manifests_are_an_exact_arch_pair() -> None:
    manifests = {}
    raw_source_hashes = {}
    for target in ("sm100", "sm103"):
        source_path, manifest_path = _artifact_paths(target)
        manifest, manifest_raw = blackwell_jit._load_abi_manifest(
            manifest_path, target
        )
        blackwell_jit._validate_source_header(
            source_path.read_bytes(), manifest, manifest_raw, target
        )
        manifests[target] = copy.deepcopy(manifest)
        raw_source_hashes[target] = blackwell_jit._source_define(
            source_path.read_text(encoding="utf-8"),
            "FLASHINFER_BLACKWELL_BF16_FP4_RAW_SOURCE_SHA256",
        )

    assert manifests["sm100"].pop("arch") == "sm_100a"
    assert manifests["sm103"].pop("arch") == "sm_103a"
    assert manifests["sm100"] == manifests["sm103"]
    assert raw_source_hashes["sm100"] == raw_source_hashes["sm103"]


@pytest.mark.parametrize(
    ("capability", "target"), [((10, 0), "sm100"), ((10, 3), "sm103")]
)
def test_compute_capability_routing_is_exact(
    capability: tuple[int, int], target: str
) -> None:
    assert blackwell_jit._target_for_capability(capability) == target


@pytest.mark.parametrize("capability", [(9, 0), (10, 1), (12, 0)])
def test_compute_capability_routing_rejects_other_targets(
    capability: tuple[int, int],
) -> None:
    with pytest.raises(ValueError, match="requires compute capability 10.0 or 10.3"):
        blackwell_jit._target_for_capability(capability)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version=3"),
        ("arch", "sm_103a", "architecture does not match"),
        ("tma_abi", "value", "requires pointer TMA ABI"),
        ("tensor_map_abi", {}, "incompatible TensorMap ABI"),
        ("dispatcher", {}, "incompatible dispatch routing"),
        ("composite_routes", [], "incompatible composite routing"),
        ("workspaces", [], "incompatible workspace ABI"),
        ("variants", [], "requires 74 variants"),
    ],
)
def test_manifest_rejects_incompatible_abi_fields(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manifest = _legacy_variant_manifest("sm100")
    _bind_legacy_variant_fixture(monkeypatch, manifest)
    manifest[field] = value
    path = tmp_path / "generated.abi.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        blackwell_jit._load_abi_manifest(path, "sm100")


def test_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    _, checked_in_manifest = _artifact_paths("sm100")
    raw = checked_in_manifest.read_text(encoding="utf-8").replace(
        '"schema_version": 3', '"schema_version": 3, "schema_version": 3', 1
    )
    path = tmp_path / "generated.abi.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate ABI manifest key"):
        blackwell_jit._load_abi_manifest(path, "sm100")


@pytest.mark.parametrize(
    "mutation",
    [
        "kernel_symbol",
        "module_ident",
        "threads_bool",
        "smem_bytes",
        "arg_plan",
        "tma_descriptor",
        "launch_grid",
    ],
)
def test_manifest_rejects_deep_variant_abi_mutations(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    manifest = _legacy_variant_manifest("sm100")
    _bind_legacy_variant_fixture(monkeypatch, manifest)
    variant = manifest["variants"][0]
    if mutation == "kernel_symbol":
        variant["kernel_symbol"] += "_mutated"
    elif mutation == "module_ident":
        variant["module_ident"] = "unbound_0000000000"
    elif mutation == "threads_bool":
        variant["threads"] = True
    elif mutation == "smem_bytes":
        variant["smem_bytes"] += 1
    elif mutation == "arg_plan":
        variant["arg_plan"][0][1] = "not_A"
    elif mutation == "tma_descriptor":
        variant["tma_descriptors"][0]["descriptor_name"] = "not_A"
    elif mutation == "launch_grid":
        variant["launch_grid"]["z"]["expression"]["value"] = 2
    else:  # pragma: no cover
        raise AssertionError(f"unknown mutation {mutation}")
    path = tmp_path / "generated.abi.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="ABI manifest"):
        blackwell_jit._load_abi_manifest(path, "sm100")


def test_source_header_rejects_abi_version_2() -> None:
    source_path, manifest_path = _artifact_paths("sm100")
    manifest, manifest_raw = blackwell_jit._load_abi_manifest(
        manifest_path, "sm100"
    )
    source_raw = source_path.read_bytes().replace(
        b"FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION 3",
        b"FLASHINFER_BLACKWELL_BF16_FP4_ABI_VERSION 2",
        1,
    )

    with pytest.raises(ValueError, match="incompatible ABI version"):
        blackwell_jit._validate_source_header(
            source_raw, manifest, manifest_raw, "sm100"
        )


def test_source_header_rejects_kernel_symbol_drift() -> None:
    source_path, manifest_path = _artifact_paths("sm100")
    manifest, manifest_raw = blackwell_jit._load_abi_manifest(
        manifest_path, "sm100"
    )
    kernel_symbol = sorted(blackwell_jit._manifest_kernel_symbols(manifest))[0]
    kernel_symbol = kernel_symbol.encode()
    source_raw = source_path.read_bytes().replace(
        kernel_symbol, kernel_symbol + b"_mutated", 1
    )

    with pytest.raises(ValueError, match="kernel symbols do not match"):
        blackwell_jit._validate_source_header(
            source_raw, manifest, manifest_raw, "sm100"
        )


def test_source_and_manifest_both_participate_in_cache_identity() -> None:
    nvcc = Path("/opt/cuda/bin/nvcc")
    base = blackwell_jit._source_package_key(
        "sm100", b"source-a", b"manifest-a", b"binding", nvcc
    )

    assert base != blackwell_jit._source_package_key(
        "sm100", b"source-b", b"manifest-a", b"binding", nvcc
    )
    assert base != blackwell_jit._source_package_key(
        "sm100", b"source-a", b"manifest-b", b"binding", nvcc
    )


@pytest.mark.parametrize(
    ("backend", "out_dtype"),
    [
        ("blackwell-native", torch.bfloat16),
        ("blackwell-native", torch.float16),
        ("blackwell-tiled", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("enable_pdl", [False, True])
@pytest.mark.parametrize("explicit_alpha", [False, True])
def test_synthetic_launch_preserves_alpha_out_and_pdl(
    monkeypatch: pytest.MonkeyPatch,
    backend: blackwell_backend.BlackwellBf16Fp4Backend,
    out_dtype: torch.dtype,
    enable_pdl: bool,
    explicit_alpha: bool,
) -> None:
    a = torch.zeros((2, 16), dtype=torch.bfloat16)
    if backend == "blackwell-native":
        b = torch.zeros((64, 8), dtype=torch.uint8)
        b_descale = torch.zeros((64, 1), dtype=torch.uint8).view(torch.float8_e4m3fn)
        layout_code = 0
    else:
        b = torch.zeros((1, 128), dtype=torch.int32)
        b_descale = torch.zeros((1, 64), dtype=torch.uint8)
        layout_code = 1
    alpha = torch.ones((1,), dtype=torch.float32) if explicit_alpha else None
    out = torch.empty((2, 64), dtype=out_dtype)
    captured: list[tuple] = []

    monkeypatch.setattr(
        blackwell_backend, "_require_blackwell_source_arch", lambda _device: None
    )
    monkeypatch.setattr(
        blackwell_backend,
        "_get_blackwell_bf16_fp4_module",
        lambda: SimpleNamespace(run=lambda *args: captured.append(args)),
    )

    result = blackwell_backend._compute_blackwell_bf16_fp4(
        a,
        b,
        b_descale,
        alpha,
        out_dtype,
        out,
        16,
        enable_pdl,
        backend,
    )

    assert result is out
    assert len(captured) == 1
    args = captured[0]
    assert len(args) == 7
    assert args[0] is a
    assert args[1] is b
    assert args[2].data_ptr() == b_descale.data_ptr()
    assert args[2].dtype == torch.uint8
    assert args[4] is out
    assert args[5:] == (layout_code, enable_pdl)
    if explicit_alpha:
        assert args[3] is alpha
    else:
        assert args[3].data_ptr() == a.data_ptr()
