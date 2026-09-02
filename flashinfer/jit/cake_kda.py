"""Manifest-verified JIT loader for the exported Cake KDA portfolio."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast, get_args

from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir,
    get_kda_csrc_dir,
)
from .core import JitSpec, logger

CakeKDATarget = Literal["sm100a", "sm103a"]
CakeKDAFamily = Literal[
    "bounded_bf16_evolution",
    "bounded_bf16_affine_prefix",
    "bounded_bf16_affine_h12_prefix",
    "bounded_fp32_serving",
    "bounded_fp32_affine_prefix",
    "bounded_fp32_affine_h12_prefix",
    "unbounded_bf16_serving",
    "unbounded_affine_prefix",
]
CakeKDASequenceFamily = Literal["unbounded_affine_prefix",]
CakeKDARole = Literal["main", "prepare", "chain", "map", "scan", "correction"]
CakeKDATMAABI = Literal["grid_constant", "pointer"]

_MANIFEST = "cake_kda_prefill_portfolio_export_manifest.json"
_TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
}
_SUPPORTED_FAMILIES = frozenset(get_args(CakeKDAFamily))
_SUPPORTED_ROLES = frozenset(get_args(CakeKDARole))


@dataclass(frozen=True)
class CakeKDAModuleSpec:
    """One exact-architecture source closure from the standard Cake export."""

    target: CakeKDATarget
    family: CakeKDAFamily
    policy: str
    role: CakeKDARole
    name: str
    module_ident: str
    closure_sha256: str
    compile_flags: tuple[str, ...]
    device_path: Path
    binding_path: Path
    use_pdl: bool
    tma_abi: CakeKDATMAABI
    tma_workspace_bytes: int


@dataclass(frozen=True)
class CakeKDASequenceSpec:
    """One manifest-sealed prepared multi-kernel source closure."""

    target: CakeKDATarget
    family: CakeKDASequenceFamily
    name: str
    closure_sha256: str
    compile_flags: tuple[str, ...]
    stage_order: tuple[str, ...]
    arg_plan: tuple[tuple[str, str], ...]
    device_paths: tuple[Path, ...]
    binding_path: Path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake KDA portfolio export manifest: {message}")


def _resolve_source(csrc_dir: Path, raw_path: object, label: str) -> Path:
    _require(isinstance(raw_path, str), f"{label} must be a path")
    assert isinstance(raw_path, str)
    relative = PurePosixPath(raw_path)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "kda")
        and len(relative.parts) == 3,
        f"{label} must name one csrc/kda file",
    )
    path = csrc_dir / relative.name
    _require(path.is_file(), f"{label} does not exist: {path}")
    return path


@functools.cache
def get_cake_kda_module_specs() -> tuple[CakeKDAModuleSpec, ...]:
    """Load and verify the complete manifest-sealed source closure."""

    csrc_dir = get_kda_csrc_dir()
    payload: Any = json.loads((csrc_dir / _MANIFEST).read_text())
    _require(isinstance(payload, dict), "root must be an object")
    _require(payload.get("schema") == "cake.library_export.v1", "schema mismatch")
    _require(payload.get("producer") == "cake", "producer mismatch")
    _require(payload.get("library") == "flashinfer", "library mismatch")
    _require(payload.get("name") == "cake_kda_prefill_portfolio", "name mismatch")
    _require(payload.get("artifact_kind") == "source_only", "artifact kind mismatch")

    build = payload.get("build_contract")
    _require(isinstance(build, dict), "build contract missing")
    _require(
        build.get("translation_unit_model") == "separate_device_and_binding",
        "device and binding translation units must compile separately",
    )
    _require(build.get("binary_payloads") is False, "binary payloads forbidden")
    infrastructure = build.get("target_infrastructure")
    _require(isinstance(infrastructure, dict), "target infrastructure missing")
    _require(
        infrastructure.get("required_headers") == ["tvm_ffi_utils.h"],
        "target header contract mismatch",
    )

    contract = payload.get("contract")
    _require(isinstance(contract, dict), "contract missing")
    _require(contract.get("architectures") == ["sm_100a", "sm_103a"], "architectures")
    suites = contract.get("shape_suites")
    _require(isinstance(suites, dict) and bool(suites), "shape suites missing")
    suite_rows = 0
    for suite, labels in suites.items():
        _require(
            isinstance(suite, str)
            and bool(suite)
            and isinstance(labels, list)
            and bool(labels)
            and all(isinstance(label, str) and bool(label) for label in labels)
            and len(set(labels)) == len(labels),
            f"{suite!r} must contain unique nonempty labels",
        )
        suite_rows += len(labels)
    denominator = contract.get("shape_denominator")
    _require(
        isinstance(denominator, list)
        and bool(denominator)
        and all(isinstance(label, str) and bool(label) for label in denominator)
        and len(set(denominator)) == len(denominator)
        and contract.get("shape_count") == len(denominator),
        "shape denominator must contain shape_count unique nonempty rows",
    )
    _require(
        suite_rows == len(denominator)
        and {label for labels in suites.values() for label in labels}
        == set(denominator),
        "shape suites must exactly partition the denominator",
    )
    required_problem_shapes = contract.get("required_problem_shapes")
    _require(
        isinstance(required_problem_shapes, list)
        and len(set(required_problem_shapes)) == len(required_problem_shapes)
        and set(required_problem_shapes).issubset(denominator),
        "required predecessor/problem shapes must be a unique denominator subset",
    )
    core_base_shape_count = contract.get("core_base_shape_count")
    core_state_arms = contract.get("core_state_arms")
    if core_base_shape_count is not None or core_state_arms is not None:
        _require(
            isinstance(core_base_shape_count, int)
            and not isinstance(core_base_shape_count, bool)
            and core_base_shape_count > 0,
            "core_base_shape_count must be positive",
        )
        _require(
            isinstance(core_state_arms, list)
            and bool(core_state_arms)
            and all(isinstance(arm, str) and bool(arm) for arm in core_state_arms)
            and len(set(core_state_arms)) == len(core_state_arms),
            "core_state_arms must be unique nonempty labels",
        )
        _require(
            isinstance(suites.get("core177"), list)
            and len(suites["core177"]) == core_base_shape_count * len(core_state_arms),
            "core suite size must equal base shapes times state arms",
        )
    timing_requirement = contract.get("timing_requirement")
    _require(
        isinstance(timing_requirement, str)
        and "source_export_interleaved" in timing_requirement,
        "interleaved timing requirement missing",
    )

    files = payload.get("files")
    _require(isinstance(files, list) and bool(files), "file inventory missing")
    file_sha256: dict[str, str] = {}
    for index, item in enumerate(files):
        _require(isinstance(item, dict), f"files[{index}] must be an object")
        raw_path = item.get("path")
        digest = item.get("sha256")
        _require(isinstance(raw_path, str), f"files[{index}].path missing")
        _require(
            isinstance(digest, str) and len(digest) == 64, f"files[{index}].sha256"
        )
        path = _resolve_source(csrc_dir, raw_path, f"files[{index}].path")
        _require(
            hashlib.sha256(path.read_bytes()).hexdigest() == digest,
            f"files[{index}] hash",
        )
        _require(raw_path not in file_sha256, f"duplicate file {raw_path}")
        file_sha256[raw_path] = digest

    modules = payload.get("modules")
    target_by_arch = {arch: target for target, arch in _TARGET_ARCH.items()}
    _require(isinstance(modules, list) and bool(modules), "module inventory missing")
    observed: set[tuple[str, str, str, str]] = set()
    specs: list[CakeKDAModuleSpec] = []
    for index, item in enumerate(modules):
        label = f"modules[{index}]"
        _require(isinstance(item, dict), f"{label} must be an object")
        arch = item.get("arch")
        route = item.get("route")
        role = item.get("role")
        _require(arch in target_by_arch, f"{label}.arch unsupported")
        _require(isinstance(route, dict), f"{label}.route missing")
        family = route.get("family")
        policy = route.get("policy")
        _require(family in _SUPPORTED_FAMILIES, f"{label}.family unsupported")
        _require(
            isinstance(policy, str) and bool(policy),
            f"{label}.policy must be nonempty",
        )
        _require(role in _SUPPORTED_ROLES, f"{label}.role unsupported")
        typed_family = cast(CakeKDAFamily, family)
        typed_role = cast(CakeKDARole, role)
        key = (arch, family, policy, role)
        _require(key not in observed, f"duplicate module {key}")
        observed.add(key)

        units = item.get("translation_units")
        _require(isinstance(units, dict), f"{label}.translation_units")
        _require(units.get("compile_separately") is True, f"{label}.compile_separately")
        device_raw = units.get("device")
        binding_raw = units.get("binding")
        device_path = _resolve_source(csrc_dir, device_raw, f"{label}.device")
        binding_path = _resolve_source(csrc_dir, binding_raw, f"{label}.binding")
        _require(
            device_raw in file_sha256 and binding_raw in file_sha256, f"{label}.files"
        )
        closure = item.get("closure")
        _require(isinstance(closure, list) and len(closure) == 2, f"{label}.closure")
        closure_map = {
            entry.get("path"): entry.get("sha256")
            for entry in closure
            if isinstance(entry, dict)
        }
        _require(
            closure_map
            == {
                device_raw: file_sha256[device_raw],
                binding_raw: file_sha256[binding_raw],
            },
            f"{label}.closure mismatch",
        )
        module_ident = item.get("module_ident")
        closure_sha256 = item.get("closure_sha256")
        compile_flags = item.get("compile_flags")
        launch = item.get("launch")
        arg_plan = item.get("arg_plan")
        tma_abi = item.get("tma_abi")
        tma_workspace_bytes = item.get("tma_workspace_bytes")
        _require(item.get("ffi_entry") == "run", f"{label}.ffi_entry")
        _require(
            isinstance(arg_plan, list)
            and all(
                isinstance(entry, list)
                and len(entry) == 2
                and all(isinstance(part, str) and part for part in entry)
                for entry in arg_plan
            ),
            f"{label}.arg_plan",
        )
        _require(tma_abi in ("grid_constant", "pointer"), f"{label}.tma_abi")
        _require(
            isinstance(tma_workspace_bytes, int) and tma_workspace_bytes >= 0,
            f"{label}.tma_workspace_bytes",
        )
        workspace_args = [
            entry
            for entry in arg_plan
            if entry == ["workspace", "tma_descriptor_workspace"]
        ]
        if tma_abi == "grid_constant":
            _require(
                tma_workspace_bytes == 0 and not workspace_args,
                f"{label} grid-constant ABI must not request TMA workspace",
            )
        else:
            _require(
                tma_workspace_bytes > 0
                and tma_workspace_bytes % 128 == 0
                and workspace_args == [["workspace", "tma_descriptor_workspace"]],
                f"{label} pointer ABI requires one aligned caller TMA workspace",
            )
        _require(isinstance(item.get("name"), str), f"{label}.name")
        _require(isinstance(module_ident, str), f"{label}.module_ident")
        _require(
            isinstance(closure_sha256, str) and len(closure_sha256) == 64,
            f"{label}.closure_sha256",
        )
        _require(
            isinstance(compile_flags, list)
            and all(isinstance(flag, str) for flag in compile_flags),
            f"{label}.compile_flags",
        )
        _require(isinstance(launch, dict), f"{label}.launch")
        specs.append(
            CakeKDAModuleSpec(
                target=cast(CakeKDATarget, target_by_arch[arch]),
                family=typed_family,
                policy=policy,
                role=typed_role,
                name=item["name"],
                module_ident=module_ident,
                closure_sha256=closure_sha256,
                compile_flags=tuple(compile_flags),
                device_path=device_path,
                binding_path=binding_path,
                use_pdl=launch.get("use_pdl") is True,
                tma_abi=tma_abi,
                tma_workspace_bytes=tma_workspace_bytes,
            )
        )

    observed_arch_families = {
        (arch, family) for arch, family, _policy, _role in observed
    }
    expected_arch_families = {
        (arch, family) for arch in target_by_arch for family in _SUPPORTED_FAMILIES
    }
    _require(
        observed_arch_families == expected_arch_families,
        "every architecture must contain every portfolio family",
    )
    specs.sort(key=lambda spec: (spec.target, spec.family, spec.policy, spec.role))
    return tuple(specs)


@functools.cache
def get_cake_kda_sequence_specs() -> tuple[CakeKDASequenceSpec, ...]:
    """Load the two allocation-free unbounded affine sequences."""

    module_specs = get_cake_kda_module_specs()
    module_by_identity = {
        (spec.target, spec.name, spec.role): spec for spec in module_specs
    }
    csrc_dir = get_kda_csrc_dir()
    payload: Any = json.loads((csrc_dir / _MANIFEST).read_text())
    files = payload.get("files")
    assert isinstance(files, list)
    file_sha256 = {
        item["path"]: item["sha256"]
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("sha256"), str)
    }
    raw_sequences = payload.get("sequences")
    _require(
        isinstance(raw_sequences, list) and len(raw_sequences) == 2,
        "prepared sequence inventory mismatch",
    )
    expected: set[tuple[CakeKDATarget, CakeKDASequenceFamily]] = {
        (target, family)
        for target in cast(tuple[CakeKDATarget, ...], tuple(_TARGET_ARCH))
        for family in cast(
            tuple[CakeKDASequenceFamily, ...], ("unbounded_affine_prefix",)
        )
    }
    observed: set[tuple[CakeKDATarget, CakeKDASequenceFamily]] = set()
    specs: list[CakeKDASequenceSpec] = []
    target_by_arch = {arch: target for target, arch in _TARGET_ARCH.items()}
    for index, item in enumerate(raw_sequences):
        label = f"sequences[{index}]"
        _require(isinstance(item, dict), f"{label} must be an object")
        arch = item.get("arch")
        route = item.get("route")
        _require(arch in target_by_arch, f"{label}.arch unsupported")
        _require(isinstance(route, dict), f"{label}.route missing")
        family = cast(CakeKDASequenceFamily, route.get("family"))
        target = cast(CakeKDATarget, target_by_arch[arch])
        key = (target, family)
        _require(key in expected, f"{label} unsupported route {key}")
        _require(key not in observed, f"duplicate sequence {key}")
        observed.add(key)
        _require(
            route.get("policy") == "affine_prepared_sequence",
            f"{label}.route policy",
        )
        _require(item.get("role") == "composite", f"{label}.role")
        expected_name = "cake_kda_affine_unbounded_softplus_sequence"
        _require(item.get("name") == expected_name, f"{label}.name")
        _require(item.get("ffi_entry") == "run", f"{label}.ffi_entry")
        _require(
            item.get("ffi_abi") == "packed_positional",
            f"{label}.ffi_abi",
        )
        stage_order = item.get("stage_order")
        _require(
            stage_order == ["main", "map", "scan", "correction"],
            f"{label}.stage order",
        )
        stages = item.get("stages")
        _require(isinstance(stages, list) and len(stages) == 4, f"{label}.stages")
        expected_stage_specs = (
            get_cake_kda_module_spec(target, family, "affine_split_main", "main"),
            get_cake_kda_module_spec(target, family, "affine_split_map", "map"),
            get_cake_kda_module_spec(
                target,
                "unbounded_affine_prefix",
                "affine_prefix_scan",
                "scan",
            ),
            get_cake_kda_module_spec(
                target,
                family,
                "affine_split_correction",
                "correction",
            ),
        )
        device_raw: list[str] = []
        for stage_index, (stage, expected_spec, stage_name) in enumerate(
            zip(stages, expected_stage_specs, stage_order, strict=True)
        ):
            stage_label = f"{label}.stages[{stage_index}]"
            _require(isinstance(stage, dict), f"{stage_label} must be an object")
            _require(stage.get("name") == stage_name, f"{stage_label}.name")
            module_ref = stage.get("module")
            _require(isinstance(module_ref, dict), f"{stage_label}.module")
            identity = (
                target,
                module_ref.get("name"),
                cast(CakeKDARole, module_ref.get("role")),
            )
            _require(
                module_by_identity.get(identity) == expected_spec,
                f"{stage_label}.module identity",
            )
            raw_device = stage.get("device")
            _require(isinstance(raw_device, str), f"{stage_label}.device")
            _require(
                _resolve_source(csrc_dir, raw_device, f"{stage_label}.device")
                == expected_spec.device_path,
                f"{stage_label}.device path",
            )
            device_raw.append(raw_device)
        units = item.get("translation_units")
        _require(isinstance(units, dict), f"{label}.translation_units")
        _require(units.get("compile_separately") is True, f"{label}.compile_separately")
        _require(units.get("devices") == device_raw, f"{label}.device closure")
        binding_raw = units.get("binding")
        binding_path = _resolve_source(csrc_dir, binding_raw, f"{label}.binding")
        closure = item.get("closure")
        _require(
            isinstance(closure, list) and len(closure) == 5,
            f"{label}.closure",
        )
        closure_map = {
            entry.get("path"): entry.get("sha256")
            for entry in closure
            if isinstance(entry, dict)
        }
        expected_closure = {
            path: file_sha256[path] for path in [*device_raw, binding_raw]
        }
        _require(closure_map == expected_closure, f"{label}.closure mismatch")
        compile_flags = item.get("compile_flags")
        _require(
            isinstance(compile_flags, list)
            and all(isinstance(flag, str) for flag in compile_flags),
            f"{label}.compile_flags",
        )
        arg_plan = item.get("arg_plan")
        _require(
            isinstance(arg_plan, list)
            and len(arg_plan) > 0
            and all(
                isinstance(entry, list)
                and len(entry) == 2
                and all(isinstance(part, str) and part for part in entry)
                for entry in arg_plan
            ),
            f"{label}.arg_plan",
        )
        closure_sha256 = item.get("closure_sha256")
        _require(
            isinstance(closure_sha256, str) and len(closure_sha256) == 64,
            f"{label}.closure_sha256",
        )
        specs.append(
            CakeKDASequenceSpec(
                target=target,
                family=family,
                name=expected_name,
                closure_sha256=closure_sha256,
                compile_flags=tuple(compile_flags),
                stage_order=tuple(stage_order),
                arg_plan=tuple(tuple(entry) for entry in arg_plan),
                device_paths=tuple(
                    _resolve_source(csrc_dir, path, f"{label}.device")
                    for path in device_raw
                ),
                binding_path=binding_path,
            )
        )
    _require(observed == expected, "prepared sequence set mismatch")
    specs.sort(key=lambda spec: (spec.target, spec.family))
    return tuple(specs)


def cake_kda_is_available() -> bool:
    return bool(get_cake_kda_module_specs()) and len(get_cake_kda_sequence_specs()) == 2


def get_cake_kda_module_spec(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
) -> CakeKDAModuleSpec:
    for spec in get_cake_kda_module_specs():
        if (spec.target, spec.family, spec.policy, spec.role) == (
            target,
            family,
            policy,
            role,
        ):
            return spec
    raise ValueError(f"unsupported Cake KDA module: {target}/{family}/{policy}/{role}")


def get_cake_kda_sequence_spec(
    target: CakeKDATarget,
    family: CakeKDASequenceFamily,
) -> CakeKDASequenceSpec:
    for spec in get_cake_kda_sequence_specs():
        if (spec.target, spec.family) == (target, family):
            return spec
    raise ValueError(f"unsupported Cake KDA prepared sequence: {target}/{family}")


@functools.cache
def gen_cake_kda_module(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
) -> JitSpec:
    spec = get_cake_kda_module_spec(target, family, policy, role)
    jit_spec = gen_kda_jit_spec(
        name=f"{spec.module_ident}_{target}_{spec.closure_sha256}",
        sources=[spec.device_path, spec.binding_path],
        target=target,
        target_define=_TARGET_DEFINE[target],
        csrc_dir=get_kda_csrc_dir(),
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=spec.compile_flags,
    )
    logger.info(
        "Generated Cake KDA portfolio JIT spec: "
        f"target={target}, family={family}, policy={policy}, role={role}"
    )
    return jit_spec


@functools.cache
def get_cake_kda_module(
    target: CakeKDATarget,
    family: CakeKDAFamily,
    policy: str,
    role: CakeKDARole = "main",
):
    return gen_cake_kda_module(target, family, policy, role).build_and_load()


@functools.cache
def gen_cake_kda_sequence(
    target: CakeKDATarget,
    family: CakeKDASequenceFamily,
) -> JitSpec:
    spec = get_cake_kda_sequence_spec(target, family)
    jit_spec = gen_kda_jit_spec(
        name=f"{spec.name}_{target}_{spec.closure_sha256}",
        sources=[*spec.device_paths, spec.binding_path],
        target=target,
        target_define=_TARGET_DEFINE[target],
        csrc_dir=get_kda_csrc_dir(),
        include_dir=get_flashinfer_include_dir(),
        extra_cuda_cflags=spec.compile_flags,
    )
    logger.info(
        "Generated Cake KDA prepared-sequence JIT spec: "
        f"target={target}, family={family}, stages={spec.stage_order}"
    )
    return jit_spec


@functools.cache
def get_cake_kda_sequence(
    target: CakeKDATarget,
    family: CakeKDASequenceFamily,
):
    return gen_cake_kda_sequence(target, family).build_and_load()


__all__ = [
    "CakeKDAFamily",
    "CakeKDAModuleSpec",
    "CakeKDARole",
    "CakeKDASequenceFamily",
    "CakeKDASequenceSpec",
    "CakeKDATarget",
    "cake_kda_is_available",
    "gen_cake_kda_module",
    "gen_cake_kda_sequence",
    "get_cake_kda_module",
    "get_cake_kda_module_spec",
    "get_cake_kda_module_specs",
    "get_cake_kda_sequence",
    "get_cake_kda_sequence_spec",
    "get_cake_kda_sequence_specs",
]
