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

"""Pack sanitized per-target promotions into one exact public inventory."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Mapping


PUBLIC_RECEIPT_KIND = "generated_program_public_promotion_receipt"
PACK_KIND = "flashinfer.generated_program_pack"
IMPORT_KIND = "flashinfer.generated_program_promotion"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_ARCHITECTURE = re.compile(r"sm_([0-9]{2,3})a\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_FRAGMENT_KIND = "flashinfer.generated_program_pack.fragment"
_FRAGMENT_TARGETS = ("sm100a", "sm103a")
_FRAGMENT_ROUTE_COUNT = 48
_FRAGMENT_SELECTOR_ARGUMENTS = {
    "fixed_layout",
    "gpu_arch",
    "num_heads",
    "sequence_lengths",
    "sm_count",
    "store_final_state",
    "use_initial_state",
}
_FRAGMENT_TOPOLOGY_COUNTS = {1: 32, 2: 9, 4: 7}
_HOST_ACTIVITY_ROLES = ("beta_tma_refresh", "affine_torch_epilogue")
_AFFINE_HOST_ACTIVITY_ROLES = (
    _HOST_ACTIVITY_ROLES[0],
    _HOST_ACTIVITY_ROLES[0],
    _HOST_ACTIVITY_ROLES[0],
    _HOST_ACTIVITY_ROLES[1],
)
_AFFINE_EPILOGUE_ONLY_ROLES = (_HOST_ACTIVITY_ROLES[1],)
_FRAGMENT_ACTIVITY_TOPOLOGY_COUNTS = {
    (1, ()): 23,
    (1, (_HOST_ACTIVITY_ROLES[0],)): 9,
    (2, ()): 9,
    (4, _AFFINE_HOST_ACTIVITY_ROLES): 4,
    (4, _AFFINE_EPILOGUE_ONLY_ROLES): 3,
}
_RECEIPT_KEYS = {
    "architecture",
    "artifacts",
    "contracts",
    "kind",
    "mode",
    "name",
    "route_count",
    "route_denominator_sha256",
    "runtime_inventory",
    "runtime_inventory_identity",
    "schema_version",
}
_ARTIFACT_KEYS = {"executable", "id", "kind", "path", "sha256", "size_bytes"}
_CONTRACT_KEYS = {"correctness", "performance"}
_DENOMINATOR_KEYS = {"denominator_sha256"}
_PUBLICIZED_RECEIPT_KEYS = _RECEIPT_KEYS - {
    "runtime_inventory",
    "runtime_inventory_identity",
}
_PUBLICIZED_ARTIFACT_KEYS = _ARTIFACT_KEYS - {"id"}
_FRAGMENT_KEYS = {
    "architecture",
    "build",
    "contract",
    "dispatcher",
    "dispatcher_seed_identity",
    "kind",
    "mode",
    "modules",
    "pack_kind",
    "package_shared_library",
    "route_denominator_sha256",
    "routes",
    "schema_version",
    "seeds",
    "selector",
    "target",
}
_FRAGMENT_REF_KEYS = {"artifact_id", "kind", "path", "sha256", "size_bytes"}
_FRAGMENT_MODULE_KEYS = {
    "build_output",
    "build_receipt",
    "build_recipe",
    "cubin",
    "cuda_source",
    "entry_point",
    "host_source",
    "id",
    "kernel_name",
    "module_ident",
    "shared_library",
}
_BUILD_RECEIPT_KEYS = {
    "compile_options",
    "cooperative",
    "cubin_sha256",
    "cubin_size_bytes",
    "cuda_source_sha256",
    "cuda_source_size_bytes",
    "host_source_sha256",
    "host_source_size_bytes",
    "tma_abi",
    "use_pdl",
}
_FRAGMENT_ROUTE_KEYS = {
    "id",
    "module_ids",
    "public_activity_contract",
    "route",
    "route_index",
    "seed_id",
    "selector_facts",
}
_FRAGMENT_SELECTOR_KEYS = {"arguments", "kind", "route_count"}
_FRAGMENT_BUILD_KEYS = {"kind", "outputs", "recipe"}
_ACTIVITY_KEYS = {
    "device_kernel_names",
    "expected_activity_segments",
    "expected_fixed_host_activity_markers",
    "expected_host_activity_count",
    "host_roles",
}
_ACTIVITY_SEGMENT_KEYS = {"activity_count", "fixed_markers"}
_FINAL_INVENTORY_KEYS = {
    "architecture",
    "contract",
    "dispatcher",
    "dispatcher_seed_identity",
    "mode",
    "modules",
    "routes",
    "schema_version",
    "seeds",
}


class PromotionPackError(ValueError):
    """A sanitized input or packed public inventory is invalid."""


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PromotionPackError(message)


def _relative(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    assert isinstance(value, str)
    path = PurePosixPath(value)
    _require(
        "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and value not in (".", "..")
        and ".." not in path.parts,
        f"{label} must be a normalized relative path",
    )
    return path


def _sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"{label} must be one lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _safe_id(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _SAFE_ID.fullmatch(value) is not None,
        f"{label} must be a safe identifier",
    )
    assert isinstance(value, str)
    return value


def _identifier(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None,
        f"{label} must be a Python identifier",
    )
    assert isinstance(value, str)
    return value


def _string_list(
    value: object,
    label: str,
    *,
    allow_empty: bool = False,
    allow_duplicates: bool = False,
) -> list[str]:
    _require(
        isinstance(value, list)
        and (allow_empty or bool(value))
        and all(isinstance(item, str) and bool(item) for item in value),
        f"{label} must be an ordered string list",
    )
    assert isinstance(value, list)
    result = list(value)
    _require(
        allow_duplicates or len(result) == len(set(result)),
        f"{label} contains duplicates",
    )
    return result


def _activity_segments(
    value: object,
    label: str,
    *,
    module_count: int,
) -> list[dict[str, object]]:
    _require(
        isinstance(value, list) and len(value) == module_count + 1,
        f"{label} must contain one segment around every device stage",
    )
    assert isinstance(value, list)
    result: list[dict[str, object]] = []
    for index, item in enumerate(value):
        segment_label = f"{label} {index}"
        _require(
            isinstance(item, dict) and set(item) == _ACTIVITY_SEGMENT_KEYS,
            f"{segment_label} is invalid",
        )
        assert isinstance(item, dict)
        activity_count = item.get("activity_count")
        _require(
            isinstance(activity_count, int)
            and not isinstance(activity_count, bool)
            and activity_count >= 0,
            f"{segment_label} activity count is invalid",
        )
        fixed_markers = _string_list(
            item.get("fixed_markers"),
            f"{segment_label} fixed markers",
            allow_empty=True,
            allow_duplicates=True,
        )
        result.append(
            {
                "activity_count": activity_count,
                "fixed_markers": fixed_markers,
            }
        )
    return result


def _expected_activity_identity(
    module_count: int,
    roles: tuple[str, ...],
) -> tuple[int, list[str], list[dict[str, object]]] | None:
    empty = {"activity_count": 0, "fixed_markers": []}
    copy = {
        "activity_count": 1,
        "fixed_markers": ["direct_copy_kernel_cuda"],
    }
    if (module_count, roles) == (1, ()):
        return 0, [], [dict(empty), dict(empty)]
    if (module_count, roles) == (1, (_HOST_ACTIVITY_ROLES[0],)):
        return 1, ["direct_copy_kernel_cuda"], [dict(copy), dict(empty)]
    if (module_count, roles) == (2, ()):
        return 0, [], [dict(empty), dict(empty), dict(empty)]
    if (module_count, roles) == (4, _AFFINE_HOST_ACTIVITY_ROLES):
        return (
            6,
            ["direct_copy_kernel_cuda"] * 3,
            [
                dict(copy),
                dict(copy),
                dict(empty),
                dict(copy),
                {"activity_count": 3, "fixed_markers": []},
            ],
        )
    if (module_count, roles) == (4, _AFFINE_EPILOGUE_ONLY_ROLES):
        return (
            3,
            [],
            [
                dict(empty),
                dict(empty),
                dict(empty),
                dict(empty),
                {"activity_count": 3, "fixed_markers": []},
            ],
        )
    return None


def _load_json(path: Path) -> dict[str, object]:
    _require(path.is_file() and not path.is_symlink(), f"not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PromotionPackError(f"could not read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _target(architecture: object) -> str:
    _require(isinstance(architecture, str), "architecture must be a string")
    assert isinstance(architecture, str)
    match = _ARCHITECTURE.fullmatch(architecture)
    _require(match is not None, f"unsupported architecture: {architecture!r}")
    assert match is not None
    return f"sm{match.group(1)}a"


def _safe_file(root: Path, relative: PurePosixPath) -> Path:
    _require(
        root.is_dir() and not root.is_symlink(),
        f"input root is not a real directory: {root}",
    )
    current = root.resolve()
    for component in relative.parts:
        current = current / component
        _require(not current.is_symlink(), f"input path traverses a symlink: {current}")
    _require(current.is_file(), f"input artifact is not a regular file: {current}")
    return current


def _input_files(root: Path) -> set[str]:
    observed: set[str] = set()
    for directory, directories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in directories:
            _require(
                not (base / name).is_symlink(),
                f"input contains a directory symlink: {base / name}",
            )
        for name in files:
            path = base / name
            _require(not path.is_symlink(), f"input contains a file symlink: {path}")
            _require(
                stat.S_ISREG(path.stat().st_mode),
                f"input contains a non-regular file: {path}",
            )
            observed.add(path.relative_to(root).as_posix())
    return observed


def _validated_input(root: Path, *, mode: str) -> dict[str, object]:
    receipt = _load_json(root / "promotion-receipt.json")
    _require(set(receipt) == _RECEIPT_KEYS, "public receipt envelope is invalid")
    _require(
        receipt.get("kind") == PUBLIC_RECEIPT_KIND
        and receipt.get("schema_version") == SCHEMA_VERSION,
        "public receipt kind/schema is invalid",
    )
    _require(
        receipt.get("mode") == mode,
        "public receipt mode differs from the selected mode",
    )
    _require(
        isinstance(receipt.get("name"), str)
        and _NAME.fullmatch(str(receipt["name"])) is not None,
        "public receipt name is invalid",
    )
    target = _target(receipt.get("architecture"))
    inventory = receipt.get("runtime_inventory")
    _require(
        isinstance(inventory, dict),
        "public receipt runtime_inventory must be an object",
    )
    _require(
        receipt.get("runtime_inventory_identity") == _digest(inventory),
        "public receipt runtime inventory identity is invalid",
    )
    contracts = receipt.get("contracts")
    _require(
        isinstance(contracts, dict) and set(contracts) == _CONTRACT_KEYS,
        "public receipt contracts envelope is invalid",
    )
    assert isinstance(contracts, dict)
    for contract_name in sorted(_CONTRACT_KEYS):
        contract = contracts.get(contract_name)
        _require(
            isinstance(contract, dict) and set(contract) == _DENOMINATOR_KEYS,
            f"public receipt {contract_name} contract is invalid",
        )
        assert isinstance(contract, dict)
        _sha256(
            contract.get("denominator_sha256"),
            f"public receipt {contract_name} denominator",
        )
    artifacts = receipt.get("artifacts")
    _require(
        isinstance(artifacts, list) and bool(artifacts),
        "public receipt has no artifacts",
    )
    expected = {"promotion-receipt.json"}
    artifact_ids: set[str] = set()
    normalized: list[dict[str, object]] = []
    for index, raw in enumerate(artifacts):
        _require(
            isinstance(raw, dict) and set(raw) == _ARTIFACT_KEYS,
            f"artifact {index} envelope is invalid",
        )
        artifact_id = raw.get("id")
        _require(
            isinstance(artifact_id, str)
            and bool(artifact_id)
            and artifact_id not in artifact_ids,
            f"artifact {index} id is invalid or repeated",
        )
        assert isinstance(artifact_id, str)
        artifact_ids.add(artifact_id)
        relative = _relative(raw.get("path"), f"artifact {artifact_id} path")
        expected.add(relative.as_posix())
        path = _safe_file(root, relative)
        digest = _sha256(raw.get("sha256"), f"artifact {artifact_id} sha256")
        size = raw.get("size_bytes")
        executable = raw.get("executable")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            "artifact size is invalid",
        )
        _require(isinstance(executable, bool), "artifact executable flag is invalid")
        _require(
            _sha256_file(path) == (digest, size), f"artifact bytes drifted: {relative}"
        )
        _require(
            bool(path.stat().st_mode & 0o111) == executable,
            f"artifact mode drifted: {relative}",
        )
        normalized.append(dict(raw))
    _require(
        _input_files(root) == expected,
        "public input file closure differs from its receipt",
    )
    return {**receipt, "target": target, "artifacts": normalized}


def _contract_denominators(value: object, *, label: str) -> dict[str, str]:
    _require(
        isinstance(value, dict) and set(value) == _CONTRACT_KEYS,
        f"{label} contracts envelope is invalid",
    )
    assert isinstance(value, dict)
    result: dict[str, str] = {}
    for name in sorted(_CONTRACT_KEYS):
        contract = value.get(name)
        _require(
            isinstance(contract, dict) and "denominator_sha256" in contract,
            f"{label} {name} contract is invalid",
        )
        assert isinstance(contract, dict)
        result[name] = _sha256(
            contract.get("denominator_sha256"),
            f"{label} {name} denominator",
        )
    return result


def _validated_publicized_input(
    root: Path,
    *,
    expected_target: str,
    mode: str,
    name: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    receipt = _load_json(root / "promotion-receipt.json")
    _require(
        set(receipt) == _PUBLICIZED_RECEIPT_KEYS,
        "publicized receipt envelope is invalid",
    )
    _require(
        receipt.get("kind") == PUBLIC_RECEIPT_KIND
        and receipt.get("schema_version") == SCHEMA_VERSION,
        "publicized receipt kind/schema is invalid",
    )
    _require(
        receipt.get("mode") == mode,
        "publicized receipt mode differs from the selected mode",
    )
    _require(receipt.get("name") == name, "publicized receipt name differs")
    _require(
        _target(receipt.get("architecture")) == expected_target,
        f"publicized receipt architecture differs from {expected_target}",
    )
    _require(
        receipt.get("route_count") == _FRAGMENT_ROUTE_COUNT,
        "publicized receipt route count differs",
    )
    _sha256(
        receipt.get("route_denominator_sha256"),
        "publicized receipt route denominator",
    )
    denominators = _contract_denominators(
        receipt.get("contracts"), label="publicized receipt"
    )
    raw_artifacts = receipt.get("artifacts")
    _require(
        isinstance(raw_artifacts, list) and bool(raw_artifacts),
        "publicized receipt has no artifacts",
    )
    assert isinstance(raw_artifacts, list)
    artifacts: dict[str, dict[str, object]] = {}
    expected_files = {"promotion-receipt.json"}
    fragment_paths: list[PurePosixPath] = []
    for index, raw in enumerate(raw_artifacts):
        _require(
            isinstance(raw, dict) and set(raw) == _PUBLICIZED_ARTIFACT_KEYS,
            f"publicized artifact {index} envelope is invalid",
        )
        assert isinstance(raw, dict)
        relative = _relative(raw.get("path"), f"publicized artifact {index} path")
        path_key = relative.as_posix()
        _require(path_key not in artifacts, "publicized artifact paths repeat")
        path = _safe_file(root, relative)
        digest = _sha256(raw.get("sha256"), f"publicized artifact {index} sha256")
        size = raw.get("size_bytes")
        executable = raw.get("executable")
        kind = raw.get("kind")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"publicized artifact {index} size is invalid",
        )
        _require(
            isinstance(executable, bool),
            f"publicized artifact {index} executable flag is invalid",
        )
        _require(
            isinstance(kind, str) and bool(kind),
            f"publicized artifact {index} kind is invalid",
        )
        _require(
            _sha256_file(path) == (digest, size),
            f"publicized artifact bytes drifted: {path_key}",
        )
        _require(
            bool(path.stat().st_mode & 0o111) == executable,
            f"publicized artifact mode drifted: {path_key}",
        )
        artifacts[path_key] = dict(raw)
        expected_files.add(path_key)
        if relative.name == "fragment.json":
            fragment_paths.append(relative)
    _require(
        _input_files(root) == expected_files,
        "publicized input file closure differs from its receipt",
    )
    _require(len(fragment_paths) == 1, "publicized input must contain one fragment")
    fragment = _load_json(_safe_file(root, fragment_paths[0]))
    _require(
        fragment.get("kind") == _FRAGMENT_KIND,
        "publicized fragment kind is invalid",
    )
    return (
        {**receipt, "contracts": denominators},
        artifacts,
        fragment,
    )


def _fragment_reference(
    value: object,
    *,
    artifacts: Mapping[str, dict[str, object]],
    identities: dict[str, tuple[str, str, str, int]],
    path_ids: dict[str, str],
    label: str,
    installed: bool,
) -> tuple[dict[str, object], dict[str, object] | None]:
    _require(
        isinstance(value, dict) and set(value) == _FRAGMENT_REF_KEYS,
        f"{label} reference is invalid",
    )
    assert isinstance(value, dict)
    artifact_id = _safe_id(value.get("artifact_id"), f"{label} artifact id")
    kind = value.get("kind")
    _require(isinstance(kind, str) and bool(kind), f"{label} kind is invalid")
    assert isinstance(kind, str)
    relative = _relative(value.get("path"), f"{label} path")
    path_key = relative.as_posix()
    digest = _sha256(value.get("sha256"), f"{label} sha256")
    size = value.get("size_bytes")
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label} size is invalid",
    )
    assert isinstance(size, int)
    identity = (kind, path_key, digest, size)
    previous = identities.setdefault(artifact_id, identity)
    _require(previous == identity, f"{label} artifact id resolves inconsistently")
    previous_id = path_ids.setdefault(path_key, artifact_id)
    _require(previous_id == artifact_id, f"{label} path resolves to multiple ids")
    artifact = artifacts.get(path_key)
    if artifact is not None:
        _require(
            artifact.get("kind") == kind
            and artifact.get("sha256") == digest
            and artifact.get("size_bytes") == size,
            f"{label} differs from its publicized artifact",
        )
    _require(not installed or artifact is not None, f"{label} is not installed")
    return dict(value), artifact


def _optional_fragment_reference(
    value: object,
    **kwargs: object,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    if value is None:
        return None, None
    return _fragment_reference(value, **kwargs)


def _install_artifact(
    reference: Mapping[str, object],
    artifact: Mapping[str, object] | None,
    *,
    selected: dict[str, dict[str, object]],
    label: str,
    kind: str,
    suffixes: tuple[str, ...],
    executable: bool,
) -> None:
    _require(artifact is not None, f"{label} is not installed")
    assert artifact is not None
    artifact_id = str(reference["artifact_id"])
    path = PurePosixPath(str(reference["path"]))
    _require(reference["kind"] == kind, f"{label} kind differs")
    _require(path.suffix in suffixes, f"{label} suffix differs")
    _require(
        artifact.get("executable") is executable,
        f"{label} executable flag differs",
    )
    record = {**artifact, "id": artifact_id}
    previous = selected.setdefault(artifact_id, record)
    _require(previous == record, f"{label} installed identity differs")


def _validated_build_receipt(value: object, *, label: str) -> dict[str, object]:
    _require(
        isinstance(value, dict) and set(value) == _BUILD_RECEIPT_KEYS,
        f"{label} build receipt is invalid",
    )
    assert isinstance(value, dict)
    for field in ("cuda_source", "cubin", "host_source"):
        _sha256(value.get(f"{field}_sha256"), f"{label} {field} sha256")
        size = value.get(f"{field}_size_bytes")
        _require(
            isinstance(size, int) and not isinstance(size, bool) and size >= 0,
            f"{label} {field} size is invalid",
        )
    options = _string_list(value.get("compile_options"), f"{label} compile options")
    _require(value.get("tma_abi") == "pointer", f"{label} tensor-map ABI differs")
    _require(
        type(value.get("cooperative")) is bool and type(value.get("use_pdl")) is bool,
        f"{label} launch flags are invalid",
    )
    return {**value, "compile_options": options}


def _compact_reference(value: Mapping[str, object]) -> dict[str, object]:
    return {
        "artifact_id": value["artifact_id"],
        "sha256": value["sha256"],
    }


def _reference_matches_receipt(
    reference: Mapping[str, object],
    receipt: Mapping[str, object],
    *,
    field: str,
    label: str,
) -> None:
    _require(
        reference.get("sha256") == receipt.get(f"{field}_sha256")
        and reference.get("size_bytes") == receipt.get(f"{field}_size_bytes"),
        f"{label} differs from its build receipt",
    )


def _normalized_fragment_input(
    root: Path,
    *,
    expected_target: str,
    mode: str,
    name: str,
    runtime_contract: Mapping[str, object],
    dispatcher_run_entrypoint: str,
    dispatcher_select_entrypoint: str,
) -> tuple[dict[str, object], dict[str, object]]:
    receipt, artifacts, fragment = _validated_publicized_input(
        root,
        expected_target=expected_target,
        mode=mode,
        name=name,
    )
    _require(set(fragment) == _FRAGMENT_KEYS, "fragment envelope is invalid")
    architecture = receipt["architecture"]
    _require(
        fragment.get("kind") == _FRAGMENT_KIND
        and fragment.get("schema_version") == SCHEMA_VERSION
        and fragment.get("pack_kind") == PACK_KIND,
        "fragment kind/schema is invalid",
    )
    _require(
        fragment.get("target") == expected_target
        and fragment.get("architecture") == architecture
        and fragment.get("mode") == mode,
        "fragment target identity differs",
    )
    _require(
        fragment.get("contract") == dict(runtime_contract),
        "fragment runtime contract differs",
    )

    identities: dict[str, tuple[str, str, str, int]] = {}
    path_ids: dict[str, str] = {}
    selected: dict[str, dict[str, object]] = {}

    dispatcher, dispatcher_artifact = _fragment_reference(
        fragment.get("dispatcher"),
        artifacts=artifacts,
        identities=identities,
        path_ids=path_ids,
        label="fragment dispatcher",
        installed=True,
    )
    _install_artifact(
        dispatcher,
        dispatcher_artifact,
        selected=selected,
        label="fragment dispatcher",
        kind="python_source",
        suffixes=(".py",),
        executable=False,
    )
    dispatcher_record = {
        **_compact_reference(dispatcher),
        "run_entrypoint": dispatcher_run_entrypoint,
        "select_entrypoint": dispatcher_select_entrypoint,
    }

    package_library, package_artifact = _fragment_reference(
        fragment.get("package_shared_library"),
        artifacts=artifacts,
        identities=identities,
        path_ids=path_ids,
        label="fragment package library",
        installed=True,
    )
    _require(
        package_library["kind"] == "shared_library"
        and PurePosixPath(str(package_library["path"])).suffix == ".so"
        and package_artifact is not None
        and package_artifact.get("executable") is False,
        "fragment package library identity differs",
    )

    raw_modules = fragment.get("modules")
    _require(
        isinstance(raw_modules, list) and bool(raw_modules),
        "fragment modules must be non-empty",
    )
    assert isinstance(raw_modules, list)
    module_ids: list[str] = []
    module_positions: dict[str, int] = {}
    module_kernel_names: dict[str, str] = {}
    final_modules: list[dict[str, object]] = []
    raw_recipes: list[dict[str, object]] = []
    raw_outputs: list[dict[str, object]] = []
    output_paths: set[str] = set()
    for index, raw_module in enumerate(raw_modules):
        label = f"fragment module {index}"
        _require(
            isinstance(raw_module, dict) and set(raw_module) == _FRAGMENT_MODULE_KEYS,
            f"{label} envelope is invalid",
        )
        assert isinstance(raw_module, dict)
        module_id = _safe_id(raw_module.get("id"), f"{label} id")
        _require(module_id not in module_positions, f"{label} id repeats")
        module_positions[module_id] = index
        module_ids.append(module_id)
        module_ident = _identifier(raw_module.get("module_ident"), f"{label} ident")
        entry_point = _identifier(raw_module.get("entry_point"), f"{label} entry point")
        kernel_name = raw_module.get("kernel_name")
        _require(
            isinstance(kernel_name, str) and bool(kernel_name),
            f"{label} kernel name is invalid",
        )
        assert isinstance(kernel_name, str)
        module_kernel_names[module_id] = kernel_name
        build_receipt = _validated_build_receipt(
            raw_module.get("build_receipt"), label=label
        )

        source, source_artifact = _optional_fragment_reference(
            raw_module.get("cuda_source"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} CUDA source",
            installed=mode == "cuda",
        )
        host, host_artifact = _fragment_reference(
            raw_module.get("host_source"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} host source",
            installed=mode == "cuda",
        )
        shared, shared_artifact = _fragment_reference(
            raw_module.get("shared_library"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} shared library",
            installed=mode == "cubin",
        )
        cubin, cubin_artifact = _optional_fragment_reference(
            raw_module.get("cubin"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} cubin",
            installed=mode == "cubin",
        )
        recipe, recipe_artifact = _optional_fragment_reference(
            raw_module.get("build_recipe"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} build recipe",
            installed=mode == "cuda",
        )
        build_output, build_output_artifact = _fragment_reference(
            raw_module.get("build_output"),
            artifacts=artifacts,
            identities=identities,
            path_ids=path_ids,
            label=f"{label} build output",
            installed=mode == "cubin",
        )
        _require(
            host["kind"] == "host_source"
            and PurePosixPath(str(host["path"])).suffix in (".cc", ".cpp")
            and (host_artifact is None or host_artifact.get("executable") is False),
            f"{label} host evidence differs",
        )
        _require(
            shared["kind"] == "shared_library"
            and PurePosixPath(str(shared["path"])).suffix == ".so"
            and (shared_artifact is None or shared_artifact.get("executable") is False),
            f"{label} shared evidence differs",
        )
        _require(
            build_output["kind"] == "cubin"
            and PurePosixPath(str(build_output["path"])).suffix == ".cubin"
            and (
                build_output_artifact is None
                or build_output_artifact.get("executable") is False
            ),
            f"{label} build output evidence differs",
        )
        if cubin is not None:
            _require(
                cubin["kind"] == "cubin"
                and PurePosixPath(str(cubin["path"])).suffix == ".cubin"
                and (
                    cubin_artifact is None or cubin_artifact.get("executable") is False
                ),
                f"{label} cubin evidence differs",
            )
        _reference_matches_receipt(
            host,
            build_receipt,
            field="host_source",
            label=f"{label} host source",
        )
        _reference_matches_receipt(
            build_output,
            build_receipt,
            field="cubin",
            label=f"{label} build output",
        )
        if cubin is not None:
            _reference_matches_receipt(
                cubin,
                build_receipt,
                field="cubin",
                label=f"{label} cubin",
            )
        if source is not None:
            _reference_matches_receipt(
                source,
                build_receipt,
                field="cuda_source",
                label=f"{label} CUDA source",
            )

        if mode == "cubin":
            _require(
                source is None and recipe is None,
                f"{label} cubin mode has CUDA build fields",
            )
            _require(
                cubin is not None
                and cubin_artifact is not None
                and build_output == cubin
                and build_output_artifact == cubin_artifact,
                f"{label} exact cubin identity differs",
            )
            _install_artifact(
                cubin,
                cubin_artifact,
                selected=selected,
                label=f"{label} cubin",
                kind="cubin",
                suffixes=(".cubin",),
                executable=False,
            )
            _install_artifact(
                shared,
                shared_artifact,
                selected=selected,
                label=f"{label} shared library",
                kind="shared_library",
                suffixes=(".so",),
                executable=False,
            )
            final_module = {
                "build_output": None,
                "cubin": _compact_reference(cubin),
                "entry_point": entry_point,
                "host": _compact_reference(host),
                "id": module_id,
                "module_ident": module_ident,
                "recipe": None,
                "shared_library": _compact_reference(shared),
                "source": None,
            }
        else:
            _require(
                source is not None
                and source_artifact is not None
                and recipe is not None
                and recipe_artifact is not None
                and cubin is not None
                and build_output == cubin
                and build_output_artifact == cubin_artifact,
                f"{label} CUDA mode closure differs",
            )
            _install_artifact(
                source,
                source_artifact,
                selected=selected,
                label=f"{label} CUDA source",
                kind="cuda_source",
                suffixes=(".cu",),
                executable=False,
            )
            _install_artifact(
                host,
                host_artifact,
                selected=selected,
                label=f"{label} host source",
                kind="host_source",
                suffixes=(".cc", ".cpp"),
                executable=False,
            )
            _install_artifact(
                recipe,
                recipe_artifact,
                selected=selected,
                label=f"{label} build recipe",
                kind="build_recipe",
                suffixes=(".py",),
                executable=True,
            )
            output_path = _relative(build_output["path"], f"{label} build output path")
            _require(
                output_path.suffix == ".cubin"
                and output_path.as_posix() not in output_paths,
                f"{label} build output path differs",
            )
            output_paths.add(output_path.as_posix())
            final_module = {
                "build_output": {
                    "id": build_output["artifact_id"],
                    "path": output_path.as_posix(),
                    "sha256": build_output["sha256"],
                    "size_bytes": build_output["size_bytes"],
                },
                "cubin": _compact_reference(cubin),
                "entry_point": entry_point,
                "host": _compact_reference(host),
                "id": module_id,
                "module_ident": module_ident,
                "recipe": _compact_reference(recipe),
                "shared_library": _compact_reference(shared),
                "source": _compact_reference(source),
            }
            raw_recipes.append(recipe)
            raw_outputs.append(build_output)
        final_modules.append(final_module)

    raw_build = fragment.get("build")
    _require(
        isinstance(raw_build, dict) and set(raw_build) == _FRAGMENT_BUILD_KEYS,
        "fragment build envelope is invalid",
    )
    assert isinstance(raw_build, dict)
    if mode == "cubin":
        _require(
            raw_build == {"kind": "prebuilt", "recipe": None, "outputs": []},
            "fragment prebuilt closure differs",
        )
    else:
        _require(bool(raw_recipes), "fragment CUDA recipe is missing")
        _require(
            all(recipe == raw_recipes[0] for recipe in raw_recipes)
            and raw_build.get("kind") == "nvrtc"
            and raw_build.get("recipe") == raw_recipes[0]
            and raw_build.get("outputs") == raw_outputs,
            "fragment CUDA build closure differs",
        )

    selector = fragment.get("selector")
    _require(
        isinstance(selector, dict) and set(selector) == _FRAGMENT_SELECTOR_KEYS,
        "fragment selector envelope is invalid",
    )
    assert isinstance(selector, dict)
    selector_arguments = _string_list(
        selector.get("arguments"), "fragment selector arguments"
    )
    _require(
        selector.get("kind") == "exact_selector_facts"
        and selector.get("route_count") == _FRAGMENT_ROUTE_COUNT,
        "fragment selector denominator differs",
    )
    _require(
        set(selector_arguments) == _FRAGMENT_SELECTOR_ARGUMENTS,
        "fragment selector argument set differs",
    )

    raw_seeds = fragment.get("seeds")
    _require(
        isinstance(raw_seeds, list) and bool(raw_seeds),
        "fragment seeds must be non-empty",
    )
    assert isinstance(raw_seeds, list)
    seeds: list[dict[str, object]] = []
    seed_by_id: dict[str, dict[str, object]] = {}
    for index, raw_seed in enumerate(raw_seeds):
        label = f"fragment seed {index}"
        _require(
            isinstance(raw_seed, dict) and set(raw_seed) == {"id"},
            f"{label} envelope is invalid",
        )
        assert isinstance(raw_seed, dict)
        seed_id = _safe_id(raw_seed.get("id"), f"{label} id")
        seed = {"id": seed_id}
        _require(seed_id not in seed_by_id, f"{label} id repeats")
        seed_by_id[seed_id] = seed
        seeds.append(seed)

    raw_routes = fragment.get("routes")
    _require(
        isinstance(raw_routes, list) and len(raw_routes) == _FRAGMENT_ROUTE_COUNT,
        "fragment route denominator differs",
    )
    assert isinstance(raw_routes, list)
    _require(
        fragment.get("route_denominator_sha256")
        == hashlib.sha256(_canonical(raw_routes)).hexdigest(),
        "fragment route denominator identity differs",
    )
    _require(
        fragment.get("dispatcher_seed_identity")
        == _digest(
            {"dispatcher": dispatcher, "seeds": raw_seeds, "routes": raw_routes}
        ),
        "fragment dispatcher/seed identity differs",
    )
    routes: list[dict[str, object]] = []
    logical_routes: list[dict[str, object]] = []
    route_ids: set[str] = set()
    referenced_seed_ids: set[str] = set()
    topology_counts = {module_count: 0 for module_count in _FRAGMENT_TOPOLOGY_COUNTS}
    activity_topology_counts = {
        topology: 0 for topology in _FRAGMENT_ACTIVITY_TOPOLOGY_COUNTS
    }
    for index, raw_route in enumerate(raw_routes):
        label = f"fragment route {index}"
        _require(
            isinstance(raw_route, dict) and set(raw_route) == _FRAGMENT_ROUTE_KEYS,
            f"{label} envelope is invalid",
        )
        assert isinstance(raw_route, dict)
        route_id = _safe_id(raw_route.get("id"), f"{label} id")
        _require(route_id not in route_ids, f"{label} id repeats")
        route_ids.add(route_id)
        _require(raw_route.get("route_index") == index, f"{label} order differs")
        route_name = raw_route.get("route")
        _require(
            isinstance(route_name, str) and bool(route_name),
            f"{label} route name is invalid",
        )
        seed_id = _safe_id(raw_route.get("seed_id"), f"{label} seed id")
        _require(seed_id in seed_by_id, f"{label} references an unknown seed")
        referenced_seed_ids.add(seed_id)
        route_modules = _string_list(
            raw_route.get("module_ids"),
            f"{label} modules",
            allow_duplicates=True,
        )
        _require(
            all(module_id in module_positions for module_id in route_modules),
            f"{label} references an unknown module",
        )
        module_count = len(route_modules)
        _require(
            module_count in topology_counts,
            f"{label} module count is outside the fixed topology",
        )
        topology_counts[module_count] += 1
        selector_facts = raw_route.get("selector_facts")
        _require(
            isinstance(selector_facts, dict)
            and set(selector_facts) == set(selector_arguments),
            f"{label} selector denominator differs",
        )
        assert isinstance(selector_facts, dict)
        _require(
            selector_facts.get("gpu_arch") == architecture,
            f"{label} selector architecture differs",
        )
        sm_count = selector_facts.get("sm_count")
        _require(
            isinstance(sm_count, int)
            and not isinstance(sm_count, bool)
            and sm_count > 0,
            f"{label} selector SM count is invalid",
        )
        activity = raw_route.get("public_activity_contract")
        _require(
            isinstance(activity, dict) and set(activity) == _ACTIVITY_KEYS,
            f"{label} activity contract is invalid",
        )
        assert isinstance(activity, dict)
        expected_names = [module_kernel_names[item] for item in route_modules]
        host_roles = _string_list(
            activity.get("host_roles"),
            f"{label} host roles",
            allow_empty=True,
            allow_duplicates=True,
        )
        markers = _string_list(
            activity.get("expected_fixed_host_activity_markers"),
            f"{label} host markers",
            allow_empty=True,
            allow_duplicates=True,
        )
        segments = _activity_segments(
            activity.get("expected_activity_segments"),
            f"{label} activity segments",
            module_count=module_count,
        )
        host_count = activity.get("expected_host_activity_count")
        roles = tuple(host_roles)
        expected_activity = _expected_activity_identity(module_count, roles)
        _require(
            activity.get("device_kernel_names") == expected_names
            and expected_activity is not None
            and isinstance(host_count, int)
            and not isinstance(host_count, bool)
            and host_count == expected_activity[0]
            and markers == expected_activity[1]
            and segments == expected_activity[2],
            f"{label} activity identity differs",
        )
        activity_topology = (module_count, roles)
        _require(
            activity_topology in activity_topology_counts,
            f"{label} activity denominator differs from its route topology",
        )
        activity_topology_counts[activity_topology] += 1
        final_route = {
            "id": route_id,
            "module_ids": route_modules,
            "seed_id": seed_id,
            "selector": dict(selector_facts),
        }
        routes.append(final_route)
        logical_routes.append(
            {
                "activity": {
                    "expected_activity_segments": segments,
                    "expected_fixed_host_activity_markers": markers,
                    "expected_host_activity_count": host_count,
                    "host_roles": host_roles,
                },
                "id": route_id,
                "module_count": module_count,
                "route_index": index,
                "selector": {
                    key: value
                    for key, value in selector_facts.items()
                    if key not in {"gpu_arch", "sm_count"}
                },
            }
        )

    _require(
        topology_counts == _FRAGMENT_TOPOLOGY_COUNTS,
        "fragment topology denominator differs",
    )
    _require(
        referenced_seed_ids == set(seed_by_id),
        "fragment seed denominator differs from route references",
    )
    _require(
        activity_topology_counts == _FRAGMENT_ACTIVITY_TOPOLOGY_COUNTS,
        "fragment activity denominator differs",
    )

    dispatcher_seed = {
        "contract": dict(runtime_contract),
        "dispatcher": dispatcher_record,
        "routes": routes,
        "seeds": seeds,
    }
    inventory = {
        "architecture": architecture,
        "contract": dict(runtime_contract),
        "dispatcher": dispatcher_record,
        "dispatcher_seed_identity": _digest(dispatcher_seed),
        "mode": mode,
        "modules": final_modules,
        "routes": routes,
        "schema_version": SCHEMA_VERSION,
        "seeds": seeds,
    }
    _require(set(inventory) == _FINAL_INVENTORY_KEYS, "runtime inventory differs")
    route_denominator = hashlib.sha256(_canonical(routes)).hexdigest()
    normalized_receipt = {
        "architecture": architecture,
        "artifacts": list(selected.values()),
        "contracts": {
            contract_name: {"denominator_sha256": denominator}
            for contract_name, denominator in receipt["contracts"].items()
        },
        "kind": PUBLIC_RECEIPT_KIND,
        "mode": mode,
        "name": name,
        "route_count": len(routes),
        "route_denominator_sha256": route_denominator,
        "runtime_inventory": inventory,
        "runtime_inventory_identity": _digest(inventory),
        "schema_version": SCHEMA_VERSION,
    }
    logical = {
        "routes": logical_routes,
        "selector_arguments": selector_arguments,
    }
    return normalized_receipt, logical


def _write_normalized_input(
    source_root: Path,
    target_root: Path,
    receipt: Mapping[str, object],
) -> None:
    target_root.mkdir()
    artifacts = receipt.get("artifacts")
    assert isinstance(artifacts, list)
    for artifact in artifacts:
        assert isinstance(artifact, dict)
        relative = _relative(artifact["path"], "normalized artifact path")
        source = _safe_file(source_root, relative)
        destination = target_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if artifact["executable"] else 0o644)
    (target_root / "promotion-receipt.json").write_bytes(_canonical(receipt) + b"\n")


def pack_public_promotions(
    inputs: Mapping[str, Path],
    *,
    mode: str,
    name: str,
    target: Path,
    runtime_manifest_destination: str,
) -> dict[str, object]:
    """Pack one selected mode across named targets into an importer payload."""

    _require(mode in ("cuda", "cubin"), "mode must be 'cuda' or 'cubin'")
    _require(
        _NAME.fullmatch(name) is not None, "name is not a safe promotion identifier"
    )
    _require(bool(inputs), "at least one target input is required")
    runtime_destination = _relative(
        runtime_manifest_destination, "runtime manifest destination"
    )
    loaded: dict[str, dict[str, object]] = {}
    for expected_target, root in inputs.items():
        receipt = _validated_input(Path(root).absolute(), mode=mode)
        _require(
            receipt["target"] == expected_target,
            f"input target label differs from {expected_target}",
        )
        _require(expected_target not in loaded, f"duplicate target: {expected_target}")
        loaded[expected_target] = receipt
    names = {str(receipt["name"]) for receipt in loaded.values()}
    _require(names == {name}, "public input names differ from the selected name")
    correctness = {
        str(receipt["contracts"]["correctness"]["denominator_sha256"])
        for receipt in loaded.values()
    }
    performance = {
        str(receipt["contracts"]["performance"]["denominator_sha256"])
        for receipt in loaded.values()
    }
    _require(
        len(correctness) == len(performance) == 1,
        "public input contract denominators differ",
    )

    target = target.absolute()
    _require(
        not target.exists() and not target.is_symlink(),
        f"refusing to overwrite pack target: {target}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.incomplete-", dir=target.parent)
    )
    try:
        payload = temporary / "payload"
        payload.mkdir()
        importer_artifacts: list[dict[str, object]] = []
        entries: list[dict[str, object]] = []
        for target_name in sorted(loaded):
            receipt = loaded[target_name]
            installed: list[dict[str, object]] = []
            source_root = Path(inputs[target_name]).absolute()
            for artifact in receipt["artifacts"]:
                assert isinstance(artifact, dict)
                relative = _relative(artifact["path"], "public artifact path")
                source_relative = PurePosixPath(target_name) / relative
                destination = (
                    PurePosixPath("csrc/generated_programs")
                    / name
                    / target_name
                    / relative
                )
                source = _safe_file(source_root, relative)
                output = payload.joinpath(*source_relative.parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, output)
                output.chmod(0o755 if artifact["executable"] else 0o644)
                _require(
                    _sha256_file(output)
                    == (artifact["sha256"], artifact["size_bytes"]),
                    "packed artifact differs from its public input",
                )
                importer_artifacts.append(
                    {
                        "destination": destination.as_posix(),
                        "executable": artifact["executable"],
                        "sha256": artifact["sha256"],
                        "size_bytes": artifact["size_bytes"],
                        "source": source_relative.as_posix(),
                    }
                )
                installed.append(
                    {
                        **artifact,
                        "path": destination.as_posix(),
                    }
                )
            entries.append(
                {
                    "architecture": receipt["architecture"],
                    "artifact_root": (
                        PurePosixPath("csrc/generated_programs") / name / target_name
                    ).as_posix(),
                    "artifacts": installed,
                    "route_count": receipt["route_count"],
                    "route_denominator_sha256": receipt["route_denominator_sha256"],
                    "runtime_inventory": receipt["runtime_inventory"],
                    "runtime_inventory_identity": receipt["runtime_inventory_identity"],
                    "target": target_name,
                }
            )
        pack_manifest = {
            "contract_denominators": {
                "correctness": next(iter(correctness)),
                "performance": next(iter(performance)),
            },
            "entries": entries,
            "kind": PACK_KIND,
            "mode": mode,
            "name": name,
            "schema_version": SCHEMA_VERSION,
        }
        runtime_source = PurePosixPath("runtime-manifest.json")
        runtime_bytes = _canonical(pack_manifest) + b"\n"
        (payload / runtime_source).write_bytes(runtime_bytes)
        importer_artifacts.append(
            {
                "destination": runtime_destination.as_posix(),
                "executable": False,
                "sha256": hashlib.sha256(runtime_bytes).hexdigest(),
                "size_bytes": len(runtime_bytes),
                "source": runtime_source.as_posix(),
            }
        )
        importer_artifacts.sort(key=lambda artifact: str(artifact["destination"]))
        importer_manifest = {
            "artifacts": importer_artifacts,
            "kind": IMPORT_KIND,
            "mode": mode,
            "name": name,
            "schema_version": SCHEMA_VERSION,
        }
        (temporary / "promotion-manifest.json").write_bytes(
            _canonical(importer_manifest) + b"\n"
        )
        (temporary / "pack-manifest.json").write_bytes(runtime_bytes)
        os.rename(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return pack_manifest


def pack_public_fragment_promotions(
    inputs: Mapping[str, Path],
    *,
    mode: str,
    name: str,
    target: Path,
    runtime_manifest_destination: str,
    runtime_contract: Mapping[str, object],
    dispatcher_run_entrypoint: str,
    dispatcher_select_entrypoint: str,
) -> dict[str, object]:
    """Compose two exact publicized fragments through the generic packer."""

    _require(mode in ("cuda", "cubin"), "mode must be 'cuda' or 'cubin'")
    _require(
        tuple(sorted(inputs)) == _FRAGMENT_TARGETS,
        f"fragment inputs must be exactly {list(_FRAGMENT_TARGETS)}",
    )
    _require(
        isinstance(runtime_contract, Mapping) and bool(runtime_contract),
        "runtime contract must be a non-empty object",
    )
    try:
        normalized_contract = json.loads(_canonical(dict(runtime_contract)))
    except (TypeError, ValueError) as exc:
        raise PromotionPackError("runtime contract is not canonical JSON") from exc
    _require(
        isinstance(normalized_contract, dict), "runtime contract must be an object"
    )
    run_entrypoint = _identifier(dispatcher_run_entrypoint, "dispatcher run entrypoint")
    select_entrypoint = _identifier(
        dispatcher_select_entrypoint, "dispatcher select entrypoint"
    )
    _require(
        run_entrypoint != select_entrypoint,
        "dispatcher run/select entrypoints must differ",
    )
    target = target.absolute()
    _require(
        not target.exists() and not target.is_symlink(),
        f"refusing to overwrite pack target: {target}",
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.fragments-", dir=target.parent
    ) as temporary_name:
        temporary = Path(temporary_name)
        normalized_inputs: dict[str, Path] = {}
        expected_logical: dict[str, object] | None = None
        for target_name in _FRAGMENT_TARGETS:
            source_root = Path(inputs[target_name]).absolute()
            receipt, logical = _normalized_fragment_input(
                source_root,
                expected_target=target_name,
                mode=mode,
                name=name,
                runtime_contract=normalized_contract,
                dispatcher_run_entrypoint=run_entrypoint,
                dispatcher_select_entrypoint=select_entrypoint,
            )
            if expected_logical is None:
                expected_logical = logical
            else:
                _require(
                    logical == expected_logical,
                    "fragment logical route topology differs across targets",
                )
            normalized_root = temporary / target_name
            _write_normalized_input(source_root, normalized_root, receipt)
            normalized_inputs[target_name] = normalized_root
        return pack_public_promotions(
            normalized_inputs,
            mode=mode,
            name=name,
            target=target,
            runtime_manifest_destination=runtime_manifest_destination,
        )


__all__ = [
    "PACK_KIND",
    "PromotionPackError",
    "pack_public_fragment_promotions",
    "pack_public_promotions",
]
