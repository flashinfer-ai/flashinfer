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
"""

import functools
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_flashinfer_include_dir as _get_flash_kda_include_dir,
    get_kda_csrc_dir as _get_flash_kda_csrc_dir,
)
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from .flash_kda_nvrtc import prepare_generated_flash_kda_cubin

FlashKDAVariant = Literal[
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "m128_n16_short",
    "persistent_m128",
    "piece_persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
    "bt16_prepare_chain_m64_s8",
]
FlashKDATarget = Literal["sm100a", "sm100f", "sm103a"]
GeneratedFlashKDATarget = Literal["sm100a", "sm103a"]

FLASH_KDA_VARIANTS: tuple[FlashKDAVariant, ...] = (
    "m64",
    "m128",
    "m128_tensor_state_decay",
    "m128_h12_short",
    "m128_h12_long",
    "m128_n16",
    "m128_n16_checkpoint",
    "m128_n16_short",
    "persistent_m128",
    "piece_persistent_m128",
    "small_bh_m128",
    "bt16_prepare",
    "bt16_prepare_beta_tma",
    "bt16_chain_m64_s7",
    "bt16_chain_m64_s8",
    "bt16_chain_m64_s9",
    "bt16_prepare_chain_m64_s8",
)

_FLASH_KDA_TARGETS: tuple[FlashKDATarget, ...] = (
    "sm100a",
    "sm100f",
    "sm103a",
)
_FLASH_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm100f": "-DFLASHINFER_FLASH_KDA_TARGET_FAMILY=100",
    "sm103a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=3",
}

_FLASH_KDA_GENERATED_METADATA_NAME = "flashkda_generated_variant_metadata.json"
_FLASH_KDA_GENERATED_RECEIPT_NAME = "flashkda_generated_generation_receipt.json"
_FLASH_KDA_GENERATED_CLOSURE_ROLES = (
    "selector_binding",
    "sanitized_body",
    "abi_wrapper",
    "generated_common_wrapper",
    "bt16_descriptor_common",
    "public_common_include",
)
_FLASH_KDA_GENERATED_ARCH_TARGETS: Mapping[str, GeneratedFlashKDATarget] = {
    "sm_100a": "sm100a",
    "sm_103a": "sm103a",
}
_FLASH_KDA_GENERATED_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_FLASH_KDA_GENERATED_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_FLASH_KDA_TARGET_MINOR=3",
}
_FLASH_KDA_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_FLASH_KDA_CACHE_IDENT_RE = re.compile(r"[0-9a-f]{10}")


class _GeneratedFlashKDASelectorNotFoundError(ValueError):
    """The sealed generated portfolio does not cover one valid runtime shape."""


@dataclass(frozen=True)
class GeneratedFlashKDASource:
    """One ordered, hash-sealed member of a generated module's source closure."""

    role: str
    path: str
    sha256: str


GeneratedFlashKDASelectorKey = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[tuple[str, object], ...],
]


@dataclass(frozen=True)
class GeneratedFlashKDAPhysicalSelector:
    """One receipt-backed semantic key for an exact physical module."""

    arch: str
    route: str
    route_role: str
    abi_family: str
    state_mode: str
    family_specialization: tuple[tuple[str, object], ...]
    selector_key_sha256: str
    selector_key_json: str

    @property
    def key(self) -> GeneratedFlashKDASelectorKey:
        return (
            self.arch,
            self.route,
            self.route_role,
            self.abi_family,
            self.state_mode,
            self.family_specialization,
        )


@dataclass(frozen=True)
class GeneratedFlashKDAModule:
    """Receipt-closed physical FlashKDA module selected by the dispatcher."""

    variant_id: str
    arch: str
    target: GeneratedFlashKDATarget
    module_ident: str
    abi_family: str
    abi_variant: str
    state_mode: str
    route_role: str
    binding_relpath: str
    body_relpath: str
    abi_wrapper_relpath: str
    launch_contract_sha256: str
    source_closure_sha256: str
    cache_ident: str
    source_closure: tuple[GeneratedFlashKDASource, ...]
    physical_selectors: tuple[GeneratedFlashKDAPhysicalSelector, ...]


# Keep every frozen cache key tied to its complete generated-plus-integration
# implementation. This prevents an installed JIT/AOT cache from satisfying a
# refreshed export or binding specialization after an in-place package upgrade.
_FLASH_KDA_MODULE_IDENTS = {
    "m64": "535ed3e2ce",
    "m128": "ec6d5fdb56",
    "m128_tensor_state_decay": "9614ba2d29",
    "m128_h12_short": "47c46019cc",
    "m128_h12_long": "b813a7edd3",
    "m128_n16": "a00baf7312",
    # Generated body, binding, and shared binding header, separated by NUL
    # bytes without a trailing separator. Keep this route's cache key tied to
    # all compiled content.
    "m128_n16_checkpoint": "ef6484d679",
    "m128_n16_short": "3f90fe2347",
    "persistent_m128": "4a2c82bde2",
    "piece_persistent_m128": "dd8e3a5ca0",
    "small_bh_m128": "b2593f3697",
    "bt16_prepare": "2c6cc4c1f6",
    "bt16_prepare_beta_tma": "d9394ce430",
    "bt16_chain_m64_s7": "350dbb8897",
    "bt16_chain_m64_s8": "9e1ea1ef2d",
    "bt16_chain_m64_s9": "e83ce16115",
    "bt16_prepare_chain_m64_s8": "6c392ef667",
}

_FLASH_KDA_BINDING_STEMS = {
    "m64": "flashkda_bf16_fused_m64",
    "m128": "flashkda_bf16_fused_m128",
    "m128_tensor_state_decay": "flashkda_bf16_fused_m128",
    "m128_h12_short": "cake_flashkda_bf16_fused_m128_h12",
    "m128_h12_long": "cake_flashkda_bf16_fused_m128_h12",
    "m128_n16": "cake_flashkda_bf16_fused_m128_n16",
    "m128_n16_checkpoint": "flashkda_bf16_fused_m128_n16_checkpoint",
    "m128_n16_short": "cake_flashkda_bf16_fused_m128_n16",
    "persistent_m128": "cake_flashkda_bf16_persistent_m128",
    "piece_persistent_m128": "cake_flashkda_bf16_piece_persistent_m128",
    "small_bh_m128": "cake_flashkda_bf16_small_bh_m128",
    "bt16_prepare": "cake_flashkda_bf16_bt16_prepare",
    "bt16_prepare_beta_tma": "cake_flashkda_bf16_bt16_prepare_beta_tma",
    "bt16_chain_m64_s7": "cake_flashkda_bf16_bt16_chain_m64_s7",
    "bt16_chain_m64_s8": "cake_flashkda_bf16_bt16_chain_m64",
    "bt16_chain_m64_s9": "cake_flashkda_bf16_bt16_chain_m64_s9",
}

_FLASH_KDA_VARIANT_DEFINES = {
    "m128_n16_short": "-DFLASHINFER_FLASH_KDA_N16_SHORT=1",
    "m128_tensor_state_decay": "-DFLASHINFER_FLASH_KDA_TENSOR_STATE_DECAY=1",
    "m128_h12_short": "-DFLASHINFER_FLASH_KDA_H12_SHORT=1",
    "m128_h12_long": "-DFLASHINFER_FLASH_KDA_H12_LONG=1",
}


def _canonical_json_sha256(value: object) -> str:
    payload = _canonical_json(value).encode()
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _resolve_generated_source(csrc_dir: Path, relpath: str) -> Path:
    """Resolve one package-relative generated source without allowing escape."""

    candidate = Path(relpath)
    if (
        candidate.is_absolute()
        or candidate.parts[:2] != ("csrc", "kda")
        or ".." in candidate.parts
    ):
        raise ValueError(
            f"generated FlashKDA source path must be relative to csrc/kda: {relpath!r}"
        )
    resolved = csrc_dir.joinpath(*candidate.parts[2:])
    if not resolved.is_file():
        raise FileNotFoundError(f"generated FlashKDA source not found: {resolved}")
    return resolved


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _FLASH_KDA_SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase full SHA-256 digest")
    return value


def _parse_generated_selector_key(
    value: object,
    *,
    label: str,
) -> GeneratedFlashKDAPhysicalSelector:
    if not isinstance(value, dict) or set(value) != {
        "arch",
        "route",
        "route_role",
        "abi_family",
        "state_mode",
        "family_specialization_vector",
    }:
        raise ValueError(f"{label} has an unsupported selector_key schema")
    arch = value.get("arch")
    route = value.get("route")
    route_role = value.get("route_role")
    abi_family = value.get("abi_family")
    state_mode = value.get("state_mode")
    if not all(
        isinstance(item, str) and item
        for item in (arch, route, route_role, abi_family, state_mode)
    ):
        raise ValueError(f"{label} has an empty selector identity field")
    vector_value = value.get("family_specialization_vector")
    if not isinstance(vector_value, list):
        raise ValueError(f"{label} family_specialization_vector must be a list")
    vector: list[tuple[str, object]] = []
    seen_fields: set[str] = set()
    for field_index, item in enumerate(vector_value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
        ):
            raise ValueError(
                f"{label} specialization item {field_index} must be [name, value]"
            )
        field, field_value = item
        if field in seen_fields:
            raise ValueError(f"{label} repeats specialization field {field!r}")
        if not isinstance(field_value, (str, int, float, bool)):
            raise ValueError(
                f"{label} specialization field {field!r} has a non-scalar value"
            )
        if isinstance(field_value, float) and not math.isfinite(field_value):
            raise ValueError(f"{label} specialization field {field!r} is not finite")
        seen_fields.add(field)
        vector.append((field, field_value))
    selector_key_json = _canonical_json(value)
    return GeneratedFlashKDAPhysicalSelector(
        arch=arch,
        route=route,
        route_role=route_role,
        abi_family=abi_family,
        state_mode=state_mode,
        family_specialization=tuple(vector),
        selector_key_sha256=hashlib.sha256(selector_key_json.encode()).hexdigest(),
        selector_key_json=selector_key_json,
    )


def _parse_generated_physical_selector(
    value: object,
    *,
    label: str,
) -> GeneratedFlashKDAPhysicalSelector:
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "selector_key",
        "selector_key_sha256",
        "family_specialization",
    }:
        raise ValueError(f"{label} has an unsupported physical_selector schema")
    if value.get("schema_version") != 1:
        raise ValueError(f"{label} has an unsupported physical_selector version")
    selector = _parse_generated_selector_key(
        value.get("selector_key"), label=f"{label} selector_key"
    )
    selector_key_sha256 = _require_sha256(
        value.get("selector_key_sha256"), f"{label} selector_key_sha256"
    )
    if selector_key_sha256 != selector.selector_key_sha256:
        raise ValueError(f"{label} selector_key digest mismatch")
    specialization = value.get("family_specialization")
    if not isinstance(specialization, dict) or specialization != dict(
        selector.family_specialization
    ):
        raise ValueError(
            f"{label} family_specialization does not match its ordered vector"
        )
    return selector


@functools.cache
def get_flash_kda_generated_registry() -> Mapping[str, GeneratedFlashKDAModule]:
    """Load and verify the source-closed generated physical-module registry.

    The registry is intentionally lazy: importing FlashInfer or using the
    checkpoint fallback does not parse or compile the generated portfolio.
    On first generated access, every selector and every member of its ordered
    source closure is rehashed before any :class:`JitSpec` can be created.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    metadata_path = csrc_dir / _FLASH_KDA_GENERATED_METADATA_NAME
    receipt_path = csrc_dir / _FLASH_KDA_GENERATED_RECEIPT_NAME
    missing = [path for path in (metadata_path, receipt_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "generated FlashKDA registry is not materialized; missing "
            + ", ".join(str(path) for path in missing)
        )

    metadata = json.loads(metadata_path.read_text())
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("unsupported generated FlashKDA metadata schema")
    if metadata.get("physical_selector_schema_version") != 1:
        raise ValueError("unsupported generated FlashKDA physical-selector schema")
    metadata_selector_index_sha256 = _require_sha256(
        metadata.get("physical_selector_index_sha256"),
        "generated FlashKDA metadata physical_selector_index_sha256",
    )
    rows = metadata.get("variants")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generated FlashKDA metadata has no variants")
    if not isinstance(receipt, dict) or receipt.get("schema_version") != 1:
        raise ValueError("unsupported generated FlashKDA receipt schema")
    if receipt.get("status") != "passed":
        raise ValueError("generated FlashKDA receipt status is not passed")
    if receipt.get("optimization_level_one_absent") is not True:
        raise ValueError("generated FlashKDA receipt does not prove O1 absence")
    if receipt.get("public_confidentiality_scan") != "passed":
        raise ValueError(
            "generated FlashKDA receipt does not prove public-safe source content"
        )
    if receipt.get("source_closure_status") != "passed":
        raise ValueError("generated FlashKDA receipt is not source-closed")
    if receipt.get("physical_selector_schema_version") != 1:
        raise ValueError("generated FlashKDA receipt lacks physical-selector evidence")
    if receipt.get("physical_selector_collision_count") != 0:
        raise ValueError("generated FlashKDA receipt has selector collisions")
    if receipt.get("launch_contract_schema_version") != 2:
        raise ValueError("unsupported generated FlashKDA launch-contract schema")
    receipt_selector_index_sha256 = _require_sha256(
        receipt.get("physical_selector_index_sha256"),
        "generated FlashKDA receipt physical_selector_index_sha256",
    )
    if receipt_selector_index_sha256 != metadata_selector_index_sha256:
        raise ValueError(
            "generated FlashKDA metadata and receipt selector indexes differ"
        )
    for digest_name in (
        "dispatcher_contract_sha256",
        "runtime_compile_factory_contracts_sha256",
        "runtime_physical_sequences_sha256",
    ):
        _require_sha256(receipt.get(digest_name), f"generated FlashKDA {digest_name}")
    _require_sha256(
        receipt.get("source_closure_table_sha256"),
        "generated FlashKDA source_closure_table_sha256",
    )
    if receipt.get("variant_metadata_sha256") != _canonical_json_sha256(metadata):
        raise ValueError("generated FlashKDA metadata digest does not match receipt")
    if receipt.get("variant_count") != len(rows):
        raise ValueError("generated FlashKDA variant count does not match receipt")
    if receipt.get("binding_tu_count") != len(rows):
        raise ValueError("generated FlashKDA selector count does not match receipt")

    modules: dict[str, GeneratedFlashKDAModule] = {}
    arches: set[str] = set()
    body_paths: set[str] = set()
    abi_wrapper_relpaths: set[str] = set()
    closure_table: list[dict[str, str]] = []
    module_order: list[tuple[str, str]] = []
    selector_index: list[dict[str, object]] = []
    selector_owners: dict[str, str] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"generated FlashKDA variant {index} is not an object")
        label = f"generated FlashKDA variant {index}"
        arch = row.get("arch")
        if arch not in _FLASH_KDA_GENERATED_ARCH_TARGETS:
            raise ValueError(f"{label} has unsupported exact architecture: {arch!r}")
        target = _FLASH_KDA_GENERATED_ARCH_TARGETS[arch]
        arches.add(arch)

        variant_id = row.get("variant_id")
        module_ident = row.get("module_ident")
        binding_relpath = row.get("binding_relpath")
        body_relpath = row.get("body_relpath")
        abi_wrapper_relpath = row.get("abi_wrapper_relpath")
        abi_family = row.get("abi_family")
        abi_variant = row.get("abi_variant")
        state_mode = row.get("state_mode")
        launch_contract = row.get("launch_contract")
        if (
            not isinstance(launch_contract, dict)
            or launch_contract.get("schema_version") != 2
        ):
            raise ValueError(f"{label} has an unsupported launch contract")
        value_tma_rank = launch_contract.get("value_tma_rank")
        if (
            value_tma_rank == 0
            if abi_family in ("bt16_prepare", "affine_scan")
            else value_tma_rank in (3, 4)
        ) is not True:
            raise ValueError(f"{label} has an invalid sealed V TensorMap rank")
        route_role = (
            launch_contract.get("route_role")
            if isinstance(launch_contract, dict)
            else None
        )
        cache_ident = row.get("cache_ident")
        if not all(
            isinstance(value, str) and value
            for value in (
                variant_id,
                module_ident,
                binding_relpath,
                body_relpath,
                abi_wrapper_relpath,
                abi_family,
                abi_variant,
                state_mode,
                route_role,
            )
        ):
            raise ValueError(f"{label} has missing identity or source fields")
        if variant_id != f"{arch}:{module_ident}":
            raise ValueError(f"{label} variant_id is not arch:module_ident")
        if variant_id in modules:
            raise ValueError(f"duplicate generated FlashKDA variant: {variant_id}")
        if (
            not isinstance(cache_ident, str)
            or _FLASH_KDA_CACHE_IDENT_RE.fullmatch(cache_ident) is None
        ):
            raise ValueError(f"{label} has invalid source-closure cache_ident")
        launch_contract_sha256 = _require_sha256(
            row.get("launch_contract_sha256"), f"{label} launch_contract_sha256"
        )
        if launch_contract_sha256 != _canonical_json_sha256(launch_contract):
            raise ValueError(f"{label} launch-contract digest mismatch")
        if state_mode not in ("bf16", "bf16_f32_dependency", "fp32", "none"):
            raise ValueError(f"{label} has unsupported state mode")

        physical_selector_rows = row.get("physical_selectors")
        if not isinstance(physical_selector_rows, list) or not physical_selector_rows:
            raise ValueError(f"{label} has no receipt-backed physical selectors")
        physical_selectors_sha256 = _require_sha256(
            row.get("physical_selectors_sha256"),
            f"{label} physical_selectors_sha256",
        )
        sorted_selector_rows = sorted(physical_selector_rows, key=_canonical_json)
        if physical_selector_rows != sorted_selector_rows:
            raise ValueError(f"{label} physical selectors are not canonical-order")
        if physical_selectors_sha256 != _canonical_json_sha256(sorted_selector_rows):
            raise ValueError(f"{label} physical-selector digest mismatch")
        physical_selectors: list[GeneratedFlashKDAPhysicalSelector] = []
        observed_route_roles = (
            launch_contract.get("observed_route_roles")
            if isinstance(launch_contract, dict)
            else None
        )
        if not isinstance(observed_route_roles, list) or not all(
            isinstance(item, str) and item for item in observed_route_roles
        ):
            raise ValueError(f"{label} has no observed launch route roles")
        for selector_index_in_variant, selector_row in enumerate(
            physical_selector_rows
        ):
            selector_label = f"{label} physical selector {selector_index_in_variant}"
            selector = _parse_generated_physical_selector(
                selector_row, label=selector_label
            )
            if (
                selector.arch != arch
                or selector.abi_family != abi_family
                or selector.state_mode != state_mode
            ):
                raise ValueError(
                    f"{selector_label} disagrees with its variant identity"
                )
            semantic_route_role = f"{selector.route}:{selector.route_role}"
            if semantic_route_role not in observed_route_roles:
                raise ValueError(
                    f"{selector_label} route role is not launch-contract observed"
                )
            previous_owner = selector_owners.get(selector.selector_key_json)
            if previous_owner is not None:
                raise ValueError(
                    "generated FlashKDA physical selector collision: "
                    f"{selector.selector_key_json} maps to {previous_owner} "
                    f"and {variant_id}"
                )
            selector_owners[selector.selector_key_json] = variant_id
            selector_key = json.loads(selector.selector_key_json)
            selector_index.append(
                {
                    "selector_key": selector_key,
                    "selector_key_sha256": selector.selector_key_sha256,
                    "variant_id": variant_id,
                }
            )
            physical_selectors.append(selector)

        closure_rows = row.get("source_closure")
        if not isinstance(closure_rows, list) or len(closure_rows) != len(
            _FLASH_KDA_GENERATED_CLOSURE_ROLES
        ):
            raise ValueError(f"{label} does not have the complete ordered closure")
        closure: list[GeneratedFlashKDASource] = []
        closure_bytes: list[bytes] = []
        for expected_role, source_row in zip(
            _FLASH_KDA_GENERATED_CLOSURE_ROLES, closure_rows, strict=True
        ):
            if not isinstance(source_row, dict):
                raise ValueError(f"{label} has a non-object closure member")
            role = source_row.get("role")
            relpath = source_row.get("path")
            sha256 = _require_sha256(
                source_row.get("sha256"), f"{label} {expected_role} sha256"
            )
            if role != expected_role or not isinstance(relpath, str):
                raise ValueError(
                    f"{label} expected closure role {expected_role}, got {role!r}"
                )
            source_path = _resolve_generated_source(csrc_dir, relpath)
            source_bytes = source_path.read_bytes()
            if hashlib.sha256(source_bytes).hexdigest() != sha256:
                raise ValueError(f"{label} source digest mismatch: {relpath}")
            closure_bytes.append(source_bytes)
            closure.append(GeneratedFlashKDASource(role, relpath, sha256))

        if closure[0].path != binding_relpath:
            raise ValueError(f"{label} selector closure does not match binding_relpath")
        if closure[1].path != body_relpath:
            raise ValueError(f"{label} body closure does not match body_relpath")
        if closure[2].path != abi_wrapper_relpath:
            raise ValueError(
                f"{label} ABI wrapper closure does not match abi_wrapper_relpath"
            )
        if row.get("binding_sha256") != closure[0].sha256:
            raise ValueError(f"{label} binding digest is not source-closed")
        if row.get("body_sha256") != closure[1].sha256:
            raise ValueError(f"{label} body digest is not source-closed")
        calculated_closure_sha256 = hashlib.sha256(
            b"\0".join(closure_bytes)
        ).hexdigest()
        if row.get("source_closure_sha256") != calculated_closure_sha256:
            raise ValueError(f"{label} source_closure_sha256 is not content-closed")
        if cache_ident != calculated_closure_sha256[:10]:
            raise ValueError(f"{label} cache_ident does not seal its source closure")
        if any(
            "_o1" in value.lower()
            for value in (variant_id, module_ident, *[item.path for item in closure])
        ):
            raise ValueError(f"{label} contains a forbidden O1 variant or source")

        body_paths.add(body_relpath)
        abi_wrapper_relpaths.add(abi_wrapper_relpath)
        module_order.append((arch, module_ident))
        closure_table.append(
            {
                "variant_id": variant_id,
                "source_closure_sha256": calculated_closure_sha256,
            }
        )
        modules[variant_id] = GeneratedFlashKDAModule(
            variant_id=variant_id,
            arch=arch,
            target=target,
            module_ident=module_ident,
            abi_family=abi_family,
            abi_variant=abi_variant,
            state_mode=state_mode,
            route_role=route_role,
            binding_relpath=binding_relpath,
            body_relpath=body_relpath,
            abi_wrapper_relpath=abi_wrapper_relpath,
            launch_contract_sha256=launch_contract_sha256,
            source_closure_sha256=calculated_closure_sha256,
            cache_ident=cache_ident,
            source_closure=tuple(closure),
            physical_selectors=tuple(physical_selectors),
        )

    if arches != set(_FLASH_KDA_GENERATED_ARCH_TARGETS):
        raise ValueError("generated FlashKDA registry is missing an exact architecture")
    if module_order != sorted(module_order):
        raise ValueError(
            "generated FlashKDA variants are not ordered by (arch, module_ident)"
        )
    if receipt.get("unique_body_count") != len(body_paths):
        raise ValueError("generated FlashKDA unique body count does not match receipt")
    if receipt.get("abi_wrapper_count") != len(abi_wrapper_relpaths):
        raise ValueError("generated FlashKDA ABI wrapper count does not match receipt")
    if receipt.get("abi_wrapper_count") != 8:
        raise ValueError(
            "generated FlashKDA registry does not contain all eight ABI wrappers"
        )
    if receipt.get("source_closure_table_sha256") != _canonical_json_sha256(
        closure_table
    ):
        raise ValueError("generated FlashKDA source closure table digest differs")
    selector_index.sort(
        key=lambda row: (
            _canonical_json(row["selector_key"]),
            row["variant_id"],
        )
    )
    if receipt.get("physical_selector_count") != len(selector_index):
        raise ValueError("generated FlashKDA physical-selector count differs")
    if metadata_selector_index_sha256 != _canonical_json_sha256(selector_index):
        raise ValueError("generated FlashKDA physical-selector index digest differs")
    return MappingProxyType(modules)


@functools.cache
def get_flash_kda_generated_selector_registry() -> Mapping[
    GeneratedFlashKDASelectorKey, GeneratedFlashKDAModule
]:
    """Return the collision-free, receipt-backed semantic module index."""

    index: dict[GeneratedFlashKDASelectorKey, GeneratedFlashKDAModule] = {}
    for module in get_flash_kda_generated_registry().values():
        for selector in module.physical_selectors:
            if selector.key in index:
                # The source registry checks canonical JSON keys as well. Keep
                # this independent guard so Python equality cannot silently
                # collapse two differently encoded selector records.
                raise ValueError(
                    "generated FlashKDA selector registry is not collision-free"
                )
            index[selector.key] = module
    return MappingProxyType(index)


def get_flash_kda_generated_module_for_selector(
    selector_key: Mapping[str, object],
) -> GeneratedFlashKDAModule:
    """Resolve one exact module from a runtime-computed physical selector."""

    selector = _parse_generated_selector_key(
        dict(selector_key), label="generated FlashKDA runtime selector"
    )
    try:
        return get_flash_kda_generated_selector_registry()[selector.key]
    except KeyError as error:
        raise _GeneratedFlashKDASelectorNotFoundError(
            "unsupported generated FlashKDA physical selector: "
            f"{selector.selector_key_json}"
        ) from error


def load_flash_kda_generated_module_for_selector(
    selector_key: Mapping[str, object],
):
    """Resolve and load exactly one receipt-backed generated module."""

    module = get_flash_kda_generated_module_for_selector(selector_key)
    return load_flash_kda_generated_module(module.variant_id)


def get_flash_kda_generated_variant_ids(
    target: GeneratedFlashKDATarget,
) -> tuple[str, ...]:
    """Return receipt order for one exact target without creating JIT specs."""

    if target not in _FLASH_KDA_GENERATED_NVCC_FLAGS:
        raise ValueError(f"unsupported generated FlashKDA target: {target}")
    return tuple(
        variant_id
        for variant_id, module in get_flash_kda_generated_registry().items()
        if module.target == target
    )


def get_flash_kda_generated_uri(variant_id: str) -> str:
    """Return the full-source-closure JIT/AOT key for one physical module."""

    try:
        module = get_flash_kda_generated_registry()[variant_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported generated FlashKDA variant: {variant_id}"
        ) from error
    return (
        f"flash_kda_generated_{module.target}_{module.module_ident}_"
        f"{module.cache_ident}"
    )


@functools.cache
def gen_flash_kda_generated_module(variant_id: str) -> JitSpec:
    """Create one exact-target JIT spec containing only its selector TU."""

    try:
        module = get_flash_kda_generated_registry()[variant_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported generated FlashKDA variant: {variant_id}"
        ) from error
    csrc_dir = _get_flash_kda_csrc_dir()
    spec = gen_jit_spec(
        name=get_flash_kda_generated_uri(variant_id),
        sources=[_resolve_generated_source(csrc_dir, module.binding_relpath)],
        extra_cuda_cflags=[
            *_FLASH_KDA_GENERATED_NVCC_FLAGS[module.target],
            _FLASH_KDA_GENERATED_TARGET_DEFINE[module.target],
            "-DFLASHKDA_GENERATED_EMBEDDED_CUBIN=1",
            "-DTVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API=1",
            f"-DFLASHKDA_GENERATED_CUBIN_IDENT={module.module_ident}",
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_flash_kda_include_dir(),
        ],
        embedded_cubin_factory=functools.partial(
            prepare_generated_flash_kda_cubin,
            selector_path=_resolve_generated_source(csrc_dir, module.binding_relpath),
            body_path=_resolve_generated_source(csrc_dir, module.body_relpath),
            module_ident=module.module_ident,
            target=module.target,
        ),
    )
    logger.info(
        "Generated FlashKDA physical module %s JIT spec: %s",
        variant_id,
        spec.name,
    )
    return spec


@functools.cache
def load_flash_kda_generated_module(variant_id: str):
    """Build and load exactly one receipt-selected physical module."""

    return gen_flash_kda_generated_module(variant_id).build_and_load()


def get_flash_kda_uri(variant: FlashKDAVariant, target: FlashKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in FLASH_KDA_VARIANTS:
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    if target not in _FLASH_KDA_TARGETS:
        raise ValueError(f"unsupported FlashKDA target: {target}")
    module_ident = _FLASH_KDA_MODULE_IDENTS[variant]
    return f"flash_kda_bf16_{variant}_{module_ident}_{target}"


@functools.cache
def gen_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget) -> JitSpec:
    """Generate one legacy exact-SM100a or SM100-family JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. CUDA 12.8 uses the exact ``sm_100a`` target on B200. CUDA 12.9 and
    newer use one ``sm_100f`` target validated on CC 10.0 and CC 10.3.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    include_dir = _get_flash_kda_include_dir()
    uri = get_flash_kda_uri(variant, target)
    if variant == "bt16_prepare_chain_m64_s8":
        sources = [
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_chain_m64_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_chain_m64_binding.cu",
        ]
    else:
        sources = [csrc_dir / f"{_FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"]
    missing_sources = [source for source in sources if not source.exists()]
    if missing_sources:
        raise FileNotFoundError(
            f"FlashKDA binding source not found: {missing_sources[0]}"
        )

    extra_cuda_cflags = [
        *(
            [_FLASH_KDA_VARIANT_DEFINES[variant]]
            if variant in _FLASH_KDA_VARIANT_DEFINES
            else []
        ),
        *(
            ["-DFLASHINFER_FLASH_KDA_COMBINED_BT16=1"]
            if variant == "bt16_prepare_chain_m64_s8"
            else []
        ),
    ]
    spec = gen_kda_jit_spec(
        name=uri,
        sources=sources,
        target=target,
        target_define=_FLASH_KDA_TARGET_DEFINE[target],
        csrc_dir=csrc_dir,
        include_dir=include_dir,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    logger.info(f"Generated FlashKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_flash_kda_m64_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed N=1, H=64 two-CTA M64 module."""

    return gen_flash_kda_module("m64", target)


def gen_flash_kda_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the general packed/fixed M128 module."""

    return gen_flash_kda_module("m128", target)


def gen_flash_kda_m128_tensor_state_decay_module(
    target: FlashKDATarget,
) -> JitSpec:
    """Generate the full-tile SM103 tensor state-decay M128 module."""

    return gen_flash_kda_module("m128_tensor_state_decay", target)


def gen_flash_kda_m128_h12_short_module(target: FlashKDATarget) -> JitSpec:
    """Generate the short-sequence H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_short", target)


def gen_flash_kda_m128_h12_long_module(target: FlashKDATarget) -> JitSpec:
    """Generate the pair-packed-beta H12 N32 M128 module."""

    return gen_flash_kda_module("m128_h12_long", target)


def gen_flash_kda_m128_n16_module(target: FlashKDATarget) -> JitSpec:
    """Generate the H12 packed/fixed M128 module with a 16-token chunk."""

    return gen_flash_kda_module("m128_n16", target)


def gen_flash_kda_m128_n16_checkpoint_module(target: FlashKDATarget) -> JitSpec:
    """Generate the N16 M128 module with checkpoint TMA stores."""

    return gen_flash_kda_module("m128_n16_checkpoint", target)


def gen_flash_kda_m128_n16_short_module(target: FlashKDATarget) -> JitSpec:
    """Generate the generic one-tile M128 module with one N16 stage."""

    return gen_flash_kda_module("m128_n16_short", target)


def gen_flash_kda_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the SM100-only static-binned persistent M128 module."""

    return gen_flash_kda_module("persistent_m128", target)


def gen_flash_kda_piece_persistent_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the recurrence-piece persistent M128 module."""

    return gen_flash_kda_module("piece_persistent_m128", target)


def gen_flash_kda_small_bh_m128_module(target: FlashKDATarget) -> JitSpec:
    """Generate the fixed-layout small-BH owner/helper M128 module."""

    return gen_flash_kda_module("small_bh_m128", target)


def gen_flash_kda_bt16_prepare_module(target: FlashKDATarget) -> JitSpec:
    """Generate the scalar-beta BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare", target)


def gen_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget) -> JitSpec:
    """Generate the beta-TMA BT16 factor-preparation module."""

    return gen_flash_kda_module("bt16_prepare_beta_tma", target)


def gen_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget) -> JitSpec:
    """Generate the two-resident S7 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s7", target)


def gen_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget) -> JitSpec:
    """Generate the canonical S8 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s8", target)


def gen_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget) -> JitSpec:
    """Generate the underfilled-grid S9 BT16 recurrence-chain module."""

    return gen_flash_kda_module("bt16_chain_m64_s9", target)


def gen_flash_kda_bt16_prepare_chain_m64_s8_module(
    target: FlashKDATarget,
) -> JitSpec:
    """Generate the combined scalar-prepare plus S8 chain launcher."""

    return gen_flash_kda_module("bt16_prepare_chain_m64_s8", target)


@functools.cache
def load_flash_kda_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Build or load one physical, target-specific FlashKDA module."""

    module = gen_flash_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded FlashKDA {variant} {target} module")
    return module


def load_flash_kda_m64_module(target: FlashKDATarget):
    """Load the fixed N=1, H=64 two-CTA M64 module."""

    return load_flash_kda_module("m64", target)


def load_flash_kda_m128_module(target: FlashKDATarget):
    """Load the general packed/fixed M128 module."""

    return load_flash_kda_module("m128", target)


def load_flash_kda_m128_tensor_state_decay_module(target: FlashKDATarget):
    """Load the full-tile SM103 tensor state-decay M128 module."""

    return load_flash_kda_module("m128_tensor_state_decay", target)


def load_flash_kda_m128_h12_short_module(target: FlashKDATarget):
    """Load the short-sequence H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_short", target)


def load_flash_kda_m128_h12_long_module(target: FlashKDATarget):
    """Load the pair-packed-beta H12 N32 M128 module."""

    return load_flash_kda_module("m128_h12_long", target)


def load_flash_kda_m128_n16_module(target: FlashKDATarget):
    """Load the H12 packed/fixed M128 module with a 16-token chunk."""

    return load_flash_kda_module("m128_n16", target)


def load_flash_kda_m128_n16_short_module(target: FlashKDATarget):
    """Load the generic one-tile M128 module with one N16 stage."""

    return load_flash_kda_module("m128_n16_short", target)


def load_flash_kda_persistent_m128_module(target: FlashKDATarget):
    """Load the SM100-only static-binned persistent M128 module."""

    return load_flash_kda_module("persistent_m128", target)


def load_flash_kda_piece_persistent_m128_module(target: FlashKDATarget):
    """Load the recurrence-piece persistent M128 module."""

    return load_flash_kda_module("piece_persistent_m128", target)


def load_flash_kda_small_bh_m128_module(target: FlashKDATarget):
    """Load the fixed-layout small-BH owner/helper M128 module."""

    return load_flash_kda_module("small_bh_m128", target)


def load_flash_kda_bt16_prepare_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare", target)


def load_flash_kda_bt16_prepare_beta_tma_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare_beta_tma", target)


def load_flash_kda_bt16_chain_m64_s7_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s7", target)


def load_flash_kda_bt16_chain_m64_s8_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s8", target)


def load_flash_kda_bt16_chain_m64_s9_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_chain_m64_s9", target)


def load_flash_kda_bt16_prepare_chain_m64_s8_module(target: FlashKDATarget):
    return load_flash_kda_module("bt16_prepare_chain_m64_s8", target)


def get_flash_kda_prefill_module(variant: FlashKDAVariant, target: FlashKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_flash_kda_module(variant, target)


__all__ = [
    "FLASH_KDA_VARIANTS",
    "FlashKDATarget",
    "FlashKDAVariant",
    "GeneratedFlashKDAModule",
    "GeneratedFlashKDAPhysicalSelector",
    "GeneratedFlashKDASelectorKey",
    "GeneratedFlashKDASource",
    "GeneratedFlashKDATarget",
    "gen_flash_kda_bt16_chain_m64_s7_module",
    "gen_flash_kda_bt16_chain_m64_s8_module",
    "gen_flash_kda_bt16_chain_m64_s9_module",
    "gen_flash_kda_bt16_prepare_chain_m64_s8_module",
    "gen_flash_kda_bt16_prepare_beta_tma_module",
    "gen_flash_kda_bt16_prepare_module",
    "gen_flash_kda_m64_module",
    "gen_flash_kda_m128_module",
    "gen_flash_kda_m128_tensor_state_decay_module",
    "gen_flash_kda_m128_h12_short_module",
    "gen_flash_kda_m128_h12_long_module",
    "gen_flash_kda_m128_n16_module",
    "gen_flash_kda_m128_n16_checkpoint_module",
    "gen_flash_kda_m128_n16_short_module",
    "gen_flash_kda_piece_persistent_m128_module",
    "gen_flash_kda_persistent_m128_module",
    "gen_flash_kda_small_bh_m128_module",
    "gen_flash_kda_module",
    "gen_flash_kda_generated_module",
    "get_flash_kda_generated_registry",
    "get_flash_kda_generated_module_for_selector",
    "get_flash_kda_generated_selector_registry",
    "get_flash_kda_generated_uri",
    "get_flash_kda_generated_variant_ids",
    "get_flash_kda_prefill_module",
    "get_flash_kda_uri",
    "load_flash_kda_m64_module",
    "load_flash_kda_m128_module",
    "load_flash_kda_m128_tensor_state_decay_module",
    "load_flash_kda_m128_h12_short_module",
    "load_flash_kda_m128_h12_long_module",
    "load_flash_kda_m128_n16_module",
    "load_flash_kda_m128_n16_short_module",
    "load_flash_kda_piece_persistent_m128_module",
    "load_flash_kda_persistent_m128_module",
    "load_flash_kda_small_bh_m128_module",
    "load_flash_kda_bt16_chain_m64_s7_module",
    "load_flash_kda_bt16_chain_m64_s8_module",
    "load_flash_kda_bt16_chain_m64_s9_module",
    "load_flash_kda_bt16_prepare_chain_m64_s8_module",
    "load_flash_kda_bt16_prepare_beta_tma_module",
    "load_flash_kda_bt16_prepare_module",
    "load_flash_kda_module",
    "load_flash_kda_generated_module",
    "load_flash_kda_generated_module_for_selector",
]
