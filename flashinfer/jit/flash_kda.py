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
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping

from . import env as jit_env
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
from .utils import write_if_different

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
_FLASH_KDA_GENERATED_STATE_MODES = (
    "none",
    "bf16",
    "fp32",
    "bf16_f32_dependency",
)
_FLASH_KDA_GENERATED_ABI_VARIANTS = ("default", "serving", "vtile")
# Every ABI family has one target-owned wrapper header; the vtile variant of
# the fused M128 family is a preprocessor branch of the direct wrapper.
_FLASH_KDA_GENERATED_ABI_WRAPPER = {
    "direct_m128": "flashkda_generated_direct_m128_binding.cuh",
    "vtile_m128": "flashkda_generated_direct_m128_binding.cuh",
    "m64": "flashkda_generated_m64_binding.cuh",
    "scalar_lpt_m128": "flashkda_generated_scalar_lpt_m128_binding.cuh",
    "taskized_persistent_m128": "flashkda_generated_taskized_persistent_m128_binding.cuh",
    "small_bh_m128": "flashkda_generated_small_bh_m128_binding.cuh",
    "bt16_prepare": "flashkda_generated_bt16_prepare_binding.cuh",
    "bt16_chain": "flashkda_generated_bt16_chain_binding.cuh",
    "affine_scan": "flashkda_generated_affine_scan_binding.cuh",
}
# Shared headers compiled into every generated selector translation unit, in
# the order they enter the cache key after the rendered selector and the body.
_FLASH_KDA_GENERATED_SHARED_HEADERS = (
    "flashkda_generated_binding_common.cuh",
    "cake_flashkda_bt16_binding_common.cuh",
    "flashkda_binding_common.cuh",
)
_FLASH_KDA_LOCAL_INCLUDE_RE = re.compile(r'^\s*#include\s+"([^"]+)"', re.MULTILINE)
_FLASH_KDA_C_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class _GeneratedFlashKDASelectorNotFoundError(ValueError):
    """The generated portfolio does not cover one valid runtime shape."""


@dataclass(frozen=True)
class GeneratedFlashKDASelector:
    """One semantic key the dispatcher may resolve to a physical module."""

    route: str
    route_role: str
    specialization: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class GeneratedFlashKDAModule:
    """One physical generated FlashKDA module and its launch contract."""

    variant_id: str
    arch: str
    target: GeneratedFlashKDATarget
    module_ident: str
    body: str
    kernel: str
    threads: int
    smem_bytes: int
    use_pdl: bool
    state_mode: str
    abi_family: str
    abi_variant: str
    tma_tile_tokens: int
    value_rows: int
    value_tma_rank: int
    pair_packed_beta: bool
    affine_dependency: str
    selectors: tuple[GeneratedFlashKDASelector, ...]


GeneratedFlashKDASelectorKey = tuple[
    str,
    str,
    str,
    str,
    str,
    tuple[tuple[str, str, object], ...],
]

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


def _specialization_key(
    items: tuple[tuple[str, object], ...],
) -> tuple[tuple[str, str, object], ...]:
    """Order-free, type-aware specialization identity (``True`` is not ``1``)."""

    return tuple(sorted((name, type(value).__name__, value) for name, value in items))


def _selector_key(
    *,
    arch: str,
    route: str,
    route_role: str,
    abi_family: str,
    state_mode: str,
    specialization: tuple[tuple[str, object], ...],
) -> GeneratedFlashKDASelectorKey:
    return (
        arch,
        route,
        route_role,
        abi_family,
        state_mode,
        _specialization_key(specialization),
    )


def _parse_specialization(
    value: object, *, label: str
) -> tuple[tuple[str, object], ...]:
    """Accept either a mapping or an ordered ``[[name, value], ...]`` vector."""

    if isinstance(value, Mapping):
        items = list(value.items())
    elif isinstance(value, list) and all(
        isinstance(item, (list, tuple)) and len(item) == 2 for item in value
    ):
        items = [(item[0], item[1]) for item in value]
    else:
        raise ValueError(
            f"{label} specialization must be a mapping or [name, value] list"
        )
    seen: set[str] = set()
    for name, field_value in items:
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError(f"{label} has an empty or repeated specialization field")
        if not isinstance(field_value, (str, int, float, bool)):
            raise ValueError(f"{label} specialization field {name!r} is not scalar")
        seen.add(name)
    return tuple(items)


def _read_text(path: Path, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path.read_text()


@functools.cache
def get_flash_kda_generated_registry() -> Mapping[str, GeneratedFlashKDAModule]:
    """Load the generated physical-module table, keyed by ``arch:module_ident``.

    The table is lazy: importing FlashInfer or using the checkpoint fallback does
    not read it. Source files are read only when a module's JIT spec is created.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    metadata_path = csrc_dir / _FLASH_KDA_GENERATED_METADATA_NAME
    metadata = json.loads(_read_text(metadata_path, "generated FlashKDA module table"))
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 2:
        raise ValueError("unsupported generated FlashKDA module table schema")
    rows = metadata.get("modules")
    if not isinstance(rows, list) or not rows:
        raise ValueError("generated FlashKDA module table has no modules")

    modules: dict[str, GeneratedFlashKDAModule] = {}
    for index, row in enumerate(rows):
        label = f"generated FlashKDA module {index}"
        if not isinstance(row, dict):
            raise ValueError(f"{label} is not an object")
        arch = row.get("arch")
        if arch not in _FLASH_KDA_GENERATED_ARCH_TARGETS:
            raise ValueError(f"{label} has unsupported exact architecture: {arch!r}")
        module_ident = row.get("module_ident")
        if (
            not isinstance(module_ident, str)
            or _FLASH_KDA_C_IDENTIFIER_RE.fullmatch(module_ident) is None
        ):
            raise ValueError(f"{label} module_ident is not a C identifier")
        variant_id = f"{arch}:{module_ident}"
        if variant_id in modules:
            raise ValueError(f"duplicate generated FlashKDA module: {variant_id}")
        state_mode = row.get("state_mode")
        if state_mode not in _FLASH_KDA_GENERATED_STATE_MODES:
            raise ValueError(f"{label} has unsupported state mode {state_mode!r}")
        abi_family = row.get("abi_family")
        if abi_family not in _FLASH_KDA_GENERATED_ABI_WRAPPER:
            raise ValueError(f"{label} has unsupported ABI family {abi_family!r}")
        abi_variant = row.get("abi_variant")
        if abi_variant not in _FLASH_KDA_GENERATED_ABI_VARIANTS:
            raise ValueError(f"{label} has unsupported ABI variant {abi_variant!r}")
        if (state_mode == "none") != (abi_family in ("bt16_prepare", "affine_scan")):
            raise ValueError(f"{label} state mode disagrees with its ABI family")
        for name in ("body", "kernel", "affine_dependency"):
            if not isinstance(row.get(name), str) or not row[name]:
                raise ValueError(f"{label} field {name!r} must be a nonempty string")
        for name in (
            "threads",
            "smem_bytes",
            "tma_tile_tokens",
            "value_rows",
            "value_tma_rank",
        ):
            if not isinstance(row.get(name), int) or isinstance(row[name], bool):
                raise ValueError(f"{label} field {name!r} must be an integer")
        for name in ("use_pdl", "pair_packed_beta"):
            if not isinstance(row.get(name), bool):
                raise ValueError(f"{label} field {name!r} must be a boolean")
        selector_rows = row.get("selectors")
        if not isinstance(selector_rows, list) or not selector_rows:
            raise ValueError(f"{label} has no selectors")
        selectors = []
        for selector_index, selector_row in enumerate(selector_rows):
            selector_label = f"{label} selector {selector_index}"
            if not isinstance(selector_row, dict) or set(selector_row) != {
                "route",
                "route_role",
                "specialization",
            }:
                raise ValueError(f"{selector_label} has an unsupported schema")
            route = selector_row["route"]
            route_role = selector_row["route_role"]
            if not all(isinstance(item, str) and item for item in (route, route_role)):
                raise ValueError(f"{selector_label} has an empty route identity")
            selectors.append(
                GeneratedFlashKDASelector(
                    route=route,
                    route_role=route_role,
                    specialization=_parse_specialization(
                        selector_row["specialization"], label=selector_label
                    ),
                )
            )
        modules[variant_id] = GeneratedFlashKDAModule(
            variant_id=variant_id,
            arch=arch,
            target=_FLASH_KDA_GENERATED_ARCH_TARGETS[arch],
            module_ident=module_ident,
            body=row["body"],
            kernel=row["kernel"],
            threads=row["threads"],
            smem_bytes=row["smem_bytes"],
            use_pdl=row["use_pdl"],
            state_mode=state_mode,
            abi_family=abi_family,
            abi_variant=abi_variant,
            tma_tile_tokens=row["tma_tile_tokens"],
            value_rows=row["value_rows"],
            value_tma_rank=row["value_tma_rank"],
            pair_packed_beta=row["pair_packed_beta"],
            affine_dependency=row["affine_dependency"],
            selectors=tuple(selectors),
        )
    return MappingProxyType(modules)


@functools.cache
def get_flash_kda_generated_selector_registry() -> Mapping[
    GeneratedFlashKDASelectorKey, GeneratedFlashKDAModule
]:
    """Return the collision-free tuple-keyed selector index."""

    index: dict[GeneratedFlashKDASelectorKey, GeneratedFlashKDAModule] = {}
    for module in get_flash_kda_generated_registry().values():
        for selector in module.selectors:
            key = _selector_key(
                arch=module.arch,
                route=selector.route,
                route_role=selector.route_role,
                abi_family=module.abi_family,
                state_mode=module.state_mode,
                specialization=selector.specialization,
            )
            previous = index.get(key)
            if previous is not None:
                raise ValueError(
                    "generated FlashKDA selector collision: "
                    f"{key} maps to {previous.variant_id} and {module.variant_id}"
                )
            index[key] = module
    return MappingProxyType(index)


_FLASH_KDA_SELECTOR_FIELDS = frozenset(
    {
        "arch",
        "route",
        "route_role",
        "abi_family",
        "state_mode",
        "family_specialization_vector",
    }
)


def get_flash_kda_generated_module_for_selector(
    selector_key: Mapping[str, object],
) -> GeneratedFlashKDAModule:
    """Resolve one exact module from a runtime-computed physical selector."""

    if set(selector_key) != _FLASH_KDA_SELECTOR_FIELDS:
        raise ValueError(
            "generated FlashKDA runtime selector has an unsupported schema"
        )
    identity = tuple(
        selector_key[name]
        for name in ("arch", "route", "route_role", "abi_family", "state_mode")
    )
    if not all(isinstance(item, str) and item for item in identity):
        raise ValueError(
            "generated FlashKDA runtime selector has an empty identity field"
        )
    arch, route, route_role, abi_family, state_mode = identity
    key = _selector_key(
        arch=arch,
        route=route,
        route_role=route_role,
        abi_family=abi_family,
        state_mode=state_mode,
        specialization=_parse_specialization(
            selector_key["family_specialization_vector"],
            label="generated FlashKDA runtime selector",
        ),
    )
    try:
        return get_flash_kda_generated_selector_registry()[key]
    except KeyError as error:
        raise _GeneratedFlashKDASelectorNotFoundError(
            f"unsupported generated FlashKDA physical selector: {key}"
        ) from error


def load_flash_kda_generated_module_for_selector(
    selector_key: Mapping[str, object],
):
    """Resolve and load exactly one generated module."""

    module = get_flash_kda_generated_module_for_selector(selector_key)
    return load_flash_kda_generated_module(module.variant_id)


def get_flash_kda_generated_variant_ids(
    target: GeneratedFlashKDATarget,
) -> tuple[str, ...]:
    """Return table order for one exact target without creating JIT specs."""

    if target not in _FLASH_KDA_GENERATED_NVCC_FLAGS:
        raise ValueError(f"unsupported generated FlashKDA target: {target}")
    return tuple(
        variant_id
        for variant_id, module in get_flash_kda_generated_registry().items()
        if module.target == target
    )


def _generated_module(variant_id: str) -> GeneratedFlashKDAModule:
    try:
        return get_flash_kda_generated_registry()[variant_id]
    except KeyError as error:
        raise ValueError(
            f"unsupported generated FlashKDA variant: {variant_id}"
        ) from error


def render_flash_kda_generated_binding(
    module: GeneratedFlashKDAModule, *, target: GeneratedFlashKDATarget
) -> str:
    """Render the selector translation unit for one module and exact target."""

    target_minor = {"sm100a": 0, "sm103a": 3}[target]
    state_mode = {
        "none": "FLASHKDA_GENERATED_STATE_NONE",
        "bf16": "FLASHKDA_GENERATED_STATE_BF16",
        "fp32": "FLASHKDA_GENERATED_STATE_FP32",
        "bf16_f32_dependency": "FLASHKDA_GENERATED_STATE_BF16_F32_DEPENDENCY",
    }[module.state_mode]
    lines = [
        "/* Selector translation unit rendered by flashinfer.jit.flash_kda. */",
        "#ifndef FLASHINFER_FLASH_KDA_TARGET_MINOR",
        '#error "JIT spec must define FLASHINFER_FLASH_KDA_TARGET_MINOR"',
        "#endif",
        f"static_assert(FLASHINFER_FLASH_KDA_TARGET_MINOR == {target_minor},",
        '              "binding compiled for the wrong exact target");',
        "",
        f'#define FLASHKDA_GENERATED_BODY_FILE "{module.body}"',
        f"#define FLASHKDA_GENERATED_KERNEL {module.kernel}",
        f"#define FLASHKDA_GENERATED_THREADS {module.threads}",
        f"#define FLASHKDA_GENERATED_SMEM_BYTES {module.smem_bytes}",
        f"#define FLASHKDA_GENERATED_USE_PDL {int(module.use_pdl)}",
        f"#define FLASHKDA_GENERATED_STATE_MODE {state_mode}",
        "#define FLASHKDA_GENERATED_ABI_VARIANT "
        f"FLASHKDA_GENERATED_VARIANT_{module.abi_variant.upper()}",
        f"#define FLASHKDA_GENERATED_TMA_TILE_TOKENS {module.tma_tile_tokens}",
        f"#define FLASHKDA_GENERATED_VALUE_ROWS {module.value_rows}",
        f"#define FLASHKDA_GENERATED_VALUE_TMA_RANK {module.value_tma_rank}",
        f"#define FLASHKDA_GENERATED_PAIR_PACKED_BETA {int(module.pair_packed_beta)}",
        "#define FLASHKDA_GENERATED_AFFINE_DEPENDENCY "
        f"FLASHKDA_GENERATED_AFFINE_{module.affine_dependency.upper()}",
        f'#include "{_FLASH_KDA_GENERATED_ABI_WRAPPER[module.abi_family]}"',
        "",
    ]
    return "\n".join(lines)


@functools.cache
def _flash_kda_generated_closure(variant_id: str) -> tuple[Path, str, str]:
    """Return (body path, rendered selector, content cache ident) for one module.

    The cache ident seals every byte the selector translation unit compiles:
    the rendered selector, the generated body, its ABI wrapper, and the shared
    binding headers. A changed body or integration header therefore cannot
    reuse a stale JIT/AOT artifact.
    """

    module = _generated_module(variant_id)
    csrc_dir = _get_flash_kda_csrc_dir()
    body_path = csrc_dir / module.body
    selector = render_flash_kda_generated_binding(module, target=module.target)
    closure = [
        selector.encode(),
        _read_text(body_path, f"generated FlashKDA body for {variant_id}").encode(),
        _read_text(
            csrc_dir / _FLASH_KDA_GENERATED_ABI_WRAPPER[module.abi_family],
            f"generated FlashKDA ABI wrapper for {variant_id}",
        ).encode(),
        *(
            _read_text(csrc_dir / name, "generated FlashKDA shared header").encode()
            for name in _FLASH_KDA_GENERATED_SHARED_HEADERS
        ),
    ]
    cache_ident = hashlib.sha256(b"\0".join(closure)).hexdigest()[:10]
    return body_path, selector, cache_ident


def get_flash_kda_generated_uri(variant_id: str) -> str:
    """Return the content-derived JIT/AOT key for one physical module."""

    module = _generated_module(variant_id)
    _, _, cache_ident = _flash_kda_generated_closure(variant_id)
    uri = f"flash_kda_generated_{module.target}_{module.module_ident}_{cache_ident}"
    if module.abi_family == "direct_m128" and module.abi_variant == "serving":
        # Serving direct-M128 retains its native prepared-launch ABI while the
        # kernel body is compiled by the same NVRTC path as the source. Keep it
        # distinct from older direct-source artifacts in the JIT cache.
        uri += "_direct_nvrtc_v1"
    return uri


@functools.cache
def gen_flash_kda_generated_module(variant_id: str) -> JitSpec:
    """Create one exact-target JIT spec containing only its rendered selector TU."""

    module = _generated_module(variant_id)
    csrc_dir = _get_flash_kda_csrc_dir()
    body_path, selector, _ = _flash_kda_generated_closure(variant_id)
    uri = get_flash_kda_generated_uri(variant_id)
    selector_path = (
        jit_env.FLASHINFER_GEN_SRC_DIR / uri / f"{module.module_ident}_binding.cu"
    )
    write_if_different(selector_path, selector)
    direct_serving = (
        module.abi_family == "direct_m128" and module.abi_variant == "serving"
    )
    torch_include_paths: list[Path] = []
    torch_ldflags: list[str] | None = None
    if direct_serving:
        from torch.utils.cpp_extension import include_paths, library_paths

        torch_include_paths = [Path(path) for path in include_paths()]
        torch_ldflags = [
            *(f"-L{path}" for path in library_paths()),
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-ltorch_python",
        ]
    embedded_flags = [
        "-DFLASHKDA_GENERATED_EMBEDDED_CUBIN=1",
        "-DTVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API=1",
        f"-DFLASHKDA_GENERATED_CUBIN_IDENT={module.module_ident}",
    ]
    spec = gen_jit_spec(
        name=uri,
        sources=[selector_path],
        extra_cuda_cflags=[
            *_FLASH_KDA_GENERATED_NVCC_FLAGS[module.target],
            _FLASH_KDA_GENERATED_TARGET_DEFINE[module.target],
            *embedded_flags,
            *(
                [
                    "-std=c++20",
                    "-DFLASHKDA_GENERATED_DIRECT_SOURCE_ABI=1",
                    "-UPy_LIMITED_API",
                ]
                if direct_serving
                else []
            ),
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_flash_kda_include_dir(),
            *torch_include_paths,
        ],
        extra_ldflags=torch_ldflags,
        embedded_cubin_factory=functools.partial(
            prepare_generated_flash_kda_cubin,
            body_path=body_path,
            kernel_name=module.kernel,
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
    """Build and load exactly one table-selected physical module."""

    return gen_flash_kda_generated_module(variant_id).build_and_load()


@functools.cache
def _load_flash_kda_generated_direct_python_factory(variant_id: str):
    """Resolve the native prepared-launch factory from the exact loaded module."""

    module = load_flash_kda_generated_module(variant_id)
    factory = module.make_direct_python_launcher
    # The factory calls the CPython C API and therefore must retain the GIL.
    # It runs before the optional beta copy and returns a native callable that
    # submits the exact PyTorch copy and prepared kernel without returning to
    # Python between the two GPU launches.
    factory.release_gil = False
    return module, factory


def _flash_kda_static_sources(variant: FlashKDAVariant) -> list[Path]:
    csrc_dir = _get_flash_kda_csrc_dir()
    if variant == "bt16_prepare_chain_m64_s8":
        return [
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_chain_m64_binding.cu",
            csrc_dir / "cake_flashkda_bf16_bt16_prepare_chain_m64_binding.cu",
        ]
    return [csrc_dir / f"{_FLASH_KDA_BINDING_STEMS[variant]}_binding.cu"]


def _flash_kda_static_flags(variant: FlashKDAVariant) -> list[str]:
    flags = []
    if variant in _FLASH_KDA_VARIANT_DEFINES:
        flags.append(_FLASH_KDA_VARIANT_DEFINES[variant])
    if variant == "bt16_prepare_chain_m64_s8":
        flags.append("-DFLASHINFER_FLASH_KDA_COMBINED_BT16=1")
    return flags


@functools.cache
def _flash_kda_static_ident(variant: FlashKDAVariant) -> str:
    """Content-derived cache ident of one static schedule's compiled closure.

    The closure is the variant's binding sources plus every ``csrc/kda`` header
    or body they include, directly or transitively, in first-encounter order,
    followed by the variant's compile definitions. Target-owned infrastructure
    outside ``csrc/kda`` (``tvm_ffi_utils.h``) is part of the FlashInfer build,
    not of the schedule, and is excluded.
    """

    csrc_dir = _get_flash_kda_csrc_dir()
    pending = _flash_kda_static_sources(variant)
    ordered: list[Path] = []
    seen: set[Path] = set()
    while pending:
        path = pending.pop(0)
        if path in seen:
            continue
        seen.add(path)
        text = _read_text(path, f"FlashKDA {variant} source")
        ordered.append(path)
        for name in _FLASH_KDA_LOCAL_INCLUDE_RE.findall(text):
            included = csrc_dir / name
            if included.is_file() and included not in seen:
                pending.append(included)
    payload = b"\0".join(
        [
            *(path.read_bytes() for path in ordered),
            *(flag.encode() for flag in _flash_kda_static_flags(variant)),
        ]
    )
    return hashlib.sha256(payload).hexdigest()[:10]


def get_flash_kda_uri(variant: FlashKDAVariant, target: FlashKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in FLASH_KDA_VARIANTS:
        raise ValueError(f"unsupported FlashKDA variant: {variant}")
    if target not in _FLASH_KDA_TARGETS:
        raise ValueError(f"unsupported FlashKDA target: {target}")
    return f"flash_kda_bf16_{variant}_{_flash_kda_static_ident(variant)}_{target}"


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
    spec = gen_kda_jit_spec(
        name=uri,
        sources=_flash_kda_static_sources(variant),
        target=target,
        target_define=_FLASH_KDA_TARGET_DEFINE[target],
        csrc_dir=csrc_dir,
        include_dir=include_dir,
        extra_cuda_cflags=_flash_kda_static_flags(variant),
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
    "GeneratedFlashKDASelector",
    "GeneratedFlashKDASelectorKey",
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
    "render_flash_kda_generated_binding",
]
