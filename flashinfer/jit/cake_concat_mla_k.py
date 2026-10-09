"""JIT loader for the source-only Blackwell concat MLA K backend."""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, Mapping

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

CakeConcatMLAKTarget = Literal["sm100f", "sm103a", "sm107a"]
# The single delivered module is arch-neutral source text (vector copy, no
# tcgen05/TMA); the manifest pins the sm_103a render but each target below
# compiles the same two translation units with its own flags.
_TARGET_FLAGS = {
    "sm100f": sm100f_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
    "sm107a": sm107a_nvcc_flags,
}

_MANIFEST_NAME = "cake_concat_mla_k_import_manifest.json"
_SCHEMA = "cake.library_export.v5"
_RENDER_ARCH = "sm_103a"
_MODULE_NAME_RE = re.compile(r"cake_concat_mla_k_[0-9a-f]{20}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_EXPECTED_CONTRACT = {
    "arches": [_RENDER_ARCH],
    "backend": "cake",
    "correctness": "byte_exact_copy_and_broadcast",
    "dtypes": [
        "bfloat16",
        "float16",
        "float8_e4m3fn",
        "float8_e5m2",
    ],
    "fixed_shape": {
        "nope_dim": 128,
        "num_heads": 128,
        "output_dim": 192,
        "rope_dim": 64,
    },
    "input_layouts": ["contiguous", "nope_strided", "both_strided"],
    "integration_files": [
        "flashinfer/concat_ops.py",
        "flashinfer/jit/cake_concat_mla_k.py",
    ],
    # The host wrapper mirrors this launch rule: one CTA copies two 1-byte
    # tokens or one 2-byte token (six 16-byte vectors per thread either way).
    "launch_policy": {
        "block": [512, 1, 1],
        "grid": "ceil(tokens / tokens_per_cta[element_bytes]), 1, 1",
        "min_blocks_per_sm": 2,
        "tokens_per_cta": {"element_bytes_1": 2, "element_bytes_2": 1},
    },
    "mutation": "caller_owned_k_in_place",
    "operator": "concat_mla_k",
    "output_layouts": [
        "caller_owned_uninitialized",
        "caller_owned_leading_strided_uninitialized",
    ],
    "public_api": "flashinfer.concat_ops.concat_mla_k",
    "signature": "concat_mla_k(k, k_nope, k_rope) -> None",
    "stages": ["vector_copy"],
    "stages_by_arch": {_RENDER_ARCH: ["vector_copy"]},
}
_EXPECTED_BUILD_CONTRACT = {
    "architecture_source": "modules[].arch",
    "binary_payloads": False,
    "source_license_header_sha256": (
        "dcf2449c2d70596e2c32bf1fa23720023965ff05857b85772decee99d830bf2f"
    ),
    "target_infrastructure": {
        "binding_runtime": "flashinfer_tvm_ffi_utils",
        "headers_owned_by_target": True,
        "required_headers": ["tvm_ffi_utils.h"],
    },
    "translation_unit_model": "separate_device_and_binding",
}
_EXPECTED_ARG_PLAN = [
    ["buffer", "k"],
    ["buffer", "k_nope"],
    ["buffer", "k_rope"],
    ["parameter", "element_bytes"],
    ["parameter", "k_stride_0_bytes"],
    ["parameter", "k_stride_1_bytes"],
    ["parameter", "k_nope_stride_0_bytes"],
    ["parameter", "k_nope_stride_1_bytes"],
    ["parameter", "k_rope_stride_0_bytes"],
    ["parameter", "tokens"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]
_EXPECTED_LAUNCH = {
    "block": [512, 1, 1],
    "cluster": [1, 1, 1],
    "cluster_scheduling_policy": "spread",
    "cooperative": False,
    "dynamic_smem_bytes": 0,
    "max_persistent_clusters": None,
    "persistent_ctas_per_sm": 1,
    "use_pdl": False,
}
_EXPECTED_BINDING_INFRASTRUCTURE = {
    "runtime": "flashinfer_tvm_ffi_utils",
    "target_owned_headers": ["tvm_ffi_utils.h"],
}
_EXPECTED_ROUTE_TEMPLATE = "flashinfer_blackwell_concat_mla_k_vector_copy"
_HASH_KEYS = ("arg_plan_sha256", "closure_sha256")


@dataclass(frozen=True)
class CakeConcatMLAKModuleSpec:
    """Verified source closure, cache identity and launch rule of the exported module."""

    module_ident: str
    closure_sha256: str
    device_path: Path
    binding_path: Path
    tokens_per_cta: Mapping[int, int]
    block: tuple[int, int, int]


def _require_manifest(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake concat MLA K import manifest: {message}")


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "concat_mla"
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "concat_mla"
    for candidate in (installed, checkout):
        if (candidate / _MANIFEST_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        "Cake concat MLA K sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def _verify_source(
    csrc_dir: Path, item: object, expected_path: str, label: str
) -> Path:
    _require_manifest(isinstance(item, dict), f"{label} must be an object")
    assert isinstance(item, dict)
    path_value = item.get("path")
    _require_manifest(
        path_value == expected_path, f"{label}.path must be {expected_path}"
    )
    relative = PurePosixPath(expected_path)
    _require_manifest(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "concat_mla")
        and len(relative.parts) == 3,
        f"{label}.path must name one csrc/concat_mla file",
    )
    path = csrc_dir / relative.name
    _require_manifest(path.is_file(), f"{label}.path does not exist: {path}")
    sha256_value = item.get("sha256")
    _require_manifest(
        isinstance(sha256_value, str)
        and _SHA256_RE.fullmatch(sha256_value) is not None,
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    _require_manifest(
        actual == sha256_value,
        f"{label}.sha256 mismatch: {actual} != {sha256_value}",
    )
    _require_manifest(item.get("bytes") == len(payload), f"{label}.bytes mismatch")
    return path


def _compact_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


@functools.cache
def get_cake_concat_mla_k_module_spec() -> CakeConcatMLAKModuleSpec:
    """Read and verify the complete source-only module receipt."""

    csrc_dir = _get_csrc_dir()
    payload: Any = json.loads((csrc_dir / _MANIFEST_NAME).read_text())
    _require_manifest(isinstance(payload, dict), "root must be an object")
    _require_manifest(payload.get("schema") == _SCHEMA, "schema")
    _require_manifest(payload.get("producer") == "cake", "producer")
    _require_manifest(payload.get("public_namespace") == "cake", "public_namespace")
    _require_manifest(payload.get("artifact_kind") == "source_only", "artifact_kind")
    _require_manifest(payload.get("library") == "flashinfer", "library")
    _require_manifest(payload.get("name") == "cake_concat_mla_k", "name")
    _require_manifest(payload.get("sequences") == [], "sequences")
    _require_manifest(payload.get("contract") == _EXPECTED_CONTRACT, "contract")
    _require_manifest(
        payload.get("build_contract") == _EXPECTED_BUILD_CONTRACT,
        "build_contract",
    )

    modules = payload.get("modules")
    _require_manifest(isinstance(modules, list) and len(modules) == 1, "modules")
    module = modules[0]
    _require_manifest(isinstance(module, dict), "modules[0] must be an object")
    name = module.get("name")
    _require_manifest(
        isinstance(name, str) and _MODULE_NAME_RE.fullmatch(name) is not None,
        "modules[0].name",
    )
    assert isinstance(name, str)
    _require_manifest(module.get("arch") == _RENDER_ARCH, "modules[0].arch")
    _require_manifest(module.get("role") == "kernel", "modules[0].role")
    _require_manifest(module.get("ffi_entry") == "run", "modules[0].ffi_entry")
    _require_manifest(
        module.get("kernel_symbol") == f"kernel_{name}", "modules[0].kernel_symbol"
    )
    module_ident = module.get("module_ident")
    _require_manifest(
        module_ident == f"{name}_{_RENDER_ARCH}", "modules[0].module_ident"
    )
    assert isinstance(module_ident, str)
    # FlashInfer owns the per-target flags; the render carries none.
    _require_manifest(module.get("compile_flags") == [], "modules[0].compile_flags")
    _require_manifest(module.get("tma_abi") == "grid_constant", "modules[0].tma_abi")
    _require_manifest(module.get("specializations") == {}, "modules[0].specializations")
    _require_manifest(
        module.get("noncontiguous_tensors") == ["k", "k_nope", "k_rope"],
        "modules[0].noncontiguous_tensors",
    )
    _require_manifest(
        module.get("binding_infrastructure") == _EXPECTED_BINDING_INFRASTRUCTURE,
        "modules[0].binding_infrastructure",
    )
    _require_manifest(
        module.get("arg_plan") == _EXPECTED_ARG_PLAN, "modules[0].arg_plan"
    )
    _require_manifest(
        module.get("arg_plan_sha256") == _compact_sha256(_EXPECTED_ARG_PLAN),
        "modules[0].arg_plan_sha256",
    )
    _require_manifest(module.get("launch") == _EXPECTED_LAUNCH, "modules[0].launch")
    route = module.get("route")
    _require_manifest(
        isinstance(route, dict)
        and route.get("template") == _EXPECTED_ROUTE_TEMPLATE
        and route.get("stage") == "vector_copy"
        and route.get("specialization") == {},
        "modules[0].route",
    )
    source_build = module.get("source_build")
    _require_manifest(
        isinstance(source_build, dict) and source_build.get("backend") == "cuda_cpp",
        "modules[0].source_build",
    )
    device_value = f"csrc/concat_mla/{name}_kernel.cu"
    binding_value = f"csrc/concat_mla/{name}_binding.cu"
    _require_manifest(
        module.get("translation_units")
        == {
            "binding": binding_value,
            "compile_separately": True,
            "device": device_value,
        },
        "modules[0].translation_units",
    )
    closure = module.get("closure")
    _require_manifest(
        isinstance(closure, list) and len(closure) == 2,
        "modules[0].closure",
    )
    by_path = {
        item.get("path"): item
        for item in closure
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    _require_manifest(device_value in by_path, "device closure missing")
    _require_manifest(binding_value in by_path, "binding closure missing")
    device_path = _verify_source(
        csrc_dir, by_path[device_value], device_value, "modules[0].device"
    )
    binding_path = _verify_source(
        csrc_dir, by_path[binding_value], binding_value, "modules[0].binding"
    )
    files = payload.get("files")
    _require_manifest(isinstance(files, list) and len(files) == 2, "files")
    file_receipts = {
        item.get("path"): (item.get("kind"), item.get("sha256"), item.get("bytes"))
        for item in files
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    _require_manifest(
        file_receipts
        == {
            device_value: (
                "device_source",
                by_path[device_value].get("sha256"),
                by_path[device_value].get("bytes"),
            ),
            binding_value: (
                "tvm_ffi_binding",
                by_path[binding_value].get("sha256"),
                by_path[binding_value].get("bytes"),
            ),
        },
        "files must match the complete module closure",
    )
    closure_sha256 = module.get("closure_sha256")
    _require_manifest(
        isinstance(closure_sha256, str)
        and _SHA256_RE.fullmatch(closure_sha256) is not None,
        "modules[0].closure_sha256",
    )
    assert isinstance(closure_sha256, str)
    identity_input = {
        key: value for key, value in module.items() if key not in _HASH_KEYS
    }
    _require_manifest(
        closure_sha256 == _compact_sha256(identity_input),
        "modules[0].closure_sha256 mismatch",
    )
    tokens_per_cta = _EXPECTED_CONTRACT["launch_policy"]["tokens_per_cta"]
    block = _EXPECTED_LAUNCH["block"]
    return CakeConcatMLAKModuleSpec(
        module_ident=module_ident,
        closure_sha256=closure_sha256,
        device_path=device_path,
        binding_path=binding_path,
        tokens_per_cta={
            1: int(tokens_per_cta["element_bytes_1"]),
            2: int(tokens_per_cta["element_bytes_2"]),
        },
        block=(int(block[0]), int(block[1]), int(block[2])),
    )


def cake_concat_mla_k_tokens_per_cta(element_bytes: int) -> int:
    """Token window one CTA copies (the manifest's ``launch_policy``)."""

    spec = get_cake_concat_mla_k_module_spec()
    try:
        return spec.tokens_per_cta[int(element_bytes)]
    except KeyError:
        raise ValueError(
            f"the Cake concat MLA K module has no launch rule for {element_bytes}-byte elements"
        ) from None


def cake_concat_mla_k_target(device) -> CakeConcatMLAKTarget:
    """Select the physical JIT target for one supported Blackwell device."""

    import torch

    from .cpp_ext import is_cuda_version_at_least

    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    if capability == (10, 0):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError(
                "Cake concat MLA K on compute capability 10.0 requires CUDA "
                "12.9 or newer for the sm_100f family target"
            )
        return "sm100f"
    if capability == (10, 3):
        return "sm103a"
    if capability == (10, 7):
        if not is_cuda_version_at_least("13.0"):
            raise RuntimeError(
                "Cake concat MLA K on compute capability 10.7 requires CUDA "
                "13.0 or newer for the sm_107a target"
            )
        return "sm107a"
    raise RuntimeError(
        "the Cake concat MLA K backend requires compute capability 10.0 "
        "(SM100f), 10.3 (SM103a) or 10.7 (SM107a), got "
        f"{capability[0]}.{capability[1]}"
    )


def get_cake_concat_mla_k_uri(target: CakeConcatMLAKTarget) -> str:
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake concat MLA K target: {target}")
    spec = get_cake_concat_mla_k_module_spec()
    return f"{spec.module_ident}_{target}_{spec.closure_sha256}"


@functools.cache
def gen_cake_concat_mla_k_module(target: CakeConcatMLAKTarget) -> JitSpec:
    """Generate one target-specific JIT module from separate source units."""

    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake concat MLA K target: {target}")
    spec = get_cake_concat_mla_k_module_spec()
    csrc_dir = _get_csrc_dir()
    jit_spec = gen_jit_spec(
        name=get_cake_concat_mla_k_uri(target),
        sources=[spec.device_path, spec.binding_path],
        extra_cuda_cflags=[*_TARGET_FLAGS[target]],
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        needs_device_linking=True,
    )
    logger.info("Generated Cake concat MLA K %s JIT spec: %s", target, jit_spec.name)
    return jit_spec


@functools.cache
def load_cake_concat_mla_k_module(target: CakeConcatMLAKTarget):
    module = gen_cake_concat_mla_k_module(target).build_and_load()
    logger.info("Loaded Cake concat MLA K %s module", target)
    return module


def get_cake_concat_mla_k_module(device):
    return load_cake_concat_mla_k_module(cake_concat_mla_k_target(device))


__all__ = [
    "CakeConcatMLAKTarget",
    "CakeConcatMLAKModuleSpec",
    "cake_concat_mla_k_target",
    "cake_concat_mla_k_tokens_per_cta",
    "gen_cake_concat_mla_k_module",
    "get_cake_concat_mla_k_module",
    "get_cake_concat_mla_k_module_spec",
    "get_cake_concat_mla_k_uri",
    "load_cake_concat_mla_k_module",
]
