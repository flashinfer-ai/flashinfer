"""JIT loader for the generated SM103 request-ordered paged-decode program."""

from __future__ import annotations

import functools
import glob
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

from filelock import FileLock

from . import env as jit_env
from .core import logger


_MANIFEST_NAME = "cake_fmha_request_ordered_paged_decode_manifest.json"
_SCHEMA = "flashinfer.cake_fmha_request_ordered_paged_decode.v1"
_CONTRACT = {
    "head_dim": 256,
    "kv_dtype": "float8_e4m3fn",
    "num_kv_heads": 1,
    "num_q_heads": 8,
    "page_size": 64,
    "q_len": [1, 6],
    "query_output_dtype": "bfloat16",
    "request_order": "optional_device_int32",
    "softmax_accumulation_dtype": "float32",
}

_RUNTIME_LENGTHS_STEM = "cake_fmha_request_ordered_paged_decode_runtime_lengths"
_RUNTIME_LENGTHS_MANIFEST = _RUNTIME_LENGTHS_STEM + "_manifest.json"
_RUNTIME_LENGTHS_GRID_RULE = "q_len,head_groups,batch*min(workspace_parts,max(1,floor(sm_count/(batch*q_len*head_groups))))"
_RUNTIME_LENGTHS_CAPACITY_RULE = (
    "max(16,min(256,next_power_of_2(ceil(sm_count/(batch*q_len*head_groups)))))"
)
_ONE_PAGE_STEM = _RUNTIME_LENGTHS_STEM + "_one_page"
_ONE_PAGE_MANIFEST = _ONE_PAGE_STEM + "_manifest.json"


def _one_page_source_root() -> Path:
    directory = "request_ordered_paged_decode_runtime_lengths_one_page"
    for root in (
        jit_env.FLASHINFER_CSRC_DIR / "cake_fmha" / directory,
        Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha" / directory,
    ):
        if (root / _ONE_PAGE_MANIFEST).is_file():
            return root
    raise FileNotFoundError(
        "request-ordered one-page runtime-length sources were not found"
    )


@functools.cache
def get_cake_fmha_request_ordered_one_page_manifest() -> dict[str, Any]:
    """Authenticate profiles selected by immutable per-table page capacity."""
    root = _one_page_source_root()
    payload = json.loads((root / _ONE_PAGE_MANIFEST).read_text())
    _require(
        payload.get("schema") == "flashinfer.cake_fmha_request_ordered_one_page.v1",
        "one-page schema",
    )
    _require(payload.get("target") == "sm_103a", "one-page target")
    contract = payload.get("contract")
    _require(isinstance(contract, dict), "one-page contract")
    sm_count, min_pairs = contract.get("sm_count"), contract.get("min_pairs")
    _require(type(sm_count) is int and sm_count > 0, "one-page SM count")
    _require(
        type(min_pairs) is int and 1 <= min_pairs <= 8388608, "one-page granularity"
    )
    _require(
        contract
        == {
            "query_dtype": "bfloat16",
            "kv_dtype": "float8_e4m3fn",
            "output_dtype": "bfloat16",
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
            "page_size": 64,
            "kv_layout": "HND",
            "causal": True,
            "batch_capacity": 256,
            "sm_count": sm_count,
            "min_pairs": min_pairs,
            "workspace_parts": 16,
            "physical_pages_per_table": 1,
            "scheduler": "runtime_lengths_one_page",
            "request_order_semantics": "original_logical_request_index",
        },
        "one-page contract fields",
    )
    modules, bindings = payload.get("modules"), payload.get("bindings")
    _require(
        isinstance(modules, list) and isinstance(bindings, list), "one-page inventory"
    )
    _require(
        payload.get("module_count") == len(modules) == len(bindings) == 16,
        "one-page profile count",
    )
    names = _verify_modules(root, modules)
    _require(
        all(name.startswith(_ONE_PAGE_STEM + "_") for name in names),
        "one-page module prefix",
    )
    profiles = set()
    for binding in bindings:
        q = binding.get("q_len")
        modes = tuple(
            binding.get(key)
            for key in ("write_lse", "device_scales", "request_order_enabled")
        )
        _require(type(q) is int and q in (1, 6), "one-page Q length")
        _require(all(type(mode) is bool for mode in modes), "one-page mode flags")
        profile = (q, *modes)
        _require(profile not in profiles, "duplicate one-page profile")
        profiles.add(profile)
        _require(binding.get("module_name") in names, "one-page module binding")
        smem = binding.get("dynamic_smem_bytes")
        _require(type(smem) is int and smem > 0, "one-page shared memory")
        _require(
            binding
            == {
                "module_name": binding["module_name"],
                "batch_size_min": 1,
                "batch_size_max": 256,
                "q_len": q,
                "num_q_heads": 8,
                "num_kv_heads": 1,
                "write_lse": modes[0],
                "device_scales": modes[1],
                "request_order_enabled": modes[2],
                "workspace_parts": 16,
                "shared_plan_capacity": 256,
                "min_pairs": min_pairs,
                "physical_pages_per_table": 1,
                "grid_rule": "q_len,head_groups,batch",
                "dynamic_smem_bytes": smem,
                "total_tiles_rule": "batch*workspace_parts",
            },
            "one-page binding fields",
        )
    _require(
        {binding["module_name"] for binding in bindings} == names,
        "one-page module coverage",
    )
    return payload


def _runtime_length_split_capacity(batch: int, q_len: int, sm_count: int) -> int:
    required = (sm_count + batch * q_len - 1) // (batch * q_len)
    return max(16, min(256, 1 << (required - 1).bit_length()))


def _runtime_lengths_source_root() -> Path:
    directory = "request_ordered_paged_decode_runtime_lengths"
    candidates = (
        jit_env.FLASHINFER_CSRC_DIR / "cake_fmha" / directory,
        Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha" / directory,
    )
    for candidate in candidates:
        if (candidate / _RUNTIME_LENGTHS_MANIFEST).is_file():
            return candidate
    raise FileNotFoundError("request-ordered runtime-length sources were not found")


@functools.cache
def get_cake_fmha_request_ordered_runtime_lengths_manifest() -> dict[str, Any]:
    """Authenticate the family that reads lengths and order during each launch."""
    root = _runtime_lengths_source_root()
    payload = json.loads((root / _RUNTIME_LENGTHS_MANIFEST).read_text())
    _require(isinstance(payload, dict), "runtime-length root")
    _require(
        payload.get("schema")
        == "flashinfer.cake_fmha_request_ordered_runtime_lengths.v2",
        "runtime-length schema",
    )
    _require(payload.get("target") == "sm_103a", "runtime-length target")
    contract = payload.get("contract")
    _require(isinstance(contract, dict), "runtime-length contract")
    sm_count, min_pairs = contract.get("sm_count"), contract.get("min_pairs")
    _require(type(sm_count) is int and sm_count > 0, "runtime-length SM count")
    _require(
        type(min_pairs) is int and 1 <= min_pairs <= 8388608,
        "runtime-length granularity",
    )
    _require(
        contract
        == {
            "query_dtype": "bfloat16",
            "kv_dtype": "float8_e4m3fn",
            "output_dtype": "bfloat16",
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "head_dim": 256,
            "page_size": 64,
            "kv_layout": "HND",
            "causal": True,
            "split_capacities": [16, 32, 64, 128, 256],
            "split_capacity_rule": _RUNTIME_LENGTHS_CAPACITY_RULE,
            "batch_capacity": 256,
            "sm_count": sm_count,
            "scheduler": "runtime_lengths",
            "min_pairs": min_pairs,
            "request_order_semantics": "original_logical_request_index",
        },
        "runtime-length contract fields",
    )
    modules, bindings = payload.get("modules"), payload.get("bindings")
    _require(isinstance(modules, list) and bool(modules), "runtime-length modules")
    _require(isinstance(bindings, list) and bool(bindings), "runtime-length bindings")
    _require(payload.get("module_count") == len(modules), "runtime-length module count")
    names = _verify_modules(root, modules)
    _require(
        all(name.startswith(_RUNTIME_LENGTHS_STEM + "_") for name in names),
        "runtime-length module prefix",
    )
    seen = set()
    covered: dict[tuple[Any, ...], set[int]] = {}
    for index, binding in enumerate(bindings):
        _require(isinstance(binding, dict), f"runtime-length bindings[{index}]")
        _require(binding.get("module_name") in names, "runtime-length module binding")
        q_len = binding.get("q_len")
        _require(type(q_len) is int and q_len in (1, 6), "runtime-length Q length")
        modes = tuple(
            binding.get(key)
            for key in ("write_lse", "device_scales", "request_order_enabled")
        )
        _require(all(type(mode) is bool for mode in modes), "runtime-length mode flags")
        profile = (q_len, *modes)
        capacity = binding.get("workspace_parts")
        _require(
            type(capacity) is int and capacity in (16, 32, 64, 128, 256),
            "runtime-length split capacity",
        )
        first, last = binding.get("batch_size_min"), binding.get("batch_size_max")
        _require(
            type(first) is int and type(last) is int and 1 <= first <= last <= 256,
            "runtime-length batch interval",
        )
        key = (*profile, capacity)
        _require(key not in seen, "duplicate runtime-length capacity profile")
        seen.add(key)
        batches = set(range(first, last + 1))
        _require(
            all(
                _runtime_length_split_capacity(batch, q_len, sm_count) == capacity
                for batch in batches
            ),
            "runtime-length capacity selector mismatch",
        )
        previous = covered.setdefault(profile, set())
        _require(
            not previous.intersection(batches),
            "overlapping runtime-length batch intervals",
        )
        previous.update(batches)
        smem = binding.get("dynamic_smem_bytes")
        _require(type(smem) is int and smem > 0, "runtime-length shared memory")
        expected = {
            "module_name": binding["module_name"],
            "batch_size_min": first,
            "batch_size_max": last,
            "q_len": q_len,
            "num_q_heads": 8,
            "num_kv_heads": 1,
            "write_lse": modes[0],
            "device_scales": modes[1],
            "request_order_enabled": modes[2],
            "workspace_parts": capacity,
            "shared_plan_capacity": 256,
            "min_pairs": min_pairs,
            "grid_rule": _RUNTIME_LENGTHS_GRID_RULE,
            "dynamic_smem_bytes": smem,
            "total_tiles_rule": "batch*workspace_parts",
        }
        _require(binding == expected, "runtime-length binding fields")
    _require(
        all(batches == set(range(1, 257)) for batches in covered.values()),
        "runtime-length profile has a batch coverage gap",
    )
    _require(
        {binding["module_name"] for binding in bindings} == names,
        "runtime-length module coverage",
    )
    return payload


_FP8Q_MANIFEST_NAME = "cake_fmha_request_ordered_paged_decode_fp8q_manifest.json"
_FP8Q_CONTRACT: dict[str, Any] = {
    "query_dtype": "float8_e4m3fn",
    "output_dtype": "bfloat16",
    "kv_dtype": "float8_e4m3fn",
    "softmax_accumulation_dtype": "float32",
    "num_q_heads": 32,
    "num_kv_heads": 2,
    "head_dim": 256,
    "page_size": 64,
    "q_len": 6,
    "batch_sizes": [64, 128, 160, 192, 224, 256],
    "minimum_kv_len": 6,
    "write_lse": False,
    "kv_strides": [32768, 256, 512, 1],
    "shared_page_table": True,
    "request_order": "device_int32_permutation",
}


def _fp8q_source_root(*, split: bool = False) -> Path:
    directory = "request_ordered_paged_decode_fp8q" + ("_split" if split else "")
    manifest_name = _FP8Q_SPLIT_MANIFEST_NAME if split else _FP8Q_MANIFEST_NAME
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha" / directory
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha" / directory
    for candidate in (installed, checkout):
        if (candidate / manifest_name).is_file():
            return candidate
    raise FileNotFoundError("request-ordered FP8-Q sources were not found")


@functools.cache
def get_cake_fmha_request_ordered_fp8q_manifest() -> dict[str, Any]:
    """Authenticate the independent FP8-query, BF16-output export contract."""
    root = _fp8q_source_root()
    manifest = json.loads((root / _FP8Q_MANIFEST_NAME).read_text(encoding="utf-8"))
    _require(
        manifest.get("schema") == "flashinfer.cake_fmha_request_ordered_fp8q.v1",
        "FP8-Q schema",
    )
    _require(manifest.get("target") == "sm_103a", "FP8-Q target")
    _require(manifest.get("contract") == _FP8Q_CONTRACT, "FP8-Q contract")
    modules, bindings = manifest.get("modules"), manifest.get("bindings")
    _require(
        isinstance(modules, list) and isinstance(bindings, list), "FP8-Q inventory"
    )
    names = _verify_modules(root, modules)
    _require(
        len(modules) == len(bindings) == len(_FP8Q_CONTRACT["batch_sizes"]),
        "FP8-Q inventory count",
    )
    _require(manifest.get("module_count") == len(modules), "FP8-Q module count")
    seen = set()
    for binding in bindings:
        batch = binding.get("batch_size")
        _require(
            batch in _FP8Q_CONTRACT["batch_sizes"] and batch not in seen, "FP8-Q batch"
        )
        seen.add(batch)
        expected = dict(
            batch_size=batch,
            q_len=6,
            num_q_heads=32,
            num_kv_heads=2,
            workspace_parts=1,
            grid=[1, 1, batch * 2],
            total_tiles=batch * 2,
            write_lse=False,
            query_dtype="float8_e4m3fn",
        )
        _require(
            {key: value for key, value in binding.items() if key != "module_name"}
            == expected,
            "FP8-Q binding",
        )
        _require(binding.get("module_name") in names, "FP8-Q module binding")
    _require(
        {binding["module_name"] for binding in bindings} == names,
        "FP8-Q module coverage",
    )
    return manifest


_FP8Q_SPLIT_MANIFEST_NAME = (
    "cake_fmha_request_ordered_paged_decode_fp8q_split_manifest.json"
)
_FP8Q_SPLIT_CONTRACT: dict[str, Any] = {
    "query_dtype": "float8_e4m3fn",
    "output_dtype": "bfloat16",
    "kv_dtype": "float8_e4m3fn",
    "num_q_heads": 32,
    "num_kv_heads": 2,
    "head_dim": 256,
    "page_size": 64,
    "q_len": 6,
    "batch_splits": [[8, 8], [27, 2], [32, 2]],
    "minimum_kv_len": 6,
    "write_lse": [False, True],
    "lse_base": 2,
    "partial_dtype": "float32",
    "kv_strides": [32768, 256, 512, 1],
    "shared_page_table": True,
    "request_order": "device_int32_permutation",
}


@functools.cache
def get_cake_fmha_request_ordered_fp8q_split_manifest() -> dict[str, Any]:
    """Authenticate compound FP8-query producer/reducer plans."""
    root = _fp8q_source_root(split=True)
    manifest = json.loads(
        (root / _FP8Q_SPLIT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    _require(
        manifest.get("schema") == "flashinfer.cake_fmha_request_ordered_fp8q_split.v1",
        "FP8-Q split schema",
    )
    _require(manifest.get("target") == "sm_103a", "FP8-Q split target")
    _require(manifest.get("contract") == _FP8Q_SPLIT_CONTRACT, "FP8-Q split contract")
    modules, bindings = manifest.get("modules"), manifest.get("bindings")
    _require(
        isinstance(modules, list) and isinstance(bindings, list),
        "FP8-Q split inventory",
    )
    _require(
        len(modules) == manifest.get("module_count") == 7 and len(bindings) == 6,
        "FP8-Q split inventory count",
    )
    producers = [m for m in modules if m.get("tma_workspace_bytes") == 384]
    reducers = [m for m in modules if m.get("tma_workspace_bytes") == 0]
    _require(len(producers) == 3 and len(reducers) == 4, "FP8-Q split module roles")
    producer_names = _verify_modules(root, producers)
    reducer_names = _verify_modules(root, reducers, tma_workspace_bytes=0)
    seen = set()
    for row in bindings:
        batch, splits, write_lse = (
            row.get("batch_size"),
            row.get("workspace_parts"),
            row.get("write_lse"),
        )
        _require(
            [batch, splits] in _FP8Q_SPLIT_CONTRACT["batch_splits"]
            and type(write_lse) is bool,
            "FP8-Q split geometry",
        )
        _require((batch, write_lse) not in seen, "FP8-Q split duplicate binding")
        seen.add((batch, write_lse))
        expected = dict(
            batch_size=batch,
            q_len=6,
            num_q_heads=32,
            num_kv_heads=2,
            workspace_parts=splits,
            grid=[1, 1, batch * 2 * splits],
            total_tiles=batch * 2 * splits,
            write_lse=write_lse,
            query_dtype="float8_e4m3fn",
            reducer_grid=[batch * 6 * 32, 1, 1],
        )
        _require(
            {
                k: v
                for k, v in row.items()
                if k not in ("module_name", "reducer_module_name")
            }
            == expected,
            "FP8-Q split binding",
        )
        _require(
            row.get("module_name") in producer_names
            and row.get("reducer_module_name") in reducer_names,
            "FP8-Q split module binding",
        )
    _require(
        {row["module_name"] for row in bindings} == producer_names
        and {row["reducer_module_name"] for row in bindings} == reducer_names,
        "FP8-Q split module coverage",
    )
    return manifest


@dataclass(frozen=True)
class CakeFmhaRequestOrderedModuleSpec:
    """One authenticated generated source pair."""

    name: str
    closure_sha256: str
    device_path: Path
    binding_path: Path
    module_ident: str
    kernel_symbol: str
    ffi_entry: str
    compile_options: tuple[str, ...]
    tma_workspace_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid request-ordered FMHA manifest: {message}")


def _source_root(
    num_q_heads: int = 8, num_kv_heads: int = 1, *, runtime_q: bool = False
) -> Path:
    _require((num_q_heads, num_kv_heads) in ((8, 1), (32, 2)), "head geometry")
    suffix = "_32q2" if (num_q_heads, num_kv_heads) == (32, 2) else ""
    directory = "request_ordered_paged_decode" + suffix
    manifest_name = (
        "cake_fmha_request_ordered_paged_decode"
        + suffix
        + ("_runtime_q" if runtime_q else "")
        + "_manifest.json"
    )
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_fmha" / directory
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_fmha" / directory
    for candidate in (installed, checkout):
        if (candidate / manifest_name).is_file():
            return candidate
    raise FileNotFoundError(
        "request-ordered Cake FMHA sources were not found; checked "
        f"{installed} and {checkout}"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verified_source(root: Path, value: object, digest: object, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label}.path")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("generated", "sm_103a")
        and len(relative.parts) == 3,
        f"{label}.path",
    )
    path = root.joinpath(*relative.parts)
    _require(path.is_file(), f"{label}.path does not exist")
    _require(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label}.sha256",
    )
    _require(_sha256(path) == digest, f"{label}.sha256 mismatch")
    return path


def _verify_modules(
    root: Path, modules: list[dict[str, Any]], *, tma_workspace_bytes: int = 384
) -> set[str]:
    names: set[str] = set()
    for index, module in enumerate(modules):
        _require(isinstance(module, dict), f"modules[{index}]")
        _require(module.get("arch") == "sm_103a", f"modules[{index}].arch")
        name = module.get("name")
        _require(
            isinstance(name, str)
            and name.startswith("cake_fmha_request_ordered_paged_decode_")
            and name.replace("_", "").isalnum(),
            f"modules[{index}].name",
        )
        _require(name not in names, f"duplicate module {name}")
        names.add(name)
        _verified_source(
            root,
            module.get("device_path"),
            module.get("device_sha256"),
            f"modules[{index}].device",
        )
        _verified_source(
            root,
            module.get("binding_path"),
            module.get("binding_sha256"),
            f"modules[{index}].binding",
        )
        closure = module.get("closure_sha256")
        _require(
            isinstance(closure, str)
            and len(closure) == 64
            and all(character in "0123456789abcdef" for character in closure),
            f"modules[{index}].closure_sha256",
        )
        for field in ("module_ident", "kernel_symbol", "ffi_entry"):
            value = module.get(field)
            _require(
                isinstance(value, str)
                and bool(value)
                and value.replace("_", "a").isalnum(),
                f"modules[{index}].{field}",
            )
        _require(
            module.get("binding_mode") == "embedded_cubin",
            f"modules[{index}].binding_mode",
        )
        _require(
            module.get("compile_options") == ["--use_fast_math"],
            f"modules[{index}].compile_options",
        )
        _require(
            module.get("tma_workspace_bytes") == tma_workspace_bytes,
            f"modules[{index}].tma_workspace_bytes",
        )
    return names


_B1_Q6_S76_BINDING: dict[str, Any] = {
    "name": "b1_q6_s76",
    "batch_size": 1,
    "q_len": 6,
    "num_q_heads": 32,
    "num_kv_heads": 2,
    "num_kv_splits": 76,
    "write_lse": False,
    "ordered": True,
    "grid": [76, 2, 1],
    "workspace_parts": 76,
    "total_tiles": 1,
    "scratch_layout": {
        "schema": "peer_split_major_padded128_pair_stats_v21",
        "partial_o_elements_per_batch": 4980736,
        "partial_stats_elements_per_batch": 38912,
        "kv_groups": 2,
        "splits": 76,
        "padded_rows": 128,
        "live_rows": 96,
        "head_dim": 256,
        "stats_fields": ["raw_max", "sum"],
        "partial_o_dtype": "bfloat16",
        "partial_stats_dtype": "float32",
    },
}


@functools.cache
def get_cake_fmha_request_ordered_manifest(
    num_q_heads: int = 8, num_kv_heads: int = 1
) -> dict[str, Any]:
    """Load and authenticate the generated-program route ledger."""

    root = _source_root(num_q_heads, num_kv_heads)
    suffix = "_32q2" if (num_q_heads, num_kv_heads) == (32, 2) else ""
    manifest_name = "cake_fmha_request_ordered_paged_decode" + suffix + "_manifest.json"
    payload: Any = json.loads((root / manifest_name).read_text())
    _require(isinstance(payload, dict), "root")
    _require(payload.get("schema") == _SCHEMA, "schema")
    _require(payload.get("target") == "sm_103a", "target")
    _require(payload.get("shape_count") == 43, "shape_count")
    supplemental = payload.get("supplemental_bindings", [])
    _require(isinstance(supplemental, list), "supplemental_bindings")
    _require(
        len(supplemental) in ((0, 1) if suffix else (0,)), "supplemental_bindings count"
    )
    module_count = 13 + len(supplemental)
    _require(payload.get("module_count") == module_count, "module_count")
    expected_contract = dict(
        _CONTRACT, num_q_heads=num_q_heads, num_kv_heads=num_kv_heads
    )
    if suffix:
        expected_contract.update(
            q_groups_per_kv=[1, 2], query_heads_per_work_group=[8, 16]
        )
    _require(payload.get("contract") == expected_contract, "contract")
    modules = payload.get("modules")
    routes = payload.get("routes")
    _require(isinstance(modules, list) and len(modules) == module_count, "modules")
    _require(isinstance(routes, list) and len(routes) == 43, "routes")
    names = _verify_modules(root, modules)
    for binding in supplemental:
        _require(isinstance(binding, dict), "supplemental binding")
        _require(binding.get("module_name") in names, "supplemental module_name")
        _require(
            {key: value for key, value in binding.items() if key != "module_name"}
            == _B1_Q6_S76_BINDING,
            "supplemental B1/Q6/S76 binding",
        )
    route_names: set[str] = set()
    for index, route in enumerate(routes):
        _require(isinstance(route, dict), f"routes[{index}]")
        shape = route.get("shape")
        _require(
            isinstance(shape, str) and bool(shape) and shape not in route_names,
            f"routes[{index}].shape",
        )
        route_names.add(shape)
        _require(route.get("module_name") in names, f"routes[{index}].module_name")
        plan = route.get("build_plan")
        _require(isinstance(plan, dict), f"routes[{index}].build_plan")
        _require(plan.get("q_len") in (1, 6), f"routes[{index}].build_plan.q_len")
        if suffix:
            wide_q6 = plan.get("fused_q6") is True and plan.get("write_lse") is False
            _require(
                plan.get("num_q_heads") == 32
                and plan.get("num_kv_heads") == 2
                and plan.get("q_groups_per_kv") == (1 if wide_q6 else 2)
                and plan.get("query_heads_per_work_group", 8) == (16 if wide_q6 else 8),
                f"routes[{index}].build_plan.head_geometry",
            )
            _require(
                route.get("args", {}).get("params", {}).get("num_qo_heads") == 32
                and route.get("args", {}).get("params", {}).get("num_kv_heads") == 2,
                f"routes[{index}].args.head_geometry",
            )
    return payload


@functools.cache
def get_cake_fmha_request_ordered_runtime_q_manifest(
    num_q_heads: int = 8, num_kv_heads: int = 1
) -> dict[str, Any]:
    """Authenticate the supplemental bindings whose query length is grid.x."""
    root = _source_root(num_q_heads, num_kv_heads, runtime_q=True)
    suffix = "_32q2" if (num_q_heads, num_kv_heads) == (32, 2) else ""
    stem = "cake_fmha_request_ordered_paged_decode" + suffix + "_runtime_q"
    payload: Any = json.loads((root / (stem + "_manifest.json")).read_text())
    _require(isinstance(payload, dict), "root")
    _require(
        payload.get("schema") == "flashinfer.cake_fmha_request_ordered_runtime_q.v1",
        "schema",
    )
    _require(payload.get("target") == "sm_103a", "target")
    _require(payload.get("module_count") == 2, "module_count")
    _require(payload.get("binding_count") == 2, "binding_count")
    expected_contract = dict(
        _CONTRACT,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_groups_per_kv=num_q_heads // num_kv_heads // 8,
        query_heads_per_work_group=8,
        q_len={
            "kind": "uniform_positive_integer",
            "excluding": [1, 6],
            "value_source": "grid.x",
        },
        ordered=True,
        num_split=1,
        workspace_parts=1,
    )
    _require(payload.get("contract") == expected_contract, "contract")
    modules = payload.get("modules")
    bindings = payload.get("bindings")
    _require(isinstance(modules, list) and len(modules) == 2, "modules")
    _require(isinstance(bindings, list) and len(bindings) == 2, "bindings")
    names = _verify_modules(root, modules)
    _require(all(name.startswith(stem + "_") for name in names), "module prefix")
    for index, binding in enumerate(bindings):
        _require(isinstance(binding, dict), f"bindings[{index}]")
        _require(binding.get("module_name") in names, f"bindings[{index}].module_name")
        _require(type(binding.get("write_lse")) is bool, f"bindings[{index}].write_lse")
    _require(
        {binding["write_lse"] for binding in bindings} == {False, True}, "LSE modes"
    )
    _require(
        {binding["module_name"] for binding in bindings} == names, "binding modules"
    )
    return payload


@functools.cache
def get_cake_fmha_request_ordered_module_spec(
    name: str,
) -> CakeFmhaRequestOrderedModuleSpec:
    geometry = (
        (32, 2)
        if name.startswith("cake_fmha_request_ordered_paged_decode_32q2_")
        else (8, 1)
    )
    if name.startswith(_ONE_PAGE_STEM + "_"):
        root = _one_page_source_root()
        manifest = get_cake_fmha_request_ordered_one_page_manifest()
    elif name.startswith(_RUNTIME_LENGTHS_STEM + "_"):
        root = _runtime_lengths_source_root()
        manifest = get_cake_fmha_request_ordered_runtime_lengths_manifest()
    elif name.startswith("cake_fmha_request_ordered_paged_decode_fp8q_split_"):
        root = _fp8q_source_root(split=True)
        manifest = get_cake_fmha_request_ordered_fp8q_split_manifest()
    elif name.startswith("cake_fmha_request_ordered_paged_decode_fp8q_"):
        root = _fp8q_source_root()
        manifest = get_cake_fmha_request_ordered_fp8q_manifest()
    else:
        runtime_q = "_runtime_q_" in name
        root = _source_root(*geometry, runtime_q=runtime_q)
        reader = (
            get_cake_fmha_request_ordered_runtime_q_manifest
            if runtime_q
            else get_cake_fmha_request_ordered_manifest
        )
        manifest = reader(*geometry)
    matches = [module for module in manifest["modules"] if module["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown request-ordered FMHA module: {name}")
    module = matches[0]
    spec = CakeFmhaRequestOrderedModuleSpec(
        name=name,
        closure_sha256=module["closure_sha256"],
        device_path=_verified_source(
            root,
            module["device_path"],
            module["device_sha256"],
            f"module {name} device",
        ),
        binding_path=_verified_source(
            root,
            module["binding_path"],
            module["binding_sha256"],
            f"module {name} binding",
        ),
        module_ident=module["module_ident"],
        kernel_symbol=module["kernel_symbol"],
        ffi_entry=module["ffi_entry"],
        compile_options=tuple(module["compile_options"]),
        tma_workspace_bytes=module["tma_workspace_bytes"],
    )
    binding = spec.binding_path.read_text(encoding="utf-8")
    _require(
        binding.count(f"TVM_FFI_EMBED_CUBIN({spec.module_ident});") == 1,
        f"module {name} embedded-cubin declaration",
    )
    _require(
        (
            binding.count(
                f"EmbedCubinModule_{spec.module_ident}::Global()->mod.GetKernel("
                f'"{spec.kernel_symbol}")'
            )
            == 2
            if spec.tma_workspace_bytes
            else binding.count(
                f"TVM_FFI_EMBED_CUBIN_GET_KERNEL({spec.module_ident}, "
                f'"{spec.kernel_symbol}")'
            )
            == 1
        ),
        f"module {name} ordinary and capture kernel lookups",
    )
    _require(
        binding.count(f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({spec.ffi_entry},") == 1,
        f"module {name} FFI entry",
    )
    return spec


@functools.cache
def _cuda_include_dirs() -> tuple[Path, ...]:
    candidates: list[str] = []
    for variable in ("CUDA_HOME", "CUDA_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(str(Path(value) / "include"))
    nvcc = shutil.which("nvcc")
    if nvcc:
        candidates.append(str(Path(nvcc).resolve().parent.parent / "include"))
    candidates.append("/usr/local/cuda/include")
    for entry in sys.path:
        if not entry:
            continue
        candidates.extend(sorted(glob.glob(str(Path(entry) / "nvidia/cu*/include"))))
        candidates.append(str(Path(entry) / "nvidia/cuda_runtime/include"))
        candidates.append(str(Path(entry) / "triton/backends/nvidia/include"))
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = Path(candidate).resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if any(
            (resolved / marker).is_file()
            for marker in (
                "cuda_bf16.h",
                "cuda.h",
                "cuda_runtime.h",
                "crt/host_config.h",
            )
        ):
            result.append(resolved)
    _require(bool(result), "CUDA headers for generated FMHA are unavailable")
    return tuple(result)


def _nvrtc_options(spec: CakeFmhaRequestOrderedModuleSpec) -> tuple[str, ...]:
    options = [
        "--gpu-architecture=sm_103a",
        "-std=c++17",
        "-default-device",
    ]
    for include in _cuda_include_dirs():
        options.append(f"-I{include}")
        cccl = include / "cccl"
        if (cccl / "cuda/std").is_dir():
            options.append(f"-I{cccl}")
    options.extend(spec.compile_options)
    return tuple(options)


def _result_ok(result: object) -> bool:
    try:
        return int(cast(Any, result)) == 0
    except TypeError:
        return getattr(result, "value", result) == 0


def _compile_log(nvrtc: object, program: object) -> str:
    api = cast(Any, nvrtc)
    result, size = api.nvrtcGetProgramLogSize(program)
    if not _result_ok(result) or size <= 1:
        return ""
    log = b"\0" * size
    (result,) = api.nvrtcGetProgramLog(program, log)
    return log.decode(errors="replace").rstrip("\0") if _result_ok(result) else ""


def _file_identity(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
            size += len(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": size}


@functools.cache
def _nvrtc_toolchain_identity() -> dict[str, object]:
    from cuda.bindings import nvrtc

    result, major, minor = nvrtc.nvrtcVersion()
    if not _result_ok(result):
        raise RuntimeError(f"nvrtcVersion failed: {result}")
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        raise RuntimeError("cannot resolve the loaded NVRTC library")
    libraries: set[Path] = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        raw = line.rpartition(" ")[2]
        if "libnvrtc" in Path(raw).name:
            libraries.add(Path(raw).resolve(strict=True))
    if not libraries:
        raise RuntimeError("cannot resolve the loaded NVRTC library")
    return {
        "nvrtc_version": [int(major), int(minor)],
        "loaded_libraries": [_file_identity(path) for path in sorted(libraries)],
    }


def _compile_cubin(spec: CakeFmhaRequestOrderedModuleSpec) -> bytes:
    """Compile the exported source with the same NVRTC option model as Cake."""

    from cuda.bindings import nvrtc

    source = spec.device_path.read_bytes()
    options = _nvrtc_options(spec)
    if any("o1" in option.lower() for option in options):
        raise RuntimeError(f"forbidden O1 option in Cake FMHA NVRTC flags: {options}")
    result, program = nvrtc.nvrtcCreateProgram(source, b"kernel.cu", 0, [], [])
    if not _result_ok(result):
        raise RuntimeError(f"nvrtcCreateProgram failed for {spec.name}: {result}")
    try:
        encoded = [option.encode() for option in options]
        (result,) = nvrtc.nvrtcCompileProgram(program, len(encoded), encoded)
        if not _result_ok(result):
            raise RuntimeError(
                f"NVRTC compilation failed for {spec.name}: {result}\n"
                f"{_compile_log(nvrtc, program)}"
            )
        result, size = nvrtc.nvrtcGetCUBINSize(program)
        if not _result_ok(result):
            raise RuntimeError(f"nvrtcGetCUBINSize failed for {spec.name}: {result}")
        cubin = b"\0" * size
        (result,) = nvrtc.nvrtcGetCUBIN(program, cubin)
        if not _result_ok(result):
            raise RuntimeError(f"nvrtcGetCUBIN failed for {spec.name}: {result}")
        return cubin
    finally:
        nvrtc.nvrtcDestroyProgram(program)


def _cached_cubin(
    spec: CakeFmhaRequestOrderedModuleSpec,
) -> tuple[bytes, Path]:
    identity = hashlib.sha256(
        json.dumps(
            {
                "closure_sha256": spec.closure_sha256,
                "compile_options": list(_nvrtc_options(spec)),
                "nvrtc_toolchain": _nvrtc_toolchain_identity(),
                "target": "sm_103a",
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()[:20]
    build_directory = (
        jit_env.FLASHINFER_JIT_DIR
        / "cake_fmha_request_ordered"
        / f"{spec.name}_{identity}"
    )
    build_directory.mkdir(parents=True, exist_ok=True)
    cubin_path = build_directory / f"{spec.module_ident}.cubin"
    receipt_path = build_directory / f"{spec.module_ident}.json"
    with FileLock(f"{cubin_path}.lock", thread_local=False):
        reusable = False
        if cubin_path.is_file() and receipt_path.is_file():
            try:
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                cubin = cubin_path.read_bytes()
                reusable = receipt == {
                    "cache_identity": identity,
                    "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                    "cubin_size_bytes": len(cubin),
                }
            except (OSError, TypeError, ValueError):
                reusable = False
        if not reusable:
            cubin = _compile_cubin(spec)
            temporary = cubin_path.with_name(f".{cubin_path.name}.{os.getpid()}.tmp")
            temporary_receipt = receipt_path.with_name(
                f".{receipt_path.name}.{os.getpid()}.tmp"
            )
            try:
                temporary.write_bytes(cubin)
                os.replace(temporary, cubin_path)
                temporary_receipt.write_text(
                    json.dumps(
                        {
                            "cache_identity": identity,
                            "cubin_sha256": hashlib.sha256(cubin).hexdigest(),
                            "cubin_size_bytes": len(cubin),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary_receipt, receipt_path)
            finally:
                temporary.unlink(missing_ok=True)
                temporary_receipt.unlink(missing_ok=True)
        cubin = cubin_path.read_bytes()
    _require(bool(cubin), f"empty cubin for {spec.name}")
    return cubin, build_directory


@functools.cache
def load_cake_fmha_request_ordered_module(name: str):
    """NVRTC-compile and load one exact SM103 generated-program member."""

    import torch
    from tvm_ffi import cpp

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("request-ordered Cake FMHA requires compute capability 10.3")
    spec = get_cake_fmha_request_ordered_module_spec(name)
    root = spec.binding_path.parents[2]
    cubin, build_directory = _cached_cubin(spec)
    result = cpp.load_inline(
        build_directory.name,
        cpp_sources=spec.binding_path.read_text(encoding="utf-8"),
        embed_cubin={spec.module_ident: cubin},
        extra_include_paths=[
            *(str(path) for path in _cuda_include_dirs()),
            str(root.parents[1]),
            str(root.parents[2] / "include"),
            str(jit_env.FLASHINFER_CSRC_DIR),
        ],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_directory),
    )
    _require(
        callable(getattr(result, spec.ffi_entry, None)),
        f"missing FFI entry {spec.ffi_entry}",
    )
    logger.info("Loaded request-ordered Cake FMHA module %s", name)
    return result


__all__ = [
    "CakeFmhaRequestOrderedModuleSpec",
    "get_cake_fmha_request_ordered_manifest",
    "get_cake_fmha_request_ordered_module_spec",
    "get_cake_fmha_request_ordered_runtime_lengths_manifest",
    "load_cake_fmha_request_ordered_module",
]
