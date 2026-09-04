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

Source-only JIT registry for the generated FP32-indexed KDA portfolio.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from glob import glob
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, Literal, Optional, Protocol, cast

import torch
from filelock import FileLock

from ..utils import get_compute_capability

GeneratedKDAIndexedTarget = Literal["sm100a", "sm103a"]

_CATALOG_KIND = "flashinfer.generated_kda_indexed_prefill.source_catalog"
_IMPORT_RECEIPT_KIND = "flashinfer.generated_kda_indexed_prefill.import_receipt"
_CATALOG_NAME = "flashkda_generated_indexed_variant_metadata.json"
_RECEIPT_NAME = "flashkda_generated_indexed_generation_receipt.json"
_SOURCE_PREFIX = "flashkda_generated_indexed_"
_CATALOG_SCHEMA_VERSION = 2
_EXPECTED_MODULE_COUNTS = {
    "sm100a": 18,
    "sm103a": 19,
}
_EXPECTED_STATE_POOL_CAPACITY = 257
_TARGET_ARCHITECTURES: dict[GeneratedKDAIndexedTarget, str] = {
    "sm100a": "sm_100a",
    "sm103a": "sm_103a",
}
_TARGET_COMPUTE_CAPABILITIES: dict[tuple[int, int], GeneratedKDAIndexedTarget] = {
    (10, 0): "sm100a",
    (10, 3): "sm103a",
}
_EXPECTED_RUNTIME_CONTRACT = {
    "A_log_dtype": "float32",
    "beta_dtype": "bfloat16",
    "beta_is_logit": True,
    "checkpoint_mode": "none",
    "dt_bias_dtype": "float32",
    "gate_kind": "softplus",
    "head_dim": 128,
    "head_relationship": "equal_q_kv",
    "initial_state": "indexed_float32_pool",
    "operation": "recurrent_kda_prefill",
    "output_dtype": "bfloat16",
    "qkv_dtype": "bfloat16",
    "targets": ["sm100a", "sm103a"],
}
_SHA256_CHARACTERS = frozenset("0123456789abcdef")
_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SAFE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:+-]*\Z")
_NVRTC_TIMEOUT_SECONDS = 600


class GeneratedKDAIndexedPrefillError(RuntimeError):
    """The source-only indexed-prefill package or runtime contract is invalid."""


@dataclass(frozen=True)
class _SourceRecord:
    path: Path
    relative: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ModuleRecord:
    id: str
    module_ident: str
    entry_point: str
    kernel_name: str
    compile_options: tuple[str, ...]
    cuda_source: _SourceRecord
    host_source: _SourceRecord
    cubin_sha256: str
    cubin_size_bytes: int


@dataclass(frozen=True)
class _TargetRecord:
    target: GeneratedKDAIndexedTarget
    architecture: str
    toolchain_identity: str
    build_recipe: _SourceRecord
    dispatcher: _SourceRecord
    modules: tuple[_ModuleRecord, ...]


class _PreparedLaunch(Protocol):
    def launch(self) -> object: ...

    def close(self) -> None: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GeneratedKDAIndexedPrefillError(message)


def _object(value: object, fields: set[str], label: str) -> Mapping[str, object]:
    _require(
        isinstance(value, Mapping) and set(value) == fields,
        f"{label} fields differ",
    )
    assert isinstance(value, Mapping)
    return value


def _identifier(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _C_IDENTIFIER.fullmatch(value) is not None,
        f"{label} must be a C identifier",
    )
    assert isinstance(value, str)
    return value


def _safe_token(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and _SAFE_TOKEN.fullmatch(value) is not None
        and ".." not in value,
        f"{label} must be a path-free token",
    )
    assert isinstance(value, str)
    return value


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _full_sha256(value: object, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= _SHA256_CHARACTERS,
        f"{label} must be one full lowercase SHA-256",
    )
    assert isinstance(value, str)
    return value


def _full_identity(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and value.startswith("sha256:"),
        f"{label} must be a SHA-256 identity",
    )
    assert isinstance(value, str)
    _full_sha256(value.removeprefix("sha256:"), label)
    return value


def _recipe_toolchain_identity(record: _SourceRecord, label: str) -> str:
    payload = record.path.read_bytes()
    _require(
        len(payload) == record.size_bytes and _sha256(payload) == record.sha256,
        f"{label} content identity differs",
    )
    try:
        module = ast.parse(payload.decode("utf-8"), filename=str(record.path))
    except (UnicodeDecodeError, SyntaxError) as error:
        raise GeneratedKDAIndexedPrefillError(f"{label} is invalid Python") from error
    assignments = [
        node
        for node in module.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "EXPECTED_TOOLCHAIN_IDENTITY"
    ]
    _require(len(assignments) == 1, f"{label} toolchain identity is unresolved")
    try:
        value = ast.literal_eval(assignments[0].value)
    except (ValueError, TypeError) as error:
        raise GeneratedKDAIndexedPrefillError(
            f"{label} toolchain identity is not literal"
        ) from error
    return _full_identity(value, f"{label} toolchain identity")


def _source_root() -> Path:
    """Locate imported sources in an installed package or source checkout."""

    from . import env as jit_env

    candidates = (
        jit_env.FLASHINFER_CSRC_DIR / "kda",
        Path(__file__).resolve().parents[2] / "csrc" / "kda",
    )
    for candidate in candidates:
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate.resolve(strict=True)
    raise GeneratedKDAIndexedPrefillError(
        "Generated indexed FlashKDA sources are not installed"
    )


def _safe_relative(value: object, label: str) -> PurePosixPath:
    _require(isinstance(value, str) and bool(value), f"{label} must be a path")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and relative.as_posix() == value
        and ".." not in relative.parts
        and "." not in relative.parts,
        f"{label} must be a normalized relative path",
    )
    _require(
        len(relative.parts) == 1 and relative.name.startswith(_SOURCE_PREFIX),
        f"{label} must be one semantic generated indexed-FlashKDA filename",
    )
    return relative


def _source_record(root: Path, value: object, label: str) -> _SourceRecord:
    record = _object(value, {"path", "sha256", "size_bytes"}, label)
    relative = _safe_relative(record["path"], f"{label}.path")
    size = record["size_bytes"]
    _require(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label}.size_bytes is invalid",
    )
    assert isinstance(size, int) and not isinstance(size, bool)
    digest = _full_sha256(record["sha256"], f"{label}.sha256")
    path = root.joinpath(*relative.parts)
    _require(path.is_file() and not path.is_symlink(), f"{label} is unavailable")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as error:
        raise GeneratedKDAIndexedPrefillError(
            f"{label} escapes the source root"
        ) from error
    payload = path.read_bytes()
    _require(
        len(payload) == size and _sha256(payload) == digest,
        f"{label} content identity differs",
    )
    return _SourceRecord(
        path=path,
        relative=relative.as_posix(),
        sha256=digest,
        size_bytes=size,
    )


def _installed_source_manifest(root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(root.glob(f"{_SOURCE_PREFIX}*")):
        _require(
            not path.is_symlink(), "Generated KDA source closure contains a symlink"
        )
        if path.is_dir():
            continue
        _require(
            path.is_file(), "Generated KDA source closure contains a non-regular file"
        )
        relative = path.name
        if relative == _RECEIPT_NAME:
            continue
        payload = path.read_bytes()
        records.append(
            {
                "path": relative,
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
        )
    return records


def _verify_catalog_receipt(root: Path, catalog_payload: bytes) -> dict[str, str]:
    receipt_path = root / _RECEIPT_NAME
    _require(
        receipt_path.is_file() and not receipt_path.is_symlink(),
        "Generated KDA import receipt is unavailable",
    )
    try:
        receipt = json.loads(receipt_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeneratedKDAIndexedPrefillError(
            "Generated KDA import receipt is invalid"
        ) from error
    receipt = _object(
        receipt,
        {"kind", "schema_version", "catalog_sha256", "inputs", "outputs", "passed"},
        "import receipt",
    )
    _require(
        receipt["kind"] == _IMPORT_RECEIPT_KIND
        and type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["passed"] is True,
        "Generated KDA import receipt identity differs",
    )
    _require(
        _full_sha256(receipt["catalog_sha256"], "import receipt.catalog_sha256")
        == _sha256(catalog_payload),
        "Generated KDA source catalog differs from its import receipt",
    )
    inputs = receipt["inputs"]
    _require(
        isinstance(inputs, list) and len(inputs) == len(_TARGET_ARCHITECTURES),
        "Generated KDA import receipt input denominator differs",
    )
    assert isinstance(inputs, list)
    input_archives: dict[str, str] = {}
    for index, target in enumerate(_TARGET_ARCHITECTURES):
        item = _object(
            inputs[index],
            {"target", "archive_sha256"},
            f"import receipt.inputs[{index}]",
        )
        _require(
            item["target"] == target,
            "Generated KDA import receipt target order differs",
        )
        input_archives[target] = _full_sha256(
            item["archive_sha256"],
            f"import receipt.inputs[{index}].archive_sha256",
        )
    outputs = receipt["outputs"]
    _require(
        isinstance(outputs, list) and outputs == _installed_source_manifest(root),
        "Generated KDA installed source closure differs from its import receipt",
    )
    return input_archives


def _read_catalog(root: Path) -> tuple[_TargetRecord, ...]:
    root = root.resolve(strict=True)
    catalog_path = root / _CATALOG_NAME
    _require(
        catalog_path.is_file() and not catalog_path.is_symlink(),
        "Generated KDA source catalog is unavailable",
    )
    catalog_payload = catalog_path.read_bytes()
    input_archives = _verify_catalog_receipt(root, catalog_payload)
    try:
        catalog = json.loads(catalog_payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise GeneratedKDAIndexedPrefillError(
            "Generated KDA source catalog is invalid"
        ) from error
    catalog = _object(catalog, {"kind", "schema_version", "targets"}, "catalog")
    _require(
        catalog["kind"] == _CATALOG_KIND
        and type(catalog["schema_version"]) is int
        and catalog["schema_version"] == _CATALOG_SCHEMA_VERSION,
        "Generated KDA source catalog identity differs",
    )
    raw_targets = catalog["targets"]
    _require(isinstance(raw_targets, list), "catalog.targets must be a list")
    assert isinstance(raw_targets, list)
    targets: list[_TargetRecord] = []
    for target_index, raw_target in enumerate(raw_targets):
        label = f"catalog.targets[{target_index}]"
        target = _object(
            raw_target,
            {
                "target",
                "architecture",
                "input_archive_sha256",
                "input_fragment_sha256",
                "route_denominator_sha256",
                "dispatcher_seed_identity",
                "toolchain_identity",
                "contract",
                "build_recipe",
                "dispatcher",
                "modules",
            },
            label,
        )
        target_name = target["target"]
        _require(
            isinstance(target_name, str) and target_name in _TARGET_ARCHITECTURES,
            f"{label}.target is invalid",
        )
        assert isinstance(target_name, str)
        target_name = cast(GeneratedKDAIndexedTarget, target_name)
        _require(
            target["architecture"] == _TARGET_ARCHITECTURES[target_name],
            f"{label}.architecture differs",
        )
        input_archive_sha256 = _full_sha256(
            target["input_archive_sha256"], f"{label}.input_archive_sha256"
        )
        _require(
            input_archive_sha256 == input_archives[target_name],
            f"{label}.input_archive_sha256 differs from the import receipt",
        )
        _full_sha256(target["input_fragment_sha256"], f"{label}.input_fragment_sha256")
        _full_sha256(
            target["route_denominator_sha256"],
            f"{label}.route_denominator_sha256",
        )
        seed_identity = target["dispatcher_seed_identity"]
        _require(
            isinstance(seed_identity, str) and seed_identity.startswith("sha256:"),
            f"{label}.dispatcher_seed_identity is invalid",
        )
        assert isinstance(seed_identity, str)
        _full_sha256(
            seed_identity.removeprefix("sha256:"),
            f"{label}.dispatcher_seed_identity",
        )
        _require(
            target["contract"] == _EXPECTED_RUNTIME_CONTRACT,
            f"{label}.contract differs",
        )
        toolchain_identity = _full_identity(
            target["toolchain_identity"],
            f"{label}.toolchain_identity",
        )
        build_recipe = _source_record(
            root,
            target["build_recipe"],
            f"{label}.build_recipe",
        )
        _require(
            build_recipe.path.suffix == ".py"
            and _recipe_toolchain_identity(build_recipe, f"{label}.build_recipe")
            == toolchain_identity,
            f"{label}.build_recipe identity differs",
        )
        dispatcher = _source_record(root, target["dispatcher"], f"{label}.dispatcher")
        _require(
            dispatcher.path.suffix == ".py",
            f"{label}.dispatcher must be Python source",
        )
        raw_modules = target["modules"]
        _require(
            isinstance(raw_modules, list)
            and len(raw_modules) == _EXPECTED_MODULE_COUNTS[target_name],
            f"{label}.modules denominator differs",
        )
        assert isinstance(raw_modules, list)
        modules: list[_ModuleRecord] = []
        for module_index, raw_module in enumerate(raw_modules):
            module_label = f"{label}.modules[{module_index}]"
            module = _object(
                raw_module,
                {
                    "id",
                    "module_ident",
                    "entry_point",
                    "kernel_name",
                    "compile_options",
                    "cooperative",
                    "tma_abi",
                    "use_pdl",
                    "cuda_source",
                    "host_source",
                    "expected_cubin",
                },
                module_label,
            )
            module_id = _safe_token(module["id"], f"{module_label}.id")
            module_ident = _identifier(
                module["module_ident"], f"{module_label}.module_ident"
            )
            entry_point = _identifier(
                module["entry_point"], f"{module_label}.entry_point"
            )
            kernel_name = _identifier(
                module["kernel_name"], f"{module_label}.kernel_name"
            )
            _require(
                type(module["cooperative"]) is bool
                and type(module["use_pdl"]) is bool
                and module["tma_abi"] == "pointer",
                f"{module_label} launch metadata is invalid",
            )
            options = module["compile_options"]
            _require(
                isinstance(options, list)
                and all(isinstance(option, str) and option for option in options),
                f"{module_label}.compile_options are invalid",
            )
            assert isinstance(options, list)
            compile_options = cast(list[str], options)
            cuda_source = _source_record(
                root, module["cuda_source"], f"{module_label}.cuda_source"
            )
            host_source = _source_record(
                root, module["host_source"], f"{module_label}.host_source"
            )
            _require(
                cuda_source.path.suffix == ".cu"
                and host_source.path.suffix in {".cc", ".cpp", ".cxx"},
                f"{module_label} source suffix differs",
            )
            expected_cubin = _object(
                module["expected_cubin"],
                {"sha256", "size_bytes"},
                f"{module_label}.expected_cubin",
            )
            cubin_size = expected_cubin["size_bytes"]
            _require(
                isinstance(cubin_size, int)
                and not isinstance(cubin_size, bool)
                and cubin_size > 0,
                f"{module_label}.expected_cubin.size_bytes is invalid",
            )
            assert isinstance(cubin_size, int) and not isinstance(cubin_size, bool)
            modules.append(
                _ModuleRecord(
                    id=module_id,
                    module_ident=module_ident,
                    entry_point=entry_point,
                    kernel_name=kernel_name,
                    compile_options=tuple(compile_options),
                    cuda_source=cuda_source,
                    host_source=host_source,
                    cubin_sha256=_full_sha256(
                        expected_cubin["sha256"],
                        f"{module_label}.expected_cubin.sha256",
                    ),
                    cubin_size_bytes=cubin_size,
                )
            )
        module_ids = [module.id for module in modules]
        _require(
            len(module_ids) == len(set(module_ids)),
            f"{label}.modules contain duplicate ids",
        )
        targets.append(
            _TargetRecord(
                target=target_name,
                architecture=str(target["architecture"]),
                toolchain_identity=toolchain_identity,
                build_recipe=build_recipe,
                dispatcher=dispatcher,
                modules=tuple(modules),
            )
        )
    _require(
        [target.target for target in targets] == list(_TARGET_ARCHITECTURES),
        "catalog target order or denominator differs",
    )
    return tuple(targets)


@functools.cache
def _catalog() -> tuple[_TargetRecord, ...]:
    return _read_catalog(_source_root())


def _target_record(target: GeneratedKDAIndexedTarget) -> _TargetRecord:
    for record in _catalog():
        if record.target == target:
            return record
    raise GeneratedKDAIndexedPrefillError(
        f"Generated KDA target {target!r} is unavailable"
    )


def _target_for_device(device: torch.device) -> GeneratedKDAIndexedTarget:
    capability = get_compute_capability(device)
    try:
        return _TARGET_COMPUTE_CAPABILITIES[capability]
    except KeyError as error:
        raise GeneratedKDAIndexedPrefillError(
            f"Generated KDA indexed prefill requires CC 10.0 or 10.3, got {capability}"
        ) from error


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
        if entry:
            candidates.extend(
                sorted(glob(str(Path(entry) / "nvidia" / "cu*" / "include")))
            )
            candidates.append(str(Path(entry) / "nvidia" / "cuda_runtime" / "include"))
            candidates.append(
                str(Path(entry) / "triton" / "backends" / "nvidia" / "include")
            )
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        path = Path(candidate).resolve()
        if path in seen:
            continue
        seen.add(path)
        if any(
            (path / marker).is_file()
            for marker in ("cuda.h", "cuda_runtime.h", "cuda_bf16.h")
        ):
            result.append(path)
    _require(
        bool(result), "CUDA headers required by the generated source are unavailable"
    )
    return tuple(result)


def _compile_exact_cubin(module: _ModuleRecord, target: _TargetRecord) -> bytes:
    source = module.cuda_source.path.read_bytes()
    _require(
        len(source) == module.cuda_source.size_bytes
        and _sha256(source) == module.cuda_source.sha256,
        f"CUDA source drifted for {module.id}",
    )
    worker = Path(__file__).with_name("flash_kda_indexed_nvrtc.py")
    _require(
        worker.is_file() and not worker.is_symlink(),
        "isolated Generated KDA NVRTC worker is unavailable",
    )
    with tempfile.TemporaryDirectory(prefix="generated_kda_nvrtc_") as temporary:
        output = Path(temporary) / "generated_kda_kernel.cubin"
        command = [
            sys.executable,
            "-I",
            str(worker),
            "--source",
            str(module.cuda_source.path),
            "--output",
            str(output),
            "--architecture",
            target.architecture,
            "--toolchain-identity",
            target.toolchain_identity,
            "--source-sha256",
            module.cuda_source.sha256,
            "--source-size-bytes",
            str(module.cuda_source.size_bytes),
            "--cubin-sha256",
            module.cubin_sha256,
            "--cubin-size-bytes",
            str(module.cubin_size_bytes),
        ]
        command.extend(f"--include-dir={include}" for include in _cuda_include_dirs())
        command.extend(
            f"--compile-option={option}" for option in module.compile_options
        )
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=_NVRTC_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as error:
            raise GeneratedKDAIndexedPrefillError(
                f"isolated NVRTC compilation timed out for {target.target}/{module.id}"
            ) from error
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise GeneratedKDAIndexedPrefillError(
                f"isolated NVRTC compilation failed for {target.target}/{module.id}: "
                f"{detail or 'worker returned no diagnostic'}"
            )
        _require(
            output.is_file() and not output.is_symlink(),
            f"isolated NVRTC worker produced no cubin for {target.target}/{module.id}",
        )
        cubin = output.read_bytes()
    _require(
        len(cubin) == module.cubin_size_bytes and _sha256(cubin) == module.cubin_sha256,
        f"rebuilt cubin identity differs for {target.target}/{module.id}",
    )
    return cubin


def _module_build_directory(module: _ModuleRecord, target: _TargetRecord) -> Path:
    from . import env as jit_env

    identity = hashlib.sha256(
        (
            target.target
            + module.id
            + module.cuda_source.sha256
            + module.host_source.sha256
            + module.cubin_sha256
            + target.toolchain_identity
            + target.build_recipe.sha256
        ).encode()
    ).hexdigest()[:16]
    return jit_env.FLASHINFER_JIT_DIR / (
        f"generated_kda_indexed_{target.target}_{module.module_ident}_{identity}"
    )


def _exact_cubin(module: _ModuleRecord, target: _TargetRecord) -> tuple[bytes, Path]:
    build_directory = _module_build_directory(module, target)
    build_directory.mkdir(parents=True, exist_ok=True)
    cubin_path = build_directory / f"{module.module_ident}.cubin"
    with FileLock(build_directory / f"{module.module_ident}.lock", thread_local=False):
        if cubin_path.is_file():
            cubin = cubin_path.read_bytes()
            _require(
                len(cubin) == module.cubin_size_bytes
                and _sha256(cubin) == module.cubin_sha256,
                f"cached cubin identity differs for {target.target}/{module.id}",
            )
        else:
            cubin = _compile_exact_cubin(module, target)
            with tempfile.NamedTemporaryFile(
                dir=build_directory,
                prefix=f".{module.module_ident}.",
                suffix=".tmp.cubin",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(cubin)
            os.replace(temporary, cubin_path)
    return cubin, build_directory


@functools.cache
def _load_entrypoint(
    target_name: GeneratedKDAIndexedTarget, module_id: str
) -> Callable[..., object]:
    from tvm_ffi import cpp

    target = _target_record(target_name)
    module = next((record for record in target.modules if record.id == module_id), None)
    _require(module is not None, f"unknown Generated KDA module id {module_id!r}")
    assert module is not None
    host_source = module.host_source.path.read_bytes()
    _require(
        len(host_source) == module.host_source.size_bytes
        and _sha256(host_source) == module.host_source.sha256,
        f"host source drifted for {module.id}",
    )
    cubin, build_directory = _exact_cubin(module, target)
    module_name = build_directory.name
    loaded = cpp.load_inline(
        module_name,
        cpp_sources=host_source.decode("utf-8"),
        embed_cubin={module.module_ident: cubin},
        extra_include_paths=[str(path) for path in _cuda_include_dirs()],
        extra_cflags=["-O3"],
        extra_ldflags=["-lcuda"],
        build_directory=str(build_directory),
    )
    entrypoint = getattr(loaded, module.entry_point, None)
    _require(
        callable(entrypoint),
        f"loaded host module lacks entrypoint {module.entry_point!r}",
    )
    return entrypoint


class _LazyEntrypoint:
    __slots__ = ("_module_id", "_target")

    def __init__(self, target: GeneratedKDAIndexedTarget, module_id: str) -> None:
        self._target = target
        self._module_id = module_id

    def __call__(self, *args: object) -> object:
        return _load_entrypoint(self._target, self._module_id)(*args)


def _load_dispatcher_source(record: _TargetRecord) -> dict[str, Callable[..., object]]:
    payload = record.dispatcher.path.read_bytes()
    _require(
        len(payload) == record.dispatcher.size_bytes
        and _sha256(payload) == record.dispatcher.sha256,
        f"dispatcher source drifted for {record.target}",
    )
    namespace: dict[str, Any] = {
        "__file__": str(record.dispatcher.path),
        "__name__": f"generated_kda_indexed_dispatcher_{record.target}",
    }
    try:
        exec(compile(payload, str(record.dispatcher.path), "exec"), namespace)
    except (SyntaxError, UnicodeDecodeError) as error:
        raise GeneratedKDAIndexedPrefillError(
            f"dispatcher source is invalid for {record.target}"
        ) from error
    module_ids = namespace.get("FLASHINFER_MODULE_IDS")
    expected_ids = tuple(module.id for module in record.modules)
    _require(
        module_ids == expected_ids,
        f"dispatcher/module order differs for {record.target}",
    )
    binder = namespace.get("bind_loaded_modules")
    _require(callable(binder), f"dispatcher binder is unavailable for {record.target}")
    lazy_modules = {
        module.id: _LazyEntrypoint(record.target, module.id)
        for module in record.modules
    }
    bound = binder(lazy_modules)
    _require(
        isinstance(bound, Mapping)
        and set(bound) == {"select_fp32_indexed_schedule_route", "prepare_fwd"}
        and all(callable(value) for value in bound.values()),
        f"dispatcher public ABI differs for {record.target}",
    )
    return dict(bound)


@functools.cache
def get_flash_kda_indexed_prefill_dispatcher(
    target: GeneratedKDAIndexedTarget,
) -> Mapping[str, Callable[..., object]]:
    """Bind one verified generated dispatcher to lazy source-built modules."""

    return MappingProxyType(_load_dispatcher_source(_target_record(target)))


def flash_kda_indexed_prefill_is_eligible(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    use_gate_in_kernel: bool,
    lower_bound: Optional[float],
    cu_seqlens: Optional[torch.Tensor],
    ssm_state_indices: Optional[torch.Tensor],
    num_spec_tokens: Optional[int],
    num_accepted_tokens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    initial_state_source: Optional[torch.Tensor],
    initial_state_indices: Optional[torch.Tensor],
    beta_is_logit: bool,
    seq_order: Optional[torch.Tensor],
    prefill_workspace: Optional[object],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Return whether a call is inside the exact exported FP32-indexed domain."""

    if (
        not isinstance(q, torch.Tensor)
        or q.ndim != 4
        or not q.is_cuda
        or get_compute_capability(q.device) not in _TARGET_COMPUTE_CAPABILITIES
        or q.shape[1] <= 1
        or q.shape[-1] != 128
        or q.dtype != torch.bfloat16
        or not q.is_contiguous()
    ):
        return False
    batch_size, tokens, heads, _ = q.shape
    if batch_size <= 0 or heads <= 0:
        return False
    if any(
        not isinstance(tensor, torch.Tensor)
        or tensor.device != q.device
        or tensor.dtype != torch.bfloat16
        or tensor.shape != q.shape
        or not tensor.is_contiguous()
        for tensor in (k, v, g)
    ):
        return False
    if (
        not isinstance(beta, torch.Tensor)
        or beta.device != q.device
        or beta.dtype != torch.bfloat16
        or beta.shape != (batch_size, tokens, heads)
        or not beta.is_contiguous()
    ):
        return False
    if (
        not isinstance(A_log, torch.Tensor)
        or A_log.device != q.device
        or A_log.dtype != torch.float32
        or A_log.shape != (heads,)
        or not A_log.is_contiguous()
        or not isinstance(dt_bias, torch.Tensor)
        or dt_bias.device != q.device
        or dt_bias.dtype != torch.float32
        or dt_bias.shape != (heads, 128)
        or not dt_bias.is_contiguous()
    ):
        return False
    num_sequences = batch_size
    if cu_seqlens is not None:
        if (
            batch_size != 1
            or not isinstance(cu_seqlens, torch.Tensor)
            or cu_seqlens.device != q.device
            or cu_seqlens.dtype != torch.int64
            or cu_seqlens.ndim != 1
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() < 2
        ):
            return False
        num_sequences = cu_seqlens.numel() - 1
    state_inner = heads * 128 * 128
    if (
        not isinstance(initial_state, torch.Tensor)
        or initial_state.device != q.device
        or initial_state.dtype != torch.float32
        or initial_state.shape != (_EXPECTED_STATE_POOL_CAPACITY, heads, 128, 128)
        or tuple(initial_state.stride()) != (state_inner, 128 * 128, 128, 1)
        or initial_state.storage_offset() != 0
        or not isinstance(ssm_state_indices, torch.Tensor)
        or ssm_state_indices.device != q.device
        or ssm_state_indices.dtype != torch.int32
        or ssm_state_indices.ndim != 1
        or ssm_state_indices.numel() != num_sequences
        or not ssm_state_indices.is_contiguous()
    ):
        return False
    if output is not None and (
        not isinstance(output, torch.Tensor)
        or output.device != q.device
        or output.dtype != torch.bfloat16
        or output.shape != q.shape
        or not output.is_contiguous()
    ):
        return False
    return (
        use_qk_l2norm_in_kernel
        and use_gate_in_kernel
        and beta_is_logit
        and isinstance(lower_bound, (int, float))
        and not isinstance(lower_bound, bool)
        and float(lower_bound) == -5.0
        and num_spec_tokens is None
        and num_accepted_tokens is None
        and initial_state_source is None
        and initial_state_indices is None
        and seq_order is None
        and prefill_workspace is None
        and state_checkpoints is None
        and checkpoint_cu_starts is None
        and checkpoint_every_n_tokens == 0
    )


def _run_flash_kda_indexed_prefill(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: Optional[float],
    initial_state: torch.Tensor,
    output_final_state: bool,
    lower_bound: float,
    cu_seqlens: Optional[torch.Tensor],
    output: Optional[torch.Tensor],
    state_indices: torch.Tensor,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Prepare, launch, and close one generated FP32-indexed KDA dispatch."""

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Generated KDA indexed prefill must be warmed and launched outside CUDA graph capture"
        )
    from ..kda_prefill import _check_output_does_not_overlap_inputs

    if output is not None:
        _check_output_does_not_overlap_inputs(
            output,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
        )
    target = _target_for_device(q.device)
    dispatcher = get_flash_kda_indexed_prefill_dispatcher(target)
    out = torch.empty_like(q) if output is None else output
    prepared_object = dispatcher["prepare_fwd"](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        output=out,
        seq_order=None,
        prefill_workspace=None,
        state_indices=state_indices,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=0,
    )
    _require(
        callable(getattr(prepared_object, "launch", None))
        and callable(getattr(prepared_object, "close", None)),
        "generated dispatcher returned an invalid prepared launch",
    )
    prepared = cast(_PreparedLaunch, prepared_object)
    try:
        result = prepared.launch()
    finally:
        prepared.close()
    _require(
        isinstance(result, tuple) and len(result) == 2,
        "generated dispatcher returned an invalid public result",
    )
    assert isinstance(result, tuple)
    expected_state = initial_state if output_final_state else None
    _require(
        result[0] is out and result[1] is expected_state,
        "generated dispatcher returned objects outside the public ABI",
    )
    return out, expected_state


__all__ = [
    "GeneratedKDAIndexedPrefillError",
    "flash_kda_indexed_prefill_is_eligible",
    "get_flash_kda_indexed_prefill_dispatcher",
]
