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

Checksum-bound source loader for the TRT-LLM MLA Blackwell backend.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

import torch
from filelock import FileLock
from tvm_ffi import cpp

from . import env as jit_env
from .core import logger


_TARGETS = {
    "sm_100a_148": ("sm_100a", 148),
    "sm_100a_152": ("sm_100a", 152),
    "sm_103a_148": ("sm_103a", 148),
    "sm_103a_152": ("sm_103a", 152),
}
_TARGET_ORDER = tuple(_TARGETS)
_ARCH_CAPABILITIES = {"sm_100a": (10, 0), "sm_103a": (10, 3)}
_SOURCE_CATALOG_RELATIVE_PATH = Path("generated") / "source_catalog.json"
_DOMAIN_DEVICE_COUNTS = {
    "mla_bf16_vquarter": 1,
    "mla_bf16_vhalf": 1,
    "mla_bf16_unsplit": 1,
    "mla_bf16_clc": 8,
    "mla_bf16_tail": 1,
    "mla_fp8_tail": 1,
    "mla_fp8_p32_qk_l2": 1,
    "mla_fp8_page64_pdl": 2,
    "mla_bf16_native_split8_pdl": 2,
}
_DOMAIN_ORDER = tuple(_DOMAIN_DEVICE_COUNTS)
_EXPORTED_COMPILE_FLAGS = ["--use_fast_math"]
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_C_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _source_dir() -> Path:
    """Locate the generated MLA source package in installs and checkouts."""

    packaged = jit_env.FLASHINFER_CSRC_DIR / "mla" / "trtllm_mla_blackwell"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[2] / "csrc" / "mla" / "trtllm_mla_blackwell"


def _source_record(
    value: object,
    *,
    domain: str,
    kind: str,
    target: str | None = None,
    index: int | None = None,
) -> Mapping[str, object]:
    location_parts = [domain]
    if target is not None:
        location_parts.append(target)
    location_parts.append(kind if index is None else f"{kind}[{index}]")
    location = "/".join(location_parts)
    expected_keys = {"path", "sha256"}
    if kind == "device_source":
        expected_keys.update({"module_ident", "compile_flags"})
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise RuntimeError(f"TRT-LLM MLA Blackwell catalog {location} schema is invalid")

    relative = value["path"]
    sha256 = value["sha256"]
    suffix = ".cpp" if kind == "host_source" else ".cu"
    if not isinstance(relative, str):
        raise RuntimeError(f"TRT-LLM MLA Blackwell catalog {location} path is invalid")
    path = PurePosixPath(relative)
    if (
        not relative
        or "\\" in relative
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != relative
        or path.suffix != suffix
    ):
        raise RuntimeError(
            f"TRT-LLM MLA Blackwell catalog {location} path is noncanonical: {relative!r}"
        )
    if not isinstance(sha256, str) or _HEX_SHA256.fullmatch(sha256) is None:
        raise RuntimeError(f"TRT-LLM MLA Blackwell catalog {location} sha256 is invalid")

    if kind == "device_source":
        module_ident = value["module_ident"]
        compile_flags = value["compile_flags"]
        if (
            not isinstance(module_ident, str)
            or _C_IDENTIFIER.fullmatch(module_ident) is None
        ):
            raise RuntimeError(
                f"TRT-LLM MLA Blackwell catalog {location} module identity is invalid"
            )
        if compile_flags != _EXPORTED_COMPILE_FLAGS:
            raise RuntimeError(
                f"TRT-LLM MLA Blackwell catalog {location} compile flags differ from "
                f"the exported contract: {compile_flags!r}"
            )
    return value


@functools.cache
def _source_catalog() -> Mapping[str, object]:
    """Load and validate the exact physical-target generated-source catalog."""

    catalog_path = _source_dir() / _SOURCE_CATALOG_RELATIVE_PATH
    if not catalog_path.is_file():
        raise RuntimeError(
            f"TRT-LLM MLA Blackwell generated-source catalog is missing: {catalog_path}"
        )
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"TRT-LLM MLA Blackwell generated-source catalog is unreadable: {catalog_path}"
        ) from error
    if not isinstance(catalog, dict) or set(catalog) != {
        "schema_version",
        "target_order",
        "targets",
        "domain_order",
        "domains",
    }:
        raise RuntimeError("TRT-LLM MLA Blackwell generated-source catalog schema is invalid")
    expected_targets = {
        target: {"arch": arch, "multi_processor_count": multi_processor_count}
        for target, (arch, multi_processor_count) in _TARGETS.items()
    }
    if (
        isinstance(catalog["schema_version"], bool)
        or catalog["schema_version"] != 3
        or catalog["target_order"] != list(_TARGET_ORDER)
        or not isinstance(catalog["targets"], dict)
        or tuple(catalog["targets"]) != _TARGET_ORDER
        or catalog["targets"] != expected_targets
        or catalog["domain_order"] != list(_DOMAIN_ORDER)
    ):
        raise RuntimeError(
            "TRT-LLM MLA Blackwell generated-source catalog identity is invalid"
        )

    domains = catalog["domains"]
    if not isinstance(domains, dict) or tuple(domains) != _DOMAIN_ORDER:
        raise RuntimeError(
            "TRT-LLM MLA Blackwell generated-source catalog domain topology is invalid"
        )
    source_paths: set[str] = set()
    module_idents: set[str] = set()
    device_source_count = {target: 0 for target in _TARGET_ORDER}
    for domain, expected_device_count in _DOMAIN_DEVICE_COUNTS.items():
        profile = domains[domain]
        if not isinstance(profile, dict) or set(profile) != {
            "host_source",
            "device_sources",
        }:
            raise RuntimeError(
                f"TRT-LLM MLA Blackwell catalog domain {domain!r} schema is invalid"
            )
        host = _source_record(
            profile["host_source"], domain=domain, kind="host_source"
        )
        devices_by_target = profile["device_sources"]
        if (
            not isinstance(devices_by_target, dict)
            or tuple(devices_by_target) != _TARGET_ORDER
        ):
            raise RuntimeError(
                f"TRT-LLM MLA catalog domain {domain!r} target "
                "inventory is invalid"
            )
        host_path = str(host["path"])
        if host_path != f"host/{domain}.cpp" or host_path in source_paths:
            raise RuntimeError("TRT-LLM MLA catalog host source paths are invalid")
        source_paths.add(host_path)
        expected_idents: tuple[str, ...] | None = None
        for target in _TARGET_ORDER:
            devices = devices_by_target[target]
            if not isinstance(devices, list) or len(devices) != expected_device_count:
                raise RuntimeError(
                    f"TRT-LLM MLA catalog domain {domain!r}/{target} must contain "
                    f"exactly {expected_device_count} device sources"
                )
            records = [
                _source_record(
                    device,
                    domain=domain,
                    target=target,
                    kind="device_source",
                    index=index,
                )
                for index, device in enumerate(devices)
            ]
            idents = tuple(str(record["module_ident"]) for record in records)
            if len(set(idents)) != len(idents):
                raise RuntimeError(
                    "TRT-LLM MLA catalog contains duplicate device module "
                    "identities"
                )
            if expected_idents is None:
                expected_idents = idents
                if any(ident in module_idents for ident in idents):
                    raise RuntimeError(
                        "TRT-LLM MLA catalog contains duplicate device module "
                        "identities"
                    )
                module_idents.update(idents)
            elif idents != expected_idents:
                raise RuntimeError(
                    f"TRT-LLM MLA catalog domain {domain!r} device identity "
                    "order differs across targets"
                )
            paths = [str(record["path"]) for record in records]
            expected_paths = [f"device/{target}/{ident}.cu" for ident in idents]
            if paths != expected_paths or any(path in source_paths for path in paths):
                raise RuntimeError(
                    "TRT-LLM MLA catalog device source paths are invalid"
                )
            source_paths.update(paths)
            device_source_count[target] += len(records)
    if len(module_idents) != 18 or device_source_count != {
        target: 18 for target in _TARGET_ORDER
    }:
        raise RuntimeError(
            "TRT-LLM MLA generated-source catalog must contain exactly 18 "
            "device sources for each supported target"
        )
    return catalog


def _domain_profile(domain: str, target: str) -> Mapping[str, object]:
    if domain not in _DOMAIN_DEVICE_COUNTS:
        raise ValueError(
            f"unknown TRT-LLM MLA domain {domain!r}; expected one of "
            f"{list(_DOMAIN_ORDER)!r}"
        )
    if target not in _TARGETS:
        raise ValueError(
            f"unsupported TRT-LLM MLA target {target!r}; expected one of "
            f"{list(_TARGET_ORDER)!r}"
        )
    domains = _source_catalog()["domains"]
    assert isinstance(domains, dict)
    profile = domains[domain]
    assert isinstance(profile, dict)
    return profile


def _target_key(device: torch.device | int | str | None = None) -> str:
    capability = torch.cuda.get_device_capability(device)
    multi_processor_count = torch.cuda.get_device_properties(
        device
    ).multi_processor_count
    for target, (arch, expected_multi_processor_count) in _TARGETS.items():
        if (
            capability == _ARCH_CAPABILITIES[arch]
            and multi_processor_count == expected_multi_processor_count
        ):
            return target
    raise ValueError(
        "TRT-LLM MLA requires one of the exact targets "
        f"{list(_TARGET_ORDER)!r}, got compute capability "
        f"{capability[0]}.{capability[1]} with {multi_processor_count} SMs"
    )


def _sealed_source_bytes(
    source_dir: Path,
    record: Mapping[str, object],
) -> tuple[Path, bytes]:
    relative = record["path"]
    expected_sha256 = record["sha256"]
    assert isinstance(relative, str)
    assert isinstance(expected_sha256, str)
    generated_root = (source_dir / "generated").resolve()
    path = (generated_root / relative).resolve()
    if generated_root not in path.parents:
        raise RuntimeError(
            f"TRT-LLM MLA Blackwell generated source path escapes its package: {path}"
        )
    if not path.is_file():
        raise RuntimeError(f"TRT-LLM MLA Blackwell generated source is missing: {path}")
    payload = path.read_bytes()
    actual_sha256 = hashlib.sha256(payload).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "TRT-LLM MLA Blackwell generated source identity drift: "
            f"{path} has sha256={actual_sha256}, expected {expected_sha256}"
        )
    return path, payload


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the TRT-LLM MLA Blackwell backend")
    return Path(candidate).resolve()


def _digest_field(digest, value: bytes) -> None:
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


@functools.cache
def _load_domain_module(domain: str, target: str):
    """Compile and load one catalog-bound domain for an exact physical target."""

    profile = _domain_profile(domain, target)
    arch, multi_processor_count = _TARGETS[target]
    source_dir = _source_dir()
    host = profile["host_source"]
    devices_by_target = profile["device_sources"]
    assert isinstance(host, dict)
    assert isinstance(devices_by_target, dict)
    devices = devices_by_target[target]
    assert isinstance(devices, list)
    _, host_payload = _sealed_source_bytes(source_dir, host)
    nvcc = _nvcc()

    digest = hashlib.sha256()
    for value in (
        domain.encode(),
        target.encode(),
        arch.encode(),
        str(multi_processor_count).encode(),
        str(nvcc).encode(),
        host_payload,
    ):
        _digest_field(digest, value)
    resolved_devices: list[tuple[str, Path, tuple[str, ...]]] = []
    for device in devices:
        assert isinstance(device, dict)
        module_ident = device["module_ident"]
        compile_flags = device["compile_flags"]
        assert isinstance(module_ident, str)
        assert isinstance(compile_flags, list)
        device_path, device_payload = _sealed_source_bytes(source_dir, device)
        flags = tuple(compile_flags)
        for value in (
            module_ident.encode(),
            device_payload,
            "\0".join(flags).encode(),
        ):
            _digest_field(digest, value)
        resolved_devices.append((module_ident, device_path, flags))

    module_name = f"trtllm_mla_{domain}_{target}_{digest.hexdigest()[:16]}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    lock_path = build_dir / f"{module_name}.lock"
    with FileLock(lock_path, thread_local=False):
        cubins: dict[str, bytes] = {}
        for module_ident, device_path, compile_flags in resolved_devices:
            cubin_path = build_dir / f"{module_ident}.cubin"
            if not cubin_path.is_file():
                temporary = build_dir / f"{module_ident}.{os.getpid()}.tmp.cubin"
                command = [
                    str(nvcc),
                    "-cubin",
                    f"-arch={arch}",
                    "--std=c++17",
                    "-O3",
                    "-I",
                    str(nvcc.parent.parent / "include"),
                    *compile_flags,
                    str(device_path),
                    "-o",
                    str(temporary),
                ]
                process = subprocess.run(command, text=True, capture_output=True)
                if process.returncode != 0:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(
                        f"TRT-LLM MLA Blackwell nvcc failed for {domain}/"
                        f"{module_ident} ({target}, {arch}):\n{process.stderr}"
                    )
                os.replace(temporary, cubin_path)
            cubins[module_ident] = cubin_path.read_bytes()

        module = cpp.load_inline(
            module_name,
            cpp_sources=host_payload.decode("utf-8"),
            embed_cubin=cubins,
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )
    logger.info(
        "Loaded TRT-LLM MLA domain %s for target %s (%s)", domain, target, arch
    )
    return module


def get_domain_module(
    domain: str, device: torch.device | int | str | None = None
):
    """Return the cached source-built module for one exact public domain."""

    if domain not in _DOMAIN_DEVICE_COUNTS:
        raise ValueError(
            f"unknown TRT-LLM MLA domain {domain!r}; expected one of "
            f"{list(_DOMAIN_ORDER)!r}"
        )
    return _load_domain_module(domain, _target_key(device))


__all__ = ["get_domain_module"]
