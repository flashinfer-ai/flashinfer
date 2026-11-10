"""JIT loader for the generated SM103 request-ordered paged-decode programs."""

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
_SCHEMA = "flashinfer.cake_fmha_request_ordered_paged_decode.v2"
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
_KINDS = ("persistent", "two_wave", "low_q1_cga")
_SPECIALIZATIONS = ("Q_LEN", "WRITE_LSE", "NUM_SPLIT")


@dataclass(frozen=True)
class CakeFmhaRequestOrderedModuleSpec:
    """One program instantiation: a device source, its launcher and its defines."""

    name: str
    device_path: Path
    binding_path: Path
    module_ident: str
    kernel_symbol: str
    ffi_entry: str
    compile_options: tuple[str, ...]
    defines: tuple[tuple[str, int], ...]
    tma_workspace_bytes: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid request-ordered FMHA manifest: {message}")


def _source_root() -> Path:
    installed = (
        jit_env.FLASHINFER_CSRC_DIR / "cake_fmha" / "request_ordered_paged_decode"
    )
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_fmha"
        / "request_ordered_paged_decode"
    )
    for candidate in (installed, checkout):
        if (candidate / _MANIFEST_NAME).is_file():
            return candidate
    raise FileNotFoundError(
        "request-ordered Cake FMHA sources were not found; checked "
        f"{installed} and {checkout}"
    )


def _source(root: Path, value: object, label: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{label}")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("generated", "sm_103a")
        and len(relative.parts) == 3,
        f"{label}",
    )
    path = root.joinpath(*relative.parts)
    _require(path.is_file(), f"{label} does not exist")
    return path


def _identifier(value: object) -> bool:
    return isinstance(value, str) and bool(value) and value.replace("_", "a").isalnum()


@functools.cache
def get_cake_fmha_request_ordered_manifest() -> dict[str, Any]:
    """Load the module table: programs, their defines and the exact exported routes."""

    root = _source_root()
    payload: Any = json.loads((root / _MANIFEST_NAME).read_text())
    _require(isinstance(payload, dict), "root")
    _require(payload.get("schema") == _SCHEMA, "schema")
    _require(payload.get("target") == "sm_103a", "target")
    _require(payload.get("contract") == _CONTRACT, "contract")
    modules = payload.get("modules")
    _require(isinstance(modules, list) and bool(modules), "modules")
    modules_by_name: dict[str, dict[str, Any]] = {}
    for index, module in enumerate(modules):
        _require(isinstance(module, dict), f"modules[{index}]")
        name = module.get("name")
        _require(
            isinstance(name, str)
            and name.startswith("cake_fmha_request_ordered_paged_decode_")
            and name.replace("_", "").isalnum(),
            f"modules[{index}].name",
        )
        _require(name not in modules_by_name, f"duplicate module {name}")
        _source(root, module.get("device_path"), f"modules[{index}].device_path")
        _source(root, module.get("binding_path"), f"modules[{index}].binding_path")
        for field in ("module_ident", "kernel_symbol", "ffi_entry"):
            _require(_identifier(module.get(field)), f"modules[{index}].{field}")
        _require(
            module.get("compile_options") == ["--use_fast_math"],
            f"modules[{index}].compile_options",
        )
        _require(
            module.get("tma_workspace_bytes") == 384,
            f"modules[{index}].tma_workspace_bytes",
        )
        defines = module.get("defines")
        _require(
            isinstance(defines, dict)
            and set(defines) <= set(_SPECIALIZATIONS)
            and all(isinstance(value, int) for value in defines.values()),
            f"modules[{index}].defines",
        )
        _require(module.get("kind") in _KINDS, f"modules[{index}].kind")
        _require(module.get("q_len") in (1, 6), f"modules[{index}].q_len")
        _require(
            isinstance(module.get("write_lse"), bool), f"modules[{index}].write_lse"
        )
        _require(
            isinstance(module.get("num_split"), int) and module["num_split"] >= 1,
            f"modules[{index}].num_split",
        )
        expected_defines = {
            "Q_LEN": module["q_len"],
            "NUM_SPLIT": module["num_split"],
            "WRITE_LSE": int(module["write_lse"]),
        }
        _require(
            all(defines[key] == expected_defines[key] for key in defines),
            f"modules[{index}].defines disagree with q_len/num_split/write_lse",
        )
        modules_by_name[name] = module
    routes = payload.get("exact_routes")
    _require(isinstance(routes, list), "exact_routes")
    for index, route in enumerate(routes):
        _require(isinstance(route, dict), f"exact_routes[{index}]")
        _require(
            route.get("module") in modules_by_name, f"exact_routes[{index}].module"
        )
        target = modules_by_name[route["module"]]
        _require(
            route.get("q_len") == target["q_len"]
            and route.get("write_lse") is target["write_lse"],
            f"exact_routes[{index}] disagrees with module {route['module']}",
        )
        lengths = route.get("kv_lens")
        _require(
            isinstance(lengths, dict)
            and isinstance(lengths.get("period"), list)
            and bool(lengths["period"])
            and all(isinstance(value, int) for value in lengths["period"])
            and isinstance(lengths.get("count"), int)
            and lengths["count"] >= len(lengths["period"]),
            f"exact_routes[{index}].kv_lens",
        )
        grid = route.get("grid")
        _require(
            isinstance(grid, list)
            and len(grid) == 3
            and all(isinstance(v, int) for v in grid),
            f"exact_routes[{index}].grid",
        )
    return payload


@functools.cache
def get_cake_fmha_request_ordered_module_spec(
    name: str,
) -> CakeFmhaRequestOrderedModuleSpec:
    root = _source_root()
    manifest = get_cake_fmha_request_ordered_manifest()
    matches = [module for module in manifest["modules"] if module["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"unknown request-ordered FMHA module: {name}")
    module = matches[0]
    spec = CakeFmhaRequestOrderedModuleSpec(
        name=name,
        device_path=_source(root, module["device_path"], f"module {name} device"),
        binding_path=_source(root, module["binding_path"], f"module {name} binding"),
        module_ident=module["module_ident"],
        kernel_symbol=module["kernel_symbol"],
        ffi_entry=module["ffi_entry"],
        compile_options=tuple(module["compile_options"]),
        defines=tuple(sorted(module["defines"].items())),
        tma_workspace_bytes=module["tma_workspace_bytes"],
    )
    binding = spec.binding_path.read_text(encoding="utf-8")
    shared = "CAKE_RO_EMBED(CAKE_RO_MODULE_IDENT);" in binding
    embed = (
        "CAKE_RO_EMBED(CAKE_RO_MODULE_IDENT);"
        if shared
        else f"TVM_FFI_EMBED_CUBIN({spec.module_ident});"
    )
    _require(binding.count(embed) == 1, f"module {name} embedded-cubin declaration")
    module_global = (
        "CAKE_RO_CONCAT(EmbedCubinModule_, CAKE_RO_MODULE_IDENT)::Global()"
        if shared
        else f"EmbedCubinModule_{spec.module_ident}::Global()"
    )
    _require(
        binding.count(f'{module_global}->mod.GetKernel("{spec.kernel_symbol}")') == 1,
        f"module {name} kernel lookup",
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
    options.extend(f"-D{name}={value}" for name, value in spec.defines)
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
                "binding_sha256": _file_identity(spec.binding_path)["sha256"],
                "compile_options": list(_nvrtc_options(spec)),
                "device_sha256": _file_identity(spec.device_path)["sha256"],
                "module_ident": spec.module_ident,
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
    """NVRTC-compile and load one SM103 program instantiation."""

    import torch
    from tvm_ffi import cpp

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("request-ordered Cake FMHA requires compute capability 10.3")
    spec = get_cake_fmha_request_ordered_module_spec(name)
    root = _source_root()
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
        extra_cflags=["-O3", f"-DCAKE_RO_MODULE_IDENT={spec.module_ident}"],
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
    "load_cake_fmha_request_ordered_module",
]
