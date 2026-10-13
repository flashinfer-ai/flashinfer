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

"""JIT loader for the Cake SM120 MXFP4 x MXFP8 hybrid-transport MegaMoE closure.

The closure under ``../src/`` is manifest-driven: one ``cake_sm120_megamoe_manifest.json``
seals (SHA-256) the hand-written host binding, the ONE generated parametric device source
(``cake_sm120_megamoe_kernel.cuh`` and the part files it includes), four three-line
instance wrappers (M128 / M64 row schedule x with / without the LSA copier CTAs) and the
generated host geometry (``cake_sm120_megamoe_geometry.json``: switches, constants, buffer
extents, kernel ABI as expressions of the compile-time parameters).

Two kinds of JIT modules are built from it, both ordinary :class:`JitSpec` objects (so
``flashinfer/aot.py`` can package them):

* the host binding (transport session, probe, TMA encode, cooperative launch), one module;
* one device module per parameter set and transport flavor: ``nvcc -cubin`` compiles the
  M128 and M64 instances of the flavor for ``-DMOE_RANKS=... -DMOE_EXPERTS=... ...``
  (:data:`DEVICE_PARAMETERS`) and the two cubins are embedded into a small carrier
  (``embedded_cubin_factory``).  The binding loads them with the driver API because the
  kernel is a cooperative persistent kernel launched with ``cuLaunchCooperativeKernel``
  rather than a ``<<<>>>`` host stub.

Both link NCCL 2.30.7 (Device API: LSA + GIN), found like ``flashinfer/jit/comm.py``
finds NCCL: ``NCCL_HOME``, then the ``nvidia-nccl`` wheel.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping


from ......jit.core import JitSpec, gen_jit_spec, logger, sm120a_nvcc_flags

_SOURCE_DIR = Path(__file__).resolve().parents[1] / "src"
_MANIFEST = "cake_sm120_megamoe_manifest.json"
_MANIFEST_FORMAT = "cake-sm120-megamoe-manifest/3"
_GEOMETRY_FORMAT = "cake-sm120-megamoe-geometry/1"
# Hand-written carrier of the two embedded cubins of one device module (not generated).
_CUBIN_CARRIER = "cake_sm120_megamoe_cubins.cu"
# Index = the MOE_INSTANCE value compiled by the wrapper of that name.
INSTANCE_NAMES = ("m128_copier", "m64_copier", "m128_nocopier", "m64_nocopier")
# Compile-time parameters of the device source, in -D order (MOE_INSTANCE comes from the wrapper).
DEVICE_PARAMETERS = (
    "MOE_RANKS",
    "MOE_EXPERTS",
    "MOE_TOPK",
    "MOE_MAX_ROWS",
    "MOE_HIDDEN",
    "MOE_INTERMEDIATE",
    "MOE_SHARED_EXPERT",
    "MOE_SWIGLU_LIMIT",
    "MOE_CODEC_OVF_BLOCKS",
    "MOE_LSA_COPIERS",
)
_NVCC_FLAGS = (
    "-cubin",
    "--generate-code=arch=compute_120a,code=sm_120a",
    "-std=c++17",
    "-O3",
)
EXPECTED_NCCL_VERSION_INTEGER = 23007  # NCCL 2.30.7: GIN device API

# Device modules packaged by ``flashinfer/aot.py``: DeepSeek-V4-Flash routed experts
# (E256 / H4096 / I2048 / top-k 6) with the in-kernel FP8 shared expert and the SwiGLU
# clamp, 4096 tokens per rank, default ``all_gin`` transport (copier-free flavor), EP4 and
# EP8.  Other parameter sets compile on first use.
AOT_DEVICE_PARAMETERS: tuple[tuple[int, ...], ...] = (
    (4, 256, 6, 4096, 4096, 2048, 1, 10, 16, 1),
    (8, 256, 6, 4096, 4096, 2048, 1, 10, 16, 1),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nccl_home() -> Path | None:
    nccl_home = os.environ.get("NCCL_HOME")
    if nccl_home:
        return Path(nccl_home).expanduser()
    try:
        import nvidia.nccl  # type: ignore[import-not-found]

        return Path(nvidia.nccl.__path__[0])
    except (ImportError, AttributeError, IndexError):
        return None


def nccl_installation() -> tuple[Path, Path]:
    """Locate NCCL 2.30.7 headers (``nccl.h`` + ``nccl_device.h``) and ``libnccl.so.2``.

    Search order (as ``flashinfer/jit/comm.py``): ``NCCL_HOME``, then the
    ``nvidia-nccl-cu13`` wheel (``nvidia/nccl``).
    """

    root = _nccl_home()
    if root is not None:
        include = root / "include"
        header = include / "nccl.h"
        if header.is_file() and (include / "nccl_device.h").is_file():
            match = re.search(
                r"#define\s+NCCL_VERSION_CODE\s+(\d+)", header.read_text()
            )
            version = int(match.group(1)) if match else None
            if version != EXPECTED_NCCL_VERSION_INTEGER:
                raise RuntimeError(
                    f"Cake SM120 MegaMoE requires NCCL {EXPECTED_NCCL_VERSION_INTEGER} "
                    f"headers, {header} reports {version}"
                )
            for lib_dir in (root / "lib", root / "lib64"):
                library = lib_dir / "libnccl.so.2"
                if library.is_file():
                    return include.resolve(), library.resolve()
    raise RuntimeError(
        "NCCL 2.30.7 development files (nccl.h, nccl_device.h, libnccl.so.2) were not found; "
        "set NCCL_HOME or install nvidia-nccl-cu13==2.30.7"
    )


def cake_sm120_megamoe_available() -> bool:
    """True when the sealed sources and NCCL 2.30.7 (headers + library) are present."""

    try:
        read_manifest()
        nccl_installation()
    except (OSError, RuntimeError):
        return False
    return True


@functools.cache
def read_manifest() -> dict[str, Any]:
    """Load and SHA-256-verify the sealed closure manifest (binding, device sources, geometry)."""

    manifest = json.loads((_SOURCE_DIR / _MANIFEST).read_text())
    if manifest.get("format") != _MANIFEST_FORMAT:
        raise RuntimeError(
            f"unexpected Cake SM120 MegaMoE manifest format {manifest.get('format')!r}"
        )
    if manifest.get("arch") != "sm_120a":
        raise RuntimeError(f"unexpected device arch {manifest.get('arch')!r}")
    device = manifest["device"]
    sealed = [
        (manifest["binding"]["path"], manifest["binding"]["sha256"]),
        (manifest["geometry"]["path"], manifest["geometry"]["sha256"]),
        *device["sources"].items(),
        *((w["path"], w["sha256"]) for w in device["instances"]),
    ]
    for name, digest in sealed:
        path = _SOURCE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(f"Cake SM120 MegaMoE source not found: {path}")
        if _sha256(path) != digest:
            raise RuntimeError(f"Cake SM120 MegaMoE source digest mismatch: {path}")
    names = tuple(
        w["name"] for w in sorted(device["instances"], key=lambda w: w["instance"])
    )
    if names != INSTANCE_NAMES:
        raise RuntimeError(f"unexpected device instance set {names}")
    return manifest


@functools.cache
def read_geometry() -> dict[str, Any]:
    """The generated host geometry tables (verified by :func:`read_manifest`)."""

    manifest = read_manifest()
    geometry = json.loads((_SOURCE_DIR / manifest["geometry"]["path"]).read_text())
    if geometry.get("format") != _GEOMETRY_FORMAT:
        raise RuntimeError(
            f"unexpected Cake SM120 MegaMoE geometry format {geometry.get('format')!r}"
        )
    return geometry


def device_defines(parameters: tuple[int, ...]) -> list[str]:
    """``-D`` flags of one parameter set (ordered as :data:`DEVICE_PARAMETERS`)."""

    if len(parameters) != len(DEVICE_PARAMETERS):
        raise ValueError(f"expected {len(DEVICE_PARAMETERS)} device parameters")
    return [
        f"-D{n}={int(v)}" for n, v in zip(DEVICE_PARAMETERS, parameters, strict=True)
    ]


@functools.cache
def gen_cake_sm120_megamoe_module() -> JitSpec:
    """JIT spec for the host binding (transport session, probe, TMA encode, cooperative launch)."""

    manifest = read_manifest()
    nccl_include, nccl_library = nccl_installation()
    identity = manifest["binding"]["sha256"][:20]
    spec = gen_jit_spec(
        name=f"cake_sm120_megamoe_binding_{identity}",
        sources=[_SOURCE_DIR / manifest["binding"]["path"]],
        extra_cuda_cflags=[*sm120a_nvcc_flags],
        extra_ldflags=[
            "-lcuda",
            str(nccl_library),
            f"-Wl,-rpath,{nccl_library.parent}",
        ],
        extra_include_paths=[_SOURCE_DIR, nccl_include],
    )
    logger.info(f"Generated Cake SM120 MegaMoE binding JIT spec: {spec.name}")
    return spec


def _nvcc() -> str:
    from ......jit.cpp_ext import get_cuda_path

    nvcc = Path(get_cuda_path()) / "bin" / "nvcc"
    if nvcc.is_file():
        return str(nvcc)
    return os.environ.get("NVCC", "nvcc")


def _flavor_instances(copiers: bool) -> dict[int, dict[str, Any]]:
    """Manifest records of the flavor's instances, keyed by row-schedule height (128, 64)."""

    return {
        int(w["task_m"]): w
        for w in read_manifest()["device"]["instances"]
        if bool(w["copiers_active"]) == copiers
    }


def _compile_cubins(
    build_dir: Path, *, parameters: tuple[int, ...], copiers: bool
) -> Mapping[str, Path]:
    """``embedded_cubin_factory``: compile the flavor's M128 / M64 cubins into ``build_dir``.

    Runs under the module's build lock.  A cubin is rebuilt when it is missing or older
    than the sealed sources; the module's Ninja graph depends on the cubin paths.
    Unsupported parameter combinations fail in ``nvcc`` with the source's own ``#error`` /
    ``static_assert`` message (the backend checks the same guards first).
    """

    manifest = read_manifest()
    nccl_include, _ = nccl_installation()
    nvcc = _nvcc()
    defines = device_defines(parameters)
    newest_source = max(
        (_SOURCE_DIR / name).stat().st_mtime
        for name in (*manifest["device"]["sources"], _MANIFEST)
    )
    build_dir.mkdir(parents=True, exist_ok=True)

    def compile_one(task_m: int, wrapper: dict[str, Any]) -> tuple[str, Path]:
        cubin = build_dir / f"cake_sm120_megamoe_{wrapper['name']}.cubin"
        if not cubin.is_file() or cubin.stat().st_mtime < newest_source:
            temporary = build_dir / f".{cubin.name}.{os.getpid()}.tmp"
            command = [
                nvcc,
                *_NVCC_FLAGS,
                *defines,
                f"-I{_SOURCE_DIR}",
                f"-I{nccl_include}",
                str(_SOURCE_DIR / wrapper["path"]),
                "-o",
                str(temporary),
            ]
            logger.info(
                f"Compiling Cake SM120 MegaMoE {wrapper['name']} for {' '.join(defines)}"
            )
            completed = subprocess.run(
                command, capture_output=True, text=True, check=False
            )
            if completed.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    "nvcc failed for the Cake SM120 MegaMoE device source "
                    f"({wrapper['name']}, {' '.join(defines)}):\n"
                    + (completed.stderr or completed.stdout)
                )
            os.replace(temporary, cubin)
        return f"cake_sm120_megamoe_m{task_m}", cubin

    # The two instances are independent ~10-20 s nvcc runs; build them concurrently.
    with ThreadPoolExecutor(max_workers=2) as pool:
        built = list(
            pool.map(
                lambda item: compile_one(*item), _flavor_instances(copiers).items()
            )
        )
    return dict(built)


@functools.cache
def gen_cake_sm120_megamoe_kernel_module(
    parameters: tuple[int, ...], copiers: bool
) -> JitSpec:
    """JIT spec of the device instances of one parameter set and transport flavor.

    ``parameters`` follows :data:`DEVICE_PARAMETERS`; ``copiers`` selects the instances
    with LSA copier CTAs (``m128_copier`` / ``m64_copier``) or without
    (``m128_nocopier`` / ``m64_nocopier``).  The module exports ``cubin_image(task_m)``.
    """

    manifest = read_manifest()
    defines = device_defines(parameters)
    identity = hashlib.sha256(
        json.dumps(
            {
                "sources": manifest["device"]["sources"],
                "wrappers": sorted(
                    w["sha256"] for w in _flavor_instances(copiers).values()
                ),
                "defines": defines,
                "flags": _NVCC_FLAGS,
                "nccl": EXPECTED_NCCL_VERSION_INTEGER,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()
    flavor = "copier" if copiers else "nocopier"
    return gen_jit_spec(
        name=f"cake_sm120_megamoe_{flavor}_{identity[:20]}",
        sources=[_SOURCE_DIR / _CUBIN_CARRIER],
        extra_cuda_cflags=[*sm120a_nvcc_flags],
        extra_ldflags=["-lcuda"],
        embedded_cubin_factory=functools.partial(
            _compile_cubins, parameters=tuple(parameters), copiers=copiers
        ),
    )


def gen_cake_sm120_megamoe_aot_modules() -> list[JitSpec]:
    """The binding and the :data:`AOT_DEVICE_PARAMETERS` device modules (all-GIN flavor)."""

    return [
        gen_cake_sm120_megamoe_module(),
        *(
            gen_cake_sm120_megamoe_kernel_module(parameters, False)
            for parameters in AOT_DEVICE_PARAMETERS
        ),
    ]


@functools.cache
def device_cubin(parameters: tuple[int, ...], instance: int) -> bytes:
    """The cubin of one device instance (``instance`` indexes :data:`INSTANCE_NAMES`).

    Loaded from the packaged AOT module or the JIT cache, compiled on first use otherwise.
    Only the instances a deployment launches are built (the two row schedules of the
    copier or copier-free flavor its transport table needs).
    """

    wrapper = next(
        w for w in read_manifest()["device"]["instances"] if w["instance"] == instance
    )
    module = gen_cake_sm120_megamoe_kernel_module(
        tuple(parameters), bool(wrapper["copiers_active"])
    ).build_and_load()
    return bytes(module.cubin_image(int(wrapper["task_m"])))


def device_cubin_resource_usage(
    parameters: tuple[int, ...], instance: int
) -> dict[str, int]:
    """Per-thread resources of one compiled device instance (no GPU needed).

    Uses :func:`device_cubin` (same modules) and reads
    ``cuobjdump -res-usage``: ``reg`` registers, ``stack`` stack frame bytes, ``local``
    local memory bytes.  A register spill shows up as stack / local growth.
    """

    import tempfile

    cubin = device_cubin(parameters, instance)
    cuobjdump = Path(_nvcc()).with_name("cuobjdump")
    tool = str(cuobjdump) if cuobjdump.is_file() else "cuobjdump"
    with tempfile.NamedTemporaryFile(suffix=".cubin") as handle:
        handle.write(cubin)
        handle.flush()
        completed = subprocess.run(
            [tool, "-res-usage", handle.name],
            capture_output=True,
            text=True,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cuobjdump -res-usage failed for {INSTANCE_NAMES[instance]}: "
            + (completed.stderr or completed.stdout)
        )
    usage: dict[str, int] = {}
    for key in ("REG", "STACK", "LOCAL"):
        match = re.search(rf"\b{key}:(\d+)", completed.stdout)
        if match is None:
            raise RuntimeError(
                f"cuobjdump -res-usage reported no {key} for {INSTANCE_NAMES[instance]}"
            )
        usage[key.lower()] = int(match.group(1))
    return usage


def check_exact_sm120(device: Any = None) -> None:
    import torch

    from ......utils import get_compute_capability

    resolved = torch.device("cuda") if device is None else torch.device(device)
    capability = get_compute_capability(resolved)
    # The cubins are built for ``sm_120a``; arch-specific code does not load on 12.1.
    if capability != (12, 0):
        raise RuntimeError(
            f"Cake SM120 MegaMoE requires compute capability 12.0, got {capability[0]}.{capability[1]}"
        )


@functools.cache
def _build_and_load() -> Any:
    module = gen_cake_sm120_megamoe_module().build_and_load()
    observed = int(module.nccl_runtime_version())
    if observed != EXPECTED_NCCL_VERSION_INTEGER:
        raise RuntimeError(
            f"Cake SM120 MegaMoE requires NCCL {EXPECTED_NCCL_VERSION_INTEGER}, the loaded library reports {observed}"
        )
    logger.info("Loaded Cake SM120 MegaMoE binding module")
    return module


def load_cake_sm120_megamoe_module(*, device: Any = None) -> Any:
    """Build or load the binding for an exact SM120 device (fail closed)."""

    check_exact_sm120(device)
    return _build_and_load()


__all__ = [
    "AOT_DEVICE_PARAMETERS",
    "DEVICE_PARAMETERS",
    "EXPECTED_NCCL_VERSION_INTEGER",
    "INSTANCE_NAMES",
    "cake_sm120_megamoe_available",
    "check_exact_sm120",
    "device_cubin",
    "device_cubin_resource_usage",
    "device_defines",
    "gen_cake_sm120_megamoe_aot_modules",
    "gen_cake_sm120_megamoe_kernel_module",
    "gen_cake_sm120_megamoe_module",
    "load_cake_sm120_megamoe_module",
    "nccl_installation",
    "read_geometry",
    "read_manifest",
]
