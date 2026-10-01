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

from __future__ import annotations

import functools
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Optional

from ..compilation_context import CompilationContext
from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger
from .utils import write_if_different

_GENERATED_DIR = "generated"
# The root translation unit; its kernel bodies live in the ``source_files`` parts the manifest
# lists (``cake_sampling_kernels_part<N>.cuh``, each under the repository's 5 MiB file limit),
# which the root includes in order.
_SOURCE_FILE = f"{_GENERATED_DIR}/cake_sampling_kernels.cu"
_MAX_SOURCE_FILE_BYTES = 5 * 1024 * 1024
_MANIFEST_FILE = f"{_GENERATED_DIR}/manifest.json"
_BINDING_HEADER = "cake_sampling_binding.cuh"
# The kernels use thread-block clusters, distributed shared memory, programmatic dependent launch
# and redux.sync only, i.e. the sm_90 feature set, so one frozen source serves every compute
# capability 9.x / 10.x / 11.x / 12.x device.  It is compiled once into a single fatbin with one
# -gencode per target architecture; the targets come from FlashInfer's ``CompilationContext``:
# ``FLASHINFER_CUDA_ARCH_LIST`` when set (AOT builds on hosts without a GPU), otherwise the
# capabilities of the visible devices.
SUPPORTED_MAJOR_VERSIONS: tuple[int, ...] = (9, 10, 11, 12)
_MANIFEST_KEYS = {
    "buckets",
    "codegen_arch",
    "compile_flags",
    "fused_block_tail_kcap",
    "fused_tail_kcap",
    "kernel_count",
    "kernel_symbols",
    "min_compute_capability",
    "schema_version",
    "semantics",
    "slab_entries",
    "source_files",
    "source_sha256",
    "stage1",
    "stage23",
    "tma_abi",
}
_LOADED: dict[str, Any] = {}


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_sampling"
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "cake_sampling"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "frozen radix sampling sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry (``(9, "0a")`` -> ``(9, 0)``)."""
    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Compute capabilities the module is built for, in ascending order.

    Read once per process from ``CompilationContext`` (``FLASHINFER_CUDA_ARCH_LIST`` when set,
    else the visible devices) and restricted to :data:`SUPPORTED_MAJOR_VERSIONS`.  Empty when the
    process has neither an arch list nor a CUDA device.
    """
    context = CompilationContext()
    return tuple(
        sorted(
            {
                _capability_of(major, minor)
                for major, minor in context.TARGET_CUDA_ARCHS
                if int(major) in SUPPORTED_MAJOR_VERSIONS
            }
        )
    )


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the frozen kernels are built for it, else ``None``.

    A device whose capability is not among the build targets (a major outside
    :data:`SUPPORTED_MAJOR_VERSIONS`, or an architecture left out of ``FLASHINFER_CUDA_ARCH_LIST``)
    takes the ``top_k_first`` fallback instead of failing at launch.
    """
    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


def supported_capabilities() -> tuple[tuple[int, int], ...]:
    """Capabilities the frozen kernels are built for in this process."""
    return target_capabilities()


def nvcc_flags() -> list[str]:
    """``-gencode`` flags for every build target (plus FlashInfer's common flags).

    Raises ``RuntimeError`` when no target has a supported major version.
    """
    return CompilationContext().get_nvcc_flags_list(
        supported_major_versions=list(SUPPORTED_MAJOR_VERSIONS)
    )


def _reject_duplicate_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(f"radix sampling manifest has duplicate key {key!r}")
        document[key] = value
    return document


@functools.cache
def load_manifest() -> dict[str, Any]:
    """Load and verify the frozen manifest (``csrc/cake_sampling/generated/manifest.json``)."""
    csrc = _get_csrc_dir()
    source = csrc / _SOURCE_FILE
    manifest_path = csrc / _MANIFEST_FILE
    binding = csrc / _BINDING_HEADER
    missing = [p.name for p in (source, manifest_path, binding) if not p.is_file()]
    if missing:
        raise RuntimeError(
            f"radix sampling source package is incomplete: missing {', '.join(missing)}"
        )
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("radix sampling manifest is invalid JSON") from error
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise RuntimeError("radix sampling manifest schema is invalid")
    if manifest["schema_version"] != 1 or manifest["tma_abi"] != "pointer":
        raise RuntimeError("radix sampling manifest identity is invalid")
    min_cc = manifest["min_compute_capability"]
    if (
        not isinstance(min_cc, list)
        or len(min_cc) != 2
        or not all(isinstance(v, int) for v in min_cc)
        or tuple(min_cc) != (min(SUPPORTED_MAJOR_VERSIONS), 0)
    ):
        raise RuntimeError(
            "radix sampling manifest min_compute_capability does not cover the compiled targets"
        )
    if not re.fullmatch(r"sm_[0-9]+a", str(manifest["codegen_arch"])):
        raise RuntimeError("radix sampling manifest codegen_arch is invalid")
    source_bytes = b"".join(_verified_source_files(csrc, manifest).values())
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise RuntimeError("radix sampling source identity is invalid")
    symbols = list(manifest["kernel_symbols"])
    stage_symbols = [v["symbol"] for v in manifest["stage1"]] + [
        v["symbol"] for v in manifest["stage23"]
    ]
    if symbols != stage_symbols or len(symbols) != manifest["kernel_count"]:
        raise RuntimeError("radix sampling manifest kernel inventory is inconsistent")
    if any(not isinstance(v.get("fused_tail"), bool) for v in manifest["stage1"]):
        raise RuntimeError("radix sampling manifest stage-1 entries lack fused_tail")
    if any(not isinstance(v.get("fused_block_tail"), bool) for v in manifest["stage1"]):
        raise RuntimeError(
            "radix sampling manifest stage-1 entries lack fused_block_tail"
        )
    if any(v["fused_block_tail"] and not v["fused_tail"] for v in manifest["stage1"]):
        raise RuntimeError(
            "radix sampling manifest: a whole-CTA tail without the two-warp tail"
        )
    if any(not isinstance(v.get("coarse_sample"), bool) for v in manifest["stage1"]):
        raise RuntimeError("radix sampling manifest stage-1 entries lack coarse_sample")
    if any(v["coarse_sample"] and not v["stream"] for v in manifest["stage1"]):
        raise RuntimeError(
            "radix sampling manifest: a coarse-sample build of a register-resident variant"
        )
    if any(v["coarse_sample"] and v["fused_block_tail"] for v in manifest["stage1"]):
        raise RuntimeError(
            "radix sampling manifest: a build with both the coarse sample and the whole-CTA tail"
        )
    if any(not isinstance(v.get("spec_sample"), bool) for v in manifest["stage1"]):
        raise RuntimeError("radix sampling manifest stage-1 entries lack spec_sample")
    if any(v["spec_sample"] and not v["stream"] for v in manifest["stage1"]):
        raise RuntimeError(
            "radix sampling manifest: a speculative-sample build of a register-resident variant"
        )
    if any(
        v["spec_sample"] and (v["fused_block_tail"] or v["coarse_sample"])
        for v in manifest["stage1"]
    ):
        raise RuntimeError(
            "radix sampling manifest: a speculative-sample build together with the coarse sample "
            "or the whole-CTA tail"
        )
    builds = [
        (
            v["cluster"],
            v["ept"],
            bool(v["stream"]),
            bool(v["fused_block_tail"]),
            bool(v["coarse_sample"]),
            bool(v["spec_sample"]),
        )
        for v in manifest["stage1"]
    ]
    if len(set(builds)) != len(builds):
        raise RuntimeError("radix sampling manifest stage-1 builds are not unique")
    if any(
        (bt or ws or sp) and (c, e, s, False, False, False) not in set(builds)
        for c, e, s, bt, ws, sp in builds
    ):
        raise RuntimeError(
            "radix sampling manifest: a twin build without its default build"
        )
    for v in manifest["stage23"]:
        feats = v.get("features")
        flags = v.get("variant_flags")
        if not isinstance(feats, list) or not all(isinstance(f, str) for f in feats):
            raise RuntimeError(
                "radix sampling manifest stage-2/3 entries lack features"
            )
        if not isinstance(flags, int) or isinstance(flags, bool) or flags < 0:
            raise RuntimeError(
                "radix sampling manifest stage-2/3 entries lack variant_flags"
            )
    if len(
        {(v["threads"], v["items"], v["variant_flags"]) for v in manifest["stage23"]}
    ) != len(manifest["stage23"]):
        raise RuntimeError("radix sampling manifest stage-2/3 variants are not unique")
    for symbol in symbols:
        definitions = re.findall(
            rb"(?<![A-Za-z0-9_])" + re.escape(symbol.encode()) + rb"\(", source_bytes
        )
        if len(definitions) != 1:
            raise RuntimeError(
                f"radix sampling source does not define {symbol} exactly once"
            )
    return manifest


def _verified_source_files(csrc: Path, manifest: dict[str, Any]) -> dict[str, bytes]:
    """The frozen source files (root first, then its include parts) with their bytes, each verified.

    Every entry of ``manifest["source_files"]`` must be a regular file directly inside the generated
    directory whose size and SHA-256 match; the first entry must be the root translation unit, and
    the root must include exactly the remaining parts in order.
    """
    entries = manifest["source_files"]
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("radix sampling manifest source_files is invalid")
    files: dict[str, bytes] = {}
    for entry in entries:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256", "bytes"}
            or not isinstance(entry["path"], str)
            or not re.fullmatch(r"[A-Za-z0-9_.-]+\.(?:cu|cuh)", entry["path"])
            or entry["path"] in files
        ):
            raise RuntimeError("radix sampling manifest source_files entry is invalid")
        path = csrc / _GENERATED_DIR / entry["path"]
        if not path.is_file():
            raise RuntimeError(
                f"radix sampling source package is incomplete: missing {entry['path']}"
            )
        data = path.read_bytes()
        if (
            len(data) != entry["bytes"]
            or hashlib.sha256(data).hexdigest() != entry["sha256"]
        ):
            raise RuntimeError(
                f"radix sampling source identity is invalid: {entry['path']}"
            )
        if len(data) > _MAX_SOURCE_FILE_BYTES:
            raise RuntimeError(
                f"radix sampling source file exceeds the size limit: {entry['path']}"
            )
        files[entry["path"]] = data
    root = Path(_SOURCE_FILE).name
    if next(iter(files)) != root:
        raise RuntimeError(
            "radix sampling manifest source_files must start with the root source"
        )
    included = re.findall(
        rb'^#include "([^"\n]+)"\s*$', files[root], flags=re.MULTILINE
    )
    if [name.decode() for name in included] != list(files)[1:]:
        raise RuntimeError(
            "radix sampling root source does not include exactly the listed parts"
        )
    return files


def _binding_source(manifest: dict[str, Any]) -> str:
    min_major, min_minor = manifest["min_compute_capability"]
    stage1 = " ".join(
        f"X({v['symbol']}, {v['cluster']}, {v['ept']}, {1 if v['stream'] else 0}, "
        f"{v['block_threads']}, {v['dynamic_smem_bytes']}, {1 if v['fused_tail'] else 0}, "
        f"{1 if v['fused_block_tail'] else 0}, {1 if v['coarse_sample'] else 0}, "
        f"{1 if v['spec_sample'] else 0})"
        for v in manifest["stage1"]
    )
    stage23 = " ".join(
        f"X({v['symbol']}, {v['threads']}, {v['items']}, {int(v['variant_flags'])}, "
        f"{v['dynamic_smem_bytes']})"
        for v in manifest["stage23"]
    )
    return f"""\
/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */
#define CAKE_SAMPLING_BODY_FILE "{_SOURCE_FILE}"
#define CAKE_SAMPLING_MIN_MAJOR {min_major}
#define CAKE_SAMPLING_MIN_MINOR {min_minor}
#define CAKE_SAMPLING_SLAB {manifest["slab_entries"]}
#define CAKE_SAMPLING_FUSED_TAIL_KCAP {manifest["fused_tail_kcap"]}
#define CAKE_SAMPLING_FUSED_BLOCK_TAIL_KCAP {manifest["fused_block_tail_kcap"]}
#define CAKE_SAMPLING_STAGE1_TABLE(X) {stage1}
#define CAKE_SAMPLING_STAGE23_TABLE(X) {stage23}
#include "{_BINDING_HEADER}"
"""


def _module_identity(manifest: dict[str, Any]) -> str:
    """JIT module name: sealed over the frozen source, manifest and binding.

    The target architectures are not part of the name: FlashInfer's JIT workspace directory is
    already keyed by the ``CompilationContext`` target set, so one name maps to one fatbin per
    target set.
    """
    csrc = _get_csrc_dir()
    digest = hashlib.sha256()
    for data in _verified_source_files(csrc, manifest).values():
        digest.update(data)
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update((csrc / _BINDING_HEADER).read_bytes())
    digest.update(_binding_source(manifest).encode())
    return f"cake_sampling_{digest.hexdigest()[:20]}"


def get_cake_sampling_uri() -> str:
    return _module_identity(load_manifest())


@functools.cache
def gen_cake_sampling_module() -> JitSpec:
    """One JIT spec compiling the frozen source for every ``CompilationContext`` target."""
    manifest = load_manifest()
    csrc = _get_csrc_dir()
    uri = _module_identity(manifest)
    binding = jit_env.FLASHINFER_GEN_SRC_DIR / uri / "cake_sampling_binding.cu"
    write_if_different(binding, _binding_source(manifest))
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=nvcc_flags() + list(manifest["compile_flags"]),
        extra_include_paths=[csrc, csrc.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info(
        "Generated frozen radix sampling JIT spec %s for compute capabilities %s",
        spec.name,
        ", ".join(f"{a}.{b}" for a, b in supported_capabilities()),
    )
    return spec


def load_cake_sampling_module():
    module = _LOADED.get("module")
    if module is None:
        module = gen_cake_sampling_module().build_and_load()
        _LOADED["module"] = module
        logger.info("Loaded frozen radix sampling module")
    return module


def is_cake_sampling_module_loaded() -> bool:
    return "module" in _LOADED


__all__ = [
    "SUPPORTED_MAJOR_VERSIONS",
    "gen_cake_sampling_module",
    "get_cake_sampling_uri",
    "is_cake_sampling_module_loaded",
    "load_cake_sampling_module",
    "load_manifest",
    "nvcc_flags",
    "supported_capabilities",
    "supported_capability",
    "target_capabilities",
]
