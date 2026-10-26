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
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
)

# Generated SM120 DeepSeek-V4 NVFP4 cache writers: fused GPT-J RoPE + group-16 NVFP4
# quantization + paged insert into the 384-byte record read by the Cake sparse-MLA
# decode / prefill family (``cake_sparse_mla_dsv4_nvfp4``). Own JIT module: the attention
# module is never rebuilt when the writers are regenerated.
_FAMILY = "cake_dsv4_nvfp4_rope_insert"
_SOURCE_SUBDIR = Path("cake_dsv4") / "sm_120a"
_MANIFEST_NAME = f"{_FAMILY}_manifest.json"
# GB202 / GB10 only: the generated device code is emitted for the SM120 instruction surface
# (register-only scalar fp32 math, warp shuffles, ``cvt.rn.satfinite.e4m3x2.f32`` /
# ``cvt.rn.satfinite.e2m1x2.f32``) and pairs with the SM120-only attention family.
_SUPPORTED_MAJOR_VERSIONS = [12]
# Identity carried by the hand-written placeholder manifest until the exporter writes the family.
_UNEXPORTED_IDENTITY = "unexported"

_REQUIRED_MANIFEST_KEYS = (
    "identity",
    "kernel_commit",
    "generator",
    "generated_at",
    "sources",
    "entries",
    "q_head_padded_choices",
    "compress_ratios",
    "slot_dtypes",
    "bytes_per_token",
    "data_bytes_per_token",
    "scale_bytes_per_token",
    "head_dim",
    "rope_dim",
    "threads",
)
_REQUIRED_ENTRIES = ("qkv", "kv")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _candidate_dirs() -> tuple[Path, Path]:
    return (
        jit_env.FLASHINFER_CSRC_DIR / _SOURCE_SUBDIR,
        _repo_root() / "csrc" / _SOURCE_SUBDIR,
    )


def _source_dir() -> Optional[Path]:
    for candidate in _candidate_dirs():
        if (candidate / _MANIFEST_NAME).is_file():
            return candidate
    return None


@functools.cache
def cake_dsv4_nvfp4_rope_insert_manifest() -> Dict[str, Any]:
    """The generated family manifest: provenance, record constants, dispatch choices and the source list.

    Raises ``FileNotFoundError`` naming both searched locations when the manifest is absent and
    ``ValueError`` when it lacks a required key or FFI entry name.
    """

    source_dir = _source_dir()
    if source_dir is None:
        packaged, source_tree = _candidate_dirs()
        raise FileNotFoundError(
            "Cake SM120 DSV4 NVFP4 RoPE-insert sources not found (the generated family "
            f"manifest {_MANIFEST_NAME} is absent). Checked:\n"
            f"  - {packaged}\n  - {source_tree}"
        )
    with open(source_dir / _MANIFEST_NAME, encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(
            f"Cake SM120 DSV4 NVFP4 RoPE-insert manifest lacks {missing!r}"
        )
    missing_entries = [
        name for name in _REQUIRED_ENTRIES if name not in manifest["entries"]
    ]
    if missing_entries:
        raise ValueError(
            "Cake SM120 DSV4 NVFP4 RoPE-insert manifest lacks the FFI entries "
            f"{missing_entries!r}"
        )
    return manifest


def _missing_sources(source_dir: Path, manifest: Dict[str, Any]) -> List[Path]:
    return [
        source_dir / name
        for name in manifest["sources"]
        if not (source_dir / name).is_file()
    ]


def cake_dsv4_nvfp4_rope_insert_available() -> bool:
    """True when the generated kernel family is present: a non-placeholder manifest and every listed source."""

    source_dir = _source_dir()
    if source_dir is None:
        return False
    try:
        manifest = cake_dsv4_nvfp4_rope_insert_manifest()
    except ValueError:
        return False
    if manifest["identity"] == _UNEXPORTED_IDENTITY:
        return False
    return not _missing_sources(source_dir, manifest)


def gen_cake_dsv4_nvfp4_rope_insert_module() -> JitSpec:
    """JIT spec for the Cake SM120 (GB202 / GB10) DeepSeek-V4 NVFP4 fused RoPE + quantize + insert writers.

    One generated kernel translation unit (every exported variant: padded Q head count x Q RoPE
    on/off x slot dtype for the QKV form, compress ratio x slot dtype for the KV form) plus the
    TVM-FFI host binding with the two entries named by the manifest; the module name carries the
    generated family's identity so a regenerated kernel set never reuses a stale build. The
    include path lets the binding reach the SM120 NVFP4 cache-layout parser
    (``sparse_mla_sm120/dsv4_nvfp4_validation.h``) next to the shared ``tvm_ffi_utils.h``.

    Raises ``FileNotFoundError`` while the tree only holds the placeholder manifest.
    """

    manifest = cake_dsv4_nvfp4_rope_insert_manifest()
    source_dir = _source_dir()
    assert source_dir is not None
    missing = _missing_sources(source_dir, manifest)
    if manifest["identity"] == _UNEXPORTED_IDENTITY or missing:
        raise FileNotFoundError(
            "Cake SM120 DSV4 NVFP4 RoPE-insert kernels are not exported into this tree "
            f"(manifest identity {manifest['identity']!r}; missing sources: "
            f"{[str(path) for path in missing]}). Regenerate csrc/{_SOURCE_SUBDIR} with the "
            "exporter named in the manifest."
        )
    compilation_context = current_compilation_context
    if not compilation_context.TARGET_CUDA_ARCHS:
        compilation_context = refresh_current_compilation_context()
    nvcc_flags = compilation_context.get_nvcc_flags_list(
        supported_major_versions=_SUPPORTED_MAJOR_VERSIONS
    )
    return gen_jit_spec(
        f"{_FAMILY}_sm120a_{manifest['identity'][:12]}",
        [source_dir / name for name in manifest["sources"]],
        extra_cuda_cflags=nvcc_flags,
        extra_include_paths=[
            source_dir,
            jit_env.FLASHINFER_CSRC_DIR,
            jit_env.FLASHINFER_CSRC_DIR / "sparse_mla_sm120",
        ],
    )


__all__ = [
    "cake_dsv4_nvfp4_rope_insert_available",
    "cake_dsv4_nvfp4_rope_insert_manifest",
    "gen_cake_dsv4_nvfp4_rope_insert_module",
]
