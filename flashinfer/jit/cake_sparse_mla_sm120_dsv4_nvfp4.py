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
from typing import Any, Dict

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
)

_FAMILY = "cake_sparse_mla_dsv4_nvfp4"
_SOURCE_SUBDIR = Path("cake_dsv4") / "sm_120a"
_MANIFEST_NAME = f"{_FAMILY}_manifest.json"
# GB202 only: the generated device code uses mma.sync kind::mxf4nvf4 block-scaled MMA, cp.async 16-byte
# gathers with zero-fill and CTA mbarriers, and its register / shared-memory schedule is tuned for SM120.
_SUPPORTED_MAJOR_VERSIONS = [12]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / _SOURCE_SUBDIR
    if (packaged / _MANIFEST_NAME).is_file():
        return packaged
    source_tree = _repo_root() / "csrc" / _SOURCE_SUBDIR
    if (source_tree / _MANIFEST_NAME).is_file():
        return source_tree
    raise FileNotFoundError(
        "Cake SM120 DSV4 NVFP4 sparse-MLA sources not found. Checked:\n"
        f"  - {packaged}\n  - {source_tree}"
    )


@functools.cache
def cake_sparse_mla_sm120_dsv4_nvfp4_manifest() -> Dict[str, Any]:
    """The generated family manifest: provenance, geometry constants and the source list."""

    with open(_source_dir() / _MANIFEST_NAME, encoding="utf-8") as handle:
        manifest = json.load(handle)
    for key in (
        "identity",
        "entry",
        "prefill_entry",
        "sources",
        "head_counts",
        "prefill_head_counts",
        "prefill_head_tiles",
        "heads_per_block",
        "candidates_per_chunk",
        "max_chunks_per_block",
        "head_dim",
        "value_dim",
        "bytes_per_token",
    ):
        if key not in manifest:
            raise ValueError(f"Cake SM120 DSV4 NVFP4 manifest lacks {key!r}")
    return manifest


def gen_cake_sparse_mla_sm120_dsv4_nvfp4_module() -> JitSpec:
    """JIT spec for the Cake SM120 (GB202) DeepSeek-V4 NVFP4 sparse-MLA decode + prefill families.

    One translation unit per supported head count for the decode family (single-cache decode,
    dual-cache decode and split merge kernels), one per prefill head count (head tiles x single /
    dual cache x one-item / persistent CTAs) plus the TVM-FFI host binding with both entries; the
    module name carries the generated family's identity so a regenerated kernel set never reuses a
    stale build.
    """

    manifest = cake_sparse_mla_sm120_dsv4_nvfp4_manifest()
    source_dir = _source_dir()
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
        extra_include_paths=[source_dir],
    )


__all__ = [
    "cake_sparse_mla_sm120_dsv4_nvfp4_manifest",
    "gen_cake_sparse_mla_sm120_dsv4_nvfp4_module",
]
