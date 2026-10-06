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
from typing import Any, Dict, Optional

from . import env as jit_env
from .core import (
    JitSpec,
    current_compilation_context,
    gen_jit_spec,
    refresh_current_compilation_context,
)

# Family name of the generated DeepSeek-V4.1 mixed-cache (528 B FP8 main + 288 B V41_FP4 extra, BF16 Q)
# sparse-MLA decode kernels. Distinct from ``cake_sparse_mla_dsv4_nvfp4`` (the 384 B NVFP4 family) so the
# two generated kernel sets never share a translation unit, header, manifest or JIT module name.
_FAMILY = "cake_sparse_mla_dsv41_mixed"
_SOURCE_SUBDIR = Path("cake_dsv4") / "sm_120a"
_MANIFEST_NAME = f"{_FAMILY}_manifest.json"
# GB202 / GB10 only: the generated device code uses the SM120 mma.sync tensor-core surface, cp.async 16-byte
# gathers with zero-fill and CTA mbarriers, and its register / shared-memory schedule is tuned for SM120.
_SUPPORTED_MAJOR_VERSIONS = [12]

_REQUIRED_MANIFEST_KEYS = (
    "identity",
    "entry",
    "sources",
    "head_counts",
    "heads_per_block",
    "candidates_per_chunk",
    "max_chunks_per_block",
    "head_dim",
    "value_dim",
    "main_bytes_per_token",
    "extra_bytes_per_token",
    "precisions",
    "decode_params",
    "merge_params",
)


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


def cake_sparse_mla_sm120_dsv41_mixed_available() -> bool:
    """True when the generated kernel family (its manifest) is present in this tree."""

    return _source_dir() is not None


@functools.cache
def cake_sparse_mla_sm120_dsv41_mixed_manifest() -> Dict[str, Any]:
    """The generated family manifest: provenance, geometry constants and the source list.

    Raises ``FileNotFoundError`` naming both searched locations when the family has not been
    exported into this tree.
    """

    source_dir = _source_dir()
    if source_dir is None:
        packaged, source_tree = _candidate_dirs()
        raise FileNotFoundError(
            "Cake SM120 DSv4.1 mixed-cache sparse-MLA sources not found (the generated "
            f"family manifest {_MANIFEST_NAME} is absent). Checked:\n"
            f"  - {packaged}\n  - {source_tree}"
        )
    with open(source_dir / _MANIFEST_NAME, encoding="utf-8") as handle:
        manifest = json.load(handle)
    missing = [key for key in _REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Cake SM120 DSv4.1 mixed-cache manifest lacks {missing!r}")
    return manifest


def gen_cake_sparse_mla_sm120_dsv41_mixed_module() -> JitSpec:
    """JIT spec for the Cake SM120 (GB202 / GB10) DeepSeek-V4.1 mixed-cache sparse-MLA decode family.

    One translation unit per supported head count (single-cache decode, dual-cache decode and
    split merge kernels for every exported compute precision) plus the TVM-FFI host binding; the
    module name carries the generated family's identity so a regenerated kernel set never reuses
    a stale build.
    """

    manifest = cake_sparse_mla_sm120_dsv41_mixed_manifest()
    source_dir = _source_dir()
    assert source_dir is not None
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
    "cake_sparse_mla_sm120_dsv41_mixed_available",
    "cake_sparse_mla_sm120_dsv41_mixed_manifest",
    "gen_cake_sparse_mla_sm120_dsv41_mixed_module",
]
