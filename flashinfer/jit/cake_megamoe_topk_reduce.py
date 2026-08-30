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
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags
from .utils import write_if_different

_SOURCE_FILE = "cake_megamoe_topk_reduce_kernels.cu"
_MANIFEST_FILE = "manifest.json"
_BINDING_HEADER = "cake_megamoe_topk_reduce_binding.cuh"
_KERNEL_SYMBOL = "kernel_cake_megamoe_workspace_topk_reduce_bfloat16_h4096_k6"
_MANIFEST_KEYS = {
    "arch",
    "compile_flags",
    "constraints",
    "kernel_count",
    "kernel_symbols",
    "launch",
    "schema_version",
    "source_sha256",
    "tma_abi",
}
_CONSTRAINTS = {
    "capacities": [256, 4096],
    "dtype": "bfloat16",
    "hidden_size": 4096,
    "top_k": 6,
}
_LAUNCH = {
    "block_threads": 128,
    "dynamic_smem_bytes": 0,
    "grid_x": "4 * num_tokens",
}
_LOADED_MODULE: Any | None = None


def _get_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "cake_megamoe_topk_reduce"
    if installed.is_dir():
        return installed
    checkout = (
        Path(__file__).resolve().parents[2]
        / "csrc"
        / "cake_megamoe_topk_reduce"
    )
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "frozen MegaMoE TopK-reduce sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def _reject_duplicate_manifest_keys(pairs):
    document = {}
    for key, value in pairs:
        if key in document:
            raise RuntimeError(
                f"MegaMoE TopK-reduce manifest has duplicate key {key!r}"
            )
        document[key] = value
    return document


def _program_source() -> tuple[Path, dict[str, Any]]:
    csrc_dir = _get_csrc_dir()
    source = csrc_dir / _SOURCE_FILE
    manifest_path = csrc_dir / _MANIFEST_FILE
    binding_header = csrc_dir / _BINDING_HEADER
    missing = [
        path.name
        for path in (source, manifest_path, binding_header)
        if not path.is_file()
    ]
    if missing:
        raise RuntimeError(
            "MegaMoE TopK-reduce source package is incomplete: missing "
            + ", ".join(missing)
        )

    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_manifest_keys,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("MegaMoE TopK-reduce manifest is invalid JSON") from error

    source_bytes = source.read_bytes()
    expected = {
        "schema_version": 1,
        "arch": "sm_100a",
        "compile_flags": [],
        "tma_abi": "pointer",
        "kernel_count": 1,
        "launch": _LAUNCH,
        "constraints": _CONSTRAINTS,
        "kernel_symbols": [_KERNEL_SYMBOL],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_KEYS
        or manifest != expected
    ):
        raise RuntimeError("MegaMoE TopK-reduce manifest identity is invalid")
    if source_bytes.count(_KERNEL_SYMBOL.encode()) != 1:
        raise RuntimeError(
            "MegaMoE TopK-reduce source does not define exactly one frozen kernel symbol"
        )
    return source, manifest


def _module_identity(
    source: Path, manifest: dict[str, Any], binding_header: Path
) -> str:
    digest = hashlib.sha256()
    digest.update(source.read_bytes())
    digest.update(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    digest.update(binding_header.read_bytes())
    digest.update(_binding_source().encode())
    digest.update(b"sm_100a")
    return f"cake_megamoe_topk_reduce_sm100a_{digest.hexdigest()[:20]}"


def _binding_source() -> str:
    return f"""\
/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#define CAKE_MEGAMOE_TOPK_REDUCE_BODY_FILE "{_SOURCE_FILE}"
#define CAKE_MEGAMOE_TOPK_REDUCE_KERNEL {_KERNEL_SYMBOL}
#define CAKE_MEGAMOE_TOPK_REDUCE_THREADS {_LAUNCH["block_threads"]}
#define CAKE_MEGAMOE_TOPK_REDUCE_SMEM_BYTES {_LAUNCH["dynamic_smem_bytes"]}

#include "{_BINDING_HEADER}"
"""


def get_cake_megamoe_topk_reduce_uri() -> str:
    source, manifest = _program_source()
    return _module_identity(source, manifest, _get_csrc_dir() / _BINDING_HEADER)


@functools.cache
def gen_cake_megamoe_topk_reduce_module() -> JitSpec:
    source, manifest = _program_source()
    csrc_dir = source.parent
    uri = _module_identity(source, manifest, csrc_dir / _BINDING_HEADER)
    binding = (
        jit_env.FLASHINFER_GEN_SRC_DIR
        / uri
        / "cake_megamoe_topk_reduce_binding.cu"
    )
    write_if_different(binding, _binding_source())
    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=sm100a_nvcc_flags,
        extra_include_paths=[csrc_dir, csrc_dir.parent, _get_include_dir()],
        use_fast_math=False,
    )
    logger.info("Generated frozen MegaMoE TopK-reduce JIT spec: %s", spec.name)
    return spec


def load_cake_megamoe_topk_reduce_module():
    global _LOADED_MODULE
    if _LOADED_MODULE is None:
        _LOADED_MODULE = gen_cake_megamoe_topk_reduce_module().build_and_load()
        logger.info("Loaded frozen MegaMoE TopK-reduce module")
    return _LOADED_MODULE


def is_cake_megamoe_topk_reduce_module_loaded() -> bool:
    """Whether the reducer can launch without lazy build/load work."""

    return _LOADED_MODULE is not None


def get_cake_megamoe_topk_reduce_module():
    return load_cake_megamoe_topk_reduce_module()


def run_cake_megamoe_topk_reduce(
    partials: torch.Tensor,
    out: torch.Tensor,
    num_tokens: int,
) -> None:
    """Launch the reducer on ``partials``' current CUDA stream."""

    stream = torch.cuda.current_stream(device=partials.device).cuda_stream
    get_cake_megamoe_topk_reduce_module().run(
        partials,
        out,
        num_tokens,
        stream,
    )


__all__ = [
    "gen_cake_megamoe_topk_reduce_module",
    "get_cake_megamoe_topk_reduce_module",
    "get_cake_megamoe_topk_reduce_uri",
    "load_cake_megamoe_topk_reduce_module",
    "is_cake_megamoe_topk_reduce_module_loaded",
    "run_cake_megamoe_topk_reduce",
]
