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
from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def _q_token_kv_block_sparse_source_paths() -> tuple[list[Path], list[str | Path]]:
    """Resolve checkout-local QToken-KvBlock-Sparse-Attention sources, with installed-package fallback."""

    checkout = Path(__file__).resolve().parents[2]
    checkout_sources = [
        checkout / "csrc" / "prims_ts_q_token_kv_block_sparse_metadata.cu",
        checkout / "csrc" / "prims_ts_q_token_kv_block_sparse_metadata_jit_binding.cu",
    ]
    checkout_include = checkout / "include"
    if all(path.is_file() for path in checkout_sources) and checkout_include.is_dir():
        return checkout_sources, [checkout_include]
    return [
        jit_env.FLASHINFER_CSRC_DIR / "prims_ts_q_token_kv_block_sparse_metadata.cu",
        jit_env.FLASHINFER_CSRC_DIR
        / "prims_ts_q_token_kv_block_sparse_metadata_jit_binding.cu",
    ], []


@functools.cache
def gen_prims_ts_q_token_kv_block_sparse_metadata_module() -> JitSpec:
    """Build the JIT spec for direct Q1 and grouped sort-union metadata."""

    sources, include_paths = _q_token_kv_block_sparse_source_paths()
    return gen_jit_spec(
        "prims_ts_q_token_kv_block_sparse_metadata",
        sources,
        extra_include_paths=include_paths,
        extra_cuda_cflags=["-lineinfo"],
    )
