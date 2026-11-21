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

from pathlib import Path

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm107a_nvcc_flags


def get_msa_blockscore_indexer_csrc_dir() -> Path:
    """Locate the indexer CUDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "msa_blockscore_indexer"
    if installed.exists():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "msa_blockscore_indexer"
    if checkout.exists():
        return checkout
    raise FileNotFoundError(
        "MSA block-score indexer CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def gen_msa_blockscore_indexer_sm107_module() -> JitSpec:
    """JIT spec for the SM107 MSA block-score top-k indexer.

    One module exporting ``run``: tcgen05 BF16 block-MAX scoring plus top-k
    selection (forced own block + top-15 strictly-prior blocks, deterministic
    lower-block-id tie-break). SM107-only (``sm_107a``); the host-side gate is
    in :mod:`flashinfer.msa_ops.blockscore_indexer`.
    """
    csrc_dir = get_msa_blockscore_indexer_csrc_dir()
    return gen_jit_spec(
        "msa_blockscore_indexer_sm107",
        [
            csrc_dir / "msa_blockscore_indexer_sm107_staged.cu",
            csrc_dir / "msa_blockscore_indexer_sm107_fused.cu",
            csrc_dir / "msa_blockscore_indexer_sm107_binding.cu",
        ],
        extra_cuda_cflags=sm107a_nvcc_flags,
        extra_ldflags=["-lcuda"],
    )
