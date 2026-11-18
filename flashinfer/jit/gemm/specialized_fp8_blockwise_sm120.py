# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""JIT registration of the specialized SM120 fp8 blockwise-scale decode GEMMs.

One module holding the agent-generated, shape-specialized kernels behind the
``"specialized"`` backend of ``gemm_fp8_nt_groupwise`` (see
``flashinfer/gemm/specialized_fp8_blockwise_sm120.py``). The module is pinned
to the exact ``sm_120a`` target: the kernels rely on the 64 KB shared-memory
carveout and 256-bit global loads that the family target does not admit, and
the allowlist was measured on SM120 only (SM121 resolves to the stock
backends). ``flashinfer/aot.py`` prebuilds it for jit-cache installs under the
same exact target.
"""

from pathlib import Path

from .. import env as jit_env
from ..core import JitSpec, gen_jit_spec


def gen_specialized_gemm_fp8_blockwise_sm120_module() -> JitSpec:
    src_dir = jit_env.FLASHINFER_CSRC_DIR / "specialized_gemm_fp8_blockwise_sm120"
    sources = sorted(Path(src_dir).glob("*.cu"))
    return gen_jit_spec(
        "specialized_gemm_fp8_blockwise_sm120",
        sources,
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "-std=c++17",
            "--extra-device-vectorization",
            "-Xptxas",
            "-O3",
            "-Xptxas",
            "--allow-expensive-optimizations=true",
            # single-token form REQUIRED: cpp_ext only treats "-gencode=..."
            # as a module arch override. The sm_120f family target caps
            # static smem at 48KB while these kernels opt into 64KB of
            # dynamic smem (sm_120a permits it), so the module must build
            # for sm_120a exactly.
            "-gencode=arch=compute_120a,code=sm_120a",
        ],
    )
