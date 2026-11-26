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

from ..core import JitSpec, gen_jit_spec, sm107a_nvcc_flags
from .. import env as jit_env


def gen_mm_mxfp8_smallm_sm107_module() -> JitSpec:
    """JIT spec for the SM107 small-M MXFP8 GEMM fast path.

    ``csrc/mm_mxfp8_smallm_sm107.cu`` is a single translation unit holding one
    kernel template instantiated per allowlisted shape family plus its TVM-FFI
    binding. The kernel targets sm_107a only (it uses the fused scaled
    ``cvt.rn.scaled::n2::ue8m0.bf16x2.e4m3x2`` up-conversion, ``add.rn.f32x2``
    and thread-block cluster DSMEM reductions), so the spec must keep the
    ``compute_107a`` gencode and is only built when the dispatch gate in
    ``flashinfer/gemm/mxfp8_smallm_sm107.py`` has already matched an SM107
    device. ``-lcuda`` is required for ``cuStreamGetGreenCtx``.
    """
    return gen_jit_spec(
        "mm_mxfp8_smallm_sm107",
        [jit_env.FLASHINFER_CSRC_DIR / "mm_mxfp8_smallm_sm107.cu"],
        extra_cuda_cflags=sm107a_nvcc_flags,
        extra_ldflags=["-lcuda"],
        extra_include_paths=[jit_env.FLASHINFER_CSRC_DIR],
    )
