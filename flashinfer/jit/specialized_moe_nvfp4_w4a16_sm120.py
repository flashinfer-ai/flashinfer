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

"""JIT spec for the specialized SM120 NVFP4-W4A16 routed-MoE kernels
(contract: E=128, top_k=6, hidden=2688, intermediate=1856, non-gated
ReLU^2, bf16 activations). Exact-architecture module: sm_120a only;
``cvt.rn.bf16x2.e2m1x2`` requires a CUDA >= 13.1-line ptxas."""

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags

_SOURCES = (
    "moe_kernel.cu",
    "fast_t1.cu",
    "fixed_prefill.cu",
    "persistent_decode.cu",
    "wide_prefill.cu",
)


def gen_specialized_moe_nvfp4_w4a16_sm120_module() -> JitSpec:
    src_dir = jit_env.FLASHINFER_CSRC_DIR / "specialized_moe_nvfp4_w4a16_sm120"
    return gen_jit_spec(
        "specialized_moe_nvfp4_w4a16_sm120",
        [src_dir / s for s in _SOURCES],
        extra_cuda_cflags=sm120a_nvcc_flags + ["-std=c++17", "-lineinfo"],
        extra_ldflags=["-lcuda"],
    )
