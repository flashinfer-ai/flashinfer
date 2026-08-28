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

from .core import JitSpec, gen_jit_spec, sm120a_nvcc_flags

MOE_ROUTING_SM120_MODULE_NAME = "moe_routing_sm120"

# The kernel source lives next to the op it serves (registered as package data
# in pyproject.toml, like the experimental fused-GDN-decode source) so all code
# of the experimental fused MoE routing op stays together under
# flashinfer/fused_moe/experimental/.  It is only ever read from there; JIT
# output goes to FLASHINFER_GEN_SRC_DIR / FLASHINFER_JIT_DIR, never back into
# the package.
_MOE_ROUTING_KERNEL_DIR = (
    Path(__file__).resolve().parent.parent / "fused_moe" / "experimental" / "kernel"
)


def gen_moe_routing_sm120_module() -> JitSpec:
    """MoE routing prologue, align and weighted-sum finalize for SM120.

    One translation unit holds all three entry points and every supported token
    count, so the whole surface is covered by a single compiled module and
    CUDA-graph capture readiness is a single check.  There is no AOT entry for
    it: the module is built on the first non-capturing dispatch (see
    ``flashinfer/fused_moe/experimental/README.md``).
    """
    return gen_jit_spec(
        MOE_ROUTING_SM120_MODULE_NAME,
        [_MOE_ROUTING_KERNEL_DIR / "moe_routing_sm120.cu"],
        extra_cuda_cflags=sm120a_nvcc_flags,
    )
