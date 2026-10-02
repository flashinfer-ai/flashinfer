"""
Copyright (c) 2024 by FlashInfer team.

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

from typing import Optional, Tuple

from . import env as jit_env
from .core import JitSpec, gen_jit_spec


def gen_topk_module() -> JitSpec:
    return gen_jit_spec(
        "topk",
        [
            jit_env.FLASHINFER_CSRC_DIR / "topk.cu",
            jit_env.FLASHINFER_CSRC_DIR / "cub_topk.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_topk_binding.cu",
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_fast_topk_clusters_binding.cu",
        ],
        extra_cuda_cflags=["-lineinfo"],
    )


def gen_sglang_dsv4_topk_module(
    capability: Optional[Tuple[int, int]] = None,
) -> JitSpec:
    """Vendored sglang DeepSeek-V4 ragged top-k (benchmark reference backend).

    The vendored sources are C++20 (concepts, designated initializers);
    ``-std=c++20`` here replaces gen_jit_spec's default ``-std=c++17``.
    sglang's headers require ``-DSGL_CUDA_ARCH=<__CUDA_ARCH__ value>`` and
    static_assert that it equals the device-pass target, so the module is
    built for exactly ONE architecture: a single ``-gencode`` for
    ``capability`` (the current device's when ``None``), which makes cpp_ext
    drop the global multi-arch list, with the arch in the module name so
    hosts of different architectures never share a cached ``.so``.
    """
    import torch

    from ..compilation_context import CompilationContext, cutlass_supports_sm107

    src_dir = jit_env.FLASHINFER_CSRC_DIR / "sglang_dsv4"
    if capability is None:
        capability = torch.cuda.get_device_capability()
    major, minor = CompilationContext._normalize_cuda_arch(*capability)
    arch = f"{major}{minor}"
    # SGL_CUDA_ARCH must equal the __CUDA_ARCH__ of the device pass, i.e. the
    # gencode target rather than the raw capability: Rubin (10.7) builds as
    # the sm100f family target while the bundled CUTLASS lacks native
    # compute_107a (the rule cpp_ext applies to the global list), and the
    # device pass then sees 1000.
    if arch == "107a" and not cutlass_supports_sm107():
        arch = "100f"
    sgl_arch = int("".join(ch for ch in arch if ch.isdigit())) * 10
    return gen_jit_spec(
        f"sglang_dsv4_topk_sm{arch}",
        [
            src_dir / "sglang_dsv4_topk.cu",
            src_dir / "flashinfer_sglang_dsv4_topk_binding.cu",
        ],
        extra_cflags=["-std=c++20", f"-DSGL_CUDA_ARCH={sgl_arch}"],
        extra_cuda_cflags=[
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            "-std=c++20",
            "-lineinfo",
            f"-DSGL_CUDA_ARCH={sgl_arch}",
        ],
        extra_include_paths=[src_dir],
    )
