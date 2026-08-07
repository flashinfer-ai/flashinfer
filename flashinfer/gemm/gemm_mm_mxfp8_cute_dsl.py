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
import importlib.metadata
from typing import List, Literal, Optional

import torch

from ..autotuner import OptimizationProfile, TunableRunner
from ..jit.cpp_ext import get_cuda_version
from ..utils import get_device_sm_count, supported_compute_capability
from .gemm_mm_fp4_cute_dsl import (
    _compile_block_scaled_gemm,
    _prepare_alpha_for_launch,
)


@functools.cache
def _b12x_mxfp8_dsl_supported() -> bool:
    # Probe distribution metadata rather than import cutlass.cute, which
    # takes seconds. Cached because the lookup hits the filesystem and this
    # runs on every call.
    try:
        dsl_version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        return False
    from packaging import version as pkg_version

    try:
        return pkg_version.Version(dsl_version).release[:2] >= (4, 6)
    except pkg_version.InvalidVersion:
        return False


@supported_compute_capability([120, 121])
def _b12x_gemm_mxfp8_requirement(
    a: torch.Tensor,
    b: torch.Tensor,
    a_descale: torch.Tensor,
    b_descale: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    out_dtype: torch.dtype = torch.bfloat16,
    use_8x4_sf_layout: bool = True,
    backend: Literal["cutlass", "cute-dsl", "trtllm", "cudnn", "b12x", "auto"] = "auto",
):
    if get_cuda_version().major < 13:
        if backend != "b12x":
            return False
        raise ValueError(
            "b12x mm_mxfp8 requires CUDA 13 or later. "
            f"Current CUDA version: {get_cuda_version()}."
        )
    if use_8x4_sf_layout or a_descale.ndim != 1 or b_descale.ndim != 1:
        if backend != "b12x":
            return False
        raise ValueError(
            "b12x mm_mxfp8 requires 1D 128x4-swizzled block scales. "
            "Use mxfp8_quantize(..., is_sf_swizzled_layout=True)."
        )
    if a.shape[1] % 128 != 0:
        if backend != "b12x":
            return False
        raise ValueError(
            "b12x mm_mxfp8 requires the contraction dim K to be a multiple of "
            f"128 (one full BK128 tile). Got K={a.shape[1]}."
        )
    if not _b12x_mxfp8_dsl_supported():
        if backend != "b12x":
            return False
        raise ValueError(
            "b12x mm_mxfp8 requires nvidia-cutlass-dsl >= 4.6.0 "
            "(cute.nvgpu.warp.MmaMXF8Op)."
        )
    return True


# Module-level kernel cache for b12x MXFP8 GEMM.
_B12X_MM_MXFP8_KERNEL_CACHE: dict[tuple, tuple] = {}


def _b12x_gemm_mxfp8_runner(
    sm_major: int,
    sm_minor: int,
    enable_pdl: bool,
    out_dtype: torch.dtype,
):
    """Create a b12x MXFP8 GEMM runner for SM12x.

    Same warp-level MMA kernel as the b12x FP4 backend, driven with MXFP8
    operands (m16n8k32 atom, UE8M0 scales, BK128) and the MXFP8 tile regimes.
    """
    import cutlass

    from .kernels.dense_blockscaled_gemm_sm120_b12x import (
        Sm120B12xBlockScaledDenseGemmKernel,
        _select_default_dense_gemm_plan,
    )

    from ..cute_dsl.utils import torch_to_cutlass_dtype

    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(
            f"b12x backend does not support output dtype {out_dtype}. "
            f"Supported: torch.bfloat16, torch.float16."
        )
    c_cutlass_dtype = torch_to_cutlass_dtype(out_dtype)

    def _default_dense_plan(m, n, real_k, device):
        return _select_default_dense_gemm_plan(
            m,
            n,
            real_k,
            get_device_sm_count(device),
            is_mxfp8=True,
            expected_m=m,
        )

    class B12xMxfp8GemmRunner(TunableRunner):
        """TunableRunner for b12x block-scaled MXFP8 dense GEMM on SM12x."""

        def get_valid_tactics(
            self,
            inputs: List[torch.Tensor],
            profile: OptimizationProfile,
        ) -> list:
            (a, b, a_descale, b_descale, _, out, _) = inputs
            m = a.shape[0]
            real_k = a.shape[1]
            n = b.shape[1]

            sf_vec_size = 32
            ab_dtype = cutlass.Float8E4M3FN
            sf_dtype = cutlass.Float8E8M0FNU
            batch_size = 1

            valid_tactics = []

            def _add(mma_tiler_mn, swap_ab):
                # can_implement takes no m, so validity is M-independent.
                if not Sm120B12xBlockScaledDenseGemmKernel.can_implement(
                    ab_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    (1, 1),
                    n,
                    real_k,
                    batch_size,
                    "k",
                    "k",
                    "n",
                    swap_ab=swap_ab,
                ):
                    return
                for use_prefetch in (False, True):
                    tac = (mma_tiler_mn, (1, 1), swap_ab, use_prefetch, "sm120", None)
                    if tac not in valid_tactics:
                        valid_tactics.append(tac)

            # A few tiles for the tuner to profile (a larger grid overfits the
            # bucket representative and makes picks noisier).
            for mma_tiler_mn in [
                (16, 128),
                (32, 128),
                (64, 64),
                (64, 128),
                (128, 128),
            ]:
                _add(mma_tiler_mn, swap_ab=False)

            # Also include the default-path tile so the tuner can't pick worse
            # than static.
            plan = _default_dense_plan(m, n, real_k, a.device)
            _add(plan.mma_tiler_mn, swap_ab=plan.swap_ab)
            return valid_tactics

        def forward(
            self,
            inputs: List[torch.Tensor],
            tactic=None,
            do_preparation: bool = False,
            **kwargs,
        ):
            (a, b, a_descale, b_descale, _, out, _) = inputs
            m = a.shape[0]
            real_k = a.shape[1]
            n = b.shape[1]

            sf_vec_size = 32
            sf_dtype = cutlass.Float8E8M0FNU
            batch_size = 1

            if tactic is None or tactic == -1:
                plan = _default_dense_plan(m, n, real_k, a.device)
                tactic = (
                    plan.mma_tiler_mn,
                    (1, 1),
                    plan.swap_ab,
                    False,
                    "sm120",
                    None,
                )

            (
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
            ) = tactic

            kernel_a, kernel_b = a, b.T
            kernel_a_sf, kernel_b_sf = a_descale, b_descale

            sf_m = (m + 127) // 128
            sf_n = (n + 127) // 128
            sf_k = (real_k // sf_vec_size + 3) // 4

            cache_key = (
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                swap_ab,
                use_prefetch,
                kernel_type,
                use_tma_store,
                enable_pdl,
                out_dtype,
            )

            make_kernel = lambda: Sm120B12xBlockScaledDenseGemmKernel(
                sf_vec_size,
                mma_tiler_mn,
                cluster_shape_mn,
                mma_k=32,
                tile_k=128,
                use_prefetch=use_prefetch,
                enable_pdl=enable_pdl,
                swap_ab=swap_ab,
            )

            # swap_ab is applied inside the kernel, so the public C tensor
            # stays row-major (m, n).
            compiled_gemm, _ = _compile_block_scaled_gemm(
                _B12X_MM_MXFP8_KERNEL_CACHE,
                cache_key,
                make_kernel,
                ab_cutlass_dtype=cutlass.Float8E4M3FN,
                sf_dtype=sf_dtype,
                c_cutlass_dtype=c_cutlass_dtype,
                ab_assumed_align=32,
                cluster_shape_mn=cluster_shape_mn,
                swap_ab=False,
                sf_m=sf_m,
                sf_n=sf_n,
                sf_k=sf_k,
                batch_size=batch_size,
            )

            alpha_for_launch = _prepare_alpha_for_launch(None, a.device)

            compiled_gemm(
                kernel_a,
                kernel_b,
                out,
                sf_m,
                sf_n,
                sf_k,
                kernel_a_sf.data_ptr(),
                kernel_b_sf.data_ptr(),
                alpha_for_launch,
            )
            return out

    return B12xMxfp8GemmRunner()
