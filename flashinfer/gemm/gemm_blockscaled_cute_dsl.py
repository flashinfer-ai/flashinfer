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

"""Tactic enumeration shared by FP4 and MXFP8 CuTe-DSL runners."""

import torch

from .kernels.utils import _CLUSTER_SHAPE_MN_CANDIDATES, _SM100_MMA_TILER_MN_CANDIDATES


def _get_approximate_cta_nums(m, n, tile_mn, cluster_shape_mn):
    tile_m, tile_n = tile_mn
    cluster_m, cluster_n = cluster_shape_mn
    ctas_m = ((m + tile_m - 1) // tile_m + cluster_m - 1) // cluster_m * cluster_m
    ctas_n = ((n + tile_n - 1) // tile_n + cluster_n - 1) // cluster_n * cluster_n
    return ctas_m * ctas_n


def _get_sm100_block_scaled_tactics(
    m,
    n,
    real_k,
    ab_dtype,
    sf_dtype,
    sf_vec_size,
    c_cutlass_dtype,
    device,
):
    """Enumerate valid SM100 block-scaled GEMM tactics for autotuning.

    Returns list of (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
    tuples.  Shared by both the mxfp8 and fp4 cute-dsl runners.
    """
    from .kernels.dense_blockscaled_gemm_sm100 import (
        Sm100BlockScaledPersistentDenseGemmKernel,
    )

    batch_size = 1
    n_aligned = n % 8 == 0

    valid_tactics = []
    for mma_tiler_mn in _SM100_MMA_TILER_MN_CANDIDATES:
        for cluster_shape_mn in _CLUSTER_SHAPE_MN_CANDIDATES:
            for swap_ab in (False, True):
                if not n_aligned:
                    continue

                if swap_ab:
                    c_major = "m"
                    kernel_m, kernel_n = n, m
                else:
                    c_major = "n"
                    kernel_m, kernel_n = m, n

                if not Sm100BlockScaledPersistentDenseGemmKernel.can_implement(
                    ab_dtype,
                    sf_dtype,
                    sf_vec_size,
                    c_cutlass_dtype,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    kernel_m,
                    kernel_n,
                    real_k,
                    batch_size,
                    "k",
                    "k",
                    c_major,
                ):
                    continue

                for use_prefetch in (False, True):
                    if use_prefetch and real_k < 8192:
                        cta_nums = _get_approximate_cta_nums(
                            kernel_m, kernel_n, mma_tiler_mn, cluster_shape_mn
                        )
                        sm_count = torch.cuda.get_device_properties(
                            device
                        ).multi_processor_count
                        cta_wave_ratio = cta_nums / sm_count
                        if not (0.5 < cta_wave_ratio < 1.0 or real_k >= 8192):
                            continue

                    valid_tactics.append(
                        (mma_tiler_mn, cluster_shape_mn, swap_ab, use_prefetch)
                    )
    return valid_tactics
