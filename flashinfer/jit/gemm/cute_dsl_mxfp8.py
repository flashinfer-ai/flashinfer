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

"""Persistent/cluster split-K MXFP8 exports for the CuTe-DSL AOT builder.

Warp split-K uses a separate experimental compiler and specializes on N/K;
it remains JIT-compiled rather than adding an unbounded shape inventory.
"""

import torch

from ..cute_dsl_aot import CuteDslAOTOperation

MM_MXFP8_AOT_ARCHS = frozenset({"sm100a", "sm103a", "sm107a"})


def mm_mxfp8_aot_configs(arch: str) -> dict[str, tuple]:
    """Cover vLLM's MXFP8 dtypes with persistent and cluster split-K tactics.

    The public mm_mxfp8 API enables PDL for SM100-family runners; native
    Rubin has a separate non-PDL interface. Both output dtypes are used by
    vLLM v0.32.0.
    """
    if arch not in MM_MXFP8_AOT_ARCHS:
        return {}
    import cutlass

    from ...cute_dsl.utils import torch_to_cutlass_dtype
    from ...gemm.gemm_blockscaled_cute_dsl import _get_sm100_block_scaled_tactics
    from ...gemm.gemm_mm_fp4_cute_dsl import _blockscaled_kernel_disk_name
    from ...gemm.gemm_mm_mxfp8_cute_dsl import _mm_mxfp8_cache_key
    from ...gemm.kernels.dense_blockscaled_gemm_sm100_splitk import (
        Sm100BlockScaledSplitKGemmKernel as SplitK,
    )
    from ...gemm.kernels.utils import _get_sm107_mxfp8_tactics

    configs: dict[str, tuple] = {}
    for dtype in (torch.float16, torch.bfloat16):
        c_dtype = torch_to_cutlass_dtype(dtype)
        tactics: set[tuple] = set()
        # K=8192 enables both prefetch settings without a device query. The
        # low-M/N representatives expose every narrow-tile specialization.
        for m, n in (
            (8, 8192),
            (8192, 8),
            (16, 8192),
            (8192, 16),
            (32, 8192),
            (8192, 32),
            (8192, 8192),
        ):
            tactics.update(
                (*tactic, 1)
                for tactic in _get_sm100_block_scaled_tactics(
                    m,
                    n,
                    8192,
                    cutlass.Float8E4M3FN,
                    cutlass.Float8E8M0FNU,
                    32,
                    c_dtype,
                    None,
                )
            )
        for m in (8, 16, 32):
            for slices in SplitK.SUPPORTED_SPLIT_K_SLICES:
                if SplitK.is_valid_tactic(m, 8192, cutlass.Float8E4M3FN, slices):
                    tactics.add(
                        (SplitK.mma_tiler_mn_for_m(m), (1, 1), True, False, slices)
                    )
        keys = [
            _mm_mxfp8_cache_key(tactic, True, dtype)
            for tactic in sorted(tactics, key=repr)
        ]
        if arch == "sm107a":
            keys.extend(
                _mm_mxfp8_cache_key(tactic, False, dtype, native_sm107=True)
                for tactic in _get_sm107_mxfp8_tactics(8192, 8192, 8192, c_dtype)
            )
        for key in keys:
            name = _blockscaled_kernel_disk_name(key, 1, None)
            if name in configs and configs[name] != key:
                raise RuntimeError(f"MXFP8 specialization name collision: {name}")
            configs[name] = key
    return dict(sorted(configs.items()))


def make_mm_mxfp8_aot_spec(cache_key):
    """Use runtime construction and symbolic shapes/occupancy for CPU exports."""
    import cutlass

    from ...cute_dsl.utils import torch_to_cutlass_dtype
    from ...gemm.gemm_mm_fp4_cute_dsl import (
        _blockscaled_gemm_cache_key_files,
        _blockscaled_kernel_disk_name,
        _make_blockscaled_gemm_compile_fn,
    )
    from ...gemm.gemm_mm_mxfp8_cute_dsl import _make_mm_mxfp8_kernel
    from ..cute_dsl_core import JitSpecCuteDsl, _hash_source_files

    compile_fn = _make_blockscaled_gemm_compile_fn(
        _make_mm_mxfp8_kernel(cache_key),
        ab_cutlass_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_cutlass_dtype=torch_to_cutlass_dtype(cache_key[8]),
        ab_assumed_align=16,
        swap_ab=cache_key[3],
        sf_m=1,
        sf_n=1,
        sf_k=1,
        batch_size=1,
        max_active_clusters=1,
        dynamic_max_active_clusters=True,
    )
    return JitSpecCuteDsl(
        "mm_mxfp8",
        _blockscaled_kernel_disk_name(cache_key, 1, None),
        compile_fn,
        _hash_source_files(_blockscaled_gemm_cache_key_files()),
    )


MM_MXFP8_OPERATION = CuteDslAOTOperation(
    name="mm_mxfp8",
    architectures=MM_MXFP8_AOT_ARCHS,
    configs=mm_mxfp8_aot_configs,
    make_spec=make_mm_mxfp8_aot_spec,
)
