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

"""FP4 configuration/spec generation for the CuTe-DSL AOT builder."""

from itertools import product
import torch
from ..cute_dsl_aot import CuteDslAOTOperation

MM_FP4_AOT_ARCHS = frozenset({"sm100a", "sm103a", "sm107a"})


def mm_fp4_aot_configs(arch: str) -> dict[str, tuple]:
    """Enumerate every runtime FP4 specialization, before autotune ranking.

    Shapes and occupancy are runtime arguments. Three aligned representative
    shapes expose the full tactic space: narrow-N SM100 tactics require at
    most 32 rows/columns; wide tiles require a larger shape. K=8192 includes deep-K
    and split-K variants and disables the device-dependent prefetch filter.
    """
    if arch not in MM_FP4_AOT_ARCHS:
        return {}

    from ...cute_dsl.utils import torch_to_cutlass_dtype
    from ...gemm.gemm_mm_fp4_cute_dsl import (
        _get_mm_fp4_cute_dsl_tactics,
        _get_mm_fp4_kernel_classes,
        _blockscaled_kernel_disk_name,
        _mm_fp4_cache_key,
        mm_fp4_l2_policy,
        per_token_alpha_mode,
    )

    sm_version = int(arch[2:-1])
    _get_mm_fp4_kernel_classes(sm_version, require_native=True)

    payloads: dict[str, tuple] = {}
    for use_nvfp4, dtype, pdl in product(
        (True, False), (torch.float16, torch.bfloat16), (False, True)
    ):
        sf_vec_size = 16 if use_nvfp4 else 32
        tactics = set()
        for m, n in ((32, 8192), (8192, 32), (8192, 8192)):
            tactics.update(
                _get_mm_fp4_cute_dsl_tactics(
                    m,
                    n,
                    8192,
                    use_nvfp4,
                    torch_to_cutlass_dtype(dtype),
                    None,
                    sm_version,
                )
            )
        for tactic in sorted(tactics, key=repr):
            tile, _, swap, _, kernel_type, _ = tactic
            # Both L2 regions for every persistent tactic, and both alpha
            # modes where the public API serves per-token alpha.
            policies = {mm_fp4_l2_policy(m, tile, swap, kernel_type) for m in (1, 8192)}
            alpha_modes = (
                (False, True)
                if kernel_type in ("sm100", "sm100sk") and sm_version != 107
                else (False,)
            )
            for policy, per_token in product(sorted(policies, key=str), alpha_modes):
                key = _mm_fp4_cache_key(
                    sf_vec_size,
                    tactic,
                    pdl,
                    dtype,
                    per_token_alpha_mode(per_token, swap),
                    policy,
                )
                name = _blockscaled_kernel_disk_name(key, 1, None)
                if name in payloads and payloads[name] != key:
                    raise RuntimeError(f"FP4 specialization name collision: {name}")
                payloads[name] = key
    return dict(sorted(payloads.items()))


def make_mm_fp4_aot_spec(cache_key):
    """Construct the runtime FP4 specialization with symbolic shape/occupancy args."""
    import cutlass
    from ...cute_dsl.utils import torch_to_cutlass_dtype
    from ...gemm.gemm_mm_fp4_cute_dsl import (
        _blockscaled_gemm_cache_key_files,
        _blockscaled_kernel_disk_name,
        _make_mm_fp4_kernel,
        _make_blockscaled_gemm_compile_fn,
    )
    from ..cute_dsl_core import JitSpecCuteDsl, _hash_source_files

    sf_vec_size, _, _, swap_ab, _, _, _, _, out_dtype, alpha_mode, _ = cache_key
    compile_fn = _make_blockscaled_gemm_compile_fn(
        _make_mm_fp4_kernel(cache_key),
        ab_cutlass_dtype=cutlass.Uint8,
        sf_dtype=cutlass.Float8E4M3FN if sf_vec_size == 16 else cutlass.Float8E8M0FNU,
        c_cutlass_dtype=torch_to_cutlass_dtype(out_dtype),
        ab_assumed_align=32,
        swap_ab=swap_ab,
        sf_m=1,
        sf_n=1,
        sf_k=1,
        batch_size=1,
        max_active_clusters=1,
        per_token_alpha=alpha_mode,
        dynamic_max_active_clusters=True,
    )
    return JitSpecCuteDsl(
        "mm_fp4",
        _blockscaled_kernel_disk_name(cache_key, 1, None),
        compile_fn,
        _hash_source_files(_blockscaled_gemm_cache_key_files()),
    )


MM_FP4_OPERATION = CuteDslAOTOperation(
    name="mm_fp4",
    architectures=MM_FP4_AOT_ARCHS,
    configs=mm_fp4_aot_configs,
    make_spec=make_mm_fp4_aot_spec,
)
