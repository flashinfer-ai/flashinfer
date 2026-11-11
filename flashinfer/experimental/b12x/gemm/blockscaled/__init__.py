"""Prepared dense block-scaled GEMM: ``C = (A·SFA) @ (B·SFB)``.

Functional and caller-buffer operations share the SM120 warp-MMA engine (no
TMEM, no tcgen05, no 2-CTA). Recipes: NVFP4 (Float4E2M1 values, e4m3 scales, vec 16),
MXFP4 (e8m0 scales, vec 32), MXFP8 (e4m3 values, e8m0 scales, vec 32), and
tensor-scaled FP8. ``mm`` accepts raw ``(values, scales)`` operand pairs or a
weight returned by ``pack_weight``. A packed MXFP8 weight accepts either a
BF16/FP16 activation or a prequantized ``(values, scales)`` pair with compact
row-major or F8_128x4-swizzled UE8M0 scales. Pass swizzled storage flattened
(or as its native 6D view); a 2D ``[M,K/32]`` scale is interpreted as compact.
``expected_m`` and precision constraints are immutable query coordinates.

IQ2_XS and IQ2_XXS use BF16 activations and inline descriptor decoding in the same dense
warp-MMA engine. ``pack_weight(blocks, recipe='iq2_xs')`` losslessly rearranges
CUDA uint8 ``[N,K/256,74]`` safetensors payloads; scales are embedded, so no
separate scale argument is accepted. K must be divisible by 256 and N by 8.
``recipe="iq2_xxs"`` accepts 66-byte blocks and retains exactly 66 bytes per
256 weights in the same compute path. Codec-specific loading specializes at
compile time. Only A16 activation precision is supported for these recipes.
``recipe="q8_0"`` takes uint8 ``[N,K/32,34]`` blocks, retains INT8 payloads
and FP16 scales, and uses the same engines without a codebook.

``plan(query)`` declares an invocation without compiling or allocating.
``PreparationSession`` selects and primes its prepared ``Plan`` before
execution with an explicit plan. ``mm`` also accepts NVFP4, MXFP8, tensor-FP8,
and serialized operands without a plan: first use prepares the heuristic
configuration without autotuning and retains a power-of-two row capacity.
Warm each required capacity before graph capture or frozen kernel resolution.
The explicit ``w4a16``/``w8a16`` entry points require a plan. A16 uses
BF16 activations and inline weight dequantization; it performs no activation
scaling. ``plan_regimes`` combines exact static-shape variants with one bounded
dynamic-row execution while keeping runtime dispatch behind the custom-op boundary.

NVFP4 packed weights require ``pack_weight(..., recipe='nvfp4', global_scale=g,
global_scale_kind='multiplier')``. ``'reciprocal'`` accepts a weight quantizer
multiplier without creating another tensor. NVFP4 quantized activation calls
also require ``activation_global_scale``, the activation quantizer multiplier.
Global scales must be finite and strictly positive; reconstructed weights must
fit BF16. Quantized activation execution requires K divisible by 128 and N by 8.

Caller-owned output/workspace buffers are described by the query and reserved
before durable binding. Concurrent owners use disjoint buffers. Functional
forms retain their output allocation contract, while graph replay uses fixed
captured addresses. Concurrent executions sharing a heuristic plan must be
serialized; independent owners should prepare separate plans.
"""

from __future__ import annotations


from typing import TYPE_CHECKING

from ..._lib.meta import OpMeta, Provenance, install_lazy_api

META = OpMeta(
    name="blockscaled",
    group="gemm",
    api_style="planned",
    entry_points=(
        "Weight",
        "NVFP4LinearWeight",
        "IQ2XSLinearWeight",
        "BlockQuantLinearWeight",
        "BlockscaledQuery",
        "BlockscaledConfig",
        "FixedBlockscaledQuery",
        "plan",
        "plan_regimes",
        "query_from_call",
        "mm",
        "mm_mxfp4",
        "mm_nvfp4",
        "mm_block_fp8",
        "pack_weight",
        "is_supported",
        "w4a16",
        "w8a16",
        "workspace_size",
    ),
    dtypes=("bf16", "fp16", "fp32", "fp8_e4m3", "fp4_e2m1"),
    recipes=("nvfp4", "mxfp4", "mxfp8", "iq2_xs", "iq2_xxs", "q8_0"),
    requires=("triton",),
    provenance=Provenance(
        repo="https://github.com/lukealonso/b12x",
        commit="6627d342",
        paths=("b12x/gemm/dense.py",),
    ),
    test_path="tests/experimental/b12x/gemm/test_blockscaled.py",
    since="0.7.0",
)

if TYPE_CHECKING:  # static analysis only; runtime resolution is lazy
    from .api import (  # noqa: F401
        Weight,
        NVFP4LinearWeight,
        IQ2XSLinearWeight,
        BlockQuantLinearWeight,
        BlockscaledQuery,
        BlockscaledConfig,
        FixedBlockscaledQuery,
        plan,
        plan_regimes,
        query_from_call,
        is_supported,
        mm,
        mm_block_fp8,
        mm_mxfp4,
        mm_nvfp4,
        pack_weight,
        w4a16,
        w8a16,
        workspace_size,
    )

install_lazy_api(globals(), META)
