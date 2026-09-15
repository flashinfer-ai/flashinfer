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

import triton
import triton.language as tl


@triton.jit
def _pack_e2m1(even, odd):
    # PTX packs its first source into the high nibble, so pass odd first.
    packed = tl.inline_asm_elementwise(
        asm="""{
            .reg .b8 b0, b1, b2, b3;
            mov.b32 {b0, b1, b2, b3}, 0;
            cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;
            mov.b32 $0, {b0, b1, b2, b3};
        }""",
        constraints="=r,f,f",
        args=[even, odd],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    )
    return packed.to(tl.uint8)


@triton.jit
def _rmsnorm_nvfp4_kernel(
    X,
    Gamma,
    Q,
    Scales,
    GlobalScale,
    K: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SWIZZLED: tl.constexpr,
    HAS_GLOBAL_SCALE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_K)
    x = tl.load(X + row * K + cols, cols < K, other=0).to(tl.float32)

    # All valid columns participate before any group can be normalized.
    rstd = tl.rsqrt(tl.sum(x * x, axis=0) / K + EPS)
    gamma = tl.load(Gamma + cols, cols < K, other=0).to(tl.float32)
    y = (x * rstd) * gamma

    # Keep normalization in FP32; do not materialize or round a BF16 Y.
    groups = tl.reshape(y, (BLOCK_K // 16, 16))
    amax = tl.max(tl.abs(groups), axis=1)
    global_scale = tl.load(GlobalScale) if HAS_GLOBAL_SCALE else 1.0
    sf8 = tl.minimum((amax / 6.0) * global_scale, 448.0).to(
        tl.float8e4nv, fp_downcast_rounding="rtne"
    )
    scale = sf8.to(tl.float32)
    inv_scale = tl.where(scale > 0, global_scale / scale, 0.0)
    normalized = tl.reshape(groups * inv_scale[:, None], (BLOCK_K // 2, 2))
    even, odd = tl.split(normalized)
    packed = _pack_e2m1(even, odd)

    pairs = tl.arange(0, BLOCK_K // 2)
    tl.store(Q + row * (K // 2) + pairs, packed, pairs < K // 2)

    # Scale layout [ceil(M/128), ceil(K/64), 32, 4, 4].
    g = tl.arange(0, BLOCK_K // 16)
    k_tiles: tl.constexpr = triton.cdiv(K // 16, 4)
    if SWIZZLED:
        offset = (
            (row // 128 * k_tiles + g // 4) * 512
            + row % 32 * 16
            + (row % 128 // 32) * 4
            + g % 4
        )
    else:
        offset = row * (K // 16) + g
    tl.store(Scales + offset, sf8.to(tl.uint8, bitcast=True), g < K // 16)
