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

"""Math, dtype and SMEM-descriptor helpers for the MLA decode kernels."""

from functools import partial
import math
import struct
import numbers

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.experimental import primitives as prims

# E4M3FN finite maximum used when clamping FP8 output conversion.
FP8_E4M3_MAX = 448.0

fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")

fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd="rn")
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd="rn")
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd="rn")


def canonicalize_softmax_rescale_threshold_log2(
    threshold: object, *, qkv_dtype: str
) -> float:
    """Range-check ``softmax_rescale_threshold_log2``; return it float32-rounded.

    Zero keeps the exact rescale.  A positive value is compiled into the
    kernel, which compares it with the log2-scaled row-max increase.  Any zero
    (including ``-0.0``) returns ``0.0`` so it selects the default kernel.
    """

    if isinstance(threshold, bool) or not isinstance(threshold, numbers.Real):
        raise TypeError("softmax_rescale_threshold_log2 must be a real number")
    try:
        value = float(threshold)
    except OverflowError:
        value = math.inf
    if value == 0.0:
        # The default needs no per-dtype bound.
        return 0.0
    bound = max_softmax_rescale_threshold_log2(qkv_dtype)
    if math.isfinite(value) and 0.0 < value <= bound:
        # The kernel compares in float32: canonicalize so values equal in
        # float32 share one compiled kernel, and reject positive values that
        # round to zero (they would change the FP8 P scale without skipping).
        value = struct.unpack("<f", struct.pack("<f", value))[0]
        if value == 0.0:
            raise ValueError(
                "softmax_rescale_threshold_log2 must be 0 or at least the smallest "
                "positive float32"
            )
    if not math.isfinite(value) or not 0.0 <= value <= bound:
        raise ValueError(
            "softmax_rescale_threshold_log2 must be a finite number in "
            f"[0, {bound:g}] for {qkv_dtype} P"
        )
    return value


def ceil_div(a, b):
    """Return integer ceil(a / b) for positive integer-like values."""
    return (a + b - 1) // b


@cute.jit
def final_lse(value, base_on_e: cutlass.Constexpr[bool]):
    """Convert final statistics; partial/reduction statistics stay base2."""
    # Preserve the caller's base-2 value for subsequent split normalization.
    if cutlass.const_expr(base_on_e):
        return value * Float32(0.6931471805599453)
    return value


def max_softmax_rescale_threshold_log2(qkv_dtype: str) -> float:
    """Return the largest legal ``softmax_rescale_threshold_log2`` per P dtype.

    P reaches ``2**T``. ``"e4m3"`` caps at 8 binades: P is quantized at
    ``448 * 2**-T``, and a larger T would push more of the row into E4M3
    subnormals. ``"bf16"`` caps at 32 binades (P <= ``2**32``). Both bounds
    match the trtllm-gen FMHA ``skipCorrThreshold`` checks for E4M3 and BF16
    BMM2.
    """

    if qkv_dtype == "e4m3":
        return 8.0
    if qkv_dtype == "bf16":
        return 32.0
    raise ValueError(f"unsupported qkv_dtype={qkv_dtype!r}")


def mma_k_step_for_qkv(cfg) -> int:
    """Return the K step consumed by one tcgen05 MMA instruction."""
    return 32 if cfg.is_fp8_qkv() else 16


def mma_kind_for_qkv(cfg):
    """Return the tcgen05 MMA kind for QK and PV operations."""
    if cfg.is_fp8_qkv():
        return prims.Tcgen05MMAKind.F8F6F4
    return prims.Tcgen05MMAKind.F16


def output_dtype(cfg):
    """Return the output element dtype for MLA decode examples."""
    if getattr(cfg, "use_fp8_output", 0) == 1:
        return cutlass.Float8E4M3FN
    return cutlass.BFloat16


def p_desc_layout(cfg):
    """Return the UMMA descriptor layout for P in SMEM."""
    del cfg
    return 4


def p_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for P in SMEM."""
    del cfg
    return 16


def p_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for P in SMEM."""
    del cfg
    return 512


def partial_output_dtype(cfg):
    """Return the split-KV partial-O dtype used before the final reduction."""
    if getattr(cfg, "num_ctas_per_seq_kv", 1) > 1:
        return cutlass.BFloat16
    return output_dtype(cfg)


def qk_desc_layout(cfg):
    """Return the UMMA descriptor layout for the latent QK operand."""
    return qk_desc_layout_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def qk_desc_layout_for_head_dim(cfg, head_dim: int):
    """Return the UMMA descriptor layout for a QK operand head dimension."""
    if cfg.is_fp8_qkv() and head_dim == 64:
        return 4
    return 2


def qk_desc_leading_byte_offset(cfg) -> int:
    """Return the descriptor leading byte offset for latent QK."""
    return qk_desc_leading_byte_offset_for_head_dim(
        cfg, cfg.mma_qk_tiler[0] // cfg.num_mma_ctas, cfg.mma_qk_tiler_k
    )


def qk_desc_leading_byte_offset_for_head_dim(cfg, tile_rows: int, head_dim: int) -> int:
    """Return the descriptor leading byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return tile_rows * head_dim * cfg.qkv_dtype_bytes
    return 16


def qk_desc_stride_byte_offset(cfg) -> int:
    """Return the descriptor stride byte offset for latent QK."""
    return qk_desc_stride_byte_offset_for_head_dim(cfg, cfg.mma_qk_tiler_k)


def qk_desc_stride_byte_offset_for_head_dim(cfg, head_dim: int) -> int:
    """Return the descriptor stride byte offset for one QK head dimension."""
    if cfg.is_fp8_qkv():
        return qkv_major_k_stride_bytes_for(cfg, head_dim)
    return 1024


def qkv_dtype(cfg):
    """Return the shared Q/K/V element dtype for MLA decode examples."""
    if cfg.is_fp8_qkv():
        return cutlass.Float8E4M3FN
    return cutlass.BFloat16


def qkv_major_k_stride_bytes_for(cfg, head_dim: int) -> int:
    """Return the shared-memory major-K stride for one Q/K/V head dimension."""
    num_smem_cols = 128 // cfg.qkv_dtype_bytes
    rows_per_smem_row = max(1, num_smem_cols // head_dim)
    if rows_per_smem_row == 1:
        rows_per_swizzle_block = 8
    elif rows_per_smem_row == 2:
        rows_per_swizzle_block = 4
    elif rows_per_smem_row == 4:
        rows_per_swizzle_block = 2
    else:
        rows_per_swizzle_block = 1
    return 128 * rows_per_swizzle_block


def qkv_smem_swizzle(cfg):
    """Return the SMEM swizzle used by staged Q/K/V tensors."""
    if cfg.is_fp8_qkv() and cfg.head_dim_per_stage_kv == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B


def qkv_smem_swizzle_for_head_dim(cfg, head_dim: int):
    """Return the staged Q/K/V SMEM swizzle for one head-dimension width."""
    if cfg.is_fp8_qkv() and head_dim == 64:
        return prims.Tcgen05SmemSwizzle.SWIZZLE_64B
    return prims.Tcgen05SmemSwizzle.SWIZZLE_128B
