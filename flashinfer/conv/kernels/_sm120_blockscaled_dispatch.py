# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""SM120 block-scaled GEMM dispatch helpers shared by the cooperative and
pingpong examples in this directory. Both examples are otherwise standalone;
they only depend on this file (and the rest of the cutlass DSL library) and
not on each other.

Dispatch table:
    Same-dtype paths:
        (FP4, FP4, *, Float8E4M3FN, 16) -> MmaMXF4NVF4Op,  use_mxf8f6f4=False, mma_K=64
        (FP4, FP4, *, Float8E8M0FNU, 32) -> MmaMXF4Op,    use_mxf8f6f4=False, mma_K=64
        (FP8, FP8, *, Float8E8M0FNU, 32) -> MmaMXF8Op,    use_mxf8f6f4=True,  mma_K=32
            (FP8 same-dtype is restricted to a_dtype == b_dtype.)
    Mixed-precision:
        (FP4, FP8, *, Float8E8M0FNU, 32) -> MmaMXF8F6F4Op, use_mxf8f6f4=True, mma_K=32
        (FP8, FP4, *, Float8E8M0FNU, 32) -> MmaMXF8F6F4Op, use_mxf8f6f4=True, mma_K=32
            (Same-width mixed-FP8 (E4M3 + E5M2) and FP6 mixed pairs are not supported.)

`use_mxf8f6f4` is the third argument to
`cutlass.utils.blackwell_helpers.get_permutation_mnk` and selects perm_k = 32
(FP8 / mixed warp-MMA shape (16,8,32)) instead of perm_k = 64 (FP4 shape (16,8,64)).
"""

import cutlass
import cutlass.cute as cute


# Number of bits to shift the FP4 byte left by before mma.sync.kind::mxf8f6f4.
# ldsm.b4x16_p64 places the FP4 nibble in the LOW half of the 8-bit register
# byte; mma.sync.kind::mxf8f6f4 reads the FP4 from the MIDDLE of the byte. The
# fixed `<< 2` shift moves it into position. See the C++ reference
# implementation `cute::fp4_shift_A/B` in
# cutlass/include/cute/atom/mma_traits_sm120.hpp.
FP4_SHIFT_BITS = 2

_FP8_DTYPES = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)


def make_ldmatrix_atom(operand_dtype, transpose, num_matrices=4, mixed_mode=False):
    """Build a warp-level ldmatrix copy atom for the SM120 block-scaled GEMM
    A/B SMEM->RMEM path.
    """
    if mixed_mode and operand_dtype.width == 4:
        assert not transpose, "LdMatrix8x16x8bOp does not support transpose"
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x16x8bOp(
                transpose=False,
                num_matrices=num_matrices,
                unpack_bits=4,
            ),
            cutlass.Int8,
        )
    atom_element_type = operand_dtype
    if mixed_mode and operand_dtype.width < 8:
        atom_element_type = cutlass.Int8
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=num_matrices,
        ),
        atom_element_type,
    )


def make_sm120_blockscaled_mma_op(a_dtype, b_dtype, acc_dtype, sf_dtype, sf_vec_size):
    """Dispatch the SM120 block-scaled MMA op for the given operand / scale
    factor combination.

    Returns
    -------
    Tuple[MmaOp, bool]
        (mma_op, use_mxf8f6f4) where use_mxf8f6f4 is True for FP8 / mixed
        paths (atom shape (16,8,32)) and False for FP4 same-dtype paths
        (atom shape (16,8,64)). `use_mxf8f6f4` is consumed by
        `cutlass.utils.blackwell_helpers.get_permutation_mnk`.

    Raises
    ------
    ValueError
        On any unsupported combination, with a diagnostic that names the
        offending combination and the supported alternatives.
    """
    # Same-dtype paths use the FP4 / FP8 same-dtype atoms.
    if a_dtype == b_dtype:
        if a_dtype == cutlass.Float4E2M1FN:
            if sf_vec_size == 16:
                if sf_dtype != cutlass.Float8E4M3FN:
                    raise ValueError(
                        f"Float4E2M1FN + sf_vec_size=16 requires sf_dtype=Float8E4M3FN, "
                        f"got sf_dtype={sf_dtype}"
                    )
                return (
                    cute.nvgpu.warp.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype),
                    False,
                )
            if sf_vec_size == 32:
                if sf_dtype != cutlass.Float8E8M0FNU:
                    raise ValueError(
                        f"Float4E2M1FN + sf_vec_size=32 requires sf_dtype=Float8E8M0FNU, "
                        f"got sf_dtype={sf_dtype}"
                    )
                return (
                    cute.nvgpu.warp.MmaMXF4Op(a_dtype, acc_dtype, sf_dtype),
                    False,
                )
            raise ValueError(
                f"Float4E2M1FN requires sf_vec_size in (16, 32), got {sf_vec_size}"
            )
        if a_dtype in _FP8_DTYPES:
            if sf_vec_size != 32:
                raise ValueError(
                    f"FP8 ab_dtype ({a_dtype}) requires sf_vec_size=32, "
                    f"got {sf_vec_size}"
                )
            if sf_dtype != cutlass.Float8E8M0FNU:
                raise ValueError(
                    f"FP8 ab_dtype + sf_vec_size=32 requires sf_dtype=Float8E8M0FNU, "
                    f"got sf_dtype={sf_dtype}"
                )
            return (
                cute.nvgpu.warp.MmaMXF8Op(a_dtype, acc_dtype, sf_dtype),
                True,
            )
        raise ValueError(
            f"Unsupported same-dtype ab_dtype={a_dtype} for SM120 block-scaled GEMM. "
            f"Supported same-dtype: Float4E2M1FN, Float8E4M3FN, Float8E5M2."
        )
    # Mixed-precision FP4 x FP8 paths.
    is_a_fp4_b_fp8 = a_dtype == cutlass.Float4E2M1FN and b_dtype in _FP8_DTYPES
    is_a_fp8_b_fp4 = a_dtype in _FP8_DTYPES and b_dtype == cutlass.Float4E2M1FN
    if is_a_fp4_b_fp8 or is_a_fp8_b_fp4:
        if sf_vec_size != 32:
            raise ValueError(
                f"FP4 x FP8 mixed-precision requires sf_vec_size=32, got {sf_vec_size}"
            )
        if sf_dtype != cutlass.Float8E8M0FNU:
            raise ValueError(
                f"FP4 x FP8 mixed-precision requires sf_dtype=Float8E8M0FNU, "
                f"got sf_dtype={sf_dtype}"
            )
        return (
            cute.nvgpu.warp.MmaMXF8F6F4Op(a_dtype, b_dtype, acc_dtype, sf_dtype),
            True,
        )
    # Reject same-width mixed-FP8 (E4M3 + E5M2) and FP6 with named diagnostics.
    if a_dtype in _FP8_DTYPES and b_dtype in _FP8_DTYPES:
        raise ValueError(
            f"same-width mixed-FP8 (a_dtype={a_dtype}, b_dtype={b_dtype}) is not supported. "
            f"Supported FP4 x FP8 mixed pairs: (Float4E2M1FN, Float8E4M3FN), "
            f"(Float4E2M1FN, Float8E5M2), (Float8E4M3FN, Float4E2M1FN), "
            f"(Float8E5M2, Float4E2M1FN). Same-dtype FP8 uses MmaMXF8Op."
        )
    raise ValueError(
        f"Unsupported (a_dtype={a_dtype}, b_dtype={b_dtype}, sf_dtype={sf_dtype}, "
        f"sf_vec_size={sf_vec_size}) for SM120 block-scaled GEMM. FP6 mixed pairs "
        f"are not supported; supported mixed pairs are FP4 x FP8 only."
    )
