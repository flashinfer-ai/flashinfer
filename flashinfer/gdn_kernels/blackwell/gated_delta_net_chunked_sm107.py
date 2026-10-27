# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Rubin (SM107) specialization of the chunked Gated Delta Net prefill kernel."""

import cutlass
import cutlass.cute as cute

from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel


class RubinGatedDeltaNetChunkedKernel(GatedDeltaNetChunkedKernel):
    """SM107 specialization of the Blackwell chunked GDN kernel."""

    arch = "sm_107"
    state_load_l1c_evict_priority = cute.nvgpu.CacheEvictionPriority.EVICT_NORMAL

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # SM107 allows 576 TMEM columns; the extra 64 did not help.
        self.tmem_alloc_cols = 512
        if not self.use_initial_state:
            # Rebalanced from 216/256/32 to cut spills in the lightweight warps.
            self.num_regs_compute_group_0 = 208
            self.num_regs_compute_group_1 = 256
            self.num_regs_other = 40
        assert (
            self.num_regs_compute_group_0
            + self.num_regs_compute_group_1
            + self.num_regs_other
            <= 504
        ), "register budget above 504 hangs the CTA on SM107"

    @cute.jit
    def _scale_convert_shared_acc_fragment(
        self, frag, tGr, scalar, out_frag, out_dtype, sub
    ):
        """Packed f32x2 variant of the base-class hook."""
        num = cute.size(tGr.shape[0])
        if cutlass.const_expr(num % 2 == 0):
            s = cutlass.Float32(scalar)
            for k in cutlass.range_constexpr(0, num, 2):
                m0, m1 = cute.arch.mul_packed_f32x2(
                    (frag[k, 0, sub], frag[k + 1, 0, sub]),
                    (tGr[k, 0, sub], tGr[k + 1, 0, sub]),
                )
                f0, f1 = cute.arch.mul_packed_f32x2((m0, m1), (s, s))
                frag[k, 0, sub] = f0
                frag[k + 1, 0, sub] = f1
            out_frag[None, 0, sub].store(frag[None, 0, sub].load().to(out_dtype))
        else:
            for k in cutlass.range(
                cute.size(tGr.shape[0]), vectorize=True, unroll_full=True
            ):
                frag[k, 0, sub] = frag[k, 0, sub] * tGr[k, 0, sub] * scalar
            out_frag[None, 0, sub].store(frag[None, 0, sub].load().to(out_dtype))

    @cute.jit
    def _apply_beta_upconvert_ainv(self, tCrAICal, tKKrBeta, tCrAIAcc):
        """Fuse the half->f32 up-convert with the beta multiply in one packed FMA."""
        num = cute.size(tCrAIAcc)
        if cutlass.const_expr(
            self.inverse_dtype in (cutlass.Float16, cutlass.BFloat16) and num % 2 == 0
        ):
            zero = cutlass.Float32(0.0)
            for k in cutlass.range_constexpr(0, num, 2):
                if cutlass.const_expr(self.inverse_dtype == cutlass.Float16):
                    p0, p1 = cute.arch.fma_packed_f32x2_f16x2_f32x2_f32x2(
                        (tCrAICal[k], tCrAICal[k + 1]),
                        (tKKrBeta[k], tKKrBeta[k + 1]),
                        (zero, zero),
                    )
                else:
                    p0, p1 = cute.arch.fma_packed_f32x2_bf16x2_f32x2_f32x2(
                        (tCrAICal[k], tCrAICal[k + 1]),
                        (tKKrBeta[k], tKKrBeta[k + 1]),
                        (zero, zero),
                    )
                tCrAIAcc[k] = p0
                tCrAIAcc[k + 1] = p1
        else:
            tCrAIAcc.store(tCrAICal.load().to(self.acc_dtype))
            for k in cutlass.range(
                cute.size(tCrAIAcc), vectorize=True, unroll_full=True
            ):
                tCrAIAcc[k] = tCrAIAcc[k] * tKKrBeta[k]
