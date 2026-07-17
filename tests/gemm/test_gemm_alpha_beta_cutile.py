# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gemm_alpha_beta_cutile kernel (GEMM with alpha/beta scaling)."""

import random

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.gemm import gemm_alpha_beta
from flashinfer.utils import get_compute_capability


class Test_FlashInfer_Matmul_Alpha_Beta:
    @staticmethod
    def reference(
        a, b, c, trans_a=False, trans_b=True, alpha=1.0, beta=0.0, dtype=torch.bfloat16
    ):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        return torch.addmm(
            c, a.to(dtype), b.to(dtype), beta=beta, alpha=alpha, out=c
        ).to(dtype)

    @staticmethod
    def prepare_data(m, n, k, trans_a, trans_b, dtype):
        device = torch.device("cuda")

        a_size = m * k
        b_size = k * n
        a = torch.rand(a_size, device=device, dtype=torch.float16).to(dtype)
        b = torch.rand(b_size, device=device, dtype=torch.float16).to(dtype)

        if trans_a:
            a = a.view(k, m)
        else:
            a = a.view(m, k)
        if trans_b:
            b = b.view(n, k)
        else:
            b = b.view(k, n)

        return a, b

    @pytest.mark.parametrize("m, n, k", [(4096, 4096, 4096), (8192, 8192, 8192)])
    @pytest.mark.parametrize(
        "dtype, out_dtype",
        [
            (torch.float16, torch.float16),
            (torch.bfloat16, torch.bfloat16),
            (torch.float8_e4m3fn, torch.bfloat16),
        ],
    )
    @pytest.mark.parametrize("trans_a, trans_b", [(False, True)])
    @pytest.mark.parametrize("alpha, beta", [(1.0, 0.0), (1.5, 2.0)])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(
        self, m, n, k, dtype, out_dtype, trans_a, trans_b, alpha, beta, backend
    ):
        if backend == "cutile" and not is_cuda_tile_available():
            pytest.skip("cuda.tile not available")
        cc_num = get_compute_capability(torch.device("cuda:0"))[0] * 10
        if not gemm_alpha_beta.is_backend_supported(backend, cc_num):
            pytest.skip(
                f"gemm_alpha_beta {backend} backend not supported on compute capability {cc_num}."
            )

        torch.manual_seed(0)
        random.seed(0)
        a, b = self.prepare_data(m, n, k, trans_a, trans_b, dtype)
        c = torch.rand((m, n), device=a.device, dtype=out_dtype)
        ref_c = c.clone()

        result = gemm_alpha_beta(
            a, b, c, trans_a, trans_b, alpha, beta, backend=backend
        )
        ref = self.reference(a, b, ref_c, trans_a, trans_b, alpha, beta, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("m, n, k", [(256, 256, 256), (512, 128, 320)])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_beta_zero_ignores_nan_c(self, m, n, k, dtype, backend):
        """beta==0 must not read C: an uninitialized/NaN C must not poison the
        output (guards against ``0 * NaN`` in the epilogue)."""
        if backend == "cutile" and not is_cuda_tile_available():
            pytest.skip("cuda.tile not available")
        cc_num = get_compute_capability(torch.device("cuda:0"))[0] * 10
        if not gemm_alpha_beta.is_backend_supported(backend, cc_num):
            pytest.skip(
                f"gemm_alpha_beta {backend} backend not supported on compute capability {cc_num}."
            )

        torch.manual_seed(0)
        random.seed(0)
        a, b = self.prepare_data(m, n, k, False, True, dtype)
        # C is deliberately full of NaNs; with beta==0 it must be ignored.
        c = torch.full((m, n), float("nan"), device=a.device, dtype=dtype)

        result = gemm_alpha_beta(
            a, b, c, False, True, alpha=1.5, beta=0.0, backend=backend
        )
        assert not torch.isnan(result).any(), "beta==0 leaked NaN from C into output"
        ref = (1.5 * (a.to(dtype) @ b.t().to(dtype))).to(dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
