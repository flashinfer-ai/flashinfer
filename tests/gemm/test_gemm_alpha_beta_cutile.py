# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gemm_alpha_beta_cutile kernel (GEMM with alpha/beta scaling)."""

import importlib.util
import pathlib
import random
import sys

import pytest
import torch

_REPO = pathlib.Path(__file__).resolve().parent.parent.parent


def _load_module(name, rel_path):
    path = _REPO / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_common = _load_module("cutile_common", "flashinfer/gemm/kernels/cutile/cutile_common.py")
is_cuda_tile_available = _common.is_cuda_tile_available

if not is_cuda_tile_available():
    pytest.skip("cuda.tile not available", allow_module_level=True)

_mod = _load_module(
    "gemm_alpha_beta_cutile",
    "flashinfer/gemm/kernels/cutile/gemm_alpha_beta_cutile.py",
)
gemm_alpha_beta = _mod.gemm_alpha_beta


class Test_FlashInfer_Matmul_Alpha_Beta:
    @staticmethod
    def reference(a, b, c, trans_a=False, trans_b=True, alpha=1.0, beta=0.0, dtype=torch.bfloat16):
        if trans_a:
            a = a.t()
        if trans_b:
            b = b.t()
        return torch.addmm(c, a.to(dtype), b.to(dtype), beta=beta, alpha=alpha, out=c).to(dtype)

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
        [(torch.float16, torch.float16), (torch.bfloat16, torch.bfloat16), (torch.float8_e4m3fn, torch.bfloat16)],
    )
    @pytest.mark.parametrize("trans_a, trans_b", [(False, True)])
    @pytest.mark.parametrize("alpha, beta", [(1.0, 0.0), (1.5, 2.0)])
    def test_op(self, m, n, k, dtype, out_dtype, trans_a, trans_b, alpha, beta):
        torch.manual_seed(0)
        random.seed(0)
        a, b = self.prepare_data(m, n, k, trans_a, trans_b, dtype)
        c = torch.rand((m, n), device=a.device, dtype=out_dtype)
        ref_c = c.clone()

        result = gemm_alpha_beta(a, b, c, trans_a, trans_b, alpha, beta)
        ref = self.reference(a, b, ref_c, trans_a, trans_b, alpha, beta, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
