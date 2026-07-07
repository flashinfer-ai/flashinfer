# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for masked_bmm_cutile kernel (masked batched matrix multiply)."""

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
    "masked_bmm_cutile",
    "flashinfer/gemm/kernels/cutile/masked_bmm_cutile.py",
)
masked_bmm = _mod.masked_bmm


def enumerate_m_grouped_masked():
    max_m = 4096

    cases = [
        (6, 512),
        # DeepGEMM default cases
        (1, 1024),
        (2, 512),
        (4, 256),
    ]
    # more GB200 cases
    num_experts = 288
    num_experts_per_token = 8
    for num_ranks in [4, 8, 16, 32, 36, 48, 72]:
        for num_tokens in [64, 128, 256, 384, 512, 768, 1024]:
            num_groups = num_experts // num_ranks
            expected_m_per_group = num_tokens * num_experts_per_token // num_groups
            cases.append((num_groups, expected_m_per_group))

    for num_groups, expected_m_per_group in cases:
        for n, k in (
            (4096, 7168),
            (7168, 2048),
        ):
            yield dict(
                num_groups=num_groups,
                max_m=max_m,
                expected_m_per_group=expected_m_per_group,
                n=n,
                k=k,
            )


def create_masked_m(num_groups, expected_m_per_group, max_m):
    masked_m = torch.empty((num_groups,), dtype=torch.int32, device="cuda")
    for j in range(num_groups):
        masked_m[j] = int(expected_m_per_group * random.uniform(0.7, 1.3))
    assert masked_m.amax().item() <= max_m
    return masked_m


class Test_FlashInfer_MaskedBMM:
    @staticmethod
    def reference(a, b, m_mask, trans_a=False, trans_b=False):
        if trans_a:
            a = torch.transpose(a, 1, 2)
        if trans_b:
            b = torch.transpose(b, 1, 2)
        return torch.bmm(a, b)

    @staticmethod
    def prepare_data(num_groups, max_m, expected_m_per_group, n, k, trans_a, trans_b, dtype):
        device = torch.device("cuda")
        q = num_groups
        m = max_m

        if trans_a:
            a_shape = (q, k, m)
        else:
            a_shape = (q, m, k)

        if trans_b:
            b_shape = (q, n, k)
        else:
            b_shape = (q, k, n)

        m_mask = create_masked_m(
            num_groups=num_groups,
            expected_m_per_group=expected_m_per_group,
            max_m=max_m,
        )

        a = torch.rand(a_shape, device=device, dtype=dtype, requires_grad=False)
        b = torch.rand(b_shape, device=device, dtype=dtype, requires_grad=False)

        a_impl = a.clone()
        b_impl = b.clone()
        # Set all the elements beyond the m_mask to 0
        if trans_a:
            for i in range(num_groups):
                a[i, :, m_mask[i]:] = 0
        else:
            for i in range(num_groups):
                a[i, m_mask[i]:, :] = 0

        return a, b, a_impl, b_impl, m_mask

    @pytest.mark.parametrize(
        "num_groups, max_m, expected_m_per_group, n, k",
        [
            (
                case["num_groups"],
                case["max_m"],
                case["expected_m_per_group"],
                case["n"],
                case["k"],
            )
            for case in list(enumerate_m_grouped_masked())[:2]  # Use smaller set for correctness
        ],
    )
    @pytest.mark.parametrize("dtype", [torch.float16])
    @pytest.mark.parametrize("trans_a", [False, True])
    @pytest.mark.parametrize("trans_b", [False, True])
    def test_op(
        self,
        num_groups,
        max_m,
        expected_m_per_group,
        n,
        k,
        dtype,
        trans_a,
        trans_b,
    ):
        torch.manual_seed(0)
        random.seed(0)
        a_ref, b_ref, a_impl, b_impl, m_mask = self.prepare_data(
            num_groups, max_m, expected_m_per_group, n, k, trans_a, trans_b, dtype
        )

        ref_c = self.reference(a_ref, b_ref, m_mask, trans_a, trans_b)
        c = masked_bmm(a_impl, b_impl, m_mask, trans_a, trans_b)

        for i in range(num_groups):
            c[i, m_mask[i]:, :] = 0
        torch.testing.assert_close(ref_c, c, atol=1e-2, rtol=1e-2)
