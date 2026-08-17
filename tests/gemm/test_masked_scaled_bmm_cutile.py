# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for masked_scaled_bmm_cutile kernel (masked FP8/FP4 block-scaled batched matrix multiply)."""

import random

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.gemm import masked_scaled_bmm
from flashinfer.utils import get_compute_capability

try:
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    HAS_MXFP = True
except ImportError:
    HAS_MXFP = False


def create_masked_m(num_groups, expected_m_per_group, max_m):
    """Draw random per-group row counts (rounded up to a multiple of 128), bounded by `max_m`."""
    masked_m = torch.empty((num_groups,), dtype=torch.int32, device="cuda")
    for j in range(num_groups):
        masked_m[j] = (
            int(expected_m_per_group * random.uniform(0.7, 1.3) + 127) // 128 * 128
        )
    assert masked_m.amax().item() <= max_m
    return masked_m


class Test_FlashInfer_MaskedScaledBMM:
    """Correctness tests for masked_scaled_bmm on the cuTile backend."""

    @staticmethod
    def initialize_block_scaled(
        num_groups,
        max_m,
        expected_m_per_group,
        N,
        K,
        block_scale_type="nvfp4",
        compute_reference=False,
        trans_a=False,
        trans_b=True,
        out_dtype=torch.bfloat16,
    ):
        """Build packed FP8/FP4 A/B, their MX-swizzled block scales, the per-group mask,
        and (optionally) a dequantized torch.bmm reference.

        Mirrors the ocean validation data-prep: A/B are generated with
        ``MXFP4Tensor`` (col-major, K-packed for fp4), and the scales are stored in
        the 5D TMA-descriptor layout ``[Q, m_tiles, k_tiles, 2, 256]`` the cuTile
        kernel expects.
        """
        VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
        assert block_scale_type in [
            "nvfp4",
            "mxfp4",
            "mxfp8",
            "mixed",
        ], f"Invalid block scale type: {block_scale_type}"

        Q = num_groups
        M = max_m
        assert not trans_a and trans_b, "Only NT layout is supported"
        a_shape = (Q, M, K)
        b_shape = (Q, N, K)

        m_mask = create_masked_m(
            num_groups=num_groups,
            expected_m_per_group=expected_m_per_group,
            max_m=max_m,
        )

        device = "cuda"
        a_ref = MXFP4Tensor(size=(a_shape), device=device).random()
        b_ref = MXFP4Tensor(size=(b_shape), device=device).random()
        if block_scale_type in ["mxfp8", "mixed"]:
            a_ref = a_ref.to(torch.float32)
            a = a_ref.to(torch.float8_e4m3fn)
        else:
            # Pack two fp4 elements per byte along K
            a = a_ref.to_packed_tensor(dim=2)

        if block_scale_type == "mxfp8":
            b_ref = b_ref.to(torch.float32)
            b = b_ref.to(torch.float8_e4m3fn)
        else:
            b = b_ref.to_packed_tensor(dim=2)

        b_ref = b_ref.to(torch.float32)
        b_ref = torch.transpose(b_ref, 1, 2)

        # Focus on NT layout, so k is the leading dimension of both A and B.
        a_scale_shape = [Q, M // 128, K // VEC_SIZE // 4, 32, 16]
        b_scale_shape = [Q, N // 128, K // VEC_SIZE // 4, 32, 16]
        epsilon = 1e-8
        a_scale = torch.rand(a_scale_shape, device=device) + epsilon
        b_scale = torch.rand(b_scale_shape, device=device) + epsilon
        if block_scale_type == "nvfp4":
            a_scale = a_scale.to(torch.float8_e4m3fn)
            b_scale = b_scale.to(torch.float8_e4m3fn)
            a_scale_ref = a_scale
            b_scale_ref = b_scale
        elif block_scale_type in ["mxfp4", "mxfp8", "mixed"]:
            a_scale_ref = MXScaleTensor(a_scale)
            b_scale_ref = MXScaleTensor(b_scale)
            a_scale = a_scale_ref.data
            b_scale = b_scale_ref.data

        # 5D TMA descriptor [Q, rep_m, rep_k, 2, 256] with uint8 elements.
        a_scale = a_scale.reshape(Q, a_scale_shape[1], a_scale.shape[2], 2, 256)
        b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)

        reference = None
        if compute_reference:
            a_scale_ref = a_scale_ref.to(torch.float32)
            b_scale_ref = b_scale_ref.to(torch.float32)

            def unpack_scale(packed):
                packed = packed.reshape(*packed.shape[:-2], 32, 4, 4)
                num_chunk_q, num_chunk_m, num_chunk_k, _, _, _ = packed.shape
                return (
                    packed.permute(0, 1, 4, 3, 2, 5)
                    .reshape(num_chunk_q, num_chunk_m * 128, num_chunk_k * 4)
                    .contiguous()
                )

            a_scale_ref = unpack_scale(a_scale_ref).repeat_interleave(VEC_SIZE, dim=2)[
                :Q, :M, :K
            ]
            b_scale_ref = (
                unpack_scale(b_scale_ref)
                .repeat_interleave(VEC_SIZE, dim=2)
                .permute(0, 2, 1)
                .contiguous()[:Q, :K, :N]
            )
            a_ref_float = a_ref.to(torch.float32)
            for i in range(Q):
                a_ref_float[i, m_mask[i] :, :] = 0

            reference = torch.bmm(a_ref_float * a_scale_ref, b_ref * b_scale_ref).to(
                out_dtype
            )

        return a, b, a_scale, b_scale, m_mask, reference

    @pytest.mark.parametrize(
        "num_groups, max_m, expected_m_per_group, n, k",
        [
            (2, 512, 64, 256, 256),
            (4, 512, 128, 256, 256),
        ],
    )
    @pytest.mark.parametrize("block_scale_type", ["mxfp8", "nvfp4"])
    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [True])
    @pytest.mark.parametrize("out_dtype", [torch.bfloat16])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(
        self,
        num_groups,
        max_m,
        expected_m_per_group,
        n,
        k,
        block_scale_type,
        trans_a,
        trans_b,
        out_dtype,
        backend,
    ):
        """cuTile masked_scaled_bmm must match the dequantized torch.bmm reference."""
        if backend == "cutile" and not is_cuda_tile_available():
            pytest.skip("cuda.tile not available")
        if not HAS_MXFP:
            pytest.skip(
                "triton.tools.mxfp not available (needed for FP4/FP8 data prep)"
            )
        cc_major, cc_minor = get_compute_capability(torch.device("cuda:0"))
        cc_num = cc_major * 10 + cc_minor
        if not masked_scaled_bmm.is_backend_supported(backend, cc_num):
            pytest.skip(
                f"masked_scaled_bmm {backend} backend not supported on compute capability {cc_num}."
            )
        if block_scale_type == "nvfp4" and (cc_major, cc_minor) == (10, 3):
            pytest.skip("ct.unpack_from_bytes not available in cuTile on sm103")

        torch.manual_seed(0)
        random.seed(0)
        a, b, a_scale, b_scale, m_mask, ref_c = self.initialize_block_scaled(
            num_groups,
            max_m,
            expected_m_per_group,
            n,
            k,
            block_scale_type,
            True,
            trans_a,
            trans_b,
            out_dtype,
        )

        c = masked_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            m_mask,
            block_scale_type,
            max_m_device=None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )

        for i in range(num_groups):
            c[i, m_mask[i] :, :] = 0

        torch.testing.assert_close(ref_c, c, atol=1e-2, rtol=1e-2)
