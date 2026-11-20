# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ragged_scaled_bmm_cutile kernel (ragged FP8/FP4 block-scaled batched matrix multiply)."""

import random

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.gemm import ragged_scaled_bmm
from flashinfer.utils import get_compute_capability

try:
    from triton.tools.mxfp import MXFP4Tensor, MXScaleTensor

    HAS_MXFP = True
except ImportError:
    HAS_MXFP = False


def create_ragged_m_segments(num_groups, m, ELEM_PER_BYTE_A):
    """Create non-even, 128-aligned M segments for ragged BMM."""
    total_m = num_groups * m
    segment_sizes = []
    num_items = 16 * ELEM_PER_BYTE_A
    alignment_sf = 128

    for _ in range(num_groups - 1):
        size = int(m * random.uniform(0.5, 1.5))
        size = (size // num_items) * num_items
        size = (size // alignment_sf) * alignment_sf
        segment_sizes.append(size)

    remaining = total_m - sum(segment_sizes)
    assert (
        remaining > 0 and remaining % num_items == 0 and remaining % alignment_sf == 0
    )
    segment_sizes.append(remaining)

    segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        segment_offsets[i + 1] = segment_offsets[i] + segment_sizes[i]

    max_m = max(segment_sizes)
    return max_m, segment_offsets


class Test_FlashInfer_RaggedScaledBMM:
    """Correctness tests for ragged_scaled_bmm on the cuTile backend."""

    @staticmethod
    def reference(a, b, segment_offsets):
        """torch.mm reference over each ragged segment (NT layout, dequantized inputs)."""
        total_m, K = a.shape
        Q, K_b, N = b.shape

        c = torch.zeros((total_m, N), device=a.device, dtype=a.dtype)
        for q in range(Q):
            start_offset = segment_offsets[q].item()
            end_offset = segment_offsets[q + 1].item()
            assert end_offset - start_offset > 0
            a_segment = a[start_offset:end_offset, :]
            b_segment = b[q, :, :]
            c[start_offset:end_offset, :] = torch.mm(a_segment, b_segment)
        return c

    @staticmethod
    def initialize_block_scaled(
        num_groups,
        M,
        N,
        K,
        block_scale_type="nvfp4",
        compute_reference=False,
        trans_a=False,
        trans_b=True,
    ):
        """Build packed FP8/FP4 ragged A, batched B, their MX-swizzled block scales,
        the segment offsets, and (optionally) a dequantized reference.

        Mirrors the ocean validation data-prep: A is a ragged stack ``(total_m, K)``,
        B is batched ``(Q, N, K)``, and the scales are stored in the 5D
        TMA-descriptor layout the cuTile kernel expects.
        """
        VEC_SIZE = 16 if block_scale_type == "nvfp4" else 32
        assert block_scale_type in [
            "nvfp4",
            "mxfp4",
            "mxfp8",
            "mixed",
        ], f"Invalid block scale type: {block_scale_type}"

        Q = num_groups
        assert N % 8 == 0, "N must be divisible by 8"
        assert K % 128 == 0, "K must be divisible by 128"
        assert not trans_a and trans_b, "Only NT layout is supported"
        ELEM_PER_BYTE_A = 2 if "fp4" in block_scale_type else 1

        max_m, segment_offsets = create_ragged_m_segments(
            num_groups=num_groups,
            m=M,
            ELEM_PER_BYTE_A=ELEM_PER_BYTE_A,
        )
        total_m = segment_offsets[-1].item()
        assert total_m == num_groups * M

        device = "cuda"
        a_ref = MXFP4Tensor(size=(total_m, K), device=device).random()
        b_ref = MXFP4Tensor(size=(Q, N, K), device=device).random()
        if block_scale_type in ["mxfp8", "mixed"]:
            a_ref = a_ref.to(torch.float32)
            a = a_ref.to(torch.float8_e4m3fn)
        else:
            a = a_ref.to_packed_tensor(dim=1)

        if block_scale_type == "mxfp8":
            b_ref = b_ref.to(torch.float32)
            b = b_ref.to(torch.float8_e4m3fn)
        else:
            b = b_ref.to_packed_tensor(dim=2)

        b_ref = b_ref.to(torch.float32)
        b_ref = torch.transpose(b_ref, 1, 2)

        # Focus on NT layout, so k is the leading dimension of both A and B.
        a_scale_shape = [total_m // 128, K // VEC_SIZE // 4, 32, 16]
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

        a_scale = a_scale.reshape(a_scale_shape[0], a_scale.shape[1], 2, 256)
        b_scale = b_scale.reshape(Q, b_scale_shape[1], b_scale.shape[2], 2, 256)

        c = None
        if compute_reference:
            packed_a_scale_ref = (
                a_scale_ref.to(torch.float32)
                .reshape(a_scale_shape[0], a_scale.shape[1], 32, 4, 4)
                .permute(0, 3, 2, 1, 4)
                .reshape(a_scale_shape[0] * 128, a_scale.shape[1] * 4)
                .contiguous()
            )
            unpacked_a_scale_ref = packed_a_scale_ref.repeat_interleave(
                VEC_SIZE, dim=1
            ).contiguous()[:total_m, :K]

            packed_b_scale_ref = (
                b_scale_ref.to(torch.float32)
                .reshape(Q, b_scale_shape[1], b_scale.shape[2], 32, 4, 4)
                .permute(0, 1, 4, 3, 2, 5)
                .reshape(Q, b_scale_shape[1] * 128, b_scale.shape[2] * 4)
                .contiguous()
            )
            unpacked_b_scale_ref = (
                packed_b_scale_ref.repeat_interleave(VEC_SIZE, dim=2)
                .permute(0, 2, 1)
                .contiguous()[:Q, :K, :N]
            )
            a_ref_float = a_ref.to(torch.float32)

            c = Test_FlashInfer_RaggedScaledBMM.reference(
                a_ref_float * unpacked_a_scale_ref,
                b_ref * unpacked_b_scale_ref,
                segment_offsets,
            )

        return a, b, a_scale, b_scale, segment_offsets, max_m, c

    @pytest.mark.parametrize("num_groups", [3])
    @pytest.mark.parametrize("m", [512])
    @pytest.mark.parametrize("n", [2048])
    @pytest.mark.parametrize("k", [1024])
    @pytest.mark.parametrize("block_scale_type", ["mxfp8", "nvfp4"])
    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [True])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(
        self,
        num_groups,
        m,
        n,
        k,
        block_scale_type,
        trans_a,
        trans_b,
        backend,
    ):
        """cuTile ragged_scaled_bmm must match the dequantized per-segment torch.mm reference."""
        if backend == "cutile" and not is_cuda_tile_available():
            pytest.skip("cuda.tile not available")
        if not HAS_MXFP:
            pytest.skip(
                "triton.tools.mxfp not available (needed for FP4/FP8 data prep)"
            )
        cc_major, cc_minor = get_compute_capability(torch.device("cuda:0"))
        cc_num = cc_major * 10 + cc_minor
        if not ragged_scaled_bmm.is_backend_supported(backend, cc_num):
            pytest.skip(
                f"ragged_scaled_bmm {backend} backend not supported on compute capability {cc_num}."
            )
        if block_scale_type == "nvfp4" and (cc_major, cc_minor) == (10, 3):
            pytest.skip("ct.unpack_from_bytes not available in cuTile on sm103")

        torch.manual_seed(0)
        random.seed(0)
        (
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            ref_c,
        ) = self.initialize_block_scaled(
            num_groups,
            m,
            n,
            k,
            block_scale_type,
            True,
            trans_a,
            trans_b,
        )

        c = ragged_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            block_scale_type,
            transpose_a=trans_a,
            transpose_b=trans_b,
            backend=backend,
        )

        torch.testing.assert_close(ref_c, c.to(torch.float32), atol=1e-2, rtol=1e-2)
