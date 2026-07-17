# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ragged_block_scaled_bmm_cutile kernel (ragged FP8 block-scaled batched matrix multiply)."""

import random

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.gemm import ragged_block_scaled_bmm
from flashinfer.utils import get_compute_capability


class Test_FlashInfer_RaggedBlockScaledBMM:
    @staticmethod
    def create_ragged_m_segments(num_groups, m, ELEM_PER_BYTE_A, alignment=16):
        """Create non-even M segments for ragged BMM.

        Args:
            num_groups: Number of groups/batches
            m: Average segment size
            ELEM_PER_BYTE_A: Elements per byte for A matrix
            alignment: Segment size alignment (default 16, use 128 for CuTile)
        """
        # Create random segment sizes that sum to approximately total_m
        total_m = num_groups * m
        segment_sizes = []
        num_items = alignment * ELEM_PER_BYTE_A

        # Generate random segment sizes
        for _ in range(num_groups - 1):
            # Random size between 0.5x and 1.5x expected size
            size = int(m * random.uniform(0.5, 1.5))
            size = (size // num_items) * num_items
            segment_sizes.append(size)

        # Last segment gets the remaining size
        remaining = total_m - sum(segment_sizes)
        assert remaining > 0 and remaining % num_items == 0
        segment_sizes.append(remaining)

        # Create segment offsets
        segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
        for i in range(num_groups):
            segment_offsets[i + 1] = segment_offsets[i] + segment_sizes[i]

        max_m = max(segment_sizes)

        return max_m, segment_offsets

    @staticmethod
    def create_aligned_m_segments(num_groups, m, block_m=128):
        """Create M segments aligned to BLOCK_M for CuTile.

        CuTile's tile-based indexing requires segment offsets to be
        multiples of BLOCK_M for correct operation.

        Args:
            num_groups: Number of groups/batches
            m: Segment size (should be multiple of block_m)
            block_m: Block size for M dimension (default 128)
        """
        # Ensure m is a multiple of block_m
        aligned_m = ((m + block_m - 1) // block_m) * block_m

        # Create even segment offsets (all segments same size)
        segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
        for i in range(num_groups):
            segment_offsets[i + 1] = segment_offsets[i] + aligned_m

        max_m = aligned_m

        return max_m, segment_offsets, aligned_m

    @staticmethod
    def reference(
        a_fp8,
        b_fp8,
        a_scale,
        b_scale,
        segment_offsets,
        block_n,
        block_k,
        trans_a=False,
        trans_b=True,
        out_dtype=torch.bfloat16,
    ):
        """Reference for ragged block-scaled (FP8) BMM.

        For the NT layout the kernel supports, the reference is flashinfer's own
        native grouped FP8 block-scaled GEMM (``group_gemm_fp8_nt_groupwise``,
        trtllm backend) with K-major scales — the same grouped FP8 block-scaled
        math — so the cuTile kernel is validated against a production flashinfer
        kernel rather than a fresh torch impl. Falls back to the torch dequant
        loop if the native op is unavailable on this device.
        """
        if (not trans_a) and trans_b:
            try:
                from flashinfer.gemm import group_gemm_fp8_nt_groupwise

                return group_gemm_fp8_nt_groupwise(
                    a_fp8,
                    b_fp8,
                    a_scale,
                    b_scale,
                    segment_offsets.to(torch.int32),
                    scale_granularity_mnk=(1, block_n, block_k),
                    scale_major_mode="K",
                    out_dtype=out_dtype,
                    backend="trtllm",
                )
            except Exception:
                pass  # fall through to torch reference below

        a = a_fp8.float()
        b = b_fp8.float()
        # Get dimensions
        total_m, K = a.shape
        Q, N, K_b = b.shape

        assert K_b == K, f"K dimensions must match: {K} != {K_b}"

        # Initialize output tensor
        c = torch.zeros((total_m, N), device=a.device, dtype=out_dtype)

        # Process each segment
        for q in range(Q):
            start_offset = segment_offsets[q].item()
            end_offset = segment_offsets[q + 1].item()
            segment_size = end_offset - start_offset
            assert segment_size > 0

            # Extract segment from flattened matrix a
            a_segment = a[start_offset:end_offset, :]  # Shape: [segment_size, K]
            a_scale_segment = a_scale[
                start_offset:end_offset, :
            ]  # Shape: [segment_size, k_tiles]

            b_segment = b[q, :, :]  # Shape: [N, K]
            b_scale_segment = b_scale[q, :, :]  # Shape: [n_tiles, k_tiles]

            # Expand block-level scales to match data dimensions
            # a_scale: [segment_size, k_tiles] -> [segment_size, K]
            a_scale_expanded = torch.repeat_interleave(a_scale_segment, block_k, dim=1)[
                :, :K
            ]

            # b_scale: [n_tiles, k_tiles] -> [N, K]
            b_scale_expanded = torch.repeat_interleave(b_scale_segment, block_n, dim=0)[
                :N, :
            ]
            b_scale_expanded = torch.repeat_interleave(
                b_scale_expanded, block_k, dim=1
            )[:, :K]

            # Compute matrix multiplication for this segment
            # (a * a_scale) @ (b * b_scale).T
            c_segment = torch.mm(
                a_segment * a_scale_expanded, (b_segment * b_scale_expanded).t()
            ).to(out_dtype)

            # Store the result in the output tensor
            c[start_offset:end_offset, :] = c_segment

        return c

    @staticmethod
    def prepare_data(
        num_groups,
        M,
        N,
        K,
        trans_a=False,
        trans_b=True,
        out_dtype=torch.bfloat16,
        use_aligned_segments=False,
    ):
        Q = num_groups
        assert not trans_a and trans_b, "Only NT layout is supported"
        device = torch.device("cuda")
        factor_for_scale = 1e-2
        block_n = 128
        block_k = 128
        fp8_info = torch.finfo(torch.float8_e4m3fn)
        fp8_max, fp8_min = fp8_info.max, fp8_info.min

        if use_aligned_segments:
            # CuTile requires segment offsets aligned to BLOCK_M (128)
            max_m, segment_offsets, aligned_m = (
                Test_FlashInfer_RaggedBlockScaledBMM.create_aligned_m_segments(
                    num_groups=num_groups,
                    m=M,
                    block_m=128,  # BLOCK_M for CuTile
                )
            )
            total_m = segment_offsets[-1].item()
        else:
            # Supports non-aligned segments
            max_m, segment_offsets = (
                Test_FlashInfer_RaggedBlockScaledBMM.create_ragged_m_segments(
                    num_groups=num_groups,
                    m=M,
                    ELEM_PER_BYTE_A=1,
                    alignment=16,
                )
            )
            total_m = segment_offsets[-1].item()
            assert total_m == num_groups * M

        A_fp32 = (
            (
                torch.rand(total_m, K, dtype=torch.float32, device=device).normal_(
                    mean=0.0, std=0.3
                )
                - 0.5
            )
            * 2
            * fp8_max
        )
        A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        B_fp32 = (
            (
                torch.rand(Q, N, K, dtype=torch.float32, device=device).normal_(
                    mean=0.0, std=0.3
                )
                - 0.5
            )
            * 2
            * fp8_max
        )
        B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

        n_tiles = (N + block_n - 1) // block_n
        k_tiles = (K + block_k - 1) // block_k

        As = (
            torch.rand(total_m, k_tiles, dtype=torch.float32, device=device)
            * factor_for_scale
        )
        Bs = (
            torch.rand(Q, n_tiles, k_tiles, dtype=torch.float32, device=device)
            * factor_for_scale
        )

        ref_c = Test_FlashInfer_RaggedBlockScaledBMM.reference(
            A_fp8,
            B_fp8,
            As,
            Bs,
            segment_offsets,
            block_n,
            block_k,
            trans_a,
            trans_b,
            out_dtype,
        )

        return A_fp8, B_fp8, As, Bs, ref_c, segment_offsets, max_m

    @pytest.mark.parametrize("num_groups", [4])
    @pytest.mark.parametrize("m", [128, 512])
    @pytest.mark.parametrize("n, k", [(2048, 2048)])
    @pytest.mark.parametrize(
        "dtype, out_dtype", [(torch.float8_e4m3fn, torch.bfloat16)]
    )
    @pytest.mark.parametrize("trans_a, trans_b", [(False, True)])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op(self, num_groups, m, n, k, dtype, out_dtype, trans_a, trans_b, backend):
        if torch.cuda.get_device_capability()[0] == 8 and "float8" in dtype.__repr__():
            pytest.skip("FP8 is not supported on sm80 (Ampere).")

        if backend == "cutile" and not is_cuda_tile_available():
            pytest.skip("cuda.tile not available")
        cc_num = get_compute_capability(torch.device("cuda:0"))[0] * 10
        if not ragged_block_scaled_bmm.is_backend_supported(backend, cc_num):
            pytest.skip(
                f"ragged_block_scaled_bmm {backend} backend not supported on compute capability {cc_num}."
            )

        torch.manual_seed(0)
        random.seed(0)

        (
            a,
            b,
            a_scale,
            b_scale,
            ref_c,
            segment_offsets,
            max_m,
        ) = self.prepare_data(
            num_groups,
            m,
            n,
            k,
            trans_a,
            trans_b,
            out_dtype,
            use_aligned_segments=True,
        )

        c = ragged_block_scaled_bmm(
            a,
            b,
            a_scale,
            b_scale,
            segment_offsets,
            max_m,
            max_m_device=None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )

        torch.testing.assert_close(ref_c, c, atol=1.0, rtol=1.0)
