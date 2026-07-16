# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ragged_bmm_cutile kernel (ragged batched matrix multiply)."""

import random

import pytest
import torch

from flashinfer.cutile.cutile_common import is_cuda_tile_available
from flashinfer.gemm import ragged_bmm
from flashinfer.utils import get_compute_capability


def _skip_if_backend_unavailable(backend):
    if backend == "cutile" and not is_cuda_tile_available():
        pytest.skip("cuda.tile not available")
    cc_num = get_compute_capability(torch.device("cuda:0"))[0] * 10
    if not ragged_bmm.is_backend_supported(backend, cc_num):
        pytest.skip(
            f"ragged_bmm {backend} backend not supported on compute capability {cc_num}."
        )


def create_ragged_m_segments(num_groups, m, dtype, align_to=None):
    """Create non-even M segments for ragged BMM

    Args:
        num_groups: Number of batches/groups
        m: Average segment size
        dtype: Data type
        align_to: If specified, align segment sizes to this value
    """
    total_m = num_groups * m
    segment_sizes = []
    itemsize = dtype.itemsize
    num_items = 16 // itemsize

    # Use align_to if specified, otherwise use default alignment
    alignment = align_to if align_to is not None else num_items

    # Generate random segment sizes
    for _ in range(num_groups - 1):
        size = int(m * random.uniform(0.5, 1.5))
        size = (size // alignment) * alignment
        if size < alignment:
            size = alignment
        segment_sizes.append(size)

    remaining = total_m - sum(segment_sizes)
    remaining = (remaining // alignment) * alignment
    if remaining < alignment:
        remaining = alignment
    segment_sizes.append(remaining)

    actual_total_m = sum(segment_sizes)

    segment_offsets = torch.zeros(num_groups + 1, dtype=torch.int32, device="cuda")
    for i in range(num_groups):
        segment_offsets[i + 1] = segment_offsets[i] + segment_sizes[i]

    max_m = max(segment_sizes)
    return max_m, segment_offsets, actual_total_m


class Test_FlashInfer_RaggedBMM:
    @staticmethod
    def reference(a, b, segment_offsets, trans_a=False, trans_b=True, out_dtype=None):
        """Reference for ragged BMM with non-even M segments.

        For fp16/bf16 inputs (the ``trans_a=False, trans_b=True`` case the kernel
        supports), the reference is flashinfer's own native grouped GEMM
        (``SegmentGEMMWrapper``, sm80/sm90 CUTLASS): ``ragged_bmm`` here computes
        ``y[i] = a[i] @ b[q].T`` with ``b = [Q, N, K]``, which is exactly
        SegmentGEMM with ``weight_column_major=True``. This validates the cuTile
        kernel against a production flashinfer kernel rather than a fresh torch impl.
        Falls back to the torch loop for fp8 inputs (no native fp8 grouped GEMM).
        """
        if out_dtype is None:
            out_dtype = a.dtype
        if (not trans_a) and trans_b and a.dtype in (torch.float16, torch.bfloat16):
            from flashinfer.gemm import SegmentGEMMWrapper

            num_groups = b.shape[0]
            ws = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=a.device)
            seg = SegmentGEMMWrapper(ws)
            return seg.run(
                a,
                b,
                batch_size=num_groups,
                weight_column_major=True,
                seg_indptr=segment_offsets.to(torch.int64),
            ).to(out_dtype)
        # Torch fallback: fp8 inputs, or non-standard transpose.
        if trans_a:
            a = torch.transpose(a, 0, 1)
        if trans_b:
            b = torch.transpose(b, 1, 2)

        total_m, K = a.shape
        Q, K_b, N = b.shape

        if out_dtype is None:
            out_dtype = a.dtype

        c = torch.zeros((total_m, N), device=a.device, dtype=out_dtype)

        for q in range(Q):
            start_offset = segment_offsets[q].item()
            end_offset = segment_offsets[q + 1].item()
            segment_size = end_offset - start_offset
            assert segment_size > 0
            a_segment = a[start_offset:end_offset, :]
            b_segment = b[q, :, :]
            c_segment = torch.mm(a_segment.to(out_dtype), b_segment.to(out_dtype))
            c[start_offset:end_offset, :] = c_segment

        return c

    @staticmethod
    def prepare_data(num_groups, m, n, k, trans_a, trans_b, dtype):
        device = torch.device("cuda")

        max_m, segment_offsets, actual_total_m = create_ragged_m_segments(
            num_groups=num_groups,
            m=m,
            dtype=dtype,
            align_to=128,
        )

        total_m = segment_offsets[-1].item()

        if trans_a:
            a_shape = (k, total_m)
        else:
            a_shape = (total_m, k)

        if trans_b:
            b_shape = (num_groups, n, k)
        else:
            b_shape = (num_groups, k, n)

        a = torch.rand(
            a_shape, device=device, dtype=torch.float16, requires_grad=False
        ).to(dtype)
        b = torch.rand(
            b_shape, device=device, dtype=torch.float16, requires_grad=False
        ).to(dtype)

        return a, b, max_m, segment_offsets

    @pytest.mark.parametrize("trans_a", [False])
    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize(
        "num_groups, m, n, k",
        [(4, 256, 256, 256), (2, 128, 128, 128), (4, 512, 512, 512)],
    )
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_shapes(self, trans_a, trans_b, dtype, num_groups, m, n, k, backend):
        if trans_a or not trans_b:
            pytest.skip("ragged_bmm only supports trans_a=False, trans_b=True")
        _skip_if_backend_unavailable(backend)

        torch.manual_seed(0)
        random.seed(0)
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(
            num_groups, m, n, k, trans_a, trans_b, dtype
        )

        result = ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )
        ref = self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("m, n, k", [(256, 256, 256)])
    @pytest.mark.parametrize("num_groups", [1, 4, 8])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_num_groups(self, dtype, m, n, k, num_groups, backend):
        _skip_if_backend_unavailable(backend)
        torch.manual_seed(0)
        random.seed(0)
        trans_a = False
        trans_b = True
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(
            num_groups, m, n, k, trans_a, trans_b, dtype
        )

        result = ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )
        ref = self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("num_groups, m, n, k", [(4, 256, 256, 256)])
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_dtypes(self, num_groups, m, n, k, dtype, backend):
        if torch.cuda.get_device_capability()[0] == 8 and "float8" in dtype.__repr__():
            pytest.skip("FP8 is not supported on sm80 (Ampere).")
        _skip_if_backend_unavailable(backend)

        torch.manual_seed(0)
        random.seed(0)
        trans_a = False
        trans_b = True
        out_dtype = torch.bfloat16
        a, b, max_m, segment_offsets = self.prepare_data(
            num_groups, m, n, k, trans_a, trans_b, dtype
        )

        result = ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )
        ref = self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.parametrize("dtype", [torch.bfloat16])
    @pytest.mark.parametrize("num_groups, m, n, k", [(4, 256, 256, 256)])
    @pytest.mark.parametrize("trans_a", [False, True])
    @pytest.mark.parametrize("trans_b", [False, True])
    @pytest.mark.parametrize("backend", ["cutile"])
    def test_op_transpose(self, dtype, num_groups, m, n, k, trans_a, trans_b, backend):
        if trans_a or not trans_b:
            pytest.skip("ragged_bmm only supports trans_a=False, trans_b=True")
        _skip_if_backend_unavailable(backend)

        torch.manual_seed(0)
        random.seed(0)
        out_dtype = dtype
        a, b, max_m, segment_offsets = self.prepare_data(
            num_groups, m, n, k, trans_a, trans_b, dtype
        )

        result = ragged_bmm(
            a,
            b,
            segment_offsets,
            max_m,
            None,
            transpose_a=trans_a,
            transpose_b=trans_b,
            out_dtype=out_dtype,
            backend=backend,
        )
        ref = self.reference(a, b, segment_offsets, trans_a, trans_b, out_dtype)
        torch.testing.assert_close(result, ref, atol=1e-2, rtol=1e-2)
