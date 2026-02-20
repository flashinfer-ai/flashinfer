################################################################################
#
# Copyright (c) 2017 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
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
#
################################################################################

"""
Test for concat_mla_k kernel.

Validates that the FlashInfer CUDA kernel correctly concatenates k_nope and k_rope
tensors for MLA attention, matching a reference PyTorch implementation.

Tests cover:
- Multiple dtypes: float16, bfloat16, float8_e4m3fn
- Contiguous and non-contiguous (strided) inputs
- Various token counts including edge cases
"""

from typing import Sequence

import pytest
import torch

from flashinfer.concat_ops import concat_mla_k

# MLA configuration constants (must match kernel expectations)
NUM_LOCAL_HEADS = 128
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
K_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM


def reference_concat_mla_k(
    k: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor
) -> None:
    """Reference PyTorch implementation of concat_mla_k."""
    k[..., :QK_NOPE_HEAD_DIM] = k_nope
    k[..., QK_NOPE_HEAD_DIM:] = k_rope


@pytest.fixture(params=[torch.float16, torch.bfloat16, torch.float8_e4m3fn])
def dtype(request: pytest.FixtureRequest) -> torch.dtype:
    return request.param


@pytest.fixture(params=[1, 7, 128, 1024, 4096])
def num_tokens(request: pytest.FixtureRequest) -> int:
    return request.param


def _make_random_tensor(
    shape: Sequence[int], dtype: torch.dtype, device: str = "cuda"
) -> torch.Tensor:
    """Create a random tensor, handling FP8 which doesn't support randn."""
    if dtype == torch.float8_e4m3fn:
        # Generate in float16 then cast (fp8 doesn't support randn directly)
        return torch.randn(shape, dtype=torch.float16, device=device).to(dtype)
    return torch.randn(shape, dtype=dtype, device=device)


def _to_comparable(t: torch.Tensor) -> torch.Tensor:
    """Convert tensor to float32 for comparison (needed for fp8)."""
    return t.float()


class TestConcatMLAKContiguous:
    """Test concat_mla_k with contiguous inputs."""

    def test_correctness(self, dtype: torch.dtype, num_tokens: int) -> None:
        k_nope = _make_random_tensor(
            (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM), dtype
        )
        k_rope = _make_random_tensor((num_tokens, 1, QK_ROPE_HEAD_DIM), dtype)

        # FlashInfer kernel output
        k = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
        )
        concat_mla_k(k, k_nope, k_rope)

        # Reference output
        k_ref = torch.empty_like(k)
        reference_concat_mla_k(k_ref, k_nope, k_rope)

        torch.testing.assert_close(
            _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
        )


class TestConcatMLAKStrided:
    """Test concat_mla_k with non-contiguous (strided) inputs."""

    def test_strided_nope(self, dtype: torch.dtype, num_tokens: int) -> None:
        """k_nope is a slice of a larger tensor (non-contiguous)."""
        nope_container = _make_random_tensor(
            (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128), dtype
        )
        k_nope = nope_container[:, :, :QK_NOPE_HEAD_DIM]
        assert not k_nope.is_contiguous()

        k_rope = _make_random_tensor((num_tokens, 1, QK_ROPE_HEAD_DIM), dtype)

        k = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
        )
        concat_mla_k(k, k_nope, k_rope)

        k_ref = torch.empty_like(k)
        reference_concat_mla_k(k_ref, k_nope, k_rope)

        torch.testing.assert_close(
            _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
        )

    def test_strided_rope(self, dtype: torch.dtype, num_tokens: int) -> None:
        """k_rope is a slice of a larger tensor (non-contiguous)."""
        k_nope = _make_random_tensor(
            (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM), dtype
        )

        rope_container = _make_random_tensor(
            (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM), dtype
        )
        k_rope = rope_container[:, :, -QK_ROPE_HEAD_DIM:]
        # With num_tokens=1, the slice may still be contiguous (single-element leading dims),
        # so only assert for num_tokens > 1
        if num_tokens > 1:
            assert not k_rope.is_contiguous()

        k = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
        )
        concat_mla_k(k, k_nope, k_rope)

        k_ref = torch.empty_like(k)
        reference_concat_mla_k(k_ref, k_nope, k_rope)

        torch.testing.assert_close(
            _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
        )

    def test_strided_output(self, dtype: torch.dtype, num_tokens: int) -> None:
        """Output k is a slice of a larger tensor (non-contiguous token stride)."""
        k_nope = _make_random_tensor(
            (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM), dtype
        )
        k_rope = _make_random_tensor((num_tokens, 1, QK_ROPE_HEAD_DIM), dtype)

        # Create output with extra padding between heads
        k_container = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM + 64), dtype=dtype, device="cuda"
        )
        k = k_container[:, :, :K_HEAD_DIM]
        assert not k.is_contiguous()

        concat_mla_k(k, k_nope, k_rope)

        k_ref = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
        )
        reference_concat_mla_k(k_ref, k_nope, k_rope)

        torch.testing.assert_close(
            _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
        )

    def test_all_strided(self, dtype: torch.dtype, num_tokens: int) -> None:
        """All tensors are non-contiguous slices."""
        nope_container = _make_random_tensor(
            (num_tokens, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM + 128), dtype
        )
        k_nope = nope_container[:, :, :QK_NOPE_HEAD_DIM]

        rope_container = _make_random_tensor(
            (num_tokens, 1, 128 + QK_ROPE_HEAD_DIM), dtype
        )
        k_rope = rope_container[:, :, -QK_ROPE_HEAD_DIM:]

        k_container = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM + 64), dtype=dtype, device="cuda"
        )
        k = k_container[:, :, :K_HEAD_DIM]

        concat_mla_k(k, k_nope, k_rope)

        k_ref = torch.empty(
            (num_tokens, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
        )
        reference_concat_mla_k(k_ref, k_nope, k_rope)

        torch.testing.assert_close(
            _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
        )


class TestConcatMLAKEdgeCases:
    """Test edge cases."""

    def test_zero_tokens(self) -> None:
        """Zero token count should be a no-op."""
        dtype = torch.bfloat16
        k = torch.empty((0, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda")
        k_nope = torch.empty(
            (0, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM), dtype=dtype, device="cuda"
        )
        k_rope = torch.empty((0, 1, QK_ROPE_HEAD_DIM), dtype=dtype, device="cuda")
        # Should not crash
        concat_mla_k(k, k_nope, k_rope)

    def test_single_token(self) -> None:
        """Single token should work correctly for all dtypes."""
        for dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn]:
            k_nope = _make_random_tensor((1, NUM_LOCAL_HEADS, QK_NOPE_HEAD_DIM), dtype)
            k_rope = _make_random_tensor((1, 1, QK_ROPE_HEAD_DIM), dtype)
            k = torch.empty(
                (1, NUM_LOCAL_HEADS, K_HEAD_DIM), dtype=dtype, device="cuda"
            )
            concat_mla_k(k, k_nope, k_rope)

            k_ref = torch.empty_like(k)
            reference_concat_mla_k(k_ref, k_nope, k_rope)

            torch.testing.assert_close(
                _to_comparable(k), _to_comparable(k_ref), atol=0, rtol=0
            )
