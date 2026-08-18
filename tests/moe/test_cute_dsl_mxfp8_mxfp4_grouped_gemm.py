"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch
from torch.nn import functional as F

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.tllm_enums import (
    ActivationType,
    DEFAULT_SWIGLU_ALPHA,
    DEFAULT_SWIGLU_BETA,
    DEFAULT_SWIGLU_LIMIT,
)
from flashinfer.utils import is_sm100a_supported


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuTeDSL is not available"
)
# The marker is evaluated at import time, so short-circuit on CPU-only hosts:
# is_sm100a_supported queries the current device.
sm100_required = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_sm100a_supported(torch.device("cuda"))),
    reason="Requires an SM100-family GPU with CUDA 12.8+",
)


def _interleave_linear_and_gate(x: torch.Tensor, group_size: int = 64) -> torch.Tensor:
    """Convert logical [linear, gate] rows to the FC1 kernel's interleaving."""
    num_experts, rows, k = x.shape
    intermediate_size = rows // 2
    assert rows % (2 * group_size) == 0
    return (
        x.view(num_experts, 2, intermediate_size // group_size, group_size, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def _deinterleave_linear_and_gate(
    x: torch.Tensor, group_size: int = 64
) -> torch.Tensor:
    """Inverse of :func:`_interleave_linear_and_gate`."""
    num_experts, rows, k = x.shape
    intermediate_size = rows // 2
    assert rows % (2 * group_size) == 0
    return (
        x.view(num_experts, intermediate_size // group_size, 2, group_size, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def _unswizzle_mxfp8_scales_128x4(
    sf: torch.Tensor, rows: int, columns: int
) -> torch.Tensor:
    """Expose canonical 128x4 E8M0 storage as logical [row, K/32]."""
    sf_vec_size = 32
    num_m_tiles = (rows + 127) // 128
    num_k_tiles = (columns + sf_vec_size * 4 - 1) // (sf_vec_size * 4)
    swizzled = sf.reshape(num_m_tiles, num_k_tiles, 32, 4, 4)
    linear = swizzled.transpose(1, 3).reshape(num_m_tiles * 128, num_k_tiles * 4)
    return linear[:rows, : columns // sf_vec_size].contiguous()


def _e8m0_to_float(sf: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 bytes, including FlashInfer's zero-block code."""
    decoded = torch.ldexp(
        torch.ones_like(sf, dtype=torch.float32), sf.to(torch.int32) - 127
    )
    return torch.where(sf == 0, torch.zeros_like(decoded), decoded)


def _dequantize_mxfp8_linear(
    values: torch.Tensor, scales: torch.Tensor
) -> torch.Tensor:
    rows, columns = values.shape
    logical_scales = scales.reshape(rows, columns // 32)
    return (
        values.float().reshape(rows, columns // 32, 32)
        * _e8m0_to_float(logical_scales).unsqueeze(-1)
    ).reshape(rows, columns)


def _dequantize_mxfp8_swizzled(
    values: torch.Tensor, scales: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, columns = values.shape
    logical_scales = _unswizzle_mxfp8_scales_128x4(scales, rows, columns)
    dequantized = (
        values.float().reshape(rows, columns // 32, 32)
        * _e8m0_to_float(logical_scales).unsqueeze(-1)
    ).reshape(rows, columns)
    return dequantized, logical_scales


def _quantize_mxfp4_grouped(
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed weights, MMA-layout scales, and a dequantized reference."""
    from flashinfer import e2m1_and_ufp8sf_scale_to_float, fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    num_experts, rows, k = weights.shape
    flat = weights.reshape(num_experts * rows, k).contiguous()
    packed, swizzled_sf = fp4_quantize(
        flat,
        global_scale=torch.ones(1, dtype=torch.float32, device=weights.device),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=True,
    )
    dequantized = e2m1_and_ufp8sf_scale_to_float(
        packed.detach().cpu(),
        swizzled_sf.detach().cpu().view(torch.uint8).reshape(-1),
        torch.ones(1, dtype=torch.float32),
        sf_vec_size=32,
        ufp8_type=0,
        is_sf_swizzled_layout=True,
    ).to(weights.device)
    mma_sf = convert_sf_to_mma_layout(
        swizzled_sf,
        m=rows,
        k=k,
        num_groups=num_experts,
        sf_vec_size=32,
    )
    return (
        packed.view(num_experts, rows, k // 2),
        mma_sf,
        dequantized.view(num_experts, rows, k),
    )


def _assert_numerically_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    min_cosine: float,
    max_relative_l2: float,
) -> None:
    actual_f = actual.float().reshape(-1)
    expected_f = expected.float().reshape(-1)
    assert torch.isfinite(actual_f).all()
    assert torch.isfinite(expected_f).all()
    expected_norm = torch.linalg.vector_norm(expected_f).clamp_min(1e-6)
    relative_l2 = (
        torch.linalg.vector_norm(actual_f - expected_f) / expected_norm
    ).item()
    cosine = F.cosine_similarity(actual_f, expected_f, dim=0).item()
    assert cosine > min_cosine, (
        f"cosine similarity {cosine:.6f} is below {min_cosine:.6f}"
    )
    assert relative_l2 < max_relative_l2, (
        f"relative L2 error {relative_l2:.6f} exceeds {max_relative_l2:.6f}"
    )


@cute_dsl_available
class TestMxfp8Mxfp4CanImplement:
    """Compile-time contracts; these tests do not require a CUDA device."""

    def test_gemm1_dtype_and_unpack_alignment(self):
        import cutlass

        from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
            BlockScaledContiguousGatherGroupedGemmKernel,
        )

        def can_implement(*, k=256, a_dtype=None, b_dtype=None, sf_dtype=None, vec=32):
            return BlockScaledContiguousGatherGroupedGemmKernel.can_implement(
                a_dtype or cutlass.Float8E4M3FN,
                b_dtype or cutlass.Float4E2M1FN,
                sf_dtype or cutlass.Float8E8M0FNU,
                vec,
                cutlass.Float8E4M3FN,
                (128, 128),
                (1, 1),
                128,
                256,
                k,
                4,
                a_major="k",
                b_major="k",
                c_major="n",
            )

        assert can_implement()
        assert not can_implement(
            a_dtype=cutlass.Float4E2M1FN, b_dtype=cutlass.Float8E4M3FN
        )
        assert not can_implement(sf_dtype=cutlass.Float8E4M3FN)
        assert not can_implement(vec=16)
        assert not can_implement(k=64)

    def test_gemm2_dtype_and_unpack_alignment(self):
        import cutlass

        from flashinfer.fused_moe.cute_dsl.blackwell.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel,
        )

        def can_implement(
            *,
            k=128,
            n=256,
            mma_tiler_mn=(128, 128),
            a_dtype=None,
            b_dtype=None,
            sf_dtype=None,
            vec=32,
        ):
            cluster_shape_mn = (2 if mma_tiler_mn[0] == 256 else 1, 1)
            return (
                Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    a_dtype or cutlass.Float8E4M3FN,
                    b_dtype or cutlass.Float4E2M1FN,
                    sf_dtype or cutlass.Float8E8M0FNU,
                    vec,
                    cutlass.BFloat16,
                    cutlass.Float32,
                    mma_tiler_mn,
                    cluster_shape_mn,
                    128,
                    n,
                    k,
                    4,
                    a_major="k",
                    b_major="k",
                    out_major="n",
                )
            )

        assert can_implement()
        assert not can_implement(
            a_dtype=cutlass.Float4E2M1FN, b_dtype=cutlass.Float8E4M3FN
        )
        assert not can_implement(sf_dtype=cutlass.Float8E4M3FN)
        assert not can_implement(vec=16)
        assert not can_implement(k=64)
        for mma_tiler_mn, n in [
            ((128, 64), 128),
            ((128, 192), 384),
            ((256, 64), 128),
            ((256, 192), 384),
        ]:
            assert can_implement(mma_tiler_mn=mma_tiler_mn, n=n)

        # N must cover whole 128-element groups: the B scale-factor layout tiles
        # the N extent into 128x4 E8M0 atoms, so a partial group would
        # under-describe the weight scales.
        assert not can_implement(mma_tiler_mn=(128, 64), n=192)
        # A partial trailing N tile is allowed: the finalize epilogue clamps the
        # bulk-reduce size to the remaining columns.
        assert can_implement(mma_tiler_mn=(128, 192), n=256)

        def can_implement_nvfp4(*, n, mma_tiler_mn):
            return (
                Sm100BlockScaledContiguousGroupedGemmFinalizeFusionKernel.can_implement(
                    cutlass.Float4E2M1FN,
                    cutlass.Float4E2M1FN,
                    cutlass.Float8E4M3FN,
                    16,
                    cutlass.BFloat16,
                    cutlass.Float32,
                    mma_tiler_mn,
                    (1, 1),
                    128,
                    n,
                    128,
                    4,
                    a_major="k",
                    b_major="k",
                    out_major="n",
                )
            )

        # The predicated bulk-reduce contract is shared by NVFP4 and mixed mode.
        assert can_implement_nvfp4(n=384, mma_tiler_mn=(128, 192))
        assert can_implement_nvfp4(n=256, mma_tiler_mn=(128, 192))
        assert can_implement_nvfp4(n=128, mma_tiler_mn=(128, 256))


@cute_dsl_available
@sm100_required
class TestMxfp8Mxfp4TwoStageMoe:
    @pytest.mark.parametrize(
        "mma_tiler_mn,cluster_shape_mn,n",
        [
            pytest.param((128, 64), (1, 1), 128, id="1cta-n64"),
            pytest.param((128, 192), (1, 1), 384, id="1cta-n192"),
            pytest.param((256, 64), (2, 1), 128, id="2cta-n64"),
            pytest.param((256, 192), (2, 1), 384, id="2cta-n192"),
            # Partial trailing N tile: the finalize epilogue predicates the
            # bulk-reduce with valid_columns, so n need not divide tile N.
            pytest.param((128, 192), (1, 1), 256, id="1cta-n192-partial"),
        ],
    )
    def test_gemm2_finalize_tile_n_variants(self, mma_tiler_mn, cluster_shape_mn, n):
        """Exercise the N=64/192 SFB offset paths without running GEMM1."""
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4,
        )

        torch.manual_seed(20260703)
        device = torch.device("cuda")
        num_tokens = 17
        num_experts = 2
        topk = 2
        k = 128
        tile_size = mma_tiler_mn[0]
        permuted_m = num_experts * tile_size

        intermediate_bf16 = (
            torch.randn(permuted_m, k, dtype=torch.bfloat16, device=device) / 2
        )
        intermediate, intermediate_sf = mxfp8_quantize(
            intermediate_bf16, is_sf_swizzled_layout=True
        )
        intermediate_dequant, _ = _dequantize_mxfp8_swizzled(
            intermediate, intermediate_sf
        )

        permuted_idx_to_expanded_idx = torch.full(
            (permuted_m,), -1, dtype=torch.int32, device=device
        )
        expanded_token_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
        for expert_idx in range(num_experts):
            start = expert_idx * tile_size
            permuted_idx_to_expanded_idx[start : start + num_tokens] = (
                expanded_token_ids * topk + expert_idx
            )
        tile_idx_to_expert_idx = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        tile_idx_to_mn_limit = torch.tensor(
            [expert_idx * tile_size + num_tokens for expert_idx in range(num_experts)],
            dtype=torch.int32,
            device=device,
        )
        num_non_exiting_tiles = torch.tensor(
            [num_experts], dtype=torch.int32, device=device
        )
        token_final_scales = (
            torch.tensor([0.4, 0.6], dtype=torch.float32, device=device)
            .expand(num_tokens, topk)
            .contiguous()
        )

        weights = (
            torch.randn(num_experts, n, k, dtype=torch.bfloat16, device=device) / 4
        )
        weights_packed, weights_sf, weights_dequant = _quantize_mxfp4_grouped(weights)
        alpha = torch.ones(num_experts, dtype=torch.float32, device=device)

        output = blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4(
            a=intermediate,
            b=weights_packed,
            a_scale=intermediate_sf,
            b_scale=weights_sf,
            alpha=alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            enable_pdl=False,
        )

        valid_rows = torch.cat(
            [
                torch.arange(
                    expert_idx * tile_size,
                    expert_idx * tile_size + num_tokens,
                    device=device,
                )
                for expert_idx in range(num_experts)
            ]
        )
        expanded_idx = permuted_idx_to_expanded_idx[valid_rows].long()
        token_ids = torch.div(expanded_idx, topk, rounding_mode="floor")
        topk_ids = expanded_idx % topk
        expert_ids = topk_ids
        assert torch.equal(
            tile_idx_to_expert_idx[valid_rows // tile_size].long(), expert_ids
        )

        gemm2_reference = torch.bmm(
            weights_dequant[expert_ids],
            intermediate_dequant[valid_rows].unsqueeze(-1),
        ).squeeze(-1)
        gemm2_reference *= token_final_scales[token_ids, topk_ids].unsqueeze(-1)
        final_reference = torch.zeros(num_tokens, n, dtype=torch.float32, device=device)
        final_reference.index_add_(0, token_ids, gemm2_reference)

        assert output.shape == (num_tokens, n)
        assert output.dtype is torch.bfloat16
        _assert_numerically_close(
            output,
            final_reference.to(torch.bfloat16),
            min_cosine=0.97,
            max_relative_l2=0.25,
        )

    @pytest.mark.parametrize(
        "mma_tiler_mn,cluster_shape_mn,swiglu_alpha,swiglu_beta,swiglu_limit",
        [
            pytest.param(
                (128, 128),
                (1, 1),
                DEFAULT_SWIGLU_ALPHA,
                DEFAULT_SWIGLU_BETA,
                DEFAULT_SWIGLU_LIMIT,
                id="swiglu-tile128",
            ),
            pytest.param(
                (256, 128),
                (2, 1),
                1.702,
                1.0,
                7.0,
                id="swiglu-oai-tile256",
            ),
        ],
    )
    def test_gemm1_quantize_then_gemm2_finalize(
        self,
        mma_tiler_mn,
        cluster_shape_mn,
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
    ):
        from flashinfer import mxfp8_quantize
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_gather_grouped_gemm_act_fusion import (
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4,
        )
        from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (
            blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4,
        )

        torch.manual_seed(42)
        device = torch.device("cuda")
        num_tokens = 17
        num_experts = 2
        topk = 2
        hidden_size = 256
        intermediate_size = 128
        tile_size = mma_tiler_mn[0]

        x_bf16 = (
            torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
            / 2
        )
        x_mxfp8, x_sf = mxfp8_quantize(
            x_bf16, is_sf_swizzled_layout=False, alignment=256
        )
        x_sf = x_sf.reshape(num_tokens, hidden_size // 32)
        x_dequant = _dequantize_mxfp8_linear(x_mxfp8, x_sf)

        token_selected_experts = (
            torch.tensor([0, 1], dtype=torch.int32, device=device)
            .expand(num_tokens, topk)
            .contiguous()
        )
        token_final_scales = (
            torch.tensor([0.4, 0.6], dtype=torch.float32, device=device)
            .expand(num_tokens, topk)
            .contiguous()
        )

        # Construct the exact mapping contract produced by moe_sort without
        # pulling its unrelated C++ JIT into this focused CuTeDSL test. Each
        # token routes to both experts, and each expert occupies one padded tile.
        permuted_m = num_experts * tile_size
        permuted_idx_to_expanded_idx = torch.full(
            (permuted_m,), -1, dtype=torch.int32, device=device
        )
        expanded_token_ids = torch.arange(num_tokens, dtype=torch.int32, device=device)
        for expert_idx in range(num_experts):
            start = expert_idx * tile_size
            permuted_idx_to_expanded_idx[start : start + num_tokens] = (
                expanded_token_ids * topk + expert_idx
            )
        tile_idx_to_expert_idx = torch.arange(
            num_experts, dtype=torch.int32, device=device
        )
        tile_idx_to_mn_limit = torch.tensor(
            [expert_idx * tile_size + num_tokens for expert_idx in range(num_experts)],
            dtype=torch.int32,
            device=device,
        )
        num_non_exiting_tiles = torch.tensor(
            [num_experts], dtype=torch.int32, device=device
        )

        w1_logical = (
            torch.randn(
                num_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.bfloat16,
                device=device,
            )
            / 2
        )
        w1_interleaved = _interleave_linear_and_gate(w1_logical)
        w1_packed, w1_sf, w1_interleaved_dequant = _quantize_mxfp4_grouped(
            w1_interleaved
        )
        w1_dequant = _deinterleave_linear_and_gate(w1_interleaved_dequant)

        w2_logical = (
            torch.randn(
                num_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.bfloat16,
                device=device,
            )
            / 4
        )
        w2_packed, w2_sf, w2_dequant = _quantize_mxfp4_grouped(w2_logical)
        alpha = torch.ones(num_experts, dtype=torch.float32, device=device)

        intermediate, intermediate_sf = (
            blockscaled_contiguous_gather_grouped_gemm_act_fusion_mxfp8_mxfp4(
                a=x_mxfp8,
                b=w1_packed,
                a_scale=x_sf,
                b_scale=w1_sf,
                alpha=alpha,
                tile_idx_to_expert_idx=tile_idx_to_expert_idx,
                tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                token_id_mapping=permuted_idx_to_expanded_idx,
                num_non_exiting_tiles=num_non_exiting_tiles,
                topk=topk,
                mma_tiler_mn=mma_tiler_mn,
                cluster_shape_mn=cluster_shape_mn,
                enable_pdl=False,
                activation_type=ActivationType.Swiglu,
                swiglu_alpha=swiglu_alpha,
                swiglu_beta=swiglu_beta,
                swiglu_limit=swiglu_limit,
            )
        )

        assert permuted_m == permuted_idx_to_expanded_idx.numel()
        assert intermediate.shape == (permuted_m, intermediate_size)
        assert intermediate.dtype is torch.float8_e4m3fn
        assert intermediate_sf.dtype is torch.uint8
        assert intermediate_sf.shape == (
            32,
            4,
            permuted_m // 128,
            4,
            (intermediate_size + 127) // 128,
            1,
        )

        active_rows = int(num_non_exiting_tiles.item()) * tile_size
        row_ids = torch.arange(permuted_m, device=device)
        valid = (row_ids < active_rows) & (permuted_idx_to_expanded_idx >= 0)
        valid_rows = torch.nonzero(valid, as_tuple=False).squeeze(1)
        expanded_idx = permuted_idx_to_expanded_idx[valid_rows].long()
        token_ids = torch.div(expanded_idx, topk, rounding_mode="floor")
        topk_ids = expanded_idx % topk
        expert_ids = token_selected_experts[token_ids, topk_ids].long()
        assert torch.equal(
            tile_idx_to_expert_idx[valid_rows // tile_size].long(), expert_ids
        )

        intermediate_dequant, logical_intermediate_sf = _dequantize_mxfp8_swizzled(
            intermediate, intermediate_sf
        )
        assert (logical_intermediate_sf[valid_rows] != 0xFF).all()
        assert torch.isfinite(intermediate_dequant[valid_rows]).all()

        gemm1_reference = torch.bmm(
            w1_dequant[expert_ids], x_dequant[token_ids].unsqueeze(-1)
        ).squeeze(-1)
        linear = gemm1_reference[:, :intermediate_size]
        gate = gemm1_reference[:, intermediate_size:]
        gate = gate.clamp(max=swiglu_limit)
        linear = linear.clamp(min=-swiglu_limit, max=swiglu_limit)
        activation_reference = (
            gate * torch.sigmoid(swiglu_alpha * gate) * (linear + swiglu_beta)
        )
        activation_q, activation_sf = mxfp8_quantize(
            activation_reference.to(torch.bfloat16),
            is_sf_swizzled_layout=False,
        )
        activation_qdq = _dequantize_mxfp8_linear(activation_q, activation_sf)
        _assert_numerically_close(
            intermediate_dequant[valid_rows],
            activation_qdq,
            min_cosine=0.97,
            max_relative_l2=0.25,
        )

        actual_scale_codes = logical_intermediate_sf[valid_rows].to(torch.int16)
        reference_scale_codes = activation_sf.reshape(
            valid_rows.numel(), intermediate_size // 32
        ).to(torch.int16)
        scale_code_delta = (actual_scale_codes - reference_scale_codes).abs()
        assert (scale_code_delta <= 1).float().mean().item() >= 0.95

        output = blockscaled_contiguous_grouped_gemm_finalize_fusion_mxfp8_mxfp4(
            a=intermediate,
            b=w2_packed,
            a_scale=intermediate_sf,
            b_scale=w2_sf,
            alpha=alpha,
            tile_idx_to_expert_idx=tile_idx_to_expert_idx,
            num_non_exiting_tiles=num_non_exiting_tiles,
            tile_idx_to_mn_limit=tile_idx_to_mn_limit,
            permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
            token_final_scales=token_final_scales,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            enable_pdl=False,
        )
        assert output.shape == (num_tokens, hidden_size)
        assert output.dtype is torch.bfloat16
        assert torch.isfinite(output).all()

        gemm2_reference = torch.bmm(
            w2_dequant[expert_ids],
            intermediate_dequant[valid_rows].unsqueeze(-1),
        ).squeeze(-1)
        gemm2_reference *= token_final_scales[token_ids, topk_ids].unsqueeze(-1)
        final_reference = torch.zeros(
            num_tokens, hidden_size, dtype=torch.float32, device=device
        )
        final_reference.index_add_(0, token_ids, gemm2_reference)
        _assert_numerically_close(
            output,
            final_reference.to(torch.bfloat16),
            min_cosine=0.97,
            max_relative_l2=0.25,
        )


if __name__ == "__main__":
    pytest.main([__file__])
