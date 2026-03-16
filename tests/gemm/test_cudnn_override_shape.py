"""
Tests for cuDNN GEMM operations using is_override_shape_enabled API.

A single cuDNN graph is compiled once with a "cache shape" (large M).
At execution time, the actual M is passed via override_shapes / override_strides,
so no graph rebuild is triggered for varying M values.

Requires:
  - CUDA compute capability == SM100 / SM103
  - cuDNN frontend >= 1.20 / backend_version >= 92100
"""

import pytest
import torch

from flashinfer.gemm.gemm_base import (
    CUDNN_AVAILABLE,
    build_cudnn_gemm_bf16_graph_override_shape,
    execute_cudnn_gemm_bf16_graph_override_shape,
    build_cudnn_fp4_gemm_graph_override_shape,
    execute_cudnn_fp4_gemm_graph_override_shape,
    build_cudnn_mxfp8_gemm_graph_override_shape,
    execute_cudnn_mxfp8_gemm_graph_override_shape,
    is_cudnn_override_shape_available,
    _calculate_block_scale_dims,
)
from flashinfer.utils import get_compute_capability


def _skip_if_no_cudnn():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN not available")


def _skip_if_override_shape_not_supported():
    if not CUDNN_AVAILABLE:
        pytest.skip("cuDNN not available")
    if not is_cudnn_override_shape_available():
        pytest.skip(
            "cuDNN override-shape requires higher version of cuDNN backend and frontend"
        )


def _skip_if_not_sm100():
    major, minor = get_compute_capability(torch.device("cuda"))
    if major * 10 + minor < 100:
        pytest.skip("override-shape GEMM requires SM100+ (Blackwell)")


# ============================================================================
# BF16 GEMM with override_shape
# ============================================================================


class TestCudnnBf16OverrideShape:
    """Single compiled plan handles multiple M dimensions for BF16 GEMM."""

    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_bf16_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100()

        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        in_dtype = torch.bfloat16
        out_dtype = torch.bfloat16

        # Build graph once with cache_m
        graph = build_cudnn_gemm_bf16_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            device=device,
            cache_m=cache_m,
            is_a_k_major=True,
            is_b_k_major=True,
        )

        workspace = torch.empty(
            graph.get_workspace_size(), dtype=torch.uint8, device=device
        )

        b = torch.randn(1, n, k, dtype=in_dtype, device=device).transpose(1, 2)

        for m in dynamic_ms:
            a = torch.randn(1, m, k, dtype=in_dtype, device=device)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            ref = torch.bmm(a.float(), b.float()).to(out_dtype)

            execute_cudnn_gemm_bf16_graph_override_shape(
                graph, a, b, out, workspace, tactic=0
            )
            torch.cuda.synchronize()

            assert torch.allclose(ref, out, rtol=5e-2, atol=5e-2), (
                f"BF16 override_shape failed for m={m}, n={n}, k={k}: "
                f"max_abs_err={(ref - out).abs().max().item():.4f}, "
                f"max_rel_err={((ref - out).abs() / (ref.abs() + 1e-8)).max().item():.4f}"
            )


# ============================================================================
# NVFP4 GEMM with override_shape
# ============================================================================


class TestCudnnNVFp4OverrideShape:
    """Single compiled plan handles multiple M dimensions for NVFP4 GEMM."""

    @pytest.mark.skipif(
        not CUDNN_AVAILABLE,
        reason="cuDNN not available",
    )
    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_nvfp4_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100()

        import cudnn

        if cudnn.backend_version() < 91002:
            pytest.skip("FP4 requires cuDNN backend >= 91002")
        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        block_size = 16
        out_dtype = torch.bfloat16

        # Compute block scale dims using cache_m
        _, block_scale_dim_n, block_scale_dim_k = _calculate_block_scale_dims(
            cache_m, n, k, block_size
        )

        # Build graph once with cache_m
        graph = build_cudnn_fp4_gemm_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            ab_type=cudnn.data_type.FP4_E2M1,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            block_size=block_size,
            device=device,
            alpha_is_not_none=False,
            use_nvfp4=True,
            cache_m=cache_m,
        )

        workspace = torch.empty(
            graph.get_workspace_size(),
            dtype=torch.uint8,
            device=device,
        )

        # FP4 E2M1 lookup table: index is the 4-bit pattern (0–15)
        # Encoding: sign(1) | exp(2) | mantissa(1), bias=1
        FP4_E2M1_LUT = torch.tensor(
            [
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
                3.0,
                4.0,
                6.0,
                -0.0,
                -0.5,
                -1.0,
                -1.5,
                -2.0,
                -3.0,
                -4.0,
                -6.0,
            ],
            dtype=torch.float32,
            device=device,
        )

        # B is fixed across all dynamic_ms
        b_packed = torch.randint(
            0, 256, (1, n, k // 2), dtype=torch.uint8, device=device
        ).transpose(1, 2)
        b_descale = torch.ones(
            1,
            block_scale_dim_n,
            block_scale_dim_k,
            dtype=torch.float8_e4m3fn,
            device=device,
        ).transpose(1, 2)

        for m in dynamic_ms:
            block_scale_dim_m, _, _ = _calculate_block_scale_dims(m, n, k, block_size)

            a_packed = torch.randint(
                0, 256, (1, m, k // 2), dtype=torch.uint8, device=device
            )
            a_descale = torch.ones(
                1,
                block_scale_dim_m,
                block_scale_dim_k,
                dtype=torch.float8_e4m3fn,
                device=device,
            )

            # Execute with cached graph (override_shape)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            execute_cudnn_fp4_gemm_graph_override_shape(
                graph,
                a_packed,
                b_packed,
                a_descale,
                b_descale,
                alpha=None,
                c_final=out,
                workspace_buffer=workspace,
                tactic=0,
            )
            torch.cuda.synchronize()

            # Correctness check: dequantize FP4 E2M1 via LUT and compare with
            # FP32 bmm reference. Descales are all 1.0, so no scaling needed.
            # A packing: a_packed (1, m, k//2), low nibble = even k, high = odd k
            a_fp32 = torch.empty(1, m, k, dtype=torch.float32, device=device)
            a_fp32[:, :, 0::2] = FP4_E2M1_LUT[(a_packed & 0x0F).long()]
            a_fp32[:, :, 1::2] = FP4_E2M1_LUT[((a_packed >> 4) & 0x0F).long()]
            # B packing: b_packed (1, k//2, n), low nibble = even k, high = odd k
            b_fp32 = torch.empty(1, k, n, dtype=torch.float32, device=device)
            b_fp32[:, 0::2, :] = FP4_E2M1_LUT[(b_packed & 0x0F).long()]
            b_fp32[:, 1::2, :] = FP4_E2M1_LUT[((b_packed >> 4) & 0x0F).long()]
            ref = torch.bmm(a_fp32, b_fp32).to(out_dtype)

            assert torch.allclose(ref, out, rtol=1e-1, atol=1.0), (
                f"NVFP4 override_shape failed for m={m}, n={n}, k={k}: "
                f"max_abs_err={(ref - out).abs().max().item():.4f}, "
                f"max_rel_err="
                f"{((ref - out).abs() / (ref.abs() + 1e-8)).max().item():.4f}"
            )


# ============================================================================
# MXFP8 GEMM with override_shape
# ============================================================================


class TestCudnnMXFp8OverrideShape:
    """Single compiled plan handles multiple M dimensions for MXFP8 GEMM."""

    @pytest.mark.skipif(
        not CUDNN_AVAILABLE,
        reason="cuDNN not available",
    )
    @pytest.mark.parametrize(
        "cache_m,dynamic_ms",
        [
            (2048, [1, 4, 16, 32, 64, 128, 512, 1024, 2048]),
            (4096, [1, 8, 64, 256, 1024, 4096]),
        ],
    )
    @pytest.mark.parametrize("n", [1024, 2048])
    @pytest.mark.parametrize("k", [1024, 2048])
    def test_mxfp8_override_shape_dynamic_m(self, cache_m, dynamic_ms, n, k):
        _skip_if_no_cudnn()
        _skip_if_override_shape_not_supported()
        _skip_if_not_sm100()

        import cudnn
        from flashinfer.gemm.gemm_base import _torch_data_type_to_cudnn_data_type

        device = torch.device("cuda")
        block_size = 32
        out_dtype = torch.bfloat16

        # Compute block scale dims using cache_m
        block_scale_dim_m_cache, block_scale_dim_n, block_scale_dim_k = (
            _calculate_block_scale_dims(cache_m, n, k, block_size)
        )

        # Build graph once with cache_m
        graph = build_cudnn_mxfp8_gemm_graph_override_shape(
            batch=1,
            n=n,
            k=k,
            a_type=cudnn.data_type.FP8_E4M3,
            b_type=cudnn.data_type.FP8_E4M3,
            o_type=_torch_data_type_to_cudnn_data_type(out_dtype),
            block_size=block_size,
            device=device,
            cache_m=cache_m,
        )

        workspace = torch.empty(
            graph.get_workspace_size(),
            dtype=torch.uint8,
            device=device,
        )

        # Use values 0–126 to avoid NaN FP8_E4M3 bit patterns (0x7F, 0xFF).
        b = torch.randint(
            0, 127, (1, n, k), dtype=torch.uint8, device=device
        ).transpose(1, 2)
        b_descale = torch.ones(
            1,
            block_scale_dim_n,
            block_scale_dim_k,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ).transpose(1, 2)

        for m in dynamic_ms:
            block_scale_dim_m, _, _ = _calculate_block_scale_dims(m, n, k, block_size)

            # Use values 0–126 to avoid NaN FP8_E4M3 bit patterns (0x7F, 0xFF).
            a = torch.randint(0, 127, (1, m, k), dtype=torch.uint8, device=device)
            a_descale = torch.ones(
                1,
                block_scale_dim_m,
                block_scale_dim_k,
                dtype=torch.float8_e8m0fnu,
                device=device,
            )

            # Execute with cached graph (override_shape)
            out = torch.empty(1, m, n, dtype=out_dtype, device=device)
            execute_cudnn_mxfp8_gemm_graph_override_shape(
                graph,
                a,
                b,
                a_descale,
                b_descale,
                c_final=out,
                workspace_buffer=workspace,
                tactic=0,
            )
            torch.cuda.synchronize()

            # Correctness check: reinterpret uint8 as FP8_E4M3, compute FP32
            # bmm reference. Descales are all 1.0 (2^0), so no scaling needed.
            # A: (1, m, k) contiguous uint8 → float8_e4m3fn → float32
            a_fp32 = a.view(torch.float8_e4m3fn).to(torch.float32)
            # B logical shape is (1, k, n) with stride [n*k, 1, k]; make
            # contiguous before view so dtype reinterpretation is valid.
            b_fp32 = b.contiguous().view(torch.float8_e4m3fn).to(torch.float32)
            ref = torch.bmm(a_fp32, b_fp32).to(out_dtype)

            assert torch.allclose(ref, out, rtol=5e-2, atol=5e-2), (
                f"MXFP8 override_shape failed for m={m}, n={n}, k={k}: "
                f"max_abs_err={(ref - out).abs().max().item():.4f}, "
                f"max_rel_err="
                f"{((ref - out).abs() / (ref.abs() + 1e-8)).max().item():.4f}"
            )
