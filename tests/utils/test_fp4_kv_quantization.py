"""
Copyright (c) 2025 by FlashInfer team.

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

import flashinfer

# E2M1 lookup table for reference dequantization
E2M1_LUT = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


def reference_dequant(fp4_data, block_scales, global_scale_val, output_dtype):
    """Pure PyTorch reference dequantization."""
    M, K_packed = fp4_data.shape
    K = K_packed * 2

    output = torch.zeros((M, K), dtype=torch.float32, device=fp4_data.device)

    fp4_np = fp4_data.cpu().numpy()
    scales_np = block_scales.cpu().numpy()

    for row in range(M):
        for col in range(0, K, 2):
            packed_byte = fp4_np[row, col // 2]
            fp4_lo = packed_byte & 0xF
            fp4_hi = (packed_byte >> 4) & 0xF

            scale_idx = col // 16
            scale_fp8_byte = scales_np[row, scale_idx]
            # Interpret byte as FP8 E4M3
            scale_val = torch.tensor(
                [scale_fp8_byte], dtype=torch.uint8
            ).view(torch.float8_e4m3fn).float().item()

            output[row, col] = E2M1_LUT[fp4_lo] * scale_val * global_scale_val
            output[row, col + 1] = E2M1_LUT[fp4_hi] * scale_val * global_scale_val

    return output.to(output_dtype).to(fp4_data.device)


def _make_global_scale(val, scale_mode):
    """Create global_scale in different forms based on scale_mode.

    Returns (global_scale, scale_on_host) tuple.
    - "device": tensor on CUDA, scale_on_host=False
    - "host": tensor on CPU, scale_on_host=True
    - "float": plain Python float, scale_on_host not needed (auto-detected)
    """
    if scale_mode == "device":
        return torch.tensor([val], dtype=torch.float32, device="cuda"), False
    elif scale_mode == "host":
        return torch.tensor([val], dtype=torch.float32, device="cpu"), True
    else:  # "float"
        return val, True  # scale_on_host is ignored when passing float, but set for clarity


SHAPES = [(128, 64), (256, 128), (1, 32), (2048, 2048)]
DTYPES = [torch.bfloat16, torch.float16]
SCALE_MODES = ["device", "host", "float"]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_mode", SCALE_MODES)
def test_nvfp4_kv_dequant(shape, dtype, scale_mode):
    """Test dequantization kernel against PyTorch reference."""
    M, K = shape
    torch.manual_seed(42)

    # Generate random FP4 packed data (each byte holds 2 FP4 values, 0-15 per nibble)
    fp4_data = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="cuda")

    # Generate random block scales as FP8 E4M3 bytes
    # Use values that are valid FP8 E4M3 (avoid NaN/Inf for stability)
    block_scales = torch.randint(1, 120, (M, K // 16), dtype=torch.uint8, device="cuda")

    global_scale_val = 0.5
    global_scale, scale_on_host = _make_global_scale(global_scale_val, scale_mode)

    # CUDA kernel output
    if scale_mode == "float":
        output = flashinfer.nvfp4_kv_dequantize(
            fp4_data, block_scales, global_scale, output_dtype=dtype
        )
    else:
        output = flashinfer.nvfp4_kv_dequantize(
            fp4_data, block_scales, global_scale, output_dtype=dtype,
            scale_on_host=scale_on_host,
        )

    # Reference output
    ref = reference_dequant(fp4_data, block_scales, global_scale_val, dtype)

    torch.testing.assert_close(output.float(), ref.float(), atol=1e-3, rtol=1e-3)


def get_compute_capability():
    props = torch.cuda.get_device_properties(0)
    return props.major * 10 + props.minor


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_mode", SCALE_MODES)
def test_nvfp4_kv_quant(shape, dtype, scale_mode):
    """Test quantization kernel output shapes and basic validity."""
    cc = get_compute_capability()
    if cc < 100:
        pytest.skip(f"SM{cc} does not support NVFP4 quantization (requires SM100+)")

    M, K = shape
    torch.manual_seed(42)

    input_data = torch.randn((M, K), dtype=dtype, device="cuda")
    # Use global_scale=1.0 to avoid FP8 E4M3 block scale underflow
    # (large global_scale causes 1/global_scale * vecMax/6 to underflow to FP8 zero)
    global_scale_val = 1.0
    global_scale, scale_on_host = _make_global_scale(global_scale_val, scale_mode)

    if scale_mode == "float":
        fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(input_data, global_scale)
    else:
        fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(
            input_data, global_scale, scale_on_host=scale_on_host
        )

    # Check shapes
    assert fp4_output.shape == (M, K // 2)
    assert fp4_output.dtype == torch.uint8
    assert block_scales.shape == (M, K // 16)
    assert block_scales.dtype == torch.uint8

    # Check that FP4 values are in valid range (each nibble 0-15)
    assert (fp4_output <= 255).all()


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_mode", SCALE_MODES)
def test_nvfp4_kv_roundtrip(shape, dtype, scale_mode):
    """Test quantize -> dequantize roundtrip error is within FP4 precision."""
    cc = get_compute_capability()
    if cc < 100:
        pytest.skip(f"SM{cc} does not support NVFP4 quantization (requires SM100+)")

    M, K = shape
    torch.manual_seed(42)

    input_data = torch.randn((M, K), dtype=dtype, device="cuda")
    # Use global_scale=1.0 to avoid FP8 E4M3 block scale underflow
    global_scale_val = 1.0
    global_scale, scale_on_host = _make_global_scale(global_scale_val, scale_mode)

    # Quantize
    if scale_mode == "float":
        fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(input_data, global_scale)
    else:
        fp4_output, block_scales = flashinfer.nvfp4_kv_quantize(
            input_data, global_scale, scale_on_host=scale_on_host
        )

    # Dequantize
    if scale_mode == "float":
        reconstructed = flashinfer.nvfp4_kv_dequantize(
            fp4_output, block_scales, global_scale, output_dtype=dtype
        )
    else:
        reconstructed = flashinfer.nvfp4_kv_dequantize(
            fp4_output, block_scales, global_scale, output_dtype=dtype,
            scale_on_host=scale_on_host,
        )

    # FP4 E2M1 has very limited precision (only 16 representable values),
    # so we check relative error with generous tolerance
    input_float = input_data.float()
    recon_float = reconstructed.float()

    # Compute per-element relative error where input is non-negligible
    mask = input_float.abs() > 1e-6
    if mask.any():
        rel_error = ((input_float[mask] - recon_float[mask]).abs() /
                     input_float[mask].abs().clamp(min=1e-6))
        # FP4 quantization can have up to ~50% relative error for some values,
        # but on average should be much better
        assert rel_error.mean() < 0.5, f"Mean relative error too high: {rel_error.mean():.4f}"

    # Also check that the overall cosine similarity is reasonable
    cos_sim = torch.nn.functional.cosine_similarity(
        input_float.flatten().unsqueeze(0),
        recon_float.flatten().unsqueeze(0),
    )
    assert cos_sim > 0.8, f"Cosine similarity too low: {cos_sim:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
