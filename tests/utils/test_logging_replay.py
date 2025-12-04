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

"""
Tests for Level 10 API logging with actual FlashInfer APIs.

This test suite verifies that Level 10 logging (tensor dumping) works correctly
with all decorated FlashInfer APIs. For each API, we:
1. Run the API with Level 10 logging enabled
2. Verify a dump was created
3. Load the dump using replay_from_dump
4. Run the API again with replayed inputs
5. Verify: original_output ≈ dumped_output ≈ replayed_output
"""

import os
import sys

import pytest
import torch

from flashinfer.utils import get_compute_capability
from tests.utils_fp8 import to_float8


def _clean_flashinfer_modules():
    """Remove flashinfer modules from sys.modules."""
    modules_to_delete = [k for k in sys.modules.keys() if k.startswith("flashinfer")]
    for module in modules_to_delete:
        del sys.modules[module]


@pytest.fixture
def level10_environment(tmp_path):
    """Set up test environment and clean up after each test."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
    }

    # Set up test environment
    dump_dir = tmp_path / "test_dumps"
    os.environ["FLASHINFER_LOGLEVEL"] = "10"
    os.environ["FLASHINFER_LOGDEST"] = "stdout"
    os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
    os.environ["FLASHINFER_DUMP_MAX_COUNT"] = "1000"
    os.environ["FLASHINFER_DUMP_MAX_SIZE_GB"] = "10"

    # Force reimport to pick up new environment variables
    _clean_flashinfer_modules()

    yield dump_dir

    # Restore original environment
    for key, value in original_env.items():
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]

    # Force reimport
    _clean_flashinfer_modules()


def verify_and_replay_dump(
    dump_dir, original_output, func_to_replay, expected_dumps=1, dump_idx=0
):
    """Helper to verify dump creation and replay functionality."""
    from flashinfer.api_logging import replay_sequence

    # Replay sequence
    results = replay_sequence(str(dump_dir), device="cuda")

    # Filter results if checking for specific function
    if func_to_replay:
        target_name = func_to_replay.__name__
        results = [
            res
            for res in results
            if res.get("metadata", {}).get("function_name") == target_name
        ]

    assert len(results) >= expected_dumps, (
        f"Expected at least {expected_dumps} dumps, found {len(results)}"
    )

    # Get the target result (usually the last one if multiple)
    if dump_idx == -1:
        result = results[-1]
    else:
        result = results[dump_idx]

    # Verify comparison passed
    assert result["comparison_match"] is True

    # Verify execution result matches original (in-memory check)
    execution_result = result["execution_result"]
    assert torch.allclose(original_output, execution_result, atol=1e-5, rtol=1e-3)


def test_replay_sequence(level10_environment):
    """Test replaying a sequence of calls."""
    from flashinfer import single_decode_with_kv_cache
    from flashinfer.api_logging import replay_sequence

    # Generate two calls
    kv_len = 128
    num_qo_heads, num_kv_heads = 32, 8
    head_dim = 128

    # Call 1
    q1 = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    k1 = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v1 = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    _ = single_decode_with_kv_cache(q1, k1, v1)

    # Call 2
    q2 = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    k2 = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v2 = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    _ = single_decode_with_kv_cache(q2, k2, v2)

    # Replay sequence
    results = replay_sequence(str(level10_environment), device="cuda")

    assert len(results) == 2
    for res in results:
        assert res["comparison_match"] is True
        assert "execution_result" in res


def test_mm_fp8_replay(level10_environment):
    """Test Level 10 logging with mm_fp8 API."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] not in [10]:
        pytest.skip("mm_fp8 is only supported on Blackwell GPUs.")

    from flashinfer import mm_fp8, autotune, prepare_low_latency_gemm_weights

    # Test configuration
    m, n, k = 4, 2560, 8192
    input_dtype = torch.float8_e4m3fn
    mat2_dtype = torch.float8_e4m3fn

    # Create inputs
    torch.manual_seed(42)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    global_scale = input_inv_s * mat2_inv_s

    _cache_permute_indices = {}
    prepared_weights = prepare_low_latency_gemm_weights(
        mat2_fp8, _cache_permute_indices
    )

    # Run API with Level 10 logging (without pre-allocated output)
    with autotune():
        original_output = mm_fp8(
            input_fp8,
            prepared_weights,
            global_scale,
        )

    verify_and_replay_dump(level10_environment, original_output, mm_fp8)


def test_bmm_fp8_replay(level10_environment):
    """Test Level 10 logging with bmm_fp8 API."""

    from flashinfer import bmm_fp8, autotune

    # Test configuration
    b, m, n, k = 1, 48, 80, 64
    input_dtype = torch.float8_e4m3fn
    mat2_dtype = torch.float8_e4m3fn
    res_dtype = torch.bfloat16
    backend = "cudnn"

    # Create inputs
    torch.manual_seed(42)
    input = torch.randn([b, m, k], device="cuda", dtype=torch.bfloat16)
    input_fp8, input_inv_s = to_float8(input, dtype=input_dtype)

    mat2 = torch.randn([b, n, k], device="cuda", dtype=torch.bfloat16).transpose(-2, -1)
    mat2_fp8, mat2_inv_s = to_float8(mat2, dtype=mat2_dtype)

    # Run API with Level 10 logging
    with autotune():
        original_output = bmm_fp8(
            input_fp8,
            mat2_fp8,
            input_inv_s,
            mat2_inv_s,
            res_dtype,
            backend=backend,
        )

    verify_and_replay_dump(level10_environment, original_output, bmm_fp8)


def test_mm_fp4_replay(level10_environment):
    """Test Level 10 logging with mm_fp4 API."""
    compute_capability = get_compute_capability(torch.device("cuda"))
    compute_capability_number = compute_capability[0] * 10 + compute_capability[1]

    from flashinfer import mm_fp4, autotune, nvfp4_quantize, SfLayout

    backend = "cudnn"
    if not mm_fp4.is_backend_supported(backend, compute_capability_number):
        pytest.skip(
            f"Skipping test for {backend} because it is not supported on compute capability {compute_capability_number}."
        )

    # Test configuration
    m, n, k = 48, 128, 128
    res_dtype = torch.bfloat16
    use_128x4_sf_layout = True

    # Create inputs
    torch.manual_seed(42)
    input = torch.randn([m, k], device="cuda", dtype=torch.bfloat16)
    mat2 = torch.randn([n, k], device="cuda", dtype=torch.bfloat16)

    a_sf_layout = SfLayout.layout_128x4 if use_128x4_sf_layout else SfLayout.layout_8x4
    global_sf_input = (448 * 6) / input.float().abs().nan_to_num().max()
    global_sf_mat2 = (448 * 6) / mat2.float().abs().nan_to_num().max()

    input_fp4, input_inv_s = nvfp4_quantize(
        input, global_sf_input, sfLayout=a_sf_layout, do_shuffle=False
    )
    mat2_fp4, mat2_inv_s = nvfp4_quantize(
        mat2, global_sf_mat2, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )

    alpha = 1.0 / (global_sf_input * global_sf_mat2)
    block_size = 16

    # Run API with Level 10 logging (without pre-allocated output)
    with autotune():
        original_output = mm_fp4(
            input_fp4,
            mat2_fp4.T,
            input_inv_s,
            mat2_inv_s.T,
            alpha,
            res_dtype,
            block_size=block_size,
            use_8x4_sf_layout=not use_128x4_sf_layout,
            backend=backend,
            use_nvfp4=True,
            skip_check=False,
        )

    verify_and_replay_dump(level10_environment, original_output, mm_fp4)


def test_single_prefill_with_kv_cache_replay(level10_environment):
    """Test Level 10 logging with single_prefill_with_kv_cache API."""
    from flashinfer import single_prefill_with_kv_cache

    # Test configuration
    qo_len, kv_len = 127, 501
    num_qo_heads, num_kv_heads = 4, 1
    head_dim = 128
    causal = False

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    # Run API with Level 10 logging
    o = single_prefill_with_kv_cache(q, k, v, causal=causal)

    original_output = o.clone()

    verify_and_replay_dump(
        level10_environment, original_output, single_prefill_with_kv_cache
    )


def test_single_decode_with_kv_cache_replay(level10_environment):
    """Test Level 10 logging with single_decode_with_kv_cache API."""
    from flashinfer import single_decode_with_kv_cache

    # Test configuration
    kv_len = 1024
    num_qo_heads, num_kv_heads = 32, 8
    head_dim = 128

    # Create inputs
    torch.manual_seed(42)
    q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)

    # Run API with Level 10 logging
    o = single_decode_with_kv_cache(q, k, v)

    original_output = o.clone()

    verify_and_replay_dump(
        level10_environment, original_output, single_decode_with_kv_cache
    )


def test_cli_replay(level10_environment):
    """Test the CLI replay command."""
    from click.testing import CliRunner
    from flashinfer.__main__ import cli
    from flashinfer import single_decode_with_kv_cache

    # Create some data
    kv_len = 128
    num_qo_heads, num_kv_heads = 32, 8
    head_dim = 128
    q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16)
    single_decode_with_kv_cache(q, k, v)

    runner = CliRunner()
    result = runner.invoke(cli, ["replay", "--dir", str(level10_environment)])

    assert result.exit_code == 0
    assert "Replaying session from" in result.output
    assert "Passed" in result.output
    assert "Summary: 1 passed" in result.output


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
