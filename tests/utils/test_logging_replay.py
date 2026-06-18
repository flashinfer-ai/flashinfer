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
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
        "FLASHINFER_DUMP_SAFETENSORS": os.environ.get("FLASHINFER_DUMP_SAFETENSORS"),
    }

    # Set up test environment
    dump_dir = tmp_path / "test_dumps"
    os.environ["FLASHINFER_LOGLEVEL"] = "10"
    os.environ["FLASHINFER_LOGDEST"] = "stdout"
    os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
    os.environ["FLASHINFER_DUMP_MAX_COUNT"] = "1000"
    os.environ["FLASHINFER_DUMP_MAX_SIZE_GB"] = "10"
    # Clear any existing filters and safetensors mode
    if "FLASHINFER_DUMP_INCLUDE" in os.environ:
        del os.environ["FLASHINFER_DUMP_INCLUDE"]
    if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
        del os.environ["FLASHINFER_DUMP_EXCLUDE"]
    if "FLASHINFER_DUMP_SAFETENSORS" in os.environ:
        del os.environ["FLASHINFER_DUMP_SAFETENSORS"]

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


# =============================================================================
# Tests for FLASHINFER_DUMP_INCLUDE / FLASHINFER_DUMP_EXCLUDE filtering
# =============================================================================


def test_dump_include_filter(tmp_path):
    """Test that FLASHINFER_DUMP_INCLUDE only dumps matching functions."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment with include filter for decode only
        dump_dir = tmp_path / "test_dumps_include"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_INCLUDE"] = "*decode*"
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode (should be dumped)
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Call prefill (should NOT be dumped due to include filter)
        qo_len = 64
        q_prefill = torch.randn(
            qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)

        # Check that only decode was dumped (filter for directories only, exclude session.jsonl)
        dump_subdirs = (
            [d for d in dump_dir.iterdir() if d.is_dir()] if dump_dir.exists() else []
        )
        assert len(dump_subdirs) == 1, f"Expected 1 dump, found {len(dump_subdirs)}"

        # Verify the dump is for decode
        dump_name = dump_subdirs[0].name
        assert "decode" in dump_name.lower(), f"Expected decode dump, got {dump_name}"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_dump_exclude_filter(tmp_path):
    """Test that FLASHINFER_DUMP_EXCLUDE skips matching functions."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment with exclude filter for prefill
        dump_dir = tmp_path / "test_dumps_exclude"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_EXCLUDE"] = "*prefill*"
        if "FLASHINFER_DUMP_INCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_INCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode (should be dumped)
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Call prefill (should NOT be dumped due to exclude filter)
        qo_len = 64
        q_prefill = torch.randn(
            qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)

        # Check that only decode was dumped (filter for directories only, exclude session.jsonl)
        dump_subdirs = (
            [d for d in dump_dir.iterdir() if d.is_dir()] if dump_dir.exists() else []
        )
        assert len(dump_subdirs) == 1, f"Expected 1 dump, found {len(dump_subdirs)}"

        # Verify the dump is for decode
        dump_name = dump_subdirs[0].name
        assert "decode" in dump_name.lower(), f"Expected decode dump, got {dump_name}"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_dump_include_and_exclude_combined(tmp_path):
    """Test that FLASHINFER_DUMP_INCLUDE and FLASHINFER_DUMP_EXCLUDE work together."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment: include all single_* APIs but exclude prefill
        dump_dir = tmp_path / "test_dumps_combined"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_INCLUDE"] = "single_*"
        os.environ["FLASHINFER_DUMP_EXCLUDE"] = "*prefill*"

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode (matches include, not excluded -> should be dumped)
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Call prefill (matches include BUT also matches exclude -> NOT dumped)
        qo_len = 64
        q_prefill = torch.randn(
            qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)

        # Check that only decode was dumped (filter for directories only, exclude session.jsonl)
        dump_subdirs = (
            [d for d in dump_dir.iterdir() if d.is_dir()] if dump_dir.exists() else []
        )
        assert len(dump_subdirs) == 1, f"Expected 1 dump, found {len(dump_subdirs)}"

        # Verify the dump is for decode
        dump_name = dump_subdirs[0].name
        assert "decode" in dump_name.lower(), f"Expected decode dump, got {dump_name}"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_dump_include_no_match(tmp_path):
    """Test that no dumps are created when include filter matches nothing."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment with include filter that matches nothing
        dump_dir = tmp_path / "test_dumps_nomatch"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_INCLUDE"] = "nonexistent_function_xyz"
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode (should NOT be dumped - doesn't match include filter)
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Check that no dumps were created (filter for directories only, exclude session.jsonl)
        dump_subdirs = (
            [d for d in dump_dir.iterdir() if d.is_dir()] if dump_dir.exists() else []
        )
        assert len(dump_subdirs) == 0, f"Expected 0 dumps, found {len(dump_subdirs)}"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_dump_multiple_include_patterns(tmp_path):
    """Test that multiple comma-separated include patterns work."""
    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment with multiple include patterns
        dump_dir = tmp_path / "test_dumps_multi"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_INCLUDE"] = "*decode*, *prefill*"
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode (should be dumped)
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Call prefill (should also be dumped)
        qo_len = 64
        q_prefill = torch.randn(
            qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v_prefill = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_prefill_with_kv_cache(q_prefill, k_prefill, v_prefill)

        # Check that both were dumped (filter for directories only, exclude session.jsonl)
        dump_subdirs = (
            [d for d in dump_dir.iterdir() if d.is_dir()] if dump_dir.exists() else []
        )
        assert len(dump_subdirs) == 2, f"Expected 2 dumps, found {len(dump_subdirs)}"

        # Verify we have one decode and one prefill
        dump_names = [d.name.lower() for d in dump_subdirs]
        has_decode = any("decode" in name for name in dump_names)
        has_prefill = any("prefill" in name for name in dump_names)
        assert has_decode, "Expected decode dump not found"
        assert has_prefill, "Expected prefill dump not found"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_jsonl_format(tmp_path):
    """Test that JSONL format is used correctly for metadata files."""
    import json

    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment
        dump_dir = tmp_path / "test_dumps_jsonl"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        if "FLASHINFER_DUMP_INCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_INCLUDE"]
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Verify dump directory was created
        assert dump_dir.exists(), "Dump directory was not created"

        # Find the dump subdirectory
        dump_subdirs = [d for d in dump_dir.iterdir() if d.is_dir()]
        assert len(dump_subdirs) == 1, (
            f"Expected 1 dump subdir, found {len(dump_subdirs)}"
        )
        dump_subdir = dump_subdirs[0]

        # Verify per-dump metadata.jsonl exists (not metadata.json)
        metadata_jsonl_path = dump_subdir / "metadata.jsonl"
        metadata_json_path = dump_subdir / "metadata.json"
        assert metadata_jsonl_path.exists(), "metadata.jsonl was not created"
        assert not metadata_json_path.exists(), "metadata.json should not exist"

        # Verify per-dump metadata.jsonl has 2 lines (inputs_saved + completed)
        with open(metadata_jsonl_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 2, (
            f"Expected 2 lines in metadata.jsonl, found {len(lines)}"
        )

        # Verify first line has execution_status="inputs_saved"
        first_record = json.loads(lines[0])
        assert first_record["execution_status"] == "inputs_saved"

        # Verify second line has execution_status="completed"
        second_record = json.loads(lines[1])
        assert second_record["execution_status"] == "completed"
        assert "output_metadata" in second_record
        assert second_record["tensor_info"]["output_tensor_keys"]

        # Verify central session.jsonl exists
        session_jsonl_path = dump_dir / "session.jsonl"
        assert session_jsonl_path.exists(), "session.jsonl was not created"

        # Verify session.jsonl has 2 lines (inputs_saved + completed for this call)
        with open(session_jsonl_path, "r") as f:
            session_lines = [line.strip() for line in f if line.strip()]
        assert len(session_lines) == 2, (
            f"Expected 2 lines in session.jsonl, found {len(session_lines)}"
        )

        # Verify session.jsonl records match per-dump records
        session_first = json.loads(session_lines[0])
        session_second = json.loads(session_lines[1])
        assert session_first["execution_status"] == "inputs_saved"
        assert session_second["execution_status"] == "completed"

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_session_jsonl_multiple_calls(tmp_path):
    """Test that session.jsonl accumulates records from multiple API calls."""
    import json

    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
    }

    try:
        # Set up environment
        dump_dir = tmp_path / "test_dumps_session"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        if "FLASHINFER_DUMP_INCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_INCLUDE"]
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache, single_prefill_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call 1: decode
        q1 = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k1 = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v1 = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q1, k1, v1)

        # Call 2: prefill
        qo_len = 64
        q2 = torch.randn(
            qo_len, num_qo_heads, head_dim, device="cuda", dtype=torch.float16
        )
        k2 = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v2 = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_prefill_with_kv_cache(q2, k2, v2)

        # Verify session.jsonl has 4 lines (2 per call: inputs_saved + completed)
        session_jsonl_path = dump_dir / "session.jsonl"
        assert session_jsonl_path.exists(), "session.jsonl was not created"

        with open(session_jsonl_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 4, f"Expected 4 lines in session.jsonl, found {len(lines)}"

        # Verify the structure: inputs_saved, completed, inputs_saved, completed
        records = [json.loads(line) for line in lines]
        assert records[0]["execution_status"] == "inputs_saved"
        assert records[1]["execution_status"] == "completed"
        assert records[2]["execution_status"] == "inputs_saved"
        assert records[3]["execution_status"] == "completed"

        # Verify we have both function names
        func_names = {r["function_name"] for r in records}
        assert "single_decode_with_kv_cache" in func_names
        assert "single_prefill_with_kv_cache" in func_names

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_safetensors_format(tmp_path):
    """Test that FLASHINFER_DUMP_SAFETENSORS uses safetensors format."""
    pytest.importorskip("safetensors", reason="safetensors package not installed")

    import json

    # Store original environment
    original_env = {
        "FLASHINFER_LOGLEVEL": os.environ.get("FLASHINFER_LOGLEVEL"),
        "FLASHINFER_LOGDEST": os.environ.get("FLASHINFER_LOGDEST"),
        "FLASHINFER_DUMP_DIR": os.environ.get("FLASHINFER_DUMP_DIR"),
        "FLASHINFER_DUMP_INCLUDE": os.environ.get("FLASHINFER_DUMP_INCLUDE"),
        "FLASHINFER_DUMP_EXCLUDE": os.environ.get("FLASHINFER_DUMP_EXCLUDE"),
        "FLASHINFER_DUMP_SAFETENSORS": os.environ.get("FLASHINFER_DUMP_SAFETENSORS"),
    }

    try:
        # Set up environment with safetensors enabled
        dump_dir = tmp_path / "test_dumps_safetensors"
        os.environ["FLASHINFER_LOGLEVEL"] = "10"
        os.environ["FLASHINFER_LOGDEST"] = "stdout"
        os.environ["FLASHINFER_DUMP_DIR"] = str(dump_dir)
        os.environ["FLASHINFER_DUMP_SAFETENSORS"] = "1"
        if "FLASHINFER_DUMP_INCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_INCLUDE"]
        if "FLASHINFER_DUMP_EXCLUDE" in os.environ:
            del os.environ["FLASHINFER_DUMP_EXCLUDE"]

        # Force reimport
        _clean_flashinfer_modules()

        from flashinfer import single_decode_with_kv_cache

        # Test configuration
        kv_len = 128
        num_qo_heads, num_kv_heads = 32, 8
        head_dim = 128

        # Call decode
        q = torch.randn(num_qo_heads, head_dim, device="cuda", dtype=torch.float16)
        k = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        v = torch.randn(
            kv_len, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
        )
        single_decode_with_kv_cache(q, k, v)

        # Verify dump directory was created
        assert dump_dir.exists(), "Dump directory was not created"

        # Find the dump subdirectory
        dump_subdirs = [d for d in dump_dir.iterdir() if d.is_dir()]
        assert len(dump_subdirs) == 1, (
            f"Expected 1 dump subdir, found {len(dump_subdirs)}"
        )
        dump_subdir = dump_subdirs[0]

        # Verify safetensors files exist (not .pt files)
        inputs_safetensors = dump_subdir / "inputs.safetensors"
        outputs_safetensors = dump_subdir / "outputs.safetensors"
        inputs_pt = dump_subdir / "inputs.pt"
        outputs_pt = dump_subdir / "outputs.pt"

        assert inputs_safetensors.exists(), "inputs.safetensors was not created"
        assert outputs_safetensors.exists(), "outputs.safetensors was not created"
        assert not inputs_pt.exists(), "inputs.pt should not exist in safetensors mode"
        assert not outputs_pt.exists(), (
            "outputs.pt should not exist in safetensors mode"
        )

        # Verify metadata has tensor_format field
        with open(dump_subdir / "metadata.jsonl", "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        last_record = json.loads(lines[-1])
        assert last_record.get("tensor_format") == "safetensors", (
            "tensor_format should be 'safetensors'"
        )

        # Verify replay works with safetensors format
        _clean_flashinfer_modules()

        from flashinfer.api_logging import replay_sequence

        results = replay_sequence(str(dump_dir), device="cuda")
        assert len(results) == 1
        assert results[0]["comparison_match"] is True

    finally:
        # Restore environment
        for key, value in original_env.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        _clean_flashinfer_modules()


def test_safetensors_replay_auto_detection(tmp_path):
    """Test that replay auto-detects safetensors format."""
    pytest.importorskip("safetensors", reason="safetensors package not installed")

    from safetensors.torch import save_file

    # Create a mock dump with safetensors files
    dump_subdir = tmp_path / "mock_dump"
    dump_subdir.mkdir()

    # Create mock input tensors
    input_tensors = {
        "arg_0": torch.randn(32, 128, dtype=torch.float16),
        "arg_1": torch.randn(128, 8, 128, dtype=torch.float16),
    }
    save_file(input_tensors, str(dump_subdir / "inputs.safetensors"))

    # Create mock metadata.jsonl
    import json

    metadata = {
        "function_name": "test_function",
        "module": "test_module",
        "call_sequence": 1,
        "timestamp": "20260108_120000_000",
        "process_id": os.getpid(),
        "input_metadata": {},
        "output_metadata": {},
        "tensor_info": {
            "input_tensor_keys": ["arg_0", "arg_1"],
            "output_tensor_keys": [],
            "input_size_bytes": 0,
            "input_size_mb": 0,
        },
        "tensor_format": "safetensors",
        "execution_status": "inputs_saved",
    }
    with open(dump_subdir / "metadata.jsonl", "w") as f:
        f.write(json.dumps(metadata) + "\n")

    # Force reimport to get clean state
    _clean_flashinfer_modules()

    from flashinfer.api_logging import replay_from_dump

    # Replay should auto-detect safetensors format
    result = replay_from_dump(
        str(dump_subdir), compare_outputs=False, device="cpu", run=False
    )

    # Verify tensors were loaded
    assert len(result["args"]) == 2
    assert result["args"][0].shape == (32, 128)
    assert result["args"][1].shape == (128, 8, 128)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
