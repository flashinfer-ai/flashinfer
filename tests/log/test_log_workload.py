"""
Test workload dumping functionality for FlashInfer API logging.

This test verifies that when FLASHINFER_BENCH_LOG is enabled, the decorator
correctly dumps tensor arguments to safetensors files.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
import torch

# Set environment variables BEFORE importing flashinfer
# This ensures the decorator picks up the settings at module load time
_TEST_BENCH_DIR = tempfile.mkdtemp(prefix="flashinfer_bench_test_")
os.environ["FLASHINFER_BENCH_LOG"] = "1"
os.environ["FLASHINFER_BENCH_LOG_DIR"] = _TEST_BENCH_DIR

import flashinfer


def generate_random_inputs(
    batch_size: int,
    max_seq_len: int,
    num_attention_heads: int = 32,
    num_key_value_heads: int = 4,
    head_dim: int = 128,
    page_size: int = 1,
    device: str = "cuda",
):
    """Generate random inputs for testing batch decode attention."""
    # Generate random sequence lengths for each batch
    seq_lens = torch.randint(
        1, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    # Calculate total pages needed (page_size = 1 means num_pages = total_tokens)
    total_pages_needed = seq_lens.sum().item()

    # Generate kv_indptr based on sequence lengths
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(seq_lens, dim=0)

    # Generate kv_indices (page indices for each sequence)
    kv_indices = torch.arange(total_pages_needed, dtype=torch.int32, device=device)

    # For page_size=1, last page always has 1 token
    kv_last_page_len = torch.ones(batch_size, dtype=torch.int32, device=device)

    # Generate query tensor
    q = torch.randn(
        batch_size, num_attention_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Generate K and V caches with extra pages
    num_pages = total_pages_needed + 100
    k_cache = torch.randn(
        num_pages,
        page_size,
        num_key_value_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_cache = torch.randn(
        num_pages,
        page_size,
        num_key_value_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )

    # Attention scale
    sm_scale = 1.0 / np.sqrt(head_dim)

    return {
        "q": q,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_len": kv_last_page_len,
        "sm_scale": sm_scale,
        "seq_lens": seq_lens,
    }


def get_dumped_safetensors_files(func_name: str) -> list:
    """Get list of safetensors files for a given function."""
    func_dir = os.path.join(_TEST_BENCH_DIR, func_name)
    if not os.path.exists(func_dir):
        return []
    return [
        os.path.join(func_dir, f)
        for f in os.listdir(func_dir)
        if f.endswith(".safetensors")
    ]


def load_safetensors(file_path: str) -> dict:
    """Load tensors from a safetensors file."""
    from safetensors.torch import load_file

    return load_file(file_path)


@pytest.fixture(scope="module", autouse=True)
def cleanup_bench_dir():
    """Clean up bench directory after all tests."""
    yield
    # Cleanup after tests
    if os.path.exists(_TEST_BENCH_DIR):
        shutil.rmtree(_TEST_BENCH_DIR)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_workload_dump_batch_decode():
    """Test that batch decode attention dumps workloads correctly."""
    # Skip if safetensors not installed
    pytest.importorskip("safetensors")

    device = "cuda"
    batch_size = 4
    max_seq_len = 32

    # Constants
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    page_size = 1

    # Generate inputs
    inputs = generate_random_inputs(
        batch_size,
        max_seq_len,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_size,
        device,
    )

    # Setup FlashInfer wrapper
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)

    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    # Plan the attention computation
    decode_wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"],
    )

    # Store original tensors for comparison
    original_q = inputs["q"].clone()

    # Run FlashInfer - this should trigger workload dump
    output, lse = decode_wrapper.run(
        inputs["q"],
        (inputs["k_cache"], inputs["v_cache"]),
        return_lse=True,
    )

    # Verify output is valid
    assert output is not None
    assert output.shape == (batch_size, num_attention_heads, head_dim)

    # Check that safetensors files were created
    # The run method is decorated, so look for BatchDecodeWithPagedKVCacheWrapper_run
    func_name = "BatchDecodeWithPagedKVCacheWrapper_run"
    safetensors_files = get_dumped_safetensors_files(func_name)

    assert len(safetensors_files) > 0, (
        f"No safetensors files found in {_TEST_BENCH_DIR}/{func_name}. "
        f"Directory contents: {os.listdir(_TEST_BENCH_DIR) if os.path.exists(_TEST_BENCH_DIR) else 'N/A'}"
    )

    # Load the most recent safetensors file
    latest_file = max(safetensors_files, key=os.path.getmtime)
    loaded_tensors = load_safetensors(latest_file)

    # Verify that the key tensors were dumped
    # The parameter names should match the function signature
    assert "q" in loaded_tensors, (
        f"'q' not found in dumped tensors: {list(loaded_tensors.keys())}"
    )

    # For paged_kv_cache which is a tuple, it should be dumped as paged_kv_cache_0 and paged_kv_cache_1
    assert "paged_kv_cache_0" in loaded_tensors or "k_cache" in loaded_tensors, (
        f"KV cache not found in dumped tensors: {list(loaded_tensors.keys())}"
    )

    # Verify tensor values match (compare on CPU)
    loaded_q = loaded_tensors["q"]
    original_q_cpu = original_q.cpu()
    assert torch.allclose(
        loaded_q.to(original_q_cpu.dtype), original_q_cpu, atol=1e-6
    ), "Dumped 'q' tensor does not match original"

    print("  - Safetensors file: ", latest_file)
    print("  - Dumped tensors: ", list(loaded_tensors.keys()))
    print("  - q shape: ", loaded_tensors["q"].shape)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_workload_dump_tensor_shapes():
    """Test that dumped tensors have correct shapes."""
    pytest.importorskip("safetensors")

    device = "cuda"
    batch_size = 2
    max_seq_len = 16
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    page_size = 1

    inputs = generate_random_inputs(
        batch_size,
        max_seq_len,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        page_size,
        device,
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    decode_wrapper.plan(
        indptr=inputs["kv_indptr"],
        indices=inputs["kv_indices"],
        last_page_len=inputs["kv_last_page_len"],
        num_qo_heads=num_attention_heads,
        num_kv_heads=num_key_value_heads,
        head_dim=head_dim,
        page_size=page_size,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        sm_scale=inputs["sm_scale"],
    )

    # Run to trigger dump
    _ = decode_wrapper.run(inputs["q"], (inputs["k_cache"], inputs["v_cache"]))

    # Check dumped files
    func_name = "BatchDecodeWithPagedKVCacheWrapper_run"
    safetensors_files = get_dumped_safetensors_files(func_name)

    assert len(safetensors_files) > 0, "No safetensors files created"

    latest_file = max(safetensors_files, key=os.path.getmtime)
    loaded_tensors = load_safetensors(latest_file)

    # Verify q shape
    if "q" in loaded_tensors:
        expected_q_shape = (batch_size, num_attention_heads, head_dim)
        assert loaded_tensors["q"].shape == expected_q_shape, (
            f"q shape mismatch: expected {expected_q_shape}, got {loaded_tensors['q'].shape}"
        )

    print("  - Shapes verified correctly")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_multiple_runs_create_multiple_files():
    """Test that multiple runs create multiple safetensors files."""
    pytest.importorskip("safetensors")

    device = "cuda"
    batch_size = 2
    max_seq_len = 8
    num_attention_heads = 32
    num_key_value_heads = 4
    head_dim = 128
    page_size = 1

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer,
        kv_layout="NHD",
    )

    func_name = "BatchDecodeWithPagedKVCacheWrapper_run"

    # Count existing files before test
    initial_files = set(get_dumped_safetensors_files(func_name))

    num_runs = 3
    for _ in range(num_runs):
        inputs = generate_random_inputs(
            batch_size,
            max_seq_len,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            page_size,
            device,
        )

        decode_wrapper.plan(
            indptr=inputs["kv_indptr"],
            indices=inputs["kv_indices"],
            last_page_len=inputs["kv_last_page_len"],
            num_qo_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads,
            head_dim=head_dim,
            page_size=page_size,
            pos_encoding_mode="NONE",
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            sm_scale=inputs["sm_scale"],
        )

        _ = decode_wrapper.run(inputs["q"], (inputs["k_cache"], inputs["v_cache"]))

    # Count files after runs
    final_files = set(get_dumped_safetensors_files(func_name))
    new_files = final_files - initial_files

    assert len(new_files) >= num_runs, (
        f"Expected at least {num_runs} new safetensors files, got {len(new_files)}"
    )

    print(f"  - Created {len(new_files)} new safetensors files from {num_runs} runs")


if __name__ == "__main__":
    # Run tests manually
    print("Running workload dump tests...")
    print(f"Bench log directory: {_TEST_BENCH_DIR}")

    try:
        test_workload_dump_batch_decode()
        test_workload_dump_tensor_shapes()
        test_multiple_runs_create_multiple_files()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    finally:
        # Cleanup
        if os.path.exists(_TEST_BENCH_DIR):
            print("Cleaning up ", _TEST_BENCH_DIR)
            shutil.rmtree(_TEST_BENCH_DIR)
