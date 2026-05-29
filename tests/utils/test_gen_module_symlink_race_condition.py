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

import os
import tempfile
from pathlib import Path
from multiprocessing import Pool


def gen_fused_moe_worker_process(temp_dir):
    """
    Worker function that calls gen_trtllm_gen_fused_moe_sm100_module end-to-end.

    Each process will:
    1. Patch FLASHINFER_CUBIN_DIR in all modules to use the shared temp directory
    2. Call gen_trtllm_gen_fused_moe_sm100_module (downloads artifacts, creates symlinks)
    3. Verify the symlink is correct
    """
    from flashinfer.jit import env as jit_env
    from flashinfer.jit import cubin_loader

    jit_env.FLASHINFER_CUBIN_DIR = Path(temp_dir)
    cubin_loader.FLASHINFER_CUBIN_DIR = Path(temp_dir)

    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    gen_trtllm_gen_fused_moe_sm100_module()

    # Verify the symlink was created correctly.
    symlink_path = (
        Path(temp_dir)
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )
    assert symlink_path.is_symlink(), f"Expected {symlink_path} to be a symlink"

    # Verify we can read a header through the symlink
    headers = [
        p
        for p in symlink_path.iterdir()
        if p.is_file() and p.suffix in (".h", ".cuh", ".hpp")
    ]
    assert len(headers) > 0, f"No headers found through symlink at {symlink_path}"
    for header in headers:
        content = header.read_bytes()
        assert len(content) > 0, f"Header {header.name} is empty"

    return True


def test_gen_fused_moe_symlink_race_condition(num_iterations=100, num_processes=10):
    """
    End-to-end test for race conditions in gen_trtllm_gen_fused_moe_sm100_module.

    Multiple processes concurrently call the real gen_trtllm_gen_fused_moe_sm100_module,
    which downloads artifacts via get_artifact() and races on ensure_symlink().

    Uses a single shared temp directory across iterations so artifacts are cached
    after the first download. Between iterations, the symlink is deleted to
    re-trigger the race condition without re-downloading.

    Args:
        num_iterations: Number of times to repeat the test
        num_processes: Number of concurrent processes per iteration
    """
    import shutil
    import torch
    from flashinfer.utils import is_sm100a_supported, is_sm12x_supported

    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm12x_supported(device)):
        print("Skipping: gen_trtllm_gen_fused_moe_sm100_module requires SM100 or SM12x")
        return

    temp_dir = tempfile.mkdtemp(prefix="flashinfer_test_fused_moe_symlink_")
    symlink_path = (
        Path(temp_dir)
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )

    try:
        with Pool(processes=num_processes) as pool:
            for iteration in range(num_iterations):
                # Delete the symlink to re-trigger the race, but keep
                # downloaded artifacts cached in temp_dir.
                if symlink_path.is_symlink():
                    symlink_path.unlink()

                results = pool.map(
                    gen_fused_moe_worker_process,
                    [temp_dir] * num_processes,
                )

                assert all(results), (
                    f"Iteration {iteration + 1}/{num_iterations}: some processes failed"
                )

                if (iteration + 1) % 10 == 0 or iteration == 0:
                    print(
                        f"Iteration {iteration + 1}/{num_iterations}: "
                        f"{num_processes} processes all verified symlink successfully"
                    )

        print(
            f"\nAll gen_fused_moe symlink race tests passed: "
            f"{num_iterations} iterations × {num_processes} processes"
        )

    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    test_gen_fused_moe_symlink_race_condition()
