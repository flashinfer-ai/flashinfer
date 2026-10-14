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


def _bmm_export_symlink_path(gen_src_dir):
    """Location where the BMM export headers are symlinked for C++ includes."""
    return (
        Path(gen_src_dir)
        / "trtllm_export"
        / "fused_moe_trtllm_sm100"
        / "flashinfer"
        / "trtllm"
        / "batched_gemm"
        / "trtllmGen_bmm_export"
    )


def gen_fused_moe_worker_process(dirs):
    """
    Worker function that calls gen_trtllm_gen_fused_moe_sm100_module end-to-end.

    Each process will:
    1. Patch FLASHINFER_CUBIN_DIR (artifact download cache) and
       FLASHINFER_GEN_SRC_DIR (where the symlink is created) to use the shared
       temp directories
    2. Call gen_trtllm_gen_fused_moe_sm100_module (downloads artifacts, creates symlinks)
    3. Verify the symlink is correct
    """
    cubin_dir, gen_src_dir = dirs

    from flashinfer.jit import env as jit_env
    from flashinfer.jit import cubin_loader

    jit_env.FLASHINFER_CUBIN_DIR = Path(cubin_dir)
    cubin_loader.FLASHINFER_CUBIN_DIR = Path(cubin_dir)
    # The symlink is created under FLASHINFER_GEN_SRC_DIR, so it must be
    # redirected too or the test would race on the real workspace directory.
    jit_env.FLASHINFER_GEN_SRC_DIR = Path(gen_src_dir)

    from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    gen_trtllm_gen_fused_moe_sm100_module()

    # Verify the symlink was created correctly.
    symlink_path = _bmm_export_symlink_path(gen_src_dir)
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

    The artifact cache and the generated-source directory are kept as separate
    subdirectories so the assertion pins the symlink to FLASHINFER_GEN_SRC_DIR
    and would catch it moving back under FLASHINFER_CUBIN_DIR.

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
    cubin_dir = Path(temp_dir) / "cubins"
    gen_src_dir = Path(temp_dir) / "generated"
    cubin_dir.mkdir(parents=True, exist_ok=True)
    gen_src_dir.mkdir(parents=True, exist_ok=True)
    symlink_path = _bmm_export_symlink_path(gen_src_dir)

    try:
        with Pool(processes=num_processes) as pool:
            for iteration in range(num_iterations):
                # Delete the symlink to re-trigger the race, but keep
                # downloaded artifacts cached in cubin_dir.
                if symlink_path.is_symlink():
                    symlink_path.unlink()

                results = pool.map(
                    gen_fused_moe_worker_process,
                    [(cubin_dir, gen_src_dir)] * num_processes,
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
