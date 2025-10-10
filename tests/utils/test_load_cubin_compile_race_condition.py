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

import os
import tempfile
import pytest
from pathlib import Path
from multiprocessing import Pool


def worker_process(temp_dir):
    """
    Worker function that each process executes.

    Each process will:
    1. Set FLASHINFER_CUBIN_DIR environment variable
    2. Import and call get_cubin with the same target file
    3. Read the file from FLASHINFER_CUBIN_DIR
    4. Return the file content
    """
    # Set environment variable for this process
    os.environ["FLASHINFER_CUBIN_DIR"] = temp_dir

    # Import here to ensure FLASHINFER_CUBIN_DIR is set before module loads
    from flashinfer.artifacts import ArtifactPath

    # Define the target file - same for all processes
    include_path = f"{ArtifactPath.TRTLLM_GEN_BMM}/include"
    header_name = "flashinferMetaInfo"

    # Read the file from FLASHINFER_CUBIN_DIR
    # NOTE(Zihao): instead of using metainfo, we directly read from the file path,
    # that aligns with how we compile the kernel.
    file_path = Path(temp_dir) / include_path / f"{header_name}.h"
    with open(file_path, "rb") as f:
        content = f.read()

    return content


@pytest.mark.skip(reason="Incompatible with pytest due to multiprocessing usage.")
def test_load_cubin_race_condition(num_iterations, num_processes):
    """
    Test race condition when multiple processes concurrently call get_cubin
    for the same file.

    Test steps:
    1. Set up a temporary FLASHINFER_CUBIN_DIR
    2. Launch multiple processes
    3. Each process calls get_cubin for the same target file
    4. Each process reads the downloaded file
    5. Verify all processes read the same content
    6. Repeat multiple times to increase chance of detecting race conditions

    Args:
        num_iterations: Number of times to repeat the test
        num_processes: Number of concurrent processes per iteration
    """
    import shutil

    for iteration in range(num_iterations):
        # Create a temporary directory for FLASHINFER_CUBIN_DIR
        temp_dir = tempfile.mkdtemp(prefix="flashinfer_test_cubin_")

        try:
            # Launch multiple processes concurrently
            with Pool(processes=num_processes) as pool:
                results = pool.map(worker_process, [temp_dir] * num_processes)

            # Verify all processes read the same content
            assert len(results) == num_processes, (
                f"Expected {num_processes} results, got {len(results)}"
            )

            # All results should be identical
            first_content = results[0]
            for i, content in enumerate(results):
                assert content == first_content, (
                    f"Iteration {iteration + 1}/{num_iterations}, Process {i} read different content. "
                    f"Expected length {len(first_content)}, got {len(content)}"
                )

            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(
                    f"Iteration {iteration + 1}/{num_iterations}: {num_processes} processes all read the same content"
                )

        finally:
            # Clean up temporary directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    print(
        f"\nAll tests passed: {num_iterations} iterations × {num_processes} processes"
    )


if __name__ == "__main__":
    # NOTE(Zihao): do not use pytest to run this test
    test_load_cubin_race_condition(100, 10)
