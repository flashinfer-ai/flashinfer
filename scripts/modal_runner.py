#!/usr/bin/env python3
"""
Modal runner script for FlashInfer testing and benchmarking.

Usage:
    # Run tests (default B200)
    modal run scripts/modal_runner.py --command "pytest tests/utils/test_norm.py -v"

    # Specify GPU type
    modal run scripts/modal_runner.py --gpu H100 --command "pytest tests/attention/test_hopper.py -v"

Requirements:
    pip install modal
    python3 -m modal setup
"""

import sys

SUPPORTED_GPUS = ["B200", "H100", "A100", "A10G", "L4", "T4"]

# Parse --gpu argument before importing modal (to set GPU_TYPE correctly)
GPU_TYPE = "B200"  # default
for i, arg in enumerate(sys.argv):
    if arg == "--gpu" and i + 1 < len(sys.argv):
        GPU_TYPE = sys.argv[i + 1].upper()
        break

import modal

# Use the official FlashInfer CI image from DockerHub, with local source mounted
image = (
    modal.Image.from_registry(
        "flashinfer/flashinfer-ci-cu130",
        add_python=None,  # Image already has Python
    )
    .env({"FLASHINFER_WORKSPACE_BASE": "/cache/flashinfer"})
    .add_local_dir(
        ".",
        remote_path="/workspace/flashinfer",
        ignore=[
            ".git",
            "__pycache__",
            ".pytest_cache",
            "*.egg-info",
            "build",
            "dist",
            ".cache",
            "*.pyc",
            "*.so",
        ],
    )
)

app = modal.App("flashinfer-runner")

# Persistent volume for JIT cache
jit_cache = modal.Volume.from_name("flashinfer-jit-cache", create_if_missing=True)


def _run_flashinfer_command(command: str) -> str:
    """
    Run a command in the FlashInfer environment.
    """
    import os
    import subprocess

    os.chdir("/workspace/flashinfer")

    # Append to PYTHONPATH
    pythonpath = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = (
        f"/workspace/flashinfer:{pythonpath}" if pythonpath else "/workspace/flashinfer"
    )

    # Generate _build_meta.py for correct version
    build_meta_path = "/workspace/flashinfer/flashinfer/_build_meta.py"
    if not os.path.exists(build_meta_path):
        version = "0.0.0+unknown"
        version_file = "/workspace/flashinfer/version.txt"
        if os.path.exists(version_file):
            with open(version_file) as f:
                version = f.read().strip()
        with open(build_meta_path, "w") as f:
            f.write('"""Build metadata for flashinfer package."""\n')
            f.write(f'__version__ = "{version}"\n')
            f.write('__git_version__ = "modal"\n')
        print(f"=== Generated _build_meta.py with version {version} ===")

    # Initialize submodules (since .git is not fully mounted)
    if not os.path.exists("3rdparty/cutlass/include"):
        print("=== Downloading CUTLASS ===")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/NVIDIA/cutlass.git",
                "3rdparty/cutlass",
            ],
            check=True,
        )

    if not os.path.exists("3rdparty/spdlog/include"):
        print("=== Downloading spdlog ===")
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/gabime/spdlog.git",
                "3rdparty/spdlog",
            ],
            check=True,
        )

    # Run the user command
    print(f"=== Running command: {command} ===")
    import shlex

    result = subprocess.run(
        shlex.split(command),
        text=True,
        capture_output=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Commit JIT cache to persistent volume
    print("=== Committing JIT cache ===")
    jit_cache.commit()

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")

    return "Command completed successfully"


# Only create function for the selected GPU type
@app.function(
    image=image,
    gpu=GPU_TYPE,
    timeout=3600,
    memory=32768,
    volumes={"/cache/flashinfer": jit_cache},
)
def run_gpu(command: str) -> str:
    return _run_flashinfer_command(command)


@app.local_entrypoint()
def main(
    command: str = "pytest tests/utils/test_norm.py -v",
    gpu: str = "B200",
):
    """
    Main entrypoint for running FlashInfer commands on Modal.

    Args:
        command: Shell command to run
        gpu: GPU type (B200, H100, A100, A10G, L4, T4)
    """
    gpu = gpu.upper()
    if gpu not in SUPPORTED_GPUS:
        raise ValueError(f"Unknown GPU type: {gpu}. Available: {SUPPORTED_GPUS}")

    print(f"GPU: {gpu}")
    print(f"Command: {command}")

    result = run_gpu.remote(command)
    print(result)
