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
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests  # type: ignore[import-untyped]
import shutil

from .jit.core import logger
from .jit.cubin_loader import (
    FLASHINFER_CUBINS_REPOSITORY,
    get_cubin,
    FLASHINFER_CUBIN_DIR,
)


import logging
from contextlib import contextmanager


@contextmanager
def temp_env_var(key, value):
    old_value = os.environ.get(key, None)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


@contextmanager
def patch_logger_for_tqdm(logger):
    """
    Context manager to patch the logger so that log messages are displayed using tqdm.write,
    preventing interference with tqdm progress bars.
    """
    import tqdm

    class TqdmLoggingHandler(logging.Handler):
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.write(msg, end="\n")
            except Exception:
                self.handleError(record)

    # Save original handlers and level
    original_handlers = logger.handlers[:]
    original_level = logger.level

    # Remove all existing handlers to prevent duplicate output
    for h in original_handlers:
        logger.removeHandler(h)

    # Add our tqdm-aware handler
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        # Remove tqdm handler and restore original handlers and level
        logger.removeHandler(handler)
        for h in original_handlers:
            logger.addHandler(h)
        logger.setLevel(original_level)


def get_available_cubin_files(source, retries=3, delay=5, timeout=10):
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(source, timeout=timeout)
            response.raise_for_status()
            hrefs = re.findall(r'\<a href=".*\.cubin">', response.text)
            files = [(h[9:-8], ".cubin") for h in hrefs]
            return files

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Fetching available files {source}: attempt {attempt} failed: {e}"
            )

            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logger.error("Max retries reached. Fetch failed.")
                return []


class ArtifactPath:
    TRTLLM_GEN_FMHA: str = "037e528e719ec3456a7d7d654f26b805e44c63b1/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "037e528e719ec3456a7d7d654f26b805e44c63b1/batched_gemm-8704aa4-ba3b00d/"
    )
    TRTLLM_GEN_GEMM: str = (
        "037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e/"
    )
    CUDNN_SDPA: str = "4c623163877c8fef5751c9c7a59940cd2baae02e/fmha/cudnn/"
    DEEPGEMM: str = "d25901733420c7cddc1adf799b0d4639ed1e162f/deep-gemm/"


class MetaInfoHash:
    TRTLLM_GEN_FMHA: str = (
        "0ff77215b86997665cf75973e13cd2932f551d46b4e008f851d32d47e1d9560f"
    )
    TRTLLM_GEN_BMM: str = (
        "34bdfe7acfd49f5fb8b48e06d56e6a5ad88b951c730552f228fc5f614f7632a8"
    )
    DEEPGEMM: str = "69aa277b7f3663ed929e73f9c57301792b8c594dac15a465b44a5d151b6a1d50"
    TRTLLM_GEN_GEMM: str = (
        "0345358c916d990709f9670e113e93f35c76aa22715e2d5128ec2ca8740be5ba"
    )


def get_cubin_file_list():
    cubin_files = [
        (ArtifactPath.TRTLLM_GEN_FMHA + "include/flashInferMetaInfo", ".h"),
        (ArtifactPath.TRTLLM_GEN_GEMM + "include/flashinferMetaInfo", ".h"),
        (ArtifactPath.TRTLLM_GEN_BMM + "include/flashinferMetaInfo", ".h"),
    ]
    for kernel in [
        ArtifactPath.TRTLLM_GEN_FMHA,
        ArtifactPath.TRTLLM_GEN_BMM,
        ArtifactPath.TRTLLM_GEN_GEMM,
        ArtifactPath.DEEPGEMM,
    ]:
        cubin_files += [
            (kernel + name, extension)
            for name, extension in get_available_cubin_files(
                FLASHINFER_CUBINS_REPOSITORY + "/" + kernel
            )
        ]
    return cubin_files


def download_artifacts():
    import tqdm

    with temp_env_var("FLASHINFER_CUBIN_CHECKSUM_DISABLED", "1"):
        cubin_files = get_cubin_file_list()
        num_threads = int(os.environ.get("FLASHINFER_CUBIN_DOWNLOAD_THREADS", "4"))
        pool = ThreadPoolExecutor(num_threads)
        futures = []
        for name, extension in cubin_files:
            ret = pool.submit(get_cubin, name, "", extension)
            futures.append(ret)
        results = []
        with (
            patch_logger_for_tqdm(logger),
            tqdm(total=len(futures), desc="Downloading cubins") as pbar,
        ):
            for ret in as_completed(futures):
                result = ret.result()
                results.append(result)
                pbar.update(1)
        all_success = all(results)
    if not all_success:
        raise RuntimeError("Failed to download cubins")


def get_artifacts_status():
    """
    Check which cubins are already downloaded and return (num_downloaded, total).
    Does not download any cubins.
    """
    cubin_files = get_cubin_file_list()
    status = []
    for name, extension in cubin_files:
        # get_cubin stores cubins in FLASHINFER_CUBIN_DIR with the same relative path
        # Remove any leading slashes from name
        rel_path = name.lstrip("/")
        local_path = os.path.join(FLASHINFER_CUBIN_DIR, rel_path)
        exists = os.path.isfile(local_path + extension)
        status.append((name, extension, exists))
    return status


def clear_cubin():
    if os.path.exists(FLASHINFER_CUBIN_DIR):
        print(f"Clearing cubin directory: {FLASHINFER_CUBIN_DIR}")
        shutil.rmtree(FLASHINFER_CUBIN_DIR)
    else:
        print(f"Cubin directory does not exist: {FLASHINFER_CUBIN_DIR}")
