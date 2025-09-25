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
from urllib.parse import urljoin
import requests  # type: ignore[import-untyped]
import shutil

from .jit.core import logger
from .jit.cubin_loader import (
    FLASHINFER_CUBINS_REPOSITORY,
    get_cubin,
    FLASHINFER_CUBIN_DIR,
)


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
    TRTLLM_GEN_FMHA: str = "7206d64e67f4c8949286246d6e2e07706af5d223/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "e6f22dcc3fdeb29ff87af2f4a2cb3d30b8d273e0/batched_gemm-45beda1-ee6a802/"
    )
    TRTLLM_GEN_GEMM: str = (
        "037e528e719ec3456a7d7d654f26b805e44c63b1/gemm-8704aa4-f91dc9e/"
    )
    CUDNN_SDPA: str = "4c623163877c8fef5751c9c7a59940cd2baae02e/fmha/cudnn/"
    DEEPGEMM: str = "51d730202c9eef782f06ecc950005331d85c5d4b/deep-gemm/"


class MetaInfoHash:
    TRTLLM_GEN_FMHA: str = (
        "2f605255e71d673768f5bece66dde9e2e9f4c873347bfe8fefcffbf86a3c847d"
    )
    TRTLLM_GEN_BMM: str = (
        "c98b4ce69a39fd41556d67033c30ea814ef76b0a2fe16e798e55baf0104acc34"
    )
    DEEPGEMM: str = "b4374f857c3066089c4ec6b5e79e785559fa2c05ce2623710b0b04bf86414a48"
    TRTLLM_GEN_GEMM: str = (
        "0345358c916d990709f9670e113e93f35c76aa22715e2d5128ec2ca8740be5ba"
    )


def get_cubin_file_list():
    base = FLASHINFER_CUBINS_REPOSITORY.rstrip("/")
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
                urljoin(base + "/", kernel)
            )
        ]
    return cubin_files


def download_artifacts():
    from tqdm.contrib.logging import tqdm_logging_redirect

    # use a shared session to make use of HTTP keep-alive and reuse of
    # HTTPS connections.
    session = requests.Session()

    with temp_env_var("FLASHINFER_CUBIN_CHECKSUM_DISABLED", "1"):
        cubin_files = get_cubin_file_list()
        num_threads = int(os.environ.get("FLASHINFER_CUBIN_DOWNLOAD_THREADS", "4"))
        with tqdm_logging_redirect(
            total=len(cubin_files), desc="Downloading cubins"
        ) as pbar:

            def update_pbar_cb(_) -> None:
                pbar.update(1)

            with ThreadPoolExecutor(num_threads) as pool:
                futures = []
                for name, extension in cubin_files:
                    fut = pool.submit(get_cubin, name, "", extension, session)
                    fut.add_done_callback(update_pbar_cb)
                    futures.append(fut)

                results = [fut.result() for fut in as_completed(futures)]

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
