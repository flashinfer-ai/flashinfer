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

from .jit.core import logger
from .jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY, get_cubin


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
    TRTLLM_GEN_FMHA: str = "6c74964c96684c3e674340f7e35fc20ad909a9a0/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "6c74964c96684c3e674340f7e35fc20ad909a9a0/batched_gemm-8704aa4-ba3b00d/"
    )
    TRTLLM_GEN_GEMM: str = (
        "6c74964c96684c3e674340f7e35fc20ad909a9a0/gemm-8704aa4-f91dc9e/"
    )
    CUDNN_SDPA: str = "4c623163877c8fef5751c9c7a59940cd2baae02e/fmha/cudnn/"
    DEEPGEMM: str = "d25901733420c7cddc1adf799b0d4639ed1e162f/deep-gemm/"


class MetaInfoHash:
    TRTLLM_GEN_FMHA: str = (
        "5a41a165d4d5e956d4cccd0a7d1627dbdcaccf4d07a9cfcc8055ef0cb52e0c87"
    )
    TRTLLM_GEN_BMM: str = (
        "3edf4847059d465182779436397ece3d5fb45c3360a1d1abda3b71e35f957caa"
    )
    DEEPGEMM: str = "69aa277b7f3663ed929e73f9c57301792b8c594dac15a465b44a5d151b6a1d50"
    TRTLLM_GEN_GEMM: str = (
        "c6265cf047fc5d2208a37c54b5d720f4755b1215de5f7434ad24ffbc81c31c27"
    )


def download_artifacts() -> bool:
    env_backup = os.environ.get("FLASHINFER_CUBIN_CHECKSUM_DISABLED", None)
    os.environ["FLASHINFER_CUBIN_CHECKSUM_DISABLED"] = "1"
    cubin_files = [
        (ArtifactPath.TRTLLM_GEN_FMHA + "flashInferMetaInfo", ".h"),
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
    pool = ThreadPoolExecutor(4)
    futures = []
    for name, extension in cubin_files:
        ret = pool.submit(get_cubin, name, "", extension)
        futures.append(ret)
    results = []
    for ret in as_completed(futures):
        result = ret.result()
        results.append(result)
    all_success = all(results)
    if not env_backup:
        os.environ.pop("FLASHINFER_CUBIN_CHECKSUM_DISABLED")
    else:
        os.environ["FLASHINFER_CUBIN_CHECKSUM_DISABLED"] = env_backup

    return all_success
