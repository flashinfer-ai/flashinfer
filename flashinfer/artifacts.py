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
    TRTLLM_GEN_FMHA: str = "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/batched_gemm-6492001-c97c649/"
    )
    TRTLLM_GEN_GEMM: str = (
        "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/gemm-6492001-434a6e1/"
    )
    CUDNN_SDPA: str = "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/fmha/cudnn/"
    DEEPGEMM: str = "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/deep-gemm/"


class MetaInfoHash:
    TRTLLM_GEN_FMHA: str = (
        "9f8e809647a205f80547ad813892cec9b92ca086110878e135ba9e1be9ce805c"
    )
    TRTLLM_GEN_BMM: str = (
        "7709b92d29263c9e7eb2060a20233c6f93f0d8361f115b2eec68f1e437f48f89"
    )
    DEEPGEMM: str = "c17e9d4ca3593774cab4e788cb87ad152c2db220d52311ec2a014bda5286753d"
    TRTLLM_GEN_GEMM: str = (
        "9be7fada4f4eced19c4414dfe0860da0b1ca7cc38fec85c007534d88307e23df"
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
