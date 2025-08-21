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
    TRTLLM_GEN_FMHA: str = "c8e0abb4b0438880a2b0a9b68449e3cf1513aadf/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "364304c7693814410e18e4bae11d8da011860117/batched_gemm-6492001-c97c649/"
    )
    TRTLLM_GEN_GEMM: str = (
        "5d347c6234c9f0e7f1ab6519ea933183b48216ed/gemm-32110eb-434a6e1/"
    )
    CUDNN_SDPA: str = "4c623163877c8fef5751c9c7a59940cd2baae02e/fmha/cudnn/"
    DEEPGEMM: str = "d25901733420c7cddc1adf799b0d4639ed1e162f/deep-gemm/"


class MetaInfoHash:
    TRTLLM_GEN_FMHA: str = (
        "0d124e546c8a2e9fa59499625e8a6d140a2465573d4a3944f9d29f29f73292fb"
    )
    TRTLLM_GEN_BMM: str = (
        "a2543b8fce60bebe071df40ef349edca32cea081144a4516b0089bd1487beb2b"
    )
    DEEPGEMM: str = "69aa277b7f3663ed929e73f9c57301792b8c594dac15a465b44a5d151b6a1d50"
    TRTLLM_GEN_GEMM: str = (
        "a00ef9d834cb66c724ec7c72337bc955dc53070a65a6f68b34f852d144fa6ea3"
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
