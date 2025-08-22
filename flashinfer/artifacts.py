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
from typing import Any, Type, ClassVar
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests  # type: ignore[import-untyped]

from .jit.core import logger
from .jit.cubin_loader import FLASHINFER_CUBINS_REPOSITORY, get_cubin
from .jit import env as jit_env


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


class ArtifactPathBase:
    TRTLLM_GEN_FMHA: ClassVar[str]
    TRTLLM_GEN_FMHA_INCLUDE_PATH: ClassVar[str]
    TRTLLM_GEN_BMM: ClassVar[str]
    TRTLLM_GEN_GEMM: ClassVar[str]
    CUDNN_SDPA: ClassVar[str]
    DEEPGEMM: ClassVar[str]

    def __init_subclass__(cls: Type["ArtifactPathBase"], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if jit_env.FLASHINFER_CUDA_VERSION.major >= 13:
            cls.TRTLLM_GEN_FMHA = (
                "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/fmha/trtllm-gen/"
            )
            cls.TRTLLM_GEN_FMHA_INCLUDE_PATH = cls.TRTLLM_GEN_FMHA + "include/"
            cls.TRTLLM_GEN_BMM = (
                "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/batched_gemm-6492001-c97c649/"
            )
            cls.TRTLLM_GEN_GEMM = (
                "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/gemm-6492001-434a6e1/"
            )
            cls.CUDNN_SDPA = "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/fmha/cudnn/"
            cls.DEEPGEMM = "d8c2e4e646bd7e73ea79f06ae52b4ba13adddc64/deep-gemm/"
        else:
            cls.TRTLLM_GEN_FMHA = (
                "c8e0abb4b0438880a2b0a9b68449e3cf1513aadf/fmha/trtllm-gen/"
            )
            cls.TRTLLM_GEN_FMHA_INCLUDE_PATH = cls.TRTLLM_GEN_FMHA
            cls.TRTLLM_GEN_BMM = (
                "364304c7693814410e18e4bae11d8da011860117/batched_gemm-6492001-c97c649/"
            )
            cls.TRTLLM_GEN_GEMM = (
                "5d347c6234c9f0e7f1ab6519ea933183b48216ed/gemm-32110eb-434a6e1/"
            )
            cls.CUDNN_SDPA = "4c623163877c8fef5751c9c7a59940cd2baae02e/fmha/cudnn/"
            cls.DEEPGEMM = "d25901733420c7cddc1adf799b0d4639ed1e162f/deep-gemm/"


class MetaInfoHashBase:
    TRTLLM_GEN_FMHA: ClassVar[str]
    TRTLLM_GEN_BMM: ClassVar[str]
    DEEPGEMM: ClassVar[str]
    TRTLLM_GEN_GEMM: ClassVar[str]

    def __init_subclass__(cls: Type["MetaInfoHashBase"], **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if jit_env.FLASHINFER_CUDA_VERSION.major >= 13:
            cls.TRTLLM_GEN_FMHA = (
                "9f8e809647a205f80547ad813892cec9b92ca086110878e135ba9e1be9ce805c"
            )
            cls.TRTLLM_GEN_BMM = (
                "7709b92d29263c9e7eb2060a20233c6f93f0d8361f115b2eec68f1e437f48f89"
            )
            cls.DEEPGEMM = (
                "c17e9d4ca3593774cab4e788cb87ad152c2db220d52311ec2a014bda5286753d"
            )
            cls.TRTLLM_GEN_GEMM = (
                "9be7fada4f4eced19c4414dfe0860da0b1ca7cc38fec85c007534d88307e23df"
            )
        else:
            cls.TRTLLM_GEN_FMHA = (
                "0d124e546c8a2e9fa59499625e8a6d140a2465573d4a3944f9d29f29f73292fb"
            )
            cls.TRTLLM_GEN_BMM = (
                "a2543b8fce60bebe071df40ef349edca32cea081144a4516b0089bd1487beb2b"
            )
            cls.DEEPGEMM = (
                "69aa277b7f3663ed929e73f9c57301792b8c594dac15a465b44a5d151b6a1d50"
            )
            cls.TRTLLM_GEN_GEMM = (
                "a00ef9d834cb66c724ec7c72337bc955dc53070a65a6f68b34f852d144fa6ea3"
            )


class ArtifactPath(ArtifactPathBase):
    pass


class MetaInfoHash(MetaInfoHashBase):
    pass


def download_artifacts() -> bool:
    env_backup = os.environ.get("FLASHINFER_CUBIN_CHECKSUM_DISABLED", None)
    os.environ["FLASHINFER_CUBIN_CHECKSUM_DISABLED"] = "1"

    cubin_files = [
        (ArtifactPath.TRTLLM_GEN_FMHA_INCLUDE_PATH + "flashInferMetaInfo", ".h"),
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
