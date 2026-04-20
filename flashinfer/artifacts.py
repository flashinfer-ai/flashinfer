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

from dataclasses import dataclass
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator
import requests  # type: ignore[import-untyped]
import shutil

# Create logger for artifacts module to avoid circular import with jit.core
logger = logging.getLogger("flashinfer.artifacts")
logger.setLevel(os.getenv("FLASHINFER_LOGGING_LEVEL", "INFO").upper())
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())

from .jit.cubin_loader import (
    FLASHINFER_CUBINS_REPOSITORY,
    safe_urljoin,
    FLASHINFER_CUBIN_DIR,
    download_file,
    verify_cubin,
)


from contextlib import contextmanager


@contextmanager
def temp_env_var(key: str, value: str):
    old_value = os.environ.get(key, None)
    os.environ[key] = value
    try:
        yield
    finally:
        if old_value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = old_value


def get_available_cubin_files(
    source: str, retries: int = 3, delay: int = 5, timeout: int = 10
) -> tuple[str, ...]:
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(source, timeout=timeout)
            response.raise_for_status()
            hrefs = re.findall(r'\<a href=".*\.cubin">', response.text)
            return tuple((h[9:-8] + ".cubin") for h in hrefs)

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"Fetching available files {source}: attempt {attempt} failed: {e}"
            )

            if attempt < retries:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)

    # TODO: check if we really want to return an empty collection here instead of crashing.
    logger.error("Max retries reached. Fetch failed.")
    return tuple()


def get_available_header_files(
    source: str, retries: int = 3, delay: int = 5, timeout: int = 10
) -> tuple[str, ...]:
    """
    Recursively navigates through child directories (e.g., include/) and finds
    all *.h header files, returning them as a tuple of relative paths.
    """
    result: list[str] = []

    def fetch_directory(url: str, prefix: str = "") -> None:
        for attempt in range(1, retries + 1):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()

                # Find all .h header files in this directory
                header_hrefs = re.findall(r'<a href="([^"]+\.h)">', response.text)
                for h in header_hrefs:
                    result.append(prefix + h if prefix else h)

                # Find all subdirectories (links ending with /)
                dir_hrefs = re.findall(r'<a href="([^"]+/)">', response.text)
                for d in dir_hrefs:
                    # Skip parent directory links
                    if d == "../" or d.startswith(".."):
                        continue
                    subdir_url = safe_urljoin(url, d)
                    subdir_prefix = prefix + d if prefix else d
                    fetch_directory(subdir_url, subdir_prefix)

                return  # Success, exit retry loop

            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Fetching available header files {url}: attempt {attempt} failed: {e}"
                )

                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)

        logger.error(f"Max retries reached for {url}. Fetch failed.")

    fetch_directory(source)
    logger.info(f"result: {result}")
    return tuple(result)


@dataclass(frozen=True)
class ArtifactPath:
    """
    This class is used to store the paths of the cubin files in artifactory.
    The paths are generated in cubin publishing script logs (accessible by codeowners).
    When compiling new cubins for backend directories, update the corresponding path.
    """

    TRTLLM_GEN_FMHA: str = "55bba55929d4093682e32d817bd11ffb0441c749/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "39a9d28268f43475a757d5700af135e1e58c9849/batched_gemm-5ee61af-2b9855b/"
    )
    TRTLLM_GEN_GEMM: str = (
        "31e75d429ff3f710de1251afdd148185f53da44d/gemm-4daf11e-1fddea2/"
    )
    CUDNN_SDPA: str = "a72d85b019dc125b9f711300cb989430f762f5a6/fmha/cudnn/"
    # For DEEPGEMM, we also need to update KernelMap.KERNEL_MAP_HASH in flashinfer/deep_gemm.py
    DEEPGEMM: str = "a72d85b019dc125b9f711300cb989430f762f5a6/deep-gemm/"
    DSL_FMHA: str = "b0adf88479d74ce564eaeb8b3e12547806e6b9c0/fmha/cute-dsl/"
    DSL_FMHA_ARCHS: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_110a")


class CheckSumHash:
    """
    This class is used to store the checksums of the cubin files in artifactory.
    The sha256 hashes are generated in cubin publishing script logs (accessible by codeowners).
    When updating the ArtifactPath for backend directories, update the corresponding hash.
    """

    TRTLLM_GEN_FMHA: str = (
        "f2c0aad1e74391c4267a2f9a20ec819358b59e04588385cffb452ed341500b99"
    )
    TRTLLM_GEN_BMM: str = (
        "db06db7f36a2a9395a2041ff6ac016fe664874074413a2ed90797f91ef17e0f6"
    )
    DEEPGEMM: str = "1a2a166839042dbd2a57f48051c82cd1ad032815927c753db269a4ed10d0ffbf"
    TRTLLM_GEN_GEMM: str = (
        "64b7114a429ea153528dd4d4b0299363d7320964789eb5efaefec66f301523c7"
    )
    # SHA256 of the checksums.txt manifest file per cpu-arch/sm-arch,
    # NOT hashes of individual kernel .so files.
    DSL_FMHA_CHECKSUMS: dict[str, dict[str, str]] = {
        "x86_64": {
            "sm_100a": "533b56cf30fed9ecee5e305eb50cbf3c1a86ed7ef946f7c0b2530792c391452b",
            "sm_103a": "13eaa3b5119e7f0d295e5d9f9c15ae282c2c2fe19aec306dfb9ac1aa9eb9feab",
            "sm_110a": "fb68dc3917f28de610ef5d52745186280d7f58863d019f31f3498463e6336ca6",
        },
        "aarch64": {
            "sm_100a": "4b968ac45e82672c77c46496bcf4a6383aee36cd461381dd77eb9346de0a169a",
            "sm_103a": "038f78e23cfe365caed93b5b2d575a1a7c0650f43567e59f4bbd4f6eb6d1c3b0",
            "sm_110a": "44c4c2404173e7119fd95a338a0da32e23fb8ca5e9d92e67e0b7433aaed8d416",
        },
    }
    map_checksums: dict[str, str] = {
        safe_urljoin(ArtifactPath.TRTLLM_GEN_FMHA, "checksums.txt"): TRTLLM_GEN_FMHA,
        safe_urljoin(ArtifactPath.TRTLLM_GEN_BMM, "checksums.txt"): TRTLLM_GEN_BMM,
        safe_urljoin(ArtifactPath.DEEPGEMM, "checksums.txt"): DEEPGEMM,
        safe_urljoin(ArtifactPath.TRTLLM_GEN_GEMM, "checksums.txt"): TRTLLM_GEN_GEMM,
        **{
            safe_urljoin(
                ArtifactPath.DSL_FMHA, f"{cpu_arch}/{sm_arch}/checksums.txt"
            ): sha
            for cpu_arch, sm_checksums in DSL_FMHA_CHECKSUMS.items()
            for sm_arch, sha in sm_checksums.items()
        },
    }


def get_checksums(subdirs):
    checksums = {}
    for subdir in subdirs:
        uri = safe_urljoin(
            FLASHINFER_CUBINS_REPOSITORY, safe_urljoin(subdir, "checksums.txt")
        )
        checksum_path = FLASHINFER_CUBIN_DIR / safe_urljoin(subdir, "checksums.txt")
        download_file(uri, checksum_path)
        with open(checksum_path, "r") as f:
            for line in f:
                sha256, filename = line.strip().split()

                # Distinguish between all meta info header files
                if ".h" in filename:
                    filename = safe_urljoin(subdir, filename)
                checksums[filename] = sha256
    return checksums


def _get_host_cpu_arch() -> str:
    """Return CPU architecture string matching artifactory layout."""
    import platform

    machine = platform.machine()
    if machine in ("aarch64", "arm64"):
        return "aarch64"
    return "x86_64"


def get_subdir_file_list() -> Generator[tuple[str, str], None, None]:
    base = FLASHINFER_CUBINS_REPOSITORY
    cpu_arch = _get_host_cpu_arch()

    cubin_dirs = [
        ArtifactPath.TRTLLM_GEN_FMHA,
        ArtifactPath.TRTLLM_GEN_BMM,
        ArtifactPath.TRTLLM_GEN_GEMM,
        ArtifactPath.DEEPGEMM,
        # DSL FMHA: per cpu-arch and sm-arch subdirectories
        *(
            safe_urljoin(ArtifactPath.DSL_FMHA, f"{cpu_arch}/{arch}/")
            for arch in ArtifactPath.DSL_FMHA_ARCHS
        ),
    ]

    # Get checksums of all files
    checksums = get_checksums(cubin_dirs)

    # The meta info header files first.
    yield (
        safe_urljoin(ArtifactPath.TRTLLM_GEN_FMHA, "include/flashInferMetaInfo.h"),
        checksums[
            safe_urljoin(ArtifactPath.TRTLLM_GEN_FMHA, "include/flashInferMetaInfo.h")
        ],
    )
    yield (
        safe_urljoin(ArtifactPath.TRTLLM_GEN_GEMM, "include/flashinferMetaInfo.h"),
        checksums[
            safe_urljoin(ArtifactPath.TRTLLM_GEN_GEMM, "include/flashinferMetaInfo.h")
        ],
    )
    yield (
        safe_urljoin(ArtifactPath.TRTLLM_GEN_BMM, "include/flashinferMetaInfo.h"),
        checksums[
            safe_urljoin(ArtifactPath.TRTLLM_GEN_BMM, "include/flashinferMetaInfo.h")
        ],
    )

    # All the actual kernel cubin's.
    for cubin_dir in cubin_dirs:
        checksum_path = safe_urljoin(cubin_dir, "checksums.txt")
        yield (checksum_path, CheckSumHash.map_checksums[checksum_path])
        for name in get_available_cubin_files(safe_urljoin(base, cubin_dir)):
            yield (safe_urljoin(cubin_dir, name), checksums[name])
        for name in get_available_header_files(safe_urljoin(base, cubin_dir)):
            full_path = safe_urljoin(cubin_dir, name)
            yield (full_path, checksums[full_path])


def download_artifacts() -> None:
    from tqdm.contrib.logging import tqdm_logging_redirect

    # use a shared session to make use of HTTP keep-alive and reuse of
    # HTTPS connections.
    session = requests.Session()
    cubin_files = list[tuple[str, str]](get_subdir_file_list())
    num_threads = int(os.environ.get("FLASHINFER_CUBIN_DOWNLOAD_THREADS", "4"))
    with tqdm_logging_redirect(
        total=len(cubin_files), desc="Downloading cubins"
    ) as pbar:

        def update_pbar_cb(_) -> None:
            pbar.update(1)

        with ThreadPoolExecutor(num_threads) as pool:
            futures = []
            for name, _ in cubin_files:
                source = safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, name)
                local_path = FLASHINFER_CUBIN_DIR / name
                # Ensure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)
                fut = pool.submit(
                    download_file, source, str(local_path), session=session
                )
                fut.add_done_callback(update_pbar_cb)
                futures.append(fut)

            results = [fut.result() for fut in as_completed(futures)]

    all_success = all(results)
    if not all_success:
        raise RuntimeError("Failed to download cubins")

    # Check checksums of all downloaded cubins
    for name, checksum in cubin_files:
        local_path = FLASHINFER_CUBIN_DIR / name
        if not verify_cubin(str(local_path), checksum):
            raise RuntimeError("Failed to download cubins: checksum mismatch")


def get_artifacts_status() -> tuple[tuple[str, bool], ...]:
    """
    Check which cubins are already downloaded and return (num_downloaded, total).
    Does not download any cubins.
    """
    cubin_files = get_subdir_file_list()

    def _check_file_status(file_name: str) -> tuple[str, bool]:
        # get_artifact stores files in FLASHINFER_CUBIN_DIR with the same relative path
        # Remove any leading slashes from name
        local_path = os.path.join(FLASHINFER_CUBIN_DIR, file_name)
        exists = os.path.isfile(local_path)
        return (file_name, exists)

    return tuple(_check_file_status(file_name) for file_name, _ in cubin_files)


def clear_cubin():
    if os.path.exists(FLASHINFER_CUBIN_DIR):
        logger.info(f"Clearing cubin directory: {FLASHINFER_CUBIN_DIR}")
        shutil.rmtree(FLASHINFER_CUBIN_DIR)
    else:
        logger.info(f"Cubin directory does not exist: {FLASHINFER_CUBIN_DIR}")
