"""
Copyright (c) 2025-2026 by FlashInfer team.

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

    # The trtllm-gen packages below are single-package, multi-architecture: one
    # publish carries the Blackwell (sm100f/sm103a) and Rubin (sm107a) cubins.
    TRTLLM_GEN_FMHA: str = "2d6a5a029eefcc388ec0ceb87efb55d8bcce5c3c/fmha/trtllm-gen/"
    TRTLLM_GEN_BMM: str = (
        "1d145b82ac60add55ea213863523f12d63005651/batched_gemm-09795a1-31ee4e5/"
    )
    TRTLLM_GEN_GEMM: str = (
        "7b1fc253cd6237950e76310873f4acf4d97a3904/gemm-b738138-25754e6/"
    )
    CUDNN_SDPA: str = "a72d85b019dc125b9f711300cb989430f762f5a6/fmha/cudnn/"
    # For DEEPGEMM, we also need to update KernelMap.KERNEL_MAP_HASH in flashinfer/deep_gemm.py
    DEEPGEMM: str = "7ec7ac40b9fd48172651b77ff2ebe20d79decc39/deep-gemm/"
    DSL_FMHA: str = "5b34f84266cbc2135066ce96885b664992535670/fmha/cute-dsl/"
    DSL_FMHA_ARCHS: tuple[str, ...] = (
        "sm_100a",
        "sm_103a",
        "sm_107a",
        "sm_110a",
    )


class CheckSumHash:
    """
    This class is used to store the checksums of the cubin files in artifactory.
    The sha256 hashes are generated in cubin publishing script logs (accessible by codeowners).
    When updating the ArtifactPath for backend directories, update the corresponding hash.
    """

    TRTLLM_GEN_FMHA: str = (
        "d79b5c51fc8597fac57dae0da4afa114fb2014575e4ec3df099ad856d97cabc3"
    )
    TRTLLM_GEN_BMM: str = (
        "e071273ce357ee3e8d40ce905dac03d2a6078f6c5869ca3b7d1f1d146643f009"
    )
    DEEPGEMM: str = "09e961d4e3852a6cf81b3482d0604c09dcb1f69c1b7936f535c9ee2f53335184"
    TRTLLM_GEN_GEMM: str = (
        "ca9d4f956f3fb63bff3066db88fa7ccf08b00f4b0b2751cc14ba72454fd01638"
    )
    # SHA256 of the checksums.txt manifest file per cpu-arch/sm-arch,
    # NOT hashes of individual kernel .so files.
    DSL_FMHA_CHECKSUMS: dict[str, dict[str, str]] = {
        "x86_64": {
            "sm_100a": "832c303bb9b386af590d3efc294681859829b91991975fd2e188a5d7dc30c461",
            "sm_103a": "57322c10ddbbe9072c7ded41e2856fdf9d4276fbd79ac4bc825af0cd78844da6",
            "sm_107a": "8480678539adf622f8395e875923471bce683be7158694556bd6c536eadeaa45",
            "sm_110a": "4f6f0f3a868f0e9171c8ab217e6d2a87fde46b02a9417d8f55f1a779c53fa9fb",
        },
        "aarch64": {
            "sm_100a": "064cfcac21886c3e16b5007ca769f1b93111db7a63864db1e026a53e61fe20fe",
            "sm_103a": "1631a884738d706f5bc39bf4032bcd54b25ba0a0d734c26f388dce4ae32093c9",
            "sm_107a": "0348ef0b74dffa67c0c9662d3f567b26dccf46c356009f315860f31a9550e207",
            "sm_110a": "8f16f510d159797432bda92d55d0d82d65d3e19080f9af1b751dec4146cdcbe6",
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
        if not download_file(uri, checksum_path) and not checksum_path.is_file():
            # Without this the next open() fails with a bare FileNotFoundError on
            # the local cache path, which hides the real cause: the artifact pin
            # is unreachable (typo'd/unpublished pin, or network/mirror failure).
            raise RuntimeError(
                f"Failed to fetch the checksum manifest for artifact pin '{subdir}' "
                f"from {uri}. Check that the pin exists in "
                f"{FLASHINFER_CUBINS_REPOSITORY} and is reachable."
            )
        with open(checksum_path, "r") as f:
            for line in f:
                sha256, filename = line.strip().split()

                # Key every entry by its full path. Bare filenames are not
                # unique across subdirs: two pins built from different sources
                # can ship identically named kernels, so a flat dict would let
                # the subdir processed last silently overwrite the earlier
                # one's hashes and fail verification for every shared name.
                checksums[safe_urljoin(subdir, filename)] = sha256
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
            full_path = safe_urljoin(cubin_dir, name)
            yield (full_path, checksums[full_path])
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
