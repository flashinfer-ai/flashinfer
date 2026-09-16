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

    raise RuntimeError(
        f"Failed to fetch the cubin artifact index {source} after {retries} attempts"
    )


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

        raise RuntimeError(
            f"Failed to fetch the header artifact index {url} after {retries} attempts"
        )

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
    DSL_FMHA: str = "6efb974aae4c012d9fb317c2ef360210b3f352e4/fmha/cute-dsl/"
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
            "sm_100a": "7bb9eb497d295a6471ce85d0913e3b4955fd9a7d918ddd9b92882cba41af5365",
            "sm_103a": "3c0dc183a6f73fe3f0705a4b6d6fe8de667cf1dd1908bfe27422327faa16e27b",
            "sm_107a": "70be29547f3d9b2e7e22981865de20f35eef61fa3b6493443680b058e7523dd4",
            "sm_110a": "817e55486f3c35fe1841dccafc9b7fe34e50aff99edc4a15da952ace123d9edd",
        },
        "aarch64": {
            "sm_100a": "f1395b80f2c8917fd1f52dca0bf0a37efc6f74fc594c245284076c72bf7e8130",
            "sm_103a": "c4a44f8be82d9544f18eb7e139712d9fe3b09b30d890b5d09e3cbc5a203a1544",
            "sm_107a": "fe30ea44d746c630c0347ed357ce317de29968da3e1d848a1d6a6864cbd25b7e",
            "sm_110a": "634636627af7a98f8e1ff8c8332baac91a2e2f4f3dddbf647b6da9d60a8085ca",
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

    cubin_files = list[tuple[str, str]](get_subdir_file_list())
    num_threads = int(os.environ.get("FLASHINFER_CUBIN_DOWNLOAD_THREADS", "4"))
    max_retries = os.environ.get("FLASHINFER_CUBIN_MAX_RETRIES")
    retry_window_seconds = int(os.environ.get("FLASHINFER_CUBIN_RETRY_WINDOW_SECONDS", "0"))
    retry_deadline = (
        time.monotonic() + retry_window_seconds if retry_window_seconds > 0 else None
    )

    cached_files: set[str] = set()
    files_to_download: list[tuple[str, str]] = []
    for name, checksum in cubin_files:
        local_path = FLASHINFER_CUBIN_DIR / name
        if local_path.is_file():
            try:
                if verify_cubin(str(local_path), checksum):
                    cached_files.add(name)
                    continue
            except OSError as e:
                logger.warning(f"Failed to read cached artifact {local_path}: {e}")
        files_to_download.append((name, checksum))

    logger.info(
        "Using %d checksum-verified cached artifacts; downloading %d artifacts",
        len(cached_files),
        len(files_to_download),
    )

    with tqdm_logging_redirect(
        total=len(cubin_files), desc="Downloading cubins"
    ) as pbar:
        pbar.update(len(cached_files))

        def update_pbar_cb(_) -> None:
            pbar.update(1)

        def _download_within_retry_window(
            source_path: str,
            destination_path: str,
            artifact_name: str,
            kwargs: dict[str, object],
        ) -> bool:
            with requests.Session() as session:
                request_kwargs = dict(kwargs)
                request_kwargs["session"] = session
                while True:
                    if download_file(source_path, destination_path, **request_kwargs):
                        return True
                    if retry_deadline is None:
                        return False
                    remaining = retry_deadline - time.monotonic()
                    if remaining <= 0:
                        logger.error(
                            "Retry window exhausted for %s after %d seconds",
                            artifact_name,
                            retry_window_seconds,
                        )
                        return False
                    logger.warning(
                        "Download failed for %s; retrying while %0.2f seconds remain in retry window",
                        artifact_name,
                        remaining,
                    )
                    time.sleep(min(5.0, remaining))

        with ThreadPoolExecutor(num_threads) as pool:
            futures = []
            for name, _ in files_to_download:
                source = safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, name)
                local_path = FLASHINFER_CUBIN_DIR / name
                # Ensure parent directory exists
                local_path.parent.mkdir(parents=True, exist_ok=True)
                download_kwargs: dict[str, object] = {}
                if max_retries is not None:
                    download_kwargs["retries"] = int(max_retries)
                fut = pool.submit(
                    _download_within_retry_window,
                    source,
                    str(local_path),
                    name,
                    download_kwargs,
                )
                fut.add_done_callback(update_pbar_cb)
                futures.append(fut)

            results = [fut.result() for fut in as_completed(futures)]

    all_success = all(results)
    if not all_success:
        raise RuntimeError("Failed to download cubins")

    # Cached artifacts were verified before they were skipped. Verify each file
    # fetched in this invocation before allowing it into the wheel.
    for name, checksum in files_to_download:
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
