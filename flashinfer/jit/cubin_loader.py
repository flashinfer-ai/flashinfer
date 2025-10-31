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

import ctypes
import hashlib
import os
from urllib.parse import urljoin
import shutil
import time
import uuid

import filelock

from .core import logger
from .env import FLASHINFER_CUBIN_DIR

# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)


def safe_urljoin(base, path):
    """Join URLs ensuring base is treated as a directory."""
    if not base.endswith("/"):
        base += "/"
    return urljoin(base, path)


def download_file(
    source: str,
    destination: str,
    retries: int = 4,
    delay: int = 5,
    timeout: int = 10,
    lock_timeout: int = 30,
    session=None,
):
    """
    Downloads a file from a URL or copies from a local path to a destination.
    If the filesystem supports atomic file rename operations, the destination file is
    either written completely or not at all with respect to concurrent access.

    Parameters:
    - source (str): The URL or local file path of the file to download.
    - destination (str): The local file path to save the downloaded/copied file.
    - retries (int): Number of retry attempts for URL downloads (default: 3).
    - delay (int): Initial delay in seconds for exponential backoff (default: 5).
    - timeout (int): Timeout for the HTTP request in seconds (default: 10).
    - lock_timeout (int): Timeout in seconds for the file lock (default: 30).

    Returns:
    - bool: True if download or copy is successful, False otherwise.
    """

    import requests  # type: ignore[import-untyped]

    if session is None:
        session = requests.Session()

    lock_path = f"{destination}.lock"  # Lock file path
    lock = filelock.FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock:
            logger.info(f"Acquired lock for {destination}")

            temp_path = f"{destination}.{uuid.uuid4().hex}.tmp"

            # Handle local file copy
            if os.path.exists(source):
                try:
                    shutil.copy(source, temp_path)
                    os.replace(temp_path, destination)  # Atomic rename
                    logger.info(f"File copied successfully: {destination}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to copy local file: {e}")
                    return False
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

            # Handle URL downloads with exponential backoff
            for attempt in range(1, retries + 1):
                try:
                    response = session.get(source, timeout=timeout)
                    response.raise_for_status()

                    with open(temp_path, "wb") as file:
                        file.write(response.content)

                    # Atomic rename to prevent readers from seeing partial writes
                    os.replace(temp_path, destination)

                    logger.info(
                        f"File downloaded successfully: {source} -> {destination}"
                    )
                    return True

                except requests.exceptions.RequestException as e:
                    logger.warning(
                        f"Downloading {source}: attempt {attempt} failed: {e}"
                    )

                    if attempt < retries:
                        backoff_delay = delay * (2 ** (attempt - 1))
                        logger.info(f"Retrying in {backoff_delay} seconds...")
                        time.sleep(backoff_delay)
                    else:
                        logger.error("Max retries reached. Download failed.")
                        return False
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    except filelock.Timeout:
        logger.error(
            f"Failed to acquire lock for {destination} within {lock_timeout} seconds."
        )
        return False


def get_meta_hash(checksums_bytes: bytes) -> str:
    """
    Parse the checksums.txt file and get the hash of corresponding flashinferMetaInfo.h file
    """
    checksums_lines = checksums_bytes.decode("utf-8").splitlines()
    for line in checksums_lines:
        sha256, filename = line.strip().split()
        if ".h" in filename:
            return sha256
    raise ValueError("Invalid checksums.txt, no flashinferMetaInfo.h found")


def verify_cubin(cubin_path: str, expected_sha256: str) -> bool:
    """
    Verify the cubin file against the sha256 checksum.
    """
    with open(cubin_path, "rb") as f:
        data = f.read()
    actual_sha256 = hashlib.sha256(data).hexdigest()
    if actual_sha256 != expected_sha256:
        logger.warning(
            f"sha256 mismatch (expected {expected_sha256} actual {actual_sha256}) for {cubin_path}"
        )
        return False
    return True


def load_cubin(cubin_path: str, sha256: str) -> bytes:
    """
    Load a cubin from the provide local path and
    ensure that the sha256 signature matches.

    Return None on failure.
    """
    logger.debug(f"Loading from {cubin_path}")
    try:
        with open(cubin_path, mode="rb") as f:
            cubin = f.read()
            if os.getenv("FLASHINFER_CUBIN_CHECKSUM_DISABLED"):
                return cubin
            m = hashlib.sha256()
            m.update(cubin)
            actual_sha = m.hexdigest()
            if sha256 == actual_sha:
                return cubin
            logger.warning(
                f"sha256 mismatch (expected {sha256} actual {actual_sha}) for {cubin_path}"
            )
    except Exception:
        pass
    return b""


def get_cubin(file_name: str, sha256: str, session=None) -> bytes:
    """
    Load a cubin from the local cache directory with {file_name} and
    ensure that the sha256 signature matches.

    If the kernel does not exist in the cache, it will downloaded.

    Returns:
    None on failure.
    """
    cubin_path = str(FLASHINFER_CUBIN_DIR / file_name)
    cubin = load_cubin(cubin_path, sha256)
    if cubin:
        return cubin
    # either the file does not exist or it is corrupted, we'll download a new one.

    uri = safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, file_name)
    logger.info(f"Fetching cubin {file_name} from {uri}")
    download_file(uri, cubin_path, session=session)
    return load_cubin(cubin_path, sha256)


def convert_to_ctypes_char_p(data: bytes):
    return ctypes.c_char_p(data)


# Keep a reference to the callback for each loaded library to prevent GC
dll_cubin_handlers = {}


def setup_cubin_loader(dll_path: str) -> None:
    if dll_path in dll_cubin_handlers:
        return

    _LIB = ctypes.CDLL(dll_path)

    # Define the correct callback type
    CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

    def get_cubin_callback(name: bytes, sha256: bytes):
        # Both name and sha256 are bytes (c_char_p)
        cubin = get_cubin(name.decode("utf-8"), sha256.decode("utf-8"))
        _LIB.FlashInferSetCurrentCubin(
            convert_to_ctypes_char_p(cubin), ctypes.c_int(len(cubin))
        )

    # Create the callback and keep a reference to prevent GC
    cb = CALLBACK_TYPE(get_cubin_callback)
    dll_cubin_handlers[dll_path] = cb

    _LIB.FlashInferSetCubinCallback(cb)
