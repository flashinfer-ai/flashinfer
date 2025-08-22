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
import shutil
import time

import filelock

from .core import logger
from .env import FLASHINFER_CUBIN_DIR

# This is the storage path for the cubins, it can be replaced
# with a local path for testing.
FLASHINFER_CUBINS_REPOSITORY = os.environ.get(
    "FLASHINFER_CUBINS_REPOSITORY",
    "https://edge.urm.nvidia.com/artifactory/sw-kernelinferencelibrary-public-generic-local/",
)


def download_file(source, local_path, retries=3, delay=5, timeout=10, lock_timeout=30):
    """
    Downloads a file from a URL or copies from a local path to a destination.

    Parameters:
    - source (str): The URL or local file path of the file to download.
    - local_path (str): The local file path to save the downloaded/copied file.
    - retries (int): Number of retry attempts for URL downloads (default: 3).
    - delay (int): Delay in seconds between retries (default: 5).
    - timeout (int): Timeout for the HTTP request in seconds (default: 10).
    - lock_timeout (int): Timeout in seconds for the file lock (default: 30).

    Returns:
    - bool: True if download or copy is successful, False otherwise.
    """

    import requests  # type: ignore[import-untyped]

    lock_path = f"{local_path}.lock"  # Lock file path
    lock = filelock.FileLock(lock_path, timeout=lock_timeout)

    try:
        with lock:
            logger.info(f"Acquired lock for {local_path}")

            # Handle local file copy
            if os.path.exists(source):
                try:
                    shutil.copy(source, local_path)
                    logger.info(f"File copied successfully: {local_path}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to copy local file: {e}")
                    return False

            # Handle URL downloads
            for attempt in range(1, retries + 1):
                try:
                    response = requests.get(source, timeout=timeout)
                    response.raise_for_status()

                    with open(local_path, "wb") as file:
                        file.write(response.content)

                    logger.info(
                        f"File downloaded successfully: {source} -> {local_path}"
                    )
                    return True

                except requests.exceptions.RequestException as e:
                    logger.warning(
                        f"Downloading {source}: attempt {attempt} failed: {e}"
                    )

                    if attempt < retries:
                        logger.info(f"Retrying in {delay} seconds...")
                        time.sleep(delay)
                    else:
                        logger.error("Max retries reached. Download failed.")
                        return False

    except filelock.Timeout:
        logger.error(
            f"Failed to acquire lock for {local_path} within {lock_timeout} seconds."
        )
        return False

    finally:
        # Clean up the lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)
            logger.info(f"Lock file {lock_path} removed.")


def load_cubin(cubin_path, sha256) -> bytes:
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


def get_cubin(name, sha256, file_extension=".cubin"):
    """
    Load a cubin from the local cache directory with {name} and
    ensure that the sha256 signature matches.

    If the kernel does not exist in the cache, it will downloaded.

    Returns:
    None on failure.
    """
    cubin_fname = name + file_extension
    cubin_path = FLASHINFER_CUBIN_DIR / cubin_fname
    cubin = load_cubin(cubin_path, sha256)
    if cubin:
        return cubin
    # either the file does not exist or it is corrupted, we'll download a new one.
    uri = FLASHINFER_CUBINS_REPOSITORY + "/" + cubin_fname
    logger.info(f"Fetching cubin {name} from {uri}")
    download_file(uri, cubin_path)
    return load_cubin(cubin_path, sha256)


def convert_to_ctypes_char_p(data: bytes):
    return ctypes.c_char_p(data)


# Keep a reference to the callback for each loaded library to prevent GC
dll_cubin_handlers = {}


def setup_cubin_loader(dll_path: str):
    if dll_path in dll_cubin_handlers:
        return

    _LIB = ctypes.CDLL(dll_path)

    # Define the correct callback type
    CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_char_p)

    def get_cubin_callback(name, sha256):
        # Both name and sha256 are bytes (c_char_p)
        cubin = get_cubin(name.decode("utf-8"), sha256.decode("utf-8"))
        _LIB.FlashInferSetCurrentCubin(
            convert_to_ctypes_char_p(cubin), ctypes.c_int(len(cubin))
        )

    # Create the callback and keep a reference to prevent GC
    cb = CALLBACK_TYPE(get_cubin_callback)
    dll_cubin_handlers[dll_path] = cb

    _LIB.FlashInferSetCubinCallback(cb)


dll_metainfo_handlers = {}


def setup_metainfo_loader(dll_path: str):
    if dll_path in dll_metainfo_handlers:
        return

    _LIB = ctypes.CDLL(dll_path)

    # Define the correct callback type
    CALLBACK_TYPE = ctypes.CFUNCTYPE(
        None, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p
    )

    def get_metainfo_callback(name, sha256, extension):
        metainfo = get_cubin(
            name.decode("utf-8"), sha256.decode("utf-8"), extension.decode("utf-8")
        )
        _LIB.FlashInferSetCurrentMetaInfo(
            convert_to_ctypes_char_p(metainfo), ctypes.c_int(len(metainfo))
        )

    # Create the callback and keep a reference to prevent GC
    cb = CALLBACK_TYPE(get_metainfo_callback)
    dll_metainfo_handlers[dll_path] = cb

    _LIB.FlashInferSetMetaInfoCallback(cb)
