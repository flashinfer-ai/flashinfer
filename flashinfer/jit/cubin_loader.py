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
import pathlib
import subprocess
from urllib.parse import urljoin
import shutil
import time
from typing import List, Union
import uuid

import filelock

from .core import logger
from .env import FLASHINFER_CUBIN_DIR

# Track cubins compiled from source (.cu files shipped in the repo).
# These bypass sha256 verification since their hash differs from the
# pre-built cubin hash in flashinferMetaInfo.h.
_source_compiled_cubins: set[str] = set()

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


def get_meta_hash(
    checksums_bytes: bytes, target_file: str = "flashinferMetaInfo.h"
) -> str:
    """
    Parse the checksums.txt file and get the hash of corresponding flashinferMetaInfo.h file
    """
    checksums_lines = checksums_bytes.decode("utf-8").splitlines()
    for line in checksums_lines:
        sha256, filename = line.strip().split()
        # Match on path segment boundary to avoid substring collisions
        # (e.g. "Enums.h" must not match "BatchedGemmEnums.h")
        if filename.lower() == target_file.lower() or filename.lower().endswith(
            "/" + target_file.lower()
        ):
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


def get_artifact(file_name: str, sha256: str, session=None) -> bytes:
    """Load an artifact (cubin, header, checksum, etc.) from the local cache.

    Checks ``FLASHINFER_CUBIN_DIR / file_name`` first.  If the file is missing
    or its SHA-256 doesn't match, it is downloaded from
    ``FLASHINFER_CUBINS_REPOSITORY``.

    Returns the file contents as bytes, or empty bytes on failure.
    """
    local_path = str(FLASHINFER_CUBIN_DIR / file_name)
    data = load_cubin(local_path, sha256)
    if data:
        return data

    if os.getenv("FLASHINFER_NO_DOWNLOAD"):
        raise RuntimeError(
            f"Artifact not found locally: {file_name} "
            f"(looked at {local_path}). "
            f"FLASHINFER_NO_DOWNLOAD is set — refusing to download. "
            f"This means flashinfer-cubin is missing this file."
        )

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    uri = safe_urljoin(FLASHINFER_CUBINS_REPOSITORY, file_name)
    logger.info(f"Fetching {file_name} from {uri}")
    download_file(uri, local_path, session=session)
    return load_cubin(local_path, sha256)


# Backward-compatible alias
get_cubin = get_artifact


def compile_source_cubins(
    source_dir: Union[str, pathlib.Path],
    artifact_path: str,
    include_paths: List[Union[str, pathlib.Path]],
    nvcc_flags: List[str],
) -> None:
    """
    Compile .cu source files to .cubin and place them in the cubin cache
    directory where getCubin() will find them at runtime.

    This supports the hybrid cubin model where some trtllm-gen kernels are
    shipped as source (.cu files in the repo) rather than pre-built cubins
    downloaded from artifactory.

    Parameters:
    - source_dir: Directory containing .cu source files
    - artifact_path: Artifact subdirectory (e.g. ArtifactPath.TRTLLM_GEN_BMM),
      used to place output cubins where getCubin() expects them
    - include_paths: Include directories for nvcc compilation
    - nvcc_flags: Flags passed to nvcc for compilation
    """
    source_dir = pathlib.Path(source_dir)
    if not source_dir.exists():
        logger.debug(f"Source cubin directory does not exist, skipping: {source_dir}")
        return

    cu_files = sorted(source_dir.glob("*.cu"))
    if not cu_files:
        logger.debug(f"No .cu files found in {source_dir}, skipping")
        return

    # Debug flags for development/testing
    _DBG_KERNEL_FILTER = (
        None  # None to compile all, or list of substrings (all must match)
    )
    _DBG_FORCE_RECOMPILE = False  # True to skip cache check and always recompile
    _DBG_NPROC = 0  # 0 = auto (cpu_count), 1 = sequential, >1 = parallel with N workers
    if _DBG_KERNEL_FILTER:
        cu_files = [f for f in cu_files if all(s in f.name for s in _DBG_KERNEL_FILTER)]
        logger.debug(
            f"_DBG_KERNEL_FILTER active: {_DBG_KERNEL_FILTER} -> {len(cu_files)} files"
        )
    if _DBG_FORCE_RECOMPILE:
        logger.debug("_DBG_FORCE_RECOMPILE enabled")

    logger.info(f"compile_source_cubins: {len(cu_files)} .cu files to process")

    from .cpp_ext import get_cuda_path

    cuda_home = get_cuda_path()
    nvcc = os.environ.get("FLASHINFER_NVCC", os.path.join(cuda_home, "bin", "nvcc"))

    # Output directory: same location as downloaded cubins
    output_dir = FLASHINFER_CUBIN_DIR / artifact_path
    output_dir.mkdir(parents=True, exist_ok=True)

    include_flags = []
    for p in include_paths:
        include_flags.extend(["-I", str(p)])

    # Gather files to compile vs skip
    to_compile = []
    skipped_count = 0
    for cu_file in cu_files:
        kernel_name = cu_file.stem
        cubin_file = output_dir / f"{kernel_name}.cubin"
        cubin_rel_path = f"{artifact_path}/{kernel_name}.cubin"

        # Skip compilation if cubin is newer than source (unless force recompile)
        if (
            not _DBG_FORCE_RECOMPILE
            and cubin_file.exists()
            and cubin_file.stat().st_mtime > cu_file.stat().st_mtime
        ):
            logger.debug(f"Cached source cubin up-to-date: {cubin_rel_path}")
            _source_compiled_cubins.add(cubin_rel_path)
            skipped_count += 1
        else:
            to_compile.append((cu_file, cubin_file, cubin_rel_path))

    if not to_compile:
        logger.info(f"compile_source_cubins: 0 compiled, {skipped_count} cached")
        return

    # Helper function to compile a single file
    def compile_one(args):
        cu_file, cubin_file, cubin_rel_path = args
        cmd = (
            [nvcc]
            + nvcc_flags
            + include_flags
            + ["-o", str(cubin_file), "-c", str(cu_file)]
        )
        logger.debug(f"Compiling: {cu_file.name}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return (False, cu_file.name, result.stdout, result.stderr)
        return (True, cubin_rel_path, None, None)

    # Determine parallelism
    import multiprocessing

    nproc = _DBG_NPROC if _DBG_NPROC > 0 else multiprocessing.cpu_count()

    if nproc <= 1:
        # Sequential compilation
        logger.info(
            f"compile_source_cubins: compiling {len(to_compile)} files sequentially"
        )
        for args in to_compile:
            success, name, stdout, stderr = compile_one(args)
            if not success:
                raise RuntimeError(
                    f"Failed to compile {name} to cubin:\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            _source_compiled_cubins.add(name)
    else:
        # Parallel compilation with progress logging
        logger.info(
            f"compile_source_cubins: compiling {len(to_compile)} files with {nproc} workers"
        )
        from concurrent.futures import ThreadPoolExecutor
        import threading

        completed_count = 0
        results = []
        lock = threading.Lock()

        def compile_and_track(args):
            nonlocal completed_count
            result = compile_one(args)
            with lock:
                completed_count += 1
            return result

        with ThreadPoolExecutor(max_workers=nproc) as executor:
            futures = [executor.submit(compile_and_track, args) for args in to_compile]
            total = len(futures)

            # Progress logging in main thread
            last_logged = 0
            while True:
                done_count = sum(1 for f in futures if f.done())
                if done_count > last_logged:
                    logger.info(
                        f"compile_source_cubins: ({done_count}/{total}) compiled"
                    )
                    last_logged = done_count
                if done_count == total:
                    break
                time.sleep(3)

            # Collect results and check for errors
            for future in futures:
                results.append(future.result())

        for success, name, stdout, stderr in results:
            if not success:
                raise RuntimeError(
                    f"Failed to compile {name} to cubin:\n"
                    f"stdout: {stdout}\nstderr: {stderr}"
                )
            _source_compiled_cubins.add(name)

    logger.info(
        f"compile_source_cubins: {len(to_compile)} compiled, {skipped_count} cached"
    )


def ensure_symlink(
    link: Union[str, pathlib.Path], target: Union[str, pathlib.Path]
) -> None:
    """Create or update a symlink, removing any stale file/directory at *link*.

    This is used to map C++ include paths (e.g.
    ``CUBIN_DIR/flashinfer/trtllm/batched_gemm/trtllmGen_bmm_export``) to the
    canonical artifact directory where ``get_artifact()`` stores downloaded files.
    """
    link = pathlib.Path(link)
    target = pathlib.Path(target)

    link.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(link) + ".lock"
    lock = filelock.FileLock(lock_path, timeout=60)
    with lock:
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.resolve() == target.resolve():
                return  # already correct
            # Stale symlink or directory from a previous version; remove it.
            if link.is_symlink() or link.is_file():
                link.unlink()
            else:
                shutil.rmtree(link)
        link.symlink_to(target)


def verify_symlinked_headers(
    symlink_path: Union[str, pathlib.Path],
    headers: list,
    checksums: bytes,
) -> None:
    """Verify that headers accessible through the symlink match expected checksums.

    This catches stale cached headers after e.g. ``git checkout`` to a branch
    with a different artifact version.  Each header file is read through the
    symlink and its SHA-256 is compared against the hash from ``checksums.txt``.
    """
    symlink_path = pathlib.Path(symlink_path)
    for header in headers:
        header_path = symlink_path / header
        expected_hash = get_meta_hash(checksums, header)
        if not header_path.exists():
            raise RuntimeError(
                f"Header {header} not found at {header_path}. "
                f"Try clearing the cache: rm -rf {FLASHINFER_CUBIN_DIR}"
            )
        actual_hash = hashlib.sha256(header_path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Header {header} at {header_path} has wrong checksum "
                f"(expected {expected_hash}, got {actual_hash}). "
                f"This can happen after switching branches. "
                f"Try clearing the cache: rm -rf {FLASHINFER_CUBIN_DIR}"
            )


def download_cuda_ptx_header(artifact_path: str, sha256: str) -> pathlib.Path:
    """
    Download cuda_ptx.h from artifactory into the cubin cache directory.

    The file is downloaded via ``get_artifact()`` and then symlinked so that
    ``#include <cuda_ptx/cuda_ptx.h>`` resolves when FLASHINFER_CUBIN_DIR is
    on the include path.

    Returns the include-root directory (FLASHINFER_CUBIN_DIR).
    """
    file_name = f"{artifact_path}/cuda_ptx.h"
    result = get_artifact(file_name, sha256)
    assert result, "cuda_ptx.h not found"

    # ``#include <cuda_ptx/cuda_ptx.h>`` needs a ``cuda_ptx/`` directory
    # under the include root.  The downloaded file lives under the
    # artifact_path subdirectory (e.g. ``<hash>/cuda_ptx-<hash>/``),
    # so create a symlink ``cuda_ptx -> <artifact_path>`` if needed.
    canonical_dir = FLASHINFER_CUBIN_DIR / "cuda_ptx"
    source_dir = FLASHINFER_CUBIN_DIR / artifact_path
    ensure_symlink(canonical_dir, source_dir)

    return FLASHINFER_CUBIN_DIR


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
        cubin = get_artifact(name.decode("utf-8"), sha256.decode("utf-8"))
        _LIB.FlashInferSetCurrentCubin(
            convert_to_ctypes_char_p(cubin), ctypes.c_int(len(cubin))
        )

    # Create the callback and keep a reference to prevent GC
    cb = CALLBACK_TYPE(get_cubin_callback)
    dll_cubin_handlers[dll_path] = cb

    _LIB.FlashInferSetCubinCallback(cb)
