"""Strict compatibility guard for MegaMoE's NVSHMEM hybrid shim.

NVSHMEM 3.7 does not expose a public per-peer transport override.  MegaMoE's
hybrid path therefore needs one small device helper that clears a peer's P2P
mapping before the first RMA.  This module fail-closes on every ABI input and
compiles that helper against the headers installed next to the selected
NVSHMEM device bitcode.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Dict, Tuple

import nvshmem.core
from cuda.pathfinder import find_nvidia_header_directory


_SUPPORTED_NVSHMEM_VERSION = (3, 7, 0)
_SUPPORTED_DEVICE_BITCODE_SHA256 = {
    # NVSHMEM 3.7.0 wheel in cutedsl_460_public.sqsh.
    "a434965d5d6b08b64589c16621d4e5648257e426bb7863f3d5d482108f7d17e0",
}
_VERSION_MACROS = (
    "NVSHMEM_VENDOR_MAJOR_VERSION",
    "NVSHMEM_VENDOR_MINOR_VERSION",
    "NVSHMEM_VENDOR_PATCH_VERSION",
)


@dataclass(frozen=True)
class NvshmemCompatArtifacts:
    device_bitcode: str
    helper_bitcode: str
    include_dir: str
    version: Tuple[int, int, int]
    device_bitcode_sha256: str

    @property
    def link_libraries(self) -> str:
        return f"{self.device_bitcode},{self.helper_bitcode}"


def find_nvshmem_device_bitcode() -> str:
    """Resolve the selected device bitcode without accepting unknown ABI."""

    explicit = os.environ.get("NVSHMEM_DEVICE_BITCODE")
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(
                f"NVSHMEM_DEVICE_BITCODE does not exist: {path}"
            )
        return str(path)
    try:
        return nvshmem.core.find_device_bitcode_library()
    except Exception as first_error:
        include_dir = Path(find_nvidia_header_directory("nvshmem"))
        candidates = (
            include_dir.parent / "lib/libnvshmem_device_sm_90.bc",
            include_dir.parent / "lib/libnvshmem_device.bc",
            Path("/usr/local/cuda/targets/x86_64-linux/lib/libnvshmem_device.bc"),
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        raise RuntimeError(
            "Unable to locate NVSHMEM device bitcode; set "
            "NVSHMEM_DEVICE_BITCODE explicitly."
        ) from first_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _header_version(include_dir: Path) -> Tuple[int, int, int]:
    version_header = include_dir / "non_abi/nvshmem_version.h"
    text = version_header.read_text(encoding="utf-8")
    values: Dict[str, int] = {}
    for name in _VERSION_MACROS:
        match = re.search(rf"^#define\s+{name}\s+(\d+)\s*$", text, re.MULTILINE)
        if match is None:
            raise RuntimeError(f"missing {name} in {version_header}")
        values[name] = int(match.group(1))
    return tuple(values[name] for name in _VERSION_MACROS)


def _library_version() -> Tuple[int, int, int]:
    version = nvshmem.core.get_version().libnvshmem_version
    try:
        parsed = tuple(int(part) for part in version.split("."))
    except ValueError as exc:
        raise RuntimeError(f"unparseable NVSHMEM library version {version!r}") from exc
    if len(parsed) != 3:
        raise RuntimeError(f"unparseable NVSHMEM library version {version!r}")
    return parsed


def _compile_helper(include_dir: Path, cache_key: str) -> Path:
    source = Path(__file__).with_name("nvshmem_compat.cu")
    if not source.is_file():
        raise RuntimeError(f"NVSHMEM compatibility source is missing: {source}")
    compiler = shutil.which(os.environ.get("MEGAMOE_NVSHMEM_CLANG", "clang++"))
    if compiler is None:
        raise RuntimeError(
            "clang++ is required to compile the NVSHMEM compatibility helper"
        )
    cache_root = Path(
        os.environ.get(
            "MEGAMOE_NVSHMEM_COMPAT_CACHE",
            Path.home() / ".cache/megamoe/nvshmem_compat",
        )
    )
    cache_root.mkdir(parents=True, exist_ok=True)
    output = cache_root / f"nvshmem_compat_{cache_key}.bc"
    if output.is_file():
        return output
    temporary = output.with_suffix(f".tmp.{os.getpid()}.bc")
    command = [
        compiler,
        "-c",
        "-emit-llvm",
        "-O2",
        "-x",
        "c++",
        "-target",
        "nvptx64-nvidia-cuda",
        "-march=sm_90",
        "-D__x86_64__=1",
        "-isystem",
        "/usr/include/x86_64-linux-gnu",
        f"-I{include_dir}",
        str(source),
        "-o",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        os.replace(temporary, output)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "failed to compile the NVSHMEM compatibility helper against the "
            f"installed headers:\n{exc.stderr}"
        ) from exc
    return output


def prepare_nvshmem_hybrid_compat(
    device_bitcode: str | None = None,
) -> NvshmemCompatArtifacts:
    """Validate library/header/bitcode identity, then build the helper.

    Any mismatch disables hybrid transport by raising before NVSHMEM init or
    any device-state write can occur.  There is deliberately no unsafe
    override.
    """

    include_dir = Path(find_nvidia_header_directory("nvshmem")).resolve()
    header_version = _header_version(include_dir)
    library_version = _library_version()
    if header_version != _SUPPORTED_NVSHMEM_VERSION:
        raise RuntimeError(
            "NVSHMEM hybrid compatibility helper is not validated for header "
            f"version {header_version}; expected {_SUPPORTED_NVSHMEM_VERSION}"
        )
    if library_version != header_version:
        raise RuntimeError(
            "NVSHMEM host library/header version mismatch: "
            f"library={library_version}, headers={header_version}"
        )

    bitcode_path = Path(
        device_bitcode or find_nvshmem_device_bitcode()
    ).expanduser().resolve()
    bitcode_hash = _sha256(bitcode_path)
    if bitcode_hash not in _SUPPORTED_DEVICE_BITCODE_SHA256:
        raise RuntimeError(
            "NVSHMEM hybrid compatibility helper refuses unvalidated device "
            f"bitcode {bitcode_path} (sha256={bitcode_hash}). Update the "
            "version/hash/ABI assertions together before enabling hybrid."
        )

    source_hash = _sha256(Path(__file__).with_name("nvshmem_compat.cu"))
    cache_key = f"{'.'.join(map(str, header_version))}_{bitcode_hash[:16]}_{source_hash[:16]}"
    helper = _compile_helper(include_dir, cache_key)
    return NvshmemCompatArtifacts(
        device_bitcode=str(bitcode_path),
        helper_bitcode=str(helper),
        include_dir=str(include_dir),
        version=header_version,
        device_bitcode_sha256=bitcode_hash,
    )


__all__ = [
    "NvshmemCompatArtifacts",
    "find_nvshmem_device_bitcode",
    "prepare_nvshmem_hybrid_compat",
]
