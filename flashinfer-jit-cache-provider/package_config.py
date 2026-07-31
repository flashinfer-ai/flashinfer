"""Build configuration shared by the jit-cache provider backend and setup.py."""

import os
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROVIDER_SOURCE_DIR = Path(__file__).resolve().parent / "flashinfer_jit_cache_provider"


def normalize_cuda_architecture(architecture: str) -> tuple[str, str]:
    """Return ``(nvcc_arch, provider_tag)`` for a CUDA architecture spelling."""
    normalized = architecture.strip().lower()
    for prefix in ("compute_", "sm_", "sm"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break

    dotted_match = re.fullmatch(r"(\d{1,2})\.(\d)([af]?)", normalized)
    compact_match = re.fullmatch(r"(\d{2,3})([af]?)", normalized)
    if dotted_match is not None:
        major = int(dotted_match.group(1))
        minor = int(dotted_match.group(2))
        suffix = dotted_match.group(3)
    elif compact_match is not None:
        digits, suffix = compact_match.groups()
        major = int(digits[:-1])
        minor = int(digits[-1])
    else:
        raise ValueError(
            f"Invalid FLASHINFER_JIT_CACHE_PROVIDER_ARCH={architecture!r}; "
            "use a value such as '8.0', '9.0a', or 'sm120f'"
        )

    nvcc_arch = f"{major}.{minor}{suffix}"
    provider_tag = f"sm{major}{minor}{suffix}"
    return nvcc_arch, provider_tag


def get_package_version() -> str:
    version_file = PROJECT_ROOT / "version.txt"
    version = version_file.read_text().strip() if version_file.exists() else "0.0.0"

    dev_suffix = os.environ.get("FLASHINFER_DEV_RELEASE_SUFFIX", "").strip()
    if dev_suffix:
        version = f"{version}.dev{dev_suffix}"

    local_version = os.environ.get("FLASHINFER_LOCAL_VERSION", "").strip()
    if local_version:
        version = f"{version}+{local_version}"
    return version


@dataclass(frozen=True)
class ProviderBuildConfig:
    cuda_architecture: str
    provider_tag: str
    distribution: str
    package: str
    version: str


def get_provider_build_config() -> ProviderBuildConfig:
    architecture = os.environ.get("FLASHINFER_JIT_CACHE_PROVIDER_ARCH", "").strip()
    if not architecture:
        raise RuntimeError(
            "FLASHINFER_JIT_CACHE_PROVIDER_ARCH must select exactly one provider "
            "architecture, for example '8.0' or '12.0f'"
        )

    cuda_architecture, provider_tag = normalize_cuda_architecture(architecture)
    return ProviderBuildConfig(
        cuda_architecture=cuda_architecture,
        provider_tag=provider_tag,
        distribution=f"flashinfer-jit-cache-{provider_tag}",
        package=f"flashinfer_jit_cache.providers.{provider_tag}",
        version=get_package_version(),
    )
