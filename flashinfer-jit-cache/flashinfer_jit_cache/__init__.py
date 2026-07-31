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

import importlib.metadata
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, FrozenSet, Mapping, Tuple


logger = logging.getLogger(__name__)

JIT_CACHE_PROVIDER_ENTRY_POINT_GROUP = "flashinfer.jit_cache.providers"
JIT_CACHE_PROVIDER_SCHEMA_VERSION = 1

# Get the path to the AOT modules directory within this package
jit_cache_dir = Path(__file__).parent / "jit_cache"


def _normalize_cuda_architecture(architecture: str) -> str:
    """Normalize CUDA architecture spellings to names such as ``sm90a``."""
    normalized = architecture.strip().lower()
    for prefix in ("compute_", "sm_", "sm"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    normalized = normalized.replace(".", "").replace("_", "")
    if not re.fullmatch(r"\d{2,3}[af]?", normalized):
        raise ValueError(f"Invalid CUDA architecture {architecture!r}")
    return f"sm{normalized}"


@dataclass(frozen=True)
class JitCacheProvider:
    """Description of one installed jit-cache binary provider."""

    provider_id: str
    distribution: str
    version: str
    jit_cache_dir: Path
    cuda_architectures: FrozenSet[str]
    modules: FrozenSet[str]


def _provider_from_mapping(
    provider_id: str, manifest: Mapping[str, Any]
) -> JitCacheProvider:
    schema_version = manifest.get("schema_version")
    if schema_version != JIT_CACHE_PROVIDER_SCHEMA_VERSION:
        raise ValueError(
            f"Provider {provider_id!r} uses unsupported manifest schema "
            f"{schema_version!r}; expected {JIT_CACHE_PROVIDER_SCHEMA_VERSION}"
        )

    required_fields = (
        "distribution",
        "version",
        "jit_cache_dir",
        "cuda_architectures",
        "modules",
    )
    missing_fields = [field for field in required_fields if field not in manifest]
    if missing_fields:
        raise ValueError(
            f"Provider {provider_id!r} is missing fields: {', '.join(missing_fields)}"
        )

    cache_dir = Path(str(manifest["jit_cache_dir"])).resolve()
    cuda_architectures = frozenset(
        _normalize_cuda_architecture(str(architecture))
        for architecture in manifest["cuda_architectures"]
    )
    modules = frozenset(str(module) for module in manifest["modules"])
    if not cuda_architectures:
        raise ValueError(f"Provider {provider_id!r} has no CUDA architectures")
    if not modules:
        raise ValueError(f"Provider {provider_id!r} has no modules")

    return JitCacheProvider(
        provider_id=provider_id,
        distribution=str(manifest["distribution"]),
        version=str(manifest["version"]),
        jit_cache_dir=cache_dir,
        cuda_architectures=cuda_architectures,
        modules=modules,
    )


def _provider_entry_points() -> Tuple[importlib.metadata.EntryPoint, ...]:
    entry_points = importlib.metadata.entry_points()
    if hasattr(entry_points, "select"):
        return tuple(entry_points.select(group=JIT_CACHE_PROVIDER_ENTRY_POINT_GROUP))
    return tuple(entry_points.get(JIT_CACHE_PROVIDER_ENTRY_POINT_GROUP, ()))


def get_jit_cache_providers() -> Tuple[JitCacheProvider, ...]:
    """Discover installed binary providers registered through package metadata."""
    providers = []
    for entry_point in _provider_entry_points():
        try:
            factory = entry_point.load()
            manifest = factory()
            if not isinstance(manifest, Mapping):
                raise TypeError("provider factory must return a mapping")
            providers.append(_provider_from_mapping(entry_point.name, manifest))
        except Exception as error:
            logger.warning(
                "Ignoring invalid flashinfer jit-cache provider %s: %s",
                entry_point.name,
                error,
            )
    return tuple(sorted(providers, key=lambda provider: provider.provider_id))


def get_jit_cache_dir() -> str:
    """Get the directory containing the AOT compiled modules."""
    return str(jit_cache_dir)


try:
    from ._build_meta import __version__ as __version__
    from ._build_meta import __git_version__ as __git_version__
except ModuleNotFoundError:
    __version__ = "0.0.0+unknown"
    __git_version__ = "unknown"


__all__ = [
    "JIT_CACHE_PROVIDER_ENTRY_POINT_GROUP",
    "JIT_CACHE_PROVIDER_SCHEMA_VERSION",
    "JitCacheProvider",
    "get_jit_cache_dir",
    "get_jit_cache_providers",
]
