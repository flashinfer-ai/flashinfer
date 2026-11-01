"""
Copyright (c) 2026 by FlashInfer team.

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

# Experimental Cake backend: non-deterministic (``deterministic=False``) context-parallel (KCP) packed KDA (Kimi
# Delta Attention) chunked training on SM100 / SM103 / SM107, the counterpart of the deterministic package
# ``flashinfer.experimental.cake_kda_det_train_kcp``.  The public entry points are in ``flashinfer.cake_kda_train_kcp``
# (``chunk_kda_kcp``, ``chunk_kda_fwd_kcp``, ``chunk_kda_bwd_kcp``, ``partition_kcp``, ``build_context_kcp``, the
# byte queries; ``deterministic=False`` selects this package); the per-rank launch sequences and the autograd
# Function (``cake_kcp``), the partition and context plan (``cake_kcp_plan``), the registry of the KCP programs
# (``cake_kcp_jit``) and their generated sources live here.  The chunk stages a rank runs at its own shape are
# those of the non-deterministic single-rank package ``flashinfer.experimental.cake.cake_kda_train``; this package
# imports no module of either deterministic package.  The names below resolve on first access, so importing the
# package imports neither the host module (``torch.distributed``) nor the loader.

from importlib import import_module
from typing import Any

# public name -> the submodule that defines it
_SUBMODULE_OF = {
    "ChunkKDAFunctionKcp": "cake_kcp",
    "CallKcp": "cake_kcp",
    "OptionsKcp": "cake_kcp",
    "RecomputeHandleKcp": "cake_kcp",
    "SavedKcp": "cake_kcp",
    "WorkspaceKcp": "cake_kcp",
    "chunk_kda_bwd_kcp": "cake_kcp",
    "chunk_kda_fwd_kcp": "cake_kcp",
    "chunk_kda_kcp": "cake_kcp",
    "ContextKcp": "cake_kcp_plan",
    "FragmentKcp": "cake_kcp_plan",
    "PartitionKcp": "cake_kcp_plan",
    "build_context_kcp": "cake_kcp_plan",
    "kda_comm_bytes_kcp": "cake_kcp_plan",
    "kda_saved_bytes_kcp": "cake_kcp_plan",
    "kda_workspace_bytes_kcp": "cake_kcp_plan",
    "partition_kcp": "cake_kcp_plan",
}


def __getattr__(name: str) -> Any:
    submodule = _SUBMODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{submodule}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULE_OF))


__all__ = [
    "CallKcp",  # noqa: F822 - resolved by __getattr__
    "ChunkKDAFunctionKcp",  # noqa: F822 - resolved by __getattr__
    "ContextKcp",  # noqa: F822 - resolved by __getattr__
    "FragmentKcp",  # noqa: F822 - resolved by __getattr__
    "OptionsKcp",  # noqa: F822 - resolved by __getattr__
    "PartitionKcp",  # noqa: F822 - resolved by __getattr__
    "RecomputeHandleKcp",  # noqa: F822 - resolved by __getattr__
    "SavedKcp",  # noqa: F822 - resolved by __getattr__
    "WorkspaceKcp",  # noqa: F822 - resolved by __getattr__
    "build_context_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_bwd_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_fwd_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_comm_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_saved_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_workspace_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "partition_kcp",  # noqa: F822 - resolved by __getattr__
]
