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

# Experimental Cake backend: deterministic context-parallel (KCP) packed KDA (Kimi Delta Attention) chunked
# training on SM100 / SM103 / SM107.  The public entry points are in ``flashinfer.cake_kda_det_train_kcp``
# (``chunk_kda_det_kcp``, ``chunk_kda_det_fwd_kcp``, ``chunk_kda_det_bwd_kcp``, ``partition_det_kcp``,
# ``build_context_det_kcp``, the byte queries); the per-rank launch sequences and the autograd Function
# (``cake_kcp``), the partition and context plan (``cake_kcp_plan``), the registry of the KCP programs
# (``cake_kcp_jit``) and their generated sources live here.  The chunk stages a rank runs at its own shape are
# those of ``flashinfer.experimental.cake_kda_det_train``.  The names below resolve on first access, so importing
# the package imports neither the host module (``torch.distributed``) nor the loader.  The names before the det
# rename (``KcpContext``, ``chunk_kda_kcp``, ...) stay bound to the same objects in ``cake_kcp`` and
# ``cake_kcp_plan`` and are not in ``__all__``.

from importlib import import_module
from typing import Any

# public name -> the submodule that defines it
_SUBMODULE_OF = {
    "ChunkKDADetFunctionKcp": "cake_kcp",
    "CallDetKcp": "cake_kcp",
    "OptionsDetKcp": "cake_kcp",
    "RecomputeHandleDetKcp": "cake_kcp",
    "SavedDetKcp": "cake_kcp",
    "WorkspaceDetKcp": "cake_kcp",
    "chunk_kda_det_bwd_kcp": "cake_kcp",
    "chunk_kda_det_fwd_kcp": "cake_kcp",
    "chunk_kda_det_kcp": "cake_kcp",
    "reduce_gate_param_grads_det_kcp": "cake_kcp",
    "ContextDetKcp": "cake_kcp_plan",
    "FragmentDetKcp": "cake_kcp_plan",
    "PartitionDetKcp": "cake_kcp_plan",
    "build_context_det_kcp": "cake_kcp_plan",
    "kda_det_comm_bytes_kcp": "cake_kcp_plan",
    "kda_det_saved_bytes_kcp": "cake_kcp_plan",
    "kda_det_workspace_bytes_kcp": "cake_kcp_plan",
    "partition_det_kcp": "cake_kcp_plan",
}


def __getattr__(name: str) -> Any:
    submodule = _SUBMODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{submodule}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULE_OF))


__all__ = [
    "CallDetKcp",  # noqa: F822 - resolved by __getattr__
    "ChunkKDADetFunctionKcp",  # noqa: F822 - resolved by __getattr__
    "ContextDetKcp",  # noqa: F822 - resolved by __getattr__
    "FragmentDetKcp",  # noqa: F822 - resolved by __getattr__
    "OptionsDetKcp",  # noqa: F822 - resolved by __getattr__
    "PartitionDetKcp",  # noqa: F822 - resolved by __getattr__
    "RecomputeHandleDetKcp",  # noqa: F822 - resolved by __getattr__
    "SavedDetKcp",  # noqa: F822 - resolved by __getattr__
    "WorkspaceDetKcp",  # noqa: F822 - resolved by __getattr__
    "build_context_det_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_det_bwd_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_det_fwd_kcp",  # noqa: F822 - resolved by __getattr__
    "chunk_kda_det_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_det_comm_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_det_saved_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "kda_det_workspace_bytes_kcp",  # noqa: F822 - resolved by __getattr__
    "partition_det_kcp",  # noqa: F822 - resolved by __getattr__
    "reduce_gate_param_grads_det_kcp",  # noqa: F822 - resolved by __getattr__
]
