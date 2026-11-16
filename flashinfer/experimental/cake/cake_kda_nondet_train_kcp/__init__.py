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

# Experimental Cake backend: non-deterministic context-parallel (KCP) packed KDA (Kimi Delta Attention) chunked
# training on SM100 / SM103 / SM107.  The public entry points are in ``flashinfer.cake_kda_nondet_train_kcp``
# (``cake_kda_nondet_kcp``, ``cake_kda_nondet_train_kcp_forward``, ``cake_kda_nondet_train_kcp_backward``,
# ``cake_kda_nondet_train_kcp_partition``, ``cake_kda_nondet_train_kcp_build_context``, the byte queries, the
# checkpoint hand-off's ``cake_kda_nondet_train_kcp_handoff_bytes`` / ``cake_kda_nondet_train_kcp_handoff_stats``);
# the per-rank launch sequences and the autograd Function (``cake_kcp``), the partition and context plan
# (``cake_kcp_plan``), the registry of the KCP programs (``cake_kcp_jit``) and their generated sources live here.
# The chunk stages a rank runs at its own shape are those of ``flashinfer.experimental.cake.cake_kda_nondet_train``.
# The names below resolve on first access, so importing the package imports neither the host module
# (``torch.distributed``) nor the loader.

from importlib import import_module
from typing import Any

# public name -> the submodule that defines it
_SUBMODULE_OF = {
    "CakeKDANondetKcpCall": "cake_kcp",
    "CakeKDANondetKcpContext": "cake_kcp_plan",
    "CakeKDANondetKcpFragment": "cake_kcp_plan",
    "CakeKDANondetKcpFunction": "cake_kcp",
    "CakeKDANondetKcpOptions": "cake_kcp",
    "CakeKDANondetKcpPartition": "cake_kcp_plan",
    "CakeKDANondetKcpRecomputeHandle": "cake_kcp",
    "CakeKDANondetKcpSaved": "cake_kcp",
    "CakeKDANondetKcpWorkspace": "cake_kcp",
    "cake_kda_nondet_kcp": "cake_kcp",
    "cake_kda_nondet_train_kcp_backward": "cake_kcp",
    "cake_kda_nondet_train_kcp_build_context": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_comm_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_forward": "cake_kcp",
    "cake_kda_nondet_train_kcp_handoff_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_handoff_stats": "cake_kcp",
    "cake_kda_nondet_train_kcp_partition": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_reduce_gate_param_grads": "cake_kcp",
    "cake_kda_nondet_train_kcp_saved_bytes": "cake_kcp_plan",
    "cake_kda_nondet_train_kcp_workspace_bytes": "cake_kcp_plan",
}


def __getattr__(name: str) -> Any:
    submodule = _SUBMODULE_OF.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f"{__name__}.{submodule}"), name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_SUBMODULE_OF))


__all__ = [
    "CakeKDANondetKcpCall",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpContext",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpFragment",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpFunction",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpOptions",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpPartition",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpRecomputeHandle",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpSaved",  # noqa: F822 - resolved by __getattr__
    "CakeKDANondetKcpWorkspace",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_kcp",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_backward",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_build_context",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_comm_bytes",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_forward",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_handoff_bytes",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_handoff_stats",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_partition",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_reduce_gate_param_grads",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_saved_bytes",  # noqa: F822 - resolved by __getattr__
    "cake_kda_nondet_train_kcp_workspace_bytes",  # noqa: F822 - resolved by __getattr__
]
