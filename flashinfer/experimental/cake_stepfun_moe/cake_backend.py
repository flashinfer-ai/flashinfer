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

Cake backend: the StepFun fused MoE on exact SM100 / SM103 (cake-build/1 form).

The forward is a sequence of exported Cake kernels -- routing (Renormalize
top-k, one or two launches), FC1 with the StepFun epilogue
``clamp(up, -L, L) * min(silu(gate), L)``, the NVFP4 per-token requantization
(per-token family only), FC2 and finalize -- each a program of ``cake_jit``
reached through its generated TVM-FFI binding.  This module owns what the
trtllm-gen C++ stage runners owned: family and unit selection, the shape
legality of a (family, tile), the host plan (``max_num_ctas``, grids, the
routing window, the finalize variant), the buffer set, the keyword-to-kernel
argument binding and the ordered launch sequence.  The bindings own the
tensor checks, the TMA descriptors, the dynamic-smem opt-in and the cluster /
cooperative / PDL launch attributes.

A prepared launch performs no allocation and no host synchronisation and is
CUDA-graph capturable; the caller runs it once eagerly before capturing.
Nothing here reads the environment: every switch is a keyword, resolved by
the ``flashinfer.fused_moe`` facade.

Scaffold state: the registry literals of ``cake_jit`` are empty until the
generated-program export delivers them, so the host plan and the prepared
launches land with that delivery; the device helpers below work on any
checkout.
"""

from __future__ import annotations

import functools

import torch

from .cake_jit import PROGRAMS

SUPPORTED_COMPUTE_CAPABILITIES: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------


def _device_index(device: torch.device) -> int:
    index = device.index
    return torch.cuda.current_device() if index is None else int(index)


@functools.cache
def _device_facts(index: int) -> tuple[tuple[int, int], int]:
    """(compute capability, SM count) of CUDA device ``index``: queried once per process."""
    capability = torch.cuda.get_device_capability(index)
    properties = torch.cuda.get_device_properties(index)
    return (int(capability[0]), int(capability[1])), int(
        properties.multi_processor_count
    )


def arch_for(device: torch.device) -> str:
    """Exact architecture name of ``device`` (``sm_100a`` or ``sm_103a``).

    Raises ``NotImplementedError`` naming the compute capability for any other
    device: the StepFun programs are exported for these two targets only and
    nothing falls back.
    """
    capability, _ = _device_facts(_device_index(device))
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise NotImplementedError(
            "the Cake StepFun fused-MoE programs are exported for exact SM100 and "
            f"SM103 only, got SM{capability[0]}{capability[1]} (device {device})"
        )
    return arch


def sm_count(device: torch.device) -> int:
    """Streaming-multiprocessor count of ``device`` (the cooperative routing grid base)."""
    return _device_facts(_device_index(device))[1]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _require_registry() -> None:
    if not PROGRAMS:
        raise NotImplementedError(
            "The generated StepFun fused-MoE programs are not registered in this "
            "checkout yet (empty PROGRAMS in "
            "flashinfer.experimental.cake_stepfun_moe.cake_jit; the registry is "
            "filled by the Cake generated-program export)"
        )
