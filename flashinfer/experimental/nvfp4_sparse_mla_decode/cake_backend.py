"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Generated-program backend: NVFP4 sparse MLA decode (SM100 / SM103).

``backend="cake"`` of :func:`flashinfer.mla.nvfp4_sparse_mla_decode`: the same
operator and tensor ABI as the hand-written ``backend="cuda"`` kernel (E4M3
``[T, 16, 576]`` queries, 352-byte ``nvfp4_ds_mla`` cache rows, int32
``[T, topk]`` row ids with ``-1`` marking an empty slot, BF16 ``[T, 16, 512]``
output), served by a family of generated kernels.  One thread-block cluster of
``C`` CTAs serves one query token (grid ``(T, C, 1)``, cluster ``(1, C, 1)``):
CTA ``c`` owns the keys ``[c * keys_per_cta, (c + 1) * keys_per_cta)``, walks
them in 32-key stages with 8 math warps, 4 softmax warps and 2 loader warps,
and either writes the normalized output directly (``C = 1``) or pushes its
fp32 partial ``(O, m, l)`` through distributed shared memory to the cluster
peers that own the 8-dim output groups, which merge and store.  ``C`` is one of
1, 2, 3, 4, 5, 6 or 8 -- one physical module per (architecture, ``C``) -- and
is planned per launch from the driver's one-wave cluster capacity unless the
caller forces it with ``num_ctas_per_token``.  Nothing is allocated at launch
(only the optional output), so a prepared runner is CUDA Graph safe.  See
``README.md`` in this package.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import torch
import tvm_ffi

from .backend import NUM_HEADS, V_HEAD_DIM, _check_inputs
from .cake_jit import (
    ARCHES,
    CLUSTER_SIZES,
    MODULES,
    load_nvfp4_sparse_mla_decode_cake_module,
    registered_clusters,
    select_module,
)

__all__ = [
    "CLUSTER_SIZES",
    "MAX_KEYS_PER_CTA",
    "NVFP4SparseMLADecodeCakeRunner",
    "PLAN_CTAS",
    "SMEM_BYTES",
    "SMEM_BYTES_CLUSTER",
    "STAGE_KEYS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "THREADS",
    "cluster_capacity_for",
    "generated_program_available",
    "is_valid_split",
    "keys_per_cta",
    "nvfp4_sparse_mla_decode",
    "plan_ctas",
    "prepare_nvfp4_sparse_mla_decode",
]

SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
assert set(SUPPORTED_COMPUTE_CAPABILITIES.values()) == set(ARCHES)

# Generated-kernel constants (copied from the program that the export renders;
# the registry's launch records must agree, see ``_check_launch_record``).
THREADS = 448  # 8 math warps + 4 softmax warps + 2 loader warps
STAGE_KEYS = 32  # keys per pipeline stage; keys_per_cta is a multiple of it
MAX_KEYS_PER_CTA = 2048  # entries of the per-CTA shared-memory index table
SMEM_BYTES = 192256  # dynamic shared memory of the generated program, cluster size 1 (pool + mbarriers)
SMEM_BYTES_CLUSTER = 226816  # dynamic shared memory of the generated program, cluster size > 1
# Split sizes the automatic plan considers, largest first; 1 is the fallback
# when no split leaves every CTA a full stage.  7 is not a variant of the program.
PLAN_CTAS = (8, 6, 5, 4, 3, 2)
assert tuple(sorted((1, *PLAN_CTAS))) == CLUSTER_SIZES
LOG2E = math.log2(math.e)

# Exact keyword set of the generated program's ``run`` entry (bound by the
# export's argument plan); ``grid`` is expanded to ``grid_x/y/z``.
MAIN_KWARGS = (
    "q",
    "kv",
    "indices",
    "out",
    "topk",
    "keys_per_cta",
    "qk_scale",
    "out_scale",
    "grid",
)


# ---------------------------------------------------------------------------
# Host planner (pure functions; unit-tested without a GPU)
# ---------------------------------------------------------------------------


def keys_per_cta(topk: int, num_ctas: int) -> int:
    """Keys each CTA of a ``num_ctas`` split owns: ``topk / num_ctas`` rounded up to whole 32-key stages."""
    kpc = -(-int(topk) // int(num_ctas))
    kpc = -(-kpc // STAGE_KEYS) * STAGE_KEYS
    if kpc > MAX_KEYS_PER_CTA:
        raise ValueError(
            f"{kpc} keys per CTA exceed the {MAX_KEYS_PER_CTA}-entry index table"
        )
    return kpc


def plan_ctas(*, num_tokens: int, topk: int, capacity: Dict[int, int]) -> int:
    """Key splits per token: the largest of 8, 6, 5, 4, 3, 2 giving every CTA at least one full 32-key stage
    whose ``num_tokens`` clusters are co-resident in one wave (``capacity[c]`` clusters of ``c`` CTAs run at
    once); otherwise the smallest such size (several waves); 1 when no split leaves a full stage per CTA."""
    valid = [
        c
        for c in PLAN_CTAS
        if topk // c >= STAGE_KEYS and keys_per_cta(topk, c) * c >= topk
    ]
    for c in valid:
        if num_tokens <= capacity.get(c, 0):
            return c
    return valid[-1] if valid else 1


def is_valid_split(topk: int, num_ctas: int) -> bool:
    """Whether the program admits ``num_ctas`` CTAs per token for ``topk`` keys.

    ``num_ctas`` must be 1 or one of :data:`PLAN_CTAS` (7 is not a variant), the
    per-CTA share must fit the index table, and for a cluster the rounded share
    must not leave a CTA without keys (``ceil(topk / keys_per_cta) == num_ctas``).
    """
    num_ctas = int(num_ctas)
    if int(topk) <= 0 or (num_ctas != 1 and num_ctas not in PLAN_CTAS):
        return False
    try:
        kpc = keys_per_cta(topk, num_ctas)
    except ValueError:
        return False
    return num_ctas == 1 or -(-int(topk) // kpc) == num_ctas


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def _device_arch(device: torch.device | int) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "the generated NVFP4 sparse MLA decode program requires compute capability "
            f"10.0 or 10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(
    device: torch.device, num_ctas: Optional[int] = None
) -> bool:
    """True when this checkout registers the program for ``device``.

    With ``num_ctas`` the check names the module of that cluster size; without
    it every cluster size of the program must be registered (the automatic plan
    may select any of them).
    """
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    if arch is None:
        return False
    registered = registered_clusters(arch)
    if num_ctas is None:
        return bool(MODULES) and set(CLUSTER_SIZES) <= registered
    return int(num_ctas) in registered


def _max_active_clusters(module: Any, record: dict[str, Any], device_index: int) -> int:
    launch = record["main"]["launch"]
    block = [int(v) for v in launch["block"]] + [1, 1, 1]
    cluster = [int(v) for v in launch["cluster"]] + [1, 1, 1]
    return int(
        module.max_active_clusters(
            int(device_index),
            block[0],
            block[1],
            block[2],
            cluster[0],
            cluster[1],
            cluster[2],
            int(launch["dynamic_smem_bytes"]),
            bool(launch["cooperative"]),
        )
    )


@functools.cache
def cluster_capacity_for(device_index: int) -> Dict[int, int]:
    """``{cluster size: co-resident clusters}`` of the device for the program's split sizes.

    Asks each registered split's own module (``cudaOccupancyMaxActiveClusters``
    with that module's launch configuration), so the first automatic plan on a
    device builds every :data:`PLAN_CTAS` module.
    """
    arch = _device_arch(device_index)
    capacity: Dict[int, int] = {}
    with torch.cuda.device(device_index):
        for c in PLAN_CTAS:
            name = select_module(arch, c)
            module = load_nvfp4_sparse_mla_decode_cake_module(name, "main")
            capacity[c] = _max_active_clusters(module, MODULES[name], device_index)
    return capacity


def _check_launch_record(record: dict[str, Any], num_ctas: int) -> None:
    """The registry's launch resources must be the generated-kernel constants this host side assumes."""
    launch = record["main"]["launch"]
    expected = {
        "block": [THREADS, 1, 1],
        "cluster": [1, num_ctas, 1],
        "cooperative": False,
        "dynamic_smem_bytes": SMEM_BYTES if num_ctas == 1 else SMEM_BYTES_CLUSTER,
    }
    actual = {
        "block": [int(v) for v in launch["block"]],
        "cluster": [int(v) for v in launch["cluster"]],
        "cooperative": bool(launch["cooperative"]),
        "dynamic_smem_bytes": int(launch["dynamic_smem_bytes"]),
    }
    if actual != expected:
        raise RuntimeError(
            f"registered launch resources {actual} of the cluster-size-{num_ctas} module "
            f"differ from the generated-kernel constants {expected}"
        )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NVFP4SparseMLADecodeCakeRunner:
    """Launch the prepared decode.

    Calling the runner or ``launch()`` submits one kernel on the current stream
    with no CUDA allocation and no host synchronization, writes the bound
    ``out`` and returns it.  The kernel reads ``query`` / ``kv_cache`` /
    ``indices`` on device at every launch, so the same runner (or a CUDA Graph
    capturing it) stays valid when the caller writes new values into those
    buffers.  Prepare a new runner when shapes, scales or tensor bindings change.
    """

    module_name: str
    arch: str
    num_ctas: int
    keys_per_cta: int
    grid: tuple[int, int, int]
    out: torch.Tensor
    entry: Callable[..., Any]
    arguments: tuple

    def launch(self) -> torch.Tensor:
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch


def prepare_nvfp4_sparse_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    num_ctas_per_token: Optional[int] = None,
) -> NVFP4SparseMLADecodeCakeRunner:
    """Validate the inputs, plan the cluster size and bind the launch.

    Every allocation happens here (only the optional ``out``); the returned
    runner launches with none.  The JIT module of the selected cluster size
    (and, for the automatic plan, of every planned size) is built and loaded
    here, so prepare outside CUDA Graph capture.
    """
    _check_inputs(query, kv_cache, indices, out)
    device = query.device
    arch = _device_arch(device)
    num_tokens, topk = int(query.shape[0]), int(indices.shape[1])
    if num_tokens == 0:
        raise ValueError(
            "an empty batch has nothing to launch; nvfp4_sparse_mla_decode returns the empty output"
        )
    if topk <= 0:
        raise ValueError(
            f"indices must hold at least one slot per token, got [num_tokens, {topk}]"
        )
    if out is None:
        out = torch.empty(
            (num_tokens, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    if num_ctas_per_token is None:
        num_ctas = plan_ctas(
            num_tokens=num_tokens,
            topk=topk,
            capacity=cluster_capacity_for(device_index),
        )
        if not is_valid_split(topk, num_ctas):
            raise ValueError(
                f"the planned split {num_ctas} leaves an empty CTA for topk={topk}; "
                "pass num_ctas_per_token"
            )
    else:
        num_ctas = int(num_ctas_per_token)
        if not is_valid_split(topk, num_ctas):
            raise ValueError(
                f"num_ctas_per_token={num_ctas} is not valid for topk={topk}: the generated "
                f"program serves cluster sizes {CLUSTER_SIZES} whose per-CTA share of at most "
                f"{MAX_KEYS_PER_CTA} keys (whole {STAGE_KEYS}-key stages) leaves no CTA empty"
            )
    kpc = keys_per_cta(topk, num_ctas)

    module_name = select_module(arch, num_ctas)
    record = MODULES[module_name]
    _check_launch_record(record, num_ctas)
    physical = record["main"]
    with torch.cuda.device(device_index):
        module = load_nvfp4_sparse_mla_decode_cake_module(module_name, "main")
    grid = (num_tokens, num_ctas, 1)
    main_kwargs = dict(
        q=query.view(torch.int32),
        kv=kv_cache.view(-1),
        indices=indices,
        out=out,
        topk=topk,
        keys_per_cta=kpc,
        qk_scale=float(bmm1_scale) * LOG2E,
        out_scale=float(bmm2_scale),
        grid=grid,
    )
    assert tuple(main_kwargs) == MAIN_KWARGS
    grid_dims = dict(zip(("grid_x", "grid_y", "grid_z"), grid, strict=True))
    arguments = tuple(
        grid_dims[name] if kind == "grid" else main_kwargs[name]
        for kind, name in physical["arg_plan"]
    )
    entry = getattr(module, physical["ffi_entry"])
    return NVFP4SparseMLADecodeCakeRunner(
        module_name, arch, num_ctas, kpc, grid, out, entry, arguments
    )


def nvfp4_sparse_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    bmm1_scale: float,
    bmm2_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    num_ctas_per_token: Optional[int] = None,
    backend: str = "cake",
) -> torch.Tensor:
    """``flashinfer.mla.nvfp4_sparse_mla_decode`` semantics on the generated program (one launch)."""
    if backend != "cake":
        raise ValueError("this implementation serves backend='cake' only")
    if query.shape[0] == 0:  # empty batch: validate, allocate, no launch
        _check_inputs(query, kv_cache, indices, out)
        if out is None:
            out = torch.empty(
                (0, NUM_HEADS, V_HEAD_DIM), dtype=torch.bfloat16, device=query.device
            )
        return out
    return prepare_nvfp4_sparse_mla_decode(
        query,
        kv_cache,
        indices,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=out,
        num_ctas_per_token=num_ctas_per_token,
    )()
