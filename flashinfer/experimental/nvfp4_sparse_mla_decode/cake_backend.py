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
output), served by a family of generated tcgen05 kernels.  One thread-block
cluster of ``C`` CTAs serves one query token (grid ``(T, C, 1)``, cluster
``(1, C, 1)``, 512 threads): CTA ``c`` owns the keys ``[c * keys_per_cta,
(c + 1) * keys_per_cta)`` with ``keys_per_cta = ceil(topk / C)``, gathers
their raw rows with TMA, dequantizes them and computes ``S`` and ``O`` on the
tensor core in 128-key tiles, and either writes the normalized output directly
(``C = 1``) or pushes its fp32 partial ``(O, m, l)`` through distributed shared
memory to the cluster peers that own the 8-dim output groups, which merge and
store.  ``C`` is one of 1, 2, 3, 4, 5, 6 or 8, and every cluster size has one
program per split bound (at most 64, 128, 192 or 256 keys per CTA, or any) --
delivered for both architectures (``cake_jit.MODULES``).  ``C`` is planned per
launch from the driver's one-wave cluster capacity unless the caller forces it
with ``num_ctas_per_token``; the split bound follows from ``keys_per_cta``.
Nothing is allocated at launch (only the optional output), so a prepared runner
is CUDA Graph safe.  See ``README.md`` in this package.
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
    ARCH_BY_CAPABILITY,
    ARCHES,
    CLUSTER_SIZES,
    MAX_KEYS,
    MODULES,
    arch_for_capability,
    launch_args,
    launch_grid,
    load,
    program_max_keys,
    record_for,
    registered_clusters,
    select_module,
)

__all__ = [
    "CLUSTER_SIZES",
    "MAX_KEYS",
    "MIN_PLAN_KEYS",
    "NVFP4SparseMLADecodeCakeRunner",
    "PLAN_CTAS",
    "SUPPORTED_COMPUTE_CAPABILITIES",
    "THREADS",
    "cluster_capacity_for",
    "generated_program_available",
    "is_valid_split",
    "keys_per_cta",
    "nvfp4_sparse_mla_decode",
    "plan_ctas",
    "prepare_nvfp4_sparse_mla_decode",
    "program_max_keys",
    "split_cost",
]

SUPPORTED_COMPUTE_CAPABILITIES = dict(ARCH_BY_CAPABILITY)
assert set(SUPPORTED_COMPUTE_CAPABILITIES.values()) == set(ARCHES)

# Generated-kernel constant (copied from the programs that the export renders;
# the registry's launch records must agree, see ``_check_launch_record``).
THREADS = 512  # 4 layout, 4 softmax, 2 MMA, 2 TMA gather and 4 tile-0 copy warps
# Split sizes the automatic plan considers, largest first; 1 is the fallback
# when no split leaves every CTA MIN_PLAN_KEYS keys.  7 is not a variant of the
# program.
PLAN_CTAS = (8, 6, 5, 4, 3, 2)
assert tuple(sorted((1, *PLAN_CTAS))) == CLUSTER_SIZES
# Fewest keys per CTA the automatic plan gives a split.
MIN_PLAN_KEYS = 32
LOG2E = math.log2(math.e)

# Exact keyword set of the generated program's ``run`` entry (bound through the
# record's ``arg_plan`` under its ``logical`` names); ``grid`` is expanded to
# ``grid_x/y/z``.
MAIN_KWARGS = (
    "q",
    "kv",
    "kv_rows",
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
    """Keys each CTA of a ``num_ctas`` split owns: ``ceil(topk / num_ctas)``."""
    return -(-int(topk) // int(num_ctas))


def split_cost(keys: int) -> int:
    """Work of one CTA that owns ``keys`` keys, in 64-key halves of a 128-key
    tile: the programs of at most 256 keys skip an empty half tile, the general
    program walks whole tiles."""
    keys = int(keys)
    return -(-keys // 64) if keys <= 256 else 2 * -(-keys // 128)


def plan_ctas(*, num_tokens: int, topk: int, capacity: Dict[int, int]) -> int:
    """Key splits per token: among 8, 6, 5, 4, 3, 2 giving every CTA at least
    :data:`MIN_PLAN_KEYS` keys and leaving no CTA without keys
    (:func:`is_valid_split`) whose ``num_tokens`` clusters are co-resident in one
    wave (``capacity[c]`` clusters of ``c`` CTAs run at once), the one with the
    least work per CTA (:func:`split_cost`), then the fewest CTAs; otherwise the
    smallest such size (several waves); 1 when no split qualifies."""
    topk = int(topk)
    valid = [
        c for c in PLAN_CTAS if topk // c >= MIN_PLAN_KEYS and is_valid_split(topk, c)
    ]
    one_wave = [c for c in valid if num_tokens <= capacity.get(c, 0)]
    if one_wave:
        return min(one_wave, key=lambda c: (split_cost(keys_per_cta(topk, c)), c))
    return valid[-1] if valid else 1


def is_valid_split(topk: int, num_ctas: int) -> bool:
    """Whether the programs admit ``num_ctas`` CTAs per token for ``topk`` keys.

    ``num_ctas`` must be 1 or one of :data:`PLAN_CTAS` (7 is not a variant), and
    for a cluster the per-CTA share must not leave a CTA without keys
    (``ceil(topk / keys_per_cta) == num_ctas``).
    """
    topk, num_ctas = int(topk), int(num_ctas)
    if topk <= 0 or (num_ctas != 1 and num_ctas not in PLAN_CTAS):
        return False
    return num_ctas == 1 or -(-topk // keys_per_cta(topk, num_ctas)) == num_ctas


# ---------------------------------------------------------------------------
# Device / registry queries
# ---------------------------------------------------------------------------


def _device_arch(device: torch.device | int) -> str:
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = arch_for_capability(capability)
    if arch is None:
        raise ValueError(
            "the generated NVFP4 sparse MLA decode program requires compute capability "
            f"10.0 or 10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(
    device: torch.device, num_ctas: Optional[int] = None
) -> bool:
    """True when this checkout delivers the programs for ``device``.

    With ``num_ctas`` the check names the programs of that cluster size (every
    split bound); without it every cluster size must be delivered (the automatic
    plan may select any of them).
    """
    arch = arch_for_capability(tuple(torch.cuda.get_device_capability(device)))
    if arch is None:
        return False
    registered = registered_clusters(arch)
    if num_ctas is None:
        return bool(MODULES) and set(CLUSTER_SIZES) <= registered
    return int(num_ctas) in registered


def _max_active_clusters(module: Any, record: dict[str, Any], device_index: int) -> int:
    launch = record["launch"]
    block = [int(v) for v in launch["block"]] + [1, 1, 1]
    cluster = [int(v) for v in launch["cluster"]] + [1, 1, 1]
    return int(
        getattr(module, record["occupancy_entry"])(
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
    """``{cluster size: co-resident clusters}`` of the device for the split sizes.

    Asks the general program of each split size (its ``max_active_clusters``
    entry, the ``cudaOccupancyMaxActiveClusters`` of that program's launch
    configuration; every program runs one CTA per SM), so the first automatic
    plan on a device builds the general library of every :data:`PLAN_CTAS` size.
    """
    arch = _device_arch(device_index)
    capacity: Dict[int, int] = {}
    with torch.cuda.device(device_index):
        for c in PLAN_CTAS:
            name = record_for(c, 0)
            module = load(name, arch)
            capacity[c] = _max_active_clusters(module, MODULES[name], device_index)
    return capacity


def _check_launch_record(record: dict[str, Any], num_ctas: int) -> None:
    """The registry's launch record must be the launch this host side assumes."""
    launch = record["launch"]
    expected = {
        "block": (THREADS, 1, 1),
        "cluster": (1, num_ctas, 1),
        "cooperative": False,
        "use_pdl": False,
    }
    actual = {
        "block": tuple(int(v) for v in launch["block"]),
        "cluster": tuple(int(v) for v in launch["cluster"]),
        "cooperative": bool(launch["cooperative"]),
        "use_pdl": launch["use_pdl"],
    }
    if actual != expected or int(launch["dynamic_smem_bytes"]) <= 0:
        raise RuntimeError(
            f"registered launch record {dict(launch)} of the cluster-size-{num_ctas} program "
            f"differs from the launch {expected} with positive dynamic shared memory"
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
    max_keys: int
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
    runner launches with none.  The library of the selected program (and, for
    the automatic plan, the general library of every planned size) is built and
    loaded here, so prepare outside CUDA Graph capture.
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
                f"programs serve cluster sizes {CLUSTER_SIZES} whose per-CTA share "
                "ceil(topk / C) leaves no CTA empty"
            )
    kpc = keys_per_cta(topk, num_ctas)

    module_name = select_module(arch, num_ctas, kpc)
    record = MODULES[module_name]
    _check_launch_record(record, num_ctas)
    with torch.cuda.device(device_index):
        module = load(module_name, arch)
    grid = launch_grid(record, num_tokens=num_tokens)
    rows = kv_cache.view(-1, 352)
    main_kwargs = dict(
        q=query.view(torch.int32),
        kv=rows.view(torch.int32),
        kv_rows=rows,
        indices=indices,
        out=out,
        topk=topk,
        keys_per_cta=kpc,
        qk_scale=float(bmm1_scale) * LOG2E,
        out_scale=float(bmm2_scale),
        grid=grid,
    )
    assert tuple(main_kwargs) == MAIN_KWARGS
    # The record's argument plan over ``main_kwargs`` under its logical names, then the grid axes.
    arguments = launch_args(record, main_kwargs, grid)
    entry = getattr(module, record["ffi_entry"])
    return NVFP4SparseMLADecodeCakeRunner(
        module_name,
        arch,
        num_ctas,
        kpc,
        int(record["max_keys"]),
        grid,
        out,
        entry,
        arguments,
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
