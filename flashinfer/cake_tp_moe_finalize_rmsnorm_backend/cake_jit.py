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

Host plan and loader glue of the Cake tensor-parallel MoE finalize kernels
(``flashinfer.comm.tp_moe_finalize_rmsnorm``: the top-8 routed GEMM2 rows of
every rank reduced over the group, the per-expert RMSNorm and the routing
weights applied, hidden size 4096, BF16; SM100 / SM103).

Every record of ``MODULES`` is one mechanically exported production build: a
CUDA device translation unit, a tvm-ffi binding and an occupancy query
compiled together for each architecture the record names.  The record carries
the host plan only: the ``library`` stem, the architectures it is delivered
for, the ``build`` keywords it was rendered with, its program kind and world
size, its launch record, its grid rule, its selection rule, its workspace rule
and the kernel-parameter -> logical-argument map.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags, verifying every source digest before the
first build.

A route is the list of records one public call may launch at one world size:
``tp<world>`` per architecture (``ROUTES``).  Each record admits a window of
token counts on each of its architectures (``select_rule``: ``above < n_tok
<= at_most``, the last window unbounded); the windows of a route partition
every positive token count, so ``selected_module`` always answers where the
route exists.  An architecture without the route ships no program for that
world size.  The grid follows the record's ``grid_rule`` (``launch_grid``):
one cluster of ``split`` CTAs per token (``eo``, ``onehop``), one CTA per
token (``rsag``), ``(split, ceil(n_tok / world), world)`` clusters
(``token1``) or one cooperative grid of ``G = 1 (mod world)`` CTAs
(``tokown``); every CTA of a launch is resident at once, so the grid is held
to the record's co-residency, read once per (record, architecture, device)
from its occupancy query (``occupancy_entry``).

Every record owns a Lamport workspace (``workspace_rule``): the symmetric
rings its kernel declares, each filled with the kernel's sentinel word, one of
them behind a multicast view, and its rank-local counters and scratch.  The
host (``cake_backend``) provisions it once per group from the record alone,
sized at the record's largest routed token count, and passes the kernel its
device address table, its local pointers and its multicast address through
the kernel's own parameters; a launch is one kernel, with no host
synchronisation, and replays under CUDA graphs.
"""

from __future__ import annotations

import functools
from typing import Any, Mapping, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) and the ``ROUTES`` table are
# rewritten by the exporter from the rendered programs; every other field is
# host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_tp_moe_finalize_rmsnorm_09a7bb8908c5bf9cdd04": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "gemm2",
            ],
            [
                "parameter",
                "row_stride",
            ],
            [
                "parameter",
                "rows",
            ],
            [
                "buffer",
                "idx",
            ],
            [
                "buffer",
                "expert_weights",
            ],
            [
                "parameter",
                "weights_fp32",
            ],
            [
                "buffer",
                "norm_weight",
            ],
            [
                "buffer",
                "output",
            ],
            [
                "parameter",
                "n_tok",
            ],
            [
                "parameter",
                "slots",
            ],
            [
                "parameter",
                "eps",
            ],
            [
                "parameter",
                "gain_center",
            ],
            [
                "parameter",
                "mode",
            ],
            [
                "parameter",
                "rank",
            ],
            [
                "buffer",
                "peer_table",
            ],
            [
                "buffer",
                "gen",
            ],
            [
                "parameter",
                "b_mcast",
            ],
            [
                "grid",
                "grid_x",
            ],
            [
                "grid",
                "grid_y",
            ],
            [
                "grid",
                "grid_z",
            ],
        ],
        "build": {
            "broadcast": "multicast",
            "fence": "auto",
            "pack": "sector",
            "pdl": False,
            "push": "weak",
            "split": 2,
            "trace": False,
        },
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "eo",
            "world": 2,
            "split": 2,
        },
        "kernel_symbol": "kernel_cake_tp_moe_finalize_rmsnorm_09a7bb8908c5bf9cdd04",
        "launch": {
            "block": (
                256,
                1,
                1,
            ),
            "cluster": (
                2,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_tp_moe_finalize_rmsnorm_09a7bb8908c5bf9cdd04",
        "logical": {
            "gemm2": "gemm2_permuted",
            "row_stride": "row_stride",
            "rows": "rows",
            "idx": "expanded_idx_to_permuted_idx",
            "expert_weights": "expert_weights_words",
            "weights_fp32": "weights_fp32",
            "norm_weight": "norm_weight",
            "output": "output",
            "n_tok": "n_tok",
            "slots": "slots",
            "eps": "eps",
            "gain_center": "gain_center",
            "mode": "weight_before_norm",
            "rank": "rank",
            "peer_table": "peer_table",
            "gen": "gen",
            "b_mcast": "partial_multicast",
        },
        "occupancy_entry": "max_active_clusters",
        "program": "eo",
        "route": "tp2",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        0,
                        32,
                    ],
                ],
                "sm_103a": [
                    [
                        33,
                        64,
                    ],
                ],
            },
        },
        "workspace_rule": {
            "kind": "lamport",
            "program": "eo",
            "world": 2,
            "slot_parameter": "slots",
            "buffers": [
                {
                    "name": "scatter",
                    "placement": "symmetric",
                    "bytes_per_slot": 196608,
                    "fill": 2147516416,
                    "multicast": None,
                    "view": None,
                },
                {
                    "name": "partial",
                    "placement": "symmetric",
                    "bytes_per_slot": 98304,
                    "fill": 2147483648,
                    "multicast": "b_mcast",
                    "view": None,
                },
                {
                    "name": "gen",
                    "placement": "local",
                    "bytes_per_slot": 4,
                    "fill": 0,
                    "multicast": None,
                    "view": "gen",
                },
            ],
            "tables": [
                {
                    "parameter": "peer_table",
                    "entries": [
                        "scatter",
                        "partial",
                    ],
                },
            ],
        },
        "world": 2,
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "tp2": [
            "cake_tp_moe_finalize_rmsnorm_09a7bb8908c5bf9cdd04",
        ],
    },
    "sm_103a": {
        "tp2": [
            "cake_tp_moe_finalize_rmsnorm_09a7bb8908c5bf9cdd04",
        ],
    },
}

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 /
# GB300 (``sm_103a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
HIDDEN_SIZE = 4096
TOPK = 8

#: The world sizes the kernels are built for (one route per world size) and
#: the program kinds a record may be (its grid rule kind).
WORLDS = (2, 4, 8)
PROGRAM_KINDS = ("onehop", "rsag", "tokown", "token1", "eo")
SELECT_KIND = "rows_between"
GRID_KINDS = PROGRAM_KINDS
WORKSPACE_KIND = "lamport"

#: The logical argument vocabulary of ``run_module`` ``values``: the call's
#: tensors and scalars, then the workspace handles the host passes under the
#: kernel's own names (the device address table, the local pointers, the
#: ``u64`` multicast addresses).
LOGICAL_TENSORS = (
    "gemm2_permuted",
    "expanded_idx_to_permuted_idx",
    "expert_weights_words",
    "norm_weight",
    "output",
    "peer_table",
    "gen",
    "part",
    "ctr",
    "ctl",
)
LOGICAL_SCALARS = (
    "row_stride",
    "rows",
    "weights_fp32",
    "n_tok",
    "slots",
    "max_ctas",
    "eps",
    "gain_center",
    "weight_before_norm",
    "rank",
    "partial_multicast",
    "rows_multicast",
    "bcast_multicast",
)


def route_key(world: int) -> str:
    """The route one public call at world size ``world`` launches from: ``tp<world>``."""
    if world not in WORLDS:
        raise ValueError(
            f"unsupported Cake TP MoE finalize world size {world!r}; the package ships {WORLDS}"
        )
    return f"tp{world}"


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""
    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake TP MoE finalize target"
        )
    return arch


def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def route_available(arch: str, route: str) -> bool:
    """Whether ``arch`` ships a program of ``route``."""
    return bool(ROUTES.get(arch, {}).get(route))


def route_modules(arch: str, route: str) -> tuple[str, ...]:
    """The records of ``route`` on ``arch`` in window order (empty when the route is not exported there)."""
    return tuple(ROUTES.get(arch, {}).get(route, ()))


def admits(rule: dict[str, Any] | None, n_tok: int, arch: str) -> bool:
    """The record's selection rule: ``None`` admits every call; ``rows_between`` admits ``n_tok`` on ``arch``
    when one of the architecture's windows holds ``above < n_tok <= at_most`` (``at_most`` ``None`` = unbounded)."""
    if rule is None:
        return True
    if rule.get("kind") != SELECT_KIND:
        raise ValueError(
            f"unknown Cake TP MoE finalize select rule {rule.get('kind')!r}"
        )
    windows = rule["windows"].get(arch, ())
    return any(
        n_tok > int(above) and (at_most is None or n_tok <= int(at_most))
        for above, at_most in windows
    )


def selected_module(arch: str, route: str, n_tok: int) -> str:
    """The record of ``route`` whose window admits ``n_tok`` tokens on ``arch``."""
    if n_tok <= 0:
        raise ValueError(
            f"Cake TP MoE finalize launches need a positive token count, got {n_tok}"
        )
    for name in route_modules(arch, route):
        if admits(MODULES[name].get("select_rule"), n_tok, arch):
            return name
    raise NotImplementedError(
        f"no exported program of route {route!r} on {arch} admits {n_tok} tokens"
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake TP MoE finalize module {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    ``cake_build.load_library`` serves a prebuilt ``aot/`` library when a
    builder compiled it and compiles the manifest's sources otherwise.
    """
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


@functools.cache
def cluster_capacity(name: str, arch: str, device_index: int) -> int:
    """How many clusters (CTAs, for a record without a cluster) of record ``name``'s launch shape device
    ``device_index`` holds at once: the record's occupancy query over its launch record, asked once per (record,
    architecture, device).  Loads the library on first use: call it outside CUDA graph capture."""
    record = MODULES[name]
    launch = record["launch"]
    query = getattr(load(name, arch), record["occupancy_entry"])
    block = [int(value) for value in launch["block"]]
    cluster = [int(value) for value in launch["cluster"]]
    return int(
        query(
            int(device_index),
            *block,
            *cluster,
            int(launch["dynamic_smem_bytes"]),
            bool(launch["cooperative"]),
        )
    )


def tokown_grid(n_tok: int, capacity: int, world: int) -> int:
    """The persistent program's CTA count: the smallest ``G = 1 (mod world)`` of at least ``n_tok`` CTAs when it is
    co-resident, else the largest co-resident one."""
    grid = n_tok + (1 - n_tok) % world
    return grid if grid <= capacity else capacity - (capacity - 1) % world


def launch_grid(
    record: dict[str, Any],
    n_tok: int,
    *,
    device_index: int,
    values: Mapping[str, Any] | None = None,
    capacity: int | None = None,
) -> tuple[int, int, int]:
    """Launch grid of one record under its recorded grid rule for ``n_tok`` tokens, held to the record's
    co-residency on the device (every CTA of a launch must be resident at once: the ranks wait on each other).
    ``values`` are the launch's logical arguments when the caller has them: the slot count the workspace was
    provisioned with bounds the launch (``slots`` >= ``n_tok``; the persistent program's grid <= ``max_ctas``).
    ``capacity`` is the co-residency the caller holds the launch to instead of this device's own (the smallest
    over the ranks of a group, agreed once at initialization)."""
    if n_tok <= 0:
        raise ValueError(
            f"Cake TP MoE finalize launches need a positive token count, got {n_tok}"
        )
    rule = record["grid_rule"]
    kind = rule.get("kind")
    world = int(rule["world"])
    if capacity is None:
        capacity = cluster_capacity(
            record["library"], device_arch(device_index), device_index
        )
    capacity = int(capacity)
    if kind == "tokown":
        if values is not None:
            capacity = min(capacity, int(values["max_ctas"]))
        return (tokown_grid(n_tok, max(1, capacity), world), 1, 1)
    if values is not None and n_tok > int(values["slots"]):
        raise ValueError(
            f"{record['library']}: {n_tok} tokens exceed the {values['slots']} slots its workspace holds"
        )
    if kind in ("eo", "onehop"):
        if n_tok > capacity:
            raise RuntimeError(
                f"{record['library']}: {n_tok} clusters of {rule['split']} CTAs exceed the device's "
                f"co-residency of {capacity}"
            )
        return (int(rule["split"]) * n_tok, 1, 1)
    if kind == "rsag":
        if n_tok > capacity:
            raise RuntimeError(
                f"{record['library']}: {n_tok} CTAs exceed the device's co-residency of {capacity}"
            )
        return (n_tok, 1, 1)
    if kind == "token1":
        groups = -(-n_tok // world)
        if groups * world > capacity:
            raise RuntimeError(
                f"{record['library']}: {groups * world} clusters of {rule['split']} CTAs exceed the "
                f"device's co-residency of {capacity}"
            )
        return (int(rule["split"]), groups, world)
    raise ValueError(f"unknown Cake TP MoE finalize grid rule {kind!r}")


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: Sequence[int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``ffi_entry``: ``values`` by logical name in ``arg_plan`` order,
    the grid axes where the plan names them."""
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    logical = record["logical"]
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {record['library']} needs logical argument {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    return tuple(args)


def run_module(
    name: str, values: dict[str, Any], *, rows: int, device_index: int
) -> None:
    """Launch one exported module with logical argument ``values`` (``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS``)
    for ``rows`` tokens (the call's ``n_tok``) on the current torch stream of device ``device_index``."""
    record = MODULES[name]
    grid = launch_grid(record, rows, device_index=device_index, values=values)
    args = launch_args(record, values, grid)
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "GRID_KINDS",
    "HIDDEN_SIZE",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "PROGRAM_KINDS",
    "ROUTES",
    "SELECT_KIND",
    "TOPK",
    "WORKSPACE_KIND",
    "WORLDS",
    "admits",
    "arch_for_capability",
    "cluster_capacity",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "route_available",
    "route_key",
    "route_modules",
    "run_module",
    "selected_module",
    "sm_count",
    "specs_for_arch",
    "tokown_grid",
]
