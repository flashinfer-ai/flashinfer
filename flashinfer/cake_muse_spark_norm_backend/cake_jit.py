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

Host plan and loader glue of the Cake single-GPU norm kernels of the Muse
Spark family: today the gated residual with the next RMSNorm fused
(``flashinfer.norm.gated_residual_next_rmsnorm``, hidden size 8192, BF16
branch and FP32 residual stream; SM100 / SM103).

Every record of ``MODULES`` is one mechanically exported production build: a
CUDA device translation unit, a tvm-ffi binding and an occupancy query
compiled together for each architecture the record names.  The record carries
the host plan only: the ``library`` stem, the architectures it is delivered
for, the ``build`` keywords it was rendered with, its launch record, its grid
rule, its selection rule and the kernel-parameter -> logical-argument map.
What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route is the list of records one public call may launch:
``<op>_<variant>`` per architecture (``ROUTES``), the variant naming which of
the call's optional inputs the programs take (``alpha`` a tensor or a scalar,
``post_weight`` present or not).  Each record admits a window of row counts on
each of its architectures (``select_rule``: ``above < rows <= at_most``, the
last window unbounded); the windows of a route partition every positive row
count, so ``selected_module`` always answers.  The grid follows the record's
``grid_rule``: one cluster of ``cluster`` CTAs per row, or a persistent cluster
loop of ``min(rows, co-resident clusters)`` clusters, the co-residency read once
per (record, architecture, device) from the record's occupancy query
(``occupancy_entry``).  No program owns a workspace (``workspace_rule`` is
``None`` on every record).  Batch size is a runtime scalar, so a changed row
count never recompiles anything.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) and the ``ROUTES`` table are
# rewritten by the exporter from the rendered programs; every other field is
# host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_muse_spark_norm_1aa28374ae397e05d09f": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            [
                "buffer",
                "branch",
            ],
            [
                "buffer",
                "residual",
            ],
            [
                "buffer",
                "alpha",
            ],
            [
                "buffer",
                "beta",
            ],
            [
                "buffer",
                "next_weight",
            ],
            [
                "buffer",
                "post_weight",
            ],
            [
                "buffer",
                "residual_out",
            ],
            [
                "buffer",
                "ffn_input",
            ],
            [
                "parameter",
                "M",
            ],
            [
                "parameter",
                "post_eps",
            ],
            [
                "parameter",
                "next_eps",
            ],
            [
                "parameter",
                "post_gain_center",
            ],
            [
                "parameter",
                "next_gain_center",
            ],
            [
                "parameter",
                "alpha_value",
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
            "H": 8192,
            "cluster": 1,
            "exit_sync": False,
            "expect_hoist": False,
            "issue_order": "source",
            "l2_hint": None,
            "load_order": "vectors_first",
            "min_blocks": 0,
            "packed": True,
            "pdl": False,
            "pdl_launch": "statistic",
            "persistent": True,
            "post_weight": True,
            "prefetch": True,
            "prefetch_depth": 1,
            "reload_fp32": False,
            "resident": True,
            "rsqrt_approx": False,
            "store_all_lanes": False,
            "tensor_alpha": True,
            "threads": 512,
            "tma_stages": 0,
            "vec": 16,
            "wait_form": "acquire_cluster",
        },
        "ffi_entry": "run",
        "grid_rule": {
            "cluster": 1,
            "kind": "persistent_rows",
        },
        "kernel_symbol": "kernel_cake_muse_spark_norm_1aa28374ae397e05d09f",
        "launch": {
            "block": (
                512,
                1,
                1,
            ),
            "cluster": (
                1,
                1,
                1,
            ),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_muse_spark_norm_1aa28374ae397e05d09f",
        "logical": {
            "M": "rows",
            "alpha": "alpha",
            "alpha_value": "alpha_value",
            "beta": "beta",
            "branch": "branch",
            "ffn_input": "ffn_input",
            "next_eps": "next_eps",
            "next_gain_center": "next_gain_center",
            "next_weight": "next_weight",
            "post_eps": "post_eps",
            "post_gain_center": "post_gain_center",
            "post_weight": "post_weight",
            "residual": "residual",
            "residual_out": "residual_out",
        },
        "occupancy_entry": "max_active_clusters",
        "op": "gated_residual_next_rmsnorm",
        "route": "gated_residual_next_rmsnorm_alpha_tensor_pw",
        "select_rule": {
            "kind": "rows_between",
            "windows": {
                "sm_100a": [
                    [
                        1536,
                        None,
                    ],
                ],
                "sm_103a": [
                    [
                        1536,
                        None,
                    ],
                ],
            },
        },
        "workspace_rule": None,
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_1aa28374ae397e05d09f",
        ],
    },
    "sm_103a": {
        "gated_residual_next_rmsnorm_alpha_tensor_pw": [
            "cake_muse_spark_norm_1aa28374ae397e05d09f",
        ],
    },
}

# One verified build per architecture: B200 / GB200 (``sm_100a``) and B300 /
# GB300 (``sm_103a``); ``sm_107a`` is an exported target of the family with no
# routed program yet.  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
HIDDEN_SIZE = 8192

#: The operators of the package and the variants of each (the route suffixes a
#: public call maps onto: which optional inputs the programs take).
OPS = ("gated_residual_next_rmsnorm",)
VARIANTS: dict[str, tuple[str, ...]] = {
    "gated_residual_next_rmsnorm": (
        "alpha_tensor_pw",
        "alpha_scalar_nopw",
        "alpha_scalar_pw",
        "alpha_tensor_nopw",
    ),
}
SELECT_KIND = "rows_between"
GRID_KINDS = ("rows", "persistent_rows")

#: The logical argument vocabulary of ``run_module`` ``values``.
LOGICAL_TENSORS = (
    "branch",
    "residual",
    "alpha",
    "beta",
    "next_weight",
    "post_weight",
    "residual_out",
    "ffn_input",
)
LOGICAL_SCALARS = (
    "rows",
    "post_eps",
    "next_eps",
    "post_gain_center",
    "next_gain_center",
    "alpha_value",
)


def route_key(op: str, variant: str) -> str:
    """The route one public call of ``op`` with ``variant`` launches from: ``<op>_<variant>``."""
    if op not in OPS:
        raise ValueError(f"unknown Cake norm operator {op!r}; the package ships {OPS}")
    if variant not in VARIANTS[op]:
        raise ValueError(
            f"unknown variant {variant!r} of {op}; its variants are {VARIANTS[op]}"
        )
    return f"{op}_{variant}"


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
            f"device {device_index} (capability {capability}) is not an exported Cake norm target"
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


def admits(rule: dict[str, Any] | None, rows: int, arch: str) -> bool:
    """The record's selection rule: ``None`` admits every call; ``rows_between`` admits ``rows`` on ``arch``
    when one of the architecture's windows holds ``above < rows <= at_most`` (``at_most`` ``None`` = unbounded)."""
    if rule is None:
        return True
    if rule.get("kind") != SELECT_KIND:
        raise ValueError(f"unknown Cake norm select rule {rule.get('kind')!r}")
    windows = rule["windows"].get(arch, ())
    return any(
        rows > int(above) and (at_most is None or rows <= int(at_most))
        for above, at_most in windows
    )


def selected_module(arch: str, route: str, rows: int) -> str:
    """The record of ``route`` whose window admits ``rows`` on ``arch``."""
    if rows <= 0:
        raise ValueError(f"Cake norm launches need a positive row count, got {rows}")
    for name in route_modules(arch, route):
        if admits(MODULES[name].get("select_rule"), rows, arch):
            return name
    raise NotImplementedError(
        f"no exported program of route {route!r} on {arch} admits {rows} rows"
    )


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake norm module {name} is not delivered for {arch}; its architectures: {record['arches']}"
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
    """How many clusters of record ``name``'s launch shape device ``device_index`` holds at once: the record's
    occupancy query over its launch record, asked once per (record, architecture, device).  Loads the library on
    first use: call it outside CUDA graph capture."""
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


def launch_grid(
    record: dict[str, Any], rows: int, *, device_index: int
) -> tuple[int, int, int]:
    """Launch grid of one record under its recorded grid rule: ``rows x cluster`` CTAs (one cluster per row), or
    ``min(rows, co-resident clusters) x cluster`` CTAs for the persistent row loop."""
    if rows <= 0:
        raise ValueError(f"Cake norm launches need a positive row count, got {rows}")
    rule = record["grid_rule"]
    cluster = int(rule["cluster"])
    kind = rule.get("kind")
    if kind == "rows":
        return (rows * cluster, 1, 1)
    if kind == "persistent_rows":
        capacity = cluster_capacity(
            record["library"], device_arch(device_index), device_index
        )
        return (min(rows, max(1, capacity)) * cluster, 1, 1)
    raise ValueError(f"unknown Cake norm grid rule {kind!r}")


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
    over ``rows`` rows on the current torch stream of device ``device_index``."""
    record = MODULES[name]
    grid = launch_grid(record, rows, device_index=device_index)
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
    "OPS",
    "ROUTES",
    "SELECT_KIND",
    "VARIANTS",
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
]
