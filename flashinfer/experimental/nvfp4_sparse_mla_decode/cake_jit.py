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

Host plan and loader glue of the Cake experimental NVFP4 sparse MLA decode
program (SM100 / SM103), the ``backend="cake"`` of
:func:`flashinfer.mla.nvfp4_sparse_mla_decode`.

Contract of the program: ``out[t, h] = out_scale * softmax_k(qk_scale *
q[t, h] . K[indices[t, k]]) @ V[indices[t, k]]`` over E4M3 ``[T, 16, 576]``
queries (``q``, handed to the kernel as int32 words), 352-byte NVFP4 cache
rows (``kv``, the flat uint8 row pool), int32 ``[T, topk]`` row ids with
``-1`` marking an empty slot (``indices``) and a BF16 ``[T, 16, 512]`` output
(``out``); ``qk_scale`` is the base-2 softmax scale (``bmm1_scale *
log2(e)``), ``out_scale`` the output scale and ``keys_per_cta`` the keys each
CTA of the token's cluster owns.  One thread-block cluster of ``C`` CTAs
serves one query token (grid ``(T, C, 1)``, cluster ``(1, C, 1)``, 448
threads): CTA ``c`` walks the keys ``[c * keys_per_cta, (c + 1) *
keys_per_cta)`` in 32-key stages and either writes the normalized output
directly (``C = 1``) or merges its fp32 partials with the cluster peers
through distributed shared memory.  ``C`` is one of :data:`CLUSTER_SIZES`,
one program per cluster size; :mod:`.cake_backend` plans it per launch.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit, a tvm-ffi binding and an occupancy query unit,
compiled once per architecture.  A record carries the host plan only: the
program's ``library`` stem (its ``build.json`` entries are
``<library>_<arch>``), the architectures it is delivered for, the source
``module`` and ``build`` keywords it was rendered from, its launch record,
the argument plan of its ``run`` entry, the kernel-parameter -> logical-
argument map, the cluster size and the stage constants of the host planner
(``cluster``, ``keys_per_stage``, ``max_keys_per_cta``), the grid rule
:func:`launch_grid` evaluates, the name of the occupancy entry
(``occupancy_entry``: ``max_active_clusters(device, block_x, block_y,
block_z, cluster_x, cluster_y, cluster_z, dynamic_smem_bytes, cooperative)``,
the one-wave cluster capacity the planner asks each program for) and the
workspace rule (``None``: the kernel owns no scratch).  What to compile --
sources, digests, nvcc flags -- is recorded once in ``build.json`` next to
this file; :mod:`.cake_build` serves the library from a prebuilt ``aot/``
file when a builder compiled it and otherwise through FlashInfer's JIT from
the same sources and flags.  Every ``build.json`` entry records
``--use_fast_math`` (the flag the kernel was validated and measured with);
the loader passes ``use_fast_math=False`` so that the manifest, not the
loader, states it.

The derived leaves of a record (``kernel_symbol``, ``launch.block``,
``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
``launch.use_pdl``, ``arg_plan``, ``logical``) are rewritten by the exporter
from the rendered binding; every other field is the host plan.  Every value
is a literal: FlashInfer's manifest checker and the exporter read this file
with ``ast.literal_eval``.

The program name ``cake_nvfp4_sparse_mla_decode_<20 hex>`` is the first
twenty hex digits of the SHA-256 of the compact JSON ``{"build": <build
keywords, sorted, every default spelled>, "module": <dotted source module>}``
of the kernel the program was rendered from; a changed keyword is a new
program.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

from . import cake_build

# Two delivered architectures: B200 / GB200 (``sm_100a``) and B300 / GB300
# (``sm_103a``), each compiled with its exact ``-gencode`` flag set
# (``build.json`` ``arch_flags``).
ARCHES = ("sm_100a", "sm_103a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}

# Thread-block cluster sizes (CTAs per query token) the program is delivered
# for, one record each; 7 is not a variant of the program (see
# ``cake_backend.PLAN_CTAS``).
CLUSTER_SIZES = (1, 2, 3, 4, 5, 6, 8)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
# ``launch.dynamic_smem_bytes`` is the kernel's computed shared memory at the
# last render.
MODULES: dict[str, dict[str, Any]] = {
    "cake_nvfp4_sparse_mla_decode_b321ca1b7dfea243bca6": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 1,
        },
        "cluster": 1,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_b321ca1b7dfea243bca6",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_b321ca1b7dfea243bca6",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 192256,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_b321ca1b7dfea243bca6",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_9db88fc2ce17eaf2d27e": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 2,
        },
        "cluster": 2,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_9db88fc2ce17eaf2d27e",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_9db88fc2ce17eaf2d27e",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 2, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_9db88fc2ce17eaf2d27e",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_525c3da3db366af4b77c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 3,
        },
        "cluster": 3,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_525c3da3db366af4b77c",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_525c3da3db366af4b77c",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 3, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_525c3da3db366af4b77c",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_b5c9cbf01c57fe28ab5c": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 4,
        },
        "cluster": 4,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_b5c9cbf01c57fe28ab5c",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_b5c9cbf01c57fe28ab5c",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 4, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_b5c9cbf01c57fe28ab5c",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_79fa77f600e1320149dd": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 5,
        },
        "cluster": 5,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_79fa77f600e1320149dd",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_79fa77f600e1320149dd",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 5, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_79fa77f600e1320149dd",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_5339f8b94959bd44e7f4": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 6,
        },
        "cluster": 6,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_5339f8b94959bd44e7f4",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_5339f8b94959bd44e7f4",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 6, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_5339f8b94959bd44e7f4",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
    "cake_nvfp4_sparse_mla_decode_79fc3ddfe4254b845585": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "kv"],
            ["buffer", "indices"],
            ["buffer", "out"],
            ["parameter", "topk"],
            ["parameter", "keys_per_cta"],
            ["parameter", "qk_scale"],
            ["parameter", "out_scale"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "build": {
            "cluster": 8,
        },
        "cluster": 8,
        "ffi_entry": "run",
        "grid_rule": {
            "kind": "tokens_by_cluster",
        },
        "kernel": "cake_nvfp4_sparse_mla_decode_79fc3ddfe4254b845585",
        "kernel_symbol": "kernel_cake_nvfp4_sparse_mla_decode_79fc3ddfe4254b845585",
        "keys_per_stage": 32,
        "launch": {
            "block": (448, 1, 1),
            "cluster": (1, 8, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 226816,
            "use_pdl": False,
        },
        "library": "cake_nvfp4_sparse_mla_decode_79fc3ddfe4254b845585",
        "logical": {
            "indices": "indices",
            "keys_per_cta": "keys_per_cta",
            "kv": "kv",
            "out": "out",
            "out_scale": "out_scale",
            "q": "q",
            "qk_scale": "qk_scale",
            "topk": "topk",
        },
        "max_keys_per_cta": 2048,
        "module": "tart.mla_decode_dsv32_sparse_nvfp4_kv_fp8_q.sm100_sm103.flashinfer_blackwell_nvfp4_ds_mla_sparse_decode",
        "occupancy_entry": "max_active_clusters",
        "workspace_rule": None,
    },
}


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target; the capability is immutable per process)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake NVFP4 sparse MLA decode target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def record_for(cluster: int) -> str:
    """The name of the one record of ``cluster`` CTAs per token (``RuntimeError``
    when none or more than one is: the route has one program per cluster size)."""
    names = [
        name for name, record in MODULES.items() if int(record["cluster"]) == cluster
    ]
    if len(names) != 1:
        raise RuntimeError(
            f"Cake NVFP4 sparse MLA decode delivers {len(names)} programs of cluster "
            f"size {cluster}; the delivered cluster sizes are {CLUSTER_SIZES}"
        )
    return names[0]


def registered_clusters(arch: str) -> set[int]:
    """Cluster sizes whose program is delivered for ``arch``."""
    return {
        int(record["cluster"])
        for record in MODULES.values()
        if arch in record["arches"]
    }


def select_module(arch: str, cluster: int) -> str:
    """The record name of the program of ``cluster`` CTAs per token delivered for
    ``arch`` (``NotImplementedError`` when this checkout delivers none)."""
    if int(cluster) not in registered_clusters(arch):
        raise NotImplementedError(
            f"The generated NVFP4 sparse MLA decode program (cluster size {cluster}) "
            f"is not delivered for {arch} in this checkout"
        )
    return record_for(int(cluster))


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake NVFP4 sparse MLA decode record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch`` (its ``run``
    and ``max_active_clusters`` entries): the prebuilt ``aot/`` library when a
    builder compiled it, else FlashInfer's JIT build of the same sources and
    flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    return cake_build.jit_specs(arch)


def launch_grid(record: dict[str, Any], *, num_tokens: int) -> tuple[int, int, int]:
    """Launch grid of one record for ``num_tokens`` query tokens under its
    ``tokens_by_cluster`` rule: one cluster of the record's ``cluster`` CTAs per
    token, the cluster along the grid's y axis."""
    rule = record["grid_rule"]
    if rule["kind"] != "tokens_by_cluster":
        raise ValueError(
            f"unknown Cake NVFP4 sparse MLA decode grid rule: {rule['kind']!r}"
        )
    num_tokens, cluster = int(num_tokens), int(record["cluster"])
    if num_tokens <= 0:
        raise ValueError(
            f"the grid rule needs a positive token count, got num_tokens={num_tokens}"
        )
    return (num_tokens, cluster, 1)


def launch_args(
    record: dict[str, Any], values: dict[str, Any], grid: tuple[int, int, int]
) -> tuple[Any, ...]:
    """The positional arguments of the record's ``run`` entry: ``values`` by the
    logical name of each kernel parameter of ``arg_plan`` and the grid axes."""
    logical = record["logical"]
    args: list[Any] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(int(grid["xyz".index(key[-1])]))
            continue
        if key not in logical:
            raise ValueError(f"kernel parameter {key!r} has no logical name")
        try:
            args.append(values[logical[key]])
        except KeyError:
            raise ValueError(
                f"the launch needs logical argument {logical[key]!r}"
            ) from None
    return tuple(args)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid`` (the
    record's :func:`launch_grid`) on the current torch stream of the device."""
    record = MODULES[name]
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*launch_args(record, values, grid))


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CLUSTER_SIZES",
    "MODULES",
    "arch_for_capability",
    "device_arch",
    "launch_args",
    "launch_grid",
    "library_name",
    "load",
    "record_for",
    "registered_clusters",
    "run_module",
    "select_module",
    "sm_count",
    "specs_for_arch",
]
