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

Registry and loader glue of the Cake QSA sparse paged GQA decode over an
NVFP4 KV cache for SM80 through SM90: the split-KV decode (its plain build and
its cooperative build, which merges the splits in the same launch) and the
separate split-KV merge with the sigmoid output gate.

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled for ``sm_80`` and
``sm_90a`` (one source, both architectures).  A record carries the host plan
only: the program's ``library`` stem (its ``build.json`` entry is
``<library>_<arch>``), the architectures it is delivered for, the ``stage``
of the launch order it fills and its ``variant``, the ``select_rule`` that
picks the records of one call from the host-known form, the ``grid_rule``
that shapes its launch, the ``workspace_rule`` naming the scratch it writes,
the kernel-parameter -> logical-argument map (the identity: a logical name is the kernel parameter's name)
and the argument plan of its ``run`` entry.  What to compile -- sources,
digests, nvcc flags -- is recorded once in ``build.json`` next to this file;
:mod:`.cake_build` serves the library from a prebuilt ``aot/`` file when a
builder compiled it and otherwise through FlashInfer's JIT from the same
sources and flags.  Every ``build.json`` entry records ``--use_fast_math``
(the flag the kernels were validated and measured with).

The rules are data, evaluated here (:func:`select_records`,
:func:`launch_grid`) from host-known scalars: the call's *form* (``"split"``:
the plain decode, then the merge when the call has more than one split;
``"fused"``: the cooperative decode alone, its CTAs merging the splits
through a counter per (row, KV head)), the row count ``rows``, the KV head
count ``kv_heads``, the query head count ``qo_heads`` and the split count
``splits``.  No rule reads a tensor.  Which form a call takes, and its split
count, is the host's decision (:mod:`.cake_backend`), not a rule.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

# The two verified builds, each compiled with its exact ``-gencode`` flag set
# (``build.json`` ``arch_flags``): ``sm_80`` serves every compute capability
# 8.x device, ``sm_90a`` the H100 (9.0).  No other capability is a target.
ARCHES = ("sm_80", "sm_90a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (8, 0): "sm_80",
    (8, 6): "sm_80",
    (8, 7): "sm_80",
    (8, 9): "sm_80",
    (9, 0): "sm_90a",
}
PREFIX = "cake_qsa_nvfp4"

# The launch order of a call: the decode (one of its two builds) and, in the
# split form with more than one split, the merge behind it.
STAGES = ("decode", "merge")
# The two forms a call takes (``select_rule`` kind ``form``).
FORMS = ("split", "fused")
# The merge's split bound: a (row, head) has at most this many partials (one
# per lane of the merge's four warps).  The host's split rule never exceeds it.
MAX_SPLITS = 128
# The merge's warps per CTA: above this many splits one CTA merges one head
# with every warp owning a chunk of splits; at or below, one warp per head and
# this many heads per CTA (``grid_rule`` kind ``rows_head_groups``).
MERGE_WARPS = 4
# The decode's tile: the selection is split into tiles of this many tokens.
TILE_TOKENS = 64
# The decode's block and the resident CTAs per SM its split rule counts on.
DECODE_THREADS = 128
DECODE_CTAS_PER_SM = 2

# Logical argument names the host backend binds: the kernels' own parameter
# names (``record["logical"]`` is the identity map).  Decode: ``q`` BF16
# ``[rows, qo_heads, 256]`` (strided), ``k_data`` / ``v_data`` uint8
# ``[blocks, page, kv_heads, 128]`` packed E2M1 (any strides with a contiguous
# last dim), ``k_sf`` / ``v_sf`` their E4M3 block scales as uint8 views
# ``[blocks, page, kv_heads, 16]``, ``indices`` int32 ``[rows, topk + 1]``,
# ``block_table`` int32 ``[requests, table_width]``, ``token_to_req`` int32
# ``[rows]``, ``gate`` BF16 ``[rows, qo_heads, 256]`` (dense), ``out`` BF16
# ``[rows, qo_heads, 256]`` (strided), ``partial_out`` FP32 ``[splits, rows,
# qo_heads, 256]`` and ``partial_lse`` FP32 ``[splits, rows, qo_heads]``
# (``None`` at one split), ``merge_counters`` uint32 ``[rows * kv_heads]``
# (the fused build's group counters; ``None`` for the plain build).  Merge:
# ``partial_out`` / ``partial_lse`` / ``gate`` / ``out`` as above.
LOGICAL_TENSORS = (
    "q",
    "k_data",
    "v_data",
    "k_sf",
    "v_sf",
    "indices",
    "block_table",
    "token_to_req",
    "gate",
    "out",
    "partial_out",
    "partial_lse",
    "merge_counters",
)
LOGICAL_SCALARS = (
    "q_stride_row",
    "q_stride_head",
    "data_stride_block",
    "data_stride_token",
    "data_stride_head",
    "sf_stride_block",
    "sf_stride_token",
    "sf_stride_head",
    "indices_stride_row",
    "table_stride_req",
    "out_stride_row",
    "out_stride_head",
    "score_scale",
    "output_scale",
    "num_rows",
    "num_requests",
    "num_blocks",
    "page_div",
    "table_width",
    "topk",
    "group_size",
    "num_qo_heads",
    "num_tiles",
    "num_splits",
)

# The kernel parameter names of each program, in parameter order: the identity
# ``logical`` map of its record and the order its ``arg_plan`` follows.  The
# decode's plain and cooperative builds take the same parameters.
DECODE_PARAMETERS = (
    "q",
    "k_data",
    "v_data",
    "k_sf",
    "v_sf",
    "indices",
    "block_table",
    "token_to_req",
    "gate",
    "out",
    "partial_out",
    "partial_lse",
    "merge_counters",
    "q_stride_row",
    "q_stride_head",
    "data_stride_block",
    "data_stride_token",
    "data_stride_head",
    "sf_stride_block",
    "sf_stride_token",
    "sf_stride_head",
    "indices_stride_row",
    "table_stride_req",
    "out_stride_row",
    "out_stride_head",
    "score_scale",
    "output_scale",
    "num_rows",
    "num_requests",
    "num_blocks",
    "page_div",
    "table_width",
    "topk",
    "group_size",
    "num_qo_heads",
    "num_tiles",
    "num_splits",
)
MERGE_PARAMETERS = (
    "partial_out",
    "partial_lse",
    "gate",
    "out",
    "out_stride_row",
    "out_stride_head",
    "num_splits",
    "num_qo_heads",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``,
# ``launch``, ``arg_plan``, ``logical``); the host plan (``stage``, ``variant``,
# ``select_rule``, ``grid_rule``, ``workspace_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal:
# FlashInfer's manifest checker and the exporter read this file with
# ``ast.literal_eval``.  Record names are the stages, suffixed with the variant
# where a stage has several programs.
MODULES: dict[str, dict[str, Any]] = {
    "decode": {
        "arches": ["sm_80", "sm_90a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_5bf192272d1a4b0d1f89",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_5bf192272d1a4b0d1f89",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "split"},
        "stage": "decode",
        "variant": "plain",
        "workspace_rule": {"kind": "split_partials"},
    },
    "decode_coop": {
        "arches": ["sm_80", "sm_90a"],
        "arg_plan": [
            ["buffer", "q"],
            ["buffer", "k_data"],
            ["buffer", "v_data"],
            ["buffer", "k_sf"],
            ["buffer", "v_sf"],
            ["buffer", "indices"],
            ["buffer", "block_table"],
            ["buffer", "token_to_req"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "merge_counters"],
            ["parameter", "q_stride_row"],
            ["parameter", "q_stride_head"],
            ["parameter", "data_stride_block"],
            ["parameter", "data_stride_token"],
            ["parameter", "data_stride_head"],
            ["parameter", "sf_stride_block"],
            ["parameter", "sf_stride_token"],
            ["parameter", "sf_stride_head"],
            ["parameter", "indices_stride_row"],
            ["parameter", "table_stride_req"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "score_scale"],
            ["parameter", "output_scale"],
            ["parameter", "num_rows"],
            ["parameter", "num_requests"],
            ["parameter", "num_blocks"],
            ["parameter", "page_div"],
            ["parameter", "table_width"],
            ["parameter", "topk"],
            ["parameter", "group_size"],
            ["parameter", "num_qo_heads"],
            ["parameter", "num_tiles"],
            ["parameter", "num_splits"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_kv_heads_splits"},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_4e706fc93843b564281d",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 49792,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_4e706fc93843b564281d",
        "logical": {
            "block_table": "block_table",
            "data_stride_block": "data_stride_block",
            "data_stride_head": "data_stride_head",
            "data_stride_token": "data_stride_token",
            "gate": "gate",
            "group_size": "group_size",
            "indices": "indices",
            "indices_stride_row": "indices_stride_row",
            "k_data": "k_data",
            "k_sf": "k_sf",
            "merge_counters": "merge_counters",
            "num_blocks": "num_blocks",
            "num_qo_heads": "num_qo_heads",
            "num_requests": "num_requests",
            "num_rows": "num_rows",
            "num_splits": "num_splits",
            "num_tiles": "num_tiles",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "output_scale": "output_scale",
            "page_div": "page_div",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
            "q": "q",
            "q_stride_head": "q_stride_head",
            "q_stride_row": "q_stride_row",
            "score_scale": "score_scale",
            "sf_stride_block": "sf_stride_block",
            "sf_stride_head": "sf_stride_head",
            "sf_stride_token": "sf_stride_token",
            "table_stride_req": "table_stride_req",
            "table_width": "table_width",
            "token_to_req": "token_to_req",
            "topk": "topk",
            "v_data": "v_data",
            "v_sf": "v_sf",
        },
        "select_rule": {"kind": "form", "value": "fused"},
        "stage": "decode",
        "variant": "coop",
        "workspace_rule": {"kind": "fused_partials_counters"},
    },
    "merge": {
        "arches": ["sm_80", "sm_90a"],
        "arg_plan": [
            ["buffer", "partial_out"],
            ["buffer", "partial_lse"],
            ["buffer", "gate"],
            ["buffer", "out"],
            ["parameter", "out_stride_row"],
            ["parameter", "out_stride_head"],
            ["parameter", "num_splits"],
            ["parameter", "num_qo_heads"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_head_groups", "warps": 4},
        "kernel_symbol": "kernel_cake_qsa_nvfp4_fb94b71eaa79ef2379de",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_qsa_nvfp4_fb94b71eaa79ef2379de",
        "logical": {
            "gate": "gate",
            "num_qo_heads": "num_qo_heads",
            "num_splits": "num_splits",
            "out": "out",
            "out_stride_head": "out_stride_head",
            "out_stride_row": "out_stride_row",
            "partial_lse": "partial_lse",
            "partial_out": "partial_out",
        },
        "select_rule": {"kind": "form", "value": "split"},
        "stage": "merge",
        "variant": None,
        "workspace_rule": None,
    },
}


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    return tuple(name for name, rec in MODULES.items() if rec["stage"] == stage)


def _admits(rule: dict[str, Any], *, form: str) -> bool:
    kind = rule["kind"]
    if kind == "form":
        return rule["value"] == form
    raise ValueError(f"unknown Cake QSA NVFP4 decode selection rule: {kind!r}")


def select_record(stage: str, *, form: str) -> str:
    """The one record of ``stage`` the call's ``form`` selects.

    ``form`` is ``"split"`` (the plain decode and, above one split, the merge)
    or ``"fused"`` (the cooperative decode, which merges its splits itself).
    The ``merge`` stage has no record in the fused form (:func:`select_records`
    leaves it out); asking for one is an error.  Exactly one record of a stage
    admits a form it serves, or the registry is inconsistent.
    """
    if form not in FORMS:
        raise ValueError(f"unknown form {form!r}; the forms are {FORMS}")
    admitted = [
        name
        for name in records_of_stage(stage)
        if _admits(MODULES[name]["select_rule"], form=form)
    ]
    if len(admitted) != 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} admit form={form!r}; exactly one must"
        )
    return admitted[0]


def select_records(*, form: str, splits: int) -> tuple[str, ...]:
    """The records one call launches, in launch order: the decode of ``form``
    and, in the split form with ``splits > 1``, the merge."""
    if int(splits) < 1:
        raise ValueError(f"splits must be at least 1, got {splits}")
    names = [select_record("decode", form=form)]
    if form == "split" and int(splits) > 1:
        names.append(select_record("merge", form=form))
    return tuple(names)


def merge_heads_per_cta(splits: int) -> int:
    """The heads one merge CTA serves: one above ``MERGE_WARPS`` splits (a warp
    per chunk of splits), ``MERGE_WARPS`` at or below (a warp per head)."""
    return 1 if int(splits) > MERGE_WARPS else MERGE_WARPS


def launch_grid(
    name: str, *, rows: int, kv_heads: int, qo_heads: int, splits: int
) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule.

    ``rows_kv_heads_splits`` launches ``(rows, kv_heads, splits)`` (the decode:
    one CTA per (row, KV head, split)); ``rows_head_groups`` launches ``(rows,
    ceil(qo_heads / merge_heads_per_cta(splits)), 1)`` (the merge).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    if int(rows) < 1 or int(kv_heads) < 1 or int(qo_heads) < 1 or int(splits) < 1:
        raise ValueError(
            f"record {name}: rows={rows}, kv_heads={kv_heads}, qo_heads={qo_heads}, "
            f"splits={splits} must all be at least one"
        )
    if kind == "rows_kv_heads_splits":
        return (int(rows), int(kv_heads), int(splits))
    if kind == "rows_head_groups":
        if int(rule["warps"]) != MERGE_WARPS:
            raise ValueError(
                f"record {name}: grid rule warps {rule['warps']} is not the merge's {MERGE_WARPS}"
            )
        per = merge_heads_per_cta(splits)
        return (int(rows), -(-int(qo_heads) // per), 1)
    raise ValueError(f"unknown Cake QSA NVFP4 decode grid rule: {kind!r}")


def arch_for_capability(capability: tuple[int, int]) -> str | None:
    """The exported architecture of one device capability, if any."""
    values = tuple(int(value) for value in capability)
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when the device
    is not an exported target)."""
    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported "
            f"Cake QSA NVFP4 decode target; the exported architectures are {ARCHES}"
        )
    return arch


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""
    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake QSA NVFP4 decode record {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    from . import (
        cake_build,
    )  # written by the exporter beside build.json; absent before the first render

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""
    from . import (
        cake_build,
    )  # written by the exporter beside build.json; absent before the first render

    return cake_build.jit_specs(arch)


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter (buffers
    and scalars alike) and ``(None, axis)`` per grid argument, in ``run``
    order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
    if not record["arg_plan"]:
        raise RuntimeError(
            f"record {name} has no argument plan: the package was not exported "
            "(the exporter writes the derived leaves of cake_jit.MODULES)"
        )
    plan: list[tuple[str | None, int]] = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            plan.append((None, "xyz".index(key[-1])))
        else:
            if key not in logical:
                raise ValueError(
                    f"record {name}: kernel parameter {key!r} has no logical name"
                )
            plan.append((logical[key], -1))
    return tuple(plan)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    grid: tuple[int, int, int],
    device_index: int,
) -> None:
    """Launch record ``name`` with logical argument ``values`` on ``grid``.

    ``values`` maps the names of ``LOGICAL_TENSORS`` / ``LOGICAL_SCALARS`` to
    tensors (``None`` for an absent optional tensor) and Python scalars; the
    record maps its kernel parameters onto that vocabulary and ``grid`` is the
    call's launch grid (:func:`launch_grid`).  The kernel runs on the current
    torch stream of the device.
    """
    record = MODULES[name]
    args = []
    for logical_name, axis in _argument_plan(name):
        if logical_name is None:
            args.append(int(grid[axis]))
        else:
            try:
                args.append(values[logical_name])
            except KeyError:
                raise ValueError(
                    f"record {name} needs logical argument {logical_name!r}"
                ) from None
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "DECODE_CTAS_PER_SM",
    "DECODE_PARAMETERS",
    "DECODE_THREADS",
    "FORMS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_SPLITS",
    "MERGE_PARAMETERS",
    "MERGE_WARPS",
    "MODULES",
    "PREFIX",
    "STAGES",
    "TILE_TOKENS",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "merge_heads_per_cta",
    "records_of_stage",
    "run_module",
    "select_record",
    "select_records",
    "sm_count",
    "specs_for_arch",
]
