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

Registry and loader glue of the Cake dense NVFP4 MLA decode on Rubin (SM107):
the query quantizer, the mixed-cluster decode (its two builds) and the
split-KV merge (its three warps-per-row instantiations).

Every record of ``MODULES`` is one mechanically exported program: a CUDA
device translation unit plus a tvm-ffi binding, compiled for ``sm_107a``.  A
record carries the host plan only: the program's ``library`` stem (its
``build.json`` entry is ``<library>_<arch>``), the architectures it is
delivered for, the ``stage`` of the launch order it fills and its ``variant``,
the ``select_rule`` that picks one variant of a stage from the host-known
plan scalars, the ``grid_rule`` that shapes its launch, the kernel-parameter
-> logical-argument map and the argument plan of its ``run`` entry.  What to
compile -- sources, digests, nvcc flags -- is recorded once in ``build.json``
next to this file; :mod:`.cake_build` serves the library from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's
JIT from the same sources and flags.

The rules are data, evaluated here (:func:`select_record`, :func:`launch_grid`)
from three host-known plan scalars: the packed (token, head) row count
``rows``, the work-item count ``items`` and the count ``n_full`` of full-tile
items among them.  No rule reads a tensor.  The ``decode`` stage has two
builds of one kernel: ``decode`` (``mixed``) runs clusters of four on the
full-tile items and the pair body on the double items of half tiles;
``decode_full`` (``full``) is the same kernel built for a plan without half
tiles (``n_full == items``), whose four-CTA body skips the double-item test at
each item's end.  The ``merge`` stage has three instantiations of one kernel,
chosen by the row count (four warps per row up to 1024 rows, two up to 2048,
one beyond); the host launches it only when the plan has more than one
split.  The ``quantize`` stage has one program.

The derived leaves of a record (``kernel_symbol``, ``launch``, ``arg_plan``,
``logical``) are rewritten by the exporter from the rendered binding; the
other fields are the host plan and never change.
"""

from __future__ import annotations

import functools
from typing import Any

import tvm_ffi

# One verified build: R200 (``sm_107a``), compiled with its exact ``-gencode``
# flag set (``build.json`` ``arch_flags``); no family fallback.
ARCHES = ("sm_107a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 7): "sm_107a",
}
PREFIX = "cake_rubin_mla_nvfp4_decode"

# The launch order of the complete call: the BF16 -> NVFP4 query quantizer,
# the decode (one of its two builds) and, when the plan has more than one
# split, the split-KV merge behind it.
STAGES = ("quantize", "decode", "merge")
# The merge's split-slot bound: a row has at most this many partials (two per
# lane of the merge's statistics pass).
MAX_SPLITS = 64

# Logical argument names the host backend binds: the kernels' own parameter
# names (``record["logical"]`` is the identity map).  Decode: ``QN`` / ``QS`` /
# ``QR`` the query's packed E2M1 bytes, E4M3 block scales and E4M3 rope as 2-D
# ``[rows, 256 | 32 | 64]`` uint8 views (TMA operands), ``CKV`` / ``CKS`` /
# ``KPE`` the cache's as ``[pages * 64, 256 | 32 | 64]`` uint8 views (strided
# rows allowed), ``ckv_words`` / ``cks_words`` the latent and block scales as
# int32 words from the same storage, ``O`` BF16 ``[rows, 512]``, ``LSE`` FP32
# ``[rows]`` (natural log), ``partial_o`` BF16 ``[rows, max_splits, 512]``,
# ``partial_lse`` FP32 ``[rows, max_splits]`` (log2 domain), ``q_scale`` FP32
# ``[rows]``, ``item_table`` int32 ``[items, 6]``, ``page_table`` int32
# ``[B, max_pages]``, ``seq_lens`` int32 ``[B]`` (this rank's local lengths),
# ``q_indptr`` int32 ``[B + 1]``, ``row_limit`` int32 ``[rows]``, ``O_T`` /
# ``O_T16`` the output again as the epilogue's 32- and 16-row store maps,
# ``PO_T`` / ``PO_T16`` the partials again.  Merge: ``partial_o`` /
# ``partial_lse`` / ``O`` as above and ``lse`` FP32 ``[rows]``.  Quantizer:
# ``q`` BF16 ``[rows, 576]``, ``q_nope`` uint32 ``[rows, 64]``, ``q_sf`` uint8
# ``[rows, 32]``, ``q_rope`` uint32 ``[rows, 16]``, ``q_scale`` FP32 ``[rows]``.
LOGICAL_TENSORS = (
    "QN",
    "QS",
    "QR",
    "CKV",
    "CKS",
    "KPE",
    "ckv_words",
    "cks_words",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "q_scale",
    "item_table",
    "page_table",
    "seq_lens",
    "q_indptr",
    "row_limit",
    "O_T",
    "PO_T",
    "O_T16",
    "PO_T16",
    "lse",
    "q",
    "q_nope",
    "q_sf",
    "q_rope",
)
LOGICAL_SCALARS = (
    "num_heads",
    "max_pages",
    "items",
    "max_splits",
    "ckv_stride_words",
    "cks_stride_words",
    "logit_scale",
    "o_factor",
    "n_full",
    "rows",
    "num_split",
    "has_lse",
    "c_nope",
    "c_rope",
    "kpe_scale",
    "ckv_scale",
)

# The kernel parameter names of each program, in parameter order: the identity
# ``logical`` map of its record and the order its ``arg_plan`` follows.
DECODE_PARAMETERS = (
    "QN",
    "QS",
    "QR",
    "CKV",
    "CKS",
    "KPE",
    "ckv_words",
    "cks_words",
    "O",
    "LSE",
    "partial_o",
    "partial_lse",
    "q_scale",
    "item_table",
    "page_table",
    "seq_lens",
    "q_indptr",
    "row_limit",
    "num_heads",
    "max_pages",
    "items",
    "max_splits",
    "ckv_stride_words",
    "cks_stride_words",
    "logit_scale",
    "o_factor",
    "O_T",
    "PO_T",
    "O_T16",
    "PO_T16",
    "n_full",
)
MERGE_PARAMETERS = (
    "partial_o",
    "partial_lse",
    "O",
    "lse",
    "rows",
    "num_split",
    "has_lse",
)
QUANTIZE_PARAMETERS = (
    "q",
    "q_nope",
    "q_sf",
    "q_rope",
    "q_scale",
    "rows",
    "c_nope",
    "c_rope",
    "kpe_scale",
    "ckv_scale",
)

# Filled mechanically by the exporter (the derived leaves ``kernel_symbol``, ``launch``, ``arg_plan``,
# ``logical``); the host plan (``stage``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this
# file with ``ast.literal_eval``.  Record names are the stages, suffixed with the variant where a stage has
# several programs.
MODULES: dict[str, dict[str, Any]] = {
    "decode": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "ctas_per_item", "ctas": 4},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rubin_mla_nvfp4_decode_211b49b794a7df09262e",
        "logical": {
            "CKS": "CKS",
            "CKV": "CKV",
            "KPE": "KPE",
            "LSE": "LSE",
            "O": "O",
            "O_T": "O_T",
            "O_T16": "O_T16",
            "PO_T": "PO_T",
            "PO_T16": "PO_T16",
            "QN": "QN",
            "QR": "QR",
            "QS": "QS",
            "cks_stride_words": "cks_stride_words",
            "cks_words": "cks_words",
            "ckv_stride_words": "ckv_stride_words",
            "ckv_words": "ckv_words",
            "item_table": "item_table",
            "items": "items",
            "logit_scale": "logit_scale",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "n_full": "n_full",
            "num_heads": "num_heads",
            "o_factor": "o_factor",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_scale": "q_scale",
            "row_limit": "row_limit",
            "seq_lens": "seq_lens",
        },
        "select_rule": {"kind": "all_full", "value": False},
        "stage": "decode",
        "variant": "mixed",
    },
    "decode_full": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "ctas_per_item", "ctas": 4},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rubin_mla_nvfp4_decode_8d0e1f24232434e74d49",
        "logical": {
            "CKS": "CKS",
            "CKV": "CKV",
            "KPE": "KPE",
            "LSE": "LSE",
            "O": "O",
            "O_T": "O_T",
            "O_T16": "O_T16",
            "PO_T": "PO_T",
            "PO_T16": "PO_T16",
            "QN": "QN",
            "QR": "QR",
            "QS": "QS",
            "cks_stride_words": "cks_stride_words",
            "cks_words": "cks_words",
            "ckv_stride_words": "ckv_stride_words",
            "ckv_words": "ckv_words",
            "item_table": "item_table",
            "items": "items",
            "logit_scale": "logit_scale",
            "max_pages": "max_pages",
            "max_splits": "max_splits",
            "n_full": "n_full",
            "num_heads": "num_heads",
            "o_factor": "o_factor",
            "page_table": "page_table",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "q_indptr": "q_indptr",
            "q_scale": "q_scale",
            "row_limit": "row_limit",
            "seq_lens": "seq_lens",
        },
        "select_rule": {"kind": "all_full", "value": True},
        "stage": "decode",
        "variant": "full",
    },
    "merge_w1": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 8},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_rubin_mla_nvfp4_decode_3a007c4be08b7b22f587",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {"kind": "rows_above", "value": 2048},
        "stage": "merge",
        "variant": "w1",
    },
    "merge_w2": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 4},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_rubin_mla_nvfp4_decode_6fd468e4645fbef52b7e",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {"kind": "rows_between", "lo": 1025, "hi": 2048},
        "stage": "merge",
        "variant": "w2",
    },
    "merge_w4": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 2},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_rubin_mla_nvfp4_decode_66cbdc7d0d294087c5da",
        "logical": {
            "O": "O",
            "has_lse": "has_lse",
            "lse": "lse",
            "num_split": "num_split",
            "partial_lse": "partial_lse",
            "partial_o": "partial_o",
            "rows": "rows",
        },
        "select_rule": {"kind": "rows_at_most", "value": 1024},
        "stage": "merge",
        "variant": "w4",
    },
    "quantize": {
        "arches": ["sm_107a"],
        "arg_plan": [],
        "ffi_entry": "run",
        "grid_rule": {"kind": "rows_per_cta", "rows": 4},
        "kernel_symbol": "",
        "launch": {
            "block": (0, 0, 0),
            "cluster": (0, 0, 0),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": True,
        },
        "library": "cake_rubin_mla_nvfp4_decode_302684a62be285e34842",
        "logical": {
            "c_nope": "c_nope",
            "c_rope": "c_rope",
            "ckv_scale": "ckv_scale",
            "kpe_scale": "kpe_scale",
            "q": "q",
            "q_nope": "q_nope",
            "q_rope": "q_rope",
            "q_scale": "q_scale",
            "q_sf": "q_sf",
            "rows": "rows",
        },
        "select_rule": {"kind": "always"},
        "stage": "quantize",
        "variant": None,
    },
}


def records_of_stage(stage: str) -> tuple[str, ...]:
    """The record names of ``stage`` in registry order."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; the stages are {STAGES}")
    return tuple(name for name, rec in MODULES.items() if rec["stage"] == stage)


def _admits(rule: dict[str, Any], *, rows: int, n_full: int, items: int) -> bool:
    kind = rule["kind"]
    if kind == "always":
        return True
    if kind == "all_full":
        return (n_full == items) == bool(rule["value"])
    if kind == "rows_above":
        return rows > int(rule["value"])
    if kind == "rows_between":
        return int(rule["lo"]) <= rows <= int(rule["hi"])
    if kind == "rows_at_most":
        return rows <= int(rule["value"])
    raise ValueError(f"unknown Cake Rubin NVFP4 MLA decode selection rule: {kind!r}")


def select_record(stage: str, *, rows: int, n_full: int, items: int) -> str:
    """The one record of ``stage`` the plan scalars select.

    ``rows`` is the packed (token, head) row count of the call, ``items`` the
    plan's work-item count and ``n_full`` how many of them are full-tile items
    (the plan lists those first).  The ``decode`` stage switches on
    ``all_full``: ``decode_full`` when ``n_full == items`` (no half tile),
    else ``decode``.  The ``merge`` stage switches on the row count
    (``rows_at_most`` / ``rows_between`` (inclusive) / ``rows_above``: four
    warps per row up to 1024 rows, two up to 2048, one beyond); whether the
    merge runs at all (``max_splits > 1``) is the host's decision, not a rule.
    ``always`` is the rule of a single-program stage.  Exactly one record
    admits every problem, or the registry is inconsistent.
    """
    admitted = [
        name
        for name in records_of_stage(stage)
        if _admits(
            MODULES[name]["select_rule"],
            rows=int(rows),
            n_full=int(n_full),
            items=int(items),
        )
    ]
    if len(admitted) != 1:
        raise ValueError(
            f"stage {stage!r}: records {admitted} admit rows={rows}, n_full={n_full}, "
            f"items={items}; exactly one must"
        )
    return admitted[0]


def launch_grid(name: str, *, items: int, rows: int) -> tuple[int, int, int]:
    """The launch grid of record ``name`` under its recorded grid rule.

    ``ctas_per_item`` launches ``(ctas * items, 1, 1)`` (the decode: four CTAs
    per work item, the clusters of four and the pairs decoded from the CTA
    id); ``rows_per_cta`` launches ``(ceil(rows / rows_per_cta), 1, 1)`` (the
    merge: ``8 / warps_per_row`` rows per CTA; the quantizer: four rows per
    CTA).
    """
    rule = MODULES[name]["grid_rule"]
    kind = rule["kind"]
    if kind == "ctas_per_item":
        if int(items) < 1:
            raise ValueError(f"record {name}: items={items} must be at least one")
        return (int(rule["ctas"]) * int(items), 1, 1)
    if kind == "rows_per_cta":
        if int(rows) < 1:
            raise ValueError(f"record {name}: rows={rows} must be at least one")
        per = int(rule["rows"])
        return (-(-int(rows) // per), 1, 1)
    raise ValueError(f"unknown Cake Rubin NVFP4 MLA decode grid rule: {kind!r}")


def merge_warps_per_row(rows: int) -> int:
    """Warps per row of the merge record ``select_record("merge", ...)`` picks for ``rows``: 4 up to 1024 rows, 2 up
    to 2048, 1 beyond (each step keeps the grid near 512 CTAs)."""
    return 4 if rows <= 1024 else 2 if rows <= 2048 else 1


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
            f"Cake Rubin NVFP4 MLA decode target; the exported architectures are {ARCHES}"
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
            f"Cake Rubin NVFP4 MLA decode record {name} is not delivered for {arch}; "
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
    """The record's plan as ``(logical name, -1)`` per kernel parameter (buffers,
    TMA operands and scalars alike) and ``(None, axis)`` per grid argument, in
    ``run`` order; computed once."""
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
    tensors and Python scalars; the record maps its kernel parameters onto that
    vocabulary and ``grid`` is the plan's launch grid (:func:`launch_grid`).
    The kernel runs on the current torch stream of the device.
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
    "DECODE_PARAMETERS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MAX_SPLITS",
    "MERGE_PARAMETERS",
    "MODULES",
    "PREFIX",
    "QUANTIZE_PARAMETERS",
    "STAGES",
    "arch_for_capability",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "merge_warps_per_row",
    "records_of_stage",
    "run_module",
    "select_record",
    "sm_count",
    "specs_for_arch",
]
