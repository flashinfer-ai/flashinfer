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

Registry and loader glue of the Cake SM110 (Thor) GQA decode programs.

Every record of ``MODULES`` is one mechanically exported program: a CUDA device
translation unit plus a tvm-ffi binding, compiled for ``sm_110a``.  A record
carries the host plan only: the program's ``library`` stem (its ``build.json``
entry is ``<library>_sm_110a``), the architectures it is delivered for, the
``stage`` it fills, its ``variant`` (the route), the ``select_rule`` that picks
it from the host-known shape facts, the ``grid_rule`` that shapes its launch,
the kernel-parameter -> logical-argument map and the argument plan of its
``run`` entry.  What to compile -- sources, digests, nvcc flags -- is recorded
once in ``build.json`` next to this file; :mod:`.cake_build` serves the library
from a prebuilt ``aot/`` file when a builder compiled it and otherwise through
FlashInfer's JIT from the same sources and flags.

The geometry is fixed: Hq = 32 query heads, Hkv = 8 KV heads, D = 128, one
FP16 query token per request over a contiguous FP16 cache.  The rules are
data, evaluated here (:func:`select_record`, :func:`launch_grid`) from the
batch, the cache capacity and the caller's split-count request.  A ``regions``
rule is a list of rows; a row admits a batch range and a capacity range for one
class of split request (``one``: ``num_splits`` None or 1; ``many``: an explicit
count above 1) and carries the split count the route launches with (an integer
of ``ACCEPTED_SPLITS`` on a ``one`` row, ``"runtime"`` -- the caller's count --
on a ``many`` row); the ``otherwise`` rule of each class, with its own count,
takes what no row admits.  :func:`select_route` answers ``(record, count)``;
every ``one`` row launches one split today: one request over at most
``SHORT_CAPACITY_MAX`` keys takes ``short`` (the CUDA-core program: one
256-thread CTA per request and KV head, the K/V rows read as 16-byte vectors
into registers before the length lands), one request over 65..128 keys ``long``
(the two-slot K/V ring), one request over 129..256 keys and two requests over at
most 512 keys the one-split 64-token-tile build of the split-KV kernel
(``split_n64_direct``), one or two requests over more keys its four-slot-ring
twin (``split_n64_r4_direct``), three or more requests the one-split
32-token-tile build (``split_n32_direct``); an explicit split count above one
takes ``split_n64`` for one or two requests under a capacity of 4096 and
``split_n32`` otherwise, whose launch grid carries the count.  The public ``num_splits`` keeps the accepted
set ``ACCEPTED_SPLITS``; another value raises ``ValueError`` as before.  A split
record's ``workspace`` leaf names the four buffers the host allocates
(``partial_O``, ``partial_max``, ``partial_sum``, ``completed``; the counters
zero at prepare and reset by the kernel, so a captured graph replays without a
reset node).  No rule reads a tensor.

The derived leaves of a record (``arg_plan``, ``logical``) are rewritten by
the exporter from the rendered binding; the other fields are the host plan and
never change.
"""

from __future__ import annotations

import functools
from typing import Any

from . import cake_build

# One verified build: Thor (``sm_110a``, compute capability 11.0), compiled with
# its exact ``-gencode`` flag set (``build.json`` ``arch_flags``); no fallback.
ARCHES = ("sm_110a",)
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {(11, 0): "sm_110a"}

NUM_Q_HEADS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
# One request over capacities through this value takes the short route.
SHORT_CAPACITY_MAX = 64
# The public ``num_splits`` values (the kernel takes any count; the API keeps this set).
ACCEPTED_SPLITS = (1, 2, 4, 8, 10, 16)

# Logical argument names the host backend binds; every record maps its kernel
# parameter names onto this vocabulary (``record["logical"]``).  ``Q`` is the
# TMA descriptor source over the query viewed ``[B, 4, 8, 128]`` (the four
# query heads of a KV head side by side), ``K`` and ``V`` the two selectors of
# the cache ``[B, 2, 8, capacity, 128]``; ``softmax_scale_log2`` is
# ``q_scale / sqrt(128) / ln 2``; the split records add the workspace
# ``partial_O`` f32 ``[B, 32, S, 128]``, ``partial_max`` / ``partial_sum`` f32
# ``[B, 32, S]`` and ``completed`` u32 ``[B * 8]``.
LOGICAL_TENSORS = (
    "Q",
    "K",
    "V",
    "O",
    "sequence_lengths",
    "partial_O",
    "partial_max",
    "partial_sum",
    "completed",
)
LOGICAL_SCALARS = ("softmax_scale_log2",)

# Filled mechanically by the exporter (the derived leaves ``arg_plan``, ``logical``); the host plan
# (``stage``, ``variant``, ``select_rule``, ``grid_rule``, ``library``, ``arches``)
# is written by hand.  Every value is a literal: FlashInfer's manifest checker and the exporter read this file
# with ``ast.literal_eval``.  A record is named by its route.
MODULES: dict[str, dict[str, Any]] = {
    "short": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "batch_heads", "x_factor": 1},
        "library": "cake_sm110_gqa_decode_01f1434b7424417bec5c",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {
            "kind": "regions",
            "regions": [
                {
                    "splits": "one",
                    "batch": [1, 1],
                    "capacity": [1, 64],
                    "num_splits": 1,
                },
            ],
        },
        "stage": "decode",
        "variant": "short",
    },
    "long": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "batch_heads", "x_factor": 1},
        "library": "cake_sm110_gqa_decode_daa3a415e53b721fe538",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {
            "kind": "regions",
            "regions": [
                {
                    "splits": "one",
                    "batch": [1, 1],
                    "capacity": [65, 128],
                    "num_splits": 1,
                },
            ],
        },
        "stage": "decode",
        "variant": "long",
    },
    "split_n32_direct": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "completed"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "split_pairs", "num_splits": 1},
        "library": "cake_sm110_gqa_decode_d8f05d2bfbbde5459902",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "completed": "completed",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {"kind": "otherwise", "splits": "one", "num_splits": 1},
        "stage": "decode",
        "variant": "split_n32_direct",
        "workspace": ["partial_O", "partial_max", "partial_sum", "completed"],
    },
    "split_n64_direct": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "completed"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "split_pairs", "num_splits": 1},
        "library": "cake_sm110_gqa_decode_e5128e91902434559902",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "completed": "completed",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {
            "kind": "regions",
            "regions": [
                {
                    "splits": "one",
                    "batch": [1, 1],
                    "capacity": [129, 256],
                    "num_splits": 1,
                },
                {
                    "splits": "one",
                    "batch": [2, 2],
                    "capacity": [1, 512],
                    "num_splits": 1,
                },
            ],
        },
        "stage": "decode",
        "variant": "split_n64_direct",
        "workspace": ["partial_O", "partial_max", "partial_sum", "completed"],
    },
    "split_n64_r4_direct": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "completed"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "split_pairs", "num_splits": 1},
        "library": "cake_sm110_gqa_decode_f0ed92bc2f7a1e380d70",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "completed": "completed",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {
            "kind": "regions",
            "regions": [
                {
                    "splits": "one",
                    "batch": [1, 1],
                    "capacity": [257, None],
                    "num_splits": 1,
                },
                {
                    "splits": "one",
                    "batch": [2, 2],
                    "capacity": [513, None],
                    "num_splits": 1,
                },
            ],
        },
        "stage": "decode",
        "variant": "split_n64_r4_direct",
        "workspace": ["partial_O", "partial_max", "partial_sum", "completed"],
    },
    "split_n32": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "completed"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "split_pairs", "num_splits": "runtime"},
        "library": "cake_sm110_gqa_decode_b7f31235adbc7ac497d0",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "completed": "completed",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {"kind": "otherwise", "splits": "many", "num_splits": "runtime"},
        "stage": "decode",
        "variant": "split_n32",
        "workspace": ["partial_O", "partial_max", "partial_sum", "completed"],
    },
    "split_n64": {
        "arches": ["sm_110a"],
        "arg_plan": [
            ["buffer", "Q"],
            ["buffer", "K"],
            ["buffer", "V"],
            ["buffer", "partial_O"],
            ["buffer", "partial_max"],
            ["buffer", "partial_sum"],
            ["buffer", "completed"],
            ["buffer", "O"],
            ["buffer", "sequence_lengths"],
            ["parameter", "softmax_scale_log2"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {"kind": "split_pairs", "num_splits": "runtime"},
        "library": "cake_sm110_gqa_decode_3f0576c3791ebdd23b5a",
        "logical": {
            "K": "K",
            "O": "O",
            "Q": "Q",
            "V": "V",
            "completed": "completed",
            "partial_O": "partial_O",
            "partial_max": "partial_max",
            "partial_sum": "partial_sum",
            "sequence_lengths": "sequence_lengths",
            "softmax_scale_log2": "softmax_scale_log2",
        },
        "select_rule": {
            "kind": "regions",
            "regions": [
                {
                    "splits": "many",
                    "batch": [1, 2],
                    "capacity": [1, 4095],
                    "num_splits": "runtime",
                },
            ],
        },
        "stage": "decode",
        "variant": "split_n64",
        "workspace": ["partial_O", "partial_max", "partial_sum", "completed"],
    },
}


def _split_class(num_splits: int | None) -> str:
    """``one`` for no request or a single split, ``many`` for an explicit count above one."""
    return "one" if num_splits in (None, 1) else "many"


def _inside(bounds: dict[str, Any], batch: int, capacity: int) -> bool:
    (b_lo, b_hi), (c_lo, c_hi) = bounds["batch"], bounds["capacity"]
    return int(b_lo) <= batch <= (batch if b_hi is None else int(b_hi)) and int(
        c_lo
    ) <= capacity <= (capacity if c_hi is None else int(c_hi))


def _count(row_count: Any, num_splits: int | None, name: str) -> int:
    if row_count == "runtime":
        if num_splits is None or type(num_splits) is not int or num_splits < 1:
            raise ValueError(
                f"record {name}: a run-time split count needs the caller's positive num_splits"
            )
        return num_splits
    return int(row_count)


def select_route(
    *, batch: int, capacity: int, num_splits: int | None = None
) -> tuple[str, int]:
    """The record that serves a decode problem of ``batch`` requests over a
    cache of ``capacity`` tokens and the split count it launches with.
    ``num_splits`` ``None`` or ``1`` leaves both to the size rule: the one row
    of the ``one`` class admitting the shape names the record and the count
    (one split on every row today); an accepted count above one takes the
    run-time-split program of the shape's tile with that many splits; a value
    outside ``ACCEPTED_SPLITS`` raises ``ValueError`` (no silent ignore).  When
    no row admits the shape, the class's ``otherwise`` record answers with its
    count."""
    if batch < 1 or capacity < 1:
        raise ValueError("batch and capacity must be positive")
    if num_splits is not None and (
        type(num_splits) is not int or num_splits not in ACCEPTED_SPLITS
    ):
        raise ValueError("num_splits must be None or an integer in {1,2,4,8,10,16}")
    wanted = _split_class(num_splits)
    hits = [
        (name, row)
        for name, record in MODULES.items()
        if record["select_rule"]["kind"] == "regions"
        for row in record["select_rule"]["regions"]
        if row["splits"] == wanted and _inside(row, batch, capacity)
    ]
    if len(hits) > 1:
        names = [name for name, _ in hits]
        raise ValueError(
            f"records {names} all admit batch={batch}, capacity={capacity}, num_splits={num_splits}"
        )
    if hits:
        ((name, row),) = hits
        return name, _count(row["num_splits"], num_splits, name)
    fallback = [
        (name, record["select_rule"])
        for name, record in MODULES.items()
        if record["select_rule"]["kind"] == "otherwise"
        and record["select_rule"]["splits"] == wanted
    ]
    if len(fallback) != 1:
        raise ValueError(
            f"the registry has {len(fallback)} fallback records of class {wanted!r}; one is expected"
        )
    ((name, rule),) = fallback
    return name, _count(rule["num_splits"], num_splits, name)


def select_record(*, batch: int, capacity: int, num_splits: int | None = None) -> str:
    """The record of :func:`select_route` (the host's size rule; the split
    count is the route's)."""
    return select_route(batch=batch, capacity=capacity, num_splits=num_splits)[0]


def launch_grid(
    name: str, *, batch: int, num_splits: int | None = None
) -> tuple[int, int, int]:
    """The launch grid of record ``name``: under its ``batch_heads`` grid rule
    ``(batch * NUM_KV_HEADS * x_factor, 1, 1)``, one CTA per request and KV
    head; under its ``split_pairs`` rule ``(num_splits, batch * NUM_KV_HEADS,
    1)``, the split count the rule's literal (the one-split builds) or the
    caller's (``"runtime"``; a positive int is required)."""
    rule = MODULES[name]["grid_rule"]
    if rule["kind"] == "batch_heads":
        if num_splits not in (None, 1):
            raise ValueError(
                f"record {name}: a one-CTA route takes no split count, got {num_splits!r}"
            )
        return (int(batch) * NUM_KV_HEADS * int(rule.get("x_factor", 1)), 1, 1)
    if rule["kind"] == "split_pairs":
        fixed = rule["num_splits"]
        if fixed == "runtime":
            if num_splits is None or type(num_splits) is not int or num_splits < 1:
                raise ValueError(
                    f"record {name}: a run-time split launch needs a positive num_splits"
                )
            splits = num_splits
        else:
            if num_splits not in (None, int(fixed)):
                raise ValueError(
                    f"record {name}: its split count is fixed at {fixed}, got {num_splits!r}"
                )
            splits = int(fixed)
        return (splits, int(batch) * NUM_KV_HEADS, 1)
    raise ValueError(f"unknown Cake SM110 GQA decode grid rule: {rule['kind']!r}")


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
            f"Cake SM110 GQA decode target; the exported architectures are {ARCHES}"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of record ``name`` built for ``arch``: ``<library>_<arch>``."""
    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake SM110 GQA decode record {name} is not delivered for {arch}; its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``: the
    prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's JIT
    build of the same sources and flags (``cake_build.load_library``)."""
    return cake_build.load_library(library_name(name, arch))


@functools.cache
def _argument_plan(name: str) -> tuple[tuple[str | None, int], ...]:
    """The record's plan as ``(logical name, -1)`` per kernel parameter and
    ``(None, axis)`` per grid argument, in ``run`` order; computed once."""
    record = MODULES[name]
    logical = record["logical"]
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


def argument_list(
    name: str, values: dict[str, Any], *, grid: tuple[int, int, int]
) -> list[Any]:
    """The positional arguments of record ``name``'s ``run`` entry: ``values``
    (logical name -> tensor or Python scalar) in the plan's order, the three
    grid extents in place.  Bound once at prepare time so a replay interprets
    no plan."""
    args: list[Any] = []
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
    return args


def entry(name: str, device_index: int) -> Any:
    """The ``run`` entry of record ``name`` for the device's architecture."""
    return getattr(load(name, device_arch(device_index)), MODULES[name]["ffi_entry"])


__all__ = [
    "ACCEPTED_SPLITS",
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "HEAD_DIM",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "NUM_KV_HEADS",
    "NUM_Q_HEADS",
    "SHORT_CAPACITY_MAX",
    "arch_for_capability",
    "argument_list",
    "device_arch",
    "entry",
    "launch_grid",
    "library_name",
    "load",
    "select_record",
    "select_route",
]
