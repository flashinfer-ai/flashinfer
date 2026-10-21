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

Host plan and loader glue of the Cake packed width-4 depthwise causal
convolution + SiLU training kernels (SM100 / SM103 / SM107).

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only: the
kernel's ``library`` stem, the architectures it is delivered for, its launch
record, grid rule, workspace rule and the kernel-parameter -> logical-argument
map.  What to compile -- sources, digests, nvcc flags -- is recorded once in
``build.json`` next to this file; :mod:`.cake_build` serves the library
``<library>_<arch>`` from a prebuilt ``aot/`` file when a builder compiled it
and otherwise through FlashInfer's JIT from the same sources and flags,
verifying every source digest before the first build.

A route lists every program one public call may launch: ``fwd`` (``y`` from
``x``, ``weight`` and ``cu_seqlens``) and ``bwd`` (``dx`` and the deterministic
FP32 ``dweight`` in one cooperative launch).  The forward launches its route in
order.  The backward route of an architecture may carry several programs of
the kernel and launches exactly one per call, chosen by the channel count: a
record's ``select_rule`` (``channels_at_least``: ``C >= channels[arch]``)
names the channel counts it owns, of several owners the one of the highest
threshold runs, the record without a rule is the architecture's fallback, and
:func:`selected_modules` applies that rule (a pure function of the
architecture and ``C``; the host never reads the data).
Token and channel counts are runtime scalars, so a changed problem size never
recompiles anything; building every library of the architecture
(:func:`specs_for_arch`) before a CUDA Graph capture keeps the JIT out of the
capture.

The host plan is the three functions :func:`plan` (the kernel scalars),
:func:`launch_grid` (the grid) and :func:`partial_rows` (the backward
workspace rows); everything else derives from them.  The public workspace is
sized by the largest grid of the architecture's backward programs
(:func:`workspace_record`, the capacity owner); each program writes and
reduces the leading rows of its own grid.

The program name ``cake_causal_conv1d_train_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted>,
"module": <dotted kernel module>}`` of the kernel the program was rendered
from; the forward is built with its kernel file's default keywords; the
backward with ``vec=8, copy="tma2d"`` (eight channels, 16-byte vectors, per
thread; 2-D TMA row copies) on sm_100a and sm_103a, with ``vec=8,
min_blocks=2`` (two CTAs per SM) on sm_107a, and the backwards the
``channels_at_least`` rules select, builds of ``vec=8, copy="tma2d",
stage_tokens=8, min_blocks=2`` (eight-token stages, two CTAs per SM): as is
on sm_100a from C = 512 and on sm_103a from 768, with ``fold_overlap=True,
fast_roll=True`` (the dweight fold buffer over the stage slots, the stage body
as two rolled passes of four tokens) on sm_103a for 512 <= C < 768 and on
sm_107a for 512 <= C < 1024, as is on sm_107a for 1024 <= C < 2048, and
with ``fast_roll=True, roll_prefetch=True`` (each rolled pass loads the
next pass's rows before it computes) on sm_107a from 2048.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_causal_conv1d_train_f193c5247980992f4f81": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "y"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # The grid is the co-resident CTA count, ``ctas_per_sm x SM count``:
        # CTA ``b`` owns channel tile ``b % n_ch_tiles`` and, with the other
        # CTAs of that tile, an equal contiguous share of the token axis.  Any
        # positive grid is correct; ``ctas_per_sm`` is the measured
        # co-residency of the program: 4 CTAs per SM on B200 (sm_100a, 107
        # registers) and GB300 (sm_103a, 109 registers), the register limit
        # at 128 threads (ncu launch__occupancy_limit_registers); sm_107a is
        # taken as the same launch bound.  ``geometry`` is the kernel's build
        # geometry the host plan derives its scalars from.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_f193c5247980992f4f81",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_f193c5247980992f4f81",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "fwd",
        "workspace_rule": None,
    },
    "cake_causal_conv1d_train_21cc8e5b2b49282a47ef": {
        "arches": [
            "sm_100a",
            "sm_103a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # Cooperative (a grid-wide barrier precedes the fixed-order dweight
        # reduction): the grid is exactly the co-resident CTA count.
        # ``ctas_per_sm``: 3 CTAs per SM (168 registers at 128 threads, the
        # launch bound; ncu launch__occupancy_limit_registers), the occupancy
        # query's answer on B200 and GB300.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and reduces in a fixed order; nothing is zeroed by the host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_a93b636beb1d573942bf": {
        "arches": [
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # Cooperative (a grid-wide barrier precedes the fixed-order dweight
        # reduction): the grid is exactly the co-resident CTA count.
        # ``ctas_per_sm``: 2 CTAs per SM, the program's ``min_blocks`` 2 launch
        # bound (234 registers at 128 threads; ncu
        # launch__occupancy_limit_registers), the occupancy query's answer on
        # R200; the ``min_blocks`` 3 build of the other two architectures
        # measured 10 % slower at C = 128 on its 212 SMs.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_a93b636beb1d573942bf",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_a93b636beb1d573942bf",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_a93b636beb1d573942bf",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
        },
        "role": "bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows (row ``b // n_ch_tiles`` belongs to CTA ``b``) the kernel fully
        # writes and reduces in a fixed order; nothing is zeroed by the host.
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    # Rendered by the exporter from the family adapter's scaffold: the backward
    # with stage_tokens 8, copy tma2d, min_blocks 2 (2 CTAs per SM); it runs when
    # C >= select_rule.channels[arch] and no routed record of a higher threshold
    # admits C; below every threshold the fallback record (no select_rule) runs.
    "cake_causal_conv1d_train_90351ecca30604957251": {
        "arches": [
            "sm_100a",
            "sm_103a",
            "sm_107a",
        ],
        "arg_plan": [
            ["buffer", "x"],
            ["buffer", "w"],
            ["buffer", "cu"],
            ["buffer", "dy"],
            ["buffer", "dx"],
            ["buffer", "dw"],
            ["buffer", "partial"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["parameter", "search_steps"],
            ["parameter", "ct_vec"],
            ["parameter", "n_sub"],
            ["parameter", "n_ch_tiles"],
            ["buffer", "x_map"],
            ["buffer", "dy_map"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_90351ecca30604957251",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_90351ecca30604957251",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_90351ecca30604957251",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "role": "bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 768,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
}

ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "bwd": [
            "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
            "cake_causal_conv1d_train_90351ecca30604957251",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
    "sm_103a": {
        "bwd": [
            "cake_causal_conv1d_train_21cc8e5b2b49282a47ef",
            "cake_causal_conv1d_train_90351ecca30604957251",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
    "sm_107a": {
        "bwd": [
            "cake_causal_conv1d_train_a93b636beb1d573942bf",
            "cake_causal_conv1d_train_90351ecca30604957251",
        ],
        "fwd": [
            "cake_causal_conv1d_train_f193c5247980992f4f81",
        ],
    },
}

# One verified build per architecture: B200 (``sm_100a``), B300 (``sm_103a``)
# and R200 (``sm_107a``).  Every architecture is compiled with its exact
# ``-gencode`` flag set (``build.json`` ``arch_flags``); there is no family
# fallback.
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
ARCH_BY_CAPABILITY: dict[tuple[int, int], str] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
    (10, 7): "sm_107a",
}
KINDS = ("fwd", "bwd")
# The selection-rule vocabulary of a backward record (``record["select_rule"]
# ["kind"]``); of the records whose rule admits a call the one of the highest
# threshold runs, a backward record without a rule is its route's fallback and
# a forward record never carries one.
SELECT_KINDS = ("channels_at_least",)
# The convolution width: ``weight`` and ``dweight`` are ``[C, WIDTH]`` and tap
# ``WIDTH - 1`` multiplies the current token.
WIDTH = 4
# A forward thread and a backward thread each own one 16-byte vector of
# eight BF16 channels, so ``C`` is a multiple of eight; the ``vec`` of every
# record's geometry divides this.
CHANNEL_MULTIPLE = 8
# The kernels take ``S``, ``C`` and the plan scalars as 32-bit integers.
INT32_MAX = 2**31 - 1
# Logical argument names the public API binds; every module record maps its
# kernel parameter names onto this vocabulary (``record["logical"]``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "partial",
)
LOGICAL_SCALARS = (
    "S",
    "C",
    "num_docs",
    "search_steps",
    "ct_vec",
    "n_sub",
    "n_ch_tiles",
)


def arch_for_capability(device_capability: Sequence[int]) -> str | None:
    """Name the exported architecture for one device capability, if any."""

    values = [int(value) for value in device_capability]
    if len(values) != 2:
        return None
    return ARCH_BY_CAPABILITY.get((values[0], values[1]))


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one route on ``arch`` (every program the route may launch).

    The forward launches them all, in order; the backward launches the one
    :func:`selected_modules` picks for the call's channel count.
    """

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A backward record without a rule (no ``select_rule`` key, or ``None``) is
    its route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others; when several records own a call, the one of the highest threshold
    runs.  Forward records carry no rule.
    """

    return record.get("select_rule")


def _threshold(rule: dict[str, Any], *, arch: str) -> int | None:
    """The channel count from which a selection rule owns the calls on ``arch``, ``None`` where it never does
    (``ValueError`` for an unknown kind)."""

    kind = rule["kind"]
    if kind == "channels_at_least":
        channels = rule["channels"]
        return int(channels[arch]) if arch in channels else None
    raise ValueError(
        f"unknown Cake causal conv1d training selection rule: {kind!r} "
        f"(known: {SELECT_KINDS})"
    )


def _admits(rule: dict[str, Any], *, arch: str, C: int) -> bool:
    """Whether a selection rule owns ``C`` channels on ``arch`` (``ValueError`` for an unknown kind)."""

    threshold = _threshold(rule, arch=arch)
    return threshold is not None and int(C) >= threshold


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one public call launches on ``arch`` for ``C`` channels.

    The forward launches every program of its route (:func:`route_modules`).
    The backward route may carry several programs of the kernel and launches
    exactly one: of the records whose :func:`select_rule` admits ``C`` on
    ``arch`` the one of the highest threshold, else the route's single fallback
    record (the one without a rule).  The choice is a pure function of the
    architecture and the channel count -- ``S`` is accepted and unused, and the
    host never reads the data -- so a given ``(S, C, device)`` always runs the
    same program and ``dweight`` stays bitwise reproducible.  Two admitting
    records of one threshold, or zero or several fallbacks, are registry errors
    (``RuntimeError``); an unknown rule kind is a ``ValueError``.
    """

    names = route_modules(arch, kind)
    if kind == "fwd":
        return names
    positive: list[tuple[int, str]] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
        elif _admits(rule, arch=arch, C=C):
            positive.append((_threshold(rule, arch=arch), name))
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported backward route on {arch} declares {len(fallback)} fallback "
            "programs (records without a select_rule); expected exactly one"
        )
    if not positive:
        return (fallback[0],)
    positive.sort(reverse=True)
    if len(positive) > 1 and positive[0][0] == positive[1][0]:
        tied = [name for threshold, name in positive if threshold == positive[0][0]]
        raise RuntimeError(
            f"the exported backward route on {arch} selects {len(tied)} programs "
            f"for C={int(C)} at one threshold {positive[0][0]} ({tied}); thresholds "
            "must be distinct"
        )
    return (positive[0][1],)


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The backward module records of ``arch`` that own a ``partial`` workspace, in route order.

    Every backward program declares a ``workspace_rule``; a route without one
    is a registry error (``RuntimeError``).
    """

    owners = tuple(
        MODULES[name]
        for name in route_modules(arch, "bwd")
        if MODULES[name]["workspace_rule"] is not None
    )
    if not owners:
        raise RuntimeError(
            f"the exported backward route on {arch} declares no workspace owner; "
            "expected at least one"
        )
    return owners


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the backward ``partial`` workspace on ``arch``.

    The public workspace serves every backward program of the route
    (:func:`workspace_owners`), each of which writes the leading rows of its
    own grid.  The owners share one build geometry, so for every ``C`` their
    rows, ``ceil(ctas_per_sm x SM count / n_ch_tiles)``, and bytes grow with
    ``ctas_per_sm`` alone: the record with the largest ``ctas_per_sm`` (route
    order breaks ties) bounds them all.  Owners of different geometries are a
    registry error (``RuntimeError``).
    """

    owners = workspace_owners(arch)
    geometries = {
        tuple(sorted(owner["grid_rule"]["geometry"].items())) for owner in owners
    }
    if len(geometries) != 1:
        raise RuntimeError(
            f"the backward programs on {arch} declare {len(geometries)} build "
            "geometries; one workspace serves them only when they share one"
        )
    return max(owners, key=lambda owner: int(owner["grid_rule"]["ctas_per_sm"]))


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    return -(-int(numerator) // int(denominator))


def channel_split(C: int, *, geometry: dict[str, int]) -> tuple[int, int, int]:
    """``(ct_vec, n_sub, n_ch_tiles)`` of ``C`` channels under a record's build geometry.

    ``ct_vec = min(C / vec, threads)`` channel vectors per channel tile (one
    thread each), ``n_sub = threads // ct_vec`` token sub-tiles per CTA and
    ``n_ch_tiles = ceil(C / vec / ct_vec)`` channel tiles.
    """

    C = int(C)
    vec = int(geometry["vec"])
    threads = int(geometry["threads"])
    if C <= 0 or C % vec:
        raise ValueError(f"C={C} must be a positive multiple of {vec}")
    channel_vectors = C // vec
    ct_vec = min(channel_vectors, threads)
    return ct_vec, threads // ct_vec, _ceil_div(channel_vectors, ct_vec)


def plan(S: int, C: int, num_docs: int, *, geometry: dict[str, int]) -> dict[str, int]:
    """The kernel's launch scalars for one problem under a record's build geometry.

    ``geometry`` is ``record["grid_rule"]["geometry"]``: ``threads`` per CTA
    and ``vec`` BF16 channels per thread.  The scalars are ``S``, ``C``,
    ``num_docs``, :func:`channel_split`'s ``ct_vec`` / ``n_sub`` /
    ``n_ch_tiles`` and ``search_steps``, the number of halvings of the
    device-side binary search over ``cu_seqlens`` (``max(1, ceil(log2
    num_docs))``).  Every value is a pure function of ``(S, C, num_docs)`` and
    the geometry: the host never reads ``cu_seqlens``, and the kernel derives
    its token ranges from ``S`` and the grid.
    """

    S, C, num_docs = int(S), int(C), int(num_docs)
    if S < 0 or num_docs < 0:
        raise ValueError(f"S and num_docs must be non-negative, got {(S, num_docs)}")
    ct_vec, n_sub, n_ch_tiles = channel_split(C, geometry=geometry)
    return {
        "S": S,
        "C": C,
        "num_docs": num_docs,
        "search_steps": max(1, (num_docs - 1).bit_length()),
        "ct_vec": ct_vec,
        "n_sub": n_sub,
        "n_ch_tiles": n_ch_tiles,
    }


@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of one CUDA device (immutable per process)."""

    import torch

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def launch_grid(record: dict[str, Any], *, device_index: int) -> tuple[int, int, int]:
    """Launch grid of one module on ``device_index`` under its recorded grid rule.

    ``co_resident`` modules launch exactly ``ctas_per_sm x SM count`` CTAs: the
    co-resident CTA count of the program, which the cooperative backward never
    exceeds.  The grid depends on the device only, so the deterministic
    ``dweight`` order is a pure function of ``(S, C, device)`` and one
    workspace serves every token count.
    """

    rule = record["grid_rule"]
    if rule["kind"] != "co_resident":
        raise ValueError(
            f"unknown Cake causal conv1d training grid rule: {rule['kind']!r}"
        )
    return (max(1, int(rule["ctas_per_sm"]) * sm_count(device_index)), 1, 1)


def partial_rows(grid_x: int, n_ch_tiles: int) -> int:
    """Rows of the backward ``partial`` workspace for a grid: ``ceil(grid_x / n_ch_tiles)``.

    CTA ``b`` of the grid owns channel tile ``b % n_ch_tiles`` and partial row
    ``b // n_ch_tiles``; the kernel writes every row it reduces.
    """

    return _ceil_div(grid_x, n_ch_tiles)


def partial_bytes(record: dict[str, Any], C: int, rows: int) -> int:
    """Bytes of ``rows`` FP32 ``[C, taps]`` partial rows of the module's workspace rule."""

    rule = record["workspace_rule"]
    if rule is None or rule["kind"] != "partial_rows":
        raise ValueError(f"module {record['kernel_symbol']} owns no backward workspace")
    return 4 * int(rule["taps"]) * int(C) * int(rows)


def workspace_layout(
    record: dict[str, Any], C: int, *, device_index: int
) -> dict[str, int]:
    """Byte layout of the caller-owned backward workspace of one module for ``C`` channels.

    The workspace is one CUDA ``uint8`` buffer holding the FP32 ``[rows, C, taps]``
    ``dweight`` partials, ``rows = partial_rows(grid, n_ch_tiles)`` for the
    device's launch grid; it depends on ``C`` and the device only, so one
    buffer serves every token count.  The kernel fully writes the rows it
    reduces, so nothing is zeroed between launches or CUDA Graph replays.
    """

    grid_x = launch_grid(record, device_index=device_index)[0]
    _, _, n_ch_tiles = channel_split(C, geometry=record["grid_rule"]["geometry"])
    rows = partial_rows(grid_x, n_ch_tiles)
    return {
        "grid": grid_x,
        "rows": rows,
        "taps": int(record["workspace_rule"]["taps"]),
        "total_bytes": partial_bytes(record, C, rows),
    }


@functools.cache
def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    import torch

    capability = tuple(torch.cuda.get_device_capability(device_index))
    arch = arch_for_capability(capability)
    if arch is None:
        raise RuntimeError(
            f"device {device_index} (capability {capability}) is not an exported Cake causal conv1d training target"
        )
    return arch


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library without a prebuilt file (for
    ahead-of-time warm-up through FlashInfer's ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(name: str, values: dict[str, Any], *, device_index: int) -> None:
    """Launch one exported module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is the
    module's :func:`launch_grid` on ``device_index``, and the kernel runs on
    the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    grid = launch_grid(record, device_index=device_index)
    grid_values = {"grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    args = []
    for kind, key in record["arg_plan"]:
        if kind == "grid":
            args.append(grid_values[key])
            continue
        logical_name = logical[key]
        if logical_name not in values:
            raise ValueError(
                f"module {name} needs {logical_name!r} (kernel parameter {key!r})"
            )
        args.append(values[logical_name])
    module = load(name, device_arch(device_index))
    with tvm_ffi.use_torch_stream():
        getattr(module, record["ffi_entry"])(*args)


__all__ = [
    "ARCHES",
    "ARCH_BY_CAPABILITY",
    "CHANNEL_MULTIPLE",
    "INT32_MAX",
    "KINDS",
    "LOGICAL_SCALARS",
    "LOGICAL_TENSORS",
    "MODULES",
    "ROUTES",
    "SELECT_KINDS",
    "WIDTH",
    "arch_for_capability",
    "channel_split",
    "device_arch",
    "launch_grid",
    "library_name",
    "load",
    "partial_bytes",
    "partial_rows",
    "plan",
    "route_applies",
    "route_modules",
    "run_module",
    "select_rule",
    "selected_modules",
    "sm_count",
    "specs_for_arch",
    "workspace_layout",
    "workspace_owners",
    "workspace_record",
]
