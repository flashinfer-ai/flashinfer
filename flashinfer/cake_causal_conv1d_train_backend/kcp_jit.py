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

Host plan and loader glue of the Cake context-parallel (KCP) packed width-4
depthwise causal convolution + SiLU training kernels (SM100 / SM103 / SM107):
the halo-aware programs the KCP host flow (:mod:`.kcp`) launches, recorded
beside the zero-history registry :mod:`.cake_jit`, whose tables this module
never touches.

Every module below is one mechanically exported production build: a CUDA
device translation unit plus a tvm-ffi binding compiled together for exactly
one architecture.  ``MODULES`` and ``ROUTES`` carry the host plan only, in the
vocabulary of :mod:`.cake_jit`: the kernel's ``library`` stem, the
architectures it is delivered for, its launch record, grid rule, workspace
rule, selection rule and the kernel-parameter -> logical-argument map.  What to
compile -- sources, digests, nvcc flags -- is recorded once in ``build.json``
next to this file (its entries sit beside the zero-history entries);
:mod:`.cake_build` serves the library ``<library>_<arch>`` from a prebuilt
``aot/`` file when a builder compiled it and otherwise through FlashInfer's JIT
from the same sources and flags, verifying every source digest first.

The routes of an architecture are the three KCP kinds: ``kcp_fwd`` (one
program: ``y`` and the fragments' history rows ``hist`` from ``x``, ``weight``,
``cu_seqlens`` and the context's history tables), ``kcp_bwd`` (the fallback
program and the big-channel ``_st8`` program, selected by the channel count
exactly as the zero-history backward route selects its programs: a record's
``select_rule`` ``channels_at_least`` owns ``C >= channels[arch]``, the record
without a rule is the fallback; ``dx``, the deterministic FP32 ``dweight`` and
the predecessor fragments' tail derivatives ``dpre_tail`` in one cooperative
launch) and ``kcp_bwd_head`` (one program: the head-row derivatives
``dpre_head`` every consumer fragment sends back; its grid is
``max(1, n_frag x ceil(C / (threads x vec)))``, computed by :mod:`.kcp` from
the record's ``grid_rule.geometry`` and launched through the record's
``arg_plan`` / ``logical`` / ``ffi_entry`` and :func:`load`).

The host plan reuses :mod:`.cake_jit`'s functions over a record (:func:`plan`,
:func:`launch_grid`, :func:`partial_rows`, :func:`partial_bytes`,
:func:`workspace_layout`, :func:`channel_split`) unchanged; the functions bound
to the tables (:func:`route_modules`, :func:`selected_modules`,
:func:`workspace_record`, :func:`library_name`, :func:`load`,
:func:`run_module`) are spelled here over ``MODULES`` and ``ROUTES``.  The
logical tensor names are :mod:`.cake_jit`'s plus the KCP tables ``hist``,
``hist_row_off``, ``dpre_tail``, ``tail_row_off``, ``dpre_head`` and
``head_row_off``; the two ``CUtensorMap`` parameters ``x_map`` / ``dy_map`` of
the TMA backward programs map onto ``x`` / ``dy`` (the binding encodes the
descriptors from the tensors at every launch); the scalars are
:mod:`.cake_jit`'s.

The program name ``cake_causal_conv1d_train_<20 hex>`` is the first twenty hex
digits of the SHA-256 of the compact JSON ``{"build": <build keywords, sorted>,
"module": <dotted kernel module>}`` of the kernel the program was rendered
from: the forward and the head program with their kernel files' default
keywords; the backward with ``vec=8, copy="tma2d"`` on sm_100a and sm_103a,
with ``vec=8, min_blocks=2`` on sm_107a, and the big-channel backward with
``vec=8, copy="tma2d", stage_tokens=8, min_blocks=2`` -- the builds of the
zero-history backward, which the KCP kernels extend.
"""

from __future__ import annotations

import functools
from typing import Any, Sequence

import tvm_ffi

from . import cake_build
from . import cake_jit as _zero_history
from .cake_jit import (  # noqa: F401  (re-exported: the host plan over a record is the zero-history one)
    ARCH_BY_CAPABILITY,
    ARCHES,
    CHANNEL_MULTIPLE,
    INT32_MAX,
    LOGICAL_SCALARS,
    SELECT_KINDS,
    WIDTH,
    arch_for_capability,
    channel_split,
    launch_grid,
    partial_bytes,
    partial_rows,
    plan,
    sm_count,
    workspace_layout,
)

# The derived leaves of every record (``kernel_symbol``, ``launch.block``,
# ``launch.cluster``, ``launch.cooperative``, ``launch.dynamic_smem_bytes``,
# ``launch.use_pdl``, ``arg_plan`` and ``logical``) are rewritten by the
# exporter from the rendered program; every other field is host plan.
MODULES: dict[str, dict[str, Any]] = {
    "cake_causal_conv1d_train_675940a7a4fc0dba4b53": {
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
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
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
        # The grid is the co-resident CTA count, ``ctas_per_sm x SM count``,
        # as for the zero-history forward: CTA ``b`` owns channel tile
        # ``b % n_ch_tiles`` and an equal contiguous share of the token axis.
        # ``ctas_per_sm`` is the measured co-residency of the program (the
        # occupancy query's answer, 4 CTAs per SM at 112 registers on GB300;
        # sm_100a / sm_107a are taken as the same launch bound); ``geometry``
        # is the kernel's build geometry the host plan derives its scalars
        # from.  Besides ``y`` the forward writes ``hist`` (BF16 history rows
        # ``[n_hist_rows, C]``) at the rows ``hist_row_off`` (int32
        # ``[n_frag, 3]``, ``-1`` = none) names.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 4,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 33792,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "search_steps": "search_steps",
            "w": "w",
            "x": "x",
            "y": "y",
        },
        "role": "kcp_fwd",
        "workspace_rule": None,
    },
    "cake_causal_conv1d_train_5288d132a5cc927ae729": {
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
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
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
        # reduction): the grid is exactly the co-resident CTA count, 3 CTAs
        # per SM (168 registers at 128 threads, the launch bound), the
        # occupancy query's answer on B200 and GB300 for this build.  The
        # fallback program of the route (no ``select_rule``).  Besides ``dx``
        # and ``dw`` it writes the predecessors' tail derivatives
        # ``dpre_tail`` (FP32 ``[n_tail_rows, C]``) at the rows
        # ``tail_row_off`` (int32 ``[n_frag, 3]``, ``-1`` = none) names and
        # reads the forward's ``hist`` rows through ``hist_row_off``.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 3,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_5288d132a5cc927ae729",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_5288d132a5cc927ae729",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_5288d132a5cc927ae729",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "role": "kcp_bwd",
        # The FP32 ``partial`` workspace: ``[ceil(grid / n_ch_tiles), C, taps]``
        # rows of per-CTA dweight partials, reduced in fixed order after the
        # grid barrier (:func:`partial_rows`, :func:`partial_bytes`).
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039": {
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
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
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
        # The sm_107a fallback backward (``vec=8, min_blocks=2``: two CTAs per
        # SM, bulk copies, no tensor maps), cooperative like every backward
        # program; the grid is its co-resident CTA count.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 50176,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "x": "x",
        },
        "role": "kcp_bwd",
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_7485529a1755e5bd3036": {
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
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_tail"],
            ["buffer", "tail_row_off"],
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
        # The big-channel backward (``stage_tokens=8``, eight-row TMA boxes per
        # stage, two CTAs per SM): it runs when ``C >= select_rule.channels
        # [arch]``, else the architecture's fallback record runs -- the
        # zero-history backward's thresholds, which these programs' geometry
        # measured.  Cooperative; the grid is its co-resident CTA count.
        "grid_rule": {
            "cooperative": True,
            "ctas_per_sm": 2,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_7485529a1755e5bd3036",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_7485529a1755e5bd3036",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": True,
            "dynamic_smem_bytes": 82944,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_7485529a1755e5bd3036",
        "logical": {
            "C": "C",
            "S": "S",
            "ct_vec": "ct_vec",
            "cu": "cu",
            "dpre_tail": "dpre_tail",
            "dw": "dw",
            "dx": "dx",
            "dy": "dy",
            "dy_map": "dy",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "n_ch_tiles": "n_ch_tiles",
            "n_sub": "n_sub",
            "num_docs": "num_docs",
            "partial": "partial",
            "search_steps": "search_steps",
            "tail_row_off": "tail_row_off",
            "w": "w",
            "x": "x",
            "x_map": "x",
        },
        "role": "kcp_bwd",
        "select_rule": {
            "channels": {
                "sm_100a": 512,
                "sm_103a": 512,
                "sm_107a": 1024,
            },
            "kind": "channels_at_least",
        },
        "workspace_rule": {
            "kind": "partial_rows",
            "taps": 4,
        },
    },
    "cake_causal_conv1d_train_17cd42fb05079bbbf840": {
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
            ["buffer", "hist"],
            ["buffer", "hist_row_off"],
            ["buffer", "dpre_head"],
            ["buffer", "head_row_off"],
            ["parameter", "S"],
            ["parameter", "C"],
            ["parameter", "num_docs"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "ffi_entry": "run",
        # The head-row derivative program: ``dpre_head`` (FP32 ``[n_head_rows,
        # C]``) at the rows ``head_row_off`` (int32 ``[n_frag, 3, 3]``, ``-1``
        # = none) names, from the fragment's first three tokens, the history
        # rows and ``dy``.  Its grid is ``max(1, n_frag x ceil(C / (threads x
        # vec)))`` -- a work-item count, computed by the host flow from
        # ``geometry`` (the grid strides over the items) -- not the co-resident
        # count; ``ctas_per_sm`` records the measured co-residency (5 CTAs per
        # SM at 96 registers and no shared memory, the occupancy query's
        # answer on GB300) for a device-bound launch.
        "grid_rule": {
            "cooperative": False,
            "ctas_per_sm": 5,
            "geometry": {
                "threads": 128,
                "vec": 8,
            },
            "kind": "co_resident",
        },
        "kernel": "cake_causal_conv1d_train_17cd42fb05079bbbf840",
        "kernel_symbol": "kernel_cake_causal_conv1d_train_17cd42fb05079bbbf840",
        "launch": {
            "block": (128, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        },
        "library": "cake_causal_conv1d_train_17cd42fb05079bbbf840",
        "logical": {
            "C": "C",
            "S": "S",
            "cu": "cu",
            "dpre_head": "dpre_head",
            "dy": "dy",
            "head_row_off": "head_row_off",
            "hist": "hist",
            "hist_row_off": "hist_row_off",
            "num_docs": "num_docs",
            "w": "w",
            "x": "x",
        },
        "role": "kcp_bwd_head",
        "workspace_rule": None,
    },
}

# architecture -> KCP kind -> the records the route may launch, in launch
# order.  The forward and the head route carry one program; the backward
# route carries the architecture's fallback program first and the
# big-channel ``_st8`` program second (:func:`selected_modules` launches one).
ROUTES: dict[str, dict[str, list[str]]] = {
    "sm_100a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_5288d132a5cc927ae729",
            "cake_causal_conv1d_train_7485529a1755e5bd3036",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        ],
    },
    "sm_103a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_5288d132a5cc927ae729",
            "cake_causal_conv1d_train_7485529a1755e5bd3036",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        ],
    },
    "sm_107a": {
        "kcp_bwd": [
            "cake_causal_conv1d_train_6ea6d5fc9e7fa3f2d039",
            "cake_causal_conv1d_train_7485529a1755e5bd3036",
        ],
        "kcp_bwd_head": [
            "cake_causal_conv1d_train_17cd42fb05079bbbf840",
        ],
        "kcp_fwd": [
            "cake_causal_conv1d_train_675940a7a4fc0dba4b53",
        ],
    },
}

# The KCP kinds (the zero-history kinds are ``cake_jit.KINDS``).
KINDS = ("kcp_fwd", "kcp_bwd", "kcp_bwd_head")
# The history / halo rows every fragment carries (the convolution width minus one).
HALO = WIDTH - 1
# Logical argument names the KCP host flow binds: the zero-history vocabulary
# (``cake_jit.LOGICAL_TENSORS``, spelled out so the registry stays a literal)
# plus the KCP tables.  ``hist``: BF16 history rows ``[n_hist_rows, C]``;
# ``hist_row_off``: int32 ``[n_frag, 3]`` (``-1`` = none); ``dpre_tail``: FP32
# ``[n_tail_rows, C]`` tail derivatives with ``tail_row_off`` int32
# ``[n_frag, 3]``; ``dpre_head``: FP32 ``[n_head_rows, C]`` head derivatives
# with ``head_row_off`` int32 ``[n_frag, 3, 3]``.  The scalars are the
# zero-history scalars (``cake_jit.LOGICAL_SCALARS``).
LOGICAL_TENSORS = (
    "x",
    "w",
    "cu",
    "y",
    "dy",
    "dx",
    "dw",
    "partial",
    "hist",
    "hist_row_off",
    "dpre_tail",
    "tail_row_off",
    "dpre_head",
    "head_row_off",
)


def route_applies(*, device_capability: Sequence[int], kind: str) -> bool:
    """Whether the verified export owns one KCP training call on a device."""

    arch = arch_for_capability(device_capability)
    if arch is None:
        return False
    return kind in ROUTES.get(arch, {})


def route_modules(arch: str, kind: str) -> tuple[str, ...]:
    """Ordered module names of one KCP route on ``arch`` (every program the route may launch).

    The forward and the head route launch their one program; the backward
    launches the one :func:`selected_modules` picks for the call's channel
    count.
    """

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    names = ROUTES.get(arch, {}).get(kind)
    if not names:
        raise ValueError(
            f"unsupported Cake causal conv1d KCP training route {kind!r} on {arch!r}; "
            f"exported architectures: {ARCHES}"
        )
    return tuple(names)


def select_rule(record: dict[str, Any]) -> dict[str, Any] | None:
    """The selection rule of one module record, ``None`` when the record carries none.

    A backward record without a rule (no ``select_rule`` key, or ``None``) is
    its route's fallback program.  ``{"kind": "channels_at_least", "channels":
    {arch: C_min, ...}}`` owns the calls whose channel count is at least
    ``channels[arch]`` on the architectures it names and never hits on the
    others.  Forward and head records carry no rule.
    """

    return record.get("select_rule")


def selected_modules(
    arch: str, kind: str, *, C: int, S: int | None = None
) -> tuple[str, ...]:
    """Ordered module names one KCP call launches on ``arch`` for ``C`` channels.

    The forward and the head route launch every program of their route
    (:func:`route_modules`, one each).  The backward route carries the
    architecture's fallback program and the big-channel program and launches
    exactly one: the record whose :func:`select_rule` admits ``C`` on ``arch``
    when one does, else the route's single fallback record (the one without a
    rule) -- the rule of the zero-history backward.  The choice is a pure
    function of the architecture and the channel count (``S`` is accepted and
    unused, and the host never reads the data), so a given ``(S, C, device)``
    always runs the same program and ``dweight`` stays bitwise reproducible.
    Several admitting records, or zero or several fallbacks, are registry
    errors (``RuntimeError``); an unknown rule kind is a ``ValueError``.
    """

    names = route_modules(arch, kind)
    if kind != "kcp_bwd":
        return names
    positive: list[str] = []
    fallback: list[str] = []
    for name in names:
        rule = select_rule(MODULES[name])
        if rule is None:
            fallback.append(name)
        elif _zero_history._admits(rule, arch=arch, C=C):
            positive.append(name)
    if len(fallback) != 1:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} declares {len(fallback)} fallback "
            "programs (records without a select_rule); expected exactly one"
        )
    if len(positive) > 1:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} selects {len(positive)} programs "
            f"for C={int(C)} ({positive}); expected at most one"
        )
    if positive:
        return (positive[0],)
    return (fallback[0],)


def workspace_owners(arch: str) -> tuple[dict[str, Any], ...]:
    """The KCP backward module records of ``arch`` that own a ``partial`` workspace, in route order."""

    owners = tuple(
        MODULES[name]
        for name in route_modules(arch, "kcp_bwd")
        if MODULES[name]["workspace_rule"] is not None
    )
    if not owners:
        raise RuntimeError(
            f"the exported KCP backward route on {arch} declares no workspace owner; "
            "expected at least one"
        )
    return owners


def workspace_record(arch: str) -> dict[str, Any]:
    """The capacity owner of the KCP backward ``partial`` workspace on ``arch``.

    As for the zero-history route: the owners share one build geometry, so
    their rows and bytes grow with ``ctas_per_sm`` alone and the record with
    the largest ``ctas_per_sm`` (route order breaks ties) bounds them all.
    """

    owners = workspace_owners(arch)
    geometries = {
        tuple(sorted(owner["grid_rule"]["geometry"].items())) for owner in owners
    }
    if len(geometries) != 1:
        raise RuntimeError(
            f"the KCP backward programs on {arch} declare {len(geometries)} build "
            "geometries; one workspace serves them only when they share one"
        )
    return max(owners, key=lambda owner: int(owner["grid_rule"]["ctas_per_sm"]))


def head_grid(record: dict[str, Any], *, n_frag: int, C: int) -> tuple[int, int, int]:
    """The head program's grid: ``max(1, n_frag x ceil(C / (threads x vec)))`` work items over the record's geometry."""

    geometry = record["grid_rule"]["geometry"]
    channels_per_cta = int(geometry["threads"]) * int(geometry["vec"])
    if channels_per_cta <= 0:
        raise ValueError(
            "the head record's geometry must cover at least one channel per CTA"
        )
    items = int(n_frag) * (-(-int(C) // channels_per_cta))
    return (max(1, items), 1, 1)


def device_arch(device_index: int) -> str:
    """Exported architecture of one CUDA device (``RuntimeError`` when it is not an exported target)."""

    return _zero_history.device_arch(device_index)


def library_name(name: str, arch: str) -> str:
    """``build.json`` library of KCP module record ``name`` built for ``arch``: ``<library>_<arch>``."""

    record = MODULES[name]
    if arch not in record["arches"]:
        raise RuntimeError(
            f"Cake causal conv1d KCP training module {name} is not delivered for {arch}; "
            f"its architectures: {record['arches']}"
        )
    return f"{record['library']}_{arch}"


@functools.cache
def load(name: str, arch: str) -> Any:
    """The loaded tvm-ffi module of KCP record ``name`` built for ``arch``.

    The prebuilt ``aot/`` library when a builder compiled it, else FlashInfer's
    JIT build of the same sources and flags (``cake_build.load_library``).
    """

    return cake_build.load_library(library_name(name, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specifications of every ``arch`` library of the package's ``build.json`` without a prebuilt file
    (the zero-history and the KCP entries alike; for ahead-of-time warm-up through FlashInfer's
    ``build_jit_specs``)."""

    return cake_build.jit_specs(arch)


def run_module(
    name: str,
    values: dict[str, Any],
    *,
    device_index: int,
    grid: tuple[int, int, int] | None = None,
) -> None:
    """Launch one exported KCP module with logical argument ``values``.

    ``values`` maps the logical names in ``LOGICAL_TENSORS`` /
    ``LOGICAL_SCALARS`` to tensors and Python scalars; the module record maps
    its kernel parameter names onto that vocabulary.  The launch grid is the
    module's :func:`launch_grid` on ``device_index`` (the co-resident count)
    unless ``grid`` gives it (the head program's work-item grid,
    :func:`head_grid`), and the kernel runs on the current torch stream.
    """

    record = MODULES[name]
    logical = record["logical"]
    if grid is None:
        grid = launch_grid(record, device_index=device_index)
    grid_values = {
        "grid_x": int(grid[0]),
        "grid_y": int(grid[1]),
        "grid_z": int(grid[2]),
    }
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
    "HALO",
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
    "head_grid",
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
