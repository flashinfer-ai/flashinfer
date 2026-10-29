# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
"""CPU tests of the Cake single-GPU norm package (``flashinfer.cake_muse_spark_norm_backend``): the registry
records and routes of its three operators as the exporter wrote them, the selection and launch rules over the
literals, the variant / route / support helpers of the host module, and ``build.json`` under FlashInfer's manifest
checker."""

import hashlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.cake_muse_spark_norm_backend as pkg
from flashinfer.cake_muse_spark_norm_backend import cake_build
from flashinfer.cake_muse_spark_norm_backend import cake_jit as loader

ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(pkg.__file__).parent
OP = pkg.GATED_RESIDUAL_NEXT_RMSNORM
OP_A = "gated_residual_rmsnorm"
OP_C = "rmsnorm_sigmoid_gate"
BLACKWELL = ("sm_100a", "sm_103a")
EVERY = (*BLACKWELL, "sm_107a")
# The kernel parameters of each operator's programs, in binding order (the grid triple follows).
PARAMETERS = {
    OP: (
        "branch",
        "residual",
        "alpha",
        "beta",
        "next_weight",
        "post_weight",
        "residual_out",
        "ffn_input",
        "M",
        "post_eps",
        "next_eps",
        "post_gain_center",
        "next_gain_center",
        "alpha_value",
    ),
    OP_A: (
        "branch",
        "residual",
        "weight",
        "alpha",
        "beta",
        "out",
        "M",
        "eps",
        "gain_center",
    ),
    OP_C: ("x", "gate", "tokens", "heads", "x_stride", "gate_stride", "eps"),
}
# The logical names of op C's parameters (ops A and B map every parameter to itself but ``M`` -> ``rows``).
LOGICAL_C = {
    "x": "input",
    "gate": "gate",
    "tokens": "tokens",
    "heads": "heads",
    "x_stride": "input_stride",
    "gate_stride": "gate_stride",
    "eps": "eps",
}
# The variants with exported programs: three of op B's four forms (the fourth runs through the post_weight
# programs), both modes of op A in the plain and the PDL build, op C's BF16 form in the plain and the PDL build (one
# PDL table per architecture, sm_107a's over the R200 graph-replay tables).
SHIPPED_VARIANTS = {
    OP: ("alpha_tensor_pw", "alpha_scalar_nopw", "alpha_scalar_pw"),
    OP_A: ("mode0", "mode0_pdl", "mode1", "mode1_pdl"),
    OP_C: ("bf16", "bf16_pdl"),
}
# Op C's windows of record per head count: (heads, above, at_most) -> the selecting build keywords.
OP_C_WINDOWS = {
    (128, 0, 8): {
        "threads": 256,
        "lanes": 32,
        "cluster": 16,
        "tcluster": 1,
        "order": "tokens",
    },
    (128, 8, 256): {
        "threads": 256,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 16,
        "order": "tokens",
    },
    (128, 256, None): {
        "threads": 256,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 1,
        "order": "heads",
    },
    (32, 0, 1024): {
        "threads": 256,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 16,
        "order": "tokens",
    },
    (32, 1024, None): {
        "threads": 256,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 1,
        "order": "tokens",
    },
    (16, 0, 1024): {
        "threads": 128,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 16,
        "order": "tokens",
    },
    (16, 1024, None): {
        "threads": 128,
        "lanes": 8,
        "cluster": 1,
        "tcluster": 1,
        "order": "tokens",
    },
}
# Op C's windows of record on sm_107a: thirteen programs over fifteen windows; the token counts of each head count are
# partitioned but for the declared gaps, where the FlashInfer kernel measured ahead: 128 heads at 1 and 3-8 tokens, 32
# heads at 10-64 and 257-767.  The four R200 bandwidth-band programs (the packed four-row forms at 256 and 64 threads,
# the two-row 128-thread token cluster and the 64-thread 256-bit form) own the rows above the run-1 routes: 128 heads
# 384-511 and from 2561 the 256-thread packed form, 512-2560 the 64-thread one; 32 heads 768-2047 the two-row token
# cluster, from 2048 the 64-thread packed form; 16 heads 2048-8191 the 256-bit form and from 8192 the multi-token
# 256-bit form (a 128-thread CTA holding four 16-head tokens, two rows per thread).


def _form(threads: int, lanes: int, **knobs) -> dict:
    """The selecting build keywords of an op C program: its block, its lanes per row and the knobs off their
    defaults (one CTA, no token cluster, the token-major order, one row per thread)."""
    return {
        "threads": threads,
        "lanes": lanes,
        "cluster": 1,
        "tcluster": 1,
        "order": "tokens",
        "rows_per_thread": 1,
        **knobs,
    }


OP_C_WINDOWS_SM107A = {
    (128, 1, 2): _form(256, 32, cluster=16),
    (128, 8, 64): _form(256, 8, tcluster=16, order="heads"),
    (128, 64, 256): _form(256, 8, rows_per_thread=2),
    (128, 256, 383): _form(256, 8, order="heads"),
    (128, 383, 511): _form(256, 8, rows_per_thread=4, packed=True),
    (128, 511, 2560): _form(64, 8, rows_per_thread=4, packed=True),
    (128, 2560, None): _form(256, 8, rows_per_thread=4, packed=True),
    (32, 0, 9): _form(256, 8, tcluster=16),
    (32, 64, 256): _form(128, 8, tcluster=8),
    (32, 767, 2047): _form(128, 8, tcluster=16, rows_per_thread=2),
    (32, 2047, None): _form(64, 8, rows_per_thread=4, packed=True),
    (16, 0, 1024): _form(128, 8, tcluster=16),
    (16, 1024, 2047): _form(128, 8),
    (16, 2047, 8191): _form(64, 4, width=256),
    (16, 8191, None): _form(128, 4, width=256, rows_per_thread=2, tpc=4),
}
# The declared gaps of the sm_107a routes, (above, at_most) ranges of rows (tokens for op C) no program admits, keyed
# (route, head count or None) like the partition test's window classes; every other route partitions its counts. Op C's
# PDL route leaves gaps of its own there (one token at every head count, 128 heads at 25-39 and 41-47, 32 heads at 14,
# 25-63, 66-69 and 71-257: the
# CuTe DSL kernel under PDL ahead of every Cake form; below them the 128-thread tail-free form's PDL build serves 128
# heads at 9-24 and 32 heads at 10-12), spelled under GAPS_PDL as well.
GAPS_SM107A = {
    (f"{OP_A}_mode0", None): ((128, 256),),
    (f"{OP_A}_mode0_pdl", None): ((128, 256),),
    (f"{OP_C}_bf16", 128): ((0, 1), (2, 8)),
    (f"{OP_C}_bf16", 32): ((9, 64), (256, 767)),
    (f"{OP_C}_bf16_pdl", 128): ((0, 1), (24, 39), (40, 47)),
    (f"{OP_C}_bf16_pdl", 32): ((0, 1), (13, 14), (24, 63), (65, 69), (70, 257)),
    (f"{OP_C}_bf16_pdl", 16): ((0, 1),),
}
# The declared gaps of the PDL routes, keyed (arch, route, head count or None): the row counts where the CuTe DSL kernel
# under PDL measured ahead of both the PDL build and the plain build of the routed program, so the facade's auto keeps
# it there.  Op A's mode-0 routes declare gaps (one on sm_100a, two on sm_103a); op C's bf16_pdl tables declare the token counts the CuTe DSL
# kernel under PDL wins in the graph regime (one token at every head count, 128 heads around 64 tokens and 32 heads
# around 128 and 256 tokens on the Blackwell architectures; one token at every head count, 128 heads at 25-39 and 41-47,
# 32 heads at 14, 25-63, 66-69 and 71-257 on sm_107a); the windows and gaps partition the token count for every head
# count and keep the plain route's boundaries a program touches (a gap may span one).
GAPS_PDL = {
    ("sm_100a", f"{OP_A}_mode0_pdl", None): ((768, 4095),),
    ("sm_103a", f"{OP_A}_mode0_pdl", None): ((1024, 2048), (4096, 8191)),
    ("sm_100a", f"{OP_C}_bf16_pdl", 128): ((0, 1), (16, 64)),
    ("sm_100a", f"{OP_C}_bf16_pdl", 32): ((0, 1), (64, 128), (192, 257)),
    ("sm_100a", f"{OP_C}_bf16_pdl", 16): ((0, 1),),
    ("sm_103a", f"{OP_C}_bf16_pdl", 128): ((0, 1), (32, 64)),
    ("sm_103a", f"{OP_C}_bf16_pdl", 32): ((0, 1), (64, 128), (192, 256)),
    ("sm_103a", f"{OP_C}_bf16_pdl", 16): ((0, 1),),
    ("sm_107a", f"{OP_C}_bf16_pdl", 128): ((0, 1), (24, 39), (40, 47)),
    ("sm_107a", f"{OP_C}_bf16_pdl", 32): (
        (0, 1),
        (13, 14),
        (24, 63),
        (65, 69),
        (70, 257),
    ),
    ("sm_107a", f"{OP_C}_bf16_pdl", 16): ((0, 1),),
}
# Op A's Blackwell PDL routes, (at_most, program) windows per (arch, mode), the last unbounded: a program named by its
# place in the plain route (c2 / c4 / rsa the three small-M composites, pro the prologue whole-row program, psb the
# parked whole-row program) stands for its PDL build; (program, "plain") for the plain build the PDL route borrows
# (launched without the attribute: the PDL build measured slower than the CuTe DSL kernel under PDL there and the
# plain build faster; no op A window does so since the re-measurement, which gave every such row to the prologue
# program's PDL build); None for a declared gap.
OP_A_PLAIN_ORDER = ("c2", "c4", "rsa", "pro", "psb")
# The parked program's late-store form (the parked program's build with these keywords changed): the sixth record
# of sm_100a's mode-1 plain route, owning 512-1024 and 2049-4096 rows, and the fourth of sm_103a's, owning 129-2048.
OP_A_LATE_STORE = {"psb_late": {"planes_path": "sts_late"}}
# Op A's Blackwell plain routes per (arch, mode): (place, windows) rows in route order, a window (above, at_most).
# The whole-row programs follow the three small-M composites: on sm_100a the prologue program owns 129-383 and the
# rows above the parked band, the parked program 384-2048 (mode 0: the prologue program from 2049; mode 1: the parked
# program at 384-511 and 1025-2048 around the late-store form's 512-1024, the late-store form again at 2049-4096 and
# the prologue program from 4097); on sm_103a,
# re-measured on GB300, mode 0 parks 129-224 and 257-2048 around the prologue program's 225-256 and mode 1 hands
# 129-2048 to the late-store form, the prologue program keeping 2049- (the parked program's own build left the route).
OP_A_PLAIN_WINDOWS = {
    ("sm_100a", 0): (
        ("c2", [(0, 16)]),
        ("c4", [(16, 64)]),
        ("rsa", [(64, 128)]),
        ("pro", [(128, 383), (2048, None)]),
        ("psb", [(383, 2048)]),
    ),
    ("sm_100a", 1): (
        ("c2", [(0, 16)]),
        ("c4", [(16, 64)]),
        ("rsa", [(64, 128)]),
        ("pro", [(128, 383), (4096, None)]),
        ("psb", [(383, 511), (1024, 2048)]),
        ("psb_late", [(511, 1024), (2048, 4096)]),
    ),
    ("sm_103a", 0): (
        ("c2", [(0, 16)]),
        ("c4", [(16, 64)]),
        ("rsa", [(64, 128)]),
        ("psb", [(128, 224), (256, 2048)]),
        ("pro", [(224, 256), (2048, None)]),
    ),
    ("sm_103a", 1): (
        ("c2", [(0, 16)]),
        ("c4", [(16, 64)]),
        ("rsa", [(64, 128)]),
        ("psb_late", [(128, 2048)]),
        ("pro", [(2048, None)]),
    ),
}
# The sm_100a PDL routes' forms of the prologue whole-row program that ship under PDL only (no plain-route record):
# the build of the prologue program's PDL build with these keywords changed.
OP_A_PDL_ONLY = {
    "abel": {"ab_hint": "el"},
    "pk": {"packed": True},
    "h": {"hoist": True},
}
OP_A_PDL_WINDOWS = {
    ("sm_100a", 0): (
        (16, "c2"),
        (64, "c4"),
        (128, "rsa"),
        (192, "psb"),
        (256, "abel"),
        (768, "pro"),
        (4095, None),
        (8192, "pk"),
        (None, "pro"),
    ),
    ("sm_100a", 1): (
        (16, "c2"),
        (64, "c4"),
        (128, "rsa"),
        (256, "abel"),
        (512, "pro"),
        (768, "pk"),
        (1024, "pro"),
        (8192, "pk"),
        (None, "pro"),
    ),
    ("sm_103a", 0): (
        (16, "c2"),
        (64, "c4"),
        (128, "rsa"),
        (255, "abel"),
        (256, "psb"),
        (384, "pk"),
        (512, "pro"),
        (768, "pk"),
        (1024, "pro"),
        (2048, None),
        (4096, "pro"),
        (8191, None),
        (None, "pro"),
    ),
    ("sm_103a", 1): (
        (16, "c2"),
        (64, "c4"),
        (128, "rsa"),
        (256, "abel"),
        (512, "pro"),
        (768, "pk"),
        (1024, "pro"),
        (2048, "pk"),
        (None, "pro"),
    ),
}
# Op C's PDL route per architecture: (heads, above, at_most) -> (the selecting keywords of the program, as OP_C_WINDOWS
# or OP_C_WINDOWS_SM107A spells them, its form): "pdl" the PDL build, "pdl_late" the late-signal PDL build of the two
# token-cluster programs, "plain" the plain route's build borrowed (16 heads above 1024 tokens on the Blackwell
# architectures; on sm_103a also 128 heads at 257-2048 tokens and 32 heads at 1025-8192 tokens, where the PDL build is
# 1-6 % behind the plain build and no launch overlaps its predecessor; on sm_107a the 64-thread packed form at 128 heads
# 513-2560 and 32 heads above 2048, the 128-thread tail-free form at 16 heads 1025-2047, the 256-bit form at 16 heads
# 2048-6143 and the multi-token 256-bit form at 16 heads 12288-32767, where its plain build is 1.5-4.9 % ahead of its
# PDL build).  Contiguous over the token count per head count but for the declared gaps; every boundary of the plain
# route a program touches is one of these (the pdl table splits a plain window between the pdl and the pdl_late build,
# or between the borrowed plain build and the pdl build; a gap may span one).
_C16, _CT16, _HM, _T256, _T128_CT16, _T128 = (
    OP_C_WINDOWS[key]
    for key in (
        (128, 0, 8),
        (128, 8, 256),
        (128, 256, None),
        (32, 1024, None),
        (16, 0, 1024),
        (16, 1024, None),
    )
)
_CT16_HM, _R2, _CT8, _P1, _P2, _P3, _P4, _P5 = (
    OP_C_WINDOWS_SM107A[key]
    for key in (
        (128, 8, 64),
        (128, 64, 256),
        (32, 64, 256),
        (128, 383, 511),
        (128, 511, 2560),
        (32, 767, 2047),
        (16, 2047, 8191),
        (16, 8191, None),
    )
)
OP_C_PDL_WINDOWS = {
    "sm_100a": {
        (128, 1, 8): (_C16, "pdl"),
        (128, 8, 16): (_CT16, "pdl_late"),
        (128, 64, 256): (_CT16, "pdl_late"),
        (128, 256, None): (_HM, "pdl"),
        (32, 1, 64): (_CT16, "pdl_late"),
        (32, 128, 192): (_CT16, "pdl_late"),
        (32, 257, 1024): (_CT16, "pdl_late"),
        (32, 1024, None): (_T256, "pdl"),
        (16, 1, 8): (_T128_CT16, "pdl"),
        (16, 8, 256): (_T128_CT16, "pdl_late"),
        (16, 256, 1024): (_T128_CT16, "pdl"),
        (16, 1024, None): (_T128, "plain"),
    },
    "sm_103a": {
        (128, 1, 8): (_C16, "pdl"),
        (128, 8, 32): (_CT16, "pdl"),
        (128, 64, 256): (_CT16, "pdl_late"),
        (128, 256, 2048): (_HM, "plain"),
        (128, 2048, None): (_HM, "pdl"),
        (32, 1, 8): (_CT16, "pdl"),
        (32, 8, 64): (_CT16, "pdl_late"),
        (32, 128, 192): (_CT16, "pdl_late"),
        (32, 256, 1024): (_CT16, "pdl_late"),
        (32, 1024, 8192): (_T256, "plain"),
        (32, 8192, None): (_T256, "pdl"),
        (16, 1, 8): (_T128_CT16, "pdl"),
        (16, 8, 64): (_T128_CT16, "pdl_late"),
        (16, 64, 1024): (_T128_CT16, "pdl"),
        (16, 1024, None): (_T128, "plain"),
    },
    "sm_107a": {
        (128, 1, 2): (_C16, "pdl"),
        (128, 2, 8): (_C16, "pdl"),
        (128, 8, 24): (_T128, "pdl"),
        (128, 39, 40): (_T128, "pdl"),
        (128, 47, 64): (_T128, "pdl"),
        (128, 64, 256): (_R2, "pdl"),
        (128, 256, 383): (_HM, "pdl"),
        (128, 383, 511): (_P1, "pdl"),
        (128, 511, 512): (_P2, "pdl"),
        (128, 512, 2560): (_P2, "plain"),
        (128, 2560, None): (_P1, "pdl"),
        (32, 1, 9): (_CT16, "pdl"),
        (32, 9, 13): (_T128, "pdl"),
        (32, 14, 24): (_T128, "pdl"),
        (32, 63, 64): (_T128, "pdl"),
        (32, 64, 65): (_T128, "pdl"),
        (32, 69, 70): (_T128, "pdl"),
        (32, 257, 767): (_T256, "pdl"),
        (32, 767, 2047): (_P3, "pdl"),
        (32, 2047, 2048): (_P2, "pdl"),
        (32, 2048, None): (_P2, "plain"),
        (16, 1, 1024): (_T128_CT16, "pdl"),
        (16, 1024, 2047): (_T128, "plain"),
        (16, 2047, 6143): (_P4, "plain"),
        (16, 6143, 8191): (_P5, "pdl"),
        (16, 8191, 12287): (_P5, "pdl"),
        (16, 12287, 32767): (_P5, "plain"),
        (16, 32767, None): (_P5, "pdl"),
    },
}
# Op A's windows of record on sm_107a, one tuple of windows per record: the two composites alternate below 65 rows
# (mode 1 keeps the whole-row program from 33), the whole-row persistent program owns 257-4096 rows in both modes,
# the TMA-ring persistent program (residual ring in mode 0, residual-and-weight ring filled first in mode 1) the
# rows beyond; mode 1 parks 129-256 (mode 0 leaves them to the FlashInfer kernel).
OP_A_WINDOWS_SM107A = {
    0: {
        ((0, 1), (16, 32)),
        ((1, 16), (32, 64)),
        ((64, 128),),
        ((256, 4096),),
        ((4096, None),),
    },
    1: {
        ((0, 1), (16, 32)),
        ((1, 16),),
        ((32, 128),),
        ((128, 256),),
        ((256, 4096),),
        ((4096, None),),
    },
}
FAMILY_FLAGS = ("--prec-div=true", "--prec-sqrt=true", "--ftz=false")
NAME = re.compile(r"cake_muse_spark_norm_[0-9a-f]{20}")


def _variant(op: str, build: dict) -> str:
    if op == OP:
        return pkg.gated_residual_next_rmsnorm_variant(
            build["tensor_alpha"], build["post_weight"]
        )
    if op == OP_C:
        return "bf16" + ("_pdl" if build["pdl"] else "")
    return f"mode{build['mode']}" + ("_pdl" if build["pdl"] else "")


def _route_windows(record: dict, arch: str, route: str) -> list:
    """The windows ``route`` gives ``record`` on ``arch``: the rule's ``routes`` entry of a PDL route borrowing the
    record's plain build, else the record's own windows (the loader's ``admits`` reading)."""
    rule = record["select_rule"]
    return rule.get("routes", {}).get(route, rule["windows"]).get(arch, [])


def _owners(arch: str, route: str) -> set:
    """The records a route lists on an architecture: its own and the plain builds it borrows there."""
    return {
        name
        for name, record in loader.MODULES.items()
        if arch in record["arches"]
        and (
            record["route"] == route
            or arch in record["select_rule"].get("routes", {}).get(route, {})
        )
    }


def _routes_of(arch: str) -> set:
    """Every shipped variant's route (the same set on every architecture)."""
    del arch
    return {
        loader.route_key(op, variant)
        for op in loader.OPS
        for variant in SHIPPED_VARIANTS[op]
    }


def _op_a_programs(arch: str, mode: int) -> dict:
    """The plain route's records of a mode on ``arch`` by their place (OP_A_PLAIN_WINDOWS): the route lists them
    in window order, so a place is told by its build -- the three small-M composites by position, the whole-row
    programs by beta's plane (``planes_smem`` none: the prologue program; beta parked in shared memory: the parked
    program, its late-store form under the ``planes_path`` of OP_A_LATE_STORE)."""
    names = loader.route_modules(arch, loader.route_key(OP_A, f"mode{mode}"))
    places = [place for place, _windows in OP_A_PLAIN_WINDOWS[(arch, mode)]]
    assert len(names) == len(places), (arch, mode, names)
    programs = dict(zip(places[:3], names[:3], strict=True))
    for name in names[3:]:
        build = loader.MODULES[name]["build"]
        if build["planes_smem"] == "none":
            place = "pro"
        else:
            assert build["planes_smem"] == "beta", (arch, mode, name)
            late = OP_A_LATE_STORE["psb_late"]["planes_path"]
            place = "psb_late" if build["planes_path"] == late else "psb"
        assert place not in programs, (arch, mode, place)
        programs[place] = name
    assert list(programs) == places, (arch, mode, names)
    if "psb_late" in programs:
        # the parked program's build with the keywords changed; sm_103a's mode-1 route lacks the parked program
        # itself, so its record comes from the sm_100a route
        parked = programs.get("psb") or _op_a_programs("sm_100a", mode)["psb"]
        assert loader.MODULES[programs["psb_late"]]["build"] == {
            **loader.MODULES[parked]["build"],
            **OP_A_LATE_STORE["psb_late"],
        }, (arch, mode)
    return programs


def _op_a_pdl_only(programs: dict, program: str) -> str:
    """The record of a PDL-only form of op A's prologue program (OP_A_PDL_ONLY): the prologue program's PDL build
    with the form's keywords, found among every record (it is in no plain route)."""
    base = loader.MODULES[_pdl_build_of(programs["pro"])]["build"]
    build = {**base, **OP_A_PDL_ONLY[program]}
    (twin,) = [
        other
        for other, record in loader.MODULES.items()
        if record["op"] == OP_A and record["build"] == build
    ]
    return twin


def _pdl_build_of(name: str) -> str:
    """The record of the PDL build of ``name``'s program: the same build keywords with ``pdl`` set (the two mode-1
    composites' PDL builds carry a ``packed_widen`` of their own beside it)."""
    plain = loader.MODULES[name]
    ignored = ("pdl", "packed_widen")
    key = {k: v for k, v in plain["build"].items() if k not in ignored}
    (twin,) = [
        other
        for other, record in loader.MODULES.items()
        if record["op"] == plain["op"]
        and record["build"]["pdl"] is True
        and {k: v for k, v in record["build"].items() if k not in ignored} == key
    ]
    return twin


def _cluster_dims(build: dict) -> tuple:
    """The launch cluster of a build: the cluster over one row's CTAs, or op C's token-axis cluster
    (``(1, tcluster, 1)`` under the head-major order)."""
    if build["cluster"] > 1 or "tcluster" not in build:
        return (build["cluster"], 1, 1)
    if build["tcluster"] <= 1:
        return (1, 1, 1)
    return (
        (1, build["tcluster"], 1)
        if build["order"] == "heads"
        else (build["tcluster"], 1, 1)
    )


def test_variant_and_route_helpers():
    assert pkg.gated_residual_next_rmsnorm_variant(True, True) == "alpha_tensor_pw"
    assert pkg.gated_residual_next_rmsnorm_variant(False, False) == "alpha_scalar_nopw"
    assert pkg.gated_residual_next_rmsnorm_variant(False, True) == "alpha_scalar_pw"
    assert pkg.gated_residual_next_rmsnorm_variant(True, False) == "alpha_tensor_nopw"
    for tensor_alpha, has_post_weight in ((True, True), (False, False), (False, True)):
        route = pkg.gated_residual_next_rmsnorm_route(tensor_alpha, has_post_weight)
        assert (
            route
            == f"{OP}_{pkg.gated_residual_next_rmsnorm_variant(tensor_alpha, has_post_weight)}"
        )
    assert pkg.gated_residual_next_rmsnorm_route(True, False) == f"{OP}_alpha_tensor_pw"
    assert loader.route_key(OP, "alpha_tensor_nopw") == f"{OP}_alpha_tensor_nopw"
    with pytest.raises(ValueError, match="unknown Cake norm operator"):
        loader.route_key("rmsnorm", "alpha_tensor_pw")
    with pytest.raises(ValueError, match="unknown variant"):
        loader.route_key(OP, "alpha_vector")
    assert loader.arch_for_capability((10, 0)) == "sm_100a"
    assert loader.arch_for_capability([10, 3]) == "sm_103a"
    assert loader.arch_for_capability((10, 7)) == "sm_107a"
    assert loader.arch_for_capability((9, 0)) is None
    assert loader.arch_for_capability((10,)) is None
    assert tuple(loader.ARCHES) == ("sm_100a", "sm_103a", "sm_107a")
    assert set(loader.ARCH_BY_CAPABILITY.values()) == set(loader.ARCHES)
    assert tuple(loader.OPS) == (OP, OP_A, OP_C)
    assert loader.route_key(OP_A, "mode1_pdl") == f"{OP_A}_mode1_pdl"
    assert loader.route_key(OP_C, "bf16") == f"{OP_C}_bf16"
    assert loader.SELECT_KINDS == (loader.SELECT_KIND, "tokens_heads_between")
    assert loader.GRID_KINDS == ("rows", "persistent_rows", "tokens_heads")
    with pytest.raises(ValueError, match="unknown variant"):
        loader.route_key(OP_C, "fp8")
    with pytest.raises(ValueError, match="unknown variant"):
        loader.route_key(OP_A, "mode2")
    for op in loader.OPS:
        assert set(SHIPPED_VARIANTS[op]) <= set(loader.VARIANTS[op])


def test_support_follows_the_device_architecture(monkeypatch):
    device = torch.device("cuda", 0)
    for capability, arch in (((10, 0), "sm_100a"), ((10, 3), "sm_103a")):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *_, c=capability: c
        )
        assert pkg._arch(device) == arch
        for tensor_alpha in (True, False):
            for has_post_weight in (True, False):
                assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
                    device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight
                )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    assert pkg._arch(device) == "sm_107a"
    for tensor_alpha in (True, False):
        for has_post_weight in (True, False):
            assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
                device, tensor_alpha=tensor_alpha, has_post_weight=has_post_weight
            )
    scalar_pw = dict(tensor_alpha=False, has_post_weight=True)
    for rows in (1, 16, 128, 129, 255, 256, 4096, 10**6):
        assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
            device, rows=rows, **scalar_pw
        )
    assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
        device, rows=1, tensor_alpha=True, has_post_weight=True
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert pkg._arch(device) is None
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(device)
    assert pkg._arch(torch.device("cpu")) is None
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(torch.device("cpu"))


def test_support_follows_the_row_count_through_the_route_windows(monkeypatch):
    """With ``rows`` the predicate asks the route's windows: a row count no record admits (a gap of the route, where
    the FlashInfer kernel measured ahead) is unsupported, so the facade's ``auto`` keeps the CuTe branch there."""
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    route = loader.route_key(OP, "alpha_tensor_pw")
    names = loader.route_modules("sm_100a", route)
    for rows in (1, 16, 17, 512, 100000):
        assert pkg.is_cake_gated_residual_next_rmsnorm_supported(device, rows=rows)
        assert loader.selected_module("sm_100a", route, rows) in names
    gapped = {
        "arches": ["sm_100a"],
        "route": route,
        "select_rule": {
            "kind": "rows_between",
            "windows": {"sm_100a": [[0, 128], [256, None]]},
        },
    }
    monkeypatch.setitem(loader.MODULES, "gapped_record", gapped)
    monkeypatch.setitem(loader.ROUTES["sm_100a"], route, ["gapped_record"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    assert pkg.is_cake_gated_residual_next_rmsnorm_supported(device)
    assert pkg.is_cake_gated_residual_next_rmsnorm_supported(device, rows=128)
    assert pkg.is_cake_gated_residual_next_rmsnorm_supported(device, rows=257)
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(device, rows=129)
    assert not pkg.is_cake_gated_residual_next_rmsnorm_supported(device, rows=256)
    with pytest.raises(NotImplementedError, match="admits 200 rows"):
        loader.selected_module("sm_100a", route, 200)


def test_registry_records_reference_the_build_manifest():
    """Every ``MODULES`` record names its ``build.json`` libraries and nothing to compile: one entry per
    (library, architecture) with the kernel, binding and occupancy sources under ``csrc/``, their digests and the
    family's numerics flags; the registry itself records the host plan only."""
    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    assert set(manifest["arch_flags"]) >= set(BLACKWELL)
    assert manifest["common"]["requires"]["cuda_toolkit"]["min"]
    libraries = cake_build.libraries_by_name()
    assert (PACKAGE_DIR / "csrc" / "cake_ffi_checks.h").is_file()
    referenced = set()
    logical_values = set()
    for name, record in loader.MODULES.items():
        assert NAME.fullmatch(name), name
        assert record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert record["ffi_entry"] == "run"
        assert record["occupancy_entry"] == "max_active_clusters"
        op = record["op"]
        assert op in loader.OPS, name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        arches = tuple(record["arches"])
        assert arches and set(arches) <= set(EVERY), name
        build = record["build"]
        if op != OP_C:
            assert build["H"] == pkg.HIDDEN_SIZE
        variant = _variant(op, build)
        assert variant in SHIPPED_VARIANTS[op], name
        assert record["route"] == loader.route_key(op, variant)
        launch = record["launch"]
        assert tuple(launch["block"]) == (build["threads"], 1, 1)
        assert tuple(launch["cluster"]) == _cluster_dims(build)
        assert launch["cooperative"] is False
        assert launch["use_pdl"] is build["pdl"]
        if op == OP:
            assert build["pdl"] is False
        else:
            assert build["pdl"] is variant.endswith("_pdl")
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        )
        rule = record["grid_rule"]
        assert rule["cluster"] == build["cluster"]
        if op == OP_C:
            assert rule["kind"] == "tokens_heads"
            assert rule["rows_per_cta"] == (
                build["threads"] // build["lanes"] * build["rows_per_thread"]
            )
            assert rule["tcluster"] == build["tcluster"]
            assert rule["order"] == build["order"]
            assert rule["interleave"] == build["interleave"]
            assert build["tail"] in ("none", "clamp")
        else:
            assert rule["kind"] == (
                "persistent_rows" if build["persistent"] else "rows"
            )
        assert rule["kind"] in loader.GRID_KINDS
        select = record["select_rule"]
        assert select["kind"] == (
            loader.SELECT_KINDS[1] if op == OP_C else loader.SELECT_KIND
        )
        assert set(select["windows"]) == set(arches)
        for borrowing, per_arch in select.get("routes", {}).items():
            # a plain build a PDL route borrows: the PDL route's windows of this record, launched without the attribute
            assert borrowing == loader.route_key(op, f"{variant}_pdl"), name
            assert set(per_arch) <= set(arches) and launch["use_pdl"] is False, name
            assert all(per_arch.values()), name
        parameters = [key for kind, key in record["arg_plan"] if kind != "grid"]
        assert tuple(parameters) == PARAMETERS[op]
        assert [key for kind, key in record["arg_plan"] if kind == "grid"] == [
            "grid_x",
            "grid_y",
            "grid_z",
        ]
        assert record["logical"] == (
            LOGICAL_C
            if op == OP_C
            else {key: ("rows" if key == "M" else key) for key in PARAMETERS[op]}
        )
        logical_values |= set(record["logical"].values())
        for arch in arches:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            sources = {
                f"csrc/{name}_kernel.cu",
                f"csrc/{name}_binding.cu",
                f"csrc/{name}_occupancy.cu",
            }
            assert set(entry["sources"]) == sources, library
            assert set(entry["sha256"]) == sources
            for source, digest in entry["sha256"].items():
                assert (
                    hashlib.sha256((PACKAGE_DIR / source).read_bytes()).hexdigest()
                    == digest
                ), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            flags = tuple(entry["nvcc_flags"])
            assert flags[: len(FAMILY_FLAGS)] == FAMILY_FLAGS, library
            # nvcc's default --fmad=true is left off the entries; nothing opts into fast math.
            assert flags == FAMILY_FLAGS, library
            assert entry["entry"] == "run" and entry["arch"] == arch
            assert entry["min_cuda"]
    assert referenced == set(libraries)
    assert logical_values == set(loader.LOGICAL_TENSORS) | set(loader.LOGICAL_SCALARS)
    assert {record["op"] for record in loader.MODULES.values()} == set(loader.OPS)


def test_routes_partition_the_row_counts():
    """Every route of an architecture lists its records in window order (the plain builds a PDL route borrows among
    them); read through the route called, the windows of each row-count class partition the positive counts but for
    the declared gaps (sm_107a's, and the Blackwell PDL routes')."""
    for arch, routes in loader.ROUTES.items():
        assert arch in EVERY
        assert set(routes) == _routes_of(arch)
        for route, names in routes.items():
            assert len(names) == len(set(names))
            assert tuple(names) == loader.route_modules(arch, route)
            assert set(names) == _owners(arch, route), (arch, route)
            keyed = (
                loader.MODULES[names[0]]["select_rule"]["kind"]
                == "tokens_heads_between"
            )
            classes: dict = {}
            for name in names:
                windows = _route_windows(loader.MODULES[name], arch, route)
                assert windows, (arch, route, name)
                for window in windows:
                    *head, above, at_most = window
                    assert len(head) == (1 if keyed else 0)
                    classes.setdefault(head[0] if head else None, []).append(
                        (int(above), at_most, name)
                    )
            assert set(classes) == ({128, 32, 16} if keyed else {None})
            for heads, windows in classes.items():
                windows.sort()
                bounds = [(0, windows[0][0])] + [
                    (w[1], v[0]) for w, v in zip(windows[:-1], windows[1:], strict=True)
                ]
                assert all(at_most <= above for at_most, above in bounds), (
                    arch,
                    route,
                    heads,
                )
                gaps = tuple(
                    (at_most, above) for at_most, above in bounds if at_most != above
                )
                expected = (
                    GAPS_SM107A.get((route, heads), ())
                    if arch == "sm_107a"
                    else GAPS_PDL.get((arch, route, heads), ())
                )
                assert gaps == expected, (arch, route, heads)
                assert windows[-1][1] is None and all(
                    w[1] is not None for w in windows[:-1]
                )
                top = windows[-1][0] + 2
                for rows in range(1, top + 1):
                    owners = [
                        name
                        for above, at_most, name in windows
                        if above < rows <= (at_most or rows)
                    ]
                    if any(above < rows <= at_most for above, at_most in gaps):
                        assert not owners
                        with pytest.raises(
                            NotImplementedError, match=f"admits {rows} rows"
                        ):
                            loader.selected_module(arch, route, rows, heads)
                    else:
                        (owner,) = owners
                        assert loader.selected_module(arch, route, rows, heads) == owner
                assert (
                    loader.selected_module(arch, route, 10**6, heads) == windows[-1][2]
                )
            if keyed:
                with pytest.raises(ValueError, match="needs the head count"):
                    loader.selected_module(arch, route, 1)
                with pytest.raises(NotImplementedError, match="of 64 heads"):
                    loader.selected_module(arch, route, 1, 64)
    assert loader.route_available("sm_107a", loader.route_key(OP, "alpha_tensor_pw"))
    assert (
        len(loader.route_modules("sm_107a", loader.route_key(OP, "alpha_tensor_pw")))
        == 2
    )
    scalar_pw = loader.route_key(OP, "alpha_scalar_pw")
    small_rows, whole_range = loader.route_modules("sm_107a", scalar_pw)
    for rows in (1, 128):
        assert loader.selected_module("sm_107a", scalar_pw, rows) == small_rows
    for rows in (129, 255, 256, 10**6):
        assert loader.selected_module("sm_107a", scalar_pw, rows) == whole_range
    unexported = loader.route_key(OP, "alpha_tensor_nopw")
    assert not loader.route_available("sm_100a", unexported)
    assert loader.route_modules("sm_100a", unexported) == ()
    with pytest.raises(NotImplementedError, match="no exported program"):
        loader.selected_module("sm_100a", unexported, 1)
    with pytest.raises(ValueError, match="positive row count"):
        loader.selected_module("sm_100a", loader.route_key(OP, "alpha_tensor_pw"), 0)
    assert loader.admits(None, 1, "sm_100a")
    rule = {"kind": "rows_between", "windows": {"sm_100a": [[0, 16], [128, 512]]}}
    assert [
        r for r in (1, 16, 17, 128, 129, 512, 513) if loader.admits(rule, r, "sm_100a")
    ] == [1, 16, 129, 512]
    assert not loader.admits(rule, 1, "sm_103a")
    with pytest.raises(ValueError, match="unknown Cake norm select rule"):
        loader.admits({"kind": "always"}, 1, "sm_100a")
    keyed = {
        "kind": "tokens_heads_between",
        "windows": {"sm_100a": [[128, 0, 8], [32, 1024, None]]},
    }
    assert [
        r for r in (1, 8, 9, 1024, 1025) if loader.admits(keyed, r, "sm_100a", 128)
    ] == [1, 8]
    assert [
        r for r in (1, 8, 9, 1024, 1025) if loader.admits(keyed, r, "sm_100a", 32)
    ] == [1025]
    assert not loader.admits(keyed, 1, "sm_100a", 16)
    assert not loader.admits(keyed, 1, "sm_103a", 128)
    with pytest.raises(ValueError, match="needs the head count"):
        loader.admits(keyed, 1, "sm_100a")
    # a rule's routes entry: the windows another route reads in place of the record's own
    borrowed = {
        "kind": "rows_between",
        "windows": {"sm_100a": [[0, 16]]},
        "routes": {"other": {"sm_100a": [[512, 768]]}},
    }
    assert loader.admits(borrowed, 16, "sm_100a") and not loader.admits(
        borrowed, 600, "sm_100a"
    )
    assert loader.admits(borrowed, 600, "sm_100a", route="other")
    assert not loader.admits(borrowed, 16, "sm_100a", route="other")
    assert loader.admits(
        borrowed, 16, "sm_100a", route="own"
    )  # a route the rule does not name reads the windows
    assert not loader.admits(borrowed, 600, "sm_103a", route="other")


def test_rmsnorm_sigmoid_gate_routes_window_the_token_count_per_head_count():
    """Op C's one route per architecture holds six programs over seven windows: three head counts (128 / 32 / 16),
    each partitioning the token counts, the 16-CTA token-cluster program shared by the 128-head mid band and the
    32-head band; the launch cluster follows the build (a cluster over the head blocks, a token-axis cluster, or
    none) and the grid is the kernel's two-dimensional one."""
    for arch in BLACKWELL:
        route = loader.route_key(OP_C, "bf16")
        names = loader.route_modules(arch, route)
        assert len(names) == 6, (arch, route)
        windows = {}
        for name in names:
            record = loader.MODULES[name]
            for heads, above, at_most in record["select_rule"]["windows"][arch]:
                windows[(heads, above, at_most)] = name
        assert set(windows) == set(OP_C_WINDOWS)
        for key, selecting in OP_C_WINDOWS.items():
            record = loader.MODULES[windows[key]]
            build = record["build"]
            assert {k: build[k] for k in selecting} == selecting, key
            assert build["tail"] == "none" and build["pdl"] is False
            assert tuple(record["launch"]["cluster"]) == _cluster_dims(build), key
            assert key[0] % record["grid_rule"]["rows_per_cta"] == 0, key
        for heads, tokens, key in (
            (128, 1, (128, 0, 8)),
            (128, 8, (128, 0, 8)),
            (128, 9, (128, 8, 256)),
            (128, 256, (128, 8, 256)),
            (128, 257, (128, 256, None)),
            (128, 16384, (128, 256, None)),
            (32, 1, (32, 0, 1024)),
            (32, 1024, (32, 0, 1024)),
            (32, 1025, (32, 1024, None)),
            (16, 1024, (16, 0, 1024)),
            (16, 1025, (16, 1024, None)),
        ):
            assert loader.selected_module(arch, route, tokens, heads) == windows[key]
        assert windows[(128, 8, 256)] == windows[(32, 0, 1024)]
        assert windows[(32, 1024, None)] != windows[(128, 256, None)]
    cluster = loader.MODULES[windows[(128, 0, 8)]]
    assert loader.launch_grid(cluster, 3, device_index=0, values={"heads": 128}) == (
        16,
        3,
        1,
    )
    token_cluster = loader.MODULES[windows[(128, 8, 256)]]
    assert loader.launch_grid(
        token_cluster, 9, device_index=0, values={"heads": 128}
    ) == (16, 4, 1)
    assert loader.launch_grid(
        token_cluster, 1024, device_index=0, values={"heads": 32}
    ) == (1024, 1, 1)
    head_major = loader.MODULES[windows[(128, 256, None)]]
    assert loader.launch_grid(
        head_major, 1024, device_index=0, values={"heads": 128}
    ) == (4, 1024, 1)
    plain = loader.MODULES[windows[(16, 1024, None)]]
    assert loader.launch_grid(plain, 4096, device_index=0, values={"heads": 16}) == (
        4096,
        1,
        1,
    )
    with pytest.raises(ValueError, match="must carry 'heads'"):
        loader.launch_grid(plain, 4096, device_index=0)


def test_rmsnorm_sigmoid_gate_pdl_route_names_a_build_form_per_window():
    """Op C's PDL route has windows of its own per architecture: each names the PDL build of a routed program, its
    late-signal PDL build or the plain build the route borrows (16 heads above 1024 tokens everywhere, the whole-row
    programs up to 2048 / 8192 tokens on sm_103a, the 64-thread packed form at 128 heads 513-2560 and 32 heads above
    2048, the 256-bit form at 16 heads 2048-6143 and the multi-token form at 16 heads 12288-32767 on sm_107a: the plain
    record's plan, launched without the attribute); the declared gaps (``GAPS_PDL``: one token at every head count; 128
    heads around 64 tokens and 32 heads around 128 and 256 tokens on the Blackwell architectures, 128 heads at 25-39 and
    41-47 and 32 heads at 14, 25-63, 66-69 and 71-257 on sm_107a) belong to no record; the 128-thread tail-free form's
    PDL build serves 128 heads at 9-24, 40 and 48-64 and 32 heads at 10-13, 15-24, 64-65 and 70 on sm_107a (the
    small-token windows the fill-in sessions widened).  The records appear in the route in window order, each once."""
    plain_route = loader.route_key(OP_C, "bf16")
    route = loader.route_key(OP_C, "bf16_pdl")
    for arch in EVERY:
        plain_names = loader.route_modules(arch, plain_route)
        plain_by_key = {
            tuple(window): name
            for name in plain_names
            for window in loader.MODULES[name]["select_rule"]["windows"][arch]
        }
        names = loader.route_modules(arch, route)
        expected = OP_C_PDL_WINDOWS[arch]
        windows = {}
        for name in names:
            for window in _route_windows(loader.MODULES[name], arch, route):
                windows[tuple(window)] = name
        assert set(windows) == set(expected), arch
        order: list = []
        for key in sorted(expected, key=lambda k: (-k[0], k[1])):
            selecting, form = expected[key]
            name = windows[key]
            if name not in order:
                order.append(name)
            record = loader.MODULES[name]
            build = record["build"]
            assert {k: build[k] for k in selecting} == selecting, (arch, key)
            # the two-rows-per-thread form clamps its tail; every other routed build is tail-free
            assert build["tail"] == ("clamp" if selecting is _R2 else "none"), (
                arch,
                key,
            )
            assert build["pdl"] is (form != "plain") and record["launch"][
                "use_pdl"
            ] is (form != "plain"), (arch, key)
            assert build["pdl_late"] is (form == "pdl_late"), (arch, key)
            assert build["pdl_launch_top"] is False
            assert tuple(record["launch"]["cluster"]) == _cluster_dims(build), key
            heads, above, at_most = key
            for tokens in (above + 1, at_most or above + 4096):
                assert loader.selected_module(arch, route, tokens, heads) == name, (
                    arch,
                    key,
                    tokens,
                )
            if form == "plain":
                # the plain route's record of the same program: its window holds the borrowed one
                containing = next(
                    w
                    for w in plain_by_key
                    if w[0] == heads
                    and w[1] <= above
                    and (w[2] is None or (at_most is not None and at_most <= w[2]))
                )
                assert name == plain_by_key[containing] and name in plain_names, (
                    arch,
                    key,
                )
                assert record["route"] == plain_route
                assert list(key) in record["select_rule"]["routes"][route][arch], (
                    arch,
                    key,
                )
            else:
                assert (
                    record["route"] == route and "routes" not in record["select_rule"]
                )
                assert arch in record["arches"] and set(record["arches"]) <= set(EVERY)
                assert name not in plain_names
        assert list(names) == order, arch
        # the pdl builds of five programs (four on sm_100a: the 16-CTA token-cluster program's pdl build keeps no
        # window there), two pdl_late builds and the borrowed plain builds: the 16-head tail-free program on the
        # Blackwell architectures, plus the two whole-row programs on sm_103a; on sm_107a the pdl builds of eleven
        # programs and four borrowed plain builds (the two token clusters' pdl builds left the route, the 128-thread
        # tail-free form's PDL build joined it at 128 heads 9-24 and 32 heads 10-12; the multi-token form lends
        # its plain build at 16 heads 12288-32767)
        assert len(names) == {"sm_100a": 7, "sm_103a": 10, "sm_107a": 15}[arch], arch
        for heads, gaps in (
            (128, GAPS_PDL.get((arch, route, 128), ())),
            (32, GAPS_PDL.get((arch, route, 32), ())),
            (16, GAPS_PDL.get((arch, route, 16), ())),
        ):
            for low, high in gaps:
                for tokens in (low + 1, high):
                    with pytest.raises(
                        NotImplementedError,
                        match=f"admits {tokens} rows of {heads} heads",
                    ):
                        loader.selected_module(arch, route, tokens, heads)
    # the plain build a PDL window borrows launches from the plain record on every architecture
    for arch in EVERY:
        assert route in loader.ROUTES[arch] and loader.route_available(arch, route)
    borrowed = loader.selected_module("sm_107a", route, 1024, 128)
    assert borrowed == loader.selected_module("sm_107a", plain_route, 1024, 128)
    assert loader.MODULES[borrowed]["launch"]["use_pdl"] is False


def test_rmsnorm_sigmoid_gate_routes_on_sm_107a():
    """On sm_107a op C's route holds thirteen programs over fifteen windows: 128 heads at 2 tokens, 9-64 (the head-major
    16-CTA token cluster), 65-256 (two rows per thread), 257-383 (head-major), 384-511 and from 2561 (the 256-thread
    packed four-row form) and 512-2560 (its 64-thread form); 32 heads at 1-9 (the 16-CTA token cluster), 65-256 (the
    128-thread 8-CTA token cluster), 768-2047 (the two-row 128-thread token cluster) and from 2048 (the 64-thread packed
    form); 16 heads as on the Blackwell architectures up to 2047, the 64-thread 256-bit form at 2048-8191 and the
    multi-token 256-bit form (a 128-thread CTA holding four 16-head tokens) from 8192.  128 heads at 1 and 3-8 tokens
    and 32 heads at 10-64 and 257-767 are declared gaps: no program admits them."""
    route = loader.route_key(OP_C, "bf16")
    names = loader.route_modules("sm_107a", route)
    assert len(names) == 13
    windows = {}
    for name in names:
        for heads, above, at_most in loader.MODULES[name]["select_rule"]["windows"][
            "sm_107a"
        ]:
            windows[(heads, above, at_most)] = name
    assert set(windows) == set(OP_C_WINDOWS_SM107A)
    for key, selecting in OP_C_WINDOWS_SM107A.items():
        record = loader.MODULES[windows[key]]
        build = record["build"]
        assert {k: build[k] for k in selecting} == selecting, key
        assert build["pdl"] is False
        assert tuple(record["launch"]["cluster"]) == _cluster_dims(build), key
        rule = record["grid_rule"]
        if rule.get("tpc", 1) > 1:
            # a multi-token CTA holds tpc whole tokens of rows_per_cta / tpc heads: the one head count it serves
            assert key[0] * rule["tpc"] == rule["rows_per_cta"], key
        else:
            assert key[0] % rule["rows_per_cta"] == 0, key
    # the 64-thread packed form serves 128 heads at 512-2560 and 32 heads from 2048; the 256-thread one two windows
    assert windows[(128, 511, 2560)] == windows[(32, 2047, None)]
    assert windows[(128, 383, 511)] == windows[(128, 2560, None)]
    assert windows[(128, 8, 64)] != windows[(128, 256, 383)]
    for heads, tokens, key in (
        (128, 2, (128, 1, 2)),
        (128, 9, (128, 8, 64)),
        (128, 64, (128, 8, 64)),
        (128, 65, (128, 64, 256)),
        (128, 256, (128, 64, 256)),
        (128, 257, (128, 256, 383)),
        (128, 383, (128, 256, 383)),
        (128, 384, (128, 383, 511)),
        (128, 511, (128, 383, 511)),
        (128, 512, (128, 511, 2560)),
        (128, 2560, (128, 511, 2560)),
        (128, 2561, (128, 2560, None)),
        (128, 16384, (128, 2560, None)),
        (32, 1, (32, 0, 9)),
        (32, 9, (32, 0, 9)),
        (32, 65, (32, 64, 256)),
        (32, 256, (32, 64, 256)),
        (32, 768, (32, 767, 2047)),
        (32, 2047, (32, 767, 2047)),
        (32, 2048, (32, 2047, None)),
        (32, 16384, (32, 2047, None)),
        (16, 1, (16, 0, 1024)),
        (16, 1024, (16, 0, 1024)),
        (16, 1025, (16, 1024, 2047)),
        (16, 2047, (16, 1024, 2047)),
        (16, 2048, (16, 2047, 8191)),
        (16, 8191, (16, 2047, 8191)),
        (16, 8192, (16, 8191, None)),
        (16, 16384, (16, 8191, None)),
    ):
        assert (
            loader.selected_module("sm_107a", route, tokens, heads) == windows[key]
        ), (heads, tokens)
    for heads, tokens in (
        (128, 1),
        (128, 3),
        (128, 8),
        (32, 10),
        (32, 64),
        (32, 257),
        (32, 767),
    ):
        with pytest.raises(
            NotImplementedError, match=f"admits {tokens} rows of {heads} heads"
        ):
            loader.selected_module("sm_107a", route, tokens, heads)
    two_rows = loader.MODULES[windows[(128, 64, 256)]]
    assert (
        two_rows["grid_rule"]["rows_per_cta"] == 64
        and two_rows["build"]["tail"] == "clamp"
    )
    assert loader.launch_grid(two_rows, 65, device_index=0, values={"heads": 128}) == (
        65,
        2,
        1,
    )
    packed = loader.MODULES[windows[(128, 511, 2560)]]
    assert packed["grid_rule"]["rows_per_cta"] == 32 and packed["build"]["packed"]
    assert loader.launch_grid(packed, 512, device_index=0, values={"heads": 128}) == (
        512,
        4,
        1,
    )
    multi = loader.MODULES[windows[(16, 8191, None)]]
    assert multi["grid_rule"]["rows_per_cta"] == 64 and multi["grid_rule"]["tpc"] == 4
    assert multi["build"]["tpc"] == 4 and multi["build"]["rows_per_thread"] == 2
    for tokens, blocks in ((8192, 2048), (8193, 2049), (16384, 4096)):
        grid = loader.launch_grid(multi, tokens, device_index=0, values={"heads": 16})
        assert grid == (blocks, 1, 1), tokens
    for key in (
        (128, 8, 64),
        (128, 64, 256),
        (128, 383, 511),
        (128, 511, 2560),
        (32, 64, 256),
        (32, 767, 2047),
        (16, 2047, 8191),
        (16, 8191, None),
    ):
        assert loader.MODULES[windows[key]]["arches"] == ["sm_107a"], key


# Row counts probing every window edge of op A's Blackwell routes (OP_A_PLAIN_WINDOWS, OP_A_PDL_WINDOWS, GAPS_PDL).
OP_A_PROBE_ROWS = (
    1,
    16,
    17,
    64,
    65,
    96,
    128,
    129,
    192,
    193,
    224,
    225,
    256,
    257,
    383,
    384,
    511,
    512,
    768,
    769,
    1024,
    1025,
    2048,
    2049,
    4095,
    4096,
    4097,
    8191,
    8192,
    10**6,
)


def test_gated_residual_rmsnorm_routes_window_the_row_count():
    """Op A's Blackwell plain routes (two architectures x two modes) hold the programs of OP_A_PLAIN_WINDOWS over
    the windows 1-16 / 17-64 / 65-128 of the three small-M composites and the whole-row programs' windows, which
    partition the rows above 128 (the parked program's late-store form a sixth record on sm_100a in mode 1 and in
    the parked program's place on sm_103a in mode 1); the records appear in the route in window order, each once,
    and the programs of a variant differ only by their windows."""
    for (arch, mode), table in OP_A_PLAIN_WINDOWS.items():
        route = loader.route_key(OP_A, f"mode{mode}")
        names = loader.route_modules(arch, route)
        programs = _op_a_programs(arch, mode)
        assert list(names) == [programs[place] for place, _windows in table], (
            arch,
            route,
        )
        windows = [
            [
                tuple(window)
                for window in loader.MODULES[name]["select_rule"]["windows"][arch]
            ]
            for name in names
        ]
        assert windows == [expected for _place, expected in table], (arch, route)
        for rows in OP_A_PROBE_ROWS:
            (owner,) = [
                name
                for name, bounds in zip(names, windows, strict=True)
                if any(above < rows <= (at_most or rows) for above, at_most in bounds)
            ]
            assert loader.selected_module(arch, route, rows) == owner, (
                arch,
                route,
                rows,
            )
        if "psb_late" in programs:
            late_form = loader.MODULES[programs["psb_late"]]["build"]
            assert late_form["planes_smem"] == "beta"
            assert late_form["planes_path"] == "sts_late"
        for place in ("rsa", "pro", "psb", "psb_late"):
            if place not in programs:
                continue
            window = loader.MODULES[programs[place]]
            assert window["grid_rule"] == {"cluster": 1, "kind": "rows", "row_ctas": 1}
            assert window["build"]["threads"] == 512 and window["build"]["split"] == 1
            assert tuple(window["launch"]["block"]) == (512, 1, 1)
        if "psb" in programs:
            parked = loader.MODULES[programs["psb"]]["build"]
            assert parked["planes_smem"] == "beta" and parked["planes_path"] == "sts"
        assert loader.MODULES[programs["pro"]]["build"]["planes_smem"] == "none"
        for name in names:
            build = loader.MODULES[name]["build"]
            assert build["mode"] == mode and build["pdl"] is False


def test_gated_residual_rmsnorm_pdl_routes_window_the_row_count_per_architecture():
    """Op A's Blackwell PDL routes have windows of their own per (architecture, mode): each names the PDL build of a
    program of the plain route, a PDL-only form of the prologue program (OP_A_PDL_ONLY: sm_100a, and abel and pk on sm_103a), the plain build
    itself (borrowed: launched without the attribute where the PDL build measured slower than the CuTe DSL kernel
    under PDL and the plain build faster) or no program (a declared gap, where neither did); the records appear in
    the route in window order, each once."""
    for (arch, mode), table in OP_A_PDL_WINDOWS.items():
        programs = _op_a_programs(arch, mode)
        plain_route = loader.route_key(OP_A, f"mode{mode}")
        route = loader.route_key(OP_A, f"mode{mode}_pdl")
        names = loader.route_modules(arch, route)
        order: list = []
        above = 0
        for at_most, key in table:
            low = above + 1
            high = at_most if at_most is not None else above + 4096
            if key is None:
                for rows in (low, high):
                    with pytest.raises(
                        NotImplementedError, match=f"admits {rows} rows"
                    ):
                        loader.selected_module(arch, route, rows)
                above = at_most
                continue
            program, form = (key, "pdl") if isinstance(key, str) else key
            if form == "plain":
                name = programs[program]
            elif program in OP_A_PDL_ONLY:
                name = _op_a_pdl_only(programs, program)
            else:
                name = _pdl_build_of(programs[program])
            if name not in order:
                order.append(name)
            record = loader.MODULES[name]
            for rows in (low, high):
                assert loader.selected_module(arch, route, rows) == name, (
                    arch,
                    route,
                    rows,
                )
            assert record["build"]["mode"] == mode
            assert record["build"]["pdl"] is (form == "pdl"), (arch, route, key)
            assert record["launch"]["use_pdl"] is (form == "pdl"), (arch, route, key)
            if form == "plain":
                assert record["route"] == plain_route
                assert [above, at_most] in record["select_rule"]["routes"][route][arch]
            else:
                assert (
                    record["route"] == route and "routes" not in record["select_rule"]
                )
            above = at_most
        assert list(names) == order, (arch, route)
    # the sm_107a PDL routes keep the plain tables (no graph-replay table of record there): the PDL build of every
    # plain program but the TMA-ring program, whose plain build they borrow above 4096 rows (no PDL build is shipped)
    for mode in (0, 1):
        route = loader.route_key(OP_A, f"mode{mode}_pdl")
        plain = loader.route_modules("sm_107a", loader.route_key(OP_A, f"mode{mode}"))
        pdl = loader.route_modules("sm_107a", route)
        ring = plain[-1]
        assert loader.MODULES[ring]["build"]["tma_stages"] == 3
        assert [_pdl_build_of(name) for name in plain[:-1]] + [ring] == list(pdl)
        for name in pdl[:-1]:
            assert "routes" not in loader.MODULES[name]["select_rule"]
        assert loader.MODULES[ring]["select_rule"]["routes"] == {
            route: {"sm_107a": [[4096, None]]}
        }
        assert loader.MODULES[ring]["build"]["pdl"] is False
        assert loader.MODULES[ring]["launch"]["use_pdl"] is False


def test_gated_residual_rmsnorm_routes_on_sm_107a():
    """On sm_107a op A's four routes hold five (mode 0) or six (mode 1) programs: the two composites alternate over
    1 / 2-16 / 17-32 / 33-64 rows in mode 0 (mode 1 keeps the whole-row program from 33), the whole-row program owns
    65-128 (33-128 in mode 1), the whole-row persistent program (one 1024-thread CTA per SM) owns 257-4096 and the
    TMA-ring persistent program the rows beyond (a plain build in both launch forms: the PDL routes borrow it); mode 1
    also parks 129-256, which mode 0 leaves to the FlashInfer kernel (a declared gap)."""
    for variant in SHIPPED_VARIANTS[OP_A]:
        mode = int(variant[4])
        route = loader.route_key(OP_A, variant)
        names = loader.route_modules("sm_107a", route)
        assert len(names) == (6 if mode else 5), route
        windows = {
            tuple(
                tuple(w)
                for w in loader.MODULES[name]["select_rule"]["windows"]["sm_107a"]
            ): name
            for name in names
        }
        assert set(windows) == OP_A_WINDOWS_SM107A[mode], route

        def owner(rows):
            (name,) = [
                name
                for ws, name in windows.items()
                for above, at_most in ws
                if above < rows <= (at_most or rows)
            ]
            return name

        samples = [1, 2, 16, 17, 32, 33, 64, 65, 128, 257, 8192, 8193, 10**6] + (
            [129, 256] if mode else []
        )
        for rows in samples:
            assert loader.selected_module("sm_107a", route, rows) == owner(rows), (
                route,
                rows,
            )
        if not mode:
            for rows in (129, 200, 256):
                with pytest.raises(NotImplementedError, match=f"admits {rows} rows"):
                    loader.selected_module("sm_107a", route, rows)
        persistent = loader.MODULES[owner(1024)]
        build = persistent["build"]
        assert (
            build["persistent"] is True
            and build["threads"] == 1024
            and build["split"] == 1
        )
        assert persistent["grid_rule"]["kind"] == "persistent_rows"
        assert persistent["grid_rule"]["cluster"] == 1
        assert tuple(persistent["launch"]["block"]) == (1024, 1, 1)
        assert persistent["arches"] == ["sm_107a"]
        ring = loader.MODULES[owner(10**6)]
        assert ring["build"]["tma_stages"] == 3 and ring["build"]["persistent"] is True
        assert ring["build"]["threads"] == 1024 and ring["build"]["pdl"] is False
        assert ring["build"]["ring_content"] == ("both" if mode else "res")
        assert ring["build"]["fill_first"] is bool(mode)
        assert ring["arches"] == ["sm_107a"]
        if mode:
            parked = loader.MODULES[owner(200)]["build"]
            assert parked["planes_smem"] == "beta" and parked["planes_path"] == "sts"
        for name in names:
            record = loader.MODULES[name]
            borrowed = "routes" in record["select_rule"]
            assert record["build"]["mode"] == mode
            assert record["build"]["pdl"] is (variant.endswith("_pdl") and not borrowed)


def test_gated_residual_rmsnorm_helpers_and_support(monkeypatch):
    """The variant of a gated residual RMSNorm call is its mode (``weight`` applied before the norm or absent) and
    its launch form; on the Blackwell architectures every variant's route is exported, the plain routes' windows
    partition the positive row counts and the mode-0 PDL routes leave 769-4095 (sm_100a) / 1025-2048 and 4097-8191 (sm_103a) rows
    to the FlashInfer kernel under PDL; on sm_107a mode 0 leaves 129-256 rows to it in both launch forms."""
    assert pkg.GATED_RESIDUAL_RMSNORM == OP_A
    assert pkg.gated_residual_rmsnorm_variant(False, False) == "mode0"
    assert pkg.gated_residual_rmsnorm_variant(False, True) == "mode0_pdl"
    assert pkg.gated_residual_rmsnorm_variant(True, False) == "mode1"
    assert pkg.gated_residual_rmsnorm_variant(True, True) == "mode1_pdl"
    for weight_before_norm in (False, True):
        for enable_pdl in (False, True):
            variant = pkg.gated_residual_rmsnorm_variant(weight_before_norm, enable_pdl)
            assert variant in SHIPPED_VARIANTS[OP_A]
            assert (
                pkg.gated_residual_rmsnorm_route(weight_before_norm, enable_pdl)
                == f"{OP_A}_{variant}"
            )
    device = torch.device("cuda", 0)
    for capability, arch in (((10, 0), "sm_100a"), ((10, 3), "sm_103a")):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *_, c=capability: c
        )
        for weight_before_norm in (False, True):
            for enable_pdl in (False, True):
                form = dict(
                    weight_before_norm=weight_before_norm, enable_pdl=enable_pdl
                )
                assert pkg.is_cake_gated_residual_rmsnorm_supported(device, **form)
                route = pkg.gated_residual_rmsnorm_route(**form)
                gaps = GAPS_PDL.get((arch, route, None), ())
                assert bool(gaps) is (enable_pdl and not weight_before_norm)
                for rows in OP_A_PROBE_ROWS:
                    expected = not any(low < rows <= high for low, high in gaps)
                    assert (
                        bool(
                            pkg.is_cake_gated_residual_rmsnorm_supported(
                                device, rows=rows, **form
                            )
                        )
                        is expected
                    ), (arch, form, rows)
                assert not pkg.is_cake_gated_residual_rmsnorm_supported(
                    device, rows=0, **form
                )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    for weight_before_norm in (False, True):
        for enable_pdl in (False, True):
            form = dict(weight_before_norm=weight_before_norm, enable_pdl=enable_pdl)
            assert pkg.is_cake_gated_residual_rmsnorm_supported(device, **form)
            for rows in (1, 16, 17, 32, 33, 64, 65, 128, 257, 8192, 8193, 10**6):
                assert pkg.is_cake_gated_residual_rmsnorm_supported(
                    device, rows=rows, **form
                ), (form, rows)
            for rows in (129, 200, 256):
                assert (
                    bool(
                        pkg.is_cake_gated_residual_rmsnorm_supported(
                            device, rows=rows, **form
                        )
                    )
                    is weight_before_norm
                ), (form, rows)
            assert not pkg.is_cake_gated_residual_rmsnorm_supported(
                device, rows=0, **form
            )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert not pkg.is_cake_gated_residual_rmsnorm_supported(device)
    assert not pkg.is_cake_gated_residual_rmsnorm_supported(torch.device("cpu"))


def test_gated_residual_rmsnorm_support_follows_the_row_count_through_gaps(monkeypatch):
    """A gap in a route's windows (no record admits the row count) is unsupported for that row count only; the other
    variants keep their own windows."""
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 3))
    route = pkg.gated_residual_rmsnorm_route(True, True)
    gapped = {
        "arches": ["sm_103a"],
        "route": route,
        "select_rule": {
            "kind": "rows_between",
            "windows": {"sm_103a": [[0, 64], [256, 2048]]},
        },
    }
    monkeypatch.setitem(loader.MODULES, "gapped_a", gapped)
    monkeypatch.setitem(loader.ROUTES["sm_103a"], route, ["gapped_a"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    form = dict(weight_before_norm=True, enable_pdl=True)
    assert pkg.is_cake_gated_residual_rmsnorm_supported(device, **form)
    for rows in (1, 64, 257, 2048):
        assert pkg.is_cake_gated_residual_rmsnorm_supported(device, rows=rows, **form)
    for rows in (65, 256, 2049):
        assert not pkg.is_cake_gated_residual_rmsnorm_supported(
            device, rows=rows, **form
        )
    with pytest.raises(NotImplementedError, match="admits 100 rows"):
        loader.selected_module("sm_103a", route, 100)
    assert pkg.is_cake_gated_residual_rmsnorm_supported(
        device, rows=100, weight_before_norm=False, enable_pdl=True
    )
    assert pkg.is_cake_gated_residual_rmsnorm_supported(
        device, rows=100, weight_before_norm=True, enable_pdl=False
    )


def test_rmsnorm_sigmoid_gate_helpers_and_support(monkeypatch):
    """The sigmoid gate's variant is its launch form (``bf16``, ``bf16_pdl``); the plain route's windows name the head
    counts 128, 32 and 16 and partition the positive token counts of each on the Blackwell architectures, so the
    predicate holds for those head counts at every token count and fails for every other head count (a 2-D
    ``(rows, 64)`` call is one head); the PDL route's windows leave the declared gaps to the CuTe DSL kernel under
    PDL; on sm_107a the plain route's declared gaps of 128 and 32 heads are unsupported token counts and the PDL
    route leaves gaps of its own."""
    assert pkg.RMSNORM_SIGMOID_GATE == OP_C
    assert pkg.HEAD_DIM == 64 and pkg.ROW_ALIGNMENT_BYTES == 32
    assert pkg.GATE_STRIDE_ELEMENTS == 16
    assert pkg.rmsnorm_sigmoid_gate_variant() == "bf16"
    assert pkg.rmsnorm_sigmoid_gate_variant(True) == "bf16_pdl"
    assert pkg.rmsnorm_sigmoid_gate_route() == f"{OP_C}_bf16"
    assert pkg.rmsnorm_sigmoid_gate_route(False) == f"{OP_C}_bf16"
    assert pkg.rmsnorm_sigmoid_gate_route(True) == f"{OP_C}_bf16_pdl"
    device = torch.device("cuda", 0)
    routed_heads = sorted({key[0] for key in OP_C_WINDOWS})
    assert routed_heads == [16, 32, 128]
    for capability, arch in (((10, 0), "sm_100a"), ((10, 3), "sm_103a")):
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda *_, c=capability: c
        )
        assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device)
        pdl_route = pkg.rmsnorm_sigmoid_gate_route(True)
        assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, enable_pdl=True)
        for heads in routed_heads:
            assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, enable_pdl=True
            )
            gaps = GAPS_PDL.get((arch, pdl_route, heads), ())
            for tokens in (
                1,
                2,
                7,
                8,
                9,
                64,
                65,
                255,
                256,
                257,
                1023,
                1024,
                1025,
                16384,
            ):
                expected = not any(low < tokens <= high for low, high in gaps)
                assert (
                    bool(
                        pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                            device, heads=heads, tokens=tokens, enable_pdl=True
                        )
                    )
                    is expected
                ), (arch, heads, tokens)
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, tokens=0, enable_pdl=True
            )
        for heads in (1, 2, 8, 64, 256):
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, enable_pdl=True
            )
        for heads in routed_heads:
            assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=heads)
            for tokens in (1, 8, 9, 256, 257, 1024, 1025, 16384):
                assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                    device, heads=heads, tokens=tokens
                ), (capability, heads, tokens)
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, tokens=0
            )
        for heads in (1, 2, 8, 64, 256):
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=heads)
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, tokens=64
            )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 7))
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device)
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, enable_pdl=True)
    for heads in routed_heads:
        for enable_pdl in (False, True):
            assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, enable_pdl=enable_pdl
            )
            assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                device, heads=heads, tokens=0, enable_pdl=enable_pdl
            )
    for heads, tokens, supported in (
        (128, 1, False),
        (128, 2, True),
        (128, 3, False),
        (128, 8, False),
        (128, 9, True),
        (128, 64, True),
        (128, 256, True),
        (128, 1024, True),
        (128, 16384, True),
        (32, 1, True),
        (32, 9, True),
        (32, 10, False),
        (32, 64, False),
        (32, 65, True),
        (32, 256, True),
        (32, 257, False),
        (32, 767, False),
        (32, 768, True),
        (32, 1025, True),
        (16, 1, True),
        (16, 1024, True),
        (16, 1025, True),
        (64, 64, False),
    ):
        assert (
            bool(
                pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                    device, heads=heads, tokens=tokens
                )
            )
            is supported
        ), (heads, tokens)
    # the PDL route on sm_107a: gaps of its own (one token at every head count, 128 heads at 25-39 and 41-47, 32 heads
    # at 14, 25-63, 66-69 and 71-257), the borrowed plain builds admitted like any window
    for heads, tokens, supported in (
        (128, 1, False),
        (128, 2, True),
        (128, 3, True),
        (128, 8, True),
        (128, 9, True),
        (128, 24, True),
        (128, 25, False),
        (128, 39, False),
        (128, 40, True),
        (128, 41, False),
        (128, 47, False),
        (128, 48, True),
        (128, 64, True),
        (128, 511, True),
        (128, 512, True),
        (128, 2560, True),
        (128, 2561, True),
        (128, 16384, True),
        (32, 1, False),
        (32, 2, True),
        (32, 9, True),
        (32, 10, True),
        (32, 12, True),
        (32, 13, True),
        (32, 14, False),
        (32, 15, True),
        (32, 17, True),
        (32, 18, True),
        (32, 24, True),
        (32, 25, False),
        (32, 63, False),
        (32, 64, True),
        (32, 65, True),
        (32, 66, False),
        (32, 69, False),
        (32, 70, True),
        (32, 71, False),
        (32, 128, False),
        (32, 129, False),
        (32, 256, False),
        (32, 257, False),
        (32, 767, True),
        (32, 2048, True),
        (32, 2049, True),
        (16, 1, False),
        (16, 2, True),
        (16, 2047, True),
        (16, 2048, True),
        (64, 64, False),
    ):
        assert (
            bool(
                pkg.is_cake_rmsnorm_sigmoid_gate_supported(
                    device, heads=heads, tokens=tokens, enable_pdl=True
                )
            )
            is supported
        ), (heads, tokens)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device)
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(torch.device("cpu"))


def test_rmsnorm_sigmoid_gate_support_follows_the_head_and_token_counts_through_gaps(
    monkeypatch,
):
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    route = pkg.rmsnorm_sigmoid_gate_route()
    gapped = {
        "arches": ["sm_100a"],
        "route": route,
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {"sm_100a": [[128, 0, 8], [128, 256, None], [32, 0, 1024]]},
        },
    }
    monkeypatch.setitem(loader.MODULES, "gapped_c", gapped)
    monkeypatch.setitem(loader.ROUTES["sm_100a"], route, ["gapped_c"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device)
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=128)
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=128, tokens=8)
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=128, tokens=257)
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=128, tokens=9)
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=128, tokens=256)
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=32, tokens=1024)
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=32, tokens=1025)
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(device, heads=16)
    with pytest.raises(NotImplementedError, match="admits 100 rows of 128 heads"):
        loader.selected_module("sm_100a", route, 100, 128)
    # a plain build the PDL route borrows: the PDL route reads the record's routes entry, the plain route its windows
    pdl_route = pkg.rmsnorm_sigmoid_gate_route(True)
    borrowed = {
        "arches": ["sm_100a"],
        "route": route,
        "select_rule": {
            "kind": "tokens_heads_between",
            "windows": {"sm_100a": [[16, 0, None]]},
            "routes": {pdl_route: {"sm_100a": [[16, 1024, None]]}},
        },
        "launch": {"use_pdl": False},
    }
    monkeypatch.setitem(loader.MODULES, "borrowed_c", borrowed)
    monkeypatch.setitem(loader.ROUTES["sm_100a"], pdl_route, ["borrowed_c"])
    monkeypatch.setattr(
        pkg, "_ADMIT_INDEX", {}
    )  # the replaced route rebuilds its admit index
    assert pkg.is_cake_rmsnorm_sigmoid_gate_supported(
        device, heads=16, tokens=1025, enable_pdl=True
    )
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
        device, heads=16, tokens=1024, enable_pdl=True
    )
    assert not pkg.is_cake_rmsnorm_sigmoid_gate_supported(
        device, heads=32, tokens=1025, enable_pdl=True
    )
    assert loader.selected_module("sm_100a", pdl_route, 1025, 16) == "borrowed_c"
    with pytest.raises(NotImplementedError, match="admits 1024 rows of 16 heads"):
        loader.selected_module("sm_100a", pdl_route, 1024, 16)


def test_the_launch_attribute_follows_the_record():
    """A record's ``launch.use_pdl`` is its binding's programmatic-stream-serialization attribute: in the binding of
    every PDL build, absent from every plain build's -- the plain builds the PDL routes borrow included, so a call of
    a PDL route in such a window launches without the attribute; op C's route borrows on every architecture (one
    plain build on sm_100a, three on sm_103a, four on sm_107a: the 16-head windows the PDL build loses) and op A's
    two PDL routes borrow the TMA-ring program's plain build on sm_107a (above 4096 rows); the Blackwell op A routes
    borrow nothing since the window re-measurement."""
    attribute = "cudaLaunchAttributeProgrammaticStreamSerialization"
    borrowed = {}
    for name, record in loader.MODULES.items():
        binding = (PACKAGE_DIR / "csrc" / f"{name}_binding.cu").read_text()
        assert (attribute in binding) is record["launch"]["use_pdl"], name
        assert record["launch"]["use_pdl"] is record["build"]["pdl"], name
        for route, per_arch in record["select_rule"].get("routes", {}).items():
            assert record["launch"]["use_pdl"] is False and attribute not in binding
            for arch in per_arch:
                borrowed.setdefault(arch, set()).add(route)
    op_c = loader.route_key(OP_C, "bf16_pdl")
    op_a_pdl = {
        loader.route_key(OP_A, "mode0_pdl"),
        loader.route_key(OP_A, "mode1_pdl"),
    }
    assert borrowed == {
        "sm_100a": {op_c},
        "sm_103a": {op_c},
        "sm_107a": {op_c, *op_a_pdl},
    }
    for arch, routes in borrowed.items():
        for route in routes:
            names = loader.route_modules(arch, route)
            forms = [loader.MODULES[n]["launch"]["use_pdl"] for n in names]
            plain = (
                {"sm_100a": 1, "sm_103a": 3, "sm_107a": 4}[arch] if route == op_c else 1
            )
            assert (
                forms.count(False) == plain and forms.count(True) == len(names) - plain
            ), (arch, route)


def test_launch_grid_and_args_follow_the_record(monkeypatch):
    record = next(  # a persistent record that sm_107a does not ship, whatever the registry's order
        r
        for r in loader.MODULES.values()
        if r["grid_rule"]["kind"] == "persistent_rows" and "sm_107a" not in r["arches"]
    )
    cluster = record["grid_rule"]["cluster"]
    monkeypatch.setattr(loader, "device_arch", lambda index: record["arches"][0])
    for capacity, rows, expected in (
        (148, 7, 7),
        (148, 4096, 148),
        (0, 5, 1),
        (3, 3, 3),
    ):
        monkeypatch.setattr(loader, "cluster_capacity", lambda *a, c=capacity: c)
        assert loader.launch_grid(record, rows, device_index=0) == (
            expected * cluster,
            1,
            1,
        )
    plain = {"library": "x", "grid_rule": {"kind": "rows", "cluster": 4}}
    assert loader.launch_grid(plain, 10, device_index=0) == (40, 1, 1)
    two_d = {
        "library": "x",
        "grid_rule": {
            "kind": "tokens_heads",
            "cluster": 1,
            "tcluster": 16,
            "order": "tokens",
            "interleave": 1,
            "rows_per_cta": 32,
        },
    }
    assert loader.launch_grid(two_d, 9, device_index=0, values={"heads": 128}) == (
        16,
        4,
        1,
    )
    assert loader.tokens_heads_grid(two_d["grid_rule"], 9, 128) == (16, 4, 1)
    multi = {**two_d["grid_rule"], "tcluster": 1, "rows_per_cta": 64, "tpc": 4}
    assert loader.tokens_heads_grid(multi, 8193, 16) == (2049, 1, 1)
    assert loader.launch_grid(
        {"library": "x", "grid_rule": multi}, 8192, device_index=0, values={"heads": 16}
    ) == (2048, 1, 1)
    with pytest.raises(ValueError, match="positive counts"):
        loader.tokens_heads_grid(two_d["grid_rule"], 0, 128)
    with pytest.raises(ValueError, match="positive row count"):
        loader.launch_grid(plain, 0, device_index=0)
    with pytest.raises(ValueError, match="unknown Cake norm grid rule"):
        loader.launch_grid(
            {"library": "x", "grid_rule": {"kind": "tokens", "cluster": 1}},
            1,
            device_index=0,
        )
    values = {
        logical: object()
        for logical in (*loader.LOGICAL_TENSORS, *loader.LOGICAL_SCALARS)
    }
    args = loader.launch_args(record, values, (7, 1, 1))
    assert args[:-3] == tuple(
        values[record["logical"][key]] for key in PARAMETERS[record["op"]]
    )
    assert args[-3:] == (7, 1, 1)
    with pytest.raises(ValueError, match="needs logical argument 'rows'"):
        loader.launch_args(
            record, {k: v for k, v in values.items() if k != "rows"}, (1, 1, 1)
        )
    name = record["library"]
    assert (
        loader.library_name(name, record["arches"][0])
        == f"{name}_{record['arches'][0]}"
    )
    with pytest.raises(RuntimeError, match="not delivered for sm_107a"):
        loader.library_name(name, "sm_107a")


def _checker():
    path = ROOT / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


def _package(checker):
    return checker.Package(
        name=pkg.__name__,
        path=str(PACKAGE_DIR.relative_to(ROOT)),
        loader="cake_jit.py",
        refs=checker.refs_generic,
    )


def test_manifest_passes_the_flashinfer_checker():
    checker = _checker()
    package = _package(checker)
    manifest = cake_build.manifest()
    findings = [
        *checker.check_manifest(package, PACKAGE_DIR, manifest),
        *checker.check_cuda_toolkit_floor(package, manifest),
        *checker.check_references(package, PACKAGE_DIR, manifest),
    ]
    requirements = ROOT / "requirements.txt"
    if requirements.is_file():
        findings.extend(checker.check_tvm_ffi_pin(package, manifest, requirements))
    assert findings == []


@pytest.mark.xfail(
    reason="pyproject.toml package-data entry for the package is a shared-file edit outside this branch",
    strict=False,
)
def test_package_data_lists_the_package():
    checker = _checker()
    table = checker.package_data(ROOT / "pyproject.toml")
    assert (
        checker.check_package_data(_package(checker), cake_build.manifest(), table)
        == []
    )


def test_the_package_alignment_covers_the_widest_global_access_of_every_kernel():
    """The package admits 32-byte aligned storage alone (``ALIGNMENT_BYTES``, ``ROW_ALIGNMENT_BYTES`` and the norm
    facade's ``_CAKE_NORM_ALIGNMENT`` agree, and the sigmoid gate's token stride is a 32-byte multiple of BF16
    elements): the widest global vector access of every shipped kernel text is 256 bits, so a view aligned to 16
    bytes only never reaches a program (``"auto"`` runs the CuTe DSL kernel there)."""
    from flashinfer import norm

    assert (
        pkg.ALIGNMENT_BYTES
        == pkg.ROW_ALIGNMENT_BYTES
        == norm._CAKE_NORM_ALIGNMENT
        == 32
    )
    assert pkg.GATE_STRIDE_ELEMENTS * torch.bfloat16.itemsize == pkg.ROW_ALIGNMENT_BYTES
    access = re.compile(r"\b(?:ld|st)\.global[\w.:]*?\.v(\d+)\.[bfsu](\d+)\b")
    widest_by_kernel = {}
    for path in sorted((PACKAGE_DIR / "csrc").glob("*_kernel.cu")):
        widths = [
            int(lanes) * int(bits) // 8
            for lanes, bits in access.findall(path.read_text())
        ]
        widest_by_kernel[path.name] = max(widths, default=0)
    assert widest_by_kernel, "no kernel text under csrc/"
    assert all(width <= pkg.ALIGNMENT_BYTES for width in widest_by_kernel.values()), (
        widest_by_kernel
    )
    assert pkg.ALIGNMENT_BYTES in widest_by_kernel.values(), (
        "no kernel text carries a 256-bit vector access"
    )


OP_B_SM100A_WINDOWS = {
    "alpha_tensor_pw": [
        [(0, 4)],
        [(4, 16)],
        [(16, 64), (128, 512)],
        [(64, 128)],
        [(512, 1151)],
        [(1151, 4096)],
        [(4096, None)],
    ],
    "alpha_scalar_nopw": [
        [(0, 1)],
        [(1, 64)],
        [(64, 148)],
        [(148, 256)],
        [(256, 383), (4096, None)],
        [(383, 4096)],
    ],
    "alpha_scalar_pw": [[(0, None)]],
}
OP_B_SM100A_POSITIONS = {
    "alpha_tensor_pw": (
        (1, 0),
        (4, 0),
        (5, 1),
        (16, 1),
        (17, 2),
        (64, 2),
        (65, 3),
        (128, 3),
        (129, 2),
        (512, 2),
        (513, 4),
        (1151, 4),
        (1152, 5),
        (1536, 5),
        (3072, 5),
        (4096, 5),
        (4097, 6),
        (10**6, 6),
    ),
    "alpha_scalar_nopw": (
        (1, 0),
        (2, 1),
        (64, 1),
        (65, 2),
        (148, 2),
        (149, 3),
        (256, 3),
        (257, 4),
        (383, 4),
        (384, 5),
        (512, 5),
        (3072, 5),
        (4096, 5),
        (4097, 4),
        (10**6, 4),
    ),
    "alpha_scalar_pw": ((1, 0), (255, 0), (256, 0), (10**6, 0)),
}
OP_B_SM100A_EDGE_ROWS = (
    1,
    4,
    5,
    16,
    17,
    64,
    65,
    128,
    129,
    148,
    149,
    256,
    257,
    383,
    384,
    512,
    513,
    1151,
    1152,
    4096,
    4097,
    10**6,
)


def _build_without(record: str, key: str) -> dict:
    return {k: v for k, v in loader.MODULES[record]["build"].items() if k != key}


def test_gated_residual_next_rmsnorm_routes_window_the_row_count_on_sm_100a(
    monkeypatch,
):
    """Op B's sm_100a routes: the route with a tensor alpha and a post weight holds seven records (the
    persistent four-CTA program with a two-row L2 prefetch owns 1152-4096 rows and differs from the plain
    persistent four-CTA program, which owns 17-64 and 129-512, only by that prefetch), the route with a
    scalar alpha and no post weight holds six (the L2 load-hint form owns 384-4096 rows and differs from
    its neighbour, which owns 257-383 and 4097 up, only by the hint), and the route with a scalar alpha and
    a post weight holds one; every row count from 1 up is served on all three."""
    for variant, expected in OP_B_SM100A_WINDOWS.items():
        route = loader.route_key(OP, variant)
        names = loader.route_modules("sm_100a", route)
        assert len(names) == len(expected), route
        windows = [
            [
                tuple(window)
                for window in loader.MODULES[name]["select_rule"]["windows"]["sm_100a"]
            ]
            for name in names
        ]
        assert windows == expected, route
        for rows, position in OP_B_SM100A_POSITIONS[variant]:
            assert loader.selected_module("sm_100a", route, rows) == names[position], (
                route,
                rows,
            )
    tensor_pw = loader.route_modules("sm_100a", loader.route_key(OP, "alpha_tensor_pw"))
    plain, prefetch = (loader.MODULES[tensor_pw[i]]["build"] for i in (2, 5))
    assert prefetch["prefetch_l2"] == 2 and plain["prefetch_l2"] == 0
    assert _build_without(tensor_pw[5], "prefetch_l2") == _build_without(
        tensor_pw[2], "prefetch_l2"
    )
    scalar_nopw = loader.route_modules(
        "sm_100a", loader.route_key(OP, "alpha_scalar_nopw")
    )
    neighbour, hinted = (loader.MODULES[scalar_nopw[i]]["build"] for i in (4, 5))
    assert hinted["l2_hint"] == "ld_ll" and neighbour["l2_hint"] is None
    assert _build_without(scalar_nopw[5], "l2_hint") == _build_without(
        scalar_nopw[4], "l2_hint"
    )
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 0))
    for tensor_alpha, has_post_weight in ((True, True), (False, False), (False, True)):
        for rows in OP_B_SM100A_EDGE_ROWS:
            assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
                device,
                rows=rows,
                tensor_alpha=tensor_alpha,
                has_post_weight=has_post_weight,
            ), (tensor_alpha, has_post_weight, rows)


OP_B_SM103A_WINDOWS = {
    "alpha_tensor_pw": [
        [(0, 32)],
        [(32, 64), (128, 512)],
        [(64, 128)],
        [(512, 768), (1151, 3584)],
        [(768, 1151)],
        [(3584, None)],
    ],
    "alpha_scalar_nopw": [
        [(0, 32)],
        [(32, 128), (320, 384), (2560, None)],
        [(128, 256)],
        [(256, 320), (384, 2560)],
    ],
    "alpha_scalar_pw": [[(0, None)]],
}
OP_B_SM103A_POSITIONS = {
    "alpha_tensor_pw": (
        (1, 0),
        (32, 0),
        (33, 1),
        (64, 1),
        (65, 2),
        (128, 2),
        (129, 1),
        (512, 1),
        (513, 3),
        (768, 3),
        (769, 4),
        (1024, 4),
        (1151, 4),
        (1152, 3),
        (2048, 3),
        (3072, 3),
        (3584, 3),
        (3585, 5),
        (10**6, 5),
    ),
    "alpha_scalar_nopw": (
        (1, 0),
        (32, 0),
        (33, 1),
        (128, 1),
        (129, 2),
        (256, 2),
        (257, 3),
        (320, 3),
        (321, 1),
        (384, 1),
        (385, 3),
        (1024, 3),
        (2048, 3),
        (2560, 3),
        (2561, 1),
        (10**6, 1),
    ),
    "alpha_scalar_pw": ((1, 0), (255, 0), (256, 0), (10**6, 0)),
}
OP_B_SM103A_EDGE_ROWS = (
    1,
    32,
    33,
    64,
    65,
    128,
    129,
    256,
    257,
    320,
    321,
    384,
    385,
    512,
    513,
    768,
    769,
    1151,
    1152,
    2560,
    2561,
    3584,
    3585,
    10**6,
)


def test_gated_residual_next_rmsnorm_routes_window_the_row_count_on_sm_103a(
    monkeypatch,
):
    """Op B's sm_103a routes: the route with a tensor alpha and a post weight holds six records (the
    persistent four-CTA program with a two-row L2 prefetch owns 513-768 and 1152-3584 rows and differs
    from the plain persistent four-CTA program, which owns 33-64 and 129-512, only by that prefetch; the
    quarter-row ring program that owned 1537-3072 rows is gone, so no record of the route rings its rows),
    the route with a scalar alpha and no post weight holds four (the L2 load-hint form owns 257-320 and
    385-2560 rows and differs from its neighbour, which owns 33-128, 321-384 and 2561 up, only by the
    hint), and the route with a scalar alpha and a post weight holds one; the two new programs ship on
    sm_103a alone, and every row count from 1 up is served on all three routes."""
    for variant, expected in OP_B_SM103A_WINDOWS.items():
        route = loader.route_key(OP, variant)
        names = loader.route_modules("sm_103a", route)
        assert len(names) == len(expected), route
        windows = [
            [
                tuple(window)
                for window in loader.MODULES[name]["select_rule"]["windows"]["sm_103a"]
            ]
            for name in names
        ]
        assert windows == expected, route
        for rows, position in OP_B_SM103A_POSITIONS[variant]:
            assert loader.selected_module("sm_103a", route, rows) == names[position], (
                route,
                rows,
            )
    tensor_pw = loader.route_modules("sm_103a", loader.route_key(OP, "alpha_tensor_pw"))
    plain, prefetch = (loader.MODULES[tensor_pw[i]]["build"] for i in (1, 3))
    assert prefetch["prefetch_l2"] == 2 and plain["prefetch_l2"] == 0
    assert _build_without(tensor_pw[3], "prefetch_l2") == _build_without(
        tensor_pw[1], "prefetch_l2"
    )
    assert loader.MODULES[tensor_pw[3]]["arches"] == ["sm_103a"]
    assert all(loader.MODULES[name]["build"]["ring"] is None for name in tensor_pw)
    scalar_nopw = loader.route_modules(
        "sm_103a", loader.route_key(OP, "alpha_scalar_nopw")
    )
    neighbour, hinted = (loader.MODULES[scalar_nopw[i]]["build"] for i in (1, 3))
    assert hinted["l2_hint"] == "ld_ll" and neighbour["l2_hint"] is None
    assert _build_without(scalar_nopw[3], "l2_hint") == _build_without(
        scalar_nopw[1], "l2_hint"
    )
    assert loader.MODULES[scalar_nopw[3]]["arches"] == ["sm_103a"]
    device = torch.device("cuda", 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (10, 3))
    for tensor_alpha, has_post_weight in ((True, True), (False, False), (False, True)):
        for rows in OP_B_SM103A_EDGE_ROWS:
            assert pkg.is_cake_gated_residual_next_rmsnorm_supported(
                device,
                rows=rows,
                tensor_alpha=tensor_alpha,
                has_post_weight=has_post_weight,
            ), (tensor_alpha, has_post_weight, rows)
