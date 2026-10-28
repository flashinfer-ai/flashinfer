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

Behavioural tests of the Cake all-gather matmul backend that need no GPU:
route-table coverage, launch geometry, the main tile-width rule and its
delivered-class table, operand classification, input validation, the
symmetric-memory rendezvous guard and the public dispatch. The multi-GPU
numerical test lives in ``test_all_gather_matmul_cake_e2e.py``.
"""

from __future__ import annotations

import gc
import importlib
import types
import weakref
from types import SimpleNamespace

import pytest
import torch

from flashinfer.jit import cake_all_gather_matmul as loader

backend = importlib.import_module(
    "flashinfer.comm.all_gather_matmul.cake_all_gather_matmul"
)


def _main_programs():
    for world_size in loader.SUPPORTED_WORLD_SIZES:
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            for b_layout in loader.B_LAYOUTS:
                yield world_size, dtype_name, b_layout


def _n128_classes(arch):
    """``(world_size, dtype, b_layout)`` classes with a delivered ``_n128`` main on
    ``arch``, read from the emitted table (three-element lists)."""

    return {
        (int(world_size), str(dtype_name), str(b_layout))
        for world_size, dtype_name, b_layout in loader.N128_CLASSES[arch]
    }


def _n128_programs():
    """Every class with a delivered ``_n128`` main (union over the architectures)."""

    classes = set()
    for arch in loader.ARCH_FLAGS:
        classes |= _n128_classes(arch)
    return sorted(classes)


def _cg2_classes(arch):
    """``(world_size, dtype, b_layout)`` classes with a delivered ``_cg2`` main (the
    CTA-pair twin of the N256 main) on ``arch``, read from the emitted table."""

    return {
        (int(world_size), str(dtype_name), str(b_layout))
        for world_size, dtype_name, b_layout in loader.CG2_CLASSES[arch]
    }


def _cg2_programs():
    """Every class with a delivered ``_cg2`` main (union over the architectures)."""

    classes = set()
    for arch in loader.ARCH_FLAGS:
        classes |= _cg2_classes(arch)
    return sorted(classes)


def _cg2s_classes(arch):
    """``(world_size, dtype, b_layout, form)`` classes with a delivered spill-wave
    N-split pair main on ``arch``, read from the emitted table."""

    return {
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in loader.CG2S_CLASSES[arch]
    }


def _cg2s_programs():
    """Every class with a delivered spill-wave N-split pair main (union over the
    architectures)."""

    classes = set()
    for arch in loader.ARCH_FLAGS:
        classes |= _cg2s_classes(arch)
    return sorted(classes)


def _order_classes(arch):
    """``(world_size, dtype, b_layout, form)`` classes with a delivered tile-order
    twin of the one-CTA N256 main on ``arch``, read from the emitted table."""

    return {
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in loader.ORDER_CLASSES[arch]
    }


def _order_programs():
    """Every class with a delivered tile-order twin (union over the architectures)."""

    classes = set()
    for arch in loader.ARCH_FLAGS:
        classes |= _order_classes(arch)
    return sorted(classes)


def _cg1s_classes(arch):
    """``(world_size, dtype, b_layout, form)`` classes with a delivered last-pass
    N-split twin of the one-CTA N256 main on ``arch``, read from the emitted
    table."""

    return {
        (int(world_size), str(dtype_name), str(b_layout), str(form))
        for world_size, dtype_name, b_layout, form in loader.CG1S_CLASSES[arch]
    }


def _cg1s_programs():
    """Every class with a delivered last-pass N-split twin (union over the
    architectures)."""

    classes = set()
    for arch in loader.ARCH_FLAGS:
        classes |= _cg1s_classes(arch)
    return sorted(classes)


def _spill_point_form(arch, world_size, dtype_name, b_layout, padded, n):
    """Form of the emitted spill point naming the class, padded rows and ``n`` on
    ``arch``, or ``None`` when no point does."""

    for entry in loader.CG2_SPILL_POINTS[arch]:
        if tuple(entry[:5]) == (world_size, dtype_name, b_layout, padded, n):
            return str(entry[5])
    return None


def test_route_table_covers_every_world_size_dtype_layout_and_phase():
    for phase in (0, 1):
        assert loader.barrier_program(phase) in loader.PROGRAMS
    assert loader.barrier_program(0) != loader.barrier_program(1)
    mains = set()
    for world_size, dtype_name, b_layout in _main_programs():
        program = loader.main_program(world_size, dtype_name, b_layout)
        assert program in loader.PROGRAMS
        mains.add(program)
    # One main kernel per (world size, dtype, weight layout): twelve distinct programs.
    assert len(mains) == 12
    # One ``_n128`` twin per class of the delivered-class table (union over the
    # architectures), each a program of its own; no other ``_n128`` route exists.
    twins = set()
    for world_size, dtype_name, b_layout in _n128_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, block_n=128)
        assert program in loader.PROGRAMS and program not in mains
        twins.add(program)
    assert len(twins) == len(_n128_programs())
    assert {key for key in loader.ROUTES if key.endswith("_n128")} == {
        f"main_{dtype_name}_ws{world_size}_{b_layout}_n128"
        for world_size, dtype_name, b_layout in _n128_programs()
    }
    # One ``_cg2`` CTA-pair twin per class of the delivered-class table, each a
    # program of its own; no other ``_cg2`` route exists.
    pairs = set()
    for world_size, dtype_name, b_layout in _cg2_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form="cg2")
        assert (
            program in loader.PROGRAMS and program not in mains and program not in twins
        )
        pairs.add(program)
    assert len(pairs) == len(_cg2_programs())
    assert {key for key in loader.ROUTES if key.endswith("_cg2")} == {
        f"main_{dtype_name}_ws{world_size}_{b_layout}_cg2"
        for world_size, dtype_name, b_layout in _cg2_programs()
    }
    # One spill-wave N-split pair twin per class of the delivered spill-class
    # table, each a program of its own; no other spill route exists.
    spills = set()
    for world_size, dtype_name, b_layout, form in _cg2s_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert program in loader.PROGRAMS and program not in mains | twins | pairs
        spills.add(program)
    assert len(spills) == len(_cg2s_programs())
    spill_keys = {key for key in loader.ROUTES if key.endswith(("_cg2s1", "_cg2s2"))}
    assert spill_keys == {
        f"main_{dtype_name}_ws{world_size}_{b_layout}_{form}"
        for world_size, dtype_name, b_layout, form in _cg2s_programs()
    }
    # One tile-order twin of the one-CTA main per class of the delivered
    # order-class table, each a program of its own; no other ``_po`` / ``_pi``
    # route exists.
    orders = set()
    for world_size, dtype_name, b_layout, form in _order_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert program in loader.PROGRAMS
        assert program not in mains | twins | pairs | spills
        orders.add(program)
    assert len(orders) == len(_order_programs())
    order_keys = {key for key in loader.ROUTES if key.endswith(("_po", "_pi"))}
    assert order_keys == {
        f"main_{dtype_name}_ws{world_size}_{b_layout}_{form}"
        for world_size, dtype_name, b_layout, form in _order_programs()
    }
    # One last-pass N-split twin of the one-CTA main per class of the delivered
    # class table, each a program of its own; no other ``_cg1s2`` route exists.
    splits = set()
    for world_size, dtype_name, b_layout, form in _cg1s_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert program in loader.PROGRAMS
        assert program not in mains | twins | pairs | spills | orders
        splits.add(program)
    assert len(splits) == len(_cg1s_programs())
    split_keys = {key for key in loader.ROUTES if key.endswith("_cg1s2")}
    assert split_keys == {
        f"main_{dtype_name}_ws{world_size}_{b_layout}_{form}"
        for world_size, dtype_name, b_layout, form in _cg1s_programs()
    }
    kslice_routes = {key for key in loader.ROUTES if loader._is_kslice_stage(key)}
    expected_routes = (
        2
        + 12
        + 2
        + len(twins)
        + len(pairs)
        + len(spills)
        + len(orders)
        + len(splits)
        + len(kslice_routes)
    )
    assert len(loader.ROUTES) == expected_routes
    # ws2 never has a twin: asking for one names the missing route.
    with pytest.raises(ValueError, match="main_bfloat16_ws2_n_major_n128"):
        loader.main_program(2, "bfloat16", "n_major", block_n=128)
    # The pair twin exists at the public width only, and only for the delivered forms.
    with pytest.raises(ValueError, match="block_n=256 only"):
        loader.main_program(8, "bfloat16", "n_major", block_n=128, form="cg2")
    with pytest.raises(ValueError, match="form"):
        loader.main_program(8, "bfloat16", "n_major", form="cg3")
    # A spill twin exists only for the classes with a point, at the public width,
    # and only at the form of that point (the four-tile form is delivered nowhere).
    with pytest.raises(ValueError, match="no all-gather matmul main program"):
        loader.main_program(8, "bfloat16", "n_major", form="cg2s2")
    with pytest.raises(ValueError, match="no all-gather matmul main program"):
        loader.main_program(4, "bfloat16", "k_major", form="cg2s1")
    with pytest.raises(ValueError, match="block_n=256 only"):
        loader.main_program(4, "bfloat16", "k_major", block_n=128, form="cg2s2")
    assert loader.peer_push_program() in loader.PROGRAMS
    assert loader.peer_push_wide_program() in loader.PROGRAMS
    assert loader.peer_push_program() != loader.peer_push_wide_program()
    assert set(loader.ROUTES.values()) == set(loader.PROGRAMS)


def test_n128_table_lists_ws4_and_ws8_classes_per_architecture_under_the_frozen_rule():
    """The table restricts the frozen rule to delivered programs: one entry per
    exported architecture, classes of world sizes 4 and 8 only, and the rule
    inputs copied from the kernel module (148 SMs; at most two N256 waves on
    sm_100a, one on sm_103a)."""

    assert set(loader.N128_CLASSES) == set(loader.ARCH_FLAGS)
    assert loader.SM_COUNT_BY_ARCH == {"sm_100a": 148, "sm_103a": 148}
    assert loader.N128_MAX_WAVES_BY_ARCH == {"sm_100a": 2, "sm_103a": 1}
    assert loader.SM_PUSH_BANDS_BY_ARCH == {
        "sm_100a": [[8, 1024, 1619]],
        "sm_103a": [[8, 1024, 1619]],
    }
    assert loader.PUSH_LAUNCH_ROWS_BY_ARCH == {
        "sm_100a": [[4, 512, 16, 8]],
        "sm_103a": [[4, 512, 16, 8], [8, 1024, 16, 8]],
    }
    assert loader.PUSH_KSLICES_BY_ARCH == {"sm_100a": [], "sm_103a": [[8, 512, 4]]}
    assert loader.MAIN_BLOCK_NS == (256, 128) and loader.BLOCK_N == 256
    for arch in loader.ARCH_FLAGS:
        classes = _n128_classes(arch)
        assert len(classes) == len(loader.N128_CLASSES[arch]), arch
        for world_size, dtype_name, b_layout in classes:
            assert world_size in (4, 8)
            assert dtype_name in loader.SUPPORTED_DTYPES.values()
            assert b_layout in loader.B_LAYOUTS


def test_cg2_n128_precedence_table_lists_every_architecture():
    """The CTA-pair precedence rule: on sm_100a the CTA pair takes
    precedence over the N128 wave rule (a call inside a measured rectangle of a
    delivered pair class keeps 256 so that ``main_form`` selects the pair twin),
    on sm_103a it does not; the table is copied from the kernel module and lists
    every exported architecture. Outside a rectangle the wave rule stands."""

    assert loader.CG2_N128_PRECEDENCE_ARCHES == _CG2_N128_PRECEDENCE_ARCHES
    assert loader.CG2_N128_PRECEDENCE_ARCHES == {"sm_100a": True, "sm_103a": False}
    assert set(loader.CG2_N128_PRECEDENCE_ARCHES) == set(loader.ARCH_FLAGS)
    for dtype_name in ("bfloat16", "float16"):
        cls = {"dtype_name": dtype_name, "b_layout": "k_major"}
        assert (
            loader.main_block_n(512, 2560, world_size=4, arch="sm_100a", **cls) == 256
        )
        assert (
            loader.main_form(
                512, 2560, world_size=4, arch="sm_100a", block_n=256, **cls
            )
            == "cg2"
        )
        assert loader.main_block_n(512, 2816, world_size=4, arch="sm_100a", **cls) == (
            128 if dtype_name == "bfloat16" else 256
        )
    bf16_n = {"dtype_name": "bfloat16", "b_layout": "n_major"}
    assert loader.main_block_n(512, 2560, world_size=4, arch="sm_100a", **bf16_n) == 256
    bf16_k = {"dtype_name": "bfloat16", "b_layout": "k_major"}
    assert loader.main_block_n(1024, 256, world_size=4, arch="sm_103a", **bf16_k) == 128


# ``[world_size, dtype, b_layout, min_padded_rows, max_padded_rows, min_cols,
# max_cols]`` rectangles; a class may list several, pairwise disjoint.
_CG2_REGIONS = {
    "sm_100a": [
        # The nine sm_100a rows measured on B200, as points on padded rows and exact N
        # (M=1025 pads to 1152): the ws2 bf16 n_major perf rows, the ws4 bf16 k_major QKV rows and the ws4 / ws8
        # bf16 k_major gate_up M=512 / M=1024 rows.
        [2, "bfloat16", "n_major", 2048, 2048, 2048, 2048],
        [2, "bfloat16", "n_major", 4096, 4096, 2048, 2048],
        [4, "bfloat16", "k_major", 1024, 1024, 2560, 2560],
        [4, "bfloat16", "k_major", 1152, 1152, 2560, 2560],
        [4, "bfloat16", "k_major", 2048, 2048, 2560, 2560],
        [4, "bfloat16", "k_major", 512, 512, 14336, 14336],
        [4, "bfloat16", "k_major", 1024, 1024, 14336, 14336],
        [8, "bfloat16", "k_major", 512, 512, 7168, 7168],
        [8, "bfloat16", "k_major", 1024, 1024, 7168, 7168],
        # The three measured ws2 M=1024 rows (N = 2048), as points: the bf16 n_major
        # class (its M=1000 row pads to the same point), the bf16 k_major class and
        # the fp16 k_major class at exactly 1024 padded rows.
        [2, "bfloat16", "n_major", 1024, 1024, 2048, 2048],
        [2, "bfloat16", "k_major", 1024, 1024, 2048, 2048],
        [2, "float16", "k_major", 1024, 1024, 2048, 2048],
        # The two measured SM-push ws4 M=512 rows (N = 2560), as points: the bf16
        # k_major and fp16 k_major classes at exactly 512 padded rows (the sm_100a
        # CTA-pair precedence rule: the pair takes precedence over the N128 wave
        # rule there, CG2_N128_PRECEDENCE_ARCHES).
        [4, "bfloat16", "k_major", 512, 512, 2560, 2560],
        [4, "float16", "k_major", 512, 512, 2560, 2560],
    ],
    "sm_103a": [
        # The two measured rectangles (min_cols 1 keeps every call in them routed).
        [2, "bfloat16", "n_major", 2048, 5760, 1, 2048],
        [4, "bfloat16", "k_major", 1024, 1536, 1, 6058],
        # The two measured gate_up M=512 rows, as points; the M=2048 gate_up rows are
        # not routed.
        [4, "bfloat16", "k_major", 512, 512, 14336, 14336],
        [8, "bfloat16", "k_major", 512, 512, 7168, 7168],
        # The three measured ws2 M=1024 rows (N = 2048), as points: the bf16 n_major
        # class (its M=1000 row pads to the same point), the bf16 k_major class and
        # the fp16 k_major class at exactly 1024 padded rows.
        [2, "bfloat16", "n_major", 1024, 1024, 2048, 2048],
        [2, "bfloat16", "k_major", 1024, 1024, 2048, 2048],
        [2, "float16", "k_major", 1024, 1024, 2048, 2048],
        # The two measured SM-push ws4 M=512 rows (N = 2560), as points: the bf16
        # k_major and fp16 k_major classes at exactly 512 padded rows.
        [4, "bfloat16", "k_major", 512, 512, 2560, 2560],
        [4, "float16", "k_major", 512, 512, 2560, 2560],
    ],
}
_CG2_CLASSES = {
    "sm_100a": {
        (2, "bfloat16", "n_major"),
        (2, "bfloat16", "k_major"),
        (2, "float16", "k_major"),
        (4, "bfloat16", "k_major"),
        (4, "float16", "k_major"),
        (8, "bfloat16", "k_major"),
    },
    "sm_103a": {
        (2, "bfloat16", "n_major"),
        (2, "bfloat16", "k_major"),
        (2, "float16", "k_major"),
        (4, "bfloat16", "k_major"),
        (4, "float16", "k_major"),
        (8, "bfloat16", "k_major"),
    },
}


def test_cg2_tables_list_the_measured_pair_regions_and_classes_per_architecture():
    """The CTA-pair twin is delivered for the classes with a measured rectangle:
    on sm_103a the ws2 bf16 n_major class at 2048..5760 padded rows / N <= 2048
    and at its M=1024 point (exactly 1024 padded rows at N = 2048), the ws2 bf16
    k_major and fp16 k_major classes at their M=1024 points (exactly 1024 padded
    rows at N = 2048), the ws4 bf16 k_major class at 1024..1536 padded rows /
    N <= 6058, at its gate_up point (exactly 512 padded rows at N = 14336) and
    at its QKV M=512 point (exactly 512 padded rows at N = 2560), the ws4 fp16
    k_major class at its M=512 point (exactly 512 padded rows at N = 2560), and
    the ws8 bf16 k_major class at its gate_up point (exactly 512 padded rows at
    N = 7168; the M=2048 gate_up rows are not routed); on sm_100a
    the ws2 bf16 n_major class at its three measured points (exactly 1024, 2048
    or 4096 padded rows at N = 2048), the ws2 bf16 k_major and fp16 k_major
    classes at their M=1024 points (exactly 1024 padded rows at N = 2048), the
    ws4 bf16 k_major class at its six (exactly 512, 1024, 1152 or 2048 padded
    rows at N = 2560; exactly 512 or 1024 padded rows at N = 14336), the ws4
    fp16 k_major class at its M=512 point (exactly 512 padded rows at N = 2560)
    and the ws8 bf16 k_major class at its two (exactly 512 or 1024 padded rows
    at N = 7168). The rule inputs are the kernel module's (clusters of two).
    On sm_100a the pair takes precedence over the N128 wave rule
    (``CG2_N128_PRECEDENCE_ARCHES``, the sm_100a CTA-pair precedence rule)."""

    assert set(loader.CG2_CLASSES) == set(loader.CG2_REGIONS) == set(loader.ARCH_FLAGS)
    assert loader.CG2_REGIONS == _CG2_REGIONS
    assert loader.MAIN_FORMS == ("cg1", "cg2", "cg2s1", "cg2s2", "po", "pi", "cg1s2")
    assert loader.PAIR_FORMS == ("cg2", "cg2s1", "cg2s2")
    assert loader.ORDER_FORMS == ("po", "pi")
    assert loader.ONE_CTA_SPLIT_FORMS == ("cg1s2",)
    assert loader.CG2_CLUSTER_DIMS == (2, 1, 1)
    for arch in loader.ARCH_FLAGS:
        classes = _cg2_classes(arch)
        assert len(classes) == len(loader.CG2_CLASSES[arch])
        assert classes == _CG2_CLASSES[arch], arch
        # every region belongs to a delivered class and every delivered class has a region
        assert {tuple(entry[:3]) for entry in loader.CG2_REGIONS[arch]} == classes, arch
    for retired in (
        "CG2_MIN_M_TILES_BY_ARCH",
        "CG2_MAX_M_TILES",
        "CG2_WS2_MIN_PADDED_ROWS",
        "chunk_m_tiles",
    ):
        assert not hasattr(loader, retired), retired


# ``[world_size, dtype, b_layout, padded_rows, n, form]`` points: the spill-wave
# N-split pair program of a measured row, on exact padded rows and exact N. The
# two measured bf16 k_major gate_up rows -- the ws4 M=512 row and the ws8 M=512
# row -- on each architecture, and on sm_100a the ws4 bf16 k_major QKV M=2048
# row, each to the form its entry names (``cg2s2``: two 256 x 128 spill tiles;
# ``cg2s1``, four 256 x 64, has no point).
_CG2_SPILL_POINTS = {
    "sm_100a": [
        [4, "bfloat16", "k_major", 512, 14336, "cg2s2"],
        [8, "bfloat16", "k_major", 512, 7168, "cg2s2"],
        [4, "bfloat16", "k_major", 2048, 2560, "cg2s2"],
    ],
    "sm_103a": [
        [4, "bfloat16", "k_major", 512, 14336, "cg2s2"],
        [8, "bfloat16", "k_major", 512, 7168, "cg2s2"],
    ],
}
_CG2S_CLASSES = {
    "sm_100a": {
        (4, "bfloat16", "k_major", "cg2s2"),
        (8, "bfloat16", "k_major", "cg2s2"),
    },
    "sm_103a": {
        (4, "bfloat16", "k_major", "cg2s2"),
        (8, "bfloat16", "k_major", "cg2s2"),
    },
}


def test_cg2_spill_point_table_lists_the_measured_points_per_architecture():
    """The spill-wave N-split pair twins are delivered for the classes with a
    measured point: on both architectures the ws4 bf16 k_major class at exactly
    512 padded rows / N = 14336 and the ws8 bf16 k_major class at exactly 512
    padded rows / N = 7168, on sm_100a the ws4 bf16 k_major class at exactly
    2048 padded rows / N = 2560 too, each to the form its pinned entry names.
    Every point lies inside a pair rectangle of its class and names a delivered
    spill class; the form order gives the host sequence's positional form
    codes."""

    assert set(loader.CG2_SPILL_POINTS) == set(loader.ARCH_FLAGS)
    assert set(loader.CG2S_CLASSES) == set(loader.ARCH_FLAGS)
    assert loader.CG2_SPILL_POINTS == _CG2_SPILL_POINTS
    for arch in loader.ARCH_FLAGS:
        classes = _cg2s_classes(arch)
        assert len(classes) == len(loader.CG2S_CLASSES[arch])
        assert classes == _CG2S_CLASSES[arch], arch
        # every point belongs to a delivered spill class and every delivered
        # spill class has a point
        points = loader.CG2_SPILL_POINTS[arch]
        assert {(*entry[:3], entry[5]) for entry in points} == classes, arch
        assert len({tuple(entry[:5]) for entry in points}) == len(points), arch
        for world_size, dtype_name, b_layout, padded, n, form in points:
            assert (world_size, dtype_name, b_layout) in _cg2_classes(arch), arch
            assert form in loader.PAIR_FORMS[1:] and b_layout == "k_major"
            assert padded % loader.BLOCK_M == 0 and n % loader.BLOCK_N == 0
            rectangles = [
                entry[3:]
                for entry in loader.CG2_REGIONS[arch]
                if tuple(entry[:3]) == (world_size, dtype_name, b_layout)
            ]
            assert any(
                min_rows <= padded <= max_rows and min_cols <= n <= max_cols
                for min_rows, max_rows, min_cols, max_cols in rectangles
            ), (arch, world_size, dtype_name, b_layout, padded, n)
    assert [loader.main_form_code(form) for form in loader.MAIN_FORMS] == [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
    ]
    with pytest.raises(ValueError, match="form"):
        loader.main_form_code("pair")


# ``[world_size, dtype, b_layout, padded_rows, n, form]`` tile-order points: the
# one-CTA N256 main's ``po`` / ``pi`` twins (another tile -> (peer, B panel) map
# of the persistent walk; the one-CTA launch and grid) on exact padded rows
# (always 128: one M tile, one chunk) and exact N. One point is delivered, on
# sm_103a alone: the four-rank bf16 K-major calls of 128 padded rows at
# N = 14336 (1..128 rows) run the ``po`` twin; sm_100a lists none, and no
# ``pi`` twin is delivered anywhere.
_TILE_ORDER_POINTS = {
    "sm_100a": [],
    "sm_103a": [[4, "bfloat16", "k_major", 128, 14336, "po"]],
}
_ORDER_CLASSES = {"sm_100a": set(), "sm_103a": {(4, "bfloat16", "k_major", "po")}}
_ORDER_POINT_ROUTE = "main_bfloat16_ws4_k_major_po"
# ``(rows, n, world_size)`` bf16 K-major calls at the point (1..128 rows pad to
# the one 128-row tile) and off it (129 pads to 256; the neighbouring column
# multiples; the eight-rank gate_up M=125 call; a two-rank call).
_ORDER_POINT_CALLS = ((1, 14336, 4), (125, 14336, 4), (128, 14336, 4))
_OFF_POINT_CALLS = (
    (129, 14336, 4),
    (256, 14336, 4),
    (125, 14080, 4),
    (125, 14592, 4),
    (125, 7168, 8),
    (1, 256, 2),
)


def test_tile_order_point_table_lists_the_ws4_bf16_k_major_point_on_sm_103a_and_the_forms_take_the_one_cta_grid():
    """The tile-order twins of the one-CTA main are form codes 4 / 5 of the host
    sequence. One point is delivered, on sm_103a alone: ``main_form`` selects
    ``po`` for exactly the four-rank bf16 K-major calls of 128 padded rows at
    N = 14336 there, one ``_po`` route ships (a program of its own next to the
    one-CTA main of the class) and ``main_program`` refuses the form for every
    other class and the ``pi`` form by name; sm_100a keeps every call on the
    one-CTA program. The twins' grid is the one-CTA grid at the public width and
    they exist at that width only."""

    assert set(loader.TILE_ORDER_POINTS) == set(loader.ARCH_FLAGS)
    assert set(loader.ORDER_CLASSES) == set(loader.ARCH_FLAGS)
    assert loader.TILE_ORDER_POINTS == _TILE_ORDER_POINTS
    for arch in loader.ARCH_FLAGS:
        assert _order_classes(arch) == _ORDER_CLASSES[arch], arch
        assert loader._order_classes(arch) == frozenset(_ORDER_CLASSES[arch]), arch
        on_point = "po" if arch == "sm_103a" else None
        for rows, n, world_size in _ORDER_POINT_CALLS:
            assert (
                loader._tile_order_point(
                    arch, world_size, "bfloat16", "k_major", loader.padded_rows(rows), n
                )
                == on_point
            ), (arch, rows)
            assert loader.main_form(
                rows,
                n,
                world_size=world_size,
                arch=arch,
                dtype_name="bfloat16",
                b_layout="k_major",
                block_n=256,
            ) == (on_point or "cg1"), (arch, rows)
        for rows, n, world_size in _OFF_POINT_CALLS:
            assert (
                loader._tile_order_point(
                    arch, world_size, "bfloat16", "k_major", loader.padded_rows(rows), n
                )
                is None
            ), (arch, rows, n, world_size)
            assert (
                loader.main_form(
                    rows,
                    n,
                    world_size=world_size,
                    arch=arch,
                    dtype_name="bfloat16",
                    b_layout="k_major",
                    block_n=256,
                )
                == "cg1"
            ), (arch, rows, n, world_size)
        # The other layout and dtype of the point geometry, and the 128-wide call,
        # stay on the one-CTA program on both architectures.
        for dtype_name, b_layout, block_n in (
            ("bfloat16", "n_major", 256),
            ("float16", "k_major", 256),
            ("bfloat16", "k_major", 128),
        ):
            assert (
                loader.main_form(
                    125,
                    14336,
                    world_size=4,
                    arch=arch,
                    dtype_name=dtype_name,
                    b_layout=b_layout,
                    block_n=block_n,
                )
                == "cg1"
            ), (arch, dtype_name, b_layout, block_n)
    # Exactly one order route: the ``_po`` twin of the ws4 bf16 k_major class, a
    # program of its own next to the class's one-CTA main. The ``_po`` suffix is
    # the route key's: a delivered program is named by the export's content
    # fingerprint (``cake_all_gather_matmul_<hash>``), not by its kernel symbol.
    assert {key for key in loader.ROUTES if key.endswith(("_po", "_pi"))} == {
        _ORDER_POINT_ROUTE
    }
    twin = loader.main_program(4, "bfloat16", "k_major", form="po")
    assert twin == loader.ROUTES[_ORDER_POINT_ROUTE]
    assert twin in loader.PROGRAMS
    assert twin != loader.main_program(4, "bfloat16", "k_major")
    for form in loader.ORDER_FORMS:
        assert loader.main_form_code(form) == 4 + loader.ORDER_FORMS.index(form)
        assert form not in loader.PAIR_FORMS
        assert (
            loader.main_grid(125, 14336, world_size=4, sm_count=148, form=form)
            == loader.main_grid(125, 14336, world_size=4, sm_count=148)
            == (148, 1, 1)
        )
        with pytest.raises(ValueError, match="block_n=256 only"):
            loader.main_grid(
                125, 14336, world_size=4, sm_count=148, block_n=128, form=form
            )
        with pytest.raises(ValueError, match="block_n=256 only"):
            loader._main_route_key(4, "bfloat16", "k_major", 128, form)
    # The ``pi`` twin is delivered for no class; the ``po`` twin for the point's
    # class alone.
    with pytest.raises(
        ValueError, match="no all-gather matmul main program is delivered"
    ):
        loader.main_program(4, "bfloat16", "k_major", block_n=256, form="pi")
    for world_size, dtype_name, b_layout in (
        (8, "bfloat16", "k_major"),
        (4, "bfloat16", "n_major"),
        (4, "float16", "k_major"),
        (2, "bfloat16", "k_major"),
    ):
        with pytest.raises(
            ValueError, match="no all-gather matmul main program is delivered"
        ):
            loader.main_program(world_size, dtype_name, b_layout, form="po")
    with pytest.raises(ValueError, match="no entry for arch 'sm_90a'"):
        loader._tile_order_point("sm_90a", 4, "bfloat16", "k_major", 128, 14336)
    with pytest.raises(ValueError, match="no entry for arch 'sm_90a'"):
        loader._order_classes("sm_90a")


# ``[world_size, dtype, b_layout, padded_rows, n, form]`` one-CTA last-pass
# N-split points: the one-CTA N256 main's ``cg1s2`` twin (its last persistent
# pass runs each output tile as two 128-wide half items on otherwise idle CTAs;
# the one-CTA launch and grid) on exact padded rows and exact N. One point is
# delivered, on sm_103a alone: the eight-rank bf16 K-major calls of 1152 padded
# rows at N = 1280 (1025..1152 rows) run the ``cg1s2`` twin; sm_100a lists none.
_CG1S_POINTS = {
    "sm_100a": [],
    "sm_103a": [[8, "bfloat16", "k_major", 1152, 1280, "cg1s2"]],
}
_CG1S_CLASSES = {"sm_100a": set(), "sm_103a": {(8, "bfloat16", "k_major", "cg1s2")}}
_CG1S_POINT_ROUTE = "main_bfloat16_ws8_k_major_cg1s2"
# ``(rows, n, world_size)`` bf16 K-major calls at the point (1025..1152 rows pad
# to 1152) and off it (1024 pads to 1024, the SM push band row; 1153 pads to
# 1280; the next column multiple; the four-rank call, inside the sm_103a pair
# rectangle of its class; a two-rank call).
_CG1S_POINT_CALLS = ((1025, 1280, 8), (1100, 1280, 8), (1152, 1280, 8))
_CG1S_OFF_POINT_CALLS = (
    (1024, 1280, 8),
    (1153, 1280, 8),
    (1025, 1536, 8),
    (1025, 1280, 4),
    (1025, 1280, 2),
)
# The off-point calls a pair rectangle of their class takes, per architecture
# (the four-rank call on sm_103a); every other off-point call stays on cg1.
_CG1S_OFF_POINT_PAIRED = {"sm_100a": set(), "sm_103a": {(1025, 1280, 4)}}


def test_cg1s_point_table_lists_the_ws8_bf16_k_major_point_on_sm_103a_and_the_form_takes_the_one_cta_grid():
    """The one-CTA main's last-pass N-split twin (``cg1s2``) is form code 6 of
    the host sequence. One point is delivered, on sm_103a alone: ``main_form``
    selects ``cg1s2`` for exactly the eight-rank bf16 K-major calls of 1152
    padded rows at N = 1280 there, one ``_cg1s2`` route ships (a program of its
    own next to the one-CTA main of the class, delivered for sm_103a only) and
    ``main_program`` refuses the form for every other class and at the 128-wide
    tile; sm_100a keeps every call of the point geometry on the one-CTA program.
    The twin's grid is the one-CTA grid at the public width."""

    assert set(loader.CG1S_POINTS) == set(loader.ARCH_FLAGS)
    assert set(loader.CG1S_CLASSES) == set(loader.ARCH_FLAGS)
    assert loader.CG1S_POINTS == _CG1S_POINTS
    for arch in loader.ARCH_FLAGS:
        assert _cg1s_classes(arch) == _CG1S_CLASSES[arch], arch
        assert loader._cg1s_classes(arch) == frozenset(_CG1S_CLASSES[arch]), arch
        on_point = "cg1s2" if arch == "sm_103a" else None
        for rows, n, world_size in _CG1S_POINT_CALLS:
            padded = loader.padded_rows(rows)
            assert (
                loader._cg1s_point(arch, world_size, "bfloat16", "k_major", padded, n)
                == on_point
            ), (arch, rows)
            # The point is no tile-order point: one form per one-CTA call.
            assert (
                loader._tile_order_point(
                    arch, world_size, "bfloat16", "k_major", padded, n
                )
                is None
            ), (arch, rows)
            assert loader.main_form(
                rows,
                n,
                world_size=world_size,
                arch=arch,
                dtype_name="bfloat16",
                b_layout="k_major",
                block_n=256,
            ) == (on_point or "cg1"), (arch, rows)
        for rows, n, world_size in _CG1S_OFF_POINT_CALLS:
            assert (
                loader._cg1s_point(
                    arch, world_size, "bfloat16", "k_major", loader.padded_rows(rows), n
                )
                is None
            ), (arch, rows, n, world_size)
            paired = (rows, n, world_size) in _CG1S_OFF_POINT_PAIRED[arch]
            assert loader.main_form(
                rows,
                n,
                world_size=world_size,
                arch=arch,
                dtype_name="bfloat16",
                b_layout="k_major",
                block_n=256,
            ) == ("cg2" if paired else "cg1"), (arch, rows, n, world_size)
        # The other layout and dtype of the point geometry, and the 128-wide call,
        # stay on the one-CTA program on both architectures.
        for dtype_name, b_layout, block_n in (
            ("bfloat16", "n_major", 256),
            ("float16", "k_major", 256),
            ("bfloat16", "k_major", 128),
        ):
            assert (
                loader.main_form(
                    1025,
                    1280,
                    world_size=8,
                    arch=arch,
                    dtype_name=dtype_name,
                    b_layout=b_layout,
                    block_n=block_n,
                )
                == "cg1"
            ), (arch, dtype_name, b_layout, block_n)
    # Exactly one last-pass N-split route: the ``_cg1s2`` twin of the ws8 bf16
    # k_major class, a program of its own next to the class's one-CTA main,
    # delivered for sm_103a alone and compiled into that architecture's host
    # sequence of the class only.
    assert {key for key in loader.ROUTES if key.endswith("_cg1s2")} == {
        _CG1S_POINT_ROUTE
    }
    twin = loader.main_program(8, "bfloat16", "k_major", form="cg1s2")
    assert twin == loader.ROUTES[_CG1S_POINT_ROUTE]
    assert twin in loader.PROGRAMS
    assert twin != loader.main_program(8, "bfloat16", "k_major")
    assert loader.PROGRAMS[twin]["arches"] == ["sm_103a"]
    for arch in loader.ARCH_FLAGS:
        sequence = loader.sequence_name(8, "bfloat16", "k_major", arch)
        delivered = (8, "bfloat16", "k_major", "cg1s2") in _CG1S_CLASSES[arch]
        assert (twin in loader.sequence_programs(sequence)) == delivered, arch
        assert ("main_cg1s2" in loader.SEQUENCES[sequence]["programs"]) == delivered
    assert len(loader.MAIN_FORMS) == 7
    for form in loader.ONE_CTA_SPLIT_FORMS:
        assert loader.main_form_code(form) == 6
        assert form not in loader.PAIR_FORMS and form not in loader.ORDER_FORMS
        for sm_count, grid in ((148, (148, 1, 1)), (105, (105, 1, 1))):
            assert (
                loader.main_grid(1025, 1280, world_size=8, sm_count=sm_count, form=form)
                == loader.main_grid(1025, 1280, world_size=8, sm_count=sm_count)
                == grid
            )
        with pytest.raises(ValueError, match="block_n=256 only"):
            loader.main_grid(
                1025, 1280, world_size=8, sm_count=148, block_n=128, form=form
            )
        with pytest.raises(ValueError, match="block_n=256 only"):
            loader._main_route_key(8, "bfloat16", "k_major", 128, form)
        with pytest.raises(ValueError, match="block_n=256 only"):
            loader.main_program(8, "bfloat16", "k_major", block_n=128, form=form)
    # Code 7 names no form.
    with pytest.raises(ValueError, match="form"):
        loader.main_form_code("cg1s3")
    # The twin is delivered for the point's class alone.
    for world_size, dtype_name, b_layout in (
        (4, "bfloat16", "k_major"),
        (8, "bfloat16", "n_major"),
        (8, "float16", "k_major"),
        (2, "bfloat16", "k_major"),
    ):
        with pytest.raises(
            ValueError, match="no all-gather matmul main program is delivered"
        ):
            loader.main_program(world_size, dtype_name, b_layout, form="cg1s2")
    with pytest.raises(ValueError, match="no entry for arch 'sm_90a'"):
        loader._cg1s_point("sm_90a", 8, "bfloat16", "k_major", 1152, 1280)
    with pytest.raises(ValueError, match="no entry for arch 'sm_90a'"):
        loader._cg1s_classes("sm_90a")


def test_every_program_lists_its_device_source_only():
    for program, row in loader.PROGRAMS.items():
        assert len(row["sources"]) == 1, program
        assert row["sources"][0].startswith("csrc/cake_all_gather_matmul/")
        assert row["sources"][0].endswith("_kernel.cu")
        assert len(row["block"]) == 3 and row["block"][0] > 0
        assert row["dynamic_smem_bytes"] >= 0
        assert row["arches"] and set(row["arches"]) <= set(loader.ARCH_FLAGS), program


def test_every_route_has_one_host_sequence_per_architecture():
    expected = set()
    for arch in loader.ARCH_FLAGS:
        for world_size, dtype_name, b_layout in _main_programs():
            name = loader.sequence_name(world_size, dtype_name, b_layout, arch)
            expected.add(name)
            row = loader.SEQUENCES[name]
            assert row["arches"] == [arch]
            assert (row["world_size"], row["dtype"], row["b_layout"]) == (
                world_size,
                dtype_name,
                b_layout,
            )
            programs = loader.sequence_programs(name)
            # The sequence compiles the route's device units (one each) with
            # its launcher; the compile order carries no meaning.
            assert len(row["sources"]) == len(programs) + 1
            assert set(row["sources"][:-1]) == {
                loader.PROGRAMS[program]["sources"][0] for program in programs
            }
            assert row["sources"][-1].startswith(
                "csrc/cake_all_gather_matmul/cake_all_gather_matmul_sequence_"
            )
            # The file is this class's sequence for one architecture of
            # ``ARCH_FLAGS``; a row that builds from another architecture's
            # file names its own host namespace through ``sequence_flags``.
            file_arches = [
                candidate
                for candidate in loader.ARCH_FLAGS
                if row["sources"][-1].endswith(
                    loader.sequence_name(world_size, dtype_name, b_layout, candidate)
                    + ".cu"
                )
            ]
            assert len(file_arches) == 1
            if file_arches != [arch]:
                (flag,) = row["sequence_flags"]
                assert flag.startswith("-DCAKE_SEQ_SHIM=cake_host_shim_seq_")
            # A sequence of a world size the SM push route serves (4 and 8, as
            # ``uses_sm_push`` decides) compiles both SM push programs: the route
            # and the push launch are decided per call (rows, world size). No
            # two-rank call takes that route, so a world-size-2 sequence compiles
            # neither.
            sm_push = row["world_size"] >= 4
            assert (loader.peer_push_program() in programs) is sm_push
            assert (loader.peer_push_wide_program() in programs) is sm_push
            # The ``_n128`` twin travels with the sequence exactly for the classes
            # with a delivered twin on this architecture (the per-call width is
            # chosen by ``main_block_n``, so both mains are compiled in).
            route = {
                loader.barrier_program(0),
                loader.barrier_program(1),
                loader.main_program(world_size, dtype_name, b_layout),
            }
            if sm_push:
                route.add(loader.peer_push_program())
                route.add(loader.peer_push_wide_program())
            delivered = (world_size, dtype_name, b_layout) in _n128_classes(arch)
            assert ("main_n128" in row["programs"]) is delivered
            if delivered:
                twin = loader.main_program(
                    world_size, dtype_name, b_layout, block_n=128
                )
                assert row["programs"]["main_n128"] == twin
                assert arch in loader.PROGRAMS[twin]["arches"]
                route.add(twin)
            # The ``_cg2`` CTA-pair twin likewise (the per-call form is chosen by
            # ``main_form``, so the N256 main and its pair twin are compiled in).
            pair_delivered = (world_size, dtype_name, b_layout) in _cg2_classes(arch)
            assert ("main_cg2" in row["programs"]) is pair_delivered
            if pair_delivered:
                pair = loader.main_program(world_size, dtype_name, b_layout, form="cg2")
                assert row["programs"]["main_cg2"] == pair
                assert arch in loader.PROGRAMS[pair]["arches"]
                route.add(pair)
            # A spill-wave N-split pair twin likewise, for the classes with a
            # delivered point on this architecture (the per-call form is chosen
            # by ``main_form`` over the point table; its pair base travels too).
            for form in loader.PAIR_FORMS[1:]:
                spill_class = (world_size, dtype_name, b_layout, form)
                spill_delivered = spill_class in _cg2s_classes(arch)
                assert (f"main_{form}" in row["programs"]) is spill_delivered
                if spill_delivered:
                    assert pair_delivered
                    spill = loader.main_program(
                        world_size, dtype_name, b_layout, form=form
                    )
                    assert row["programs"][f"main_{form}"] == spill
                    assert arch in loader.PROGRAMS[spill]["arches"]
                    route.add(spill)
            # A tile-order twin likewise, for the classes with a delivered point of
            # that form on this architecture (the per-call form is chosen by
            # ``main_form`` over the tile-order point table: the ws4 bf16 k_major
            # ``po`` class on sm_103a, none on sm_100a).
            for form in loader.ORDER_FORMS:
                order_class = (world_size, dtype_name, b_layout, form)
                order_delivered = order_class in loader._order_classes(arch)
                assert (f"main_{form}" in row["programs"]) is order_delivered
                if order_delivered:
                    order_twin = loader.main_program(
                        world_size, dtype_name, b_layout, form=form
                    )
                    assert row["programs"][f"main_{form}"] == order_twin
                    assert arch in loader.PROGRAMS[order_twin]["arches"]
                    route.add(order_twin)
            # The one-CTA main's last-pass N-split twin likewise, for the classes
            # with a delivered point of that form on this architecture (the ws8
            # bf16 k_major ``cg1s2`` class on sm_103a, none on sm_100a).
            for form in loader.ONE_CTA_SPLIT_FORMS:
                split_class = (world_size, dtype_name, b_layout, form)
                split_delivered = split_class in loader._cg1s_classes(arch)
                assert (f"main_{form}" in row["programs"]) is split_delivered
                if split_delivered:
                    split_twin = loader.main_program(
                        world_size, dtype_name, b_layout, form=form
                    )
                    assert row["programs"][f"main_{form}"] == split_twin
                    assert arch in loader.PROGRAMS[split_twin]["arches"]
                    route.add(split_twin)
            # The per-K-slice readiness twins of the class's slice points on this
            # architecture travel with the sequence under their ``_ks<S>`` stage
            # names: the push twin of each point's launched program family and the
            # main twin of every width and form a call at a point can resolve to
            # (each a delivered program of this architecture); a class without a
            # point carries none.
            kslice_stages = {
                stage: program
                for stage, program in row["programs"].items()
                if loader._is_kslice_stage(stage)
            }
            slice_rows = [
                row_entry
                for row_entry in loader.push_kslice_rows(arch)
                if row_entry[0] == world_size
            ]
            assert bool(kslice_stages) is bool(slice_rows)
            for stage, program in kslice_stages.items():
                base, _separator, slices = stage.rpartition("_ks")
                assert int(slices) in {k_slices for _ws, _rows, k_slices in slice_rows}
                assert arch in loader.PROGRAMS[program]["arches"]
                if base in ("peer_push", "peer_push_wide"):
                    assert program == loader.ROUTES[stage]
                else:
                    block_n = 128 if base.startswith("main_n128") else 256
                    if base in ("main", "main_n128"):
                        form = "cg1"
                    else:
                        form = base[len("main_") :]
                    assert program == loader.main_program(
                        world_size,
                        dtype_name,
                        b_layout,
                        block_n=block_n,
                        form=form,
                        k_slices=int(slices),
                    )
                route.add(program)
            assert len(programs) == len(route) and set(programs) == route
    assert set(loader.SEQUENCES) == expected
    with pytest.raises(ValueError, match="host sequence"):
        loader.sequence_name(16, "bfloat16", "n_major", "sm_100a")


def test_every_program_is_delivered_for_both_architectures():
    for push in (loader.peer_push_program(), loader.peer_push_wide_program()):
        assert loader.PROGRAMS[push]["arches"] == ["sm_100a", "sm_103a"]
    for phase in (0, 1):
        assert loader.PROGRAMS[loader.barrier_program(phase)]["arches"] == [
            "sm_100a",
            "sm_103a",
        ]
    for world_size, dtype_name, b_layout in _main_programs():
        assert loader.PROGRAMS[loader.main_program(world_size, dtype_name, b_layout)][
            "arches"
        ] == [
            "sm_100a",
            "sm_103a",
        ]
    # An ``_n128`` main is delivered for exactly the architectures whose table
    # lists its class (a class routed on one architecture only ships one arch).
    for world_size, dtype_name, b_layout in _n128_programs():
        twin = loader.main_program(world_size, dtype_name, b_layout, block_n=128)
        assert loader.PROGRAMS[twin]["arches"] == sorted(
            arch
            for arch in loader.ARCH_FLAGS
            if (world_size, dtype_name, b_layout) in _n128_classes(arch)
        ), twin
    # A ``_cg2`` main likewise (every pair class -- ws2 bf16 n_major, ws2 bf16
    # k_major, ws2 fp16 k_major, ws4 bf16 k_major, ws4 fp16 k_major and ws8 bf16
    # k_major -- has pair-routed rows on both architectures; ws4 fp16 k_major
    # since the sm_100a CTA-pair precedence rule, its M=512 N=2560 point).
    assert sorted(_cg2_programs()) == [
        (2, "bfloat16", "k_major"),
        (2, "bfloat16", "n_major"),
        (2, "float16", "k_major"),
        (4, "bfloat16", "k_major"),
        (4, "float16", "k_major"),
        (8, "bfloat16", "k_major"),
    ]
    for world_size, dtype_name, b_layout in _cg2_programs():
        pair = loader.main_program(world_size, dtype_name, b_layout, form="cg2")
        pair_arches = sorted(
            arch
            for arch in loader.ARCH_FLAGS
            if (world_size, dtype_name, b_layout) in _cg2_classes(arch)
        )
        assert loader.PROGRAMS[pair]["arches"] == pair_arches, pair
        assert pair_arches == ["sm_100a", "sm_103a"], pair
    # A spill-wave N-split pair twin is delivered for exactly the architectures
    # whose point table names its class (the pinned spill classes: the ws4 and
    # ws8 bf16 k_major classes on both architectures, each at the form of its
    # points).
    assert sorted(_cg2s_programs()) == sorted(
        {cls for classes in _CG2S_CLASSES.values() for cls in classes}
    )
    for world_size, dtype_name, b_layout, form in _cg2s_programs():
        spill = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert loader.PROGRAMS[spill]["arches"] == sorted(
            arch
            for arch in loader.ARCH_FLAGS
            if (world_size, dtype_name, b_layout, form) in _cg2s_classes(arch)
        ), spill
        assert loader.PROGRAMS[spill]["arches"] == ["sm_100a", "sm_103a"], spill
    # A tile-order twin of the one-CTA main is delivered for exactly the
    # architectures whose point table names its class (the pinned order class:
    # the ws4 bf16 k_major ``po`` twin on sm_103a alone).
    assert sorted(_order_programs()) == sorted(
        {cls for classes in _ORDER_CLASSES.values() for cls in classes}
    )
    for world_size, dtype_name, b_layout, form in _order_programs():
        twin = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert loader.PROGRAMS[twin]["arches"] == sorted(
            arch
            for arch in loader.ARCH_FLAGS
            if (world_size, dtype_name, b_layout, form) in _order_classes(arch)
        ), twin
        assert loader.PROGRAMS[twin]["arches"] == ["sm_103a"], twin
    # The one-CTA main's last-pass N-split twin likewise (the pinned class: the
    # ws8 bf16 k_major ``cg1s2`` twin on sm_103a alone).
    assert sorted(_cg1s_programs()) == sorted(
        {cls for classes in _CG1S_CLASSES.values() for cls in classes}
    )
    for world_size, dtype_name, b_layout, form in _cg1s_programs():
        twin = loader.main_program(world_size, dtype_name, b_layout, form=form)
        assert loader.PROGRAMS[twin]["arches"] == sorted(
            arch
            for arch in loader.ARCH_FLAGS
            if (world_size, dtype_name, b_layout, form) in _cg1s_classes(arch)
        ), twin
        assert loader.PROGRAMS[twin]["arches"] == ["sm_103a"], twin
    sm103_sequence = loader.sequence_name(8, "bfloat16", "n_major", "sm_103a")
    loader.spec.cache_clear()
    try:
        with pytest.raises(ValueError, match="sm_100a"):
            loader.spec(sm103_sequence, "sm_100a")
    finally:
        loader.spec.cache_clear()


def test_barrier_and_push_programs_use_static_shared_memory_only():
    for phase in (0, 1):
        program = loader.barrier_program(phase)
        assert loader.launch_block(program) == (32, 1, 1)
        assert loader.dynamic_smem_bytes(program) == 0
    for push in (loader.peer_push_program(), loader.peer_push_wide_program()):
        assert loader.launch_block(push) == (128, 1, 1)
        assert loader.dynamic_smem_bytes(push) == 0


def test_push_launch_follows_the_per_peer_payload():
    # 128 padded rows x K x 2 B = 2 MiB per peer keep the 32-CTA launch; 512 rows
    # (8 MiB) take the wide launch; the threshold itself is on the wide side. The
    # eight-rank calls of these row counts have no launch row on either
    # architecture (the eight-rank launch row sits at 1024 padded rows, below).
    assert loader.push_payload_bytes(128, 2) == 2 * 1024 * 1024
    assert loader.push_payload_bytes(512, 2) == 8 * 1024 * 1024
    for arch in _BOTH:
        rule = lambda rows, world_size=8, arch=arch: loader.push_launch(  # noqa: E731
            rows, 2, world_size=world_size, arch=arch
        )
        assert rule(128) == ("peer_push", 32, 1)
        assert rule(512) == ("peer_push_wide", 8, 8)
        assert rule(128)[1:] == (loader.PUSH_CTAS_PER_PEER, loader.PUSH_UNROLL)
        assert rule(512)[1:] == (
            loader.PUSH_WIDE_CTAS_PER_PEER,
            loader.PUSH_WIDE_UNROLL,
        )
        threshold_rows = loader.PUSH_PAYLOAD_THRESHOLD_BYTES // (loader.K * 2)
        assert threshold_rows % loader.BLOCK_M == 0
        assert rule(threshold_rows)[0] == "peer_push_wide"
        assert rule(threshold_rows - loader.BLOCK_M)[0] == "peer_push"
        assert rule(128, world_size=4) == ("peer_push", 32, 1)
    for key in ("peer_push", "peer_push_wide"):
        assert loader.ROUTES[key] in loader.PROGRAMS
    assert loader.PUSH_CARVEOUT == -1


def test_push_launch_rows_give_the_sm_103a_listed_calls_sixteen_wide_ctas():
    """``PUSH_LAUNCH_ROWS_BY_ARCH`` replaces the payload rule for exactly the listed
    (world size, padded rows): on ``sm_103a`` the four-rank calls of exactly 512
    padded rows and the eight-rank calls of exactly 1024 padded rows (the SM push
    band row) launch 16 CTAs per peer of the wide program (the rule's 8 x 8
    elsewhere); the four-rank 128-row calls keep 32 x 1 and every other call of
    either world size the payload rule (``sm_100a`` lists the four-rank 512-row
    row too; its own test follows). An arch outside the table raises."""

    assert loader.push_launch_rows("sm_103a") == ((4, 512, 16, 8), (8, 1024, 16, 8))
    assert loader.push_launch_rows("sm_100a") == ((4, 512, 16, 8),)
    assert loader.push_launch(512, 2, world_size=4, arch="sm_103a") == (
        "peer_push_wide",
        16,
        8,
    )
    assert loader.push_launch(1024, 2, world_size=8, arch="sm_103a") == (
        "peer_push_wide",
        16,
        8,
    )
    # Keyed on the padded row count, not the payload.
    assert loader.push_launch(512, 4, world_size=4, arch="sm_103a") == (
        "peer_push_wide",
        16,
        8,
    )
    assert loader.push_launch(1024, 4, world_size=8, arch="sm_103a") == (
        "peer_push_wide",
        16,
        8,
    )
    assert loader.push_launch(512, 2, world_size=4, arch="sm_100a") == (
        "peer_push_wide",
        16,
        8,
    )
    assert loader.push_launch(1024, 2, world_size=8, arch="sm_100a") == (
        "peer_push_wide",
        8,
        8,
    )
    # Each row is keyed on its own world size: the other world size at the same
    # row count keeps the payload rule.
    assert loader.push_launch(512, 2, world_size=8, arch="sm_103a") == (
        "peer_push_wide",
        8,
        8,
    )
    assert loader.push_launch(1024, 2, world_size=4, arch="sm_103a") == (
        "peer_push_wide",
        8,
        8,
    )
    assert loader.push_launch(128, 2, world_size=4, arch="sm_103a") == (
        "peer_push",
        32,
        1,
    )
    for rows in (256, 384, 640, 1024):
        assert loader.push_launch(rows, 2, world_size=4, arch="sm_103a") == (
            "peer_push_wide",
            8,
            8,
        )
    for rows in (256, 384, 512, 640, 1152, 2048):
        assert loader.push_launch(rows, 2, world_size=8, arch="sm_103a") == (
            "peer_push_wide",
            8,
            8,
        )
    for arch in ("sm_90a", "sm_120a", ""):
        with pytest.raises(ValueError, match="no SM push launch rows"):
            loader.push_launch(512, 2, world_size=4, arch=arch)
        with pytest.raises(ValueError, match="no SM push launch rows"):
            loader.push_launch_rows(arch)


def test_push_kslices_table_lists_the_eight_rank_512_row_calls_at_four_slices_on_sm_103a():
    """``PUSH_KSLICES_BY_ARCH`` (per-K-slice readiness on the SM push chain) lists one
    row: on ``sm_103a`` the SM-push calls of world size 8 and exactly 512 padded rows
    publish four readiness words per (peer, chunk) -- the host sequence launches their
    ``_ks4`` push and main twins -- and every other call on either architecture
    publishes one; the backend sizes a workspace by the widest call of its capacity
    (the pitch's chunk count wherever the row is out of reach or narrower) and a
    captured call clears ``world_size * num_chunks * k_slices`` words. An arch outside
    the table raises."""

    assert loader.PUSH_KSLICES_BY_ARCH == {"sm_100a": [], "sm_103a": [[8, 512, 4]]}
    assert loader.push_kslice_rows("sm_103a") == ((8, 512, 4),)
    assert loader.push_kslice_rows("sm_100a") == ()
    assert loader.push_kslices(512, world_size=8, arch="sm_103a") == 4
    assert loader.max_push_kslices(8, "sm_103a") == 4
    for arch in _BOTH:
        for world_size in (2, 4, 8):
            for padded in (128, 256, 384, 512, 640, 1024, 2432, 65536):
                listed = (arch, world_size, padded) == ("sm_103a", 8, 512)
                assert loader.push_kslices(
                    padded, world_size=world_size, arch=arch
                ) == (4 if listed else 1)
                chunks = loader.chunk_plan(padded)[2]
                reached = (arch, world_size) == ("sm_103a", 8) and padded >= 512
                assert loader.push_signal_slots(
                    padded, world_size=world_size, arch=arch
                ) == (max(chunks, 4) if reached else chunks)
            if (arch, world_size) != ("sm_103a", 8):
                assert loader.max_push_kslices(world_size, arch) == 1
        assert backend._readiness_words(4, 512, arch) == 4 * loader.chunk_plan(512)[2]
        assert (
            backend._readiness_words(4, 65536, arch) == 4 * loader.chunk_plan(65536)[2]
        )
    assert backend._readiness_words(8, 512, "sm_100a") == 8
    assert backend._readiness_words(8, 512, "sm_103a") == 32
    for arch in ("sm_90a", "sm_120a", ""):
        with pytest.raises(ValueError, match="no SM push readiness slice rows"):
            loader.push_kslices(512, world_size=4, arch=arch)
        with pytest.raises(ValueError, match="no SM push readiness slice rows"):
            loader.push_kslice_rows(arch)
        with pytest.raises(ValueError, match="no SM push readiness slice rows"):
            loader.max_push_kslices(8, arch)
        with pytest.raises(ValueError, match="no SM push readiness slice rows"):
            loader.push_signal_slots(512, world_size=8, arch=arch)
    with pytest.raises(ValueError, match="pitch must be positive"):
        loader.push_signal_slots(0, world_size=8, arch="sm_103a")


def test_push_signal_slots_size_the_readiness_pad_by_the_widest_call_of_the_capacity():
    """With the ``sm_103a`` readiness slice row, a workspace is sized by the largest
    words-per-rank-row count any call up to its pitch publishes: the row's chunk count
    times its slice count once the pitch can serve the row, the pitch's own chunk count
    wherever that is larger (a large workspace never grows past the one-word sizing) or
    the row is out of reach; the other world sizes and the other architecture keep the
    chunk count. ``_publish_scratch`` refuses a signal pad too small for the widened
    view and sizes the generation's ticket counters by the same count."""

    slots = loader.push_signal_slots
    assert loader.max_push_kslices(8, "sm_103a") == 4
    assert loader.max_push_kslices(4, "sm_103a") == 1
    assert loader.max_push_kslices(8, "sm_100a") == 1
    assert slots(128, world_size=8, arch="sm_103a") == 1
    assert slots(384, world_size=8, arch="sm_103a") == 1
    for pitch in (512, 640, 2048, 2432, 2560, 7296):
        assert slots(pitch, world_size=8, arch="sm_103a") == 4, pitch
    assert loader.chunk_plan(7296)[2] == 3 and loader.chunk_plan(7424)[2] == 4
    assert slots(7424, world_size=8, arch="sm_103a") == 4
    assert slots(9856, world_size=8, arch="sm_103a") == 5
    assert loader.chunk_plan(65536)[2] == 27
    assert slots(65536, world_size=8, arch="sm_103a") == 27
    for pitch in (128, 512, 2048, 65536):
        chunks = loader.chunk_plan(pitch)[2]
        for world_size in (2, 4):
            assert slots(pitch, world_size=world_size, arch="sm_103a") == chunks
        assert slots(pitch, world_size=8, arch="sm_100a") == chunks
    assert backend._readiness_words(8, 512, "sm_103a") == 32
    assert backend._readiness_words(8, 512, "sm_100a") == 8
    assert backend._readiness_words(8, 65536, "sm_103a") == 8 * 27
    # Publication at 512 rows per peer on eight ranks: 32 words (128 bytes) must fit the pad.
    call = backend._Call(0, 0, 8, "g", "bfloat16", "sm_103a", 512, 1280, "k_major")

    class _PadHandle(_GrowthHandle):
        def __init__(self, signal_pad_size):
            self.signal_pad_size = signal_pad_size

    workspace = _published_workspace(push_buffers=None)
    workspace.world_size, workspace.arch = 8, "sm_103a"
    scratch = torch.empty(8, 512, 8, dtype=torch.bfloat16)
    backend._publish_scratch(workspace, call, 512, scratch, _PadHandle(2048))
    assert workspace.max_chunks == 1 and workspace.signal_slots == 4
    assert tuple(workspace.signal_pad.shape) == (32,)
    workspace = _published_workspace(push_buffers=None)
    workspace.world_size, workspace.arch = 8, "sm_103a"
    with pytest.raises(RuntimeError, match="cannot hold 32 readiness words"):
        backend._publish_scratch(workspace, call, 512, scratch, _PadHandle(64))
    # The same pitch on the other architecture keeps the one-word view and never reads the pad size.
    workspace = _published_workspace(push_buffers=None)
    workspace.world_size = 8
    backend._publish_scratch(
        workspace,
        backend._Call(0, 0, 8, "g", "bfloat16", "sm_100a", 512, 1280, "k_major"),
        512,
        scratch,
        _GrowthHandle(),
    )
    assert workspace.max_chunks == 1 and workspace.signal_slots == 1
    assert tuple(workspace.signal_pad.shape) == (8,)


def test_push_launch_rows_give_the_sm_100a_four_rank_512_row_calls_sixteen_wide_ctas():
    """The ``sm_100a`` four-rank row is the same: the four-rank calls of exactly 512
    padded rows launch 16 CTAs per peer of the wide program (the payload rule's
    8 x 8 there), keyed on the padded row count; the four-rank 128-row calls keep
    32 x 1 and every two-rank and eight-rank call the payload rule (``sm_100a`` has
    no eight-rank launch row: its 1024-row band calls launch the payload rule's
    8 x 8)."""

    assert loader.push_launch_rows("sm_100a") == ((4, 512, 16, 8),)
    assert loader.push_launch_rows("sm_100a") == loader.push_launch_rows("sm_103a")[:1]
    assert loader.push_launch(1024, 2, world_size=8, arch="sm_100a") == (
        "peer_push_wide",
        8,
        8,
    )
    for arch in _BOTH:
        assert loader.push_launch(512, 2, world_size=4, arch=arch) == (
            "peer_push_wide",
            16,
            8,
        )
        assert loader.push_launch(512, 4, world_size=4, arch=arch) == (
            "peer_push_wide",
            16,
            8,
        )
        assert loader.push_launch(512, 2, world_size=8, arch=arch) == (
            "peer_push_wide",
            8,
            8,
        )
        assert loader.push_launch(512, 2, world_size=2, arch=arch) == (
            "peer_push_wide",
            8,
            8,
        )
        assert loader.push_launch(128, 2, world_size=4, arch=arch) == (
            "peer_push",
            32,
            1,
        )
        for rows in (256, 384, 640, 1024):
            assert loader.push_launch(rows, 2, world_size=4, arch=arch) == (
                "peer_push_wide",
                8,
                8,
            )


def test_main_programs_share_one_block_and_dynamic_shared_memory_contract():
    blocks = set()
    smem = set()
    for world_size, dtype_name, b_layout in _main_programs():
        program = loader.main_program(world_size, dtype_name, b_layout)
        blocks.add(loader.launch_block(program))
        smem.add(loader.dynamic_smem_bytes(program))
    assert blocks == {(loader.MAIN_THREADS, 1, 1)}
    assert len(smem) == 1 and smem.pop() > 0
    # The ``_n128`` twins share the N256 block (one thread count per sequence)
    # and one dynamic shared memory value of their own, read per program from
    # the twin's build receipt (never assumed equal to the N256 value).
    twin_blocks = set()
    twin_smem = set()
    for world_size, dtype_name, b_layout in _n128_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, block_n=128)
        twin_blocks.add(loader.launch_block(program))
        twin_smem.add(loader.dynamic_smem_bytes(program))
    assert twin_blocks == {(loader.MAIN_THREADS, 1, 1)}
    assert len(twin_smem) == 1 and twin_smem.pop() > 0
    # The ``_cg2`` CTA-pair twins share the N256 block as well and one dynamic
    # shared memory value of their own (six 32 KiB stages per CTA, read from the
    # twin's build receipt).
    pair_blocks = set()
    pair_smem = set()
    for world_size, dtype_name, b_layout in _cg2_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form="cg2")
        pair_blocks.add(loader.launch_block(program))
        pair_smem.add(loader.dynamic_smem_bytes(program))
    assert pair_blocks == {(loader.MAIN_THREADS, 1, 1)}
    assert len(pair_smem) == 1 and pair_smem.pop() > 0
    # The spill-wave N-split pair twins share the N256 block as well (one thread
    # count per sequence) and, per form, one dynamic shared memory value of
    # their own read from the twin's build receipt.
    spill_blocks = set()
    spill_smem = {}
    for world_size, dtype_name, b_layout, form in _cg2s_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form=form)
        spill_blocks.add(loader.launch_block(program))
        spill_smem.setdefault(form, set()).add(loader.dynamic_smem_bytes(program))
    assert spill_blocks == {(loader.MAIN_THREADS, 1, 1)}
    for form, values in spill_smem.items():
        assert len(values) == 1 and values.pop() > 0, form
    # The one-CTA main's last-pass N-split twin shares the N256 block as well
    # and, per form, one dynamic shared memory value of its own read from the
    # twin's build receipt.
    split_blocks = set()
    split_smem = {}
    for world_size, dtype_name, b_layout, form in _cg1s_programs():
        program = loader.main_program(world_size, dtype_name, b_layout, form=form)
        split_blocks.add(loader.launch_block(program))
        split_smem.setdefault(form, set()).add(loader.dynamic_smem_bytes(program))
    assert split_blocks == {(loader.MAIN_THREADS, 1, 1)}
    for form, values in split_smem.items():
        assert len(values) == 1 and values.pop() > 0, form


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (1, 128),
        (125, 128),
        (128, 128),
        (129, 256),
        (1025, 1152),
        (19456, 19456),
    ],
)
def test_padded_rows_rounds_up_to_the_mma_tile(rows, expected):
    assert loader.padded_rows(rows) == expected


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (125, (128, 128, 1)),
        (128, (128, 128, 1)),
        (1025, (1152, 1152, 1)),
        (2432, (2432, 2432, 1)),
        (2433, (2560, 2432, 2)),
        (2560, (2560, 2432, 2)),
        (16384, (16384, 2432, 7)),
        (19456, (19456, 2432, 8)),
    ],
)
def test_chunk_plan_pads_rows_and_pushes_at_most_nineteen_row_blocks_per_chunk(
    rows, expected
):
    assert loader.chunk_plan(rows) == expected


@pytest.mark.parametrize(
    ("rows", "n", "world_size", "sm_count", "expected"),
    [
        (125, 1280, 8, 148, (40, 1, 1)),  # 8 x 1 x 5 tiles < SM count
        (512, 1280, 8, 148, (148, 1, 1)),  # 8 x 4 x 5 = 160 tiles: one CTA per SM
        (512, 2560, 4, 148, (148, 1, 1)),
        (125, 7168, 2, 148, (56, 1, 1)),  # 2 x 1 x 28
        (1025, 14336, 4, 148, (148, 1, 1)),
        (125, 1280, 8, 160, (40, 1, 1)),
    ],
)
def test_main_grid_is_one_cta_per_sm_bounded_by_the_total_tiles(
    rows, n, world_size, sm_count, expected
):
    assert (
        loader.main_grid(rows, n, world_size=world_size, sm_count=sm_count) == expected
    )


def test_main_grid_requires_a_positive_world_size_and_sm_count():
    with pytest.raises(ValueError, match="sm_count"):
        loader.main_grid(512, 1280, world_size=8, sm_count=0)
    with pytest.raises(ValueError, match="world_size"):
        loader.main_grid(512, 1280, world_size=0, sm_count=148)


@pytest.mark.parametrize(
    ("rows", "n", "world_size", "sm_count", "block_n", "tiles", "expected"),
    [
        (125, 1280, 8, 148, 256, 40, (40, 1, 1)),  # 8 x 1 x 5 N256 tiles
        (125, 1280, 8, 148, 128, 80, (80, 1, 1)),  # the same call on the twin
        (512, 1280, 8, 148, 256, 160, (148, 1, 1)),
        (512, 1280, 8, 148, 128, 320, (148, 1, 1)),  # one CTA per SM either way
        (125, 2560, 4, 148, 128, 80, (80, 1, 1)),
        (125, 2048, 8, 148, 128, 128, (128, 1, 1)),
        (125, 7168, 2, 148, 256, 56, (56, 1, 1)),  # ws2 only ever runs the N256 program
        (1025, 14336, 4, 148, 256, 2016, (148, 1, 1)),
    ],
)
def test_main_grid_and_main_tiles_count_the_tiles_of_the_selected_width(
    rows, n, world_size, sm_count, block_n, tiles, expected
):
    assert loader.main_tiles(rows, n, world_size=world_size, block_n=block_n) == tiles
    assert (
        loader.main_grid(
            rows, n, world_size=world_size, sm_count=sm_count, block_n=block_n
        )
        == expected
    )
    if block_n == 256:
        # The public width is the default.
        assert (
            loader.main_grid(rows, n, world_size=world_size, sm_count=sm_count)
            == expected
        )


def test_main_tiles_accepts_only_the_delivered_widths_and_a_width_multiple_n():
    with pytest.raises(ValueError, match="block_n"):
        loader.main_tiles(125, 1280, world_size=8, block_n=64)
    with pytest.raises(ValueError, match="multiple of 256"):
        loader.main_grid(125, 1152, world_size=8, sm_count=148)
    with pytest.raises(ValueError, match="multiple of 128"):
        loader.main_tiles(125, 1216, world_size=8, block_n=128)
    with pytest.raises(ValueError, match="positive"):
        loader.main_tiles(125, 0, world_size=8, block_n=256)


@pytest.mark.parametrize(
    ("rows", "n", "world_size", "sm_count", "pair_tiles", "expected"),
    [
        # 8 M tiles -> 4 pairs per peer; 8 N tiles; 2 peers: 64 pair tiles, every pair gets a cluster.
        (1024, 2048, 2, 148, 64, (128, 1, 1)),
        # 9 M tiles -> 5 pairs (the last pair's odd CTA idles); 4 x 5 x 8 = 160 > 74 clusters.
        (1025, 2048, 4, 148, 160, (148, 1, 1)),
        # 20 M tiles in two chunks (19 + 1): 10 + 1 pairs; 2 x 11 x 8 = 176.
        (2433, 2048, 2, 148, 176, (148, 1, 1)),
        # 512 M tiles in 27 chunks (26 x 19 + 18): 26 x 10 + 9 = 269 pairs; x 2 x 8.
        (65536, 2048, 2, 148, 4304, (148, 1, 1)),
        # One M tile: one pair per peer and N tile (the rule never selects the pair form here).
        (125, 1280, 8, 148, 40, (80, 1, 1)),
        # An odd SM count rounds down to whole clusters.
        (65536, 2048, 2, 147, 4304, (146, 1, 1)),
    ],
)
def test_main_grid_of_the_pair_main_is_even_and_bounded_by_the_pair_tiles(
    rows, n, world_size, sm_count, pair_tiles, expected
):
    assert loader.main_pair_tiles(rows, n, world_size=world_size) == pair_tiles
    grid = loader.main_grid(
        rows, n, world_size=world_size, sm_count=sm_count, form="cg2"
    )
    assert grid == expected and grid[0] % 2 == 0 and grid[0] <= 2 * pair_tiles
    assert grid[0] == 2 * max(1, min(pair_tiles, sm_count // 2))
    # The spill-wave N-split pair twins launch the same even grid.
    for form in loader.PAIR_FORMS[1:]:
        spill_grid = loader.main_grid(
            rows, n, world_size=world_size, sm_count=sm_count, form=form
        )
        assert spill_grid == grid, form
    # The one-CTA grid of the same call is the default form.
    assert loader.main_grid(
        rows, n, world_size=world_size, sm_count=sm_count
    ) == loader.main_grid(rows, n, world_size=world_size, sm_count=sm_count, form="cg1")


def test_main_grid_of_the_pair_main_exists_at_the_public_width_only():
    with pytest.raises(ValueError, match="block_n=256 only"):
        loader.main_grid(
            1024, 2048, world_size=2, sm_count=148, block_n=128, form="cg2"
        )
    with pytest.raises(ValueError, match="block_n=256 only"):
        loader.main_grid(
            1024, 2048, world_size=2, sm_count=148, block_n=128, form="cg2s1"
        )
    with pytest.raises(ValueError, match="form"):
        loader.main_grid(1024, 2048, world_size=2, sm_count=148, form="pair")
    with pytest.raises(ValueError, match="multiple of 256"):
        loader.main_pair_tiles(1024, 1152, world_size=2)


# The exporter's pair-over-N128 precedence flags, copied from the kernel module
# (the sm_100a CTA-pair precedence rule): where True, a default-path call inside
# a measured pair rectangle of its class keeps 256.
_CG2_N128_PRECEDENCE_ARCHES = {"sm_100a": True, "sm_103a": False}


def _frozen_rule(rows, n, *, world_size, arch, dtype_name, b_layout):
    """The exporter's N128 rule without the delivered-class table: at world sizes
    4 and 8 the twin is selected when ``2 * ceil(T256 / S) > ceil(2 * T256 / S)``
    for ``T256`` N256 tiles and ``S`` SMs, within the architecture's
    ``N128_MAX_WAVES_BY_ARCH`` N256 waves -- unless, on an architecture listed
    True in ``_CG2_N128_PRECEDENCE_ARCHES``, the call lies inside a measured pair
    rectangle of its ``(world_size, dtype_name, b_layout)`` class
    (``_CG2_REGIONS``): the CTA pair takes precedence and the call keeps 256."""

    if world_size not in (4, 8):
        return 256
    sm_count = loader.SM_COUNT_BY_ARCH[arch]
    t256 = loader.main_tiles(rows, n, world_size=world_size, block_n=256)
    w256 = -(-t256 // sm_count)
    w128 = -(-(2 * t256) // sm_count)
    cap = w256 <= loader.N128_MAX_WAVES_BY_ARCH[arch]
    padded = loader.padded_rows(rows)
    inside = any(
        entry[3] <= padded <= entry[4] and entry[5] <= n <= entry[6]
        for entry in _CG2_REGIONS[arch]
        if tuple(entry[:3]) == (world_size, dtype_name, b_layout)
    )
    if _CG2_N128_PRECEDENCE_ARCHES[arch] and inside:
        return 256
    return 128 if 2 * w256 > w128 and cap else 256


_BOTH = ("sm_100a", "sm_103a")
# Rows of the Cake export inventory (``exports/all_gather_matmul/shapes.json`` of
# the exporting Cake tree; it is not delivered with this test) with the tile
# width the exporter resolved them to under the frozen rule (``_frozen_rule``
# with S = 148 and the per-architecture wave cap ``N128_MAX_WAVES_BY_ARCH``:
# two N256 waves on sm_100a, one on sm_103a), keyed by the architectures the
# row is exported on. The rows the rule routes to the ``_n128`` main on some
# architecture are embedded verbatim: their classes define ``N128_CLASSES``
# per architecture (six routed rows on sm_100a, the three one-wave M=125
# rows on sm_103a). The other rows are at least one N256 row per
# world-size-4/8 class (the wave cap or the fill test fails there) and two ws2
# rows (always 256).
_N256 = {"sm_100a": 256, "sm_103a": 256}
# One N256 wave (40 or 64 tiles): within the cap of both architectures.
_N128 = {"sm_100a": 128, "sm_103a": 128}
# Two N256 waves (160 tiles against three on the twin): within the sm_100a cap
# of two, beyond the sm_103a cap of one.
_N128_SM100A = {"sm_100a": 128, "sm_103a": 256}
_INVENTORY_ROWS = (
    # (label, (rows, n, world_size, dtype, b_layout), {arch: block_n})
    (
        "tp8_correctness_llama_qkv_bf16_m512_n1280",
        (512, 1280, 8, "bfloat16", "n_major"),
        _N128_SM100A,
    ),
    # The sm_103a-only packed-QKV row: two waves, beyond the sm_103a cap.
    (
        "ws4_packed_qkv_bf16_m512_n2560",
        (512, 2560, 4, "bfloat16", "n_major"),
        {"sm_103a": 256},
    ),
    (
        "llama_tp8_qkv_bf16_m125_kmajor",
        (125, 1280, 8, "bfloat16", "k_major"),
        _N128,
    ),
    (
        "llama_tp8_qkv_bf16_m512_kmajor",
        (512, 1280, 8, "bfloat16", "k_major"),
        _N128_SM100A,
    ),
    (
        "llama_tp4_qkv_bf16_m125_kmajor",
        (125, 2560, 4, "bfloat16", "k_major"),
        _N128,
    ),
    # Two N256 waves inside the (4, bf16 | fp16, k_major) 512 x 2560 rectangles:
    # on sm_100a the pair takes precedence over the wave rule (the CTA-pair
    # precedence rule), on sm_103a the cap of one fails -- 256 on both.
    (
        "llama_tp4_qkv_bf16_m512_kmajor",
        (512, 2560, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "kmajor_ws4_fp16_m512_n2560",
        (512, 2560, 4, "float16", "k_major"),
        _N256,
    ),
    (
        "kmajor_ws8_fp16_m512_n1280",
        (512, 1280, 8, "float16", "k_major"),
        _N128_SM100A,
    ),
    (
        "tail_ws8_bf16_m125_n2048",
        (125, 2048, 8, "bfloat16", "n_major"),
        _N128,
    ),
    # The sm_103a-only ws8 512-row readiness-coverage rows: 32 / 64 N256 tiles
    # (one wave on the twin too) route the two bf16 rows to the twin; the fp16
    # n_major row (160 tiles, two waves) stays at 256, and its class has no
    # delivered twin on sm_103a anyway.
    (
        "kmajor_ws8_bf16_m512_n256",
        (512, 256, 8, "bfloat16", "k_major"),
        {"sm_103a": 128},
    ),
    (
        "tp8_correctness_bf16_m512_n512",
        (512, 512, 8, "bfloat16", "n_major"),
        {"sm_103a": 128},
    ),
    (
        "tp8_correctness_fp16_m512_n1280",
        (512, 1280, 8, "float16", "n_major"),
        {"sm_103a": 256},
    ),
    # 320 N256 tiles = three waves: beyond the cap of both architectures.
    (
        "llama_tp8_qkv_bf16_m1024_kmajor",
        (1024, 1280, 8, "bfloat16", "k_major"),
        _N256,
    ),
    # 360 N256 tiles = three waves: beyond the cap of both architectures; on
    # sm_103a its 1152 padded rows at N = 1280 are the one-CTA main's last-pass
    # N-split point.
    (
        "llama_tp8_qkv_bf16_m1025_kmajor",
        (1025, 1280, 8, "bfloat16", "k_major"),
        _N256,
    ),
    # 224 N256 tiles: 2 * ceil(224 / 148) = 4 = ceil(448 / 148), no idle slot saved.
    (
        "llama_tp8_gate_up_bf16_m125_kmajor",
        (125, 7168, 8, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_gate_up_bf16_m125_kmajor",
        (125, 14336, 4, "bfloat16", "k_major"),
        _N256,
    ),
    # The gate_up rows (896 / 3584 N256 tiles: beyond the cap of both
    # architectures); on both architectures the M=512 rows run the spill-wave
    # N-split pair twin of their point; on sm_100a the M=1024 rows (measured
    # points) keep the CTA-pair twin and the M=1025 (1152 padded rows) / M=2048
    # rows keep the one-CTA program; on sm_103a the M=1024 / M=1025 / M=2048
    # rows (not routed) keep the one-CTA program.
    (
        "llama_tp4_gate_up_bf16_m512_kmajor",
        (512, 14336, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp8_gate_up_bf16_m512_kmajor",
        (512, 7168, 8, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_gate_up_bf16_m1024_kmajor",
        (1024, 14336, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp8_gate_up_bf16_m1024_kmajor",
        (1024, 7168, 8, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_gate_up_bf16_m1025_kmajor",
        (1025, 14336, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp8_gate_up_bf16_m1025_kmajor",
        (1025, 7168, 8, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_gate_up_bf16_m2048_kmajor",
        (2048, 14336, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp8_gate_up_bf16_m2048_kmajor",
        (2048, 7168, 8, "bfloat16", "k_major"),
        _N256,
    ),
    # The pair-routed N256 rows (ws2 always 256; 320 / 360 / 640
    # N256 tiles at ws4 = three to five waves, beyond the cap of both
    # architectures): on sm_103a the measured rectangles route them, on sm_100a the
    # measured points (the m2048 QKV row is pair-routed on sm_100a only).
    (
        "perf_bf16_m2048",
        (2048, 2048, 2, "bfloat16", "n_major"),
        _N256,
    ),
    (
        "perf_bf16_m4096",
        (4096, 2048, 2, "bfloat16", "n_major"),
        _N256,
    ),
    (
        "llama_tp4_qkv_bf16_m1024_kmajor",
        (1024, 2560, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_qkv_bf16_m1025_kmajor",
        (1025, 2560, 4, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "llama_tp4_qkv_bf16_m2048_kmajor",
        (2048, 2560, 4, "bfloat16", "k_major"),
        _N256,
    ),
    # The N256 coverage rows of the float16 k_major classes (640 tiles: five waves).
    (
        "kmajor_ws4_fp16_m2048_n2560",
        (2048, 2560, 4, "float16", "k_major"),
        _N256,
    ),
    (
        "kmajor_ws8_fp16_m2048_n1280",
        (2048, 1280, 8, "float16", "k_major"),
        _N256,
    ),
    (
        "ws4_perf_bf16_m1024",
        (1024, 2048, 4, "bfloat16", "n_major"),
        _N256,
    ),
    (
        "tp8_perf_bf16_m1024",
        (1024, 2048, 8, "bfloat16", "n_major"),
        _N256,
    ),
    # Classes without any routed row (like the ws4 bfloat16 n_major class above):
    # the float16 n_major mains have no twin.
    (
        "tail_ws4_fp16_m1025_n2048",
        (1025, 2048, 4, "float16", "n_major"),
        _N256,
    ),
    (
        "tp8_correctness_fp16_m16384",
        (16384, 2048, 8, "float16", "n_major"),
        _N256,
    ),
    # The ws2 M=1024 rows (N = 2048; the M=1000 row pads to 1024): on both
    # architectures the measured points route them to the CTA-pair twin (the ws2
    # k_major classes' only twins).
    (
        "perf_bf16_m1024",
        (1024, 2048, 2, "bfloat16", "n_major"),
        _N256,
    ),
    (
        "tail_ws2_bf16_m1000_n2048",
        (1000, 2048, 2, "bfloat16", "n_major"),
        _N256,
    ),
    (
        "kmajor_ws2_bf16_m1024_n2048",
        (1024, 2048, 2, "bfloat16", "k_major"),
        _N256,
    ),
    (
        "kmajor_ws2_fp16_m1024_n2048",
        (1024, 2048, 2, "float16", "k_major"),
        _N256,
    ),
)


@pytest.mark.parametrize(
    ("label", "shape", "widths"),
    _INVENTORY_ROWS,
    ids=[row[0] for row in _INVENTORY_ROWS],
)
def test_main_block_n_routes_the_inventory_rows_like_the_exporter(label, shape, widths):
    """The delivered loader resolves every inventory row to the width the
    exporter built its route with; on the inventory the delivered-class table
    never changes the outcome of the frozen rule."""

    rows, n, world_size, dtype_name, b_layout = shape
    cls = {"dtype_name": dtype_name, "b_layout": b_layout}
    for arch, expected in widths.items():
        assert (
            loader.main_block_n(
                rows,
                n,
                world_size=world_size,
                arch=arch,
                dtype_name=dtype_name,
                b_layout=b_layout,
            )
            == expected
        ), (label, arch)
        assert (
            _frozen_rule(rows, n, world_size=world_size, arch=arch, **cls) == expected
        ), (label, arch)


@pytest.mark.parametrize(
    ("label", "shape", "widths"),
    _INVENTORY_ROWS,
    ids=[row[0] for row in _INVENTORY_ROWS],
)
def test_main_form_routes_the_inventory_rows_by_the_region_table(label, shape, widths):
    """The loader's ``main_form`` on the inventory is the frozen region rule:
    the CTA-pair form for an N256 row whose padded rows and N lie inside some
    measured rectangle of its class on the architecture -- on sm_103a
    perf_bf16_m1024 / m2048 / m4096 and tail_ws2_bf16_m1000_n2048 (ws2 bf16
    n_major), kmajor_ws2_bf16_m1024_n2048 (ws2 bf16 k_major),
    kmajor_ws2_fp16_m1024_n2048 (ws2 fp16 k_major), llama_tp4_qkv_bf16_m512 /
    m1024 / m1025_kmajor and llama_tp4_gate_up_bf16_m512_kmajor (ws4 bf16
    k_major), kmajor_ws4_fp16_m512_n2560 (ws4 fp16 k_major) and
    llama_tp8_gate_up_bf16_m512_kmajor (ws8 bf16 k_major); on sm_100a
    perf_bf16_m1024 / m2048 / m4096 and tail_ws2_bf16_m1000_n2048 (ws2 bf16
    n_major), kmajor_ws2_bf16_m1024_n2048 (ws2 bf16 k_major),
    kmajor_ws2_fp16_m1024_n2048 (ws2 fp16 k_major), llama_tp4_qkv_bf16_m512 /
    m1024 / m1025 / m2048_kmajor and llama_tp4_gate_up_bf16_m512 / m1024_kmajor
    (ws4 bf16 k_major), kmajor_ws4_fp16_m512_n2560 (ws4 fp16 k_major) and
    llama_tp8_gate_up_bf16_m512 / m1024_kmajor (ws8 bf16 k_major; the measured
    sm_100a points, the two M=512 N=2560 ws4 points since the sm_100a CTA-pair
    precedence rule); the one-CTA form for every other row.
    Of the pair rows, the spill points -- llama_tp4_gate_up_bf16_m512_kmajor
    and llama_tp8_gate_up_bf16_m512_kmajor on both architectures,
    llama_tp4_qkv_bf16_m2048_kmajor on sm_100a -- run the spill-wave N-split
    pair twin of the form their point names instead;
    llama_tp4_qkv_bf16_m1024_kmajor stays on the CTA-pair twin. Of the one-CTA
    rows, llama_tp4_gate_up_bf16_m125_kmajor on sm_103a (128 padded rows at
    N = 14336: the one tile-order point) runs the ``po`` tile-order twin of the
    one-CTA program and llama_tp8_qkv_bf16_m1025_kmajor on sm_103a (1152 padded
    rows at N = 1280: the one last-pass N-split point) runs the ``cg1s2``
    last-pass N-split twin of the one-CTA program; every other row, and those
    rows on sm_100a, runs the one-CTA program itself."""

    rows, n, world_size, dtype_name, b_layout = shape
    for arch, block_n in widths.items():
        form = loader.main_form(
            rows,
            n,
            world_size=world_size,
            arch=arch,
            dtype_name=dtype_name,
            b_layout=b_layout,
            block_n=block_n,
        )
        padded = loader.padded_rows(rows)
        delivered = (world_size, dtype_name, b_layout) in _cg2_classes(arch)
        rectangles = [
            entry[3:]
            for entry in loader.CG2_REGIONS[arch]
            if tuple(entry[:3]) == (world_size, dtype_name, b_layout)
        ]
        qualifies = block_n == 256 and any(
            min_rows <= padded <= max_rows and min_cols <= n <= max_cols
            for min_rows, max_rows, min_cols, max_cols in rectangles
        )
        point = _spill_point_form(arch, world_size, dtype_name, b_layout, padded, n)
        order = loader._tile_order_point(
            arch, world_size, dtype_name, b_layout, padded, n
        )
        order_delivered = (world_size, dtype_name, b_layout, order) in _order_classes(
            arch
        )
        split = loader._cg1s_point(arch, world_size, dtype_name, b_layout, padded, n)
        split_delivered = (world_size, dtype_name, b_layout, split) in _cg1s_classes(
            arch
        )
        if qualifies and delivered:
            expected_form = point or "cg2"
        elif block_n == 256 and order is not None and order_delivered:
            expected_form = order
        elif block_n == 256 and split is not None and split_delivered:
            expected_form = split
        else:
            expected_form = "cg1"
        assert form == expected_form, (label, arch)
        expected_split = (
            label
            in {
                "sm_103a": {"llama_tp8_qkv_bf16_m1025_kmajor"},
                "sm_100a": set(),
            }[arch]
        )
        assert (split is not None) is expected_split, (label, arch)
        assert (form in loader.ONE_CTA_SPLIT_FORMS) is expected_split, (label, arch)
        assert (form == "cg1s2") is expected_split, (label, arch)
        expected_order = (
            arch == "sm_103a" and label == "llama_tp4_gate_up_bf16_m125_kmajor"
        )
        assert (order is not None) is expected_order, (label, arch)
        assert (form in loader.ORDER_FORMS) is expected_order, (label, arch)
        assert (form == "po") is expected_order, (label, arch)
        expected_pair = (
            label
            in {
                "sm_103a": {
                    "perf_bf16_m1024",
                    "perf_bf16_m2048",
                    "perf_bf16_m4096",
                    "tail_ws2_bf16_m1000_n2048",
                    "kmajor_ws2_bf16_m1024_n2048",
                    "kmajor_ws2_fp16_m1024_n2048",
                    "llama_tp4_qkv_bf16_m512_kmajor",
                    "llama_tp4_qkv_bf16_m1024_kmajor",
                    "llama_tp4_qkv_bf16_m1025_kmajor",
                    "kmajor_ws4_fp16_m512_n2560",
                    "llama_tp4_gate_up_bf16_m512_kmajor",
                    "llama_tp8_gate_up_bf16_m512_kmajor",
                },
                "sm_100a": {
                    "perf_bf16_m1024",
                    "perf_bf16_m2048",
                    "perf_bf16_m4096",
                    "tail_ws2_bf16_m1000_n2048",
                    "kmajor_ws2_bf16_m1024_n2048",
                    "kmajor_ws2_fp16_m1024_n2048",
                    "llama_tp4_qkv_bf16_m1024_kmajor",
                    "llama_tp4_qkv_bf16_m1025_kmajor",
                    "llama_tp4_qkv_bf16_m2048_kmajor",
                    "llama_tp4_qkv_bf16_m512_kmajor",
                    "kmajor_ws4_fp16_m512_n2560",
                    "llama_tp4_gate_up_bf16_m512_kmajor",
                    "llama_tp4_gate_up_bf16_m1024_kmajor",
                    "llama_tp8_gate_up_bf16_m512_kmajor",
                    "llama_tp8_gate_up_bf16_m1024_kmajor",
                },
            }[arch]
        )
        assert (form in loader.PAIR_FORMS) is expected_pair, (label, arch)
        expected_spill = (
            label
            in {
                "sm_103a": {
                    "llama_tp4_gate_up_bf16_m512_kmajor",
                    "llama_tp8_gate_up_bf16_m512_kmajor",
                },
                "sm_100a": {
                    "llama_tp4_gate_up_bf16_m512_kmajor",
                    "llama_tp8_gate_up_bf16_m512_kmajor",
                    "llama_tp4_qkv_bf16_m2048_kmajor",
                },
            }[arch]
        )
        assert (point is not None) is expected_spill, (label, arch)
        assert (form in loader.PAIR_FORMS[1:]) is expected_spill, (label, arch)
        # Every inventory row the rule qualifies belongs to a delivered class (the table is derived from the rows).
        assert delivered or not qualifies, (label, arch)


@pytest.mark.parametrize("arch", _BOTH)
def test_main_form_selects_the_pair_twin_only_inside_a_measured_region_at_the_public_width(
    arch,
):
    for world_size, dtype_name, b_layout in _main_programs():
        n = {2: 2048, 4: 2560, 8: 1280}[world_size]
        rule = lambda rows, n=n, block_n=256: loader.main_form(  # noqa: E731
            rows,
            n,
            world_size=world_size,
            arch=arch,
            dtype_name=dtype_name,
            b_layout=b_layout,
            block_n=block_n,
        )
        if (world_size, dtype_name, b_layout) not in _cg2_classes(arch):
            # No delivered pair twin: the one-CTA program whatever the row count.
            assert (world_size, dtype_name, b_layout) not in _CG2_CLASSES[arch]
            for rows in (1, 1024, 1025, 2048, 4096, 32768, 65536):
                assert rule(rows) == "cg1", (
                    arch,
                    world_size,
                    dtype_name,
                    b_layout,
                    rows,
                )
            continue
        assert (world_size, dtype_name, b_layout) in _CG2_CLASSES[arch]
        if arch == "sm_100a":
            # The measured sm_100a points: exact padded rows at exact N.
            assert world_size in (2, 4, 8)
            if (world_size, b_layout) == (2, "n_major"):
                # (2, bfloat16, n_major): exactly 1024, 2048 or 4096 padded rows at
                # N = 2048 (897..1023, 1921..2047 and 3969..4095 pad onto the
                # points; 1025 pads to 1152 and leaves).
                assert dtype_name == "bfloat16"
                assert [
                    rule(rows)
                    for rows in (897, 1000, 1023, 1024, 1921, 2048, 3969, 4096)
                ] == ["cg2"] * 8
                assert [
                    rule(rows)
                    for rows in (
                        1,
                        896,
                        1025,
                        1152,
                        1920,
                        2049,
                        3000,
                        3968,
                        4097,
                        5760,
                        8192,
                        65536,
                    )
                ] == ["cg1"] * 12
                assert rule(1024, n=1792) == "cg1" and rule(1024, n=2304) == "cg1"
                assert rule(2048, n=1792) == "cg1" and rule(2048, n=2304) == "cg1"
                assert rule(4096, n=1792) == "cg1" and rule(4096, n=2304) == "cg1"
            elif world_size == 2:
                # (2, bfloat16, k_major) and (2, float16, k_major): exactly 1024
                # padded rows at exactly N = 2048 (897..1023 pad onto the point;
                # 1025 pads to 1152 and leaves); the classes have no other point.
                assert dtype_name in ("bfloat16", "float16") and b_layout == "k_major"
                assert [rule(rows) for rows in (897, 1000, 1023, 1024)] == ["cg2"] * 4
                assert [
                    rule(rows)
                    for rows in (
                        1,
                        896,
                        1025,
                        1152,
                        1921,
                        2048,
                        4096,
                        5760,
                        8192,
                        65536,
                    )
                ] == ["cg1"] * 10
                assert rule(1024, n=1792) == "cg1" and rule(1024, n=2304) == "cg1"
                for cols in (2047, 2049):
                    with pytest.raises(ValueError, match="multiple of 256"):
                        rule(1024, n=cols)
            elif (world_size, dtype_name) == (4, "float16"):
                # (4, float16, k_major), the measured SM-push ws4 M=512 point
                # (kmajor_ws4_fp16_m512_n2560; the sm_100a CTA-pair precedence
                # rule): exactly 512 padded rows at exactly N = 2560 run the
                # CTA-pair twin (385..511 pad onto the point; 513 pads to 640 and
                # leaves); the class has no other rectangle, no spill point and no
                # tile-order point, so every other call stays cg1.
                cls = (4, "float16", "k_major")
                assert b_layout == "k_major"
                assert _spill_point_form(arch, *cls, 512, 2560) is None
                assert loader._tile_order_point(arch, *cls, 512, 2560) is None
                assert [rule(rows) for rows in (385, 500, 511, 512)] == ["cg2"] * 4
                assert [
                    rule(rows)
                    for rows in (
                        1,
                        128,
                        384,
                        513,
                        640,
                        1024,
                        1025,
                        1536,
                        2048,
                        4096,
                        65536,
                    )
                ] == ["cg1"] * 11
                assert rule(512, n=2304) == "cg1" and rule(512, n=2816) == "cg1"
                assert rule(512, n=14336) == "cg1" and rule(1024, n=14336) == "cg1"
                assert rule(512, block_n=128) == "cg1"
                for cols in (2559, 2561):
                    with pytest.raises(ValueError, match="multiple of 256"):
                        rule(512, n=cols)
            elif world_size == 4:
                # (4, bfloat16, k_major): exactly 512 (the QKV M=512 point under the
                # sm_100a CTA-pair precedence rule: 385..511 pad onto it, 513 pads to
                # 640 and leaves; no spill point), 1024, 1152 or 2048 padded rows
                # at N = 2560 (897..1024, 1025..1152 and 1921..2048 pad onto the
                # points; M = 1000 is inside the 1024 point); the 1024 and 1152
                # points carry no spill point, the 2048 point is the QKV M=2048
                # spill point (llama_tp4_qkv_bf16_m2048_kmajor): the spill-wave
                # N-split pair twin of its form.
                cls = (4, "bfloat16", "k_major")
                assert (dtype_name, b_layout) == cls[1:]
                assert _spill_point_form(arch, *cls, 1024, 2560) is None
                assert _spill_point_form(arch, *cls, 1152, 2560) is None
                p_qkv = _spill_point_form(arch, *cls, 2048, 2560)
                assert p_qkv in loader.PAIR_FORMS[1:]
                # No tile-order point on sm_100a: the 1..128-row gate_up calls
                # keep the one-CTA program (asserted below with rows 1 / 128).
                assert loader._tile_order_point(arch, *cls, 128, 14336) is None
                assert rule(128, n=14336) == "cg1"
                assert [
                    rule(rows)
                    for rows in (897, 1000, 1024, 1025, 1100, 1152, 1921, 2048)
                ] == ["cg2"] * 6 + [p_qkv] * 2
                assert _spill_point_form(arch, *cls, 512, 2560) is None
                assert [rule(rows) for rows in (385, 500, 511, 512)] == ["cg2"] * 4
                assert [
                    rule(rows)
                    for rows in (1, 896, 1153, 1280, 1536, 1920, 2049, 4096, 65536)
                ] == ["cg1"] * 9
                assert rule(384) == "cg1" and rule(513) == "cg1" and rule(640) == "cg1"
                assert rule(1024, n=2304) == "cg1" and rule(1024, n=2816) == "cg1"
                assert rule(1024, n=5888) == "cg1" and rule(512, n=2560) == "cg2"
                assert rule(512, n=2304) == "cg1" and rule(512, n=2816) == "cg1"
                # ... and the gate_up points: exactly 512 or 1024 padded rows at
                # N = 14336 (385..512 and 897..1024 pad onto them; 513 pads to 640,
                # 1025 to 1152 and 2048 stays one-CTA). The 512 point is the spill
                # point (llama_tp4_gate_up m512): the spill-wave N-split pair twin
                # of its form; the 1024 point keeps the CTA-pair twin.
                p_gate = _spill_point_form(arch, *cls, 512, 14336)
                assert p_gate in loader.PAIR_FORMS[1:]
                assert [
                    rule(rows, n=14336)
                    for rows in (385, 500, 511, 512, 897, 1000, 1023, 1024)
                ] == [p_gate] * 4 + ["cg2"] * 4
                assert [
                    rule(rows, n=14336)
                    for rows in (
                        1,
                        384,
                        513,
                        640,
                        896,
                        1025,
                        1152,
                        1921,
                        2048,
                        2049,
                        4096,
                    )
                ] == ["cg1"] * 11
                assert rule(512, n=14080) == "cg1" and rule(512, n=14592) == "cg1"
                assert rule(1024, n=14080) == "cg1" and rule(1024, n=14592) == "cg1"
            else:
                # (8, bfloat16, k_major): exactly 512 or 1024 padded rows at
                # N = 7168 (385..512 and 897..1024 pad onto the points; 513 pads
                # to 640, 1025 to 1152); every N = 1280 call stays one-CTA. The
                # 512 point is the spill point (llama_tp8_gate_up m512): the
                # spill-wave N-split pair twin of its form; the 1024 point keeps
                # the CTA-pair twin. No last-pass N-split point on sm_100a: the
                # M=1025 QKV call (1152 padded rows at N = 1280) keeps the one-CTA
                # program.
                cls = (8, "bfloat16", "k_major")
                assert (dtype_name, b_layout) == cls[1:]
                p_ws8 = _spill_point_form(arch, *cls, 512, 7168)
                assert p_ws8 in loader.PAIR_FORMS[1:]
                assert loader._cg1s_point(arch, *cls, 1152, 1280) is None
                assert [rule(rows) for rows in (1, 512, 1024, 1025, 2048, 4096)] == [
                    "cg1"
                ] * 6
                assert [
                    rule(rows, n=7168)
                    for rows in (385, 500, 511, 512, 897, 1000, 1023, 1024)
                ] == [p_ws8] * 4 + ["cg2"] * 4
                assert [
                    rule(rows, n=7168)
                    for rows in (
                        1,
                        384,
                        513,
                        640,
                        896,
                        1025,
                        1152,
                        1921,
                        2048,
                        2049,
                        4096,
                    )
                ] == ["cg1"] * 11
                assert rule(512, n=6912) == "cg1" and rule(512, n=7424) == "cg1"
                assert rule(1024, n=6912) == "cg1" and rule(1024, n=7424) == "cg1"
            # The 128-wide program has no pair form, whatever the row count.
            assert rule(1024, block_n=128) == "cg1" and rule(2048, block_n=128) == "cg1"
            if world_size == 4:
                assert rule(512, n=14336, block_n=128) == "cg1"
            elif world_size == 8:
                assert rule(512, n=7168, block_n=128) == "cg1"
            continue
        assert arch == "sm_103a"
        if (world_size, b_layout) == (2, "n_major"):
            # (2, bfloat16, n_major): 2048..5760 padded rows and N <= 2048, plus the
            # M=1024 point (exactly 1024 padded rows at N = 2048; 897..1023 pad onto
            # it, 1025 pads to 1152 and leaves, 1920 padded rows stay one-CTA).
            assert [
                rule(rows)
                for rows in (1, 896, 1025, 1152, 1920, 5761, 6000, 8192, 65536)
            ] == ["cg1"] * 9
            assert [
                rule(rows)
                for rows in (897, 1000, 1023, 1024, 1921, 2048, 3000, 4096, 5760)
            ] == ["cg2"] * 9
            assert rule(2048, n=1792) == "cg2" and rule(2048, n=2304) == "cg1"
            assert rule(1024, n=1792) == "cg1" and rule(1024, n=2304) == "cg1"
        elif world_size == 2:
            # (2, bfloat16, k_major) and (2, float16, k_major): exactly 1024 padded
            # rows at exactly N = 2048 (897..1023 pad onto the point; 1025 pads to
            # 1152 and leaves); the classes have no other rectangle.
            assert dtype_name in ("bfloat16", "float16") and b_layout == "k_major"
            assert [rule(rows) for rows in (897, 1000, 1023, 1024)] == ["cg2"] * 4
            assert [
                rule(rows)
                for rows in (1, 896, 1025, 1152, 1921, 2048, 4096, 5760, 8192, 65536)
            ] == ["cg1"] * 10
            assert rule(1024, n=1792) == "cg1" and rule(1024, n=2304) == "cg1"
            for cols in (2047, 2049):
                with pytest.raises(ValueError, match="multiple of 256"):
                    rule(1024, n=cols)
        elif (world_size, dtype_name) == (4, "float16"):
            # (4, float16, k_major), the measured SM-push ws4 M=512 point
            # (kmajor_ws4_fp16_m512_n2560): exactly 512 padded rows at exactly
            # N = 2560 run the CTA-pair twin (385..511 pad onto the point; 513 pads
            # to 640 and leaves); the class has no other rectangle, no spill point
            # and no tile-order point, so every other call stays cg1.
            cls = (4, "float16", "k_major")
            assert b_layout == "k_major"
            assert _spill_point_form(arch, *cls, 512, 2560) is None
            assert loader._tile_order_point(arch, *cls, 512, 2560) is None
            assert [rule(rows) for rows in (385, 500, 511, 512)] == ["cg2"] * 4
            assert [
                rule(rows)
                for rows in (1, 128, 384, 513, 640, 1024, 1025, 1536, 2048, 4096, 65536)
            ] == ["cg1"] * 11
            assert rule(512, n=2304) == "cg1" and rule(512, n=2816) == "cg1"
            assert rule(512, n=14336) == "cg1" and rule(1024, n=14336) == "cg1"
            assert rule(512, block_n=128) == "cg1"
            for cols in (2559, 2561):
                with pytest.raises(ValueError, match="multiple of 256"):
                    rule(512, n=cols)
        elif world_size == 4:
            # (4, bfloat16, k_major) rectangle: 1024..1536 padded rows, N <= 6058;
            # every call of the rectangle runs the CTA-pair twin (the QKV M=1024
            # row at N = 2560 is no spill point: 897..1024 pad onto it and keep
            # the twin like 1025..1536); and the measured SM-push ws4 M=512 point
            # (llama_tp4_qkv_bf16_m512_kmajor): exactly 512 padded rows at exactly
            # N = 2560 run the twin too (385..511 pad onto the point; 513 pads to
            # 640 and leaves; no spill point there either).
            assert _spill_point_form(arch, 4, "bfloat16", "k_major", 1024, 2560) is None
            assert _spill_point_form(arch, 4, "bfloat16", "k_major", 512, 2560) is None
            assert [
                rule(rows) for rows in (1, 384, 513, 640, 896, 1537, 2048, 4096, 65536)
            ] == ["cg1"] * 9
            assert [
                rule(rows)
                for rows in (385, 500, 511, 512, 897, 1000, 1024, 1025, 1100, 1536)
            ] == ["cg2"] * 10
            assert rule(512, n=2304) == "cg1" and rule(512, n=2816) == "cg1"
            assert rule(1024, n=2304) == "cg2" and rule(1024, n=2816) == "cg2"
            assert (
                rule(1024, n=5888) == "cg2"
                and rule(1024, n=6144) == "cg1"
                and rule(1024, n=14336) == "cg1"
            )
            # The gate_up point (llama_tp4_gate_up m512): exactly 512 padded rows at
            # exactly N = 14336, the spill-wave N-split pair twin (385..511 pad to
            # 512); 513 pads to 640 (cg1); the M=2048 row (not routed) and the
            # neighbouring column multiples stay cg1.
            p_gate = _spill_point_form(arch, 4, "bfloat16", "k_major", 512, 14336)
            assert p_gate in loader.PAIR_FORMS[1:]
            gate_rows = (385, 500, 511, 512)
            assert [rule(rows, n=14336) for rows in gate_rows] == [p_gate] * 4
            assert [
                rule(rows, n=14336)
                for rows in (129, 256, 513, 640, 1024, 1536, 1921, 2048, 2049, 4096)
            ] == ["cg1"] * 10
            assert rule(512, n=14080) == "cg1" and rule(512, n=14592) == "cg1"
            assert rule(2048, n=14080) == "cg1" and rule(2048, n=14592) == "cg1"
            assert rule(512, n=14336, block_n=128) == "cg1"
            # The tile-order point (llama_tp4_gate_up m125): exactly 128 padded rows
            # at exactly N = 14336 run the ``po`` tile-order twin of the one-CTA
            # program (1..128 pad onto it; 129 pads to 256 and leaves); the
            # neighbouring column multiples and the 128-wide call stay cg1.
            p_order = loader._tile_order_point(
                arch, 4, "bfloat16", "k_major", 128, 14336
            )
            assert p_order == "po" and p_order in loader.ORDER_FORMS
            assert [rule(rows, n=14336) for rows in (1, 125, 128)] == ["po"] * 3
            assert rule(128, n=14080) == "cg1" and rule(128, n=14592) == "cg1"
            assert rule(125, n=14336, block_n=128) == "cg1"
            for cols in (14335, 14337):
                with pytest.raises(ValueError, match="multiple of 256"):
                    rule(512, n=cols)
        else:
            # (8, bfloat16, k_major), the gate_up point (llama_tp8_gate_up m512):
            # exactly 512 padded rows at exactly N = 7168, the spill-wave N-split
            # pair twin; the class has no other rectangle, so the M=2048 row (not
            # routed) and every N = 1280 call stay off the pair forms.
            p_ws8 = _spill_point_form(arch, 8, "bfloat16", "k_major", 512, 7168)
            assert p_ws8 in loader.PAIR_FORMS[1:]
            # The last-pass N-split point (llama_tp8_qkv m1025): exactly 1152
            # padded rows at exactly N = 1280 run the ``cg1s2`` twin of the
            # one-CTA program (1025..1152 pad onto it; 1024 and 1153 leave); the
            # neighbouring column multiples and the 128-wide call stay cg1.
            p_split = loader._cg1s_point(arch, 8, "bfloat16", "k_major", 1152, 1280)
            assert p_split == "cg1s2" and p_split in loader.ONE_CTA_SPLIT_FORMS
            assert [rule(rows) for rows in (1, 512, 1024, 1153, 2048, 4096)] == [
                "cg1"
            ] * 6
            assert [rule(rows) for rows in (1025, 1100, 1152)] == ["cg1s2"] * 3
            assert rule(1152, n=1024) == "cg1" and rule(1152, n=1536) == "cg1"
            assert rule(1152, block_n=128) == "cg1"
            assert [rule(rows, n=7168) for rows in (500, 511, 512)] == [p_ws8] * 3
            assert [
                rule(rows, n=7168)
                for rows in (1, 513, 640, 1024, 1025, 1536, 1921, 2048, 2049, 4096)
            ] == ["cg1"] * 10
            assert rule(512, n=6912) == "cg1" and rule(512, n=7424) == "cg1"
            assert rule(2048, n=6912) == "cg1" and rule(2048, n=7424) == "cg1"
            assert rule(512, n=7168, block_n=128) == "cg1"
            for cols in (7167, 7169):
                with pytest.raises(ValueError, match="multiple of 256"):
                    rule(512, n=cols)
        # The 128-wide program has no pair form, whatever the row count.
        assert rule(1024, block_n=128) == "cg1" and rule(2048, block_n=128) == "cg1"
    with pytest.raises(ValueError, match="sm_90a"):
        loader.main_form(
            1024,
            2048,
            world_size=2,
            arch="sm_90a",
            dtype_name="bfloat16",
            b_layout="n_major",
            block_n=256,
        )
    with pytest.raises(ValueError, match="multiple of 256"):
        loader.main_form(
            1024,
            1152,
            world_size=2,
            arch=arch,
            dtype_name="bfloat16",
            b_layout="n_major",
            block_n=256,
        )
    with pytest.raises(ValueError, match="block_n"):
        loader.main_form(
            1024,
            2048,
            world_size=2,
            arch=arch,
            dtype_name="bfloat16",
            b_layout="n_major",
            block_n=64,
        )


def test_n128_classes_are_exactly_the_classes_of_the_routed_inventory_rows():
    for arch in loader.ARCH_FLAGS:
        routed = {
            tuple(shape[2:])
            for _label, shape, widths in _INVENTORY_ROWS
            if widths.get(arch) == 128
        }
        assert _n128_classes(arch) == routed, arch
    # Six routed rows define four classes on sm_100a (the CTA-pair precedence
    # rule: the two ws4 M=512 N=2560 k_major rows keep 256 under the pair
    # precedence, so the fp16 ws4 k_major class delivers no twin); the five
    # one-wave rows (three M=125 rows, two ws8 512-row coverage rows) define
    # three on sm_103a (a subset of the sm_100a classes).
    assert len(_n128_classes("sm_100a")) == 4
    assert len(_n128_classes("sm_103a")) == 3
    assert _n128_classes("sm_103a") < _n128_classes("sm_100a")


@pytest.mark.parametrize("arch", _BOTH)
def test_main_block_n_selects_the_twin_only_inside_the_table_and_the_wave_cap(arch):
    for world_size in (4, 8):
        n = 2560 if world_size == 4 else 1280
        for dtype_name in loader.SUPPORTED_DTYPES.values():
            for b_layout in loader.B_LAYOUTS:
                delivered = (world_size, dtype_name, b_layout) in _n128_classes(arch)
                cls = {"dtype_name": dtype_name, "b_layout": b_layout}
                # 40 N256 tiles (one wave): the rule qualifies every class, the
                # table alone decides.
                assert (
                    _frozen_rule(125, n, world_size=world_size, arch=arch, **cls) == 128
                )
                assert loader.main_block_n(
                    125,
                    n,
                    world_size=world_size,
                    arch=arch,
                    dtype_name=dtype_name,
                    b_layout=b_layout,
                ) == (128 if delivered else 256), (world_size, dtype_name, b_layout)
                # 160 N256 tiles (two waves, three on the twin): within the
                # sm_100a cap of two (the table decides), beyond the sm_103a
                # cap of one (256 whatever the table says).
                # The sm_100a CTA-pair precedence rule: inside the (4, bf16 | fp16,
                # k_major) 512 x 2560 rectangles on sm_100a the pair takes precedence
                # over the wave rule -- the frozen rule and the loader keep 256.
                pair_point = (arch, world_size, b_layout) == ("sm_100a", 4, "k_major")
                two_waves = 128 if arch == "sm_100a" else 256
                assert _frozen_rule(
                    512, n, world_size=world_size, arch=arch, **cls
                ) == (256 if pair_point else two_waves)
                assert loader.main_block_n(
                    512,
                    n,
                    world_size=world_size,
                    arch=arch,
                    dtype_name=dtype_name,
                    b_layout=b_layout,
                ) == (two_waves if delivered and not pair_point else 256)
                # 320 N256 tiles (three waves): beyond the cap for every class.
                assert (
                    loader.main_block_n(
                        1024,
                        n,
                        world_size=world_size,
                        arch=arch,
                        dtype_name=dtype_name,
                        b_layout=b_layout,
                    )
                    == 256
                )
    # ws2 always runs the N256 program, whatever the class and shape.
    for dtype_name in loader.SUPPORTED_DTYPES.values():
        for b_layout in loader.B_LAYOUTS:
            assert (
                loader.main_block_n(
                    125,
                    1280,
                    world_size=2,
                    arch=arch,
                    dtype_name=dtype_name,
                    b_layout=b_layout,
                )
                == 256
            )


def test_main_block_n_keeps_a_class_without_a_delivered_twin_on_the_n256_program():
    # (4, bfloat16, n_major) has no routed inventory row on either architecture
    # (its narrowest row, ws4_packed_qkv_bf16_m512_n2560, is sm_103a-only and
    # spans two N256 waves, beyond the sm_103a cap of one): the two-wave call
    # stays at 256 on both architectures although the sm_100a rule alone would
    # select the twin.
    cls = {"dtype_name": "bfloat16", "b_layout": "n_major"}
    assert _frozen_rule(512, 2560, world_size=4, arch="sm_100a", **cls) == 128
    assert _frozen_rule(512, 2560, world_size=4, arch="sm_103a", **cls) == 256
    for arch in _BOTH:
        assert (4, "bfloat16", "n_major") not in _n128_classes(arch)
    widths = [
        loader.main_block_n(
            512,
            2560,
            world_size=4,
            arch=arch,
            dtype_name="bfloat16",
            b_layout="n_major",
        )
        for arch in _BOTH
    ]
    assert widths == [256, 256]
    # (8, float16, k_major) is routed on sm_100a only (kmajor_ws8_fp16_m512_n1280,
    # two waves): the one-wave call qualifies on both architectures but takes
    # the twin only where the class delivers it.
    fp16_k = {"dtype_name": "float16", "b_layout": "k_major"}
    for arch in _BOTH:
        assert _frozen_rule(125, 1280, world_size=8, arch=arch, **fp16_k) == 128
    assert (8, "float16", "k_major") in _n128_classes("sm_100a")
    assert (8, "float16", "k_major") not in _n128_classes("sm_103a")
    widths = [
        loader.main_block_n(
            125,
            1280,
            world_size=8,
            arch=arch,
            dtype_name="float16",
            b_layout="k_major",
        )
        for arch in _BOTH
    ]
    assert widths == [128, 256]


def test_main_block_n_validates_the_architecture_before_the_world_size_shortcut():
    for world_size in (2, 4, 8):
        with pytest.raises(ValueError, match="sm_90a"):
            loader.main_block_n(
                125,
                1280,
                world_size=world_size,
                arch="sm_90a",
                dtype_name="bfloat16",
                b_layout="n_major",
            )
    with pytest.raises(ValueError, match="multiple of 256"):
        loader.main_block_n(
            125,
            1152,
            world_size=8,
            arch="sm_100a",
            dtype_name="bfloat16",
            b_layout="n_major",
        )


@pytest.mark.parametrize("arch", _BOTH)
@pytest.mark.parametrize(
    ("world_size", "rows", "cols", "expected"),
    [
        (8, 125, 1280, True),
        (8, 125, 7168, True),  # 128 padded rows serve N up to 10240
        (8, 125, 10240, True),
        (8, 125, 10496, False),  # one tile too wide
        (8, 512, 1280, True),
        (8, 512, 4096, True),  # 512 padded rows serve N up to 4096
        (8, 512, 7168, False),  # the GEMM would wait behind the SM push: copy engines
        (8, 513, 1280, False),  # pads to 640 rows
        (8, 1024, 2048, False),  # beyond the row cap; wider than the band (N <= 1619)
        (8, 1025, 1280, False),  # pads to 1152 rows: no band
        (4, 500, 2560, True),
        (4, 125, 14336, False),
        (4, 512, 14336, False),
        (4, 256, 8192, True),  # 256 padded rows serve N up to 8192
        (4, 256, 8448, False),
        (4, 1024, 1280, False),  # the band is an eight-rank band
        (4, 1025, 2048, False),
        (2, 125, 1280, False),  # a single peer: the copy engine wins
        (2, 512, 2048, False),
    ],
)
def test_sm_push_serves_at_most_512_padded_rows_within_the_width_ceiling_at_ws4_and_ws8(
    arch, world_size, rows, cols, expected
):
    """The base rule is the same on both architectures; the rows outside every band keep it."""

    assert (
        loader.uses_sm_push(rows=rows, world_size=world_size, cols=cols, arch=arch)
        is expected
    )
    assert loader.SM_PUSH_MAX_ROWS == 512
    assert (loader.SM_PUSH_COLS_INTERCEPT, loader.SM_PUSH_COLS_PER_ROW) == (12288, 16)
    assert [loader.sm_push_max_cols(rows) for rows in (128, 256, 384, 512)] == [
        10240,
        8192,
        6144,
        4096,
    ]


def test_sm_push_bands_route_the_eight_rank_1024_row_calls_up_to_n_1619():
    """``SM_PUSH_BANDS_BY_ARCH`` adds, on both architectures, the eight-rank calls of exactly
    1024 padded rows with ``N <= 1619`` (the ``llama_tp8_qkv_bf16_m1024_kmajor`` row, N=1280);
    the N=2048 row of that row count, the 1025-row call (1152 padded) and every four-rank call
    keep the copy engines. An architecture outside the table is an error."""

    assert loader.sm_push_bands("sm_103a") == ((8, 1024, 1619),)
    assert loader.sm_push_bands("sm_100a") == ((8, 1024, 1619),)
    for rows in (897, 1000, 1024):  # every M padding to 1024 rows
        assert (
            loader.uses_sm_push(rows=rows, world_size=8, cols=1280, arch="sm_103a")
            is True
        )
        assert (
            loader.uses_sm_push(rows=rows, world_size=8, cols=1536, arch="sm_103a")
            is True
        )
        assert (
            loader.uses_sm_push(rows=rows, world_size=8, cols=2048, arch="sm_103a")
            is False
        )
        assert (
            loader.uses_sm_push(rows=rows, world_size=8, cols=1280, arch="sm_100a")
            is True
        )
        assert (
            loader.uses_sm_push(rows=rows, world_size=8, cols=2048, arch="sm_100a")
            is False
        )
        assert (
            loader.uses_sm_push(rows=rows, world_size=4, cols=1280, arch="sm_103a")
            is False
        )
    for arch in _BOTH:
        assert (
            loader.uses_sm_push(rows=896, world_size=8, cols=1280, arch=arch) is False
        )
        assert (
            loader.uses_sm_push(rows=1025, world_size=8, cols=1280, arch=arch) is False
        )
    # The band row's per-peer payload (1024 x 8192 x 2 B = 16 MiB) selects the wide
    # push program; on sm_103a its launch row raises the CTAs per peer from the
    # rule's 8 to 16, on sm_100a (the band, no launch row) the payload rule keeps 8 x 8.
    assert loader.push_payload_bytes(1024, 2) == 16 * 1024 * 1024
    assert loader.push_launch(1024, 2, world_size=8, arch="sm_103a") == (
        "peer_push_wide",
        16,
        loader.PUSH_WIDE_UNROLL,
    )
    assert loader.push_launch(1024, 2, world_size=8, arch="sm_100a") == (
        "peer_push_wide",
        loader.PUSH_WIDE_CTAS_PER_PEER,
        loader.PUSH_WIDE_UNROLL,
    )
    for arch in ("sm_90a", "sm_120a", ""):
        with pytest.raises(ValueError, match="no SM push bands"):
            loader.uses_sm_push(rows=125, world_size=8, cols=1280, arch=arch)
        with pytest.raises(ValueError, match="no SM push bands"):
            loader.sm_push_reachable(world_size=8, cols=1280, arch=arch)


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_barrier_flag_pad_has_two_epochs_and_two_mailbox_banks_per_phase(world_size):
    assert loader.barrier_flag_words(world_size) == 2 + 4 * world_size


def test_spec_names_carry_the_exact_architecture(monkeypatch):
    # Build-target discovery reads FLASHINFER_CUDA_ARCH_LIST when set; pin it so
    # the test does not depend on the visible devices of the test host.
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    loader.spec.cache_clear()
    try:
        specs = {}
        for arch in ("sm_100a", "sm_103a"):
            sequence = loader.sequence_name(2, "bfloat16", "n_major", arch)
            spec = loader.spec(sequence, arch)
            assert spec.name == f"{loader.SOURCE_PACKAGE}_sequence_{sequence}"
            assert sequence.endswith(arch)
            assert loader.ARCH_FLAGS[arch][0] in spec.extra_cuda_cflags
            assert [str(path).split("/")[-1] for path in spec.sources] == [
                source.split("/")[-1]
                for source in loader.SEQUENCES[sequence]["sources"]
            ]
            specs[arch] = spec
        assert specs["sm_100a"] is not specs["sm_103a"]
    finally:
        loader.spec.cache_clear()


@pytest.mark.parametrize("n", [256, 1280, 2048, 7168, 14336])
def test_weight_layout_is_classified_from_strides_without_copies(n):
    n_major = torch.empty(8192, n, dtype=torch.bfloat16, device="meta")
    k_major = torch.empty(n, 8192, dtype=torch.bfloat16, device="meta").t()
    assert loader.weight_layout(n_major) == "n_major"
    assert loader.weight_layout(k_major) == "k_major"
    assert loader.weight_tma_source(n_major, "n_major").shape == (1, 8192, n)
    assert loader.weight_tma_source(k_major, "k_major").shape == (1, n, 8192)
    assert loader.weight_tma_source(k_major, "k_major").is_contiguous()


def test_weight_layout_rejects_other_stride_patterns():
    padded = torch.empty(8192, 4096, dtype=torch.bfloat16, device="meta")[:, :2048]
    with pytest.raises(ValueError, match="strides"):
        loader.weight_layout(padded)
    with pytest.raises(ValueError, match=r"\[8192, N\]"):
        loader.weight_layout(
            torch.empty(4096, 2048, dtype=torch.bfloat16, device="meta")
        )


def _fake_group(world_size, rank, name="fake_group"):
    return SimpleNamespace(group_name=name, _world_size=world_size, _rank=rank)


def _patch_distributed(
    monkeypatch,
    *,
    world_size,
    rank,
    backend_name="nccl",
    arch="sm_100a",
    symm_backend="CUDA",
):
    monkeypatch.setattr(backend.dist, "is_available", lambda: True)
    monkeypatch.setattr(backend.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(backend.dist, "get_backend", lambda group: backend_name)
    monkeypatch.setattr(backend.dist, "get_world_size", lambda group: group._world_size)
    monkeypatch.setattr(backend.dist, "get_rank", lambda group: group._rank)
    monkeypatch.setattr(backend.symm_mem, "get_backend", lambda device: symm_backend)
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability={"sm_100a": (10, 0), "sm_103a": (10, 3)}[arch],
            arch=arch,
            sm_count=148,
        ),
    )


def _meta_pair(rows, n, dtype=torch.bfloat16):
    inp = torch.empty(rows, 8192, dtype=dtype, device="meta")
    w = torch.empty(8192, n, dtype=dtype, device="meta")
    return inp, w


class _CudaLike:
    """Tensor metadata stand-in: shape, strides, dtype and a CUDA device without a GPU."""

    def __init__(self, shape, dtype, *, contiguous=True, index=0, stride=None):
        self._shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device("cuda", index)
        self._contiguous = contiguous
        if stride is None:
            stride = []
            acc = 1
            for dim in reversed(self._shape):
                stride.append(acc)
                acc *= dim
            stride = tuple(reversed(stride))
        self._stride = tuple(stride)

    @property
    def shape(self):
        return self._shape

    @property
    def ndim(self):
        return len(self._shape)

    def stride(self):
        return self._stride

    def is_contiguous(self):
        return self._contiguous

    def t(self):
        return _CudaLike(
            self._shape[::-1],
            self.dtype,
            contiguous=self._contiguous and self.ndim < 2,
            index=self.device.index,
            stride=self._stride[::-1],
        )

    def view(self, *shape):
        # Metadata-only reshape of a dense (possibly transposed-to-dense) view.
        return _CudaLike(shape, self.dtype, index=self.device.index)


def _k_major_weight(n, dtype=torch.bfloat16):
    return _CudaLike((8192, n), dtype, contiguous=False, stride=(1, 8192))


@pytest.mark.parametrize("world_size", [2, 4, 8])
@pytest.mark.parametrize("rows", [1, 125, 512, 1025, 16384])
@pytest.mark.parametrize("n", [256, 1280, 2048, 7168, 14336])
def test_validation_admits_any_rows_and_any_width_multiple_of_256_in_both_layouts(
    monkeypatch, world_size, rows, n
):
    _patch_distributed(monkeypatch, world_size=world_size, rank=0)
    for weight, layout in (
        (_CudaLike((8192, n), torch.bfloat16), "n_major"),
        (_k_major_weight(n), "k_major"),
    ):
        call = backend._validate(
            _CudaLike((rows, 8192), torch.bfloat16), weight, _fake_group(world_size, 0)
        )
        assert (call.world_size, call.rows, call.n, call.b_layout, call.arch) == (
            world_size,
            rows,
            n,
            layout,
            "sm_100a",
        )


@pytest.mark.parametrize("n", [128, 1000, 1281, 2049])
def test_validation_rejects_widths_that_are_not_a_multiple_of_256(monkeypatch, n):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="multiple of 256"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, n), torch.bfloat16),
            _fake_group(2, 0),
        )


def test_validation_rejects_strided_inputs_and_padded_weights_instead_of_copying(
    monkeypatch,
):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="contiguous"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16, contiguous=False),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )
    with pytest.raises(ValueError, match="strides"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16, contiguous=False, stride=(4096, 1)),
            _fake_group(2, 0),
        )


def test_validation_rejects_wrong_k_and_empty_inputs(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    with pytest.raises(ValueError, match="K=8192"):
        backend._validate(
            _CudaLike((256, 4096), torch.bfloat16),
            _CudaLike((4096, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )
    with pytest.raises(ValueError, match="positive"):
        backend._validate(
            _CudaLike((0, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )


def test_validation_rejects_unsupported_world_sizes_and_backends(monkeypatch):
    _patch_distributed(monkeypatch, world_size=3, rank=0)
    with pytest.raises(ValueError, match="world size 2, 4, or 8"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(3, 0),
        )
    _patch_distributed(monkeypatch, world_size=2, rank=0, backend_name="gloo")
    with pytest.raises(ValueError, match="NCCL"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )


@pytest.mark.parametrize("symm_backend", ["CUDA", "NVSHMEM", "cuda", "nvshmem"])
def test_validation_admits_the_cuda_and_nvshmem_symmetric_memory_backends(
    monkeypatch, symm_backend
):
    _patch_distributed(monkeypatch, world_size=2, rank=0, symm_backend=symm_backend)
    device = torch.device("cuda", 0)
    assert backend.symmetric_memory_backend(device) == symm_backend.upper()
    assert backend.supports_symmetric_memory_backend(device)
    call = backend._validate(
        _CudaLike((256, 8192), torch.bfloat16),
        _CudaLike((8192, 2048), torch.bfloat16),
        _fake_group(2, 0),
    )
    assert call.rows == 256


@pytest.mark.parametrize("symm_backend", ["NCCL", None, "XPU"])
def test_validation_rejects_other_symmetric_memory_backends_without_selecting_one(
    monkeypatch, symm_backend
):
    _patch_distributed(monkeypatch, world_size=2, rank=0, symm_backend=symm_backend)
    selected = []
    monkeypatch.setattr(backend.symm_mem, "set_backend", selected.append)
    device = torch.device("cuda", 0)
    assert not backend.supports_symmetric_memory_backend(device)
    with pytest.raises(ValueError, match=r"\['CUDA', 'NVSHMEM'\]"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )
    assert selected == []


def test_capture_sentinel_is_above_every_eager_epoch():
    assert backend._CAPTURE_READY_TARGET == 2**32 - 1
    state = backend._LaunchState(rank=0, world_size=2)
    assert state.ready_epoch < backend._CAPTURE_READY_TARGET - 1


def test_captured_call_clears_only_its_readiness_words_with_a_stream_memset(
    monkeypatch,
):
    """The clear is a ``memset32_`` on the pad's leading words (no kernel) on the
    caller's current stream, so a captured call issues exactly the eager kernel
    sequence."""
    calls = []
    monkeypatch.setattr(
        backend.torch.ops.symm_mem,
        "memset32_",
        lambda pad, offset, val, count: calls.append((pad, offset, val, count)),
    )
    pad = torch.zeros(4 * 8, dtype=torch.uint32)
    workspace = backend._Workspace(
        dtype=torch.bfloat16,
        rank=0,
        world_size=4,
        device_index=0,
        arch="sm_100a",
        signal_pad=pad,
    )
    backend._clear_readiness(workspace, 4 * 3)
    assert calls == [(pad, 0, 0, 12)]


def test_prepare_session_refuses_to_allocate_or_grow_inside_capture(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 256, 2048, "n_major")
    with pytest.raises(RuntimeError, match="inside CUDA-graph capture"):
        backend._prepare_session(
            call, _fake_group(2, 0, name="g"), torch.bfloat16, capacity_rows=256
        )


def test_workspace_materializes_the_sm_push_tables_at_prepare_time():
    """The copy-engine route's peer tables are address tuples built when the
    workspace grows; ``materialize`` resolves the SM push route's device tables,
    and only when that route is reachable for the prepared group."""

    workspace = backend._Workspace(
        dtype=torch.bfloat16, rank=1, world_size=4, device_index=0, arch="sm_100a"
    )
    resolved = []
    workspace.push_tables = lambda: resolved.append(True)
    workspace.materialize(sm_push=False)
    assert resolved == []
    workspace.materialize(sm_push=True)
    assert resolved == [True]


def test_sm_push_tables_are_reachable_exactly_when_some_row_count_takes_that_route():
    """``_prepare_session`` materializes by ``sm_push_reachable``: the widest ceiling of the base
    rule (128 padded rows) or a band of the device's architecture; the band (N <= 1619 at eight
    ranks, on both architectures) is already reachable through the base rule at 128 rows, so the
    predicate agrees with the 128-row probe on both architectures for every group."""

    reachable = {
        (2, 1280): False,
        (4, 2560): True,
        (4, 10240): True,
        (4, 10496): False,
        (8, 1280): True,
        (8, 1619): True,
        (8, 14336): False,
    }
    for arch in _BOTH:
        for (world_size, cols), expected in reachable.items():
            assert (
                loader.sm_push_reachable(world_size=world_size, cols=cols, arch=arch)
                is expected
            ), (arch, world_size, cols)
            assert loader.sm_push_reachable(
                world_size=world_size, cols=cols, arch=arch
            ) is (
                loader.uses_sm_push(
                    rows=loader.BLOCK_M, world_size=world_size, cols=cols, arch=arch
                )
                or any(
                    world_size == band_ws and cols <= band_cols
                    for band_ws, _band_rows, band_cols in loader.sm_push_bands(arch)
                )
            )


def _published_workspace(push_buffers):
    """A workspace one generation old, as ``_publish_scratch`` finds it at a growth (CPU stand-ins)."""

    workspace = backend._Workspace(
        dtype=torch.bfloat16,
        rank=0,
        world_size=2,
        device_index=0,
        arch="sm_100a",
    )
    workspace.pitch = 512
    workspace.scratch = torch.empty(2, 512, 8, dtype=torch.bfloat16)
    workspace.scratch_handle = object()
    workspace.signal_pad = torch.zeros(8, dtype=torch.uint32)
    workspace.peer_scratch = ("old peer slice",)
    workspace.push_buffers = push_buffers
    workspace.comm_stream = types.SimpleNamespace(cuda_stream=7)
    return workspace


class _GrowthHandle:
    """Rendezvous handle of a grown scratch: CPU readiness pads and remote views."""

    def get_signal_pad(self, peer, shape, dtype, offset):
        return torch.zeros(shape, dtype=dtype)

    def get_remote_tensor(self, peer, shape, dtype):
        return torch.empty(shape, dtype=dtype)


def test_growth_resolves_the_sm_push_tables_again_when_they_existed():
    """A launcher prepared before the growth keeps both routes: the SM push
    tables address the scratch generation, so they are resolved again."""

    workspace = _published_workspace(push_buffers=("payload", "signals", "counters"))
    resolved = []
    workspace.push_tables = lambda: resolved.append(True)
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 1024, 2048, "n_major")
    grown = torch.empty(2, 1024, 8, dtype=torch.bfloat16)
    backend._publish_scratch(workspace, call, 1024, grown, _GrowthHandle())
    assert workspace.pitch == 1024 and workspace.scratch is grown
    assert workspace.max_chunks == loader.chunk_plan(1024)[2]
    assert resolved == [True]


def test_growth_without_push_tables_resolves_none():
    workspace = _published_workspace(push_buffers=None)
    resolved = []
    workspace.push_tables = lambda: resolved.append(True)
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 1024, 2048, "n_major")
    grown = torch.empty(2, 1024, 8, dtype=torch.bfloat16)
    backend._publish_scratch(workspace, call, 1024, grown, _GrowthHandle())
    assert workspace.push_buffers is None and resolved == []


def test_prepare_session_refuses_to_grow_a_captured_workspace_before_any_collective(
    monkeypatch,
):
    """A captured graph bakes this generation's addresses on every rank: a larger
    call raises a named error before any synchronize or collective, without
    poisoning the state."""

    call, group, stream = _prepare_fixture(monkeypatch, "captured_growth")
    state = backend._launch_state(call, group)
    state.flags = torch.zeros(4, dtype=torch.uint32)
    workspace = backend._workspace(call, group, torch.bfloat16)
    workspace.pitch = 512
    workspace.captured = True

    def rendezvous(*args, **kwargs):
        raise AssertionError("no collective may run for a refused growth")

    monkeypatch.setattr(backend, "_rendezvous", rendezvous)
    with pytest.raises(RuntimeError, match="captured into a CUDA graph"):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=1024)
    assert stream.synchronized == 0
    assert state.poisoned is False and workspace.pitch == 512


def test_prepare_session_refuses_to_build_the_host_sequence_inside_capture(
    monkeypatch,
):
    call, group, stream = _prepare_fixture(monkeypatch, "jit_capture")
    state = backend._launch_state(call, group)
    state.flags = torch.zeros(4, dtype=torch.uint32)
    workspace = backend._workspace(call, group, torch.bfloat16)
    workspace.pitch = 512
    workspace.sequences = {}
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)

    def load(*args):
        raise AssertionError("no JIT build may run inside capture")

    monkeypatch.setattr(backend.loader, "load", load)
    with pytest.raises(RuntimeError, match="host sequence inside CUDA-graph capture"):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
    assert stream.synchronized == 0 and state.poisoned is False


def test_prepare_session_refuses_to_resolve_push_tables_inside_capture(monkeypatch):
    """Resolving the SM push tables allocates; under capture that is a named refusal."""

    _patch_distributed(monkeypatch, world_size=8, rank=0)
    call = backend._Call(0, 0, 8, "g8", "bfloat16", "sm_103a", 512, 1280, "n_major")
    group = _fake_group(8, 0, name="g8")
    state = backend._launch_state(call, group)
    state.flags = torch.zeros(16, dtype=torch.uint32)
    workspace = backend._workspace(call, group, torch.bfloat16)
    workspace.pitch = 512
    workspace.sequences = {"n_major": object()}
    workspace.push_buffers = None
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="SM push route's device tables inside"):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
    assert state.poisoned is False and workspace.push_buffers is None


def test_validation_rejects_devices_outside_the_exported_architectures(monkeypatch):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(capability=(9, 0), arch=None, sm_count=132),
    )
    with pytest.raises(ValueError, match="SM100 or SM103"):
        backend._validate(
            _CudaLike((256, 8192), torch.bfloat16),
            _CudaLike((8192, 2048), torch.bfloat16),
            _fake_group(2, 0),
        )


def test_out_must_match_the_gathered_output_contract():
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 250, 2048, "n_major")
    like = _CudaLike((250, 8192), torch.bfloat16)
    with pytest.raises(ValueError, match=r"\[500, 2048\]"):
        backend._output(_CudaLike((250, 2048), torch.bfloat16), call, like)
    with pytest.raises(ValueError, match="dtype"):
        backend._output(_CudaLike((500, 2048), torch.float16), call, like)
    out = _CudaLike((500, 2048), torch.bfloat16)
    assert backend._output(out, call, like) is out


def test_backend_entrypoint_rejects_non_cake_route():
    inp, w = _meta_pair(256, 2048)
    with pytest.raises(ValueError, match="exactly 'cake'"):
        backend.all_gather_matmul_cake(inp, w, _fake_group(2, 0), backend="auto")


def test_public_entrypoint_routes_backend_cake_to_the_backend(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )
    calls = []
    result = object()

    def fake_backend(inp, w, group, *, backend, verbose):
        calls.append((inp, w, group, backend, verbose))
        return result

    monkeypatch.setattr(backend, "all_gather_matmul_cake", fake_backend)
    inp, w, group = object(), object(), object()
    assert dispatcher.all_gather_matmul(inp, w, group, backend="cake") is result
    assert calls == [(inp, w, group, "cake", False)]


def test_public_prepare_forwards_the_row_capacity(monkeypatch):
    dispatcher = importlib.import_module(
        "flashinfer.comm.all_gather_matmul.all_gather_matmul"
    )
    calls = []
    launcher = object()

    def fake_prepare(inp, w, group, *, max_rows, verbose):
        calls.append((inp, w, group, max_rows, verbose))
        return launcher

    monkeypatch.setattr(backend, "_prepare_all_gather_matmul_cake", fake_prepare)
    inp, w, group = object(), object(), object()
    assert dispatcher.prepare_all_gather_matmul(inp, w, group) is launcher
    assert (
        dispatcher.prepare_all_gather_matmul(
            inp, w, group, backend="cake", max_rows=2048
        )
        is launcher
    )
    assert calls == [(inp, w, group, None, False), (inp, w, group, 2048, False)]
    with pytest.raises(ValueError, match="'auto' or 'cake'"):
        dispatcher.prepare_all_gather_matmul(inp, w, group, backend="cutile")


def test_prepared_launcher_serves_every_row_count_up_to_its_capacity(monkeypatch):
    _patch_distributed(monkeypatch, world_size=8, rank=3, arch="sm_103a")
    weight = _k_major_weight(1280)
    weight.data_ptr = lambda: 0x1000
    sample = _CudaLike((512, 8192), torch.bfloat16)
    call = backend._validate(sample, weight, _fake_group(8, 3))
    state = backend._LaunchState(rank=3, world_size=8)
    workspace = backend._Workspace(
        dtype=torch.bfloat16,
        rank=3,
        world_size=8,
        device_index=0,
        arch="sm_103a",
        pitch=2048,
    )
    group = _fake_group(8, 3)
    launcher = backend._PreparedLauncher(
        group=group,
        group_id=id(group),
        call=call,
        max_rows=2048,
        device=sample.device,
        dtype=sample.dtype,
        weight=weight,
        weight_fingerprint=backend._fingerprint(weight),
        state=state,
        workspace=workspace,
    )
    # The frozen launcher resolves the weight's tensor-map source view once.
    assert launcher.weight_source is not None
    assert tuple(launcher.weight_source.shape) == (1, 1280, 8192)
    for rows in (1, 125, 512, 1025, 2048):
        bound = launcher._validate_input(_CudaLike((rows, 8192), torch.bfloat16))
        assert (bound.rows, bound.n, bound.b_layout) == (rows, 1280, "k_major")
    with pytest.raises(ValueError, match=r"\[1, 2048\]"):
        launcher._validate_input(_CudaLike((2049, 8192), torch.bfloat16))
    with pytest.raises(ValueError, match="contiguous"):
        launcher._validate_input(
            _CudaLike((512, 8192), torch.bfloat16, contiguous=False)
        )
    with pytest.raises(ValueError, match="dtype"):
        launcher._validate_input(_CudaLike((512, 8192), torch.float16))
    with pytest.raises(ValueError, match=r"\[M, 8192\]"):
        launcher._validate_input(_CudaLike((512, 4096), torch.bfloat16))


def test_launch_refuses_a_workspace_smaller_than_the_padded_rows():
    state = backend._LaunchState(rank=0, world_size=2)
    workspace = backend._Workspace(
        dtype=torch.bfloat16,
        rank=0,
        world_size=2,
        device_index=0,
        arch="sm_100a",
        pitch=128,
    )
    call = backend._Call(0, 0, 2, "g", "bfloat16", "sm_100a", 129, 2048, "n_major")
    with pytest.raises(
        RuntimeError, match="holds 128 rows per peer, the call needs 256"
    ):
        backend._launch(state, workspace, call, None, None, None)
    # A capacity shortfall is a caller error, not a failed collective.
    assert state.poisoned is False


def test_cross_stream_join_records_a_fresh_event_and_skips_capture(monkeypatch):
    """The join must never reuse an event that a CUDA graph capture re-recorded."""

    assert not hasattr(backend._LaunchState, "tail_event")

    class _Stream:
        def __init__(self, handle):
            self.cuda_stream = handle
            self.joined = []

        def wait_stream(self, other):
            self.joined.append(other)

    first, second, third = _Stream(11), _Stream(22), _Stream(33)
    current = {11: first, 22: second, 33: third}
    monkeypatch.setattr(
        backend.torch.cuda, "current_stream", lambda index: current[handles[-1]]
    )
    handles = [11]
    state = backend._LaunchState(rank=0, world_size=2)
    assert backend._join_previous_tail(state, 11, 0, capturing=False) is False
    backend._remember_tail(state, 11, 0)
    assert state.tail_stream is first and state.tail_handle == 11
    assert backend._join_previous_tail(state, 11, 0, capturing=False) is False
    handles.append(22)
    assert backend._join_previous_tail(state, 22, 0, capturing=True) is False
    assert backend._join_previous_tail(state, 22, 0, capturing=False) is True
    assert second.joined == [first] and first.joined == []
    backend._remember_tail(state, 22, 0)
    assert state.tail_stream is second
    handles.append(33)
    assert backend._join_previous_tail(state, 33, 0, capturing=True) is False
    assert third.joined == []
    # Re-remembering the same stream keeps the cached torch stream object.
    backend._remember_tail(state, 22, 0)
    assert state.tail_stream is second


class _Recording:
    def __init__(self, *, index=0):
        self.device = torch.device("cuda", index)
        self.streams = []

    def record_stream(self, stream):
        self.streams.append(stream)


def _launch_fixture(
    monkeypatch,
    *,
    rows,
    n,
    world_size,
    arch,
    b_layout="n_major",
    dtype_name="bfloat16",
):
    call = backend._Call(0, 1, world_size, "g", dtype_name, arch, rows, n, b_layout)
    state = backend._LaunchState(rank=1, world_size=world_size, flag_peers=("flags",))
    workspace = backend._Workspace(
        dtype={"bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name],
        rank=1,
        world_size=world_size,
        device_index=0,
        arch=arch,
        pitch=max(2048, backend.loader.padded_rows(rows)),
    )
    # The launch reads the device facts once (the SM count bounds the grid);
    # pin them to the call's architecture so no GPU is queried and the
    # delivered-class table of that architecture decides the tile width.
    monkeypatch.setattr(
        loader,
        "device_facts",
        lambda index: loader.DeviceFacts(
            capability={"sm_100a": (10, 0), "sm_103a": (10, 3)}[arch],
            arch=arch,
            sm_count=148,
        ),
    )
    workspace.scratch = "scratch"
    workspace.signal_pad = "signal_pad"
    workspace.comm_stream = "comm_stream"
    workspace.comm_handle = 77
    workspace.peer_scratch_ptrs = ("peer_scratch",)
    workspace.peer_signal_ptrs = ("peer_signals",)
    workspace.push_buffers = ("payload", "signals", "counters")

    class _Sequence:
        def __init__(self):
            self.calls = []

        def run(self, *args):
            self.calls.append(("run", args))

        def run_push(self, *args):
            self.calls.append(("run_push", args))

    sequence = _Sequence()
    workspace.sequences = {b_layout: sequence}
    monkeypatch.setattr(backend, "_current_stream_handle", lambda index: 4242)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(backend, "_remember_tail", lambda state, handle, index: None)
    return call, state, workspace, sequence


def test_launch_is_one_host_sequence_call_with_the_workspace_tables(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=1025, n=2048, world_size=4, arch="sm_100a"
    )
    inp = _Recording()
    backend._launch(state, workspace, call, inp, "w_source", "out")
    # 288 N256 tiles: two waves, and halving the width saves no idle slot
    # (2 * 2 = ceil(576 / 148)), so the call runs the N256 main; its class
    # (4, bfloat16, n_major) has no measured pair region, so it runs the
    # one-CTA form. The width and the form travel to the sequence right after
    # grid_z.
    grid = backend.loader.main_grid(
        1025, 2048, world_size=4, sm_count=148, block_n=256, form="cg1"
    )
    assert grid == (148, 1, 1)
    assert sequence.calls == [
        (
            "run",
            (
                inp,
                "scratch",
                "w_source",
                "out",
                "signal_pad",
                ("flags",),
                ("peer_scratch",),
                ("peer_signals",),
                1,
                1025,
                workspace.pitch,
                0,
                1,
                *grid,
                256,
                0,
                4242,
                77,
            ),
        )
    ]
    # The pushes read ``inp`` on the communication stream.
    assert inp.streams == ["comm_stream"]
    assert (state.next_phase, state.ready_epoch) == (1, 1)
    backend._launch(state, workspace, call, inp, "w_source", "out")
    assert sequence.calls[-1][1][11:13] == (1, 2)
    assert (state.next_phase, state.ready_epoch) == (0, 2)
    assert state.poisoned is False


def test_small_rows_use_the_sm_push_sequence_entry_on_the_callers_stream(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=512, n=1280, world_size=8, arch="sm_100a"
    )
    inp = _Recording()
    backend._launch(state, workspace, call, inp, "w_source", "out")
    (entry, args) = sequence.calls[0]
    assert entry == "run_push"
    assert args[6:9] == ("payload", "signals", "counters")
    assert args[12:14] == (0, 1)
    # 160 N256 tiles at ws8 (two waves, three on the twin): within the sm_100a
    # cap of two the rule selects the ``_n128`` main of the (8, bfloat16,
    # n_major) class delivered on sm_100a; the SM push entry receives the width
    # after grid_z like the copy-engine entry.
    assert args[14:17] == backend.loader.main_grid(
        512, 1280, world_size=8, sm_count=148, block_n=128
    )
    # Four M tiles per chunk and the 128-wide program: the one-CTA form (form code 0).
    assert args[17:] == (128, 0, 4242, 77) and len(args) == 21
    # No communication stream is involved on the SM push route.
    assert inp.streams == []
    assert state.next_phase == 1 and state.ready_epoch == 1


def _pinned_spill_code(arch, world_size, dtype_name, b_layout, padded, n):
    """Host-sequence form code of the pinned spill point of ``arch`` naming the
    class, padded rows and ``n`` (``_CG2_SPILL_POINTS``): the launch rows below
    follow the pinned table when a point's form changes."""

    [form] = [
        entry[5]
        for entry in _CG2_SPILL_POINTS[arch]
        if tuple(entry[:5]) == (world_size, dtype_name, b_layout, padded, n)
    ]
    return loader.main_form_code(form)


# The two gate_up points of each architecture (the ws4 M=512 row and the ws8
# M=512 row) and the third sm_100a point (the ws4 QKV M=2048 row).
_GATE_WS4_CODE = _pinned_spill_code("sm_103a", 4, "bfloat16", "k_major", 512, 14336)
_GATE_WS8_CODE = _pinned_spill_code("sm_103a", 8, "bfloat16", "k_major", 512, 7168)
_GATE_WS4_CODE_SM100A = _pinned_spill_code(
    "sm_100a", 4, "bfloat16", "k_major", 512, 14336
)
_GATE_WS8_CODE_SM100A = _pinned_spill_code(
    "sm_100a", 8, "bfloat16", "k_major", 512, 7168
)
_QKV_WS4_M2048_CODE_SM100A = _pinned_spill_code(
    "sm_100a", 4, "bfloat16", "k_major", 2048, 2560
)


def _pinned_order_code(world_size, dtype_name, b_layout, padded, n):
    """Host-sequence form code of the pinned sm_103a tile-order point naming the
    class, padded rows and ``n`` (``_TILE_ORDER_POINTS``): the launch rows below
    follow the pinned table when a point's form changes."""

    [form] = [
        entry[5]
        for entry in _TILE_ORDER_POINTS["sm_103a"]
        if tuple(entry[:5]) == (world_size, dtype_name, b_layout, padded, n)
    ]
    return loader.main_form_code(form)


# The sm_103a tile-order point: the ws4 gate_up M=125 row (128 padded rows).
_ORDER_WS4_CODE = _pinned_order_code(4, "bfloat16", "k_major", 128, 14336)


def _pinned_cg1s_code(world_size, dtype_name, b_layout, padded, n):
    """Host-sequence form code of the pinned sm_103a last-pass N-split point
    naming the class, padded rows and ``n`` (``_CG1S_POINTS``): the launch rows
    below follow the pinned table when a point's form changes."""

    [form] = [
        entry[5]
        for entry in _CG1S_POINTS["sm_103a"]
        if tuple(entry[:5]) == (world_size, dtype_name, b_layout, padded, n)
    ]
    return loader.main_form_code(form)


# The sm_103a last-pass N-split point: the ws8 QKV M=1025 row (1152 padded rows).
_CG1S_WS8_CODE = _pinned_cg1s_code(8, "bfloat16", "k_major", 1152, 1280)


@pytest.mark.parametrize(
    (
        "rows",
        "n",
        "world_size",
        "arch",
        "dtype_name",
        "b_layout",
        "entry",
        "block_n",
        "form_code",
    ),
    [
        # Two N256 waves: within the sm_100a cap of two, beyond the sm_103a cap of one.
        (512, 1280, 8, "sm_100a", "bfloat16", "n_major", "run_push", 128, 0),
        (512, 1280, 8, "sm_103a", "bfloat16", "n_major", "run_push", 256, 0),
        # One N256 wave: within the cap of both architectures.
        (125, 1280, 8, "sm_100a", "bfloat16", "k_major", "run_push", 128, 0),
        (125, 1280, 8, "sm_103a", "bfloat16", "k_major", "run_push", 128, 0),
        # (8, float16, k_major) has a delivered twin on sm_100a only (two-wave row).
        (512, 1280, 8, "sm_100a", "float16", "k_major", "run_push", 128, 0),
        (125, 1280, 8, "sm_103a", "float16", "k_major", "run_push", 256, 0),
        # (4, bfloat16, n_major) has no delivered twin: its sm_103a-only row spans
        # two waves (beyond the sm_103a cap), so neither architecture ships one.
        (512, 2560, 4, "sm_103a", "bfloat16", "n_major", "run_push", 256, 0),
        (512, 2560, 4, "sm_100a", "bfloat16", "n_major", "run_push", 256, 0),
        # The band (eight ranks, exactly 1024 padded rows, N <= 1619; both architectures)
        # takes the SM push route beyond the base rule; three N256 waves keep the N256 main.
        (1024, 1280, 8, "sm_103a", "bfloat16", "k_major", "run_push", 256, 0),
        (1024, 1280, 8, "sm_100a", "bfloat16", "k_major", "run_push", 256, 0),
        # Three N256 waves exceed the cap: the copy-engine route runs the N256 main;
        # a call inside a measured pair rectangle of its class on the architecture
        # selects its CTA-pair twin (form code 1): on sm_103a the ws2 bf16
        # n_major calls of 2048..5760 padded rows (N <= 2048) and the ws4 bf16
        # k_major calls of 1024..1536 padded rows (N <= 6058), the QKV M=1024
        # and M=1025 (1152 padded) rows among them: that rectangle holds no
        # spill point.
        (2048, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 1),
        (4096, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 1),
        (1024, 2560, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 1),
        (1025, 2560, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 1),
        # The sm_103a gate_up points: ws4 bf16 k_major at exactly 512 padded rows
        # and N = 14336, ws8 bf16 k_major at exactly 512 padded rows and N = 7168
        # (500 pads to 512; 513 pads to 640 and leaves) select the spill-wave
        # N-split pair twin of the form each pinned point names (the code derived
        # from ``_CG2_SPILL_POINTS``); the M=2048 gate_up rows are not routed.
        (512, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, _GATE_WS4_CODE),
        (2048, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (500, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, _GATE_WS4_CODE),
        # The sm_103a tile-order point: ws4 bf16 k_major at exactly 128 padded rows
        # and N = 14336 (125 pads to 128; 129 pads to 256 and leaves) selects the
        # ``po`` tile-order twin of the one-CTA program (form code 4) on the
        # copy-engine route with the one-CTA grid; sm_100a has no point.
        (125, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, _ORDER_WS4_CODE),
        (128, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, _ORDER_WS4_CODE),
        (129, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (125, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        # The sm_103a last-pass N-split point: ws8 bf16 k_major at exactly 1152
        # padded rows and N = 1280 (1025 pads to 1152; 1153 pads to 1280 and
        # leaves) selects the ``cg1s2`` twin of the one-CTA program (form code 6)
        # on the copy-engine route with the one-CTA grid; sm_100a has no point.
        (1025, 1280, 8, "sm_103a", "bfloat16", "k_major", "run", 256, _CG1S_WS8_CODE),
        (1152, 1280, 8, "sm_103a", "bfloat16", "k_major", "run", 256, _CG1S_WS8_CODE),
        (1153, 1280, 8, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (1025, 1280, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (512, 7168, 8, "sm_103a", "bfloat16", "k_major", "run", 256, _GATE_WS8_CODE),
        (2048, 7168, 8, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (513, 7168, 8, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        # The measured SM-push ws4 M=512 points on sm_103a: ws4 bf16 k_major and ws4
        # fp16 k_major at exactly 512 padded rows and N = 2560 (500 pads to 512;
        # 513 pads to 640 and leaves) select the CTA-pair twin (form code 1) on
        # the SM push route (512 padded rows at N = 2560 is within the base push
        # rule; 640 padded rows take the copy engines); the points carry no spill
        # point. On sm_100a the same two points are rectangles and the pair takes
        # precedence over the N128 wave rule (the CTA-pair precedence rule,
        # CG2_N128_PRECEDENCE_ARCHES): form code 1 at 256 there too.
        (512, 2560, 4, "sm_103a", "bfloat16", "k_major", "run_push", 256, 1),
        (500, 2560, 4, "sm_103a", "bfloat16", "k_major", "run_push", 256, 1),
        (513, 2560, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (512, 2560, 4, "sm_103a", "float16", "k_major", "run_push", 256, 1),
        (500, 2560, 4, "sm_103a", "float16", "k_major", "run_push", 256, 1),
        (513, 2560, 4, "sm_103a", "float16", "k_major", "run", 256, 0),
        (512, 2560, 4, "sm_100a", "bfloat16", "k_major", "run_push", 256, 1),
        (512, 2560, 4, "sm_100a", "float16", "k_major", "run_push", 256, 1),
        # The measured points on sm_100a: ws2 bf16 n_major at
        # exactly 1024, 2048 or 4096 padded rows and N = 2048, ws4 bf16 k_major at
        # exactly 1024, 1152 or 2048 padded rows and N = 2560 (1000 pads to 1024
        # and takes the twin; 1153 pads to 1280 and leaves; 3000 pads to 3072 and
        # leaves) and at exactly 512 or 1024 padded rows and N = 14336, ws8 bf16
        # k_major at exactly 512 or 1024 padded rows and N = 7168 (500 / 1000 pad
        # to 512 / 1024 and take the twin; 513 / 1025 pad to 640 / 1152 and
        # leave; the M=2048 gate_up rows stay one-CTA). The sm_100a gate_up
        # points -- ws4 bf16 k_major at exactly 512 padded rows and N = 14336,
        # ws8 bf16 k_major at exactly 512 padded rows and N = 7168 (500 pads to
        # 512) -- and the ws4 QKV M=2048 point (exactly 2048 padded rows at
        # N = 2560; 1921 pads to 2048) select the spill-wave N-split pair twin of
        # the form each pinned point names; the 1024-row points keep the CTA-pair
        # twin.
        (2048, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 1),
        (4096, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 1),
        (1024, 2560, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1025, 2560, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (
            2048,
            2560,
            4,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _QKV_WS4_M2048_CODE_SM100A,
        ),
        (
            1921,
            2560,
            4,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _QKV_WS4_M2048_CODE_SM100A,
        ),
        (1000, 2560, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1153, 2560, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (3000, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 0),
        (1024, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 1),
        (
            512,
            14336,
            4,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _GATE_WS4_CODE_SM100A,
        ),
        (
            500,
            14336,
            4,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _GATE_WS4_CODE_SM100A,
        ),
        (513, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (1024, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1000, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1025, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (2048, 14336, 4, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (
            512,
            7168,
            8,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _GATE_WS8_CODE_SM100A,
        ),
        (
            500,
            7168,
            8,
            "sm_100a",
            "bfloat16",
            "k_major",
            "run",
            256,
            _GATE_WS8_CODE_SM100A,
        ),
        (513, 7168, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (1024, 7168, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1000, 7168, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1025, 7168, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (2048, 7168, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        # The measured ws2 M=1024 points on both architectures: bf16 n_major, bf16
        # k_major and fp16 k_major at exactly 1024 padded rows and N = 2048 (1000
        # pads to 1024 and takes the twin; 1025 pads to 1152 and leaves).
        (1024, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 1),
        (1000, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 1),
        (1024, 2048, 2, "sm_103a", "bfloat16", "k_major", "run", 256, 1),
        (1024, 2048, 2, "sm_103a", "float16", "k_major", "run", 256, 1),
        (1025, 2048, 2, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (1536, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 0),
        (1000, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 1),
        (1024, 2048, 2, "sm_100a", "bfloat16", "k_major", "run", 256, 1),
        (1024, 2048, 2, "sm_100a", "float16", "k_major", "run", 256, 1),
        (1025, 2048, 2, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (1536, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 0),
        # Outside every rectangle or another class: the one-CTA N256 main.
        (8192, 2048, 2, "sm_103a", "bfloat16", "n_major", "run", 256, 0),
        (2048, 2560, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (1024, 14336, 4, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        # Outside the band (N > 1619) the 1024-row eight-rank call keeps the copy engines.
        (1024, 2048, 8, "sm_103a", "bfloat16", "k_major", "run", 256, 0),
        (1024, 2048, 8, "sm_100a", "bfloat16", "k_major", "run", 256, 0),
        (1025, 2048, 4, "sm_103a", "bfloat16", "n_major", "run", 256, 0),
        (65536, 2048, 2, "sm_103a", "float16", "n_major", "run", 256, 0),
        (2048, 2048, 2, "sm_100a", "float16", "n_major", "run", 256, 0),
        (1025, 2048, 4, "sm_100a", "bfloat16", "n_major", "run", 256, 0),
        (256, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 0),
        (896, 2048, 2, "sm_100a", "bfloat16", "n_major", "run", 256, 0),
    ],
)
def test_launch_passes_the_loader_tile_width_and_form_after_grid_z_on_both_routes(
    monkeypatch,
    rows,
    n,
    world_size,
    arch,
    dtype_name,
    b_layout,
    entry,
    block_n,
    form_code,
):
    """Both host-sequence entries receive ``block_n`` (128 for a call the
    loader routes to a delivered ``_n128`` main on that architecture within its
    wave cap, 256 otherwise) and the form code (1 for an N256 call inside a
    measured pair region of its class on that architecture, whose class
    delivers the CTA-pair twin there; the point's form code -- 2 for the
    four-tile split, 3 for the two-tile split -- when the call also names a
    delivered spill-wave N-split point of its class; 4 / 5 for an N256 call the
    pair rule leaves on the one-CTA program that names a delivered tile-order
    point of its class; 6 for such a call that names a delivered last-pass
    N-split point of its class; 0 otherwise) positionally after ``grid_z``, with
    the grid of that width and form."""

    call, state, workspace, sequence = _launch_fixture(
        monkeypatch,
        rows=rows,
        n=n,
        world_size=world_size,
        arch=arch,
        b_layout=b_layout,
        dtype_name=dtype_name,
    )
    backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    [(name, args)] = sequence.calls
    assert name == entry
    grid_z = 16 if entry == "run_push" else 15
    assert len(args) == grid_z + 5
    form = backend.loader.MAIN_FORMS[form_code]
    assert backend.loader.main_form_code(form) == form_code
    grid = backend.loader.main_grid(
        rows, n, world_size=world_size, sm_count=148, block_n=block_n, form=form
    )
    assert args[grid_z - 2 : grid_z + 1] == grid
    if form in backend.loader.PAIR_FORMS:
        assert grid[0] % 2 == 0
    if form in backend.loader.ORDER_FORMS or form in backend.loader.ONE_CTA_SPLIT_FORMS:
        assert grid == backend.loader.main_grid(
            rows, n, world_size=world_size, sm_count=148
        )
    assert args[grid_z + 1 :] == (block_n, form_code, 4242, 77)
    assert block_n == backend.loader.main_block_n(
        rows,
        n,
        world_size=world_size,
        arch=arch,
        dtype_name=dtype_name,
        b_layout=b_layout,
    )
    assert form == backend.loader.main_form(
        rows,
        n,
        world_size=world_size,
        arch=arch,
        dtype_name=dtype_name,
        b_layout=b_layout,
        block_n=block_n,
    )


@pytest.mark.parametrize(
    "rows, n, world_size, arch, entry, target_index, k_slices",
    [
        (1025, 2048, 4, "sm_100a", "run", 12, 1),
        (512, 1280, 8, "sm_103a", "run_push", 13, 4),
    ],
)
def test_captured_launch_publishes_the_sentinel_and_clears_its_words_on_both_routes(
    monkeypatch, rows, n, world_size, arch, entry, target_index, k_slices
):
    """Under capture both routes pass the sentinel as ``ready_target``, leave the
    eager epoch untouched, skip ``record_stream`` and clear exactly the
    ``world_size * num_chunks * k_slices`` words the peers wrote (the readiness
    slice table's count of the SM push call -- four words per (peer, chunk) on
    the eight-rank 512-row call of ``sm_103a`` -- and one word per (peer, chunk)
    off the table), after the sequence."""

    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=rows, n=n, world_size=world_size, arch=arch
    )
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    cleared = []
    monkeypatch.setattr(
        backend, "_clear_readiness", lambda ws, words: cleared.append((ws, words))
    )
    inp = _Recording()
    backend._launch(state, workspace, call, inp, "w_source", "out")
    (name, args) = sequence.calls[0]
    assert name == entry
    assert args[target_index] == backend._CAPTURE_READY_TARGET
    assert state.ready_epoch == 0
    assert inp.streams == []
    assert (
        loader.push_kslices(loader.padded_rows(rows), world_size=world_size, arch=arch)
        == k_slices
    )
    assert cleared == [
        (workspace, world_size * backend.loader.chunk_plan(rows)[2] * k_slices)
    ]
    assert state.next_phase == 1 and state.poisoned is False
    assert workspace.captured is True


def test_captured_sm_push_launch_refuses_unmaterialized_tables(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=512, n=1280, world_size=8, arch="sm_103a"
    )
    monkeypatch.setattr(backend.torch.cuda, "is_current_stream_capturing", lambda: True)
    workspace.push_buffers = None
    with pytest.raises(RuntimeError, match="prepare the launcher eagerly"):
        backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    assert sequence.calls == []
    # A caller error before any submission: the group stays usable and uncaptured.
    assert state.poisoned is False and workspace.captured is False


def test_launch_refuses_an_unprepared_layout_without_poisoning(monkeypatch):
    call, state, workspace, _sequence = _launch_fixture(
        monkeypatch, rows=256, n=2048, world_size=2, arch="sm_100a"
    )
    workspace.sequences = {}
    with pytest.raises(RuntimeError, match="host sequence"):
        backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    assert state.poisoned is False


def test_sequence_failure_poisons_the_launch_state(monkeypatch):
    call, state, workspace, sequence = _launch_fixture(
        monkeypatch, rows=256, n=2048, world_size=2, arch="sm_100a"
    )

    def failing(*args):
        raise RuntimeError("boom")

    sequence.run = failing
    with pytest.raises(RuntimeError, match="boom"):
        backend._launch(state, workspace, call, _Recording(), "w_source", "out")
    assert state.poisoned is True


class _FakeHandle:
    """Symmetric-memory handle stand-in: topology, ``buffer_size`` and CPU peer buffers."""

    def __init__(self, *, rank, world_size, buffer_size):
        self.rank = rank
        self.world_size = world_size
        self.buffer_size = buffer_size

    def get_buffer(self, peer, shape, dtype, offset):
        return torch.empty(shape, dtype=dtype)


def _patch_symmetric_memory(monkeypatch, handles):
    """``symm_mem.empty`` hands out CPU tensors; ``rendezvous`` returns the scripted handles in order."""

    allocated = []

    def empty(*shape, dtype, device):
        tensor = torch.empty(*shape, dtype=dtype)
        allocated.append(weakref.ref(tensor))
        return tensor

    def rendezvous(tensor, group):
        assert group == "g"
        return handles.pop(0)

    monkeypatch.setattr(backend.symm_mem, "empty", empty)
    monkeypatch.setattr(backend.symm_mem, "rendezvous", rendezvous)
    return allocated


def _guard_call(rank=1, world_size=2):
    return backend._Call(
        0, rank, world_size, "g", "bfloat16", "sm_100a", 128, 2048, "n_major"
    )


def _allocate_words(words):
    return lambda: backend.symm_mem.empty(words, dtype=torch.uint32, device=0)


def test_rendezvous_reallocates_past_a_stale_handle_and_releases_the_parked_allocation(
    monkeypatch,
):
    """A handle smaller than its tensor is torch's cached handle of a freed allocation (pre-pytorch#192579)."""

    words = 64
    stale = _FakeHandle(rank=1, world_size=2, buffer_size=4 * words - 4)
    fresh = _FakeHandle(rank=1, world_size=2, buffer_size=4 * words)
    allocated = _patch_symmetric_memory(monkeypatch, [stale, fresh])
    votes = []
    monkeypatch.setattr(
        backend, "_group_max", lambda mask, call, group: votes.append(mask) or mask
    )
    monkeypatch.setitem(backend._RENDEZVOUS_STATS, "stale_retries", 0)
    with pytest.warns(RuntimeWarning, match="pytorch#192579"):
        [(tensor, handle)] = backend._rendezvous(
            _guard_call(), "group", [("barrier flag", _allocate_words(words))]
        )
    assert handle is fresh and tensor.numel() == words
    assert votes == [1, 0]
    assert backend._RENDEZVOUS_STATS["stale_retries"] == 1
    gc.collect()
    # The parked (stale) allocation is released once the loop ends; the returned one lives on.
    assert allocated[0]() is None
    assert allocated[1]() is tensor


def test_rendezvous_votes_once_per_round_and_retries_only_the_stale_allocations(
    monkeypatch,
):
    """Flags and scratch of one growth round share one all-reduce; only the stale one is re-allocated."""

    words, scratch_words = 32, 64
    flags_ok = _FakeHandle(rank=1, world_size=2, buffer_size=4 * words)
    scratch_stale = _FakeHandle(rank=1, world_size=2, buffer_size=4 * scratch_words - 4)
    scratch_ok = _FakeHandle(rank=1, world_size=2, buffer_size=4 * scratch_words)
    allocated = _patch_symmetric_memory(
        monkeypatch, [flags_ok, scratch_stale, scratch_ok]
    )
    votes = []
    monkeypatch.setattr(
        backend, "_group_max", lambda mask, call, group: votes.append(mask) or mask
    )
    monkeypatch.setitem(backend._RENDEZVOUS_STATS, "stale_retries", 0)
    with pytest.warns(RuntimeWarning, match="scratch"):
        (flags, flag_handle), (scratch, scratch_handle) = backend._rendezvous(
            _guard_call(),
            "group",
            [
                ("barrier flag", _allocate_words(words)),
                ("scratch", _allocate_words(scratch_words)),
            ],
        )
    # One vote per round (a bitmask over the round's allocations), not one per allocation.
    assert votes == [0b10, 0]
    assert flag_handle is flags_ok and scratch_handle is scratch_ok
    assert flags.numel() == words and scratch.numel() == scratch_words
    assert len(allocated) == 3 and backend._RENDEZVOUS_STATS["stale_retries"] == 1
    gc.collect()
    assert allocated[1]() is None
    assert allocated[0]() is flags and allocated[2]() is scratch


def test_rendezvous_follows_a_peer_stale_vote_so_every_rank_reallocates(monkeypatch):
    words = 64
    first = _FakeHandle(rank=0, world_size=2, buffer_size=4 * words)
    second = _FakeHandle(rank=0, world_size=2, buffer_size=4 * words)
    allocated = _patch_symmetric_memory(monkeypatch, [first, second])
    peer_votes = iter([1, 0])
    local_masks = []
    monkeypatch.setattr(
        backend,
        "_group_max",
        lambda mask, call, group: local_masks.append(mask) or next(peer_votes),
    )
    monkeypatch.setitem(backend._RENDEZVOUS_STATS, "stale_retries", 3)
    [(tensor, handle)] = backend._rendezvous(
        _guard_call(rank=0), "group", [("scratch", _allocate_words(words))]
    )
    # This rank's handles were fine; the peer's stale vote still made it re-allocate.
    assert local_masks == [0, 0]
    assert handle is second and len(allocated) == 2
    assert backend._RENDEZVOUS_STATS["stale_retries"] == 4
    gc.collect()
    assert allocated[0]() is None and allocated[1]() is tensor


def test_rendezvous_gives_up_after_the_bounded_attempts_with_a_named_error(monkeypatch):
    words = 64
    handles = [
        _FakeHandle(rank=1, world_size=2, buffer_size=4)
        for _ in range(backend._RENDEZVOUS_ATTEMPTS)
    ]
    allocated = _patch_symmetric_memory(monkeypatch, handles)
    monkeypatch.setattr(backend, "_group_max", lambda mask, call, group: mask)
    monkeypatch.setitem(backend._RENDEZVOUS_STATS, "stale_retries", 1)
    with pytest.raises(backend.StaleSymmetricHandleError, match="max_rows"):
        backend._rendezvous(
            _guard_call(), "group", [("scratch", _allocate_words(words))]
        )
    assert issubclass(backend.StaleSymmetricHandleError, RuntimeError)
    assert len(allocated) == backend._RENDEZVOUS_ATTEMPTS and not handles
    gc.collect()
    assert all(ref() is None for ref in allocated)


def test_rendezvous_rejects_a_topology_mismatch_before_voting(monkeypatch):
    _patch_symmetric_memory(
        monkeypatch, [_FakeHandle(rank=0, world_size=4, buffer_size=4 * 64)]
    )
    monkeypatch.setattr(
        backend,
        "_group_max",
        lambda mask, call, group: pytest.fail("no vote on a topology mismatch"),
    )
    with pytest.raises(RuntimeError, match="topology"):
        backend._rendezvous(_guard_call(), "group", [("scratch", _allocate_words(64))])


def test_publish_flags_publishes_every_peer_pointer_of_the_cleared_pad():
    call = _guard_call(rank=1, world_size=2)
    words = backend.loader.barrier_flag_words(2)
    fresh = _FakeHandle(rank=1, world_size=2, buffer_size=4 * words)
    flags = torch.zeros(words, dtype=torch.uint32)
    state = backend._LaunchState(rank=1, world_size=2)
    backend._publish_flags(state, call, flags, fresh)
    assert state.flag_handle is fresh and state.flags is flags
    assert len(state.flag_peers) == 2


def test_rendezvous_clears_fresh_allocations_before_the_single_vote(monkeypatch):
    """The round zeroes the local pads, synchronizes and votes: the vote is also the post-allocation barrier."""

    words = 64
    fresh = _FakeHandle(rank=1, world_size=2, buffer_size=4 * words)
    _patch_symmetric_memory(monkeypatch, [fresh])
    order = []
    monkeypatch.setattr(
        backend, "_group_max", lambda mask, call, group: order.append("vote") or mask
    )
    monkeypatch.setattr(
        backend.torch.cuda,
        "current_stream",
        lambda index: type(
            "S", (), {"synchronize": lambda self: order.append("sync")}
        )(),
    )
    [(tensor, handle)] = backend._rendezvous(
        _guard_call(),
        "group",
        [("barrier flag", _allocate_words(words))],
        clear=lambda index, tensor, handle: order.append(("clear", index))
        or tensor.zero_(),
    )
    assert order == [("clear", 0), "sync", "vote"] and not tensor.any()


class _SyncStream:
    def __init__(self):
        self.synchronized = 0

    def synchronize(self):
        self.synchronized += 1


def _prepare_fixture(monkeypatch, name):
    _patch_distributed(monkeypatch, world_size=2, rank=0)
    stream = _SyncStream()
    monkeypatch.setattr(backend.torch.cuda, "current_stream", lambda index: stream)
    monkeypatch.setattr(
        backend.torch.cuda, "is_current_stream_capturing", lambda: False
    )
    monkeypatch.setattr(backend.loader, "load", lambda *a: object())
    group = _fake_group(2, 0, name=name)
    call = backend._Call(0, 0, 2, name, "bfloat16", "sm_100a", 512, 2048, "n_major")
    return call, group, stream


def test_prepare_session_rendezvous_plan_covers_flags_and_scratch_in_one_round(
    monkeypatch,
):
    call, group, stream = _prepare_fixture(monkeypatch, "one_round")
    plans = []

    def rendezvous(call, group, plan, *, clear):
        plans.append([what for what, _ in plan])
        flags = torch.full((4,), 7, dtype=torch.uint32)
        clear(0, flags, None)
        assert not flags.any()  # the barrier-flag pad is cleared before the vote
        raise backend.StaleSymmetricHandleError("stop here")

    monkeypatch.setattr(backend, "_rendezvous", rendezvous)
    with pytest.raises(backend.StaleSymmetricHandleError):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
    # First prepare of a group: barrier flags and the scratch growth share one rendezvous round.
    assert plans == [["barrier flag", "scratch"]]


def test_prepare_session_keeps_the_state_usable_after_a_rank_uniform_stale_exhaustion(
    monkeypatch,
):
    call, group, stream = _prepare_fixture(monkeypatch, "stale_exhaustion")

    def exhausted(call, group, plan, *, clear):
        raise backend.StaleSymmetricHandleError("stale")

    monkeypatch.setattr(backend, "_rendezvous", exhausted)
    with pytest.raises(backend.StaleSymmetricHandleError):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
    state = backend._launch_state(call, group)
    assert state.poisoned is False and stream.synchronized == 1


def test_prepare_session_poisons_the_state_when_a_growth_collective_fails(monkeypatch):
    call, group, stream = _prepare_fixture(monkeypatch, "growth_failure")

    def failing(call, group, plan, *, clear):
        raise RuntimeError("nvshmem_malloc failed")

    monkeypatch.setattr(backend, "_rendezvous", failing)
    with pytest.raises(RuntimeError, match="nvshmem_malloc"):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
    state = backend._launch_state(call, group)
    assert state.poisoned is True
    with pytest.raises(RuntimeError, match="poisoned"):
        backend._prepare_session(call, group, torch.bfloat16, capacity_rows=512)
