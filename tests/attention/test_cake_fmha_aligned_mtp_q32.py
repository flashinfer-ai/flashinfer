"""Aligned speculative-decode (MTP verify) grouped paged decode program of the
Cake FMHA module (``backend="cake"``): the cluster program of 2 CTAs whose
32-row query tile holds the 1 to 4 query tokens of a KV head's 8 query heads.

Host-only: the module's Python mirror of the source selector against the source
selector test's expectations (copied here: the pinned point and its neighbours,
the shape rows per target at each target's SM count of record, the trtllm-gen
split at other SM counts), the pinned program record against the generated
loader, the load-time checks against a stand-in loader (each record that differs
from the pinned launch fails closed), and the shape rows through
``select_cake_fmha_decode_route`` per target and SM count, including the
launches outside the program's contract.
GPU: the shape rows through ``trtllm_batch_decode_with_kv_cache(backend="cake")``
against an FP32 bottom-right-causal reference at atol = rtol = 1e-2 (the
reference's largest ``|O|`` above 1e-2 on every row, so no row passes
vacuously), the route taken, CUDA-graph replay bitwise equal to the eager result
on the aligned program (also with shorter lengths under the captured KV bound),
and the kernel tests' cases of 3 and 4 query tokens on the program.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from types import SimpleNamespace

import pytest
import torch

import flashinfer.cake_fmha as cake_api
import flashinfer.jit
from flashinfer.decode import trtllm_batch_decode_with_kv_cache
from flashinfer.jit.cake_program import GRID_ARGS, CakeProgram

HEAD_DIM = 128
PAGE_SIZE = 16
# Query heads per KV head of the program (8 heads x 4 tokens: one 32-row tile).
GROUP = 8
MAX_Q_LEN = 4
ATOL = 1e-2
RTOL = 1e-2
# The reference's largest |O| must exceed this on every row: a peaked softmax.  A
# smaller reference would pass the tolerance check without exercising the kernel.
MIN_REF_ABSMAX = 1e-2
COMPONENT = "decode_aligned_bf16_mtp_q32"
PRESENT = "decode_balanced_bf16"
# The selector's verdicts -> the program identifiers of the generated loader.
PROGRAMS = {("cga", 2): "ALIGNED_MTP_Q32_CGA2_BF16"}
# Dynamic shared memory of the program's launch (the kernel's computed size).
PROGRAM_SMEM = {"ALIGNED_MTP_Q32_CGA2_BF16": 201088}
ARCHS = ("sm_100a", "sm_103a")
# The module's device target -> the program target the selector reads.
TARGET_ARCH = {"sm100a": "sm_100a", "sm103a": "sm_103a"}
# The binding's host parameters in the kernel's order (then the grid arguments).
HOST_PARAMS = (
    "Q",
    "K",
    "V",
    "O",
    "page_table",
    "seq_lens_kv",
    "max_pages_per_seq",
    "softmax_scale_log2",
    "num_q_heads",
    "num_kv_heads",
    "heads_per_kv",
    "batch_size",
    "seq_len_q",
    "skip_corr_threshold",
)

# The sixteen long-context lengths; a batch of requests cycles the list.
# fmt: off
LONG_CONTEXT_LENGTHS = [
    8193, 57345, 73729, 81921, 98305, 106497, 114689, 131073,
    139265, 147457, 163841, 180225, 196609, 212993, 229377, 237569,
]
# fmt: on


def _cycled(batch: int) -> list[int]:
    return [LONG_CONTEXT_LENGTHS[i % len(LONG_CONTEXT_LENGTHS)] for i in range(batch)]


# The measured shape rows: name -> (num_kv_heads, q_len, seq_lens); eight query
# heads per KV head, BF16, page 16, HND, one page table shared by K and V, causal
# speculative decode with a uniform q_len per request.
SHAPE_ROWS = {
    "uniform_b8_hkv8_s4096_mtp3": (8, 3, [4096] * 8),
    "uniform_b8_hkv8_s4096_mtp7": (8, 7, [4096] * 8),
    "agentx_b16_hkv1_mtp7": (1, 7, _cycled(16)),
    "uniform_b1_hkv1_s60007_mtp7": (1, 7, [60007]),
    "ragged_b4_hkv2_mtp7": (2, 7, [300, 257, 5000, 777]),
}

# The kernel tests' cases of 3 and 4 query tokens: name -> (num_q_heads,
# num_kv_heads, q_len, seq_lens): the mtp3 row, ragged lengths with page tails,
# short requests (the mask cuts inside the first tile; a request of exactly q_len
# tokens; one cluster rank active) and a full 32-row tile with a long request.
# Each runs on the program with the selector held to it.  The cases of 1 and 2
# query tokens never reach the route: the balanced branch it sits in admits 3 to
# 8 tokens (and 1 token only above its single-token band).
CASE_ROWS = {
    "case_b8_hq64_hkv8_s4096_q3": (64, 8, 3, [4096] * 8),
    "case_b4_hq16_hkv2_ragged_q3": (16, 2, 3, [300, 257, 5000, 777]),
    "case_b3_hq8_hkv1_short_q3": (8, 1, 3, [3, 129, 300]),
    "case_b2_hq16_hkv2_long_q4": (16, 2, 4, [65536, 2048]),
}

# --- BEGIN SEL_Q32CGA_EXPECTATIONS (the source selector test's tables; copied verbatim)
# The SM count of each target's records of derivation (the trtllm-gen split
# reads it) and the other SM counts the split tests sweep.
SM_COUNT: dict[str, int] = {"sm_100a": 148, "sm_103a": 152}
# fmt: off
OTHER_SM_COUNTS: list[int] = [
    1, 2, 63, 64, 100, 127, 128, 132, 148, 152, 160, 191, 192, 255, 256, 296,
]
# fmt: on
# Batch, KV-head, max-KV and q_len values the selector is swept over.
BATCHES: list[int] = [1, 2, 4, 7, 8, 9, 16, 32, 64, 128]
KV_HEADS: list[int] = [1, 2, 4, 7, 8, 9, 16]
# fmt: off
MAX_KV: list[int] = [
    1, 3, 255, 256, 257, 511, 512, 513, 1024, 1025, 2048, 4095, 4096, 4097, 8192,
    8193, 65536, 237569,
]
# fmt: on
Q_LENS: list[int] = [1, 2, 3, 4]
# (arch, batch, kv_heads, max_kv, q_len, program) at the target's SM count of
# record: the pinned points.
PINNED: list[tuple[str, int, int, int, int, tuple[str, int]]] = [
    ("sm_100a", 8, 8, 4096, 3, ("cga", 2)),
]
# (batch, kv_heads, max_kv, q_len): the pinned point's neighbours on every axis,
# on both targets the present route; the first four have the pinned point's 64
# (request, KV head) tiles and so the same split.
NEIGHBOUR_POINTS: list[tuple[int, int, int, int]] = [
    (4, 16, 4096, 3),
    (16, 4, 4096, 3),
    (64, 1, 4096, 3),
    (32, 2, 4096, 3),
    (7, 8, 4096, 3),
    (9, 8, 4096, 3),
    (8, 7, 4096, 3),
    (8, 9, 4096, 3),
    (8, 8, 4095, 3),
    (8, 8, 4097, 3),
    (8, 8, 4096, 2),
    (8, 8, 4096, 4),
    (8, 8, 4096, 1),
    (16, 8, 4096, 3),
    (8, 8, 2048, 3),
    (8, 8, 8192, 3),
]
# row -> (batch, KV heads, max KV length, q_len, {arch: program}) of the standings
# rows at each target's SM count of record (None: the present route).
SEL_SHAPE_ROWS: dict[
    str, tuple[int, int, int, int, dict[str, tuple[str, int] | None]]
] = {
    "uniform_b8_hkv8_s4096_mtp3": (
        8,
        8,
        4096,
        3,
        {"sm_100a": ("cga", 2), "sm_103a": None},
    ),
    "uniform_b8_hkv8_s4096_mtp7": (8, 8, 4096, 7, {"sm_100a": None, "sm_103a": None}),
    "agentx_b16_hkv1_mtp7": (16, 1, 237569, 7, {"sm_100a": None, "sm_103a": None}),
    "uniform_b1_hkv1_s60007_mtp7": (1, 1, 60007, 7, {"sm_100a": None, "sm_103a": None}),
    "ragged_b4_hkv2_mtp7": (4, 2, 5000, 7, {"sm_100a": None, "sm_103a": None}),
}
# (sm_count, program) of the pinned point (8 requests x 8 KV heads at 4096
# tokens, 3 tokens, sm_100a): the split is min(8, sm_count // 64).
SPLIT_EDGES: list[tuple[int, tuple[str, int] | None]] = [
    (127, None),
    (128, ("cga", 2)),
    (148, ("cga", 2)),
    (152, ("cga", 2)),
    (191, ("cga", 2)),
    (192, None),
    (296, None),
]
# --- END SEL_Q32CGA_EXPECTATIONS ---

NEIGHBOURS = [(arch, *point) for arch in ARCHS for point in NEIGHBOUR_POINTS]


def _ctas_kv(batch: int, kv_heads: int, max_kv: int, sm_count: int) -> int:
    # The trtllm-gen split of the 32-row grouped class at q_len <= 4: the CTAs one
    # (request, KV head) tile's KV sequence is split over.
    return min(-(-max_kv // 512), max(1, sm_count // (batch * kv_heads)))


def _select(batch, kv_heads, max_kv, q_len, arch, sm_count=None, **overrides):
    kwargs = dict(
        q_len=q_len,
        num_q_heads=GROUP * kv_heads,
        arch=arch,
        sm_count=SM_COUNT[arch] if sm_count is None else sm_count,
    )
    kwargs.update(overrides)
    return cake_api.spec_q32_cga_program(batch, kv_heads, max_kv, **kwargs)


@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_shape_rows_are_the_selector_rows(row) -> None:
    assert set(SHAPE_ROWS) == set(SEL_SHAPE_ROWS)
    num_kv_heads, q_len, seq_lens = SHAPE_ROWS[row]
    shape = (len(seq_lens), num_kv_heads, max(seq_lens), q_len)
    assert shape == SEL_SHAPE_ROWS[row][:4]


def test_case_rows_spell_the_kernel_test_groups() -> None:
    for num_q_heads, num_kv_heads, q_len, seq_lens in CASE_ROWS.values():
        assert num_q_heads == GROUP * num_kv_heads
        assert 3 <= q_len <= MAX_Q_LEN
        assert min(seq_lens) >= q_len


# --- The selector mirror against the source selector test ------------------


def test_aligned_mtp_q32_selector_constants_of_record() -> None:
    assert cake_api.SEL_Q32CGA_ARCHS == ARCHS
    assert cake_api.SEL_Q32CGA_GROUP == GROUP
    assert cake_api.SEL_Q32CGA_MAX_Q_LEN == MAX_Q_LEN
    assert cake_api.SEL_Q32CGA_KV_PER_CTA_SPLIT == 512
    assert cake_api.SEL_Q32CGA_MAX_CGA_CTAS == 16
    assert set(cake_api.SEL_Q32CGA_PINS) == set(ARCHS)


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_mtp_q32_selector_pins_name_shipped_programs(arch) -> None:
    for _point, program in cake_api.SEL_Q32CGA_PINS[arch]:
        assert program in PROGRAMS, (arch, program)


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_mtp_q32_selector_gives_one_verdict_on_every_shape(arch) -> None:
    pinned = {point for point, _ in cake_api.SEL_Q32CGA_PINS[arch]}
    verdicts = (None, *PROGRAMS)
    for batch in BATCHES:
        for kv_heads in KV_HEADS:
            for max_kv in MAX_KV:
                for q_len in Q_LENS:
                    for sm_count in [SM_COUNT[arch], *OTHER_SM_COUNTS]:
                        got = _select(batch, kv_heads, max_kv, q_len, arch, sm_count)
                        point = (batch, kv_heads, max_kv, q_len)
                        assert got in verdicts, (point, sm_count, got)
                        # A program only at a pinned point, and only where the
                        # split at this SM count picks its cluster size.
                        assert got is None or point in pinned, (point, got)
                        assert (
                            got is None
                            or _ctas_kv(batch, kv_heads, max_kv, sm_count) == got[1]
                        ), (point, sm_count, got)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(q_len=0),
        dict(q_len=5),
        dict(q_len=7),
        dict(num_q_heads=8 * GROUP + 1),
        dict(num_q_heads=8 * GROUP - 1),
        dict(num_q_heads=16 * GROUP),
        dict(arch="sm_90a"),
        dict(arch="sm_120a"),
        dict(arch="sm_100f"),
        dict(sm_count=0),
        dict(sm_count=-148),
    ],
)
def test_aligned_mtp_q32_selector_other_shapes_keep_the_present_route(kwargs) -> None:
    for max_kv in MAX_KV:
        base = dict(
            q_len=3,
            num_q_heads=8 * GROUP,
            arch=ARCHS[0],
            sm_count=SM_COUNT[ARCHS[0]],
        )
        base.update(kwargs)
        assert cake_api.spec_q32_cga_program(8, 8, max_kv, **base) is None, max_kv


@pytest.mark.parametrize("arch", ARCHS)
def test_aligned_mtp_q32_selector_empty_launch_keeps_the_present_route(arch) -> None:
    select, sm = cake_api.spec_q32_cga_program, SM_COUNT[arch]
    heads = 8 * GROUP
    assert (
        select(0, 8, 4096, q_len=3, num_q_heads=heads, arch=arch, sm_count=sm) is None
    )
    assert select(8, 0, 4096, q_len=3, num_q_heads=0, arch=arch, sm_count=sm) is None
    assert select(8, 8, 0, q_len=3, num_q_heads=heads, arch=arch, sm_count=sm) is None
    assert (
        select(-8, -8, 4096, q_len=3, num_q_heads=-heads, arch=arch, sm_count=sm)
        is None
    )


@pytest.mark.parametrize(
    "case", PINNED, ids=lambda c: f"{c[0]}-b{c[1]}-hkv{c[2]}-kv{c[3]}-q{c[4]}"
)
def test_aligned_mtp_q32_selector_pinned_points_route_as_derived(case) -> None:
    arch, batch, kv_heads, max_kv, q_len, program = case
    assert _select(batch, kv_heads, max_kv, q_len, arch) == program


@pytest.mark.parametrize(
    "case", NEIGHBOURS, ids=lambda c: f"{c[0]}-b{c[1]}-hkv{c[2]}-kv{c[3]}-q{c[4]}"
)
def test_aligned_mtp_q32_selector_neighbours_keep_the_present_route(case) -> None:
    arch, batch, kv_heads, max_kv, q_len = case
    assert _select(batch, kv_heads, max_kv, q_len, arch) is None


@pytest.mark.parametrize("arch", ARCHS)
@pytest.mark.parametrize("row", sorted(SEL_SHAPE_ROWS))
def test_aligned_mtp_q32_selector_shape_rows_route_as_served(row, arch) -> None:
    batch, kv_heads, max_kv, q_len, want = SEL_SHAPE_ROWS[row]
    assert _select(batch, kv_heads, max_kv, q_len, arch) == want[arch]


def test_aligned_mtp_q32_selector_sm103a_pins_nothing() -> None:
    for batch in BATCHES:
        for kv_heads in KV_HEADS:
            for max_kv in MAX_KV:
                for q_len in Q_LENS:
                    for sm_count in [SM_COUNT["sm_103a"], *OTHER_SM_COUNTS]:
                        got = _select(
                            batch, kv_heads, max_kv, q_len, "sm_103a", sm_count
                        )
                        assert got is None, (batch, kv_heads, max_kv, q_len, sm_count)


@pytest.mark.parametrize(
    "case", PINNED, ids=lambda c: f"{c[0]}-b{c[1]}-hkv{c[2]}-kv{c[3]}-q{c[4]}"
)
def test_aligned_mtp_q32_selector_programs_follow_the_split(case) -> None:
    arch, batch, kv_heads, max_kv, q_len, program = case
    for sm_count in range(1, 4 * SM_COUNT[arch]):
        ctas = _ctas_kv(batch, kv_heads, max_kv, sm_count)
        expect = program if ctas == program[1] else None
        got = _select(batch, kv_heads, max_kv, q_len, arch, sm_count)
        assert got == expect, (sm_count, expect)


@pytest.mark.parametrize("sm_count, program", SPLIT_EDGES)
def test_aligned_mtp_q32_selector_split_edges_of_the_pinned_point(
    sm_count, program
) -> None:
    assert _select(8, 8, 4096, 3, "sm_100a", sm_count) == program


# --- The pinned program record ----------------------------------------------


def test_cake_fmha_aligned_mtp_q32_program_table_and_launch_grid() -> None:
    """The identifier table pins one record per program; the launch grid (the
    parity lane's host grid) is ``(CTAs, KV heads, requests)``: one cluster of
    the program's CTAs per (request, KV head)."""

    assert cake_api._ALIGNED_MTP_Q32_COMPONENT == COMPONENT
    assert cake_api._ALIGNED_MTP_Q32_PROGRAM_IDS == PROGRAMS
    table = cake_api._ALIGNED_MTP_Q32_PROGRAMS
    assert sorted(table) == sorted(PROGRAMS.values())
    for (_kind, ctas), ident in PROGRAMS.items():
        table_ctas, name = table[ident]
        assert table_ctas == ctas
        assert name.startswith("cake_fmha_aligned_mtp_q32_")
        digest = name.removeprefix("cake_fmha_aligned_mtp_q32_")
        assert len(digest) == 20 and set(digest) <= set("0123456789abcdef")
        record = SimpleNamespace(cluster=(ctas, 1, 1))
        grid = cake_api.cake_fmha_aligned_mtp_q32_launch_grid
        assert grid(record, 8, 8) == (ctas, 8, 8)
        assert grid(record, 3, 2) == (ctas, 2, 3)
    assert tuple(cake_api._ALIGNED_MTP_Q32_HOST_PARAMS) == HOST_PARAMS
    assert cake_api._ALIGNED_MTP_Q32_SKIP_CORR_THRESHOLD == 0.0


def test_cake_fmha_aligned_mtp_q32_program_table_matches_the_generated_loader() -> None:
    """The pinned record names the generated loader's program of the same
    identifier (the load-time checks, host-only): name, cluster, block, binding
    argument order, tensors and both targets; the loader publishes exactly that
    program."""

    from flashinfer.jit import cake_fmha_aligned_mtp_q32 as loader

    names = []
    for ident, (ctas, name) in cake_api._ALIGNED_MTP_Q32_PROGRAMS.items():
        program = getattr(loader, ident)
        assert isinstance(program, CakeProgram)
        assert program.name == name
        assert tuple(program.cluster) == (ctas, 1, 1)
        assert tuple(program.block) == (512, 1, 1)
        assert tuple(program.args) == (*HOST_PARAMS, *GRID_ARGS)
        assert tuple(program.tensors) == HOST_PARAMS[:6]
        assert {"sm_100a", "sm_103a"} <= set(program.targets)
        assert program.use_pdl is False and program.cooperative is False
        assert program.dynamic_smem_bytes == PROGRAM_SMEM[ident]
        names.append(name)
    assert sorted(p.name for p in loader.PROGRAMS) == sorted(names)


# --- The load-time checks against a stand-in loader (host-only) -------------


def _stand_in_records() -> dict[str, CakeProgram]:
    """One record per identifier with exactly the launch the module pins."""

    records = {}
    for ident, (ctas, name) in cake_api._ALIGNED_MTP_Q32_PROGRAMS.items():
        records[ident] = CakeProgram(
            name=name,
            operator_dir="cake_fmha_aligned_mtp_q32",
            kernel="kernel",
            sources=("kernel.cu", "binding.cu"),
            targets=ARCHS,
            compile_options=tuple((arch, ()) for arch in ARCHS),
            args=(*HOST_PARAMS, *GRID_ARGS),
            tensors=HOST_PARAMS[:6],
            block=(512, 1, 1),
            cluster=(ctas, 1, 1),
            dynamic_smem_bytes=PROGRAM_SMEM[ident],
            use_pdl=False,
            cooperative=False,
            needs_device_linking=False,
        )
    return records


def _install_stand_in_loader(monkeypatch, records) -> None:
    """Publish ``records`` as ``flashinfer.jit.cake_fmha_aligned_mtp_q32`` for the
    test; the loader is imported lazily, at load."""

    loader = types.ModuleType("flashinfer.jit.cake_fmha_aligned_mtp_q32")
    for ident, record in records.items():
        setattr(loader, ident, record)
    loader.PROGRAMS = tuple(records.values())
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setattr(
        flashinfer.jit, "cake_fmha_aligned_mtp_q32", loader, raising=False
    )


def _aligned_mtp_q32_route(aligned_program, target="sm100a"):
    return cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=8,
        q_len=3,
        num_q_heads=GROUP * 8,
        num_kv_heads=8,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=False,
        component=COMPONENT,
        page_size=PAGE_SIZE,
        aligned_program=aligned_program,
    )


def _record_defects(ident, records):
    """Records that differ from the pinned launch of ``ident`` (None: the loader
    publishes no such identifier), with the load-time refusal each must meet."""

    record = records[ident]
    ctas = record.cluster[0]
    swapped_args = (record.args[1], record.args[0], *record.args[2:])
    no_seq_len_q = tuple(arg for arg in record.args if arg != "seq_len_q")
    return {
        "missing": (None, "publishes no CakeProgram"),
        "not_a_cake_program": (
            SimpleNamespace(**dataclasses.asdict(record)),
            "publishes no CakeProgram",
        ),
        "another_name": (
            dataclasses.replace(record, name="cake_fmha_aligned_mtp_q32_" + "0" * 20),
            "this module pins",
        ),
        "no_sm_100a_build": (
            dataclasses.replace(record, targets=("sm_103a",)),
            "ships no 'sm_100a' build",
        ),
        "argument_order": (
            dataclasses.replace(record, args=swapped_args),
            "binding takes",
        ),
        "no_grid_arguments": (
            dataclasses.replace(record, args=HOST_PARAMS),
            "binding takes",
        ),
        "no_seq_len_q": (
            dataclasses.replace(record, args=no_seq_len_q),
            "binding takes",
        ),
        "cluster_size": (
            dataclasses.replace(record, cluster=(2 * ctas, 1, 1)),
            "launches cluster",
        ),
        "cluster_shape": (
            dataclasses.replace(record, cluster=(ctas, 1, 2)),
            "launches cluster",
        ),
    }


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("ident", sorted(PROGRAMS.values()))
def test_cake_fmha_aligned_mtp_q32_load_accepts_the_pinned_record(
    monkeypatch, ident, target
) -> None:
    records = _stand_in_records()
    _install_stand_in_loader(monkeypatch, records)
    monkeypatch.setattr(CakeProgram, "entry", lambda self, arch: (self.name, arch))
    module = cake_api._load_cake_fmha_aligned_mtp_q32_decode_module(
        _aligned_mtp_q32_route(ident, target)
    )
    assert isinstance(module, cake_api.CakeFmhaAlignedMtpQ32DecodeModule)
    assert module.program is records[ident]
    assert module.target == TARGET_ARCH[target]
    assert module.entry == (records[ident].name, TARGET_ARCH[target])


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "not_a_cake_program",
        "another_name",
        "no_sm_100a_build",
        "argument_order",
        "no_grid_arguments",
        "no_seq_len_q",
        "cluster_size",
        "cluster_shape",
    ],
)
@pytest.mark.parametrize("ident", sorted(PROGRAMS.values()))
def test_cake_fmha_aligned_mtp_q32_load_fails_closed_on_a_record_that_differs(
    monkeypatch, ident, defect
) -> None:
    records = _stand_in_records()
    bad, message = _record_defects(ident, records)[defect]
    if bad is None:
        del records[ident]
    else:
        records[ident] = bad
    _install_stand_in_loader(monkeypatch, records)

    def no_entry(self, arch):
        raise AssertionError("a refused record reached the build")

    monkeypatch.setattr(CakeProgram, "entry", no_entry)
    with pytest.raises(RuntimeError, match=message):
        cake_api._load_cake_fmha_aligned_mtp_q32_decode_module(
            _aligned_mtp_q32_route(ident)
        )


def test_cake_fmha_aligned_mtp_q32_load_fails_closed_on_an_unknown_program(
    monkeypatch,
) -> None:
    _install_stand_in_loader(monkeypatch, _stand_in_records())
    with pytest.raises(RuntimeError, match="names no program"):
        cake_api._load_cake_fmha_aligned_mtp_q32_decode_module(
            _aligned_mtp_q32_route("ALIGNED_MTP_Q32_CGA4_BF16")
        )


# --- The shape rows through the module's route selection (host-only) --------


def _decode_kwargs(
    *,
    num_kv_heads,
    q_len,
    seq_lens,
    num_q_heads=None,
    stacked_kv=False,
    sm_count=148,
):
    """Host-side ``select_cake_fmha_decode_route`` inputs of one shape row.

    A four-page pool is enough: the selector reads shapes, strides and dtypes.
    ``stacked_kv`` hands the K / V views of one ``[pages, 2, ...]`` tensor, the
    layout the aligned program cannot address.
    """

    batch_size = len(seq_lens)
    num_q_heads = GROUP * num_kv_heads if num_q_heads is None else num_q_heads
    dtype = torch.bfloat16
    query = torch.empty((batch_size * q_len, num_q_heads, HEAD_DIM), dtype=dtype)
    if stacked_kv:
        key, value = torch.empty(
            (4, 2, num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype=dtype
        ).unbind(1)
    else:
        key = torch.empty((4, num_kv_heads, PAGE_SIZE, HEAD_DIM), dtype=dtype)
        value = torch.empty_like(key)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32)
    max_seq_len = int(seq_lens_t.max())
    return dict(
        query=query,
        key_cache=key,
        value_cache=value,
        out=torch.empty_like(query),
        workspace_buffer=torch.empty(
            cake_api.cake_fmha_balanced_workspace_bytes(sm_count, q_len),
            dtype=torch.uint8,
        ),
        block_tables=torch.zeros(
            (batch_size, (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE), dtype=torch.int32
        ),
        seq_lens=seq_lens_t,
        batch_size=batch_size,
        q_len=q_len,
        max_seq_len=max_seq_len,
        window_left=-1,
        bmm1_scale=HEAD_DIM**-0.5,
        bmm2_scale=1.0,
        o_scale=1.0,
        sinks=None,
        kv_layout="HND",
        uses_shared_paged_kv_idx=True,
        cum_seq_lens_q=None,
        key_block_scales=None,
        value_block_scales=None,
        skip_softmax_threshold_scale_factor=None,
        enable_block_sparse_attention=False,
        multi_ctas_kv_counter_buffer=torch.zeros(
            cake_api.cake_fmha_balanced_counter_bytes(sm_count, q_len),
            dtype=torch.uint8,
        ),
    )


def _select_route(kwargs):
    return cake_api.select_cake_fmha_decode_route(kwargs["query"].device, **kwargs)


def _patch_host_only(monkeypatch, target, sm_count):
    monkeypatch.setattr(cake_api, "_cake_fmha_target", lambda device: target)
    monkeypatch.setattr(cake_api, "_smallm_sm_count", lambda device: sm_count)
    monkeypatch.setattr(
        cake_api,
        "_is_cake_fmha_decode_native_bf16_available",
        lambda *args, **kwargs: True,
    )


def _expect_row_route(route, *, target, program, num_kv_heads, q_len, seq_lens) -> None:
    assert route is not None
    if program is None:
        # The module's present speculative-decode route, unchanged.
        assert route.component == PRESENT
        assert route.aligned_program is None
        return
    assert route == cake_api.CakeFmhaDecodeRoute(
        target=target,
        batch_size=len(seq_lens),
        q_len=q_len,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        has_sink=False,
        has_window=False,
        use_scale_ptr=False,
        retain_kv_l2=(max(seq_lens) + 127) // 128 <= 9,
        component=COMPONENT,
        page_size=PAGE_SIZE,
        aligned_program=PROGRAMS[program],
    )
    assert cake_api.cake_fmha_route_is_optimized(route)
    assert cake_api._route_components(route) == (PROGRAMS[program],)


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_mtp_q32_route_serves_shape_rows_as_derived(
    monkeypatch, target, row
) -> None:
    arch = TARGET_ARCH[target]
    _patch_host_only(monkeypatch, target, SM_COUNT[arch])
    num_kv_heads, q_len, seq_lens = SHAPE_ROWS[row]
    kwargs = _decode_kwargs(
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
        sm_count=SM_COUNT[arch],
    )
    _expect_row_route(
        _select_route(kwargs),
        target=target,
        program=SEL_SHAPE_ROWS[row][4][arch],
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
    )


@pytest.mark.parametrize("target", sorted(TARGET_ARCH))
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_fmha_aligned_mtp_q32_route_reads_the_device_sm_count(
    monkeypatch, target, row
) -> None:
    """The route takes the program the selector names at the device's SM count:
    on another SM count the trtllm-gen split may pick another cluster size, and
    the present route then serves the launch."""

    arch = TARGET_ARCH[target]
    num_kv_heads, q_len, seq_lens = SHAPE_ROWS[row]
    for sm_count in OTHER_SM_COUNTS:
        _patch_host_only(monkeypatch, target, sm_count)
        kwargs = _decode_kwargs(
            num_kv_heads=num_kv_heads,
            q_len=q_len,
            seq_lens=seq_lens,
            sm_count=sm_count,
        )
        _expect_row_route(
            _select_route(kwargs),
            target=target,
            program=cake_api.spec_q32_cga_program(
                len(seq_lens),
                num_kv_heads,
                max(seq_lens),
                q_len=q_len,
                num_q_heads=GROUP * num_kv_heads,
                arch=arch,
                sm_count=sm_count,
            ),
            num_kv_heads=num_kv_heads,
            q_len=q_len,
            seq_lens=seq_lens,
        )


@pytest.mark.parametrize("program", sorted(PROGRAMS))
def test_cake_fmha_aligned_mtp_q32_route_fails_closed_outside_its_contract(
    monkeypatch, program
) -> None:
    _patch_host_only(monkeypatch, "sm100a", SM_COUNT["sm_100a"])
    # Hold the selector to one program so the contract checks around it decide.
    monkeypatch.setattr(cake_api, "spec_q32_cga_program", lambda *a, **k: program)
    num_kv_heads, q_len, seq_lens = 8, 3, [4096] * 8
    batch_size = len(seq_lens)
    rows = batch_size * q_len
    num_q_heads = GROUP * num_kv_heads
    dtype = torch.bfloat16
    kwargs = _decode_kwargs(
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
        sm_count=SM_COUNT["sm_100a"],
    )
    route = _select_route(kwargs)
    assert route is not None and route.component == COMPONENT
    assert route.aligned_program == PROGRAMS[program]

    stacked = _decode_kwargs(
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
        stacked_kv=True,
        sm_count=SM_COUNT["sm_100a"],
    )
    assert not stacked["key_cache"].is_contiguous()
    pages = kwargs["block_tables"].shape[1]
    for mutation in (
        {"sinks": torch.zeros(num_q_heads, dtype=torch.float32)},
        {"window_left": 128},
        {"bmm1_scale": torch.ones(1, dtype=torch.float32)},
        {"bmm2_scale": 0.5},
        {"bmm2_scale": torch.ones(1, dtype=torch.float32)},
        {"o_scale": 0.5},
        {"kv_layout": "NHD"},
        {"lse": torch.empty((rows, num_q_heads), dtype=torch.float32)},
        {"block_tables": kwargs["block_tables"].to(torch.uint32)},
        {"seq_lens": kwargs["seq_lens"].to(torch.uint32)},
        {"key_cache": stacked["key_cache"], "value_cache": stacked["value_cache"]},
        {"value_cache": stacked["value_cache"]},
        {"workspace_buffer": torch.empty(16, dtype=torch.uint8)},
        {
            "query": torch.empty((rows, num_q_heads, 2 * HEAD_DIM), dtype=dtype)[
                ..., :HEAD_DIM
            ]
        },
        {
            "key_cache": torch.empty((4, num_kv_heads, 32, HEAD_DIM), dtype=dtype),
            "value_cache": torch.empty((4, num_kv_heads, 32, HEAD_DIM), dtype=dtype),
        },
        {
            "uses_shared_paged_kv_idx": False,
            "block_tables": torch.zeros((batch_size, 2, pages), dtype=torch.int32),
        },
        {
            "key_block_scales": torch.ones(
                (4, num_kv_heads, PAGE_SIZE), dtype=torch.float32
            )
        },
        {
            "q_len": 2,
            "query": torch.empty((2 * batch_size, num_q_heads, HEAD_DIM), dtype=dtype),
            "out": torch.empty((2 * batch_size, num_q_heads, HEAD_DIM), dtype=dtype),
        },
        {
            "query": torch.empty((rows, num_q_heads // 2, HEAD_DIM), dtype=dtype),
            "out": torch.empty((rows, num_q_heads // 2, HEAD_DIM), dtype=dtype),
        },
        {
            "query": torch.empty((rows, 2 * num_q_heads, HEAD_DIM), dtype=dtype),
            "out": torch.empty((rows, 2 * num_q_heads, HEAD_DIM), dtype=dtype),
        },
    ):
        mutated = {**kwargs, **mutation}
        selected = _select_route(mutated)
        assert selected is None or selected.component != COMPONENT, mutation
        if selected is not None:
            assert selected.aligned_program is None, mutation


# --- GPU: the shape rows through backend="cake" -----------------------------

_requires_blackwell = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Cake FMHA requires SM100 or SM103",
)


def _mtp_case(device, *, num_q_heads, num_kv_heads, q_len, seq_lens, seed):
    """Query ``[batch x q_len, num_q_heads, 128]`` (``randn x 3``: a peaked
    softmax, so ``|O|`` sits well above the tolerance), separate contiguous K / V
    pools ``[pages, num_kv_heads, 16, 128]`` (``randn``; the layout the aligned
    program reads in place), a shuffled page table and the device lengths."""

    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(seq_lens)
    pages_per_req = [(s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens]
    total_pages = sum(pages_per_req)
    query = (
        torch.randn(
            (batch * q_len, num_q_heads, HEAD_DIM), device=device, generator=generator
        )
        * 3
    ).to(torch.bfloat16)
    pool_shape = (total_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM)
    key = torch.randn(
        pool_shape, device=device, generator=generator, dtype=torch.bfloat16
    )
    value = torch.randn(
        pool_shape, device=device, generator=generator, dtype=torch.bfloat16
    )
    order = torch.randperm(total_pages, device=device, generator=generator)
    block_tables = torch.zeros(
        (batch, max(pages_per_req)), dtype=torch.int32, device=device
    )
    cursor = 0
    for b, n in enumerate(pages_per_req):
        block_tables[b, :n] = order[cursor : cursor + n]
        cursor += n
    return (
        query,
        key,
        value,
        block_tables,
        torch.tensor(seq_lens, dtype=torch.int32, device=device),
    )


def _reference_decode(query, key, value, block_tables, seq_lens, *, q_len, scale):
    """FP32 bottom-right-causal GQA decode over the paged HND pools: token ``j``
    of request ``b`` attends the first ``seq_lens[b] - (q_len - 1 - j)`` tokens,
    query head ``h`` reads KV head ``h // group``."""

    batch = int(seq_lens.numel())
    num_q_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    group = num_q_heads // num_kv_heads
    out = torch.empty(query.shape, dtype=torch.float32, device=query.device)
    for b in range(batch):
        length = int(seq_lens[b])
        pages = block_tables[b, : (length + PAGE_SIZE - 1) // PAGE_SIZE]
        k = key[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        v = value[pages].permute(1, 0, 2, 3).reshape(num_kv_heads, -1, HEAD_DIM)
        k = k[:, :length].float()
        v = v[:, :length].float()
        for j in range(q_len):
            visible = length - (q_len - 1 - j)
            row = b * q_len + j
            q = query[row].float().view(num_kv_heads, group, HEAD_DIM)
            logits = torch.einsum("hgd,hnd->hgn", q, k[:, :visible]) * scale
            probs = torch.softmax(logits, dim=-1)
            out[row] = torch.einsum("hgn,hnd->hgd", probs, v[:, :visible]).reshape(
                num_q_heads, HEAD_DIM
            )
    return out


def _shorter_lengths(seq_lens, q_len):
    """A second length vector under the same captured KV bound: the longest
    request keeps its length, every other request loses a third (never below
    ``q_len`` tokens)."""

    shorter = [max(q_len, s - s // 3) for s in seq_lens]
    longest = max(range(len(seq_lens)), key=seq_lens.__getitem__)
    shorter[longest] = seq_lens[longest]
    return shorter


def _device_sm_count(device) -> int:
    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _check_row(
    monkeypatch,
    *,
    query,
    key,
    value,
    block_tables,
    seq_lens_t,
    seq_lens,
    q_len,
    expect_program,
):
    """One shape row through ``backend="cake"``: the route taken, the FP32
    reference at the tolerance with a non-vacuous reference, and CUDA-graph
    replay -- bitwise equal to the eager result on the aligned program (the
    present route is held to the tolerance), then under shorter lengths."""

    device = query.device
    scale = HEAD_DIM**-0.5
    max_seq_len = max(seq_lens)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    bitwise = expect_program is not None

    # Record the route the dispatch resolves (decode.py imports the resolver
    # from the module at call time).
    resolved = []
    resolve = cake_api._resolve_cake_fmha_decode_module

    def spy(resolve_device, route):
        resolved.append(route)
        return resolve(resolve_device, route)

    monkeypatch.setattr(cake_api, "_resolve_cake_fmha_decode_module", spy)

    def run():
        return trtllm_batch_decode_with_kv_cache(
            query,
            (key, value),
            workspace,
            block_tables,
            seq_lens_t,
            max_seq_len,
            bmm1_scale=scale,
            bmm2_scale=1.0,
            window_left=-1,
            kv_layout="HND",
            q_len_per_req=q_len,
            backend="cake",
        )

    def reference():
        return _reference_decode(
            query, key, value, block_tables, seq_lens_t, q_len=q_len, scale=scale
        )

    actual = run()
    torch.cuda.synchronize()
    route = resolved[-1]
    assert route is not None
    if bitwise:
        assert route.component == COMPONENT
        assert route.aligned_program == PROGRAMS[expect_program]
    else:
        assert route.component == PRESENT
        assert route.aligned_program is None
    expected = reference()
    assert float(expected.abs().amax()) > MIN_REF_ABSMAX
    torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)

    def check_replay(replayed, wanted):
        if bitwise:
            assert torch.equal(replayed, actual)
        torch.testing.assert_close(replayed.float(), wanted, atol=ATOL, rtol=RTOL)

    # Graph replay: the captured launch must reproduce the eager result and,
    # with shorter lengths under the same KV bound, follow the device lengths
    # without a re-plan.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run()
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    check_replay(replayed, expected)
    shorter = _shorter_lengths(seq_lens, q_len)
    seq_lens_t.copy_(torch.tensor(shorter, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    expected_short = reference()
    assert float(expected_short.abs().amax()) > MIN_REF_ABSMAX
    torch.testing.assert_close(replayed.float(), expected_short, atol=ATOL, rtol=RTOL)
    seq_lens_t.copy_(torch.tensor(seq_lens, dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    check_replay(replayed, expected)


@_requires_blackwell
@pytest.mark.parametrize("row", sorted(SHAPE_ROWS))
def test_cake_decode_aligned_mtp_q32_shape_rows_match_reference_and_replay(
    monkeypatch, row
) -> None:
    device = torch.device("cuda")
    arch = TARGET_ARCH[cake_api._cake_fmha_target(device)]
    sm_count = _device_sm_count(device)
    num_kv_heads, q_len, seq_lens = SHAPE_ROWS[row]
    # The program the selector names at this device's SM count: the derivation's
    # verdict on a device of the SM count of record.
    program = cake_api.spec_q32_cga_program(
        len(seq_lens),
        num_kv_heads,
        max(seq_lens),
        q_len=q_len,
        num_q_heads=GROUP * num_kv_heads,
        arch=arch,
        sm_count=sm_count,
    )
    if sm_count == SM_COUNT[arch]:
        assert program == SEL_SHAPE_ROWS[row][4][arch]
    query, key, value, block_tables, seq_lens_t = _mtp_case(
        device,
        num_q_heads=GROUP * num_kv_heads,
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads + q_len,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        q_len=q_len,
        expect_program=program,
    )


@_requires_blackwell
@pytest.mark.parametrize("program", sorted(PROGRAMS))
@pytest.mark.parametrize("case", sorted(CASE_ROWS))
def test_cake_decode_aligned_mtp_q32_kernel_cases_match_reference_on_the_program(
    monkeypatch, case, program
) -> None:
    """The kernel tests' cases of 3 and 4 query tokens on the program itself:
    the selector is held to the program so the small shapes (ragged lengths,
    page tails, a request of exactly q_len tokens, one active cluster rank, a
    full 32-row tile) exercise it."""

    device = torch.device("cuda")
    num_q_heads, num_kv_heads, q_len, seq_lens = CASE_ROWS[case]
    monkeypatch.setattr(cake_api, "spec_q32_cga_program", lambda *a, **k: program)
    query, key, value, block_tables, seq_lens_t = _mtp_case(
        device,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        q_len=q_len,
        seq_lens=seq_lens,
        seed=20261102 + len(seq_lens) + num_kv_heads + q_len,
    )
    _check_row(
        monkeypatch,
        query=query,
        key=key,
        value=value,
        block_tables=block_tables,
        seq_lens_t=seq_lens_t,
        seq_lens=seq_lens,
        q_len=q_len,
        expect_program=program,
    )
