# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests of the generated programs behind ``minimax_h3_sm120_varlen_attention_fp8`` (no CUDA
device): each program constant of ``flashinfer.jit.cake_minimax_h3_sm120_quant_varlen_attention``
matches its two translation units (the kernel symbol the device unit defines and the binding declares,
the binding namespace, the exported ``run`` entry, the FFI checks header), its digests follow from the
file bytes, its JIT spec carries the target flags, ``--fmad=false`` and the recorded options, and the
host module's grid rules, its choice of the overlapped form and its log2-domain scale are what the
kernels expect."""

import hashlib
import math
import re

import pytest
import torch

from flashinfer.diffusion_ops import cake_minimax_h3_sm120_quant_varlen_attention as ops
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_sm120_quant_varlen_attention import (
    ATTENTION,
    ATTENTION_LEAD,
    ATTENTION_TAIL,
    KV_STATS,
    PREPROCESS_HEAD_MAJOR,
    PREPROCESS_HEAD_MAJOR_BURST,
    PROGRAMS,
    QUANTIZE,
)
from flashinfer.jit.core import sm120a_nvcc_flags

NAMED = {
    "KV_STATS": KV_STATS,
    "QUANTIZE": QUANTIZE,
    "ATTENTION": ATTENTION,
    "PREPROCESS_HEAD_MAJOR": PREPROCESS_HEAD_MAJOR,
    "PREPROCESS_HEAD_MAJOR_BURST": PREPROCESS_HEAD_MAJOR_BURST,
    "ATTENTION_LEAD": ATTENTION_LEAD,
    "ATTENTION_TAIL": ATTENTION_TAIL,
}
DYNAMIC_SMEM = {
    "KV_STATS": 16512,
    "QUANTIZE": 128,
    "ATTENTION": 82944,
    "PREPROCESS_HEAD_MAJOR": 33024,
    "PREPROCESS_HEAD_MAJOR_BURST": 33024,
    "ATTENTION_LEAD": 82944,
    "ATTENTION_TAIL": 82944,
}
# One CTA of two 256-thread workers per SM for the producers, 256 threads for every other program.
BLOCK = {
    name: 512 if name.startswith("PREPROCESS_HEAD_MAJOR") else 256 for name in NAMED
}
# The overlapped form's lead and tail attention are programmatic dependents of the launch before
# them; the sequential launches and the producers start after every earlier launch of the stream.
PDL = {"ATTENTION_LEAD", "ATTENTION_TAIL"}
STATS_ARGS = (
    "K",
    "V",
    "tile_table",
    "seg_begin",
    "seg_len",
    "seg_tile_begin",
    "partials",
    "counters",
    "mean_k",
    "v_scale",
    "num_tiles",
    "num_heads",
)
QUANTIZE_ARGS = (
    "Q",
    "K",
    "V",
    "cu_seqlens",
    "mean_k",
    "v_scale",
    "q8",
    "k8",
    "vt8",
    "q_scale",
    "k_scale",
    "total_tokens",
    "total_padded",
    "num_segments",
    "num_heads",
    "num_kblocks",
)
ATTENTION_ARGS = (
    "Q_map",
    "K_map",
    "V_map",
    "O",
    "q_scale",
    "k_scale",
    "v_scale",
    "seg_begin",
    "seg_len",
    "unit_table",
    "total_units",
    "num_heads",
    "num_kblocks",
    "softmax_scale_log2",
)
PREPROCESS_HEAD_MAJOR_ARGS = (
    "Q",
    "K",
    "V",
    "tile_table",
    "seg_begin",
    "seg_len",
    "seg_tile_begin",
    "cu_seqlens",
    "partials",
    "counters",
    "mean_k",
    "v_scale",
    "q8",
    "k8",
    "vt8",
    "q_scale",
    "k_scale",
    "hctr",
    "num_tiles",
    "total_tokens",
    "total_padded",
    "num_segments",
    "num_heads",
    "num_kblocks",
)
ATTENTION_POLL_ARGS = (*ATTENTION_ARGS, "hctr", "lead_ctas", "tail_ctas")
HOST_ARGS = {
    "KV_STATS": STATS_ARGS,
    "QUANTIZE": QUANTIZE_ARGS,
    "ATTENTION": ATTENTION_ARGS,
    "PREPROCESS_HEAD_MAJOR": PREPROCESS_HEAD_MAJOR_ARGS,
    "PREPROCESS_HEAD_MAJOR_BURST": PREPROCESS_HEAD_MAJOR_ARGS,
    "ATTENTION_LEAD": ATTENTION_POLL_ARGS,
    "ATTENTION_TAIL": ATTENTION_POLL_ARGS,
}


def test_the_loader_publishes_the_seven_programs() -> None:
    assert tuple(NAMED.values()) == PROGRAMS
    assert len({program.name for program in PROGRAMS}) == 7
    for identifier, program in NAMED.items():
        assert program.name.startswith("cake_minimax_h3_sm120_quant_varlen_attention_")
        assert program.operator_dir == "cake_minimax_h3_sm120_quant_varlen_attention"
        assert program.targets == ("sm_120a",)
        assert program.kernel == f"kernel_{program.name}"
        assert program.args == (*HOST_ARGS[identifier], *cake_program.GRID_ARGS)
        assert set(program.tensors) <= set(program.args)
        assert program.launch_record() == {
            "block": (BLOCK[identifier], 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": DYNAMIC_SMEM[identifier],
            "use_pdl": identifier in PDL,
        }
        assert program.needs_device_linking is False


@pytest.mark.parametrize("identifier", sorted(NAMED))
def test_each_program_matches_its_units(identifier: str) -> None:
    program = NAMED[identifier]
    kernel_path, binding_path = program.source_paths()
    assert kernel_path.name == f"{program.name}_kernel.cu"
    assert binding_path.name == f"{program.name}_binding.cu"
    kernel, binding = kernel_path.read_bytes(), binding_path.read_bytes()
    assert re.search(rb"__global__[^;{]*\b" + program.kernel.encode() + rb"\b", kernel)
    assert program.kernel.encode() in binding
    assert f"namespace cake_binding_{program.kernel}".encode() in binding
    assert b"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding
    assert b"cake_ffi_checks.h" in binding
    # the units end in exactly one newline (FlashInfer's end-of-file hook must be a no-op)
    for data in (kernel, binding):
        assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    for target in program.targets:
        options = program.options_for(target)
        digest = hashlib.sha256()
        digest.update(kernel)
        digest.update(binding)
        digest.update(
            cake_program.json.dumps(
                {"options": list(options), "target": target},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        expected = digest.hexdigest()[: cake_program.DIGEST_DIGITS]
        assert program.digest(target) == expected
        assert program.module_name(target) == f"{program.name}_{target}_{expected}"
        spec = program.spec(target)
        flags = [str(flag) for flag in spec.extra_cuda_cflags]
        tail = [*sm120a_nvcc_flags, "--fmad=false", *options]
        assert flags[-len(tail) :] == tail
        assert "-use_fast_math" not in flags and "--use_fast_math" not in flags
        assert spec.name == program.module_name(target)
    with pytest.raises(KeyError):
        program.options_for("sm_100a")


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 0), "sm_120a"), ((12, 1), None), ((10, 0), None), ((9, 0), None)],
)
def test_exact_architecture_router(capability, expected) -> None:
    for program in PROGRAMS:
        assert program.target_for(capability) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    ops._fp8_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 1))
    with pytest.raises(RuntimeError, match="exact compute capability 12.0"):
        ops._fp8_target(0)
    ops._fp8_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    assert ops._fp8_target(0) == "sm_120a"
    ops._fp8_target.cache_clear()


@pytest.mark.parametrize(
    ("num_sms", "heads", "facts", "expected"),
    [
        (188, 1, {"num_tiles": 1, "num_kblocks": 1, "num_units": 1}, (1, 1, 1)),
        (188, 3, {"num_tiles": 5, "num_kblocks": 4, "num_units": 15}, (15, 12, 15)),
        (
            170,
            56,
            {"num_tiles": 262, "num_kblocks": 262, "num_units": 14672},
            (680, 680, 170),
        ),
        (
            188,
            56,
            {"num_tiles": 262, "num_kblocks": 262, "num_units": 14672},
            (752, 752, 188),
        ),
        (188, 56, {"num_tiles": 262}, (752, None, None)),
        (188, 56, {"num_kblocks": 262}, (None, 752, None)),
        (188, 56, {"num_units": 14672}, (None, None, 188)),
    ],
)
def test_launch_grids_follow_the_plan_facts(num_sms, heads, facts, expected) -> None:
    grids = ops.minimax_h3_sm120_quant_varlen_attention_launch_grids(
        num_sms, num_heads=heads, **facts
    )
    assert grids == expected
    with pytest.raises(ValueError):
        ops.minimax_h3_sm120_quant_varlen_attention_launch_grids(
            0, num_heads=heads, **facts
        )


def test_attention_grid_is_the_plan_grid() -> None:
    # The unit table's slot order is laid out for the plan's grid; the launch takes the same count.
    bounds = ops.normalize_cu_seqlens((0, 0, 129, 129, 500, 4824))
    for num_sms in (1, 7, 170, 188):
        plan = ops.MiniMaxH3VarlenPlan(bounds, 5, torch.device("cpu"), num_sms)
        grids = ops.minimax_h3_sm120_quant_varlen_attention_launch_grids(
            num_sms, num_heads=5, num_units=plan.num_units
        )
        assert grids[2] == plan.grid


@pytest.mark.parametrize(
    ("num_sms", "num_units", "facts", "expected"),
    [
        # width 32 on 188 SMs: the burst producers on every SM
        (188, 1792, None, (156, 188, 32)),
        (188, 0, None, (1, 188, 32)),
        # width 32 on 170 SMs: the default producers (two workers per CTA) over at most twice the
        # width on the SMs the lead leaves
        (170, 1792, None, (138, 32, 32)),
        (170, 16, None, (16, 64, 32)),
        # width 64 (a light plan: 4 x 1536 tokens): the producers (two workers per CTA) over at
        # most twice the width on the SMs the lead leaves
        (170, 1792, (6144, 4 * 1536 * 1536, 6144), (106, 64, 64)),
        (188, 1792, (6144, 4 * 1536 * 1536, 6144), (124, 64, 64)),
        (170, 16, (6144, 4 * 1536 * 1536, 6144), (16, 128, 64)),
        (132, 1792, None, None),
    ],
)
def test_overlap_grids_take_one_producer_cta_per_free_sm(
    num_sms, num_units, facts, expected
) -> None:
    # (lead, producers, tail): the lead over the SMs the overlap width leaves, the tail over the
    # width; the producers over every SM for width 32 on 188 SMs (the burst build), else over at
    # most twice the width on the SMs the lead leaves
    tokens, work, padded = facts or (None, None, None)
    grids = ops.minimax_h3_sm120_quant_varlen_attention_overlap_grids(
        num_sms,
        num_units=num_units,
        total_tokens=tokens,
        work=work,
        padded_tokens=padded,
    )
    assert grids == expected
    if grids is not None:
        lead, producers, tail = grids
        if num_sms == 188 and tail == 32:
            assert producers == num_sms
        else:
            assert lead + producers <= num_sms
    with pytest.raises(ValueError):
        ops.minimax_h3_sm120_quant_varlen_attention_overlap_grids(0, num_units=1)


@pytest.mark.parametrize(
    ("bounds", "num_sms", "width"),
    [
        # attention-weighted mean segment length (work / tokens) at most 1792 on 170 SMs and 2304 on
        # 188 SMs: width 64
        ((0, 1536, 3072, 4608, 6144), 170, 64),
        ((0, 1536, 3072, 4608, 6144), 188, 64),
        ((0, 1792, 3584, 5376, 7168), 170, 64),
        ((0, 1793, 3586, 5379, 7172), 170, 32),
        ((0, 2048, 4096), 188, 64),
        ((0, 2048, 4096), 170, 32),
        ((0, 2304, 4608), 188, 64),
        ((0, 2305, 4610), 188, 32),
        # a mean of at least 4096 over more than 16384 padded tokens: 64 on 170 SMs, 48 on 188 SMs
        ((0, 8368, 16736, 25104, 33472), 170, 64),
        ((0, 8368, 16736, 25104, 33472), 188, 48),
        ((0, 4096, 8192, 12288, 16384, 20480), 188, 48),
        ((0, 4095, 8190, 12285, 16380, 20475), 188, 32),
        ((0, 16385), 188, 48),
        # the rest keep the shipped width
        ((0, 16384), 170, 32),
        ((0, 257, 4567, 4824), 188, 32),
        ((0, 4096), 170, 32),
    ],
)
def test_overlap_width_follows_the_plan(bounds, num_sms, width) -> None:
    plan = ops.MiniMaxH3VarlenPlan(bounds, 56, torch.device("cpu"), num_sms)
    assert plan.overlap
    grids = ops.minimax_h3_sm120_quant_varlen_attention_overlap_grids(
        num_sms,
        num_units=plan.num_units,
        total_tokens=plan.total_tokens,
        work=plan.work,
        padded_tokens=plan.padded_tokens,
    )
    lead = max(1, min(num_sms - width, plan.num_units))
    burst = num_sms == 188 and width == 32
    producers = num_sms if burst else min(2 * width, num_sms - lead)
    assert grids == (lead, producers, width)
    expected = PREPROCESS_HEAD_MAJOR_BURST if burst else PREPROCESS_HEAD_MAJOR
    assert ops.minimax_h3_sm120_quant_varlen_attention_programs(plan)[0] is expected


@pytest.mark.parametrize(
    ("bounds", "heads", "num_sms", "overlap"),
    [
        # the band's edges: 2 x 2048 (2^23) and one 24576-token segment (24576^2)
        ((0, 2048, 4096), 56, 170, "default"),
        ((0, 2047, 4094), 56, 170, None),
        ((0, 24576), 56, 188, "default"),
        ((0, 24577), 56, 188, "head_major"),
        ((0, 257, 4567, 4824), 56, 188, "burst"),
        ((0, 8368, 16736, 25104, 33472), 56, 170, "default"),
        # other head counts and SM counts keep the sequential launches
        ((0, 257, 4567, 4824), 8, 170, None),
        ((0, 257, 4567, 4824), 56, 132, None),
        ((0, 0, 129, 129, 500), 56, 170, None),
        # above the band: 56 heads on 170 or 188 SMs take the head-major producers, other head and
        # SM counts the statistics and quantizer launches
        ((0, 33472), 56, 170, "head_major"),
        ((0, 3000, 44000, 53000), 56, 188, "head_major"),
        ((0, 24577), 8, 188, None),
        ((0, 24577), 56, 132, None),
        # past the token bound (65536 on 170 SMs, 98304 on 188 SMs) the statistics and quantizer
        # launches again
        ((0, 65536), 56, 170, "head_major"),
        ((0, 65537), 56, 170, None),
        ((0, 65537), 56, 188, "head_major"),
        ((0, 98304), 56, 188, "head_major"),
        ((0, 98305), 56, 188, None),
    ],
)
def test_the_overlapped_form_serves_the_overlap_band(
    bounds, heads, num_sms, overlap
) -> None:
    plan = ops.MiniMaxH3VarlenPlan(bounds, heads, torch.device("cpu"), num_sms)
    lengths = [b - a for a, b in zip(bounds, bounds[1:], strict=False)]
    assert plan.work == sum(length * length for length in lengths)
    assert plan.overlap is (overlap not in (None, "head_major"))
    assert plan.head_major is (overlap == "head_major")
    programs = ops.minimax_h3_sm120_quant_varlen_attention_programs(plan)
    if overlap == "head_major":
        # the head-major producers over every SM, then the sequential attention
        assert programs == (PREPROCESS_HEAD_MAJOR, ATTENTION)
    elif overlap:
        # launch order: producers (plain; the burst build for a plan of overlap width 32 on 188
        # SMs), lead attention, tail attention
        producers = (
            PREPROCESS_HEAD_MAJOR_BURST if overlap == "burst" else PREPROCESS_HEAD_MAJOR
        )
        assert programs == (producers, ATTENTION_LEAD, ATTENTION_TAIL)
    else:
        assert programs == (KV_STATS, QUANTIZE, ATTENTION)


def test_head_major_unit_table_publishes_heads_in_order() -> None:
    # Every unit once, head by head; within a head the longest units first, equal costs in segment
    # and tile order.
    bounds = (0, 257, 4567, 4824)
    plan = ops.MiniMaxH3VarlenPlan(bounds, 56, torch.device("cpu"), 170)
    assert plan.overlap
    table = plan.unit_table_head_major.view(-1, 2).tolist()
    assert len(table) == plan.num_units
    units = [(seg, packed >> 16, packed & 0xFFFF) for seg, packed in table]
    assert sorted(units) == sorted(
        (seg, packed >> 16, packed & 0xFFFF)
        for seg, packed in plan.unit_table.view(-1, 2).tolist()
    )
    per_head = plan.num_units // 56
    assert [head for _seg, head, _t in units] == [
        head for head in range(56) for _ in range(per_head)
    ]
    # segment 1 (4310 tokens, 34 tiles) is the most expensive; segments 0 and 2 (257 tokens, 3
    # tiles each) tie
    first_head = [(seg, t) for seg, _head, t in units[:per_head]]
    assert first_head == [(1, t) for t in range(34)] + [
        (seg, t) for seg in (0, 2) for t in range(3)
    ]
    for head in range(1, 56):
        chunk = units[head * per_head : (head + 1) * per_head]
        assert [(seg, t) for seg, _head, t in chunk] == first_head


def test_overlapped_form_warms_up_once_outside_capture(monkeypatch) -> None:
    # The producers, then the lead and the tail attention.  The first call on a device whose own
    # stream takes part in no graph capture first runs the three programs once with no work on the
    # call's head-control words; a call whose stream is capturing runs none.  The capture query
    # asks about the call's own device and stream.
    launches = []
    selected = []

    def entries(target: str, burst: bool):
        selected.append((target, burst))
        return tuple(
            lambda *args, _slot=slot: launches.append((_slot, args))
            for slot in ("producers", "lead", "tail")
        )

    monkeypatch.setattr(ops, "_overlap_entries", entries)
    monkeypatch.setattr(ops, "_OVERLAP_LOADED", set())
    queries = []
    capturing = [True]

    def query(index: int, stream: int) -> bool:
        queries.append((index, stream))
        return capturing[0]

    monkeypatch.setattr(ops, "_capturing", query)
    producer_tensors = ("q", "k", "v", "hctr")
    attention_tensors = ("q8_rows", "out", "unit_table_head_major")
    # statistics tiles, tokens, padded tokens, segments, heads, 128-key blocks
    producer_facts = (5, 2100, 2176, 3, 56, 17)
    # units, heads, 128-key blocks, log2-domain scale, head-control words
    attention_facts = (51, 56, 17, 0.25, "hctr")

    def call(grids=(106, 64, 64)) -> None:
        ops._launch_overlapped(
            "sm_120a",
            1,
            4321,
            producer_tensors,
            producer_facts,
            attention_tensors,
            attention_facts,
            grids,
        )

    work = [
        ("producers", (*producer_tensors, *producer_facts, 64, 1, 1)),
        ("lead", (*attention_tensors, *attention_facts, 106, 64, 106, 1, 1)),
        ("tail", (*attention_tensors, *attention_facts, 106, 64, 64, 1, 1)),
    ]
    no_work = [
        ("producers", (*producer_tensors, 0, 2100, 2176, 1, 1, 0, 1, 1, 1)),
        ("lead", (*attention_tensors, 0, 1, 1, 0.25, "hctr", 1, 1, 1, 1, 1)),
        ("tail", (*attention_tensors, 0, 1, 1, 0.25, "hctr", 1, 1, 1, 1, 1)),
    ]
    call()  # the call's own stream is capturing: the call's launches only
    assert launches == work
    launches.clear()
    capturing[0] = False
    call()  # the first eager call: the no-work instance, then the call
    assert launches == no_work + work
    launches.clear()
    call()
    assert launches == work
    assert queries == [(1, 4321), (1, 4321)]
    assert selected == [("sm_120a", False)] * 3
    # the burst producers (188 SMs, a tail of 32) are a module of their own: their first eager
    # call warms them up once more
    launches.clear()
    call((156, 188, 32))
    burst_work = [
        ("producers", (*producer_tensors, *producer_facts, 188, 1, 1)),
        ("lead", (*attention_tensors, *attention_facts, 156, 32, 156, 1, 1)),
        ("tail", (*attention_tensors, *attention_facts, 156, 32, 32, 1, 1)),
    ]
    assert launches == no_work + burst_work
    assert selected[-1] == ("sm_120a", True)


@pytest.mark.parametrize(
    ("grids", "burst"),
    [
        # 188 SMs, width 32: the burst producers on every SM
        ((156, 188, 32), True),
        ((1, 188, 32), True),
        # 170 SMs at width 32, and the light and long widths: the default producers
        ((138, 32, 32), False),
        ((16, 64, 32), False),
        ((106, 64, 64), False),
        ((124, 64, 64), False),
        ((140, 48, 48), False),
        ((140, 96, 48), False),
    ],
)
def test_burst_route_follows_the_overlap_grids(monkeypatch, grids, burst) -> None:
    # The route: the grids ``overlap_grids`` gives a plan of width 32 on 188 SMs launch
    # PREPROCESS_HEAD_MAJOR_BURST over their producer grid (every SM); every other plan keeps
    # PREPROCESS_HEAD_MAJOR.
    selected = []
    monkeypatch.setattr(
        ops,
        "_overlap_entries",
        lambda target, b: selected.append(b) or (lambda *a: None,) * 3,
    )
    monkeypatch.setattr(ops, "_capturing", lambda *_: True)
    ops._launch_overlapped(
        "sm_120a",
        0,
        1,
        (),
        (5, 2100, 2176, 3, 56, 17),
        (),
        (51, 56, 17, 0.25, "h"),
        grids,
    )
    assert selected == [burst]
    assert ops._overlap_programs(burst)[0] is (
        PREPROCESS_HEAD_MAJOR_BURST if burst else PREPROCESS_HEAD_MAJOR
    )


def test_burst_grids_need_the_tail_of_their_build() -> None:
    # A producer grid above two workers' CTAs per tail CTA is the burst launch, valid only beside
    # a tail of the burst build's P.
    with pytest.raises(AssertionError):
        ops._overlap_burst((140, 188, 48))
    with pytest.raises(AssertionError):
        ops._overlap_burst((106, 188, 64))


def test_burst_build_stops_ctas_from_the_host_width() -> None:
    # The burst build's P (its CTAs from bid P on stop early) is the host's width of the burst
    # route: the device unit compares the CTA index with that literal; the default producers do
    # not.
    kernel = PREPROCESS_HEAD_MAJOR_BURST.source_paths()[0].read_text()
    default = PREPROCESS_HEAD_MAJOR.source_paths()[0].read_text()
    p = ops._OVERLAP_BURST_P
    assert p == 32
    assert ops._OVERLAP_BURST_WIDTH == {188: 32}
    assert len(re.findall(rf"\bbid >= {p}\b", kernel)) == 2  # the two workers' claims
    assert not re.search(r"\bbid >= ", default)


def test_softmax_scale_log2_rounds_as_the_kernel_takes_it() -> None:
    scale = 1.0 / math.sqrt(128)
    value = ops._softmax_scale_log2(scale)
    assert value == scale * 1.4426950408889634
    assert math.log2(math.e) == ops._LOG2E
    assert (
        torch.tensor(value, dtype=torch.float32).item()
        == torch.tensor(scale * math.log2(math.e), dtype=torch.float32).item()
    )


def test_public_names_are_exported() -> None:
    for name in (
        "minimax_h3_sm120_quant_varlen_attention_launch_grids",
        "minimax_h3_sm120_quant_varlen_attention_overlap_grids",
        "minimax_h3_sm120_quant_varlen_attention_programs",
        "minimax_h3_sm120_varlen_attention_fp8",
        "workspace_bytes",
    ):
        assert name in ops.__all__ and callable(getattr(ops, name))
