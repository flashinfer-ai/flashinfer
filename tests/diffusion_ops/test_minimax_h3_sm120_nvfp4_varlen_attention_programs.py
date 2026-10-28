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
"""CPU tests of the generated programs behind ``minimax_h3_sm120_varlen_attention_nvfp4`` (``pv_mode="fp4"``)
and ``minimax_h3_sm120_varlen_attention_nvfp4_nodelta`` (no CUDA device): each program constant of
``flashinfer.jit.cake_minimax_h3_sm120_nvfp4_varlen_attention`` matches its two translation units (the kernel
symbol the device unit defines and the binding declares, the binding namespace, the exported ``run`` entry,
the FFI checks header), its digests follow from the file bytes, its JIT spec carries the target flags,
``--fmad=false`` and the recorded options, and the host module's program routing, grid rules (the
overlapped form's included), head-major unit rows and Q slot count are what the kernels expect; the
overlapped form launches its producers first, then the lead and the tail attention, and the shared host
helpers key workspace sets by owner (stream, and capture under CUDA-graph capture) and lock per stream."""

import contextlib
import hashlib
import math
import re
import types

import pytest
import torch

from flashinfer.diffusion_ops import cake_minimax_h3_sm120_nvfp4_varlen_attention as ops
from flashinfer.diffusion_ops import (
    cake_minimax_h3_sm120_quant_varlen_attention as quant_ops,
)
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
)
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    ATTENTION,
    ATTENTION_LEAD,
    ATTENTION_NODELTA,
    ATTENTION_NODELTA_LEAD,
    ATTENTION_NODELTA_TAIL,
    ATTENTION_SFSEL_LEAD,
    ATTENTION_SFSEL_TAIL,
    ATTENTION_TAIL,
    ATTENTION_WS,
    ATTENTION_WS_NODELTA,
    ATTENTION_WS_NODELTA_LEAD,
    ATTENTION_WS_NODELTA_TAIL,
    PREPROCESS,
    PREPROCESS_HEAD_MAJOR,
    PREPROCESS_HEAD_MAJOR_KA,
    PROGRAMS,
    QUANTIZE_KV,
    QUANTIZE_KV_HBLK64,
    STATS,
)
from flashinfer.jit.core import sm120a_nvcc_flags

NAMED = {
    "PREPROCESS": PREPROCESS,
    "STATS": STATS,
    "QUANTIZE_KV": QUANTIZE_KV,
    "QUANTIZE_KV_HBLK64": QUANTIZE_KV_HBLK64,
    "ATTENTION": ATTENTION,
    "ATTENTION_NODELTA": ATTENTION_NODELTA,
    "ATTENTION_WS": ATTENTION_WS,
    "ATTENTION_WS_NODELTA": ATTENTION_WS_NODELTA,
    "PREPROCESS_HEAD_MAJOR": PREPROCESS_HEAD_MAJOR,
    "ATTENTION_LEAD": ATTENTION_LEAD,
    "ATTENTION_TAIL": ATTENTION_TAIL,
    "ATTENTION_NODELTA_LEAD": ATTENTION_NODELTA_LEAD,
    "ATTENTION_NODELTA_TAIL": ATTENTION_NODELTA_TAIL,
    "ATTENTION_WS_NODELTA_LEAD": ATTENTION_WS_NODELTA_LEAD,
    "ATTENTION_WS_NODELTA_TAIL": ATTENTION_WS_NODELTA_TAIL,
    "PREPROCESS_HEAD_MAJOR_KA": PREPROCESS_HEAD_MAJOR_KA,
    "ATTENTION_SFSEL_LEAD": ATTENTION_SFSEL_LEAD,
    "ATTENTION_SFSEL_TAIL": ATTENTION_SFSEL_TAIL,
}
# The overlapped form's producers: its first launch, a plain launch (no programmatic dependent launch).
PLAIN = ("PREPROCESS_HEAD_MAJOR", "PREPROCESS_HEAD_MAJOR_KA")
DYNAMIC_SMEM = {
    "PREPROCESS": 16512,
    "STATS": 17024,
    "QUANTIZE_KV": 0,
    "QUANTIZE_KV_HBLK64": 0,
    "ATTENTION": 66560,
    "ATTENTION_NODELTA": 48128,
    "ATTENTION_WS": 80896,
    "ATTENTION_WS_NODELTA": 80896,
    "PREPROCESS_HEAD_MAJOR": 49536,
    "ATTENTION_LEAD": 66560,
    "ATTENTION_TAIL": 66560,
    "ATTENTION_NODELTA_LEAD": 48128,
    "ATTENTION_NODELTA_TAIL": 48128,
    "ATTENTION_WS_NODELTA_LEAD": 80896,
    "ATTENTION_WS_NODELTA_TAIL": 80896,
    "PREPROCESS_HEAD_MAJOR_KA": 49536,
    "ATTENTION_SFSEL_LEAD": 66560,
    "ATTENTION_SFSEL_TAIL": 66560,
}
THREADS = {identifier: 256 for identifier in NAMED} | {
    "ATTENTION_WS": 384,
    "ATTENTION_WS_NODELTA": 384,
    "ATTENTION_WS_NODELTA_LEAD": 384,
    "ATTENTION_WS_NODELTA_TAIL": 384,
    # three 256-thread producer workers per CTA
    "PREPROCESS_HEAD_MAJOR": 768,
    "PREPROCESS_HEAD_MAJOR_KA": 768,
}
PREPROCESS_ARGS = (
    "Q",
    "K",
    "V",
    "tile_table",
    "seg_tile_begin",
    "tok_seg",
    "cu_seqlens",
    "partials",
    "counters",
    "mean_k",
    "q_mean",
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "qm4",
    "qm_sf",
    "vt4",
    "v_sf",
    "barrier",
    "num_tiles",
    "total_tokens",
    "total_padded",
    "num_heads",
    "num_kblocks",
)
STATS_ARGS = (
    "Q",
    "K",
    "tile_table",
    "seg_tile_begin",
    "partials",
    "counters",
    "mean_k",
    "q_mean",
    "q4",
    "q_sf",
    "qm4",
    "qm_sf",
    "num_tiles",
    "num_heads",
    "q_slots",
)
QUANTIZE_ARGS = (
    "Q",
    "K",
    "V",
    "tok_seg",
    "cu_seqlens",
    "seg_tile_begin",
    "mean_k",
    "q_mean",
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "qm4",
    "qm_sf",
    "vt4",
    "v_sf",
    "total_tokens",
    "total_padded",
    "num_heads",
    "num_kblocks",
)
ATTENTION_ARGS = (
    "Q_map",
    "K_map",
    "KSF_map",
    "VT_map",
    "VSF_map",
    "O",
    "q_sf",
    "qm4",
    "qm_sf",
    "seg_tile_begin",
    "unit_table",
    "total_units",
    "num_heads",
    "num_kblocks",
    "softmax_scale_log2",
)
PREPROCESS_HEAD_MAJOR_ARGS = (*PREPROCESS_ARGS[:19], "hctr", *PREPROCESS_ARGS[20:])
ATTENTION_OVERLAP_ARGS = (
    *ATTENTION_ARGS,
    "hctr",
    "units_per_head",
    "lead_ctas",
    "tail_ctas",
)
HOST_ARGS = {
    "PREPROCESS": PREPROCESS_ARGS,
    "STATS": STATS_ARGS,
    "QUANTIZE_KV": QUANTIZE_ARGS,
    "QUANTIZE_KV_HBLK64": QUANTIZE_ARGS,
    "ATTENTION": ATTENTION_ARGS,
    "ATTENTION_NODELTA": ATTENTION_ARGS,
    "ATTENTION_WS": ATTENTION_ARGS,
    "ATTENTION_WS_NODELTA": ATTENTION_ARGS,
    "PREPROCESS_HEAD_MAJOR": PREPROCESS_HEAD_MAJOR_ARGS,
    "ATTENTION_LEAD": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_TAIL": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_NODELTA_LEAD": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_NODELTA_TAIL": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_WS_NODELTA_LEAD": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_WS_NODELTA_TAIL": ATTENTION_OVERLAP_ARGS,
    "PREPROCESS_HEAD_MAJOR_KA": PREPROCESS_HEAD_MAJOR_ARGS,
    "ATTENTION_SFSEL_LEAD": ATTENTION_OVERLAP_ARGS,
    "ATTENTION_SFSEL_TAIL": ATTENTION_OVERLAP_ARGS,
}


def test_the_loader_publishes_the_eighteen_programs() -> None:
    assert tuple(NAMED.values()) == PROGRAMS
    assert ops._NVFP4_PROGRAMS == PROGRAMS
    assert len({program.name for program in PROGRAMS}) == 18
    for identifier, program in NAMED.items():
        assert program.name.startswith("cake_minimax_h3_sm120_nvfp4_varlen_attention_")
        assert program.operator_dir == "cake_minimax_h3_sm120_nvfp4_varlen_attention"
        assert program.targets == ("sm_120a",)
        assert program.kernel == f"kernel_{program.name}"
        assert program.args == (*HOST_ARGS[identifier], *cake_program.GRID_ARGS)
        assert set(program.tensors) <= set(program.args)
        assert program.launch_record() == {
            "block": (THREADS[identifier], 1, 1),
            "cluster": (1, 1, 1),
            # the fused pre-processing's grid barrier needs every CTA co-resident
            "cooperative": identifier == "PREPROCESS",
            "dynamic_smem_bytes": DYNAMIC_SMEM[identifier],
            # the overlapped form's producers start after every earlier launch of the stream; its lead
            # and tail attention are programmatic dependents of the producers and of the lead
            "use_pdl": identifier not in PLAIN,
        }
        assert program.needs_device_linking is False
    assert {
        identifier
        for identifier, program in NAMED.items()
        if not program.launch_record()["use_pdl"]
    } == set(PLAIN)
    # the two producer builds share their argument list and launch, not their units
    assert PREPROCESS_HEAD_MAJOR.args == PREPROCESS_HEAD_MAJOR_KA.args
    assert (
        PREPROCESS_HEAD_MAJOR.launch_record()
        == PREPROCESS_HEAD_MAJOR_KA.launch_record()
    )
    assert PREPROCESS_HEAD_MAJOR.name != PREPROCESS_HEAD_MAJOR_KA.name
    # the compensated overlap's two lead / tail builds share their launch, not their units
    for ship, sfsel in (
        (ATTENTION_LEAD, ATTENTION_SFSEL_LEAD),
        (ATTENTION_TAIL, ATTENTION_SFSEL_TAIL),
    ):
        assert ship.launch_record() == sfsel.launch_record()
        assert ship.name != sfsel.name
    # the two quantizer work orders and the four attention programs share their argument lists
    assert QUANTIZE_KV.args == QUANTIZE_KV_HBLK64.args
    assert QUANTIZE_KV.name != QUANTIZE_KV_HBLK64.name
    attention = (ATTENTION, ATTENTION_NODELTA, ATTENTION_WS, ATTENTION_WS_NODELTA)
    assert len({program.args for program in attention}) == 1
    overlap = (
        ATTENTION_LEAD,
        ATTENTION_TAIL,
        ATTENTION_NODELTA_LEAD,
        ATTENTION_NODELTA_TAIL,
        ATTENTION_WS_NODELTA_LEAD,
        ATTENTION_WS_NODELTA_TAIL,
        ATTENTION_SFSEL_LEAD,
        ATTENTION_SFSEL_TAIL,
    )
    assert len({program.args for program in overlap}) == 1


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
    ops._nvfp4_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (10, 3))
    with pytest.raises(RuntimeError, match="exact compute capability 12.0"):
        ops._nvfp4_target(0)
    ops._nvfp4_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    assert ops._nvfp4_target(0) == "sm_120a"
    ops._nvfp4_target.cache_clear()


def _plan(bounds, num_heads=1):
    tokens = bounds[-1]
    padded = max(1, -(-tokens // 128)) * 128
    tiles = sum(
        -(-(b - a) // 128)
        for a, b in zip(bounds[:-1], bounds[1:], strict=False)
        if b > a
    )
    return types.SimpleNamespace(
        bounds=bounds,
        total_tokens=tokens,
        num_segments=len(bounds) - 1,
        padded_tokens=padded,
        num_heads=num_heads,
        num_units=tiles * num_heads,
    )


# The overlapped form in launch order: producers, lead attention, tail attention.
OVERLAP = (PREPROCESS_HEAD_MAJOR, ATTENTION_LEAD, ATTENTION_TAIL)
# The compensated overlapped form of a plan above the fused pre-processing on 170 SMs: the lead and tail whose
# MMAs select each tile's B scale record.
OVERLAP_SFSEL = (PREPROCESS_HEAD_MAJOR, ATTENTION_SFSEL_LEAD, ATTENTION_SFSEL_TAIL)
OVERLAP_NODELTA = (
    PREPROCESS_HEAD_MAJOR,
    ATTENTION_NODELTA_LEAD,
    ATTENTION_NODELTA_TAIL,
)
# The uncompensated overlapped form on 188 SMs: the warp-specialised lead and tail behind the producers whose K
# items wait for their head's K sums in the worker's second warp.
OVERLAP_WS_NODELTA = (
    PREPROCESS_HEAD_MAJOR_KA,
    ATTENTION_WS_NODELTA_LEAD,
    ATTENTION_WS_NODELTA_TAIL,
)
PRODUCERS = (PREPROCESS_HEAD_MAJOR, PREPROCESS_HEAD_MAJOR_KA)


def _fused(plan) -> bool:
    """Whether ``plan`` runs the fused pre-processing (2048 to 16384 padded tokens)."""
    return 2048 <= plan.padded_tokens <= 16384


def _overlap(num_sms: int, fused: bool) -> tuple:
    return OVERLAP_SFSEL if num_sms == 170 and not fused else OVERLAP


def _overlap_nodelta(num_sms: int) -> tuple:
    return OVERLAP_WS_NODELTA if num_sms == 188 else OVERLAP_NODELTA


@pytest.mark.parametrize(
    ("bounds", "num_sms", "pre", "attention", "attention_nodelta"),
    [
        ((0, 1), 188, (STATS, QUANTIZE_KV), ATTENTION, ATTENTION_NODELTA),
        ((0, 1920), 188, (STATS, QUANTIZE_KV), ATTENTION_WS, ATTENTION_WS_NODELTA),
        ((0, 1920), 170, (STATS, QUANTIZE_KV), ATTENTION, ATTENTION_NODELTA),
        ((0, 2048), 187, (PREPROCESS,), ATTENTION, ATTENTION_WS_NODELTA),
        # on 170 to 187 SMs (without an overlap width) only the uncompensated route of a fused
        # pre-processing plan takes the warp-specialised form
        ((0, 2049), 171, (PREPROCESS,), ATTENTION, ATTENTION_WS_NODELTA),
        ((0, 16384), 186, (PREPROCESS,), ATTENTION, ATTENTION_WS_NODELTA),
        ((0, 16385), 171, (STATS, QUANTIZE_KV_HBLK64), ATTENTION, ATTENTION_NODELTA),
        ((0, 3000, 3000), 171, (PREPROCESS,), ATTENTION, ATTENTION_NODELTA),
        ((0, 2049), 169, (PREPROCESS,), ATTENTION, ATTENTION_NODELTA),
        # the mean segment counts the empty segments
        ((0, 3000, 3000), 189, (PREPROCESS,), ATTENTION, ATTENTION_NODELTA),
        (
            (0, 0, 129, 129, 500),
            188,
            (STATS, QUANTIZE_KV),
            ATTENTION,
            ATTENTION_NODELTA,
        ),
        # the compensated route's warp-specialised form stops below a mean of 16384 tokens
        ((0, 16384), 189, (PREPROCESS,), ATTENTION, ATTENTION_WS_NODELTA),
        ((0, 16385), 189, (STATS, QUANTIZE_KV_HBLK64), ATTENTION, ATTENTION_WS_NODELTA),
        (
            (0, 8368, 16736, 25104, 33472),
            189,
            (STATS, QUANTIZE_KV_HBLK64),
            ATTENTION_WS,
            ATTENTION_WS_NODELTA,
        ),
    ],
)
def test_program_routing_follows_the_plan_and_the_sm_count(
    bounds, num_sms, pre, attention, attention_nodelta
) -> None:
    plan = _plan(bounds)
    assert ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms) == (*pre, attention)
    assert ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms, delta=False) == (
        *pre,
        attention_nodelta,
    )


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 2048),
        (0, 2049),
        (0, 3000, 3000),
        (0, 4096),
        (0, 1100, 2100, 6000),
        (0, 16384),
    ],
)
@pytest.mark.parametrize("num_sms", [170, 188])
def test_fused_plans_take_the_overlapped_form_on_the_overlap_sm_counts(
    bounds, num_sms
) -> None:
    plan = _plan(bounds)
    assert _fused(plan)
    programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms)
    assert programs == _overlap(num_sms, fused=True)
    nodelta = _overlap_nodelta(num_sms)
    assert ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms, delta=False) == nodelta
    # a plan below the fused pre-processing keeps the separate launches
    small = _plan((0, 1920))
    for delta in (True, False):
        programs = ops.minimax_h3_sm120_nvfp4_programs(small, num_sms, delta=delta)
        assert not any(producers in programs for producers in PRODUCERS)


@pytest.mark.parametrize(
    ("bounds", "num_sms", "delta_overlaps", "nodelta_overlaps"),
    [
        ((0, 16385), 170, True, True),
        ((0, 16385), 188, True, True),
        ((0, 5000, 5000, 21000, 21129), 170, True, True),
        ((0, 5000, 5000, 21000, 21129), 188, True, True),
        ((0, 8368, 16736, 25104, 33472), 170, True, True),
        ((0, 8368, 16736, 25104, 33472), 188, True, True),
        ((0, 33472), 170, True, True),
        ((0, 33472), 188, True, True),
        # the compensated route overlaps every long plan on 170 and 188 SMs; on 188 SMs the uncompensated route
        # overlaps up to an attention-weighted mean segment of 40960 tokens
        ((0, 36864), 188, True, True),
        ((0, 36865), 188, True, True),
        ((0, 38592), 188, True, True),
        ((0, 38592), 170, True, True),
        ((0, 40960), 170, True, True),
        ((0, 40960), 188, True, True),
        ((0, 40961), 170, True, True),
        ((0, 40961), 188, True, False),
        ((0, 48768), 170, True, True),
        ((0, 48768), 188, True, False),
        ((0, 109901, 109952), 170, True, True),
        ((0, 109901, 109952), 188, True, False),
        # a plan whose separate launches would not take the warp-specialised attention (mean segment, empty
        # segments included, below 1536 tokens) keeps the overlap
        ((0,) * 33 + (49152,), 188, True, True),
        ((0,) * 32 + (49152,), 188, True, False),
    ],
)
def test_long_plans_take_the_overlapped_form_below_the_segment_bound(
    bounds, num_sms, delta_overlaps, nodelta_overlaps
) -> None:
    plan = _plan(bounds)
    programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms)
    if delta_overlaps:
        assert programs == _overlap(num_sms, fused=False)
    else:
        assert programs == (STATS, QUANTIZE_KV_HBLK64, ATTENTION)
    programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms, delta=False)
    if nodelta_overlaps:
        assert programs == _overlap_nodelta(num_sms)
    else:
        assert programs == (STATS, QUANTIZE_KV_HBLK64, ATTENTION_WS_NODELTA)
    # an SM count without an overlap width keeps the separate launches
    for delta in (True, False):
        programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms + 1, delta=delta)
        assert not any(producers in programs for producers in PRODUCERS)


@pytest.mark.parametrize(
    ("bounds", "nodelta_188"),
    [
        # fused pre-processing plans
        ((0, 2048), OVERLAP_WS_NODELTA),
        ((0, 257, 4567, 4824), OVERLAP_WS_NODELTA),
        ((0, 4096), OVERLAP_WS_NODELTA),
        ((0, 16384), OVERLAP_WS_NODELTA),
        # long plans up to an attention-weighted mean segment of 40960 tokens
        ((0, 8368, 16736, 25104, 33472), OVERLAP_WS_NODELTA),
        ((0, 33472), OVERLAP_WS_NODELTA),
        ((0, 38592), OVERLAP_WS_NODELTA),
        ((0, 40960), OVERLAP_WS_NODELTA),
        # above it the separate launches with the warp-specialised attention
        ((0, 40961), (STATS, QUANTIZE_KV_HBLK64, ATTENTION_WS_NODELTA)),
        ((0, 58944), (STATS, QUANTIZE_KV_HBLK64, ATTENTION_WS_NODELTA)),
        ((0, 109901, 109952), (STATS, QUANTIZE_KV_HBLK64, ATTENTION_WS_NODELTA)),
        # a plan below the fused pre-processing keeps the separate launches
        ((0, 1920), (STATS, QUANTIZE_KV, ATTENTION_WS_NODELTA)),
    ],
)
def test_uncompensated_overlap_is_warp_specialised_on_188_sms(
    bounds, nodelta_188
) -> None:
    """On 188 SMs the uncompensated overlapped form launches the warp-specialised lead and tail; on 170 SMs
    it keeps the poll lead and tail, and the compensated route keeps its overlapped form on both."""
    plan = _plan(bounds)
    assert ops.minimax_h3_sm120_nvfp4_programs(plan, 188, delta=False) == nodelta_188
    on_170 = ops.minimax_h3_sm120_nvfp4_programs(plan, 170, delta=False)
    if nodelta_188 == OVERLAP_WS_NODELTA:
        assert on_170 == OVERLAP_NODELTA
        assert ops.minimax_h3_sm120_nvfp4_programs(plan, 188) == OVERLAP
        on_170_delta = ops.minimax_h3_sm120_nvfp4_programs(plan, 170)
        assert on_170_delta == _overlap(170, _fused(plan))
    assert ATTENTION_WS_NODELTA_LEAD not in on_170
    assert ATTENTION_WS_NODELTA_TAIL not in on_170
    for num_sms in (170, 188):
        for program in (ATTENTION_WS_NODELTA_LEAD, ATTENTION_WS_NODELTA_TAIL):
            assert program not in ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms)


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 1920),
        (0, 2048),
        (0, 257, 4567, 4824),
        (0, 1100, 2100, 6000),
        (0, 16384),
        (0, 16385),
        (0, 8368, 16736, 25104, 33472),
        (0, 40960),
        (0, 40961),
        (0, 109901, 109952),
        (0,) * 33 + (49152,),
    ],
)
@pytest.mark.parametrize("num_sms", [84, 170, 187, 188, 189])
def test_k_sum_wait_producers_only_behind_the_warp_specialised_uncompensated_lead(
    bounds, num_sms
) -> None:
    """``PREPROCESS_HEAD_MAJOR_KA`` is a property of the plan's form: the producers of exactly the
    uncompensated overlapped form whose lead is ``ATTENTION_WS_NODELTA_LEAD`` (188 SMs); every other
    overlapped form keeps ``PREPROCESS_HEAD_MAJOR`` and no separate-launch plan takes either."""
    plan = _plan(bounds)
    for delta in (True, False):
        programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms, delta=delta)
        ws_form = programs[0] in PRODUCERS and not delta and num_sms == 188
        assert (programs == OVERLAP_WS_NODELTA) == ws_form
        assert (programs[0] is PREPROCESS_HEAD_MAJOR_KA) == ws_form


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 1920),
        (0, 2048),
        (0, 257, 4567, 4824),
        (0, 1100, 2100, 6000),
        (0, 16384),
        (0, 16385),
        (0, 8368, 16736, 25104, 33472),
        (0, 40961),
        (0, 109901, 109952),
    ],
)
@pytest.mark.parametrize("num_sms", [84, 169, 170, 171, 187, 188, 189])
def test_sfsel_lead_and_tail_only_in_the_long_compensated_overlap_on_170_sms(
    bounds, num_sms
) -> None:
    """``ATTENTION_SFSEL_LEAD`` / ``ATTENTION_SFSEL_TAIL`` are a property of the plan's form: the lead and tail of
    exactly the compensated overlapped form of a plan above the fused pre-processing on the SM counts of
    ``_OVERLAP_SFSEL_SMS`` (170); every other compensated overlap, the plans of the fused pre-processing on
    170 SMs included, keeps ``ATTENTION_LEAD`` / ``ATTENTION_TAIL`` and no other form takes either."""
    plan = _plan(bounds)
    fused = _fused(plan)
    for delta in (True, False):
        programs = ops.minimax_h3_sm120_nvfp4_programs(plan, num_sms, delta=delta)
        overlapped = programs[0] in PRODUCERS
        ws_form = overlapped and not delta and num_sms == 188
        sfsel_form = overlapped and delta and num_sms == 170 and not fused
        assert (programs == OVERLAP_SFSEL) == sfsel_form
        assert (programs == OVERLAP) == (overlapped and delta and not sfsel_form)
        sfsel = (ATTENTION_SFSEL_LEAD, ATTENTION_SFSEL_TAIL)
        assert any(p in programs for p in sfsel) == (programs == OVERLAP_SFSEL)
        assert (ATTENTION_WS_NODELTA_LEAD in programs) == ws_form
        assert PREPROCESS_HEAD_MAJOR_KA not in programs[1:]


@pytest.mark.parametrize(
    ("num_sms", "num_units", "expected"),
    [
        # (lead, producers, tail): the producers take one CTA (three workers) per SM the lead leaves free,
        # at most 3 * P CTAs
        (170, 1792, (128, 42, 42)),
        (188, 1792, (142, 46, 46)),
        (170, 16, (16, 126, 42)),
        (188, 1, (1, 138, 46)),
        (188, 0, (1, 138, 46)),
        (171, 1792, None),
        (187, 1792, None),
        (84, 1792, None),
    ],
)
def test_overlap_grids_follow_the_sm_count(num_sms, num_units, expected) -> None:
    assert (
        ops.minimax_h3_sm120_nvfp4_overlap_grids(num_sms, num_units=num_units)
        == expected
    )
    with pytest.raises(ValueError):
        ops.minimax_h3_sm120_nvfp4_overlap_grids(0, num_units=num_units)


@pytest.mark.parametrize(
    ("num_sms", "padded_tokens", "expected"),
    [
        (170, None, (128, 42, 42)),
        (170, 16384, (128, 42, 42)),
        (170, 16512, (106, 64, 64)),
        (188, 16384, (142, 46, 46)),
        (188, 16512, (142, 46, 46)),
        (171, 16512, None),
    ],
)
def test_long_plans_overlap_at_the_long_width(num_sms, padded_tokens, expected) -> None:
    assert (
        ops.minimax_h3_sm120_nvfp4_overlap_grids(
            num_sms, num_units=1792, padded_tokens=padded_tokens
        )
        == expected
    )


# The custom op's tensor arguments from ``tok_seg`` through ``qm_sf_words``: stand-ins, the launches are
# recorded.
IMPL_TENSORS = (
    "tok_seg",
    "cu_seqlens",
    "seg_tile_begin",
    "tile_table",
    "unit_table",
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "qm4",
    "qm_sf",
    "vt4",
    "v_sf",
    "q_mean",
    "mean_k",
    "partials",
    "counters",
    "barrier",
    "hctr",
    "q4_rows",
    "k4_rows",
    "k_sf_rows",
    "vt4_rows",
    "v_sf_rows",
    "q_sf_words",
    "qm_sf_words",
)


@pytest.mark.parametrize(
    "overlap", [OVERLAP, OVERLAP_SFSEL, OVERLAP_NODELTA, OVERLAP_WS_NODELTA]
)
def test_overlapped_form_launches_the_producers_first(overlap, monkeypatch) -> None:
    """The overlapped form launches the producers, then the lead attention, then the tail attention, with
    the arguments of their kernels.  Before the first eager call of a form on a device, one no-work
    instance runs in the same order on the call's own head-control words (one head, one CTA per launch);
    a call during graph capture runs none."""
    launches = []
    monkeypatch.setattr(
        ops,
        "_nvfp4_entry",
        lambda _target, slot: lambda *args: launches.append((slot, args)),
    )
    monkeypatch.setattr(
        ops,
        "tvm_ffi",
        types.SimpleNamespace(use_raw_stream=lambda *_args: contextlib.nullcontext()),
    )
    monkeypatch.setattr(ops, "_ffi_device", lambda _index: None)
    monkeypatch.setattr(ops, "_NVFP4_OVERLAP_LOADED", set())
    capturing = [True]
    monkeypatch.setattr(ops, "_capturing", lambda _index, _stream: capturing[0])
    producers, lead, tail = (ops._NVFP4_PROGRAMS.index(program) for program in overlap)
    tokens, heads, scale = 2100, 3, 0.25
    q = torch.empty(tokens, heads, MINIMAX_H3_HEAD_DIM)
    # statistics tiles, Q slots, padded tokens, K blocks, units
    facts = (5, 1, 2176, 17, 51)
    # producer / second / lead slots, their grids, the tail slot and grid, units per head
    grids = (producers, -1, lead, 42, 1, 128, tail, 42, 17)

    def call() -> None:
        ops._minimax_h3_sm120_varlen_attention_nvfp4_impl(
            q, "k", "v", "out", *IMPL_TENSORS, *facts, *grids, scale, 7, "sm_120a"
        )

    producer_args = (
        q,
        "k",
        "v",
        "tile_table",
        "seg_tile_begin",
        "tok_seg",
        "cu_seqlens",
        "partials",
        "counters",
        "mean_k",
        "q_mean",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "qm4",
        "qm_sf",
        "vt4",
        "v_sf",
    )
    attention_args = (
        "q4_rows",
        "k4_rows",
        "k_sf_rows",
        "vt4_rows",
        "v_sf_rows",
        "out",
        "q_sf_words",
        "qm4",
        "qm_sf_words",
        "seg_tile_begin",
        "unit_table",
    )
    work = [
        (producers, (*producer_args, "hctr", 5, tokens, 2176, heads, 17, 42, 1, 1)),
        (lead, (*attention_args, 51, heads, 17, scale, "hctr", 17, 128, 42, 128, 1, 1)),
        (tail, (*attention_args, 51, heads, 17, scale, "hctr", 17, 128, 42, 42, 1, 1)),
    ]
    no_work = [
        (producers, (*producer_args, "hctr", 0, tokens, 0, 1, 17, 1, 1, 1)),
        (lead, (*attention_args, 0, 1, 17, scale, "hctr", 1, 1, 1, 1, 1, 1)),
        (tail, (*attention_args, 0, 1, 17, scale, "hctr", 1, 1, 1, 1, 1, 1)),
    ]
    call()  # during graph capture: the call's launches only
    assert launches == work
    launches.clear()
    capturing[0] = False
    call()  # the first eager call: the no-work instance, then the call
    assert launches == no_work + work
    launches.clear()
    call()
    assert launches == work


def test_workspace_owners_and_stream_locks(monkeypatch) -> None:
    """An eager call's workspace owner is its (device, stream); a call during graph capture also carries
    the capture's id, so graphs captured one after another on one stream get separate workspace sets.  The
    host lock is one per (device, stream), whatever the capture."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert quant_ops._owner(0, 1234) == (0, 1234)
    capture_ids = iter((7, 8, 0))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(quant_ops, "_capture_id", lambda _stream: next(capture_ids))
    assert quant_ops._owner(0, 1234) == (0, 1234, 7)
    assert quant_ops._owner(0, 1234) == (0, 1234, 8)
    # a stream that takes part in no capture is an eager owner
    assert quant_ops._owner(0, 1234) == (0, 1234)
    # torch's capture query reads the current device's stream; a call on another device asks CUDA about
    # its own stream, capturing or not
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(quant_ops, "_capture_id", lambda _stream: 9)
    assert quant_ops._owner(1, 4321) == (1, 4321, 9)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(quant_ops, "_capture_id", lambda _stream: 0)
    assert quant_ops._owner(1, 4321) == (1, 4321)
    lock = quant_ops._stream_lock(0, 1234)
    assert lock is quant_ops._stream_lock(0, 1234)
    assert lock is not quant_ops._stream_lock(0, 1235)
    assert lock is not quant_ops._stream_lock(1, 1234)


def test_head_major_unit_rows_cover_every_unit_once() -> None:
    bounds = (0, 0, 129, 129, 500)
    heads = 3
    plan = _plan(bounds, heads)
    rows = ops._plan_rows_head_major(plan, torch.device("cpu", 0)).view(-1, 4).tolist()
    assert len(rows) == plan.num_units == 5 * heads
    expected = [
        [seg, (head << 16) | tile, bounds[seg], bounds[seg + 1]]
        for head in range(heads)
        for seg, tile in ((1, 0), (1, 1), (3, 0), (3, 1), (3, 2))
    ]
    assert rows == expected


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        ((0, 1), 1),
        ((0, 128), 1),
        ((0, 129), 2),
        ((0, 512), 4),
        # a segment longer than four Q blocks is split into one-block statistics tiles
        ((0, 513), 1),
        ((0, 0, 129, 129, 500), 3),
        ((0, 257, 4567, 4824), 3),
        ((0, 600, 1000), 4),
    ],
)
def test_q_slots_is_the_largest_statistics_tile_in_q_blocks(bounds, expected) -> None:
    assert ops._q_slots(bounds) == expected


@pytest.mark.parametrize(
    ("num_sms", "heads", "facts", "expected"),
    [
        (
            188,
            1,
            {
                "num_stats_tiles": 1,
                "q_slots": 1,
                "padded_tokens": 128,
                "num_units": 1,
            },
            (4, 2, 4, 1),
        ),
        (
            188,
            7,
            {
                "num_stats_tiles": 3,
                "q_slots": 4,
                "padded_tokens": 512,
                "num_units": 28,
            },
            (112, 105, 112, 28),
        ),
        (
            188,
            56,
            {
                "num_stats_tiles": 262,
                "q_slots": 1,
                "padded_tokens": 33536,
                "num_units": 14672,
            },
            (564, 752, 1504, 188),
        ),
        (
            170,
            56,
            {
                "num_stats_tiles": 262,
                "q_slots": 1,
                "padded_tokens": 33536,
                "num_units": 14672,
            },
            (510, 680, 1360, 170),
        ),
        (188, 56, {"num_stats_tiles": 262, "q_slots": 1}, (None, 752, None, None)),
        (188, 56, {"padded_tokens": 33536}, (None, None, 1504, None)),
        (
            188,
            56,
            {"num_stats_tiles": 262, "padded_tokens": 33536},
            (564, None, 1504, None),
        ),
        (188, 56, {"num_units": 14672}, (None, None, None, 188)),
    ],
)
def test_launch_grids_follow_the_plan_facts(num_sms, heads, facts, expected) -> None:
    grids = ops.minimax_h3_sm120_nvfp4_launch_grids(num_sms, num_heads=heads, **facts)
    assert grids == expected
    with pytest.raises(ValueError):
        ops.minimax_h3_sm120_nvfp4_launch_grids(0, num_heads=heads, **facts)


def test_softmax_scale_log2_is_the_kernels_constant() -> None:
    assert ops._LOG2E == 1.4426950408889634
    scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    assert ops._softmax_scale_log2(scale) == scale * 1.4426950408889634


def test_public_names_are_exported() -> None:
    for name in (
        "minimax_h3_sm120_nvfp4_launch_grids",
        "minimax_h3_sm120_nvfp4_overlap_grids",
        "minimax_h3_sm120_nvfp4_programs",
        "minimax_h3_sm120_varlen_attention_nvfp4",
        "minimax_h3_sm120_varlen_attention_nvfp4_nodelta",
        "workspace_bytes_nvfp4",
    ):
        assert name in ops.__all__ and callable(getattr(ops, name))
