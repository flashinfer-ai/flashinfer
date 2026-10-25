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
``--fmad=false`` and the recorded options, and the host module's program routing, grid rule and Q slot
count are what the kernels expect."""

import hashlib
import math
import re
import types

import pytest
import torch

from flashinfer.diffusion_ops import cake_minimax_h3_sm120_nvfp4_varlen_attention as ops
from flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_varlen_attention import (
    MINIMAX_H3_HEAD_DIM,
)
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    ATTENTION,
    ATTENTION_NODELTA,
    ATTENTION_WS,
    ATTENTION_WS_NODELTA,
    PREPROCESS,
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
}
DYNAMIC_SMEM = {
    "PREPROCESS": 16512,
    "STATS": 17024,
    "QUANTIZE_KV": 0,
    "QUANTIZE_KV_HBLK64": 0,
    "ATTENTION": 66560,
    "ATTENTION_NODELTA": 48128,
    "ATTENTION_WS": 80896,
    "ATTENTION_WS_NODELTA": 80896,
}
THREADS = {identifier: 256 for identifier in NAMED} | {
    "ATTENTION_WS": 384,
    "ATTENTION_WS_NODELTA": 384,
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
HOST_ARGS = {
    "PREPROCESS": PREPROCESS_ARGS,
    "STATS": STATS_ARGS,
    "QUANTIZE_KV": QUANTIZE_ARGS,
    "QUANTIZE_KV_HBLK64": QUANTIZE_ARGS,
    "ATTENTION": ATTENTION_ARGS,
    "ATTENTION_NODELTA": ATTENTION_ARGS,
    "ATTENTION_WS": ATTENTION_ARGS,
    "ATTENTION_WS_NODELTA": ATTENTION_ARGS,
}


def test_the_loader_publishes_the_eight_programs() -> None:
    assert tuple(NAMED.values()) == PROGRAMS
    assert ops._NVFP4_PROGRAMS == PROGRAMS
    assert len({program.name for program in PROGRAMS}) == 8
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
            "use_pdl": True,
        }
        assert program.needs_device_linking is False
    # the two quantizer work orders and the four attention programs share their argument lists
    assert QUANTIZE_KV.args == QUANTIZE_KV_HBLK64.args
    assert QUANTIZE_KV.name != QUANTIZE_KV_HBLK64.name
    attention = (ATTENTION, ATTENTION_NODELTA, ATTENTION_WS, ATTENTION_WS_NODELTA)
    assert len({program.args for program in attention}) == 1


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


def _plan(bounds):
    tokens = bounds[-1]
    padded = max(1, -(-tokens // 128)) * 128
    return types.SimpleNamespace(
        bounds=bounds,
        total_tokens=tokens,
        num_segments=len(bounds) - 1,
        padded_tokens=padded,
    )


@pytest.mark.parametrize(
    ("bounds", "num_sms", "pre", "attention", "attention_nodelta"),
    [
        ((0, 1), 188, (STATS, QUANTIZE_KV), ATTENTION, ATTENTION_NODELTA),
        ((0, 1920), 188, (STATS, QUANTIZE_KV), ATTENTION_WS, ATTENTION_WS_NODELTA),
        ((0, 1920), 170, (STATS, QUANTIZE_KV), ATTENTION, ATTENTION_NODELTA),
        ((0, 2048), 188, (PREPROCESS,), ATTENTION_WS, ATTENTION_WS_NODELTA),
        ((0, 2049), 170, (PREPROCESS,), ATTENTION, ATTENTION_NODELTA),
        # the mean segment counts the empty segments
        ((0, 3000, 3000), 188, (PREPROCESS,), ATTENTION, ATTENTION_NODELTA),
        (
            (0, 0, 129, 129, 500),
            188,
            (STATS, QUANTIZE_KV),
            ATTENTION,
            ATTENTION_NODELTA,
        ),
        # the compensated route's warp-specialised form stops below a mean of 16384 tokens
        ((0, 16384), 188, (PREPROCESS,), ATTENTION, ATTENTION_WS_NODELTA),
        ((0, 16385), 188, (STATS, QUANTIZE_KV_HBLK64), ATTENTION, ATTENTION_WS_NODELTA),
        (
            (0, 8368, 16736, 25104, 33472),
            188,
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
            (376, 752, 1504, 188),
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
            (340, 680, 1360, 170),
        ),
        (188, 56, {"num_stats_tiles": 262, "q_slots": 1}, (None, 752, None, None)),
        (188, 56, {"padded_tokens": 33536}, (None, None, 1504, None)),
        (
            188,
            56,
            {"num_stats_tiles": 262, "padded_tokens": 33536},
            (376, None, 1504, None),
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
        "minimax_h3_sm120_nvfp4_programs",
        "minimax_h3_sm120_varlen_attention_nvfp4",
        "minimax_h3_sm120_varlen_attention_nvfp4_nodelta",
        "workspace_bytes_nvfp4",
    ):
        assert name in ops.__all__ and callable(getattr(ops, name))
