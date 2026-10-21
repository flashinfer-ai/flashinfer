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
"""CPU tests of the generated programs behind ``minimax_h3_sm120_varlen_attention_nvfp4(pv_mode="fp8")``
(no CUDA device): each program constant of ``flashinfer.jit.cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention``
matches its two translation units (the kernel symbol the device unit defines and the binding declares, the
binding namespace, the exported ``run`` entry, the FFI checks header), its digests follow from the file bytes,
its JIT spec carries the target flags, ``--fmad=false`` and the recorded options, and the host module's
grid rule, program routing and workspace estimate are what the kernels expect."""

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
from flashinfer.jit.cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention import (
    ATTENTION_BASE,
    ATTENTION_SHORT,
    KV_STATS,
    PROGRAMS,
    QUANTIZE,
    QUANTIZE_MB3,
)
from flashinfer.jit.core import sm120a_nvcc_flags

NAMED = {
    "KV_STATS": KV_STATS,
    "QUANTIZE": QUANTIZE,
    "QUANTIZE_MB3": QUANTIZE_MB3,
    "ATTENTION_BASE": ATTENTION_BASE,
    "ATTENTION_SHORT": ATTENTION_SHORT,
}
DYNAMIC_SMEM = {
    "KV_STATS": 16512,
    "QUANTIZE": 0,
    "QUANTIZE_MB3": 0,
    "ATTENTION_BASE": 60416,
    "ATTENTION_SHORT": 60416,
}
STATS_ARGS = (
    "K",
    "V",
    "tile_table",
    "partials",
    "knorm_part",
    "counters",
    "mean_k",
    "v_scale",
    "k_scale",
    "num_tiles",
    "num_heads",
)
QUANTIZE_ARGS = (
    "Q",
    "K",
    "V",
    "tok_seg",
    "mean_k",
    "v_scale",
    "q4",
    "k4",
    "q_sf",
    "k_sf",
    "vt8",
    "q_scale",
    "k_scale",
    "total_tokens",
    "total_padded",
    "num_heads",
    "num_kblocks",
)
ATTENTION_ARGS = (
    "Q_map",
    "K_map",
    "KSF_map",
    "V_map",
    "O",
    "q_scale",
    "k_scale",
    "v_scale",
    "q_sf",
    "unit_table",
    "total_units",
    "num_heads",
    "num_kblocks",
    "softmax_scale_log2",
)
HOST_ARGS = {
    "KV_STATS": STATS_ARGS,
    "QUANTIZE": QUANTIZE_ARGS,
    "QUANTIZE_MB3": QUANTIZE_ARGS,
    "ATTENTION_BASE": ATTENTION_ARGS,
    "ATTENTION_SHORT": ATTENTION_ARGS,
}


def test_the_loader_publishes_the_five_programs() -> None:
    assert tuple(NAMED.values()) == PROGRAMS
    assert len({program.name for program in PROGRAMS}) == 5
    for identifier, program in NAMED.items():
        assert program.name.startswith(
            "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention_"
        )
        assert (
            program.operator_dir == "cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention"
        )
        assert program.targets == ("sm_120a",)
        assert program.kernel == f"kernel_{program.name}"
        assert program.args == (*HOST_ARGS[identifier], *cake_program.GRID_ARGS)
        assert program.args[-3:] == cake_program.GRID_ARGS
        assert set(program.tensors) <= set(program.args)
        assert program.launch_record() == {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": DYNAMIC_SMEM[identifier],
            "use_pdl": True,
        }
        assert program.needs_device_linking is False
    # the two quantizer builds and the two attention schedules share their argument lists, not their programs
    assert QUANTIZE.args == QUANTIZE_MB3.args and QUANTIZE.name != QUANTIZE_MB3.name
    assert ATTENTION_BASE.args == ATTENTION_SHORT.args
    assert ATTENTION_BASE.name != ATTENTION_SHORT.name


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
    assert b"CAKE_KERNEL_SYMBOL" not in kernel and b"CAKE_KERNEL_SYMBOL" not in binding
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
    [((12, 0), "sm_120a"), ((10, 0), None), ((10, 3), None), ((9, 0), None)],
)
def test_exact_architecture_router(capability, expected) -> None:
    for program in PROGRAMS:
        assert program.target_for(capability) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    ops._fp8pv_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (10, 3))
    with pytest.raises(RuntimeError, match="exact compute capability 12.0"):
        ops._fp8pv_target(0)
    ops._fp8pv_target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    assert ops._fp8pv_target(0) == "sm_120a"
    ops._fp8pv_target.cache_clear()


@pytest.mark.parametrize(
    ("bounds", "quantize", "attention"),
    [
        ((0, 1), QUANTIZE_MB3, ATTENTION_SHORT),
        ((0, 1024), QUANTIZE_MB3, ATTENTION_SHORT),
        ((0, 1025), QUANTIZE, ATTENTION_SHORT),
        ((0, 4096), QUANTIZE, ATTENTION_SHORT),
        ((0, 4097), QUANTIZE, ATTENTION_BASE),
        ((0, 0, 129, 129, 500), QUANTIZE_MB3, ATTENTION_SHORT),
        ((0, 257, 4567, 4824), QUANTIZE, ATTENTION_BASE),
        ((0, 8368, 16736, 25104, 33472), QUANTIZE, ATTENTION_BASE),
    ],
)
def test_program_routing_follows_the_token_and_segment_counts(
    bounds, quantize, attention
) -> None:
    plan = types.SimpleNamespace(bounds=bounds, total_tokens=bounds[-1])
    assert ops.minimax_h3_sm120_nvfp4_fp8pv_programs(plan) == (quantize, attention)


@pytest.mark.parametrize(
    ("num_sms", "heads", "facts", "expected"),
    [
        (
            188,
            1,
            {"num_stats_tiles": 1, "padded_tokens": 128, "num_units": 1},
            (1, 4, 1),
        ),
        (
            188,
            7,
            {"num_stats_tiles": 7, "padded_tokens": 1024, "num_units": 63},
            (49, 224, 63),
        ),
        (
            188,
            56,
            {"num_stats_tiles": 262, "padded_tokens": 33536, "num_units": 14672},
            (752, 1504, 188),
        ),
        (
            170,
            56,
            {"num_stats_tiles": 262, "padded_tokens": 33536, "num_units": 14672},
            (680, 1360, 170),
        ),
        (188, 56, {"num_stats_tiles": 262}, (752, None, None)),
        (188, 56, {"padded_tokens": 33536}, (None, 1504, None)),
        (188, 56, {"num_units": 14672}, (None, None, 188)),
    ],
)
def test_launch_grids_follow_the_plan_facts(num_sms, heads, facts, expected) -> None:
    grids = ops.minimax_h3_sm120_nvfp4_fp8pv_launch_grids(
        num_sms, num_heads=heads, **facts
    )
    assert grids == expected
    with pytest.raises(ValueError):
        ops.minimax_h3_sm120_nvfp4_fp8pv_launch_grids(0, num_heads=heads, **facts)


def test_softmax_scale_log2_is_the_kernel_tests_constant() -> None:
    # The kernel tests fix ``SOFTMAX_SCALE * log2(e)`` with ``SOFTMAX_SCALE = 1 / sqrt(128)``; the host's
    # default scale is spelled the same way, so the two arms of the kitchen's parity harness pass one float.
    scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    value = ops._softmax_scale_log2(scale)
    assert value == scale * math.log2(math.e)
    assert (
        torch.tensor(value, dtype=torch.float32).item()
        == torch.tensor(scale * math.log2(math.e), dtype=torch.float32).item()
    )


def test_workspace_estimate_covers_the_route_buffers() -> None:
    tokens, heads, segments = 900, 7, 3
    nkb = 8
    padded = nkb * 128
    tiles = -(-tokens // 128) + segments
    expected = (
        2 * tokens * heads * 64  # q4, k4
        + tokens * heads * 8  # q_sf
        + heads * nkb * 1024  # k_sf
        + heads * 128 * padded  # vt8
        + 4
        * (
            tokens * heads  # q_scale
            + segments * heads  # k_scale
            + 2 * segments * heads * 128  # mean_k, v_scale
            + tiles * heads * 256  # partials
            + tiles * heads  # knorm_part
            + segments * heads  # counters
        )
    )
    assert ops.workspace_bytes_nvfp4_fp8pv(tokens, heads, segments) == expected
    assert ops.workspace_bytes_nvfp4_fp8pv(1, 1, 1) > 0


def test_public_names_are_exported() -> None:
    for name in (
        "minimax_h3_sm120_nvfp4_fp8pv_launch_grids",
        "minimax_h3_sm120_nvfp4_fp8pv_programs",
        "workspace_bytes_nvfp4_fp8pv",
    ):
        assert name in ops.__all__ and callable(getattr(ops, name))
