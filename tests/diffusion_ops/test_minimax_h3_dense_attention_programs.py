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
"""CPU tests of the generated program behind ``minimax_h3_dense_attention`` (no CUDA
device): the ``ATTENTION`` constant of ``flashinfer.jit.cake_minimax_h3_dense_attention``
matches its two translation units (the kernel symbol the device unit defines and the
binding declares, the binding namespace, the exported ``run`` entry, the FFI checks
header), its digests follow from the file bytes, its JIT spec carries the target flags,
``--fmad=false`` and no fast math, and the host module's architecture router, launch plan
and workspace size are what the kernel expects."""

import hashlib
import math
import re

import pytest
import torch

from flashinfer.diffusion_ops import cake_minimax_h3_dense_attention as ops
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_dense_attention import ATTENTION, PROGRAMS
from flashinfer.jit.core import (
    sm90a_nvcc_flags,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm120a_nvcc_flags,
    sm121a_nvcc_flags,
)

HOST_ARGS = (
    "Q_map",
    "K_map",
    "V_map",
    "O",
    "partial_ws",
    "split_counters",
    "tokens",
    "num_q_tiles",
    "num_kv_tiles",
    "total_work",
    "split_base",
    "split_k",
    "split_tiles",
    "softmax_scale_log2",
    "q_scale_bf16x2",
)
TENSORS = ("Q_map", "K_map", "V_map", "O", "partial_ws", "split_counters")
TARGET_NVCC_FLAGS = {
    "sm_90a": sm90a_nvcc_flags,
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_120a": sm120a_nvcc_flags,
    "sm_121a": sm121a_nvcc_flags,
}


def test_the_loader_publishes_the_program() -> None:
    assert PROGRAMS == (ATTENTION,)
    assert ATTENTION.name.startswith("cake_minimax_h3_dense_attention_")
    assert ATTENTION.operator_dir == "cake_minimax_h3_dense_attention"
    assert ATTENTION.targets == ("sm_90a", "sm_100a", "sm_103a", "sm_120a", "sm_121a")
    assert ATTENTION.kernel == f"kernel_{ATTENTION.name}"
    assert ATTENTION.args == (*HOST_ARGS, *cake_program.GRID_ARGS)
    assert tuple(ATTENTION.tensors) == TENSORS
    assert ATTENTION.launch_record() == {
        "block": (256, 1, 1),
        "cluster": (1, 1, 1),
        "cooperative": False,
        "dynamic_smem_bytes": 99456,
        "use_pdl": False,
    }
    assert ATTENTION.needs_device_linking is False


def test_the_program_matches_its_units() -> None:
    kernel_path, binding_path = ATTENTION.source_paths()
    assert kernel_path.name == f"{ATTENTION.name}_kernel.cu"
    assert binding_path.name == f"{ATTENTION.name}_binding.cu"
    kernel, binding = kernel_path.read_bytes(), binding_path.read_bytes()
    assert re.search(
        rb"__global__[^;{]*\b" + ATTENTION.kernel.encode() + rb"\b", kernel
    )
    assert ATTENTION.kernel.encode() in binding
    assert f"namespace cake_binding_{ATTENTION.kernel}".encode() in binding
    assert b"TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in binding
    assert b"cake_ffi_checks.h" in binding
    # the units end in exactly one newline (FlashInfer's end-of-file hook must be a no-op)
    for data in (kernel, binding):
        assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    for target in ATTENTION.targets:
        options = ATTENTION.options_for(target)
        assert options == ()
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
        assert ATTENTION.digest(target) == expected
        assert ATTENTION.module_name(target) == f"{ATTENTION.name}_{target}_{expected}"
        spec = ATTENTION.spec(target)
        flags = [str(flag) for flag in spec.extra_cuda_cflags]
        tail = [*TARGET_NVCC_FLAGS[target], "--fmad=false"]
        assert flags[-len(tail) :] == tail
        assert "-use_fast_math" not in flags and "--use_fast_math" not in flags
        assert spec.name == ATTENTION.module_name(target)
    with pytest.raises(KeyError):
        ATTENTION.options_for("sm_110a")


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((9, 0), "sm_90a"),
        ((10, 0), "sm_100a"),
        ((10, 3), "sm_103a"),
        ((12, 0), "sm_120a"),
        ((12, 1), "sm_121a"),
        ((8, 0), None),
        ((10, 7), None),
        ((11, 0), None),
    ],
)
def test_exact_architecture_router(capability, expected) -> None:
    assert ATTENTION.target_for(capability) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    ops._target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (11, 0))
    with pytest.raises(RuntimeError, match=r"9\.0, 10\.0, 10\.3, 12\.0 or 12\.1"):
        ops._target(0)
    ops._target.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _index: (12, 0))
    assert ops._target(0) == "sm_120a"
    ops._target.cache_clear()


@pytest.mark.parametrize(
    ("tokens", "num_sms", "expected"),
    [
        (1, 188, (56, 1, 1, 56, 56, 1, 1)),
        (128, 188, (56, 1, 2, 56, 56, 1, 2)),
        (129, 188, (112, 2, 3, 112, 112, 1, 3)),
        (257, 188, (168, 3, 5, 168, 168, 1, 5)),
        (1023, 188, (188, 8, 16, 520, 376, 2, 8)),
        (4097, 188, (188, 33, 65, 1848, 1848, 1, 65)),
        (15493, 188, (188, 122, 243, 6896, 6768, 2, 122)),
        (131072, 188, (188, 1024, 2048, 57372, 57340, 8, 256)),
        (1023, 170, (170, 8, 16, 448, 448, 1, 16)),
        (15493, 170, (170, 122, 243, 6960, 6800, 5, 49)),
        (4097, 1, (1, 33, 65, 1848, 1848, 1, 65)),
    ],
)
def test_launch_plan(tokens, num_sms, expected) -> None:
    plan = ops.minimax_h3_dense_attention_launch_plan(tokens, num_sms)
    assert tuple(plan) == expected
    items = plan.num_q_tiles * ops.MINIMAX_H3_NUM_HEADS
    assert plan.num_q_tiles == math.ceil(tokens / 128)
    assert plan.num_kv_tiles == math.ceil(tokens / 64)
    assert plan.grid == min(items, num_sms)
    # every split range holds at least one K/V tile
    assert (plan.split_k - 1) * plan.split_tiles < plan.num_kv_tiles
    assert plan.total_work == plan.split_base + (items - plan.split_base) * plan.split_k


@pytest.mark.parametrize(("tokens", "num_sms"), [(0, 188), (131073, 188), (1, 0)])
def test_launch_plan_rejects_out_of_range_inputs(tokens, num_sms) -> None:
    with pytest.raises(ValueError):
        ops.minimax_h3_dense_attention_launch_plan(tokens, num_sms)


def test_workspace_holds_one_partial_slot_and_one_counter_per_cta() -> None:
    assert ops.minimax_h3_dense_attention_workspace_bytes(188) == 188 * (
        68 * 256 * 4 + 4
    )
    with pytest.raises(ValueError):
        ops.minimax_h3_dense_attention_workspace_bytes(0)


def test_launch_constants() -> None:
    # the softmax scale is 1.0 (the query carries 1 / sqrt(128)); the kernel takes log2(e)
    softmax_scale_log2 = ops._SOFTMAX_SCALE_LOG2
    assert softmax_scale_log2 == math.log2(math.e)
    # two BF16 copies of the query scale
    query_scale_bf16x2 = ops._QUERY_SCALE_BF16X2
    scale_bits = (
        torch.tensor([ops.MINIMAX_H3_QUERY_SCALE_BF16], dtype=torch.bfloat16)
        .view(torch.int16)
        .item()
        & 0xFFFF
    )
    assert query_scale_bf16x2 == (scale_bits << 16) | scale_bits


def test_public_names_are_exported() -> None:
    for name in (
        "minimax_h3_dense_attention",
        "minimax_h3_dense_attention_launch_plan",
        "minimax_h3_dense_attention_workspace_bytes",
    ):
        assert name in ops.__all__ and callable(getattr(ops, name))
