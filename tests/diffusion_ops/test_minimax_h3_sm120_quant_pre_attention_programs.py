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
"""CPU tests of the generated programs behind ``minimax_h3_fp8_pre_attention`` and
``minimax_h3_nvfp4_pre_attention`` (no CUDA device): each program constant of
``flashinfer.jit.cake_minimax_h3_sm120_quant_pre_attention`` matches its two translation units
(the kernel symbol the device unit defines and the binding declares, the binding namespace, the
exported ``run`` entry, the FFI checks header), its digests follow from the file bytes, its JIT spec
carries the target flags, ``--fmad=false`` and the recorded options, and the host module's program
table, FP8 MMA form rule, NVFP4 tile-order rule and grid rule are what the kernels expect."""

import hashlib
import re
import types

import pytest
import torch

from flashinfer.diffusion_ops import cake_minimax_h3_sm120_quant_pre_attention as ops
from flashinfer.jit import cake_program
from flashinfer.jit.cake_minimax_h3_sm120_quant_pre_attention import (
    FP8_BF16,
    FP8_E4M3,
    FP8_MX_BF16,
    FP8_MX_E4M3,
    FP8_MX_NVFP4,
    FP8_NVFP4,
    NORM_FP8,
    NORM_NVFP4,
    NVFP4_BF16,
    NVFP4_E4M3,
    NVFP4_E4M3_G1,
    NVFP4_NVFP4,
    PROGRAMS,
)
from flashinfer.jit.core import sm120a_nvcc_flags

NAMED = {
    "NORM_FP8": NORM_FP8,
    "NORM_NVFP4": NORM_NVFP4,
    "FP8_BF16": FP8_BF16,
    "FP8_E4M3": FP8_E4M3,
    "FP8_NVFP4": FP8_NVFP4,
    "FP8_MX_BF16": FP8_MX_BF16,
    "FP8_MX_E4M3": FP8_MX_E4M3,
    "FP8_MX_NVFP4": FP8_MX_NVFP4,
    "NVFP4_BF16": NVFP4_BF16,
    "NVFP4_E4M3": NVFP4_E4M3,
    "NVFP4_NVFP4": NVFP4_NVFP4,
    "NVFP4_E4M3_G1": NVFP4_E4M3_G1,
}
GEMMS = {name: program for name, program in NAMED.items() if "NORM" not in name}
NORM_FP8_ARGS = (
    "x",
    "x_norm_weight",
    "adaln_scale",
    "adaln_shift",
    "adaln_index",
    "act_q",
    "act_scale",
    "M",
    "eps",
)
NORM_NVFP4_ARGS = (
    "x",
    "x_norm_weight",
    "adaln_scale",
    "adaln_shift",
    "adaln_index",
    "act_q",
    "act_sf",
    "act_global_scale",
    "M",
    "eps",
)
GEMM_ARGS = (
    "A",
    "B",
    "SFA",
    "SFB",
    "act_scale",
    "w_scale",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos_sin",
    "q_out",
    "k_out",
    "v_out",
    "q_sf",
    "k_sf",
    "v_sf",
    "M",
    "num_m_tiles",
    "total_tiles",
    "eps",
    "alpha",
    "out_scale_q",
    "out_scale_k",
    "out_scale_v",
    "sf_mul_q",
    "sf_mul_k",
    "sf_mul_v",
)
# The stages compute 128 bytes of dynamic shared memory; the FP8 GEMM four ring stages of
# 256 x 64 A and 128 x 64 B bytes with the row statistics, the NVFP4 GEMM three stages with
# their scale tiles, each behind the 1024-byte-aligned barrier prefix.
DYNAMIC_SMEM = {"NORM_FP8": 128, "NORM_NVFP4": 128}
DYNAMIC_SMEM.update(
    {name: 99840 if name.startswith("FP8") else 86528 for name in GEMMS}
)
DENSE_FP8_MMA = rb"mma\.sync\.aligned\.m16n8k32\.row\.col\.f32\.e4m3\.e4m3\.f32"
UNIT_SCALE_FP8_MMA = rb"kind::mxf8f6f4\.block_scale\.scale_vec::1X"
NVFP4_MMA = rb"kind::mxf4nvf4\.block_scale"


def _host_args(identifier: str) -> tuple[str, ...]:
    if identifier == "NORM_FP8":
        return NORM_FP8_ARGS
    if identifier == "NORM_NVFP4":
        return NORM_NVFP4_ARGS
    return GEMM_ARGS


def test_the_loader_publishes_the_twelve_programs() -> None:
    assert tuple(NAMED.values()) == PROGRAMS
    assert len({program.name for program in PROGRAMS}) == 12
    for identifier, program in NAMED.items():
        assert program.name.startswith("cake_minimax_h3_sm120_quant_pre_attention_")
        assert program.operator_dir == "cake_minimax_h3_sm120_quant_pre_attention"
        assert program.targets == ("sm_120a",)
        assert program.kernel == f"kernel_{program.name}"
        assert program.args == (*_host_args(identifier), *cake_program.GRID_ARGS)
        assert set(program.tensors) <= set(program.args)
        assert program.launch_record() == {
            "block": (128 if identifier.startswith("NORM") else 256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": DYNAMIC_SMEM[identifier],
            "use_pdl": False,
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


@pytest.mark.parametrize("identifier", sorted(GEMMS))
def test_each_gemm_issues_its_mma_form(identifier: str) -> None:
    kernel = GEMMS[identifier].source_paths()[0].read_bytes()
    if identifier.startswith("FP8_MX"):
        assert re.search(UNIT_SCALE_FP8_MMA, kernel)
        assert not re.search(DENSE_FP8_MMA, kernel)
    elif identifier.startswith("FP8"):
        assert re.search(DENSE_FP8_MMA, kernel)
        assert not re.search(UNIT_SCALE_FP8_MMA, kernel)
    else:
        assert re.search(NVFP4_MMA, kernel)


def test_the_host_table_names_each_gemm_once() -> None:
    assert ops._FP8_GEMM == {
        (False, 0): FP8_BF16,
        (False, 1): FP8_E4M3,
        (False, 2): FP8_NVFP4,
        (True, 0): FP8_MX_BF16,
        (True, 1): FP8_MX_E4M3,
        (True, 2): FP8_MX_NVFP4,
    }
    assert ops._NVFP4_GEMM == {
        (False, 0): NVFP4_BF16,
        (False, 1): NVFP4_E4M3,
        (False, 2): NVFP4_NVFP4,
        (True, 0): NVFP4_BF16,
        (True, 1): NVFP4_E4M3_G1,
        (True, 2): NVFP4_NVFP4,
    }
    assert ops._OUT_MODES == {"bf16": 0, "e4m3": 1, "nvfp4": 2}
    assert ops._SFB_ROWS * ops._SFB_ROW_BYTES == (
        ops.MINIMAX_H3_QKV_WIDTH * ops.MINIMAX_H3_HIDDEN // ops.MINIMAX_H3_SF_BLOCK
    )


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((12, 0), "sm_120a"), ((12, 1), None), ((10, 0), None), ((9, 0), None)],
)
def test_exact_architecture_router(capability, expected) -> None:
    for program in PROGRAMS:
        assert program.target_for(capability) == expected


def _properties(major: int, minor: int, name: str, sms: int):
    return types.SimpleNamespace(
        major=major, minor=minor, name=name, multi_processor_count=sms
    )


@pytest.mark.parametrize(
    ("name", "sms", "unit_scale"),
    [
        ("NVIDIA GeForce RTX 5090", 170, True),
        ("NVIDIA RTX PRO 6000 Blackwell Server Edition", 188, False),
        ("NVIDIA RTX PRO 6000 Blackwell Workstation Edition", 188, False),
    ],
)
def test_device_facts_fp8_mma_form_and_nvfp4_tile_order(
    monkeypatch, name, sms, unit_scale
) -> None:
    ops._device_facts.cache_clear()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: _properties(12, 0, name, sms),
    )
    assert ops._device_facts(0) == ("sm_120a", sms, name)
    assert ops._fp8_unit_scale_mma(0) is unit_scale
    assert ops._nvfp4_one_m_tile_order(0) is unit_scale
    ops._device_facts.cache_clear()


@pytest.mark.parametrize("capability", [(12, 1), (10, 0), (9, 0)])
def test_device_facts_reject_other_architectures(monkeypatch, capability) -> None:
    ops._device_facts.cache_clear()
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _index: _properties(*capability, "NVIDIA GPU", 132),
    )
    with pytest.raises(RuntimeError, match="compute capability 12.0"):
        ops._device_facts(0)
    ops._device_facts.cache_clear()


@pytest.mark.parametrize(
    ("rows", "sms", "expected"),
    [
        (1, 188, (1, 168, 1, 168)),
        (129, 170, (129, 168, 1, 168)),
        (256, 188, (256, 168, 1, 168)),
        (257, 188, (257, 188, 2, 336)),
        (38592, 188, (1504, 188, 151, 25368)),
        (109952, 170, (1360, 170, 430, 72240)),
    ],
)
def test_launch_plan(rows, sms, expected) -> None:
    assert ops._launch_plan(rows, sms) == expected
