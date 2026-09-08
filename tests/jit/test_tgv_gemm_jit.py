# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import torch

from flashinfer.jit import core as jit_core
from flashinfer.jit import env as jit_env
from flashinfer.jit.gemm.core import gen_tgv_gemm_sm10x_module


def test_tgv_gemm_target_specific_jit_specs_and_aot_inventory(monkeypatch, tmp_path):
    monkeypatch.setattr(
        jit_core.current_compilation_context,
        "TARGET_CUDA_ARCHS",
        {(10, "0a"), (10, "3a")},
    )
    monkeypatch.setattr(jit_env, "FLASHINFER_GEN_SRC_DIR", tmp_path / "generated")

    specs = [
        gen_tgv_gemm_sm10x_module(torch.bfloat16, use_sm_100f=False),
        gen_tgv_gemm_sm10x_module(torch.float16, use_sm_100f=False),
        gen_tgv_gemm_sm10x_module(torch.bfloat16, use_sm_100f=True),
        gen_tgv_gemm_sm10x_module(torch.float16, use_sm_100f=True),
    ]
    assert [spec.name for spec in specs] == [
        "tgv_gemm_bf16_sm100a",
        "tgv_gemm_fp16_sm100a",
        "tgv_gemm_bf16_sm100f",
        "tgv_gemm_fp16_sm100f",
    ]
    assert [spec.sources[1].parent.name for spec in specs] == [
        "gen_tgv_gemm_bf16_sm100a",
        "gen_tgv_gemm_fp16_sm100a",
        "gen_tgv_gemm_bf16_sm100f",
        "gen_tgv_gemm_fp16_sm100f",
    ]
    assert [
        [flag for flag in spec.extra_cuda_cflags if flag.startswith("-gencode=")]
        for spec in specs
    ] == [
        ["-gencode=arch=compute_100a,code=sm_100a"],
        ["-gencode=arch=compute_100a,code=sm_100a"],
        ["-gencode=arch=compute_100f,code=sm_100f"],
        ["-gencode=arch=compute_100f,code=sm_100f"],
    ]

    from flashinfer import aot

    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    for generator_name in [
        "gen_spdlog_module",
        "gen_gemm_module",
        "gen_bgmv_moe_module",
        "gen_hash_topk_module",
        "gen_fp4_quantization_sm100_module",
        "gen_cutlass_fused_moe_sm100_module",
        "gen_gemm_sm100_module",
        "gen_gemm_sm100_module_cutlass_fp4",
        "gen_gemm_sm100_module_cutlass_nvfp4_svdquant",
        "gen_gemm_sm100_module_cutlass_fp8",
        "gen_gemm_sm100_module_cutlass_mxfp8",
        "gen_mxfp8_quantization_sm100_module",
        "gen_trtllm_gen_gemm_module",
        "gen_trtllm_low_latency_gemm_module",
        "gen_trtllm_gen_fused_moe_sm100_module",
        "gen_moe_utils_module",
        "gen_mm_bf16_cublaslt_module",
        "gen_cudnn_fmha_module",
    ]:
        monkeypatch.setattr(
            aot,
            generator_name,
            lambda *args, _name=generator_name, **kwargs: SimpleNamespace(name=_name),
        )

    inventory = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm100": True, "sm100f": True},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )
    assert [spec.name for spec in inventory if spec.name.startswith("tgv_gemm_")] == [
        "tgv_gemm_bf16_sm100a",
        "tgv_gemm_fp16_sm100a",
        "tgv_gemm_bf16_sm100f",
        "tgv_gemm_fp16_sm100f",
    ]
