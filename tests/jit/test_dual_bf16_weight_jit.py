# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace


def test_dual_bf16_weight_module_is_registered_for_sm100_aot(monkeypatch):
    from flashinfer import aot

    target_name = "dual_bf16_weight_gemm_sm100"
    stubbed_generators = [
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
        "gen_tgv_gemm_sm10x_module",
        "gen_mxfp8_quantization_sm100_module",
        "gen_trtllm_gen_gemm_module",
        "gen_trtllm_low_latency_gemm_module",
        "gen_trtllm_gen_fused_moe_sm100_module",
        "gen_trtllm_gen_routing_module",
        "gen_mm_bf16_cublaslt_module",
    ]
    for name in stubbed_generators:
        monkeypatch.setattr(
            aot,
            name,
            lambda *args, _name=name, **kwargs: SimpleNamespace(name=_name),
        )
    monkeypatch.setattr(
        aot,
        "gen_dual_bf16_weight_gemm_sm100_module",
        lambda: SimpleNamespace(name=target_name),
    )
    monkeypatch.setattr(
        aot, "gen_spdlog_module", lambda: SimpleNamespace(name="spdlog")
    )
    monkeypatch.setattr(aot, "gen_attention", lambda *args: ())
    monkeypatch.setattr(
        aot, "gen_cudnn_fmha_module", lambda: SimpleNamespace(name="cudnn")
    )

    specs = aot.gen_all_modules(
        [],
        [],
        [],
        [],
        [],
        [],
        {"sm100": True},
        False,
        False,
        False,
        True,
        False,
        False,
        False,
    )

    assert [spec.name for spec in specs].count(target_name) == 1
