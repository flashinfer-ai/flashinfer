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
"""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available
from flashinfer.cute_dsl.utils import is_cute_dsl_arch_supported
from flashinfer.tllm_enums import ActivationType

from .utils import create_moe_tensors


def _is_sm100_family() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() in ((10, 0), (10, 3))


pytestmark = [
    pytest.mark.skipif(
        not _is_sm100_family(), reason="requires the SM100/SM103 GEMM1 kernel"
    ),
    pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe DSL unavailable"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and not is_cute_dsl_arch_supported(
            *torch.cuda.get_device_capability(), native_only=True
        ),
        reason="installed CuTe DSL cannot target the current GPU",
    ),
]


def _set_quant_mode(monkeypatch: pytest.MonkeyPatch, deterministic: bool) -> None:
    values = {
        "FLASHINFER_NVFP4_4OVER6": "1" if deterministic else "0",
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1" if deterministic else "0",
        "FLASHINFER_NVFP4_4OVER6_ERR_MODE": "MSE",
        "FLASHINFER_NVFP4_4OVER6_ERR_USE_FAST_MATH": "1" if deterministic else "0",
        "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1" if deterministic else "0",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


@pytest.mark.parametrize("deterministic_quant", [False, True])
@pytest.mark.parametrize(
    ("tile_size", "gemm1_n", "activation_type", "gated"),
    [
        pytest.param(128, 128, ActivationType.Swiglu, True, id="m128-n128-swiglu"),
        pytest.param(128, 256, ActivationType.Swiglu, True, id="m128-n256-swiglu"),
        pytest.param(256, 128, ActivationType.Relu2, False, id="m256-n128-relu2"),
        pytest.param(256, 256, ActivationType.Relu2, False, id="m256-n256-relu2"),
    ],
)
def test_gemm1_aux_amax_and_per_token_output_are_bitwise_equal(
    monkeypatch: pytest.MonkeyPatch,
    tile_size: int,
    gemm1_n: int,
    activation_type: ActivationType,
    gated: bool,
    deterministic_quant: bool,
):
    """Check producer values and the exact legacy/accelerated MoE boundary."""
    import flashinfer.fused_moe.cute_dsl.fused_moe as fused_moe_module

    _set_quant_mode(monkeypatch, deterministic_quant)

    num_tokens, top_k, num_experts = 8, 2, 16
    hidden_size, intermediate_size = 256, 512
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
        gated=gated,
        use_per_token_activation=True,
    )

    # Give every expert exactly one routed row. The actual and maximum tile
    # counts are then identical, so every materialized row and every aux cell
    # is written and can be checked without consulting internal sort metadata.
    tensors["token_selected_experts"].copy_(
        torch.arange(
            num_tokens * top_k,
            device="cuda",
            dtype=torch.int32,
        ).reshape(num_tokens, top_k)
    )
    tensors["token_final_scales"].copy_(
        torch.tensor([0.375, 0.625], device="cuda", dtype=torch.float32).repeat(
            num_tokens, 1
        )
    )

    kwargs = {
        "x": tensors["x"],
        "x_sf": tensors["x_sf"],
        "token_selected_experts": tensors["token_selected_experts"],
        "token_final_scales": tensors["token_final_scales"],
        "w1_weight": tensors["w1_weight"],
        "w1_weight_sf": tensors["w1_weight_sf"],
        "w1_alpha": tensors["w1_alpha"],
        "fc2_input_scale": tensors["fc2_input_scale"],
        "w2_weight": tensors["w2_weight"],
        "w2_weight_sf": tensors["w2_weight_sf"],
        "w2_alpha": tensors["w2_alpha"],
        "num_experts": num_experts,
        "top_k": top_k,
        "num_local_experts": num_experts,
        "tile_size": tile_size,
        "gemm1_mma_tiler_mn": (tile_size, gemm1_n),
        "gemm1_cluster_shape_mn": (tile_size // 128, 1),
        "gemm2_mma_tiler_mn": (tile_size, 128),
        "gemm2_cluster_shape_mn": (tile_size // 128, 1),
        "use_async_memset": False,
        "use_fused_finalize": False,
        "enable_pdl": True,
        "activation_type": activation_type.value,
        "per_token_scale": tensors["x_per_token_scale"],
    }

    monkeypatch.setenv("FLASHINFER_CUTEDSL_MOE_PER_TOKEN_AUX_AMAX", "0")
    legacy_output = fused_moe_module._moe_core_impl(**kwargs)

    original_quantize = fused_moe_module.nvfp4_quantize_per_token_cute_dsl
    calls_checked = 0

    def checked_quantize(
        input: torch.Tensor,
        global_scale_inv: torch.Tensor,
        sf_layout: int,
        enable_pdl: bool,
        input_amax: torch.Tensor | None = None,
    ):
        nonlocal calls_checked
        assert input_amax is not None
        output_tile_n = gemm1_n // (2 if gated else 1)
        expected_amax = (
            input.float()
            .abs()
            .reshape(input.shape[0], input.shape[1] // output_tile_n, output_tile_n)
            .amax(dim=2)
        )
        assert torch.equal(input_amax, expected_amax)

        legacy_quantized = original_quantize(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=False,
        )
        accelerated_quantized = original_quantize(
            input,
            global_scale_inv,
            sf_layout=sf_layout,
            enable_pdl=enable_pdl,
            input_amax=input_amax,
        )
        for accelerated, legacy in zip(
            accelerated_quantized, legacy_quantized, strict=True
        ):
            assert torch.equal(accelerated, legacy)
        calls_checked += 1
        return accelerated_quantized

    monkeypatch.setattr(
        fused_moe_module,
        "nvfp4_quantize_per_token_cute_dsl",
        checked_quantize,
    )
    monkeypatch.setenv("FLASHINFER_CUTEDSL_MOE_PER_TOKEN_AUX_AMAX", "1")
    accelerated_output = fused_moe_module._moe_core_impl(**kwargs)

    assert calls_checked == 1
    assert torch.equal(accelerated_output, legacy_output)
