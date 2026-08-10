"""Integration contracts for the MXFP8 x MXFP4 CuTeDSL fused-MoE API."""

import inspect

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available


def _is_sm100_family() -> bool:
    return bool(
        torch.cuda.is_available() and torch.cuda.get_device_properties(0).major == 10
    )


cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="CuTeDSL is not available"
)
sm100_required = pytest.mark.skipif(
    not _is_sm100_family(), reason="Requires an SM100-family GPU"
)


@cute_dsl_available
class TestMxfp8Mxfp4FusedMoeContracts:
    def test_tactic_namespace_is_separate_and_complete(self):
        from flashinfer.fused_moe.cute_dsl.mixed_tuner import (
            ALL_MXFP8_MXFP4_MOE_TACTICS,
            CuteDslFusedMoEMxfp8Mxfp4Runner,
        )
        from flashinfer.fused_moe.cute_dsl.tuner import (
            ALL_MOE_TACTICS,
            CuteDslFusedMoENvfp4Runner,
        )

        assert len(ALL_MXFP8_MXFP4_MOE_TACTICS) == 32
        assert len(set(ALL_MXFP8_MXFP4_MOE_TACTICS)) == 32
        assert len(ALL_MOE_TACTICS) == 16
        assert CuteDslFusedMoEMxfp8Mxfp4Runner is not CuteDslFusedMoENvfp4Runner
        assert {tactic[2][0][1] for tactic in ALL_MXFP8_MXFP4_MOE_TACTICS} == {
            64,
            128,
            192,
            256,
        }
        assert {tactic[2][0][1] for tactic in ALL_MOE_TACTICS} == {128, 256}

    def test_public_signature_has_no_nvfp4_fc2_scale(self):
        from flashinfer import cute_dsl_fused_moe_mxfp8_mxfp4
        from flashinfer.fused_moe.cute_dsl import (
            CuteDslMxfp8Mxfp4MoEWrapper,
        )

        functional_params = inspect.signature(cute_dsl_fused_moe_mxfp8_mxfp4).parameters
        wrapper_params = inspect.signature(CuteDslMxfp8Mxfp4MoEWrapper.run).parameters
        assert "fc2_input_scale" not in functional_params
        assert "fc2_input_scale" not in wrapper_params
        assert "tactic" in functional_params
        assert "tactic" in wrapper_params

    def test_json_tactics_are_canonicalized_and_checked(self):
        from flashinfer.fused_moe.cute_dsl.mixed_tuner import (
            ALL_MXFP8_MXFP4_MOE_TACTICS,
            canonicalize_mxfp8_mxfp4_tactic,
        )

        tactic = ALL_MXFP8_MXFP4_MOE_TACTICS[5]
        json_form = [
            tactic[0],
            [list(tactic[1][0]), list(tactic[1][1]), False],
            [list(tactic[2][0]), list(tactic[2][1]), False],
        ]
        assert canonicalize_mxfp8_mxfp4_tactic(json_form) == tactic
        with pytest.raises(ValueError, match="unsupported"):
            canonicalize_mxfp8_mxfp4_tactic(
                [128, [[128, 128], [1, 1], False], [[128, 96], [1, 1], False]]
            )


def _interleave_linear_and_gate(x: torch.Tensor) -> torch.Tensor:
    experts, rows, k = x.shape
    intermediate = rows // 2
    return (
        x.view(experts, 2, intermediate // 64, 64, k)
        .transpose(1, 2)
        .contiguous()
        .view_as(x)
    )


def _quantize_mxfp4_grouped(weights: torch.Tensor):
    from flashinfer import fp4_quantize
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    experts, rows, k = weights.shape
    packed, scale = fp4_quantize(
        weights.reshape(experts * rows, k).contiguous(),
        global_scale=torch.ones(1, dtype=torch.float32, device=weights.device),
        sf_vec_size=32,
        sf_use_ue8m0=True,
        is_sf_swizzled_layout=True,
    )
    return packed.view(experts, rows, k // 2), convert_sf_to_mma_layout(
        scale,
        m=rows,
        k=k,
        num_groups=experts,
        sf_vec_size=32,
    )


def _make_inputs():
    from flashinfer import mxfp8_quantize

    torch.manual_seed(20260720)
    device = torch.device("cuda")
    num_tokens, num_experts, top_k = 17, 2, 2
    hidden_size, intermediate_size = 256, 128
    x_bf16 = torch.randn(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=device
    ).mul_(0.5)
    x, x_sf = mxfp8_quantize(x_bf16, is_sf_swizzled_layout=False, alignment=256)
    x_sf = x_sf.view(torch.uint8).reshape(num_tokens, hidden_size // 32)
    selected = (
        torch.arange(num_experts, dtype=torch.int32, device=device)
        .expand(num_tokens, top_k)
        .contiguous()
    )
    final_scales = (
        torch.tensor([0.4, 0.6], dtype=torch.float32, device=device)
        .expand(num_tokens, top_k)
        .contiguous()
    )
    w1 = torch.randn(
        num_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.25)
    w2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    ).mul_(0.25)
    w1_q, w1_sf = _quantize_mxfp4_grouped(_interleave_linear_and_gate(w1))
    w2_q, w2_sf = _quantize_mxfp4_grouped(w2)
    alpha = torch.ones(num_experts, dtype=torch.float32, device=device)
    return {
        "x": x,
        "x_sf": x_sf,
        "token_selected_experts": selected,
        "token_final_scales": final_scales,
        "w1_weight": w1_q,
        "w1_weight_sf": w1_sf,
        "w1_alpha": alpha,
        "w2_weight": w2_q,
        "w2_weight_sf": w2_sf,
        "w2_alpha": alpha,
        "num_experts": num_experts,
        "top_k": top_k,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_tokens": num_tokens,
    }


@cute_dsl_available
@sm100_required
class TestMxfp8Mxfp4FusedMoeEndToEnd:
    def test_functional_and_preallocated_wrapper_agree(self, monkeypatch):
        from flashinfer import (
            CuteDslMxfp8Mxfp4MoEWrapper,
            cute_dsl_fused_moe_mxfp8_mxfp4,
        )
        from flashinfer.autotuner import AutoTuner
        from flashinfer.fused_moe.cute_dsl.mixed_tuner import (
            DEFAULT_MXFP8_MXFP4_MOE_TACTIC,
        )

        tensors = _make_inputs()
        call_args = {
            key: tensors[key]
            for key in (
                "x",
                "x_sf",
                "token_selected_experts",
                "token_final_scales",
                "w1_weight",
                "w1_weight_sf",
                "w1_alpha",
                "w2_weight",
                "w2_weight_sf",
                "w2_alpha",
            )
        }
        functional = cute_dsl_fused_moe_mxfp8_mxfp4(
            **call_args,
            num_experts=tensors["num_experts"],
            top_k=tensors["top_k"],
            tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC,
            enable_pdl=False,
        )

        wrapper = CuteDslMxfp8Mxfp4MoEWrapper(
            num_experts=tensors["num_experts"],
            top_k=tensors["top_k"],
            hidden_size=tensors["hidden_size"],
            intermediate_size=tensors["intermediate_size"],
            max_num_tokens=tensors["num_tokens"],
            enable_pdl=False,
        )
        monkeypatch.setattr(
            AutoTuner,
            "choose_one",
            lambda *args, **kwargs: pytest.fail(
                "explicit-tactic wrapper call must bypass AutoTuner.choose_one"
            ),
        )
        wrapper_result_1 = wrapper.run(
            **call_args, tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC
        )
        torch.cuda.synchronize()
        first_ptr = wrapper_result_1.data_ptr()
        snapshot = wrapper_result_1.clone()
        wrapper_result_2 = wrapper.run(
            **call_args, tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC
        )
        torch.cuda.synchronize()

        assert wrapper_result_2.data_ptr() == first_ptr
        assert (
            wrapper_result_2.shape
            == functional.shape
            == (
                tensors["num_tokens"],
                tensors["hidden_size"],
            )
        )
        assert wrapper_result_2.dtype is torch.bfloat16
        assert torch.isfinite(wrapper_result_2).all()
        assert torch.allclose(snapshot, wrapper_result_2, atol=0.5, rtol=0.05)
        assert torch.allclose(functional, wrapper_result_2, atol=0.5, rtol=0.05)

        # A capacity-backed wrapper must also produce correctly-sized views
        # for smaller active batches without reallocating its output storage.
        smaller_args = dict(call_args)
        for name in (
            "x",
            "x_sf",
            "token_selected_experts",
            "token_final_scales",
        ):
            smaller_args[name] = call_args[name][:9]
        smaller_result = wrapper.run(
            **smaller_args, tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC
        )
        torch.cuda.synchronize()
        assert smaller_result.shape == (9, tensors["hidden_size"])
        assert smaller_result.data_ptr() == first_ptr
        assert torch.isfinite(smaller_result).all()

        other_stream = torch.cuda.Stream()
        with (
            torch.cuda.stream(other_stream),
            pytest.raises(RuntimeError, match="bound to the CUDA stream"),
        ):
            wrapper.run(**call_args, tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC)

    def test_rejects_nvfp4_style_contract(self):
        from flashinfer import cute_dsl_fused_moe_mxfp8_mxfp4
        from flashinfer.fused_moe.cute_dsl.mixed_tuner import (
            DEFAULT_MXFP8_MXFP4_MOE_TACTIC,
        )

        tensors = _make_inputs()
        with pytest.raises(TypeError, match="token_final_scales"):
            cute_dsl_fused_moe_mxfp8_mxfp4(
                x=tensors["x"],
                x_sf=tensors["x_sf"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"].to(torch.bfloat16),
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"],
                w1_alpha=tensors["w1_alpha"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                num_experts=tensors["num_experts"],
                top_k=tensors["top_k"],
                tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC,
            )

        with pytest.raises(ValueError, match="MMA scale strides"):
            cute_dsl_fused_moe_mxfp8_mxfp4(
                x=tensors["x"],
                x_sf=tensors["x_sf"],
                token_selected_experts=tensors["token_selected_experts"],
                token_final_scales=tensors["token_final_scales"],
                w1_weight=tensors["w1_weight"],
                w1_weight_sf=tensors["w1_weight_sf"].contiguous(),
                w1_alpha=tensors["w1_alpha"],
                w2_weight=tensors["w2_weight"],
                w2_weight_sf=tensors["w2_weight_sf"],
                w2_alpha=tensors["w2_alpha"],
                num_experts=tensors["num_experts"],
                top_k=tensors["top_k"],
                tactic=DEFAULT_MXFP8_MXFP4_MOE_TACTIC,
            )


if __name__ == "__main__":
    pytest.main([__file__])
