"""Unit tests for ``materialize_fused_moe_weights``."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _bf16_moe_config(*, num_experts, local_num_experts, offset, intermediate, top_k=4):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=intermediate,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=512),
    )


def _nvfp4_moe_config(*, num_experts, local_num_experts, offset, intermediate, top_k=4):
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmFp4Config,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(
            intermediate_size=intermediate,
            local_expert_offset=offset,
            local_num_experts=local_num_experts,
        ),
        backend=BackendOptions(candidates=(TrtllmFp4Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=512),
    )


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available(),
    reason="CUDA required for weight layout transforms",
)
class TestMaterializeFusedMoeWeights:
    def test_bf16_matches_manual_prepare(self):
        import torch

        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
            materialize_fused_moe_weights,
        )

        device = "cuda"
        local_n, hidden, intermediate = 2, 256, 128
        w13 = torch.randn(local_n, 2 * intermediate, hidden, device=device).to(
            torch.bfloat16
        )
        w2 = torch.randn(local_n, hidden, intermediate, device=device).to(
            torch.bfloat16
        )
        cfg = _bf16_moe_config(
            num_experts=4,
            local_num_experts=local_n,
            offset=0,
            intermediate=intermediate,
        )
        canonical = MoEWeightPack(w13=w13, w2=w2)

        pack_a = materialize_fused_moe_weights(canonical, cfg)
        pack_b = materialize_fused_moe_weights(canonical, cfg)
        view_a = pack_a.get_view("trtllm_bf16_routed")
        view_b = pack_b.get_view("trtllm_bf16_routed")
        for key in ("gemm1_weights", "gemm2_weights"):
            torch.testing.assert_close(view_a[key], view_b[key])
        assert view_a["gemm1_weights"].dtype == torch.bfloat16

    def test_bf16_view_keys(self):
        import torch

        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
            materialize_fused_moe_weights,
        )

        device = "cuda"
        local_n, hidden, intermediate = 2, 256, 128
        w13 = torch.randn(local_n, 2 * intermediate, hidden, device=device).to(
            torch.bfloat16
        )
        w2 = torch.randn(local_n, hidden, intermediate, device=device).to(
            torch.bfloat16
        )
        cfg = _bf16_moe_config(
            num_experts=4,
            local_num_experts=local_n,
            offset=0,
            intermediate=intermediate,
        )
        pack = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)
        view = pack.get_view("trtllm_bf16_routed")
        assert set(view) == {"gemm1_weights", "gemm2_weights"}

    @pytest.mark.skipif(
        not __import__("torch").cuda.is_available()
        or __import__("torch").cuda.get_device_capability()[0] < 10,
        reason="NVFP4 weight prep needs SM100+",
    )
    def test_nvfp4_trtllm_matches_manual_prepare(self):
        import torch

        from flashinfer.fused_moe.api import (
            MoEWeightPack as FusedMoEWeightPack,
            TrtllmFp4Config,
        )
        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
            materialize_fused_moe_weights,
        )

        device = "cuda"
        local_n, hidden, intermediate = 2, 256, 128
        w13 = (
            torch.randn(local_n, 2 * intermediate, hidden, device=device).to(
                torch.bfloat16
            )
            * 0.1
        )
        w2 = (
            torch.randn(local_n, hidden, intermediate, device=device).to(torch.bfloat16)
            * 0.1
        )
        cfg = _nvfp4_moe_config(
            num_experts=4,
            local_num_experts=local_n,
            offset=0,
            intermediate=intermediate,
        )

        got = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)
        manual = FusedMoEWeightPack()
        manual.prepare_for(
            "trtllm_fp4_routed",
            TrtllmFp4Config.prepare_weights(
                w13,
                w2,
                num_local_experts=local_n,
                hidden_size=hidden,
                intermediate_size=intermediate,
                device=device,
            ),
        )
        got_view = got.get_view("trtllm_fp4_routed")
        manual_view = manual.get_view("trtllm_fp4_routed")
        assert set(got_view) == set(manual_view)
        for key in manual_view:
            if got_view[key].dtype.is_floating_point:
                torch.testing.assert_close(got_view[key], manual_view[key])
            else:
                assert torch.equal(got_view[key], manual_view[key])

    def test_unsupported_variant_raises(self):
        import torch

        from flashinfer.fused_moe.api import (
            BackendOptions,
            ExecutionConfig,
            ExpertConfig,
            MoEConfig,
            QuantConfig,
            QuantVariant,
            RoutingConfig,
            TrtllmBf16Config,
        )
        from flashinfer.moe_ep import MoEWeightPack
        from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
            materialize_fused_moe_weights,
        )

        device = "cuda"
        w13 = torch.randn(1, 256, 128, device=device).to(torch.bfloat16)
        w2 = torch.randn(1, 128, 128, device=device).to(torch.bfloat16)
        cfg = MoEConfig(
            routing=RoutingConfig(num_experts=2, top_k=2),
            quant=QuantConfig(variant=QuantVariant.NVFP4),
            experts=ExpertConfig(intermediate_size=128, local_num_experts=1),
            backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
            execution=ExecutionConfig(tune_max_num_tokens=64),
        )
        with pytest.raises(ValueError, match="No fused_moe backend"):
            materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)
