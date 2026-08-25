"""Focused source/CPU gates for NVFP4 CuTeDSL MegaMoE ReLU2 support."""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace
from unittest import mock

import pytest

pytest.importorskip("flashinfer.moe_ep.kernel_src.cutedsl_megamoe")


def test_public_config_preserves_swiglu_default_and_legacy_positions():
    from flashinfer.moe_ep import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    # The fourth/fifth positional slots predate activation and must remain the
    # clamp aliases; activation was appended to preserve this construction.
    cfg = Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        128, 2, "legacy-kernel-name", 3.0, 3.0
    )
    assert cfg.kernel_name == "legacy-kernel-name"
    assert cfg.gate_up_clamp == 3.0
    assert cfg.activation_clamp == 3.0
    assert cfg.activation == "swiglu"


def test_shim_config_preserves_legacy_positions_and_separates_compile_key():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Config,
        MegaMoENvfp4Frontend,
    )

    # The first optional positional slot remains mma_tiler_mnk.
    legacy = MegaMoENvfp4Config(0, 1, 64, 2, 4, 128, 256, (128, 128, 256))
    assert legacy.mma_tiler_mnk == (128, 128, 256)
    assert legacy.activation == "swiglu"

    relu2 = MegaMoENvfp4Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=64,
        num_topk=2,
        num_total_experts=4,
        hidden=128,
        intermediate=256,
        activation="relu2",
    )
    assert (
        MegaMoENvfp4Frontend(legacy)._mega_compile_key()
        != MegaMoENvfp4Frontend(relu2)._mega_compile_key()
    )


def test_relu2_config_rejects_clamps_and_unknown_activation():
    from flashinfer.moe_ep import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.nvfp4 import (
        MegaMoENvfp4Config,
    )

    with pytest.raises(ValueError, match="does not support"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            activation="relu2",
            gate_up_clamp=1.0,
        )
    with pytest.raises(ValueError, match="activation must"):
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
            activation="gelu",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="does not support"):
        MegaMoENvfp4Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=2,
            num_total_experts=4,
            hidden=128,
            intermediate=256,
            activation="relu2",
            gate_up_clamp=1.0,
        )


def test_nemotron_semantic_fc1_is_padded_to_exact_internal_geometry():
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        _pad_relu2_fc1_plane,
    )

    semantic_i = 2688
    semantic = torch.arange(semantic_i * 4, dtype=torch.int64).view(1, semantic_i, 4)
    padded = _pad_relu2_fc1_plane(semantic, intermediate_size=semantic_i, name="w13")
    assert padded.shape == (1, 5376, 4)
    torch.testing.assert_close(padded[:, :semantic_i], semantic)
    assert torch.count_nonzero(padded[:, semantic_i:]) == 0

    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4_dtype is not None:
        packed = torch.ones(1, semantic_i, 4, dtype=torch.uint8).view(fp4_dtype)
        packed_padded = _pad_relu2_fc1_plane(
            packed, intermediate_size=semantic_i, name="packed_w13"
        )
        assert packed_padded.shape == (1, 5376, 4)
        assert torch.count_nonzero(packed_padded.view(torch.uint8)[:, semantic_i:]) == 0

    with pytest.raises(ValueError, match="semantic intermediate dimension 2688"):
        _pad_relu2_fc1_plane(
            padded,
            intermediate_size=semantic_i,
            name="w13",
        )


def test_packed_modelopt_relu2_weights_expand_and_interleave():
    import torch

    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
    )

    experts, semantic_i, hidden = 1, 64, 64
    pack = MoEWeightPack(
        w13=torch.full((experts, semantic_i, hidden // 2), 0x11, dtype=torch.uint8),
        w2=torch.full((experts, hidden, semantic_i // 2), 0x22, dtype=torch.uint8),
        w13_scale=torch.ones(experts, semantic_i, hidden // 16, dtype=torch.uint8),
        w2_scale=torch.ones(experts, hidden, semantic_i // 16, dtype=torch.uint8),
    )
    (fc1_weight, fc1_sf), (fc2_weight, fc2_sf) = preprocess_mega_weights(
        pack,
        intermediate_size=semantic_i,
        hidden_size=hidden,
        activation="relu2",
    )
    assert fc1_weight.shape == (experts, hidden // 2, 2 * semantic_i)
    assert fc2_weight.shape == (experts, semantic_i // 2, hidden)
    assert fc1_sf.shape[0] == experts
    assert fc2_sf.shape[0] == experts

    # FC1 columns are 16 semantic W1 values then 16 zero padding values.
    fc1_bytes = fc1_weight.view(torch.uint8)
    for offset in range(0, 2 * semantic_i, 32):
        assert torch.all(fc1_bytes[:, :, offset : offset + 16] == 0x11)
        assert torch.count_nonzero(fc1_bytes[:, :, offset + 16 : offset + 32]) == 0


def test_relu2_reference_proves_padding_plane_is_ignored():
    import torch

    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        relu2_reference_from_internal_fc1,
    )

    semantic = torch.tensor([[-3.0, -0.5, 0.0, 2.0]])
    padding_a = torch.tensor([[1e20, -1e20, 7.0, -9.0]])
    padding_b = torch.randn_like(padding_a) * 1e6
    out_a = relu2_reference_from_internal_fc1(
        torch.cat((semantic, padding_a), dim=-1), alpha=2.0
    )
    out_b = relu2_reference_from_internal_fc1(
        torch.cat((semantic, padding_b), dim=-1), alpha=2.0
    )
    torch.testing.assert_close(out_a, out_b)
    torch.testing.assert_close(out_a, torch.tensor([[0.0, 0.0, 0.0, 16.0]]))


def test_device_specialization_never_indexes_padding_values():
    import cutlass.cute as cute

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.relu2 import (
        SwapABRelu2Fc1Epilogue,
        SwapABRelu2Fp4Epilogue,
    )

    activation_tree = ast.parse(
        textwrap.dedent(inspect.getsource(SwapABRelu2Fc1Epilogue.alpha_swiglu_clamp))
    )
    padding_reads = [
        node
        for node in ast.walk(activation_tree)
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "up_rmem"
    ]
    assert not padding_reads, "ReLU2 device math must not read padding values"

    run_source = inspect.getsource(SwapABRelu2Fp4Epilogue.run)
    assert "SwapABRelu2Fc1Epilogue(" in run_source
    assert "SwapABFc1Epilogue(" not in run_source

    # CuTeDSL consumes live annotations from @cute.jit functions.  A future
    # annotations import silently turns these into strings and breaks tracing.
    run_signature = inspect.signature(SwapABRelu2Fp4Epilogue.run)
    assert run_signature.parameters["fc1_output"].annotation is cute.Tensor


def test_relu2_kernel_adapter_preserves_exact_vendor_callable(monkeypatch):
    from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import relu2

    structural_values = {
        field: index for index, field in enumerate(relu2._STRUCTURAL_EPILOGUE_FIELDS)
    }
    vendor_epilogue = SimpleNamespace(**structural_values)

    def fake_vendor_setup(kernel):
        kernel.epilogue = vendor_epilogue

    class FakeRelu2Epilogue:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            for field, value in structural_values.items():
                setattr(self, field, value)

    monkeypatch.setattr(Sm100MegaMoEKernel, "_setup_attributes", fake_vendor_setup)
    monkeypatch.setattr(Sm100MegaMoEKernel, "name", lambda self: "vendor-name")
    monkeypatch.setattr(relu2, "SwapABRelu2Fp4Epilogue", FakeRelu2Epilogue)

    kernel = object.__new__(Sm100MegaMoEKernel)
    kernel.gate_up_clamp = None
    kernel.mma_tiler = (128, 128, 256)
    kernel.cluster_shape_mn = (1, 1)
    kernel.use_2cta_instrs = False
    kernel.sf_vec_size = 16
    kernel.fc1_output_dtype = object()
    kernel.combine_format = object()
    kernel.non_ubulk_fc2_store = True
    kernel.in_kernel_fc2_reduce = False
    kernel.token_back_by_dispatch = False
    kernel.epi_flag_batch = (1, 1)
    kernel.acc_dtype = object()
    kernel.static_expert_shape = (32, 5376, 1024)

    adapted = relu2.configure_sm100_relu2_megamoe_kernel(kernel)
    assert type(adapted) is Sm100MegaMoEKernel
    assert adapted.__call__.__func__ is Sm100MegaMoEKernel.__call__
    assert adapted.name() == "vendor-name_activation_relu2"
    assert adapted._flashinfer_activation == "relu2"

    adapted._setup_attributes()
    assert isinstance(adapted.epilogue, FakeRelu2Epilogue)
    assert adapted.epilogue.kwargs["gate_up_clamp"] is None

    class UnsupportedSubclass(Sm100MegaMoEKernel):
        pass

    with pytest.raises(TypeError, match="exact Sm100MegaMoEKernel"):
        relu2.configure_sm100_relu2_megamoe_kernel(object.__new__(UnsupportedSubclass))


def test_workspace_pool_key_contains_activation():
    import torch

    from flashinfer.moe_ep import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.backend import (
        Nvfp4CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.config import (
        Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    fp = FleetParams(
        num_experts=512,
        max_tokens_per_rank=128,
        token_hidden_size=1024,
    )
    bootstrap = BootstrapConfig(world_size=4, rank=0, auto_bootstrap=False)
    kernels = []
    for activation in ("swiglu", "relu2"):
        kernel = Nvfp4CutedslMegaKernelBackend(
            Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=2688,
                top_k=22,
                activation=activation,
            )
        )
        kernel.bind_ep_bootstrap(bootstrap)
        kernels.append(kernel)
    with mock.patch.object(torch.cuda, "current_device", return_value=0):
        swiglu_key = kernels[0]._workspace_pool_key(fp)
        relu2_key = kernels[1]._workspace_pool_key(fp)
    assert swiglu_key != relu2_key
    assert swiglu_key[9:11] == (5376, "swiglu")
    assert relu2_key[9:11] == (5376, "relu2")


def test_knob_cache_never_crosses_activation(monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
        lookup_knobs,
        record_knobs,
    )

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    key = dict(
        dtype="nvfp4",
        world_size=4,
        hidden=1024,
        intermediate=5376,
        num_experts=512,
        topk=22,
        max_tokens=128,
        combine_dtype="bf16",
        device="GB200-test",
    )
    swiglu = {"flag_batch": 4}
    relu2 = {"flag_batch": 8}
    record_knobs(swiglu, activation="swiglu", **key)
    assert lookup_knobs(activation="swiglu", **key) == swiglu
    assert lookup_knobs(activation="relu2", **key) is None
    record_knobs(relu2, activation="relu2", **key)
    assert lookup_knobs(activation="swiglu", **key) == swiglu
    assert lookup_knobs(activation="relu2", **key) == relu2
