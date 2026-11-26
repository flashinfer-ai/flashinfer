# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Preparation and constructor contracts for the combined provider export."""

from types import SimpleNamespace

import pytest
import torch


def test_public_nvfp4_preparation_matches_default_selector():
    from flashinfer import moe_ep
    from flashinfer.moe_ep.backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import (
        Sm100BlockScaledMoeConfig,
        validate_standard_weights,
    )

    weights = moe_ep.MoEWeightPack(
        torch.zeros(1, 256, 128, dtype=torch.uint8),
        torch.zeros(1, 256, 64, dtype=torch.uint8),
        torch.full((1, 256, 16), 127, dtype=torch.uint8),
        torch.full((1, 256, 8), 127, dtype=torch.uint8),
    )
    dimensions = dict(hidden_size=256, intermediate_size=128)
    transformed = moe_ep.preprocess_nvfp4_cutedsl_mega_weights(weights, **dimensions)
    config = Sm100BlockScaledMoeConfig(
        num_total_experts=1,
        max_tokens_per_rank=16,
        num_topk=1,
        hidden=256,
        intermediate=128,
        rank=0,
        world_size=1,
        quant_kind="nvfp4",
    )
    validate_standard_weights(transformed, config, torch.device("cpu"))
    legacy = preprocess_mega_weights(weights, **dimensions)
    validate_transformed_mega_weights(legacy, world_size=1, num_experts=1, **dimensions)
    for default_leg, legacy_leg in zip(transformed, legacy, strict=True):
        assert default_leg[1].dtype == torch.float8_e4m3fn
        assert legacy_leg[1].dtype == torch.uint8


@pytest.mark.parametrize("variant", ["inference", "genphase"])
@pytest.mark.parametrize("scale_dtype", [torch.uint8, torch.float8_e4m3fn, torch.int8])
def test_sm100_nvfp4_loads_transformed_scale_views(monkeypatch, variant, scale_dtype):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.backends.mega.kernel.sm100 import standard_cutedsl
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe import sources

    validate = sources.validate_standard_weights
    monkeypatch.setattr(standard_cutedsl, "validate_mega_arch", lambda: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        sources,
        "validate_standard_weights",
        lambda weights, config, device: validate(weights, config, torch.device("cpu")),
    )

    pack = moe_ep.MoEWeightPack(
        torch.zeros(1, 256, 128, dtype=torch.uint8),
        torch.zeros(1, 256, 64, dtype=torch.uint8),
        torch.full((1, 256, 16), 56, dtype=torch.uint8),
        torch.full((1, 256, 8), 56, dtype=torch.uint8),
    )
    typed = moe_ep.preprocess_nvfp4_cutedsl_mega_weights(
        pack, hidden_size=256, intermediate_size=128
    )
    transformed = tuple((weight, scale.view(scale_dtype)) for weight, scale in typed)
    args = (
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        moe_ep.FleetParams(1, 16, 256),
        None,
        moe_ep.MegaConfig(
            megakernel=moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=128, top_k=1, kernel_variant=variant
            ),
            preprocess_weights=False,
            transformed_weights=transformed,
        ),
    )
    if scale_dtype == torch.int8:
        with pytest.raises(ValueError, match="requires .*float8_e4m3fn scales"):
            moe_ep.MoEEpMegaLayer(*args)
        return

    layer = moe_ep.MoEEpMegaLayer(*args)
    try:
        prepared = layer.get_transformed_weights()
        if scale_dtype == torch.float8_e4m3fn:
            assert prepared is transformed
        for (weight, scale), (original_w, original_sf) in zip(
            prepared, transformed, strict=True
        ):
            assert weight is original_w
            assert scale.dtype == torch.float8_e4m3fn
            assert scale.data_ptr() == original_sf.data_ptr()
    finally:
        layer.destroy()


@pytest.mark.parametrize("name", ["bf16", "bf16_mxfp8"])
@pytest.mark.parametrize(
    "positional, keywords, expected",
    [
        ((), {}, False),
        ((True,), {}, True),
        ((False,), {}, False),
        ((True,), {"apply_routing_weights_before_fc2": False}, False),
        ((False,), {"apply_topk_in_fc1": True}, True),
        (
            (),
            {"apply_topk_in_fc1": True, "apply_routing_weights_before_fc2": False},
            True,
        ),
    ],
)
def test_reference_routing_binding(monkeypatch, name, positional, keywords, expected):
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification import (
        kernel_helpers,
    )

    def reference(a, b, c, d, e, f, g, h, apply_topk_in_fc1=True):
        return apply_topk_in_fc1

    module = SimpleNamespace(
        compute_megamoe_reference=reference,
        compute_megamoe_reference_mxfp8_bf16=reference,
    )
    monkeypatch.setattr(kernel_helpers.importlib, "import_module", lambda _: module)
    function = getattr(kernel_helpers, "compute_megamoe_reference_" + name)
    assert function(*([None] * 8), *positional, **keywords) is expected


@pytest.mark.parametrize("hidden, static, width", [(12, False, 4), (256, True, 8)])
def test_sm90_bf16_reducer_keeps_narrow_fallback(hidden, static, width):
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.topk_reduce import (
        CombineFormat,
        TopkReduce,
    )

    reducer = TopkReduce(hidden, 2, CombineFormat.parse("bf16"), sm_arch="sm_90")
    assert reducer._use_sm90_static is static
    assert reducer.hidden_per_thread == width


@pytest.mark.parametrize(
    "world_size, topk, expert_count, error",
    [
        (256, 1, 256, None),
        (257, 1, 257, "world_size must not exceed"),
        (1, 256, 512, None),
        (1, 257, 512, "topk must not exceed"),
    ],
)
def test_genphase_routing_metadata_limits(world_size, topk, expert_count, error):
    import cutlass

    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.api import (
        ImplDesc,
        ProblemDesc,
    )
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.sources.kernel_src.blackwell.inference.genphase.token_comm_gen_specialized import (
        GenphaseTokenComm,
    )

    problem = ProblemDesc(
        dict(
            world_size=world_size,
            expert_count=expert_count,
            topk=topk,
            max_tokens_per_rank=4,
            hidden_size=256,
            quant_kind="nvfp4",
            topk_index_dtype=cutlass.Int32,
            apply_topk_at_fc1=False,
        )
    )
    implementation = ImplDesc(
        dict(
            token_padding_block=128,
            sf_padding_block=128,
            pusher_cta_count=24,
            refine_participant_groups=2,
            refine_output_stages=2,
        )
    )
    if error is not None:
        with pytest.raises(ValueError, match=error):
            GenphaseTokenComm(problem, implementation)
    else:
        comm = GenphaseTokenComm(problem, implementation)
        assert (comm.world_size, comm.topk) == (world_size, topk)


def test_sm90_reducer_rejects_noncanonical_mxfp8_scales():
    import cutlass
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.topk_reduce import (
        CombineFormat,
        TopkReduce,
    )

    unsupported = CombineFormat(cutlass.Float8E4M3FN, cutlass.BFloat16, 16)
    with pytest.raises(ValueError, match="does not support"):
        TopkReduce(256, 2, unsupported, sm_arch="sm_90")
    supported = CombineFormat.parse("32e4m3xe8m0")
    assert TopkReduce(256, 2, supported, sm_arch="sm_90")._use_sm90_static


@pytest.mark.parametrize("vector_bytes", [8, 16, None])
def test_internode_reducer_vector_width(vector_bytes):
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.topk_reduce import (
        CombineFormat,
        TopkReduce,
    )

    options = dict(num_remote_nodes=1, vector_bytes=vector_bytes)
    if vector_bytes == 8:
        with pytest.raises(ValueError, match="requires vector_bytes=16"):
            TopkReduce(256, 2, CombineFormat.parse("bf16"), **options)
    else:
        reducer = TopkReduce(256, 2, CombineFormat.parse("bf16"), **options)
        assert reducer.vector_bytes == 16


@pytest.mark.parametrize("in_kernel", [False, True])
def test_mixed_persistent_finalize_matches_active_reducer(in_kernel, monkeypatch):
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.pre_unification.bf16_mxfp8 import (
        MegaMoEBf16Mxfp8Config,
    )
    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_mxfp8_bf16_glu.megamoe_kernel_mxfp8_bf16 import (
        Sm100MegaMoEMxfp8Bf16Kernel,
    )

    from flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_mxfp8_bf16_glu import (
        megamoe_kernel_mxfp8_bf16 as provider,
    )

    monkeypatch.setattr(
        provider.cutlass_utils,
        "HardwareInfo",
        lambda: SimpleNamespace(get_device_multiprocessor_count=lambda: 132),
    )
    config = MegaMoEBf16Mxfp8Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=16,
        num_topk=2,
        num_total_experts=4,
        hidden=256,
        intermediate=128,
        enable_in_kernel_fc2_reduce=in_kernel,
        in_kernel_fc2_reduce=in_kernel,
        use_persistent_finalize_kernel=True,
    )
    kernel = Sm100MegaMoEMxfp8Bf16Kernel(
        world_size=1,
        local_rank=0,
        num_topk=2,
        max_tokens_per_rank=16,
        hidden=256,
        static_expert_shape=(4, 256, 256),
        fc2_in_kernel_topk_reduce=in_kernel,
        topk_reduce_persistent=config.persistent_finalize_enabled,
    )
    assert kernel.topk_reduce_persistent is not in_kernel
    names = [region.name for region in kernel._build_local_region_specs()]
    assert names.count("topk_reduce_counters") == int(not in_kernel)
