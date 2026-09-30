from __future__ import annotations

from dataclasses import replace
import gc
import weakref

import pytest
import torch

from b12x.moe import fused_moe
from b12x._lib.intrinsics import swizzle_block_scale



def weight_plan(**kwargs):
    return fused_moe.plan_weights(
        source=kwargs.pop("source", fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MODELOPT_NVFP4,
            w13_layout=fused_moe.W13Layout.W13,
        )),
        geometry=fused_moe.MoEGeometry(num_experts=8, hidden_size=128, intermediate_size=128),
        activation=kwargs.pop("activation", fused_moe.ActivationSpec(
            mode=fused_moe.ActivationMode.AUTO, nonlinearity="silu", io_dtype=torch.bfloat16,
        )), **kwargs,
    )


@pytest.mark.parametrize("field,value", [
    ("io_dtype", torch.float16), ("nonlinearity", "relu2"),
])
def test_auto_rejects_unsupported_activation(field, value):
    activation = replace(weight_plan().activation, **{field: value})
    with pytest.raises(ValueError, match="automatic MoE precision"):
        weight_plan(activation=activation)


def test_auto_requires_nvfp4_native_storage():
    plan = weight_plan()
    assert plan._impl.quant_modes == {"nvfp4", "w4a16"}
    assert plan.prepared_format.packing is fused_moe.WeightPacking.SOURCE_NATIVE
    with pytest.raises(ValueError, match="up/gate"):
        weight_plan(source=replace(plan.source, w13_layout=fused_moe.W13Layout.W31))
    with pytest.raises(ValueError, match="source-native"):
        weight_plan(constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.MMA_PACKED,
        ))
    with pytest.raises(ValueError, match="ModelOpt NVFP4"):
        weight_plan(source=fused_moe.PackedSource(
            format=fused_moe.PackedSourceFormat.MXFP4_E8M0_K32,
        ))


def test_uniform_nvfp4_a16_requires_only_mma_packing():
    activation = replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16)
    plan = weight_plan(activation=activation)
    assert plan.prepared_format.available_packings == {fused_moe.WeightPacking.MMA_PACKED}
    with pytest.raises(ValueError, match="uniform NVFP4 W4A16 requires mma_packed"):
        weight_plan(activation=activation, constraints=fused_moe.WeightPlanConstraints(
            required_packing=fused_moe.WeightPacking.SOURCE_NATIVE,
        ))


@pytest.mark.parametrize("mode,larger_recipe", (("a4", "nvfp4"), ("a8", "w4a8_nvfp4"), ("auto", "nvfp4_auto")))
def test_a16_cutoff_uses_existing_precision_candidates(mode, larger_recipe):
    from types import SimpleNamespace
    from b12x.moe.fused_moe._impl import B12XFP4ExpertWeights, _PreparedWeightRepresentation
    from b12x.preparation import DeviceIdentity

    base = weight_plan()
    plan = weight_plan(
        activation=replace(base.activation, mode=mode, a16_max_tokens=32),
        source=replace(base.source, w13_layout="w13" if mode == "auto" else "w31"),
    )
    assert plan._impl.w13_layout == "w13"
    assert plan._impl.quant_modes == {"w4a16", "nvfp4" if mode == "auto" else larger_recipe}
    w1 = torch.empty((8, 256, 64), dtype=torch.uint8)
    w2 = torch.empty((8, 128, 64), dtype=torch.uint8)
    experts = fused_moe.PreparedExperts(plan=plan, _impl=B12XFP4ExpertWeights(
        plan=plan._impl,
        w1_fp4=w1, w2_fp4=w2,
        w1_blockscale=torch.empty((8, 256, 8), dtype=torch.uint8),
        w2_blockscale=torch.empty((8, 128, 8), dtype=torch.uint8),
        w1_alphas=torch.ones(8), w2_alphas=torch.ones(8),
        a1_gscale=torch.ones(8), a2_gscale=torch.ones(8),
        representation=_PreparedWeightRepresentation(
            quant_mode="w4a16", layout="source_native",
            value=SimpleNamespace(w13=w1, w2=w2),
        ),
    ))
    declaration = fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(
            max_tokens=128, top_k=2, warmup_token_counts=(4, 33),
        ),
    )
    assert declaration.token_counts == (4, 32, 33, 128)
    device = DeviceIdentity("nvidia", (12, 0), 188, "SM120")
    for rows, child in declaration.variants.items():
        assert child.query.quant_mode == ("w4a16" if rows <= 32 else larger_recipe)
        configs = [config for _, config in child.contract.eligible_plan(child.query, device).candidates]
        assert configs
        if rows <= 32:
            assert all(config.backend == "w4a16" for config in configs)
        elif mode != "auto":
            assert all(config.backend != "w4a16" for config in configs)


@pytest.mark.parametrize("name", ["w13_blockscale", "w2_blockscale"])
@pytest.mark.parametrize("invalid", ["truncated", "strided", "dtype"])
def test_native_nvfp4_preparation_validates_scale_storage(name, invalid):
    from b12x.moe._shared.kernels.w4a16.prepare import prepare_w4a16_modelopt_native_weights
    inputs = dict(
        w13_fp4=torch.empty((1, 256, 64), dtype=torch.uint8),
        w2_fp4=torch.empty((1, 128, 64), dtype=torch.uint8),
        w13_global_scale=torch.ones(1), w2_global_scale=torch.ones(1),
        w13_blockscale=torch.empty((1, 256, 8), dtype=torch.uint8),
        w2_blockscale=torch.empty((1, 128, 8), dtype=torch.uint8),
        activation="silu",
    )
    scales = inputs[name]
    if invalid == "truncated":
        inputs[name] = scales[:, :-1].contiguous()
    elif invalid == "strided":
        inputs[name] = scales.transpose(1, 2)
    else:
        inputs[name] = scales.float()
    with pytest.raises((ValueError, TypeError), match=name):
        prepare_w4a16_modelopt_native_weights(**inputs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_uniform_nvfp4_a16_does_not_retain_source_scales():
    plan = weight_plan(activation=replace(weight_plan().activation, mode=fused_moe.ActivationMode.A16))
    weights = fused_moe.PackedWeights(
        w13=torch.zeros((8, 256, 64), dtype=torch.uint8, device="cuda"),
        w2=torch.zeros((8, 128, 64), dtype=torch.uint8, device="cuda"),
        w13_block_scales=swizzle_block_scale(torch.ones((8, 256, 8), device="cuda").to(torch.float8_e4m3fn)),
        w2_block_scales=swizzle_block_scale(torch.ones((8, 128, 8), device="cuda").to(torch.float8_e4m3fn)),
        w13_global_scales=torch.ones(8, device="cuda"),
        w2_global_scales=torch.ones(8, device="cuda"),
    )
    source_scales = (weakref.ref(weights.w13_block_scales), weakref.ref(weights.w2_block_scales))
    experts = fused_moe.prepare_weights(plan=plan, weights=weights)
    prepared = experts._impl.representation_for("w4a16")
    assert prepared.weight_layout == "packed"
    assert prepared.w13.data_ptr() == weights.w13.data_ptr()
    assert prepared.w2.data_ptr() == weights.w2.data_ptr()
    assert prepared.w13_scale is experts._impl.w1_blockscale
    assert prepared.w2_scale is experts._impl.w2_blockscale
    del weights
    gc.collect()
    assert all(ref() is None for ref in source_scales)


@pytest.mark.parametrize("num_experts,hidden_size,intermediate_size,topk", (
    (8, 256, 256, 2), (256, 6144, 256, 8),
))
@pytest.mark.parametrize("counts", ((1, 2, 4, 7, 8), (1, 2, 3, 4, 5, 6, 8)))
def test_auto_native_decode_retains_launches_and_replays_shared_storage(
    tmp_path, monkeypatch, num_experts, hidden_size, intermediate_size, topk, counts,
):
    from b12x.preparation import PreparationSession, PreparedCall
    from b12x.moe._shared.kernels.w4a16 import kernel
    from b12x.testing.reference.helpers import require_b12x
    from b12x.testing.reference.w4a16_reference import compare_to_reference, moe_reference_w4a16
    from .test_fused_moe import make_modelopt_weights

    require_b12x()
    torch.manual_seed(4196)
    raw = make_modelopt_weights(
        experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
    )
    w13, s13, g13, w2, s2, g2 = raw
    scales = torch.ones(num_experts, device="cuda")
    plan = fused_moe.plan_weights(
        source=fused_moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=fused_moe.ActivationSpec(mode="auto", nonlinearity="silu", io_dtype=torch.bfloat16),
        geometry=fused_moe.MoEGeometry(
            num_experts=num_experts, hidden_size=hidden_size, intermediate_size=intermediate_size,
        ),
    )
    experts = fused_moe.prepare_weights(plan=plan, weights=fused_moe.PackedWeights(
        w13=w13, w2=w2, w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=g13, w2_global_scales=g2,
        input_scale=scales, intermediate_scale=scales,
    ))
    native = experts._impl.representation_for("w4a16")
    for tensor, original in (
        (native.w13, w13), (native.w2, w2), (native.w13_scale, s13), (native.w2_scale, s2),
        (native.micro_w13_scale, s13), (native.micro_w2_scale, s2),
    ):
        assert tensor.data_ptr() == original.data_ptr()
    originals = [tensor.clone() for tensor in raw]
    x = torch.randn(8, hidden_size, dtype=torch.bfloat16, device="cuda") * 0.25
    ids = torch.stack([torch.randperm(num_experts, device="cuda")[:topk] for _ in range(8)])
    probabilities = torch.softmax(torch.randn(8, topk, device="cuda"), dim=-1)
    execution = fused_moe.plan_execution(
        experts=experts, capacity=fused_moe.ExecutionCapacity(
            max_tokens=8, top_k=topk, warmup_token_counts=counts,
        ),
    )

    def make_call(m):
        def call(state):
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                            for spec in state.scratch.scratch_specs())
            binding = state.bind(scratch=scratch, a=x[:m], topk_ids=ids[:m],
                                 topk_weights=probabilities[:m], output=torch.empty_like(x[:m]))
            assert state.config.backend == "w4a16" and state.config.w4a16_route_mode == "direct"
            assert binding.route_pack_launches is None
            return PreparedCall(run=lambda: state.run(binding), owners=scratch)
        return call

    with PreparationSession(device=x.device, autotune=False, compile_workers=2, cache_dir=tmp_path) as session:
        session.prepare((execution.request(
            name="native-auto", prepare_calls={m: make_call(m) for m in counts},
        ),))
        bindings = []
        unprepared_rows = next(rows for rows in range(1, 8) if rows not in counts)
        for m in (*counts, unprepared_rows):
            state = execution.variants[m if m in counts else 8].prepared.state
            scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
                            for spec in state.scratch.scratch_specs())
            for dtype in (torch.int32, torch.int64):
                binding = fused_moe.bind(
                    execution, scratch=scratch, a=x[:m], topk_ids=ids[:m].to(dtype),
                    topk_weights=probabilities[:m], output=torch.empty_like(x[:m]),
                )
                assert (binding.route_pack_launches is None) is (m in counts)
                bindings.append((m, binding, scratch))

        launched = []
        launch_flat = kernel._w4a16_small_m_direct_launch_flat

        def retained_launch(*args, launcher=None, **kwargs):
            assert launcher is not None
            launched.append((launcher.m, launcher.topk_ids_dtype))
            return launch_flat(*args, launcher=launcher, **kwargs)

        def forbidden_resolution(*args, **kwargs):
            raise AssertionError("prepared replay resolved a kernel through a module cache")

        kernel.clear_w4a16_kernel_cache()
        monkeypatch.setattr(kernel, "_compile_w4a16_small_m_direct", forbidden_resolution)
        monkeypatch.setattr(kernel, "compile_w4a16_fused_moe", forbidden_resolution)
        monkeypatch.setattr(kernel, "_w4a16_small_m_direct_launch_flat", retained_launch)
        session.freeze()
        for m, binding, scratch in bindings:
            expected = moe_reference_w4a16(
                x[:m], *raw, binding.topk_ids, probabilities[:m],
                num_experts, hidden_size, intermediate_size,
            )
            actual = fused_moe.run(binding=binding)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual)
            assert compare_to_reference(actual, expected).cos > 0.9999
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fused_moe.run(binding=binding)
            pointers = tuple(tensor.data_ptr() for tensor in (*scratch, binding.a, actual))
            for replay in range(3):
                if replay == 1:
                    binding.a.neg_()
                    binding.topk_ids.add_(1).remainder_(num_experts)
                    expected = moe_reference_w4a16(
                        binding.a, *raw, binding.topk_ids, probabilities[:m],
                        num_experts, hidden_size, intermediate_size,
                    )
                actual.fill_(float("nan"))
                allocated = torch.cuda.memory_stats()["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocated
                assert tuple(tensor.data_ptr() for tensor in (*scratch, binding.a, actual)) == pointers
                assert torch.isfinite(actual).all()
                assert compare_to_reference(actual, expected).cos > 0.9999
            graph.reset()
        assert set(launched) == {(m, dtype) for m in counts for dtype in (torch.int32, torch.int64)}
        for tensor, original in zip(raw, originals):
            torch.testing.assert_close(tensor.view(torch.uint8), original.view(torch.uint8), rtol=0, atol=0)
