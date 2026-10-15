"""Online CSF compression during weight preparation preserves native expert output."""

from dataclasses import replace

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_nvfp4_phase_kernels import _build_domain
from ..conftest import require_b12x
from ..quantization.test_mxfp4_csf import fixture


def _replay_matches_native(native, online, buffers, *, e, h, tokens, topk, device):
    """Graph-replay both arms with poisoned shared scratch; outputs must be bitwise equal."""
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in (native, online)
    ]
    source = torch.randn(tokens, h, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    probabilities = torch.softmax(torch.randn(tokens, topk, device=device), -1)

    def operands(specs, output):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device) for s in specs
        )
        return scratch, dict(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )

    def bind(plan, output):
        scratch, arguments = operands(plan.scratch_specs(), output)
        return moe.bind(plan, **arguments), scratch

    def prepare(state):
        output = torch.empty_like(source)
        scratch, arguments = operands(state.scratch.scratch_specs(), output)
        binding = state.bind(**arguments)
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"online-csf-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        outputs = [torch.empty_like(source) for _ in plans]
        bound = [bind(p, o) for p, o in zip(plans, outputs, strict=True)]
        session.freeze()
        graphs = []
        for binding, _ in bound:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.run(binding=binding)
            graphs.append(graph)
        for _ in range(3):
            source.neg_()
            ids.add_(1).remainder_(e)
            for buffer in buffers:
                buffer.view(torch.uint8).fill_(0x7F)
            for output in outputs:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(outputs[0])
            torch.testing.assert_close(outputs[1], outputs[0], rtol=0, atol=0)
        for graph in graphs:
            graph.reset()


def _assert_scales_released(online, native_scales, buffers):
    impl = online._impl
    assert online.plan.scale_compression == "csf"
    assert impl.w1_blockscale.data_ptr() == buffers[0].data_ptr()
    assert impl.w2_blockscale.data_ptr() == buffers[1].data_ptr()
    owned = {t.untyped_storage().data_ptr() for t in native_scales}
    retained = [
        t
        for t in vars(impl).values()
        if isinstance(t, torch.Tensor) and t.untyped_storage().data_ptr() in owned
    ]
    assert not retained


@pytest.mark.parametrize("tokens", [1, 17, 128])
@pytest.mark.parametrize("h,n", [(512, 192), (5120, 576)])
@pytest.mark.parametrize("nonlinearity", ["silu", "situ"])
def test_mxfp4_online_compression_matches_native_experts(tokens, h, n, nonlinearity):
    device = require_b12x()
    e, topk = 8, 2
    _, s13 = fixture(2 * n, h // 32, finite_scales=True)
    _, s2 = fixture(h, n // 32, finite_scales=True)
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device=device)
    one = torch.ones(e, device=device)

    def plan(compression):
        return moe.plan_weights(
            source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=moe.ActivationSpec(
                mode="a8", nonlinearity=nonlinearity, io_dtype=torch.bfloat16
            ),
            geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
            constraints=moe.WeightPlanConstraints(scale_compression=compression),
        )

    def packed(scales):
        return moe.PackedWeights(
            w13=w13.clone(),
            w2=w2.clone(),
            w13_block_scales=scales[0],
            w2_block_scales=scales[1],
            w13_global_scales=one,
            w2_global_scales=one,
        )

    # Native preparation rewrites its source scales in place; copy them first.
    sources = (s13.clone(), s2.clone())
    online_weights = packed(sources)
    native = moe.prepare_weights(plan=plan(None), weights=packed((s13, s2)))
    buffers = tuple(
        torch.empty_like(scale).view(torch.uint8)
        for scale in (native._impl.w1_blockscale, native._impl.w2_blockscale)
    )
    online = moe.prepare_weights(
        plan=plan("csf"), weights=online_weights, scale_scratch=buffers
    )
    _assert_scales_released(online, sources, buffers)
    _replay_matches_native(
        native, online, buffers, e=e, h=h, tokens=tokens, topk=topk, device=device
    )


@pytest.mark.parametrize("tokens", [1, 17, 33])
@pytest.mark.parametrize("activation_mode", ["a4", "a16"])
@pytest.mark.parametrize("n", [64, 192, 320])
def test_nvfp4_online_compression_matches_native_experts(tokens, activation_mode, n):
    device = require_b12x()
    e, h, topk = 8, 256, 2
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=topk, seed=931)
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)

    def plan(compression):
        return moe.plan_weights(
            source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
            activation=moe.ActivationSpec(
                mode=activation_mode,
                nonlinearity="silu",
                io_dtype=torch.bfloat16,
                swiglu_limit=10.0,
            ),
            geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
            constraints=moe.WeightPlanConstraints(scale_compression=compression),
        )

    packed = moe.PackedWeights(
        w13=domain["w13_packed"],
        w2=domain["w2_packed"],
        w13_block_scales=s13,
        w2_block_scales=s2,
        w13_global_scales=one,
        w2_global_scales=one,
        input_scale=one,
        intermediate_scale=one,
        immutable_input_scales=True,
    )
    # Native preparation may normalize FC1 weights and scales in place; copy them first.
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    sources = (s13.clone(), s2.clone())
    online_weights = replace(
        packed,
        w13=packed.w13.clone(),
        w2=packed.w2.clone(),
        w13_block_scales=sources[0],
        w2_block_scales=sources[1],
    )
    native = moe.prepare_weights(plan=plan(None), weights=packed)
    online = moe.prepare_weights(
        plan=plan("csf"), weights=online_weights, scale_scratch=buffers
    )
    _assert_scales_released(online, sources, buffers)
    _replay_matches_native(
        native, online, buffers, e=e, h=h, tokens=tokens, topk=topk, device=device
    )


def test_online_compression_requires_supported_plans_and_scratch():
    device = require_b12x()
    with pytest.raises(ValueError, match="None or 'csf'"):
        moe.WeightPlanConstraints(scale_compression="ans")
    with pytest.raises(ValueError, match="gated native MXFP4"):
        moe.plan_weights(
            source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
            activation=moe.ActivationSpec(
                mode="a4", nonlinearity="silu", io_dtype=torch.bfloat16
            ),
            geometry=moe.MoEGeometry(
                num_experts=8, hidden_size=512, intermediate_size=192
            ),
            constraints=moe.WeightPlanConstraints(scale_compression="csf"),
        )
    plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=8, hidden_size=512, intermediate_size=192),
    )
    scratch = tuple(torch.empty(64, dtype=torch.uint8, device=device) for _ in range(2))
    weights = moe.PackedWeights(
        w13=torch.zeros(8, 384, 256, dtype=torch.uint8, device=device),
        w2=torch.zeros(8, 512, 96, dtype=torch.uint8, device=device),
        w13_block_scales=torch.zeros(8, 384, 16, dtype=torch.uint8, device=device),
        w2_block_scales=torch.zeros(8, 512, 6, dtype=torch.uint8, device=device),
        w13_global_scales=torch.ones(8, device=device),
        w2_global_scales=torch.ones(8, device=device),
    )
    with pytest.raises(ValueError, match="requires a scale_compression plan"):
        moe.prepare_weights(plan=plan, weights=weights, scale_scratch=scratch)
    compressing = replace(plan, scale_compression="csf")
    with pytest.raises(ValueError, match="two scale_scratch buffers"):
        moe.prepare_weights(plan=compressing, weights=weights)
    with pytest.raises(ValueError, match="separate from the native scales"):
        moe.prepare_weights(
            plan=compressing,
            weights=weights,
            scale_scratch=(weights.w13_block_scales, scratch[1]),
        )
