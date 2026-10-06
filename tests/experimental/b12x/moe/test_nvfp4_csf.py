"""Compressed scales preserve native NVFP4 expert output under graph replay."""

from dataclasses import replace

import numpy as np
import pytest
import torch

from b12x._lib.quant.nvfp4_csf import make_nvfp4_csf_batch
from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from .test_nvfp4_phase_kernels import _build_domain
from ..conftest import require_b12x


def compress_fixture(swizzled, rows, columns, raw=False):
    """Build valid byte-window records independently from GPU expansion."""
    e = swizzled.shape[0]
    logical = swizzled.view(torch.uint8).reshape(e, rows // 128, columns // 4, 32, 4, 4)
    logical = logical.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, rows, columns)
    fixed, exceptions = [], []
    for source in logical.cpu().numpy():
        base = np.minimum(source.min(axis=1), 240).astype(np.uint8)
        offsets = source.astype(np.int16) - base[:, None]
        outside = offsets > 15
        offsets[outside] = 0
        offsets = offsets.astype(np.uint8)
        packed = offsets[:, ::2] | (offsets[:, 1::2] << 4)
        fixed.append(
            np.concatenate((base.reshape(-1, 16), packed.reshape(rows // 16, -1)), 1)
        )
        pos = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(pos | (source.ravel()[pos].astype(np.uint32) << 24))
    if raw:
        return moe.CsfScalePlanes(
            tuple(torch.from_numpy(p) for p in fixed),
            tuple(torch.from_numpy(p) for p in exceptions),
        )
    return make_nvfp4_csf_batch(
        fixed, exceptions, rows=rows, columns=columns, device=swizzled.device
    )


@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("tokens", [1, 17, 33])
@pytest.mark.parametrize("activation_mode", ["a4", "a16"])
@pytest.mark.parametrize("n", [64, 128, 192, 320])
def test_native_expert_output_and_shared_scratch_poisoned_replay(
    tokens, activation_mode, raw, n, inline_scales=None, autotune=False,
    deterministic=True, tile_m=16, after_plan=None,
):
    device = require_b12x()
    e, h, topk = 8, 256, 2
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=topk, seed=931)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode=activation_mode,
            nonlinearity="silu",
            io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
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
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(packed, w13=packed.w13.clone(), w2=packed.w2.clone(),
                       w13_block_scales=buffers[0], w2_block_scales=buffers[1]),
        w13_scales=compress_fixture(s13, 2 * n, h // 16, raw=raw),
        w2_scales=compress_fixture(s2, h, n // 16, raw=raw),
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    # W4A16 reads compressed scales per pipeline stage up to a token limit;
    # larger calls expand the routed experts from the same storage first.
    from b12x.moe.fused_moe._impl import W4A16_CSF_STAGE_MAX_TOKENS

    stage_scales = activation_mode == "a16" and tokens <= W4A16_CSF_STAGE_MAX_TOKENS
    if activation_mode == "a16":
        assert experts[1].plan._impl.w4a16_scale_format == "e4m3_k16_csf"
        assert experts[1]._impl.w4a16_expanded is not None
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    override = None if inline_scales is None else MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=tile_m, dynamic_route_mode="grouped",
        nvfp4_inline_scales=inline_scales,
    )
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=deterministic),
            override=override if i == 1 else (
                replace(override, nvfp4_inline_scales=False) if override else None
            ),
        )
        for i, owner in enumerate(experts)
    ]
    if after_plan is not None:
        after_plan()
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=scratch,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(
        device=device, autotune=autotune, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"nvfp4-scale-storage-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        owners, outputs, bindings, graphs = [], [], [], []
        for plan in plans:
            scratch = tuple(
                torch.empty(s.shape, dtype=s.dtype, device=device)
                for s in plan.scratch_specs()
            )
            output = torch.empty_like(source)
            binding = moe.bind(
                plan,
                a=source,
                topk_ids=ids,
                topk_weights=probabilities,
                output=output,
                scratch=scratch,
                input_scales_static=True,
            )
            owners.append(scratch)
            outputs.append(output)
            bindings.append(binding)
        session.freeze()
        for binding in bindings:
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
            for binding in bindings:
                if binding.barrier_count is not None:
                    binding.barrier_count.fill_(123)
                    binding.barrier_epoch.fill_(-42)
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            assert torch.isfinite(outputs[0]).all() and torch.count_nonzero(outputs[0])
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            if inline_scales or stage_scales:
                assert all(torch.all(buf.view(torch.uint8) == 0x7F) for buf in buffers)
            elif activation_mode == "a16":
                # Larger calls rebuild the routed experts' scales in the scratch.
                assert not torch.all(buffers[0].view(torch.uint8) == 0x7F)
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("inline_scales", [False, True])
@pytest.mark.parametrize("tokens,n", [(1, 64), (17, 320)])
def test_scale_decoder_override_preserves_expert_graph_output(inline_scales, tokens, n):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens, "a4", True, n, inline_scales=inline_scales
    )


@pytest.mark.parametrize("tile_m", [16, 32, 64, 128])
def test_split_gate_scale_operands_preserve_native_output_for_consumer_counts(tile_m):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        33, "a4", True, 320, inline_scales=True, tile_m=tile_m
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_indexed_scale_programs_are_declared_for_preparation(deterministic):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        1, "a4", True, 128, inline_scales=False, autotune=True,
        deterministic=deterministic,
    )


@pytest.mark.parametrize("n", [128, 320])
def test_large_a16_calls_expand_compressed_scales_per_call(n, monkeypatch):
    from b12x.moe.fused_moe import _impl

    monkeypatch.setattr(_impl, "W4A16_CSF_STAGE_MAX_TOKENS", 64)
    test_native_expert_output_and_shared_scratch_poisoned_replay(300, "a16", False, n)


def test_a16_stage_scales_reuse_the_planned_launch_for_live_counts():
    """One capacity plan serves smaller live counts without resolving new kernels."""
    device = require_b12x()
    e, h, n, topk, capacity = 8, 256, 128, 2, 64
    domain = _build_domain(E=e, K=h, n=n, m=capacity, top_k=topk, seed=977)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode="a16", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
    packed = moe.PackedWeights(
        w13=domain["w13_packed"], w2=domain["w2_packed"],
        w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=one, w2_global_scales=one,
        input_scale=one, intermediate_scale=one, immutable_input_scales=True,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    # Native preparation repacks its package in place: compress the source first.
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(packed, w13=packed.w13.clone(), w2=packed.w2.clone(),
                       w13_block_scales=buffers[0], w2_block_scales=buffers[1]),
        w13_scales=compress_fixture(s13, 2 * n, h // 16),
        w2_scales=compress_fixture(s2, h, n // 16),
    )
    experts = (
        moe.prepare_weights(plan=weight_plan, weights=packed),
        moe.prepare_weights(plan=weight_plan, weights=compressed),
    )
    plans = tuple(
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in experts
    )
    x, ids, weights = domain["x"], domain["topk_ids"], domain["topk_weights"]

    def bind(plan, rows):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device) for s in plan.scratch_specs()
        )
        output = torch.empty_like(x[:rows])
        binding = moe.bind(plan, a=x[:rows], topk_ids=ids[:rows], topk_weights=weights[:rows],
                           output=output, scratch=scratch, input_scales_static=True)
        return binding, output, scratch

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(x)
        binding = state.bind(a=x, topk_ids=ids, topk_weights=weights, output=output,
                             scratch=scratch, input_scales_static=True)
        return PreparedCall(run=lambda: state.run(binding), output=output, owners=(scratch, binding))

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(tuple(
            p.request(name=f"csf-live-{i}", prepare_call=prepare) for i, p in enumerate(plans)
        ))
        session.freeze()
        for rows in (1, 7, 33, capacity):
            native, compressed = (bind(plan, rows) for plan in plans)
            for buffer in buffers:
                buffer.view(torch.uint8).fill_(0x7F)
            moe.run(binding=native[0])
            moe.run(binding=compressed[0])
            torch.cuda.synchronize()
            assert torch.isfinite(native[1]).all() and torch.count_nonzero(native[1])
            torch.testing.assert_close(compressed[1], native[1], rtol=0, atol=0)
            # Stage reads leave the shared expansion scratch untouched.
            assert all(torch.all(buf.view(torch.uint8) == 0x7F) for buf in buffers)


@pytest.mark.parametrize("activation_mode,backend", [("a16", None), ("a4", None), ("a4", "dynamic")])
def test_prefetched_scales_replace_the_per_call_expansion(activation_mode, backend):
    """expand_scales() fills the shared scratch for a later scales_expanded call."""
    from b12x.moe.fused_moe._impl import W4A16_CSF_STAGE_MAX_TOKENS

    device = require_b12x()
    e, h, n, topk = 8, 256, 128, 2
    # A16 expands only above the stage-read limit; A4 expands every call.
    tokens = W4A16_CSF_STAGE_MAX_TOKENS + 64 if activation_mode == "a16" else 33
    domain = _build_domain(E=e, K=h, n=n, m=tokens, top_k=topk, seed=937)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="modelopt_nvfp4", w13_layout="w13"),
        activation=moe.ActivationSpec(
            mode=activation_mode, nonlinearity="silu", io_dtype=torch.bfloat16,
            swiglu_limit=10.0,
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    s13 = domain["w13_sfb"].view(torch.float8_e4m3fn).view(e, 2 * n, h // 16)
    s2 = domain["w2_sfb"].view(torch.float8_e4m3fn).view(e, h, n // 16)
    one = torch.ones(e, device=device)
    packed = moe.PackedWeights(
        w13=domain["w13_packed"], w2=domain["w2_packed"],
        w13_block_scales=s13, w2_block_scales=s2,
        w13_global_scales=one, w2_global_scales=one,
        input_scale=one, intermediate_scale=one, immutable_input_scales=True,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    compressed = moe.Nvfp4CsfWeights(
        packed=replace(packed, w13=packed.w13.clone(), w2=packed.w2.clone(),
                       w13_block_scales=buffers[0], w2_block_scales=buffers[1]),
        w13_scales=compress_fixture(s13, 2 * n, h // 16),
        w2_scales=compress_fixture(s2, h, n // 16),
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    assert not moe.expand_scales(experts[0])
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    override = None if backend is None else MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=16, dynamic_route_mode="grouped", nvfp4_inline_scales=False,
    )
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False}, override=override,
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in experts
    ]
    source, ids, probabilities = domain["x"], domain["topk_ids"], domain["topk_weights"]
    scratches = [
        tuple(torch.empty(s.shape, dtype=s.dtype, device=device) for s in p.scratch_specs())
        for p in plans
    ]

    def call(index, **kwargs):
        output = torch.full_like(source, float("nan"))
        binding = moe.bind(
            plans[index], a=source, topk_ids=ids, topk_weights=probabilities,
            output=output, scratch=scratches[index], input_scales_static=True, **kwargs,
        )
        moe.run(binding=binding)
        torch.cuda.synchronize()
        return output

    def prepare(state):
        scratch = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source, topk_ids=ids, topk_weights=probabilities, output=output,
            scratch=scratch, input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(scratch, binding)
        )

    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        session.prepare(
            tuple(
                p.request(name=f"nvfp4-scale-prefetch-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        reference = call(0)
        assert torch.isfinite(reference).all() and torch.count_nonzero(reference)
        poison = lambda: [b.view(torch.uint8).fill_(0x7F) for b in buffers]  # noqa: E731
        poison()
        stale = call(1, scales_expanded=True)
        if activation_mode == "a4" and backend is None:
            # Small FP4-activation calls run the micro kernel, whose barrier
            # reset is fused into the expansion: they expand regardless.
            torch.testing.assert_close(stale, reference, rtol=0, atol=0)
        else:
            assert not torch.equal(stale, reference)
        # ... and a prefetch on a side stream restores the native result.
        poison()
        side = torch.cuda.Stream(device)
        side.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(side):
            assert moe.expand_scales(experts[1])
        torch.cuda.current_stream(device).wait_stream(side)
        torch.testing.assert_close(call(1, scales_expanded=True), reference, rtol=0, atol=0)
        # Without the flag the call still expands its own routed experts.
        poison()
        torch.testing.assert_close(call(1), reference, rtol=0, atol=0)

        if torch.cuda.device_count() > 1:
            poison()
            torch.cuda.synchronize(device)
            with torch.cuda.device(1):
                assert moe.expand_scales(experts[1])
                assert torch.cuda.current_device() == 1
            torch.cuda.synchronize(device)
            torch.testing.assert_close(call(1, scales_expanded=True), reference, rtol=0, atol=0)

        output = torch.empty_like(source)
        binding = moe.bind(
            plans[1], a=source, topk_ids=ids, topk_weights=probabilities,
            output=output, scratch=scratches[1], input_scales_static=True,
            scales_expanded=True,
        )
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            moe.expand_scales(experts[1])
            moe.run(binding=binding)
        for _ in range(3):
            poison()
            output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            torch.testing.assert_close(output, reference, rtol=0, atol=0)
        graph.reset()


@pytest.mark.parametrize("declared,changed,tokens", [(64, 1536, 300), (1536, 0, 33)])
def test_stage_scale_capacity_control_is_retained(declared, changed, tokens, monkeypatch):
    from b12x.moe.fused_moe import _impl

    monkeypatch.setattr(_impl, "W4A16_CSF_STAGE_MAX_TOKENS", declared)
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens, "a16", False, 128,
        after_plan=lambda: monkeypatch.setattr(_impl, "W4A16_CSF_STAGE_MAX_TOKENS", changed),
    )
