"""MXFP4-CSF preserves native W4A8 expert math and serialized scale scratch."""

import pytest
import torch

from b12x.moe import fused_moe as moe
from b12x.preparation import PreparationSession, PreparedCall
from ..conftest import require_b12x
from ..quantization.test_mxfp4_csf import fixture


@pytest.mark.parametrize("tokens", [1, 17, 33, 128])
@pytest.mark.parametrize("h,n", [(512, 192), (5120, 576), (512, 256)])
@pytest.mark.parametrize("raw", [False, True])
def test_native_expert_output_and_shared_scratch_poisoned_replay(tokens, h, n, raw):
    device = require_b12x()
    e, topk = 8, 2
    first, s13 = fixture(2 * n, h // 32, finite_scales=True)
    second, s2 = fixture(h, n // 32, finite_scales=True)
    if raw:

        def cpu_planes(batch):
            offsets = batch.exception_offsets.cpu().tolist()
            fixed = tuple(t.cpu() for t in batch.fixed)
            exceptions = batch.exceptions.cpu()
            return moe.CsfScalePlanes(
                fixed,
                tuple(
                    exceptions[start:end]
                    for start, end in zip(offsets[:-1], offsets[1:], strict=True)
                ),
            )

        first, second = cpu_planes(first), cpu_planes(second)
    w13 = torch.randint(0, 256, (e, 2 * n, h // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (e, h, n // 2), dtype=torch.uint8, device=device)
    one = torch.ones(e, device=device)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout="w31"),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(num_experts=e, hidden_size=h, intermediate_size=n),
    )
    packed = moe.PackedWeights(
        w13=w13.clone(),
        w2=w2.clone(),
        w13_block_scales=s13,
        w2_block_scales=s2,
        w13_global_scales=one,
        w2_global_scales=one,
    )
    buffers = (torch.empty_like(s13), torch.empty_like(s2))
    compressed = moe.Mxfp4CsfWeights(
        w13=w13,
        w2=w2,
        w13_scales=first,
        w2_scales=second,
        w13_scale_scratch=buffers[0],
        w2_scale_scratch=buffers[1],
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    impl = experts[1]._impl
    assert impl.w1_blockscale.data_ptr() == buffers[0].data_ptr()
    assert impl.w2_blockscale.data_ptr() == buffers[1].data_ptr()
    assert impl.representation.value.w13_sfb.data_ptr() == buffers[0].data_ptr()
    assert impl.representation.value.w2_sfb.data_ptr() == buffers[1].data_ptr()
    for attr in ("w1_fp4", "w2_fp4", "w1_blockscale", "w2_blockscale"):
        assert torch.equal(
            getattr(experts[0]._impl, attr), getattr(experts[1]._impl, attr)
        )
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in experts
    ]
    source = torch.randn(tokens, h, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    probabilities = torch.softmax(torch.randn(tokens, topk, device=device), -1)

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
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"mxfp4-scale-storage-{i}", prepare_call=prepare)
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
        for graph in graphs:
            graph.reset()


def _csf_planes(rows, columns, *, experts, seed):
    """CSF planes with sparse exceptions in most native tiles and dense rows
    whose tiles exceed the inline records (raw tiles)."""
    import numpy as np

    from b12x._lib.quant.x4t_scales import make_x4t_scale_batch

    device = require_b12x()
    rng = np.random.default_rng(seed + rows + columns)
    fixed, exceptions, grids = [], [], []
    for _ in range(experts):
        bases = rng.integers(120, 126, rows).astype(np.uint8)
        grid = bases[:, None] + rng.integers(0, 2, (rows, columns)).astype(np.uint8)
        grid.reshape(-1)[::97] = rng.integers(116, 130, grid.reshape(-1)[::97].shape)
        for row in rng.choice(rows - 4, 3, replace=False):
            grid[row : row + 4] = rng.integers(114, 132, (4, columns))
        selected = grid == bases[:, None] + 1
        outside = (grid != bases[:, None]) & ~selected
        selectors = np.packbits(selected, axis=1, bitorder="little")
        fixed.append(
            torch.from_numpy(
                np.concatenate(
                    (bases.reshape(-1, 16), selectors.reshape(rows // 16, -1)), 1
                )
            )
        )
        position = np.flatnonzero(outside).astype(np.uint32)
        exceptions.append(
            torch.from_numpy(
                position | (grid.reshape(-1)[position].astype(np.uint32) << 24)
            )
        )
        grids.append(grid)
    batch = make_x4t_scale_batch(
        fixed, exceptions, rows=rows, columns=columns, device=device
    )
    return batch, torch.from_numpy(np.stack(grids)).to(device)


def _prepare_arms(h, n, layout, monkeypatch, *, experts=8, seed=0):
    """Native, per-call expansion and inline CSF experts of one MXFP4 layer."""
    device = require_b12x()
    first, s13 = _csf_planes(2 * n, h // 32, experts=experts, seed=seed)
    second, s2 = _csf_planes(h, n // 32, experts=experts, seed=seed + 1)
    w13 = torch.randint(
        0, 256, (experts, 2 * n, h // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(0, 256, (experts, h, n // 2), dtype=torch.uint8, device=device)
    one = torch.ones(experts, device=device)
    weight_plan = moe.plan_weights(
        source=moe.PackedSource(format="fp4_e8m0_k32", w13_layout=layout),
        activation=moe.ActivationSpec(
            mode="a8", nonlinearity="silu", io_dtype=torch.bfloat16
        ),
        geometry=moe.MoEGeometry(
            num_experts=experts, hidden_size=h, intermediate_size=n
        ),
    )
    native = moe.prepare_weights(
        plan=weight_plan,
        weights=moe.PackedWeights(
            w13=w13.clone(),
            w2=w2.clone(),
            w13_block_scales=s13,
            w2_block_scales=s2,
            w13_global_scales=one,
            w2_global_scales=one,
        ),
    )
    arms, scratch = [native], []
    for inline in ("0", "1"):
        monkeypatch.setenv("B12X_W4A8_CSF_INLINE", inline)
        buffers = (torch.empty_like(s13), torch.empty_like(s2))
        arms.append(
            moe.prepare_weights(
                plan=weight_plan,
                weights=moe.Mxfp4CsfWeights(
                    w13=w13.clone(),
                    w2=w2.clone(),
                    w13_scales=first,
                    w2_scales=second,
                    w13_scale_scratch=buffers[0],
                    w2_scale_scratch=buffers[1],
                ),
            )
        )
        scratch.append(buffers)
    monkeypatch.delenv("B12X_W4A8_CSF_INLINE")
    return arms, scratch


def _assert_bitwise(expected, actual):
    assert torch.isfinite(expected).all() and torch.count_nonzero(expected)
    assert torch.equal(expected.view(torch.int16), actual.view(torch.int16))


@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("layout", ["w13", "w31"])
@pytest.mark.parametrize(
    "h,n,tokens",
    [
        (h, n, tokens)
        for h, n in ((5120, 576), (1024, 320), (512, 256))
        for tokens in (1, 8, 17, 64, 512)
        # Deterministic N128 decode at 8 tokens plans direct routing, which
        # that reduction contract does not support (independent of CSF).
        if not (n % 128 == 0 and tokens == 8)
    ],
)
def test_inline_scales_match_native_and_expansion_under_poisoned_replay(
    h, n, tokens, layout, ids_dtype, monkeypatch, after_plan=None
):
    """Inline reads equal native scales and per-call expansion bit for bit.

    The shared expansion scratch is poisoned with NaN E8M0 bytes before every
    replay; inline launches never read or write it. N % 128 == 0 keeps the
    per-call expansion (no inline storage).
    """
    device = require_b12x()
    e, topk = 8, 2
    (native, expanded, inline), scratch = _prepare_arms(h, n, layout, monkeypatch)
    compact = n % 128 == 64
    assert expanded._impl.mxfp4_csf is not None
    assert expanded._impl.mxfp4_csf_inline is None
    assert not expanded.plan._impl.w4a8_csf_inline
    assert inline.plan._impl.w4a8_csf_inline == compact
    assert (inline._impl.mxfp4_csf_inline is not None) == compact
    assert (inline._impl.mxfp4_csf is None) == compact
    if compact:
        # Canonical scale storage stays the caller's scratch; launches read
        # the inline planes instead.
        assert inline._impl.w1_blockscale.data_ptr() == scratch[1][0].data_ptr()
    arms = (native, expanded, inline)
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in arms
    ]
    if after_plan is not None:
        after_plan()
    source = torch.randn(tokens, h, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(tokens)]
    ).to(ids_dtype)
    probabilities = torch.softmax(torch.randn(tokens, topk, device=device), -1)

    def prepare(state):
        buffers = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=buffers,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(buffers, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"mxfp4-csf-inline-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        owners, outputs, graphs = [], [], []
        for plan in plans:
            buffers = tuple(
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
                scratch=buffers,
                input_scales_static=True,
            )
            owners.append((buffers, binding))
            outputs.append(output)
        session.freeze()
        for _, binding in owners:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                moe.run(binding=binding)
            graphs.append(graph)
        for _ in range(3):
            source.neg_()
            ids.add_(1).remainder_(e)
            for buffer in scratch[1]:
                buffer.view(torch.uint8).fill_(0xFF)
            for output in outputs:
                output.fill_(float("nan"))
            allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert (
                torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
            )
            _assert_bitwise(outputs[0], outputs[1])
            _assert_bitwise(outputs[0], outputs[2])
            if compact:
                assert all(
                    bool((buffer.view(torch.uint8) == 0xFF).all())
                    for buffer in scratch[1]
                )
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("capacity", [8, 512])
def test_inline_capacity_plan_reuses_its_launches_for_live_counts(
    capacity, monkeypatch
):
    """One planned capacity serves smaller live counts under frozen resolution."""
    device = require_b12x()
    e, h, n, topk = 8, 5120, 576, 6
    (native, _, inline), scratch = _prepare_arms(h, n, "w31", monkeypatch, seed=3)
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=capacity, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in (native, inline)
    ]
    source = torch.randn(capacity, h, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(capacity)]
    ).to(torch.int32)
    probabilities = torch.softmax(torch.randn(capacity, topk, device=device), -1)

    def bind(plan, rows):
        buffers = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in plan.scratch_specs()
        )
        output = torch.empty_like(source[:rows])
        binding = moe.bind(
            plan,
            a=source[:rows],
            topk_ids=ids[:rows],
            topk_weights=probabilities[:rows],
            output=output,
            scratch=buffers,
            input_scales_static=True,
        )
        return binding, output, buffers

    def prepare(state):
        buffers = tuple(
            torch.empty(s.shape, dtype=s.dtype, device=device)
            for s in state.scratch.scratch_specs()
        )
        output = torch.empty_like(source)
        binding = state.bind(
            a=source,
            topk_ids=ids,
            topk_weights=probabilities,
            output=output,
            scratch=buffers,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: state.run(binding), output=output, owners=(buffers, binding)
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(name=f"csf-inline-live-{i}", prepare_call=prepare)
                for i, p in enumerate(plans)
            )
        )
        session.freeze()
        for rows in sorted({1, 3, capacity // 2 + 1, capacity}):
            (reference, expected, _), (binding, actual, _) = (
                bind(plan, rows) for plan in plans
            )
            for buffer in scratch[1]:
                buffer.view(torch.uint8).fill_(0xFF)
            moe.run(binding=reference)
            moe.run(binding=binding)
            torch.cuda.synchronize()
            _assert_bitwise(expected, actual)


@pytest.mark.parametrize("tokens", [17, 512])
def test_inline_plans_above_the_limit_expand_the_inline_storage(tokens, monkeypatch):
    """Plans above the inline token limit run native kernels over expanded scales.

    The limit is lowered so that small capacities take the large-call path: the
    poisoned scratch receives every expert's native scales from the inline
    storage before each replay, and outputs equal native scales bit for bit.
    """
    from b12x.moe.fused_moe import _impl

    device = require_b12x()
    e, h, n, topk = 8, 5120, 576, 6
    (native, _, inline), scratch = _prepare_arms(h, n, "w31", monkeypatch, seed=5)
    monkeypatch.setattr(_impl, "W4A8_CSF_INLINE_MAX_TOKENS", 16)
    plans = [
        moe.plan_execution(
            experts=owner,
            capacity=moe.ExecutionCapacity(max_tokens=tokens, top_k=topk),
            invocation={"fast_math": False},
            routing=moe.RoutingSpec(deterministic_output=True),
        )
        for owner in (native, inline)
    ]
    monkeypatch.setattr(_impl, "W4A8_CSF_INLINE_MAX_TOKENS", 1536)
    source = torch.randn(tokens, h, dtype=torch.bfloat16, device=device) * 0.1
    ids = torch.stack(
        [torch.randperm(e, device=device)[:topk] for _ in range(tokens)]
    ).to(torch.int32)
    probabilities = torch.softmax(torch.randn(tokens, topk, device=device), -1)

    def bind(plan):
        buffers = tuple(
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
            scratch=buffers,
            input_scales_static=True,
        )
        return PreparedCall(
            run=lambda: moe.run(binding=binding),
            output=output,
            owners=(buffers, binding),
        )

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            tuple(
                p.request(
                    name=f"csf-inline-large-{i}",
                    prepare_call=lambda state, p=p: bind(p),
                )
                for i, p in enumerate(plans)
            )
        )
        calls = [bind(plan) for plan in plans]
        session.freeze()
        graphs = []
        for call in calls:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                call.run()
            graphs.append(graph)
        for _ in range(2):
            source.neg_()
            ids.add_(1).remainder_(e)
            for buffer in scratch[1]:
                buffer.view(torch.uint8).fill_(0xFF)
            for call in calls:
                call.output.fill_(float("nan"))
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            _assert_bitwise(calls[0].output, calls[1].output)
            reference = native._impl.representation.value
            for buffer, expected in zip(
                scratch[1], (reference.w13_sfb, reference.w2_sfb), strict=True
            ):
                assert torch.equal(
                    buffer.view(torch.uint8).view(-1),
                    expected.view(torch.uint8).view(-1),
                )
        for graph in graphs:
            graph.reset()


def test_inline_requires_compact_split_kernels():
    """Inline storage is accepted only by the compact N64 kernels that read it."""
    from b12x.moe._shared.kernels.w4a8_compact_projection import (
        W4A8CompactMicroProjectionKernel,
    )
    from b12x.moe._shared.kernels.w4a8_phase1 import W4A8MaterializedPhase1Kernel
    from b12x.moe._shared.kernels.w4a8_phase2 import W4A8MaterializedPhase2Kernel

    with pytest.raises(ValueError):
        W4A8CompactMicroProjectionKernel(1024, 256, 2, csf_inline=True)
    with pytest.raises(ValueError):
        W4A8MaterializedPhase1Kernel(source_tile_m=16, csf_inline=True)
    with pytest.raises(ValueError):
        W4A8MaterializedPhase2Kernel(source_tile_m=16, csf_inline=True)


def test_inline_capacity_control_is_retained_after_declaration(monkeypatch):
    from b12x.moe.fused_moe import _impl

    monkeypatch.setattr(_impl, "W4A8_CSF_INLINE_MAX_TOKENS", 1536)
    test_inline_scales_match_native_and_expansion_under_poisoned_replay(
        1024,
        320,
        17,
        "w31",
        torch.int32,
        monkeypatch,
        after_plan=lambda: monkeypatch.setattr(_impl, "W4A8_CSF_INLINE_MAX_TOKENS", 0),
    )
