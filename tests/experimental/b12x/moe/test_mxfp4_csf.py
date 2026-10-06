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
