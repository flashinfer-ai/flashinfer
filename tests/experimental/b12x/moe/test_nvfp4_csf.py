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
    deterministic=True,
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
        packed=replace(packed, w13_block_scales=buffers[0], w2_block_scales=buffers[1]),
        w13_scales=compress_fixture(s13, 2 * n, h // 16, raw=raw),
        w2_scales=compress_fixture(s2, h, n // 16, raw=raw),
    )
    experts = [
        moe.prepare_weights(plan=weight_plan, weights=w) for w in (packed, compressed)
    ]
    from b12x.moe.fused_moe._tuning import MoeDecodeConfig

    override = None if inline_scales is None else MoeDecodeConfig(
        backend="dynamic", route_planner="internal", max_active_clusters=None,
        dynamic_tile_m=16, dynamic_route_mode="grouped",
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
            if inline_scales:
                assert all(torch.all(buf.view(torch.uint8) == 0x7F) for buf in buffers)
        for graph in graphs:
            graph.reset()


@pytest.mark.parametrize("inline_scales", [False, True])
@pytest.mark.parametrize("tokens,n", [(1, 64), (17, 320)])
def test_scale_decoder_override_preserves_expert_graph_output(inline_scales, tokens, n):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        tokens, "a4", True, n, inline_scales=inline_scales
    )


@pytest.mark.parametrize("deterministic", [False, True])
def test_indexed_scale_programs_are_declared_for_preparation(deterministic):
    test_native_expert_output_and_shared_scratch_poisoned_replay(
        1, "a4", True, 128, inline_scales=False, autotune=True,
        deterministic=deterministic,
    )
