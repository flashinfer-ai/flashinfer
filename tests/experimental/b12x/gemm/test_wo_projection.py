"""Prepared WO projection numerical behavior."""
from __future__ import annotations

from contextlib import ExitStack
from itertools import permutations

import pytest
import torch

from b12x.gemm import wo_projection as wo
from b12x.gemm._shared.wo_mxfp8 import dequantize_mxfp8_rows_torch, quantize_wo_projection_weights_mxfp8_torch
from b12x.preparation import PreparedCall, PreparationSession
from ..conftest import require_b12x


@pytest.mark.parametrize("tokens", [1, 6, 8, 129])
def test_caller_scratch_views_do_not_initialize_scale_padding(tokens):
    from b12x.gemm._shared.wo_mxfp8 import _materialize_wo_projection_scratch

    caps = wo.Caps(device="cpu", max_tokens=129, groups=4,
                   group_width=512, rank=128, hidden=256)
    state = _materialize_wo_projection_scratch(caps, config=wo.WoProjectionConfig())
    scratch = tuple(torch.full(spec.shape, 255, dtype=spec.dtype, device=spec.device)
                    for spec in state.scratch_specs())
    before = tuple(tensor.clone() for tensor in scratch)
    views = state._views_from_scratch(scratch=scratch, tokens=tokens)
    assert views.x_q.values.shape[0] == tokens
    assert views.output.shape == (tokens, 256, 1)
    for actual, expected in zip(scratch, before, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [1, 6, 8])
def test_decode_lowering_retains_declared_packed_weight_contract(rows):
    from types import SimpleNamespace
    from b12x.preparation import DeviceIdentity

    device = SimpleNamespace(ordinal=0, identity=DeviceIdentity(
        "nvidia", (12, 0), 188, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ))
    caps = wo.Caps(device="cuda:0", max_tokens=rows, groups=4,
                   group_width=4096, rank=1024, hidden=4096)
    plan = wo.plan(caps, invocation={"sfb_k_replicated": True, "wo_a_tiled": True, "wo_b_tiled": True})
    config = plan.contract.default_config(plan.query, device.identity)
    plan.contract.validate_query(plan.query, device.identity)
    jobs = plan._compile_jobs(config, device)
    ordinary_a = jobs[0].args[0]
    assert ordinary_a["b_tile_major"] and ordinary_a["sfb_k_reuse"]
    assert tuple(ordinary_a["mma_tiler_mn"]) == (16, 64)
    fused, = (job.args[0] for job in jobs if job.factory.endswith(":_compile_dense_fused_quant_lowering"))
    assert fused["b_tile_major"] and fused["sfb_k_replicated"]
    assert tuple(fused["mma_tiler_mn"]) in ((16, 64), (16, 128))
    assert fused["m"] == rows and fused["source_shape"] == (rows, 1024, 4)


@pytest.mark.parametrize("rows,width,expected", [
    (1, 4096, (16, 64)), (6, 4096, (16, 64)), (8, 4096, (16, 64)),
    (9, 4096, (64, 64)), (15, 4096, (64, 64)), (16, 4096, (32, 64)),
    (9, 512, (32, 64)), (15, 512, (32, 64)), (17, 512, (64, 64)),
])
def test_first_projection_preserves_declared_row_tile(rows, width, expected):
    from types import SimpleNamespace
    from b12x.preparation import DeviceIdentity

    device = SimpleNamespace(ordinal=0, identity=DeviceIdentity(
        "nvidia", (12, 0), 188, "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
    ))
    caps = wo.Caps(device="cuda:0", max_tokens=rows, groups=4,
                   group_width=width, rank=1024, hidden=4096)
    plan = wo.plan(caps)
    config = plan.contract.default_config(plan.query, device.identity)
    a = plan._compile_jobs(config, device)[0].args[0]
    assert tuple(a["mma_tiler_mn"]) == expected
    assert not a["b_tile_major"]


def test_declared_weight_contract_rejects_incompatible_binding():
    from types import SimpleNamespace
    from b12x.gemm.wo_projection._preparation import _PreparedWO

    caps = wo.Caps(device="cpu", max_tokens=6, groups=4,
                   group_width=512, rank=1024, hidden=4096)
    query = wo.plan(caps, invocation={"sfb_k_replicated": True, "wo_b_tiled": True}).query
    state = _PreparedWO(None, query, None, None, None, None)
    with pytest.raises(ValueError, match="replicated block scales"):
        state._check_weights(SimpleNamespace(sfb_k_replicated=False))
    with pytest.raises(ValueError, match="tiled WO-B"):
        state._check_weights(SimpleNamespace(sfb_k_replicated=True,
                                            wo_b=SimpleNamespace(values_tiled=None)))
    caps = wo.Caps(device="cpu", max_tokens=6, groups=4,
                   group_width=4096, rank=1024, hidden=4096)
    state = _PreparedWO(None, wo.plan(caps, invocation={"wo_a_tiled": True}).query,
                        None, None, None, None)
    with pytest.raises(ValueError, match="tiled WO-A"):
        state._check_weights(SimpleNamespace(wo_a=SimpleNamespace(values_tiled=None)))


def _prepared(caps, source, weights, *, invocation=None):
    resources = ExitStack()
    specs = ()
    declaration = wo.plan(caps, invocation=invocation or {})

    def prepare(state):
        nonlocal specs
        specs = state._scratch_state.scratch_specs()
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=caps.device) for spec in specs)
        binding = state.bind(scratch=scratch, source_tgd=source, weights=weights)
        return PreparedCall(run=lambda: state.run(binding))

    session = resources.enter_context(PreparationSession(device=caps.device, autotune=False, compile_workers=2))
    resources.enter_context(session.prepare((declaration.request(
        name="wo", prepare_call=prepare,
    ),)))
    plan = declaration
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=caps.device) for spec in specs)
    return resources, plan, wo.bind(plan, scratch=scratch, source_tgd=source, weights=weights)


def test_plan_bind_run_singleton_group_matches_quantized_reference() -> None:
    require_b12x()
    torch.manual_seed(31005)
    tokens, groups, group_width, rank, hidden = 3, 1, 512, 128, 128
    source = torch.randn((tokens, groups, group_width), device="cuda", dtype=torch.bfloat16) / 4
    wo_a = torch.randn((groups, rank, group_width), device="cuda", dtype=torch.bfloat16) / group_width**0.5
    wo_b = torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16) / (groups * rank) ** 0.5
    weights = quantize_wo_projection_weights_mxfp8_torch(wo_a, wo_b)
    caps = wo.Caps(device=source.device, max_tokens=tokens, groups=groups, group_width=group_width, rank=rank, hidden=hidden)
    resources, plan, binding = _prepared(caps, source, weights)
    with resources:
        actual = wo.run(binding=binding, plan=plan)
        x = wo.quantize_input(source, plan=plan)
        tmp = (dequantize_mxfp8_rows_torch(x.values, x.scale_rows) @ dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows).T).to(torch.bfloat16).unsqueeze(-1)
        tmp_q = wo.quantize_input_b(tmp, plan=plan)
        expected = dequantize_mxfp8_rows_torch(tmp_q.values, tmp_q.scale_rows) @ dequantize_mxfp8_rows_torch(weights.wo_b.values, weights.wo_b.scale_rows).T
        torch.testing.assert_close(actual, expected.to(actual.dtype), rtol=0, atol=0)

def test_inverse_rope_prepared_execution_rejects_mismatched_runtime_pointer_dtype() -> None:
    require_b12x()
    tokens, groups, heads_per_group, nope_dim, rope_dim, rank, hidden = 1, 1, 1, 96, 32, 128, 128
    o = torch.randn((tokens, groups * heads_per_group, nope_dim + rope_dim), device="cuda", dtype=torch.bfloat16)
    positions = torch.zeros((tokens,), device="cuda", dtype=torch.int64)
    cos_sin_cache = torch.randn((4, rope_dim), device="cuda", dtype=torch.bfloat16)
    weights = quantize_wo_projection_weights_mxfp8_torch(
        torch.randn((groups, rank, heads_per_group * (nope_dim + rope_dim)), device="cuda", dtype=torch.bfloat16),
        torch.randn((hidden, groups * rank), device="cuda", dtype=torch.bfloat16),
    )
    caps = wo.Caps(
        device=o.device, max_tokens=tokens, groups=groups, group_width=heads_per_group * (nope_dim + rope_dim),
        rank=rank, hidden=hidden,
    )
    declaration = wo.plan(
        caps,
        invocation={"operation": "inv_rope", "heads_per_group": heads_per_group,
                    "nope_dim": nope_dim, "rope_dim": rope_dim},
    )
    resources = ExitStack()
    specs = ()

    def prepare(state):
        nonlocal specs
        specs = state._scratch_state.scratch_specs()
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device) for spec in specs)
        binding = state.bind_inv_rope(
            scratch=scratch, o=o, positions=positions, cos_sin_cache=cos_sin_cache, weights=weights,
            heads_per_group=heads_per_group, nope_dim=nope_dim, rope_dim=rope_dim,
        )
        return PreparedCall(run=lambda: state.run_inv_rope(binding))

    session = resources.enter_context(PreparationSession(device=o.device, autotune=False, compile_workers=2))
    resources.enter_context(session.prepare((declaration.request(
        name="wo-inv", prepare_call=prepare,
    ),)))
    scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=o.device) for spec in specs)
    with resources:
        plan = declaration
        with pytest.raises(ValueError, match="dtypes differ"):
            wo.bind_inv_rope(
                plan, scratch=scratch, o=o, positions=positions.to(torch.int32),
                cos_sin_cache=cos_sin_cache, weights=weights, heads_per_group=heads_per_group,
                nope_dim=nope_dim, rope_dim=rope_dim,
            )




@torch.no_grad()
def test_prefill_chunk_remainders_reuse_launchers_and_replay():
    require_b12x()
    from dataclasses import replace
    from b12x.preparation._measurement import no_compilation

    device = torch.device("cuda", torch.cuda.current_device())
    capacity, groups, width, rank, hidden = 4096, 2, 4096, 1024, 5120
    counts = (1, 129, 3575, 3582, capacity)
    source = torch.randn(capacity, 16, 512, dtype=torch.bfloat16, device=device) / 8
    positions = torch.arange(capacity, device=device).remainder_(64)
    angles = torch.randn(64, 32, device=device)
    table = torch.cat((angles.cos(), angles.sin()), dim=-1).bfloat16()
    weights = quantize_wo_projection_weights_mxfp8_torch(
        torch.randn(groups, rank, width, device=device, dtype=torch.bfloat16) / width**0.5,
        torch.randn(hidden, groups * rank, device=device, dtype=torch.bfloat16) / (groups * rank)**0.5,
    )
    caps = wo.Caps(device=device, max_tokens=capacity, groups=groups, group_width=width, rank=rank, hidden=hidden)
    invocation = dict(operation="inv_rope", heads_per_group=8, nope_dim=448, rope_dim=64)
    prefill = wo.plan(caps, invocation={**invocation, "dynamic_tokens": True})
    exact = {rows: wo.plan(replace(caps, max_tokens=rows), invocation=invocation) for rows in counts}

    def bind(state, scratch, rows):
        return state.bind_inv_rope(
            scratch=scratch, o=source[:rows], positions=positions[:rows], cos_sin_cache=table,
            weights=weights, heads_per_group=8, nope_dim=448, rope_dim=64,
        )

    def prepare(state):
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device)
                        for spec in state._scratch_state.scratch_specs())
        binding = bind(state, scratch, state.query.max_tokens)
        return PreparedCall(run=lambda: state.run_inv_rope(binding))

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(tuple(plan.request(name=f"wo.{index}", prepare_call=prepare)
                              for index, plan in enumerate((prefill, *exact.values()))))
        scratch = tuple(torch.empty(spec.shape, dtype=spec.dtype, device=device) for spec in prefill.scratch_specs())
        reference_scratch = tuple(torch.empty_like(tensor) for tensor in scratch)
        state = prefill.prepared.state
        launchers = (state.ordinary_a.gemm, state.ordinary_b.gemm,
                     state.quantizers.inv_rope, state.quantizers.group_major)
        session.freeze()
        for rows in counts:
            reference_state = exact[rows].prepared.state
            reference = bind(reference_state, reference_scratch, rows)
            binding = bind(state, scratch, rows)
            for tensor in scratch:
                tensor.fill_(255)
            with no_compilation():
                expected = reference_state.run_inv_rope(reference).clone()
                actual = state.run_inv_rope(binding)
            torch.cuda.synchronize(device)
            assert actual.shape == (rows, hidden)
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            torch.testing.assert_close(binding.x_q.values.view(torch.uint8), reference.x_q.values.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(binding.x_q.scale_rows.view(torch.uint8), reference.x_q.scale_rows.view(torch.uint8), rtol=0, atol=0)
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
            assert torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.99999
            assert launchers == (state.ordinary_a.gemm, state.ordinary_b.gemm,
                                 state.quantizers.inv_rope, state.quantizers.group_major)
            if rows != 3575:
                continue
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture(), torch.cuda.graph(graph):
                    replayed = state.run_inv_rope(binding)
                source[:rows].neg_()
                positions[:rows].add_(1).remainder_(table.shape[0])
                for tensor in scratch:
                    tensor.fill_(255)
                replayed.fill_(float("nan"))
                pointers = tuple(tensor.data_ptr() for tensor in (*scratch, replayed))
                allocated = torch.cuda.memory_allocated(device)
                graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert tuple(tensor.data_ptr() for tensor in (*scratch, replayed)) == pointers
                with no_compilation():
                    expected = reference_state.run_inv_rope(reference)
                torch.testing.assert_close(replayed, expected, rtol=0.01, atol=0.002)
                assert torch.isfinite(replayed).all() and torch.count_nonzero(replayed) > 0
            finally:
                graph.reset()
        with pytest.raises(ValueError, match="exact prepared token count"):
            bind(exact[capacity].prepared.state, reference_scratch, 3575)
        with pytest.raises(ValueError, match="planned token capacity"):
            state._check_tokens(capacity + 1)


@pytest.mark.parametrize("groups,width,rank,hidden", [(4, 4096, 1024, 4096), (2, 4096, 1024, 5120)])
@torch.no_grad()
def test_decode_projection_replays_with_poisoned_caller_scratch(groups, width, rank, hidden):
    """Prepared decode must not depend on padding from an earlier invocation."""
    require_b12x()
    from b12x.preparation._measurement import no_compilation

    device = torch.device("cuda", torch.cuda.current_device())
    counts = (1, 6, 8)
    source = torch.randn(8, groups, width, dtype=torch.bfloat16, device=device) / 8
    weights = quantize_wo_projection_weights_mxfp8_torch(
        torch.randn(groups, rank, width, device=device, dtype=torch.bfloat16) / width**0.5,
        torch.randn(hidden, groups * rank, device=device, dtype=torch.bfloat16) / (groups * rank)**0.5,
    )
    for rows in counts:
        caps = wo.Caps(device=device, max_tokens=rows, groups=groups,
                       group_width=width, rank=rank, hidden=hidden)
        resources, plan, binding = _prepared(caps, source[:rows], weights)
        with resources:
            expected = wo.run(binding=binding, plan=plan).clone()
            graph = torch.cuda.CUDAGraph()
            try:
                with no_compilation(), torch.cuda.graph(graph):
                    replayed = wo.run(binding=binding, plan=plan)
                for operand in (binding.x_q, binding.tmp_q):
                    operand.scale_mma.view(torch.uint8).fill_(255)
                    operand.values.view(torch.uint8).fill_(255)
                binding.tmp.fill_(float("nan"))
                replayed.fill_(float("nan"))
                pointers = tuple(tensor.data_ptr() for tensor in (replayed, binding.tmp, binding.x_q.values))
                allocated = torch.cuda.memory_allocated(device)
                with no_compilation():
                    graph.replay()
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated
                assert pointers == tuple(tensor.data_ptr() for tensor in (replayed, binding.tmp, binding.x_q.values))
                assert torch.isfinite(replayed).all() and torch.count_nonzero(replayed) > 0
                torch.testing.assert_close(replayed, expected, rtol=0.01, atol=0.002)
            finally:
                graph.reset()


def _assert_wo_b_quantized_reduction(actual, binding, state):
    """Accept only the declared split-K arithmetic, including BF16 atomics.

    Four atomic partials can arrive in any order. Comparing two native runs
    as if they promised a fixed order rejects valid results near cancellation.
    Enumerate all orders of the independently calculated quantized partials
    and retain the same pointwise tolerance for the nearest valid result.
    """
    state.quantizers.quantize_b(binding.tmp, binding.tmp_q)
    left = dequantize_mxfp8_rows_torch(binding.tmp_q.values, binding.tmp_q.scale_rows)
    right = dequantize_mxfp8_rows_torch(binding.weights.wo_b.values, binding.weights.wo_b.scale_rows)
    policy = state.fused_b.lowering.policy
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        if policy.split_k_slices == 1 or not policy.split_k_atomic_bf16:
            reference = (left @ right.T).to(actual.dtype)
        else:
            slices = policy.split_k_slices
            assert slices in (2, 4) and left.shape[1] % slices == 0
            width = left.shape[1] // slices
            partials = [
                (left[:, index * width:(index + 1) * width]
                 @ right[:, index * width:(index + 1) * width].T).bfloat16()
                for index in range(slices)
            ]
            reference = torch.zeros_like(actual)
            distance = torch.full_like(actual.float(), float("inf"))
            for order in permutations(range(slices)):
                candidate = torch.zeros_like(actual)
                for index in order:
                    candidate = (candidate.float() + partials[index].float()).bfloat16()
                error = (actual.float() - candidate.float()).abs()
                reference = torch.where(error < distance, candidate, reference)
                distance = torch.minimum(distance, error)
        torch.testing.assert_close(actual, reference, rtol=0.01, atol=0.002)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


@pytest.mark.parametrize("seed", [20260919, 31005, 0])
@pytest.mark.parametrize("rows", [1, 6, 8])
@torch.no_grad()
def test_packed_decode_layout_matches_generic_projection(rows, seed):
    """Packed checkpoint metadata must change storage access, not quantization."""
    require_b12x()
    from b12x.preparation._measurement import no_compilation

    torch.manual_seed(seed)
    device = torch.device("cuda", torch.cuda.current_device())
    groups, width, rank, hidden = 4, 4096, 1024, 4096
    source = torch.randn(rows, groups, width, device=device, dtype=torch.bfloat16) / 8
    # Block scales vary across N and K; uniform scales would hide a bad index.
    scales_a = torch.exp2(torch.randint(-2, 3, (groups * rank // 128, width // 128), device=device).float())
    scales_b = torch.exp2(torch.randint(-2, 3, (hidden // 128, groups * rank // 128), device=device).float())
    weights = wo.pack_weights(
        (torch.randn(groups * rank, width, device=device) / width**0.5).to(torch.float8_e4m3fn),
        scales_a,
        (torch.randn(hidden, groups * rank, device=device) / (groups * rank)**0.5).to(torch.float8_e4m3fn),
        scales_b,
        groups=groups, group_width=width, rank=rank, hidden=hidden,
    )
    assert weights.wo_a.values_tiled is not None and weights.wo_b.values_tiled is not None
    assert weights.sfb_k_replicated
    caps = wo.Caps(device=device, max_tokens=rows, groups=groups,
                   group_width=width, rank=rank, hidden=hidden)
    with ExitStack() as resources:
        plans, bindings = {}, {}
        for arm, invocation in (("generic", {}), ("packed", {"sfb_k_replicated": True, "wo_a_tiled": True, "wo_b_tiled": True})):
            owned, plans[arm], bindings[arm] = _prepared(caps, source, weights, invocation=invocation)
            resources.enter_context(owned)
        packed = plans["packed"].prepared.state.fused_b.lowering
        assert packed.b_tile_major and packed.sfb_k_replicated
        first = plans["packed"].prepared.state.ordinary_a.lowering
        assert first.mma_tiler_mn == (16, 64) and first.b_tile_major and first.sfb_k_reuse
        graph = torch.cuda.CUDAGraph()
        try:
            with no_compilation(), torch.cuda.graph(graph):
                actual = wo.run(binding=bindings["packed"], plan=plans["packed"])
            for sign in (1, -1):
                source.mul_(sign)
                actual.fill_(float("nan"))
                bindings["packed"].x_q.scale_mma.view(torch.uint8).fill_(255)
                pointer, allocated = actual.data_ptr(), torch.cuda.memory_allocated(device)
                with no_compilation():
                    graph.replay()
                    expected = wo.run(binding=bindings["generic"], plan=plans["generic"])
                torch.cuda.synchronize(device)
                assert torch.cuda.memory_allocated(device) == allocated and actual.data_ptr() == pointer
                assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
                # Layout/provenance flags must preserve input quantization and
                # WO-A exactly, independently of WO-B atomic arrival order.
                for field in ("values", "scale_rows"):
                    torch.testing.assert_close(
                        getattr(bindings["packed"].x_q, field).view(torch.uint8),
                        getattr(bindings["generic"].x_q, field).view(torch.uint8),
                        rtol=0, atol=0,
                    )
                torch.testing.assert_close(bindings["packed"].tmp, bindings["generic"].tmp, rtol=0, atol=0)
                left = dequantize_mxfp8_rows_torch(
                    bindings["packed"].x_q.values, bindings["packed"].x_q.scale_rows,
                )
                right = dequantize_mxfp8_rows_torch(weights.wo_a.values, weights.wo_a.scale_rows)
                previous_tf32 = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = False
                try:
                    first_reference = torch.einsum("mkg,nkg->mng", left, right).bfloat16()
                    torch.testing.assert_close(bindings["packed"].tmp, first_reference, rtol=0.01, atol=0.002)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous_tf32
                if packed.policy.split_k_slices <= 2:
                    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.002)
                for name, output in (("packed", actual), ("generic", expected)):
                    _assert_wo_b_quantized_reduction(output, bindings[name], plans[name].prepared.state)
                assert torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0) > 0.99999
        finally:
            graph.reset()
