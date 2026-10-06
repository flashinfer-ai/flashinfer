"""A prepared plan survives Dynamo, graph caching, and changed-input CUDA replay."""
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from b12x.preparation import FrozenMapping, PreparationSession, PreparedCall, require_prepared
from b12x.norm import hyperconnection as hc
from b12x.norm.hyperconnection import _impl, _preparation
from ..conftest import require_b12x


def test_prepared_native_swiglu_eager_compile_and_capture(monkeypatch):
    device = require_b12x()
    source = torch.tensor(
        [[-20.0, -2.0, -0.5, 0.5, 1.5, 20.0, 20.0, -20.0, 1.75, 1.75, -1.75, 20.0]],
        device=device, dtype=torch.bfloat16,
    )
    output = torch.empty((1, 6), device=device, dtype=torch.bfloat16)
    declaration = hc.plan(
        hc.Caps(device=device, max_tokens=1, hidden_size=6),
        invocation=FrozenMapping({"operation": "swiglu", "limit": 2.0}),
    )
    assert declaration.prepared is None

    def call(state):
        return PreparedCall(run=lambda: _impl.run_swiglu_impl(
            source, limit=2.0, out=output, plan=state,
        ))

    request = declaration.request(name="swiglu", prepare_call=call)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        assert declaration.prepared is not None
        plan = declaration

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared plan attempted configuration or compilation")

        monkeypatch.setattr(_preparation, "compile_hyperconnection", forbidden)
        monkeypatch.setattr(_preparation, "TUNING", replace(_preparation.TUNING, default_config=forbidden))

        def consume(x, out, prepared):
            return hc.run_swiglu(x, limit=2.0, out=out, plan=prepared)

        def expected():
            gate, up = source.chunk(2, dim=1)
            return (F.silu(gate.float().clamp(max=2.0)) * up.float().clamp(-2.0, 2.0)).bfloat16()

        consume(source, output, plan)
        torch.testing.assert_close(output, expected(), rtol=0, atol=0)
        compiled = torch.compile(consume, fullgraph=True)
        compiled(source, output, plan)
        torch.testing.assert_close(output, expected(), rtol=0, atol=0)
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    consume(source, output, plan)
            pointer = output.data_ptr()
            source.add_(0.25)
            output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize(device)
            assert output.data_ptr() == pointer
            torch.testing.assert_close(output, expected(), rtol=0, atol=0)
        finally:
            graph.reset()


def test_mhc_runtime_projection_splits_share_code_across_prepared_plans(monkeypatch):
    from b12x._lib import compiler
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl as impl
    from b12x.testing.mhc import make_inputs, post_reference, pre_reference

    device, hidden = require_b12x(), 4096
    requests, inputs = [], {}
    for rows in (1, 128):
        residual, x, fn, scale, bias = make_inputs(tokens=rows, hidden_size=hidden, seed=17, device=device,)
        _, post, comb = pre_reference(residual, fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,)
        norm = torch.ones(hidden, device=device, dtype=torch.bfloat16)
        for splits in (1, 16):
            name = f"m{rows}-s{splits}"
            tensors = (x, residual, post, comb, fn, scale, bias, norm)
            inputs[name] = tensors
            config = mhc.MhcConfig(
                backend="tf32_tma", projection_tile_m=16, projection_tile_n=8,
                projection_tile_k=256, projection_num_stages=1,
                projection_num_m_warps=1, projection_num_n_warps=1, projection_k_splits=splits,
            )
            declaration = mhc.plan(
                mhc.Caps(device=device, max_tokens=rows, hidden_size=hidden, split_k=64),
                invocation={
                    "operation": "post_pre", "output_mode": "functional",
                    "has_norm_weight": True, "rms_eps": 1e-6, "hc_eps": 1e-6,
                    "sinkhorn_iters": 20, "norm_eps": 1e-6,
                },
                override=config,
            )

            def prepare(state, tensors=tensors):
                x, residual, post, comb, fn, scale, bias, norm = tensors
                return PreparedCall(run=lambda: impl._b12x_mhc_post_pre_impl(
                    x, residual, post, comb, fn, scale, bias,
                    rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
                    norm_weight=norm, norm_eps=1e-6, _state=state,
                ))

            requests.append(declaration.request(name=name, prepare_call=prepare))

    def consume(x, residual, post, comb, fn, scale, bias, norm, plan):
        return mhc.run_post_pre(
            x, residual, post, comb, fn, scale, bias,
            rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
            norm_weight=norm, norm_eps=1e-6, plan=plan,
        )

    def check(outputs, tensors):
        x, residual, post, comb, fn, scale, bias, norm = tensors
        carry = post_reference(x, residual, post, comb)
        torch.testing.assert_close(outputs[0], carry, rtol=0, atol=2e-2)
        # The downstream oracle consumes the actual permitted BF16 carry
        # rounding, rather than propagating a different reference FMA ordering.
        raw, expected_post, expected_comb = pre_reference(outputs[0], fn, scale, bias, rms_eps=1e-6, hc_eps=1e-6,
        sinkhorn_iters=20, y_dtype=torch.float32,)
        expected_y = (
            raw.bfloat16().float() * torch.rsqrt(raw.square().mean(-1, keepdim=True) + 1e-6)
            * norm.float()
        ).bfloat16()
        for actual, expected in zip(outputs[1:], (expected_post, expected_comb, expected_y)):
            assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
            similarity = F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            assert similarity >= 0.9998
        torch.testing.assert_close(outputs[1], expected_post, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(outputs[2], expected_comb, rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(outputs[3], expected_y, rtol=1e-2, atol=2e-2)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare(requests)
        plans = {request.name: request.plan for request in requests}
        projection, finalizers = set(), set()
        for plan in plans.values():
            for program in require_prepared(plan, "norm.mhc").launchers["partial"].__b12x_programs__:
                if "mhc_prefill_tf32_project_tma_" in program.name:
                    projection.add(program)
                if "mhc_finalize_gram_" in program.name:
                    finalizers.add(program)
        assert len(projection) == 1
        assert len(finalizers) == 2

        def forbidden(*args, **kwargs):
            raise AssertionError("prepared runtime reached compiler or loader")

        monkeypatch.setattr(compiler, "compile", forbidden)
        monkeypatch.setattr(compiler, "_load_cute_compile_from_disk", forbidden)
        compiled = torch.compile(consume, fullgraph=True)
        for name, tensors in inputs.items():
            plan = plans[name]
            check(consume(*tensors, plan), tensors)
            check(compiled(*tensors, plan), tensors)
            graph = torch.cuda.CUDAGraph()
            try:
                with session.capture():
                    with torch.cuda.graph(graph):
                        outputs = consume(*tensors, plan)
                tensors[1].add_(0.03125)
                for output in outputs:
                    output.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize(device)
                check(outputs, tensors)
            finally:
                graph.reset()


@pytest.mark.parametrize("provided", [False, True])
def test_dense_prepared_plan_compiles_through_the_graph_cache(provided, monkeypatch):
    from b12x import gemm
    from b12x._lib.intrinsics import as_grouped_scale_view_mx

    device = require_b12x()
    rows, columns, width, batch = 3, 128, 256, 2
    generator = torch.Generator(device=device).manual_seed(19)
    a_storage = (torch.randn(batch, rows, width, generator=generator, device=device) * 0.1).to(torch.float8_e4m3fn)
    b_storage = (torch.randn(batch, columns, width, generator=generator, device=device) * 0.1).to(torch.float8_e4m3fn)
    a, b = a_storage.permute(1, 2, 0), b_storage.permute(1, 2, 0)

    def scales(count):
        storage = torch.full(
            (batch, ((count + 127) // 128) * ((width // 32 + 3) // 4) * 512),
            127, dtype=torch.uint8, device=device,
        )
        return as_grouped_scale_view_mx(storage, count, width)

    sfa, sfb = scales(rows), scales(columns)
    output = (
        torch.empty(batch, rows, columns, device=device, dtype=torch.bfloat16).permute(1, 2, 0)
        if provided else None
    )
    query = gemm.DenseGemmQuery(
        recipe="mxfp8", entry_point="gemm.mm", weight_storage="native",
        output_dtype="bfloat16", batch=batch, max_rows=rows,
        in_features=width, out_features=columns,
        output_mode="provided" if provided else "functional", alpha_mode="unit",
    )
    declaration = gemm.plan(query)
    request = declaration.request(
        name="dense",
        prepare_call=lambda state: PreparedCall(
            run=lambda: state.run(a, sfa, b, sfb, None, output, None, torch.bfloat16, None),
        ),
    )
    expected = torch.bmm(a_storage.float(), b_storage.float().transpose(1, 2)).bfloat16().permute(1, 2, 0)

    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        plan = declaration

        def consume(a, sfa, b, sfb, prepared):
            return gemm.mm((a, sfa), (b, sfb), out=output, plan=prepared)

        import torch._inductor.config as inductor_config
        with inductor_config.patch(fx_graph_cache=True):
            compiled = torch.compile(consume, fullgraph=True)
            actual = compiled(a, sfa, b, sfb, plan)
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)


@pytest.mark.parametrize("operation", ("pre", "post_pre"))
def test_mhc_lagged_pair_prepared_replays_bf16_collapse(operation, monkeypatch):
    """The paired coefficient ABI is prepared once and replays changed inputs."""
    from b12x._lib import compiler
    from b12x.norm import mhc
    from b12x.norm.mhc import _impl as impl
    from b12x.norm.mhc import _preparation as mhc_preparation
    from b12x.testing.mhc import make_inputs, post_reference, pre_reference

    device, tokens, hidden = require_b12x(), 1, 4096
    residual, x, fn, scale, bias = make_inputs(
        tokens=tokens, hidden_size=hidden, seed=923_101, device=device,
    )
    pre_input = x
    pre_fn = fn.reshape(24, 4, hidden).sum(dim=1).contiguous()
    incoming = torch.tensor([[0.75, -0.5, 0.25, 0.125]], device=device)
    next_mix = torch.empty_like(incoming)
    norm = torch.linspace(0.5, 1.5, hidden, device=device).bfloat16()
    _, prev_post, prev_comb = pre_reference(
        residual, fn, scale, bias, rms_eps=1e-20, hc_eps=1e-6, sinkhorn_iters=20,
    )
    declaration = mhc.plan(
        mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden, split_k=64),
        invocation=FrozenMapping({
            "operation": operation, "lagged_mix": True, "output_mode": "functional",
            "has_norm_weight": True, "rms_eps": 1e-20, "hc_eps": 1e-6,
            "sinkhorn_iters": 20, "norm_eps": 1e-20,
        }),
    )

    def prepare(state):
        if operation == "pre":
            return PreparedCall(run=lambda: impl._b12x_mhc_pre_impl(
                pre_input, pre_fn, scale, bias, pre_mix=incoming, pre_out=next_mix,
                norm_weight=norm, rms_eps=1e-20, hc_eps=1e-6,
                sinkhorn_iters=20, norm_eps=1e-20, _state=state,
            ))
        return PreparedCall(run=lambda: impl._b12x_mhc_post_pre_impl(
            x, residual, prev_post.contiguous(), prev_comb.contiguous(), fn, scale, bias,
            pre_mix=incoming, pre_out=next_mix, norm_weight=norm, rms_eps=1e-20,
            hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20, _state=state,
        ))

    request = declaration.request(name=operation, prepare_call=prepare)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((request,))
        plan = declaration

        def consume():
            if operation == "pre":
                return mhc.run_pre(
                    pre_input, pre_fn, scale, bias, pre_mix=incoming, pre_out=next_mix,
                    norm_weight=norm, rms_eps=1e-20, hc_eps=1e-6,
                    sinkhorn_iters=20, norm_eps=1e-20, plan=plan,
                )
            return mhc.run_post_pre(
                x, residual, prev_post.contiguous(), prev_comb.contiguous(), fn, scale, bias,
                pre_mix=incoming, pre_out=next_mix, norm_weight=norm, rms_eps=1e-20,
                hc_eps=1e-6, sinkhorn_iters=20, norm_eps=1e-20, plan=plan,
            )

        outputs = consume()
        current = pre_input.unsqueeze(1).expand(-1, 4, -1) if operation == "pre" else post_reference(
            x, residual, prev_post, prev_comb,
        )
        collapsed = (incoming.unsqueeze(-1) * current.float()).sum(1).bfloat16()
        expected_y = (
            collapsed.float()
            * torch.rsqrt(collapsed.float().square().mean(-1, keepdim=True) + 1e-20)
            * norm.float()
        ).bfloat16()
        torch.testing.assert_close(outputs[3], expected_y, rtol=0, atol=8e-3)

        monkeypatch.setattr(compiler, "compile", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared lagged runtime attempted compilation")
        ))
        monkeypatch.setattr(mhc_preparation, "compile_mhc", lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prepared lagged runtime attempted materialization")
        ))
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture():
                with torch.cuda.graph(graph):
                    captured = consume()
            pointers = tuple(t.data_ptr() for t in (*captured, next_mix))
            residual.add_(1)
            if operation == "pre":
                pre_input.add_(0.5)
            incoming.copy_(incoming.roll(1, dims=1))
            graph.replay()
            torch.cuda.synchronize(device)
            assert tuple(t.data_ptr() for t in (*captured, next_mix)) == pointers
            eager_after = consume()
            for replayed, eager in zip(captured, eager_after, strict=True):
                torch.testing.assert_close(replayed, eager, rtol=0, atol=0)
        finally:
            graph.reset()
