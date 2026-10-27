"""Fully bound mHC launches survive compilation and allocation-stable graph replay."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.mhc import make_inputs, pre_reference

from ..conftest import require_b12x
from ._mhc import bind, prepare


@pytest.mark.parametrize("tokens", [1, 65])
@pytest.mark.parametrize("normalized", [False, True])
@torch.no_grad()
def test_bound_mhc_compiled_graph(tokens, normalized):
    device, hidden = require_b12x(), 4096
    residual, x, fn, scale, bias = make_inputs(
        tokens=tokens, hidden_size=hidden, seed=9420 + tokens, device=device
    )
    options = dict(rms_eps=1e-5, hc_eps=1e-6, sinkhorn_iters=20)
    _, previous_post, previous_comb = pre_reference(
        residual, fn, scale, bias, **options
    )
    if normalized:
        options.update(
            norm_weight=torch.randn(hidden, device=device, dtype=torch.bfloat16),
            norm_eps=1e-5,
        )
    pre_args = (residual, fn, scale, bias)
    post_pre_args = (
        x,
        residual,
        previous_post.contiguous(),
        previous_comb.contiguous(),
        fn,
        scale,
        bias,
    )
    output = torch.empty_like(residual)
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        pre_plan = prepare(session, "pre", pre_args, options)
        post_pre_plan = prepare(session, "post_pre", post_pre_args, options)
        pre_binding, post_pre_binding = bind(pre_plan), bind(post_pre_plan)
        post_plan = mhc.plan(
            mhc.Caps(device=device, max_tokens=tokens, hidden_size=hidden),
            invocation={"operation": "post", "output_mode": "provided"},
        )
        session.prepare(
            (
                post_plan.request(
                    name="post",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: _impl._b12x_mhc_post_impl(
                            x,
                            residual,
                            previous_post.contiguous(),
                            previous_comb.contiguous(),
                            out=output,
                            _state=state,
                        )
                    ),
                ),
            )
        )

        def forward():
            mhc.run_pre(*pre_args, **options, binding=pre_binding)
            mhc.run_post_pre(*post_pre_args, **options, binding=post_pre_binding)
            mhc.run_post(
                x,
                post_pre_binding.out,
                pre_binding.post_buffer,
                pre_binding.comb_buffer,
                out=output,
                plan=post_plan,
            )

        traced = make_fx(forward)()
        operations = [
            str(n.target) for n in traced.graph.nodes if n.op == "call_function"
        ]
        assert operations.count("b12x.mhc_bound.default") == 2
        assert "b12x.mhc_post_out.default" in operations
        compiled = torch.compile(forward, fullgraph=True)
        compiled()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled()
        session.freeze()
        actual = [
            *(
                getattr(binding, name)
                for binding in (pre_binding, post_pre_binding)
                for name in ("out", "y", "post_buffer", "comb_buffer")
            ),
            output,
        ]
        for _ in range(3):
            residual.normal_()
            x.normal_()
            _impl._b12x_mhc_pre_impl(
                *pre_args, **options, binding=pre_binding, _state=pre_binding.state
            )
            _impl._b12x_mhc_post_pre_impl(
                *post_pre_args,
                **options,
                binding=post_pre_binding,
                _state=post_pre_binding.state,
            )
            _impl._b12x_mhc_post_impl(
                x,
                post_pre_binding.out,
                pre_binding.post_buffer,
                pre_binding.comb_buffer,
                out=output,
                _state=_impl._prepared_state(post_plan, output_mode="provided"),
            )
            expected = [value.clone() for value in actual]
            for value in actual:
                value.zero_()
            torch.cuda.synchronize()
            before = torch.cuda.memory_stats()
            graph.replay()
            torch.cuda.synchronize()
            after = torch.cuda.memory_stats()
            for key in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[key] == after[key]
            for value, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
        graph.reset()
