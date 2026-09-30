"""Prepare mHC invocations using the regression's real inputs and trial-owned scratch."""
from dataclasses import replace

import torch

from b12x.norm import mhc
from b12x.norm.mhc import _impl
from b12x.norm.mhc._tuning import TUNING
from b12x.preparation import FrozenMapping, PreparedCall


def declaration(operation, args, options, *, capacity=None, output_mode="provided", backend=None, lagged_prepare=None, partials_per_cta=None):
    residual = args[0 if operation == "pre" else 1]
    norm = options.get("norm_weight")
    invocation = dict(
        operation=operation, output_mode=output_mode,
        expanded_residual=operation == "pre" and residual.ndim == 3,
        lagged_mix=options.get("pre_mix") is not None,
        has_norm_weight=norm is not None,
        norm_weight_dtype="bfloat16" if norm is None else str(norm.dtype).removeprefix("torch."),
        has_fn_bf16=options.get("fn_bf16") is not None,
        rms_eps=options["rms_eps"], hc_eps=options["hc_eps"],
        sinkhorn_iters=options["sinkhorn_iters"], norm_eps=options.get("norm_eps", 0.0),
    )
    caps = mhc.Caps(device=residual.device, max_tokens=capacity or residual.shape[0],
                    hidden_size=residual.shape[-1])
    plan = mhc.plan(caps, invocation=FrozenMapping(invocation))
    if backend is not None or lagged_prepare is not None or partials_per_cta is not None:
        config = TUNING.configure(plan.query, device=None).default
        if backend is not None:
            config = replace(config, backend=backend, lagged_prepare=False, partials_per_cta=4)
        if lagged_prepare is not None:
            config = replace(config, lagged_prepare=lagged_prepare,
                             partials_per_cta=config.partials_per_cta if lagged_prepare else 4)
        if partials_per_cta is not None:
            config = replace(config, partials_per_cta=partials_per_cta)
        plan = mhc.plan(caps, invocation=FrozenMapping(invocation), override=config)
    return plan


def prepare(session, operation, args, options, *, plan=None, **plan_options):
    if plan is None:
        plan = declaration(operation, args, options, **plan_options)
    residual = args[0 if operation == "pre" else 1]
    rows, hidden = residual.shape[0], residual.shape[-1]
    entry = getattr(_impl, f"_b12x_mhc_{operation}_impl")

    def call(state):
        if state.query.output_mode == "functional":
            trial_options = dict(options)
            if trial_options.get("pre_mix") is not None:
                trial_options["pre_out"] = torch.empty_like(trial_options["pre_mix"])
            return PreparedCall(run=lambda: entry(*args, **trial_options, _state=state))
        spec, = state.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=residual.device)
        bound = state.bind(
            scratch=scratch, tokens=rows,
            out=torch.empty((rows, 4, hidden), dtype=torch.bfloat16, device=residual.device),
            y=torch.empty((rows, hidden), dtype=torch.bfloat16, device=residual.device),
            post=torch.empty((rows, 4), device=residual.device),
            comb=torch.empty((rows, 4, 4), device=residual.device),
            pre_out=torch.empty((rows, 4), device=residual.device) if state.query.lagged_mix else None,
        )
        trial_options = {name: value for name, value in options.items()
                         if name not in ("pre_out", "residual_out", "y_out", "post_out", "comb_out")}
        return PreparedCall(run=lambda: entry(*args, **trial_options, binding=bound, _state=state),
                            owners=(scratch, bound))

    session.prepare((plan.request(name=f"mhc-{operation}", prepare_call=call),))
    return plan


def bind(plan, *, tokens=None, pre_out=None, y=None):
    q = plan.query
    rows = q.max_tokens if tokens is None else tokens
    spec, = plan.scratch_specs()
    device = spec.device
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
    return mhc.bind(
        plan, scratch=scratch, tokens=rows,
        out=torch.empty((rows, 4, q.hidden_size), dtype=torch.bfloat16, device=device),
        y=torch.empty((rows, q.hidden_size), dtype=torch.bfloat16, device=device) if y is None else y,
        post=torch.empty((rows, 4), device=device), comb=torch.empty((rows, 4, 4), device=device),
        pre_out=pre_out,
    )


def prepare_collapse(session, source, mix, output):
    plan = mhc.plan(
        mhc.Caps(device=source.device, max_tokens=source.shape[0], hidden_size=source.shape[-1]),
        invocation=FrozenMapping({"operation": "collapse", "collapse_weighted": mix is not None}),
    )
    session.prepare((plan.request(name="collapse", prepare_call=lambda state: PreparedCall(
        run=lambda: _impl._run_collapse_impl(source, mix, out=output, _state=state),
    )),))
    return plan


import pytest
from b12x.preparation import PreparationSession
from ..conftest import require_b12x


@pytest.fixture
def mhc_session():
    with PreparationSession(device=require_b12x(), autotune=False, compile_workers=2) as session:
        yield session
