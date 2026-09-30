"""Concrete HyperConnection and mHC preparation benchmark transactions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from b12x.preparation import FrozenMapping, PreparedCall, PreparationRequest


def _scratch(layout, device):
    """Allocate the caller-owned byte layout or its single scratch spec."""
    if hasattr(layout, "scratch_specs"):
        (spec,) = layout.scratch_specs()
        return torch.empty(spec.shape, dtype=spec.dtype, device=device)
    return torch.empty((layout.nbytes,), dtype=torch.uint8, device=device)


def _producer(output, eps):
    from vllm import _custom_ops as ops

    source = torch.randn_like(output)
    weight = torch.ones(output.shape[-1], dtype=output.dtype, device=output.device)

    def produce():
        ops.rms_norm(
            output.view(-1, output.shape[-1]),
            source.view(-1, source.shape[-1]),
            weight,
            eps,
        )

    return produce, (source, weight)


@dataclass
class _Expected:
    role: str
    inputs: dict
    options: dict


def _hyper_fixture(*, m, hidden, streams, lowrank, eps, device):
    def rand(shape):
        return torch.randn(shape, device=device, dtype=torch.bfloat16)

    fixture = {
        "state": rand((m, streams * hidden)),
        "weight": rand((streams * hidden,)) * 0.05,
        "projected_down": rand((m, lowrank)),
        "logits": rand((m, streams * hidden)),
        "injection": rand((m, streams)),
        "block_output": rand((m, hidden)),
    }
    fixture["producer"], fixture["producer_owners"] = _producer(fixture["state"], eps)
    fixture["projected_producer"], fixture["projected_producer_owners"] = _producer(
        fixture["projected_down"], eps
    )
    return fixture


def _hyper_call_factory(*, role, fixture, eps, device):
    """Build a fresh trial call from a materialized operation state.

    The fixture owns immutable benchmark parameters and activation producers once;
    every candidate/retained call receives disjoint output capacity.
    """
    from b12x.norm.hyperconnection import _impl

    def make_call(state):
        binding = None
        outputs = ()
        if role not in ("combine", "combine_norm"):
            outputs = {
                name: torch.empty(shape, device=device, dtype=torch.bfloat16)
                for name, shape in state.output_shapes().items()
            }
            binding = state.bind(**outputs, tokens=state.caps.max_tokens)
        source, weight = fixture["state"], fixture["weight"]
        projected = fixture["projected_down"]
        logits, block = fixture["logits"], fixture["block_output"]
        injection = fixture["injection"]
        if role == "grouped_rmsnorm":
            run = lambda: _impl.run_grouped_rmsnorm_impl(
                source, weight, eps=eps, plan=state, out=binding.normalized_capacity
            )
            inputs = dict(state=source, weight=weight)
        elif role == "scaled_silu":
            run = lambda: _impl.run_scaled_silu_impl(
                projected, plan=state, out=binding.bottleneck_capacity
            )
            inputs = dict(projected_down=projected)
        elif role == "gate_mean":
            run = lambda: _impl.run_gate_mean_impl(
                source, logits, plan=state, out=binding.block_input_capacity
            )
            inputs = dict(normalized=source, gate_logits=logits)
        elif role == "combine":
            run = lambda: _impl.run_combine_impl(
                source, block, injection, plan=state
            )
            inputs = dict(state=source, block_output=block, injection_logits=injection)
        else:
            run = lambda: _impl.run_combine_norm_impl(
                source, block, injection, weight, eps=eps, plan=state
            )
            inputs = dict(
                state=source,
                block_output=block,
                injection_logits=injection,
                next_norm_weight=weight,
            )
        options = {"streams": state.caps.streams}
        if role in ("grouped_rmsnorm", "combine_norm"):
            options["eps"] = eps
        if role == "scaled_silu":
            produce, producer_owners = (
                fixture["projected_producer"], fixture["projected_producer_owners"]
            )
        else:
            produce, producer_owners = fixture["producer"], fixture["producer_owners"]
        mutable = (
            source,
            projected,
            logits,
            block,
            injection,
            *(outputs.values() if isinstance(outputs, dict) else ()),
        )
        initial = tuple(value.clone() for value in mutable)

        def restore():
            for value, snapshot in zip(mutable, initial, strict=True):
                value.copy_(snapshot)

        return PreparedCall(
            run=run,
            produce=produce,
            reset=restore,
            restore=restore,
            owners=(
                _Expected(role, inputs, options), binding, fixture, outputs, producer_owners,
            ),
        )

    return make_call


def _hyper_requests(metadata, device, rows):
    from b12x.norm import hyperconnection as op

    hidden, streams, lowrank = (
        int(metadata[key]) for key in ("hidden_size", "hc_count", "hc_lowrank")
    )
    eps = float(metadata.get("rms_norm_eps", 1e-6))
    requests = []
    for m in rows:
        fixture = _hyper_fixture(
            m=m, hidden=hidden, streams=streams, lowrank=lowrank, eps=eps, device=device
        )
        caps = op.Caps(
            device=device, max_tokens=m, hidden_size=hidden, streams=streams, lowrank=lowrank
        )
        for role in (
            "grouped_rmsnorm", "scaled_silu", "gate_mean", "combine", "combine_norm",
        ):
            declaration = op.plan(
                caps,
                invocation=FrozenMapping({"operation": role, "eps": eps}),
            )
            call = _hyper_call_factory(
                role=role, fixture=fixture, eps=eps, device=device
            )
            requests.append(
                declaration.request(
                    name=f"norm.hyperconnection.{role}.m{m}",
                    prepare_call=call,
                    benchmark_call=call,
                    retain_benchmark_call=True,
                )
            )
    return requests


def _mhc_fixture(*, m, hidden, streams, eps, device):
    def rand(shape, dtype=torch.bfloat16):
        return torch.randn(shape, device=device, dtype=dtype) * 0.1

    fixture = dict(
        x=rand((m, hidden)),
        residual=rand((m, streams, hidden)),
        fn=rand((24, streams * hidden), torch.float32),
        fn_pre=rand((24, hidden), torch.float32),
        hc_scale=torch.ones(3, device=device),
        hc_base=rand((24,), torch.float32),
        prev_post=torch.full((m, streams), 0.5, device=device),
        prev_comb=torch.eye(streams, device=device).expand(m, streams, streams).contiguous(),
        norm_weight=torch.ones(hidden, device=device, dtype=torch.bfloat16),
    )
    fixture["produce"], fixture["producer_owners"] = _producer(fixture["x"], eps)
    return fixture


def _mhc_call_factory(*, role, fixture, eps, hc_eps, sinkhorn_iters, device):
    """Bind only the materialized mHC state; no candidate is a serving binding."""
    from b12x.norm.mhc import _impl

    def make_call(state):
        m, hidden = state.query.max_tokens, state.query.hidden_size
        streams = _impl.MHC_MULT
        opts = dict(rms_eps=eps, hc_eps=hc_eps, sinkhorn_iters=sinkhorn_iters)
        if role == "post":
            output = torch.empty((m, streams, hidden), device=device, dtype=torch.bfloat16)
            inputs = {name: fixture[name] for name in ("x", "residual", "prev_post", "prev_comb")}
            run = lambda: _impl._b12x_mhc_post_impl(**inputs, out=output, _state=state)
            binding = None
        else:
            binding = state.bind(
                scratch=_scratch(state, device),
                tokens=m,
                expected_m=m,
                y=torch.empty((m, hidden), device=device, dtype=torch.bfloat16),
                post=torch.empty((m, streams), device=device),
                comb=torch.empty((m, streams, streams), device=device),
                out=torch.empty((m, streams, hidden), device=device, dtype=torch.bfloat16),
            )
            if role == "pre":
                inputs = {name: fixture[name] for name in ("hc_scale", "hc_base")}
                inputs.update(residual=fixture["x"], fn=fixture["fn_pre"])
                run = lambda: _impl._b12x_mhc_pre_impl(
                    **inputs,
                    **opts,
                    norm_weight=fixture["norm_weight"],
                    norm_eps=eps,
                    binding=binding,
                    _state=state,
                )
            else:
                inputs = {
                    name: fixture[name]
                    for name in ("x", "residual", "prev_post", "prev_comb", "fn", "hc_scale", "hc_base")
                }
                run = lambda: _impl._b12x_mhc_post_pre_impl(
                    **inputs,
                    **opts,
                    norm_weight=fixture["norm_weight"],
                    norm_eps=eps,
                    binding=binding,
                    _state=state,
                )
        mutable = tuple(
            value
            for value in (
                fixture["x"],
                fixture["residual"],
                fixture["prev_post"],
                fixture["prev_comb"],
                None if binding is None else binding.y,
                None if binding is None else binding.post_buffer,
                None if binding is None else binding.comb_buffer,
                None if binding is None else binding.out,
                output if role == "post" else None,
            )
            if value is not None
        )
        initial = tuple(value.clone() for value in mutable)

        def restore():
            for value, snapshot in zip(mutable, initial, strict=True):
                value.copy_(snapshot)

        return PreparedCall(
            run=run,
            produce=fixture["produce"],
            reset=restore,
            restore=restore,
            owners=(
                _Expected(
                    "mhc_" + role, inputs,
                    dict(**opts, norm_weight=fixture["norm_weight"]),
                ),
                binding,
                fixture,
                fixture["producer_owners"],
            ),
        )

    return make_call


def _mhc_requests(metadata, device, rows):
    from b12x.norm import mhc as op

    hidden = int(metadata["hidden_size"])
    streams = int(metadata.get("hc_mult", 4))
    if streams != op.MULT:
        raise ValueError("native mHC requires four residual streams")
    eps = float(metadata.get("rms_norm_eps", 1e-5))
    hc_eps = float(metadata.get("hc_eps", 1e-6))
    sinkhorn_iters = int(metadata.get("hc_sinkhorn_iters", 20))
    requests = []
    for m in rows:
        fixture = _mhc_fixture(
            m=m, hidden=hidden, streams=streams, eps=eps, device=device
        )
        caps = op.Caps(
            device=device,
            max_tokens=m,
            hidden_size=hidden,
            split_k=streams * hidden // op.DEFAULT_BLOCK_K,
        )
        for role in ("pre", "post", "post_pre"):
            invocation = {"operation": role, "output_mode": "provided"}
            if role in ("pre", "post_pre"):
                invocation.update(
                    has_norm_weight=True,
                    norm_weight_dtype="bfloat16",
                    rms_eps=eps,
                    hc_eps=hc_eps,
                    sinkhorn_iters=sinkhorn_iters,
                    norm_eps=eps,
                )
            declaration = op.plan(caps, invocation=FrozenMapping(invocation))
            call = _mhc_call_factory(
                role=role,
                fixture=fixture,
                eps=eps,
                hc_eps=hc_eps,
                sinkhorn_iters=sinkhorn_iters,
                device=device,
            )
            requests.append(
                declaration.request(
                    name=f"norm.mhc.{role}.m{m}",
                    prepare_call=call,
                    benchmark_call=call,
                    retain_benchmark_call=True,
                )
            )
    return requests


def make_benchmark_requests(
    metadata: Mapping, *, device: torch.device, rows: tuple[int, ...]
) -> list[PreparationRequest]:
    if "hc_count" in metadata:
        return _hyper_requests(metadata, device, rows)
    if metadata.get("mhc"):
        return _mhc_requests(metadata, device, rows)
    return []


def test_expected(call: PreparedCall):
    """Numerical oracle used exclusively by the external acceptance benchmark."""
    context = next(owner for owner in call.owners if isinstance(owner, _Expected))
    role, x, opts = context.role, context.inputs, context.options
    if not role.startswith("mhc_"):
        from b12x.norm.hyperconnection import reference

        return getattr(reference, role)(**x, **opts)
    if role == "mhc_pre":
        residual = x["residual"].unsqueeze(1).expand(-1, 4, -1)
        fn = x["fn"].repeat(1, 4) / 4
    else:
        residual = (
            x["prev_post"].unsqueeze(-1) * x["x"].unsqueeze(1).float()
            + (x["prev_comb"].unsqueeze(-1) * x["residual"].unsqueeze(2).float()).sum(1)
        ).to(x["x"].dtype)
        if role == "mhc_post":
            return residual
        fn = x["fn"]
    flat = residual.flatten(1).float()
    mixes = torch.nn.functional.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(-1, keepdim=True) + opts["rms_eps"]
    )
    scale, bias, epsilon = x["hc_scale"], x["hc_base"], opts["hc_eps"]
    pre = torch.sigmoid(mixes[:, :4] * scale[0] + bias[:4]) + epsilon
    post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + bias[4:8])
    comb = (
        torch.softmax(mixes[:, 8:].view(-1, 4, 4) * scale[2] + bias[8:].view(4, 4), -1)
        + epsilon
    )
    comb = comb / (comb.sum(-2, keepdim=True) + epsilon)
    for _ in range(opts["sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + epsilon)
        comb = comb / (comb.sum(-2, keepdim=True) + epsilon)
    y = (pre.unsqueeze(-1) * residual.float()).sum(1)
    rms_scale = torch.rsqrt(y.square().mean(-1, keepdim=True) + opts["rms_eps"])
    y = (y.to(residual.dtype).float() * rms_scale * opts["norm_weight"].float()).to(
        residual.dtype
    )
    return residual, post, comb, y
