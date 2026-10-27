"""Compilation boundaries for mHC with fully caller-owned storage."""

import torch

from b12x.preparation.types import plan_from_handle, require_prepared

_BUFFER_FIELDS = ("partials", "y", "post_buffer", "comb_buffer", "out")


@torch.library.custom_op("b12x::mhc_bound", mutates_args=("buffers",))
def _bound(
    inputs: list[torch.Tensor],
    buffers: list[torch.Tensor],
    norm_weight: torch.Tensor | None,
    pre_mix: torch.Tensor | None,
    fn_bf16: torch.Tensor | None,
    operation: str,
    plan_handle: int,
    expected_m: int | None,
    rms_eps: float,
    hc_eps: float,
    sinkhorn_iters: int,
    norm_eps: float,
) -> None:
    from . import _impl

    plan = plan_from_handle(plan_handle)
    state = require_prepared(plan, "norm.mhc")
    binding = _impl.B12XMHCBinding(
        state=state,
        plan=plan,
        expected_m=expected_m,
        pre_out=buffers[5] if len(buffers) == 6 else None,
        **dict(zip(_BUFFER_FIELDS, buffers[:5], strict=True)),
    )
    options = dict(
        binding=binding,
        _state=state,
        norm_weight=norm_weight,
        pre_mix=pre_mix,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        sinkhorn_iters=sinkhorn_iters,
        norm_eps=norm_eps,
    )
    if operation == "pre":
        _impl._b12x_mhc_pre_impl(*inputs, **options)
    elif operation == "post_pre":
        _impl._b12x_mhc_post_pre_impl(*inputs, fn_bf16=fn_bf16, **options)
    else:
        raise ValueError("bound mHC requires pre or post_pre")


@_bound.register_fake
def _fake_bound(
    inputs,
    buffers,
    norm_weight,
    pre_mix,
    fn_bf16,
    operation,
    plan_handle,
    expected_m,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    norm_eps,
):
    pass


def complete(binding, pre_mix):
    return (
        binding is not None
        and all(getattr(binding, field) is not None for field in _BUFFER_FIELDS)
        and (pre_mix is None or binding.pre_out is not None)
    )


def run(
    binding,
    inputs,
    *,
    operation,
    norm_weight,
    pre_mix,
    fn_bf16=None,
    rms_eps,
    hc_eps,
    sinkhorn_iters,
    norm_eps,
    **outputs,
):
    if binding.state is not require_prepared(binding.plan, "norm.mhc"):
        raise ValueError("binding and plan have different prepared states")
    extras = [name for name, value in outputs.items() if value is not None]
    if extras:
        raise ValueError(
            "mHC binding owns scratch and output buffers; do not also pass "
            + ", ".join(extras)
        )
    buffers = [getattr(binding, field) for field in _BUFFER_FIELDS]
    if binding.pre_out is not None:
        buffers.append(binding.pre_out)
    _bound(
        list(inputs),
        buffers,
        norm_weight,
        pre_mix,
        fn_bf16,
        operation,
        binding.plan.handle,
        binding.expected_m,
        float(rms_eps),
        float(hc_eps),
        int(sinkhorn_iters),
        float(norm_eps),
    )
    tokens = inputs[0 if operation == "pre" else 1].shape[0]
    return tuple(
        value if value.shape[0] == tokens else value[:tokens]
        for value in (binding.out, binding.post_buffer, binding.comb_buffer, binding.y)
    )


@torch.library.custom_op("b12x::mhc_post_out", mutates_args=("out",))
def post(
    x: torch.Tensor,
    residual: torch.Tensor,
    prev_post: torch.Tensor,
    prev_comb: torch.Tensor,
    out: torch.Tensor,
    plan_handle: int,
) -> None:
    from . import _impl

    _impl._b12x_mhc_post_impl(
        x,
        residual,
        prev_post,
        prev_comb,
        out=out,
        _state=_impl._prepared_state(
            plan_from_handle(plan_handle), output_mode="provided"
        ),
    )


@post.register_fake
def _fake_post(x, residual, prev_post, prev_comb, out, plan_handle):
    pass
