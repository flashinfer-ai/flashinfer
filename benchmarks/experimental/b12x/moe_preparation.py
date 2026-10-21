"""Prepared-call adapters for native fused-MoE benchmarks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import torch

from b12x.preparation import PreparedCall
from b12x.preparation.types import require_prepared


def encode_csf_32x2_source(scales: torch.Tensor, *, quantization: str):
    """Encode native scale bytes and export the csf_32x2_v1 checkpoint planes.

    CPU indexing and transfers happen during benchmark setup, before preparation
    and timing. The exporter preserves the scale bytes without requantization.
    """
    from b12x._lib.quant.csf_encode import encode_csf_source
    from b12x.loader._csf_blocks import export_csf_blocks
    from b12x.loader._csf_source import upload_csf_planes

    source = encode_csf_source(scales, quantization=quantization)
    fixed, exceptions = (t.cpu().numpy() for t in source.components)
    starts = source.starts.cpu().tolist()
    planes = tuple(
        export_csf_blocks(
            fixed[index],
            exceptions[first:last],
            rows=source.rows,
            columns=source.columns,
            quantization=quantization,
        )
        for index, (first, last) in enumerate(zip(starts[:-1], starts[1:], strict=True))
    )
    return upload_csf_planes(planes, scales.device)


def prepare_csf_32x2_weights(*, plan, weights, scale_scratch):
    """Prepare expert weights from csf_32x2_v1 sources and caller-owned scratch."""
    from b12x.moe import fused_moe as moe
    from b12x.moe.fused_moe._impl import _unswizzle_block_scale_bytes

    if plan.scale_compression != "csf":
        raise ValueError("csf_32x2_v1 requires a CSF-compatible weight plan")
    nvfp4 = plan.source.format.value == "modelopt_nvfp4"
    geometry = plan.geometry
    e, h, n = geometry.num_experts, geometry.hidden_size, geometry.intermediate_size
    group = 16 if nvfp4 else 32
    sources = tuple(
        encode_csf_32x2_source(
            _unswizzle_block_scale_bytes(scales, rows, columns)
            if nvfp4
            else scales.view(torch.uint8).reshape(e, rows, columns).contiguous(),
            quantization="nvfp4" if nvfp4 else "mxfp4",
        )
        for scales, rows, columns in zip(
            (weights.w13_block_scales, weights.w2_block_scales),
            (2 * n, h),
            (h // group, n // group),
            strict=True,
        )
    )
    if nvfp4:
        compressed = moe.Nvfp4CsfWeights(
            packed=replace(
                weights,
                w13_block_scales=scale_scratch[0],
                w2_block_scales=scale_scratch[1],
            ),
            w13_scales=sources[0],
            w2_scales=sources[1],
        )
    else:
        if any(
            scale is not None and not bool(torch.all(scale == 1))
            for scale in (weights.w13_global_scales, weights.w2_global_scales)
        ):
            raise ValueError("MXFP4-CSF requires unit global weight scales")
        compressed = moe.Mxfp4CsfWeights(
            weights.w13, weights.w2, *sources, *scale_scratch
        )
    prepared = moe.prepare_weights(
        plan=replace(plan, scale_compression=None), weights=compressed
    )
    return replace(prepared, plan=replace(prepared.plan, scale_compression="csf"))


def prepared_call(
    *, output: object, bind: Callable[[object, object], object]
) -> Callable[[object], PreparedCall]:
    """Bind one benchmark invocation with the session-owned exact-M scratch."""

    def factory(state: object) -> PreparedCall:
        scratch = tuple(
            torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
            for spec in state.scratch.scratch_specs()
        )
        binding = bind(state, scratch)
        return PreparedCall(run=binding.run, output=output, owners=(binding, scratch))

    return factory


def scratch_for(plan: object) -> tuple[torch.Tensor, ...]:
    """Allocate caller-owned scratch for one prepared exact-M plan."""
    state = require_prepared(plan, plan.component_id)
    return tuple(
        torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        for spec in state.scratch.scratch_specs()
    )


def request_for_capacity(
    declaration: object,
    *,
    name: str,
    calls: dict[int, Callable[[object], PreparedCall]],
    benchmark_calls: dict[int, Callable[[object], PreparedCall]] | None = None,
    dependencies: tuple[str, ...] = (),
):
    """Make the scalar or exact-M composite request without a legacy warmup path."""
    counts = getattr(declaration, "token_counts", None)
    if benchmark_calls is None:
        benchmark_calls = calls
    if counts is None:
        if len(calls) != 1:
            raise ValueError("scalar MoE declaration requires one prepared call")
        return declaration.request(
            name=name,
            prepare_call=next(iter(calls.values())),
            benchmark_call=next(iter(benchmark_calls.values())),
            dependencies=dependencies,
        )
    return declaration.request(
        name=name,
        prepare_calls=calls,
        benchmark_calls=benchmark_calls,
        dependencies=dependencies,
    )
