"""Prepare lossless scale blocks for ordinary native expert kernels."""

from dataclasses import replace

import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane
from b12x._lib.quant._csf_prepare import (
    expand_source_scales,
    prepare_csf_payload,
    validate_source,
)
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x.loader._csf_source import CsfSource, upload_csf_planes
from .weights import CsfBlockPlanes, Mxfp4CsfWeights, Nvfp4CsfWeights, PackedWeights


def prepare_csf_block_weights(plan, weights):
    """Apply native preparation once and retain only compressed resident scales."""
    from .planning import ActivationMode, prepare_weights

    nvfp4 = isinstance(weights, Nvfp4CsfWeights)
    mode = plan.activation.mode
    if mode not in (
        (ActivationMode.A4, ActivationMode.A16)
        if nvfp4
        else (ActivationMode.A8, ActivationMode.A16)
    ):
        raise ValueError("CSF scales require native FP4 expert activation modes")
    if plan.activation.a16_max_tokens and mode is not ActivationMode.A16:
        raise ValueError(
            "CSF expansion requires a uniform native scale layout per weight plan"
        )
    geometry = plan.geometry
    e, h, n = geometry.num_experts, geometry.hidden_size, geometry.intermediate_size
    quantization = "nvfp4" if nvfp4 else "mxfp4"
    group_size = 16 if nvfp4 else 32
    device = weights.packed.w13.device if nvfp4 else weights.w13.device
    sources = tuple(
        upload_csf_planes(source.planes, device)
        if isinstance(source, CsfBlockPlanes)
        else source.to(device)
        for source in (weights.w13_scales, weights.w2_scales)
    )
    for source, rows, columns in zip(
        sources, (2 * n, h), (h // group_size, n // group_size), strict=True
    ):
        if not isinstance(source, CsfSource) or (
            source.num_experts,
            source.rows,
            source.columns,
            source.quantization,
        ) != (e, rows, columns, quantization):
            raise ValueError(
                "CSF block geometry must match its native expert weight plan"
            )
    for source in sources:
        validate_source(source)
    buffers = (
        (weights.packed.w13_block_scales, weights.packed.w2_block_scales)
        if nvfp4
        else (weights.w13_scale_scratch, weights.w2_scale_scratch)
    )
    if nvfp4:
        native_weights = weights.packed
        source_outputs = buffers
    else:
        source_outputs = tuple(
            buffer.view(torch.uint8)
            .reshape(-1)[: e * rows * columns]
            .reshape(e, rows, columns)
            for buffer, rows, columns in zip(
                buffers, (2 * n, h), (h // 32, n // 32), strict=True
            )
        )
        unit = torch.ones(e, dtype=torch.float32, device=weights.w13.device)
        native_weights = PackedWeights(
            w13=weights.w13,
            w2=weights.w2,
            w13_block_scales=source_outputs[0],
            w2_block_scales=source_outputs[1],
            w13_global_scales=unit,
            w2_global_scales=unit,
        )
    initial_planes = None
    if nvfp4 and mode is ActivationMode.A4:
        initial_planes = tuple(
            Nvfp4CompactBlockPlane(prepare_csf_payload(source)) for source in sources
        )
        Nvfp4CsfDecoder.prepare(*initial_planes, *source_outputs).decode_all(
            *source_outputs
        )
    else:
        for source, output in zip(sources, source_outputs, strict=True):
            expand_source_scales(source, output, swizzled=nvfp4)
    tables = (None, None)
    if nvfp4 and mode is ActivationMode.A16:
        from b12x.moe._shared.kernels.w4a16.prepare import (
            _nvfp4_compute_scale_factor,
            _process_nvfp4_packed_scales,
        )

        alphabet = torch.arange(
            256, dtype=torch.uint8, device=native_weights.w13.device
        ).view(torch.float8_e4m3fn)
        alphabet = (
            alphabet[:, None].expand(256, 4).contiguous().to(plan.activation.io_dtype)
        )
        tables = tuple(
            _process_nvfp4_packed_scales(
                alphabet,
                scale_factor=_nvfp4_compute_scale_factor(s, plan.activation.io_dtype),
            )
            .view(torch.uint8)[:, 0]
            .contiguous()
            for s in source_outputs
        )
    elif not nvfp4:
        table = torch.arange(256, device=device).clamp_max_(247).to(torch.uint8)
        tables = (table, table)
    prepared = prepare_weights(plan=plan, weights=native_weights)
    rep = prepared._impl.representation
    native = None if rep is None else rep.value
    a16 = mode is ActivationMode.A16
    destination = 1 if a16 else 0 if nvfp4 else 3 if n % 128 == 64 else 2
    rotation = n if (plan.source.w13_layout.value == ("w13" if a16 else "w31")) else 0
    if (
        a16
        and not nvfp4
        and getattr(native, "micro_w13_scale", None) is not None
        and rotation
    ):
        raise ValueError(
            "Source-native MXFP4 CSF requires gate/up weight order for one shared scale grid"
        )
    targets = (
        (native.w13_scale, native.w2_scale)
        if a16
        else (prepared._impl.w1_blockscale, prepared._impl.w2_blockscale)
    )
    outputs = []
    for buffer, target in zip(buffers, targets, strict=True):
        if (
            not buffer.is_contiguous()
            or buffer.device != target.device
            or buffer.numel() * buffer.element_size()
            < target.numel() * target.element_size()
        ):
            raise ValueError(
                "Caller-owned CSF scale scratch must cover the prepared native layout"
            )
        output = (
            buffer.view(torch.uint8)
            .reshape(-1)[: target.numel() * target.element_size()]
            .view(target.dtype)
            .view_as(target)
        )
        output.copy_(target)
        outputs.append(output)
    if a16:
        replacements = dict(w13_scale=outputs[0], w2_scale=outputs[1])
        if getattr(native, "micro_w13_scale", None) is not None:
            replacements.update(micro_w13_scale=outputs[0], micro_w2_scale=outputs[1])
        native = replace(native, **replacements)
    elif not nvfp4:
        native = replace(native, w13_sfb=outputs[0], w2_sfb=outputs[1])
    planes = tuple(
        initial_planes[index]
        if initial_planes is not None and not rot
        else Nvfp4CompactBlockPlane(
            prepare_csf_payload(
                source,
                destination=destination,
                rotation=rot,
                value_table=table,
                group_rows=group if destination in (2, 3) else None,
            ),
            layout=int(a16),
        )
        for index, (source, table, rot, group) in enumerate(
            zip(sources, tables, (rotation, 0), (n, h), strict=True)
        )
    )
    decoder = Nvfp4CsfDecoder.prepare(*planes, *outputs)
    return replace(
        prepared,
        _impl=replace(
            prepared._impl,
            representation=None if rep is None else replace(rep, value=native),
            w1_blockscale=outputs[0],
            w2_blockscale=outputs[1],
            nvfp4_csf=decoder,
        ),
    )


def prepare_online_csf_weights(plan, weights, scale_scratch):
    """Compress native expert block scales losslessly, then prepare them as CSF.

    The GPU encoder writes the slab checkpoint encoding, so the experts follow
    the same preparation, expansion plans, and kernels as compressed checkpoint
    scales. ``scale_scratch`` supplies the two caller-owned expansion buffers;
    the prepared experts no longer reference the native block-scale tensors.
    """
    from b12x._lib.quant.csf_encode import encode_csf_source
    from ._impl import _unswizzle_block_scale_bytes

    if not isinstance(weights, PackedWeights):
        raise TypeError("online CSF compression requires native PackedWeights")
    if scale_scratch is None or len(scale_scratch) != 2:
        raise ValueError("online CSF compression requires two scale_scratch buffers")
    nvfp4 = plan.source.format.value == "modelopt_nvfp4"
    geometry = plan.geometry
    e, h, n = geometry.num_experts, geometry.hidden_size, geometry.intermediate_size
    group = 16 if nvfp4 else 32
    native = (weights.w13_block_scales, weights.w2_block_scales)
    owners = {t.untyped_storage().data_ptr() for t in native}
    if (
        any(
            not isinstance(t, torch.Tensor)
            or t.device != weights.w13.device
            or not t.is_contiguous()
            for t in scale_scratch
        )
        or len({t.untyped_storage().data_ptr() for t in scale_scratch}) != 2
        or owners & {t.untyped_storage().data_ptr() for t in scale_scratch}
    ):
        raise ValueError(
            "CSF scale scratch must be two contiguous device buffers separate from the native scales"
        )
    if not nvfp4 and any(
        t is not None and not bool(torch.all(t == 1))
        for t in (weights.w13_global_scales, weights.w2_global_scales)
    ):
        raise ValueError("MXFP4-CSF requires unit global weight scales")
    sources = []
    for scales, rows, columns in zip(
        native, (2 * n, h), (h // group, n // group), strict=True
    ):
        logical = (
            _unswizzle_block_scale_bytes(scales, rows, columns)
            if nvfp4
            else scales.view(torch.uint8).reshape(e, rows, columns).contiguous()
        )
        sources.append(
            encode_csf_source(logical, quantization="nvfp4" if nvfp4 else "mxfp4")
        )
        del logical
    inner = replace(plan, scale_compression=None)
    if nvfp4:
        compressed = Nvfp4CsfWeights(
            packed=replace(
                weights,
                w13_block_scales=scale_scratch[0],
                w2_block_scales=scale_scratch[1],
            ),
            w13_scales=sources[0],
            w2_scales=sources[1],
        )
    else:
        compressed = Mxfp4CsfWeights(
            weights.w13, weights.w2, sources[0], sources[1], *scale_scratch
        )
    prepared = prepare_csf_block_weights(inner, compressed)
    return replace(prepared, plan=replace(prepared.plan, scale_compression="csf"))
