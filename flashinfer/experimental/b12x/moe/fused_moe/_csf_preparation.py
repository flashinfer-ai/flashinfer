"""Prepare lossless scale blocks for ordinary native expert kernels."""

from dataclasses import replace

import numpy as np
import torch

from b12x._lib.quant.csf_blocks import Nvfp4CompactBlockPlane, upload_csf_block_payload
from b12x._lib.quant.nvfp4_csf import Nvfp4CsfDecoder
from b12x.loader._csf_blocks import (
    concatenate_csf_blocks,
    logical_csf_components,
    slice_csf_blocks,
    transform_csf_blocks,
)
from .weights import CsfBlockPlanes, Nvfp4CsfWeights, PackedWeights


def _logical(plane):
    bases, fields, positions, values = logical_csf_components(plane)
    grid = bases[:, None] + fields
    grid.ravel()[positions] = values
    return grid


def _rotate(plane, rotation):
    if not rotation:
        return plane
    return concatenate_csf_blocks(
        (
            slice_csf_blocks(plane, row_range=(rotation, plane.rows)),
            slice_csf_blocks(plane, row_range=(0, rotation)),
        )
    )


def _fill_source_scales(planes, output, nvfp4):
    for expert, plane in enumerate(planes):
        logical = _logical(plane)
        if nvfp4:
            rows, columns = output.shape[1:]
            grid = np.zeros((rows, columns), dtype=np.uint8)
            grid[: plane.rows, : plane.columns] = logical
            logical = (
                grid.reshape(rows // 128, 4, 32, columns // 4, 4)
                .transpose(0, 3, 2, 1, 4)
                .copy()
                .reshape(rows, columns)
            )
        output[expert].view(torch.uint8).copy_(torch.from_numpy(logical))


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
    if not isinstance(weights.w2_scales, CsfBlockPlanes):
        raise TypeError("Both expert projections must use CSF block scale planes")
    geometry = plan.geometry
    e, h, n = geometry.num_experts, geometry.hidden_size, geometry.intermediate_size
    quantization = "nvfp4" if nvfp4 else "mxfp4"
    group_size = 16 if nvfp4 else 32
    sources = (weights.w13_scales.planes, weights.w2_scales.planes)
    for source, rows, columns in zip(
        sources, (2 * n, h), (h // group_size, n // group_size), strict=True
    ):
        if len(source) != e or any(
            (p.rows, p.columns, p.quantization) != (rows, columns, quantization)
            for p in source
        ):
            raise ValueError(
                "CSF block geometry must match its native expert weight plan"
            )
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
    for source, output in zip(sources, source_outputs, strict=True):
        _fill_source_scales(source, output, nvfp4)
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
            .cpu()
            .numpy()
            for s in source_outputs
        )
    elif not nvfp4:
        table = np.minimum(np.arange(256), 247).astype(np.uint8)
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
        Nvfp4CompactBlockPlane(
            upload_csf_block_payload(
                tuple(
                    _rotate(
                        transform_csf_blocks(p, table) if table is not None else p, rot
                    )
                    for p in source
                ),
                outputs[0].device,
                native_codes=True,
                group_record_bytes=3,
                destination=destination,
                group_rows=group if destination in (2, 3) else None,
            ),
            layout=int(a16),
        )
        for source, table, rot, group in zip(
            sources, tables, (rotation, 0), (n, h), strict=True
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
