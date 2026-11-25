# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""cuDNN graphs for batch-invariant GEMMs."""

import functools
import inspect
import threading
from collections import OrderedDict

import torch

from .. import gemm_base
from ..gemm_base import UIDs
from .api import (
    CLASS,
    _binding_layout,
    _check_plan_options,
    _require_disjoint,
    prepare_bias,
    prepare_operands,
)

_BF16_UIDS = (UIDs.A_UID.value, UIDs.B_UID.value, UIDs.O_UID.value)
_BIASED_BF16_UIDS = (*_BF16_UIDS, UIDs.BIAS_UID.value)
_BF16_LAYOUTS: OrderedDict[tuple, tuple] = OrderedDict()
_BF16_LAYOUT_LOCK = threading.Lock()


def clear_bf16_layouts():
    with _BF16_LAYOUT_LOCK:
        _BF16_LAYOUTS.clear()


def bf16_gemm(a, b, out, bias, config, chunks, workspace):
    # Cache only validated metadata and the arguments to the existing graph LRU.
    # Pointers, tensor ownership and overlap checks remain per execution.
    try:
        key = (
            _binding_layout(a),
            _binding_layout(b),
            None if bias is None else _binding_layout(bias),
            None if out is None else _binding_layout(out),
        )
        geometry = _BF16_LAYOUTS.get(key)
    except (AttributeError, RuntimeError, TypeError):
        key = geometry = None

    original_a = None
    if geometry is None:
        input_a, input_out = a, out
        a, out, _, m, n, k = prepare_operands(a, b, out)
        bias = prepare_bias(bias, n, a)
        if m == 0:
            return out
        if a.stride(0) < k:
            a = a.as_strided((1, k), (k, 1)) if m == 1 else a.contiguous()
        if a is not input_a and (input_out is not None or workspace is not None):
            original_a = input_a
        geometry = (
            a.shape,
            a.stride(),
            b.shape,
            b.stride(),
            a.device,
            () if bias is None else bias.shape,
            () if bias is None else bias.stride(),
        )
        # Inputs which required a copy or a normalized view must be prepared
        # again on every call. They still share the same compiled graph.
        if key is not None and a is input_a and (input_out is None or out is input_out):
            with _BF16_LAYOUT_LOCK:
                _BF16_LAYOUTS[key] = geometry
                if len(_BF16_LAYOUTS) > 512:
                    _BF16_LAYOUTS.popitem(last=False)

    a_shape, a_stride, b_shape, b_stride, device, bias_shape, bias_stride = geometry
    if out is None:
        out = torch.empty((a_shape[0], b_shape[1]), dtype=torch.bfloat16, device=device)
    if config is not None or chunks is not None:
        _check_plan_options(CLASS, config, chunks, b_shape[1], b_shape[0], "bf16")
    gemm_base._require_cudnn_batch_invariant(CLASS)
    with gemm_base._batch_invariant_device(device):
        try:
            graph = gemm_base.build_cudnn_gemm_bf16_graph(
                a_shape,
                a_stride,
                b_shape,
                b_stride,
                gemm_base.cudnn.data_type.BFLOAT16,
                device,
                bias is not None,
                bias_shape,
                bias_stride,
                batch_invariant=True,
            )
        except gemm_base.cudnn.cudnnGraphNotSupportedError as exc:
            raise gemm_base._batch_invariant_plan_error(CLASS, exc) from exc
        buffers = (a, b, out) if bias is None else (a, b, out, bias)
        uids = _BF16_UIDS if bias is None else _BIASED_BF16_UIDS
        try:
            return gemm_base._execute_cudnn_batch_invariant(
                graph, buffers, uids, out, workspace, CLASS, original_a
            )
        except ValueError as exc:
            if "disjoint" not in str(exc):
                raise
            writes = [(out, "the output")]
            if workspace is not None and graph._fi_bi_workspace_size:
                writes.append((workspace, "the workspace"))
            reads = [(a, "A"), (b, "B")]
            if bias is not None:
                reads.append((bias, "bias"))
            _require_disjoint(CLASS, writes, reads)
            raise


def _layout(t):
    return tuple(t.shape), t.stride(), t.dtype


_QUANTIZED_UIDS = (
    UIDs.A_UID.value,
    UIDs.B_UID.value,
    UIDs.A_SCALE_UID.value,
    UIDs.B_SCALE_UID.value,
    UIDs.O_UID.value,
)
_QUANTIZED_ALPHA_UIDS = (*_QUANTIZED_UIDS, UIDs.ALPHA_UID.value)


@functools.lru_cache(maxsize=None)
def _unit_alpha(device):
    # The blocking host-to-device copy publishes a fully initialized scalar
    # for subsequent eager calls and captures on every stream of this device.
    return torch.tensor([1.0], dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=2048)
def build_quantized_graph(
    device,
    a_layout,
    b_layout,
    sa_layout,
    sb_layout,
    out_layout,
    block_size,
    enable_pdl,
):
    """Cache a static graph for the device and complete operand layouts."""
    cudnn = gemm_base.cudnn
    fp32 = cudnn.data_type.FLOAT
    g = cudnn.pygraph(compute_data_type=fp32, intermediate_data_type=fp32)
    packed_types = (torch.uint8, getattr(torch, "float4_e2m1fn_x2", None))
    has_alpha = b_layout[2] in packed_types

    def operand(uid, layout):
        shape, stride, dtype = layout
        if dtype in packed_types:
            # Packed FP4 strides in torch are in bytes, in cuDNN in elements.
            shape, stride = gemm_base._real_fp4_shape_stride(shape, stride)
            shape, stride = shape[-2:], stride[-2:]
            cu_type = cudnn.data_type.FP4_E2M1
        else:
            cu_type = gemm_base._torch_data_type_to_cudnn_data_type(dtype)
        return g.tensor(dim=shape, stride=stride, data_type=cu_type, uid=uid.value)

    def scale(uid, layout):
        if block_size in (16, 32):
            m, n = out_layout[0][-2:]
            k = a_layout[0][-1] * (2 if a_layout[2] in packed_types else 1)
            rows = m if uid == UIDs.A_SCALE_UID else n
            padded_rows = (rows + 127) // 128 * 128
            padded_blocks = (k // block_size + 3) // 4 * 4
            shape = (padded_rows, padded_blocks)
            stride = (padded_blocks, 1)
            if uid == UIDs.B_SCALE_UID:
                shape, stride = shape[::-1], stride[::-1]
            cu_type = (
                cudnn.data_type.FP8_E4M3
                if block_size == 16
                else cudnn.data_type.FP8_E8M0
            )
            return g.tensor(
                dim=shape,
                stride=stride,
                data_type=cu_type,
                reordering_type=cudnn.tensor_reordering.F8_128x4,
                uid=uid.value,
            )
        shape, stride, _ = layout
        return g.tensor(dim=shape, stride=stride, data_type=fp32, uid=uid.value)

    a, b = operand(UIDs.A_UID, a_layout), operand(UIDs.B_UID, b_layout)
    sa, sb = scale(UIDs.A_SCALE_UID, sa_layout), scale(UIDs.B_SCALE_UID, sb_layout)
    if block_size:
        # Groupwise B has one FP32 scale per 128 x 128 block. MX and NV
        # weights each have one swizzled scale per K block and output column.
        n_block = 128 if block_size == 128 else 1
        a = g.block_scale_dequantize(a, sa, block_size=[1, block_size]).set_data_type(
            fp32
        )
        b = g.block_scale_dequantize(
            b, sb, block_size=[block_size, n_block]
        ).set_data_type(fp32)
    if enable_pdl and "enable_pdl" not in inspect.signature(g.matmul).parameters:
        raise cudnn.cudnnGraphNotSupportedError(
            "the installed cuDNN Frontend cannot request PDL for a batch-invariant GEMM"
        )
    mm_kwargs = {"enable_pdl": True} if enable_pdl else {}
    y = g.matmul(A=a, B=b, **mm_kwargs).set_data_type(fp32)
    if not block_size:
        # The FP8 per-tensor class forms the FP32 scale product before
        # multiplying the accumulator and rounding to the output dtype.
        alpha = g.mul(a=sa, b=sb).set_data_type(fp32)
        y = g.mul(a=y, b=alpha)
    elif has_alpha:
        alpha = g.tensor(dim=[1], stride=[1], data_type=fp32, uid=UIDs.ALPHA_UID.value)
        y = g.mul(a=y, b=alpha)
    shape, stride, dtype = out_layout
    y.set_dim(shape).set_stride(stride).set_output(True).set_data_type(
        gemm_base._torch_data_type_to_cudnn_data_type(dtype)
    ).set_uid(UIDs.O_UID.value)
    g.select_numeric_notes([cudnn.numerical_note.BATCH_INVARIANT])
    gemm_base._finalize_cudnn_graph_for_tactic(g, -1, [cudnn.heur_mode.A])
    g._fi_bi_workspace_size = g.get_workspace_size()
    g._fi_bi_default_alpha = _unit_alpha(device) if has_alpha else None
    return g


def quantized_gemm(
    cls,
    a,
    b,
    sa,
    sb,
    out,
    out_fresh,
    *,
    caller_a,
    block_size=0,
    alpha=None,
    enable_pdl=False,
    workspace=None,
    oversized_scales=False,
):
    """Execute the FE graph using the public entry's validated tensors."""
    gemm_base._require_cudnn_batch_invariant(cls)

    if a.stride(-2) < a.size(-1):
        # Expanded or overlapping A rows have the same scalar values when
        # materialized. Give the GEMM a complete row pitch for the TMA load.
        if a.size(-2) == 1:
            stride = list(a.stride())
            stride[-2] = a.size(-1)
            a = a.as_strided(a.shape, stride)
        else:
            a = a.contiguous()
    # Both operand preparation and row materialization may replace A before
    # cuDNN sees it. Caller-provided writes must also be disjoint from its source.
    original_a = (
        caller_a
        if a is not caller_a and (not out_fresh or workspace is not None)
        else None
    )
    swizzled = block_size in (16, 32)
    with gemm_base._batch_invariant_device(a.device):
        try:
            graph = build_quantized_graph(
                a.device,
                _layout(a),
                _layout(b),
                None if swizzled else _layout(sa),
                None if swizzled else _layout(sb),
                _layout(out),
                block_size,
                enable_pdl,
            )
        except gemm_base.cudnn.cudnnGraphNotSupportedError as exc:
            raise gemm_base._batch_invariant_plan_error(cls, exc) from exc
        # The cuDNN tensor descriptors specify the FP4 pairs and scale format
        # stored in these byte tensors. Bind their original buffers directly.
        buffers = (a, b, sa, sb, out)
        uids = _QUANTIZED_UIDS
        if graph._fi_bi_default_alpha is not None:
            if alpha is None:
                alpha = graph._fi_bi_default_alpha
            buffers += (alpha,)
            uids = _QUANTIZED_ALPHA_UIDS
        try:
            if oversized_scales and (not out_fresh or workspace is not None):
                return gemm_base._execute_cudnn_batch_invariant(
                    graph,
                    buffers,
                    uids,
                    out,
                    workspace,
                    cls,
                    original_a,
                    extra_reads=((sa, "A's scales"), (sb, "B's scales")),
                )
            return gemm_base._execute_cudnn_batch_invariant(
                graph, buffers, uids, out, workspace, cls, original_a
            )
        except ValueError as exc:
            if "disjoint" not in str(exc):
                raise
            writes = [(out, "the output")]
            if workspace is not None and graph._fi_bi_workspace_size:
                writes.append((workspace, "the workspace"))
            if not block_size:
                sa_name, sb_name = "a scale", "a scale"
            elif block_size == 128:
                sa_name, sb_name = "a_scale", "b_scale"
            else:
                sa_name, sb_name = "A's scales", "B's scales"
            reads = [(a, "A"), (b, "B"), (sa, sa_name), (sb, sb_name)]
            if alpha is not None:
                reads.append((alpha, "alpha"))
            _require_disjoint(cls, writes, reads)
            raise
