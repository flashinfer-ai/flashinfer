# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Declaration-only SM80 native addressing contract shared by graph and adapters."""


def data_layout_mismatch(name, shape, stride, *, thd=False):
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok

    shape, stride = tuple(shape), tuple(stride)
    if shape[-1] % 8:
        return f"SM80 {name}: head dimension must be a multiple of 8; got {shape[-1]}"
    # THD has its own packed-layout validator. Its batch stride is unread:
    # every sequence origin comes from device prefixes or physical offsets.
    if not thd and not dense_layout_ok(shape, stride):
        return f"SM80 {name}: native storage requires non-overlapping strides and contiguous head dimension; got stride {stride}"
    axes = (1, 2) if thd else (0, 1, 2)
    if any((shape[i] > 1 or (thd and i == 2)) and stride[i] % 8 for i in axes):
        return f"SM80 {name}: native storage requires stepped strides to be multiples of 8 elements; got stride {stride}"
    return None


def forward_thd_layout_mismatch(name, shape, stride):
    """Packed token-major rows may have independent head and token padding."""
    reason = data_layout_mismatch(name, shape, stride, thd=True)
    if reason is not None:
        return reason
    _, heads, _, width = map(int, shape)
    _, head_stride, token_stride, inner_stride = map(int, stride)
    # Batch is never stepped. A singleton head's pitch is also unread.
    row_span = (heads - 1) * head_stride + width
    if inner_stride != 1 or (heads > 1 and head_stride < width) or token_stride < row_span:
        return f"SM80 THD {name}: native packed rows require non-overlapping token/head strides and contiguous head dimension; got stride {tuple(stride)}"
    return None


def forward_adapter_mismatch(api):
    if api._rope_max_s:
        return "SM80 RoPE forward requires staging and is not supported by the native adapter"
    for role in ("q", "k", "v", "o"):
        desc = getattr(api, role + "_desc")
        reason = forward_thd_layout_mismatch(role, desc.shape, desc.stride) if api.thd else data_layout_mismatch(role, desc.shape, desc.stride)
        if reason is not None:
            return reason
    return None


def graph_mismatch(facts, _version=None):
    """No tensor-framework imports, allocation, compilation or graph mutation."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import is_contiguous

    ports = (
        facts.port_layouts
        if facts.is_backward
        else tuple(
            (name, tensor.get_dim(), tensor.get_stride())
            for name, tensor in (("q", facts.q_t), ("k", facts.k_t), ("v", facts.v_t), ("o", facts.o_t))
            if tensor is not None
        )
    )
    for name, shape, stride in ports:
        reason = data_layout_mismatch(name, shape, stride, thd=facts.thd)
        if reason is not None:
            return reason
    if facts.is_backward:
        if facts.has_bias != facts.has_dbias:
            return "SM80 bias and dbias must both be declared for native backward"
        if facts.has_sink != facts.has_dsink:
            return "SM80 sink and dsink must both be declared for native backward"
        for name in ("bias", "dbias", "sink", "dsink"):
            tensor = getattr(facts, name + "_t")
            if tensor is not None and not is_contiguous(tensor.get_dim(), tensor.get_stride()):
                return f"SM80 {name}: native backward requires contiguous storage; got stride {tuple(tensor.get_stride())}"
    return None
