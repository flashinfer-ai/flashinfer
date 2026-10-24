# SPDX-License-Identifier: Apache-2.0
"""Public-call adapter; unsupported calls retain the official implementation.

Fast calls use native finite FP32 log gates and matching immutable CPU sequence
metadata. Linear/raw gates, checkpoints and non-FP32 state use the official path.
"""

from typing import Any

import torch

_plans: dict[tuple, Any] = {}


def run(q, k, v, **kwargs):
    from flashinfer.gdn_prefill import chunk_gated_delta_rule as official

    fallback = lambda: official(q, k, v, backend="flashinfer", **kwargs)
    g, beta = kwargs["g"], kwargs["beta"]
    cu = kwargs["cu_seqlens"]
    initial, final = kwargs["initial_state"], kwargs["output_state"]
    if (
        q.ndim != 3
        or k.shape != q.shape
        or v.ndim != 3
        or q.shape[0] != v.shape[0]
        or q.shape[-1] != 128
        or v.shape[-1] != 128
        or q.shape[1] == 0
        or v.shape[1] % q.shape[1]
        or q.dtype not in (torch.bfloat16, torch.float16)
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or not q.is_cuda
        or any(not x.is_contiguous() or x.device != q.device for x in (q, k, v))
        or kwargs["gate_domain"] != "log"
        or kwargs["use_gate_in_kernel"]
        or kwargs["beta_is_logit"]
        or kwargs["A_log"] is not None
        or kwargs["dt_bias"] is not None
        or g is None
        or beta is None
        or g.dtype != torch.float32
        or beta.dtype != torch.float32
        or g.shape != v.shape[:-1]
        or beta.shape != g.shape
        or g.device != q.device
        or beta.device != q.device
        or cu is None
        or cu.ndim != 1
        or cu.dtype not in (torch.int32, torch.int64)
        or cu.device != q.device
        or not cu.is_contiguous()
        or kwargs["cu_seqlens_cpu"] is None
        or kwargs["state_indices"] is not None
        or kwargs["checkpoint_every_n_tokens"] != 0
        or kwargs["state_checkpoints"] is not None
        or kwargs["checkpoint_cu_starts"] is not None
        or kwargs["_cp_chunk_len"] is not None
        or any(x is not None and x.dtype != torch.float32 for x in (initial, final))
    ):
        return fallback()
    if any(x is not None and x.requires_grad for x in (q, k, v, g, beta, initial)):
        return fallback()
    props = torch.cuda.get_device_properties(q.device)
    if (props.major, props.minor, props.multi_processor_count) != (12, 0, 156):
        return fallback()
    use_cp = kwargs["use_cp"]
    if use_cp not in ("auto", True, False):
        return fallback()
    host = kwargs["cu_seqlens_cpu"]
    if isinstance(host, torch.Tensor):
        if (
            host.device.type != "cpu"
            or host.ndim != 1
            or host.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("cu_seqlens_cpu must be a 1D int32/int64 CPU tensor")
        host = host.tolist()
    values = tuple(host)
    if (
        len(values) != cu.numel()
        or any(type(x) is not int for x in values)
        or not values
        or values[0] != 0
        or values[-1] != q.shape[0]
        or any(b < a for a, b in zip(values, values[1:], strict=False))
    ):
        raise ValueError("cu_seqlens_cpu must match packed sequence offsets")
    lengths = tuple(b - a for a, b in zip(values, values[1:], strict=False))
    if not lengths or min(lengths) <= 0:
        return fallback()
    expected_state = (len(lengths), v.shape[1], 128, 128)
    if any(
        x is not None and (x.shape != expected_state or x.device != q.device)
        for x in (initial, final)
    ):
        return fallback()
    output = kwargs["output"]
    if output is not None and (
        output.shape != v.shape or output.dtype != q.dtype or output.device != q.device
    ):
        return fallback()
    hint = kwargs["max_seqlen"]
    if hint is not None and (
        type(hint) is not int or hint < max(lengths) or hint > q.shape[0]
    ):
        raise ValueError("max_seqlen must bound the supplied sequence lengths")
    key = (lengths, v.shape[1], str(q.device), cu.dtype, str(use_cp))
    if key not in _plans and len(_plans) >= 64:
        return fallback()
    from ._api import build_plan, chunk_gated_delta_rule

    if key not in _plans:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "warm SM120 CuTe sequence metadata before CUDA graph capture"
            )
        _plans[key] = build_plan(
            lengths, v.shape[1], q.device, cu.dtype, use_cp is not False, use_cp is True
        )
    out, state = chunk_gated_delta_rule(
        q[None],
        k[None],
        v[None],
        g[None],
        beta[None],
        scale=kwargs["scale"],
        initial_state=initial,
        output_final_state=kwargs["output_final_state"] or final is not None,
        use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
        cu_seqlens=cu,
        state_v_first=True,
        plan=_plans[key],
    )
    out = out.squeeze(0)
    if output is not None:
        output.copy_(out)
        out = output
    if state is not None and final is not None:
        final.copy_(state)
        state = final
    return (out, state) if kwargs["output_final_state"] else out
