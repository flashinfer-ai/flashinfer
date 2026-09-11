# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

from dataclasses import dataclass
import torch


@dataclass
class _Plan:
    signature: tuple
    metadata: object
    provider_path: str


@dataclass
class _FrostPlan:
    signature: tuple
    arithmetic: str
    out: torch.Tensor
    lse: torch.Tensor
    workspace: dict
    backend: str = "frost"


def dispatch_decode(
    q,
    swa_cache,
    global_cache,
    swa_indices,
    global_indices,
    sink,
    *,
    plan=None,
    backend="flashmla",
    arithmetic=None,
):
    if backend == "flashmla":
        if arithmetic is not None or isinstance(plan, _FrostPlan):
            raise ValueError("Frost arithmetic/plan cannot be used with FlashMLA")
        return decode(
            q, swa_cache, global_cache, swa_indices, global_indices, sink, plan=plan
        )
    if backend != "frost":
        raise ValueError("decode backend must be 'frost' or 'flashmla'")
    arithmetic = "bf16x3" if arithmetic is None else arithmetic
    if arithmetic not in ("tf32x3", "bf16x3"):
        raise ValueError("Frost arithmetic must be 'tf32x3' or 'bf16x3'")
    if q.device.type != "cuda":
        raise ValueError("Frost decode requires CUDA inputs")
    tensors = (q, swa_cache, global_cache, swa_indices, global_indices, sink)
    signature = tuple(
        None if x is None else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
        for x in tensors
    )
    if plan is not None and (
        not isinstance(plan, _FrostPlan)
        or plan.signature != signature
        or plan.arithmetic != arithmetic
    ):
        raise ValueError("Frost decode plan declaration/arithmetic mismatch")
    from .decode_fp32 import decode_fp32

    with torch.cuda.device(q.device):
        if plan is None and torch.cuda.is_current_stream_capturing():
            raise ValueError("prepare a Frost decode plan before CUDA Graph capture")
        out, lse, workspace = decode_fp32(
            *tensors,
            out=None if plan is None else plan.out,
            lse=None if plan is None else plan.lse,
            workspace=None if plan is None else plan.workspace,
            recipe="bf16x3_tcgen" if arithmetic == "bf16x3" else "tf32x3",
            head_tile=None if arithmetic == "bf16x3" else 16,
        )
    if plan is None:
        plan = _FrostPlan(signature, arithmetic, out, lse, workspace)
    return out, lse, plan


def decode(q, swa_cache, global_cache, swa_indices, global_indices, sink, *, plan=None):
    import flash_mla

    if q.device.type != "cuda" or torch.cuda.get_device_capability(q.device) not in (
        (10, 0),
        (10, 3),
    ):
        raise ValueError("V4.1 mixed-cache decode currently requires SM100/SM103")
    if q.ndim != 4 or q.shape[-2:] != (64, 512) or q.dtype != torch.bfloat16:
        raise ValueError("Q must be BF16 [B,Sq,64,512]")
    if (global_cache is None) != (global_indices is None):
        raise ValueError(
            "global cache and indices must both be provided or both be None"
        )
    declarations = (q, swa_cache, global_cache, swa_indices, global_indices, sink)
    tensors = tuple(x for x in declarations if x is not None)
    for x in tensors:
        if x.device != q.device or not x.is_contiguous():
            raise ValueError("all inputs must be contiguous on Q's device")
        if x.requires_grad and torch.is_grad_enabled():
            raise ValueError("decode is inference-only")
    for cache, width in ((swa_cache, 528), (global_cache, 288)):
        if cache is None:
            continue
        if (
            cache.ndim != 4
            or cache.shape[2:] != (1, width)
            or cache.dtype != torch.uint8
            or cache.shape[1] not in (32, 64, 128)
        ):
            raise ValueError("invalid V4.1 packed cache declaration")
    for indices in (swa_indices, global_indices):
        if indices is None:
            continue
        if (
            indices.ndim != 3
            or indices.shape[:2] != q.shape[:2]
            or indices.dtype != torch.int32
        ):
            raise ValueError("indices must be int32 [B,Sq,K]")
    if sink.shape != (64,) or sink.dtype != torch.float32:
        raise ValueError("sink must be FP32 [64]")
    signature = tuple(
        None if x is None else (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
        for x in declarations
    )
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise ValueError(
                "warm up decode and supply its plan before CUDA Graph capture"
            )
        metadata, _ = flash_mla.get_mla_metadata()
        plan = _Plan(signature, metadata, flash_mla.__file__)
    elif (
        not isinstance(plan, _Plan)
        or plan.signature != signature
        or plan.provider_path != flash_mla.__file__
    ):
        raise ValueError("decode plan declaration/provider mismatch; create a new plan")
    # Length tensors stay None: changing their VALUES would invalidate scheduling
    # metadata despite identical declarations. Invalid slots are represented -1.
    with torch.cuda.device(q.device):
        out, lse = flash_mla.flash_mla_with_kvcache(
            q,
            swa_cache,
            None,
            None,
            512,
            plan.metadata,
            softmax_scale=512**-0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=sink,
            extra_k_cache=global_cache,
            extra_indices_in_kvcache=global_indices,
        )
    return out, lse, plan
