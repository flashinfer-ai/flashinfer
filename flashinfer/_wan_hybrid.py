"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools

import torch


@functools.cache
def _get_wan_hybrid_attention_module(target: str):
    from .jit.wan_hybrid import gen_wan_hybrid_attention_module

    return gen_wan_hybrid_attention_module(target).build_and_load()


@functools.cache
def _get_wan_hybrid_dispatch_module(target: str):
    from .jit.wan_hybrid import gen_wan_hybrid_dispatch_module

    return gen_wan_hybrid_dispatch_module(target).build_and_load()


def _wan_hybrid_attention_target(device: torch.device | str | int) -> str:
    from .jit.wan_hybrid import _wan_hybrid_target

    target, _ = _wan_hybrid_target(device)
    return target


def _descriptor_signature(q, k, out, workspace) -> tuple[int, ...]:
    views = workspace._attention_abi_views
    return tuple(
        tensor.data_ptr()
        for tensor in (
            q,
            k,
            views.vt,
            views.sfvt_lo,
            views.sfvt_hi,
            out,
        )
    )


def wan_hybrid_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace,
    sm_scale: float,
) -> None:
    del v
    signature = _descriptor_signature(q, k, out, workspace)
    prepare_descriptors = workspace._descriptor_signature != signature
    if prepare_descriptors and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "wan_hybrid attention must be prewarmed with the same q, k, out, "
            "and workspace tensors before CUDA Graph capture"
        )

    target = _wan_hybrid_attention_target(q.device)
    module = _get_wan_hybrid_attention_module(target)
    views = workspace._attention_abi_views
    module.wan_hybrid_attention(
        q,
        k,
        views.vt,
        views.sfvt_lo,
        views.sfvt_hi,
        out,
        workspace._descriptor_storage,
        prepare_descriptors,
        sm_scale,
    )
    if prepare_descriptors:
        workspace._descriptor_signature = signature


def wan_hybrid_dispatch_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace,
    sm_scale: float,
) -> None:
    signature = _descriptor_signature(q, k, out, workspace)
    prepare_descriptors = workspace._descriptor_signature != signature
    if prepare_descriptors and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "wan_hybrid attention must be prewarmed with the same q, k, out, "
            "and workspace tensors before CUDA Graph capture"
        )

    target = _wan_hybrid_attention_target(q.device)
    module = _get_wan_hybrid_dispatch_module(target)
    views = workspace._attention_abi_views
    module.wan_hybrid_dispatch(
        q,
        k,
        v,
        views.vt,
        views.sfvt_lo,
        views.sfvt_hi,
        out,
        workspace._descriptor_storage,
        prepare_descriptors,
        sm_scale,
    )
    if prepare_descriptors:
        workspace._descriptor_signature = signature
