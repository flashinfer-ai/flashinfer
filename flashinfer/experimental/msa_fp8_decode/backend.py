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

from __future__ import annotations

import math

import torch

from ...msa_ops._blackwell_sm100 import (
    _HEAD_DIM,
    MSASparseAttentionWorkspace,
    _bind_workspace,
    _check_warmed_launch,
    _launch_signature,
    _record_successful_launch,
    _select_target,
    _stream_ptr,
    _workspace_buffer,
)


def _packed_decode_tensor(tensor, name, shape, dtype, device, *, contiguous=True):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if tuple(tensor.shape) != tuple(shape) or tensor.dtype != dtype:
        raise ValueError(f"{name} must have shape {tuple(shape)} and dtype {dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def run_packed_fp8_decode(
    q,
    k,
    v,
    q2k_indices,
    *,
    page_table,
    seqused_k,
    seqlen_q,
    k_scale,
    v_scale,
    softmax_scale,
    workspace,
    out,
):
    """Use native block-sparse attention without repacking FP8 K/V or Q."""
    from ...decode import trtllm_batch_decode_with_kv_cache
    from ...utils import (
        get_device_sm_count,
        get_trtllm_gen_multi_ctas_kv_counter_bytes,
    )
    from .jit import load_msa_decode_metadata_module

    if isinstance(softmax_scale, torch.Tensor):
        raise TypeError("softmax_scale must be a host float, not a torch.Tensor")
    scale = _HEAD_DIM**-0.5 if softmax_scale is None else float(softmax_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("softmax_scale must be finite and positive")
    if not isinstance(workspace, MSASparseAttentionWorkspace):
        raise TypeError("packed FP8 decode requires an MSASparseAttentionWorkspace")
    if not q.is_cuda or q.ndim != 3 or q.shape[2] != _HEAD_DIM:
        raise ValueError("q must be CUDA [total_q, num_qo_heads, 128]")
    total_q, hq, _ = q.shape
    if not isinstance(seqlen_q, int):
        raise TypeError("seqlen_q must be a host integer")
    if not 1 <= seqlen_q <= 8 or total_q % seqlen_q:
        raise ValueError(
            "q rows must equal batch_size * seqlen_q, with seqlen_q in [1, 8]"
        )
    batch_size = total_q // seqlen_q
    if not 1 <= batch_size <= 256:
        raise ValueError("packed FP8 decode requires batch_size in [1, 256]")
    if k.ndim != 4 or k.shape[0] <= 0:
        raise ValueError("k/v must be packed paged views [num_pages, Hkv, 128, 128]")
    hkv = k.shape[1]
    if hkv not in (1, 4) or hq != hkv * 16:
        raise ValueError("packed FP8 decode supports head layouts 16/1 and 64/4")
    device = q.device
    _packed_decode_tensor(q, "q", (total_q, hq, 128), torch.bfloat16, device)
    for tensor, name in ((k, "k"), (v, "v")):
        _packed_decode_tensor(
            tensor,
            name,
            (k.shape[0], hkv, 128, 128),
            torch.float8_e4m3fn,
            device,
            contiguous=False,
        )
        expected = (hkv * 128 * 256, 128 * 256, 256, 1)
        if any(
            size > 1 and stride != want
            for size, stride, want in zip(
                tensor.shape, tensor.stride(), expected, strict=True
            )
        ):
            raise ValueError(
                "k/v must be split views of packed HND FP8 KV with token stride 256"
            )
    if (
        k.untyped_storage().data_ptr() != v.untyped_storage().data_ptr()
        or v.data_ptr() - k.data_ptr() != 128
    ):
        raise ValueError("k/v must share the same packed FP8 KV allocation")
    _packed_decode_tensor(
        q2k_indices,
        "q2k_indices",
        (hkv, total_q, 16),
        torch.int32,
        device,
        contiguous=False,
    )
    _packed_decode_tensor(
        seqused_k,
        "seqused_k",
        (batch_size,),
        torch.int32,
        device,
        contiguous=False,
    )
    if not isinstance(page_table, torch.Tensor) or page_table.ndim != 2:
        raise ValueError("page_table must have shape [batch_size, max_pages]")
    if not 1 <= page_table.shape[1] <= 2048:
        raise ValueError("page_table capacity must be in [1, 2048] pages")
    _packed_decode_tensor(
        page_table,
        "page_table",
        (batch_size, page_table.shape[1]),
        torch.int32,
        device,
        contiguous=False,
    )
    for tensor, name in ((k_scale, "k_scale"), (v_scale, "v_scale")):
        if not isinstance(tensor, torch.Tensor) or tensor.numel() != 1:
            raise ValueError(f"{name} must be a CUDA float32 scalar tensor")
        _packed_decode_tensor(tensor, name, tensor.shape, torch.float32, device)
    if out is not None:
        _packed_decode_tensor(out, "out", q.shape, torch.bfloat16, device)
    target = _select_target(device)

    capturing = torch.cuda.is_current_stream_capturing()
    stream_ptr = _stream_ptr(device)
    with workspace._lock:
        _bind_workspace(
            workspace, device=device, stream_ptr=stream_ptr, capturing=capturing
        )

        def buffer(name, shape, dtype):
            return _workspace_buffer(
                workspace, "packed_fp8_" + name, shape, dtype=dtype, device=device
            )

        if out is None:
            out = buffer("out", tuple(q.shape), torch.bfloat16)
        # TRT-LLM-gen multi-CTA-KV scratch contract
        # (include/flashinfer/trtllm/fmha/kernelParams.h "partial buffers" and
        # csrc/fmhaReduction.cu): one float2 partial-softmax slot plus one BF16
        # partial-O row of head_dim elements per launched CTA, and
        # computeCtaAndClusterConfig caps the launch at the SM count, so the
        # demand is input-shape independent. The factor 32 covers any
        # generation kernel's mStepQ. This route disables LSE output, so the
        # softmax-statistics slab behind the generic 128 MiB wrapper workspace
        # is never allocated here.
        scratch = buffer(
            "scratch",
            (32 * get_device_sm_count(device) * (8 + 2 * _HEAD_DIM),),
            torch.uint8,
        )
        pages = buffer("pages", (hkv, total_q, 16), torch.int32)
        lengths = buffer("lengths", (hkv, total_q), torch.int32)
        qk_scale = buffer("qk_scale", (1,), torch.float32)
        previous_counter = workspace._buffers.get("packed_fp8_counter")
        counter = buffer(
            "counter",
            (
                get_trtllm_gen_multi_ctas_kv_counter_bytes(
                    total_q, hq, get_device_sm_count(device)
                ),
            ),
            torch.uint8,
        )
        if counter is not previous_counter:
            counter.zero_()
        signature = _launch_signature(
            variant="packed_fp8_decode",
            target=target,
            tensors=(
                q,
                k,
                v,
                q2k_indices,
                page_table,
                seqused_k,
                k_scale,
                v_scale,
                out,
                scratch,
                pages,
                lengths,
                qk_scale,
                counter,
            ),
            scalars=(seqlen_q, scale),
            grid=((total_q * hkv + 3) // 4, 1, 1),
        )
        _check_warmed_launch(workspace, signature, capturing=capturing)
        load_msa_decode_metadata_module(target).prepare(
            q2k_indices,
            page_table,
            seqused_k,
            k_scale,
            pages,
            lengths,
            qk_scale,
            seqlen_q,
            k.shape[0],
            scale,
        )
        # Flatten to Q1 requests: every query/KV-head keeps its own sparse row.
        result = trtllm_batch_decode_with_kv_cache(
            q,
            (k, v),
            scratch,
            pages,
            lengths,
            16 * 128,
            bmm1_scale_log2=qk_scale,
            bmm2_scale=v_scale,
            out=out,
            backend="trtllm-gen",
            q_len_per_req=1,
            enable_block_sparse_attention=True,
            multi_ctas_kv_counter_buffer=counter,
        )
        _record_successful_launch(workspace, signature, capturing=capturing)
    return result
