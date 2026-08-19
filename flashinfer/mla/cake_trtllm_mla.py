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

import math
from typing import Literal, Optional

import torch

from ..jit.cake_trtllm_mla import get_cake_trtllm_mla_module
from ..jit.cpp_ext import is_cuda_version_at_least
from ..utils import get_compute_capability, get_device_sm_count, log2e

_CAKE_HEADS = 128
_CAKE_HEAD_DIM = 576
_CAKE_VALUE_DIM = 512
_CAKE_PAGE_SIZE = 32
_CAKE_MAX_SEQUENCE_LENGTH = 1024
_CAKE_MAX_Q_LEN = 16
_CAKE_CLUSTER_SIZE = 2


def _check_cake_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def cake_trtllm_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: Optional[list[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    enable_pdl: Optional[bool],
    uses_shared_paged_kv_idx: bool,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    enable_dcp: bool,
    backend: Literal["cake"] = "cake",
) -> torch.Tensor:
    """Run the verified low-batch BF16 Cake schedule on exact SM103a."""

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    if get_compute_capability(query.device) != (10, 3):
        major, minor = get_compute_capability(query.device)
        raise RuntimeError(
            f"Cake TRT-LLM MLA requires compute capability 10.3, got {major}.{minor}"
        )
    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError("Cake TRT-LLM MLA on SM103a requires CUDA 12.9 or newer")

    _check_cake_tensor(query, name="query", dtype=torch.bfloat16, device=query.device)
    if query.ndim != 4 or query.shape[2:] != (_CAKE_HEADS, _CAKE_HEAD_DIM):
        raise ValueError("query must have shape [B, q_len, 128, 576]")
    batch, q_len = int(query.shape[0]), int(query.shape[1])
    if batch <= 0 or batch > 4:
        raise ValueError(f"Cake TRT-LLM MLA batch must be in [1, 4], got {batch}")
    if q_len <= 0 or q_len > _CAKE_MAX_Q_LEN:
        raise ValueError(
            f"Cake TRT-LLM MLA q_len must be in [1, {_CAKE_MAX_Q_LEN}], got {q_len}"
        )

    device = query.device
    _check_cake_tensor(kv_cache, name="kv_cache", dtype=torch.bfloat16, device=device)
    if kv_cache.ndim not in (3, 4):
        raise ValueError("kv_cache must be a 3D or 4D paged tensor")
    if tuple(kv_cache.shape[-2:]) != (_CAKE_PAGE_SIZE, _CAKE_HEAD_DIM):
        raise ValueError("kv_cache must have page size 32 and head dimension 576")
    if kv_cache.ndim == 4 and kv_cache.shape[1] != 1:
        raise ValueError("4D kv_cache must have a singleton head axis")

    _check_cake_tensor(
        block_tables, name="block_tables", dtype=torch.int32, device=device
    )
    if (
        block_tables.ndim != 2
        or block_tables.shape[0] != batch
        or block_tables.shape[1] < _CAKE_MAX_SEQUENCE_LENGTH // _CAKE_PAGE_SIZE
    ):
        raise ValueError("block_tables must have contiguous shape [B, width >= 32]")
    if seq_lens is None:
        raise ValueError("seq_lens is required for the Cake backend")
    _check_cake_tensor(seq_lens, name="seq_lens", dtype=torch.int32, device=device)
    if tuple(seq_lens.shape) != (batch,):
        raise ValueError("seq_lens must have shape [B]")

    if (qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim) != (128, 512, 64):
        raise ValueError(
            "Cake TRT-LLM MLA requires qk_nope_head_dim=128, "
            "kv_lora_rank=512, and qk_rope_head_dim=64"
        )
    if max_seq_len != _CAKE_MAX_SEQUENCE_LENGTH:
        raise ValueError("Cake TRT-LLM MLA requires max_seq_len=1024")
    if sparse_mla_top_k != 0 or sparse_mla_top_k_lens is not None:
        raise ValueError("Cake TRT-LLM MLA supports dense MLA only")
    if not uses_shared_paged_kv_idx:
        raise ValueError("Cake TRT-LLM MLA requires shared paged KV indices")
    if sinks is not None:
        raise ValueError("Cake TRT-LLM MLA does not support sinks")
    if skip_softmax_threshold_scale_factor is not None:
        raise ValueError("Cake TRT-LLM MLA does not support skip-softmax")
    if enable_pdl:
        raise ValueError("Cake TRT-LLM MLA does not support PDL")
    if lse is not None or return_lse:
        raise ValueError("Cake TRT-LLM MLA does not return LSE")
    if cum_seq_lens_q is not None or max_q_len is not None:
        raise ValueError("Cake TRT-LLM MLA does not support compact variable Q")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError("Cake TRT-LLM MLA does not use a multi-CTA counter buffer")
    if enable_dcp:
        raise ValueError("Cake TRT-LLM MLA does not support DCP")
    if isinstance(bmm1_scale, torch.Tensor) or not isinstance(bmm1_scale, (int, float)):
        raise TypeError("Cake TRT-LLM MLA requires a scalar bmm1_scale")
    if isinstance(bmm2_scale, torch.Tensor) or not isinstance(bmm2_scale, (int, float)):
        raise TypeError("Cake TRT-LLM MLA requires a scalar bmm2_scale")
    if not math.isfinite(float(bmm1_scale)) or float(bmm1_scale) <= 0:
        raise ValueError("bmm1_scale must be finite and positive")
    if float(bmm2_scale) != 1.0:
        raise ValueError("Cake TRT-LLM MLA requires bmm2_scale=1.0")

    expected_out_shape = (batch, q_len, _CAKE_HEADS, _CAKE_VALUE_DIM)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=device)
    else:
        _check_cake_tensor(out, name="out", dtype=torch.bfloat16, device=device)
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")

    total_work_items = batch * q_len * 2
    grid_x = min(get_device_sm_count(device), total_work_items * _CAKE_CLUSTER_SIZE)
    grid_x = max(_CAKE_CLUSTER_SIZE, grid_x // _CAKE_CLUSTER_SIZE * _CAKE_CLUSTER_SIZE)
    get_cake_trtllm_mla_module().run(
        query,
        kv_cache,
        out,
        block_tables,
        seq_lens,
        q_len,
        int(block_tables.shape[1]),
        _CAKE_MAX_SEQUENCE_LENGTH,
        float(bmm1_scale) * log2e,
        total_work_items,
        grid_x,
        int(torch.cuda.current_stream(device).cuda_stream),
    )
    return out


__all__ = ["cake_trtllm_mla_decode"]
