"""
Copyright (c) 2025 by FlashInfer team.

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

import functools

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.moe_utils import gen_moe_utils_module
from flashinfer.utils import device_support_pdl

_ROUTING_TILE_TOKENS_DIM = 128
_workspace_pool: dict[tuple[str, int | None], "_RawLogitsTopkWorkspace"] = {}


@functools.lru_cache(maxsize=1)
def _get_moe_utils_module():
    spec = gen_moe_utils_module()
    return spec.build_and_load()


def _get_cuda_stream_ptr(device: torch.device) -> int:
    return torch.cuda.current_stream(device=device).cuda_stream


class _RawLogitsTopkWorkspace:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.topk_weights_bf16 = torch.empty((0,), dtype=torch.bfloat16, device=device)
        self.topk_packed = torch.empty((0,), dtype=torch.int32, device=device)
        self.expert_counts = torch.empty((0,), dtype=torch.int32, device=device)
        self.permuted_idx_size = torch.empty((0,), dtype=torch.int32, device=device)
        self.expanded_idx_to_permuted_idx = torch.empty(
            (0,), dtype=torch.int32, device=device
        )
        self.permuted_idx_to_token_idx = torch.empty((0,), dtype=torch.int32, device=device)
        self.cta_idx_to_batch_idx = torch.empty((0,), dtype=torch.int32, device=device)
        self.cta_idx_to_mn_limit = torch.empty((0,), dtype=torch.int32, device=device)
        self.num_non_exiting_ctas = torch.empty((0,), dtype=torch.int32, device=device)

    @staticmethod
    def _ensure_capacity(tensor: torch.Tensor, numel: int) -> torch.Tensor:
        if tensor.numel() >= numel:
            return tensor
        return torch.empty((numel,), dtype=tensor.dtype, device=tensor.device)

    def get_views(
        self,
        num_tokens: int,
        topk: int,
        num_experts: int,
        max_num_tiles: int,
        max_num_permuted_tokens: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        expanded_size = num_tokens * topk
        expert_counts_size = max(512, 2 * num_experts)

        self.topk_weights_bf16 = self._ensure_capacity(self.topk_weights_bf16, expanded_size)
        self.topk_packed = self._ensure_capacity(self.topk_packed, expanded_size)
        self.expert_counts = self._ensure_capacity(self.expert_counts, expert_counts_size)
        self.permuted_idx_size = self._ensure_capacity(self.permuted_idx_size, 1)
        self.expanded_idx_to_permuted_idx = self._ensure_capacity(
            self.expanded_idx_to_permuted_idx, expanded_size
        )
        self.permuted_idx_to_token_idx = self._ensure_capacity(
            self.permuted_idx_to_token_idx, max_num_permuted_tokens
        )
        self.cta_idx_to_batch_idx = self._ensure_capacity(
            self.cta_idx_to_batch_idx, max_num_tiles
        )
        self.cta_idx_to_mn_limit = self._ensure_capacity(self.cta_idx_to_mn_limit, max_num_tiles)
        self.num_non_exiting_ctas = self._ensure_capacity(self.num_non_exiting_ctas, 1)

        return (
            self.topk_weights_bf16[:expanded_size].view(num_tokens, topk),
            self.topk_packed[:expanded_size].view(num_tokens, topk),
            self.expert_counts[:expert_counts_size],
            self.permuted_idx_size[:1],
            self.expanded_idx_to_permuted_idx[:expanded_size],
            self.permuted_idx_to_token_idx[:max_num_permuted_tokens],
            self.cta_idx_to_batch_idx[:max_num_tiles],
            self.cta_idx_to_mn_limit[:max_num_tiles],
            self.num_non_exiting_ctas[:1],
        )


def _get_workspace(device: torch.device) -> _RawLogitsTopkWorkspace:
    key = (device.type, device.index)
    ws = _workspace_pool.get(key)
    if ws is None:
        ws = _RawLogitsTopkWorkspace(device)
        _workspace_pool[key] = ws
    return ws


def _get_max_num_tiles(
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    tile_size: int,
) -> int:
    # Mirrors TRTLLM GroupedGemmInputsHelper.get_max_num_tiles.
    num_expanded_tokens = num_tokens * top_k
    if num_expanded_tokens <= num_local_experts:
        return num_expanded_tokens
    num_remaining_tokens = num_expanded_tokens - num_local_experts
    return num_local_experts + (num_remaining_tokens + tile_size - 1) // tile_size


def _validate_args(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> None:
    if topk_weights.ndim != 2 or topk_ids.ndim != 2 or gating_output.ndim != 2:
        raise ValueError(
            "Expected 2D tensors for topk_weights, topk_ids, and gating_output."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            f"topk_weights/topk_ids shape mismatch: {topk_weights.shape} vs {topk_ids.shape}"
        )
    if topk_weights.shape[0] != gating_output.shape[0]:
        raise ValueError(
            "Batch size mismatch: "
            f"{topk_weights.shape[0]} (output) vs {gating_output.shape[0]} (gating_output)"
        )
    if topk_weights.dtype != torch.float32:
        raise ValueError(
            f"Expected topk_weights dtype float32, got {topk_weights.dtype}"
        )
    if not topk_weights.is_contiguous():
        raise ValueError("Expected topk_weights to be contiguous.")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f"Expected topk_ids dtype int32 or int64, got {topk_ids.dtype}"
        )
    if not topk_ids.is_contiguous():
        raise ValueError("Expected topk_ids to be contiguous.")
    if gating_output.dtype != torch.bfloat16:
        raise ValueError(
            "TRTLLM routingRenormalize path expects bf16 gating_output, got "
            f"{gating_output.dtype}"
        )
    if (
        topk_weights.device != gating_output.device
        or topk_ids.device != gating_output.device
    ):
        raise ValueError(
            "topk_weights, topk_ids, and gating_output must be on the same device."
        )
    if gating_output.device.type != "cuda":
        raise ValueError("TRTLLM routingRenormalize path only supports CUDA tensors.")
    if not isinstance(renormalize, bool):
        raise ValueError(f"renormalize must be bool, got {type(renormalize)}")

    topk = topk_weights.shape[1]
    num_experts = gating_output.shape[1]
    if topk < 1:
        raise ValueError(f"Invalid top-k: {topk}")
    if topk > num_experts:
        raise ValueError(f"Invalid top-k {topk} for num_experts={num_experts}")
    if topk > 10:
        raise ValueError(f"TRTLLM routingRenormalize supports top-k <= 10, got {topk}")
    if num_experts > 512:
        raise ValueError(
            f"TRTLLM routingRenormalize supports num_experts <= 512, got {num_experts}"
        )
    if num_experts % 4 != 0:
        raise ValueError(
            f"TRTLLM routingRenormalize expects num_experts % 4 == 0, got {num_experts}"
        )


@flashinfer_api
def fused_topk_raw_logits(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = True,
) -> None:
    """TRTLLM routingRenormalize raw-logits top-k.

    This API intentionally uses only TRTLLM's routingRenormalize path. There is no
    fallback path in this implementation.

    Supported configuration:
    - ``gating_output`` dtype ``torch.bfloat16``
    - routing score mode fixed to raw-logits + optional post-topk softmax
    - routing tile size fixed to 128
    """

    _validate_args(
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        gating_output=gating_output,
        renormalize=renormalize,
    )

    num_tokens, num_experts = gating_output.shape
    topk = topk_weights.shape[1]
    max_num_tiles = _get_max_num_tiles(
        num_tokens=num_tokens,
        top_k=topk,
        num_local_experts=num_experts,
        tile_size=_ROUTING_TILE_TOKENS_DIM,
    )
    max_num_permuted_tokens = max_num_tiles * _ROUTING_TILE_TOKENS_DIM
    use_pdl = device_support_pdl(gating_output.device)

    (
        topk_weights_bf16,
        topk_packed,
        expert_counts,
        permuted_idx_size,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        cta_idx_to_batch_idx,
        cta_idx_to_mn_limit,
        num_non_exiting_ctas,
    ) = _get_workspace(gating_output.device).get_views(
        num_tokens=num_tokens,
        topk=topk,
        num_experts=num_experts,
        max_num_tiles=max_num_tiles,
        max_num_permuted_tokens=max_num_permuted_tokens,
    )

    if not gating_output.is_contiguous():
        gating_output = gating_output.contiguous()
    # Initialize tail entries so searchsorted sees a monotonic full-length array.
    cta_idx_to_batch_idx.zero_()
    cta_idx_to_mn_limit.fill_(torch.iinfo(torch.int32).max)
    _get_moe_utils_module()["flashinfer_fused_topk_raw_logits_trtllm_renormalize"](
        topk_weights_bf16.data_ptr(),
        topk_packed.data_ptr(),
        gating_output.data_ptr(),
        num_tokens,
        num_experts,
        topk,
        renormalize,
        use_pdl,
        _ROUTING_TILE_TOKENS_DIM,
        expert_counts.data_ptr(),
        permuted_idx_size.data_ptr(),
        expanded_idx_to_permuted_idx.data_ptr(),
        permuted_idx_to_token_idx.data_ptr(),
        cta_idx_to_batch_idx.data_ptr(),
        cta_idx_to_mn_limit.data_ptr(),
        num_non_exiting_ctas.data_ptr(),
        _get_cuda_stream_ptr(gating_output.device),
    )

    # Recover expert ids entirely on-device from routing metadata. We intentionally
    # avoid host syncs (e.g. tensor.item()) to keep this CUDA-graph safe.
    expanded = expanded_idx_to_permuted_idx.view(num_tokens, topk)
    cta_idx = torch.searchsorted(cta_idx_to_mn_limit, expanded, right=True)
    topk_ids_i32 = cta_idx_to_batch_idx[cta_idx].to(torch.int32)
    topk_weights.copy_(topk_weights_bf16.float())
    if topk_ids.dtype == torch.int32:
        topk_ids.copy_(topk_ids_i32)
    else:
        topk_ids.copy_(topk_ids_i32.to(torch.int64))
