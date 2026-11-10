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

------------------------------------------------------------------------------
Multi-LoRA delta builders for the routed TRT-LLM fused MoE (single node).

They compute the per-(token, expert) LoRA deltas for a SwiGLU MoE expert FFN by driving
the BGMV shrink/expand kernels (``bgmv_moe.py``):

  * FC1 (gate_up) -> ``bgmv_moe_gemm1_lora_delta`` -> ``[T, k, 2I]``, fed into
    ``trtllm_*_moe`` as ``gemm1_lora_delta`` (added pre-SwiGLU).
  * FC2 (down)    -> ``bgmv_moe_gemm2_lora_delta`` -> ``[T, H]``, added to the MoE output.

LoRA weights are caller-managed: the builders take the ``[num_slices, num_experts]`` int64
base-pointer tables (``w_ptr``) and the per-adapter ``lora_stride``, built once with
:func:`flashinfer.fused_moe.fill_w_ptr` over banks ``[max_loras, num_experts, *, *]``.
``topk_ids`` is the UNPACKED expert id per (token, slot).
"""

from typing import Tuple

import torch

from ..api_logging import flashinfer_api
from .bgmv_moe import bgmv_moe_expand, bgmv_moe_shrink


def _expanded_pairs(
    topk_ids: torch.Tensor, lora_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the token-major (``p = token*top_k + slot``) per-pair routing arrays.

    ``lora_ids`` stays per-token ``[T]`` (the kernels look up the adapter via the real
    token, which ``sorted_token_ids`` carries). Returns int64 ``token_per_pair [P]`` and
    ``expert_per_pair [P]`` on ``topk_ids.device``.
    """
    T, k = topk_ids.shape
    device = topk_ids.device
    token_per_pair = torch.arange(
        T, device=device, dtype=torch.int64
    ).repeat_interleave(k)
    expert_per_pair = topk_ids.reshape(-1).to(torch.int64)
    return token_per_pair, expert_per_pair


@flashinfer_api
def bgmv_moe_gemm1_lora_delta(
    hidden_states: torch.Tensor,
    w_ptr_a: torch.Tensor,
    lora_stride_a: int,
    w_ptr_b: torch.Tensor,
    lora_stride_b: int,
    topk_ids: torch.Tensor,
    lora_ids: torch.Tensor,
    rank: int,
    intermediate_size: int,
    *,
    lora_dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FC1 (gate_up_proj) LoRA delta for a routed MoE, in the layout consumed by
    ``trtllm_*_moe``'s ``gemm1_lora_delta``.

    For each routed pair ``(token t, slot j)`` with expert ``e = topk_ids[t, j]`` and
    adapter ``l = lora_ids[t]`` (skipped when ``l < 0``)::

        delta[t, j] = scale * concat( B_gate[l,e] @ (A_gate[l,e] @ x[t]),
                                      B_up[l,e]   @ (A_up[l,e]   @ x[t]) )

    Unweighted and kept per-(token, slot) (it is added before the nonlinear SwiGLU, so it
    must not be summed over experts nor scaled by routing weights).

    **Performant (shared-A) LoRA** is selected by passing 1D ``[num_experts]`` pointer tables
    (no slice dim): a single shared ``A`` feeds both slices and ``B`` is fused horizontally
    into one ``[2I, rank]`` matrix, so the delta is ``scale * B_fused @ (A_shared @ x[t])`` —
    one shrink + one fused expand instead of two of each.

    Parameters
    ----------
    hidden_states : torch.Tensor
        ``[T, H]`` FFN input. Cast to ``lora_dtype`` for the LoRA path, independent of the
        FP8/MXFP8 base weights.
    w_ptr_a : torch.Tensor
        LoRA-A int64 base-pointer table, from :func:`fill_w_ptr`. ``[2, num_experts]``
        (regular) points to ``[A_gate, A_up]``; ``[num_experts]`` 1D (no slice dim) selects
        **performant LoRA** (one shared A).
    lora_stride_a : int
        Element stride between adapters in the A bank(s) (the ``fill_w_ptr`` return value).
    w_ptr_b : torch.Tensor
        LoRA-B int64 base-pointer table. ``[2, num_experts]`` (regular) points to
        ``[B_gate, B_up]``; ``[num_experts]`` 1D selects performant LoRA (B fused as
        ``[2I, rank]``: gate rows ``0:I``, up rows ``I:2I``).
    lora_stride_b : int
        Element stride between adapters in the B bank(s).
    topk_ids : torch.Tensor
        ``[T, top_k]`` int — UNPACKED routed expert id per (token, slot).
    lora_ids : torch.Tensor
        ``[T]`` int — adapter id per token, ``-1`` for no adapter.
    rank : int
        LoRA rank (the A/B contraction dim).
    intermediate_size : int
        ``I`` — the per-slice (gate / up) output width; total FC1 width is ``2*I``.
    lora_dtype : torch.dtype
        Dtype of the LoRA weights the ``w_ptr`` tables point to (bf16/fp16).
    scale : float
        LoRA ``alpha / rank`` scaling applied to the delta.
    out_dtype : torch.dtype
        Output dtype; bf16 to match ``gemm1_lora_delta``'s required dtype.

    Returns
    -------
    torch.Tensor
        ``[T, top_k, 2*I]`` in ``out_dtype``. Pass as ``gemm1_lora_delta``.
    """
    if w_ptr_a.dim() != w_ptr_b.dim():
        raise ValueError(
            "w_ptr_a and w_ptr_b must have the same rank (both 2D or both 1D)"
        )
    performant = w_ptr_a.dim() == 1
    if performant:
        w_ptr_a = w_ptr_a.reshape(1, -1)
        w_ptr_b = w_ptr_b.reshape(1, -1)
    else:
        assert w_ptr_a.shape[0] == 2 and w_ptr_b.shape[0] == 2, (
            "regular FC1 LoRA is a 2-slice (gate, up) GLU projection"
        )

    T, _ = hidden_states.shape
    k = topk_ids.shape[1]
    P = T * k
    inter = intermediate_size
    device = hidden_states.device

    token_per_pair, expert_per_pair = _expanded_pairs(topk_ids, lora_ids)
    lora_idx = lora_ids.to(torch.int64)
    x = hidden_states.to(lora_dtype)

    num_slices = 1 if performant else 2

    # Shrink: x @ A -> [num_slices, P, rank]. Per-token input read (default mode).
    shrink_out = torch.zeros(num_slices, P, rank, dtype=lora_dtype, device=device)
    bgmv_moe_shrink(
        shrink_out,
        x,
        w_ptr_a,
        token_per_pair,
        expert_per_pair,
        lora_idx,
        lora_stride_a,
        per_pair_input=False,
    )

    # Expand: shrink_out @ B = [P, 2I], per-pair unweighted store; zeroed so skipped pairs stay 0.
    if performant:
        slice_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
        output_slices = [2 * inter]
    else:
        slice_start_loc = torch.tensor([0, inter], dtype=torch.int64, device=device)
        output_slices = [inter, inter]
    unit_w = torch.ones(P, dtype=torch.float32, device=device)  # ignored by the kernel
    y = torch.zeros(P, 2 * inter, dtype=torch.float32, device=device)
    bgmv_moe_expand(
        y,
        shrink_out,
        w_ptr_b,
        token_per_pair,
        expert_per_pair,
        unit_w,
        lora_idx,
        slice_start_loc,
        output_slices,
        lora_stride_b,
        finalize=False,
    )

    if scale != 1.0:
        y = y * scale
    return y.view(T, k, 2 * inter).to(out_dtype)


@flashinfer_api
def bgmv_moe_gemm2_lora_delta(
    gemm1_activation_output: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    w_ptr_a: torch.Tensor,
    lora_stride_a: int,
    w_ptr_b: torch.Tensor,
    lora_stride_b: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    lora_ids: torch.Tensor,
    rank: int,
    hidden_size: int,
    *,
    lora_dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FC2 (down_proj) LoRA delta for a routed MoE, to be ADDED to the MoE output.

    Consumes the post-SwiGLU activation returned by ``trtllm_*_moe`` (called with
    ``gemm1_lora_delta`` set and ``do_finalize=True``). For each routed pair
    ``(token t, slot j)`` with expert ``e`` and adapter ``l``::

        delta[t] = scale * Σ_j  w[t, j] * ( B_down[l,e] @ (A_down[l,e] @ a[t, j]) )

    Weighted and combined over experts (added after FC2, post-combine).

    Parameters
    ----------
    gemm1_activation_output : torch.Tensor
        ``[padded_rows, I]`` PERMUTED, post-SwiGLU activation (trtllm return). ``I`` (the
        down-projection input width) is read from ``shape[1]``.
    expanded_idx_to_permuted_idx : torch.Tensor
        ``[T*top_k]`` int — maps expanded index ``token*top_k+slot`` to the permuted row
        (trtllm return); ``< 0`` marks an inactive slot.
    w_ptr_a : torch.Tensor
        ``[1, num_experts]`` int64 base-pointer table for ``[A_down]``, from :func:`fill_w_ptr`.
    lora_stride_a : int
        Element stride between adapters in the A_down bank.
    w_ptr_b : torch.Tensor
        ``[1, num_experts]`` int64 base-pointer table for ``[B_down]``.
    lora_stride_b : int
        Element stride between adapters in the B_down bank.
    topk_ids, topk_weights : torch.Tensor
        ``[T, top_k]`` routed expert ids (int) and per-expert combine weights (f32).
    lora_ids : torch.Tensor
        ``[T]`` int adapter id per token (``-1`` = none).
    rank : int
        LoRA rank.
    hidden_size : int
        ``H`` — the down-projection output width.
    lora_dtype : torch.dtype
        Dtype of the LoRA weights / activation gather buffer (bf16/fp16).
    scale : float
        LoRA ``alpha / rank`` scaling.
    out_dtype : torch.dtype
        Output dtype (match the MoE output, e.g. bf16).

    Returns
    -------
    torch.Tensor
        ``[T, H]`` in ``out_dtype``. Add to the MoE output.
    """
    assert w_ptr_a.shape[0] == 1 and w_ptr_b.shape[0] == 1, (
        "FC2 LoRA is a single down-projection slice"
    )
    T, k = topk_ids.shape
    P = T * k
    inter = gemm1_activation_output.shape[1]
    hidden = hidden_size
    device = gemm1_activation_output.device

    token_per_pair, expert_per_pair = _expanded_pairs(topk_ids, lora_ids)
    lora_idx = lora_ids.to(torch.int64)

    # Gather permuted activation into expanded [P, I] order; inactive slots (perm < 0) stay 0.
    perm = expanded_idx_to_permuted_idx.to(torch.int64)
    valid = perm >= 0
    a_exp = torch.zeros(P, inter, dtype=lora_dtype, device=device)
    a_exp[valid] = gemm1_activation_output[perm[valid]].to(lora_dtype)

    # Shrink: a_exp @ A_down -> [1, P, rank]. Per-pair input read (per_pair_input=True).
    shrink_out = torch.zeros(1, P, rank, dtype=lora_dtype, device=device)
    bgmv_moe_shrink(
        shrink_out,
        a_exp,
        w_ptr_a,
        token_per_pair,
        expert_per_pair,
        lora_idx,
        lora_stride_a,
        per_pair_input=True,
    )

    # Expand (finalize): shrink_out @ B_down -> [T, H], routing-weighted combine over experts.
    slice_start_loc = torch.tensor([0], dtype=torch.int64, device=device)
    topk_w = topk_weights.reshape(P).to(torch.float32)
    y = torch.zeros(T, hidden, dtype=torch.float32, device=device)
    bgmv_moe_expand(
        y,
        shrink_out,
        w_ptr_b,
        token_per_pair,
        expert_per_pair,
        topk_w,
        lora_idx,
        slice_start_loc,
        [hidden],
        lora_stride_b,
        finalize=True,
    )

    if scale != 1.0:
        y = y * scale
    return y.to(out_dtype)
