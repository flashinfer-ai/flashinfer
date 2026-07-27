# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-rank MegaMoE host driver (NVFP4 dispatch + fc1+fc2 + combine).

Parallel sibling of ``runner_fc12.py``.  ``runner_fc12.py`` is the
single-rank fused fc1+fc2 perf testbed (no cross-rank comm); this
runner adds dispatch and combine on top of the same fc1+fc2 math and
runs them all in one kernel.  Ground truth is the eyeball-reviewable
``compute_megamoe_reference`` in ``mega_reference.py``; everything in
this file is glue.

Design choices worth knowing:

  * Inputs are generated deterministically across ranks from a shared
    seed (numpy rng + ``torch.from_numpy(...).cuda()``).  Every rank
    reproduces the full ``(num_ranks, num_tokens_per_rank, *)`` view
    and the ``(num_ranks, num_experts_per_rank, ...)`` weight bank in
    lock-step -- zero NCCL/NVSHMEM traffic during input gen.

  * The reference is computed against the full global view on every
    rank; each rank picks its own ``(T, K, hidden)`` slice for validate.
    ``num_ranks`` times the reference work but immune to host-side
    comm bugs.

  * Workspace partitioning is the kernel's business.  Host asks for
    ``(local_ws_bytes, shared_ws_bytes)`` and hands back two opaque
    byte buffers; kernel does its own region split internally.

Kernel-client contract assumed below (lives in ``megamoe_kernel.py``)::

    class MegaMoEKernel:
        def __init__(self, problem_const, impl_options): ...
        def get_workspace_sizes(self, ...) -> Tuple[int, int]:
            \"\"\"(local_ws_bytes, shared_ws_bytes)\"\"\"
        def __call__(self,
            activation, activation_sf,           # NVFP4 + fp8 plain SF
            topk_idx, topk_weights,
            fc1_weight, fc1_weight_sf,
            fc2_weight, fc2_weight_sf,
            output_activation,                   # (T, hidden) final output
            local_workspace, shared_workspace,   # opaque uint8
            # ``peer_rank_ptr_mapper_host`` carries runtime peer offsets;
            # kernel.__call__ packs it into SymBuffer{world_size}.
            stream): ...
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from itertools import permutations
import numpy as np
import torch
import triton
import triton.language as tl


_FORM_B_PERM_K_LIMIT = 4  # K! perms: K<=4 -> <=24, manageable


@triton.jit
def _form_b_all_perms_kernel(
    src_ptr,  # (T, K, H) bf16, contiguous
    dst_ptr,  # (T, H, n_perms) bf16, contiguous
    perms_ptr,  # (n_perms, K) int32
    T,
    K,
    H,
    BLOCK_H: tl.constexpr,
    K_const: tl.constexpr,
    n_perms_const: tl.constexpr,
):
    """Enumerate every K-axis permutation's bf16 sequential-add result.

    Triton's ``tl.bfloat16`` add lowers to PTX ``add.bf16x2`` on sm_90+,
    which is the same round-to-nearest bf16 op the hardware atomic-add
    path uses internally -- so this kernel's candidates are
    bit-comparable to ``red.global.add.noftz.v2.bf16x2`` observable
    outputs (modulo the atomic-add ordering, which is exactly what we
    enumerate).
    """
    t = tl.program_id(0)
    h_off = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < H
    for p in tl.static_range(n_perms_const):
        acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)
        for k_pos in tl.static_range(K_const):
            k = tl.load(perms_ptr + p * K_const + k_pos)
            v = tl.load(
                src_ptr + t * K * H + k * H + h_off,
                mask=mask,
                other=0,
            ).to(tl.bfloat16)
            acc = acc + v
        tl.store(
            dst_ptr + t * H * n_perms_const + h_off * n_perms_const + p,
            acc,
            mask=mask,
        )


def _form_b_enumerate_perms(ref_K_terms: torch.Tensor) -> torch.Tensor:
    """Return ``(T, H, K!)`` bf16 tensor of every K-axis permutation's
    bf16 sequential-add candidate result.  Hard-rejects ``K > 4`` (K!
    grows factorially; K=4 -> 24 perms is the practical ceiling for
    full ordering enumeration).
    """
    assert ref_K_terms.dtype == torch.bfloat16 and ref_K_terms.is_cuda
    T, K, H = ref_K_terms.shape
    if K > _FORM_B_PERM_K_LIMIT:
        raise ValueError(
            f"_form_b_enumerate_perms: K={K} > {_FORM_B_PERM_K_LIMIT}; "
            f"K! enumeration is intractable beyond K=4."
        )
    perms = list(permutations(range(K)))
    n_perms = len(perms)
    perms_t = torch.tensor(perms, dtype=torch.int32, device=ref_K_terms.device)
    src = ref_K_terms.contiguous()
    dst = torch.empty(
        (T, H, n_perms),
        dtype=torch.bfloat16,
        device=ref_K_terms.device,
    )
    BLOCK_H = 256
    grid = (T, triton.cdiv(H, BLOCK_H))
    _form_b_all_perms_kernel[grid](
        src,
        dst,
        perms_t,
        T,
        K,
        H,
        BLOCK_H=BLOCK_H,
        K_const=K,
        n_perms_const=n_perms,
    )
    return dst


def _quantile_compat(x: torch.Tensor, qs: List[float]) -> List[float]:
    """torch.quantile-equivalent that handles >16M tensors via numpy.

    ``torch.quantile`` errors out with "input tensor is too large" past
    ~2^24 elements (CPU sort backed); ``np.quantile`` is partition-based
    and has no such cap.  Both paths return the same fp64 quantiles.
    """
    if x.numel() <= 16_777_216:
        return (
            torch.quantile(
                x,
                torch.tensor(qs, dtype=x.dtype, device=x.device),
            )
            .cpu()
            .tolist()
        )
    return np.quantile(x.cpu().numpy(), np.asarray(qs)).tolist()


def _form_b_bitwise_match(
    actual_reduced: torch.Tensor,  # (T, H) bf16 or fp32 (bit cast via .to(bf16))
    ref_K_terms: torch.Tensor,  # (T, K, H) bf16
) -> Tuple[torch.Tensor, int]:
    """For each (t, h) cell, return True iff ``actual_reduced[t, h]`` is
    bit-exactly equal to SOME of the K! bf16-sequential-add permutations
    of ``ref_K_terms[t, :, h]``.

    Used by reduce modes (``fc2_transpose_redg`` / ``fc2_ublkredg``) to validate the device-side
    ``red.global.add.noftz.v2.bf16x2`` flow is algorithmically correct
    (modulo non-deterministic atomic-add ordering): every K! permutation
    is a legal observable outcome of K concurrent atomic adds, so actual
    hitting ANY of them proves the only discrepancy from a host fp32 ref
    is ordering, not a real bug.  100% match additionally confirms the
    hardware atomic-add rounds per-step (bf16+bf16->bf16) rather than
    keeping a wider intermediate -- because if it kept fp32 accum, the
    observable result set would NOT be exhausted by bf16-sequential
    candidates.

    Returns ``(match_mask: (T, H) bool, n_perms: int)``.
    """
    candidates = _form_b_enumerate_perms(ref_K_terms)  # (T, H, K!)
    n_perms = candidates.shape[-1]
    actual_bf16 = actual_reduced.to(torch.bfloat16)
    match_mask = (candidates == actual_bf16.unsqueeze(-1)).any(dim=-1)
    return match_mask, n_perms


@triton.jit
def _bf16_seq_sum_k_kernel(
    src_ptr,  # (T, K, H) bf16 contiguous
    dst_ptr,  # (T, H)    bf16 contiguous
    T: tl.constexpr,
    K: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Per-(token, hidden-block) sequential bf16 add over the K axis.

    Mirrors the device-side ``red.global.add.noftz.v2.bf16x2`` precision
    (= bf16 round-to-nearest per add) so the form-B validate comparison
    captures only the atomic-ordering difference, not the host-vs-device
    accumulator precision gap (host fp32 sum would over-estimate ref).
    """
    t = tl.program_id(0)
    h_off = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = h_off < H
    acc = tl.zeros((BLOCK_H,), dtype=tl.bfloat16)
    for k in tl.static_range(K):
        v = tl.load(src_ptr + t * K * H + k * H + h_off, mask=mask, other=0).to(
            tl.bfloat16
        )
        acc = acc + v
    tl.store(dst_ptr + t * H + h_off, acc, mask=mask)


def _bf16_seq_sum_k(src: torch.Tensor) -> torch.Tensor:
    """Triton-backed ``sum(dim=1)`` over (T, K, H) bf16, accumulating in
    bf16 to match device-side atomic-add precision (see kernel docstring).
    """
    assert src.dtype == torch.bfloat16 and src.is_cuda
    T, K, H = src.shape
    src = src.contiguous()
    dst = torch.empty((T, H), dtype=torch.bfloat16, device=src.device)
    BLOCK_H = 256
    grid = (T, triton.cdiv(H, BLOCK_H))
    _bf16_seq_sum_k_kernel[grid](src, dst, T, K, H, BLOCK_H=BLOCK_H)
    return dst


# Ensure absolute package imports work when run as a script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import cutlass

from moe_nvfp4_swapab.epilogue_refactor import (
    SwapABSwigluFp4Epilogue as _SwapABEpilogue,
)
from moe_nvfp4_swapab.mega_reference import compute_megamoe_reference
from src.token_comm import CombineFormat
from common.megamoe_constants import Nvfp4BlockSize
from moe_nvfp4_swapab.runner_common import (
    _DataDtype,
    _Fp4DecodeTable,
    _ScaleDtype,
    _stack_byte_reinterpretable_tensors,
    ceil_div,
    make_nvfp4_tensor_from_torch_rng as _make_nvfp4_tensor_from_torch_rng,
    make_raw_scale_tensor_from_torch_rng as _make_raw_scale_tensor_from_torch_rng,
    round_up,
    to_blocked,
)
from moe_nvfp4_swapab.runner_fc12 import (
    ImplDesc,
    MiscDesc,
)

_SwapABEpilogueTokenTile = _SwapABEpilogue._EpilogueTokenTileSize
_SwapABEpilogueIntermediateAlignment = 2 * _SwapABEpilogue._EpilogueFc1GateUpInterleave


# =============================================================================
# Problem descriptor
# =============================================================================


@dataclass
class TokenCommProblemDesc:
    """Multi-rank, pre-dispatch problem configuration.

    Each rank holds ``num_tokens_per_rank`` input-token slots and the
    routing layer picks ``num_topk`` experts out of ``num_total_experts``.
    Per-rank locality (``num_experts_per_rank = num_total_experts //
    world_size``) is implicit.

    ``num_tokens_per_rank``: input-token slot count per rank, equal across
    all EP ranks.  This is the dispatch-protocol assumption -- the
    per-rank ``input_token_buffer`` has a fixed shape so peer TMA loads
    can use a static descriptor.  In real EP deployments the per-rank
    slot count comes from ``global_batch / DP_size`` (times seq_len if
    no sequence-parallel), which is also equal across EP ranks under
    balanced DP.  Variable-length sequences / drop-token routing are
    handled by setting individual slots' ``topk_idx`` to -1 (masked
    routing) rather than by ragged buffer shapes -- v1 doesn't yet wire
    masked routing through but the buffer layout already accommodates it.

    Contrast ``runner_fc12.ProblemDesc``: that describes the post-dispatch
    world (already-grouped tokens per local expert), and its
    ``balance_route`` flag controls expert-side balance of an
    already-routed stream.  Here, ``route_distribution`` controls how
    the routing table is *generated* in the first place.

    v1 layout / dtype locks (inherited from ``runner_fc12``): NVFP4 data,
    fp8 plain K-major SF on inputs (no atom swizzle at the user boundary
    -- the kernel's dispatch warps swizzle SF internally before fc1 reads
    them), bf16/fp16 ``fc2_output_dtype``.

    TODO: expose ``activation_global_scale`` / ``fc1_weight_global_scale``
    / ``fc2_weight_global_scale`` / ``norm_const`` as fields once they're
    randomized; v1 pins to 1.0 matching ``runner_fc12.py``.
    """

    world_size: int = 0
    num_tokens_per_rank: int = 0
    num_topk: int = 0
    num_total_experts: int = 0
    hidden: int = 0
    intermediate: int = 0
    fc2_output_dtype: torch.dtype = torch.bfloat16
    # Cross-rank combine wire format (drives the host ref's quant round-trip and,
    # once wired, the kernel's fc2 encoder). Default is the bf16 no-quant baseline.
    combine_format: CombineFormat = CombineFormat.parse("bf16")
    # ``balanced``: locally uniform random block assignment, with exact
    # per-expert balance on each full E-token block and randomized tails.
    # ``power_law``: Zipf-rank-frequency, matches the empirical
    # token-to-expert distribution shape reported by the TRT team.
    route_distribution: Literal["balanced", "power_law"] = "balanced"
    # Only consulted when route_distribution == "power_law".  1.0 = classic
    # Zipf (top ~10% experts absorb ~50% tokens); larger = more skewed; 0
    # degenerates to uniform.
    power_law_exponent: float = 1.0
    # DeepSeek-V4 ``config.swiglu_limit``: asymmetric clamp on the real
    # (post-fc1_alpha) gate/up pre-activations before SiLU
    # (``gate=clamp(gate,max=limit)``, ``up=clamp(up,-limit,+limit)``).  Part
    # of the model math, so it lives in the problem desc and drives both the
    # reference and the kernel build.  None disables the clamp.
    gate_up_clamp: Optional[float] = None

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_total_experts // self.world_size

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError(f"world_size must be positive, got {self.world_size}.")
        if self.num_tokens_per_rank <= 0:
            raise ValueError(
                f"num_tokens_per_rank must be positive, got {self.num_tokens_per_rank}."
            )
        if self.num_topk <= 0:
            raise ValueError(f"num_topk must be positive, got {self.num_topk}.")
        if self.num_total_experts <= 0:
            raise ValueError(
                f"num_total_experts must be positive, got {self.num_total_experts}."
            )
        if self.num_total_experts % self.world_size != 0:
            raise ValueError(
                f"num_total_experts ({self.num_total_experts}) must be divisible "
                f"by world_size ({self.world_size})."
            )
        if self.num_topk > self.num_total_experts:
            raise ValueError(
                f"num_topk ({self.num_topk}) > num_total_experts "
                f"({self.num_total_experts}); each token's K slots must select "
                f"distinct experts."
            )
        if self.hidden <= 0 or self.hidden % (2 * Nvfp4BlockSize) != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a positive multiple of "
                f"{2 * Nvfp4BlockSize} (NVFP4 data leg: 4-bit packed yields "
                f"hidden/2 bytes/row; TMA leading-row 16-byte align requires "
                f"hidden % 32 == 0).  The SF leg's stricter %64 alignment "
                f"is NOT enforced at the hidden level: callers may keep "
                f"hidden at %32 granularity and zero-pad activation_sf "
                f"along K_sf to the next multiple of 4 FP8 SFs."
            )
        if self.intermediate <= 0:
            raise ValueError(f"intermediate must be positive, got {self.intermediate}.")
        if self.intermediate % _SwapABEpilogueIntermediateAlignment != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a positive multiple of "
                f"{_SwapABEpilogueIntermediateAlignment} for the current "
                "swapAB epilogue."
            )
        if (self.intermediate // 2) % Nvfp4BlockSize != 0:
            raise ValueError(
                f"intermediate/2 ({self.intermediate // 2}) must be a multiple of "
                f"sf_vec_size ({Nvfp4BlockSize})."
            )
        if self.fc2_output_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                f"fc2_output_dtype must be torch.bfloat16 or torch.float16, "
                f"got {self.fc2_output_dtype}."
            )
        if self.route_distribution not in ("balanced", "power_law"):
            raise ValueError(
                f"route_distribution must be 'balanced' or 'power_law', "
                f"got {self.route_distribution!r}."
            )
        if self.power_law_exponent < 0.0:
            raise ValueError(
                f"power_law_exponent must be non-negative, got "
                f"{self.power_law_exponent}."
            )
        if self.gate_up_clamp is not None and self.gate_up_clamp < 0.0:
            raise ValueError(
                f"gate_up_clamp must be None or non-negative, got {self.gate_up_clamp}."
            )

    def __str__(self) -> str:
        dtype_name = lambda t: str(t).split(".")[-1]
        route_str = self.route_distribution
        if self.route_distribution == "power_law":
            route_str = f"power_law(exponent={self.power_law_exponent:.3g})"
        clamp_str = "off" if self.gate_up_clamp is None else f"{self.gate_up_clamp:.4g}"
        return (
            f"TokenCommProblemDesc: world={self.world_size} "
            f"tokens_per_rank={self.num_tokens_per_rank} topk={self.num_topk} "
            f"experts(total={self.num_total_experts}, "
            f"per_rank={self.num_experts_per_rank}) | "
            f"hidden={self.hidden} intermediate={self.intermediate} | "
            f"fc2_output_dtype={dtype_name(self.fc2_output_dtype)} | "
            f"combine={self.combine_format} | "
            f"route={route_str} | swiglu_clamp={clamp_str}"
        )


# =============================================================================
# Routing-table generators (vectorized, deterministic)
# =============================================================================


def _generate_topk_idx_balanced(
    num_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_total_experts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Locally uniform random routing over E-token blocks.

    Each source rank is partitioned into padded blocks of ``num_total_experts``
    consecutive tokens.  Within every block, each top-k column traverses a
    random permutation of all experts exactly once, shifted by a distinct
    per-column offset.  This makes every token's K experts distinct while
    keeping the source-rank-to-expert and source-rank-to-target-rank traffic
    balanced in each local block.

    Properties:
      - The K experts per token are pairwise distinct.
      - Every full block contributes exactly ``num_topk`` tokens to every
        expert from each source rank.
      - Non-divisible token counts are handled by generating one padded block
        and truncating it, so tail imbalance is random instead of structural.
    """
    assert num_topk <= num_total_experts, (
        f"num_topk ({num_topk}) > num_total_experts ({num_total_experts})"
    )
    num_padded_tokens_per_rank = (
        (num_tokens_per_rank + num_total_experts - 1)
        // num_total_experts
        * num_total_experts
    )
    num_blocks = num_padded_tokens_per_rank // num_total_experts

    expert_permutations = rng.random(
        (num_ranks, num_blocks, num_total_experts)
    ).argsort(axis=-1)
    topk_offsets = rng.random((num_ranks, num_blocks, num_total_experts)).argsort(
        axis=-1
    )[..., :num_topk]

    token_offsets = np.arange(num_total_experts)[None, None, :, None]
    expert_indices = (token_offsets + topk_offsets[:, :, None, :]) % num_total_experts
    topk_blocks = np.take_along_axis(
        expert_permutations[..., None],
        expert_indices,
        axis=2,
    )
    return topk_blocks.reshape(
        num_ranks,
        num_padded_tokens_per_rank,
        num_topk,
    )[:, :num_tokens_per_rank, :].astype(np.int64)


def _generate_topk_idx_power_law(
    num_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_total_experts: int,
    exponent: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Power-law (Zipf-rank-frequency) routing.

    Matches the token-to-expert distribution shape observed by the TRT
    team: the expert with popularity rank ``r`` (1-indexed) receives
    probability proportional to ``1 / r ** exponent``.  ``exponent = 1.0``
    is the classic Zipf law (top ~10% experts absorb ~50% of tokens);
    larger exponents skew further; ``exponent = 0`` degenerates to uniform.

    A random permutation of expert ids carries the popularity rank, so
    "which experts are hot" varies with the seed instead of always
    being ``expert_id 0..N``.

    Vectorized via Gumbel-top-K: ``argpartition(log p + Gumbel)`` over
    the expert axis is statistically equivalent to
    ``rng.choice(p=p, replace=False)`` but runs as one numpy op instead
    of ``num_ranks * num_tokens_per_rank`` Python iterations.
    """
    assert num_topk <= num_total_experts, (
        f"num_topk ({num_topk}) > num_total_experts ({num_total_experts})"
    )
    popularity_rank_to_expert = rng.permutation(num_total_experts)
    rank_freq = 1.0 / (np.arange(num_total_experts) + 1).astype(np.float64) ** exponent
    probs = np.empty(num_total_experts, dtype=np.float64)
    probs[popularity_rank_to_expert] = rank_freq / rank_freq.sum()

    log_probs = np.log(np.maximum(probs, 1e-30))
    num_token_slots = num_ranks * num_tokens_per_rank
    uniform_noise = rng.random(size=(num_token_slots, num_total_experts))
    gumbel_noise = -np.log(-np.log(np.maximum(uniform_noise, 1e-30)) + 1e-30)
    scores = log_probs[None, :] + gumbel_noise
    # ``argpartition`` returns the top-K indices in unspecified order; routing
    # correctness doesn't depend on intra-token order, so this is fine.
    topk_flat = np.argpartition(-scores, kth=num_topk - 1, axis=-1)[:, :num_topk]
    return topk_flat.reshape(num_ranks, num_tokens_per_rank, num_topk).astype(np.int64)


def _get_remote_rank_comm_matrices(
    topk_idx: np.ndarray,
    world_size: int,
    num_total_experts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return remote dispatch and fc2-return routed-token count matrices.

    Unit is routed token after topk expansion: one original token with K
    selected experts contributes K routed-token counts.  ``dispatch[src, dst]``
    counts routed tokens whose expert is owned by ``dst`` and whose source
    token lives on ``src``.  ``fc2_return`` is the reverse direction for the
    same routed tokens, so it is the transpose of dispatch at count
    granularity.  Local entries are zeroed because they do not contribute
    inter-rank traffic.
    """
    if num_total_experts % world_size != 0:
        raise ValueError(
            f"num_total_experts ({num_total_experts}) must be divisible by "
            f"world_size ({world_size})."
        )
    if topk_idx.ndim != 3 or topk_idx.shape[0] != world_size:
        raise ValueError(
            f"topk_idx must have shape (world_size, tokens_per_rank, topk), "
            f"got {topk_idx.shape} for world_size={world_size}."
        )

    num_experts_per_rank = num_total_experts // world_size
    valid = topk_idx >= 0
    src_ranks = np.broadcast_to(
        np.arange(world_size, dtype=np.int64)[:, None, None],
        topk_idx.shape,
    )[valid]
    dst_ranks = (topk_idx[valid] // num_experts_per_rank).astype(np.int64)

    dispatch = np.zeros((world_size, world_size), dtype=np.int64)
    np.add.at(dispatch, (src_ranks, dst_ranks), 1)
    np.fill_diagonal(dispatch, 0)
    return dispatch, dispatch.T.copy()


def _print_remote_rank_comm_matrices(
    topk_idx: np.ndarray,
    world_size: int,
    num_total_experts: int,
) -> None:
    """Print remote rank-to-rank routed-token counts for route inspection."""
    dispatch, fc2_return = _get_remote_rank_comm_matrices(
        topk_idx,
        world_size,
        num_total_experts,
    )

    def _print_matrix(title: str, matrix: np.ndarray) -> None:
        col_width = 12
        print(f"{title} [unit: routed tokens after topk count]")
        src_dst_label = "src\\dst"
        header = f"{src_dst_label:>8}" + "".join(
            f"{f'dst{dst}':>{col_width}}" for dst in range(world_size)
        )
        print(header)
        for src in range(world_size):
            row = "".join(
                f"{int(matrix[src, dst]):>{col_width}}" for dst in range(world_size)
            )
            print(f"{f'src{src}':>8}{row}")
        print(
            f"  row_sums_routed_tokens={matrix.sum(axis=1).astype(np.int64).tolist()} "
            f"col_sums_routed_tokens={matrix.sum(axis=0).astype(np.int64).tolist()} "
            f"total_remote_routed_tokens={int(matrix.sum())}"
        )

    print(
        "---- remote rank traffic from topk_idx "
        "[unit: routed tokens after topk count] ----"
    )
    _print_matrix("dispatch input tokens: source rank -> expert-owner rank", dispatch)
    _print_matrix("fc2 return tokens: expert-owner rank -> source rank", fc2_return)


def _generate_topk_weights(
    num_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    rng: torch.Generator,
) -> torch.Tensor:
    """Per-(rank, token, slot) routing weight uniformly in ``[0.5, 1.5]``.

    NOT a faithful softmax output (those sum to 1 along the K axis); this
    is just a moderate-magnitude positive value picked to keep
    ``swiglu * topk_weight`` within fp8 SF range.  Reference and kernel
    see the same tensor so the absolute scale doesn't affect correctness,
    only the local numerical regime.
    """
    return (
        torch.rand(
            (num_ranks, num_tokens_per_rank, num_topk),
            dtype=torch.float32,
            device="cuda",
            generator=rng,
        )
        + 0.5
    )


# =============================================================================
# NVFP4 / SF tensor builders (host-side numpy rng -> cuda)
# =============================================================================


# =============================================================================
# Symmetric-heap allocation
# =============================================================================


# MEGA_NO_DIST=1: opt out of torch.distributed + NVSHMEM init so the
# runner can be launched as plain ``python mega_runner.py`` (e.g. under
# compute-sanitizer).  Single-rank only; sym tensors degrade to plain
# CUDA ``torch.zeros`` and the SymBuffer offsets list is hard-coded to
# ``(0,) * world_size``.
_NO_DIST: bool = bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def _sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    """Zero-initialised symmetric-heap tensor.

    Restricted to dtypes nvshmem4py natively supports (``uint8``, ``int32``,
    ``int64``, ``float32``, ``bfloat16``, ``float16``).  NVFP4 / fp8 go
    through the byte-buf reinterpret helpers below.

    Caller is responsible for freeing via ``nvshmem.core.free_tensor``
    before ``nvshmem.core.finalize()``.
    """
    if _NO_DIST:
        return torch.zeros(shape, dtype=dtype, device="cuda")
    import nvshmem.core

    tensor = nvshmem.core.tensor(shape, dtype=dtype)
    tensor.zero_()
    return tensor


def _sym_zeros_byte_view(
    logical_shape: Tuple[int, ...],
    target_dtype: torch.dtype,
) -> torch.Tensor:
    """Sym-heap allocation for dtypes nvshmem4py doesn't natively support
    (NVFP4 / fp8): allocate uint8 byte buffer, ``.view(target_dtype)``.

    NVFP4 halves the last logical dim (2 fp4 elements packed per byte);
    fp8 keeps shape as-is.  Returns a tensor whose ``.shape`` is the
    storage shape -- callers that want a different packed dim should
    permute after this returns.
    """
    if target_dtype == _DataDtype:
        if not logical_shape or logical_shape[-1] % 2 != 0:
            raise ValueError(
                f"NVFP4 sym view needs non-empty logical_shape with even last dim, "
                f"got {logical_shape}."
            )
        storage_shape = (*logical_shape[:-1], logical_shape[-1] // 2)
    elif target_dtype == _ScaleDtype:
        storage_shape = tuple(logical_shape)
    else:
        raise ValueError(
            f"_sym_zeros_byte_view: dtype must be {_DataDtype} or {_ScaleDtype}, "
            f"got {target_dtype}."
        )
    total_bytes = 1
    for dim_size in storage_shape:
        total_bytes *= dim_size
    return (
        _sym_zeros((total_bytes,), torch.uint8)
        .view(target_dtype)
        .reshape(storage_shape)
    )


def _compute_peer_offsets(
    sym_tensor: torch.Tensor, world_size: int
) -> Tuple[int, Tuple[int, ...]]:
    """Host-side peer-delta table for one sym-heap allocation.

    Returns ``(local_base, peer_offsets_list)`` where:
      * ``local_base`` = ``int(sym_tensor.data_ptr())`` -- the local
        rank's symmetric-heap base byte address.
      * ``peer_offsets_list[r] = peer_r_base - local_base`` in bytes,
        as a length-``world_size`` ``Tuple[int, ...]``.  Per NVSHMEM
        convention, ``peer_offsets_list[local_rank] == 0``.

    NVSHMEM keeps every rank's symmetric heap laid out identically (same
    allocation order -> same offsets), so this single per-peer constant
    works for every sub-allocation in the heap.  These ints are passed
    through ``SymBufferHost`` as runtime launch args; kernel.__call__
    packs them into ``SymBuffer{world_size}`` before entering device
    code.  The offsets land in the kernel parameter bank, and a
    runtime-indexed peer lookup lowers to a single ``ld.param.b64`` (=
    ``LDC.U64``) with no GMEM traffic at all.

    The legacy version returned a ``(world_size,) int64`` device tensor
    to obtain an ``LDC.E.64`` indexed load via an extra GMEM step; the
    byval ``SymBuffer`` removes that indirection entirely (one LDC
    against the param bank, zero GMEM).
    """
    if _NO_DIST:
        local_base = int(sym_tensor.data_ptr())
        return local_base, tuple(0 for _ in range(world_size))
    import nvshmem.core

    local_base = int(sym_tensor.data_ptr())
    peer_offsets_list = tuple(
        int(nvshmem.core.get_peer_tensor(sym_tensor, peer).data_ptr()) - local_base
        for peer in range(world_size)
    )
    return local_base, peer_offsets_list


# =============================================================================
# Tester
# =============================================================================


class MegaMoETester:
    """Multi-rank MegaMoE host driver.

    Lifecycle:

      1. ``generate_inputs``: deterministically reproduce the global view
         on EVERY rank from ``misc.seed``; slice own-rank into the sym
         heap.  Weights stay on regular cuda (no cross-rank access).
      2. ``compute_reference``: run ``compute_megamoe_reference`` on the
         global view; slice own ``(T, K, hidden)`` into ``combine_output_ref``.
      3. ``allocate_workspaces``: ``kernel.get_workspace_sizes()`` ->
         allocate local (cuda) + shared (sym) byte buffers; derive the
         ``(symmetric_base, peer_offsets_list)`` host-side ints used to
         build the kernel-side ``SymBuffer`` (covers every sym sub-region).
      4. ``run_kernel``: placeholder until the kernel side is wired.
      5. ``validate``: bf16-grade compare own-rank ``combine_output``
         against ``combine_output_ref``.

    None of steps 1-3 emit NCCL/NVSHMEM traffic; only step 4 (kernel
    launch) and step 5's compare touch sym memory.
    """

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
    ) -> None:
        self.problem = problem
        self.impl = impl
        self.misc = misc
        self.rank = rank
        if impl.fc2_reduces_topk and misc.ref_compute_graph != "deepgemm":
            raise ValueError(
                "in-kernel fc2 reduce requires ref_compute_graph='deepgemm'; "
                f"got {misc.ref_compute_graph!r}.  The REDG path can only "
                "atomic-add terms whose topk score was already absorbed before "
                "fc2."
            )
        # Single source of truth for world size; ``local_rank`` was only
        # needed for the upstream ``torch.cuda.set_device`` inside the
        # NCCL+NVSHMEM bootstrap, the tester itself doesn't reference it.
        self.world_size = problem.world_size

        torch.manual_seed(misc.seed)
        np.random.seed(misc.seed)
        self._np_rng: np.random.Generator = np.random.default_rng(misc.seed)
        self._torch_cuda_rng = torch.Generator(device="cuda")
        self._torch_cuda_rng.manual_seed(misc.seed)

        # Own-rank inputs on the symmetric heap (peer-pulled by dispatch).
        self.my_activation: Optional[torch.Tensor] = None
        self.my_activation_sf: Optional[torch.Tensor] = None
        self.my_topk_idx: Optional[torch.Tensor] = None
        self.my_topk_weights: Optional[torch.Tensor] = None

        # Own-rank weights on regular cuda (no cross-rank access).
        self.my_fc1_weight: Optional[torch.Tensor] = None
        self.my_fc1_weight_sf: Optional[torch.Tensor] = None
        self.my_fc2_weight: Optional[torch.Tensor] = None
        self.my_fc2_weight_sf: Optional[torch.Tensor] = None

        # Full global view (regular cuda) -- kept so ``compute_reference``
        # can feed it without re-rolling the rng.
        self._global_activation: Optional[torch.Tensor] = None
        self._global_activation_sf: Optional[torch.Tensor] = None
        self._global_topk_idx: Optional[torch.Tensor] = None
        self._global_topk_weights: Optional[torch.Tensor] = None
        self._global_fc1_weight: Optional[torch.Tensor] = None
        self._global_fc1_weight_sf: Optional[torch.Tensor] = None
        self._global_fc2_weight: Optional[torch.Tensor] = None
        self._global_fc2_weight_sf: Optional[torch.Tensor] = None
        self._global_fc1_alpha: Optional[torch.Tensor] = None
        self._global_fc2_alpha: Optional[torch.Tensor] = None
        self._global_fc1_norm_const: Optional[torch.Tensor] = None

        # Public 2D (T, hidden) combined output the kernel produces.
        self.output_activation: Optional[torch.Tensor] = None
        self.combine_output_ref: Optional[torch.Tensor] = None
        self.combine_reduced_output_ref: Optional[torch.Tensor] = None

        self.local_workspace: Optional[torch.Tensor] = None
        self.shared_workspace: Optional[torch.Tensor] = None
        # Host-side SymBuffer payload.  It is passed as a runtime arg via
        # SymBufferHost and packed into SymBuffer{world_size} in
        # kernel.__call__.  No GMEM tensor is allocated for these offsets.
        self.symmetric_base: Optional[int] = None
        self.peer_offsets_list: Optional[Tuple[int, ...]] = None

        self._kernel = None
        self._compiled_kernel = None
        self._use_torch_profiler = False
        self._perf_warmup = 1
        self._perf_iters = 1

        # v1 pinned expert-wise epilogue args.
        self.my_fc1_alpha: Optional[torch.Tensor] = None
        self.my_fc2_alpha: Optional[torch.Tensor] = None
        self.my_fc1_norm_const: Optional[torch.Tensor] = None

    def set_torch_profiler_enabled(self, enabled: bool) -> None:
        """Enable optional torch profiler timing for the target kernel launch."""
        self._use_torch_profiler = enabled

    def set_perf_iters(self, warmup: int, iters: int) -> None:
        """Configure steady-state timing: `warmup` untimed launches then
        `iters` back-to-back timed launches (torch-profiler path only)."""
        self._perf_warmup = max(0, int(warmup))
        self._perf_iters = max(1, int(iters))

    def _check_cuda_rng_consistency(self) -> None:
        """Weakly verify CUDA RNG streams stayed aligned across ranks."""
        if not (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        ):
            return

        random_parts = torch.randint(
            0,
            1 << 31,
            (2, 16),
            dtype=torch.int64,
            device="cuda",
            generator=self._torch_cuda_rng,
        )
        sentinel = (random_parts[0] << 31) ^ random_parts[1]
        gathered = [torch.empty_like(sentinel) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, sentinel)

        reference = gathered[0]
        mismatched_ranks = [
            rank
            for rank, value in enumerate(gathered)
            if not torch.equal(value, reference)
        ]
        if mismatched_ranks:
            raise RuntimeError(
                f"CUDA RNG state mismatch across ranks after generate_inputs; "
                f"mismatched ranks: {mismatched_ranks}."
            )

    # ------------------------------------------------------------------
    # Step 1: deterministic input + weight generation
    # ------------------------------------------------------------------

    def generate_inputs(self) -> None:
        problem = self.problem
        rng = self._np_rng
        num_ranks = problem.world_size
        num_tokens_per_rank = problem.num_tokens_per_rank
        num_topk = problem.num_topk
        hidden = problem.hidden
        intermediate = problem.intermediate
        intermediate_downproj = intermediate // 2
        num_experts_per_rank = problem.num_experts_per_rank
        scale_blocksize = Nvfp4BlockSize
        hidden_sf_cols = ceil_div(hidden, scale_blocksize)
        intermediate_downproj_sf_cols = ceil_div(intermediate_downproj, scale_blocksize)

        # ---- Activation (NVFP4 packed in hidden) + plain K-major SF.
        # SF lives in the user-facing plain layout -- the kernel's dispatch
        # warps re-stage it into the SFB atom layout inside l1_sf_buffer
        # before fc1 reads it; the boundary stays plain so the host
        # reference can dequant directly.
        self._global_activation = _make_nvfp4_tensor_from_torch_rng(
            self._torch_cuda_rng,
            (num_ranks, num_tokens_per_rank, hidden),
            packed_dim=-1,
            perf_run=self.misc.perf_run,
        )
        self._global_activation_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_tokens_per_rank,
            hidden,
            blocksize=scale_blocksize,
            strict=True,
        ).reshape(num_ranks, num_tokens_per_rank, hidden_sf_cols)

        # ---- Routing table.
        if problem.route_distribution == "balanced":
            topk_idx_np = _generate_topk_idx_balanced(
                num_ranks,
                num_tokens_per_rank,
                num_topk,
                problem.num_total_experts,
                rng,
            )
        else:
            topk_idx_np = _generate_topk_idx_power_law(
                num_ranks,
                num_tokens_per_rank,
                num_topk,
                problem.num_total_experts,
                problem.power_law_exponent,
                rng,
            )
        topk_weights = _generate_topk_weights(
            num_ranks,
            num_tokens_per_rank,
            num_topk,
            self._torch_cuda_rng,
        )
        if self.rank == 0:
            _print_remote_rank_comm_matrices(
                topk_idx_np,
                num_ranks,
                problem.num_total_experts,
            )
        self._global_topk_idx = torch.from_numpy(topk_idx_np).cuda()
        self._global_topk_weights = topk_weights

        # ---- Weights.  fc1 packs along hidden (K of fc1); fc2 packs along
        # intermediate_downproj (K of fc2).
        self._global_fc1_weight = _make_nvfp4_tensor_from_torch_rng(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, hidden, intermediate),
            packed_dim=2,
            perf_run=self.misc.perf_run,
        )
        self._global_fc1_weight_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * intermediate,
            hidden,
            blocksize=scale_blocksize,
            strict=True,
        ).reshape(num_ranks, num_experts_per_rank, intermediate, hidden_sf_cols)

        self._global_fc2_weight = _make_nvfp4_tensor_from_torch_rng(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, intermediate_downproj, hidden),
            packed_dim=2,
            perf_run=self.misc.perf_run,
        )
        self._global_fc2_weight_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * hidden,
            intermediate_downproj,
            blocksize=scale_blocksize,
            strict=True,
        ).reshape(
            num_ranks,
            num_experts_per_rank,
            hidden,
            intermediate_downproj_sf_cols,
        )
        epi_arg_shape = (num_ranks, num_experts_per_rank)
        self._global_fc1_alpha = (
            torch.randint(
                1,
                5,
                epi_arg_shape,
                generator=self._torch_cuda_rng,
                device="cuda",
            ).to(torch.float32)
            * 0.5
        )
        self._global_fc2_alpha = (
            torch.randint(
                1,
                5,
                epi_arg_shape,
                generator=self._torch_cuda_rng,
                device="cuda",
            ).to(torch.float32)
            * 0.5
        )
        self._global_fc1_norm_const = (
            torch.randint(
                2,
                5,
                epi_arg_shape,
                generator=self._torch_cuda_rng,
                device="cuda",
            ).to(torch.float32)
            * 0.5
        )

        # ---- Atom-swizzle the weight SFs.  fc1_weight / fc2_weight are
        # consumed by the base kernel directly (NOT routed through dispatch
        # like the activation SF), so the host has to feed them in the
        # 32x4x4 atom-swizzled, flat 2D ``(experts, flat_atom_size)`` layout
        # that ``tile_atom_to_shape_SF`` and the TMA SFA descriptor expect.
        # Mirrors the lean ``runner_fc12``'s ``assemble_raw_scales_stacked_expert``
        # pipeline.  Plain ``_global_fc1/2_weight_sf`` are kept untouched for
        # ``compute_megamoe_reference`` (which dequantizes from the raw plain
        # layout).
        fc1_sf_swizzled = [
            to_blocked(self._global_fc1_weight_sf[r, e])
            for r in range(num_ranks)
            for e in range(num_experts_per_rank)
        ]
        fc1_flat_sf_size = fc1_sf_swizzled[0].numel()
        self._global_fc1_weight_sf_swizzled = _stack_byte_reinterpretable_tensors(
            fc1_sf_swizzled, dim=0
        ).view(num_ranks, num_experts_per_rank, fc1_flat_sf_size)

        fc2_sf_swizzled = [
            to_blocked(self._global_fc2_weight_sf[r, e])
            for r in range(num_ranks)
            for e in range(num_experts_per_rank)
        ]
        fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
        self._global_fc2_weight_sf_swizzled = _stack_byte_reinterpretable_tensors(
            fc2_sf_swizzled, dim=0
        ).view(num_ranks, num_experts_per_rank, fc2_flat_sf_size)

        # ---- Stage own-rank inputs into the symmetric heap.
        own_activation = self._global_activation[self.rank]
        own_activation_sf = self._global_activation_sf[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        # NVFP4 / fp8 go through uint8 byte-buf views (nvshmem4py doesn't
        # natively support these dtypes); copy via uint8 to dodge any
        # NVFP4-to-NVFP4 assignment quirks between allocators.
        self.my_activation = _sym_zeros_byte_view(
            (num_tokens_per_rank, hidden),
            _DataDtype,
        )
        self.my_activation.view(torch.uint8).copy_(own_activation.view(torch.uint8))

        # SF leg caller contract: the K_sf axis (= ``ceil(hidden, 16)``)
        # must be a multiple of 4 fp8 SFs so dispatch_pull's LDG.32 byte
        # stride (= ``sf_uint32_per_token * 4 bytes/token``) matches the
        # host row stride.  Pad-to-multiple-of-4 + zero-fill on the host
        # side; the kernel never reads past ``ceil(hidden/16)`` valid SFs
        # because the trailing K_sf elements are paired with NVFP4 data
        # entries that fall in TMA's OOB-fill-0 region.
        #
        # ``_sym_zeros_byte_view`` already zero-inits the symmetric heap
        # allocation, so we only need to copy the first ``hidden_sf_cols``
        # uint8 columns; the trailing padded columns remain at zero.  Use
        # the uint8 byte view to dodge fp8 dtype's spotty support for
        # tensor-indexing assignment across torch versions.
        hidden_sf_cols_padded = round_up(hidden_sf_cols, 4)
        self.my_activation_sf = _sym_zeros_byte_view(
            (num_tokens_per_rank, hidden_sf_cols_padded),
            _ScaleDtype,
        )
        self.my_activation_sf.view(torch.uint8)[:, :hidden_sf_cols].copy_(
            own_activation_sf.view(torch.uint8)
        )

        self.my_topk_idx = _sym_zeros(tuple(own_topk_idx.shape), torch.int64)
        self.my_topk_idx.copy_(own_topk_idx)

        self.my_topk_weights = _sym_zeros(tuple(own_topk_weights.shape), torch.float32)
        self.my_topk_weights.copy_(own_topk_weights)

        # ---- Own-rank weights stay on regular cuda.  DO NOT ``.contiguous()``:
        # ``_make_nvfp4_tensor_from_torch_rng`` uses an internal permute to put
        # the packed dim mid-tensor; ``.contiguous()`` would re-pack to
        # row-major and break the K-as-stride-1 invariant the dequant +
        # GEMM path depends on.  Leading-dim slicing keeps the existing
        # stride pattern.
        self.my_fc1_weight = self._global_fc1_weight[self.rank]
        # Kernel consumes atom-swizzled (E, flat_atom_size); reference path
        # (``compute_megamoe_reference``) still uses the plain
        # ``_global_fc1_weight_sf`` directly to dequantize.
        self.my_fc1_weight_sf = self._global_fc1_weight_sf_swizzled[self.rank]
        self.my_fc2_weight = self._global_fc2_weight[self.rank]
        self.my_fc2_weight_sf = self._global_fc2_weight_sf_swizzled[self.rank]
        self.my_fc1_alpha = self._global_fc1_alpha[self.rank]
        self.my_fc2_alpha = self._global_fc2_alpha[self.rank]
        self.my_fc1_norm_const = self._global_fc1_norm_const[self.rank]

        # ---- Public 2D (T, hidden) combined output.  The kernel owns the
        # per-topk (T, K, H) combine staging internally; the caller only
        # provides/consumes the final reduced result.  Placement depends on the
        # reduce mode:
        #   * in_kernel_reduce: this IS the cross-rank ``red.global.add.v2.bf16x2``
        #     target, so it MUST live on the sym heap; ``_sym_zeros`` also gives
        #     the atomic-add-accumulate caller contract (start from zero).
        #   * separate_kernel_reduce: peers write the internal staging instead;
        #     this only receives the local tail reduce, so allocate it locally
        #     (avoids any symmetric-heap caching ambiguity, matches deployment).
        if self.impl.fc2_reduces_topk:
            self.output_activation = _sym_zeros(
                (num_tokens_per_rank, hidden),
                problem.fc2_output_dtype,
            )
        else:
            if problem.fc2_output_dtype != torch.bfloat16:
                raise ValueError(
                    "separate_kernel_reduce currently expects BF16 output, "
                    f"got {problem.fc2_output_dtype}."
                )
            self.output_activation = torch.empty(
                (num_tokens_per_rank, hidden),
                dtype=problem.fc2_output_dtype,
                device="cuda",
            )

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: reference
    # ------------------------------------------------------------------

    def compute_reference(self) -> None:
        if self.misc.skip_ref_check:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        reference = compute_megamoe_reference(
            input_activation=self._global_activation,
            input_activation_sf=self._global_activation_sf,
            input_topk_idx=self._global_topk_idx,
            input_topk_weights=self._global_topk_weights,
            fc1_weight=self._global_fc1_weight,
            fc1_weight_sf=self._global_fc1_weight_sf,
            fc2_weight=self._global_fc2_weight,
            fc2_weight_sf=self._global_fc2_weight_sf,
            fc1_alpha=self._global_fc1_alpha,
            fc2_alpha=self._global_fc2_alpha,
            fc1_norm_const=self._global_fc1_norm_const,
            ref_compute_graph=self.misc.ref_compute_graph,
            combine_format=self.problem.combine_format,
            gate_up_clamp=self.problem.gate_up_clamp,
        )
        self.combine_output_ref = reference.combine_output[self.rank].contiguous()
        self.combine_reduced_output_ref = reference.combine_reduced_output[
            self.rank
        ].contiguous()

    # ------------------------------------------------------------------
    # Step 3: workspace allocation
    # ------------------------------------------------------------------

    def allocate_workspaces(self) -> None:
        """Allocate the two opaque byte buffers the kernel partitions internally.

        ``local_workspace`` lives in plain CUDA device memory (no peer
        access; counter / fc1_output / pool slots etc. are all local).
        ``shared_workspace`` lives on the NVSHMEM symmetric heap so the
        dispatch warps' ``peer_rank_ptr_mapper.map`` writes can reach peer ranks.
        ``(symmetric_base, peer_offsets_list)`` is computed once from the
        symmetric workspace's base address -- NVSHMEM guarantees the same
        delta works for every sub-allocation in the heap, so the same
        table also covers ``my_activation`` / ``my_activation_sf`` /
        ``my_topk_weights`` / ``combine_output``.  ``run_kernel`` passes
        these ints through ``SymBufferHost`` so kernel.__call__ can pack
        them into SymBuffer{world_size}.

        Caller invariant: ``self._kernel`` must already be instantiated
        (``run_kernel`` is responsible for that step before invoking us).
        """
        if self._kernel is None:
            raise RuntimeError(
                "allocate_workspaces called before self._kernel was "
                "instantiated; run_kernel must create the kernel first."
            )

        local_ws_bytes, shared_ws_bytes = self._kernel.get_workspace_sizes()

        self.local_workspace = torch.zeros(
            (local_ws_bytes,),
            dtype=torch.uint8,
            device="cuda",
        )
        self.shared_workspace = _sym_zeros((shared_ws_bytes,), torch.uint8)
        self.symmetric_base, self.peer_offsets_list = _compute_peer_offsets(
            self.shared_workspace,
            self.world_size,
        )

    # ------------------------------------------------------------------
    # Step 4: kernel launch
    # ------------------------------------------------------------------

    @staticmethod
    def _profiler_event_cuda_time_us(event) -> float:
        """Return CUDA/device time in microseconds from a torch profiler event."""
        for attr_name in ("device_time_total", "cuda_time_total"):
            value = getattr(event, attr_name, None)
            if value is not None:
                return float(value)
        return 0.0

    def _report_torch_profiler_kernel_time(self, prof, num_iters: int = 1) -> None:
        """Gather and print per-rank ``kernel_cutlass*`` CUDA times.

        ``prof.events()`` aggregates the CUDA time over all occurrences of
        each event, so when the timed region launched the pipeline
        ``num_iters`` times the sums are divided down to a per-launch time.
        ``mega`` and ``topk_reduce`` are reported separately.
        """
        mega_time_us = 0.0
        topk_time_us = 0.0
        matched_event_names = []
        for event in prof.events():
            event_name = getattr(event, "key", "")
            if not event_name.startswith("kernel_cutlass"):
                continue
            event_time_us = self._profiler_event_cuda_time_us(event)
            matched_event_names.append(event_name)
            if event_name.startswith("kernel_cutlass__reduce_"):
                topk_time_us += event_time_us
            else:
                mega_time_us += event_time_us

        if not matched_event_names:
            mega_time_us = float("nan")
            topk_time_us = float("nan")
        else:
            # Aggregate sums cover ``num_iters`` back-to-back launches in the
            # timed region; divide down to a per-launch (steady-state) time.
            mega_time_us /= max(1, num_iters)
            topk_time_us /= max(1, num_iters)
            if mega_time_us == 0.0:
                mega_time_us = float("nan")
            if topk_time_us == 0.0:
                topk_time_us = float("nan")
        finite_parts = [
            time_us for time_us in (mega_time_us, topk_time_us) if np.isfinite(time_us)
        ]
        total_time_us = sum(finite_parts) if finite_parts else float("nan")

        dist_active = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if dist_active:
            local_times = torch.tensor(
                [mega_time_us, topk_time_us, total_time_us],
                dtype=torch.float64,
                device="cuda",
            )
            gathered_times = [
                torch.empty_like(local_times) for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_times, local_times)
            rank_times_us = [
                [float(value) for value in rank_time.tolist()]
                for rank_time in gathered_times
            ]
        else:
            rank_times_us = [[mega_time_us, topk_time_us, total_time_us]]

        if self.rank == 0:
            mega_values = [rank_times[0] for rank_times in rank_times_us]
            finite_mega = [
                (rank, time_us)
                for rank, time_us in enumerate(mega_values)
                if np.isfinite(time_us)
            ]
            if finite_mega:
                min_rank, min_mega_us = min(finite_mega, key=lambda x: x[1])
                min_topk_us = rank_times_us[min_rank][1]
                min_total_us = rank_times_us[min_rank][2]
            else:
                min_rank = -1
                min_mega_us = float("nan")
                min_topk_us = float("nan")
                min_total_us = float("nan")

            def _fmt(time_us: float) -> str:
                return f"{time_us:.2f} us" if np.isfinite(time_us) else "n/a"

            print("---- torch profiler target pipeline CUDA time ----")
            print(
                f"  min_rank_by_mega=rank {min_rank}: "
                f"mega={_fmt(min_mega_us)} "
                f"topk_reduce={_fmt(min_topk_us)} "
                f"total={_fmt(min_total_us)}"
            )
            print("  mega:")
            for rank, times in enumerate(rank_times_us):
                print(f"    rank_{rank}: {_fmt(times[0])}")
            has_any_topk = any(np.isfinite(times[1]) for times in rank_times_us)
            if has_any_topk:
                print("  topk:")
                for rank, times in enumerate(rank_times_us):
                    print(f"    rank_{rank}: {_fmt(times[1])}")
            if not matched_event_names:
                print("  warning: no kernel_cutlass* events matched")

    def _launch_target_kernels_with_optional_torch_profiler(
        self,
        runtime_kwargs,
    ) -> None:
        """Launch the target kernel pipeline, optionally under torch profiler.

        The compiled kernel now owns the separate-kernel-reduce tail internally
        (launched on the same stream inside ``__call__``), so this is a single
        compiled-kernel call; the reduce still shows up as its own
        ``kernel_cutlass_topk_reduce_*`` device event in the profiler.
        """
        if self._compiled_kernel is None:
            raise RuntimeError("compiled kernel is not available")

        def _launch() -> None:
            self._compiled_kernel(**runtime_kwargs)

        if not self._use_torch_profiler:
            _launch()
            torch.cuda.synchronize()
            return

        for _ in range(self._perf_warmup):
            self._compiled_kernel(**runtime_kwargs)
        torch.cuda.synchronize()

        # Timed: `_perf_iters` back-to-back launches, no host sync between them.
        n_iters = max(1, self._perf_iters)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
        ) as prof:
            for _ in range(n_iters):
                _launch()
            torch.cuda.synchronize()
        self._report_torch_profiler_kernel_time(prof, num_iters=n_iters)

    def run_kernel(self) -> None:
        """Compile + launch ``Sm100MegaMoEKernel`` on the current cuda stream.

        Steps (lazy-imported so the harness paths -- ``generate_inputs`` +
        ``compute_reference`` -- can still be exercised on machines without
        a working cute install):

          1. Instantiate the kernel with the static expert shape derived
             from ``ProblemDesc`` and the impl-side tile config.
          2. Allocate the opaque local + shared workspaces via
             ``self.allocate_workspaces()``.
          3. Convert every host-side torch tensor into a cute.Tensor with
             ``cutlass.torch.from_dlpack`` + ``mark_layout_dynamic``.
          4. ``cute.compile`` once.
          5. Launch on the current cuda stream.
        """
        if (
            self.my_activation is None
            or self.my_activation_sf is None
            or self.my_topk_idx is None
            or self.my_topk_weights is None
            or self.my_fc1_weight is None
            or self.my_fc1_weight_sf is None
            or self.my_fc2_weight is None
            or self.my_fc2_weight_sf is None
            or self.my_fc1_alpha is None
            or self.my_fc2_alpha is None
            or self.my_fc1_norm_const is None
            or self.output_activation is None
        ):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        # SF leg caller contract guard: the host-supplied
        # ``activation_sf`` row stride must already be padded to a
        # multiple of 4 fp8 SFs so dispatch_pull's LDG.32 byte stride
        # (= ``sf_uint32_per_token * 4 bytes/token``) matches.  Catch
        # any future caller drift here rather than relying on a silent
        # cross-token-row misalignment manifesting as a numeric mismatch.
        if self.my_activation_sf.shape[-1] % 4 != 0:
            raise ValueError(
                f"activation_sf.shape[-1] ({self.my_activation_sf.shape[-1]}) "
                f"must be a multiple of 4 (4 FP8 SFs per uint32 dispatch "
                f"LDG.32 wire format).  Caller must pad K_sf along the "
                f"hidden axis with zero bytes before staging onto the "
                f"symmetric heap."
            )

        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from common.megamoe_constants import SfPaddingBlock
        from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel
        from src.sym_buffer import SymBufferHost

        # -- 1. Kernel instance --
        # MegaMoE currently requires ``static_expert_shape != None`` (see
        # the subclass docstring): the dispatch SMEM sizing + the pool
        # capacity formulas need ``num_experts_per_rank`` /
        # ``intermediate_gateup`` / ``hidden`` as codegen-time constants.
        static_expert_shape = (
            self.problem.num_experts_per_rank,
            self.problem.intermediate,
            self.problem.hidden,
        )

        cluster_size = self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(cluster_size)
        group_hint = self.impl.group_hint
        if group_hint is None:
            group_hint = max_active_clusters

        self._kernel = Sm100MegaMoEKernel(
            mma_tiler_mnk=self.impl.mma_tiler_mnk,
            cluster_shape_mnk=self.impl.cluster_shape_mnk,
            use_2cta_instrs=self.impl.use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=_SwapABEpilogueTokenTile,
            sf_padding_block=SfPaddingBlock,
            load_balance_mode=self.impl.load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=self.impl.force_static_sched,
            clc_bundle_size=self.impl.clc_bundle_size,
            num_sched_stages=self.impl.num_sched_stages,
            world_size=self.world_size,
            num_topk=self.problem.num_topk,
            max_tokens_per_rank=self.problem.num_tokens_per_rank,
            hidden=self.problem.hidden,
            fc2_output_dtype=cutlass.BFloat16,
            combine_format=self.problem.combine_format,
            non_ubulk_fc2_store=self.impl.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=self.impl.in_kernel_fc2_reduce,
            token_back_mode=self.impl.token_back_mode,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            gate_up_clamp=self.problem.gate_up_clamp,
            flag_batch=self.impl.flag_batch,
            epi_flag_batch=self.impl.epi_flag_batch,
        )

        # -- 2. Workspaces (local cuda + sym-heap) --
        self.allocate_workspaces()

        # -- 3. Torch -> cute --
        # Same align defaults as runner_fc12: 16 B for general tensors,
        # 4 B for the i32 / fp32 counter / sized buffers.
        def _to_cute(
            tensor: torch.Tensor, assumed_align: int = 16, force_static_layout=False
        ):
            cute_tensor = cutlass_torch.from_dlpack(
                tensor,
                assumed_align=assumed_align,
            )
            if force_static_layout:
                return cute_tensor
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        activation_cute = _to_cute(self.my_activation)
        activation_sf_cute = _to_cute(self.my_activation_sf)
        topk_idx_cute = _to_cute(self.my_topk_idx)
        topk_weights_cute = _to_cute(self.my_topk_weights)
        fc1_weight_cute = _to_cute(self.my_fc1_weight)
        fc1_weight_sf_cute = _to_cute(self.my_fc1_weight_sf)
        fc2_weight_cute = _to_cute(self.my_fc2_weight)
        fc2_weight_sf_cute = _to_cute(self.my_fc2_weight_sf)
        fc1_alpha_cute = _to_cute(self.my_fc1_alpha, assumed_align=4)
        fc2_alpha_cute = _to_cute(self.my_fc2_alpha, assumed_align=4)
        fc1_norm_const_cute = _to_cute(self.my_fc1_norm_const, assumed_align=4)
        output_activation_cute = _to_cute(self.output_activation)

        # Opaque byte workspaces -> raw uint8 gmem base pointers (no tensor). The
        # kernel addresses them by base + Int64 byte offset; a tensor's shape would
        # be ignored AND overflow cute's 32-bit memref shape field once the
        # internalized combine staging pushes shared_workspace past 2 GiB.
        from cutlass.cute.typing import AddressSpace as _AddressSpace

        def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
            return cute.runtime.make_ptr(
                cutlass.Uint8,
                tensor.data_ptr(),
                _AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        local_workspace_cute = _to_cute_ptr(self.local_workspace)
        shared_workspace_cute = _to_cute_ptr(self.shared_workspace)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- 4. cute.compile --
        # Runtime payload for the peer pointer mapper.  The generated host
        # wrapper packs these scalar args into ``SymBuffer{world_size}``
        # before launching the device kernel.
        peer_rank_ptr_mapper_host = SymBufferHost(
            base_addr=self.symmetric_base,
            offsets=tuple(self.peer_offsets_list),
            rank_idx=self.rank,
            num_max_ranks=self.world_size,
        )

        runtime_kwargs = dict(
            activation=activation_cute,
            activation_sf=activation_sf_cute,
            topk_idx=topk_idx_cute,
            topk_weights=topk_weights_cute,
            fc1_weight=fc1_weight_cute,
            fc1_weight_sf=fc1_weight_sf_cute,
            fc2_weight=fc2_weight_cute,
            fc2_weight_sf=fc2_weight_sf_cute,
            fc1_alpha=fc1_alpha_cute,
            fc2_alpha=fc2_alpha_cute,
            fc1_norm_const=fc1_norm_const_cute,
            output_activation=output_activation_cute,
            local_workspace=local_workspace_cute,
            shared_workspace=shared_workspace_cute,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket:
            compile_kwargs["options"] = "iket"

        self._compiled_kernel = cute.compile(self._kernel, **compile_kwargs)

        # The separate-kernel-reduce tail is now launched inside the kernel's
        # ``__call__`` (same stream, right after the mega kernel), so the runner
        # no longer compiles or chains a standalone reduce.

        # -- 5. Launch --
        #
        # ``profile_friendly`` mode wraps the launch in 4 markers:
        #
        #   (a) ``torch.cuda.synchronize()`` BEFORE the launch -- drains
        #       every setup-side GPU op off the stream so the capture
        #       launch sits alone in the post-sync timeline.
        #
        #   (b) Cross-rank ``dist.barrier()`` BEFORE cudaProfilerStart --
        #       aligns every rank to the same wall-clock moment before
        #       opening the capture window.  Without this, a rank that
        #       finishes setup early opens cudaProfilerStart, launches
        #       its mega kernel, and then spin-waits inside the kernel's
        #       NVSHMEM cross-rank ops for slower ranks to reach the
        #       same launch point; that spin time gets baked into the
        #       captured kernel duration and pollutes per-rank timing.
        #       The NCCL barrier kernel itself runs OUTSIDE the
        #       cudaProfilerStart/Stop window, so even though nsys
        #       observes it, the ``--capture-range=cudaProfilerApi`` flag
        #       keeps it out of the .nsys-rep timeline.
        #
        #   (c) ``cudaProfilerStart`` / ``cudaProfilerStop`` around the
        #       launch -- driver-level markers honoured by
        #       ``nsys --capture-range=cudaProfilerApi --capture-range-end=stop``
        #       and ``ncu --capture-range=cudaProfilerApi``.
        #
        #   (d) Cross-rank ``dist.barrier()`` AFTER cudaProfilerStop --
        #       ensures every rank has closed its capture window before
        #       advancing into the post-launch path (``free_tensor`` /
        #       ``nvshmem.finalize`` / etc.).  Without this, a fast rank
        #       can start freeing its sym tensors while a slower rank's
        #       mega kernel is still touching the sym heap, which races
        #       the symmetric-memory teardown and is the leading suspect
        #       behind multi-rank ``nsys`` runs hanging post-capture.
        #
        # ``_NO_DIST=1`` / uninitialised ``torch.distributed`` paths skip
        # both barriers silently so single-rank smoke runs and the
        # ``MEGA_NO_DIST=1`` debugger path are unaffected.
        #
        # Outside ``profile_friendly`` mode the launch path is unchanged.
        if self.misc.profile_friendly:
            import nvtx

            torch.cuda.synchronize()
            _dist_active = (
                torch.distributed.is_available() and torch.distributed.is_initialized()
            )
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
            with nvtx.annotate("cute_dsl_prof"):
                self._launch_target_kernels_with_optional_torch_profiler(
                    runtime_kwargs,
                )
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
        else:
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
            )

    # ------------------------------------------------------------------
    # Step 5: validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Compare the rank's final ``(token, hidden)`` combine output
        against the reference.

        The final downstream consumer of MegaMoE is a per-token, all-topk-
        aggregated activation tensor of shape ``(num_tokens_per_rank,
        hidden)`` -- i.e. ``sum_k combine_output[t, k, :]``.  The runner
        validates against that final form regardless of which side did
        the topk reduce:

          * non-reduce modes (form A, default): the
            kernel writes one fc2 cell per ``(src_token, src_topk)``
            into a ``(T, K, H)`` combine_output; a standalone CuTeDSL
            kernel applies topk weights, accumulates K in FP32, and
            stores BF16 ``(T, H)``.  The host K-axis sum is only used to
            build the validation reference, with the same final BF16 cast.

          * reduce modes (form B): the kernel emits
            a ``(T, 1, H)`` already-K-reduced output via cross-rank
            ``red.global.add.v2.bf16x2`` atomic adds.  The runner squeezes
            the singleton K axis and reduces the reference's K axis on
            host to match.

        bf16 tolerance: ``5e-2`` atol / rtol, identical to the runner's
        existing acceptance band; the K-axis sum adds at most ~K worth
        of bf16 round-off, which is well below this band for the v1
        configurations.  Form B (device-side reduce) is slightly noisier
        in principle because atomic-add ordering is non-deterministic,
        but the same atol covers it -- bf16 quantisation dwarfs the
        ordering-driven rounding drift.
        """
        if self.misc.skip_ref_check:
            return
        if self.output_activation is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")
        if self.combine_reduced_output_ref is None:
            raise RuntimeError("validate requires reduced reference output.")

        atol = 5e-2
        rtol = 5e-2

        if self.misc.ref_compute_graph == "transformers" and self.impl.fc2_reduces_topk:
            raise NotImplementedError(
                "transformers ref graph requires topk weighting during the "
                "host/combine reduce; in-kernel fc2 reduce has already "
                "collapsed the topk axis."
            )

        # Both modes now expose the same 2D (T, hidden) ``output_activation``.
        actual_reduced = self.output_activation.to(torch.float32)
        if self.impl.fc2_reduces_topk:
            # in_kernel_reduce: K already collapsed on device; the reference is
            # the host bf16 sequential K-sum of the per-topk reference cells.
            ref_reduced = _bf16_seq_sum_k(self.combine_output_ref).to(torch.float32)
        else:
            ref_reduced = self.combine_reduced_output_ref.to(torch.float32)

        diff = (actual_reduced - ref_reduced).abs()

        if self.rank == 0 and os.environ.get("MEGA_PEEK_WORKSPACE", "0") == "1":
            self._peek_workspace(actual_reduced, ref_reduced, diff)

        # Form B branching:
        #   K <= _FORM_B_PERM_K_LIMIT: exhaustive K! ordering bitwise check
        #     -- every K concurrent atomic adds can be observed in any of K!
        #     serial orders; if every cell hits SOME order's bf16-seq-sum,
        #     the redg path is algorithmically correct (= not a bug).
        #   K > _FORM_B_PERM_K_LIMIT: K! is intractable; switch to a
        #     percentile observatory on abs/rel diff and accept under a
        #     conservative threshold (real algorithmic bugs typically
        #     manifest as O(1) rel diff -- way above the threshold).
        if self.impl.fc2_reduces_topk:
            K = self.combine_output_ref.shape[1]
            if K <= _FORM_B_PERM_K_LIMIT:
                match_mask, n_perms = _form_b_bitwise_match(
                    actual_reduced,
                    self.combine_output_ref,
                )
                n_matched = int(match_mask.sum().item())
                n_total = match_mask.numel()
                pct = 100.0 * n_matched / max(n_total, 1)
                if self.rank == 0:
                    print(
                        f"---- form-B bitwise-ordering check "
                        f"(K={K}, perms={n_perms}): "
                        f"{n_matched}/{n_total} cells matched "
                        f"({pct:.2f}%) ----"
                    )
                # Cross-rank agreement MUST be a collective every rank
                # reaches: a passing rank returning early while a failing rank
                # falls through to a later all_gather desyncs NCCL and hangs.
                # Gather the per-rank pass flag here so all ranks branch
                # identically (all return, or all raise).
                local_bitwise_pass = n_matched == n_total
                rank_bitwise_passes = [local_bitwise_pass]
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    pass_tensor = torch.tensor(
                        [int(local_bitwise_pass)],
                        device=actual_reduced.device,
                        dtype=torch.int32,
                    )
                    gathered_passes = [
                        torch.empty_like(pass_tensor) for _ in range(self.world_size)
                    ]
                    torch.distributed.all_gather(gathered_passes, pass_tensor)
                    rank_bitwise_passes = [bool(t.item()) for t in gathered_passes]
                if all(rank_bitwise_passes):
                    if self.rank == 0:
                        print(
                            f"Validation PASSED on all ranks "
                            f"(form B bitwise: 100% cells match some "
                            f"K!={n_perms} ordering)"
                        )
                    return
                if not local_bitwise_pass:
                    bad = (~match_mask).nonzero(as_tuple=False)
                    n_local_bad = int(bad.shape[0])
                    sample = bad[: min(8, n_local_bad)].cpu().tolist()
                    print(
                        f"[rank {self.rank}] form-B bitwise miss: "
                        f"{n_local_bad} cell(s); sample "
                        f"(token, hidden)={sample}"
                    )
                    sys.stdout.flush()
                failed_ranks = [r for r, ok in enumerate(rank_bitwise_passes) if not ok]
                n_bad = int((diff > atol).sum().item())
                raise AssertionError(
                    f"[rank {self.rank}] form-B bitwise check failed on ranks "
                    f"{failed_ranks}; this rank matched {n_matched}/{n_total} "
                    f"cells to some K!={n_perms} ordering "
                    f"(max_diff={diff.max().item():.4g} "
                    f"mean_diff={diff.mean().item():.4g} n_bad={n_bad})."
                )
            else:
                # K > _FORM_B_PERM_K_LIMIT: K! enumeration intractable.
                # Two-tier validation:
                #   Tier 1 -- bulk noise floor sanity: rel_diff p90 must be
                #     under P90_REL_THRESHOLD (a small generous bound that
                #     real algo bugs would blow through; cancellation +
                #     bf16 ULP noise of mainstream cells stays well below).
                #   Tier 2 -- per-cell Wilkinson bf16 K-axis-accum bound on
                #     the top-N rel_diff outliers: every K! ordering's
                #     bf16-seq-sum result is provably within
                #     ``SAFETY * K * sum(|ref_K_terms|) / 256`` of the
                #     fp32 exact sum (Wilkinson 1963 round-off analysis).
                #     ``abs_diff <= bound`` <=> ``actual`` may come from
                #     SOME ordering -> no algo bug.
                P90_REL_THRESHOLD = 0.10
                N_VERIFY = 500
                # Wilkinson 1963 tight upper bound on bf16 K-axis
                # sequential round-off; no slack.  Cell PASS <=>
                # ``actual`` is mathematically reachable by SOME K!
                # ordering's bf16-seq-sum of ``ref_K_terms``.  Any
                # excess <=> bug.
                SAFETY = 1.0

                abs_diff = (actual_reduced - ref_reduced).abs()
                rel_diff = abs_diff / ref_reduced.abs().clamp(min=1.0)
                qs = [0.5, 0.9, 0.99, 0.999, 1.0]
                abs_q = _quantile_compat(abs_diff.flatten(), qs)
                rel_q = _quantile_compat(rel_diff.flatten(), qs)
                if self.rank == 0:
                    print(
                        f"---- form-B percentile observatory "
                        f"(K={K} > _FORM_B_PERM_K_LIMIT="
                        f"{_FORM_B_PERM_K_LIMIT}; bitwise enumeration "
                        f"skipped) ----"
                    )
                    print(
                        f"  abs diff: p50={abs_q[0]:.4g} p90={abs_q[1]:.4g} "
                        f"p99={abs_q[2]:.4g} p99.9={abs_q[3]:.4g} "
                        f"max={abs_q[4]:.4g}"
                    )
                    print(
                        f"  rel diff: p50={rel_q[0]:.4g} p90={rel_q[1]:.4g} "
                        f"p99={rel_q[2]:.4g} p99.9={rel_q[3]:.4g} "
                        f"max={rel_q[4]:.4g}"
                    )
                # Tier 1: bulk p90 rel sanity (catches systematic biases
                # that affect the majority of cells -- a real algo bug
                # would push the median up, not just the tail).
                if rel_q[1] > P90_REL_THRESHOLD:
                    raise AssertionError(
                        f"[rank {self.rank}] form-B bulk noise floor "
                        f"violated: p90_rel={rel_q[1]:.4g} > "
                        f"{P90_REL_THRESHOLD}.  The majority of cells "
                        f"have unexpectedly large drift; this is "
                        f"systematic, not tail outliers."
                    )
                # Tier 2: per-cell Wilkinson bound on top-N rel_diff cells.
                # rel_diff is the right "suspicion" metric for outliers
                # because small-ref + bf16-noise blows up there; abs_diff
                # alone misses them (they have small abs but huge rel).
                n_cells_total = rel_diff.numel()
                n_verify = min(N_VERIFY, n_cells_total)
                flat_rel = rel_diff.flatten()
                top_indices = torch.topk(flat_rel, n_verify).indices
                H = ref_reduced.shape[1]
                suspect_t = (top_indices // H).cpu()
                suspect_h = (top_indices % H).cpu()
                # Gather per-cell K terms (n_verify, K) then sum of |x|.
                # combine_output_ref is (T, K, H) bf16; advanced index
                # combines suspect_t and suspect_h elementwise.
                cells_ref_K = self.combine_output_ref[suspect_t, :, suspect_h].to(
                    torch.float32
                )
                cells_abs_sum = cells_ref_K.abs().sum(dim=1)
                bound = SAFETY * K * cells_abs_sum / 256.0
                suspect_abs_diff = abs_diff.flatten()[top_indices].cpu()
                bound_cpu = bound.cpu()
                excess = (suspect_abs_diff - bound_cpu).clamp(min=0)
                n_violators = int((excess > 0).sum().item())
                if n_violators > 0:
                    worst_idx = int(excess.argmax().item())
                    worst_t = int(suspect_t[worst_idx].item())
                    worst_h = int(suspect_h[worst_idx].item())
                    raise AssertionError(
                        f"[rank {self.rank}] form-B Wilkinson bf16-accum "
                        f"bound violated on {n_violators}/{n_verify} "
                        f"top-rel-diff suspect cells (SAFETY={SAFETY}x).  "
                        f"Worst: token={worst_t}, hidden={worst_h}, "
                        f"abs_diff="
                        f"{suspect_abs_diff[worst_idx].item():.4g}, "
                        f"bound={bound_cpu[worst_idx].item():.4g}, "
                        f"excess={excess[worst_idx].item():.4g}, "
                        f"ref="
                        f"{ref_reduced[worst_t, worst_h].item():.4g}, "
                        f"actual="
                        f"{actual_reduced[worst_t, worst_h].item():.4g}, "
                        f"sum_|ref_K_terms|="
                        f"{cells_abs_sum[worst_idx].item():.4g}.  "
                        f"See percentiles above."
                    )
                if self.rank == 0:
                    print(
                        f"Validation PASSED on all ranks (form B: "
                        f"p90_rel={rel_q[1]:.3g} <= {P90_REL_THRESHOLD}; "
                        f"top-{n_verify} rel-diff cells all within "
                        f"{SAFETY}x Wilkinson bf16-K-accum bound)"
                    )
                return

        bitwise_mismatch_count = None
        if self.impl.fc2_reduces_topk:
            local_pass = torch.allclose(
                actual_reduced,
                ref_reduced,
                atol=atol,
                rtol=rtol,
            )
        else:
            bitwise_mismatch_count = int((diff != 0).sum().item())
            local_pass = bitwise_mismatch_count == 0
        local_max_diff = diff.max().detach().reshape(1)
        rank_max_diffs = [float(local_max_diff.item())]
        rank_passes = [local_pass]
        _dist_active = (
            torch.distributed.is_available() and torch.distributed.is_initialized()
        )
        if _dist_active:
            gathered_diffs = [
                torch.empty_like(local_max_diff) for _ in range(self.world_size)
            ]
            pass_tensor = torch.tensor(
                [int(local_pass)], device=diff.device, dtype=torch.int32
            )
            gathered_passes = [
                torch.empty_like(pass_tensor) for _ in range(self.world_size)
            ]
            torch.distributed.all_gather(gathered_diffs, local_max_diff)
            torch.distributed.all_gather(gathered_passes, pass_tensor)
            rank_max_diffs = [float(rank_diff.item()) for rank_diff in gathered_diffs]
            rank_passes = [bool(rank_pass.item()) for rank_pass in gathered_passes]

        if not local_pass:
            print(self.output_activation)
            print(self.combine_output_ref)
            num_bad = (
                bitwise_mismatch_count
                if bitwise_mismatch_count is not None
                else int((diff > atol).sum().item())
            )
            reduce_label = (
                "form-B in-kernel fc2 reduce"
                if self.impl.fc2_reduces_topk
                else "CuTeDSL form-A topk reduce"
            )
            raise AssertionError(
                f"[rank {self.rank}] combine_output mismatch after "
                f"{reduce_label}: "
                f"max_diff={diff.max().item():.4g} "
                f"mean_diff={diff.mean().item():.4g} "
                f"n_bad={num_bad}/{actual_reduced.numel()} "
                f"({100 * num_bad / max(actual_reduced.numel(), 1):.2f}%)"
            )

        if self.rank == 0 and all(rank_passes):
            diff_summary = ", ".join(
                f"rank {rank}: max_diff={max_diff:.4g}"
                for rank, max_diff in enumerate(rank_max_diffs)
            )
            print(f"Validation PASSED on all ranks ({diff_summary})")

    # ------------------------------------------------------------------
    # Workspace peek (MEGA_PEEK_WORKSPACE=1)
    # ------------------------------------------------------------------

    def _peek_workspace(
        self,
        actual_reduced: torch.Tensor,
        ref_reduced: torch.Tensor,
        diff: torch.Tensor,
    ) -> None:
        """Dump intermediate workspace regions on a failing rank-0 run.

        Read-only peek into the local + shared workspace partitioned by
        the kernel, mostly meant to localise a combine mismatch to one
        of: (a) dispatch (routing / metadata), (b) fc1 (acc / SwiGLU
        cast), or (c) fc2 epi (STG indexing / peer_rank_ptr_mapper.map peer
        redirect).

        Gated by ``MEGA_PEEK_WORKSPACE=1`` so it never fires on perf or
        clean validate runs.  All slicing assumes ``_kernel`` has cached
        the region specs (it does, after ``run_kernel``).
        """
        from moe_nvfp4_swapab.megamoe_kernel import _layout_regions

        local_specs = self._kernel._local_region_specs
        shared_specs = self._kernel._shared_region_specs
        local_offsets, _ = _layout_regions(local_specs)
        shared_offsets, _ = _layout_regions(shared_specs)
        local_by_name = self._kernel._local_region_by_name
        shared_by_name = self._kernel._shared_region_by_name

        def _region(ws, by_name, offsets, name, torch_dtype):
            spec = by_name[name]
            off = offsets[name]
            return ws[off : off + spec.nbytes].view(torch_dtype)

        # ---- 1. expert routing counters (low32=tokens, high32=publishers*ranks).
        ercs = _region(
            self.shared_workspace,
            shared_by_name,
            shared_offsets,
            "expert_recv_count_sum",
            torch.int64,
        )
        print("---- expert_recv_count_sum ----")
        for i in range(ercs.shape[0]):
            v = int(ercs[i].item()) & 0xFFFFFFFFFFFFFFFF
            print(
                f"  expert[{i}]: tokens={v & 0xFFFFFFFF} "
                f"publishers*ranks={(v >> 32) & 0xFFFFFFFF}"
            )

        # ---- 2. token_src_metadata: (pool_slot) -> (src_rank, src_token, src_topk).
        mbuf = _region(
            self.local_workspace,
            local_by_name,
            local_offsets,
            "token_src_metadata",
            torch.int32,
        )
        md_u32 = mbuf.reshape(-1, 3)
        n_slots = md_u32.shape[0]
        sample_rows = sorted(
            set(
                list(range(min(8, n_slots)))
                + [n_slots // 2 - 1, n_slots // 2, n_slots // 2 + 1]
                + list(range(max(0, n_slots - 4), n_slots))
            )
        )
        print(
            f"---- token_src_metadata (pool_slot -> (src_rank, src_token, src_topk)) "
            f"n_slots={n_slots} ----"
        )
        for s in sample_rows:
            r = md_u32[s].cpu().tolist()
            print(f"  slot[{s}]: src_rank={r[0]} src_token={r[1]} src_topk={r[2]}")

        # ---- 2a. duplicate (src_rank, src_token, src_topk) detection.
        #
        # In a well-formed dispatch every valid pool slot maps to a unique
        # ``(src_rank, src_token, src_topk)`` triple (= one cell of
        # ``combine_output``).  If two pool slots share a triple, both will
        # fire their fc2 STG onto the same ``combine_output[src_token,
        # src_topk, :]`` row -- exactly the race-condition shape we saw
        # (non-determinism + all bad cells concentrated on one src_topk
        # slot).  This pass enumerates the offending triples and the pool
        # slots that map to each, so the bug can be localised to a
        # specific expert / advertise-card slot range.
        #
        # Padding slots (where dispatch never wrote metadata) sit at the
        # default ``(0, 0, 0)`` triple; they show up as one big
        # multi-collision cluster.  We separate them out and report the
        # padding-slot count plus the count of OTHER collision triples.
        md_cpu = md_u32.cpu()
        # Build a single int64 key per slot so we can rely on torch.unique.
        # Key bit layout: bits 0..15 = src_topk, bits 16..47 = src_token,
        # bits 48..63 = src_rank.  src_token can in principle exceed 16
        # bits (e.g. 1M-token bench config), so we use a 32-bit field for
        # safety -- the 64-bit key still fits.
        key = (
            md_cpu[:, 0].to(torch.int64) << 48
            | (md_cpu[:, 1].to(torch.int64) & 0xFFFFFFFF) << 16
            | (md_cpu[:, 2].to(torch.int64) & 0xFFFF)
        )
        unique_keys, inverse, counts = torch.unique(
            key, return_inverse=True, return_counts=True
        )
        coll_mask = counts > 1
        n_collisions = int(coll_mask.sum().item())
        print(
            f"---- duplicate metadata triples (n_collision_triples={n_collisions}) ----"
        )
        if n_collisions == 0:
            print("  (no duplicates -- metadata is 1:1 with combine_output cells)")
        else:
            coll_keys = unique_keys[coll_mask]
            coll_counts = counts[coll_mask]
            # Rank collision triples by replication count (worst first).
            sort_idx = torch.argsort(coll_counts, descending=True)
            shown = 0
            shown_limit = 8
            for i in sort_idx.tolist():
                k = int(coll_keys[i].item())
                c = int(coll_counts[i].item())
                src_rank = (k >> 48) & 0xFFFF
                src_token = (k >> 16) & 0xFFFFFFFF
                src_topk = k & 0xFFFF
                # Find which pool_slots map here.
                slot_ids = (key == k).nonzero(as_tuple=False).squeeze(-1)
                slot_sample = slot_ids[: min(8, slot_ids.numel())].tolist()
                tag = ""
                if src_rank == 0 and src_token == 0 and src_topk == 0:
                    tag = "  [LIKELY PADDING / UNWRITTEN]"
                print(
                    f"  (src_rank={src_rank}, src_token={src_token}, "
                    f"src_topk={src_topk}) x {c}{tag}"
                )
                print(f"    first {len(slot_sample)} slot ids: {slot_sample}")
                shown += 1
                if shown >= shown_limit:
                    remaining = n_collisions - shown
                    if remaining > 0:
                        print(f"  ... ({remaining} more collision triples elided)")
                    break

        # ---- 2b. expected-vs-actual (src_token, src_topk) reverse check.
        #
        # Single-rank only: every ``(src_token, src_topk)`` in
        # ``[0, T) x [0, K)`` routes to this rank and should appear in
        # ``token_src_metadata`` exactly once.  Multi-rank would need
        # filtering by ``my_topk_idx`` (rank ownership of each (t, k));
        # left as future work.
        #
        # ``missing`` = pairs that should have been stamped but weren't
        # (= dispatch dropped the metadata STG, leaving the slot at the
        # default ``(0, 0, 0)``).  ``extras`` = pairs stamped that
        # shouldn't exist in routing (= dispatch wrote garbage).  Both
        # are direct evidence of a dispatch bug.
        #
        # Histogram by ``src_topk`` is the high-signal bit: if missing
        # concentrates on a single src_topk, it pairs 1:1 with the
        # worst-diff topk slot downstream.
        #
        # Valid-slot identification: use ``fc1_output`` non-zero row mask
        # rather than ``(md != (0,0,0))``, because the legitimate routing
        # pair ``(src_token=0, src_topk=0)`` would otherwise collide with
        # the padding default and be mis-flagged as missing.  fc1_output
        # is written iff the slot carries a valid token instance, so
        # ``fc1_output non-zero`` is the correct "this slot is in use"
        # discriminator.  Recomputed inline rather than threaded from the
        # later fc1_output block to keep the two passes independent.
        fc1_spec_rc = local_by_name["fc1_output"]
        fc1_off_rc = local_offsets["fc1_output"]
        fc1_raw_rc = self.local_workspace[
            fc1_off_rc : fc1_off_rc + fc1_spec_rc.nbytes
        ].view(torch.uint8)
        row_bytes_rc = fc1_spec_rc.shape[1] // 2 if len(fc1_spec_rc.shape) >= 2 else 0
        rc_ready = row_bytes_rc > 0 and fc1_raw_rc.numel() % row_bytes_rc == 0
        if rc_ready:
            fc1_2d_rc = fc1_raw_rc.view(-1, row_bytes_rc).cpu()
            valid_slot_mask = (fc1_2d_rc != 0).any(dim=1)
            T = self.problem.num_tokens_per_rank
            K = self.problem.num_topk
            # Multi-rank: also need src_rank to disambiguate (same
            # (t, k) on different source ranks lives in different
            # pool slots on the receiver).  Single-rank degenerates to
            # the previous behaviour (src_rank == 0 everywhere).
            seen_triples = set(map(tuple, md_cpu[valid_slot_mask, :].tolist()))
            # Expected universe: every (src_rank, src_token, src_topk)
            # whose routed expert lands on THIS rank.  Per-rank
            # routing is built by replaying ``_generate_topk_idx_*``;
            # to avoid recomputing it the harness keeps the global
            # tensor in ``self._global_topk_idx`` (shape
            # ``(num_ranks, T, K)``).  Filter by
            # ``expert // num_experts_per_rank == self.rank``.
            num_experts_per_rank = self.problem.num_total_experts // self.world_size
            expected_triples = set()
            if hasattr(self, "_global_topk_idx") and self._global_topk_idx is not None:
                gti = self._global_topk_idx.cpu()  # (R, T, K)
                for sr in range(self.world_size):
                    for t in range(T):
                        for k in range(K):
                            e = int(gti[sr, t, k].item())
                            if e // num_experts_per_rank == self.rank:
                                expected_triples.add((sr, t, k))
            expected_pairs = expected_triples  # alias kept for parity
            seen_pairs = seen_triples
            missing = expected_pairs - seen_pairs
            extras = seen_pairs - expected_pairs
            print(
                f"---- expected-vs-actual (src_rank, src_token, src_topk) "
                f"reverse check (world={self.world_size}, T={T}, K={K}, "
                f"expected={len(expected_triples)}, observed_unique="
                f"{len(seen_pairs)}) ----"
            )
            print(f"  missing (expected but never stamped): {len(missing)}")
            print(f"  extras  (stamped but not in routing): {len(extras)}")
            if missing:
                topk_hist = [0] * K
                rank_hist = [0] * self.world_size
                for sr, _, k in missing:
                    if 0 <= k < K:
                        topk_hist[k] += 1
                    if 0 <= sr < self.world_size:
                        rank_hist[sr] += 1
                print(
                    f"  missing by src_topk: "
                    f"{ {k: c for k, c in enumerate(topk_hist) if c} }"
                )
                print(
                    f"  missing by src_rank: "
                    f"{ {r: c for r, c in enumerate(rank_hist) if c} }"
                )
                sample = sorted(missing)[:16]
                print(f"  first 16 missing (src_rank, src_token, src_topk): {sample}")
            if extras:
                topk_hist = [0] * K
                rank_hist = [0] * self.world_size
                for sr, _, k in extras:
                    if 0 <= k < K:
                        topk_hist[k] += 1
                    if 0 <= sr < self.world_size:
                        rank_hist[sr] += 1
                print(
                    f"  extras by src_topk: "
                    f"{ {k: c for k, c in enumerate(topk_hist) if c} }"
                )
                print(
                    f"  extras by src_rank: "
                    f"{ {r: c for r, c in enumerate(rank_hist) if c} }"
                )
                sample = sorted(extras)[:16]
                print(f"  first 16 extras (src_rank, src_token, src_topk): {sample}")
        else:
            print(
                "---- expected-vs-actual reverse check skipped "
                "(fc1_output peek unavailable) ----"
            )

        # ---- 3. fc1_output non-zero row mask (per pool_slot) + hash.
        #
        # The hash (sum modulo a large prime over the raw bytes) is a
        # quick fingerprint to compare ``fc1_output`` across two runs
        # of the same cmd: if the hash differs the race is on the
        # write side of fc1_output (dispatch pull -> fc1 GEMM / fc1
        # epi); if the hash matches but ``combine_output`` still
        # differs the race is on the read / fc2 phase side.
        fc1_spec = local_by_name["fc1_output"]
        fc1_off = local_offsets["fc1_output"]
        fc1_raw = self.local_workspace[fc1_off : fc1_off + fc1_spec.nbytes].view(
            torch.uint8
        )
        row_bytes = fc1_spec.shape[1] // 2 if len(fc1_spec.shape) >= 2 else 0
        if row_bytes > 0 and fc1_raw.numel() % row_bytes == 0:
            fc1_2d = fc1_raw.view(-1, row_bytes).cpu()
            nz = (fc1_2d != 0).any(dim=1)
            zero_rows = (~nz).nonzero(as_tuple=False).squeeze(-1).tolist()
            # Cheap byte fingerprint: sum over uint32 view modulo a
            # 31-bit prime.  Stable across runs iff the underlying
            # bytes are identical -- collisions are vanishingly rare
            # for the workspaces we look at.
            fc1_u32 = fc1_raw.view(torch.int32).cpu().to(torch.int64)
            fc1_hash = int((fc1_u32 & 0xFFFFFFFF).sum().item()) % 0x7FFFFFFF
            print(
                f"---- fc1_output zero-row mask "
                f"shape={tuple(fc1_2d.shape)} all_zero_rows={len(zero_rows)}/{fc1_2d.shape[0]} "
                f"hash=0x{fc1_hash:08x} ----"
            )
            if zero_rows:
                print(f"  first 16 zero rows: {zero_rows[:16]}")
                print(f"  last 16 zero rows : {zero_rows[-16:]}")

            # ---- 3b. fc1_output * SF dequant sample (head/tail rows).
            #
            # cute SF layout (``tile_atom_to_shape_SF((M, K, L=1), vec)``):
            #   shape  ((32, 4, M_rest), (vec, 4, K_rest))
            #   stride ((16, 4, 512*K_rest), (0, 1, 512))
            # Per-byte offset for logical (m, k_element):
            #   m_inner=(m%32)*16 + ((m//32)%4)*4 + (m//128)*512*K_atoms
            #   k     =(k%vec=collapsed) + ((k//vec)%4)*1 + (k//(vec*4))*512
            sf_vec_size = Nvfp4BlockSize
            intermediate_downproj_int = fc1_spec.shape[1]
            K_sf_blocks_total = intermediate_downproj_int // sf_vec_size
            K_atoms = (K_sf_blocks_total + 3) // 4
            sf_spec = local_by_name["fc1_output_sf"]
            sf_off = local_offsets["fc1_output_sf"]
            sf_raw = (
                self.local_workspace[sf_off : sf_off + sf_spec.nbytes]
                .view(torch.uint8)
                .cpu()
            )

            def _sf_byte_offset(m: int, sf_block: int) -> int:
                return (
                    (m % 32) * 16
                    + ((m // 32) % 4) * 4
                    + (m // 128) * (512 * K_atoms)
                    + (sf_block % 4)
                    + (sf_block // 4) * 512
                )

            n_show_rows = 4
            n_show_cells = 16
            n_rows_total = fc1_2d.shape[0]
            sample_row_ids = list(range(min(n_show_rows, n_rows_total))) + [
                r
                for r in range(
                    max(0, n_rows_total - n_show_rows),
                    n_rows_total,
                )
                if r >= n_show_rows  # avoid duplicates when rows < 2*n_show
            ]
            print(
                f"---- fc1_output * SF dequant sample "
                f"(rows {sample_row_ids}; SF per {sf_vec_size}-element block) ----"
            )
            for r in sample_row_ids:
                # fp4 values for this row.
                row_bytes_t = fc1_2d[r]  # uint8, len=row_bytes
                lo = (row_bytes_t & 0x0F).to(torch.int64)
                hi = ((row_bytes_t >> 4) & 0x0F).to(torch.int64)
                fp4_idx = torch.stack([lo, hi], dim=-1).reshape(-1)
                fp4_vals = _Fp4DecodeTable[fp4_idx]
                # SF bytes for this row (one byte per sf_block).
                sf_byte_idxs = torch.tensor(
                    [_sf_byte_offset(r, sb) for sb in range(K_sf_blocks_total)],
                    dtype=torch.int64,
                )
                sf_bytes = sf_raw[sf_byte_idxs]
                sf_fp32 = sf_bytes.view(torch.float8_e4m3fn).to(torch.float32)
                # Apply per-block SF.
                dequant = (
                    fp4_vals.reshape(K_sf_blocks_total, sf_vec_size)
                    * sf_fp32.unsqueeze(-1)
                ).reshape(-1)

                n_cells_actual = min(n_show_cells, dequant.numel())
                head_cells = [f"{v:+.2f}" for v in dequant[:n_cells_actual].tolist()]
                sf_show = [f"{s:+.3f}" for s in sf_fp32.tolist()]
                print(
                    f"  row[{r}]: dequant min={dequant.min().item():+.3f} "
                    f"max={dequant.max().item():+.3f} "
                    f"mean={dequant.mean().item():+.3f} "
                    f"nnz={int((fp4_vals != 0).sum().item())}"
                    f"/{fp4_vals.numel()} | "
                    f"first {n_cells_actual}: {head_cells} | "
                    f"SF: {sf_show}"
                )

            # ---- 3c. fc1_output * SF percentiles over valid rows only.
            # ``nz`` already marks valid pool rows (= any non-zero byte
            # in the row); padding rows are TMA-OOB-fill-0 so they are
            # zero and dominate the global distribution if not filtered.
            valid_row_ids = nz.nonzero(as_tuple=False).squeeze(-1).tolist()
            n_valid = len(valid_row_ids)
            if n_valid > 0:
                # Bulk dequant whole pool's fp4 first, then index valid
                # rows + SF gather.  fc1_2d is (n_rows, row_bytes) uint8
                # already on CPU.
                full_lo = (fc1_2d & 0x0F).to(torch.int64)
                full_hi = ((fc1_2d >> 4) & 0x0F).to(torch.int64)
                full_fp4_idx = torch.stack(
                    [full_lo, full_hi],
                    dim=-1,
                ).reshape(fc1_2d.shape[0], -1)
                full_fp4_vals = _Fp4DecodeTable[full_fp4_idx]
                # Per-(valid_row, sf_block) SF byte offsets, gathered
                # into one tensor then sliced back.
                sf_offsets_all = torch.tensor(
                    [
                        _sf_byte_offset(r, sb)
                        for r in valid_row_ids
                        for sb in range(K_sf_blocks_total)
                    ],
                    dtype=torch.int64,
                )
                sf_bytes_all = sf_raw[sf_offsets_all].view(
                    n_valid,
                    K_sf_blocks_total,
                )
                sf_fp32_all = sf_bytes_all.view(torch.float8_e4m3fn).to(torch.float32)
                valid_fp4 = full_fp4_vals[valid_row_ids]
                dequant_all = (
                    valid_fp4.reshape(
                        n_valid,
                        K_sf_blocks_total,
                        sf_vec_size,
                    )
                    * sf_fp32_all.unsqueeze(-1)
                ).reshape(n_valid, -1)
                all_vals = dequant_all.flatten()
                abs_vals = all_vals.abs()
                qs = [0.5, 0.9, 0.99, 0.999, 1.0]
                abs_q = _quantile_compat(abs_vals, qs)
                sf_q = _quantile_compat(sf_fp32_all.flatten(), qs)
                print(
                    f"---- fc1_output * SF percentiles "
                    f"(valid_rows={n_valid}, cells={all_vals.numel()}) ----"
                )
                print(
                    f"  |val|:   p50={abs_q[0]:.4g} p90={abs_q[1]:.4g} "
                    f"p99={abs_q[2]:.4g} p99.9={abs_q[3]:.4g} "
                    f"max={abs_q[4]:.4g}"
                )
                print(
                    f"  SF:      p50={sf_q[0]:.4g} p90={sf_q[1]:.4g} "
                    f"p99={sf_q[2]:.4g} p99.9={sf_q[3]:.4g} "
                    f"max={sf_q[4]:.4g}"
                )
                print(
                    f"  signed:  mean={all_vals.mean().item():+.4g} "
                    f"std={all_vals.std().item():.4g} "
                    f"nnz%={100 * (all_vals != 0).float().mean().item():.2f}"
                )
        else:
            print(
                f"---- fc1_output peek skipped (spec.shape={fc1_spec.shape}, "
                f"raw nbytes={fc1_raw.numel()}) ----"
            )

        # ---- 3a. fc1_output_sf and l1_sf_buffer hash for cross-run diffing.
        #
        # Same purpose as the fc1_output hash above: stable byte hash
        # across runs => buffer content is deterministic; varying hash
        # => race on the write side of that buffer.  l1_sf_buffer is
        # the dispatch-pull-written input SF for the fc1 phase;
        # fc1_output_sf is fc1 epi's per-thread STG output SF for the
        # fc2 phase.  Pairing the three hashes (fc1_output above +
        # these two) lets us localise a race to one of the three
        # write stages.
        for buf_name in ("fc1_output_sf", "l1_sf_buffer"):
            spec = local_by_name[buf_name]
            off = local_offsets[buf_name]
            raw = self.local_workspace[off : off + spec.nbytes].view(torch.uint8)
            u32 = raw.view(torch.int32).cpu().to(torch.int64)
            buf_hash = int((u32 & 0xFFFFFFFF).sum().item()) % 0x7FFFFFFF
            print(
                f"---- {buf_name} hash=0x{buf_hash:08x} "
                f"(spec.shape={spec.shape}, nbytes={raw.numel()}) ----"
            )

        # ---- 3a2. routing dump for the tokens that show a non-zero
        # ``diff_sum`` (= K-reduced abs diff on the host-side reduce
        # path).  In power_law a subset of tokens is suspected to be
        # deterministic-error tokens (token 0 + a handful of others);
        # dumping each such token's 11 routing experts lets us check
        # whether they share a structural property (e.g. all hit one
        # specific heavy-tail expert, or all hit a particular task-tile
        # boundary).
        if hasattr(self, "_global_topk_idx") and self._global_topk_idx is not None:
            host_diff_per_token = diff.sum(dim=-1).cpu()  # (T,)
            bad_token_ids = (
                (host_diff_per_token > 0).nonzero(as_tuple=False).squeeze(-1).tolist()
            )
            n_bad_tokens = len(bad_token_ids)
            topk_idx_cpu = self._global_topk_idx[self.rank].cpu()
            topk_w_cpu = (
                self._global_topk_weights[self.rank].cpu()
                if hasattr(self, "_global_topk_weights")
                and self._global_topk_weights is not None
                else None
            )
            # Per-LOCAL-expert recv count (``ercs`` only covers this
            # rank's ``num_experts_per_rank`` experts; global expert
            # ids that route to OTHER ranks have no entry here).
            n_local_e = ercs.shape[0]
            num_experts_per_rank = self.problem.num_total_experts // self.world_size
            expert_recv_local = [
                int(ercs[i].item()) & 0xFFFFFFFF for i in range(n_local_e)
            ]

            def _recv_for(e_global):
                owner = e_global // num_experts_per_rank
                if owner == self.rank:
                    local = e_global - owner * num_experts_per_rank
                    if 0 <= local < n_local_e:
                        return str(expert_recv_local[local])
                return f"@r{owner}"  # remote-rank expert: count not visible here

            dump_limit = 8
            print(
                f"---- routing for tokens with diff_sum > 0 "
                f"(count={n_bad_tokens}, showing first {min(dump_limit, n_bad_tokens)}) ----"
            )
            for t in bad_token_ids[:dump_limit]:
                t_routes = topk_idx_cpu[t].tolist()
                ds = float(host_diff_per_token[t].item())
                pieces = []
                for k_idx, e in enumerate(t_routes):
                    e_int = int(e)
                    wt = (
                        f"w={topk_w_cpu[t, k_idx].item():.3g}"
                        if topk_w_cpu is not None
                        else ""
                    )
                    pieces.append(f"k{k_idx}=e{e_int}(recv={_recv_for(e_int)},{wt})")
                print(f"  token[{t}] diff_sum={ds:.4g}  " + " ".join(pieces))

        # ---- 4. worst-diff tokens (on UNREDUCED (T, K, H) abs diff).
        #
        # ``diff`` is ``|actual_reduced - ref_reduced|`` over the
        # K-axis-reduced view -- topk reduction can cancel
        # opposite-sign errors and undercount real per-cell errors.
        # The peek instead works on the raw ``|actual - ref|`` cell
        # tensor of shape ``(T, K, H)`` so every cell-level error is
        # visible.
        # separate_kernel_reduce: the per-topk (T, K, H) staging is internal to
        #   the kernel, so read it back from the shared-workspace
        #   ``combine_quant`` region for a raw per-cell diff (bf16 combine only).
        # in_kernel_reduce: device output is already (T, H) (K reduced on
        #   device); the per-(K, H) cell diff is meaningless, so reduce ref's K
        #   axis on host and diff in (T, H).
        if self.impl.fc2_reduces_topk:
            actual_full = self.output_activation.to(torch.float32)
            ref_full = self.combine_output_ref.to(torch.float32).sum(dim=1)
            unreduced_diff = (actual_full - ref_full).abs()  # (T, H)
            per_token_diff_sum = unreduced_diff.sum(dim=-1)  # (T,)
        else:
            combine_quant_raw = _region(
                self.shared_workspace,
                shared_by_name,
                shared_offsets,
                "combine_quant",
                self.problem.fc2_output_dtype,
            )
            actual_full = combine_quant_raw.reshape(self.combine_output_ref.shape).to(
                torch.float32
            )
            ref_full = self.combine_output_ref.to(torch.float32)
            unreduced_diff = (actual_full - ref_full).abs()  # (T, K, H)
            per_token_diff_sum = unreduced_diff.sum(dim=(1, 2))  # (T,)
        n_worst_tokens = min(8, per_token_diff_sum.numel())
        worst_tokens = torch.topk(per_token_diff_sum, k=n_worst_tokens)
        bad_tokens = worst_tokens.indices.cpu().tolist()
        bad_diff_sums = worst_tokens.values.cpu().tolist()

        n_worst_cells_per_token = 6
        H = unreduced_diff.shape[-1]
        if self.impl.fc2_reduces_topk:
            print(
                "---- worst-diff tokens (top "
                f"{n_worst_tokens} by sum over hidden; per token: "
                f"top {n_worst_cells_per_token} hidden cells; K already "
                "reduced on device) ----"
            )
            for t, ds in zip(bad_tokens, bad_diff_sums, strict=False):
                token_diff = unreduced_diff[t]  # (H,)
                n_cells = min(n_worst_cells_per_token, token_diff.numel())
                top_cells = torch.topk(token_diff, k=n_cells)
                cell_idxs = top_cells.indices.cpu().tolist()
                print(
                    f"  token[{t}]: diff_sum={ds:.4g}  "
                    f"max_cell_diff={token_diff.max().item():.4g}"
                )
                for h in cell_idxs:
                    a = float(actual_full[t, h].item())
                    r = float(ref_full[t, h].item())
                    d = float(token_diff[h].item())
                    print(
                        f"    (hidden={h}): actual={a:.4g}  ref={r:.4g}  diff={d:.4g}"
                    )
        else:
            print(
                "---- worst-diff tokens (top "
                f"{n_worst_tokens} by sum over (topk, hidden); per token: "
                f"top {n_worst_cells_per_token} (topk, hidden) cells) ----"
            )
            for t, ds in zip(bad_tokens, bad_diff_sums, strict=False):
                token_diff = unreduced_diff[t]  # (K, H)
                flat = token_diff.flatten()
                n_cells = min(n_worst_cells_per_token, flat.numel())
                top_cells = torch.topk(flat, k=n_cells)
                cell_idxs = top_cells.indices.cpu().tolist()
                print(
                    f"  token[{t}]: diff_sum={ds:.4g}  "
                    f"max_cell_diff={token_diff.max().item():.4g}"
                )
                for ci in cell_idxs:
                    k = ci // H
                    h = ci % H
                    a = float(actual_full[t, k, h].item())
                    r = float(ref_full[t, k, h].item())
                    d = float(token_diff[k, h].item())
                    print(
                        f"    (topk={k}, hidden={h}): "
                        f"actual={a:.4g}  ref={r:.4g}  diff={d:.4g}"
                    )

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self.rank == 0:
            print(self.problem)
            print(self.impl)
            print(self.misc)
            print(
                f"MegaMoETester: torch_profiler="
                f"{'on' if self._use_torch_profiler else 'off'}"
            )

        self.generate_inputs()

        if self.rank == 0:
            for name, tensor in (
                ("my_activation", self.my_activation),
                ("my_activation_sf", self.my_activation_sf),
                ("my_topk_idx", self.my_topk_idx),
                ("my_fc1_weight", self.my_fc1_weight),
                ("output_activation", self.output_activation),
            ):
                if tensor is None:
                    print(f"[rank {self.rank}] {name:18s} not allocated")
                    continue
                print(
                    f"[rank {self.rank}] {name:18s} "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )

        skip_ref = self.misc.skip_ref_check or self.misc.profile_friendly
        if not skip_ref:
            self.compute_reference()

        # Once the kernel is wired, this is also the place ``allocate_workspaces``
        # gets called (depends on self._kernel being instantiated inside run_kernel).
        self.run_kernel()

        if not skip_ref:
            self.validate()

        if self.rank == 0:
            print("DONE")


# =============================================================================
# CLI
# =============================================================================


def _parse_tuple(argument: str) -> Tuple[int, ...]:
    return tuple(int(value) for value in argument.split(","))


def _parse_output_dtype(argument: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    if argument not in mapping:
        raise argparse.ArgumentTypeError(
            f"fc2_output_dtype must be one of {sorted(mapping.keys())}, got {argument!r}"
        )
    return mapping[argument]


def _parse_combine_format(argument: str) -> CombineFormat:
    try:
        return CombineFormat.parse(argument)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaMoE NVFP4 multi-rank fused dispatch+fc12+combine runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num_tokens_per_rank", type=int, default=128)
    parser.add_argument("--num_topk", type=int, default=4)
    parser.add_argument("--num_total_experts", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--fc2_output_dtype",
        type=_parse_output_dtype,
        default=torch.bfloat16,
    )
    parser.add_argument(
        "--combine_format",
        type=_parse_combine_format,
        default=CombineFormat.parse("bf16"),
        help="Cross-rank combine wire format: 'bf16' (no-quant baseline), "
        "'16e2m1xbf16' (fp4 + per-16 bf16 amax), or '32e4m3xe8m0' (MXFP8).",
    )
    parser.add_argument(
        "--route_distribution",
        type=str,
        default="balanced",
        choices=["balanced", "power_law"],
    )
    parser.add_argument(
        "--power_law_exponent",
        type=float,
        default=1.0,
        help="Zipf exponent for --route_distribution power_law; 1.0 = classic "
        "Zipf, larger = more skewed, 0 = uniform.",
    )
    parser.add_argument(
        "--gate_up_clamp",
        type=float,
        default=None,
        help="DeepSeek-V4 swiglu_limit: asymmetric clamp on the real gate/up "
        "pre-activations (gate<=limit, |up|<=limit) before SiLU. "
        "Omitted/None disables the clamp; must be non-negative.",
    )

    parser.add_argument("--mma_tiler_mnk", type=str, default="128,128,256")
    parser.add_argument("--cluster_shape_mnk", type=str, default="1,1,1")
    parser.add_argument("--use_2cta_instrs", action="store_true", default=False)
    parser.add_argument(
        "--enable_static_expert_shape", action="store_true", default=False
    )
    parser.add_argument("--dynamic_sched", action="store_true", default=False)
    parser.add_argument("--clc_bundle_size", type=int, default=None)
    parser.add_argument("--num_sched_stages", type=int, default=None)
    parser.add_argument(
        "--load_balance_mode",
        type=str,
        default="static",
        choices=["static", "atomic_counter"],
    )
    parser.add_argument("--group_hint", type=int, default=None)
    parser.add_argument("--flag_batch", type=int, default=4)
    parser.add_argument(
        "--epi_flag_batch",
        type=str,
        default="1,1",
        help="(fc1,fc2) done-counter publish batch in comma form",
    )
    parser.add_argument(
        "--use_bulk_fc2_store",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--in_kernel_fc2_reduce",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--token_back_mode",
        type=str,
        default="epi_warps",
        choices=["epi_warps", "standalone_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 push-back runs: epi_warps (epilogue "
        "STG redirect, default), standalone_warps (dedicated warps "
        "12-15), or reuse_dispatch_warps (dispatch warps 8-11).",
    )
    parser.add_argument("--perf_run", action="store_true", default=False)
    parser.add_argument("--skip_ref_check", action="store_true", default=False)
    parser.add_argument("--profile_friendly", action="store_true", default=False)
    parser.add_argument("--use_torch_profiler", action="store_true", default=False)
    # Default to 0,1 to avoid testing time.
    parser.add_argument("--perf_warmup", type=int, default=0)
    parser.add_argument("--perf_iters", type=int, default=1)
    parser.add_argument("--enable_debug_checks", action="store_true", default=False)
    parser.add_argument(
        "--ref_compute_graph",
        type=str,
        default="deepgemm",
        choices=["transformers", "deepgemm"],
    )
    parser.add_argument("--enable_iket", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Bootstrap NCCL + NVSHMEM, build the tester, run one pass.

    Launcher::

        torchrun --nproc_per_node=4 -m moe_nvfp4_swapab.mega_runner \\
            --num_total_experts 32 --route_distribution balanced
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if _NO_DIST:
        # MEGA_NO_DIST=1: bypass torch.distributed + NVSHMEM init so the
        # script can be launched as plain ``python mega_runner.py`` under
        # compute-sanitizer / debugger.  Single-rank only.
        torch.cuda.set_device(0)
        rank = 0
        world_size = 1
    else:
        from src.bootstrap import init_dist_and_nvshmem

        _local_rank, rank, world_size, _ = init_dist_and_nvshmem()

    problem = TokenCommProblemDesc(
        world_size=world_size,
        num_tokens_per_rank=args.num_tokens_per_rank,
        num_topk=args.num_topk,
        num_total_experts=args.num_total_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        fc2_output_dtype=args.fc2_output_dtype,
        combine_format=args.combine_format,
        route_distribution=args.route_distribution,
        power_law_exponent=args.power_law_exponent,
        gate_up_clamp=args.gate_up_clamp,
    )

    if problem.num_topk > 32:
        raise ValueError(
            f"num_topk ({problem.num_topk}) > 32 is unsupported by the current. Shit inherited from DeepGEMM."
        )

    impl = ImplDesc(
        mma_tiler_mnk=_parse_tuple(args.mma_tiler_mnk),
        cluster_shape_mnk=_parse_tuple(args.cluster_shape_mnk),
        use_2cta_instrs=args.use_2cta_instrs,
        enable_static_expert_shape=args.enable_static_expert_shape,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        non_ubulk_fc2_store=not args.use_bulk_fc2_store,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        token_back_mode=args.token_back_mode,
        flag_batch=args.flag_batch,
        epi_flag_batch=_parse_tuple(args.epi_flag_batch),
    )

    misc = MiscDesc(
        perf_run=args.perf_run,
        skip_ref_check=args.skip_ref_check,
        # MiscDesc field kept as ``run_target_kernel_only`` to keep
        # runner_fc12.py's reads working unmodified.  ``profile_friendly``
        # is the @property alias defined alongside the field; both spellings
        # read the same boolean.
        run_target_kernel_only=args.profile_friendly,
        enable_debug_checks=args.enable_debug_checks,
        ref_compute_graph=args.ref_compute_graph,
        enable_iket=args.enable_iket,
        seed=args.seed,
    )

    tester = MegaMoETester(problem, impl, misc, rank=rank)
    tester.set_torch_profiler_enabled(args.use_torch_profiler)
    tester.set_perf_iters(args.perf_warmup, args.perf_iters)

    return_code = 0
    try:
        tester.run()
    except NotImplementedError as exc:
        # Expected until the MegaMoE kernel side is wired; the host
        # orchestration above is the part being smoke-tested for now.
        if rank == 0:
            print(f"[mega_runner] kernel launch skipped: {exc}")

    if not _NO_DIST:
        # nvshmem_free/finalize are collective barriers; an unsynchronized or
        # GC-driven teardown deadlocks once per-rank free order diverges.  So
        # just barrier-align, then os._exit and let the driver reclaim the heap
        # on exit.  os._exit skips finalizers/GC, hence the manual flush.
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
