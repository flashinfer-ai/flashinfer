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
import gc
from contextlib import nullcontext
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch


_FORM_B_PERM_K_LIMIT = 4   # K! perms: K<=4 -> <=24, manageable


def _form_b_enumerate_perms(ref_K_terms: torch.Tensor) -> torch.Tensor:
    """Return ``(T, H, K!)`` bf16 tensor of every K-axis permutation's
    bf16 sequential-add candidate result.  Hard-rejects ``K > 4`` (K!
    grows factorially; K=4 -> 24 perms is the practical ceiling for
    full ordering enumeration).
    """
    assert ref_K_terms.dtype == torch.bfloat16 and ref_K_terms.is_cuda
    _, K, _ = ref_K_terms.shape
    if K > _FORM_B_PERM_K_LIMIT:
        raise ValueError(
            f"_form_b_enumerate_perms: K={K} > {_FORM_B_PERM_K_LIMIT}; "
            f"K! enumeration is intractable beyond K=4."
        )
    from .cute_ref_ops import bf16_all_perm_sums
    return bf16_all_perm_sums(ref_K_terms)


def _quantile_compat(x: torch.Tensor, qs: List[float]) -> List[float]:
    """torch.quantile-equivalent that handles >16M tensors via numpy.

    ``torch.quantile`` errors out with "input tensor is too large" past
    ~2^24 elements (CPU sort backed); ``np.quantile`` is partition-based
    and has no such cap.  Both paths return the same fp64 quantiles.
    """
    if x.numel() <= 16_777_216:
        return torch.quantile(
            x, torch.tensor(qs, dtype=x.dtype, device=x.device),
        ).cpu().tolist()
    return np.quantile(x.cpu().numpy(), np.asarray(qs)).tolist()


def _form_b_bitwise_match(
    actual_reduced: torch.Tensor,  # (T, H) bf16 or fp32 (bit cast via .to(bf16))
    ref_K_terms: torch.Tensor,     # (T, K, H) bf16
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


def _bf16_seq_sum_k(src: torch.Tensor) -> torch.Tensor:
    """cuTeDSL ``sum(dim=1)`` over (T, K, H) bf16, accumulating in bf16
    to match device-side atomic-add precision.
    """
    assert src.dtype == torch.bfloat16 and src.is_cuda
    from .cute_ref_ops import bf16_seq_sum_k
    return bf16_seq_sum_k(src)

# Ensure absolute package imports work when run as a script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(os.path.dirname(_PKG_DIR))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import cutlass

from .epilogue_refactor import SwapABSwigluFp4Epilogue as _SwapABEpilogue
from .mega_reference import compute_megamoe_reference
from ..comm.megamoe_config import MegaMoEConfig, MegaMoEDiagnostics, MegaMoEExperimental
from ..comm.nccl_gin_layout import NcclGinWindowLayout, build_megamoe_gin_window_layout
from ..comm.nccl_gin_runtime import (
    NcclGinContext,
    NcclGinUnavailableError,
    NcclLsaContext,
    create_nccl_gin_context,
    create_nccl_lsa_context,
    register_nccl_gin_window,
    register_nccl_lsa_window,
    validate_nccl_gin_rail_topology,
)
from ..comm.reference import internode_dispatch_prep_oracle, internode_stage1_layout_oracle
from ..comm.relaunch_protocol import (
    build_expert_pool_layout,
    pool_row_slots,
    expected_nvlink_state,
    nvlink_calls_per_launch,
    packed_recv_sum,
)
from ..comm.config import sf_atom_int32_offset_numpy
from ..comm.token_comm import CombineFormat
from ..common.megamoe_constants import Nvfp4BlockSize
from .runner_common import (
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
from .runner_fc12_common import resolve_group_hint
from .runner_fc12 import ImplDesc, MiscDesc
from .simulate_fc1_fc2_sched import FusedFc12Simulator

_SwapABEpilogueTokenTile = _SwapABEpilogue._EpilogueTokenTileSize
_SwapABEpilogueIntermediateAlignment = (
    2 * _SwapABEpilogue._EpilogueFc1GateUpInterleave
)


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
                f"gate_up_clamp must be None or non-negative, got "
                f"{self.gate_up_clamp}."
            )

    def __str__(self) -> str:
        dtype_name = lambda t: str(t).split(".")[-1]
        route_str = self.route_distribution
        if self.route_distribution == "power_law":
            route_str = f"power_law(exponent={self.power_law_exponent:.3g})"
        clamp_str = (
            "off" if self.gate_up_clamp is None
            else f"{self.gate_up_clamp:.4g}"
        )
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
    topk_offsets = rng.random(
        (num_ranks, num_blocks, num_total_experts)
    ).argsort(axis=-1)[..., :num_topk]

    token_offsets = np.arange(num_total_experts)[None, None, :, None]
    expert_indices = (
        token_offsets + topk_offsets[:, :, None, :]
    ) % num_total_experts
    topk_blocks = np.take_along_axis(
        expert_permutations[..., None], expert_indices, axis=2,
    )
    return topk_blocks.reshape(
        num_ranks, num_padded_tokens_per_rank, num_topk,
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
        topk_idx, world_size, num_total_experts,
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
    return torch.rand(
        (num_ranks, num_tokens_per_rank, num_topk),
        dtype=torch.float32, device="cuda", generator=rng,
    ) + 0.5


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
#
# Initialised from the environment at import time: the other legacy runners
# (moe_bf16_glu, moe_hopper_*, moe_mxfp8_*) import ``_NO_DIST`` from this module
# and never call ``_set_no_dist``, so a plain ``False`` here would send their
# ``MEGA_NO_DIST=1`` single-process runs into ``init_dist_and_nvshmem``.
# ``main()`` re-applies the parsed ``MegaMoEDiagnostics.no_dist`` (same source).
_NO_DIST: bool = bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def _set_no_dist(enabled: bool) -> None:
    """Record ``MegaMoEDiagnostics.no_dist`` for the module-level symmetric-heap
    helpers (``_sym_zeros``, ``_compute_peer_offsets``); ``main()`` calls this
    once right after parsing the diagnostics."""
    global _NO_DIST
    _NO_DIST = bool(enabled)


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
    logical_shape: Tuple[int, ...], target_dtype: torch.dtype,
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
    return _sym_zeros((total_bytes,), torch.uint8).view(target_dtype).reshape(storage_shape)


def _multicast_base(sym_tensor: torch.Tensor) -> int:
    """POC (multimem barrier): absolute multicast (NVLS) VA base of a symmetric
    tensor, or 0 when MEGA_MULTIMEM_BARRIER is off / NVLS unavailable. Only
    created on demand (an NVLS multicast costs an NVSHMEM team)."""
    if _NO_DIST or os.environ.get("MEGA_MULTIMEM_BARRIER", "0") != "1":
        return 0
    try:
        import nvshmem.core

        mc = nvshmem.core.get_multicast_tensor(
            nvshmem.core.Teams.TEAM_WORLD, sym_tensor
        )
        return int(mc.data_ptr())
    except Exception as exc:  # noqa: BLE001
        print(f"[mega_runner] multicast mapping failed ({exc}); barrier stays baseline.")
        return 0


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
        int(nvshmem.core.get_peer_tensor(sym_tensor, peer).data_ptr())
        - local_base
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
         on EVERY rank from ``misc.seed``; stage own-rank peer inputs in
         NVSHMEM (legacy) or the NCCL LSA arena (relay). Weights stay local.
      2. ``compute_reference``: run ``compute_megamoe_reference`` on the
         global view; slice own ``(T, K, hidden)`` into ``combine_output_ref``.
      3. ``allocate_workspaces``: ``kernel.get_workspace_sizes()`` ->
         allocate local CUDA memory plus either legacy NVSHMEM shared memory
         or one registered LSA arena; derive the corresponding peer deltas.
      4. ``run_kernel``: compile and launch the fused MegaMoE kernel.
      5. ``validate``: bf16-grade compare own-rank ``combine_output``
         against ``combine_output_ref``.

    Relay workspace setup performs collective NCCL communicator/window
    registration; data movement still begins only with the kernel launch.
    """

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        nccl_gin_context: Optional[NcclGinContext] = None,
        nccl_lsa_context: Optional[NcclLsaContext] = None,
        config: Optional[MegaMoEConfig] = None,
    ) -> None:
        self.problem = problem
        self.impl = impl
        self.misc = misc
        # Codegen options and diagnostics of the kernel (pre_unification/comm/megamoe_config.py).
        self.config = config if config is not None else MegaMoEConfig()
        self.rank = rank
        if impl.fc2_reduces_topk and misc.ref_compute_graph != "deepgemm":
            raise ValueError(
                "in-kernel fc2 reduce requires ref_compute_graph='deepgemm'; "
                f"got {misc.ref_compute_graph!r}.  The REDG path can only "
                "atomic-add terms whose topk score was already absorbed before "
                "fc2."
            )
        # Single source of truth for world size; ``local_rank`` was only
        # needed for ``torch.cuda.set_device`` during runtime bootstrap;
        # the tester itself does not reference it.
        self.world_size = problem.world_size

        torch.manual_seed(misc.seed)
        np.random.seed(misc.seed)
        self._np_rng: np.random.Generator = np.random.default_rng(misc.seed)
        self._torch_cuda_rng = torch.Generator(device="cuda")
        self._torch_cuda_rng.manual_seed(misc.seed)

        # Own-rank peer inputs: NVSHMEM for legacy, NCCL LSA for relay.
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
        self._scheduled_routed_tokens_per_rank: Optional[Tuple[int, ...]] = None
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
        self.gin_workspace: Optional[torch.Tensor] = None
        self._lsa_arena: Optional[torch.Tensor] = None
        self._lsa_arena_regions = {}
        self._gin_window_layout: Optional[NcclGinWindowLayout] = None
        self._gin_internode_send_count: Optional[torch.Tensor] = None
        self._gin_internode_recv_splits: Optional[torch.Tensor] = None
        self._gin_relay_send_buffer: Optional[torch.Tensor] = None
        self._gin_relay_recv_buffer: Optional[torch.Tensor] = None
        self._gin_intra_node_reduce_buf: Optional[torch.Tensor] = None
        self._gin_internode_combine_buffer: Optional[torch.Tensor] = None
        self._gin_context = nccl_gin_context
        self._lsa_context = nccl_lsa_context
        self._gin_runtime = None
        self._lsa_runtime = None
        # Host-side peer mapper payload. Legacy intranode mode uses the
        # world-wide NVSHMEM mapper; relay mode stores only node-local NCCL LSA
        # offsets and rejects every cross-logical-node map.
        self.symmetric_base: Optional[int] = None
        self.peer_offsets_list: Optional[Tuple[int, ...]] = None

        self._kernel = None
        self._compiled_kernel = None
        self._max_active_clusters: Optional[int] = None
        self._use_torch_profiler = False
        self._perf_warmup = 1
        self._perf_iters = 1
        self._perf_sync_each_launch = False
        self._validate_each_launch = False
        self._validated_launch_count = 0
        self._reported_scheduler_group_layout = False

        # v1 pinned expert-wise epilogue args.
        self.my_fc1_alpha: Optional[torch.Tensor] = None
        self.my_fc2_alpha: Optional[torch.Tensor] = None
        self.my_fc1_norm_const: Optional[torch.Tensor] = None

    def set_torch_profiler_enabled(self, enabled: bool) -> None:
        """Enable optional torch profiler timing for the target kernel launch."""
        self._use_torch_profiler = enabled

    def set_perf_iters(self, warmup: int, iters: int) -> None:
        """Configure profiler timing launch counts."""
        self._perf_warmup = max(0, int(warmup))
        self._perf_iters = max(1, int(iters))

    def set_perf_sync_each_launch(self, enabled: bool) -> None:
        """Synchronize every repeated profiler launch when explicitly enabled.

        This diagnostic/workaround mode prevents any rank from enqueuing the
        next fused launch until all ranks complete the current launch. It
        deliberately changes timing
        semantics and must not be reported as back-to-back steady-state
        throughput.
        """
        self._perf_sync_each_launch = bool(enabled)

    def set_validate_each_launch(self, enabled: bool) -> None:
        """Validate fused-tail cleanup and phase state after every launch."""

        self._validate_each_launch = bool(enabled)

    def _workspace_region(self, *, shared: bool, name: str) -> torch.Tensor:
        """Return a typed torch view of one production workspace region."""

        if self._kernel is None:
            raise RuntimeError("workspace region lookup requires a built kernel")
        workspace = self.shared_workspace if shared else self.local_workspace
        if workspace is None:
            raise RuntimeError("workspace region lookup requires allocated storage")
        specs = (
            self._kernel._shared_region_by_name
            if shared else self._kernel._local_region_by_name
        )
        offsets = (
            self._kernel._shared_offsets
            if shared else self._kernel._local_offsets
        )
        spec = specs[name]
        if spec.cute_dtype is cutlass.Int32:
            torch_dtype = torch.int32
        elif spec.cute_dtype is cutlass.Int64:
            torch_dtype = torch.int64
        else:
            raise TypeError(
                f"diagnostic region {name} has unsupported dtype "
                f"{spec.cute_dtype}"
            )
        return workspace.narrow(
            0, offsets[name], spec.nbytes,
        ).view(torch_dtype).reshape(spec.shape)

    def _pipeline_timing_region(self) -> Optional[torch.Tensor]:
        if not self.config.diagnostics.pipeline_timing:
            return None
        return self._workspace_region(shared=False, name="pipeline_timing")

    def _combine_timing_landmarks_region(self) -> Optional[torch.Tensor]:
        if not self.config.diagnostics.pipeline_timing:
            return None
        return self._workspace_region(
            shared=False, name="combine_timing_landmarks"
        )

    def _fc_tile_timeline_region(self) -> Optional[torch.Tensor]:
        if not self.config.diagnostics.fc_tile_timeline:
            return None
        return self._workspace_region(shared=False, name="fc_tile_timeline")

    def _report_fc_tile_timeline(self) -> None:
        """Print the per-tile FC1/FC2 timeline of the last captured launch."""
        snapshot = getattr(self, "_fc_tile_timeline_snapshot", None)
        if snapshot is None:
            snapshot = self._fc_tile_timeline_region()
        if snapshot is None:
            return
        from ..common.fc_tile_timeline_host import report_fc_tile_timeline
        from ..common.pipeline_timing import PIPELINE_STAGE_DISPATCH

        dispatch_start_ns = None
        timing = getattr(self, "_fc_tile_pipeline_timing_snapshot", None)
        if timing is None:
            timing = self._pipeline_timing_region()
        if timing is not None:
            starts = timing[:, :, PIPELINE_STAGE_DISPATCH, 0]
            valid = starts[starts > 0]
            if valid.numel():
                dispatch_start_ns = int(valid.min().item())
        report_fc_tile_timeline(
            snapshot, rank=self.rank, dispatch_start_ns=dispatch_start_ns,
            dump=self.config.diagnostics.fc_tile_timeline_dump,
        )

    def _hang_dump(self, tag: str) -> None:
        """Print which pipeline stages have completed while a launch is stuck.

        Reads the pipeline-timing region over a side stream while the kernel
        still occupies the launch stream; each (CTA, warp) writes a stage's
        start/end pair when that stage function completes, so a stage with
        fewer ends than expected is where the hang sits. Diagnostic only
        (``MEGA_HANG_DUMP_SEC``), requires ``MEGA_PIPELINE_TIMING=1``.
        """
        try:
            timing = self._pipeline_timing_region()
            if timing is None:
                print(f"[hang-dump] rank={self.rank} {tag}: pipeline timing disabled", flush=True)
                return
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                host = timing.detach().to("cpu", non_blocking=True)
            side.synchronize()
            from ..common.pipeline_timing import PIPELINE_STAGE_NAMES
            ends = host[..., 1]
            starts = host[..., 0]
            parts = []
            for idx, name in enumerate(PIPELINE_STAGE_NAMES):
                if idx >= host.shape[2]:
                    break
                done = int((ends[:, :, idx] > 0).sum().item())
                started = int((starts[:, :, idx] > 0).sum().item())
                if done or started:
                    parts.append(f"{name}={done}/{started}")
            print(f"[hang-dump] rank={self.rank} {tag} stages(done/started per CTA-warp): " + " ".join(parts), flush=True)
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"[hang-dump] rank={self.rank} {tag} failed: {exc!r}", flush=True)

    def _start_hang_dump_watchdog(self, tag: str):
        """Arm ``MEGA_HANG_DUMP_SEC`` dumps at 1x and 2x the delay; returns timers."""
        import threading

        seconds = self.config.diagnostics.hang_dump_sec
        if seconds <= 0:
            return []
        timers = []
        for k in (1, 2):
            t = threading.Timer(seconds * k, self._hang_dump, args=(f"{tag}+{int(seconds * k)}s",))
            t.daemon = True
            t.start()
            timers.append(t)
        return timers

    def _reset_pipeline_timing(self) -> None:
        timing = self._pipeline_timing_region()
        combine_landmarks = self._combine_timing_landmarks_region()
        if timing is not None and combine_landmarks is not None:
            from ..common.pipeline_timing_host import reset_pipeline_timing

            reset_pipeline_timing(timing, combine_landmarks)
        fc_tiles = self._fc_tile_timeline_region()
        if fc_tiles is not None:
            fc_tiles.zero_()

    def _capture_pipeline_timing(
        self,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Queue same-stream device copies before the next timing reset."""
        timing = self._pipeline_timing_region()
        combine_landmarks = self._combine_timing_landmarks_region()
        if timing is None or combine_landmarks is None:
            return None
        fc_tiles = self._fc_tile_timeline_region()
        if fc_tiles is not None:
            # The last capture wins: the report covers the final timed launch.
            self._fc_tile_timeline_snapshot = fc_tiles.clone()
            self._fc_tile_pipeline_timing_snapshot = timing.clone()
        return timing.clone(), combine_landmarks.clone()

    def _report_pipeline_timing(
        self,
        snapshot: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if snapshot is None:
            timing = self._pipeline_timing_region()
            combine_landmarks = self._combine_timing_landmarks_region()
        else:
            timing, combine_landmarks = snapshot
        if timing is not None and combine_landmarks is not None:
            from ..common.pipeline_timing_host import report_pipeline_timing

            report_pipeline_timing(
                timing, combine_landmarks, rank=self.rank,
                theory_us=self.config.diagnostics.combine_theory_us,
            )

    def _report_pipeline_timing_samples(
        self,
        snapshots: List[Tuple[torch.Tensor, torch.Tensor]],
        full_call_critical_ranks: Optional[List[int]] = None,
    ) -> None:
        if not snapshots:
            return
        from ..common.pipeline_timing_host import report_pipeline_timing_samples

        report_pipeline_timing_samples(
            snapshots,
            rank=self.rank,
            full_call_critical_ranks=full_call_critical_ranks,
        )

    def _validate_relaunch_state_local(self) -> None:
        """Check zero prefixes and the exact persistent NVLink phase."""

        if self._kernel is None:
            raise RuntimeError("relaunch validation requires a built kernel")
        if self.local_workspace is None or self.shared_workspace is None:
            raise RuntimeError("relaunch validation requires workspaces")

        local_bytes, shared_bytes = (
            self._kernel.require_zero_workspace_leading_bytes
        )
        local_nonzero = int(torch.count_nonzero(
            self.local_workspace.narrow(0, 0, local_bytes)
        ).item())
        shared_nonzero = int(torch.count_nonzero(
            self.shared_workspace.narrow(0, 0, shared_bytes)
        ).item())
        if local_nonzero or shared_nonzero:
            raise AssertionError(
                "fused-tail zero-prefix residue: "
                f"local_nonzero_bytes={local_nonzero} "
                f"shared_nonzero_bytes={shared_nonzero}"
            )

        ncu_padding = self.config.diagnostics.use_ncu
        calls_per_launch = nvlink_calls_per_launch(
            enable_nccl_gin=self.impl.enable_nccl_gin,
            enable_lsa_relay=(
                self.impl.enable_internode_relay
                and self.world_size > self.impl.ranks_per_node
            ),
            ncu_padding=ncu_padding,
        )
        compiled_base_calls = int(
            self._kernel.token_comm.nvlink_barriers_per_launch
        )
        compiled_padding = (
            int(self._kernel.token_comm.nvlink_ncu_padding)
            if ncu_padding else 0
        )
        if calls_per_launch != compiled_base_calls + compiled_padding:
            raise AssertionError(
                "host/device NVLink call-count contract drift: "
                f"host={calls_per_launch} device="
                f"{compiled_base_calls + compiled_padding}"
            )

        expected = expected_nvlink_state(
            launches=self._validated_launch_count + 1,
            calls_per_launch=calls_per_launch,
            ranks_per_node=self.impl.ranks_per_node,
        )
        counter = int(self._workspace_region(
            shared=False, name="nvlink_barrier_counter",
        ).reshape(-1)[0].item())
        signals_tensor = self._workspace_region(
            shared=True, name="nvlink_barrier_signal",
        ).reshape(-1)
        signals = tuple(int(value) for value in signals_tensor.cpu().tolist())
        if counter != expected.counter or signals != expected.signals:
            raise AssertionError(
                "persistent NVLink phase mismatch: "
                f"launch={self._validated_launch_count + 1} "
                f"counter={counter}/{expected.counter} "
                f"signals={signals}/{expected.signals}"
            )

        if self.impl.enable_nccl_gin:
            self.validate_internode_stage1_metadata()

    def _validate_relaunch_state(self, phase: str, iteration: int) -> None:
        """Run local checks and make every rank report the same verdict."""

        error = ""
        try:
            self._validate_relaunch_state_local()
        except Exception as exc:  # noqa: BLE001 - aggregate exact rank failure
            error = f"rank={self.rank}: {type(exc).__name__}: {exc}"

        errors = [error]
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            errors = [""] * self.world_size
            torch.distributed.all_gather_object(errors, error)
        failures = [entry for entry in errors if entry]
        if failures:
            raise AssertionError(
                "per-launch relaunch validation failed: " + " | ".join(failures)
            )

        self._validated_launch_count += 1
        if self.rank == 0:
            print(
                f"[relaunch-validate] PASS launch="
                f"{self._validated_launch_count} phase={phase} "
                f"iteration={iteration}",
                flush=True,
            )

    def _check_cuda_rng_consistency(self) -> None:
        """Weakly verify CUDA RNG streams stayed aligned across ranks."""
        if not (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            return

        random_parts = torch.randint(
            0, 1 << 31, (2, 16),
            dtype=torch.int64, device="cuda", generator=self._torch_cuda_rng,
        )
        sentinel = (random_parts[0] << 31) ^ random_parts[1]
        gathered = [torch.empty_like(sentinel) for _ in range(self.world_size)]
        torch.distributed.all_gather(gathered, sentinel)

        reference = gathered[0]
        mismatched_ranks = [
            rank for rank, value in enumerate(gathered)
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
            self._torch_cuda_rng, (num_ranks, num_tokens_per_rank, hidden),
            packed_dim=-1, perf_run=self.misc.perf_run,
        )
        self._global_activation_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_tokens_per_rank, hidden, blocksize=scale_blocksize,
            strict=True
        ).reshape(num_ranks, num_tokens_per_rank, hidden_sf_cols)

        # ---- Routing table.
        if problem.route_distribution == "balanced":
            topk_idx_np = _generate_topk_idx_balanced(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, rng,
            )
        else:
            topk_idx_np = _generate_topk_idx_power_law(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, problem.power_law_exponent, rng,
            )
        topk_weights = _generate_topk_weights(
            num_ranks, num_tokens_per_rank, num_topk, self._torch_cuda_rng,
        )
        if self.rank == 0:
            _print_remote_rank_comm_matrices(
                topk_idx_np, num_ranks, problem.num_total_experts,
            )
        valid_expert_ids = topk_idx_np[topk_idx_np >= 0]
        expert_owner_ranks = valid_expert_ids // num_experts_per_rank
        self._scheduled_routed_tokens_per_rank = tuple(
            int(count)
            for count in np.bincount(
                expert_owner_ranks,
                minlength=num_ranks,
            ).tolist()
        )
        self._global_topk_idx = torch.from_numpy(topk_idx_np).cuda()
        self._global_topk_weights = topk_weights

        # ---- Weights.  fc1 packs along hidden (K of fc1); fc2 packs along
        # intermediate_downproj (K of fc2).
        self._global_fc1_weight = _make_nvfp4_tensor_from_torch_rng(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, hidden, intermediate),
            packed_dim=2, perf_run=self.misc.perf_run,
        )
        self._global_fc1_weight_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * intermediate,
            hidden, blocksize=scale_blocksize, strict=True
        ).reshape(num_ranks, num_experts_per_rank, intermediate, hidden_sf_cols)

        self._global_fc2_weight = _make_nvfp4_tensor_from_torch_rng(
            self._torch_cuda_rng,
            (num_ranks, num_experts_per_rank, intermediate_downproj, hidden),
            packed_dim=2, perf_run=self.misc.perf_run,
        )
        self._global_fc2_weight_sf = _make_raw_scale_tensor_from_torch_rng(
            self._torch_cuda_rng,
            num_ranks * num_experts_per_rank * hidden,
            intermediate_downproj, blocksize=scale_blocksize,
            strict=True
        ).reshape(
            num_ranks, num_experts_per_rank, hidden, intermediate_downproj_sf_cols,
        )
        epi_arg_shape = (num_ranks, num_experts_per_rank)
        self._global_fc1_alpha = torch.randint(
            1, 5, epi_arg_shape, generator=self._torch_cuda_rng, device="cuda",
        ).to(torch.float32) * 0.5
        self._global_fc2_alpha = torch.randint(
            1, 5, epi_arg_shape, generator=self._torch_cuda_rng, device="cuda",
        ).to(torch.float32) * 0.5
        self._global_fc1_norm_const = torch.randint(
            2, 5, epi_arg_shape, generator=self._torch_cuda_rng, device="cuda",
        ).to(torch.float32) * 0.5

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
        self._global_fc1_weight_sf_swizzled = (
            _stack_byte_reinterpretable_tensors(fc1_sf_swizzled, dim=0)
            .view(num_ranks, num_experts_per_rank, fc1_flat_sf_size)
        )

        fc2_sf_swizzled = [
            to_blocked(self._global_fc2_weight_sf[r, e])
            for r in range(num_ranks)
            for e in range(num_experts_per_rank)
        ]
        fc2_flat_sf_size = fc2_sf_swizzled[0].numel()
        self._global_fc2_weight_sf_swizzled = (
            _stack_byte_reinterpretable_tensors(fc2_sf_swizzled, dim=0)
            .view(num_ranks, num_experts_per_rank, fc2_flat_sf_size)
        )

        # ---- Stage own-rank inputs into peer-accessible storage.
        # Relay mode must size/register one LSA arena before carving inputs;
        # legacy intranode mode retains its individual NVSHMEM allocations.
        if self.impl.enable_internode_relay:
            self._ensure_kernel()
            self.allocate_workspaces()

        own_activation = self._global_activation[self.rank]
        own_activation_sf = self._global_activation_sf[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        # NVFP4 / fp8 go through uint8 byte-buf views (nvshmem4py doesn't
        # natively support these dtypes); copy via uint8 to dodge any
        # NVFP4-to-NVFP4 assignment quirks between allocators.
        if not self.impl.enable_internode_relay:
            self.my_activation = _sym_zeros_byte_view(
                (num_tokens_per_rank, hidden), _DataDtype,
            )
        if self.my_activation is None:
            raise RuntimeError("activation storage was not allocated")
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
        if not self.impl.enable_internode_relay:
            self.my_activation_sf = _sym_zeros_byte_view(
                (num_tokens_per_rank, hidden_sf_cols_padded), _ScaleDtype,
            )
        if self.my_activation_sf is None:
            raise RuntimeError("activation SF storage was not allocated")
        self.my_activation_sf.view(torch.uint8)[:, :hidden_sf_cols].copy_(
            own_activation_sf.view(torch.uint8)
        )

        if not self.impl.enable_internode_relay:
            self.my_topk_idx = _sym_zeros(
                tuple(own_topk_idx.shape), torch.int64,
            )
            self.my_topk_weights = _sym_zeros(
                tuple(own_topk_weights.shape), torch.float32,
            )
        if self.my_topk_idx is None or self.my_topk_weights is None:
            raise RuntimeError("routing storage was not allocated")
        self.my_topk_idx.copy_(own_topk_idx)
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

        if not self.impl.enable_internode_relay:
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
        self.combine_reduced_output_ref = (
            reference.combine_reduced_output[self.rank].contiguous()
        )

    def _ensure_kernel(self) -> None:
        """Instantiate the static kernel before sizing the relay LSA arena."""
        if self._kernel is not None:
            return

        import cutlass
        import cutlass.utils as utils

        from ..common.megamoe_constants import SfPaddingBlock
        from .megamoe_kernel import Sm100MegaMoEKernel

        static_expert_shape = (
            self.problem.num_experts_per_rank,
            self.problem.intermediate,
            self.problem.hidden,
        )
        cluster_size = (
            self.impl.cluster_shape_mnk[0] * self.impl.cluster_shape_mnk[1]
        )
        self._max_active_clusters = int(
            utils.HardwareInfo().get_max_active_clusters(cluster_size)
        )
        group_hint = resolve_group_hint(
            self.impl.group_hint, self._max_active_clusters
        )

        self._kernel = Sm100MegaMoEKernel(
            config=self.config,
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
            ranks_per_node=(
                self.impl.ranks_per_node
                if self.impl.ranks_per_node is not None
                else self.world_size
            ),
            combine_format=self.problem.combine_format,
            non_ubulk_fc2_store=self.impl.non_ubulk_fc2_store,
            in_kernel_fc2_reduce=self.impl.in_kernel_fc2_reduce,
            # Until the two-stage internode combine is wired, GIN relay
            # outputs must stay in the LSA FC2 workspace; reuse dispatch warps
            # for node-local token-back and leave cross-node terms staged.
            token_back_mode=(
                "reuse_dispatch_warps"
                if self.impl.enable_nccl_gin
                else self.impl.token_back_mode
            ),
            internode=self.impl.internode,
            enable_internode_relay=self.impl.enable_internode_relay,
            enable_nccl_gin=self.impl.enable_nccl_gin,
            # The GIN specialization has complete Stage 1/2/3 payload
            # publication; LSA-only relay scaffolding remains intra-only.
            publish_relay_counts_to_gemm=self.impl.enable_nccl_gin,
            apply_topk_in_fc1=self.misc.ref_compute_graph == "deepgemm",
            gate_up_clamp=self.problem.gate_up_clamp,
            flag_batch=self.impl.flag_batch,
            epi_flag_batch=self.impl.epi_flag_batch,
        )
        if self.rank == 0:
            # The compiled-kernel cache key; it records every codegen knob,
            # including the environment-driven ones (register split, SF TMEM
            # staging), so a run's kernel variant is verifiable from its log.
            print(f"[kernel] name={self._kernel.name()}", flush=True)

    # ------------------------------------------------------------------
    # Step 3: workspace allocation
    # ------------------------------------------------------------------

    @staticmethod
    def _arena_view(
        arena: torch.Tensor,
        byte_offset: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        numel = int(np.prod(shape, dtype=np.int64))
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        return arena.narrow(0, byte_offset, nbytes).view(dtype).reshape(shape)

    def allocate_workspaces(self) -> None:
        """Allocate local memory and the peer-accessible communication arena.

        Legacy intranode mode preserves the NVSHMEM allocations and its one
        world-wide offset table. Internode relay mode instead registers one NCCL
        LSA arena containing every object reached through the peer mapper:
        activation data/SFs, top-k weights, shared workspace (including FC2
        output staging/counters), and the optional in-kernel-reduce output.
        One registered root is required because NCCL peer deltas are window
        specific; using one root keeps the existing base-plus-delta mapper ABI.
        """
        if self.local_workspace is not None:
            return
        self.mc_base = 0
        if self._kernel is None:
            raise RuntimeError(
                "allocate_workspaces called before self._kernel was instantiated"
            )

        local_ws_bytes, shared_ws_bytes = self._kernel.get_workspace_sizes()
        self.local_workspace = torch.zeros(
            (local_ws_bytes,), dtype=torch.uint8, device="cuda",
        )

        if not self.impl.enable_internode_relay:
            self.shared_workspace = _sym_zeros(
                (shared_ws_bytes,), torch.uint8,
            )
            self.symmetric_base, self.peer_offsets_list = _compute_peer_offsets(
                self.shared_workspace, self.world_size,
            )
            self.mc_base = _multicast_base(self.shared_workspace)
            return

        tokens = self.problem.num_tokens_per_rank
        hidden = self.problem.hidden
        topk = self.problem.num_topk
        hidden_sf_cols = round_up(ceil_div(hidden, Nvfp4BlockSize), 4)

        regions = {}
        cursor = 0

        def reserve(name: str, nbytes: int, align: int = 128) -> int:
            nonlocal cursor
            cursor = round_up(cursor, align)
            regions[name] = (cursor, nbytes)
            cursor += nbytes
            return regions[name][0]

        activation_shape = (tokens, hidden // 2)
        activation_sf_shape = (tokens, hidden_sf_cols)
        topk_weights_shape = (tokens, topk)
        if self.impl.enable_nccl_gin:
            self._gin_window_layout = build_megamoe_gin_window_layout(
                world_size=self.world_size,
                ranks_per_node=int(
                    self.impl.ranks_per_node or self.world_size
                ),
                num_experts_per_rank=self.problem.num_experts_per_rank,
                max_relay_tokens_per_node=tokens,
                hidden_bytes=hidden // 2,
                sf_uint32_per_token=self._kernel.sf_uint32_per_token,
                num_topk=topk,
                combine_hidden=hidden,
            )
            gin_transport_bytes = max(
                self._gin_window_layout.total_bytes,
                round_up(self.impl.gin_window_bytes, 128),
            )
            gin_transport_offset = reserve(
                "gin_transport", gin_transport_bytes, 128,
            )
            if gin_transport_offset != 0:
                raise AssertionError(
                    "GIN transport layout must start at registered-window offset 0"
                )
        reserve("activation", int(np.prod(activation_shape)), 128)
        reserve("activation_sf", int(np.prod(activation_sf_shape)), 128)
        reserve("topk_weights", int(np.prod(topk_weights_shape)) * 4, 128)
        reserve("shared_workspace", shared_ws_bytes, 128)
        if self.impl.fc2_reduces_topk:
            output_bytes = tokens * hidden * torch.empty(
                (), dtype=self.problem.fc2_output_dtype,
            ).element_size()
            reserve("output_activation", output_bytes, 128)

        arena_bytes = round_up(cursor, 128)
        if self.impl.enable_nccl_gin:
            arena_bytes = max(arena_bytes, round_up(self.impl.gin_window_bytes, 128))

        if _NO_DIST:
            self._lsa_arena = torch.zeros(
                (arena_bytes,), dtype=torch.uint8, device="cuda",
            )
            window = None
        else:
            from nccl.core.interop.torch import empty as nccl_torch_empty

            self._lsa_arena = nccl_torch_empty(
                (arena_bytes,),
                dtype=torch.uint8,
                device=torch.cuda.current_device(),
            )
            self._lsa_arena.zero_()
            if self.impl.enable_nccl_gin:
                if self._gin_context is None:
                    raise RuntimeError(
                        "NCCL GIN context must exist before LSA arena registration"
                    )
                self._gin_runtime = register_nccl_gin_window(
                    self._gin_context, self._lsa_arena,
                )
                self._gin_context = None
                window = self._gin_runtime.window
                self.gin_workspace = self._lsa_arena
            else:
                if self._lsa_context is None:
                    raise RuntimeError(
                        "NCCL LSA context must exist before arena registration"
                    )
                self._lsa_runtime = register_nccl_lsa_window(
                    self._lsa_context, self._lsa_arena,
                )
                self._lsa_context = None
                window = self._lsa_runtime.window

        self._lsa_arena_regions = regions
        if self.impl.enable_nccl_gin:
            if self._gin_window_layout is None:
                raise AssertionError("GIN layout was not constructed")
            self._gin_internode_send_count = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.internode_send_count_offset,
                self._gin_window_layout.internode_send_count_shape,
                torch.int32,
            )
            self._gin_internode_recv_splits = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.internode_recv_splits_offset,
                self._gin_window_layout.internode_recv_splits_shape,
                torch.int32,
            )
            assert self._gin_window_layout.relay_send_buffer_offset is not None
            assert self._gin_window_layout.relay_send_buffer_shape is not None
            assert self._gin_window_layout.relay_recv_buffer_offset is not None
            assert self._gin_window_layout.relay_recv_buffer_shape is not None
            self._gin_relay_send_buffer = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.relay_send_buffer_offset,
                self._gin_window_layout.relay_send_buffer_shape,
                torch.uint8,
            )
            self._gin_relay_recv_buffer = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.relay_recv_buffer_offset,
                self._gin_window_layout.relay_recv_buffer_shape,
                torch.uint8,
            )
            assert self._gin_window_layout.intra_node_reduce_buf_offset is not None
            assert self._gin_window_layout.intra_node_reduce_buf_shape is not None
            assert self._gin_window_layout.internode_combine_buffer_offset is not None
            assert self._gin_window_layout.internode_combine_buffer_shape is not None
            self._gin_intra_node_reduce_buf = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.intra_node_reduce_buf_offset,
                self._gin_window_layout.intra_node_reduce_buf_shape,
                torch.bfloat16,
            )
            self._gin_internode_combine_buffer = self._arena_view(
                self._lsa_arena,
                self._gin_window_layout.internode_combine_buffer_offset,
                self._gin_window_layout.internode_combine_buffer_shape,
                torch.bfloat16,
            )
        self.my_activation = self._arena_view(
            self._lsa_arena, regions["activation"][0],
            activation_shape, _DataDtype,
        )
        self.my_activation_sf = self._arena_view(
            self._lsa_arena, regions["activation_sf"][0],
            activation_sf_shape, _ScaleDtype,
        )
        self.my_topk_weights = self._arena_view(
            self._lsa_arena, regions["topk_weights"][0],
            topk_weights_shape, torch.float32,
        )
        self.shared_workspace = self._arena_view(
            self._lsa_arena, regions["shared_workspace"][0],
            (shared_ws_bytes,), torch.uint8,
        )
        self.my_topk_idx = torch.zeros(
            (tokens, topk), dtype=torch.int64, device="cuda",
        )
        if self.impl.fc2_reduces_topk:
            self.output_activation = self._arena_view(
                self._lsa_arena, regions["output_activation"][0],
                (tokens, hidden), self.problem.fc2_output_dtype,
            )
        else:
            self.output_activation = torch.empty(
                (tokens, hidden),
                dtype=self.problem.fc2_output_dtype,
                device="cuda",
            )

        self.symmetric_base = int(self._lsa_arena.data_ptr())
        ranks_per_node = int(self.impl.ranks_per_node or self.world_size)
        if _NO_DIST:
            self.peer_offsets_list = tuple(0 for _ in range(ranks_per_node))
        else:
            node_base = (self.rank // ranks_per_node) * ranks_per_node
            peer_ptrs = []
            for peer in range(node_base, node_base + ranks_per_node):
                peer_ptr = window.get_peer_device_pointer(peer)
                if peer_ptr is None:
                    raise RuntimeError(
                        f"NCCL LSA peer {peer} is unreachable from rank {self.rank}"
                    )
                peer_ptrs.append(int(peer_ptr))
            self.peer_offsets_list = tuple(
                peer_ptr - self.symmetric_base for peer_ptr in peer_ptrs
            )

        if self.rank == 0:
            runtime_kind = "GIN+LSA" if self.impl.enable_nccl_gin else "LSA"
            print(
                f"NCCL {runtime_kind} arena ready: bytes={arena_bytes} "
                f"regions={regions} ranks_per_node={ranks_per_node}",
                flush=True,
            )

    def close(self) -> None:
        """Release registered windows before their owning communicators."""
        try:
            if self._gin_runtime is not None:
                torch.cuda.synchronize()
                self._gin_runtime.close()
                self._gin_runtime = None
            if self._lsa_runtime is not None:
                torch.cuda.synchronize()
                self._lsa_runtime.close()
                self._lsa_runtime = None
        finally:
            if self._gin_context is not None:
                self._gin_context.close()
                self._gin_context = None
            if self._lsa_context is not None:
                self._lsa_context.close()
                self._lsa_context = None

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

    def _report_cuda_event_samples(
        self,
        event_pairs: List[Tuple[torch.cuda.Event, torch.cuda.Event]],
    ) -> Optional[List[int]]:
        """Report iteration-wise rank maxima from paired CUDA events."""
        local_us = [
            float(start.elapsed_time(end)) * 1000.0
            for start, end in event_pairs
        ]
        gathered_us: List[List[float]] = [local_us]
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            gathered_us = [
                [] for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(gathered_us, local_us)
        if self.rank != 0:
            return None
        sample_counts = {len(samples) for samples in gathered_us}
        if len(sample_counts) != 1 or not sample_counts or 0 in sample_counts:
            raise ValueError(
                "CUDA event sample-count mismatch across ranks: "
                f"{sorted(sample_counts)}"
            )
        matrix = np.asarray(gathered_us, dtype=np.float64)
        if not np.all(np.isfinite(matrix)):
            raise ValueError("CUDA event samples contain non-finite values")
        iteration_max_us = np.max(matrix, axis=0)
        max_ranks = np.argmax(matrix, axis=0)
        rank_means_us = np.mean(matrix, axis=1)
        max_rank_mean_rank = int(np.argmax(rank_means_us))
        max_rank_mean_samples = matrix[max_rank_mean_rank]
        critical_ranks = ",".join(map(str, max_ranks.tolist()))
        launch_mode = (
            "synchronized"
            if self._perf_sync_each_launch else "back_to_back"
        )
        profiler_collection = (
            "off"
            if self.config.diagnostics.cuda_event_only
            else "on"
        )
        print(
            "[pipeline-cuda-event-global] "
            f"ranks={matrix.shape[0]} samples={matrix.shape[1]} "
            f"iter_rank_max_mean_us={np.mean(iteration_max_us):.3f} "
            f"iter_rank_max_median_us={np.median(iteration_max_us):.3f} "
            f"iter_rank_max_p90_us={np.percentile(iteration_max_us, 90):.3f} "
            f"iter_rank_max_min_us={np.min(iteration_max_us):.3f} "
            f"iter_rank_max_max_us={np.max(iteration_max_us):.3f} "
            f"max_rank_mean_rank={max_rank_mean_rank} "
            f"max_rank_mean_us={rank_means_us[max_rank_mean_rank]:.3f} "
            f"max_rank_median_us={np.median(max_rank_mean_samples):.3f} "
            f"critical_ranks={critical_ranks} "
            "headline=mean_of_iteration_rank_max "
            "deepep_compatible_headline=max_of_rank_means "
            "timing=cuda_event_pipeline_bracket "
            f"torch_profiler_collection={profiler_collection} "
            f"launch_mode={launch_mode}",
            flush=True,
        )
        return [int(rank) for rank in max_ranks.tolist()]

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
            # Aggregate sums cover ``num_iters`` launches in the timed region;
            # divide down to a per-launch target-kernel time. In synchronized
            # mode the barriers are outside the matched kernel_cutlass events.
            mega_time_us /= max(1, num_iters)
            topk_time_us /= max(1, num_iters)
            if mega_time_us == 0.0:
                mega_time_us = float("nan")
            if topk_time_us == 0.0:
                topk_time_us = float("nan")
        finite_parts = [
            time_us
            for time_us in (mega_time_us, topk_time_us)
            if np.isfinite(time_us)
        ]
        total_time_us = sum(finite_parts) if finite_parts else float("nan")

        dist_active = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
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
            total_values = [rank_times[2] for rank_times in rank_times_us]
            finite_totals = [
                (rank, time_us)
                for rank, time_us in enumerate(total_values)
                if np.isfinite(time_us)
            ]
            if finite_totals:
                min_duration_rank, min_total_us = min(
                    finite_totals, key=lambda x: x[1]
                )
                max_duration_rank, max_total_us = max(
                    finite_totals, key=lambda x: x[1]
                )
            else:
                min_duration_rank = -1
                max_duration_rank = -1
                min_total_us = float("nan")
                max_total_us = float("nan")

            def _fmt(time_us: float) -> str:
                return f"{time_us:.2f} us" if np.isfinite(time_us) else "n/a"

            # ``problem.intermediate`` is the FC1 gate+up output width J,
            # not the post-SwiGLU model dimension I=J/2. Count the actual
            # matrices executed by this runner: FC1 HxJ plus FC2 (J/2)xH.
            gemm_flops_per_routed_token = (
                2 * self.problem.hidden * self.problem.intermediate
                + 2
                * (self.problem.intermediate // 2)
                * self.problem.hidden
            )

            def _format_rank_summary(rank: int, total_us: float) -> str:
                if (
                    rank < 0
                    or self._scheduled_routed_tokens_per_rank is None
                    or rank >= len(self._scheduled_routed_tokens_per_rank)
                    or not np.isfinite(total_us)
                    or total_us <= 0.0
                ):
                    return "n/a"
                routed_tokens = self._scheduled_routed_tokens_per_rank[rank]
                algorithmic_flops = routed_tokens * gemm_flops_per_routed_token
                effective_tflops = algorithmic_flops / total_us / 1.0e6
                return (
                    f"rank {rank}: total={_fmt(total_us)} "
                    f"scheduled_routed_tokens={routed_tokens} "
                    f"algorithmic_gemm_tflops={effective_tflops:.4f}"
                )

            print("---- torch profiler target pipeline CUDA time ----")
            print(
                "  timing_semantics: "
                + (
                    "target-kernel CUDA time under synchronized launches "
                    "(not back-to-back throughput)"
                    if self._perf_sync_each_launch
                    else "back-to-back steady-state latency"
                )
            )
            print(
                "  flop_convention: actual runner GEMMs, "
                "FC1 H->J plus FC2 J/2->H, "
                f"{gemm_flops_per_routed_token} FLOP/routed-token"
            )
            print(
                "  min_duration_rank="
                f"{_format_rank_summary(min_duration_rank, min_total_us)}"
            )
            print(
                "  max_duration_rank="
                f"{_format_rank_summary(max_duration_rank, max_total_us)}"
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

        One compiled-kernel call owns the full pipeline. By default it launches
        TopkReduce separately on the same stream; the opt-in ordered GIN tail
        reduction folds that worker into the megakernel itself.
        """
        if self._compiled_kernel is None:
            raise RuntimeError("compiled kernel is not available")
        hang_timers = self._start_hang_dump_watchdog("launch_sequence")
        try:
            self._launch_target_kernels_body(runtime_kwargs)
        finally:
            for t in hang_timers:
                t.cancel()

    def _launch_target_kernels_body(self, runtime_kwargs) -> None:
        def _launch() -> None:
            self._reset_pipeline_timing()
            self._compiled_kernel(**runtime_kwargs)

        def _distributed_barrier() -> None:
            if (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            ):
                torch.distributed.barrier()

        def _launch_synchronized(
            phase: str,
            iteration: int,
            *,
            event_pair: Optional[
                Tuple[torch.cuda.Event, torch.cuda.Event]
            ] = None,
            capture_timing: bool = False,
        ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
            print(
                f"[perf-launch] BEGIN rank={self.rank} phase={phase} "
                f"iteration={iteration}",
                flush=True,
            )
            if event_pair is not None:
                event_pair[0].record()
            _launch()
            if event_pair is not None:
                event_pair[1].record()
            snapshot = (
                self._capture_pipeline_timing() if capture_timing else None
            )
            # This is the first host API that must observe a device fault from
            # this launch. If it raises, the missing END marker attributes the
            # failure to this exact phase/iteration. Surviving ranks may wait
            # in the barrier until the launcher tears the failed job down.
            torch.cuda.synchronize()
            _distributed_barrier()
            if self._validate_each_launch:
                self._validate_relaunch_state(phase, iteration)
                _distributed_barrier()
            print(
                f"[perf-launch] END rank={self.rank} phase={phase} "
                f"iteration={iteration}",
                flush=True,
            )
            return snapshot

        if not self._use_torch_profiler:
            _launch()
            torch.cuda.synchronize()
            self._report_pipeline_timing()
            self._report_fc_tile_timeline()
            return

        if self._perf_sync_each_launch:
            # Drain compile/setup work and align every rank before launch epoch
            # zero. Later epochs are aligned by the post-launch barrier.
            torch.cuda.synchronize()
            _distributed_barrier()
            if self.rank == 0:
                print(
                    "[perf-launch] mode=synchronized-launch "
                    "timing_semantics=not-back-to-back-throughput",
                    flush=True,
                )

        for iteration in range(self._perf_warmup):
            if self._perf_sync_each_launch:
                _launch_synchronized("warmup", iteration)
            else:
                _launch()
        if not self._perf_sync_each_launch:
            torch.cuda.synchronize()

        # Default timing remains back-to-back. The opt-in synchronized mode is
        # a lifecycle workaround/diagnostic with different timing semantics.
        n_iters = max(1, self._perf_iters)
        event_pairs: List[
            Tuple[torch.cuda.Event, torch.cuda.Event]
        ] = []
        timing_snapshots: List[Tuple[torch.Tensor, torch.Tensor]] = []
        # Keep Python GC pauses out of the measured launch sequence.
        gc.collect()
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            profiler_context = (
                nullcontext(None)
                if self.config.diagnostics.cuda_event_only
                else torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                )
            )
            with profiler_context as prof:
                # Profiler startup is per-process. Align only after every rank has
                # entered the context so its skew is not counted as NVSHMEM spin
                # time in the first timed kernel.
                _distributed_barrier()
                torch.cuda.synchronize()
                for iteration in range(n_iters):
                    event_pair = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    event_pairs.append(event_pair)
                    if self._perf_sync_each_launch:
                        snapshot = _launch_synchronized(
                            "timed",
                            iteration,
                            event_pair=event_pair,
                            capture_timing=self.config.diagnostics.pipeline_timing,
                        )
                    else:
                        event_pair[0].record()
                        _launch()
                        event_pair[1].record()
                        snapshot = self._capture_pipeline_timing()
                    if snapshot is not None:
                        timing_snapshots.append(snapshot)
                if not self._perf_sync_each_launch:
                    torch.cuda.synchronize()
        finally:
            if gc_was_enabled:
                gc.enable()
        full_call_critical_ranks = self._report_cuda_event_samples(
            event_pairs
        )
        if prof is not None:
            self._report_torch_profiler_kernel_time(prof, num_iters=n_iters)
        if timing_snapshots:
            # Preserve the legacy one-sample records for existing parsers while
            # reporting all timed launches under a distinct summary prefix.
            self._report_pipeline_timing(timing_snapshots[-1])
            self._report_fc_tile_timeline()
            self._report_pipeline_timing_samples(
                timing_snapshots,
                full_call_critical_ranks=full_call_critical_ranks,
            )

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

        lifecycle_trace_enabled = self.config.diagnostics.lifecycle_trace

        def _trace_lifecycle(phase: str) -> None:
            if lifecycle_trace_enabled:
                print(
                    f"[lifecycle] rank={self.rank} phase={phase}",
                    flush=True,
                )

        _trace_lifecycle("run_kernel.enter")

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
                f"hidden axis with zero bytes before peer-visible staging."
            )

        _trace_lifecycle("imports.begin")
        import cuda.bindings.driver as cuda
        import cutlass.cute as cute
        import cutlass.torch as cutlass_torch
        import cutlass.utils as utils

        from ..common.megamoe_constants import SfPaddingBlock
        from .megamoe_kernel import Sm100MegaMoEKernel
        from ..comm.sym_buffer import SymBufferHost

        _trace_lifecycle("imports.end")
        # -- 1. Kernel instance --
        _trace_lifecycle("ensure_kernel.begin")
        self._ensure_kernel()
        _trace_lifecycle("ensure_kernel.end")
        if self._max_active_clusters is None:
            raise RuntimeError("kernel occupancy was not initialized")
        max_active_clusters = self._max_active_clusters

        # -- 2. Workspaces (local cuda + sym-heap) --
        _trace_lifecycle("allocate_workspaces.begin")
        self.allocate_workspaces()
        _trace_lifecycle("allocate_workspaces.end")

        # -- 3. Torch -> cute --
        _trace_lifecycle("argument_lowering.begin")
        # Same align defaults as runner_fc12: 16 B for general tensors,
        # 4 B for the i32 / fp32 counter / sized buffers.
        def _to_cute(tensor: torch.Tensor, assumed_align: int = 16, force_static_layout = False):
            cute_tensor = cutlass_torch.from_dlpack(
                tensor, assumed_align=assumed_align,
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

        if self.impl.enable_nccl_gin:
            if (
                self._gin_runtime is None
                or self._gin_internode_send_count is None
                or self._gin_internode_recv_splits is None
                or self._gin_internode_combine_buffer is None
            ):
                raise RuntimeError(
                    "GIN runtime and registered metadata views must exist "
                    "before compiling the Stage 1 kernel"
                )
            from nccl.core.device import cute as nccl_cute

            gin_dev_comm_cute = nccl_cute.DevComm(
                self._gin_runtime.dev_comm,
            )
            gin_window_cute = nccl_cute.Window(
                self._gin_runtime.window,
            )
            gin_internode_send_count_cute = _to_cute(
                self._gin_internode_send_count,
                assumed_align=128,
                force_static_layout=True,
            )
            gin_internode_recv_splits_cute = _to_cute(
                self._gin_internode_recv_splits,
                assumed_align=128,
                force_static_layout=True,
            )
            gin_internode_combine_buffer_cute = _to_cute(
                self._gin_internode_combine_buffer,
                assumed_align=128,
                force_static_layout=True,
            )
        else:
            gin_dev_comm_cute = None
            gin_window_cute = None
            gin_internode_send_count_cute = None
            gin_internode_recv_splits_cute = None
            gin_internode_combine_buffer_cute = None

        # Opaque byte workspaces -> raw uint8 gmem base pointers (no tensor). The
        # kernel addresses them by base + Int64 byte offset; a tensor's shape would
        # be ignored AND overflow cute's 32-bit memref shape field once the
        # internalized combine staging pushes shared_workspace past 2 GiB.
        from cutlass.cute.typing import AddressSpace as _AddressSpace

        def _to_cute_ptr(tensor: torch.Tensor, assumed_align: int = 16):
            return cute.runtime.make_ptr(
                cutlass.Uint8, tensor.data_ptr(), _AddressSpace.gmem,
                assumed_align=assumed_align,
            )

        local_workspace_cute = _to_cute_ptr(self.local_workspace)
        shared_workspace_cute = _to_cute_ptr(self.shared_workspace)

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        # -- 4. cute.compile --
        # Runtime payload for the peer pointer mapper. Relay mode carries only
        # logical-node peers and traps on a cross-node global rank; legacy mode
        # retains the world-wide NVSHMEM mapper.
        if self.symmetric_base is None or self.peer_offsets_list is None:
            raise RuntimeError("peer pointer offsets were not initialized")
        if self.impl.enable_internode_relay:
            from ..comm.lsa_peer_mapper import LsaPeerMapperHost

            ranks_per_node = int(self.impl.ranks_per_node or self.world_size)
            peer_rank_ptr_mapper_host = LsaPeerMapperHost(
                base_addr=self.symmetric_base,
                offsets=tuple(self.peer_offsets_list),
                rank_idx=self.rank,
                num_max_ranks=ranks_per_node,
            )
            peer_rank_ptr_mapper_host.validate()
        else:
            from ..comm.sym_buffer import SymBufferHost

            mc_signal_base = (
                self.mc_base + int(self._kernel._shared_offsets["nvlink_barrier_signal"])
                if self.mc_base else 0
            )
            peer_rank_ptr_mapper_host = SymBufferHost(
                base_addr=self.symmetric_base,
                offsets=tuple(self.peer_offsets_list),
                rank_idx=self.rank,
                num_max_ranks=self.world_size,
                mc_base=mc_signal_base,
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
            gin_dev_comm=gin_dev_comm_cute,
            gin_window=gin_window_cute,
            gin_internode_send_count=gin_internode_send_count_cute,
            gin_internode_recv_splits=gin_internode_recv_splits_cute,
            gin_internode_combine_buffer=(
                gin_internode_combine_buffer_cute
            ),
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
        compile_target = self._kernel
        if not self.impl.enable_nccl_gin:
            # Use a dedicated entry point so GIN resources are not part of the
            # compiled non-GIN function's runtime ABI at all.
            compile_target = self._kernel.launch_without_gin
            for gin_arg in (
                "gin_dev_comm",
                "gin_window",
                "gin_internode_send_count",
                "gin_internode_recv_splits",
                "gin_internode_combine_buffer",
            ):
                runtime_kwargs.pop(gin_arg)

        compile_kwargs = dict(runtime_kwargs)
        compile_kwargs["max_active_clusters"] = max_active_clusters
        if self.misc.enable_iket and "iket" not in os.environ.get(
            "CUTE_DSL_COMPILER_OPT", ""
        ).split(","):
            raise RuntimeError(
                "IKET profiling requires CUTE_DSL_COMPILER_OPT=iket before "
                "the Python process starts"
            )

        _trace_lifecycle("argument_lowering.end")
        _trace_lifecycle("compile.begin")
        self._compiled_kernel = cute.compile(compile_target, **compile_kwargs)
        _trace_lifecycle("compile.end")

        # The separate-kernel-reduce tail is now launched inside the kernel's
        # ``__call__`` (same stream, right after the mega kernel), so the runner
        # no longer compiles or chains a standalone reduce.

        # -- 5. Launch --
        #
        # ``profile_friendly`` mode wraps the launch in five steps:
        #
        #   (a) ``torch.cuda.synchronize()`` before the launch drains every
        #       setup-side GPU operation off the stream.
        #
        #   (b) A cross-rank barrier before cudaProfilerStart aligns completed
        #       setup. Its NCCL kernel stays outside the capture window.
        #
        #   (c) A dedicated CPU/Gloo group is created and warmed before
        #       cudaProfilerStart. Its host-only barrier runs after profiler
        #       startup and immediately before the target launch. This avoids
        #       comparing wall clocks from different hosts, whose clock skew
        #       would otherwise appear as an in-kernel barrier wait. Because
        #       this group is explicitly Gloo, the rendezvous launches no CUDA
        #       work into the capture window.
        #
        #   (d) ``cudaProfilerStart`` / ``cudaProfilerStop`` select the target
        #       launch for Nsight Systems and Nsight Compute capture-range support.
        #
        #   (e) A cross-rank barrier after cudaProfilerStop keeps a fast rank
        #       from freeing peer-visible storage while another rank is still
        #       using it.
        #
        # ``_NO_DIST=1`` and uninitialized ``torch.distributed`` paths skip
        # both rendezvous points so single-rank debugging is unchanged.
        #
        # Outside ``profile_friendly`` mode the launch path is unchanged.
        _trace_lifecycle("launch_sequence.begin")
        if self.misc.profile_friendly:
            import nvtx

            torch.cuda.synchronize()
            _dist_active = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            profile_host_group = None
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
                profile_host_group = torch.distributed.new_group(backend="gloo")
                # Warm lazy Gloo setup before the profiler capture starts.
                torch.distributed.barrier(group=profile_host_group)
            # Nsight Systems and Nsight Compute can select exactly this launch through the CUDA
            # profiler API. The pre-launch collectives are intentionally outside
            # the capture window; the compiled call synchronizes before Stop.
            torch.cuda.profiler.start()
            try:
                if profile_host_group is not None:
                    torch.distributed.barrier(group=profile_host_group)
                with nvtx.annotate("cute_dsl_prof"):
                    self._launch_target_kernels_with_optional_torch_profiler(
                        runtime_kwargs,
                    )
            finally:
                torch.cuda.synchronize()
                torch.cuda.profiler.stop()
            if _dist_active:
                torch.distributed.barrier()
                torch.cuda.synchronize()
        else:
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
            )
        _trace_lifecycle("launch_sequence.end")

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
            ref_reduced = _bf16_seq_sum_k(
                self.combine_output_ref
            ).to(torch.float32)
        else:
            ref_reduced = self.combine_reduced_output_ref.to(torch.float32)

        if self.impl.enable_nccl_gin:
            # Match the production reduction tree exactly: local-node expert
            # terms remain individual BF16 cells, while each remote node first
            # reduces its terms in FP32 and casts one node partial to BF16.
            # TopkReduce then adds local cells followed by remote-node partials
            # in FP32 and casts the final activation to BF16.
            protocol_acc = torch.zeros_like(ref_reduced, dtype=torch.float32)
            ref_terms = self.combine_output_ref
            topk = self.my_topk_idx
            my_node = self.rank // self.impl.ranks_per_node
            term_weights = None
            if self.misc.ref_compute_graph == "transformers":
                term_weights = self.my_topk_weights.to(torch.float32)
            for k in range(ref_terms.shape[1]):
                expert_node = (
                    topk[:, k]
                    // self.problem.num_experts_per_rank
                    // self.impl.ranks_per_node
                )
                term = ref_terms[:, k].to(torch.float32)
                if term_weights is not None:
                    term = term * term_weights[:, k, None]
                local_mask = expert_node == my_node
                protocol_acc[local_mask] += term[local_mask]
            for remote_node in range(self.world_size // self.impl.ranks_per_node):
                if remote_node == my_node:
                    continue
                node_partial = torch.zeros_like(
                    protocol_acc, dtype=torch.float32
                )
                for k in range(ref_terms.shape[1]):
                    expert_node = (
                        topk[:, k]
                        // self.problem.num_experts_per_rank
                        // self.impl.ranks_per_node
                    )
                    term = ref_terms[:, k].to(torch.float32)
                    if term_weights is not None:
                        term = term * term_weights[:, k, None]
                    remote_mask = expert_node == remote_node
                    node_partial[remote_mask] += term[remote_mask]
                protocol_acc += node_partial.to(torch.bfloat16).to(torch.float32)
            protocol_ref = protocol_acc.to(torch.bfloat16).to(torch.float32)
            ideal_diff = (protocol_ref - ref_reduced).abs()
            print(
                f"[rank {self.rank}] Stage 4 protocol rounding: "
                f"max_diff={ideal_diff.max().item():.4g} "
                f"mean_diff={ideal_diff.mean().item():.4g}"
            )
            ref_reduced = protocol_ref

        diff = (actual_reduced - ref_reduced).abs()

        if self.rank == 0 and self.config.diagnostics.peek_workspace:
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
                    actual_reduced, self.combine_output_ref,
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
                        torch.empty_like(pass_tensor)
                        for _ in range(self.world_size)
                    ]
                    torch.distributed.all_gather(gathered_passes, pass_tensor)
                    rank_bitwise_passes = [
                        bool(t.item()) for t in gathered_passes
                    ]
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
                    sample = bad[:min(8, n_local_bad)].cpu().tolist()
                    print(
                        f"[rank {self.rank}] form-B bitwise miss: "
                        f"{n_local_bad} cell(s); sample "
                        f"(token, hidden)={sample}"
                    )
                    sys.stdout.flush()
                failed_ranks = [
                    r for r, ok in enumerate(rank_bitwise_passes) if not ok
                ]
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
                cells_ref_K = self.combine_output_ref[
                    suspect_t, :, suspect_h
                ].to(torch.float32)
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
                actual_reduced, ref_reduced, atol=atol, rtol=rtol,
            )
        else:
            bitwise_mismatch_count = int((diff != 0).sum().item())
            local_pass = bitwise_mismatch_count == 0
        local_max_diff = diff.max().detach().reshape(1)
        rank_max_diffs = [float(local_max_diff.item())]
        rank_passes = [local_pass]
        _dist_active = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
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
            if (
                self.impl.enable_nccl_gin
                and self.local_workspace is not None
                and self.shared_workspace is not None
                and self._gin_internode_combine_buffer is not None
            ):
                phase_spec = self._kernel._local_region_by_name[
                    "combine_phase_snapshot"
                ]
                phase_off = self._kernel._local_offsets[
                    "combine_phase_snapshot"
                ]
                phase = int(
                    self.local_workspace[
                        phase_off:phase_off + phase_spec.nbytes
                    ].view(torch.int32)[0].item()
                ) & 1
                combine_spec = self._kernel._shared_region_by_name[
                    "combine_quant"
                ]
                combine_off = self._kernel._shared_offsets["combine_quant"]
                local_cells = self.shared_workspace[
                    combine_off:combine_off + combine_spec.nbytes
                ].view(torch.bfloat16).reshape(combine_spec.shape)[phase]
                remote_rows = self._gin_internode_combine_buffer[phase]
                slot_map = None
                # Slot-indexed landing plane: gather each token's row
                # through the device-written token -> message-slot map.
                slot_spec = self._kernel._local_region_by_name[
                    "token_node_slot"
                ]
                slot_off = self._kernel._local_offsets["token_node_slot"]
                slot_map = self.local_workspace[
                    slot_off:slot_off + slot_spec.nbytes
                ].view(torch.int32).reshape(slot_spec.shape)
                topk = self.my_topk_idx
                ref_terms = self.combine_output_ref
                my_node = self.rank // self.impl.ranks_per_node
                local_max = 0.0
                local_cells_checked = 0
                for k in range(ref_terms.shape[1]):
                    nodes = (
                        topk[:, k]
                        // self.problem.num_experts_per_rank
                        // self.impl.ranks_per_node
                    )
                    mask = nodes == my_node
                    if bool(mask.any()):
                        local_max = max(
                            local_max,
                            float(
                                (
                                    local_cells[mask, k].float()
                                    - ref_terms[mask, k].float()
                                ).abs().max().item()
                            ),
                        )
                        local_cells_checked += int(mask.sum().item())
                remote_summaries = []
                for slot in range(self._gin_internode_combine_buffer.shape[1]):
                    remote_node = slot + (1 if slot >= my_node else 0)
                    expected = torch.zeros_like(
                        remote_rows[slot], dtype=torch.float32
                    )
                    has_remote = torch.zeros(
                        (topk.shape[0],), device=topk.device, dtype=torch.bool
                    )
                    for k in range(ref_terms.shape[1]):
                        nodes = (
                            topk[:, k]
                            // self.problem.num_experts_per_rank
                            // self.impl.ranks_per_node
                        )
                        mask = nodes == remote_node
                        expected[mask] += ref_terms[mask, k].float()
                        has_remote |= mask
                    remote_max = 0.0
                    if bool(has_remote.any()):
                        actual_rows = remote_rows[slot, has_remote]
                        if slot_map is not None:
                            token_ids = torch.nonzero(
                                has_remote, as_tuple=False
                            ).flatten()
                            actual_rows = remote_rows[slot][
                                slot_map[slot, token_ids].long()
                            ]
                        remote_max = float(
                            (
                                actual_rows.float()
                                - expected[has_remote].to(torch.bfloat16).float()
                            ).abs().max().item()
                        )
                    remote_summaries.append(
                        f"slot{slot}->node{remote_node}:tokens="
                        f"{int(has_remote.sum().item())},max_diff={remote_max:.4g}"
                    )
                print(
                    f"[rank {self.rank}] Stage 4 debug: phase={phase} "
                    f"local_cells={local_cells_checked} local_max_diff={local_max:.4g} "
                    + " ".join(remote_summaries)
                )
                print(
                    f"[rank {self.rank}] Stage 4 token0: "
                    f"topk={topk[0].tolist()} "
                    f"actual={self.output_activation[0, :8].float().tolist()} "
                    f"local_cells={local_cells[0, :, :8].float().tolist()} "
                    f"remote_rows={remote_rows[:, 0, :8].float().tolist()} "
                    f"ref_terms={ref_terms[0, :, :8].float().tolist()}"
                )
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
            print(
                f"Validation PASSED on all ranks "
                f"({diff_summary})"
            )

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
        from .megamoe_kernel import _layout_regions

        local_specs = self._kernel._local_region_specs
        shared_specs = self._kernel._shared_region_specs
        local_offsets, _ = _layout_regions(local_specs)
        shared_offsets, _ = _layout_regions(shared_specs)
        local_by_name = self._kernel._local_region_by_name
        shared_by_name = self._kernel._shared_region_by_name

        def _region(ws, by_name, offsets, name, torch_dtype):
            if name not in by_name and name in shared_by_name:
                ws, by_name, offsets = self.shared_workspace, shared_by_name, shared_offsets
            spec = by_name[name]
            off = offsets[name]
            return ws[off:off + spec.nbytes].view(torch_dtype)

        # ---- 1. expert routing counters (low32=tokens, high32=publishers*ranks).
        ercs = _region(
            self.shared_workspace, shared_by_name, shared_offsets,
            "expert_recv_count_sum", torch.int64,
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
            self.local_workspace, local_by_name, local_offsets,
            "token_src_metadata", torch.int32,
        )
        md_u32 = mbuf.reshape(-1, 3)
        n_slots = md_u32.shape[0]
        sample_rows = sorted(set(
            list(range(min(8, n_slots)))
            + [n_slots // 2 - 1, n_slots // 2, n_slots // 2 + 1]
            + list(range(max(0, n_slots - 4), n_slots))
        ))
        print(
            f"---- token_src_metadata (pool_slot -> (src_rank, src_token, src_topk)) "
            f"n_slots={n_slots} ----"
        )
        for s in sample_rows:
            r = md_u32[s].cpu().tolist()
            print(
                f"  slot[{s}]: src_rank={r[0]} src_token={r[1]} src_topk={r[2]}"
            )

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
            f"---- duplicate metadata triples (n_collision_triples="
            f"{n_collisions}) ----"
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
                slot_sample = slot_ids[:min(8, slot_ids.numel())].tolist()
                tag = ""
                if src_rank == 0 and src_token == 0 and src_topk == 0:
                    tag = "  [LIKELY PADDING / UNWRITTEN]"
                print(
                    f"  (src_rank={src_rank}, src_token={src_token}, "
                    f"src_topk={src_topk}) x {c}{tag}"
                )
                print(
                    f"    first {len(slot_sample)} slot ids: {slot_sample}"
                )
                shown += 1
                if shown >= shown_limit:
                    remaining = n_collisions - shown
                    if remaining > 0:
                        print(
                            f"  ... ({remaining} more collision triples elided)"
                        )
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
            fc1_off_rc:fc1_off_rc + fc1_spec_rc.nbytes
        ].view(torch.uint8)
        row_bytes_rc = (
            fc1_spec_rc.shape[1] // 2 if len(fc1_spec_rc.shape) >= 2 else 0
        )
        rc_ready = (
            row_bytes_rc > 0
            and fc1_raw_rc.numel() % row_bytes_rc == 0
        )
        if rc_ready:
            fc1_2d_rc = fc1_raw_rc.view(-1, row_bytes_rc).cpu()
            valid_slot_mask = (fc1_2d_rc != 0).any(dim=1)
            T = self.problem.num_tokens_per_rank
            K = self.problem.num_topk
            # Multi-rank: also need src_rank to disambiguate (same
            # (t, k) on different source ranks lives in different
            # pool slots on the receiver).  Single-rank degenerates to
            # the previous behaviour (src_rank == 0 everywhere).
            seen_triples = set(
                map(tuple, md_cpu[valid_slot_mask, :].tolist())
            )
            # Expected universe: every (src_rank, src_token, src_topk)
            # whose routed expert lands on THIS rank.  Per-rank
            # routing is built by replaying ``_generate_topk_idx_*``;
            # to avoid recomputing it the harness keeps the global
            # tensor in ``self._global_topk_idx`` (shape
            # ``(num_ranks, T, K)``).  Filter by
            # ``expert // num_experts_per_rank == self.rank``.
            num_experts_per_rank = (
                self.problem.num_total_experts // self.world_size
            )
            expected_triples = set()
            if (
                hasattr(self, "_global_topk_idx")
                and self._global_topk_idx is not None
            ):
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
                f"---- expected-vs-actual reverse check skipped "
                f"(fc1_output peek unavailable) ----"
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
        fc1_raw = self.local_workspace[
            fc1_off:fc1_off + fc1_spec.nbytes
        ].view(torch.uint8)
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
                print(
                    f"  first 16 zero rows: {zero_rows[:16]}"
                )
                print(
                    f"  last 16 zero rows : {zero_rows[-16:]}"
                )

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
            sf_raw = self.local_workspace[
                sf_off:sf_off + sf_spec.nbytes
            ].view(torch.uint8).cpu()

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
                r for r in range(
                    max(0, n_rows_total - n_show_rows), n_rows_total,
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
                    [
                        _sf_byte_offset(r, sb)
                        for sb in range(K_sf_blocks_total)
                    ],
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
                head_cells = [
                    f"{v:+.2f}" for v in dequant[:n_cells_actual].tolist()
                ]
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
                    [full_lo, full_hi], dim=-1,
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
                    n_valid, K_sf_blocks_total,
                )
                sf_fp32_all = (
                    sf_bytes_all.view(torch.float8_e4m3fn).to(torch.float32)
                )
                valid_fp4 = full_fp4_vals[valid_row_ids]
                dequant_all = (
                    valid_fp4.reshape(
                        n_valid, K_sf_blocks_total, sf_vec_size,
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
            raw = self.local_workspace[off:off + spec.nbytes].view(torch.uint8)
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
                (host_diff_per_token > 0).nonzero(as_tuple=False)
                .squeeze(-1).tolist()
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
            num_experts_per_rank = (
                self.problem.num_total_experts // self.world_size
            )
            expert_recv_local = [
                int(ercs[i].item()) & 0xFFFFFFFF
                for i in range(n_local_e)
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
                    pieces.append(
                        f"k{k_idx}=e{e_int}(recv={_recv_for(e_int)},{wt})"
                    )
                print(
                    f"  token[{t}] diff_sum={ds:.4g}  " + " ".join(pieces)
                )

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
                self.shared_workspace, shared_by_name, shared_offsets,
                "combine_quant", self.problem.fc2_output_dtype,
            )
            combine_shape = shared_by_name["combine_quant"].shape
            if self.impl.enable_nccl_gin:
                # GIN uses phase-double-buffered local cells. Match the
                # post-kernel TopkReduce by inspecting only the phase that the
                # fused kernel snapshotted before advancing the rail generation.
                phase_raw = _region(
                    self.local_workspace, local_by_name, local_offsets,
                    "combine_phase_snapshot", torch.int32,
                )
                phase = int(phase_raw[0].item()) & 1
                combine_quant_raw = combine_quant_raw.reshape(combine_shape)[phase]
            actual_full = (
                combine_quant_raw.reshape(self.combine_output_ref.shape)
                .to(torch.float32)
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
            for t, ds in zip(bad_tokens, bad_diff_sums):
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
                        f"    (hidden={h}): "
                        f"actual={a:.4g}  ref={r:.4g}  diff={d:.4g}"
                    )
        else:
            print(
                "---- worst-diff tokens (top "
                f"{n_worst_tokens} by sum over (topk, hidden); per token: "
                f"top {n_worst_cells_per_token} (topk, hidden) cells) ----"
            )
            for t, ds in zip(bad_tokens, bad_diff_sums):
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

    def _report_actual_scheduler_group_layout(
        self,
        actual_scheduled: np.ndarray,
        actual_intra: np.ndarray,
        actual_recv_per_source: np.ndarray,
    ) -> None:
        """Print one deterministic record of the device scheduler layout.

        ``expert_scheduled_count_snapshot`` is the post-redistribution token-size
        view consumed by the scheduler. MegaMoE rounds its intranode and relay
        segments independently, so the earlier intranode-only recv-sum snapshot
        is insufficient for this path.
        """
        if self._reported_scheduler_group_layout:
            return
        if self._kernel is None or self._max_active_clusters is None:
            raise RuntimeError("scheduler group reporting requires a built kernel")

        total_counts = [
            int(value) for value in actual_scheduled.reshape(-1).tolist()
        ]
        intra_counts = [
            int(value) for value in actual_intra.reshape(-1).tolist()
        ]
        relay_counts = [
            total - intra
            for total, intra in zip(total_counts, intra_counts)
        ]
        published_relay_counts = [
            int(value)
            for value in actual_recv_per_source.sum(axis=0).reshape(-1).tolist()
        ]
        if relay_counts != published_relay_counts:
            raise AssertionError(
                "scheduler relay-count view disagrees with published source "
                f"counts: packed={relay_counts} published={published_relay_counts}"
            )

        offsets = np.cumsum(total_counts, dtype=np.int64).tolist()
        cluster_tile_tokens = int(self._kernel.cluster_tile_tokens)
        cluster_tile_n_post_swap = int(
            self.impl.mma_tiler_mnk[0] * self.impl.cluster_shape_mnk[0]
        )
        num_fc1_n_blocks = ceil_div(
            self.problem.intermediate, cluster_tile_n_post_swap
        )
        num_fc2_n_blocks = ceil_div(
            self.problem.hidden, cluster_tile_n_post_swap
        )
        resolved_hint = int(self._kernel.group_hint)
        simulator = FusedFc12Simulator(
            offsets=offsets,
            cta_tile_token=cluster_tile_tokens,
            num_fc1_n_blocks=num_fc1_n_blocks,
            num_fc2_n_blocks=num_fc2_n_blocks,
            group_hint=resolved_hint,
            intra_token_sizes=intra_counts,
            segment_major=self._kernel.sched_segment_major,
            segment_fc1_first=self._kernel.sched_segment_fc1_first,
            merge_segments=self._kernel.merge_token_segments,
        )
        report = simulator.layout_summary()
        report.update(
            {
                "combine_wide_ring_rounds": int(
                    self._kernel.token_comm.combine_wide_ring_rounds
                ),
                "diag_fanout_skip": int(self._kernel.token_comm.diag_fanout_skip),
                "fc2_done_gpu_scope": int(self._kernel.token_comm.fc2_done_gpu_scope),
                "diag_stage1_flush": int(self._kernel.token_comm.diag_stage1_flush),
                "early_stage1": int(self._kernel.token_comm.gin_early_stage1),
                "direct_l1_sf_staging": int(self._kernel.token_comm.direct_l1_sf_staging),
                "merge_token_segments": int(self._kernel.merge_token_segments),
                "tail_topk_reduce": int(self._kernel.tail_topk_reduce),
                "cluster_tile_n_post_swap": cluster_tile_n_post_swap,
                "cluster_tile_tokens": cluster_tile_tokens,
                "intra_counts": intra_counts,
                "launch_active_clusters": int(self._max_active_clusters),
                "num_fc1_n_blocks": num_fc1_n_blocks,
                "num_fc2_n_blocks": num_fc2_n_blocks,
                "rank": self.rank,
                "relay_counts": relay_counts,
                "requested_hint": (
                    "one" if self.impl.group_hint is None
                    else "auto" if self.impl.group_hint == 0
                    else int(self.impl.group_hint)
                ),
                "resolved_hint": resolved_hint,
                "total_counts": total_counts,
            }
        )
        print(
            "[group-hint-layout] "
            + json.dumps(report, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        self._reported_scheduler_group_layout = True

    def validate_internode_stage1_metadata(self) -> None:
        """Check exact Stage 1 GIN splits, LSA redistribution, and layout."""
        if not self.impl.enable_nccl_gin:
            return
        if (
            self._global_topk_idx is None
            or self._gin_internode_recv_splits is None
            or self.local_workspace is None
            or self._kernel is None
        ):
            raise RuntimeError(
                "Stage 1 metadata validation requires generated routing and "
                "completed GIN/local workspaces"
            )

        ranks_per_node = int(self.impl.ranks_per_node or self.world_size)
        topk = self._global_topk_idx.detach().cpu().numpy()
        all_prep = [
            internode_dispatch_prep_oracle(
                topk[src_rank],
                rank=src_rank,
                world_size=self.world_size,
                ranks_per_node=ranks_per_node,
                num_experts_per_rank=self.problem.num_experts_per_rank,
            )
            for src_rank in range(self.world_size)
        ]
        all_send = np.stack(
            [prep.internode_send_count for prep in all_prep], axis=0,
        )
        all_send_token_count = np.stack(
            [prep.internode_send_token_count for prep in all_prep], axis=0,
        )

        intra = np.zeros(
            (self.problem.num_experts_per_rank,), dtype=np.int32,
        )
        expected_recv_table = np.zeros(
            (self.world_size, self.problem.num_experts_per_rank),
            dtype=np.int64,
        )
        my_node = self.rank // ranks_per_node
        node_rank_begin = my_node * ranks_per_node
        node_rank_end = node_rank_begin + ranks_per_node
        for src_rank in range(node_rank_begin, node_rank_end):
            for expert_id_raw in topk[src_rank].reshape(-1):
                expert_id = int(expert_id_raw)
                if expert_id < 0:
                    continue
                dst_rank = (
                    expert_id // self.problem.num_experts_per_rank
                )
                if dst_rank == self.rank:
                    local_expert = (
                        expert_id % self.problem.num_experts_per_rank
                    )
                    intra[local_expert] += np.int32(1)
                    expected_recv_table[src_rank, local_expert] += np.int64(1)

        expected = internode_stage1_layout_oracle(
            all_send,
            all_send_token_count,
            rank=self.rank,
            world_size=self.world_size,
            ranks_per_node=ranks_per_node,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            intra_token_count=intra,
            token_padding_block=self._kernel.cluster_tile_tokens,
            publish_relay_counts_to_gemm=True,
            merge_segments=self._kernel.merge_token_segments,
        )

        def local_i32_region(name: str) -> np.ndarray:
            spec = self._kernel._local_region_by_name[name]
            if spec.cute_dtype is not cutlass.Int32:
                raise AssertionError(
                    f"{name} must be Int32, got {spec.cute_dtype}"
                )
            offset = self._kernel._local_offsets[name]
            numel = int(np.prod(spec.shape, dtype=np.int64))
            tensor = self.local_workspace.narrow(
                0, offset, numel * 4,
            ).view(torch.int32).reshape(spec.shape)
            return tensor.detach().cpu().numpy()

        def local_i64_region(name: str) -> np.ndarray:
            spec = self._kernel._local_region_by_name[name]
            if spec.cute_dtype is not cutlass.Int64:
                raise AssertionError(
                    f"{name} must be Int64, got {spec.cute_dtype}"
                )
            offset = self._kernel._local_offsets[name]
            numel = int(np.prod(spec.shape, dtype=np.int64))
            tensor = self.local_workspace.narrow(
                0, offset, numel * 8,
            ).view(torch.int64).reshape(spec.shape)
            return tensor.detach().cpu().numpy()

        actual_recv_splits = (
            self._gin_internode_recv_splits.detach().cpu().numpy()
        )
        actual_recv_per_source = local_i32_region(
            "internode_recv_count_per_source_snapshot"
        )
        actual_recv_table = local_i64_region(
            "expert_recv_count_snapshot"
        )
        actual_recv_sum = local_i64_region(
            "expert_recv_count_sum_snapshot"
        )
        actual_intra = local_i32_region("intra_token_count_snapshot")
        actual_expert_pool_base = local_i32_region(
            "expert_pool_base_snapshot"
        )
        actual_expert_sf_pool_base = local_i32_region(
            "expert_sf_pool_base_snapshot"
        )
        actual_relay_pool_base = local_i32_region(
            "relay_pool_base_snapshot"
        )
        actual_relay_sf_pool_base = local_i32_region(
            "relay_sf_pool_base_snapshot"
        )
        actual_scheduled = local_i32_region(
            "expert_scheduled_count_snapshot"
        )
        actual_pool_end = local_i64_region("pool_end_snapshot").reshape(-1)
        actual_arrival = local_i32_region(
            "l1_arrival_count_snapshot"
        ).reshape(-1)
        expected_arrival = np.zeros_like(actual_arrival)
        if self._max_active_clusters is None:
            raise RuntimeError("packed recv validation requires SM count")
        expected_recv_sum = np.asarray(
            [
                packed_recv_sum(
                    expected_recv_table[
                        node_rank_begin:node_rank_end, local_e
                    ],
                    sm_count=self._max_active_clusters,
                )
                for local_e in range(self.problem.num_experts_per_rank)
            ],
            dtype=np.int64,
        )
        expected_pool = build_expert_pool_layout(
            intra.tolist(),
            expected.internode_recv_count_per_source.T.tolist(),
            token_padding_block=self._kernel.cluster_tile_tokens,
            sf_padding_block=self._kernel.sf_padding_block,
            merge_segments=self._kernel.merge_token_segments,
        )
        expected_pool_end = np.asarray(
            [expected_pool.data_end, expected_pool.sf_end], dtype=np.int64,
        )
        tile = int(self._kernel.cluster_tile_tokens)
        for _, _, _, _, _, counter_slot in pool_row_slots(
            expected_pool, intra.tolist(),
            expected.internode_recv_count_per_source.T.tolist(), tile=tile,
        ):
            expected_arrival[counter_slot] += np.int32(1)

        checks = (
            (
                "expert_recv_count",
                actual_recv_table,
                expected_recv_table,
            ),
            (
                "expert_recv_count_sum",
                actual_recv_sum,
                expected_recv_sum,
            ),
            (
                "intra_token_count",
                actual_intra,
                intra,
            ),
            (
                "internode_recv_splits",
                actual_recv_splits,
                expected.internode_recv_splits,
            ),
            (
                "internode_recv_count_per_source",
                actual_recv_per_source,
                expected.internode_recv_count_per_source,
            ),
            (
                "expert_pool_base",
                actual_expert_pool_base,
                np.asarray(expected_pool.expert_pool_base, dtype=np.int32),
            ),
            (
                "expert_sf_pool_base",
                actual_expert_sf_pool_base,
                np.asarray(expected_pool.expert_sf_pool_base, dtype=np.int32),
            ),
            (
                "relay_pool_base",
                actual_relay_pool_base,
                expected.relay_pool_base,
            ),
            (
                "relay_sf_pool_base",
                actual_relay_sf_pool_base,
                np.asarray(expected_pool.relay_sf_pool_base, dtype=np.int32),
            ),
            (
                "expert_scheduled_count",
                actual_scheduled,
                expected.expert_scheduled_count,
            ),
            (
                "pool_end",
                actual_pool_end,
                expected_pool_end,
            ),
            (
                "l1_arrival_count",
                actual_arrival,
                expected_arrival,
            ),
        )
        for name, actual, wanted in checks:
            if not np.array_equal(actual, wanted):
                mismatch = np.argwhere(actual != wanted)
                first = tuple(int(v) for v in mismatch[0])
                raise AssertionError(
                    f"[rank {self.rank}] Stage 1 {name} mismatch at {first}: "
                    f"actual={int(actual[first])} expected={int(wanted[first])}; "
                    f"actual_sum={int(actual.sum())} "
                    f"expected_sum={int(wanted.sum())}"
                )

        if expected_pool.data_end > self._kernel.pool_token_capacity:
            raise AssertionError(
                "oracle DATA pool end exceeds allocated capacity: "
                f"end={expected_pool.data_end} "
                f"capacity={self._kernel.pool_token_capacity}"
            )
        if expected_pool.sf_end > self._kernel.pool_sf_capacity:
            raise AssertionError(
                "oracle SF pool end exceeds allocated capacity: "
                f"end={expected_pool.sf_end} "
                f"capacity={self._kernel.pool_sf_capacity}"
            )

        node_experts = ranks_per_node * self.problem.num_experts_per_rank
        actual_expert_hit_total = int(
            actual_recv_splits[:, :node_experts].sum()
        )
        expected_expert_hit_total = int(
            expected.internode_recv_splits[:, :node_experts].sum()
        )
        actual_message_total = int(
            actual_recv_splits[:, node_experts].sum()
        )
        expected_message_total = int(
            expected.internode_recv_splits[:, node_experts].sum()
        )
        if actual_expert_hit_total != expected_expert_hit_total:
            raise AssertionError("received expert-hit split total mismatch")
        if actual_message_total != expected_message_total:
            raise AssertionError("received message-count total mismatch")
        if int(actual_recv_per_source.sum()) != int(
            expected.internode_recv_count_per_source.sum()
        ):
            raise AssertionError("redistributed relay total mismatch")

        self._report_actual_scheduler_group_layout(
            actual_scheduled,
            actual_intra,
            actual_recv_per_source,
        )

        # Do not put a distributed barrier on the success path: if one rank's
        # exact check fails, its peers must also be free to unwind instead of
        # hanging here and hiding the first mismatch behind the job timeout.
        print(
            f"[rank {self.rank}] Stage 1 metadata validation PASSED: "
            f"expert_hit_total={actual_expert_hit_total} "
            f"message_total={actual_message_total} "
            f"redistributed_total={int(actual_recv_per_source.sum())} "
            f"scheduled_total={int(actual_scheduled.sum())}",
            flush=True,
        )

    def validate_internode_stage1_payload(self) -> None:
        """Validate every received relay message byte-for-byte."""
        if not self.impl.enable_nccl_gin:
            return
        if (
            self._gin_window_layout is None
            or self._gin_window_layout.relay_message is None
            or self._gin_relay_recv_buffer is None
            or self._gin_internode_recv_splits is None
            or self._global_activation is None
            or self._global_activation_sf is None
            or self._global_topk_idx is None
            or self._global_topk_weights is None
        ):
            raise RuntimeError(
                "Stage 1 payload validation requires the registered relay "
                "buffer and generated global inputs"
            )

        ranks_per_node = int(self.impl.ranks_per_node or self.world_size)
        rank_in_node = self.rank % ranks_per_node
        my_node = self.rank // ranks_per_node
        num_experts_per_rank = self.problem.num_experts_per_rank
        node_experts = ranks_per_node * num_experts_per_rank
        layout = self._gin_window_layout.relay_message
        message_bytes = layout.message_bytes
        sf_wire_bytes = self._kernel.sf_uint32_per_token * 4

        topk = self._global_topk_idx.detach().cpu().numpy()
        activation = (
            self._global_activation.view(torch.uint8).detach().cpu().numpy()
        )
        activation_sf = (
            self._global_activation_sf.view(torch.uint8)
            .detach().cpu().numpy()
        )
        weights = self._global_topk_weights.detach().cpu().numpy()
        actual_splits = (
            self._gin_internode_recv_splits.detach().cpu().numpy()
        )
        actual_messages = self._gin_relay_recv_buffer.detach().cpu().numpy()

        validated_messages = 0
        for src_slot in range(actual_messages.shape[0]):
            src_node = src_slot if src_slot < my_node else src_slot + 1
            src_rank = src_node * ranks_per_node + rank_in_node
            send_slot = my_node if my_node < src_node else my_node - 1
            prep = internode_dispatch_prep_oracle(
                topk[src_rank],
                rank=src_rank,
                world_size=self.world_size,
                ranks_per_node=ranks_per_node,
                num_experts_per_rank=num_experts_per_rank,
            )
            expected_count = int(prep.tokens_to_node_count[send_slot])
            metadata_count = int(actual_splits[src_slot, node_experts])
            if metadata_count != expected_count:
                raise AssertionError(
                    f"[rank {self.rank}] relay message count from source "
                    f"node {src_node}: metadata={metadata_count} "
                    f"expected={expected_count}"
                )

            actual_by_token = {}
            for slot in range(metadata_count):
                msg = actual_messages[src_slot, slot]
                packed = int.from_bytes(msg[:8].tobytes(), "little")
                src_token = packed & 0xFFFFFFFF
                header_hi = packed >> 32
                header_rank = (header_hi >> 16) & 0xFFFF
                header_topk = header_hi & 0xFFFF
                if header_rank != src_rank or header_topk != 0xFFFF:
                    raise AssertionError(
                        f"[rank {self.rank}] invalid relay header at "
                        f"source_slot={src_slot} slot={slot}: "
                        f"src_rank={header_rank} src_topk={header_topk:#x}"
                    )
                if src_token in actual_by_token:
                    raise AssertionError(
                        f"[rank {self.rank}] duplicate relay token {src_token} "
                        f"from source rank {src_rank}"
                    )
                actual_by_token[src_token] = msg

            expected_tokens = [
                int(v)
                for v in prep.tokens_to_node[send_slot, :expected_count]
            ]
            if set(actual_by_token) != set(expected_tokens):
                raise AssertionError(
                    f"[rank {self.rank}] relay token set mismatch from rank "
                    f"{src_rank}: actual={sorted(actual_by_token)} "
                    f"expected={sorted(expected_tokens)}"
                )

            for token_idx in expected_tokens:
                expected = np.zeros((message_bytes,), dtype=np.uint8)
                header_hi = (src_rank << 16) | 0xFFFF
                packed = (header_hi << 32) | (token_idx & 0xFFFFFFFF)
                expected[
                    layout.src_metadata_offset:
                    layout.src_metadata_offset + 8
                ] = np.frombuffer(packed.to_bytes(8, "little"), dtype=np.uint8)

                for topk_slot in range(self.problem.num_topk):
                    expert_id = int(topk[src_rank, token_idx, topk_slot])
                    route_word = 0xFFFF
                    weight_bytes = bytes(4)
                    if expert_id >= 0:
                        dst_rank = expert_id // num_experts_per_rank
                        if dst_rank // ranks_per_node == my_node:
                            node_expert = (
                                (dst_rank % ranks_per_node)
                                * num_experts_per_rank
                                + expert_id % num_experts_per_rank
                            )
                            route_word = node_expert | (topk_slot << 16)
                            weight_bytes = np.float32(
                                weights[src_rank, token_idx, topk_slot]
                            ).tobytes()
                    route_off = layout.route_words_offset + topk_slot * 4
                    weight_off = layout.topk_weights_offset + topk_slot * 4
                    expected[route_off:route_off + 4] = np.frombuffer(
                        np.uint32(route_word).tobytes(), dtype=np.uint8,
                    )
                    expected[weight_off:weight_off + 4] = np.frombuffer(
                        weight_bytes, dtype=np.uint8,
                    )

                token_payload = activation[src_rank, token_idx].reshape(-1)
                expected[
                    layout.token_data_offset:
                    layout.token_data_offset + token_payload.size
                ] = token_payload
                sf_payload = activation_sf[src_rank, token_idx].reshape(-1)
                expected[
                    layout.sf_data_offset:
                    layout.sf_data_offset + sf_payload.size
                ] = sf_payload
                # Any uint32 padding after the valid SF bytes, plus the message
                # tail up to the 128B stride, remains zero by construction.
                if sf_payload.size > sf_wire_bytes:
                    raise AssertionError(
                        f"SF payload {sf_payload.size} exceeds wire plane "
                        f"{sf_wire_bytes}"
                    )

                actual = actual_by_token[token_idx]
                if not np.array_equal(actual, expected):
                    mismatch = np.flatnonzero(actual != expected)
                    byte_off = int(mismatch[0])
                    raise AssertionError(
                        f"[rank {self.rank}] relay payload mismatch from "
                        f"src_rank={src_rank} token={token_idx} byte={byte_off}: "
                        f"actual={int(actual[byte_off])} "
                        f"expected={int(expected[byte_off])}"
                    )
                validated_messages += 1

        print(
            f"[rank {self.rank}] Stage 1 payload validation PASSED: "
            f"messages={validated_messages} message_bytes={message_bytes}",
            flush=True,
        )


    def validate_internode_dispatch_l1(self) -> None:
        """Validate the Stage 2/3 relay result in the physical L1 pools.

        Fan-out reservations are concurrent, so arrival order is deliberately
        unspecified. Match each physical relay slot by its materialized
        (src_rank, src_token, src_topk) metadata, then compare token bytes,
        SF words, and the routing weight exactly against the originating rank.
        """
        if not self.impl.enable_nccl_gin:
            return
        if (
            self.local_workspace is None
            or self._kernel is None
            or self._global_activation is None
            or self._global_activation_sf is None
            or self._global_topk_idx is None
            or self._global_topk_weights is None
        ):
            raise RuntimeError(
                "Stage 2/3 L1 validation requires completed local workspaces "
                "and generated global inputs"
            )

        def local_region(name: str, dtype: torch.dtype) -> np.ndarray:
            shared = name in self._kernel._shared_region_by_name
            specs = self._kernel._shared_region_by_name if shared else self._kernel._local_region_by_name
            offsets = self._kernel._shared_offsets if shared else self._kernel._local_offsets
            workspace = self.shared_workspace if shared else self.local_workspace
            spec = specs[name]
            offset = offsets[name]
            numel = int(np.prod(spec.shape, dtype=np.int64))
            element_bytes = torch.empty((), dtype=dtype).element_size()
            tensor = workspace.narrow(
                0, offset, numel * element_bytes,
            ).view(dtype).reshape(spec.shape)
            return tensor.detach().cpu().numpy()

        ranks_per_node = int(self.impl.ranks_per_node or self.world_size)
        my_node = self.rank // ranks_per_node
        num_nodes = self.world_size // ranks_per_node
        num_experts_per_rank = self.problem.num_experts_per_rank
        node_experts = ranks_per_node * num_experts_per_rank
        sf_words_per_token = self._kernel.sf_uint32_per_token

        topk = self._global_topk_idx.detach().cpu().numpy()
        activation = (
            self._global_activation.view(torch.uint8).detach().cpu().numpy()
        )
        activation_sf = (
            self._global_activation_sf.view(torch.uint8)
            .detach().cpu().numpy()
        )
        weights = self._global_topk_weights.detach().cpu().numpy()

        l1_token = local_region("l1_token_buffer", torch.uint8)
        l1_sf = local_region("l1_sf_buffer", torch.int32).reshape(-1)
        l1_weights = local_region(
            "l1_topk_weights_buffer", torch.float32,
        )
        l1_metadata = local_region("token_src_metadata", torch.uint8)
        relay_pool_base = local_region(
            "relay_pool_base_snapshot", torch.int32,
        )
        recv_per_source = local_region(
            "internode_recv_count_per_source_snapshot", torch.int32,
        )

        # Reconstruct DATA/SF bases from routing counts. DATA uses cluster
        # token tiles; SF uses its own atom padding, including merged experts.
        intra_count = np.zeros((num_experts_per_rank,), dtype=np.int32)
        local_rank_begin = my_node * ranks_per_node
        local_rank_end = local_rank_begin + ranks_per_node
        for src_rank in range(local_rank_begin, local_rank_end):
            for expert_id_raw in topk[src_rank].reshape(-1):
                expert_id = int(expert_id_raw)
                if (
                    self.rank * num_experts_per_rank
                    <= expert_id
                    < (self.rank + 1) * num_experts_per_rank
                ):
                    intra_count[
                        expert_id - self.rank * num_experts_per_rank
                    ] += np.int32(1)
        expected_pool = build_expert_pool_layout(
            intra_count.tolist(), recv_per_source.T.tolist(),
            token_padding_block=self._kernel.cluster_tile_tokens,
            sf_padding_block=self._kernel.sf_padding_block,
            merge_segments=self._kernel.merge_token_segments,
        )
        np.testing.assert_array_equal(relay_pool_base, expected_pool.relay_pool_base)
        relay_sf_pool_base = np.asarray(expected_pool.relay_sf_pool_base, dtype=np.int32)

        validated_hits = 0
        for src_node in range(num_nodes):
            if src_node == my_node:
                continue
            src_slot = src_node if src_node < my_node else src_node - 1
            for local_expert in range(num_experts_per_rank):
                global_expert = (
                    self.rank * num_experts_per_rank + local_expert
                )
                expected_keys = []
                for src_rank_in_node in range(ranks_per_node):
                    src_rank = src_node * ranks_per_node + src_rank_in_node
                    for src_token in range(self.problem.num_tokens_per_rank):
                        for src_topk in range(self.problem.num_topk):
                            if (
                                int(topk[src_rank, src_token, src_topk])
                                == global_expert
                            ):
                                expected_keys.append(
                                    (src_rank, src_token, src_topk)
                                )

                expected_count = len(expected_keys)
                published_count = int(
                    recv_per_source[src_slot, local_expert]
                )
                if published_count != expected_count:
                    raise AssertionError(
                        f"[rank {self.rank}] Stage 2/3 relay count mismatch "
                        f"for src_node={src_node} local_expert={local_expert}: "
                        f"published={published_count} expected={expected_count}"
                    )

                pool_base = int(relay_pool_base[local_expert, src_slot])
                actual_keys = []
                for source_offset in range(expected_count):
                    pool_idx = pool_base + source_offset
                    md = l1_metadata[pool_idx]
                    packed = int.from_bytes(md[:8].tobytes(), "little")
                    src_token = packed & 0xFFFFFFFF
                    header_hi = packed >> 32
                    src_rank = (header_hi >> 16) & 0xFFFF
                    src_topk = header_hi & 0xFFFF
                    key = (src_rank, src_token, src_topk)
                    actual_keys.append(key)

                    if not (
                        src_node * ranks_per_node
                        <= src_rank
                        < (src_node + 1) * ranks_per_node
                        and src_token < self.problem.num_tokens_per_rank
                        and src_topk < self.problem.num_topk
                    ):
                        raise AssertionError(
                            f"[rank {self.rank}] invalid Stage 3 metadata at "
                            f"pool={pool_idx}: key={key} src_node={src_node}"
                        )
                    if (
                        int(topk[src_rank, src_token, src_topk])
                        != global_expert
                    ):
                        raise AssertionError(
                            f"[rank {self.rank}] Stage 3 route mismatch at "
                            f"pool={pool_idx}: key={key} routes to "
                            f"expert={int(topk[src_rank, src_token, src_topk])}, "
                            f"expected={global_expert}"
                        )

                    expected_token = activation[
                        src_rank, src_token
                    ].reshape(-1)
                    actual_token = l1_token[pool_idx].reshape(-1)
                    if not np.array_equal(actual_token, expected_token):
                        byte_off = int(
                            np.flatnonzero(actual_token != expected_token)[0]
                        )
                        raise AssertionError(
                            f"[rank {self.rank}] Stage 3 token mismatch at "
                            f"pool={pool_idx} key={key} byte={byte_off}: "
                            f"actual={int(actual_token[byte_off])} "
                            f"expected={int(expected_token[byte_off])}"
                        )

                    expected_weight_bits = np.asarray(
                        weights[src_rank, src_token, src_topk],
                        dtype=np.float32,
                    ).view(np.uint32).item()
                    actual_weight_bits = np.asarray(
                        l1_weights[pool_idx], dtype=np.float32,
                    ).view(np.uint32).item()
                    if actual_weight_bits != expected_weight_bits:
                        raise AssertionError(
                            f"[rank {self.rank}] Stage 3 weight mismatch at "
                            f"pool={pool_idx} key={key}: "
                            f"actual_bits={actual_weight_bits:#010x} "
                            f"expected_bits={expected_weight_bits:#010x}"
                        )

                    expected_sf_words = np.frombuffer(
                        activation_sf[src_rank, src_token]
                        .reshape(-1).tobytes(),
                        dtype=np.int32,
                        count=sf_words_per_token,
                    )
                    sf_token_axis = (
                        int(relay_sf_pool_base[local_expert, src_slot])
                        + source_offset
                    )
                    for sf_word in range(sf_words_per_token):
                        sf_pos = int(
                            sf_atom_int32_offset_numpy(
                                sf_token_axis,
                                sf_word,
                                num_k_atoms=sf_words_per_token,
                            )
                        )
                        if int(l1_sf[sf_pos]) != int(
                            expected_sf_words[sf_word]
                        ):
                            raise AssertionError(
                                f"[rank {self.rank}] Stage 3 SF mismatch at "
                                f"pool={pool_idx} key={key} word={sf_word}: "
                                f"actual={int(l1_sf[sf_pos]) & 0xFFFFFFFF:#010x} "
                                f"expected={int(expected_sf_words[sf_word]) & 0xFFFFFFFF:#010x}"
                            )
                    validated_hits += 1

                if sorted(actual_keys) != sorted(expected_keys):
                    raise AssertionError(
                        f"[rank {self.rank}] Stage 3 provenance set mismatch "
                        f"for src_node={src_node} local_expert={local_expert}: "
                        f"actual={sorted(actual_keys)} "
                        f"expected={sorted(expected_keys)}"
                    )

        expected_total = int(recv_per_source.sum())
        if validated_hits != expected_total:
            raise AssertionError(
                f"[rank {self.rank}] Stage 3 validated-hit total mismatch: "
                f"actual={validated_hits} expected={expected_total}"
            )
        print(
            f"[rank {self.rank}] Stage 2/3 L1 validation PASSED: "
            f"relay_hits={validated_hits} node_experts={node_experts}",
            flush=True,
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

        if self.rank == 0:
            print(
                "MegaMoETester: perf_launch_mode="
                + (
                    "synchronized"
                    if self._perf_sync_each_launch
                    else "back-to-back"
                )
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

        if self.impl.enable_nccl_gin:
            self.validate_internode_stage1_metadata()
            self.validate_internode_stage1_payload()
            diag_fanout_skip = int(self._kernel.token_comm.diag_fanout_skip)
            if diag_fanout_skip != 0:
                # DIAGNOSTIC ONLY: the fan-out skipped the SF scatter and/or
                # the DATA copy, so the L1 pool is intentionally wrong.
                print(
                    f"[rank {self.rank}] Stage 2/3 L1 validation SKIPPED: "
                    f"MEGA_DIAG_FANOUT_SKIP={diag_fanout_skip} (diagnostic, "
                    "outputs invalid)",
                    flush=True,
                )
            else:
                self.validate_internode_dispatch_l1()

        if not skip_ref:
            self.validate()

        if self.rank == 0:
            print("DONE")


# =============================================================================
# CLI
# =============================================================================


def _parse_group_hint(text: str):
    """CLI group hint: ``one`` -> None, ``auto`` -> 0, else a positive int."""
    value = str(text).strip().lower()
    if value == "one":
        return None
    if value == "auto":
        return 0
    hint = int(value)
    if hint <= 0:
        raise ValueError(f"--group_hint must be one, auto, or positive, got {text!r}")
    return hint


def _parse_tuple(argument: str) -> Tuple[int, ...]:
    return tuple(int(value) for value in argument.split(","))


def _parse_output_dtype(argument: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
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
        raise argparse.ArgumentTypeError(str(exc))


def _resolve_nccl_gin_resource_counts(
    impl: ImplDesc,
    *,
    world_size: int,
    num_experts_per_rank: int,
    max_relay_tokens_per_node: int,
    hidden_bytes: int,
    sf_uint32_per_token: int,
    num_topk: int,
    combine_hidden: int,
) -> Tuple[int, int]:
    """Resolve the exact Stage 1 signal count and optional library barriers."""
    ranks_per_node = int(impl.ranks_per_node or world_size)
    layout = build_megamoe_gin_window_layout(
        world_size=world_size,
        ranks_per_node=ranks_per_node,
        num_experts_per_rank=num_experts_per_rank,
        max_relay_tokens_per_node=max_relay_tokens_per_node,
        hidden_bytes=hidden_bytes,
        sf_uint32_per_token=sf_uint32_per_token,
        num_topk=num_topk,
        combine_hidden=combine_hidden,
    )
    # Allocate the complete planned window signal space now. Stage 1 consumes
    # the split and relay-node slots; reserving the combine slots prevents a
    # later window/resource ABI change when the combine path is enabled.
    required_signals = layout.total_signal_count
    if impl.gin_signal_count is None:
        signal_count = required_signals
    else:
        signal_count = impl.gin_signal_count
        if signal_count < required_signals:
            raise ValueError(
                f"gin_signal_count={signal_count} is smaller than the "
                f"Stage 1 requirement {required_signals}."
            )

    # MegaMoE uses ordinary indexed signals for its replay-safe custom rail
    # barrier. The runtime-smoke CLI can still request NCCL library barriers,
    # but production should not allocate them by default.
    barrier_count = (
        impl.gin_barrier_count
        if impl.gin_barrier_count is not None
        else 0
    )
    return signal_count, barrier_count


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
        "--fc2_output_dtype", type=_parse_output_dtype, default=torch.bfloat16,
    )
    parser.add_argument(
        "--combine_format", type=_parse_combine_format,
        default=CombineFormat.parse("bf16"),
        help="Cross-rank combine wire format: 'bf16' (no-quant baseline), "
             "'16e2m1xbf16' (fp4 + per-16 bf16 amax), or '32e4m3xe8m0' (MXFP8).",
    )
    parser.add_argument(
        "--route_distribution", type=str, default="balanced",
        choices=["balanced", "power_law"],
    )
    parser.add_argument(
        "--power_law_exponent", type=float, default=1.0,
        help="Zipf exponent for --route_distribution power_law; 1.0 = classic "
             "Zipf, larger = more skewed, 0 = uniform.",
    )
    parser.add_argument(
        "--gate_up_clamp", type=float, default=None,
        help="DeepSeek-V4 swiglu_limit: asymmetric clamp on the real gate/up "
             "pre-activations (gate<=limit, |up|<=limit) before SiLU. "
             "Omitted/None disables the clamp; must be non-negative.",
    )

    parser.add_argument("--mma_tiler_mnk", type=str, default="128,128,256")
    parser.add_argument("--cluster_shape_mnk", type=str, default="1,1,1")
    parser.add_argument("--use_2cta_instrs", action="store_true", default=False)
    parser.add_argument("--enable_static_expert_shape", action="store_true", default=False)
    parser.add_argument("--dynamic_sched", action="store_true", default=False)
    parser.add_argument("--clc_bundle_size", type=int, default=None)
    parser.add_argument("--num_sched_stages", type=int, default=None)
    parser.add_argument(
        "--load_balance_mode", type=str, default="static",
        choices=["static", "atomic_counter"],
    )
    parser.add_argument(
        "--group_hint", type=str, default=None,
        help="one (single scheduler group), auto (max active clusters) or a "
             "positive FC1 tile threshold per group. Default: one with "
             "--internode (validated GIN default, 2026-09-05), auto for the "
             "NVLink-domain kernel (main's value)",
    )
    parser.add_argument(
        "--task_reg_cnt", type=int, default=72,
        help="Registers per thread of the sched/TMA/MMA task warpgroup, a "
             "multiple of 8 in [24, 112] (MegaMoEConfig.task_reg_cnt).",
    )
    parser.add_argument(
        "--tma_cache_hint", type=int, default=0, choices=(0, 1, 2, 3),
        help="TMA L2 cache-hint mask on the mainloop operand loads: 1 = "
             "weights evict-first, 2 = token / FC1-output B operand "
             "evict-last (MegaMoEConfig.tma_cache_hint).",
    )
    parser.add_argument(
        "--tail_topk_reduce", action=argparse.BooleanOptionalAction,
        default=False,
        help="Ordered FP32 top-k reduce inside the GIN kernel tail instead of "
             "the separate TopkReduce launch (MegaMoEConfig.tail_topk_reduce; "
             "requires --enable_nccl_gin).",
    )
    parser.add_argument(
        "--early_stage1", action=argparse.BooleanOptionalAction, default=False,
        help="Opt-in dispatch candidate (MegaMoEExperimental.early_stage1): "
             "relay warps skip dispatch_barrier, grid-flat Stage 1 pack, Step 3 "
             "metadata posted by the Stage 1 issuer lane. Requires --enable_nccl_gin.",
    )
    parser.add_argument(
        "--direct_l1_sf_staging", action=argparse.BooleanOptionalAction, default=False,
        help="Opt-in dispatch candidate (MegaMoEExperimental.direct_l1_sf_staging): "
             "staged scale-factor rows scattered by the owner's dispatch warps. "
             "Requires --enable_nccl_gin.",
    )
    parser.add_argument("--flag_batch", type=int, default=4)
    parser.add_argument(
        "--epi_flag_batch", type=str, default=None,
        help="(fc1,fc2) done-counter publish batch in comma form. Default: 1,4 "
             "with --internode (the FC2 batch of 4 amortizes the system-scope "
             "release that otherwise bounds consecutive FC2 tiles, GEMM-stage "
             "attribution 2026-09-08), 1,1 for the NVLink-domain kernel (main)",
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
    parser.add_argument(
        "--internode",
        action="store_true",
        default=False,
        help="Master switch of the internode (multi-node) code paths: implies "
             "--enable_internode_relay and --enable_nccl_gin (either of those "
             "also turns it on). Without it the kernel is main's NVLink-domain "
             "megakernel: every internode path compiled out, one logical node, "
             "--ranks_per_node must equal the world size.",
    )
    parser.add_argument(
        "--enable_internode_relay",
        action="store_true",
        default=False,
        help="Enable the internode relay path: reserve warps 12-15 for relay "
             "work and run the 16-warp synchronization layout.",
    )
    parser.add_argument(
        "--ranks_per_node",
        type=int,
        default=None,
        help="Ranks per physical node for internode routing metadata. Defaults "
             "to LOCAL_WORLD_SIZE under torchrun, or world_size when unset.",
    )
    parser.add_argument(
        "--enable_nccl_gin",
        action="store_true",
        default=False,
        help="Enable the NCCL GIN Stage 1 metadata exchange and custom "
             "cross-node rail barrier on the registered LSA arena.",
    )
    parser.add_argument(
        "--gin_connection_type",
        type=str,
        default="rail",
        choices=["rail", "full"],
        help="NCCL GIN connection topology for the device communicator.",
    )
    parser.add_argument("--gin_signal_count", type=int, default=None)
    parser.add_argument("--gin_barrier_count", type=int, default=None)
    parser.add_argument("--gin_context_count", type=int, default=None)
    parser.add_argument(
        "--gin_strong_signals_required",
        action="store_true",
        default=False,
        help="Request NCCL GIN strong-signal resources from the dev communicator.",
    )
    parser.add_argument(
        "--gin_va_signals_required",
        action="store_true",
        default=False,
        help="Request NCCL GIN VA-signal resources from the dev communicator.",
    )
    parser.add_argument(
        "--gin_force_enable",
        action="store_true",
        default=False,
        help="Pass gin_force_enable=True to NCCL devComm requirements.",
    )
    parser.add_argument(
        "--gin_window_bytes",
        type=int,
        default=4096,
        help="Minimum bytes reserved at registered-window offset 0 for "
             "GIN transport state; the Stage 1 layout can enlarge it.",
    )
    parser.add_argument("--perf_run", action="store_true", default=False)
    parser.add_argument("--skip_ref_check", action="store_true", default=False)
    parser.add_argument("--profile_friendly", action="store_true", default=False)
    parser.add_argument("--use_torch_profiler", action="store_true", default=False)
    parser.add_argument(
        "--perf_sync_each_launch",
        action="store_true",
        default=False,
        help="Synchronize and rank-barrier after every profiler warmup/timed "
             "launch. This is a repeated-launch diagnostic/workaround, not "
             "a back-to-back throughput measurement.",
    )
    parser.add_argument(
        "--validate_each_launch",
        action="store_true",
        default=False,
        help="After every synchronized repeated launch, validate both zero "
             "prefixes, the persistent NVLink phase, and GIN Stage-1 metadata.",
    )
    # Default to 0,1 to avoid testing time.
    parser.add_argument("--perf_warmup", type=int, default=0)
    parser.add_argument("--perf_iters", type=int, default=1)
    parser.add_argument("--enable_debug_checks", action="store_true", default=False)
    parser.add_argument(
        "--ref_compute_graph", type=str, default="deepgemm",
        choices=["transformers", "deepgemm"],
    )
    parser.add_argument("--enable_iket", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def _init_dist_without_nvshmem() -> Tuple[int, int, int]:
    """Initialize torch distributed/CUDA without importing NVSHMEM."""
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    except KeyError as exc:
        raise RuntimeError(
            "relay mode requires torchrun RANK/LOCAL_RANK/WORLD_SIZE"
        ) from exc
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="cpu:gloo,cuda:nccl")
    return local_rank, rank, world_size


def main(argv: Optional[List[str]] = None) -> int:
    """Bootstrap the selected peer runtime, build the tester, and run once.

    Launcher::

        torchrun --nproc_per_node=4 -m flashinfer.moe_ep.kernels.cutedsl_megamoe.pre_unification.moe_nvfp4_swapab.mega_runner \\
            --num_total_experts 32 --route_distribution balanced
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # The diagnostic MEGA_* variables are parsed exactly once, here; the
    # kernel modules read only the resulting objects.
    diagnostics = MegaMoEDiagnostics.from_env()
    _set_no_dist(diagnostics.no_dist)
    atomic_batch_text = os.environ.get("MEGA_TOKEN_BACK_ATOMIC_BATCH", "1")
    if not atomic_batch_text.isdigit() or int(atomic_batch_text) < 1:
        raise ValueError(
            "MEGA_TOKEN_BACK_ATOMIC_BATCH must be a positive integer, got "
            f"{atomic_batch_text!r}"
        )
    config = MegaMoEConfig(
        task_reg_cnt=args.task_reg_cnt,
        tma_cache_hint=args.tma_cache_hint,
        tail_topk_reduce=args.tail_topk_reduce,
        token_back_atomic_batch=int(atomic_batch_text),
        experimental=MegaMoEExperimental(
            early_stage1=args.early_stage1,
            direct_l1_sf_staging=args.direct_l1_sf_staging,
        ),
        diagnostics=diagnostics,
    )

    if args.validate_each_launch and not (
        args.use_torch_profiler and args.perf_sync_each_launch
    ):
        parser.error(
            "--validate_each_launch requires --use_torch_profiler and "
            "--perf_sync_each_launch"
        )

    # Master switch: --internode implies the production relay + GIN configuration, and
    # either sub-flag turns the switch on; everything else is the NVLink-domain kernel.
    if args.internode:
        args.enable_internode_relay = True
        args.enable_nccl_gin = True
    internode = bool(args.enable_internode_relay or args.enable_nccl_gin)
    if args.epi_flag_batch is None:
        args.epi_flag_batch = "1,4" if internode else "1,1"
    if args.group_hint is None:
        args.group_hint = "one" if internode else "auto"

    if _NO_DIST:
        # MEGA_NO_DIST=1: bypass torch.distributed + NVSHMEM init so the
        # script can be launched as plain ``python mega_runner.py`` under
        # compute-sanitizer / debugger.  Single-rank only.
        torch.cuda.set_device(0)
        rank = 0
        world_size = 1
    elif args.enable_internode_relay:
        _local_rank, rank, world_size = _init_dist_without_nvshmem()
    else:
        from ..comm.bootstrap import init_dist_and_nvshmem
        _local_rank, rank, world_size, _ = init_dist_and_nvshmem()

    if not internode:
        # One logical node: the NVLink domain is the whole world, whatever torchrun's
        # LOCAL_WORLD_SIZE says (a two-host NVLink domain is still one node here).
        if args.ranks_per_node not in (None, world_size):
            raise ValueError(
                "--ranks_per_node needs --internode (or --enable_internode_relay / "
                f"--enable_nccl_gin); without it ranks_per_node is the world size {world_size}"
            )
        ranks_per_node = world_size
    elif args.ranks_per_node is None:
        ranks_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    else:
        ranks_per_node = args.ranks_per_node
    if ranks_per_node <= 0:
        raise ValueError(
            f"ranks_per_node must be positive, got {ranks_per_node}."
        )
    if world_size % ranks_per_node != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by "
            f"ranks_per_node ({ranks_per_node})."
        )

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
            f"num_topk ({problem.num_topk}) > 32 is unsupported by the "
            "current DeepGEMM-derived implementation."
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
        group_hint=_parse_group_hint(args.group_hint),
        non_ubulk_fc2_store=not args.use_bulk_fc2_store,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        token_back_mode=args.token_back_mode,
        internode=internode,
        enable_internode_relay=args.enable_internode_relay,
        ranks_per_node=ranks_per_node,
        enable_nccl_gin=args.enable_nccl_gin,
        gin_connection_type=args.gin_connection_type,
        gin_signal_count=args.gin_signal_count,
        gin_barrier_count=args.gin_barrier_count,
        gin_context_count=args.gin_context_count,
        gin_strong_signals_required=args.gin_strong_signals_required,
        gin_va_signals_required=args.gin_va_signals_required,
        gin_force_enable=args.gin_force_enable,
        gin_window_bytes=args.gin_window_bytes,
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

    tester: Optional[MegaMoETester] = None
    nccl_gin_context: Optional[NcclGinContext] = None
    nccl_lsa_context: Optional[NcclLsaContext] = None
    return_code = 0
    try:
        if impl.enable_nccl_gin:
            if not impl.enable_internode_relay:
                raise ValueError(
                    "--enable_nccl_gin requires --enable_internode_relay"
                )
            signal_count, barrier_count = _resolve_nccl_gin_resource_counts(
                impl,
                world_size=world_size,
                num_experts_per_rank=problem.num_experts_per_rank,
                max_relay_tokens_per_node=problem.num_tokens_per_rank,
                hidden_bytes=problem.hidden // 2,
                sf_uint32_per_token=(
                    round_up(
                        ceil_div(problem.hidden, Nvfp4BlockSize), 4
                    ) // 4
                ),
                num_topk=problem.num_topk,
                combine_hidden=problem.hidden,
            )
            nccl_gin_context = create_nccl_gin_context(
                rank=rank,
                world_size=world_size,
                signal_count=signal_count,
                barrier_count=barrier_count,
                connection_type=impl.gin_connection_type,
                context_count=impl.gin_context_count,
                strong_signals_required=impl.gin_strong_signals_required,
                gin_va_signals_required=impl.gin_va_signals_required,
                gin_force_enable=impl.gin_force_enable,
            )
            validate_nccl_gin_rail_topology(
                nccl_gin_context.support,
                world_size=world_size,
                ranks_per_node=ranks_per_node,
            )
            if rank == 0:
                print(
                    "NCCL GIN context ready: "
                    f"dev_comm_ptr=0x{nccl_gin_context.dev_comm_ptr:x} "
                    f"connection={nccl_gin_context.connection_type} "
                    f"gin_force_enable={nccl_gin_context.gin_force_enable} "
                    f"signals={nccl_gin_context.signal_count} "
                    f"barriers={nccl_gin_context.barrier_count} "
                    f"support={nccl_gin_context.support}"
                )

        elif impl.enable_internode_relay and not _NO_DIST:
            nccl_lsa_context = create_nccl_lsa_context(
                rank=rank,
                world_size=world_size,
            )
            if rank == 0:
                print("NCCL LSA context ready", flush=True)

        tester = MegaMoETester(
            problem,
            impl,
            misc,
            rank=rank,
            nccl_gin_context=nccl_gin_context,
            nccl_lsa_context=nccl_lsa_context,
            config=config,
        )
        nccl_gin_context = None
        nccl_lsa_context = None
        tester.set_torch_profiler_enabled(args.use_torch_profiler)
        tester.set_perf_iters(args.perf_warmup, args.perf_iters)
        tester.set_perf_sync_each_launch(args.perf_sync_each_launch)
        tester.set_validate_each_launch(args.validate_each_launch)
        tester.run()
    except NotImplementedError as exc:
        # Expected until the MegaMoE kernel side is wired; the host
        # orchestration above is the part being smoke-tested for now.
        if rank == 0:
            print(f"[mega_runner] kernel launch skipped: {exc}")
    except NcclGinUnavailableError as exc:
        return_code = 1
        if rank == 0:
            print(f"[mega_runner] NCCL GIN unavailable: {exc}")
    finally:
        if tester is not None:
            tester.close()
        elif nccl_gin_context is not None:
            nccl_gin_context.close()
        elif nccl_lsa_context is not None:
            nccl_lsa_context.close()

    if not _NO_DIST:
        # Rank-align before process exit. Legacy NVSHMEM free/finalize can
        # deadlock when per-rank GC order diverges, so retain the established
        # os._exit policy for normal runs; relay mode has already closed its
        # NCCL window/comm. IKET runs need normal Python process teardown
        # because libsmodel_injection writes decoded trace JSON from a finalizer.
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        sys.stdout.flush()
        sys.stderr.flush()
        if not args.enable_iket:
            os._exit(return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
