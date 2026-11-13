# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Multi-rank SM120 NVFP4-weight/NVFP4-activation MegaMoE host driver.

This runner owns the distributed split path: dispatch, FC1, FC2 and combine.
Ground truth is the eyeball-reviewable
``compute_megamoe_reference`` in ``mega_reference.py``; everything in
this file is glue.

Design choices worth knowing:

  * Inputs are generated deterministically across ranks from a shared
    seed (numpy rng + ``torch.from_numpy(...).cuda()``).  Every rank
    reproduces the full ``(num_ranks, num_tokens_per_rank, *)`` view
    and the ``(num_ranks, num_experts_per_rank, ...)`` weight bank in
    lock-step -- zero NCCL traffic during input gen.

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
            activation, activation_sf,           # packed NVFP4 + E4M3 SF
            topk_idx, topk_weights,
            fc1_weight, fc1_weight_sf,       # packed E2M1 + E8M0
            fc2_weight, fc2_weight_sf,       # packed E2M1 + E8M0
            combine_output,                      # sym heap
            local_workspace, shared_workspace,   # opaque uint8
            # ``peer_rank_ptr_mapper_host`` carries runtime peer offsets;
            # kernel.__call__ packs it into SymBuffer{world_size}.
            stream): ...
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List, Literal, Optional, Tuple

import numpy as np
import torch
from .transport import gin_bridge
from .runtime.low_precision_dlpack import wrap as wrap_dlpack
from cuda.bindings import driver as cuda


_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS = 256
_IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS = _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS - 1
_IBGDA_OWNER_PLAN_SLOTS = 2


# Ensure absolute package imports work when run as a script.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_PKG_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

from moe_sm120_nvfp4_split.combine_reference import (
    enforce_combine_reports,
    flat_topk_bf16_reference,
    make_combine_report,
    owner_grouped_bf16_reference,
)
from moe_sm120_nvfp4_split.combine_validation import rank_local_staging_report

from moe_sm120_nvfp4_split.support.common.megamoe_constants import Nvfp4BlockSize
from moe_sm120_nvfp4_split.sm120_mma import (
    INTERMEDIATE_ALIGNMENT,
)
from moe_sm120_nvfp4_split.runner_common import (
    _stack_byte_reinterpretable_tensors,
    ceil_div,
    Nvfp4DataDtype,
    Nvfp4ScaleDtype,
    round_up,
    to_blocked,
)
from moe_sm120_nvfp4_split.mega_reference import compute_megamoe_reference_nvfp4
from moe_sm120_nvfp4_split.config import (
    ImplDesc,
    MiscDesc,
)
from moe_sm120_nvfp4_split.heuristic import (
    MegaMoEHeuristicInput,
    MegaMoEHeuristicOverrides,
    MegaMoEKernelConfig,
    select_megamoe_config,
    select_compatibility_config,
)
from moe_sm120_nvfp4_split.api import (
    MegaMoECompileSpec,
    MegaMoEProblemSpec,
    SplitKernelBuildOptions,
    build_split_kernels,
    is_calibrated_execution,
)
from moe_sm120_nvfp4_split.jit_config import Sm120JitConfig
from moe_sm120_nvfp4_split.transport.hybrid_transport import HybridTransportState
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

    ``route_distribution`` controls how the pre-dispatch routing table is
    generated; K1 groups the routed rows into local expert pools.

    v1 layout / dtype locks: packed NVFP4 data, E4M3 plain K-major SF on inputs
    (no atom swizzle at the user boundary
    -- the kernel's dispatch warps swizzle SF internally before fc1 reads
    them), BF16 ``fc2_output_dtype``.

    TODO: expose ``activation_global_scale`` / ``fc1_weight_global_scale``
    / ``fc2_weight_global_scale`` / ``norm_const`` as fields once they're
    randomized; the reference currently pins them to 1.0.
    """

    world_size: int = 0
    num_tokens_per_rank: int = 0
    num_topk: int = 0
    num_total_experts: int = 0
    hidden: int = 0
    intermediate: int = 0
    fc2_output_dtype: torch.dtype = torch.bfloat16
    # ``balanced``: locally uniform random block assignment, with exact
    # per-expert balance on each full E-token block and randomized tails.
    # ``power_law``: Zipf-rank-frequency, matches the empirical
    # token-to-expert distribution shape reported by the TRT team.
    route_distribution: Literal[
        "balanced", "power_law", "zero_count_peer"
    ] = "balanced"
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
        if self.hidden <= 0 or self.hidden % 32 != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be a positive multiple of "
                "32 for packed NVFP4 TMA alignment."
            )
        if self.intermediate <= 0:
            raise ValueError(f"intermediate must be positive, got {self.intermediate}.")
        if self.intermediate % INTERMEDIATE_ALIGNMENT != 0:
            raise ValueError(
                f"intermediate ({self.intermediate}) must be a positive multiple of "
                f"{INTERMEDIATE_ALIGNMENT} for the current "
                "swapAB epilogue."
            )
        if (self.intermediate // 2) % Nvfp4BlockSize != 0:
            raise ValueError(
                f"intermediate/2 ({self.intermediate // 2}) must be a multiple of "
                f"sf_vec_size ({Nvfp4BlockSize})."
            )
        if self.fc2_output_dtype != torch.bfloat16:
            raise ValueError(
                f"fc2_output_dtype must be torch.bfloat16, "
                f"got {self.fc2_output_dtype}."
            )
        if self.route_distribution not in (
            "balanced", "power_law", "zero_count_peer"
        ):
            raise ValueError(
                "route_distribution must be 'balanced', 'power_law', or "
                f"'zero_count_peer', "
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


def _generate_topk_idx_zero_count_peer(
    num_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_total_experts: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Balanced diagnostic route with the last owner receiving zero rows."""

    if num_ranks < 2:
        raise ValueError("zero_count_peer routing requires at least two ranks")
    experts_per_rank = num_total_experts // num_ranks
    active_experts = num_total_experts - experts_per_rank
    if num_topk > active_experts:
        raise ValueError(
            f"topk={num_topk} exceeds {active_experts} active experts after "
            "reserving one zero-count peer"
        )
    return _generate_topk_idx_balanced(
        num_ranks,
        num_tokens_per_rank,
        num_topk,
        active_experts,
        rng,
    )


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
# NVFP4 tensor builders (host-side torch RNG -> CUDA)
# =============================================================================


_TorchActivationDtype = Nvfp4DataDtype
_TorchWeightDtype = torch.float4_e2m1fn_x2


def _make_fp8_tensor(
    torch_rng: torch.Generator,
    shape: Tuple[int, ...],
    data_dtype: torch.dtype,
    *,
    perf_run: bool,
) -> torch.Tensor:
    """Build a finite FP8 tensor for MXFP8 input or weight data."""

    n = 1
    for s in shape:
        n *= s
    if perf_run:
        # Generate storage bytes directly. The previous int64 randint used 8x
        # the final tensor size and OOMed while building full DSV4 weights.
        # Restricting codes to the finite positive range is sufficient for a
        # data-independent performance run and needs no conversion workspace.
        if data_dtype == torch.float8_e4m3fn:
            code_limit = 127
        elif data_dtype == torch.float8_e5m2:
            code_limit = 124
        else:
            raise ValueError(f"Unsupported fp8 dtype: {data_dtype}")
        flat_bytes = torch.randint(
            0, code_limit, (n,), dtype=torch.uint8, device="cuda", generator=torch_rng,
        )
        return flat_bytes.view(data_dtype).reshape(shape)

    # Correctness inputs use sparse, exactly representable values.  Generate
    # them in chunks so full DSV4 weights do not require two 63-GiB FP32
    # temporaries in addition to the final FP8 tensor.
    nz_pct = float(os.environ.get("MXFP8_NONZERO_PCT", "1")) / 100.0
    chunk_elems = int(os.environ.get("MXFP8_GEN_CHUNK_ELEMS", str(16 * 1024 * 1024)))
    flat = torch.empty((n,), dtype=data_dtype, device="cuda")
    for start in range(0, n, chunk_elems):
        count = min(chunk_elems, n - start)
        fp32 = torch.zeros((count,), dtype=torch.float32, device="cuda")
        rand = torch.rand((count,), device="cuda", generator=torch_rng)
        fp32[rand < nz_pct] = 0.5
        fp32[(rand >= nz_pct) & (rand < 2.0 * nz_pct)] = -0.5
        flat[start : start + count].copy_(fp32)
    return flat.reshape(shape)


def _make_fp4_weight_tensor(
    torch_rng: torch.Generator,
    logical_shape: Tuple[int, ...],
    *,
    perf_run: bool,
) -> torch.Tensor:
    """Create packed E2M1 storage for a K-major logical weight tensor.

    ``torch.float4_e2m1fn_x2`` exposes one storage byte as one torch element,
    while CuTe's DLPack adapter exposes the two packed nibbles as two logical
    ``Float4E2M1FN`` elements.  Therefore the torch-visible last dimension is
    half of the logical K dimension.
    """

    if not logical_shape or logical_shape[-1] % 2:
        raise ValueError(
            f"packed E2M1 requires an even logical K, got {logical_shape}"
        )
    physical_shape = (*logical_shape[:-1], logical_shape[-1] // 2)
    nbytes = math.prod(physical_shape)
    if perf_run:
        packed = torch.randint(
            0,
            256,
            (nbytes,),
            dtype=torch.uint8,
            device="cuda",
            generator=torch_rng,
        )
    else:
        # Sparse, exactly representable +/-0.5 values. E2M1 codes 0x1/0x9
        # occupy the low/high nibble in logical element order.
        nz_pct = float(os.environ.get("MXFP4_NONZERO_PCT", "1")) / 100.0
        chunk_bytes = int(
            os.environ.get("MXFP4_GEN_CHUNK_BYTES", str(16 * 1024 * 1024))
        )
        packed = torch.empty((nbytes,), dtype=torch.uint8, device="cuda")
        for start in range(0, nbytes, chunk_bytes):
            count = min(chunk_bytes, nbytes - start)
            low_r = torch.rand((count,), device="cuda", generator=torch_rng)
            high_r = torch.rand((count,), device="cuda", generator=torch_rng)
            low = torch.where(
                low_r < nz_pct,
                torch.ones_like(low_r, dtype=torch.uint8),
                torch.where(
                    low_r < 2.0 * nz_pct,
                    torch.full_like(low_r, 9, dtype=torch.uint8),
                    torch.zeros_like(low_r, dtype=torch.uint8),
                ),
            )
            high = torch.where(
                high_r < nz_pct,
                torch.ones_like(high_r, dtype=torch.uint8),
                torch.where(
                    high_r < 2.0 * nz_pct,
                    torch.full_like(high_r, 9, dtype=torch.uint8),
                    torch.zeros_like(high_r, dtype=torch.uint8),
                ),
            )
            packed[start : start + count] = low | (high << 4)
    return packed.view(_TorchWeightDtype).reshape(physical_shape)


def _make_nvfp4_scale_tensor(
    torch_rng: torch.Generator,
    non_k_size: int,
    k_size: int,
    *,
    blocksize: int,
) -> torch.Tensor:
    """Plain per-block E4M3 NVFP4 scale tensor."""

    num_scale_cols = ceil_div(k_size, blocksize)
    rand = torch.rand(
        (non_k_size, num_scale_cols), device="cuda", generator=torch_rng
    )
    return torch.where(
        rand < 0.80,
        torch.ones_like(rand),
        torch.full_like(rand, 1.5),
    ).to(Nvfp4ScaleDtype)


# =============================================================================
# Symmetric-heap allocation
# =============================================================================


# MEGA_NO_DIST=1: opt out of torch.distributed + communication init so the
# runner can be launched as plain ``python mega_runner.py`` (e.g. under
# compute-sanitizer).  Single-rank only; sym tensors degrade to plain
# CUDA ``torch.zeros`` and the SymBuffer offsets list is hard-coded to
# ``(0,) * world_size``.
_NO_DIST: bool = bool(int(os.environ.get("MEGA_NO_DIST", "0")))


def _sym_zeros(shape: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if _NO_DIST:
        return torch.zeros(shape, dtype=dtype, device='cuda')
    return gin_bridge.arena_zeros(shape, dtype)


def _sym_zeros_byte_view(
    logical_shape: Tuple[int, ...], target_dtype: torch.dtype,
) -> torch.Tensor:
    """P2P arena allocation using byte views for packed low-precision dtypes."""

    if target_dtype not in (
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        Nvfp4DataDtype,
        Nvfp4ScaleDtype,
    ):
        raise ValueError(f"_sym_zeros_byte_view: unsupported dtype {target_dtype}.")
    storage_shape = tuple(logical_shape)
    if target_dtype == Nvfp4DataDtype:
        if storage_shape[-1] % 2:
            raise ValueError(
                f"packed NVFP4 logical row must be even, got {storage_shape}."
            )
        storage_shape = (*storage_shape[:-1], storage_shape[-1] // 2)
    total_bytes = 1
    for dim_size in storage_shape:
        total_bytes *= dim_size
    return _sym_zeros((total_bytes,), torch.uint8).view(target_dtype).reshape(storage_shape)


def _compute_peer_offsets(sym_tensor, world_size):
    if _NO_DIST:
        return int(sym_tensor.data_ptr()), (0,) * world_size
    return gin_bridge.peer_offsets(sym_tensor, world_size)


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

    None of steps 1-3 emit NCCL traffic; only step 4 (kernel
    launch) and step 5's compare touch sym memory.
    """

    def __init__(
        self,
        problem: TokenCommProblemDesc,
        impl: ImplDesc,
        misc: MiscDesc,
        *,
        rank: int,
        kernel_config: MegaMoEKernelConfig,
        hybrid_force_ibgda_mask: int = 0,
        parallel_context=None,
        split_launch: Literal["sequential", "green_graph"] = "sequential",
        jit_config: Optional[Sm120JitConfig] = None,
        dual_k2_ctas_per_sm: Optional[int] = None,
    ) -> None:
        self.problem = problem
        self.impl = impl
        self.misc = misc
        self.kernel_config = kernel_config
        self._dual_k2_ctas_per_sm = (
            int(os.environ.get("MEGA_DUAL_K2_CTAS_PER_SM", "2"))
            if dual_k2_ctas_per_sm is None else dual_k2_ctas_per_sm
        )
        if type(self._dual_k2_ctas_per_sm) is not int or self._dual_k2_ctas_per_sm <= 0:
            raise ValueError("dual K2 CTAs/SM must be a positive integer")
        self.jit_config = (
            jit_config if jit_config is not None else Sm120JitConfig.from_environment()
        )
        self.comm_backend = kernel_config.kernel_comm_backend
        self.rank = rank
        if hybrid_force_ibgda_mask < 0:
            raise ValueError("hybrid_force_ibgda_mask must be non-negative")
        if hybrid_force_ibgda_mask & (1 << rank):
            raise ValueError("hybrid transport must not mask the local PE")
        if hybrid_force_ibgda_mask >> problem.world_size:
            raise ValueError(
                "hybrid_force_ibgda_mask contains a PE outside the EP world"
            )
        self.hybrid_force_ibgda_mask = hybrid_force_ibgda_mask
        self.parallel_context = parallel_context
        self.global_rank = (
            parallel_context.global_rank
            if parallel_context is not None else rank
        )
        self.global_world_size = (
            parallel_context.global_world_size
            if parallel_context is not None else problem.world_size
        )
        self.data_parallel_rank = (
            parallel_context.data_parallel_rank
            if parallel_context is not None else 0
        )
        self.data_parallel_size = (
            parallel_context.data_parallel_size
            if parallel_context is not None else 1
        )
        self.tensor_parallel_rank = (
            parallel_context.tensor_parallel_rank
            if parallel_context is not None else 0
        )
        self.tensor_parallel_size = (
            parallel_context.tensor_parallel_size
            if parallel_context is not None else 1
        )
        self._ep_group = (
            parallel_context.ep_group
            if parallel_context is not None else None
        )
        self._tp_group = (
            parallel_context.tp_group
            if parallel_context is not None else None
        )
        self._ep_global_ranks = (
            parallel_context.ep_global_ranks
            if parallel_context is not None
            else tuple(range(problem.world_size))
        )
        if split_launch not in ("sequential", "green_graph"):
            raise ValueError(f"unsupported split_launch={split_launch!r}")
        device_sms = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
        if kernel_config.total_sms != device_sms:
            raise ValueError(
                f"SM120 config must cover all {device_sms} device SMs, got "
                f"{kernel_config.total_sms}."
            )
        k2_warp_count = kernel_config.k2_warps
        k2_mma_tiler_mnk = kernel_config.k2_tile
        k2_num_ab_stages = kernel_config.k2_stages
        if k2_warp_count not in (8, 12):
            raise ValueError(
                f"k2_warp_count must be 8 or 12, got {k2_warp_count}."
            )
        self.split_launch = split_launch
        self.torch_activation_dtype = _TorchActivationDtype
        self.torch_weight_dtype = _TorchWeightDtype
        if (
            k2_mma_tiler_mnk[0] != 64
            or k2_mma_tiler_mnk[1] not in (16, 32, 64, 128)
        ):
            raise ValueError(
                "K2 tiler must be M64 with N in (16, 32, 64, 128), got "
                f"{k2_mma_tiler_mnk}."
            )
        self.k2_warp_count = k2_warp_count
        if k2_num_ab_stages not in (1, 2, 3, 4, 5):
            raise ValueError(
                f"k2_num_ab_stages must be in 1..5; constructor enforces SMEM feasibility, got "
                f"{k2_num_ab_stages}."
            )
        self.k2_mma_tiler_mnk = k2_mma_tiler_mnk
        self.k2_num_ab_stages = k2_num_ab_stages
        self.k2_tail_reclaim = kernel_config.k2_tail_reclaim
        self.k2_ready_queue = kernel_config.k2_ready_queue
        self.k1_ready_queue = kernel_config.k1_ready_queue
        self.dispatch_rank_cache = kernel_config.dispatch_rank_cache
        self.rank_local_combine = kernel_config.rank_local_combine
        if self.k2_tail_reclaim and split_launch != "green_graph":
            raise ValueError(
                "K2 tail reclaim requires --split_launch green_graph."
            )
        # Single source of truth for world size; ``local_rank`` was only
        # needed for the upstream ``torch.cuda.set_device`` inside the
        # NCCL bootstrap, the tester itself doesn't reference it.
        self.world_size = problem.world_size

        torch.manual_seed(misc.seed)
        np.random.seed(misc.seed)
        self._np_rng: np.random.Generator = np.random.default_rng(misc.seed)
        self._torch_cuda_rng = torch.Generator(device="cuda")
        self._torch_cuda_rng.manual_seed(misc.seed)
        self._weight_cuda_rng = torch.Generator(device="cuda")
        self._weight_cuda_rng.manual_seed(
            misc.seed + self.tensor_parallel_rank * 1_000_003
        )
        self._epi_cuda_rng = torch.Generator(device="cuda")
        self._epi_cuda_rng.manual_seed(misc.seed + 2_000_003)

        # Inputs arrive pre-arranged for the caller's TP mapping. Each TP rank
        # receives the same logical token rows and dispatch peer-pulls them
        # directly from the symmetric heap; MegaMoE performs no TP input copy.
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

        self.combine_output: Optional[torch.Tensor] = None
        self.combine_reduced_output: Optional[torch.Tensor] = None
        self.rank_combine_ready: Optional[torch.Tensor] = None
        self._rank_combine_epoch = 0
        self.output_tensor: Optional[torch.Tensor] = None
        self.output_tensor_ref: Optional[torch.Tensor] = None
        self.combine_output_ref: Optional[torch.Tensor] = None
        self.combine_reduced_output_ref: Optional[torch.Tensor] = None
        self.rank_local_output_ref: Optional[torch.Tensor] = None
        self._tp_reference_reduced = False
        self._combine_output_ref_global: Optional[torch.Tensor] = None
        self._tp_k3_chunk_count = kernel_config.tp_k3_chunks
        self._tp_k3_chunk_runtime_kwargs = None
        self._tp_k3_chunk_outputs = None

        self.local_workspace: Optional[torch.Tensor] = None
        self.shared_workspace: Optional[torch.Tensor] = None
        from moe_sm120_nvfp4_split.transport.hybrid_transport import (
            HybridTransportState,
        )

        self._transport = HybridTransportState.from_config(kernel_config)
        # Host-side SymBuffer payload.  It is passed as a runtime arg via
        # SymBufferHost and packed into SymBuffer{world_size} in
        # kernel.__call__.  No GMEM tensor is allocated for these offsets.
        self.symmetric_base: Optional[int] = None
        self.peer_offsets_list: Optional[Tuple[int, ...]] = None

        self._kernel = None
        self._compiled_kernel = None
        self._kernel_k1 = None
        self._kernel_k2 = None
        self._kernel_k2_drain = None
        self._kernel_k2_finalizer = None
        self._compiled_kernel_k1 = None
        self._compiled_kernel_k2 = None
        self._compiled_kernel_k2_drain = None
        self._compiled_kernel_k2_finalizer = None
        # Materialize every communication function before starting persistent
        # receivers. A first-use module load may synchronize the context and
        # deadlock behind a receiver that is waiting for that very sender.
        # CUDA_MODULE_LOADING=EAGER alone is insufficient for CuTe CUlibrary.
        def checked(result):
            if int(result[0]) != 0:
                raise RuntimeError(f'Communication function preload failed: {result}')
            return result[1] if len(result) == 2 else result[1:]
        loaded = set()
        for name, compiled in vars(self._transport).items():
            if not name.startswith('compiled_') or compiled is None:
                continue
            library = int(compiled.library)
            if library in loaded:
                continue
            loaded.add(library)
            module = checked(cuda.cuLibraryGetModule(cuda.CUlibrary(library)))
            count = checked(cuda.cuModuleGetFunctionCount(module))
            for function in checked(cuda.cuModuleEnumerateFunctions(count, module)):
                checked(cuda.cuFuncLoad(function))
        self._runtime_kwargs_k1 = None
        self._runtime_kwargs_k2 = None
        self._runtime_kwargs_k2_drain = None
        self._runtime_kwargs_k2_finalizer = None
        self._k1_torch_stream = None
        self._k2_torch_stream = None
        self._native_green_resources = None
        self._native_green_graph = None
        self._graph_root_stream = None
        self._host_control_group = (
            parallel_context.ep_control_group
            if parallel_context is not None else None
        )
        self._device_launch_started = False
        self._device_launch_completed = False
        self._compiled_topk_reduce = None
        self._use_torch_profiler = False
        self._use_cuda_events = False
        self._perf_warmup = 1
        self._perf_iters = 1
        self._dynamic_route_replays = 0
        self._transport_replay_validation = False
        self._verify_replay_bit_determinism = False
        self._phase_timing_enabled = (
            os.environ.get("MEGA_IBGDA_PHASE_TIMING", "0") == "1"
        )
        self._phase_timing_collect = False
        self._phase_timing_events = {}
        self._split_globaltimer_enabled = self.jit_config.enable_globaltimer
        self._split_k2_tile_trace_enabled = (
            self.jit_config.enable_k2_tile_trace
        )
        if self._split_k2_tile_trace_enabled:
            self._split_globaltimer_enabled = True
        self._split_green_trace: Optional[torch.Tensor] = None

        # v1 pinned expert-wise epilogue args.
        self.my_fc1_alpha: Optional[torch.Tensor] = None
        self.my_fc2_alpha: Optional[torch.Tensor] = None
        self.my_fc1_norm_const: Optional[torch.Tensor] = None

    def set_torch_profiler_enabled(self, enabled: bool) -> None:
        """Enable optional torch profiler timing for the target kernel launch."""
        self._use_torch_profiler = enabled

    def set_cuda_event_timing_enabled(self, enabled: bool) -> None:
        """Enable low-overhead per-launch CUDA-event timing."""
        self._use_cuda_events = enabled

    def set_perf_iters(self, warmup: int, iters: int) -> None:
        """Configure steady-state timing: `warmup` untimed launches then
        `iters` back-to-back timed launches."""
        self._perf_warmup = max(0, int(warmup))
        self._perf_iters = max(1, int(iters))

    def set_dynamic_route_replays(self, replays: int) -> None:
        """Change the GPU-resident top-k route before every timed replay.

        This is a benchmark stress mode for the runtime route/count control
        plane.  Route generation and the host-visible fingerprints are kept
        outside CUDA-event timing.
        """
        self._dynamic_route_replays = max(0, int(replays))
        if self._dynamic_route_replays:
            # Every replay installs a new GPU-resident route.  Separate
            # negative-index warmups create a benchmark-only epoch boundary
            # that production never executes; the replay series itself is
            # sufficient to reach steady state.
            self._perf_warmup = 0
            self._perf_iters = self._dynamic_route_replays

    def set_transport_replay_validation(
        self, *, byte_exact: bool, bit_determinism: bool
    ) -> None:
        if bit_determinism and self._dynamic_route_replays:
            raise ValueError(
                "bit determinism requires a static route across replays"
            )
        self._transport_replay_validation = byte_exact
        self._verify_replay_bit_determinism = bit_determinism

    def _update_runtime_route(self, replay: int) -> None:
        if not self._dynamic_route_replays:
            return
        if self.my_topk_idx is None:
            raise RuntimeError("dynamic route update requires generated inputs")
        tokens = self.problem.num_tokens_per_rank
        topk = self.problem.num_topk
        experts = self.problem.num_total_experts
        token = torch.arange(tokens, dtype=torch.int64, device="cuda")
        # Keep the affine base a permutation for non-power-of-two expert
        # counts as well; the original multiplier had gcd=3 with 384.
        route_multiplier = 6364136223846793005
        while math.gcd(route_multiplier, experts) != 1:
            route_multiplier += 2
        replay_term = (
            (replay + 17) * 2862933555777941757
        ) % 9223372036854775783
        mixed = (
            token * route_multiplier
            + (self.rank + 1) * 1442695040888963407
            + replay_term
        )
        strides = torch.tensor(
            (1, 5, 7, 11, 13, 17, 19, 23),
            dtype=torch.int64,
            device="cuda",
        )
        stride = strides[torch.remainder(mixed, strides.numel())]
        slots = torch.arange(topk, dtype=torch.int64, device="cuda")
        stress_pattern = os.environ.get(
            "MEGA_DYNAMIC_ROUTE_PATTERN", "balanced"
        )
        if stress_pattern == "balanced":
            route = torch.remainder(
                torch.remainder(mixed, experts)[:, None]
                + stride[:, None] * slots[None, :],
                experts,
            )
        else:
            if self.world_size != 4 or experts % self.world_size:
                raise ValueError(
                    "dynamic owner stress patterns currently require EP4 "
                    "and experts divisible by four"
                )
            per_owner = experts // self.world_size
            slot_hash = torch.remainder(
                mixed[:, None]
                + slots[None, :] * 2862933555777941757,
                9223372036854775783,
            )
            local_expert = torch.remainder(
                slot_hash + slots[None, :] * 17,
                per_owner,
            )
            use_power_law = stress_pattern == "power_law_owner" or (
                stress_pattern == "mixed_owner_stress" and replay % 2 == 0
            )
            use_zero_owner = stress_pattern == "zero_count_owner" or (
                stress_pattern == "mixed_owner_stress" and replay % 2 != 0
            )
            if use_power_law:
                # EP4 owner probabilities 6:3:2:1.  Expert IDs remain
                # uniform inside each owner so the balanced pool-capacity
                # specialization is sufficient for this control-plane test.
                bucket = torch.remainder(slot_hash, 12)
                owner = torch.where(
                    bucket < 6,
                    torch.zeros_like(bucket),
                    torch.where(
                        bucket < 9,
                        torch.ones_like(bucket),
                        torch.where(
                            bucket < 11,
                            torch.full_like(bucket, 2),
                            torch.full_like(bucket, 3),
                        ),
                    ),
                )
            elif use_zero_owner:
                excluded_owner = replay % self.world_size
                owner = torch.remainder(slot_hash, self.world_size - 1)
                owner = owner + (owner >= excluded_owner).to(owner.dtype)
            else:
                raise ValueError(
                    "MEGA_DYNAMIC_ROUTE_PATTERN must be balanced, "
                    "power_law_owner, zero_count_owner, or mixed_owner_stress"
                )
            route = owner * per_owner + local_expert
        self.my_topk_idx.copy_(route)
        # Dynamic-route mode is a benchmark-only input producer.  Complete
        # that producer before entering the production launch path, matching
        # the real caller contract that topk_idx is already GPU-resident and
        # ready when MegaMoE is invoked.
        torch.cuda.current_stream().synchronize()
        # Dynamic-route mode emulates a caller-provided GPU-resident route.
        # Every EP source must finish publishing it before remote owners read.
        self._ep_barrier()

    def _ep_all_gather(self, output_tensors, input_tensor) -> None:
        """Gather over this rank's EP world, never over the global world."""
        torch.distributed.all_gather(
            output_tensors,
            input_tensor,
            group=self._ep_group,
        )

    def _ep_barrier(self) -> None:
        """Synchronize only the NUMA-local EP world."""
        torch.distributed.barrier(group=self._ep_group)

    def _tp_all_reduce(
        self,
        tensor: torch.Tensor,
        *,
        async_op: bool = False,
    ):
        """Sum row-parallel FC2 partials for one logical EP rank."""
        if self.tensor_parallel_size <= 1:
            return None
        if self._tp_group is None:
            raise RuntimeError(
                "tensor_parallel_size > 1 requires a TP process group."
            )
        # NativeGreenContextGraph.launch() records graph completion on the
        # caller stream before returning. ProcessGroupNCCL then establishes
        # the usual current-stream dependency for this tensor, so a device-wide
        # fence before or after the collective is both redundant and harmful to
        # K3/all-reduce pipelining.
        return torch.distributed.all_reduce(
            tensor,
            group=self._tp_group,
            async_op=async_op,
        )

    def _tp_reduce_reference_output(self) -> None:
        if (
            self.combine_reduced_output_ref is not None
            and not self._tp_reference_reduced
        ):
            self._tp_all_reduce(self.combine_reduced_output_ref)
            self._tp_reference_reduced = True

    def _launch_chunked_k3_tp_reduce(self) -> None:
        if (
            self._compiled_topk_reduce is None
            or self._tp_k3_chunk_runtime_kwargs is None
            or self._tp_k3_chunk_outputs is None
        ):
            raise RuntimeError("chunked TP K3 runtime is not initialized.")

        for runtime_kwargs, _output_chunk in zip(
            self._tp_k3_chunk_runtime_kwargs,
            self._tp_k3_chunk_outputs,
        ):
            self._compiled_topk_reduce(**runtime_kwargs)
        if self._tp_reduce_enabled():
            if self.combine_reduced_output is None:
                raise RuntimeError("TP all-reduce needs reduced output storage.")
            self._tp_all_reduce(self.combine_reduced_output)
            self._tp_reduce_reference_output()

    def _tp_reduce_enabled(self) -> bool:
        return (
            self.tensor_parallel_size > 1
            and os.environ.get("MEGA_SPLIT_DISABLE_TP_REDUCE", "0") != "1"
        )

    def _tp_reduce_actual_output(self) -> None:
        if not self._tp_reduce_enabled():
            return
        if self.combine_reduced_output is None:
            raise RuntimeError("TP reduce requires the K3 reduced output.")
        self._tp_all_reduce(self.combine_reduced_output)
        # Keep reference collectives out of compute_reference(): launching a
        # TP NCCL collective there can overlap communication/CuTeDSL setup on the
        # other execution context.  Once K3 has completed, reduce the actual
        # output and its reference back-to-back in the same TP-group order.
        self._tp_reduce_reference_output()

    def _ensure_host_control_group(self) -> None:
        """Create a CPU/Gloo control plane before any rank starts compiling."""
        if (
            self.world_size <= 1
            or not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or self._host_control_group is not None
        ):
            return
        timeout_s = int(os.environ.get("MEGAMOE_HOST_BARRIER_TIMEOUT_S", "180"))
        self._host_control_group = torch.distributed.new_group(
            ranks=list(self._ep_global_ranks),
            backend="gloo",
            timeout=timedelta(seconds=timeout_s),
        )

    def _host_control_barrier(self, phase: str) -> None:
        """Fail on a missing rank before any persistent device wait begins."""
        self._ensure_host_control_group()
        if self._host_control_group is None:
            return
        timeout_s = int(os.environ.get("MEGAMOE_HOST_BARRIER_TIMEOUT_S", "180"))
        try:
            torch.distributed.monitored_barrier(
                group=self._host_control_group,
                timeout=timedelta(seconds=timeout_s),
                wait_all_ranks=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"rank {self.rank} failed host rendezvous at {phase}; "
                "the MegaMoE device kernels were not launched"
            ) from exc

    @staticmethod
    def _validate_kernel_tail_topology(kernel, label: str) -> None:
        """Reject CTA layouts that cannot satisfy the kernel-tail barrier."""
        actual = int(kernel.threads_per_cta)
        expected = int(kernel.token_comm.kernel_tail_threads)
        if actual != expected:
            raise RuntimeError(
                f"{label} tail topology mismatch: launch has {actual} "
                f"threads/CTA but token_comm waits for {expected}. "
                "Launching this kernel would deadlock."
            )

    def close_split_execution_contexts(self) -> None:
        """Release Green Context resources without waiting on a failed launch."""
        safe_to_synchronize = (
            not self._device_launch_started or self._device_launch_completed
        )
        if not safe_to_synchronize:
            if self.rank == 0:
                print(
                    "[mega_runner] abandoning Green Context handles after an "
                    "incomplete device launch; skipping blocking stream sync",
                    file=sys.stderr,
                )
            return
        if self._native_green_graph is not None:
            self._native_green_graph.close()
            self._native_green_graph = None
        if self._native_green_resources is not None:
            self._native_green_resources.close()
            self._native_green_resources = None

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
        self._ep_all_gather(gathered, sentinel)

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

        activation_dtype = self.torch_activation_dtype

        # ---- Activation (packed E2M1 + plain K-major E4M3 scale).
        # Scale lives in the user-facing plain layout; dispatch warps re-stage
        # it into the MMA scale layout inside l1_sf_buffer before FC1 reads it.
        self._global_activation = _make_fp4_weight_tensor(
            self._torch_cuda_rng,
            (num_ranks, num_tokens_per_rank, hidden),
            perf_run=self.misc.perf_run,
        )
        self._global_activation_sf = _make_nvfp4_scale_tensor(
            self._torch_cuda_rng,
            num_ranks * num_tokens_per_rank,
            hidden,
            blocksize=scale_blocksize,
        ).reshape(num_ranks, num_tokens_per_rank, hidden_sf_cols)

        # ---- Routing table.
        if problem.route_distribution == "balanced":
            topk_idx_np = _generate_topk_idx_balanced(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, rng,
            )
        elif problem.route_distribution == "power_law":
            topk_idx_np = _generate_topk_idx_power_law(
                num_ranks, num_tokens_per_rank, num_topk,
                problem.num_total_experts, problem.power_law_exponent, rng,
            )
        else:
            topk_idx_np = _generate_topk_idx_zero_count_peer(
                num_ranks,
                num_tokens_per_rank,
                num_topk,
                problem.num_total_experts,
                rng,
            )
        topk_weights = _generate_topk_weights(
            num_ranks, num_tokens_per_rank, num_topk, self._torch_cuda_rng,
        )
        if self.rank == 0:
            _print_remote_rank_comm_matrices(
                topk_idx_np, num_ranks, problem.num_total_experts,
            )
        self._global_topk_idx = torch.from_numpy(topk_idx_np).cuda()
        self._global_topk_weights = topk_weights

        # ---- Weights.  fc1/fc2 are stored in GEMM K-major logical layouts.
        # TP ranks use independent intermediate-dimension shards while every
        # EP rank in the same TP world deterministically reproduces the same
        # full expert bank.  TP1 intentionally retains the original RNG
        # stream for bit-for-bit backward compatibility.
        # TP0 continues from the common activation/routing RNG stream just as
        # the original TP1 runner does.  Higher TP ranks use independent,
        # deterministic streams for their complementary weight shards.
        weight_rng = (
            self._torch_cuda_rng
            if self.tensor_parallel_rank == 0
            else self._weight_cuda_rng
        )
        epi_rng = (
            self._torch_cuda_rng
            if self.tensor_parallel_size == 1
            else self._epi_cuda_rng
        )
        # Packed FP4 weights are physically byte tensors with half-sized K:
        # fc1 logical (E, gateup, hidden), fc2 logical (E, hidden, downproj).
        self._global_fc1_weight = _make_fp4_weight_tensor(
            weight_rng,
            (num_ranks, num_experts_per_rank, intermediate, hidden),
            perf_run=self.misc.perf_run,
        )
        self._global_fc1_weight_sf = _make_nvfp4_scale_tensor(
            weight_rng,
            num_ranks * num_experts_per_rank * intermediate,
            hidden,
            blocksize=scale_blocksize,
        ).reshape(num_ranks, num_experts_per_rank, intermediate, hidden_sf_cols)

        self._global_fc2_weight = _make_fp4_weight_tensor(
            weight_rng,
            (num_ranks, num_experts_per_rank, hidden, intermediate_downproj),
            perf_run=self.misc.perf_run,
        )
        self._global_fc2_weight_sf = _make_nvfp4_scale_tensor(
            weight_rng,
            num_ranks * num_experts_per_rank * hidden,
            intermediate_downproj,
            blocksize=scale_blocksize,
        ).reshape(
            num_ranks, num_experts_per_rank, hidden, intermediate_downproj_sf_cols,
        )
        epi_arg_shape = (num_ranks, num_experts_per_rank)
        self._global_fc1_alpha = torch.randint(
            1, 5, epi_arg_shape, generator=epi_rng, device="cuda",
        ).to(torch.float32) * 0.5
        self._global_fc2_alpha = torch.randint(
            1, 5, epi_arg_shape, generator=epi_rng, device="cuda",
        ).to(torch.float32) * 0.5
        self._global_fc1_norm_const = torch.randint(
            2, 5, epi_arg_shape, generator=epi_rng, device="cuda",
        ).to(torch.float32) * 0.5

        # ---- Atom-swizzle the weight SFs.  fc1_weight / fc2_weight are
        # consumed by the base kernel directly (NOT routed through dispatch
        # like the activation SF), so the host has to feed them in the
        # 32x4x4 atom-swizzled, flat 2D ``(experts, flat_atom_size)`` layout
        # that ``tile_atom_to_shape_SF`` and the TMA SFA descriptor expect.
        # Plain ``_global_fc1/2_weight_sf`` are kept untouched for the Torch
        # reference, which dequantizes from the raw plain layout.
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

        # ---- Stage the caller-provided TP-mapped input directly into the
        # symmetric heap. TP peers are expected to receive matching logical
        # token rows before entering MegaMoE.
        own_activation = self._global_activation[self.rank]
        own_activation_sf = self._global_activation_sf[self.rank]
        own_topk_idx = self._global_topk_idx[self.rank]
        own_topk_weights = self._global_topk_weights[self.rank]

        # Packed E2M1/E4M3 use byte-buffer views in the NCCL arena;
        # the original element types are retained by the tensor views.
        self.my_activation = _sym_zeros_byte_view(
            (num_tokens_per_rank, hidden), self.torch_activation_dtype,
        )
        self.my_activation.view(torch.uint8).copy_(
            own_activation.view(torch.uint8)
        )

        # SF leg caller contract: the K_sf axis (= ``ceil(hidden, 16)``)
        # must be a multiple of 4 fp8 SFs so dispatch_pull's LDG.32 byte
        # stride (= ``sf_uint32_per_token * 4 bytes/token``) matches the
        # host row stride.  Pad-to-multiple-of-4 + zero-fill on the host
        # side; the kernel never reads past ``ceil(hidden/32)`` valid SFs
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
            (num_tokens_per_rank, hidden_sf_cols_padded), Nvfp4ScaleDtype,
        )
        self.my_activation_sf.view(torch.uint8)[:, :hidden_sf_cols].copy_(
            own_activation_sf.view(torch.uint8)
        )

        self.my_topk_idx = _sym_zeros(tuple(own_topk_idx.shape), torch.int64)
        self.my_topk_idx.copy_(own_topk_idx)

        self.my_topk_weights = _sym_zeros(
            tuple(own_topk_weights.shape), torch.float32
        )
        self.my_topk_weights.copy_(own_topk_weights)

        # ---- Own-rank weights stay on regular cuda.  Keep the permuted
        # K-major stride pattern; ``.contiguous()`` would convert to row-major
        # and break the dequant + GEMM path's layout contract.
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

        if self.misc.perf_run and self.misc.skip_ref_check:
            # The reference path needs the full world-sized expert banks, but
            # a target-only performance run consumes only this EP rank's
            # shard.  Materialize that shard with the original non-contiguous
            # K-major strides, then release the otherwise-unused global banks.
            # This is essential for full DSV4 + GPU-heap staging on 72-GiB
            # RTX PRO 5000 cards and does not change any kernel input values.
            self.my_fc1_weight = self.my_fc1_weight.clone(
                memory_format=torch.preserve_format
            )
            self.my_fc1_weight_sf = self.my_fc1_weight_sf.clone()
            self.my_fc2_weight = self.my_fc2_weight.clone(
                memory_format=torch.preserve_format
            )
            self.my_fc2_weight_sf = self.my_fc2_weight_sf.clone()
            self.my_fc1_alpha = self.my_fc1_alpha.clone()
            self.my_fc2_alpha = self.my_fc2_alpha.clone()
            self.my_fc1_norm_const = self.my_fc1_norm_const.clone()
            for name in (
                "_global_fc1_weight",
                "_global_fc1_weight_sf",
                "_global_fc1_weight_sf_swizzled",
                "_global_fc2_weight",
                "_global_fc2_weight_sf",
                "_global_fc2_weight_sf_swizzled",
                "_global_fc1_alpha",
                "_global_fc2_alpha",
                "_global_fc1_norm_const",
            ):
                setattr(self, name, None)
            torch.cuda.empty_cache()

        # K2 writes one BF16 row per (source token, top-k slot); K3 performs
        # the only supported top-k reduction into the public (T, H) output.
        self.combine_output = _sym_zeros(
            (num_tokens_per_rank, num_topk, hidden),
            problem.fc2_output_dtype,
        )
        self.combine_reduced_output = torch.zeros(
            (num_tokens_per_rank, hidden),
            dtype=problem.fc2_output_dtype,
            device="cuda",
        )
        self.output_tensor = self.combine_reduced_output

        torch.cuda.synchronize()
        self._check_cuda_rng_consistency()

    # ------------------------------------------------------------------
    # Step 2: reference
    # ------------------------------------------------------------------

    def compute_reference(self, *, force: bool = False) -> None:
        if self.misc.skip_ref_check and not force:
            return
        if self._global_activation is None:
            raise RuntimeError("compute_reference requires generate_inputs first.")

        combine_ref_global = compute_megamoe_reference_nvfp4(
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
            fc2_output_dtype=self.problem.fc2_output_dtype,
            gate_up_clamp=self.problem.gate_up_clamp,
        )
        self._combine_output_ref_global = combine_ref_global
        self.combine_output_ref = combine_ref_global[self.rank].contiguous()
        self.combine_reduced_output_ref = flat_topk_bf16_reference(
            self.combine_output_ref
        ).contiguous()
        self.output_tensor_ref = self.combine_reduced_output_ref
        self.rank_local_output_ref = None
        if self.rank_local_combine:
            self.rank_local_output_ref = owner_grouped_bf16_reference(
                self.combine_output_ref,
                self.my_topk_idx,
                world_size=self.world_size,
                num_experts_per_rank=self.problem.num_experts_per_rank,
            ).contiguous()
        # FC1 is column-parallel and FC2 is row-parallel across TP.  The
        # reference is TP-reduced immediately after the actual K3 output in
        # _tp_reduce_actual_output, keeping both NCCL calls after the
        # CuTeDSL/communication kernel work.
        if self._tp_reduce_enabled():
            self._tp_reference_reduced = False

    # ------------------------------------------------------------------
    # Step 3: workspace allocation
    # ------------------------------------------------------------------

    @staticmethod
    def _byte_workspace_view(
        workspace: torch.Tensor,
        byte_offset: int,
        dtype: torch.dtype,
        shape,
    ) -> torch.Tensor:
        nbytes = int(np.prod(shape)) * torch.empty((), dtype=dtype).element_size()
        return workspace.narrow(0, byte_offset, nbytes).view(dtype).view(*shape)

    def _phase_timing_begin(self, name: str, stream=None):
        if not (
            self._phase_timing_enabled
            and self._phase_timing_collect
            and self.comm_backend == "gin"
        ):
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(torch.cuda.current_stream() if stream is None else stream)
        return name, start, end

    def _phase_timing_end(self, handle, stream=None) -> None:
        if handle is None:
            return
        name, start, end = handle
        end.record(torch.cuda.current_stream() if stream is None else stream)
        self._phase_timing_events.setdefault(name, []).append((start, end))

    def _report_phase_timings(self) -> None:
        if not self._phase_timing_enabled or not self._phase_timing_events:
            return
        if self.rank == 0:
            print("---- IBGDA phase critical-path time ----")
        for name, pairs in self._phase_timing_events.items():
            local_us = torch.tensor(
                [start.elapsed_time(end) * 1000.0 for start, end in pairs],
                dtype=torch.float64,
                device="cuda",
            )
            gathered = [torch.empty_like(local_us) for _ in range(self.world_size)]
            self._ep_all_gather(gathered, local_us)
            if self.rank == 0:
                critical_us = torch.stack(gathered).amax(dim=0).cpu().numpy()
                print(
                    f"  {name}: n={len(critical_us)} "
                    f"min={float(np.min(critical_us)):.2f} us "
                    f"p10={float(np.percentile(critical_us, 10)):.2f} us "
                    f"p50={float(np.median(critical_us)):.2f} us "
                    f"p90={float(np.percentile(critical_us, 90)):.2f} us "
                    f"max={float(np.max(critical_us)):.2f} us"
                )

    def _ibgda_streaming_dispatch_begin(self) -> None:
        if not getattr(self, '_gin_receiver_preloaded', False):
            # Load/allocate receiver resources before any persistent ACK wait.
            # Epoch zero and the initialized zero-count control records make
            # this an empty receive: no payload, signal, ACK or GIN RMA.
            assert self._transport.dispatch_epoch == 1
            assert torch.count_nonzero(self._transport.dispatch_control_rx.view(torch.int32)[::32]).item() == 0
            prewarm_args = list(self._transport.dispatch_ring_receiver_runtime_args)
            prewarm_args[-2] = 0
            prewarm_args[-1] = cuda.CUstream(self._transport.dispatch_recv_stream.cuda_stream)
            self._transport.compiled_dispatch_ring_receiver(*prewarm_args)
            self._transport.dispatch_recv_stream.synchronize()
            self._gin_receiver_preloaded = True
        tokens = self.problem.num_tokens_per_rank
        own = self.rank
        activation_by_rank = self._transport.activation_rx.view(
            self.world_size, tokens, *self.my_activation.shape[1:]
        )
        sf_by_rank = self._transport.sf_rx.view(
            self.world_size, tokens, *self.my_activation_sf.shape[1:]
        )
        weight_by_rank = self._transport.topk_weight_rx.view(
            self.world_size, tokens, self.problem.num_topk
        )
        phase = self._phase_timing_begin("dispatch.streaming_ring")
        self._transport.dispatch_epoch += 1
        launch_stream = torch.cuda.current_stream()
        route_phase = self._phase_timing_begin("route.dispatch_builder")
        route_args = list(self._transport.dispatch_route_builder_runtime_args)
        route_args[-2] = self._transport.dispatch_epoch
        route_ready = torch.cuda.Event()
        route_ready.record(launch_stream)
        self._transport.route_stream.wait_event(route_ready)
        route_args[-1] = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.compiled_dispatch_route_builder(*route_args)
        route_done = torch.cuda.Event()
        route_done.record(self._transport.route_stream)
        launch_stream.wait_event(route_done)
        self._phase_timing_end(route_phase)
        activation_by_rank[own].copy_(self.my_activation)
        sf_by_rank[own].copy_(self.my_activation_sf)
        weight_by_rank[own].copy_(self.my_topk_weights)
        self._transport.dispatch_ready[0:1].fill_(
            self._transport.dispatch_epoch
        )
        self._transport.dispatch_ready[
            1 + own * tokens:1 + (own + 1) * tokens
        ].fill_(self._transport.dispatch_epoch)

        sender_args = list(self._transport.dispatch_ring_sender_runtime_args)
        ack_args = list(self._transport.dispatch_ring_ack_runtime_args)
        receiver_args = list(self._transport.dispatch_ring_receiver_runtime_args)
        sender_args[-2] = self._transport.dispatch_epoch
        ack_args[-2] = self._transport.dispatch_epoch
        receiver_args[-2] = self._transport.dispatch_epoch
        ready = torch.cuda.Event()
        ready.record(launch_stream)
        for stream in (
            self._transport.dispatch_send_stream,
            self._transport.dispatch_recv_stream,
        ):
            stream.wait_event(ready)
        plan_stream = self._transport.owner_plan_stream
        plan_stream.wait_event(ready)
        with torch.cuda.stream(plan_stream):
            count_push_args = list(
                self._transport.owner_plan_count_push_runtime_args
            )
            route_push_args = list(
                self._transport.owner_plan_route_push_runtime_args
            )
            count_push_args[-2] = self._transport.dispatch_epoch
            route_push_args[-2] = self._transport.dispatch_epoch
            count_push_args[-1] = cuda.CUstream(plan_stream.cuda_stream)
            route_push_args[-1] = cuda.CUstream(plan_stream.cuda_stream)
            self._transport.compiled_owner_plan_count_push(*count_push_args)
            self._transport.owner_plan_count_push_done = torch.cuda.Event()
            self._transport.owner_plan_count_push_done.record(plan_stream)
            # Route blocks and activation payloads use independent TX streams;
            # owner metadata joins both at source/expert granularity.
            self._transport.compiled_owner_plan_route_push(*route_push_args)
            self._transport.owner_plan_route_push_done = torch.cuda.Event()
            self._transport.owner_plan_route_push_done.record(plan_stream)
        # Give the IBGDA producer a head start before enqueuing persistent
        # polling CTAs.  Launching the receiver first can occupy communication
        # progress resources while every CTA is waiting on its first signal.
        with torch.cuda.stream(self._transport.dispatch_send_stream):
            sender_phase = self._phase_timing_begin(
                "dispatch.tx_sender",
                stream=self._transport.dispatch_send_stream,
            )
            self._transport.compiled_dispatch_ring_sender(*sender_args)
            self._phase_timing_end(
                sender_phase,
                stream=self._transport.dispatch_send_stream,
            )
            # Publish TX-grid completion before joining terminal credits.
            # Combine may reuse the TX partition as soon as dispatch has
            # submitted all payloads; the ACK waiter remains ordered on this
            # stream and independently protects the next epoch's slot reuse.
            self._transport.dispatch_send_done = torch.cuda.Event()
            self._transport.dispatch_send_done.record(
                self._transport.dispatch_send_stream
            )
            self._transport.compiled_dispatch_ring_ack_waiter(*ack_args)
            self._transport.dispatch_credit_done = torch.cuda.Event()
            self._transport.dispatch_credit_done.record(
                self._transport.dispatch_send_stream
            )
        with torch.cuda.stream(self._transport.dispatch_recv_stream):
            receiver_phase = self._phase_timing_begin(
                "dispatch.rx_receiver",
                stream=self._transport.dispatch_recv_stream,
            )
            self._transport.compiled_dispatch_ring_receiver(*receiver_args)
            self._phase_timing_end(
                receiver_phase,
                stream=self._transport.dispatch_recv_stream,
            )
            self._transport.dispatch_recv_done = torch.cuda.Event()
            self._transport.dispatch_recv_done.record(
                self._transport.dispatch_recv_stream
            )
        # Record dispatch completion on the receiver stream itself.  Ending
        # this phase after K1/K2 joined would fold compute overlap into the
        # reported dispatch time.
        self._phase_timing_end(
            phase, stream=self._transport.dispatch_recv_stream
        )
        self._transport.dispatch_ring_phase = None

    def _ibgda_streaming_dispatch_end(self) -> None:
        current = torch.cuda.current_stream()
        current.wait_event(self._transport.dispatch_recv_done)
        current.wait_event(self._transport.dispatch_send_done)

    def _wait_for_previous_streaming_replay(self) -> None:
        """Order replay-local resets after all prior transport consumers.

        This is the production dependency: terminal credits protect cyclic
        slots and work queues, while receive completion protects the local
        ready/control records.  Host synchronization and rank barriers remain
        a debug/stress aid, not part of the performance path.
        """
        current = torch.cuda.current_stream()
        for event in (
            self._transport.dispatch_credit_done,
            self._transport.dispatch_recv_done,
            self._transport.ring_credit_done,
            self._transport.ring_recv_done,
            self._transport.owner_plan_route_push_done,
            self._transport.owner_metadata_done,
        ):
            if event is not None:
                current.wait_event(event)

    def _ibgda_build_runtime_owner_route_plan(self) -> None:
        """Install this replay's K1 and combine metadata entirely on GPU."""
        phase = self._phase_timing_begin("route.owner_control_builder")
        current = torch.cuda.current_stream()
        if self._transport.ring_credit_done is not None:
            # ring_send_counts is also the previous epoch's ACK snapshot.
            self._transport.route_stream.wait_event(
                self._transport.ring_credit_done
            )
        self._transport.route_stream.wait_event(
            self._transport.owner_plan_count_push_done
        )
        count_args = list(self._transport.owner_count_builder_runtime_args)
        count_args[-2] = self._transport.dispatch_epoch
        count_args[-1] = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.compiled_owner_count_builder(*count_args)
        count_done = torch.cuda.Event()
        count_done.record(self._transport.route_stream)
        current.wait_event(count_done)
        # Metadata is produced incrementally after the compact count record.
        # K1 and combine TX consume per-source/per-expert ready epochs; the
        # complete metadata event is not an admission barrier.
        self._transport.owner_metadata_stream.wait_event(count_done)
        metadata_args = list(
            self._transport.owner_metadata_builder_runtime_args
        )
        metadata_args[-2] = self._transport.dispatch_epoch
        metadata_args[-1] = cuda.CUstream(
            self._transport.owner_metadata_stream.cuda_stream
        )
        self._transport.compiled_owner_metadata_builder(*metadata_args)
        self._transport.owner_metadata_done = torch.cuda.Event()
        self._transport.owner_metadata_done.record(
            self._transport.owner_metadata_stream
        )
        self._phase_timing_end(phase)

    def _ibgda_dispatch_inputs(self) -> None:
        self._ibgda_streaming_dispatch_begin()

    def _ibgda_streaming_combine_begin(self) -> None:
        """Start per-channel combine RX/TX without a full-plan barrier."""
        self._transport.ring_done = None
        current = torch.cuda.current_stream()
        self._transport.ring_epoch += 1
        if self._transport.ring_epoch != self._transport.dispatch_epoch:
            raise RuntimeError("dispatch/combine TX handoff epochs diverged")
        sender_args = list(self._transport.ring_sender_runtime_args)
        ack_args = list(self._transport.ring_ack_runtime_args)
        receiver_args = list(self._transport.ring_receiver_runtime_args)
        sender_args[-2] = self._transport.ring_epoch
        ack_args[-2] = self._transport.ring_epoch
        receiver_args[-2] = self._transport.ring_epoch
        launch_ready = torch.cuda.Event()
        launch_ready.record(current)
        self._transport.ring_stream.wait_event(launch_ready)
        self._transport.ring_recv_stream.wait_event(launch_ready)
        # Admission is exclusively the device-side per-peer/channel epoch
        # handoff. Terminal ACK/credit recovery remains independently ordered
        # and protects cyclic slot reuse across replays.
        self._transport.ring_phase = self._phase_timing_begin(
            "combine.streaming_ring"
        )
        with torch.cuda.stream(self._transport.ring_recv_stream):
            receiver_phase = self._phase_timing_begin(
                "combine.rx_receiver",
                stream=self._transport.ring_recv_stream,
            )
            self._transport.compiled_ring_receiver(*receiver_args)
            self._phase_timing_end(
                receiver_phase,
                stream=self._transport.ring_recv_stream,
            )
            self._transport.ring_recv_done = torch.cuda.Event()
            self._transport.ring_recv_done.record(self._transport.ring_recv_stream)
        with torch.cuda.stream(self._transport.ring_stream):
            sender_phase = self._phase_timing_begin(
                "combine.tx_sender",
                stream=self._transport.ring_stream,
            )
            self._transport.compiled_ring_sender(*sender_args)
            self._phase_timing_end(
                sender_phase,
                stream=self._transport.ring_stream,
            )
            # K3 depends on submission/receive completion, not credit return.
            self._transport.ring_done = torch.cuda.Event()
            self._transport.ring_done.record(self._transport.ring_stream)
            self._transport.compiled_ring_ack_waiter(*ack_args)
            self._transport.ring_credit_done = torch.cuda.Event()
            self._transport.ring_credit_done.record(
                self._transport.ring_stream
            )

    def _ibgda_streaming_combine_end(self) -> None:
        """Join the GPU ring without a CPU synchronize or rank barrier."""
        current = torch.cuda.current_stream()
        current.wait_event(self._transport.ring_done)
        current.wait_event(self._transport.ring_recv_done)
        self._phase_timing_end(self._transport.ring_phase)

    def _validate_dispatch_handoff_epoch(self, *, verbose: bool = False) -> None:
        """Require one successful epoch publication for every peer/channel.

        Requiring the complete matrix to equal the current epoch after the
        independent terminal ACK waiter has joined proves that every
        peer/channel published this replay's handoff.
        """
        handoff = self._transport.dispatch_tx_handoff
        if handoff is None:
            return
        expected = int(self._transport.dispatch_epoch)
        mismatches = torch.nonzero(handoff != expected, as_tuple=False)
        if mismatches.numel():
            entries = [
                (
                    int(peer),
                    int(channel),
                    int(handoff[peer, channel].item()),
                )
                for peer, channel in mismatches[:16].cpu().tolist()
            ]
            raise AssertionError(
                "dispatch TX handoff epoch mismatch: "
                f"expected={expected}, entries={entries}"
            )
        if verbose and self.rank == 0:
            print(
                "Dispatch TX handoff validation PASSED: "
                f"epoch={expected}, peer_channels={handoff.numel()}, "
                "direct epoch publication complete; terminal ACK/credit joined "
                "independently",
                flush=True,
            )
    def allocate_workspaces(self) -> None:
        """Allocate the two opaque byte buffers the kernel partitions internally.

        ``local_workspace`` normally lives in plain CUDA device memory.  The
        GIN path places it in an NCCL-registered GPU allocation so the
        peer-major K2 output staging region has a registered lkey and can be
        used directly as an RMA source; counters / FC1 output remain local
        accesses even though their backing allocation is registered.
        ``shared_workspace`` lives in the NCCL LSA peer-mapped arena so the
        dispatch warps' ``peer_rank_ptr_mapper.map`` writes can reach peer ranks.
        ``(symmetric_base, peer_offsets_list)`` is computed once from the
        P2P arena base address -- the allocator preserves the same
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

        if self.comm_backend == "gin":
            gin_bridge.prepare_network_arena(self, local_ws_bytes, _IBGDA_OWNER_PLAN_SLOTS, _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS, _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS)
            self.local_workspace = gin_bridge.sym_zeros(
                (local_ws_bytes,), torch.uint8
            )
        else:
            self.local_workspace = torch.zeros(
                (local_ws_bytes,), dtype=torch.uint8, device="cuda",
            )
        self.shared_workspace = _sym_zeros((shared_ws_bytes,), torch.uint8)
        if self.rank_local_combine:
            self.rank_combine_ready = _sym_zeros(
                (self.problem.num_tokens_per_rank + 1, self.world_size),
                torch.int32,
            )
        if self.comm_backend == "gin":
            tokens = self.problem.num_tokens_per_rank
            max_routes_per_source = tokens * self.problem.num_topk
            # Fixed-capacity GPU control buffers; their logical lengths
            # are rebuilt from the current topk_idx every replay.
            self._transport.max_pair_routes = max_routes_per_source
            self._transport.dispatch_send_token_indices = torch.full(
                (self.world_size, tokens),
                -1,
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.dispatch_send_counts = torch.zeros(
                (self.world_size,), dtype=torch.int32, device="cuda"
            )
            owner_token_shape = (
                self.world_size,
                self.problem.num_experts_per_rank,
            )
            self._transport.dispatch_owner_token_counts = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_token_bases = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_token_cursors = torch.zeros(
                owner_token_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.expert_pool_bases = torch.zeros(
                (self.problem.num_experts_per_rank,),
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.combine_source_expert_bases = torch.zeros(
                (self.world_size, self.problem.num_experts_per_rank),
                dtype=torch.int32,
                device="cuda",
            )
            activation_rx_shape = (
                self.world_size * tokens,
                *self.my_activation.shape[1:],
            )
            self._transport.activation_rx = torch.zeros(
                activation_rx_shape, dtype=torch.uint8, device="cuda"
            ).view(self.my_activation.dtype)
            sf_rx_shape = (
                self.world_size * tokens,
                *self.my_activation_sf.shape[1:],
            )
            self._transport.sf_rx = torch.zeros(
                sf_rx_shape, dtype=torch.uint8, device="cuda"
            ).view(self.my_activation_sf.dtype)
            self._transport.topk_weight_rx = torch.zeros(
                (self.world_size * tokens, self.problem.num_topk),
                dtype=torch.float32,
                device="cuda",
            )
            # Each source builds one compact expert-grouped record per owner.
            # Aligned 256-int64 payload blocks use the same nonblocking
            # payload/fence/epoch protocol as the data ring. The count record
            # has an independent commit so stable extents install first.
            max_route_blocks = (
                max_routes_per_source
                + _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS
                - 1
            ) // _IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS
            self._transport.owner_plan_route_capacity = (
                max_route_blocks
                + self.problem.num_experts_per_rank
            ) * _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS
            count_shape = (
                _IBGDA_OWNER_PLAN_SLOTS,
                self.world_size,
                _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
            )
            route_shape = (
                _IBGDA_OWNER_PLAN_SLOTS,
                self.world_size,
                self._transport.owner_plan_route_capacity,
            )
            self._transport.owner_plan_count_rx = gin_bridge.sym_zeros(
                count_shape, torch.int64
            )
            self._transport.owner_plan_count_signal_rx = gin_bridge.sym_zeros(
                (_IBGDA_OWNER_PLAN_SLOTS, self.world_size),
                torch.int64,
            )
            self._transport.owner_plan_count_signal_tx = gin_bridge.sym_zeros(
                (_IBGDA_OWNER_PLAN_SLOTS, self.world_size),
                torch.int64,
            )
            self._transport.owner_plan_count_tx = gin_bridge.sym_zeros(
                count_shape, torch.int64
            )
            self._transport.owner_plan_route_rx = gin_bridge.sym_zeros(
                route_shape, torch.int64
            )
            self._transport.owner_plan_route_signal_rx = gin_bridge.sym_zeros(
                (
                    _IBGDA_OWNER_PLAN_SLOTS,
                    self.world_size,
                    self.problem.num_experts_per_rank,
                ),
                torch.int64,
            )
            self._transport.owner_plan_route_signal_tx = gin_bridge.sym_zeros(
                (
                    _IBGDA_OWNER_PLAN_SLOTS,
                    self.world_size,
                    self.problem.num_experts_per_rank,
                ),
                torch.int64,
            )
            self._transport.owner_plan_route_tx = gin_bridge.sym_zeros(
                route_shape, torch.int64
            )
            plan_shape = (
                self.world_size,
                self.problem.num_experts_per_rank,
            )
            self._transport.owner_plan_route_bases_tx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.dispatch_owner_route_output_bases = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.owner_plan_route_cursors_tx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            self._transport.owner_plan_route_bases_rx = torch.zeros(
                plan_shape, dtype=torch.int32, device="cuda"
            )
            channels = self._transport.dispatch_ring_channels
            slots = self._transport.dispatch_ring_slots
            chunk_tokens = self._kernel.token_comm.ibgda_dispatch_chunk_tokens
            act_bpt = self.my_activation[0].numel() * self.my_activation.element_size()
            sf_bpt = self.my_activation_sf[0].numel() * self.my_activation_sf.element_size()
            weight_bpt = self.my_topk_weights[0].numel() * self.my_topk_weights.element_size()
            dispatch_payload_bytes = chunk_tokens * (
                4 + act_bpt + sf_bpt + weight_bpt
            )
            dispatch_slot_bytes = (
                (dispatch_payload_bytes + 8 + 15) // 16
            ) * 16
            self._transport.dispatch_ring_rx = gin_bridge.sym_zeros(
                (
                    self.world_size,
                    channels,
                    slots,
                    dispatch_slot_bytes,
                ),
                torch.uint8,
            )
            self._transport.dispatch_ring_tx = gin_bridge.sym_zeros(
                (
                    self.world_size,
                    channels,
                    slots,
                    dispatch_slot_bytes,
                ),
                torch.uint8,
            )
            self._transport.dispatch_ring_ack = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.dispatch_ring_ack_tx = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.dispatch_control_rx = gin_bridge.sym_zeros(
                (self.world_size, 128), torch.uint8
            )
            self._transport.dispatch_control_tx = gin_bridge.sym_zeros(
                (self.world_size, 128), torch.uint8
            )
            dispatch_tasks = self.world_size * channels
            self._transport.dispatch_send_work_queue = torch.zeros(
                (2 * dispatch_tasks + self.kernel_config.tx_sms + 1,),
                dtype=torch.int32,
                device="cuda",
            )
            self._transport.dispatch_recv_work_queue = torch.zeros(
                (2 * dispatch_tasks + self.kernel_config.rx_sms + 1,),
                dtype=torch.int32,
                device="cuda",
            )
            # Transport epochs start at one and are incremented before launch.
            # Initialize the publication matrix to the same predecessor epoch
            # so each channel can CAS epoch-1 -> epoch exactly once.
            self._transport.dispatch_tx_handoff = torch.full(
                (self.world_size, channels),
                1,
                dtype=torch.int32,
                device="cuda",
            )
            ready_words = self._kernel.token_comm.ibgda_dispatch_ready_words
            ready_off = self._kernel._local_offsets["ibgda_sf_staging"]
            self._transport.dispatch_ready = self.local_workspace[
                ready_off:ready_off + ready_words * 4
            ].view(torch.int32)
            plan_ready_words = (
                self.world_size * self.problem.num_experts_per_rank
            )
            plan_ready_off = ready_off + ready_words * 4
            self._transport.owner_plan_ready = self.local_workspace[
                plan_ready_off:plan_ready_off + plan_ready_words * 4
            ].view(torch.int32)
            # The bounded ring slots below are the only combine payload
            # storage; no full routed-output replica is allocated.
            channels = self._transport.combine_ring_channels
            slots = self._transport.combine_ring_slots
            chunk_rows = self.kernel_config.combine_chunk_rows
            if chunk_rows <= 0:
                raise ValueError(
                    "combine_chunk_rows must be positive"
                )
            payload_bytes = (
                self.problem.hidden * self.combine_output.element_size()
            )
            slot_bytes = (
                (chunk_rows * (4 + payload_bytes) + 8 + 15) // 16
            ) * 16
            self._transport.ring_rx = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, slot_bytes),
                torch.uint8,
            )
            self._transport.ring_tx = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, slot_bytes),
                torch.uint8,
            )
            self._transport.ring_ack = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            self._transport.ring_ack_tx = gin_bridge.sym_zeros(
                (self.world_size, channels, slots, 128), torch.uint8
            )
            shape = (self.world_size, self._transport.max_pair_routes)
            self._transport.ring_send_rows = torch.zeros(
                shape, dtype=torch.int64, device="cuda"
            )
            self._transport.ring_send_ready_slots = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_ready_values = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_output_rows = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.combine_recv_output_rows = torch.zeros(
                shape, dtype=torch.int32, device="cuda"
            )
            self._transport.ring_send_counts = torch.zeros(
                (self.world_size,), dtype=torch.int32, device="cuda"
            )
            self.symmetric_base = int(self.shared_workspace.data_ptr())
            self.peer_offsets_list = tuple(0 for _ in range(self.world_size))
        else:
            self.symmetric_base, self.peer_offsets_list = _compute_peer_offsets(
                self.shared_workspace, self.world_size,
            )
            gin_bridge.configure(self)

    def _compile_ibgda_gpu_push_kernels(self, cute_module) -> None:
        gin_bridge.configure(self)
        import cuda.bindings.driver as cuda
        from cutlass.cute.runtime import from_dlpack
        from moe_sm120_nvfp4_split.transport.gin_gpu_comm import (
            GinGpuOwnerPlanPush,
        )
        from moe_sm120_nvfp4_split.transport.gin_gpu_ring import (
            GinGpuRingCombineReceiver,
            GinGpuRingCombineSender,
            GinGpuRingDispatchReceiver,
            GinGpuRingDispatchSender,
        )
        from moe_sm120_nvfp4_split.transport.gpu_runtime_route import (
            GpuChunkedOwnerRoutePlanBuilder,
            GpuDispatchRouteBuilder,
        )

        def comm_to_cute(tensor):
            return from_dlpack(wrap_dlpack(tensor)).mark_layout_dynamic()

        # communication progress must be established before the long-lived
        # receiver polling CTA is admitted.  A high-priority producer
        # stream prevents the scheduler from choosing the waiter first.
        if self._transport.dispatch_send_stream is None:
            self._transport.dispatch_send_stream = torch.cuda.Stream(
                priority=-1
            )
            self._transport.dispatch_recv_stream = torch.cuda.Stream(
                priority=-1
            )
        if self._transport.owner_plan_stream is None:
            self._transport.owner_plan_stream = torch.cuda.Stream(priority=-1)
        if self._transport.owner_metadata_stream is None:
            self._transport.owner_metadata_stream = torch.cuda.Stream(priority=0)
        chunk_tokens = self._kernel.token_comm.ibgda_dispatch_chunk_tokens
        act_bpt = (
            self.my_activation[0].numel()
            * self.my_activation.element_size()
        )
        sf_bpt = (
            self.my_activation_sf[0].numel()
            * self.my_activation_sf.element_size()
        )
        weight_bpt = (
            self.my_topk_weights[0].numel()
            * self.my_topk_weights.element_size()
        )
        # Ownership is a protocol property, not an SM-count heuristic: every
        # (peer, channel) has one lifetime owner CTA.  When an explicit tuning
        # override selects four channels, two lightweight owner CTAs may share
        # each of the 16 direction-specific communication SMs.
        dispatch_tasks = (
            self.world_size * self._transport.dispatch_ring_channels
        )
        dispatch_tx_workers = dispatch_tasks
        dispatch_rx_workers = dispatch_tasks
        self._transport.dispatch_ring_sender_kernel = (
            GinGpuRingDispatchSender(
                self.world_size,
                self.problem.num_tokens_per_rank,
                act_bpt,
                sf_bpt,
                weight_bpt,
                chunk_tokens=chunk_tokens,
                channels=self._transport.dispatch_ring_channels,
                slots=self._transport.dispatch_ring_slots,
                local_handoff_windows=(
                    self.kernel_config.dispatch_local_handoff_windows
                ),
                remote_handoff_windows=(
                    self.kernel_config.dispatch_remote_handoff_windows
                ),
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
                worker_ctas=dispatch_tx_workers,
            )
        )
        self._transport.dispatch_ring_receiver_kernel = (
            GinGpuRingDispatchReceiver(
                self._transport.dispatch_ring_sender_kernel,
                worker_ctas=dispatch_rx_workers,
            )
        )
        self._transport.dispatch_epoch = 1
        dispatch_sender_args = (
            comm_to_cute(self._transport.dispatch_ring_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_tx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(self._transport.dispatch_control_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_control_tx.reshape(-1)),
            comm_to_cute(self.my_activation.view(torch.uint8).reshape(-1)),
            comm_to_cute(
                self.my_activation_sf.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self.my_topk_weights.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_send_token_indices.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_send_counts),
            comm_to_cute(self._transport.dispatch_tx_handoff.reshape(-1)),
            comm_to_cute(self._transport.dispatch_send_work_queue),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_send_stream.cuda_stream),
        )
        dispatch_receiver_args = (
            comm_to_cute(self._transport.dispatch_ring_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(
                self._transport.dispatch_ring_ack_tx.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_control_rx.reshape(-1)),
            comm_to_cute(self._transport.dispatch_recv_work_queue),
            comm_to_cute(
                self._transport.activation_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.sf_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(
                self._transport.topk_weight_rx.view(torch.uint8).reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_ready),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_recv_stream.cuda_stream),
        )
        dispatch_ack_args = (
            comm_to_cute(self._transport.dispatch_ring_ack.reshape(-1)),
            comm_to_cute(self._transport.dispatch_send_counts),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(self._transport.dispatch_send_stream.cuda_stream),
        )
        self._transport.owner_plan_push_kernel = (
            GinGpuOwnerPlanPush(
                self.world_size,
                self.problem.num_experts_per_rank,
                self._transport.owner_plan_route_capacity,
                block_elements=(
                    _IBGDA_OWNER_PLAN_BLOCK_ELEMENTS
                ),
                plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
                completion_credit=(
                    self.problem.num_tokens_per_rank <= 128
                ),
            )
        )
        plan_stream = self._transport.owner_plan_stream
        owner_plan_count_push_args = (
            comm_to_cute(
                self._transport.owner_plan_count_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_signal_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(plan_stream.cuda_stream),
        )
        owner_plan_route_push_args = (
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_signal_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_bases_tx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            cuda.CUstream(plan_stream.cuda_stream),
        )

        if self._transport.ring_stream is None:
            self._transport.ring_stream = torch.cuda.Stream(priority=0)
            self._transport.ring_recv_stream = torch.cuda.Stream(priority=0)
        payload_bytes = (
            self.problem.hidden * self.combine_output.element_size()
        )
        chunk_rows = self.kernel_config.combine_chunk_rows
        # Keep the worker grid resident in its direction-specific Green
        # Context.  Workers own disjoint peer/channel stripes, avoiding both
        # head-of-line blocking and an oversubscribed polling grid.
        combine_tasks = (
            self.world_size * self._transport.combine_ring_channels
        )
        combine_tx_workers = combine_tasks
        combine_rx_workers = combine_tasks
        self._transport.ring_sender_kernel = GinGpuRingCombineSender(
            self.world_size,
            payload_bytes,
            self._transport.max_pair_routes,
            chunk_rows=chunk_rows,
            channels=self._transport.combine_ring_channels,
            slots=self._transport.combine_ring_slots,
            hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            worker_ctas=combine_tx_workers,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            blocking_put=(
                self.kernel_config.combine_blocking_put
                or self.problem.num_tokens_per_rank <= 128
            ),
            progress_windows=(self.problem.num_tokens_per_rank <= 128),
        )
        self._transport.ring_receiver_kernel = GinGpuRingCombineReceiver(
            self.world_size,
            payload_bytes,
            self._transport.max_pair_routes,
            chunk_rows=chunk_rows,
            channels=self._transport.combine_ring_channels,
            slots=self._transport.combine_ring_slots,
            hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            worker_ctas=combine_rx_workers,
            num_experts_per_rank=self.problem.num_experts_per_rank,
            count_block_elements=_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
            plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
            progress_windows=(self.problem.num_tokens_per_rank <= 128),
        )
        max_chunks_per_peer = (
            self._transport.max_pair_routes + chunk_rows - 1
        ) // chunk_rows
        self._transport.combine_ready_queue_done = torch.zeros(
            (self.world_size * max_chunks_per_peer,),
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_ready_queue_probe = torch.full(
            (
                self.world_size
                * self._transport.combine_ring_channels
                * 4,
            ),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_ready_queue_selected = torch.full(
            (
                self.world_size
                * self._transport.combine_ring_channels,
            ),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        combine_tasks = (
            self.world_size * self._transport.combine_ring_channels
        )
        self._transport.combine_send_work_queue = torch.zeros(
            (
                2 * combine_tasks
                + self._transport.ring_sender_kernel.worker_ctas
                + 2,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        self._transport.combine_recv_work_queue = torch.zeros(
            (
                combine_tasks
                + self._transport.ring_receiver_kernel.worker_ctas
                + 1,
            ),
            dtype=torch.int32,
            device="cuda",
        )
        kernel = self._kernel
        fc2_spec = kernel._local_region_by_name["fc2_output_workspace"]
        fc2_done_spec = kernel._local_region_by_name["fc2_done_counter"]
        fc2_ws_full = self.local_workspace[
            kernel._local_offsets["fc2_output_workspace"]:
            kernel._local_offsets["fc2_output_workspace"] + fc2_spec.nbytes
        ]
        # The full worst-case pool can exceed the signed 32-bit memref
        # length accepted by the current CuTe runtime.  Ring rows address
        # a dynamically populated prefix.  Keep the descriptor below
        # INT32_MAX bytes; device indexing is pointer based and the
        # backing allocation remains the kernel's full worst-case pool.
        row_bytes = self.problem.hidden * self.combine_output.element_size()
        descriptor_row_limit = max(1, (2**31 - 1) // row_bytes)
        used_pool_rows = min(
            self._kernel.pool_token_capacity,
            descriptor_row_limit,
        )
        fc2_used_bytes = (
            used_pool_rows
            * self.problem.hidden
            * self.combine_output.element_size()
        )
        fc2_ws = fc2_ws_full.narrow(0, 0, fc2_used_bytes)
        fc2_done = self.local_workspace[
            kernel._local_offsets["fc2_done_counter"]:
            kernel._local_offsets["fc2_done_counter"] + fc2_done_spec.nbytes
        ].view(torch.int32)
        direct_map_byte_offset = (
            kernel._local_offsets["ibgda_sf_staging"]
            + kernel.ibgda_direct_stage_map_offset_i32 * 4
        )
        self._transport.combine_pool_to_stage_rows = self.local_workspace[
            direct_map_byte_offset:
            direct_map_byte_offset + kernel.pool_token_capacity * 4
        ].view(torch.int32)
        self._transport.ring_epoch = 1
        ring_sender_args = (
            comm_to_cute(self._transport.ring_rx.reshape(-1)),
            comm_to_cute(self._transport.ring_tx.reshape(-1)),
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(fc2_ws),
            comm_to_cute(
                self._transport.ring_send_output_rows.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_slots.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_values.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_counts),
            # Keep the host memref in BF16. Flattening the full output to
            # bytes can exceed CuTe's signed 32-bit layout-size limit.
            # The ring kernel performs its own vector recasts and Int64 indexing.
            comm_to_cute(self.combine_output.reshape(-1)),
            comm_to_cute(fc2_done),
            comm_to_cute(self._transport.combine_ready_queue_done),
            comm_to_cute(self._transport.combine_ready_queue_probe),
            comm_to_cute(self._transport.combine_ready_queue_selected),
            comm_to_cute(self._transport.dispatch_tx_handoff.reshape(-1)),
            comm_to_cute(self._transport.combine_send_work_queue),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_stream.cuda_stream),
        )
        ring_ack_args = (
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(self._transport.ring_send_counts),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_stream.cuda_stream),
        )
        ring_receiver_args = (
            comm_to_cute(self._transport.ring_rx.reshape(-1)),
            comm_to_cute(self._transport.ring_ack.reshape(-1)),
            comm_to_cute(self._transport.ring_ack_tx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_count_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.combine_recv_output_rows.reshape(-1)
            ),
            comm_to_cute(self.combine_output.reshape(-1)),
            comm_to_cute(self._transport.combine_recv_work_queue),
            self.rank,
            self._transport.ring_epoch,
            cuda.CUstream(self._transport.ring_recv_stream.cuda_stream),
        )


        kernel = self._kernel
        meta = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["src_token_topk_idx"],
            torch.int32,
            kernel._shared_region_by_name["src_token_topk_idx"].shape,
        )
        counts = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["expert_recv_count"],
            torch.int64,
            kernel._shared_region_by_name["expert_recv_count"].shape,
        )
        sums = self._byte_workspace_view(
            self.shared_workspace,
            kernel._shared_offsets["expert_recv_count_sum"],
            torch.int64,
            kernel._shared_region_by_name["expert_recv_count_sum"].shape,
        )
        if self._transport.route_stream is None:
            self._transport.route_stream = torch.cuda.Stream(priority=0)
        route_stream = cuda.CUstream(self._transport.route_stream.cuda_stream)
        self._transport.dispatch_route_builder_kernel = GpuDispatchRouteBuilder(
            self.world_size,
            self.problem.num_tokens_per_rank,
            self.problem.num_topk,
            self.problem.num_experts_per_rank,
            owner_route_chunk_elements=_IBGDA_OWNER_PLAN_PAYLOAD_ELEMENTS,
            owner_route_capacity=self._transport.owner_plan_route_capacity,
            owner_plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
        )
        dispatch_route_builder_args = (
            comm_to_cute(self.my_topk_idx.reshape(-1)),
            comm_to_cute(
                self._transport.dispatch_send_token_indices.reshape(-1)
            ),
            comm_to_cute(self._transport.dispatch_send_counts),
            comm_to_cute(self._transport.owner_plan_count_tx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_route_bases_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_route_output_bases.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_cursors_tx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_counts.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_bases.reshape(-1)
            ),
            comm_to_cute(
                self._transport.dispatch_owner_token_cursors.reshape(-1)
            ),
            comm_to_cute(self._transport.owner_plan_route_tx.reshape(-1)),
            comm_to_cute(
                self._transport.combine_recv_output_rows.reshape(-1)
            ),
            self._transport.dispatch_epoch,
            route_stream,
        )
        self._transport.owner_route_builder_kernel = (
            GpuChunkedOwnerRoutePlanBuilder(
                self.world_size,
                self.problem.num_tokens_per_rank,
                self.problem.num_topk,
                self.problem.num_experts_per_rank,
                kernel.token_padding_block,
                self._kernel_k2.k2_token_tile_tokens,
                (
                    self._kernel_k2.token_comm
                    .fc2_publishes_per_token_cluster_tile
                ),
                self._transport.k1_sm_count,
                self._transport.owner_plan_route_capacity,
                route_block_elements=_IBGDA_OWNER_PLAN_BLOCK_ELEMENTS,
                plan_slots=_IBGDA_OWNER_PLAN_SLOTS,
                hybrid_force_ibgda_mask=self.hybrid_force_ibgda_mask,
            )
        )
        owner_count_builder_args = (
            comm_to_cute(self._transport.owner_plan_count_rx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_count_signal_rx.reshape(-1)
            ),
            comm_to_cute(counts.reshape(-1)),
            comm_to_cute(sums.reshape(-1)),
            comm_to_cute(self._transport.expert_pool_bases),
            comm_to_cute(
                self._transport.combine_source_expert_bases.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_counts),
            comm_to_cute(
                self._transport.owner_plan_route_bases_rx.reshape(-1)
            ),
            self.rank,
            self._transport.dispatch_epoch,
            route_stream,
        )
        metadata_stream = cuda.CUstream(
            self._transport.owner_metadata_stream.cuda_stream
        )
        owner_metadata_builder_args = (
            comm_to_cute(self._transport.owner_plan_route_rx.reshape(-1)),
            comm_to_cute(
                self._transport.owner_plan_route_signal_rx.reshape(-1)
            ),
            comm_to_cute(
                self._transport.owner_plan_route_bases_rx.reshape(-1)
            ),
            comm_to_cute(meta.reshape(-1)),
            comm_to_cute(counts.reshape(-1)),
            comm_to_cute(sums.reshape(-1)),
            comm_to_cute(self._transport.expert_pool_bases),
            comm_to_cute(
                self._transport.combine_source_expert_bases.reshape(-1)
            ),
            comm_to_cute(self._transport.ring_send_rows.reshape(-1)),
            comm_to_cute(
                self._transport.ring_send_output_rows.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_slots.reshape(-1)
            ),
            comm_to_cute(
                self._transport.ring_send_ready_values.reshape(-1)
            ),
            comm_to_cute(
                self._transport.combine_pool_to_stage_rows.reshape(-1)
            ),
            comm_to_cute(self._transport.owner_plan_ready),
            self._transport.dispatch_epoch,
            metadata_stream,
        )
        link_options = f"--link-libraries={gin_bridge.device_bitcode()}"
        self._transport.compiled_owner_plan_count_push = cute_module.compile(
            self._transport.owner_plan_push_kernel.count_launcher,
            *owner_plan_count_push_args,
            options=link_options,
        )
        self._transport.compiled_owner_plan_route_push = cute_module.compile(
            self._transport.owner_plan_push_kernel.route_launcher,
            *owner_plan_route_push_args,
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_sender = cute_module.compile(
            self._transport.dispatch_ring_sender_kernel.launcher,
            *dispatch_sender_args,
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_ack_waiter = cute_module.compile(
            self._transport.dispatch_ring_sender_kernel.ack_launcher,
            *dispatch_ack_args,
            options=link_options,
        )
        self._transport.compiled_dispatch_ring_receiver = cute_module.compile(
            self._transport.dispatch_ring_receiver_kernel.launcher,
            *dispatch_receiver_args,
            options=link_options,
        )
        self._transport.compiled_ring_sender = cute_module.compile(
            self._transport.ring_sender_kernel.launcher,
            *ring_sender_args,
            options=link_options,
        )
        self._transport.compiled_ring_ack_waiter = cute_module.compile(
            self._transport.ring_sender_kernel.ack_launcher,
            *ring_ack_args,
            options=link_options,
        )
        self._transport.compiled_ring_receiver = cute_module.compile(
            self._transport.ring_receiver_kernel.launcher,
            *ring_receiver_args,
            options=link_options,
        )
        self._transport.compiled_dispatch_route_builder = cute_module.compile(
            self._transport.dispatch_route_builder_kernel.launcher,
            *dispatch_route_builder_args,
        )
        self._transport.compiled_owner_count_builder = cute_module.compile(
            self._transport.owner_route_builder_kernel.count_launcher,
            *owner_count_builder_args,
            options=link_options,
        )
        self._transport.compiled_owner_metadata_builder = cute_module.compile(
            self._transport.owner_route_builder_kernel.metadata_launcher,
            *owner_metadata_builder_args,
            options=link_options,
        )
        self._transport.dispatch_ring_sender_runtime_args = dispatch_sender_args
        self._transport.dispatch_ring_ack_runtime_args = dispatch_ack_args
        self._transport.dispatch_ring_receiver_runtime_args = dispatch_receiver_args
        self._transport.ring_sender_runtime_args = ring_sender_args
        self._transport.ring_ack_runtime_args = ring_ack_args
        self._transport.ring_receiver_runtime_args = ring_receiver_args
        self._transport.dispatch_route_builder_runtime_args = (
            dispatch_route_builder_args
        )
        self._transport.owner_plan_count_push_runtime_args = (
            owner_plan_count_push_args
        )
        self._transport.owner_plan_route_push_runtime_args = (
            owner_plan_route_push_args
        )
        self._transport.owner_count_builder_runtime_args = (
            owner_count_builder_args
        )
        self._transport.owner_metadata_builder_runtime_args = (
            owner_metadata_builder_args
        )
        compiled_kernels = (
            self._transport.compiled_owner_plan_count_push,
            self._transport.compiled_owner_plan_route_push,
            self._transport.compiled_dispatch_ring_sender,
            self._transport.compiled_dispatch_ring_ack_waiter,
            self._transport.compiled_dispatch_ring_receiver,
            self._transport.compiled_ring_sender,
            self._transport.compiled_ring_ack_waiter,
            self._transport.compiled_ring_receiver,
        )
        # Materialize every communication function before starting persistent
        # receivers. A first-use module load may synchronize the context and
        # deadlock behind a receiver that is waiting for that very sender.
        # CUDA_MODULE_LOADING=EAGER alone is insufficient for CuTe CUlibrary.
        def checked(result):
            if int(result[0]) != 0:
                raise RuntimeError(f'Communication function preload failed: {result}')
            return result[1] if len(result) == 2 else result[1:]
        loaded = set()
        for name, compiled in vars(self._transport).items():
            if not name.startswith('compiled_') or compiled is None:
                continue
            library = int(compiled.library)
            if library in loaded:
                continue
            loaded.add(library)
            module = checked(cuda.cuLibraryGetModule(cuda.CUlibrary(library)))
            count = checked(cuda.cuModuleGetFunctionCount(module))
            for function in checked(cuda.cuModuleEnumerateFunctions(count, module)):
                checked(cuda.cuFuncLoad(function))
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
            if event_name.startswith("kernel_cutlass_topk_reduce_"):
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
            self._ep_all_gather(gathered_times, local_times)
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

    def _reset_local_counters(self) -> None:
        """Zero only the local flag/counter regions between back-to-back
        launches.  The caller must first drain the preceding replay because
        these regions are also consumed by persistent dispatch/combine CTAs.
        The kernel does not reset its accumulating local counters
        (``fc1_done_counter`` / ``fc2_done_counter`` spin thresholds,
        ``load_balance_counter`` work-stealing cursor, ``expert_send_count`` /
        ``l1_arrival_count`` / ``atomic_counter`` dispatch write-cursors, plus
        the phase-flip ``grid_sync_counter``), so a re-launch with stale values
        deadlocks or writes out of bounds. ``nvlink_barrier_counter`` and its
        peer-written signal slots are generation state and survive replay. Data
        buffers (token / act / sf / fc1_output / fc2_output_workspace / topk
        weights) are fully overwritten each launch and are deliberately NOT
        zeroed -- a large memset between launches desyncs the ranks and pollutes
        the steady-state entry-barrier timing. Peer-written control regions
        below are safe to reset only after the preceding replay has drained;
        K1's device-side entry barrier joins every rank after this launch-
        ordered reset before any peer can publish the next replay's counts.
        A memset is not a ``kernel_cutlass*`` event, so it stays out of that
        timing."""
        kernel = self._kernel
        if kernel is None or self.local_workspace is None:
            return
        for name in (
            "l1_arrival_count",
            "expert_send_count",
            "grid_sync_counter",
            "fc1_done_counter",
            "fc2_done_counter",
            "atomic_counter",
            "load_balance_counter",
            "k2_ready_queue_desc",
            "k2_ready_queue_ready",
            "k2_ready_queue_state",
            "dispatch_rank_cache_state",
            "rank_combine_tile_done",
        ):
            if name in kernel._local_offsets:
                off = kernel._local_offsets[name]
                nbytes = kernel._local_region_by_name[name].nbytes
                self.local_workspace[off:off + nbytes].zero_()
        if "rank_combine_route_map" in kernel._local_offsets:
            off = kernel._local_offsets["rank_combine_route_map"]
            nbytes = kernel._local_region_by_name[
                "rank_combine_route_map"
            ].nbytes
            self.local_workspace[off:off + nbytes].view(torch.int32).fill_(-1)
        if self.rank_combine_ready is not None:
            self._rank_combine_epoch += 1
            self.rank_combine_ready[-1, 0].fill_(
                self._rank_combine_epoch
            )
        # K2 uses the high 32 bits of expert_recv_count_sum as K1's
        # count-publication marker. The symmetric workspace survives CUDA
        # Graph replays, so leaving this marker intact lets K2 consume stale
        # expert extents before K1 has published the current replay. Reset
        # only this small control region; K1 still overwrites the data pools.
        if (
            self._kernel_k2_finalizer is None
            and self.shared_workspace is not None
            and "expert_recv_count_sum" in kernel._shared_offsets
        ):
            for name in (
                "expert_recv_count",
                "expert_recv_count_sum",
                "rank_combine_route_count",
            ):
                if name in kernel._shared_offsets:
                    off = kernel._shared_offsets[name]
                    nbytes = kernel._shared_region_by_name[name].nbytes
                    self.shared_workspace[off:off + nbytes].zero_()
        if self.comm_backend == "gin":
            ready_bytes = (
                self._kernel.token_comm.ibgda_dispatch_ready_words
                + self._kernel.token_comm.ibgda_owner_plan_ready_words
            ) * 4
            ready_off = self._kernel._local_offsets["ibgda_sf_staging"]
            self.local_workspace[
                ready_off:ready_off + ready_bytes
            ].zero_()
            counts = self._byte_workspace_view(
                self.shared_workspace,
                kernel._shared_offsets["expert_recv_count"],
                torch.int64,
                kernel._shared_region_by_name["expert_recv_count"].shape,
            )
            counts.zero_()
            self._transport.dispatch_send_counts.zero_()
            self._transport.dispatch_send_work_queue.zero_()
            self._transport.dispatch_recv_work_queue.zero_()
            self._transport.combine_send_work_queue.zero_()
            self._transport.combine_recv_work_queue.zero_()
            self._transport.dispatch_owner_token_counts.zero_()
            self._transport.dispatch_owner_token_cursors.zero_()
            self._transport.ring_send_counts.zero_()
            self._transport.owner_plan_count_tx.zero_()
            self._transport.owner_plan_route_cursors_tx.zero_()
        if self._split_green_trace is not None:
            self._split_green_trace.zero_()
        if self._transport.combine_ready_queue_done is not None:
            self._transport.combine_ready_queue_done.zero_()
        if self._transport.combine_ready_queue_probe is not None:
            self._transport.combine_ready_queue_probe.fill_(-1)
        if self._transport.combine_ready_queue_selected is not None:
            self._transport.combine_ready_queue_selected.fill_(-1)
        # Every valid pool row is overwritten by this epoch's incremental
        # metadata builder before K2 can consume its expert-ready word.  Do
        # not clear the full direct-stage map between replays: the memset is
        # unnecessary and turns rank-local launch skew into remote wait time
        # at large token counts.

    def _reset_before_launch(self) -> None:
        """Reset state not owned by an in-graph finalization protocol.

        The dispatch cache is intentionally reset on the caller stream.  The
        Green Graph launch records its launch-ready event after that reset,
        so K1 cannot observe stale cache entries; the caller stream also waits
        for graph completion before the next reset.  Capturing the memset in
        the graph did not provide a reliable admission edge after Green
        Context kernel-node rebinding.
        """
        graph_owns_epoch = (
            self.split_launch == "green_graph"
            and (
                self.k2_tail_reclaim
                or self._kernel_k2_finalizer is not None
            )
        )
        if not graph_owns_epoch:
            self._reset_local_counters()

    def _report_split_green_trace(self) -> None:
        """Report K1/K2 envelopes and per-CTA K2 attribution."""
        if self._split_green_trace is None:
            return

        from moe_sm120_nvfp4_split.split_timestamp import (
            FIELD_FIRST_WORK,
            FIELD_FIRST_TILE_ID,
            FIELD_KERNEL_ENTRY,
            FIELD_KERNEL_EXIT,
            FIELD_LAST_WORK,
            FIELD_LAST_TILE_ID,
            FIELD_MAINLOOP_NS,
            FIELD_READY_WAIT_CALLS,
            FIELD_READY_WAIT_NS,
            FIELD_STORE_NS,
            FIELD_TILE_COUNT,
            FIELD_TMA_A_WAIT_CALLS,
            FIELD_TMA_A_WAIT_NS,
            FIELD_TMA_A_TIMED_CALLS,
            FIELD_TMA_B_WAIT_CALLS,
            FIELD_TMA_B_WAIT_NS,
            FIELD_TMA_B_TIMED_CALLS,
            ROLE_K1,
            ROLE_K2,
            K2_TILE_FIELD_BF16_PACK_NS,
            K2_TILE_FIELD_DEQUEUE_BEGIN,
            K2_TILE_FIELD_DEQUEUE_END,
            K2_TILE_FIELD_LDSM_QMMA_NS,
            K2_TILE_FIELD_PEER_STORE_NS,
            K2_TILE_FIELD_PHASE_ADVANCE_NS,
            K2_TILE_FIELD_TILE_BEGIN,
            K2_TILE_FIELD_TILE_END,
            K2_TILE_FIELD_TMA_A_WAIT_NS,
            K2_TILE_FIELD_TMA_B_WAIT_NS,
            K2_TILE_TRACE_CTA_CAPACITY,
            K2_TILE_TRACE_FIELDS,
            K2_TILE_TRACE_TILES_PER_CTA,
            TILE_ID_COORD_BASE,
            TILE_ID_EXPERT_BASE,
            TRACE_BASE_WORDS,
            TRACE_CTA_CAPACITY,
            TRACE_FIELDS,
            TRACE_ROLES,
        )

        gathered = [
            torch.empty_like(self._split_green_trace)
            for _ in range(self.world_size)
        ]
        self._ep_all_gather(gathered, self._split_green_trace)
        if self.rank != 0:
            return

        def _role_metrics(
            trace: torch.Tensor, role: int
        ) -> tuple[int, int, int, int, int]:
            rows = trace[role]
            kernel_valid = (
                (rows[:, FIELD_KERNEL_ENTRY] > 0)
                & (rows[:, FIELD_KERNEL_EXIT] > rows[:, FIELD_KERNEL_ENTRY])
            )
            work_valid = (
                (rows[:, FIELD_FIRST_WORK] > 0)
                & (rows[:, FIELD_LAST_WORK] >= rows[:, FIELD_FIRST_WORK])
            )
            if not bool(kernel_valid.any().item()):
                return (0, 0, 0, 0, 0)
            kernel_rows = rows[kernel_valid]
            start = int(kernel_rows[:, FIELD_KERNEL_ENTRY].min().item())
            end = int(kernel_rows[:, FIELD_KERNEL_EXIT].max().item())
            first = 0
            last = 0
            if bool(work_valid.any().item()):
                work_rows = rows[work_valid]
                first = int(work_rows[:, FIELD_FIRST_WORK].min().item())
                last = int(work_rows[:, FIELD_LAST_WORK].max().item())
            return (start, first, last, end, int(kernel_valid.sum().item()))

        def _us(ns: int) -> float:
            return ns / 1000.0

        def _pct(values: torch.Tensor, q: float) -> float:
            if values.numel() == 0:
                return 0.0
            return float(torch.quantile(values.to(torch.float64), q).item())

        def _decode_tile_id(value: int) -> tuple[int, int, int]:
            expert = value // TILE_ID_EXPERT_BASE
            remainder = value % TILE_ID_EXPERT_BASE
            tile_n = remainder // TILE_ID_COORD_BASE
            tile_m = remainder % TILE_ID_COORD_BASE
            return (expert, tile_n, tile_m)

        def _sampled_wait_estimate(
            rows: torch.Tensor,
            ns_field: int,
            calls_field: int,
            timed_calls_field: int,
        ) -> torch.Tensor:
            raw_ns = rows[:, ns_field].to(torch.float64)
            calls = rows[:, calls_field].to(torch.float64)
            timed_calls = rows[:, timed_calls_field].to(torch.float64)
            scale = torch.where(
                timed_calls > 0,
                calls / torch.clamp(timed_calls, min=1),
                torch.zeros_like(timed_calls),
            )
            return raw_ns * scale

        print("---- split Green Context %globaltimer overlap ----")
        for rank, rank_tensor in enumerate(gathered):
            trace = rank_tensor[:TRACE_BASE_WORDS].reshape(
                TRACE_ROLES, TRACE_CTA_CAPACITY, TRACE_FIELDS
            ).cpu()
            k1_start, k1_first, k1_last, k1_end, k1_ctas = _role_metrics(
                trace, ROLE_K1
            )
            k2_start, k2_first, k2_last, k2_end, k2_ctas = _role_metrics(
                trace, ROLE_K2
            )
            kernel_overlap = max(
                0, min(k1_end, k2_end) - max(k1_start, k2_start)
            )
            active_overlap = max(
                0, min(k1_last, k2_last) - max(k1_first, k2_first)
            )
            kernel_union = max(k1_end, k2_end) - min(k1_start, k2_start)
            k1_kernel = k1_end - k1_start
            k2_kernel = k2_end - k2_start
            k1_active = max(0, k1_last - k1_first)
            k2_active = max(0, k2_last - k2_first)
            print(
                f"  rank_{rank}: "
                f"K1={_us(k1_kernel):.3f} us "
                f"(first_tile={_us(max(0, k1_first - k1_start)):.3f} us, "
                f"active={_us(k1_active):.3f}, ctas={k1_ctas}) "
                f"K2={_us(k2_kernel):.3f} us "
                f"(active={_us(k2_active):.3f}, ctas={k2_ctas}) "
                f"kernel_overlap={_us(kernel_overlap):.3f} us "
                f"active_overlap={_us(active_overlap):.3f} us "
                f"union={_us(kernel_union):.3f} us "
                f"K2_start_wait={_us(max(0, k2_first - k2_start)):.3f} us "
                f"K2_tail={_us(max(0, k2_end - k2_last)):.3f} us"
            )
            k1_rows = trace[ROLE_K1]
            k1_valid = (
                (k1_rows[:, FIELD_KERNEL_ENTRY] > 0)
                & (
                    k1_rows[:, FIELD_KERNEL_EXIT]
                    > k1_rows[:, FIELD_KERNEL_ENTRY]
                )
                & (k1_rows[:, FIELD_TILE_COUNT] > 0)
            )
            if bool(k1_valid.any().item()):
                k1_rows = k1_rows[k1_valid]
                k1_tma_a_ns = _sampled_wait_estimate(
                    k1_rows,
                    FIELD_TMA_A_WAIT_NS,
                    FIELD_TMA_A_WAIT_CALLS,
                    FIELD_TMA_A_TIMED_CALLS,
                )
                k1_tma_b_ns = _sampled_wait_estimate(
                    k1_rows,
                    FIELD_TMA_B_WAIT_NS,
                    FIELD_TMA_B_WAIT_CALLS,
                    FIELD_TMA_B_TIMED_CALLS,
                )
                print(
                    "    K1 CTA-sum attribution: "
                    f"TMA_A_wait~={_us(int(k1_tma_a_ns.sum().item())):.3f} us/"
                    f"{int(k1_rows[:, FIELD_TMA_A_WAIT_CALLS].sum().item())} stalls, "
                    f"TMA_B_wait~={_us(int(k1_tma_b_ns.sum().item())):.3f} us/"
                    f"{int(k1_rows[:, FIELD_TMA_B_WAIT_CALLS].sum().item())} stalls, "
                    f"mainloop="
                    f"{_us(int(k1_rows[:, FIELD_MAINLOOP_NS].sum().item())):.3f} us, "
                    f"epilogue="
                    f"{_us(int(k1_rows[:, FIELD_STORE_NS].sum().item())):.3f} us, "
                    f"TMA_A/B~ p90="
                    f"{_us(int(_pct(k1_tma_a_ns, 0.90))):.3f}/"
                    f"{_us(int(_pct(k1_tma_b_ns, 0.90))):.3f} us, "
                    f"max="
                    f"{_us(int(k1_tma_a_ns.max().item())):.3f}/"
                    f"{_us(int(k1_tma_b_ns.max().item())):.3f} us"
                )

            rows = trace[ROLE_K2]
            valid = (
                (rows[:, FIELD_KERNEL_ENTRY] > 0)
                & (rows[:, FIELD_KERNEL_EXIT] > rows[:, FIELD_KERNEL_ENTRY])
                & (rows[:, FIELD_TILE_COUNT] > 0)
            )
            if not bool(valid.any().item()):
                continue
            cta_ids = torch.arange(TRACE_CTA_CAPACITY, dtype=torch.int64)[valid]
            rows = rows[valid]
            tile_count = rows[:, FIELD_TILE_COUNT]
            lifetime = rows[:, FIELD_KERNEL_EXIT] - rows[:, FIELD_KERNEL_ENTRY]
            active = rows[:, FIELD_LAST_WORK] - rows[:, FIELD_FIRST_WORK]
            start_gap = rows[:, FIELD_FIRST_WORK] - rows[:, FIELD_KERNEL_ENTRY]
            final_tail = rows[:, FIELD_KERNEL_EXIT] - rows[:, FIELD_LAST_WORK]
            ready_ns = rows[:, FIELD_READY_WAIT_NS]
            tma_a_ns = _sampled_wait_estimate(
                rows,
                FIELD_TMA_A_WAIT_NS,
                FIELD_TMA_A_WAIT_CALLS,
                FIELD_TMA_A_TIMED_CALLS,
            )
            tma_b_ns = _sampled_wait_estimate(
                rows,
                FIELD_TMA_B_WAIT_NS,
                FIELD_TMA_B_WAIT_CALLS,
                FIELD_TMA_B_TIMED_CALLS,
            )
            mainloop_ns = rows[:, FIELD_MAINLOOP_NS]
            store_ns = rows[:, FIELD_STORE_NS]
            compute_body_ns = torch.clamp(
                mainloop_ns - tma_a_ns - tma_b_ns, min=0
            )
            scheduler_gap_ns = torch.clamp(
                active - mainloop_ns - store_ns, min=0
            )
            tile_mean = float(tile_count.to(torch.float64).mean().item())
            tile_cv = 0.0
            if tile_mean > 0.0 and tile_count.numel() > 1:
                tile_cv = float(
                    tile_count.to(torch.float64).std(unbiased=False).item()
                    / tile_mean
                )
            print(
                f"    K2 CTA work: tiles total={int(tile_count.sum().item())} "
                f"min/p50/p90/max="
                f"{int(tile_count.min().item())}/"
                f"{_pct(tile_count, 0.50):.1f}/"
                f"{_pct(tile_count, 0.90):.1f}/"
                f"{int(tile_count.max().item())}, cv={tile_cv:.3f}; "
                f"lifetime p50/p90/max="
                f"{_us(int(_pct(lifetime, 0.50))):.3f}/"
                f"{_us(int(_pct(lifetime, 0.90))):.3f}/"
                f"{_us(int(lifetime.max().item())):.3f} us"
            )
            print(
                "    K2 CTA-sum attribution: "
                f"ready_wait={_us(int(ready_ns.sum().item())):.3f} us/"
                f"{int(rows[:, FIELD_READY_WAIT_CALLS].sum().item())} calls, "
                f"TMA_A_wait~={_us(int(tma_a_ns.sum().item())):.3f} us/"
                f"{int(rows[:, FIELD_TMA_A_WAIT_CALLS].sum().item())} stalls, "
                f"TMA_B_wait~={_us(int(tma_b_ns.sum().item())):.3f} us/"
                f"{int(rows[:, FIELD_TMA_B_WAIT_CALLS].sum().item())} stalls, "
                f"LDS+QMMA~={_us(int(compute_body_ns.sum().item())):.3f} us, "
                f"peer_store={_us(int(store_ns.sum().item())):.3f} us, "
                f"scheduler/gaps~={_us(int(scheduler_gap_ns.sum().item())):.3f} us"
            )
            print(
                "    K2 CTA tails: "
                f"start_gap p90/max={_us(int(_pct(start_gap, 0.90))):.3f}/"
                f"{_us(int(start_gap.max().item())):.3f} us, "
                f"final_barrier p90/max={_us(int(_pct(final_tail, 0.90))):.3f}/"
                f"{_us(int(final_tail.max().item())):.3f} us, "
                f"ready_wait p90/max={_us(int(_pct(ready_ns, 0.90))):.3f}/"
                f"{_us(int(ready_ns.max().item())):.3f} us, "
                f"TMA_A/B~ max={_us(int(tma_a_ns.max().item())):.3f}/"
                f"{_us(int(tma_b_ns.max().item())):.3f} us"
            )
            critical_index = int(rows[:, FIELD_LAST_WORK].argmax().item())
            critical = rows[critical_index]
            print(
                f"    K2 critical CTA={int(cta_ids[critical_index].item())}: "
                f"tiles={int(critical[FIELD_TILE_COUNT].item())}, "
                f"first(e,n,m)={_decode_tile_id(int(critical[FIELD_FIRST_TILE_ID].item()))}, "
                f"last(e,n,m)={_decode_tile_id(int(critical[FIELD_LAST_TILE_ID].item()))}, "
                f"lifetime={_us(int(lifetime[critical_index].item())):.3f} us, "
                f"ready={_us(int(critical[FIELD_READY_WAIT_NS].item())):.3f} us, "
                f"TMA_A/B~={_us(int(tma_a_ns[critical_index].item())):.3f}/"
                f"{_us(int(tma_b_ns[critical_index].item())):.3f} us, "
                f"mainloop={_us(int(critical[FIELD_MAINLOOP_NS].item())):.3f} us, "
                f"store={_us(int(critical[FIELD_STORE_NS].item())):.3f} us, "
                f"final={_us(int(final_tail[critical_index].item())):.3f} us"
            )

            if self._split_k2_tile_trace_enabled:
                tile_rows = rank_tensor[TRACE_BASE_WORDS:].reshape(
                    K2_TILE_TRACE_CTA_CAPACITY,
                    K2_TILE_TRACE_TILES_PER_CTA,
                    K2_TILE_TRACE_FIELDS,
                ).cpu()
                tile_valid = (
                    tile_rows[:, :, K2_TILE_FIELD_TILE_END]
                    > tile_rows[:, :, K2_TILE_FIELD_TILE_BEGIN]
                )
                if bool(tile_valid.any().item()):
                    samples = tile_rows[tile_valid]
                    dequeue_ns = torch.clamp(
                        samples[:, K2_TILE_FIELD_DEQUEUE_END]
                        - samples[:, K2_TILE_FIELD_DEQUEUE_BEGIN],
                        min=0,
                    )
                    tile_wall_ns = torch.clamp(
                        samples[:, K2_TILE_FIELD_TILE_END]
                        - samples[:, K2_TILE_FIELD_TILE_BEGIN],
                        min=0,
                    )
                    print(
                        "    K2 per-tile exact CTA-sum: "
                        f"tiles={samples.shape[0]}, "
                        f"dequeue={_us(int(dequeue_ns.sum().item())):.3f} us, "
                        f"TMA_A_wait={_us(int(samples[:, K2_TILE_FIELD_TMA_A_WAIT_NS].sum().item())):.3f} us, "
                        f"TMA_B_wait={_us(int(samples[:, K2_TILE_FIELD_TMA_B_WAIT_NS].sum().item())):.3f} us, "
                        f"LDSM+QMMA={_us(int(samples[:, K2_TILE_FIELD_LDSM_QMMA_NS].sum().item())):.3f} us, "
                        f"BF16_pack={_us(int(samples[:, K2_TILE_FIELD_BF16_PACK_NS].sum().item())):.3f} us, "
                        f"peer_store={_us(int(samples[:, K2_TILE_FIELD_PEER_STORE_NS].sum().item())):.3f} us, "
                        f"phase_advance={_us(int(samples[:, K2_TILE_FIELD_PHASE_ADVANCE_NS].sum().item())):.3f} us"
                    )
                    print(
                        "    K2 per-tile wall p50/p90/max="
                        f"{_us(int(_pct(tile_wall_ns, 0.50))):.3f}/"
                        f"{_us(int(_pct(tile_wall_ns, 0.90))):.3f}/"
                        f"{_us(int(tile_wall_ns.max().item())):.3f} us; "
                        "dequeue p50/p90/max="
                        f"{_us(int(_pct(dequeue_ns, 0.50))):.3f}/"
                        f"{_us(int(_pct(dequeue_ns, 0.90))):.3f}/"
                        f"{_us(int(dequeue_ns.max().item())):.3f} us"
                    )

    def _report_split_trace_replays(
        self, replay_traces: list[torch.Tensor]
    ) -> None:
        """Report the production Green-Graph K2 final-tail distribution.

        The optional replay capture synchronizes only after each measured end
        event, so it does not add work inside the CUDA-event interval.  Normal
        benchmark runs never enter this diagnostic path.
        """
        if not replay_traces:
            return

        from moe_sm120_nvfp4_split.split_timestamp import (
            FIELD_KERNEL_EXIT,
            FIELD_LAST_WORK,
            K2_TILE_FIELD_TILE_BEGIN,
            K2_TILE_FIELD_TILE_END,
            K2_TILE_FIELD_TILE_ID,
            K2_TILE_TRACE_CTA_CAPACITY,
            K2_TILE_TRACE_FIELDS,
            K2_TILE_TRACE_TILES_PER_CTA,
            ROLE_K2,
            TRACE_BASE_WORDS,
            TRACE_CTA_CAPACITY,
            TRACE_FIELDS,
            TRACE_ROLES,
        )

        tails = []
        tile_sets: list[set[int]] = []
        tile_counts: list[int] = []
        for replay_trace in replay_traces:
            trace = replay_trace[:TRACE_BASE_WORDS].reshape(
                TRACE_ROLES, TRACE_CTA_CAPACITY, TRACE_FIELDS
            )
            rows = trace[ROLE_K2]
            valid = (
                (rows[:, FIELD_KERNEL_EXIT] > 0)
                & (rows[:, FIELD_LAST_WORK] > 0)
                & (rows[:, FIELD_KERNEL_EXIT] >= rows[:, FIELD_LAST_WORK])
            )
            if bool(valid.any().item()):
                rows = rows[valid]
                tails.append(
                    rows[:, FIELD_KERNEL_EXIT].max()
                    - rows[:, FIELD_LAST_WORK].max()
                )
            else:
                tails.append(torch.tensor(0, dtype=torch.int64))

            tile_rows = replay_trace[TRACE_BASE_WORDS:].reshape(
                K2_TILE_TRACE_CTA_CAPACITY,
                K2_TILE_TRACE_TILES_PER_CTA,
                K2_TILE_TRACE_FIELDS,
            )
            tile_valid = (
                tile_rows[:, :, K2_TILE_FIELD_TILE_END]
                > tile_rows[:, :, K2_TILE_FIELD_TILE_BEGIN]
            )
            tile_ids = tile_rows[:, :, K2_TILE_FIELD_TILE_ID][tile_valid]
            tile_counts.append(int(tile_ids.numel()))
            tile_sets.append({int(value) for value in tile_ids.cpu().tolist()})

        reference_tiles = tile_sets[0]
        local_tile_summary = [
            {
                "count": tile_counts[index],
                "unique": len(tiles),
                "missing": len(reference_tiles - tiles),
                "extra": len(tiles - reference_tiles),
            }
            for index, tiles in enumerate(tile_sets)
        ]
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            gathered_tile_summary = [
                None for _ in range(self.global_world_size)
            ]
            torch.distributed.all_gather_object(
                gathered_tile_summary, local_tile_summary
            )
        else:
            gathered_tile_summary = [local_tile_summary]

        local_tail_ns = torch.stack(tails).to(torch.float64)
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
        ):
            gathered = [
                torch.empty_like(local_tail_ns)
                for _ in range(self.global_world_size)
            ]
            torch.distributed.all_gather(gathered, local_tail_ns)
            critical_tail_ns = torch.stack(gathered).amax(dim=0)
        else:
            critical_tail_ns = local_tail_ns

        if self.global_rank == 0:
            print("---- production Green Graph K2 replay tile sets ----")
            for rank, summaries in enumerate(gathered_tile_summary):
                print(f"  rank {rank}: {summaries}")
            values = critical_tail_ns.cpu().numpy() / 1000.0
            print("---- production Green Graph K2 final-tail %globaltimer ----")
            print(
                f"  replays={len(values)} "
                f"min={float(np.min(values)):.3f} us "
                f"p50={float(np.median(values)):.3f} us "
                f"p90={float(np.percentile(values, 90)):.3f} us "
                f"p99={float(np.percentile(values, 99)):.3f} us "
                f"max={float(np.max(values)):.3f} us"
            )

    def _launch_target_kernels_with_optional_torch_profiler(
        self,
        runtime_kwargs,
        topk_reduce_runtime_kwargs,
    ) -> None:
        """Launch the target kernel pipeline, optionally under torch profiler."""
        if self._compiled_kernel_k1 is None or self._compiled_kernel_k2 is None:
            raise RuntimeError("compiled split kernels are not available")
        pool_to_sf = {}
        if (
            topk_reduce_runtime_kwargs is not None
            and self._compiled_topk_reduce is None
        ):
            raise RuntimeError("compiled topk reduce kernel is not available")

        def _launch() -> None:
            use_ibgda = self.comm_backend == "gin"
            if use_ibgda:
                self._ibgda_dispatch_inputs()
                self._ibgda_build_runtime_owner_route_plan()
                self._ibgda_streaming_combine_begin()
            if self.split_launch == "sequential":
                self._compiled_kernel_k1(**self._runtime_kwargs_k1)
                # Sequential mode is a bring-up option and intentionally
                # exposes the K1/K2 boundary.
                torch.cuda.synchronize()
                if (
                    torch.distributed.is_available()
                    and torch.distributed.is_initialized()
                ):
                    self._ep_barrier()
                self._compiled_kernel_k2(**self._runtime_kwargs_k2)
                if self._compiled_kernel_k2_finalizer is not None:
                    self._compiled_kernel_k2_finalizer(
                        **self._runtime_kwargs_k2_finalizer
                    )
            else:
                phase = self._phase_timing_begin("k1_k2.compute")
                self._native_green_graph.launch(torch.cuda.current_stream())
                self._phase_timing_end(phase)

            if use_ibgda:
                self._ibgda_streaming_dispatch_end()
                self._ibgda_streaming_combine_end()

            if topk_reduce_runtime_kwargs is not None:
                phase = self._phase_timing_begin("k3.topk_reduce")
                if self._tp_k3_chunk_runtime_kwargs is not None:
                    self._launch_chunked_k3_tp_reduce()
                elif use_ibgda or self.split_launch == "sequential":
                    self._compiled_topk_reduce(**topk_reduce_runtime_kwargs)
                    self._tp_reduce_actual_output()
                else:
                    # The non-IBGDA Green graph captures K3.
                    self._tp_reduce_actual_output()
                self._phase_timing_end(phase)

        if not self._use_torch_profiler and not self._use_cuda_events:
            self._reset_before_launch()
            _launch()
            torch.cuda.synchronize()
            if self.misc.enable_debug_checks:
                self._validate_dispatch_handoff_epoch(verbose=True)
            return

        if self._use_cuda_events:
            drain_streaming_transport_replays = self.comm_backend in (
                "gin",
                "gin_hybrid",
            )
            strict_replay_barriers = (
                drain_streaming_transport_replays
                and self.misc.enable_debug_checks
            )
            # Phase events are also the explicit cross-stream progress
            # boundaries for the Green-Context compute graph and the two
            # communication rings.  Create them during warmup as well, then discard
            # their timings.  Correctness must not depend on whether the
            # caller asks us to report the warmup phases.
            self._phase_timing_events = {}
            self._phase_timing_collect = self._phase_timing_enabled
            for warmup_index in range(self._perf_warmup):
                if drain_streaming_transport_replays:
                    self._wait_for_previous_streaming_replay()
                self._update_runtime_route(
                    warmup_index - self._perf_warmup
                )
                self._reset_before_launch()
                if strict_replay_barriers:
                    # JIT and reset duration differs per rank. Do not let an
                    # early PE fill the GPU with persistent polling CTAs before
                    # every peer has reached the same epoch's launch point.
                    torch.cuda.synchronize()
                    self._ep_barrier()
                _launch()
                # Warmup is outside the measured interval.  Drain each replay
                # before reusing the device-side epoch/control records: a
                # queue of persistent communication polling kernels can otherwise
                # amplify rank launch skew and let a later replay occupy the
                # progress resources needed to retire an earlier replay.
                if strict_replay_barriers:
                    torch.cuda.synchronize()
                    if drain_streaming_transport_replays:
                        self._validate_dispatch_handoff_epoch()
                    # A terminal slot ACK protects that slot, but does not say
                    # that every PE has retired all compute/control consumers
                    # of the epoch. Keep resets and epoch publication globally
                    # ordered between benchmark replays.
                    self._ep_barrier()
                self._phase_timing_events = {}
            self._phase_timing_collect = False
            self._phase_timing_events = {}
            self._phase_timing_collect = True

            starts = [
                torch.cuda.Event(enable_timing=True) for _ in range(self._perf_iters)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True) for _ in range(self._perf_iters)
            ]
            replay_reference = None
            replay_all_rank_reference = None
            replay_all_rank_topk = None
            replay_byte_diff = 0
            replay_first_mismatch = -1
            replay_workspace_references = {}
            replay_workspace_first_mismatch = {
                "l1_token_buffer": -1,
                "fc1_output": -1,
            }
            replay_canonical_fc1_reference = None
            replay_canonical_fc1_keys = None
            replay_canonical_fc1_first_mismatch = -1
            replay_canonical_fc1_byte_diff = 0
            replay_live_rows = None
            diagnose_fc1 = (
                os.environ.get("MEGA_DIAG_FC1_REPLAY", "0") == "1"
                or os.environ.get("MEGA_DIAG_CANONICAL_FC1", "0") == "1"
            )
            if diagnose_fc1:
                all_counts = torch.bincount(
                    self._global_topk_idx.reshape(-1),
                    minlength=self.problem.num_total_experts,
                ).cpu()
                expert_begin = self.rank * self.problem.num_experts_per_rank
                local_counts = all_counts[
                    expert_begin : expert_begin
                    + self.problem.num_experts_per_rank
                ].tolist()
                physical_rows = []
                physical_begin = 0
                padding = self._kernel.token_padding_block
                for count in local_counts:
                    physical_rows.extend(
                        range(physical_begin, physical_begin + int(count))
                    )
                    physical_begin += (
                        (int(count) + padding - 1) // padding
                    ) * padding
                replay_live_rows = torch.tensor(
                    physical_rows, dtype=torch.long, device="cuda"
                )
            trace_replays = (
                []
                if (
                    self._split_green_trace is not None
                    and os.environ.get("MEGA_SPLIT_TRACE_REPLAYS", "0") == "1"
                )
                else None
            )
            # A streaming communication replay owns the ready/credit rings and their
            # backing slot storage until both the sender and receiver terminal
            # ACKs have joined the caller stream.  Queueing the next replay
            # before that join has retired lets its reset and epoch publication
            # race the previous persistent CTAs.  At large token counts this is
            # nondeterministic: a later polling grid can also consume the SMs
            # needed by the earlier replay to make forward progress.
            #
            # Synchronizing the per-replay end event preserves CUDA-event
            # device timing (host wait time is outside start..end) while making
            # the benchmark obey the same sequential dependency as production
            # MoE steps.  Ordinary non-communication launches remain back-to-back.
            for replay, (start, end) in enumerate(zip(starts, ends)):
                if drain_streaming_transport_replays:
                    self._wait_for_previous_streaming_replay()
                self._update_runtime_route(replay)
                self._reset_before_launch()
                if strict_replay_barriers:
                    torch.cuda.synchronize()
                    self._ep_barrier()
                if self._split_k2_tile_trace_enabled:
                    # Clearing the dense trace buffer is asynchronous and can
                    # otherwise turn rank launch skew into apparent K2 startup.
                    torch.cuda.synchronize()
                    if (
                        torch.distributed.is_available()
                        and torch.distributed.is_initialized()
                    ):
                        self._ep_barrier()
                    torch.cuda.synchronize()
                start.record()
                _launch()
                end.record()
                if drain_streaming_transport_replays:
                    # Submit/query every Green-Context stream once before the
                    # host blocks on the replay end event.  Without this
                    # non-blocking driver flush, immediate event synchronization
                    # can leave mutually dependent persistent grids resident
                    # but unsubmitted on RTX Pro 5000.
                    progress_events = (
                        ("dispatch_send", self._transport.dispatch_send_done),
                        ("dispatch_credit", self._transport.dispatch_credit_done),
                        ("dispatch_recv", self._transport.dispatch_recv_done),
                        ("combine_send", self._transport.ring_done),
                        ("combine_recv", self._transport.ring_recv_done),
                        ("combine_credit", self._transport.ring_credit_done),
                        (
                            "route_push",
                            self._transport.owner_plan_route_push_done,
                        ),
                        ("metadata", self._transport.owner_metadata_done),
                    )
                    poll_attempt = 0
                    while True:
                        for _, event in progress_events:
                            if event is not None:
                                event.query()
                        end_ready = end.query()
                        terminal_credits_ready = (
                            self._transport.dispatch_credit_done.query()
                            and self._transport.ring_credit_done.query()
                        )
                        if end_ready and terminal_credits_ready:
                            break
                        # Querying every dependent stream once is sufficient
                        # to flush the Green-Context launches into the driver.
                        # Yield briefly while the GPU makes progress, then use
                        # a microsecond-scale capped backoff.  A 100 ms sleep
                        # leaves short MoE replays idle long enough to lose the
                        # steady-state clock/launch behavior being measured.
                        if poll_attempt < 8:
                            time.sleep(0)
                        else:
                            backoff_us = min(100, 1 << min(poll_attempt - 8, 7))
                            time.sleep(backoff_us * 1.0e-6)
                        poll_attempt += 1
                    if strict_replay_barriers:
                        torch.cuda.synchronize()
                        self._validate_dispatch_handoff_epoch()
                        self._ep_barrier()
                if os.environ.get("MEGA_RANK_LOCAL_VALIDATE_EVERY_REPLAY", "0") == "1":
                    # Debug/stress only.  The reference runs after the timed
                    # end event and verifies every changing route, rather than
                    # merely the final replay.  Performance sweeps leave this
                    # disabled.
                    end.synchronize()
                    gathered_routes = [
                        torch.empty_like(self.my_topk_idx)
                        for _ in range(self.world_size)
                    ]
                    self._ep_all_gather(gathered_routes, self.my_topk_idx)
                    self._global_topk_idx.copy_(torch.stack(gathered_routes))
                    self.compute_reference(force=True)
                    self.validate(force=True)
                if trace_replays is not None:
                    # Synchronize outside the measured start..end interval,
                    # then preserve this replay before the next reset clears
                    # the compact CTA trace.
                    end.synchronize()
                    trace_replays.append(
                        self._split_green_trace.detach().cpu().clone()
                    )
                if self._verify_replay_bit_determinism:
                    torch.cuda.synchronize()
                    current = self.combine_output.view(torch.uint8)
                    if os.environ.get("MEGA_DIAG_OUTPUT_MATCH", "0") == "1":
                        if replay_all_rank_reference is None:
                            gathered_output = [
                                torch.empty_like(self.combine_output)
                                for _ in range(self.world_size)
                            ]
                            gathered_topk = [
                                torch.empty_like(self.my_topk_idx)
                                for _ in range(self.world_size)
                            ]
                            self._ep_all_gather(
                                gathered_output, self.combine_output
                            )
                            self._ep_all_gather(gathered_topk, self.my_topk_idx)
                            replay_all_rank_reference = [
                                value.detach().clone()
                                for value in gathered_output
                            ]
                            replay_all_rank_topk = [
                                value.detach().clone() for value in gathered_topk
                            ]
                    if diagnose_fc1:
                        for region_name in replay_workspace_first_mismatch:
                            region = self._kernel._local_region_by_name[
                                region_name
                            ]
                            offset = self._kernel._local_offsets[region_name]
                            all_region_bytes = self.local_workspace[
                                offset : offset + region.nbytes
                            ]
                            region_bytes = all_region_bytes.reshape(
                                self._kernel.pool_token_capacity, -1
                            ).index_select(0, replay_live_rows).reshape(-1)
                            reference = replay_workspace_references.get(
                                region_name
                            )
                            if reference is None:
                                replay_workspace_references[region_name] = (
                                    region_bytes.clone()
                                )
                            elif (
                                replay_workspace_first_mismatch[region_name]
                                < 0
                                and bool((region_bytes != reference).any().item())
                            ):
                                replay_workspace_first_mismatch[
                                    region_name
                                ] = replay
                    if os.environ.get("MEGA_DIAG_CANONICAL_FC1", "0") == "1":
                        kernel = self._kernel
                        fc1_region = kernel._local_region_by_name["fc1_output"]
                        fc1_offset = kernel._local_offsets["fc1_output"]
                        fc1_rows = self.local_workspace[
                            fc1_offset : fc1_offset + fc1_region.nbytes
                        ].reshape(kernel.pool_token_capacity, -1)
                        fc1_rows = fc1_rows.index_select(0, replay_live_rows)
                        md_region = kernel._local_region_by_name[
                            "token_src_metadata"
                        ]
                        md_offset = kernel._local_offsets["token_src_metadata"]
                        metadata = self.local_workspace[
                            md_offset : md_offset + md_region.nbytes
                        ].view(torch.int64)
                        metadata = metadata.index_select(0, replay_live_rows)
                        metadata_hi = metadata >> 32
                        metadata_rank = (metadata_hi >> 16) & 0xFFFF
                        metadata_token = metadata & 0xFFFFFFFF
                        metadata_topk = metadata_hi & 0xFFFF
                        metadata_key = (
                            (
                                metadata_rank
                                * self.problem.num_tokens_per_rank
                                + metadata_token
                            )
                            * self.problem.num_topk
                            + metadata_topk
                        )
                        metadata_key, canonical_order = metadata_key.sort()
                        canonical_fc1 = fc1_rows.index_select(
                            0, canonical_order
                        ).reshape(-1)
                        if replay_canonical_fc1_reference is None:
                            replay_canonical_fc1_reference = canonical_fc1.clone()
                            replay_canonical_fc1_keys = metadata_key.clone()
                        else:
                            key_diff = int(
                                (
                                    metadata_key != replay_canonical_fc1_keys
                                ).sum().item()
                            )
                            fc1_byte_diff = int(
                                (
                                    canonical_fc1
                                    != replay_canonical_fc1_reference
                                ).sum().item()
                            )
                            replay_canonical_fc1_byte_diff += fc1_byte_diff
                            if (
                                (key_diff or fc1_byte_diff)
                                and replay_canonical_fc1_first_mismatch < 0
                            ):
                                replay_canonical_fc1_first_mismatch = replay
                                print(
                                    f"[rank {self.rank}] canonical FC1 mismatch "
                                    f"replay={replay} key_diff={key_diff} "
                                    f"byte_diff={fc1_byte_diff}",
                                    flush=True,
                                )
                    if (
                        replay < 3
                        and os.environ.get("MEGA_DIAG_QUEUE_STATE", "0")
                        == "1"
                    ):
                        kernel = self._kernel
                        region = kernel._local_region_by_name[
                            "k2_ready_queue_state"
                        ]
                        queue_state = self._byte_workspace_view(
                            self.local_workspace,
                            kernel._local_offsets["k2_ready_queue_state"],
                            torch.int32,
                            region.shape,
                        ).detach().cpu().tolist()
                        as_bf16 = self.combine_output.float()
                        print(
                            f"[rank {self.rank}] replay={replay} "
                            f"queue_state={queue_state} "
                            f"output_sum={float(as_bf16.sum().item()):.6e} "
                            f"output_absmax={float(as_bf16.abs().max().item()):.6e}",
                            flush=True,
                        )
                    if replay_reference is None:
                        replay_reference = current.clone()
                    else:
                        from .combine_reference import byte_difference
                        current_byte_diff = byte_difference(current, replay_reference)
                        replay_byte_diff += current_byte_diff
                        if current_byte_diff and replay_first_mismatch < 0:
                            replay_first_mismatch = replay
                            if os.environ.get("MEGA_DIAG_OUTPUT_COORDS", "0") == "1":
                                current_bf16 = self.combine_output.detach()
                                reference_bf16 = replay_reference.view(
                                    torch.bfloat16
                                ).reshape_as(current_bf16)
                                different = current_bf16 != reference_bf16
                                coords = different.nonzero()[:16].cpu().tolist()
                                slot_counts = different.any(dim=-1).sum(dim=0).cpu().tolist()
                                details = []
                                matched_rows = set()
                                for token_idx, topk_idx, hidden_idx in coords:
                                    expert = int(
                                        self.my_topk_idx[token_idx, topk_idx].item()
                                    )
                                    detail = {
                                            "coord": [token_idx, topk_idx, hidden_idx],
                                            "expert": expert,
                                            "owner": expert
                                            // self.problem.num_experts_per_rank,
                                            "current": float(
                                                current_bf16[
                                                    token_idx, topk_idx, hidden_idx
                                                ].item()
                                            ),
                                            "reference": float(
                                                reference_bf16[
                                                    token_idx, topk_idx, hidden_idx
                                                ].item()
                                            ),
                                        }
                                    row_key = (token_idx, topk_idx)
                                    if (
                                        replay_all_rank_reference is not None
                                        and row_key not in matched_rows
                                    ):
                                        matched_rows.add(row_key)
                                        candidate_rows = []
                                        candidate_coords = []
                                        for source_rank in range(self.world_size):
                                            mask = (
                                                replay_all_rank_topk[source_rank]
                                                == expert
                                            )
                                            source_coords = mask.nonzero()
                                            if source_coords.numel() == 0:
                                                continue
                                            candidate_rows.append(
                                                replay_all_rank_reference[
                                                    source_rank
                                                ][mask]
                                            )
                                            candidate_coords.extend(
                                                [
                                                    [
                                                        source_rank,
                                                        int(coord[0].item()),
                                                        int(coord[1].item()),
                                                    ]
                                                    for coord in source_coords
                                                ]
                                            )
                                        if candidate_rows:
                                            candidates = torch.cat(
                                                candidate_rows, dim=0
                                            )
                                            current_row = current_bf16[
                                                token_idx, topk_idx
                                            ]
                                            exact = (
                                                candidates == current_row
                                            ).all(dim=-1)
                                            mean_abs = (
                                                candidates.float()
                                                - current_row.float()
                                            ).abs().mean(dim=-1)
                                            nearest = int(
                                                mean_abs.argmin().item()
                                            )
                                            max_abs = float(
                                                (
                                                    candidates[nearest].float()
                                                    - current_row.float()
                                                )
                                                .abs()
                                                .max()
                                                .item()
                                            )
                                            detail["row_match"] = {
                                                "exact": bool(exact.any().item()),
                                                "nearest": candidate_coords[nearest],
                                                "nearest_mean_abs": float(
                                                    mean_abs[nearest].item()
                                                ),
                                                "nearest_max_abs": max_abs,
                                            }
                                    details.append(detail)
                                print(
                                    f"[rank {self.rank}] first output mismatch "
                                    f"replay={replay} changed_rows_by_slot="
                                    f"{slot_counts} details={details}",
                                    flush=True,
                                )
                    # Diagnostic only: determine whether dual-resident K2
                    # corruption is internal to one replay or caused by rank
                    # skew when ownership advances to the next graph epoch.
                    self._ep_barrier()
            torch.cuda.synchronize()
            if self._verify_replay_bit_determinism:
                diff = torch.tensor(
                    [replay_byte_diff], dtype=torch.int64, device="cuda"
                )
                gathered_diff = [
                    torch.empty_like(diff) for _ in range(self.world_size)
                ]
                self._ep_all_gather(gathered_diff, diff)
                rank_diffs = [int(value.item()) for value in gathered_diff]
                if any(rank_diffs):
                    first = torch.tensor(
                        [replay_first_mismatch],
                        dtype=torch.int64,
                        device="cuda",
                    )
                    gathered_first = [
                        torch.empty_like(first) for _ in range(self.world_size)
                    ]
                    self._ep_all_gather(gathered_first, first)
                    first_mismatches = [
                        int(value.item()) for value in gathered_first
                    ]
                    raise AssertionError(
                        "static-route replay bit determinism failed: "
                        + ", ".join(
                            f"rank {rank}: byte_diff={value}, "
                            f"first_replay={first_mismatches[rank]}"
                            for rank, value in enumerate(rank_diffs)
                        )
                        + (
                            "; local FC1 first mismatch="
                            + repr(replay_workspace_first_mismatch)
                            if replay_workspace_references
                            else ""
                        )
                    )
                if self.rank == 0:
                    print(
                        f"Replay bit determinism PASSED for "
                        f"{self._perf_iters} launches on all ranks",
                        flush=True,
                    )
                    if replay_workspace_references:
                        print(
                            "Diagnostic FC1 first mismatch: "
                            f"{replay_workspace_first_mismatch}",
                            flush=True,
                        )
                    if replay_canonical_fc1_reference is not None:
                        print(
                            "Canonical FC1 replay diagnostic: "
                            f"first_mismatch="
                            f"{replay_canonical_fc1_first_mismatch}, "
                            f"total_byte_diff="
                            f"{replay_canonical_fc1_byte_diff}",
                            flush=True,
                        )
            if self.misc.enable_debug_checks:
                self._validate_dispatch_handoff_epoch(verbose=True)
            self._phase_timing_collect = False
            local_us = torch.tensor(
                [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)],
                dtype=torch.float64,
                device="cuda",
            )
            gathered = [torch.empty_like(local_us) for _ in range(self.world_size)]
            self._ep_all_gather(gathered, local_us)
            if self.rank == 0:
                critical_us = torch.stack(gathered).amax(dim=0).cpu().numpy()
                parallel_suffix = (
                    f" [DP {self.data_parallel_rank}/"
                    f"{self.data_parallel_size}, TP "
                    f"{self.tensor_parallel_rank}/"
                    f"{self.tensor_parallel_size}, EP{self.world_size}]"
                )
                print(
                    "---- CUDA-event target pipeline critical-path time"
                    f"{parallel_suffix} ----"
                )
                print(
                    f"  iters={len(critical_us)} "
                    f"min={float(np.min(critical_us)):.2f} us "
                    f"p10={float(np.percentile(critical_us, 10)):.2f} us "
                    f"p50={float(np.median(critical_us)):.2f} us "
                    f"p90={float(np.percentile(critical_us, 90)):.2f} us "
                    f"max={float(np.max(critical_us)):.2f} us"
                )
            self._report_phase_timings()
            if self.parallel_context is not None:
                # The default process group is the global Gloo control plane.
                # This post-timing gather gives one true all-rank critical path
                # without creating another NCCL communicator or perturbing the
                # measured GPU interval.
                global_local_us = local_us.cpu()
                global_gathered = [
                    torch.empty_like(global_local_us)
                    for _ in range(self.global_world_size)
                ]
                torch.distributed.all_gather(
                    global_gathered,
                    global_local_us,
                )
                if self.global_rank == 0:
                    global_critical_us = (
                        torch.stack(global_gathered).amax(dim=0).numpy()
                    )
                    print(
                        f"---- CUDA-event global {self.global_world_size}-rank critical-path time ----"
                    )
                    print(
                        f"  global_iters={len(global_critical_us)} "
                        f"min={float(np.min(global_critical_us)):.2f} us "
                        f"p10={float(np.percentile(global_critical_us, 10)):.2f} us "
                        f"p50={float(np.median(global_critical_us)):.2f} us "
                        f"p90={float(np.percentile(global_critical_us, 90)):.2f} us "
                        f"max={float(np.max(global_critical_us)):.2f} us"
                    )
            if trace_replays is not None:
                self._report_split_trace_replays(trace_replays)
            return

        # Warm up (untimed): reset the local counters before each launch so the
        # re-launch sees clean spin thresholds; the kernel's entry barrier
        # re-aligns ranks each launch, so no host barrier is needed between them.
        for _ in range(self._perf_warmup):
            self._reset_before_launch()
            _launch()
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
                self._reset_before_launch()
                _launch()
            torch.cuda.synchronize()
        self._report_torch_profiler_kernel_time(prof, num_iters=n_iters)

    def run_kernel(self) -> None:
        """Compile + launch ``Sm120MegaMoENvfp4SwapABKernel`` on the current cuda stream.

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
            or self.combine_output is None
        ):
            raise RuntimeError("run_kernel requires generate_inputs first.")

        # Build the failure-detecting CPU control plane before per-rank JIT.
        # A rank that fails compilation can then be detected before any peer
        # enters the device-side communication barrier.
        self._ensure_host_control_group()

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

        from moe_sm120_nvfp4_split.support.src.sym_buffer import SymBufferHost
        from moe_sm120_nvfp4_split.kernel_combine_reduce import (
            compile_topk_reduce,
        )

        # -- 1. Kernel instance --
        cluster_size = 1
        max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
            cluster_size
        )
        reserved_comm_sms = 0
        if self.split_launch == "green_graph":
            if (
                self.comm_backend == "gin"
            ):
                from moe_sm120_nvfp4_split.runtime.green_context import (
                    NativeGreenContextResources,
                )

                self._native_green_resources = (
                    NativeGreenContextResources.create(
                        k1_sm_count=self.kernel_config.k1_sms,
                        k2_sm_count=self.kernel_config.k2_sms,
                        comm_tx_sm_count=self.kernel_config.tx_sms,
                        comm_rx_sm_count=self.kernel_config.rx_sms,
                        # Fixed-ownership dispatch/combine data streams and
                        # the finite route-push producer use the communication
                        # Green Contexts.  Owner metadata stays on its ordinary
                        # CUDA stream: placing a polling metadata grid beside
                        # the 16 persistent dispatch-RX CTAs can prevent the
                        # receiver's complete peer/channel set from residing.
                        num_comm_streams=5,
                    )
                )
                reserved_comm_sms = (
                    self._native_green_resources.comm_sm_count
                )
                k1_active_clusters = (
                    self._native_green_resources.k1_sm_count
                )
                k2_partition_clusters = (
                    self._native_green_resources.k2_sm_count
                )
                if self.rank == 0:
                    print(
                        "[mega_runner] Green Context SM split: "
                        f"K1={k1_active_clusters}, "
                        "COMM="
                        f"{self._native_green_resources.comm_sm_count} "
                        "(TX="
                        f"{self._native_green_resources.comm_tx_sm_count}, "
                        "RX="
                        f"{self._native_green_resources.comm_rx_sm_count}), "
                        f"K2={k2_partition_clusters}",
                        flush=True,
                    )
            else:
                from moe_sm120_nvfp4_split.runtime.green_context import (
                    query_green_context_sm_counts,
                )

                (
                    k1_active_clusters,
                    k2_partition_clusters,
                ) = query_green_context_sm_counts(
                    k1_sm_count=self.kernel_config.k1_sms,
                )
                if k2_partition_clusters != self.kernel_config.k2_sms:
                    raise RuntimeError(
                        "CUDA K2 partition differs from heuristic: "
                        f"actual={k2_partition_clusters}, "
                        f"expected={self.kernel_config.k2_sms}."
                    )
        else:
            k1_active_clusters = max_active_clusters
            k2_partition_clusters = max_active_clusters
        k2_ctas_per_sm = 2 if self.k2_warp_count == 12 else 1
        if self.k2_warp_count == 12:
            k2_ctas_per_sm = self._dual_k2_ctas_per_sm
        k2_active_clusters = k2_partition_clusters * k2_ctas_per_sm
        k2_drain_launch_clusters = (
            k1_active_clusters * k2_ctas_per_sm
            if self.k2_tail_reclaim
            else 0
        )
        self._transport.k1_sm_count = k1_active_clusters * cluster_size
        compile_spec = MegaMoECompileSpec(
            problem=MegaMoEProblemSpec(
                tokens_per_rank=self.problem.num_tokens_per_rank,
                num_topk=self.problem.num_topk,
                num_total_experts=self.problem.num_total_experts,
                hidden=self.problem.hidden,
                intermediate=self.problem.intermediate,
                expert_parallel_size=self.world_size,
                expert_parallel_rank=self.rank,
                data_parallel_size=self.data_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
                gate_up_clamp=self.problem.gate_up_clamp,
            ),
            kernel=self.kernel_config,
            jit=self.jit_config,
            build=SplitKernelBuildOptions(
                group_hint=self.impl.group_hint,
                flag_batch=self.impl.flag_batch,
                epi_flag_batch=self.impl.epi_flag_batch,
                concurrent_k1_k2=self.split_launch == "green_graph",
                k1_active_clusters=k1_active_clusters,
                # The build API receives physical K2 SMs and derives the
                # launch grid from the selected K2 mapping. Passing the
                # already-expanded grid here would double it a second time.
                k2_active_clusters=k2_partition_clusters,
            ),
        )
        kernel_bundle = build_split_kernels(compile_spec)
        self._kernel_k1 = kernel_bundle.k1
        self._kernel_k2 = kernel_bundle.k2
        self._kernel_k2_drain = kernel_bundle.k2_drain
        self._kernel_k2_finalizer = kernel_bundle.k2_finalizer
        self._kernel_cache_key = kernel_bundle.cache_key
        if self.rank == 0:
            print(f"[SM120 JIT cache key] {self._kernel_cache_key}")

        self._validate_kernel_tail_topology(self._kernel_k1, "K1")
        if self.k2_warp_count != 12:
            self._validate_kernel_tail_topology(self._kernel_k2, "K2")
        if self.k2_tail_reclaim and self.k2_warp_count != 12:
            self._validate_kernel_tail_topology(
                self._kernel_k2_drain, "K2 drain"
            )
        if self._kernel_k2_finalizer is not None:
            self._validate_kernel_tail_topology(
                self._kernel_k2_finalizer, "K2 finalizer"
            )
        if self.k2_ready_queue:
            queue_consumers = k2_active_clusters
            if self.k2_tail_reclaim:
                queue_consumers += k2_drain_launch_clusters
            if queue_consumers > 256:
                raise RuntimeError(
                    "K2 ready-queue DONE sentinel capacity is 256 but "
                    f"{queue_consumers} scheduler CTAs may consume it."
                )
        self._kernel = self._kernel_k1
        workspace_sizes = (
            kernel_bundle.local_workspace_bytes,
            kernel_bundle.shared_workspace_bytes,
        )
        if self.rank == 0:
            print(
                "Split-MegaMoE launch: "
                f"mode={self.split_launch} "
                f"K1_clusters={k1_active_clusters} "
                f"K2_partition_clusters={k2_partition_clusters} "
                f"K2_grid_clusters={k2_active_clusters} "
                f"K2_drain_launch_ctas={k2_drain_launch_clusters} "
                f"K2_ctas_per_sm={k2_ctas_per_sm} "
                f"comm_reserved_sms={reserved_comm_sms} "
                "partition="
                f"{self.kernel_config.k1_sms}/"
                f"{self.kernel_config.tx_sms}/"
                f"{self.kernel_config.rx_sms}/"
                f"{self.kernel_config.k2_sms} "
                f"K1_compute_warps={self._kernel_k1.threads_per_cta // 32} "
                f"K2_warps={self._kernel_k2.threads_per_cta // 32} "
                f"K2_compute_warps={len(self._kernel_k2.compute_warp_id)} "
                f"K2_threads={self._kernel_k2.threads_per_cta} "
                f"K2_tiler={self.k2_mma_tiler_mnk} "
                f"K2_stages={self.k2_num_ab_stages} "
                f"K2_tail_reclaim={self.k2_tail_reclaim} "
                f"K2_ready_queue={self.k2_ready_queue} "
                f"TP_K3_chunks={self._tp_k3_chunk_count}"
            )

        # -- 2. Workspaces (local cuda + sym-heap) --
        self.allocate_workspaces()
        if self._split_globaltimer_enabled:
            from moe_sm120_nvfp4_split.split_timestamp import (
                K2_TILE_TRACE_WORDS,
                TRACE_BASE_WORDS,
                TRACE_CTA_CAPACITY,
                TRACE_FIELDS,
                TRACE_ROLES,
            )

            trace_words = TRACE_BASE_WORDS
            if self._split_k2_tile_trace_enabled:
                trace_words += K2_TILE_TRACE_WORDS
            self._split_green_trace = torch.zeros(
                trace_words,
                dtype=torch.int64,
                device="cuda",
            )
        else:
            self._split_green_trace = None

        # -- 3. Torch -> cute --
        # Use 16-byte alignment for vector payloads and 4-byte alignment for
        # i32/fp32 control buffers.
        def _to_cute(tensor: torch.Tensor, assumed_align: int = 16, force_static_layout = False):
            cute_tensor = cutlass_torch.from_dlpack(
                wrap_dlpack(tensor), assumed_align=assumed_align,
            )
            if force_static_layout:
                return cute_tensor
            leading_dim = cutlass_torch.get_leading_dim(tensor)
            return cute_tensor.mark_layout_dynamic(leading_dim=leading_dim)

        kernel_activation = (
            self._transport.activation_rx
            if self.comm_backend == "gin"
            else self.my_activation.view(torch.uint8)
        )
        kernel_activation_sf = (
            self._transport.sf_rx
            if self.comm_backend == "gin"
            else self.my_activation_sf
        )
        kernel_topk_weights = (
            self._transport.topk_weight_rx
            if self.comm_backend == "gin"
            else self.my_topk_weights
        )
        activation_cute = _to_cute(kernel_activation)
        activation_sf_cute = _to_cute(kernel_activation_sf)
        topk_idx_cute = _to_cute(self.my_topk_idx)
        topk_weights_cute = _to_cute(kernel_topk_weights)
        fc1_weight_cute = _to_cute(self.my_fc1_weight)
        fc1_weight_sf_cute = _to_cute(self.my_fc1_weight_sf)
        fc2_weight_cute = _to_cute(self.my_fc2_weight)
        fc2_weight_sf_cute = _to_cute(self.my_fc2_weight_sf)
        fc1_alpha_cute = _to_cute(self.my_fc1_alpha, assumed_align=4)
        fc2_alpha_cute = _to_cute(self.my_fc2_alpha, assumed_align=4)
        fc1_norm_const_cute = _to_cute(self.my_fc1_norm_const, assumed_align=4)
        combine_output_cute = _to_cute(self.combine_output)
        rank_combine_ready_cute = (
            _to_cute(self.rank_combine_ready, assumed_align=4)
            if self.rank_combine_ready is not None
            else None
        )
        local_workspace_cute = _to_cute(self.local_workspace, force_static_layout=True)
        shared_workspace_cute = _to_cute(self.shared_workspace)
        green_trace_cute = (
            _to_cute(self._split_green_trace, force_static_layout=True)
            if self._split_green_trace is not None
            else None
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        if self.split_launch == "green_graph":
            self._graph_root_stream = torch.cuda.Stream(priority=0)
            self._k1_torch_stream = torch.cuda.Stream(priority=-1)
            self._k2_torch_stream = torch.cuda.Stream(priority=0)
            stream = cuda.CUstream(self._graph_root_stream.cuda_stream)
            k1_stream = cuda.CUstream(self._k1_torch_stream.cuda_stream)
            k2_stream = cuda.CUstream(self._k2_torch_stream.cuda_stream)
        else:
            k1_stream = stream
            k2_stream = stream

        if self._native_green_resources is not None:
            (
                self._transport.dispatch_send_stream,
                self._transport.dispatch_recv_stream,
                self._transport.ring_stream,
                self._transport.ring_recv_stream,
                self._transport.owner_plan_stream,
            ) = self._native_green_resources.comm_streams

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
            combine_output=combine_output_cute,
            combine_ready_flags=rank_combine_ready_cute,
            fc2_block_done_counter=None,
            local_workspace=local_workspace_cute,
            shared_workspace=shared_workspace_cute,
            green_trace=green_trace_cute,
            peer_rank_ptr_mapper_host=peer_rank_ptr_mapper_host,
            stream=stream,
        )
        self._runtime_kwargs_k1 = dict(runtime_kwargs)
        self._runtime_kwargs_k1["stream"] = k1_stream
        self._runtime_kwargs_k2 = dict(runtime_kwargs)
        self._runtime_kwargs_k2["stream"] = k2_stream
        if self.k2_tail_reclaim:
            self._runtime_kwargs_k2_drain = dict(runtime_kwargs)
            self._runtime_kwargs_k2_drain["stream"] = k1_stream
        if self._kernel_k2_finalizer is not None:
            self._runtime_kwargs_k2_finalizer = dict(runtime_kwargs)
            self._runtime_kwargs_k2_finalizer["stream"] = stream

        compile_kwargs_k1 = dict(self._runtime_kwargs_k1)
        compile_kwargs_k1["max_active_clusters"] = k1_active_clusters
        compile_kwargs_k2 = dict(self._runtime_kwargs_k2)
        compile_kwargs_k2["max_active_clusters"] = k2_active_clusters
        compile_kwargs_k2_drain = None
        if self.k2_tail_reclaim:
            compile_kwargs_k2_drain = dict(self._runtime_kwargs_k2_drain)
            compile_kwargs_k2_drain["max_active_clusters"] = (
                k2_drain_launch_clusters
            )
        compile_kwargs_k2_finalizer = None
        if self._runtime_kwargs_k2_finalizer is not None:
            compile_kwargs_k2_finalizer = dict(
                self._runtime_kwargs_k2_finalizer
            )
            compile_kwargs_k2_finalizer["max_active_clusters"] = 1
        compile_options = []
        if self.misc.enable_iket:
            compile_options.append("iket")
        if compile_options:
            options = " ".join(compile_options)
            compile_kwargs_k1["options"] = options
            compile_kwargs_k2["options"] = options
            if compile_kwargs_k2_drain is not None:
                compile_kwargs_k2_drain["options"] = options
            if compile_kwargs_k2_finalizer is not None:
                compile_kwargs_k2_finalizer["options"] = options
        # Compile the small device-RMA translation units first.  Compiling
        # them after the very large persistent K1/K2 modules can exhaust the
        # current CuTe/LLVM process and terminate inside device-bitcode link.
        if self.comm_backend == "gin":
            self._compile_ibgda_gpu_push_kernels(cute)
        if self.rank == 0:
            print("[compile timing] K1 begin", flush=True)
        compile_k1_start = time.perf_counter()
        self._compiled_kernel_k1 = cute.compile(
            self._kernel_k1, **compile_kwargs_k1
        )
        if self.rank == 0:
            print(
                "[compile timing] K1 done "
                f"{time.perf_counter() - compile_k1_start:.3f}s",
                flush=True,
            )
            print("[compile timing] K2 begin", flush=True)
        compile_k2_start = time.perf_counter()
        self._compiled_kernel_k2 = cute.compile(
            self._kernel_k2, **compile_kwargs_k2
        )
        if self.k2_tail_reclaim:
            self._compiled_kernel_k2_drain = cute.compile(
                self._kernel_k2_drain, **compile_kwargs_k2_drain
            )
        if self._kernel_k2_finalizer is not None:
            self._compiled_kernel_k2_finalizer = cute.compile(
                self._kernel_k2_finalizer,
                **compile_kwargs_k2_finalizer,
            )
        if self.rank == 0:
            print(
                "[compile timing] K2 done "
                f"{time.perf_counter() - compile_k2_start:.3f}s",
                flush=True,
            )
        self._compiled_topk_reduce = None
        topk_reduce_runtime_kwargs = None
        include_topk_reduce = (
            self.rank_local_combine
            or not self.misc.skip_ref_check
            or os.environ.get("MEGA_SPLIT_INCLUDE_K3", "0") == "1"
        )
        if include_topk_reduce:
            if self.combine_reduced_output is None:
                raise RuntimeError(
                    "form A CuTeDSL reduce requires combine_reduced_output allocation."
                )
            combine_reduced_output_cute = _to_cute(self.combine_reduced_output)
            topk_reduce_score = (
                self.my_topk_weights
                if self.misc.ref_compute_graph == "transformers"
                else None
            )
            topk_reduce_score_cute = (
                topk_weights_cute
                if self.misc.ref_compute_graph == "transformers"
                else None
            )
            topk_runtime_stream = (
                cuda.CUstream(torch.cuda.current_stream().cuda_stream)
                if self.comm_backend == "gin"
                else stream
            )
            if self.rank_local_combine:
                if self.rank_combine_ready is None:
                    raise RuntimeError(
                        "rank-local combine ready storage is missing"
                    )
                from moe_sm120_nvfp4_split.kernel_rank_local_combine import (
                    compile_rank_local_combine,
                )

                (
                    self._compiled_topk_reduce,
                    topk_reduce_runtime_kwargs,
                ) = compile_rank_local_combine(
                    self.combine_output,
                    self.rank_combine_ready,
                    self.combine_reduced_output,
                    self.my_topk_idx,
                    world_size=self.world_size,
                    num_experts_per_rank=self.problem.num_experts_per_rank,
                    stream=topk_runtime_stream,
                )
            elif self._tp_k3_chunk_count > 1:
                if (
                    self.problem.num_tokens_per_rank
                    % self._tp_k3_chunk_count
                    != 0
                ):
                    raise ValueError(
                        "tokens/rank must be divisible by TP K3 chunks: "
                        f"{self.problem.num_tokens_per_rank} vs "
                        f"{self._tp_k3_chunk_count}."
                    )
                combine_chunks = self.combine_output.chunk(
                    self._tp_k3_chunk_count, dim=0
                )
                reduced_chunks = self.combine_reduced_output.chunk(
                    self._tp_k3_chunk_count, dim=0
                )
                score_chunks = (
                    topk_reduce_score.chunk(self._tp_k3_chunk_count, dim=0)
                    if topk_reduce_score is not None
                    else (None,) * self._tp_k3_chunk_count
                )
                caller_stream = cuda.CUstream(
                    torch.cuda.current_stream().cuda_stream
                )
                topk_reduce_plan = compile_topk_reduce(
                    combine_chunks[0],
                    reduced_chunks[0],
                    score_chunks[0],
                    stream=caller_stream,
                    enable_iket=False,
                )
                self._compiled_topk_reduce = topk_reduce_plan[0]
                self._tp_k3_chunk_runtime_kwargs = []
                for combine_chunk, reduced_chunk, score_chunk in zip(
                    combine_chunks,
                    reduced_chunks,
                    score_chunks,
                ):
                    self._tp_k3_chunk_runtime_kwargs.append(
                        dict(
                            combine_cute=_to_cute(combine_chunk),
                            reduced_cute=_to_cute(reduced_chunk),
                            topk_score_cute=(
                                _to_cute(score_chunk)
                                if score_chunk is not None
                                else None
                            ),
                            stream=caller_stream,
                        )
                    )
                self._tp_k3_chunk_outputs = reduced_chunks
                # A non-None plan tells the launch path that K3 is enabled;
                # chunked K3 itself is deliberately kept outside the Green
                # graph so NCCL can consume each completed token chunk.
                topk_reduce_runtime_kwargs = (
                    self._tp_k3_chunk_runtime_kwargs[0]
                )
            else:
                topk_reduce_plan = compile_topk_reduce(
                    self.combine_output,
                    self.combine_reduced_output,
                    topk_reduce_score,
                    stream=topk_runtime_stream,
                    enable_iket=False,
                )
                self._compiled_topk_reduce = topk_reduce_plan[0]
                topk_reduce_runtime_kwargs = dict(
                    combine_cute=combine_output_cute,
                    reduced_cute=combine_reduced_output_cute,
                    topk_score_cute=topk_reduce_score_cute,
                    stream=topk_runtime_stream,
                )

        if self.split_launch == "green_graph":
            from moe_sm120_nvfp4_split.runtime.green_context import (
                NativeGreenContextGraph,
            )

            k1_executor = self._compiled_kernel_k1.to(None)
            k2_executor = self._compiled_kernel_k2.to(None)
            k2_drain_executor = (
                self._compiled_kernel_k2_drain.to(None)
                if self.k2_tail_reclaim
                else None
            )
            k2_finalizer_executor = (
                self._compiled_kernel_k2_finalizer.to(None)
                if self._compiled_kernel_k2_finalizer is not None
                else None
            )
            topk_executor = (
                self._compiled_topk_reduce.to(None)
                if self._compiled_topk_reduce is not None
                and self._tp_k3_chunk_runtime_kwargs is None
                and self.comm_backend != "gin"
                else None
            )
            graph_topk_runtime_kwargs = topk_reduce_runtime_kwargs
            self._native_green_graph = NativeGreenContextGraph.capture(
                root_stream=self._graph_root_stream,
                k1_stream=self._k1_torch_stream,
                k2_stream=self._k2_torch_stream,
                k1_executor=k1_executor,
                k2_executor=k2_executor,
                k2_drain_executor=k2_drain_executor,
                k2_finalizer_executor=k2_finalizer_executor,
                k3_executor=topk_executor,
                compiled_role_owners={0: (self._compiled_kernel_k1, self._compiled_kernel_k2_drain), 1: (self._compiled_kernel_k2,)},
                compiled_kernels=tuple(
                    compiled for compiled in (
                        self._compiled_kernel_k1,
                        self._compiled_kernel_k2,
                        self._compiled_kernel_k2_drain,
                        self._compiled_kernel_k2_finalizer,
                        self._compiled_topk_reduce,
                    ) if compiled is not None
                ),
                launch_k1=lambda: k1_executor(**self._runtime_kwargs_k1),
                launch_k2=lambda: k2_executor(**self._runtime_kwargs_k2),
                launch_k2_drain=(
                    (
                        lambda: k2_drain_executor(
                            **self._runtime_kwargs_k2_drain
                        )
                    )
                    if k2_drain_executor is not None
                    else None
                ),
                launch_k2_finalizer=(
                    lambda: k2_finalizer_executor(
                        **self._runtime_kwargs_k2_finalizer
                    )
                    if k2_finalizer_executor is not None
                    else None
                ),
                launch_k3=(
                    (
                        lambda: topk_executor(
                            **graph_topk_runtime_kwargs
                        )
                    )
                    if topk_executor is not None
                    else None
                ),
                launch_reset=(
                    self._reset_local_counters
                    if (
                        self.k2_tail_reclaim
                        or k2_finalizer_executor is not None
                    )
                    else None
                ),
                k1_sm_count=self.kernel_config.k1_sms,
                k1_grid_clusters=k1_active_clusters,
                k2_grid_clusters=k2_active_clusters,
                k2_drain_grid_clusters=(
                    k2_drain_launch_clusters
                    if k2_drain_executor is not None
                    else None
                ),
                green_resources=self._native_green_resources,
            )
            if (
                self._native_green_graph.k1_sm_count
                != k1_active_clusters
                or self._native_green_graph.k2_sm_count
                != k2_partition_clusters
            ):
                raise RuntimeError(
                    "Green Context partition changed between query and graph "
                    "instantiation."
                )

        # This rendezvous is intentionally outside CUDA-event timing.  Normal
        # runs previously launched as soon as one rank finished JIT/capture;
        # if another rank failed or was interrupted, the early rank entered an
        # unbounded device barrier and occupied its GPU indefinitely.
        self._host_control_barrier("post-compile/graph-capture")

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
        #       communication cross-rank ops for slower ranks to reach the
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
        #       communication-session teardown / etc.).  Without this, a fast rank
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
        if os.environ.get("MEGA_POSTCOMPILE_EVENT_BENCH", "0") == "1":
            self._use_cuda_events = True
            self._perf_warmup = int(
                os.environ.get("MEGA_POSTCOMPILE_WARMUP", "10")
            )
            self._perf_iters = int(
                os.environ.get("MEGA_POSTCOMPILE_ITERS", "100")
            )
        self._device_launch_completed = False
        self._device_launch_started = True
        if self.misc.profile_friendly:
            import nvtx

            torch.cuda.synchronize()
            _dist_active = (
                torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
            if _dist_active:
                self._ep_barrier()
                torch.cuda.synchronize()
            with nvtx.annotate("cute_dsl_prof"):
                self._launch_target_kernels_with_optional_torch_profiler(
                    runtime_kwargs,
                    topk_reduce_runtime_kwargs,
                )
            if _dist_active:
                self._ep_barrier()
                torch.cuda.synchronize()
        else:
            self._launch_target_kernels_with_optional_torch_profiler(
                runtime_kwargs,
                topk_reduce_runtime_kwargs,
            )
        self._device_launch_completed = True
        self._report_split_green_trace()

    # ------------------------------------------------------------------
    # Step 5: validation
    # ------------------------------------------------------------------

    def _validate_token_back_exact(self) -> dict:
        """Reconstruct combine from every owner's local FC2 workspace.

        This is the multi-rank communication oracle: metadata and FC2 rows are
        gathered after the fused kernel, then replayed onto the source rank's
        ``(token, topk)`` cells.  Token-back is a raw BF16 copy in form A, so
        the reconstructed tensor must be byte-exact.
        """
        if self._kernel is None or self.local_workspace is None:
            raise RuntimeError("token-back validation requires a completed kernel")
        if self.combine_output is None or self._global_topk_idx is None:
            raise RuntimeError("token-back validation requires generated inputs")

        if self.rank_local_combine:
            return rank_local_staging_report(self)

        local_by_name = self._kernel._local_region_by_name
        if "fc2_output_workspace" not in local_by_name:
            if self.rank == 0:
                print(
                    "Token-back workspace replay SKIPPED: "
                    f"token_back_mode={self.impl.token_back_mode} writes "
                    "FC2 results directly to combine_output and does not "
                    "allocate fc2_output_workspace"
                )
            return {
                "status": "NOT_CHECKED",
                "reason": "direct peer-store path has no independent pre-store staging",
            }

        local_offsets = self._kernel._local_offsets
        fc2_spec = local_by_name["fc2_output_workspace"]
        fc2_off = local_offsets["fc2_output_workspace"]
        md_spec = local_by_name["token_src_metadata"]
        md_off = local_offsets["token_src_metadata"]

        local_fc2 = self.local_workspace[
            fc2_off:fc2_off + fc2_spec.nbytes
        ].view(self.problem.fc2_output_dtype).reshape(fc2_spec.shape).contiguous()
        local_md = self.local_workspace[
            md_off:md_off + md_spec.nbytes
        ].view(torch.int64).contiguous()

        all_fc2 = [torch.empty_like(local_fc2) for _ in range(self.world_size)]
        all_md = [torch.empty_like(local_md) for _ in range(self.world_size)]
        if self.world_size == 1:
            all_fc2[0].copy_(local_fc2)
            all_md[0].copy_(local_md)
        else:
            self._ep_all_gather(all_fc2, local_fc2)
            self._ep_all_gather(all_md, local_md)

        expected = torch.zeros_like(self.combine_output)
        if self._kernel.ibgda_k2_direct_staging:
            local_send_rows = self._transport.ring_send_output_rows
            local_send_counts = self._transport.ring_send_counts
            all_send_rows = [
                torch.empty_like(local_send_rows)
                for _ in range(self.world_size)
            ]
            all_send_counts = [
                torch.empty_like(local_send_counts)
                for _ in range(self.world_size)
            ]
            if self.world_size == 1:
                all_send_rows[0].copy_(local_send_rows)
                all_send_counts[0].copy_(local_send_counts)
            else:
                self._ep_all_gather(all_send_rows, local_send_rows)
                self._ep_all_gather(all_send_counts, local_send_counts)
            expected_flat = expected.view(
                self.problem.num_tokens_per_rank * self.problem.num_topk,
                self.problem.hidden,
            )
            stage_base = self.rank * self._transport.max_pair_routes
            for owner in range(self.world_size):
                count = int(all_send_counts[owner][self.rank].item())
                if count == 0:
                    continue
                output_rows = all_send_rows[owner][self.rank, :count].to(
                    torch.int64
                )
                expected_flat[output_rows] = all_fc2[owner][
                    stage_base:stage_base + count, 0, :
                ]
        else:
            routes = self._global_topk_idx
            experts_per_rank = self.problem.num_experts_per_rank
            token_padding = self._kernel.token_padding_block
            for owner in range(self.world_size):
                pool_ids = []
                cursor = 0
                for local_expert in range(experts_per_rank):
                    global_expert = owner * experts_per_rank + local_expert
                    count = int((routes == global_expert).sum().item())
                    pool_ids.extend(range(cursor, cursor + count))
                    cursor += round_up(count, token_padding)
                if not pool_ids:
                    continue
                ids = torch.tensor(
                    pool_ids, dtype=torch.int64, device=local_md.device
                )
                packed = all_md[owner][ids]
                hi = packed >> 32
                src_rank = (hi >> 16) & 0xFFFF
                src_token = packed & 0xFFFFFFFF
                src_topk = hi & 0xFFFF
                mine = src_rank == self.rank
                expected[src_token[mine], src_topk[mine], :] = all_fc2[
                    owner
                ][ids[mine], 0, :]

        byte_diff = int(
            (expected.view(torch.uint8) != self.combine_output.view(torch.uint8))
            .sum().item()
        )
        local_pass = byte_diff == 0
        pass_tensor = torch.tensor(
            [int(local_pass)], dtype=torch.int32, device=self.combine_output.device
        )
        diff_tensor = torch.tensor(
            [byte_diff], dtype=torch.int64, device=self.combine_output.device
        )
        if self.world_size > 1:
            pass_all = [torch.empty_like(pass_tensor) for _ in range(self.world_size)]
            diff_all = [torch.empty_like(diff_tensor) for _ in range(self.world_size)]
            self._ep_all_gather(pass_all, pass_tensor)
            self._ep_all_gather(diff_all, diff_tensor)
            rank_passes = [bool(v.item()) for v in pass_all]
            rank_diffs = [int(v.item()) for v in diff_all]
        else:
            rank_passes = [local_pass]
            rank_diffs = [byte_diff]

        return {
            "status": "PASS" if all(rank_passes) else "FAIL",
            "source": "actual_FC2_workspace", "rank_byte_differences": rank_diffs,
        }

    def _agree_combine_report(self, report: dict) -> None:
        """Print distinct verdicts, then agree before any EP rank raises."""
        for name in ("grouped_implementation", "ordinary_topk", "staging_transport", "grouped_arithmetic"):
            report.setdefault(name, {
                "status": "NOT_CHECKED", "reason": "not requested in this validation call",
            })
        strict = os.environ.get("MEGA_STRICT_TORCH_REF", "0") == "1"
        report["strict_ordinary_topk"] = strict
        self.last_combine_validation_report = report
        print(f"[rank {self.rank}] COMBINE_VALIDATION "
              + json.dumps(report, sort_keys=True, allow_nan=False), flush=True)
        names = ("grouped_implementation", "ordinary_topk", "staging_transport", "grouped_arithmetic")
        flags = torch.tensor(
            [int(report.get(name, {}).get("status") == "FAIL") for name in names],
            device=self.output_tensor.device, dtype=torch.int32,
        )
        gathered = [torch.empty_like(flags) for _ in range(self.world_size)]
        if self.world_size == 1:
            gathered[0].copy_(flags)
        else:
            self._ep_all_gather(gathered, flags)
        verdicts = [
            {name: {"status": "FAIL" if value else "NOT_CHECKED"}
             for name, value in zip(names, row.tolist())}
            for row in gathered
        ]
        enforce_combine_reports(verdicts, strict=strict)

    def validate(self, *, force: bool = False) -> None:
        """Validate the public BF16 ``(tokens, hidden)`` output after K3."""
        if self.misc.skip_ref_check and not force:
            return
        if self.combine_output is None:
            raise RuntimeError("validate requires run_kernel first.")
        if self.combine_output_ref is None:
            raise RuntimeError("validate requires compute_reference first.")
        if self.combine_reduced_output_ref is None:
            raise RuntimeError("validate requires reduced reference output.")
        if self.output_tensor is None:
            raise RuntimeError("validate requires the public reduced output.")
        if self.output_tensor_ref is None:
            raise RuntimeError("validate requires the reduced reference output.")

        if self.rank_local_combine and self.rank_local_output_ref is None:
            raise RuntimeError("RLC validation requires the independent grouped reference")
        # All three outcomes are reported before applying numerical strictness.
        transport = (rank_local_staging_report(self) if self.rank_local_combine
                     else self._validate_token_back_exact())
        report = make_combine_report(
            self.output_tensor, self.output_tensor_ref,
            grouped_reference=self.rank_local_output_ref if self.rank_local_combine else None,
            transport=transport,
            independent_grouped_arithmetic=self.rank_local_combine,
        )
        self._agree_combine_report(report)

    # ------------------------------------------------------------------
    # Top-level orchestration
    # ------------------------------------------------------------------

    def run(self) -> None:
        if self.rank == 0:
            print(self.problem)
            print(
                "ParallelDesc: "
                f"DP={self.data_parallel_size} "
                f"TP={self.tensor_parallel_size} "
                f"EP={self.world_size} "
                f"full_intermediate="
                f"{self.problem.intermediate * self.tensor_parallel_size} "
                f"tp_intermediate_shard={self.problem.intermediate}"
            )
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
                ("combine_output", self.combine_output),
            ):
                print(
                    f"[rank {self.rank}] {name:18s} "
                    f"shape={tuple(tensor.shape)} dtype={tensor.dtype}"
                )

        skip_ref = self.misc.skip_ref_check or self.misc.profile_friendly
        # A dynamic replay replaces topk_idx before every launch; build its
        # reference from the final GPU-resident route after the timed loop.
        if not skip_ref and not self._dynamic_route_replays:
            self.compute_reference()

        # Once the kernel is wired, this is also the place ``allocate_workspaces``
        # gets called (depends on self._kernel being instantiated inside run_kernel).
        self.run_kernel()

        if self._dynamic_route_replays or self._transport_replay_validation:
            gathered_routes = [
                torch.empty_like(self.my_topk_idx)
                for _ in range(self.world_size)
            ]
            self._ep_all_gather(gathered_routes, self.my_topk_idx)
            self._global_topk_idx.copy_(torch.stack(gathered_routes))

        if self._dynamic_route_replays and not skip_ref:
            self.compute_reference(force=True)

        if self._transport_replay_validation and skip_ref:
            transport = self._validate_token_back_exact()
            self._agree_combine_report({"staging_transport": transport})

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
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    }
    if argument not in mapping:
        raise argparse.ArgumentTypeError(
            f"fc2_output_dtype must be one of {sorted(mapping.keys())}, got {argument!r}"
        )
    return mapping[argument]


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MegaMoE SM120 NVFP4 x NVFP4 split dispatch+fc12+combine runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--num_tokens_per_rank",
        type=int,
        default=128,
        help="Token rows already arranged for the caller's TP mapping and "
             "passed to each EP rank. MegaMoE does not exchange TP inputs.",
    )
    parser.add_argument("--num_topk", type=int, default=4)
    parser.add_argument("--num_total_experts", type=int, default=32)
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=1,
        help="Number of data-parallel expert replicas. Each DP/TP replica "
             "owns the complete expert-id space.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Intermediate-dimension tensor-parallel size. EP is derived as "
             "WORLD_SIZE / (DP * TP).",
    )
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1024)
    parser.add_argument(
        "--fc2_output_dtype", type=_parse_output_dtype, default=torch.bfloat16,
    )
    parser.add_argument("--tuning_config", default=None,
                        help="Qualified offline JSON; incompatible entries use the normal selector")
    parser.add_argument(
        "--route_distribution", type=str, default="balanced",
        choices=["balanced", "power_law", "zero_count_peer"],
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

    parser.add_argument(
        "--mma_tiler_mnk",
        type=str,
        default=None,
        help="K1 CTA tile. By default, select M64 x N32/N64/N128 x K128 "
             "from the expected routed rows per expert.",
    )
    parser.add_argument("--enable_static_expert_shape", action="store_true", default=False)
    parser.add_argument(
        "--group_hint",
        type=int,
        default=None,
        help="FC1 tiles per scheduler group; defaults to max_active_clusters.",
    )
    parser.add_argument("--flag_batch", type=int, default=4)
    parser.add_argument(
        "--epi_flag_batch", type=str, default="1,1",
        help="(fc1,fc2) done-counter publish batch in comma form",
    )
    parser.add_argument(
        "--token_back_mode",
        type=str,
        default=None,
        choices=["epi_warps", "reuse_dispatch_warps"],
        help="Where the cross-rank fc2 push-back runs: epi_warps (epilogue "
             "STG redirect) or reuse_dispatch_warps (IBGDA combine sender).",
    )
    parser.add_argument(
        "--comm_backend",
        type=str,
        default=None,
        choices=["p2p_direct", "gin", "gin_hybrid"],
        help="GPU-heap communication backend. gin disables peer "
             "mappings and routes every remote PE through GPU-initiated IB; "
             "gin_hybrid keeps same-NUMA P2P mappings and masks only "
             "cross-NUMA peers so their device RMAs use IBGDA.",
    )
    parser.add_argument("--perf_run", action="store_true", default=False)
    parser.add_argument("--skip_ref_check", action="store_true", default=False)
    parser.add_argument("--profile_friendly", action="store_true", default=False)
    parser.add_argument("--use_torch_profiler", action="store_true", default=False)
    parser.add_argument("--use_cuda_events", action="store_true", default=False)
    # Default to 0,1 to avoid testing time.
    parser.add_argument("--perf_warmup", type=int, default=0)
    parser.add_argument("--perf_iters", type=int, default=1)
    parser.add_argument(
        "--dynamic_route_replays",
        type=int,
        default=0,
        help="Replace --perf_iters and regenerate the GPU-resident top-k "
             "route before every timed replay (performance/stress mode).",
    )
    parser.add_argument(
        "--transport_replay_validation",
        action="store_true",
        default=False,
        help="After replay stress, gather the final GPU-resident route and "
             "run byte-exact dispatch/combine reconstruction.",
    )
    parser.add_argument(
        "--verify_replay_bit_determinism",
        action="store_true",
        default=False,
        help="Require combine_output to be byte-identical across static-route "
             "performance replays (debug/stress only).",
    )
    parser.add_argument("--enable_debug_checks", action="store_true", default=False)
    parser.add_argument(
        "--ref_compute_graph", type=str, default="deepgemm",
        choices=["transformers", "deepgemm"],
    )
    parser.add_argument("--enable_iket", action="store_true", default=False)
    parser.add_argument(
        "--split_launch",
        choices=["sequential", "green_graph"],
        default="green_graph",
        help="Launch K1/K2 serially for debugging, or through the production "
             "native CUDA Graph whose K1/K2 nodes are bound to disjoint "
             "Green Contexts.",
    )
    parser.add_argument("--k1_sms", type=int, default=None)
    parser.add_argument("--k2_sms", type=int, default=None)
    parser.add_argument("--tx_sms", type=int, default=None)
    parser.add_argument("--rx_sms", type=int, default=None)
    parser.add_argument(
        "--k2_warp_count",
        type=int,
        choices=[8, 12],
        default=None,
        help="Select compact K2 (8) or the N16 dual-N8 specialization (12).",
    )
    parser.add_argument(
        "--k2_mma_tiler_mnk",
        type=str,
        default=None,
        help="Independent lean K2 CTA tile. By default, use the same "
             "routed-rows heuristic as K1; K1 keeps --mma_tiler_mnk.",
    )
    parser.add_argument(
        "--k2_num_ab_stages",
        type=int,
        choices=[1, 2, 3],
        default=None,
        help="Independent lean K2 TMA A/B pipeline depth. By default, use "
             "the shape- and transport-dependent heuristic.",
    )
    parser.add_argument("--dispatch_chunk_tokens", type=int, default=None)
    parser.add_argument("--combine_chunk_rows", type=int, default=None)
    parser.add_argument("--dispatch_channels", type=int, default=None)
    parser.add_argument("--combine_channels", type=int, default=None)
    parser.add_argument("--dispatch_slots", type=int, default=None)
    parser.add_argument("--combine_slots", type=int, default=None)
    parser.add_argument("--dispatch_local_handoff_windows", type=int, default=None)
    parser.add_argument("--dispatch_remote_handoff_windows", type=int, default=None)
    parser.add_argument("--ibgda_rc_per_pe", type=int, default=None)
    parser.add_argument(
        "--ibgda_rc_mapping", choices=["warp", "cta", "sm"], default=None
    )
    parser.add_argument(
        "--k2_ready_queue_bundle",
        type=int,
        default=None,
        help="K2 scheduler descriptors published per ready-queue work item; "
             "omitted uses the pure heuristic.",
    )
    parser.add_argument(
        "--k2_tail_reclaim",
        choices=["on", "off"],
        default=None,
        help="Benchmark-only override for K2 workers launched on the K1 "
             "partition after K1 retires.",
    )
    parser.add_argument("--tp_k3_chunks", type=int, default=None)
    parser.add_argument(
        "--dispatch_compute_overlap", choices=["on", "off"], default=None
    )
    parser.add_argument(
        "--dispatch_warps",
        type=int,
        choices=[1, 2, 4],
        default=None,
        help="Active K1 dispatch warps per CTA; omitted uses the heuristic.",
    )
    parser.add_argument(
        "--dispatch_rank_cache",
        choices=["on", "off"],
        default=None,
        help="Cache one pulled activation row per (source rank, token).",
    )
    parser.add_argument(
        "--rank_local_combine",
        choices=["on", "off"],
        default=None,
        help="Override the heuristic that reduces routes within each owner "
             "rank before token-back. This inserts an intermediate BF16 "
             "rounding step and can differ from top-k-slot accumulation.",
    )
    parser.add_argument(
        "--numerical_contract", choices=["ordinary_topk", "owner_bf16"],
        default="ordinary_topk",
        help="Explicitly permit owner BF16 intermediate rounding before RLC selection.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Bootstrap NCCL control, GIN/LSA memory, then run the tester.

    Launcher::

        torchrun --nproc_per_node=4 -m moe_sm120_nvfp4_split.mega_runner \\
            --num_total_experts 32 --route_distribution balanced
    """
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    data_parallel_size = args.data_parallel_size
    tensor_parallel_size = args.tensor_parallel_size
    global_world_size = 1 if _NO_DIST else int(os.environ["WORLD_SIZE"])
    parallel_product = data_parallel_size * tensor_parallel_size
    if global_world_size % parallel_product:
        parser.error(
            f"WORLD_SIZE={global_world_size} must be divisible by "
            f"DP*TP={parallel_product}."
        )
    ep_size = global_world_size // parallel_product
    if args.intermediate % tensor_parallel_size:
        parser.error(
            f"intermediate={args.intermediate} must be divisible by "
            f"tensor_parallel_size={tensor_parallel_size}."
        )
    parallel_context = None
    if _NO_DIST:
        torch.cuda.set_device(0)
        if data_parallel_size != 1 or tensor_parallel_size != 1:
            raise ValueError(
                "DP/TP sizes greater than one require a distributed "
                "torchrun launch."
            )
        rank = 0
        world_size = 1
        same_numa_remote_peers = 0
        cross_numa_remote_peers = 0
    else:
        from moe_sm120_nvfp4_split.runtime.parallelism import (
            init_nccl_memory,
            init_parallel_context,
        )

        # Build the control/process groups first, then discover the physical
        # PCI/NUMA layout.  communication must not initialize until that topology has
        # selected the transport preset and environment below.
        parallel_context = init_parallel_context(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
        )
        topology = parallel_context.ep_transport_topology
        same_numa_remote_peers = topology.same_numa_peer_count
        cross_numa_remote_peers = topology.cross_numa_peer_count
        rank = parallel_context.ep_rank
        world_size = parallel_context.ep_size

    from moe_sm120_nvfp4_split.runtime.green_context import query_sm_resource_info

    (
        device_sm_count,
        sm_min_partition,
        sm_partition_alignment,
    ) = query_sm_resource_info()
    from moe_sm120_nvfp4_split.tuning_config import load_tuning_config, hardware_identity
    offline_tuning = load_tuning_config(args.tuning_config) if args.tuning_config else None
    offline_hardware = hardware_identity() if offline_tuning is not None else None
    heuristic_input = MegaMoEHeuristicInput(
        tokens_per_rank=args.num_tokens_per_rank,
        hidden=args.hidden,
        intermediate=args.intermediate // tensor_parallel_size,
        num_topk=args.num_topk,
        num_total_experts=args.num_total_experts,
        data_parallel_size=data_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=ep_size,
        ep_same_numa_peer_count=same_numa_remote_peers,
        ep_cross_numa_peer_count=cross_numa_remote_peers,
        num_sms=device_sm_count,
        sm_min_partition=sm_min_partition,
        sm_partition_alignment=sm_partition_alignment,
    )
    heuristic_overrides = MegaMoEHeuristicOverrides(
        comm_backend=args.comm_backend,
        token_back_mode=args.token_back_mode,
        k1_tile=(
            _parse_tuple(args.mma_tiler_mnk)
            if args.mma_tiler_mnk is not None else None
        ),
        k2_tile=(
            _parse_tuple(args.k2_mma_tiler_mnk)
            if args.k2_mma_tiler_mnk is not None else None
        ),
        k2_stages=args.k2_num_ab_stages,
        k2_warps=args.k2_warp_count,
        k2_tail_reclaim=(
            args.k2_tail_reclaim == "on"
            if args.k2_tail_reclaim is not None else None
        ),
        k1_sms=args.k1_sms,
        k2_sms=args.k2_sms,
        tx_sms=args.tx_sms,
        rx_sms=args.rx_sms,
        ready_queue_bundle=args.k2_ready_queue_bundle,
        dispatch_chunk_tokens=args.dispatch_chunk_tokens,
        combine_chunk_rows=args.combine_chunk_rows,
        dispatch_channels=args.dispatch_channels,
        combine_channels=args.combine_channels,
        dispatch_slots=args.dispatch_slots,
        combine_slots=args.combine_slots,
        dispatch_local_handoff_windows=args.dispatch_local_handoff_windows,
        dispatch_remote_handoff_windows=args.dispatch_remote_handoff_windows,
        ibgda_rc_per_pe=args.ibgda_rc_per_pe,
        ibgda_rc_mapping=args.ibgda_rc_mapping,
        tp_k3_chunks=args.tp_k3_chunks,
        dispatch_warps=args.dispatch_warps,
        dispatch_compute_overlap=(
            args.dispatch_compute_overlap == "on"
            if args.dispatch_compute_overlap is not None else None
        ),
        dispatch_rank_cache=(
            args.dispatch_rank_cache == "on"
            if args.dispatch_rank_cache is not None else None
        ),
        rank_local_combine=(
            args.rank_local_combine == "on"
            if args.rank_local_combine is not None else None
        ),
    )
    # Parse JIT options once and use the same execution scope guard as the API.
    # Custom launch plumbing changes resources not represented by the cost fit.
    selection_jit = Sm120JitConfig.from_environment()
    selection_dual_ctas = int(os.environ.get("MEGA_DUAL_K2_CTAS_PER_SM", "2"))
    selection_complete_pipeline = (
        not args.skip_ref_check or os.environ.get("MEGA_SPLIT_INCLUDE_K3", "0") == "1"
    )
    selection_build = SplitKernelBuildOptions(
        group_hint=args.group_hint,
        flag_batch=args.flag_batch,
        epi_flag_batch=_parse_tuple(args.epi_flag_batch),
        concurrent_k1_k2=args.split_launch == "green_graph",
    )
    select_for_execution = (
        select_megamoe_config
        if is_calibrated_execution(
            build=selection_build, jit=selection_jit,
            dual_k2_ctas_per_sm=selection_dual_ctas,
            complete_pipeline=selection_complete_pipeline,
        )
        else select_compatibility_config
    )
    profile_kwargs = (dict(tuning_config=offline_tuning, hardware=offline_hardware)
                      if select_for_execution is select_megamoe_config else {})
    kernel_config = select_for_execution(
        heuristic_input, heuristic_overrides,
        numerical_contract=args.numerical_contract, **profile_kwargs,
    )
    if args.tuning_config and parallel_context is not None:
        # Setup-only consensus: divergent files/fallbacks must not launch
        # incompatible persistent protocols on different EP ranks.
        from dataclasses import asdict
        agreed = [None] * world_size
        torch.distributed.all_gather_object(
            agreed, dict(config=asdict(kernel_config), profile=offline_tuning.sha256,
                         hardware=offline_hardware),
            group=parallel_context.ep_control_group)
        if any(item != agreed[0] for item in agreed):
            raise ValueError('EP ranks disagree on offline tuning configuration or hardware')
    if offline_tuning is not None and rank == 0:
        if profile_kwargs:
            from moe_sm120_nvfp4_split.heuristic import explain_selection
            print("Offline tuning selection:", explain_selection(
                heuristic_input, heuristic_overrides,
                numerical_contract=args.numerical_contract, **profile_kwargs)['profiles'])
        else:
            print("Offline tuning ignored: nondefault execution requires compatibility fallback")
    if rank == 0 and kernel_config.rank_local_combine:
        print(
            "WARNING: rank_local_combine is active. It rounds one BF16 "
            "partial per owner rank before K3, so output "
            "need not be bit-identical to top-k-slot accumulation."
        )
    requested_comm_backend = kernel_config.comm_backend
    hybrid_transport = requested_comm_backend == "gin_hybrid"
    ring_transport = requested_comm_backend in (
        "gin", "gin_hybrid"
    )
    if ring_transport:
        if kernel_config.token_back_mode != "reuse_dispatch_warps":
            parser.error(
                f"--comm_backend {requested_comm_backend} requires "
                "--token_back_mode reuse_dispatch_warps"
            )
    hybrid_force_ibgda_mask = 0
    if hybrid_transport:
        if parallel_context is None:
            parser.error('Network transport requires distributed topology discovery')
        hybrid_force_ibgda_mask = parallel_context.ep_transport_topology.force_ibgda_peer_mask
    if not _NO_DIST:
        parallel_context = init_nccl_memory(parallel_context, gin_bridge.p2p_capacity(
            args.num_tokens_per_rank, args.hidden, args.num_topk,
            args.num_total_experts, world_size, kernel_config.rank_local_combine))

    if args.intermediate % tensor_parallel_size != 0:
        raise ValueError(
            f"full intermediate {args.intermediate} must be divisible by "
            f"tensor_parallel_size {tensor_parallel_size}."
        )
    tp_intermediate = args.intermediate // tensor_parallel_size

    problem = TokenCommProblemDesc(
        world_size=world_size,
        num_tokens_per_rank=args.num_tokens_per_rank,
        num_topk=args.num_topk,
        num_total_experts=args.num_total_experts,
        hidden=args.hidden,
        intermediate=tp_intermediate,
        fc2_output_dtype=args.fc2_output_dtype,
        route_distribution=args.route_distribution,
        power_law_exponent=args.power_law_exponent,
        gate_up_clamp=args.gate_up_clamp,
    )

    if world_size != ep_size:
        raise RuntimeError(
            f"bootstrap EP size {world_size} differs from heuristic EP size "
            f"{ep_size}."
        )
    if rank == 0:
        print(
            "[SM120 split heuristic] "
            f"rows_per_expert={kernel_config.expected_rows_per_expert:.2f}, "
            f"backend={kernel_config.comm_backend}, "
            f"K1={kernel_config.k1_tile}, K2={kernel_config.k2_tile}, "
            f"K2_stages={kernel_config.k2_stages}, "
            f"K2_tail_reclaim={kernel_config.k2_tail_reclaim}, "
            f"dispatch_warps={kernel_config.dispatch_warps}, "
            f"rank_cache={kernel_config.dispatch_rank_cache}, "
            f"rank_combine={kernel_config.rank_local_combine}, "
            "SMs(K1/TX/RX/K2)="
            f"{kernel_config.k1_sms}/{kernel_config.tx_sms}/"
            f"{kernel_config.rx_sms}/{kernel_config.k2_sms}, "
            f"transport(rows/channels/slots/handoff_local/remote)="
            f"{kernel_config.combine_chunk_rows}/"
            f"{kernel_config.combine_channels}/"
            f"{kernel_config.combine_slots}/"
            f"{kernel_config.dispatch_local_handoff_windows}/"
            f"{kernel_config.dispatch_remote_handoff_windows}"
        )

    impl = ImplDesc(
        mma_tiler_mnk=kernel_config.k1_tile,
        enable_static_expert_shape=args.enable_static_expert_shape,
        group_hint=args.group_hint,
        token_back_mode=kernel_config.token_back_mode,
        flag_batch=args.flag_batch,
        epi_flag_batch=_parse_tuple(args.epi_flag_batch),
    )

    misc = MiscDesc(
        perf_run=args.perf_run,
        skip_ref_check=args.skip_ref_check,
        # ``profile_friendly`` is the property alias used by this runner.
        run_target_kernel_only=args.profile_friendly,
        enable_debug_checks=args.enable_debug_checks,
        ref_compute_graph=args.ref_compute_graph,
        enable_iket=args.enable_iket,
        seed=args.seed,
    )

    tester = MegaMoETester(
        problem,
        impl,
        misc,
        rank=rank,
        kernel_config=kernel_config,
        hybrid_force_ibgda_mask=hybrid_force_ibgda_mask,
        parallel_context=parallel_context,
        split_launch=args.split_launch,
        jit_config=selection_jit,
        dual_k2_ctas_per_sm=selection_dual_ctas,
    )
    tester.set_torch_profiler_enabled(args.use_torch_profiler)
    tester.set_cuda_event_timing_enabled(args.use_cuda_events)
    tester.set_perf_iters(args.perf_warmup, args.perf_iters)
    if args.dynamic_route_replays:
        if not args.perf_run:
            parser.error("--dynamic_route_replays requires --perf_run")
        tester.set_dynamic_route_replays(args.dynamic_route_replays)
    if args.transport_replay_validation or args.verify_replay_bit_determinism:
        if not args.perf_run or not args.use_cuda_events:
            parser.error(
                "replay validation requires --perf_run --use_cuda_events"
            )
        if args.verify_replay_bit_determinism and args.dynamic_route_replays:
            parser.error(
                "--verify_replay_bit_determinism requires a static route"
            )
        tester.set_transport_replay_validation(
            byte_exact=args.transport_replay_validation,
            bit_determinism=args.verify_replay_bit_determinism,
        )

    return_code = 0
    try:
        tester.run()
    except NotImplementedError as exc:
        # Expected until the MegaMoE kernel side is wired; the host
        # orchestration above is the part being smoke-tested for now.
        if rank == 0:
            print(f"[mega_runner] kernel launch skipped: {exc}")
    finally:
        tester.close_split_execution_contexts()

    if not _NO_DIST:
        # Drain device work and align ranks before process exit. Regular CLI
        # runs retain the tested fast-exit lifecycle; the profiling path below
        # performs explicit collective NCCL window/communicator teardown.
        # os._exit skips Python finalizers, hence the manual output flush.
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            # The global default group is the CPU/Gloo control plane.  A short
            # grace interval after the barrier prevents an early os._exit
            # from closing sockets while a late rank is returning from it.
            torch.distributed.barrier()
            time.sleep(0.5)
        sys.stdout.flush()
        sys.stderr.flush()
        # ``os._exit`` intentionally skips Python/native finalizers, including
        # IKET's tracker dump.  Profiling runs instead perform a collective,
        # explicitly ordered teardown and return normally so the injection
        # library can write its tracker/trace files.  Regular runs retain the
        # synchronized fast-exit behavior below.
        if os.environ.get("MEGAMOE_IKET_PROFILE_TEARDOWN") == "1":
            gin_bridge.close()
            torch.distributed.destroy_process_group()
            return return_code
        os._exit(return_code)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
