"""DSV4 configuration constants and host-side index utilities (no GPU dependencies).

Pool sizing follows mega_moe (`layout/mega_moe.cuh::get_num_max_pool_tokens` and
`get_num_padded_sf_pool_tokens`) so the workspace is large enough for the
worst-case routing distribution. The plan's `kNumPaddedSFPoolTokens=1088`
reflects an expected-case sizing only — see goal-tracker Open Issues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def _align_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _num_max_pool_tokens(
    num_ranks: int,
    num_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
    block_m: int,
) -> int:
    num_max_recv = num_ranks * num_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    raw = num_max_recv * num_max_experts_per_token + num_experts_per_rank * (
        block_m - 1
    )
    return _align_up(raw, block_m)


def _num_padded_sf_pool_tokens(
    num_max_pool_tokens: int, block_m: int, sf_block_m: int
) -> int:
    return (num_max_pool_tokens // block_m) * sf_block_m


@dataclass(frozen=True)
class DSV4Config:
    # DeepSeek V4 routing config (DeepGEMM/tests/test_mega_moe.py defaults):
    #   num_experts=384, num_topk=6, hidden=7168, intermediate_hidden=3072.
    # 4-rank EP target -> num_experts_per_rank = 384 / 4 = 96.
    #
    # `hidden` is the logical token element count (always 7168 for DSV4).
    # `hidden_bytes` is the wire-format byte count per token, which differs
    # by token dtype:
    #   FP8 dispatch:   hidden_bytes = hidden     × 1   = 7168 (default)
    #   NVFP4 dispatch: hidden_bytes = hidden / 2 × 1   = 3584 (see DSV4_nvfp4)
    # All in-kernel byte arithmetic (TMA descriptors, SMEM pull buffer, mbarrier
    # expect_tx) uses `hidden_bytes`; element-count math (sf groups, oracle
    # logical layout) uses `hidden`.
    num_ranks: int = 4
    hidden: int = 7168
    hidden_bytes: int = 7168
    num_topk: int = 6
    num_experts_per_rank: int = 96
    num_total_experts: int = 384
    block_m: int = 192
    sf_block_m: int = 256
    num_tokens_per_rank: int = 1024
    num_sms: int = 152
    sf_uint32_per_token: int = 56  # FP8 vec_size=128 -> hidden / 128 = 56
    # Downstream consumer's task-tile granularity, drives the
    # ``l1_arrival_count`` release-counter index granularity inside
    # ``_dispatch_pull``.  Decoupled from ``block_m`` (which controls pool
    # layout / TMA-store granularity only) so that the fused fc12 fc1 spin
    # sees one counter per work tile.  Default ``None`` falls back to
    # ``block_m`` via ``effective_cluster_tile_m``, recovering the legacy
    # per-pool-block counter layout for the standalone dispatch path.  Set
    # explicitly when fusing with a downstream GEMM whose task-tile
    # granularity differs from the dispatch pool granularity.  See
    # ``fc12_integrate_comm.md`` §4 (C3) for the full contract.
    cluster_tile_m: Optional[int] = None

    @property
    def effective_cluster_tile_m(self) -> int:
        return self.cluster_tile_m if self.cluster_tile_m is not None else self.block_m

    @property
    def num_max_pool_tokens(self) -> int:
        return _num_max_pool_tokens(
            self.num_ranks,
            self.num_tokens_per_rank,
            self.num_topk,
            self.num_experts_per_rank,
            self.block_m,
        )

    @property
    def num_padded_sf_pool_tokens(self) -> int:
        return _num_padded_sf_pool_tokens(
            self.num_max_pool_tokens, self.block_m, self.sf_block_m
        )

    @property
    def num_max_pool_blocks(self) -> int:
        return self.num_max_pool_tokens // self.block_m

    @property
    def num_max_task_tiles(self) -> int:
        """Sized for the release-counter array (``l1_arrival_count``).

        Upper-bound on the number of distinct ``task_tile_idx`` slots
        ``_dispatch_pull`` will release-add into across all experts:
        ``sum_e ceil_div(valid_e, cluster_tile_m)``.  We bound it by the
        same worst-case pool sizing formula used for
        ``num_max_pool_tokens`` (rounded up to ``effective_cluster_tile_m``
        granularity).
        """
        ctm = self.effective_cluster_tile_m
        return (self.num_max_pool_tokens + ctm - 1) // ctm


DSV4 = DSV4Config()

# NVFP4 dispatch variant. Same logical model (hidden=7168, topk=6, ...) but the
# token wire format is NVFP4 (4-bit data + 8-bit shared scale-factor per 16
# elements):
#   * hidden_bytes = 7168 / 2 = 3584  (4 bits/elem packed)
#   * sf_uint32_per_token = 7168 / 16 / 4 = 112  (vec_size=16, 4 SF/uint32)
# Combine still uses BF16; this only changes the dispatch-side wire bytes and
# the SF granularity (LDG loop runs 4 passes instead of 2).
DSV4_nvfp4 = DSV4Config(
    hidden_bytes=3584,
    sf_uint32_per_token=112,
)

# 8-rank EP across 2 nodes (GB200 NVL72 same-clique cross-node NVLink).
# 384 experts / 8 ranks = 48 experts per rank.
DSV4_8rank = DSV4Config(
    num_ranks=8,
    num_experts_per_rank=48,
)

DSV4_8rank_nvfp4 = DSV4Config(
    num_ranks=8,
    num_experts_per_rank=48,
    hidden_bytes=3584,
    sf_uint32_per_token=112,
)

# 6-rank EP across 2 nodes (e.g. 3 GPUs/node * 2 nodes). 384/6 = 64.
DSV4_6rank = DSV4Config(
    num_ranks=6,
    num_experts_per_rank=64,
)

DSV4_6rank_nvfp4 = DSV4Config(
    num_ranks=6,
    num_experts_per_rank=64,
    hidden_bytes=3584,
    sf_uint32_per_token=112,
)

# Lookup table for CLI / test parametrization. Keep keys stable; scripts read
# them from --config.
DSV4_CONFIGS = {
    "DSV4": DSV4,
    "DSV4_nvfp4": DSV4_nvfp4,
    "DSV4_8rank": DSV4_8rank,
    "DSV4_8rank_nvfp4": DSV4_8rank_nvfp4,
    "DSV4_6rank": DSV4_6rank,
    "DSV4_6rank_nvfp4": DSV4_6rank_nvfp4,
}

MAX_SLOT: int = DSV4.num_tokens_per_rank * DSV4.num_topk  # 6144

# Host-side mirror of ``TokenSrcMetadata.nbytes`` (one packed i64 per pool
# token).  config stays GPU-free, so it cannot import the device object; keep
# this in sync with ``src/token_comm.py``.
TOKEN_METADATA_BYTES: int = 8


def transform_sf_token_idx_numpy(token_idx_in_expert):
    """Host-side replica of mega_moe transform_sf_token_idx (UTCCP 4x32 swizzle)."""
    t = np.asarray(token_idx_in_expert, dtype=np.int32)
    idx = t % np.int32(DSV4.block_m)
    return (
        (t // np.int32(DSV4.block_m)) * np.int32(DSV4.sf_block_m)
        + (idx & np.int32(-128))
        + (idx & np.int32(31)) * np.int32(4)
        + ((idx >> np.int32(5)) & np.int32(3))
    ).astype(np.int32)
