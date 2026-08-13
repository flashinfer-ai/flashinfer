# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Stateless SM90 FP8 epilogue helpers shared by both operand orders."""

import dataclasses
from typing import Any, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Float32
from cutlass.cutlass_dsl import Int64

from common.megamoe_constants import Log2E
from common.moe_utils import fmax, fmin
from src.token_comm import TokenSrcMetadata


@cute.jit
def clamp_and_swiglu_sm90(
    t_swiglu: cute.Tensor,
    t_up: cute.Tensor,
    t_gate: cute.Tensor,
    glu_clamp,
    prob: Float32,
) -> None:
    """Apply the optional gate/up clamp and scalar SM90 SwiGLU."""
    if cutlass.const_expr(glu_clamp is not None):
        for i in cutlass.range_constexpr(cute.size(t_up)):
            t_gate[i] = fmin(t_gate[i], glu_clamp)
            t_up[i] = fmin(t_up[i], glu_clamp)
            t_up[i] = fmax(t_up[i], -glu_clamp)

    for i in cutlass.range_constexpr(cute.size(t_swiglu)):
        neg_gate_log2e = t_gate[i] * -Log2E
        exp_val = cute.math.exp2(neg_gate_log2e, fastmath=True)
        sigmoid = cute.arch.rcp_approx(exp_val + Float32(1.0))
        t_swiglu[i] = t_up[i] * t_gate[i] * sigmoid * prob


@dataclasses.dataclass(frozen=True)
class Fc2OutputDest:
    """Resolve direct or peer-mapped FC2 output rows."""

    tensor: cute.Tensor
    metadata: Optional[cute.Tensor] = None
    peer_rank_ptr_mapper: Any = None
    reduce_topk_in_kernel: bool = False

    def __post_init__(self) -> None:
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError(
                "Fc2OutputDest: ``metadata`` and ``peer_rank_ptr_mapper`` must be "
                "both None (direct mode) or both non-None (MegaMoE / indirect "
                "mode).  Got metadata="
                f"{'set' if self.metadata is not None else 'None'}, "
                "peer_rank_ptr_mapper="
                f"{'set' if self.peer_rank_ptr_mapper is not None else 'None'}."
            )

    @cute.jit
    def resolve_token_row(self, pool_token_global) -> cute.Tensor:
        if cutlass.const_expr(self.metadata is None):
            return cute.slice_(self.tensor, (pool_token_global, 0, None))

        md = TokenSrcMetadata.load(
            self.metadata.iterator.toint()
            + Int64(pool_token_global) * Int64(TokenSrcMetadata.nbytes)
        )
        src_rank = md.src_rank
        src_token = md.src_token
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            src_topk = cutlass.Int32(0)
        else:
            src_topk = md.src_topk
        local_row = cute.slice_(self.tensor, (src_token, src_topk, None))
        peer_iter = self.peer_rank_ptr_mapper.ptr_map_to_rank(
            local_row.iterator, src_rank,
        )
        return cute.make_tensor(peer_iter, local_row.layout)


@cute.jit
def tma_store_fc1_output(
    sC,
    stage_idx,
    tma_atom_fc1_output: cute.CopyAtom,
    g_fc1_output_subtile_view: cute.Tensor,
    valid_tokens,
) -> None:
    """Issue one WG-private FC1 TMA store for a non-empty token tile."""
    sC_stage = cute.slice_(sC, (None, None, stage_idx))
    g_fc1_output_2d = cute.slice_(g_fc1_output_subtile_view, (None, None, 0))
    bSG_sC, bSG_g = cpasync.tma_partition(
        tma_atom_fc1_output,
        0,
        cute.make_layout(1),
        cute.group_modes(sC_stage, 0, 2),
        cute.group_modes(g_fc1_output_2d, 0, 2),
    )

    tile_has_valid = valid_tokens > cutlass.Int32(0)
    if tile_has_valid:
        with cute.arch.elect_one():
            cute.copy(tma_atom_fc1_output, bSG_sC, bSG_g)


@cute.jit
def stg_fc1_block_scale_row(
    real_fc1_output_sf: cute.Tensor,
    scale_col_idx,
    token_idx,
    scale: Float32,
) -> None:
    """Store one FP32 blockwise FC1-output scale."""
    sf_base = cute.local_tile(
        real_fc1_output_sf,
        (1, 1, 1),
        (token_idx, scale_col_idx, cutlass.Int32(0)),
    )
    gmem_sf = cute.make_tensor(sf_base.iterator, cute.make_layout(1))
    rmem_sf = cute.make_rmem_tensor((1,), cutlass.Float32)
    rmem_sf[0] = scale
    cute.autovec_copy(rmem_sf, gmem_sf)
