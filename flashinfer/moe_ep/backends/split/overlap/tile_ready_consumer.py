# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GEMM2 ``tile_ready`` consumer: wait flags, then ship dense permuted ``C``.

Producer contract (GEMM2 epilogue, opt-in ``tile_ready``)
---------------------------------------------------------
* One ``int32`` flag per CTA output tile, index ``m_tile * num_n_tiles + n_tile``.
* ``cta_m`` / ``cta_n`` follow :func:`gemm2_cta_tile_mn` (pinned 128 x 128
  on the fused-gemm2-combine path). A band is ``cta_m`` **permuted** rows.
* Store is ``st.release.gpu`` of ``1``. This kernel pairs it with
  ``ld.acquire.gpu``. The caller zeros the buffer before each GEMM2 launch.
* Flags live in permuted GEMM M-space. Payload is dense ``C[p, :]``.

Wait rule (Finding 009)
-----------------------
A band is hidden-complete only after **every** column tile is flagged.
Wait only GEMM2's live M-tiles (``num_non_exiting_tiles``).

Combine dest
------------
Each GEMM row is one expert output for one dispatched token. Combine sends it
back to that token's **home GPU** at that token's **original index**.

``permuted_idx[p]`` is only the dispatch-buffer row (which local expert).
Home GPU and original index live in ``src_info`` from
:func:`combine_src_info_from_packed` (packed as
``home_rank * tokens_per_rank + token_index``, ``-1`` unused). Skip
``expanded < 0`` and ``src_info < 0``. Do not infer the home GPU from the
dispatch-buffer column.

Ship is ``m02_ship<K>``: 16-byte ``ld.global.v4.u32`` / ``st.global.v4.u32``,
K loads issued before any store, one ``fence.sys`` per CTA at kernel exit.
Default ``K=8``, ``threads=256``, ``max_ctas=8``. Consecutive live ``p`` with
contiguous inbox addresses become one bulk put (in-kernel RLE).

Launch on a **second stream** after GEMM2 is queued; do not PDL-wait on GEMM2.
JIT with ``compile_only=True`` before the GEMM2 host launch so a flag-waiter
is never live across that invoke.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.cute_dsl.fp4_common import (
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_v4_u32,
)
from flashinfer.cute_dsl.utils import make_ptr

# Re-exported so the consumer and the GEMM2 producer can never disagree on the
# flag-buffer geometry.
from flashinfer.fused_moe.cute_dsl.blockscaled_contiguous_grouped_gemm_finalize_fusion import (  # noqa: E501
    gemm2_cta_tile_mn,
    gemm2_tile_ready_numel,
)

_SHIP_UNROLLS = (1, 2, 4, 8)


def expert_major_dest(
    expanded: int, tokens_per_rank: int, world_size: int
) -> Tuple[int, int, int]:
    """Map an EXPERT_MAJOR pack row to ``(expert, dest_rank, window_slot)``.

    ``window_slot`` is the dense index inside the dest-rank window, not the
    dest token index. Use :func:`combine_src_info_from_packed` for the token.
    """
    cap = tokens_per_rank * world_size
    expert = expanded // cap
    local_slot = expanded % cap
    dest_rank = local_slot // tokens_per_rank
    window_slot = local_slot % tokens_per_rank
    return expert, dest_rank, window_slot


# Unused dest / padding fingerprint. Chosen so a live bf16 row cannot match it
# via :func:`_row_fingerprint` (4x int16 packed into int64).
_FP_UNUSED = torch.iinfo(torch.int64).max


def _row_fingerprint(x: torch.Tensor) -> torch.Tensor:
    """Pack the first four bf16 bit-patterns of each row into ``int64``."""
    if x.shape[-1] < 4:
        raise ValueError(
            f"fingerprint needs hidden >= 4, got hidden={int(x.shape[-1])}"
        )
    head = x[..., :4].contiguous().view(torch.int16).to(torch.int64) & 0xFFFF
    return (
        head[..., 0]
        | (head[..., 1] << 16)
        | (head[..., 2] << 32)
        | (head[..., 3] << 48)
    )


row_fingerprint = _row_fingerprint
ROW_FP_UNUSED = _FP_UNUSED


def combine_src_info_from_packed(
    packed: torch.Tensor,
    dest_fp_all: torch.Tensor,
    topk_all: torch.Tensor,
    *,
    local_expert_offset: int,
    num_local_experts: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build combine ``src_info`` from the dispatch payload permutation.

    Combine is: send this expert output back to the GPU that owned the token,
    at that token's original index. Dispatch does not keep tokens in home-GPU
    order, so this matches packed activations to dest hidden states.

    Stored value is ``home_rank * tokens_per_rank + token_index`` (``-1`` unused).

    ``packed`` is ``[num_local_experts, world * tpr, hidden]``. ``dest_fp_all``
    is ``[world, tpr]`` int64 from :func:`_row_fingerprint` on dest hidden
    states. ``topk_all`` is ``[world, tpr, dest_top_k]``. Returns
    ``int32 [num_local_experts, world * tpr]``; unused slots are ``-1``.
    """
    if packed.dim() != 3:
        raise ValueError(
            "packed must be [num_local_experts, world * tpr, hidden], "
            f"got {tuple(packed.shape)}"
        )
    if dest_fp_all.dim() != 2 or dest_fp_all.dtype != torch.int64:
        raise ValueError(
            "dest_fp_all must be int64 [world, tpr], "
            f"got {dest_fp_all.dtype} {tuple(dest_fp_all.shape)}"
        )
    if topk_all.dim() != 3:
        raise ValueError(
            "topk_all must be [world, tokens_per_rank, dest_top_k], "
            f"got {tuple(topk_all.shape)}"
        )
    world, tpr, _k = (int(s) for s in topk_all.shape)
    cap = world * tpr
    if int(packed.shape[0]) != num_local_experts or int(packed.shape[1]) != cap:
        raise ValueError(
            f"packed shape {tuple(packed.shape)} does not match "
            f"num_local_experts={num_local_experts} cap={cap}"
        )
    if tuple(dest_fp_all.shape) != (world, tpr):
        raise ValueError(
            f"dest_fp_all shape {tuple(dest_fp_all.shape)} != ({world}, {tpr})"
        )
    if out is None:
        src_info = torch.full(
            (num_local_experts, cap),
            -1,
            dtype=torch.int32,
            device=packed.device,
        )
    else:
        if out.shape != (num_local_experts, cap) or out.dtype != torch.int32:
            raise ValueError(
                f"out must be int32 [{num_local_experts}, {cap}], got "
                f"{out.dtype} {tuple(out.shape)}"
            )
        src_info = out
        src_info.fill_(-1)

    packed_fp = _row_fingerprint(packed)
    device = packed.device
    unused = torch.tensor(_FP_UNUSED, dtype=torch.int64, device=device)
    neg1 = torch.tensor(-1, dtype=torch.int32, device=device)
    home = (
        torch.arange(world, device=device, dtype=torch.int64)
        .unsqueeze(1)
        .expand(world, tpr)
    )
    token = (
        torch.arange(tpr, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(world, tpr)
    )
    loc = (home * tpr + token).reshape(-1)
    nloc = world * tpr
    for e in range(num_local_experts):
        selected = topk_all.eq(local_expert_offset + e).any(dim=-1)
        cand_fp = torch.where(selected, dest_fp_all, unused).reshape(-1)
        order = torch.argsort(cand_fp)
        sorted_fp = cand_fp[order]
        sorted_loc = loc[order]
        p_fp = packed_fp[e]
        pos = torch.searchsorted(sorted_fp, p_fp).clamp(max=nloc - 1)
        hit = (sorted_fp[pos] == p_fp) & (p_fp != unused)
        src_info[e] = torch.where(hit, sorted_loc[pos].to(dtype=torch.int32), neg1)
    return src_info


def peer_ptrs_from_peer_out(peer_out: torch.Tensor) -> torch.Tensor:
    """Build ``int64[world]`` bases from a contiguous 4-D peer tensor.

    ``peer_out`` shape is ``[world, num_local_experts, tokens_per_rank, hidden]``.
    """
    if peer_out.dim() != 4:
        raise ValueError(
            "peer_out must be [world, num_local_experts, tokens_per_rank, hidden], "
            f"got shape {tuple(peer_out.shape)}"
        )
    if not peer_out.is_contiguous():
        raise ValueError("peer_out must be contiguous")
    world = peer_out.shape[0]
    rank_stride_bytes = peer_out.stride(0) * peer_out.element_size()
    base = peer_out.data_ptr()
    return (
        torch.arange(world, device=peer_out.device, dtype=torch.int64)
        * rank_stride_bytes
        + base
    )


@dsl_user_op
def _fence_sys(*, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [],
        "fence.sys;",
        "",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _nanosleep(n: Int32, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [Int32(n).ir_value(loc=loc, ip=ip)],
        "nanosleep.u32 $0;",
        "r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@cute.jit
def _m02_ship(
    src_ptr: Int64,
    dst_ptr: Int64,
    n_uint4: Int32,
    tid: Int32,
    stride: Int32,
    unroll: cutlass.Constexpr,
) -> None:
    """Copy ``n_uint4`` 16-byte words. K=8 issues 8 loads before any store."""
    w = tid
    word = Int64(16)
    if cutlass.const_expr(unroll == 8):
        last = stride * Int32(7)
        step = stride * Int32(8)
        while w + last < n_uint4:
            a00, a01, a02, a03 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(0)) * word
            )
            a10, a11, a12, a13 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(1)) * word
            )
            a20, a21, a22, a23 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(2)) * word
            )
            a30, a31, a32, a33 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(3)) * word
            )
            a40, a41, a42, a43 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(4)) * word
            )
            a50, a51, a52, a53 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(5)) * word
            )
            a60, a61, a62, a63 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(6)) * word
            )
            a70, a71, a72, a73 = ld_global_v4_u32(
                src_ptr + Int64(w + stride * Int32(7)) * word
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(0)) * word, a00, a01, a02, a03
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(1)) * word, a10, a11, a12, a13
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(2)) * word, a20, a21, a22, a23
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(3)) * word, a30, a31, a32, a33
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(4)) * word, a40, a41, a42, a43
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(5)) * word, a50, a51, a52, a53
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(6)) * word, a60, a61, a62, a63
            )
            st_global_v4_u32(
                dst_ptr + Int64(w + stride * Int32(7)) * word, a70, a71, a72, a73
            )
            w += step
    while w < n_uint4:
        v0, v1, v2, v3 = ld_global_v4_u32(src_ptr + Int64(w) * word)
        st_global_v4_u32(dst_ptr + Int64(w) * word, v0, v1, v2, v3)
        w += stride


def _smem_i32(smem, n: int):
    ptr = smem.allocate_array(cutlass.Int32, n, byte_alignment=4)
    return cute.make_tensor(ptr, cute.make_layout((n,)))


class TileReadyConsumerKernel:
    def __init__(
        self,
        *,
        cta_m: int,
        threads: int,
        unroll: int,
        no_wait: bool,
        no_ship: bool,
        max_ctas: int,
    ):
        self.cta_m = cta_m
        self.threads = threads
        self.unroll = unroll
        self.no_wait = no_wait
        self.no_ship = no_ship
        self.max_ctas = max_ctas

    @cute.kernel
    def kernel(
        self,
        tile_ready: cute.Tensor,
        gemm2_c: cute.Tensor,
        permuted_idx: cute.Tensor,
        peer_ptrs: cute.Tensor,
        src_info: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        shipped_rows: cute.Tensor,
        permuted_m: cutlass.Int32,
        hidden: cutlass.Int32,
        tokens_per_rank: cutlass.Int32,
        world: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        src_rank: cutlass.Int32,
        num_n: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdx, _, _ = cute.arch.grid_dim()

        smem = cutlass.utils.SmemAllocator()
        s_start = _smem_i32(smem, self.cta_m)
        s_n = _smem_i32(smem, self.cta_m)
        s_dest = _smem_i32(smem, self.cta_m)
        s_expert = _smem_i32(smem, self.cta_m)
        s_slot = _smem_i32(smem, self.cta_m)
        s_nruns = _smem_i32(smem, 1)
        cute.arch.sync_threads()

        nne = cute.arch.make_warp_uniform(cutlass.Int32(num_non_exiting_tiles[0]))
        row_bytes = Int64(hidden) * Int64(2)
        uint4_per_row = hidden // 8
        cap = tokens_per_rank * world
        inbox_src_stride = Int64(num_local_experts) * Int64(tokens_per_rank) * row_bytes

        band = bidx
        while band < nne:
            if cutlass.const_expr(not self.no_wait):
                col = tidx
                while col < num_n:
                    flag_idx = band * num_n + col
                    ready = cutlass.Int32(0)
                    while ready == 0:
                        ready = cute.arch.load(
                            tile_ready.iterator + flag_idx,
                            cutlass.Int32,
                            sem="acquire",
                            scope="gpu",
                        )
                        if ready == 0:
                            _nanosleep(Int32(32))
                    col += Int32(self.threads)
            cute.arch.sync_threads()

            if cutlass.const_expr(not self.no_ship):
                band_start = band * Int32(self.cta_m)
                if tidx == 0:
                    nruns = Int32(0)
                    active = Int32(0)
                    run_start = Int32(0)
                    run_n = Int32(0)
                    run_dest = Int32(0)
                    run_expert = Int32(0)
                    run_slot = Int32(0)
                    for i in cutlass.range(self.cta_m, unroll_full=True):
                        p = band_start + i
                        live = Int32(0)
                        dest_rank = Int32(0)
                        expert = Int32(0)
                        slot = Int32(0)
                        if p < permuted_m:
                            expanded = permuted_idx[p]
                            if expanded >= 0:
                                expert = expanded // cap
                                local_slot = expanded % cap
                                in_bounds = (
                                    (expert >= 0)
                                    & (expert < num_local_experts)
                                    & (local_slot >= 0)
                                    & (local_slot < cap)
                                )
                                if in_bounds:
                                    loc = src_info[(expert, local_slot)]
                                    if loc >= 0:
                                        dest_rank = loc // tokens_per_rank
                                        slot = loc % tokens_per_rank
                                        live = (
                                            Int32(dest_rank >= 0)
                                            & Int32(dest_rank < world)
                                            & Int32(slot >= 0)
                                            & Int32(slot < tokens_per_rank)
                                        )
                        if live != 0:
                            cont = (
                                (active != 0)
                                & (dest_rank == run_dest)
                                & (expert == run_expert)
                                & (slot == run_slot + run_n)
                            )
                            if cont:
                                run_n = run_n + Int32(1)
                            else:
                                if active != 0:
                                    s_start[nruns] = run_start
                                    s_n[nruns] = run_n
                                    s_dest[nruns] = run_dest
                                    s_expert[nruns] = run_expert
                                    s_slot[nruns] = run_slot
                                    nruns = nruns + Int32(1)
                                run_start = p
                                run_n = Int32(1)
                                run_dest = dest_rank
                                run_expert = expert
                                run_slot = slot
                                active = Int32(1)
                        else:
                            if active != 0:
                                s_start[nruns] = run_start
                                s_n[nruns] = run_n
                                s_dest[nruns] = run_dest
                                s_expert[nruns] = run_expert
                                s_slot[nruns] = run_slot
                                nruns = nruns + Int32(1)
                                active = Int32(0)
                    if active != 0:
                        s_start[nruns] = run_start
                        s_n[nruns] = run_n
                        s_dest[nruns] = run_dest
                        s_expert[nruns] = run_expert
                        s_slot[nruns] = run_slot
                        nruns = nruns + Int32(1)
                    s_nruns[0] = nruns
                cute.arch.sync_threads()

                nruns = s_nruns[0]
                shipped = Int32(0)
                for r in cutlass.range(self.cta_m):
                    if r < nruns:
                        p0 = s_start[r]
                        n_run = s_n[r]
                        dest_rank = s_dest[r]
                        expert = s_expert[r]
                        slot = s_slot[r]
                        dst = (
                            peer_ptrs[dest_rank]
                            + Int64(src_rank) * inbox_src_stride
                            + (Int64(expert) * Int64(tokens_per_rank) + Int64(slot))
                            * row_bytes
                        )
                        src = (
                            get_ptr_as_int64(gemm2_c, Int64(0)) + Int64(p0) * row_bytes
                        )
                        _m02_ship(
                            src,
                            dst,
                            n_run * uint4_per_row,
                            tidx,
                            Int32(self.threads),
                            self.unroll,
                        )
                        if tidx == 0:
                            shipped += n_run
                cute.arch.sync_threads()
                if tidx == 0 and shipped != 0:
                    cute.arch.atomic_add(shipped_rows.iterator, shipped)
            band += gdx

        if cutlass.const_expr(not self.no_ship):
            _fence_sys()

    @cute.jit
    def wrapper(
        self,
        tile_ready_ptr: cute.Pointer,
        gemm2_c_ptr: cute.Pointer,
        permuted_idx_ptr: cute.Pointer,
        peer_ptrs_ptr: cute.Pointer,
        src_info_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        shipped_rows_ptr: cute.Pointer,
        permuted_m: cutlass.Int32,
        hidden: cutlass.Int32,
        tokens_per_rank: cutlass.Int32,
        world: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        src_rank: cutlass.Int32,
        num_n: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        tile_ready = cute.make_tensor(
            tile_ready_ptr,
            layout=cute.make_layout((permuted_m * num_n,)),
        )
        gemm2_c = cute.make_tensor(
            gemm2_c_ptr,
            layout=cute.make_ordered_layout((permuted_m, hidden), order=(1, 0)),
        )
        permuted_idx = cute.make_tensor(
            permuted_idx_ptr, layout=cute.make_layout((permuted_m,))
        )
        peer_ptrs = cute.make_tensor(peer_ptrs_ptr, layout=cute.make_layout((world,)))
        cap = tokens_per_rank * world
        src_info = cute.make_tensor(
            src_info_ptr,
            layout=cute.make_ordered_layout((num_local_experts, cap), order=(1, 0)),
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        shipped_rows = cute.make_tensor(shipped_rows_ptr, layout=cute.make_layout((1,)))
        self.kernel(
            tile_ready,
            gemm2_c,
            permuted_idx,
            peer_ptrs,
            src_info,
            num_non_exiting_tiles,
            shipped_rows,
            permuted_m,
            hidden,
            tokens_per_rank,
            world,
            num_local_experts,
            src_rank,
            num_n,
        ).launch(
            grid=[self.max_ctas, 1, 1],
            block=[self.threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )


_kernel_cache: Dict[Tuple, Any] = {}
_cached_dummy_peer_ptrs: Optional[torch.Tensor] = None
_cached_dummy_src_info: Optional[torch.Tensor] = None


def _dummy_peer_ptrs(device: torch.device) -> torch.Tensor:
    global _cached_dummy_peer_ptrs
    if _cached_dummy_peer_ptrs is None or _cached_dummy_peer_ptrs.device != device:
        _cached_dummy_peer_ptrs = torch.zeros(1, dtype=torch.int64, device=device)
    return _cached_dummy_peer_ptrs


def _dummy_src_info(device: torch.device) -> torch.Tensor:
    global _cached_dummy_src_info
    if _cached_dummy_src_info is None or _cached_dummy_src_info.device != device:
        _cached_dummy_src_info = torch.full(
            (1, 1), -1, dtype=torch.int32, device=device
        )
    return _cached_dummy_src_info


def _get_compiled_consumer(
    *,
    cta_m: int,
    threads: int,
    unroll: int,
    no_wait: bool,
    no_ship: bool,
    max_ctas: int,
    tile_ready_ptr,
    gemm2_c_ptr,
    permuted_idx_ptr,
    peer_ptrs_ptr,
    src_info_ptr,
    num_tiles_ptr,
    shipped_ptr,
    permuted_m: int,
    hidden: int,
    tokens_per_rank: int,
    world: int,
    num_local_experts: int,
    src_rank: int,
    num_n: int,
    stream,
):
    cache_key = (cta_m, threads, unroll, no_wait, no_ship, max_ctas)
    if cache_key not in _kernel_cache:
        gemm = TileReadyConsumerKernel(
            cta_m=cta_m,
            threads=threads,
            unroll=unroll,
            no_wait=no_wait,
            no_ship=no_ship,
            max_ctas=max_ctas,
        )
        compiled = cute.compile(
            gemm.wrapper,
            tile_ready_ptr,
            gemm2_c_ptr,
            permuted_idx_ptr,
            peer_ptrs_ptr,
            src_info_ptr,
            num_tiles_ptr,
            shipped_ptr,
            permuted_m,
            hidden,
            tokens_per_rank,
            world,
            num_local_experts,
            src_rank,
            num_n,
            stream,
        )
        _kernel_cache[cache_key] = compiled
    return _kernel_cache[cache_key]


def launch_tile_ready_consumer(
    tile_ready: torch.Tensor,
    gemm2_c: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    mma_tiler_mn: Tuple[int, int],
    tokens_per_rank: int,
    world_size: int,
    num_local_experts: int,
    num_non_exiting_tiles: torch.Tensor,
    permuted_m: int,
    src_info: Optional[torch.Tensor] = None,
    *,
    peer_ptrs: Optional[torch.Tensor] = None,
    peer_out: Optional[torch.Tensor] = None,
    src_rank: int = 0,
    shipped_rows: Optional[torch.Tensor] = None,
    no_ship: bool = False,
    no_wait: bool = False,
    threads: int = 256,
    unroll: int = 8,
    max_ctas: int = 8,
    stream: Optional[torch.cuda.Stream] = None,
    compile_only: bool = False,
) -> None:
    """Launch the tile-ready consumer on ``stream`` (default: current stream).

    ``compile_only=True`` runs ``cute.compile`` (or a cache hit) and returns
    without queuing the kernel. Use that to keep a flag-waiter off the GPU
    across a later GEMM2 host launch.

    ``src_info`` is ``[num_local_experts, world * tokens_per_rank]`` int32
    home locations ``home_rank * tokens_per_rank + token_index`` (``-1``
    unused). Build it with :func:`combine_src_info_from_packed`.
    """
    if gemm2_c.dim() != 2:
        raise ValueError(
            f"gemm2_c must be 2-D [rows, hidden], got {tuple(gemm2_c.shape)}"
        )
    if gemm2_c.dtype != torch.bfloat16:
        raise ValueError(f"gemm2_c must be torch.bfloat16, got {gemm2_c.dtype}")
    if not gemm2_c.is_contiguous() or gemm2_c.device.type != "cuda":
        raise ValueError("gemm2_c must be a contiguous CUDA tensor")
    if tile_ready.dtype != torch.int32 or not tile_ready.is_contiguous():
        raise ValueError("tile_ready must be contiguous int32")
    if tile_ready.device.type != "cuda":
        raise ValueError("tile_ready must be on CUDA")
    if permuted_idx_to_expanded_idx.dtype != torch.int32:
        raise ValueError(
            f"permuted_idx_to_expanded_idx must be int32, got {permuted_idx_to_expanded_idx.dtype}"
        )
    if permuted_idx_to_expanded_idx.device.type != "cuda":
        raise ValueError("permuted_idx_to_expanded_idx must be on CUDA")
    if not permuted_idx_to_expanded_idx.is_contiguous():
        raise ValueError("permuted_idx_to_expanded_idx must be contiguous")
    if num_non_exiting_tiles.dtype != torch.int32:
        raise ValueError(
            f"num_non_exiting_tiles must be int32, got {num_non_exiting_tiles.dtype}"
        )
    if num_non_exiting_tiles.device.type != "cuda":
        raise ValueError("num_non_exiting_tiles must be on CUDA")
    if not num_non_exiting_tiles.is_contiguous():
        raise ValueError("num_non_exiting_tiles must be contiguous")
    if num_non_exiting_tiles.numel() < 1:
        raise ValueError("num_non_exiting_tiles must hold at least one int32")

    hidden = int(gemm2_c.shape[1])
    if hidden % 8 != 0:
        raise ValueError(
            f"hidden must be a multiple of 8 for 16-byte (uint4) copies, got {hidden}"
        )
    if unroll not in _SHIP_UNROLLS:
        raise ValueError(f"unroll must be one of {_SHIP_UNROLLS}, got {unroll}")
    if tokens_per_rank <= 0 or world_size <= 0 or num_local_experts <= 0:
        raise ValueError(
            "tokens_per_rank, world_size, and num_local_experts must be positive"
        )
    if max_ctas <= 0:
        raise ValueError(f"max_ctas must be positive, got {max_ctas}")
    if src_rank < 0 or src_rank >= world_size:
        raise ValueError(
            f"src_rank must be in [0, world_size), got {src_rank} vs {world_size}"
        )
    if int(gemm2_c.shape[0]) < permuted_m:
        raise ValueError(
            f"gemm2_c.shape[0]={int(gemm2_c.shape[0])} < permuted_m={permuted_m}"
        )
    if permuted_m > int(permuted_idx_to_expanded_idx.numel()):
        raise ValueError(
            f"permuted_m={permuted_m} exceeds permuted_idx_to_expanded_idx "
            f"numel={int(permuted_idx_to_expanded_idx.numel())}"
        )

    cap = tokens_per_rank * world_size
    if no_ship:
        if src_info is None:
            src_info = _dummy_src_info(gemm2_c.device)
    else:
        if src_info is None:
            raise ValueError("src_info is required unless no_ship=True")
        if src_info.dim() != 2:
            raise ValueError(
                "src_info must be [num_local_experts, world * tokens_per_rank], "
                f"got {tuple(src_info.shape)}"
            )
        if int(src_info.shape[0]) != num_local_experts or int(src_info.shape[1]) != cap:
            raise ValueError(
                f"src_info shape {tuple(src_info.shape)} does not match "
                f"nle={num_local_experts} cap={cap}"
            )
        if src_info.dtype != torch.int32:
            raise ValueError(f"src_info must be int32, got {src_info.dtype}")
        if not src_info.is_contiguous() or src_info.device.type != "cuda":
            raise ValueError("src_info must be a contiguous CUDA tensor")

    cta_m, cta_n = gemm2_cta_tile_mn(mma_tiler_mn)
    needed = gemm2_tile_ready_numel(permuted_m, hidden, mma_tiler_mn)
    if int(tile_ready.numel()) < needed:
        raise ValueError(
            f"tile_ready must have at least {needed} flags (permuted_m={permuted_m}, "
            f"hidden={hidden}, mma_tiler_mn={mma_tiler_mn}), got {int(tile_ready.numel())}"
        )
    num_n = (hidden + cta_n - 1) // cta_n

    if no_ship:
        if peer_ptrs is None:
            peer_ptrs = _dummy_peer_ptrs(gemm2_c.device)
    else:
        if peer_ptrs is None:
            if peer_out is None:
                raise ValueError(
                    "peer_ptrs or peer_out is required unless no_ship=True"
                )
            peer_ptrs = peer_ptrs_from_peer_out(peer_out)
        if peer_ptrs.dtype != torch.int64 or not peer_ptrs.is_contiguous():
            raise ValueError("peer_ptrs must be contiguous int64")
        if int(peer_ptrs.numel()) < world_size:
            raise ValueError(
                f"peer_ptrs numel={int(peer_ptrs.numel())} < world_size={world_size}"
            )

    if shipped_rows is None:
        shipped_rows = torch.zeros(1, dtype=torch.int32, device=gemm2_c.device)
    elif shipped_rows.dtype != torch.int32 or shipped_rows.numel() < 1:
        raise ValueError("shipped_rows must be a 1-element int32 tensor")

    torch_stream = stream or torch.cuda.current_stream()
    cu_stream = cuda.CUstream(torch_stream.cuda_stream)

    tile_ready_ptr = make_ptr(
        cutlass.Int32, tile_ready.data_ptr(), cute.AddressSpace.gmem
    )
    gemm2_c_ptr = make_ptr(
        BFloat16, gemm2_c.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
    )
    permuted_idx_ptr = make_ptr(
        cutlass.Int32,
        permuted_idx_to_expanded_idx.data_ptr(),
        cute.AddressSpace.gmem,
    )
    peer_ptrs_ptr = make_ptr(
        cutlass.Int64, peer_ptrs.data_ptr(), cute.AddressSpace.gmem
    )
    src_info_ptr = make_ptr(cutlass.Int32, src_info.data_ptr(), cute.AddressSpace.gmem)
    num_tiles_ptr = make_ptr(
        cutlass.Int32, num_non_exiting_tiles.data_ptr(), cute.AddressSpace.gmem
    )
    shipped_ptr = make_ptr(
        cutlass.Int32, shipped_rows.data_ptr(), cute.AddressSpace.gmem
    )

    compiled = _get_compiled_consumer(
        cta_m=cta_m,
        threads=threads,
        unroll=unroll,
        no_wait=no_wait,
        no_ship=no_ship,
        max_ctas=max_ctas,
        tile_ready_ptr=tile_ready_ptr,
        gemm2_c_ptr=gemm2_c_ptr,
        permuted_idx_ptr=permuted_idx_ptr,
        peer_ptrs_ptr=peer_ptrs_ptr,
        src_info_ptr=src_info_ptr,
        num_tiles_ptr=num_tiles_ptr,
        shipped_ptr=shipped_ptr,
        permuted_m=permuted_m,
        hidden=hidden,
        tokens_per_rank=tokens_per_rank,
        world=world_size,
        num_local_experts=num_local_experts,
        src_rank=src_rank,
        num_n=num_n,
        stream=cu_stream,
    )

    if compile_only:
        return

    compiled(
        tile_ready_ptr,
        gemm2_c_ptr,
        permuted_idx_ptr,
        peer_ptrs_ptr,
        src_info_ptr,
        num_tiles_ptr,
        shipped_ptr,
        permuted_m,
        hidden,
        tokens_per_rank,
        world_size,
        num_local_experts,
        src_rank,
        num_n,
        stream=cu_stream,
    )
