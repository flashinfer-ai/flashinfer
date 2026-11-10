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

"""CuTe DSL ReplaySSM commit from verify-produced delta-rule corrections.

Verify starts from a frozen checkpoint ``h0`` and solves the single-chunk
delta rule for every drafted token, producing the correction
``u_t = beta_t * (v_t - (g_t * h_{t-1})^T k_t)``.  Because the chunk system is
unit lower triangular, ``u_t`` depends only on tokens ``1..t``; the rows of a
full-window ``u`` are valid for every accepted prefix length ``a``.  Commit
therefore needs no recurrence:

    h_a = gamma_a * h0 + sum_{j <= a} (gamma_a / gamma_j) * k_j u_j^T,

with ``gamma_t = exp(L_t)`` and ``L_t`` the cumulative log decay.  The weight
``gamma_a / gamma_j = exp(L_a - L_j)`` lies in (0, 1].  This kernel streams
each checkpoint once and applies this scale plus rank-``a`` update with FP32
FMAs, so state accuracy matches a sequential FP32 replay given ``u``.

``u`` embeds ``h0``: it must come from the verify call that read the same,
unmodified checkpoint content.
"""

import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from ..jit.cute_dsl_core import build_and_load_cute_dsl_kernel
from .cute_dsl_cache_naming import make_kernel_name
from .device_target import gdn_compile_options, gdn_device_target
import cuda.bindings.driver as cuda


_F32 = cutlass.Float32
_BF16 = cutlass.BFloat16
_K = 128
_V = 128
_CHANNELS_PER_LANE = 4
_ROWS_PER_TILE = 8
_L2_EPS = 1.0e-6
# Longest window: the decay prefix is one warp scan over the window's tokens.
MAX_WINDOW = 32


def _aligned_tensor(tensor: cute.Tensor, alignment: int) -> cute.Tensor:
    """Attach alignment guaranteed by base validation and compact strides."""
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            pointer.dtype,
            pointer.toint(),
            pointer.memspace,
            assumed_align=alignment,
        ),
        tensor.layout,
    )


@cute.jit
def _decay_weights(
    log_decay: cutlass.Float32,
    count: cutlass.Int32,
    lane: cutlass.Int32,
):
    """Lane j returns ``exp(L_{count-1} - L_j)``; also returns ``exp(L_{count-1})``.

    ``log_decay`` holds lane j's per-token log gate (zero past the window) and
    is turned into the inclusive prefix ``L_j`` with a warp scan, which covers
    windows of up to ``MAX_WINDOW`` tokens.
    """
    prefix = log_decay
    for offset in (1, 2, 4, 8, 16):
        neighbor = cute.arch.shuffle_sync_up(prefix, offset)
        if lane >= offset:
            prefix += neighbor
    last = cute.arch.shuffle_sync(prefix, count - 1)
    return cute.exp(last - prefix, fastmath=True), cute.exp(last, fastmath=True)


@cute.jit
def _rank_update_tile(
    s_state: cute.Tensor,
    destination: cute.Tensor,
    state_scale: cutlass.Float32,
    s_u: cute.Tensor,
    s_key: cute.Tensor,
    count: cutlass.Int32,
    lane: cutlass.Int32,
    tile: cutlass.Constexpr[int],
    warp: cutlass.Int32,
):
    """Store ``state_scale * h0 + sum_{j < count} u_j[rows] (w_j k_j)^T``.

    Lanes own four consecutive K columns of eight rows, so every shared and
    global access is a 128-bit vector and ``u`` is a warp broadcast.
    """
    copy_f32x4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), _F32, num_bits_per_copy=128
    )
    result = cute.make_rmem_tensor((_CHANNELS_PER_LANE, _ROWS_PER_TILE), _F32)
    cute.copy(copy_f32x4, s_state[(None, lane, None, tile, warp)], result)
    scale_pair = (state_scale, state_scale)
    for row in cutlass.range_constexpr(_ROWS_PER_TILE):
        result[(0, row)], result[(1, row)] = cute.arch.mul_packed_f32x2(
            (result[(0, row)], result[(1, row)]), scale_pair
        )
        result[(2, row)], result[(3, row)] = cute.arch.mul_packed_f32x2(
            (result[(2, row)], result[(3, row)]), scale_pair
        )
    coeff = cute.make_rmem_tensor((_ROWS_PER_TILE,), _F32)
    key = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _F32)
    for step in cutlass.range(count, unroll=4):
        cute.copy(copy_f32x4, s_u[(None, step, tile, warp)], coeff)
        cute.copy(copy_f32x4, s_key[(None, lane, step)], key)
        key01 = (key[0], key[1])
        key23 = (key[2], key[3])
        for row in cutlass.range_constexpr(_ROWS_PER_TILE):
            pair = (coeff[row], coeff[row])
            result[(0, row)], result[(1, row)] = cute.arch.fma_packed_f32x2(
                pair, key01, (result[(0, row)], result[(1, row)])
            )
            result[(2, row)], result[(3, row)] = cute.arch.fma_packed_f32x2(
                pair, key23, (result[(2, row)], result[(3, row)])
            )
    cute.copy(copy_f32x4, result, destination)


@cute.kernel
def _gdn_replayssm_u_commit_kernel(
    state: cute.Tensor,
    u: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    WARPS_PER_CTA: cutlass.Constexpr[int],
    TILES_PER_WARP: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[bool],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    HAS_TRACK: cutlass.Constexpr[bool],
):
    row_tile, i_n, layer_head = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    i_layer = layer_head // HV
    i_hv = layer_head % HV
    i_h = i_hv // (HV // H)
    slots_per_layer = state.shape[7] // NUM_LAYERS

    smem = utils.SmemAllocator()
    tile_elems = _ROWS_PER_TILE * _K
    s_state = smem.allocate_tensor(
        _F32,
        cute.make_layout(
            (_CHANNELS_PER_LANE, 32, _ROWS_PER_TILE, TILES_PER_WARP, WARPS_PER_CTA),
            stride=(
                1,
                _CHANNELS_PER_LANE,
                _K,
                tile_elems,
                TILES_PER_WARP * tile_elems,
            ),
        ),
        16,
    )
    # Raw u rows of each warp's tiles; filled by cp.async in 16-byte halves.
    u_tile = T * _ROWS_PER_TILE
    s_u = smem.allocate_tensor(
        _F32,
        cute.make_layout(
            (_ROWS_PER_TILE, T, TILES_PER_WARP, WARPS_PER_CTA),
            stride=(1, _ROWS_PER_TILE, u_tile, TILES_PER_WARP * u_tile),
        ),
        16,
    )
    s_u_fill = cute.make_tensor(
        s_u.iterator,
        cute.make_layout(
            (4, 2, T, TILES_PER_WARP, WARPS_PER_CTA),
            stride=(1, 4, _ROWS_PER_TILE, u_tile, TILES_PER_WARP * u_tile),
        ),
    )
    key_layout = cute.make_layout(
        (_CHANNELS_PER_LANE, 32, T), stride=(1, _CHANNELS_PER_LANE, _K)
    )
    s_key = smem.allocate_tensor(_F32, key_layout, 16)
    s_track_key = s_key
    if cutlass.const_expr(HAS_TRACK):
        s_track_key = smem.allocate_tensor(_F32, key_layout, 16)

    requested_slot = state_indices[i_n]
    n_commit = accept_lens[i_n]
    if requested_slot > NULL_BLOCK_ID and n_commit > 0:
        # Int64: layers * slots * HV * V * K passes 2^31 elements on large pools.
        flat_slot = cutlass.Int64(i_layer) * slots_per_layer + requested_slot

        copy_f32x4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), _F32, num_bits_per_copy=128
        )
        copy_bf16x4 = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), _BF16, num_bits_per_copy=64
        )
        copy_async = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            _F32,
            num_bits_per_copy=128,
        )

        # Each warp streams its own u rows and state tiles through private
        # shared memory.  Everything is in flight before the CTA prologue, and
        # tile i is updated while tiles i+1.. are still arriving.
        # Only the accepted prefix of u is read.
        for u_pass in cutlass.range_constexpr((2 * T + 31) // 32):
            half_row = u_pass * 32 + lane
            if half_row < 2 * T and half_row < 2 * n_commit:
                for tile in cutlass.range_constexpr(TILES_PER_WARP):
                    cute.copy(
                        copy_async,
                        _aligned_tensor(
                            u[
                                (
                                    None,
                                    half_row % 2,
                                    tile,
                                    warp,
                                    row_tile,
                                    half_row // 2,
                                    i_hv,
                                    flat_slot,
                                )
                            ],
                            16,
                        ),
                        s_u_fill[(None, half_row % 2, half_row // 2, tile, warp)],
                    )
        for tile in cutlass.range_constexpr(TILES_PER_WARP):
            cute.copy(
                copy_async,
                _aligned_tensor(
                    state[(None, lane, None, tile, warp, row_tile, i_hv, flat_slot)],
                    16,
                ),
                s_state[(None, lane, None, tile, warp)],
            )
            cute.arch.cp_async_commit_group()

        # The decay weights w_j = gamma_last / gamma_j are folded into the
        # normalized keys, so the inner loop is a plain rank update with u.
        log_decay = _F32(0.0)
        if lane < n_commit:
            log_decay = log_g[(lane, i_hv, flat_slot)]
        weight, state_scale = _decay_weights(log_decay, n_commit, lane)
        track_live = cutlass.Boolean(False)
        track_count = cutlass.Int32(0)
        track_weight = weight
        track_scale = state_scale
        flat_track_slot = flat_slot
        if cutlass.const_expr(HAS_TRACK):
            requested_track_slot = track_indices[i_n]
            track_count = track_steps[i_n] + 1
            track_live = (
                requested_track_slot > NULL_BLOCK_ID
                and track_count >= 1
                and track_count <= n_commit
            )
            if track_live:
                flat_track_slot = (
                    cutlass.Int64(i_layer) * slots_per_layer + requested_track_slot
                )
                track_weight, track_scale = _decay_weights(log_decay, track_count, lane)

        # Warps normalize interleaved key rows once per CTA.
        key_bf16 = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _BF16)
        key = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _F32)
        weighted_key = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _F32)
        for step in cutlass.range(warp, n_commit, WARPS_PER_CTA, unroll=1):
            cute.copy(
                copy_bf16x4,
                _aligned_tensor(rawk[(None, lane, step, i_h, flat_slot)], 8),
                key_bf16,
            )
            key_norm = _F32(0.0)
            for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                key[channel] = key_bf16[channel].to(_F32)
                key_norm += key[channel] * key[channel]
            key_scale = _F32(1.0)
            if cutlass.const_expr(USE_QK_L2NORM):
                key_norm = cute.arch.warp_reduction_sum(key_norm)
                key_scale = cute.rsqrt(key_norm + _L2_EPS)
            step_weight = cute.arch.shuffle_sync(weight, step) * key_scale
            for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                weighted_key[channel] = key[channel] * step_weight
            cute.copy(copy_f32x4, weighted_key, s_key[(None, lane, step)])
            if cutlass.const_expr(HAS_TRACK):
                step_weight = cute.arch.shuffle_sync(track_weight, step) * key_scale
                if track_live and step < track_count:
                    for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                        weighted_key[channel] = key[channel] * step_weight
                    cute.copy(copy_f32x4, weighted_key, s_track_key[(None, lane, step)])
        cute.arch.barrier()

        for tile in cutlass.range_constexpr(TILES_PER_WARP):
            cute.arch.cp_async_wait_group(TILES_PER_WARP - 1 - tile)
            cute.arch.sync_warp()
            if cutlass.const_expr(HAS_TRACK):
                if track_live:
                    _rank_update_tile(
                        s_state,
                        _aligned_tensor(
                            state[
                                (
                                    None,
                                    lane,
                                    None,
                                    tile,
                                    warp,
                                    row_tile,
                                    i_hv,
                                    flat_track_slot,
                                )
                            ],
                            16,
                        ),
                        track_scale,
                        s_u,
                        s_track_key,
                        track_count,
                        lane,
                        tile,
                        warp,
                    )
            _rank_update_tile(
                s_state,
                _aligned_tensor(
                    state[(None, lane, None, tile, warp, row_tile, i_hv, flat_slot)],
                    16,
                ),
                state_scale,
                s_u,
                s_key,
                n_commit,
                lane,
                tile,
                warp,
            )


@cute.jit
def _launch_gdn_replayssm_u_commit(
    state: cute.Tensor,
    u: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    T: cutlass.Constexpr[int],
    WARPS_PER_CTA: cutlass.Constexpr[int],
    TILES_PER_WARP: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[bool],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    HAS_TRACK: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    rows_per_warp = TILES_PER_WARP * _ROWS_PER_TILE
    rows_per_cta = WARPS_PER_CTA * rows_per_warp
    row_tiles = _V // rows_per_cta

    state_view = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            (
                _CHANNELS_PER_LANE,
                32,
                _ROWS_PER_TILE,
                TILES_PER_WARP,
                WARPS_PER_CTA,
                row_tiles,
                HV,
                state.shape[0],
            ),
            stride=(
                1,
                _CHANNELS_PER_LANE,
                _K,
                _ROWS_PER_TILE * _K,
                rows_per_warp * _K,
                rows_per_cta * _K,
                _V * _K,
                HV * _V * _K,
            ),
        ),
    )
    u_view = cute.make_tensor(
        u.iterator,
        cute.make_layout(
            (
                4,
                2,
                TILES_PER_WARP,
                WARPS_PER_CTA,
                row_tiles,
                T,
                HV,
                u.shape[0],
            ),
            stride=(
                1,
                4,
                _ROWS_PER_TILE,
                rows_per_warp,
                rows_per_cta,
                _V,
                T * _V,
                HV * T * _V,
            ),
        ),
    )
    rawk_view = cute.make_tensor(
        rawk.iterator,
        cute.make_layout(
            (_CHANNELS_PER_LANE, 32, T, H, rawk.shape[0]),
            stride=(1, _CHANNELS_PER_LANE, _K, T * _K, H * T * _K),
        ),
    )
    log_g_view = cute.make_tensor(
        log_g.iterator,
        cute.make_layout((T, HV, log_g.shape[0]), stride=(1, T, HV * T)),
    )

    _gdn_replayssm_u_commit_kernel(
        state_view,
        u_view,
        rawk_view,
        log_g_view,
        state_indices,
        accept_lens,
        track_indices,
        track_steps,
        NUM_LAYERS,
        H,
        HV,
        T,
        WARPS_PER_CTA,
        TILES_PER_WARP,
        USE_QK_L2NORM,
        NULL_BLOCK_ID,
        HAS_TRACK,
    ).launch(
        grid=(row_tiles, state_indices.shape[0], NUM_LAYERS * HV),
        block=(WARPS_PER_CTA * 32, 1, 1),
        stream=stream,
    )


_CACHE: dict = {}


def _mark_leading_dynamic(tensor: torch.Tensor) -> cute.Tensor:
    return from_dlpack(tensor, assumed_align=16).mark_compact_shape_dynamic(
        mode=0,
        stride_order=tuple(range(tensor.dim())),
        divisibility=1,
    )


def commit_gdn_replayssm_u_all_layers(
    checkpoint_state: torch.Tensor,
    u_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    track_state_indices: torch.Tensor | None = None,
    track_steps: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    *,
    warps_per_cta: int = 4,
    tiles_per_warp: int = 2,
) -> None:
    """Fold verify-produced ``u`` windows into every layer's FP32 checkpoint.

    Shapes: checkpoint ``[layers, slots, HV, 128, 128]`` FP32 (K-last),
    ``u_cache`` ``[layers, slots, HV, T, 128]`` FP32, ``rawk_cache``
    ``[layers, slots, H, T, 128]`` BF16 and ``g_cache``
    ``[layers, slots, HV, T]`` FP32.  Index and accept-length semantics match
    :func:`commit_gdn_replayssm_fold_all_layers`.  Validation is done by the
    caller.
    """
    num_layers, num_slots, HV, V, K = checkpoint_state.shape
    T = u_cache.shape[3]
    H = rawk_cache.shape[2]
    target = gdn_device_target(checkpoint_state.device)
    has_track = track_state_indices is not None
    if has_track:
        track_indices = track_state_indices
    else:
        track_indices = ssm_state_indices
        track_steps = accept_lens

    flat_slots = num_layers * num_slots
    state = checkpoint_state.view(flat_slots, HV, V, K)
    u = u_cache.view(flat_slots, HV, T, V)
    rawk = rawk_cache.view(flat_slots, H, T, K)
    log_g = g_cache.view(flat_slots, HV, T)

    key = (
        target.compile_key,
        num_layers,
        H,
        HV,
        T,
        warps_per_cta,
        tiles_per_warp,
        bool(use_qk_l2norm_in_kernel),
        int(null_block_id),
        bool(has_track),
    )
    cache = _CACHE.setdefault(key, {})
    stream = cuda.CUstream(
        torch.cuda.current_stream(device=checkpoint_state.device).cuda_stream
    )
    compile_options = gdn_compile_options(
        checkpoint_state.device, cute.EnableTVMFFI(True), cute.GenerateLineInfo(True)
    )
    if "compiled" not in cache:
        cache["compiled"] = build_and_load_cute_dsl_kernel(
            "gdn_replayssm_u_commit",
            make_kernel_name("ucommit", target.arch, *key[1:]),
            lambda: cute.compile[compile_options](
                _launch_gdn_replayssm_u_commit,
                _mark_leading_dynamic(state),
                _mark_leading_dynamic(u),
                _mark_leading_dynamic(rawk),
                _mark_leading_dynamic(log_g),
                _mark_leading_dynamic(ssm_state_indices),
                _mark_leading_dynamic(accept_lens),
                _mark_leading_dynamic(track_indices),
                _mark_leading_dynamic(track_steps),
                NUM_LAYERS=num_layers,
                H=H,
                HV=HV,
                T=T,
                WARPS_PER_CTA=warps_per_cta,
                TILES_PER_WARP=tiles_per_warp,
                USE_QK_L2NORM=bool(use_qk_l2norm_in_kernel),
                NULL_BLOCK_ID=int(null_block_id),
                HAS_TRACK=bool(has_track),
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
    cache["compiled"](
        state,
        u,
        rawk,
        log_g,
        ssm_state_indices,
        accept_lens,
        track_indices,
        track_steps,
        stream,
    )


__all__ = ["commit_gdn_replayssm_u_all_layers"]
