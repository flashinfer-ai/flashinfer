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

"""CuTe DSL ReplaySSM fold-every-commit kernel for Gated DeltaNet.

Replay each request's accepted raw ``(v, k, log-g, beta)`` prefix into an FP32
checkpoint in place, with optional writes to an extra tracked state slot.

The optimized SM100 mapping keeps one 8-row state stripe in each warp's
registers.  A CTA owns a contiguous V-row tile, so checkpoint loads and stores
are 128-bit vectorized along K while all recurrence traffic stays on chip.
"""

import torch
import cutlass
import cutlass.cute as cute
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
_ROWS_PER_WARP = 8
_L2_EPS = 1.0e-6


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


@cute.kernel
def _gdn_replayssm_fold_kernel(
    state: cute.Tensor,
    rawv: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    MAX_CACHE_LEN: cutlass.Constexpr[int],
    WARPS_PER_CTA: cutlass.Constexpr[int],
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
    slots_per_layer = state.shape[6] // NUM_LAYERS

    requested_slot = state_indices[i_n]
    n_commit = accept_lens[i_n]
    is_live = requested_slot > NULL_BLOCK_ID and n_commit > 0
    safe_slot = requested_slot
    if not is_live:
        safe_slot = cutlass.Int32(0)
    flat_slot = i_layer * slots_per_layer + safe_slot

    copy_f32x4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), _F32, num_bits_per_copy=128
    )
    copy_bf16x4 = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), _BF16, num_bits_per_copy=64
    )

    state_registers = cute.make_rmem_tensor((_CHANNELS_PER_LANE, _ROWS_PER_WARP), _F32)
    if is_live:
        global_state = _aligned_tensor(
            state[(None, lane, None, warp, row_tile, i_hv, flat_slot)], 16
        )
        cute.copy(copy_f32x4, global_state, state_registers)

        key = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _F32)
        key_bf16 = cute.make_rmem_tensor((_CHANNELS_PER_LANE,), _BF16)

        if cutlass.const_expr(HAS_TRACK):
            requested_track_slot = track_indices[i_n]
            track_step = track_steps[i_n]
            track_live = requested_track_slot > NULL_BLOCK_ID
            safe_track_slot = requested_track_slot
            if not track_live:
                safe_track_slot = cutlass.Int32(0)
            flat_track_slot = i_layer * slots_per_layer + safe_track_slot

        for t in cutlass.range(n_commit, unroll=1):
            cute.copy(
                copy_bf16x4,
                _aligned_tensor(rawk[(None, lane, t, i_h, flat_slot)], 8),
                key_bf16,
            )
            key_norm = _F32(0.0)
            for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                key[channel] = key_bf16[channel].to(_F32)
                key_norm += key[channel] * key[channel]
            key_norm = cute.arch.warp_reduction_sum(key_norm)
            if cutlass.const_expr(USE_QK_L2NORM):
                inv_key_norm = cute.rsqrt(key_norm + _L2_EPS)
                for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                    key[channel] *= inv_key_norm

            decay = _F32(0.0)
            beta_value = _F32(0.0)
            if lane == 0:
                decay = cute.exp(log_g[(t, i_hv, flat_slot)], fastmath=True)
                beta_value = beta[(t, i_hv, flat_slot)]
            decay = cute.arch.shuffle_sync(decay, 0)
            beta_value = cute.arch.shuffle_sync(beta_value, 0)

            raw_value = _F32(0.0)
            if lane < _ROWS_PER_WARP:
                raw_value = rawv[(lane, warp, row_tile, t, i_hv, flat_slot)].to(_F32)

            # Interleave four independent row reductions so the scheduler can
            # cover each SHFL/FADD dependency chain with useful work from the
            # other rows.  The per-row summation order remains unchanged.
            for row_group in cutlass.range_constexpr(_ROWS_PER_WARP // 4):
                row0 = row_group * 4
                row1 = row0 + 1
                row2 = row0 + 2
                row3 = row0 + 3
                prediction0 = _F32(0.0)
                prediction1 = _F32(0.0)
                prediction2 = _F32(0.0)
                prediction3 = _F32(0.0)
                for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                    state_registers[(channel, row0)] *= decay
                    state_registers[(channel, row1)] *= decay
                    state_registers[(channel, row2)] *= decay
                    state_registers[(channel, row3)] *= decay
                    prediction0 += state_registers[(channel, row0)] * key[channel]
                    prediction1 += state_registers[(channel, row1)] * key[channel]
                    prediction2 += state_registers[(channel, row2)] * key[channel]
                    prediction3 += state_registers[(channel, row3)] * key[channel]
                for offset in (16, 8, 4, 2, 1):
                    prediction0 += cute.arch.shuffle_sync_bfly(
                        prediction0, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    prediction1 += cute.arch.shuffle_sync_bfly(
                        prediction1, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    prediction2 += cute.arch.shuffle_sync_bfly(
                        prediction2, offset=offset, mask=-1, mask_and_clamp=31
                    )
                    prediction3 += cute.arch.shuffle_sync_bfly(
                        prediction3, offset=offset, mask=-1, mask_and_clamp=31
                    )
                value0 = cute.arch.shuffle_sync(raw_value, row0)
                value1 = cute.arch.shuffle_sync(raw_value, row1)
                value2 = cute.arch.shuffle_sync(raw_value, row2)
                value3 = cute.arch.shuffle_sync(raw_value, row3)
                delta0 = (value0 - prediction0) * beta_value
                delta1 = (value1 - prediction1) * beta_value
                delta2 = (value2 - prediction2) * beta_value
                delta3 = (value3 - prediction3) * beta_value
                for channel in cutlass.range_constexpr(_CHANNELS_PER_LANE):
                    state_registers[(channel, row0)] += delta0 * key[channel]
                    state_registers[(channel, row1)] += delta1 * key[channel]
                    state_registers[(channel, row2)] += delta2 * key[channel]
                    state_registers[(channel, row3)] += delta3 * key[channel]

            if cutlass.const_expr(HAS_TRACK):
                if t == track_step and track_live:
                    track_state = _aligned_tensor(
                        state[
                            (
                                None,
                                lane,
                                None,
                                warp,
                                row_tile,
                                i_hv,
                                flat_track_slot,
                            )
                        ],
                        16,
                    )
                    cute.copy(copy_f32x4, state_registers, track_state)

        cute.copy(copy_f32x4, state_registers, global_state)


@cute.jit
def _launch_gdn_replayssm_fold(
    state: cute.Tensor,
    rawv: cute.Tensor,
    rawk: cute.Tensor,
    log_g: cute.Tensor,
    beta: cute.Tensor,
    state_indices: cute.Tensor,
    accept_lens: cute.Tensor,
    track_indices: cute.Tensor,
    track_steps: cute.Tensor,
    NUM_LAYERS: cutlass.Constexpr[int],
    H: cutlass.Constexpr[int],
    HV: cutlass.Constexpr[int],
    MAX_CACHE_LEN: cutlass.Constexpr[int],
    WARPS_PER_CTA: cutlass.Constexpr[int],
    USE_QK_L2NORM: cutlass.Constexpr[bool],
    NULL_BLOCK_ID: cutlass.Constexpr[int],
    HAS_TRACK: cutlass.Constexpr[bool],
    stream: cuda.CUstream,
):
    rows_per_cta = WARPS_PER_CTA * _ROWS_PER_WARP
    row_tiles = _V // rows_per_cta

    state_view = cute.make_tensor(
        state.iterator,
        cute.make_layout(
            (
                _CHANNELS_PER_LANE,
                32,
                _ROWS_PER_WARP,
                WARPS_PER_CTA,
                row_tiles,
                HV,
                state.shape[0],
            ),
            stride=(
                1,
                _CHANNELS_PER_LANE,
                _K,
                _ROWS_PER_WARP * _K,
                rows_per_cta * _K,
                _V * _K,
                HV * _V * _K,
            ),
        ),
    )
    rawv_view = cute.make_tensor(
        rawv.iterator,
        cute.make_layout(
            (
                _ROWS_PER_WARP,
                WARPS_PER_CTA,
                row_tiles,
                MAX_CACHE_LEN,
                HV,
                rawv.shape[0],
            ),
            stride=(
                1,
                _ROWS_PER_WARP,
                rows_per_cta,
                _V,
                MAX_CACHE_LEN * _V,
                HV * MAX_CACHE_LEN * _V,
            ),
        ),
    )
    rawk_view = cute.make_tensor(
        rawk.iterator,
        cute.make_layout(
            (_CHANNELS_PER_LANE, 32, MAX_CACHE_LEN, H, rawk.shape[0]),
            stride=(
                1,
                _CHANNELS_PER_LANE,
                _K,
                MAX_CACHE_LEN * _K,
                H * MAX_CACHE_LEN * _K,
            ),
        ),
    )
    gate_layout = cute.make_layout(
        (MAX_CACHE_LEN, HV, log_g.shape[0]),
        stride=(1, MAX_CACHE_LEN, HV * MAX_CACHE_LEN),
    )
    log_g_view = cute.make_tensor(log_g.iterator, gate_layout)
    beta_view = cute.make_tensor(beta.iterator, gate_layout)

    _gdn_replayssm_fold_kernel(
        state_view,
        rawv_view,
        rawk_view,
        log_g_view,
        beta_view,
        state_indices,
        accept_lens,
        track_indices,
        track_steps,
        NUM_LAYERS,
        H,
        HV,
        MAX_CACHE_LEN,
        WARPS_PER_CTA,
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


def commit_gdn_replayssm_fold_all_layers(
    checkpoint_state: torch.Tensor,
    rawv_cache: torch.Tensor,
    rawk_cache: torch.Tensor,
    g_cache: torch.Tensor,
    beta_cache: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    accept_lens: torch.Tensor,
    max_cache_len: int,
    num_k_heads: int,
    track_state_indices: torch.Tensor | None = None,
    track_steps: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = True,
    null_block_id: int = -1,
    *,
    backend: str = "auto",
) -> None:
    """Replay the accepted prefix into every layer's FP32 checkpoint.

    Checkpoints have shape ``[layers, slots, HV, 128, 128]`` (FP32, K-last).
    Raw K/V windows have shape ``[layers, slots, H/HV, T, 128]`` (BF16);
    log-decay and beta have shape ``[layers, slots, HV, T]`` (FP32).
    All tensors must be contiguous and on the same SM100/SM103 CUDA device.

    ``ssm_state_indices`` and ``accept_lens`` are int32 vectors of length B.
    Each live slot must be unique and in range; accepted lengths must lie in
    [0, T]. Slots <= ``null_block_id`` and zero accepted lengths are no-ops.
    Optional ``track_state_indices`` and zero-based ``track_steps`` save the
    state after a selected accepted token. Track slots must be distinct from
    each other and all live checkpoint slots. A step outside the accepted
    prefix or a null track slot leaves its destination unchanged.

    Verify leaves checkpoints frozen and overwrites these windows. Call commit
    once after choosing the accepted prefix, before verifying the next window.
    The cache is neither shifted nor cleared; unaccepted tokens are discarded.
    Use the same normalization setting in verify and commit. Index values and
    aliasing constraints are caller responsibilities (no device synchronization).
    Only FP32 checkpoints are supported.
    """
    if checkpoint_state.ndim != 5:
        raise ValueError("checkpoint_state must be [layers, slots, HV, V, K]")
    num_layers, num_slots, HV, V, K = checkpoint_state.shape
    target = gdn_device_target(checkpoint_state.device)
    if target.major != 10:
        raise ValueError("ReplaySSM commit currently requires SM100/SM103")
    if num_layers < 1 or num_slots < 1 or max_cache_len < 1:
        raise ValueError("layers, slots, and max_cache_len must be positive")
    H = int(num_k_heads)
    if ssm_state_indices.ndim != 1:
        raise ValueError("ssm_state_indices must be one-dimensional")
    if null_block_id < -1:
        raise ValueError("null_block_id must be >= -1")
    B = ssm_state_indices.shape[0]
    has_track = track_state_indices is not None or track_steps is not None
    if has_track and (track_state_indices is None or track_steps is None):
        raise ValueError("track_state_indices and track_steps are paired")
    if (K, V) != (_K, _V):
        raise ValueError(f"CuTe ReplaySSM fold requires K=V=128, got K={K}, V={V}")
    if H <= 0 or HV <= 0 or HV % H != 0:
        raise ValueError(f"HV={HV} must be divisible by H={H}")
    warps_per_cta = 4
    if backend not in ("auto", "simt", "tcgen05"):
        raise ValueError("backend must be auto, simt, or tcgen05")

    expected = (
        (
            rawv_cache,
            (num_layers, num_slots, HV, max_cache_len, V),
            torch.bfloat16,
            "rawv_cache",
        ),
        (
            rawk_cache,
            (num_layers, num_slots, H, max_cache_len, K),
            torch.bfloat16,
            "rawk_cache",
        ),
        (g_cache, (num_layers, num_slots, HV, max_cache_len), torch.float32, "g_cache"),
        (
            beta_cache,
            (num_layers, num_slots, HV, max_cache_len),
            torch.float32,
            "beta_cache",
        ),
    )
    if checkpoint_state.dtype != torch.float32:
        raise TypeError("checkpoint_state must be float32")
    for tensor, shape, dtype, name in expected:
        if tuple(tensor.shape) != tuple(shape):
            raise ValueError(
                f"{name} must have shape {shape}, got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")

    tensors = (checkpoint_state, rawv_cache, rawk_cache, g_cache, beta_cache)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("checkpoint and ReplaySSM cache tensors must be contiguous")
    if not all(t.device == checkpoint_state.device for t in tensors):
        raise ValueError("all checkpoint/cache tensors must be on one device")
    for tensor, name in (
        (ssm_state_indices, "ssm_state_indices"),
        (accept_lens, "accept_lens"),
    ):
        if tensor.shape != (B,) or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be contiguous int32 with shape [{B}]")
        if not tensor.is_contiguous() or tensor.device != checkpoint_state.device:
            raise ValueError(f"{name} must be contiguous on the checkpoint device")
    if has_track:
        assert track_state_indices is not None and track_steps is not None
        for tensor, name in (
            (track_state_indices, "track_state_indices"),
            (track_steps, "track_steps"),
        ):
            if tensor.shape != (B,) or tensor.dtype != torch.int32:
                raise ValueError(f"{name} must be contiguous int32 with shape [{B}]")
            if not tensor.is_contiguous() or tensor.device != checkpoint_state.device:
                raise ValueError(f"{name} must be contiguous on the checkpoint device")
        track_indices = track_state_indices
    else:
        track_indices = ssm_state_indices
        track_steps = accept_lens

    # Compact strides preserve alignment only when the base is aligned too.
    for name, tensor in (
        ("checkpoint_state", checkpoint_state),
        ("rawv_cache", rawv_cache),
        ("rawk_cache", rawk_cache),
        ("g_cache", g_cache),
        ("beta_cache", beta_cache),
        ("ssm_state_indices", ssm_state_indices),
        ("accept_lens", accept_lens),
        ("track_state_indices", track_indices),
        ("track_steps", track_steps),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must have a 16-byte aligned base address")

    supports_tcgen = max_cache_len == 8 and use_qk_l2norm_in_kernel and not has_track
    if backend == "tcgen05" and not supports_tcgen:
        raise ValueError("tcgen05 commit requires normalized T=8 without tracking")
    if B == 0:
        return
    if supports_tcgen and backend != "simt":
        from .gdn_replayssm_spec_fold_tcgen import run_replayssm_fold_tcgen

        run_replayssm_fold_tcgen(
            checkpoint_state,
            rawv_cache,
            rawk_cache,
            g_cache,
            beta_cache,
            ssm_state_indices,
            accept_lens,
            null_block_id=null_block_id,
        )
        return

    flat_slots = num_layers * num_slots
    state = checkpoint_state.view(flat_slots, HV, V, K)
    rawv = rawv_cache.view(flat_slots, HV, max_cache_len, V)
    rawk = rawk_cache.view(flat_slots, H, max_cache_len, K)
    log_g = g_cache.view(flat_slots, HV, max_cache_len)
    beta = beta_cache.view(flat_slots, HV, max_cache_len)

    key = (
        target.compile_key,
        num_layers,
        H,
        HV,
        max_cache_len,
        warps_per_cta,
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
            "gdn_replayssm_spec_fold",
            make_kernel_name("fold", target.arch, *key[1:]),
            lambda: cute.compile[compile_options](
                _launch_gdn_replayssm_fold,
                _mark_leading_dynamic(state),
                _mark_leading_dynamic(rawv),
                _mark_leading_dynamic(rawk),
                _mark_leading_dynamic(log_g),
                _mark_leading_dynamic(beta),
                _mark_leading_dynamic(ssm_state_indices),
                _mark_leading_dynamic(accept_lens),
                _mark_leading_dynamic(track_indices),
                _mark_leading_dynamic(track_steps),
                NUM_LAYERS=num_layers,
                H=H,
                HV=HV,
                MAX_CACHE_LEN=max_cache_len,
                WARPS_PER_CTA=warps_per_cta,
                USE_QK_L2NORM=bool(use_qk_l2norm_in_kernel),
                NULL_BLOCK_ID=int(null_block_id),
                HAS_TRACK=bool(has_track),
                stream=stream,
            ),
            extra_key_files=(__file__,),
        )
    cache["compiled"](
        state,
        rawv,
        rawk,
        log_g,
        beta,
        ssm_state_indices,
        accept_lens,
        track_indices,
        track_steps,
        stream,
    )


__all__ = ["commit_gdn_replayssm_fold_all_layers"]
