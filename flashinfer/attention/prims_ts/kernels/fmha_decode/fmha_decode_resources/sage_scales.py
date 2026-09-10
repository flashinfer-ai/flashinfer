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

"""Sage attention scale addressing for the decode softmax and epilogue.

Every scale load of the kernel goes through this module. The softmax bodies
load ``sfQ`` of their Q row once per work tile and the raw ``sfK`` of each
KV tile's scale groups; the epilogue receives the per-channel V scales for one
output column range.

The contiguous provider indexes the trtllm-gen flat layout through
``flat_scale_slot`` of :mod:`flashinfer.attention.prims_ts.sage`, the module
that owns the layout. Tokens beyond the sequence end (masked score columns)
and Q rows beyond the valid row count are clamped to the last valid slot:
their scores are masked or discarded, but their scale must stay finite so
masked columns exponentiate to zero.

During the two softmax passes a tile's ``sfK`` words live in the lane's
rotating register array (:class:`RegisterSageKScales`), which hands a pass one
fragment's ``factor * sfK`` at a time (``open``, ``fragment``, ``advance``);
the passes apply ``sfQ`` once per tile themselves.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.experimental.task_scheduling.resources import StageInfo

from ....sage import flat_scale_slot, log2_block_size
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import fmul2

Constexpr = cutlass.Constexpr


@cute.jit
def _load_scale(scale_addr: Int64, index: Int32) -> Float32:
    """Return one FP32 scale loaded from a global memory base address."""
    value_ptr = cutlass.inttoptr(
        scale_addr + Int64(index) * Int64(4),
        mem_space=1,
        dtype=Float32,
    )
    return Float32(value_ptr.load(count=1, alignment=4)[0])


@cute.jit
def load_q_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    q_scale_addr: Int64,
    q_scale_head_stride: Int32,
    *,
    kv_head_idx: Int32,
    local_head_idx: Int32,
    batch_idx: Int32,
    q_token_idx: Int32,
) -> Float32:
    """Return ``sfQ`` for one Q row (token and Q head of one KV head group)."""
    heads_q_per_kv = Int32(cfg.heads_q_per_kv)
    q_head_idx = kv_head_idx * heads_q_per_kv + cute.math.min(
        local_head_idx, heads_q_per_kv - Int32(1)
    )
    q_token_idx = cute.math.min(q_token_idx, Int32(cfg.max_seq_len_q - 1))
    slot = flat_scale_slot(
        batch_idx,
        q_token_idx,
        Int32(cfg.max_seq_len_q),
        log2_block_size(cfg.sage_q_block_size),
    )
    return _load_scale(q_scale_addr, q_head_idx * q_scale_head_stride + slot)


@cute.jit
def load_k_scale(
    cfg: Constexpr[FmhaDecodeConfig],
    k_scale_ptr: cute.Pointer,
    k_scale_head_stride: Int32,
    *,
    kv_head_idx: Int32,
    batch_idx: Int32,
    seq_len_kv: Int32,
    kv_token_idx: Int32,
) -> Float32:
    """Return ``sfK`` for one KV token of one KV head."""
    kv_token_idx = cute.math.min(kv_token_idx, seq_len_kv - Int32(1))
    slot = flat_scale_slot(
        batch_idx,
        kv_token_idx,
        seq_len_kv,
        log2_block_size(cfg.sage_k_block_size),
    )
    return _load_scale(k_scale_ptr.toint(), kv_head_idx * k_scale_head_stride + slot)


def sage_scale_arr_size(cfg: FmhaDecodeConfig) -> int:
    """Return the dequantization multipliers owned by one softmax lane per tile."""
    return cfg.num_softmax_score_fragments * cfg.sage_k_groups_per_fragment


@cute.jit
def scale_pairs_in_place(
    values: cutlass.Array, factor: Float32, count: Constexpr[int]
) -> None:
    """Multiply ``count`` values in place by ``factor``, two at a time."""
    assert count % 2 == 0
    for pair_base in cutlass.range_constexpr(0, count, 2):
        values[pair_base], values[pair_base + 1] = fmul2(
            (factor, factor),
            (Float32(values[pair_base]), Float32(values[pair_base + 1])),
        )


@cute.jit
def load_lane_k_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    *,
    k_scale_ptr: cute.Pointer,
    k_scale_head_stride: Int32,
    kv_head_idx: Int32,
    batch_idx: Int32,
    seq_len_kv: Int32,
    fragment_first_tokens: tuple,
) -> cutlass.Array:
    """Return ``sfK`` for every scale group of the lane's fragments.

    ``fragment_first_tokens[f]`` is the KV token of the first score register of
    fragment ``f``; group ``g`` of that fragment starts
    ``g * softmax_score_fragment_regs / sage_k_groups_per_fragment`` tokens
    later. Entry ``f * groups + g`` of the result is that group's scale.
    """
    groups = cfg.sage_k_groups_per_fragment
    group_tokens = cfg.softmax_score_fragment_regs // groups
    scales = cutlass.Array(
        Float32, sage_scale_arr_size(cfg), space=cutlass.AddressSpace.rmem
    )
    for fragment_idx in cutlass.range_constexpr(cfg.num_softmax_score_fragments):
        for group_idx in cutlass.range_constexpr(groups):
            scales[fragment_idx * groups + group_idx] = load_k_scale(
                cfg,
                k_scale_ptr,
                k_scale_head_stride,
                kv_head_idx=kv_head_idx,
                batch_idx=batch_idx,
                seq_len_kv=seq_len_kv,
                kv_token_idx=fragment_first_tokens[fragment_idx]
                + Int32(group_idx * group_tokens),
            )
    return scales


@dataclass
class RegisterSageKScales:
    """The lane's ``sfK`` multipliers as a rotating register array.

    K blocks of 16 tokens and larger have at most two scale groups per K32
    fragment, so a lane holds ``sage_scale_arr_size`` words per tile (eight
    for the 16-token block on KV256). The routed ``sage_scale_arr`` carries
    the raw ``sfK`` of the lane's fragment groups in fragment order. A pass
    opens it with its factor, which is applied once per tile, reads the
    leading ``groups`` entries for the fragment at hand and rotates the
    array down by one fragment, so neither the unrolled nor the rolled
    fragment loop indexes registers at run time.
    """

    cfg: FmhaDecodeConfig

    @property
    def routed_words(self) -> int:
        """Return the entries of the routed ``sage_scale_arr``."""
        return sage_scale_arr_size(self.cfg)

    @cute.jit
    def open(
        self,
        stage_info: StageInfo,
        scale_arr: cutlass.Array,
        factor: Float32 | None,
    ) -> cutlass.Array:
        """Return the tile's ``factor * sfK`` words as a fresh rotating array."""
        _ = stage_info
        arr_size = sage_scale_arr_size(self.cfg)
        words = cutlass.Array(Float32, arr_size, space=cutlass.AddressSpace.rmem)
        for entry in cutlass.range_constexpr(arr_size):
            words[entry] = Float32(scale_arr[entry])
        if cutlass.const_expr(factor is not None):
            scale_pairs_in_place(words, factor, arr_size)
        return words

    @cute.jit
    def fragment(
        self, words: cutlass.Array, fragment: Int32, factor: Float32 | None
    ) -> cutlass.Array:
        """Return the leading fragment's ``groups`` words (``factor`` is in them)."""
        _ = fragment, factor
        groups = self.cfg.sage_k_groups_per_fragment
        values = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
        for group_idx in cutlass.range_constexpr(groups):
            values[group_idx] = Float32(words[group_idx])
        return values

    @cute.jit
    def advance(self, words: cutlass.Array) -> None:
        """Rotate the array down by one fragment."""
        groups = self.cfg.sage_k_groups_per_fragment
        for entry in cutlass.range_constexpr(sage_scale_arr_size(self.cfg) - groups):
            words[entry] = Float32(words[entry + groups])


SageKScales = RegisterSageKScales


def make_sage_k_scales(cfg: FmhaDecodeConfig) -> SageKScales | None:
    """Return the K scale strategy of one softmax instance, ``None`` without Sage."""
    if not cfg.use_sage_attention:
        return None
    return RegisterSageKScales(cfg)


def staged_v_channel_scale_entries(cfg: FmhaDecodeConfig) -> int:
    """Return the SMEM floats holding one KV head's V scales and means.

    The scales occupy ``[0, headdim)``; with ``sage_v_mean`` the means follow
    at ``[headdim, 2 * headdim)``.
    """
    return cfg.headdim * (2 if cfg.sage_v_mean else 1)


@cute.jit
def stage_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    v_scale_ptr: cute.Pointer,
    v_mean_ptr: cute.Pointer | None,
    staged: cutlass.Array,
    *,
    kv_head_idx: Int32,
    thread_idx: Int32,
    num_threads: Constexpr[int],
) -> None:
    """Copy one KV head's per-channel V scales and means into SMEM.

    Both tensors are ``[Hkv, D]`` FP32. The ``num_threads`` callers each
    move ``headdim / num_threads`` channels, so the global loads are issued
    once per tile while the correction warps are otherwise idle, instead of
    inside the output store loop where their latency lands on the tile tail.
    Callers order the writes before the epilogue reads with a barrier, and
    place a second barrier after the last read of a tile so the next tile's
    writes cannot overtake it.
    """
    assert cfg.headdim % num_threads == 0
    head_base = Int64(kv_head_idx) * Int64(cfg.headdim)
    for base in cutlass.range_constexpr(0, cfg.headdim, num_threads):
        channel = Int32(base) + thread_idx
        byte_offset = (head_base + Int64(channel)) * Int64(4)
        staged[channel] = Float32(
            cutlass.inttoptr(
                v_scale_ptr.toint() + byte_offset, mem_space=1, dtype=Float32
            ).load(count=1, alignment=4)[0]
        )
        if cutlass.const_expr(cfg.sage_v_mean):
            staged[channel + Int32(cfg.headdim)] = Float32(
                cutlass.inttoptr(
                    v_mean_ptr.toint() + byte_offset, mem_space=1, dtype=Float32
                ).load(count=1, alignment=4)[0]
            )


@cute.jit
def load_staged_v_channel_scales(
    cfg: Constexpr[FmhaDecodeConfig],
    staged: cutlass.Array,
    *,
    first_col: Int32,
    count: Constexpr[int],
) -> tuple[cutlass.Array, cutlass.Array]:
    """Return ``count`` staged V scales and means from ``first_col``.

    Without ``sage_v_mean`` the returned mean array is zero so callers can
    apply one fused multiply-add.
    """
    assert count % 4 == 0
    scales = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    means = cutlass.Array(Float32, count, space=cutlass.AddressSpace.rmem)
    for chunk in cutlass.range_constexpr(0, count, 4):
        chunk_col = first_col + Int32(chunk)
        scale_vec = (staged.data_ptr() + chunk_col).load(count=4, alignment=16)
        for elem in cutlass.range_constexpr(4):
            scales[chunk + elem] = Float32(scale_vec[elem])
        if cutlass.const_expr(cfg.sage_v_mean):
            mean_vec = (staged.data_ptr() + chunk_col + Int32(cfg.headdim)).load(
                count=4, alignment=16
            )
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(mean_vec[elem])
        else:
            for elem in cutlass.range_constexpr(4):
                means[chunk + elem] = Float32(0.0)
    return scales, means
