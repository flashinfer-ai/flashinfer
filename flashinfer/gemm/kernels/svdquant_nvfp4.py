# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Fused NVFP4-SVDQuant linear for FlashInfer (B200 / sm_100a): an NVFP4 main GEMM plus a bf16 low-rank
# (LoRA-style) correction, Y = dequant(nvfp4(X) @ nvfp4(R)^T) + (X @ L2^T) @ L1^T.

"""
Self-contained runtime for the fused NVFP4 SVDQuant linear layer (B200 / sm_100a).

This single module holds the full deployable pipeline -- the (quant + down) front-end and the fused
dual-TMEM main+low-rank GEMM -- behind a small public surface:

  * prepare_svdquant_state / mm_fp4_svdquant : the reusable, CUDA-graph-safe host fast-path.
  * run_kernel_a / run_quantizer / run_down_projection : the on-device (quant + down) front-end.
  * run_kernel_b : the fused NVFP4 main GEMM + bf16 low-rank up-projection, one fused-add epilogue.
  * _run_kernels : the private on-device kernel sequence shared by the fast-path and the harness.

FRONT-END (run_kernel_a). Emits BOTH halves the fused linear consumes, on device, via two device
kernels (X is read from gmem twice -- once each). Both reads are device-side (no CPU staging):

  1. REAL on-device packed Xq (E2M1 4-bit nibbles, 2/byte, K-major: even K in the low nibble, odd K
     in the high nibble) + sf_x (E4M3 per-16-element block scales, plain [M, I//16] byte layout)
     computed against a PROVIDED per-tensor global scale gscale_x: per-16-block amax over K,
     sf = E4M3-round(block_amax / (6 * gscale_x)), Xq = E2M1-round(x / (sf * gscale_x)), using
     cutlass.cute's f32->E4M3 / f32->E2M1 hardware conversions (the same round-to-nearest conversions
     the block-scaled GEMM consumes). The on-device E2M1/E4M3 quantizer (ActivationQuantizer) is a
     tcgen05-free CuteDSL kernel: ONE THREAD PER 16-ELEMENT K-BLOCK -- a flat grid over the M*(I/16)
     independent NVFP4 blocks (thread g -> row m = g//nblk, block b = g%nblk, m-major so a warp reads
     32 contiguous blocks of one row, coalesced). Each thread reads its 16 X elements from gmem (f32),
     reduces the block amax, rounds the E4M3 block scale and each element to E2M1, and packs two
     nibbles/byte (signed-zero canonicalized to +0). sf_x is the plain [M, I//16] E4M3 byte format the
     reference packer produces, so nvfp4_gemm._make_sf_mma_tensor / _fp4_from_packed consume it
     UNCHANGED. The quantizer consumes f32 X (NOT bf16): quantizing the full-precision X is
     dequant-parity with the reference packer of the same f32 X (a few E4M3/E2M1 round-to-nearest TIES
     differ between cute's and torch's conversions, so a same-device bytewise compare shows a few tie
     mismatches -- each a single E2M1 grid step -- not bytewise-exact); feeding bf16 X instead would
     impose a rank-independent input-rounding floor.
  2. bf16 D = X @ L2^T   [M, r]   -- the low-rank "down" projection (L2 is [r, I], so the contraction
     is over K = I, producing r output columns). The bf16 GEMM (DownProjection) is a public-wheel
     tcgen05 bf16 MMA with a multi-stage TMA K-loop (make_trivial_tiled_mma + PipelineTmaUmma),
     contracting the FULL K = I. The MMA tile is specialized by shape (down_tile_for_shape): r<=64 uses
     a SKINNY 64-col N-tile, and the long-K family ALSO halves the M-tile to 64 (more M-tile CTAs, the
     occupancy lever for the X-stream-bound long-K down) and uses split-K (the long K-loop partitioned
     into concurrent slices whose f32 partials SplitKReduce sums to bf16 D).

A single-kernel read-once fusion (one f32 X read driving both the quant stores and the bf16 D MMA) is
blocked by an MLIR compile-time explosion when the NVFP4 quant body is co-located in the same module
as the tcgen05 UMMA + multi-stage TMA pipeline (it is the co-location, not unrolling); the runtime
deadlock is solved but the compile wall is not, so the production front-end is the two device kernels
above. They feed the fused main identically.

FUSED MAIN (run_kernel_b). ONE public-wheel CuteDSL kernel holds TWO heterogeneous TMEM accumulators
and combines them in a single fused-add epilogue, over the real problem sizes (multi-tile grid +
multi-stage main K-loop):

  main_acc = NVFP4 block-scaled MMA  Xq @ Rq^T,  contract over K = I (LARGE) via a multi-stage
             pipelined TMA/UMMA loop; E4M3 vec-16 block scales; the per-tensor FP32 global scale
             (gscale_x * gscale_w) is folded OUTSIDE the MMA (epilogue).
  lora_acc = bf16 MMA  D @ L1^T,  contract over K = r (<= 64) -- a SINGLE tile / different K-atom on an
             independent pipeline: the D/L1 TMA is issued BEFORE the main K-loop (overlapping it) and
             the bf16 MMA AFTER it; one acc commit then gates both accumulators to the epilogue.
  epilogue : Y = dequant(main_acc) * (gscale_x * gscale_w) + lora_acc  -> bf16.

`enable_lora` (compile-time): True -> the full fused dual-TMEM add epilogue; False -> the plain-NVFP4
variant (identical main schedule, the lora load/MMA/add are NOT issued, store dequant(main) only). The
two variants share a bit-identical main accumulation, so with L1=L2=0 (=> D=0 => lora_acc=+0, exact in
IEEE-754) the fused output reproduces the plain output bit-for-bit through the REAL fused epilogue.

A 2-CTA cooperative variant (KernelB2CTA) computes a 256-row cluster M-tile with a CtaGroup.TWO cluster
(128 rows/CTA), halving the M-tile count for the wide-O / up-bound rows that the per-(M,O)-tile
dual-TMEM add dominates; _wide_o_2cta selects it. The 1-CTA KernelB is unchanged for every other shape.

Public nvidia-cutlass-dsl wheel ONLY -- no private-module dependency.
"""
import os
import sys
from typing import Type

import cuda.bindings.driver as cuda
import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.pipeline as pipeline
import cutlass.torch as cutlass_torch
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

# --- self-contained runtime: no local-kernel imports, no reference-oracle import ---
# This deployable runtime is oracle-free (FP4_E2M1_MAX / E4M3_MAX are inlined above), and the packed-FP4 /
# E4M3-SF operand builders + the SF MMA-layout converter that used to live in nvfp4_gemm.py /
# blockscaled_gemm_sm100.py are inlined below, so this file is a single import-clean module (the flashinfer
# mirror needs no co-located siblings). _HERE is kept (NO sys.path mutation): harness helpers (e.g.
# _flux_shapes) and the dev shims resolve bench/problem_sizes.yaml relative to it.
_HERE = os.path.dirname(os.path.abspath(__file__))
from cutlass.cute.runtime import from_dlpack  # noqa: E402  (used by the inlined operand builders below)

# --- front-end (quant + down) dtype aliases / constants ---
LORA_DTYPE = cutlass.BFloat16     # low-rank operands (X cast to bf16, L2, D) + the resident X tile
ACC_DTYPE = cutlass.Float32
MMA_TILER_MN = (128, 128)         # (default M tile, default down-projection N tile)
SKINNY_N_TILE = 64                # skinny down-projection N tile for r<=64: D=X@L2^T has only r<=64
                                  # output columns, so a 64-col tile avoids paying the full 128-col tile
LONG_K_M_TILE = 64                # long-K down M tile: halving M -> 2x the M-tile CTAs (occupancy)
LONG_K_SPLIT = 2                   # long-K down split-K factor: partition the long K-loop into 2 slices
                                  # run concurrently -> a shorter dependent K-loop; the WINNING long-K
                                  # lever (GB200 r18_down_split: 320->172 us, 1.84x; S=4 no better than S=2)
LONG_K_THRESHOLD = 24576          # I >= this selects the long-K occupancy + split-K path (single_linear2)
BF16_TILE_K = 64                  # bf16 tcgen05 K-tile (inst-K 16 x 4)
SF_BLOCK = 16                     # NVFP4 block size
FP4_E2M1_MAX = 6.0
E4M3_MAX = 448.0      # local (was reference_svdquant_nvfp4.E4M3_MAX); keeps the deployable runtime oracle-free
QUANT_M_TILE = 128                # quantizer threads per CTA (now one thread per 16-element K-block)

# --- fused main (NVFP4 + low-rank) dtype aliases / constants ---
ELEM_DTYPE = cutlass.Float4E2M1FN     # E2M1 4-bit main operands
SF_DTYPE = cutlass.Float8E4M3FN       # E4M3 block scales (NVF4, not the E8M0 MX default)
OUT_DTYPE = cutlass.BFloat16
SF_VEC = 16
NVFP4_INST_K = 64                     # tcgen05 NVFP4 instruction K
NVFP4_TILE_K = 256                    # 64 * 4 K-blocks per main K-tile


# --- inlined operand builders + SF MMA-layout converter (relocated verbatim from the former
# kernels/nvfp4_gemm.py and the vendored kernels/blockscaled_gemm_sm100.py) so this runtime has zero
# local-kernel imports. ELEM_DTYPE / SF_DTYPE / SF_VEC are the shared constants defined above. ----------
_ATOM_M = (32, 4)
_ATOM_K = 4


def _ceil_div(a, b):
    return (a + b - 1) // b


@cute.jit
def cvt_sf_MKL_to_M32x4xrm_K4xrk_L(
    sf_ref_tensor: cute.Tensor,
    sf_mma_tensor: cute.Tensor,
):
    """Convert scale factor tensor from MKL layout to mma specification M(32x4xrest_m)xK(4xrest_k)xL layout"""
    # sf_mma_tensor has flatten shape (32, 4, rest_m, 4, rest_k, l)
    # group to ((32, 4, rest_m), (4, rest_k), l)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 0, 3)
    sf_mma_tensor = cute.group_modes(sf_mma_tensor, 1, 3)
    for i in cutlass.range(cute.size(sf_ref_tensor)):
        mkl_coord = sf_ref_tensor.layout.get_hier_coord(i)
        sf_mma_tensor[mkl_coord] = sf_ref_tensor[mkl_coord]


_cvt_sf_to_mma = cvt_sf_MKL_to_M32x4xrm_K4xrk_L


def _make_sf_mma_tensor(values_2d, mn, k):
    """Build the swizzled MMA-layout E4M3 scale-factor cute tensor.

    values_2d[mn, sf_k] (optional) are the per-block E4M3 scales to inject; when None the
    scratch keeps random values (for data-independent timing). Mirrors the example's
    scale-factor construction so the M(32x4)xK(4) MMA layout is exactly what the kernel reads.
    """
    l = 1
    sf_k = _ceil_div(k, SF_VEC)
    ref_shape = (l, mn, sf_k)
    mma_shape = (
        l,
        _ceil_div(mn, _ATOM_M[0] * _ATOM_M[1]),
        _ceil_div(sf_k, _ATOM_K),
        _ATOM_M[0],
        _ATOM_M[1],
        _ATOM_K,
    )
    ref_cpu = cutlass_torch.create_and_permute_torch_tensor(
        ref_shape, torch.float32, permute_order=(1, 2, 0),
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=1, max_val=3),
    )  # shape (mn, sf_k, l)
    if values_2d is not None:
        ref_cpu[:, :, 0] = values_2d.to(torch.float32)
    mma_cpu = cutlass_torch.create_and_permute_torch_tensor(
        mma_shape, torch.float32, permute_order=(3, 4, 1, 5, 2, 0),
        init_type=cutlass_torch.TensorInitType.RANDOM,
        init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1),
    )
    _cvt_sf_to_mma(from_dlpack(ref_cpu), from_dlpack(mma_cpu))  # ref MKL -> MMA swizzle
    mma_cuda = mma_cpu.cuda()
    cute_tensor, cute_torch = cutlass_torch.cute_tensor_like(
        mma_cpu, SF_DTYPE, is_dynamic_layout=True, assumed_align=16
    )
    cute_tensor = cutlass_torch.convert_cute_tensor(
        mma_cuda, cute_tensor, SF_DTYPE, is_dynamic_layout=True
    )
    return cute_tensor, cute_torch


def _fp4_from_packed(mode0, k, packed_2d):
    """A (m,k,l) or B (n,k,l) Float4E2M1FN cute tensor built directly from the raw packed NVFP4
    bytes (2 E2M1 nibbles/byte, K fastest) — no f32 round-trip, no dequantized float. The kernel
    stores fp4 packed 2/byte, all mode0*k elements row-major into the first mode0*k/2 contiguous
    storage bytes of the int8 staging buffer (verified byte-identical to the f32->FP4 path by
    probe_fp4_encoding). packed_2d: uint8 [mode0, k//2] from pack_nvfp4 / pack_with_gscale."""
    l = 1
    ref0 = cutlass_torch.matrix(l, mode0, k, False, cutlass.Float32)  # (mode0,k,l) k-major strides
    buf = torch.empty_like(ref0, dtype=torch.int8, device="cuda")     # storage offset i*k+j
    buf.zero_()
    flat = buf.as_strided((mode0 * k,), (1,))                         # linear view of storage
    flat[: mode0 * k // 2] = packed_2d.reshape(-1).to(torch.int8).cuda()
    t = from_dlpack(buf, assumed_align=16)
    t.element_type = ELEM_DTYPE
    t = t.mark_layout_dynamic(leading_dim=cutlass_torch.get_leading_dim(buf))
    t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=32)
    return t, buf


# Small-M-aware long-K split-K factors, keyed by (I, M, r). The contraction dim I is part of the key
# so different long-K shapes never collide at the same (M, r): single_linear2 (I=24576) and a future
# long-K shape (e.g. double_ff_out, I=18432) each carry their own entries and each divides ITS OWN
# K-tile count. The long-K down GEMM (D = X @ L2^T, K=I, N=r<=64) is OCCUPANCY-STARVED at small M: with
# the 64-row M tile it launches only ceil(M/64) M-tile CTAs to stream the long K, so the down runs at a
# fraction of the 148 SMs until split-K adds num_k_split concurrent K-slice CTAs (target ~128 resident
# CTAs = M_tiles * split). The per-(m,n) parallel SplitKReduce is ~flat across the split (~14 us), so it
# no longer caps the split factor.
# single_linear2 (I=24576, K-tile count 384) GB200 FULL-PIPELINE selection over the valid divisors
# {2,3,4,6,8,12,16}, REPEATED (runs r1_split_select + r2_core), median per (M, rank):
#   M=1024 -> 8 (both ranks, ROBUST: +11.5/+12.1% vs +12-15% runner-ups, consistent across both runs).
#   M=2048 -> 4 (both ranks). split-4 vs split-8 is a within-MAD TIE at M=2048 -- the per-rank "best"
#     FLIPS between runs (r1: r32->4, r64->8; r2: r32->8, r64->4), and split-4 averages marginally
#     better across both runs (~+13.35% vs ~+13.68%). split-4 is the principled pick: 32 M-tiles x 4 =
#     128 CTAs EXACTLY saturates the 148 SMs (ncu r1_ncu), while split-8 over-subscribes to 256 with
#     no gain. So the selector resolves to both ranks agreeing per M, on REPEATED evidence (not r1
#     noise) -- a documented selection per the plan's repeated-evidence allowance.
# Every factor MUST divide the shape's own K-tile count (I // BF16_TILE_K). (I, M, r) not in the table
# (incl. large M) falls back to LONG_K_SPLIT -> unchanged.
_LONG_K_SPLIT_BY_IMR = {
    (24576, 1024, 32): 8,
    (24576, 1024, 64): 8,
    (24576, 2048, 32): 4,
    (24576, 2048, 64): 4,
    # double_ff_out (I=18432, K-tile count 288): split=9 (-> 32 K-tiles/CTA) is the robust full-pipeline
    # pick across M=1024/2048/4096 x r{32,64} -- the argmin at M=2048 and M=4096 (both ranks) and
    # within-MAD of the argmin at M=1024 (where split 16/18 edge it by < 1 MAD but over-subscribe).
    # At M=1024, 16 M-tiles x 9 = 144 CTAs ~= the 148 SMs (the saturation pick). Admitting these rows
    # takes double_ff_out's down off skinny-N onto the m64 occupancy + split-K path, cutting M=1024
    # ~+77% toward the long-K floor (~+12-14%). GB200 r0_dffout_split full-pipeline split sweep.
    (18432, 1024, 32): 9,
    (18432, 1024, 64): 9,
    (18432, 2048, 32): 9,
    (18432, 2048, 64): 9,
    (18432, 4096, 32): 9,
    (18432, 4096, 64): 9,
    # double_ff_out mid-M M=4608: split=2 (72 m64-tiles x 2 = 144 CTAs ~= 148 SMs); split=9
    # over-subscribes here and measures worse (GB200 r0_dffout_4608). +26% -> ~+14% (long-K floor).
    (18432, 4608, 32): 2,
    (18432, 4608, 64): 2,
}


def _long_k_split_for_m(I, M, r):
    """Long-K down split-K factor for contraction I, batch M, and rank r -- (I, M, r)-aware: the I key
    isolates each long-K shape (so single_linear2 and a same-(M,r) long-K sibling never collide), and
    the full-pipeline optimum also differs by rank at M=2048. Falls back to LONG_K_SPLIT when M is None
    or (I, M, r) is not in the table (so large-M and non-shape-aware callers are unchanged). Asserts the
    factor divides the shape's own K-tile count so the per-CTA K-loop (kps = k_tile_cnt // num_k_split)
    covers K exactly (no silently-dropped tiles)."""
    split = LONG_K_SPLIT if M is None else _LONG_K_SPLIT_BY_IMR.get((I, M, r), LONG_K_SPLIT)
    k_tiles = I // BF16_TILE_K
    assert k_tiles % split == 0, (
        f"long-K split factor {split} does not divide the K-tile count {k_tiles} (I={I}); "
        f"choose a divisor of {k_tiles}"
    )
    return split


# Long-K down admission. single_linear2 (I in _LONG_K_I_ALLM) takes the occupancy + split-K path for
# EVERY M; any other shape takes it ONLY at an (I, M, r) that carries an explicit split entry, so
# admitting a new long-K shape/M is a deliberate, isolated table entry -- never a blunt I>=threshold
# spill that would also switch its untuned (e.g. large-M) rows. (M is None -> non-shape-aware caller,
# admitted only via the all-M set.)
_LONG_K_I_ALLM = (24576,)


def _is_long_k(I, M, r):
    return I in _LONG_K_I_ALLM or (M is not None and (I, M, r) in _LONG_K_SPLIT_BY_IMR)


# Occupancy-starved I=6144 down admission, keyed by (I, O, M, r). Several block layers share the same
# I=6144 down (D = X @ L2^T, K=6144) whose few M-tiles starve the 148 SMs at small/mid M; the m64
# occupancy tile + a saturation split (8192/M -> ~128 CTAs at m64) cuts the overhead. The key carries O
# (not just I) so it isolates the TARGET shape from its I=6144 SIBLINGS -- single_linear1 (O=55296,
# the shipped 2-CTA win) and any other O stay byte-identical. GB200 full-pipeline sweeps
# (r0_dattn_split / r0_dffin_split / r0_*_4608):
#   double_attn (square, O=6144): down occupancy-bound; +53/56% -> ~+14-17% at M=1024 (residual is the
#     small-native X-re-read floor).
#   double_ff_in (wide-O-ish, O=36864): its DOWN dominated (not the up-projection); +19.5% -> ~+5-8%
#     (under budget) -- profile-first paid off: the lever is the down, not the 2-CTA epilogue.
# split = 8/4/2 for M=1024/2048/4096 (saturation; the noisy per-(M,r) argmins are within-MAD), 2 at M=4608.
_OCC_DOWN_SPLIT_BY_IOMR = {
    (6144, 6144, 1024, 32): 8, (6144, 6144, 1024, 64): 8,      # double_attn (square)
    (6144, 6144, 2048, 32): 4, (6144, 6144, 2048, 64): 4,
    (6144, 6144, 4096, 32): 2, (6144, 6144, 4096, 64): 2,
    (6144, 6144, 4608, 32): 2, (6144, 6144, 4608, 64): 2,
    (6144, 36864, 1024, 32): 8, (6144, 36864, 1024, 64): 8,    # double_ff_in (wide-O-ish; down-bound)
    (6144, 36864, 2048, 32): 4, (6144, 36864, 2048, 64): 4,
    (6144, 36864, 4096, 32): 2, (6144, 36864, 4096, 64): 2,
}


def _is_occ_down(I, O, M, r):
    return O is not None and M is not None and (I, O, M, r) in _OCC_DOWN_SPLIT_BY_IOMR


def _occ_down_split_for(I, O, M, r):
    split = _OCC_DOWN_SPLIT_BY_IOMR[(I, O, M, r)]
    k_tiles = I // BF16_TILE_K
    assert k_tiles % split == 0, (
        f"occupancy-down split factor {split} does not divide the K-tile count {k_tiles} (I={I}); "
        f"choose a divisor of {k_tiles}"
    )
    return split


# Offline-autotuned down configs (experiments/dkg/autotune_dispatch.py, run.r1_autotune): for these
# residual rows, m128 + split-K BEATS the round-0 m64 occ-down / long-K -- the split provides the
# occupancy the smaller 64-row tile gave, while the 128-row tile is MMA-efficient. Wins (vs the round-0
# config, same-harness 3-rep medians, robust beyond MAD): single_linear1 M1024 +14->+4%, single_linear2
# mid/large-M +13->+8%, double_ff_out +12->+8%, double_attn +15->+10%. The win is the DOWN TILE (best at
# num_ab_stage=3; a deeper KernelB pipeline added only ~0.01-0.43pp, NOT baked -- avoids a new dispatch
# dimension). (I, O, M, r) -> (m_tile, num_k_split); n_tile stays the skinny 64. Takes precedence over the
# long-K / occ-down tables for exactly these measured rows; all other rows are byte-identical.
_AUTOTUNED_DOWN_BY_IOMR = {
    # double_attn (square): M1024 is NOT baked -- BOTH the r3 m128 picks AND the EXACT r4 bests (r32 m64
    # split32 s4, r64 m128 split6 s4) were paired-rejected (r3_dattn_gate + r5_dattn_gate): the target deltas
    # are within the ~200us tiny-shape same-node noise (r5: a byte-identical-dispatch context_embed row
    # "regressed" +1.93% in the SAME run). M1024 keeps round-0's m64 occ-down split8.
    (6144, 6144, 4096, 32): (128, 4),  (6144, 6144, 4096, 64): (128, 4),
    (6144, 6144, 4608, 32): (128, 4),  (6144, 6144, 4608, 64): (128, 4),
    (6144, 6144, 8192, 32): (128, 2),  (6144, 6144, 8192, 64): (128, 2),
    (18432, 6144, 2048, 32): (128, 9), (18432, 6144, 2048, 64): (128, 9),     # double_ff_out (long-K)
    (18432, 6144, 4096, 32): (128, 4), (18432, 6144, 4096, 64): (128, 4),
    (18432, 6144, 4608, 32): (128, 4), (18432, 6144, 4608, 64): (128, 4),
    (18432, 6144, 8192, 32): (128, 2), (18432, 6144, 8192, 64): (128, 2),
    (24576, 6144, 1024, 32): (128, 16), (24576, 6144, 1024, 64): (128, 16),   # single_linear2 small-M: m128+split16/8
    (24576, 6144, 2048, 32): (128, 8),  (24576, 6144, 2048, 64): (128, 8),    #   beats the SHIPPED m64 split-K (~+12 -> ~+8%, round 2)
    (24576, 6144, 4096, 32): (128, 4), (24576, 6144, 4096, 64): (128, 4),     # single_linear2 (long-K mid/large-M)
    (24576, 6144, 4608, 32): (128, 4), (24576, 6144, 4608, 64): (128, 4),
    (24576, 6144, 8192, 32): (128, 2), (24576, 6144, 8192, 64): (128, 2),
    (6144, 55296, 1024, 32): (128, 12), (6144, 55296, 1024, 64): (128, 16),   # single_linear1 (wide-O; r4 m64+s4 pick was pair-rejected TGT-FLAT -> kept round-3 m128)
    # round-13 retune (run.r13_autotune + verify sweep r13_sweep2): these rows round 1 never searched used a
    # poor DEFAULT down-tile (split1); m128 + high split-K wins, sweep-confirmed (BOTH ranks improve ~6pp
    # consistently), 1-CTA/2-CTA state unchanged, grid 120/120. double_attn M2048's autotune "win" did NOT
    # reproduce in the verify sweep (r32 +8.6% but r64 +12.3% > prod) -> tiny-shape (~240us native) cross-run
    # noise, NOT baked (kept at prod like M1024; the square shape is at the X-re-read/noise floor).
    (6144, 55296, 2048, 32): (128, 16), (6144, 55296, 2048, 64): (128, 8),    # single_linear1 M2048: +9.4/+10.9 -> +3.6/+4.6% (2-CTA path)
    (18432, 6144, 1024, 32): (128, 18), (18432, 6144, 1024, 64): (128, 16),   # double_ff_out M1024: +9.8/+11.4 -> +5.2/+8.7%
}


def _is_autotuned_down(I, O, M, r):
    return O is not None and M is not None and (I, O, M, r) in _AUTOTUNED_DOWN_BY_IOMR


def _autotuned_down_for(I, O, M, r):
    m_tile, split = _AUTOTUNED_DOWN_BY_IOMR[(I, O, M, r)]
    k_tiles = I // BF16_TILE_K
    assert m_tile in (64, 128), f"autotuned m_tile {m_tile} must be 64 or 128 (I={I},O={O},M={M},r={r})"
    assert k_tiles % split == 0, (
        f"autotuned split {split} does not divide the K-tile count {k_tiles} (I={I}); choose a divisor"
    )
    return m_tile, split


def down_tile_for_shape(I, r, M=None, O=None):
    """The down-projection MMA (m_tile, n_tile), zero-padded N, and split-K factor for a layer with
    contraction I, output rank r, and optional batch M. D=X@L2^T has only r<=64 output columns over
    a long K=I. Selection -> (m_tile, n_tile, r_pad, num_k_split):
      * r>64                          -> (128, 128, ..., 1): full tile (informational rank; no specialization).
      * r<=64, NOT long-K (_is_long_k) -> (128, 64, ..., 1): skinny-N (avoids the 128-col N waste).
      * r<=64, long-K (_is_long_k)     -> ( 64, 64, ..., split): the long-K path -- a 64 M-tile
        (2x M-tile CTAs) AND split-K. The split factor is M-AWARE (_long_k_split_for_m): small M is
        occupancy-starved streaming the long K, so a larger split saturates the SMs (~128 resident
        CTAs = M_tiles * split). The factor divides the K-tile count; split writes f32 partials
        reduced to bf16 D by SplitKReduce. M=64 is a valid 1-CTA tcgen05 bf16 M-mode; the epilogue
        t2r op adapts via get_tmem_load_op. (M=None -> the default LONG_K_SPLIT, so non-shape-aware
        callers and every M not in the small-M table are unchanged.)
    r is padded UP to a multiple of n_tile -- the padded B/L2 rows produce zero output columns the caller
    slices off.

    O (the output dim) is threaded through so the down-tile selection can be made shape-aware: I and O
    together identify the layer, which isolates a same-I family member (e.g. the square I==O case) from
    its siblings. The current branches do not yet read O -- it is wired for the per-shape down-tile
    specialization that selects on it."""
    if r > SKINNY_N_TILE:
        m_tile, n_tile, num_k_split = MMA_TILER_MN[0], MMA_TILER_MN[1], 1
    elif _is_autotuned_down(I, O, M, r):
        m_tile, num_k_split = _autotuned_down_for(I, O, M, r)   # measured winner (overrides long-K/occ-down)
        n_tile = SKINNY_N_TILE
    elif _is_long_k(I, M, r):
        m_tile, n_tile = LONG_K_M_TILE, SKINNY_N_TILE
        num_k_split = _long_k_split_for_m(I, M, r)
    elif _is_occ_down(I, O, M, r):
        m_tile, n_tile = LONG_K_M_TILE, SKINNY_N_TILE
        num_k_split = _occ_down_split_for(I, O, M, r)
    else:
        m_tile, n_tile, num_k_split = MMA_TILER_MN[0], SKINNY_N_TILE, 1
    r_pad = ((r + n_tile - 1) // n_tile) * n_tile
    return m_tile, n_tile, r_pad, num_k_split


class ActivationQuantizer:
    """REAL on-device NVFP4 activation quantizer (tcgen05-free CuteDSL).

    run_kernel_a's quantizer half: the production front-end runs this quantizer + the down-projection
    as two device kernels (the read-once single-kernel fusion is BLOCKED by an MLIR compile wall --
    see the module header).

    Grid: ONE THREAD PER 16-ELEMENT K-BLOCK -- a flat grid over the M*(I/16) independent blocks
    (thread g -> block (m = g // nblk, b = g % nblk), m-major so a warp reads 32 contiguous blocks
    of one row -> coalesced). Each thread reads its 16 X elements from gmem (f32, K-major logical
    [M, I, 1]), computes the block amax, the E4M3 block scale sf = E4M3_round(amax / (6*gscale)),
    rounds each element x/(sf*gscale) to E2M1, and packs two nibbles/byte. Emits the plain
    [M, I//16] E4M3 sf bytes and the [M, I//2] packed-E2M1 Xq bytes -- the same byte format
    reference.pack_with_gscale produces. No TMA / MMA / pipeline -> no dynamic-control-flow
    variable hoisting; the only loops are static Python ranges over the 16 block elements / 8 pairs.

    (The earlier one-thread-per-ROW grid launched only M threads -- a few thousand -- each serially
    walking a whole I-element row, leaving the GPU ~1% utilized [~5676 us for single_linear2]. The
    per-block grid launches M*(I/16) threads [millions], so the quantizer is bandwidth-bound; the
    per-block math is IDENTICAL, so the emitted Xq/sf bytes are unchanged.)

    The quantizer consumes f32 X (NOT bf16): NVFP4 quant has no MMA that forces a bf16 operand,
    so quantizing the full-precision X is DEQUANT-PARITY with reference.pack_with_gscale(X_f32)
    (a few E4M3/E2M1 round-to-nearest TIES differ between cute's and torch's conversions, so the
    same-device bytewise compare shows tie MISMATCHes -- not bytewise-exact) and the downstream
    NVFP4 main reproduces the fp32 reference's main. Feeding bf16 X instead would impose a
    rank-independent ~38 dB floor (input rounding), which is why both this helper and the fused
    kernel quantize f32.
    """

    def __init__(self):
        self.threads_per_cta = QUANT_M_TILE

    @cute.jit
    def __call__(
        self,
        x_tensor: cute.Tensor,       # X  [M, I, 1]  f32, K-major
        xq_tensor: cute.Tensor,      # Xq [M, I//2]  uint8 packed E2M1 (output)
        sf_tensor: cute.Tensor,      # sf [M, I//16] uint8 E4M3 block scales (output, plain layout)
        gscale: cutlass.Float32,     # provided per-tensor global scale gscale_x
        stream: cuda.CUstream,
    ):
        M = x_tensor.shape[0]
        nblk = x_tensor.shape[1] // SF_BLOCK                 # I // 16 (one thread per 16-block)
        grid = (cute.ceil_div(M * nblk, self.threads_per_cta), 1, 1)
        self.kernel(x_tensor, xq_tensor, sf_tensor, gscale).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,             # X  [M, I, 1]  f32, K-major
        mXq: cute.Tensor,            # Xq [M, I//2]  uint8 (gmem)
        mSF: cute.Tensor,            # sf [M, I//16] uint8 (gmem)
        gscale: cutlass.Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * self.threads_per_cta + tidx   # flat index over the M*(I/16) 16-blocks
        M = mX.shape[0]
        nblk = cute.size(mSF, mode=[1])          # I // 16 (compile-time)
        gm = g // nblk                            # this thread's M-row (nblk compile-time -> static div/mod)
        b = g % nblk                              # this thread's 16-block within the row

        # Guard the tail: flat threads past M*nblk map to rows >= M and do nothing.
        if gm < M:
            k0 = b * SF_BLOCK
            amax = cutlass.Float32(0.0)
            for j in range(SF_BLOCK):
                v = cutlass.Float32(mX[gm, k0 + j, 0])
                amax = cute.arch.fmax(amax, cute.math.absf(v))
            amax = cute.arch.fmax(amax, cutlass.Float32(1e-8))            # ref clamp_min(1e-8)
            sf_e4 = cutlass.Float8E4M3FN(amax / (cutlass.Float32(FP4_E2M1_MAX) * gscale))
            mSF[gm, b] = sf_e4.bitcast(cutlass.Uint8)
            eff = cute.arch.fmax(cutlass.Float32(sf_e4.to(cutlass.Float32)) * gscale,
                                 cutlass.Float32(1e-12))                  # ref clamp_min(1e-12)
            # E2M1-round each element, pack two nibbles/byte (even K low / odd K high).
            frag = cute.make_fragment(SF_BLOCK, cutlass.Float32)
            for j in range(SF_BLOCK):
                frag[j] = cutlass.Float32(mX[gm, k0 + j, 0]) / eff
            q = frag.load().to(cutlass.Float4E2M1FN)                      # f32 -> E2M1 (round-nearest)
            qfrag = cute.make_fragment(SF_BLOCK, cutlass.Float4E2M1FN)
            qfrag.store(q)
            qbytes = cute.recast_tensor(qfrag, cutlass.Uint8)             # 2 nibbles/byte
            for jj in range(SF_BLOCK // 2):
                # canonicalize -0 -> +0 per nibble (pack_with_gscale drops -0): keep both
                # magnitudes (0x77) always; keep each sign bit only when its 3 magnitude
                # bits != 0. Done in Int32 (uint8 shift-to-bit7 is fragile).
                bb = qbytes[jj].to(cutlass.Int32) & cutlass.Int32(0xFF)
                mag_lo = bb & cutlass.Int32(0x7)
                mag_hi = (bb >> cutlass.Int32(4)) & cutlass.Int32(0x7)
                nz_lo = (mag_lo + cutlass.Int32(7)) >> cutlass.Int32(3)   # 1 if mag!=0 else 0
                nz_hi = (mag_hi + cutlass.Int32(7)) >> cutlass.Int32(3)
                mask = cutlass.Int32(0x77) | (nz_lo << cutlass.Int32(3)) | (nz_hi << cutlass.Int32(7))
                mXq[gm, b * (SF_BLOCK // 2) + jj] = (bb & mask).to(cutlass.Uint8)
        return


class DownProjection:
    """The CuteDSL bf16 D = X @ L2^T projection, contracting the full K = I.

    One CTA-group (CtaGroup.ONE), 128 threads. A multi-stage TMA pipeline streams K-tiles
    of the X tile (A operand, M-rows) and the L2 tile (B operand, N=r-rows) into SMEM and a
    single TMEM bf16 accumulator integrates them across the whole K = I. Each X tile is
    read from gmem once (TMA G->S) to drive the projection.

    The MMA tile is configurable (`m_tile`, `n_tile`; selected by down_tile_for_shape). D's output is
    only r<=64 columns, so the SKINNY n_tile=64 computes a 64-col tile instead of padding to 128 (halving
    the tensor-core N work). For the long-K family (I>=24576) the M tile is ALSO halved to 64, doubling
    the M-tile CTA count -- the OCCUPANCY lever for the X-stream-bound long-K down (measured
    occupancy-bound, not N-width-bound). The epilogue TMEM-load copy atom is DERIVED from the tile via
    sm100_utils.get_tmem_load_op (an f32-accumulator / N-major-bf16-D config): M=128 -> Ld32x32b with
    Repetition == n_tile (128->x128, 64->x64); M=64 -> num_dp=16 selects a different op. Deriving it keeps
    every (m_tile, n_tile) correct without hardcoding the op family.
    """

    def __init__(self, num_ab_stage: int = 3, n_tile: int = MMA_TILER_MN[1], m_tile: int = MMA_TILER_MN[0],
                 num_k_split: int = 1):
        self.threads_per_cta = 128
        self.num_tmem_alloc_cols = 512
        self.num_ab_stage = num_ab_stage
        self.num_acc_stage = 1
        self.n_tile = n_tile
        self.m_tile = m_tile
        self.mma_tiler_mn = (m_tile, n_tile)
        # split-K: partition the long K-loop into num_k_split contiguous slices, run concurrently across
        # num_k_split CTAs per M/N tile (grid z = the D output's L dim), each integrating its slice into
        # its own partial D[m,:,s]; a separate reduction sums the partials. num_k_split=1 == the original.
        self.num_k_split = num_k_split

    @cute.jit
    def __call__(
        self,
        x_tensor: cute.Tensor,       # X  [M, I]  bf16, K-major (A operand)
        l2_tensor: cute.Tensor,      # L2 [r, I]  bf16, K-major (B operand; D = X @ L2^T)
        d_tensor: cute.Tensor,       # D  [M, r]  bf16, N-major (output)
        stream: cuda.CUstream,
    ):
        self.x_dtype: Type[cutlass.Numeric] = x_tensor.element_type
        self.l2_dtype: Type[cutlass.Numeric] = l2_tensor.element_type
        self.d_dtype: Type[cutlass.Numeric] = d_tensor.element_type

        self.mma_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], BF16_TILE_K)
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.x_dtype,
            self.l2_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            ACC_DTYPE,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_mn,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        # Multi-stage SMEM layouts for A (X) and B (L2): the staged-K mainloop reuses
        # `num_ab_stage` SMEM buffers as it streams K-tiles of the full K = I.
        self.x_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.x_dtype, self.num_ab_stage
        )
        self.l2_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.l2_dtype, self.num_ab_stage
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA atoms (one-stage slice of the staged SMEM layout describes the per-tile copy).
        x_smem_layout_one = cute.slice_(self.x_smem_layout, (None, None, None, 0))
        tma_atom_x, tma_tensor_x = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            x_tensor, x_smem_layout_one, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        l2_smem_layout_one = cute.slice_(self.l2_smem_layout, (None, None, None, 0))
        tma_atom_l2, tma_tensor_l2 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            l2_tensor, l2_smem_layout_one, self.mma_tiler, tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        x_copy_size = cute.size_in_bytes(self.x_dtype, x_smem_layout_one)
        l2_copy_size = cute.size_in_bytes(self.l2_dtype, l2_smem_layout_one)
        self.num_tma_load_bytes = (x_copy_size + l2_copy_size) * atom_thr_size

        grid = (
            cute.ceil_div(d_tensor.shape[0], self.mma_tiler[0]),
            cute.ceil_div(d_tensor.shape[1], self.mma_tiler[1]),
            d_tensor.shape[2],
        )

        self.kernel(
            tiled_mma,
            tma_atom_x, tma_tensor_x,
            tma_atom_l2, tma_tensor_l2,
            d_tensor,
            self.x_smem_layout, self.l2_smem_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_x: cute.CopyAtom, mX_mkl: cute.Tensor,
        tma_atom_l2: cute.CopyAtom, mL2_nkl: cute.Tensor,
        mD_mnl: cute.Tensor,
        x_smem_layout: cute.ComposedLayout,
        l2_smem_layout: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        mma_tile_coord_mnl = (
            bidx // cute.size(tiled_mma.thr_id.shape),
            bidy,
            bidz,
        )

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sX = smem.allocate_tensor(self.x_dtype, x_smem_layout.outer, 128, x_smem_layout.inner)
        sL2 = smem.allocate_tensor(self.l2_dtype, l2_smem_layout.outer, 128, l2_smem_layout.inner)

        # AB pipeline: producer (TMA) fills a stage, consumer (MMA) drains it; the accumulator
        # pipeline gates the single MMA-result handoff to the epilogue.
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_tma_load_bytes,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta
            ),
        ).make_participants()

        # Tile the global tensors; the K mode stays a free index so we can loop over K-tiles.
        gX = cute.local_tile(mX_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gL2 = cute.local_tile(mL2_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gD = cute.local_tile(mD_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgX = thr_mma.partition_A(gX)
        tCgL2 = thr_mma.partition_B(gL2)
        tCgD = thr_mma.partition_C(gD)

        tXsX, tXgX = cpasync.tma_partition(
            tma_atom_x, 0, cute.make_layout(1),
            cute.group_modes(sX, 0, 3), cute.group_modes(tCgX, 0, 3),
        )
        tL2sL2, tL2gL2 = cpasync.tma_partition(
            tma_atom_l2, 0, cute.make_layout(1),
            cute.group_modes(sL2, 0, 3), cute.group_modes(tCgL2, 0, 3),
        )

        tCrX = tiled_mma.make_fragment_A(sX)
        tCrL2 = tiled_mma.make_fragment_B(sL2)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # TMEM accumulator.
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)

        # Slice the TMA source down to this CTA's M / N tile. X/L2 have L=1 (their L-index is always 0);
        # the output D's L-dim holds the split-K partials (L = num_k_split), so D is sliced by bidz (the
        # split index) -- each split's CTA writes its own partial D[m,:,bidz]. For num_k_split=1, bidz=0,
        # so this is the original single-partition down.
        tXgX = tXgX[(None, mma_tile_coord_mnl[0], None, 0)]
        tL2gL2 = tL2gL2[(None, mma_tile_coord_mnl[1], None, 0)]

        # K-tile count from the static layout mode (compile-time int even though the tensor's
        # runtime M/K extents are marked dynamic) -> a plain Python loop, so `k_tile != 0` below
        # is a compile-time predicate (mirrors the vendored GEMM mainloop). For split-K, each CTA runs
        # only its 1/num_k_split contiguous slice (kps tiles, compile-time) -> a shorter dependent K-loop
        # run concurrently across the num_k_split CTAs; k_base is this split's first global K-tile.
        k_tile_cnt = cute.size(gX, mode=[3])
        kps = k_tile_cnt // self.num_k_split
        k_base = mma_tile_coord_mnl[2] * kps

        #
        # Mainloop over K-tiles: warp 0 produces (TMA load) and consumes (MMA-accumulate) each
        # K-tile through the staged AB pipeline, integrating the full K = I into one accumulator.
        #
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()
            for k_tile in range(kps):
                ab_empty = ab_producer.acquire_and_advance()
                idx = ab_empty.index
                # Global K-tile for this split. The unsplit path keeps a STATIC index (k_tile); split-K
                # offsets by this split's k_base (a runtime split index) -- a dynamic TMA coordinate.
                if cutlass.const_expr(self.num_k_split == 1):
                    kg = k_tile
                else:
                    kg = k_base + k_tile
                cute.copy(tma_atom_x, tXgX[(None, kg)], tXsX[(None, idx)],
                          tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_l2, tL2gL2[(None, kg)], tL2sL2[(None, idx)],
                          tma_bar_ptr=ab_empty.barrier)

                ab_full = ab_consumer.wait_and_advance()
                stage = ab_full.index
                # First K-tile of THIS split initializes its partial accumulator; later K-tiles accumulate.
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    tCrX[(None, None, None, stage)],
                    tCrL2[(None, None, None, stage)],
                    tCtAcc,
                )
                ab_full.release()
            acc_empty.commit()

        #
        # Epilogue: t2r-copy the bf16 accumulator and store D [M, r].
        #
        # The TMEM-load t2r copy atom is DERIVED from the (m_tile, n_tile) tile + dtypes via
        # get_tmem_load_op (D is N-major => ROW_MAJOR C). M=128 -> Ld32x32b(Repetition==n_tile, so
        # 128->x128 / 64->x64); M=64 -> num_dp=16 selects a different op. Deriving it (vs hardcoding the
        # op family) keeps every (m_tile, n_tile) from down_tile_for_shape correct.
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.mma_tiler, utils.LayoutEnum.ROW_MAJOR, self.d_dtype, ACC_DTYPE,
            cute.make_layout(self.mma_tiler_mn), False,
        )
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gD = thr_copy_t2r.partition_D(tCgD)

        rshape = tTR_gD[None, None, None, None, 0, 0, 0].shape
        tTR_rAcc = cute.make_rmem_tensor(rshape, ACC_DTYPE)
        tTR_rD = cute.make_rmem_tensor(rshape, self.d_dtype)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.d_dtype)
        tTR_gD = tTR_gD[(None, None, None, None, *mma_tile_coord_mnl)]

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        acc_full = acc_consumer.wait_and_advance()

        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        tTR_rD.store(tTR_rAcc.load().to(self.d_dtype))
        cute.copy(simt_atom, tTR_rD, tTR_gD)

        acc_full.release()
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)
        return


class SplitKReduce:
    """Sum the split-K f32 partial down output D_partial[M, r, S] over the S splits -> bf16 D[M, r].

    tcgen05-free CuteDSL. ONE THREAD PER (m, n) OUTPUT ELEMENT: a flat grid over the M*r_pad
    independent outputs (thread g -> m = g // r_pad, n = g % r_pad, m-major), each summing its S f32
    partials in FIXED ORDER and storing one bf16.

    (The earlier grid was one thread per M-ROW -- only ceil(M/128) CTAs, e.g. 8 at M=1024 -- each
    thread serially walking all r_pad*S elements, so the reduce was THREAD-STARVED and its latency
    GREW with the split factor [the small-batch single-CTA-ish split-K-reduce slowdown]. The
    per-element grid launches M*r_pad threads [~64x more CTAs], so the reduce is occupancy/bandwidth-
    bound and stays cheap as S rises -- which is what lets the long-K down use a larger split-K. The
    per-thread, fixed-order f32 accumulation [+0.0 init, same S order] is UNCHANGED, so the bf16 D is
    BIT-IDENTICAL to the per-row reduce: zero-LoRA [D=0] stays bit-exact and the nonzero-LoRA SQNR is
    unchanged. The padded n in [r, r_pad) sum to the down's zero partials and are sliced off by the
    caller, exactly as before.) The partials are f32 (accurate cross-split sum); the output is bf16,
    the same D the down used to emit, so the downstream up-projection / fused main consume it UNCHANGED."""

    def __init__(self):
        self.threads_per_cta = 128

    @cute.jit
    def __call__(
        self,
        partial_tensor: cute.Tensor,     # D_partial [M, r, S]  f32 (input)
        out_tensor: cute.Tensor,         # D         [M, r, 1]  bf16 (output)
        stream: cuda.CUstream,
    ):
        self.out_dtype: Type[cutlass.Numeric] = out_tensor.element_type
        M = partial_tensor.shape[0]
        out_cols = out_tensor.shape[1]            # produce EXACTLY the output's columns (<= the partials' r_pad)
        total = M * out_cols                      # one thread per (m, n) output element
        grid = (cute.ceil_div(total, self.threads_per_cta), 1, 1)
        self.kernel(partial_tensor, out_tensor).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1], cluster=(1, 1, 1), stream=stream,
        )
        return

    @cute.kernel
    def kernel(self, mP: cute.Tensor, mO: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        g = bidx * self.threads_per_cta + tidx
        M = mP.shape[0]
        ncols = cute.size(mO, mode=[1])          # OUTPUT columns (reads the first ncols of the partials' r_pad)
        nsplit = cute.size(mP, mode=[2])         # num_k_split (compile-time)
        total = M * ncols
        if g < total:
            m = g // ncols
            n = g % ncols
            acc = cutlass.Float32(0.0)
            for s in range(nsplit):
                acc = acc + cutlass.Float32(mP[m, n, s])
            mO[m, n, 0] = acc.to(self.out_dtype)
        return


# ---------------------------------------------------------------------------
# Host helpers: front-end input builders.
# ---------------------------------------------------------------------------

# Backing torch tensors for the production cute-tensor wrappers must outlive the cute.Tensor
# (DLPack does not own the storage). We pin them on these wrappers so they are not GC'd while a
# compiled kernel still references the DLPack view.
def _kmajor_cuda_tensor(values_2d, cutlass_dtype, torch_dtype):
    """(mn, k, 1) K-major cute tensor whose storage IS the CUDA input -- NO CPU staging.

    The production front-end inputs are already on CUDA; we build a contiguous [mn, k, 1] K-major
    CUDA view (k fastest) and wrap it via DLPack directly, marking the SAME dynamic / compact
    layout metadata `cutlass_torch.matrix(1, mn, k, False, dtype)` would (leading_dim=1, mode-1
    compact, stride_order (2,0,1)). For 16/32-bit operands the bytes ARE the values, so no f32
    convert is needed; the backing tensor is pinned on the cute tensor to keep it alive."""
    assert values_2d.is_cuda, "production front-end input builder requires a CUDA tensor"
    # [mn, k] contiguous on CUDA -> [mn, k, 1] (k stride 1). .contiguous() is a device->device
    # copy (or no-op); it never routes through host. The original input is untouched.
    backing = values_2d.to(dtype=torch_dtype).contiguous().unsqueeze(-1)  # [mn, k, 1], k-major
    cute_t = cute.runtime.from_dlpack(backing, assumed_align=16)
    cute_t.element_type = cutlass_dtype
    cute_t = cute_t.mark_layout_dynamic(leading_dim=1)
    cute_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
    cute_t._svdquant_backing = backing            # keep storage alive with the cute tensor
    return cute_t


def _bf16_kmajor_tensor(values_2d):
    """(mn, k, 1) K-major bf16 cute tensor backed directly by CUDA storage (NO `.cpu()` staging).
    Same K-major / dynamic-layout metadata as the dual-MMA spike's bf16 operand builder."""
    return _kmajor_cuda_tensor(values_2d, LORA_DTYPE, torch.bfloat16)


def _f32_kmajor_tensor(values_2d):
    """(mn, k, 1) K-major f32 cute tensor backed directly by CUDA storage (NO `.cpu()` staging).
    The quantizer's A-input -- full-precision f32, so the on-device quant reproduces
    reference.pack_with_gscale(X_f32) (no bf16 input rounding)."""
    return _kmajor_cuda_tensor(values_2d, cutlass.Float32, torch.float32)


def _u8_rowmajor_tensor(M, cols):
    """A CUDA uint8 [M, cols] row-major output cute tensor (+ its torch view)."""
    t = torch.zeros(M, cols, dtype=torch.uint8, device="cuda")
    cute_t = cute.runtime.from_dlpack(t, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    return cute_t, t


def assert_no_cpu_staging_in_input_builders():
    """Guard: the PRODUCTION front-end input builders must NOT route CUDA
    inputs through host. We monkeypatch torch.Tensor.cpu to raise if invoked on a CUDA tensor
    while the builders run, then build a representative [M, I, 1] / [r, I, 1] CUDA operand each.
    Raises AssertionError if any builder stages through `.cpu()`; returns silently on success."""
    real_cpu = torch.Tensor.cpu
    tripped = {"hit": False}

    def _guarded_cpu(self, *a, **k):
        if self.is_cuda:
            tripped["hit"] = True
            raise AssertionError("production front-end input builder staged a CUDA tensor via .cpu()")
        return real_cpu(self, *a, **k)

    M, I, r = 256, 256, 64
    Xf = torch.randn(M, I, device="cuda", dtype=torch.float32)
    Xb = Xf.to(torch.bfloat16)
    L2b = torch.randn(r, I, device="cuda", dtype=torch.bfloat16)
    torch.Tensor.cpu = _guarded_cpu
    try:
        _f32_kmajor_tensor(Xf)          # quantizer A-input (f32)
        _bf16_kmajor_tensor(Xb)         # down-projection A-input (bf16)
        _bf16_kmajor_tensor(L2b)        # down-projection B-input (bf16)
    finally:
        torch.Tensor.cpu = real_cpu
    assert not tripped["hit"], "production front-end input builders must not CPU-stage CUDA inputs"
    print("[guard] production front-end input builders: no .cpu() staging of CUDA inputs -> PASS")


# Compile caches keyed by shape so JIT time is excluded from reuse.
_QUANT_CACHE = {}
_DOWN_CACHE = {}


def _compiled_quantizer(M, I):
    key = (M, I, "f32")
    if key not in _QUANT_CACHE:
        x_t = _f32_kmajor_tensor(torch.zeros(M, I, dtype=torch.float32, device="cuda"))
        xq_t, _ = _u8_rowmajor_tensor(M, I // 2)
        sf_t, _ = _u8_rowmajor_tensor(M, I // SF_BLOCK)
        quant = ActivationQuantizer()
        stream = cutlass_torch.default_stream()
        _QUANT_CACHE[key] = cute.compile(quant, x_t, xq_t, sf_t, cutlass.Float32(0.0), stream)
    return _QUANT_CACHE[key]


def _compiled_down(M, I, r_pad, n_tile=MMA_TILER_MN[1], m_tile=MMA_TILER_MN[0], num_ab_stage=3):
    key = (M, I, r_pad, n_tile, m_tile, str(LORA_DTYPE), BF16_TILE_K, num_ab_stage)
    if key not in _DOWN_CACHE:
        x_t = _bf16_kmajor_tensor(torch.zeros(M, I, dtype=torch.bfloat16, device="cuda"))
        l2_t = _bf16_kmajor_tensor(torch.zeros(r_pad, I, dtype=torch.bfloat16, device="cuda"))
        d_ref = cutlass_torch.matrix(1, M, r_pad, False, LORA_DTYPE)
        d_t, _ = cutlass_torch.cute_tensor_like(d_ref, LORA_DTYPE, True, 16)
        d_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
        proj = DownProjection(num_ab_stage, n_tile, m_tile)
        stream = cutlass_torch.default_stream()
        _DOWN_CACHE[key] = cute.compile(proj, x_t, l2_t, d_t, stream)
    return _DOWN_CACHE[key]


_REDUCE_CACHE = {}


def _compiled_down_split(M, I, r_pad, n_tile, m_tile, num_k_split, num_ab_stage=3):
    """The split-K down: writes f32 [M, r_pad, num_k_split] partials (each split integrates its 1/S K-slice)."""
    key = (M, I, r_pad, n_tile, m_tile, num_k_split, "splitK", num_ab_stage)
    if key not in _DOWN_CACHE:
        x_t = _bf16_kmajor_tensor(torch.zeros(M, I, dtype=torch.bfloat16, device="cuda"))
        l2_t = _bf16_kmajor_tensor(torch.zeros(r_pad, I, dtype=torch.bfloat16, device="cuda"))
        d_ref = cutlass_torch.matrix(num_k_split, M, r_pad, False, ACC_DTYPE)   # f32 [M, r_pad, S] partials
        d_t, _ = cutlass_torch.cute_tensor_like(d_ref, ACC_DTYPE, True, 16)
        d_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
        proj = DownProjection(num_ab_stage, n_tile, m_tile, num_k_split)
        stream = cutlass_torch.default_stream()
        _DOWN_CACHE[key] = cute.compile(proj, x_t, l2_t, d_t, stream)
    return _DOWN_CACHE[key]


def _compiled_reduce(M, r_pad, num_k_split, r_out=None):
    """SplitKReduce: f32 [M, r_pad, num_k_split] partials -> bf16 [M, r_out, 1] D (sum over the splits).
    r_out defaults to r_pad; pass r_out=r to write EXACTLY the r real columns (the padded columns
    [r, r_pad) are zero and are skipped) so the reduce can write straight into the fused main's D operand
    [M, r] with no separate buffer / handoff copy. The output uses the SAME _kmajor layout as the fused
    main's D (_kmajor_cuda_tensor), so the compiled reduce writes that exact layout. r_out <= r_pad."""
    r_out = r_pad if r_out is None else r_out
    key = (M, r_pad, num_k_split, r_out)
    if key not in _REDUCE_CACHE:
        p_ref = cutlass_torch.matrix(num_k_split, M, r_pad, False, ACC_DTYPE)
        p_t, _ = cutlass_torch.cute_tensor_like(p_ref, ACC_DTYPE, True, 16)
        p_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
        o_t = _kmajor_cuda_tensor(torch.zeros(M, r_out, dtype=torch.bfloat16, device="cuda"),
                                  LORA_DTYPE, torch.bfloat16)
        red = SplitKReduce()
        stream = cutlass_torch.default_stream()
        _REDUCE_CACHE[key] = cute.compile(red, p_t, o_t, stream)
    return _REDUCE_CACHE[key]


def run_quantizer(X, gscale_x):
    """run_kernel_a's quantizer half (one of the production front-end's two device kernels):
    emit (Xq uint8 [M, I//2], sf_x_bytes uint8 [M, I//16]) on CUDA, computed in-kernel (NOT via
    reference.pack_with_gscale). X is taken on its own device (already-CUDA inputs are not CPU-
    staged); the kernel quantizes the full-precision f32 X, so the emitted bytes are dequant-parity
    with reference.pack_with_gscale(X_f32) (a few E4M3/E2M1 round-to-nearest TIES differ between
    cute's and torch's conversions -- see check_quant_parity)."""
    M, I = X.shape
    Xf = X.to(device="cuda", dtype=torch.float32)
    x_tensor = _f32_kmajor_tensor(Xf)                # [M, I, 1] K-major f32
    xq_tensor, xq_torch = _u8_rowmajor_tensor(M, I // 2)
    sf_tensor, sf_torch = _u8_rowmajor_tensor(M, I // SF_BLOCK)
    compiled = _compiled_quantizer(M, I)
    stream = cutlass_torch.default_stream()
    compiled(x_tensor, xq_tensor, sf_tensor, cutlass.Float32(gscale_x), stream)
    torch.cuda.synchronize()
    return xq_torch, sf_torch                        # Xq uint8, sf uint8 (CUDA)


def run_down_projection(X, L2, O=None):
    """run_kernel_a's down-projection half (one of the production front-end's two device kernels):
    run the CuteDSL bf16 D = X @ L2^T kernel and return D [M, r] bf16 on CUDA.

    The MMA tile is shape-aware (down_tile_for_shape): r<=64 uses the SKINNY 64-col N-tile; the long-K
    family (I>=24576) ALSO halves the M-tile to 64 AND uses SPLIT-K (num_k_split>1): the split-K down
    writes f32 [M, r_pad, S] partials which SplitKReduce sums to bf16 D. L2 is zero-padded to r_pad rows
    (a multiple of n_tile): the padded rows contribute zero output columns, which we slice off."""
    M, I = X.shape
    r = L2.shape[0]
    m_tile, n_tile, r_pad, num_k_split = down_tile_for_shape(I, r, M, O=O)
    Xb = X.to(device="cuda", dtype=torch.bfloat16)
    L2b = L2.to(device="cuda", dtype=torch.bfloat16)
    L2_pad = L2b
    if r_pad != r:
        L2_pad = torch.zeros(r_pad, I, dtype=torch.bfloat16, device="cuda")
        L2_pad[:r] = L2b

    x_tensor = _bf16_kmajor_tensor(Xb)               # [M, I]     K-major bf16 (A operand)
    l2_tensor = _bf16_kmajor_tensor(L2_pad)          # [r_pad, I] K-major bf16 (B operand, N = r_pad)
    stream = cutlass_torch.default_stream()

    if num_k_split == 1:
        d_ref = cutlass_torch.matrix(1, M, r_pad, False, LORA_DTYPE)
        d_tensor, d_torch = cutlass_torch.cute_tensor_like(d_ref, LORA_DTYPE, True, 16)
        d_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
        _compiled_down(M, I, r_pad, n_tile, m_tile)(x_tensor, l2_tensor, d_tensor, stream)
        torch.cuda.synchronize()
        return d_torch[:, :r, 0]                      # D bf16 (CUDA)

    # split-K: f32 [M, r_pad, S] partials -> SplitKReduce -> bf16 [M, r] D, writing the r REAL columns
    # directly into a K-major D (the padded columns [r, r_pad) are zero and are skipped); no slice/copy.
    p_ref = cutlass_torch.matrix(num_k_split, M, r_pad, False, ACC_DTYPE)
    p_tensor, _ = cutlass_torch.cute_tensor_like(p_ref, ACC_DTYPE, True, 16)
    p_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
    o_tensor = _kmajor_cuda_tensor(torch.zeros(M, r, dtype=torch.bfloat16, device="cuda"),
                                   LORA_DTYPE, torch.bfloat16)
    _compiled_down_split(M, I, r_pad, n_tile, m_tile, num_k_split)(x_tensor, l2_tensor, p_tensor, stream)
    _compiled_reduce(M, r_pad, num_k_split, r)(p_tensor, o_tensor, stream)
    torch.cuda.synchronize()
    return o_tensor._svdquant_backing[:, :, 0]        # D bf16 [M, r] (CUDA)


def run_kernel_a(X, L2, gscale_x, O=None):
    """PRODUCTION front-end (CURRENT: two device kernels -- ActivationQuantizer +
    DownProjection). Emits REAL on-device Xq/sf_x AND bf16 D = X @ L2^T; both reads are device-side
    (already-CUDA inputs are NOT CPU-staged). Returns (Xq uint8 [M, I//2], sf_x_bytes uint8
    [M, I//16], D bf16 [M, r] on CUDA). Xq/sf_x are computed in-kernel (NOT pack_with_gscale).

    The SINGLE-kernel read-once fusion (one f32 X read driving both quant and D) is BLOCKED by an MLIR
    compile-time wall that a compile-cheap quant body does NOT fix: the deadlock is solved (warp-0 sole
    consumer), but a fully NON-UNROLLED quant body still walls cute.compile() even on a small shape
    (GB200 py.ro_compile2) -- the blowup is from co-locating the quant body with the tcgen05 UMMA
    module at all, not from unrolling. A read-once fusion needs a fundamentally different emission (not
    a loop tweak) -- see the module header. The two-kernel path feeds the fused main identically
    (same Xq/sf_x/D), so this does not affect the fused main."""
    Xq, sf_x_bytes = run_quantizer(X, gscale_x)      # ActivationQuantizer (REAL on-device quant)
    D = run_down_projection(X, L2, O=O)              # DownProjection (bf16 tcgen05 MMA)
    return Xq, sf_x_bytes, D


def _provided_gscale_x(X):
    # Clamp amax to the reference/bridge floor (reference uses amax.clamp_min(1e-8)) so an all-zero X yields a
    # finite gscale instead of 0 -- a 0 here would divide to inf/NaN E4M3 block scales in the quantizer. No-op
    # for any non-zero activation.
    return (X.abs().amax().clamp_min(1e-8) / (FP4_E2M1_MAX * E4M3_MAX)).item()


def _flux_shapes():
    """All six FLUX.2 shapes from bench/problem_sizes.yaml as (name, M, I, O)."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(_HERE), "bench", "problem_sizes.yaml")))
    return [(s["name"], s["M"], s["I"], s["O"]) for s in cfg["shapes"]]


class KernelB:
    """One kernel, two heterogeneous TMEM accumulators (NVFP4 main + bf16 low-rank), fused add,
    over the full K = I via a multi-stage pipelined main loop. `enable_lora=False` compiles the
    plain-NVFP4 variant (same main schedule, lora path disabled)."""

    def __init__(self, lora_k: int, enable_lora: bool = True, num_ab_stage: int = 3,
                 num_tmem_alloc_cols: int = 512, num_acc_stage: int = 1):
        self.threads_per_cta = 128
        self.lora_k = lora_k                 # r (low-rank contraction depth, <= 64)
        self.enable_lora = bool(enable_lora)
        # num_tmem_alloc_cols: TMEM columns reserved per CTA (full TMEM = 512 -> 1 CTA/SM). Reducing it
        # (when the accumulators fit) lets >1 CTA reside per SM so one CTA's epilogue overlaps another's
        # MMA -- the occupancy lever for this latency-bound kernel (ncu: 6.2% occ, 27% compute).
        self.num_tmem_alloc_cols = num_tmem_alloc_cols
        self.num_ab_stage = num_ab_stage     # main NVFP4 pipeline depth (large K = I)
        self.num_lora_stage = 1              # lora K = r fits one tile
        self.num_acc_stage = num_acc_stage

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor,       # Xq  [M, I]  packed E2M1, K-major
        b_tensor: cute.Tensor,       # Rq  [O, I]  packed E2M1, K-major
        sfa_tensor: cute.Tensor,     # SFA E4M3 (MMA-swizzled iterator)
        sfb_tensor: cute.Tensor,     # SFB E4M3 (MMA-swizzled iterator)
        d_tensor: cute.Tensor,       # D   [M, r]  bf16, K-major (= X @ L2^T)
        l1_tensor: cute.Tensor,      # L1  [O, r]  bf16, K-major
        c_tensor: cute.Tensor,       # Y   [M, O]  bf16, N-major
        global_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a_tensor.element_type
        self.b_dtype: Type[cutlass.Numeric] = b_tensor.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa_tensor.element_type
        self.lora_dtype: Type[cutlass.Numeric] = d_tensor.element_type
        self.c_dtype: Type[cutlass.Numeric] = c_tensor.element_type

        # --- main (NVFP4 block-scaled) tile + tiled_mma ---
        self.mma_tiler = (MMA_TILER_MN[0], MMA_TILER_MN[1], NVFP4_TILE_K)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype,
            OperandMajorMode.K, OperandMajorMode.K,
            self.sf_dtype, SF_VEC, tcgen05.CtaGroup.ONE, MMA_TILER_MN,
        )
        # --- low-rank (bf16) tile + tiled_mma: SAME (M, N) tile, contract K = r ---
        self.lora_tiler = (MMA_TILER_MN[0], MMA_TILER_MN[1], self.lora_k)
        lora_mma = sm100_utils.make_trivial_tiled_mma(
            self.lora_dtype, self.lora_dtype,
            OperandMajorMode.K, OperandMajorMode.K,
            ACC_DTYPE, tcgen05.CtaGroup.ONE, MMA_TILER_MN,
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )

        # Re-wrap the SF tensors with the layout derived from the FULL A/B shape (iterator swizzled).
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, SF_VEC)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, SF_VEC)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        # Multi-stage SMEM layouts -- main operands + scales (the staged-K mainloop reuses
        # num_ab_stage buffers as it streams the K = I tiles).
        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage)
        self.sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage)
        # Single-stage SMEM layouts -- bf16 low-rank operands (K = r is one tile).
        self.d_smem_layout = sm100_utils.make_smem_layout_a(
            lora_mma, self.lora_tiler, self.lora_dtype, self.num_lora_stage)
        self.l1_smem_layout = sm100_utils.make_smem_layout_b(
            lora_mma, self.lora_tiler, self.lora_dtype, self.num_lora_stage)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # TMA atoms -- main NVFP4 path (per-tile slice of the staged SMEM describes the copy).
        a_smem_layout = cute.slice_(self.a_smem_layout, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            a_tensor, a_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            b_tensor, b_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
        )
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfa_tensor, sfa_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            sfb_tensor, sfb_smem_layout, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        # TMA atoms -- bf16 low-rank path
        d_smem_layout = cute.slice_(self.d_smem_layout, (None, None, None, 0))
        tma_atom_d, tma_tensor_d = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            d_tensor, d_smem_layout, self.lora_tiler, lora_mma, self.cluster_layout_vmnk.shape,
        )
        l1_smem_layout = cute.slice_(self.l1_smem_layout, (None, None, None, 0))
        tma_atom_l1, tma_tensor_l1 = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            l1_tensor, l1_smem_layout, self.lora_tiler, lora_mma, self.cluster_layout_vmnk.shape,
        )

        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        d_copy_size = cute.size_in_bytes(self.lora_dtype, d_smem_layout)
        l1_copy_size = cute.size_in_bytes(self.lora_dtype, l1_smem_layout)
        # Per-stage TX byte counts: the main AB pipeline carries A/B/SFA/SFB each k_tile; the lora
        # pipeline carries D/L1 once.
        self.num_ab_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size) * atom_thr_size
        self.num_lora_tma_load_bytes = (d_copy_size + l1_copy_size) * atom_thr_size

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.mma_tiler[0]),
            cute.ceil_div(c_tensor.shape[1], self.mma_tiler[1]),
            c_tensor.shape[2],
        )

        self.kernel(
            tiled_mma, lora_mma,
            tma_atom_a, tma_tensor_a,
            tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa,
            tma_atom_sfb, tma_tensor_sfb,
            tma_atom_d, tma_tensor_d,
            tma_atom_l1, tma_tensor_l1,
            c_tensor, global_scale,
            self.a_smem_layout, self.b_smem_layout,
            self.sfa_smem_layout, self.sfb_smem_layout,
            self.d_smem_layout, self.l1_smem_layout,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        lora_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom, mD_mkl: cute.Tensor,
        tma_atom_l1: cute.CopyAtom, mL1_nkl: cute.Tensor,
        mC_mnl: cute.Tensor,
        global_scale: cutlass.Float32,
        a_smem_layout: cute.ComposedLayout,
        b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: cute.Layout,
        sfb_smem_layout: cute.Layout,
        d_smem_layout: cute.ComposedLayout,
        l1_smem_layout: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        mma_tile_coord_mnl = (
            bidx // cute.size(tiled_mma.thr_id.shape),
            bidy,
            bidz,
        )

        #
        # Shared storage: pipeline barriers (main AB + lora AB + acc) + the two operand sets.
        #
        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            lora_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_lora_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # main NVFP4 operands (multi-stage)
        sA = smem.allocate_tensor(self.a_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
        sB = smem.allocate_tensor(self.b_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
        sSFA = smem.allocate_tensor(self.sf_dtype, sfa_smem_layout, 128)
        sSFB = smem.allocate_tensor(self.sf_dtype, sfb_smem_layout, 128)
        # bf16 low-rank operands (single stage)
        sD = smem.allocate_tensor(self.lora_dtype, d_smem_layout.outer, 128, d_smem_layout.inner)
        sL1 = smem.allocate_tensor(self.lora_dtype, l1_smem_layout.outer, 128, l1_smem_layout.inner)

        #
        # Pipelines: a multi-stage main AB pair (A/B/SFA/SFB), a single-stage lora AB pair (D/L1),
        # and one acc pair gating the MMA-result handoff to the epilogue. Warp 0 is the SOLE
        # consumer of both AB pipelines (avoids the single-consumer-PipelineTmaUmma over-rotation).
        #
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=ab_consumer_group,
            tx_count=self.num_ab_tma_load_bytes,
        ).make_participants()
        lora_producer, lora_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.lora_mbar_ptr.data_ptr(),
            num_stages=self.num_lora_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_lora_tma_load_bytes,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_cta),
        ).make_participants()

        #
        # Tile global tensors; the K mode stays a free index so the mainloop can stream K-tiles.
        #
        gA = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gD = cute.local_tile(mD_mkl, cute.slice_(self.lora_tiler, (None, 0, None)), (None, None, None))
        gL1 = cute.local_tile(mL1_nkl, cute.slice_(self.lora_tiler, (0, None, None)), (None, None, None))
        gC = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))

        #
        # MMA partitions.
        #
        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        tCgSFB = thr_mma.partition_B(gSFB)
        tCgC = thr_mma.partition_C(gC)

        thr_lora = lora_mma.get_slice(mma_tile_coord_v)
        tLgD = thr_lora.partition_A(gD)
        tLgL1 = thr_lora.partition_B(gL1)

        #
        # TMA partitions.
        #
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1),
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1),
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa, 0, cute.make_layout(1),
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, 0, cute.make_layout(1),
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tDsD, tDgD = cpasync.tma_partition(
            tma_atom_d, 0, cute.make_layout(1),
            cute.group_modes(sD, 0, 3), cute.group_modes(tLgD, 0, 3))
        tL1sL1, tL1gL1 = cpasync.tma_partition(
            tma_atom_l1, 0, cute.make_layout(1),
            cute.group_modes(sL1, 0, 3), cute.group_modes(tLgL1, 0, 3))

        #
        # MMA fragments.
        #
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tLrD = lora_mma.make_fragment_A(sD)
        tLrL1 = lora_mma.make_fragment_B(sL1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        lora_acc_shape = lora_mma.partition_shape_C(self.mma_tiler[:2])
        tLtAcc_fake = lora_mma.make_fragment_C(lora_acc_shape)

        #
        # Allocate TMEM: main_acc at base, lora_acc next, then SFA/SFB for the NVFP4 MMA
        # (column-offset chaining; both variants allocate identically so main_acc is bit-stable).
        #
        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        lora_acc_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc), dtype=ACC_DTYPE)
        tLtAcc = cute.make_tensor(lora_acc_ptr, tLtAcc_fake.layout)

        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_smem_layout, (None, None, None, 0)))
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tLtAcc),
            dtype=self.sf_dtype)
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_smem_layout, (None, None, None, 0)))
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr
            + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tLtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA),
            dtype=self.sf_dtype)
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        #
        # S2T copy plumbing for SFA/SFB (NVFP4 block scales SMEM -> TMEM); staged source.
        #
        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), self.sf_dtype)
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, thr_copy_s2t_sfa.partition_S(tCsSFA_compact))
        tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)

        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, thr_copy_s2t_sfb.partition_S(tCsSFB_compact))
        tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        #
        # Slice TMA source/dest to this CTA tile; the main A/B/SFA/SFB keep a free K-tile mode.
        #
        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tDgD = tDgD[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tL1gL1 = tL1gL1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        # K-tile count from the static layout mode (compile-time int even with dynamic M/K), so the
        # plain Python loop keeps `tiled_mma.set(ACCUMULATE, k_tile != 0)` a compile-time predicate.
        k_tile_cnt = cute.size(gA, mode=[3])

        #
        # Compute. Warp 0 drives both AB pipelines and issues both MMAs.
        #
        if warp_idx == 0:
            acc_empty = acc_producer.acquire_and_advance()

            # OVERLAP SCHEDULE (wide-O lever): issue the low-rank D/L1 TMA load BEFORE the main K-loop so
            # it streams in flight DURING the main NVFP4 GEMM (the two use independent pipelines), then
            # consume it AFTER the loop. The bf16 lora MMA itself is tiny; this hides its TMA-load
            # latency behind the long main K-loop. The main K-loop accumulation order is UNCHANGED, and
            # the plain variant (enable_lora=False) issues NONE of the lora ops -- so its main schedule
            # is byte-identical and the zero-LoRA bit-match (lora_acc=+0 with D=0) still holds.
            if cutlass.const_expr(self.enable_lora):
                lora_empty = lora_producer.acquire_and_advance()
                lidx = lora_empty.index
                cute.copy(tma_atom_d, tDgD[(None, 0)], tDsD[(None, lidx)], tma_bar_ptr=lora_empty.barrier)
                cute.copy(tma_atom_l1, tL1gL1[(None, 0)], tL1sL1[(None, lidx)], tma_bar_ptr=lora_empty.barrier)

            # main NVFP4 mainloop: stream the full K = I, accumulating into main_acc.
            for k_tile in range(k_tile_cnt):
                ab_empty = ab_producer.acquire_and_advance()
                idx = ab_empty.index
                cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, idx)], tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, idx)], tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, idx)], tma_bar_ptr=ab_empty.barrier)
                cute.copy(tma_atom_sfb, tBgSFB[(None, k_tile)], tBsSFB[(None, idx)], tma_bar_ptr=ab_empty.barrier)

                ab_full = ab_consumer.wait_and_advance()
                stage = ab_full.index
                cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, None, stage)], tCtSFA_s2t)
                cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, stage)], tCtSFB_s2t)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                cute.gemm(
                    tiled_mma, tCtAcc,
                    [tCrA[(None, None, None, stage)], tCtSFA],
                    [tCrB[(None, None, None, stage)], tCtSFB],
                    tCtAcc,
                )
                ab_full.release()

            # The low-rank D/L1 finished loading during the main loop -> consume it now: D @ L1^T ->
            # lora_acc (bf16, K = r one tile). One acc commit then gates BOTH accumulators to the epilogue.
            if cutlass.const_expr(self.enable_lora):
                lora_full = lora_consumer.wait_and_advance()
                lstage = lora_full.index
                lora_mma.set(tcgen05.Field.ACCUMULATE, False)
                cute.gemm(
                    lora_mma, tLtAcc,
                    tLrD[(None, None, None, lstage)],
                    tLrL1[(None, None, None, lstage)],
                    tLtAcc,
                )
                lora_full.release()

            acc_empty.commit()

        #
        # Epilogue: t2r-copy the accumulator(s), combine, store.
        #   fused : Y = dequant(main_acc) * global_scale + lora_acc  -> bf16
        #   plain : Y = dequant(main_acc) * global_scale             -> bf16
        #
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, ACC_DTYPE)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)

        rshape = tTR_gC[None, None, None, None, 0, 0, 0].shape
        tTR_rAcc = cute.make_rmem_tensor(rshape, ACC_DTYPE)
        tTR_rC = cute.make_rmem_tensor(rshape, self.c_dtype)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]

        if cutlass.const_expr(self.enable_lora):
            tiled_copy_t2r_lora = tcgen05.make_tmem_copy(copy_atom_t2r, tLtAcc)
            thr_copy_t2r_lora = tiled_copy_t2r_lora.get_slice(tidx)
            tTR_tLora = thr_copy_t2r_lora.partition_S(tLtAcc)
            tTR_rLora = cute.make_rmem_tensor(rshape, ACC_DTYPE)

        if warp_idx == 0:
            cute.arch.relinquish_tmem_alloc_permit()
        acc_full = acc_consumer.wait_and_advance()

        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        out_vec = tTR_rAcc.load() * global_scale            # fold per-tensor global scale here
        if cutlass.const_expr(self.enable_lora):
            cute.copy(tiled_copy_t2r_lora, tTR_tLora, tTR_rLora)
            out_vec = out_vec + tTR_rLora.load()
        tTR_rC.store(out_vec.to(self.c_dtype))
        cute.copy(simt_atom, tTR_rC, tTR_gC)

        acc_full.release()
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)
        return


# ---------------------------------------------------------------------------
# Wide-O 2-CTA path (single_linear1, O/I >= 8): a CtaGroup.TWO cooperative variant -- a 256-row
# M-tile computed by a 2-CTA cluster (128 rows/CTA) HALVES the M-tile count -> halves the per-(M,O)
# dual-TMEM epilogue add. Built from the GB200-validated dual_mma_spike_2cta mechanics (warp0 of BOTH
# CTAs issues multicast TMAs; the LEADER issues the MMAs; allocator-aware relinquish; tmem.free on all
# CTAs) plus this module's multi-stage K-loop + the lora-prologue overlap + enable_lora plain variant.
# The 1-CTA KernelB above is UNCHANGED (other shapes use it verbatim).
# ---------------------------------------------------------------------------
MMA_TILER_MN_2CTA = (256, 128)
CTA_GROUP_2 = tcgen05.CtaGroup.TWO
CLUSTER_SHAPE_MN_2CTA = (2, 1)


class KernelB2CTA:
    """2-CTA cooperative heterogeneous dual-TMEM fused main (wide-O). Same I/O contract as KernelB;
    `enable_lora=False` is the plain (main-only) variant with the identical main schedule."""

    def __init__(self, lora_k: int, enable_lora: bool = True, num_ab_stage: int = 3):
        self.threads_per_cta = 128
        self.lora_k = lora_k
        self.enable_lora = bool(enable_lora)
        self.num_tmem_alloc_cols = 512
        self.num_ab_stage = num_ab_stage
        self.num_lora_stage = 1
        self.num_acc_stage = 1

    @cute.jit
    def __call__(
        self,
        a_tensor: cute.Tensor, b_tensor: cute.Tensor,
        sfa_tensor: cute.Tensor, sfb_tensor: cute.Tensor,
        d_tensor: cute.Tensor, l1_tensor: cute.Tensor,
        c_tensor: cute.Tensor, global_scale: cutlass.Float32, stream: cuda.CUstream,
    ):
        self.a_dtype = a_tensor.element_type
        self.b_dtype = b_tensor.element_type
        self.sf_dtype = sfa_tensor.element_type
        self.lora_dtype = d_tensor.element_type
        self.c_dtype = c_tensor.element_type

        # 2-CTA tiled MMAs (256-row cluster tile). thr_id.shape == 2 -> use_2cta.
        self.mma_tiler = (MMA_TILER_MN_2CTA[0], MMA_TILER_MN_2CTA[1], NVFP4_TILE_K)
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, OperandMajorMode.K, OperandMajorMode.K,
            self.sf_dtype, SF_VEC, CTA_GROUP_2, MMA_TILER_MN_2CTA,
        )
        self.lora_tiler = (MMA_TILER_MN_2CTA[0], MMA_TILER_MN_2CTA[1], self.lora_k)
        lora_mma = sm100_utils.make_trivial_tiled_mma(
            self.lora_dtype, self.lora_dtype, OperandMajorMode.K, OperandMajorMode.K,
            ACC_DTYPE, CTA_GROUP_2, MMA_TILER_MN_2CTA,
        )
        # SFB: a SEPARATE 1-CTA blockscaled MMA (M halved for 2-CTA, N round-up 128); the weight block
        # scales are replicated across the peer CTA.
        mma_inst_shape_mn_sfb = (MMA_TILER_MN_2CTA[0] // 2, cute.round_up(MMA_TILER_MN_2CTA[1], 128))
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype, self.b_dtype, OperandMajorMode.K, OperandMajorMode.K,
            self.sf_dtype, SF_VEC, tcgen05.CtaGroup.ONE, mma_inst_shape_mn_sfb,
        )
        self.mma_tiler_sfb = (mma_inst_shape_mn_sfb[0], mma_inst_shape_mn_sfb[1], NVFP4_TILE_K)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*CLUSTER_SHAPE_MN_2CTA, 1)), (tiled_mma.thr_id.shape,))
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*CLUSTER_SHAPE_MN_2CTA, 1)), (tiled_mma_sfb.thr_id.shape,))

        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_tensor.shape, SF_VEC)
        sfa_tensor = cute.make_tensor(sfa_tensor.iterator, sfa_layout)
        sfb_layout = blockscaled_utils.tile_atom_to_shape_SF(b_tensor.shape, SF_VEC)
        sfb_tensor = cute.make_tensor(sfb_tensor.iterator, sfb_layout)

        self.a_smem_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage)
        self.b_smem_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage)
        self.sfa_smem_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage)
        self.sfb_smem_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, self.num_ab_stage)
        self.d_smem_layout = sm100_utils.make_smem_layout_a(
            lora_mma, self.lora_tiler, self.lora_dtype, self.num_lora_stage)
        self.l1_smem_layout = sm100_utils.make_smem_layout_b(
            lora_mma, self.lora_tiler, self.lora_dtype, self.num_lora_stage)
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        # Cluster-aware TMA atoms (multicast across the cluster), NOT the generic 1-CTA op.
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN_2CTA, tiled_mma.thr_id)
        a_sl = cute.slice_(self.a_smem_layout, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a_tensor, a_sl, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN_2CTA, tiled_mma.thr_id)
        b_sl = cute.slice_(self.b_smem_layout, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b_tensor, b_sl, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN_2CTA, tiled_mma.thr_id)
        sfa_sl = cute.slice_(self.sfa_smem_layout, (None, None, None, 0))
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op, sfa_tensor, sfa_sl, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(CLUSTER_SHAPE_MN_2CTA, tiled_mma.thr_id)
        sfb_sl = cute.slice_(self.sfb_smem_layout, (None, None, None, 0))
        tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op, sfb_tensor, sfb_sl, self.mma_tiler_sfb, tiled_mma_sfb,
            self.cluster_layout_sfb_vmnk.shape, internal_type=cutlass.Int16)
        d_op = sm100_utils.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN_2CTA, lora_mma.thr_id)
        d_sl = cute.slice_(self.d_smem_layout, (None, None, None, 0))
        tma_atom_d, tma_tensor_d = cute.nvgpu.make_tiled_tma_atom_A(
            d_op, d_tensor, d_sl, self.lora_tiler, lora_mma, self.cluster_layout_vmnk.shape)
        l1_op = sm100_utils.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN_2CTA, lora_mma.thr_id)
        l1_sl = cute.slice_(self.l1_smem_layout, (None, None, None, 0))
        tma_atom_l1, tma_tensor_l1 = cute.nvgpu.make_tiled_tma_atom_B(
            l1_op, l1_tensor, l1_sl, self.lora_tiler, lora_mma, self.cluster_layout_vmnk.shape)

        self.num_ab_tma_load_bytes = (
            cute.size_in_bytes(self.a_dtype, a_sl) + cute.size_in_bytes(self.b_dtype, b_sl)
            + cute.size_in_bytes(self.sf_dtype, sfa_sl) + cute.size_in_bytes(self.sf_dtype, sfb_sl)
        ) * atom_thr_size
        self.num_lora_tma_load_bytes = (
            cute.size_in_bytes(self.lora_dtype, d_sl) + cute.size_in_bytes(self.lora_dtype, l1_sl)
        ) * atom_thr_size

        grid = (
            cute.ceil_div(c_tensor.shape[0], self.mma_tiler[0]) * atom_thr_size,
            cute.ceil_div(c_tensor.shape[1], self.mma_tiler[1]),
            c_tensor.shape[2],
        )
        self.kernel(
            tiled_mma, lora_mma, tiled_mma_sfb,
            tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b,
            tma_atom_sfa, tma_tensor_sfa, tma_atom_sfb, tma_tensor_sfb,
            tma_atom_d, tma_tensor_d, tma_atom_l1, tma_tensor_l1,
            c_tensor, global_scale,
            self.a_smem_layout, self.b_smem_layout, self.sfa_smem_layout, self.sfb_smem_layout,
            self.d_smem_layout, self.l1_smem_layout,
            self.cluster_layout_vmnk, self.cluster_layout_sfb_vmnk,
        ).launch(
            grid=grid, block=[self.threads_per_cta, 1, 1],
            cluster=(*CLUSTER_SHAPE_MN_2CTA, 1), stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma, lora_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom, mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom, mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom, mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom, mSFB_nkl: cute.Tensor,
        tma_atom_d: cute.CopyAtom, mD_mkl: cute.Tensor,
        tma_atom_l1: cute.CopyAtom, mL1_nkl: cute.Tensor,
        mC_mnl: cute.Tensor, global_scale: cutlass.Float32,
        a_smem_layout: cute.ComposedLayout, b_smem_layout: cute.ComposedLayout,
        sfa_smem_layout: cute.Layout, sfb_smem_layout: cute.Layout,
        d_smem_layout: cute.ComposedLayout, l1_smem_layout: cute.ComposedLayout,
        cluster_layout_vmnk: cute.Layout, cluster_layout_sfb_vmnk: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        mma_tile_coord_mnl = (bidx // cute.size(tiled_mma.thr_id.shape), bidy, bidz)

        @cute.struct
        class SharedStorage:
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            lora_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_lora_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sA = smem.allocate_tensor(self.a_dtype, a_smem_layout.outer, 128, a_smem_layout.inner)
        sB = smem.allocate_tensor(self.b_dtype, b_smem_layout.outer, 128, b_smem_layout.inner)
        sSFA = smem.allocate_tensor(self.sf_dtype, sfa_smem_layout, 128)
        sSFB = smem.allocate_tensor(self.sf_dtype, sfb_smem_layout, 128)
        sD = smem.allocate_tensor(self.lora_dtype, d_smem_layout.outer, 128, d_smem_layout.inner)
        sL1 = smem.allocate_tensor(self.lora_dtype, l1_smem_layout.outer, 128, l1_smem_layout.inner)

        # 2-CTA pipelines: the ab consumer count is the multicast-CTA count (vendored), NOT 1.
        num_mcast_ctas_a = cute.size(cluster_layout_vmnk.shape[2])
        num_mcast_ctas_b = cute.size(cluster_layout_vmnk.shape[1])
        ab_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_mcast_ctas_a + num_mcast_ctas_b - 1),
            tx_count=self.num_ab_tma_load_bytes, cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        lora_producer, lora_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.lora_mbar_ptr.data_ptr(), num_stages=self.num_lora_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_mcast_ctas_a + num_mcast_ctas_b - 1),
            tx_count=self.num_lora_tma_load_bytes, cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(), num_stages=self.num_acc_stage,
            producer_group=ab_producer_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.threads_per_cta),
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        gA = cute.local_tile(mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        gSFA = cute.local_tile(mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gSFB = cute.local_tile(mSFB_nkl, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
        gD = cute.local_tile(mD_mkl, cute.slice_(self.lora_tiler, (None, 0, None)), (None, None, None))
        gL1 = cute.local_tile(mL1_nkl, cute.slice_(self.lora_tiler, (0, None, None)), (None, None, None))
        gC = cute.local_tile(mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None))

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA)
        tCgB = thr_mma.partition_B(gB)
        tCgSFA = thr_mma.partition_A(gSFA)
        thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
        tCgSFB = thr_mma_sfb.partition_B(gSFB)
        tCgC = thr_mma.partition_C(gC)
        thr_lora = lora_mma.get_slice(mma_tile_coord_v)
        tLgD = thr_lora.partition_A(gD)
        tLgL1 = thr_lora.partition_B(gL1)

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sA, 0, 3), cute.group_modes(tCgA, 0, 3))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sB, 0, 3), cute.group_modes(tCgB, 0, 3))
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sSFA, 0, 3), cute.group_modes(tCgSFA, 0, 3))
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout,
            cute.group_modes(sSFB, 0, 3), cute.group_modes(tCgSFB, 0, 3))
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        tDsD, tDgD = cpasync.tma_partition(
            tma_atom_d, block_in_cluster_coord_vmnk[2], a_cta_layout,
            cute.group_modes(sD, 0, 3), cute.group_modes(tLgD, 0, 3))
        tL1sL1, tL1gL1 = cpasync.tma_partition(
            tma_atom_l1, block_in_cluster_coord_vmnk[1], b_cta_layout,
            cute.group_modes(sL1, 0, 3), cute.group_modes(tLgL1, 0, 3))

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tLrD = lora_mma.make_fragment_A(sD)
        tLrL1 = lora_mma.make_fragment_B(sL1)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)
        lora_acc_shape = lora_mma.partition_shape_C(self.mma_tiler[:2])
        tLtAcc_fake = lora_mma.make_fragment_C(lora_acc_shape)

        tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads_per_cta)
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_alloc_barrier,
            is_two_cta=True, two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        acc_tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
        tCtAcc = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
        lora_acc_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc), dtype=ACC_DTYPE)
        tLtAcc = cute.make_tensor(lora_acc_ptr, tLtAcc_fake.layout)

        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_smem_layout, (None, None, None, 0)))
        sfa_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tLtAcc), dtype=self.sf_dtype)
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
        tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfb_smem_layout, (None, None, None, 0)))
        sfb_tmem_ptr = cute.recast_ptr(
            acc_tmem_ptr + tcgen05.find_tmem_tensor_col_offset(tCtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tLtAcc)
            + tcgen05.find_tmem_tensor_col_offset(tCtSFA), dtype=self.sf_dtype)
        tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

        copy_atom_s2t = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(CTA_GROUP_2), self.sf_dtype)
        tCsSFA_compact = cute.filter_zeros(sSFA)
        tCtSFA_compact = cute.filter_zeros(tCtSFA)
        tiled_copy_s2t_sfa = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFA_compact)
        thr_copy_s2t_sfa = tiled_copy_s2t_sfa.get_slice(0)
        tCsSFA_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfa, thr_copy_s2t_sfa.partition_S(tCsSFA_compact))
        tCtSFA_s2t = thr_copy_s2t_sfa.partition_D(tCtSFA_compact)
        tCsSFB_compact = cute.filter_zeros(sSFB)
        tCtSFB_compact = cute.filter_zeros(tCtSFB)
        tiled_copy_s2t_sfb = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSFB_compact)
        thr_copy_s2t_sfb = tiled_copy_s2t_sfb.get_slice(0)
        tCsSFB_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t_sfb, thr_copy_s2t_sfb.partition_S(tCsSFB_compact))
        tCtSFB_s2t = thr_copy_s2t_sfb.partition_D(tCtSFB_compact)

        tAgA = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgB = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tAgSFA = tAgSFA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tBgSFB = tBgSFB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]
        tDgD = tDgD[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
        tL1gL1 = tL1gL1[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

        a_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
        b_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
        sfb_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)

        k_tile_cnt = cute.size(gA, mode=[3])

        # warp0 of BOTH CTAs PRODUCES (multicast TMA); the LEADER also CONSUMES (the MMAs). The leader
        # and peer paths are SEPARATE `if is_leader_cta` regions so the leader-only pipeline handles
        # (acc_empty, ab_full, ...) never cross a dynamic-control-flow boundary (the cute tracer forbids
        # using a variable defined in one runtime `if` inside another). The peer issues the same
        # multicast TMAs to fill the cluster pipeline buffers for the leader's MMAs.
        if warp_idx == 0:
            if is_leader_cta:
                acc_empty = acc_producer.acquire_and_advance()
                # Lora-prologue overlap: issue the low-rank D/L1 TMA BEFORE the main K-loop.
                if cutlass.const_expr(self.enable_lora):
                    lora_empty = lora_producer.acquire_and_advance()
                    lidx = lora_empty.index
                    cute.copy(tma_atom_d, tDgD[(None, 0)], tDsD[(None, lidx)], tma_bar_ptr=lora_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_l1, tL1gL1[(None, 0)], tL1sL1[(None, lidx)], tma_bar_ptr=lora_empty.barrier, mcast_mask=b_mask)
                for k_tile in range(k_tile_cnt):
                    ab_empty = ab_producer.acquire_and_advance()
                    idx = ab_empty.index
                    cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=b_mask)
                    cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_sfb, tBgSFB[(None, k_tile)], tBsSFB[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=sfb_mask)
                    ab_full = ab_consumer.wait_and_advance()
                    stage = ab_full.index
                    cute.copy(tiled_copy_s2t_sfa, tCsSFA_s2t[(None, None, None, None, stage)], tCtSFA_s2t)
                    cute.copy(tiled_copy_s2t_sfb, tCsSFB_s2t[(None, None, None, None, stage)], tCtSFB_s2t)
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                    cute.gemm(tiled_mma, tCtAcc,
                              [tCrA[(None, None, None, stage)], tCtSFA],
                              [tCrB[(None, None, None, stage)], tCtSFB], tCtAcc)
                    ab_full.release()
                if cutlass.const_expr(self.enable_lora):
                    lora_full = lora_consumer.wait_and_advance()
                    lstage = lora_full.index
                    lora_mma.set(tcgen05.Field.ACCUMULATE, False)
                    cute.gemm(lora_mma, tLtAcc,
                              tLrD[(None, None, None, lstage)], tLrL1[(None, None, None, lstage)], tLtAcc)
                    lora_full.release()
                acc_empty.commit()
            else:
                # PEER: producer only (same multicast TMAs; no acc, no MMA, no consume).
                if cutlass.const_expr(self.enable_lora):
                    lora_empty = lora_producer.acquire_and_advance()
                    lidx = lora_empty.index
                    cute.copy(tma_atom_d, tDgD[(None, 0)], tDsD[(None, lidx)], tma_bar_ptr=lora_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_l1, tL1gL1[(None, 0)], tL1sL1[(None, lidx)], tma_bar_ptr=lora_empty.barrier, mcast_mask=b_mask)
                for k_tile in range(k_tile_cnt):
                    ab_empty = ab_producer.acquire_and_advance()
                    idx = ab_empty.index
                    cute.copy(tma_atom_a, tAgA[(None, k_tile)], tAsA[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_b, tBgB[(None, k_tile)], tBsB[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=b_mask)
                    cute.copy(tma_atom_sfa, tAgSFA[(None, k_tile)], tAsSFA[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=a_mask)
                    cute.copy(tma_atom_sfb, tBgSFB[(None, k_tile)], tBsSFB[(None, idx)], tma_bar_ptr=ab_empty.barrier, mcast_mask=sfb_mask)

        # Epilogue (per CTA: each CTA's 128 rows of the 256-row cluster tile).
        op = tcgen05.Ld32x32bOp(tcgen05.Repetition.x128, tcgen05.Pack.NONE)
        copy_atom_t2r = cute.make_copy_atom(op, ACC_DTYPE)
        tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tCtAcc)
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        tTR_tAcc = thr_copy_t2r.partition_S(tCtAcc)
        tTR_gC = thr_copy_t2r.partition_D(tCgC)
        rshape = tTR_gC[None, None, None, None, 0, 0, 0].shape
        tTR_rAcc = cute.make_rmem_tensor(rshape, ACC_DTYPE)
        tTR_rC = cute.make_rmem_tensor(rshape, self.c_dtype)
        simt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.c_dtype)
        tTR_gC = tTR_gC[(None, None, None, None, *mma_tile_coord_mnl)]
        if cutlass.const_expr(self.enable_lora):
            tiled_copy_t2r_lora = tcgen05.make_tmem_copy(copy_atom_t2r, tLtAcc)
            thr_copy_t2r_lora = tiled_copy_t2r_lora.get_slice(tidx)
            tTR_tLora = thr_copy_t2r_lora.partition_S(tLtAcc)
            tTR_rLora = cute.make_rmem_tensor(rshape, ACC_DTYPE)

        tmem.relinquish_alloc_permit()
        acc_full = acc_consumer.wait_and_advance()
        cute.copy(tiled_copy_t2r, tTR_tAcc, tTR_rAcc)
        out_vec = tTR_rAcc.load() * global_scale
        if cutlass.const_expr(self.enable_lora):
            cute.copy(tiled_copy_t2r_lora, tTR_tLora, tTR_rLora)
            out_vec = out_vec + tTR_rLora.load()
        tTR_rC.store(out_vec.to(self.c_dtype))
        cute.copy(simt_atom, tTR_rC, tTR_gC)
        acc_full.release()
        cute.arch.barrier()
        tmem.free(acc_tmem_ptr)
        return


# ---------------------------------------------------------------------------
# Host: build inputs (from the front-end's emitted bytes + prepacked weights), run the fused main.
# ---------------------------------------------------------------------------

_COMPILE_CACHE = {}


# Explicit (I, O, M) allowlist for the UP-bound square / wide-O-ish rows the 2-CTA dual-TMEM epilogue
# also targets (the per-(M,O)-tile lora_acc add rivals/exceeds the down on these per the ncu floor
# table): double_attn (square, I=O=6144) M4096/M4608 + double_ff_in (wide-O-ish, I=6144, O=36864)
# M4608. All satisfy the 256-row cluster M-tile (M % 256 == 0) and are >8% at both ranks. single_linear1
# (O/I>=8) keeps its shipped heuristic path; every non-listed shape stays on the validated 1-CTA KernelB.
_WIDE_O_2CTA_ALLOWLIST = {
    (6144, 6144, 4096),    # double_attn  M4096 (square)
    (6144, 6144, 4608),    # double_attn  M4608 (square)
    (6144, 36864, 4608),   # double_ff_in M4608 (wide-O-ish)
}


def _wide_o_2cta(I, O, M):
    """Shape selection for the 2-CTA cooperative dual-TMEM path: a 256-row cluster M-tile must divide M,
    AND the shape is either wide-O (O/I >= 8 -> single_linear1, the shipped path) or one of the explicitly
    allow-listed UP-bound square / wide-O-ish rows. Every other shape uses the validated 1-CTA KernelB."""
    if M % MMA_TILER_MN_2CTA[0] != 0:
        return False
    if O >= 8 * I:
        return True
    return (I, O, M) in _WIDE_O_2CTA_ALLOWLIST


def _resolve_use_2cta(M, I, O, use_2cta=None):
    """Resolve the production path selector. `None` (the default for production callers) auto-selects
    the wide-O 2-CTA path via `_wide_o_2cta`; an explicit bool stays available for the `--check-2cta`
    diagnostic. An explicit `True` on a shape that violates the 2-CTA preconditions is REJECTED rather
    than silently compiling an unsupported partial path."""
    if use_2cta is None:
        return _wide_o_2cta(I, O, M)
    if use_2cta and not _wide_o_2cta(I, O, M):
        raise ValueError(
            f"use_2cta=True requires the wide-O 2-CTA preconditions (O/I>=8 and M%%256==0); "
            f"got M={M} I={I} O={O} (O/I={O / I:.2f}).")
    return bool(use_2cta)


# Production num_ab_stage (main NVFP4 K-loop pipeline depth) dispatch, keyed by (I, O, M, r). Default 3
# (the validated depth); a deeper stage is baked here ONLY for a paired-confirmed TGT-IMPR autotuner row.
# A stage change IS a dispatch-path change (it re-keys the compile cache + the SMEM staging depth), so it is
# emitted in the sweep CSV + the dispatch-map audit alongside the down (m_tile,n_tile,split) + the main path.
_NUM_AB_STAGE_BY_IOMR = {
    # (I, O, M, r): num_ab_stage -- populated ONLY by paired-confirmed wins. EMPTY: rounds 4-5's FULL
    # cross-product found NO stage win that survives the 8-round paired gate (single_linear1 M1024 r4 s4 was
    # TGT-FLAT @ run.r4_sl1_gate; double_attn M1024 r4 s4's target delta was within the tiny-shape noise @
    # run.r5_dattn_gate, which also surfaced byte-identical non-target noise). Production stays s3 everywhere.
}


def num_ab_stage_for_shape(I, O, M, r):
    """The dispatched main-loop pipeline depth for (I,O,M,r); 3 unless a paired-confirmed win baked a deeper one."""
    return _NUM_AB_STAGE_BY_IOMR.get((I, O, M, r), 3)


def _compiled_kernel_b(M, I, O, r, enable_lora, num_ab_stage=3, use_2cta=False,
                       num_tmem_alloc_cols=512, num_acc_stage=1):
    """Compile cache keyed by the static shape/variant (JIT excluded from any timing)."""
    key = (M, I, O, r, bool(enable_lora), num_ab_stage, bool(use_2cta), num_tmem_alloc_cols, num_acc_stage)
    if key in _COMPILE_CACHE:
        return _COMPILE_CACHE[key]
    # Representative tensors with the right shapes/dtypes drive the JIT trace.
    a_tensor, _ = _fp4_from_packed(M, I, torch.zeros(M, I // 2, dtype=torch.uint8).cuda())
    b_tensor, _ = _fp4_from_packed(O, I, torch.zeros(O, I // 2, dtype=torch.uint8).cuda())
    sfa_tensor, _ = _make_sf_mma_tensor(torch.ones(M, I // SF_VEC), M, I)
    sfb_tensor, _ = _make_sf_mma_tensor(torch.ones(O, I // SF_VEC), O, I)
    d_tensor = _kmajor_cuda_tensor(torch.zeros(M, r, dtype=torch.bfloat16).cuda(),
                                   LORA_DTYPE, torch.bfloat16)
    l1_tensor = _kmajor_cuda_tensor(torch.zeros(O, r, dtype=torch.bfloat16).cuda(),
                                    LORA_DTYPE, torch.bfloat16)
    c_ref = cutlass_torch.matrix(1, M, O, False, OUT_DTYPE)
    c_tensor, _ = cutlass_torch.cute_tensor_like(c_ref, OUT_DTYPE, True, 16)
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
    stream = cutlass_torch.default_stream()
    cls = KernelB2CTA if use_2cta else KernelB
    if use_2cta:
        kb = cls(r, enable_lora=enable_lora, num_ab_stage=num_ab_stage)
    else:
        kb = cls(r, enable_lora=enable_lora, num_ab_stage=num_ab_stage,
                 num_tmem_alloc_cols=num_tmem_alloc_cols, num_acc_stage=num_acc_stage)
    compiled = cute.compile(kb, a_tensor, b_tensor, sfa_tensor, sfb_tensor,
                            d_tensor, l1_tensor, c_tensor, cutlass.Float32(1.0), stream)
    _COMPILE_CACHE[key] = compiled
    return compiled


def run_kernel_b(Xq, sf_x_bytes, D, Rq, sf_w_bytes, L1, gscale_x, gscale_w, r,
                 enable_lora=True, num_ab_stage=3, use_2cta=None):
    """Run the fused main on the front-end's emitted (Xq, sf_x, D) + prepacked (Rq, sf_w, gscale_w) +
    bf16 L1. Returns Y [M, O] bf16 on CUDA. enable_lora=False compiles/runs the plain (main-only)
    variant. use_2cta=None auto-selects the wide-O 2-CTA path for O/I>=8 & M%256==0 (single_linear1)."""
    M = Xq.shape[0]
    I = Xq.shape[1] * 2
    O = Rq.shape[0]
    use_2cta = _resolve_use_2cta(M, I, O, use_2cta)
    # _make_sf_mma_tensor stages the scales on a CPU tensor and copies them in; the front-end emits sf_x
    # on CUDA, so bring both scale-byte buffers to host explicitly (an unambiguous host->host copy) rather
    # than relying on PyTorch's implicit cross-device slice assignment. Values (hence bit-exactness) are
    # unchanged; run_kernel_b is the correctness path (timing uses the device-side SF swizzle instead).
    sf_x = sf_x_bytes.cpu().view(torch.float8_e4m3fn).float()
    sf_w = sf_w_bytes.cpu().view(torch.float8_e4m3fn).float()
    global_scale = float(gscale_x * gscale_w)

    a_tensor, _ = _fp4_from_packed(M, I, Xq.cuda())
    b_tensor, _ = _fp4_from_packed(O, I, Rq.cuda())
    sfa_tensor, _ = _make_sf_mma_tensor(sf_x, M, I)
    sfb_tensor, _ = _make_sf_mma_tensor(sf_w, O, I)
    # bf16 low-rank operands, CUDA-resident (no CPU staging): D [M, r], L1 [O, r], both K-major.
    d_tensor = _kmajor_cuda_tensor(D.to(torch.bfloat16), LORA_DTYPE, torch.bfloat16)
    l1_tensor = _kmajor_cuda_tensor(L1.to(device="cuda", dtype=torch.bfloat16),
                                    LORA_DTYPE, torch.bfloat16)

    c_ref = cutlass_torch.matrix(1, M, O, False, OUT_DTYPE)
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(c_ref, OUT_DTYPE, True, 16)
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)

    compiled = _compiled_kernel_b(M, I, O, r, enable_lora, num_ab_stage, use_2cta)
    stream = cutlass_torch.default_stream()
    compiled(a_tensor, b_tensor, sfa_tensor, sfb_tensor, d_tensor, l1_tensor,
             c_tensor, cutlass.Float32(global_scale), stream)
    torch.cuda.synchronize()
    return c_torch[:, :, 0]


# ---------------------------------------------------------------------------
# On-device SF swizzle (the official harness / fast-path uses this in place of host staging).
# ---------------------------------------------------------------------------

_SF_PERM_CACHE = {}


def _sf_swizzle_perm(mn, k):
    """A CUDA gather permutation that reproduces _make_sf_mma_tensor's MMA-swizzled E4M3 layout
    ON DEVICE: replicate the same ref(MKL)->mma(M32x4xK4) swizzle on a Float32 arange (NOT float8 --
    that would saturate the index at ~448), so the swizzled mma tensor holds, at each logical position,
    its source row-major index. Then for any row-major [mn, sf_k] E4M3 byte tensor,
    swizzled_logical = sf_row_major.flatten()[perm]. Cached per (mn, k)."""
    key = (mn, k)
    if key not in _SF_PERM_CACHE:
        sf_k = (k + SF_VEC - 1) // SF_VEC
        ref_shape = (1, mn, sf_k)
        mma_shape = (1, (mn + _ATOM_M[0] * _ATOM_M[1] - 1) // (_ATOM_M[0] * _ATOM_M[1]),
                     (sf_k + _ATOM_K - 1) // _ATOM_K,
                     _ATOM_M[0], _ATOM_M[1], _ATOM_K)
        # The swizzle is a pure position permutation, but it is materialized by routing the source
        # row-major indices through a FLOAT32 cute copy, which is exact only to 2^24. Large activations
        # (batch x resolution, e.g. M=27556 sf_k=768 -> 21.2M) would silently round to garbage indices.
        # Fix: split each index into 14-bit limbs (each < 2^14, exactly float32-representable), swizzle
        # every limb through the SAME copy, and recombine in int64. The permutation moves all limbs
        # identically, so recombining yields the exact swizzled source index for any mn. Cached per (mn,k).
        n = mn * sf_k
        SHIFT = 14
        n_limbs = 1
        while (1 << (SHIFT * n_limbs)) < max(n, 1):
            n_limbs += 1
        src = torch.arange(n, dtype=torch.int64).reshape(mn, sf_k)
        limbs = []
        for li in range(n_limbs):
            part = ((src >> (SHIFT * li)) & ((1 << SHIFT) - 1)).to(torch.float32)
            ref = cutlass_torch.create_and_permute_torch_tensor(
                ref_shape, torch.float32, permute_order=(1, 2, 0),
                init_type=cutlass_torch.TensorInitType.RANDOM,
                init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1))   # (mn, sf_k, 1)
            ref[:, :, 0] = part
            mma = cutlass_torch.create_and_permute_torch_tensor(
                mma_shape, torch.float32, permute_order=(3, 4, 1, 5, 2, 0),
                init_type=cutlass_torch.TensorInitType.RANDOM,
                init_config=cutlass_torch.RandomInitConfig(min_val=0, max_val=1))
            _cvt_sf_to_mma(cute.runtime.from_dlpack(ref), cute.runtime.from_dlpack(mma))
            limbs.append(mma.reshape(-1).round().to(torch.int64))
        perm = torch.zeros_like(limbs[0])
        for li in range(n_limbs):
            perm = perm | (limbs[li] << (SHIFT * li))
        _SF_PERM_CACHE[key] = perm.cuda()
    return _SF_PERM_CACHE[key]


def _refresh_sfa_device(sf_row_major_u8_cuda, perm, sfa_torch, g_buf=None):
    """On-device SF swizzle (no host staging): gather the row-major E4M3 bytes by `perm` and write them
    into `sfa_torch`'s (strided) storage with copy_ (which respects strides), so the cute SF tensor
    wrapping sfa_torch now reflects the swizzled sf_x. Timed inside _one().

    g_buf (optional): a pre-allocated uint8 [perm.numel()] buffer. When given, the gather uses
    torch.index_select(..., out=g_buf) so the per-call path allocates NOTHING (required for CUDA-graph
    capture); when None, the fancy-index gather (which allocates a temp) is used (timing/validation).
    Both produce byte-identical results."""
    flat = sf_row_major_u8_cuda.reshape(-1)
    if g_buf is None:
        g = flat[perm]                                             # fancy-index gather (allocates a temp)
    else:
        torch.index_select(flat, 0, perm, out=g_buf)               # alloc-free gather into the pre-alloc buffer
        g = g_buf
    sfa_torch.view(torch.uint8).copy_(g.reshape(sfa_torch.shape))  # strided write into the cute buffer


def check_sf_swizzle(mn, k):
    """Verify the on-device perm-gather SF swizzle is BIT-IDENTICAL to the host _make_sf_mma_tensor."""
    sf_k = (k + SF_VEC - 1) // SF_VEC
    sf_u8 = torch.randint(1, 220, (mn, sf_k), dtype=torch.uint8)
    sf_f8 = sf_u8.view(torch.float8_e4m3fn)
    _, ref_torch = _make_sf_mma_tensor(sf_f8.float(), mn, k)        # host swizzle (reference)
    perm = _sf_swizzle_perm(mn, k)
    dev_torch = torch.empty_like(ref_torch)
    _refresh_sfa_device(sf_u8.cuda(), perm, dev_torch)
    ok = torch.equal(dev_torch.view(torch.uint8).cpu(), ref_torch.view(torch.uint8).cpu())
    print(f"[sf-swizzle {mn}x{k}] device perm-gather == host swizzle: {'OK (bit-identical)' if ok else 'MISMATCH'} "
          f"({ref_torch.numel()} E4M3 elems)")
    return ok


# ---------------------------------------------------------------------------
# Deployable host fast-path: prepare_svdquant_state(...) + mm_fp4_svdquant(x, state).
# A reusable, CUDA-graph-safe dispatch. ALL per-call host work (weight pack consumption, weight-SF
# swizzle, scale fold, buffer alloc, JIT) is hoisted into prepare_svdquant_state; mm_fp4_svdquant does
# NO torch.cuda.synchronize / .cpu() / .item() / per-call allocation. The on-device kernel sequence
# (quantizer -> down/reduce -> the fused main) and every dispatch tuple are byte-identical to the
# dataflow-connected timing path, so device latency and zero-LoRA bit-exactness are unchanged -- only
# the host dispatch around the kernels changes.
# ---------------------------------------------------------------------------

def prepare_svdquant_state(M, Rq, sf_w_bytes, L1, L2, gscale_x, gscale_w, r,
                           enable_lora=True, pre_quant_scale=None,
                           num_ab_stage=None, use_2cta=None, external_quant=False,
                           num_tmem_alloc_cols=512, num_acc_stage=1):
    """Build a reusable dispatch state for one (M, I, O, r) layer shape. Off the hot path:
      * derive I, O from the prepacked weight (Rq [O, I//2], sf_w_bytes [O, I//16]);
      * swizzle the CONSTANT weight SF (sfb) ONCE (host swizzle, prepare-time only -- never per call);
      * pre-allocate the quantizer/down/main operands + the alloc-free SFA gather buffer;
      * compile (JIT-excluded) the quantizer, down/reduce, and the fused main for the shape;
      * fold the STATIC calibrated gscale_x and (gscale_x*gscale_w) into cutlass.Float32 (graph-safe;
        no per-call .item()).
    gscale_x must be a precomputed scalar (the ModelOpt calibrated static activation scale) for the
    CUDA-graph path; an eager-only dynamic scale is computed by the caller before prepare. enable_lora
    builds the full fused state; enable_lora=False builds the plain-NVFP4 (main-only) state (L1/L2 may
    be None). The dispatch (down tile/split, num_ab_stage, main-kernel 1-CTA/2-CTA) matches the
    production selectors exactly -- this changes no device behavior."""
    I = Rq.shape[1] * 2
    O = Rq.shape[0]
    sf_k = I // SF_VEC
    if num_ab_stage is None:
        num_ab_stage = num_ab_stage_for_shape(I, O, M, r)
    use_2cta = _resolve_use_2cta(M, I, O, use_2cta)   # same validated selector run_kernel_b enforces
    m_tile, n_tile, r_pad, num_k_split = down_tile_for_shape(I, r, M, O=O)

    # Quantizer stage. xq_t/sf_t are the NVFP4 activation operands (and the injection targets).
    # external_quant=True (host framework injects its own NVFP4 activation via ext_xq/ext_sf): the
    # on-device quantizer and its f32 X backing are never used, so skip them -- drops the M*I*4 f32
    # allocation per state (the dominant per-state memory at large M/batch) and the quantizer JIT.
    xq_t, xq_torch = _u8_rowmajor_tensor(M, I // 2)
    sf_t, sf_torch = _u8_rowmajor_tensor(M, sf_k)
    if external_quant:
        xf_t = None
        quant = None
    else:
        xf_t = _f32_kmajor_tensor(torch.zeros(M, I, dtype=torch.float32, device="cuda"))
        quant = _compiled_quantizer(M, I)

    # Down stage (fused only): a FIXED bf16 X backing + the real L2 (padded to r_pad) -> D, written
    # straight into the fused main's D operand. d_tensor exists in both variants (the plain main ignores it).
    xb_t = None
    down_call = None
    d_tensor = _kmajor_cuda_tensor(torch.zeros(M, r, dtype=torch.bfloat16, device="cuda"),
                                   LORA_DTYPE, torch.bfloat16)
    if enable_lora:
        xb_t = _bf16_kmajor_tensor(torch.zeros(M, I, dtype=torch.bfloat16, device="cuda"))
        L2b = L2.to(device="cuda", dtype=torch.bfloat16)
        if r_pad != r:
            L2_pad = torch.zeros(r_pad, I, dtype=torch.bfloat16, device="cuda")
            L2_pad[:r] = L2b
            L2b = L2_pad
        l2_t = _bf16_kmajor_tensor(L2b)
        if num_k_split == 1:
            d_ref = cutlass_torch.matrix(1, M, r_pad, False, LORA_DTYPE)
            d_down_t, d_down_torch = cutlass_torch.cute_tensor_like(d_ref, LORA_DTYPE, True, 16)
            d_down_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)
            _down = _compiled_down(M, I, r_pad, n_tile, m_tile)

            def down_call(st, _down=_down, xb=xb_t, l2=l2_t, dd=d_down_t, ddt=d_down_torch, dk=d_tensor, rr=r):
                _down(xb, l2, dd, st)
                dk._svdquant_backing[:, :, 0] = ddt[:, :rr, 0]      # padded [M, r_pad] -> the fused main's D [M, r]
        else:
            _down = _compiled_down_split(M, I, r_pad, n_tile, m_tile, num_k_split)
            _reduce = _compiled_reduce(M, r_pad, num_k_split, r)
            p_ref = cutlass_torch.matrix(num_k_split, M, r_pad, False, ACC_DTYPE)
            p_t, _ = cutlass_torch.cute_tensor_like(p_ref, ACC_DTYPE, True, 16)
            p_t.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)

            def down_call(st, _down=_down, _reduce=_reduce, xb=xb_t, l2=l2_t, p=p_t, dk=d_tensor):
                _down(xb, l2, p, st)
                _reduce(p, dk, st)        # reduce writes the r real cols straight into the fused main's D -- no copy

    # Fused-main operands. A / SFA are refreshed per call from the quantizer outputs; the weights are FIXED.
    a_tensor, a_buf = _fp4_from_packed(M, I, torch.zeros(M, I // 2, dtype=torch.uint8).cuda())
    b_tensor, b_buf = _fp4_from_packed(O, I, Rq.to(device="cuda"))   # b_buf MUST be retained (raw-packed operand does not own its storage)
    sfa_tensor, sfa_torch = _make_sf_mma_tensor(torch.ones(M, sf_k), M, I)
    # CONSTANT weight SF, swizzled ONCE here (prepare-time host swizzle; never per call):
    sf_w_f = sf_w_bytes.detach().to("cpu").view(torch.float8_e4m3fn).float()
    sfb_tensor, _ = _make_sf_mma_tensor(sf_w_f, O, I)
    l1_src = L1 if (enable_lora and L1 is not None) else torch.zeros(O, r, dtype=torch.bfloat16, device="cuda")
    l1_tensor = _kmajor_cuda_tensor(l1_src.to(device="cuda", dtype=torch.bfloat16),
                                    LORA_DTYPE, torch.bfloat16)
    c_ref = cutlass_torch.matrix(1, M, O, False, OUT_DTYPE)
    c_tensor, c_torch = cutlass_torch.cute_tensor_like(c_ref, OUT_DTYPE, True, 16)
    c_tensor.mark_compact_shape_dynamic(mode=1, stride_order=(2, 0, 1), divisibility=8)

    perm = _sf_swizzle_perm(M, I)
    g_buf = torch.empty(perm.numel(), dtype=torch.uint8, device="cuda")   # alloc-free SFA gather buffer
    pqs_dev = None
    if pre_quant_scale is not None:
        pqs_dev = pre_quant_scale.detach().to(device="cuda", dtype=torch.float32).reshape(1, I)

    kb = _compiled_kernel_b(M, I, O, r, enable_lora, num_ab_stage, use_2cta,
                            num_tmem_alloc_cols=num_tmem_alloc_cols, num_acc_stage=num_acc_stage)
    return {
        "M": M, "I": I, "O": O, "r": r, "enable_lora": bool(enable_lora),
        "quant": quant, "xf_t": xf_t, "xq_t": xq_t, "sf_t": sf_t, "xq_torch": xq_torch, "sf_torch": sf_torch,
        "xb_t": xb_t, "down_call": down_call, "d_tensor": d_tensor,
        "a_tensor": a_tensor, "a_flat": a_buf.as_strided((M * I,), (1,)), "a_buf": a_buf, "b_buf": b_buf,
        "b_tensor": b_tensor, "sfa_tensor": sfa_tensor, "sfa_torch": sfa_torch, "sfb_tensor": sfb_tensor,
        "l1_tensor": l1_tensor, "c_tensor": c_tensor, "c_torch": c_torch,
        "perm": perm, "g_buf": g_buf, "pqs": pqs_dev,
        "gx": cutlass.Float32(float(gscale_x)), "gs": cutlass.Float32(float(gscale_x) * float(gscale_w)),
        "kb": kb, "stream": cutlass_torch.default_stream(),
        "down_m_tile": m_tile, "down_n_tile": n_tile, "down_split": num_k_split, "num_ab_stage": num_ab_stage,
        "use_2cta": use_2cta,
    }


def _run_kernels(c, st, ext_xq=None, ext_sf=None):
    """PRIVATE: the on-device kernel sequence (quant -> repack A -> on-device SFA refresh -> down/reduce ->
    the fused main) on the activation ALREADY staged in c['xf_t']/c['xb_t']. No host sync / copy / allocation
    (the only transient is the Xq int8 view, an in-pool reinterpret). Used by mm_fp4_svdquant (after it stages
    the activation) and by the timing harness for a kernel-sequence-only graph (load excluded). Returns the
    c_torch output view.

    External activation quant (numerical-parity hook): if ext_xq/ext_sf are given, the on-device quantizer is
    SKIPPED and the residual main GEMM uses the supplied NVFP4 activation instead (e.g. from TRT-LLM
    fp4_quantize, to match its rounding). ext_xq: uint8 [M, I//2] packed E2M1 (same packing as the internal
    Xq); ext_sf: uint8 PLAIN [M, I//16] E4M3 (gets the same on-device SFA swizzle as the internal sf_x).
    The bf16 down-projection (LoRA) still uses the staged f32 x_hat, unchanged."""
    M, I = c["M"], c["I"]
    if ext_xq is not None:
        c["xq_torch"].copy_(ext_xq.view(torch.uint8).reshape(c["xq_torch"].shape))
        c["sf_torch"].copy_(ext_sf.view(torch.uint8).reshape(c["sf_torch"].shape))
    else:
        c["quant"](c["xf_t"], c["xq_t"], c["sf_t"], c["gx"], st)           # X -> Xq + sf_x (on device)
    c["a_flat"][: M * I // 2] = c["xq_torch"].reshape(-1).view(torch.int8)  # Xq -> the fused main's A operand (byte reinterpret, no alloc)
    _refresh_sfa_device(c["sf_torch"].reshape(-1), c["perm"], c["sfa_torch"], g_buf=c["g_buf"])  # alloc-free SFA
    if c["enable_lora"]:
        c["down_call"](st)                                # SAME X, L2 -> D, into the fused main's D operand
    c["kb"](c["a_tensor"], c["b_tensor"], c["sfa_tensor"], c["sfb_tensor"],
            c["d_tensor"], c["l1_tensor"], c["c_tensor"], c["gs"], st)
    return c["c_torch"][:, :, 0]


def mm_fp4_svdquant(x, state, ext_xq=None, ext_sf=None):
    """The single public entry: run the fused NVFP4-SVDQuant linear (NVFP4 main GEMM + bf16 LoRA) for
    activation x (CUDA, [M, I]) using a state from prepare_svdquant_state. Graph-safe: NO
    torch.cuda.synchronize, .cpu(), .item(), or per-call CUDA allocation on this path.

    ext_xq/ext_sf (optional): externally-quantized NVFP4 activation for the residual main GEMM, to match
    a host framework's own quantizer (e.g. TRT-LLM/SGLang/vLLM fp4_quantize) for numerical parity. When
    given, the built-in on-device quantizer is skipped. ext_xq: uint8 [M, I//2] packed E2M1; ext_sf:
    uint8 PLAIN [M, I//16] E4M3. The bf16 LoRA down-projection still uses the staged x_hat (unchanged).

    Output aliasing: returns Y [M, O] bf16 as a VIEW into the state's reusable output storage. The caller
    must consume or clone it BEFORE the next mm_fp4_svdquant call on the SAME state (a subsequent call
    overwrites the storage), and must not host-read it before the stream completes / the graph replays."""
    c = state
    M, I = c["M"], c["I"]
    # Launch on the CURRENT stream (not a fixed default): under torch.cuda.graph this is the capture
    # stream, so the cute kernels are recorded + ordered with the per-call buffer-update torch ops; in
    # eager it is the default stream (unchanged behavior). This is what makes apply CUDA-graph-capturable.
    st = cutlass_torch.current_stream()
    if not x.is_cuda or tuple(x.shape) != (M, I):
        raise ValueError(f"mm_fp4_svdquant: activation must be a CUDA [{M}, {I}] tensor, "
                         f"got {tuple(x.shape)} on {x.device} (caller reshapes to [M, I] first)")
    if ext_xq is not None:
        # Injection path: the on-device quantizer (the only consumer of the f32 backing xf_back) is
        # SKIPPED, so staging x through f32 is pure overhead -- nsys: copy(x->f32) + mul(pqs) + copy(->bf16)
        # = ~44% of per-call GPU time at K=12288. Form the bf16 LoRA down-input x_hat = x*pqs DIRECTLY
        # (bf16, matching eager which also builds x_hat in bf16); the f32 backing is never touched (and may
        # be None when the state was built with external_quant=True to save the M*I*4 allocation).
        if c["enable_lora"]:
            xb = c["xb_t"]._svdquant_backing[:, :, 0]
            xb.copy_(x)
            if c["pqs"] is not None:
                xb.mul_(c["pqs"])
    else:
        xf_back = c["xf_t"]._svdquant_backing             # [M, I, 1] f32 (only the on-device quant path needs it)
        xf_back[:, :, 0].copy_(x)                         # new activation -> f32 backing (feeds on-device quant)
        if c["pqs"] is not None:
            xf_back[:, :, 0].mul_(c["pqs"])               # smoothing (pre_quant_scale), in place
        if c["enable_lora"]:
            c["xb_t"]._svdquant_backing[:, :, 0].copy_(xf_back[:, :, 0])   # SAME X -> bf16 down input
    return _run_kernels(c, st, ext_xq=ext_xq, ext_sf=ext_sf)


__all__ = [
    # --- module-level imports re-exported so the shims + consumers see them via `from ... import *` ---
    "os", "sys", "Type",
    "cuda", "torch",
    "cutlass", "cute", "utils", "pipeline", "cutlass_torch",
    "cpasync", "tcgen05", "OperandMajorMode", "sm100_utils", "blockscaled_utils",
    "E4M3_MAX", "_HERE",
    # --- inlined operand builders + SF MMA-layout converter (formerly nvfp4_gemm / blockscaled) ---
    "from_dlpack", "_ATOM_M", "_ATOM_K", "_ceil_div",
    "cvt_sf_MKL_to_M32x4xrm_K4xrk_L", "_cvt_sf_to_mma", "_make_sf_mma_tensor", "_fp4_from_packed",
    # --- front-end dtype aliases / constants ---
    "LORA_DTYPE", "ACC_DTYPE", "MMA_TILER_MN", "SKINNY_N_TILE", "LONG_K_M_TILE", "LONG_K_SPLIT",
    "LONG_K_THRESHOLD", "BF16_TILE_K", "SF_BLOCK", "FP4_E2M1_MAX", "QUANT_M_TILE",
    # --- fused-main dtype aliases / constants ---
    "ELEM_DTYPE", "SF_DTYPE", "OUT_DTYPE", "SF_VEC", "NVFP4_INST_K", "NVFP4_TILE_K",
    "MMA_TILER_MN_2CTA", "CTA_GROUP_2", "CLUSTER_SHAPE_MN_2CTA",
    # --- down-tile dispatch tables + selectors ---
    "_LONG_K_SPLIT_BY_IMR", "_long_k_split_for_m", "_LONG_K_I_ALLM", "_is_long_k",
    "_OCC_DOWN_SPLIT_BY_IOMR", "_is_occ_down", "_occ_down_split_for",
    "_AUTOTUNED_DOWN_BY_IOMR", "_is_autotuned_down", "_autotuned_down_for", "down_tile_for_shape",
    # --- kernel classes ---
    "ActivationQuantizer", "DownProjection", "SplitKReduce", "KernelB", "KernelB2CTA",
    # --- input builders ---
    "_kmajor_cuda_tensor", "_bf16_kmajor_tensor", "_f32_kmajor_tensor", "_u8_rowmajor_tensor",
    "assert_no_cpu_staging_in_input_builders",
    # --- compile caches + factories ---
    "_QUANT_CACHE", "_DOWN_CACHE", "_REDUCE_CACHE",
    "_compiled_quantizer", "_compiled_down", "_compiled_down_split", "_compiled_reduce",
    "_COMPILE_CACHE", "_compiled_kernel_b",
    # --- front-end runtime entries ---
    "run_quantizer", "run_down_projection", "run_kernel_a", "_provided_gscale_x", "_flux_shapes",
    # --- fused-main dispatch selectors ---
    "_WIDE_O_2CTA_ALLOWLIST", "_wide_o_2cta", "_resolve_use_2cta",
    "_NUM_AB_STAGE_BY_IOMR", "num_ab_stage_for_shape",
    # --- fused-main runtime entries ---
    "run_kernel_b",
    # --- on-device SF swizzle ---
    "_SF_PERM_CACHE", "_sf_swizzle_perm", "_refresh_sfa_device", "check_sf_swizzle",
    # --- deployable host fast-path ---
    "prepare_svdquant_state", "_run_kernels", "mm_fp4_svdquant",
]
