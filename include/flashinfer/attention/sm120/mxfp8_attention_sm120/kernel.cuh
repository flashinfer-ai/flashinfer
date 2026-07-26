/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
// S3 MXFP8/per-tensor-FP8 ragged prefill kernel for SM120/SM121 (consumer Blackwell).
// Warp-specialized producer/consumer with a persistent LPT tile scheduler; see
// tile_scheduler.cuh for the work-list contract the host launcher must build.
// clang-format off
#include <cmath>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>
#include <cute/atom/mma_atom.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/atom/copy_traits_sm90_tma.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/numeric_types.h>
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"

#include "mxfp8_mma.cuh"
#include "tile_scheduler.cuh"
// clang-format on

namespace flashinfer {
namespace mxfp8_attention_sm120 {

using namespace cute;
namespace mxfp8 = flashinfer::sm120_mxfp8;

using Element = cutlass::float_e4m3_t;
using ElementSF = cutlass::float_ue8m0_t;
#ifndef S3_HEAD_DIM
#define S3_HEAD_DIM 128
#endif
constexpr int kHeadDim = S3_HEAD_DIM, kBlockM = 128, kBlockN = 64, SFVecSize = 32, kStages = 2;
// SF atom is irreducibly 128-key-aligned (Blk_MN=128, a TMEM-format leftover; see memory
// sf-128-atom-is-tmem-artifact). So the DATA tile is kBlockN (=64, the register-win lever) but
// the SF tiles stay 128-key: each 64-key data block loads the 128-key SF atom containing it
// (redundant for the 2 sibling 64-blocks, cheap since SF is bytes) and the consumer indexes its
// (nb&1) 64-key half. kSFBlockN decouples the SF key-tile from the data key-tile.
constexpr int kSFBlockN = cute::max(int(kBlockN), 128);  // = 128
// This is the real-64 path: the consumer slices the 128-key SF atom to the (nb&1) 64-key half
// (subSFK / tOrSFV index / kNCol = kBlockN/4). That slicing is correct ONLY for kBlockN==64 (data
// tile = exactly half the SF atom). The kBlockN=128 no-slicing path was master through commit
// 221c24c; it's preserved in git history (recover there if a selectable 128 path is ever wanted).
static_assert(kBlockN == 64, "real-64 SF-128/data-64 half-slicing assumes kBlockN==64");
// The block-scaled SF smem atom + TMA box are inherently 128 along the contraction
// (cutlass Blk_SF granularity); the gmem SF layout pads K to 128 for head_dim < 128
// too. So all SF tiles use kSFPadHD=128 while the DATA path uses the real kHeadDim
// (the QK MMA contracts only kHeadDim, reading the first kHeadDim/32 of the 4 SF blocks).
constexpr int kSFPadHD = (kHeadDim < 128 ? 128 : kHeadDim);
// P is the softmax output: post-max-subtraction it is bounded in [0,1] with the row argmax
// exactly 1.0, so a FIXED scale 256.0 (se=-8) maps it into e4m3's normal range and NEVER
// saturates (1.0*256=256 <= 448). Dropping the per-32-block amax/quad_reduce + the SF
// smem store/gather: tOrSFP becomes the compile-time constant byte (-8+127)=119. The dynamic
// per-block path is kept behind this flag only so the oracle can A/B the precision delta.
#ifndef S3_P_DYNAMIC_SCALE
#define S3_P_DYNAMIC_SCALE 0
#endif
// Decoupled so the bench can isolate the two costs separately: (a) the per-block
// amax/quad_reduce, (b) the SF smem store+gather. kPConstSF=1 (fill tOrSFP with the
// constant byte, no smem) is only valid when the scale is fixed; a dynamic scale
// MUST transit smem. Default: fixed scale -> const SF.
#ifndef S3_P_CONST_SF
#define S3_P_CONST_SF (S3_P_DYNAMIC_SCALE ? 0 : 1)
#endif
constexpr bool kPDynamicScale = (S3_P_DYNAMIC_SCALE != 0);
constexpr bool kPConstSF = (S3_P_CONST_SF != 0);
constexpr int kPScaleExp = -8;  // fixed: scale = 2^-se = 256.0
// S5: fill the PV-A operand (tOrP) directly from the QK accumulator via an intra-quad
// __shfl, dropping the S2 smem-transpose round-trip (2 NamedBarriers + sP write + ldmatrix
// read, per n_block). Valid only with a fixed P scale (kPConstSF): the SF is then a
// compile-time constant so ONLY the e4m3 data shuffles (no SF shuffle, unlike SageAttention's
// dynamic-nvfp4 path). Set S3_P_SMEM=1 to fall back to the smem oracle path (the bit-exact
// reference, and the only path the dynamic-scale A/B can use since a dynamic SF must transit smem).
#ifndef S3_P_SMEM
#define S3_P_SMEM 0
#endif

// S6a (serving): partial last KV block "kFillZero" for V. When kv_len is not a
// multiple of kBlockN (the common case once prefix/chunked masking is on), the
// padded keys [valid, kBlockN) of the last block are loaded into smem as whatever
// the producer's buffer holds. The QK mask sets their P=0, but PV still reads V +
// V-SF for ALL keys: a NaN byte in the padded V data (e4m3 0x7F/0xFF) OR its SF
// (ue8m0 0xFF) makes 0*NaN=NaN, poisoning accO. K is immune (masked to -inf BEFORE
// softmax); V is not -- the same K/V asymmetry FlashInfer's NVFP4 prefill fixes with
// a kFillZero V-SF load. We can't predicate a TMA load per-row, so we sanitize in
// smem/registers: zero the padded V DATA columns in sV (covers the straddling block's
// masked keys, whose 32-block SF is real/finite for the valid keys it shares) and
// finite-ize tOrSFV for the fully-masked 32-key tiles (whose SF may be garbage NaN).
// Set 0 to reproduce the 0*NaN poison (the adversarial test injects 0xFF pad).
#ifndef S3_V_KFILLZERO
#define S3_V_KFILLZERO 1
#endif

// S6b (KV dtype contract): SFSource selects WHERE the block-scale factors come from.
//  - kMxFp8 (default): real per-32 ue8m0 SF streamed via TMA from a block-scaled source
//    (what s3_e2e + the torchao oracle feed). The full block-scaled MMA. Bit-exact unchanged.
//  - kUniformFp8: a mainstream PER-TENSOR fp8 cache (torch.float8_e4m3fn + scalar q/k/v_scale,
//    as FA3 / FlashInfer-fp8 / SageAttention store it). e4m3 bytes are IDENTICAL to ours, so the
//    DATA path is untouched; only the scale differs. A per-tensor scalar == a block-scaled tensor
//    with all block exponents EQUAL, so we set every SF block to the uniform byte 127 (2^0 = 1.0)
//    -> the block-scaled MMA degenerates to a plain fp8 MMA at zero extra cost, and the scalars
//    fold OUT of the cache: q_scale*k_scale into sm_scale (host) and v_scale into o_scale (the PV
//    epilogue). The SF TMA loads are SKIPPED ENTIRELY (the cache carries NO scale factors). This
//    makes block-scaled mxfp8 a strict SUPERSET of per-tensor fp8 -- one kernel eats both.
//  [SF-bytes contract] SKIPPING the SF TMA means transaction_bytes MUST also drop the SF
//  contribution (TmaData* without TmaSFBytes*), or the TMA-arrival barrier waits forever for bytes
//  that never land -> a producer/consumer DEADLOCK (100% GPU hang, no output), the SAME failure
//  class as the n_block_max contract. The two are paired: skip-the-load and shrink-the-byte-count
//  live together under `if constexpr (kLoadSF)` / the transaction_bytes ternary -- edit together.
enum class SFSource : int { kMxFp8 = 0, kUniformFp8 = 1 };
constexpr uint8_t kUniformSFByte = 127;  // ue8m0 for 2^0 = 1.0 (biased exp 0 + 127)

constexpr int NBLK = kHeadDim / SFVecSize;             // SF blocks along head_dim (QK contraction)
constexpr int NKB = kBlockN / SFVecSize;               // SF blocks along keys (PV contraction) = 4
constexpr int kNWarps = 12, kNThreads = kNWarps * 32;  // 384
constexpr int NumMmaThreads = 256, NumCopyThreads = 128;
constexpr float kLog2e = 1.4426950408889634f;
constexpr int kQuantBarrier = 0;  // named barrier id for P-smem handoff
constexpr int kVFillBarrier = 1;  // named barrier id for partial-block V kFillZero

using AtomMXF8 =
    cute::SM120::BLOCKSCALED::SM120_16x8x32_TN_VS<Element, Element, float, ElementSF, SFVecSize>;
using TileShape_MNK = Shape<Int<kBlockM>, Int<kBlockN>, Int<kHeadDim>>;
using TiledMmaQK = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{},
                                           Tile<_128, _32, Int<kHeadDim>>{}));
// PV: O[M=q, N=head_dim] = P[q, K=keys] * Vt[head_dim, K=keys]. TileK = kBlockN.
using TiledMmaPV = decltype(make_tiled_mma(AtomMXF8{}, Layout<Shape<_8, _1, _1>>{},
                                           Tile<_128, _32, Int<kBlockN>>{}));

namespace ccd = cutlass::gemm::collective::detail;
using SmemLayoutAtomQ = decltype(ccd::sm120_rr_smem_selector<Element, Int<kHeadDim>>());
using SmemLayoutQ = decltype(tile_to_shape(SmemLayoutAtomQ{}, select<0, 2>(TileShape_MNK{})));
// K rides a kStages ring (keys x head_dim x stage).
using SmemLayoutK = decltype(tile_to_shape(
    SmemLayoutAtomQ{}, make_shape(Int<kBlockN>{}, Int<kHeadDim>{}, Int<kStages>{})));
// P-smem transpose buffer [q, key] (single, per-nb scratch) and V [head_dim, keys].
// V stays depth-1 (the ring depth is a perf knob, orthogonal to S3's online-O goal):
// K(2 stages)+V(2 stages) = 96KB data > the 99KB sm120 opt-in smem limit.
using SmemLayoutAtomKeys = decltype(ccd::sm120_rr_smem_selector<Element, Int<kBlockN>>());
using SmemLayoutP =
    decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kBlockM>{}, Int<kBlockN>{})));
using SmemLayoutVt =
    decltype(tile_to_shape(SmemLayoutAtomKeys{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{})));

// Canonical block-scaled SF smem atom (128x128 operand; MMA_NSF=1). Reused for SFQ,
// and (head_dim==kBlockN) for the K and V rings.
using BlkSF = cutlass::detail::Sm1xxBlockScaledConfig<SFVecSize>;
static constexpr int MMA_NSF = size<2>(typename TiledMmaQK::AtomShape_MNK{}) / SFVecSize;
using Blk_MN = typename BlkSF::Blk_MN;
using Blk_SF = typename BlkSF::Blk_SF;
using Blk_Elems = decltype(Blk_MN{} * Blk_SF{});
using mnBasicBlockShape = Shape<_32, _4>;
using mnBasicBlockStride = Stride<_16, _4>;
using kBasicBlockShape = Shape<Int<SFVecSize>, Int<MMA_NSF>>;
using kBasicBlockStride = Stride<_0, _1>;
using sSF_strideMN = decltype(prepend(Blk_Elems{}, mnBasicBlockStride{}));
using sSF_shapeK = decltype(prepend(
    make_shape(Blk_SF{} / Int<MMA_NSF>{}, Int<kSFPadHD>{} / Int<SFVecSize>{} / Blk_SF{}),
    kBasicBlockShape{}));
using sSFA_shapeM = decltype(prepend(Int<kBlockM>{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFA_strideK = decltype(prepend(
    make_stride(Int<MMA_NSF>{}, Int<kBlockM>{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutSFQ = decltype(make_layout(make_shape(sSFA_shapeM{}, sSF_shapeK{}),
                                           make_stride(sSF_strideMN{}, sSFA_strideK{})));
using sSFBTileShape_N = Int<kSFBlockN>;
using sSFB_shapeN = decltype(prepend(sSFBTileShape_N{} / Blk_MN{}, mnBasicBlockShape{}));
using sSFB_strideK = decltype(prepend(
    make_stride(Int<MMA_NSF>{}, sSFBTileShape_N{} / Blk_MN{} * Blk_Elems{}), kBasicBlockStride{}));
using SmemLayoutAtomSFK = decltype(make_layout(make_shape(sSFB_shapeN{}, sSF_shapeK{}),
                                               make_stride(sSF_strideMN{}, sSFB_strideK{})));
// add stage dim (S1 pattern).
using SmemLayoutSFK = decltype(make_layout(
    append(shape(SmemLayoutAtomSFK{}), Int<kStages>{}),
    append(stride(SmemLayoutAtomSFK{}), size(filter_zeros(SmemLayoutAtomSFK{})))));
using SmemLayoutSFV = SmemLayoutAtomSFK;  // V depth-1

using SmemCopyAtomData = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
using SmemCopyAtomSF = Copy_Atom<UniversalCopy<ElementSF>, ElementSF>;

// S6a-2 (GQA): SF gmem layouts carry a trailing head (L) mode -- tile_atom_to_shape_SF*
// takes a 4D problem (M,N,K,L) -> rank-3 SF layout (M,K,L) / (N,K,L). L=1 (single head)
// is byte-identical to the old rank-2 form (the L slab is a single tile), so dense stays
// bit-exact; multi-head selects L=head before local_tile.
using LayoutSF = decltype(BlkSF::tile_atom_to_shape_SFA(
    make_shape(int(kBlockM), int(kBlockN), int(kSFPadHD), int(1))));
// V is the PV B operand [head_dim, keys], block-scaled along KEYS -> SFB layout (head_dim, keys),
// NOT K's SFA (keys, head_dim). Per-block type for the TMA descriptor; the full
// (head_dim x seqlen_k) runtime layout has the SAME C++ type (all extents dynamic int).
using LayoutSFV = decltype(BlkSF::tile_atom_to_shape_SFB(
    make_shape(int(kBlockM), int(kHeadDim), int(kSFBlockN), int(1))));

// S6a-2 (GQA): the data TMA tensors gain a trailing head mode (token-major production
// layout [token, head, head_dim]: head_dim is stride-1 (inside the box), head is a
// non-tiled batch coordinate sliced before local_tile). Stride VALUES are irrelevant to
// the descriptor TYPE (all dynamic int), only rank/staticness must match the host build.
using TMA_Q =
    decltype(make_tma_copy(SM90_TMA_LOAD{},
                           make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                       make_shape(int(kBlockM), int(kHeadDim), int(1)),
                                       make_stride(int(kHeadDim), _1{}, int(kHeadDim))),
                           SmemLayoutQ{}, select<0, 2>(TileShape_MNK{}), _1{}));
using TMA_K =
    decltype(make_tma_copy(SM90_TMA_LOAD{},
                           make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                                       make_shape(int(8 * kBlockN), int(kHeadDim), int(1)),
                                       make_stride(int(kHeadDim), _1{}, int(kHeadDim))),
                           SmemLayoutK{}(_, _, _0{}), select<1, 2>(TileShape_MNK{}), _1{}));
using TMA_V = decltype(make_tma_copy(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<Element const*>(nullptr)),
                make_shape(int(kHeadDim), int(8 * kBlockN), int(1)),
                make_stride(int(8 * kBlockN), _1{}, int(kHeadDim * 8 * kBlockN))),
    SmemLayoutVt{}, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}), _1{}));
using TMA_SFQ = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFQ{}, make_shape(Int<kBlockM>{}, Int<kSFPadHD>{}), _1{}));
using TMA_SFK = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{}, make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSF{}),
    SmemLayoutSFK{}(_, _, _0{}), make_shape(Int<kSFBlockN>{}, Int<kSFPadHD>{}), _1{}));
using TMA_SFV = decltype(make_tma_copy<uint16_t>(
    SM90_TMA_LOAD{},
    make_tensor(make_gmem_ptr(static_cast<ElementSF const*>(nullptr)), LayoutSFV{}),
    SmemLayoutSFV{}, make_shape(Int<kSFPadHD>{}, Int<kSFBlockN>{}), _1{}));

using PipeQ = cutlass::PipelineTmaAsync<1>;
using PipeK = cutlass::PipelineTmaAsync<kStages>;
using PipeV = cutlass::PipelineTmaAsync<1>;
using StateQ = cutlass::PipelineState<1>;
using StateK = cutlass::PipelineState<kStages>;
using StateV = cutlass::PipelineState<1>;

struct Params {
  TMA_Q tma_q;
  TMA_K tma_k;
  TMA_V tma_v;
  TMA_SFQ tma_sfq;
  TMA_SFK tma_sfk;
  TMA_SFV tma_sfv;
  LayoutSF layout_sfq;   // seqlen_q x 128 (Q spans all m_blocks)
  LayoutSFV layout_sfv;  // head_dim x seqlen_k (V spans all n_blocks, scale along keys)
  int seqlen_q, seqlen_k, n_block_total;
  int num_qo_heads, num_kv_heads;  // GQA: Q indexes qo_head, K/V index kv_head=qo_head/group_size
                                   // (1,1 = single-head dense)
  int const* tile_kv_len;  // OPTIONAL [num_q_tiles]: per-q-tile key count (variable-cost / varlen
                           // proxy); nullptr -> seqlen_k
  float sm_scale;
  // S6b kUniformFp8: per-tensor v_scale folded into the PV output (O *= o_scale). Defaults to 1.0
  // so the kMxFp8 path (and every existing call site) is bit-exact. q_scale*k_scale fold into
  // sm_scale on the host (the score path needs no kernel change -- it already multiplies sm_scale).
  float o_scale = 1.0f;
  float* out_O;     // token-major [seqlen_q, head_dim, num_qo_heads]: O(q,hd,h) at
                    // q*num_qo_heads*head_dim + h*head_dim + hd
  float* out_lse;   // head-major [num_qo_heads, seqlen_q]: lse(h,q) at h*seqlen_q + q
  float* out_l;     // head-major [num_qo_heads, seqlen_q] row_sum (cross-check)
  float* out_Ppre;  // [seqlen_q, seqlen_k] device pre-quant fp32 P (host re-quantizes the SAME P)
  float* out_Mnb;   // [seqlen_q, n_block_total] running max after each block (-inf if skipped)
  float* out_dbg;   // [seqlen_q, seqlen_k] OPTIONAL (nullptr to skip): device dequantized requant-P
};

struct SharedStorage {
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutQ>> sQ;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutK>> sK;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutVt>> sV;
  alignas(1024) cute::ArrayEngine<Element, cute::cosize_v<SmemLayoutP>> sP;
  alignas(128) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFQ>> sSFQ;
  alignas(128) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFK>> sSFK;
  alignas(128) cute::ArrayEngine<ElementSF, cute::cosize_v<SmemLayoutSFV>> sSFV;
  alignas(128) ElementSF sSFP[kBlockM * NKB];  // simple [q_local][keyblock]
  struct {
    alignas(16) PipeQ::SharedStorage pipeline_q;
    alignas(16) PipeK::SharedStorage pipeline_k;
    alignas(16) PipeV::SharedStorage pipeline_v;
  };
};

// Split data vs SF so the kUniformFp8 SFSource (no SF TMA) can drop the SF contribution from
// transaction_bytes -- the [SF-bytes contract] above. TmaBytes* (data+SF) keep the kMxFp8 path
// byte-identical; the kernel picks data-only when !kLoadSF.
static constexpr uint32_t TmaDataQ = cute::cosize_v<SmemLayoutQ> * sizeof(Element);
static constexpr uint32_t TmaDataK = cute::cosize_v<SmemLayoutK> / kStages * sizeof(Element);
static constexpr uint32_t TmaDataV = cute::cosize_v<SmemLayoutVt> * sizeof(Element);
static constexpr uint32_t TmaSFBytesQ = cute::cosize_v<SmemLayoutSFQ> * sizeof(ElementSF);
static constexpr uint32_t TmaSFBytesK = cute::cosize_v<SmemLayoutSFK> / kStages * sizeof(ElementSF);
static constexpr uint32_t TmaSFBytesV = cute::cosize_v<SmemLayoutSFV> * sizeof(ElementSF);
static constexpr uint32_t TmaBytesQ = TmaDataQ + TmaSFBytesQ;
static constexpr uint32_t TmaBytesK = TmaDataK + TmaSFBytesK;
static constexpr uint32_t TmaBytesV = TmaDataV + TmaSFBytesV;

// e4m3 + ue8m0 quant given the block's ue8m0 scale-exponent.
__device__ __forceinline__ Element quant_e4m3(float v, int scale_exp) {
  return Element(v * exp2f(float(-scale_exp)));
}
// OCP-MX scale exponent for an e4m3 block: floor(log2(amax)) - 8 (e4m3 emax=8), bit-extracted.
__host__ __device__ __forceinline__ int mx_scale_exp_bits(uint32_t b) {
  int e = int((b >> 23) & 0xFF) - 127;
  int se = e - 8;
  return se < -127 ? -127 : (se > 127 ? 127 : se);
}
__device__ __forceinline__ int mx_scale_exp(float amax) {
  if (!(amax > 0.f)) return -127;
  return mx_scale_exp_bits(__float_as_uint(amax));
}

template <class Op>
__device__ __forceinline__ float quad_reduce(float v, Op op) {
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 2));
  v = op(v, __shfl_xor_sync(uint32_t(-1), v, 1));
  return v;
}

template <typename TileScheduler, bool Causal, SFSource Src = SFSource::kMxFp8>
__global__ void __launch_bounds__(kNThreads, 1)
    s3_kernel(CUTE_GRID_CONSTANT Params const params,
              CUTE_GRID_CONSTANT typename TileScheduler::Params const sched_params) {
  extern __shared__ char smem_raw[];
  auto& ss = *reinterpret_cast<SharedStorage*>(smem_raw);
  // kMxFp8 -> stream real per-block SF via TMA; kUniformFp8 -> synthesize a constant byte 127 in
  // registers and SKIP every SF TMA (paired with the data-only transaction_bytes below).
  constexpr bool kLoadSF = (Src == SFSource::kMxFp8);

  int const wg = cutlass::canonical_warp_group_idx();  // 0=producer
  int const warp_in_wg = cutlass::canonical_warp_idx_sync() % 4;
  int const elect = cute::elect_one_sync();

  PipeQ::Params pq;
  pq.role = (wg == 0) ? PipeQ::ThreadCategory::Producer : PipeQ::ThreadCategory::Consumer;
  pq.is_leader = (threadIdx.x % cutlass::NumThreadsPerWarpGroup == 0);
  pq.num_consumers = NumMmaThreads;
  // [SF-bytes contract] data+SF when kLoadSF, data-only otherwise -- MUST match the SF TMA copies
  // skipped under `if constexpr (kLoadSF)` in the producer, or the barrier hangs (see SFSource).
  pq.transaction_bytes = kLoadSF ? TmaBytesQ : TmaDataQ;
  PipeQ pipeline_q(ss.pipeline_q, pq, Shape<_1, _1, _1>{});
  PipeK::Params pk;
  pk.role = (wg == 0) ? PipeK::ThreadCategory::Producer : PipeK::ThreadCategory::Consumer;
  pk.is_leader = pq.is_leader;
  pk.num_consumers = NumMmaThreads;
  pk.transaction_bytes = kLoadSF ? TmaBytesK : TmaDataK;
  PipeK pipeline_k(ss.pipeline_k, pk, Shape<_1, _1, _1>{});
  PipeV::Params pv;
  pv.role = (wg == 0) ? PipeV::ThreadCategory::Producer : PipeV::ThreadCategory::Consumer;
  pv.is_leader = pq.is_leader;
  pv.num_consumers = NumMmaThreads;
  pv.transaction_bytes = kLoadSF ? TmaBytesV : TmaDataV;
  PipeV pipeline_v(ss.pipeline_v, pv, Shape<_1, _1, _1>{});
  __syncthreads();

  Tensor sQ = make_tensor(make_smem_ptr(ss.sQ.begin()), SmemLayoutQ{});
  Tensor sK = make_tensor(make_smem_ptr(ss.sK.begin()), SmemLayoutK{});
  Tensor sV = make_tensor(make_smem_ptr(ss.sV.begin()), SmemLayoutVt{});
  Tensor sP = make_tensor(make_smem_ptr(ss.sP.begin()), SmemLayoutP{});
  Tensor sSFQ = make_tensor(make_smem_ptr(ss.sSFQ.begin()), SmemLayoutSFQ{});
  Tensor sSFK = make_tensor(make_smem_ptr(ss.sSFK.begin()), SmemLayoutSFK{});
  Tensor sSFV = make_tensor(make_smem_ptr(ss.sSFV.begin()), SmemLayoutSFV{});

  int const n_block_total = params.n_block_total;
  float const sm_scale = params.sm_scale;
  float const sm_scale_log2 = sm_scale * kLog2e;
  TileScheduler scheduler;

  if (wg == 0) {
    // -------- producer --------
    cutlass::arch::warpgroup_reg_dealloc<24>();
    if (warp_in_wg == 0 && elect) {
      // 3D coordinate tensors (..., head): the trailing head mode is sliced per work-item
      // below (Q/SFQ by qo_head, K/V/SFK/SFV by kv_head). SFK's full nominal shape is built
      // in-kernel from seqlen_k+num_kv_heads (tile_atom_to_shape_SFA is CUTE_HOST_DEVICE) so
      // the n-block index stays WITHIN nominal -- matches FlashInfer, no out-of-nominal
      // arithmetic. Descriptors (host, over the full/packed tensors) are unchanged.
      Tensor mQ3d = params.tma_q.get_tma_tensor(
          make_shape(int(params.seqlen_q), int(kHeadDim), int(params.num_qo_heads)));
      Tensor mK3d = params.tma_k.get_tma_tensor(
          make_shape(int(params.seqlen_k), int(kHeadDim), int(params.num_kv_heads)));
      Tensor mV3d = params.tma_v.get_tma_tensor(
          make_shape(int(kHeadDim), int(params.seqlen_k), int(params.num_kv_heads)));
      Tensor mSFQ3d = params.tma_sfq.get_tma_tensor(shape(params.layout_sfq));
      Tensor mSFK3d = params.tma_sfk.get_tma_tensor(shape(BlkSF::tile_atom_to_shape_SFA(make_shape(
          int(params.seqlen_k), int(kBlockN), int(kHeadDim), int(params.num_kv_heads)))));
      Tensor mSFV3d = params.tma_sfv.get_tma_tensor(shape(params.layout_sfv));
      auto bq = params.tma_q.get_slice(_0{});
      auto bsq = params.tma_sfq.get_slice(_0{});
      auto bk = params.tma_k.get_slice(_0{});
      auto bsk = params.tma_sfk.get_slice(_0{});
      auto bv = params.tma_v.get_slice(_0{});
      auto bsv = params.tma_sfv.get_slice(_0{});
      StateQ wq = cutlass::make_producer_start_state<PipeQ>();
      StateK wk = cutlass::make_producer_start_state<PipeK>();
      StateV wv = cutlass::make_producer_start_state<PipeV>();

      for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
           work = scheduler.get_next_work(sched_params, work)) {
        // S6a: ragged work-tuple. Dense schedulers return qo_indptr=kv_indptr=0 and
        // qo_len/kv_len = the single problem's lengths, so this degenerates bit-exactly.
        // Every request is 128-padded along its sequence, so qo_indptr/kv_indptr are
        // multiples of the tile (one request-tile-base offset addresses data AND blocked SF).
        auto const bc = work.get_block_coord(sched_params);
        int const q_tile_local = get<0>(bc);
        int const qo_head_idx = get<1>(bc), kv_head_idx = get<2>(bc);
        int const qo_indptr = get<3>(bc), kv_indptr = get<4>(bc), qo_len = get<5>(bc),
                  kv_len = get<6>(bc);
        int const q_tile_global = qo_indptr / kBlockM + q_tile_local;
        int const kv_tile_base = kv_indptr / kBlockN;
        int const nb_tile = params.tile_kv_len
                                ? (params.tile_kv_len[q_tile_local] + kBlockN - 1) / kBlockN
                                : (kv_len + kBlockN - 1) / kBlockN;
        // [n_block_max contract] PRODUCER copy -- this is the TMA-load trip count and it MUST be
        // byte-identical to the CONSUMER's n_block_max (grep "[n_block_max contract]"; the other
        // site is in the consumer warpgroup). If they diverge the kernel DEADLOCKS, it does not
        // misanswer: the consumer waits on K/V tiles the producer never loads (or vice versa) and
        // both warpgroups block forever (symptom = 100% GPU util, no output/error). offset_q =
        // kv_len - qo_len is the slice-3 append/causal shift; edit BOTH sites together. See
        // docs/gotcha.md "A causal-mask change in a warp-specialized kernel is TWO edits".
        int const offset_q = kv_len - qo_len;
        int const n_block_max =
            Causal ? cute::min(nb_tile,
                               ((q_tile_local + 1) * kBlockM + offset_q + kBlockN - 1) / kBlockN)
                   : nb_tile;

        // GQA: slice the head mode (Q/SFQ by qo_head, K/V/SFK/SFV by kv_head). Single-head
        // dense passes qo_head=kv_head=0 (the only slice) -> identical addressing.
        Tensor mQ = mQ3d(_, _, qo_head_idx);
        Tensor mSFQ = mSFQ3d(_, _, qo_head_idx);
        Tensor mK = mK3d(_, _, kv_head_idx);
        Tensor mSFK = mSFK3d(_, _, kv_head_idx);
        Tensor mV = mV3d(_, _, kv_head_idx);
        Tensor mSFV = mSFV3d(_, _, kv_head_idx);

        Tensor gQ = local_tile(mQ, select<0, 2>(TileShape_MNK{}), make_coord(q_tile_global, _0{}));
        Tensor gSFQ =
            local_tile(mSFQ, select<0, 2>(TileShape_MNK{}), make_coord(q_tile_global, _0{}));
        Tensor gK = local_tile(mK, select<1, 2>(TileShape_MNK{}), make_coord(_, _0{}));  // (N,K,nb)
        Tensor gSFK = local_tile(mSFK, make_shape(Int<kSFBlockN>{}, Int<kHeadDim>{}),
                                 make_coord(_, _0{}));  // SF tiled by 128-atom
        Tensor gV = local_tile(mV, make_shape(Int<kHeadDim>{}, Int<kBlockN>{}),
                               make_coord(_0{}, _));  // (hd,N,nb)
        Tensor gSFV = local_tile(mSFV, make_shape(Int<kHeadDim>{}, Int<kSFBlockN>{}),
                                 make_coord(_0{}, _));  // SF tiled by 128-atom
        Tensor tKgK = group_modes<0, 3>(bk.partition_S(gK));
        Tensor tKsK = group_modes<0, 3>(bk.partition_D(sK));
        Tensor tKgSFK = group_modes<0, 3>(bsk.partition_S(gSFK));
        Tensor tKsSFK = group_modes<0, 3>(bsk.partition_D(sSFK));
        Tensor tVgV = group_modes<0, 3>(bv.partition_S(gV));
        Tensor tVsV = group_modes<0, 3>(bv.partition_D(sV));  // V depth-1: dest has no stage
        Tensor tVgSFV = group_modes<0, 3>(bsv.partition_S(gSFV));
        Tensor tVsSFV = group_modes<0, 3>(bsv.partition_D(sSFV));

        pipeline_q.producer_acquire(wq);
        copy(params.tma_q.with(*pipeline_q.producer_get_barrier(wq), 0), bq.partition_S(gQ),
             bq.partition_D(sQ));
        // [SF-bytes contract] SF TMAs skipped for kUniformFp8 (per-tensor cache has no SF); the
        // consumer synthesizes byte 127 and transaction_bytes above already dropped these bytes.
        if constexpr (kLoadSF)
          copy(params.tma_sfq.with(*pipeline_q.producer_get_barrier(wq), 0), bsq.partition_S(gSFQ),
               bsq.partition_D(sSFQ));
        ++wq;
        for (int nb = 0; nb < n_block_max; ++nb) {
          pipeline_k.producer_acquire(wk);
          int const nbg = kv_tile_base + nb;  // global 64-key DATA-tile index into the packed K/V
          int const sfg =
              nbg / 2;  // 128-key SF-atom index containing this data block (requests 128-padded)
          copy(params.tma_k.with(*pipeline_k.producer_get_barrier(wk), 0), tKgK(_, nbg),
               tKsK(_, wk.index()));
          if constexpr (kLoadSF)
            copy(params.tma_sfk.with(*pipeline_k.producer_get_barrier(wk), 0), tKgSFK(_, sfg),
                 tKsSFK(_, wk.index()));
          ++wk;
          pipeline_v.producer_acquire(wv);
          copy(params.tma_v.with(*pipeline_v.producer_get_barrier(wv), 0), tVgV(_, nbg), tVsV);
          if constexpr (kLoadSF)
            copy(params.tma_sfv.with(*pipeline_v.producer_get_barrier(wv), 0), tVgSFV(_, sfg),
                 tVsSFV);
          ++wv;
        }
      }
    }
  } else {
    // -------- consumers --------
    cutlass::arch::warpgroup_reg_alloc<232>();
    int const tid = threadIdx.x - NumCopyThreads;  // 0..255
    int const warp = tid / 32, lane = tid % 32;
    TiledMmaQK mma_qk;
    TiledMmaPV mma_pv;
    auto thr_qk = mma_qk.get_thread_slice(tid);
    auto thr_pv = mma_pv.get_thread_slice(tid);

    // QK operands
    Tensor tSrQ = thr_qk.partition_fragment_A(sQ);
    Tensor tSrK = thr_qk.partition_fragment_B(sK(_, _, _0{}));
    Tensor tSrSFQ = mxfp8::partition_fragment_SFA(sSFQ, thr_qk);
    Tensor tSrSFK = mxfp8::partition_fragment_SFB(sSFK(_, _, _0{}), thr_qk);
    auto scQ = make_tiled_copy_A(SmemCopyAtomData{}, mma_qk);
    auto tscQ = scQ.get_thread_slice(tid);
    auto scK = make_tiled_copy_B(SmemCopyAtomData{}, mma_qk);
    auto tscK = scK.get_thread_slice(tid);
    auto ts_qk = tile_shape(mma_qk);
    auto scSFQ = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFA_TV(mma_qk),
                                      make_shape(size<0>(ts_qk), size<2>(ts_qk)));
    auto scSFK = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_qk),
                                      make_shape(size<1>(ts_qk), size<2>(ts_qk)));
    auto tscSFQ = scSFQ.get_thread_slice(tid);
    auto tscSFK = scSFK.get_thread_slice(tid);

    // PV operands
    Tensor tOrP = thr_pv.partition_fragment_A(sP);
    Tensor tOrV = thr_pv.partition_fragment_B(sV);
    Tensor tOrSFP = mxfp8::partition_fragment_SFA(sSFQ, thr_pv);  // canonical SFA layout (cosize-1)
    Tensor tOrSFV = mxfp8::partition_fragment_SFB(sSFV, thr_pv);
    auto scP = make_tiled_copy_A(SmemCopyAtomData{}, mma_pv);
    auto tscP = scP.get_thread_slice(tid);
    auto scV = make_tiled_copy_B(SmemCopyAtomData{}, mma_pv);
    auto tscV = scV.get_thread_slice(tid);
    auto ts_pv = tile_shape(mma_pv);
    auto scSFV = make_tiled_copy_impl(SmemCopyAtomSF{}, mxfp8::get_layoutSFB_TV(mma_pv),
                                      make_shape(size<1>(ts_pv), size<2>(ts_pv)));
    auto tscSFV = scSFV.get_thread_slice(tid);
    Tensor sfp_coord = mxfp8::partition_SFA(
        make_identity_tensor(make_shape(Int<kBlockM>{}, Int<kBlockN>{})), thr_pv);

    // real-64: the SF stays 128-key resident (atom-aligned), the DATA is 64. Slice the QK-B SF
    // register fragment (mode-1 = (4,4)) to the 64-key half h = nb&1: split the inner 4 into (2,2),
    // fix the trailing coord to h, regroup -> (4,2)=8 keys matching the 64-key data operand.
    auto subSFK = [](auto const& f,
                     auto h) {  // h: cute::Int<half> -> STATIC slice offset (S9: no local demotion)
      auto m1 = get<1>(f.layout());
      auto a = get<0>(m1);
      auto b = get<1>(m1);
      auto nb = shape(b);
      auto sb = stride(b);
      auto t = make_tensor(
          f.data(),
          make_layout(get<0>(f.layout()),
                      make_layout(make_shape(shape(a), make_shape(nb / _2{}, _2{})),
                                  make_stride(stride(a), make_stride(sb, sb * (nb / _2{})))),
                      get<2>(f.layout())))(_, make_coord(_, make_coord(_, h)), _);
      return group_modes<1, 3>(t);
    };
    auto max_op = [](float a, float b) { return fmaxf(a, b); };
    auto add_op = [](float a, float b) { return a + b; };

    // token-major [seqlen_q, head_dim, num_qo_heads]; sliced by qo_head per work-item below.
    Tensor mO3d = make_tensor(
        make_gmem_ptr(params.out_O),
        make_layout(make_shape(int(params.seqlen_q), int(kHeadDim), int(params.num_qo_heads)),
                    make_stride(int(params.num_qo_heads * kHeadDim), _1{}, int(kHeadDim))));
    Tensor mPpre = make_tensor(make_gmem_ptr(params.out_Ppre),
                               make_layout(make_shape(int(params.seqlen_q), int(params.seqlen_k)),
                                           make_stride(int(params.seqlen_k), _1{})));

    StateQ rq;
    StateK rk;
    StateV rv;
    for (auto work = scheduler.get_initial_work(sched_params); work.is_valid(sched_params);
         work = scheduler.get_next_work(sched_params, work)) {
      auto const bc = work.get_block_coord(sched_params);
      int const q_tile_local = get<0>(bc);
      int const qo_head_idx = get<1>(bc);
      int const qo_indptr = get<3>(bc), kv_indptr = get<4>(bc), qo_len = get<5>(bc),
                kv_len = get<6>(bc);
      int const q_tile_global = qo_indptr / kBlockM + q_tile_local;
      int const kv_tile_base = kv_indptr / kBlockN;
      Tensor mO = mO3d(_, _, qo_head_idx);  // GQA: this work-item's output head
      int const nb_tile = params.tile_kv_len
                              ? (params.tile_kv_len[q_tile_local] + kBlockN - 1) / kBlockN
                              : (kv_len + kBlockN - 1) / kBlockN;
      // slice-3: append/decode-at-end causal. offset_q = kv_len - qo_len >= 0 (FlashInfer
      // convention: the qo_len queries sit at the END of the kv_len keys, so request-local
      // query m attends keys [0, m + offset_q]). qo_len==kv_len (slice-1/2) -> offset_q=0,
      // both formulas below reduce EXACTLY to the old ones (dense/ragged stay bit-exact).
      // [n_block_max contract] CONSUMER copy -- MUST stay byte-identical to the PRODUCER's
      // n_block_max (grep "[n_block_max contract]"; other site is in the producer warpgroup).
      // Divergence DEADLOCKS the pipeline (not a wrong number) -- see that site + docs/gotcha.md.
      int const offset_q = kv_len - qo_len;
      int const n_block_max =
          Causal ? cute::min(nb_tile,
                             ((q_tile_local + 1) * kBlockM + offset_q + kBlockN - 1) / kBlockN)
                 : nb_tile;

      // online-softmax row state (this thread owns 2 rows).
      float row_max[2] = {-INFINITY, -INFINITY};
      float row_sum[2] = {0.f, 0.f};
      // PV output accumulator (q x head_dim), running-rescaled across n_blocks.
      Tensor accO = partition_fragment_C(mma_pv, select<0, 2>(TileShape_MNK{}));
      Tensor accO_rc = make_tensor(
          accO.data(), make_layout(make_layout(get<0, 1>(accO.layout()), get<1>(accO.layout())),
                                   make_layout(get<0, 0>(accO.layout()), get<2>(accO.layout()))));
      clear(accO);

      {
        auto t = pipeline_q.consumer_try_wait(rq);
        pipeline_q.consumer_wait(rq, t);
        copy(scQ, tscQ.partition_S(as_position_independent_swizzle_tensor(sQ)),
             tscQ.retile_D(tSrQ));
        if constexpr (kLoadSF)
          copy(scSFQ, tscSFQ.partition_S(as_position_independent_swizzle_tensor(sSFQ)),
               tscSFQ.retile_D(tSrSFQ));
        else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tSrSFQ); ++i) tSrSFQ(i) = ElementSF::bitcast(kUniformSFByte);
        }
        pipeline_q.consumer_release(rq);
        ++rq;
      }

      // S9: drive the n_block loop in even/odd PAIRS so the SF half h = nb&1 is a
      // COMPILE-TIME constant (cute::Int<0/1>). The runtime (nb&1) indexing demoted the
      // SF register fragments to a 160B stack frame (SASS: 64x LDL.U8 + 32x STL.U8 per
      // n_block; ncu: local = ~90% of L1TEX sectors at 0.3B/sector, LSU the #1 issue
      // pipe). Static h keeps the SF bytes in registers -- identical selection values,
      // bit-exact same math. step() is instantiated once per half (Int<0>/Int<1>).
      Tensor accS =
          partition_fragment_C(mma_qk, select<0, 1>(TileShape_MNK{}));  // ((2,2),1,8) at kBlockN=64
      // reduction view: ((row=2,MMA_M=1),(col=2,MMA_N)) = 2 rows x kBlockN/2 cols.
      Tensor accS_rc = make_tensor(
          accS.data(), make_layout(make_layout(get<0, 1>(accS.layout()), get<1>(accS.layout())),
                                   make_layout(get<0, 0>(accS.layout()), get<2>(accS.layout()))));
      constexpr int kNRow = 2, kNCol = kBlockN / 4;

      auto step = [&](int nb, auto hc) {
        constexpr int h = decltype(hc)::value;
        // ---- QK ----
        {
          auto t = pipeline_k.consumer_try_wait(rk);
          pipeline_k.consumer_wait(rk, t);
          int stage = rk.index();
          copy(scK, tscK.partition_S(as_position_independent_swizzle_tensor(sK(_, _, stage))),
               tscK.retile_D(tSrK));
          if constexpr (kLoadSF)
            copy(scSFK,
                 tscSFK.partition_S(as_position_independent_swizzle_tensor(sSFK(_, _, stage))),
                 tscSFK.retile_D(tSrSFK));
          else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tSrSFK); ++i) tSrSFK(i) = ElementSF::bitcast(kUniformSFByte);
          }
        }
        clear(accS);
        auto tSrSFK_h = subSFK(tSrSFK, hc);  // 64-key SF half for this data block (static h)
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tSrQ); ++k)
          cute::gemm(mma_qk, make_zip_tensor(tSrQ(_, _, k), tSrSFQ(_, _, k)),
                     make_zip_tensor(tSrK(_, _, k), tSrSFK_h(_, _, k)), accS);
        pipeline_k.consumer_release(rk);
        ++rk;

        // request-local masking. Causal diagonal is shifted by offset_q = kv_len - qo_len
        // (slice-3): query m attends keys [0, m + offset_q]. qo_len==kv_len -> offset_q=0 ->
        // diagonal at local 0 (slice-1/2). partial_n drops padded keys past kv_len in the last
        // tile (non-causal needs this; causal already excludes them via the shifted diagonal).
        // Dense non-causal full tile: both conditions false -> whole block elided (bit-exact).
        // S9c: skip FULLY-UNMASKED blocks outright (the causal-varlen common case -- every
        // block below the diagonal): the whole elementwise pass reduces to one bound check.
        // The mask is per-thread on this thread's own accS elements, so full_n diverging
        // within a warp is fine. Same masks produced, bit-exact.
        bool const partial_n = ((nb + 1) * kBlockN > kv_len);
        int const m_base =
            q_tile_local * kBlockM + warp * 16 + (lane / 4);  // this thread's lowest row
        bool const full_n = !partial_n && (!Causal || (nb + 1) * kBlockN <= m_base + offset_q + 1);
        if (!full_n) {
          int const col_base = (lane % 4) * 2;     // col(ni) = (ni/2)*8 + col_base + (ni%2)
          int const kmax = kv_len - nb * kBlockN;  // mask col >= kmax (padded keys)
          CUTLASS_PRAGMA_UNROLL
          for (int mi = 0; mi < kNRow; ++mi) {
            int const thr = m_base + mi * 8 + offset_q - nb * kBlockN;  // causal: mask col > thr
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < kNCol; ++ni) {
              int col = (ni / 2) * 8 + col_base + (ni % 2);
              if ((Causal && col > thr) || col >= kmax) accS_rc(mi, ni) = -INFINITY;
            }
          }
        }

        // ---- online softmax + accO rescale factor ----
        float scores_scale[2];
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          float m_prev = row_max[mi];
          float m_cur = m_prev;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) m_cur = fmaxf(m_cur, accS_rc(mi, ni));
          m_cur = quad_reduce(m_cur, max_op);
          row_max[mi] = m_cur;
          float ss_mi = exp2f((m_prev - m_cur) * sm_scale_log2);  // first tile: m_prev=-inf -> 0
          scores_scale[mi] = ss_mi;
          // Subtract the max BEFORE scaling so the argmax gives exp2(0)=1.0 EXACTLY.
          // The `accS*sm - m_cur*sm` form lets the compiler contract to fma(accS, sm, -m_cur*sm),
          // which at the argmax yields the rounding error of m_cur*sm (~ -1.7e-7), not 0 -- so the
          // max P came out 0.99999988 < 1.0, dropping floor(log2) to -1, forcing the block scale
          // exponent one too low, and SATURATING the max element to 448*2^-9 = 0.875 (the 7/8 bug).
          float m_sub =
              (m_cur == -INFINITY) ? 0.f : m_cur;  // fully-masked row: keep -inf entries -> 0
          row_sum[mi] *= ss_mi;
          CUTLASS_PRAGMA_UNROLL
          for (int ni = 0; ni < kNCol; ++ni) {
            float p = exp2f((accS_rc(mi, ni) - m_sub) * sm_scale_log2);
            accS_rc(mi, ni) = p;
            row_sum[mi] += p;
          }
        }

        // dump pre-quant float P (host re-quantizes the IDENTICAL P) and running max --
        // the bit-exact reference replays the online algo from these device-side dumps.
        // Guarded by the pointer so a timing build (out_Ppre=nullptr) skips the full-P
        // gmem write, which otherwise dominates the kernel time.
        if (params.out_Ppre != nullptr) {
          Tensor gPre = local_tile(mPpre, select<0, 1>(TileShape_MNK{}),
                                   make_coord(q_tile_global, kv_tile_base + nb));
          copy(accS, thr_qk.partition_C(gPre));
          if (params.out_Mnb != nullptr) {
            CUTLASS_PRAGMA_UNROLL
            for (int mi = 0; mi < kNRow; ++mi) {
              int q = q_tile_global * kBlockM + warp * 16 + (lane / 4) + mi * 8;
              if ((lane % 4) == 0) params.out_Mnb[q * n_block_total + nb] = row_max[mi];
            }
          }
        }

        // ---- quantize P (e4m3) and deliver it to the PV-A operand tOrP ----
#if S3_P_SMEM
        // S2 oracle path: quantize to a register fragment, scatter to a swizzled smem
        // transpose buffer by logical (q,key), ldmatrix back as PV-A. Two NamedBarriers
        // bracket the shared sP buffer. Also the only path valid for a dynamic per-block SF.
        cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier)
            .sync();  // prev nb's PV readers done
        Tensor rP = make_fragment_like<Element>(accS);
        Tensor rP_rc = make_tensor(rP.data(), accS_rc.layout());
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < kNRow; ++mi) {
          int q_local = warp * 16 + (lane / 4) + mi * 8;  // 0..127 within this m_block tile
          CUTLASS_PRAGMA_UNROLL
          for (int sfi = 0; sfi < NKB; ++sfi) {
            int se;
            if constexpr (kPDynamicScale) {
              float amax = 0.f;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(accS_rc(mi, sfi * 8 + j)));
              amax = quad_reduce(amax, max_op);
              se = mx_scale_exp(amax);
            } else {
              se = kPScaleExp;  // P<=1.0 guaranteed -> constant scale 256.0, no per-block amax
            }
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < 8; ++j)
              rP_rc(mi, sfi * 8 + j) = quant_e4m3(accS_rc(mi, sfi * 8 + j), se);
            if (!kPConstSF && (lane % 4) == 0)
              ss.sSFP[q_local * NKB + sfi] = ElementSF::bitcast(uint8_t(se + 127));
            if (params.out_dbg) {  // dequantized requant-P, indexed by logical (q, key)
              int q = q_tile_global * kBlockM + warp * 16 + (lane / 4) + mi * 8;
              CUTLASS_PRAGMA_UNROLL
              for (int j = 0; j < 8; ++j) {
                int ni = sfi * 8 + j;
                int col = (ni / 2) * 8 + (lane % 4) * 2 + (ni % 2);
                params.out_dbg[q * params.seqlen_k + (kv_tile_base + nb) * kBlockN + col] =
                    float(rP_rc(mi, ni)) * exp2f(float(se));
              }
            }
          }
        }
        copy(rP, thr_qk.partition_C(as_position_independent_swizzle_tensor(sP)));
        cutlass::arch::NamedBarrier(NumMmaThreads, kQuantBarrier).sync();  // P/SF visible
        copy(scP, tscP.partition_S(as_position_independent_swizzle_tensor(sP)),
             tscP.retile_D(tOrP));
        if constexpr (kPConstSF) {  // constant P scale -> no smem SF, no gather
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP); ++i)
            tOrSFP(i) = ElementSF::bitcast(uint8_t(kPScaleExp + 127));
        } else {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < size(tOrSFP); ++i) {
            auto c = sfp_coord(i);
            tOrSFP(i) = ss.sSFP[int(get<0>(c)) * NKB + int(get<1>(c)) / SFVecSize];
          }
        }
#else
        // S5 path: quantize accS to e4m3 in registers, then intra-quad __shfl directly into
        // the PV-A operand tOrP -- no sP, no NamedBarrier. The QK-C key partition (2 adjacent
        // keys/lane, comb stride 8) differs from PV-A's (4 contiguous keys/lane), but the
        // redistribution is entirely within a 4-lane quad, so a shuffle (not smem) suffices.
        static_assert(kPConstSF,
                      "S5 shuffle path needs a fixed P scale (const SF); build -DS3_P_SMEM=1 for "
                      "the dynamic-scale oracle");
        // Pack this thread's 32 quantized P bytes per q-row into 8 little-endian uint32 words:
        // word g (ni=4g..4g+3) holds keys {16g+2ql, +1, +8, +9} for this lane's ql=lane%4.
        uint32_t qw[kNRow][kNCol / 4];
        CUTLASS_PRAGMA_UNROLL
        for (int r = 0; r < kNRow; ++r) {
          CUTLASS_PRAGMA_UNROLL
          for (int g = 0; g < kNCol / 4; ++g) {
            uint32_t w = 0;
            CUTLASS_PRAGMA_UNROLL
            for (int b = 0; b < 4; ++b)
              w |= uint32_t(quant_e4m3(accS_rc(r, 4 * g + b), kPScaleExp).storage) << (8 * b);
            qw[r][g] = w;
          }
        }
        if (params.out_dbg) {  // dequantized requant-P, indexed by logical (q, key)
          CUTLASS_PRAGMA_UNROLL
          for (int r = 0; r < kNRow; ++r) {
            int q = q_tile_global * kBlockM + warp * 16 + (lane / 4) + r * 8;
            CUTLASS_PRAGMA_UNROLL
            for (int ni = 0; ni < kNCol; ++ni) {
              int col = (ni / 2) * 8 + (lane % 4) * 2 + (ni % 2);
              params.out_dbg[q * params.seqlen_k + (kv_tile_base + nb) * kBlockN + col] =
                  float(quant_e4m3(accS_rc(r, ni), kPScaleExp)) * exp2f(float(kPScaleExp));
            }
          }
        }
        // Gather: PV-A lane L's uint32 (q-row r, e2, mk) takes the low/high 16b of source
        // word g=e2+2mk from quad lanes {2(L&1), 2(L&1)+1}; half=(L>>1)&1 picks low vs high.
        {
          Tensor tOrP_u32 = recast<uint32_t>(tOrP);  // ((1,2,2),1,MMA_K)
          int const qb = lane & ~3, off = 2 * (lane & 1), half = (lane >> 1) & 1;
          CUTLASS_PRAGMA_UNROLL
          for (int mk = 0; mk < size<2>(tOrP_u32); ++mk) {
            CUTLASS_PRAGMA_UNROLL
            for (int e2 = 0; e2 < 2; ++e2) {
              int const g = e2 + 2 * mk;
              CUTLASS_PRAGMA_UNROLL
              for (int r = 0; r < kNRow; ++r) {
                uint32_t wlo = __shfl_sync(0xffffffffu, qw[r][g], qb + off);
                uint32_t whi = __shfl_sync(0xffffffffu, qw[r][g], qb + off + 1);
                uint32_t lo = half ? (wlo >> 16) : (wlo & 0xffffu);
                uint32_t hi = half ? (whi >> 16) : (whi & 0xffffu);
                tOrP_u32(make_coord(_0{}, r, e2), _0{}, mk) = lo | (hi << 16);
              }
            }
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(tOrSFP); ++i)
          tOrSFP(i) = ElementSF::bitcast(uint8_t(kPScaleExp + 127));
#endif
        {
          auto t = pipeline_v.consumer_try_wait(rv);
          pipeline_v.consumer_wait(rv, t);
#if S3_V_KFILLZERO
          // Partial last block: zero the padded V DATA columns [valid, kBlockN) so a NaN there
          // can't meet masked P (0*NaN=NaN). One cooperative pass + barrier before the ldmatrix.
          if (partial_n) {
            int const valid = kv_len - nb * kBlockN;  // valid keys in this block, 1..kBlockN
            int const npad = kBlockN - valid;         // padded keys [valid, kBlockN)
            Tensor sV_pi = as_position_independent_swizzle_tensor(sV);
            for (int idx = tid; idx < kHeadDim * npad; idx += NumMmaThreads)
              sV_pi(idx / npad, valid + idx % npad) = Element(0);
            cutlass::arch::NamedBarrier(NumMmaThreads, kVFillBarrier).sync();
          }
#endif
          copy(scV, tscV.partition_S(as_position_independent_swizzle_tensor(sV)),
               tscV.retile_D(tOrV));
          if constexpr (kLoadSF)
            copy(scSFV, tscSFV.partition_S(as_position_independent_swizzle_tensor(sSFV)),
                 tscSFV.retile_D(tOrSFV));
          else {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tOrSFV); ++i) tOrSFV(i) = ElementSF::bitcast(kUniformSFByte);
          }
          // the ldmatrix drained sV/sSFV into registers -> release EARLY so the producer's
          // V(nb+1) TMA overlaps the rescale/PV work below (the gemm reads registers, not smem).
          pipeline_v.consumer_release(rv);
          ++rv;
        }
#if S3_V_KFILLZERO
        // Fully-masked 32-key tiles (all keys >= kv_len) may carry a garbage NaN SF (ue8m0 0xFF);
        // replace with a finite byte. Their DATA was zeroed above. The straddling tile keeps its
        // real SF (shared with valid keys), whose masked keys are already 0 in the data.
        if (partial_n) {
          int const valid = kv_len - nb * kBlockN;
          CUTLASS_PRAGMA_UNROLL
          for (int k = 0; k < NKB; ++k)
            if (k * SFVecSize >= valid) {
              Tensor sfk = tOrSFV(_, _, h * NKB + k);  // V-SF k-blocks for this 64-key half
              CUTLASS_PRAGMA_UNROLL
              for (int i = 0; i < size(sfk); ++i) sfk(i) = ElementSF::bitcast(uint8_t(0));
            }
        }
#endif

        // ---- rescale accO in registers, then PV accumulates DIRECTLY onto it ----
        // accO = accO*scores_scale (64 FMUL); accO += P*V (gemm C=D=accO). This replaces
        // the accB block accumulator + telescope and frees 64 registers (s9e) -- headroom
        // that lets the S9d V-load hoist stay spill-free, and lets ptxas overlap the
        // independent QK/PV QMMAs across the softmax ALU chain. accO native ((a,b),0,c):
        // b = get<0,1> = M-row (scores_scale is per M-row), a = N column-pair.
        // NOTE: NOT bit-exact vs the old accB+telescope add order (~1e-7 rel, well inside
        // the fp64-oracle tolerance).
        CUTLASS_PRAGMA_UNROLL
        for (int a = 0; a < 2; ++a) {
          CUTLASS_PRAGMA_UNROLL
          for (int b = 0; b < 2; ++b) {
            CUTLASS_PRAGMA_UNROLL
            for (int c = 0; c < size<2>(accO); ++c) {
              auto coord = make_coord(make_coord(a, b), _0{}, c);
              accO(coord) = accO(coord) * scores_scale[b];
            }
          }
        }
        CUTLASS_PRAGMA_UNROLL
        for (int k = 0; k < size<2>(tOrP); ++k)
          cute::gemm(mma_pv, make_zip_tensor(tOrP(_, _, k), tOrSFP(_, _, k)),
                     make_zip_tensor(tOrV(_, _, k), tOrSFV(_, _, h * NKB + k)), accO);  // V-SF half
      };
      for (int nb = 0; nb < n_block_max; nb += 2) {
        step(nb, cute::Int<0>{});                                // even block: SF half 0
        if (nb + 1 < n_block_max) step(nb + 1, cute::Int<1>{});  // odd block: SF half 1
      }

      // ---- epilogue: finalize row_sum, normalize O, write LSE ----
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 2; ++mi) row_sum[mi] = quad_reduce(row_sum[mi], add_op);
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < size<0>(accO_rc); ++mi) {
        // o_scale folds the per-tensor v_scale (kUniformFp8); 1.0 for kMxFp8 -> bit-exact
        // 1/row_sum.
        float inv = (row_sum[mi] == 0.f) ? 0.f : params.o_scale / row_sum[mi];
        CUTLASS_PRAGMA_UNROLL
        for (int ni = 0; ni < size<1>(accO_rc); ++ni) accO_rc(mi, ni) *= inv;
      }
      Tensor gO = local_tile(mO, select<0, 2>(TileShape_MNK{}), make_coord(q_tile_global, _0{}));
      // S9d: the C fragment's mode-0 is a stride-1 column PAIR (8B, provably aligned: col
      // pairs are even, rows are 512B apart) -- but plain cute::copy assumes 128b alignment
      // and falls back to scalar STG.E (SASS: 16.1/32B per sector). Force 64b vectorization
      // -> STG.E.64, halving epilogue store instructions and L2 write sectors.
      copy(AutoVectorizingCopyWithAssumedAlignment<64>{}, accO, thr_pv.partition_C(gO));
      CUTLASS_PRAGMA_UNROLL
      for (int mi = 0; mi < 2; ++mi) {
        int q_local = q_tile_local * kBlockM + warp * 16 + (lane / 4) + mi * 8;
        int q = q_tile_global * kBlockM + warp * 16 + (lane / 4) + mi * 8;
        if (q_local < qo_len && (lane % 4) == 0) {
          int qh = qo_head_idx * params.seqlen_q + q;  // head-major lse/l
          params.out_l[qh] = row_sum[mi];
          params.out_lse[qh] =
              (row_sum[mi] == 0.f) ? -INFINITY : (row_max[mi] * sm_scale + logf(row_sum[mi]));
        }
      }
    }
  }
}

}  // namespace mxfp8_attention_sm120
}  // namespace flashinfer
