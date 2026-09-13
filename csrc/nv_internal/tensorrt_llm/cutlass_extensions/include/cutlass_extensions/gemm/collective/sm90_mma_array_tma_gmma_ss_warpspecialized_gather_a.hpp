/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// SM90 grouped (ptr-array) warp-specialized GMMA mainloop that loads operand B
// through TMA but loads operand A by ROW-GATHERING with cp.async from an
// unpermuted activation buffer. Row m of group g is read from
//
//   ptr_A[g] + ptr_gather_a_index[g][m] * row_pitch + k,
//
// where row_pitch = stride<0>(dA[g]) = gemm_k
//
// CONSUMER-GATHERED A (how it maps onto the forked kernel layer,
// cutlass_extensions/gemm/kernel/sm90_gemm_array_tma_warpspecialized_cooperative_gather_a.hpp):
//  * The Mainloop producer warp runs load(): B-only TMA, issued by the
//    elect_one_sync leader. The pipeline full barrier keeps the STOCK
//    semantics: arrival count NumProducerThreadEvents = 1 (the TMA leader's
//    arrive-and-expect-tx) and TmaTransactionBytes covers B's stage bytes
//    ONLY. The full/empty barriers govern B alone.
//  * Each CONSUMER warpgroup gathers ITS OWN rows of A inside mma_gather():
//    the cooperative TiledMma stacks its two warpgroups on M, so warpgroup w
//    only ever feeds wgmma descriptors that read the 64-row slabs
//    { m : (m / GatherSlabM) % MmaWarpGroups == w } of the smem A tile (see
//    the ThrLayoutVMNK static_asserts below that pin this mapping). Per
//    k-tile, the 128 threads of warpgroup w issue an unpredicated tiled
//    cp.async (16B vectors along K) covering exactly those BLK_M/2 rows,
//    sourced from the gathered gmem view (CustomStride/IndexedGather performs
//    the row-id lookup) and written into the standard GMMA-swizzled
//    SmemLayoutA stage.
//
// A-pipeline (no mbarrier for A):
//  * Issue: for k-tile t the warpgroup issues its cp.asyncs and then
//    cp.async.commit_group, GatherLookahead (=D) tiles ahead of consumption:
//    a prologue issues tiles 0..D-1 (one commit-group each), and each GMMA
//    iteration t first issues tile t+D. Past the last tile an EMPTY
//    commit-group is still made, so "one group per iteration" holds and the
//    wait below is exact.
//  * Wait: before the GMMA on tile t, each thread runs
//    cp.async.wait_group<D> -- committed groups at that point number
//    D + t + 1, so at most D pending means groups 0..t (FIFO) have retired,
//    i.e. this thread's copies for tile t have landed -- followed by
//    fence.proxy.async.shared::cta (cp.async writes are generic-proxy;
//    wgmma reads smem through the async proxy) and a per-warpgroup
//    NamedBarrier over its 128 threads so every thread's copies (not just
//    the caller's) are visible. Distinct barrier per warpgroup: user ids 0/1
//    (= hardware barriers 8/9 after CUTLASS's FirstUserBarrier offset).
//    These are free in this kernel: the epilogue uses only the reserved
//    EpilogueBarrier (hw id 1), and the scheduler / load-order / mainloop
//    pipelines are all mbarrier-based.
//  * A-buffer reuse safety: only warpgroup w ever writes or wgmma-reads its
//    own half of an A stage, so ordering is intra-warpgroup: the buffer for
//    tile t+D was last read by GMMA(t + D - Stages), and the issue at
//    iteration t is preceded by iteration t-1's warpgroup_wait<K_PIPE_MMAS>,
//    which guarantees GMMAs up to t - 1 - K_PIPE_MMAS have completed.
//    D <= Stages - K_PIPE_MMAS - 1 (static_assert below) therefore makes the
//    rewrite safe; across work tiles, mma_tail's warpgroup_wait<0> retires
//    every read before the next tile's prologue issues.
//  * B reuse is unchanged: consumer_release with K_PIPE_MMAS lag hands B
//    stages back to the TMA producer through the empty barrier.
//
// Hard requirements / reliances (see also can_implement()):
//  (a) M-tail rows are handled IN-KERNEL: each warpgroup precomputes a
//      K-invariant row-validity predicate over its slabs once per work tile
//      and SKIPS the copies of rows at/beyond the group's valid count -- no
//      gather-index read and no activation read is performed for them, so
//      the index buffer needs no tail padding or sanitization. Their smem
//      rows hold stale data, which only feeds output rows the epilogue's
//      M-predication drops (never a kept accumulator). Along K, all full
//      k-tiles are unpredicated; only the final k-tile of a problem with
//      K % BLK_K != 0 switches to a predicated ZFILL copy (cp.async src-size
//      0 zero-fills the masked 16B vectors of smem, gmem is not dereferenced)
//      -- the same K-tail guarantee the dense path gets from TMA bounds
//      checking, paid only on that one tile. The K tail must ZFILL rather
//      than skip because it contributes additively to kept accumulators.
//  (b) 1x1x1 cluster only (static_assert in the dispatch policy): cp.async
//      cannot multicast, so A traffic cannot be shared across a cluster, and
//      B's multicast masks stay trivial.
//  (c) A is 16-bit, K-major (row-major [tokens, K]), with K % 8 == 0 so every
//      gathered row supports 16B-aligned 16B vectors (the activation base
//      pointer itself must be 16B aligned, which cannot be checked host-side
//      for device pointer arrays -- same limitation as the CUTLASS TMA path).
//
// Template parameters are IDENTICAL to the CUTLASS specialization, so a builder
// can rebind by swapping only the dispatch policy.

#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm80.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/copy_traits_sm90_tma.hpp"  // make_tma_copy when included standalone
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/detail/layout.hpp"  // cutlass::detail::check_alignment
#include "cutlass/gemm/collective/collective_mma_decl.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass_extensions/gemm/dispatch_policy_gather_a.hpp"
#include "cutlass_extensions/util/gather_tensor.hpp"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop, TMA for B + consumer-warpgroup cp.async row-gather for A
template <int Stages, class ClusterShape, class KernelSchedule, class TileShape_, class ElementA_,
          class StrideA_, class ElementB_, class StrideB_, class TiledMma_, class GmemTiledCopyA_,
          class SmemLayoutAtomA_, class SmemCopyAtomA_, class TransformA_, class GmemTiledCopyB_,
          class SmemLayoutAtomB_, class SmemCopyAtomB_, class TransformB_>
struct CollectiveMma<
    MainloopSm90ArrayTmaGmmaWarpSpecializedGatherA<Stages, ClusterShape, KernelSchedule>,
    TileShape_, ElementA_, StrideA_, ElementB_, StrideB_, TiledMma_, GmemTiledCopyA_,
    SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_,
    SmemCopyAtomB_, TransformB_> {
  //
  // Type Aliases
  //
  using DispatchPolicy =
      MainloopSm90ArrayTmaGmmaWarpSpecializedGatherA<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA =
      GmemTiledCopyA_;  // Unused: A is gathered with cp.async, kept for builder-compatible arity
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  using PipelineParams = typename MainloopPipeline::Params;
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));

  // CUTLASS semantics: the full barrier's arrival count is just the TMA
  // leader's arrive-and-expect-tx (B bytes). A never touches the barrier --
  // the consumer warpgroups gather it themselves and synchronize with
  // cp.async.wait_group + a per-warpgroup named barrier (see file header).
  static constexpr int NumProducerThreadEvents = 1;

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;

  static_assert(rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  // The A gather requires 16-bit activations in row-major [tokens, K] form so
  // rows are contiguous 16B-vectorizable runs.
  static_assert(cutlass::sizeof_bits<ElementA>::value == 16,
                "Gather-A mainloop requires 16-bit A elements (bf16/fp16).");
  static_assert(::cutlass::gemm::detail::is_major<1, StrideA>(),
                "Gather-A mainloop requires K-major (row-major [M,K]) operand A.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideA>(), Step<_2, _1, _3>,
                          Step<_1, _2, _3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t<::cutlass::gemm::detail::is_major<0, StrideB>(), Step<_2, _1, _3>,
                          Step<_1, _2, _3>>{}));

  static_assert(DispatchPolicy::Stages >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert(
      cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
          cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
      "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA =
      cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB =
      cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  //
  // Consumer-warpgroup A-gather geometry.
  //
  // The cooperative TiledMma stacks MmaWarpGroups (=2) GMMA atoms on M, each
  // atom spanning GatherSlabM (=64) rows and executed by one warpgroup. With
  // BLK_M = repetitions * (GatherSlabM * MmaWarpGroups), warpgroup w's wgmma
  // descriptors read exactly the 64-row slabs whose slab index is congruent
  // to w modulo MmaWarpGroups (CuTe's logical_divide puts the atom mode
  // fastest, so tiled-MMA M repetitions advance in GatherSlabM*MmaWarpGroups
  // strides). Warpgroup w therefore gathers exactly those slabs; the
  // ThrLayoutVMNK asserts below pin the thread->atom-row mapping this relies
  // on instead of hardcoding "wg 0 owns rows 0..63".
  static constexpr int MmaWarpGroups = cute::size(TiledMma{}) / NumThreadsPerWarpGroup;
  static constexpr int GatherSlabM = cute::size<0>(typename TiledMma::AtomShape_MNK{});
  static_assert(cute::size(typename TiledMma::AtomThrID{}) == NumThreadsPerWarpGroup,
                "Gather-A mainloop expects one warpgroup per GMMA atom.");
  static_assert(cute::size<1>(typename TiledMma::ThrLayoutVMNK{}) == MmaWarpGroups &&
                    cute::size<2>(typename TiledMma::ThrLayoutVMNK{}) == 1 &&
                    cute::size<3>(typename TiledMma::ThrLayoutVMNK{}) == 1,
                "Gather-A mainloop expects the cooperative TiledMma: atoms stacked on M only.");
  static_assert(
      cute::stride<1>(typename TiledMma::ThrLayoutVMNK{}) == NumThreadsPerWarpGroup,
      "Gather-A mainloop expects warpgroup w to own GMMA atom-row w (wg-major thread numbering).");
  static_assert(cute::size<0>(TileShape{}) % (GatherSlabM * MmaWarpGroups) == 0,
                "BLK_M must be a whole number of (slab x warpgroup) M repetitions.");
  static constexpr int GatherSlabsPerWg =
      cute::size<0>(TileShape{}) / (GatherSlabM * MmaWarpGroups);

  // A row-gather copy executed by the 128 threads of one consumer warpgroup
  // over one (GatherSlabM, BLK_K) slab: 16B cp.async vectors along K,
  // GatherThreadsK consecutive threads cover one row's BLK_K (so each warp
  // reads whole 128B lines of a gathered row), GatherThreadsM rows in
  // parallel. For the shipped gather tiles (BLK_K = 64, bf16/fp16) this is
  // (16 rows x 8 K-threads) iterated GatherSlabM/16 = 4 times: 4 cp.asyncs
  // (+4 L1-resident index loads) per thread per slab per k-tile.
  static constexpr int GatherVectorElemsA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 8
  static_assert(size<2>(TileShape{}) % GatherVectorElemsA == 0,
                "BLK_K must be a multiple of the 16B gather vector.");
  static constexpr int GatherThreadsK = size<2>(TileShape{}) / GatherVectorElemsA;
  static_assert(GatherThreadsK <= NumThreadsPerWarpGroup &&
                    NumThreadsPerWarpGroup % GatherThreadsK == 0,
                "BLK_K vectors must tile the warpgroup evenly.");
  static constexpr int GatherThreadsM = NumThreadsPerWarpGroup / GatherThreadsK;
  static_assert(GatherSlabM % GatherThreadsM == 0, "Gather rows must tile the slab evenly.");
  using GatherCopyAtomA = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>, InternalElementA>;
  using GatherTiledCopyA = decltype(make_tiled_copy(
      GatherCopyAtomA{},
      Layout<Shape<Int<GatherThreadsM>, Int<GatherThreadsK>>,
             Stride<Int<GatherThreadsK>, _1>>{},       // K-fastest thread order
      Layout<Shape<_1, Int<GatherVectorElemsA>>>{}));  // 8 values along K (16B)
  // Predicated twin for the final k-tile of K % BLK_K != 0 problems: masked
  // vectors issue cp.async with src-size 0, which zero-fills their 16B of
  // smem without touching gmem (file header (a)). K % GatherVectorElemsA == 0
  // is still required (checked host-side) so validity is uniform within each
  // 16B vector.
  using GatherCopyAtomAZfill =
      Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL_ZFILL<cute::uint128_t>, InternalElementA>;
  using GatherTiledCopyAZfill = decltype(make_tiled_copy(
      GatherCopyAtomAZfill{},
      Layout<Shape<Int<GatherThreadsM>, Int<GatherThreadsK>>, Stride<Int<GatherThreadsK>, _1>>{},
      Layout<Shape<_1, Int<GatherVectorElemsA>>>{}));

  // Lookahead depth D of the consumer A gather: the prologue fills D stages,
  // then each iteration issues stage t+D before consuming stage t.
  // D <= Stages - K_PIPE_MMAS - 1 is the A-buffer reuse safety condition
  // (see file header); 3 stages of lookahead are plenty to hide gmem latency
  // behind the GMMA k-tile pipeline.
  static constexpr int GatherLookahead =
      (K_PIPE_MAX - K_PIPE_MMAS - 1) < 3 ? (K_PIPE_MAX - K_PIPE_MMAS - 1) : 3;
  static_assert(
      GatherLookahead >= 1,
      "Gather-A mainloop needs Stages >= K_PIPE_MMAS + 2 to overlap the A gather at all.");
  static_assert(GatherLookahead <= K_PIPE_MAX - K_PIPE_MMAS - 1,
                "A-buffer reuse safety: warpgroup_wait<K_PIPE_MMAS> must retire the last read of a "
                "stage before it is rewritten GatherLookahead tiles ahead.");

  // Assumption: StrideB is congruent with Problem_NK
  using TMA_B = decltype(make_tma_copy(
      GmemTiledCopyB{},
      make_tensor(static_cast<InternalElementB const*>(nullptr),
                  repeat_like(InternalStrideB{}, int32_t(0)), InternalStrideB{}),
      SmemLayoutB{}(_, _, cute::Int<0>{}), make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
      size<0>(ClusterShape{})));  // mcast along M mode for this N load, if any

  struct SharedStorage {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    // Only B streams through TMA; A needs no tensormap.
    struct TensorMapStorage : cute::aligned_struct<128, _0> {
      cute::TmaDescriptor smem_tensormap_B;
    } tensormaps;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel = !cute::is_same_v<InternalStrideA, StrideA>;

  // Host side kernel arguments
  struct Arguments {
    // ptr_A[g] is the SHARED unpermuted activation base (same for all groups).
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    // ptr_gather_a_index[g] is group g's slice of the permuted source-token-id
    // array; entries beyond the group's M (up to the BLK_M tile boundary) MUST
    // be 0 (see file header).
    int const** ptr_gather_a_index;
  };

  // Device side kernel params
  struct Params {
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    void* tensormaps;
    InternalElementA const** ptr_A;
    StrideA dA;
    InternalElementB const** ptr_B;
    StrideB dB;
    int const** ptr_gather_a_index;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape problem_shapes,
                                                  Arguments const& args, void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are only used to create
    // tensormap/tma desc. These will be replaced with correct values before the initial tma load.
    auto init_shape = repeat_like(typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);
    // Batches/Groups are managed by using appropriate pointers to input matrices
    const uint32_t init_L = 1;
    // NOTE: Since TMA desc creation with nullptr not possible until 12.6, we use an initial address
    // even when tensor addresses are on device. This address is never used.
    InternalElementB const* ptr_B_first_batch = reinterpret_cast<InternalElementB const*>(
        reinterpret_cast<uint64_t>(args.ptr_B) &
        0xFFFFFFFFFFFFFFF0);  // Address must be 16B-aligned

    InternalStrideB stride_b;
    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access regardless.
      stride_b = InternalStrideB{};
    } else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      stride_b = args.dB;
    }
    (void)init_M;
    Tensor tensor_b =
        make_tensor(ptr_B_first_batch, make_layout(make_shape(init_N, init_K, init_L), stride_b));
    TMA_B tma_load_b =
        make_tma_copy(GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}),
                      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
                      size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any

    void* tensormaps = workspace;

    return {tma_load_b, TmaTransactionBytes,
            tensormaps, reinterpret_cast<InternalElementA const**>(args.ptr_A),
            args.dA,    reinterpret_cast<InternalElementB const**>(args.ptr_B),
            args.dB,    args.ptr_gather_a_index};
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape, Arguments const& args,
                                   int sm_count) {
    constexpr uint32_t NumInputTensors = 1;  // B only; A has no tensormap
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);
    // Allocate gmem space for input tensormaps per each SM
    return (NumInputTensors * SizeOfCuTensorMap * sm_count);
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(ProblemShape const& problem_shape,
                                              Arguments const& args, void* workspace,
                                              cudaStream_t stream,
                                              CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  static bool can_implement(ProblemShape problem_shapes, Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL = append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        implementable =
            implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                                 cute::make_shape(N, K, L), InternalStrideB{});
        // A is row-gathered with 16B cp.async vectors: the contiguous extent K
        // (== the row pitch) must be a multiple of the vector width.
        implementable = implementable && (K % GatherVectorElemsA == 0);
        // The gather issues full, unpredicated k-tiles; a K tail would read
        // past the end of a token's row (see file header). Stricter than the
        // 16B check on purpose: rejecting here beats silent NaN.
        implementable = implementable && (K % size<2>(TileShape{}) == 0);
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for "
          "TMA/gather.\n");
    }
    return implementable;
  }

  // B bytes ONLY: A is gathered by the consumer warpgroups with cp.async and
  // never arrives on the pipeline barrier.
  static constexpr uint32_t TmaTransactionBytes =
      cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
                             static_cast<uint32_t>(sizeof_bits<ElementB>::value));

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the contract that the
  // returned tuple must contain at least two elements, with the first two elements being:
  // gA_mkl - The gathered gmem view of A after a local tile: shape (BLK_M,BLK_K,m,k,l)
  // gB_nkl - The tma tensor, B after a local tile so it has shape (BLK_N,BLK_K,n,k,l)
  // The initial call (group is not known yet) only fixes the tuple's TYPE and
  // tile counts; the kernel always routes through tensors_perform_update()
  // before the first load/mma, which rebuilds this tuple for the actual group.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                Params const& mainloop_params, int32_t group = 0) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t init_L = 1;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensor B -- get it from TMA
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N, K, init_L));  // (n,k,l)

    // A: plain gmem view of the unpermuted activations, with the m-mode stride
    // replaced by IndexedGather(row_ids) * row_pitch.
    auto dA = [&]() {
      if constexpr (IsGroupedGemmKernel) {
        return mainloop_params.dA[group];
      } else {
        return mainloop_params.dA;
      }
    }();
    Tensor mA_mkl = cutlass::util::make_gather_tensor(
        make_gmem_ptr(mainloop_params.ptr_A[group]), make_shape(M, K, init_L),
        make_stride(get<0>(dA), _1{}, get<2>(dA)),  // (row_pitch, 1, unused-l)
        cutlass::util::IndexedGather<int const*>{
            mainloop_params.ptr_gather_a_index[group]});  // (m,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _),
                               Step<X, _1, _1>{});  // (BLK_N,BLK_K,n,k,l)

    return cute::make_tuple(gA_mkl, gB_nkl);
  }

  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective -- Mainloop warp: B via TMA only. A is gathered by
  // the consumer warpgroups in mma_gather(); do NOT arrive for A here.
  template <class TensorA, class TensorB, class TensorMapB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(Params const& mainloop_params, MainloopPipeline pipeline,
                           PipelineState smem_pipe_write,
                           cute::tuple<TensorA, TensorB> const& load_inputs,
                           cute::tuple<TensorMapB> const& input_tensormaps,
                           BlockCoord const& blk_coord, KTileIterator k_tile_iter, int k_tile_count,
                           int thread_idx, uint32_t block_rank_in_cluster,
                           TensorStorage& shared_tensors) {
    int lane_predicate = cute::elect_one_sync();

    if (lane_predicate) {
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()),
                              SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

      //
      // Prepare the TMA load for B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                      block_rank_in_cluster / cluster_shape_x};

      Tensor gB_nkl = get<1>(load_inputs);

      auto block_tma_b = mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);  // (BLK_N,BLK_K,k)

      Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m, cluster_local_block_id.y, Int<0>{}));
        }
      }

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for (; k_tile_count > 0; --k_tile_count) {
        // LOCK smem_pipe_write for _writing_.
        // As pipeline leader, this arrives on the full barrier expecting
        // TmaTransactionBytes (B only).
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

        int write_stage = smem_pipe_write.index();
        copy(mainloop_params.tma_load_b.with(get<0>(input_tensormaps), *tma_barrier, mcast_mask_b),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
        ++k_tile_iter;

        // Advance smem_pipe_write
        ++smem_pipe_write;
      }
    }
  }

  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster.
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  /// Perform a collective-scoped matrix multiply-accumulate, gathering this
  /// warpgroup's rows of operand A with cp.async along the way.
  /// Consumer Perspective. Called by both MMA warpgroups; thread_idx is the
  /// thread's index within the MMA warpgroups (0..size(TiledMma)-1).
  ///
  /// The A copies carry two predicates (file header (a)):
  ///  - M rows at/beyond m_extent (the group's valid row count) are SKIPPED:
  ///    no index read, no activation read; the epilogue's M-predication
  ///    discards the corresponding output rows.
  ///  - Along K only the final k-tile is ZFILL-predicated, and only when
  ///    k_tail_extent < BLK_K (k_tail_extent = K - (k_tile_count-1)*BLK_K,
  ///    i.e. the valid element count of the last k-tile).
  template <class FrgTensorC, class TensorA, class TensorB, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void mma_gather(MainloopPipeline pipeline, PipelineState smem_pipe_read,
                                 FrgTensorC& accum,
                                 cute::tuple<TensorA, TensorB> const& load_inputs,
                                 BlockCoord const& blk_coord, KTileIterator k_tile_iter,
                                 int k_tile_count, int k_tail_extent, int m_extent, int thread_idx,
                                 TensorStorage& shared_tensors, Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(
        cute::is_void_v<SmemCopyAtomA>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(
        cute::is_void_v<SmemCopyAtomB>,
        "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()),
                            SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                      stride<0>(typename TiledMma::BLayout{}) == 0 and
                      size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                      size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be "
                  "NumThreadsPerWarpGroup");

    Layout warp_group_thread_layout =
        make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));               // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));               // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));  // PIPE

    //
    // A-gather partitioning: this warpgroup's 64-row slabs of the A tile.
    //

    // View the A stage buffer through InternalElementA so src/dst value types
    // match for the cp.async atom (both are 16-bit; GMMA reads it as ValTypeA).
    Tensor sA_gather = make_tensor(
        make_smem_ptr(reinterpret_cast<InternalElementA*>(shared_tensors.smem_A.data())),
        SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)

    // The gathered gmem view of A. The gathered tensor is built with L = 1
    // (its group slice is selected by tensors_perform_update), so the l
    // coordinate is always 0 here regardless of blk_coord's L index.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA_mkl = get<0>(load_inputs);             // (BLK_M,BLK_K,m,k,l) gathered
    Tensor gA = gA_mkl(_, _, m_coord, _, Int<0>{});  // (BLK_M,BLK_K,k)

    // Split the M mode into GatherSlabM-row slabs; warpgroup w owns slab
    // indices congruent to w modulo MmaWarpGroups (see class comment).
    Tensor gA_slabs = flat_divide(gA, Shape<Int<GatherSlabM>>{});  // (SlabM,M_SLAB,BLK_K,k)
    Tensor sA_slabs =
        flat_divide(sA_gather, Shape<Int<GatherSlabM>>{});  // (SlabM,M_SLAB,BLK_K,PIPE)

    GatherTiledCopyA gather_copy_a{};
    ThrCopy thr_copy_a = gather_copy_a.get_slice(thread_idx % NumThreadsPerWarpGroup);

    // K-tail predicate, built once: validity depends only on this thread's
    // static K offsets within a (GatherSlabM, BLK_K) slab. Consulted only for
    // the final k-tile when K % BLK_K != 0; K % GatherVectorElemsA == 0
    // (host-checked) makes validity uniform within each 16B vector, so
    // testing the vector's first-element coordinate suffices.
    bool const has_k_tail = k_tail_extent < size<2>(TileShape{});
    Tensor cA_slab = make_identity_tensor(make_shape(Int<GatherSlabM>{}, size<2>(TileShape{})));
    Tensor tAcA = thr_copy_a.partition_S(cA_slab);  // (CPY,CPY_M,CPY_K) -> (m,k)
    Tensor tApA = make_tensor<bool>(make_shape(size<1>(tAcA), size<2>(tAcA)));
    CUTLASS_PRAGMA_UNROLL
    for (int m = 0; m < size<0>(tApA); ++m) {
      CUTLASS_PRAGMA_UNROLL
      for (int k = 0; k < size<1>(tApA); ++k) {
        tApA(m, k) = static_cast<int>(get<1>(tAcA(0, m, k))) < k_tail_extent;
      }
    }

    // M-row predicate, built once per work tile: row validity is K-invariant.
    // Rows at or beyond the group's valid count (m_extent) belong to the
    // group's final, partial M-tile; their copies are SKIPPED entirely, so a
    // masked row performs no gather-index read and no activation-row read --
    // out-of-range indices are simply never dereferenced, with no host-side
    // index padding or tail sanitization required. Skipping (unlike the K
    // tail's ZFILL) is sound because a masked M row only feeds its own output
    // row, which the epilogue's M-predication drops; it never contributes to
    // a kept accumulator.
    Tensor tAmA = make_tensor<bool>(make_shape(Int<GatherSlabsPerWg>{}, size<1>(tAcA)));
    bool slab_all_valid[GatherSlabsPerWg];
    CUTLASS_PRAGMA_UNROLL
    for (int s = 0; s < GatherSlabsPerWg; ++s) {
      int const slab = warp_group_idx + s * MmaWarpGroups;
      int const slab_origin =
          static_cast<int>(m_coord) * static_cast<int>(size<0>(TileShape{})) + slab * GatherSlabM;
      slab_all_valid[s] = slab_origin + GatherSlabM <= m_extent;
      CUTLASS_PRAGMA_UNROLL
      for (int m = 0; m < size<1>(tAcA); ++m) {
        tAmA(s, m) = slab_origin + static_cast<int>(get<0>(tAcA(0, m, 0))) < m_extent;
      }
    }

    // A-issue state: runs GatherLookahead k-tiles ahead of the GMMA consumption.
    auto k_tile_iter_issue = k_tile_iter;
    int k_tiles_to_issue = k_tile_count;
    PipelineState smem_pipe_issue = smem_pipe_read;

    // Issue this warpgroup's cp.asyncs for the next unissued k-tile (if any)
    // and ALWAYS commit exactly one cp.async group -- empty past the last
    // k-tile -- so the FIFO group accounting behind cp_async_wait
    // <GatherLookahead> stays exact (see file header). The k-tile check is
    // warpgroup-uniform, so the tail branch does not diverge.
    auto issue_a_group = [&]() {
      if (k_tiles_to_issue > 0) {
        int write_stage = smem_pipe_issue.index();
        bool const is_tail_tile = has_k_tail && (k_tiles_to_issue == 1);
        CUTLASS_PRAGMA_UNROLL
        for (int s = 0; s < GatherSlabsPerWg; ++s) {
          int slab = warp_group_idx + s * MmaWarpGroups;
          Tensor tAgA = thr_copy_a.partition_S(
              gA_slabs(_, slab, _, *k_tile_iter_issue));  // (CPY,CPY_M,CPY_K)
          Tensor tAsA =
              thr_copy_a.partition_D(sA_slabs(_, slab, _, write_stage));  // (CPY,CPY_M,CPY_K)
          if (slab_all_valid[s]) {
            if (is_tail_tile) {
              copy_if(GatherTiledCopyAZfill{}, tApA, tAgA, tAsA);
            } else {
              copy(gather_copy_a, tAgA, tAsA);
            }
          } else {
            CUTLASS_PRAGMA_UNROLL
            for (int m = 0; m < size<1>(tAsA); ++m) {
              if (tAmA(s, m)) {
                if (is_tail_tile) {
                  copy_if(GatherTiledCopyAZfill{}, tApA(m, _), tAgA(_, m, _), tAsA(_, m, _));
                } else {
                  copy(gather_copy_a, tAgA(_, m, _), tAsA(_, m, _));
                }
              }
            }
          }
        }
        ++k_tile_iter_issue;
        --k_tiles_to_issue;
        ++smem_pipe_issue;
      }
      cute::cp_async_fence();
    };

    // Wait until this thread's cp.asyncs for the k-tile about to be consumed
    // have retired (all but the newest GatherLookahead groups), make the
    // generic-proxy cp.async writes visible to the async proxy (wgmma reads
    // smem descriptors through the async proxy), then barrier the warpgroup
    // so every thread's copies -- not just the caller's -- are known landed.
    // User barrier id = warp_group_idx (hardware barrier 8 + warp_group_idx):
    // free in this kernel, see file header.
    auto wait_a_stage = [&]() {
      cute::cp_async_wait<GatherLookahead>();
      cutlass::arch::fence_view_async_shared();
      cutlass::arch::NamedBarrier::sync(NumThreadsPerWarpGroup,
                                        static_cast<uint32_t>(warp_group_idx));
    };

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 < K_PIPE_MMAS) && (K_PIPE_MMAS < K_PIPE_MAX),
                  "ERROR : Incorrect number of MMAs in flight");
    // The prologue below hard-codes a single tile with manual ScaleOut
    // handling; the general multi-tile GMMA prologue of the CUTLASS mma() is
    // not replicated.
    static_assert(K_PIPE_MMAS == 1, "mma_gather assumes a single GMMA prologue tile.");

    // A-gather prologue: fill the first GatherLookahead stages. Safe against
    // buffer reuse across work tiles because mma_tail's warpgroup_wait<0>
    // retired all of this warpgroup's reads before we got here.
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < GatherLookahead; ++i) {
      issue_a_group();
    }

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
    warpgroup_fence_operand(accum);
    if (k_tile_count > 0) {
      // Keep the A pipeline GatherLookahead tiles ahead of consumption.
      issue_a_group();

      // WAIT on smem_pipe_read until B's data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      // WAIT on this warpgroup's A gather for the same stage
      wait_a_stage();

      int read_stage = smem_pipe_read.index();
      warpgroup_arrive();
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_, _, k_block, read_stage), tCrB(_, _, k_block, read_stage),
                   accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();

      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // Keep the A pipeline GatherLookahead tiles ahead of consumption. The
      // stage being rewritten was last read GatherLookahead + K_PIPE_MMAS + 1
      // GMMAs ago at most, which the previous iteration's
      // warpgroup_wait<K_PIPE_MMAS> has already retired (see file header).
      issue_a_group();

      // WAIT on smem_pipe_read until B's data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      // WAIT on this warpgroup's A gather for the same stage
      wait_a_stage();

      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      cute::gemm(tiled_mma, tCrA(_, _, _, read_stage), tCrB(_, _, _, read_stage),
                 accum);  // (V,M,K) x (V,N,K) => (V,M,N)
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write
      /// is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);

      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release,
                               int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(
          smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }

  //
  // Methods to perform different parts of TMA/Tensormap modifications
  // (B only -- A has no tensormap; its per-group rebinding happens in
  // tensors_perform_update()).
  //

  CUTLASS_DEVICE auto tensormaps_init(Params const& mainloop_params,
                                      TensorMapStorage& shared_tensormaps, int32_t sm_count,
                                      int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap =
        reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx];

    if (cute::elect_one_sync()) {
      // Bringing tensormaps from params to smem for modification later
      Tensor pB_tensormap =
          make_tensor(mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
      Tensor sB_tensormap =
          make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }
    __syncwarp();

    return cute::make_tuple(tma_desc_b);
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void tensormaps_replace_global_address(TensorMapStorage& shared_tensormaps,
                                         Params const& mainloop_params, int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                    mainloop_params.ptr_B[next_batch]);
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM (to be done by
  // single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps, Params const& mainloop_params, int32_t next_group,
      ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);
    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_B = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0, 0, 0, 0, 0};

    InternalElementB const* ptr_B = nullptr;
    Tensor tensor_b =
        make_tensor(ptr_B, make_shape(N, K, Int<1>{}), mainloop_params.dB[next_group]);

    cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_b, tensor_b, prob_shape_B,
                                             prob_stride_B);

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<InternalElementB>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(shared_tensormaps.smem_tensormap_B,
                                                            prob_shape_B, prob_stride_B);
  }

  template <class TensorMapB, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_perform_update(TensorMapStorage& shared_tensormaps,
                                                Params const& mainloop_params,
                                                cute::tuple<TensorMapB> const& input_tensormaps,
                                                ProblemShape_MNKL problem_shape_mnkl,
                                                int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params, next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(shared_tensormaps, mainloop_params, next_batch,
                                                    problem_shape_mnkl);
      }
    }
  }

  template <class TensorMapB>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(TensorMapStorage& shared_tensormaps,
                                                  cute::tuple<TensorMapB> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps), shared_tensormaps.smem_tensormap_B);
  }

  // The entire warp must call this function collectively (that is, the instructions are aligned)
  template <class TensorMapB>
  CUTLASS_DEVICE void tensormaps_fence_acquire(cute::tuple<TensorMapB> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
  }

  // Rebind the gathered A view (base pointer, gather-index slice, shape and
  // row pitch) to the next group. B's addressing is handled entirely by its
  // tensormap update, so its tile view is rebuilt unchanged from the same TMA
  // coord tensor. Called on batch change by the producer Mainloop warp AND by
  // the consumer warpgroups (which need the gathered gA view for mma_gather).
  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE InputTensors tensors_perform_update(InputTensors const& input_tensors,
                                                     Params const& mainloop_params,
                                                     ProblemShape_MNKL problem_shape_mnkl,
                                                     int32_t next_batch) {
    // Grouped: shape, row pitch, base pointer and index slice all change.
    // Ptr-array: shapes/strides are shared across batches, but A's base
    // pointer and gather-index slice still change, so rebuild either way.
    return load_init(problem_shape_mnkl, mainloop_params, next_batch);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
