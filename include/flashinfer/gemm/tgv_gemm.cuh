/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cassert>
#include <cstdio>
#include <iostream>

// Cutlass includes
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/grid_dependency_control.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include <cutlass/gemm/collective/builders/sm100_common.inl>  // mma/smem selector, umma::major

// CuTe includes
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/tmem_allocator_sm100.hpp>  // TMEM allocator for SM100
#include <cute/tensor.hpp>                     // CuTe tensor implementation

namespace flashinfer {
namespace gemm {

using namespace cute;

#ifndef gpuErrChk
#define gpuErrChk(ans)                     \
  {                                        \
    gpuAssert2((ans), __FILE__, __LINE__); \
  }
inline void gpuAssert2(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}
#endif

// copied from cutlass/arch/barrier.h, this is the non blocking version of cute::wait_barrier,
// here mbarrier.try_wait provides the best effort wait, when it completes (fail or succeed), it
// will return true/false subsequent instructions are not blocked on the try_wait, so mma and other
// instructions can continue to execute return true if the barrier is ready, false if the barrier is
// not ready
//
// cute::wait_barrier is the blocking version, when the mbarrier.try_wait finishes and fails, it
// will spin loop until the barrier is ready this means the bra instruction (in the implementation
// of cute::wait_barrier) after the try_wait will not be issued until the try_wait completes even if
// it returns true in the first call of try_wait, the bra instruction needs to wait for the result
// of the try_wait to complete (some cycles) and the instructions after the bra instruction (i.e.
// mma) will not be issued because bra is blocked on try_wait to complete
CUTLASS_DEVICE
static bool try_wait_barrier(uint64_t& smem_barrier, uint32_t phase) {
  uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&smem_barrier);
  uint32_t waitComplete;

  asm volatile(
      "{\n\t"
      ".reg .pred P1; \n\t"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2; \n\t"
      "selp.b32 %0, 1, 0, P1; \n\t"
      "}"
      : "=r"(waitComplete)
      : "r"(smem_int_ptr), "r"(phase));

  return static_cast<bool>(waitComplete);
}

// Helper methods to create layouts
// A always has the shape (M, K, L), L is the batch dimension
// it can has arbitrary stride as long as K or M is contiguous
template <int CTA_M, int CTA_K, cute::UMMA::Major UmmaMajorA>
auto make_layout_A(int m, int k, int l, int stride_m, int stride_k, int stride_l) {
  if constexpr (UmmaMajorA == cute::UMMA::Major::K) {
    assert(stride_k == 1);  // gemm/bmm requires K contiguous
    // for gemm, A is weight: (M,K,L):(K,_1,M*K) L=1
    // for bmm: A is weight: (M,K,L):(K,_1,M*K)
    return make_layout(make_shape(m, k, l), make_stride(stride_m, Int<1>{}, stride_l));
  } else {
    assert(stride_m == 1);  // gemm/bmm requires M contiguous
    // for gemm, A is weight: (M,K,L):(_1,M,M*K) L=1
    return make_layout(make_shape(m, k, l), make_stride(Int<1>{}, stride_k, stride_l));
  }
}

// B always has the shape (N, K, L), L is the batch dimension
// it can has arbitrary stride as long as K or N is contiguous
template <int CTA_N, int CTA_K, cute::UMMA::Major UmmaMajorB>
auto make_layout_B(int n, int k, int l, int stride_n, int stride_k, int stride_l) {
  if constexpr (UmmaMajorB == cute::UMMA::Major::K) {
    assert(stride_k == 1);  // gemm/bmm requires K contiguous
    // for gemm, B is activation: (N,K,L):(K,_1,N*K) L=1
    // for bmm: B is activation: (N,K,L):(L*K,_1,K)
    return make_layout(make_shape(n, k, l), make_stride(stride_n, Int<1>{}, stride_l));
  } else {
    assert(stride_n == 1);  // gemm/bmm requires N contiguous
    // for gemm, B is activation: (N,K,L):(_1,N,N*K) L=1
    return make_layout(make_shape(n, k, l), make_stride(Int<1>{}, stride_k, stride_l));
  }
}

// C always has the shape (M, N, L), L is the batch dimension
// it can has arbitrary stride as long as M is contiguous
template <int CTA_M, int CTA_N>
auto make_layout_C(int m, int n, int l, int stride_m, int stride_n, int stride_l) {
  assert(stride_m == 1);  // gemm/bmm requires M contiguous
  // for gemm, C is output: (M,N,L):(_1,M,M*N) L=1
  // for bmm: C is output: (M,N,L):(_1,M*L,M)
  return make_layout(make_shape(m, n, l), make_stride(Int<1>{}, stride_n, stride_l));
}

// Bias always has the shape (M) and broadcasted to (N, L), which means it's 1 element per M
// coordinate So we pretend it has the shape (M, N, L) same as C, but change the stride of N and L
// to 0 such that the underlying storage is still a size M vector i.e. the coordinate space is (M,
// N, L), but the underlying storage space is (M)
template <int CTA_M, int CTA_N>
auto make_layout_Bias(int m, int n, int l, int stride_m, int stride_n, int stride_l) {
  assert(stride_m == 1);  // gemm/bmm requires M contiguous
  // for gemm, (M,N,L):(_1,0,0) L=1
  // TODO: not supporting bmm
  return make_layout(make_shape(m, n, l), make_stride(Int<1>{}, Int<0>{}, Int<0>{}));
}

// simplified from cutlass/include/cutlass/gemm/kernel/static_tile_scheduler.hpp
// allow for easy extention to other types of tile scheduler
// it represents the cta will process this output tile
struct WorkTileInfo {
  int32_t M_idx = 0;
  int32_t N_idx = 0;
  int32_t L_idx = 0;
  bool is_valid_tile = false;

  CUTLASS_HOST_DEVICE
  bool is_valid() const { return is_valid_tile; }

  CUTLASS_HOST_DEVICE
  static WorkTileInfo invalid_work_tile() { return {-1, -1, -1, false}; }
};

// The shared memory buffers for A and B matrices.
template <class TypeA,        // Tensor A data type
          class TypeB,        // Tensor B data type
          class ASmemLayout,  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
          class BSmemLayout,  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
          int DMA_Stage>
struct SharedStorage {
  alignas(128) cute::ArrayEngine<
      TypeA, cute::cosize_v<ASmemLayout>> A;  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
  alignas(128) cute::ArrayEngine<
      TypeB, cute::cosize_v<BSmemLayout>> B;  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)

  alignas(16) cute::uint64_t
      tma_mma_full_barrier[DMA_Stage];  // Barrier between TMA and MMA, TMA tells MMA the tile is
                                        // ready/full, MMA can start consuming it
  alignas(16) cute::uint64_t
      tma_mma_empty_barrier[DMA_Stage];  // Barrier between MMA and TMA, MMA tells TMA the tile is
                                         // empty, TMA can start loading the next tile

  alignas(16) cute::uint64_t
      tma_epilog_full_barrier;  // Barrier between TMA_B and epilog, TMA_B tells epilog the
                                // activation loads are all issued, epilog can start loading bias if
                                // there is any
  alignas(16) cute::uint64_t
      mma_epilog_full_barrier;  // Barrier between MMA and epilog, MMA tells epilog the tile is
                                // ready/full, epilog can start consuming it
  // don't use it since we only support 1 mma stage for low latency, i.e. mma and epilog are
  // serialized
  // alignas(16) cute::uint64_t mma_epilog_empty_barrier;  // Barrier between epilog and MMA, epilog
  // tells MMA the tile is empty, MMA can start loading the next tile

  alignas(16) cute::uint32_t
      tmem_base_ptr;  // Base pointer for TMEM allocation, TMA will allocate TMEM here

  CUTE_DEVICE constexpr auto tensor_sA() {
    return make_tensor(make_smem_ptr(A.begin()), ASmemLayout{});
  }
  CUTE_DEVICE constexpr auto tensor_sB() {
    return make_tensor(make_smem_ptr(B.begin()), BSmemLayout{});
  }
};

template <class SharedStorage, class ATensor, class TmaAtomA, class TiledMMA, int CTA_M, int CTA_K,
          int DMA_Stage>
CUTLASS_DEVICE void DMA_A_warp(
    SharedStorage& shared_storage, WorkTileInfo work_tile_info, ATensor mA,
    // when passing tma descriptor as function argument, it has to be pass by pointer/reference, if
    // pass by value, it will live on local memory (i.e. the stack) and the tma unit cannot access
    // the local memory
    TmaAtomA const* tma_atom_A, TiledMMA tiled_mma, int pdl_count) {
  // exit warp if the tile is invalid
  if (!work_tile_info.is_valid()) {
    return;
  }

  // Represent the SMEM buffers for A, in the view of mma shape
  // Note: Partitioned tensors use tXgY naming convention:
  //  tXgY -> The partitioning pattern (give the same tensor a new view) tX applied to tensor gY
  //  tC means the tensor is partitioned into the mma shape, i.e. ((Mma_M, Mma_K), ..., ...)
  //  tA means the tensor is partitioned into the tma shape, i.e. (TMA, ..., ...)
  //  g means gmem, s means smem, t means tmem, r means rmem
  // so tCsA means the mma partitioned smem tensor sA
  Tensor tCsA = shared_storage.tensor_sA();  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)

  // now let's tile the gmem tensor first into the tile this CTA needs, which is (CTA_M, CTA_K,
  // Tiles_K) and Gemm_K = CTA_K * Tiles_K, Tiles_K is the number k block local_tile first tile mA
  // (Gemm_M, Gemm_K, Gemm_L) into (CTA_M, CTA_K, Tiles_M, Tiles_K, Gemm_L) the tiler (CTA_M, CTA_K)
  // applies to the first 2 modes of mA, i.e. M and K and it index into Tiles_M, Tiles_K and Gemm_L
  // mode with index work_tile_info.M_idx (i.e. blockIdx.x), _ and work_tile_info.L_idx (i.e.
  // blockIdx.z) so the result shape is (CTA_M, CTA_K, Tiles_K)
  Tensor gA = local_tile(
      mA, make_shape(Int<CTA_M>{}, Int<CTA_K>{}),
      make_coord(work_tile_info.M_idx, _, work_tile_info.L_idx));  // (CTA_M, CTA_K, Tiles_K)

  // now we want to partition (give the same tensor a new view) the gmem tensor into the mma shape,
  // to match the smem layout in sm100 mma can be shared by multiple cta, i.e. tcgen05.mma.2sm, so
  // we need to get the mma slice for this cta in cute's support for sm100, each thread is mma for a
  // cta, i.e. tcgen05.mma.1sm has 1 thread in tiled_mma, tcgen05.mma.2sm has 2 threads in tiled_mma
  // so we can get the mma slice for this cta by getting the slice of the tiled_mma
  ThrMMA cta_mma = tiled_mma.get_slice(0);  // 1 sm mma only has 1 thread in tiled_mma
  // tCgA means the mma partitioned gmem tensor gA
  Tensor tCgA = cta_mma.partition_A(gA);  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_K)

  // now we want to partition (give the same tensor a new view) the gmem tensor into the tma shape
  // in preparation for tma copy
  //   For A tensor: The group_modes<0,3> transforms the ((Mma_M, Mma_K), NumMma_M, NumMma_K,
  //   Tiles_K)-shaped tensor
  //      into (((Mma_M, Mma_K), NumMma_M, NumMma_K), Tiles_K). It groups mode [0, 3) of input into
  //      a single mode-0, The partitioning only pays attention to mode-0, the MMA Tile MK.
  //   Simply put, the TMA will be responsible for everything in mode-0 with a single call to
  //   cute::copy. The tma_partition reorders and offsets mode-0 according to the tma_x atom and the
  //   multicast info.
  auto [tAgA, tAsA] = tma_partition(*tma_atom_A, Int<0>{},  // cta_coord: 1x1 cga
                                    Layout<_1>{},  // cta_layout: CTA coord -> logical multicast id,
                                                   // no multicast, just identity layout
                                    group_modes<0, 3>(tCsA), group_modes<0, 3>(tCgA));
  // tAgA: ((TMA, NumTma_M, NumTma_K), Tiles_K) get coalesced to ((TMA, NumTma_K), Tiles_K) since
  // NumTma_M = 1 tAsA: ((TMA, NumTma_M, NumTma_K), DMA_Stage) get coalesced to ((TMA, NumTma_K),
  // DMA_Stage) since NumTma_M = 1 the shape of the TMA box is (CTA_M, 128B)

  // Calculate total bytes that TMA will transfer each tile to track completion in 1 DMA_Stage
  int tma_transaction_bytes = sizeof(make_tensor_like(tAsA(_, 0)));

  /*if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
      printf("cta_mma:\t"); print(cta_mma); print("\n");
      printf("tCsA:\t"); print(tCsA); print("\n");  // ((Mma_M, Mma_K), NumMma_M, NumMma_K,
  DMA_Stage) printf("gA:\t"); print(gA); print("\n");      // (CTA_M, CTA_K, Tiles_K)
      printf("tCgA:\t"); print(tCgA); print("\n");  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, Tiles_K)
      printf("tAgA:\t"); print(tAgA); print("\n");  // ((TMA, NumTma_K), Tiles_K)
      printf("tAsA:\t"); print(tAsA); print("\n");  // ((TMA, NumTma_K), DMA_Stage)
      printf("TmaBytes: %d\n", tma_transaction_bytes);
  }*/

  // initial phase bit for tma_mma_empty_barrier is 0, and it denotes smem slot is empty
  // we wait on the "old" phase bit of 1, when it is "flipped" to 0 (the initial phase bit), the
  // smem slot is empty so tma_mma_empty_barrier_phase_bit denotes the phase bit before the flip we
  // are waiting on example for DMA_Stage = 2, i.e. 2 smem slots, 2 empty barriers a and b kblock 0:
  // old phase bit of barrier a is 1, wait for barrier a's phase bit to change to 0 (the initial
  // phase bit), 0 denotes slot a is empty, 1 denotes slot a is used by mma kblock 1: old phase bit
  // of barrier b is 1, wait for barrier b's phase bit to change to 0 (the initial phase bit), 0
  // denotes slot b is empty, 1 denotes slot b is used by mma kblock 2: old phase bit of barrier a
  // is 0, wait for barrier a's phase bit to change to 1 (flipped once), 1 denotes slot a is empty,
  // 0 denotes slot a is used by mma kblock 3: old phase bit of barrier b is 0, wait for barrier b's
  // phase bit to change to 1 (flipped once), 1 denotes slot b is empty, 0 denotes slot b is used by
  // mma kblock 4: old phase bit of barrier a is 1, wait for barrier a's phase bit to change to 0
  // (flipped twice), 0 denotes slot a is empty, 1 denotes slot a is used by mma kblock 5: old phase
  // bit of barrier b is 1, wait for barrier b's phase bit to change to 0 (flipped twice), 0 denotes
  // slot b is empty, 1 denotes slot b is used by mma
  // ...
  int tma_mma_empty_barrier_phase_bit = 1;

  // iterate over kblock
  for (int k_tile = 0; k_tile < size<3>(tCgA); ++k_tile) {
    // wait_barrier's input argument is the old phase bit
    // it waits for the smem slot to be empty to start loading the next tile
    wait_barrier(shared_storage.tma_mma_empty_barrier[k_tile % DMA_Stage],
                 tma_mma_empty_barrier_phase_bit);

    // if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
    //     printf("[DMA_A] barrier empty, kblock %d\n", k_tile);
    // }

    if (elect_one_sync()) {
      // set the barrier transaction bytes to the number of bytes to load the tile, has to be done
      // by a single thread
      set_barrier_transaction_bytes(shared_storage.tma_mma_full_barrier[k_tile % DMA_Stage],
                                    tma_transaction_bytes);
      // load A tile into smem (CTA_M, CTA_K)
      copy(tma_atom_A->with(shared_storage.tma_mma_full_barrier[k_tile % DMA_Stage]),
           tAgA(_, k_tile), tAsA(_, k_tile % DMA_Stage));
    }

    // for every DMA_Stage number of iterations, we flip the phase bit such that we reuse the empty
    // barrier for another round of loading but now the meaning of the phase bit is flipped, example
    // with DMA_Stage = 2 kblock 0,1: 0 denotes slot is empty, 1 denotes slot is being used by mma
    // kblock 2,3: 1 denotes slot is empty, 0 denotes slot is being used by mma
    // kblock 4,5: 0 denotes slot is empty, 1 denotes slot is being used by mma
    // ...
    if ((k_tile % DMA_Stage) == (DMA_Stage - 1)) {
      tma_mma_empty_barrier_phase_bit ^= 1;
    }
    // do griddepcontrol.launch_dependents at a specific k_tile count
    if (k_tile == pdl_count) {
      cutlass::arch::launch_dependent_grids();
    }
  }

  // if pdl_count is -1, do griddepcontrol.launch_dependents at the end
  cutlass::arch::launch_dependent_grids();
}

template <class SharedStorage, class BTensor, class TmaAtomB, class TiledMMA, int CTA_N, int CTA_K,
          int DMA_Stage>
CUTLASS_DEVICE void DMA_B_warp(
    SharedStorage& shared_storage, WorkTileInfo work_tile_info, BTensor mB,
    // when passing tma descriptor as function argument, it has to be pass by pointer/reference, if
    // pass by value, it will live on local memory (i.e. the stack) and the tma unit cannot access
    // the local memory
    TmaAtomB const* tma_atom_B, TiledMMA tiled_mma) {
  // exit warp if the tile is invalid
  if (!work_tile_info.is_valid()) {
    return;
  }

  // similar to DMA_A_warp
  Tensor tCsB = shared_storage.tensor_sB();  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
  Tensor gB = local_tile(
      mB, make_shape(Int<CTA_N>{}, Int<CTA_K>{}),
      make_coord(work_tile_info.N_idx, _, work_tile_info.L_idx));  // (CTA_N, CTA_K, Tiles_K)
  ThrMMA cta_mma = tiled_mma.get_slice(0);  // 1 sm mma only has 1 thread in tiled_mma
  Tensor tCgB = cta_mma.partition_B(gB);    // ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_K)
  auto [tBgB, tBsB] = tma_partition(*tma_atom_B, Int<0>{},  // cta_coord: 1x1 cga
                                    Layout<_1>{},  // cta_layout: CTA coord -> logical multicast id,
                                                   // no multicast, just identity layout
                                    group_modes<0, 3>(tCsB), group_modes<0, 3>(tCgB));
  // tBgB: ((TMA, NumTma_N, NumTma_K), Tiles_K) get coalesced to ((TMA, NumTma_K), Tiles_K) since
  // NumTma_N = 1 tBsB: ((TMA, NumTma_N, NumTma_K), DMA_Stage) get coalesced to ((TMA, NumTma_K),
  // DMA_Stage) since NumTma_N = 1

  int tma_transaction_bytes = sizeof(make_tensor_like(tBsB(_, 0)));

  /*if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
      printf("cta_mma:\t"); print(cta_mma); print("\n");
      printf("tCsB:\t"); print(tCsB); print("\n");  // ((Mma_N, Mma_K), NumMma_N, NumMma_K,
  DMA_Stage) printf("gB:\t"); print(gB); print("\n");      // (CTA_N, CTA_K, Tiles_K)
      printf("tCgB:\t"); print(tCgB); print("\n");  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, Tiles_K)
      printf("tBgB:\t"); print(tBgB); print("\n");  // ((TMA, NumTma_K), Tiles_K)
      printf("tBsB:\t"); print(tBsB); print("\n");  // ((TMA, NumTma_K), DMA_Stage)
      printf("TmaBytes: %d\n", tma_transaction_bytes);
  }*/

  int tma_mma_empty_barrier_phase_bit = 1;

  // only do griddepcontrol.wait on B loading, B is activation
  cutlass::arch::wait_on_dependent_grids();

  // iterate over kblock
  for (int k_tile = 0; k_tile < size<3>(tCgB); ++k_tile) {
    wait_barrier(shared_storage.tma_mma_empty_barrier[k_tile % DMA_Stage],
                 tma_mma_empty_barrier_phase_bit);

    // if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
    //     printf("[DMA_B] barrier empty, kblock %d\n", k_tile);
    // }

    if (elect_one_sync()) {
      set_barrier_transaction_bytes(shared_storage.tma_mma_full_barrier[k_tile % DMA_Stage],
                                    tma_transaction_bytes);
      // load B tile into smem (CTA_N, CTA_K)
      copy(tma_atom_B->with(shared_storage.tma_mma_full_barrier[k_tile % DMA_Stage]),
           tBgB(_, k_tile), tBsB(_, k_tile % DMA_Stage));
    }

    if ((k_tile % DMA_Stage) == (DMA_Stage - 1)) {
      tma_mma_empty_barrier_phase_bit ^= 1;
    }
  }

  // all activation/B loads are issued, can issue bias load now, signal the epilog warp
  arrive_barrier(shared_storage.tma_epilog_full_barrier);
}

template <class SharedStorage, class CTensor, class TiledMMA, int CTA_M, int CTA_N, int DMA_Stage>
CUTLASS_DEVICE void MMA_warp(SharedStorage& shared_storage, WorkTileInfo work_tile_info, CTensor mC,
                             TiledMMA tiled_mma, int k_tile_count,
                             cutlass::arch::NamedBarrier& tmem_allocation_result_barrier) {
  // exit warp if the tile is invalid
  if (!work_tile_info.is_valid()) {
    return;
  }

  // get the local slice of C
  Tensor gC = local_tile(mC, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
                         make_coord(work_tile_info.M_idx, work_tile_info.N_idx,
                                    work_tile_info.L_idx));  // (CTA_M, CTA_N)

  // smem tensor prepared by DMA_A and DMA_B
  Tensor tCsA = shared_storage.tensor_sA();  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
  Tensor tCsB = shared_storage.tensor_sB();  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)

  // MMA Fragment Allocation
  // We allocate "fragments" which are SMEM descriptors that serve as inputs to cute::gemm
  // operations. For tcgen05.mma operations:
  // - Matrices A and B are sourced from SMEM
  // - tCrA and tCrB provide descriptor views of tCsA and tCsB respectively
  // - The first mode of each descriptor represents the SMEM for a single MMA operation
  ThrMMA cta_mma = tiled_mma.get_slice(0);      // 1 sm mma only has 1 thread in tiled_mma
  Tensor tCrA = cta_mma.make_fragment_A(tCsA);  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
  Tensor tCrB = cta_mma.make_fragment_B(tCsB);  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
  // then partition it to mma shape
  Tensor tCgC = cta_mma.partition_C(gC);  // ((Mma_M, Mma_N), NumMma_M, NumMma_N)

  // TMEM Allocation
  // On SM100 architecture, accumulators are stored exclusively in tensor memory (TMEM).
  // ThrMma's make_fragment_C() creates a TMEM tensor with the appropriate layout for the
  // accumulator. tCtAcc is a view of the accumulator tensor (it doesn't really hold any actual
  // value) it's currently empty, it has another member which is tmem base ptr which is unset rn
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);  // ((Mma_M, Mma_N), NumMma_M, NumMma_N)

  using TmemAllocator = cute::TMEM::Allocator1Sm;
  TmemAllocator tmem_allocator{};
  // only use half of tmem to allow overlapping
  // tmem has 128 lane, 512 column, each word is 4B, 256KB in total, our accumulator can only use 64
  // lanes if CTA_M=64
  static_assert(CTA_N * 4 < TmemAllocator::Sm100TmemCapacityColumns * 4 / 2,
                "Accumulator is too large to fit in half of tmem");
  tmem_allocator.allocate(TmemAllocator::Sm100TmemCapacityColumns / 2,
                          &shared_storage.tmem_base_ptr);
  // notify epilog warp that tmem allocation is complete
  tmem_allocation_result_barrier.arrive();

  // relinquish early so that prefetch cta can be launched
  tmem_allocator.release_allocation_lock();

  // update the tmem base ptr of the accumulator tensor
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  /*if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
      printf("gC:\t"); print(gC); print("\n");          // (CTA_M, CTA_N)
      printf("tCgC:\t"); print(tCgC); print("\n");      // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
      printf("tCrA:\t"); print(tCrA); print("\n");      // ((Mma_M, Mma_K), NumMma_M, NumMma_K,
  DMA_Stage) printf("tCrB:\t"); print(tCrB); print("\n");      // ((Mma_N, Mma_K), NumMma_N,
  NumMma_K, DMA_Stage) printf("tCtAcc:\t"); print(tCtAcc); print("\n");  // ((Mma_M, Mma_N),
  NumMma_M, NumMma_N) printf("tmem_base_ptr:\t%d\n", shared_storage.tmem_base_ptr);
  }*/

  // initial phase bit for tma_mma_full_barrier is 0
  // example for DMA_Stage = 2, i.e. 2 smem slots, 2 full barriers a and b
  // kblock 0: old phase bit of barrier a is 0, wait for barrier a's phase bit to change to 1, 1
  // denotes slot a is full, 0 denotes slot a is not ready kblock 1: old phase bit of barrier b is
  // 0, wait for barrier b's phase bit to change to 1, 1 denotes slot b is full, 0 denotes slot b is
  // not ready kblock 2: old phase bit of barrier a is 1, wait for barrier a's phase bit to change
  // to 0, 0 denotes slot a is full, 1 denotes slot a is not ready kblock 3: old phase bit of
  // barrier b is 1, wait for barrier b's phase bit to change to 0, 0 denotes slot b is full, 1
  // denotes slot b is not ready kblock 4: old phase bit of barrier a is 0, wait for barrier a's
  // phase bit to change to 1, 1 denotes slot a is full, 0 denotes slot a is not ready kblock 5: old
  // phase bit of barrier b is 0, wait for barrier b's phase bit to change to 1, 1 denotes slot b is
  // full, 0 denotes slot b is not ready
  // ...
  int tma_mma_full_barrier_phase_bit = 0;

  // Set mma accumulate option to zero so that the first MMA instruction will clear the TMEM
  // accumulator. UMMA::ScaleOut::Zero means C = A * B UMMA::ScaleOut::One means C += A * B
  tiled_mma.accumulate_ = UMMA::ScaleOut::Zero;

  // this is not sol mma loop implementation, put it here for educational purpose since it's more
  // clean
#if 0
    // explicitly manage the stage_idx instead of just doing k_tile % DMA_Stage, this is because we are mainloop latency bound, when DMA_Stage is non power of 2,
    // the mode will take long in the mainloop, so we get rid of the mod to make the mainloop tighter
    // this makes non power of 2 DMA_Stage faster, but makes power of 2 DMA_Stage (e.g. 16) slightly slower (2.5%) since moding power of 2 is free
    int stage_idx = 0;
    // iterate over kblock
    for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
        // wait for tma_mma_full_barrier to be full
        wait_barrier(shared_storage.tma_mma_full_barrier[stage_idx], tma_mma_full_barrier_phase_bit);

        //if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
        //    printf("[MMA] barrier full, kblock %d\n", k_tile);
        //}

        // tcgen05.mma instructions require single-thread execution:
        // - Only one warp performs the MMA-related loop operations
        // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
        // - No explicit elect_one_sync region is needed from the user
        // manually unroll the NumMma_K loop, such that each cute::gemm is issuing exactly one mma instruction
        // and the accumulate option is set to one after the first mma instruction
        // the benefit of this is we avoid explicit zeroing the accumulator in the prolog, but the code looks longer
        for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
            // execute a Mma_M x Mma_N x Mma_K GEMM
            cute::gemm(tiled_mma, tCrA(_,_,k_block, stage_idx), tCrB(_,_,k_block, stage_idx), tCtAcc);
            // after the first mma instruction, we need to start accumulating the result
            // i.e. basically do C += A * B
            tiled_mma.accumulate_ = UMMA::ScaleOut::One;
        }
        // notify the DMA warp that the all the MMA issued prior to this tcgen05.commit is done, the smem slot is now empty and can be reused
        cutlass::arch::umma_arrive(&shared_storage.tma_mma_empty_barrier[stage_idx]);

        stage_idx++;
        // for every DMA_Stage number of iterations, we flip the phase bit such that we reuse the full barrier for another round of mma
        if (stage_idx == DMA_Stage) {
            tma_mma_full_barrier_phase_bit ^= 1;
            stage_idx = 0;
        }
    }
#else

  // the above mma loop is not sol latency as in the wait_barrier will take some cycles and a spin
  // loop, blocking subsequent instructions to issue, the book keeping instruction for tcgen05.mma
  // can't overlap with wait_barrier we follow the cutlass mma pattern
  // cutlass/include/cutlass/gemm/collective/sm100_mma_warpspecialized.hpp to overlap wait_barrier
  // with mma
  //
  //   for each kblock
  //     if not predicate
  //       try wait (blocking, spin loop)
  //     update stage idx
  //     try wait for next iter, and return predicate
  //     mma
  //
  // recommended to use ptxas (13.1) to get max perf
  int stage_idx = 0;
  int old_stage_idx = 0;
  // peel off the first iteration, try wait for the first stage
  bool waitComplete = try_wait_barrier(shared_storage.tma_mma_full_barrier[stage_idx],
                                       tma_mma_full_barrier_phase_bit);
  // iterate over kblock
  for (int k_tile = 0; k_tile < k_tile_count; ++k_tile) {
    // wait for tma_mma_full_barrier to be full
    if (!waitComplete) {
      wait_barrier(shared_storage.tma_mma_full_barrier[stage_idx], tma_mma_full_barrier_phase_bit);
    }

    // update stage idx
    old_stage_idx = stage_idx;
    stage_idx++;
    // for every DMA_Stage number of iterations, we flip the phase bit such that we reuse the full
    // barrier for another round of mma
    if (stage_idx == DMA_Stage) {
      tma_mma_full_barrier_phase_bit ^= 1;
      stage_idx = 0;
    }
    // try wait for the next iteration
    if (k_tile < (k_tile_count - 1)) {
      waitComplete = try_wait_barrier(shared_storage.tma_mma_full_barrier[stage_idx],
                                      tma_mma_full_barrier_phase_bit);
    }

    // if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
    //     printf("[MMA] barrier full, kblock %d\n", k_tile);
    // }

    // tcgen05.mma instructions require single-thread execution:
    // - Only one warp performs the MMA-related loop operations
    // - CuTe operations internally manage the single-thread execution of tcgen05.mma and tcgen05.cp
    // - No explicit elect_one_sync region is needed from the user
    // manually unroll the NumMma_K loop, such that each cute::gemm is issuing exactly one mma
    // instruction and the accumulate option is set to one after the first mma instruction the
    // benefit of this is we avoid explicit zeroing the accumulator in the prolog, but the code
    // looks longer
    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // execute a Mma_M x Mma_N x Mma_K GEMM
      cute::gemm(tiled_mma, tCrA(_, _, k_block, old_stage_idx), tCrB(_, _, k_block, old_stage_idx),
                 tCtAcc);
      // after the first mma instruction, we need to start accumulating the result
      // i.e. basically do C += A * B
      tiled_mma.accumulate_ = UMMA::ScaleOut::One;
    }
    // notify the DMA warp that the all the MMA issued prior to this tcgen05.commit is done, the
    // smem slot is now empty and can be reused
    cutlass::arch::umma_arrive(&shared_storage.tma_mma_empty_barrier[old_stage_idx]);
  }
#endif

  // notify the epilog warp that the MMA is done, the tmem slot is now full and can be consumed
  cutlass::arch::umma_arrive(&shared_storage.mma_epilog_full_barrier);
}

template <class SharedStorage, class CTensor, class BiasTensor, class TiledMMA, int CTA_M,
          int CTA_N, bool NO_BIAS>
CUTLASS_DEVICE void EPILOG_warp(SharedStorage& shared_storage, WorkTileInfo work_tile_info,
                                CTensor mC, BiasTensor mBias, TiledMMA tiled_mma,
                                cutlass::arch::NamedBarrier& tmem_allocation_result_barrier) {
  // exit warp if the tile is invalid
  if (!work_tile_info.is_valid()) {
    return;
  }

  // get the local slice of C
  Tensor gC = local_tile(mC, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
                         make_coord(work_tile_info.M_idx, work_tile_info.N_idx,
                                    work_tile_info.L_idx));  // (CTA_M, CTA_N)
  ThrMMA cta_mma = tiled_mma.get_slice(0);  // 1 sm mma only has 1 thread in tiled_mma
  Tensor tCgC = cta_mma.partition_C(gC);    // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
  // since tCtAcc is a view of the accumulator tensor, it's safe to create a new view in the epilog
  // warp too
  //
  // the code is at cutlass/include/cute/atom/mma_traits_sm100.hpp, under struct tmem_frg
  // for the example mma size of bf16 M64 N32 K16, the layout of tCtAcc is:
  // tCtAcc:  tmem_[32b](0x0000.0000) o (((_16,_4),_32),_1,_1):(((_65536,_2097152),_1),_0,_0)
  // this represents ((Mma_M, Mma_N), NumMma_M, NumMma_N), so NumMma_M = 1, NumMma_N = 1
  // Mma_M = (16, 4) = (Mma_M_per_subp, NumSubp), Mma_N = 32, there are 4 subpartitions, and
  // Mma_M=64 means per subpartition we only use 16 lane of tmem now let's look at the stride, tmem
  // is always N major, so Mma_N's stride is 1 tmem addr is [31:16->lane, 15:0->column], and for
  // B200 specifically there is 128 lane, 512 column, each word is 4B so the stride between 2
  // consecutive lane lane is 16bit=65536B, which explains the stride of Mma_M_per_subp and the
  // stride between 2 subpartitions is 65536*32dp=2097152B, which explains the stride of NumSubp
  Tensor tCtAcc = cta_mma.make_fragment_C(tCgC);  // ((Mma_M, Mma_N), NumMma_M, NumMma_N)

  // wait for tmem allocation to complete
  tmem_allocation_result_barrier.arrive_and_wait();

  // update tmem base ptr of the accumulator tensor
  tCtAcc.data() = shared_storage.tmem_base_ptr;

  // Create the tiled copy operation for the accumulator (TMEM -> RMEM)
  // according to the cutlass epilog reference code sm100_get_tmem_load_op() in
  // cutlass/include/cutlass/epilogue/collective/builders/sm100_builder.inl for M=64, M/N major
  // output, we want SM100_TMEM_LOAD_16dp256b1x we need to use the 16dp version because M=64, each
  // sub partition uses 16 lane of tmem in mma, the other 16 lane of tmem is not used, there are 32
  // lane per sub partition other tcgen05.ld layout works too, the code will always functional, it's
  // just a matter of good/bad performance
  TiledCopy tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_16dp256b1x{}, tCtAcc);
  // epilog tid is from 128 to 255, need to offset by -128 when getting the per thread slice
  ThrCopy thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x - 128);

  // tD means the partitioning pattern of tcgen05.ld
  // now we get the per thread slice of the accumulator tensor
  //
  // for the example mma size of bf16 M64 N32 K16, and tcgen05.ld atom of
  // SM100_TMEM_LOAD_16dp256b1x, the layout of tDtAcc is: tCtAcc:  tmem_[32b](0x0000.0000) o
  // (((_16,_4),_32),_1,_1):(((_65536,_2097152),_1),_0,_0) tDtAcc:  tmem_[32b](0x0000.0000) o
  // (((_8,_16),_1),_4,_1,_1):(((_1,_65536),_0),_8,_0,_0) basically the source partitioning does
  // ((16, 4), 32) : ((65536, 2097152), 1) -> (((8, 16), 1), 4) : (((1, 65536), 0), 8) somehow
  // tDtAcc is the per subpartition slice of tCtAcc, and the shape is (((8, 16), 1), 4) = (((CpyS_N,
  // CpyS_M), NumCpy_M), NumCpy_N) with CpyS_N=8, CpyS_M=16=Mma_M_per_subp, NumCpy_M=1, NumCpy_N=4,
  // Mma_N = CpyS_N * NumCpy_N = 32 the stride is pretty self explanatory, the stride between 2
  // consecutive dp lane is 16bit=65536B, which explains the stride of CpyS_M and since N is
  // contiguous, the stride is 1 according to the ptx page of tcgen05.ld.16dp256bit, the (((CpyS_N,
  // CpyS_M), NumCpy_M), NumCpy_N) shape is easy to explain (CpyS_N, CpyS_M) is the amount of data
  // per tcgen05.ld.16dp256bit.x1 instruction can copy, which is 16lane * 8col NumCpy_N is the
  // number of times we want to replicate this (CpyS_N, CpyS_M) atom along the N dimension, since
  // NumCpy_N=4, we replicate 4 times
  //
  // alternatively, we can use x4 version of the tcgen05.ld.16dp256bit instruction suffix instead of
  // x1, this will replicate (CpyS_N, CpyS_M) 4 times along the N dimension effectively creating a
  // copy atom of shape (CpyS_N * 4, CpyS_M)
  Tensor tDtAcc = thr_t2r_copy.partition_S(
      tCtAcc);  // (CpyS, NumCpy_M, NumCpy_N), per subpartition view of the accumulator tensor
  // we want to partition tCgC for 2 reasons:
  // 1. we just want to get its post partition shape for allocating rmem space for the accumulator
  // 2. we want partition it for storing back the result to gmem, tDgC is the per thread slice of
  // the output tensor
  //
  // for the example mma size of bf16 M64 N32 K16, and tcgen05.ld atom of
  // SM100_TMEM_LOAD_16dp256b1x, C is row major, the layouts are: tCgC:
  // gmem_ptr[32b](0x7f5897000000) o ((_64,_32),_1,_1):((32,_1),_0,_0) tDgC:
  // gmem_ptr[32b](0x7f5897000000) o (((_2,_2),_1),_4,_1,_1):(((_1,256),_0),_8,_0,_0) tCgC just
  // describes the layout of the output tile this cta is responsible for, for different tiles, the
  // layout is the same, the only difference is the base addr tDgC represents which value in tCgC is
  // stored in this thread's rmem if we were doing tcgen05.ld from tmem to gmem according to the ptx
  // page of tcgen05.ld.16dp256bit, for thread 0, each tcgen05.ld will fill 4 registers, and the
  // coordinates (in output tile) of the 4 registers are: (0, 0), (0, 1), (8, 0), (8, 1). this
  // explains the layout tDgC (2,2) : (1,256) represents the 4 registers CpyD=(2,2) similarly, the
  // value 4 in tDgC's shape represents we replicate tcgen05.ld.16dp256bit 4 (NumCpy_N) times along
  // the N dimension with stride 8, each copy would fill 4 registers so tDgC's layout represents the
  // mapping between register idx -> gmem addr but we want the destination to be in rmem, so we want
  // a layout of register idx -> rmem addr we do this by allocating a rmem tensor tDrAcc
  // make_tensor<AccType> with the shape of tDgC tDrAcc:  ptr[32b](0x7f83e5fffc20) o
  // (((_2,_2),_1),_4,_1,_1):(((_1,_2),_0),_4,_0,_0) we can see that the shape is the same as tDgC,
  // but the stride is different, and the (2, 2) CpyD atom are stored contiguously in rmem with
  // stride (1, 2) now tDgC and tDrAcc has the same shape, meaning they have the same register idx
  // coordinate space
  Tensor tDgC = thr_t2r_copy.partition_D(
      tCgC);  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the output tensor
  using AccType = typename decltype(tCtAcc)::value_type;
  // allocate per thread rmem space for the accumulator, the shape is the same as the post partition
  // shape of the output tensor
  Tensor tDrAcc = make_tensor<AccType>(
      shape(tDgC));  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the accumulator tensor

  // now we do the data type conversion from fp32 to bf16/fp8, following the basic sm100 no smem
  // epilog code in cutlass/include/cutlass/epilogue/collective/sm100_epilogue_nosmem.hpp
  using TypeC = typename decltype(gC)::value_type;
  // allocate per thread rmem space for the converted output tensor, the shape is the same as the
  // post partition shape of the output tensor the only difference with tDrAcc is here per element
  // is TypeC instead of AccType
  Tensor tDrC = make_tensor<TypeC>(
      shape(tDgC));  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the converted output tensor
  // create a converter, convert 1 value at a time, from AccType to TypeC
  cutlass::NumericConverter<TypeC, AccType> converter;

  // finally we need to calculate the predicate for rmem->gmem store, because the problem shape may
  // not be a multiple of rmem/smem tile size there might be out of bounds access to gmem if we
  // don't predicate the store properly if you use tma that oob is automatically handled, you don't
  // need to predicate, for ld.global/cp.async/st.global, you need to predicate explicitly
  //
  // an identity tensor is a tensor that has the same shape as the output tensor, and it maps the
  // (m, n, l) coordinates to the payload which is a tuple of (m, n, l) so coordC(m, n, l) = (m, n,
  // l)
  Tensor coordC = make_identity_tensor(shape(mC));  // (M,N,L) -> (m,n,l)
  // create the local tile of coordC, similar to gC
  Tensor cC = local_tile(coordC, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
                         make_coord(work_tile_info.M_idx, work_tile_info.N_idx,
                                    work_tile_info.L_idx));  // (CTA_M, CTA_N)
  // partition it for the mma shape to allow thr_t2r_copy happy, otherwise we can't directly
  // partition cC?
  Tensor tCcC = cta_mma.partition_C(cC);  // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
  // tDcC has the same shape as tDgC/tDrAcc/tDrC, except its payload is the coordinate of the output
  // tensor
  Tensor tDcC = thr_t2r_copy.partition_D(
      tCcC);  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the output tensor
  // now we create the predicate tensor of the same shape
  Tensor tDpC = make_tensor<bool>(shape(tDcC));  // (CpyD, NumCpy_M, NumCpy_N)

  // construct the predicate tensor
  CUTLASS_PRAGMA_UNROLL
  for (int t = 0; t < size(tDpC); t++) {
    // compare the current coordinate with the problem shape (M, N, L), if it's out of bounds, the
    // predicate is false
    tDpC(t) = elem_less(tDcC(t), shape(mC));
  }

  // support the optional bias, since this code is not on the critical path, we leave it here even
  // if we don't have bias we do the same procedure again, but for the bias tensor get the local
  // slice of bias
  Tensor gBias = local_tile(mBias, make_shape(Int<CTA_M>{}, Int<CTA_N>{}),
                            make_coord(work_tile_info.M_idx, work_tile_info.N_idx,
                                       work_tile_info.L_idx));  // (CTA_M, CTA_N)
  Tensor tCgBias = cta_mma.partition_C(gBias);  // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
  // tDgBias has the same shape as tDgC/tDrAcc/tDrC
  Tensor tDgBias = thr_t2r_copy.partition_D(
      tCgBias);  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the output tensor
  using TypeBias = typename decltype(tDgBias)::value_type;
  // allocate per thread rmem space for the bias tensor, the shape is the same as the post partition
  // shape of the bias tensor
  Tensor tDrBias = make_tensor<TypeBias>(
      shape(tDgBias));  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the bias tensor
  // create a converter for bias, convert 1 value at a time, from TypeBias to AccType
  cutlass::NumericConverter<AccType, TypeBias> converterBias;
  // allocate per thread rmem space for the converted (to AccType) bias tensor, the shape is the
  // same as the post partition shape of the bias tensor
  Tensor tDrBiasAcc = make_tensor<AccType>(
      shape(tDgBias));  // (CpyD, NumCpy_M, NumCpy_N), per thread slice of the converted bias tensor

  /*if ((threadIdx.x == 128) && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
      printf("tiled_t2r_copy:\t"); print(tiled_t2r_copy); print("\n");
      printf("thr_t2r_copy:\t"); print(thr_t2r_copy); print("\n");
      printf("gC:\t"); print(gC); print("\n");           // (CTA_M, CTA_N)
      printf("tCgC:\t"); print(tCgC); print("\n");       // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
      printf("tCtAcc:\t"); print(tCtAcc); print("\n");   // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
      printf("tDtAcc:\t"); print(tDtAcc); print("\n");   // (CpyS, NumCpy_M, NumCpy_N)
      printf("tDgC:\t"); print(tDgC); print("\n");       // (CpyD, NumCpy_M, NumCpy_N)
      printf("tDrAcc:\t"); print(tDrAcc); print("\n");   // (CpyD, NumCpy_M, NumCpy_N)
      printf("tDrC:\t"); print(tDrC); print("\n");       // (CpyD, NumCpy_M, NumCpy_N)

      printf("tmem_base_ptr:\t%d\n", shared_storage.tmem_base_ptr);
      printf("TypeC: %llu\n", sizeof(TypeC));
      printf("AccType: %llu\n", sizeof(AccType));

      printf("coordC:\t"); print(coordC); print("\n");   // (M,N,L) -> (m,n,l)
      printf("cC:\t"); print(cC); print("\n");           // (CTA_M, CTA_N)
      printf("tCcC:\t"); print(tCcC); print("\n");       // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
      printf("tDcC:\t"); print(tDcC); print("\n");       // (CpyD, NumCpy_M, NumCpy_N)
      printf("tDpC:\t"); print(tDpC); print("\n");       // (CpyD, NumCpy_M, NumCpy_N)

      printf("gBias:\t"); print(gBias); print("\n");     // (CTA_M, CTA_N)
      printf("tCgBias:\t"); print(tCgBias); print("\n"); // ((Mma_M, Mma_N), NumMma_M, NumMma_N)
      printf("tDgBias:\t"); print(tDgBias); print("\n"); // (CpyD, NumCpy_M, NumCpy_N)
      printf("tDrBias:\t"); print(tDrBias); print("\n"); // (CpyD, NumCpy_M, NumCpy_N)
      printf("tDrBiasAcc:\t"); print(tDrBiasAcc); print("\n"); // (CpyD, NumCpy_M, NumCpy_N)
  }*/

  // initial phase bit for tma_epilog_full_barrier is 0
  // it is flipped to 1 when the TMA_B warp is done, and now we can start the bias load
  int tma_epilog_full_barrier_phase_bit = 0;
  // wait for the TMA_B warp to finish
  wait_barrier(shared_storage.tma_epilog_full_barrier, tma_epilog_full_barrier_phase_bit);

  // if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
  //     printf("[BIAS LOAD] barrier full\n");
  // }

  // now we can start the bias load
  if constexpr (!NO_BIAS) {
    // 1. load bias into rmem from gmem
    // use copy_if to predicate the load, to handle oob load from bias tensor
    copy_if(tDpC, tDgBias, tDrBias);
    // 2. convert the bias data from TypeBias to AccType
    CUTE_UNROLL
    for (int i = 0; i < tDrBias.size(); i++) {
      tDrBiasAcc[i] = converterBias(tDrBias[i]);
    }
  }

  // initial phase bit for mma_epilog_full_barrier is 0
  // it is flipped to 1 when the MMA warp is done, and now we can start the epilog
  int mma_epilog_full_barrier_phase_bit = 0;
  // wait for the MMA warp to finish
  wait_barrier(shared_storage.mma_epilog_full_barrier, mma_epilog_full_barrier_phase_bit);

  // if (elect_one_sync() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
  //     printf("[EPILOG] barrier full\n");
  // }

  // now we can start the epilog
  // 3. so we have the tmem->rmem copy source being tDtAcc (CpyS, NumCpy_M, NumCpy_N) and the copy
  // destination being tDgC (CpyD, NumCpy_M, NumCpy_N) the copy operation is basically: for each
  // NumCpy_M:
  //   for each NumCpy_N:
  //     tcgen05.ld.16dp256bit.x1 CpyS -> CpyD
  //
  // each tcgen05.ld instruction copys 16lane * 8col (CpyS) for each tmem subpartition to 128
  // thread's rmem (4 (CpyD) registers per thread)
  //
  // load tmem -> rmem
  copy(tiled_t2r_copy, tDtAcc, tDrAcc);

  // 4. optionally accumulate the bias with accumulator
  if constexpr (!NO_BIAS) {
    // accumulate the bias with accumulator
    CUTE_UNROLL
    for (int i = 0; i < tDrAcc.size(); i++) {
      tDrAcc[i] += tDrBiasAcc[i];
    }
  }

  // 5. convert the output data from AccType to TypeC
  CUTE_UNROLL
  for (int i = 0; i < tDrC.size(); i++) {
    tDrC[i] = converter(tDrAcc[i]);
  }

  // 6. now tDrC holds the converted output data in rmem, we finally need to copy it to gmem
  // recall that from above tDrAcc/tDrC and tDgC has the same shape, meaning they have the same
  // register idx coordinate space the copy operation is simply: for m in NumCpy_M:
  //   for n in NumCpy_N:
  //     for idx in register_idx:
  //       if tDpC(idx, m, n):
  //         tDgC(idx, m, n) = tDrC(idx, m, n) // this uses st.global
  // this rmem->gmem store is unfortunately very uncoalesced because the gmem addr across threads
  // are not contiguous but we hope it's fine since we run small batch size use copy_if to predicate
  // the store, to handle oob store to C
  //
  // store rmem -> gmem
  copy_if(tDpC, tDrC, tDgC);
}

// A * B = C
template <class SharedStorage, class ATensor, class BTensor, class CTensor, class BiasTensor,
          class TmaAtomA, class TmaAtomB, class TiledMMA, int CTA_M, int CTA_N, int CTA_K,
          int DMA_Stage, bool NO_BIAS>
__global__ void tgv_gemm_device(ATensor mA, BTensor mB, CTensor mC, BiasTensor mBias,
                                CUTE_GRID_CONSTANT TmaAtomA const tma_atom_A,
                                CUTE_GRID_CONSTANT TmaAtomB const tma_atom_B, TiledMMA tiled_mma,
                                int pdl_count) {
  // if (threadIdx.x == 0) {
  //     printf("[%d, %d, %d] gemm_device\n", blockIdx.x, blockIdx.y, blockIdx.z);
  // }

  // WorkTileInfo, for non persistent static scheduler, cta id is the work tile info
  WorkTileInfo work_tile_info{.M_idx = (int32_t)blockIdx.x,
                              .N_idx = (int32_t)blockIdx.y,
                              .L_idx = (int32_t)blockIdx.z,
                              .is_valid_tile = true};

  // Allocate SMEM
  extern __shared__ char shared_memory[];
  SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(shared_memory);

  int warp_idx = cutlass::canonical_warp_idx_sync();

  // there are 2 kinds of barriers: named barrier and mbarrier barrier
  // 1. named barrier dates back to ampere, it's the bar.arv/bar.sync instruction, 16 hw barrier in
  // hopper sm
  //    it handles synchronization between threads in a cta and nothing else
  // 2. mbarrier barrier is new in hopper, it's mbarrier.phasechk/mbarrier.arrive instruction, it's
  // 64bit in smem per barrier,
  //    it is more sw programmable than named barrier
  //    it can be used for multiple purposes
  //    (a) synchronization between threads in a cta like named barrier
  //    (b) synchronization within a CGA
  //    (c) support transaction count, i.e. synchronization between TMA and SM
  //
  // phase initialize to 0, expected arrive count is 1, arrival count initialize to 0
  // set transaction byte will arrive (arrive count +1) at the barrier and set transaction byte
  // dram request will increment the transaction byte to reach the expected tx count
  // when both the tx count and arrive count reach the expected value, the barrier flipped to phase
  // 1 arrive count reset to 0, waiting for the next transaction byte set
  //
  // if you want 2 DMA warp to use the same barrier to signal MMA warp, you can set expected arrive
  // count to 2 and let each DMA warp set the barrier transaction bytes to the number of bytes for
  // that warp
  //
  // vectorize the barrier initialization, only warp 0 does initialization
  if (warp_idx == 0) {
    // only transaction barrier because tma arrive on it, 2 thread (tma) arrive one for DMA_A warp
    // and one for DMA_B warp
    cutlass::arch::detail::initialize_barrier_array_aligned<
        cutlass::arch::ClusterTransactionBarrier, DMA_Stage>(shared_storage.tma_mma_full_barrier,
                                                             /* arrival count */ 2);
    // 1 thread (mma) arrive to signal DMA_A and DMA_B warp
    cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier,
                                                            DMA_Stage>(
        shared_storage.tma_mma_empty_barrier, /* arrival count */ 1);
    // 32 thread/1 warp (tma_B) arrive to signal epilog
    cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier, 1>(
        &shared_storage.tma_epilog_full_barrier, /* arrival count */ 32);
    // 1 thread (mma) arrive to signal epilog
    cutlass::arch::detail::initialize_barrier_array_aligned<cutlass::arch::ClusterBarrier, 1>(
        &shared_storage.mma_epilog_full_barrier, /* arrival count */ 1);
  }
  // Sync tmem allocation status between MMA and epilogue warps within CTA
  // 32 threads (mma) + 128 threads (epilog) to sync
  cutlass::arch::NamedBarrier tmem_allocation_result_barrier(
      32 + 128, cutlass::arch::ReservedNamedBarriers::TmemAllocBarrier);

  // handle the case where K is not divisible by CTA_K
  int k_tile_count = cutlass::ceil_div(size<1>(mA), CTA_K);

  /*if (thread0() && (blockIdx.x == 0) && (blockIdx.y == 0) && (blockIdx.z == 0)) {
      // Represent the SMEM buffers for A
      Tensor tCsA = shared_storage.tensor_sA();         // ((Mma_M, Mma_K), NumMma_M, NumMma_K,
  DMA_Stage) Tensor tCsB = shared_storage.tensor_sB();         // ((Mma_N, Mma_K), NumMma_N,
  NumMma_K, DMA_Stage)

      printf("tCsA:\t"); print(tCsA); print("\n");
      printf("tCsB:\t"); print(tCsB); print("\n");
      printf("k_tile_count:\t%d\n", k_tile_count);
  }*/

  // barrier initialization needs to be visible to all warps
  // defer it as late as possible to allow some thread divergence in prolog
  cutlass::arch::fence_barrier_init();
  __syncthreads();

  if (warp_idx == 0) {
    DMA_A_warp<SharedStorage, ATensor, TmaAtomA, TiledMMA, CTA_M, CTA_K, DMA_Stage>(
        shared_storage, work_tile_info, mA, &tma_atom_A, tiled_mma, pdl_count);
  } else if (warp_idx == 1) {
    DMA_B_warp<SharedStorage, BTensor, TmaAtomB, TiledMMA, CTA_N, CTA_K, DMA_Stage>(
        shared_storage, work_tile_info, mB, &tma_atom_B, tiled_mma);
  } else if (warp_idx == 2) {
    MMA_warp<SharedStorage, CTensor, TiledMMA, CTA_M, CTA_N, DMA_Stage>(
        shared_storage, work_tile_info, mC, tiled_mma, k_tile_count,
        tmem_allocation_result_barrier);
  } else if (warp_idx >= 4) {
    EPILOG_warp<SharedStorage, CTensor, BiasTensor, TiledMMA, CTA_M, CTA_N, NO_BIAS>(
        shared_storage, work_tile_info, mC, mBias, tiled_mma, tmem_allocation_result_barrier);
  }

  __syncthreads();

  // deallocate TMEM
  if (warp_idx == 0) {
    using TmemAllocator = cute::TMEM::Allocator1Sm;
    TmemAllocator tmem_allocator{};
    tmem_allocator.free(shared_storage.tmem_base_ptr, TmemAllocator::Sm100TmemCapacityColumns / 2);
  }
}

// L is the batch dimension, this kernel can handle both gemm and bmm
//
// here we use UMMA::Major to specify the gmem layout of A and B. It is defined in
// cutlass/include/cute/arch/mma_sm100_desc.hpp MajorMN means M/N is the contiguous dimension, and
// MajorK means K is the non-contiguous dimension for example, the most common version for bs1
// inference is TNN in BLAS term, T stands for transposed (row major), N stands for non-transposed
// (column major) A shape (M, K, L), where K is contiguous, other layout is free to choose ((M, K)
// row major in BLAS term) B shape (N, K, L), where K is contiguous, other layout is free to choose
// ((K, N) column major in BLAS term) C shape (M, N, L), where M is contiguous, other layout is free
// to choose ((M, N) column major in BLAS term) Bias shape (M), where M is contiguous
//
// but for MLA bmm2, we need to support NNN such that A is M major
//
// bias is optional, if nullptr, it means no bias
template <class TypeA, class TypeB, class TypeC, class AccType, class TypeBias, int CTA_M,
          int CTA_N, int CTA_K, int DMA_Stage, cute::UMMA::Major UmmaMajorA,
          cute::UMMA::Major UmmaMajorB>
void tgv_gemm_host(
    TypeA* device_ptr_A, TypeB* device_ptr_B, TypeC* device_ptr_C, TypeBias* device_ptr_Bias,
    int Gemm_M, int Gemm_N, int Gemm_K, int Gemm_L, int stride_A_M, int stride_A_K, int stride_A_L,
    int stride_B_N, int stride_B_K, int stride_B_L, int stride_C_M, int stride_C_N, int stride_C_L,
    bool pdl,
    int pdl_count = -1,  // which kblock count to do griddepcontrol.launch_dependents, default is
                         // griddepcontrol.launch_dependents at the end of DMA_A warp
    cudaStream_t stream = 0) {
  Layout layout_A = make_layout_A<CTA_M, CTA_K, UmmaMajorA>(
      Gemm_M, Gemm_K, Gemm_L, stride_A_M, stride_A_K,
      stride_A_L);  // (M, K, L), where K or M is contiguous, other layout is free to choose
  Layout layout_B = make_layout_B<CTA_N, CTA_K, UmmaMajorB>(
      Gemm_N, Gemm_K, Gemm_L, stride_B_N, stride_B_K,
      stride_B_L);  // (N, K, L), where K or N is contiguous, other layout is free to choose
  Layout layout_C = make_layout_C<CTA_M, CTA_N>(
      Gemm_M, Gemm_N, Gemm_L, stride_C_M, stride_C_N,
      stride_C_L);  // (M, N, L), where M is contiguous, other layout is free to choose
  Layout layout_Bias = make_layout_Bias<CTA_M, CTA_N>(
      Gemm_M, Gemm_N, Gemm_L, stride_C_M, stride_C_N,
      stride_C_L);  // (M, N, L), where M is contiguous, other stride is 0

  // how we handle oob:
  //   oob for A and B are handled by TMA
  //   oob for C is explicitly handled by predicate in the epilog since it uses simple st.global
  //   epilog oob for bias is explicitly handled by predicate in the bias load it uses simple
  //   ld.global epilog
  // assert(Gemm_M % CTA_M == 0);
  // assert(Gemm_N % CTA_N == 0);
  // assert(Gemm_K % CTA_K == 0);

  /*std::cout << "Running for problem shape (MxNxKxL): " << Gemm_M << "x" << Gemm_N << "x" << Gemm_K
  << "x" << Gemm_L << std::endl; std::cout << "with tile size CTA_M: " << CTA_M << ", CTA_N: " <<
  CTA_N << ", CTA_K: " << CTA_K << std::endl; std::cout << "layout_A: "; print(layout_A); std::cout
  << std::endl << "layout_B: "; print(layout_B); std::cout << std::endl << "layout_C: ";
  print(layout_C);
  std::cout << std::endl;*/

  // create the cute tensor
  Tensor mA = make_tensor(make_gmem_ptr(device_ptr_A), layout_A);  // (Gemm_M, Gemm_K, Gemm_L)
  Tensor mB = make_tensor(make_gmem_ptr(device_ptr_B), layout_B);  // (Gemm_N, Gemm_K, Gemm_L)
  Tensor mC = make_tensor(make_gmem_ptr(device_ptr_C), layout_C);  // (Gemm_M, Gemm_N, Gemm_L)
  Tensor mBias =
      make_tensor(make_gmem_ptr(device_ptr_Bias), layout_Bias);  // (Gemm_M, Gemm_N, Gemm_L)

  // print("mA:\t"); print(mA); print("\n");
  // print("mB:\t"); print(mB); print("\n");
  // print("mC:\t"); print(mC); print("\n");
  // print("mBias:\t"); print(mBias); print("\n");

  // tiled mma just contain a single mma atom/instruction, not stacking anything
  // we need to pass 3 additional argument to specify tcgen05.mma instruction
  // 1. data type
  // 2. mma_m and mma_n, mma_k is always 32B, i.e. mma_k=32 for fp8, mma_k=16 for bf16
  // 3. layout of A and B in smem, cute::UMMA::Major::MN means M/N is contiguous,
  // cute::UMMA::Major::K means K is contiguous
  //    tcgen05.mma supports arbitrary combination of A/B smem layout, it will issue different
  //    instructions for different layout however, the layout here has to match the gmem layout, as
  //    in gmem->smem there is no transpose, K major smem layout means gmem layout is also K major
  // sm100_make_1sm_trivial_tiled_mma is defined in
  // cutlass/gemm/collective/builders/sm100_common.inl
  //
  // here we use the cutlass utility to choose mma atom for us based on the data type and tile size
  // this is equivalent to the following code for bf16:
  // bf16 * bf16 = fp32, SS means A and B are both sourced from smem
  // TiledMMA tiled_mma = make_tiled_mma(SM100_MMA_F16BF16_SS<TypeA, TypeB, AccType, // Mma's A, B,
  // and Accumulator types
  //                                                          CTA_M, CTA_N, // Mma M and N
  //                                                          dimensions UMMA::Major::K,
  //                                                          UMMA::Major::K>{});  // A and B
  //                                                          layouts are both K major
  // tcgen05.mma only support CTA_M=64 or 128 for 1 cta mode
  static_assert((CTA_M == 64) || (CTA_M == 128), "CTA_M must be 64 or 128");
  TiledMMA tiled_mma = cutlass::gemm::collective::detail::sm100_make_1sm_trivial_tiled_mma<
      TypeA, TypeB, AccType,                      // Mma's A, B, and Accumulator types
      Shape<Int<CTA_M>, Int<CTA_N>, Int<CTA_K>>,  // TileShape_MNK
      Shape<_1, _1, _1>,                          // ClusterShape_MNK
      UmmaMajorA, UmmaMajorB>();                  // A and B layouts

  // We can also print and inspect the tiled_mma
  // print(tiled_mma);

  // Pre-partitioned smem Tile Shape (CTA_M, CTA_K, DMA_Stage) to post-partitioned smem tile shape
  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) (Mma_M, Mma_K) is the shape of A tile in 1 mma
  // instruction NumMma_M is the number of iterations in M mode, NumMma_K is the number of
  // iterations in K mode NumMma_M * NumMma_K is the total number of mma instructions for this tile
  // CTA_M = Mma_M * NumMma_M, CTA_N = Mma_N * NumMma_N, CTA_K = Mma_K * NumMma_K
  // function is defined in cutlass/include/cute/atom/mma_atom.hpp
  auto mma_shape_A =
      partition_shape_A(tiled_mma, make_shape(Int<CTA_M>{}, Int<CTA_K>{}, Int<DMA_Stage>{}));
  // Pre-partitioned smem Tile Shape (CTA_N, CTA_K, DMA_Stage) to post-partitioned smem tile shape
  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) (Mma_N, Mma_K) is the shape of B tile in 1 mma
  // instruction
  auto mma_shape_B =
      partition_shape_B(tiled_mma, make_shape(Int<CTA_N>{}, Int<CTA_K>{}, Int<DMA_Stage>{}));

  // Print and inspect mma_shape_A, and mma_shape_B for this example. note that this is only the
  // shape of the smem tile, not the actual smem layout
  // print("mma_shape_A:\t"); print(mma_shape_A); print("\n");  // mma_shape_A:  ((Mma_M, Mma_K),
  // NumMma_M, NumMma_K, DMA_Stage) print("mma_shape_B:\t"); print(mma_shape_B); print("\n");  //
  // mma_shape_B:  ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)

  // for bf16 the smem swizzle atom for K major is Sw<3,4,3> o smem_ptr[16b](unset) o
  // (_8,_64):(_64,_1) i.e. it's 8x128B (Shape(M8, K64)) contiguous chunk, and we stack multiple of
  // this swizzle atom to get the final smem layout for a tile of A and B
  // print("swizzle atom:\t"); print(UMMA::Layout_K_SW128_Atom<TypeA>{}); print("\n");
  // for bf16 the smem swizzle atom for M/N major is Sw<3,4,3> o smem_ptr[16b](unset) o
  // (_64,_8):(_1,_64) i.e. it's 128Bx8 (Shape(M64, K8)) contiguous chunk, and we stack multiple of
  // this swizzle atom to get the final smem layout for a tile of A and B
  // print("swizzle atom:\t"); print(UMMA::Layout_MN_SW128_Atom<TypeA>{}); print("\n");

  // we finally create the smem layout for tile A and B here
  // tile_to_mma_shape takes two arguments:
  // 1. the swizzle atom
  // 2. the expected post-mma-partitioned smem tile shape, e.g. ((Mma_M, Mma_K), NumMma_M, NumMma_K,
  // DMA_Stage) for A it stacks/replicates the swizzle atom such that the final shape of the stacked
  // tensor is the expected post-mma-partitioned smem tile shape and the layout of that tensor is
  // the smem layout for tile A and B it is defined in
  // cutlass/include/cute/atom/mma_traits_sm100.hpp we can also use the old tile_to_shape api to get
  // the same functionality
  //
  // for bf16 K major A tile, conceptually the function does three steps:
  // 1. it reconstruct the pre-partitioned smem tile shape (CTA_M, CTA_K, DMA_Stage) from the
  // post-mma-partitioned smem tile shape ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
  // 2. it stacks/replicates the (M8, K64) swizzle atom along the M and K mode, such that the final
  // shape of the stacked tensor is (CTA_M, CTA_K),
  //    similar to a blocked product, the replication happens along both the M and K mode, the
  //    optional "order" argument of the function specifies which mode got replicated first, Step<1,
  //    2> specifies that the M mode got replicated first, then the K mode, Step<2, 1> specifies
  //    that the K mode got replicated first, then the M mode. Both are legal replication orders.
  //    This is essentially the functionality of tile_to_shape function defined in
  //    cutlass/include/cute/layout.hpp
  // 3. after we get the stacked layout with shape (CTA_M, CTA_K, DMA_Stage), we create a new view
  // (layout) of the tensor with
  //    post-mma-partitioned smem tile shape ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage), this
  //    is achieved by a tiled_divide of the layout from step 2. Now we can slice into ((Mma_M,
  //    Mma_K), NumMma_M, NumMma_K, DMA_Stage) and will get the smem tile for each mma instruction
  //    with the correct layout
  // The result of this is the complete smem layout for tile A with shape ((Mma_M, Mma_K), NumMma_M,
  // NumMma_K, DMA_Stage)
  //
  // More note about how the stacked layout work in practice, imagine our smem K major tile A is
  // (M64, K128) and out swizzle atom is (M8, K64) for bf16, mma instruction is (M64, N32, K16) as
  // we said, there are two ways to stack the swizzle atom, Step<1, 2> (M first 64/8=8 times, then K
  // for 128/64=2 times) and Step<2, 1> (K first 2 times, then M for 8 times) so we have 8*2=16
  // swizzle atoms in total, and we will show below both stacking order would work correctly
  // 1. Step<1, 2>, after the stacking, the first mma instruction requires a tile A of (M64, K16),
  // so the first 8 (atom 0-7) (M8, K64) swizzle atoms participate in this mma instruction
  //    we set stride dimension byte offset of the A smem descriptor to be 8x128B (the (M8, K64)
  //    swizzle atom size), meaning that it will load swizzle atoms from smem at 8x128B stride and
  //    the first 8 (atom 0-7) (M8, K64) swizzle atoms are loaded to participate in the first mma
  //    instruction
  // The atom layout for the (M64, K128) tile is :
  // atom 0,  atom 8
  // atom 1,  atom 9
  // atom 2,  atom 10
  // atom 3,  atom 11
  // atom 4,  atom 12
  // atom 5,  atom 13
  // atom 6,  atom 14
  // atom 7,  atom 15
  // 2. Step<2, 1>, after the stacking, the first mma instruction requires a tile A of (M64, K16),
  // so 8 (M8, K64) swizzle atoms (atom 0, 2, 4, ..., 14) participate in this mma instruction
  //    instead of setting stride dimension byte offset of the A smem descriptor to be 8x128B, we
  //    set it to be 8x128Bx2, meaning that it will load swizzle atoms from smem at 8x128Bx2 stride
  //    and every other swizzle atom (atom 0, 2, 4, ..., 14) out of the total 16 swizzle atoms
  //    participate in the first mma instruction
  // The atom layout for the (M64, K128) tile is :
  // atom 0,  atom 1
  // atom 2,  atom 3
  // atom 4,  atom 5
  // atom 6,  atom 7
  // atom 8,  atom 9
  // atom 10, atom 11
  // atom 12, atom 13
  // atom 14, atom 15
  //
  // there is one final piece, atom 0 has K128B==K64, and mma_K is 16, so atom 0 participates in 4
  // mma instructions, and each instruction reads 1/4 of atom 0. this is achieved by setting the
  // base addr of the A tile in smem descriptor to different value such that the tensor core can
  // index into the correct part of atom 0. Without getting into too much details, you increment the
  // base addr by 32B for each mma instruction to read the corresponding part of atom 0, and 4
  // instructions in total covers the 128B K dimension.
  //
  // below is the extreme detail which you really don't need to know, but just for your reference,
  // we show the layout of atom 0, each quark is 16B, i.e. (M1, K8) in bf16 since each atom is (M8,
  // K64), atom 0 contains 8x8=64 (M1, K8) quarks below is the layout of atom 0, each entry is a
  // quark, and we denote which mma instruction it belongs to, since an atom participates in 4 mma
  // instructions mma0 mma0 mma1 mma1 mma2 mma2 mma3 mma3 mma0 mma0 mma1 mma1 mma2 mma2 mma3 mma3
  // mma1 mma1 mma0 mma0 mma3 mma3 mma2 mma2
  // mma1 mma1 mma0 mma0 mma3 mma3 mma2 mma2
  // mma2 mma2 mma3 mma3 mma0 mma0 mma1 mma1
  // mma2 mma2 mma3 mma3 mma0 mma0 mma1 mma1
  // mma3 mma3 mma2 mma2 mma1 mma1 mma0 mma0
  // mma3 mma3 mma2 mma2 mma1 mma1 mma0 mma0
  // so every mma instruction increments its smem descriptor base addr by 32B (2 quarks) to read the
  // next 1/4 of atom 0 this is the case for row 0/1, as in increment base addr by 32B, we go from
  // mma0's quark 0 to mma1's quark 0, and so on for row 2/3, swizzle handles the addr change for
  // us, a 32B increment in base addr will go from mma0's quark 0 to mma0's quark 1 (despite the
  // actual addr is a decrement of 32B, i.e. an increment of base addr by 32B results in a decrement
  // of addr by 32B in row 2/3 after swizzle) in any case, our smem descriptor base addr update
  // follows row 0/1, increment by 32B you get the descriptor for the next mma instruction
  //
  // here we use the cutlass utility to choose smem swizzle atom for us based on the data type and
  // tile size, it tries to find the biggest swizzle atom for the given dtype, tile size and gmem
  // layout this is equivalent to the following code for bf16 K major A/B tile: SW128 means swizzle
  // 128B, and K dimension has contiguous 128B auto SmemLayoutAtomA =
  // UMMA::Layout_K_SW128_Atom<TypeA>{}; auto SmemLayoutAtomB = UMMA::Layout_K_SW128_Atom<TypeB>{};
  auto SmemLayoutAtomA =
      cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorA,  // gmem layout of A
                                                             TypeA,       // data type of A
                                                             Int<CTA_M>,
                                                             Int<CTA_K>>();  // tile size of A
  auto SmemLayoutAtomB =
      cutlass::gemm::collective::detail::sm100_smem_selector<UmmaMajorB,  // gmem layout of B
                                                             TypeB,       // data type of B
                                                             Int<CTA_N>,
                                                             Int<CTA_K>>();  // tile size of B
  // finally construct the smem layout for tile A and B based on the swizzle atom and mma shape
  auto sA_layout = UMMA::tile_to_mma_shape(
      SmemLayoutAtomA, mma_shape_A);  // ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
  auto sB_layout = UMMA::tile_to_mma_shape(
      SmemLayoutAtomB, mma_shape_B);  // ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)

  // Print and inspect SmemLayoutAtomA, SmemLayoutAtomB, sA_layout and sB_layout for this example.
  // print("SmemLayoutAtomA:\t"); print(SmemLayoutAtomA); print("\n");
  // print("SmemLayoutAtomB:\t"); print(SmemLayoutAtomB); print("\n");
  // print("sA_layout:\t"); print(sA_layout); print("\n");
  // print("sB_layout:\t"); print(sB_layout); print("\n");

  // Now we can find the SMEM allocation size
  using SMEMStorage =
      SharedStorage<TypeA, TypeB, decltype(sA_layout), decltype(sB_layout), DMA_Stage>;

  // create TMA descriptors for A and B matrices
  Copy_Atom tma_atom_A = make_tma_atom(
      SM90_TMA_LOAD{},               // TMA Load Op, sm100 reuses sm90 tma atom
      mA,                            // Source GMEM tensor
      sA_layout(_, _, _, Int<0>{}),  // Destination SMEM layout for 1 DMA_Stage, ((Mma_M, Mma_K),
                                     // NumMma_M, NumMma_K)
      make_shape(Int<CTA_M>{}, Int<CTA_K>{})  // TMA box shape, it's cosize must match the cosize of
                                              // the destination smem layout
  );
  // this is an arithmetic tuple, denoting the coordinate of the top-left corner of the TMA box
  Tensor mA_tma = tma_atom_A.get_tma_tensor(shape(mA));  // (Gemm_M, Gemm_K, Gemm_L)
  // print("tma_atom_A:\t"); print(tma_atom_A); print("\n");
  // print("mA_tma:\t"); print(mA_tma); print("\n");

  Copy_Atom tma_atom_B = make_tma_atom(
      SM90_TMA_LOAD{},               // TMA Load Op, sm100 reuses sm90 tma atom
      mB,                            // Source GMEM tensor
      sB_layout(_, _, _, Int<0>{}),  // Destination SMEM layout for 1 DMA_Stage, ((Mma_N, Mma_K),
                                     // NumMma_N, NumMma_K)
      make_shape(Int<CTA_N>{}, Int<CTA_K>{})  // TMA box shape, it's cosize must match the cosize of
                                              // the destination smem layout
  );
  // this is an arithmetic tuple, denoting the coordinate of the top-left corner of the TMA box
  Tensor mB_tma = tma_atom_B.get_tma_tensor(shape(mB));  // (Gemm_N, Gemm_K, Gemm_L)
  // print("tma_atom_B:\t"); print(tma_atom_B); print("\n");
  // print("mB_tma:\t"); print(mB_tma); print("\n");

  int smemBytes = sizeof(SMEMStorage);

  // invoke the kernel
  cudaLaunchConfig_t config;
  cudaLaunchAttribute attrs[1];
  // each batch is a set of separate CTA
  config.gridDim = dim3{(uint32_t)cutlass::ceil_div(Gemm_M, CTA_M),
                        (uint32_t)cutlass::ceil_div(Gemm_N, CTA_N), (uint32_t)Gemm_L};
  config.blockDim = 256;  // 8 warps
  config.dynamicSmemBytes = smemBytes;
  config.stream = stream;
  attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attrs[0].val.programmaticStreamSerializationAllowed = 1;
  config.attrs = attrs;
  config.numAttrs = pdl ? 1 : 0;

  // Launch kernel based on bias availability
  // We have to duplicate the code since two kernel instances are of different types
  if (device_ptr_Bias != nullptr) {
    auto* kernel_instance =
        &tgv_gemm_device<SMEMStorage, decltype(mA_tma), decltype(mB_tma), decltype(mC),
                         decltype(mBias), decltype(tma_atom_A), decltype(tma_atom_B),
                         decltype(tiled_mma), CTA_M, CTA_N, CTA_K, DMA_Stage, false>;
    gpuErrChk(cudaFuncSetAttribute(*kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   smemBytes));
    gpuErrChk(cudaLaunchKernelEx(&config, kernel_instance, mA_tma, mB_tma, mC, mBias, tma_atom_A,
                                 tma_atom_B, tiled_mma, pdl_count));
  } else {
    auto* kernel_instance =
        &tgv_gemm_device<SMEMStorage, decltype(mA_tma), decltype(mB_tma), decltype(mC),
                         decltype(mBias), decltype(tma_atom_A), decltype(tma_atom_B),
                         decltype(tiled_mma), CTA_M, CTA_N, CTA_K, DMA_Stage, true>;
    gpuErrChk(cudaFuncSetAttribute(*kernel_instance, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   smemBytes));
    gpuErrChk(cudaLaunchKernelEx(&config, kernel_instance, mA_tma, mB_tma, mC, mBias, tma_atom_A,
                                 tma_atom_B, tiled_mma, pdl_count));
  }
}

}  // namespace gemm
}  // namespace flashinfer
