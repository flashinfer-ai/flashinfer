/*
 * Copyright (c) 2026 by the PatchShift Conv3d contributors.
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

#include <flashinfer/conv3d/patchshift/kernels.cuh>

#include "launcher.cuh"

namespace flashinfer::conv3d::patchshift::host {
namespace {

namespace kernel = ::flashinfer::conv3d::patchshift::detail;

enum InputMapMask : uint32_t {
  kInputM128 = 1u << 0,
  kInputHybridC32 = 1u << 1,
  kInputCompactP32 = 1u << 2,
  kInputCompactQ8 = 1u << 3,
  kInputCompactQ4 = 1u << 4,
  kInputCompactP1C64 = 1u << 5,
  kInputId40PTailC64 = 1u << 6,
  kInputId40QTailC64 = 1u << 7,
  kInputM64 = 1u << 8,
  kInputM64CompactQ4 = 1u << 9,
};

__device__ __forceinline__ void ReplaceTensorMapAddress(TensorMap* map, Element* input) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  uint64_t map_address = reinterpret_cast<uint64_t>(map);
  uint64_t input_address = reinterpret_cast<uint64_t>(input);
  asm volatile("tensormap.replace.tile.global_address.global.b1024.b64 [%0], %1;\n"
               :
               : "l"(map_address), "l"(input_address)
               : "memory");
#else
  (void)map;
  (void)input;
#endif
}

__global__ void UpdateInputMapAddresses(DescriptorWorkspace* workspace, Element* input,
                                        uint32_t mask) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  if (mask & kInputM128) ReplaceTensorMapAddress(&workspace->input_m128, input);
  if (mask & kInputHybridC32) ReplaceTensorMapAddress(&workspace->input_hybrid_c32, input);
  if (mask & kInputCompactP32) ReplaceTensorMapAddress(&workspace->input_compact_p32, input);
  if (mask & kInputCompactQ8) ReplaceTensorMapAddress(&workspace->input_compact_q8, input);
  if (mask & kInputCompactQ4) ReplaceTensorMapAddress(&workspace->input_compact_q4, input);
  if (mask & kInputCompactP1C64) ReplaceTensorMapAddress(&workspace->input_compact_p1_c64, input);
  if (mask & kInputId40PTailC64) ReplaceTensorMapAddress(&workspace->input_id40_ptail_c64, input);
  if (mask & kInputId40QTailC64) ReplaceTensorMapAddress(&workspace->input_id40_qtail_c64, input);
  if (mask & kInputM64) ReplaceTensorMapAddress(&workspace->input_m64, input);
  if (mask & kInputM64CompactQ4) ReplaceTensorMapAddress(&workspace->input_m64_compact_q4, input);
  asm volatile("fence.proxy.tensormap::generic.release.gpu;\n" ::: "memory");
#else
  (void)workspace;
  (void)input;
  (void)mask;
#endif
}

}  // namespace

ConcurrencyMode GetConcurrencyMode(const Conv3dProblem& problem, int multi_processor_count) {
  auto const& opts = problem;
  struct {
    int multiProcessorCount;
  } device_prop{multi_processor_count};

  using namespace kernel;
#include "select_policy.inl"

  // These launches write disjoint output-channel or spatial intervals.  ID18
  // is deliberately excluded: its cluster-A main grid needs a graph
  // launch-completion dependency before its edge CTAs can safely overlap.
  bool use_disjoint_main_auxiliary =
      (((use_exact_m32_tail || use_exact_m64_tail) && m128_tiles > 0 && m64_tiles > 0) ||
       use_hybrid_compact_p1_c96) &&
      !use_split_cluster_a_compact_edges;
  return use_disjoint_main_auxiliary ? ConcurrencyMode::kDisjointMainAuxiliary
                                     : ConcurrencyMode::kSequential;
}

Status UpdateInputMaps(DescriptorWorkspace* workspace, Element* input, const Conv3dProblem& problem,
                       int multi_processor_count, cudaStream_t stream) {
  auto const& opts = problem;
  struct {
    int multiProcessorCount;
  } device_prop{multi_processor_count};

  using namespace kernel;
#include "select_policy.inl"

  uint32_t input_map_mask = 0;
  if (m128_tiles > 0) input_map_mask |= kInputM128;
  if (use_hybrid_c64_c32) input_map_mask |= kInputHybridC32;
  if (use_compact_spatial) input_map_mask |= kInputCompactP32 | kInputCompactQ8;
  if (use_compact_qtail_q2_single_launch) input_map_mask |= kInputCompactQ4;
  if (use_hybrid_compact_p1_c96) input_map_mask |= kInputCompactP1C64;
  if (use_m256_cluster_b_c64_exact_id40) {
    input_map_mask |= kInputId40PTailC64 | kInputId40QTailC64;
  }
  if (m64_tiles > 0) input_map_mask |= kInputM64;
  if (use_m64n128_d1_c32_micro) input_map_mask |= kInputM64CompactQ4;
  UpdateInputMapAddresses<<<1, 1, 0, stream>>>(workspace, input, input_map_mask);
  return Status::Cuda(cudaGetLastError());
}

Status Launch(DescriptorWorkspace* workspace, Element* input, Element* output,
              const Conv3dProblem& problem, int multi_processor_count, cudaStream_t stream,
              LaunchPart part) {
  auto const& opts = problem;
  struct {
    int multiProcessorCount;
  } device_prop{multi_processor_count};

  using namespace kernel;
#include "select_policy.inl"

#define RETURN_IF_CUDA_ERROR(call) \
  do {                             \
    cudaError_t error_ = (call);   \
    if (error_ != cudaSuccess) {   \
      return Status::Cuda(error_); \
    }                              \
  } while (0)

  TensorMap* d_input_map_m128 = &workspace->input_m128;
  TensorMap* d_input_map_hybrid_c32 = &workspace->input_hybrid_c32;
  TensorMap* d_input_map_compact_p32 = &workspace->input_compact_p32;
  TensorMap* d_input_map_compact_q8 = &workspace->input_compact_q8;
  TensorMap* d_input_map_compact_q4 = &workspace->input_compact_q4;
  TensorMap* d_input_map_compact_p1_c64 = &workspace->input_compact_p1_c64;
  TensorMap* d_input_map_id40_ptail_c64 = &workspace->input_id40_ptail_c64;
  TensorMap* d_input_map_id40_qtail_c64 = &workspace->input_id40_qtail_c64;
  TensorMap* d_weight_map_m128 = &workspace->weight_m128;
  TensorMap* d_input_map_m64 = &workspace->input_m64;
  TensorMap* d_input_map_m64_compact_q4 = &workspace->input_m64_compact_q4;
  TensorMap* d_weight_map_m64 = &workspace->weight_m64;
  Element* d_output = output;
  (void)input;

  int q_tiles = use_hybrid_exact_w31 ? 1 : (opts.w + kOutQ - 1) / kOutQ;
  int p_tiles_m128 = (opts.h + kMainOutP - 1) / kMainOutP;
  int p_tiles_m64 = use_m64n128_d1_c32_micro ? (opts.h + kM64N128MicroOutP - 1) / kM64N128MicroOutP
                    : (use_m32_path || use_m64_p16) ? (opts.h + kM64P16OutP - 1) / kM64P16OutP
                                                    : (opts.h + kTailOutP - 1) / kTailOutP;
  int compact_spatial_tasks =
      use_compact_qtail_q2_single_launch
          ? compact_q2_spatial_tasks
          : (use_compact_ptail1_single_launch
                 ? p1_compact_spatial_tasks
                 : compact_full_q_tiles * compact_full_p_tiles +
                       (compact_p_tail > 0 ? compact_full_q_tiles : 0) +
                       (compact_q_tail > 0 ? (opts.h + kCompactQOutP - 1) / kCompactQOutP : 0));
  int m128_spatial_tasks = use_hybrid_compact_p1_c96 ? compact_full_q_tiles * compact_full_p_tiles
                                                     : compact_spatial_tasks;
  dim3 grid_m128 = use_compact_spatial ? dim3(m128_spatial_tasks, 1, opts.n * opts.d * m128_tiles)
                                       : dim3(q_tiles, p_tiles_m128, opts.n * opts.d * m128_tiles);
  // The x dimension is expressed in physical CTAs.  clusterDim.x=2 binds
  // each adjacent pair to one logical spatial tile, while rank 0/1 select
  // adjacent M128 output-channel tiles (a logical M256 macro tile).
  dim3 grid_m256_cluster_b(2 * q_tiles, p_tiles_m128, opts.n * opts.d * (m128_tiles / 2));
  int spatial_tiles_m128 = use_split_cluster_a_compact_edges
                               ? compact_full_p_tiles * compact_full_q_tiles
                               : p_tiles_m128 * q_tiles;
  int cluster_a_group_size = use_cluster_a_group4 ? 4 : 2;
  dim3 grid_cluster_a_spatial(
      cluster_a_group_size *
          ((spatial_tiles_m128 + cluster_a_group_size - 1) / cluster_a_group_size),
      1, opts.n * opts.d * m128_tiles);
  dim3 grid_hybrid_cluster_a4(4 * ((q_tiles + 3) / 4), 1, 4);
  dim3 grid_m64(q_tiles, p_tiles_m64, opts.n * opts.d * m64_tiles);
  dim3 grid_m32_shallow_cluster4(4 * q_tiles, p_tiles_m64, 1);
  dim3 grid_m64_cluster_b(2 * q_tiles, p_tiles_m64, opts.n * opts.d * (m64_tiles / 2));
  if (m128_tiles > 0 && part != LaunchPart::kAuxiliary) {
    if (use_hybrid_cluster_a4_exact_p15) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_hybrid_c96_exact_p15_cluster_a4_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else if (use_cluster_a_spatial_c64_k64) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
          use_cluster_a_exact_n2d2   ? general_m128_cluster_a_spatial_c64_k64_kernel<4, true>
          : use_cluster_a_exact_n1d8 ? general_m128_cluster_a_spatial_c64_k64_kernel<4, false, true>
          : use_cluster_a_exact_n1d4
              ? general_m128_cluster_a_spatial_c64_k64_kernel<4, false, false, true>
          : use_cluster_a_exact_n2d4
              ? general_m128_cluster_a_spatial_c64_k64_kernel<2, false, false, false, true>
          : use_cluster_a_exact_id18
              ? general_m128_cluster_a_spatial_c64_k64_kernel<4, false, false, false, false, true>
          : use_cluster_a_group4 ? general_m128_cluster_a_spatial_c64_k64_kernel<4, false>
                                 : general_m128_cluster_a_spatial_c64_k64_kernel<2, false>,
          cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    } else if (use_hybrid_c64_c32) {
      if (use_hybrid_compact_p1_c96) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_hybrid_main_exact_h17_w840_kernel,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      } else if (use_hybrid_exact_p15) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_hybrid_c64_c32_b2a3_kernel<false, false, false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_hybrid_exact_w31) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true, false, false, false, false,
                                                        false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_hybrid_exact_spatial) {
        if (use_hybrid_exact_h16_w840) {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true, false, false, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        } else {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        }
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_hybrid_c64_c32_b2a3_kernel<false, false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      }
    } else if (use_m256_cluster_b_c64_k64) {
      if (use_padded_m256_k160) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m256_cluster_b_c64_k64_kernel<false, 160, false, true, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_padded_m256_k192) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m256_cluster_b_c64_k64_kernel<false, 192, false, true, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_m256_cluster_b_c64_exact_id40) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m256_cluster_b_c64_k64_kernel<false, 0, true, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_m256_cluster_b_c64_exact_d4_c128) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m256_cluster_b_c64_k64_kernel<false, 0, false, true, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_m256_cluster_b_c64_eight_warp_store) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m256_cluster_b_c64_k64_kernel<false, 0, false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_m256_cluster_b_c64_optimized_partial) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m256_cluster_b_c64_k64_kernel<true>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m256_cluster_b_c64_k64_kernel<false>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      }
    } else if (use_m256_cluster_b_c32) {
      if (use_partial_m128_epilogue) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m256_cluster_b_c32_kernel<true>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m256_cluster_b_c32_kernel<false>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      }
    } else if (use_k64_c64_b2a3_k32a) {
      if (k64_c64_exact_kout == 96) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k64_c64_b2a3_k32a_kernel<false, false, 96>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (k64_c64_exact_kout == 120) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k64_c64_b2a3_k32a_kernel<false, false, 120>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_k64_c64_exact_k128) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k64_c64_b2a3_k32a_kernel<false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_partial_m128_epilogue) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m128n256_k64_c64_b2a3_k32a_kernel<true>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m128n256_k64_c64_b2a3_k32a_kernel<false>,
                                                  cudaFuncAttributePreferredSharedMemoryCarveout,
                                                  cudaSharedmemCarveoutMaxShared));
      }
    } else if (use_c16_path) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m128n256_k32_deep_ilp_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else {
      if (use_compact_spatial) {
        if (use_compact_qtail_q2_single_launch && use_compact_ptail1_single_launch) {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_k32_deep_b_c32_kernel<false, true, true, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        } else if (use_compact_qtail_q2_single_launch) {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_k32_deep_b_c32_kernel<false, true, false, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        } else if (use_compact_ptail1_single_launch) {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_k32_deep_b_c32_kernel<false, true, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        } else {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128n256_k32_deep_b_c32_kernel<false, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        }
      } else if (exact_aligned_kout == 96) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 96>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (exact_aligned_kout == 120) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 120>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (exact_aligned_kout == 160) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 160>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_exact_p15_full_q_m128) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else if (use_partial_m128_epilogue) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<true, false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m128n256_k32_deep_b_c32_kernel<false, false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      }
    }
  }
  if (part != LaunchPart::kMain) {
    if (use_split_cluster_a_compact_edges) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_id18_p1_q1_compact_edge_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else if (use_hybrid_compact_p1_c96) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_hybrid_ptail1_exact_h17_w840_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    }
  }
  if (m64_tiles > 0 && !use_c16_path && part != LaunchPart::kMain) {
    if (use_m64n128_d1_c32_micro) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m64n128_d1_c32_micro_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else if (use_m32_d1_c128_shallow) {
      if (use_m32_d1_c128_shallow_cluster4) {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            general_m32n256_d1_c128_shallow_c64_kernel<true, true>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            use_m32_d1_c128_shallow_exact ? general_m32n256_d1_c128_shallow_c64_kernel<true>
                                          : general_m32n256_d1_c128_shallow_c64_kernel<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      }
    } else if (use_m32_p16_c64) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m32n256_k64_p16_b2a3_c64_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else if (use_m32_path) {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(general_m32n256_k32_p16_b2a3_c32_kernel,
                                                cudaFuncAttributePreferredSharedMemoryCarveout,
                                                cudaSharedmemCarveoutMaxShared));
    } else if (use_m64_p16_c64) {
      if (use_m64_cluster_b_c64) {
        if (opts.d == 2) {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128_cluster_b_m64_p16_c64_kernel<2, true, false>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        } else {
          RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
              general_m128_cluster_b_m64_p16_c64_kernel<3, true>,
              cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
        }
      } else {
        RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
            use_m64_p16_c64_exact ? general_m64n256_k64_p16_b2a6_c64_kernel<true>
                                  : general_m64n256_k64_p16_b2a6_c64_kernel<false>,
            cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
      }
    } else {
      RETURN_IF_CUDA_ERROR(cudaFuncSetAttribute(
          use_m64_p16 ? general_m64n256_k32_p16_b2a3_c32_kernel
                      : general_m64n256_k32_deep_b_c32_multi_issuer_tail_kernel,
          cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared));
    }
  }
  auto launch_main = [&](cudaStream_t stream) -> Status {
    if (m128_tiles > 0) {
      if (use_cluster_a_spatial_c64_k64) {
        cudaLaunchConfig_t config{};
        config.gridDim = grid_cluster_a_spatial;
        config.blockDim = dim3(kClusterBM256Threads, 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = cluster_a_group_size;
        attribute.val.clusterDim.y = 1;
        attribute.val.clusterDim.z = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        if (use_cluster_a_exact_n2d2) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<4, true>, d_input_map_m128,
              d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        } else if (use_cluster_a_exact_n1d8) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<4, false, true>,
              d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k));
        } else if (use_cluster_a_exact_n1d4) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<4, false, false, true>,
              d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k));
        } else if (use_cluster_a_exact_n2d4) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<2, false, false, false, true>,
              d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k));
        } else if (use_cluster_a_exact_id18) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config,
              general_m128_cluster_a_spatial_c64_k64_kernel<4, false, false, false, false, true>,
              d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k));
        } else if (use_cluster_a_group4) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<4, false>, d_input_map_m128,
              d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        } else {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m128_cluster_a_spatial_c64_k64_kernel<2, false>, d_input_map_m128,
              d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        }
      } else if (use_hybrid_cluster_a4_exact_p15) {
        cudaLaunchConfig_t config{};
        config.gridDim = grid_hybrid_cluster_a4;
        config.blockDim = dim3(256, 1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = 4;
        attribute.val.clusterDim.y = 1;
        attribute.val.clusterDim.z = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
            &config, general_hybrid_c96_exact_p15_cluster_a4_kernel, d_input_map_m128,
            d_input_map_hybrid_c32, d_weight_map_m128, d_output));
      } else if (use_hybrid_c64_c32) {
        if (use_hybrid_compact_p1_c96) {
          general_hybrid_main_exact_h17_w840_kernel<<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
              d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output);
        } else if (use_hybrid_exact_p15) {
          general_m128n256_hybrid_c64_c32_b2a3_kernel<false, false, false, true>
              <<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
                  d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c64_groups, c32_groups, opts.k);
        } else if (use_hybrid_exact_w31) {
          general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true, false, false, false, false,
                                                      false, true>
              <<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
                  d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c64_groups, c32_groups, opts.k);
        } else if (use_hybrid_exact_spatial) {
          if (use_hybrid_exact_h16_w840) {
            general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true, false, false, true>
                <<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output, opts.n,
                    opts.d, opts.h, opts.w, c64_groups, c32_groups, opts.k);
          } else {
            general_m128n256_hybrid_c64_c32_b2a3_kernel<false, true>
                <<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output, opts.n,
                    opts.d, opts.h, opts.w, c64_groups, c32_groups, opts.k);
          }
        } else {
          general_m128n256_hybrid_c64_c32_b2a3_kernel<false, false>
              <<<grid_m128, kHybridC64C32Threads, 0, stream>>>(
                  d_input_map_m128, d_input_map_hybrid_c32, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c64_groups, c32_groups, opts.k);
        }
      } else if (use_m256_cluster_b_c64_k64 || use_m256_cluster_b_c32) {
        // Cluster launch is required for DSM/TMA multicast.  Arithmetic is
        // still two independent, architecturally legal 1SM M128N256 MMAs;
        // only the activation B tile is shared across the two CTA ranks. The
        // C64 variant reuses one publication across two K32 halves.
        cudaLaunchConfig_t config{};
        config.gridDim = grid_m256_cluster_b;
        config.blockDim = dim3((use_m256_cluster_b_c64_exact_id40 ||
                                use_m256_cluster_b_c64_eight_warp_store || use_padded_m256)
                                   ? 256
                                   : kClusterBM256Threads,
                               1, 1);
        config.dynamicSmemBytes = 0;
        config.stream = stream;
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeClusterDimension;
        attribute.val.clusterDim.x = use_m256_cluster_b_c64_exact_id40 ? 4 : 2;
        attribute.val.clusterDim.y = 1;
        attribute.val.clusterDim.z = 1;
        config.attrs = &attribute;
        config.numAttrs = 1;
        if (use_m256_cluster_b_c64_exact_id40) {
          config.gridDim = dim3(4 * 33, 1, 4);
        }
        if (use_m256_cluster_b_c64_k64 && use_padded_m256_k160) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c64_k64_kernel<false, 160, false, true, true>,
              d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output, opts.n, opts.d,
              opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_k64 && use_padded_m256_k192) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c64_k64_kernel<false, 192, false, true, true>,
              d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output, opts.n, opts.d,
              opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_exact_id40) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c64_k64_kernel<false, 0, true, true>,
              d_input_map_m128, d_input_map_id40_ptail_c64, d_input_map_id40_qtail_c64,
              d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_exact_d4_c128) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c64_k64_kernel<false, 0, false, true, true>,
              d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output, opts.n, opts.d,
              opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_eight_warp_store) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c64_k64_kernel<false, 0, false, true>,
              d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output, opts.n, opts.d,
              opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_k64 && use_m256_cluster_b_c64_optimized_partial) {
          RETURN_IF_CUDA_ERROR(
              cudaLaunchKernelEx(&config, general_m256_cluster_b_c64_k64_kernel<true>,
                                 d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output,
                                 opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        } else if (use_m256_cluster_b_c64_k64) {
          RETURN_IF_CUDA_ERROR(
              cudaLaunchKernelEx(&config, general_m256_cluster_b_c64_k64_kernel<false>,
                                 d_input_map_m128, nullptr, nullptr, d_weight_map_m128, d_output,
                                 opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
        } else if (use_partial_m128_epilogue) {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
              &config, general_m256_cluster_b_c32_kernel<true>, d_input_map_m128, d_weight_map_m128,
              d_output, opts.n, opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k));
        } else {
          RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(&config, general_m256_cluster_b_c32_kernel<false>,
                                                  d_input_map_m128, d_weight_map_m128, d_output,
                                                  opts.n, opts.d, opts.h, opts.w, c32_groups,
                                                  c16_groups, opts.k));
        }
      } else if (use_k64_c64_b2a3_k32a) {
        if (k64_c64_exact_kout == 96) {
          general_m128n256_k64_c64_b2a3_k32a_kernel<false, false, 96>
              <<<grid_m128, kK64C64Threads, 0, stream>>>(d_input_map_m128, d_weight_map_m128,
                                                         d_output, opts.n, opts.d, opts.h, opts.w,
                                                         c64_groups, opts.k);
        } else if (k64_c64_exact_kout == 120) {
          general_m128n256_k64_c64_b2a3_k32a_kernel<false, false, 120>
              <<<grid_m128, kK64C64Threads, 0, stream>>>(d_input_map_m128, d_weight_map_m128,
                                                         d_output, opts.n, opts.d, opts.h, opts.w,
                                                         c64_groups, opts.k);
        } else if (use_k64_c64_exact_k128) {
          general_m128n256_k64_c64_b2a3_k32a_kernel<false, true>
              <<<grid_m128, kK64C64Threads, 0, stream>>>(d_input_map_m128, d_weight_map_m128,
                                                         d_output, opts.n, opts.d, opts.h, opts.w,
                                                         c64_groups, opts.k);
        } else if (use_partial_m128_epilogue) {
          general_m128n256_k64_c64_b2a3_k32a_kernel<true><<<grid_m128, kK64C64Threads, 0, stream>>>(
              d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k);
        } else {
          general_m128n256_k64_c64_b2a3_k32a_kernel<false>
              <<<grid_m128, kK64C64Threads, 0, stream>>>(d_input_map_m128, d_weight_map_m128,
                                                         d_output, opts.n, opts.d, opts.h, opts.w,
                                                         c64_groups, opts.k);
        }
      } else if (use_c16_path) {
        general_m128n256_k32_deep_ilp_kernel<<<grid_m128, kDeepIlpThreads, 0, stream>>>(
            d_input_map_m128, d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w,
            c32_groups, c16_groups, opts.k, 0, 0);
      } else {
        if (use_compact_spatial) {
          if (use_compact_qtail_q2_single_launch && use_compact_ptail1_single_launch) {
            general_m128n256_k32_deep_b_c32_kernel<false, true, true, true>
                <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_compact_p32, d_input_map_compact_q8,
                    d_input_map_compact_q4, d_weight_map_m128, d_output, opts.n, opts.d, opts.h,
                    opts.w, c32_groups, c16_groups, opts.k, compact_full_q_tiles,
                    compact_full_p_tiles, compact_p_tail, compact_q_tail);
          } else if (use_compact_qtail_q2_single_launch) {
            general_m128n256_k32_deep_b_c32_kernel<false, true, false, true>
                <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_compact_p32, d_input_map_compact_q8,
                    d_input_map_compact_q4, d_weight_map_m128, d_output, opts.n, opts.d, opts.h,
                    opts.w, c32_groups, c16_groups, opts.k, compact_full_q_tiles,
                    compact_full_p_tiles, compact_p_tail, compact_q_tail);
          } else if (use_compact_ptail1_single_launch) {
            general_m128n256_k32_deep_b_c32_kernel<false, true, true>
                <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_compact_p32, d_input_map_compact_q8, nullptr,
                    d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
                    c16_groups, opts.k, compact_full_q_tiles, compact_full_p_tiles, compact_p_tail,
                    compact_q_tail);
          } else {
            general_m128n256_k32_deep_b_c32_kernel<false, true>
                <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                    d_input_map_m128, d_input_map_compact_p32, d_input_map_compact_q8, nullptr,
                    d_weight_map_m128, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
                    c16_groups, opts.k, compact_full_q_tiles, compact_full_p_tiles, compact_p_tail,
                    compact_q_tail);
          }
        } else if (exact_aligned_kout == 96) {
          general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 96>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        } else if (exact_aligned_kout == 120) {
          general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 120>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        } else if (exact_aligned_kout == 160) {
          general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, false, 160>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        } else if (use_exact_p15_full_q_m128) {
          general_m128n256_k32_deep_b_c32_kernel<false, false, false, false, true>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        } else if (use_partial_m128_epilogue) {
          general_m128n256_k32_deep_b_c32_kernel<true, false>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        } else {
          general_m128n256_k32_deep_b_c32_kernel<false, false>
              <<<grid_m128, kDeepBC32Threads, 0, stream>>>(
                  d_input_map_m128, nullptr, nullptr, nullptr, d_weight_map_m128, d_output, opts.n,
                  opts.d, opts.h, opts.w, c32_groups, c16_groups, opts.k, -1, 0, 0, 0);
        }
      }
    }
    return Status::Cuda(cudaGetLastError());
  };

  auto launch_tail = [&](cudaStream_t stream) -> Status {
    if (m64_tiles > 0) {
      if (use_m64n128_d1_c32_micro) {
        general_m64n128_d1_c32_micro_kernel<<<
            dim3(kM64N128MicroFullTasks + kM64N128MicroCompactTasks, 1, 1), kM64N128MicroThreads, 0,
            stream>>>(d_input_map_m64, d_input_map_m64_compact_q4, d_weight_map_m64, d_output,
                      opts.h, opts.w);
      } else if (use_m32_d1_c128_shallow) {
        if (use_m32_d1_c128_shallow_cluster4) {
          cudaLaunchConfig_t config{};
          config.gridDim = grid_m32_shallow_cluster4;
          config.blockDim = dim3(kM32P16Threads, 1, 1);
          config.dynamicSmemBytes = 0;
          config.stream = stream;
          cudaLaunchAttribute attribute{};
          attribute.id = cudaLaunchAttributeClusterDimension;
          attribute.val.clusterDim.x = 4;
          attribute.val.clusterDim.y = 1;
          attribute.val.clusterDim.z = 1;
          config.attrs = &attribute;
          config.numAttrs = 1;
          RETURN_IF_CUDA_ERROR(
              cudaLaunchKernelEx(&config, general_m32n256_d1_c128_shallow_c64_kernel<true, true>,
                                 d_input_map_m64, d_weight_map_m64, d_output, opts.h, opts.w));
        } else if (use_m32_d1_c128_shallow_exact) {
          general_m32n256_d1_c128_shallow_c64_kernel<true><<<grid_m64, kM32P16Threads, 0, stream>>>(
              d_input_map_m64, d_weight_map_m64, d_output, opts.h, opts.w);
        } else {
          general_m32n256_d1_c128_shallow_c64_kernel<false>
              <<<grid_m64, kM32P16Threads, 0, stream>>>(d_input_map_m64, d_weight_map_m64, d_output,
                                                        opts.h, opts.w);
        }
      } else if (use_m32_p16_c64) {
        general_m32n256_k64_p16_b2a3_c64_kernel<<<grid_m64, kM32P16Threads, 0, stream>>>(
            d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups,
            opts.k, m64_output_base);
      } else if (use_m32_path) {
        general_m32n256_k32_p16_b2a3_c32_kernel<<<grid_m64, kM32P16Threads, 0, stream>>>(
            d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
            c16_groups, opts.k, m64_output_base);
      } else if (use_m64_p16_c64) {
        if (use_m64_cluster_b_c64) {
          cudaLaunchConfig_t config{};
          config.gridDim = grid_m64_cluster_b;
          config.blockDim = dim3(256, 1, 1);
          config.dynamicSmemBytes = 0;
          config.stream = stream;
          cudaLaunchAttribute attribute{};
          attribute.id = cudaLaunchAttributeClusterDimension;
          attribute.val.clusterDim.x = 2;
          attribute.val.clusterDim.y = 1;
          attribute.val.clusterDim.z = 1;
          config.attrs = &attribute;
          config.numAttrs = 1;
          if (opts.d == 2) {
            RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
                &config, general_m128_cluster_b_m64_p16_c64_kernel<2, true, false>, d_input_map_m64,
                d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
          } else {
            RETURN_IF_CUDA_ERROR(cudaLaunchKernelEx(
                &config, general_m128_cluster_b_m64_p16_c64_kernel<3, true>, d_input_map_m64,
                d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c64_groups, opts.k));
          }
        } else if (use_m64_p16_c64_exact) {
          general_m64n256_k64_p16_b2a6_c64_kernel<true><<<grid_m64, kM64P16Threads, 0, stream>>>(
              d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k, m64_output_base);
        } else {
          general_m64n256_k64_p16_b2a6_c64_kernel<false><<<grid_m64, kM64P16Threads, 0, stream>>>(
              d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w,
              c64_groups, opts.k, m64_output_base);
        }
      } else if (use_m64_p16) {
        general_m64n256_k32_p16_b2a3_c32_kernel<<<grid_m64, kM64P16Threads, 0, stream>>>(
            d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
            c16_groups, opts.k, m64_output_base);
      } else if (use_c16_path) {
        general_m64n256_k32_tail_kernel<<<grid_m64, kTailThreads, 0, stream>>>(
            d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
            c16_groups, opts.k, m64_output_base);
      } else {
        general_m64n256_k32_deep_b_c32_multi_issuer_tail_kernel<<<
            grid_m64, kM64DeepBC32MultiIssuerThreads, 0, stream>>>(
            d_input_map_m64, d_weight_map_m64, d_output, opts.n, opts.d, opts.h, opts.w, c32_groups,
            c16_groups, opts.k, m64_output_base);
      }
    }
    return Status::Cuda(cudaGetLastError());
  };

  dim3 grid_split_compact_edges(
      compact_q2_spatial_tasks - compact_full_q_tiles * compact_full_p_tiles, 1,
      opts.n * opts.d * m128_tiles);
  auto launch_spatial_edge = [&](cudaStream_t stream) -> Status {
    if (use_hybrid_compact_p1_c96) {
      dim3 grid_hybrid_ptail1(7, 1, 4);
      general_hybrid_ptail1_exact_h17_w840_kernel<<<grid_hybrid_ptail1, kHybridC64C32Threads, 0,
                                                    stream>>>(
          d_input_map_compact_p1_c64, d_input_map_compact_p32, d_weight_map_m128, d_output);
    } else if (use_split_cluster_a_compact_edges) {
      general_id18_p1_q1_compact_edge_kernel<<<grid_split_compact_edges, 128, 0, stream>>>(
          d_input_map_compact_p32, d_input_map_compact_q4, d_weight_map_m128, d_output);
    }
    return Status::Cuda(cudaGetLastError());
  };

  if (part != LaunchPart::kAuxiliary) {
    Status status = launch_main(stream);
    if (!status.ok()) return status;
  }
  if (part != LaunchPart::kMain) {
    Status status = launch_tail(stream);
    if (!status.ok()) return status;
    status = launch_spatial_edge(stream);
    if (!status.ok()) return status;
  }

#undef RETURN_IF_CUDA_ERROR
  return Status::Success();
}

}  // namespace flashinfer::conv3d::patchshift::host

// Descriptor construction shares the launch policy constants and kernel
// definitions above. Keep it in this CUDA translation unit so the public
// kernel headers are instantiated exactly once in the JIT library.
#include "prepare_descriptors.inl"
