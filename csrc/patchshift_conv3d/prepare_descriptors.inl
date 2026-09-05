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
#include <flashinfer/conv3d/patchshift/weight_layout.cuh>

#include "launcher.cuh"
#include "tensor_maps.cuh"

namespace flashinfer::conv3d::patchshift::host {

namespace detail = ::flashinfer::conv3d::patchshift::detail;

Status PrepareDescriptors(DescriptorWorkspace* workspace, Element* input, Element* packed_m128,
                          Element* packed_m64, Element* packed_m32, const Conv3dProblem& problem,
                          int multi_processor_count, cudaStream_t stream) {
  auto const& opts = problem;
  struct {
    int multiProcessorCount;
  } device_prop{multi_processor_count};

  using namespace detail;
#include "select_policy.inl"

  DescriptorWorkspace host_workspace{};
  CUresult driver_error = CUDA_SUCCESS;
#define RETURN_IF_DRIVER_ERROR(call)       \
  do {                                     \
    driver_error = (call);                 \
    if (driver_error != CUDA_SUCCESS) {    \
      return Status::Driver(driver_error); \
    }                                      \
  } while (0)

  int supergroup_count = patchshift::Supergroups(problem.c);
  if (m128_tiles > 0) {
    bool use_c64_map = use_cluster_a_spatial_c64_k64 || use_hybrid_c64_c32 ||
                       use_m256_cluster_b_c64_k64 || use_k64_c64_b2a3_k32a;
    int input_p = use_hybrid_cluster_a4_exact_p15 ? kHybridExactP15InputP : kMainInputP;
    if (use_c64_map) {
      RETURN_IF_DRIVER_ERROR(MakeInputC64Map(&host_workspace.input_m128, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c, kPitch,
                                             input_p));
    } else if (use_c16_path) {
      RETURN_IF_DRIVER_ERROR(MakeInputMap(&host_workspace.input_m128, input, problem.n, problem.d,
                                          problem.h, problem.w, problem.c, kPitch, input_p));
    } else {
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_m128, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c, kPitch,
                                             input_p));
    }
    RETURN_IF_DRIVER_ERROR(MakePackedWeightMap(&host_workspace.weight_m128, packed_m128,
                                               m128_tiles * supergroup_count * 3, kMainM,
                                               patchshift::kK16GroupsPerPackedStage));

    if (use_m256_cluster_b_c64_exact_id40) {
      RETURN_IF_DRIVER_ERROR(MakeInputC64Map(&host_workspace.input_id40_ptail_c64, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c,
                                             kId40PTailPitch, kId40PTailInputP));
      RETURN_IF_DRIVER_ERROR(MakeInputC64Map(&host_workspace.input_id40_qtail_c64, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c,
                                             kId40CompactPitch, kId40CompactInputP));
    }
    if (use_hybrid_c64_c32) {
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_hybrid_c32, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c, kPitch,
                                             input_p));
    }
    if (use_compact_spatial) {
      int compact_p_pitch = use_compact_ptail1_single_launch ? kCompactPTail1Pitch : kCompactPitchP;
      int compact_p_input_p =
          use_compact_ptail1_single_launch ? kCompactPTail1InputP : kCompactPInputP;
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_compact_p32, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c,
                                             compact_p_pitch, compact_p_input_p));
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_compact_q8, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c,
                                             kCompactPitchQ, kCompactQInputP));
      if (use_compact_qtail_q2_single_launch) {
        RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_compact_q4, input, problem.n,
                                               problem.d, problem.h, problem.w, problem.c,
                                               kCompactQ2Pitch, kCompactQ2InputP));
      }
    }
    if (use_hybrid_compact_p1_c96) {
      RETURN_IF_DRIVER_ERROR(MakeInputC64Map(&host_workspace.input_compact_p1_c64, input, problem.n,
                                             problem.d, problem.h, problem.w, problem.c,
                                             kCompactPTail1Pitch, kCompactPTail1InputP));
    }
  }

  if (m64_tiles > 0) {
    if (use_m64n128_d1_c32_micro) {
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_m64, input, problem.n, problem.d,
                                             problem.h, problem.w, problem.c, kM64N128MicroPitch,
                                             kM64N128MicroInputP));
    } else if (use_m32_p16_c64 || use_m64_p16_c64) {
      RETURN_IF_DRIVER_ERROR(MakeInputC64Map(&host_workspace.input_m64, input, problem.n, problem.d,
                                             problem.h, problem.w, problem.c, kPitch,
                                             kM64P16InputP));
    } else if (use_c16_path) {
      RETURN_IF_DRIVER_ERROR(MakeInputMap(&host_workspace.input_m64, input, problem.n, problem.d,
                                          problem.h, problem.w, problem.c, kPitch, kTailInputP));
    } else {
      int input_p = (use_m32_path || use_m64_p16) ? kM64P16InputP : kTailInputP;
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(&host_workspace.input_m64, input, problem.n, problem.d,
                                             problem.h, problem.w, problem.c, kPitch, input_p));
    }

    int tile_m = use_m32_path ? 32 : 64;
    Element* packed = use_m32_path ? packed_m32 : packed_m64;
    size_t tile_elements = patchshift::PackedWeightNumel(problem.c, tile_m, tile_m);
    packed += size_t(m64_output_base / tile_m) * tile_elements;
    RETURN_IF_DRIVER_ERROR(MakePackedWeightMap(&host_workspace.weight_m64, packed,
                                               m64_tiles * supergroup_count * 3, tile_m,
                                               patchshift::kK16GroupsPerPackedStage));

    if (use_m64n128_d1_c32_micro) {
      RETURN_IF_DRIVER_ERROR(MakeInputC32Map(
          &host_workspace.input_m64_compact_q4, input, problem.n, problem.d, problem.h, problem.w,
          problem.c, kM64N128MicroCompactPitch, kM64N128MicroCompactInputP));
    }
  }

#undef RETURN_IF_DRIVER_ERROR
  cudaError_t runtime_error = cudaMemcpyAsync(workspace, &host_workspace, sizeof(host_workspace),
                                              cudaMemcpyHostToDevice, stream);
  if (runtime_error != cudaSuccess) return Status::Cuda(runtime_error);
  // host_workspace is stack storage. The explicit prepare phase may
  // synchronize, while the subsequent Launch path remains asynchronous and
  // CUDA-graph capturable.
  return Status::Cuda(cudaStreamSynchronize(stream));
}

}  // namespace flashinfer::conv3d::patchshift::host
