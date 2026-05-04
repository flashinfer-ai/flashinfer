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
#ifndef FLASHINFER_GROUP_GEMM_MXFP4_GROUPWISE_SM120_CUH_
#define FLASHINFER_GROUP_GEMM_MXFP4_GROUPWISE_SM120_CUH_

#include <cassert>
#include <iterator>

#include "../allocator.h"
#include "../cutlass_utils.cuh"
#include "../utils.cuh"

namespace flashinfer {

namespace group_gemm {

using namespace cute;

// Function to safely offset an pointer that may contain sub-byte types (FP4/INT4)
namespace {
template <class T>
__host__ __device__ __forceinline__ constexpr T* safe_inc_ptr(T* ptr, size_t offset) {
  constexpr int adjustment = (sizeof_bits<T>::value < 8) ? (8 / sizeof_bits<T>::value) : 1;
  assert(offset % adjustment == 0 && "Attempt to offset index to sub-byte");
  return ptr + offset / adjustment;
}
}  // namespace

template <typename T>
using ptr_t = T*;

template <bool SwapAB, int ScaleGranularity, typename ScaleConfig, typename ElementA,
          typename ElementB, typename ElementSFA, typename ElementSFB, typename ElementD,
          typename ProblemShape, typename StrideA, typename StrideB, typename StrideD,
          typename LayoutSFA, typename LayoutSFB>
__global__ void compute_sm120_cutlass_group_gemm_args(
    ElementA* A, ElementB* B, ElementSFA* SFA, ElementSFB* SFB, ElementD* D, int* m_indptr, int n,
    int k, int num_groups, ProblemShape* problem_sizes, const ElementA** A_ptr,
    const ElementB** B_ptr, const ElementSFA** SFA_ptr, const ElementSFB** SFB_ptr,
    ElementD** D_ptr, StrideA* stride_A, StrideB* stride_B, StrideD* stride_D,
    LayoutSFA* layout_SFA, LayoutSFB* layout_SFB) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_groups) {
    return;
  }
  constexpr size_t alignment_swizzled_mn = 128;
  constexpr size_t alignment_swizzled_k = static_cast<size_t>(ScaleGranularity) * 4;
  size_t sf_n = (static_cast<size_t>(n) + alignment_swizzled_mn - 1) / alignment_swizzled_mn *
                alignment_swizzled_mn;
  size_t swizzled_k = (static_cast<size_t>(k) + alignment_swizzled_k - 1) / alignment_swizzled_k *
                      alignment_swizzled_k;
  size_t sf_k = swizzled_k / static_cast<size_t>(ScaleGranularity);
#if (__CUDACC_VER_MAJOR__ >= 12 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("griddepcontrol.wait;");
  asm volatile("griddepcontrol.launch_dependents;");
#endif
  int m_offset = m_indptr[i];
  int m_offset_next = m_indptr[i + 1];
  size_t m = static_cast<size_t>(m_offset_next) - static_cast<size_t>(m_offset);
  // This formulation ensures that sf_m_offset_next - sf_m_offset >= m_offset_next - m_offset
  size_t sf_m_offset =
      (static_cast<size_t>(m_offset) + static_cast<size_t>(i) * (alignment_swizzled_mn - 1)) /
      alignment_swizzled_mn * alignment_swizzled_mn;

  if constexpr (SwapAB) {
    problem_sizes[i] = ProblemShape(n, m, k);
    stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {n, k, 1});
    stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {m, k, 1});
    stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {n, m, 1});
    B_ptr[i] = safe_inc_ptr(B, static_cast<size_t>(m_offset) * static_cast<size_t>(k));
    A_ptr[i] =
        safe_inc_ptr(A, static_cast<size_t>(i) * static_cast<size_t>(n) * static_cast<size_t>(k));
    D_ptr[i] = safe_inc_ptr(D, static_cast<size_t>(m_offset) * static_cast<size_t>(n));
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(static_cast<int>(sf_n), static_cast<int>(m), static_cast<int>(swizzled_k), 1));
    SFA_ptr[i] = safe_inc_ptr(
        SFA, static_cast<size_t>(i) * static_cast<size_t>(sf_n) * static_cast<size_t>(sf_k));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(static_cast<int>(sf_n), static_cast<int>(m), static_cast<int>(swizzled_k), 1));
    SFB_ptr[i] = safe_inc_ptr(SFB, static_cast<size_t>(sf_m_offset) * static_cast<size_t>(sf_k));
  } else {
    problem_sizes[i] = ProblemShape(m, n, k);
    stride_A[i] = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    stride_B[i] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    stride_D[i] = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    A_ptr[i] = safe_inc_ptr(A, static_cast<size_t>(m_offset) * static_cast<size_t>(k));
    B_ptr[i] =
        safe_inc_ptr(B, static_cast<size_t>(i) * static_cast<size_t>(n) * static_cast<size_t>(k));
    D_ptr[i] = safe_inc_ptr(D, static_cast<size_t>(m_offset) * static_cast<size_t>(n));
    layout_SFA[i] = ScaleConfig::tile_atom_to_shape_SFA(
        make_shape(static_cast<int>(m), static_cast<int>(sf_n), static_cast<int>(swizzled_k), 1));
    SFA_ptr[i] = safe_inc_ptr(SFA, static_cast<size_t>(sf_m_offset) * static_cast<size_t>(sf_k));
    layout_SFB[i] = ScaleConfig::tile_atom_to_shape_SFB(
        make_shape(static_cast<int>(m), static_cast<int>(sf_n), static_cast<int>(swizzled_k), 1));
    SFB_ptr[i] = safe_inc_ptr(
        SFB, static_cast<size_t>(i) * static_cast<size_t>(sf_n) * static_cast<size_t>(sf_k));
  }
}

template <int TileM, int TileN, int TileK, bool SwapAB, typename DTypeInA, typename DTypeInB,
          typename DTypeSFA, typename DTypeSFB, typename DTypeOut>
cudaError_t CutlassMXFP4GroupwiseScaledGroupGEMMSM120(
    void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,
    size_t float_buffer_size_in_bytes, DTypeInA* A, DTypeInB* B, DTypeSFA* SFA, DTypeSFB* SFB,
    DTypeOut* D, int* m_indptr, int n, int k, int num_groups, cudaStream_t stream, int device_id);

}  // namespace group_gemm

}  // namespace flashinfer

// There is a bug in some versions of GCC where large symbol names lead to issues
// In order to not invoke those bugs, we opt to use a macro to expand into a
// simplified symbol name
// TileM is our M tile for the GEMM
// TileN is our N tile for the GEMM
// TileK is our K tile for the GEMM
// SwapAB is whether to swap A and B
// DTypeInA: data type of input matrix A (m × k)
// DTypeInB: data type of input matrix B (k × n) — block-scaled MXFP4 format
// DTypeSFA: data type of scale factors for A (m × (k / ScaleGranularity))
// DTypeSFB: data type of scale factors for B (n × (k / ScaleGranularity))
// DTypeOut: data type of output matrix D (m × n)
// DTypeInAName, DTypeInBName, DTypeSFAName, DTypeSFBName, DTypeOutName:
//     stringified type names used to form unique C++ symbol names (e.g., "fp4", "int4", "fp8",
//     "fp16", "bf16")

#define INSTANTIATE_GROUP_GEMM_MXFP4_GROUPWISE_SM120(                                                                                                                             \
    TileM, TileN, TileK, SwapAB, DTypeInA, DTypeInB, DTypeSFA, DTypeSFB, DTypeOut, DTypeInAName,                                                                                  \
    DTypeInBName, DTypeSFAName, DTypeSFBName, DTypeOutName)                                                                                                                       \
  inline cudaError_t                                                                                                                                                              \
      CutlassMXFP4GroupwiseScaledGroupGEMMSM120_##TileM##_##TileN##_##TileK##_##SwapAB##_##DTypeInAName##_##DTypeInBName##_##DTypeSFAName##_##DTypeSFBName##_##DTypeOutName(      \
          void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,                                                                                                  \
          size_t float_buffer_size_in_bytes, DTypeInA* A, DTypeInB* B, DTypeSFA* SFA,                                                                                             \
          DTypeSFB* SFB, DTypeOut* D, int* m_indptr, int n, int k, int num_groups,                                                                                                \
          cudaStream_t stream, int device_id) {                                                                                                                                   \
    if (num_groups == 0) {                                                                                                                                                        \
      return cudaSuccess;                                                                                                                                                         \
    }                                                                                                                                                                             \
    using ElementA = std::conditional_t<SwapAB, DTypeInB, DTypeInA>;                                                                                                              \
    using ElementSFA = std::conditional_t<SwapAB, DTypeSFB, DTypeSFA>;                                                                                                            \
    constexpr int AlignmentA = 128 / cutlass::sizeof_bits<ElementA>::value;                                                                                                       \
    using ElementB = std::conditional_t<SwapAB, DTypeInA, DTypeInB>;                                                                                                              \
    using ElementSFB = std::conditional_t<SwapAB, DTypeSFA, DTypeSFB>;                                                                                                            \
    constexpr int AlignmentB = 128 / cutlass::sizeof_bits<ElementB>::value;                                                                                                       \
    using ElementD = DTypeOut;                                                                                                                                                    \
    using ElementC = void;                                                                                                                                                        \
    using LayoutC = void;                                                                                                                                                         \
    constexpr int AlignmentC = 0;                                                                                                                                                 \
    constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;                                                                                                       \
    FLASHINFER_CHECK(k % std::max(AlignmentA, AlignmentB) == 0, "k must be divisible by %d",                                                                                      \
                     max(AlignmentA, AlignmentB));                                                                                                                                \
    FLASHINFER_CHECK(n % AlignmentD == 0, "n must be divisible by %d", AlignmentD);                                                                                               \
    using ElementAccumulator = float;                                                                                                                                             \
    using ElementCompute = float;                                                                                                                                                 \
    using ElementAMainloop = cute::tuple<ElementA, ElementSFA>;                                                                                                                   \
    using ElementBMainloop = cute::tuple<ElementB, ElementSFB>;                                                                                                                   \
    using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;                                                                                                  \
    using LayoutA = cutlass::layout::RowMajor;                                                                                                                                    \
    using LayoutB = cutlass::layout::ColumnMajor;                                                                                                                                 \
    using LayoutD =                                                                                                                                                               \
        std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;                                                                                      \
    using ClusterShape = Shape<_1, _1, _1>;                                                                                                                                       \
    using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;                                                                                                 \
    using ThreadBlockShape = Shape<Int<TileM>, Int<TileN>, Int<TileK>>;                                                                                                           \
    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<                                                                                         \
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ThreadBlockShape,                                                                                        \
        ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,                                                                                        \
        ElementCompute, ElementC, LayoutD*, AlignmentD, ElementD, LayoutD*, AlignmentD,                                                                                           \
        EpilogueSchedule>::CollectiveOp;                                                                                                                                          \
    using MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto;                                                                                                       \
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<                                                                                             \
        cutlass::arch::Sm120, cutlass::arch::OpClassBlockScaledTensorOp, ElementAMainloop,                                                                                        \
        LayoutA*, AlignmentA, ElementBMainloop, LayoutB*, AlignmentB, ElementAccumulator,                                                                                         \
        ThreadBlockShape, ClusterShape,                                                                                                                                           \
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(                                                                                                       \
            sizeof(typename CollectiveEpilogue::SharedStorage))>,                                                                                                                 \
        MainloopSchedule>::CollectiveOp;                                                                                                                                          \
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop,                                                                                     \
                                                            CollectiveEpilogue, void>;                                                                                            \
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;                                                                                                         \
    using StrideA = typename Gemm::GemmKernel::InternalStrideA;                                                                                                                   \
    using StrideB = typename Gemm::GemmKernel::InternalStrideB;                                                                                                                   \
    using StrideD = typename Gemm::GemmKernel::InternalStrideD;                                                                                                                   \
    using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;                                                                                      \
    using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFA;                                                                                           \
    using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::InternalLayoutSFB;                                                                                           \
    constexpr int ScaleGranularity = Gemm::GemmKernel::CollectiveMainloop::TiledMma::SFVecSize;                                                                                   \
    static_assert(ScaleGranularity == 32);                                                                                                                                        \
    AlignedAllocator allocator(int_buffer, int_buffer_size_in_bytes);                                                                                                             \
    auto problem_sizes = allocator.aligned_alloc<typename ProblemShape::UnderlyingProblemShape>(                                                                                  \
        num_groups * sizeof(typename ProblemShape::UnderlyingProblemShape), 16,                                                                                                   \
        "sm120_groupwise_group_gemm_problem_sizes");                                                                                                                              \
    auto A_ptr = allocator.aligned_alloc<const typename Gemm::ElementA*>(                                                                                                         \
        num_groups * sizeof(const typename Gemm::ElementA*), 16,                                                                                                                  \
        "sm120_groupwise_group_gemm_A_ptr");                                                                                                                                      \
    auto B_ptr = allocator.aligned_alloc<const typename Gemm::ElementB*>(                                                                                                         \
        num_groups * sizeof(const typename Gemm::ElementB*), 16,                                                                                                                  \
        "sm120_groupwise_group_gemm_B_ptr");                                                                                                                                      \
    auto D_ptr = allocator.aligned_alloc<typename Gemm::EpilogueOutputOp::ElementOutput*>(                                                                                        \
        num_groups * sizeof(typename Gemm::EpilogueOutputOp::ElementOutput*), 16,                                                                                                 \
        "sm120_groupwise_group_gemm_D_ptr");                                                                                                                                      \
    auto SFA_ptr = allocator.aligned_alloc<const ElementSFA*>(                                                                                                                    \
        num_groups * sizeof(const ElementSFA*), 16, "sm120_groupwise_group_gemm_SFA_ptr");                                                                                        \
    auto SFB_ptr = allocator.aligned_alloc<const ElementSFB*>(                                                                                                                    \
        num_groups * sizeof(const ElementSFB*), 16, "sm120_groupwise_group_gemm_SFB_ptr");                                                                                        \
    auto stride_A = allocator.aligned_alloc<StrideA>(num_groups * sizeof(StrideA), 16,                                                                                            \
                                                     "sm120_groupwise_group_gemm_stride_A");                                                                                      \
    auto stride_B = allocator.aligned_alloc<StrideB>(num_groups * sizeof(StrideB), 16,                                                                                            \
                                                     "sm120_groupwise_group_gemm_stride_B");                                                                                      \
    auto stride_D = allocator.aligned_alloc<StrideD>(num_groups * sizeof(StrideD), 16,                                                                                            \
                                                     "sm120_groupwise_group_gemm_stride_D");                                                                                      \
    auto layout_SFA = allocator.aligned_alloc<LayoutSFA>(num_groups * sizeof(LayoutSFA), 16,                                                                                      \
                                                         "sm120_groupwise_group_gemm_layout_SFA");                                                                                \
    auto layout_SFB = allocator.aligned_alloc<LayoutSFB>(num_groups * sizeof(LayoutSFB), 16,                                                                                      \
                                                         "sm120_groupwise_group_gemm_layout_SFB");                                                                                \
    int num_threads = std::min(num_groups, 1024);                                                                                                                                 \
    int num_blocks = (num_groups + num_threads - 1) / num_threads;                                                                                                                \
    cudaLaunchConfig_t config;                                                                                                                                                    \
    config.gridDim = num_blocks;                                                                                                                                                  \
    config.blockDim = num_threads;                                                                                                                                                \
    config.dynamicSmemBytes = 0;                                                                                                                                                  \
    config.stream = stream;                                                                                                                                                       \
    cudaLaunchAttribute attrs[1];                                                                                                                                                 \
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;                                                                                                             \
    attrs[0].val.programmaticStreamSerializationAllowed = true;                                                                                                                   \
    config.numAttrs = 1;                                                                                                                                                          \
    config.attrs = attrs;                                                                                                                                                         \
    auto prepare_args_kernel =                                                                                                                                                    \
        compute_sm120_cutlass_group_gemm_args<SwapAB, ScaleGranularity, ScaleConfig, ElementA,                                                                                    \
                                              ElementB, ElementSFA, ElementSFB, ElementD,                                                                                         \
                                              ProblemShape::UnderlyingProblemShape, StrideA,                                                                                      \
                                              StrideB, StrideD, LayoutSFA, LayoutSFB>;                                                                                            \
    if constexpr (SwapAB) {                                                                                                                                                       \
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, prepare_args_kernel, B, A, SFB, SFA, D,                                                                                    \
                                              m_indptr, n, k, num_groups, problem_sizes, A_ptr,                                                                                   \
                                              B_ptr, SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B,                                                                                 \
                                              stride_D, layout_SFA, layout_SFB));                                                                                                 \
    } else {                                                                                                                                                                      \
      FLASHINFER_CUDA_CALL(cudaLaunchKernelEx(&config, prepare_args_kernel, A, B, SFA, SFB, D,                                                                                    \
                                              m_indptr, n, k, num_groups, problem_sizes, A_ptr,                                                                                   \
                                              B_ptr, SFA_ptr, SFB_ptr, D_ptr, stride_A, stride_B,                                                                                 \
                                              stride_D, layout_SFA, layout_SFB));                                                                                                 \
    }                                                                                                                                                                             \
    thread_local int last_device_id = -1;                                                                                                                                         \
    thread_local int sm_count = 0;                                                                                                                                                \
    if (last_device_id != device_id) {                                                                                                                                            \
      last_device_id = device_id;                                                                                                                                                 \
      sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count();                                                                                                \
    }                                                                                                                                                                             \
    cutlass::KernelHardwareInfo hw_info;                                                                                                                                          \
    hw_info.device_id = device_id;                                                                                                                                                \
    hw_info.sm_count = sm_count;                                                                                                                                                  \
                                                                                                                                                                                  \
    typename Gemm::Arguments arguments{                                                                                                                                           \
        cutlass::gemm::GemmUniversalMode::kGrouped,                                                                                                                               \
        {num_groups, problem_sizes, /*problem_sizes_host=*/nullptr},                                                                                                              \
        {                                                                                                                                                                         \
            A_ptr,                                                                                                                                                                \
            stride_A,                                                                                                                                                             \
            B_ptr,                                                                                                                                                                \
            stride_B,                                                                                                                                                             \
            SFA_ptr,                                                                                                                                                              \
            layout_SFA,                                                                                                                                                           \
            SFB_ptr,                                                                                                                                                              \
            layout_SFB,                                                                                                                                                           \
        },                                                                                                                                                                        \
        {                                                                                                                                                                         \
            {},                                                                                                                                                                   \
            nullptr,                                                                                                                                                              \
            nullptr,                                                                                                                                                              \
            D_ptr,                                                                                                                                                                \
            stride_D,                                                                                                                                                             \
        },                                                                                                                                                                        \
        hw_info};                                                                                                                                                                 \
    auto& fusion_args = arguments.epilogue.thread;                                                                                                                                \
    fusion_args.alpha = 1.0f;                                                                                                                                                     \
    fusion_args.beta = 0.0f;                                                                                                                                                      \
    Gemm gemm;                                                                                                                                                                    \
    size_t workspace_size = Gemm::get_workspace_size(arguments);                                                                                                                  \
    AlignedAllocator float_allocator(float_buffer, float_buffer_size_in_bytes);                                                                                                   \
    auto workspace_ptr = float_allocator.aligned_alloc<void>(                                                                                                                     \
        workspace_size, 16, "sm120_groupwise_group_gemm_float_workspace");                                                                                                        \
    CUTLASS_CHECK(gemm.can_implement(arguments));                                                                                                                                 \
    CUTLASS_CHECK(gemm.initialize(arguments, workspace_ptr));                                                                                                                     \
    CUTLASS_CHECK(gemm.run(stream, /*cuda_adapter=*/nullptr, /*launch_with_pdl=*/true));                                                                                          \
    return cudaSuccess;                                                                                                                                                           \
  }                                                                                                                                                                               \
                                                                                                                                                                                  \
  template <>                                                                                                                                                                     \
  cudaError_t CutlassMXFP4GroupwiseScaledGroupGEMMSM120<TileM, TileN, TileK, SwapAB, DTypeInA,                                                                                    \
                                                        DTypeInB, DTypeSFA, DTypeSFB, DTypeOut>(                                                                                  \
      void* int_buffer, size_t int_buffer_size_in_bytes, void* float_buffer,                                                                                                      \
      size_t float_buffer_size_in_bytes, DTypeInA* A, DTypeInB* B, DTypeSFA* SFA, DTypeSFB* SFB,                                                                                  \
      DTypeOut* D, int* m_indptr, int n, int k, int num_groups, cudaStream_t stream,                                                                                              \
      int device_id) {                                                                                                                                                            \
    return CutlassMXFP4GroupwiseScaledGroupGEMMSM120_##TileM##_##TileN##_##TileK##_##SwapAB##_##DTypeInAName##_##DTypeInBName##_##DTypeSFAName##_##DTypeSFBName##_##DTypeOutName( \
        int_buffer, int_buffer_size_in_bytes, float_buffer, float_buffer_size_in_bytes, A, B, SFA,                                                                                \
        SFB, D, m_indptr, n, k, num_groups, stream, device_id);                                                                                                                   \
  }

#endif  // FLASHINFER_GROUP_GEMM_MXFP4_GROUPWISE_SM120_CUH_
