// Copyright (c) 2026 FlashInfer team.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cutlass/device_kernel.h>
#include <flashinfer/allocator.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <flashinfer/gemm/group_gemm_sm90.cuh>
#include <limits>
#include <type_traits>

#ifndef SM90_PUSH_BF16_FAMILY_MASK
#define SM90_PUSH_BF16_FAMILY_MASK 3
#endif

#ifndef SM90_PUSH_BF16_M64_BLOCK_N
#define SM90_PUSH_BF16_M64_BLOCK_N 128
#endif

#ifndef SM90_PUSH_BF16_M64_BLOCK_K
#define SM90_PUSH_BF16_M64_BLOCK_K 128
#endif

#ifndef SM90_PUSH_BF16_M64_STAGES
#define SM90_PUSH_BF16_M64_STAGES 3
#endif

#ifndef SM90_PUSH_BF16_M64_CLUSTER_M
#define SM90_PUSH_BF16_M64_CLUSTER_M 1
#endif

#ifndef SM90_PUSH_BF16_M64_SCHEDULE
#define SM90_PUSH_BF16_M64_SCHEDULE 0
#endif

#ifndef SM90_PUSH_BF16_M128_BLOCK_N
#define SM90_PUSH_BF16_M128_BLOCK_N 128
#endif

#ifndef SM90_PUSH_BF16_M128_BLOCK_K
#define SM90_PUSH_BF16_M128_BLOCK_K 128
#endif

#ifndef SM90_PUSH_BF16_M128_STAGES
#define SM90_PUSH_BF16_M128_STAGES 3
#endif

#ifndef SM90_PUSH_BF16_M128_CLUSTER_M
#define SM90_PUSH_BF16_M128_CLUSTER_M 2
#endif

#ifndef SM90_PUSH_BF16_M128_SCHEDULE
#define SM90_PUSH_BF16_M128_SCHEDULE 1
#endif

#ifndef SM90_PUSH_BF16_SWAP_AB
#define SM90_PUSH_BF16_SWAP_AB 0
#endif

namespace flashinfer::sm90_push_bf16 {

using ProblemShape = cutlass::gemm::GroupProblemShape<cute::Shape<int, int, int>>;
using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;
using Element = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ArchTag = cutlass::arch::Sm90;
using OperatorClass = cutlass::arch::OpClassTensorOp;
using EpilogueSchedule = cutlass::epilogue::PtrArrayNoSmemWarpSpecialized;

constexpr int kAlignment = 128 / cutlass::sizeof_bits<Element>::value;
constexpr int kFamilyMask = SM90_PUSH_BF16_FAMILY_MASK;
constexpr bool kEnableM64 = (kFamilyMask & 1) != 0;
constexpr bool kEnableM128 = (kFamilyMask & 2) != 0;
constexpr bool kSwapAB = SM90_PUSH_BF16_SWAP_AB != 0;
constexpr int kNumMTileFamilies = static_cast<int>(kEnableM64) + static_cast<int>(kEnableM128);

static_assert(kFamilyMask >= 1 && kFamilyMask <= 3);
static_assert(!kSwapAB || kNumMTileFamilies == 1);

enum class MTileFamily : int32_t {
  kM64 = 0,
  kM128 = 1,
};

template <int BlockM, int BlockN, int BlockK, int Stages, int ClusterM, int Schedule, bool SwapAB>
struct GemmTraits {
  static_assert(BlockM == 64 || BlockM == 128);
  static_assert(BlockN == 64 || BlockN == 128);
  static_assert(BlockK == 64 || BlockK == 128);
  static_assert(Stages >= 2 && Stages <= 4);
  static_assert(ClusterM == 1 || ClusterM == 2);
  static_assert(Schedule == 0 || Schedule == 1);
  static_assert(Schedule == 0 || (SwapAB ? BlockN >= 128 : BlockM >= 128));
  using TileShape =
      std::conditional_t<SwapAB,
                         cute::Shape<cute::Int<BlockN>, cute::Int<BlockM>, cute::Int<BlockK>>,
                         cute::Shape<cute::Int<BlockM>, cute::Int<BlockN>, cute::Int<BlockK>>>;
  using ClusterShape = cute::Shape<cute::Int<ClusterM>, cute::_1, cute::_1>;
  using KernelSchedule =
      std::conditional_t<Schedule == 0, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
                         cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutD =
      std::conditional_t<SwapAB, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      ArchTag, OperatorClass, TileShape, ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator, ElementAccumulator,
      Element, LayoutD*, kAlignment, Element, LayoutD*, kAlignment, EpilogueSchedule>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, OperatorClass, Element, LayoutA*, kAlignment, Element, LayoutB*, kAlignment,
      ElementAccumulator, TileShape, ClusterShape, cutlass::gemm::collective::StageCount<Stages>,
      KernelSchedule>::CollectiveOp;
  using GemmKernel =
      cutlass::gemm::kernel::GemmUniversal<ProblemShape, CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
  using StrideA = typename GemmKernel::InternalStrideA;
  using StrideB = typename GemmKernel::InternalStrideB;
  using StrideC = typename GemmKernel::InternalStrideC;
  using StrideD = typename GemmKernel::InternalStrideD;
  static_assert(sizeof(typename GemmKernel::SharedStorage) <= 227 * 1024);
  static constexpr int kBlockM = BlockM;
  static constexpr int kBlockN = BlockN;
  static constexpr int kBlockK = BlockK;
  static constexpr int kStages = Stages;
  static constexpr int kClusterM = ClusterM;
  static constexpr int kSchedule = Schedule;
  static constexpr bool kSwapAB = SwapAB;
};

using M64Traits = GemmTraits<64, SM90_PUSH_BF16_M64_BLOCK_N, SM90_PUSH_BF16_M64_BLOCK_K,
                             SM90_PUSH_BF16_M64_STAGES, SM90_PUSH_BF16_M64_CLUSTER_M,
                             SM90_PUSH_BF16_M64_SCHEDULE, kSwapAB>;
using M128Traits = GemmTraits<128, SM90_PUSH_BF16_M128_BLOCK_N, SM90_PUSH_BF16_M128_BLOCK_K,
                              SM90_PUSH_BF16_M128_STAGES, SM90_PUSH_BF16_M128_CLUSTER_M,
                              SM90_PUSH_BF16_M128_SCHEDULE, kSwapAB>;
using CanonicalTraits = std::conditional_t<kEnableM128, M128Traits, M64Traits>;
using StrideA = typename CanonicalTraits::StrideA;
using StrideB = typename CanonicalTraits::StrideB;
using StrideC = typename CanonicalTraits::StrideC;
using StrideD = typename CanonicalTraits::StrideD;

#if SM90_PUSH_BF16_FAMILY_MASK & 1
static_assert(std::is_same_v<StrideA, typename M64Traits::StrideA>);
static_assert(std::is_same_v<StrideB, typename M64Traits::StrideB>);
static_assert(std::is_same_v<StrideC, typename M64Traits::StrideC>);
static_assert(std::is_same_v<StrideD, typename M64Traits::StrideD>);
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
static_assert(std::is_same_v<StrideA, typename M128Traits::StrideA>);
static_assert(std::is_same_v<StrideB, typename M128Traits::StrideB>);
static_assert(std::is_same_v<StrideC, typename M128Traits::StrideC>);
static_assert(std::is_same_v<StrideD, typename M128Traits::StrideD>);
#endif

struct Bf16ExpertSchedule {
  int64_t begin;
  int32_t rows;
  int32_t reserved;
};

struct Bf16ScheduleHeader {
  uint64_t epoch;
  int64_t row_capacity;
  int32_t num_experts;
  int32_t reserved;
};

struct RowFamilySelection {
  int64_t m128_rows;
  int64_t m64_begin;
  int64_t m64_rows;
};

__host__ __device__ constexpr RowFamilySelection select_row_families(int64_t begin, int64_t rows) {
  if constexpr (kEnableM64 && !kEnableM128) {
    return RowFamilySelection{0, begin, rows};
  }
  if constexpr (!kEnableM64 && kEnableM128) {
    return RowFamilySelection{rows, begin + rows, 0};
  }
  int64_t const full_m128_rows = rows / 128 * 128;
  int64_t const remainder = rows - full_m128_rows;
  return RowFamilySelection{full_m128_rows + (remainder > 64 ? remainder : 0),
                            begin + full_m128_rows,
                            remainder > 0 && remainder <= 64 ? remainder : 0};
}

static_assert(select_row_families(0, 0).m128_rows == 0 && select_row_families(0, 0).m64_rows == 0);
static_assert(kFamilyMask != 1 || (select_row_families(0, 65).m128_rows == 0 &&
                                   select_row_families(0, 65).m64_rows == 65));
static_assert(kFamilyMask != 2 || (select_row_families(0, 65).m128_rows == 65 &&
                                   select_row_families(0, 65).m64_rows == 0));
static_assert(kFamilyMask != 3 || (select_row_families(0, 64).m128_rows == 0 &&
                                   select_row_families(0, 64).m64_rows == 64));
static_assert(kFamilyMask != 3 || (select_row_families(0, 129).m128_rows == 128 &&
                                   select_row_families(0, 129).m64_rows == 1));

struct ScheduleWorkspaceLayout {
  size_t header;
  size_t experts;
  size_t total_bytes;
};

inline ScheduleWorkspaceLayout make_schedule_workspace_layout(int num_experts) {
  AlignedAllocator allocator;
  ScheduleWorkspaceLayout layout{};
  layout.header =
      allocator.aligned_alloc_offset(sizeof(Bf16ScheduleHeader), 16, "bf16_gemm_schedule_header");
  layout.experts =
      allocator.aligned_alloc_offset(static_cast<size_t>(num_experts) * sizeof(Bf16ExpertSchedule),
                                     16, "bf16_gemm_expert_schedule");
  layout.total_bytes = allocator.num_allocated_bytes();
  return layout;
}

struct WorkspaceView {
  UnderlyingProblemShape* problem_shapes;
  Element const** activation_ptrs;
  Element const** weight_ptrs;
  Element const** source_ptrs;
  Element** output_ptrs;
  StrideA* activation_strides;
  StrideB* weight_strides;
  StrideC* source_strides;
  StrideD* output_strides;
  void* cutlass_workspace;
};

struct WorkspaceLayout {
  size_t problem_shapes;
  size_t activation_ptrs;
  size_t weight_ptrs;
  size_t source_ptrs;
  size_t output_ptrs;
  size_t activation_strides;
  size_t weight_strides;
  size_t source_strides;
  size_t output_strides;
  size_t cutlass_workspaces[kNumMTileFamilies];
  size_t total_bytes;
};

inline WorkspaceLayout make_workspace_layout(int num_experts, size_t cutlass_workspace_bytes) {
  AlignedAllocator allocator;
  size_t const family_entries = static_cast<size_t>(num_experts) * kNumMTileFamilies;
  WorkspaceLayout layout{};
  layout.problem_shapes = allocator.aligned_alloc_offset(
      family_entries * sizeof(UnderlyingProblemShape), 16, "bf16_gemm_problem_shapes");
  layout.activation_ptrs = allocator.aligned_alloc_offset(family_entries * sizeof(Element const*),
                                                          16, "bf16_gemm_activation_ptrs");
  layout.weight_ptrs = allocator.aligned_alloc_offset(family_entries * sizeof(Element const*), 16,
                                                      "bf16_gemm_weight_ptrs");
  layout.source_ptrs = allocator.aligned_alloc_offset(family_entries * sizeof(Element const*), 16,
                                                      "bf16_gemm_source_ptrs");
  layout.output_ptrs = allocator.aligned_alloc_offset(family_entries * sizeof(Element*), 16,
                                                      "bf16_gemm_output_ptrs");
  layout.activation_strides = allocator.aligned_alloc_offset(family_entries * sizeof(StrideA), 16,
                                                             "bf16_gemm_activation_strides");
  layout.weight_strides = allocator.aligned_alloc_offset(family_entries * sizeof(StrideB), 16,
                                                         "bf16_gemm_weight_strides");
  layout.source_strides = allocator.aligned_alloc_offset(family_entries * sizeof(StrideC), 16,
                                                         "bf16_gemm_source_strides");
  layout.output_strides = allocator.aligned_alloc_offset(family_entries * sizeof(StrideD), 16,
                                                         "bf16_gemm_output_strides");
  for (int family = 0; family < kNumMTileFamilies; ++family) {
    layout.cutlass_workspaces[family] =
        allocator.aligned_alloc_offset(cutlass_workspace_bytes, 64, "bf16_gemm_cutlass_workspace");
  }
  layout.total_bytes = allocator.num_allocated_bytes();
  return layout;
}

__host__ __device__ constexpr int family_slot(MTileFamily family) {
  if (family == MTileFamily::kM64) return 0;
  return kEnableM64 ? 1 : 0;
}

template <typename T>
inline T* workspace_ptr(void* workspace, size_t offset) {
  return reinterpret_cast<T*>(static_cast<uint8_t*>(workspace) + offset);
}

inline WorkspaceView bind_workspace(void* workspace, size_t workspace_bytes, int num_experts,
                                    size_t cutlass_workspace_bytes, MTileFamily family) {
  WorkspaceLayout const layout = make_workspace_layout(num_experts, cutlass_workspace_bytes);
  if (workspace == nullptr || workspace_bytes < layout.total_bytes) return {};
  int const slot = family_slot(family);
  size_t const family_offset = static_cast<size_t>(slot) * static_cast<size_t>(num_experts);
  WorkspaceView view{};
  view.problem_shapes =
      workspace_ptr<UnderlyingProblemShape>(workspace, layout.problem_shapes) + family_offset;
  view.activation_ptrs =
      workspace_ptr<Element const*>(workspace, layout.activation_ptrs) + family_offset;
  view.weight_ptrs = workspace_ptr<Element const*>(workspace, layout.weight_ptrs) + family_offset;
  view.source_ptrs = workspace_ptr<Element const*>(workspace, layout.source_ptrs) + family_offset;
  view.output_ptrs = workspace_ptr<Element*>(workspace, layout.output_ptrs) + family_offset;
  view.activation_strides =
      workspace_ptr<StrideA>(workspace, layout.activation_strides) + family_offset;
  view.weight_strides = workspace_ptr<StrideB>(workspace, layout.weight_strides) + family_offset;
  view.source_strides = workspace_ptr<StrideC>(workspace, layout.source_strides) + family_offset;
  view.output_strides = workspace_ptr<StrideD>(workspace, layout.output_strides) + family_offset;
  view.cutlass_workspace = workspace_ptr<uint8_t>(workspace, layout.cutlass_workspaces[slot]);
  return view;
}

__device__ __forceinline__ void trap_invalid_schedule(int code) {
  printf("sm90_push_bf16_gemm: invalid prepared schedule, code=%d\n", code);
  asm volatile("trap;");
}

template <typename Traits>
__device__ __forceinline__ void prepare_family_arguments(Bf16ExpertSchedule schedule, int expert,
                                                         int n, int k, Element const* activation,
                                                         Element const* weights, Element* output,
                                                         WorkspaceView view) {
  constexpr int BlockM = Traits::kBlockM;
  RowFamilySelection const selection = select_row_families(schedule.begin, schedule.rows);
  int64_t const begin = BlockM == 64 ? selection.m64_begin : schedule.begin;
  int const rows =
      BlockM == 64 ? static_cast<int>(selection.m64_rows) : static_cast<int>(selection.m128_rows);
  if constexpr (Traits::kSwapAB) {
    view.problem_shapes[expert] = UnderlyingProblemShape(n, rows, k);
    view.activation_ptrs[expert] = weights + static_cast<int64_t>(expert) * n * k;
    view.weight_ptrs[expert] =
        rows == 0 ? activation : activation + begin * static_cast<int64_t>(k);
    view.source_ptrs[expert] = rows == 0 ? output : output + begin * static_cast<int64_t>(n);
    view.output_ptrs[expert] = rows == 0 ? output : output + begin * static_cast<int64_t>(n);
    view.activation_strides[expert] = cutlass::make_cute_packed_stride(StrideA{}, {n, k, 1});
    view.weight_strides[expert] = cutlass::make_cute_packed_stride(StrideB{}, {rows, k, 1});
    view.source_strides[expert] = cutlass::make_cute_packed_stride(StrideC{}, {n, rows, 1});
    view.output_strides[expert] = cutlass::make_cute_packed_stride(StrideD{}, {n, rows, 1});
    return;
  }
  view.problem_shapes[expert] = UnderlyingProblemShape(rows, n, k);
  view.activation_ptrs[expert] =
      rows == 0 ? activation : activation + begin * static_cast<int64_t>(k);
  view.weight_ptrs[expert] = weights + static_cast<int64_t>(expert) * n * k;
  view.source_ptrs[expert] = rows == 0 ? output : output + begin * static_cast<int64_t>(n);
  view.output_ptrs[expert] = rows == 0 ? output : output + begin * static_cast<int64_t>(n);
  view.activation_strides[expert] = cutlass::make_cute_packed_stride(StrideA{}, {rows, k, 1});
  view.weight_strides[expert] = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  view.source_strides[expert] = cutlass::make_cute_packed_stride(StrideC{}, {rows, n, 1});
  view.output_strides[expert] = cutlass::make_cute_packed_stride(StrideD{}, {rows, n, 1});
}

__global__ void prepare_schedule_and_arguments_kernel(
    int64_t const* offsets, Bf16ScheduleHeader* header, Bf16ExpertSchedule* schedules,
    bool write_schedule, int num_experts, int64_t row_capacity, uint64_t epoch, int n, int k,
    Element const* activation, Element const* weights, Element* output, WorkspaceView m64_view,
    WorkspaceView m128_view) {
  int const expert = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (expert == 0) {
    if (write_schedule) {
      header->epoch = epoch;
      header->row_capacity = row_capacity;
      header->num_experts = num_experts;
      header->reserved = 0;
    } else if (epoch == 0 || header->epoch != epoch || header->num_experts != num_experts ||
               header->row_capacity != row_capacity) {
      trap_invalid_schedule(3);
    }
  }
  if (expert >= num_experts) return;

  Bf16ExpertSchedule schedule;
  if (write_schedule) {
    int64_t const begin = offsets[expert];
    int64_t const end = offsets[expert + 1];
    if (begin < 0 || (expert == 0 && begin != 0) || end < begin || end > row_capacity) {
      trap_invalid_schedule(1);
    }
    int64_t const rows = end - begin;
    if (rows > std::numeric_limits<int32_t>::max()) trap_invalid_schedule(2);
    schedule = Bf16ExpertSchedule{begin, static_cast<int32_t>(rows), 0};
    schedules[expert] = schedule;
  } else {
    schedule = schedules[expert];
  }
#if SM90_PUSH_BF16_FAMILY_MASK & 1
  prepare_family_arguments<M64Traits>(schedule, expert, n, k, activation, weights, output,
                                      m64_view);
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
  prepare_family_arguments<M128Traits>(schedule, expert, n, k, activation, weights, output,
                                       m128_view);
#endif
}

template <typename Traits>
inline typename Traits::Gemm::Arguments make_arguments(WorkspaceView const& workspace,
                                                       int num_experts, int device_id,
                                                       int sm_count) {
  cutlass::KernelHardwareInfo hardware_info;
  hardware_info.device_id = device_id;
  hardware_info.sm_count = sm_count;
  typename Traits::Gemm::EpilogueOutputOp::Params epilogue{ElementAccumulator(1.0f),
                                                           ElementAccumulator(0.0f)};
  return
      typename Traits::Gemm::Arguments{cutlass::gemm::GemmUniversalMode::kGrouped,
                                       {num_experts, workspace.problem_shapes, nullptr},
                                       {workspace.activation_ptrs, workspace.activation_strides,
                                        workspace.weight_ptrs, workspace.weight_strides},
                                       {epilogue, workspace.source_ptrs, workspace.source_strides,
                                        workspace.output_ptrs, workspace.output_strides},
                                       hardware_info};
}

struct KernelResources {
  int32_t blocks_per_sm;
  int32_t num_regs;
  int32_t local_memory_bytes;
  int32_t dynamic_smem_bytes;
};

template <typename Traits>
inline cudaError_t query_kernel_resources(KernelResources* resources) {
  using Kernel = typename Traits::GemmKernel;
  auto kernel = cutlass::device_kernel<Kernel>;
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) return status;
  int const dynamic_smem_bytes = static_cast<int>(sizeof(typename Kernel::SharedStorage));
  if (dynamic_smem_bytes >= 48 * 1024) {
    status = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                  dynamic_smem_bytes);
    if (status != cudaSuccess) return status;
  }
  int blocks_per_sm = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, kernel, Kernel::MaxThreadsPerBlock, dynamic_smem_bytes);
  if (status != cudaSuccess) return status;
  resources->blocks_per_sm = blocks_per_sm;
  resources->num_regs = attributes.numRegs;
  resources->local_memory_bytes = static_cast<int32_t>(attributes.localSizeBytes);
  resources->dynamic_smem_bytes = dynamic_smem_bytes;
  return cudaSuccess;
}

template <typename Traits>
inline cudaError_t launch_gemm_family(WorkspaceView const& view, int num_experts, int device_id,
                                      int sm_count, cudaStream_t stream) {
  using Gemm = typename Traits::Gemm;
  auto arguments = make_arguments<Traits>(view, num_experts, device_id, sm_count);
  Gemm gemm;
  cutlass::Status cutlass_status = gemm.can_implement(arguments);
  if (cutlass_status != cutlass::Status::kSuccess) return cudaErrorNotSupported;
  cutlass_status = gemm.initialize(arguments, view.cutlass_workspace, stream);
  if (cutlass_status != cutlass::Status::kSuccess) return cudaErrorInitializationError;
  cutlass_status = gemm.run(stream);
  if (cutlass_status != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
  return cudaSuccess;
}

inline cudaError_t launch_grouped_gemm(void* workspace, size_t workspace_bytes,
                                       void* schedule_workspace, size_t schedule_workspace_bytes,
                                       int num_experts, int max_rows, int n, int k,
                                       __nv_bfloat16 const* activation,
                                       __nv_bfloat16 const* weights, __nv_bfloat16* output,
                                       int64_t const* offsets, uint64_t epoch,
                                       bool prepare_schedule, size_t cutlass_workspace_bytes,
                                       int device_id, int sm_count, cudaStream_t stream) {
  auto const schedule_layout = make_schedule_workspace_layout(num_experts);
  if (schedule_workspace == nullptr || schedule_workspace_bytes < schedule_layout.total_bytes) {
    return cudaErrorInvalidValue;
  }
  auto* header = workspace_ptr<Bf16ScheduleHeader>(schedule_workspace, schedule_layout.header);
  auto* schedules = workspace_ptr<Bf16ExpertSchedule>(schedule_workspace, schedule_layout.experts);
  int constexpr kThreads = 128;
  int const blocks = (num_experts + kThreads - 1) / kThreads;
  WorkspaceView m64_view{};
  WorkspaceView m128_view{};
#if SM90_PUSH_BF16_FAMILY_MASK & 1
  m64_view = bind_workspace(workspace, workspace_bytes, num_experts, cutlass_workspace_bytes,
                            MTileFamily::kM64);
  if (m64_view.cutlass_workspace == nullptr) return cudaErrorInvalidValue;
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 2
  m128_view = bind_workspace(workspace, workspace_bytes, num_experts, cutlass_workspace_bytes,
                             MTileFamily::kM128);
  if (m128_view.cutlass_workspace == nullptr) return cudaErrorInvalidValue;
#endif
  prepare_schedule_and_arguments_kernel<<<blocks, kThreads, 0, stream>>>(
      offsets, header, schedules, prepare_schedule, num_experts, max_rows, epoch, n, k,
      reinterpret_cast<Element const*>(activation), reinterpret_cast<Element const*>(weights),
      reinterpret_cast<Element*>(output), m64_view, m128_view);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  if (max_rows == 0) return cudaSuccess;

#if SM90_PUSH_BF16_FAMILY_MASK & 2
  status = launch_gemm_family<M128Traits>(m128_view, num_experts, device_id, sm_count, stream);
  if (status != cudaSuccess) return status;
#endif
#if SM90_PUSH_BF16_FAMILY_MASK & 1
  status = launch_gemm_family<M64Traits>(m64_view, num_experts, device_id, sm_count, stream);
#endif
  return status;
}

}  // namespace flashinfer::sm90_push_bf16
