/*
 * Copyright (c) 2023 by FlashInfer team.
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
#include <cstdint>

#include "../../allocator.h"
#include "collective/fmha_fusion.hpp"
#include "collective/sm100_fmha_fwd_epilogue_tma_warpspecialized.hpp"
#include "collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp"
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "device/fmha.hpp"
#include "kernel/fmha_tile_scheduler.hpp"
#include "kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask>
struct FwdRunner {
  using Element = DTypeIn;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = DTypeOut;

  // Q K D ((H_R, H_KV), B)
  using ProblemShapeVarlen =
      cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;  // Q D (H_G H_R)
  using StrideK = cute::tuple<int, _1, cute::tuple<_0, int>>;   // K D (H_G H_R)
  using StrideV = cute::tuple<_1, int, cute::tuple<_0, int>>;   // D V (H_G H_R)
  // NOTE(Zihao): use markus's trick for tma store
  using StrideO =
      cute::tuple<int, _1, cute::tuple<cute::tuple<int, int>, int>>;  // Q D (H_G H_R) CUMULATIVE_Q
  using StrideLSE = cute::tuple<int, cute::tuple<_1, int>>;           // Q (H_G H_R)

  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeQK, TileShapePV, StrideQ,
      StrideK, StrideV, ActiveMask>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, typename Mainloop::TileShapePV>;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeVarlen, Mainloop, Epilogue,
          cutlass::fmha::kernel::HostPrecomputedTileScheduler>>;
  using LayoutQ = typename Mainloop::LayoutQ;
  using LayoutK = typename Mainloop::LayoutK;
  using LayoutV = typename Mainloop::LayoutV;
  using LayoutO = typename Epilogue::LayoutO;
  using LayoutLSE = typename Epilogue::LayoutLSE;

  static cudaError_t run(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         IdType* work_indptr, IdType* qo_tile_indices, IdType* qo_head_indices,
                         IdType* batch_indices, DTypeOut* o, float* maybe_lse, int mask_mode_code,
                         double sm_scale, double q_scale, double k_scale, double v_scale,
                         double o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, cudaStream_t stream) {
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    StrideQ stride_Q;
    StrideK stride_K;
    StrideV stride_V;
    StrideO stride_O;
    StrideLSE stride_LSE;

    int h_r = num_qo_heads / num_kv_heads;
    assert(num_qo_heads % num_kv_heads == 0);
    ProblemShapeVarlen problem_shape = cute::make_tuple(
        VariableLength{qo_segment_offsets}, VariableLength{kv_segment_offsets}, head_dim_qk,
        cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size));

    stride_Q = make_stride(q_stride_n, _1{}, make_stride(q_stride_h, h_r * q_stride_h));
    stride_O = make_stride(
        num_qo_heads * head_dim_vo, _1{},
        make_stride(make_stride(head_dim_vo, h_r * head_dim_vo), num_qo_heads * head_dim_vo));
    stride_K = make_stride(k_stride_n, _1{}, make_stride(_0{}, k_stride_h));
    stride_V = make_stride(_1{}, v_stride_n, make_stride(_0{}, v_stride_h));
    stride_LSE = make_stride(num_qo_heads, make_stride(_1{}, h_r));

    auto shape_Q = make_shape(total_qo_len, head_dim_qk, make_shape(h_r, num_kv_heads));
    auto shape_O = make_shape(max_qo_len, head_dim_vo,
                              make_shape(make_shape(h_r, num_kv_heads), max_qo_len + total_qo_len));
    auto shape_K = make_shape(total_kv_len, head_dim_qk, make_shape(h_r, num_kv_heads));
    auto shape_V = make_shape(head_dim_vo, total_kv_len, make_shape(h_r, num_kv_heads));
    auto shape_LSE = make_shape(total_qo_len, make_shape(h_r, num_kv_heads));

    LayoutQ layout_Q = make_layout(shape_Q, stride_Q);
    LayoutK layout_K = make_layout(shape_K, stride_K);
    LayoutV layout_V = make_layout(shape_V, stride_V);
    LayoutO layout_O = make_layout(shape_O, stride_O);
    LayoutLSE layout_LSE = make_layout(shape_LSE, stride_LSE);

    typename Operation::Arguments arguments{
        problem_shape,
        {q, layout_Q, k, layout_K, v, layout_V, sm_scale, q_scale, k_scale, v_scale, o_scale},
        {o - max_qo_len * get<0>(stride_O), layout_O, maybe_lse, layout_LSE, max_qo_len},
        {work_indptr, qo_tile_indices, qo_head_indices, batch_indices},
        hw_info};

    Operation op;

    // NOTE(Zihao): workspace size is not used at this moment
    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    AlignedAllocator allocator(workspace_buffer, workspace_size);
    uint8_t* workspace_ptr =
        allocator.aligned_alloc<uint8_t>(workspace_size, 16, "fmha_cutlass_sm100_workspace");

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    status = op.initialize(arguments, workspace_ptr);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    // Run
    status = op.run(stream);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }
    return cudaSuccess;
  }
};

template <typename DTypeIn, typename DTypeOut, typename IdType, class TileShapeQK,
          class TileShapePV, class ActiveMask>
cudaError_t run_fmha_fwd(void* workspace_buffer, DTypeIn* q, DTypeIn* k, DTypeIn* v,
                         IdType* qo_segment_offsets, IdType* kv_segment_offsets,
                         IdType* work_indptr, IdType* qo_tile_indices, IdType* qo_head_indices,
                         IdType* batch_indices, DTypeOut* o, float* maybe_lse, int mask_mode_code,
                         double sm_scale, double q_scale, double k_scale, double v_scale,
                         double o_scale, int num_qo_heads, int num_kv_heads, int head_dim_qk,
                         int head_dim_vo, int q_stride_n, int q_stride_h, int k_stride_n,
                         int k_stride_h, int v_stride_n, int v_stride_h, int batch_size,
                         int total_qo_len, int total_kv_len, int max_qo_len, cudaStream_t stream) {
  return FwdRunner<DTypeIn, DTypeOut, IdType, TileShapeQK, TileShapePV, ActiveMask>::run(
      workspace_buffer, q, k, v, qo_segment_offsets, kv_segment_offsets, work_indptr,
      qo_tile_indices, qo_head_indices, batch_indices, o, maybe_lse, mask_mode_code, sm_scale,
      q_scale, k_scale, v_scale, o_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo,
      q_stride_n, q_stride_h, k_stride_n, k_stride_h, v_stride_n, v_stride_h, batch_size,
      total_qo_len, total_kv_len, max_qo_len, stream);
}

};  // namespace flashinfer
