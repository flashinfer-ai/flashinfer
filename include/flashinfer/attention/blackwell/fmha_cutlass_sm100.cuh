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
#include "pytorch_extension_utils.h"

namespace flashinfer {

using namespace cute;
using namespace cutlass::fmha::collective;
using namespace cutlass::fmha::kernel;
using namespace cutlass::fmha::device;

template <typename DTypeIn, typename DTypeOut, class TileShapeQK, class TileShapePV,
          class ActiveMask>
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
  using StrideLSE = cute::tuple<_1, cute::tuple<int, int>>;           // Q (H_G H_R)

  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShapeQK, TileShapePV, StrideQ,
      StrideK, StrideV, ActiveMask>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, typename Mainloop::TileShapePV>;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeVarlen, Mainloop, Epilogue,
          typename std::conditional<std::is_same<ActiveMask, CausalMask>::value,
                                    cutlass::fmha::kernel::NaiveTileScheduler,
                                    cutlass::fmha::kernel::PersistentTileScheduler>::type>>;
  using LayoutQ = typename Mainloop::LayoutQ;
  using LayoutK = typename Mainloop::LayoutK;
  using LayoutV = typename Mainloop::LayoutV;
  using LayoutO = typename Epilogue::LayoutO;
  using LayoutLSE = typename Epilogue::LayoutLSE;

  static void run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_lens, at::Tensor kv_lens,
                  at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets, at::Tensor o,
                  std::optional<at::Tensor> maybe_lse, int mask_mode_code, double sm_scale,
                  int num_qo_heads, int num_kv_heads, int head_dim_qk, int head_dim_vo,
                  int batch_size, int total_qo_len, int total_kv_len, int max_qo_len,
                  int max_kv_len) {
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
        VariableLength{max_qo_len, static_cast<int*>(qo_segment_offsets.data_ptr()),
                       static_cast<int*>(qo_lens.data_ptr())},
        VariableLength{max_kv_len, static_cast<int*>(kv_segment_offsets.data_ptr()),
                       static_cast<int*>(kv_lens.data_ptr())},
        head_dim_qk, cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size));

    stride_Q =
        make_stride(num_qo_heads * head_dim_qk, _1{}, make_stride(head_dim_qk, h_r * head_dim_qk));
    stride_O = make_stride(
        num_qo_heads * head_dim_vo, _1{},
        make_stride(make_stride(head_dim_vo, h_r * head_dim_vo), num_qo_heads * head_dim_vo));
    stride_K = make_stride(num_kv_heads * head_dim_qk, _1{}, make_stride(_0{}, head_dim_qk));
    stride_V = make_stride(_1{}, num_kv_heads * head_dim_vo, make_stride(_0{}, head_dim_vo));
    stride_LSE = make_stride(_1{}, make_stride(total_qo_len, total_qo_len * h_r));

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
        {static_cast<Element*>(q.data_ptr()), layout_Q, static_cast<Element*>(k.data_ptr()),
         layout_K, static_cast<Element*>(v.data_ptr()), layout_V},
        {static_cast<ElementOut*>(o.data_ptr()) - max_qo_len * get<0>(stride_O), layout_O,
         static_cast<ElementAccumulatorPV*>(maybe_lse.value().data_ptr()), layout_LSE},
        hw_info};

    Operation op;

    size_t workspace_size = 0;
    workspace_size = Operation::get_workspace_size(arguments);
    cutlass::DeviceAllocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = cutlass::Status::kSuccess;
    status = op.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "This kernel is not supported. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    status = op.initialize(arguments, workspace.get());
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to initialize the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    // Run
    status = op.run();
    if (status != cutlass::Status::kSuccess) {
      std::cerr << "Failed to launch the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(cudaGetLastError()) << std::endl;
    }

    cudaError_t result = cudaDeviceSynchronize();
    if (result != cudaSuccess) {
      std::cerr << "Error running the CUTLASS kernel. Last CUDA error is: "
                << cudaGetErrorString(result) << std::endl;
    }
  }
};

template <typename DTypeIn, typename DTypeOut, class TileShapeQK, class TileShapePV,
          class ActiveMask>
void run_fmha_fwd(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_lens, at::Tensor kv_lens,
                  at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets, at::Tensor o,
                  std::optional<at::Tensor> maybe_lse, int mask_mode_code, double sm_scale,
                  int num_qo_heads, int num_kv_heads, int head_dim_qk, int head_dim_vo,
                  int batch_size, int total_qo_len, int total_kv_len, int max_qo_len,
                  int max_kv_len) {
  FwdRunner<DTypeIn, DTypeOut, TileShapeQK, TileShapePV, ActiveMask>::run(
      q, k, v, qo_lens, kv_lens, qo_segment_offsets, kv_segment_offsets, o, maybe_lse,
      mask_mode_code, sm_scale, num_qo_heads, num_kv_heads, head_dim_qk, head_dim_vo, batch_size,
      total_qo_len, total_kv_len, max_qo_len, max_kv_len);
}

};  // namespace flashinfer
