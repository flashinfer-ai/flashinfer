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

template <class TileShape, class DispatchPolicy, class ActiveMask>
struct FwdRunner {
  using Element = cutlass::half_t;
  using ElementAccumulatorQK = float;
  using ElementAccumulatorPV = float;
  using ElementOut = cutlass::half_t;

  // Q K D ((H_R, H_KV), B)
  using ProblemShapeVarlen =
      cute::tuple<VariableLength, VariableLength, int, cute::tuple<cute::tuple<int, int>, int>>;

  using StrideQ = cute::tuple<int, _1, cute::tuple<int, int>>;  // Q D (H_G H_R B)
  using StrideK = cute::tuple<int, _1, cute::tuple<_0, int>>;   // K D (H_G H_R B)
  using StrideV = StrideK;
  using StrideO = cute::tuple<int, _1, cute::tuple<int, int>>;
  using StrideLSE = cute::tuple<_1, cute::tuple<int, int>>;  // Q (H_G H_R)

  using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
      Element, ElementAccumulatorQK, ElementAccumulatorPV, TileShape, StrideQ, StrideK, StrideV,
      ActiveMask>;
  using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
      ElementOut, ElementAccumulatorPV, typename Mainloop::TileShapePV>;
  using Operation =
      cutlass::fmha::device::FMHA<cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
          ProblemShapeVarlen, Mainloop, Epilogue, cutlass::fmha::kernel::PersistentTileScheduler>>;
  using LayoutQ = typename Mainloop::LayoutQ;
  using LayoutKV = typename Mainloop::LayoutKV;
  using LayoutO = typename Epilogue::LayoutO;
  using LayoutLSE = typename Epilogue::LayoutLSE;

  void run(at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor qo_lens, at::Tensor kv_lens,
           at::Tensor qo_segment_offsets, at::Tensor kv_segment_offsets, at::Tensor o,
           std::optional<at::Tensor> maybe_lse, int mask_mode_code, double sm_scale,
           int num_qo_heads, int num_kv_heads, int head_dim, int batch_size, int total_qo_len,
           int total_kv_len, int max_qo_len, int max_kv_len) {
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
        head_dim, cute::make_tuple(cute::make_tuple(h_r, num_kv_heads), batch_size));

    stride_Q = make_stride(num_qo_heads * head_dim, _1{}, make_stride(head_dim, h_r * head_dim));
    stride_O = stride_Q;
    stride_K = make_stride(num_kv_heads * head_dim, _1{}, make_stride(_0{}, head_dim));
    stride_V = stride_K;
    stride_LSE = make_stride(_1{}, make_stride(total_qo_len, total_qo_len * h_r));

    LayoutQ layout_Q =
        make_layout(make_shape(total_qo_len, head_dim, make_shape(h_r, num_kv_heads)), stride_Q);
    LayoutKV layout_K =
        make_layout(make_shape(total_kv_len, head_dim, make_shape(h_r, num_kv_heads)), stride_K);
    LayoutKV layout_V =
        make_layout(make_shape(total_kv_len, head_dim, make_shape(h_r, num_kv_heads)), stride_V);
    LayoutO layout_O =
        make_layout(make_shape(total_qo_len, head_dim, make_shape(h_r, num_kv_heads)), stride_O);
    LayoutLSE layout_LSE =
        make_layout(make_shape(total_qo_len, make_shape(h_r, num_kv_heads)), stride_LSE);

    typename Operation::Arguments arguments{
        problem_shape,
        {static_cast<Element*>(q.data_ptr()), layout_Q, static_cast<Element*>(k.data_ptr()),
         layout_K, static_cast<Element*>(v.data_ptr()), layout_V},
        {static_cast<ElementOut*>(o.data_ptr()), layout_O,
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

};  // namespace flashinfer
