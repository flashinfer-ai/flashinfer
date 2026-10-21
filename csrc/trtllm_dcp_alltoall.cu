/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <tvm/ffi/container/tuple.h>

#include <cstdlib>
#include <string>
#include <vector>

#include "tensorrt_llm/kernels/helixAllToAll.h"
#include "tvm_ffi_utils.h"

#ifdef FLASHINFER_DCP_GENERATED
#include "cake_dcp_alltoall_dispatch.cuh"
#endif

using tvm::ffi::Tensor;
using tvm::ffi::TensorView;

namespace {

static int getEnvChannelCount() {
  static int cached = -1;
  if (cached < 0) {
    const char* env = std::getenv("DCP_A2A_CHANNEL_COUNT");
    cached = (env && std::string(env) != "0") ? std::atoi(env) : 0;
  }
  return cached;
}

int64_t getDcpWorkspaceSizePerRank(int64_t cp_size) {
  return static_cast<int64_t>(
      tensorrt_llm::kernels::computeHelixWorkspaceSizePerRank(static_cast<int>(cp_size)));
}

void initializeDcpWorkspaceOp(TensorView workspace, int64_t cp_rank, int64_t cp_size) {
  CHECK_INPUT_TYPE(workspace, dl_int64);
  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2) << "workspace must be 2D";
  TVM_FFI_ICHECK_EQ(workspace.size(0), cp_size) << "workspace first dim must equal cp_size";
  TVM_FFI_ICHECK(cp_rank >= 0 && cp_rank < cp_size) << "cp_rank must be in [0, cp_size)";

  auto stream = get_current_stream();
  auto* global_ptr = reinterpret_cast<uint64_t*>(workspace.data_ptr());
  auto* local_ptr = global_ptr + cp_rank * workspace.stride(0);

  tensorrt_llm::kernels::initializeHelixWorkspace(local_ptr, static_cast<int>(cp_size), stream);
}

void checkDcpInputs(TensorView partial_o, TensorView softmax_stats, TensorView workspace,
                    int64_t cp_size) {
  CHECK_INPUT(partial_o);
  CHECK_INPUT(softmax_stats);
  CHECK_CUDA(workspace);

  auto po_dtype_code = encode_dlpack_dtype(partial_o.dtype());
  TVM_FFI_ICHECK(po_dtype_code == float16_code || po_dtype_code == bfloat16_code)
      << "partial_o must be half or bfloat16";
  CHECK_INPUT_TYPE(softmax_stats, dl_float32);
  CHECK_INPUT_TYPE(workspace, dl_int64);

  TVM_FFI_ICHECK(partial_o.ndim() >= 2) << "partial_o must have at least 2 dimensions";
  TVM_FFI_ICHECK(softmax_stats.ndim() >= 2) << "softmax_stats must have at least 2 dimensions";
  TVM_FFI_ICHECK_EQ(partial_o.ndim(), softmax_stats.ndim())
      << "partial_o and softmax_stats must have same number of dimensions";

  int64_t kv_lora_rank = partial_o.size(-1);
  TVM_FFI_ICHECK_EQ(partial_o.size(-2), cp_size)
      << "partial_o second-to-last dim must equal cp_size";
  TVM_FFI_ICHECK_EQ(softmax_stats.size(-2), cp_size)
      << "softmax_stats second-to-last dim must equal cp_size";
  TVM_FFI_ICHECK(softmax_stats.size(-1) % 2 == 0 && softmax_stats.size(-1) >= 2)
      << "softmax_stats last dim must be even and >= 2";

  for (int i = 0; i < partial_o.ndim() - 2; i++) {
    TVM_FFI_ICHECK_EQ(partial_o.size(i), softmax_stats.size(i)) << "batch dimensions must match";
  }

  int64_t po_elem_size = get_element_size(partial_o);
  TVM_FFI_ICHECK(kv_lora_rank * po_elem_size % 16 == 0)
      << "partial_o last dim must be 16-byte aligned";

  TVM_FFI_ICHECK_EQ(workspace.ndim(), 2) << "workspace must be 2D";
  TVM_FFI_ICHECK_EQ(workspace.size(0), cp_size) << "workspace first dim must equal cp_size";
}

void checkDcpOutput(TensorView input, TensorView output, char const* name) {
  CHECK_INPUT(output);
  TVM_FFI_ICHECK(output.dtype() == input.dtype()) << name << " dtype must match its input";
  TVM_FFI_ICHECK_EQ(output.ndim(), input.ndim()) << name << " rank must match its input";
  for (int i = 0; i < input.ndim(); i++) {
    TVM_FFI_ICHECK_EQ(output.size(i), input.size(i)) << name << " shape must match its input";
  }
  TVM_FFI_ICHECK(output.device().device_id == input.device().device_id)
      << name << " must live on the input device";
}

// Exchanges ``partial_o`` / ``softmax_stats`` into the caller-provided,
// already validated output tensors.
void runDcpAllToAll(TensorView partial_o, TensorView softmax_stats, TensorView partial_o_out,
                    TensorView softmax_stats_out, TensorView workspace, int64_t cp_rank,
                    int64_t cp_size, bool enable_pdl) {
  int64_t kv_lora_rank = partial_o.size(-1);
  int64_t po_elem_size = get_element_size(partial_o);
  bool allowVariableField1 = softmax_stats.size(-1) > 2;

  int64_t entry_count = 1;
  for (int i = 0; i < partial_o.ndim() - 2; i++) {
    entry_count *= partial_o.size(i);
  }

  int64_t ss_last = softmax_stats.size(-1);
  int64_t ss_elem_size = get_element_size(softmax_stats);

  // stride(1) of a contiguous [entry_count, cp_size, D] view = D
  // HelixFieldInfo.stride is in bytes: D * elem_size
  tensorrt_llm::kernels::HelixAllToAllParams params;

  params.sendFields[0].dataPtr = static_cast<uint8_t*>(partial_o.data_ptr());
  params.sendFields[0].elementCount = static_cast<int>(kv_lora_rank);
  params.sendFields[0].elementSize = static_cast<int>(po_elem_size);
  params.sendFields[0].stride = static_cast<int>(kv_lora_rank * po_elem_size);

  params.recvFields[0].dataPtr = static_cast<uint8_t*>(partial_o_out.data_ptr());
  params.recvFields[0].elementCount = static_cast<int>(kv_lora_rank);
  params.recvFields[0].elementSize = static_cast<int>(po_elem_size);
  params.recvFields[0].stride = static_cast<int>(kv_lora_rank * po_elem_size);

  params.sendFields[1].dataPtr = static_cast<uint8_t*>(softmax_stats.data_ptr());
  params.sendFields[1].elementCount = static_cast<int>(ss_last);
  params.sendFields[1].elementSize = static_cast<int>(ss_elem_size);
  params.sendFields[1].stride = static_cast<int>(ss_last * ss_elem_size);

  params.recvFields[1].dataPtr = static_cast<uint8_t*>(softmax_stats_out.data_ptr());
  params.recvFields[1].elementCount = static_cast<int>(ss_last);
  params.recvFields[1].elementSize = static_cast<int>(ss_elem_size);
  params.recvFields[1].stride = static_cast<int>(ss_last * ss_elem_size);

  params.entryCount = static_cast<int>(entry_count);
  params.workspace = reinterpret_cast<uint64_t*>(workspace.data_ptr());
  params.workspaceStrideInU64 = workspace.stride(0);

  params.cpRank = static_cast<int>(cp_rank);
  params.cpSize = static_cast<int>(cp_size);
  params.channelCount = getEnvChannelCount();
  params.maxChannelCount =
      tensorrt_llm::kernels::computeHelixMaxChannelCount(static_cast<int>(cp_size));

  auto stream = get_current_stream();
#ifdef FLASHINFER_DCP_GENERATED
  if (flashinfer::comm::dcp::LaunchGeneratedDcpAllToAll(params, allowVariableField1, enable_pdl,
                                                        stream)) {
    return;
  }
#endif
  tensorrt_llm::kernels::launchHelixAllToAll(params, allowVariableField1, enable_pdl, stream);
}

tvm::ffi::Tuple<Tensor, Tensor> alltoallDcpNativeOp(TensorView partial_o, TensorView softmax_stats,
                                                    TensorView workspace, int64_t cp_rank,
                                                    int64_t cp_size, bool enable_pdl) {
  checkDcpInputs(partial_o, softmax_stats, workspace, cp_size);

  // Build output shapes matching inputs
  std::vector<int64_t> po_shape(partial_o.ndim());
  for (int i = 0; i < partial_o.ndim(); i++) {
    po_shape[i] = partial_o.size(i);
  }
  std::vector<int64_t> ss_shape(softmax_stats.ndim());
  for (int i = 0; i < softmax_stats.ndim(); i++) {
    ss_shape[i] = softmax_stats.size(i);
  }

  Tensor partial_o_out =
      alloc_tensor(tvm::ffi::Shape(po_shape), partial_o.dtype(), partial_o.device());
  Tensor softmax_stats_out =
      alloc_tensor(tvm::ffi::Shape(ss_shape), softmax_stats.dtype(), softmax_stats.device());

  runDcpAllToAll(partial_o, softmax_stats, partial_o_out, softmax_stats_out, workspace, cp_rank,
                 cp_size, enable_pdl);

  return tvm::ffi::Tuple(partial_o_out, softmax_stats_out);
}

// Same exchange into preallocated outputs, so callers that capture CUDA graphs
// or reuse output buffers do not pay a per-call allocation.
void alltoallDcpNativeIntoOp(TensorView partial_o, TensorView softmax_stats,
                             TensorView partial_o_out, TensorView softmax_stats_out,
                             TensorView workspace, int64_t cp_rank, int64_t cp_size,
                             bool enable_pdl) {
  checkDcpInputs(partial_o, softmax_stats, workspace, cp_size);
  checkDcpOutput(partial_o, partial_o_out, "partial_o_out");
  checkDcpOutput(softmax_stats, softmax_stats_out, "softmax_stats_out");
  runDcpAllToAll(partial_o, softmax_stats, partial_o_out, softmax_stats_out, workspace, cp_rank,
                 cp_size, enable_pdl);
}

}  // namespace

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_dcp_workspace_size_per_rank, getDcpWorkspaceSizePerRank);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(initialize_dcp_workspace, initializeDcpWorkspaceOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(alltoall_dcp_native, alltoallDcpNativeOp);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(alltoall_dcp_native_into, alltoallDcpNativeIntoOp);
