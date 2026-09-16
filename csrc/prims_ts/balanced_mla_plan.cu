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

// PrimsTS binding for the graph-capturable device scheduler. This file validates caller-owned
// tensors and adapts them to the native scheduling parameters.

#include <algorithm>
#include <cstdint>
#include <limits>

#include "balanced_mla_scheduler.cuh"
#include "tvm_ffi_utils.h"

namespace {

using flashinfer::prims_ts::BalancedCombineDescriptor;
using flashinfer::prims_ts::BalancedSchedDeviceParams;
using flashinfer::prims_ts::BalancedSchedMetadata;
using flashinfer::prims_ts::BalancedWorkDescriptor;

void check_cuda_i32(tvm::ffi::TensorView tensor, char const* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_int32) << name << " must have dtype int32";
}

int32_t checked_nonnegative_i32(int64_t value, char const* name) {
  TVM_FFI_ICHECK_GE(value, 0) << name << " must be non-negative";
  TVM_FFI_ICHECK_LE(value, std::numeric_limits<int32_t>::max()) << name << " is too large";
  return static_cast<int32_t>(value);
}

void check_same_cuda_device(tvm::ffi::TensorView tensor, DLDevice const& expected,
                            char const* name) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, kDLCUDA) << name << " must be a CUDA tensor";
  TVM_FFI_ICHECK_EQ(tensor.device().device_id, expected.device_id)
      << name << " must be on the work-descriptor device";
}

}  // namespace

void BuildPrimsBalancedMLAPlanDevice(
    tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView work_descriptors,
    tvm::ffi::TensorView partition_offsets, tvm::ffi::TensorView combine_descriptors,
    tvm::ffi::TensorView num_combine_descriptors, tvm::ffi::TensorView plan_metadata,
    tvm::ffi::TensorView scheduler_workspace, tvm::ffi::TensorView cost_model_table,
    int64_t num_partitions_arg, int64_t k_tile_tokens_arg, int64_t max_seq_len_arg,
    bool select_cost_model_on_device, bool use_optimized_schedule,
    int64_t forced_target_piece_tiles) {
  check_cuda_i32(seq_lens, "seq_lens");
  check_cuda_i32(work_descriptors, "work_descriptors");
  check_cuda_i32(partition_offsets, "partition_offsets");
  check_cuda_i32(combine_descriptors, "combine_descriptors");
  check_cuda_i32(num_combine_descriptors, "num_combine_descriptors");
  check_cuda_i32(plan_metadata, "plan_metadata");
  check_cuda_i32(cost_model_table, "cost_model_table");
  TVM_FFI_ICHECK_EQ(scheduler_workspace.device().device_type, kDLCUDA)
      << "scheduler_workspace must be a CUDA tensor";
  TVM_FFI_ICHECK(scheduler_workspace.IsContiguous()) << "scheduler_workspace must be contiguous";
  TVM_FFI_ICHECK_EQ(scheduler_workspace.dtype(), dl_uint8)
      << "scheduler_workspace must have dtype uint8";

  TVM_FFI_ICHECK_EQ(seq_lens.ndim(), 1) << "seq_lens must be rank 1";
  TVM_FFI_ICHECK_EQ(work_descriptors.ndim(), 2) << "work_descriptors must be rank 2";
  TVM_FFI_ICHECK_EQ(work_descriptors.size(1), 4)
      << "work_descriptors must have packed shape [capacity, 4]";
  TVM_FFI_ICHECK_EQ(partition_offsets.ndim(), 1) << "partition_offsets must be rank 1";
  TVM_FFI_ICHECK_EQ(combine_descriptors.ndim(), 2) << "combine_descriptors must be rank 2";
  TVM_FFI_ICHECK_EQ(combine_descriptors.size(1), 4)
      << "combine_descriptors must have packed shape [capacity, 4]";
  TVM_FFI_ICHECK_EQ(num_combine_descriptors.ndim(), 1) << "num_combine_descriptors must be rank 1";
  TVM_FFI_ICHECK_GE(num_combine_descriptors.size(0), 1)
      << "num_combine_descriptors must hold one value";
  TVM_FFI_ICHECK_EQ(plan_metadata.ndim(), 1) << "plan_metadata must be rank 1";
  TVM_FFI_ICHECK_GE(plan_metadata.size(0), static_cast<int32_t>(BalancedSchedMetadata::kCount))
      << "device plan_metadata must hold five values";
  TVM_FFI_ICHECK_EQ(scheduler_workspace.ndim(), 1) << "scheduler_workspace must be rank 1";
  TVM_FFI_ICHECK_EQ(cost_model_table.ndim(), 2) << "cost_model_table must be rank 2";
  TVM_FFI_ICHECK_EQ(cost_model_table.size(0), flashinfer::prims_ts::kBalancedDeviceCostBucketCount)
      << "cost_model_table must contain seven workload buckets";
  TVM_FFI_ICHECK_EQ(cost_model_table.size(1),
                    flashinfer::prims_ts::kBalancedDeviceCostCoefficientCount)
      << "cost_model_table must contain six coefficients per bucket";

  TVM_FFI_ICHECK_GT(num_partitions_arg, 0) << "num_partitions must be positive";
  TVM_FFI_ICHECK_LE(num_partitions_arg, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_GT(k_tile_tokens_arg, 0) << "k_tile_tokens must be positive";
  TVM_FFI_ICHECK_LE(k_tile_tokens_arg, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_GE(max_seq_len_arg, 0) << "max_seq_len must be non-negative";
  TVM_FFI_ICHECK_LE(max_seq_len_arg, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_LE(seq_lens.size(0), std::numeric_limits<int32_t>::max() - num_partitions_arg)
      << "batch_size plus num_partitions exceeds int32 capacity";

  int32_t const batch_size = static_cast<int32_t>(seq_lens.size(0));
  int32_t const num_partitions = static_cast<int32_t>(num_partitions_arg);
  int32_t const descriptor_capacity = batch_size + num_partitions;
  TVM_FFI_ICHECK_GE(work_descriptors.size(0), descriptor_capacity)
      << "work_descriptors does not have the N + P capacity";
  TVM_FFI_ICHECK_EQ(partition_offsets.size(0), num_partitions + 1)
      << "partition_offsets has the wrong size";
  int32_t const reducer_capacity = std::min(batch_size, num_partitions);
  TVM_FFI_ICHECK_GE(combine_descriptors.size(0), reducer_capacity)
      << "combine_descriptors does not have min(batch_size, num_partitions) capacity";
  size_t const required_workspace =
      flashinfer::prims_ts::getBalancedSchedDeviceWorkspaceSize(batch_size, num_partitions);
  TVM_FFI_ICHECK_GE(static_cast<size_t>(scheduler_workspace.size(0)), required_workspace)
      << "scheduler_workspace is too small";

  DLDevice const output_device = work_descriptors.device();
  check_same_cuda_device(seq_lens, output_device, "seq_lens");
  check_same_cuda_device(partition_offsets, output_device, "partition_offsets");
  check_same_cuda_device(combine_descriptors, output_device, "combine_descriptors");
  check_same_cuda_device(num_combine_descriptors, output_device, "num_combine_descriptors");
  check_same_cuda_device(plan_metadata, output_device, "plan_metadata");
  check_same_cuda_device(scheduler_workspace, output_device, "scheduler_workspace");
  check_same_cuda_device(cost_model_table, output_device, "cost_model_table");

  BalancedSchedDeviceParams params{};
  params.batchSize = batch_size;
  params.blockSizeN = static_cast<int32_t>(k_tile_tokens_arg);
  params.numSmParts = num_partitions;
  params.maxSeqLen = static_cast<int32_t>(max_seq_len_arg);
  params.seqLensKvPtr = static_cast<int32_t const*>(seq_lens.data_ptr());
  params.workDescriptorCapacity = static_cast<int32_t>(work_descriptors.size(0));
  params.combineDescriptorCapacity = static_cast<int32_t>(combine_descriptors.size(0));
  params.workDescriptorPtr = reinterpret_cast<BalancedWorkDescriptor*>(work_descriptors.data_ptr());
  params.workDescriptorOffsetsPtr = static_cast<int32_t*>(partition_offsets.data_ptr());
  params.combineDescriptorPtr =
      reinterpret_cast<BalancedCombineDescriptor*>(combine_descriptors.data_ptr());
  params.numCombineDescriptorsDevicePtr = static_cast<int32_t*>(num_combine_descriptors.data_ptr());
  params.planMetadataDevicePtr = static_cast<int32_t*>(plan_metadata.data_ptr());
  params.costModelTablePtr = static_cast<int32_t const*>(cost_model_table.data_ptr());
  params.selectCostModelOnDevice = select_cost_model_on_device;
  params.useOptimizedSchedule = use_optimized_schedule;
  params.workspacePtr = scheduler_workspace.data_ptr();
  params.workspaceBytes = static_cast<size_t>(scheduler_workspace.size(0));
  params.stream = get_stream(output_device);
  params.forcedTargetPieceTiles =
      checked_nonnegative_i32(forced_target_piece_tiles, "forced_target_piece_tiles");

  flashinfer::prims_ts::runBalancedSchedDevice(params);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_prims_balanced_mla_plan_device,
                              BuildPrimsBalancedMLAPlanDevice);
