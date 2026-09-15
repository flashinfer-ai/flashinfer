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

// PrimsTS binding for its balanced host scheduler. The scheduling algorithm and packed descriptor
// emitter live in prims_balanced_mla_scheduler.cu; this file validates caller-owned tensors and
// adapts the Python-selected PrimsTS cost model to the native parameters.

#include <cstdint>
#include <limits>

#include "prims_balanced_mla_scheduler.cuh"
#include "tvm_ffi_utils.h"

namespace {

using flashinfer::prims_ts::BalancedCombineDescriptor;
using flashinfer::prims_ts::BalancedSchedParams;
using flashinfer::prims_ts::BalancedWorkDescriptor;

void check_i32(tvm::ffi::TensorView tensor, char const* name, DLDeviceType device_type) {
  TVM_FFI_ICHECK_EQ(tensor.device().device_type, device_type)
      << name << (device_type == kDLCPU ? " must be a host tensor" : " must be a CUDA tensor");
  TVM_FFI_ICHECK(tensor.IsContiguous()) << name << " must be contiguous";
  TVM_FFI_ICHECK_EQ(tensor.dtype(), dl_int32) << name << " must have dtype int32";
}

int32_t checked_nonnegative_i32(int64_t value, char const* name) {
  TVM_FFI_ICHECK_GE(value, 0) << name << " must be non-negative";
  TVM_FFI_ICHECK_LE(value, std::numeric_limits<int32_t>::max()) << name << " is too large";
  return static_cast<int32_t>(value);
}

}  // namespace

void BuildPrimsBalancedMLAPlan(tvm::ffi::TensorView seq_lens, tvm::ffi::TensorView work_descriptors,
                               tvm::ffi::TensorView partition_offsets,
                               tvm::ffi::TensorView combine_descriptors,
                               tvm::ffi::TensorView num_combine_descriptors,
                               tvm::ffi::TensorView plan_metadata, int64_t num_partitions_arg,
                               int64_t k_tile_tokens_arg, int64_t cost_per_k_tile,
                               int64_t fixed_piece_cost, int64_t split_fixed_cost,
                               int64_t split_piece_cost, int64_t reducer_fixed_cost,
                               int64_t reducer_piece_cost, int64_t forced_target_piece_tiles) {
  check_i32(seq_lens, "seq_lens", kDLCPU);
  check_i32(work_descriptors, "work_descriptors", kDLCUDA);
  check_i32(partition_offsets, "partition_offsets", kDLCUDA);
  check_i32(combine_descriptors, "combine_descriptors", kDLCUDA);
  check_i32(num_combine_descriptors, "num_combine_descriptors", kDLCUDA);
  check_i32(plan_metadata, "plan_metadata", kDLCPU);

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
  TVM_FFI_ICHECK_GE(plan_metadata.size(0), 3) << "plan_metadata must hold three values";

  TVM_FFI_ICHECK_GT(num_partitions_arg, 0) << "num_partitions must be positive";
  TVM_FFI_ICHECK_LE(num_partitions_arg, std::numeric_limits<int32_t>::max());
  TVM_FFI_ICHECK_GT(k_tile_tokens_arg, 0) << "k_tile_tokens must be positive";
  TVM_FFI_ICHECK_LE(k_tile_tokens_arg, std::numeric_limits<int32_t>::max());
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

  DLDevice const output_device = work_descriptors.device();
  auto check_same_device = [&](tvm::ffi::TensorView tensor, char const* name) {
    TVM_FFI_ICHECK_EQ(tensor.device().device_type, output_device.device_type)
        << name << " must be on the work-descriptor device";
    TVM_FFI_ICHECK_EQ(tensor.device().device_id, output_device.device_id)
        << name << " must be on the work-descriptor device";
  };
  check_same_device(partition_offsets, "partition_offsets");
  check_same_device(combine_descriptors, "combine_descriptors");
  check_same_device(num_combine_descriptors, "num_combine_descriptors");

  int32_t const* seq_ptr = static_cast<int32_t const*>(seq_lens.data_ptr());
  for (int32_t request_idx = 0; request_idx < batch_size; ++request_idx) {
    TVM_FFI_ICHECK_GE(seq_ptr[request_idx], 0) << "seq_lens must be non-negative";
  }

  BalancedSchedParams params{};
  params.batchSize = batch_size;
  params.blockSizeN = static_cast<int32_t>(k_tile_tokens_arg);
  params.numSmParts = num_partitions;
  params.seqLensKvPtr = seq_ptr;
  params.seqLensOnHost = true;
  params.workDescriptorCapacity = static_cast<int32_t>(work_descriptors.size(0));
  params.combineDescriptorCapacity = static_cast<int32_t>(combine_descriptors.size(0));
  params.workDescriptorPtr = reinterpret_cast<BalancedWorkDescriptor*>(work_descriptors.data_ptr());
  params.workDescriptorOffsetsPtr = static_cast<int32_t*>(partition_offsets.data_ptr());
  params.combineDescriptorPtr =
      reinterpret_cast<BalancedCombineDescriptor*>(combine_descriptors.data_ptr());
  params.numCombineDescriptorsDevicePtr = static_cast<int32_t*>(num_combine_descriptors.data_ptr());
  params.stream = get_stream(output_device);
  params.costPerBlock = checked_nonnegative_i32(cost_per_k_tile, "cost_per_k_tile");
  params.fixedPieceCost = checked_nonnegative_i32(fixed_piece_cost, "fixed_piece_cost");
  params.splitFixedCost = checked_nonnegative_i32(split_fixed_cost, "split_fixed_cost");
  params.splitPieceCost = checked_nonnegative_i32(split_piece_cost, "split_piece_cost");
  params.combineFixedCost = checked_nonnegative_i32(reducer_fixed_cost, "reducer_fixed_cost");
  params.combinePieceCost = checked_nonnegative_i32(reducer_piece_cost, "reducer_piece_cost");
  params.forcedTargetPieceTiles =
      checked_nonnegative_i32(forced_target_piece_tiles, "forced_target_piece_tiles");
  params.planMetadataHostPtr = static_cast<int32_t*>(plan_metadata.data_ptr());

  flashinfer::prims_ts::runBalancedSchedHost(params, params.numCombineDescriptorsDevicePtr,
                                             descriptor_capacity);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(build_prims_balanced_mla_plan, BuildPrimsBalancedMLAPlan);
