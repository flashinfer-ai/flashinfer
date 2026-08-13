/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 */

#include "flashkda_binding_common.cuh"

#define uint8_t flashkda_k1_parallel_uint8_t
#define uint16_t flashkda_k1_parallel_uint16_t
#define uint32_t flashkda_k1_parallel_uint32_t
#define uint64_t flashkda_k1_parallel_uint64_t
#define int32_t flashkda_k1_parallel_int32_t
#define int16_t flashkda_k1_parallel_int16_t
#include "flashkda_bf16_fused_m128_k1_parallel.cu"
#undef uint8_t
#undef uint16_t
#undef uint32_t
#undef uint64_t
#undef int32_t
#undef int16_t

namespace flashinfer {
namespace flash_kda {

constexpr int64_t kK1ParallelPacketBytes = 31520;

void RunM128K1Parallel(TensorView q, TensorView k, TensorView v, TensorView g, TensorView beta,
                       TensorView beta_tma, TensorView A_log, TensorView dt_bias,
                       TensorView cu_seqlens, TensorView seq_order, TensorView initial_state,
                       TensorView out, TensorView final_state, TensorView descriptor_storage,
                       TensorView k1_workspace, int64_t prepare_descriptors, int64_t num_heads,
                       int64_t use_initial_state, int64_t store_final_state, int64_t cluster_size,
                       int64_t mailbox_depth, double scale, double lower_bound,
                       int64_t cuda_stream) {
  TVM_FFI_ICHECK(cuda_stream >= 0) << "cuda_stream must be a non-negative stream handle";
  TVM_FFI_ICHECK(q.device().device_type == kDLCUDA) << "q must be a CUDA tensor";
  const int32_t device_id = q.device().device_id;
  ffi::CUDADeviceGuard device_guard(device_id);
  CheckFlashKDATarget(device_id);

  int major = 0;
  int minor = 0;
  CheckCuda(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id),
            "cudaDeviceGetAttribute(major)");
  CheckCuda(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id),
            "cudaDeviceGetAttribute(minor)");
  TVM_FFI_ICHECK(major == 10 && minor == 0)
      << "K1-parallel FlashKDA is tuned only for B200/GB200 (CC 10.0)";

  const int64_t num_seqs =
      CheckCommonInputs(q, k, v, g, beta, beta_tma, A_log, dt_bias, cu_seqlens, seq_order,
                        initial_state, out, final_state, descriptor_storage, prepare_descriptors,
                        num_heads, use_initial_state, store_final_state, scale, lower_bound);
  CheckCudaTensor(k1_workspace, "k1_workspace", device_id);
  CheckDtype(k1_workspace, "k1_workspace", dl_uint8);
  for (const auto& named : {
           std::pair<const TensorView*, const char*>(&q, "q"),
           std::pair<const TensorView*, const char*>(&k, "k"),
           std::pair<const TensorView*, const char*>(&v, "v"),
           std::pair<const TensorView*, const char*>(&g, "g"),
           std::pair<const TensorView*, const char*>(&beta, "beta"),
           std::pair<const TensorView*, const char*>(&beta_tma, "beta_tma"),
           std::pair<const TensorView*, const char*>(&A_log, "A_log"),
           std::pair<const TensorView*, const char*>(&dt_bias, "dt_bias"),
           std::pair<const TensorView*, const char*>(&cu_seqlens, "cu_seqlens"),
           std::pair<const TensorView*, const char*>(&seq_order, "seq_order"),
           std::pair<const TensorView*, const char*>(&out, "out"),
           std::pair<const TensorView*, const char*>(&descriptor_storage, "descriptor_storage"),
       }) {
    CheckNoOverlap(k1_workspace, "k1_workspace", *named.first, named.second);
  }
  if (use_initial_state != 0) {
    CheckNoOverlap(k1_workspace, "k1_workspace", initial_state, "initial_state");
  }
  if (store_final_state != 0) {
    CheckNoOverlap(k1_workspace, "k1_workspace", final_state, "final_state");
  }
  TVM_FFI_ICHECK(cluster_size == 4 || cluster_size == 8) << "cluster_size must be C4 or C8";
  TVM_FFI_ICHECK(mailbox_depth > 0 && mailbox_depth <= std::numeric_limits<int32_t>::max())
      << "mailbox_depth must be in the positive int32 range";
  TVM_FFI_ICHECK(num_heads >= 8 && num_heads % 8 == 0)
      << "K1-parallel FlashKDA requires H >= 8 and H divisible by 8";

  const int64_t num_tasks = num_seqs * num_heads;
  const int64_t packet_count = num_tasks * mailbox_depth;
  TVM_FFI_ICHECK(packet_count > 0 &&
                 packet_count <= std::numeric_limits<int64_t>::max() / kK1ParallelPacketBytes)
      << "K1 mailbox packet count is out of range";
  const int64_t flag_offset =
      (packet_count * kK1ParallelPacketBytes + int64_t{255}) & ~int64_t{255};
  TVM_FFI_ICHECK(packet_count <= (std::numeric_limits<int64_t>::max() - flag_offset) /
                                     static_cast<int64_t>(sizeof(uint32_t)))
      << "K1 mailbox flag size is out of range";
  const int64_t required_bytes =
      flag_offset + packet_count * static_cast<int64_t>(sizeof(uint32_t));
  TVM_FFI_ICHECK(k1_workspace.numel() >= required_bytes)
      << "k1_workspace requires " << required_bytes << " bytes, got " << k1_workspace.numel();
  TVM_FFI_ICHECK(reinterpret_cast<uintptr_t>(k1_workspace.data_ptr()) % 256 == 0)
      << "k1_workspace must be 256-byte aligned";

  constexpr int32_t kSmemBytes = SMEM_TOTAL;
  CheckDynamicSmemCapacity(device_id, kSmemBytes);
  CheckCuda(cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemBytes),
            "cudaFuncSetAttribute(kernel_flashkda_bf16_fused_m128_k1_parallel)");

  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(cuda_stream));
  const TmaPointers tma = EncodeTmaPointers<128>(q, k, v, g, beta_tma, out, descriptor_storage,
                                                 prepare_descriptors, stream);
  PackBetaForTmaIfNeeded(beta, beta_tma, num_heads, stream);

  auto* workspace_bytes = static_cast<unsigned char*>(k1_workspace.data_ptr());
  auto* flags = reinterpret_cast<unsigned int*>(workspace_bytes + flag_offset);
  CheckCuda(cudaMemsetAsync(flags, 0, packet_count * sizeof(uint32_t), stream),
            "cudaMemsetAsync(K1 mailbox flags)");

  const bool global_pool = cluster_size < 0;
  const int64_t grid_x_i64 = global_pool ? -cluster_size : num_tasks * cluster_size;
  TVM_FFI_ICHECK(grid_x_i64 > 0 && grid_x_i64 <= std::numeric_limits<uint32_t>::max())
      << "K1-parallel FlashKDA grid.x is out of range: " << grid_x_i64;

  cudaLaunchAttribute attribute{};
  attribute.id = cudaLaunchAttributeClusterDimension;
  attribute.val.clusterDim = {global_pool ? 1u : static_cast<uint32_t>(cluster_size), 1u, 1u};
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(static_cast<uint32_t>(grid_x_i64), 1, 1);
  config.blockDim = dim3(THREADS, 1, 1);
  config.dynamicSmemBytes = kSmemBytes;
  config.stream = stream;
  config.attrs = global_pool ? nullptr : &attribute;
  config.numAttrs = global_pool ? 0 : 1;

  CheckCuda(
      cudaLaunchKernelEx(
          &config, kernel_flashkda_bf16_fused_m128, reinterpret_cast<__nv_bfloat16*>(q.data_ptr()),
          tma.q, reinterpret_cast<__nv_bfloat16*>(k.data_ptr()), tma.k,
          reinterpret_cast<__nv_bfloat16*>(v.data_ptr()), tma.v,
          reinterpret_cast<__nv_bfloat16*>(g.data_ptr()), tma.g,
          reinterpret_cast<__nv_bfloat16*>(beta.data_ptr()), tma.beta,
          reinterpret_cast<float*>(A_log.data_ptr()), reinterpret_cast<float*>(dt_bias.data_ptr()),
          reinterpret_cast<long long*>(cu_seqlens.data_ptr()),
          reinterpret_cast<int*>(seq_order.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(initial_state.data_ptr()),
          reinterpret_cast<__nv_bfloat16*>(out.data_ptr()), tma.out,
          reinterpret_cast<__nv_bfloat16*>(final_state.data_ptr()), workspace_bytes, flags,
          static_cast<int32_t>(mailbox_depth), static_cast<int32_t>(cluster_size),
          static_cast<int32_t>(num_tasks), static_cast<int32_t>(num_heads),
          static_cast<int32_t>(use_initial_state), static_cast<int32_t>(store_final_state),
          static_cast<float>(scale), static_cast<float>(lower_bound)),
      "kernel_flashkda_bf16_fused_m128_k1_parallel launch");
}

}  // namespace flash_kda
}  // namespace flashinfer

TVM_FFI_DLL_EXPORT_TYPED_FUNC(run, flashinfer::flash_kda::RunM128K1Parallel);
