/*
 * Copyright (C) 2025 Perplexity AI
 */
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "tvm_ffi_utils.h"

#define NVSHMEMCHECK(stmt)                                                                    \
  do {                                                                                        \
    int result = (stmt);                                                                      \
    if (NVSHMEMX_SUCCESS != result) {                                                         \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result); \
      exit(-1);                                                                               \
    }                                                                                         \
  } while (0)

namespace {

Tensor get_unique_id() {
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_get_uniqueid(&uid);
  auto tensor = alloc_tensor(ffi::Shape({sizeof(uid)}), dl_uint8, cpu);
  std::memcpy(tensor->data, &uid, sizeof(uid));
  return tensor;
}

int64_t unique_id_size() { return sizeof(nvshmemx_uniqueid_t); }

int64_t init(Tensor uid, int64_t rank, int64_t world_size) {
  TVM_FFI_ICHECK_EQ(uid->device.device_type, kDLCPU) << "uid must be a host tensor";
  CHECK_INPUT_TYPE(uid, dl_uint8);
  TVM_FFI_ICHECK_EQ(get_numel(uid), sizeof(nvshmemx_uniqueid_t))
      << "Invalid unique id size. Expected: " << sizeof(nvshmemx_uniqueid_t)
      << ", Got: " << get_numel(uid);
  nvshmemx_uniqueid_t id;
  std::memcpy(&id, uid->data, sizeof(id));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &id, &attr);
  return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void finalize() { nvshmem_finalize(); }

int64_t my_pe() { return nvshmem_my_pe(); }

int64_t n_pes() { return nvshmem_n_pes(); }

// Tensor malloc_tensor(ffi::Shape shape, DLDataType dtype,
//                          const c10::Device& device) {
//   size_t size = c10::elementSize(dtype) * c10::multiply_integers(shape);
//   void* ptr = nvshmem_malloc(size);
//   if (ptr == nullptr) {
//     TVM_FFI_ICHECK(false) << "nvshmem_malloc failed. size: " << size;
//   }
//   auto tensor = alloc_tensor(ffi::Shape(shape), dtype, device);
//   std::memcpy(tensor->data, ptr, size);
//   return tensor;
// }

void barrier_all() { nvshmem_barrier_all(); }

void barrier_all_on_current_stream() {
  cudaStream_t stream = get_stream(cpu);
  nvshmemx_barrier_all_on_stream(stream);
}

void alltoall(Tensor dest, Tensor source) {
  CHECK_CONTIGUOUS(dest);
  CHECK_CONTIGUOUS(source);
  TVM_FFI_ICHECK_EQ(dest->dtype, source->dtype) << "dest and source must have the same dtype";

  size_t nbytes = get_numel(dest) * get_element_size(dest) / dest->shape[0];
  cudaStream_t stream = get_stream(dest->device);
  NVSHMEMCHECK(nvshmemx_alltoallmem_on_stream(NVSHMEM_TEAM_WORLD, (uint8_t*)dest->data,
                                              (uint8_t*)source->data, nbytes, stream));
}

void fake_alltoall(Tensor dest, Tensor source) {}

void sum_reduce(Tensor dest, Tensor source, int64_t nelems) {
  CHECK_CONTIGUOUS(dest);
  CHECK_CONTIGUOUS(source);
  TVM_FFI_ICHECK_EQ(dest->dtype, source->dtype) << "dest and source must have the same dtype";

  // Add validation and conversion
  TVM_FFI_ICHECK_GE(nelems, 0) << "nelems must be non-negative, got " << nelems;
  TVM_FFI_ICHECK_LE(nelems, SIZE_MAX) << "nelems too large: " << nelems << " > " << SIZE_MAX;
  size_t nelems_size_t = static_cast<size_t>(nelems);

  cudaStream_t stream = get_stream(dest->device);

  switch (encode_dlpack_dtype(dest->dtype)) {
    case float16_code:  // float16
      NVSHMEMCHECK(nvshmemx_half_sum_reduce_on_stream(
          NVSHMEM_TEAM_WORLD, (__half*)dest->data, (__half*)source->data, nelems_size_t, stream));
      break;
    case float32_code:  // float32
      NVSHMEMCHECK(nvshmemx_float_sum_reduce_on_stream(
          NVSHMEM_TEAM_WORLD, (float*)dest->data, (float*)source->data, nelems_size_t, stream));
      break;
    case bfloat16_code:  // bfloat16
      NVSHMEMCHECK(nvshmemx_bfloat16_sum_reduce_on_stream(
          NVSHMEM_TEAM_WORLD, (__nv_bfloat16*)dest->data, (__nv_bfloat16*)source->data,
          nelems_size_t, stream));
      break;

    default:
      TVM_FFI_ICHECK(false) << "Unsupported dtype for nvshmem_sum_reduce: " << dest->dtype;
  }
}

void fake_sum_reduce(Tensor dest, Tensor source, int64_t nelems) {}

void allreduce_on_stream_with_copy(Tensor dest_symm, Tensor source_symm, Tensor dest_local,
                                   Tensor source_local, int64_t nelems) {
  CHECK_CONTIGUOUS(dest_symm);
  CHECK_CONTIGUOUS(source_symm);
  CHECK_CONTIGUOUS(dest_local);
  CHECK_CONTIGUOUS(source_local);
  TVM_FFI_ICHECK_EQ(dest_symm->dtype, source_symm->dtype)
      << "dest_symm and source_symm must have the same dtype";
  TVM_FFI_ICHECK_EQ(dest_symm->dtype, source_local->dtype)
      << "dest_symm and source_local must have the same dtype";
  TVM_FFI_ICHECK_EQ(dest_local->dtype, source_local->dtype)
      << "dest_local and source_local must have the same dtype";

  cudaStream_t stream = get_stream(source_symm->device);

  cudaMemcpyAsync(source_symm->data, source_local->data, nelems * get_element_size(source_local),
                  cudaMemcpyDefault, stream);
  nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
  sum_reduce(dest_symm, source_symm, nelems);
  cudaMemcpyAsync(dest_local->data, dest_symm->data, nelems * get_element_size(dest_local),
                  cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
}

void fake_allreduce_on_stream_with_copy(Tensor dest_symm, Tensor source_symm, Tensor dest_local,
                                        Tensor source_local, int64_t nelems) {}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_get_unique_id, get_unique_id);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_unique_id_size, unique_id_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_init, init);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_finalize, finalize);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_my_pe, my_pe);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_n_pes, n_pes);
// TVM_FFI_DLL_EXPORT_TYPED_FUNC(malloc_tensor, malloc_tensor);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_barrier_all, barrier_all);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_barrier_all_on_current_stream, barrier_all_on_current_stream);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_alltoall, alltoall);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_fake_alltoall, fake_alltoall);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_sum_reduce, sum_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_fake_sum_reduce, fake_sum_reduce);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_allreduce_on_stream_with_copy, allreduce_on_stream_with_copy);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(nvshmem_fake_allreduce_on_stream_with_copy,
                              fake_allreduce_on_stream_with_copy);

}  // namespace
