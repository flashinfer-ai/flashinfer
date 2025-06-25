/*
 * Copyright (C) 2025 Perplexity AI
 */
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <torch/library.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#define NVSHMEMCHECK(stmt)                                                                    \
  do {                                                                                        \
    int result = (stmt);                                                                      \
    if (NVSHMEMX_SUCCESS != result) {                                                         \
      fprintf(stderr, "[%s:%d] nvshmem failed with error %d \n", __FILE__, __LINE__, result); \
      exit(-1);                                                                               \
    }                                                                                         \
  } while (0)

namespace {

at::Tensor get_unique_id() {
  nvshmemx_uniqueid_t uid = NVSHMEMX_UNIQUEID_INITIALIZER;
  nvshmemx_get_uniqueid(&uid);
  return at::from_blob(&uid, sizeof(uid), at::kByte).clone();
}

int64_t unique_id_size() { return sizeof(nvshmemx_uniqueid_t); }

int64_t init(at::Tensor uid, int64_t rank, int64_t world_size) {
  TORCH_CHECK(uid.device().is_cpu(), "uid must be a CPU tensor");
  TORCH_CHECK(uid.scalar_type() == at::kByte, "uid must be a byte tensor");
  TORCH_CHECK(uid.numel() == sizeof(nvshmemx_uniqueid_t),
              "Invalid unique id size. Expected: ", sizeof(nvshmemx_uniqueid_t),
              ", Got: ", uid.numel(), ")");
  nvshmemx_uniqueid_t id;
  std::memcpy(&id, uid.data_ptr(), sizeof(id));
  nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
  nvshmemx_set_attr_uniqueid_args(rank, world_size, &id, &attr);
  return nvshmemx_init_attr(NVSHMEMX_INIT_WITH_UNIQUEID, &attr);
}

void finalize() { nvshmem_finalize(); }

int64_t my_pe() { return nvshmem_my_pe(); }

int64_t n_pes() { return nvshmem_n_pes(); }

at::Tensor malloc_tensor(const std::vector<int64_t>& shape, c10::ScalarType dtype,
                         const c10::Device& device) {
  size_t size = c10::elementSize(dtype) * c10::multiply_integers(shape);
  void* ptr = nvshmem_malloc(size);
  if (ptr == nullptr) {
    AT_ERROR("nvshmem_malloc failed. size: ", size);
  }
  return at::from_blob(
      ptr, shape, [](void* ptr) { nvshmem_free(ptr); },
      at::TensorOptions().dtype(dtype).device(device));
}

void barrier_all() { nvshmem_barrier_all(); }

void barrier_all_on_current_stream() {
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  nvshmemx_barrier_all_on_stream(stream);
}

void alltoall(at::Tensor dest, at::Tensor source) {
  TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");

  size_t nbytes = dest.numel() * dest.itemsize() / dest.size(0);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  NVSHMEMCHECK(nvshmemx_alltoallmem_on_stream(NVSHMEM_TEAM_WORLD, (uint8_t*)dest.data_ptr(),
                                              (uint8_t*)source.data_ptr(), nbytes, stream));
}

void fake_alltoall(at::Tensor dest, at::Tensor source) {}

void sum_reduce(at::Tensor dest, at::Tensor source, int64_t nelems) {
  TORCH_CHECK(dest.is_contiguous(), "dest must be contiguous");
  TORCH_CHECK(source.is_contiguous(), "source must be contiguous");
  TORCH_CHECK(dest.scalar_type() == source.scalar_type(),
              "dest and source must have the same dtype");

  // Add validation and conversion
  TORCH_CHECK(nelems >= 0, "nelems must be non-negative, got ", nelems);
  TORCH_CHECK(nelems <= SIZE_MAX, "nelems too large: ", nelems, " > ", SIZE_MAX);
  size_t nelems_size_t = static_cast<size_t>(nelems);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  switch (dest.scalar_type()) {
    case at::kHalf:  // float16
      NVSHMEMCHECK(nvshmemx_half_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, (__half*)dest.data_ptr(),
                                                      (__half*)source.data_ptr(), nelems_size_t,
                                                      stream));
      break;
    case at::kFloat:  // float32
      NVSHMEMCHECK(nvshmemx_float_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, (float*)dest.data_ptr(),
                                                       (float*)source.data_ptr(), nelems_size_t,
                                                       stream));
      break;
    case at::kBFloat16:  // bfloat16
      NVSHMEMCHECK(nvshmemx_bfloat16_sum_reduce_on_stream(
          NVSHMEM_TEAM_WORLD, (__nv_bfloat16*)dest.data_ptr(), (__nv_bfloat16*)source.data_ptr(),
          nelems_size_t, stream));
      break;

    default:
      TORCH_CHECK(false, "Unsupported dtype for nvshmem_sum_reduce: ", dest.scalar_type());
  }
}

void fake_sum_reduce(at::Tensor dest, at::Tensor source, int64_t nelems) {}

void allreduce_on_stream_with_copy(at::Tensor dest_symm, at::Tensor source_symm,
                                   at::Tensor dest_local, at::Tensor source_local, int64_t nelems) {
  TORCH_CHECK(dest_symm.is_contiguous(), "dest_symm must be contiguous");
  TORCH_CHECK(source_symm.is_contiguous(), "source_symm must be contiguous");
  TORCH_CHECK(dest_local.is_contiguous(), "dest_local must be contiguous");
  TORCH_CHECK(source_local.is_contiguous(), "source_local must be contiguous");
  TORCH_CHECK(dest_symm.scalar_type() == source_symm.scalar_type(),
              "dest_symm and source_symm must have the same dtype");
  TORCH_CHECK(dest_symm.scalar_type() == source_local.scalar_type(),
              "dest_symm and source_local must have the same dtype");
  TORCH_CHECK(dest_local.scalar_type() == source_local.scalar_type(),
              "dest_local and source_local must have the same dtype");

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  cudaMemcpyAsync(source_symm.data_ptr(), source_local.data_ptr(),
                  nelems * source_local.element_size(), cudaMemcpyDefault, stream);
  nvshmemx_barrier_on_stream(NVSHMEM_TEAM_WORLD, stream);
  sum_reduce(dest_symm, source_symm, nelems);
  cudaMemcpyAsync(dest_local.data_ptr(), dest_symm.data_ptr(), nelems * dest_local.element_size(),
                  cudaMemcpyDefault, stream);
  cudaStreamSynchronize(stream);
}

void fake_allreduce_on_stream_with_copy(at::Tensor dest_symm, at::Tensor source_symm,
                                        at::Tensor dest_local, at::Tensor source_local,
                                        int64_t nelems) {}

TORCH_LIBRARY_FRAGMENT(TORCH_EXTENSION_NAME, m) {
  m.def("nvshmem_get_unique_id", &get_unique_id);
  m.def("nvshmem_unique_id_size", &unique_id_size);
  m.def("nvshmem_init", &init);
  m.def("nvshmem_finalize", &finalize);
  m.def("nvshmem_my_pe", &my_pe);
  m.def("nvshmem_n_pes", &n_pes);
  m.def("nvshmem_malloc", &malloc_tensor);
  m.def("nvshmem_barrier_all", &barrier_all);
  m.def("nvshmem_barrier_all_on_current_stream", &barrier_all_on_current_stream);
  m.def("nvshmem_alltoall(Tensor! dest, Tensor src) -> ()");
  m.impl("nvshmem_alltoall", c10::kCUDA, &alltoall);
  m.impl("nvshmem_alltoall", c10::kMeta, &fake_alltoall);
  m.def("nvshmem_sum_reduce(Tensor! dest, Tensor src, int nelems) -> ()");
  m.impl("nvshmem_sum_reduce", c10::kCUDA, &sum_reduce);
  m.impl("nvshmem_sum_reduce", c10::kMeta, &fake_sum_reduce);
  m.def(
      "nvshmem_allreduce_on_stream_with_copy(Tensor! dest_symm, Tensor source_symm, Tensor "
      "dest_local, Tensor source_local, int nelems) -> ()");
  m.impl("nvshmem_allreduce_on_stream_with_copy", c10::kCUDA, &allreduce_on_stream_with_copy);
  m.impl("nvshmem_allreduce_on_stream_with_copy", c10::kMeta, &fake_allreduce_on_stream_with_copy);
};

}  // namespace
