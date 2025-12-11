// flashinfer: adapted from sglang + vllm code
// refer to: https://github.com/vllm-project/vllm/blob/v0.8.2/csrc/custom_all_reduce.cu
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tuple.h>

#include <cstdint>
#include <vector>

#include "flashinfer/comm/vllm_custom_all_reduce.cuh"
#include "tvm_ffi_utils.h"

// Fake pointer type, must match fptr_t type in ops.h.
// We use this type alias to indicate when pointers are passed in as int64_t.
using fptr_t = int64_t;
static_assert(sizeof(void*) == sizeof(fptr_t));

using tvm::ffi::Array;
using tvm::ffi::Tuple;

fptr_t init_custom_ar(Array<fptr_t> fake_ipc_ptrs, TensorView rank_data, int64_t rank,
                      bool full_nvlink) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > 8) throw std::invalid_argument("world size > 8 is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported for now");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank passed in");

  vllm::Signal* ipc_ptrs[8];
  for (int i = 0; i < world_size; i++) {
    ipc_ptrs[i] = reinterpret_cast<vllm::Signal*>(fake_ipc_ptrs[i]);
  }
  return (fptr_t) new vllm::CustomAllreduce(ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank,
                                            world_size, full_nvlink);
}

/**
 * Make sure tensor t's data lies completely within ((char)t.data_ptr()) +
 * t.numel() * t.element_size(). This is slightly weaker than t.is_contiguous()
 * because it allows transpose of contiguous slice (i.e. slicing the first
 * dimension). Currently, we require this because stride information is not
 * passed into the kernels and we treat input tensors as flat.
 *
 * Examples
 * A = torch.zeros(3, 3, 3)
 * 1. A: OK
 * 2. A[1:]: OK
 * 3. A.permute(2, 0, 1): OK
 * 4. A[1:].permute(2, 0, 1): OK
 * 5. A[None].expand(2, -1, -1, -1): Not OK
 * 6. A[:, 1:, 1:]: Not OK
 */
bool _is_weak_contiguous(TensorView t) {
  auto numel = t.numel();
  auto element_size = get_element_size(t);
  return t.IsContiguous() ||
         (tvm::ffi::GetDataSize(numel, t.dtype()) - t.byte_offset() * element_size ==
          numel * element_size);
}

/**
 * Performs an out-of-place allreduce and stores result in out.
 *
 * If _reg_buffer is null, assumes inp.data_ptr() is already IPC-registered.
 * Otherwise, _reg_buffer is assumed to be IPC-registered and inp is first
 * copied into _reg_buffer.
 */
void all_reduce(fptr_t _fa, TensorView inp, TensorView out, fptr_t _reg_buffer,
                int64_t reg_buffer_sz_bytes, int64_t num_ctas) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  ffi::CUDADeviceGuard device_guard(inp.device().device_id);
  auto stream = get_stream(inp.device());

  TVM_FFI_ICHECK_EQ(inp.dtype(), out.dtype());
  TVM_FFI_ICHECK_EQ(inp.numel(), out.numel());
  TVM_FFI_ICHECK(_is_weak_contiguous(out));
  TVM_FFI_ICHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * get_element_size(inp);
  auto reg_buffer = reinterpret_cast<void*>(_reg_buffer);
  if (reg_buffer) {
    TVM_FFI_ICHECK_LE(input_size, reg_buffer_sz_bytes);
    auto status =
        cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream);
    TVM_FFI_ICHECK(status == cudaSuccess);
  } else {
    reg_buffer = inp.data_ptr();
  }
  switch (encode_dlpack_dtype(out.dtype())) {
    case float32_code: {
      fa->allreduce<float>(stream, reinterpret_cast<float*>(reg_buffer),
                           reinterpret_cast<float*>(out.data_ptr()), out.numel(), num_ctas);
      break;
    }
    case float16_code: {
      fa->allreduce<half>(stream, reinterpret_cast<half*>(reg_buffer),
                          reinterpret_cast<half*>(out.data_ptr()), out.numel(), num_ctas);
      break;
    }
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case bfloat16_code: {
      fa->allreduce<nv_bfloat16>(stream, reinterpret_cast<nv_bfloat16*>(reg_buffer),
                                 reinterpret_cast<nv_bfloat16*>(out.data_ptr()), out.numel(),
                                 num_ctas);
      break;
    }
#endif
    default:
      throw std::runtime_error("custom allreduce only supports float32, float16 and bfloat16");
  }
}

void dispose(fptr_t _fa) { delete reinterpret_cast<vllm::CustomAllreduce*>(_fa); }

int64_t meta_size() { return sizeof(vllm::Signal); }

void register_buffer(fptr_t _fa, Array<fptr_t> fake_ipc_ptrs) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  TVM_FFI_ICHECK_EQ(fake_ipc_ptrs.size(), fa->world_size_);
  void* ipc_ptrs[8];
  for (int i = 0; i < fake_ipc_ptrs.size(); i++) {
    ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  }
  fa->register_buffer(ipc_ptrs);
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
Tuple<Array<int64_t>, Array<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return Tuple<Array<int64_t>, Array<int64_t>>(Array<int64_t>(bytes), Array<int64_t>(offsets));
}

// Use vector<int64_t> to represent byte data for python binding compatibility.
void register_graph_buffers(fptr_t _fa, Array<Array<int64_t>> handles,
                            Array<Array<int64_t>> offsets) {
  auto fa = reinterpret_cast<vllm::CustomAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (int i = 0; i < handles.size(); i++) {
    bytes.emplace_back(handles[i].begin(), handles[i].end());
  }
  bytes.reserve(handles.size());
  std::vector<std::vector<int64_t>> off(offsets.size());
  for (int i = 0; i < offsets.size(); ++i) {
    off[i] = std::vector<int64_t>(offsets[i].begin(), offsets[i].end());
  }
  fa->register_graph_buffers(bytes, off);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_graph_buffer_ipc_meta, get_graph_buffer_ipc_meta);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_graph_buffers, register_graph_buffers);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispose, dispose);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(meta_size, meta_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(register_buffer, register_buffer);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(init_custom_ar, init_custom_ar);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(all_reduce, all_reduce);
