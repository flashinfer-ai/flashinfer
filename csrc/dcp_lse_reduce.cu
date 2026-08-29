#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>

#include <torch/csrc/distributed/c10d/NCCLUtils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/NCCLSymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_dev_cap.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/nccl_devcomm_manager.hpp>

#include <flashinfer/comm/dcp_lse_reduce.cuh>
#include "tvm_ffi_utils.h"

#include <climits>
#include <cstdint>
#include <string>
#include <vector>

namespace flashinfer::comm::dcp {

using c10d::symmetric_memory::NCCLDevCommManager;
using c10d::symmetric_memory::NCCLSymmetricMemory;

namespace {

template <typename T, bool BaseE>
void launch_merge(const unsigned char* workspace, size_t out_region_offset, size_t slot_out_bytes,
                  size_t lse_region_offset, size_t slot_lse_bytes, const uint32_t* state,
                  at::Tensor& output, int cp_size, int num_tokens, int max_tokens, int local_heads,
                  int head_dim, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(num_tokens), static_cast<unsigned int>(local_heads));
  const size_t smem = static_cast<size_t>(cp_size) * sizeof(float);
  MergeKernel<T, BaseE><<<grid, kMergeBlockSize, smem, stream>>>(
      workspace, out_region_offset, slot_out_bytes, lse_region_offset, slot_lse_bytes, state,
      reinterpret_cast<T*>(output.data_ptr()), cp_size, num_tokens, max_tokens, local_heads,
      head_dim);
}

}  // namespace

at::Tensor dcp_lse_reduce(const at::Tensor& partial_o, const at::Tensor& partial_lse,
                          const at::Tensor& workspace, int64_t cp_rank, int64_t cp_size,
                          bool is_lse_base_on_e, const std::string& group_name) {
#ifdef NCCL_HAS_DEVCOMM
  TORCH_CHECK(partial_o.is_cuda(), "partial_o must be a CUDA tensor");
  TORCH_CHECK(partial_lse.is_cuda(), "partial_lse must be a CUDA tensor");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
  TORCH_CHECK(partial_o.is_contiguous(), "partial_o must be contiguous");
  TORCH_CHECK(partial_lse.is_contiguous(), "partial_lse must be contiguous");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(partial_o.dim() >= 3,
              "partial_o must be at least 3-D [..., cp_size, head_dim]");
  TORCH_CHECK(partial_lse.dim() == partial_o.dim() - 1,
              "partial_lse must have one fewer dimension than partial_o");
  TORCH_CHECK(partial_o.scalar_type() == at::kHalf || partial_o.scalar_type() == at::kBFloat16,
              "partial_o must be float16 or bfloat16");
  TORCH_CHECK(partial_lse.scalar_type() == at::kFloat, "partial_lse must be float32");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be a uint8 tensor");
  TORCH_CHECK(partial_o.device() == partial_lse.device() &&
                  partial_o.device() == workspace.device(),
              "partial_o, partial_lse, and workspace must be on the same device");
  TORCH_CHECK(cp_size > 0 && cp_size <= kMaxRanks, "cp_size must be in [1, ", kMaxRanks, "]");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
  TORCH_CHECK(partial_o.size(-2) == cp_size,
              "partial_o second-to-last dimension must equal cp_size");
  TORCH_CHECK(partial_lse.size(-1) == cp_size,
              "partial_lse last dimension must equal cp_size");
  for (int64_t i = 0; i < partial_lse.dim() - 1; ++i) {
    TORCH_CHECK(partial_o.size(i) == partial_lse.size(i),
                "partial_o and partial_lse leading dimensions must match");
  }

  const int64_t head_dim_i64 = partial_o.size(-1);
  TORCH_CHECK(head_dim_i64 * partial_o.element_size() % 16 == 0,
              "partial_o rows must be 16-byte aligned");

  // Every point in the leading shape is an independent reduction entry. This
  // handles [batch, heads, cp, dim], [rows, cp, dim], and additional batch
  // dimensions without assigning semantics to any one leading axis.
  constexpr int64_t local_heads_i64 = 1;
  int64_t num_tokens_i64 = 1;
  for (int64_t i = 0; i < partial_o.dim() - 2; ++i) {
    num_tokens_i64 *= partial_o.size(i);
  }
  TORCH_CHECK(num_tokens_i64 > 0, "zero-token inputs are not supported");

  auto symm_mem = c10d::symmetric_memory::rendezvous(workspace, group_name);
  TORCH_CHECK(symm_mem != nullptr,
              "workspace must be allocated via torch symmetric memory and rendezvoused first");
  auto* nccl_hdl = dynamic_cast<NCCLSymmetricMemory*>(symm_mem.get());
  TORCH_CHECK(nccl_hdl != nullptr,
              "workspace requires the NCCL torch symmetric-memory backend");

  c10::cuda::CUDAGuard guard(partial_o.device());
  const auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(nccl_hdl->get_group_name() == group_name,
              "workspace was rendezvoused with a different process group");
  auto& manager = NCCLDevCommManager::get(partial_o.device());
  ncclComm_t comm = manager.get_comm(group_name);

  static constexpr char kDevcommKey[] = "flashinfer_decode_cp_a2a_lse_reduce";
  auto devcomm_opt = manager.get_devcomm(group_name, kDevcommKey);
  if (!devcomm_opt) {
    ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
    reqs.lsaBarrierCount = 1;
    ncclDevComm devcomm;
    C10D_NCCL_CHECK(ncclDevCommCreate(comm, &reqs, &devcomm),
                    "ncclDevCommCreate failed in decode_cp_a2a_lse_reduce");
    devcomm_opt = manager.register_devcomm(group_name, devcomm, kDevcommKey);
  }
  ncclDevComm& devcomm = devcomm_opt->get();
  TORCH_CHECK(devcomm.nRanks == cp_size, "cp_size does not match the workspace group");
  TORCH_CHECK(devcomm.rank == cp_rank, "cp_rank does not match the workspace group");
  TORCH_CHECK(devcomm.lsaSize == cp_size,
              "decode_cp_a2a_lse_reduce requires one NCCL LSA/NVLink domain");

  ncclWindow_t window = nccl_hdl->get_window();
  TORCH_CHECK(window != nullptr, "NCCL symmetric-memory window is null");
  TORCH_CHECK(workspace.storage_offset() == 0,
              "workspace must be the base symmetric-memory tensor, not a view");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 16 == 0,
              "workspace must be 16-byte aligned");
  TORCH_CHECK((nccl_hdl->get_window_offset() + kMetadataBytes) % 16 == 0,
              "payload offset within the NCCL symmetric window must be 16-byte aligned");

  const size_t workspace_bytes = static_cast<size_t>(workspace.numel());
  const size_t bytes_per_token =
      kNumSlots * static_cast<size_t>(cp_size) * static_cast<size_t>(local_heads_i64) *
      (static_cast<size_t>(head_dim_i64) * partial_o.element_size() + sizeof(float));
  TORCH_CHECK(workspace_bytes >= kMetadataBytes &&
                  (workspace_bytes - kMetadataBytes) % bytes_per_token == 0,
              "workspace has an invalid size for this tensor geometry");
  const int64_t max_tokens_i64 =
      static_cast<int64_t>((workspace_bytes - kMetadataBytes) / bytes_per_token);
  TORCH_CHECK(num_tokens_i64 <= max_tokens_i64, "input token count exceeds workspace capacity");

  TORCH_CHECK(num_tokens_i64 <= INT_MAX && local_heads_i64 <= INT_MAX &&
                  head_dim_i64 <= INT_MAX && max_tokens_i64 <= INT_MAX,
              "tensor geometry exceeds kernel integer limits");
  const int num_tokens = static_cast<int>(num_tokens_i64);
  const int local_heads = static_cast<int>(local_heads_i64);
  const int head_dim = static_cast<int>(head_dim_i64);
  const int max_tokens = static_cast<int>(max_tokens_i64);

  const size_t rows_per_slot = static_cast<size_t>(cp_size) * max_tokens * local_heads;
  const size_t slot_out_bytes = rows_per_slot * head_dim * partial_o.element_size();
  const size_t slot_lse_bytes = rows_per_slot * sizeof(float);
  const size_t window_base_offset = nccl_hdl->get_window_offset();
  const size_t out_region_window_offset = window_base_offset + kMetadataBytes;
  const size_t lse_region_window_offset =
      out_region_window_offset + kNumSlots * slot_out_bytes;
  auto* workspace_ptr = static_cast<unsigned char*>(workspace.data_ptr());
  auto* state = reinterpret_cast<uint32_t*>(workspace_ptr);

  std::vector<int64_t> output_shape(partial_o.sizes().begin(), partial_o.sizes().end() - 2);
  output_shape.push_back(head_dim_i64);
  at::Tensor output = at::empty(output_shape, partial_o.options());

  SelectSlotKernel<<<1, 1, 0, stream>>>(state);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const dim3 put_grid(static_cast<unsigned int>(num_tokens * local_heads),
                      static_cast<unsigned int>(cp_size));
  if (partial_o.scalar_type() == at::kHalf) {
    PutPackedKernel<__half><<<put_grid, kPutBlockSize, 0, stream>>>(
        reinterpret_cast<const __half*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>(), window, out_region_window_offset, slot_out_bytes,
        lse_region_window_offset, slot_lse_bytes, state, cp_rank, cp_size, num_tokens,
        max_tokens, local_heads, head_dim);
  } else {
    PutPackedKernel<__nv_bfloat16><<<put_grid, kPutBlockSize, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(partial_o.data_ptr()),
        partial_lse.data_ptr<float>(), window, out_region_window_offset, slot_out_bytes,
        lse_region_window_offset, slot_lse_bytes, state, cp_rank, cp_size, num_tokens,
        max_tokens, local_heads, head_dim);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  BarrierKernel<<<1, 32, 0, stream>>>(devcomm);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const size_t out_region_local_offset = kMetadataBytes;
  const size_t lse_region_local_offset =
      out_region_local_offset + kNumSlots * slot_out_bytes;
  if (partial_o.scalar_type() == at::kHalf) {
    if (is_lse_base_on_e) {
      launch_merge<__half, true>(workspace_ptr, out_region_local_offset, slot_out_bytes,
                                 lse_region_local_offset, slot_lse_bytes, state, output, cp_size,
                                 num_tokens, max_tokens, local_heads, head_dim, stream);
    } else {
      launch_merge<__half, false>(workspace_ptr, out_region_local_offset, slot_out_bytes,
                                  lse_region_local_offset, slot_lse_bytes, state, output, cp_size,
                                  num_tokens, max_tokens, local_heads, head_dim, stream);
    }
  } else {
    if (is_lse_base_on_e) {
      launch_merge<__nv_bfloat16, true>(
          workspace_ptr, out_region_local_offset, slot_out_bytes, lse_region_local_offset,
          slot_lse_bytes, state, output, cp_size, num_tokens, max_tokens, local_heads, head_dim,
          stream);
    } else {
      launch_merge<__nv_bfloat16, false>(
          workspace_ptr, out_region_local_offset, slot_out_bytes, lse_region_local_offset,
          slot_lse_bytes, state, output, cp_size, num_tokens, max_tokens, local_heads, head_dim,
          stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
#else
  TORCH_CHECK(false,
              "decode_cp_a2a_lse_reduce requires NCCL >= 2.29 with device communicator support");
#endif
}

}  // namespace flashinfer::comm::dcp

TORCH_LIBRARY_FRAGMENT(flashinfer, m) {
  m.def("decode_cp_a2a_lse_reduce(Tensor partial_o, Tensor partial_lse, Tensor(a!) workspace, "
        "int cp_rank, int cp_size, bool is_lse_base_on_e, str group_name) -> Tensor");
}

TORCH_LIBRARY_IMPL(flashinfer, CUDA, m) {
  m.impl("decode_cp_a2a_lse_reduce", TORCH_FN(flashinfer::comm::dcp::dcp_lse_reduce));
}

// Give FlashInfer's TVM-FFI JIT loader an exported symbol; loading this module
// also runs the TORCH_LIBRARY static initializers above.
namespace {
bool isDcpLseReduceLoaded() {
  return true;
}
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dcp_lse_reduce_loaded, isDcpLseReduceLoaded);
