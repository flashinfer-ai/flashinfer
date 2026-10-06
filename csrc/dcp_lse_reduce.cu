#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <flashinfer/comm/dcp_lse_reduce.cuh>
#include <string>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>
#include <vector>

#include "tvm_ffi_utils.h"

namespace flashinfer::comm::dcp {

namespace symm = c10d::symmetric_memory;

namespace {

// Element strides of the [tokens, heads, cp_size, head_dim] view the kernel reads.
struct InputGeometry {
  int64_t num_tokens;
  int64_t local_heads;
  int64_t o_s_tok, o_s_head, o_s_peer;
  int64_t lse_s_tok, lse_s_head, lse_s_peer;
};

// A 4-D input may be any strided view whose innermost dimension is contiguous
// (e.g. an attention output [B, cp_size * H, D] viewed as [B, H, cp_size, D]), so
// callers need not pack it first. Inputs of other ranks must be contiguous; every
// point of their leading shape is an independent reduction row.
InputGeometry DescribeInput(const at::Tensor& partial_o, const at::Tensor& partial_lse) {
  InputGeometry g{};
  if (partial_o.dim() == 4) {
    g.num_tokens = partial_o.size(0);
    g.local_heads = partial_o.size(1);
    g.o_s_tok = partial_o.stride(0);
    g.o_s_head = partial_o.stride(1);
    g.o_s_peer = partial_o.stride(2);
    g.lse_s_tok = partial_lse.stride(0);
    g.lse_s_head = partial_lse.stride(1);
    g.lse_s_peer = partial_lse.stride(2);
    return g;
  }
  TORCH_CHECK(partial_o.is_contiguous() && partial_lse.is_contiguous(),
              "partial_o and partial_lse must be contiguous unless they are 4-D");
  g.num_tokens = 1;
  for (int64_t i = 0; i < partial_o.dim() - 2; ++i) g.num_tokens *= partial_o.size(i);
  g.local_heads = 1;
  g.o_s_tok = partial_o.size(-2) * partial_o.size(-1);
  g.o_s_peer = partial_o.size(-1);
  g.lse_s_tok = partial_lse.size(-1);
  g.lse_s_peer = 1;
  return g;
}

// Every rank's workspace as mapped on this device.
struct PeerWorkspaces {
  c10::intrusive_ptr<symm::SymmetricMemory> handle;  // keeps the mappings alive for the call
  unsigned char* const* bases;                       // device array [cp_size]
  size_t payload_offset;  // workspace offset within its buffer + kPayloadOffset
};

// Works with any torch symmetric-memory backend (CUDA, NCCL, NVSHMEM): each exposes
// every rank's buffer base, null for a rank it cannot map for load/store.
PeerWorkspaces LookUpPeers(const at::Tensor& workspace, int64_t cp_rank, int64_t cp_size,
                           const std::string& group_name) {
  // create_workspace already rendezvoused (workspace, group_name): this returns the
  // cached handle and is not a collective.
  auto handle = symm::rendezvous(workspace, group_name);
  TORCH_CHECK(handle.defined(),
              "workspace must be allocated via torch symmetric memory and rendezvoused first");
  // Only evaluated when a check fails.
  const auto backend = [&] { return symm::get_backend(workspace.device()).value_or("unknown"); };
  TORCH_CHECK(handle->get_world_size() == cp_size, "cp_size (", cp_size,
              ") does not match the workspace group size (", handle->get_world_size(), ")");
  TORCH_CHECK(handle->get_rank() == cp_rank, "cp_rank (", cp_rank,
              ") does not match this rank in the workspace group (", handle->get_rank(), ")");
  const size_t offset = handle->get_offset();
  TORCH_CHECK(offset + static_cast<size_t>(workspace.numel()) <= handle->get_buffer_size(),
              "workspace extends past its symmetric-memory buffer");
  const std::vector<void*> bases = handle->get_buffer_ptrs();
  void** bases_dev = handle->get_buffer_ptrs_dev();
  TORCH_CHECK(static_cast<int64_t>(bases.size()) == cp_size && bases_dev != nullptr, "the ",
              backend(), " symmetric-memory backend exposes no peer buffer pointers");
  const size_t payload_offset = offset + kPayloadOffset;
  for (int64_t r = 0; r < cp_size; ++r) {
    TORCH_CHECK(bases[r] != nullptr, "rank ", r,
                "'s workspace is not mapped for load/store by the ", backend(),
                " symmetric-memory backend; all ranks must share one NVLink domain");
    // LL128 lines must not straddle a 128-byte boundary in any rank's workspace.
    TORCH_CHECK((reinterpret_cast<uintptr_t>(bases[r]) + payload_offset) % kLineBytes == 0, "rank ",
                r, "'s workspace payload is not 128-byte aligned");
  }
  return {std::move(handle), reinterpret_cast<unsigned char* const*>(bases_dev), payload_offset};
}

template <typename T, bool BaseE>
void LaunchFused(const at::Tensor& partial_o, const at::Tensor& partial_lse, const InputGeometry& g,
                 unsigned char* workspace, unsigned char* const* peer_workspaces,
                 size_t peer_payload_offset, at::Tensor& output, int rank, int cp_size,
                 int max_rows, int head_dim, cudaStream_t stream) {
  auto* kernel = FusedKernel<T, BaseE>;
  int blocks_per_sm = 0;
  C10_CUDA_CHECK(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, kFusedBlockSize, 0));
  const cudaDeviceProp* props = at::cuda::getCurrentDeviceProperties();
  TORCH_CHECK(props->cooperativeLaunch, "decode_cp_a2a_lse_reduce requires cooperative launch");
  const int max_cooperative_blocks = blocks_per_sm * props->multiProcessorCount;
  TORCH_CHECK(max_cooperative_blocks > 0,
              "decode_cp_a2a_lse_reduce has no cooperative launch capacity");
  // One warp per send item (row, destination) and per receive item (row, line group).
  const int64_t num_rows = g.num_tokens * g.local_heads;
  const int groups = LineGroups(head_dim, static_cast<int>(sizeof(T)));
  const int64_t warp_items = num_rows * std::max(cp_size, groups);
  const int64_t desired_blocks = (warp_items + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int grid_blocks = static_cast<int>(
      std::max<int64_t>(1, std::min<int64_t>(desired_blocks, max_cooperative_blocks)));

  const T* o_ptr = reinterpret_cast<const T*>(partial_o.data_ptr());
  const float* lse_ptr = partial_lse.data_ptr<float>();
  T* output_ptr = reinterpret_cast<T*>(output.data_ptr());
  int64_t o_s_tok = g.o_s_tok, o_s_head = g.o_s_head, o_s_peer = g.o_s_peer;
  int64_t lse_s_tok = g.lse_s_tok, lse_s_head = g.lse_s_head, lse_s_peer = g.lse_s_peer;
  int num_tokens = static_cast<int>(g.num_tokens), local_heads = static_cast<int>(g.local_heads);
  void* args[] = {
      &o_ptr,
      &lse_ptr,
      &o_s_tok,
      &o_s_head,
      &o_s_peer,
      &lse_s_tok,
      &lse_s_head,
      &lse_s_peer,
      &workspace,
      &peer_workspaces,
      &peer_payload_offset,
      &output_ptr,
      &rank,
      &cp_size,
      &num_tokens,
      &local_heads,
      &max_rows,
      &head_dim,
  };
  C10_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3(grid_blocks),
                                             dim3(kFusedBlockSize), args, 0, stream));
}

}  // namespace

at::Tensor dcp_lse_reduce(const at::Tensor& partial_o, const at::Tensor& partial_lse,
                          const at::Tensor& workspace, int64_t cp_rank, int64_t cp_size,
                          bool is_lse_base_on_e, const std::string& group_name) {
  TORCH_CHECK(partial_o.is_cuda(), "partial_o must be a CUDA tensor");
  TORCH_CHECK(partial_lse.is_cuda(), "partial_lse must be a CUDA tensor");
  TORCH_CHECK(workspace.is_cuda(), "workspace must be a CUDA tensor");
  TORCH_CHECK(workspace.is_contiguous(), "workspace must be contiguous");
  TORCH_CHECK(partial_o.dim() >= 3, "partial_o must be at least 3-D [..., cp_size, head_dim]");
  TORCH_CHECK(partial_lse.dim() == partial_o.dim() - 1,
              "partial_lse must have one fewer dimension than partial_o");
  TORCH_CHECK(partial_o.scalar_type() == at::kHalf || partial_o.scalar_type() == at::kBFloat16,
              "partial_o must be float16 or bfloat16");
  TORCH_CHECK(partial_lse.scalar_type() == at::kFloat, "partial_lse must be float32");
  TORCH_CHECK(workspace.scalar_type() == at::kByte, "workspace must be a uint8 tensor");
  TORCH_CHECK(
      partial_o.device() == partial_lse.device() && partial_o.device() == workspace.device(),
      "partial_o, partial_lse, and workspace must be on the same device");
  TORCH_CHECK(cp_size > 0 && cp_size <= kMaxRanks, "cp_size must be in [1, ", kMaxRanks, "]");
  TORCH_CHECK(cp_rank >= 0 && cp_rank < cp_size, "cp_rank must be in [0, cp_size)");
  TORCH_CHECK(partial_o.size(-2) == cp_size,
              "partial_o second-to-last dimension must equal cp_size");
  TORCH_CHECK(partial_lse.size(-1) == cp_size, "partial_lse last dimension must equal cp_size");
  for (int64_t i = 0; i < partial_lse.dim() - 1; ++i) {
    TORCH_CHECK(partial_o.size(i) == partial_lse.size(i),
                "partial_o and partial_lse leading dimensions must match");
  }

  const int64_t head_dim_i64 = partial_o.size(-1);
  const int64_t element_size = partial_o.element_size();
  TORCH_CHECK(partial_o.stride(-1) == 1, "partial_o innermost dimension must be contiguous");
  TORCH_CHECK(head_dim_i64 * element_size % 16 == 0, "partial_o rows must be 16-byte aligned");
  TORCH_CHECK(head_dim_i64 <= INT_MAX && LineGroups(static_cast<int>(head_dim_i64),
                                                    static_cast<int>(element_size)) <= kMaxGroups,
              "partial_o rows must not exceed ", kMaxGroups * kLinesPerGroup * kLineDataWords * 8,
              " bytes");
  const InputGeometry geometry = DescribeInput(partial_o, partial_lse);
  TORCH_CHECK(reinterpret_cast<uintptr_t>(partial_o.data_ptr()) % 8 == 0 &&
                  geometry.o_s_tok * element_size % 8 == 0 &&
                  geometry.o_s_head * element_size % 8 == 0 &&
                  geometry.o_s_peer * element_size % 8 == 0,
              "every partial_o row must start on an 8-byte boundary");
  const int64_t num_rows_i64 = geometry.num_tokens * geometry.local_heads;
  TORCH_CHECK(num_rows_i64 > 0, "zero-token inputs are not supported");

  TORCH_CHECK(workspace.storage_offset() == 0,
              "workspace must be the base symmetric-memory tensor, not a view");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % kLineBytes == 0,
              "workspace must be 128-byte aligned");
  const PeerWorkspaces peers = LookUpPeers(workspace, cp_rank, cp_size, group_name);

  c10::cuda::CUDAGuard guard(partial_o.device());
  const auto stream = at::cuda::getCurrentCUDAStream();

  const int head_dim = static_cast<int>(head_dim_i64);
  const size_t row_bytes = RowBytes(head_dim, static_cast<int>(element_size));
  const size_t workspace_bytes = static_cast<size_t>(workspace.numel());
  const size_t bytes_per_row = kNumSlots * static_cast<size_t>(cp_size) * row_bytes;
  TORCH_CHECK(
      workspace_bytes >= kPayloadOffset && (workspace_bytes - kPayloadOffset) % bytes_per_row == 0,
      "workspace has an invalid size for this tensor geometry");
  const int64_t max_rows_i64 =
      static_cast<int64_t>((workspace_bytes - kPayloadOffset) / bytes_per_row);
  TORCH_CHECK(num_rows_i64 <= max_rows_i64, "input token count exceeds workspace capacity");
  TORCH_CHECK(max_rows_i64 * std::max<int64_t>(cp_size, kMaxGroups) <= INT_MAX,
              "tensor geometry exceeds kernel integer limits");

  std::vector<int64_t> output_shape(partial_o.sizes().begin(), partial_o.sizes().end() - 2);
  output_shape.push_back(head_dim_i64);
  at::Tensor output = at::empty(output_shape, partial_o.options());  // contiguous
  auto* workspace_ptr = static_cast<unsigned char*>(workspace.data_ptr());
  const int rank = static_cast<int>(cp_rank);
  const int cp = static_cast<int>(cp_size);
  const int max_rows = static_cast<int>(max_rows_i64);

  if (partial_o.scalar_type() == at::kHalf) {
    if (is_lse_base_on_e) {
      LaunchFused<__half, true>(partial_o, partial_lse, geometry, workspace_ptr, peers.bases,
                                peers.payload_offset, output, rank, cp, max_rows, head_dim, stream);
    } else {
      LaunchFused<__half, false>(partial_o, partial_lse, geometry, workspace_ptr, peers.bases,
                                 peers.payload_offset, output, rank, cp, max_rows, head_dim,
                                 stream);
    }
  } else {
    if (is_lse_base_on_e) {
      LaunchFused<__nv_bfloat16, true>(partial_o, partial_lse, geometry, workspace_ptr, peers.bases,
                                       peers.payload_offset, output, rank, cp, max_rows, head_dim,
                                       stream);
    } else {
      LaunchFused<__nv_bfloat16, false>(partial_o, partial_lse, geometry, workspace_ptr,
                                        peers.bases, peers.payload_offset, output, rank, cp,
                                        max_rows, head_dim, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

}  // namespace flashinfer::comm::dcp

TORCH_LIBRARY_FRAGMENT(flashinfer, m) {
  m.def(
      "decode_cp_a2a_lse_reduce(Tensor partial_o, Tensor partial_lse, Tensor(a!) workspace, "
      "int cp_rank, int cp_size, bool is_lse_base_on_e, str group_name) -> Tensor");
}

TORCH_LIBRARY_IMPL(flashinfer, CUDA, m) {
  m.impl("decode_cp_a2a_lse_reduce", TORCH_FN(flashinfer::comm::dcp::dcp_lse_reduce));
}

// Give FlashInfer's TVM-FFI JIT loader an exported symbol; loading this module
// also runs the TORCH_LIBRARY static initializers above.
namespace {
bool isDcpLseReduceLoaded() { return true; }
}  // namespace
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dcp_lse_reduce_loaded, isDcpLseReduceLoaded);
