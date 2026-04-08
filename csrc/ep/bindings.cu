// csrc/ep/bindings.cu
//
// TVM FFI bindings for the FlashInfer Unified EP API.
// JIT-compiled by flashinfer/jit/ep.py, loaded via tvm_ffi.
//
// This file implements the COMPLETE dispatch/combine pipeline in C++:
//   1. Routing computation (expert counts, per-rank send/recv counts)
//   2. Alltoall communication via NCCL P2P (ncclSend/ncclRecv)
//   3. Layout normalization (2D <-> 3D scatter/gather kernels)
//   4. Combine: reverse alltoall + weighted reduction
//
// Python is a thin passthrough — only converts torch types to TVM FFI.
//
// Exported TVM FFI functions:
//   ep_create_group, ep_destroy_group,
//   ep_get_dispatch_layout,
//   ep_dispatch, ep_combine,
//   ep_create_handle, ep_destroy_handle,
//   ep_handle_get_num_recv, ep_handle_invoke_deferred,
//   ep_layout_scatter_2d_to_3d, ep_layout_gather_3d_to_2d

#include <cstdint>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <numeric>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>

#include <tvm/ffi/container/tuple.h>
#include <tvm/ffi/function.h>

#include "../tvm_ffi_utils.h"

using tvm::ffi::TensorView;
using tvm::ffi::Tensor;
using tvm::ffi::Tuple;
using tvm::ffi::Shape;

#define NCCL_CHECK(cmd) do {                                    \
  ncclResult_t _nccl_r = (cmd);                                 \
  if (_nccl_r != ncclSuccess) {                                 \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",                  \
            __FILE__, __LINE__, ncclGetErrorString(_nccl_r));    \
    throw std::runtime_error(ncclGetErrorString(_nccl_r));       \
  }                                                             \
} while(0)

// ─── Internal state ─────────────────────────────────────────────────

namespace {

struct EpGroupState {
  int64_t rank;
  int64_t world_size;
  int64_t num_experts;
  int64_t num_local_experts;
  int64_t top_k;
  int64_t hidden_dim;
  int64_t backend;  // 0 = deepep, 1 = nccl_ep
  int64_t cuda_graph_max_tokens;
  ncclComm_t nccl_comm;  // NCCL communicator for alltoall
  bool destroyed;
};

struct EpHandleState {
  int64_t group_id;
  int64_t num_recv_tokens;

  // Routing metadata preserved for combine (reverse alltoall)
  std::vector<int64_t> send_counts;          // [world_size] — per-rank totals sent
  std::vector<int64_t> recv_counts;          // [world_size] — per-rank totals received
  std::vector<int64_t> send_expert_counts;   // [ws * num_local] — per-rank per-expert sent
  std::vector<int64_t> recv_expert_counts;   // [ws * num_local] — per-rank per-expert recvd
  std::vector<int64_t> sort_indices;         // argsort of flat_experts by (dest_rank, expert)
  int64_t num_tokens;                        // original num_tokens (before expand)
  int64_t top_k;
  int64_t hidden_dim;

  bool has_deferred;
  bool destroyed;
};

static std::mutex g_mutex;
static int64_t g_next_group_id = 1;
static int64_t g_next_handle_id = 1;
static std::unordered_map<int64_t, EpGroupState> g_groups;
static std::unordered_map<int64_t, EpHandleState> g_handles;

// ─── Helpers ───────────────────────────────────────────────────────

inline ncclDataType_t to_nccl_dtype(DLDataType dtype) {
  if (dtype.code == kDLFloat && dtype.bits == 16) return ncclFloat16;
  if (dtype.code == kDLBfloat && dtype.bits == 16) return ncclBfloat16;
  if (dtype.code == kDLFloat && dtype.bits == 32) return ncclFloat32;
  if (dtype.code == kDLFloat && dtype.bits == 64) return ncclFloat64;
  if (dtype.code == kDLInt && dtype.bits == 32) return ncclInt32;
  if (dtype.code == kDLInt && dtype.bits == 64) return ncclInt64;
  // FP8: treat as raw bytes via ncclUint8. Caller must multiply count by elem_size
  // and divide by 1 (uint8 is 1 byte). This avoids dependency on NCCL version
  // for ncclFp8E4M3/ncclFp8E5M2 enum values which vary across releases.
  if (dtype.bits == 8) return ncclUint8;
  return ncclFloat32;  // fallback
}

// ─── Group lifecycle ────────────────────────────────────────────────

int64_t epCreateGroup(
    int64_t rank,
    int64_t world_size,
    int64_t num_experts,
    int64_t num_local_experts,
    int64_t top_k,
    int64_t hidden_dim,
    int64_t backend,
    int64_t cuda_graph_max_tokens,
    int64_t nccl_comm_ptr) {

  // Validate ncclComm_t pointer
  if (nccl_comm_ptr == 0) {
    throw std::invalid_argument(
        "ep_create_group: nccl_comm_ptr is NULL. "
        "The ncclComm_t extraction from the PyTorch process group failed. "
        "Ensure the process group uses the 'nccl' backend and has been "
        "initialized (e.g., by performing a dummy allreduce before calling "
        "create_group).");
  }

  // Validate basic parameters
  if (rank < 0 || rank >= world_size) {
    throw std::invalid_argument(
        "ep_create_group: invalid rank=" + std::to_string(rank) +
        " for world_size=" + std::to_string(world_size));
  }
  if (num_experts <= 0 || num_local_experts <= 0) {
    throw std::invalid_argument(
        "ep_create_group: num_experts=" + std::to_string(num_experts) +
        " and num_local_experts=" + std::to_string(num_local_experts) +
        " must be positive");
  }
  if (num_experts % world_size != 0) {
    throw std::invalid_argument(
        "ep_create_group: num_experts=" + std::to_string(num_experts) +
        " must be divisible by world_size=" + std::to_string(world_size));
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  int64_t id = g_next_group_id++;
  EpGroupState s;
  s.rank = rank;
  s.world_size = world_size;
  s.num_experts = num_experts;
  s.num_local_experts = num_local_experts;
  s.top_k = top_k;
  s.hidden_dim = hidden_dim;
  s.backend = backend;
  s.cuda_graph_max_tokens = cuda_graph_max_tokens;
  s.nccl_comm = reinterpret_cast<ncclComm_t>(nccl_comm_ptr);
  s.destroyed = false;
  g_groups[id] = s;

  fprintf(stderr, "[FlashInfer EP] Group %lld created: rank=%lld/%lld, "
          "experts=%lld (local=%lld), top_k=%lld, hidden=%lld, "
          "nccl_comm=%p\n",
          (long long)id, (long long)rank, (long long)world_size,
          (long long)num_experts, (long long)num_local_experts,
          (long long)top_k, (long long)hidden_dim, (void*)s.nccl_comm);

  return id;
}

void epDestroyGroup(int64_t group_id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_groups.find(group_id);
  if (it != g_groups.end()) {
    it->second.destroyed = true;
  }
}

// ─── Layout computation ─────────────────────────────────────────────

Tensor epGetDispatchLayout(
    TensorView topk_idx,   // [num_tokens, top_k] int64 on GPU
    int64_t group_id) {

  EpGroupState g;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    g = g_groups.at(group_id);
  }

  cudaStream_t stream = get_current_stream();
  int64_t num_tokens = topk_idx.size(0);
  int64_t top_k = topk_idx.size(1);
  int64_t num_local = g.num_local_experts;
  int64_t local_start = g.rank * num_local;

  // Copy topk_idx to host
  std::vector<int64_t> host_idx(num_tokens * top_k);
  cudaMemcpyAsync(host_idx.data(), topk_idx.data_ptr(),
                  num_tokens * top_k * sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Count THIS rank's tokens that target each local expert (send-side).
  // This is a local-only computation — no alltoall needed.
  // The sum across all ranks of these counts equals num_tokens * top_k
  // (each token-expert pair maps to exactly one rank's local expert).
  auto counts = alloc_tensor(Shape({num_local}), dl_int32, DLDevice{kDLCPU, 0});
  auto* counts_ptr = static_cast<int32_t*>(counts.data_ptr());
  std::memset(counts_ptr, 0, num_local * sizeof(int32_t));

  for (int64_t i = 0; i < num_tokens * top_k; ++i) {
    int64_t eid = host_idx[i];
    int64_t local_eid = eid - local_start;
    if (local_eid >= 0 && local_eid < num_local) {
      counts_ptr[local_eid]++;
    }
  }

  return counts;
}

// ─── Dispatch (scatter) — full C++ implementation ───────────────────
//
// 1. Copy topk_idx to host, compute per-rank send counts
// 2. alltoall the counts (ncclSend/ncclRecv)
// 3. Sort hidden by dest rank, alltoall the hidden states
// 4. Return recv_hidden + routing metadata in an EpHandle
//
// Returns a Tuple of:
//   [0] recv_hidden:      Tensor [total_recv, hidden_dim] on GPU
//   [1] expert_counts:    Tensor [num_local_experts] int32 on CPU
//   [2] handle_id:        int64

Tuple<Tensor, Tensor, int64_t> epDispatch(
    TensorView hidden,      // [num_tokens, hidden_dim] bf16/fp16/fp32 on GPU
    TensorView topk_idx,    // [num_tokens, top_k] int64 on GPU
    int64_t group_id,
    int64_t output_layout)  // 0 = FLAT_2D, 1 = EXPERT_MAJOR_3D
{
  EpGroupState g;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_groups.find(group_id);
    if (it == g_groups.end()) {
      throw std::invalid_argument(
          "ep_dispatch: invalid group_id=" + std::to_string(group_id) +
          ". Was the group destroyed or never created?");
    }
    g = it->second;
    if (g.destroyed) {
      throw std::invalid_argument(
          "ep_dispatch: group_id=" + std::to_string(group_id) +
          " has been destroyed");
    }
    if (g.nccl_comm == nullptr) {
      throw std::runtime_error(
          "ep_dispatch: ncclComm_t is NULL for group_id=" +
          std::to_string(group_id) +
          ". The NCCL communicator was not properly extracted from the "
          "PyTorch process group. Check _extract_nccl_comm_ptr() output.");
    }
  }

  cudaStream_t stream = get_current_stream();
  int64_t num_tokens = hidden.size(0);
  int64_t hidden_dim = hidden.size(1);
  int64_t top_k = topk_idx.size(1);
  int64_t ws = g.world_size;
  int64_t experts_per_rank = g.num_experts / ws;
  int64_t num_local = g.num_local_experts;
  int64_t local_start = g.rank * num_local;
  DLDataType dtype = hidden.dtype();
  ncclDataType_t nccl_dtype = to_nccl_dtype(dtype);
  int elem_size = (dtype.bits * dtype.lanes + 7) / 8;

  // --- 1. Copy topk_idx to host, compute routing ---

  std::vector<int64_t> host_idx(num_tokens * top_k);
  cudaMemcpyAsync(host_idx.data(), topk_idx.data_ptr(),
                  num_tokens * top_k * sizeof(int64_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // Per-expert global counts
  std::vector<int64_t> expert_counts_global(g.num_experts, 0);
  for (auto eid : host_idx) {
    if (eid >= 0 && eid < g.num_experts) expert_counts_global[eid]++;
  }

  // --- 2. Alltoall per-expert counts ---
  //
  // Each rank sends expert_counts for the dest rank's local experts.
  // send_expert_counts[r * num_local .. (r+1) * num_local] = counts for
  //   the num_local experts on rank r, from THIS rank's tokens.
  // After alltoall, recv_expert_counts[r * num_local .. (r+1) * num_local]
  //   = counts from rank r for THIS rank's local experts.

  std::vector<int64_t> send_expert_counts(ws * num_local, 0);
  for (int64_t r = 0; r < ws; ++r) {
    int64_t r_start = r * experts_per_rank;
    for (int64_t e = 0; e < num_local; ++e) {
      send_expert_counts[r * num_local + e] = expert_counts_global[r_start + e];
    }
  }

  std::vector<int64_t> recv_expert_counts(ws * num_local, 0);
  {
    auto send_gpu = alloc_tensor(
      Shape({ws * num_local}), dl_int64,
      DLDevice{kDLCUDA, (int)hidden.device().device_id});
    auto recv_gpu = alloc_tensor(
      Shape({ws * num_local}), dl_int64,
      DLDevice{kDLCUDA, (int)hidden.device().device_id});

    cudaMemcpyAsync(send_gpu.data_ptr(), send_expert_counts.data(),
                    ws * num_local * sizeof(int64_t),
                    cudaMemcpyHostToDevice, stream);

    NCCL_CHECK(ncclGroupStart());
    for (int64_t r = 0; r < ws; ++r) {
      NCCL_CHECK(ncclSend(
        static_cast<int64_t*>(send_gpu.data_ptr()) + r * num_local,
        num_local, ncclInt64, (int)r, g.nccl_comm, stream));
      NCCL_CHECK(ncclRecv(
        static_cast<int64_t*>(recv_gpu.data_ptr()) + r * num_local,
        num_local, ncclInt64, (int)r, g.nccl_comm, stream));
    }
    NCCL_CHECK(ncclGroupEnd());

    cudaMemcpyAsync(recv_expert_counts.data(), recv_gpu.data_ptr(),
                    ws * num_local * sizeof(int64_t),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }

  // Compute per-rank send/recv totals and per-local-expert recv totals
  std::vector<int64_t> send_counts(ws, 0);
  std::vector<int64_t> recv_counts(ws, 0);
  std::vector<int64_t> local_expert_recv_counts(num_local, 0);

  for (int64_t r = 0; r < ws; ++r) {
    for (int64_t e = 0; e < num_local; ++e) {
      send_counts[r] += send_expert_counts[r * num_local + e];
      recv_counts[r] += recv_expert_counts[r * num_local + e];
      local_expert_recv_counts[e] += recv_expert_counts[r * num_local + e];
    }
  }

  int64_t total_send = std::accumulate(send_counts.begin(), send_counts.end(), (int64_t)0);
  int64_t total_recv = std::accumulate(recv_counts.begin(), recv_counts.end(), (int64_t)0);

  // --- 3. Build sorted send buffer, alltoall the hidden states ---

  // Compute dest_rank for each token-expert pair and argsort
  std::vector<int64_t> dest_ranks(num_tokens * top_k);
  for (int64_t i = 0; i < num_tokens * top_k; ++i) {
    dest_ranks[i] = host_idx[i] / experts_per_rank;
  }

  std::vector<int64_t> sort_indices(num_tokens * top_k);
  std::iota(sort_indices.begin(), sort_indices.end(), 0);
  // Sort by (dest_rank, expert_id) so that within each dest rank's chunk,
  // tokens are grouped by expert. This is required for EXPERT_MAJOR_3D
  // layout and ensures the receiver can split by expert using counts alone.
  std::stable_sort(sort_indices.begin(), sort_indices.end(),
            [&](int64_t a, int64_t b) {
              if (dest_ranks[a] != dest_ranks[b])
                return dest_ranks[a] < dest_ranks[b];
              return host_idx[a] < host_idx[b];  // sub-sort by expert id
            });

  // Upload sort_indices to GPU, use a gather kernel to build send buffer
  // For simplicity and correctness, do expand + gather on GPU

  // Expand hidden: [N, H] -> [N*top_k, H] (repeat each token top_k times)
  auto expanded = alloc_tensor(
    Shape({num_tokens * top_k, hidden_dim}), dtype,
    DLDevice{kDLCUDA, (int)hidden.device().device_id});

  // Simple copy — each token repeated top_k times, then reorder by sort_indices
  // We'll do this via a CUDA kernel for efficiency
  // But first: upload sort_indices
  auto sort_idx_gpu = alloc_tensor(
    Shape({num_tokens * top_k}), dl_int64,
    DLDevice{kDLCUDA, (int)hidden.device().device_id});
  cudaMemcpyAsync(sort_idx_gpu.data_ptr(), sort_indices.data(),
                  sort_indices.size() * sizeof(int64_t),
                  cudaMemcpyHostToDevice, stream);

  // Expand + gather kernel: send_buffer[i] = hidden[sort_indices[i] / top_k]
  // We'll use a generic byte-copy approach to handle any dtype
  {
    int64_t row_bytes = hidden_dim * elem_size;
    int64_t n = num_tokens * top_k;
    // Launch a simple kernel
    auto* src = static_cast<const char*>(hidden.data_ptr());
    auto* dst = static_cast<char*>(expanded.data_ptr());
    auto* idx = static_cast<const int64_t*>(sort_idx_gpu.data_ptr());

    // We can't easily launch a template kernel from here for all dtypes,
    // so use cudaMemcpy2D in a loop on host — but that's slow.
    // Better: upload sort_indices, do gather on GPU.
    // For now, do host-side gather (works correctly, optimize later).
    cudaStreamSynchronize(stream);

    // Host-side: build send_buffer by gathering from hidden
    std::vector<char> h_hidden(num_tokens * row_bytes);
    cudaMemcpy(h_hidden.data(), hidden.data_ptr(), h_hidden.size(), cudaMemcpyDeviceToHost);

    std::vector<char> h_send(total_send * row_bytes);
    for (int64_t i = 0; i < total_send; ++i) {
      int64_t orig_tok = sort_indices[i] / top_k;
      std::memcpy(&h_send[i * row_bytes], &h_hidden[orig_tok * row_bytes], row_bytes);
    }

    // Upload send buffer to GPU
    auto send_buf = alloc_tensor(
      Shape({total_send, hidden_dim}), dtype,
      DLDevice{kDLCUDA, (int)hidden.device().device_id});
    cudaMemcpyAsync(send_buf.data_ptr(), h_send.data(),
                    h_send.size(), cudaMemcpyHostToDevice, stream);

    // Alltoall the hidden states via NCCL P2P
    auto recv_buf = alloc_tensor(
      Shape({std::max(total_recv, (int64_t)1), hidden_dim}), dtype,
      DLDevice{kDLCUDA, (int)hidden.device().device_id});

    if (total_recv > 0 || total_send > 0) {
      NCCL_CHECK(ncclGroupStart());
      int64_t send_offset = 0;
      int64_t recv_offset = 0;
      for (int64_t r = 0; r < ws; ++r) {
        int64_t s_count = send_counts[r] * hidden_dim;
        int64_t r_count = recv_counts[r] * hidden_dim;
        if (s_count > 0) {
          NCCL_CHECK(ncclSend(
            static_cast<const char*>(send_buf.data_ptr()) + send_offset * elem_size,
            s_count, nccl_dtype, (int)r, g.nccl_comm, stream));
        }
        if (r_count > 0) {
          NCCL_CHECK(ncclRecv(
            static_cast<char*>(recv_buf.data_ptr()) + recv_offset * elem_size,
            r_count, nccl_dtype, (int)r, g.nccl_comm, stream));
        }
        send_offset += s_count;
        recv_offset += r_count;
      }
      NCCL_CHECK(ncclGroupEnd());
    }

    // --- 4. Compute local expert counts (from recv side = all ranks) ---

    auto expert_counts_out = alloc_tensor(
      Shape({num_local}), dl_int32, DLDevice{kDLCPU, 0});
    auto* ec_ptr = static_cast<int32_t*>(expert_counts_out.data_ptr());
    for (int64_t e = 0; e < num_local; ++e) {
      ec_ptr[e] = (int32_t)local_expert_recv_counts[e];
    }

    // --- 5. Store handle with routing metadata ---

    int64_t handle_id;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      handle_id = g_next_handle_id++;
      EpHandleState h;
      h.group_id = group_id;
      h.num_recv_tokens = total_recv;
      h.send_counts = send_counts;
      h.recv_counts = recv_counts;
      h.send_expert_counts = send_expert_counts;
      h.recv_expert_counts = recv_expert_counts;
      h.sort_indices = sort_indices;
      h.num_tokens = num_tokens;
      h.top_k = top_k;
      h.hidden_dim = hidden_dim;
      h.has_deferred = false;
      h.destroyed = false;
      g_handles[handle_id] = std::move(h);
    }

    // --- 6. Format output based on output_layout ---

    Tensor recv_hidden_out;
    if (output_layout == 1 && total_recv > 0) {
      // EXPERT_MAJOR_3D: [num_local_experts, max_tokens_per_expert, hidden_dim]
      // recv_buf is flat [total_recv, hidden_dim], ordered by source rank.
      // We need to scatter into [E, max_tok, H] based on expert assignment.

      // Compute max tokens per expert for padding
      int64_t max_tok = 0;
      for (int64_t e = 0; e < num_local; ++e) {
        if (ec_ptr[e] > max_tok) max_tok = ec_ptr[e];
      }
      if (max_tok == 0) max_tok = 1;  // avoid zero-size tensor

      // Allocate 3D output, zero-initialized for padding
      recv_hidden_out = alloc_tensor(
        Shape({num_local, max_tok, hidden_dim}), dtype,
        DLDevice{kDLCUDA, (int)hidden.device().device_id});
      cudaMemsetAsync(recv_hidden_out.data_ptr(), 0,
                      num_local * max_tok * hidden_dim * elem_size, stream);

      // recv_buf layout: tokens from rank 0 first, then rank 1, etc.
      // Within each rank's chunk, tokens are sub-sorted by expert ID
      // (because the sender sorted by (dest_rank, expert_id)).
      //
      // recv_expert_counts[r * num_local + e] tells us exactly how many
      // tokens from rank r go to local expert e. We use this to scatter
      // into the 3D [num_local_experts, max_tok, hidden_dim] layout.

      cudaStreamSynchronize(stream);

      int64_t row_bytes_3d = hidden_dim * elem_size;
      std::vector<char> h_recv(total_recv * row_bytes_3d);
      cudaMemcpy(h_recv.data(), recv_buf.data_ptr(), h_recv.size(),
                 cudaMemcpyDeviceToHost);

      // Per-expert write cursors for the 3D output
      std::vector<int64_t> expert_write_pos(num_local, 0);

      std::vector<char> h_3d(num_local * max_tok * row_bytes_3d, 0);
      int64_t src_offset = 0;
      for (int64_t r = 0; r < ws; ++r) {
        for (int64_t e = 0; e < num_local; ++e) {
          int64_t cnt = recv_expert_counts[r * num_local + e];
          for (int64_t t = 0; t < cnt; ++t) {
            int64_t dst_off = (e * max_tok + expert_write_pos[e]) * row_bytes_3d;
            int64_t src_off = src_offset * row_bytes_3d;
            std::memcpy(&h_3d[dst_off], &h_recv[src_off], row_bytes_3d);
            expert_write_pos[e]++;
            src_offset++;
          }
        }
      }

      cudaMemcpyAsync(recv_hidden_out.data_ptr(), h_3d.data(),
                      h_3d.size(), cudaMemcpyHostToDevice, stream);

    } else if (total_recv > 0) {
      // FLAT_2D: return as-is
      recv_hidden_out = recv_buf;
    } else {
      // Empty — no tokens on this rank
      if (output_layout == 1) {
        recv_hidden_out = alloc_tensor(
          Shape({num_local, 0, hidden_dim}), dtype,
          DLDevice{kDLCUDA, (int)hidden.device().device_id});
      } else {
        recv_hidden_out = alloc_tensor(
          Shape({0, hidden_dim}), dtype,
          DLDevice{kDLCUDA, (int)hidden.device().device_id});
      }
    }

    return Tuple<Tensor, Tensor, int64_t>(recv_hidden_out, expert_counts_out, handle_id);
  }
}

// ─── Combine (gather) — full C++ implementation ─────────────────────
//
// Reverses the dispatch alltoall: sends expert outputs back to originating
// ranks, then applies topk_weights weighted reduction.
//
// Returns: Tensor [num_tokens, hidden_dim] (combined output)

Tensor epCombine(
    TensorView expert_output,  // [total_recv, hidden_dim] or [E, max_tok, H]
    TensorView topk_weights,   // [num_tokens, top_k] float32
    int64_t handle_id)
{
  EpHandleState h;
  EpGroupState g;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    auto hit = g_handles.find(handle_id);
    if (hit == g_handles.end()) {
      throw std::invalid_argument(
          "ep_combine: invalid handle_id=" + std::to_string(handle_id) +
          ". Was the handle destroyed or never created?");
    }
    h = hit->second;
    if (h.destroyed) {
      throw std::invalid_argument(
          "ep_combine: handle_id=" + std::to_string(handle_id) +
          " has been destroyed");
    }
    auto git = g_groups.find(h.group_id);
    if (git == g_groups.end() || git->second.destroyed) {
      throw std::invalid_argument(
          "ep_combine: group_id=" + std::to_string(h.group_id) +
          " (from handle) is invalid or destroyed");
    }
    g = git->second;
    if (g.nccl_comm == nullptr) {
      throw std::runtime_error(
          "ep_combine: ncclComm_t is NULL for group_id=" +
          std::to_string(h.group_id));
    }
  }

  cudaStream_t stream = get_current_stream();
  int64_t ws = g.world_size;
  int64_t hidden_dim = h.hidden_dim;
  DLDataType dtype = expert_output.dtype();
  ncclDataType_t nccl_dtype = to_nccl_dtype(dtype);
  int elem_size = (dtype.bits * dtype.lanes + 7) / 8;

  // Combine direction is REVERSE of dispatch:
  //   send_counts (combine) = recv_counts (dispatch)
  //   recv_counts (combine) = send_counts (dispatch)
  auto& c_send_counts = h.recv_counts;   // what we received, we now send back
  auto& c_recv_counts = h.send_counts;   // what we sent, we now receive back

  int64_t total_c_send = std::accumulate(c_send_counts.begin(), c_send_counts.end(), (int64_t)0);
  int64_t total_c_recv = std::accumulate(c_recv_counts.begin(), c_recv_counts.end(), (int64_t)0);

  // Flatten expert_output if 3D
  const void* send_ptr = expert_output.data_ptr();
  int64_t send_rows;
  if (expert_output.ndim() == 3) {
    send_rows = expert_output.size(0) * expert_output.size(1);
  } else {
    send_rows = expert_output.size(0);
  }

  // Trim send to actual total_c_send
  // Alltoall expert outputs back
  auto recv_buf = alloc_tensor(
    Shape({std::max(total_c_recv, (int64_t)1), hidden_dim}), dtype,
    DLDevice{kDLCUDA, (int)expert_output.device().device_id});

  if (total_c_send > 0 || total_c_recv > 0) {
    NCCL_CHECK(ncclGroupStart());
    int64_t send_offset = 0;
    int64_t recv_offset = 0;
    for (int64_t r = 0; r < ws; ++r) {
      int64_t s_elems = c_send_counts[r] * hidden_dim;
      int64_t r_elems = c_recv_counts[r] * hidden_dim;
      if (s_elems > 0) {
        NCCL_CHECK(ncclSend(
          static_cast<const char*>(send_ptr) + send_offset * elem_size,
          s_elems, nccl_dtype, (int)r, g.nccl_comm, stream));
      }
      if (r_elems > 0) {
        NCCL_CHECK(ncclRecv(
          static_cast<char*>(recv_buf.data_ptr()) + recv_offset * elem_size,
          r_elems, nccl_dtype, (int)r, g.nccl_comm, stream));
      }
      send_offset += s_elems;
      recv_offset += r_elems;
    }
    NCCL_CHECK(ncclGroupEnd());
  }

  // Unsort + weighted reduce on host (D2H → compute → H2D)
  // recv_buf is sorted by dest rank. We need to unsort to original token order,
  // reshape to [num_tokens, top_k, hidden_dim], apply weights, sum.

  cudaStreamSynchronize(stream);

  int64_t num_tokens = h.num_tokens;
  int64_t top_k = h.top_k;
  int64_t row_bytes = hidden_dim * elem_size;

  std::vector<char> h_recv(total_c_recv * row_bytes);
  if (total_c_recv > 0) {
    cudaMemcpy(h_recv.data(), recv_buf.data_ptr(), h_recv.size(), cudaMemcpyDeviceToHost);
  }

  // Unsort
  std::vector<char> h_unsorted(total_c_recv * row_bytes);
  for (int64_t i = 0; i < total_c_recv; ++i) {
    int64_t orig_pos = h.sort_indices[i];
    std::memcpy(&h_unsorted[orig_pos * row_bytes], &h_recv[i * row_bytes], row_bytes);
  }

  // Copy topk_weights to host
  std::vector<float> h_weights(num_tokens * top_k);
  cudaMemcpy(h_weights.data(), topk_weights.data_ptr(),
             num_tokens * top_k * sizeof(float), cudaMemcpyDeviceToHost);

  // Weighted reduce: combined[t] = sum_k(weights[t,k] * unsorted[t*top_k+k])
  // Output as float32 for accumulation, then cast back to dtype
  std::vector<float> h_combined(num_tokens * hidden_dim, 0.0f);

  for (int64_t t = 0; t < num_tokens; ++t) {
    for (int64_t k = 0; k < top_k; ++k) {
      float w = h_weights[t * top_k + k];
      int64_t src_idx = t * top_k + k;
      const char* src_row = &h_unsorted[src_idx * row_bytes];

      for (int64_t d = 0; d < hidden_dim; ++d) {
        float val;
        if (dtype.code == kDLBfloat && dtype.bits == 16) {
          __nv_bfloat16 bf = reinterpret_cast<const __nv_bfloat16*>(src_row)[d];
          val = __bfloat162float(bf);
        } else if (dtype.code == kDLFloat && dtype.bits == 16) {
          __half hf = reinterpret_cast<const __half*>(src_row)[d];
          val = __half2float(hf);
        } else {
          val = reinterpret_cast<const float*>(src_row)[d];
        }
        h_combined[t * hidden_dim + d] += w * val;
      }
    }
  }

  // Convert back to output dtype and upload
  auto combined = alloc_tensor(
    Shape({num_tokens, hidden_dim}), dtype,
    DLDevice{kDLCUDA, (int)expert_output.device().device_id});

  if (dtype.code == kDLBfloat && dtype.bits == 16) {
    std::vector<__nv_bfloat16> h_out(num_tokens * hidden_dim);
    for (size_t i = 0; i < h_out.size(); ++i) {
      h_out[i] = __float2bfloat16(h_combined[i]);
    }
    cudaMemcpyAsync(combined.data_ptr(), h_out.data(),
                    h_out.size() * sizeof(__nv_bfloat16),
                    cudaMemcpyHostToDevice, stream);
  } else if (dtype.code == kDLFloat && dtype.bits == 16) {
    std::vector<__half> h_out(num_tokens * hidden_dim);
    for (size_t i = 0; i < h_out.size(); ++i) {
      h_out[i] = __float2half(h_combined[i]);
    }
    cudaMemcpyAsync(combined.data_ptr(), h_out.data(),
                    h_out.size() * sizeof(__half),
                    cudaMemcpyHostToDevice, stream);
  } else {
    cudaMemcpyAsync(combined.data_ptr(), h_combined.data(),
                    h_combined.size() * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
  }

  return combined;
}

// ─── Handle lifecycle ───────────────────────────────────────────────

int64_t epCreateHandle(int64_t group_id, int64_t num_recv_tokens, bool has_deferred) {
  std::lock_guard<std::mutex> lock(g_mutex);
  int64_t id = g_next_handle_id++;
  EpHandleState h;
  h.group_id = group_id;
  h.num_recv_tokens = num_recv_tokens;
  h.num_tokens = 0;
  h.top_k = 0;
  h.hidden_dim = 0;
  h.has_deferred = has_deferred;
  h.destroyed = false;
  g_handles[id] = std::move(h);
  return id;
}

void epDestroyHandle(int64_t handle_id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_handles.find(handle_id);
  if (it != g_handles.end()) {
    it->second.destroyed = true;
  }
}

int64_t epHandleGetNumRecv(int64_t handle_id) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_handles.find(handle_id);
  if (it == g_handles.end() || it->second.destroyed) return -1;
  return it->second.num_recv_tokens;
}

int64_t epHandleInvokeDeferred(int64_t handle_id) {
  cudaStreamSynchronize(nullptr);
  return 0;
}

// ─── Layout normalization kernels (2D <-> 3D) ───────────────────────

__global__ void scatter_2d_to_3d_kernel(
    const __nv_bfloat16* __restrict__ flat,
    __nv_bfloat16* __restrict__ scattered,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_counts,
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim) {
  int expert = blockIdx.x;
  if (expert >= num_local_experts) return;
  int count = expert_counts[expert];
  int offset = expert_offsets[expert];
  for (int t = threadIdx.x; t < count * hidden_dim; t += blockDim.x) {
    int tok = t / hidden_dim;
    int h = t % hidden_dim;
    scattered[expert * max_tokens_per_expert * hidden_dim + tok * hidden_dim + h] =
        flat[offset * hidden_dim + tok * hidden_dim + h];
  }
}

__global__ void gather_3d_to_2d_kernel(
    const __nv_bfloat16* __restrict__ scattered,
    __nv_bfloat16* __restrict__ flat,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ expert_counts,
    int num_local_experts,
    int max_tokens_per_expert,
    int hidden_dim) {
  int expert = blockIdx.x;
  if (expert >= num_local_experts) return;
  int count = expert_counts[expert];
  int offset = expert_offsets[expert];
  for (int t = threadIdx.x; t < count * hidden_dim; t += blockDim.x) {
    int tok = t / hidden_dim;
    int h = t % hidden_dim;
    flat[offset * hidden_dim + tok * hidden_dim + h] =
        scattered[expert * max_tokens_per_expert * hidden_dim + tok * hidden_dim + h];
  }
}

void epLayoutScatter2dTo3d(
    TensorView flat, TensorView scattered,
    TensorView expert_offsets, TensorView expert_counts,
    int64_t num_local_experts, int64_t max_tokens_per_expert, int64_t hidden_dim) {
  scatter_2d_to_3d_kernel<<<num_local_experts, 256, 0, get_current_stream()>>>(
      static_cast<const __nv_bfloat16*>(flat.data_ptr()),
      static_cast<__nv_bfloat16*>(scattered.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<const int32_t*>(expert_counts.data_ptr()),
      (int)num_local_experts, (int)max_tokens_per_expert, (int)hidden_dim);
}

void epLayoutGather3dTo2d(
    TensorView scattered, TensorView flat,
    TensorView expert_offsets, TensorView expert_counts,
    int64_t num_local_experts, int64_t max_tokens_per_expert, int64_t hidden_dim) {
  gather_3d_to_2d_kernel<<<num_local_experts, 256, 0, get_current_stream()>>>(
      static_cast<const __nv_bfloat16*>(scattered.data_ptr()),
      static_cast<__nv_bfloat16*>(flat.data_ptr()),
      static_cast<const int32_t*>(expert_offsets.data_ptr()),
      static_cast<const int32_t*>(expert_counts.data_ptr()),
      (int)num_local_experts, (int)max_tokens_per_expert, (int)hidden_dim);
}

}  // anonymous namespace

// ─── TVM FFI exports ────────────────────────────────────────────────

TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_create_group, epCreateGroup);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_destroy_group, epDestroyGroup);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_get_dispatch_layout, epGetDispatchLayout);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_dispatch, epDispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_combine, epCombine);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_create_handle, epCreateHandle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_destroy_handle, epDestroyHandle);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_handle_get_num_recv, epHandleGetNumRecv);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_handle_invoke_deferred, epHandleInvokeDeferred);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_layout_scatter_2d_to_3d, epLayoutScatter2dTo3d);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(ep_layout_gather_3d_to_2d, epLayoutGather3dTo2d);
