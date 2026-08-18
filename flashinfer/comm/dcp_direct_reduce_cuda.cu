// Two-kernel destination-owned DCP Output/LSE reduce.
// K1: per-dest publish of all tokens, then system-scope release signal.
// K2: wait + stable LSE combine into caller/workspace output.

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace {

constexpr uint64_t kSpinLimit = 100000000ULL;

__device__ __forceinline__ void store_release_sys(uint32_t* ptr, uint32_t value) {
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("st.global.release.sys.u32 [%0], %1;" ::"l"(address), "r"(value) : "memory");
}

__device__ __forceinline__ uint32_t load_acquire_sys(const uint32_t* ptr) {
  uint32_t value;
  uint64_t address = reinterpret_cast<uint64_t>(ptr);
  asm volatile("ld.global.acquire.sys.u32 %0, [%1];" : "=r"(value) : "l"(address) : "memory");
  return value;
}

template <typename T>
__device__ __forceinline__ float to_f32(T v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __half2float(v);
  } else {
    return __bfloat162float(v);
  }
}

template <typename T>
__device__ __forceinline__ T from_f32(float v) {
  if constexpr (std::is_same_v<T, __half>) {
    return __float2half_rn(v);
  } else {
    return __float2bfloat16_rn(v);
  }
}

// One block per (token, dest). Last writer for each dest publishes the flag.
template <typename T>
__global__ void publish_signal_kernel(
    const T* __restrict__ partial_o, const float* __restrict__ partial_lse,
    const int64_t* __restrict__ peer_out_ptrs, const int64_t* __restrict__ peer_lse_ptrs,
    const int64_t* __restrict__ peer_sig_ptrs, int32_t* __restrict__ epoch_ptr,
    int32_t* __restrict__ dest_done, int rank, int world, int num_tokens, int max_tokens,
    int h_local, int d, int64_t stride_po_tok, int64_t stride_po_head, int64_t stride_pl_tok) {
  const int tok = blockIdx.x;
  const int dest = blockIdx.y;
  const int32_t next = epoch_ptr[0] + 1;
  const int parity = next & 1;
  const int n_items = h_local * d;
  const int src_head0 = dest * h_local;
  T* peer_o = reinterpret_cast<T*>(static_cast<uintptr_t>(peer_out_ptrs[dest]));
  float* peer_lse = reinterpret_cast<float*>(static_cast<uintptr_t>(peer_lse_ptrs[dest]));
  T* dst = peer_o + (static_cast<int64_t>(parity) * world + rank) * max_tokens * n_items +
           static_cast<int64_t>(tok) * n_items;
  const T* src_base = partial_o + tok * stride_po_tok + src_head0 * stride_po_head;
  const bool aligned16 = ((reinterpret_cast<uintptr_t>(src_base) & 15) == 0) &&
                         ((reinterpret_cast<uintptr_t>(dst) & 15) == 0);
  const bool packed16 = (stride_po_head == d) && (sizeof(T) == 2) && ((n_items & 7) == 0) &&
                        ((stride_po_tok & 7) == 0) && aligned16;
  if (packed16) {
    const uint4* src = reinterpret_cast<const uint4*>(src_base);
    uint4* d4 = reinterpret_cast<uint4*>(dst);
    const int nvec = n_items >> 3;
    for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
      d4[i] = src[i];
    }
  } else {
    for (int i = threadIdx.x; i < n_items; i += blockDim.x) {
      const int local_head = i / d;
      const int dim = i - local_head * d;
      dst[i] = partial_o[tok * stride_po_tok + (src_head0 + local_head) * stride_po_head + dim];
    }
  }
  float* lse_dst = peer_lse + (static_cast<int64_t>(parity) * world + rank) * max_tokens * h_local +
                   static_cast<int64_t>(tok) * h_local;
  const float* lse_src = partial_lse + tok * stride_pl_tok + src_head0;
  for (int h = threadIdx.x; h < h_local; h += blockDim.x) {
    lse_dst[h] = lse_src[h];
  }

  __syncthreads();
  __threadfence_system();
  if (threadIdx.x == 0) {
    const int finished = atomicAdd(dest_done + dest, 1);
    if (finished == num_tokens - 1) {
      uint32_t* sig = reinterpret_cast<uint32_t*>(static_cast<uintptr_t>(peer_sig_ptrs[dest]));
      store_release_sys(sig + parity * world + rank, static_cast<uint32_t>(next));
      dest_done[dest] = 0;
    }
  }
}

// Advance after every publish block has sampled the same epoch.
__global__ void advance_epoch_kernel(int32_t* epoch_ptr) {
  if (threadIdx.x == 0) {
    epoch_ptr[0] += 1;
  }
}

// One block per (token, local_head). Thread 0 waits and reduces LSE.
template <typename T>
__global__ void merge_kernel(const T* __restrict__ recv_o, const float* __restrict__ recv_lse,
                             const int32_t* __restrict__ recv_sig,
                             const int32_t* __restrict__ epoch_ptr, T* __restrict__ out,
                             float* __restrict__ lse_out, int world, int num_tokens, int max_tokens,
                             int h_local, int d, int64_t stride_out_tok, int64_t stride_out_head,
                             int64_t stride_lse_tok, int is_base_e) {
  extern __shared__ float weights[];
  const int item = blockIdx.x;
  const int token = item / h_local;
  const int head = item - token * h_local;
  const int32_t epoch = epoch_ptr[0];
  const int parity = epoch & 1;
  const int n_items = h_local * d;

  if (threadIdx.x == 0) {
    float lse_max = -INFINITY;
    uint64_t spins = 0;
    while (true) {
      int pending = 0;
      for (int src = 0; src < world; ++src) {
        const int32_t* sig = recv_sig + parity * world + src;
        if (load_acquire_sys(reinterpret_cast<const uint32_t*>(sig)) !=
            static_cast<uint32_t>(epoch)) {
          pending = 1;
        }
      }
      if (pending == 0) {
        break;
      }
      if (++spins >= kSpinLimit) {
        asm volatile("trap;");
      }
    }
    for (int src = 0; src < world; ++src) {
      const int64_t lse_off =
          ((static_cast<int64_t>(parity) * world + src) * max_tokens + token) * h_local + head;
      float value = recv_lse[lse_off];
      if (isnan(value) || value == INFINITY) {
        value = -INFINITY;
      }
      weights[src] = value;
      lse_max = fmaxf(lse_max, value);
    }
    const float m_math = (lse_max == -INFINITY) ? 0.f : lse_max;
    float sum_w = 0.f;
    for (int src = 0; src < world; ++src) {
      const float w = is_base_e ? expf(weights[src] - m_math) : exp2f(weights[src] - m_math);
      weights[src] = w;
      sum_w += w;
    }
    float final_lse = -INFINITY;
    if (sum_w > 0.f) {
      const float inv = 1.f / sum_w;
      for (int src = 0; src < world; ++src) {
        weights[src] *= inv;
      }
      final_lse = (is_base_e ? logf(sum_w) : log2f(sum_w)) + m_math;
    } else {
      for (int src = 0; src < world; ++src) {
        weights[src] = 0.f;
      }
    }
    lse_out[token * stride_lse_tok + head] = final_lse;
  }
  __syncthreads();

  for (int dim = threadIdx.x; dim < d; dim += blockDim.x) {
    float acc = 0.f;
    for (int src = 0; src < world; ++src) {
      const float nw = weights[src];
      const int64_t off =
          ((static_cast<int64_t>(parity) * world + src) * max_tokens + token) * n_items + head * d +
          dim;
      float part = to_f32(recv_o[off]);
      if (nw == 0.f) {
        part = 0.f;
      }
      acc += part * nw;
    }
    out[token * stride_out_tok + head * stride_out_head + dim] = from_f32<T>(acc);
  }
}

void launch(torch::Tensor partial_o, torch::Tensor partial_lse, torch::Tensor peer_out_ptrs,
            torch::Tensor peer_lse_ptrs, torch::Tensor peer_sig_ptrs, torch::Tensor recv_o,
            torch::Tensor recv_lse, torch::Tensor recv_sig, torch::Tensor epoch,
            torch::Tensor dest_done, torch::Tensor out, torch::Tensor lse_out, int64_t world,
            int64_t rank, int64_t max_tokens, int64_t is_base_e) {
  TORCH_CHECK(partial_o.is_cuda() && out.is_cuda());
  const int num_tokens = static_cast<int>(partial_o.size(0));
  const int h_total = static_cast<int>(partial_o.size(1));
  const int d = static_cast<int>(partial_o.size(2));
  const int h_local = h_total / static_cast<int>(world);
  auto stream = at::cuda::getCurrentCUDAStream();
  const dim3 pub_grid(static_cast<unsigned>(num_tokens), static_cast<unsigned>(world));
  constexpr int kPubThreads = 256;
  constexpr int kMergeThreads = 128;

  auto launch_typed = [&](auto dummy) {
    using T = decltype(dummy);
    publish_signal_kernel<T><<<pub_grid, kPubThreads, 0, stream>>>(
        reinterpret_cast<const T*>(partial_o.data_ptr()), partial_lse.data_ptr<float>(),
        peer_out_ptrs.data_ptr<int64_t>(), peer_lse_ptrs.data_ptr<int64_t>(),
        peer_sig_ptrs.data_ptr<int64_t>(), epoch.data_ptr<int32_t>(), dest_done.data_ptr<int32_t>(),
        static_cast<int>(rank), static_cast<int>(world), num_tokens, static_cast<int>(max_tokens),
        h_local, d, partial_o.stride(0), partial_o.stride(1), partial_lse.stride(0));
    advance_epoch_kernel<<<1, 1, 0, stream>>>(epoch.data_ptr<int32_t>());
    const dim3 merge_grid(static_cast<unsigned>(num_tokens * h_local));
    merge_kernel<T><<<merge_grid, kMergeThreads, sizeof(float) * world, stream>>>(
        reinterpret_cast<const T*>(recv_o.data_ptr()), recv_lse.data_ptr<float>(),
        recv_sig.data_ptr<int32_t>(), epoch.data_ptr<int32_t>(),
        reinterpret_cast<T*>(out.data_ptr()), lse_out.data_ptr<float>(), static_cast<int>(world),
        num_tokens, static_cast<int>(max_tokens), h_local, d, out.stride(0), out.stride(1),
        lse_out.stride(0), static_cast<int>(is_base_e));
  };

  if (partial_o.scalar_type() == at::kBFloat16) {
    launch_typed(__nv_bfloat16{});
  } else if (partial_o.scalar_type() == at::kHalf) {
    launch_typed(__half{});
  } else {
    TORCH_CHECK(false, "dcp_direct_reduce cuda supports fp16/bf16 only");
  }
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("launch", &launch); }
