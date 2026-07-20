// Ported from b12x b12x/distributed/pcie_dcp_a2a.cu @ a394fd2e (2026-07-04) -- one-time curated port.
// PCIe one-shot exchange for DCP attention output and LSE reduction.
//
// Each rank first copies its full partial output and FP32 LSE into one of two
// IPC-visible staging slots. A single system-scope barrier makes those copies
// visible, after which every rank pulls only its destination head shard from
// all peers and performs the stable LSE-weighted reduction while storing the
// final output. This deliberately follows the low-latency, one-barrier design
// of pcie_oneshot.cu rather than implementing a generic NCCL-style all-to-all.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                                \
  do {                                                                         \
    cudaError_t e = cmd;                                                       \
    if (e != cudaSuccess) {                                                    \
      std::stringstream _message;                                              \
      _message << cudaGetErrorString(e) << "\n"                                \
               << __FILE__ << ':' << __LINE__;                                 \
      throw std::runtime_error(_message.str());                                \
    }                                                                          \
  } while (0)

namespace pcie_dcp_a2a {

constexpr int kMaxBlocks = 64;
constexpr int kMaxRanks = 8;
constexpr int kFlagStride = 32;
using FlagType = uint32_t;

static int env_int(const char *name, int fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return fallback;
  }
  return std::atoi(raw);
}

// Launch-shape overrides for latency sweeps. Every rank must use identical
// values: warp geometry decides which rows each block stages, and peer
// block b only reads what writer block b staged.
static int dcp_threads_override() {
  static const int value = env_int("FLASHINFER_EXP_SM12X_PCIE_DCP_THREADS", 0);
  return value;
}

static int dcp_block_limit_override() {
  static const int value = env_int("FLASHINFER_EXP_SM12X_PCIE_DCP_BLOCK_LIMIT", 0);
  return value;
}

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
};

struct RankSignals {
  Signal *signals[kMaxRanks];
};

struct RankStaging {
  void *ptrs[kMaxRanks];
};

template <typename T> struct __align__(16) Pack {
  T values[8];
};

#define DINLINE __device__ __forceinline__

static DINLINE void store_flag(FlagType *address, FlagType value) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(value),
               "l"(address));
}

static DINLINE FlagType load_flag(FlagType *address) {
  FlagType value;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];"
               : "=r"(value)
               : "l"(address));
  return value;
}

// pre_sync orders in-kernel staging stores (from every warp of the block)
// before the flag post; without staging the flags can go out immediately.
template <int world_size, bool pre_sync>
DINLINE void start_barrier(const RankSignals &signals, Signal *self, int rank) {
  if constexpr (pre_sync) __syncthreads();
  if (threadIdx.x < world_size) {
    __threadfence_system();
    const auto value = self->self_counter[blockIdx.x][threadIdx.x] +=
        FlagType{1};
    auto *peer = &signals.signals[threadIdx.x]
                      ->peer_counter[value % 2][blockIdx.x][rank * kFlagStride];
    auto *mine =
        &self->peer_counter[value % 2][blockIdx.x][threadIdx.x * kFlagStride];
    store_flag(peer, value);
    while (load_flag(mine) != value) {
    }
  }
  __syncthreads();
}

DINLINE float to_float(half value) { return __half2float(value); }
DINLINE float to_float(nv_bfloat16 value) { return __bfloat162float(value); }

template <typename T> DINLINE T from_float(float value);

template <> DINLINE half from_float<half>(float value) {
  return __float2half(value);
}

template <> DINLINE nv_bfloat16 from_float<nv_bfloat16>(float value) {
  return __float2bfloat16(value);
}

DINLINE float sanitize_lse(float value) {
  return isfinite(value) ? value : -CUDART_INF_F;
}

// Warp-per-row LSE-weighted reduction with in-kernel staging.
//
// Reader rows are the (batch, local_head) pairs of this rank's head shard,
// and every rank strides that identical row space with identical warp
// geometry. The warp that owns reader row (b, h) also stages this rank's
// contributions for (b, h) across all destination shards (packs and LSE),
// so the block-pairwise barrier covers exactly what the matching reader
// blocks pull. Each warp reads the LSE values once per row (one lane per
// source, packs_per_head times fewer remote scalar reads than per-pack
// loads), builds the softmax weights via shuffles in rotated source order,
// and streams the row's packs with lane-parallel pulls.
template <typename T, int world_size>
__global__ void __launch_bounds__(512, 1)
    dcp_lse_reduce_kernel(const T *__restrict__ local_output,
                          const float *__restrict__ local_lse,
                          RankStaging staging, int64_t lse_offset,
                          RankSignals signals, Signal *self,
                          T *__restrict__ output, int rank, int batch,
                          int total_heads, int head_dim, bool natural_log) {
  constexpr int kPackElems = 8;
  const int heads_per_rank = total_heads / world_size;
  const int packs_per_head = head_dim / kPackElems;
  const int rows = batch * heads_per_rank;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_first = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * warps_per_block;

  const auto *local_packs = reinterpret_cast<const Pack<T> *>(local_output);
  auto *output_packs = reinterpret_cast<Pack<T> *>(output);

  auto *staging_out = reinterpret_cast<Pack<T> *>(staging.ptrs[rank]);
  auto *staging_lse = reinterpret_cast<float *>(
      reinterpret_cast<char *>(staging.ptrs[rank]) + lse_offset);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / heads_per_rank;
    const int local_head = row - batch_index * heads_per_rank;
#pragma unroll
    for (int dest = 0; dest < world_size; ++dest) {
      const int64_t source_row = int64_t(batch_index) * total_heads +
                                 dest * heads_per_rank + local_head;
      const int64_t base = source_row * packs_per_head;
      for (int pack = lane; pack < packs_per_head; pack += warpSize) {
        staging_out[base + pack] = local_packs[base + pack];
      }
      if (lane == 0) {
        staging_lse[source_row] = local_lse[source_row];
      }
    }
  }
  start_barrier<world_size, true>(signals, self, rank);

  // Rotated source pointers so every later access uses a compile-time
  // index; the self source reads the local tensors directly (still hot in
  // L2 from staging).
  const Pack<T> *rot_packs[world_size];
#pragma unroll
  for (int i = 0; i < world_size; ++i) {
    const int src = (rank + i) % world_size;
    rot_packs[i] = src == rank
                       ? local_packs
                       : reinterpret_cast<const Pack<T> *>(staging.ptrs[src]);
  }

  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / heads_per_rank;
    const int local_head = row - batch_index * heads_per_rank;
    const int global_head = rank * heads_per_rank + local_head;
    const int64_t source_row = int64_t(batch_index) * total_heads + global_head;

    // Lane i pulls rotated source i's LSE; shuffles then give every lane
    // all sources in rotated order. The pointer is derived per lane from
    // the kernel param block to avoid a runtime-indexed local array.
    float lane_lse = -CUDART_INF_F;
    if (lane < world_size) {
      const int src = (rank + lane) % world_size;
      const float *lse_ptr =
          src == rank ? local_lse
                      : reinterpret_cast<const float *>(
                            reinterpret_cast<const char *>(staging.ptrs[src]) +
                            lse_offset);
      lane_lse = sanitize_lse(lse_ptr[source_row]);
    }
    float weights[world_size];
    float max_lse = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < world_size; ++i) {
      const float value = __shfl_sync(0xffffffff, lane_lse, i);
      weights[i] = value;
      max_lse = fmaxf(max_lse, value);
    }
    if (!isfinite(max_lse)) {
      max_lse = 0.0f;
    }
    float weight_sum = 0.0f;
#pragma unroll
    for (int i = 0; i < world_size; ++i) {
      const float delta = weights[i] - max_lse;
      const float weight = isfinite(weights[i])
                               ? (natural_log ? expf(delta) : exp2f(delta))
                               : 0.0f;
      weights[i] = weight;
      weight_sum += weight;
    }
    const float inv_weight_sum = 1.0f / fmaxf(weight_sum, 1.0e-10f);

    const int64_t source_base = source_row * packs_per_head;
    const int64_t output_base = int64_t(row) * packs_per_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      float accum[kPackElems] = {};
#pragma unroll
      for (int i = 0; i < world_size; ++i) {
        const Pack<T> values = rot_packs[i][source_base + pack];
        const float weight = weights[i] * inv_weight_sum;
#pragma unroll
        for (int element = 0; element < kPackElems; ++element) {
          accum[element] += weight * to_float(values.values[element]);
        }
      }
      Pack<T> result;
#pragma unroll
      for (int element = 0; element < kPackElems; ++element) {
        result.values[element] = from_float<T>(accum[element]);
      }
      output_packs[output_base + pack] = result;
    }
  }
}

// Warp-per-output-row head gather with in-kernel staging. Every rank
// strides the identical output row space with identical warp geometry, so
// the writer warp that owns row (batch, global_head) with source == this
// rank stages exactly the packs the matching reader blocks pull. Row-major
// warps also hoist the head/source arithmetic out of the pack loop (the
// pack-strided version paid six integer divisions per 16B pack).
template <typename T, int world_size>
__global__ void __launch_bounds__(512, 1)
    all_gather_heads_kernel(const T *__restrict__ local_input,
                            RankStaging staging, RankSignals signals,
                            Signal *self, T *__restrict__ output, int rank,
                            int batch, int local_heads, int head_dim) {
  constexpr int kPackElems = 8;
  const int packs_per_head = head_dim / kPackElems;
  const int total_heads = local_heads * world_size;
  const int rows = batch * total_heads;
  const int lane = threadIdx.x & 31;
  const int warps_per_block = blockDim.x >> 5;
  const int warp_first = blockIdx.x * warps_per_block + (threadIdx.x >> 5);
  const int warp_stride = gridDim.x * warps_per_block;
  const auto *local_packs = reinterpret_cast<const Pack<T> *>(local_input);

  auto *staging_out = reinterpret_cast<Pack<T> *>(staging.ptrs[rank]);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / total_heads;
    const int global_head = row - batch_index * total_heads;
    const int source_rank = global_head / local_heads;
    if (source_rank != rank) continue;
    const int local_head = global_head - source_rank * local_heads;
    const int64_t base =
        (int64_t(batch_index) * local_heads + local_head) * packs_per_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      staging_out[base + pack] = local_packs[base + pack];
    }
  }
  start_barrier<world_size, true>(signals, self, rank);

  auto *output_packs = reinterpret_cast<Pack<T> *>(output);
  for (int row = warp_first; row < rows; row += warp_stride) {
    const int batch_index = row / total_heads;
    const int global_head = row - batch_index * total_heads;
    const int source_rank = global_head / local_heads;
    const int local_head = global_head - source_rank * local_heads;
    const auto *source_packs =
        source_rank == rank
            ? local_packs
            : reinterpret_cast<const Pack<T> *>(staging.ptrs[source_rank]);
    const int64_t source_base =
        (int64_t(batch_index) * local_heads + local_head) * packs_per_head;
    const int64_t output_base = int64_t(row) * packs_per_head;
    for (int pack = lane; pack < packs_per_head; pack += warpSize) {
      output_packs[output_base + pack] = source_packs[source_base + pack];
    }
  }
}

class PCIeDCPA2A {
public:
  int rank_;
  int world_size_;
  RankSignals signals_{};
  Signal *self_signal_;
  RankStaging staging_[2]{};
  int64_t output_capacity_elems_;
  int64_t lse_offset_;
  int64_t lse_capacity_;
  int slot_ = 0;

  PCIeDCPA2A(Signal **signals,
             const std::vector<std::array<void *, 2>> &staging,
             int64_t output_capacity_elems, int64_t lse_offset,
             int64_t lse_capacity, int rank, int world_size)
      : rank_(rank), world_size_(world_size), self_signal_(signals[rank]),
        output_capacity_elems_(output_capacity_elems), lse_offset_(lse_offset),
        lse_capacity_(lse_capacity) {
    for (int peer = 0; peer < world_size_; ++peer) {
      signals_.signals[peer] = signals[peer];
      staging_[0].ptrs[peer] = staging[peer][0];
      staging_[1].ptrs[peer] = staging[peer][1];
    }
  }

  template <typename T>
  void run(cudaStream_t stream, const T *partial_output,
           const float *partial_lse, T *output, int batch, int total_heads,
           int head_dim, bool natural_log, int threads, int block_limit) {
    const int64_t output_elems = int64_t(batch) * total_heads * head_dim;
    const int64_t lse_elems = int64_t(batch) * total_heads;
    if (output_elems > output_capacity_elems_ || lse_elems > lse_capacity_) {
      throw std::runtime_error("PCIe DCP A2A staging capacity exceeded");
    }
    if (head_dim % 8 != 0) {
      throw std::runtime_error("head_dim must be a multiple of 8");
    }
    if (total_heads % world_size_ != 0) {
      throw std::runtime_error("total_heads must be divisible by world size");
    }
    if (threads < world_size_ || threads > 1024) {
      throw std::runtime_error("invalid thread count");
    }
    if (block_limit <= 0 || block_limit > kMaxBlocks) {
      throw std::runtime_error("invalid block limit");
    }

    if (const int env_threads = dcp_threads_override(); env_threads > 0) {
      threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    }
    if (const int env_blocks = dcp_block_limit_override(); env_blocks > 0) {
      block_limit = std::min(env_blocks, kMaxBlocks);
    }
    if (threads % 32 != 0) {
      throw std::runtime_error("threads must be a multiple of 32");
    }

    // Staging happens inside the kernel (warp-per-row, before the start
    // barrier); no host staging memcpys are issued.
    const int slot = slot_++ % 2;
    const int heads_per_rank = total_heads / world_size_;
    const int rows = batch * heads_per_rank;
    const int warps_per_block = threads / 32;
    const int blocks = std::max(
        1, std::min(block_limit, (rows + warps_per_block - 1) / warps_per_block));

#define LAUNCH(world)                                                          \
  dcp_lse_reduce_kernel<T, world><<<blocks, threads, 0, stream>>>(             \
      partial_output, partial_lse, staging_[slot], lse_offset_, signals_,      \
      self_signal_, output, rank_, batch, total_heads, head_dim, natural_log)
    switch (world_size_) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    default:
      throw std::runtime_error("PCIe DCP A2A supports 2, 4, or 8 ranks");
    }
#undef LAUNCH
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }

  template <typename T>
  void all_gather_heads(cudaStream_t stream, const T *local_input, T *output,
                        int batch, int local_heads, int head_dim, int threads,
                        int block_limit) {
    const int total_heads = local_heads * world_size_;
    const int64_t output_elems = int64_t(batch) * total_heads * head_dim;
    if (output_elems > output_capacity_elems_) {
      throw std::runtime_error("PCIe DCP all-gather staging capacity exceeded");
    }
    if (head_dim % 8 != 0) {
      throw std::runtime_error("head_dim must be a multiple of 8");
    }
    if (threads < world_size_ || threads > 1024) {
      throw std::runtime_error("invalid thread count");
    }
    if (block_limit <= 0 || block_limit > kMaxBlocks) {
      throw std::runtime_error("invalid block limit");
    }

    if (const int env_threads = dcp_threads_override(); env_threads > 0) {
      threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    }
    if (const int env_blocks = dcp_block_limit_override(); env_blocks > 0) {
      block_limit = std::min(env_blocks, kMaxBlocks);
    }
    if (threads % 32 != 0) {
      throw std::runtime_error("threads must be a multiple of 32");
    }

    // Staging happens inside the kernel (warp-per-row, before the start
    // barrier); no host staging memcpy is issued.
    const int slot = slot_++ % 2;
    const int rows = batch * total_heads;
    const int warps_per_block = threads / 32;
    const int blocks = std::max(
        1, std::min(block_limit, (rows + warps_per_block - 1) / warps_per_block));

#define LAUNCH(world)                                                          \
  all_gather_heads_kernel<T, world><<<blocks, threads, 0, stream>>>(           \
      local_input, staging_[slot], signals_, self_signal_, output, rank_,      \
      batch, local_heads, head_dim)
    switch (world_size_) {
    case 2:
      LAUNCH(2);
      break;
    case 4:
      LAUNCH(4);
      break;
    case 8:
      LAUNCH(8);
      break;
    default:
      throw std::runtime_error("PCIe DCP all-gather supports 2, 4, or 8 ranks");
    }
#undef LAUNCH
    CHECK_CUDA_SUCCESS(cudaGetLastError());
  }
};

} // namespace pcie_dcp_a2a

using fptr_t = int64_t;

static fptr_t init_dcp_a2a(const std::vector<fptr_t> &signal_ptrs,
                           const std::vector<fptr_t> &staging0_ptrs,
                           const std::vector<fptr_t> &staging1_ptrs,
                           int64_t output_capacity_elems, int64_t lse_offset,
                           int64_t lse_capacity, int64_t rank) {
  const int world_size = signal_ptrs.size();
  TORCH_CHECK(world_size == 2 || world_size == 4 || world_size == 8);
  TORCH_CHECK_EQ(staging0_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK_EQ(staging1_ptrs.size(), signal_ptrs.size());
  TORCH_CHECK(rank >= 0 && rank < world_size);

  pcie_dcp_a2a::Signal *signals[pcie_dcp_a2a::kMaxRanks];
  std::vector<std::array<void *, 2>> staging(world_size);
  for (int peer = 0; peer < world_size; ++peer) {
    signals[peer] = reinterpret_cast<pcie_dcp_a2a::Signal *>(signal_ptrs[peer]);
    staging[peer] = {reinterpret_cast<void *>(staging0_ptrs[peer]),
                     reinterpret_cast<void *>(staging1_ptrs[peer])};
  }
  return reinterpret_cast<fptr_t>(
      new pcie_dcp_a2a::PCIeDCPA2A(signals, staging, output_capacity_elems,
                                   lse_offset, lse_capacity, rank, world_size));
}

static void lse_reduce_scatter(fptr_t pointer, torch::Tensor &partial_output,
                               torch::Tensor &partial_lse,
                               torch::Tensor &output, bool natural_log,
                               int64_t threads, int64_t block_limit) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(partial_output));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(partial_output.is_cuda() && partial_lse.is_cuda() &&
              output.is_cuda());
  TORCH_CHECK(partial_output.is_contiguous() && partial_lse.is_contiguous() &&
              output.is_contiguous());
  TORCH_CHECK_EQ(partial_output.dim(), 3);
  TORCH_CHECK_EQ(partial_lse.dim(), 2);
  TORCH_CHECK_EQ(output.dim(), 3);
  TORCH_CHECK_EQ(partial_lse.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(partial_output.scalar_type(), output.scalar_type());

  const int64_t batch = partial_output.size(0);
  const int64_t total_heads = partial_output.size(1);
  const int64_t head_dim = partial_output.size(2);
  TORCH_CHECK_GT(batch, 0);
  TORCH_CHECK_EQ(partial_lse.size(0), batch);
  TORCH_CHECK_EQ(partial_lse.size(1), total_heads);
  TORCH_CHECK_EQ(total_heads % runtime->world_size_, 0);
  TORCH_CHECK_EQ(output.size(0), batch);
  TORCH_CHECK_EQ(output.size(1), total_heads / runtime->world_size_);
  TORCH_CHECK_EQ(output.size(2), head_dim);

  switch (partial_output.scalar_type()) {
  case at::ScalarType::Half:
    runtime->run(stream,
                 reinterpret_cast<const half *>(partial_output.data_ptr()),
                 reinterpret_cast<const float *>(partial_lse.data_ptr()),
                 reinterpret_cast<half *>(output.data_ptr()), int(batch),
                 int(total_heads), int(head_dim), natural_log, int(threads),
                 int(block_limit));
    break;
  case at::ScalarType::BFloat16:
    runtime->run(
        stream,
        reinterpret_cast<const nv_bfloat16 *>(partial_output.data_ptr()),
        reinterpret_cast<const float *>(partial_lse.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(output.data_ptr()), int(batch),
        int(total_heads), int(head_dim), natural_log, int(threads),
        int(block_limit));
    break;
  default:
    TORCH_CHECK(false, "partial_output must be float16 or bfloat16");
  }
}

static void all_gather_heads(fptr_t pointer, torch::Tensor &local_input,
                             torch::Tensor &output, int64_t threads,
                             int64_t block_limit) {
  auto *runtime = reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(local_input));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK(local_input.is_cuda() && output.is_cuda());
  TORCH_CHECK(local_input.is_contiguous() && output.is_contiguous());
  TORCH_CHECK_EQ(local_input.dim(), 3);
  TORCH_CHECK_EQ(output.dim(), 3);
  TORCH_CHECK_EQ(local_input.scalar_type(), output.scalar_type());

  const int64_t batch = local_input.size(0);
  const int64_t local_heads = local_input.size(1);
  const int64_t head_dim = local_input.size(2);
  TORCH_CHECK_GT(batch, 0);
  TORCH_CHECK_GT(local_heads, 0);
  TORCH_CHECK_EQ(output.size(0), batch);
  TORCH_CHECK_EQ(output.size(1), local_heads * runtime->world_size_);
  TORCH_CHECK_EQ(output.size(2), head_dim);

  switch (local_input.scalar_type()) {
  case at::ScalarType::Half:
    runtime->all_gather_heads(
        stream, reinterpret_cast<const half *>(local_input.data_ptr()),
        reinterpret_cast<half *>(output.data_ptr()), int(batch),
        int(local_heads), int(head_dim), int(threads), int(block_limit));
    break;
  case at::ScalarType::BFloat16:
    runtime->all_gather_heads(
        stream,
        reinterpret_cast<const nv_bfloat16 *>(local_input.data_ptr()),
        reinterpret_cast<nv_bfloat16 *>(output.data_ptr()), int(batch),
        int(local_heads), int(head_dim), int(threads), int(block_limit));
    break;
  default:
    TORCH_CHECK(false, "local_input must be float16 or bfloat16");
  }
}

static void dispose(fptr_t pointer) {
  delete reinterpret_cast<pcie_dcp_a2a::PCIeDCPA2A *>(pointer);
}

static int64_t meta_size() { return sizeof(pcie_dcp_a2a::Signal); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("init_dcp_a2a", &init_dcp_a2a, "initialize PCIe DCP A2A");
  module.def("lse_reduce_scatter", &lse_reduce_scatter,
             "fused PCIe DCP LSE reduce-scatter");
  module.def("all_gather_heads", &all_gather_heads,
             "PCIe DCP head-dimension all-gather");
  module.def("dispose", &dispose, "dispose PCIe DCP A2A");
  module.def("meta_size", &meta_size, "signal metadata size");
}
