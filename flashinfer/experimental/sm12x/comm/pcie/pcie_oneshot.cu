// Ported from b12x b12x/distributed/pcie_oneshot.cu @ 913ac7aa (2026-07-04) -- one-time curated port.
// PCIe-only custom allreduce extension.
// System-scope barriers for cross-GPU visibility over PCIe switch fabric.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CHECK_CUDA_SUCCESS(cmd)                                         \
  do {                                                                  \
    cudaError_t e = cmd;                                                \
    if (e != cudaSuccess) {                                             \
      std::stringstream _message;                                       \
      auto s = cudaGetErrorString(e);                                   \
      _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__; \
      throw std::runtime_error(_message.str());                         \
    }                                                                   \
  } while (0)

namespace pcie_allreduce {

constexpr int kMaxBlocks = 36;
constexpr int kMaxRanks = 16;
using FlagType = uint32_t;

// 128B stride per rank to avoid PCIe false sharing.
constexpr int kFlagStride = 32;

static int env_int(const char* name, int fallback) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') return fallback;
  return std::atoi(raw);
}

// Fused kernel launch knobs, mainly for latency sweeps. Every rank must use
// identical values: launch geometry decides which columns each block stages,
// and peer block b only reads what writer block b staged. 256 threads beat
// 512 measurably at rows >= 2 on Gen5 PCIe (three packs per thread give the
// pull loop deeper memory-level parallelism than half-idle wider blocks).
static int fused_threads() {
  static const int value = [] {
    int v = env_int("FLASHINFER_EXP_SM12X_PCIE_FUSED_THREADS", 256);
    v = std::min(512, std::max(64, v));
    return (v / 32) * 32;
  }();
  return value;
}

static int fused_ctas_per_row_override() {
  static const int value = env_int("FLASHINFER_EXP_SM12X_PCIE_FUSED_CTAS_PER_ROW", 0);
  return value;
}

// Push transport requires eager slots of world_size * max_size bytes; the
// Python runtime scales the allocation when the same env toggle is set.
static bool oneshot_push_enabled() {
  static const bool value = env_int("FLASHINFER_EXP_SM12X_PCIE_ONESHOT_PUSH", 0) != 0;
  return value;
}

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
  alignas(128) FlagType rms_arrive[kMaxBlocks];
  alignas(128) FlagType rms_gen[kMaxBlocks];
  alignas(128) float rms_partial[kMaxBlocks];
};

struct __align__(16) RankData {
  const void* __restrict__ ptrs[kMaxRanks];
};

struct __align__(16) RankSignals {
  Signal* signals[kMaxRanks];
};

template <typename T, int sz>
struct __align__(alignof(T) * sz) array_t {
  T data[sz];
  using type = T;
  static constexpr int size = sz;
};

template <typename T>
struct packed_t {
  using P = array_t<T, 16 / sizeof(T)>;
  using A = array_t<float, 16 / sizeof(T)>;
};

#define DINLINE __device__ __forceinline__

DINLINE float upcast_s(float val) { return val; }
DINLINE float upcast_s(half val) { return __half2float(val); }

template <typename T>
DINLINE T downcast_s(float val);
template <>
DINLINE half downcast_s(float val) {
  return __float2half(val);
}

DINLINE half& assign_add(half& a, half b) {
  a = __hadd(a, b);
  return a;
}
DINLINE float& assign_add(float& a, float b) {
  return a += b;
}

#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
DINLINE float upcast_s(nv_bfloat16 val) { return __bfloat162float(val); }
template <>
DINLINE nv_bfloat16 downcast_s(float val) {
  return __float2bfloat16(val);
}
DINLINE nv_bfloat16& assign_add(nv_bfloat16& a, nv_bfloat16 b) {
  a = __hadd(a, b);
  return a;
}
#endif

template <typename T, int N>
DINLINE array_t<T, N>& packed_assign_add(array_t<T, N>& a, array_t<T, N> b) {
#pragma unroll
  for (int i = 0; i < N; i++) assign_add(a.data[i], b.data[i]);
  return a;
}

template <typename T, int N>
DINLINE array_t<float, N> upcast(array_t<T, N> val) {
  if constexpr (std::is_same<T, float>::value) {
    return val;
  } else {
    array_t<float, N> out;
#pragma unroll
    for (int i = 0; i < N; i++) out.data[i] = upcast_s(val.data[i]);
    return out;
  }
}

template <typename O>
DINLINE O downcast(array_t<float, O::size> val) {
  if constexpr (std::is_same<typename O::type, float>::value) {
    return val;
  } else {
    O out;
#pragma unroll
    for (int i = 0; i < O::size; i++) out.data[i] = downcast_s<typename O::type>(val.data[i]);
    return out;
  }
}

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

static DINLINE FlagType ld_flag_relaxed_gpu(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr) : "memory");
  return flag;
}

static DINLINE FlagType ld_flag_acquire_gpu(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr) : "memory");
  return flag;
}

template <int ngpus, bool is_start>
DINLINE void multi_gpu_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  if constexpr (!is_start) __syncthreads();
  if (threadIdx.x < ngpus) {
    __threadfence_system();
    auto val = self_sg->self_counter[blockIdx.x][threadIdx.x] += 1;
    auto peer_counter_ptr = &sg.signals[threadIdx.x]->peer_counter[val % 2][blockIdx.x][rank * kFlagStride];
    auto self_counter_ptr = &self_sg->peer_counter[val % 2][blockIdx.x][threadIdx.x * kFlagStride];
    st_flag_relaxed(peer_counter_ptr, val);
    while (ld_flag_relaxed(self_counter_ptr) != val)
      ;
  }
  __syncthreads();
}

template <typename P, int ngpus, typename A>
DINLINE P packed_reduce(const P* ptrs[], int idx) {
  A tmp = upcast(ptrs[0][idx]);
#pragma unroll
  for (int i = 1; i < ngpus; i++) packed_assign_add(tmp, upcast(ptrs[i][idx]));
  return downcast<P>(tmp);
}

template <typename T, int N>
DINLINE float packed_square_sum(array_t<T, N> value) {
  auto value_f = upcast(value);
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < N; i++) sum += value_f.data[i] * value_f.data[i];
  return sum;
}

DINLINE float warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

DINLINE float block_reduce_sum(float value, float* warp_sums) {
  const int lane = threadIdx.x % warpSize;
  const int warp = threadIdx.x / warpSize;
  value = warp_reduce_sum(value);
  if (lane == 0) warp_sums[warp] = value;
  __syncthreads();

  value = threadIdx.x < blockDim.x / warpSize ? warp_sums[lane] : 0.0f;
  if (warp == 0) value = warp_reduce_sum(value);
  return value;
}

// 1-stage allreduce with staggered peer reads and start-barrier-only. The end
// barrier is avoided by alternating between eager IPC buffers in a single
// ordered channel. Callers must not share a channel concurrently across CUDA
// streams; multi-stream use needs separate signal and staging buffers.
//
// With stage_input the kernel copies its own grid-stride index set from the
// local input into this channel's staging slot before the start barrier,
// replacing the host staging memcpy. Writer block b covers exactly the packs
// reader block b consumes on every rank, so the block-pairwise barrier is
// sufficient for staging visibility.
template <typename T, int ngpus, bool stage_input>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ input,
    T* __restrict__ result,
    T* __restrict__ self_staging,
    int rank,
    int size) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }
  if constexpr (stage_input) {
    const P* input_p = reinterpret_cast<const P*>(input);
    P* staging_p = reinterpret_cast<P*>(self_staging);
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
      staging_p[idx] = input_p[idx];
    }
  }
  multi_gpu_barrier<ngpus, !stage_input>(sg, self_sg, rank);
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x) {
    ((P*)result)[idx] = packed_reduce<P, ngpus, A>(rotated, idx);
  }
}

// Per-thread register budget for the fused kernel's normalize-from-registers
// path, in 16B packs. Three packs cover hidden_size 6144 at the default
// 256-thread block while staying inside the 128-register budget (four packs
// plus prefetched operands spilled to the stack and the spill traffic
// dominated the pull loop). Larger per-thread column counts fall back to
// re-reading residual_output during normalization.
constexpr int kRegPacks = 3;

// Transport modes for the fused kernel.
//  kModeRegistered: the input already lives in a peer-visible registered
//    buffer; peers pull it after the start barrier.
//  kModeStagePull: the kernel copies its columns into this rank's eager slot
//    before the start barrier (replacing the host staging memcpy); peers
//    pull staged data.
//  kModeStagePush: the kernel writes its columns into a per-source shard of
//    every peer's eager slot (PCIe posted writes run at link bandwidth while
//    pulls are round-trip bound); after the barrier the reduction reads only
//    local memory. Requires eager slots of world_size * max_size bytes.
constexpr int kModeRegistered = 0;
constexpr int kModeStagePull = 1;
constexpr int kModeStagePush = 2;

// Complete the allreduce, residual add, and RMSNorm in one launch.
//
// Geometry is uniform: gridDim.x == rows * ctas_per_row and block b serves
// row b / ctas_per_row, owning columns {(b % ctas_per_row) * blockDim.x +
// threadIdx.x + k * ctas_per_row * blockDim.x}. The same block index covers
// the same columns on every rank, so the block-pairwise start barrier is
// sufficient for staging visibility (writer block b stages exactly the
// columns reader block b consumes, for every source/destination pair).
//
// Slot reuse stays safe without an end barrier in every mode: a rank enters
// kernel k+2 only after its kernel k+1 retired, which required every rank to
// have passed kernel k+1's start barrier and hence to have fully consumed
// slot k % 2 inside kernel k.
//
// The reduction runs in fp32 registers straight through the residual add.
// With single_cta each row is normalized with block-local synchronization
// only. Otherwise each CTA publishes one fp32 square-sum partial, crosses a
// sense-reversing per-row generation barrier, and normalizes its own columns
// from registers, so no CTA re-reads the row it just wrote. All CTAs of a
// row spin, which is safe for the same reason the cross-rank start barrier
// is: the grid never exceeds kMaxBlocks <= SM count, so every block is
// resident.
template <typename T, int ngpus, bool single_cta, int mode>
__global__ void __launch_bounds__(512, 1) pcie_allreduce_fused_add_rms_norm_kernel(
    RankData* _dp,
    RankSignals sg,
    Signal* self_sg,
    const T* __restrict__ input,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ output,
    T* __restrict__ residual_output,
    T* __restrict__ self_staging,
    int rank,
    int hidden_packs,
    int ctas_per_row,
    int shard_packs,
    float epsilon) {
  using P = typename packed_t<T>::P;
  using A = typename packed_t<T>::A;
  __shared__ float warp_sums[32];
  __shared__ float s_inv_rms;

  const int row = single_cta ? blockIdx.x : blockIdx.x / ctas_per_row;
  const int row_cta = single_cta ? 0 : blockIdx.x - row * ctas_per_row;
  const int row_offset = row * hidden_packs;
  const int col_stride = single_cta ? blockDim.x : ctas_per_row * blockDim.x;
  const int col0 = row_cta * blockDim.x + threadIdx.x;
  const int max_packs = (hidden_packs + col_stride - 1) / col_stride;
  const bool reg_norm = max_packs <= kRegPacks;

  auto dp = *_dp;
  const P* rotated[ngpus];
#pragma unroll
  for (int i = 0; i < ngpus; i++) {
    rotated[i] = (const P*)dp.ptrs[(rank + i) % ngpus];
  }

  const P* input_p = reinterpret_cast<const P*>(input);
  const P* residual_p = reinterpret_cast<const P*>(residual);
  const P* weight_p = reinterpret_cast<const P*>(weight);
  P* output_p = reinterpret_cast<P*>(output);
  P* residual_output_p = reinterpret_cast<P*>(residual_output);
  P* staging_p = reinterpret_cast<P*>(self_staging);

  // Sense-reversing row barrier: latch the generation before arriving.
  // Launches are stream-serialized, so this read cannot race a peer CTA's
  // increment for this launch (that increment needs this CTA's arrival).
  FlagType row_gen = 0;
  if (!single_cta && threadIdx.x == 0) {
    row_gen = ld_flag_relaxed_gpu(&self_sg->rms_gen[row]);
  }

  // Stage this block's columns before the start barrier so the load/store
  // latency hides behind the peer flag exchange. The self contribution is
  // kept in registers instead of being re-read from staging. Residual and
  // weight stay in-loop: their local loads overlap the peer pulls and the
  // row barrier, and keeping them out of registers avoids spilling.
  P self_pk[kRegPacks];
  if (reg_norm) {
#pragma unroll
    for (int n = 0; n < kRegPacks; n++) {
      const int col = col0 + n * col_stride;
      if (col < hidden_packs) {
        const int idx = row_offset + col;
        self_pk[n] = mode == kModeRegistered ? rotated[0][idx] : input_p[idx];
        if constexpr (mode == kModeStagePull) staging_p[idx] = self_pk[n];
      }
    }
    if constexpr (mode == kModeStagePush) {
      // Peer-major order keeps each warp's stores contiguous per target so
      // they coalesce into large PCIe TLPs.
#pragma unroll
      for (int i = 1; i < ngpus; i++) {
        P* peer_shard = (P*)rotated[i] + rank * shard_packs;
#pragma unroll
        for (int n = 0; n < kRegPacks; n++) {
          const int col = col0 + n * col_stride;
          if (col < hidden_packs) peer_shard[row_offset + col] = self_pk[n];
        }
      }
    }
  } else if constexpr (mode == kModeStagePull) {
    for (int col = col0; col < hidden_packs; col += col_stride) {
      const int idx = row_offset + col;
      staging_p[idx] = input_p[idx];
    }
  } else if constexpr (mode == kModeStagePush) {
    for (int col = col0; col < hidden_packs; col += col_stride) {
      const int idx = row_offset + col;
      const P value = input_p[idx];
#pragma unroll
      for (int i = 1; i < ngpus; i++) {
        ((P*)rotated[i] + rank * shard_packs)[idx] = value;
      }
    }
  }

  // Staging modes need the pre-barrier __syncthreads (is_start == false) so
  // every thread's staging stores are fenced before the flag post.
  multi_gpu_barrier<ngpus, mode == kModeRegistered>(sg, self_sg, rank);

  float square_sum = 0.0f;
  A vals_f[kRegPacks];
  if (reg_norm) {
#pragma unroll
    for (int n = 0; n < kRegPacks; n++) {
      const int col = col0 + n * col_stride;
      if (col < hidden_packs) {
        const int idx = row_offset + col;
        A acc = upcast(self_pk[n]);
        if constexpr (mode == kModeStagePush) {
#pragma unroll
          for (int src = 0; src < ngpus; src++) {
            if (src != rank) {
              packed_assign_add(acc, upcast(rotated[0][src * shard_packs + idx]));
            }
          }
        } else {
#pragma unroll
          for (int i = 1; i < ngpus; i++) {
            packed_assign_add(acc, upcast(rotated[i][idx]));
          }
        }
        packed_assign_add(acc, upcast(residual_p[idx]));
        residual_output_p[idx] = downcast<P>(acc);
#pragma unroll
        for (int j = 0; j < A::size; j++) {
          square_sum += acc.data[j] * acc.data[j];
        }
        vals_f[n] = acc;
      }
    }
  } else {
    for (int col = col0; col < hidden_packs; col += col_stride) {
      const int idx = row_offset + col;
      A acc = upcast(mode == kModeStagePush ? input_p[idx] : rotated[0][idx]);
      if constexpr (mode == kModeStagePush) {
#pragma unroll
        for (int src = 0; src < ngpus; src++) {
          if (src != rank) {
            packed_assign_add(acc, upcast(rotated[0][src * shard_packs + idx]));
          }
        }
      } else {
#pragma unroll
        for (int i = 1; i < ngpus; i++) {
          packed_assign_add(acc, upcast(rotated[i][idx]));
        }
      }
      packed_assign_add(acc, upcast(residual_p[idx]));
      residual_output_p[idx] = downcast<P>(acc);
#pragma unroll
      for (int j = 0; j < A::size; j++) {
        square_sum += acc.data[j] * acc.data[j];
      }
    }
  }

  square_sum = block_reduce_sum(square_sum, warp_sums);
  const float hidden_size_f = static_cast<float>(hidden_packs * P::size);

  if constexpr (single_cta) {
    if (threadIdx.x == 0) {
      s_inv_rms = rsqrtf(square_sum / hidden_size_f + epsilon);
    }
  } else {
    if (threadIdx.x == 0) {
      self_sg->rms_partial[blockIdx.x] = square_sum;
      __threadfence();
      const FlagType prior = atomicAdd(&self_sg->rms_arrive[row], 1u);
      if (prior == static_cast<FlagType>(ctas_per_row - 1)) {
        // Last arriver: reset the arrival count for the next launch, then
        // release the generation. The fence orders the reset and every
        // observed peer partial before the release.
        self_sg->rms_arrive[row] = 0;
        __threadfence();
        atomicAdd(&self_sg->rms_gen[row], 1u);
      } else {
        while (ld_flag_acquire_gpu(&self_sg->rms_gen[row]) == row_gen) {
        }
      }
    }
    __syncthreads();
    if (threadIdx.x < warpSize) {
      float partial = 0.0f;
      for (int i = threadIdx.x; i < ctas_per_row; i += warpSize) {
        partial += self_sg->rms_partial[row * ctas_per_row + i];
      }
      partial = warp_reduce_sum(partial);
      if (threadIdx.x == 0) {
        s_inv_rms = rsqrtf(partial / hidden_size_f + epsilon);
      }
    }
  }
  __syncthreads();
  const float inv_rms = s_inv_rms;

  if (reg_norm) {
#pragma unroll
    for (int n = 0; n < kRegPacks; n++) {
      const int col = col0 + n * col_stride;
      if (col < hidden_packs) {
        A scale = upcast(weight_p[col]);
        A value = vals_f[n];
#pragma unroll
        for (int j = 0; j < A::size; j++) {
          value.data[j] *= inv_rms * scale.data[j];
        }
        output_p[row_offset + col] = downcast<P>(value);
      }
    }
  } else {
    for (int col = col0; col < hidden_packs; col += col_stride) {
      const int idx = row_offset + col;
      A value = upcast(residual_output_p[idx]);
      A scale = upcast(weight_p[col]);
#pragma unroll
      for (int j = 0; j < A::size; j++) {
        value.data[j] *= inv_rms * scale.data[j];
      }
      output_p[idx] = downcast<P>(value);
    }
  }
}

using IPC_KEY = std::array<uint8_t, sizeof(cudaIpcMemHandle_t)>;

class PCIeAllreduce {
 public:
  int rank_;
  int world_size_;

  RankSignals sg_;
  std::unordered_map<void*, RankData*> buffers_;
  Signal* self_sg_;

  RankData *d_rank_data_base_, *d_rank_data_end_;
  std::vector<void*> graph_unreg_buffers_;
  std::map<IPC_KEY, char*> ipc_handles_;

  bool dbuf_enabled_ = false;
  int dbuf_slot_ = 0;
  void* dbuf_raw_[2][kMaxRanks] = {};
  RankData* dbuf_rd_[2] = {};

  PCIeAllreduce(Signal** signals, void* rank_data, size_t rank_data_sz, int rank, int world_size)
      : rank_(rank),
        world_size_(world_size),
        self_sg_(signals[rank]),
        d_rank_data_base_(reinterpret_cast<RankData*>(rank_data)),
        d_rank_data_end_(d_rank_data_base_ + rank_data_sz / sizeof(RankData)) {
    for (int i = 0; i < world_size_; i++) sg_.signals[i] = signals[i];
  }

  char* open_ipc_handle(const void* ipc_handle) {
    auto [it, new_handle] = ipc_handles_.insert({*((IPC_KEY*)ipc_handle), nullptr});
    if (new_handle) {
      char* ipc_ptr;
      CHECK_CUDA_SUCCESS(cudaIpcOpenMemHandle(
          (void**)&ipc_ptr, *((const cudaIpcMemHandle_t*)ipc_handle), cudaIpcMemLazyEnablePeerAccess));
      it->second = ipc_ptr;
    }
    return it->second;
  }

  std::pair<std::string, std::vector<int64_t>> get_graph_buffer_ipc_meta() {
    auto num_buffers = graph_unreg_buffers_.size();
    auto handle_sz = sizeof(cudaIpcMemHandle_t);
    std::string handles(handle_sz * num_buffers, static_cast<char>(0));
    std::vector<int64_t> offsets(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto ptr = graph_unreg_buffers_[i];
      void* base_ptr;
      if (cuPointerGetAttribute(&base_ptr, CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, (CUdeviceptr)ptr) != CUDA_SUCCESS)
        throw std::runtime_error("failed to get pointer attr");
      CHECK_CUDA_SUCCESS(cudaIpcGetMemHandle((cudaIpcMemHandle_t*)&handles[i * handle_sz], base_ptr));
      offsets[i] = ((char*)ptr) - ((char*)base_ptr);
    }
    return std::make_pair(handles, offsets);
  }

  void check_rank_data_capacity(size_t num = 1) {
    if (d_rank_data_base_ + num > d_rank_data_end_)
      throw std::runtime_error("Rank data buffer overflow");
  }

  void register_pcie_buffers(void** ptrs0, void** ptrs1) {
    check_rank_data_capacity(2);
    for (int s = 0; s < 2; s++) {
      void** ptrs = s == 0 ? ptrs0 : ptrs1;
      RankData data;
      for (int i = 0; i < world_size_; i++) {
        data.ptrs[i] = ptrs[i];
        dbuf_raw_[s][i] = ptrs[i];
      }
      dbuf_rd_[s] = d_rank_data_base_++;
      CHECK_CUDA_SUCCESS(cudaMemcpy(dbuf_rd_[s], &data, sizeof(RankData), cudaMemcpyHostToDevice));
    }
    buffers_[ptrs0[rank_]] = dbuf_rd_[0];
    buffers_[ptrs1[rank_]] = dbuf_rd_[1];
    dbuf_enabled_ = true;
    dbuf_slot_ = 0;
  }

  void register_buffer(void** ptrs) {
    check_rank_data_capacity();
    RankData data;
    for (int i = 0; i < world_size_; i++) data.ptrs[i] = ptrs[i];
    auto d_data = d_rank_data_base_++;
    CHECK_CUDA_SUCCESS(cudaMemcpy(d_data, &data, sizeof(RankData), cudaMemcpyHostToDevice));
    buffers_[ptrs[rank_]] = d_data;
  }

  void register_graph_buffers(
      const std::vector<std::string>& handles, const std::vector<std::vector<int64_t>>& offsets) {
    auto num_buffers = graph_unreg_buffers_.size();
    check_rank_data_capacity(num_buffers);
    std::vector<RankData> rank_data(num_buffers);
    for (size_t i = 0; i < num_buffers; i++) {
      auto self_ptr = graph_unreg_buffers_[i];
      auto& rd = rank_data[i];
      for (int j = 0; j < world_size_; j++) {
        if (j != rank_) {
          char* handle = open_ipc_handle(&handles[j][i * sizeof(cudaIpcMemHandle_t)]);
          handle += offsets[j][i];
          rd.ptrs[j] = handle;
        } else {
          rd.ptrs[j] = self_ptr;
        }
      }
    }
    CHECK_CUDA_SUCCESS(
        cudaMemcpy(d_rank_data_base_, rank_data.data(), sizeof(RankData) * num_buffers, cudaMemcpyHostToDevice));
    d_rank_data_base_ += num_buffers;
    graph_unreg_buffers_.clear();
  }

  // 256-thread blocks with a low block cap measurably beat 512x36 on Gen5
  // PCIe pulls at 12KB-96KB (10.0 vs 11.8 us at 12KB, 31.9 vs 39.6 us at
  // 96KB, 8 GPUs): fewer, narrower blocks give each thread ~3 packs of
  // memory-level parallelism instead of oversubscribing the fabric.
  template <typename T>
  void allreduce(cudaStream_t stream, T* input, T* output, int size, int threads = 256, int block_limit = 8) {
    auto d = packed_t<T>::P::size;
    if (size % d != 0)
      throw std::runtime_error("allreduce requires input length to be multiple of " + std::to_string(d));
    static const int env_threads = env_int("FLASHINFER_EXP_SM12X_PCIE_ONESHOT_THREADS", 0);
    static const int env_block_limit = env_int("FLASHINFER_EXP_SM12X_PCIE_ONESHOT_BLOCK_LIMIT", 0);
    if (env_threads > 0) threads = std::min(512, std::max(64, (env_threads / 32) * 32));
    if (env_block_limit > 0) block_limit = env_block_limit;
    if (block_limit > kMaxBlocks)
      throw std::runtime_error("max supported block limit is " + std::to_string(kMaxBlocks));

    RankData* ptrs;
    T* staging = nullptr;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));

    if (dbuf_enabled_) {
      // The kernel stages the input into the eager slot itself; no host
      // staging memcpy is issued.
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      ptrs = dbuf_rd_[slot];
      staging = reinterpret_cast<T*>(dbuf_raw_[slot][rank_]);
    } else if (status == cudaStreamCaptureStatusActive) {
      throw std::runtime_error(
          "PCIe oneshot graph capture requires eager IPC buffers; construct the runtime with eager buffers or use "
          "PCIeOneshotAllReducePool");
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
      ptrs = it->second;
    }

    size /= d;
    int blocks = std::min(block_limit, (size + threads - 1) / threads);
    blocks = std::max(blocks, 1);

#define KL(ngpus)                                                                                                     \
  do {                                                                                                               \
    if (staging != nullptr) {                                                                                        \
      pcie_allreduce_kernel<T, ngpus, true>                                                                          \
          <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, input, output, staging, rank_, size);                \
    } else {                                                                                                         \
      pcie_allreduce_kernel<T, ngpus, false>                                                                         \
          <<<blocks, threads, 0, stream>>>(ptrs, sg_, self_sg_, input, output, staging, rank_, size);                \
    }                                                                                                                 \
  } while (0)
    switch (world_size_) {
      case 2:
        KL(2);
        break;
      case 4:
        KL(4);
        break;
      case 6:
        KL(6);
        break;
      case 8:
        KL(8);
        break;
      case 10:
        KL(10);
        break;
      default:
        throw std::runtime_error("only supports (2,4,6,8,10) gpus, got " + std::to_string(world_size_));
    }
#undef KL
  }

  template <typename T>
  void allreduce_fused_add_rms_norm(
      cudaStream_t stream,
      T* input,
      const T* residual,
      const T* weight,
      T* output,
      T* residual_output,
      int size,
      int hidden_size,
      float epsilon) {
    using P = typename packed_t<T>::P;
    const int pack_size = P::size;
    if (hidden_size <= 0 || size <= 0 || size % hidden_size != 0)
      throw std::runtime_error("fused allreduce RMSNorm requires complete non-empty rows");
    if (hidden_size % pack_size != 0)
      throw std::runtime_error(
          "fused allreduce RMSNorm requires hidden size to be a multiple of " + std::to_string(pack_size));

    RankData* ptrs;
    T* staging = nullptr;
    cudaStreamCaptureStatus status;
    CHECK_CUDA_SUCCESS(cudaStreamIsCapturing(stream, &status));
    if (dbuf_enabled_) {
      // The kernel stages the input into the eager slot itself; no host
      // staging memcpy is issued.
      int slot = dbuf_slot_ % 2;
      dbuf_slot_++;
      ptrs = dbuf_rd_[slot];
      staging = reinterpret_cast<T*>(dbuf_raw_[slot][rank_]);
    } else if (status == cudaStreamCaptureStatusActive) {
      throw std::runtime_error(
          "PCIe oneshot graph capture requires eager IPC buffers; construct the runtime with eager buffers or use "
          "PCIeOneshotAllReducePool");
    } else {
      auto it = buffers_.find(input);
      if (it == buffers_.end())
        throw std::runtime_error(
            "buffer address " + std::to_string(reinterpret_cast<uint64_t>(input)) + " is not registered!");
      ptrs = it->second;
    }

    int rows = size / hidden_size;
    int hidden_packs = hidden_size / pack_size;
    if (rows > kMaxBlocks)
      throw std::runtime_error(
          "fused allreduce RMSNorm supports at most " + std::to_string(kMaxBlocks) + " rows");
    const int threads = fused_threads();
    int ctas_per_row = fused_ctas_per_row_override();
    if (ctas_per_row <= 0) {
      // Measured on 8x Gen5 PCIe: the pull phase wants ~3 blocks in flight
      // (12.3 vs 13.3 us at rows=1), while extra CTAs per row only add row
      // barrier cost once rows supply that concurrency. Large hidden sizes
      // additionally need enough CTAs to keep the normalize path in
      // registers (kRegPacks packs per thread).
      const int min_ctas = (hidden_packs + threads * kRegPacks - 1) / (threads * kRegPacks);
      ctas_per_row = std::max(std::max(1, 3 / rows), min_ctas);
    }
    ctas_per_row = std::max(1, std::min(ctas_per_row, kMaxBlocks / rows));
    const int blocks = rows * ctas_per_row;
    const bool single = ctas_per_row == 1;
    const int shard_packs = size / pack_size;
    const int mode = staging == nullptr ? kModeRegistered
                     : oneshot_push_enabled() ? kModeStagePush
                                              : kModeStagePull;

#define KL(ngpus, SINGLE, MODE)                                                                                       \
  pcie_allreduce_fused_add_rms_norm_kernel<T, ngpus, SINGLE, MODE><<<blocks, threads, 0, stream>>>(                   \
      ptrs, sg_, self_sg_, input, residual, weight, output, residual_output, staging, rank_, hidden_packs,            \
      ctas_per_row, shard_packs, epsilon);
#define DISPATCH_MODE(ngpus, SINGLE)                                                                                  \
  do {                                                                                                               \
    if (mode == kModeStagePush) {                                                                                    \
      KL(ngpus, SINGLE, kModeStagePush);                                                                             \
    } else if (mode == kModeStagePull) {                                                                             \
      KL(ngpus, SINGLE, kModeStagePull);                                                                             \
    } else {                                                                                                         \
      KL(ngpus, SINGLE, kModeRegistered);                                                                            \
    }                                                                                                                 \
  } while (0)
#define DISPATCH(ngpus)                                                                                              \
  do {                                                                                                               \
    if (single) {                                                                                                    \
      DISPATCH_MODE(ngpus, true);                                                                                    \
    } else {                                                                                                         \
      DISPATCH_MODE(ngpus, false);                                                                                   \
    }                                                                                                                 \
  } while (0)
    switch (world_size_) {
      case 2:
        DISPATCH(2);
        break;
      case 4:
        DISPATCH(4);
        break;
      case 6:
        DISPATCH(6);
        break;
      case 8:
        DISPATCH(8);
        break;
      case 10:
        DISPATCH(10);
        break;
      default:
        throw std::runtime_error("only supports (2,4,6,8,10) gpus, got " + std::to_string(world_size_));
    }
#undef DISPATCH
#undef DISPATCH_MODE
#undef KL
  }

  ~PCIeAllreduce() {
    for (auto [_, ptr] : ipc_handles_) CHECK_CUDA_SUCCESS(cudaIpcCloseMemHandle(ptr));
  }
};

}  // namespace pcie_allreduce

using fptr_t = int64_t;

static fptr_t init_custom_ar(const std::vector<fptr_t>& fake_ipc_ptrs, torch::Tensor& rank_data, int64_t rank) {
  int world_size = fake_ipc_ptrs.size();
  if (world_size > pcie_allreduce::kMaxRanks)
    throw std::invalid_argument("world size > " + std::to_string(pcie_allreduce::kMaxRanks) + " is not supported");
  if (world_size % 2 != 0) throw std::invalid_argument("Odd num gpus is not supported");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  pcie_allreduce::Signal* ipc_ptrs[pcie_allreduce::kMaxRanks];
  for (int i = 0; i < world_size; i++) ipc_ptrs[i] = reinterpret_cast<pcie_allreduce::Signal*>(fake_ipc_ptrs[i]);
  return (fptr_t) new pcie_allreduce::PCIeAllreduce(ipc_ptrs, rank_data.data_ptr(), rank_data.numel(), rank, world_size);
}

static bool _is_weak_contiguous(torch::Tensor& t) {
  return t.is_contiguous() ||
         (t.storage().nbytes() - t.storage_offset() * t.element_size() == t.numel() * t.element_size());
}

static void all_reduce(fptr_t _fa, torch::Tensor& inp, torch::Tensor& out, fptr_t _reg_buffer, int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(inp));
  auto input_size = inp.numel() * inp.element_size();

  // When double-buffer is active, allreduce handles the memcpy and slot
  // alternation internally, including under CUDA graph capture.
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      fa->allreduce<float>(stream, (float*)reg_buffer, (float*)out.data_ptr(), out.numel());
      break;
    case at::ScalarType::Half:
      fa->allreduce<half>(stream, (half*)reg_buffer, (half*)out.data_ptr(), out.numel());
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      fa->allreduce<nv_bfloat16>(stream, (nv_bfloat16*)reg_buffer, (nv_bfloat16*)out.data_ptr(), out.numel());
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
}

static void all_reduce_fused_add_rms_norm(
    fptr_t _fa,
    torch::Tensor& inp,
    torch::Tensor& residual,
    torch::Tensor& weight,
    torch::Tensor& out,
    torch::Tensor& residual_out,
    double epsilon,
    fptr_t _reg_buffer,
    int64_t reg_buffer_sz_bytes) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(inp));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();

  TORCH_CHECK_GE(inp.dim(), 1);
  TORCH_CHECK_EQ(inp.scalar_type(), residual.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), weight.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
  TORCH_CHECK_EQ(inp.scalar_type(), residual_out.scalar_type());
  TORCH_CHECK_EQ(inp.device(), residual.device());
  TORCH_CHECK_EQ(inp.device(), weight.device());
  TORCH_CHECK_EQ(inp.device(), out.device());
  TORCH_CHECK_EQ(inp.device(), residual_out.device());
  TORCH_CHECK_EQ(inp.numel(), residual.numel());
  TORCH_CHECK_EQ(inp.numel(), out.numel());
  TORCH_CHECK_EQ(inp.numel(), residual_out.numel());
  TORCH_CHECK_EQ(weight.dim(), 1);
  TORCH_CHECK_EQ(weight.numel(), inp.size(-1));
  TORCH_CHECK_GE(epsilon, 0.0);
  TORCH_CHECK(_is_weak_contiguous(inp));
  TORCH_CHECK(_is_weak_contiguous(residual));
  TORCH_CHECK(weight.is_contiguous());
  TORCH_CHECK(_is_weak_contiguous(out));
  TORCH_CHECK(_is_weak_contiguous(residual_out));
  TORCH_CHECK_NE(out.data_ptr(), residual_out.data_ptr());

  auto input_size = inp.numel() * inp.element_size();
  void* reg_buffer;
  if (fa->dbuf_enabled_) {
    reg_buffer = inp.data_ptr();
  } else if (_reg_buffer) {
    reg_buffer = reinterpret_cast<void*>(_reg_buffer);
    TORCH_CHECK_LE(input_size, reg_buffer_sz_bytes);
    AT_CUDA_CHECK(cudaMemcpyAsync(reg_buffer, inp.data_ptr(), input_size, cudaMemcpyDeviceToDevice, stream));
  } else {
    reg_buffer = inp.data_ptr();
  }

#define CALL_FUSED(T)                                                                                              \
  fa->allreduce_fused_add_rms_norm<T>(                                                                             \
      stream,                                                                                                      \
      reinterpret_cast<T*>(reg_buffer),                                                                           \
      reinterpret_cast<const T*>(residual.data_ptr()),                                                             \
      reinterpret_cast<const T*>(weight.data_ptr()),                                                               \
      reinterpret_cast<T*>(out.data_ptr()),                                                                        \
      reinterpret_cast<T*>(residual_out.data_ptr()),                                                               \
      inp.numel(),                                                                                                  \
      inp.size(-1),                                                                                                 \
      static_cast<float>(epsilon));
  switch (inp.scalar_type()) {
    case at::ScalarType::Float:
      CALL_FUSED(float);
      break;
    case at::ScalarType::Half:
      CALL_FUSED(half);
      break;
#if (__CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__))
    case at::ScalarType::BFloat16:
      CALL_FUSED(nv_bfloat16);
      break;
#endif
    default:
      throw std::runtime_error("only supports float32, float16 and bfloat16");
  }
#undef CALL_FUSED
}

static void dispose(fptr_t _fa) {
  delete reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
}

static int64_t meta_size() {
  return sizeof(pcie_allreduce::Signal);
}

static void register_buffer(fptr_t _fa, const std::vector<fptr_t>& fake_ipc_ptrs) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(fake_ipc_ptrs.size() == (size_t)fa->world_size_);
  void* ipc_ptrs[pcie_allreduce::kMaxRanks];
  for (size_t i = 0; i < fake_ipc_ptrs.size(); i++) ipc_ptrs[i] = reinterpret_cast<void*>(fake_ipc_ptrs[i]);
  fa->register_buffer(ipc_ptrs);
}

static void register_pcie_buffers(fptr_t _fa, const std::vector<fptr_t>& ptrs0, const std::vector<fptr_t>& ptrs1) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  TORCH_CHECK(ptrs0.size() == (size_t)fa->world_size_);
  TORCH_CHECK(ptrs1.size() == (size_t)fa->world_size_);
  void* p0[pcie_allreduce::kMaxRanks];
  void* p1[pcie_allreduce::kMaxRanks];
  for (size_t i = 0; i < ptrs0.size(); i++) {
    p0[i] = reinterpret_cast<void*>(ptrs0[i]);
    p1[i] = reinterpret_cast<void*>(ptrs1[i]);
  }
  fa->register_pcie_buffers(p0, p1);
}

static std::tuple<std::vector<int64_t>, std::vector<int64_t>> get_graph_buffer_ipc_meta(fptr_t _fa) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  auto [handle, offsets] = fa->get_graph_buffer_ipc_meta();
  std::vector<int64_t> bytes(handle.begin(), handle.end());
  return std::make_tuple(bytes, offsets);
}

static void register_graph_buffers(
    fptr_t _fa, const std::vector<std::vector<int64_t>>& handles, const std::vector<std::vector<int64_t>>& offsets) {
  auto fa = reinterpret_cast<pcie_allreduce::PCIeAllreduce*>(_fa);
  std::vector<std::string> bytes;
  bytes.reserve(handles.size());
  for (size_t i = 0; i < handles.size(); i++) bytes.emplace_back(handles[i].begin(), handles[i].end());
  fa->register_graph_buffers(bytes, offsets);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_custom_ar", &init_custom_ar, "init PCIe allreduce");
  m.def("all_reduce", &all_reduce, "PCIe allreduce");
  m.def(
      "all_reduce_fused_add_rms_norm",
      &all_reduce_fused_add_rms_norm,
      "PCIe allreduce fused with residual add and RMSNorm");
  m.def("dispose", &dispose, "dispose PCIe allreduce");
  m.def("meta_size", &meta_size, "signal metadata size");
  m.def("register_buffer", &register_buffer, "register IPC buffer");
  m.def("register_pcie_buffers", &register_pcie_buffers, "register double-buffered IPC buffers");
  m.def("get_graph_buffer_ipc_meta", &get_graph_buffer_ipc_meta, "get graph buffer IPC meta");
  m.def("register_graph_buffers", &register_graph_buffers, "register graph buffers");
}
