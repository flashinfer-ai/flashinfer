// Ported from b12x b12x/distributed/pcie_twoshot.cu @ 7bfc9455 (2026-07-04) -- one-time curated port.
// PCIe two-shot sequence-parallel collectives with fp8 transport.
//
// Push-based: PCIe P2P writes run at full link bandwidth while P2P reads
// are latency-bound, so each rank WRITES its quantized slices into the
// destination rank's IPC staging region, crosses a block-pairwise
// system-scope barrier, then consumes staged data with LOCAL reads:
//
//   reduce_scatter_fp8: rank j pushes its per-token-quantized partials
//   for shard r into rank r's staging[j]; the owner then does a fused
//   dequant + fp32 accumulate (+ its own local contribution) + bf16
//   store. Each value is quantized exactly once at the source.
//
//   all_gather_fp8: rank j pushes its quantized shard into every peer's
//   staging[j]; each rank then dequantizes staged shards to bf16.
//
// Writer block b covers the same pack range that reader block b consumes
// (for every src/dst pair), so the per-block-index pairwise barrier is
// sufficient. Two staging slots alternate per channel so no end barrier
// is needed (same scheme as pcie_oneshot.cu); a channel must not be
// shared concurrently across CUDA streams. Sources are read directly
// from the caller's tensors (no input staging copy), which keeps the
// ops CUDA-graph-capturable given stable input allocations.

#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/all.h>
#include <torch/extension.h>

#include <array>
#include <sstream>
#include <stdexcept>
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

namespace pcie_twoshot {

constexpr int kMaxBlocks = 64;
constexpr int kMaxRanks = 16;
using FlagType = uint32_t;

// 128B stride per rank to avoid PCIe false sharing.
constexpr int kFlagStride = 32;

struct Signal {
  alignas(128) FlagType self_counter[kMaxBlocks][kMaxRanks];
  alignas(128) FlagType peer_counter[2][kMaxBlocks][kMaxRanks * kFlagStride];
};

struct __align__(16) RankPtrs {
  void* __restrict__ ptrs[kMaxRanks];
};

struct RankSignals {
  Signal* signals[kMaxRanks];
};

#define DINLINE __device__ __forceinline__

static DINLINE void st_flag_relaxed(FlagType* flag_addr, FlagType flag) {
  asm volatile("st.relaxed.sys.global.u32 [%1], %0;" ::"r"(flag), "l"(flag_addr));
}

static DINLINE FlagType ld_flag_relaxed(FlagType* flag_addr) {
  FlagType flag;
  asm volatile("ld.relaxed.sys.global.u32 %0, [%1];" : "=r"(flag) : "l"(flag_addr));
  return flag;
}

// Block-pairwise barrier: after it returns, all prior writes from every
// rank's block blockIdx.x are visible to this block.
template <int ngpus>
DINLINE void block_pair_barrier(const RankSignals& sg, Signal* self_sg, int rank) {
  __syncthreads();
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

// 16 fp8 values as one 16B vector.
struct __align__(16) Fp8Pack {
  __nv_fp8_e4m3 v[16];
};

struct __align__(16) Bf16Pack8 {
  nv_bfloat16 v[8];
};

DINLINE void dequant_accum(const Fp8Pack& p, float scale, float* acc) {
#pragma unroll
  for (int k = 0; k < 16; k++) acc[k] += float(p.v[k]) * scale;
}

DINLINE void store_bf16x16(nv_bfloat16* dst, const float* acc) {
  Bf16Pack8 lo, hi;
#pragma unroll
  for (int k = 0; k < 8; k++) {
    lo.v[k] = __float2bfloat16(acc[k]);
    hi.v[k] = __float2bfloat16(acc[k + 8]);
  }
  reinterpret_cast<Bf16Pack8*>(dst)[0] = lo;
  reinterpret_cast<Bf16Pack8*>(dst)[1] = hi;
}

// Staging layout per rank per slot (all offsets 256B-aligned host-side):
//   payload: [world][pack_stride] Fp8Pack, indexed by src rank
//   scales : [world][scale_stride] fp32 at scale_offset, indexed by src

DINLINE Fp8Pack* staged_payload(void* base, int64_t pack_stride, int src) {
  return reinterpret_cast<Fp8Pack*>(base) + int64_t(src) * pack_stride;
}

DINLINE float* staged_scale(void* base, int64_t scale_offset, int64_t scale_stride, int src) {
  return reinterpret_cast<float*>(reinterpret_cast<char*>(base) + scale_offset) +
         int64_t(src) * scale_stride;
}

// Grid-partitioned range [begin, end) for this block over `total` packs.
DINLINE void block_range(int64_t total, int64_t& begin, int64_t& end) {
  const int64_t chunk = (total + gridDim.x - 1) / gridDim.x;
  begin = int64_t(blockIdx.x) * chunk;
  end = min(begin + chunk, total);
}

template <int ngpus>
__global__ void __launch_bounds__(512, 1) rs_fp8_kernel(
    const Fp8Pack* __restrict__ src_payload, const float* __restrict__ src_scale,
    RankPtrs staging, int64_t pack_stride, int64_t scale_offset, int64_t scale_stride,
    RankSignals sg, Signal* self_sg, nv_bfloat16* __restrict__ out, int rank,
    int rows_per_rank, int row_elems) {
  const int packs_per_row = row_elems / 16;
  const int64_t shard_packs = int64_t(rows_per_rank) * packs_per_row;
  int64_t begin, end;
  block_range(shard_packs, begin, end);

  // Phase 1: push my quantized partials for every remote shard into the
  // owner's staging[rank], peers visited in rank-staggered order.
#pragma unroll 1
  for (int i = 1; i < ngpus; i++) {
    const int dst = (rank + i) % ngpus;
    void* dst_base = staging.ptrs[dst];
    Fp8Pack* dst_payload = staged_payload(dst_base, pack_stride, rank);
    float* dst_scale = staged_scale(dst_base, scale_offset, scale_stride, rank);
    const Fp8Pack* src = src_payload + int64_t(dst) * shard_packs;
    const float* src_s = src_scale + int64_t(dst) * rows_per_rank;
    for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
      dst_payload[idx] = src[idx];
    }
    // Scales for the rows covered by this block's pack range.
    const int64_t row_begin = begin / packs_per_row;
    const int64_t row_end = (end + packs_per_row - 1) / packs_per_row;
    for (int64_t row = row_begin + threadIdx.x; row < row_end; row += blockDim.x) {
      dst_scale[row] = src_s[row];
    }
  }

  block_pair_barrier<ngpus>(sg, self_sg, rank);

  // Phase 2: fused dequant + accumulate from local staging plus my own
  // (unstaged) contribution, then bf16 store.
  void* self_base = staging.ptrs[rank];
  const Fp8Pack* self_src = src_payload + int64_t(rank) * shard_packs;
  const float* self_scale_src = src_scale + int64_t(rank) * rows_per_rank;
  for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
    const int64_t row = idx / packs_per_row;
    float acc[16] = {};
    dequant_accum(self_src[idx], self_scale_src[row], acc);
#pragma unroll 1
    for (int i = 1; i < ngpus; i++) {
      const int src_rank = (rank + i) % ngpus;
      const Fp8Pack* p = staged_payload(self_base, pack_stride, src_rank);
      const float* s = staged_scale(self_base, scale_offset, scale_stride, src_rank);
      dequant_accum(p[idx], s[row], acc);
    }
    store_bf16x16(&out[idx * 16], acc);
  }
}

template <int ngpus>
__global__ void __launch_bounds__(512, 1) ag_fp8_kernel(
    const Fp8Pack* __restrict__ src_payload, const float* __restrict__ src_scale,
    RankPtrs staging, int64_t pack_stride, int64_t scale_offset, int64_t scale_stride,
    RankSignals sg, Signal* self_sg, nv_bfloat16* __restrict__ out, int rank,
    int rows_per_rank, int row_elems) {
  const int packs_per_row = row_elems / 16;
  const int64_t shard_packs = int64_t(rows_per_rank) * packs_per_row;
  int64_t begin, end;
  block_range(shard_packs, begin, end);
  const int64_t row_begin = begin / packs_per_row;
  const int64_t row_end = (end + packs_per_row - 1) / packs_per_row;

  // Phase 1: push my shard into every peer's staging[rank].
#pragma unroll 1
  for (int i = 1; i < ngpus; i++) {
    const int dst = (rank + i) % ngpus;
    void* dst_base = staging.ptrs[dst];
    Fp8Pack* dst_payload = staged_payload(dst_base, pack_stride, rank);
    float* dst_scale = staged_scale(dst_base, scale_offset, scale_stride, rank);
    for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
      dst_payload[idx] = src_payload[idx];
    }
    for (int64_t row = row_begin + threadIdx.x; row < row_end; row += blockDim.x) {
      dst_scale[row] = src_scale[row];
    }
  }

  block_pair_barrier<ngpus>(sg, self_sg, rank);

  // Phase 2: dequantize every shard (peers from local staging, own shard
  // straight from the source) into the full-width bf16 output.
  void* self_base = staging.ptrs[rank];
#pragma unroll 1
  for (int i = 0; i < ngpus; i++) {
    const int src_rank = (rank + i) % ngpus;
    const Fp8Pack* p = (src_rank == rank)
                           ? src_payload
                           : staged_payload(self_base, pack_stride, src_rank);
    const float* s = (src_rank == rank)
                         ? src_scale
                         : staged_scale(self_base, scale_offset, scale_stride, src_rank);
    nv_bfloat16* dst = out + int64_t(src_rank) * rows_per_rank * row_elems;
    for (int64_t idx = begin + threadIdx.x; idx < end; idx += blockDim.x) {
      const int64_t row = idx / packs_per_row;
      float acc[16] = {};
      dequant_accum(p[idx], s[row], acc);
      store_bf16x16(&dst[idx * 16], acc);
    }
  }
}

class PCIeTwoShot {
 public:
  int rank_;
  int world_size_;
  RankSignals sg_;
  Signal* self_sg_;

  RankPtrs staging_[2];
  int64_t pack_stride_;   // Fp8Packs per src region
  int64_t scale_offset_;  // bytes
  int64_t scale_stride_;  // floats per src region
  int64_t max_shard_packs_;
  int slot_ = 0;

  PCIeTwoShot(Signal** signals, const std::vector<std::array<void*, 2>>& staging,
              int64_t pack_stride, int64_t scale_offset, int64_t scale_stride, int rank,
              int world_size)
      : rank_(rank),
        world_size_(world_size),
        self_sg_(signals[rank]),
        pack_stride_(pack_stride),
        scale_offset_(scale_offset),
        scale_stride_(scale_stride),
        max_shard_packs_(pack_stride) {
    for (int i = 0; i < world_size_; i++) {
      sg_.signals[i] = signals[i];
      for (int s = 0; s < 2; s++) staging_[s].ptrs[i] = staging[i][s];
    }
  }

  template <typename Fn>
  void dispatch(Fn&& fn) {
    switch (world_size_) {
      case 2:
        fn(std::integral_constant<int, 2>{});
        break;
      case 4:
        fn(std::integral_constant<int, 4>{});
        break;
      case 8:
        fn(std::integral_constant<int, 8>{});
        break;
      default:
        throw std::runtime_error("pcie_twoshot supports 2, 4 or 8 gpus, got " +
                                 std::to_string(world_size_));
    }
  }

  void check_shard(int64_t rows_per_rank, int64_t row_elems) {
    if (row_elems % 16 != 0) throw std::runtime_error("row_elems must be a multiple of 16");
    const int64_t shard_packs = rows_per_rank * (row_elems / 16);
    if (shard_packs > max_shard_packs_)
      throw std::runtime_error("pcie_twoshot staging capacity exceeded");
    if (rows_per_rank > scale_stride_)
      throw std::runtime_error("pcie_twoshot scale capacity exceeded");
  }

  void reduce_scatter(cudaStream_t stream, const void* payload, const void* scale, void* out,
                      int64_t num_rows, int64_t row_elems, int threads, int block_limit) {
    if (num_rows % world_size_ != 0)
      throw std::runtime_error("num_rows must be divisible by world size");
    const int64_t rows_per_rank = num_rows / world_size_;
    check_shard(rows_per_rank, row_elems);
    const int s = slot_ % 2;
    slot_++;
    const int64_t shard_packs = rows_per_rank * (row_elems / 16);
    int blocks =
        std::max<int64_t>(1, std::min<int64_t>(block_limit, (shard_packs + threads - 1) / threads));
    dispatch([&](auto ng) {
      constexpr int ngpus = decltype(ng)::value;
      rs_fp8_kernel<ngpus><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const Fp8Pack*>(payload), reinterpret_cast<const float*>(scale),
          staging_[s], pack_stride_, scale_offset_, scale_stride_, sg_, self_sg_,
          reinterpret_cast<nv_bfloat16*>(out), rank_, int(rows_per_rank), int(row_elems));
    });
  }

  void all_gather(cudaStream_t stream, const void* payload, const void* scale, void* out,
                  int64_t rows_per_rank, int64_t row_elems, int threads, int block_limit) {
    check_shard(rows_per_rank, row_elems);
    const int s = slot_ % 2;
    slot_++;
    const int64_t shard_packs = rows_per_rank * (row_elems / 16);
    int blocks =
        std::max<int64_t>(1, std::min<int64_t>(block_limit, (shard_packs + threads - 1) / threads));
    dispatch([&](auto ng) {
      constexpr int ngpus = decltype(ng)::value;
      ag_fp8_kernel<ngpus><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const Fp8Pack*>(payload), reinterpret_cast<const float*>(scale),
          staging_[s], pack_stride_, scale_offset_, scale_stride_, sg_, self_sg_,
          reinterpret_cast<nv_bfloat16*>(out), rank_, int(rows_per_rank), int(row_elems));
    });
  }
};

}  // namespace pcie_twoshot

using fptr_t = int64_t;

static fptr_t init_twoshot(const std::vector<fptr_t>& signal_ptrs,
                           const std::vector<fptr_t>& staging0, const std::vector<fptr_t>& staging1,
                           int64_t pack_stride, int64_t scale_offset, int64_t scale_stride,
                           int64_t rank) {
  const int world_size = signal_ptrs.size();
  if (world_size != 2 && world_size != 4 && world_size != 8)
    throw std::invalid_argument("pcie_twoshot supports 2, 4 or 8 gpus");
  if (rank < 0 || rank >= world_size) throw std::invalid_argument("invalid rank");

  pcie_twoshot::Signal* signals[pcie_twoshot::kMaxRanks];
  std::vector<std::array<void*, 2>> staging(world_size);
  for (int i = 0; i < world_size; i++) {
    signals[i] = reinterpret_cast<pcie_twoshot::Signal*>(signal_ptrs[i]);
    staging[i] = {reinterpret_cast<void*>(staging0[i]), reinterpret_cast<void*>(staging1[i])};
  }
  return (fptr_t) new pcie_twoshot::PCIeTwoShot(signals, staging, pack_stride, scale_offset,
                                                scale_stride, rank, world_size);
}

static void check_fp8_inputs(torch::Tensor& payload, torch::Tensor& scale, torch::Tensor& out) {
  TORCH_CHECK(payload.is_contiguous() && scale.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(payload.scalar_type() == at::ScalarType::Float8_e4m3fn ||
              payload.scalar_type() == at::ScalarType::Byte);
  TORCH_CHECK_EQ(scale.scalar_type(), at::ScalarType::Float);
  TORCH_CHECK_EQ(out.scalar_type(), at::ScalarType::BFloat16);
}

static void reduce_scatter_fp8(fptr_t _fa, torch::Tensor& payload, torch::Tensor& scale,
                               torch::Tensor& out, int64_t threads, int64_t block_limit) {
  auto fa = reinterpret_cast<pcie_twoshot::PCIeTwoShot*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(payload));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  check_fp8_inputs(payload, scale, out);
  const int64_t num_rows = payload.size(0);
  const int64_t row_elems = payload.size(1);
  TORCH_CHECK_EQ(scale.numel(), num_rows);
  TORCH_CHECK_EQ(out.numel(), (num_rows / fa->world_size_) * row_elems);
  fa->reduce_scatter(stream, payload.data_ptr(), scale.data_ptr(), out.data_ptr(), num_rows,
                     row_elems, int(threads), int(block_limit));
}

static void all_gather_fp8(fptr_t _fa, torch::Tensor& payload, torch::Tensor& scale,
                           torch::Tensor& out, int64_t threads, int64_t block_limit) {
  auto fa = reinterpret_cast<pcie_twoshot::PCIeTwoShot*>(_fa);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(payload));
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  check_fp8_inputs(payload, scale, out);
  const int64_t rows_per_rank = payload.size(0);
  const int64_t row_elems = payload.size(1);
  TORCH_CHECK_EQ(scale.numel(), rows_per_rank);
  TORCH_CHECK_EQ(out.numel(), rows_per_rank * fa->world_size_ * row_elems);
  fa->all_gather(stream, payload.data_ptr(), scale.data_ptr(), out.data_ptr(), rows_per_rank,
                 row_elems, int(threads), int(block_limit));
}

static void dispose(fptr_t _fa) {
  delete reinterpret_cast<pcie_twoshot::PCIeTwoShot*>(_fa);
}

static int64_t meta_size() {
  return sizeof(pcie_twoshot::Signal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_twoshot", &init_twoshot, "init PCIe twoshot SP collectives");
  m.def("reduce_scatter_fp8", &reduce_scatter_fp8, "fp8-transport reduce_scatter");
  m.def("all_gather_fp8", &all_gather_fp8, "fp8-transport all_gather");
  m.def("dispose", &dispose, "dispose");
  m.def("meta_size", &meta_size, "signal metadata size");
}
