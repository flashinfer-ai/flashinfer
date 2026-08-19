typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;
typedef signed int int32_t;
typedef short int int16_t;
struct __align__(128) CakeTensorMap {
  uint64_t opaque[16];
};
template <int N>
struct __align__(128) CakeTensorMapPack {
  CakeTensorMap maps[N];
};

typedef struct __align__(64) {
  uint64_t opaque[16];
} CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
  int result;
  asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;" : "=r"(result) : "r"(x));
  return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 128

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(128) void kernel_factorized_packed_varlen_metadata(
    int* __restrict__ seq_idx_i32, long long* __restrict__ seq_idx_i64,
    int* __restrict__ chunk_indices, int* __restrict__ chunk_offsets,
    int* __restrict__ segment_starts, int* __restrict__ segment_lengths,
    int* __restrict__ sequence_offsets, int num_segments, int num_sequences, int seqlen,
    int seq_idx_int64) {
  const int tid = threadIdx.x;
  const int warp = make_warp_uniform(tid / 32);
  const int lane = tid % 32;

  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;

  // === Task calls (dependency order) ===
  int segment = bid * 128 + tid;
  if (segment < num_segments) {
    int start = chunk_indices[segment] * 128 + chunk_offsets[segment];
    int end = seqlen;
    if (segment + 1 < num_segments) {
      end = chunk_indices[segment + 1] * 128 + chunk_offsets[segment + 1];
    }
    segment_starts[segment] = start;
    segment_lengths[segment] = end - start;
    int sequence = 0;
    if (seq_idx_int64 != 0) {
      sequence = (int)seq_idx_i64[start];
    } else {
      sequence = seq_idx_i32[start];
    }
    if (segment == 0) {
      sequence_offsets[sequence] = 0;
    } else {
      int previous_start = chunk_indices[segment - 1] * 128 + chunk_offsets[segment - 1];
      int previous_sequence = 0;
      if (seq_idx_int64 != 0) {
        previous_sequence = (int)seq_idx_i64[previous_start];
      } else {
        previous_sequence = seq_idx_i32[previous_start];
      }
      if (sequence != previous_sequence) {
        sequence_offsets[sequence] = segment;
      }
    }
    if (segment == num_segments - 1) {
      sequence_offsets[num_sequences] = num_segments;
    }
  }
}

}  // extern "C"
