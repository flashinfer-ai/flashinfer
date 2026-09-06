typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
typedef unsigned long long uint64_t;
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

typedef struct __align__(64) { uint64_t opaque[16]; } CUtensorMap;

#include <cuda_bf16.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 512

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_concat_mla_k_vector_copy(uint8_t* __restrict__ k, uint8_t* __restrict__ k_nope, uint8_t* __restrict__ k_rope, int element_bytes, long long k_stride_0_bytes, long long k_stride_1_bytes, long long k_nope_stride_0_bytes, long long k_nope_stride_1_bytes, long long k_rope_stride_0_bytes)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int token = bid;
    long long token_i64 = (long long)token;
    int vectors_per_nope_head = 128 * element_bytes / 16;
    int nope_vectors = 128 * vectors_per_nope_head;
    for (int vector = tid; vector < nope_vectors; vector += 512) {
        int head = vector / vectors_per_nope_head;
        int vector_in_head = vector - head * vectors_per_nope_head;
        long long vector_bytes = (long long)vector_in_head * 16;
        long long src_offset = token_i64 * k_nope_stride_0_bytes + (long long)head * k_nope_stride_1_bytes + vector_bytes;
        long long dst_offset = token_i64 * k_stride_0_bytes + (long long)head * k_stride_1_bytes + vector_bytes;
        reinterpret_cast<int4*>(k + dst_offset)[0] = reinterpret_cast<int4*>(k_nope + src_offset)[0];
    }
    int vectors_per_rope_head = 64 * element_bytes / 16;
    int rope_vectors = 128 * vectors_per_rope_head;
    long long rope_prefix_bytes = 128 * element_bytes;
    for (int vector_1 = tid; vector_1 < rope_vectors; vector_1 += 512) {
        int head_1 = vector_1 / vectors_per_rope_head;
        int vector_in_head_1 = vector_1 - head_1 * vectors_per_rope_head;
        long long vector_bytes_1 = (long long)vector_in_head_1 * 16;
        long long src_offset_1 = token_i64 * k_rope_stride_0_bytes + vector_bytes_1;
        long long dst_offset_1 = token_i64 * k_stride_0_bytes + (long long)head_1 * k_stride_1_bytes + rope_prefix_bytes + vector_bytes_1;
        reinterpret_cast<int4*>(k + dst_offset_1)[0] = reinterpret_cast<int4*>(k_rope + src_offset_1)[0];
    }
}

} // extern "C"
