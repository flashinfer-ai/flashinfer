// Portions derived from DeepGEMM, Copyright (c) 2025 DeepSeek.
// DeepGEMM portions are licensed under MIT; see ../DEEPGEMM_NOTICE.txt.

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Deepgemm requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) DeepgemmTensorMap { uint64_t opaque[16]; };
struct __align__(64) DeepgemmTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(DeepgemmTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(DeepgemmTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) DeepgemmTensorMapPack { DeepgemmTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(DeepgemmTensorMap) >= alignof(CUtensorMap), "DeepgemmTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define DEEPGEMM_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define THREADS 256

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(256) void
kernel_deepgemm_mega_moe_v3_sm103a_2cca3344a19322a12bf6(unsigned int* __restrict__ src_a, unsigned int* __restrict__ src_b, unsigned int* __restrict__ dst_a, unsigned int* __restrict__ dst_b, int a_rows, int b_rows, int b_group_rows, int words)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));


    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // === Task calls (dependency order) ===
    int blocks_a = (a_rows * words + 255) / 256;
    if (blocks_a > blockIdx.x) {
        int index = blockIdx.x * 256 + tid;
        if (index < a_rows * words) {
            int row = index / words;
            int word = index % words;
            int group = row / a_rows;
            int local_row = row % a_rows;
            int transposed = local_row / 128 * 128 + local_row % 32 * 4 + local_row % 128 / 32;
            dst_a[(group * words + word) * a_rows + transposed] = src_a[index];
        }
    } else {
        int index_1 = (blockIdx.x - blocks_a) * 256 + tid;
        if (index_1 < b_rows * words) {
            int row_1 = index_1 / words;
            int word_1 = index_1 % words;
            int group_1 = row_1 / b_group_rows;
            int local_row_1 = row_1 % b_group_rows;
            int transposed_1 = local_row_1 / 128 * 128 + local_row_1 % 32 * 4 + local_row_1 % 128 / 32;
            dst_b[(group_1 * words + word_1) * b_group_rows + transposed_1] = src_b[index_1];
        }
    }
}

} // extern "C"
