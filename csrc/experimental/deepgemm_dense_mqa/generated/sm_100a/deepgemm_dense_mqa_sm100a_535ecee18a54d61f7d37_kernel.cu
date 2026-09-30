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
#define SMEM_WORK_PREFIX_OFF 0
#define SMEM_WORK_PREFIX_STAGE_BYTES 16
#define SMEM_WORK_PREFIX_STRIDE 16
#define SMEM_COST_PREFIX_OFF 16
#define SMEM_COST_PREFIX_STAGE_BYTES 16
#define SMEM_COST_PREFIX_STRIDE 16
#define SMEM_WARP_SUMS_OFF 32
#define SMEM_WARP_SUMS_STAGE_BYTES 64
#define SMEM_WARP_SUMS_STRIDE 64
#define SMEM_CARRY_OFF 96
#define SMEM_CARRY_STAGE_BYTES 8
#define SMEM_CARRY_STRIDE 8
#define SMEM_TOTAL 128
#define THREADS 256

#include <math_constants.h>

__device__ __forceinline__ uint32_t elect_sync() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t"
        ".reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t"
        "}\n"
        : "+r"(pred)
        : "r"(0xFFFFFFFF));
    return pred;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

extern "C" {

__global__ __launch_bounds__(256, 1) void
kernel_deepgemm_dense_mqa_sm100a_535ecee18a54d61f7d37(unsigned int* __restrict__ Starts, unsigned int* __restrict__ Ends, unsigned int* __restrict__ Metadata, unsigned int num_q_tokens, unsigned int num_kv_tokens)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    unsigned int* work_prefix = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int work_prefix_addr = smem + 0;
    unsigned int* cost_prefix = reinterpret_cast<unsigned int*>(smem_raw + 16);
    const int cost_prefix_addr = smem + 16;
    unsigned long long* warp_sums = reinterpret_cast<unsigned long long*>(smem_raw + 32);
    const int warp_sums_addr = smem + 32;
    unsigned long long* carry = reinterpret_cast<unsigned long long*>(smem_raw + 96);
    const int carry_addr = smem + 96;

    // Kernel post-init ops
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // === Task calls (dependency order) ===
    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    #pragma unroll 1
    for (unsigned int qidx = tid; qidx < 4; qidx += 256) {
        unsigned int start = 4294967295;
        unsigned int end = 0;
        #pragma unroll
        for (int qi = 0; qi < 4; qi++) {
            unsigned int _min_0 = ((qidx * 4 + (unsigned int)qi) < (num_q_tokens - 1) ? (qidx * 4 + (unsigned int)qi) : (num_q_tokens - 1));
            unsigned int row = _min_0;
            unsigned int _min_1 = ((Starts[row]) < (num_kv_tokens) ? (Starts[row]) : (num_kv_tokens));
            unsigned int _min_2 = ((start) < (_min_1) ? (start) : (_min_1));
            start = _min_2;
            unsigned int _min_3 = ((Ends[row]) < (num_kv_tokens) ? (Ends[row]) : (num_kv_tokens));
            unsigned int _max_0 = ((end) > (_min_3) ? (end) : (_min_3));
            end = _max_0;
        }
        unsigned int base = start / 4 * 4;
        unsigned int splits = (end - base + 255) / 256;
        Metadata[444 + qidx * 2] = base;
        Metadata[444 + qidx * 2 + 1] = splits;
        work_prefix[qidx] = splits;
        cost_prefix[qidx] = splits + (unsigned int)(((splits == 0) ? 0 : 1));
    }
    __syncthreads();
    unsigned long long value = 0;
    if (tid < 4) {
        value = ((unsigned long long)work_prefix[tid] << 32) + (unsigned long long)cost_prefix[tid];
    }
    unsigned long long scanned = value;
    unsigned long long _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, scanned, 1, 32);
    unsigned long long synced = _shfl_up_0;
    if (lane >= 1) {
        scanned += synced;
    }
    unsigned long long _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, scanned, 2, 32);
    unsigned long long synced_0 = _shfl_up_1;
    if (lane >= 2) {
        scanned += synced_0;
    }
    unsigned long long _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, scanned, 4, 32);
    unsigned long long synced_1 = _shfl_up_2;
    if (lane >= 4) {
        scanned += synced_1;
    }
    unsigned long long _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, scanned, 8, 32);
    unsigned long long synced_2 = _shfl_up_3;
    if (lane >= 8) {
        scanned += synced_2;
    }
    unsigned long long _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, scanned, 16, 32);
    unsigned long long synced_3 = _shfl_up_4;
    if (lane >= 16) {
        scanned += synced_3;
    }
    if (tid < 4) {
        work_prefix[tid] = (unsigned int)(scanned >> 32);
        cost_prefix[tid] = (unsigned int)scanned;
    }
    __syncthreads();
    unsigned int total_work = work_prefix[3];
    unsigned int total_cost = cost_prefix[3];
    if (tid < 148) {
        unsigned int base_1 = total_cost / 148;
        unsigned int remainder = total_cost % 148;
        int _min_4 = ((tid) < (remainder) ? (tid) : (remainder));
        unsigned int target = (unsigned int)tid * base_1 + (unsigned int)_min_4;
        unsigned int block = 4;
        unsigned int split = 0;
        unsigned int coordinate = total_work;
        if (target != total_cost) {
            block = 0;
            unsigned int candidate = block + 16384;
            if (candidate <= 4) {
                if (target >= cost_prefix[candidate - 1]) {
                    block = candidate;
                }
            }
            unsigned int candidate_0 = block + 8192;
            if (candidate_0 <= 4) {
                if (target >= cost_prefix[candidate_0 - 1]) {
                    block = candidate_0;
                }
            }
            unsigned int candidate_1 = block + 4096;
            if (candidate_1 <= 4) {
                if (target >= cost_prefix[candidate_1 - 1]) {
                    block = candidate_1;
                }
            }
            unsigned int candidate_2 = block + 2048;
            if (candidate_2 <= 4) {
                if (target >= cost_prefix[candidate_2 - 1]) {
                    block = candidate_2;
                }
            }
            unsigned int candidate_3 = block + 1024;
            if (candidate_3 <= 4) {
                if (target >= cost_prefix[candidate_3 - 1]) {
                    block = candidate_3;
                }
            }
            unsigned int candidate_4 = block + 512;
            if (candidate_4 <= 4) {
                if (target >= cost_prefix[candidate_4 - 1]) {
                    block = candidate_4;
                }
            }
            unsigned int candidate_5 = block + 256;
            if (candidate_5 <= 4) {
                if (target >= cost_prefix[candidate_5 - 1]) {
                    block = candidate_5;
                }
            }
            unsigned int candidate_6 = block + 128;
            if (candidate_6 <= 4) {
                if (target >= cost_prefix[candidate_6 - 1]) {
                    block = candidate_6;
                }
            }
            unsigned int candidate_7 = block + 64;
            if (candidate_7 <= 4) {
                if (target >= cost_prefix[candidate_7 - 1]) {
                    block = candidate_7;
                }
            }
            unsigned int candidate_8 = block + 32;
            if (candidate_8 <= 4) {
                if (target >= cost_prefix[candidate_8 - 1]) {
                    block = candidate_8;
                }
            }
            unsigned int candidate_9 = block + 16;
            if (candidate_9 <= 4) {
                if (target >= cost_prefix[candidate_9 - 1]) {
                    block = candidate_9;
                }
            }
            unsigned int candidate_10 = block + 8;
            if (candidate_10 <= 4) {
                if (target >= cost_prefix[candidate_10 - 1]) {
                    block = candidate_10;
                }
            }
            unsigned int candidate_11 = block + 4;
            if (candidate_11 <= 4) {
                if (target >= cost_prefix[candidate_11 - 1]) {
                    block = candidate_11;
                }
            }
            unsigned int candidate_12 = block + 2;
            if (candidate_12 <= 4) {
                if (target >= cost_prefix[candidate_12 - 1]) {
                    block = candidate_12;
                }
            }
            unsigned int candidate_13 = block + 1;
            if (candidate_13 <= 4) {
                if (target >= cost_prefix[candidate_13 - 1]) {
                    block = candidate_13;
                }
            }
            unsigned int cost_before = 0;
            unsigned int work_before = 0;
            if (block > 0) {
                cost_before = cost_prefix[block - 1];
                work_before = work_prefix[block - 1];
            }
            unsigned int _max_1 = ((target - cost_before) > (1) ? (target - cost_before) : (1));
            unsigned int _min_5 = ((_max_1 - 1) < (work_prefix[block] - work_before - 1) ? (_max_1 - 1) : (work_prefix[block] - work_before - 1));
            split = _min_5;
            coordinate = work_before + split;
        }
        int _min_6 = ((tid + 1) < (remainder) ? (tid + 1) : (remainder));
        unsigned int target_0 = (unsigned int)(tid + 1) * base_1 + (unsigned int)_min_6;
        unsigned int block_1 = 4;
        unsigned int split_2 = 0;
        unsigned int coordinate_3 = total_work;
        if (target_0 != total_cost) {
            block_1 = 0;
            unsigned int candidate_14 = block_1 + 16384;
            if (candidate_14 <= 4) {
                if (target_0 >= cost_prefix[candidate_14 - 1]) {
                    block_1 = candidate_14;
                }
            }
            unsigned int candidate_0_1 = block_1 + 8192;
            if (candidate_0_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_0_1 - 1]) {
                    block_1 = candidate_0_1;
                }
            }
            unsigned int candidate_1_1 = block_1 + 4096;
            if (candidate_1_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_1_1 - 1]) {
                    block_1 = candidate_1_1;
                }
            }
            unsigned int candidate_2_1 = block_1 + 2048;
            if (candidate_2_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_2_1 - 1]) {
                    block_1 = candidate_2_1;
                }
            }
            unsigned int candidate_3_1 = block_1 + 1024;
            if (candidate_3_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_3_1 - 1]) {
                    block_1 = candidate_3_1;
                }
            }
            unsigned int candidate_4_1 = block_1 + 512;
            if (candidate_4_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_4_1 - 1]) {
                    block_1 = candidate_4_1;
                }
            }
            unsigned int candidate_5_1 = block_1 + 256;
            if (candidate_5_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_5_1 - 1]) {
                    block_1 = candidate_5_1;
                }
            }
            unsigned int candidate_6_1 = block_1 + 128;
            if (candidate_6_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_6_1 - 1]) {
                    block_1 = candidate_6_1;
                }
            }
            unsigned int candidate_7_1 = block_1 + 64;
            if (candidate_7_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_7_1 - 1]) {
                    block_1 = candidate_7_1;
                }
            }
            unsigned int candidate_8_1 = block_1 + 32;
            if (candidate_8_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_8_1 - 1]) {
                    block_1 = candidate_8_1;
                }
            }
            unsigned int candidate_9_1 = block_1 + 16;
            if (candidate_9_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_9_1 - 1]) {
                    block_1 = candidate_9_1;
                }
            }
            unsigned int candidate_10_1 = block_1 + 8;
            if (candidate_10_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_10_1 - 1]) {
                    block_1 = candidate_10_1;
                }
            }
            unsigned int candidate_11_1 = block_1 + 4;
            if (candidate_11_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_11_1 - 1]) {
                    block_1 = candidate_11_1;
                }
            }
            unsigned int candidate_12_1 = block_1 + 2;
            if (candidate_12_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_12_1 - 1]) {
                    block_1 = candidate_12_1;
                }
            }
            unsigned int candidate_13_1 = block_1 + 1;
            if (candidate_13_1 <= 4) {
                if (target_0 >= cost_prefix[candidate_13_1 - 1]) {
                    block_1 = candidate_13_1;
                }
            }
            unsigned int cost_before_1 = 0;
            unsigned int work_before_1 = 0;
            if (block_1 > 0) {
                cost_before_1 = cost_prefix[block_1 - 1];
                work_before_1 = work_prefix[block_1 - 1];
            }
            unsigned int _max_2 = ((target_0 - cost_before_1) > (1) ? (target_0 - cost_before_1) : (1));
            unsigned int _min_7 = ((_max_2 - 1) < (work_prefix[block_1] - work_before_1 - 1) ? (_max_2 - 1) : (work_prefix[block_1] - work_before_1 - 1));
            split_2 = _min_7;
            coordinate_3 = work_before_1 + split_2;
        }
        Metadata[tid * 2] = block;
        Metadata[tid * 2 + 1] = split;
        Metadata[296 + tid] = coordinate_3 - coordinate;
    }
}

} // extern "C"
