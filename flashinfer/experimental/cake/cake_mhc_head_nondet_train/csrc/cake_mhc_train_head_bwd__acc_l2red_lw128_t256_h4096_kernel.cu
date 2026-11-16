/*
 * Copyright (c) 2026 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_mhc_train_head_bwd__acc_l2red_lw128_t256_h4096(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, int T, CakeFastDivmod hv)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
#if __CUDA_ARCH__ == 1000
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);
#else
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);
#endif

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    unsigned int* _stage = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int stage_addr = smem + 0;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    unsigned int token = (unsigned int)bid / 2;
    unsigned int seg = (unsigned int)bid - token * 2;
    unsigned int col = seg * 256 + (unsigned int)tid;
    if (token < (unsigned int)T) {
        unsigned int grad[4];
        {
            uint32_t _uv4_0_0;
            uint32_t _uv4_0_1;
            uint32_t _uv4_0_2;
            uint32_t _uv4_0_3;
            asm volatile("ld.global.L2::cache_hint.v4.b32 {%0, %1, %2, %3}, [%4], %5;"
                : "=r"(_uv4_0_0), "=r"(_uv4_0_1), "=r"(_uv4_0_2), "=r"(_uv4_0_3) : "l"((const void*)(reinterpret_cast<unsigned int*>(dy + ((long long)token * 4096ll + (long long)col * 8ll)) + (0))), "l"(0x12F0000000000000ULL) : "memory");
            grad[0 + 0] = _uv4_0_0;
            grad[0 + 1] = _uv4_0_1;
            grad[0 + 2] = _uv4_0_2;
            grad[0 + 3] = _uv4_0_3;
        }
        unsigned int q[4];
        uint32_t quarter_0;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_0) : "r"(grad[0]), "r"(1048592000));
        q[0] = quarter_0;
        uint32_t quarter_1;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_1) : "r"(grad[1]), "r"(1048592000));
        q[1] = quarter_1;
        uint32_t quarter_2;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_2) : "r"(grad[2]), "r"(1048592000));
        q[2] = quarter_2;
        uint32_t quarter_3;
        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(quarter_3) : "r"(grad[3]), "r"(1048592000));
        q[3] = quarter_3;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(stage_addr + (unsigned int)tid * 16), "r"(q[0]), "r"(q[1]), "r"(q[2]), "r"(q[3]) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 0, 256;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                {
                    void* _cpred_dst_1 = reinterpret_cast<void*>(dx + ((long long)token * 16384ll + (long long)seg * 2048ll));
                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                        " [%0], [%1], %2;"
                        :: "l"(_cpred_dst_1), "r"(stage_addr), "r"((uint32_t)(4096))
                        : "memory");
                }
                {
                    void* _cpred_dst_2 = reinterpret_cast<void*>(dx + ((long long)token * 16384ll + 4096ll + (long long)seg * 2048ll));
                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                        " [%0], [%1], %2;"
                        :: "l"(_cpred_dst_2), "r"(stage_addr), "r"((uint32_t)(4096))
                        : "memory");
                }
                {
                    void* _cpred_dst_3 = reinterpret_cast<void*>(dx + ((long long)token * 16384ll + 8192ll + (long long)seg * 2048ll));
                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                        " [%0], [%1], %2;"
                        :: "l"(_cpred_dst_3), "r"(stage_addr), "r"((uint32_t)(4096))
                        : "memory");
                }
                {
                    void* _cpred_dst_4 = reinterpret_cast<void*>(dx + ((long long)token * 16384ll + 12288ll + (long long)seg * 2048ll));
                    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16"
                        " [%0], [%1], %2;"
                        :: "l"(_cpred_dst_4), "r"(stage_addr), "r"((uint32_t)(4096))
                        : "memory");
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
        }
    }
}

} // extern "C"
