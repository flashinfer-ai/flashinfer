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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_RED_FOLD_OFF 0
#define SMEM_RED_FOLD_STAGE_BYTES 6144
#define SMEM_RED_FOLD_STRIDE 6144
#define SMEM_GRP_FOLD_OFF 6144
#define SMEM_GRP_FOLD_STAGE_BYTES 1024
#define SMEM_GRP_FOLD_STRIDE 1024
#define SMEM_STAGE_OFF 7168
#define SMEM_STAGE_STAGE_BYTES 196608
#define SMEM_STAGE_STRIDE 196608
#define SMEM_TOTAL 203776
#define THREADS 384

#include <math_constants.h>

extern "C" {

__global__ __launch_bounds__(384) void
kernel_cake_rmsnorm_train_1863320f3158d35eead6(float* __restrict__ partial, float* __restrict__ dw, int n_chunks)
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

    // Kernel setup ops
    float* red_fold = reinterpret_cast<float*>(smem_raw + 0);
    const int red_fold_addr = smem + 0;
    float* grp_fold = reinterpret_cast<float*>(smem_raw + 6144);
    const int grp_fold_addr = smem + 6144;
    uint8_t* stage = reinterpret_cast<uint8_t*>(smem_raw + 7168);
    const int stage_addr = smem + 7168;

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    int s = bid;
    int rsub = tid / 8;
    int cg = tid - rsub * 8;
    int col = s * 32 + cg * 4;
    float a[4];
    float c[4];
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        a[e] = 0.0f;
        c[e] = 0.0f;
    }
    int n_trips = (n_chunks + 47) / 48;
    int n_waves = (n_trips + 15) / 16;
    unsigned int slot0 = (unsigned int)stage_addr + (unsigned int)tid * 16;
    unsigned int zero_off = 0;
    #pragma unroll
    for (int d = 0; d < 16; d++) {
        int cc = rsub + d * 48;
        long long src = (long long)cc * 1536 + (long long)col;
        #pragma unroll
        for (int q = 0; q < 1; q++) {
            asm volatile(
                "{\n\t"
                ".reg .pred p;\n\t"
                "setp.ne.b32 p, %0, 0;\n\t"
                "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                "}"
                :: "r"((cc < n_chunks) ? 1 : 0), "r"(slot0 + zero_off + (unsigned int)(d * 384 * 4 * 4 + q * 16)), "l"(partial + (src + (long long)(q * 4))));
        }
    }
    asm volatile("cp.async.commit_group;");
    for (int w = 1; w < n_waves; w++) {
        unsigned int buf_w = (unsigned int)(w & 1) * 98304;
        unsigned int buf_p = (unsigned int)(w - 1 & 1) * 98304;
        #pragma unroll
        for (int d_1 = 0; d_1 < 16; d_1++) {
            int cc_1 = rsub + (w * 16 + d_1) * 48;
            long long src_1 = (long long)cc_1 * 1536 + (long long)col;
            #pragma unroll
            for (int q_1 = 0; q_1 < 1; q_1++) {
                asm volatile(
                    "{\n\t"
                    ".reg .pred p;\n\t"
                    "setp.ne.b32 p, %0, 0;\n\t"
                    "@p cp.async.cg.shared::cta.global [%1], [%2], 16;\n\t"
                    "}"
                    :: "r"((cc_1 < n_chunks) ? 1 : 0), "r"(slot0 + buf_w + (unsigned int)(d_1 * 384 * 4 * 4 + q_1 * 16)), "l"(partial + (src_1 + (long long)(q_1 * 4))));
            }
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 1;");
        unsigned int pw[64];
        #pragma unroll
        for (int d_2 = 0; d_2 < 16; d_2++) {
            #pragma unroll
            for (int q_2 = 0; q_2 < 1; q_2++) {
                __int128_t _smem_raw_0;
                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_0) : "r"(slot0 + buf_p + (unsigned int)(d_2 * 384 * 4 * 4 + q_2 * 16)));
                pw[d_2 * 4 + q_2 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[0];
                pw[(d_2 * 4 + q_2 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[1];
                pw[(d_2 * 4 + q_2 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[2];
                pw[(d_2 * 4 + q_2 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_0)[3];
            }
        }
        #pragma unroll
        for (int d_3 = 0; d_3 < 16; d_3++) {
            int cc_2 = rsub + ((w - 1) * 16 + d_3) * 48;
            if (cc_2 < n_chunks) {
                #pragma unroll
                for (int e_1 = 0; e_1 < 4; e_1++) {
                    float pf = __uint_as_float(pw[d_3 * 4 + e_1]);
                    float y = __fmaf_rn(pf, 1.0f, -c[e_1]);
                    float t = a[e_1] + y;
                    c[e_1] = t - a[e_1] - y;
                    a[e_1] = t;
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;");
    int last = n_waves - 1;
    unsigned int buf_l = (unsigned int)(last & 1) * 98304;
    unsigned int pw_1[64];
    #pragma unroll
    for (int d_4 = 0; d_4 < 16; d_4++) {
        #pragma unroll
        for (int q_3 = 0; q_3 < 1; q_3++) {
            __int128_t _smem_raw_1;
            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_1) : "r"(slot0 + buf_l + (unsigned int)(d_4 * 384 * 4 * 4 + q_3 * 16)));
            pw_1[d_4 * 4 + q_3 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[0];
            pw_1[(d_4 * 4 + q_3 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[1];
            pw_1[(d_4 * 4 + q_3 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[2];
            pw_1[(d_4 * 4 + q_3 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[3];
        }
    }
    #pragma unroll
    for (int d_5 = 0; d_5 < 16; d_5++) {
        int cc_3 = rsub + (last * 16 + d_5) * 48;
        if (cc_3 < n_chunks) {
            #pragma unroll
            for (int e_2 = 0; e_2 < 4; e_2++) {
                float pf_1 = __uint_as_float(pw_1[d_5 * 4 + e_2]);
                float y_1 = __fmaf_rn(pf_1, 1.0f, -c[e_2]);
                float t_1 = a[e_2] + y_1;
                c[e_2] = t_1 - a[e_2] - y_1;
                a[e_2] = t_1;
            }
        }
    }
    #pragma unroll
    for (int e_3 = 0; e_3 < 4; e_3++) {
        red_fold[rsub * 32 + cg * 4 + e_3] = a[e_3] - c[e_3];
    }
    __syncthreads();
    if (tid < 64) {
        int gi = tid / 8;
        int lane_1 = tid - gi * 8;
        float o[4];
        float oc[4];
        #pragma unroll
        for (int e_4 = 0; e_4 < 4; e_4++) {
            o[e_4] = 0.0f;
            oc[e_4] = 0.0f;
        }
        #pragma unroll
        for (int k = 0; k < 6; k++) {
            #pragma unroll
            for (int e_5 = 0; e_5 < 4; e_5++) {
                float d_6 = red_fold[(gi * 6 + k) * 32 + lane_1 * 4 + e_5];
                float y_2 = __fmaf_rn(d_6, 1.0f, -oc[e_5]);
                float t_2 = o[e_5] + y_2;
                oc[e_5] = t_2 - o[e_5] - y_2;
                o[e_5] = t_2;
            }
        }
        #pragma unroll
        for (int e_6 = 0; e_6 < 4; e_6++) {
            grp_fold[gi * 32 + lane_1 * 4 + e_6] = o[e_6] - oc[e_6];
        }
    }
    __syncthreads();
    if (tid < 8) {
        float o2[4];
        float oc2[4];
        #pragma unroll
        for (int e_7 = 0; e_7 < 4; e_7++) {
            o2[e_7] = 0.0f;
            oc2[e_7] = 0.0f;
        }
        #pragma unroll
        for (int gk = 0; gk < 8; gk++) {
            #pragma unroll
            for (int e_8 = 0; e_8 < 4; e_8++) {
                float d2 = grp_fold[gk * 32 + tid * 4 + e_8];
                float y_3 = __fmaf_rn(d2, 1.0f, -oc2[e_8]);
                float t_3 = o2[e_8] + y_3;
                oc2[e_8] = t_3 - o2[e_8] - y_3;
                o2[e_8] = t_3;
            }
        }
        #pragma unroll
        for (int e_9 = 0; e_9 < 4; e_9++) {
            o2[e_9] = o2[e_9] - oc2[e_9];
        }
        int ocol2 = s * 32 + tid * 4;
        {
            float4 _v4 = make_float4(o2[0 + 0], o2[0 + 1], o2[0 + 2], o2[0 + 3]);
            *reinterpret_cast<float4*>(dw + (unsigned long long)ocol2) = _v4;
        }
    }
    __syncthreads();
}

} // extern "C"
