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
#if defined(__CUDACC_RTC__) || defined(_MSC_VER)
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

#define USE_PDL 1

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


__device__ __forceinline__ void mbarrier_init(int mbar_addr, int count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
        :: "r"(mbar_addr), "r"(count) : "memory");
}



// CTA-local pipelines have short, resident producer/consumer edges.  Omitting
// suspendTimeHint keeps a miss on the lightweight TRYWAIT retry path; the
// explicit loop still makes this helper blocking until acquire succeeds.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}




// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}



extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_trtllm_mla_blackwell_3fe1f0c71bfd4d0f4835(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_stats, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int num_split, int num_reduce_ctas, int num_queries, float bmm2_scale)
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

    const int mbar_base = smem;
    #define stage_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    float* table = reinterpret_cast<float*>(smem_raw + 1024);
    const int table_addr = smem + 1024;
    float* head_ml = reinterpret_cast<float*>(smem_raw + 3072);
    const int head_ml_addr = smem + 3072;
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int stage_addr = smem + 4096;

    // Mbarrier init (14 barriers at smem_raw[0..112))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stage_full: 14 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    const int role_token = 0;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int hd_row = blockIdx.y;
    int hd_cta = blockIdx.x;
    int hd_heads = (128 - hd_cta + num_reduce_ctas - 1) / num_reduce_ctas;
    int hd_items = hd_heads * num_split;
    int hd_chunks = (hd_items + 8 - 1) / 8;
    int hd_split_base = hd_row * num_split;
    if (warp >= 1) {
        if (elect_sync()) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            #pragma unroll 1
            for (int hd_g = 0; hd_g < hd_chunks; hd_g++) {
                if (warp == 1 + hd_g % 3) {
                    int hd_g_left = hd_items - hd_g * 8;
                    int hd_g_items = ((hd_g_left >= 8) ? 8 : hd_g_left);
                    mbarrier_arrive_expect_tx(stage_full_addr + ((int)((unsigned int)hd_g % 14)) * 8, hd_g_items * 1024);
                    #pragma unroll
                    for (int hd_k = 0; hd_k < 8; hd_k++) {
                        if (hd_k < hd_g_items) {
                            int hd_c_item = hd_g * 8 + hd_k;
                            int hd_c_local = hd_c_item / num_split;
                            int hd_c_split = hd_c_item - hd_c_local * num_split;
                            int hd_c_head = hd_cta + hd_c_local * num_reduce_ctas;
                            asm volatile(
                                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                                "[%0], [%1], %2, [%3];"
                                :: "r"(stage_addr + (unsigned int)(hd_g * 8192 + hd_k * 1024)), "l"(reinterpret_cast<const uint8_t*>(partial_O + ((hd_split_base + hd_c_split) * 128 + hd_c_head) * 512)), "r"((uint32_t)(1024)), "r"(stage_full_addr + ((int)((unsigned int)hd_g % 14)) * 8)
                                : "memory");
                        }
                    }
                }
            }
        }
    }
    if (warp == 0) {
        float hd_stats[16];
        #pragma unroll
        for (int hd_q = 0; hd_q < 16; hd_q++) {
            hd_stats[hd_q] = 0.0f;
        }
        #pragma unroll
        for (int hd_q_1 = 0; hd_q_1 < 8; hd_q_1++) {
            int hd_p_item = lane + hd_q_1 * 32;
            if (hd_p_item < hd_items) {
                int hd_p_local = hd_p_item / num_split;
                int hd_p_split = hd_p_item - hd_p_local * num_split;
                int hd_p_head = hd_cta + hd_p_local * num_reduce_ctas;
                {
                    float2 _v2_0 = *reinterpret_cast<const float2*>(partial_stats + ((hd_split_base + hd_p_split) * 128 + hd_p_head) * 2 + 0);
                    hd_stats[2 * hd_q_1] = _v2_0.x;
                    hd_stats[2 * hd_q_1 + 1] = _v2_0.y;
                }
            }
        }
        #pragma unroll
        for (int hd_q_2 = 0; hd_q_2 < 8; hd_q_2++) {
            int hd_p_item_1 = lane + hd_q_2 * 32;
            if (hd_p_item_1 < hd_items) {
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(table_addr + (unsigned int)(hd_p_item_1 * 8)), "f"(hd_stats[2 * hd_q_2]), "f"(hd_stats[2 * hd_q_2 + 1]) : "memory");
            }
        }
        __syncwarp();
        #pragma unroll
        for (int hd_i = 0; hd_i < 128 / 32; hd_i++) {
            int hd_local = lane + hd_i * 32;
            if (hd_local < hd_heads) {
                float hd_merged_max = -CUDART_INF_F;
                float hd_merged_sum = 0.0f;
                #pragma unroll 4
                for (int hd_split = 0; hd_split < num_split; hd_split++) {
                    int hd_t_item = hd_local * num_split + hd_split;
                    float hd_ml[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&hd_ml[0])), "=r"(*reinterpret_cast<uint32_t*>(&hd_ml[(0) + 1]))
                        : "r"(table_addr + (unsigned int)(hd_t_item * 8)));
                    float hd_local_max = hd_ml[0];
                    float hd_local_sum = hd_ml[1];
                    float hd_next_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(hd_next_max) : "f"(hd_merged_max), "f"(hd_local_max));
                    float _exp2_1 = approx_exp2(hd_merged_max - hd_next_max);
                    float hd_prior_scale = ((hd_merged_max != -CUDART_INF_F) ? _exp2_1 : 0.0f);
                    float _exp2_2 = approx_exp2(hd_local_max - hd_next_max);
                    float hd_local_scale = ((hd_local_max != -CUDART_INF_F) ? _exp2_2 : 0.0f);
                    asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(table_addr + (unsigned int)(hd_t_item * 8)), "f"(hd_prior_scale), "f"(hd_local_scale) : "memory");
                    hd_merged_sum = __fmaf_rn(hd_merged_sum, hd_prior_scale, hd_local_sum * hd_local_scale);
                    hd_merged_max = hd_next_max;
                }
                asm volatile("st.shared.v2.f32 [%0], {%1,%2};" :: "r"(head_ml_addr + (unsigned int)(hd_local * 8)), "f"(hd_merged_max), "f"(hd_merged_sum) : "memory");
            }
        }
    }
    __syncthreads();
    float hd_acc[4];
    #pragma unroll
    for (int hd_e = 0; hd_e < 4; hd_e++) {
        hd_acc[hd_e] = 0.0f;
    }
    int hd_cur_local = 0;
    int hd_cur_split = 0;
    #pragma unroll 1
    for (int hd_g_1 = 0; hd_g_1 < hd_chunks; hd_g_1++) {
        int _stage_stage_full = (int)((unsigned int)hd_g_1 % 14);
        int _parity_stage_full = (int)((unsigned int)hd_g_1 / 14 % 2);
        mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
        #pragma unroll
        for (int hd_k_1 = 0; hd_k_1 < 8; hd_k_1++) {
            int hd_e_item = hd_g_1 * 8 + hd_k_1;
            if (hd_e_item < hd_items) {
                float hd_scales[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&hd_scales[0])), "=r"(*reinterpret_cast<uint32_t*>(&hd_scales[(0) + 1]))
                    : "r"(table_addr + (unsigned int)(hd_e_item * 8)));
                unsigned int hd_words[2];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&hd_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&hd_words[(0) + 1]))
                    : "r"(stage_addr + (unsigned int)(hd_g_1 * 8192 + hd_k_1 * 1024 + tid * 8)));
                float hd_x[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&hd_x[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(hd_words[_pair]) << 16);
                    (&hd_x[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(hd_words[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int hd_e_1 = 0; hd_e_1 < 4; hd_e_1++) {
                    hd_acc[hd_e_1] = __fmaf_rn(hd_acc[hd_e_1], hd_scales[0], hd_x[hd_e_1] * hd_scales[1]);
                }
                if (hd_cur_split == num_split - 1) {
                    float hd_final[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&hd_final[0])), "=r"(*reinterpret_cast<uint32_t*>(&hd_final[(0) + 1]))
                        : "r"(head_ml_addr + (unsigned int)(hd_cur_local * 8)));
                    float _rcp_3 = approx_rcp(hd_final[1]);
                    float hd_inv_sum = ((hd_final[1] > 0.0f) ? _rcp_3 : 0.0f);
                    int hd_head = hd_cta + hd_cur_local * num_reduce_ctas;
                    float hd_scaled[4];
                    const float2 _scale2_4 = {hd_inv_sum * bmm2_scale, hd_inv_sum * bmm2_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        reinterpret_cast<float2*>(hd_scaled)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(hd_acc)[_ls], _scale2_4);
                    unsigned int hd_packed[2];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hd_scaled[_lp*2 + 0], hd_scaled[_lp*2+1 + 0]));
                        hd_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(hd_scaled[_lp*2 + 2], hd_scaled[_lp*2+1 + 2]));
                        hd_packed[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int2*>(O + ((hd_row * 128 + hd_head) * 512 + tid * 4))[0] = reinterpret_cast<int2*>(hd_packed)[0];
                    if (tid == 0) {
                        float _log2_5;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(hd_final[1]));
                        LSE[hd_row * 128 + hd_head] = ((hd_final[1] > 0.0f) ? hd_final[0] + _log2_5 : -CUDART_INF_F);
                    }
                    #pragma unroll
                    for (int hd_e_2 = 0; hd_e_2 < 4; hd_e_2++) {
                        hd_acc[hd_e_2] = 0.0f;
                    }
                    hd_cur_split = 0;
                    hd_cur_local = hd_cur_local + 1;
                } else {
                    hd_cur_split = hd_cur_split + 1;
                }
            }
        }
    }
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
