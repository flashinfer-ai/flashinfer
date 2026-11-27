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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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




__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_trtllm_mla_blackwell_a310b2fcce209bd2556b(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ sinks, int num_q_heads, int num_split, float bmm2_scale, int enable_sink, int* __restrict__ seq_lens, int q_len, int table_width, int page_shift)
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
    float* split_weights = reinterpret_cast<float*>(smem_raw + 1024);
    const int split_weights_addr = smem + 1024;
    float* merged_lse = reinterpret_cast<float*>(smem_raw + 1280);
    const int merged_lse_addr = smem + 1280;
    __nv_bfloat16* stage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int stage_addr = smem + 1536;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stage_full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (declaration order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (batch_idx * num_q_heads + head_idx) * num_split;
    int live_request = 0;
    int live_keys = 0;
    int live = 0;
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    live_request = (int)((unsigned int)batch_idx / (unsigned int)q_len);
    live_keys = seq_lens[live_request] - q_len + (int)((unsigned int)batch_idx % (unsigned int)q_len) + 1;
    int _min_1 = ((live_keys) < (table_width << 5 + page_shift) ? (live_keys) : (table_width << 5 + page_shift));
    int _max_0 = ((_min_1) > (0) ? (_min_1) : (0));
    live_keys = _max_0;
    int live_tiles = (live_keys + 128 - 1) / 128;
    int live_run = (((live_tiles + num_split - 1) / num_split) > (1) ? ((live_tiles + num_split - 1) / num_split) : (1));
    int _min_2 = (((live_tiles + live_run - 1) / live_run) < (num_split) ? ((live_tiles + live_run - 1) / live_run) : (num_split));
    live = _min_2;
    if (warp == 1) {
        if (elect_sync()) {
            asm volatile("fence.proxy.async.global;" ::: "memory");
            #pragma unroll
            for (int c = 0; c < 8; c++) {
                if (c * 8 < live) {
                    int chunk_left = live - c * 8;
                    int chunk_splits = ((chunk_left >= 8) ? 8 : chunk_left);
                    int chunk_bytes = chunk_splits * 1024;
                    mbarrier_arrive_expect_tx(stage_full_addr + (c) * 8, chunk_bytes);
                    asm volatile(
                        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                        "[%0], [%1], %2, [%3];"
                        :: "r"(stage_addr + (unsigned int)(c * 8192)), "l"(reinterpret_cast<const uint8_t*>(partial_O + (stat_base + c * 8) * 512)), "r"((uint32_t)(chunk_bytes)), "r"(stage_full_addr + (c) * 8)
                        : "memory");
                }
            }
        }
    }
    float sink_value = 0.0f;
    if (enable_sink != 0) {
        sink_value = sinks[head_idx];
    }
    float lane_lse = -CUDART_INF_F;
    if (warp == 0) {
        if (lane < live) {
            lane_lse = partial_lse[stat_base + lane];
        }
    }
    float lane_lse_hi[1];
    #pragma unroll
    for (int h = 0; h < 1; h++) {
        lane_lse_hi[h] = -CUDART_INF_F;
        if (warp == 0) {
            if (lane + (h + 1) * 32 < live) {
                lane_lse_hi[h] = partial_lse[stat_base + lane + (h + 1) * 32];
            }
        }
    }
    if (warp == 0) {
        float local_m[1];
        float local_w[1];
        float thread_max = -CUDART_INF_F;
        #pragma unroll
        for (int i = 0; i < 1; i++) {
            int split_idx = lane + i * 32;
            local_m[i] = -CUDART_INF_F;
            local_w[i] = 0.0f;
            if (split_idx < live) {
                local_m[i] = lane_lse;
                thread_max = max_noftz(thread_max, local_m[i]);
            }
        }
        float local_m_hi[1];
        float local_w_hi[1];
        #pragma unroll
        for (int h_1 = 0; h_1 < 1; h_1++) {
            int split_hi = lane + (h_1 + 1) * 32;
            local_m_hi[h_1] = -CUDART_INF_F;
            local_w_hi[h_1] = 0.0f;
            if (split_hi < live) {
                local_m_hi[h_1] = lane_lse_hi[h_1];
                thread_max = max_noftz(thread_max, local_m_hi[h_1]);
            }
        }
        float global_max = thread_max;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            global_max = max_noftz(global_max, __shfl_xor_sync(0xFFFFFFFF, global_max, offset));
        float thread_sum = 0.0f;
        #pragma unroll
        for (int i_1 = 0; i_1 < 1; i_1++) {
            int split_idx_1 = lane + i_1 * 32;
            if (split_idx_1 < live) {
                float _exp2_3 = approx_exp2(local_m[i_1] - global_max);
                local_w[i_1] = ((local_m[i_1] == -CUDART_INF_F) ? 0.0f : _exp2_3);
                thread_sum = thread_sum + local_w[i_1];
            }
        }
        #pragma unroll
        for (int h_2 = 0; h_2 < 1; h_2++) {
            int split_hi_1 = lane + (h_2 + 1) * 32;
            if (split_hi_1 < live) {
                float _exp2_4 = approx_exp2(local_m_hi[h_2] - global_max);
                local_w_hi[h_2] = ((local_m_hi[h_2] == -CUDART_INF_F) ? 0.0f : _exp2_4);
                thread_sum = thread_sum + local_w_hi[h_2];
            }
        }
        float global_sum = thread_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            global_sum += __shfl_xor_sync(0xFFFFFFFF, global_sum, offset);
        float _rcp_5 = approx_rcp(global_sum);
        float inv_sum = ((global_sum > 0.0f) ? _rcp_5 : 0.0f);
        if (lane == 0) {
            float _log2_6;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_6) : "f"(global_sum));
            merged_lse[0] = global_max + _log2_6;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 1; i_2++) {
            int split_idx_2 = lane + i_2 * 32;
            if (split_idx_2 < live) {
                split_weights[split_idx_2] = local_w[i_2] * inv_sum;
            }
        }
        #pragma unroll
        for (int h_3 = 0; h_3 < 1; h_3++) {
            int split_hi_2 = lane + (h_3 + 1) * 32;
            if (split_hi_2 < live) {
                split_weights[split_hi_2] = local_w_hi[h_3] * inv_sum;
            }
        }
    }
    __syncthreads();
    float output_scale = bmm2_scale;
    if (enable_sink != 0) {
        float sink_exp = approx_exp2(__fmaf_rn(sink_value, 1.442695f, -merged_lse[0]));
        float _rcp_7 = approx_rcp(1.0f + sink_exp);
        output_scale = output_scale * _rcp_7;
    }
    int po_head_base = stat_base * 512;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[4];
    int d_base = tid * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        acc[e] = 0.0f;
    }
    #pragma unroll
    for (int c_1 = 0; c_1 < 8; c_1++) {
        if (c_1 * 8 < live) {
            mbarrier_wait(stage_full_addr + (c_1) * 8, 0);
            #pragma unroll
            for (int cs = 0; cs < 8; cs++) {
                if (c_1 * 8 + cs < live) {
                    int stage_split = c_1 * 8 + cs;
                    float stage_weight = split_weights[stage_split];
                    unsigned int _stage_words[2];
                    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_stage_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&_stage_words[(0) + 1]))
                        : "r"(stage_addr + (unsigned int)(stage_split * 1024 + tid * 8)));
                    float _unpack_bf16_1[4];
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_stage_words[_pair]) << 16);
                        (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_stage_words[_pair]) & 0xffff0000u);
                    }
                    #pragma unroll
                    for (int e_1 = 0; e_1 < 4; e_1++) {
                        acc[e_1] = __fmaf_rn(stage_weight, _unpack_bf16_1[e_1], acc[e_1]);
                    }
                }
            }
        }
    }
    if (live < num_split) {
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            acc[e_2] = ((__as_u32(acc[e_2]) << 1 == 0) ? 0.0f : acc[e_2]);
        }
    }
    float _scaled_0[4];
    const float2 _scale2_8 = {output_scale, output_scale};
    #pragma unroll
    for (int _ls = 0; _ls < 2; _ls++)
        reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_ls], _scale2_8);
    unsigned int _packed_0[2];
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
        _packed_0[_lp] = *(uint32_t*)&_bf2;
    }
    #pragma unroll
    for (int _lp = 0; _lp < 1; _lp++) {
        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
        _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
    }
    reinterpret_cast<int2*>(O + (o_head_base + d_base))[0] = reinterpret_cast<int2*>(_packed_0)[0];
    {
        asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
