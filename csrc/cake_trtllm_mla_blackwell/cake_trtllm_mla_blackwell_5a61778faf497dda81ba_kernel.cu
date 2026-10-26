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
#define SMEM_SPLIT_WEIGHTS_OFF 0
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 128
#define SMEM_SPLIT_WEIGHTS_STRIDE 128
#define SMEM_MERGED_LSE_OFF 128
#define SMEM_MERGED_LSE_STAGE_BYTES 4
#define SMEM_MERGED_LSE_STRIDE 4
#define SMEM_TOTAL 256
#define THREADS 128
#define USE_PDL 1

#include <math_constants.h>

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

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_trtllm_mla_blackwell_5a61778faf497dda81ba(__nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, __nv_bfloat16* __restrict__ O, float* __restrict__ sinks, int num_q_heads, int num_split, float bmm2_scale, int enable_sink)
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
    float* split_weights = reinterpret_cast<float*>(smem_raw + 0);
    const int split_weights_addr = smem + 0;
    float* merged_lse = reinterpret_cast<float*>(smem_raw + 128);
    const int merged_lse_addr = smem + 128;

    // === Task calls (declaration order) ===
    const int wg_dummy = 0;
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int stat_base = (batch_idx * num_q_heads + head_idx) * num_split;
    float sink_value = 0.0f;
    if (enable_sink != 0) {
        sink_value = sinks[head_idx];
    }
    {
        asm volatile("griddepcontrol.wait;" ::: "memory");
    }
    int prefetch_base = stat_base * 512 + tid * 4;
    float prefetched[32];
    if (num_split == 8) {
        #pragma unroll
        for (int ps = 0; ps < 8; ps++) {
            float _vec_load_0[4];
            {
                uint2 _vld_0;
                _vld_0 = *reinterpret_cast<const uint2*>(partial_O + (prefetch_base + ps * 512) + 0);
                uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&_vec_load_0[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                    (&_vec_load_0[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                }
            }
            #pragma unroll
            for (int pe = 0; pe < 4; pe++) {
                prefetched[ps * 4 + pe] = _vec_load_0[pe];
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
            if (split_idx < num_split) {
                local_m[i] = partial_lse[stat_base + split_idx];
                thread_max = max_noftz(thread_max, local_m[i]);
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
            if (split_idx_1 < num_split) {
                float _exp2_1 = approx_exp2(local_m[i_1] - global_max);
                local_w[i_1] = ((local_m[i_1] == -CUDART_INF_F) ? 0.0f : _exp2_1);
                thread_sum = thread_sum + local_w[i_1];
            }
        }
        float global_sum = thread_sum;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            global_sum += __shfl_xor_sync(0xFFFFFFFF, global_sum, offset);
        float _rcp_2 = approx_rcp(global_sum);
        float inv_sum = ((global_sum > 0.0f) ? _rcp_2 : 0.0f);
        if (lane == 0) {
            float _log2_3;
            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(global_sum));
            merged_lse[0] = global_max + _log2_3;
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 1; i_2++) {
            int split_idx_2 = lane + i_2 * 32;
            if (split_idx_2 < num_split) {
                split_weights[split_idx_2] = local_w[i_2] * inv_sum;
            }
        }
    }
    __syncthreads();
    float output_scale = bmm2_scale;
    if (enable_sink != 0) {
        float sink_exp = approx_exp2(__fmaf_rn(sink_value, 1.442695f, -merged_lse[0]));
        float _rcp_4 = approx_rcp(1.0f + sink_exp);
        output_scale = output_scale * _rcp_4;
    }
    int po_head_base = stat_base * 512;
    int o_head_base = (batch_idx * num_q_heads + head_idx) * 512;
    float acc[4];
    int d_base = tid * 4;
    #pragma unroll
    for (int e = 0; e < 4; e++) {
        acc[e] = 0.0f;
    }
    int remaining_splits = ((num_split == 8) ? 0 : num_split);
    if (num_split == 8) {
        #pragma unroll
        for (int ps_1 = 0; ps_1 < 8; ps_1++) {
            float prefetch_weight = split_weights[ps_1];
            #pragma unroll
            for (int e_1 = 0; e_1 < 4; e_1++) {
                acc[e_1] = __fmaf_rn(prefetch_weight, prefetched[ps_1 * 4 + e_1], acc[e_1]);
            }
        }
    }
    #pragma unroll 4
    for (int s = 0; s < remaining_splits; s++) {
        float split_weight = split_weights[s];
        float _vec_load_1[4];
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(partial_O + (po_head_base + s * 512 + d_base) + 0);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                (&_vec_load_1[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&_vec_load_1[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int e_2 = 0; e_2 < 4; e_2++) {
            acc[e_2] = __fmaf_rn(split_weight, _vec_load_1[e_2], acc[e_2]);
        }
    }
    float _scaled_0[4];
    const float2 _scale2_6 = {output_scale, output_scale};
    #pragma unroll
    for (int _ls = 0; _ls < 2; _ls++)
        reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(acc)[_ls], _scale2_6);
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
}

} // extern "C"
