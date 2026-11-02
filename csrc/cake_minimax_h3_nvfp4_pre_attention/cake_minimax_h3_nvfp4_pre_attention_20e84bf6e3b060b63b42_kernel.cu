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
#define SMEM_SMEM_PARTIAL_OFF 0
#define SMEM_SMEM_PARTIAL_STAGE_BYTES 16
#define SMEM_SMEM_PARTIAL_STRIDE 16
#define SMEM_TOTAL 128
#define THREADS 128
#define HIDDEN 5376
#define SCALE_COLS 336
#define PACKED_COLS 2688
#define LAUNCH_MIN_BLOCKS 4

#include <math_constants.h>

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

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_minimax_h3_nvfp4_pre_attention_20e84bf6e3b060b63b42(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, float* __restrict__ x_global_scale, uint8_t* __restrict__ activation_q, uint8_t* __restrict__ activation_sf, __nv_bfloat16* __restrict__ debug_adaln_bf16, int write_debug, float eps, int M, uint8_t* __restrict__ out_sf, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    float* smem_partial = reinterpret_cast<float*>(smem_raw + 0);
    const int smem_partial_addr = smem + 0;

    // === Task calls (declaration order) ===
    int thread = warp * 32 + lane;
    int row_in_cta = thread / 64;
    int t = thread % 64;
    int warp_in_row = warp % 2;
    int row = bid * 2 + row_in_cta;
    if (SCALE_STRIDE != 0) {
        if (ROWS_PER_DESTINATION % 128 != 0) {
            if (bid < 56 / HEADS_PER_DESTINATION) {
                unsigned int zero_pair[2];
                zero_pair[0] = 0;
                zero_pair[1] = 0;
                reinterpret_cast<int2*>(out_sf + ((unsigned long long)bid * (unsigned long long)SCALE_STRIDE + (unsigned long long)(ROWS_PER_DESTINATION / 128 * 1024 + thread * 8)))[0] = reinterpret_cast<int2*>(zero_pair)[0];
            }
        }
    }
    int table_row = -1;
    if (row < M) {
        table_row = adaln_index[row];
    }
    float global_scale = x_global_scale[0];
    unsigned int x_carrier[44];
    #pragma unroll
    for (int q = 0; q < 44; q++) {
        x_carrier[q] = 0;
    }
    if (row < M) {
        unsigned long long row_base = (unsigned long long)row * (unsigned long long)HIDDEN;
        #pragma unroll
        for (int k = 0; k < 10; k++) {
            {
                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + row_base + (unsigned long long)(k * 512) + (unsigned long long)(t * 8));
                uint4* _vdst_0 = reinterpret_cast<uint4*>(&x_carrier[4 * k]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_0[_blk] = _vptr_0[_blk];
                }
            }
        }
        if (t < 32) {
            {
                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + row_base + 5120 + (unsigned long long)(t * 8));
                uint4* _vdst_1 = reinterpret_cast<uint4*>(&x_carrier[40]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_1[_blk] = _vptr_1[_blk];
                }
            }
        }
    }
    float _unpack_bf16_0[88];
    #pragma unroll
    for (int _pair = 0; _pair < 44; _pair++) {
        (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(x_carrier[_pair]) << 16);
        (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(x_carrier[_pair]) & 0xffff0000u);
    }
    int row_valid = 0;
    int load_row = 0;
    if (table_row >= 0 && table_row < 9) {
        load_row = table_row;
        if (row < M) {
            row_valid = 1;
        }
    }
    unsigned long long table_base = (unsigned long long)load_row * (unsigned long long)HIDDEN;
    const __nv_bfloat16* w_ptr = x_norm_weight + (t * 8);
    const __nv_bfloat16* s_ptr = adaln_scale + (table_base + (unsigned long long)(t * 8));
    const __nv_bfloat16* b_ptr = adaln_shift + (table_base + (unsigned long long)(t * 8));
    float _vec_load_0[8];
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(w_ptr + 0 + 0);
        uint4 _vld_2[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_2[_blk] = _vptr_2[_blk];
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
    }
    unsigned int tab_s[4];
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(s_ptr + 0);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&tab_s[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_3[_blk] = _vptr_3[_blk];
        }
    }
    float _vec_load_1[8];
    {
        const uint4* _vptr_4 = reinterpret_cast<const uint4*>(b_ptr + 0 + 0);
        uint4 _vld_4[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_4[_blk] = _vptr_4[_blk];
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
            }
        }
    }
    float _vec_load_2[8];
    {
        const uint4* _vptr_5 = reinterpret_cast<const uint4*>(w_ptr + 512 + 0);
        uint4 _vld_5[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_5[_blk] = _vptr_5[_blk];
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
            }
        }
    }
    unsigned int tab_s_0[4];
    {
        const uint4* _vptr_6 = reinterpret_cast<const uint4*>(s_ptr + 512);
        uint4* _vdst_6 = reinterpret_cast<uint4*>(&tab_s_0[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_6[_blk] = _vptr_6[_blk];
        }
    }
    float _vec_load_3[8];
    {
        const uint4* _vptr_7 = reinterpret_cast<const uint4*>(b_ptr + 512 + 0);
        uint4 _vld_7[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_7[_blk] = _vptr_7[_blk];
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
            }
        }
    }
    float sum_sq = 0.0f;
    #pragma unroll
    for (int i = 0; i < 88; i++) {
        sum_sq = __fmaf_rn(_unpack_bf16_0[i], _unpack_bf16_0[i], sum_sq);
    }
    float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
    sum_sq += _shfl_xor_sync_8;
    float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
    sum_sq += _shfl_xor_sync_9;
    float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
    sum_sq += _shfl_xor_sync_10;
    float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
    sum_sq += _shfl_xor_sync_11;
    float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
    sum_sq += _shfl_xor_sync_12;
    if (lane == 0) {
        smem_partial[row_in_cta * 2 + warp_in_row] = sum_sq;
    }
    asm volatile("barrier.sync 1, 128;" ::: "memory");
    float total_sq = smem_partial[row_in_cta * 2];
    total_sq += smem_partial[row_in_cta * 2 + 1];
    float mean_sq = __fdiv_rn(total_sq, 5376.0f);
    float rstd = rsqrtf(mean_sq + eps);
    float2 rstd_pair = make_float2(rstd, rstd);
    float rcp_six = approx_rcp(6.0f);
    float rcp_global_scale = approx_rcp(global_scale);
    uint8_t* q_ptr = activation_q + ((unsigned long long)row * (unsigned long long)PACKED_COLS + (unsigned long long)(t * 4));
    uint8_t* sf_ptr = activation_sf + ((unsigned long long)(t / 2 % 4 + t / 8 * 512 + row % 32 * 16 + row % 128 / 32 * 4) + (unsigned long long)(row / 128) * (unsigned long long)(128 * SCALE_COLS));
    float scaled[8];
    unsigned int scale_plus_one[4];
    unsigned int one_pair = 1065369472;
    float adaln[8];
    float values[8];
    float normalized[8];
    float absolute[8];
    float quant_values[8];
    unsigned int packed[1];
    if (row_valid != 0) {
        float _vec_load_4[8];
        {
            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(w_ptr + 1024 + 0);
            uint4 _vld_13[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_13[_blk] = _vptr_13[_blk];
                uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) << 16);
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_13[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_1[4];
        {
            const uint4* _vptr_14 = reinterpret_cast<const uint4*>(s_ptr + 1024);
            uint4* _vdst_14 = reinterpret_cast<uint4*>(&tab_s_1[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_14[_blk] = _vptr_14[_blk];
            }
        }
        float _vec_load_5[8];
        {
            const uint4* _vptr_15 = reinterpret_cast<const uint4*>(b_ptr + 1024 + 0);
            uint4 _vld_15[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_15[_blk] = _vptr_15[_blk];
                uint32_t* _vpairs_15 = reinterpret_cast<uint32_t*>(&_vld_15[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) << 16);
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_15[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            int mul_offset = pair * 2;
            float2 _f32x2_mul_a_16 = make_float2(_unpack_bf16_0[mul_offset], _unpack_bf16_0[mul_offset + 1]);
            float2 x_rstd;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd) : "l"(*(const unsigned long long*)&_f32x2_mul_a_16), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_17 = make_float2(_vec_load_0[mul_offset], _vec_load_0[mul_offset + 1]);
            float2 x_rstd_w;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w) : "l"(*(const unsigned long long*)&x_rstd), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
            scaled[mul_offset] = x_rstd_w.x;
            scaled[mul_offset + 1] = x_rstd_w.y;
        }
        unsigned int scaled_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_1 = 0; q_1 < 4; q_1++) {
            uint32_t _bf16x2_add_0;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(tab_s[q_1]), "r"(one_pair));
            scale_plus_one[q_1] = _bf16x2_add_0;
        }
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
            int ada_offset = pair_1 * 2;
            float2 adaln_pair = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_1[ada_offset], _unpack_bf16_1[ada_offset + 1]), make_float2(_unpack_bf16_2[ada_offset], _unpack_bf16_2[ada_offset + 1]), make_float2(_vec_load_1[ada_offset], _vec_load_1[ada_offset + 1]));
            adaln[ada_offset] = adaln_pair.x;
            adaln[ada_offset + 1] = adaln_pair.y;
        }
        unsigned int adaln_bf16[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16[_pair]) << 16);
            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            values[j] = _unpack_bf16_3[j];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                normalized[j_1] = _unpack_bf16_1[j_1];
            }
            unsigned int _packed_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(t * 8)))[0] = reinterpret_cast<int4*>(_packed_0)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(t * 8)))[0] = reinterpret_cast<int4*>(_packed_1)[0];
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            absolute[j_2] = values[j_2];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax = max_noftz(amax, absolute[_lr]);
        }
        float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        amax = max_noftz(amax, _shfl_xor_sync_18);
        float sf_value = global_scale * (amax * rcp_six);
        uint16_t _e4m3x2_20;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_20) : "f"(0.0f), "f"(sf_value));
        uint8_t _cvt_satfinite_e4m3_f32_19 = (uint8_t)_e4m3x2_20;
        uint32_t _f16x2_decode_21;
        float sf_rounded;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_21) : "h"(_e4m3x2_20));
        uint16_t _f16_decode_21 = (uint16_t)_f16x2_decode_21;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded) : "h"(_f16_decode_21));
        float _rcp_22 = approx_rcp(sf_rounded * rcp_global_scale);
        float output_scale = fminf(_rcp_22, 3.4028235e+38f);
        float2 output_pair = make_float2(output_scale, output_scale);
        #pragma unroll
        for (int pair_2 = 0; pair_2 < 4; pair_2++) {
            int quant_offset = pair_2 * 2;
            float2 _f32x2_mul_a_23 = make_float2(values[quant_offset], values[quant_offset + 1]);
            float2 quant_pair;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&output_pair));
            quant_values[quant_offset] = quant_pair.x;
            quant_values[quant_offset + 1] = quant_pair.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_24 = q_ptr + 0;
            __builtin_assume(__isGlobal(_gst_24));
            *(reinterpret_cast<int*>(_gst_24) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_25 = sf_ptr + 0;
                __builtin_assume(__isGlobal(_gst_25));
                uint16_t _cvt_satfinite_e4m3x2_f32_26;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_26) : "f"(0.0f), "f"(sf_value));
                *reinterpret_cast<unsigned char*>(_gst_25) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_26 & 127));
            }
        }
        float _vec_load_6[8];
        {
            const uint4* _vptr_27 = reinterpret_cast<const uint4*>(w_ptr + 1536 + 0);
            uint4 _vld_27[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_27[_blk] = _vptr_27[_blk];
                uint32_t* _vpairs_27 = reinterpret_cast<uint32_t*>(&_vld_27[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) << 16);
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_27[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_2[4];
        {
            const uint4* _vptr_28 = reinterpret_cast<const uint4*>(s_ptr + 1536);
            uint4* _vdst_28 = reinterpret_cast<uint4*>(&tab_s_2[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_28[_blk] = _vptr_28[_blk];
            }
        }
        float _vec_load_7[8];
        {
            const uint4* _vptr_29 = reinterpret_cast<const uint4*>(b_ptr + 1536 + 0);
            uint4 _vld_29[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_29[_blk] = _vptr_29[_blk];
                uint32_t* _vpairs_29 = reinterpret_cast<uint32_t*>(&_vld_29[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) << 16);
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_29[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_3 = 0; pair_3 < 4; pair_3++) {
            int mul_offset_1 = pair_3 * 2;
            float2 _f32x2_mul_a_30 = make_float2(_unpack_bf16_0[8 + mul_offset_1], _unpack_bf16_0[8 + mul_offset_1 + 1]);
            float2 x_rstd_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_31 = make_float2(_vec_load_2[mul_offset_1], _vec_load_2[mul_offset_1 + 1]);
            float2 x_rstd_w_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_1) : "l"(*(const unsigned long long*)&x_rstd_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_31));
            scaled[mul_offset_1] = x_rstd_w_1.x;
            scaled[mul_offset_1 + 1] = x_rstd_w_1.y;
        }
        unsigned int scaled_bf16_3[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_3[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_3[_pair]) << 16);
            (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_3[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_2 = 0; q_2 < 4; q_2++) {
            uint32_t _bf16x2_add_1;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(tab_s_0[q_2]), "r"(one_pair));
            scale_plus_one[q_2] = _bf16x2_add_1;
        }
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_4 = 0; pair_4 < 4; pair_4++) {
            int ada_offset_1 = pair_4 * 2;
            float2 adaln_pair_1 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_4[ada_offset_1], _unpack_bf16_4[ada_offset_1 + 1]), make_float2(_unpack_bf16_5[ada_offset_1], _unpack_bf16_5[ada_offset_1 + 1]), make_float2(_vec_load_3[ada_offset_1], _vec_load_3[ada_offset_1 + 1]));
            adaln[ada_offset_1] = adaln_pair_1.x;
            adaln[ada_offset_1 + 1] = adaln_pair_1.y;
        }
        unsigned int adaln_bf16_4[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_4[_pair]) << 16);
            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_4[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            values[j_3] = _unpack_bf16_6[j_3];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_4 = 0; j_4 < 8; j_4++) {
                normalized[j_4] = _unpack_bf16_4[j_4];
            }
            unsigned int _packed_2[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(512 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_2)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_3[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(512 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_3)[0];
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            absolute[j_5] = values[j_5];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_5 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_5 = max_noftz(amax_5, absolute[_lr]);
        }
        float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, amax_5, 1);
        amax_5 = max_noftz(amax_5, _shfl_xor_sync_32);
        float sf_value_6 = global_scale * (amax_5 * rcp_six);
        uint16_t _e4m3x2_34;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_34) : "f"(0.0f), "f"(sf_value_6));
        uint8_t _cvt_satfinite_e4m3_f32_33 = (uint8_t)_e4m3x2_34;
        uint32_t _f16x2_decode_35;
        float sf_rounded_7;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_35) : "h"(_e4m3x2_34));
        uint16_t _f16_decode_35 = (uint16_t)_f16x2_decode_35;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_7) : "h"(_f16_decode_35));
        float _rcp_36 = approx_rcp(sf_rounded_7 * rcp_global_scale);
        float output_scale_8 = fminf(_rcp_36, 3.4028235e+38f);
        float2 output_pair_9 = make_float2(output_scale_8, output_scale_8);
        #pragma unroll
        for (int pair_5 = 0; pair_5 < 4; pair_5++) {
            int quant_offset_1 = pair_5 * 2;
            float2 _f32x2_mul_a_37 = make_float2(values[quant_offset_1], values[quant_offset_1 + 1]);
            float2 quant_pair_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&output_pair_9));
            quant_values[quant_offset_1] = quant_pair_1.x;
            quant_values[quant_offset_1 + 1] = quant_pair_1.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_38 = q_ptr + 256;
            __builtin_assume(__isGlobal(_gst_38));
            *(reinterpret_cast<int*>(_gst_38) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_39 = sf_ptr + 4096;
                __builtin_assume(__isGlobal(_gst_39));
                uint16_t _cvt_satfinite_e4m3x2_f32_40;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_40) : "f"(0.0f), "f"(sf_value_6));
                *reinterpret_cast<unsigned char*>(_gst_39) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_40 & 127));
            }
        }
        float _vec_load_8[8];
        {
            const uint4* _vptr_41 = reinterpret_cast<const uint4*>(w_ptr + 2048 + 0);
            uint4 _vld_41[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_41[_blk] = _vptr_41[_blk];
                uint32_t* _vpairs_41 = reinterpret_cast<uint32_t*>(&_vld_41[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) << 16);
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_41[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_10[4];
        {
            const uint4* _vptr_42 = reinterpret_cast<const uint4*>(s_ptr + 2048);
            uint4* _vdst_42 = reinterpret_cast<uint4*>(&tab_s_10[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_42[_blk] = _vptr_42[_blk];
            }
        }
        float _vec_load_9[8];
        {
            const uint4* _vptr_43 = reinterpret_cast<const uint4*>(b_ptr + 2048 + 0);
            uint4 _vld_43[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_43[_blk] = _vptr_43[_blk];
                uint32_t* _vpairs_43 = reinterpret_cast<uint32_t*>(&_vld_43[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) << 16);
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_43[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_6 = 0; pair_6 < 4; pair_6++) {
            int mul_offset_2 = pair_6 * 2;
            float2 _f32x2_mul_a_44 = make_float2(_unpack_bf16_0[16 + mul_offset_2], _unpack_bf16_0[16 + mul_offset_2 + 1]);
            float2 x_rstd_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_44), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_45 = make_float2(_vec_load_4[mul_offset_2], _vec_load_4[mul_offset_2 + 1]);
            float2 x_rstd_w_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_2) : "l"(*(const unsigned long long*)&x_rstd_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_45));
            scaled[mul_offset_2] = x_rstd_w_2.x;
            scaled[mul_offset_2 + 1] = x_rstd_w_2.y;
        }
        unsigned int scaled_bf16_11[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_11[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_11[_pair]) << 16);
            (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_11[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_3 = 0; q_3 < 4; q_3++) {
            uint32_t _bf16x2_add_2;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(tab_s_1[q_3]), "r"(one_pair));
            scale_plus_one[q_3] = _bf16x2_add_2;
        }
        float _unpack_bf16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
            int ada_offset_2 = pair_7 * 2;
            float2 adaln_pair_2 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_7[ada_offset_2], _unpack_bf16_7[ada_offset_2 + 1]), make_float2(_unpack_bf16_8[ada_offset_2], _unpack_bf16_8[ada_offset_2 + 1]), make_float2(_vec_load_5[ada_offset_2], _vec_load_5[ada_offset_2 + 1]));
            adaln[ada_offset_2] = adaln_pair_2.x;
            adaln[ada_offset_2 + 1] = adaln_pair_2.y;
        }
        unsigned int adaln_bf16_12[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_12[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_12[_pair]) << 16);
            (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_12[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            values[j_6] = _unpack_bf16_9[j_6];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_7 = 0; j_7 < 8; j_7++) {
                normalized[j_7] = _unpack_bf16_7[j_7];
            }
            unsigned int _packed_4[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_4[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1024 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_4)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_5[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_5[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1024 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_5)[0];
        }
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            absolute[j_8] = values[j_8];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_13 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_13 = max_noftz(amax_13, absolute[_lr]);
        }
        float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, amax_13, 1);
        amax_13 = max_noftz(amax_13, _shfl_xor_sync_46);
        float sf_value_14 = global_scale * (amax_13 * rcp_six);
        uint16_t _e4m3x2_48;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_48) : "f"(0.0f), "f"(sf_value_14));
        uint8_t _cvt_satfinite_e4m3_f32_47 = (uint8_t)_e4m3x2_48;
        uint32_t _f16x2_decode_49;
        float sf_rounded_15;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_49) : "h"(_e4m3x2_48));
        uint16_t _f16_decode_49 = (uint16_t)_f16x2_decode_49;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_15) : "h"(_f16_decode_49));
        float _rcp_50 = approx_rcp(sf_rounded_15 * rcp_global_scale);
        float output_scale_16 = fminf(_rcp_50, 3.4028235e+38f);
        float2 output_pair_17 = make_float2(output_scale_16, output_scale_16);
        #pragma unroll
        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
            int quant_offset_2 = pair_8 * 2;
            float2 _f32x2_mul_a_51 = make_float2(values[quant_offset_2], values[quant_offset_2 + 1]);
            float2 quant_pair_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&output_pair_17));
            quant_values[quant_offset_2] = quant_pair_2.x;
            quant_values[quant_offset_2 + 1] = quant_pair_2.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_52 = q_ptr + 512;
            __builtin_assume(__isGlobal(_gst_52));
            *(reinterpret_cast<int*>(_gst_52) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_53 = sf_ptr + 8192;
                __builtin_assume(__isGlobal(_gst_53));
                uint16_t _cvt_satfinite_e4m3x2_f32_54;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_54) : "f"(0.0f), "f"(sf_value_14));
                *reinterpret_cast<unsigned char*>(_gst_53) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_54 & 127));
            }
        }
        float _vec_load_10[8];
        {
            const uint4* _vptr_55 = reinterpret_cast<const uint4*>(w_ptr + 2560 + 0);
            uint4 _vld_55[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_55[_blk] = _vptr_55[_blk];
                uint32_t* _vpairs_55 = reinterpret_cast<uint32_t*>(&_vld_55[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) << 16);
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_55[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_18[4];
        {
            const uint4* _vptr_56 = reinterpret_cast<const uint4*>(s_ptr + 2560);
            uint4* _vdst_56 = reinterpret_cast<uint4*>(&tab_s_18[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_56[_blk] = _vptr_56[_blk];
            }
        }
        float _vec_load_11[8];
        {
            const uint4* _vptr_57 = reinterpret_cast<const uint4*>(b_ptr + 2560 + 0);
            uint4 _vld_57[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_57[_blk] = _vptr_57[_blk];
                uint32_t* _vpairs_57 = reinterpret_cast<uint32_t*>(&_vld_57[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) << 16);
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_57[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_9 = 0; pair_9 < 4; pair_9++) {
            int mul_offset_3 = pair_9 * 2;
            float2 _f32x2_mul_a_58 = make_float2(_unpack_bf16_0[24 + mul_offset_3], _unpack_bf16_0[24 + mul_offset_3 + 1]);
            float2 x_rstd_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_58), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_59 = make_float2(_vec_load_6[mul_offset_3], _vec_load_6[mul_offset_3 + 1]);
            float2 x_rstd_w_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_3) : "l"(*(const unsigned long long*)&x_rstd_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
            scaled[mul_offset_3] = x_rstd_w_3.x;
            scaled[mul_offset_3 + 1] = x_rstd_w_3.y;
        }
        unsigned int scaled_bf16_19[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_19[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_10[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_19[_pair]) << 16);
            (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_19[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_4 = 0; q_4 < 4; q_4++) {
            uint32_t _bf16x2_add_3;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(tab_s_2[q_4]), "r"(one_pair));
            scale_plus_one[q_4] = _bf16x2_add_3;
        }
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_10 = 0; pair_10 < 4; pair_10++) {
            int ada_offset_3 = pair_10 * 2;
            float2 adaln_pair_3 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_10[ada_offset_3], _unpack_bf16_10[ada_offset_3 + 1]), make_float2(_unpack_bf16_11[ada_offset_3], _unpack_bf16_11[ada_offset_3 + 1]), make_float2(_vec_load_7[ada_offset_3], _vec_load_7[ada_offset_3 + 1]));
            adaln[ada_offset_3] = adaln_pair_3.x;
            adaln[ada_offset_3 + 1] = adaln_pair_3.y;
        }
        unsigned int adaln_bf16_20[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_20[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_20[_pair]) << 16);
            (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_20[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_9 = 0; j_9 < 8; j_9++) {
            values[j_9] = _unpack_bf16_12[j_9];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_10 = 0; j_10 < 8; j_10++) {
                normalized[j_10] = _unpack_bf16_10[j_10];
            }
            unsigned int _packed_6[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_6[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1536 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_6)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_7[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_7[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1536 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_7)[0];
        }
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            absolute[j_11] = values[j_11];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_21 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_21 = max_noftz(amax_21, absolute[_lr]);
        }
        float _shfl_xor_sync_60 = __shfl_xor_sync(0xFFFFFFFF, amax_21, 1);
        amax_21 = max_noftz(amax_21, _shfl_xor_sync_60);
        float sf_value_22 = global_scale * (amax_21 * rcp_six);
        uint16_t _e4m3x2_62;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(sf_value_22));
        uint8_t _cvt_satfinite_e4m3_f32_61 = (uint8_t)_e4m3x2_62;
        uint32_t _f16x2_decode_63;
        float sf_rounded_23;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_63) : "h"(_e4m3x2_62));
        uint16_t _f16_decode_63 = (uint16_t)_f16x2_decode_63;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_23) : "h"(_f16_decode_63));
        float _rcp_64 = approx_rcp(sf_rounded_23 * rcp_global_scale);
        float output_scale_24 = fminf(_rcp_64, 3.4028235e+38f);
        float2 output_pair_25 = make_float2(output_scale_24, output_scale_24);
        #pragma unroll
        for (int pair_11 = 0; pair_11 < 4; pair_11++) {
            int quant_offset_3 = pair_11 * 2;
            float2 _f32x2_mul_a_65 = make_float2(values[quant_offset_3], values[quant_offset_3 + 1]);
            float2 quant_pair_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&output_pair_25));
            quant_values[quant_offset_3] = quant_pair_3.x;
            quant_values[quant_offset_3 + 1] = quant_pair_3.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_66 = q_ptr + 768;
            __builtin_assume(__isGlobal(_gst_66));
            *(reinterpret_cast<int*>(_gst_66) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_67 = sf_ptr + 12288;
                __builtin_assume(__isGlobal(_gst_67));
                uint16_t _cvt_satfinite_e4m3x2_f32_68;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_68) : "f"(0.0f), "f"(sf_value_22));
                *reinterpret_cast<unsigned char*>(_gst_67) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_68 & 127));
            }
        }
        float _vec_load_12[8];
        {
            const uint4* _vptr_69 = reinterpret_cast<const uint4*>(w_ptr + 3072 + 0);
            uint4 _vld_69[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_69[_blk] = _vptr_69[_blk];
                uint32_t* _vpairs_69 = reinterpret_cast<uint32_t*>(&_vld_69[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) << 16);
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_69[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_26[4];
        {
            const uint4* _vptr_70 = reinterpret_cast<const uint4*>(s_ptr + 3072);
            uint4* _vdst_70 = reinterpret_cast<uint4*>(&tab_s_26[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_70[_blk] = _vptr_70[_blk];
            }
        }
        float _vec_load_13[8];
        {
            const uint4* _vptr_71 = reinterpret_cast<const uint4*>(b_ptr + 3072 + 0);
            uint4 _vld_71[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_71[_blk] = _vptr_71[_blk];
                uint32_t* _vpairs_71 = reinterpret_cast<uint32_t*>(&_vld_71[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) << 16);
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_71[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_12 = 0; pair_12 < 4; pair_12++) {
            int mul_offset_4 = pair_12 * 2;
            float2 _f32x2_mul_a_72 = make_float2(_unpack_bf16_0[32 + mul_offset_4], _unpack_bf16_0[32 + mul_offset_4 + 1]);
            float2 x_rstd_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_72), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_73 = make_float2(_vec_load_8[mul_offset_4], _vec_load_8[mul_offset_4 + 1]);
            float2 x_rstd_w_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_4) : "l"(*(const unsigned long long*)&x_rstd_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_73));
            scaled[mul_offset_4] = x_rstd_w_4.x;
            scaled[mul_offset_4 + 1] = x_rstd_w_4.y;
        }
        unsigned int scaled_bf16_27[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_27[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_27[_pair]) << 16);
            (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_27[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_5 = 0; q_5 < 4; q_5++) {
            uint32_t _bf16x2_add_4;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_4) : "r"(tab_s_10[q_5]), "r"(one_pair));
            scale_plus_one[q_5] = _bf16x2_add_4;
        }
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_13 = 0; pair_13 < 4; pair_13++) {
            int ada_offset_4 = pair_13 * 2;
            float2 adaln_pair_4 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_13[ada_offset_4], _unpack_bf16_13[ada_offset_4 + 1]), make_float2(_unpack_bf16_14[ada_offset_4], _unpack_bf16_14[ada_offset_4 + 1]), make_float2(_vec_load_9[ada_offset_4], _vec_load_9[ada_offset_4 + 1]));
            adaln[ada_offset_4] = adaln_pair_4.x;
            adaln[ada_offset_4 + 1] = adaln_pair_4.y;
        }
        unsigned int adaln_bf16_28[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_28[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_28[_pair]) << 16);
            (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_28[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            values[j_12] = _unpack_bf16_15[j_12];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_13 = 0; j_13 < 8; j_13++) {
                normalized[j_13] = _unpack_bf16_13[j_13];
            }
            unsigned int _packed_8[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_8[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2048 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_8)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_9[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_9[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2048 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_9)[0];
        }
        #pragma unroll
        for (int j_14 = 0; j_14 < 8; j_14++) {
            absolute[j_14] = values[j_14];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_29 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_29 = max_noftz(amax_29, absolute[_lr]);
        }
        float _shfl_xor_sync_74 = __shfl_xor_sync(0xFFFFFFFF, amax_29, 1);
        amax_29 = max_noftz(amax_29, _shfl_xor_sync_74);
        float sf_value_30 = global_scale * (amax_29 * rcp_six);
        uint16_t _e4m3x2_76;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_76) : "f"(0.0f), "f"(sf_value_30));
        uint8_t _cvt_satfinite_e4m3_f32_75 = (uint8_t)_e4m3x2_76;
        uint32_t _f16x2_decode_77;
        float sf_rounded_31;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_77) : "h"(_e4m3x2_76));
        uint16_t _f16_decode_77 = (uint16_t)_f16x2_decode_77;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_31) : "h"(_f16_decode_77));
        float _rcp_78 = approx_rcp(sf_rounded_31 * rcp_global_scale);
        float output_scale_32 = fminf(_rcp_78, 3.4028235e+38f);
        float2 output_pair_33 = make_float2(output_scale_32, output_scale_32);
        #pragma unroll
        for (int pair_14 = 0; pair_14 < 4; pair_14++) {
            int quant_offset_4 = pair_14 * 2;
            float2 _f32x2_mul_a_79 = make_float2(values[quant_offset_4], values[quant_offset_4 + 1]);
            float2 quant_pair_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&output_pair_33));
            quant_values[quant_offset_4] = quant_pair_4.x;
            quant_values[quant_offset_4 + 1] = quant_pair_4.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_80 = q_ptr + 1024;
            __builtin_assume(__isGlobal(_gst_80));
            *(reinterpret_cast<int*>(_gst_80) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_81 = sf_ptr + 16384;
                __builtin_assume(__isGlobal(_gst_81));
                uint16_t _cvt_satfinite_e4m3x2_f32_82;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_82) : "f"(0.0f), "f"(sf_value_30));
                *reinterpret_cast<unsigned char*>(_gst_81) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_82 & 127));
            }
        }
        float _vec_load_14[8];
        {
            const uint4* _vptr_83 = reinterpret_cast<const uint4*>(w_ptr + 3584 + 0);
            uint4 _vld_83[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_83[_blk] = _vptr_83[_blk];
                uint32_t* _vpairs_83 = reinterpret_cast<uint32_t*>(&_vld_83[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) << 16);
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_83[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_34[4];
        {
            const uint4* _vptr_84 = reinterpret_cast<const uint4*>(s_ptr + 3584);
            uint4* _vdst_84 = reinterpret_cast<uint4*>(&tab_s_34[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_84[_blk] = _vptr_84[_blk];
            }
        }
        float _vec_load_15[8];
        {
            const uint4* _vptr_85 = reinterpret_cast<const uint4*>(b_ptr + 3584 + 0);
            uint4 _vld_85[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_85[_blk] = _vptr_85[_blk];
                uint32_t* _vpairs_85 = reinterpret_cast<uint32_t*>(&_vld_85[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) << 16);
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_85[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_15 = 0; pair_15 < 4; pair_15++) {
            int mul_offset_5 = pair_15 * 2;
            float2 _f32x2_mul_a_86 = make_float2(_unpack_bf16_0[40 + mul_offset_5], _unpack_bf16_0[40 + mul_offset_5 + 1]);
            float2 x_rstd_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_86), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_87 = make_float2(_vec_load_10[mul_offset_5], _vec_load_10[mul_offset_5 + 1]);
            float2 x_rstd_w_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_5) : "l"(*(const unsigned long long*)&x_rstd_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_87));
            scaled[mul_offset_5] = x_rstd_w_5.x;
            scaled[mul_offset_5 + 1] = x_rstd_w_5.y;
        }
        unsigned int scaled_bf16_35[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_35[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_35[_pair]) << 16);
            (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_35[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_6 = 0; q_6 < 4; q_6++) {
            uint32_t _bf16x2_add_5;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_5) : "r"(tab_s_18[q_6]), "r"(one_pair));
            scale_plus_one[q_6] = _bf16x2_add_5;
        }
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_16 = 0; pair_16 < 4; pair_16++) {
            int ada_offset_5 = pair_16 * 2;
            float2 adaln_pair_5 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_16[ada_offset_5], _unpack_bf16_16[ada_offset_5 + 1]), make_float2(_unpack_bf16_17[ada_offset_5], _unpack_bf16_17[ada_offset_5 + 1]), make_float2(_vec_load_11[ada_offset_5], _vec_load_11[ada_offset_5 + 1]));
            adaln[ada_offset_5] = adaln_pair_5.x;
            adaln[ada_offset_5 + 1] = adaln_pair_5.y;
        }
        unsigned int adaln_bf16_36[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_36[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_36[_pair]) << 16);
            (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_36[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            values[j_15] = _unpack_bf16_18[j_15];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_16 = 0; j_16 < 8; j_16++) {
                normalized[j_16] = _unpack_bf16_16[j_16];
            }
            unsigned int _packed_10[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_10[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2560 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_10)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_11[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_11[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2560 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_11)[0];
        }
        #pragma unroll
        for (int j_17 = 0; j_17 < 8; j_17++) {
            absolute[j_17] = values[j_17];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_37 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_37 = max_noftz(amax_37, absolute[_lr]);
        }
        float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, amax_37, 1);
        amax_37 = max_noftz(amax_37, _shfl_xor_sync_88);
        float sf_value_38 = global_scale * (amax_37 * rcp_six);
        uint16_t _e4m3x2_90;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_90) : "f"(0.0f), "f"(sf_value_38));
        uint8_t _cvt_satfinite_e4m3_f32_89 = (uint8_t)_e4m3x2_90;
        uint32_t _f16x2_decode_91;
        float sf_rounded_39;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_91) : "h"(_e4m3x2_90));
        uint16_t _f16_decode_91 = (uint16_t)_f16x2_decode_91;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_39) : "h"(_f16_decode_91));
        float _rcp_92 = approx_rcp(sf_rounded_39 * rcp_global_scale);
        float output_scale_40 = fminf(_rcp_92, 3.4028235e+38f);
        float2 output_pair_41 = make_float2(output_scale_40, output_scale_40);
        #pragma unroll
        for (int pair_17 = 0; pair_17 < 4; pair_17++) {
            int quant_offset_5 = pair_17 * 2;
            float2 _f32x2_mul_a_93 = make_float2(values[quant_offset_5], values[quant_offset_5 + 1]);
            float2 quant_pair_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_93), "l"(*(const unsigned long long*)&output_pair_41));
            quant_values[quant_offset_5] = quant_pair_5.x;
            quant_values[quant_offset_5 + 1] = quant_pair_5.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_94 = q_ptr + 1280;
            __builtin_assume(__isGlobal(_gst_94));
            *(reinterpret_cast<int*>(_gst_94) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_95 = sf_ptr + 20480;
                __builtin_assume(__isGlobal(_gst_95));
                uint16_t _cvt_satfinite_e4m3x2_f32_96;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_96) : "f"(0.0f), "f"(sf_value_38));
                *reinterpret_cast<unsigned char*>(_gst_95) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_96 & 127));
            }
        }
        float _vec_load_16[8];
        {
            const uint4* _vptr_97 = reinterpret_cast<const uint4*>(w_ptr + 4096 + 0);
            uint4 _vld_97[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_97[_blk] = _vptr_97[_blk];
                uint32_t* _vpairs_97 = reinterpret_cast<uint32_t*>(&_vld_97[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) << 16);
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_97[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_42[4];
        {
            const uint4* _vptr_98 = reinterpret_cast<const uint4*>(s_ptr + 4096);
            uint4* _vdst_98 = reinterpret_cast<uint4*>(&tab_s_42[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_98[_blk] = _vptr_98[_blk];
            }
        }
        float _vec_load_17[8];
        {
            const uint4* _vptr_99 = reinterpret_cast<const uint4*>(b_ptr + 4096 + 0);
            uint4 _vld_99[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_99[_blk] = _vptr_99[_blk];
                uint32_t* _vpairs_99 = reinterpret_cast<uint32_t*>(&_vld_99[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) << 16);
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_99[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_18 = 0; pair_18 < 4; pair_18++) {
            int mul_offset_6 = pair_18 * 2;
            float2 _f32x2_mul_a_100 = make_float2(_unpack_bf16_0[48 + mul_offset_6], _unpack_bf16_0[48 + mul_offset_6 + 1]);
            float2 x_rstd_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_101 = make_float2(_vec_load_12[mul_offset_6], _vec_load_12[mul_offset_6 + 1]);
            float2 x_rstd_w_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_6) : "l"(*(const unsigned long long*)&x_rstd_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
            scaled[mul_offset_6] = x_rstd_w_6.x;
            scaled[mul_offset_6 + 1] = x_rstd_w_6.y;
        }
        unsigned int scaled_bf16_43[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_43[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_43[_pair]) << 16);
            (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_43[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_7 = 0; q_7 < 4; q_7++) {
            uint32_t _bf16x2_add_6;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_6) : "r"(tab_s_26[q_7]), "r"(one_pair));
            scale_plus_one[q_7] = _bf16x2_add_6;
        }
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_19 = 0; pair_19 < 4; pair_19++) {
            int ada_offset_6 = pair_19 * 2;
            float2 adaln_pair_6 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_19[ada_offset_6], _unpack_bf16_19[ada_offset_6 + 1]), make_float2(_unpack_bf16_20[ada_offset_6], _unpack_bf16_20[ada_offset_6 + 1]), make_float2(_vec_load_13[ada_offset_6], _vec_load_13[ada_offset_6 + 1]));
            adaln[ada_offset_6] = adaln_pair_6.x;
            adaln[ada_offset_6 + 1] = adaln_pair_6.y;
        }
        unsigned int adaln_bf16_44[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_44[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_44[_pair]) << 16);
            (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_44[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_18 = 0; j_18 < 8; j_18++) {
            values[j_18] = _unpack_bf16_21[j_18];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_19 = 0; j_19 < 8; j_19++) {
                normalized[j_19] = _unpack_bf16_19[j_19];
            }
            unsigned int _packed_12[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_12[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3072 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_12)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_13[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_13[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3072 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_13)[0];
        }
        #pragma unroll
        for (int j_20 = 0; j_20 < 8; j_20++) {
            absolute[j_20] = values[j_20];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_45 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_45 = max_noftz(amax_45, absolute[_lr]);
        }
        float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, amax_45, 1);
        amax_45 = max_noftz(amax_45, _shfl_xor_sync_102);
        float sf_value_46 = global_scale * (amax_45 * rcp_six);
        uint16_t _e4m3x2_104;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_104) : "f"(0.0f), "f"(sf_value_46));
        uint8_t _cvt_satfinite_e4m3_f32_103 = (uint8_t)_e4m3x2_104;
        uint32_t _f16x2_decode_105;
        float sf_rounded_47;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_105) : "h"(_e4m3x2_104));
        uint16_t _f16_decode_105 = (uint16_t)_f16x2_decode_105;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_47) : "h"(_f16_decode_105));
        float _rcp_106 = approx_rcp(sf_rounded_47 * rcp_global_scale);
        float output_scale_48 = fminf(_rcp_106, 3.4028235e+38f);
        float2 output_pair_49 = make_float2(output_scale_48, output_scale_48);
        #pragma unroll
        for (int pair_20 = 0; pair_20 < 4; pair_20++) {
            int quant_offset_6 = pair_20 * 2;
            float2 _f32x2_mul_a_107 = make_float2(values[quant_offset_6], values[quant_offset_6 + 1]);
            float2 quant_pair_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_107), "l"(*(const unsigned long long*)&output_pair_49));
            quant_values[quant_offset_6] = quant_pair_6.x;
            quant_values[quant_offset_6 + 1] = quant_pair_6.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_108 = q_ptr + 1536;
            __builtin_assume(__isGlobal(_gst_108));
            *(reinterpret_cast<int*>(_gst_108) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_109 = sf_ptr + 24576;
                __builtin_assume(__isGlobal(_gst_109));
                uint16_t _cvt_satfinite_e4m3x2_f32_110;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_110) : "f"(0.0f), "f"(sf_value_46));
                *reinterpret_cast<unsigned char*>(_gst_109) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_110 & 127));
            }
        }
        float _vec_load_18[8];
        {
            const uint4* _vptr_111 = reinterpret_cast<const uint4*>(w_ptr + 4608 + 0);
            uint4 _vld_111[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_111[_blk] = _vptr_111[_blk];
                uint32_t* _vpairs_111 = reinterpret_cast<uint32_t*>(&_vld_111[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) << 16);
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_111[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_50[4];
        {
            const uint4* _vptr_112 = reinterpret_cast<const uint4*>(s_ptr + 4608);
            uint4* _vdst_112 = reinterpret_cast<uint4*>(&tab_s_50[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_112[_blk] = _vptr_112[_blk];
            }
        }
        float _vec_load_19[8];
        {
            const uint4* _vptr_113 = reinterpret_cast<const uint4*>(b_ptr + 4608 + 0);
            uint4 _vld_113[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_113[_blk] = _vptr_113[_blk];
                uint32_t* _vpairs_113 = reinterpret_cast<uint32_t*>(&_vld_113[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) << 16);
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_113[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_21 = 0; pair_21 < 4; pair_21++) {
            int mul_offset_7 = pair_21 * 2;
            float2 _f32x2_mul_a_114 = make_float2(_unpack_bf16_0[56 + mul_offset_7], _unpack_bf16_0[56 + mul_offset_7 + 1]);
            float2 x_rstd_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_114), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_115 = make_float2(_vec_load_14[mul_offset_7], _vec_load_14[mul_offset_7 + 1]);
            float2 x_rstd_w_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_7) : "l"(*(const unsigned long long*)&x_rstd_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_115));
            scaled[mul_offset_7] = x_rstd_w_7.x;
            scaled[mul_offset_7 + 1] = x_rstd_w_7.y;
        }
        unsigned int scaled_bf16_51[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_51[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_51[_pair]) << 16);
            (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_51[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_8 = 0; q_8 < 4; q_8++) {
            uint32_t _bf16x2_add_7;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_7) : "r"(tab_s_34[q_8]), "r"(one_pair));
            scale_plus_one[q_8] = _bf16x2_add_7;
        }
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_22 = 0; pair_22 < 4; pair_22++) {
            int ada_offset_7 = pair_22 * 2;
            float2 adaln_pair_7 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_22[ada_offset_7], _unpack_bf16_22[ada_offset_7 + 1]), make_float2(_unpack_bf16_23[ada_offset_7], _unpack_bf16_23[ada_offset_7 + 1]), make_float2(_vec_load_15[ada_offset_7], _vec_load_15[ada_offset_7 + 1]));
            adaln[ada_offset_7] = adaln_pair_7.x;
            adaln[ada_offset_7 + 1] = adaln_pair_7.y;
        }
        unsigned int adaln_bf16_52[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_52[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_52[_pair]) << 16);
            (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_52[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_21 = 0; j_21 < 8; j_21++) {
            values[j_21] = _unpack_bf16_24[j_21];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_22 = 0; j_22 < 8; j_22++) {
                normalized[j_22] = _unpack_bf16_22[j_22];
            }
            unsigned int _packed_14[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_14[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3584 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_14)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_15[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_15[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_15[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_15[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3584 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_15)[0];
        }
        #pragma unroll
        for (int j_23 = 0; j_23 < 8; j_23++) {
            absolute[j_23] = values[j_23];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_53 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_53 = max_noftz(amax_53, absolute[_lr]);
        }
        float _shfl_xor_sync_116 = __shfl_xor_sync(0xFFFFFFFF, amax_53, 1);
        amax_53 = max_noftz(amax_53, _shfl_xor_sync_116);
        float sf_value_54 = global_scale * (amax_53 * rcp_six);
        uint16_t _e4m3x2_118;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_118) : "f"(0.0f), "f"(sf_value_54));
        uint8_t _cvt_satfinite_e4m3_f32_117 = (uint8_t)_e4m3x2_118;
        uint32_t _f16x2_decode_119;
        float sf_rounded_55;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_119) : "h"(_e4m3x2_118));
        uint16_t _f16_decode_119 = (uint16_t)_f16x2_decode_119;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_55) : "h"(_f16_decode_119));
        float _rcp_120 = approx_rcp(sf_rounded_55 * rcp_global_scale);
        float output_scale_56 = fminf(_rcp_120, 3.4028235e+38f);
        float2 output_pair_57 = make_float2(output_scale_56, output_scale_56);
        #pragma unroll
        for (int pair_23 = 0; pair_23 < 4; pair_23++) {
            int quant_offset_7 = pair_23 * 2;
            float2 _f32x2_mul_a_121 = make_float2(values[quant_offset_7], values[quant_offset_7 + 1]);
            float2 quant_pair_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&output_pair_57));
            quant_values[quant_offset_7] = quant_pair_7.x;
            quant_values[quant_offset_7 + 1] = quant_pair_7.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_122 = q_ptr + 1792;
            __builtin_assume(__isGlobal(_gst_122));
            *(reinterpret_cast<int*>(_gst_122) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_123 = sf_ptr + 28672;
                __builtin_assume(__isGlobal(_gst_123));
                uint16_t _cvt_satfinite_e4m3x2_f32_124;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_124) : "f"(0.0f), "f"(sf_value_54));
                *reinterpret_cast<unsigned char*>(_gst_123) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_124 & 127));
            }
        }
        int last_col = 5120 + t % 32 * 8;
        float _vec_load_20[8];
        {
            const uint4* _vptr_125 = reinterpret_cast<const uint4*>(x_norm_weight + last_col + 0);
            uint4 _vld_125[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_125[_blk] = _vptr_125[_blk];
                uint32_t* _vpairs_125 = reinterpret_cast<uint32_t*>(&_vld_125[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_125[_pair]) << 16);
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_125[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int last_s[4];
        {
            const uint4* _vptr_126 = reinterpret_cast<const uint4*>(adaln_scale + table_base + (unsigned long long)last_col);
            uint4* _vdst_126 = reinterpret_cast<uint4*>(&last_s[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_126[_blk] = _vptr_126[_blk];
            }
        }
        float _vec_load_21[8];
        {
            const uint4* _vptr_127 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)last_col) + 0);
            uint4 _vld_127[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_127[_blk] = _vptr_127[_blk];
                uint32_t* _vpairs_127 = reinterpret_cast<uint32_t*>(&_vld_127[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_127[_pair]) << 16);
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_127[_pair]) & 0xffff0000u);
                }
            }
        }
        #pragma unroll
        for (int pair_24 = 0; pair_24 < 4; pair_24++) {
            int mul_offset_8 = pair_24 * 2;
            float2 _f32x2_mul_a_128 = make_float2(_unpack_bf16_0[64 + mul_offset_8], _unpack_bf16_0[64 + mul_offset_8 + 1]);
            float2 x_rstd_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_129 = make_float2(_vec_load_16[mul_offset_8], _vec_load_16[mul_offset_8 + 1]);
            float2 x_rstd_w_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_8) : "l"(*(const unsigned long long*)&x_rstd_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
            scaled[mul_offset_8] = x_rstd_w_8.x;
            scaled[mul_offset_8 + 1] = x_rstd_w_8.y;
        }
        unsigned int scaled_bf16_58[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_58[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_58[_pair]) << 16);
            (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_58[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_9 = 0; q_9 < 4; q_9++) {
            uint32_t _bf16x2_add_8;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_8) : "r"(tab_s_42[q_9]), "r"(one_pair));
            scale_plus_one[q_9] = _bf16x2_add_8;
        }
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_25 = 0; pair_25 < 4; pair_25++) {
            int ada_offset_8 = pair_25 * 2;
            float2 adaln_pair_8 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_25[ada_offset_8], _unpack_bf16_25[ada_offset_8 + 1]), make_float2(_unpack_bf16_26[ada_offset_8], _unpack_bf16_26[ada_offset_8 + 1]), make_float2(_vec_load_17[ada_offset_8], _vec_load_17[ada_offset_8 + 1]));
            adaln[ada_offset_8] = adaln_pair_8.x;
            adaln[ada_offset_8 + 1] = adaln_pair_8.y;
        }
        unsigned int adaln_bf16_59[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_59[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_59[_pair]) << 16);
            (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_59[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_24 = 0; j_24 < 8; j_24++) {
            values[j_24] = _unpack_bf16_27[j_24];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_25 = 0; j_25 < 8; j_25++) {
                normalized[j_25] = _unpack_bf16_25[j_25];
            }
            unsigned int _packed_16[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_16[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_16[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_16[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4096 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_16)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_17[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_17[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_17[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_17[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4096 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_17)[0];
        }
        #pragma unroll
        for (int j_26 = 0; j_26 < 8; j_26++) {
            absolute[j_26] = values[j_26];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_60 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_60 = max_noftz(amax_60, absolute[_lr]);
        }
        float _shfl_xor_sync_130 = __shfl_xor_sync(0xFFFFFFFF, amax_60, 1);
        amax_60 = max_noftz(amax_60, _shfl_xor_sync_130);
        float sf_value_61 = global_scale * (amax_60 * rcp_six);
        uint16_t _e4m3x2_132;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_132) : "f"(0.0f), "f"(sf_value_61));
        uint8_t _cvt_satfinite_e4m3_f32_131 = (uint8_t)_e4m3x2_132;
        uint32_t _f16x2_decode_133;
        float sf_rounded_62;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_133) : "h"(_e4m3x2_132));
        uint16_t _f16_decode_133 = (uint16_t)_f16x2_decode_133;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_62) : "h"(_f16_decode_133));
        float _rcp_134 = approx_rcp(sf_rounded_62 * rcp_global_scale);
        float output_scale_63 = fminf(_rcp_134, 3.4028235e+38f);
        float2 output_pair_64 = make_float2(output_scale_63, output_scale_63);
        #pragma unroll
        for (int pair_26 = 0; pair_26 < 4; pair_26++) {
            int quant_offset_8 = pair_26 * 2;
            float2 _f32x2_mul_a_135 = make_float2(values[quant_offset_8], values[quant_offset_8 + 1]);
            float2 quant_pair_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_135), "l"(*(const unsigned long long*)&output_pair_64));
            quant_values[quant_offset_8] = quant_pair_8.x;
            quant_values[quant_offset_8 + 1] = quant_pair_8.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_136 = q_ptr + 2048;
            __builtin_assume(__isGlobal(_gst_136));
            *(reinterpret_cast<int*>(_gst_136) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_137 = sf_ptr + 32768;
                __builtin_assume(__isGlobal(_gst_137));
                uint16_t _cvt_satfinite_e4m3x2_f32_138;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_138) : "f"(0.0f), "f"(sf_value_61));
                *reinterpret_cast<unsigned char*>(_gst_137) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_138 & 127));
            }
        }
        #pragma unroll
        for (int pair_27 = 0; pair_27 < 4; pair_27++) {
            int mul_offset_9 = pair_27 * 2;
            float2 _f32x2_mul_a_139 = make_float2(_unpack_bf16_0[72 + mul_offset_9], _unpack_bf16_0[72 + mul_offset_9 + 1]);
            float2 x_rstd_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_139), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_140 = make_float2(_vec_load_18[mul_offset_9], _vec_load_18[mul_offset_9 + 1]);
            float2 x_rstd_w_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_9) : "l"(*(const unsigned long long*)&x_rstd_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_140));
            scaled[mul_offset_9] = x_rstd_w_9.x;
            scaled[mul_offset_9 + 1] = x_rstd_w_9.y;
        }
        unsigned int scaled_bf16_65[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
            scaled_bf16_65[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_65[_pair]) << 16);
            (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_65[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int q_10 = 0; q_10 < 4; q_10++) {
            uint32_t _bf16x2_add_9;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_9) : "r"(tab_s_50[q_10]), "r"(one_pair));
            scale_plus_one[q_10] = _bf16x2_add_9;
        }
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_28 = 0; pair_28 < 4; pair_28++) {
            int ada_offset_9 = pair_28 * 2;
            float2 adaln_pair_9 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_28[ada_offset_9], _unpack_bf16_28[ada_offset_9 + 1]), make_float2(_unpack_bf16_29[ada_offset_9], _unpack_bf16_29[ada_offset_9 + 1]), make_float2(_vec_load_19[ada_offset_9], _vec_load_19[ada_offset_9 + 1]));
            adaln[ada_offset_9] = adaln_pair_9.x;
            adaln[ada_offset_9 + 1] = adaln_pair_9.y;
        }
        unsigned int adaln_bf16_66[4];
        #pragma unroll
        for (int _lp = 0; _lp < 4; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
            adaln_bf16_66[_lp] = *(uint32_t*)&_bf2;
        }
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_66[_pair]) << 16);
            (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_66[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int j_27 = 0; j_27 < 8; j_27++) {
            values[j_27] = _unpack_bf16_30[j_27];
        }
        if (write_debug == 2) {
            #pragma unroll
            for (int j_28 = 0; j_28 < 8; j_28++) {
                normalized[j_28] = _unpack_bf16_28[j_28];
            }
            unsigned int _packed_18[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                _packed_18[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                _packed_18[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                _packed_18[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4608 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_18)[0];
        } else if (write_debug != 0) {
            unsigned int _packed_19[4];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                _packed_19[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                _packed_19[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                _packed_19[_lp + 3] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4608 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_19)[0];
        }
        #pragma unroll
        for (int j_29 = 0; j_29 < 8; j_29++) {
            absolute[j_29] = values[j_29];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_67 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_67 = max_noftz(amax_67, absolute[_lr]);
        }
        float _shfl_xor_sync_141 = __shfl_xor_sync(0xFFFFFFFF, amax_67, 1);
        amax_67 = max_noftz(amax_67, _shfl_xor_sync_141);
        float sf_value_68 = global_scale * (amax_67 * rcp_six);
        uint16_t _e4m3x2_143;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_143) : "f"(0.0f), "f"(sf_value_68));
        uint8_t _cvt_satfinite_e4m3_f32_142 = (uint8_t)_e4m3x2_143;
        uint32_t _f16x2_decode_144;
        float sf_rounded_69;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_144) : "h"(_e4m3x2_143));
        uint16_t _f16_decode_144 = (uint16_t)_f16x2_decode_144;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_69) : "h"(_f16_decode_144));
        float _rcp_145 = approx_rcp(sf_rounded_69 * rcp_global_scale);
        float output_scale_70 = fminf(_rcp_145, 3.4028235e+38f);
        float2 output_pair_71 = make_float2(output_scale_70, output_scale_70);
        #pragma unroll
        for (int pair_29 = 0; pair_29 < 4; pair_29++) {
            int quant_offset_9 = pair_29 * 2;
            float2 _f32x2_mul_a_146 = make_float2(values[quant_offset_9], values[quant_offset_9 + 1]);
            float2 quant_pair_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_146), "l"(*(const unsigned long long*)&output_pair_71));
            quant_values[quant_offset_9] = quant_pair_9.x;
            quant_values[quant_offset_9 + 1] = quant_pair_9.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_147 = q_ptr + 2304;
            __builtin_assume(__isGlobal(_gst_147));
            *(reinterpret_cast<int*>(_gst_147) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_148 = sf_ptr + 36864;
                __builtin_assume(__isGlobal(_gst_148));
                uint16_t _cvt_satfinite_e4m3x2_f32_149;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_149) : "f"(0.0f), "f"(sf_value_68));
                *reinterpret_cast<unsigned char*>(_gst_148) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_149 & 127));
            }
        }
        if (t < 32) {
            #pragma unroll
            for (int pair_30 = 0; pair_30 < 4; pair_30++) {
                int mul_offset_10 = pair_30 * 2;
                float2 _f32x2_mul_a_150 = make_float2(_unpack_bf16_0[80 + mul_offset_10], _unpack_bf16_0[80 + mul_offset_10 + 1]);
                float2 x_rstd_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_150), "l"(*(const unsigned long long*)&rstd_pair));
                float2 _f32x2_mul_b_151 = make_float2(_vec_load_20[mul_offset_10], _vec_load_20[mul_offset_10 + 1]);
                float2 x_rstd_w_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_10) : "l"(*(const unsigned long long*)&x_rstd_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_151));
                scaled[mul_offset_10] = x_rstd_w_10.x;
                scaled[mul_offset_10 + 1] = x_rstd_w_10.y;
            }
            unsigned int scaled_bf16_0[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(scaled[_lp*2 + 0], scaled[_lp*2+1 + 0]));
                scaled_bf16_0[_lp] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_31[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_0[_pair]) << 16);
                (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scaled_bf16_0[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int q_11 = 0; q_11 < 4; q_11++) {
                uint32_t _bf16x2_add_10;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_10) : "r"(last_s[q_11]), "r"(one_pair));
                scale_plus_one[q_11] = _bf16x2_add_10;
            }
            float _unpack_bf16_32[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_32[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
                (&_unpack_bf16_32[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int pair_31 = 0; pair_31 < 4; pair_31++) {
                int ada_offset_10 = pair_31 * 2;
                float2 adaln_pair_10 = fma_f32x2_rn_ftz(make_float2(_unpack_bf16_31[ada_offset_10], _unpack_bf16_31[ada_offset_10 + 1]), make_float2(_unpack_bf16_32[ada_offset_10], _unpack_bf16_32[ada_offset_10 + 1]), make_float2(_vec_load_21[ada_offset_10], _vec_load_21[ada_offset_10 + 1]));
                adaln[ada_offset_10] = adaln_pair_10.x;
                adaln[ada_offset_10 + 1] = adaln_pair_10.y;
            }
            unsigned int adaln_bf16_1[4];
            #pragma unroll
            for (int _lp = 0; _lp < 4; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(adaln[_lp*2 + 0], adaln[_lp*2+1 + 0]));
                adaln_bf16_1[_lp] = *(uint32_t*)&_bf2;
            }
            float _unpack_bf16_33[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_33[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_1[_pair]) << 16);
                (&_unpack_bf16_33[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(adaln_bf16_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int j_30 = 0; j_30 < 8; j_30++) {
                values[j_30] = _unpack_bf16_33[j_30];
            }
            if (write_debug == 2) {
                #pragma unroll
                for (int j_31 = 0; j_31 < 8; j_31++) {
                    normalized[j_31] = _unpack_bf16_31[j_31];
                }
                unsigned int _packed_20[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_20[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_20[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_20[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_20[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(5120 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_20)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_21[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_21[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_21[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_21[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_21[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(5120 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_21)[0];
            }
            #pragma unroll
            for (int j_32 = 0; j_32 < 8; j_32++) {
                absolute[j_32] = values[j_32];
            }
            absolute[0] = fabsf(absolute[0]);
            absolute[1] = fabsf(absolute[1]);
            absolute[2] = fabsf(absolute[2]);
            absolute[3] = fabsf(absolute[3]);
            absolute[4] = fabsf(absolute[4]);
            absolute[5] = fabsf(absolute[5]);
            absolute[6] = fabsf(absolute[6]);
            absolute[7] = fabsf(absolute[7]);
            float amax_2 = absolute[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                amax_2 = max_noftz(amax_2, absolute[_lr]);
            }
            float _shfl_xor_sync_152 = __shfl_xor_sync(0xFFFFFFFF, amax_2, 1);
            amax_2 = max_noftz(amax_2, _shfl_xor_sync_152);
            float sf_value_3 = global_scale * (amax_2 * rcp_six);
            uint16_t _e4m3x2_154;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_154) : "f"(0.0f), "f"(sf_value_3));
            uint8_t _cvt_satfinite_e4m3_f32_153 = (uint8_t)_e4m3x2_154;
            uint32_t _f16x2_decode_155;
            float sf_rounded_4;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_155) : "h"(_e4m3x2_154));
            uint16_t _f16_decode_155 = (uint16_t)_f16x2_decode_155;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_4) : "h"(_f16_decode_155));
            float _rcp_156 = approx_rcp(sf_rounded_4 * rcp_global_scale);
            float output_scale_5 = fminf(_rcp_156, 3.4028235e+38f);
            float2 output_pair_6 = make_float2(output_scale_5, output_scale_5);
            #pragma unroll
            for (int pair_32 = 0; pair_32 < 4; pair_32++) {
                int quant_offset_10 = pair_32 * 2;
                float2 _f32x2_mul_a_157 = make_float2(values[quant_offset_10], values[quant_offset_10 + 1]);
                float2 quant_pair_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_157), "l"(*(const unsigned long long*)&output_pair_6));
                quant_values[quant_offset_10] = quant_pair_10.x;
                quant_values[quant_offset_10 + 1] = quant_pair_10.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_158 = q_ptr + 2560;
                __builtin_assume(__isGlobal(_gst_158));
                *(reinterpret_cast<int*>(_gst_158) + (0)) = packed[0];
            }
            if (t % 2 == 0) {
                {
                    auto _gst_159 = sf_ptr + 40960;
                    __builtin_assume(__isGlobal(_gst_159));
                    uint16_t _cvt_satfinite_e4m3x2_f32_160;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_160) : "f"(0.0f), "f"(sf_value_3));
                    *reinterpret_cast<unsigned char*>(_gst_159) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_160 & 127));
                }
            }
        }
    } else {
        #pragma unroll
        for (int j_33 = 0; j_33 < 8; j_33++) {
            values[j_33] = 0.0f;
            normalized[j_33] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_22[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_22[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_22[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_22[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_22[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(t * 8)))[0] = reinterpret_cast<int4*>(_packed_22)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_23[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_23[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_23[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_23[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_23[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(t * 8)))[0] = reinterpret_cast<int4*>(_packed_23)[0];
            }
        }
        #pragma unroll
        for (int j_34 = 0; j_34 < 8; j_34++) {
            absolute[j_34] = values[j_34];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_1 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_1 = max_noftz(amax_1, absolute[_lr]);
        }
        float _shfl_xor_sync_161 = __shfl_xor_sync(0xFFFFFFFF, amax_1, 1);
        amax_1 = max_noftz(amax_1, _shfl_xor_sync_161);
        float sf_value_1 = global_scale * (amax_1 * rcp_six);
        uint16_t _e4m3x2_163;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_163) : "f"(0.0f), "f"(sf_value_1));
        uint8_t _cvt_satfinite_e4m3_f32_162 = (uint8_t)_e4m3x2_163;
        uint32_t _f16x2_decode_164;
        float sf_rounded_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_164) : "h"(_e4m3x2_163));
        uint16_t _f16_decode_164 = (uint16_t)_f16x2_decode_164;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_1) : "h"(_f16_decode_164));
        float _rcp_165 = approx_rcp(sf_rounded_1 * rcp_global_scale);
        float output_scale_1 = fminf(_rcp_165, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_1 = make_float2(output_scale_1, output_scale_1);
            #pragma unroll
            for (int pair_33 = 0; pair_33 < 4; pair_33++) {
                int quant_offset_11 = pair_33 * 2;
                float2 _f32x2_mul_a_166 = make_float2(values[quant_offset_11], values[quant_offset_11 + 1]);
                float2 quant_pair_11;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_166), "l"(*(const unsigned long long*)&output_pair_1));
                quant_values[quant_offset_11] = quant_pair_11.x;
                quant_values[quant_offset_11 + 1] = quant_pair_11.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_167 = q_ptr + 0;
                __builtin_assume(__isGlobal(_gst_167));
                *(reinterpret_cast<int*>(_gst_167) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_168 = sf_ptr + 0;
                __builtin_assume(__isGlobal(_gst_168));
                uint16_t _cvt_satfinite_e4m3x2_f32_169;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_169) : "f"(0.0f), "f"(sf_value_1));
                *reinterpret_cast<unsigned char*>(_gst_168) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_169 & 127));
            }
        }
        #pragma unroll
        for (int j_35 = 0; j_35 < 8; j_35++) {
            values[j_35] = 0.0f;
            normalized[j_35] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_24[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_24[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_24[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_24[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_24[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(512 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_24)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_25[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_25[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_25[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_25[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_25[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(512 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_25)[0];
            }
        }
        #pragma unroll
        for (int j_36 = 0; j_36 < 8; j_36++) {
            absolute[j_36] = values[j_36];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_0 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_0 = max_noftz(amax_0, absolute[_lr]);
        }
        float _shfl_xor_sync_170 = __shfl_xor_sync(0xFFFFFFFF, amax_0, 1);
        amax_0 = max_noftz(amax_0, _shfl_xor_sync_170);
        float sf_value_1_1 = global_scale * (amax_0 * rcp_six);
        uint16_t _e4m3x2_172;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_172) : "f"(0.0f), "f"(sf_value_1_1));
        uint8_t _cvt_satfinite_e4m3_f32_171 = (uint8_t)_e4m3x2_172;
        uint32_t _f16x2_decode_173;
        float sf_rounded_2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_173) : "h"(_e4m3x2_172));
        uint16_t _f16_decode_173 = (uint16_t)_f16x2_decode_173;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_2) : "h"(_f16_decode_173));
        float _rcp_174 = approx_rcp(sf_rounded_2 * rcp_global_scale);
        float output_scale_3 = fminf(_rcp_174, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_2 = make_float2(output_scale_3, output_scale_3);
            #pragma unroll
            for (int pair_34 = 0; pair_34 < 4; pair_34++) {
                int quant_offset_12 = pair_34 * 2;
                float2 _f32x2_mul_a_175 = make_float2(values[quant_offset_12], values[quant_offset_12 + 1]);
                float2 quant_pair_12;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&output_pair_2));
                quant_values[quant_offset_12] = quant_pair_12.x;
                quant_values[quant_offset_12 + 1] = quant_pair_12.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_176 = q_ptr + 256;
                __builtin_assume(__isGlobal(_gst_176));
                *(reinterpret_cast<int*>(_gst_176) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_177 = sf_ptr + 4096;
                __builtin_assume(__isGlobal(_gst_177));
                uint16_t _cvt_satfinite_e4m3x2_f32_178;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_178) : "f"(0.0f), "f"(sf_value_1_1));
                *reinterpret_cast<unsigned char*>(_gst_177) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_178 & 127));
            }
        }
        #pragma unroll
        for (int j_37 = 0; j_37 < 8; j_37++) {
            values[j_37] = 0.0f;
            normalized[j_37] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_26[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_26[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_26[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_26[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_26[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1024 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_26)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_27[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_27[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_27[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_27[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_27[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1024 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_27)[0];
            }
        }
        #pragma unroll
        for (int j_38 = 0; j_38 < 8; j_38++) {
            absolute[j_38] = values[j_38];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_4 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_4 = max_noftz(amax_4, absolute[_lr]);
        }
        float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, amax_4, 1);
        amax_4 = max_noftz(amax_4, _shfl_xor_sync_179);
        float sf_value_5 = global_scale * (amax_4 * rcp_six);
        uint16_t _e4m3x2_181;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_181) : "f"(0.0f), "f"(sf_value_5));
        uint8_t _cvt_satfinite_e4m3_f32_180 = (uint8_t)_e4m3x2_181;
        uint32_t _f16x2_decode_182;
        float sf_rounded_6;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_182) : "h"(_e4m3x2_181));
        uint16_t _f16_decode_182 = (uint16_t)_f16x2_decode_182;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_6) : "h"(_f16_decode_182));
        float _rcp_183 = approx_rcp(sf_rounded_6 * rcp_global_scale);
        float output_scale_7 = fminf(_rcp_183, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_3 = make_float2(output_scale_7, output_scale_7);
            #pragma unroll
            for (int pair_35 = 0; pair_35 < 4; pair_35++) {
                int quant_offset_13 = pair_35 * 2;
                float2 _f32x2_mul_a_184 = make_float2(values[quant_offset_13], values[quant_offset_13 + 1]);
                float2 quant_pair_13;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_184), "l"(*(const unsigned long long*)&output_pair_3));
                quant_values[quant_offset_13] = quant_pair_13.x;
                quant_values[quant_offset_13 + 1] = quant_pair_13.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_185 = q_ptr + 512;
                __builtin_assume(__isGlobal(_gst_185));
                *(reinterpret_cast<int*>(_gst_185) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_186 = sf_ptr + 8192;
                __builtin_assume(__isGlobal(_gst_186));
                uint16_t _cvt_satfinite_e4m3x2_f32_187;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_187) : "f"(0.0f), "f"(sf_value_5));
                *reinterpret_cast<unsigned char*>(_gst_186) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_187 & 127));
            }
        }
        #pragma unroll
        for (int j_39 = 0; j_39 < 8; j_39++) {
            values[j_39] = 0.0f;
            normalized[j_39] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_28[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_28[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_28[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_28[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_28[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1536 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_28)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_29[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_29[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_29[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_29[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_29[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(1536 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_29)[0];
            }
        }
        #pragma unroll
        for (int j_40 = 0; j_40 < 8; j_40++) {
            absolute[j_40] = values[j_40];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_8 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_8 = max_noftz(amax_8, absolute[_lr]);
        }
        float _shfl_xor_sync_188 = __shfl_xor_sync(0xFFFFFFFF, amax_8, 1);
        amax_8 = max_noftz(amax_8, _shfl_xor_sync_188);
        float sf_value_9 = global_scale * (amax_8 * rcp_six);
        uint16_t _e4m3x2_190;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_190) : "f"(0.0f), "f"(sf_value_9));
        uint8_t _cvt_satfinite_e4m3_f32_189 = (uint8_t)_e4m3x2_190;
        uint32_t _f16x2_decode_191;
        float sf_rounded_10;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_191) : "h"(_e4m3x2_190));
        uint16_t _f16_decode_191 = (uint16_t)_f16x2_decode_191;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_10) : "h"(_f16_decode_191));
        float _rcp_192 = approx_rcp(sf_rounded_10 * rcp_global_scale);
        float output_scale_11 = fminf(_rcp_192, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_4 = make_float2(output_scale_11, output_scale_11);
            #pragma unroll
            for (int pair_36 = 0; pair_36 < 4; pair_36++) {
                int quant_offset_14 = pair_36 * 2;
                float2 _f32x2_mul_a_193 = make_float2(values[quant_offset_14], values[quant_offset_14 + 1]);
                float2 quant_pair_14;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_193), "l"(*(const unsigned long long*)&output_pair_4));
                quant_values[quant_offset_14] = quant_pair_14.x;
                quant_values[quant_offset_14 + 1] = quant_pair_14.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_194 = q_ptr + 768;
                __builtin_assume(__isGlobal(_gst_194));
                *(reinterpret_cast<int*>(_gst_194) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_195 = sf_ptr + 12288;
                __builtin_assume(__isGlobal(_gst_195));
                uint16_t _cvt_satfinite_e4m3x2_f32_196;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_196) : "f"(0.0f), "f"(sf_value_9));
                *reinterpret_cast<unsigned char*>(_gst_195) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_196 & 127));
            }
        }
        #pragma unroll
        for (int j_41 = 0; j_41 < 8; j_41++) {
            values[j_41] = 0.0f;
            normalized[j_41] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_30[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_30[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_30[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_30[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_30[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2048 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_30)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_31[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_31[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_31[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_31[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_31[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2048 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_31)[0];
            }
        }
        #pragma unroll
        for (int j_42 = 0; j_42 < 8; j_42++) {
            absolute[j_42] = values[j_42];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_12 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_12 = max_noftz(amax_12, absolute[_lr]);
        }
        float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, amax_12, 1);
        amax_12 = max_noftz(amax_12, _shfl_xor_sync_197);
        float sf_value_13 = global_scale * (amax_12 * rcp_six);
        uint16_t _e4m3x2_199;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_199) : "f"(0.0f), "f"(sf_value_13));
        uint8_t _cvt_satfinite_e4m3_f32_198 = (uint8_t)_e4m3x2_199;
        uint32_t _f16x2_decode_200;
        float sf_rounded_14;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_200) : "h"(_e4m3x2_199));
        uint16_t _f16_decode_200 = (uint16_t)_f16x2_decode_200;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_14) : "h"(_f16_decode_200));
        float _rcp_201 = approx_rcp(sf_rounded_14 * rcp_global_scale);
        float output_scale_15 = fminf(_rcp_201, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_5 = make_float2(output_scale_15, output_scale_15);
            #pragma unroll
            for (int pair_37 = 0; pair_37 < 4; pair_37++) {
                int quant_offset_15 = pair_37 * 2;
                float2 _f32x2_mul_a_202 = make_float2(values[quant_offset_15], values[quant_offset_15 + 1]);
                float2 quant_pair_15;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_202), "l"(*(const unsigned long long*)&output_pair_5));
                quant_values[quant_offset_15] = quant_pair_15.x;
                quant_values[quant_offset_15 + 1] = quant_pair_15.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_203 = q_ptr + 1024;
                __builtin_assume(__isGlobal(_gst_203));
                *(reinterpret_cast<int*>(_gst_203) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_204 = sf_ptr + 16384;
                __builtin_assume(__isGlobal(_gst_204));
                uint16_t _cvt_satfinite_e4m3x2_f32_205;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_205) : "f"(0.0f), "f"(sf_value_13));
                *reinterpret_cast<unsigned char*>(_gst_204) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_205 & 127));
            }
        }
        #pragma unroll
        for (int j_43 = 0; j_43 < 8; j_43++) {
            values[j_43] = 0.0f;
            normalized[j_43] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_32[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_32[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_32[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_32[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_32[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2560 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_32)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_33[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_33[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_33[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_33[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_33[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(2560 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_33)[0];
            }
        }
        #pragma unroll
        for (int j_44 = 0; j_44 < 8; j_44++) {
            absolute[j_44] = values[j_44];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_16 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_16 = max_noftz(amax_16, absolute[_lr]);
        }
        float _shfl_xor_sync_206 = __shfl_xor_sync(0xFFFFFFFF, amax_16, 1);
        amax_16 = max_noftz(amax_16, _shfl_xor_sync_206);
        float sf_value_17 = global_scale * (amax_16 * rcp_six);
        uint16_t _e4m3x2_208;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_208) : "f"(0.0f), "f"(sf_value_17));
        uint8_t _cvt_satfinite_e4m3_f32_207 = (uint8_t)_e4m3x2_208;
        uint32_t _f16x2_decode_209;
        float sf_rounded_18;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_209) : "h"(_e4m3x2_208));
        uint16_t _f16_decode_209 = (uint16_t)_f16x2_decode_209;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_18) : "h"(_f16_decode_209));
        float _rcp_210 = approx_rcp(sf_rounded_18 * rcp_global_scale);
        float output_scale_19 = fminf(_rcp_210, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_7 = make_float2(output_scale_19, output_scale_19);
            #pragma unroll
            for (int pair_38 = 0; pair_38 < 4; pair_38++) {
                int quant_offset_16 = pair_38 * 2;
                float2 _f32x2_mul_a_211 = make_float2(values[quant_offset_16], values[quant_offset_16 + 1]);
                float2 quant_pair_16;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_211), "l"(*(const unsigned long long*)&output_pair_7));
                quant_values[quant_offset_16] = quant_pair_16.x;
                quant_values[quant_offset_16 + 1] = quant_pair_16.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_212 = q_ptr + 1280;
                __builtin_assume(__isGlobal(_gst_212));
                *(reinterpret_cast<int*>(_gst_212) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_213 = sf_ptr + 20480;
                __builtin_assume(__isGlobal(_gst_213));
                uint16_t _cvt_satfinite_e4m3x2_f32_214;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_214) : "f"(0.0f), "f"(sf_value_17));
                *reinterpret_cast<unsigned char*>(_gst_213) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_214 & 127));
            }
        }
        #pragma unroll
        for (int j_45 = 0; j_45 < 8; j_45++) {
            values[j_45] = 0.0f;
            normalized[j_45] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_34[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_34[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_34[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_34[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_34[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3072 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_34)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_35[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_35[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_35[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_35[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_35[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3072 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_35)[0];
            }
        }
        #pragma unroll
        for (int j_46 = 0; j_46 < 8; j_46++) {
            absolute[j_46] = values[j_46];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_20 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_20 = max_noftz(amax_20, absolute[_lr]);
        }
        float _shfl_xor_sync_215 = __shfl_xor_sync(0xFFFFFFFF, amax_20, 1);
        amax_20 = max_noftz(amax_20, _shfl_xor_sync_215);
        float sf_value_21 = global_scale * (amax_20 * rcp_six);
        uint16_t _e4m3x2_217;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_217) : "f"(0.0f), "f"(sf_value_21));
        uint8_t _cvt_satfinite_e4m3_f32_216 = (uint8_t)_e4m3x2_217;
        uint32_t _f16x2_decode_218;
        float sf_rounded_22;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_218) : "h"(_e4m3x2_217));
        uint16_t _f16_decode_218 = (uint16_t)_f16x2_decode_218;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_22) : "h"(_f16_decode_218));
        float _rcp_219 = approx_rcp(sf_rounded_22 * rcp_global_scale);
        float output_scale_23 = fminf(_rcp_219, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_8 = make_float2(output_scale_23, output_scale_23);
            #pragma unroll
            for (int pair_39 = 0; pair_39 < 4; pair_39++) {
                int quant_offset_17 = pair_39 * 2;
                float2 _f32x2_mul_a_220 = make_float2(values[quant_offset_17], values[quant_offset_17 + 1]);
                float2 quant_pair_17;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_220), "l"(*(const unsigned long long*)&output_pair_8));
                quant_values[quant_offset_17] = quant_pair_17.x;
                quant_values[quant_offset_17 + 1] = quant_pair_17.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_221 = q_ptr + 1536;
                __builtin_assume(__isGlobal(_gst_221));
                *(reinterpret_cast<int*>(_gst_221) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_222 = sf_ptr + 24576;
                __builtin_assume(__isGlobal(_gst_222));
                uint16_t _cvt_satfinite_e4m3x2_f32_223;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_223) : "f"(0.0f), "f"(sf_value_21));
                *reinterpret_cast<unsigned char*>(_gst_222) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_223 & 127));
            }
        }
        #pragma unroll
        for (int j_47 = 0; j_47 < 8; j_47++) {
            values[j_47] = 0.0f;
            normalized[j_47] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_36[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_36[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_36[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_36[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_36[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3584 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_36)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_37[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_37[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_37[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_37[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_37[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(3584 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_37)[0];
            }
        }
        #pragma unroll
        for (int j_48 = 0; j_48 < 8; j_48++) {
            absolute[j_48] = values[j_48];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_24 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_24 = max_noftz(amax_24, absolute[_lr]);
        }
        float _shfl_xor_sync_224 = __shfl_xor_sync(0xFFFFFFFF, amax_24, 1);
        amax_24 = max_noftz(amax_24, _shfl_xor_sync_224);
        float sf_value_25 = global_scale * (amax_24 * rcp_six);
        uint16_t _e4m3x2_226;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_226) : "f"(0.0f), "f"(sf_value_25));
        uint8_t _cvt_satfinite_e4m3_f32_225 = (uint8_t)_e4m3x2_226;
        uint32_t _f16x2_decode_227;
        float sf_rounded_26;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_227) : "h"(_e4m3x2_226));
        uint16_t _f16_decode_227 = (uint16_t)_f16x2_decode_227;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_26) : "h"(_f16_decode_227));
        float _rcp_228 = approx_rcp(sf_rounded_26 * rcp_global_scale);
        float output_scale_27 = fminf(_rcp_228, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_10 = make_float2(output_scale_27, output_scale_27);
            #pragma unroll
            for (int pair_40 = 0; pair_40 < 4; pair_40++) {
                int quant_offset_18 = pair_40 * 2;
                float2 _f32x2_mul_a_229 = make_float2(values[quant_offset_18], values[quant_offset_18 + 1]);
                float2 quant_pair_18;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_229), "l"(*(const unsigned long long*)&output_pair_10));
                quant_values[quant_offset_18] = quant_pair_18.x;
                quant_values[quant_offset_18 + 1] = quant_pair_18.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_230 = q_ptr + 1792;
                __builtin_assume(__isGlobal(_gst_230));
                *(reinterpret_cast<int*>(_gst_230) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_231 = sf_ptr + 28672;
                __builtin_assume(__isGlobal(_gst_231));
                uint16_t _cvt_satfinite_e4m3x2_f32_232;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_232) : "f"(0.0f), "f"(sf_value_25));
                *reinterpret_cast<unsigned char*>(_gst_231) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_232 & 127));
            }
        }
        #pragma unroll
        for (int j_49 = 0; j_49 < 8; j_49++) {
            values[j_49] = 0.0f;
            normalized[j_49] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_38[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_38[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_38[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_38[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_38[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4096 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_38)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_39[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_39[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_39[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_39[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_39[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4096 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_39)[0];
            }
        }
        #pragma unroll
        for (int j_50 = 0; j_50 < 8; j_50++) {
            absolute[j_50] = values[j_50];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_28 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_28 = max_noftz(amax_28, absolute[_lr]);
        }
        float _shfl_xor_sync_233 = __shfl_xor_sync(0xFFFFFFFF, amax_28, 1);
        amax_28 = max_noftz(amax_28, _shfl_xor_sync_233);
        float sf_value_29 = global_scale * (amax_28 * rcp_six);
        uint16_t _e4m3x2_235;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_235) : "f"(0.0f), "f"(sf_value_29));
        uint8_t _cvt_satfinite_e4m3_f32_234 = (uint8_t)_e4m3x2_235;
        uint32_t _f16x2_decode_236;
        float sf_rounded_30;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_236) : "h"(_e4m3x2_235));
        uint16_t _f16_decode_236 = (uint16_t)_f16x2_decode_236;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_30) : "h"(_f16_decode_236));
        float _rcp_237 = approx_rcp(sf_rounded_30 * rcp_global_scale);
        float output_scale_31 = fminf(_rcp_237, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_11 = make_float2(output_scale_31, output_scale_31);
            #pragma unroll
            for (int pair_41 = 0; pair_41 < 4; pair_41++) {
                int quant_offset_19 = pair_41 * 2;
                float2 _f32x2_mul_a_238 = make_float2(values[quant_offset_19], values[quant_offset_19 + 1]);
                float2 quant_pair_19;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_238), "l"(*(const unsigned long long*)&output_pair_11));
                quant_values[quant_offset_19] = quant_pair_19.x;
                quant_values[quant_offset_19 + 1] = quant_pair_19.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_239 = q_ptr + 2048;
                __builtin_assume(__isGlobal(_gst_239));
                *(reinterpret_cast<int*>(_gst_239) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_240 = sf_ptr + 32768;
                __builtin_assume(__isGlobal(_gst_240));
                uint16_t _cvt_satfinite_e4m3x2_f32_241;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_241) : "f"(0.0f), "f"(sf_value_29));
                *reinterpret_cast<unsigned char*>(_gst_240) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_241 & 127));
            }
        }
        #pragma unroll
        for (int j_51 = 0; j_51 < 8; j_51++) {
            values[j_51] = 0.0f;
            normalized[j_51] = 0.0f;
        }
        if (row < M) {
            if (write_debug == 2) {
                unsigned int _packed_40[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                    _packed_40[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                    _packed_40[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                    _packed_40[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                    _packed_40[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4608 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_40)[0];
            } else if (write_debug != 0) {
                unsigned int _packed_41[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                    _packed_41[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                    _packed_41[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                    _packed_41[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                    _packed_41[_lp + 3] = *(uint32_t*)&_bf2;
                }
                reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(4608 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_41)[0];
            }
        }
        #pragma unroll
        for (int j_52 = 0; j_52 < 8; j_52++) {
            absolute[j_52] = values[j_52];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_32 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_32 = max_noftz(amax_32, absolute[_lr]);
        }
        float _shfl_xor_sync_242 = __shfl_xor_sync(0xFFFFFFFF, amax_32, 1);
        amax_32 = max_noftz(amax_32, _shfl_xor_sync_242);
        float sf_value_33 = global_scale * (amax_32 * rcp_six);
        uint16_t _e4m3x2_244;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_244) : "f"(0.0f), "f"(sf_value_33));
        uint8_t _cvt_satfinite_e4m3_f32_243 = (uint8_t)_e4m3x2_244;
        uint32_t _f16x2_decode_245;
        float sf_rounded_34;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_245) : "h"(_e4m3x2_244));
        uint16_t _f16_decode_245 = (uint16_t)_f16x2_decode_245;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_34) : "h"(_f16_decode_245));
        float _rcp_246 = approx_rcp(sf_rounded_34 * rcp_global_scale);
        float output_scale_35 = fminf(_rcp_246, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_12 = make_float2(output_scale_35, output_scale_35);
            #pragma unroll
            for (int pair_42 = 0; pair_42 < 4; pair_42++) {
                int quant_offset_20 = pair_42 * 2;
                float2 _f32x2_mul_a_247 = make_float2(values[quant_offset_20], values[quant_offset_20 + 1]);
                float2 quant_pair_20;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_247), "l"(*(const unsigned long long*)&output_pair_12));
                quant_values[quant_offset_20] = quant_pair_20.x;
                quant_values[quant_offset_20 + 1] = quant_pair_20.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_248 = q_ptr + 2304;
                __builtin_assume(__isGlobal(_gst_248));
                *(reinterpret_cast<int*>(_gst_248) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_249 = sf_ptr + 36864;
                __builtin_assume(__isGlobal(_gst_249));
                uint16_t _cvt_satfinite_e4m3x2_f32_250;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_250) : "f"(0.0f), "f"(sf_value_33));
                *reinterpret_cast<unsigned char*>(_gst_249) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_250 & 127));
            }
        }
        if (t < 32) {
            #pragma unroll
            for (int j_53 = 0; j_53 < 8; j_53++) {
                values[j_53] = 0.0f;
                normalized[j_53] = 0.0f;
            }
            if (row < M) {
                if (write_debug == 2) {
                    unsigned int _packed_42[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 0], normalized[_lp*2+1 + 0]));
                        _packed_42[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 2], normalized[_lp*2+1 + 2]));
                        _packed_42[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 4], normalized[_lp*2+1 + 4]));
                        _packed_42[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized[_lp*2 + 6], normalized[_lp*2+1 + 6]));
                        _packed_42[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(5120 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_42)[0];
                } else if (write_debug != 0) {
                    unsigned int _packed_43[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 0], values[_lp*2+1 + 0]));
                        _packed_43[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 2], values[_lp*2+1 + 2]));
                        _packed_43[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 4], values[_lp*2+1 + 4]));
                        _packed_43[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(values[_lp*2 + 6], values[_lp*2+1 + 6]));
                        _packed_43[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(debug_adaln_bf16 + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(5120 + t * 8)))[0] = reinterpret_cast<int4*>(_packed_43)[0];
                }
            }
            #pragma unroll
            for (int j_54 = 0; j_54 < 8; j_54++) {
                absolute[j_54] = values[j_54];
            }
            absolute[0] = fabsf(absolute[0]);
            absolute[1] = fabsf(absolute[1]);
            absolute[2] = fabsf(absolute[2]);
            absolute[3] = fabsf(absolute[3]);
            absolute[4] = fabsf(absolute[4]);
            absolute[5] = fabsf(absolute[5]);
            absolute[6] = fabsf(absolute[6]);
            absolute[7] = fabsf(absolute[7]);
            float amax_1_1 = absolute[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                amax_1_1 = max_noftz(amax_1_1, absolute[_lr]);
            }
            float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, amax_1_1, 1);
            amax_1_1 = max_noftz(amax_1_1, _shfl_xor_sync_251);
            float sf_value_2 = global_scale * (amax_1_1 * rcp_six);
            uint16_t _e4m3x2_253;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_253) : "f"(0.0f), "f"(sf_value_2));
            uint8_t _cvt_satfinite_e4m3_f32_252 = (uint8_t)_e4m3x2_253;
            uint32_t _f16x2_decode_254;
            float sf_rounded_3;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_254) : "h"(_e4m3x2_253));
            uint16_t _f16_decode_254 = (uint16_t)_f16x2_decode_254;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_3) : "h"(_f16_decode_254));
            float _rcp_255 = approx_rcp(sf_rounded_3 * rcp_global_scale);
            float output_scale_4 = fminf(_rcp_255, 3.4028235e+38f);
            if (row < M) {
                float2 output_pair_13 = make_float2(output_scale_4, output_scale_4);
                #pragma unroll
                for (int pair_43 = 0; pair_43 < 4; pair_43++) {
                    int quant_offset_21 = pair_43 * 2;
                    float2 _f32x2_mul_a_256 = make_float2(values[quant_offset_21], values[quant_offset_21 + 1]);
                    float2 quant_pair_21;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_256), "l"(*(const unsigned long long*)&output_pair_13));
                    quant_values[quant_offset_21] = quant_pair_21.x;
                    quant_values[quant_offset_21 + 1] = quant_pair_21.y;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                {
                    auto _gst_257 = q_ptr + 2560;
                    __builtin_assume(__isGlobal(_gst_257));
                    *(reinterpret_cast<int*>(_gst_257) + (0)) = packed[0];
                }
            }
            if (t % 2 == 0) {
                {
                    auto _gst_258 = sf_ptr + 40960;
                    __builtin_assume(__isGlobal(_gst_258));
                    uint16_t _cvt_satfinite_e4m3x2_f32_259;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_259) : "f"(0.0f), "f"(sf_value_2));
                    *reinterpret_cast<unsigned char*>(_gst_258) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_259 & 127));
                }
            }
        }
    }
}

} // extern "C"
