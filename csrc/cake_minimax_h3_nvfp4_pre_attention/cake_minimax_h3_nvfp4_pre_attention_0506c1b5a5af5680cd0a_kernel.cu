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
#define SMEM_X_STAGE_OFF 64
#define SMEM_X_STAGE_STAGE_BYTES 21504
#define SMEM_X_STAGE_STRIDE 21504
#define SMEM_TOTAL 21632
#define THREADS 128
#define HIDDEN 5376
#define SCALE_COLS 336
#define PACKED_COLS 2688
#define LAUNCH_MIN_BLOCKS 8

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
kernel_cake_minimax_h3_nvfp4_pre_attention_0506c1b5a5af5680cd0a(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ x_norm_weight, __nv_bfloat16* __restrict__ adaln_scale, __nv_bfloat16* __restrict__ adaln_shift, int* __restrict__ adaln_index, float* __restrict__ x_global_scale, uint8_t* __restrict__ activation_q, uint8_t* __restrict__ activation_sf, __nv_bfloat16* __restrict__ debug_adaln_bf16, int write_debug, float eps, int M, uint8_t* __restrict__ out_sf, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    uint8_t* x_stage = reinterpret_cast<uint8_t*>(smem_raw + 64);
    const int x_stage_addr = smem + 64;

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
    unsigned int x_base = x_stage_addr + (unsigned int)(row_in_cta * 10752 + t * 16);
    if (row < M) {
        __nv_bfloat16* x_row = x + ((unsigned long long)row * (unsigned long long)HIDDEN + (unsigned long long)(t * 8));
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base), "l"(x_row + 0));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 1024), "l"(x_row + 512));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 2048), "l"(x_row + 1024));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 3072), "l"(x_row + 1536));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 4096), "l"(x_row + 2048));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 5120), "l"(x_row + 2560));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 6144), "l"(x_row + 3072));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 7168), "l"(x_row + 3584));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 8192), "l"(x_row + 4096));
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
            :: "r"(x_base + 9216), "l"(x_row + 4608));
        asm volatile("cp.async.commit_group;");
        if (t < 32) {
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                :: "r"(x_base + 10240), "l"(x_row + 5120));
        }
        asm volatile("cp.async.commit_group;");
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
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w_ptr + 0 + 0);
        uint4 _vld_0[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_0[_blk] = _vptr_0[_blk];
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&_vec_load_0[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
    }
    unsigned int tab_s[4];
    {
        const uint4* _vptr_1 = reinterpret_cast<const uint4*>(s_ptr + 0);
        uint4* _vdst_1 = reinterpret_cast<uint4*>(&tab_s[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_1[_blk] = _vptr_1[_blk];
        }
    }
    float _vec_load_1[8];
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(b_ptr + 0 + 0);
        uint4 _vld_2[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_2[_blk] = _vptr_2[_blk];
            uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                (&_vec_load_1[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
            }
        }
    }
    float sum_sq = 0.0f;
    if (row < M) {
        asm volatile("cp.async.wait_group 10;");
        unsigned int words[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
            : "r"(x_base));
        float _unpack_bf16_0[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
            (&_unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            sum_sq = __fmaf_rn(_unpack_bf16_0[i], _unpack_bf16_0[i], sum_sq);
        }
        asm volatile("cp.async.wait_group 9;");
        unsigned int words_0[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_0[(0) + 3]))
            : "r"(x_base + 1024));
        float _unpack_bf16_1[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) << 16);
            (&_unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_1 = 0; i_1 < 8; i_1++) {
            sum_sq = __fmaf_rn(_unpack_bf16_1[i_1], _unpack_bf16_1[i_1], sum_sq);
        }
        asm volatile("cp.async.wait_group 8;");
        unsigned int words_1[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_1[(0) + 3]))
            : "r"(x_base + 2048));
        float _unpack_bf16_2[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) << 16);
            (&_unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_1[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_2 = 0; i_2 < 8; i_2++) {
            sum_sq = __fmaf_rn(_unpack_bf16_2[i_2], _unpack_bf16_2[i_2], sum_sq);
        }
        asm volatile("cp.async.wait_group 7;");
        unsigned int words_2[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_2[(0) + 3]))
            : "r"(x_base + 3072));
        float _unpack_bf16_3[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_2[_pair]) << 16);
            (&_unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_2[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_3 = 0; i_3 < 8; i_3++) {
            sum_sq = __fmaf_rn(_unpack_bf16_3[i_3], _unpack_bf16_3[i_3], sum_sq);
        }
        asm volatile("cp.async.wait_group 6;");
        unsigned int words_3[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3[(0) + 3]))
            : "r"(x_base + 4096));
        float _unpack_bf16_4[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_3[_pair]) << 16);
            (&_unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_3[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_4 = 0; i_4 < 8; i_4++) {
            sum_sq = __fmaf_rn(_unpack_bf16_4[i_4], _unpack_bf16_4[i_4], sum_sq);
        }
        asm volatile("cp.async.wait_group 5;");
        unsigned int words_4[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_4[(0) + 3]))
            : "r"(x_base + 5120));
        float _unpack_bf16_5[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) << 16);
            (&_unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_4[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_5 = 0; i_5 < 8; i_5++) {
            sum_sq = __fmaf_rn(_unpack_bf16_5[i_5], _unpack_bf16_5[i_5], sum_sq);
        }
        asm volatile("cp.async.wait_group 4;");
        unsigned int words_5[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_5[(0) + 3]))
            : "r"(x_base + 6144));
        float _unpack_bf16_6[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) << 16);
            (&_unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_5[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_6 = 0; i_6 < 8; i_6++) {
            sum_sq = __fmaf_rn(_unpack_bf16_6[i_6], _unpack_bf16_6[i_6], sum_sq);
        }
        asm volatile("cp.async.wait_group 3;");
        unsigned int words_6[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_6[(0) + 3]))
            : "r"(x_base + 7168));
        float _unpack_bf16_7[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_6[_pair]) << 16);
            (&_unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_6[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_7 = 0; i_7 < 8; i_7++) {
            sum_sq = __fmaf_rn(_unpack_bf16_7[i_7], _unpack_bf16_7[i_7], sum_sq);
        }
        asm volatile("cp.async.wait_group 2;");
        unsigned int words_7[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_7[(0) + 3]))
            : "r"(x_base + 8192));
        float _unpack_bf16_8[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) << 16);
            (&_unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_7[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_8 = 0; i_8 < 8; i_8++) {
            sum_sq = __fmaf_rn(_unpack_bf16_8[i_8], _unpack_bf16_8[i_8], sum_sq);
        }
        asm volatile("cp.async.wait_group 1;");
        unsigned int words_8[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_8[(0) + 3]))
            : "r"(x_base + 9216));
        float _unpack_bf16_9[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) << 16);
            (&_unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_8[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int i_9 = 0; i_9 < 8; i_9++) {
            sum_sq = __fmaf_rn(_unpack_bf16_9[i_9], _unpack_bf16_9[i_9], sum_sq);
        }
        if (t < 32) {
            asm volatile("cp.async.wait_group 0;");
            unsigned int words_9[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_9[(0) + 3]))
                : "r"(x_base + 10240));
            float _unpack_bf16_10[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) << 16);
                (&_unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_9[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int i_10 = 0; i_10 < 8; i_10++) {
                sum_sq = __fmaf_rn(_unpack_bf16_10[i_10], _unpack_bf16_10[i_10], sum_sq);
            }
        }
    }
    float _shfl_xor_sync_3 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 1);
    sum_sq += _shfl_xor_sync_3;
    float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 2);
    sum_sq += _shfl_xor_sync_4;
    float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 4);
    sum_sq += _shfl_xor_sync_5;
    float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 8);
    sum_sq += _shfl_xor_sync_6;
    float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, sum_sq, 16);
    sum_sq += _shfl_xor_sync_7;
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
    float high_only[16];
    if (row_valid != 0) {
        float _vec_load_2[8];
        {
            const uint4* _vptr_8 = reinterpret_cast<const uint4*>(w_ptr + 512 + 0);
            uint4 _vld_8[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_8[_blk] = _vptr_8[_blk];
                uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                    (&_vec_load_2[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_0[4];
        {
            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(s_ptr + 512);
            uint4* _vdst_9 = reinterpret_cast<uint4*>(&tab_s_0[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_9[_blk] = _vptr_9[_blk];
            }
        }
        float _vec_load_3[8];
        {
            const uint4* _vptr_10 = reinterpret_cast<const uint4*>(b_ptr + 512 + 0);
            uint4 _vld_10[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_10[_blk] = _vptr_10[_blk];
                uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                    (&_vec_load_3[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_10[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_10[(0) + 3]))
            : "r"(x_base));
        float _unpack_bf16_11[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) << 16);
            (&_unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_10[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair = 0; pair < 4; pair++) {
            int mul_offset = pair * 2;
            float2 _f32x2_mul_a_11 = make_float2(_unpack_bf16_11[mul_offset], _unpack_bf16_11[mul_offset + 1]);
            float2 x_rstd;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_12 = make_float2(_vec_load_0[mul_offset], _vec_load_0[mul_offset + 1]);
            float2 x_rstd_w;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w) : "l"(*(const unsigned long long*)&x_rstd), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
            scaled[mul_offset] = x_rstd_w.x;
            scaled[mul_offset + 1] = x_rstd_w.y;
        }
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            high_only[2 * j] = 0.0f;
            high_only[2 * j + 1] = scaled[j];
        }
        unsigned int high_only_bf16[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            normalized[j_1] = __uint_as_float(high_only_bf16[j_1]);
        }
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            uint32_t _bf16x2_add_0;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_0) : "r"(tab_s[q]), "r"(one_pair));
            scale_plus_one[q] = _bf16x2_add_0;
        }
        float _unpack_bf16_12[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_1 = 0; pair_1 < 4; pair_1++) {
            int ada_offset = pair_1 * 2;
            float2 adaln_pair = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset], normalized[ada_offset + 1]), make_float2(_unpack_bf16_12[ada_offset], _unpack_bf16_12[ada_offset + 1]), make_float2(_vec_load_1[ada_offset], _vec_load_1[ada_offset + 1]));
            adaln[ada_offset] = adaln_pair.x;
            adaln[ada_offset + 1] = adaln_pair.y;
        }
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            high_only[2 * j_2] = 0.0f;
            high_only[2 * j_2 + 1] = adaln[j_2];
        }
        unsigned int high_only_bf16_1[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_1[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            values[j_3] = __uint_as_float(high_only_bf16_1[j_3]);
        }
        if (write_debug == 2) {
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
        for (int j_4 = 0; j_4 < 8; j_4++) {
            absolute[j_4] = values[j_4];
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
        float _shfl_xor_sync_13 = __shfl_xor_sync(0xFFFFFFFF, amax, 1);
        amax = max_noftz(amax, _shfl_xor_sync_13);
        float sf_value = global_scale * (amax * rcp_six);
        uint16_t _e4m3x2_15;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_15) : "f"(0.0f), "f"(sf_value));
        uint8_t _cvt_satfinite_e4m3_f32_14 = (uint8_t)_e4m3x2_15;
        uint32_t _f16x2_decode_16;
        float sf_rounded;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_16) : "h"(_e4m3x2_15));
        uint16_t _f16_decode_16 = (uint16_t)_f16x2_decode_16;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded) : "h"(_f16_decode_16));
        float _rcp_17 = approx_rcp(sf_rounded * rcp_global_scale);
        float output_scale = fminf(_rcp_17, 3.4028235e+38f);
        float2 output_pair = make_float2(output_scale, output_scale);
        #pragma unroll
        for (int pair_2 = 0; pair_2 < 4; pair_2++) {
            int quant_offset = pair_2 * 2;
            float2 _f32x2_mul_a_18 = make_float2(values[quant_offset], values[quant_offset + 1]);
            float2 quant_pair;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&output_pair));
            quant_values[quant_offset] = quant_pair.x;
            quant_values[quant_offset + 1] = quant_pair.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_19 = q_ptr + 0;
            __builtin_assume(__isGlobal(_gst_19));
            *(reinterpret_cast<int*>(_gst_19) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_20 = sf_ptr + 0;
                __builtin_assume(__isGlobal(_gst_20));
                uint16_t _cvt_satfinite_e4m3x2_f32_21;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_21) : "f"(0.0f), "f"(sf_value));
                *reinterpret_cast<unsigned char*>(_gst_20) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_21 & 127));
            }
        }
        float _vec_load_4[8];
        {
            const uint4* _vptr_22 = reinterpret_cast<const uint4*>(w_ptr + 1024 + 0);
            uint4 _vld_22[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_22[_blk] = _vptr_22[_blk];
                uint32_t* _vpairs_22 = reinterpret_cast<uint32_t*>(&_vld_22[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) << 16);
                    (&_vec_load_4[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_22[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_2[4];
        {
            const uint4* _vptr_23 = reinterpret_cast<const uint4*>(s_ptr + 1024);
            uint4* _vdst_23 = reinterpret_cast<uint4*>(&tab_s_2[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_23[_blk] = _vptr_23[_blk];
            }
        }
        float _vec_load_5[8];
        {
            const uint4* _vptr_24 = reinterpret_cast<const uint4*>(b_ptr + 1024 + 0);
            uint4 _vld_24[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_24[_blk] = _vptr_24[_blk];
                uint32_t* _vpairs_24 = reinterpret_cast<uint32_t*>(&_vld_24[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) << 16);
                    (&_vec_load_5[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_24[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_3_1[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_3_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_3_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_3_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_3_1[(0) + 3]))
            : "r"(x_base + 1024));
        float _unpack_bf16_13[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_3_1[_pair]) << 16);
            (&_unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_3_1[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_3 = 0; pair_3 < 4; pair_3++) {
            int mul_offset_1 = pair_3 * 2;
            float2 _f32x2_mul_a_25 = make_float2(_unpack_bf16_13[mul_offset_1], _unpack_bf16_13[mul_offset_1 + 1]);
            float2 x_rstd_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_26 = make_float2(_vec_load_2[mul_offset_1], _vec_load_2[mul_offset_1 + 1]);
            float2 x_rstd_w_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_1) : "l"(*(const unsigned long long*)&x_rstd_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
            scaled[mul_offset_1] = x_rstd_w_1.x;
            scaled[mul_offset_1 + 1] = x_rstd_w_1.y;
        }
        #pragma unroll
        for (int j_5 = 0; j_5 < 8; j_5++) {
            high_only[2 * j_5] = 0.0f;
            high_only[2 * j_5 + 1] = scaled[j_5];
        }
        unsigned int high_only_bf16_4[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_4[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_6 = 0; j_6 < 8; j_6++) {
            normalized[j_6] = __uint_as_float(high_only_bf16_4[j_6]);
        }
        #pragma unroll
        for (int q_1 = 0; q_1 < 4; q_1++) {
            uint32_t _bf16x2_add_1;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_1) : "r"(tab_s_0[q_1]), "r"(one_pair));
            scale_plus_one[q_1] = _bf16x2_add_1;
        }
        float _unpack_bf16_14[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_14[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_14[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_4 = 0; pair_4 < 4; pair_4++) {
            int ada_offset_1 = pair_4 * 2;
            float2 adaln_pair_1 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_1], normalized[ada_offset_1 + 1]), make_float2(_unpack_bf16_14[ada_offset_1], _unpack_bf16_14[ada_offset_1 + 1]), make_float2(_vec_load_3[ada_offset_1], _vec_load_3[ada_offset_1 + 1]));
            adaln[ada_offset_1] = adaln_pair_1.x;
            adaln[ada_offset_1 + 1] = adaln_pair_1.y;
        }
        #pragma unroll
        for (int j_7 = 0; j_7 < 8; j_7++) {
            high_only[2 * j_7] = 0.0f;
            high_only[2 * j_7 + 1] = adaln[j_7];
        }
        unsigned int high_only_bf16_5[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_5[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_8 = 0; j_8 < 8; j_8++) {
            values[j_8] = __uint_as_float(high_only_bf16_5[j_8]);
        }
        if (write_debug == 2) {
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
        for (int j_9 = 0; j_9 < 8; j_9++) {
            absolute[j_9] = values[j_9];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_6 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_6 = max_noftz(amax_6, absolute[_lr]);
        }
        float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, amax_6, 1);
        amax_6 = max_noftz(amax_6, _shfl_xor_sync_27);
        float sf_value_7 = global_scale * (amax_6 * rcp_six);
        uint16_t _e4m3x2_29;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(sf_value_7));
        uint8_t _cvt_satfinite_e4m3_f32_28 = (uint8_t)_e4m3x2_29;
        uint32_t _f16x2_decode_30;
        float sf_rounded_8;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_30) : "h"(_e4m3x2_29));
        uint16_t _f16_decode_30 = (uint16_t)_f16x2_decode_30;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_8) : "h"(_f16_decode_30));
        float _rcp_31 = approx_rcp(sf_rounded_8 * rcp_global_scale);
        float output_scale_9 = fminf(_rcp_31, 3.4028235e+38f);
        float2 output_pair_10 = make_float2(output_scale_9, output_scale_9);
        #pragma unroll
        for (int pair_5 = 0; pair_5 < 4; pair_5++) {
            int quant_offset_1 = pair_5 * 2;
            float2 _f32x2_mul_a_32 = make_float2(values[quant_offset_1], values[quant_offset_1 + 1]);
            float2 quant_pair_1;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_32), "l"(*(const unsigned long long*)&output_pair_10));
            quant_values[quant_offset_1] = quant_pair_1.x;
            quant_values[quant_offset_1 + 1] = quant_pair_1.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_33 = q_ptr + 256;
            __builtin_assume(__isGlobal(_gst_33));
            *(reinterpret_cast<int*>(_gst_33) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_34 = sf_ptr + 4096;
                __builtin_assume(__isGlobal(_gst_34));
                uint16_t _cvt_satfinite_e4m3x2_f32_35;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_35) : "f"(0.0f), "f"(sf_value_7));
                *reinterpret_cast<unsigned char*>(_gst_34) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_35 & 127));
            }
        }
        float _vec_load_6[8];
        {
            const uint4* _vptr_36 = reinterpret_cast<const uint4*>(w_ptr + 1536 + 0);
            uint4 _vld_36[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_36[_blk] = _vptr_36[_blk];
                uint32_t* _vpairs_36 = reinterpret_cast<uint32_t*>(&_vld_36[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) << 16);
                    (&_vec_load_6[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_36[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_11[4];
        {
            const uint4* _vptr_37 = reinterpret_cast<const uint4*>(s_ptr + 1536);
            uint4* _vdst_37 = reinterpret_cast<uint4*>(&tab_s_11[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_37[_blk] = _vptr_37[_blk];
            }
        }
        float _vec_load_7[8];
        {
            const uint4* _vptr_38 = reinterpret_cast<const uint4*>(b_ptr + 1536 + 0);
            uint4 _vld_38[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_38[_blk] = _vptr_38[_blk];
                uint32_t* _vpairs_38 = reinterpret_cast<uint32_t*>(&_vld_38[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) << 16);
                    (&_vec_load_7[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_38[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_12[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_12[(0) + 3]))
            : "r"(x_base + 2048));
        float _unpack_bf16_15[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_15[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_12[_pair]) << 16);
            (&_unpack_bf16_15[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_12[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_6 = 0; pair_6 < 4; pair_6++) {
            int mul_offset_2 = pair_6 * 2;
            float2 _f32x2_mul_a_39 = make_float2(_unpack_bf16_15[mul_offset_2], _unpack_bf16_15[mul_offset_2 + 1]);
            float2 x_rstd_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_40 = make_float2(_vec_load_4[mul_offset_2], _vec_load_4[mul_offset_2 + 1]);
            float2 x_rstd_w_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_2) : "l"(*(const unsigned long long*)&x_rstd_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
            scaled[mul_offset_2] = x_rstd_w_2.x;
            scaled[mul_offset_2 + 1] = x_rstd_w_2.y;
        }
        #pragma unroll
        for (int j_10 = 0; j_10 < 8; j_10++) {
            high_only[2 * j_10] = 0.0f;
            high_only[2 * j_10 + 1] = scaled[j_10];
        }
        unsigned int high_only_bf16_13[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_13[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_11 = 0; j_11 < 8; j_11++) {
            normalized[j_11] = __uint_as_float(high_only_bf16_13[j_11]);
        }
        #pragma unroll
        for (int q_2 = 0; q_2 < 4; q_2++) {
            uint32_t _bf16x2_add_2;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_2) : "r"(tab_s_2[q_2]), "r"(one_pair));
            scale_plus_one[q_2] = _bf16x2_add_2;
        }
        float _unpack_bf16_16[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_7 = 0; pair_7 < 4; pair_7++) {
            int ada_offset_2 = pair_7 * 2;
            float2 adaln_pair_2 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_2], normalized[ada_offset_2 + 1]), make_float2(_unpack_bf16_16[ada_offset_2], _unpack_bf16_16[ada_offset_2 + 1]), make_float2(_vec_load_5[ada_offset_2], _vec_load_5[ada_offset_2 + 1]));
            adaln[ada_offset_2] = adaln_pair_2.x;
            adaln[ada_offset_2 + 1] = adaln_pair_2.y;
        }
        #pragma unroll
        for (int j_12 = 0; j_12 < 8; j_12++) {
            high_only[2 * j_12] = 0.0f;
            high_only[2 * j_12 + 1] = adaln[j_12];
        }
        unsigned int high_only_bf16_14[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_14[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_13 = 0; j_13 < 8; j_13++) {
            values[j_13] = __uint_as_float(high_only_bf16_14[j_13]);
        }
        if (write_debug == 2) {
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
        float amax_15 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_15 = max_noftz(amax_15, absolute[_lr]);
        }
        float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, amax_15, 1);
        amax_15 = max_noftz(amax_15, _shfl_xor_sync_41);
        float sf_value_16 = global_scale * (amax_15 * rcp_six);
        uint16_t _e4m3x2_43;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_43) : "f"(0.0f), "f"(sf_value_16));
        uint8_t _cvt_satfinite_e4m3_f32_42 = (uint8_t)_e4m3x2_43;
        uint32_t _f16x2_decode_44;
        float sf_rounded_17;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_44) : "h"(_e4m3x2_43));
        uint16_t _f16_decode_44 = (uint16_t)_f16x2_decode_44;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_17) : "h"(_f16_decode_44));
        float _rcp_45 = approx_rcp(sf_rounded_17 * rcp_global_scale);
        float output_scale_18 = fminf(_rcp_45, 3.4028235e+38f);
        float2 output_pair_19 = make_float2(output_scale_18, output_scale_18);
        #pragma unroll
        for (int pair_8 = 0; pair_8 < 4; pair_8++) {
            int quant_offset_2 = pair_8 * 2;
            float2 _f32x2_mul_a_46 = make_float2(values[quant_offset_2], values[quant_offset_2 + 1]);
            float2 quant_pair_2;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_46), "l"(*(const unsigned long long*)&output_pair_19));
            quant_values[quant_offset_2] = quant_pair_2.x;
            quant_values[quant_offset_2 + 1] = quant_pair_2.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_47 = q_ptr + 512;
            __builtin_assume(__isGlobal(_gst_47));
            *(reinterpret_cast<int*>(_gst_47) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_48 = sf_ptr + 8192;
                __builtin_assume(__isGlobal(_gst_48));
                uint16_t _cvt_satfinite_e4m3x2_f32_49;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_49) : "f"(0.0f), "f"(sf_value_16));
                *reinterpret_cast<unsigned char*>(_gst_48) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_49 & 127));
            }
        }
        float _vec_load_8[8];
        {
            const uint4* _vptr_50 = reinterpret_cast<const uint4*>(w_ptr + 2048 + 0);
            uint4 _vld_50[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_50[_blk] = _vptr_50[_blk];
                uint32_t* _vpairs_50 = reinterpret_cast<uint32_t*>(&_vld_50[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) << 16);
                    (&_vec_load_8[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_20[4];
        {
            const uint4* _vptr_51 = reinterpret_cast<const uint4*>(s_ptr + 2048);
            uint4* _vdst_51 = reinterpret_cast<uint4*>(&tab_s_20[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_51[_blk] = _vptr_51[_blk];
            }
        }
        float _vec_load_9[8];
        {
            const uint4* _vptr_52 = reinterpret_cast<const uint4*>(b_ptr + 2048 + 0);
            uint4 _vld_52[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_52[_blk] = _vptr_52[_blk];
                uint32_t* _vpairs_52 = reinterpret_cast<uint32_t*>(&_vld_52[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) << 16);
                    (&_vec_load_9[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_21[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_21[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_21[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_21[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_21[(0) + 3]))
            : "r"(x_base + 3072));
        float _unpack_bf16_17[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_17[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_21[_pair]) << 16);
            (&_unpack_bf16_17[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_21[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_9 = 0; pair_9 < 4; pair_9++) {
            int mul_offset_3 = pair_9 * 2;
            float2 _f32x2_mul_a_53 = make_float2(_unpack_bf16_17[mul_offset_3], _unpack_bf16_17[mul_offset_3 + 1]);
            float2 x_rstd_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_54 = make_float2(_vec_load_6[mul_offset_3], _vec_load_6[mul_offset_3 + 1]);
            float2 x_rstd_w_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_3) : "l"(*(const unsigned long long*)&x_rstd_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
            scaled[mul_offset_3] = x_rstd_w_3.x;
            scaled[mul_offset_3 + 1] = x_rstd_w_3.y;
        }
        #pragma unroll
        for (int j_15 = 0; j_15 < 8; j_15++) {
            high_only[2 * j_15] = 0.0f;
            high_only[2 * j_15 + 1] = scaled[j_15];
        }
        unsigned int high_only_bf16_22[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_22[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_16 = 0; j_16 < 8; j_16++) {
            normalized[j_16] = __uint_as_float(high_only_bf16_22[j_16]);
        }
        #pragma unroll
        for (int q_3 = 0; q_3 < 4; q_3++) {
            uint32_t _bf16x2_add_3;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_3) : "r"(tab_s_11[q_3]), "r"(one_pair));
            scale_plus_one[q_3] = _bf16x2_add_3;
        }
        float _unpack_bf16_18[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_18[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_18[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_10 = 0; pair_10 < 4; pair_10++) {
            int ada_offset_3 = pair_10 * 2;
            float2 adaln_pair_3 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_3], normalized[ada_offset_3 + 1]), make_float2(_unpack_bf16_18[ada_offset_3], _unpack_bf16_18[ada_offset_3 + 1]), make_float2(_vec_load_7[ada_offset_3], _vec_load_7[ada_offset_3 + 1]));
            adaln[ada_offset_3] = adaln_pair_3.x;
            adaln[ada_offset_3 + 1] = adaln_pair_3.y;
        }
        #pragma unroll
        for (int j_17 = 0; j_17 < 8; j_17++) {
            high_only[2 * j_17] = 0.0f;
            high_only[2 * j_17 + 1] = adaln[j_17];
        }
        unsigned int high_only_bf16_23[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_23[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_18 = 0; j_18 < 8; j_18++) {
            values[j_18] = __uint_as_float(high_only_bf16_23[j_18]);
        }
        if (write_debug == 2) {
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
        for (int j_19 = 0; j_19 < 8; j_19++) {
            absolute[j_19] = values[j_19];
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
        float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, amax_24, 1);
        amax_24 = max_noftz(amax_24, _shfl_xor_sync_55);
        float sf_value_25 = global_scale * (amax_24 * rcp_six);
        uint16_t _e4m3x2_57;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(sf_value_25));
        uint8_t _cvt_satfinite_e4m3_f32_56 = (uint8_t)_e4m3x2_57;
        uint32_t _f16x2_decode_58;
        float sf_rounded_26;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_58) : "h"(_e4m3x2_57));
        uint16_t _f16_decode_58 = (uint16_t)_f16x2_decode_58;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_26) : "h"(_f16_decode_58));
        float _rcp_59 = approx_rcp(sf_rounded_26 * rcp_global_scale);
        float output_scale_27 = fminf(_rcp_59, 3.4028235e+38f);
        float2 output_pair_28 = make_float2(output_scale_27, output_scale_27);
        #pragma unroll
        for (int pair_11 = 0; pair_11 < 4; pair_11++) {
            int quant_offset_3 = pair_11 * 2;
            float2 _f32x2_mul_a_60 = make_float2(values[quant_offset_3], values[quant_offset_3 + 1]);
            float2 quant_pair_3;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&output_pair_28));
            quant_values[quant_offset_3] = quant_pair_3.x;
            quant_values[quant_offset_3 + 1] = quant_pair_3.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_61 = q_ptr + 768;
            __builtin_assume(__isGlobal(_gst_61));
            *(reinterpret_cast<int*>(_gst_61) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_62 = sf_ptr + 12288;
                __builtin_assume(__isGlobal(_gst_62));
                uint16_t _cvt_satfinite_e4m3x2_f32_63;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_63) : "f"(0.0f), "f"(sf_value_25));
                *reinterpret_cast<unsigned char*>(_gst_62) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_63 & 127));
            }
        }
        float _vec_load_10[8];
        {
            const uint4* _vptr_64 = reinterpret_cast<const uint4*>(w_ptr + 2560 + 0);
            uint4 _vld_64[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_64[_blk] = _vptr_64[_blk];
                uint32_t* _vpairs_64 = reinterpret_cast<uint32_t*>(&_vld_64[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) << 16);
                    (&_vec_load_10[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_64[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_29[4];
        {
            const uint4* _vptr_65 = reinterpret_cast<const uint4*>(s_ptr + 2560);
            uint4* _vdst_65 = reinterpret_cast<uint4*>(&tab_s_29[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_65[_blk] = _vptr_65[_blk];
            }
        }
        float _vec_load_11[8];
        {
            const uint4* _vptr_66 = reinterpret_cast<const uint4*>(b_ptr + 2560 + 0);
            uint4 _vld_66[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_66[_blk] = _vptr_66[_blk];
                uint32_t* _vpairs_66 = reinterpret_cast<uint32_t*>(&_vld_66[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) << 16);
                    (&_vec_load_11[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_66[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_30[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_30[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_30[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_30[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_30[(0) + 3]))
            : "r"(x_base + 4096));
        float _unpack_bf16_19[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_19[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_30[_pair]) << 16);
            (&_unpack_bf16_19[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_30[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_12 = 0; pair_12 < 4; pair_12++) {
            int mul_offset_4 = pair_12 * 2;
            float2 _f32x2_mul_a_67 = make_float2(_unpack_bf16_19[mul_offset_4], _unpack_bf16_19[mul_offset_4 + 1]);
            float2 x_rstd_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_68 = make_float2(_vec_load_8[mul_offset_4], _vec_load_8[mul_offset_4 + 1]);
            float2 x_rstd_w_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_4) : "l"(*(const unsigned long long*)&x_rstd_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
            scaled[mul_offset_4] = x_rstd_w_4.x;
            scaled[mul_offset_4 + 1] = x_rstd_w_4.y;
        }
        #pragma unroll
        for (int j_20 = 0; j_20 < 8; j_20++) {
            high_only[2 * j_20] = 0.0f;
            high_only[2 * j_20 + 1] = scaled[j_20];
        }
        unsigned int high_only_bf16_31[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_31[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_21 = 0; j_21 < 8; j_21++) {
            normalized[j_21] = __uint_as_float(high_only_bf16_31[j_21]);
        }
        #pragma unroll
        for (int q_4 = 0; q_4 < 4; q_4++) {
            uint32_t _bf16x2_add_4;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_4) : "r"(tab_s_20[q_4]), "r"(one_pair));
            scale_plus_one[q_4] = _bf16x2_add_4;
        }
        float _unpack_bf16_20[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_20[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_20[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_13 = 0; pair_13 < 4; pair_13++) {
            int ada_offset_4 = pair_13 * 2;
            float2 adaln_pair_4 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_4], normalized[ada_offset_4 + 1]), make_float2(_unpack_bf16_20[ada_offset_4], _unpack_bf16_20[ada_offset_4 + 1]), make_float2(_vec_load_9[ada_offset_4], _vec_load_9[ada_offset_4 + 1]));
            adaln[ada_offset_4] = adaln_pair_4.x;
            adaln[ada_offset_4 + 1] = adaln_pair_4.y;
        }
        #pragma unroll
        for (int j_22 = 0; j_22 < 8; j_22++) {
            high_only[2 * j_22] = 0.0f;
            high_only[2 * j_22 + 1] = adaln[j_22];
        }
        unsigned int high_only_bf16_32[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_32[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_23 = 0; j_23 < 8; j_23++) {
            values[j_23] = __uint_as_float(high_only_bf16_32[j_23]);
        }
        if (write_debug == 2) {
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
        for (int j_24 = 0; j_24 < 8; j_24++) {
            absolute[j_24] = values[j_24];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_33 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_33 = max_noftz(amax_33, absolute[_lr]);
        }
        float _shfl_xor_sync_69 = __shfl_xor_sync(0xFFFFFFFF, amax_33, 1);
        amax_33 = max_noftz(amax_33, _shfl_xor_sync_69);
        float sf_value_34 = global_scale * (amax_33 * rcp_six);
        uint16_t _e4m3x2_71;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_71) : "f"(0.0f), "f"(sf_value_34));
        uint8_t _cvt_satfinite_e4m3_f32_70 = (uint8_t)_e4m3x2_71;
        uint32_t _f16x2_decode_72;
        float sf_rounded_35;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_72) : "h"(_e4m3x2_71));
        uint16_t _f16_decode_72 = (uint16_t)_f16x2_decode_72;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_35) : "h"(_f16_decode_72));
        float _rcp_73 = approx_rcp(sf_rounded_35 * rcp_global_scale);
        float output_scale_36 = fminf(_rcp_73, 3.4028235e+38f);
        float2 output_pair_37 = make_float2(output_scale_36, output_scale_36);
        #pragma unroll
        for (int pair_14 = 0; pair_14 < 4; pair_14++) {
            int quant_offset_4 = pair_14 * 2;
            float2 _f32x2_mul_a_74 = make_float2(values[quant_offset_4], values[quant_offset_4 + 1]);
            float2 quant_pair_4;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&output_pair_37));
            quant_values[quant_offset_4] = quant_pair_4.x;
            quant_values[quant_offset_4 + 1] = quant_pair_4.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_75 = q_ptr + 1024;
            __builtin_assume(__isGlobal(_gst_75));
            *(reinterpret_cast<int*>(_gst_75) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_76 = sf_ptr + 16384;
                __builtin_assume(__isGlobal(_gst_76));
                uint16_t _cvt_satfinite_e4m3x2_f32_77;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_77) : "f"(0.0f), "f"(sf_value_34));
                *reinterpret_cast<unsigned char*>(_gst_76) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_77 & 127));
            }
        }
        float _vec_load_12[8];
        {
            const uint4* _vptr_78 = reinterpret_cast<const uint4*>(w_ptr + 3072 + 0);
            uint4 _vld_78[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_78[_blk] = _vptr_78[_blk];
                uint32_t* _vpairs_78 = reinterpret_cast<uint32_t*>(&_vld_78[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) << 16);
                    (&_vec_load_12[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_78[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_38[4];
        {
            const uint4* _vptr_79 = reinterpret_cast<const uint4*>(s_ptr + 3072);
            uint4* _vdst_79 = reinterpret_cast<uint4*>(&tab_s_38[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_79[_blk] = _vptr_79[_blk];
            }
        }
        float _vec_load_13[8];
        {
            const uint4* _vptr_80 = reinterpret_cast<const uint4*>(b_ptr + 3072 + 0);
            uint4 _vld_80[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_80[_blk] = _vptr_80[_blk];
                uint32_t* _vpairs_80 = reinterpret_cast<uint32_t*>(&_vld_80[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) << 16);
                    (&_vec_load_13[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_80[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_39[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_39[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_39[(0) + 3]))
            : "r"(x_base + 5120));
        float _unpack_bf16_21[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_21[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_39[_pair]) << 16);
            (&_unpack_bf16_21[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_39[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_15 = 0; pair_15 < 4; pair_15++) {
            int mul_offset_5 = pair_15 * 2;
            float2 _f32x2_mul_a_81 = make_float2(_unpack_bf16_21[mul_offset_5], _unpack_bf16_21[mul_offset_5 + 1]);
            float2 x_rstd_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_82 = make_float2(_vec_load_10[mul_offset_5], _vec_load_10[mul_offset_5 + 1]);
            float2 x_rstd_w_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_5) : "l"(*(const unsigned long long*)&x_rstd_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
            scaled[mul_offset_5] = x_rstd_w_5.x;
            scaled[mul_offset_5 + 1] = x_rstd_w_5.y;
        }
        #pragma unroll
        for (int j_25 = 0; j_25 < 8; j_25++) {
            high_only[2 * j_25] = 0.0f;
            high_only[2 * j_25 + 1] = scaled[j_25];
        }
        unsigned int high_only_bf16_40[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_40[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_26 = 0; j_26 < 8; j_26++) {
            normalized[j_26] = __uint_as_float(high_only_bf16_40[j_26]);
        }
        #pragma unroll
        for (int q_5 = 0; q_5 < 4; q_5++) {
            uint32_t _bf16x2_add_5;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_5) : "r"(tab_s_29[q_5]), "r"(one_pair));
            scale_plus_one[q_5] = _bf16x2_add_5;
        }
        float _unpack_bf16_22[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_22[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_22[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_16 = 0; pair_16 < 4; pair_16++) {
            int ada_offset_5 = pair_16 * 2;
            float2 adaln_pair_5 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_5], normalized[ada_offset_5 + 1]), make_float2(_unpack_bf16_22[ada_offset_5], _unpack_bf16_22[ada_offset_5 + 1]), make_float2(_vec_load_11[ada_offset_5], _vec_load_11[ada_offset_5 + 1]));
            adaln[ada_offset_5] = adaln_pair_5.x;
            adaln[ada_offset_5 + 1] = adaln_pair_5.y;
        }
        #pragma unroll
        for (int j_27 = 0; j_27 < 8; j_27++) {
            high_only[2 * j_27] = 0.0f;
            high_only[2 * j_27 + 1] = adaln[j_27];
        }
        unsigned int high_only_bf16_41[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_41[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_28 = 0; j_28 < 8; j_28++) {
            values[j_28] = __uint_as_float(high_only_bf16_41[j_28]);
        }
        if (write_debug == 2) {
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
        float amax_42 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_42 = max_noftz(amax_42, absolute[_lr]);
        }
        float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, amax_42, 1);
        amax_42 = max_noftz(amax_42, _shfl_xor_sync_83);
        float sf_value_43 = global_scale * (amax_42 * rcp_six);
        uint16_t _e4m3x2_85;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_85) : "f"(0.0f), "f"(sf_value_43));
        uint8_t _cvt_satfinite_e4m3_f32_84 = (uint8_t)_e4m3x2_85;
        uint32_t _f16x2_decode_86;
        float sf_rounded_44;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_86) : "h"(_e4m3x2_85));
        uint16_t _f16_decode_86 = (uint16_t)_f16x2_decode_86;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_44) : "h"(_f16_decode_86));
        float _rcp_87 = approx_rcp(sf_rounded_44 * rcp_global_scale);
        float output_scale_45 = fminf(_rcp_87, 3.4028235e+38f);
        float2 output_pair_46 = make_float2(output_scale_45, output_scale_45);
        #pragma unroll
        for (int pair_17 = 0; pair_17 < 4; pair_17++) {
            int quant_offset_5 = pair_17 * 2;
            float2 _f32x2_mul_a_88 = make_float2(values[quant_offset_5], values[quant_offset_5 + 1]);
            float2 quant_pair_5;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_88), "l"(*(const unsigned long long*)&output_pair_46));
            quant_values[quant_offset_5] = quant_pair_5.x;
            quant_values[quant_offset_5 + 1] = quant_pair_5.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_89 = q_ptr + 1280;
            __builtin_assume(__isGlobal(_gst_89));
            *(reinterpret_cast<int*>(_gst_89) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_90 = sf_ptr + 20480;
                __builtin_assume(__isGlobal(_gst_90));
                uint16_t _cvt_satfinite_e4m3x2_f32_91;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_91) : "f"(0.0f), "f"(sf_value_43));
                *reinterpret_cast<unsigned char*>(_gst_90) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_91 & 127));
            }
        }
        float _vec_load_14[8];
        {
            const uint4* _vptr_92 = reinterpret_cast<const uint4*>(w_ptr + 3584 + 0);
            uint4 _vld_92[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_92[_blk] = _vptr_92[_blk];
                uint32_t* _vpairs_92 = reinterpret_cast<uint32_t*>(&_vld_92[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_92[_pair]) << 16);
                    (&_vec_load_14[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_92[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_47[4];
        {
            const uint4* _vptr_93 = reinterpret_cast<const uint4*>(s_ptr + 3584);
            uint4* _vdst_93 = reinterpret_cast<uint4*>(&tab_s_47[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_93[_blk] = _vptr_93[_blk];
            }
        }
        float _vec_load_15[8];
        {
            const uint4* _vptr_94 = reinterpret_cast<const uint4*>(b_ptr + 3584 + 0);
            uint4 _vld_94[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_94[_blk] = _vptr_94[_blk];
                uint32_t* _vpairs_94 = reinterpret_cast<uint32_t*>(&_vld_94[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_94[_pair]) << 16);
                    (&_vec_load_15[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_94[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_48[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_48[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_48[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_48[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_48[(0) + 3]))
            : "r"(x_base + 6144));
        float _unpack_bf16_23[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_23[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_48[_pair]) << 16);
            (&_unpack_bf16_23[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_48[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_18 = 0; pair_18 < 4; pair_18++) {
            int mul_offset_6 = pair_18 * 2;
            float2 _f32x2_mul_a_95 = make_float2(_unpack_bf16_23[mul_offset_6], _unpack_bf16_23[mul_offset_6 + 1]);
            float2 x_rstd_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_96 = make_float2(_vec_load_12[mul_offset_6], _vec_load_12[mul_offset_6 + 1]);
            float2 x_rstd_w_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_6) : "l"(*(const unsigned long long*)&x_rstd_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_96));
            scaled[mul_offset_6] = x_rstd_w_6.x;
            scaled[mul_offset_6 + 1] = x_rstd_w_6.y;
        }
        #pragma unroll
        for (int j_30 = 0; j_30 < 8; j_30++) {
            high_only[2 * j_30] = 0.0f;
            high_only[2 * j_30 + 1] = scaled[j_30];
        }
        unsigned int high_only_bf16_49[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_49[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_31 = 0; j_31 < 8; j_31++) {
            normalized[j_31] = __uint_as_float(high_only_bf16_49[j_31]);
        }
        #pragma unroll
        for (int q_6 = 0; q_6 < 4; q_6++) {
            uint32_t _bf16x2_add_6;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_6) : "r"(tab_s_38[q_6]), "r"(one_pair));
            scale_plus_one[q_6] = _bf16x2_add_6;
        }
        float _unpack_bf16_24[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_24[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_24[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_19 = 0; pair_19 < 4; pair_19++) {
            int ada_offset_6 = pair_19 * 2;
            float2 adaln_pair_6 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_6], normalized[ada_offset_6 + 1]), make_float2(_unpack_bf16_24[ada_offset_6], _unpack_bf16_24[ada_offset_6 + 1]), make_float2(_vec_load_13[ada_offset_6], _vec_load_13[ada_offset_6 + 1]));
            adaln[ada_offset_6] = adaln_pair_6.x;
            adaln[ada_offset_6 + 1] = adaln_pair_6.y;
        }
        #pragma unroll
        for (int j_32 = 0; j_32 < 8; j_32++) {
            high_only[2 * j_32] = 0.0f;
            high_only[2 * j_32 + 1] = adaln[j_32];
        }
        unsigned int high_only_bf16_50[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_50[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_33 = 0; j_33 < 8; j_33++) {
            values[j_33] = __uint_as_float(high_only_bf16_50[j_33]);
        }
        if (write_debug == 2) {
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
        float amax_51 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_51 = max_noftz(amax_51, absolute[_lr]);
        }
        float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, amax_51, 1);
        amax_51 = max_noftz(amax_51, _shfl_xor_sync_97);
        float sf_value_52 = global_scale * (amax_51 * rcp_six);
        uint16_t _e4m3x2_99;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_99) : "f"(0.0f), "f"(sf_value_52));
        uint8_t _cvt_satfinite_e4m3_f32_98 = (uint8_t)_e4m3x2_99;
        uint32_t _f16x2_decode_100;
        float sf_rounded_53;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_100) : "h"(_e4m3x2_99));
        uint16_t _f16_decode_100 = (uint16_t)_f16x2_decode_100;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_53) : "h"(_f16_decode_100));
        float _rcp_101 = approx_rcp(sf_rounded_53 * rcp_global_scale);
        float output_scale_54 = fminf(_rcp_101, 3.4028235e+38f);
        float2 output_pair_55 = make_float2(output_scale_54, output_scale_54);
        #pragma unroll
        for (int pair_20 = 0; pair_20 < 4; pair_20++) {
            int quant_offset_6 = pair_20 * 2;
            float2 _f32x2_mul_a_102 = make_float2(values[quant_offset_6], values[quant_offset_6 + 1]);
            float2 quant_pair_6;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&output_pair_55));
            quant_values[quant_offset_6] = quant_pair_6.x;
            quant_values[quant_offset_6 + 1] = quant_pair_6.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_103 = q_ptr + 1536;
            __builtin_assume(__isGlobal(_gst_103));
            *(reinterpret_cast<int*>(_gst_103) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_104 = sf_ptr + 24576;
                __builtin_assume(__isGlobal(_gst_104));
                uint16_t _cvt_satfinite_e4m3x2_f32_105;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_105) : "f"(0.0f), "f"(sf_value_52));
                *reinterpret_cast<unsigned char*>(_gst_104) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_105 & 127));
            }
        }
        float _vec_load_16[8];
        {
            const uint4* _vptr_106 = reinterpret_cast<const uint4*>(w_ptr + 4096 + 0);
            uint4 _vld_106[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_106[_blk] = _vptr_106[_blk];
                uint32_t* _vpairs_106 = reinterpret_cast<uint32_t*>(&_vld_106[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) << 16);
                    (&_vec_load_16[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_106[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_56[4];
        {
            const uint4* _vptr_107 = reinterpret_cast<const uint4*>(s_ptr + 4096);
            uint4* _vdst_107 = reinterpret_cast<uint4*>(&tab_s_56[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_107[_blk] = _vptr_107[_blk];
            }
        }
        float _vec_load_17[8];
        {
            const uint4* _vptr_108 = reinterpret_cast<const uint4*>(b_ptr + 4096 + 0);
            uint4 _vld_108[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_108[_blk] = _vptr_108[_blk];
                uint32_t* _vpairs_108 = reinterpret_cast<uint32_t*>(&_vld_108[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) << 16);
                    (&_vec_load_17[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_108[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_57[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_57[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_57[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_57[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_57[(0) + 3]))
            : "r"(x_base + 7168));
        float _unpack_bf16_25[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_25[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_57[_pair]) << 16);
            (&_unpack_bf16_25[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_57[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_21 = 0; pair_21 < 4; pair_21++) {
            int mul_offset_7 = pair_21 * 2;
            float2 _f32x2_mul_a_109 = make_float2(_unpack_bf16_25[mul_offset_7], _unpack_bf16_25[mul_offset_7 + 1]);
            float2 x_rstd_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_110 = make_float2(_vec_load_14[mul_offset_7], _vec_load_14[mul_offset_7 + 1]);
            float2 x_rstd_w_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_7) : "l"(*(const unsigned long long*)&x_rstd_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
            scaled[mul_offset_7] = x_rstd_w_7.x;
            scaled[mul_offset_7 + 1] = x_rstd_w_7.y;
        }
        #pragma unroll
        for (int j_35 = 0; j_35 < 8; j_35++) {
            high_only[2 * j_35] = 0.0f;
            high_only[2 * j_35 + 1] = scaled[j_35];
        }
        unsigned int high_only_bf16_58[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_58[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_36 = 0; j_36 < 8; j_36++) {
            normalized[j_36] = __uint_as_float(high_only_bf16_58[j_36]);
        }
        #pragma unroll
        for (int q_7 = 0; q_7 < 4; q_7++) {
            uint32_t _bf16x2_add_7;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_7) : "r"(tab_s_47[q_7]), "r"(one_pair));
            scale_plus_one[q_7] = _bf16x2_add_7;
        }
        float _unpack_bf16_26[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_26[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_26[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_22 = 0; pair_22 < 4; pair_22++) {
            int ada_offset_7 = pair_22 * 2;
            float2 adaln_pair_7 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_7], normalized[ada_offset_7 + 1]), make_float2(_unpack_bf16_26[ada_offset_7], _unpack_bf16_26[ada_offset_7 + 1]), make_float2(_vec_load_15[ada_offset_7], _vec_load_15[ada_offset_7 + 1]));
            adaln[ada_offset_7] = adaln_pair_7.x;
            adaln[ada_offset_7 + 1] = adaln_pair_7.y;
        }
        #pragma unroll
        for (int j_37 = 0; j_37 < 8; j_37++) {
            high_only[2 * j_37] = 0.0f;
            high_only[2 * j_37 + 1] = adaln[j_37];
        }
        unsigned int high_only_bf16_59[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_59[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_38 = 0; j_38 < 8; j_38++) {
            values[j_38] = __uint_as_float(high_only_bf16_59[j_38]);
        }
        if (write_debug == 2) {
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
        for (int j_39 = 0; j_39 < 8; j_39++) {
            absolute[j_39] = values[j_39];
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
        float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, amax_60, 1);
        amax_60 = max_noftz(amax_60, _shfl_xor_sync_111);
        float sf_value_61 = global_scale * (amax_60 * rcp_six);
        uint16_t _e4m3x2_113;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_113) : "f"(0.0f), "f"(sf_value_61));
        uint8_t _cvt_satfinite_e4m3_f32_112 = (uint8_t)_e4m3x2_113;
        uint32_t _f16x2_decode_114;
        float sf_rounded_62;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_114) : "h"(_e4m3x2_113));
        uint16_t _f16_decode_114 = (uint16_t)_f16x2_decode_114;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_62) : "h"(_f16_decode_114));
        float _rcp_115 = approx_rcp(sf_rounded_62 * rcp_global_scale);
        float output_scale_63 = fminf(_rcp_115, 3.4028235e+38f);
        float2 output_pair_64 = make_float2(output_scale_63, output_scale_63);
        #pragma unroll
        for (int pair_23 = 0; pair_23 < 4; pair_23++) {
            int quant_offset_7 = pair_23 * 2;
            float2 _f32x2_mul_a_116 = make_float2(values[quant_offset_7], values[quant_offset_7 + 1]);
            float2 quant_pair_7;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_116), "l"(*(const unsigned long long*)&output_pair_64));
            quant_values[quant_offset_7] = quant_pair_7.x;
            quant_values[quant_offset_7 + 1] = quant_pair_7.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_117 = q_ptr + 1792;
            __builtin_assume(__isGlobal(_gst_117));
            *(reinterpret_cast<int*>(_gst_117) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_118 = sf_ptr + 28672;
                __builtin_assume(__isGlobal(_gst_118));
                uint16_t _cvt_satfinite_e4m3x2_f32_119;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_119) : "f"(0.0f), "f"(sf_value_61));
                *reinterpret_cast<unsigned char*>(_gst_118) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_119 & 127));
            }
        }
        float _vec_load_18[8];
        {
            const uint4* _vptr_120 = reinterpret_cast<const uint4*>(w_ptr + 4608 + 0);
            uint4 _vld_120[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_120[_blk] = _vptr_120[_blk];
                uint32_t* _vpairs_120 = reinterpret_cast<uint32_t*>(&_vld_120[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_120[_pair]) << 16);
                    (&_vec_load_18[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_120[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int tab_s_65[4];
        {
            const uint4* _vptr_121 = reinterpret_cast<const uint4*>(s_ptr + 4608);
            uint4* _vdst_121 = reinterpret_cast<uint4*>(&tab_s_65[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_121[_blk] = _vptr_121[_blk];
            }
        }
        float _vec_load_19[8];
        {
            const uint4* _vptr_122 = reinterpret_cast<const uint4*>(b_ptr + 4608 + 0);
            uint4 _vld_122[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_122[_blk] = _vptr_122[_blk];
                uint32_t* _vpairs_122 = reinterpret_cast<uint32_t*>(&_vld_122[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_122[_pair]) << 16);
                    (&_vec_load_19[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_122[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_66[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_66[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_66[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_66[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_66[(0) + 3]))
            : "r"(x_base + 8192));
        float _unpack_bf16_27[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_27[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_66[_pair]) << 16);
            (&_unpack_bf16_27[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_66[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_24 = 0; pair_24 < 4; pair_24++) {
            int mul_offset_8 = pair_24 * 2;
            float2 _f32x2_mul_a_123 = make_float2(_unpack_bf16_27[mul_offset_8], _unpack_bf16_27[mul_offset_8 + 1]);
            float2 x_rstd_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_124 = make_float2(_vec_load_16[mul_offset_8], _vec_load_16[mul_offset_8 + 1]);
            float2 x_rstd_w_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_8) : "l"(*(const unsigned long long*)&x_rstd_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_124));
            scaled[mul_offset_8] = x_rstd_w_8.x;
            scaled[mul_offset_8 + 1] = x_rstd_w_8.y;
        }
        #pragma unroll
        for (int j_40 = 0; j_40 < 8; j_40++) {
            high_only[2 * j_40] = 0.0f;
            high_only[2 * j_40 + 1] = scaled[j_40];
        }
        unsigned int high_only_bf16_67[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_67[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_41 = 0; j_41 < 8; j_41++) {
            normalized[j_41] = __uint_as_float(high_only_bf16_67[j_41]);
        }
        #pragma unroll
        for (int q_8 = 0; q_8 < 4; q_8++) {
            uint32_t _bf16x2_add_8;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_8) : "r"(tab_s_56[q_8]), "r"(one_pair));
            scale_plus_one[q_8] = _bf16x2_add_8;
        }
        float _unpack_bf16_28[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_28[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_28[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_25 = 0; pair_25 < 4; pair_25++) {
            int ada_offset_8 = pair_25 * 2;
            float2 adaln_pair_8 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_8], normalized[ada_offset_8 + 1]), make_float2(_unpack_bf16_28[ada_offset_8], _unpack_bf16_28[ada_offset_8 + 1]), make_float2(_vec_load_17[ada_offset_8], _vec_load_17[ada_offset_8 + 1]));
            adaln[ada_offset_8] = adaln_pair_8.x;
            adaln[ada_offset_8 + 1] = adaln_pair_8.y;
        }
        #pragma unroll
        for (int j_42 = 0; j_42 < 8; j_42++) {
            high_only[2 * j_42] = 0.0f;
            high_only[2 * j_42 + 1] = adaln[j_42];
        }
        unsigned int high_only_bf16_68[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_68[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_43 = 0; j_43 < 8; j_43++) {
            values[j_43] = __uint_as_float(high_only_bf16_68[j_43]);
        }
        if (write_debug == 2) {
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
        float amax_69 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_69 = max_noftz(amax_69, absolute[_lr]);
        }
        float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, amax_69, 1);
        amax_69 = max_noftz(amax_69, _shfl_xor_sync_125);
        float sf_value_70 = global_scale * (amax_69 * rcp_six);
        uint16_t _e4m3x2_127;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_127) : "f"(0.0f), "f"(sf_value_70));
        uint8_t _cvt_satfinite_e4m3_f32_126 = (uint8_t)_e4m3x2_127;
        uint32_t _f16x2_decode_128;
        float sf_rounded_71;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_128) : "h"(_e4m3x2_127));
        uint16_t _f16_decode_128 = (uint16_t)_f16x2_decode_128;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_71) : "h"(_f16_decode_128));
        float _rcp_129 = approx_rcp(sf_rounded_71 * rcp_global_scale);
        float output_scale_72 = fminf(_rcp_129, 3.4028235e+38f);
        float2 output_pair_73 = make_float2(output_scale_72, output_scale_72);
        #pragma unroll
        for (int pair_26 = 0; pair_26 < 4; pair_26++) {
            int quant_offset_8 = pair_26 * 2;
            float2 _f32x2_mul_a_130 = make_float2(values[quant_offset_8], values[quant_offset_8 + 1]);
            float2 quant_pair_8;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&output_pair_73));
            quant_values[quant_offset_8] = quant_pair_8.x;
            quant_values[quant_offset_8 + 1] = quant_pair_8.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_131 = q_ptr + 2048;
            __builtin_assume(__isGlobal(_gst_131));
            *(reinterpret_cast<int*>(_gst_131) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_132 = sf_ptr + 32768;
                __builtin_assume(__isGlobal(_gst_132));
                uint16_t _cvt_satfinite_e4m3x2_f32_133;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_133) : "f"(0.0f), "f"(sf_value_70));
                *reinterpret_cast<unsigned char*>(_gst_132) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_133 & 127));
            }
        }
        int last_col = 5120 + t % 32 * 8;
        float _vec_load_20[8];
        {
            const uint4* _vptr_134 = reinterpret_cast<const uint4*>(x_norm_weight + last_col + 0);
            uint4 _vld_134[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_134[_blk] = _vptr_134[_blk];
                uint32_t* _vpairs_134 = reinterpret_cast<uint32_t*>(&_vld_134[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) << 16);
                    (&_vec_load_20[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_134[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int last_s[4];
        {
            const uint4* _vptr_135 = reinterpret_cast<const uint4*>(adaln_scale + table_base + (unsigned long long)last_col);
            uint4* _vdst_135 = reinterpret_cast<uint4*>(&last_s[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_135[_blk] = _vptr_135[_blk];
            }
        }
        float _vec_load_21[8];
        {
            const uint4* _vptr_136 = reinterpret_cast<const uint4*>(adaln_shift + (table_base + (unsigned long long)last_col) + 0);
            uint4 _vld_136[1];
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vld_136[_blk] = _vptr_136[_blk];
                uint32_t* _vpairs_136 = reinterpret_cast<uint32_t*>(&_vld_136[_blk]);
                #pragma unroll
                for (int _pair = 0; _pair < 4; _pair++) {
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_136[_pair]) << 16);
                    (&_vec_load_21[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_136[_pair]) & 0xffff0000u);
                }
            }
        }
        unsigned int words_74[4];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&words_74[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_74[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_74[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_74[(0) + 3]))
            : "r"(x_base + 9216));
        float _unpack_bf16_29[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_29[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_74[_pair]) << 16);
            (&_unpack_bf16_29[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_74[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_27 = 0; pair_27 < 4; pair_27++) {
            int mul_offset_9 = pair_27 * 2;
            float2 _f32x2_mul_a_137 = make_float2(_unpack_bf16_29[mul_offset_9], _unpack_bf16_29[mul_offset_9 + 1]);
            float2 x_rstd_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_137), "l"(*(const unsigned long long*)&rstd_pair));
            float2 _f32x2_mul_b_138 = make_float2(_vec_load_18[mul_offset_9], _vec_load_18[mul_offset_9 + 1]);
            float2 x_rstd_w_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_9) : "l"(*(const unsigned long long*)&x_rstd_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_138));
            scaled[mul_offset_9] = x_rstd_w_9.x;
            scaled[mul_offset_9 + 1] = x_rstd_w_9.y;
        }
        #pragma unroll
        for (int j_45 = 0; j_45 < 8; j_45++) {
            high_only[2 * j_45] = 0.0f;
            high_only[2 * j_45 + 1] = scaled[j_45];
        }
        unsigned int high_only_bf16_75[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_75[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_46 = 0; j_46 < 8; j_46++) {
            normalized[j_46] = __uint_as_float(high_only_bf16_75[j_46]);
        }
        #pragma unroll
        for (int q_9 = 0; q_9 < 4; q_9++) {
            uint32_t _bf16x2_add_9;
            asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_9) : "r"(tab_s_65[q_9]), "r"(one_pair));
            scale_plus_one[q_9] = _bf16x2_add_9;
        }
        float _unpack_bf16_30[8];
        #pragma unroll
        for (int _pair = 0; _pair < 4; _pair++) {
            (&_unpack_bf16_30[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) << 16);
            (&_unpack_bf16_30[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(scale_plus_one[_pair]) & 0xffff0000u);
        }
        #pragma unroll
        for (int pair_28 = 0; pair_28 < 4; pair_28++) {
            int ada_offset_9 = pair_28 * 2;
            float2 adaln_pair_9 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_9], normalized[ada_offset_9 + 1]), make_float2(_unpack_bf16_30[ada_offset_9], _unpack_bf16_30[ada_offset_9 + 1]), make_float2(_vec_load_19[ada_offset_9], _vec_load_19[ada_offset_9 + 1]));
            adaln[ada_offset_9] = adaln_pair_9.x;
            adaln[ada_offset_9 + 1] = adaln_pair_9.y;
        }
        #pragma unroll
        for (int j_47 = 0; j_47 < 8; j_47++) {
            high_only[2 * j_47] = 0.0f;
            high_only[2 * j_47 + 1] = adaln[j_47];
        }
        unsigned int high_only_bf16_76[8];
        #pragma unroll
        for (int _lp = 0; _lp < 8; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
            high_only_bf16_76[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int j_48 = 0; j_48 < 8; j_48++) {
            values[j_48] = __uint_as_float(high_only_bf16_76[j_48]);
        }
        if (write_debug == 2) {
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
        for (int j_49 = 0; j_49 < 8; j_49++) {
            absolute[j_49] = values[j_49];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_77 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_77 = max_noftz(amax_77, absolute[_lr]);
        }
        float _shfl_xor_sync_139 = __shfl_xor_sync(0xFFFFFFFF, amax_77, 1);
        amax_77 = max_noftz(amax_77, _shfl_xor_sync_139);
        float sf_value_78 = global_scale * (amax_77 * rcp_six);
        uint16_t _e4m3x2_141;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_141) : "f"(0.0f), "f"(sf_value_78));
        uint8_t _cvt_satfinite_e4m3_f32_140 = (uint8_t)_e4m3x2_141;
        uint32_t _f16x2_decode_142;
        float sf_rounded_79;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_142) : "h"(_e4m3x2_141));
        uint16_t _f16_decode_142 = (uint16_t)_f16x2_decode_142;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_79) : "h"(_f16_decode_142));
        float _rcp_143 = approx_rcp(sf_rounded_79 * rcp_global_scale);
        float output_scale_80 = fminf(_rcp_143, 3.4028235e+38f);
        float2 output_pair_81 = make_float2(output_scale_80, output_scale_80);
        #pragma unroll
        for (int pair_29 = 0; pair_29 < 4; pair_29++) {
            int quant_offset_9 = pair_29 * 2;
            float2 _f32x2_mul_a_144 = make_float2(values[quant_offset_9], values[quant_offset_9 + 1]);
            float2 quant_pair_9;
            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_144), "l"(*(const unsigned long long*)&output_pair_81));
            quant_values[quant_offset_9] = quant_pair_9.x;
            quant_values[quant_offset_9 + 1] = quant_pair_9.y;
        }
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
        {
            auto _gst_145 = q_ptr + 2304;
            __builtin_assume(__isGlobal(_gst_145));
            *(reinterpret_cast<int*>(_gst_145) + (0)) = packed[0];
        }
        if (t % 2 == 0) {
            {
                auto _gst_146 = sf_ptr + 36864;
                __builtin_assume(__isGlobal(_gst_146));
                uint16_t _cvt_satfinite_e4m3x2_f32_147;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_147) : "f"(0.0f), "f"(sf_value_78));
                *reinterpret_cast<unsigned char*>(_gst_146) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_147 & 127));
            }
        }
        if (t < 32) {
            unsigned int words_0_1[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&words_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&words_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words_0_1[(0) + 3]))
                : "r"(x_base + 10240));
            float _unpack_bf16_31[8];
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&_unpack_bf16_31[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words_0_1[_pair]) << 16);
                (&_unpack_bf16_31[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words_0_1[_pair]) & 0xffff0000u);
            }
            #pragma unroll
            for (int pair_30 = 0; pair_30 < 4; pair_30++) {
                int mul_offset_10 = pair_30 * 2;
                float2 _f32x2_mul_a_148 = make_float2(_unpack_bf16_31[mul_offset_10], _unpack_bf16_31[mul_offset_10 + 1]);
                float2 x_rstd_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_148), "l"(*(const unsigned long long*)&rstd_pair));
                float2 _f32x2_mul_b_149 = make_float2(_vec_load_20[mul_offset_10], _vec_load_20[mul_offset_10 + 1]);
                float2 x_rstd_w_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&x_rstd_w_10) : "l"(*(const unsigned long long*)&x_rstd_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_149));
                scaled[mul_offset_10] = x_rstd_w_10.x;
                scaled[mul_offset_10 + 1] = x_rstd_w_10.y;
            }
            #pragma unroll
            for (int j_50 = 0; j_50 < 8; j_50++) {
                high_only[2 * j_50] = 0.0f;
                high_only[2 * j_50 + 1] = scaled[j_50];
            }
            unsigned int high_only_bf16_2[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
                high_only_bf16_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int j_51 = 0; j_51 < 8; j_51++) {
                normalized[j_51] = __uint_as_float(high_only_bf16_2[j_51]);
            }
            #pragma unroll
            for (int q_10 = 0; q_10 < 4; q_10++) {
                uint32_t _bf16x2_add_10;
                asm("add.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_add_10) : "r"(last_s[q_10]), "r"(one_pair));
                scale_plus_one[q_10] = _bf16x2_add_10;
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
                float2 adaln_pair_10 = fma_f32x2_rn_ftz(make_float2(normalized[ada_offset_10], normalized[ada_offset_10 + 1]), make_float2(_unpack_bf16_32[ada_offset_10], _unpack_bf16_32[ada_offset_10 + 1]), make_float2(_vec_load_21[ada_offset_10], _vec_load_21[ada_offset_10 + 1]));
                adaln[ada_offset_10] = adaln_pair_10.x;
                adaln[ada_offset_10 + 1] = adaln_pair_10.y;
            }
            #pragma unroll
            for (int j_52 = 0; j_52 < 8; j_52++) {
                high_only[2 * j_52] = 0.0f;
                high_only[2 * j_52 + 1] = adaln[j_52];
            }
            unsigned int high_only_bf16_3[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(high_only[_lp*2 + 0], high_only[_lp*2+1 + 0]));
                high_only_bf16_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int j_53 = 0; j_53 < 8; j_53++) {
                values[j_53] = __uint_as_float(high_only_bf16_3[j_53]);
            }
            if (write_debug == 2) {
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
            float amax_4 = absolute[0];
            #pragma unroll
            for (int _lr = 1; _lr < 8; _lr++) {
                amax_4 = max_noftz(amax_4, absolute[_lr]);
            }
            float _shfl_xor_sync_150 = __shfl_xor_sync(0xFFFFFFFF, amax_4, 1);
            amax_4 = max_noftz(amax_4, _shfl_xor_sync_150);
            float sf_value_5 = global_scale * (amax_4 * rcp_six);
            uint16_t _e4m3x2_152;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_152) : "f"(0.0f), "f"(sf_value_5));
            uint8_t _cvt_satfinite_e4m3_f32_151 = (uint8_t)_e4m3x2_152;
            uint32_t _f16x2_decode_153;
            float sf_rounded_6;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_153) : "h"(_e4m3x2_152));
            uint16_t _f16_decode_153 = (uint16_t)_f16x2_decode_153;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_6) : "h"(_f16_decode_153));
            float _rcp_154 = approx_rcp(sf_rounded_6 * rcp_global_scale);
            float output_scale_7 = fminf(_rcp_154, 3.4028235e+38f);
            float2 output_pair_8 = make_float2(output_scale_7, output_scale_7);
            #pragma unroll
            for (int pair_32 = 0; pair_32 < 4; pair_32++) {
                int quant_offset_10 = pair_32 * 2;
                float2 _f32x2_mul_a_155 = make_float2(values[quant_offset_10], values[quant_offset_10 + 1]);
                float2 quant_pair_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_155), "l"(*(const unsigned long long*)&output_pair_8));
                quant_values[quant_offset_10] = quant_pair_10.x;
                quant_values[quant_offset_10 + 1] = quant_pair_10.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_156 = q_ptr + 2560;
                __builtin_assume(__isGlobal(_gst_156));
                *(reinterpret_cast<int*>(_gst_156) + (0)) = packed[0];
            }
            if (t % 2 == 0) {
                {
                    auto _gst_157 = sf_ptr + 40960;
                    __builtin_assume(__isGlobal(_gst_157));
                    uint16_t _cvt_satfinite_e4m3x2_f32_158;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_158) : "f"(0.0f), "f"(sf_value_5));
                    *reinterpret_cast<unsigned char*>(_gst_157) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_158 & 127));
                }
            }
        }
    } else {
        #pragma unroll
        for (int j_55 = 0; j_55 < 8; j_55++) {
            values[j_55] = 0.0f;
            normalized[j_55] = 0.0f;
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
        for (int j_56 = 0; j_56 < 8; j_56++) {
            absolute[j_56] = values[j_56];
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
        float _shfl_xor_sync_159 = __shfl_xor_sync(0xFFFFFFFF, amax_1, 1);
        amax_1 = max_noftz(amax_1, _shfl_xor_sync_159);
        float sf_value_1 = global_scale * (amax_1 * rcp_six);
        uint16_t _e4m3x2_161;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_161) : "f"(0.0f), "f"(sf_value_1));
        uint8_t _cvt_satfinite_e4m3_f32_160 = (uint8_t)_e4m3x2_161;
        uint32_t _f16x2_decode_162;
        float sf_rounded_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_162) : "h"(_e4m3x2_161));
        uint16_t _f16_decode_162 = (uint16_t)_f16x2_decode_162;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_1) : "h"(_f16_decode_162));
        float _rcp_163 = approx_rcp(sf_rounded_1 * rcp_global_scale);
        float output_scale_1 = fminf(_rcp_163, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_1 = make_float2(output_scale_1, output_scale_1);
            #pragma unroll
            for (int pair_33 = 0; pair_33 < 4; pair_33++) {
                int quant_offset_11 = pair_33 * 2;
                float2 _f32x2_mul_a_164 = make_float2(values[quant_offset_11], values[quant_offset_11 + 1]);
                float2 quant_pair_11;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&output_pair_1));
                quant_values[quant_offset_11] = quant_pair_11.x;
                quant_values[quant_offset_11 + 1] = quant_pair_11.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_165 = q_ptr + 0;
                __builtin_assume(__isGlobal(_gst_165));
                *(reinterpret_cast<int*>(_gst_165) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_166 = sf_ptr + 0;
                __builtin_assume(__isGlobal(_gst_166));
                uint16_t _cvt_satfinite_e4m3x2_f32_167;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_167) : "f"(0.0f), "f"(sf_value_1));
                *reinterpret_cast<unsigned char*>(_gst_166) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_167 & 127));
            }
        }
        #pragma unroll
        for (int j_57 = 0; j_57 < 8; j_57++) {
            values[j_57] = 0.0f;
            normalized[j_57] = 0.0f;
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
        for (int j_58 = 0; j_58 < 8; j_58++) {
            absolute[j_58] = values[j_58];
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
        float _shfl_xor_sync_168 = __shfl_xor_sync(0xFFFFFFFF, amax_0, 1);
        amax_0 = max_noftz(amax_0, _shfl_xor_sync_168);
        float sf_value_1_1 = global_scale * (amax_0 * rcp_six);
        uint16_t _e4m3x2_170;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_170) : "f"(0.0f), "f"(sf_value_1_1));
        uint8_t _cvt_satfinite_e4m3_f32_169 = (uint8_t)_e4m3x2_170;
        uint32_t _f16x2_decode_171;
        float sf_rounded_2;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_171) : "h"(_e4m3x2_170));
        uint16_t _f16_decode_171 = (uint16_t)_f16x2_decode_171;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_2) : "h"(_f16_decode_171));
        float _rcp_172 = approx_rcp(sf_rounded_2 * rcp_global_scale);
        float output_scale_3 = fminf(_rcp_172, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_2 = make_float2(output_scale_3, output_scale_3);
            #pragma unroll
            for (int pair_34 = 0; pair_34 < 4; pair_34++) {
                int quant_offset_12 = pair_34 * 2;
                float2 _f32x2_mul_a_173 = make_float2(values[quant_offset_12], values[quant_offset_12 + 1]);
                float2 quant_pair_12;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_173), "l"(*(const unsigned long long*)&output_pair_2));
                quant_values[quant_offset_12] = quant_pair_12.x;
                quant_values[quant_offset_12 + 1] = quant_pair_12.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_174 = q_ptr + 256;
                __builtin_assume(__isGlobal(_gst_174));
                *(reinterpret_cast<int*>(_gst_174) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_175 = sf_ptr + 4096;
                __builtin_assume(__isGlobal(_gst_175));
                uint16_t _cvt_satfinite_e4m3x2_f32_176;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_176) : "f"(0.0f), "f"(sf_value_1_1));
                *reinterpret_cast<unsigned char*>(_gst_175) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_176 & 127));
            }
        }
        #pragma unroll
        for (int j_59 = 0; j_59 < 8; j_59++) {
            values[j_59] = 0.0f;
            normalized[j_59] = 0.0f;
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
        for (int j_60 = 0; j_60 < 8; j_60++) {
            absolute[j_60] = values[j_60];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_4_1 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_4_1 = max_noftz(amax_4_1, absolute[_lr]);
        }
        float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, amax_4_1, 1);
        amax_4_1 = max_noftz(amax_4_1, _shfl_xor_sync_177);
        float sf_value_5_1 = global_scale * (amax_4_1 * rcp_six);
        uint16_t _e4m3x2_179;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_179) : "f"(0.0f), "f"(sf_value_5_1));
        uint8_t _cvt_satfinite_e4m3_f32_178 = (uint8_t)_e4m3x2_179;
        uint32_t _f16x2_decode_180;
        float sf_rounded_6_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_180) : "h"(_e4m3x2_179));
        uint16_t _f16_decode_180 = (uint16_t)_f16x2_decode_180;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_6_1) : "h"(_f16_decode_180));
        float _rcp_181 = approx_rcp(sf_rounded_6_1 * rcp_global_scale);
        float output_scale_7_1 = fminf(_rcp_181, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_3 = make_float2(output_scale_7_1, output_scale_7_1);
            #pragma unroll
            for (int pair_35 = 0; pair_35 < 4; pair_35++) {
                int quant_offset_13 = pair_35 * 2;
                float2 _f32x2_mul_a_182 = make_float2(values[quant_offset_13], values[quant_offset_13 + 1]);
                float2 quant_pair_13;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_182), "l"(*(const unsigned long long*)&output_pair_3));
                quant_values[quant_offset_13] = quant_pair_13.x;
                quant_values[quant_offset_13 + 1] = quant_pair_13.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_183 = q_ptr + 512;
                __builtin_assume(__isGlobal(_gst_183));
                *(reinterpret_cast<int*>(_gst_183) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_184 = sf_ptr + 8192;
                __builtin_assume(__isGlobal(_gst_184));
                uint16_t _cvt_satfinite_e4m3x2_f32_185;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_185) : "f"(0.0f), "f"(sf_value_5_1));
                *reinterpret_cast<unsigned char*>(_gst_184) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_185 & 127));
            }
        }
        #pragma unroll
        for (int j_61 = 0; j_61 < 8; j_61++) {
            values[j_61] = 0.0f;
            normalized[j_61] = 0.0f;
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
        for (int j_62 = 0; j_62 < 8; j_62++) {
            absolute[j_62] = values[j_62];
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
        float _shfl_xor_sync_186 = __shfl_xor_sync(0xFFFFFFFF, amax_8, 1);
        amax_8 = max_noftz(amax_8, _shfl_xor_sync_186);
        float sf_value_9 = global_scale * (amax_8 * rcp_six);
        uint16_t _e4m3x2_188;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_188) : "f"(0.0f), "f"(sf_value_9));
        uint8_t _cvt_satfinite_e4m3_f32_187 = (uint8_t)_e4m3x2_188;
        uint32_t _f16x2_decode_189;
        float sf_rounded_10;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_189) : "h"(_e4m3x2_188));
        uint16_t _f16_decode_189 = (uint16_t)_f16x2_decode_189;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_10) : "h"(_f16_decode_189));
        float _rcp_190 = approx_rcp(sf_rounded_10 * rcp_global_scale);
        float output_scale_11 = fminf(_rcp_190, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_4 = make_float2(output_scale_11, output_scale_11);
            #pragma unroll
            for (int pair_36 = 0; pair_36 < 4; pair_36++) {
                int quant_offset_14 = pair_36 * 2;
                float2 _f32x2_mul_a_191 = make_float2(values[quant_offset_14], values[quant_offset_14 + 1]);
                float2 quant_pair_14;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_191), "l"(*(const unsigned long long*)&output_pair_4));
                quant_values[quant_offset_14] = quant_pair_14.x;
                quant_values[quant_offset_14 + 1] = quant_pair_14.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_192 = q_ptr + 768;
                __builtin_assume(__isGlobal(_gst_192));
                *(reinterpret_cast<int*>(_gst_192) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_193 = sf_ptr + 12288;
                __builtin_assume(__isGlobal(_gst_193));
                uint16_t _cvt_satfinite_e4m3x2_f32_194;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_194) : "f"(0.0f), "f"(sf_value_9));
                *reinterpret_cast<unsigned char*>(_gst_193) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_194 & 127));
            }
        }
        #pragma unroll
        for (int j_63 = 0; j_63 < 8; j_63++) {
            values[j_63] = 0.0f;
            normalized[j_63] = 0.0f;
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
        for (int j_64 = 0; j_64 < 8; j_64++) {
            absolute[j_64] = values[j_64];
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
        float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, amax_12, 1);
        amax_12 = max_noftz(amax_12, _shfl_xor_sync_195);
        float sf_value_13 = global_scale * (amax_12 * rcp_six);
        uint16_t _e4m3x2_197;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_197) : "f"(0.0f), "f"(sf_value_13));
        uint8_t _cvt_satfinite_e4m3_f32_196 = (uint8_t)_e4m3x2_197;
        uint32_t _f16x2_decode_198;
        float sf_rounded_14;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_198) : "h"(_e4m3x2_197));
        uint16_t _f16_decode_198 = (uint16_t)_f16x2_decode_198;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_14) : "h"(_f16_decode_198));
        float _rcp_199 = approx_rcp(sf_rounded_14 * rcp_global_scale);
        float output_scale_15 = fminf(_rcp_199, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_5 = make_float2(output_scale_15, output_scale_15);
            #pragma unroll
            for (int pair_37 = 0; pair_37 < 4; pair_37++) {
                int quant_offset_15 = pair_37 * 2;
                float2 _f32x2_mul_a_200 = make_float2(values[quant_offset_15], values[quant_offset_15 + 1]);
                float2 quant_pair_15;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_200), "l"(*(const unsigned long long*)&output_pair_5));
                quant_values[quant_offset_15] = quant_pair_15.x;
                quant_values[quant_offset_15 + 1] = quant_pair_15.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_201 = q_ptr + 1024;
                __builtin_assume(__isGlobal(_gst_201));
                *(reinterpret_cast<int*>(_gst_201) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_202 = sf_ptr + 16384;
                __builtin_assume(__isGlobal(_gst_202));
                uint16_t _cvt_satfinite_e4m3x2_f32_203;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_203) : "f"(0.0f), "f"(sf_value_13));
                *reinterpret_cast<unsigned char*>(_gst_202) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_203 & 127));
            }
        }
        #pragma unroll
        for (int j_65 = 0; j_65 < 8; j_65++) {
            values[j_65] = 0.0f;
            normalized[j_65] = 0.0f;
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
        for (int j_66 = 0; j_66 < 8; j_66++) {
            absolute[j_66] = values[j_66];
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
        float _shfl_xor_sync_204 = __shfl_xor_sync(0xFFFFFFFF, amax_16, 1);
        amax_16 = max_noftz(amax_16, _shfl_xor_sync_204);
        float sf_value_17 = global_scale * (amax_16 * rcp_six);
        uint16_t _e4m3x2_206;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_206) : "f"(0.0f), "f"(sf_value_17));
        uint8_t _cvt_satfinite_e4m3_f32_205 = (uint8_t)_e4m3x2_206;
        uint32_t _f16x2_decode_207;
        float sf_rounded_18;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_207) : "h"(_e4m3x2_206));
        uint16_t _f16_decode_207 = (uint16_t)_f16x2_decode_207;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_18) : "h"(_f16_decode_207));
        float _rcp_208 = approx_rcp(sf_rounded_18 * rcp_global_scale);
        float output_scale_19 = fminf(_rcp_208, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_6 = make_float2(output_scale_19, output_scale_19);
            #pragma unroll
            for (int pair_38 = 0; pair_38 < 4; pair_38++) {
                int quant_offset_16 = pair_38 * 2;
                float2 _f32x2_mul_a_209 = make_float2(values[quant_offset_16], values[quant_offset_16 + 1]);
                float2 quant_pair_16;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_209), "l"(*(const unsigned long long*)&output_pair_6));
                quant_values[quant_offset_16] = quant_pair_16.x;
                quant_values[quant_offset_16 + 1] = quant_pair_16.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_210 = q_ptr + 1280;
                __builtin_assume(__isGlobal(_gst_210));
                *(reinterpret_cast<int*>(_gst_210) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_211 = sf_ptr + 20480;
                __builtin_assume(__isGlobal(_gst_211));
                uint16_t _cvt_satfinite_e4m3x2_f32_212;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_212) : "f"(0.0f), "f"(sf_value_17));
                *reinterpret_cast<unsigned char*>(_gst_211) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_212 & 127));
            }
        }
        #pragma unroll
        for (int j_67 = 0; j_67 < 8; j_67++) {
            values[j_67] = 0.0f;
            normalized[j_67] = 0.0f;
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
        for (int j_68 = 0; j_68 < 8; j_68++) {
            absolute[j_68] = values[j_68];
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
        float _shfl_xor_sync_213 = __shfl_xor_sync(0xFFFFFFFF, amax_20, 1);
        amax_20 = max_noftz(amax_20, _shfl_xor_sync_213);
        float sf_value_21 = global_scale * (amax_20 * rcp_six);
        uint16_t _e4m3x2_215;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_215) : "f"(0.0f), "f"(sf_value_21));
        uint8_t _cvt_satfinite_e4m3_f32_214 = (uint8_t)_e4m3x2_215;
        uint32_t _f16x2_decode_216;
        float sf_rounded_22;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_216) : "h"(_e4m3x2_215));
        uint16_t _f16_decode_216 = (uint16_t)_f16x2_decode_216;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_22) : "h"(_f16_decode_216));
        float _rcp_217 = approx_rcp(sf_rounded_22 * rcp_global_scale);
        float output_scale_23 = fminf(_rcp_217, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_7 = make_float2(output_scale_23, output_scale_23);
            #pragma unroll
            for (int pair_39 = 0; pair_39 < 4; pair_39++) {
                int quant_offset_17 = pair_39 * 2;
                float2 _f32x2_mul_a_218 = make_float2(values[quant_offset_17], values[quant_offset_17 + 1]);
                float2 quant_pair_17;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_218), "l"(*(const unsigned long long*)&output_pair_7));
                quant_values[quant_offset_17] = quant_pair_17.x;
                quant_values[quant_offset_17 + 1] = quant_pair_17.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_219 = q_ptr + 1536;
                __builtin_assume(__isGlobal(_gst_219));
                *(reinterpret_cast<int*>(_gst_219) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_220 = sf_ptr + 24576;
                __builtin_assume(__isGlobal(_gst_220));
                uint16_t _cvt_satfinite_e4m3x2_f32_221;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_221) : "f"(0.0f), "f"(sf_value_21));
                *reinterpret_cast<unsigned char*>(_gst_220) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_221 & 127));
            }
        }
        #pragma unroll
        for (int j_69 = 0; j_69 < 8; j_69++) {
            values[j_69] = 0.0f;
            normalized[j_69] = 0.0f;
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
        for (int j_70 = 0; j_70 < 8; j_70++) {
            absolute[j_70] = values[j_70];
        }
        absolute[0] = fabsf(absolute[0]);
        absolute[1] = fabsf(absolute[1]);
        absolute[2] = fabsf(absolute[2]);
        absolute[3] = fabsf(absolute[3]);
        absolute[4] = fabsf(absolute[4]);
        absolute[5] = fabsf(absolute[5]);
        absolute[6] = fabsf(absolute[6]);
        absolute[7] = fabsf(absolute[7]);
        float amax_24_1 = absolute[0];
        #pragma unroll
        for (int _lr = 1; _lr < 8; _lr++) {
            amax_24_1 = max_noftz(amax_24_1, absolute[_lr]);
        }
        float _shfl_xor_sync_222 = __shfl_xor_sync(0xFFFFFFFF, amax_24_1, 1);
        amax_24_1 = max_noftz(amax_24_1, _shfl_xor_sync_222);
        float sf_value_25_1 = global_scale * (amax_24_1 * rcp_six);
        uint16_t _e4m3x2_224;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_224) : "f"(0.0f), "f"(sf_value_25_1));
        uint8_t _cvt_satfinite_e4m3_f32_223 = (uint8_t)_e4m3x2_224;
        uint32_t _f16x2_decode_225;
        float sf_rounded_26_1;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_225) : "h"(_e4m3x2_224));
        uint16_t _f16_decode_225 = (uint16_t)_f16x2_decode_225;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_26_1) : "h"(_f16_decode_225));
        float _rcp_226 = approx_rcp(sf_rounded_26_1 * rcp_global_scale);
        float output_scale_27_1 = fminf(_rcp_226, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_9 = make_float2(output_scale_27_1, output_scale_27_1);
            #pragma unroll
            for (int pair_40 = 0; pair_40 < 4; pair_40++) {
                int quant_offset_18 = pair_40 * 2;
                float2 _f32x2_mul_a_227 = make_float2(values[quant_offset_18], values[quant_offset_18 + 1]);
                float2 quant_pair_18;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_227), "l"(*(const unsigned long long*)&output_pair_9));
                quant_values[quant_offset_18] = quant_pair_18.x;
                quant_values[quant_offset_18 + 1] = quant_pair_18.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_228 = q_ptr + 1792;
                __builtin_assume(__isGlobal(_gst_228));
                *(reinterpret_cast<int*>(_gst_228) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_229 = sf_ptr + 28672;
                __builtin_assume(__isGlobal(_gst_229));
                uint16_t _cvt_satfinite_e4m3x2_f32_230;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_230) : "f"(0.0f), "f"(sf_value_25_1));
                *reinterpret_cast<unsigned char*>(_gst_229) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_230 & 127));
            }
        }
        #pragma unroll
        for (int j_71 = 0; j_71 < 8; j_71++) {
            values[j_71] = 0.0f;
            normalized[j_71] = 0.0f;
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
        for (int j_72 = 0; j_72 < 8; j_72++) {
            absolute[j_72] = values[j_72];
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
        float _shfl_xor_sync_231 = __shfl_xor_sync(0xFFFFFFFF, amax_28, 1);
        amax_28 = max_noftz(amax_28, _shfl_xor_sync_231);
        float sf_value_29 = global_scale * (amax_28 * rcp_six);
        uint16_t _e4m3x2_233;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_233) : "f"(0.0f), "f"(sf_value_29));
        uint8_t _cvt_satfinite_e4m3_f32_232 = (uint8_t)_e4m3x2_233;
        uint32_t _f16x2_decode_234;
        float sf_rounded_30;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_234) : "h"(_e4m3x2_233));
        uint16_t _f16_decode_234 = (uint16_t)_f16x2_decode_234;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_30) : "h"(_f16_decode_234));
        float _rcp_235 = approx_rcp(sf_rounded_30 * rcp_global_scale);
        float output_scale_31 = fminf(_rcp_235, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_11 = make_float2(output_scale_31, output_scale_31);
            #pragma unroll
            for (int pair_41 = 0; pair_41 < 4; pair_41++) {
                int quant_offset_19 = pair_41 * 2;
                float2 _f32x2_mul_a_236 = make_float2(values[quant_offset_19], values[quant_offset_19 + 1]);
                float2 quant_pair_19;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_236), "l"(*(const unsigned long long*)&output_pair_11));
                quant_values[quant_offset_19] = quant_pair_19.x;
                quant_values[quant_offset_19 + 1] = quant_pair_19.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_237 = q_ptr + 2048;
                __builtin_assume(__isGlobal(_gst_237));
                *(reinterpret_cast<int*>(_gst_237) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_238 = sf_ptr + 32768;
                __builtin_assume(__isGlobal(_gst_238));
                uint16_t _cvt_satfinite_e4m3x2_f32_239;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_239) : "f"(0.0f), "f"(sf_value_29));
                *reinterpret_cast<unsigned char*>(_gst_238) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_239 & 127));
            }
        }
        #pragma unroll
        for (int j_73 = 0; j_73 < 8; j_73++) {
            values[j_73] = 0.0f;
            normalized[j_73] = 0.0f;
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
        for (int j_74 = 0; j_74 < 8; j_74++) {
            absolute[j_74] = values[j_74];
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
        float _shfl_xor_sync_240 = __shfl_xor_sync(0xFFFFFFFF, amax_32, 1);
        amax_32 = max_noftz(amax_32, _shfl_xor_sync_240);
        float sf_value_33 = global_scale * (amax_32 * rcp_six);
        uint16_t _e4m3x2_242;
        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_242) : "f"(0.0f), "f"(sf_value_33));
        uint8_t _cvt_satfinite_e4m3_f32_241 = (uint8_t)_e4m3x2_242;
        uint32_t _f16x2_decode_243;
        float sf_rounded_34;
        asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_243) : "h"(_e4m3x2_242));
        uint16_t _f16_decode_243 = (uint16_t)_f16x2_decode_243;
        asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_34) : "h"(_f16_decode_243));
        float _rcp_244 = approx_rcp(sf_rounded_34 * rcp_global_scale);
        float output_scale_35 = fminf(_rcp_244, 3.4028235e+38f);
        if (row < M) {
            float2 output_pair_12 = make_float2(output_scale_35, output_scale_35);
            #pragma unroll
            for (int pair_42 = 0; pair_42 < 4; pair_42++) {
                int quant_offset_20 = pair_42 * 2;
                float2 _f32x2_mul_a_245 = make_float2(values[quant_offset_20], values[quant_offset_20 + 1]);
                float2 quant_pair_20;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_245), "l"(*(const unsigned long long*)&output_pair_12));
                quant_values[quant_offset_20] = quant_pair_20.x;
                quant_values[quant_offset_20 + 1] = quant_pair_20.y;
            }
            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
            {
                auto _gst_246 = q_ptr + 2304;
                __builtin_assume(__isGlobal(_gst_246));
                *(reinterpret_cast<int*>(_gst_246) + (0)) = packed[0];
            }
        }
        if (t % 2 == 0) {
            {
                auto _gst_247 = sf_ptr + 36864;
                __builtin_assume(__isGlobal(_gst_247));
                uint16_t _cvt_satfinite_e4m3x2_f32_248;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_248) : "f"(0.0f), "f"(sf_value_33));
                *reinterpret_cast<unsigned char*>(_gst_247) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_248 & 127));
            }
        }
        if (t < 32) {
            #pragma unroll
            for (int j_75 = 0; j_75 < 8; j_75++) {
                values[j_75] = 0.0f;
                normalized[j_75] = 0.0f;
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
            for (int j_76 = 0; j_76 < 8; j_76++) {
                absolute[j_76] = values[j_76];
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
            float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, amax_1_1, 1);
            amax_1_1 = max_noftz(amax_1_1, _shfl_xor_sync_249);
            float sf_value_2 = global_scale * (amax_1_1 * rcp_six);
            uint16_t _e4m3x2_251;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_251) : "f"(0.0f), "f"(sf_value_2));
            uint8_t _cvt_satfinite_e4m3_f32_250 = (uint8_t)_e4m3x2_251;
            uint32_t _f16x2_decode_252;
            float sf_rounded_3;
            asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_252) : "h"(_e4m3x2_251));
            uint16_t _f16_decode_252 = (uint16_t)_f16x2_decode_252;
            asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_3) : "h"(_f16_decode_252));
            float _rcp_253 = approx_rcp(sf_rounded_3 * rcp_global_scale);
            float output_scale_4 = fminf(_rcp_253, 3.4028235e+38f);
            if (row < M) {
                float2 output_pair_13 = make_float2(output_scale_4, output_scale_4);
                #pragma unroll
                for (int pair_43 = 0; pair_43 < 4; pair_43++) {
                    int quant_offset_21 = pair_43 * 2;
                    float2 _f32x2_mul_a_254 = make_float2(values[quant_offset_21], values[quant_offset_21 + 1]);
                    float2 quant_pair_21;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&quant_pair_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_254), "l"(*(const unsigned long long*)&output_pair_13));
                    quant_values[quant_offset_21] = quant_pair_21.x;
                    quant_values[quant_offset_21 + 1] = quant_pair_21.y;
                }
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                {
                    auto _gst_255 = q_ptr + 2560;
                    __builtin_assume(__isGlobal(_gst_255));
                    *(reinterpret_cast<int*>(_gst_255) + (0)) = packed[0];
                }
            }
            if (t % 2 == 0) {
                {
                    auto _gst_256 = sf_ptr + 40960;
                    __builtin_assume(__isGlobal(_gst_256));
                    uint16_t _cvt_satfinite_e4m3x2_f32_257;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_f32_257) : "f"(0.0f), "f"(sf_value_2));
                    *reinterpret_cast<unsigned char*>(_gst_256) = (unsigned char)((unsigned int)(unsigned char)(_cvt_satfinite_e4m3x2_f32_257 & 127));
                }
            }
        }
    }
}

} // extern "C"
