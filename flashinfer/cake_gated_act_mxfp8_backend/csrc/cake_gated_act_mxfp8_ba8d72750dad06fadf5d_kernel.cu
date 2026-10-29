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
#define SMEM_TILE_OFF 0
#define SMEM_TILE_STAGE_BYTES 32768
#define SMEM_TILE_STRIDE 32768
#define SMEM_TOTAL 32768
#define THREADS 256
#define LAUNCH_MIN_BLOCKS 3

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


__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
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

__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    const unsigned int w = *reinterpret_cast<const unsigned int*>(&v);
    return make_float2(__uint_as_float(w << 16), __uint_as_float(w & 0xffff0000u));
}

__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

__device__ __noinline__ unsigned int band_pair(__nv_bfloat16* src, int index, int k)
{
    unsigned int _vec_load_0[1];
    {
        _vec_load_0[0] = *reinterpret_cast<const unsigned int*>(src + index + 0);
    }
    unsigned int _vec_load_1[1];
    {
        _vec_load_1[0] = *reinterpret_cast<const unsigned int*>(src + (index + k) + 0);
    }
    float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
    float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
    float2 _f32x2_mul_b_0 = make_float2(-1.442695f, -1.442695f);
    float2 t;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_0));
    float e0 = approx_exp2(t.x);
    float e1 = approx_exp2(t.y);
    float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
    float d0 = -nd.x;
    float d1 = -nd.y;
    float r0 = approx_rcp(d0);
    float r1 = approx_rcp(d1);
    float2 r = make_float2(r0, r1);
    float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
    float t0 = s.x;
    float t1 = s.y;
    float d4_8a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a) : "f"(d0), "f"(0.25f));
    float r4_8a = approx_rcp(d4_8a);
    float _fma_2;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(-d4_8a), "f"(r4_8a), "f"(1.0f));
    float _fma_1;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(r4_8a), "f"(_fma_2), "f"(r4_8a));
    float big_8a;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a) : "f"(mul_f32_rn_noftz(_fma_1, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_3;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(big_8a), "f"(8.388608e+06f));
    float whole_8a;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a) : "f"(_add_3), "f"(8.388608e+06f));
    float above_8a;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a) : "f"(-d4_8a), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_5;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_5) : "f"(big_8a), "f"(whole_8a));
    float _abs_4;
    asm("abs.f32 %0, %1;" : "=f"(_abs_4) : "f"(_sub_5));
    float _add_6;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(big_8a), "f"(0.5f));
    float _sub_7;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_7) : "f"(big_8a), "f"(0.5f));
    float rounded_8a = ((_abs_4 == 0.5f) ? ((above_8a > 0.0f) ? _add_6 : _sub_7) : whole_8a);
    float u0 = ((d0 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a, 7.888609e-31f), 1.7763568e-15f) : t0);
    float d4_8a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(d4_8a_0) : "f"(d1), "f"(0.25f));
    float r4_8a_0 = approx_rcp(d4_8a_0);
    float _fma_9;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(-d4_8a_0), "f"(r4_8a_0), "f"(1.0f));
    float _fma_8;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(r4_8a_0), "f"(_fma_9), "f"(r4_8a_0));
    float big_8a_0;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(big_8a_0) : "f"(mul_f32_rn_noftz(_fma_8, 1.2676506e+30f)), "f"(1.4073749e+14f));
    float _add_10;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(big_8a_0), "f"(8.388608e+06f));
    float whole_8a_0;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(whole_8a_0) : "f"(_add_10), "f"(8.388608e+06f));
    float above_8a_0;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(above_8a_0) : "f"(-d4_8a_0), "f"(mul_f32_rn_noftz(mul_f32_rn_noftz(big_8a_0, 7.888609e-31f), 7.1054274e-15f)), "f"(1.0f));
    float _sub_12;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_12) : "f"(big_8a_0), "f"(whole_8a_0));
    float _abs_11;
    asm("abs.f32 %0, %1;" : "=f"(_abs_11) : "f"(_sub_12));
    float _add_13;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(big_8a_0), "f"(0.5f));
    float _sub_14;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(_sub_14) : "f"(big_8a_0), "f"(0.5f));
    float rounded_8a_0 = ((_abs_11 == 0.5f) ? ((above_8a_0 > 0.0f) ? _add_13 : _sub_14) : whole_8a_0);
    float u1 = ((d1 >= 8.507059e+37f) ? mul_f32_rn_noftz(mul_f32_rn_noftz(rounded_8a_0, 7.888609e-31f), 1.7763568e-15f) : t1);
    float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32, make_float2(max_noftz(u0, 0.0f), max_noftz(u1, 0.0f)));
    float2 mul;
    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1));
    unsigned int band_pair_out = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
    return band_pair_out;
}

extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gated_act_mxfp8_ba8d72750dad06fadf5d(__nv_bfloat16* __restrict__ gated_input, uint8_t* __restrict__ row_output, uint8_t* __restrict__ col_output, uint8_t* __restrict__ row_scales, uint8_t* __restrict__ col_scales, int M, int K)
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
    uint8_t* tile = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int tile_addr = smem + 0;

    // === Task calls (declaration order) ===
    int half = tid & 1;
    int blk = tid >> 1 & 3;
    int row = tid >> 3;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int grow = by * 128 + row;
    int col = bx * 128 + blk * 32 + half * 16;
    int gate_index = grow * (2 * K) + col;
    unsigned int _vec_load_0[8];
    {
        uint32_t _iv_0_0;
        uint32_t _iv_0_1;
        uint32_t _iv_0_2;
        uint32_t _iv_0_3;
        uint32_t _iv_0_4;
        uint32_t _iv_0_5;
        uint32_t _iv_0_6;
        uint32_t _iv_0_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_0_0), "=r"(_iv_0_1), "=r"(_iv_0_2), "=r"(_iv_0_3), "=r"(_iv_0_4), "=r"(_iv_0_5), "=r"(_iv_0_6), "=r"(_iv_0_7) : "l"((const void*)(gated_input + gate_index + (0))) : "memory");
        _vec_load_0[0 + 0] = (uint32_t)_iv_0_0;
        _vec_load_0[0 + 1] = (uint32_t)_iv_0_1;
        _vec_load_0[0 + 2] = (uint32_t)_iv_0_2;
        _vec_load_0[0 + 3] = (uint32_t)_iv_0_3;
        _vec_load_0[0 + 4] = (uint32_t)_iv_0_4;
        _vec_load_0[0 + 5] = (uint32_t)_iv_0_5;
        _vec_load_0[0 + 6] = (uint32_t)_iv_0_6;
        _vec_load_0[0 + 7] = (uint32_t)_iv_0_7;
    }
    unsigned int _vec_load_1[8];
    {
        uint32_t _iv_1_0;
        uint32_t _iv_1_1;
        uint32_t _iv_1_2;
        uint32_t _iv_1_3;
        uint32_t _iv_1_4;
        uint32_t _iv_1_5;
        uint32_t _iv_1_6;
        uint32_t _iv_1_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_1_0), "=r"(_iv_1_1), "=r"(_iv_1_2), "=r"(_iv_1_3), "=r"(_iv_1_4), "=r"(_iv_1_5), "=r"(_iv_1_6), "=r"(_iv_1_7) : "l"((const void*)(gated_input + (gate_index + K) + (0))) : "memory");
        _vec_load_1[0 + 0] = (uint32_t)_iv_1_0;
        _vec_load_1[0 + 1] = (uint32_t)_iv_1_1;
        _vec_load_1[0 + 2] = (uint32_t)_iv_1_2;
        _vec_load_1[0 + 3] = (uint32_t)_iv_1_3;
        _vec_load_1[0 + 4] = (uint32_t)_iv_1_4;
        _vec_load_1[0 + 5] = (uint32_t)_iv_1_5;
        _vec_load_1[0 + 6] = (uint32_t)_iv_1_6;
        _vec_load_1[0 + 7] = (uint32_t)_iv_1_7;
    }
    int grow_b = grow + 32;
    int grow_c = grow + 64;
    int grow_d = grow + 96;
    int gate_index_3 = grow_b * (2 * K) + col;
    unsigned int vec_load[8];
    {
        uint32_t _iv_2_0;
        uint32_t _iv_2_1;
        uint32_t _iv_2_2;
        uint32_t _iv_2_3;
        uint32_t _iv_2_4;
        uint32_t _iv_2_5;
        uint32_t _iv_2_6;
        uint32_t _iv_2_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_2_0), "=r"(_iv_2_1), "=r"(_iv_2_2), "=r"(_iv_2_3), "=r"(_iv_2_4), "=r"(_iv_2_5), "=r"(_iv_2_6), "=r"(_iv_2_7) : "l"((const void*)(gated_input + gate_index_3 + (0))) : "memory");
        vec_load[0 + 0] = (uint32_t)_iv_2_0;
        vec_load[0 + 1] = (uint32_t)_iv_2_1;
        vec_load[0 + 2] = (uint32_t)_iv_2_2;
        vec_load[0 + 3] = (uint32_t)_iv_2_3;
        vec_load[0 + 4] = (uint32_t)_iv_2_4;
        vec_load[0 + 5] = (uint32_t)_iv_2_5;
        vec_load[0 + 6] = (uint32_t)_iv_2_6;
        vec_load[0 + 7] = (uint32_t)_iv_2_7;
    }
    unsigned int vec_load_1[8];
    {
        uint32_t _iv_3_0;
        uint32_t _iv_3_1;
        uint32_t _iv_3_2;
        uint32_t _iv_3_3;
        uint32_t _iv_3_4;
        uint32_t _iv_3_5;
        uint32_t _iv_3_6;
        uint32_t _iv_3_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_3_0), "=r"(_iv_3_1), "=r"(_iv_3_2), "=r"(_iv_3_3), "=r"(_iv_3_4), "=r"(_iv_3_5), "=r"(_iv_3_6), "=r"(_iv_3_7) : "l"((const void*)(gated_input + (gate_index_3 + K) + (0))) : "memory");
        vec_load_1[0 + 0] = (uint32_t)_iv_3_0;
        vec_load_1[0 + 1] = (uint32_t)_iv_3_1;
        vec_load_1[0 + 2] = (uint32_t)_iv_3_2;
        vec_load_1[0 + 3] = (uint32_t)_iv_3_3;
        vec_load_1[0 + 4] = (uint32_t)_iv_3_4;
        vec_load_1[0 + 5] = (uint32_t)_iv_3_5;
        vec_load_1[0 + 6] = (uint32_t)_iv_3_6;
        vec_load_1[0 + 7] = (uint32_t)_iv_3_7;
    }
    int gate_index_4 = grow_c * (2 * K) + col;
    unsigned int vec_load_0[8];
    {
        uint32_t _iv_4_0;
        uint32_t _iv_4_1;
        uint32_t _iv_4_2;
        uint32_t _iv_4_3;
        uint32_t _iv_4_4;
        uint32_t _iv_4_5;
        uint32_t _iv_4_6;
        uint32_t _iv_4_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_4_0), "=r"(_iv_4_1), "=r"(_iv_4_2), "=r"(_iv_4_3), "=r"(_iv_4_4), "=r"(_iv_4_5), "=r"(_iv_4_6), "=r"(_iv_4_7) : "l"((const void*)(gated_input + gate_index_4 + (0))) : "memory");
        vec_load_0[0 + 0] = (uint32_t)_iv_4_0;
        vec_load_0[0 + 1] = (uint32_t)_iv_4_1;
        vec_load_0[0 + 2] = (uint32_t)_iv_4_2;
        vec_load_0[0 + 3] = (uint32_t)_iv_4_3;
        vec_load_0[0 + 4] = (uint32_t)_iv_4_4;
        vec_load_0[0 + 5] = (uint32_t)_iv_4_5;
        vec_load_0[0 + 6] = (uint32_t)_iv_4_6;
        vec_load_0[0 + 7] = (uint32_t)_iv_4_7;
    }
    unsigned int vec_load_1_0[8];
    {
        uint32_t _iv_5_0;
        uint32_t _iv_5_1;
        uint32_t _iv_5_2;
        uint32_t _iv_5_3;
        uint32_t _iv_5_4;
        uint32_t _iv_5_5;
        uint32_t _iv_5_6;
        uint32_t _iv_5_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_5_0), "=r"(_iv_5_1), "=r"(_iv_5_2), "=r"(_iv_5_3), "=r"(_iv_5_4), "=r"(_iv_5_5), "=r"(_iv_5_6), "=r"(_iv_5_7) : "l"((const void*)(gated_input + (gate_index_4 + K) + (0))) : "memory");
        vec_load_1_0[0 + 0] = (uint32_t)_iv_5_0;
        vec_load_1_0[0 + 1] = (uint32_t)_iv_5_1;
        vec_load_1_0[0 + 2] = (uint32_t)_iv_5_2;
        vec_load_1_0[0 + 3] = (uint32_t)_iv_5_3;
        vec_load_1_0[0 + 4] = (uint32_t)_iv_5_4;
        vec_load_1_0[0 + 5] = (uint32_t)_iv_5_5;
        vec_load_1_0[0 + 6] = (uint32_t)_iv_5_6;
        vec_load_1_0[0 + 7] = (uint32_t)_iv_5_7;
    }
    int gate_index_5 = grow_d * (2 * K) + col;
    unsigned int vec_load_2[8];
    {
        uint32_t _iv_6_0;
        uint32_t _iv_6_1;
        uint32_t _iv_6_2;
        uint32_t _iv_6_3;
        uint32_t _iv_6_4;
        uint32_t _iv_6_5;
        uint32_t _iv_6_6;
        uint32_t _iv_6_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_6_0), "=r"(_iv_6_1), "=r"(_iv_6_2), "=r"(_iv_6_3), "=r"(_iv_6_4), "=r"(_iv_6_5), "=r"(_iv_6_6), "=r"(_iv_6_7) : "l"((const void*)(gated_input + gate_index_5 + (0))) : "memory");
        vec_load_2[0 + 0] = (uint32_t)_iv_6_0;
        vec_load_2[0 + 1] = (uint32_t)_iv_6_1;
        vec_load_2[0 + 2] = (uint32_t)_iv_6_2;
        vec_load_2[0 + 3] = (uint32_t)_iv_6_3;
        vec_load_2[0 + 4] = (uint32_t)_iv_6_4;
        vec_load_2[0 + 5] = (uint32_t)_iv_6_5;
        vec_load_2[0 + 6] = (uint32_t)_iv_6_6;
        vec_load_2[0 + 7] = (uint32_t)_iv_6_7;
    }
    unsigned int vec_load_1_1[8];
    {
        uint32_t _iv_7_0;
        uint32_t _iv_7_1;
        uint32_t _iv_7_2;
        uint32_t _iv_7_3;
        uint32_t _iv_7_4;
        uint32_t _iv_7_5;
        uint32_t _iv_7_6;
        uint32_t _iv_7_7;
        asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_iv_7_0), "=r"(_iv_7_1), "=r"(_iv_7_2), "=r"(_iv_7_3), "=r"(_iv_7_4), "=r"(_iv_7_5), "=r"(_iv_7_6), "=r"(_iv_7_7) : "l"((const void*)(gated_input + (gate_index_5 + K) + (0))) : "memory");
        vec_load_1_1[0 + 0] = (uint32_t)_iv_7_0;
        vec_load_1_1[0 + 1] = (uint32_t)_iv_7_1;
        vec_load_1_1[0 + 2] = (uint32_t)_iv_7_2;
        vec_load_1_1[0 + 3] = (uint32_t)_iv_7_3;
        vec_load_1_1[0 + 4] = (uint32_t)_iv_7_4;
        vec_load_1_1[0 + 5] = (uint32_t)_iv_7_5;
        vec_load_1_1[0 + 6] = (uint32_t)_iv_7_6;
        vec_load_1_1[0 + 7] = (uint32_t)_iv_7_7;
    }
    unsigned int packed[8];
    unsigned int packed_b[8];
    unsigned int packed_c[8];
    unsigned int packed_d[8];
    __nv_bfloat162 lowest = __as_bf16x2(_vec_load_0[0]);
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(_vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_0[7]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[0]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[1]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[2]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[3]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[4]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[5]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[6]));
    lowest = __hmin2_nan(lowest, __as_bf16x2(vec_load_2[7]));
    float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(lowest);
    float _min_9;
    asm("min.NaN.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_cvt_f32_0.x), "f"(_cvt_f32_0.y));
    int _warp_vote_8 = __all_sync(0xFFFFFFFF, _min_9 > -87.0f);
    if (_warp_vote_8 != 0) {
        float2 cvt_f32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_10 = make_float2(-1.442695f, -1.442695f);
        float2 t;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t) : "l"(*(const unsigned long long*)&cvt_f32), "l"(*(const unsigned long long*)&_f32x2_mul_b_10));
        float e0 = approx_exp2(t.x);
        float e1 = approx_exp2(t.y);
        float2 nd = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0, e1));
        float d0 = -nd.x;
        float d1 = -nd.y;
        float r0 = approx_rcp(d0);
        float r1 = approx_rcp(d1);
        float2 r = make_float2(r0, r1);
        float2 s = fma_f32x2_rn_noftz(r, fma_f32x2_rn_noftz(nd, r, make_float2(1.0f, 1.0f)), r);
        float2 _f32x2_mul_a_11 = mul_f32x2_noftz(cvt_f32, s);
        float2 mul;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul) : "l"(*(const unsigned long long*)&_f32x2_mul_a_11), "l"(*(const unsigned long long*)&cvt_f32_1));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul.x, mul.y)));
        float2 cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_0 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
        float2 t_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_0) : "l"(*(const unsigned long long*)&cvt_f32_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
        float e0_0 = approx_exp2(t_0.x);
        float e1_0 = approx_exp2(t_0.y);
        float2 nd_0 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_0, e1_0));
        float d0_0 = -nd_0.x;
        float d1_0 = -nd_0.y;
        float r0_0 = approx_rcp(d0_0);
        float r1_0 = approx_rcp(d1_0);
        float2 r_0 = make_float2(r0_0, r1_0);
        float2 s_0 = fma_f32x2_rn_noftz(r_0, fma_f32x2_rn_noftz(nd_0, r_0, make_float2(1.0f, 1.0f)), r_0);
        float2 _f32x2_mul_a_13 = mul_f32x2_noftz(cvt_f32_0, s_0);
        float2 mul_0;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_13), "l"(*(const unsigned long long*)&cvt_f32_1_0));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_0.x, mul_0.y)));
        float2 cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_14 = make_float2(-1.442695f, -1.442695f);
        float2 t_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_1) : "l"(*(const unsigned long long*)&cvt_f32_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_14));
        float e0_1 = approx_exp2(t_1.x);
        float e1_1 = approx_exp2(t_1.y);
        float2 nd_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_1, e1_1));
        float d0_1 = -nd_1.x;
        float d1_1 = -nd_1.y;
        float r0_1 = approx_rcp(d0_1);
        float r1_1 = approx_rcp(d1_1);
        float2 r_1 = make_float2(r0_1, r1_1);
        float2 s_1 = fma_f32x2_rn_noftz(r_1, fma_f32x2_rn_noftz(nd_1, r_1, make_float2(1.0f, 1.0f)), r_1);
        float2 _f32x2_mul_a_15 = mul_f32x2_noftz(cvt_f32_2, s_1);
        float2 mul_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&cvt_f32_1_1));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_1.x, mul_1.y)));
        float2 cvt_f32_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_2 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_16 = make_float2(-1.442695f, -1.442695f);
        float2 t_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_2) : "l"(*(const unsigned long long*)&cvt_f32_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_16));
        float e0_2 = approx_exp2(t_2.x);
        float e1_2 = approx_exp2(t_2.y);
        float2 nd_2 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_2, e1_2));
        float d0_2 = -nd_2.x;
        float d1_2 = -nd_2.y;
        float r0_2 = approx_rcp(d0_2);
        float r1_2 = approx_rcp(d1_2);
        float2 r_2 = make_float2(r0_2, r1_2);
        float2 s_2 = fma_f32x2_rn_noftz(r_2, fma_f32x2_rn_noftz(nd_2, r_2, make_float2(1.0f, 1.0f)), r_2);
        float2 _f32x2_mul_a_17 = mul_f32x2_noftz(cvt_f32_3, s_2);
        float2 mul_2;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_17), "l"(*(const unsigned long long*)&cvt_f32_1_2));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_2.x, mul_2.y)));
        float2 cvt_f32_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_3 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_18 = make_float2(-1.442695f, -1.442695f);
        float2 t_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_3) : "l"(*(const unsigned long long*)&cvt_f32_4), "l"(*(const unsigned long long*)&_f32x2_mul_b_18));
        float e0_3 = approx_exp2(t_3.x);
        float e1_3 = approx_exp2(t_3.y);
        float2 nd_3 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_3, e1_3));
        float d0_3 = -nd_3.x;
        float d1_3 = -nd_3.y;
        float r0_3 = approx_rcp(d0_3);
        float r1_3 = approx_rcp(d1_3);
        float2 r_3 = make_float2(r0_3, r1_3);
        float2 s_3 = fma_f32x2_rn_noftz(r_3, fma_f32x2_rn_noftz(nd_3, r_3, make_float2(1.0f, 1.0f)), r_3);
        float2 _f32x2_mul_a_19 = mul_f32x2_noftz(cvt_f32_4, s_3);
        float2 mul_3;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&cvt_f32_1_3));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_3.x, mul_3.y)));
        float2 cvt_f32_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_4 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_20 = make_float2(-1.442695f, -1.442695f);
        float2 t_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_4) : "l"(*(const unsigned long long*)&cvt_f32_5), "l"(*(const unsigned long long*)&_f32x2_mul_b_20));
        float e0_4 = approx_exp2(t_4.x);
        float e1_4 = approx_exp2(t_4.y);
        float2 nd_4 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_4, e1_4));
        float d0_4 = -nd_4.x;
        float d1_4 = -nd_4.y;
        float r0_4 = approx_rcp(d0_4);
        float r1_4 = approx_rcp(d1_4);
        float2 r_4 = make_float2(r0_4, r1_4);
        float2 s_4 = fma_f32x2_rn_noftz(r_4, fma_f32x2_rn_noftz(nd_4, r_4, make_float2(1.0f, 1.0f)), r_4);
        float2 _f32x2_mul_a_21 = mul_f32x2_noftz(cvt_f32_5, s_4);
        float2 mul_4;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_21), "l"(*(const unsigned long long*)&cvt_f32_1_4));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_4.x, mul_4.y)));
        float2 cvt_f32_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_5 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
        float2 t_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_5) : "l"(*(const unsigned long long*)&cvt_f32_6), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
        float e0_5 = approx_exp2(t_5.x);
        float e1_5 = approx_exp2(t_5.y);
        float2 nd_5 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_5, e1_5));
        float d0_5 = -nd_5.x;
        float d1_5 = -nd_5.y;
        float r0_5 = approx_rcp(d0_5);
        float r1_5 = approx_rcp(d1_5);
        float2 r_5 = make_float2(r0_5, r1_5);
        float2 s_5 = fma_f32x2_rn_noftz(r_5, fma_f32x2_rn_noftz(nd_5, r_5, make_float2(1.0f, 1.0f)), r_5);
        float2 _f32x2_mul_a_23 = mul_f32x2_noftz(cvt_f32_6, s_5);
        float2 mul_5;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_23), "l"(*(const unsigned long long*)&cvt_f32_1_5));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_5.x, mul_5.y)));
        float2 cvt_f32_7 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_6 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_24 = make_float2(-1.442695f, -1.442695f);
        float2 t_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_6) : "l"(*(const unsigned long long*)&cvt_f32_7), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
        float e0_6 = approx_exp2(t_6.x);
        float e1_6 = approx_exp2(t_6.y);
        float2 nd_6 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_6, e1_6));
        float d0_6 = -nd_6.x;
        float d1_6 = -nd_6.y;
        float r0_6 = approx_rcp(d0_6);
        float r1_6 = approx_rcp(d1_6);
        float2 r_6 = make_float2(r0_6, r1_6);
        float2 s_6 = fma_f32x2_rn_noftz(r_6, fma_f32x2_rn_noftz(nd_6, r_6, make_float2(1.0f, 1.0f)), r_6);
        float2 _f32x2_mul_a_25 = mul_f32x2_noftz(cvt_f32_7, s_6);
        float2 mul_6;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&cvt_f32_1_6));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_6.x, mul_6.y)));
        float2 cvt_f32_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_7 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 _f32x2_mul_b_26 = make_float2(-1.442695f, -1.442695f);
        float2 t_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_7) : "l"(*(const unsigned long long*)&cvt_f32_8), "l"(*(const unsigned long long*)&_f32x2_mul_b_26));
        float e0_7 = approx_exp2(t_7.x);
        float e1_7 = approx_exp2(t_7.y);
        float2 nd_7 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_7, e1_7));
        float d0_7 = -nd_7.x;
        float d1_7 = -nd_7.y;
        float r0_7 = approx_rcp(d0_7);
        float r1_7 = approx_rcp(d1_7);
        float2 r_7 = make_float2(r0_7, r1_7);
        float2 s_7 = fma_f32x2_rn_noftz(r_7, fma_f32x2_rn_noftz(nd_7, r_7, make_float2(1.0f, 1.0f)), r_7);
        float2 _f32x2_mul_a_27 = mul_f32x2_noftz(cvt_f32_8, s_7);
        float2 mul_7;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&cvt_f32_1_7));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_7.x, mul_7.y)));
        float2 cvt_f32_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_8 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 _f32x2_mul_b_28 = make_float2(-1.442695f, -1.442695f);
        float2 t_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_8) : "l"(*(const unsigned long long*)&cvt_f32_9), "l"(*(const unsigned long long*)&_f32x2_mul_b_28));
        float e0_8 = approx_exp2(t_8.x);
        float e1_8 = approx_exp2(t_8.y);
        float2 nd_8 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_8, e1_8));
        float d0_8 = -nd_8.x;
        float d1_8 = -nd_8.y;
        float r0_8 = approx_rcp(d0_8);
        float r1_8 = approx_rcp(d1_8);
        float2 r_8 = make_float2(r0_8, r1_8);
        float2 s_8 = fma_f32x2_rn_noftz(r_8, fma_f32x2_rn_noftz(nd_8, r_8, make_float2(1.0f, 1.0f)), r_8);
        float2 _f32x2_mul_a_29 = mul_f32x2_noftz(cvt_f32_9, s_8);
        float2 mul_8;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&cvt_f32_1_8));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_8.x, mul_8.y)));
        float2 cvt_f32_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_9 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 _f32x2_mul_b_30 = make_float2(-1.442695f, -1.442695f);
        float2 t_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_9) : "l"(*(const unsigned long long*)&cvt_f32_10), "l"(*(const unsigned long long*)&_f32x2_mul_b_30));
        float e0_9 = approx_exp2(t_9.x);
        float e1_9 = approx_exp2(t_9.y);
        float2 nd_9 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_9, e1_9));
        float d0_9 = -nd_9.x;
        float d1_9 = -nd_9.y;
        float r0_9 = approx_rcp(d0_9);
        float r1_9 = approx_rcp(d1_9);
        float2 r_9 = make_float2(r0_9, r1_9);
        float2 s_9 = fma_f32x2_rn_noftz(r_9, fma_f32x2_rn_noftz(nd_9, r_9, make_float2(1.0f, 1.0f)), r_9);
        float2 _f32x2_mul_a_31 = mul_f32x2_noftz(cvt_f32_10, s_9);
        float2 mul_9;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&cvt_f32_1_9));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_9.x, mul_9.y)));
        float2 cvt_f32_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_10 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
        float2 t_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_10) : "l"(*(const unsigned long long*)&cvt_f32_11), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
        float e0_10 = approx_exp2(t_10.x);
        float e1_10 = approx_exp2(t_10.y);
        float2 nd_10 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_10, e1_10));
        float d0_10 = -nd_10.x;
        float d1_10 = -nd_10.y;
        float r0_10 = approx_rcp(d0_10);
        float r1_10 = approx_rcp(d1_10);
        float2 r_10 = make_float2(r0_10, r1_10);
        float2 s_10 = fma_f32x2_rn_noftz(r_10, fma_f32x2_rn_noftz(nd_10, r_10, make_float2(1.0f, 1.0f)), r_10);
        float2 _f32x2_mul_a_33 = mul_f32x2_noftz(cvt_f32_11, s_10);
        float2 mul_10;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_33), "l"(*(const unsigned long long*)&cvt_f32_1_10));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_10.x, mul_10.y)));
        float2 cvt_f32_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_11 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 _f32x2_mul_b_34 = make_float2(-1.442695f, -1.442695f);
        float2 t_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_11) : "l"(*(const unsigned long long*)&cvt_f32_12), "l"(*(const unsigned long long*)&_f32x2_mul_b_34));
        float e0_11 = approx_exp2(t_11.x);
        float e1_11 = approx_exp2(t_11.y);
        float2 nd_11 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_11, e1_11));
        float d0_11 = -nd_11.x;
        float d1_11 = -nd_11.y;
        float r0_11 = approx_rcp(d0_11);
        float r1_11 = approx_rcp(d1_11);
        float2 r_11 = make_float2(r0_11, r1_11);
        float2 s_11 = fma_f32x2_rn_noftz(r_11, fma_f32x2_rn_noftz(nd_11, r_11, make_float2(1.0f, 1.0f)), r_11);
        float2 _f32x2_mul_a_35 = mul_f32x2_noftz(cvt_f32_12, s_11);
        float2 mul_11;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&cvt_f32_1_11));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_11.x, mul_11.y)));
        float2 cvt_f32_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_12 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 _f32x2_mul_b_36 = make_float2(-1.442695f, -1.442695f);
        float2 t_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_12) : "l"(*(const unsigned long long*)&cvt_f32_13), "l"(*(const unsigned long long*)&_f32x2_mul_b_36));
        float e0_12 = approx_exp2(t_12.x);
        float e1_12 = approx_exp2(t_12.y);
        float2 nd_12 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_12, e1_12));
        float d0_12 = -nd_12.x;
        float d1_12 = -nd_12.y;
        float r0_12 = approx_rcp(d0_12);
        float r1_12 = approx_rcp(d1_12);
        float2 r_12 = make_float2(r0_12, r1_12);
        float2 s_12 = fma_f32x2_rn_noftz(r_12, fma_f32x2_rn_noftz(nd_12, r_12, make_float2(1.0f, 1.0f)), r_12);
        float2 _f32x2_mul_a_37 = mul_f32x2_noftz(cvt_f32_13, s_12);
        float2 mul_12;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_12) : "l"(*(const unsigned long long*)&_f32x2_mul_a_37), "l"(*(const unsigned long long*)&cvt_f32_1_12));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_12.x, mul_12.y)));
        float2 cvt_f32_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_13 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 _f32x2_mul_b_38 = make_float2(-1.442695f, -1.442695f);
        float2 t_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_13) : "l"(*(const unsigned long long*)&cvt_f32_14), "l"(*(const unsigned long long*)&_f32x2_mul_b_38));
        float e0_13 = approx_exp2(t_13.x);
        float e1_13 = approx_exp2(t_13.y);
        float2 nd_13 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_13, e1_13));
        float d0_13 = -nd_13.x;
        float d1_13 = -nd_13.y;
        float r0_13 = approx_rcp(d0_13);
        float r1_13 = approx_rcp(d1_13);
        float2 r_13 = make_float2(r0_13, r1_13);
        float2 s_13 = fma_f32x2_rn_noftz(r_13, fma_f32x2_rn_noftz(nd_13, r_13, make_float2(1.0f, 1.0f)), r_13);
        float2 _f32x2_mul_a_39 = mul_f32x2_noftz(cvt_f32_14, s_13);
        float2 mul_13;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_13) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&cvt_f32_1_13));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_13.x, mul_13.y)));
        float2 cvt_f32_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_14 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 _f32x2_mul_b_40 = make_float2(-1.442695f, -1.442695f);
        float2 t_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_14) : "l"(*(const unsigned long long*)&cvt_f32_15), "l"(*(const unsigned long long*)&_f32x2_mul_b_40));
        float e0_14 = approx_exp2(t_14.x);
        float e1_14 = approx_exp2(t_14.y);
        float2 nd_14 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_14, e1_14));
        float d0_14 = -nd_14.x;
        float d1_14 = -nd_14.y;
        float r0_14 = approx_rcp(d0_14);
        float r1_14 = approx_rcp(d1_14);
        float2 r_14 = make_float2(r0_14, r1_14);
        float2 s_14 = fma_f32x2_rn_noftz(r_14, fma_f32x2_rn_noftz(nd_14, r_14, make_float2(1.0f, 1.0f)), r_14);
        float2 _f32x2_mul_a_41 = mul_f32x2_noftz(cvt_f32_15, s_14);
        float2 mul_14;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_14) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&cvt_f32_1_14));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_14.x, mul_14.y)));
        float2 cvt_f32_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_15 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 _f32x2_mul_b_42 = make_float2(-1.442695f, -1.442695f);
        float2 t_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_15) : "l"(*(const unsigned long long*)&cvt_f32_16), "l"(*(const unsigned long long*)&_f32x2_mul_b_42));
        float e0_15 = approx_exp2(t_15.x);
        float e1_15 = approx_exp2(t_15.y);
        float2 nd_15 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_15, e1_15));
        float d0_15 = -nd_15.x;
        float d1_15 = -nd_15.y;
        float r0_15 = approx_rcp(d0_15);
        float r1_15 = approx_rcp(d1_15);
        float2 r_15 = make_float2(r0_15, r1_15);
        float2 s_15 = fma_f32x2_rn_noftz(r_15, fma_f32x2_rn_noftz(nd_15, r_15, make_float2(1.0f, 1.0f)), r_15);
        float2 _f32x2_mul_a_43 = mul_f32x2_noftz(cvt_f32_16, s_15);
        float2 mul_15;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_15) : "l"(*(const unsigned long long*)&_f32x2_mul_a_43), "l"(*(const unsigned long long*)&cvt_f32_1_15));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_15.x, mul_15.y)));
        float2 cvt_f32_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_16 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 _f32x2_mul_b_44 = make_float2(-1.442695f, -1.442695f);
        float2 t_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_16) : "l"(*(const unsigned long long*)&cvt_f32_17), "l"(*(const unsigned long long*)&_f32x2_mul_b_44));
        float e0_16 = approx_exp2(t_16.x);
        float e1_16 = approx_exp2(t_16.y);
        float2 nd_16 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_16, e1_16));
        float d0_16 = -nd_16.x;
        float d1_16 = -nd_16.y;
        float r0_16 = approx_rcp(d0_16);
        float r1_16 = approx_rcp(d1_16);
        float2 r_16 = make_float2(r0_16, r1_16);
        float2 s_16 = fma_f32x2_rn_noftz(r_16, fma_f32x2_rn_noftz(nd_16, r_16, make_float2(1.0f, 1.0f)), r_16);
        float2 _f32x2_mul_a_45 = mul_f32x2_noftz(cvt_f32_17, s_16);
        float2 mul_16;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_16) : "l"(*(const unsigned long long*)&_f32x2_mul_a_45), "l"(*(const unsigned long long*)&cvt_f32_1_16));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_16.x, mul_16.y)));
        float2 cvt_f32_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_17 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 _f32x2_mul_b_46 = make_float2(-1.442695f, -1.442695f);
        float2 t_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_17) : "l"(*(const unsigned long long*)&cvt_f32_18), "l"(*(const unsigned long long*)&_f32x2_mul_b_46));
        float e0_17 = approx_exp2(t_17.x);
        float e1_17 = approx_exp2(t_17.y);
        float2 nd_17 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_17, e1_17));
        float d0_17 = -nd_17.x;
        float d1_17 = -nd_17.y;
        float r0_17 = approx_rcp(d0_17);
        float r1_17 = approx_rcp(d1_17);
        float2 r_17 = make_float2(r0_17, r1_17);
        float2 s_17 = fma_f32x2_rn_noftz(r_17, fma_f32x2_rn_noftz(nd_17, r_17, make_float2(1.0f, 1.0f)), r_17);
        float2 _f32x2_mul_a_47 = mul_f32x2_noftz(cvt_f32_18, s_17);
        float2 mul_17;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_17) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&cvt_f32_1_17));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_17.x, mul_17.y)));
        float2 cvt_f32_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_18 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 _f32x2_mul_b_48 = make_float2(-1.442695f, -1.442695f);
        float2 t_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_18) : "l"(*(const unsigned long long*)&cvt_f32_19), "l"(*(const unsigned long long*)&_f32x2_mul_b_48));
        float e0_18 = approx_exp2(t_18.x);
        float e1_18 = approx_exp2(t_18.y);
        float2 nd_18 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_18, e1_18));
        float d0_18 = -nd_18.x;
        float d1_18 = -nd_18.y;
        float r0_18 = approx_rcp(d0_18);
        float r1_18 = approx_rcp(d1_18);
        float2 r_18 = make_float2(r0_18, r1_18);
        float2 s_18 = fma_f32x2_rn_noftz(r_18, fma_f32x2_rn_noftz(nd_18, r_18, make_float2(1.0f, 1.0f)), r_18);
        float2 _f32x2_mul_a_49 = mul_f32x2_noftz(cvt_f32_19, s_18);
        float2 mul_18;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_18) : "l"(*(const unsigned long long*)&_f32x2_mul_a_49), "l"(*(const unsigned long long*)&cvt_f32_1_18));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_18.x, mul_18.y)));
        float2 cvt_f32_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_19 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 _f32x2_mul_b_50 = make_float2(-1.442695f, -1.442695f);
        float2 t_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_19) : "l"(*(const unsigned long long*)&cvt_f32_20), "l"(*(const unsigned long long*)&_f32x2_mul_b_50));
        float e0_19 = approx_exp2(t_19.x);
        float e1_19 = approx_exp2(t_19.y);
        float2 nd_19 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_19, e1_19));
        float d0_19 = -nd_19.x;
        float d1_19 = -nd_19.y;
        float r0_19 = approx_rcp(d0_19);
        float r1_19 = approx_rcp(d1_19);
        float2 r_19 = make_float2(r0_19, r1_19);
        float2 s_19 = fma_f32x2_rn_noftz(r_19, fma_f32x2_rn_noftz(nd_19, r_19, make_float2(1.0f, 1.0f)), r_19);
        float2 _f32x2_mul_a_51 = mul_f32x2_noftz(cvt_f32_20, s_19);
        float2 mul_19;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_19) : "l"(*(const unsigned long long*)&_f32x2_mul_a_51), "l"(*(const unsigned long long*)&cvt_f32_1_19));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_19.x, mul_19.y)));
        float2 cvt_f32_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_20 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 _f32x2_mul_b_52 = make_float2(-1.442695f, -1.442695f);
        float2 t_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_20) : "l"(*(const unsigned long long*)&cvt_f32_21), "l"(*(const unsigned long long*)&_f32x2_mul_b_52));
        float e0_20 = approx_exp2(t_20.x);
        float e1_20 = approx_exp2(t_20.y);
        float2 nd_20 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_20, e1_20));
        float d0_20 = -nd_20.x;
        float d1_20 = -nd_20.y;
        float r0_20 = approx_rcp(d0_20);
        float r1_20 = approx_rcp(d1_20);
        float2 r_20 = make_float2(r0_20, r1_20);
        float2 s_20 = fma_f32x2_rn_noftz(r_20, fma_f32x2_rn_noftz(nd_20, r_20, make_float2(1.0f, 1.0f)), r_20);
        float2 _f32x2_mul_a_53 = mul_f32x2_noftz(cvt_f32_21, s_20);
        float2 mul_20;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_20) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&cvt_f32_1_20));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_20.x, mul_20.y)));
        float2 cvt_f32_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_21 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
        float2 t_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_21) : "l"(*(const unsigned long long*)&cvt_f32_22), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
        float e0_21 = approx_exp2(t_21.x);
        float e1_21 = approx_exp2(t_21.y);
        float2 nd_21 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_21, e1_21));
        float d0_21 = -nd_21.x;
        float d1_21 = -nd_21.y;
        float r0_21 = approx_rcp(d0_21);
        float r1_21 = approx_rcp(d1_21);
        float2 r_21 = make_float2(r0_21, r1_21);
        float2 s_21 = fma_f32x2_rn_noftz(r_21, fma_f32x2_rn_noftz(nd_21, r_21, make_float2(1.0f, 1.0f)), r_21);
        float2 _f32x2_mul_a_55 = mul_f32x2_noftz(cvt_f32_22, s_21);
        float2 mul_21;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_21) : "l"(*(const unsigned long long*)&_f32x2_mul_a_55), "l"(*(const unsigned long long*)&cvt_f32_1_21));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_21.x, mul_21.y)));
        float2 cvt_f32_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_22 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 _f32x2_mul_b_56 = make_float2(-1.442695f, -1.442695f);
        float2 t_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_22) : "l"(*(const unsigned long long*)&cvt_f32_23), "l"(*(const unsigned long long*)&_f32x2_mul_b_56));
        float e0_22 = approx_exp2(t_22.x);
        float e1_22 = approx_exp2(t_22.y);
        float2 nd_22 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_22, e1_22));
        float d0_22 = -nd_22.x;
        float d1_22 = -nd_22.y;
        float r0_22 = approx_rcp(d0_22);
        float r1_22 = approx_rcp(d1_22);
        float2 r_22 = make_float2(r0_22, r1_22);
        float2 s_22 = fma_f32x2_rn_noftz(r_22, fma_f32x2_rn_noftz(nd_22, r_22, make_float2(1.0f, 1.0f)), r_22);
        float2 _f32x2_mul_a_57 = mul_f32x2_noftz(cvt_f32_23, s_22);
        float2 mul_22;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_22) : "l"(*(const unsigned long long*)&_f32x2_mul_a_57), "l"(*(const unsigned long long*)&cvt_f32_1_22));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_22.x, mul_22.y)));
        float2 cvt_f32_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 cvt_f32_1_23 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 _f32x2_mul_b_58 = make_float2(-1.442695f, -1.442695f);
        float2 t_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_23) : "l"(*(const unsigned long long*)&cvt_f32_24), "l"(*(const unsigned long long*)&_f32x2_mul_b_58));
        float e0_23 = approx_exp2(t_23.x);
        float e1_23 = approx_exp2(t_23.y);
        float2 nd_23 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_23, e1_23));
        float d0_23 = -nd_23.x;
        float d1_23 = -nd_23.y;
        float r0_23 = approx_rcp(d0_23);
        float r1_23 = approx_rcp(d1_23);
        float2 r_23 = make_float2(r0_23, r1_23);
        float2 s_23 = fma_f32x2_rn_noftz(r_23, fma_f32x2_rn_noftz(nd_23, r_23, make_float2(1.0f, 1.0f)), r_23);
        float2 _f32x2_mul_a_59 = mul_f32x2_noftz(cvt_f32_24, s_23);
        float2 mul_23;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_23) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&cvt_f32_1_23));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_23.x, mul_23.y)));
        float2 cvt_f32_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 cvt_f32_1_24 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 _f32x2_mul_b_60 = make_float2(-1.442695f, -1.442695f);
        float2 t_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_24) : "l"(*(const unsigned long long*)&cvt_f32_25), "l"(*(const unsigned long long*)&_f32x2_mul_b_60));
        float e0_24 = approx_exp2(t_24.x);
        float e1_24 = approx_exp2(t_24.y);
        float2 nd_24 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_24, e1_24));
        float d0_24 = -nd_24.x;
        float d1_24 = -nd_24.y;
        float r0_24 = approx_rcp(d0_24);
        float r1_24 = approx_rcp(d1_24);
        float2 r_24 = make_float2(r0_24, r1_24);
        float2 s_24 = fma_f32x2_rn_noftz(r_24, fma_f32x2_rn_noftz(nd_24, r_24, make_float2(1.0f, 1.0f)), r_24);
        float2 _f32x2_mul_a_61 = mul_f32x2_noftz(cvt_f32_25, s_24);
        float2 mul_24;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_24) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&cvt_f32_1_24));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_24.x, mul_24.y)));
        float2 cvt_f32_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 cvt_f32_1_25 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
        float2 t_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_25) : "l"(*(const unsigned long long*)&cvt_f32_26), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
        float e0_25 = approx_exp2(t_25.x);
        float e1_25 = approx_exp2(t_25.y);
        float2 nd_25 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_25, e1_25));
        float d0_25 = -nd_25.x;
        float d1_25 = -nd_25.y;
        float r0_25 = approx_rcp(d0_25);
        float r1_25 = approx_rcp(d1_25);
        float2 r_25 = make_float2(r0_25, r1_25);
        float2 s_25 = fma_f32x2_rn_noftz(r_25, fma_f32x2_rn_noftz(nd_25, r_25, make_float2(1.0f, 1.0f)), r_25);
        float2 _f32x2_mul_a_63 = mul_f32x2_noftz(cvt_f32_26, s_25);
        float2 mul_25;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_25) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&cvt_f32_1_25));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_25.x, mul_25.y)));
        float2 cvt_f32_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 cvt_f32_1_26 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 _f32x2_mul_b_64 = make_float2(-1.442695f, -1.442695f);
        float2 t_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_26) : "l"(*(const unsigned long long*)&cvt_f32_27), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
        float e0_26 = approx_exp2(t_26.x);
        float e1_26 = approx_exp2(t_26.y);
        float2 nd_26 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_26, e1_26));
        float d0_26 = -nd_26.x;
        float d1_26 = -nd_26.y;
        float r0_26 = approx_rcp(d0_26);
        float r1_26 = approx_rcp(d1_26);
        float2 r_26 = make_float2(r0_26, r1_26);
        float2 s_26 = fma_f32x2_rn_noftz(r_26, fma_f32x2_rn_noftz(nd_26, r_26, make_float2(1.0f, 1.0f)), r_26);
        float2 _f32x2_mul_a_65 = mul_f32x2_noftz(cvt_f32_27, s_26);
        float2 mul_26;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_26) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&cvt_f32_1_26));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_26.x, mul_26.y)));
        float2 cvt_f32_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 cvt_f32_1_27 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 _f32x2_mul_b_66 = make_float2(-1.442695f, -1.442695f);
        float2 t_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_27) : "l"(*(const unsigned long long*)&cvt_f32_28), "l"(*(const unsigned long long*)&_f32x2_mul_b_66));
        float e0_27 = approx_exp2(t_27.x);
        float e1_27 = approx_exp2(t_27.y);
        float2 nd_27 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_27, e1_27));
        float d0_27 = -nd_27.x;
        float d1_27 = -nd_27.y;
        float r0_27 = approx_rcp(d0_27);
        float r1_27 = approx_rcp(d1_27);
        float2 r_27 = make_float2(r0_27, r1_27);
        float2 s_27 = fma_f32x2_rn_noftz(r_27, fma_f32x2_rn_noftz(nd_27, r_27, make_float2(1.0f, 1.0f)), r_27);
        float2 _f32x2_mul_a_67 = mul_f32x2_noftz(cvt_f32_28, s_27);
        float2 mul_27;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_27) : "l"(*(const unsigned long long*)&_f32x2_mul_a_67), "l"(*(const unsigned long long*)&cvt_f32_1_27));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_27.x, mul_27.y)));
        float2 cvt_f32_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 cvt_f32_1_28 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 _f32x2_mul_b_68 = make_float2(-1.442695f, -1.442695f);
        float2 t_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_28) : "l"(*(const unsigned long long*)&cvt_f32_29), "l"(*(const unsigned long long*)&_f32x2_mul_b_68));
        float e0_28 = approx_exp2(t_28.x);
        float e1_28 = approx_exp2(t_28.y);
        float2 nd_28 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_28, e1_28));
        float d0_28 = -nd_28.x;
        float d1_28 = -nd_28.y;
        float r0_28 = approx_rcp(d0_28);
        float r1_28 = approx_rcp(d1_28);
        float2 r_28 = make_float2(r0_28, r1_28);
        float2 s_28 = fma_f32x2_rn_noftz(r_28, fma_f32x2_rn_noftz(nd_28, r_28, make_float2(1.0f, 1.0f)), r_28);
        float2 _f32x2_mul_a_69 = mul_f32x2_noftz(cvt_f32_29, s_28);
        float2 mul_28;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_28) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&cvt_f32_1_28));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_28.x, mul_28.y)));
        float2 cvt_f32_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 cvt_f32_1_29 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 _f32x2_mul_b_70 = make_float2(-1.442695f, -1.442695f);
        float2 t_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_29) : "l"(*(const unsigned long long*)&cvt_f32_30), "l"(*(const unsigned long long*)&_f32x2_mul_b_70));
        float e0_29 = approx_exp2(t_29.x);
        float e1_29 = approx_exp2(t_29.y);
        float2 nd_29 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_29, e1_29));
        float d0_29 = -nd_29.x;
        float d1_29 = -nd_29.y;
        float r0_29 = approx_rcp(d0_29);
        float r1_29 = approx_rcp(d1_29);
        float2 r_29 = make_float2(r0_29, r1_29);
        float2 s_29 = fma_f32x2_rn_noftz(r_29, fma_f32x2_rn_noftz(nd_29, r_29, make_float2(1.0f, 1.0f)), r_29);
        float2 _f32x2_mul_a_71 = mul_f32x2_noftz(cvt_f32_30, s_29);
        float2 mul_29;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_29) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&cvt_f32_1_29));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_29.x, mul_29.y)));
        float2 cvt_f32_31 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 cvt_f32_1_30 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 _f32x2_mul_b_72 = make_float2(-1.442695f, -1.442695f);
        float2 t_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_30) : "l"(*(const unsigned long long*)&cvt_f32_31), "l"(*(const unsigned long long*)&_f32x2_mul_b_72));
        float e0_30 = approx_exp2(t_30.x);
        float e1_30 = approx_exp2(t_30.y);
        float2 nd_30 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_30, e1_30));
        float d0_30 = -nd_30.x;
        float d1_30 = -nd_30.y;
        float r0_30 = approx_rcp(d0_30);
        float r1_30 = approx_rcp(d1_30);
        float2 r_30 = make_float2(r0_30, r1_30);
        float2 s_30 = fma_f32x2_rn_noftz(r_30, fma_f32x2_rn_noftz(nd_30, r_30, make_float2(1.0f, 1.0f)), r_30);
        float2 _f32x2_mul_a_73 = mul_f32x2_noftz(cvt_f32_31, s_30);
        float2 mul_30;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_30) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&cvt_f32_1_30));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_30.x, mul_30.y)));
    } else {
        int band = 0;
        float2 cvt_f32_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[0]));
        float2 cvt_f32_1_31 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[0]));
        float2 _f32x2_mul_b_74 = make_float2(-1.442695f, -1.442695f);
        float2 t_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31) : "l"(*(const unsigned long long*)&cvt_f32_32), "l"(*(const unsigned long long*)&_f32x2_mul_b_74));
        float e0_31 = approx_exp2(t_31.x);
        float e1_31 = approx_exp2(t_31.y);
        float2 nd_31 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31, e1_31));
        float d0_31 = -nd_31.x;
        float d1_31 = -nd_31.y;
        float r0_31 = approx_rcp(d0_31);
        float r1_31 = approx_rcp(d1_31);
        float2 r_31 = make_float2(r0_31, r1_31);
        float2 s_31 = fma_f32x2_rn_noftz(r_31, fma_f32x2_rn_noftz(nd_31, r_31, make_float2(1.0f, 1.0f)), r_31);
        float t0 = s_31.x;
        float t1 = s_31.y;
        band = band | ((fminf(t0, t1) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_75 = mul_f32x2_noftz(cvt_f32_32, make_float2(max_noftz(t0, 0.0f), max_noftz(t1, 0.0f)));
        float2 mul_31;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31) : "l"(*(const unsigned long long*)&_f32x2_mul_a_75), "l"(*(const unsigned long long*)&cvt_f32_1_31));
        packed[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_31.x, mul_31.y)));
        float2 cvt_f32_32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[1]));
        float2 cvt_f32_1_31_1 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[1]));
        float2 _f32x2_mul_b_76 = make_float2(-1.442695f, -1.442695f);
        float2 t_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_31_1) : "l"(*(const unsigned long long*)&cvt_f32_32_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_76));
        float e0_31_1 = approx_exp2(t_31_1.x);
        float e1_31_1 = approx_exp2(t_31_1.y);
        float2 nd_31_1 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_31_1, e1_31_1));
        float d0_31_1 = -nd_31_1.x;
        float d1_31_1 = -nd_31_1.y;
        float r0_31_1 = approx_rcp(d0_31_1);
        float r1_31_1 = approx_rcp(d1_31_1);
        float2 r_31_1 = make_float2(r0_31_1, r1_31_1);
        float2 s_31_1 = fma_f32x2_rn_noftz(r_31_1, fma_f32x2_rn_noftz(nd_31_1, r_31_1, make_float2(1.0f, 1.0f)), r_31_1);
        float t0_0 = s_31_1.x;
        float t1_0 = s_31_1.y;
        band = band | ((fminf(t0_0, t1_0) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_77 = mul_f32x2_noftz(cvt_f32_32_1, make_float2(max_noftz(t0_0, 0.0f), max_noftz(t1_0, 0.0f)));
        float2 mul_31_1;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_31_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_77), "l"(*(const unsigned long long*)&cvt_f32_1_31_1));
        packed[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_31_1.x, mul_31_1.y)));
        float2 cvt_f32_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[2]));
        float2 cvt_f32_1_32 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[2]));
        float2 _f32x2_mul_b_78 = make_float2(-1.442695f, -1.442695f);
        float2 t_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_32) : "l"(*(const unsigned long long*)&cvt_f32_33), "l"(*(const unsigned long long*)&_f32x2_mul_b_78));
        float e0_32 = approx_exp2(t_32.x);
        float e1_32 = approx_exp2(t_32.y);
        float2 nd_32 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_32, e1_32));
        float d0_32 = -nd_32.x;
        float d1_32 = -nd_32.y;
        float r0_32 = approx_rcp(d0_32);
        float r1_32 = approx_rcp(d1_32);
        float2 r_32 = make_float2(r0_32, r1_32);
        float2 s_32 = fma_f32x2_rn_noftz(r_32, fma_f32x2_rn_noftz(nd_32, r_32, make_float2(1.0f, 1.0f)), r_32);
        float t0_1 = s_32.x;
        float t1_1 = s_32.y;
        band = band | ((fminf(t0_1, t1_1) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_79 = mul_f32x2_noftz(cvt_f32_33, make_float2(max_noftz(t0_1, 0.0f), max_noftz(t1_1, 0.0f)));
        float2 mul_32;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_32) : "l"(*(const unsigned long long*)&_f32x2_mul_a_79), "l"(*(const unsigned long long*)&cvt_f32_1_32));
        packed[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_32.x, mul_32.y)));
        float2 cvt_f32_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[3]));
        float2 cvt_f32_1_33 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[3]));
        float2 _f32x2_mul_b_80 = make_float2(-1.442695f, -1.442695f);
        float2 t_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_33) : "l"(*(const unsigned long long*)&cvt_f32_34), "l"(*(const unsigned long long*)&_f32x2_mul_b_80));
        float e0_33 = approx_exp2(t_33.x);
        float e1_33 = approx_exp2(t_33.y);
        float2 nd_33 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_33, e1_33));
        float d0_33 = -nd_33.x;
        float d1_33 = -nd_33.y;
        float r0_33 = approx_rcp(d0_33);
        float r1_33 = approx_rcp(d1_33);
        float2 r_33 = make_float2(r0_33, r1_33);
        float2 s_33 = fma_f32x2_rn_noftz(r_33, fma_f32x2_rn_noftz(nd_33, r_33, make_float2(1.0f, 1.0f)), r_33);
        float t0_2 = s_33.x;
        float t1_2 = s_33.y;
        band = band | ((fminf(t0_2, t1_2) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_81 = mul_f32x2_noftz(cvt_f32_34, make_float2(max_noftz(t0_2, 0.0f), max_noftz(t1_2, 0.0f)));
        float2 mul_33;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_33) : "l"(*(const unsigned long long*)&_f32x2_mul_a_81), "l"(*(const unsigned long long*)&cvt_f32_1_33));
        packed[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_33.x, mul_33.y)));
        float2 cvt_f32_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[4]));
        float2 cvt_f32_1_34 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[4]));
        float2 _f32x2_mul_b_82 = make_float2(-1.442695f, -1.442695f);
        float2 t_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_34) : "l"(*(const unsigned long long*)&cvt_f32_35), "l"(*(const unsigned long long*)&_f32x2_mul_b_82));
        float e0_34 = approx_exp2(t_34.x);
        float e1_34 = approx_exp2(t_34.y);
        float2 nd_34 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_34, e1_34));
        float d0_34 = -nd_34.x;
        float d1_34 = -nd_34.y;
        float r0_34 = approx_rcp(d0_34);
        float r1_34 = approx_rcp(d1_34);
        float2 r_34 = make_float2(r0_34, r1_34);
        float2 s_34 = fma_f32x2_rn_noftz(r_34, fma_f32x2_rn_noftz(nd_34, r_34, make_float2(1.0f, 1.0f)), r_34);
        float t0_3 = s_34.x;
        float t1_3 = s_34.y;
        band = band | ((fminf(t0_3, t1_3) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_83 = mul_f32x2_noftz(cvt_f32_35, make_float2(max_noftz(t0_3, 0.0f), max_noftz(t1_3, 0.0f)));
        float2 mul_34;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_34) : "l"(*(const unsigned long long*)&_f32x2_mul_a_83), "l"(*(const unsigned long long*)&cvt_f32_1_34));
        packed[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_34.x, mul_34.y)));
        float2 cvt_f32_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[5]));
        float2 cvt_f32_1_35 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[5]));
        float2 _f32x2_mul_b_84 = make_float2(-1.442695f, -1.442695f);
        float2 t_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_35) : "l"(*(const unsigned long long*)&cvt_f32_36), "l"(*(const unsigned long long*)&_f32x2_mul_b_84));
        float e0_35 = approx_exp2(t_35.x);
        float e1_35 = approx_exp2(t_35.y);
        float2 nd_35 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_35, e1_35));
        float d0_35 = -nd_35.x;
        float d1_35 = -nd_35.y;
        float r0_35 = approx_rcp(d0_35);
        float r1_35 = approx_rcp(d1_35);
        float2 r_35 = make_float2(r0_35, r1_35);
        float2 s_35 = fma_f32x2_rn_noftz(r_35, fma_f32x2_rn_noftz(nd_35, r_35, make_float2(1.0f, 1.0f)), r_35);
        float t0_4 = s_35.x;
        float t1_4 = s_35.y;
        band = band | ((fminf(t0_4, t1_4) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_85 = mul_f32x2_noftz(cvt_f32_36, make_float2(max_noftz(t0_4, 0.0f), max_noftz(t1_4, 0.0f)));
        float2 mul_35;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_35) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&cvt_f32_1_35));
        packed[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_35.x, mul_35.y)));
        float2 cvt_f32_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[6]));
        float2 cvt_f32_1_36 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[6]));
        float2 _f32x2_mul_b_86 = make_float2(-1.442695f, -1.442695f);
        float2 t_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_36) : "l"(*(const unsigned long long*)&cvt_f32_37), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
        float e0_36 = approx_exp2(t_36.x);
        float e1_36 = approx_exp2(t_36.y);
        float2 nd_36 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_36, e1_36));
        float d0_36 = -nd_36.x;
        float d1_36 = -nd_36.y;
        float r0_36 = approx_rcp(d0_36);
        float r1_36 = approx_rcp(d1_36);
        float2 r_36 = make_float2(r0_36, r1_36);
        float2 s_36 = fma_f32x2_rn_noftz(r_36, fma_f32x2_rn_noftz(nd_36, r_36, make_float2(1.0f, 1.0f)), r_36);
        float t0_5 = s_36.x;
        float t1_5 = s_36.y;
        band = band | ((fminf(t0_5, t1_5) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_87 = mul_f32x2_noftz(cvt_f32_37, make_float2(max_noftz(t0_5, 0.0f), max_noftz(t1_5, 0.0f)));
        float2 mul_36;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_36) : "l"(*(const unsigned long long*)&_f32x2_mul_a_87), "l"(*(const unsigned long long*)&cvt_f32_1_36));
        packed[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_36.x, mul_36.y)));
        float2 cvt_f32_38 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_0[7]));
        float2 cvt_f32_1_37 = cake_bf16x2_to_f32x2(__as_bf16x2(_vec_load_1[7]));
        float2 _f32x2_mul_b_88 = make_float2(-1.442695f, -1.442695f);
        float2 t_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_37) : "l"(*(const unsigned long long*)&cvt_f32_38), "l"(*(const unsigned long long*)&_f32x2_mul_b_88));
        float e0_37 = approx_exp2(t_37.x);
        float e1_37 = approx_exp2(t_37.y);
        float2 nd_37 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_37, e1_37));
        float d0_37 = -nd_37.x;
        float d1_37 = -nd_37.y;
        float r0_37 = approx_rcp(d0_37);
        float r1_37 = approx_rcp(d1_37);
        float2 r_37 = make_float2(r0_37, r1_37);
        float2 s_37 = fma_f32x2_rn_noftz(r_37, fma_f32x2_rn_noftz(nd_37, r_37, make_float2(1.0f, 1.0f)), r_37);
        float t0_6 = s_37.x;
        float t1_6 = s_37.y;
        band = band | ((fminf(t0_6, t1_6) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_89 = mul_f32x2_noftz(cvt_f32_38, make_float2(max_noftz(t0_6, 0.0f), max_noftz(t1_6, 0.0f)));
        float2 mul_37;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_37) : "l"(*(const unsigned long long*)&_f32x2_mul_a_89), "l"(*(const unsigned long long*)&cvt_f32_1_37));
        packed[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_37.x, mul_37.y)));
        if (band != 0) {
            unsigned int _popc_0 = (unsigned int)__popc(band);
            #pragma unroll 1
            for (int _j = 0; _j < _popc_0; _j++) {
                int _find_first_set_90 = __ffs(band);
                int j = _find_first_set_90 - 1;
                band = band & band - 1;
                unsigned int pair = band_pair(gated_input, gate_index + 2 * j, K);
                packed[0] = ((j == 0) ? pair : packed[0]);
                packed[1] = ((j == 1) ? pair : packed[1]);
                packed[2] = ((j == 2) ? pair : packed[2]);
                packed[3] = ((j == 3) ? pair : packed[3]);
                packed[4] = ((j == 4) ? pair : packed[4]);
                packed[5] = ((j == 5) ? pair : packed[5]);
                packed[6] = ((j == 6) ? pair : packed[6]);
                packed[7] = ((j == 7) ? pair : packed[7]);
            }
        }
        int band_0 = 0;
        float2 cvt_f32_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[0]));
        float2 cvt_f32_1_38 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[0]));
        float2 _f32x2_mul_b_91 = make_float2(-1.442695f, -1.442695f);
        float2 t_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_38) : "l"(*(const unsigned long long*)&cvt_f32_39), "l"(*(const unsigned long long*)&_f32x2_mul_b_91));
        float e0_38 = approx_exp2(t_38.x);
        float e1_38 = approx_exp2(t_38.y);
        float2 nd_38 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_38, e1_38));
        float d0_38 = -nd_38.x;
        float d1_38 = -nd_38.y;
        float r0_38 = approx_rcp(d0_38);
        float r1_38 = approx_rcp(d1_38);
        float2 r_38 = make_float2(r0_38, r1_38);
        float2 s_38 = fma_f32x2_rn_noftz(r_38, fma_f32x2_rn_noftz(nd_38, r_38, make_float2(1.0f, 1.0f)), r_38);
        float t0_7 = s_38.x;
        float t1_7 = s_38.y;
        band_0 = band_0 | ((fminf(t0_7, t1_7) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_92 = mul_f32x2_noftz(cvt_f32_39, make_float2(max_noftz(t0_7, 0.0f), max_noftz(t1_7, 0.0f)));
        float2 mul_38;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_38) : "l"(*(const unsigned long long*)&_f32x2_mul_a_92), "l"(*(const unsigned long long*)&cvt_f32_1_38));
        packed_b[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_38.x, mul_38.y)));
        float2 cvt_f32_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[1]));
        float2 cvt_f32_1_39 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[1]));
        float2 _f32x2_mul_b_93 = make_float2(-1.442695f, -1.442695f);
        float2 t_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_39) : "l"(*(const unsigned long long*)&cvt_f32_40), "l"(*(const unsigned long long*)&_f32x2_mul_b_93));
        float e0_39 = approx_exp2(t_39.x);
        float e1_39 = approx_exp2(t_39.y);
        float2 nd_39 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_39, e1_39));
        float d0_39 = -nd_39.x;
        float d1_39 = -nd_39.y;
        float r0_39 = approx_rcp(d0_39);
        float r1_39 = approx_rcp(d1_39);
        float2 r_39 = make_float2(r0_39, r1_39);
        float2 s_39 = fma_f32x2_rn_noftz(r_39, fma_f32x2_rn_noftz(nd_39, r_39, make_float2(1.0f, 1.0f)), r_39);
        float t0_8 = s_39.x;
        float t1_8 = s_39.y;
        band_0 = band_0 | ((fminf(t0_8, t1_8) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_94 = mul_f32x2_noftz(cvt_f32_40, make_float2(max_noftz(t0_8, 0.0f), max_noftz(t1_8, 0.0f)));
        float2 mul_39;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_39) : "l"(*(const unsigned long long*)&_f32x2_mul_a_94), "l"(*(const unsigned long long*)&cvt_f32_1_39));
        packed_b[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_39.x, mul_39.y)));
        float2 cvt_f32_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[2]));
        float2 cvt_f32_1_40 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[2]));
        float2 _f32x2_mul_b_95 = make_float2(-1.442695f, -1.442695f);
        float2 t_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_40) : "l"(*(const unsigned long long*)&cvt_f32_41), "l"(*(const unsigned long long*)&_f32x2_mul_b_95));
        float e0_40 = approx_exp2(t_40.x);
        float e1_40 = approx_exp2(t_40.y);
        float2 nd_40 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_40, e1_40));
        float d0_40 = -nd_40.x;
        float d1_40 = -nd_40.y;
        float r0_40 = approx_rcp(d0_40);
        float r1_40 = approx_rcp(d1_40);
        float2 r_40 = make_float2(r0_40, r1_40);
        float2 s_40 = fma_f32x2_rn_noftz(r_40, fma_f32x2_rn_noftz(nd_40, r_40, make_float2(1.0f, 1.0f)), r_40);
        float t0_9 = s_40.x;
        float t1_9 = s_40.y;
        band_0 = band_0 | ((fminf(t0_9, t1_9) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_96 = mul_f32x2_noftz(cvt_f32_41, make_float2(max_noftz(t0_9, 0.0f), max_noftz(t1_9, 0.0f)));
        float2 mul_40;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_40) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&cvt_f32_1_40));
        packed_b[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_40.x, mul_40.y)));
        float2 cvt_f32_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[3]));
        float2 cvt_f32_1_41 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[3]));
        float2 _f32x2_mul_b_97 = make_float2(-1.442695f, -1.442695f);
        float2 t_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_41) : "l"(*(const unsigned long long*)&cvt_f32_42), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
        float e0_41 = approx_exp2(t_41.x);
        float e1_41 = approx_exp2(t_41.y);
        float2 nd_41 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_41, e1_41));
        float d0_41 = -nd_41.x;
        float d1_41 = -nd_41.y;
        float r0_41 = approx_rcp(d0_41);
        float r1_41 = approx_rcp(d1_41);
        float2 r_41 = make_float2(r0_41, r1_41);
        float2 s_41 = fma_f32x2_rn_noftz(r_41, fma_f32x2_rn_noftz(nd_41, r_41, make_float2(1.0f, 1.0f)), r_41);
        float t0_10 = s_41.x;
        float t1_10 = s_41.y;
        band_0 = band_0 | ((fminf(t0_10, t1_10) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_98 = mul_f32x2_noftz(cvt_f32_42, make_float2(max_noftz(t0_10, 0.0f), max_noftz(t1_10, 0.0f)));
        float2 mul_41;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_41) : "l"(*(const unsigned long long*)&_f32x2_mul_a_98), "l"(*(const unsigned long long*)&cvt_f32_1_41));
        packed_b[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_41.x, mul_41.y)));
        float2 cvt_f32_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[4]));
        float2 cvt_f32_1_42 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[4]));
        float2 _f32x2_mul_b_99 = make_float2(-1.442695f, -1.442695f);
        float2 t_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_42) : "l"(*(const unsigned long long*)&cvt_f32_43), "l"(*(const unsigned long long*)&_f32x2_mul_b_99));
        float e0_42 = approx_exp2(t_42.x);
        float e1_42 = approx_exp2(t_42.y);
        float2 nd_42 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_42, e1_42));
        float d0_42 = -nd_42.x;
        float d1_42 = -nd_42.y;
        float r0_42 = approx_rcp(d0_42);
        float r1_42 = approx_rcp(d1_42);
        float2 r_42 = make_float2(r0_42, r1_42);
        float2 s_42 = fma_f32x2_rn_noftz(r_42, fma_f32x2_rn_noftz(nd_42, r_42, make_float2(1.0f, 1.0f)), r_42);
        float t0_11 = s_42.x;
        float t1_11 = s_42.y;
        band_0 = band_0 | ((fminf(t0_11, t1_11) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_100 = mul_f32x2_noftz(cvt_f32_43, make_float2(max_noftz(t0_11, 0.0f), max_noftz(t1_11, 0.0f)));
        float2 mul_42;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_42) : "l"(*(const unsigned long long*)&_f32x2_mul_a_100), "l"(*(const unsigned long long*)&cvt_f32_1_42));
        packed_b[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_42.x, mul_42.y)));
        float2 cvt_f32_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[5]));
        float2 cvt_f32_1_43 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[5]));
        float2 _f32x2_mul_b_101 = make_float2(-1.442695f, -1.442695f);
        float2 t_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_43) : "l"(*(const unsigned long long*)&cvt_f32_44), "l"(*(const unsigned long long*)&_f32x2_mul_b_101));
        float e0_43 = approx_exp2(t_43.x);
        float e1_43 = approx_exp2(t_43.y);
        float2 nd_43 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_43, e1_43));
        float d0_43 = -nd_43.x;
        float d1_43 = -nd_43.y;
        float r0_43 = approx_rcp(d0_43);
        float r1_43 = approx_rcp(d1_43);
        float2 r_43 = make_float2(r0_43, r1_43);
        float2 s_43 = fma_f32x2_rn_noftz(r_43, fma_f32x2_rn_noftz(nd_43, r_43, make_float2(1.0f, 1.0f)), r_43);
        float t0_12 = s_43.x;
        float t1_12 = s_43.y;
        band_0 = band_0 | ((fminf(t0_12, t1_12) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_102 = mul_f32x2_noftz(cvt_f32_44, make_float2(max_noftz(t0_12, 0.0f), max_noftz(t1_12, 0.0f)));
        float2 mul_43;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_43) : "l"(*(const unsigned long long*)&_f32x2_mul_a_102), "l"(*(const unsigned long long*)&cvt_f32_1_43));
        packed_b[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_43.x, mul_43.y)));
        float2 cvt_f32_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[6]));
        float2 cvt_f32_1_44 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[6]));
        float2 _f32x2_mul_b_103 = make_float2(-1.442695f, -1.442695f);
        float2 t_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_44) : "l"(*(const unsigned long long*)&cvt_f32_45), "l"(*(const unsigned long long*)&_f32x2_mul_b_103));
        float e0_44 = approx_exp2(t_44.x);
        float e1_44 = approx_exp2(t_44.y);
        float2 nd_44 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_44, e1_44));
        float d0_44 = -nd_44.x;
        float d1_44 = -nd_44.y;
        float r0_44 = approx_rcp(d0_44);
        float r1_44 = approx_rcp(d1_44);
        float2 r_44 = make_float2(r0_44, r1_44);
        float2 s_44 = fma_f32x2_rn_noftz(r_44, fma_f32x2_rn_noftz(nd_44, r_44, make_float2(1.0f, 1.0f)), r_44);
        float t0_13 = s_44.x;
        float t1_13 = s_44.y;
        band_0 = band_0 | ((fminf(t0_13, t1_13) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_104 = mul_f32x2_noftz(cvt_f32_45, make_float2(max_noftz(t0_13, 0.0f), max_noftz(t1_13, 0.0f)));
        float2 mul_44;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_44) : "l"(*(const unsigned long long*)&_f32x2_mul_a_104), "l"(*(const unsigned long long*)&cvt_f32_1_44));
        packed_b[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_44.x, mul_44.y)));
        float2 cvt_f32_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load[7]));
        float2 cvt_f32_1_45 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1[7]));
        float2 _f32x2_mul_b_105 = make_float2(-1.442695f, -1.442695f);
        float2 t_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_45) : "l"(*(const unsigned long long*)&cvt_f32_46), "l"(*(const unsigned long long*)&_f32x2_mul_b_105));
        float e0_45 = approx_exp2(t_45.x);
        float e1_45 = approx_exp2(t_45.y);
        float2 nd_45 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_45, e1_45));
        float d0_45 = -nd_45.x;
        float d1_45 = -nd_45.y;
        float r0_45 = approx_rcp(d0_45);
        float r1_45 = approx_rcp(d1_45);
        float2 r_45 = make_float2(r0_45, r1_45);
        float2 s_45 = fma_f32x2_rn_noftz(r_45, fma_f32x2_rn_noftz(nd_45, r_45, make_float2(1.0f, 1.0f)), r_45);
        float t0_14 = s_45.x;
        float t1_14 = s_45.y;
        band_0 = band_0 | ((fminf(t0_14, t1_14) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_106 = mul_f32x2_noftz(cvt_f32_46, make_float2(max_noftz(t0_14, 0.0f), max_noftz(t1_14, 0.0f)));
        float2 mul_45;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_45) : "l"(*(const unsigned long long*)&_f32x2_mul_a_106), "l"(*(const unsigned long long*)&cvt_f32_1_45));
        packed_b[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_45.x, mul_45.y)));
        if (band_0 != 0) {
            unsigned int popc = (unsigned int)__popc(band_0);
            #pragma unroll 1
            for (int _j_1 = 0; _j_1 < popc; _j_1++) {
                int _find_first_set_107 = __ffs(band_0);
                int j_1 = _find_first_set_107 - 1;
                band_0 = band_0 & band_0 - 1;
                unsigned int pair_1 = band_pair(gated_input, gate_index_3 + 2 * j_1, K);
                packed_b[0] = ((j_1 == 0) ? pair_1 : packed_b[0]);
                packed_b[1] = ((j_1 == 1) ? pair_1 : packed_b[1]);
                packed_b[2] = ((j_1 == 2) ? pair_1 : packed_b[2]);
                packed_b[3] = ((j_1 == 3) ? pair_1 : packed_b[3]);
                packed_b[4] = ((j_1 == 4) ? pair_1 : packed_b[4]);
                packed_b[5] = ((j_1 == 5) ? pair_1 : packed_b[5]);
                packed_b[6] = ((j_1 == 6) ? pair_1 : packed_b[6]);
                packed_b[7] = ((j_1 == 7) ? pair_1 : packed_b[7]);
            }
        }
        int band_1 = 0;
        float2 cvt_f32_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[0]));
        float2 cvt_f32_1_46 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[0]));
        float2 _f32x2_mul_b_108 = make_float2(-1.442695f, -1.442695f);
        float2 t_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_46) : "l"(*(const unsigned long long*)&cvt_f32_47), "l"(*(const unsigned long long*)&_f32x2_mul_b_108));
        float e0_46 = approx_exp2(t_46.x);
        float e1_46 = approx_exp2(t_46.y);
        float2 nd_46 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_46, e1_46));
        float d0_46 = -nd_46.x;
        float d1_46 = -nd_46.y;
        float r0_46 = approx_rcp(d0_46);
        float r1_46 = approx_rcp(d1_46);
        float2 r_46 = make_float2(r0_46, r1_46);
        float2 s_46 = fma_f32x2_rn_noftz(r_46, fma_f32x2_rn_noftz(nd_46, r_46, make_float2(1.0f, 1.0f)), r_46);
        float t0_15 = s_46.x;
        float t1_15 = s_46.y;
        band_1 = band_1 | ((fminf(t0_15, t1_15) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_109 = mul_f32x2_noftz(cvt_f32_47, make_float2(max_noftz(t0_15, 0.0f), max_noftz(t1_15, 0.0f)));
        float2 mul_46;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_46) : "l"(*(const unsigned long long*)&_f32x2_mul_a_109), "l"(*(const unsigned long long*)&cvt_f32_1_46));
        packed_c[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_46.x, mul_46.y)));
        float2 cvt_f32_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[1]));
        float2 cvt_f32_1_47 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[1]));
        float2 _f32x2_mul_b_110 = make_float2(-1.442695f, -1.442695f);
        float2 t_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_47) : "l"(*(const unsigned long long*)&cvt_f32_48), "l"(*(const unsigned long long*)&_f32x2_mul_b_110));
        float e0_47 = approx_exp2(t_47.x);
        float e1_47 = approx_exp2(t_47.y);
        float2 nd_47 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_47, e1_47));
        float d0_47 = -nd_47.x;
        float d1_47 = -nd_47.y;
        float r0_47 = approx_rcp(d0_47);
        float r1_47 = approx_rcp(d1_47);
        float2 r_47 = make_float2(r0_47, r1_47);
        float2 s_47 = fma_f32x2_rn_noftz(r_47, fma_f32x2_rn_noftz(nd_47, r_47, make_float2(1.0f, 1.0f)), r_47);
        float t0_16 = s_47.x;
        float t1_16 = s_47.y;
        band_1 = band_1 | ((fminf(t0_16, t1_16) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_111 = mul_f32x2_noftz(cvt_f32_48, make_float2(max_noftz(t0_16, 0.0f), max_noftz(t1_16, 0.0f)));
        float2 mul_47;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_47) : "l"(*(const unsigned long long*)&_f32x2_mul_a_111), "l"(*(const unsigned long long*)&cvt_f32_1_47));
        packed_c[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_47.x, mul_47.y)));
        float2 cvt_f32_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[2]));
        float2 cvt_f32_1_48 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[2]));
        float2 _f32x2_mul_b_112 = make_float2(-1.442695f, -1.442695f);
        float2 t_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_48) : "l"(*(const unsigned long long*)&cvt_f32_49), "l"(*(const unsigned long long*)&_f32x2_mul_b_112));
        float e0_48 = approx_exp2(t_48.x);
        float e1_48 = approx_exp2(t_48.y);
        float2 nd_48 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_48, e1_48));
        float d0_48 = -nd_48.x;
        float d1_48 = -nd_48.y;
        float r0_48 = approx_rcp(d0_48);
        float r1_48 = approx_rcp(d1_48);
        float2 r_48 = make_float2(r0_48, r1_48);
        float2 s_48 = fma_f32x2_rn_noftz(r_48, fma_f32x2_rn_noftz(nd_48, r_48, make_float2(1.0f, 1.0f)), r_48);
        float t0_17 = s_48.x;
        float t1_17 = s_48.y;
        band_1 = band_1 | ((fminf(t0_17, t1_17) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_113 = mul_f32x2_noftz(cvt_f32_49, make_float2(max_noftz(t0_17, 0.0f), max_noftz(t1_17, 0.0f)));
        float2 mul_48;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_48) : "l"(*(const unsigned long long*)&_f32x2_mul_a_113), "l"(*(const unsigned long long*)&cvt_f32_1_48));
        packed_c[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_48.x, mul_48.y)));
        float2 cvt_f32_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[3]));
        float2 cvt_f32_1_49 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[3]));
        float2 _f32x2_mul_b_114 = make_float2(-1.442695f, -1.442695f);
        float2 t_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_49) : "l"(*(const unsigned long long*)&cvt_f32_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_114));
        float e0_49 = approx_exp2(t_49.x);
        float e1_49 = approx_exp2(t_49.y);
        float2 nd_49 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_49, e1_49));
        float d0_49 = -nd_49.x;
        float d1_49 = -nd_49.y;
        float r0_49 = approx_rcp(d0_49);
        float r1_49 = approx_rcp(d1_49);
        float2 r_49 = make_float2(r0_49, r1_49);
        float2 s_49 = fma_f32x2_rn_noftz(r_49, fma_f32x2_rn_noftz(nd_49, r_49, make_float2(1.0f, 1.0f)), r_49);
        float t0_18 = s_49.x;
        float t1_18 = s_49.y;
        band_1 = band_1 | ((fminf(t0_18, t1_18) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_115 = mul_f32x2_noftz(cvt_f32_50, make_float2(max_noftz(t0_18, 0.0f), max_noftz(t1_18, 0.0f)));
        float2 mul_49;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_49) : "l"(*(const unsigned long long*)&_f32x2_mul_a_115), "l"(*(const unsigned long long*)&cvt_f32_1_49));
        packed_c[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_49.x, mul_49.y)));
        float2 cvt_f32_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[4]));
        float2 cvt_f32_1_50 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[4]));
        float2 _f32x2_mul_b_116 = make_float2(-1.442695f, -1.442695f);
        float2 t_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_50) : "l"(*(const unsigned long long*)&cvt_f32_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_116));
        float e0_50 = approx_exp2(t_50.x);
        float e1_50 = approx_exp2(t_50.y);
        float2 nd_50 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_50, e1_50));
        float d0_50 = -nd_50.x;
        float d1_50 = -nd_50.y;
        float r0_50 = approx_rcp(d0_50);
        float r1_50 = approx_rcp(d1_50);
        float2 r_50 = make_float2(r0_50, r1_50);
        float2 s_50 = fma_f32x2_rn_noftz(r_50, fma_f32x2_rn_noftz(nd_50, r_50, make_float2(1.0f, 1.0f)), r_50);
        float t0_19 = s_50.x;
        float t1_19 = s_50.y;
        band_1 = band_1 | ((fminf(t0_19, t1_19) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_117 = mul_f32x2_noftz(cvt_f32_51, make_float2(max_noftz(t0_19, 0.0f), max_noftz(t1_19, 0.0f)));
        float2 mul_50;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_50) : "l"(*(const unsigned long long*)&_f32x2_mul_a_117), "l"(*(const unsigned long long*)&cvt_f32_1_50));
        packed_c[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_50.x, mul_50.y)));
        float2 cvt_f32_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[5]));
        float2 cvt_f32_1_51 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[5]));
        float2 _f32x2_mul_b_118 = make_float2(-1.442695f, -1.442695f);
        float2 t_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_51) : "l"(*(const unsigned long long*)&cvt_f32_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_118));
        float e0_51 = approx_exp2(t_51.x);
        float e1_51 = approx_exp2(t_51.y);
        float2 nd_51 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_51, e1_51));
        float d0_51 = -nd_51.x;
        float d1_51 = -nd_51.y;
        float r0_51 = approx_rcp(d0_51);
        float r1_51 = approx_rcp(d1_51);
        float2 r_51 = make_float2(r0_51, r1_51);
        float2 s_51 = fma_f32x2_rn_noftz(r_51, fma_f32x2_rn_noftz(nd_51, r_51, make_float2(1.0f, 1.0f)), r_51);
        float t0_20 = s_51.x;
        float t1_20 = s_51.y;
        band_1 = band_1 | ((fminf(t0_20, t1_20) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_119 = mul_f32x2_noftz(cvt_f32_52, make_float2(max_noftz(t0_20, 0.0f), max_noftz(t1_20, 0.0f)));
        float2 mul_51;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_51) : "l"(*(const unsigned long long*)&_f32x2_mul_a_119), "l"(*(const unsigned long long*)&cvt_f32_1_51));
        packed_c[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_51.x, mul_51.y)));
        float2 cvt_f32_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[6]));
        float2 cvt_f32_1_52 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[6]));
        float2 _f32x2_mul_b_120 = make_float2(-1.442695f, -1.442695f);
        float2 t_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_52) : "l"(*(const unsigned long long*)&cvt_f32_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_120));
        float e0_52 = approx_exp2(t_52.x);
        float e1_52 = approx_exp2(t_52.y);
        float2 nd_52 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_52, e1_52));
        float d0_52 = -nd_52.x;
        float d1_52 = -nd_52.y;
        float r0_52 = approx_rcp(d0_52);
        float r1_52 = approx_rcp(d1_52);
        float2 r_52 = make_float2(r0_52, r1_52);
        float2 s_52 = fma_f32x2_rn_noftz(r_52, fma_f32x2_rn_noftz(nd_52, r_52, make_float2(1.0f, 1.0f)), r_52);
        float t0_21 = s_52.x;
        float t1_21 = s_52.y;
        band_1 = band_1 | ((fminf(t0_21, t1_21) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_121 = mul_f32x2_noftz(cvt_f32_53, make_float2(max_noftz(t0_21, 0.0f), max_noftz(t1_21, 0.0f)));
        float2 mul_52;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_52) : "l"(*(const unsigned long long*)&_f32x2_mul_a_121), "l"(*(const unsigned long long*)&cvt_f32_1_52));
        packed_c[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_52.x, mul_52.y)));
        float2 cvt_f32_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_0[7]));
        float2 cvt_f32_1_53 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_0[7]));
        float2 _f32x2_mul_b_122 = make_float2(-1.442695f, -1.442695f);
        float2 t_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_53) : "l"(*(const unsigned long long*)&cvt_f32_54), "l"(*(const unsigned long long*)&_f32x2_mul_b_122));
        float e0_53 = approx_exp2(t_53.x);
        float e1_53 = approx_exp2(t_53.y);
        float2 nd_53 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_53, e1_53));
        float d0_53 = -nd_53.x;
        float d1_53 = -nd_53.y;
        float r0_53 = approx_rcp(d0_53);
        float r1_53 = approx_rcp(d1_53);
        float2 r_53 = make_float2(r0_53, r1_53);
        float2 s_53 = fma_f32x2_rn_noftz(r_53, fma_f32x2_rn_noftz(nd_53, r_53, make_float2(1.0f, 1.0f)), r_53);
        float t0_22 = s_53.x;
        float t1_22 = s_53.y;
        band_1 = band_1 | ((fminf(t0_22, t1_22) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_123 = mul_f32x2_noftz(cvt_f32_54, make_float2(max_noftz(t0_22, 0.0f), max_noftz(t1_22, 0.0f)));
        float2 mul_53;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_53) : "l"(*(const unsigned long long*)&_f32x2_mul_a_123), "l"(*(const unsigned long long*)&cvt_f32_1_53));
        packed_c[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_53.x, mul_53.y)));
        if (band_1 != 0) {
            unsigned int popc_1 = (unsigned int)__popc(band_1);
            #pragma unroll 1
            for (int _j_2 = 0; _j_2 < popc_1; _j_2++) {
                int _find_first_set_124 = __ffs(band_1);
                int j_2 = _find_first_set_124 - 1;
                band_1 = band_1 & band_1 - 1;
                unsigned int pair_2 = band_pair(gated_input, gate_index_4 + 2 * j_2, K);
                packed_c[0] = ((j_2 == 0) ? pair_2 : packed_c[0]);
                packed_c[1] = ((j_2 == 1) ? pair_2 : packed_c[1]);
                packed_c[2] = ((j_2 == 2) ? pair_2 : packed_c[2]);
                packed_c[3] = ((j_2 == 3) ? pair_2 : packed_c[3]);
                packed_c[4] = ((j_2 == 4) ? pair_2 : packed_c[4]);
                packed_c[5] = ((j_2 == 5) ? pair_2 : packed_c[5]);
                packed_c[6] = ((j_2 == 6) ? pair_2 : packed_c[6]);
                packed_c[7] = ((j_2 == 7) ? pair_2 : packed_c[7]);
            }
        }
        int band_2 = 0;
        float2 cvt_f32_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[0]));
        float2 cvt_f32_1_54 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[0]));
        float2 _f32x2_mul_b_125 = make_float2(-1.442695f, -1.442695f);
        float2 t_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_54) : "l"(*(const unsigned long long*)&cvt_f32_55), "l"(*(const unsigned long long*)&_f32x2_mul_b_125));
        float e0_54 = approx_exp2(t_54.x);
        float e1_54 = approx_exp2(t_54.y);
        float2 nd_54 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_54, e1_54));
        float d0_54 = -nd_54.x;
        float d1_54 = -nd_54.y;
        float r0_54 = approx_rcp(d0_54);
        float r1_54 = approx_rcp(d1_54);
        float2 r_54 = make_float2(r0_54, r1_54);
        float2 s_54 = fma_f32x2_rn_noftz(r_54, fma_f32x2_rn_noftz(nd_54, r_54, make_float2(1.0f, 1.0f)), r_54);
        float t0_23 = s_54.x;
        float t1_23 = s_54.y;
        band_2 = band_2 | ((fminf(t0_23, t1_23) == 0.0f) ? 1 : 0);
        float2 _f32x2_mul_a_126 = mul_f32x2_noftz(cvt_f32_55, make_float2(max_noftz(t0_23, 0.0f), max_noftz(t1_23, 0.0f)));
        float2 mul_54;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_54) : "l"(*(const unsigned long long*)&_f32x2_mul_a_126), "l"(*(const unsigned long long*)&cvt_f32_1_54));
        packed_d[0] = __as_u32(__float22bfloat162_rn(make_float2(mul_54.x, mul_54.y)));
        float2 cvt_f32_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[1]));
        float2 cvt_f32_1_55 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[1]));
        float2 _f32x2_mul_b_127 = make_float2(-1.442695f, -1.442695f);
        float2 t_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_55) : "l"(*(const unsigned long long*)&cvt_f32_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_127));
        float e0_55 = approx_exp2(t_55.x);
        float e1_55 = approx_exp2(t_55.y);
        float2 nd_55 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_55, e1_55));
        float d0_55 = -nd_55.x;
        float d1_55 = -nd_55.y;
        float r0_55 = approx_rcp(d0_55);
        float r1_55 = approx_rcp(d1_55);
        float2 r_55 = make_float2(r0_55, r1_55);
        float2 s_55 = fma_f32x2_rn_noftz(r_55, fma_f32x2_rn_noftz(nd_55, r_55, make_float2(1.0f, 1.0f)), r_55);
        float t0_24 = s_55.x;
        float t1_24 = s_55.y;
        band_2 = band_2 | ((fminf(t0_24, t1_24) == 0.0f) ? 2 : 0);
        float2 _f32x2_mul_a_128 = mul_f32x2_noftz(cvt_f32_56, make_float2(max_noftz(t0_24, 0.0f), max_noftz(t1_24, 0.0f)));
        float2 mul_55;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_55) : "l"(*(const unsigned long long*)&_f32x2_mul_a_128), "l"(*(const unsigned long long*)&cvt_f32_1_55));
        packed_d[1] = __as_u32(__float22bfloat162_rn(make_float2(mul_55.x, mul_55.y)));
        float2 cvt_f32_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[2]));
        float2 cvt_f32_1_56 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[2]));
        float2 _f32x2_mul_b_129 = make_float2(-1.442695f, -1.442695f);
        float2 t_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_56) : "l"(*(const unsigned long long*)&cvt_f32_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_129));
        float e0_56 = approx_exp2(t_56.x);
        float e1_56 = approx_exp2(t_56.y);
        float2 nd_56 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_56, e1_56));
        float d0_56 = -nd_56.x;
        float d1_56 = -nd_56.y;
        float r0_56 = approx_rcp(d0_56);
        float r1_56 = approx_rcp(d1_56);
        float2 r_56 = make_float2(r0_56, r1_56);
        float2 s_56 = fma_f32x2_rn_noftz(r_56, fma_f32x2_rn_noftz(nd_56, r_56, make_float2(1.0f, 1.0f)), r_56);
        float t0_25 = s_56.x;
        float t1_25 = s_56.y;
        band_2 = band_2 | ((fminf(t0_25, t1_25) == 0.0f) ? 4 : 0);
        float2 _f32x2_mul_a_130 = mul_f32x2_noftz(cvt_f32_57, make_float2(max_noftz(t0_25, 0.0f), max_noftz(t1_25, 0.0f)));
        float2 mul_56;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_56) : "l"(*(const unsigned long long*)&_f32x2_mul_a_130), "l"(*(const unsigned long long*)&cvt_f32_1_56));
        packed_d[2] = __as_u32(__float22bfloat162_rn(make_float2(mul_56.x, mul_56.y)));
        float2 cvt_f32_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[3]));
        float2 cvt_f32_1_57 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[3]));
        float2 _f32x2_mul_b_131 = make_float2(-1.442695f, -1.442695f);
        float2 t_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_57) : "l"(*(const unsigned long long*)&cvt_f32_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_131));
        float e0_57 = approx_exp2(t_57.x);
        float e1_57 = approx_exp2(t_57.y);
        float2 nd_57 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_57, e1_57));
        float d0_57 = -nd_57.x;
        float d1_57 = -nd_57.y;
        float r0_57 = approx_rcp(d0_57);
        float r1_57 = approx_rcp(d1_57);
        float2 r_57 = make_float2(r0_57, r1_57);
        float2 s_57 = fma_f32x2_rn_noftz(r_57, fma_f32x2_rn_noftz(nd_57, r_57, make_float2(1.0f, 1.0f)), r_57);
        float t0_26 = s_57.x;
        float t1_26 = s_57.y;
        band_2 = band_2 | ((fminf(t0_26, t1_26) == 0.0f) ? 8 : 0);
        float2 _f32x2_mul_a_132 = mul_f32x2_noftz(cvt_f32_58, make_float2(max_noftz(t0_26, 0.0f), max_noftz(t1_26, 0.0f)));
        float2 mul_57;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_57) : "l"(*(const unsigned long long*)&_f32x2_mul_a_132), "l"(*(const unsigned long long*)&cvt_f32_1_57));
        packed_d[3] = __as_u32(__float22bfloat162_rn(make_float2(mul_57.x, mul_57.y)));
        float2 cvt_f32_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[4]));
        float2 cvt_f32_1_58 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[4]));
        float2 _f32x2_mul_b_133 = make_float2(-1.442695f, -1.442695f);
        float2 t_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_58) : "l"(*(const unsigned long long*)&cvt_f32_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_133));
        float e0_58 = approx_exp2(t_58.x);
        float e1_58 = approx_exp2(t_58.y);
        float2 nd_58 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_58, e1_58));
        float d0_58 = -nd_58.x;
        float d1_58 = -nd_58.y;
        float r0_58 = approx_rcp(d0_58);
        float r1_58 = approx_rcp(d1_58);
        float2 r_58 = make_float2(r0_58, r1_58);
        float2 s_58 = fma_f32x2_rn_noftz(r_58, fma_f32x2_rn_noftz(nd_58, r_58, make_float2(1.0f, 1.0f)), r_58);
        float t0_27 = s_58.x;
        float t1_27 = s_58.y;
        band_2 = band_2 | ((fminf(t0_27, t1_27) == 0.0f) ? 16 : 0);
        float2 _f32x2_mul_a_134 = mul_f32x2_noftz(cvt_f32_59, make_float2(max_noftz(t0_27, 0.0f), max_noftz(t1_27, 0.0f)));
        float2 mul_58;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_58) : "l"(*(const unsigned long long*)&_f32x2_mul_a_134), "l"(*(const unsigned long long*)&cvt_f32_1_58));
        packed_d[4] = __as_u32(__float22bfloat162_rn(make_float2(mul_58.x, mul_58.y)));
        float2 cvt_f32_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[5]));
        float2 cvt_f32_1_59 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[5]));
        float2 _f32x2_mul_b_135 = make_float2(-1.442695f, -1.442695f);
        float2 t_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_59) : "l"(*(const unsigned long long*)&cvt_f32_60), "l"(*(const unsigned long long*)&_f32x2_mul_b_135));
        float e0_59 = approx_exp2(t_59.x);
        float e1_59 = approx_exp2(t_59.y);
        float2 nd_59 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_59, e1_59));
        float d0_59 = -nd_59.x;
        float d1_59 = -nd_59.y;
        float r0_59 = approx_rcp(d0_59);
        float r1_59 = approx_rcp(d1_59);
        float2 r_59 = make_float2(r0_59, r1_59);
        float2 s_59 = fma_f32x2_rn_noftz(r_59, fma_f32x2_rn_noftz(nd_59, r_59, make_float2(1.0f, 1.0f)), r_59);
        float t0_28 = s_59.x;
        float t1_28 = s_59.y;
        band_2 = band_2 | ((fminf(t0_28, t1_28) == 0.0f) ? 32 : 0);
        float2 _f32x2_mul_a_136 = mul_f32x2_noftz(cvt_f32_60, make_float2(max_noftz(t0_28, 0.0f), max_noftz(t1_28, 0.0f)));
        float2 mul_59;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_59) : "l"(*(const unsigned long long*)&_f32x2_mul_a_136), "l"(*(const unsigned long long*)&cvt_f32_1_59));
        packed_d[5] = __as_u32(__float22bfloat162_rn(make_float2(mul_59.x, mul_59.y)));
        float2 cvt_f32_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[6]));
        float2 cvt_f32_1_60 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[6]));
        float2 _f32x2_mul_b_137 = make_float2(-1.442695f, -1.442695f);
        float2 t_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_60) : "l"(*(const unsigned long long*)&cvt_f32_61), "l"(*(const unsigned long long*)&_f32x2_mul_b_137));
        float e0_60 = approx_exp2(t_60.x);
        float e1_60 = approx_exp2(t_60.y);
        float2 nd_60 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_60, e1_60));
        float d0_60 = -nd_60.x;
        float d1_60 = -nd_60.y;
        float r0_60 = approx_rcp(d0_60);
        float r1_60 = approx_rcp(d1_60);
        float2 r_60 = make_float2(r0_60, r1_60);
        float2 s_60 = fma_f32x2_rn_noftz(r_60, fma_f32x2_rn_noftz(nd_60, r_60, make_float2(1.0f, 1.0f)), r_60);
        float t0_29 = s_60.x;
        float t1_29 = s_60.y;
        band_2 = band_2 | ((fminf(t0_29, t1_29) == 0.0f) ? 64 : 0);
        float2 _f32x2_mul_a_138 = mul_f32x2_noftz(cvt_f32_61, make_float2(max_noftz(t0_29, 0.0f), max_noftz(t1_29, 0.0f)));
        float2 mul_60;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_60) : "l"(*(const unsigned long long*)&_f32x2_mul_a_138), "l"(*(const unsigned long long*)&cvt_f32_1_60));
        packed_d[6] = __as_u32(__float22bfloat162_rn(make_float2(mul_60.x, mul_60.y)));
        float2 cvt_f32_62 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_2[7]));
        float2 cvt_f32_1_61 = cake_bf16x2_to_f32x2(__as_bf16x2(vec_load_1_1[7]));
        float2 _f32x2_mul_b_139 = make_float2(-1.442695f, -1.442695f);
        float2 t_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&t_61) : "l"(*(const unsigned long long*)&cvt_f32_62), "l"(*(const unsigned long long*)&_f32x2_mul_b_139));
        float e0_61 = approx_exp2(t_61.x);
        float e1_61 = approx_exp2(t_61.y);
        float2 nd_61 = sub_f32x2_noftz(make_float2(-1.0f, -1.0f), make_float2(e0_61, e1_61));
        float d0_61 = -nd_61.x;
        float d1_61 = -nd_61.y;
        float r0_61 = approx_rcp(d0_61);
        float r1_61 = approx_rcp(d1_61);
        float2 r_61 = make_float2(r0_61, r1_61);
        float2 s_61 = fma_f32x2_rn_noftz(r_61, fma_f32x2_rn_noftz(nd_61, r_61, make_float2(1.0f, 1.0f)), r_61);
        float t0_30 = s_61.x;
        float t1_30 = s_61.y;
        band_2 = band_2 | ((fminf(t0_30, t1_30) == 0.0f) ? 128 : 0);
        float2 _f32x2_mul_a_140 = mul_f32x2_noftz(cvt_f32_62, make_float2(max_noftz(t0_30, 0.0f), max_noftz(t1_30, 0.0f)));
        float2 mul_61;
        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&mul_61) : "l"(*(const unsigned long long*)&_f32x2_mul_a_140), "l"(*(const unsigned long long*)&cvt_f32_1_61));
        packed_d[7] = __as_u32(__float22bfloat162_rn(make_float2(mul_61.x, mul_61.y)));
        if (band_2 != 0) {
            unsigned int popc_2 = (unsigned int)__popc(band_2);
            #pragma unroll 1
            for (int _j_3 = 0; _j_3 < popc_2; _j_3++) {
                int _find_first_set_141 = __ffs(band_2);
                int j_3 = _find_first_set_141 - 1;
                band_2 = band_2 & band_2 - 1;
                unsigned int pair_3 = band_pair(gated_input, gate_index_5 + 2 * j_3, K);
                packed_d[0] = ((j_3 == 0) ? pair_3 : packed_d[0]);
                packed_d[1] = ((j_3 == 1) ? pair_3 : packed_d[1]);
                packed_d[2] = ((j_3 == 2) ? pair_3 : packed_d[2]);
                packed_d[3] = ((j_3 == 3) ? pair_3 : packed_d[3]);
                packed_d[4] = ((j_3 == 4) ? pair_3 : packed_d[4]);
                packed_d[5] = ((j_3 == 5) ? pair_3 : packed_d[5]);
                packed_d[6] = ((j_3 == 6) ? pair_3 : packed_d[6]);
                packed_d[7] = ((j_3 == 7) ? pair_3 : packed_d[7]);
            }
        }
    }
    unsigned int swizzle = (unsigned int)(row >> 4 & 6 | row & 1);
    unsigned int row_base = tile_addr + (unsigned int)(row * 256);
    unsigned int chunk = (unsigned int)((tid & 7) * 2);
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk ^ swizzle) << 4)), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + ((chunk + 1 ^ swizzle) << 4)), "r"(packed[4]), "r"(packed[5]), "r"(packed[6]), "r"(packed[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk ^ (swizzle ^ 2)) << 4)), "r"(packed_b[0]), "r"(packed_b[1]), "r"(packed_b[2]), "r"(packed_b[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 8192 + ((chunk + 1 ^ (swizzle ^ 2)) << 4)), "r"(packed_b[4]), "r"(packed_b[5]), "r"(packed_b[6]), "r"(packed_b[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 16384 + ((chunk ^ (swizzle ^ 4)) << 4)), "r"(packed_c[0]), "r"(packed_c[1]), "r"(packed_c[2]), "r"(packed_c[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 16384 + ((chunk + 1 ^ (swizzle ^ 4)) << 4)), "r"(packed_c[4]), "r"(packed_c[5]), "r"(packed_c[6]), "r"(packed_c[7]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 24576 + ((chunk ^ (swizzle ^ 6)) << 4)), "r"(packed_d[0]), "r"(packed_d[1]), "r"(packed_d[2]), "r"(packed_d[3]) : "memory");
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_base + 24576 + ((chunk + 1 ^ (swizzle ^ 6)) << 4)), "r"(packed_d[4]), "r"(packed_d[5]), "r"(packed_d[6]), "r"(packed_d[7]) : "memory");
    __nv_bfloat162 amax = __as_bf16x2(0);
    __nv_bfloat162 _hxa_143 = amax;
    __nv_bfloat162 _hxb_143 = __as_bf16x2(packed[0]);
    unsigned int _hxr_143;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_143) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_143)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_143)));
    __nv_bfloat162 _max_142 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_143);
    amax = _max_142;
    __nv_bfloat162 _hxa_145 = amax;
    __nv_bfloat162 _hxb_145 = __as_bf16x2(packed[1]);
    unsigned int _hxr_145;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_145) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_145)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_145)));
    __nv_bfloat162 _max_144 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_145);
    amax = _max_144;
    __nv_bfloat162 _hxa_147 = amax;
    __nv_bfloat162 _hxb_147 = __as_bf16x2(packed[2]);
    unsigned int _hxr_147;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_147) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_147)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_147)));
    __nv_bfloat162 _max_146 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_147);
    amax = _max_146;
    __nv_bfloat162 _hxa_149 = amax;
    __nv_bfloat162 _hxb_149 = __as_bf16x2(packed[3]);
    unsigned int _hxr_149;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_149) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_149)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_149)));
    __nv_bfloat162 _max_148 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_149);
    amax = _max_148;
    __nv_bfloat162 _hxa_151 = amax;
    __nv_bfloat162 _hxb_151 = __as_bf16x2(packed[4]);
    unsigned int _hxr_151;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_151) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_151)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_151)));
    __nv_bfloat162 _max_150 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_151);
    amax = _max_150;
    __nv_bfloat162 _hxa_153 = amax;
    __nv_bfloat162 _hxb_153 = __as_bf16x2(packed[5]);
    unsigned int _hxr_153;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_153) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_153)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_153)));
    __nv_bfloat162 _max_152 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_153);
    amax = _max_152;
    __nv_bfloat162 _hxa_155 = amax;
    __nv_bfloat162 _hxb_155 = __as_bf16x2(packed[6]);
    unsigned int _hxr_155;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_155) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_155)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_155)));
    __nv_bfloat162 _max_154 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_155);
    amax = _max_154;
    __nv_bfloat162 _hxa_157 = amax;
    __nv_bfloat162 _hxb_157 = __as_bf16x2(packed[7]);
    unsigned int _hxr_157;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_157) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_157)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_157)));
    __nv_bfloat162 _max_156 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_157);
    amax = _max_156;
    unsigned int mag = __as_u32(amax) & 2147450879;
    unsigned int peer = __shfl_xor_sync(0xFFFFFFFF, mag, 1);
    __nv_bfloat162 both = __hmax2_nan(__as_bf16x2(mag), __as_bf16x2(peer));
    unsigned int folded = __as_u32(__hmax2_nan(both, __as_bf16x2(__as_u32(both) >> 16)));
    unsigned int amax_bits = (folded & 65535) << 16;
    int scale_i32 = (int)(amax_bits + 2031616 >> 23) - 8;
    if (scale_i32 < 0) {
        scale_i32 = 0;
    }
    if ((amax_bits & 2139095040) == 2139095040) {
        scale_i32 = 255;
    }
    unsigned int scale = (unsigned int)scale_i32;
    unsigned int inv = 254 - scale << 7;
    if (scale == 255) {
        inv = 32704;
    }
    unsigned int inv2 = inv | inv << 16;
    int scale_col = bx * 4 + blk;
    int row_scale_index = ((grow >> 7) * (K >> 7) + (scale_col >> 2)) * 512 + (grow & 31) * 16 + (grow >> 5 & 3) * 4 + (scale_col & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index) + (0)) = (unsigned char)(scale);
    }
    unsigned int row_words[4];
    uint32_t bf16x2_mul;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul) : "r"(packed[0]), "r"(inv2));
    uint16_t e4m3x2_bf16x2;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2) : "r"(bf16x2_mul));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1) : "r"(packed[1]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_159;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "r"(bf16x2_mul_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_159) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_158;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_158) : "h"(e4m3x2_bf16x2), "h"(_cvt_satfinite_e4m3x2_bf16x2_159));
    row_words[0] = _mov_b32_pack_u16x2_158;
    uint32_t bf16x2_mul_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_0) : "r"(packed[2]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_0;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_0) : "r"(bf16x2_mul_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_0) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_0;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_0) : "r"(packed[3]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_161;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_161) : "r"(bf16x2_mul_1_0));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_0);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_161) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_160;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_160) : "h"(e4m3x2_bf16x2_0), "h"(_cvt_satfinite_e4m3x2_bf16x2_161));
    row_words[1] = _mov_b32_pack_u16x2_160;
    uint32_t bf16x2_mul_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_2) : "r"(packed[4]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_1;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_1) : "r"(bf16x2_mul_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_1) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_1;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_1) : "r"(packed[5]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_163;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "r"(bf16x2_mul_1_1));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_1);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_163) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_162;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_162) : "h"(e4m3x2_bf16x2_1), "h"(_cvt_satfinite_e4m3x2_bf16x2_163));
    row_words[2] = _mov_b32_pack_u16x2_162;
    uint32_t bf16x2_mul_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_3) : "r"(packed[6]), "r"(inv2));
    uint16_t e4m3x2_bf16x2_2;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_2) : "r"(bf16x2_mul_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_2) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_2;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_2) : "r"(packed[7]), "r"(inv2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_165;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "r"(bf16x2_mul_1_2));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_2);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_165) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_164;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_164) : "h"(e4m3x2_bf16x2_2), "h"(_cvt_satfinite_e4m3x2_bf16x2_165));
    row_words[3] = _mov_b32_pack_u16x2_164;
    reinterpret_cast<int4*>(row_output + (grow * K + col))[0] = reinterpret_cast<int4*>(row_words + 0)[0];
    __nv_bfloat162 amax_0 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_167 = amax_0;
    __nv_bfloat162 _hxb_167 = __as_bf16x2(packed_b[0]);
    unsigned int _hxr_167;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_167) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_167)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_167)));
    __nv_bfloat162 _max_166 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_167);
    amax_0 = _max_166;
    __nv_bfloat162 _hxa_169 = amax_0;
    __nv_bfloat162 _hxb_169 = __as_bf16x2(packed_b[1]);
    unsigned int _hxr_169;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_169) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_169)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_169)));
    __nv_bfloat162 _max_168 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_169);
    amax_0 = _max_168;
    __nv_bfloat162 _hxa_171 = amax_0;
    __nv_bfloat162 _hxb_171 = __as_bf16x2(packed_b[2]);
    unsigned int _hxr_171;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_171) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_171)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_171)));
    __nv_bfloat162 _max_170 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_171);
    amax_0 = _max_170;
    __nv_bfloat162 _hxa_173 = amax_0;
    __nv_bfloat162 _hxb_173 = __as_bf16x2(packed_b[3]);
    unsigned int _hxr_173;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_173) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_173)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_173)));
    __nv_bfloat162 _max_172 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_173);
    amax_0 = _max_172;
    __nv_bfloat162 _hxa_175 = amax_0;
    __nv_bfloat162 _hxb_175 = __as_bf16x2(packed_b[4]);
    unsigned int _hxr_175;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_175) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_175)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_175)));
    __nv_bfloat162 _max_174 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_175);
    amax_0 = _max_174;
    __nv_bfloat162 _hxa_177 = amax_0;
    __nv_bfloat162 _hxb_177 = __as_bf16x2(packed_b[5]);
    unsigned int _hxr_177;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_177) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_177)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_177)));
    __nv_bfloat162 _max_176 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_177);
    amax_0 = _max_176;
    __nv_bfloat162 _hxa_179 = amax_0;
    __nv_bfloat162 _hxb_179 = __as_bf16x2(packed_b[6]);
    unsigned int _hxr_179;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_179) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_179)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_179)));
    __nv_bfloat162 _max_178 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_179);
    amax_0 = _max_178;
    __nv_bfloat162 _hxa_181 = amax_0;
    __nv_bfloat162 _hxb_181 = __as_bf16x2(packed_b[7]);
    unsigned int _hxr_181;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_181) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_181)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_181)));
    __nv_bfloat162 _max_180 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_181);
    amax_0 = _max_180;
    unsigned int mag_0 = __as_u32(amax_0) & 2147450879;
    unsigned int peer_0 = __shfl_xor_sync(0xFFFFFFFF, mag_0, 1);
    __nv_bfloat162 both_0 = __hmax2_nan(__as_bf16x2(mag_0), __as_bf16x2(peer_0));
    unsigned int folded_0 = __as_u32(__hmax2_nan(both_0, __as_bf16x2(__as_u32(both_0) >> 16)));
    unsigned int amax_bits_0 = (folded_0 & 65535) << 16;
    int scale_i32_0 = (int)(amax_bits_0 + 2031616 >> 23) - 8;
    if (scale_i32_0 < 0) {
        scale_i32_0 = 0;
    }
    if ((amax_bits_0 & 2139095040) == 2139095040) {
        scale_i32_0 = 255;
    }
    unsigned int scale_0 = (unsigned int)scale_i32_0;
    unsigned int inv_0 = 254 - scale_0 << 7;
    if (scale_0 == 255) {
        inv_0 = 32704;
    }
    unsigned int inv2_0 = inv_0 | inv_0 << 16;
    int scale_col_0 = bx * 4 + blk;
    int row_scale_index_0 = ((grow_b >> 7) * (K >> 7) + (scale_col_0 >> 2)) * 512 + (grow_b & 31) * 16 + (grow_b >> 5 & 3) * 4 + (scale_col_0 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_0) + (0)) = (unsigned char)(scale_0);
    }
    unsigned int row_words_0[4];
    uint32_t bf16x2_mul_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_4) : "r"(packed_b[0]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_3;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_3) : "r"(bf16x2_mul_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_3) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_3;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_3) : "r"(packed_b[1]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_183;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_183) : "r"(bf16x2_mul_1_3));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_3);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_183) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_182;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_182) : "h"(e4m3x2_bf16x2_3), "h"(_cvt_satfinite_e4m3x2_bf16x2_183));
    row_words_0[0] = _mov_b32_pack_u16x2_182;
    uint32_t bf16x2_mul_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_5) : "r"(packed_b[2]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_4;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_4) : "r"(bf16x2_mul_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_4) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_4;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_4) : "r"(packed_b[3]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_185;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_185) : "r"(bf16x2_mul_1_4));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_4);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_185) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_184;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_184) : "h"(e4m3x2_bf16x2_4), "h"(_cvt_satfinite_e4m3x2_bf16x2_185));
    row_words_0[1] = _mov_b32_pack_u16x2_184;
    uint32_t bf16x2_mul_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_6) : "r"(packed_b[4]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_5;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_5) : "r"(bf16x2_mul_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_5) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_5;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_5) : "r"(packed_b[5]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_187;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_187) : "r"(bf16x2_mul_1_5));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_5);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_187) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_186;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_186) : "h"(e4m3x2_bf16x2_5), "h"(_cvt_satfinite_e4m3x2_bf16x2_187));
    row_words_0[2] = _mov_b32_pack_u16x2_186;
    uint32_t bf16x2_mul_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_7) : "r"(packed_b[6]), "r"(inv2_0));
    uint16_t e4m3x2_bf16x2_6;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_6) : "r"(bf16x2_mul_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_6) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_6;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_6) : "r"(packed_b[7]), "r"(inv2_0));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_189;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "r"(bf16x2_mul_1_6));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_6);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_189) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_188;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_188) : "h"(e4m3x2_bf16x2_6), "h"(_cvt_satfinite_e4m3x2_bf16x2_189));
    row_words_0[3] = _mov_b32_pack_u16x2_188;
    reinterpret_cast<int4*>(row_output + (grow_b * K + col))[0] = reinterpret_cast<int4*>(row_words_0 + 0)[0];
    __nv_bfloat162 amax_16 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_191 = amax_16;
    __nv_bfloat162 _hxb_191 = __as_bf16x2(packed_c[0]);
    unsigned int _hxr_191;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_191) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_191)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_191)));
    __nv_bfloat162 _max_190 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_191);
    amax_16 = _max_190;
    __nv_bfloat162 _hxa_193 = amax_16;
    __nv_bfloat162 _hxb_193 = __as_bf16x2(packed_c[1]);
    unsigned int _hxr_193;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_193) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_193)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_193)));
    __nv_bfloat162 _max_192 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_193);
    amax_16 = _max_192;
    __nv_bfloat162 _hxa_195 = amax_16;
    __nv_bfloat162 _hxb_195 = __as_bf16x2(packed_c[2]);
    unsigned int _hxr_195;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_195) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_195)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_195)));
    __nv_bfloat162 _max_194 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_195);
    amax_16 = _max_194;
    __nv_bfloat162 _hxa_197 = amax_16;
    __nv_bfloat162 _hxb_197 = __as_bf16x2(packed_c[3]);
    unsigned int _hxr_197;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_197) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_197)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_197)));
    __nv_bfloat162 _max_196 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_197);
    amax_16 = _max_196;
    __nv_bfloat162 _hxa_199 = amax_16;
    __nv_bfloat162 _hxb_199 = __as_bf16x2(packed_c[4]);
    unsigned int _hxr_199;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_199) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_199)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_199)));
    __nv_bfloat162 _max_198 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_199);
    amax_16 = _max_198;
    __nv_bfloat162 _hxa_201 = amax_16;
    __nv_bfloat162 _hxb_201 = __as_bf16x2(packed_c[5]);
    unsigned int _hxr_201;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_201) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_201)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_201)));
    __nv_bfloat162 _max_200 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_201);
    amax_16 = _max_200;
    __nv_bfloat162 _hxa_203 = amax_16;
    __nv_bfloat162 _hxb_203 = __as_bf16x2(packed_c[6]);
    unsigned int _hxr_203;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_203) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_203)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_203)));
    __nv_bfloat162 _max_202 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_203);
    amax_16 = _max_202;
    __nv_bfloat162 _hxa_205 = amax_16;
    __nv_bfloat162 _hxb_205 = __as_bf16x2(packed_c[7]);
    unsigned int _hxr_205;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_205) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_205)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_205)));
    __nv_bfloat162 _max_204 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_205);
    amax_16 = _max_204;
    unsigned int mag_17 = __as_u32(amax_16) & 2147450879;
    unsigned int peer_18 = __shfl_xor_sync(0xFFFFFFFF, mag_17, 1);
    __nv_bfloat162 both_19 = __hmax2_nan(__as_bf16x2(mag_17), __as_bf16x2(peer_18));
    unsigned int folded_20 = __as_u32(__hmax2_nan(both_19, __as_bf16x2(__as_u32(both_19) >> 16)));
    unsigned int amax_bits_1 = (folded_20 & 65535) << 16;
    int scale_i32_1 = (int)(amax_bits_1 + 2031616 >> 23) - 8;
    if (scale_i32_1 < 0) {
        scale_i32_1 = 0;
    }
    if ((amax_bits_1 & 2139095040) == 2139095040) {
        scale_i32_1 = 255;
    }
    unsigned int scale_1 = (unsigned int)scale_i32_1;
    unsigned int inv_1 = 254 - scale_1 << 7;
    if (scale_1 == 255) {
        inv_1 = 32704;
    }
    unsigned int inv2_1 = inv_1 | inv_1 << 16;
    int scale_col_26 = bx * 4 + blk;
    int row_scale_index_27 = ((grow_c >> 7) * (K >> 7) + (scale_col_26 >> 2)) * 512 + (grow_c & 31) * 16 + (grow_c >> 5 & 3) * 4 + (scale_col_26 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_27) + (0)) = (unsigned char)(scale_1);
    }
    unsigned int row_words_28[4];
    uint32_t bf16x2_mul_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_8) : "r"(packed_c[0]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_7;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_7) : "r"(bf16x2_mul_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_7) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_7;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_7) : "r"(packed_c[1]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_207;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "r"(bf16x2_mul_1_7));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_7);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_207) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_206;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_206) : "h"(e4m3x2_bf16x2_7), "h"(_cvt_satfinite_e4m3x2_bf16x2_207));
    row_words_28[0] = _mov_b32_pack_u16x2_206;
    uint32_t bf16x2_mul_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_9) : "r"(packed_c[2]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_8;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_8) : "r"(bf16x2_mul_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_8) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_8;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_8) : "r"(packed_c[3]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_209;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_209) : "r"(bf16x2_mul_1_8));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_8);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_209) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_208;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_208) : "h"(e4m3x2_bf16x2_8), "h"(_cvt_satfinite_e4m3x2_bf16x2_209));
    row_words_28[1] = _mov_b32_pack_u16x2_208;
    uint32_t bf16x2_mul_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_10) : "r"(packed_c[4]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_9;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_9) : "r"(bf16x2_mul_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_9) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_9;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_9) : "r"(packed_c[5]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_211;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_211) : "r"(bf16x2_mul_1_9));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_9);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_211) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_210;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_210) : "h"(e4m3x2_bf16x2_9), "h"(_cvt_satfinite_e4m3x2_bf16x2_211));
    row_words_28[2] = _mov_b32_pack_u16x2_210;
    uint32_t bf16x2_mul_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_11) : "r"(packed_c[6]), "r"(inv2_1));
    uint16_t e4m3x2_bf16x2_10;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_10) : "r"(bf16x2_mul_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_10) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_10;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_10) : "r"(packed_c[7]), "r"(inv2_1));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_213;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "r"(bf16x2_mul_1_10));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_10);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_213) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_212;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_212) : "h"(e4m3x2_bf16x2_10), "h"(_cvt_satfinite_e4m3x2_bf16x2_213));
    row_words_28[3] = _mov_b32_pack_u16x2_212;
    reinterpret_cast<int4*>(row_output + (grow_c * K + col))[0] = reinterpret_cast<int4*>(row_words_28 + 0)[0];
    __nv_bfloat162 amax_29 = __as_bf16x2(0);
    __nv_bfloat162 _hxa_215 = amax_29;
    __nv_bfloat162 _hxb_215 = __as_bf16x2(packed_d[0]);
    unsigned int _hxr_215;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_215) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_215)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_215)));
    __nv_bfloat162 _max_214 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_215);
    amax_29 = _max_214;
    __nv_bfloat162 _hxa_217 = amax_29;
    __nv_bfloat162 _hxb_217 = __as_bf16x2(packed_d[1]);
    unsigned int _hxr_217;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_217) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_217)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_217)));
    __nv_bfloat162 _max_216 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_217);
    amax_29 = _max_216;
    __nv_bfloat162 _hxa_219 = amax_29;
    __nv_bfloat162 _hxb_219 = __as_bf16x2(packed_d[2]);
    unsigned int _hxr_219;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_219) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_219)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_219)));
    __nv_bfloat162 _max_218 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_219);
    amax_29 = _max_218;
    __nv_bfloat162 _hxa_221 = amax_29;
    __nv_bfloat162 _hxb_221 = __as_bf16x2(packed_d[3]);
    unsigned int _hxr_221;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_221) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_221)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_221)));
    __nv_bfloat162 _max_220 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_221);
    amax_29 = _max_220;
    __nv_bfloat162 _hxa_223 = amax_29;
    __nv_bfloat162 _hxb_223 = __as_bf16x2(packed_d[4]);
    unsigned int _hxr_223;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_223) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_223)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_223)));
    __nv_bfloat162 _max_222 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_223);
    amax_29 = _max_222;
    __nv_bfloat162 _hxa_225 = amax_29;
    __nv_bfloat162 _hxb_225 = __as_bf16x2(packed_d[5]);
    unsigned int _hxr_225;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_225) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_225)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_225)));
    __nv_bfloat162 _max_224 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_225);
    amax_29 = _max_224;
    __nv_bfloat162 _hxa_227 = amax_29;
    __nv_bfloat162 _hxb_227 = __as_bf16x2(packed_d[6]);
    unsigned int _hxr_227;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_227) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_227)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_227)));
    __nv_bfloat162 _max_226 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_227);
    amax_29 = _max_226;
    __nv_bfloat162 _hxa_229 = amax_29;
    __nv_bfloat162 _hxb_229 = __as_bf16x2(packed_d[7]);
    unsigned int _hxr_229;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_229) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_229)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_229)));
    __nv_bfloat162 _max_228 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_229);
    amax_29 = _max_228;
    unsigned int mag_30 = __as_u32(amax_29) & 2147450879;
    unsigned int peer_31 = __shfl_xor_sync(0xFFFFFFFF, mag_30, 1);
    __nv_bfloat162 both_32 = __hmax2_nan(__as_bf16x2(mag_30), __as_bf16x2(peer_31));
    unsigned int folded_33 = __as_u32(__hmax2_nan(both_32, __as_bf16x2(__as_u32(both_32) >> 16)));
    unsigned int amax_bits_2 = (folded_33 & 65535) << 16;
    int scale_i32_2 = (int)(amax_bits_2 + 2031616 >> 23) - 8;
    if (scale_i32_2 < 0) {
        scale_i32_2 = 0;
    }
    if ((amax_bits_2 & 2139095040) == 2139095040) {
        scale_i32_2 = 255;
    }
    unsigned int scale_3 = (unsigned int)scale_i32_2;
    unsigned int inv_2 = 254 - scale_3 << 7;
    if (scale_3 == 255) {
        inv_2 = 32704;
    }
    unsigned int inv2_2 = inv_2 | inv_2 << 16;
    int scale_col_39 = bx * 4 + blk;
    int row_scale_index_40 = ((grow_d >> 7) * (K >> 7) + (scale_col_39 >> 2)) * 512 + (grow_d & 31) * 16 + (grow_d >> 5 & 3) * 4 + (scale_col_39 & 3);
    if (half == 0) {
        *(reinterpret_cast<unsigned char*>(row_scales + row_scale_index_40) + (0)) = (unsigned char)(scale_3);
    }
    unsigned int row_words_41[4];
    uint32_t bf16x2_mul_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_12) : "r"(packed_d[0]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_11;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_11) : "r"(bf16x2_mul_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_11) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_11;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_11) : "r"(packed_d[1]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_231;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_231) : "r"(bf16x2_mul_1_11));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_11);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_231) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_230;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_230) : "h"(e4m3x2_bf16x2_11), "h"(_cvt_satfinite_e4m3x2_bf16x2_231));
    row_words_41[0] = _mov_b32_pack_u16x2_230;
    uint32_t bf16x2_mul_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_13) : "r"(packed_d[2]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_12;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_12) : "r"(bf16x2_mul_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_12) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_12;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_12) : "r"(packed_d[3]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_233;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_233) : "r"(bf16x2_mul_1_12));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_12);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_233) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_232;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_232) : "h"(e4m3x2_bf16x2_12), "h"(_cvt_satfinite_e4m3x2_bf16x2_233));
    row_words_41[1] = _mov_b32_pack_u16x2_232;
    uint32_t bf16x2_mul_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_14) : "r"(packed_d[4]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_13;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_13) : "r"(bf16x2_mul_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_13) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_13;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_13) : "r"(packed_d[5]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_235;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "r"(bf16x2_mul_1_13));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_13);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_235) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_234;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_234) : "h"(e4m3x2_bf16x2_13), "h"(_cvt_satfinite_e4m3x2_bf16x2_235));
    row_words_41[2] = _mov_b32_pack_u16x2_234;
    uint32_t bf16x2_mul_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_15) : "r"(packed_d[6]), "r"(inv2_2));
    uint16_t e4m3x2_bf16x2_14;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_14) : "r"(bf16x2_mul_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_14) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_14;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_14) : "r"(packed_d[7]), "r"(inv2_2));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_237;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "r"(bf16x2_mul_1_14));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_14);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_237) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_236;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_236) : "h"(e4m3x2_bf16x2_14), "h"(_cvt_satfinite_e4m3x2_bf16x2_237));
    row_words_41[3] = _mov_b32_pack_u16x2_236;
    reinterpret_cast<int4*>(row_output + (grow_d * K + col))[0] = reinterpret_cast<int4*>(row_words_41 + 0)[0];
    __syncthreads();
    int lane_1 = tid & 31;
    int warp_1 = tid >> 5;
    int k8 = lane_1 & 7;
    int srow = (k8 >> 1) * 32 + (lane_1 >> 3) * 2 + (k8 & 1);
    unsigned int sswizzle = (unsigned int)(srow >> 4 & 6 | srow & 1);
    unsigned int cols[32];
    unsigned int strip_chunk0_1_0 = (unsigned int)(warp_1 * 2);
    unsigned int strip_src0_1_0 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_0 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[0]), "=r"(cols[1]), "=r"(cols[2]), "=r"(cols[3])
        : "r"(strip_src0_1_0)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[4]), "=r"(cols[5]), "=r"(cols[6]), "=r"(cols[7])
        : "r"(strip_src0_1_0 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[8]), "=r"(cols[9]), "=r"(cols[10]), "=r"(cols[11])
        : "r"(strip_src0_1_0 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[12]), "=r"(cols[13]), "=r"(cols[14]), "=r"(cols[15])
        : "r"(strip_src0_1_0 + 6144)
        : "memory");
    unsigned int strip_chunk0_1_1 = (unsigned int)(warp_1 * 2 + 1);
    unsigned int strip_src0_1_1 = tile_addr + (unsigned int)(srow * 256) + ((strip_chunk0_1_1 ^ sswizzle) << 4);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[16]), "=r"(cols[17]), "=r"(cols[18]), "=r"(cols[19])
        : "r"(strip_src0_1_1)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[20]), "=r"(cols[21]), "=r"(cols[22]), "=r"(cols[23])
        : "r"(strip_src0_1_1 + 2048)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[24]), "=r"(cols[25]), "=r"(cols[26]), "=r"(cols[27])
        : "r"(strip_src0_1_1 + 4096)
        : "memory");
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(cols[28]), "=r"(cols[29]), "=r"(cols[30]), "=r"(cols[31])
        : "r"(strip_src0_1_1 + 6144)
        : "memory");
    __nv_bfloat162 amax_a = __as_bf16x2(0);
    __nv_bfloat162 amax_b = __as_bf16x2(0);
    __nv_bfloat162 _hxa_239 = amax_a;
    __nv_bfloat162 _hxb_239 = __as_bf16x2(cols[0]);
    unsigned int _hxr_239;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_239) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_239)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_239)));
    __nv_bfloat162 _max_238 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_239);
    amax_a = _max_238;
    __nv_bfloat162 _hxa_241 = amax_b;
    __nv_bfloat162 _hxb_241 = __as_bf16x2(cols[16]);
    unsigned int _hxr_241;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_241) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_241)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_241)));
    __nv_bfloat162 _max_240 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_241);
    amax_b = _max_240;
    __nv_bfloat162 _hxa_243 = amax_a;
    __nv_bfloat162 _hxb_243 = __as_bf16x2(cols[1]);
    unsigned int _hxr_243;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_243) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_243)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_243)));
    __nv_bfloat162 _max_242 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_243);
    amax_a = _max_242;
    __nv_bfloat162 _hxa_245 = amax_b;
    __nv_bfloat162 _hxb_245 = __as_bf16x2(cols[17]);
    unsigned int _hxr_245;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_245) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_245)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_245)));
    __nv_bfloat162 _max_244 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_245);
    amax_b = _max_244;
    __nv_bfloat162 _hxa_247 = amax_a;
    __nv_bfloat162 _hxb_247 = __as_bf16x2(cols[2]);
    unsigned int _hxr_247;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_247) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_247)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_247)));
    __nv_bfloat162 _max_246 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_247);
    amax_a = _max_246;
    __nv_bfloat162 _hxa_249 = amax_b;
    __nv_bfloat162 _hxb_249 = __as_bf16x2(cols[18]);
    unsigned int _hxr_249;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_249) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_249)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_249)));
    __nv_bfloat162 _max_248 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_249);
    amax_b = _max_248;
    __nv_bfloat162 _hxa_251 = amax_a;
    __nv_bfloat162 _hxb_251 = __as_bf16x2(cols[3]);
    unsigned int _hxr_251;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_251) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_251)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_251)));
    __nv_bfloat162 _max_250 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_251);
    amax_a = _max_250;
    __nv_bfloat162 _hxa_253 = amax_b;
    __nv_bfloat162 _hxb_253 = __as_bf16x2(cols[19]);
    unsigned int _hxr_253;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_253) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_253)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_253)));
    __nv_bfloat162 _max_252 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_253);
    amax_b = _max_252;
    __nv_bfloat162 _hxa_255 = amax_a;
    __nv_bfloat162 _hxb_255 = __as_bf16x2(cols[4]);
    unsigned int _hxr_255;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_255) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_255)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_255)));
    __nv_bfloat162 _max_254 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_255);
    amax_a = _max_254;
    __nv_bfloat162 _hxa_257 = amax_b;
    __nv_bfloat162 _hxb_257 = __as_bf16x2(cols[20]);
    unsigned int _hxr_257;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_257) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_257)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_257)));
    __nv_bfloat162 _max_256 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_257);
    amax_b = _max_256;
    __nv_bfloat162 _hxa_259 = amax_a;
    __nv_bfloat162 _hxb_259 = __as_bf16x2(cols[5]);
    unsigned int _hxr_259;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_259) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_259)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_259)));
    __nv_bfloat162 _max_258 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_259);
    amax_a = _max_258;
    __nv_bfloat162 _hxa_261 = amax_b;
    __nv_bfloat162 _hxb_261 = __as_bf16x2(cols[21]);
    unsigned int _hxr_261;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_261) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_261)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_261)));
    __nv_bfloat162 _max_260 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_261);
    amax_b = _max_260;
    __nv_bfloat162 _hxa_263 = amax_a;
    __nv_bfloat162 _hxb_263 = __as_bf16x2(cols[6]);
    unsigned int _hxr_263;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_263) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_263)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_263)));
    __nv_bfloat162 _max_262 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_263);
    amax_a = _max_262;
    __nv_bfloat162 _hxa_265 = amax_b;
    __nv_bfloat162 _hxb_265 = __as_bf16x2(cols[22]);
    unsigned int _hxr_265;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_265) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_265)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_265)));
    __nv_bfloat162 _max_264 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_265);
    amax_b = _max_264;
    __nv_bfloat162 _hxa_267 = amax_a;
    __nv_bfloat162 _hxb_267 = __as_bf16x2(cols[7]);
    unsigned int _hxr_267;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_267) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_267)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_267)));
    __nv_bfloat162 _max_266 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_267);
    amax_a = _max_266;
    __nv_bfloat162 _hxa_269 = amax_b;
    __nv_bfloat162 _hxb_269 = __as_bf16x2(cols[23]);
    unsigned int _hxr_269;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_269) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_269)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_269)));
    __nv_bfloat162 _max_268 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_269);
    amax_b = _max_268;
    __nv_bfloat162 _hxa_271 = amax_a;
    __nv_bfloat162 _hxb_271 = __as_bf16x2(cols[8]);
    unsigned int _hxr_271;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_271) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_271)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_271)));
    __nv_bfloat162 _max_270 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_271);
    amax_a = _max_270;
    __nv_bfloat162 _hxa_273 = amax_b;
    __nv_bfloat162 _hxb_273 = __as_bf16x2(cols[24]);
    unsigned int _hxr_273;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_273) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_273)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_273)));
    __nv_bfloat162 _max_272 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_273);
    amax_b = _max_272;
    __nv_bfloat162 _hxa_275 = amax_a;
    __nv_bfloat162 _hxb_275 = __as_bf16x2(cols[9]);
    unsigned int _hxr_275;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_275) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_275)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_275)));
    __nv_bfloat162 _max_274 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_275);
    amax_a = _max_274;
    __nv_bfloat162 _hxa_277 = amax_b;
    __nv_bfloat162 _hxb_277 = __as_bf16x2(cols[25]);
    unsigned int _hxr_277;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_277) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_277)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_277)));
    __nv_bfloat162 _max_276 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_277);
    amax_b = _max_276;
    __nv_bfloat162 _hxa_279 = amax_a;
    __nv_bfloat162 _hxb_279 = __as_bf16x2(cols[10]);
    unsigned int _hxr_279;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_279) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_279)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_279)));
    __nv_bfloat162 _max_278 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_279);
    amax_a = _max_278;
    __nv_bfloat162 _hxa_281 = amax_b;
    __nv_bfloat162 _hxb_281 = __as_bf16x2(cols[26]);
    unsigned int _hxr_281;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_281) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_281)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_281)));
    __nv_bfloat162 _max_280 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_281);
    amax_b = _max_280;
    __nv_bfloat162 _hxa_283 = amax_a;
    __nv_bfloat162 _hxb_283 = __as_bf16x2(cols[11]);
    unsigned int _hxr_283;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_283) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_283)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_283)));
    __nv_bfloat162 _max_282 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_283);
    amax_a = _max_282;
    __nv_bfloat162 _hxa_285 = amax_b;
    __nv_bfloat162 _hxb_285 = __as_bf16x2(cols[27]);
    unsigned int _hxr_285;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_285) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_285)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_285)));
    __nv_bfloat162 _max_284 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_285);
    amax_b = _max_284;
    __nv_bfloat162 _hxa_287 = amax_a;
    __nv_bfloat162 _hxb_287 = __as_bf16x2(cols[12]);
    unsigned int _hxr_287;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_287) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_287)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_287)));
    __nv_bfloat162 _max_286 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_287);
    amax_a = _max_286;
    __nv_bfloat162 _hxa_289 = amax_b;
    __nv_bfloat162 _hxb_289 = __as_bf16x2(cols[28]);
    unsigned int _hxr_289;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_289) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_289)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_289)));
    __nv_bfloat162 _max_288 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_289);
    amax_b = _max_288;
    __nv_bfloat162 _hxa_291 = amax_a;
    __nv_bfloat162 _hxb_291 = __as_bf16x2(cols[13]);
    unsigned int _hxr_291;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_291) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_291)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_291)));
    __nv_bfloat162 _max_290 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_291);
    amax_a = _max_290;
    __nv_bfloat162 _hxa_293 = amax_b;
    __nv_bfloat162 _hxb_293 = __as_bf16x2(cols[29]);
    unsigned int _hxr_293;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_293) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_293)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_293)));
    __nv_bfloat162 _max_292 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_293);
    amax_b = _max_292;
    __nv_bfloat162 _hxa_295 = amax_a;
    __nv_bfloat162 _hxb_295 = __as_bf16x2(cols[14]);
    unsigned int _hxr_295;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_295) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_295)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_295)));
    __nv_bfloat162 _max_294 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_295);
    amax_a = _max_294;
    __nv_bfloat162 _hxa_297 = amax_b;
    __nv_bfloat162 _hxb_297 = __as_bf16x2(cols[30]);
    unsigned int _hxr_297;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_297) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_297)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_297)));
    __nv_bfloat162 _max_296 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_297);
    amax_b = _max_296;
    __nv_bfloat162 _hxa_299 = amax_a;
    __nv_bfloat162 _hxb_299 = __as_bf16x2(cols[15]);
    unsigned int _hxr_299;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_299) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_299)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_299)));
    __nv_bfloat162 _max_298 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_299);
    amax_a = _max_298;
    __nv_bfloat162 _hxa_301 = amax_b;
    __nv_bfloat162 _hxb_301 = __as_bf16x2(cols[31]);
    unsigned int _hxr_301;
    asm("max.NaN.xorsign.abs.bf16x2 %0, %1, %2;" : "=r"(_hxr_301) : "r"(*reinterpret_cast<const unsigned int*>(&_hxa_301)), "r"(*reinterpret_cast<const unsigned int*>(&_hxb_301)));
    __nv_bfloat162 _max_300 = *reinterpret_cast<const __nv_bfloat162*>(&_hxr_301);
    amax_b = _max_300;
    unsigned int mag_a = __as_u32(amax_a) & 2147450879;
    unsigned int mag_b = __as_u32(amax_b) & 2147450879;
    unsigned int lows_ab = mag_a & 65535 | mag_b << 16;
    unsigned int highs_ab = mag_a >> 16 | mag_b & 4294901760u;
    __nv_bfloat162 col_max = __hmax2_nan(__as_bf16x2(lows_ab), __as_bf16x2(highs_ab));
    unsigned int col_max_bits = __as_u32(col_max);
    unsigned int amax_bits_3 = (col_max_bits & 65535) << 16;
    int scale_i32_3 = (int)(amax_bits_3 + 2031616 >> 23) - 8;
    if (scale_i32_3 < 0) {
        scale_i32_3 = 0;
    }
    if ((amax_bits_3 & 2139095040) == 2139095040) {
        scale_i32_3 = 255;
    }
    unsigned int scale_4 = (unsigned int)scale_i32_3;
    unsigned int inv_3 = 254 - scale_4 << 7;
    if (scale_4 == 255) {
        inv_3 = 32704;
    }
    unsigned int inv2_3 = inv_3 | inv_3 << 16;
    unsigned int amax_bits_4 = (col_max_bits >> 16 & 65535) << 16;
    int scale_i32_4 = (int)(amax_bits_4 + 2031616 >> 23) - 8;
    if (scale_i32_4 < 0) {
        scale_i32_4 = 0;
    }
    if ((amax_bits_4 & 2139095040) == 2139095040) {
        scale_i32_4 = 255;
    }
    unsigned int scale_6 = (unsigned int)scale_i32_4;
    unsigned int inv_4 = 254 - scale_6 << 7;
    if (scale_6 == 255) {
        inv_4 = 32704;
    }
    unsigned int inv2_5 = inv_4 | inv_4 << 16;
    int gcol_a = bx * 128 + warp_1 * 16 + (lane_1 >> 2);
    int gcol_b = gcol_a + 8;
    int out_row = by * 128 + (lane_1 & 3) * 32;
    unsigned int words_a[8];
    unsigned int words_b[8];
    uint32_t bf16x2_mul_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_16) : "r"(cols[0]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_15;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_15) : "r"(bf16x2_mul_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_15) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_15;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_15) : "r"(cols[1]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_303;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_303) : "r"(bf16x2_mul_1_15));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_15);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_303) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_302;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_302) : "h"(e4m3x2_bf16x2_15), "h"(_cvt_satfinite_e4m3x2_bf16x2_303));
    words_a[0] = _mov_b32_pack_u16x2_302;
    uint32_t bf16x2_mul_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_17) : "r"(cols[16]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_16;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_16) : "r"(bf16x2_mul_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_16) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_16;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_16) : "r"(cols[17]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_305;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_305) : "r"(bf16x2_mul_1_16));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_16);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_305) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_304;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_304) : "h"(e4m3x2_bf16x2_16), "h"(_cvt_satfinite_e4m3x2_bf16x2_305));
    words_b[0] = _mov_b32_pack_u16x2_304;
    uint32_t bf16x2_mul_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_18) : "r"(cols[2]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_17;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_17) : "r"(bf16x2_mul_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_17) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_17;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_17) : "r"(cols[3]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_307;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_307) : "r"(bf16x2_mul_1_17));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_17);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_307) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_306;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_306) : "h"(e4m3x2_bf16x2_17), "h"(_cvt_satfinite_e4m3x2_bf16x2_307));
    words_a[1] = _mov_b32_pack_u16x2_306;
    uint32_t bf16x2_mul_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_19) : "r"(cols[18]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_18;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_18) : "r"(bf16x2_mul_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_18) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_18;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_18) : "r"(cols[19]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_309;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_309) : "r"(bf16x2_mul_1_18));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_18);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_309) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_308;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_308) : "h"(e4m3x2_bf16x2_18), "h"(_cvt_satfinite_e4m3x2_bf16x2_309));
    words_b[1] = _mov_b32_pack_u16x2_308;
    uint32_t bf16x2_mul_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_20) : "r"(cols[4]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_19;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_19) : "r"(bf16x2_mul_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_19) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_19;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_19) : "r"(cols[5]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_311;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_311) : "r"(bf16x2_mul_1_19));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_19);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_311) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_310;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_310) : "h"(e4m3x2_bf16x2_19), "h"(_cvt_satfinite_e4m3x2_bf16x2_311));
    words_a[2] = _mov_b32_pack_u16x2_310;
    uint32_t bf16x2_mul_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_21) : "r"(cols[20]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_20;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_20) : "r"(bf16x2_mul_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_20) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_20;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_20) : "r"(cols[21]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_313;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_313) : "r"(bf16x2_mul_1_20));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_20);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_313) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_312;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_312) : "h"(e4m3x2_bf16x2_20), "h"(_cvt_satfinite_e4m3x2_bf16x2_313));
    words_b[2] = _mov_b32_pack_u16x2_312;
    uint32_t bf16x2_mul_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_22) : "r"(cols[6]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_21;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_21) : "r"(bf16x2_mul_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_21) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_21;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_21) : "r"(cols[7]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_315;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_315) : "r"(bf16x2_mul_1_21));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_21);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_315) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_314;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_314) : "h"(e4m3x2_bf16x2_21), "h"(_cvt_satfinite_e4m3x2_bf16x2_315));
    words_a[3] = _mov_b32_pack_u16x2_314;
    uint32_t bf16x2_mul_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_23) : "r"(cols[22]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_22;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_22) : "r"(bf16x2_mul_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_22) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_22;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_22) : "r"(cols[23]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_317;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_317) : "r"(bf16x2_mul_1_22));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_22);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_317) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_316;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_316) : "h"(e4m3x2_bf16x2_22), "h"(_cvt_satfinite_e4m3x2_bf16x2_317));
    words_b[3] = _mov_b32_pack_u16x2_316;
    uint32_t bf16x2_mul_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_24) : "r"(cols[8]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_23;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_23) : "r"(bf16x2_mul_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_23) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_23;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_23) : "r"(cols[9]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_319;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_319) : "r"(bf16x2_mul_1_23));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_23);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_319) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_318;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_318) : "h"(e4m3x2_bf16x2_23), "h"(_cvt_satfinite_e4m3x2_bf16x2_319));
    words_a[4] = _mov_b32_pack_u16x2_318;
    uint32_t bf16x2_mul_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_25) : "r"(cols[24]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_24;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_24) : "r"(bf16x2_mul_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_24) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_24;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_24) : "r"(cols[25]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_321;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_321) : "r"(bf16x2_mul_1_24));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_24);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_321) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_320;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_320) : "h"(e4m3x2_bf16x2_24), "h"(_cvt_satfinite_e4m3x2_bf16x2_321));
    words_b[4] = _mov_b32_pack_u16x2_320;
    uint32_t bf16x2_mul_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_26) : "r"(cols[10]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_25;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_25) : "r"(bf16x2_mul_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_25) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_25;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_25) : "r"(cols[11]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_323;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_323) : "r"(bf16x2_mul_1_25));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_25);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_323) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_322;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_322) : "h"(e4m3x2_bf16x2_25), "h"(_cvt_satfinite_e4m3x2_bf16x2_323));
    words_a[5] = _mov_b32_pack_u16x2_322;
    uint32_t bf16x2_mul_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_27) : "r"(cols[26]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_26;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_26) : "r"(bf16x2_mul_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_26) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_26;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_26) : "r"(cols[27]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_325;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_325) : "r"(bf16x2_mul_1_26));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_26);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_325) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_324;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_324) : "h"(e4m3x2_bf16x2_26), "h"(_cvt_satfinite_e4m3x2_bf16x2_325));
    words_b[5] = _mov_b32_pack_u16x2_324;
    uint32_t bf16x2_mul_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_28) : "r"(cols[12]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_27;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_27) : "r"(bf16x2_mul_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_27) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_27;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_27) : "r"(cols[13]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_327;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_327) : "r"(bf16x2_mul_1_27));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_27);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_327) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_326;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_326) : "h"(e4m3x2_bf16x2_27), "h"(_cvt_satfinite_e4m3x2_bf16x2_327));
    words_a[6] = _mov_b32_pack_u16x2_326;
    uint32_t bf16x2_mul_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_29) : "r"(cols[28]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_28;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_28) : "r"(bf16x2_mul_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_28) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_28;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_28) : "r"(cols[29]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_329;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_329) : "r"(bf16x2_mul_1_28));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_28);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_329) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_328;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_328) : "h"(e4m3x2_bf16x2_28), "h"(_cvt_satfinite_e4m3x2_bf16x2_329));
    words_b[6] = _mov_b32_pack_u16x2_328;
    uint32_t bf16x2_mul_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_30) : "r"(cols[14]), "r"(inv2_3));
    uint16_t e4m3x2_bf16x2_29;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_29) : "r"(bf16x2_mul_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_29) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_29;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_29) : "r"(cols[15]), "r"(inv2_3));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_331;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_331) : "r"(bf16x2_mul_1_29));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_29);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_331) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_330;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_330) : "h"(e4m3x2_bf16x2_29), "h"(_cvt_satfinite_e4m3x2_bf16x2_331));
    words_a[7] = _mov_b32_pack_u16x2_330;
    uint32_t bf16x2_mul_31;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_31) : "r"(cols[30]), "r"(inv2_5));
    uint16_t e4m3x2_bf16x2_30;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(e4m3x2_bf16x2_30) : "r"(bf16x2_mul_31));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_31);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(e4m3x2_bf16x2_30) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t bf16x2_mul_1_30;
    asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(bf16x2_mul_1_30) : "r"(cols[31]), "r"(inv2_5));
    uint16_t _cvt_satfinite_e4m3x2_bf16x2_333;
    #if !defined(CAKE_CVT_SATFINITE_E4M3X2_BF16X2_FALLBACK) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 1)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    asm("cvt.rn.satfinite.e4m3x2.bf16x2 %0, %1;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_333) : "r"(bf16x2_mul_1_30));
    #else
    {
    const uint32_t _bf16x2 = (uint32_t)(bf16x2_mul_1_30);
    const float _lo = __uint_as_float(_bf16x2 << 16);
    const float _hi = __uint_as_float(_bf16x2 & 0xFFFF0000u);
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_cvt_satfinite_e4m3x2_bf16x2_333) : "f"(_hi), "f"(_lo));
    }
    #endif
    uint32_t _mov_b32_pack_u16x2_332;
    asm("mov.b32 %0, {%1, %2};" : "=r"(_mov_b32_pack_u16x2_332) : "h"(e4m3x2_bf16x2_30), "h"(_cvt_satfinite_e4m3x2_bf16x2_333));
    words_b[7] = _mov_b32_pack_u16x2_332;
    {
        const unsigned* _raw_stv8_334 = reinterpret_cast<const unsigned*>(words_a + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_a * M + out_row))), "r"(_raw_stv8_334[0]), "r"(_raw_stv8_334[1]), "r"(_raw_stv8_334[2]), "r"(_raw_stv8_334[3]), "r"(_raw_stv8_334[4]), "r"(_raw_stv8_334[5]), "r"(_raw_stv8_334[6]), "r"(_raw_stv8_334[7]) : "memory");
    }
    {
        const unsigned* _raw_stv8_335 = reinterpret_cast<const unsigned*>(words_b + 0);
        asm volatile(
            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
            :: "l"((void*)(col_output + (gcol_b * M + out_row))), "r"(_raw_stv8_335[0]), "r"(_raw_stv8_335[1]), "r"(_raw_stv8_335[2]), "r"(_raw_stv8_335[3]), "r"(_raw_stv8_335[4]), "r"(_raw_stv8_335[5]), "r"(_raw_stv8_335[6]), "r"(_raw_stv8_335[7]) : "memory");
    }
    int row_tiles = M >> 7;
    int rblk = by * 4 + (lane_1 & 3);
    int col_scale_a = ((gcol_a >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_a & 31) * 16 + (gcol_a >> 5 & 3) * 4 + (rblk & 3);
    int col_scale_b = ((gcol_b >> 7) * row_tiles + (rblk >> 2)) * 512 + (gcol_b & 31) * 16 + (gcol_b >> 5 & 3) * 4 + (rblk & 3);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_a) + (0)) = (unsigned char)(scale_4);
    *(reinterpret_cast<unsigned char*>(col_scales + col_scale_b) + (0)) = (unsigned char)(scale_6);
}

} // extern "C"
