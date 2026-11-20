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

#if __CUDA_ARCH__ == 1030
#define LAUNCH_MIN_BLOCKS 2
#else
#define LAUNCH_MIN_BLOCKS 1
#endif

#include <math_constants.h>

__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdp_9bfd49d6bbbb3d849f30(__nv_bfloat16* __restrict__ q, __nv_bfloat16* __restrict__ k, __nv_bfloat16* __restrict__ v, __half* __restrict__ state, float* __restrict__ A_log, __nv_bfloat16* __restrict__ a, float* __restrict__ dt_bias, __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int* __restrict__ initial_state_indices, int* __restrict__ output_state_indices, int* __restrict__ ssm_state_indices, long long* __restrict__ rand_seed, CakeFastDivmod heads, int num_tokens, float scale, int disable_state_update, int use_ssm_state_indices, int philox_rounds, long long state_size_p0, long long state_stride_p1, long long state_stride_p2, long long state_stride_p0, long long k_stride_p0, long long k_stride_p1, long long k_stride_p2, long long v_stride_p0, long long v_stride_p1, long long v_stride_p2, long long b_stride_p0, long long b_stride_p1, long long b_stride_p2, long long q_stride_p0, long long q_stride_p1, long long q_stride_p2, long long a_stride_p0, long long a_stride_p1, long long a_stride_p2, long long ssm_state_indices_stride_p0, long long ssm_state_indices_stride_p1, long long out_stride_p0, long long out_stride_p1, long long out_stride_p2)
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
    float* sK = reinterpret_cast<float*>(smem_raw + 0);
    const int sK_addr = smem + 0;
    float* sQ = reinterpret_cast<float*>(smem_raw + 9216);
    const int sQ_addr = smem + 9216;
    float* sV = reinterpret_cast<float*>(smem_raw + 12288);
    const int sV_addr = smem + 12288;
    float* sAlpha = reinterpret_cast<float*>(smem_raw + 16896);
    const int sAlpha_addr = smem + 16896;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 16920);
    const int sBeta_addr = smem + 16920;
    int* sSsm = reinterpret_cast<int*>(smem_raw + 16992);
    const int sSsm_addr = smem + 16992;
    unsigned int* sNoise = reinterpret_cast<unsigned int*>(smem_raw + 17024);
    const int sNoise_addr = smem + 17024;

    // === Task calls (declaration order) ===
    uint32_t _fast_div_q_0 = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)blockIdx.x), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)blockIdx.x);
    uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)blockIdx.x) - _fast_div_q_0 * (uint32_t)(heads.divisor);
    int n = (int)_fast_div_q_0;
    int h = (int)_fast_div_r_0;
    int tid_0 = tid;
    int lane_local = lane;
    int warp_local = warp;
    int q_warp_local = warp_local;
    int u = tid_0 - tid_0 / 8 * 8;
    int g = tid_0 / 8;
    int row0 = g * 2;
    unsigned int lane_byte = (unsigned int)(u * 16);
    unsigned int group_byte = (unsigned int)(g * 8);
    long long lane_elem = (long long)(u * 4);
    long long state_pool_size = state_size_p0;
    int read_state_slot_raw = initial_state_indices[n];
    int read_state_slot_valid = ((read_state_slot_raw >= 0 && state_pool_size > (long long)read_state_slot_raw) ? 1 : 0);
    int write_state_slot_raw = output_state_indices[n];
    int write_state_slot_valid = ((write_state_slot_raw >= 0 && state_pool_size > (long long)write_state_slot_raw) ? 1 : 0);
    long long head_rows_off = (long long)h * state_stride_p1 + (long long)row0 * state_stride_p2 + lane_elem;
    long long read_rows_base = (long long)read_state_slot_raw * state_stride_p0 + head_rows_off;
    long long write_rows_base = (long long)write_state_slot_raw * state_stride_p0 + head_rows_off;
    int k_col0 = lane_local * 4;
    int steps = 18;
    unsigned int seed_lo = 0;
    unsigned int seed_hi = 0;
    if (philox_rounds != 0) {
        long long seed_value = *rand_seed;
        seed_lo = (unsigned int)seed_value;
        seed_hi = (unsigned int)(seed_value >> 32ll);
    }
    long long read_slot_head = (long long)read_state_slot_raw * state_stride_p0 + (long long)h * state_stride_p1;
    float r_q[4];
    float r_k[4];
    float bk[12];
    float bv[6];
    float bb[3];
    int bm0_0 = warp_local;
    int bm_0 = bm0_0;
    long long bki_0 = (long long)n * k_stride_p0 + (long long)bm_0 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_0;
        _vld_0 = *reinterpret_cast<const uint2*>(k + bki_0 + (long long)k_col0);
        uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
            (&bk[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_0 = (long long)n * v_stride_p0 + (long long)bm_0 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_0 = lane_local * 2;
    bv[0] = (float)v[bvi_0 + (long long)bvc_0];
    bv[1] = (float)v[bvi_0 + (long long)bvc_0 + 1ll];
    long long bbi_0 = (long long)n * b_stride_p0 + (long long)bm_0 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[0] = (float)b[bbi_0];
    int bm0_1 = 8 + warp_local;
    int bm_1 = bm0_1;
    long long bki_1 = (long long)n * k_stride_p0 + (long long)bm_1 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_1;
        _vld_1 = *reinterpret_cast<const uint2*>(k + bki_1 + (long long)k_col0);
        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
            (&bk[4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_1 = (long long)n * v_stride_p0 + (long long)bm_1 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_1 = lane_local * 2;
    bv[2] = (float)v[bvi_1 + (long long)bvc_1];
    bv[3] = (float)v[bvi_1 + (long long)bvc_1 + 1ll];
    long long bbi_1 = (long long)n * b_stride_p0 + (long long)bm_1 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[1] = (float)b[bbi_1];
    int bm0_2 = 16 + warp_local;
    int bm_2 = ((bm0_2) < (steps - 1) ? (bm0_2) : (steps - 1));
    long long bki_2 = (long long)n * k_stride_p0 + (long long)bm_2 * k_stride_p1 + (long long)h * k_stride_p2;
    {
        uint2 _vld_2;
        _vld_2 = *reinterpret_cast<const uint2*>(k + bki_2 + (long long)k_col0);
        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bk[8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
            (&bk[8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
        }
    }
    long long bvi_2 = (long long)n * v_stride_p0 + (long long)bm_2 * v_stride_p1 + (long long)h * v_stride_p2;
    int bvc_2 = lane_local * 2;
    bv[4] = (float)v[bvi_2 + (long long)bvc_2];
    bv[5] = (float)v[bvi_2 + (long long)bvc_2 + 1ll];
    long long bbi_2 = (long long)n * b_stride_p0 + (long long)bm_2 * b_stride_p1 + (long long)h * b_stride_p2;
    bb[2] = (float)b[bbi_2];
    float bq[4];
    float ba[3];
    int bt0_0 = warp_local;
    int bt_0 = ((bt0_0) < (num_tokens - 1) ? (bt0_0) : (num_tokens - 1));
    long long bqi_0 = (long long)n * q_stride_p0 + (long long)bt_0 * q_stride_p1 + (long long)h * q_stride_p2;
    {
        uint2 _vld_3;
        _vld_3 = *reinterpret_cast<const uint2*>(q + bqi_0 + (long long)k_col0);
        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3);
        #pragma unroll
        for (int _pair = 0; _pair < 2; _pair++) {
            (&bq[0 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
            (&bq[0 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
        }
    }
    long long bgi_0 = (long long)n * a_stride_p0 + (long long)bt_0 * a_stride_p1 + (long long)h * a_stride_p2;
    ba[0] = (float)a[bgi_0];
    ba[1] = dt_bias[h];
    ba[2] = A_log[h];
    if (read_state_slot_valid != 0) {
        float S[32];
        float2 S2[16];
        float k_regs[16];
        float2 k2[8];
        float v_regs[4];
        float o4[4];
        unsigned int h_words[2];
        unsigned int h_noise[16];
        {
            uint2 _vld_4;
            _vld_4 = *reinterpret_cast<const uint2*>(state + read_rows_base);
            uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[0 + _pair * 2]))
                    : "r"(_vpairs_4[_pair]));
            }
        }
        {
            uint2 _vld_5;
            _vld_5 = *reinterpret_cast<const uint2*>(state + read_rows_base + 32ll);
            uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[4 + _pair * 2]))
                    : "r"(_vpairs_5[_pair]));
            }
        }
        {
            uint2 _vld_6;
            _vld_6 = *reinterpret_cast<const uint2*>(state + read_rows_base + 64ll);
            uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[8 + _pair * 2]))
                    : "r"(_vpairs_6[_pair]));
            }
        }
        {
            uint2 _vld_7;
            _vld_7 = *reinterpret_cast<const uint2*>(state + read_rows_base + 96ll);
            uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[12 + _pair * 2]))
                    : "r"(_vpairs_7[_pair]));
            }
        }
        {
            uint2 _vld_8;
            _vld_8 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2);
            uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[16 + _pair * 2]))
                    : "r"(_vpairs_8[_pair]));
            }
        }
        {
            uint2 _vld_9;
            _vld_9 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 32ll);
            uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[20 + _pair * 2]))
                    : "r"(_vpairs_9[_pair]));
            }
        }
        {
            uint2 _vld_10;
            _vld_10 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 64ll);
            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[24 + _pair * 2]))
                    : "r"(_vpairs_10[_pair]));
            }
        }
        {
            uint2 _vld_11;
            _vld_11 = *reinterpret_cast<const uint2*>(state + read_rows_base + state_stride_p2 + 96ll);
            uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11);
            #pragma unroll
            for (int _pair = 0; _pair < 2; _pair++) {
                asm volatile(
                    "{\n\t"
                    ".reg .b16 h_lo, h_hi;\n\t"
                    ".reg .b32 f_lo, f_hi;\n\t"
                    "mov.b32 {h_lo, h_hi}, %1;\n\t"
                    "cvt.f32.f16 f_lo, h_lo;\n\t"
                    "cvt.f32.f16 f_hi, h_hi;\n\t"
                    "mov.b64 %0, {f_lo, f_hi};\n\t"
                    "}\n"
                    : "=l"(*reinterpret_cast<unsigned long long*>(&S[28 + _pair * 2]))
                    : "r"(_vpairs_11[_pair]));
            }
        }
        unsigned int noise_lane = sNoise_addr + (unsigned int)(tid_0 * 16);
        int fm_0 = warp_local;
        float2 x0 = make_float2(bk[0], bk[1]);
        float2 x1 = make_float2(bk[2], bk[3]);
        float2 sum_pair = fma_f32x2_rn_ftz(x0, x0, make_float2(0.0f, 0.0f));
        sum_pair = fma_f32x2_rn_ftz(x1, x1, sum_pair);
        float sum_x = sum_pair.x + sum_pair.y;
        float _warp_reduce_12 = sum_x;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_12 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_12, offset);
        float norm = rsqrtf(_warp_reduce_12 + 1e-06f);
        float2 norm2 = make_float2(norm, norm);
        x0 = mul_f32x2(x0, norm2);
        x1 = mul_f32x2(x1, norm2);
        sK[fm_0 * 128 + k_col0] = x0.x;
        sK[fm_0 * 128 + k_col0 + 1] = x0.y;
        sK[fm_0 * 128 + k_col0 + 2] = x1.x;
        sK[fm_0 * 128 + k_col0 + 3] = x1.y;
        int v_col = lane_local * 2;
        sV[fm_0 * 64 + v_col] = bv[0];
        sV[fm_0 * 64 + v_col + 1] = bv[1];
        if (lane_local == 0) {
            float _exp_14 = __expf(-bb[0]);
            float _rcp_13 = approx_rcp(1.0f + _exp_14);
            sBeta[fm_0] = _rcp_13;
        }
        int fm_1 = 8 + warp_local;
        float2 x0_0 = make_float2(bk[4], bk[5]);
        float2 x1_0 = make_float2(bk[6], bk[7]);
        float2 sum_pair_0 = fma_f32x2_rn_ftz(x0_0, x0_0, make_float2(0.0f, 0.0f));
        sum_pair_0 = fma_f32x2_rn_ftz(x1_0, x1_0, sum_pair_0);
        float sum_x_0 = sum_pair_0.x + sum_pair_0.y;
        float _warp_reduce_15 = sum_x_0;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_15 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_15, offset);
        float norm_0 = rsqrtf(_warp_reduce_15 + 1e-06f);
        float2 norm2_0 = make_float2(norm_0, norm_0);
        x0_0 = mul_f32x2(x0_0, norm2_0);
        x1_0 = mul_f32x2(x1_0, norm2_0);
        sK[fm_1 * 128 + k_col0] = x0_0.x;
        sK[fm_1 * 128 + k_col0 + 1] = x0_0.y;
        sK[fm_1 * 128 + k_col0 + 2] = x1_0.x;
        sK[fm_1 * 128 + k_col0 + 3] = x1_0.y;
        int v_col_0 = lane_local * 2;
        sV[fm_1 * 64 + v_col_0] = bv[2];
        sV[fm_1 * 64 + v_col_0 + 1] = bv[3];
        if (lane_local == 0) {
            float _exp_17 = __expf(-bb[1]);
            float _rcp_16 = approx_rcp(1.0f + _exp_17);
            sBeta[fm_1] = _rcp_16;
        }
        int fm_2 = 16 + warp_local;
        if (fm_2 < steps) {
            float2 x0_1 = make_float2(bk[8], bk[9]);
            float2 x1_1 = make_float2(bk[10], bk[11]);
            float2 sum_pair_1 = fma_f32x2_rn_ftz(x0_1, x0_1, make_float2(0.0f, 0.0f));
            sum_pair_1 = fma_f32x2_rn_ftz(x1_1, x1_1, sum_pair_1);
            float sum_x_1 = sum_pair_1.x + sum_pair_1.y;
            float _warp_reduce_18 = sum_x_1;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_18 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_18, offset);
            float norm_1 = rsqrtf(_warp_reduce_18 + 1e-06f);
            float2 norm2_1 = make_float2(norm_1, norm_1);
            x0_1 = mul_f32x2(x0_1, norm2_1);
            x1_1 = mul_f32x2(x1_1, norm2_1);
            sK[fm_2 * 128 + k_col0] = x0_1.x;
            sK[fm_2 * 128 + k_col0 + 1] = x0_1.y;
            sK[fm_2 * 128 + k_col0 + 2] = x1_1.x;
            sK[fm_2 * 128 + k_col0 + 3] = x1_1.y;
            int v_col_1 = lane_local * 2;
            sV[fm_2 * 64 + v_col_1] = bv[4];
            sV[fm_2 * 64 + v_col_1 + 1] = bv[5];
            if (lane_local == 0) {
                float _exp_20 = __expf(-bb[2]);
                float _rcp_19 = approx_rcp(1.0f + _exp_20);
                sBeta[fm_2] = _rcp_19;
            }
        }
        int ft_0 = warp_local;
        if (ft_0 < num_tokens) {
            float2 x0_2 = make_float2(bq[0], bq[1]);
            float2 x1_2 = make_float2(bq[2], bq[3]);
            float2 sum_pair_2 = fma_f32x2_rn_ftz(x0_2, x0_2, make_float2(0.0f, 0.0f));
            sum_pair_2 = fma_f32x2_rn_ftz(x1_2, x1_2, sum_pair_2);
            float sum_x_2 = sum_pair_2.x + sum_pair_2.y;
            float _warp_reduce_21 = sum_x_2;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
                _warp_reduce_21 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_21, offset);
            float norm_2 = rsqrtf(_warp_reduce_21 + 1e-06f);
            norm_2 = norm_2 * scale;
            float2 norm2_2 = make_float2(norm_2, norm_2);
            x0_2 = mul_f32x2(x0_2, norm2_2);
            x1_2 = mul_f32x2(x1_2, norm2_2);
            sQ[ft_0 * 128 + k_col0] = x0_2.x;
            sQ[ft_0 * 128 + k_col0 + 1] = x0_2.y;
            sQ[ft_0 * 128 + k_col0 + 2] = x1_2.x;
            sQ[ft_0 * 128 + k_col0 + 3] = x1_2.y;
            if (lane_local == 0) {
                float xv = ba[0] + ba[1];
                float softplus_x = xv;
                if (xv <= 20.0f) {
                    float _exp_23 = __expf(xv);
                    float _log2_22;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22) : "f"(1.0f + _exp_23));
                    softplus_x = _log2_22 * 0.6931472f;
                }
                float _exp_24 = __expf(ba[2]);
                float g_log = -_exp_24 * softplus_x;
                float _exp_25 = __expf(g_log);
                sAlpha[ft_0] = _exp_25;
                if (use_ssm_state_indices != 0) {
                    sSsm[ft_0] = ssm_state_indices[(long long)n * ssm_state_indices_stride_p0 + (long long)ft_0 * ssm_state_indices_stride_p1];
                }
                if (use_ssm_state_indices == 0) {
                    sSsm[ft_0] = -1;
                }
            }
        }
        __syncthreads();
        S2[0] = make_float2(S[0], S[1]);
        S2[1] = make_float2(S[2], S[3]);
        S2[2] = make_float2(S[4], S[5]);
        S2[3] = make_float2(S[6], S[7]);
        S2[4] = make_float2(S[8], S[9]);
        S2[5] = make_float2(S[10], S[11]);
        S2[6] = make_float2(S[12], S[13]);
        S2[7] = make_float2(S[14], S[15]);
        S2[8] = make_float2(S[16], S[17]);
        S2[9] = make_float2(S[18], S[19]);
        S2[10] = make_float2(S[20], S[21]);
        S2[11] = make_float2(S[22], S[23]);
        S2[12] = make_float2(S[24], S[25]);
        S2[13] = make_float2(S[26], S[27]);
        S2[14] = make_float2(S[28], S[29]);
        S2[15] = make_float2(S[30], S[31]);
        if (use_ssm_state_indices != 0) {
            if (philox_rounds != 0) {
                unsigned int key0 = seed_lo;
                unsigned int key1 = seed_hi;
                long long o_0 = read_rows_base;
                unsigned long long p0_0 = (unsigned long long)(unsigned int)o_0 * 3528531795ull;
                h_noise[0] = (unsigned int)(o_0 >> 32ll) ^ key0;
                h_noise[2] = (unsigned int)(p0_0 >> 32ull) ^ key1;
                h_noise[3] = (unsigned int)p0_0;
                long long o_1 = read_rows_base + 32ll;
                unsigned long long p0_1 = (unsigned long long)(unsigned int)o_1 * 3528531795ull;
                h_noise[4] = (unsigned int)(o_1 >> 32ll) ^ key0;
                h_noise[6] = (unsigned int)(p0_1 >> 32ull) ^ key1;
                h_noise[7] = (unsigned int)p0_1;
                long long o_2 = read_rows_base + state_stride_p2;
                unsigned long long p0_2 = (unsigned long long)(unsigned int)o_2 * 3528531795ull;
                h_noise[8] = (unsigned int)(o_2 >> 32ll) ^ key0;
                h_noise[10] = (unsigned int)(p0_2 >> 32ull) ^ key1;
                h_noise[11] = (unsigned int)p0_2;
                long long o_3 = read_rows_base + state_stride_p2 + 32ll;
                unsigned long long p0_3 = (unsigned long long)(unsigned int)o_3 * 3528531795ull;
                h_noise[12] = (unsigned int)(o_3 >> 32ll) ^ key0;
                h_noise[14] = (unsigned int)(p0_3 >> 32ull) ^ key1;
                h_noise[15] = (unsigned int)p0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long q0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long q1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(q1_0 >> 32ull) ^ key0;
                h_noise[1] = (unsigned int)q1_0;
                h_noise[2] = (unsigned int)(q0_0 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)q0_0;
                unsigned long long q0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long q1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(q1_1 >> 32ull) ^ key0;
                h_noise[5] = (unsigned int)q1_1;
                h_noise[6] = (unsigned int)(q0_1 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)q0_1;
                unsigned long long q0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long q1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(q1_2 >> 32ull) ^ key0;
                h_noise[9] = (unsigned int)q1_2;
                h_noise[10] = (unsigned int)(q0_2 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)q0_2;
                unsigned long long q0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long q1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(q1_3 >> 32ull) ^ key0;
                h_noise[13] = (unsigned int)q1_3;
                h_noise[14] = (unsigned int)(q0_3 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)q0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_0 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_0;
                h_noise[2] = (unsigned int)(s0_0 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_0;
                unsigned long long s0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_1 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_1;
                h_noise[6] = (unsigned int)(s0_1 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_1;
                unsigned long long s0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_2 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_2;
                h_noise[10] = (unsigned int)(s0_2 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_2;
                unsigned long long s0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_3 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_3;
                h_noise[14] = (unsigned int)(s0_3 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_3;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_4 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_4;
                h_noise[2] = (unsigned int)(s0_4 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_4;
                unsigned long long s0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_5 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_5;
                h_noise[6] = (unsigned int)(s0_5 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_5;
                unsigned long long s0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_6 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_6;
                h_noise[10] = (unsigned int)(s0_6 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_6;
                unsigned long long s0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_7 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_7;
                h_noise[14] = (unsigned int)(s0_7 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_7;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                unsigned long long s0_8 = (unsigned long long)h_noise[0] * 3528531795ull;
                unsigned long long s1_8 = (unsigned long long)h_noise[2] * 3449720151ull;
                h_noise[0] = (unsigned int)(s1_8 >> 32ull) ^ h_noise[1] ^ key0;
                h_noise[1] = (unsigned int)s1_8;
                h_noise[2] = (unsigned int)(s0_8 >> 32ull) ^ h_noise[3] ^ key1;
                h_noise[3] = (unsigned int)s0_8;
                unsigned long long s0_9 = (unsigned long long)h_noise[4] * 3528531795ull;
                unsigned long long s1_9 = (unsigned long long)h_noise[6] * 3449720151ull;
                h_noise[4] = (unsigned int)(s1_9 >> 32ull) ^ h_noise[5] ^ key0;
                h_noise[5] = (unsigned int)s1_9;
                h_noise[6] = (unsigned int)(s0_9 >> 32ull) ^ h_noise[7] ^ key1;
                h_noise[7] = (unsigned int)s0_9;
                unsigned long long s0_10 = (unsigned long long)h_noise[8] * 3528531795ull;
                unsigned long long s1_10 = (unsigned long long)h_noise[10] * 3449720151ull;
                h_noise[8] = (unsigned int)(s1_10 >> 32ull) ^ h_noise[9] ^ key0;
                h_noise[9] = (unsigned int)s1_10;
                h_noise[10] = (unsigned int)(s0_10 >> 32ull) ^ h_noise[11] ^ key1;
                h_noise[11] = (unsigned int)s0_10;
                unsigned long long s0_11 = (unsigned long long)h_noise[12] * 3528531795ull;
                unsigned long long s1_11 = (unsigned long long)h_noise[14] * 3449720151ull;
                h_noise[12] = (unsigned int)(s1_11 >> 32ull) ^ h_noise[13] ^ key0;
                h_noise[13] = (unsigned int)s1_11;
                h_noise[14] = (unsigned int)(s0_11 >> 32ull) ^ h_noise[15] ^ key1;
                h_noise[15] = (unsigned int)s0_11;
                key0 = key0 + 2654435769u;
                key1 = key1 + 3144134277u;
                #pragma unroll 1
                for (int _round = 5; _round < philox_rounds; _round++) {
                    unsigned long long t0_0 = (unsigned long long)h_noise[0] * 3528531795ull;
                    unsigned long long t1_0 = (unsigned long long)h_noise[2] * 3449720151ull;
                    h_noise[0] = (unsigned int)(t1_0 >> 32ull) ^ h_noise[1] ^ key0;
                    h_noise[1] = (unsigned int)t1_0;
                    h_noise[2] = (unsigned int)(t0_0 >> 32ull) ^ h_noise[3] ^ key1;
                    h_noise[3] = (unsigned int)t0_0;
                    unsigned long long t0_1 = (unsigned long long)h_noise[4] * 3528531795ull;
                    unsigned long long t1_1 = (unsigned long long)h_noise[6] * 3449720151ull;
                    h_noise[4] = (unsigned int)(t1_1 >> 32ull) ^ h_noise[5] ^ key0;
                    h_noise[5] = (unsigned int)t1_1;
                    h_noise[6] = (unsigned int)(t0_1 >> 32ull) ^ h_noise[7] ^ key1;
                    h_noise[7] = (unsigned int)t0_1;
                    unsigned long long t0_2 = (unsigned long long)h_noise[8] * 3528531795ull;
                    unsigned long long t1_2 = (unsigned long long)h_noise[10] * 3449720151ull;
                    h_noise[8] = (unsigned int)(t1_2 >> 32ull) ^ h_noise[9] ^ key0;
                    h_noise[9] = (unsigned int)t1_2;
                    h_noise[10] = (unsigned int)(t0_2 >> 32ull) ^ h_noise[11] ^ key1;
                    h_noise[11] = (unsigned int)t0_2;
                    unsigned long long t0_3 = (unsigned long long)h_noise[12] * 3528531795ull;
                    unsigned long long t1_3 = (unsigned long long)h_noise[14] * 3449720151ull;
                    h_noise[12] = (unsigned int)(t1_3 >> 32ull) ^ h_noise[13] ^ key0;
                    h_noise[13] = (unsigned int)t1_3;
                    h_noise[14] = (unsigned int)(t0_3 >> 32ull) ^ h_noise[15] ^ key1;
                    h_noise[15] = (unsigned int)t0_3;
                    key0 = key0 + 2654435769u;
                    key1 = key1 + 3144134277u;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 0), "r"(h_noise[0] & 536813567), "r"(h_noise[1] & 536813567), "r"(h_noise[2] & 536813567), "r"(h_noise[3] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 4096), "r"(h_noise[4] & 536813567), "r"(h_noise[5] & 536813567), "r"(h_noise[6] & 536813567), "r"(h_noise[7] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 8192), "r"(h_noise[8] & 536813567), "r"(h_noise[9] & 536813567), "r"(h_noise[10] & 536813567), "r"(h_noise[11] & 536813567) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(noise_lane + 12288), "r"(h_noise[12] & 536813567), "r"(h_noise[13] & 536813567), "r"(h_noise[14] & 536813567), "r"(h_noise[15] & 536813567) : "memory");
            }
#if __CUDA_ARCH__ == 1030
            #pragma unroll 2
#else
            #pragma unroll 1
#endif
            for (int t = 0; t < 6; t++) {
                float alpha = sAlpha[t];
                float2 alpha2 = make_float2(alpha, alpha);
                S2[0] = mul_f32x2(S2[0], alpha2);
                S2[1] = mul_f32x2(S2[1], alpha2);
                S2[2] = mul_f32x2(S2[2], alpha2);
                S2[3] = mul_f32x2(S2[3], alpha2);
                S2[4] = mul_f32x2(S2[4], alpha2);
                S2[5] = mul_f32x2(S2[5], alpha2);
                S2[6] = mul_f32x2(S2[6], alpha2);
                S2[7] = mul_f32x2(S2[7], alpha2);
                S2[8] = mul_f32x2(S2[8], alpha2);
                S2[9] = mul_f32x2(S2[9], alpha2);
                S2[10] = mul_f32x2(S2[10], alpha2);
                S2[11] = mul_f32x2(S2[11], alpha2);
                S2[12] = mul_f32x2(S2[12], alpha2);
                S2[13] = mul_f32x2(S2[13], alpha2);
                S2[14] = mul_f32x2(S2[14], alpha2);
                S2[15] = mul_f32x2(S2[15], alpha2);
                int m_0 = t * 3;
                unsigned int kbase_0 = sK_addr + (unsigned int)(m_0 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_0 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_0 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_0 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_0 * 256) + group_byte));
                float beta_0 = sBeta[m_0];
                float2 acc0 = make_float2(0.0f, 0.0f);
                acc0 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0);
                acc0 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0);
                float dot = acc0.x + acc0.y;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_26;
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_27;
                float _shfl_xor_sync_28 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_28;
                float vn_0 = (v_regs[0] - dot) * beta_0;
                float2 vn2_0 = make_float2(vn_0, vn_0);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_0, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_0, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_0, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_0, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_0, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_0, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_0, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_0, S2[7]);
                float2 acc0_0 = make_float2(0.0f, 0.0f);
                acc0_0 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_0);
                acc0_0 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_0);
                float dot_0 = acc0_0.x + acc0_0.y;
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 1);
                dot_0 = dot_0 + _shfl_xor_sync_29;
                float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 2);
                dot_0 = dot_0 + _shfl_xor_sync_30;
                float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, dot_0, 4);
                dot_0 = dot_0 + _shfl_xor_sync_31;
                float vn_1 = (v_regs[1] - dot_0) * beta_0;
                float2 vn2_1 = make_float2(vn_1, vn_1);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_1, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_1, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_1, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_1, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_1, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_1, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_1, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_1, S2[15]);
                int m_1 = t * 3 + 1;
                unsigned int kbase_1 = sK_addr + (unsigned int)(m_1 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_1 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_1 * 256) + group_byte));
                float beta_1 = sBeta[m_1];
                float2 acc0_1 = make_float2(0.0f, 0.0f);
                acc0_1 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_1);
                acc0_1 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_1);
                float dot_1 = acc0_1.x + acc0_1.y;
                float _shfl_xor_sync_32 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_32;
                float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_33;
                float _shfl_xor_sync_34 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_34;
                float vn_2 = (v_regs[0] - dot_1) * beta_1;
                float2 vn2_2 = make_float2(vn_2, vn_2);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_2, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_2, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_2, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_2, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_2, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_2, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_2, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_2, S2[7]);
                float2 acc0_2 = make_float2(0.0f, 0.0f);
                acc0_2 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_2);
                acc0_2 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_2);
                float dot_2 = acc0_2.x + acc0_2.y;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 1);
                dot_2 = dot_2 + _shfl_xor_sync_35;
                float _shfl_xor_sync_36 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 2);
                dot_2 = dot_2 + _shfl_xor_sync_36;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, dot_2, 4);
                dot_2 = dot_2 + _shfl_xor_sync_37;
                float vn_3 = (v_regs[1] - dot_2) * beta_1;
                float2 vn2_3 = make_float2(vn_3, vn_3);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_3, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_3, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_3, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_3, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_3, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_3, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_3, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_3, S2[15]);
                int m_2 = t * 3 + 2;
                unsigned int kbase_2 = sK_addr + (unsigned int)(m_2 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_2));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_2 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_2 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_2 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_2 * 256) + group_byte));
                float beta_2 = sBeta[m_2];
                float2 acc0_3 = make_float2(0.0f, 0.0f);
                acc0_3 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_3);
                acc0_3 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_3);
                float dot_3 = acc0_3.x + acc0_3.y;
                float _shfl_xor_sync_38 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 1);
                dot_3 = dot_3 + _shfl_xor_sync_38;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 2);
                dot_3 = dot_3 + _shfl_xor_sync_39;
                float _shfl_xor_sync_40 = __shfl_xor_sync(0xFFFFFFFF, dot_3, 4);
                dot_3 = dot_3 + _shfl_xor_sync_40;
                float vn_4 = (v_regs[0] - dot_3) * beta_2;
                float2 vn2_4 = make_float2(vn_4, vn_4);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_4, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_4, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_4, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_4, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_4, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_4, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_4, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_4, S2[7]);
                float2 acc0_4 = make_float2(0.0f, 0.0f);
                acc0_4 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_4);
                acc0_4 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_4);
                float dot_4 = acc0_4.x + acc0_4.y;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 1);
                dot_4 = dot_4 + _shfl_xor_sync_41;
                float _shfl_xor_sync_42 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 2);
                dot_4 = dot_4 + _shfl_xor_sync_42;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, dot_4, 4);
                dot_4 = dot_4 + _shfl_xor_sync_43;
                float vn_5 = (v_regs[1] - dot_4) * beta_2;
                float2 vn2_5 = make_float2(vn_5, vn_5);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_5, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_5, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_5, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_5, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_5, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_5, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_5, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_5, S2[15]);
                unsigned int qbase = sQ_addr + (unsigned int)(t * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_5 = make_float2(0.0f, 0.0f);
                acc0_5 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_5);
                acc0_5 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_5);
                float dot_5 = acc0_5.x + acc0_5.y;
                float _shfl_xor_sync_44 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 1);
                dot_5 = dot_5 + _shfl_xor_sync_44;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 2);
                dot_5 = dot_5 + _shfl_xor_sync_45;
                float _shfl_xor_sync_46 = __shfl_xor_sync(0xFFFFFFFF, dot_5, 4);
                dot_5 = dot_5 + _shfl_xor_sync_46;
                o4[0] = dot_5;
                float2 acc0_6 = make_float2(0.0f, 0.0f);
                acc0_6 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_6);
                acc0_6 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_6);
                float dot_6 = acc0_6.x + acc0_6.y;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 1);
                dot_6 = dot_6 + _shfl_xor_sync_47;
                float _shfl_xor_sync_48 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 2);
                dot_6 = dot_6 + _shfl_xor_sync_48;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, dot_6, 4);
                dot_6 = dot_6 + _shfl_xor_sync_49;
                o4[1] = dot_6;
                if (u == 0) {
                    long long out_index = (long long)n * out_stride_p0 + (long long)t * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed)[0];
                }
                int ssm_slot = sSsm[t];
                int snap_valid = ((use_ssm_state_indices != 0 && ssm_slot >= 0 && state_pool_size > (long long)ssm_slot) ? 1 : 0);
                int safe_slot = ((snap_valid != 0) ? ssm_slot : 0);
                long long snap_base = (long long)safe_slot * state_stride_p0 + head_rows_off;
                if (philox_rounds != 0) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 0));
                    uint32_t _cvt_rs_x2_f32_50;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_50) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_50;
                    uint32_t _cvt_rs_x2_f32_51;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_51) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_51;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_52;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_52) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_52;
                    uint32_t _cvt_rs_x2_f32_53;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_53) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_53;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 4096));
                    uint32_t _cvt_rs_x2_f32_54;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_54) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_54;
                    uint32_t _cvt_rs_x2_f32_55;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_55) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_55;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_56;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_56) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_56;
                    uint32_t _cvt_rs_x2_f32_57;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_57) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_57;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 8192));
                    uint32_t _cvt_rs_x2_f32_58;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_58) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_58;
                    uint32_t _cvt_rs_x2_f32_59;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_59) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_59;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_60;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_60) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_60;
                    uint32_t _cvt_rs_x2_f32_61;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_61) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_61;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&h_noise[0])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&h_noise[(0) + 3]))
                        : "r"(noise_lane + 12288));
                    uint32_t _cvt_rs_x2_f32_62;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_62) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[0]));
                    h_words[0] = _cvt_rs_x2_f32_62;
                    uint32_t _cvt_rs_x2_f32_63;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_63) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[1]));
                    h_words[1] = _cvt_rs_x2_f32_63;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rs_x2_f32_64;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_64) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[2]));
                    h_words[0] = _cvt_rs_x2_f32_64;
                    uint32_t _cvt_rs_x2_f32_65;
                    asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_65) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[3]));
                    h_words[1] = _cvt_rs_x2_f32_65;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
                if (philox_rounds == 0) {
                    uint32_t _cvt_rn_f16x2_f32_66;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_66) : "f"(S2[0].y), "f"(S2[0].x));
                    h_words[0] = _cvt_rn_f16x2_f32_66;
                    uint32_t _cvt_rn_f16x2_f32_67;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_67) : "f"(S2[1].y), "f"(S2[1].x));
                    h_words[1] = _cvt_rn_f16x2_f32_67;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_68;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_68) : "f"(S2[2].y), "f"(S2[2].x));
                    h_words[0] = _cvt_rn_f16x2_f32_68;
                    uint32_t _cvt_rn_f16x2_f32_69;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_69) : "f"(S2[3].y), "f"(S2[3].x));
                    h_words[1] = _cvt_rn_f16x2_f32_69;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_70;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_70) : "f"(S2[4].y), "f"(S2[4].x));
                    h_words[0] = _cvt_rn_f16x2_f32_70;
                    uint32_t _cvt_rn_f16x2_f32_71;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_71) : "f"(S2[5].y), "f"(S2[5].x));
                    h_words[1] = _cvt_rn_f16x2_f32_71;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_72;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_72) : "f"(S2[6].y), "f"(S2[6].x));
                    h_words[0] = _cvt_rn_f16x2_f32_72;
                    uint32_t _cvt_rn_f16x2_f32_73;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_73) : "f"(S2[7].y), "f"(S2[7].x));
                    h_words[1] = _cvt_rn_f16x2_f32_73;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_74;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_74) : "f"(S2[8].y), "f"(S2[8].x));
                    h_words[0] = _cvt_rn_f16x2_f32_74;
                    uint32_t _cvt_rn_f16x2_f32_75;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_75) : "f"(S2[9].y), "f"(S2[9].x));
                    h_words[1] = _cvt_rn_f16x2_f32_75;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_76;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_76) : "f"(S2[10].y), "f"(S2[10].x));
                    h_words[0] = _cvt_rn_f16x2_f32_76;
                    uint32_t _cvt_rn_f16x2_f32_77;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_77) : "f"(S2[11].y), "f"(S2[11].x));
                    h_words[1] = _cvt_rn_f16x2_f32_77;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 32ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_78;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_78) : "f"(S2[12].y), "f"(S2[12].x));
                    h_words[0] = _cvt_rn_f16x2_f32_78;
                    uint32_t _cvt_rn_f16x2_f32_79;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_79) : "f"(S2[13].y), "f"(S2[13].x));
                    h_words[1] = _cvt_rn_f16x2_f32_79;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 64ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                    uint32_t _cvt_rn_f16x2_f32_80;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_80) : "f"(S2[14].y), "f"(S2[14].x));
                    h_words[0] = _cvt_rn_f16x2_f32_80;
                    uint32_t _cvt_rn_f16x2_f32_81;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_81) : "f"(S2[15].y), "f"(S2[15].x));
                    h_words[1] = _cvt_rn_f16x2_f32_81;
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %3, 0; @p_ st.global.v2.b32 [%0], {%1, %2}; }" :: "l"(state + (snap_base + state_stride_p2 + 96ll)), "r"(h_words[(0) + 0]), "r"(h_words[(0) + 1]), "r"((unsigned int)(snap_valid != 0)) : "memory");
                }
            }
        }
        if (use_ssm_state_indices == 0) {
#if __CUDA_ARCH__ == 1030
            #pragma unroll 2
#else
            #pragma unroll 1
#endif
            for (int t_1 = 0; t_1 < 6; t_1++) {
                float alpha_1 = sAlpha[t_1];
                float2 alpha2_1 = make_float2(alpha_1, alpha_1);
                S2[0] = mul_f32x2(S2[0], alpha2_1);
                S2[1] = mul_f32x2(S2[1], alpha2_1);
                S2[2] = mul_f32x2(S2[2], alpha2_1);
                S2[3] = mul_f32x2(S2[3], alpha2_1);
                S2[4] = mul_f32x2(S2[4], alpha2_1);
                S2[5] = mul_f32x2(S2[5], alpha2_1);
                S2[6] = mul_f32x2(S2[6], alpha2_1);
                S2[7] = mul_f32x2(S2[7], alpha2_1);
                S2[8] = mul_f32x2(S2[8], alpha2_1);
                S2[9] = mul_f32x2(S2[9], alpha2_1);
                S2[10] = mul_f32x2(S2[10], alpha2_1);
                S2[11] = mul_f32x2(S2[11], alpha2_1);
                S2[12] = mul_f32x2(S2[12], alpha2_1);
                S2[13] = mul_f32x2(S2[13], alpha2_1);
                S2[14] = mul_f32x2(S2[14], alpha2_1);
                S2[15] = mul_f32x2(S2[15], alpha2_1);
                int m_3 = t_1 * 3;
                unsigned int kbase_3 = sK_addr + (unsigned int)(m_3 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_3));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_3 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_3 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_3 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_3 * 256) + group_byte));
                float beta_3 = sBeta[m_3];
                float2 acc0_7 = make_float2(0.0f, 0.0f);
                acc0_7 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_7);
                acc0_7 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_7);
                float dot_7 = acc0_7.x + acc0_7.y;
                float _shfl_xor_sync_82 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 1);
                dot_7 = dot_7 + _shfl_xor_sync_82;
                float _shfl_xor_sync_83 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 2);
                dot_7 = dot_7 + _shfl_xor_sync_83;
                float _shfl_xor_sync_84 = __shfl_xor_sync(0xFFFFFFFF, dot_7, 4);
                dot_7 = dot_7 + _shfl_xor_sync_84;
                float vn_6 = (v_regs[0] - dot_7) * beta_3;
                float2 vn2_6 = make_float2(vn_6, vn_6);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_6, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_6, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_6, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_6, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_6, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_6, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_6, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_6, S2[7]);
                float2 acc0_7_1 = make_float2(0.0f, 0.0f);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_7_1);
                acc0_7_1 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_7_1);
                float dot_7_1 = acc0_7_1.x + acc0_7_1.y;
                float _shfl_xor_sync_85 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 1);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_85;
                float _shfl_xor_sync_86 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 2);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_86;
                float _shfl_xor_sync_87 = __shfl_xor_sync(0xFFFFFFFF, dot_7_1, 4);
                dot_7_1 = dot_7_1 + _shfl_xor_sync_87;
                float vn_7 = (v_regs[1] - dot_7_1) * beta_3;
                float2 vn2_7 = make_float2(vn_7, vn_7);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_7, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_7, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_7, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_7, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_7, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_7, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_7, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_7, S2[15]);
                int m_4 = t_1 * 3 + 1;
                unsigned int kbase_4 = sK_addr + (unsigned int)(m_4 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_4));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_4 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_4 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_4 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_4 * 256) + group_byte));
                float beta_4 = sBeta[m_4];
                float2 acc0_8 = make_float2(0.0f, 0.0f);
                acc0_8 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_8);
                acc0_8 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_8);
                float dot_8 = acc0_8.x + acc0_8.y;
                float _shfl_xor_sync_88 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 1);
                dot_8 = dot_8 + _shfl_xor_sync_88;
                float _shfl_xor_sync_89 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 2);
                dot_8 = dot_8 + _shfl_xor_sync_89;
                float _shfl_xor_sync_90 = __shfl_xor_sync(0xFFFFFFFF, dot_8, 4);
                dot_8 = dot_8 + _shfl_xor_sync_90;
                float vn_8 = (v_regs[0] - dot_8) * beta_4;
                float2 vn2_8 = make_float2(vn_8, vn_8);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_8, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_8, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_8, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_8, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_8, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_8, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_8, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_8, S2[7]);
                float2 acc0_9 = make_float2(0.0f, 0.0f);
                acc0_9 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_9);
                acc0_9 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_9);
                float dot_9 = acc0_9.x + acc0_9.y;
                float _shfl_xor_sync_91 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 1);
                dot_9 = dot_9 + _shfl_xor_sync_91;
                float _shfl_xor_sync_92 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 2);
                dot_9 = dot_9 + _shfl_xor_sync_92;
                float _shfl_xor_sync_93 = __shfl_xor_sync(0xFFFFFFFF, dot_9, 4);
                dot_9 = dot_9 + _shfl_xor_sync_93;
                float vn_9 = (v_regs[1] - dot_9) * beta_4;
                float2 vn2_9 = make_float2(vn_9, vn_9);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_9, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_9, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_9, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_9, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_9, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_9, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_9, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_9, S2[15]);
                int m_5 = t_1 * 3 + 2;
                unsigned int kbase_5 = sK_addr + (unsigned int)(m_5 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(kbase_5));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(kbase_5 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(kbase_5 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(kbase_5 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&v_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&v_regs[(0) + 1]))
                    : "r"(sV_addr + (unsigned int)(m_5 * 256) + group_byte));
                float beta_5 = sBeta[m_5];
                float2 acc0_10 = make_float2(0.0f, 0.0f);
                acc0_10 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_10);
                acc0_10 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_10);
                float dot_10 = acc0_10.x + acc0_10.y;
                float _shfl_xor_sync_94 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 1);
                dot_10 = dot_10 + _shfl_xor_sync_94;
                float _shfl_xor_sync_95 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 2);
                dot_10 = dot_10 + _shfl_xor_sync_95;
                float _shfl_xor_sync_96 = __shfl_xor_sync(0xFFFFFFFF, dot_10, 4);
                dot_10 = dot_10 + _shfl_xor_sync_96;
                float vn_10 = (v_regs[0] - dot_10) * beta_5;
                float2 vn2_10 = make_float2(vn_10, vn_10);
                S2[0] = fma_f32x2_rn_ftz(k2[0], vn2_10, S2[0]);
                S2[1] = fma_f32x2_rn_ftz(k2[1], vn2_10, S2[1]);
                S2[2] = fma_f32x2_rn_ftz(k2[2], vn2_10, S2[2]);
                S2[3] = fma_f32x2_rn_ftz(k2[3], vn2_10, S2[3]);
                S2[4] = fma_f32x2_rn_ftz(k2[4], vn2_10, S2[4]);
                S2[5] = fma_f32x2_rn_ftz(k2[5], vn2_10, S2[5]);
                S2[6] = fma_f32x2_rn_ftz(k2[6], vn2_10, S2[6]);
                S2[7] = fma_f32x2_rn_ftz(k2[7], vn2_10, S2[7]);
                float2 acc0_11 = make_float2(0.0f, 0.0f);
                acc0_11 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_11);
                acc0_11 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_11);
                float dot_11 = acc0_11.x + acc0_11.y;
                float _shfl_xor_sync_97 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 1);
                dot_11 = dot_11 + _shfl_xor_sync_97;
                float _shfl_xor_sync_98 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 2);
                dot_11 = dot_11 + _shfl_xor_sync_98;
                float _shfl_xor_sync_99 = __shfl_xor_sync(0xFFFFFFFF, dot_11, 4);
                dot_11 = dot_11 + _shfl_xor_sync_99;
                float vn_11 = (v_regs[1] - dot_11) * beta_5;
                float2 vn2_11 = make_float2(vn_11, vn_11);
                S2[8] = fma_f32x2_rn_ftz(k2[0], vn2_11, S2[8]);
                S2[9] = fma_f32x2_rn_ftz(k2[1], vn2_11, S2[9]);
                S2[10] = fma_f32x2_rn_ftz(k2[2], vn2_11, S2[10]);
                S2[11] = fma_f32x2_rn_ftz(k2[3], vn2_11, S2[11]);
                S2[12] = fma_f32x2_rn_ftz(k2[4], vn2_11, S2[12]);
                S2[13] = fma_f32x2_rn_ftz(k2[5], vn2_11, S2[13]);
                S2[14] = fma_f32x2_rn_ftz(k2[6], vn2_11, S2[14]);
                S2[15] = fma_f32x2_rn_ftz(k2[7], vn2_11, S2[15]);
                unsigned int qbase_1 = sQ_addr + (unsigned int)(t_1 * 512) + lane_byte;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(0) + 3]))
                    : "r"(qbase_1));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(4) + 3]))
                    : "r"(qbase_1 + 128));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[8])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(8) + 3]))
                    : "r"(qbase_1 + 256));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_regs[12])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_regs[(12) + 3]))
                    : "r"(qbase_1 + 384));
                k2[0] = make_float2(k_regs[0], k_regs[1]);
                k2[1] = make_float2(k_regs[2], k_regs[3]);
                k2[2] = make_float2(k_regs[4], k_regs[5]);
                k2[3] = make_float2(k_regs[6], k_regs[7]);
                k2[4] = make_float2(k_regs[8], k_regs[9]);
                k2[5] = make_float2(k_regs[10], k_regs[11]);
                k2[6] = make_float2(k_regs[12], k_regs[13]);
                k2[7] = make_float2(k_regs[14], k_regs[15]);
                float2 acc0_12 = make_float2(0.0f, 0.0f);
                acc0_12 = fma_f32x2_rn_ftz(S2[0], k2[0], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[1], k2[1], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[2], k2[2], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[3], k2[3], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[4], k2[4], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[5], k2[5], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[6], k2[6], acc0_12);
                acc0_12 = fma_f32x2_rn_ftz(S2[7], k2[7], acc0_12);
                float dot_12 = acc0_12.x + acc0_12.y;
                float _shfl_xor_sync_100 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 1);
                dot_12 = dot_12 + _shfl_xor_sync_100;
                float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 2);
                dot_12 = dot_12 + _shfl_xor_sync_101;
                float _shfl_xor_sync_102 = __shfl_xor_sync(0xFFFFFFFF, dot_12, 4);
                dot_12 = dot_12 + _shfl_xor_sync_102;
                o4[0] = dot_12;
                float2 acc0_13 = make_float2(0.0f, 0.0f);
                acc0_13 = fma_f32x2_rn_ftz(S2[8], k2[0], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[9], k2[1], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[10], k2[2], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[11], k2[3], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[12], k2[4], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[13], k2[5], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[14], k2[6], acc0_13);
                acc0_13 = fma_f32x2_rn_ftz(S2[15], k2[7], acc0_13);
                float dot_13 = acc0_13.x + acc0_13.y;
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 1);
                dot_13 = dot_13 + _shfl_xor_sync_103;
                float _shfl_xor_sync_104 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 2);
                dot_13 = dot_13 + _shfl_xor_sync_104;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, dot_13, 4);
                dot_13 = dot_13 + _shfl_xor_sync_105;
                o4[1] = dot_13;
                if (u == 0) {
                    long long out_index_1 = (long long)n * out_stride_p0 + (long long)t_1 * out_stride_p1 + (long long)h * out_stride_p2;
                    unsigned int packed_1[1];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o4[_lp*2 + 0], o4[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<unsigned*>(out + (out_index_1 + (long long)row0))[0] = reinterpret_cast<unsigned*>(packed_1)[0];
                }
            }
        }
        if (write_state_slot_valid != 0) {
            if (disable_state_update == 0) {
                if (use_ssm_state_indices == 0) {
                    uint32_t _read_special_reg_106;
                    asm volatile("mov.u32 %0, %%tid.x;" : "=r"(_read_special_reg_106));
                    int tid_r = (int)_read_special_reg_106;
                    long long noise_at = read_slot_head + (long long)(tid_r / 8 * 2) * state_stride_p2 + (long long)((tid_r - tid_r / 8 * 8) * 4);
                    if (philox_rounds != 0) {
                        unsigned int key0_1 = seed_lo;
                        unsigned int key1_1 = seed_hi;
                        long long o_4 = noise_at;
                        unsigned long long p0_4 = (unsigned long long)(unsigned int)o_4 * 3528531795ull;
                        h_noise[0] = (unsigned int)(o_4 >> 32ll) ^ key0_1;
                        h_noise[2] = (unsigned int)(p0_4 >> 32ull) ^ key1_1;
                        h_noise[3] = (unsigned int)p0_4;
                        long long o_5 = noise_at + 32ll;
                        unsigned long long p0_5 = (unsigned long long)(unsigned int)o_5 * 3528531795ull;
                        h_noise[4] = (unsigned int)(o_5 >> 32ll) ^ key0_1;
                        h_noise[6] = (unsigned int)(p0_5 >> 32ull) ^ key1_1;
                        h_noise[7] = (unsigned int)p0_5;
                        long long o_6 = noise_at + state_stride_p2;
                        unsigned long long p0_6 = (unsigned long long)(unsigned int)o_6 * 3528531795ull;
                        h_noise[8] = (unsigned int)(o_6 >> 32ll) ^ key0_1;
                        h_noise[10] = (unsigned int)(p0_6 >> 32ull) ^ key1_1;
                        h_noise[11] = (unsigned int)p0_6;
                        long long o_7 = noise_at + state_stride_p2 + 32ll;
                        unsigned long long p0_7 = (unsigned long long)(unsigned int)o_7 * 3528531795ull;
                        h_noise[12] = (unsigned int)(o_7 >> 32ll) ^ key0_1;
                        h_noise[14] = (unsigned int)(p0_7 >> 32ull) ^ key1_1;
                        h_noise[15] = (unsigned int)p0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long q0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long q1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(q1_4 >> 32ull) ^ key0_1;
                        h_noise[1] = (unsigned int)q1_4;
                        h_noise[2] = (unsigned int)(q0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)q0_4;
                        unsigned long long q0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long q1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(q1_5 >> 32ull) ^ key0_1;
                        h_noise[5] = (unsigned int)q1_5;
                        h_noise[6] = (unsigned int)(q0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)q0_5;
                        unsigned long long q0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long q1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(q1_6 >> 32ull) ^ key0_1;
                        h_noise[9] = (unsigned int)q1_6;
                        h_noise[10] = (unsigned int)(q0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)q0_6;
                        unsigned long long q0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long q1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(q1_7 >> 32ull) ^ key0_1;
                        h_noise[13] = (unsigned int)q1_7;
                        h_noise[14] = (unsigned int)(q0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)q0_7;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_12 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_12 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_12 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_12;
                        h_noise[2] = (unsigned int)(s0_12 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_12;
                        unsigned long long s0_13 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_13 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_13 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_13;
                        h_noise[6] = (unsigned int)(s0_13 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_13;
                        unsigned long long s0_14 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_14 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_14 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_14;
                        h_noise[10] = (unsigned int)(s0_14 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_14;
                        unsigned long long s0_15 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_15 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_15 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_15;
                        h_noise[14] = (unsigned int)(s0_15 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_15;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_16 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_16 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_16 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_16;
                        h_noise[2] = (unsigned int)(s0_16 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_16;
                        unsigned long long s0_17 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_17 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_17 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_17;
                        h_noise[6] = (unsigned int)(s0_17 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_17;
                        unsigned long long s0_18 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_18 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_18 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_18;
                        h_noise[10] = (unsigned int)(s0_18 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_18;
                        unsigned long long s0_19 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_19 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_19 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_19;
                        h_noise[14] = (unsigned int)(s0_19 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_19;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        unsigned long long s0_20 = (unsigned long long)h_noise[0] * 3528531795ull;
                        unsigned long long s1_20 = (unsigned long long)h_noise[2] * 3449720151ull;
                        h_noise[0] = (unsigned int)(s1_20 >> 32ull) ^ h_noise[1] ^ key0_1;
                        h_noise[1] = (unsigned int)s1_20;
                        h_noise[2] = (unsigned int)(s0_20 >> 32ull) ^ h_noise[3] ^ key1_1;
                        h_noise[3] = (unsigned int)s0_20;
                        unsigned long long s0_21 = (unsigned long long)h_noise[4] * 3528531795ull;
                        unsigned long long s1_21 = (unsigned long long)h_noise[6] * 3449720151ull;
                        h_noise[4] = (unsigned int)(s1_21 >> 32ull) ^ h_noise[5] ^ key0_1;
                        h_noise[5] = (unsigned int)s1_21;
                        h_noise[6] = (unsigned int)(s0_21 >> 32ull) ^ h_noise[7] ^ key1_1;
                        h_noise[7] = (unsigned int)s0_21;
                        unsigned long long s0_22 = (unsigned long long)h_noise[8] * 3528531795ull;
                        unsigned long long s1_22 = (unsigned long long)h_noise[10] * 3449720151ull;
                        h_noise[8] = (unsigned int)(s1_22 >> 32ull) ^ h_noise[9] ^ key0_1;
                        h_noise[9] = (unsigned int)s1_22;
                        h_noise[10] = (unsigned int)(s0_22 >> 32ull) ^ h_noise[11] ^ key1_1;
                        h_noise[11] = (unsigned int)s0_22;
                        unsigned long long s0_23 = (unsigned long long)h_noise[12] * 3528531795ull;
                        unsigned long long s1_23 = (unsigned long long)h_noise[14] * 3449720151ull;
                        h_noise[12] = (unsigned int)(s1_23 >> 32ull) ^ h_noise[13] ^ key0_1;
                        h_noise[13] = (unsigned int)s1_23;
                        h_noise[14] = (unsigned int)(s0_23 >> 32ull) ^ h_noise[15] ^ key1_1;
                        h_noise[15] = (unsigned int)s0_23;
                        key0_1 = key0_1 + 2654435769u;
                        key1_1 = key1_1 + 3144134277u;
                        #pragma unroll 1
                        for (int _round_1 = 5; _round_1 < philox_rounds; _round_1++) {
                            unsigned long long t0_4 = (unsigned long long)h_noise[0] * 3528531795ull;
                            unsigned long long t1_4 = (unsigned long long)h_noise[2] * 3449720151ull;
                            h_noise[0] = (unsigned int)(t1_4 >> 32ull) ^ h_noise[1] ^ key0_1;
                            h_noise[1] = (unsigned int)t1_4;
                            h_noise[2] = (unsigned int)(t0_4 >> 32ull) ^ h_noise[3] ^ key1_1;
                            h_noise[3] = (unsigned int)t0_4;
                            unsigned long long t0_5 = (unsigned long long)h_noise[4] * 3528531795ull;
                            unsigned long long t1_5 = (unsigned long long)h_noise[6] * 3449720151ull;
                            h_noise[4] = (unsigned int)(t1_5 >> 32ull) ^ h_noise[5] ^ key0_1;
                            h_noise[5] = (unsigned int)t1_5;
                            h_noise[6] = (unsigned int)(t0_5 >> 32ull) ^ h_noise[7] ^ key1_1;
                            h_noise[7] = (unsigned int)t0_5;
                            unsigned long long t0_6 = (unsigned long long)h_noise[8] * 3528531795ull;
                            unsigned long long t1_6 = (unsigned long long)h_noise[10] * 3449720151ull;
                            h_noise[8] = (unsigned int)(t1_6 >> 32ull) ^ h_noise[9] ^ key0_1;
                            h_noise[9] = (unsigned int)t1_6;
                            h_noise[10] = (unsigned int)(t0_6 >> 32ull) ^ h_noise[11] ^ key1_1;
                            h_noise[11] = (unsigned int)t0_6;
                            unsigned long long t0_7 = (unsigned long long)h_noise[12] * 3528531795ull;
                            unsigned long long t1_7 = (unsigned long long)h_noise[14] * 3449720151ull;
                            h_noise[12] = (unsigned int)(t1_7 >> 32ull) ^ h_noise[13] ^ key0_1;
                            h_noise[13] = (unsigned int)t1_7;
                            h_noise[14] = (unsigned int)(t0_7 >> 32ull) ^ h_noise[15] ^ key1_1;
                            h_noise[15] = (unsigned int)t0_7;
                            key0_1 = key0_1 + 2654435769u;
                            key1_1 = key1_1 + 3144134277u;
                        }
                        uint32_t _cvt_rs_x2_f32_107;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_107) : "f"(S2[0].y), "f"(S2[0].x), "r"(h_noise[0] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_107;
                        uint32_t _cvt_rs_x2_f32_108;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_108) : "f"(S2[1].y), "f"(S2[1].x), "r"(h_noise[1] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_108;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_109;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_109) : "f"(S2[4].y), "f"(S2[4].x), "r"(h_noise[2] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_109;
                        uint32_t _cvt_rs_x2_f32_110;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_110) : "f"(S2[5].y), "f"(S2[5].x), "r"(h_noise[3] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_110;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_111;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_111) : "f"(S2[2].y), "f"(S2[2].x), "r"(h_noise[4] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_111;
                        uint32_t _cvt_rs_x2_f32_112;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_112) : "f"(S2[3].y), "f"(S2[3].x), "r"(h_noise[5] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_112;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_113;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_113) : "f"(S2[6].y), "f"(S2[6].x), "r"(h_noise[6] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_113;
                        uint32_t _cvt_rs_x2_f32_114;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_114) : "f"(S2[7].y), "f"(S2[7].x), "r"(h_noise[7] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_114;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_115;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_115) : "f"(S2[8].y), "f"(S2[8].x), "r"(h_noise[8] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_115;
                        uint32_t _cvt_rs_x2_f32_116;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_116) : "f"(S2[9].y), "f"(S2[9].x), "r"(h_noise[9] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_116;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_117;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_117) : "f"(S2[12].y), "f"(S2[12].x), "r"(h_noise[10] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_117;
                        uint32_t _cvt_rs_x2_f32_118;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_118) : "f"(S2[13].y), "f"(S2[13].x), "r"(h_noise[11] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_118;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_119;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_119) : "f"(S2[10].y), "f"(S2[10].x), "r"(h_noise[12] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_119;
                        uint32_t _cvt_rs_x2_f32_120;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_120) : "f"(S2[11].y), "f"(S2[11].x), "r"(h_noise[13] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_120;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rs_x2_f32_121;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_121) : "f"(S2[14].y), "f"(S2[14].x), "r"(h_noise[14] & 536813567));
                        h_words[0] = _cvt_rs_x2_f32_121;
                        uint32_t _cvt_rs_x2_f32_122;
                        asm("cvt.rs.f16x2.f32 %0, %1, %2, %3;" : "=r"(_cvt_rs_x2_f32_122) : "f"(S2[15].y), "f"(S2[15].x), "r"(h_noise[15] & 536813567));
                        h_words[1] = _cvt_rs_x2_f32_122;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                    if (philox_rounds == 0) {
                        uint32_t _cvt_rn_f16x2_f32_123;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_123) : "f"(S2[0].y), "f"(S2[0].x));
                        h_words[0] = _cvt_rn_f16x2_f32_123;
                        uint32_t _cvt_rn_f16x2_f32_124;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_124) : "f"(S2[1].y), "f"(S2[1].x));
                        h_words[1] = _cvt_rn_f16x2_f32_124;
                        reinterpret_cast<int2*>(state + (write_rows_base))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_125;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_125) : "f"(S2[2].y), "f"(S2[2].x));
                        h_words[0] = _cvt_rn_f16x2_f32_125;
                        uint32_t _cvt_rn_f16x2_f32_126;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_126) : "f"(S2[3].y), "f"(S2[3].x));
                        h_words[1] = _cvt_rn_f16x2_f32_126;
                        reinterpret_cast<int2*>(state + (write_rows_base + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_127;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_127) : "f"(S2[4].y), "f"(S2[4].x));
                        h_words[0] = _cvt_rn_f16x2_f32_127;
                        uint32_t _cvt_rn_f16x2_f32_128;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_128) : "f"(S2[5].y), "f"(S2[5].x));
                        h_words[1] = _cvt_rn_f16x2_f32_128;
                        reinterpret_cast<int2*>(state + (write_rows_base + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_129;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_129) : "f"(S2[6].y), "f"(S2[6].x));
                        h_words[0] = _cvt_rn_f16x2_f32_129;
                        uint32_t _cvt_rn_f16x2_f32_130;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_130) : "f"(S2[7].y), "f"(S2[7].x));
                        h_words[1] = _cvt_rn_f16x2_f32_130;
                        reinterpret_cast<int2*>(state + (write_rows_base + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_131;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_131) : "f"(S2[8].y), "f"(S2[8].x));
                        h_words[0] = _cvt_rn_f16x2_f32_131;
                        uint32_t _cvt_rn_f16x2_f32_132;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_132) : "f"(S2[9].y), "f"(S2[9].x));
                        h_words[1] = _cvt_rn_f16x2_f32_132;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_133;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_133) : "f"(S2[10].y), "f"(S2[10].x));
                        h_words[0] = _cvt_rn_f16x2_f32_133;
                        uint32_t _cvt_rn_f16x2_f32_134;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_134) : "f"(S2[11].y), "f"(S2[11].x));
                        h_words[1] = _cvt_rn_f16x2_f32_134;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 32ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_135;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_135) : "f"(S2[12].y), "f"(S2[12].x));
                        h_words[0] = _cvt_rn_f16x2_f32_135;
                        uint32_t _cvt_rn_f16x2_f32_136;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_136) : "f"(S2[13].y), "f"(S2[13].x));
                        h_words[1] = _cvt_rn_f16x2_f32_136;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 64ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                        uint32_t _cvt_rn_f16x2_f32_137;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_137) : "f"(S2[14].y), "f"(S2[14].x));
                        h_words[0] = _cvt_rn_f16x2_f32_137;
                        uint32_t _cvt_rn_f16x2_f32_138;
                        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_cvt_rn_f16x2_f32_138) : "f"(S2[15].y), "f"(S2[15].x));
                        h_words[1] = _cvt_rn_f16x2_f32_138;
                        reinterpret_cast<int2*>(state + (write_rows_base + state_stride_p2 + 96ll))[0] = reinterpret_cast<int2*>(h_words)[0];
                    }
                }
            }
        }
    }
}

} // extern "C"
