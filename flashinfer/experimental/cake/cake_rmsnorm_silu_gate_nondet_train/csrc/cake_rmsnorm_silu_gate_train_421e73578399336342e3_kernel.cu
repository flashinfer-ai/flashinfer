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

#define LAUNCH_MIN_BLOCKS 8

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


extern "C" __device__ float __nv_expf(float);


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

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_rmsnorm_silu_gate_train_421e73578399336342e3(__nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ z, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ dx, __nv_bfloat16* __restrict__ dz, unsigned long long* __restrict__ dw_win, float* __restrict__ dw_spec, unsigned int* __restrict__ dw_done, float* __restrict__ dw, int rows, CakeFastDivmod heads, long long dy_tok, long long dy_head, long long x_tok, long long x_head, long long z_tok, long long z_head, float eps, int accumulate, int weight_grad, int blocks_per_chunk)
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
    float* fold = reinterpret_cast<float*>(smem_raw + 0);
    const int fold_addr = smem + 0;
    long long* win = reinterpret_cast<long long*>(smem_raw + 8192);
    const int win_addr = smem + 8192;
    int* cell = reinterpret_cast<int*>(smem_raw + 15360);
    const int cell_addr = smem + 15360;
    int* flag = reinterpret_cast<int*>(smem_raw + 15376);
    const int flag_addr = smem + 15376;
    int* tab = reinterpret_cast<int*>(smem_raw + 15392);
    const int tab_addr = smem + 15392;

    // === Task calls (declaration order) ===
    int group = tid >> 4;
    int col = (tid & 15) * 8;
    int ndb = (rows + 7) / 8;
    int nchunks = (ndb + blocks_per_chunk - 1) / blocks_per_chunk;
    int padded_block = ((rows % 8 != 0) ? ndb - 1 : -1);
    long long heads_i64 = (long long)((uint32_t)(heads.divisor));
    unsigned long long row_stride = (unsigned long long)(unsigned int)dy_head;
    int shared_rows = ((dy_head >= 0ll) ? 1 : 0) * ((dy_head < 2147483648ll) ? 1 : 0) * ((dy_tok == heads_i64 * dy_head) ? 1 : 0) * ((x_tok == dy_tok) ? 1 : 0) * ((z_tok == dy_tok) ? 1 : 0) * ((x_head == dy_head) ? 1 : 0) * ((z_head == dy_head) ? 1 : 0);
    float wv[8];
    {
        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + col + 0);
        uint4 _vld_0[1];
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vld_0[_blk] = _vptr_0[_blk];
            uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0[_blk]);
            #pragma unroll
            for (int _pair = 0; _pair < 4; _pair++) {
                (&wv[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                (&wv[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
            }
        }
    }
    int ticket = 2147483647;
    int claim_at = -1;
    win[tid] = 0ll;
    win[128 + tid] = 0ll;
    win[256 + tid] = 0ll;
    win[384 + tid] = 0ll;
    win[512 + tid] = 0ll;
    win[640 + tid] = 0ll;
    win[768 + tid] = 0ll;
    float spec = 0.0f;
    float acc[8];
    int parity = 0;
    int mine = (nchunks - bid + num_bids - 1) / num_bids;
    #pragma unroll 1
    for (int i = 0; i < mine; i++) {
        int sb0 = (bid + i * num_bids) * blocks_per_chunk;
        acc[0] = 0.0f;
        acc[1] = 0.0f;
        acc[2] = 0.0f;
        acc[3] = 0.0f;
        acc[4] = 0.0f;
        acc[5] = 0.0f;
        acc[6] = 0.0f;
        acc[7] = 0.0f;
        int _min_1 = ((sb0 + blocks_per_chunk) < (ndb) ? (sb0 + blocks_per_chunk) : (ndb));
        #pragma unroll 1
        for (int u = sb0; u < _min_1; u++) {
            float gs[8];
            float xs[8];
            float zs[8];
            int row = u * 8 + group;
            int src = ((row) < (rows - 1) ? (row) : (rows - 1));
            if (shared_rows != 0) {
                long long at = (long long)((unsigned long long)(unsigned int)src * row_stride + (unsigned long long)col);
                {
                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(dy + at + 0);
                    uint4 _vld_2[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_2[_blk] = _vptr_2[_blk];
                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) << 16);
                            (&gs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_2[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_3 = reinterpret_cast<const uint4*>(x + at + 0);
                    uint4 _vld_3[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_3[_blk] = _vptr_3[_blk];
                        uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) << 16);
                            (&xs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_3[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_4 = reinterpret_cast<const uint4*>(z + at + 0);
                    uint4 _vld_4[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_4[_blk] = _vptr_4[_blk];
                        uint32_t* _vpairs_4 = reinterpret_cast<uint32_t*>(&_vld_4[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) << 16);
                            (&zs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_4[_pair]) & 0xffff0000u);
                        }
                    }
                }
            } else {
                uint32_t tq = (heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)src), heads.multiplier) >> heads.shift_right) : (uint32_t)((unsigned int)src);
                uint32_t th = (uint32_t)((unsigned int)src) - tq * (uint32_t)(heads.divisor);
                {
                    const uint4* _vptr_5 = reinterpret_cast<const uint4*>(dy + ((long long)tq * dy_tok + (long long)th * dy_head + (long long)col) + 0);
                    uint4 _vld_5[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_5[_blk] = _vptr_5[_blk];
                        uint32_t* _vpairs_5 = reinterpret_cast<uint32_t*>(&_vld_5[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&gs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) << 16);
                            (&gs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_5[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + ((long long)tq * x_tok + (long long)th * x_head + (long long)col) + 0);
                    uint4 _vld_6[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_6[_blk] = _vptr_6[_blk];
                        uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&xs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                            (&xs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                        }
                    }
                }
                {
                    const uint4* _vptr_7 = reinterpret_cast<const uint4*>(z + ((long long)tq * z_tok + (long long)th * z_head + (long long)col) + 0);
                    uint4 _vld_7[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_7[_blk] = _vptr_7[_blk];
                        uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&zs[0 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                            (&zs[0 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
            if (u == padded_block) {
                float total = 0.0f;
                total = __fmaf_rn(xs[0], xs[0], total);
                total = __fmaf_rn(xs[1], xs[1], total);
                total = __fmaf_rn(xs[2], xs[2], total);
                total = __fmaf_rn(xs[3], xs[3], total);
                total = __fmaf_rn(xs[4], xs[4], total);
                total = __fmaf_rn(xs[5], xs[5], total);
                total = __fmaf_rn(xs[6], xs[6], total);
                total = __fmaf_rn(xs[7], xs[7], total);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, total, 1);
                total = total + _shfl_xor_sync_8;
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
                total = total + _shfl_xor_sync_9;
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, total, 4);
                total = total + _shfl_xor_sync_10;
                float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, total, 8);
                total = total + _shfl_xor_sync_11;
                float _sqrt_12;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_12) : "f"(total * 0.0078125f + eps));
                float rstd = __frcp_rn(_sqrt_12);
                unsigned int keep = 4294967295u;
                unsigned int sign = 2147483648u;
                keep = ((row < rows) ? keep : sign);
                float dot = 0.0f;
                int fast = 0;
                float _neg_13;
                asm("neg.f32 %0, %1;" : "=f"(_neg_13) : "f"(zs[0]));
                float _neg_14;
                asm("neg.f32 %0, %1;" : "=f"(_neg_14) : "f"(zs[1]));
                float _neg_15;
                asm("neg.f32 %0, %1;" : "=f"(_neg_15) : "f"(zs[2]));
                float mz;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz) : "f"(_neg_13), "f"(_neg_14), "f"(_neg_15));
                float _abs_17;
                asm("abs.f32 %0, %1;" : "=f"(_abs_17) : "f"(zs[0]));
                float _neg_16;
                asm("neg.f32 %0, %1;" : "=f"(_neg_16) : "f"(_abs_17));
                float _abs_19;
                asm("abs.f32 %0, %1;" : "=f"(_abs_19) : "f"(zs[1]));
                float _neg_18;
                asm("neg.f32 %0, %1;" : "=f"(_neg_18) : "f"(_abs_19));
                float _abs_21;
                asm("abs.f32 %0, %1;" : "=f"(_abs_21) : "f"(zs[2]));
                float _neg_20;
                asm("neg.f32 %0, %1;" : "=f"(_neg_20) : "f"(_abs_21));
                float ma;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma) : "f"(_neg_16), "f"(_neg_18), "f"(_neg_20));
                float _neg_23;
                asm("neg.f32 %0, %1;" : "=f"(_neg_23) : "f"(zs[3]));
                float _neg_24;
                asm("neg.f32 %0, %1;" : "=f"(_neg_24) : "f"(zs[4]));
                float _max3_22;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(mz), "f"(_neg_23), "f"(_neg_24));
                mz = _max3_22;
                float _abs_27;
                asm("abs.f32 %0, %1;" : "=f"(_abs_27) : "f"(zs[3]));
                float _neg_26;
                asm("neg.f32 %0, %1;" : "=f"(_neg_26) : "f"(_abs_27));
                float _abs_29;
                asm("abs.f32 %0, %1;" : "=f"(_abs_29) : "f"(zs[4]));
                float _neg_28;
                asm("neg.f32 %0, %1;" : "=f"(_neg_28) : "f"(_abs_29));
                float _max3_25;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(ma), "f"(_neg_26), "f"(_neg_28));
                ma = _max3_25;
                float _neg_31;
                asm("neg.f32 %0, %1;" : "=f"(_neg_31) : "f"(zs[5]));
                float _neg_32;
                asm("neg.f32 %0, %1;" : "=f"(_neg_32) : "f"(zs[6]));
                float _max3_30;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(mz), "f"(_neg_31), "f"(_neg_32));
                mz = _max3_30;
                float _abs_35;
                asm("abs.f32 %0, %1;" : "=f"(_abs_35) : "f"(zs[5]));
                float _neg_34;
                asm("neg.f32 %0, %1;" : "=f"(_neg_34) : "f"(_abs_35));
                float _abs_37;
                asm("abs.f32 %0, %1;" : "=f"(_abs_37) : "f"(zs[6]));
                float _neg_36;
                asm("neg.f32 %0, %1;" : "=f"(_neg_36) : "f"(_abs_37));
                float _max3_33;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(ma), "f"(_neg_34), "f"(_neg_36));
                ma = _max3_33;
                float _neg_39;
                asm("neg.f32 %0, %1;" : "=f"(_neg_39) : "f"(zs[7]));
                float _max_38;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(mz), "f"(_neg_39));
                mz = _max_38;
                float _abs_42;
                asm("abs.f32 %0, %1;" : "=f"(_abs_42) : "f"(zs[7]));
                float _neg_41;
                asm("neg.f32 %0, %1;" : "=f"(_neg_41) : "f"(_abs_42));
                float _max_40;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(ma), "f"(_neg_41));
                ma = _max_40;
                float _abs_43;
                asm("abs.f32 %0, %1;" : "=f"(_abs_43) : "f"(zs[0]));
                float _abs_44;
                asm("abs.f32 %0, %1;" : "=f"(_abs_44) : "f"(zs[1]));
                float _abs_45;
                asm("abs.f32 %0, %1;" : "=f"(_abs_45) : "f"(zs[2]));
                float mp;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp) : "f"(_abs_43), "f"(_abs_44), "f"(_abs_45));
                float _abs_47;
                asm("abs.f32 %0, %1;" : "=f"(_abs_47) : "f"(zs[3]));
                float _abs_48;
                asm("abs.f32 %0, %1;" : "=f"(_abs_48) : "f"(zs[4]));
                float _max3_46;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_46) : "f"(mp), "f"(_abs_47), "f"(_abs_48));
                mp = _max3_46;
                float _abs_50;
                asm("abs.f32 %0, %1;" : "=f"(_abs_50) : "f"(zs[5]));
                float _abs_51;
                asm("abs.f32 %0, %1;" : "=f"(_abs_51) : "f"(zs[6]));
                float _max3_49;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(mp), "f"(_abs_50), "f"(_abs_51));
                mp = _max3_49;
                float _abs_53;
                asm("abs.f32 %0, %1;" : "=f"(_abs_53) : "f"(zs[7]));
                float _max_52;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(mp), "f"(_abs_53));
                mp = _max_52;
                if (mz <= 41.0f) {
                    if (ma <= -1.1754944e-38f) {
                        if (mp <= 3.4028235e+38f) {
                            fast = 1;
                        }
                    }
                }
                if (fast != 0) {
                    float2 rstd2 = make_float2(rstd, rstd);
                    float2 z2_0 = make_float2(zs[0], zs[1]);
                    float ta_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_0) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_0;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_0) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_54;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(ta_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_55;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(tb_0), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_0 = make_float2(_fma_54, _fma_55);
                    float2 r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_0));
                    r2_0 = fma_f32x2_rn_noftz(z2_0, make_float2(-1.925963e-08f, -1.925963e-08f), r2_0);
                    float2 nscale_0 = make_float2(__uint_as_float((__as_u32(t2_0.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_0.y) << 23) + 2147483648u));
                    float _exp2_56 = approx_exp2(r2_0.x);
                    float _exp2_57 = approx_exp2(r2_0.y);
                    float2 nden_0 = fma_f32x2_rn_noftz(nscale_0, make_float2(_exp2_56, _exp2_57), make_float2(-1.0f, -1.0f));
                    float _neg_59;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_59) : "f"(nden_0.x));
                    float _rcp_58 = approx_rcp(_neg_59);
                    float _neg_61;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_61) : "f"(nden_0.y));
                    float _rcp_60 = approx_rcp(_neg_61);
                    float2 r0_0 = make_float2(_rcp_58, _rcp_60);
                    float2 r1_0 = fma_f32x2_rn_noftz(r0_0, fma_f32x2_rn_noftz(nden_0, r0_0, make_float2(1.0f, 1.0f)), r0_0);
                    float2 sig2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, r1_0, make_float2(1.0f, 1.0f)), r1_0);
                    float2 q0_0 = fma_f32x2_rn_noftz(z2_0, r1_0, make_float2(0.0f, 0.0f));
                    float2 gate2_0 = fma_f32x2_rn_noftz(r1_0, fma_f32x2_rn_noftz(nden_0, q0_0, z2_0), q0_0);
                    float2 g2_0 = make_float2(gs[0], gs[1]);
                    float2 w2_0 = make_float2(wv[0], wv[1]);
                    float2 _f32x2_mul_a_62 = make_float2(xs[0], xs[1]);
                    float2 n2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_62), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_63 = mul_f32x2_noftz(mul_f32x2_noftz(g2_0, mul_f32x2_noftz(n2_0, w2_0)), sig2_0);
                    float2 _f32x2_mul_b_64 = fma_f32x2_rn_noftz(z2_0, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_0), make_float2(1.0f, 1.0f));
                    float2 dz2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_63), "l"(*(const unsigned long long*)&_f32x2_mul_b_64));
                    float2 a2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_0) : "l"(*(const unsigned long long*)&g2_0), "l"(*(const unsigned long long*)&gate2_0));
                    float2 dn2_0;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_0) : "l"(*(const unsigned long long*)&a2_0), "l"(*(const unsigned long long*)&w2_0));
                    float2 acc2_0 = make_float2(acc[0], acc[1]);
                    acc2_0 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_0.x) & keep), __uint_as_float(__as_u32(a2_0.y) & keep)), make_float2(__uint_as_float(__as_u32(n2_0.x) & keep), __uint_as_float(__as_u32(n2_0.y) & keep)), acc2_0);
                    acc[0] = acc2_0.x;
                    acc[1] = acc2_0.y;
                    zs[0] = dz2_0.x;
                    zs[1] = dz2_0.y;
                    gs[0] = dn2_0.x;
                    gs[1] = dn2_0.y;
                    float2 z2_1 = make_float2(zs[2], zs[3]);
                    float ta_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_1) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_1;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_1) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_65;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(ta_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_66;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_66) : "f"(tb_1), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_1 = make_float2(_fma_65, _fma_66);
                    float2 r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_1));
                    r2_1 = fma_f32x2_rn_noftz(z2_1, make_float2(-1.925963e-08f, -1.925963e-08f), r2_1);
                    float2 nscale_1 = make_float2(__uint_as_float((__as_u32(t2_1.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_1.y) << 23) + 2147483648u));
                    float _exp2_67 = approx_exp2(r2_1.x);
                    float _exp2_68 = approx_exp2(r2_1.y);
                    float2 nden_1 = fma_f32x2_rn_noftz(nscale_1, make_float2(_exp2_67, _exp2_68), make_float2(-1.0f, -1.0f));
                    float _neg_70;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_70) : "f"(nden_1.x));
                    float _rcp_69 = approx_rcp(_neg_70);
                    float _neg_72;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_72) : "f"(nden_1.y));
                    float _rcp_71 = approx_rcp(_neg_72);
                    float2 r0_1 = make_float2(_rcp_69, _rcp_71);
                    float2 r1_1 = fma_f32x2_rn_noftz(r0_1, fma_f32x2_rn_noftz(nden_1, r0_1, make_float2(1.0f, 1.0f)), r0_1);
                    float2 sig2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, r1_1, make_float2(1.0f, 1.0f)), r1_1);
                    float2 q0_1 = fma_f32x2_rn_noftz(z2_1, r1_1, make_float2(0.0f, 0.0f));
                    float2 gate2_1 = fma_f32x2_rn_noftz(r1_1, fma_f32x2_rn_noftz(nden_1, q0_1, z2_1), q0_1);
                    float2 g2_1 = make_float2(gs[2], gs[3]);
                    float2 w2_1 = make_float2(wv[2], wv[3]);
                    float2 _f32x2_mul_a_73 = make_float2(xs[2], xs[3]);
                    float2 n2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_73), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_74 = mul_f32x2_noftz(mul_f32x2_noftz(g2_1, mul_f32x2_noftz(n2_1, w2_1)), sig2_1);
                    float2 _f32x2_mul_b_75 = fma_f32x2_rn_noftz(z2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_1), make_float2(1.0f, 1.0f));
                    float2 dz2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_74), "l"(*(const unsigned long long*)&_f32x2_mul_b_75));
                    float2 a2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_1) : "l"(*(const unsigned long long*)&g2_1), "l"(*(const unsigned long long*)&gate2_1));
                    float2 dn2_1;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_1) : "l"(*(const unsigned long long*)&a2_1), "l"(*(const unsigned long long*)&w2_1));
                    float2 acc2_1 = make_float2(acc[2], acc[3]);
                    acc2_1 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_1.x) & keep), __uint_as_float(__as_u32(a2_1.y) & keep)), make_float2(__uint_as_float(__as_u32(n2_1.x) & keep), __uint_as_float(__as_u32(n2_1.y) & keep)), acc2_1);
                    acc[2] = acc2_1.x;
                    acc[3] = acc2_1.y;
                    zs[2] = dz2_1.x;
                    zs[3] = dz2_1.y;
                    gs[2] = dn2_1.x;
                    gs[3] = dn2_1.y;
                    float2 z2_2 = make_float2(zs[4], zs[5]);
                    float ta_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_2) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_2;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_2) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_76;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(ta_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_77;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(tb_2), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_2 = make_float2(_fma_76, _fma_77);
                    float2 r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_2));
                    r2_2 = fma_f32x2_rn_noftz(z2_2, make_float2(-1.925963e-08f, -1.925963e-08f), r2_2);
                    float2 nscale_2 = make_float2(__uint_as_float((__as_u32(t2_2.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_2.y) << 23) + 2147483648u));
                    float _exp2_78 = approx_exp2(r2_2.x);
                    float _exp2_79 = approx_exp2(r2_2.y);
                    float2 nden_2 = fma_f32x2_rn_noftz(nscale_2, make_float2(_exp2_78, _exp2_79), make_float2(-1.0f, -1.0f));
                    float _neg_81;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_81) : "f"(nden_2.x));
                    float _rcp_80 = approx_rcp(_neg_81);
                    float _neg_83;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_83) : "f"(nden_2.y));
                    float _rcp_82 = approx_rcp(_neg_83);
                    float2 r0_2 = make_float2(_rcp_80, _rcp_82);
                    float2 r1_2 = fma_f32x2_rn_noftz(r0_2, fma_f32x2_rn_noftz(nden_2, r0_2, make_float2(1.0f, 1.0f)), r0_2);
                    float2 sig2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, r1_2, make_float2(1.0f, 1.0f)), r1_2);
                    float2 q0_2 = fma_f32x2_rn_noftz(z2_2, r1_2, make_float2(0.0f, 0.0f));
                    float2 gate2_2 = fma_f32x2_rn_noftz(r1_2, fma_f32x2_rn_noftz(nden_2, q0_2, z2_2), q0_2);
                    float2 g2_2 = make_float2(gs[4], gs[5]);
                    float2 w2_2 = make_float2(wv[4], wv[5]);
                    float2 _f32x2_mul_a_84 = make_float2(xs[4], xs[5]);
                    float2 n2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_84), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_85 = mul_f32x2_noftz(mul_f32x2_noftz(g2_2, mul_f32x2_noftz(n2_2, w2_2)), sig2_2);
                    float2 _f32x2_mul_b_86 = fma_f32x2_rn_noftz(z2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_2), make_float2(1.0f, 1.0f));
                    float2 dz2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_85), "l"(*(const unsigned long long*)&_f32x2_mul_b_86));
                    float2 a2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_2) : "l"(*(const unsigned long long*)&g2_2), "l"(*(const unsigned long long*)&gate2_2));
                    float2 dn2_2;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_2) : "l"(*(const unsigned long long*)&a2_2), "l"(*(const unsigned long long*)&w2_2));
                    float2 acc2_2 = make_float2(acc[4], acc[5]);
                    acc2_2 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_2.x) & keep), __uint_as_float(__as_u32(a2_2.y) & keep)), make_float2(__uint_as_float(__as_u32(n2_2.x) & keep), __uint_as_float(__as_u32(n2_2.y) & keep)), acc2_2);
                    acc[4] = acc2_2.x;
                    acc[5] = acc2_2.y;
                    zs[4] = dz2_2.x;
                    zs[5] = dz2_2.y;
                    gs[4] = dn2_2.x;
                    gs[5] = dn2_2.y;
                    float2 z2_3 = make_float2(zs[6], zs[7]);
                    float ta_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_3) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_3;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_3) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_87;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(ta_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_88;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(tb_3), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_3 = make_float2(_fma_87, _fma_88);
                    float2 r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_3));
                    r2_3 = fma_f32x2_rn_noftz(z2_3, make_float2(-1.925963e-08f, -1.925963e-08f), r2_3);
                    float2 nscale_3 = make_float2(__uint_as_float((__as_u32(t2_3.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_3.y) << 23) + 2147483648u));
                    float _exp2_89 = approx_exp2(r2_3.x);
                    float _exp2_90 = approx_exp2(r2_3.y);
                    float2 nden_3 = fma_f32x2_rn_noftz(nscale_3, make_float2(_exp2_89, _exp2_90), make_float2(-1.0f, -1.0f));
                    float _neg_92;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_92) : "f"(nden_3.x));
                    float _rcp_91 = approx_rcp(_neg_92);
                    float _neg_94;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_94) : "f"(nden_3.y));
                    float _rcp_93 = approx_rcp(_neg_94);
                    float2 r0_3 = make_float2(_rcp_91, _rcp_93);
                    float2 r1_3 = fma_f32x2_rn_noftz(r0_3, fma_f32x2_rn_noftz(nden_3, r0_3, make_float2(1.0f, 1.0f)), r0_3);
                    float2 sig2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, r1_3, make_float2(1.0f, 1.0f)), r1_3);
                    float2 q0_3 = fma_f32x2_rn_noftz(z2_3, r1_3, make_float2(0.0f, 0.0f));
                    float2 gate2_3 = fma_f32x2_rn_noftz(r1_3, fma_f32x2_rn_noftz(nden_3, q0_3, z2_3), q0_3);
                    float2 g2_3 = make_float2(gs[6], gs[7]);
                    float2 w2_3 = make_float2(wv[6], wv[7]);
                    float2 _f32x2_mul_a_95 = make_float2(xs[6], xs[7]);
                    float2 n2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_95), "l"(*(const unsigned long long*)&rstd2));
                    float2 _f32x2_mul_a_96 = mul_f32x2_noftz(mul_f32x2_noftz(g2_3, mul_f32x2_noftz(n2_3, w2_3)), sig2_3);
                    float2 _f32x2_mul_b_97 = fma_f32x2_rn_noftz(z2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_3), make_float2(1.0f, 1.0f));
                    float2 dz2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_96), "l"(*(const unsigned long long*)&_f32x2_mul_b_97));
                    float2 a2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_3) : "l"(*(const unsigned long long*)&g2_3), "l"(*(const unsigned long long*)&gate2_3));
                    float2 dn2_3;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_3) : "l"(*(const unsigned long long*)&a2_3), "l"(*(const unsigned long long*)&w2_3));
                    float2 acc2_3 = make_float2(acc[6], acc[7]);
                    acc2_3 = fma_f32x2_rn_noftz(make_float2(__uint_as_float(__as_u32(a2_3.x) & keep), __uint_as_float(__as_u32(a2_3.y) & keep)), make_float2(__uint_as_float(__as_u32(n2_3.x) & keep), __uint_as_float(__as_u32(n2_3.y) & keep)), acc2_3);
                    acc[6] = acc2_3.x;
                    acc[7] = acc2_3.y;
                    zs[6] = dz2_3.x;
                    zs[7] = dz2_3.y;
                    gs[6] = dn2_3.x;
                    gs[7] = dn2_3.y;
                    dot = __fmaf_rn(gs[0], xs[0], dot);
                    dot = __fmaf_rn(gs[1], xs[1], dot);
                    dot = __fmaf_rn(gs[2], xs[2], dot);
                    dot = __fmaf_rn(gs[3], xs[3], dot);
                    dot = __fmaf_rn(gs[4], xs[4], dot);
                    dot = __fmaf_rn(gs[5], xs[5], dot);
                    dot = __fmaf_rn(gs[6], xs[6], dot);
                    dot = __fmaf_rn(gs[7], xs[7], dot);
                } else {
                    float _exp_98 = __nv_expf(zs[0] * -1.0f);
                    float den_0 = 1.0f + _exp_98;
                    float sig_0 = __fdiv_rn(1.0f, den_0);
                    float gate_0 = __fdiv_rn(zs[0], den_0);
                    float nj_0 = xs[0] * rstd;
                    float aj_0 = gs[0] * gate_0;
                    float dzj_0 = gs[0] * (nj_0 * wv[0]) * sig_0 * __fmaf_rn(zs[0], 1.0f - sig_0, 1.0f);
                    float dnj_0 = aj_0 * wv[0];
                    dot = __fmaf_rn(dnj_0, xs[0], dot);
                    acc[0] = __fmaf_rn(__uint_as_float(__as_u32(aj_0) & keep), __uint_as_float(__as_u32(nj_0) & keep), acc[0]);
                    zs[0] = dzj_0;
                    gs[0] = dnj_0;
                    float _exp_99 = __nv_expf(zs[1] * -1.0f);
                    float den_1 = 1.0f + _exp_99;
                    float sig_1 = __fdiv_rn(1.0f, den_1);
                    float gate_1 = __fdiv_rn(zs[1], den_1);
                    float nj_1 = xs[1] * rstd;
                    float aj_1 = gs[1] * gate_1;
                    float dzj_1 = gs[1] * (nj_1 * wv[1]) * sig_1 * __fmaf_rn(zs[1], 1.0f - sig_1, 1.0f);
                    float dnj_1 = aj_1 * wv[1];
                    dot = __fmaf_rn(dnj_1, xs[1], dot);
                    acc[1] = __fmaf_rn(__uint_as_float(__as_u32(aj_1) & keep), __uint_as_float(__as_u32(nj_1) & keep), acc[1]);
                    zs[1] = dzj_1;
                    gs[1] = dnj_1;
                    float _exp_100 = __nv_expf(zs[2] * -1.0f);
                    float den_2 = 1.0f + _exp_100;
                    float sig_2 = __fdiv_rn(1.0f, den_2);
                    float gate_2 = __fdiv_rn(zs[2], den_2);
                    float nj_2 = xs[2] * rstd;
                    float aj_2 = gs[2] * gate_2;
                    float dzj_2 = gs[2] * (nj_2 * wv[2]) * sig_2 * __fmaf_rn(zs[2], 1.0f - sig_2, 1.0f);
                    float dnj_2 = aj_2 * wv[2];
                    dot = __fmaf_rn(dnj_2, xs[2], dot);
                    acc[2] = __fmaf_rn(__uint_as_float(__as_u32(aj_2) & keep), __uint_as_float(__as_u32(nj_2) & keep), acc[2]);
                    zs[2] = dzj_2;
                    gs[2] = dnj_2;
                    float _exp_101 = __nv_expf(zs[3] * -1.0f);
                    float den_3 = 1.0f + _exp_101;
                    float sig_3 = __fdiv_rn(1.0f, den_3);
                    float gate_3 = __fdiv_rn(zs[3], den_3);
                    float nj_3 = xs[3] * rstd;
                    float aj_3 = gs[3] * gate_3;
                    float dzj_3 = gs[3] * (nj_3 * wv[3]) * sig_3 * __fmaf_rn(zs[3], 1.0f - sig_3, 1.0f);
                    float dnj_3 = aj_3 * wv[3];
                    dot = __fmaf_rn(dnj_3, xs[3], dot);
                    acc[3] = __fmaf_rn(__uint_as_float(__as_u32(aj_3) & keep), __uint_as_float(__as_u32(nj_3) & keep), acc[3]);
                    zs[3] = dzj_3;
                    gs[3] = dnj_3;
                    float _exp_102 = __nv_expf(zs[4] * -1.0f);
                    float den_4 = 1.0f + _exp_102;
                    float sig_4 = __fdiv_rn(1.0f, den_4);
                    float gate_4 = __fdiv_rn(zs[4], den_4);
                    float nj_4 = xs[4] * rstd;
                    float aj_4 = gs[4] * gate_4;
                    float dzj_4 = gs[4] * (nj_4 * wv[4]) * sig_4 * __fmaf_rn(zs[4], 1.0f - sig_4, 1.0f);
                    float dnj_4 = aj_4 * wv[4];
                    dot = __fmaf_rn(dnj_4, xs[4], dot);
                    acc[4] = __fmaf_rn(__uint_as_float(__as_u32(aj_4) & keep), __uint_as_float(__as_u32(nj_4) & keep), acc[4]);
                    zs[4] = dzj_4;
                    gs[4] = dnj_4;
                    float _exp_103 = __nv_expf(zs[5] * -1.0f);
                    float den_5 = 1.0f + _exp_103;
                    float sig_5 = __fdiv_rn(1.0f, den_5);
                    float gate_5 = __fdiv_rn(zs[5], den_5);
                    float nj_5 = xs[5] * rstd;
                    float aj_5 = gs[5] * gate_5;
                    float dzj_5 = gs[5] * (nj_5 * wv[5]) * sig_5 * __fmaf_rn(zs[5], 1.0f - sig_5, 1.0f);
                    float dnj_5 = aj_5 * wv[5];
                    dot = __fmaf_rn(dnj_5, xs[5], dot);
                    acc[5] = __fmaf_rn(__uint_as_float(__as_u32(aj_5) & keep), __uint_as_float(__as_u32(nj_5) & keep), acc[5]);
                    zs[5] = dzj_5;
                    gs[5] = dnj_5;
                    float _exp_104 = __nv_expf(zs[6] * -1.0f);
                    float den_6 = 1.0f + _exp_104;
                    float sig_6 = __fdiv_rn(1.0f, den_6);
                    float gate_6 = __fdiv_rn(zs[6], den_6);
                    float nj_6 = xs[6] * rstd;
                    float aj_6 = gs[6] * gate_6;
                    float dzj_6 = gs[6] * (nj_6 * wv[6]) * sig_6 * __fmaf_rn(zs[6], 1.0f - sig_6, 1.0f);
                    float dnj_6 = aj_6 * wv[6];
                    dot = __fmaf_rn(dnj_6, xs[6], dot);
                    acc[6] = __fmaf_rn(__uint_as_float(__as_u32(aj_6) & keep), __uint_as_float(__as_u32(nj_6) & keep), acc[6]);
                    zs[6] = dzj_6;
                    gs[6] = dnj_6;
                    float _exp_105 = __nv_expf(zs[7] * -1.0f);
                    float den_7 = 1.0f + _exp_105;
                    float sig_7 = __fdiv_rn(1.0f, den_7);
                    float gate_7 = __fdiv_rn(zs[7], den_7);
                    float nj_7 = xs[7] * rstd;
                    float aj_7 = gs[7] * gate_7;
                    float dzj_7 = gs[7] * (nj_7 * wv[7]) * sig_7 * __fmaf_rn(zs[7], 1.0f - sig_7, 1.0f);
                    float dnj_7 = aj_7 * wv[7];
                    dot = __fmaf_rn(dnj_7, xs[7], dot);
                    acc[7] = __fmaf_rn(__uint_as_float(__as_u32(aj_7) & keep), __uint_as_float(__as_u32(nj_7) & keep), acc[7]);
                    zs[7] = dzj_7;
                    gs[7] = dnj_7;
                }
                float _shfl_xor_sync_106 = __shfl_xor_sync(0xFFFFFFFF, dot, 1);
                dot = dot + _shfl_xor_sync_106;
                float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, dot, 2);
                dot = dot + _shfl_xor_sync_107;
                float _shfl_xor_sync_108 = __shfl_xor_sync(0xFFFFFFFF, dot, 4);
                dot = dot + _shfl_xor_sync_108;
                float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, dot, 8);
                dot = dot + _shfl_xor_sync_109;
                float coef = rstd * rstd * rstd * (dot * -0.0078125f);
                float2 coef2 = make_float2(coef, coef);
                float2 rstd2b = make_float2(rstd, rstd);
                float2 dx2_0 = fma_f32x2_rn_noftz(make_float2(xs[0], xs[1]), coef2, mul_f32x2_noftz(make_float2(gs[0], gs[1]), rstd2b));
                xs[0] = dx2_0.x;
                xs[1] = dx2_0.y;
                float2 dx2_1 = fma_f32x2_rn_noftz(make_float2(xs[2], xs[3]), coef2, mul_f32x2_noftz(make_float2(gs[2], gs[3]), rstd2b));
                xs[2] = dx2_1.x;
                xs[3] = dx2_1.y;
                float2 dx2_2 = fma_f32x2_rn_noftz(make_float2(xs[4], xs[5]), coef2, mul_f32x2_noftz(make_float2(gs[4], gs[5]), rstd2b));
                xs[4] = dx2_2.x;
                xs[5] = dx2_2.y;
                float2 dx2_3 = fma_f32x2_rn_noftz(make_float2(xs[6], xs[7]), coef2, mul_f32x2_noftz(make_float2(gs[6], gs[7]), rstd2b));
                xs[6] = dx2_3.x;
                xs[7] = dx2_3.y;
                unsigned int packed_dx[4];
                unsigned int packed_dz[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                    packed_dx[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 0], zs[_lp*2+1 + 0]));
                    packed_dz[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                    packed_dx[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 2], zs[_lp*2+1 + 2]));
                    packed_dz[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                    packed_dx[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 4], zs[_lp*2+1 + 4]));
                    packed_dz[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                    packed_dx[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 6], zs[_lp*2+1 + 6]));
                    packed_dz[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row < rows) {
                    long long orow = (long long)row * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow)[0] = reinterpret_cast<int4*>(packed_dx)[0];
                    reinterpret_cast<int4*>(dz + orow)[0] = reinterpret_cast<int4*>(packed_dz)[0];
                }
            } else {
                float total_1 = 0.0f;
                total_1 = __fmaf_rn(xs[0], xs[0], total_1);
                total_1 = __fmaf_rn(xs[1], xs[1], total_1);
                total_1 = __fmaf_rn(xs[2], xs[2], total_1);
                total_1 = __fmaf_rn(xs[3], xs[3], total_1);
                total_1 = __fmaf_rn(xs[4], xs[4], total_1);
                total_1 = __fmaf_rn(xs[5], xs[5], total_1);
                total_1 = __fmaf_rn(xs[6], xs[6], total_1);
                total_1 = __fmaf_rn(xs[7], xs[7], total_1);
                float _shfl_xor_sync_110 = __shfl_xor_sync(0xFFFFFFFF, total_1, 1);
                total_1 = total_1 + _shfl_xor_sync_110;
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, total_1, 2);
                total_1 = total_1 + _shfl_xor_sync_111;
                float _shfl_xor_sync_112 = __shfl_xor_sync(0xFFFFFFFF, total_1, 4);
                total_1 = total_1 + _shfl_xor_sync_112;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, total_1, 8);
                total_1 = total_1 + _shfl_xor_sync_113;
                float _sqrt_114;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_114) : "f"(total_1 * 0.0078125f + eps));
                float rstd_1 = __frcp_rn(_sqrt_114);
                unsigned int keep_1 = 4294967295u;
                float dot_1 = 0.0f;
                int fast_1 = 0;
                float _neg_115;
                asm("neg.f32 %0, %1;" : "=f"(_neg_115) : "f"(zs[0]));
                float _neg_116;
                asm("neg.f32 %0, %1;" : "=f"(_neg_116) : "f"(zs[1]));
                float _neg_117;
                asm("neg.f32 %0, %1;" : "=f"(_neg_117) : "f"(zs[2]));
                float mz_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mz_1) : "f"(_neg_115), "f"(_neg_116), "f"(_neg_117));
                float _abs_119;
                asm("abs.f32 %0, %1;" : "=f"(_abs_119) : "f"(zs[0]));
                float _neg_118;
                asm("neg.f32 %0, %1;" : "=f"(_neg_118) : "f"(_abs_119));
                float _abs_121;
                asm("abs.f32 %0, %1;" : "=f"(_abs_121) : "f"(zs[1]));
                float _neg_120;
                asm("neg.f32 %0, %1;" : "=f"(_neg_120) : "f"(_abs_121));
                float _abs_123;
                asm("abs.f32 %0, %1;" : "=f"(_abs_123) : "f"(zs[2]));
                float _neg_122;
                asm("neg.f32 %0, %1;" : "=f"(_neg_122) : "f"(_abs_123));
                float ma_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(ma_1) : "f"(_neg_118), "f"(_neg_120), "f"(_neg_122));
                float _neg_125;
                asm("neg.f32 %0, %1;" : "=f"(_neg_125) : "f"(zs[3]));
                float _neg_126;
                asm("neg.f32 %0, %1;" : "=f"(_neg_126) : "f"(zs[4]));
                float _max3_124;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_124) : "f"(mz_1), "f"(_neg_125), "f"(_neg_126));
                mz_1 = _max3_124;
                float _abs_129;
                asm("abs.f32 %0, %1;" : "=f"(_abs_129) : "f"(zs[3]));
                float _neg_128;
                asm("neg.f32 %0, %1;" : "=f"(_neg_128) : "f"(_abs_129));
                float _abs_131;
                asm("abs.f32 %0, %1;" : "=f"(_abs_131) : "f"(zs[4]));
                float _neg_130;
                asm("neg.f32 %0, %1;" : "=f"(_neg_130) : "f"(_abs_131));
                float _max3_127;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_127) : "f"(ma_1), "f"(_neg_128), "f"(_neg_130));
                ma_1 = _max3_127;
                float _neg_133;
                asm("neg.f32 %0, %1;" : "=f"(_neg_133) : "f"(zs[5]));
                float _neg_134;
                asm("neg.f32 %0, %1;" : "=f"(_neg_134) : "f"(zs[6]));
                float _max3_132;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_132) : "f"(mz_1), "f"(_neg_133), "f"(_neg_134));
                mz_1 = _max3_132;
                float _abs_137;
                asm("abs.f32 %0, %1;" : "=f"(_abs_137) : "f"(zs[5]));
                float _neg_136;
                asm("neg.f32 %0, %1;" : "=f"(_neg_136) : "f"(_abs_137));
                float _abs_139;
                asm("abs.f32 %0, %1;" : "=f"(_abs_139) : "f"(zs[6]));
                float _neg_138;
                asm("neg.f32 %0, %1;" : "=f"(_neg_138) : "f"(_abs_139));
                float _max3_135;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_135) : "f"(ma_1), "f"(_neg_136), "f"(_neg_138));
                ma_1 = _max3_135;
                float _neg_141;
                asm("neg.f32 %0, %1;" : "=f"(_neg_141) : "f"(zs[7]));
                float _max_140;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_140) : "f"(mz_1), "f"(_neg_141));
                mz_1 = _max_140;
                float _abs_144;
                asm("abs.f32 %0, %1;" : "=f"(_abs_144) : "f"(zs[7]));
                float _neg_143;
                asm("neg.f32 %0, %1;" : "=f"(_neg_143) : "f"(_abs_144));
                float _max_142;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_142) : "f"(ma_1), "f"(_neg_143));
                ma_1 = _max_142;
                float _abs_145;
                asm("abs.f32 %0, %1;" : "=f"(_abs_145) : "f"(zs[0]));
                float _abs_146;
                asm("abs.f32 %0, %1;" : "=f"(_abs_146) : "f"(zs[1]));
                float _abs_147;
                asm("abs.f32 %0, %1;" : "=f"(_abs_147) : "f"(zs[2]));
                float mp_1;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(mp_1) : "f"(_abs_145), "f"(_abs_146), "f"(_abs_147));
                float _abs_149;
                asm("abs.f32 %0, %1;" : "=f"(_abs_149) : "f"(zs[3]));
                float _abs_150;
                asm("abs.f32 %0, %1;" : "=f"(_abs_150) : "f"(zs[4]));
                float _max3_148;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_148) : "f"(mp_1), "f"(_abs_149), "f"(_abs_150));
                mp_1 = _max3_148;
                float _abs_152;
                asm("abs.f32 %0, %1;" : "=f"(_abs_152) : "f"(zs[5]));
                float _abs_153;
                asm("abs.f32 %0, %1;" : "=f"(_abs_153) : "f"(zs[6]));
                float _max3_151;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.NaN.f32 %0, %1, %2, %3;" : "=f"(_max3_151) : "f"(mp_1), "f"(_abs_152), "f"(_abs_153));
                mp_1 = _max3_151;
                float _abs_155;
                asm("abs.f32 %0, %1;" : "=f"(_abs_155) : "f"(zs[7]));
                float _max_154;
                asm("max.NaN.f32 %0, %1, %2;" : "=f"(_max_154) : "f"(mp_1), "f"(_abs_155));
                mp_1 = _max_154;
                if (mz_1 <= 41.0f) {
                    if (ma_1 <= -1.1754944e-38f) {
                        if (mp_1 <= 3.4028235e+38f) {
                            fast_1 = 1;
                        }
                    }
                }
                if (fast_1 != 0) {
                    float2 rstd2_1 = make_float2(rstd_1, rstd_1);
                    float2 z2_4 = make_float2(zs[0], zs[1]);
                    float ta_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_4) : "f"(zs[0]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_4;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_4) : "f"(zs[1]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_156;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(ta_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_157;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(tb_4), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_4 = make_float2(_fma_156, _fma_157);
                    float2 r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_4));
                    r2_4 = fma_f32x2_rn_noftz(z2_4, make_float2(-1.925963e-08f, -1.925963e-08f), r2_4);
                    float2 nscale_4 = make_float2(__uint_as_float((__as_u32(t2_4.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_4.y) << 23) + 2147483648u));
                    float _exp2_158 = approx_exp2(r2_4.x);
                    float _exp2_159 = approx_exp2(r2_4.y);
                    float2 nden_4 = fma_f32x2_rn_noftz(nscale_4, make_float2(_exp2_158, _exp2_159), make_float2(-1.0f, -1.0f));
                    float _neg_161;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_161) : "f"(nden_4.x));
                    float _rcp_160 = approx_rcp(_neg_161);
                    float _neg_163;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_163) : "f"(nden_4.y));
                    float _rcp_162 = approx_rcp(_neg_163);
                    float2 r0_4 = make_float2(_rcp_160, _rcp_162);
                    float2 r1_4 = fma_f32x2_rn_noftz(r0_4, fma_f32x2_rn_noftz(nden_4, r0_4, make_float2(1.0f, 1.0f)), r0_4);
                    float2 sig2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, r1_4, make_float2(1.0f, 1.0f)), r1_4);
                    float2 q0_4 = fma_f32x2_rn_noftz(z2_4, r1_4, make_float2(0.0f, 0.0f));
                    float2 gate2_4 = fma_f32x2_rn_noftz(r1_4, fma_f32x2_rn_noftz(nden_4, q0_4, z2_4), q0_4);
                    float2 g2_4 = make_float2(gs[0], gs[1]);
                    float2 w2_4 = make_float2(wv[0], wv[1]);
                    float2 _f32x2_mul_a_164 = make_float2(xs[0], xs[1]);
                    float2 n2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_164), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_165 = mul_f32x2_noftz(mul_f32x2_noftz(g2_4, mul_f32x2_noftz(n2_4, w2_4)), sig2_4);
                    float2 _f32x2_mul_b_166 = fma_f32x2_rn_noftz(z2_4, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_4), make_float2(1.0f, 1.0f));
                    float2 dz2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_165), "l"(*(const unsigned long long*)&_f32x2_mul_b_166));
                    float2 a2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_4) : "l"(*(const unsigned long long*)&g2_4), "l"(*(const unsigned long long*)&gate2_4));
                    float2 dn2_4;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_4) : "l"(*(const unsigned long long*)&a2_4), "l"(*(const unsigned long long*)&w2_4));
                    float2 acc2_4 = make_float2(acc[0], acc[1]);
                    acc2_4 = fma_f32x2_rn_noftz(a2_4, n2_4, acc2_4);
                    acc[0] = acc2_4.x;
                    acc[1] = acc2_4.y;
                    zs[0] = dz2_4.x;
                    zs[1] = dz2_4.y;
                    gs[0] = dn2_4.x;
                    gs[1] = dn2_4.y;
                    float2 z2_5 = make_float2(zs[2], zs[3]);
                    float ta_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_5) : "f"(zs[2]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_5;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_5) : "f"(zs[3]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_167;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(ta_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_168;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(tb_5), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_5 = make_float2(_fma_167, _fma_168);
                    float2 r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_5));
                    r2_5 = fma_f32x2_rn_noftz(z2_5, make_float2(-1.925963e-08f, -1.925963e-08f), r2_5);
                    float2 nscale_5 = make_float2(__uint_as_float((__as_u32(t2_5.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_5.y) << 23) + 2147483648u));
                    float _exp2_169 = approx_exp2(r2_5.x);
                    float _exp2_170 = approx_exp2(r2_5.y);
                    float2 nden_5 = fma_f32x2_rn_noftz(nscale_5, make_float2(_exp2_169, _exp2_170), make_float2(-1.0f, -1.0f));
                    float _neg_172;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_172) : "f"(nden_5.x));
                    float _rcp_171 = approx_rcp(_neg_172);
                    float _neg_174;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_174) : "f"(nden_5.y));
                    float _rcp_173 = approx_rcp(_neg_174);
                    float2 r0_5 = make_float2(_rcp_171, _rcp_173);
                    float2 r1_5 = fma_f32x2_rn_noftz(r0_5, fma_f32x2_rn_noftz(nden_5, r0_5, make_float2(1.0f, 1.0f)), r0_5);
                    float2 sig2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, r1_5, make_float2(1.0f, 1.0f)), r1_5);
                    float2 q0_5 = fma_f32x2_rn_noftz(z2_5, r1_5, make_float2(0.0f, 0.0f));
                    float2 gate2_5 = fma_f32x2_rn_noftz(r1_5, fma_f32x2_rn_noftz(nden_5, q0_5, z2_5), q0_5);
                    float2 g2_5 = make_float2(gs[2], gs[3]);
                    float2 w2_5 = make_float2(wv[2], wv[3]);
                    float2 _f32x2_mul_a_175 = make_float2(xs[2], xs[3]);
                    float2 n2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_175), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_176 = mul_f32x2_noftz(mul_f32x2_noftz(g2_5, mul_f32x2_noftz(n2_5, w2_5)), sig2_5);
                    float2 _f32x2_mul_b_177 = fma_f32x2_rn_noftz(z2_5, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_5), make_float2(1.0f, 1.0f));
                    float2 dz2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_176), "l"(*(const unsigned long long*)&_f32x2_mul_b_177));
                    float2 a2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_5) : "l"(*(const unsigned long long*)&g2_5), "l"(*(const unsigned long long*)&gate2_5));
                    float2 dn2_5;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_5) : "l"(*(const unsigned long long*)&a2_5), "l"(*(const unsigned long long*)&w2_5));
                    float2 acc2_5 = make_float2(acc[2], acc[3]);
                    acc2_5 = fma_f32x2_rn_noftz(a2_5, n2_5, acc2_5);
                    acc[2] = acc2_5.x;
                    acc[3] = acc2_5.y;
                    zs[2] = dz2_5.x;
                    zs[3] = dz2_5.y;
                    gs[2] = dn2_5.x;
                    gs[3] = dn2_5.y;
                    float2 z2_6 = make_float2(zs[4], zs[5]);
                    float ta_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_6) : "f"(zs[4]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_6;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_6) : "f"(zs[5]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_178;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_178) : "f"(ta_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_179;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_179) : "f"(tb_6), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_6 = make_float2(_fma_178, _fma_179);
                    float2 r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_6));
                    r2_6 = fma_f32x2_rn_noftz(z2_6, make_float2(-1.925963e-08f, -1.925963e-08f), r2_6);
                    float2 nscale_6 = make_float2(__uint_as_float((__as_u32(t2_6.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_6.y) << 23) + 2147483648u));
                    float _exp2_180 = approx_exp2(r2_6.x);
                    float _exp2_181 = approx_exp2(r2_6.y);
                    float2 nden_6 = fma_f32x2_rn_noftz(nscale_6, make_float2(_exp2_180, _exp2_181), make_float2(-1.0f, -1.0f));
                    float _neg_183;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_183) : "f"(nden_6.x));
                    float _rcp_182 = approx_rcp(_neg_183);
                    float _neg_185;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_185) : "f"(nden_6.y));
                    float _rcp_184 = approx_rcp(_neg_185);
                    float2 r0_6 = make_float2(_rcp_182, _rcp_184);
                    float2 r1_6 = fma_f32x2_rn_noftz(r0_6, fma_f32x2_rn_noftz(nden_6, r0_6, make_float2(1.0f, 1.0f)), r0_6);
                    float2 sig2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, r1_6, make_float2(1.0f, 1.0f)), r1_6);
                    float2 q0_6 = fma_f32x2_rn_noftz(z2_6, r1_6, make_float2(0.0f, 0.0f));
                    float2 gate2_6 = fma_f32x2_rn_noftz(r1_6, fma_f32x2_rn_noftz(nden_6, q0_6, z2_6), q0_6);
                    float2 g2_6 = make_float2(gs[4], gs[5]);
                    float2 w2_6 = make_float2(wv[4], wv[5]);
                    float2 _f32x2_mul_a_186 = make_float2(xs[4], xs[5]);
                    float2 n2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_186), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_187 = mul_f32x2_noftz(mul_f32x2_noftz(g2_6, mul_f32x2_noftz(n2_6, w2_6)), sig2_6);
                    float2 _f32x2_mul_b_188 = fma_f32x2_rn_noftz(z2_6, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_6), make_float2(1.0f, 1.0f));
                    float2 dz2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_187), "l"(*(const unsigned long long*)&_f32x2_mul_b_188));
                    float2 a2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_6) : "l"(*(const unsigned long long*)&g2_6), "l"(*(const unsigned long long*)&gate2_6));
                    float2 dn2_6;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_6) : "l"(*(const unsigned long long*)&a2_6), "l"(*(const unsigned long long*)&w2_6));
                    float2 acc2_6 = make_float2(acc[4], acc[5]);
                    acc2_6 = fma_f32x2_rn_noftz(a2_6, n2_6, acc2_6);
                    acc[4] = acc2_6.x;
                    acc[5] = acc2_6.y;
                    zs[4] = dz2_6.x;
                    zs[5] = dz2_6.y;
                    gs[4] = dn2_6.x;
                    gs[5] = dn2_6.y;
                    float2 z2_7 = make_float2(zs[6], zs[7]);
                    float ta_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(ta_7) : "f"(zs[6]), "f"(-0.00572498f), "f"(0.5f));
                    float tb_7;
                    asm("fma.rn.sat.f32 %0, %1, %2, %3;" : "=f"(tb_7) : "f"(zs[7]), "f"(-0.00572498f), "f"(0.5f));
                    float _fma_189;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_189) : "f"(ta_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float _fma_190;
                    asm("fma.rm.f32 %0, %1, %2, %3;" : "=f"(_fma_190) : "f"(tb_7), "f"(252.0f), "f"(1.2582913e+07f));
                    float2 t2_7 = make_float2(_fma_189, _fma_190);
                    float2 r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.442695f, -1.442695f), sub_f32x2_noftz(make_float2(1.2583039e+07f, 1.2583039e+07f), t2_7));
                    r2_7 = fma_f32x2_rn_noftz(z2_7, make_float2(-1.925963e-08f, -1.925963e-08f), r2_7);
                    float2 nscale_7 = make_float2(__uint_as_float((__as_u32(t2_7.x) << 23) + 2147483648u), __uint_as_float((__as_u32(t2_7.y) << 23) + 2147483648u));
                    float _exp2_191 = approx_exp2(r2_7.x);
                    float _exp2_192 = approx_exp2(r2_7.y);
                    float2 nden_7 = fma_f32x2_rn_noftz(nscale_7, make_float2(_exp2_191, _exp2_192), make_float2(-1.0f, -1.0f));
                    float _neg_194;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_194) : "f"(nden_7.x));
                    float _rcp_193 = approx_rcp(_neg_194);
                    float _neg_196;
                    asm("neg.f32 %0, %1;" : "=f"(_neg_196) : "f"(nden_7.y));
                    float _rcp_195 = approx_rcp(_neg_196);
                    float2 r0_7 = make_float2(_rcp_193, _rcp_195);
                    float2 r1_7 = fma_f32x2_rn_noftz(r0_7, fma_f32x2_rn_noftz(nden_7, r0_7, make_float2(1.0f, 1.0f)), r0_7);
                    float2 sig2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, r1_7, make_float2(1.0f, 1.0f)), r1_7);
                    float2 q0_7 = fma_f32x2_rn_noftz(z2_7, r1_7, make_float2(0.0f, 0.0f));
                    float2 gate2_7 = fma_f32x2_rn_noftz(r1_7, fma_f32x2_rn_noftz(nden_7, q0_7, z2_7), q0_7);
                    float2 g2_7 = make_float2(gs[6], gs[7]);
                    float2 w2_7 = make_float2(wv[6], wv[7]);
                    float2 _f32x2_mul_a_197 = make_float2(xs[6], xs[7]);
                    float2 n2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&n2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_197), "l"(*(const unsigned long long*)&rstd2_1));
                    float2 _f32x2_mul_a_198 = mul_f32x2_noftz(mul_f32x2_noftz(g2_7, mul_f32x2_noftz(n2_7, w2_7)), sig2_7);
                    float2 _f32x2_mul_b_199 = fma_f32x2_rn_noftz(z2_7, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sig2_7), make_float2(1.0f, 1.0f));
                    float2 dz2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dz2_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_198), "l"(*(const unsigned long long*)&_f32x2_mul_b_199));
                    float2 a2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&a2_7) : "l"(*(const unsigned long long*)&g2_7), "l"(*(const unsigned long long*)&gate2_7));
                    float2 dn2_7;
                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&dn2_7) : "l"(*(const unsigned long long*)&a2_7), "l"(*(const unsigned long long*)&w2_7));
                    float2 acc2_7 = make_float2(acc[6], acc[7]);
                    acc2_7 = fma_f32x2_rn_noftz(a2_7, n2_7, acc2_7);
                    acc[6] = acc2_7.x;
                    acc[7] = acc2_7.y;
                    zs[6] = dz2_7.x;
                    zs[7] = dz2_7.y;
                    gs[6] = dn2_7.x;
                    gs[7] = dn2_7.y;
                    dot_1 = __fmaf_rn(gs[0], xs[0], dot_1);
                    dot_1 = __fmaf_rn(gs[1], xs[1], dot_1);
                    dot_1 = __fmaf_rn(gs[2], xs[2], dot_1);
                    dot_1 = __fmaf_rn(gs[3], xs[3], dot_1);
                    dot_1 = __fmaf_rn(gs[4], xs[4], dot_1);
                    dot_1 = __fmaf_rn(gs[5], xs[5], dot_1);
                    dot_1 = __fmaf_rn(gs[6], xs[6], dot_1);
                    dot_1 = __fmaf_rn(gs[7], xs[7], dot_1);
                } else {
                    float _exp_200 = __nv_expf(zs[0] * -1.0f);
                    float den_8 = 1.0f + _exp_200;
                    float sig_8 = __fdiv_rn(1.0f, den_8);
                    float gate_8 = __fdiv_rn(zs[0], den_8);
                    float nj_8 = xs[0] * rstd_1;
                    float aj_8 = gs[0] * gate_8;
                    float dzj_8 = gs[0] * (nj_8 * wv[0]) * sig_8 * __fmaf_rn(zs[0], 1.0f - sig_8, 1.0f);
                    float dnj_8 = aj_8 * wv[0];
                    dot_1 = __fmaf_rn(dnj_8, xs[0], dot_1);
                    acc[0] = __fmaf_rn(aj_8, nj_8, acc[0]);
                    zs[0] = dzj_8;
                    gs[0] = dnj_8;
                    float _exp_201 = __nv_expf(zs[1] * -1.0f);
                    float den_9 = 1.0f + _exp_201;
                    float sig_9 = __fdiv_rn(1.0f, den_9);
                    float gate_9 = __fdiv_rn(zs[1], den_9);
                    float nj_9 = xs[1] * rstd_1;
                    float aj_9 = gs[1] * gate_9;
                    float dzj_9 = gs[1] * (nj_9 * wv[1]) * sig_9 * __fmaf_rn(zs[1], 1.0f - sig_9, 1.0f);
                    float dnj_9 = aj_9 * wv[1];
                    dot_1 = __fmaf_rn(dnj_9, xs[1], dot_1);
                    acc[1] = __fmaf_rn(aj_9, nj_9, acc[1]);
                    zs[1] = dzj_9;
                    gs[1] = dnj_9;
                    float _exp_202 = __nv_expf(zs[2] * -1.0f);
                    float den_10 = 1.0f + _exp_202;
                    float sig_10 = __fdiv_rn(1.0f, den_10);
                    float gate_10 = __fdiv_rn(zs[2], den_10);
                    float nj_10 = xs[2] * rstd_1;
                    float aj_10 = gs[2] * gate_10;
                    float dzj_10 = gs[2] * (nj_10 * wv[2]) * sig_10 * __fmaf_rn(zs[2], 1.0f - sig_10, 1.0f);
                    float dnj_10 = aj_10 * wv[2];
                    dot_1 = __fmaf_rn(dnj_10, xs[2], dot_1);
                    acc[2] = __fmaf_rn(aj_10, nj_10, acc[2]);
                    zs[2] = dzj_10;
                    gs[2] = dnj_10;
                    float _exp_203 = __nv_expf(zs[3] * -1.0f);
                    float den_11 = 1.0f + _exp_203;
                    float sig_11 = __fdiv_rn(1.0f, den_11);
                    float gate_11 = __fdiv_rn(zs[3], den_11);
                    float nj_11 = xs[3] * rstd_1;
                    float aj_11 = gs[3] * gate_11;
                    float dzj_11 = gs[3] * (nj_11 * wv[3]) * sig_11 * __fmaf_rn(zs[3], 1.0f - sig_11, 1.0f);
                    float dnj_11 = aj_11 * wv[3];
                    dot_1 = __fmaf_rn(dnj_11, xs[3], dot_1);
                    acc[3] = __fmaf_rn(aj_11, nj_11, acc[3]);
                    zs[3] = dzj_11;
                    gs[3] = dnj_11;
                    float _exp_204 = __nv_expf(zs[4] * -1.0f);
                    float den_12 = 1.0f + _exp_204;
                    float sig_12 = __fdiv_rn(1.0f, den_12);
                    float gate_12 = __fdiv_rn(zs[4], den_12);
                    float nj_12 = xs[4] * rstd_1;
                    float aj_12 = gs[4] * gate_12;
                    float dzj_12 = gs[4] * (nj_12 * wv[4]) * sig_12 * __fmaf_rn(zs[4], 1.0f - sig_12, 1.0f);
                    float dnj_12 = aj_12 * wv[4];
                    dot_1 = __fmaf_rn(dnj_12, xs[4], dot_1);
                    acc[4] = __fmaf_rn(aj_12, nj_12, acc[4]);
                    zs[4] = dzj_12;
                    gs[4] = dnj_12;
                    float _exp_205 = __nv_expf(zs[5] * -1.0f);
                    float den_13 = 1.0f + _exp_205;
                    float sig_13 = __fdiv_rn(1.0f, den_13);
                    float gate_13 = __fdiv_rn(zs[5], den_13);
                    float nj_13 = xs[5] * rstd_1;
                    float aj_13 = gs[5] * gate_13;
                    float dzj_13 = gs[5] * (nj_13 * wv[5]) * sig_13 * __fmaf_rn(zs[5], 1.0f - sig_13, 1.0f);
                    float dnj_13 = aj_13 * wv[5];
                    dot_1 = __fmaf_rn(dnj_13, xs[5], dot_1);
                    acc[5] = __fmaf_rn(aj_13, nj_13, acc[5]);
                    zs[5] = dzj_13;
                    gs[5] = dnj_13;
                    float _exp_206 = __nv_expf(zs[6] * -1.0f);
                    float den_14 = 1.0f + _exp_206;
                    float sig_14 = __fdiv_rn(1.0f, den_14);
                    float gate_14 = __fdiv_rn(zs[6], den_14);
                    float nj_14 = xs[6] * rstd_1;
                    float aj_14 = gs[6] * gate_14;
                    float dzj_14 = gs[6] * (nj_14 * wv[6]) * sig_14 * __fmaf_rn(zs[6], 1.0f - sig_14, 1.0f);
                    float dnj_14 = aj_14 * wv[6];
                    dot_1 = __fmaf_rn(dnj_14, xs[6], dot_1);
                    acc[6] = __fmaf_rn(aj_14, nj_14, acc[6]);
                    zs[6] = dzj_14;
                    gs[6] = dnj_14;
                    float _exp_207 = __nv_expf(zs[7] * -1.0f);
                    float den_15 = 1.0f + _exp_207;
                    float sig_15 = __fdiv_rn(1.0f, den_15);
                    float gate_15 = __fdiv_rn(zs[7], den_15);
                    float nj_15 = xs[7] * rstd_1;
                    float aj_15 = gs[7] * gate_15;
                    float dzj_15 = gs[7] * (nj_15 * wv[7]) * sig_15 * __fmaf_rn(zs[7], 1.0f - sig_15, 1.0f);
                    float dnj_15 = aj_15 * wv[7];
                    dot_1 = __fmaf_rn(dnj_15, xs[7], dot_1);
                    acc[7] = __fmaf_rn(aj_15, nj_15, acc[7]);
                    zs[7] = dzj_15;
                    gs[7] = dnj_15;
                }
                float _shfl_xor_sync_208 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 1);
                dot_1 = dot_1 + _shfl_xor_sync_208;
                float _shfl_xor_sync_209 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 2);
                dot_1 = dot_1 + _shfl_xor_sync_209;
                float _shfl_xor_sync_210 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 4);
                dot_1 = dot_1 + _shfl_xor_sync_210;
                float _shfl_xor_sync_211 = __shfl_xor_sync(0xFFFFFFFF, dot_1, 8);
                dot_1 = dot_1 + _shfl_xor_sync_211;
                float coef_1 = rstd_1 * rstd_1 * rstd_1 * (dot_1 * -0.0078125f);
                float2 coef2_1 = make_float2(coef_1, coef_1);
                float2 rstd2b_1 = make_float2(rstd_1, rstd_1);
                float2 dx2_4 = fma_f32x2_rn_noftz(make_float2(xs[0], xs[1]), coef2_1, mul_f32x2_noftz(make_float2(gs[0], gs[1]), rstd2b_1));
                xs[0] = dx2_4.x;
                xs[1] = dx2_4.y;
                float2 dx2_5 = fma_f32x2_rn_noftz(make_float2(xs[2], xs[3]), coef2_1, mul_f32x2_noftz(make_float2(gs[2], gs[3]), rstd2b_1));
                xs[2] = dx2_5.x;
                xs[3] = dx2_5.y;
                float2 dx2_6 = fma_f32x2_rn_noftz(make_float2(xs[4], xs[5]), coef2_1, mul_f32x2_noftz(make_float2(gs[4], gs[5]), rstd2b_1));
                xs[4] = dx2_6.x;
                xs[5] = dx2_6.y;
                float2 dx2_7 = fma_f32x2_rn_noftz(make_float2(xs[6], xs[7]), coef2_1, mul_f32x2_noftz(make_float2(gs[6], gs[7]), rstd2b_1));
                xs[6] = dx2_7.x;
                xs[7] = dx2_7.y;
                unsigned int packed_dx_1[4];
                unsigned int packed_dz_1[4];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 0], xs[_lp*2+1 + 0]));
                    packed_dx_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 0], zs[_lp*2+1 + 0]));
                    packed_dz_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 2], xs[_lp*2+1 + 2]));
                    packed_dx_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 2], zs[_lp*2+1 + 2]));
                    packed_dz_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 4], xs[_lp*2+1 + 4]));
                    packed_dx_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 4], zs[_lp*2+1 + 4]));
                    packed_dz_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(xs[_lp*2 + 6], xs[_lp*2+1 + 6]));
                    packed_dx_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zs[_lp*2 + 6], zs[_lp*2+1 + 6]));
                    packed_dz_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                if (row < rows) {
                    long long orow_1 = (long long)row * 128ll + (long long)col;
                    reinterpret_cast<int4*>(dx + orow_1)[0] = reinterpret_cast<int4*>(packed_dx_1)[0];
                    reinterpret_cast<int4*>(dz + orow_1)[0] = reinterpret_cast<int4*>(packed_dz_1)[0];
                }
            }
        }
        if (weight_grad != 0) {
            fold[i % 2 * 1024 + group * 128 + col] = acc[0];
            fold[i % 2 * 1024 + group * 128 + col + 1] = acc[1];
            fold[i % 2 * 1024 + group * 128 + col + 2] = acc[2];
            fold[i % 2 * 1024 + group * 128 + col + 3] = acc[3];
            fold[i % 2 * 1024 + group * 128 + col + 4] = acc[4];
            fold[i % 2 * 1024 + group * 128 + col + 5] = acc[5];
            fold[i % 2 * 1024 + group * 128 + col + 6] = acc[6];
            fold[i % 2 * 1024 + group * 128 + col + 7] = acc[7];
            __syncthreads();
            float colsum = 0.0f;
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                colsum = colsum + fold[i % 2 * 1024 + g * 128 + tid];
            }
            if ((__as_u32(colsum) & 2139095040) == 2139095040) {
                spec = spec + colsum;
            } else if ((__as_u32(colsum) & 2147483647) != 0) {
                unsigned int bits = __as_u32(colsum);
                unsigned int ex = bits >> 23 & 255;
                unsigned int _max_212 = ((ex) > (1) ? (ex) : (1));
                unsigned int idx = _max_212 - 1 + 38;
                unsigned int _min_213 = ((ex) < (1) ? (ex) : (1));
                unsigned long long m = (unsigned long long)(bits & 8388607 | _min_213 << 23);
                unsigned int wk = idx / 48;
                unsigned int sh = idx - wk * 48;
                long long lo = (long long)(m << (unsigned long long)sh & 281474976710655ull);
                long long hi = (long long)(m >> (unsigned long long)(48 - sh));
                if (bits >> 31 != 0) {
                    lo = -lo;
                    hi = -hi;
                }
                win[(int)wk * 128 + tid] = win[(int)wk * 128 + tid] + lo;
                if (hi != 0ll) {
                    win[((int)wk + 1) * 128 + tid] = win[((int)wk + 1) * 128 + tid] + hi;
                }
            }
        }
    }
    if (weight_grad != 0) {
        long long wsum_0 = win[tid];
        if (wsum_0 != 0ll) {
            atomicAdd(&dw_win[(bid & 3) * 7 * 128 + tid], (unsigned long long)wsum_0);
        }
        long long wsum_1 = win[128 + tid];
        if (wsum_1 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 1) * 128 + tid], (unsigned long long)wsum_1);
        }
        long long wsum_2 = win[256 + tid];
        if (wsum_2 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 2) * 128 + tid], (unsigned long long)wsum_2);
        }
        long long wsum_3 = win[384 + tid];
        if (wsum_3 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 3) * 128 + tid], (unsigned long long)wsum_3);
        }
        long long wsum_4 = win[512 + tid];
        if (wsum_4 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 4) * 128 + tid], (unsigned long long)wsum_4);
        }
        long long wsum_5 = win[640 + tid];
        if (wsum_5 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 5) * 128 + tid], (unsigned long long)wsum_5);
        }
        long long wsum_6 = win[768 + tid];
        if (wsum_6 != 0ll) {
            atomicAdd(&dw_win[((bid & 3) * 7 + 6) * 128 + tid], (unsigned long long)wsum_6);
        }
        if (spec != 0.0f) {
            atomicAdd(&dw_spec[tid], spec);
        }
    }
    __syncthreads();
    if (tid == 0) {
        int is_last = 0;
        uint32_t arrival;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(arrival) : "l"(&dw_done[0]), "r"(static_cast<uint32_t>((unsigned int)num_bids - 1)) : "memory");
        if (arrival == (unsigned int)num_bids - 1) {
            is_last = 1;
        }
        flag[0] = is_last;
    }
    __syncthreads();
    int flag_word[1];
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&flag_word[0])) : "r"(flag_addr));
    if (flag_word[0] != 0) {
        if (weight_grad != 0) {
            unsigned long long words[28];
            unsigned long long _atom_bits_215;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_215) : "l"(&dw_win[tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_214 = _atom_bits_215;
            words[0] = _atomic_exchange_214;
            unsigned long long _atom_bits_217;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_217) : "l"(&dw_win[128 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_216 = _atom_bits_217;
            words[1] = _atomic_exchange_216;
            unsigned long long _atom_bits_219;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_219) : "l"(&dw_win[256 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_218 = _atom_bits_219;
            words[2] = _atomic_exchange_218;
            unsigned long long _atom_bits_221;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_221) : "l"(&dw_win[384 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_220 = _atom_bits_221;
            words[3] = _atomic_exchange_220;
            unsigned long long _atom_bits_223;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_223) : "l"(&dw_win[512 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_222 = _atom_bits_223;
            words[4] = _atomic_exchange_222;
            unsigned long long _atom_bits_225;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_225) : "l"(&dw_win[640 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_224 = _atom_bits_225;
            words[5] = _atomic_exchange_224;
            unsigned long long _atom_bits_227;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_227) : "l"(&dw_win[768 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_226 = _atom_bits_227;
            words[6] = _atomic_exchange_226;
            unsigned long long _atom_bits_229;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_229) : "l"(&dw_win[896 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_228 = _atom_bits_229;
            words[7] = _atomic_exchange_228;
            unsigned long long _atom_bits_231;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_231) : "l"(&dw_win[1024 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_230 = _atom_bits_231;
            words[8] = _atomic_exchange_230;
            unsigned long long _atom_bits_233;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_233) : "l"(&dw_win[1152 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_232 = _atom_bits_233;
            words[9] = _atomic_exchange_232;
            unsigned long long _atom_bits_235;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_235) : "l"(&dw_win[1280 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_234 = _atom_bits_235;
            words[10] = _atomic_exchange_234;
            unsigned long long _atom_bits_237;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_237) : "l"(&dw_win[1408 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_236 = _atom_bits_237;
            words[11] = _atomic_exchange_236;
            unsigned long long _atom_bits_239;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_239) : "l"(&dw_win[1536 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_238 = _atom_bits_239;
            words[12] = _atomic_exchange_238;
            unsigned long long _atom_bits_241;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_241) : "l"(&dw_win[1664 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_240 = _atom_bits_241;
            words[13] = _atomic_exchange_240;
            unsigned long long _atom_bits_243;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_243) : "l"(&dw_win[1792 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_242 = _atom_bits_243;
            words[14] = _atomic_exchange_242;
            unsigned long long _atom_bits_245;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_245) : "l"(&dw_win[1920 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_244 = _atom_bits_245;
            words[15] = _atomic_exchange_244;
            unsigned long long _atom_bits_247;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_247) : "l"(&dw_win[2048 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_246 = _atom_bits_247;
            words[16] = _atomic_exchange_246;
            unsigned long long _atom_bits_249;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_249) : "l"(&dw_win[2176 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_248 = _atom_bits_249;
            words[17] = _atomic_exchange_248;
            unsigned long long _atom_bits_251;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_251) : "l"(&dw_win[2304 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_250 = _atom_bits_251;
            words[18] = _atomic_exchange_250;
            unsigned long long _atom_bits_253;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_253) : "l"(&dw_win[2432 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_252 = _atom_bits_253;
            words[19] = _atomic_exchange_252;
            unsigned long long _atom_bits_255;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_255) : "l"(&dw_win[2560 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_254 = _atom_bits_255;
            words[20] = _atomic_exchange_254;
            unsigned long long _atom_bits_257;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_257) : "l"(&dw_win[2688 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_256 = _atom_bits_257;
            words[21] = _atomic_exchange_256;
            unsigned long long _atom_bits_259;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_259) : "l"(&dw_win[2816 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_258 = _atom_bits_259;
            words[22] = _atomic_exchange_258;
            unsigned long long _atom_bits_261;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_261) : "l"(&dw_win[2944 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_260 = _atom_bits_261;
            words[23] = _atomic_exchange_260;
            unsigned long long _atom_bits_263;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_263) : "l"(&dw_win[3072 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_262 = _atom_bits_263;
            words[24] = _atomic_exchange_262;
            unsigned long long _atom_bits_265;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_265) : "l"(&dw_win[3200 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_264 = _atom_bits_265;
            words[25] = _atomic_exchange_264;
            unsigned long long _atom_bits_267;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_267) : "l"(&dw_win[3328 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_266 = _atom_bits_267;
            words[26] = _atomic_exchange_266;
            unsigned long long _atom_bits_269;
            asm volatile("atom.relaxed.gpu.global.exch.b64 %0, [%1], %2;" : "=l"(_atom_bits_269) : "l"(&dw_win[3456 + tid]), "l"(static_cast<unsigned long long>(0ull)) : "memory");
            unsigned long long _atomic_exchange_268 = _atom_bits_269;
            words[27] = _atomic_exchange_268;
            float _atomic_fetch_add_270 = atomicAdd(&dw_spec[tid], -0.0f);
            spec = _atomic_fetch_add_270;
            float old = 0.0f;
            if (accumulate != 0) {
                float prev = atomicAdd(&dw[tid], -0.0f);
                old = prev;
            }
            long long digits[7];
            digits[0] = 0ll;
            digits[1] = 0ll;
            digits[2] = 0ll;
            digits[3] = 0ll;
            digits[4] = 0ll;
            digits[5] = 0ll;
            digits[6] = 0ll;
            digits[0] = digits[0] + (long long)words[0];
            digits[1] = digits[1] + (long long)words[1];
            digits[2] = digits[2] + (long long)words[2];
            digits[3] = digits[3] + (long long)words[3];
            digits[4] = digits[4] + (long long)words[4];
            digits[5] = digits[5] + (long long)words[5];
            digits[6] = digits[6] + (long long)words[6];
            digits[0] = digits[0] + (long long)words[7];
            digits[1] = digits[1] + (long long)words[8];
            digits[2] = digits[2] + (long long)words[9];
            digits[3] = digits[3] + (long long)words[10];
            digits[4] = digits[4] + (long long)words[11];
            digits[5] = digits[5] + (long long)words[12];
            digits[6] = digits[6] + (long long)words[13];
            digits[0] = digits[0] + (long long)words[14];
            digits[1] = digits[1] + (long long)words[15];
            digits[2] = digits[2] + (long long)words[16];
            digits[3] = digits[3] + (long long)words[17];
            digits[4] = digits[4] + (long long)words[18];
            digits[5] = digits[5] + (long long)words[19];
            digits[6] = digits[6] + (long long)words[20];
            digits[0] = digits[0] + (long long)words[21];
            digits[1] = digits[1] + (long long)words[22];
            digits[2] = digits[2] + (long long)words[23];
            digits[3] = digits[3] + (long long)words[24];
            digits[4] = digits[4] + (long long)words[25];
            digits[5] = digits[5] + (long long)words[26];
            digits[6] = digits[6] + (long long)words[27];
            if (spec != 0.0f) {
                dw_spec[tid] = 0.0f;
            }
            if ((__as_u32(old) & 2139095040) == 2139095040) {
                spec = spec + old;
            } else if ((__as_u32(old) & 2147483647) != 0) {
                unsigned int bits_1 = __as_u32(old);
                unsigned int ex_1 = bits_1 >> 23 & 255;
                unsigned int _max_271 = ((ex_1) > (1) ? (ex_1) : (1));
                unsigned int idx_1 = _max_271 - 1 + 38;
                unsigned int _min_272 = ((ex_1) < (1) ? (ex_1) : (1));
                unsigned long long m_1 = (unsigned long long)(bits_1 & 8388607 | _min_272 << 23);
                unsigned int wk_1 = idx_1 / 48;
                unsigned int sh_1 = idx_1 - wk_1 * 48;
                long long lo_1 = (long long)(m_1 << (unsigned long long)sh_1 & 281474976710655ull);
                long long hi_1 = (long long)(m_1 >> (unsigned long long)(48 - sh_1));
                if (bits_1 >> 31 != 0) {
                    lo_1 = -lo_1;
                    hi_1 = -hi_1;
                }
                digits[0] = digits[0] + (((int)wk_1 == 0) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 0) ? hi_1 : 0ll);
                digits[1] = digits[1] + (((int)wk_1 == 1) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 1) ? hi_1 : 0ll);
                digits[2] = digits[2] + (((int)wk_1 == 2) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 2) ? hi_1 : 0ll);
                digits[3] = digits[3] + (((int)wk_1 == 3) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 3) ? hi_1 : 0ll);
                digits[4] = digits[4] + (((int)wk_1 == 4) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 4) ? hi_1 : 0ll);
                digits[5] = digits[5] + (((int)wk_1 == 5) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 5) ? hi_1 : 0ll);
                digits[6] = digits[6] + (((int)wk_1 == 6) ? lo_1 : 0ll) + (((int)wk_1 + 1 == 6) ? hi_1 : 0ll);
            }
            long long carry = 0ll;
            long long t_0 = digits[0] + carry;
            digits[0] = t_0 & 281474976710655ll;
            carry = t_0 >> 48ll;
            long long t_1 = digits[1] + carry;
            digits[1] = t_1 & 281474976710655ll;
            carry = t_1 >> 48ll;
            long long t_2 = digits[2] + carry;
            digits[2] = t_2 & 281474976710655ll;
            carry = t_2 >> 48ll;
            long long t_3 = digits[3] + carry;
            digits[3] = t_3 & 281474976710655ll;
            carry = t_3 >> 48ll;
            long long t_4 = digits[4] + carry;
            digits[4] = t_4 & 281474976710655ll;
            carry = t_4 >> 48ll;
            long long t_5 = digits[5] + carry;
            digits[5] = t_5 & 281474976710655ll;
            carry = t_5 >> 48ll;
            long long t_6 = digits[6] + carry;
            digits[6] = t_6 & 281474976710655ll;
            carry = t_6 >> 48ll;
            long long up_0 = digits[1] << 10ll & 281474976710655ll;
            digits[0] = digits[0] >> 38ll | up_0;
            long long up_1 = digits[2] << 10ll & 281474976710655ll;
            digits[1] = digits[1] >> 38ll | up_1;
            long long up_2 = digits[3] << 10ll & 281474976710655ll;
            digits[2] = digits[2] >> 38ll | up_2;
            long long up_3 = digits[4] << 10ll & 281474976710655ll;
            digits[3] = digits[3] >> 38ll | up_3;
            long long up_4 = digits[5] << 10ll & 281474976710655ll;
            digits[4] = digits[4] >> 38ll | up_4;
            long long up_5 = digits[6] << 10ll & 281474976710655ll;
            digits[5] = digits[5] >> 38ll | up_5;
            long long top = carry << 10ll & 281474976710655ll;
            digits[6] = digits[6] >> 38ll | top;
            carry = carry >> 38ll;
            double exact = (double)carry;
            exact = exact * 281474976710656.0 + (double)digits[6];
            exact = exact * 281474976710656.0 + (double)digits[5];
            exact = exact * 281474976710656.0 + (double)digits[4];
            exact = exact * 281474976710656.0 + (double)digits[3];
            exact = exact * 281474976710656.0 + (double)digits[2];
            exact = exact * 281474976710656.0 + (double)digits[1];
            exact = exact * 281474976710656.0 + (double)digits[0];
            float rounded = ((spec != 0.0f) ? spec : (float)(exact * 1.401298464324817e-45));
            dw[tid] = rounded;
        }
    }
}

} // extern "C"
