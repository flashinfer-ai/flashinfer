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
#define SMEM_RING_OFF 1024
#define SMEM_RING_STAGE_BYTES 98304
#define SMEM_RING_STRIDE 98304
#define SMEM_TOTAL 99328
#define THREADS 288

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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

extern "C" {

__global__ __launch_bounds__(288) void
kernel_cake_rmsnorm_train_682675bda578f5b7a312(__nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ u, __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ y, __nv_bfloat16* __restrict__ h_new, float* __restrict__ r, int T, float eps, long long x_stride_p0, long long u_stride_p0)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* ring = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int ring_addr = smem + 1024;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=256
            mbarrier_init(smem + 32, 256);
            mbarrier_init(smem + 40, 256);
            mbarrier_init(smem + 48, 256);
            mbarrier_init(smem + 56, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: consumers ----
    if (warp <= 7) {
        { // consumers_main
            long long x_stride = x_stride_p0;
            long long u_stride = u_stride_p0;
            int warp_0 = warp;
            int lane_1 = lane;
            int rg = warp_0;
            int g0 = lane_1;
            float h_f = 1536.0f;
            int ring_c = (int)ring_addr;
            #pragma unroll
            for (int s = 0; s < 4; s++) {
                mbarrier_arrive(empty_addr + (s) * 8);
            }
            unsigned int wreg[24];
            #pragma unroll
            for (int c = 0; c < 2; c++) {
                #pragma unroll
                for (int v = 0; v < 3; v++) {
                    {
                        const uint4* _vptr_0 = reinterpret_cast<const uint4*>(w + ((g0 + c * 32) * 8 + v * 512) + 0);
                        uint4* _vdst_0 = reinterpret_cast<uint4*>(&wreg[c * 12 + v * 4]);
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                                : "=r"(_vdst_0[_blk].x), "=r"(_vdst_0[_blk].y), "=r"(_vdst_0[_blk].z), "=r"(_vdst_0[_blk].w) : "l"((const void*)(_vptr_0 + _blk)) : "memory");
                        }
                    }
                }
            }
            unsigned int xw[24];
            int k = 0;
            #pragma unroll 1
            for (int first = bid * 8; first < T; first += num_bids * 8) {
                int row = first + rg;
                int slot = k % 4;
                int _stage_full = (int)((unsigned int)k % 4);
                int _parity_full = (int)((unsigned int)k / 4 % 2);
                mbarrier_wait(full_addr + (_stage_full) * 8, _parity_full);
                int xaddr = ring_c + slot * 24576 + rg * 3072 + g0 * 16;
                #pragma unroll
                for (int c_1 = 0; c_1 < 2; c_1++) {
                    #pragma unroll
                    for (int v_1 = 0; v_1 < 3; v_1++) {
                        __int128_t _smem_raw_1;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_1) : "r"(xaddr + c_1 * 512 + v_1 * 1024));
                        xw[c_1 * 12 + v_1 * 4] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[0];
                        xw[(c_1 * 12 + v_1 * 4) + 1] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[1];
                        xw[(c_1 * 12 + v_1 * 4) + 2] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[2];
                        xw[(c_1 * 12 + v_1 * 4) + 3] = reinterpret_cast<const uint32_t*>(&_smem_raw_1)[3];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(empty_addr + ((int)((unsigned int)k % 4)) * 8);
                if (row < T) {
                    float sum0 = 0.0f;
                    #pragma unroll
                    for (int i = 0; i < 12; i++) {
                        float _dot_2;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_2) : "r"(xw[i]), "r"(xw[i]), "f"(sum0));
                        sum0 = _dot_2;
                    }
                    float _warp_reduce_0 = sum0;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                    float sum1 = 0.0f;
                    #pragma unroll
                    for (int i_1 = 0; i_1 < 12; i_1++) {
                        float _dot_3;
                        asm volatile(
                            "{\n\t"
                            ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                            "mov.b32 {a_lo, a_hi}, %1;\n\t"
                            "mov.b32 {b_lo, b_hi}, %2;\n\t"
                            "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                            "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                            "}\n"
                            : "=f"(_dot_3) : "r"(xw[12 + i_1]), "r"(xw[12 + i_1]), "f"(sum1));
                        sum1 = _dot_3;
                    }
                    float _warp_reduce_4 = sum1;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_4 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_4, offset);
                    float total = _warp_reduce_0 + _warp_reduce_4;
                    float mean = __fdiv_rn(total, h_f);
                    float _sqrt_5;
                    asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_5) : "f"(mean + eps));
                    float rstd = __frcp_rn(_sqrt_5);
                    if (g0 == 0) {
                        r[row] = rstd;
                    }
                    long long obase = (long long)row * 1536;
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        #pragma unroll
                        for (int v_2 = 0; v_2 < 3; v_2++) {
                            float out[8];
                            #pragma unroll
                            for (int q = 0; q < 4; q++) {
                                float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(xw[c_2 * 12 + v_2 * 4 + q]));
                                float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(wreg[c_2 * 12 + v_2 * 4 + q]));
                                out[2 * q] = _cvt_f32_0.x * rstd * _cvt_f32_1.x;
                                out[2 * q + 1] = _cvt_f32_0.y * rstd * _cvt_f32_1.y;
                            }
                            unsigned int _packed_0[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            reinterpret_cast<int4*>(y + (obase + (long long)((g0 + c_2 * 32) * 8 + v_2 * 512)))[0] = reinterpret_cast<int4*>(_packed_0)[0];
                        }
                    }
                }
                k += 1;
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 8) {
        { // producer_main
            long long x_stride_p = x_stride_p0;
            int lane_p = lane;
            int ring_p = (int)ring_addr;
            int k_p = 0;
            #pragma unroll 1
            for (int first_p = bid * 8; first_p < T; first_p += num_bids * 8) {
                int slot_p = k_p % 4;
                int rows_k = ((T - first_p) < (8) ? (T - first_p) : (8));
                int sbase = ring_p + slot_p * 24576;
                int _stage_empty = (int)((unsigned int)k_p % 4);
                int _parity_empty = (int)((unsigned int)k_p / 4 % 2);
                mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                if (lane_p == 0) {
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)k_p % 4)) * 8, rows_k * 3072);
                }
                if (lane_p == 0) {
                    #pragma unroll
                    for (int i_2 = 0; i_2 < 8; i_2++) {
                        if (rows_k > i_2) {
                            long long src_i = (long long)(first_p + i_2) * x_stride_p;
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1], %2, [%3], %4;"
                                :: "r"(sbase + i_2 * 3072), "l"(reinterpret_cast<const uint8_t*>(x + src_i)), "r"((unsigned)(3072)), "r"(full_addr + ((int)((unsigned int)k_p % 4)) * 8),
                                   "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                }
                k_p += 1;
            }
        }
    }

    // Cleanup
}

} // extern "C"
