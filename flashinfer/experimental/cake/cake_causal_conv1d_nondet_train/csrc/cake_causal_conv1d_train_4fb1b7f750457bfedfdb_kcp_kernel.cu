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

#define LAUNCH_MIN_BLOCKS 1

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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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





__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_4fb1b7f750457bfedfdb_kcp(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* dw_acc, unsigned int* __restrict__ tile_tk, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_tail, int* __restrict__ tail_row_off, int* __restrict__ skip_row_off, unsigned int* claims, unsigned int* __restrict__ departs, int S, int C, int num_docs, int M, int NL, int P, int L, int n_ctas, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(dw_acc));
    __builtin_assume(__isGlobal(dw_acc));

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
    #define d_full_addr (mbar_base + 0)
    #define w_full_addr (mbar_base + 192)
    #define w_empty_addr (mbar_base + 200)
    #define e_full_addr (mbar_base + 208)
    #define e_empty_addr (mbar_base + 216)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int stage_addr = smem + 1024;
    unsigned int* rsp = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int rsp_addr = smem + 197632;
    int* dv_s = reinterpret_cast<int*>(smem_raw + 197648);
    const int desc_addr = smem + 197648;

    // Mbarrier init (28 barriers at smem_raw[0..224))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // d_full: 24 barriers, init_count=32
            mbarrier_init(smem + 0, 32);
            mbarrier_init(smem + 8, 32);
            mbarrier_init(smem + 16, 32);
            mbarrier_init(smem + 24, 32);
            mbarrier_init(smem + 32, 32);
            mbarrier_init(smem + 40, 32);
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
            mbarrier_init(smem + 72, 32);
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            mbarrier_init(smem + 184, 32);
            // --- pipeline 'work' ---
            // w_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // w_empty: 1 barriers, init_count=32
            mbarrier_init(smem + 200, 32);
            // --- pipeline 'desc' ---
            // e_full: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // e_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: compute ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 240;");
        { // compute_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            int warp_1 = tid >> 5;
            long long C64 = (long long)C;
            int dl[1];
            int ov[1];
            float wr[32];
            float w4[4];
            float win[32];
            float pend[32];
            float acc[32];
            float tailf[8];
            unsigned int pkx[4];
            unsigned int pkd[4];
            float dyf[8];
            float dp[8];
            float a4[4];
            float lo4[4];
            float lw[128];
            unsigned int fin = 0;
            int cur_ct = -1;
            int run_t0 = 0;
            int kc = 0;
            int k_run = 0;
            int frag = 0;
            int doc_start = 0;
            int doc_end = 0;
            #pragma unroll 1
            for (int _trip = 0; _trip < L * (NL + 1) + 1; _trip++) {
                int _parity_e_full = (int)((unsigned int)_trip % 2);
                mbarrier_wait(e_full_addr, _parity_e_full);
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&dl[0])) : "r"((int)desc_addr + lane % 16 * 4));
                mbarrier_arrive(e_empty_addr);
                int fl = __shfl_sync(0xFFFFFFFF, dl[0], 2);
                if (fl == 8) {
                    break;
                }
                int gl = __shfl_sync(0xFFFFFFFF, dl[0], 0);
                int un = __shfl_sync(0xFFFFFFFF, dl[0], 1);
                int d_fr = __shfl_sync(0xFFFFFFFF, dl[0], 5);
                int d_ds = __shfl_sync(0xFFFFFFFF, dl[0], 6);
                int d_de = __shfl_sync(0xFFFFFFFF, dl[0], 7);
                int d_h0 = __shfl_sync(0xFFFFFFFF, dl[0], 8);
                int d_h1 = __shfl_sync(0xFFFFFFFF, dl[0], 9);
                int d_h2 = __shfl_sync(0xFFFFFFFF, dl[0], 10);
                int is_start = fl % 2;
                int is_last = fl / 2 % 2;
                int is_end = fl / 16 % 2;
                int ct = gl / P;
                int s_ln = gl % P;
                int cbase = ct * 2048;
                int cw = cbase + warp_1 * 256;
                int _max_0 = ((C - cw) > (0) ? (C - cw) : (0));
                int wbytes = ((512) < (_max_0 * 2) ? (512) : (_max_0 * 2));
                int c0 = cbase + tid * 8;
                long long c64 = (long long)c0;
                int _max_1 = ((un) > (0) ? (un) : (0));
                int t0 = s_ln * (NL * M) + _max_1 * M;
                if (is_end != 0) {
                    t0 = t0 + M;
                }
                int t1 = ((t0 + M) < (S) ? (t0 + M) : (S));
                int n_tok = 0;
                if (un >= 0 && is_end == 0) {
                    int _max_2 = ((t1 - t0) > (0) ? (t1 - t0) : (0));
                    n_tok = _max_2;
                }
                if (is_end != 0) {
                    t1 = t0;
                }
                int t_end = t0 + M;
                int t_lim = t0 + M;
                if (is_last != 0 || is_end != 0) {
                    t_end = t0;
                    if (n_tok > 0 || is_end != 0) {
                        int _min_3 = ((t1 + 3) < (S + 1) ? (t1 + 3) : (S + 1));
                        t_end = _min_3;
                    }
                    int _min_4 = ((t1 + 3) < (S) ? (t1 + 3) : (S));
                    t_lim = _min_4;
                }
                int n_st = (t_end - t0 + 7) / 8;
                int kg = (int)kc;
                int active = 0;
                if (c0 < C && t_end > t0) {
                    active = 1;
                }
                if (is_start != 0) {
                    run_t0 = t0;
                    k_run = kg;
                    #pragma unroll
                    for (int q = 0; q < 32; q++) {
                        win[q] = 0.0f;
                        pend[q] = 0.0f;
                        acc[q] = 0.0f;
                    }
                    frag = d_fr;
                    doc_start = d_ds;
                    doc_end = d_de;
                    if (active != 0) {
                        if (ct != cur_ct) {
                            #pragma unroll
                            for (int j2 = 0; j2 < 8; j2++) {
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j2) * 4ll);
                                    wr[j2 * 4 + 0] = _v4.x;
                                    wr[j2 * 4 + 1] = _v4.y;
                                    wr[j2 * 4 + 2] = _v4.z;
                                    wr[j2 * 4 + 3] = _v4.w;
                                }
                            }
                            cur_ct = ct;
                        }
                        if (d_h0 >= 0) {
                            {
                                const uint4* _vptr_6 = reinterpret_cast<const uint4*>(x + (long long)d_h0 * C64 + c64);
                                uint4 _vld_6[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_6[_blk] = _vptr_6[_blk];
                                    uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) << 16);
                                        (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_6[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        } else if (d_h0 <= -2) {
                            {
                                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(hist + (long long)(-2 - d_h0) * C64 + c64);
                                uint4 _vld_7[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_7[_blk] = _vptr_7[_blk];
                                    uint32_t* _vpairs_7 = reinterpret_cast<uint32_t*>(&_vld_7[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[24 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) << 16);
                                        (&win[24 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_7[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        }
                        if (d_h1 >= 0) {
                            {
                                const uint4* _vptr_8 = reinterpret_cast<const uint4*>(x + (long long)d_h1 * C64 + c64);
                                uint4 _vld_8[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_8[_blk] = _vptr_8[_blk];
                                    uint32_t* _vpairs_8 = reinterpret_cast<uint32_t*>(&_vld_8[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) << 16);
                                        (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_8[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        } else if (d_h1 <= -2) {
                            {
                                const uint4* _vptr_9 = reinterpret_cast<const uint4*>(hist + (long long)(-2 - d_h1) * C64 + c64);
                                uint4 _vld_9[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_9[_blk] = _vptr_9[_blk];
                                    uint32_t* _vpairs_9 = reinterpret_cast<uint32_t*>(&_vld_9[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[16 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) << 16);
                                        (&win[16 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_9[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        }
                        if (d_h2 >= 0) {
                            {
                                const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + (long long)d_h2 * C64 + c64);
                                uint4 _vld_10[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_10[_blk] = _vptr_10[_blk];
                                    uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                        (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        } else if (d_h2 <= -2) {
                            {
                                const uint4* _vptr_11 = reinterpret_cast<const uint4*>(hist + (long long)(-2 - d_h2) * C64 + c64);
                                uint4 _vld_11[1];
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vld_11[_blk] = _vptr_11[_blk];
                                    uint32_t* _vpairs_11 = reinterpret_cast<uint32_t*>(&_vld_11[_blk]);
                                    #pragma unroll
                                    for (int _pair = 0; _pair < 4; _pair++) {
                                        (&win[8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) << 16);
                                        (&win[8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_11[_pair]) & 0xffff0000u);
                                    }
                                }
                            }
                        }
                    }
                }
                int k_stop = kg + n_st;
                int k_lim = k_stop;
                int k_peek = -1;
                if (is_last == 0 && is_end == 0) {
                    k_peek = k_stop - 3;
                }
                int t_fill = t_lim;
                if (cw >= C) {
                    t_fill = run_t0;
                }
                if (is_start != 0) {
                    if (0 < n_st) {
                        int _max_12 = ((t_fill - run_t0) > (0) ? (t_fill - run_t0) : (0));
                        int nrows = ((_max_12) < (8) ? (_max_12) : (8));
                        int sb = (int)stage_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8192;
                        if (lane == 0 && nrows == 8 && wbytes == 512) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8, 8192);
                            tma_2d_gmem2smem(sb, (&x_map), cw, run_t0, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8);
                            tma_2d_gmem2smem(sb + 4096, (&dy_map), cw, run_t0, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8);
                        } else if (lane == 0) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8, nrows * (2 * wbytes));
                            #pragma unroll 1
                            for (int irow = 0; irow < nrows; irow++) {
                                long long row64 = (long long)(run_t0 + irow) * (long long)C + (long long)cw;
                                cp_async_bulk_gmem2smem(sb + irow * 512, reinterpret_cast<const uint8_t*>(x + row64), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8);
                                cp_async_bulk_gmem2smem(sb + 4096 + irow * 512, reinterpret_cast<const uint8_t*>(dy + row64), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8);
                            }
                        } else {
                            mbarrier_arrive(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(int)kc % 3)) * 8);
                        }
                    }
                    if (1 < n_st) {
                        int _max_13 = ((t_fill - (run_t0 + 8)) > (0) ? (t_fill - (run_t0 + 8)) : (0));
                        int nrows_1 = ((_max_13) < (8) ? (_max_13) : (8));
                        int sb_1 = (int)stage_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8192;
                        if (lane == 0 && nrows_1 == 8 && wbytes == 512) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8, 8192);
                            tma_2d_gmem2smem(sb_1, (&x_map), cw, run_t0 + 8, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8);
                            tma_2d_gmem2smem(sb_1 + 4096, (&dy_map), cw, run_t0 + 8, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8);
                        } else if (lane == 0) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8, nrows_1 * (2 * wbytes));
                            #pragma unroll 1
                            for (int irow_1 = 0; irow_1 < nrows_1; irow_1++) {
                                long long row64_1 = (long long)(run_t0 + 8 + irow_1) * (long long)C + (long long)cw;
                                cp_async_bulk_gmem2smem(sb_1 + irow_1 * 512, reinterpret_cast<const uint8_t*>(x + row64_1), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8);
                                cp_async_bulk_gmem2smem(sb_1 + 4096 + irow_1 * 512, reinterpret_cast<const uint8_t*>(dy + row64_1), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8);
                            }
                        } else {
                            mbarrier_arrive(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 1) % 3)) * 8);
                        }
                    }
                    if (2 < n_st) {
                        int _max_14 = ((t_fill - (run_t0 + 16)) > (0) ? (t_fill - (run_t0 + 16)) : (0));
                        int nrows_2 = ((_max_14) < (8) ? (_max_14) : (8));
                        int sb_2 = (int)stage_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8192;
                        if (lane == 0 && nrows_2 == 8 && wbytes == 512) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8, 8192);
                            tma_2d_gmem2smem(sb_2, (&x_map), cw, run_t0 + 16, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8);
                            tma_2d_gmem2smem(sb_2 + 4096, (&dy_map), cw, run_t0 + 16, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8);
                        } else if (lane == 0) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8, nrows_2 * (2 * wbytes));
                            #pragma unroll 1
                            for (int irow_2 = 0; irow_2 < nrows_2; irow_2++) {
                                long long row64_2 = (long long)(run_t0 + 16 + irow_2) * (long long)C + (long long)cw;
                                cp_async_bulk_gmem2smem(sb_2 + irow_2 * 512, reinterpret_cast<const uint8_t*>(x + row64_2), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8);
                                cp_async_bulk_gmem2smem(sb_2 + 4096 + irow_2 * 512, reinterpret_cast<const uint8_t*>(dy + row64_2), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8);
                            }
                        } else {
                            mbarrier_arrive(d_full_addr + (warp_1 + 8 * (int)((unsigned int)((int)kc + 2) % 3)) * 8);
                        }
                    }
                }
                #pragma unroll 1
                for (int kq = kg; kq < k_stop; kq++) {
                    if (kq == k_peek) {
                        int _parity_e_full_1 = (int)((unsigned int)_trip % 2) ^ 1;
                        mbarrier_wait(e_full_addr, _parity_e_full_1);
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])) : "r"((int)desc_addr + 8));
                        if (ov[0] / 16 % 2 != 0) {
                            k_lim = k_stop + 1;
                            int _min_15 = ((t1 + 3) < (S) ? (t1 + 3) : (S));
                            t_fill = _min_15;
                        } else {
                            int t1n = ((t1 + M) < (S) ? (t1 + M) : (S));
                            int te_n = t1n;
                            t_fill = t1n;
                            if (ov[0] / 2 % 2 != 0) {
                                int _min_16 = ((t1n + 3) < (S + 1) ? (t1n + 3) : (S + 1));
                                te_n = _min_16;
                                int _min_17 = ((t1n + 3) < (S) ? (t1n + 3) : (S));
                                t_fill = _min_17;
                            }
                            int _min_18 = ((3) < ((te_n - t1 + 7) / 8) ? (3) : ((te_n - t1 + 7) / 8));
                            k_lim = k_stop + _min_18;
                        }
                        if (cw >= C) {
                            t_fill = run_t0;
                        }
                    }
                    int ks = warp_1 + 8 * (int)((unsigned int)kq % 3);
                    int _parity_d_full = (int)((unsigned int)kq / 3 % 2);
                    mbarrier_wait(d_full_addr + (ks) * 8, _parity_d_full);
                    if (active != 0) {
                        int tb = run_t0 + (kq - k_run) * 8;
                        int _max_19 = ((t_lim - tb) > (0) ? (t_lim - tb) : (0));
                        int rows_k = ((_max_19) < (8) ? (_max_19) : (8));
                        int steps_k = ((t_end - tb) < (8) ? (t_end - tb) : (8));
                        int sx0 = (int)stage_addr + ks * 8192 + lane * 16;
                        if (kq > k_run && t1 >= tb + 8 && doc_end >= tb + 8 && doc_start <= tb - 3) {
                            #pragma unroll
                            for (int u = 0; u < 8; u++) {
                                __int128_t _smem_raw_20;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_20) : "r"(sx0 + u * 512));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_20)[3];
                                __int128_t _smem_raw_21;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_21) : "r"(sx0 + 4096 + u * 512));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_21)[3];
                                #pragma unroll
                                for (int kk = 0; kk < 4; kk++) {
                                    win[u % 4 * 8 + 2 * kk] = __uint_as_float(pkx[kk] << 16);
                                    win[u % 4 * 8 + 2 * kk + 1] = __uint_as_float(pkx[kk] & 4294901760u);
                                    dyf[2 * kk] = __uint_as_float(pkd[kk] << 16);
                                    dyf[2 * kk + 1] = __uint_as_float(pkd[kk] & 4294901760u);
                                }
                                #pragma unroll
                                for (int h = 0; h < 4; h++) {
                                    float2 p2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * h], win[u % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 2], wr[8 * h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * h], win[(u % 4 + 3) % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 1], wr[8 * h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * h], win[(u % 4 + 2) % 4 * 8 + 2 * h + 1]), mul_f32x2_noftz(make_float2(wr[8 * h], wr[8 * h + 4]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * h], win[(u % 4 + 1) % 4 * 8 + 2 * h + 1])))));
                                    float2 _f32x2_mul_b_22 = make_float2(-1.442695f, -1.442695f);
                                    float2 e2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_22));
                                    float _exp2_23 = approx_exp2(e2.x);
                                    float _exp2_24 = approx_exp2(e2.y);
                                    float2 den2 = add_f32x2_noftz(make_float2(_exp2_23, _exp2_24), make_float2(1.0f, 1.0f));
                                    float _rcp_25 = approx_rcp(den2.x);
                                    float _rcp_26 = approx_rcp(den2.y);
                                    float2 sg2 = make_float2(_rcp_25, _rcp_26);
                                    float2 _f32x2_mul_b_27 = fma_f32x2_rn_noftz(p2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2), make_float2(1.0f, 1.0f));
                                    float2 g2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2) : "l"(*(const unsigned long long*)&sg2), "l"(*(const unsigned long long*)&_f32x2_mul_b_27));
                                    float2 _f32x2_mul_a_28 = make_float2(dyf[2 * h], dyf[2 * h + 1]);
                                    float2 d2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&g2));
                                    dp[2 * h] = d2.x;
                                    dp[2 * h + 1] = d2.y;
                                    #pragma unroll
                                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                        float2 a2 = fma_f32x2_rn_noftz(d2, make_float2(win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * h], win[(u % 4 + 1 + kk_1) % 4 * 8 + 2 * h + 1]), make_float2(acc[8 * h + kk_1], acc[8 * h + 4 + kk_1]));
                                        acc[8 * h + kk_1] = a2.x;
                                        acc[8 * h + 4 + kk_1] = a2.y;
                                    }
                                    #pragma unroll
                                    for (int k = 0; k < 3; k++) {
                                        float2 q2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + k], wr[8 * h + 4 + k]), d2, make_float2(pend[(u % 4 + 1 + k) % 4 * 8 + 2 * h], pend[(u % 4 + 1 + k) % 4 * 8 + 2 * h + 1]));
                                        pend[(u % 4 + 1 + k) % 4 * 8 + 2 * h] = q2.x;
                                        pend[(u % 4 + 1 + k) % 4 * 8 + 2 * h + 1] = q2.y;
                                    }
                                    float2 _f32x2_mul_a_29 = make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]);
                                    float2 r2;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&d2));
                                    pend[u % 4 * 8 + 2 * h] = r2.x;
                                    pend[u % 4 * 8 + 2 * h + 1] = r2.y;
                                }
                                long long row3 = (long long)(tb + (u - 3)) * C64 + c64;
                                unsigned int packed[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                    packed[_lp] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                    packed[_lp + 1] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                    packed[_lp + 2] = *(uint32_t*)&_bf2;
                                }
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                    packed[_lp + 3] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(dx + (row3))[0] = reinterpret_cast<int4*>(packed)[0];
                            }
                        } else if (kq == k_run && t1 >= tb + 8 && doc_end >= tb + 8) {
                            #pragma unroll
                            for (int u_1 = 0; u_1 < 8; u_1++) {
                                __int128_t _smem_raw_30;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_30) : "r"(sx0 + u_1 * 512));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[3];
                                __int128_t _smem_raw_31;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_31) : "r"(sx0 + 4096 + u_1 * 512));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_31)[3];
                                #pragma unroll
                                for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                    win[u_1 % 4 * 8 + 2 * kk_2] = __uint_as_float(pkx[kk_2] << 16);
                                    win[u_1 % 4 * 8 + 2 * kk_2 + 1] = __uint_as_float(pkx[kk_2] & 4294901760u);
                                    dyf[2 * kk_2] = __uint_as_float(pkd[kk_2] << 16);
                                    dyf[2 * kk_2 + 1] = __uint_as_float(pkd[kk_2] & 4294901760u);
                                }
                                #pragma unroll
                                for (int h_1 = 0; h_1 < 4; h_1++) {
                                    float2 p2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 3], wr[8 * h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * h_1], win[u_1 % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 2], wr[8 * h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 1], wr[8 * h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_1], wr[8 * h_1 + 4]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1 + 1])))));
                                    float2 _f32x2_mul_b_32 = make_float2(-1.442695f, -1.442695f);
                                    float2 e2_1;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_1) : "l"(*(const unsigned long long*)&p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_32));
                                    float _exp2_33 = approx_exp2(e2_1.x);
                                    float _exp2_34 = approx_exp2(e2_1.y);
                                    float2 den2_1 = add_f32x2_noftz(make_float2(_exp2_33, _exp2_34), make_float2(1.0f, 1.0f));
                                    float _rcp_35 = approx_rcp(den2_1.x);
                                    float _rcp_36 = approx_rcp(den2_1.y);
                                    float2 sg2_1 = make_float2(_rcp_35, _rcp_36);
                                    float2 _f32x2_mul_b_37 = fma_f32x2_rn_noftz(p2_1, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_1), make_float2(1.0f, 1.0f));
                                    float2 g2_1;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_1) : "l"(*(const unsigned long long*)&sg2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_37));
                                    float2 _f32x2_mul_a_38 = make_float2(dyf[2 * h_1], dyf[2 * h_1 + 1]);
                                    float2 d2_1;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_38), "l"(*(const unsigned long long*)&g2_1));
                                    dp[2 * h_1] = d2_1.x;
                                    dp[2 * h_1 + 1] = d2_1.y;
                                    #pragma unroll
                                    for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                        float2 a2_1 = fma_f32x2_rn_noftz(d2_1, make_float2(win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 1 + kk_3) % 4 * 8 + 2 * h_1 + 1]), make_float2(acc[8 * h_1 + kk_3], acc[8 * h_1 + 4 + kk_3]));
                                        acc[8 * h_1 + kk_3] = a2_1.x;
                                        acc[8 * h_1 + 4 + kk_3] = a2_1.y;
                                    }
                                    #pragma unroll
                                    for (int k_1 = 0; k_1 < 3; k_1++) {
                                        float2 q2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + k_1], wr[8 * h_1 + 4 + k_1]), d2_1, make_float2(pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1], pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1 + 1]));
                                        pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1] = q2_1.x;
                                        pend[(u_1 % 4 + 1 + k_1) % 4 * 8 + 2 * h_1 + 1] = q2_1.y;
                                    }
                                    float2 _f32x2_mul_a_39 = make_float2(wr[8 * h_1 + 3], wr[8 * h_1 + 4 + 3]);
                                    float2 r2_1;
                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_39), "l"(*(const unsigned long long*)&d2_1));
                                    pend[u_1 % 4 * 8 + 2 * h_1] = r2_1.x;
                                    pend[u_1 % 4 * 8 + 2 * h_1 + 1] = r2_1.y;
                                }
                                if (u_1 >= 3) {
                                    long long row3_1 = (long long)(tb + (u_1 - 3)) * C64 + c64;
                                    unsigned int packed_1[4];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 0], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 0]));
                                        packed_1[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 2], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 2]));
                                        packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 4], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 4]));
                                        packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 1; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2 + 6], (pend + (u_1 % 4 + 1) % 4 * 8)[_lp*2+1 + 6]));
                                        packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                                    }
                                    reinterpret_cast<int4*>(dx + (row3_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
                                }
                            }
                        } else {
                            #pragma unroll 1
                            for (int h_2 = 0; h_2 < 2; h_2++) {
                                #pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    int u_2 = h_2 * 4 + v;
                                    if (steps_k > u_2) {
                                        int tcur = tb + u_2;
                                        if (rows_k > u_2) {
                                            __int128_t _smem_raw_40;
                                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_40) : "r"(sx0 + u_2 * 512));
                                            pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[0];
                                            pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[1];
                                            pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[2];
                                            pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_40)[3];
                                            __int128_t _smem_raw_41;
                                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_41) : "r"(sx0 + 4096 + u_2 * 512));
                                            pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[0];
                                            pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[1];
                                            pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[2];
                                            pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_41)[3];
                                            #pragma unroll
                                            for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                                win[v * 8 + 2 * kk_4] = __uint_as_float(pkx[kk_4] << 16);
                                                win[v * 8 + 2 * kk_4 + 1] = __uint_as_float(pkx[kk_4] & 4294901760u);
                                                dyf[2 * kk_4] = __uint_as_float(pkd[kk_4] << 16);
                                                dyf[2 * kk_4 + 1] = __uint_as_float(pkd[kk_4] & 4294901760u);
                                            }
                                        }
                                        if (tcur >= doc_end) {
                                            int trow = tail_row_off[frag * 3];
                                            if (trow >= 0) {
                                                long long rowr = (long long)trow * C64 + c64;
                                                {
                                                    unsigned _ldv8_42_0;
                                                    unsigned _ldv8_42_1;
                                                    unsigned _ldv8_42_2;
                                                    unsigned _ldv8_42_3;
                                                    unsigned _ldv8_42_4;
                                                    unsigned _ldv8_42_5;
                                                    unsigned _ldv8_42_6;
                                                    unsigned _ldv8_42_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_42_0), "=r"(_ldv8_42_1), "=r"(_ldv8_42_2), "=r"(_ldv8_42_3), "=r"(_ldv8_42_4), "=r"(_ldv8_42_5), "=r"(_ldv8_42_6), "=r"(_ldv8_42_7) : "l"((const void*)(dpre_tail + (rowr))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_42_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_42_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_42_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_42_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_42_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_42_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_42_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_42_7);
                                                }
                                                #pragma unroll
                                                for (int j = 0; j < 8; j++) {
                                                    #pragma unroll
                                                    for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                                        w4[kk_5] = wr[j * 4 + kk_5];
                                                    }
                                                    float _fma_43;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(w4[2]), "f"(tailf[j]), "f"(pend[(v % 4 + 4 - 1) % 4 * 8 + j]));
                                                    pend[(v % 4 + 4 - 1) % 4 * 8 + j] = _fma_43;
                                                    float _fma_44;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(w4[1]), "f"(tailf[j]), "f"(pend[(v % 4 + 4 - 2) % 4 * 8 + j]));
                                                    pend[(v % 4 + 4 - 2) % 4 * 8 + j] = _fma_44;
                                                    float _fma_45;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(w4[0]), "f"(tailf[j]), "f"(pend[(v % 4 + 4 - 3) % 4 * 8 + j]));
                                                    pend[(v % 4 + 4 - 3) % 4 * 8 + j] = _fma_45;
                                                }
                                            }
                                            int trow_0 = tail_row_off[frag * 3 + 1];
                                            if (trow_0 >= 0) {
                                                long long rowr_1 = (long long)trow_0 * C64 + c64;
                                                {
                                                    unsigned _ldv8_46_0;
                                                    unsigned _ldv8_46_1;
                                                    unsigned _ldv8_46_2;
                                                    unsigned _ldv8_46_3;
                                                    unsigned _ldv8_46_4;
                                                    unsigned _ldv8_46_5;
                                                    unsigned _ldv8_46_6;
                                                    unsigned _ldv8_46_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_46_0), "=r"(_ldv8_46_1), "=r"(_ldv8_46_2), "=r"(_ldv8_46_3), "=r"(_ldv8_46_4), "=r"(_ldv8_46_5), "=r"(_ldv8_46_6), "=r"(_ldv8_46_7) : "l"((const void*)(dpre_tail + (rowr_1))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_46_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_46_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_46_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_46_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_46_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_46_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_46_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_46_7);
                                                }
                                                #pragma unroll
                                                for (int j_1 = 0; j_1 < 8; j_1++) {
                                                    #pragma unroll
                                                    for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                                                        w4[kk_6] = wr[j_1 * 4 + kk_6];
                                                    }
                                                    float _fma_47;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(w4[1]), "f"(tailf[j_1]), "f"(pend[(v % 4 + 4 + 1 - 2) % 4 * 8 + j_1]));
                                                    pend[(v % 4 + 4 + 1 - 2) % 4 * 8 + j_1] = _fma_47;
                                                    float _fma_48;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(w4[0]), "f"(tailf[j_1]), "f"(pend[(v % 4 + 4 + 1 - 3) % 4 * 8 + j_1]));
                                                    pend[(v % 4 + 4 + 1 - 3) % 4 * 8 + j_1] = _fma_48;
                                                }
                                            }
                                            int trow_1 = tail_row_off[frag * 3 + 2];
                                            if (trow_1 >= 0) {
                                                long long rowr_2 = (long long)trow_1 * C64 + c64;
                                                {
                                                    unsigned _ldv8_49_0;
                                                    unsigned _ldv8_49_1;
                                                    unsigned _ldv8_49_2;
                                                    unsigned _ldv8_49_3;
                                                    unsigned _ldv8_49_4;
                                                    unsigned _ldv8_49_5;
                                                    unsigned _ldv8_49_6;
                                                    unsigned _ldv8_49_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_49_0), "=r"(_ldv8_49_1), "=r"(_ldv8_49_2), "=r"(_ldv8_49_3), "=r"(_ldv8_49_4), "=r"(_ldv8_49_5), "=r"(_ldv8_49_6), "=r"(_ldv8_49_7) : "l"((const void*)(dpre_tail + (rowr_2))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_49_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_49_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_49_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_49_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_49_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_49_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_49_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_49_7);
                                                }
                                                #pragma unroll
                                                for (int j_2 = 0; j_2 < 8; j_2++) {
                                                    #pragma unroll
                                                    for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                                                        w4[kk_7] = wr[j_2 * 4 + kk_7];
                                                    }
                                                    float _fma_50;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(w4[0]), "f"(tailf[j_2]), "f"(pend[(v % 4 + 4 + 2 - 3) % 4 * 8 + j_2]));
                                                    pend[(v % 4 + 4 + 2 - 3) % 4 * 8 + j_2] = _fma_50;
                                                }
                                            }
                                            int own_rows = 1;
                                            #pragma unroll
                                            for (int r_s = 0; r_s < 3; r_s++) {
                                                if (skip_row_off[frag * 3 + r_s] >= 0) {
                                                    own_rows = 0;
                                                }
                                            }
                                            #pragma unroll
                                            for (int m = 1; m < 4; m++) {
                                                int tm = tcur - m;
                                                if (own_rows != 0 && tm >= run_t0 && tm < t1 && tm >= doc_start) {
                                                    long long rowm = (long long)tm * C64 + c64;
                                                    unsigned int _packed_4[4];
                                                    #pragma unroll
                                                    for (int _lp = 0; _lp < 1; _lp++) {
                                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 4 - m) % 4 * 8)[_lp*2 + 0], (pend + (v + 4 - m) % 4 * 8)[_lp*2+1 + 0]));
                                                        _packed_4[_lp] = *(uint32_t*)&_bf2;
                                                    }
                                                    #pragma unroll
                                                    for (int _lp = 0; _lp < 1; _lp++) {
                                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 4 - m) % 4 * 8)[_lp*2 + 2], (pend + (v + 4 - m) % 4 * 8)[_lp*2+1 + 2]));
                                                        _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                                                    }
                                                    #pragma unroll
                                                    for (int _lp = 0; _lp < 1; _lp++) {
                                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 4 - m) % 4 * 8)[_lp*2 + 4], (pend + (v + 4 - m) % 4 * 8)[_lp*2+1 + 4]));
                                                        _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                                                    }
                                                    #pragma unroll
                                                    for (int _lp = 0; _lp < 1; _lp++) {
                                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 4 - m) % 4 * 8)[_lp*2 + 6], (pend + (v + 4 - m) % 4 * 8)[_lp*2+1 + 6]));
                                                        _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                                                    }
                                                    reinterpret_cast<int4*>(dx + (rowm))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                                                }
                                            }
                                            #pragma unroll
                                            for (int m2 = 1; m2 < 4; m2++) {
                                                #pragma unroll
                                                for (int j3 = 0; j3 < 8; j3++) {
                                                    win[(v + 4 - m2) % 4 * 8 + j3] = 0.0f;
                                                }
                                            }
                                            if (tcur < S) {
                                                frag = 0;
                                                int frag_hi = num_docs - 1;
                                                while (frag < frag_hi) {
                                                    int mid = frag + frag_hi + 1 >> 1;
                                                    if (tcur >= cu[mid]) {
                                                        frag = mid;
                                                    } else {
                                                        frag_hi = mid - 1;
                                                    }
                                                }
                                                doc_start = cu[frag];
                                                doc_end = cu[frag + 1];
                                                int hrow_2_0 = hist_row_off[frag * 3 + 2];
                                                if (hrow_2_0 >= 0) {
                                                    long long rowh_1_0 = (long long)hrow_2_0 * C64 + c64;
                                                    {
                                                        const uint4* _vptr_51 = reinterpret_cast<const uint4*>(hist + rowh_1_0);
                                                        uint4 _vld_51[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_51[_blk] = _vptr_51[_blk];
                                                            uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
                                                            }
                                                        }
                                                    }
                                                }
                                                int hrow_2_1 = hist_row_off[frag * 3 + 1];
                                                if (hrow_2_1 >= 0) {
                                                    long long rowh_1_1 = (long long)hrow_2_1 * C64 + c64;
                                                    {
                                                        const uint4* _vptr_52 = reinterpret_cast<const uint4*>(hist + rowh_1_1);
                                                        uint4 _vld_52[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_52[_blk] = _vptr_52[_blk];
                                                            uint32_t* _vpairs_52 = reinterpret_cast<uint32_t*>(&_vld_52[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_52[_pair]) & 0xffff0000u);
                                                            }
                                                        }
                                                    }
                                                }
                                                int hrow_2_2 = hist_row_off[frag * 3];
                                                if (hrow_2_2 >= 0) {
                                                    long long rowh_1_2 = (long long)hrow_2_2 * C64 + c64;
                                                    {
                                                        const uint4* _vptr_53 = reinterpret_cast<const uint4*>(hist + rowh_1_2);
                                                        uint4 _vld_53[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_53[_blk] = _vptr_53[_blk];
                                                            uint32_t* _vpairs_53 = reinterpret_cast<uint32_t*>(&_vld_53[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_53[_pair]) & 0xffff0000u);
                                                            }
                                                        }
                                                    }
                                                }
                                            } else {
                                                doc_start = tcur + 4;
                                                doc_end = tcur + 4;
                                            }
                                        }
                                        if (tcur < S) {
                                            if (tcur < t1) {
                                                #pragma unroll
                                                for (int h_0 = 0; h_0 < 4; h_0++) {
                                                    float2 p2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_0 + 3], wr[8 * h_0 + 4 + 3]), make_float2(win[v % 4 * 8 + 2 * h_0], win[v % 4 * 8 + 2 * h_0 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_0 + 2], wr[8 * h_0 + 4 + 2]), make_float2(win[(v % 4 + 3) % 4 * 8 + 2 * h_0], win[(v % 4 + 3) % 4 * 8 + 2 * h_0 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_0 + 1], wr[8 * h_0 + 4 + 1]), make_float2(win[(v % 4 + 2) % 4 * 8 + 2 * h_0], win[(v % 4 + 2) % 4 * 8 + 2 * h_0 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_0], wr[8 * h_0 + 4]), make_float2(win[(v % 4 + 1) % 4 * 8 + 2 * h_0], win[(v % 4 + 1) % 4 * 8 + 2 * h_0 + 1])))));
                                                    float2 _f32x2_mul_b_54 = make_float2(-1.442695f, -1.442695f);
                                                    float2 e2_2;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_2) : "l"(*(const unsigned long long*)&p2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_54));
                                                    float _exp2_55 = approx_exp2(e2_2.x);
                                                    float _exp2_56 = approx_exp2(e2_2.y);
                                                    float2 den2_2 = add_f32x2_noftz(make_float2(_exp2_55, _exp2_56), make_float2(1.0f, 1.0f));
                                                    float _rcp_57 = approx_rcp(den2_2.x);
                                                    float _rcp_58 = approx_rcp(den2_2.y);
                                                    float2 sg2_2 = make_float2(_rcp_57, _rcp_58);
                                                    float2 _f32x2_mul_b_59 = fma_f32x2_rn_noftz(p2_2, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_2), make_float2(1.0f, 1.0f));
                                                    float2 g2_2;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_2) : "l"(*(const unsigned long long*)&sg2_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_59));
                                                    float2 _f32x2_mul_a_60 = make_float2(dyf[2 * h_0], dyf[2 * h_0 + 1]);
                                                    float2 d2_2;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_60), "l"(*(const unsigned long long*)&g2_2));
                                                    dp[2 * h_0] = d2_2.x;
                                                    dp[2 * h_0 + 1] = d2_2.y;
                                                    #pragma unroll
                                                    for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                                                        float2 a2_2 = fma_f32x2_rn_noftz(d2_2, make_float2(win[(v % 4 + 1 + kk_8) % 4 * 8 + 2 * h_0], win[(v % 4 + 1 + kk_8) % 4 * 8 + 2 * h_0 + 1]), make_float2(acc[8 * h_0 + kk_8], acc[8 * h_0 + 4 + kk_8]));
                                                        acc[8 * h_0 + kk_8] = a2_2.x;
                                                        acc[8 * h_0 + 4 + kk_8] = a2_2.y;
                                                    }
                                                    #pragma unroll
                                                    for (int k_2 = 0; k_2 < 3; k_2++) {
                                                        float2 q2_2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_0 + k_2], wr[8 * h_0 + 4 + k_2]), d2_2, make_float2(pend[(v % 4 + 1 + k_2) % 4 * 8 + 2 * h_0], pend[(v % 4 + 1 + k_2) % 4 * 8 + 2 * h_0 + 1]));
                                                        pend[(v % 4 + 1 + k_2) % 4 * 8 + 2 * h_0] = q2_2.x;
                                                        pend[(v % 4 + 1 + k_2) % 4 * 8 + 2 * h_0 + 1] = q2_2.y;
                                                    }
                                                    float2 _f32x2_mul_a_61 = make_float2(wr[8 * h_0 + 3], wr[8 * h_0 + 4 + 3]);
                                                    float2 r2_2;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_61), "l"(*(const unsigned long long*)&d2_2));
                                                    pend[v % 4 * 8 + 2 * h_0] = r2_2.x;
                                                    pend[v % 4 * 8 + 2 * h_0 + 1] = r2_2.y;
                                                }
                                            } else {
                                                #pragma unroll
                                                for (int h_1_1 = 0; h_1_1 < 4; h_1_1++) {
                                                    float2 p2_3 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1_1 + 3], wr[8 * h_1_1 + 4 + 3]), make_float2(win[v % 4 * 8 + 2 * h_1_1], win[v % 4 * 8 + 2 * h_1_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1_1 + 2], wr[8 * h_1_1 + 4 + 2]), make_float2(win[(v % 4 + 3) % 4 * 8 + 2 * h_1_1], win[(v % 4 + 3) % 4 * 8 + 2 * h_1_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1_1 + 1], wr[8 * h_1_1 + 4 + 1]), make_float2(win[(v % 4 + 2) % 4 * 8 + 2 * h_1_1], win[(v % 4 + 2) % 4 * 8 + 2 * h_1_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_1_1], wr[8 * h_1_1 + 4]), make_float2(win[(v % 4 + 1) % 4 * 8 + 2 * h_1_1], win[(v % 4 + 1) % 4 * 8 + 2 * h_1_1 + 1])))));
                                                    float2 _f32x2_mul_b_62 = make_float2(-1.442695f, -1.442695f);
                                                    float2 e2_3;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_3) : "l"(*(const unsigned long long*)&p2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_62));
                                                    float _exp2_63 = approx_exp2(e2_3.x);
                                                    float _exp2_64 = approx_exp2(e2_3.y);
                                                    float2 den2_3 = add_f32x2_noftz(make_float2(_exp2_63, _exp2_64), make_float2(1.0f, 1.0f));
                                                    float _rcp_65 = approx_rcp(den2_3.x);
                                                    float _rcp_66 = approx_rcp(den2_3.y);
                                                    float2 sg2_3 = make_float2(_rcp_65, _rcp_66);
                                                    float2 _f32x2_mul_b_67 = fma_f32x2_rn_noftz(p2_3, sub_f32x2_noftz(make_float2(1.0f, 1.0f), sg2_3), make_float2(1.0f, 1.0f));
                                                    float2 g2_3;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&g2_3) : "l"(*(const unsigned long long*)&sg2_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_67));
                                                    float2 _f32x2_mul_a_68 = make_float2(dyf[2 * h_1_1], dyf[2 * h_1_1 + 1]);
                                                    float2 d2_3;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&d2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_68), "l"(*(const unsigned long long*)&g2_3));
                                                    dp[2 * h_1_1] = d2_3.x;
                                                    dp[2 * h_1_1 + 1] = d2_3.y;
                                                    #pragma unroll
                                                    for (int k_3 = 0; k_3 < 3; k_3++) {
                                                        float2 q2_3 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1_1 + k_3], wr[8 * h_1_1 + 4 + k_3]), d2_3, make_float2(pend[(v % 4 + 1 + k_3) % 4 * 8 + 2 * h_1_1], pend[(v % 4 + 1 + k_3) % 4 * 8 + 2 * h_1_1 + 1]));
                                                        pend[(v % 4 + 1 + k_3) % 4 * 8 + 2 * h_1_1] = q2_3.x;
                                                        pend[(v % 4 + 1 + k_3) % 4 * 8 + 2 * h_1_1 + 1] = q2_3.y;
                                                    }
                                                    float2 _f32x2_mul_a_69 = make_float2(wr[8 * h_1_1 + 3], wr[8 * h_1_1 + 4 + 3]);
                                                    float2 r2_3;
                                                    asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&r2_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_69), "l"(*(const unsigned long long*)&d2_3));
                                                    pend[v % 4 * 8 + 2 * h_1_1] = r2_3.x;
                                                    pend[v % 4 * 8 + 2 * h_1_1 + 1] = r2_3.y;
                                                }
                                            }
                                            int t3 = tcur - 3;
                                            if (t3 >= run_t0 && t3 >= doc_start) {
                                                long long row3_2 = (long long)t3 * C64 + c64;
                                                unsigned int _packed_5[4];
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 1) % 4 * 8)[_lp*2 + 0], (pend + (v + 1) % 4 * 8)[_lp*2+1 + 0]));
                                                    _packed_5[_lp] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 1) % 4 * 8)[_lp*2 + 2], (pend + (v + 1) % 4 * 8)[_lp*2+1 + 2]));
                                                    _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 1) % 4 * 8)[_lp*2 + 4], (pend + (v + 1) % 4 * 8)[_lp*2+1 + 4]));
                                                    _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 1; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((pend + (v + 1) % 4 * 8)[_lp*2 + 6], (pend + (v + 1) % 4 * 8)[_lp*2+1 + 6]));
                                                    _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                                                }
                                                reinterpret_cast<int4*>(dx + (row3_2))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    __syncwarp();
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (kq + 3 < k_lim) {
                        int _max_70 = ((t_fill - (run_t0 + (kq + 3 - k_run) * 8)) > (0) ? (t_fill - (run_t0 + (kq + 3 - k_run) * 8)) : (0));
                        int nrows_3 = ((_max_70) < (8) ? (_max_70) : (8));
                        int sb_3 = (int)stage_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8192;
                        if (lane == 0 && nrows_3 == 8 && wbytes == 512) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8, 8192);
                            tma_2d_gmem2smem(sb_3, (&x_map), cw, run_t0 + (kq + 3 - k_run) * 8, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8);
                            tma_2d_gmem2smem(sb_3 + 4096, (&dy_map), cw, run_t0 + (kq + 3 - k_run) * 8, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8);
                        } else if (lane == 0) {
                            mbarrier_arrive_expect_tx(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8, nrows_3 * (2 * wbytes));
                            #pragma unroll 1
                            for (int irow_3 = 0; irow_3 < nrows_3; irow_3++) {
                                long long row64_3 = (long long)(run_t0 + (kq + 3 - k_run) * 8 + irow_3) * (long long)C + (long long)cw;
                                cp_async_bulk_gmem2smem(sb_3 + irow_3 * 512, reinterpret_cast<const uint8_t*>(x + row64_3), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8);
                                cp_async_bulk_gmem2smem(sb_3 + 4096 + irow_3 * 512, reinterpret_cast<const uint8_t*>(dy + row64_3), wbytes, d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8);
                            }
                        } else {
                            mbarrier_arrive(d_full_addr + (warp_1 + 8 * (int)((unsigned int)(kq + 3) % 3)) * 8);
                        }
                    }
                }
                kc += n_st;
                int _shfl_sync_71 = __shfl_sync(0xFFFFFFFF, dl[0], 2);
                fl = _shfl_sync_71;
                int _shfl_sync_72 = __shfl_sync(0xFFFFFFFF, dl[0], 0);
                gl = _shfl_sync_72;
                int _shfl_sync_73 = __shfl_sync(0xFFFFFFFF, dl[0], 1);
                un = _shfl_sync_73;
                int u_first = __shfl_sync(0xFFFFFFFF, dl[0], 4);
                is_last = fl / 2 % 2;
                is_end = fl / 16 % 2;
                int is_owner = fl / 4 % 2;
                ct = gl / P;
                c0 = ct * 2048 + tid * 8;
                if (is_last != 0 || is_end != 0) {
                    if (c0 < C) {
                        #pragma unroll
                        for (int j4 = 0; j4 < 8; j4++) {
                            asm volatile("red.global.add.v4.f32 [%0], {%1, %2, %3, %4};" :: "l"(reinterpret_cast<uint64_t>(&dw_acc[((long long)(bid % 4) * C64 + c64 + (long long)j4) * 4ll])), "f"(acc[j4 * 4]), "f"(acc[j4 * 4 + 1]), "f"(acc[j4 * 4 + 2]), "f"(acc[j4 * 4 + 3]) : "memory");
                        }
                    }
                    int wgt = un - u_first + 1;
                    if (is_owner != 0) {
                        int lu_t = gl % P * (NL * M);
                        int _max_75 = ((S - lu_t + M - 1) > (0) ? (S - lu_t + M - 1) : (0));
                        int _min_74 = ((NL) < (_max_75 / M) ? (NL) : (_max_75 / M));
                        wgt = un + 2 + (NL - _min_74);
                    }
                    __syncwarp();
                    int l1 = 0;
                    if (lane == 0) {
                        unsigned int at_old;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(at_old) : "l"(&tile_tk[ct * 8 + warp_1]), "r"(static_cast<uint32_t>((unsigned int)wgt)) : "memory");
                        int at_last = 0;
                        if (at_old + (unsigned int)wgt == (unsigned int)(P * (NL + 1))) {
                            at_last = 1;
                            tile_tk[ct * 8 + warp_1] = 0;
                        }
                        l1 = at_last;
                    }
                    int _shfl_sync_76 = __shfl_sync(0xFFFFFFFF, l1, 0);
                    if (_shfl_sync_76 != 0) {
                        fin = fin | 1 << (unsigned int)ct;
                    }
                }
            }
            __syncwarp();
            #pragma unroll 1
            for (int tt = 0; tt < (C + 2048 - 1) / 2048; tt++) {
                if ((fin >> (unsigned int)tt & 1) != 0) {
                    int cw5 = tt * 2048 + warp_1 * 256 + lane;
                    #pragma unroll
                    for (int j5 = 0; j5 < 8; j5++) {
                        int ch5 = cw5 + j5 * 32;
                        if (ch5 < C) {
                            #pragma unroll
                            for (int q5 = 0; q5 < 4; q5++) {
                                {
                                    unsigned _v4_77_0;
                                    unsigned _v4_77_1;
                                    unsigned _v4_77_2;
                                    unsigned _v4_77_3;
                                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_v4_77_0), "=r"(_v4_77_1), "=r"(_v4_77_2), "=r"(_v4_77_3) : "l"((const void*)(dw_acc + (((long long)q5 * C64 + (long long)ch5) * 4ll))) : "memory");
                                    lw[(j5 * 4 + q5) * 4 + 0] = __uint_as_float(_v4_77_0);
                                    lw[(j5 * 4 + q5) * 4 + 1] = __uint_as_float(_v4_77_1);
                                    lw[(j5 * 4 + q5) * 4 + 2] = __uint_as_float(_v4_77_2);
                                    lw[(j5 * 4 + q5) * 4 + 3] = __uint_as_float(_v4_77_3);
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int j6 = 0; j6 < 8; j6++) {
                        int ch6 = cw5 + j6 * 32;
                        if (ch6 < C) {
                            #pragma unroll
                            for (int e6 = 0; e6 < 4; e6++) {
                                a4[e6] = lw[j6 * 4 * 4 + e6];
                                lo4[e6] = 0.0f;
                                #pragma unroll
                                for (int q6 = 1; q6 < 4; q6++) {
                                    float df_s = a4[e6] + lw[(j6 * 4 + q6) * 4 + e6];
                                    float df_b = df_s - a4[e6];
                                    float df_r = a4[e6] - (df_s - df_b) + (lw[(j6 * 4 + q6) * 4 + e6] - df_b);
                                    a4[e6] = df_s;
                                    lo4[e6] = lo4[e6] + df_r;
                                }
                                a4[e6] = a4[e6] + lo4[e6];
                            }
                            {
                                float4 _v4 = make_float4(a4[0 + 0], a4[0 + 1], a4[0 + 2], a4[0 + 3]);
                                *reinterpret_cast<float4*>(dw + (long long)ch6 * 4ll) = _v4;
                            }
                        }
                    }
                    #pragma unroll
                    for (int e7 = 0; e7 < 4; e7++) {
                        a4[e7] = 0.0f;
                    }
                    #pragma unroll
                    for (int j7 = 0; j7 < 8; j7++) {
                        int ch7 = cw5 + j7 * 32;
                        if (ch7 < C) {
                            #pragma unroll
                            for (int q7 = 0; q7 < 4; q7++) {
                                {
                                    float4 _v4 = make_float4(a4[0 + 0], a4[0 + 1], a4[0 + 2], a4[0 + 3]);
                                    *reinterpret_cast<float4*>(dw_acc + ((long long)q7 * C64 + (long long)ch7) * 4ll) = _v4;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: sched ----
    if (warp == 8) {
        { // sched_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            unsigned int cv[1];
            int g_own = -1;
            if (bid < L) {
                g_own = bid;
            }
            unsigned int _phase_e_empty = 1;
            unsigned int _phase_w_empty = 1;
            unsigned int _phase_w_full = 0;
            #pragma unroll 1
            for (int _c = 0; _c < n_ctas + 1; _c++) {
                if (g_own >= 0) {
                    int lu_t_1 = g_own % P * (NL * M);
                    int _max_0_1 = ((S - lu_t_1 + M - 1) > (0) ? (S - lu_t_1 + M - 1) : (0));
                    int nlo = ((NL) < (_max_0_1 / M) ? (NL) : (_max_0_1 / M));
                    int first = 1;
                    int prev = -1;
                    #pragma unroll 1
                    for (int _o = 0; _o < NL + 1; _o++) {
                        mbarrier_wait_hint(e_empty_addr, _phase_e_empty, 10000000);
                        int cf = -1;
                        if (lane == 0) {
                            unsigned int cf_old = atomicAdd(&claims[g_own], 1);
                            int cf_f = (int)(cf_old & 65535);
                            int cf_b = (int)(cf_old >> 16);
                            if (cf_f + cf_b < nlo) {
                                cf = cf_f;
                            }
                        }
                        int cur = __shfl_sync(0xFFFFFFFF, cf, 0);
                        int fo = 4 + first;
                        if (cur < 0) {
                            if (first != 0) {
                                int fr = 0;
                                int ds = 0;
                                int de = 0;
                                int h0 = -1;
                                int h1 = -1;
                                int h2 = -1;
                                if (elect_sync()) {
                                    int da = (int)desc_addr + (int)0 * 64;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da), "r"(g_own), "r"(-1), "r"(7), "r"(0) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da + 16), "r"(0), "r"(fr), "r"(ds), "r"(de) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da + 32), "r"(h0), "r"(h1), "r"(h2), "r"(0) : "memory");
                                    mbarrier_arrive(e_full_addr);
                                }
                            } else {
                                int fr_1 = 0;
                                int ds_1 = 0;
                                int de_1 = 0;
                                int h0_1 = -1;
                                int h1_1 = -1;
                                int h2_1 = -1;
                                if (0 && prev >= 0) {
                                    int t0m = g_own % P * (NL * M) + prev * M;
                                    fr_1 = 0;
                                    int fr_1_hi = num_docs - 1;
                                    while (fr_1 < fr_1_hi) {
                                        int mid_1 = fr_1 + fr_1_hi + 1 >> 1;
                                        if (t0m >= cu[mid_1]) {
                                            fr_1 = mid_1;
                                        } else {
                                            fr_1_hi = mid_1 - 1;
                                        }
                                    }
                                    ds_1 = cu[fr_1];
                                    de_1 = cu[fr_1 + 1];
                                    if (t0m - 1 >= ds_1) {
                                        h0_1 = t0m - 1;
                                    } else {
                                        int hr0 = hist_row_off[fr_1 * 3 + 2 - (ds_1 - t0m)];
                                        if (hr0 >= 0) {
                                            h0_1 = -2 - hr0;
                                        }
                                    }
                                    if (t0m - 2 >= ds_1) {
                                        h1_1 = t0m - 2;
                                    } else {
                                        int hr1 = hist_row_off[fr_1 * 3 + 1 - (ds_1 - t0m)];
                                        if (hr1 >= 0) {
                                            h1_1 = -2 - hr1;
                                        }
                                    }
                                    if (t0m - 3 >= ds_1) {
                                        h2_1 = t0m - 3;
                                    } else {
                                        int hr2 = hist_row_off[fr_1 * 3 - (ds_1 - t0m)];
                                        if (hr2 >= 0) {
                                            h2_1 = -2 - hr2;
                                        }
                                    }
                                }
                                if (elect_sync()) {
                                    int da_1 = (int)desc_addr + (int)0 * 64;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_1), "r"(g_own), "r"(prev), "r"(20), "r"(0) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_1 + 16), "r"(0), "r"(fr_1), "r"(ds_1), "r"(de_1) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_1 + 32), "r"(h0_1), "r"(h1_1), "r"(h2_1), "r"(0) : "memory");
                                    mbarrier_arrive(e_full_addr);
                                }
                            }
                        } else {
                            if (cur == nlo - 1) {
                                fo = fo + 2;
                            }
                            int fr_2 = 0;
                            int ds_2 = 0;
                            int de_2 = 0;
                            int h0_2 = -1;
                            int h1_2 = -1;
                            int h2_2 = -1;
                            if (fo % 2 == 1 && cur >= 0) {
                                int t0m_1 = g_own % P * (NL * M) + cur * M;
                                fr_2 = 0;
                                int fr_2_hi = num_docs - 1;
                                while (fr_2 < fr_2_hi) {
                                    int mid_2 = fr_2 + fr_2_hi + 1 >> 1;
                                    if (t0m_1 >= cu[mid_2]) {
                                        fr_2 = mid_2;
                                    } else {
                                        fr_2_hi = mid_2 - 1;
                                    }
                                }
                                ds_2 = cu[fr_2];
                                de_2 = cu[fr_2 + 1];
                                if (t0m_1 - 1 >= ds_2) {
                                    h0_2 = t0m_1 - 1;
                                } else {
                                    int hr0_1 = hist_row_off[fr_2 * 3 + 2 - (ds_2 - t0m_1)];
                                    if (hr0_1 >= 0) {
                                        h0_2 = -2 - hr0_1;
                                    }
                                }
                                if (t0m_1 - 2 >= ds_2) {
                                    h1_2 = t0m_1 - 2;
                                } else {
                                    int hr1_1 = hist_row_off[fr_2 * 3 + 1 - (ds_2 - t0m_1)];
                                    if (hr1_1 >= 0) {
                                        h1_2 = -2 - hr1_1;
                                    }
                                }
                                if (t0m_1 - 3 >= ds_2) {
                                    h2_2 = t0m_1 - 3;
                                } else {
                                    int hr2_1 = hist_row_off[fr_2 * 3 - (ds_2 - t0m_1)];
                                    if (hr2_1 >= 0) {
                                        h2_2 = -2 - hr2_1;
                                    }
                                }
                            }
                            if (elect_sync()) {
                                int da_2 = (int)desc_addr + (int)0 * 64;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_2), "r"(g_own), "r"(cur), "r"(fo), "r"(0) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_2 + 16), "r"(0), "r"(fr_2), "r"(ds_2), "r"(de_2) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_2 + 32), "r"(h0_2), "r"(h1_2), "r"(h2_2), "r"(0) : "memory");
                                mbarrier_arrive(e_full_addr);
                            }
                        }
                        _phase_e_empty ^= 1;
                        if (cur < 0) {
                            break;
                        }
                        first = 0;
                        prev = cur;
                        if (cur == nlo - 1) {
                            break;
                        }
                    }
                }
                g_own = -1;
                mbarrier_wait_hint(w_empty_addr, _phase_w_empty, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(w_full_addr, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(rsp_addr + 0 * 16 + 0 * 16), "r"(w_full_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_wait(w_full_addr, _phase_w_full);
                uint32_t valid = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(valid)
                    : "r"(rsp_addr + 0 * 16 + 0 * 16)
                    : "memory");
                uint32_t cta = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(cta)
                    : "r"(rsp_addr + 0 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(w_empty_addr);
                _phase_w_empty ^= 1;
                _phase_w_full ^= 1;
                if (valid == 0) {
                    break;
                }
                if (lane == 0) {
                    uint32_t gone;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(gone) : "l"(&departs[0]), "r"(static_cast<uint32_t>((unsigned int)n_ctas - 1)) : "memory");
                }
                if (cta < (unsigned int)L) {
                    g_own = (int)cta;
                }
            }
            int rot = bid % L;
            #pragma unroll 1
            for (int _s = 0; _s < L * NL + n_ctas; _s++) {
                mbarrier_wait_hint(e_empty_addr, _phase_e_empty, 10000000);
                int best = -1;
                int tot = 0;
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    int gg = lane + 32 * j_3;
                    if (gg < L) {
                        {
                            uint32_t _scalar_bits_1;
                            asm volatile("ld.global.cg.b32 %0, [%1];"
                                : "=r"(_scalar_bits_1) : "l"((const void*)(claims + (gg))) : "memory");
                            cv[0] = (unsigned int)_scalar_bits_1;
                        }
                        int lu_t_2 = gg % P * (NL * M);
                        int _max_3 = ((S - lu_t_2 + M - 1) > (0) ? (S - lu_t_2 + M - 1) : (0));
                        int _min_2 = ((NL) < (_max_3 / M) ? (NL) : (_max_3 / M));
                        int rq = _min_2 - (int)(cv[0] & 65535) - (int)(cv[0] >> 16);
                        if (rq > 0) {
                            tot = tot + rq;
                            int rk = gg - rot;
                            if (rk < 0) {
                                rk = rk + L;
                            }
                            int kx = rq * 1024 + (1023 - rk);
                            if (kx > best) {
                                best = kx;
                            }
                        }
                    }
                }
                int _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, best, 16);
                int _max_4 = ((best) > (_shfl_xor_sync_5) ? (best) : (_shfl_xor_sync_5));
                best = _max_4;
                int _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, tot, 16);
                tot = tot + _shfl_xor_sync_6;
                int _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, best, 8);
                int _max_7 = ((best) > (_shfl_xor_sync_8) ? (best) : (_shfl_xor_sync_8));
                best = _max_7;
                int _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, tot, 8);
                tot = tot + _shfl_xor_sync_9;
                int _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, best, 4);
                int _max_10 = ((best) > (_shfl_xor_sync_11) ? (best) : (_shfl_xor_sync_11));
                best = _max_10;
                int _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, tot, 4);
                tot = tot + _shfl_xor_sync_12;
                int _shfl_xor_sync_14 = __shfl_xor_sync(0xFFFFFFFF, best, 2);
                int _max_13_1 = ((best) > (_shfl_xor_sync_14) ? (best) : (_shfl_xor_sync_14));
                best = _max_13_1;
                int _shfl_xor_sync_15 = __shfl_xor_sync(0xFFFFFFFF, tot, 2);
                tot = tot + _shfl_xor_sync_15;
                int _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, best, 1);
                int _max_16 = ((best) > (_shfl_xor_sync_17) ? (best) : (_shfl_xor_sync_17));
                best = _max_16;
                int _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, tot, 1);
                tot = tot + _shfl_xor_sync_18;
                if (best < 0) {
                    break;
                }
                int gs = 1023 - best % 1024 + rot;
                if (gs >= L) {
                    gs = gs - L;
                }
                int _min_19 = ((best / 1024 / 2) < ((tot + n_ctas - 1) / n_ctas) ? (best / 1024 / 2) : ((tot + n_ctas - 1) / n_ctas));
                int kt = ((1) > (_min_19) ? (1) : (_min_19));
                int lo_s = 0;
                int hi_s = 0;
                if (lane == 0) {
                    unsigned int st_old = atomicAdd(&claims[gs], (unsigned int)kt << 16);
                    int lu_t_3 = gs % P * (NL * M);
                    int _max_21 = ((S - lu_t_3 + M - 1) > (0) ? (S - lu_t_3 + M - 1) : (0));
                    int _min_20 = ((NL) < (_max_21 / M) ? (NL) : (_max_21 / M));
                    hi_s = _min_20 - (int)(st_old >> 16);
                    int _max_22 = (((int)(st_old & 65535)) > (hi_s - kt) ? ((int)(st_old & 65535)) : (hi_s - kt));
                    lo_s = _max_22;
                }
                int _shfl_sync_23 = __shfl_sync(0xFFFFFFFF, lo_s, 0);
                lo_s = _shfl_sync_23;
                int _shfl_sync_24 = __shfl_sync(0xFFFFFFFF, hi_s, 0);
                hi_s = _shfl_sync_24;
                #pragma unroll 1
                for (int uu = lo_s; uu < hi_s; uu++) {
                    int fs = 0;
                    if (uu == lo_s) {
                        fs = 1;
                    }
                    if (uu == hi_s - 1) {
                        fs = fs + 2;
                    }
                    mbarrier_wait_hint(e_empty_addr, _phase_e_empty, 10000000);
                    int fr_3 = 0;
                    int ds_3 = 0;
                    int de_3 = 0;
                    int h0_3 = -1;
                    int h1_3 = -1;
                    int h2_3 = -1;
                    if (fs % 2 == 1 && uu >= 0) {
                        int t0m_2 = gs % P * (NL * M) + uu * M;
                        fr_3 = 0;
                        int fr_3_hi = num_docs - 1;
                        while (fr_3 < fr_3_hi) {
                            int mid_3 = fr_3 + fr_3_hi + 1 >> 1;
                            if (t0m_2 >= cu[mid_3]) {
                                fr_3 = mid_3;
                            } else {
                                fr_3_hi = mid_3 - 1;
                            }
                        }
                        ds_3 = cu[fr_3];
                        de_3 = cu[fr_3 + 1];
                        if (t0m_2 - 1 >= ds_3) {
                            h0_3 = t0m_2 - 1;
                        } else {
                            int hr0_2 = hist_row_off[fr_3 * 3 + 2 - (ds_3 - t0m_2)];
                            if (hr0_2 >= 0) {
                                h0_3 = -2 - hr0_2;
                            }
                        }
                        if (t0m_2 - 2 >= ds_3) {
                            h1_3 = t0m_2 - 2;
                        } else {
                            int hr1_2 = hist_row_off[fr_3 * 3 + 1 - (ds_3 - t0m_2)];
                            if (hr1_2 >= 0) {
                                h1_3 = -2 - hr1_2;
                            }
                        }
                        if (t0m_2 - 3 >= ds_3) {
                            h2_3 = t0m_2 - 3;
                        } else {
                            int hr2_2 = hist_row_off[fr_3 * 3 - (ds_3 - t0m_2)];
                            if (hr2_2 >= 0) {
                                h2_3 = -2 - hr2_2;
                            }
                        }
                    }
                    if (elect_sync()) {
                        int da_3 = (int)desc_addr + (int)0 * 64;
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_3), "r"(gs), "r"(uu), "r"(fs), "r"(0) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_3 + 16), "r"(lo_s), "r"(fr_3), "r"(ds_3), "r"(de_3) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_3 + 32), "r"(h0_3), "r"(h1_3), "r"(h2_3), "r"(0) : "memory");
                        mbarrier_arrive(e_full_addr);
                    }
                    _phase_e_empty ^= 1;
                }
            }
            mbarrier_wait_hint(e_empty_addr, _phase_e_empty, 10000000);
            int fr_4 = 0;
            int ds_4 = 0;
            int de_4 = 0;
            int h0_4 = -1;
            int h1_4 = -1;
            int h2_4 = -1;
            if (elect_sync()) {
                int da_4 = (int)desc_addr + (int)0 * 64;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_4), "r"(0), "r"(-1), "r"(8), "r"(0) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_4 + 16), "r"(0), "r"(fr_4), "r"(ds_4), "r"(de_4) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(da_4 + 32), "r"(h0_4), "r"(h1_4), "r"(h2_4), "r"(0) : "memory");
                mbarrier_arrive(e_full_addr);
            }
            _phase_e_empty ^= 1;
            __syncwarp();
            if (lane == 0) {
                uint32_t done;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(done) : "l"(&departs[0]), "r"(static_cast<uint32_t>((unsigned int)n_ctas - 1)) : "memory");
                if (done == (unsigned int)n_ctas - 1) {
                    #pragma unroll 1
                    for (int zq = 0; zq < L; zq++) {
                        claims[zq] = 0;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
