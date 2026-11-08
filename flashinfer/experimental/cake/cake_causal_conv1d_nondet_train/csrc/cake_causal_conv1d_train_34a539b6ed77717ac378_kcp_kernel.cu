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


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_34a539b6ed77717ac378_kcp(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ dy, __nv_bfloat16* __restrict__ dx, float* __restrict__ dw, float* dw_acc, unsigned int* __restrict__ tile_tk, __nv_bfloat16* __restrict__ hist, int* __restrict__ hist_row_off, float* __restrict__ dpre_tail, int* __restrict__ tail_row_off, unsigned int* claims, int S, int C, int num_docs, int M, int NL, int P, int L, int n_ctas, const __grid_constant__ CUtensorMap x_map, const __grid_constant__ CUtensorMap dy_map)
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
                                for (int j = 0; j < 8; j++) {
                                    #pragma unroll
                                    for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                                        w4[kk_1] = wr[j * 4 + kk_1];
                                    }
                                    float _fma_23;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(w4[1]), "f"(win[(u % 4 + 2) % 4 * 8 + j]), "f"(mul_f32_rn_noftz(w4[0], win[(u % 4 + 1) % 4 * 8 + j])));
                                    float _fma_22;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(w4[2]), "f"(win[(u % 4 + 3) % 4 * 8 + j]), "f"(_fma_23));
                                    float fma;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma) : "f"(w4[3]), "f"(win[u % 4 * 8 + j]), "f"(_fma_22));
                                    float e = -fma * 1.442695f;
                                    float _exp2_24 = approx_exp2(e);
                                    float den = 1.0f + _exp2_24;
                                    float sg = approx_rcp(den);
                                    float _fma_25;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(fma), "f"(1.0f - sg), "f"(1.0f));
                                    dp[j] = dyf[j] * (sg * _fma_25);
                                    #pragma unroll
                                    for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                                        acc[j * 4 + kk_2] = __fmaf_rn(dp[j], win[(u % 4 + 1 + kk_2) % 4 * 8 + j], acc[j * 4 + kk_2]);
                                    }
                                    float _fma_26;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(w4[0]), "f"(dp[j]), "f"(pend[(u % 4 + 1) % 4 * 8 + j]));
                                    pend[(u % 4 + 1) % 4 * 8 + j] = _fma_26;
                                    float _fma_27;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(w4[1]), "f"(dp[j]), "f"(pend[(u % 4 + 2) % 4 * 8 + j]));
                                    pend[(u % 4 + 2) % 4 * 8 + j] = _fma_27;
                                    float _fma_28;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(w4[2]), "f"(dp[j]), "f"(pend[(u % 4 + 3) % 4 * 8 + j]));
                                    pend[(u % 4 + 3) % 4 * 8 + j] = _fma_28;
                                    pend[u % 4 * 8 + j] = w4[3] * dp[j];
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
                                __int128_t _smem_raw_29;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_29) : "r"(sx0 + u_1 * 512));
                                pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[0];
                                pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[1];
                                pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[2];
                                pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_29)[3];
                                __int128_t _smem_raw_30;
                                asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_30) : "r"(sx0 + 4096 + u_1 * 512));
                                pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[0];
                                pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[1];
                                pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[2];
                                pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_30)[3];
                                #pragma unroll
                                for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                    win[u_1 % 4 * 8 + 2 * kk_3] = __uint_as_float(pkx[kk_3] << 16);
                                    win[u_1 % 4 * 8 + 2 * kk_3 + 1] = __uint_as_float(pkx[kk_3] & 4294901760u);
                                    dyf[2 * kk_3] = __uint_as_float(pkd[kk_3] << 16);
                                    dyf[2 * kk_3 + 1] = __uint_as_float(pkd[kk_3] & 4294901760u);
                                }
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 8; j_1++) {
                                    #pragma unroll
                                    for (int kk_4 = 0; kk_4 < 4; kk_4++) {
                                        w4[kk_4] = wr[j_1 * 4 + kk_4];
                                    }
                                    float _fma_32;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(w4[1]), "f"(win[(u_1 % 4 + 2) % 4 * 8 + j_1]), "f"(mul_f32_rn_noftz(w4[0], win[(u_1 % 4 + 1) % 4 * 8 + j_1])));
                                    float _fma_31;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(w4[2]), "f"(win[(u_1 % 4 + 3) % 4 * 8 + j_1]), "f"(_fma_32));
                                    float fma_1;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_1) : "f"(w4[3]), "f"(win[u_1 % 4 * 8 + j_1]), "f"(_fma_31));
                                    float e_1 = -fma_1 * 1.442695f;
                                    float _exp2_33 = approx_exp2(e_1);
                                    float den_1 = 1.0f + _exp2_33;
                                    float sg_1 = approx_rcp(den_1);
                                    float _fma_34;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(fma_1), "f"(1.0f - sg_1), "f"(1.0f));
                                    dp[j_1] = dyf[j_1] * (sg_1 * _fma_34);
                                    #pragma unroll
                                    for (int kk_5 = 0; kk_5 < 4; kk_5++) {
                                        acc[j_1 * 4 + kk_5] = __fmaf_rn(dp[j_1], win[(u_1 % 4 + 1 + kk_5) % 4 * 8 + j_1], acc[j_1 * 4 + kk_5]);
                                    }
                                    float _fma_35;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(w4[0]), "f"(dp[j_1]), "f"(pend[(u_1 % 4 + 1) % 4 * 8 + j_1]));
                                    pend[(u_1 % 4 + 1) % 4 * 8 + j_1] = _fma_35;
                                    float _fma_36;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(w4[1]), "f"(dp[j_1]), "f"(pend[(u_1 % 4 + 2) % 4 * 8 + j_1]));
                                    pend[(u_1 % 4 + 2) % 4 * 8 + j_1] = _fma_36;
                                    float _fma_37;
                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(w4[2]), "f"(dp[j_1]), "f"(pend[(u_1 % 4 + 3) % 4 * 8 + j_1]));
                                    pend[(u_1 % 4 + 3) % 4 * 8 + j_1] = _fma_37;
                                    pend[u_1 % 4 * 8 + j_1] = w4[3] * dp[j_1];
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
                            for (int h = 0; h < 2; h++) {
                                #pragma unroll
                                for (int v = 0; v < 4; v++) {
                                    int u_2 = h * 4 + v;
                                    if (steps_k > u_2) {
                                        int tcur = tb + u_2;
                                        if (rows_k > u_2) {
                                            __int128_t _smem_raw_38;
                                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_38) : "r"(sx0 + u_2 * 512));
                                            pkx[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[0];
                                            pkx[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[1];
                                            pkx[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[2];
                                            pkx[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_38)[3];
                                            __int128_t _smem_raw_39;
                                            asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_39) : "r"(sx0 + 4096 + u_2 * 512));
                                            pkd[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[0];
                                            pkd[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[1];
                                            pkd[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[2];
                                            pkd[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_39)[3];
                                            #pragma unroll
                                            for (int kk_6 = 0; kk_6 < 4; kk_6++) {
                                                win[v * 8 + 2 * kk_6] = __uint_as_float(pkx[kk_6] << 16);
                                                win[v * 8 + 2 * kk_6 + 1] = __uint_as_float(pkx[kk_6] & 4294901760u);
                                                dyf[2 * kk_6] = __uint_as_float(pkd[kk_6] << 16);
                                                dyf[2 * kk_6 + 1] = __uint_as_float(pkd[kk_6] & 4294901760u);
                                            }
                                        }
                                        if (tcur >= doc_end) {
                                            int trow = tail_row_off[frag * 3];
                                            if (trow >= 0) {
                                                long long rowr = (long long)trow * C64 + c64;
                                                {
                                                    unsigned _ldv8_40_0;
                                                    unsigned _ldv8_40_1;
                                                    unsigned _ldv8_40_2;
                                                    unsigned _ldv8_40_3;
                                                    unsigned _ldv8_40_4;
                                                    unsigned _ldv8_40_5;
                                                    unsigned _ldv8_40_6;
                                                    unsigned _ldv8_40_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_40_0), "=r"(_ldv8_40_1), "=r"(_ldv8_40_2), "=r"(_ldv8_40_3), "=r"(_ldv8_40_4), "=r"(_ldv8_40_5), "=r"(_ldv8_40_6), "=r"(_ldv8_40_7) : "l"((const void*)(dpre_tail + (rowr))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_40_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_40_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_40_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_40_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_40_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_40_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_40_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_40_7);
                                                }
                                                #pragma unroll
                                                for (int j_2 = 0; j_2 < 8; j_2++) {
                                                    #pragma unroll
                                                    for (int kk_7 = 0; kk_7 < 4; kk_7++) {
                                                        w4[kk_7] = wr[j_2 * 4 + kk_7];
                                                    }
                                                    float _fma_41;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(w4[2]), "f"(tailf[j_2]), "f"(pend[(v % 4 + 4 - 1) % 4 * 8 + j_2]));
                                                    pend[(v % 4 + 4 - 1) % 4 * 8 + j_2] = _fma_41;
                                                    float _fma_42;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(w4[1]), "f"(tailf[j_2]), "f"(pend[(v % 4 + 4 - 2) % 4 * 8 + j_2]));
                                                    pend[(v % 4 + 4 - 2) % 4 * 8 + j_2] = _fma_42;
                                                    float _fma_43;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(w4[0]), "f"(tailf[j_2]), "f"(pend[(v % 4 + 4 - 3) % 4 * 8 + j_2]));
                                                    pend[(v % 4 + 4 - 3) % 4 * 8 + j_2] = _fma_43;
                                                }
                                            }
                                            int trow_0 = tail_row_off[frag * 3 + 1];
                                            if (trow_0 >= 0) {
                                                long long rowr_1 = (long long)trow_0 * C64 + c64;
                                                {
                                                    unsigned _ldv8_44_0;
                                                    unsigned _ldv8_44_1;
                                                    unsigned _ldv8_44_2;
                                                    unsigned _ldv8_44_3;
                                                    unsigned _ldv8_44_4;
                                                    unsigned _ldv8_44_5;
                                                    unsigned _ldv8_44_6;
                                                    unsigned _ldv8_44_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_44_0), "=r"(_ldv8_44_1), "=r"(_ldv8_44_2), "=r"(_ldv8_44_3), "=r"(_ldv8_44_4), "=r"(_ldv8_44_5), "=r"(_ldv8_44_6), "=r"(_ldv8_44_7) : "l"((const void*)(dpre_tail + (rowr_1))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_44_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_44_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_44_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_44_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_44_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_44_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_44_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_44_7);
                                                }
                                                #pragma unroll
                                                for (int j_3 = 0; j_3 < 8; j_3++) {
                                                    #pragma unroll
                                                    for (int kk_8 = 0; kk_8 < 4; kk_8++) {
                                                        w4[kk_8] = wr[j_3 * 4 + kk_8];
                                                    }
                                                    float _fma_45;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(w4[1]), "f"(tailf[j_3]), "f"(pend[(v % 4 + 4 + 1 - 2) % 4 * 8 + j_3]));
                                                    pend[(v % 4 + 4 + 1 - 2) % 4 * 8 + j_3] = _fma_45;
                                                    float _fma_46;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(w4[0]), "f"(tailf[j_3]), "f"(pend[(v % 4 + 4 + 1 - 3) % 4 * 8 + j_3]));
                                                    pend[(v % 4 + 4 + 1 - 3) % 4 * 8 + j_3] = _fma_46;
                                                }
                                            }
                                            int trow_1 = tail_row_off[frag * 3 + 2];
                                            if (trow_1 >= 0) {
                                                long long rowr_2 = (long long)trow_1 * C64 + c64;
                                                {
                                                    unsigned _ldv8_47_0;
                                                    unsigned _ldv8_47_1;
                                                    unsigned _ldv8_47_2;
                                                    unsigned _ldv8_47_3;
                                                    unsigned _ldv8_47_4;
                                                    unsigned _ldv8_47_5;
                                                    unsigned _ldv8_47_6;
                                                    unsigned _ldv8_47_7;
                                                    asm volatile(
                                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                                        : "=r"(_ldv8_47_0), "=r"(_ldv8_47_1), "=r"(_ldv8_47_2), "=r"(_ldv8_47_3), "=r"(_ldv8_47_4), "=r"(_ldv8_47_5), "=r"(_ldv8_47_6), "=r"(_ldv8_47_7) : "l"((const void*)(dpre_tail + (rowr_2))) : "memory");
                                                    tailf[0 + 0] = __uint_as_float(_ldv8_47_0);
                                                    tailf[0 + 1] = __uint_as_float(_ldv8_47_1);
                                                    tailf[0 + 2] = __uint_as_float(_ldv8_47_2);
                                                    tailf[0 + 3] = __uint_as_float(_ldv8_47_3);
                                                    tailf[0 + 4] = __uint_as_float(_ldv8_47_4);
                                                    tailf[0 + 5] = __uint_as_float(_ldv8_47_5);
                                                    tailf[0 + 6] = __uint_as_float(_ldv8_47_6);
                                                    tailf[0 + 7] = __uint_as_float(_ldv8_47_7);
                                                }
                                                #pragma unroll
                                                for (int j_4 = 0; j_4 < 8; j_4++) {
                                                    #pragma unroll
                                                    for (int kk_9 = 0; kk_9 < 4; kk_9++) {
                                                        w4[kk_9] = wr[j_4 * 4 + kk_9];
                                                    }
                                                    float _fma_48;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(w4[0]), "f"(tailf[j_4]), "f"(pend[(v % 4 + 4 + 2 - 3) % 4 * 8 + j_4]));
                                                    pend[(v % 4 + 4 + 2 - 3) % 4 * 8 + j_4] = _fma_48;
                                                }
                                            }
                                            #pragma unroll
                                            for (int m = 1; m < 4; m++) {
                                                int tm = tcur - m;
                                                if (tm >= run_t0 && tm < t1 && tm >= doc_start) {
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
                                                        const uint4* _vptr_49 = reinterpret_cast<const uint4*>(hist + rowh_1_0);
                                                        uint4 _vld_49[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_49[_blk] = _vptr_49[_blk];
                                                            uint32_t* _vpairs_49 = reinterpret_cast<uint32_t*>(&_vld_49[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 1) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_49[_pair]) & 0xffff0000u);
                                                            }
                                                        }
                                                    }
                                                }
                                                int hrow_2_1 = hist_row_off[frag * 3 + 1];
                                                if (hrow_2_1 >= 0) {
                                                    long long rowh_1_1 = (long long)hrow_2_1 * C64 + c64;
                                                    {
                                                        const uint4* _vptr_50 = reinterpret_cast<const uint4*>(hist + rowh_1_1);
                                                        uint4 _vld_50[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_50[_blk] = _vptr_50[_blk];
                                                            uint32_t* _vpairs_50 = reinterpret_cast<uint32_t*>(&_vld_50[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 2) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_50[_pair]) & 0xffff0000u);
                                                            }
                                                        }
                                                    }
                                                }
                                                int hrow_2_2 = hist_row_off[frag * 3];
                                                if (hrow_2_2 >= 0) {
                                                    long long rowh_1_2 = (long long)hrow_2_2 * C64 + c64;
                                                    {
                                                        const uint4* _vptr_51 = reinterpret_cast<const uint4*>(hist + rowh_1_2);
                                                        uint4 _vld_51[1];
                                                        #pragma unroll
                                                        for (int _blk = 0; _blk < 1; _blk++) {
                                                            _vld_51[_blk] = _vptr_51[_blk];
                                                            uint32_t* _vpairs_51 = reinterpret_cast<uint32_t*>(&_vld_51[_blk]);
                                                            #pragma unroll
                                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                                (&win[(v % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) << 16);
                                                                (&win[(v % 4 + 4 - 3) % 4 * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_51[_pair]) & 0xffff0000u);
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
                                                for (int j_5 = 0; j_5 < 8; j_5++) {
                                                    #pragma unroll
                                                    for (int kk_10 = 0; kk_10 < 4; kk_10++) {
                                                        w4[kk_10] = wr[j_5 * 4 + kk_10];
                                                    }
                                                    float _fma_53;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(w4[1]), "f"(win[(v % 4 + 2) % 4 * 8 + j_5]), "f"(mul_f32_rn_noftz(w4[0], win[(v % 4 + 1) % 4 * 8 + j_5])));
                                                    float _fma_52;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(w4[2]), "f"(win[(v % 4 + 3) % 4 * 8 + j_5]), "f"(_fma_53));
                                                    float fma_2;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_2) : "f"(w4[3]), "f"(win[v % 4 * 8 + j_5]), "f"(_fma_52));
                                                    float e_2 = -fma_2 * 1.442695f;
                                                    float _exp2_54 = approx_exp2(e_2);
                                                    float den_2 = 1.0f + _exp2_54;
                                                    float sg_2 = approx_rcp(den_2);
                                                    float _fma_55;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(fma_2), "f"(1.0f - sg_2), "f"(1.0f));
                                                    dp[j_5] = dyf[j_5] * (sg_2 * _fma_55);
                                                    #pragma unroll
                                                    for (int kk_11 = 0; kk_11 < 4; kk_11++) {
                                                        acc[j_5 * 4 + kk_11] = __fmaf_rn(dp[j_5], win[(v % 4 + 1 + kk_11) % 4 * 8 + j_5], acc[j_5 * 4 + kk_11]);
                                                    }
                                                    float _fma_56;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(w4[0]), "f"(dp[j_5]), "f"(pend[(v % 4 + 1) % 4 * 8 + j_5]));
                                                    pend[(v % 4 + 1) % 4 * 8 + j_5] = _fma_56;
                                                    float _fma_57;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(w4[1]), "f"(dp[j_5]), "f"(pend[(v % 4 + 2) % 4 * 8 + j_5]));
                                                    pend[(v % 4 + 2) % 4 * 8 + j_5] = _fma_57;
                                                    float _fma_58;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(w4[2]), "f"(dp[j_5]), "f"(pend[(v % 4 + 3) % 4 * 8 + j_5]));
                                                    pend[(v % 4 + 3) % 4 * 8 + j_5] = _fma_58;
                                                    pend[v % 4 * 8 + j_5] = w4[3] * dp[j_5];
                                                }
                                            } else {
                                                #pragma unroll
                                                for (int j_6 = 0; j_6 < 8; j_6++) {
                                                    #pragma unroll
                                                    for (int kk_12 = 0; kk_12 < 4; kk_12++) {
                                                        w4[kk_12] = wr[j_6 * 4 + kk_12];
                                                    }
                                                    float _fma_60;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(w4[1]), "f"(win[(v % 4 + 2) % 4 * 8 + j_6]), "f"(mul_f32_rn_noftz(w4[0], win[(v % 4 + 1) % 4 * 8 + j_6])));
                                                    float _fma_59;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(w4[2]), "f"(win[(v % 4 + 3) % 4 * 8 + j_6]), "f"(_fma_60));
                                                    float fma_3;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(fma_3) : "f"(w4[3]), "f"(win[v % 4 * 8 + j_6]), "f"(_fma_59));
                                                    float e_3 = -fma_3 * 1.442695f;
                                                    float _exp2_61 = approx_exp2(e_3);
                                                    float den_3 = 1.0f + _exp2_61;
                                                    float sg_3 = approx_rcp(den_3);
                                                    float _fma_62;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(fma_3), "f"(1.0f - sg_3), "f"(1.0f));
                                                    dp[j_6] = dyf[j_6] * (sg_3 * _fma_62);
                                                    float _fma_63;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(w4[0]), "f"(dp[j_6]), "f"(pend[(v % 4 + 1) % 4 * 8 + j_6]));
                                                    pend[(v % 4 + 1) % 4 * 8 + j_6] = _fma_63;
                                                    float _fma_64;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_64) : "f"(w4[1]), "f"(dp[j_6]), "f"(pend[(v % 4 + 2) % 4 * 8 + j_6]));
                                                    pend[(v % 4 + 2) % 4 * 8 + j_6] = _fma_64;
                                                    float _fma_65;
                                                    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_65) : "f"(w4[2]), "f"(dp[j_6]), "f"(pend[(v % 4 + 3) % 4 * 8 + j_6]));
                                                    pend[(v % 4 + 3) % 4 * 8 + j_6] = _fma_65;
                                                    pend[v % 4 * 8 + j_6] = w4[3] * dp[j_6];
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
                        int _max_66 = ((t_fill - (run_t0 + (kq + 3 - k_run) * 8)) > (0) ? (t_fill - (run_t0 + (kq + 3 - k_run) * 8)) : (0));
                        int nrows_3 = ((_max_66) < (8) ? (_max_66) : (8));
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
                int _shfl_sync_67 = __shfl_sync(0xFFFFFFFF, dl[0], 2);
                fl = _shfl_sync_67;
                int _shfl_sync_68 = __shfl_sync(0xFFFFFFFF, dl[0], 0);
                gl = _shfl_sync_68;
                int _shfl_sync_69 = __shfl_sync(0xFFFFFFFF, dl[0], 1);
                un = _shfl_sync_69;
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
                        int _max_71 = ((S - lu_t + M - 1) > (0) ? (S - lu_t + M - 1) : (0));
                        int _min_70 = ((NL) < (_max_71 / M) ? (NL) : (_max_71 / M));
                        wgt = un + 2 + (NL - _min_70);
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
                    int _shfl_sync_72 = __shfl_sync(0xFFFFFFFF, l1, 0);
                    if (_shfl_sync_72 != 0) {
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
                                    unsigned _v4_73_0;
                                    unsigned _v4_73_1;
                                    unsigned _v4_73_2;
                                    unsigned _v4_73_3;
                                    asm volatile("ld.global.cg.v4.b32 {%0, %1, %2, %3}, [%4];"
                                        : "=r"(_v4_73_0), "=r"(_v4_73_1), "=r"(_v4_73_2), "=r"(_v4_73_3) : "l"((const void*)(dw_acc + (((long long)q5 * C64 + (long long)ch5) * 4ll))) : "memory");
                                    lw[(j5 * 4 + q5) * 4 + 0] = __uint_as_float(_v4_73_0);
                                    lw[(j5 * 4 + q5) * 4 + 1] = __uint_as_float(_v4_73_1);
                                    lw[(j5 * 4 + q5) * 4 + 2] = __uint_as_float(_v4_73_2);
                                    lw[(j5 * 4 + q5) * 4 + 3] = __uint_as_float(_v4_73_3);
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
            int ncan = 0;
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
                ncan = ncan + 1;
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
                for (int j_7 = 0; j_7 < 8; j_7++) {
                    int gg = lane + 32 * j_7;
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
            if (lane == 0) {
                unsigned int done;
                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                    : "=r"(done) : "l"(&claims[L]), "r"(static_cast<uint32_t>((unsigned int)(1 + ncan))) : "memory");
                if (done + (unsigned int)(1 + ncan) == (unsigned int)n_ctas) {
                    #pragma unroll 1
                    for (int zq = 0; zq < L + 1; zq++) {
                        claims[zq] = 0;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
