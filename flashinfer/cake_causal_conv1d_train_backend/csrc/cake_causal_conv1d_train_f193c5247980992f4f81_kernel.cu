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
#define SMEM_STAGE_BUF_OFF 1024
#define SMEM_STAGE_BUF_STAGE_BYTES 32768
#define SMEM_STAGE_BUF_STRIDE 32768
#define SMEM_TOTAL 33792
#define THREADS 128
#define LAUNCH_MIN_BLOCKS 4

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



__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_causal_conv1d_train_f193c5247980992f4f81(__nv_bfloat16* __restrict__ x, float* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ y, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles)
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

    // Kernel setup ops
    uint8_t* stage_buf = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int stage_buf_addr = smem + 1024;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stage_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int rank = bid / n_ch_tiles;
    int g_ct = (num_bids - ct + n_ch_tiles - 1) / n_ch_tiles;
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 8;
    int streams = g_ct * n_sub;
    int _max_0 = (((S + streams - 1) / streams) > (1) ? ((S + streams - 1) / streams) : (1));
    int L = _max_0;
    int t0 = (rank * n_sub + sub) * L;
    long long c64 = (long long)c0;
    int streams_0 = g_ct * n_sub;
    int _max_1 = (((S + streams_0 - 1) / streams_0) > (1) ? ((S + streams_0 - 1) / streams_0) : (1));
    int L_1 = _max_1;
    int base = rank * n_sub * L_1;
    int t0_2 = base + sub * L_1;
    int _min_0 = ((t0_2 + L_1) < (S) ? (t0_2 + L_1) : (S));
    int t_end = _min_0;
    int active = 0;
    if (sub < n_sub && c0 < C && t0_2 < S) {
        active = 1;
    }
    int my_rows = 0;
    if (active != 0) {
        my_rows = t_end - t0_2;
    }
    int _max_2 = ((S - base) > (0) ? (S - base) : (0));
    int _min_1 = ((_max_2) < (L_1) ? (_max_2) : (L_1));
    int rows0 = _min_1;
    int n_stages = (rows0 + 8 - 1) / 8;
    int _max_3 = (((S - base) / L_1) > (0) ? ((S - base) / L_1) : (0));
    int _min_2 = ((_max_3) < (n_sub) ? (_max_3) : (n_sub));
    int n_full = _min_2;
    int straddle = 0;
    if (n_full < n_sub) {
        int _max_4 = ((S - base - n_full * L_1) > (0) ? (S - base - n_full * L_1) : (0));
        int _min_3 = ((_max_4) < (L_1) ? (_max_4) : (L_1));
        straddle = _min_3;
    }
    int row_bytes = ct_vec * 16;
    long long cbase64 = (long long)(ct * ct_vec * 8);
    int _min_4 = ((row_bytes) < ((C - ct * ct_vec * 8) * 2) ? (row_bytes) : ((C - ct * ct_vec * 8) * 2));
    int cbytes = _min_4;
    int _min_5 = ((n_stages) < (2) ? (n_stages) : (2));
    #pragma unroll 1
    for (int k0 = 0; k0 < _min_5; k0++) {
        int slot0 = k0;
        if (tid == 0) {
            int _max_5 = ((L_1 - k0 * 8) > (0) ? (L_1 - k0 * 8) : (0));
            int _min_6 = ((_max_5) < (8) ? (_max_5) : (8));
            int full_rows = _min_6;
            int _max_6 = ((straddle - k0 * 8) > (0) ? (straddle - k0 * 8) : (0));
            int _min_7 = ((_max_6) < (8) ? (_max_6) : (8));
            int tail_rows = _min_7;
            mbarrier_arrive_expect_tx(stage_full_addr + (slot0) * 8, (n_full * full_rows + tail_rows) * cbytes);
        }
        if (tid - sub * ct_vec == 0) {
            int _max_7 = ((my_rows - k0 * 8) > (0) ? (my_rows - k0 * 8) : (0));
            int _min_8 = ((_max_7) < (8) ? (_max_7) : (8));
            int rows_k = _min_8;
            #pragma unroll 1
            for (int i = 0; i < rows_k; i++) {
                int tn = t0_2 + k0 * 8 + i;
                long long src = (long long)tn * C64 + cbase64;
                int saddr = (int)stage_buf_addr + ((slot0 * 8 + i) * 128 + sub * ct_vec) * 16;
                cp_async_bulk_gmem2smem(saddr, reinterpret_cast<const uint8_t*>(x + src), cbytes, stage_full_addr + (slot0) * 8);
            }
        }
    }
    float wr[32];
    float w4[4];
    int wsm_addr = 0;
    float win[32];
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; i_1++) {
        win[i_1] = 0.0f;
    }
    int lo = 0;
    int hi = num_docs - 1;
    int doc_start = 0;
    int doc_end = 0;
    if (active != 0) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            {
                float4 _v4 = *reinterpret_cast<const float4*>(w + (c64 + (long long)j) * 4);
                wr[j * 4 + 0] = _v4.x;
                wr[j * 4 + 1] = _v4.y;
                wr[j * 4 + 2] = _v4.z;
                wr[j * 4 + 3] = _v4.w;
            }
        }
        #pragma unroll 1
        for (int step = 0; step < search_steps; step++) {
            int mid = lo + hi + 1 >> 1;
            if (t0_2 >= cu[mid]) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        doc_start = cu[lo];
        doc_end = cu[lo + 1];
        #pragma unroll
        for (int kh = 0; kh < 3; kh++) {
            int tk = t0_2 - 3 + kh;
            if (tk >= doc_start) {
                long long rowk = (long long)tk * C64 + c64;
                {
                    const uint4* _vptr_1 = reinterpret_cast<const uint4*>(x + rowk);
                    uint4 _vld_1[1];
                    #pragma unroll
                    for (int _blk = 0; _blk < 1; _blk++) {
                        _vld_1[_blk] = _vptr_1[_blk];
                        uint32_t* _vpairs_1 = reinterpret_cast<uint32_t*>(&_vld_1[_blk]);
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) << 16);
                            (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_1[_pair]) & 0xffff0000u);
                        }
                    }
                }
            }
        }
    }
    unsigned int pk[4];
    float out[8];
    #pragma unroll 1
    for (int k = 0; k < n_stages; k++) {
        int slot = k % 2;
        int parity = k / 2 & 1;
        mbarrier_wait(stage_full_addr + (slot) * 8, parity);
        if (active != 0) {
            int tb = t0_2 + k * 8;
            int _max_8 = ((my_rows - k * 8) > (0) ? (my_rows - k * 8) : (0));
            int _min_9 = ((_max_8) < (8) ? (_max_8) : (8));
            int rows_k_1 = _min_9;
            int sbase = (int)stage_buf_addr + (slot * 8 * 128 + tid) * 16;
            int interior = 0;
            if (interior != 0) {
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    __int128_t _smem_b128_2;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_2) : "r"(sbase + u * 2048));
                    *reinterpret_cast<uint32_t*>(&pk[0]) = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[0];
                    *reinterpret_cast<uint32_t*>(&pk[(0) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[1];
                    *reinterpret_cast<uint32_t*>(&pk[(0) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[2];
                    *reinterpret_cast<uint32_t*>(&pk[(0) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_b128_2)[3];
                    #pragma unroll
                    for (int kk = 0; kk < 4; kk++) {
                        win[u % 4 * 8 + 2 * kk] = __uint_as_float(pk[kk] << 16);
                        win[u % 4 * 8 + 2 * kk + 1] = __uint_as_float(pk[kk] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 8; j_1++) {
                        #pragma unroll
                        for (int kk_1 = 0; kk_1 < 4; kk_1++) {
                            w4[kk_1] = wr[j_1 * 4 + kk_1];
                        }
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(w4[0]), "f"(win[(u % 4 + 1) % 4 * 8 + j_1]));
                        float _fma_0;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(w4[1]), "f"(win[(u % 4 + 2) % 4 * 8 + j_1]), "f"(_mul_0));
                        float _fma_1;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(w4[2]), "f"(win[(u % 4 + 3) % 4 * 8 + j_1]), "f"(_fma_0));
                        float _fma_2;
                        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(w4[3]), "f"(win[u % 4 * 8 + j_1]), "f"(_fma_1));
                        float e = (-_fma_2) * 1.4426950408889634f;
                        float _exp2_0 = approx_exp2(e);
                        float den = 1.0f + _exp2_0;
                        float _rcp_0 = approx_rcp(den);
                        out[j_1] = _fma_2 * _rcp_0;
                    }
                    long long row = (long long)(tb + u) * C64 + c64;
                    {
                        __nv_bfloat162 _pk[4];
                        _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                        _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                        _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                        _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y))[row + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            } else {
                #pragma unroll
                for (int u_1 = 0; u_1 < 8; u_1++) {
                    if (rows_k_1 > u_1) {
                        int tcur = tb + u_1;
                        __int128_t _smem_b128_3;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_3) : "r"(sbase + u_1 * 2048));
                        *reinterpret_cast<uint32_t*>(&pk[0]) = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[0];
                        *reinterpret_cast<uint32_t*>(&pk[(0) + 1]) = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[1];
                        *reinterpret_cast<uint32_t*>(&pk[(0) + 2]) = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[2];
                        *reinterpret_cast<uint32_t*>(&pk[(0) + 3]) = reinterpret_cast<const uint32_t*>(&_smem_b128_3)[3];
                        #pragma unroll
                        for (int kk_2 = 0; kk_2 < 4; kk_2++) {
                            win[u_1 % 4 * 8 + 2 * kk_2] = __uint_as_float(pk[kk_2] << 16);
                            win[u_1 % 4 * 8 + 2 * kk_2 + 1] = __uint_as_float(pk[kk_2] & 4294901760u);
                        }
                        if (tcur >= doc_end) {
                            int lo2 = 0;
                            int hi2 = num_docs - 1;
                            #pragma unroll 1
                            for (int step2 = 0; step2 < search_steps; step2++) {
                                int mid2 = lo2 + hi2 + 1 >> 1;
                                if (tcur >= cu[mid2]) {
                                    lo2 = mid2;
                                } else {
                                    hi2 = mid2 - 1;
                                }
                            }
                            doc_start = cu[lo2];
                            doc_end = cu[lo2 + 1];
                            #pragma unroll
                            for (int m = 1; m < 4; m++) {
                                #pragma unroll
                                for (int j_2 = 0; j_2 < 8; j_2++) {
                                    win[(u_1 % 4 + 4 - m) % 4 * 8 + j_2] = 0.0f;
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            #pragma unroll
                            for (int kk_3 = 0; kk_3 < 4; kk_3++) {
                                w4[kk_3] = wr[j_3 * 4 + kk_3];
                            }
                            float _mul_1;
                            asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(w4[0]), "f"(win[(u_1 % 4 + 1) % 4 * 8 + j_3]));
                            float _fma_3;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(w4[1]), "f"(win[(u_1 % 4 + 2) % 4 * 8 + j_3]), "f"(_mul_1));
                            float _fma_4;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(w4[2]), "f"(win[(u_1 % 4 + 3) % 4 * 8 + j_3]), "f"(_fma_3));
                            float _fma_5;
                            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(w4[3]), "f"(win[u_1 % 4 * 8 + j_3]), "f"(_fma_4));
                            float e_1 = (-_fma_5) * 1.4426950408889634f;
                            float _exp2_1 = approx_exp2(e_1);
                            float den_1 = 1.0f + _exp2_1;
                            float _rcp_1 = approx_rcp(den_1);
                            out[j_3] = _fma_5 * _rcp_1;
                        }
                        long long row_1 = (long long)tcur * C64 + c64;
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(out[0 + 0], out[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(out[0 + 2], out[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(out[0 + 4], out[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(out[0 + 6], out[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(y))[row_1 + 0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
        }
        __syncthreads();
        int kn = k + 2;
        if (kn < n_stages) {
            if (tid == 0) {
                int _max_9 = ((L_1 - kn * 8) > (0) ? (L_1 - kn * 8) : (0));
                int _min_10 = ((_max_9) < (8) ? (_max_9) : (8));
                int full_rows_1 = _min_10;
                int _max_10 = ((straddle - kn * 8) > (0) ? (straddle - kn * 8) : (0));
                int _min_11 = ((_max_10) < (8) ? (_max_10) : (8));
                int tail_rows_1 = _min_11;
                mbarrier_arrive_expect_tx(stage_full_addr + (slot) * 8, (n_full * full_rows_1 + tail_rows_1) * cbytes);
            }
            if (tid - sub * ct_vec == 0) {
                int _max_11 = ((my_rows - kn * 8) > (0) ? (my_rows - kn * 8) : (0));
                int _min_12 = ((_max_11) < (8) ? (_max_11) : (8));
                int rows_k_2 = _min_12;
                #pragma unroll 1
                for (int i_2 = 0; i_2 < rows_k_2; i_2++) {
                    int tn_1 = t0_2 + kn * 8 + i_2;
                    long long src_1 = (long long)tn_1 * C64 + cbase64;
                    int saddr_1 = (int)stage_buf_addr + ((slot * 8 + i_2) * 128 + sub * ct_vec) * 16;
                    cp_async_bulk_gmem2smem(saddr_1, reinterpret_cast<const uint8_t*>(x + src_1), cbytes, stage_full_addr + (slot) * 8);
                }
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"

