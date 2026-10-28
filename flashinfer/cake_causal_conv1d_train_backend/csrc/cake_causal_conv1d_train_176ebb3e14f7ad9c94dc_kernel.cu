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
#define SMEM_CELLS_OFF 33792
#define SMEM_CELLS_STAGE_BYTES 24
#define SMEM_CELLS_STRIDE 24
#define SMEM_TOTAL 33920
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


__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
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
kernel_cake_causal_conv1d_train_176ebb3e14f7ad9c94dc(__nv_bfloat16* __restrict__ x, void* __restrict__ w, int* __restrict__ cu, __nv_bfloat16* __restrict__ y, int* __restrict__ work, int S, int C, int num_docs, int search_steps, int ct_vec, int n_sub, int n_ch_tiles, int flags)
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
    int* cells = reinterpret_cast<int*>(smem_raw + 33792);
    const int cells_addr = smem + 33792;

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

    // === Task calls (declaration order) ===
    asm volatile("griddepcontrol.wait;" ::: "memory");
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    long long C64 = (long long)C;
    int ct = bid % n_ch_tiles;
    int g_ct = (num_bids - ct + n_ch_tiles - 1) / n_ch_tiles;
    int sub = tid / ct_vec;
    int v = tid - sub * ct_vec;
    int c0 = (ct * ct_vec + v) * 8;
    int lane_on = 0;
    if (sub < n_sub && c0 < C) {
        lane_on = 1;
    }
    float wr[32];
    float br[8];
    if (lane_on != 0) {
        if ((flags & 1) != 0) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                {
                    uint2 _vld_0;
                    _vld_0 = *reinterpret_cast<const uint2*>(reinterpret_cast<const __nv_bfloat16*>(w) + ((long long)c0 + (long long)j) * 4);
                    uint32_t* _vpairs_0 = reinterpret_cast<uint32_t*>(&_vld_0);
                    #pragma unroll
                    for (int _pair = 0; _pair < 2; _pair++) {
                        (&wr[j * 4 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) << 16);
                        (&wr[j * 4 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_0[_pair]) & 0xffff0000u);
                    }
                }
            }
        } else {
            #pragma unroll
            for (int j_1 = 0; j_1 < 8; j_1++) {
                {
                    float4 _v4 = *reinterpret_cast<const float4*>(reinterpret_cast<const float*>(w) + ((long long)c0 + (long long)j_1) * 4);
                    wr[j_1 * 4 + 0] = _v4.x;
                    wr[j_1 * 4 + 1] = _v4.y;
                    wr[j_1 * 4 + 2] = _v4.z;
                    wr[j_1 * 4 + 3] = _v4.w;
                }
            }
        }
    }
    long long c64 = (long long)c0;
    int chunk = n_sub * 16;
    int nch = (S + chunk - 1) / chunk;
    int row_bytes = ct_vec * 16;
    long long cbase64 = (long long)(ct * ct_vec * 8);
    int cbytes = ((row_bytes) < ((C - ct * ct_vec * 8) * 2) ? (row_bytes) : ((C - ct * ct_vec * 8) * 2));
    int last = 0;
    if (tid == 0) {
        int _atomic_fetch_add_2 = atomicAdd(&work[ct], 1);
        last = _atomic_fetch_add_2;
        cells[0] = last;
        cells[4] = last;
        cells[1] = last;
        cells[5] = last;
    }
    __syncthreads();
    #pragma unroll 1
    for (int k0 = 0; k0 < 2; k0++) {
        int qn = cells[4 + k0];
        if (qn < nch) {
            int stage_t0 = qn * n_sub * 16 + (int)((unsigned int)k0 % 2) * 8;
            if (tid == 0) {
                int rem = S - stage_t0;
                int _max_3 = ((rem + 8) > (0) ? (rem + 8) : (0));
                int n_full = (((int)((unsigned int)_max_3 / 16)) < (n_sub) ? ((int)((unsigned int)_max_3 / 16)) : (n_sub));
                int tail_rows = 0;
                if (n_full < n_sub) {
                    int _max_5 = ((rem - n_full * 16) > (0) ? (rem - n_full * 16) : (0));
                    int _min_4 = ((_max_5) < (8) ? (_max_5) : (8));
                    tail_rows = _min_4;
                }
                mbarrier_arrive_expect_tx(stage_full_addr + ((int)((unsigned int)k0 % 2)) * 8, (n_full * 8 + tail_rows) * cbytes);
            }
            if (tid - sub * ct_vec == 0 && sub < n_sub) {
                int ts_n = stage_t0 + sub * 16;
                int _max_6 = ((S - ts_n) > (0) ? (S - ts_n) : (0));
                int rows_n = ((_max_6) < (8) ? (_max_6) : (8));
                #pragma unroll 1
                for (int i = 0; i < rows_n; i++) {
                    long long src = (long long)(ts_n + i) * C64 + cbase64;
                    int saddr = (int)stage_buf_addr + (((int)((unsigned int)k0 % 2) * 8 + i) * 128 + sub * ct_vec) * 16;
                    cp_async_bulk_gmem2smem(saddr, reinterpret_cast<const uint8_t*>(x + src), cbytes, stage_full_addr + ((int)((unsigned int)k0 % 2)) * 8);
                }
            }
        }
    }
    float win[32];
    #pragma unroll
    for (int i_1 = 0; i_1 < 32; i_1++) {
        win[i_1] = 0.0f;
    }
    int doc_start = 0;
    int doc_end = 0;
    unsigned int pk[4];
    float out[8];
    #pragma unroll 1
    for (int k = 0; k < 2 * nch; k++) {
        int q = cells[(int)((unsigned int)k % 4)];
        if (q >= nch) {
            break;
        }
        int kn = k + 2;
        int grab = 0;
        if (tid == 0 && (unsigned int)kn % 2 == 0) {
            grab = 1;
        }
        int pend = 0;
        if (grab != 0) {
            int _atomic_fetch_add_7 = atomicAdd(&work[ct], 1);
            pend = _atomic_fetch_add_7;
        }
        int kst = (int)((unsigned int)k % 2);
        int ts = q * chunk + sub * 16 + kst * 8;
        int rows_k = 0;
        if (lane_on != 0) {
            int _max_9 = ((S - ts) > (0) ? (S - ts) : (0));
            int _min_8 = ((_max_9) < (8) ? (_max_9) : (8));
            rows_k = _min_8;
        }
        if (kst == 0 && rows_k > 0) {
            if (ts >= doc_end) {
                int lo = 0;
                int hi = num_docs - 1;
                #pragma unroll 1
                for (int step = 0; step < search_steps; step++) {
                    int mid = lo + hi + 1 >> 1;
                    if (ts >= cu[mid]) {
                        lo = mid;
                    } else {
                        hi = mid - 1;
                    }
                }
                doc_start = cu[lo];
                doc_end = cu[lo + 1];
            }
            #pragma unroll
            for (int kh = 0; kh < 3; kh++) {
                int tk = ts - 3 + kh;
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    win[(kh + 1) * 8 + j_2] = 0.0f;
                }
                if (tk >= doc_start) {
                    long long rowk = (long long)tk * C64 + c64;
                    {
                        const uint4* _vptr_10 = reinterpret_cast<const uint4*>(x + rowk);
                        uint4 _vld_10[1];
                        #pragma unroll
                        for (int _blk = 0; _blk < 1; _blk++) {
                            _vld_10[_blk] = _vptr_10[_blk];
                            uint32_t* _vpairs_10 = reinterpret_cast<uint32_t*>(&_vld_10[_blk]);
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) << 16);
                                (&win[(kh + 1) * 8 + _blk * 8 + _pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(_vpairs_10[_pair]) & 0xffff0000u);
                            }
                        }
                    }
                }
            }
        }
        int _stage_stage_full = (int)((unsigned int)k % 2);
        int _parity_stage_full = (int)((unsigned int)k / 2 % 2);
        mbarrier_wait(stage_full_addr + (_stage_stage_full) * 8, _parity_stage_full);
        if (rows_k > 0) {
            int sbase = (int)stage_buf_addr + ((int)((unsigned int)k % 2) * 8 * 128 + tid) * 16;
            int interior = 0;
            if (rows_k == 8 && ts + 8 <= doc_end) {
                interior = 1;
            }
            if (interior != 0) {
                #pragma unroll
                for (int u = 0; u < 8; u++) {
                    __int128_t _smem_raw_11;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_11) : "r"(sbase + u * 2048));
                    pk[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[0];
                    pk[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[1];
                    pk[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[2];
                    pk[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_11)[3];
                    #pragma unroll
                    for (int kk2 = 0; kk2 < 4; kk2++) {
                        win[u % 4 * 8 + 2 * kk2] = __uint_as_float(pk[kk2] << 16);
                        win[u % 4 * 8 + 2 * kk2 + 1] = __uint_as_float(pk[kk2] & 4294901760u);
                    }
                    #pragma unroll
                    for (int h = 0; h < 4; h++) {
                        float2 p2 = fma_f32x2_rn_noftz(make_float2(wr[8 * h + 3], wr[8 * h + 4 + 3]), make_float2(win[u % 4 * 8 + 2 * h], win[u % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 2], wr[8 * h + 4 + 2]), make_float2(win[(u % 4 + 3) % 4 * 8 + 2 * h], win[(u % 4 + 3) % 4 * 8 + 2 * h + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h + 1], wr[8 * h + 4 + 1]), make_float2(win[(u % 4 + 2) % 4 * 8 + 2 * h], win[(u % 4 + 2) % 4 * 8 + 2 * h + 1]), mul_f32x2_noftz(make_float2(wr[8 * h + 0], wr[8 * h + 4 + 0]), make_float2(win[(u % 4 + 1) % 4 * 8 + 2 * h], win[(u % 4 + 1) % 4 * 8 + 2 * h + 1])))));
                        float2 _f32x2_mul_b_12 = make_float2(-1.442695f, -1.442695f);
                        float2 e2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_12));
                        float _exp2_13 = approx_exp2(e2.x);
                        float _exp2_14 = approx_exp2(e2.y);
                        float2 den2 = add_f32x2_noftz(make_float2(_exp2_13, _exp2_14), make_float2(1.0f, 1.0f));
                        float _rcp_15 = approx_rcp(den2.x);
                        float _rcp_16 = approx_rcp(den2.y);
                        float2 _f32x2_mul_b_17 = make_float2(_rcp_15, _rcp_16);
                        float2 o2;
                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2) : "l"(*(const unsigned long long*)&p2), "l"(*(const unsigned long long*)&_f32x2_mul_b_17));
                        out[2 * h] = o2.x;
                        out[2 * h + 1] = o2.y;
                    }
                    long long row_f = (long long)(ts + u) * C64 + c64;
                    unsigned int _packed_f[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                        _packed_f[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                        _packed_f[_lp + 1] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                        _packed_f[_lp + 2] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 1; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                        _packed_f[_lp + 3] = *(uint32_t*)&_bf2;
                    }
                    reinterpret_cast<int4*>(y + (row_f))[0] = reinterpret_cast<int4*>(_packed_f)[0];
                }
            } else {
                #pragma unroll
                for (int u_1 = 0; u_1 < 8; u_1++) {
                    if (rows_k > u_1) {
                        int tcur = ts + u_1;
                        __int128_t _smem_raw_18;
                        asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_raw_18) : "r"(sbase + u_1 * 2048));
                        pk[0] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[0];
                        pk[1] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[1];
                        pk[2] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[2];
                        pk[3] = reinterpret_cast<const uint32_t*>(&_smem_raw_18)[3];
                        #pragma unroll
                        for (int kk2_1 = 0; kk2_1 < 4; kk2_1++) {
                            win[u_1 % 4 * 8 + 2 * kk2_1] = __uint_as_float(pk[kk2_1] << 16);
                            win[u_1 % 4 * 8 + 2 * kk2_1 + 1] = __uint_as_float(pk[kk2_1] & 4294901760u);
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
                                for (int j_3 = 0; j_3 < 8; j_3++) {
                                    win[(u_1 % 4 + 4 - m) % 4 * 8 + j_3] = 0.0f;
                                }
                            }
                        }
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 4; h_1++) {
                            float2 p2_1 = fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 3], wr[8 * h_1 + 4 + 3]), make_float2(win[u_1 % 4 * 8 + 2 * h_1], win[u_1 % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 2], wr[8 * h_1 + 4 + 2]), make_float2(win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 3) % 4 * 8 + 2 * h_1 + 1]), fma_f32x2_rn_noftz(make_float2(wr[8 * h_1 + 1], wr[8 * h_1 + 4 + 1]), make_float2(win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 2) % 4 * 8 + 2 * h_1 + 1]), mul_f32x2_noftz(make_float2(wr[8 * h_1 + 0], wr[8 * h_1 + 4 + 0]), make_float2(win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1], win[(u_1 % 4 + 1) % 4 * 8 + 2 * h_1 + 1])))));
                            float2 _f32x2_mul_b_19 = make_float2(-1.442695f, -1.442695f);
                            float2 e2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&e2_1) : "l"(*(const unsigned long long*)&p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_19));
                            float _exp2_20 = approx_exp2(e2_1.x);
                            float _exp2_21 = approx_exp2(e2_1.y);
                            float2 den2_1 = add_f32x2_noftz(make_float2(_exp2_20, _exp2_21), make_float2(1.0f, 1.0f));
                            float _rcp_22 = approx_rcp(den2_1.x);
                            float _rcp_23 = approx_rcp(den2_1.y);
                            float2 _f32x2_mul_b_24 = make_float2(_rcp_22, _rcp_23);
                            float2 o2_1;
                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&o2_1) : "l"(*(const unsigned long long*)&p2_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_24));
                            out[2 * h_1] = o2_1.x;
                            out[2 * h_1 + 1] = o2_1.y;
                        }
                        long long row = (long long)tcur * C64 + c64;
                        unsigned int _packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 0], out[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 2], out[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 4], out[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out[_lp*2 + 6], out[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(y + (row))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                    }
                }
            }
        }
        if (tid == 0) {
            if (grab != 0) {
                last = pend;
            }
            cells[(int)((unsigned int)kn % 4)] = last;
        }
        __syncthreads();
        int qn_1 = cells[(int)((unsigned int)kn % 4)];
        if (qn_1 < nch) {
            int stage_t0_1 = qn_1 * n_sub * 16 + (int)((unsigned int)(k + 2) % 2) * 8;
            if (tid == 0) {
                int rem_1 = S - stage_t0_1;
                int _max_25 = ((rem_1 + 8) > (0) ? (rem_1 + 8) : (0));
                int n_full_1 = (((int)((unsigned int)_max_25 / 16)) < (n_sub) ? ((int)((unsigned int)_max_25 / 16)) : (n_sub));
                int tail_rows_1 = 0;
                if (n_full_1 < n_sub) {
                    int _max_27 = ((rem_1 - n_full_1 * 16) > (0) ? (rem_1 - n_full_1 * 16) : (0));
                    int _min_26 = ((_max_27) < (8) ? (_max_27) : (8));
                    tail_rows_1 = _min_26;
                }
                mbarrier_arrive_expect_tx(stage_full_addr + ((int)((unsigned int)(k + 2) % 2)) * 8, (n_full_1 * 8 + tail_rows_1) * cbytes);
            }
            if (tid - sub * ct_vec == 0 && sub < n_sub) {
                int ts_n_1 = stage_t0_1 + sub * 16;
                int _max_28 = ((S - ts_n_1) > (0) ? (S - ts_n_1) : (0));
                int rows_n_1 = ((_max_28) < (8) ? (_max_28) : (8));
                #pragma unroll 1
                for (int i_2 = 0; i_2 < rows_n_1; i_2++) {
                    long long src_1 = (long long)(ts_n_1 + i_2) * C64 + cbase64;
                    int saddr_1 = (int)stage_buf_addr + (((int)((unsigned int)(k + 2) % 2) * 8 + i_2) * 128 + sub * ct_vec) * 16;
                    cp_async_bulk_gmem2smem(saddr_1, reinterpret_cast<const uint8_t*>(x + src_1), cbytes, stage_full_addr + ((int)((unsigned int)(k + 2) % 2)) * 8);
                }
            }
        }
    }
    if (tid == 0) {
        uint32_t done;
        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
            : "=r"(done) : "l"(&reinterpret_cast<unsigned int*>(work)[n_ch_tiles + ct]), "r"(static_cast<uint32_t>((unsigned int)(g_ct - 1))) : "memory");
        if ((int)done == g_ct - 1) {
            *(reinterpret_cast<int*>(work) + (ct)) = 0;
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
