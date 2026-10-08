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

#define CAKE_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_KV8_OFF 1024
#define SMEM_KV8_STAGE_BYTES 32768
#define SMEM_KV8_STRIDE 32768
#define SMEM_IDENT8_OFF 66560
#define SMEM_IDENT8_STAGE_BYTES 8192
#define SMEM_IDENT8_STRIDE 8192
#define SMEM_Q16_OFF 74752
#define SMEM_Q16_STAGE_BYTES 8192
#define SMEM_Q16_STRIDE 8192
#define SMEM_P16_OFF 82944
#define SMEM_P16_STAGE_BYTES 4096
#define SMEM_P16_STRIDE 4096
#define SMEM_MARK_OFF 82944
#define SMEM_MARK_STAGE_BYTES 4096
#define SMEM_MARK_STRIDE 4096
#define SMEM_RED_OFF 91136
#define SMEM_RED_STAGE_BYTES 512
#define SMEM_RED_STRIDE 512
#define SMEM_LRED_OFF 91648
#define SMEM_LRED_STAGE_BYTES 512
#define SMEM_LRED_STRIDE 512
#define SMEM_PREF_OFF 92160
#define SMEM_PREF_STAGE_BYTES 512
#define SMEM_PREF_STRIDE 512
#define SMEM_CNT_OFF 92672
#define SMEM_CNT_STAGE_BYTES 32
#define SMEM_CNT_STRIDE 32
#define SMEM_HDR_OFF 92704
#define SMEM_HDR_STAGE_BYTES 32
#define SMEM_HDR_STRIDE 32
#define SMEM_META_PG_OFF 92736
#define SMEM_META_PG_STAGE_BYTES 256
#define SMEM_META_PG_STRIDE 256
#define SMEM_META_BLK_OFF 92992
#define SMEM_META_BLK_STAGE_BYTES 256
#define SMEM_META_BLK_STRIDE 256
#define SMEM_META_MASK_OFF 93248
#define SMEM_META_MASK_STAGE_BYTES 256
#define SMEM_META_MASK_STRIDE 256
#define SMEM_TOTAL 93568
#define THREADS 128
#define TRACE 0
#define LAUNCH_MIN_BLOCKS 2

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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_8a72afd33b6d6c36048b(unsigned int* __restrict__ Q32, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ q2k_indices, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int total_q, int batch, int num_q_heads, int num_kv_heads, int max_pages, int use_q_offset, float softmax_scale_log2, unsigned int zero_u32, unsigned long long* __restrict__ trace)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define kv_full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* kv8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int kv8_addr = smem + 1024;
    uint8_t* ident8 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int ident8_addr = smem + 66560;
    __half* q16 = reinterpret_cast<__half*>(smem_raw + 74752);
    const int q16_addr = smem + 74752;
    __half* p16 = reinterpret_cast<__half*>(smem_raw + 82944);
    const int p16_addr = smem + 82944;
    uint8_t* mark = reinterpret_cast<uint8_t*>(smem_raw + 82944);
    const int mark_addr = smem + 82944;
    float* red = reinterpret_cast<float*>(smem_raw + 91136);
    const int red_addr = smem + 91136;
    float* lred = reinterpret_cast<float*>(smem_raw + 91648);
    const int lred_addr = smem + 91648;
    int* pref = reinterpret_cast<int*>(smem_raw + 92160);
    const int pref_addr = smem + 92160;
    int* cnt = reinterpret_cast<int*>(smem_raw + 92672);
    const int cnt_addr = smem + 92672;
    int* hdr = reinterpret_cast<int*>(smem_raw + 92704);
    const int hdr_addr = smem + 92704;
    int* meta_pg = reinterpret_cast<int*>(smem_raw + 92736);
    const int meta_pg_addr = smem + 92736;
    int* meta_blk = reinterpret_cast<int*>(smem_raw + 92992);
    const int meta_blk_addr = smem + 92992;
    unsigned int* meta_mask = reinterpret_cast<unsigned int*>(smem_raw + 93248);
    const int meta_mask_addr = smem + 93248;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // === Task calls (dependency order) ===
    unsigned long long stamps[12];
    stamps[0] = 0;
    stamps[1] = 0;
    stamps[2] = 0;
    stamps[3] = 0;
    stamps[4] = 0;
    stamps[5] = 0;
    stamps[6] = 0;
    stamps[7] = 0;
    stamps[8] = 0;
    stamps[9] = 0;
    stamps[10] = 0;
    stamps[11] = 0;
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
    int gi = blockIdx.x;
    int kv_head = blockIdx.y;
    int tid_1 = threadIdx.x;
    int lane_0 = lane;
    int warp_1 = warp;
    int g = lane_0 / 4;
    int cq = lane_0 - g * 4;
    int key0 = warp_1 * 16 + g;
    int key1 = key0 + 8;
    if (warp_1 == 0) {
        int found = 0;
        int base_g = 0;
        int sel_b = 0;
        int sel_tok0 = 0;
        int sel_ntok = 0;
        int sel_cu = 0;
        int sel_qlen = 0;
        int nchunks = (batch + 31) / 32;
        #pragma unroll 1
        for (int c = 0; c < nchunks; c++) {
            if (found == 0) {
                int bb = c * 32 + lane_0;
                int cnt_b = 0;
                int cu_lo = 0;
                int cu_hi = 0;
                if (bb < batch) {
                    cu_lo = cu_seqlens_q[bb];
                    cu_hi = cu_seqlens_q[bb + 1];
                    cnt_b = (cu_hi - cu_lo + 4 - 1) / 4;
                }
                int incl = cnt_b;
                int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, incl, 1, 32);
                int up = _shfl_up_0;
                if (lane_0 >= 1) {
                    incl = incl + up;
                }
                int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, incl, 2, 32);
                int up_0 = _shfl_up_1;
                if (lane_0 >= 2) {
                    incl = incl + up_0;
                }
                int _shfl_up_2 = __shfl_up_sync(0xFFFFFFFF, incl, 4, 32);
                int up_1 = _shfl_up_2;
                if (lane_0 >= 4) {
                    incl = incl + up_1;
                }
                int _shfl_up_3 = __shfl_up_sync(0xFFFFFFFF, incl, 8, 32);
                int up_2 = _shfl_up_3;
                if (lane_0 >= 8) {
                    incl = incl + up_2;
                }
                int _shfl_up_4 = __shfl_up_sync(0xFFFFFFFF, incl, 16, 32);
                int up_3 = _shfl_up_4;
                if (lane_0 >= 16) {
                    incl = incl + up_3;
                }
                int excl = incl - cnt_b;
                int hit = 0;
                if (bb < batch) {
                    if (gi >= base_g + excl) {
                        if (gi < base_g + incl) {
                            hit = 1;
                        }
                    }
                }
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit != 0);
                unsigned int bal = _vote_0;
                if (bal != 0) {
                    int _ffs_0 = __ffs(bal);
                    int src_l = _ffs_0 - 1;
                    sel_b = c * 32 + src_l;
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, cu_lo, src_l);
                    sel_cu = _shfl_0;
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, cu_hi, src_l);
                    int hi_s = _shfl_1;
                    int _shfl_2 = __shfl_sync(0xFFFFFFFF, excl, src_l);
                    int ex_s = _shfl_2;
                    int _shfl_3 = __shfl_sync(0xFFFFFFFF, cnt_b, src_l);
                    int cnt_s = _shfl_3;
                    int lg = gi - base_g - ex_s;
                    lg = cnt_s - 1 - lg;
                    sel_tok0 = lg * 4;
                    sel_qlen = hi_s - sel_cu;
                    sel_ntok = sel_qlen - sel_tok0;
                    if (sel_ntok > 4) {
                        sel_ntok = 4;
                    }
                    found = 1;
                }
                int _shfl_4 = __shfl_sync(0xFFFFFFFF, incl, 31);
                base_g = base_g + _shfl_4;
            }
        }
        if (lane_0 == 0) {
            int qoff = 0;
            if (found != 0) {
                if (use_q_offset != 0) {
                    qoff = q_offset[sel_b];
                }
                if (use_q_offset == 0) {
                    qoff = seqused_k[sel_b] - sel_qlen;
                }
            }
            hdr[0] = sel_b;
            hdr[1] = sel_tok0;
            hdr[2] = sel_ntok;
            hdr[3] = sel_cu;
            hdr[4] = qoff + sel_tok0;
        }
    }
    if (warp_1 != 0) {
        int sid = tid_1 - 32;
        int ngrp_z = (max_pages + 31) / 32;
        unsigned int zz[4];
        zz[0] = 0;
        zz[1] = 0;
        zz[2] = 0;
        zz[3] = 0;
        int grp_z = sid;
        if (grp_z < ngrp_z) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z * 32)), "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z * 32 + 16)), "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3]) : "memory");
        }
        int grp_z_0 = sid + 96;
        if (grp_z_0 < ngrp_z) {
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z_0 * 32)), "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z_0 * 32 + 16)), "r"(zz[0]), "r"(zz[1]), "r"(zz[2]), "r"(zz[3]) : "memory");
        }
        int ich_lin = sid;
        if (ich_lin < 512) {
            int irow = ich_lin / 8;
            int ich = ich_lin - irow * 8;
            int iword = irow / 4;
            unsigned int ione = 56 << 8 * (irow & 3);
            unsigned int wv[4];
            wv[0] = 0;
            if (ich * 4 == iword) {
                wv[0] = ione;
            }
            wv[1] = 0;
            if (ich * 4 + 1 == iword) {
                wv[1] = ione;
            }
            wv[2] = 0;
            if (ich * 4 + 2 == iword) {
                wv[2] = ione;
            }
            wv[3] = 0;
            if (ich * 4 + 3 == iword) {
                wv[3] = ione;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow * 128 + (ich * 16 ^ (irow & 7) << 4))), "r"(wv[0]), "r"(wv[1]), "r"(wv[2]), "r"(wv[3]) : "memory");
        }
        int ich_lin_1 = sid + 96;
        if (ich_lin_1 < 512) {
            int irow_1 = ich_lin_1 / 8;
            int ich_1 = ich_lin_1 - irow_1 * 8;
            int iword_1 = irow_1 / 4;
            unsigned int ione_1 = 56 << 8 * (irow_1 & 3);
            unsigned int wv_1[4];
            wv_1[0] = 0;
            if (ich_1 * 4 == iword_1) {
                wv_1[0] = ione_1;
            }
            wv_1[1] = 0;
            if (ich_1 * 4 + 1 == iword_1) {
                wv_1[1] = ione_1;
            }
            wv_1[2] = 0;
            if (ich_1 * 4 + 2 == iword_1) {
                wv_1[2] = ione_1;
            }
            wv_1[3] = 0;
            if (ich_1 * 4 + 3 == iword_1) {
                wv_1[3] = ione_1;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow_1 * 128 + (ich_1 * 16 ^ (irow_1 & 7) << 4))), "r"(wv_1[0]), "r"(wv_1[1]), "r"(wv_1[2]), "r"(wv_1[3]) : "memory");
        }
        int ich_lin_2 = sid + 192;
        if (ich_lin_2 < 512) {
            int irow_2 = ich_lin_2 / 8;
            int ich_2 = ich_lin_2 - irow_2 * 8;
            int iword_2 = irow_2 / 4;
            unsigned int ione_2 = 56 << 8 * (irow_2 & 3);
            unsigned int wv_2[4];
            wv_2[0] = 0;
            if (ich_2 * 4 == iword_2) {
                wv_2[0] = ione_2;
            }
            wv_2[1] = 0;
            if (ich_2 * 4 + 1 == iword_2) {
                wv_2[1] = ione_2;
            }
            wv_2[2] = 0;
            if (ich_2 * 4 + 2 == iword_2) {
                wv_2[2] = ione_2;
            }
            wv_2[3] = 0;
            if (ich_2 * 4 + 3 == iword_2) {
                wv_2[3] = ione_2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow_2 * 128 + (ich_2 * 16 ^ (irow_2 & 7) << 4))), "r"(wv_2[0]), "r"(wv_2[1]), "r"(wv_2[2]), "r"(wv_2[3]) : "memory");
        }
        int ich_lin_3 = sid + 288;
        if (ich_lin_3 < 512) {
            int irow_3 = ich_lin_3 / 8;
            int ich_3 = ich_lin_3 - irow_3 * 8;
            int iword_3 = irow_3 / 4;
            unsigned int ione_3 = 56 << 8 * (irow_3 & 3);
            unsigned int wv_3[4];
            wv_3[0] = 0;
            if (ich_3 * 4 == iword_3) {
                wv_3[0] = ione_3;
            }
            wv_3[1] = 0;
            if (ich_3 * 4 + 1 == iword_3) {
                wv_3[1] = ione_3;
            }
            wv_3[2] = 0;
            if (ich_3 * 4 + 2 == iword_3) {
                wv_3[2] = ione_3;
            }
            wv_3[3] = 0;
            if (ich_3 * 4 + 3 == iword_3) {
                wv_3[3] = ione_3;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow_3 * 128 + (ich_3 * 16 ^ (irow_3 & 7) << 4))), "r"(wv_3[0]), "r"(wv_3[1]), "r"(wv_3[2]), "r"(wv_3[3]) : "memory");
        }
        int ich_lin_4 = sid + 384;
        if (ich_lin_4 < 512) {
            int irow_4 = ich_lin_4 / 8;
            int ich_4 = ich_lin_4 - irow_4 * 8;
            int iword_4 = irow_4 / 4;
            unsigned int ione_4 = 56 << 8 * (irow_4 & 3);
            unsigned int wv_4[4];
            wv_4[0] = 0;
            if (ich_4 * 4 == iword_4) {
                wv_4[0] = ione_4;
            }
            wv_4[1] = 0;
            if (ich_4 * 4 + 1 == iword_4) {
                wv_4[1] = ione_4;
            }
            wv_4[2] = 0;
            if (ich_4 * 4 + 2 == iword_4) {
                wv_4[2] = ione_4;
            }
            wv_4[3] = 0;
            if (ich_4 * 4 + 3 == iword_4) {
                wv_4[3] = ione_4;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow_4 * 128 + (ich_4 * 16 ^ (irow_4 & 7) << 4))), "r"(wv_4[0]), "r"(wv_4[1]), "r"(wv_4[2]), "r"(wv_4[3]) : "memory");
        }
        int ich_lin_5 = sid + 480;
        if (ich_lin_5 < 512) {
            int irow_5 = ich_lin_5 / 8;
            int ich_5 = ich_lin_5 - irow_5 * 8;
            int iword_5 = irow_5 / 4;
            unsigned int ione_5 = 56 << 8 * (irow_5 & 3);
            unsigned int wv_5[4];
            wv_5[0] = 0;
            if (ich_5 * 4 == iword_5) {
                wv_5[0] = ione_5;
            }
            wv_5[1] = 0;
            if (ich_5 * 4 + 1 == iword_5) {
                wv_5[1] = ione_5;
            }
            wv_5[2] = 0;
            if (ich_5 * 4 + 2 == iword_5) {
                wv_5[2] = ione_5;
            }
            wv_5[3] = 0;
            if (ich_5 * 4 + 3 == iword_5) {
                wv_5[3] = ione_5;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(ident8_addr + (unsigned int)(irow_5 * 128 + (ich_5 * 16 ^ (irow_5 & 7) << 4))), "r"(wv_5[0]), "r"(wv_5[1]), "r"(wv_5[2]), "r"(wv_5[3]) : "memory");
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    int b_ = hdr[0];
    int tok0 = hdr[1];
    int n_tok = hdr[2];
    int cu_b = hdr[3];
    int qpos0 = hdr[4];
    int blk_e[1];
    int e = tid_1;
    blk_e[0] = -1;
    if (e < 64) {
        int tl_e = e / 16;
        int k_e = e - tl_e * 16;
        if (tl_e < n_tok) {
            int bv = q2k_indices[(kv_head * total_q + cu_b + tok0 + tl_e) * 16 + k_e];
            if (bv >= 0) {
                if (bv < max_pages) {
                    blk_e[0] = bv;
                }
            }
        }
    }
    if (blk_e[0] >= 0) {
        {
            uint32_t _byte_0 = static_cast<uint32_t>(1) & 0xFFu;
            uint32_t _addr_0 = static_cast<uint32_t>(mark_addr + (unsigned int)blk_e[0]);
            asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_0), "r"(_byte_0) : "memory");
        }
    }
    unsigned int qw[16];
    int vid = tid_1;
    int qn = vid / 16;
    int qv = vid - qn * 16;
    qw[0] = 0;
    qw[1] = 0;
    qw[2] = 0;
    qw[3] = 0;
    int tl_q = qn >> 3;
    if (tl_q < n_tok) {
        int hg_q = qn - (tl_q << 3);
        int qrow = (cu_b + tok0 + tl_q) * num_q_heads + kv_head * 8 + hg_q;
        int qbase = qrow * 64 + qv * 4;
        qw[0] = Q32[qbase];
        qw[1] = Q32[qbase + 1];
        qw[2] = Q32[qbase + 2];
        qw[3] = Q32[qbase + 3];
    }
    int vid_2 = tid_1 + 128;
    int qn_3 = vid_2 / 16;
    int qv_4 = vid_2 - qn_3 * 16;
    qw[4] = 0;
    qw[5] = 0;
    qw[6] = 0;
    qw[7] = 0;
    int tl_q_5 = qn_3 >> 3;
    if (tl_q_5 < n_tok) {
        int hg_q_1 = qn_3 - (tl_q_5 << 3);
        int qrow_1 = (cu_b + tok0 + tl_q_5) * num_q_heads + kv_head * 8 + hg_q_1;
        int qbase_1 = qrow_1 * 64 + qv_4 * 4;
        qw[4] = Q32[qbase_1];
        qw[5] = Q32[qbase_1 + 1];
        qw[6] = Q32[qbase_1 + 2];
        qw[7] = Q32[qbase_1 + 3];
    }
    int vid_6 = tid_1 + 256;
    int qn_7 = vid_6 / 16;
    int qv_8 = vid_6 - qn_7 * 16;
    qw[8] = 0;
    qw[9] = 0;
    qw[10] = 0;
    qw[11] = 0;
    int tl_q_9 = qn_7 >> 3;
    if (tl_q_9 < n_tok) {
        int hg_q_2 = qn_7 - (tl_q_9 << 3);
        int qrow_2 = (cu_b + tok0 + tl_q_9) * num_q_heads + kv_head * 8 + hg_q_2;
        int qbase_2 = qrow_2 * 64 + qv_8 * 4;
        qw[8] = Q32[qbase_2];
        qw[9] = Q32[qbase_2 + 1];
        qw[10] = Q32[qbase_2 + 2];
        qw[11] = Q32[qbase_2 + 3];
    }
    int vid_10 = tid_1 + 384;
    int qn_11 = vid_10 / 16;
    int qv_12 = vid_10 - qn_11 * 16;
    qw[12] = 0;
    qw[13] = 0;
    qw[14] = 0;
    qw[15] = 0;
    int tl_q_13 = qn_11 >> 3;
    if (tl_q_13 < n_tok) {
        int hg_q_3 = qn_11 - (tl_q_13 << 3);
        int qrow_3 = (cu_b + tok0 + tl_q_13) * num_q_heads + kv_head * 8 + hg_q_3;
        int qbase_3 = qrow_3 * 64 + qv_12 * 4;
        qw[12] = Q32[qbase_3];
        qw[13] = Q32[qbase_3 + 1];
        qw[14] = Q32[qbase_3 + 2];
        qw[15] = Q32[qbase_3 + 3];
    }
    int vid2 = tid_1;
    int qn2 = vid2 / 16;
    int qv2 = vid2 - qn2 * 16;
    int qslab = qv2 / 8;
    int qcol = (qv2 - qslab * 8) * 16;
    int qoff2 = qslab * 4096 + qn2 * 128 + (qcol ^ (qn2 & 7) << 4);
    uint32_t _bf16x2_to_f16x2_0;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_0) : "r"(qw[0]));
    qw[0] = _bf16x2_to_f16x2_0;
    uint32_t _bf16x2_to_f16x2_1;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_1) : "r"(qw[1]));
    qw[1] = _bf16x2_to_f16x2_1;
    uint32_t _bf16x2_to_f16x2_2;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_2) : "r"(qw[2]));
    qw[2] = _bf16x2_to_f16x2_2;
    uint32_t _bf16x2_to_f16x2_3;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_3) : "r"(qw[3]));
    qw[3] = _bf16x2_to_f16x2_3;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff2), "r"(qw[0]), "r"(qw[1]), "r"(qw[2]), "r"(qw[3]) : "memory");
    int vid2_14 = tid_1 + 128;
    int qn2_15 = vid2_14 / 16;
    int qv2_16 = vid2_14 - qn2_15 * 16;
    int qslab_17 = qv2_16 / 8;
    int qcol_18 = (qv2_16 - qslab_17 * 8) * 16;
    int qoff2_19 = qslab_17 * 4096 + qn2_15 * 128 + (qcol_18 ^ (qn2_15 & 7) << 4);
    uint32_t _bf16x2_to_f16x2_4;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_4) : "r"(qw[4]));
    qw[4] = _bf16x2_to_f16x2_4;
    uint32_t _bf16x2_to_f16x2_5;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_5) : "r"(qw[5]));
    qw[5] = _bf16x2_to_f16x2_5;
    uint32_t _bf16x2_to_f16x2_6;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_6) : "r"(qw[6]));
    qw[6] = _bf16x2_to_f16x2_6;
    uint32_t _bf16x2_to_f16x2_7;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_7) : "r"(qw[7]));
    qw[7] = _bf16x2_to_f16x2_7;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff2_19), "r"(qw[4]), "r"(qw[5]), "r"(qw[6]), "r"(qw[7]) : "memory");
    int vid2_20 = tid_1 + 256;
    int qn2_21 = vid2_20 / 16;
    int qv2_22 = vid2_20 - qn2_21 * 16;
    int qslab_23 = qv2_22 / 8;
    int qcol_24 = (qv2_22 - qslab_23 * 8) * 16;
    int qoff2_25 = qslab_23 * 4096 + qn2_21 * 128 + (qcol_24 ^ (qn2_21 & 7) << 4);
    uint32_t _bf16x2_to_f16x2_8;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_8) : "r"(qw[8]));
    qw[8] = _bf16x2_to_f16x2_8;
    uint32_t _bf16x2_to_f16x2_9;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_9) : "r"(qw[9]));
    qw[9] = _bf16x2_to_f16x2_9;
    uint32_t _bf16x2_to_f16x2_10;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_10) : "r"(qw[10]));
    qw[10] = _bf16x2_to_f16x2_10;
    uint32_t _bf16x2_to_f16x2_11;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_11) : "r"(qw[11]));
    qw[11] = _bf16x2_to_f16x2_11;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff2_25), "r"(qw[8]), "r"(qw[9]), "r"(qw[10]), "r"(qw[11]) : "memory");
    int vid2_26 = tid_1 + 384;
    int qn2_27 = vid2_26 / 16;
    int qv2_28 = vid2_26 - qn2_27 * 16;
    int qslab_29 = qv2_28 / 8;
    int qcol_30 = (qv2_28 - qslab_29 * 8) * 16;
    int qoff2_31 = qslab_29 * 4096 + qn2_27 * 128 + (qcol_30 ^ (qn2_27 & 7) << 4);
    uint32_t _bf16x2_to_f16x2_12;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_12) : "r"(qw[12]));
    qw[12] = _bf16x2_to_f16x2_12;
    uint32_t _bf16x2_to_f16x2_13;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_13) : "r"(qw[13]));
    qw[13] = _bf16x2_to_f16x2_13;
    uint32_t _bf16x2_to_f16x2_14;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_14) : "r"(qw[14]));
    qw[14] = _bf16x2_to_f16x2_14;
    uint32_t _bf16x2_to_f16x2_15;
    asm(
        "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
        "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
        "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
        "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
        "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
        "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
        : "=r"(_bf16x2_to_f16x2_15) : "r"(qw[15]));
    qw[15] = _bf16x2_to_f16x2_15;
    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff2_31), "r"(qw[12]), "r"(qw[13]), "r"(qw[14]), "r"(qw[15]) : "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    int ngrp = (max_pages + 31) / 32;
    unsigned int mw[8];
    mw[0] = 0;
    mw[1] = 0;
    mw[2] = 0;
    mw[3] = 0;
    mw[4] = 0;
    mw[5] = 0;
    mw[6] = 0;
    mw[7] = 0;
    if (tid_1 < ngrp) {
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
            : "r"(mark_addr + (unsigned int)(tid_1 * 32)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
            : "r"(mark_addr + (unsigned int)(tid_1 * 32) + 16));
    }
    int c_t = 0;
    int _popc_0 = __popc(mw[0]);
    c_t = c_t + _popc_0;
    int _popc_1 = __popc(mw[1]);
    c_t = c_t + _popc_1;
    int _popc_2 = __popc(mw[2]);
    c_t = c_t + _popc_2;
    int _popc_3 = __popc(mw[3]);
    c_t = c_t + _popc_3;
    int _popc_4 = __popc(mw[4]);
    c_t = c_t + _popc_4;
    int _popc_5 = __popc(mw[5]);
    c_t = c_t + _popc_5;
    int _popc_6 = __popc(mw[6]);
    c_t = c_t + _popc_6;
    int _popc_7 = __popc(mw[7]);
    c_t = c_t + _popc_7;
    int incl_t = c_t;
    int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, incl_t, 1, 32);
    int up2 = _shfl_up_5;
    if (lane_0 >= 1) {
        incl_t = incl_t + up2;
    }
    int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, incl_t, 2, 32);
    int up2_32 = _shfl_up_6;
    if (lane_0 >= 2) {
        incl_t = incl_t + up2_32;
    }
    int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, incl_t, 4, 32);
    int up2_33 = _shfl_up_7;
    if (lane_0 >= 4) {
        incl_t = incl_t + up2_33;
    }
    int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, incl_t, 8, 32);
    int up2_34 = _shfl_up_8;
    if (lane_0 >= 8) {
        incl_t = incl_t + up2_34;
    }
    int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, incl_t, 16, 32);
    int up2_35 = _shfl_up_9;
    if (lane_0 >= 16) {
        incl_t = incl_t + up2_35;
    }
    if (lane_0 == 31) {
        cnt[warp_1] = incl_t;
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    int pos_t = incl_t - c_t;
    int nblk_u = 0;
    int cw_ = cnt[0];
    nblk_u = nblk_u + cw_;
    if (warp_1 > 0) {
        pos_t = pos_t + cw_;
    }
    int cw__36 = cnt[1];
    nblk_u = nblk_u + cw__36;
    if (warp_1 > 1) {
        pos_t = pos_t + cw__36;
    }
    int cw__37 = cnt[2];
    nblk_u = nblk_u + cw__37;
    if (warp_1 > 2) {
        pos_t = pos_t + cw__37;
    }
    int cw__38 = cnt[3];
    nblk_u = nblk_u + cw__38;
    if (warp_1 > 3) {
        pos_t = pos_t + cw__38;
    }
    pref[tid_1] = pos_t;
    if (c_t != 0) {
        if ((mw[0] & 1) != 0) {
            int blk_u = tid_1 * 32;
            int pg_u = page_table[b_ * max_pages + blk_u];
            int blk_st = blk_u;
            if (pg_u < 0) {
                pg_u = 0;
                blk_st = 4194304;
            }
            meta_blk[pos_t] = blk_st;
            meta_pg[pos_t] = pg_u * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[0] >> 8 & 1) != 0) {
            int blk_u_1 = tid_1 * 32 + 1;
            int pg_u_1 = page_table[b_ * max_pages + blk_u_1];
            int blk_st_1 = blk_u_1;
            if (pg_u_1 < 0) {
                pg_u_1 = 0;
                blk_st_1 = 4194304;
            }
            meta_blk[pos_t] = blk_st_1;
            meta_pg[pos_t] = pg_u_1 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[0] >> 16 & 1) != 0) {
            int blk_u_2 = tid_1 * 32 + 2;
            int pg_u_2 = page_table[b_ * max_pages + blk_u_2];
            int blk_st_2 = blk_u_2;
            if (pg_u_2 < 0) {
                pg_u_2 = 0;
                blk_st_2 = 4194304;
            }
            meta_blk[pos_t] = blk_st_2;
            meta_pg[pos_t] = pg_u_2 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[0] >> 24 & 1) != 0) {
            int blk_u_3 = tid_1 * 32 + 3;
            int pg_u_3 = page_table[b_ * max_pages + blk_u_3];
            int blk_st_3 = blk_u_3;
            if (pg_u_3 < 0) {
                pg_u_3 = 0;
                blk_st_3 = 4194304;
            }
            meta_blk[pos_t] = blk_st_3;
            meta_pg[pos_t] = pg_u_3 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[1] & 1) != 0) {
            int blk_u_4 = tid_1 * 32 + 4;
            int pg_u_4 = page_table[b_ * max_pages + blk_u_4];
            int blk_st_4 = blk_u_4;
            if (pg_u_4 < 0) {
                pg_u_4 = 0;
                blk_st_4 = 4194304;
            }
            meta_blk[pos_t] = blk_st_4;
            meta_pg[pos_t] = pg_u_4 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[1] >> 8 & 1) != 0) {
            int blk_u_5 = tid_1 * 32 + 4 + 1;
            int pg_u_5 = page_table[b_ * max_pages + blk_u_5];
            int blk_st_5 = blk_u_5;
            if (pg_u_5 < 0) {
                pg_u_5 = 0;
                blk_st_5 = 4194304;
            }
            meta_blk[pos_t] = blk_st_5;
            meta_pg[pos_t] = pg_u_5 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[1] >> 16 & 1) != 0) {
            int blk_u_6 = tid_1 * 32 + 4 + 2;
            int pg_u_6 = page_table[b_ * max_pages + blk_u_6];
            int blk_st_6 = blk_u_6;
            if (pg_u_6 < 0) {
                pg_u_6 = 0;
                blk_st_6 = 4194304;
            }
            meta_blk[pos_t] = blk_st_6;
            meta_pg[pos_t] = pg_u_6 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[1] >> 24 & 1) != 0) {
            int blk_u_7 = tid_1 * 32 + 4 + 3;
            int pg_u_7 = page_table[b_ * max_pages + blk_u_7];
            int blk_st_7 = blk_u_7;
            if (pg_u_7 < 0) {
                pg_u_7 = 0;
                blk_st_7 = 4194304;
            }
            meta_blk[pos_t] = blk_st_7;
            meta_pg[pos_t] = pg_u_7 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[2] & 1) != 0) {
            int blk_u_8 = tid_1 * 32 + 8;
            int pg_u_8 = page_table[b_ * max_pages + blk_u_8];
            int blk_st_8 = blk_u_8;
            if (pg_u_8 < 0) {
                pg_u_8 = 0;
                blk_st_8 = 4194304;
            }
            meta_blk[pos_t] = blk_st_8;
            meta_pg[pos_t] = pg_u_8 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[2] >> 8 & 1) != 0) {
            int blk_u_9 = tid_1 * 32 + 8 + 1;
            int pg_u_9 = page_table[b_ * max_pages + blk_u_9];
            int blk_st_9 = blk_u_9;
            if (pg_u_9 < 0) {
                pg_u_9 = 0;
                blk_st_9 = 4194304;
            }
            meta_blk[pos_t] = blk_st_9;
            meta_pg[pos_t] = pg_u_9 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[2] >> 16 & 1) != 0) {
            int blk_u_10 = tid_1 * 32 + 8 + 2;
            int pg_u_10 = page_table[b_ * max_pages + blk_u_10];
            int blk_st_10 = blk_u_10;
            if (pg_u_10 < 0) {
                pg_u_10 = 0;
                blk_st_10 = 4194304;
            }
            meta_blk[pos_t] = blk_st_10;
            meta_pg[pos_t] = pg_u_10 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[2] >> 24 & 1) != 0) {
            int blk_u_11 = tid_1 * 32 + 8 + 3;
            int pg_u_11 = page_table[b_ * max_pages + blk_u_11];
            int blk_st_11 = blk_u_11;
            if (pg_u_11 < 0) {
                pg_u_11 = 0;
                blk_st_11 = 4194304;
            }
            meta_blk[pos_t] = blk_st_11;
            meta_pg[pos_t] = pg_u_11 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[3] & 1) != 0) {
            int blk_u_12 = tid_1 * 32 + 12;
            int pg_u_12 = page_table[b_ * max_pages + blk_u_12];
            int blk_st_12 = blk_u_12;
            if (pg_u_12 < 0) {
                pg_u_12 = 0;
                blk_st_12 = 4194304;
            }
            meta_blk[pos_t] = blk_st_12;
            meta_pg[pos_t] = pg_u_12 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[3] >> 8 & 1) != 0) {
            int blk_u_13 = tid_1 * 32 + 12 + 1;
            int pg_u_13 = page_table[b_ * max_pages + blk_u_13];
            int blk_st_13 = blk_u_13;
            if (pg_u_13 < 0) {
                pg_u_13 = 0;
                blk_st_13 = 4194304;
            }
            meta_blk[pos_t] = blk_st_13;
            meta_pg[pos_t] = pg_u_13 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[3] >> 16 & 1) != 0) {
            int blk_u_14 = tid_1 * 32 + 12 + 2;
            int pg_u_14 = page_table[b_ * max_pages + blk_u_14];
            int blk_st_14 = blk_u_14;
            if (pg_u_14 < 0) {
                pg_u_14 = 0;
                blk_st_14 = 4194304;
            }
            meta_blk[pos_t] = blk_st_14;
            meta_pg[pos_t] = pg_u_14 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[3] >> 24 & 1) != 0) {
            int blk_u_15 = tid_1 * 32 + 12 + 3;
            int pg_u_15 = page_table[b_ * max_pages + blk_u_15];
            int blk_st_15 = blk_u_15;
            if (pg_u_15 < 0) {
                pg_u_15 = 0;
                blk_st_15 = 4194304;
            }
            meta_blk[pos_t] = blk_st_15;
            meta_pg[pos_t] = pg_u_15 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[4] & 1) != 0) {
            int blk_u_16 = tid_1 * 32 + 16;
            int pg_u_16 = page_table[b_ * max_pages + blk_u_16];
            int blk_st_16 = blk_u_16;
            if (pg_u_16 < 0) {
                pg_u_16 = 0;
                blk_st_16 = 4194304;
            }
            meta_blk[pos_t] = blk_st_16;
            meta_pg[pos_t] = pg_u_16 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[4] >> 8 & 1) != 0) {
            int blk_u_17 = tid_1 * 32 + 16 + 1;
            int pg_u_17 = page_table[b_ * max_pages + blk_u_17];
            int blk_st_17 = blk_u_17;
            if (pg_u_17 < 0) {
                pg_u_17 = 0;
                blk_st_17 = 4194304;
            }
            meta_blk[pos_t] = blk_st_17;
            meta_pg[pos_t] = pg_u_17 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[4] >> 16 & 1) != 0) {
            int blk_u_18 = tid_1 * 32 + 16 + 2;
            int pg_u_18 = page_table[b_ * max_pages + blk_u_18];
            int blk_st_18 = blk_u_18;
            if (pg_u_18 < 0) {
                pg_u_18 = 0;
                blk_st_18 = 4194304;
            }
            meta_blk[pos_t] = blk_st_18;
            meta_pg[pos_t] = pg_u_18 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[4] >> 24 & 1) != 0) {
            int blk_u_19 = tid_1 * 32 + 16 + 3;
            int pg_u_19 = page_table[b_ * max_pages + blk_u_19];
            int blk_st_19 = blk_u_19;
            if (pg_u_19 < 0) {
                pg_u_19 = 0;
                blk_st_19 = 4194304;
            }
            meta_blk[pos_t] = blk_st_19;
            meta_pg[pos_t] = pg_u_19 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[5] & 1) != 0) {
            int blk_u_20 = tid_1 * 32 + 20;
            int pg_u_20 = page_table[b_ * max_pages + blk_u_20];
            int blk_st_20 = blk_u_20;
            if (pg_u_20 < 0) {
                pg_u_20 = 0;
                blk_st_20 = 4194304;
            }
            meta_blk[pos_t] = blk_st_20;
            meta_pg[pos_t] = pg_u_20 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[5] >> 8 & 1) != 0) {
            int blk_u_21 = tid_1 * 32 + 20 + 1;
            int pg_u_21 = page_table[b_ * max_pages + blk_u_21];
            int blk_st_21 = blk_u_21;
            if (pg_u_21 < 0) {
                pg_u_21 = 0;
                blk_st_21 = 4194304;
            }
            meta_blk[pos_t] = blk_st_21;
            meta_pg[pos_t] = pg_u_21 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[5] >> 16 & 1) != 0) {
            int blk_u_22 = tid_1 * 32 + 20 + 2;
            int pg_u_22 = page_table[b_ * max_pages + blk_u_22];
            int blk_st_22 = blk_u_22;
            if (pg_u_22 < 0) {
                pg_u_22 = 0;
                blk_st_22 = 4194304;
            }
            meta_blk[pos_t] = blk_st_22;
            meta_pg[pos_t] = pg_u_22 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[5] >> 24 & 1) != 0) {
            int blk_u_23 = tid_1 * 32 + 20 + 3;
            int pg_u_23 = page_table[b_ * max_pages + blk_u_23];
            int blk_st_23 = blk_u_23;
            if (pg_u_23 < 0) {
                pg_u_23 = 0;
                blk_st_23 = 4194304;
            }
            meta_blk[pos_t] = blk_st_23;
            meta_pg[pos_t] = pg_u_23 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[6] & 1) != 0) {
            int blk_u_24 = tid_1 * 32 + 24;
            int pg_u_24 = page_table[b_ * max_pages + blk_u_24];
            int blk_st_24 = blk_u_24;
            if (pg_u_24 < 0) {
                pg_u_24 = 0;
                blk_st_24 = 4194304;
            }
            meta_blk[pos_t] = blk_st_24;
            meta_pg[pos_t] = pg_u_24 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[6] >> 8 & 1) != 0) {
            int blk_u_25 = tid_1 * 32 + 24 + 1;
            int pg_u_25 = page_table[b_ * max_pages + blk_u_25];
            int blk_st_25 = blk_u_25;
            if (pg_u_25 < 0) {
                pg_u_25 = 0;
                blk_st_25 = 4194304;
            }
            meta_blk[pos_t] = blk_st_25;
            meta_pg[pos_t] = pg_u_25 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[6] >> 16 & 1) != 0) {
            int blk_u_26 = tid_1 * 32 + 24 + 2;
            int pg_u_26 = page_table[b_ * max_pages + blk_u_26];
            int blk_st_26 = blk_u_26;
            if (pg_u_26 < 0) {
                pg_u_26 = 0;
                blk_st_26 = 4194304;
            }
            meta_blk[pos_t] = blk_st_26;
            meta_pg[pos_t] = pg_u_26 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[6] >> 24 & 1) != 0) {
            int blk_u_27 = tid_1 * 32 + 24 + 3;
            int pg_u_27 = page_table[b_ * max_pages + blk_u_27];
            int blk_st_27 = blk_u_27;
            if (pg_u_27 < 0) {
                pg_u_27 = 0;
                blk_st_27 = 4194304;
            }
            meta_blk[pos_t] = blk_st_27;
            meta_pg[pos_t] = pg_u_27 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[7] & 1) != 0) {
            int blk_u_28 = tid_1 * 32 + 28;
            int pg_u_28 = page_table[b_ * max_pages + blk_u_28];
            int blk_st_28 = blk_u_28;
            if (pg_u_28 < 0) {
                pg_u_28 = 0;
                blk_st_28 = 4194304;
            }
            meta_blk[pos_t] = blk_st_28;
            meta_pg[pos_t] = pg_u_28 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[7] >> 8 & 1) != 0) {
            int blk_u_29 = tid_1 * 32 + 28 + 1;
            int pg_u_29 = page_table[b_ * max_pages + blk_u_29];
            int blk_st_29 = blk_u_29;
            if (pg_u_29 < 0) {
                pg_u_29 = 0;
                blk_st_29 = 4194304;
            }
            meta_blk[pos_t] = blk_st_29;
            meta_pg[pos_t] = pg_u_29 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[7] >> 16 & 1) != 0) {
            int blk_u_30 = tid_1 * 32 + 28 + 2;
            int pg_u_30 = page_table[b_ * max_pages + blk_u_30];
            int blk_st_30 = blk_u_30;
            if (pg_u_30 < 0) {
                pg_u_30 = 0;
                blk_st_30 = 4194304;
            }
            meta_blk[pos_t] = blk_st_30;
            meta_pg[pos_t] = pg_u_30 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
        if ((mw[7] >> 24 & 1) != 0) {
            int blk_u_31 = tid_1 * 32 + 28 + 3;
            int pg_u_31 = page_table[b_ * max_pages + blk_u_31];
            int blk_st_31 = blk_u_31;
            if (pg_u_31 < 0) {
                pg_u_31 = 0;
                blk_st_31 = 4194304;
            }
            meta_blk[pos_t] = blk_st_31;
            meta_pg[pos_t] = pg_u_31 * num_kv_heads + kv_head;
            meta_mask[pos_t] = 0;
            pos_t = pos_t + 1;
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    if (blk_e[0] >= 0) {
        int grp_e = blk_e[0] >> 5;
        int o_e = blk_e[0] & 31;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
            : "r"(mark_addr + (unsigned int)(grp_e * 32)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
            : "r"(mark_addr + (unsigned int)(grp_e * 32) + 16));
        int rank = pref[grp_e];
        int nb_ = o_e;
        if (nb_ > 0) {
            if (nb_ >= 4) {
                int _popc_8 = __popc(mw[0]);
                rank = rank + _popc_8;
            }
            if (nb_ < 4) {
                unsigned int lowmask = (1 << 8 * (nb_ & 3)) - 1;
                int _popc_9 = __popc(mw[0] & lowmask);
                rank = rank + _popc_9;
            }
        }
        int nb__0 = o_e - 4;
        if (nb__0 > 0) {
            if (nb__0 >= 4) {
                int _popc_10 = __popc(mw[1]);
                rank = rank + _popc_10;
            }
            if (nb__0 < 4) {
                unsigned int lowmask_1 = (1 << 8 * (nb__0 & 3)) - 1;
                int _popc_11 = __popc(mw[1] & lowmask_1);
                rank = rank + _popc_11;
            }
        }
        int nb__1 = o_e - 8;
        if (nb__1 > 0) {
            if (nb__1 >= 4) {
                int _popc_12 = __popc(mw[2]);
                rank = rank + _popc_12;
            }
            if (nb__1 < 4) {
                unsigned int lowmask_2 = (1 << 8 * (nb__1 & 3)) - 1;
                int _popc_13 = __popc(mw[2] & lowmask_2);
                rank = rank + _popc_13;
            }
        }
        int nb__2 = o_e - 12;
        if (nb__2 > 0) {
            if (nb__2 >= 4) {
                int _popc_14 = __popc(mw[3]);
                rank = rank + _popc_14;
            }
            if (nb__2 < 4) {
                unsigned int lowmask_3 = (1 << 8 * (nb__2 & 3)) - 1;
                int _popc_15 = __popc(mw[3] & lowmask_3);
                rank = rank + _popc_15;
            }
        }
        int nb__3 = o_e - 16;
        if (nb__3 > 0) {
            if (nb__3 >= 4) {
                int _popc_16 = __popc(mw[4]);
                rank = rank + _popc_16;
            }
            if (nb__3 < 4) {
                unsigned int lowmask_4 = (1 << 8 * (nb__3 & 3)) - 1;
                int _popc_17 = __popc(mw[4] & lowmask_4);
                rank = rank + _popc_17;
            }
        }
        int nb__4 = o_e - 20;
        if (nb__4 > 0) {
            if (nb__4 >= 4) {
                int _popc_18 = __popc(mw[5]);
                rank = rank + _popc_18;
            }
            if (nb__4 < 4) {
                unsigned int lowmask_5 = (1 << 8 * (nb__4 & 3)) - 1;
                int _popc_19 = __popc(mw[5] & lowmask_5);
                rank = rank + _popc_19;
            }
        }
        int nb__5 = o_e - 24;
        if (nb__5 > 0) {
            if (nb__5 >= 4) {
                int _popc_20 = __popc(mw[6]);
                rank = rank + _popc_20;
            }
            if (nb__5 < 4) {
                unsigned int lowmask_6 = (1 << 8 * (nb__5 & 3)) - 1;
                int _popc_21 = __popc(mw[6] & lowmask_6);
                rank = rank + _popc_21;
            }
        }
        int nb__6 = o_e - 28;
        if (nb__6 > 0) {
            if (nb__6 >= 4) {
                int _popc_22 = __popc(mw[7]);
                rank = rank + _popc_22;
            }
            if (nb__6 < 4) {
                unsigned int lowmask_7 = (1 << 8 * (nb__6 & 3)) - 1;
                int _popc_23 = __popc(mw[7] & lowmask_7);
                rank = rank + _popc_23;
            }
        }
        int tl_i = tid_1 / 16;
        unsigned int bit = 1 << tl_i;
        atomicAdd_block(&meta_mask[rank], bit);
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    if (nblk_u > 0) {
        int page_head = meta_pg[0];
        if (warp == 1) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(kv_full_addr, 32768);
                tma_3d_gmem2smem(kv8_addr, (&K), 0, 0, page_head, kv_full_addr);
                tma_3d_gmem2smem(kv8_addr + 8192, (&K), 0, 64, page_head, kv_full_addr);
                tma_3d_gmem2smem(kv8_addr + 16384, (&V), 0, 0, page_head, kv_full_addr);
                tma_3d_gmem2smem(kv8_addr + 24576, (&V), 0, 64, page_head, kv_full_addr);
            }
        }
    }
    if (nblk_u > 1) {
        int page_head_1 = meta_pg[1];
        if (warp == 1) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(kv_full_addr + 8, 32768);
                tma_3d_gmem2smem(kv8_addr + 32768, (&K), 0, 0, page_head_1, kv_full_addr + 8);
                tma_3d_gmem2smem(kv8_addr + 32768 + 8192, (&K), 0, 64, page_head_1, kv_full_addr + 8);
                tma_3d_gmem2smem(kv8_addr + 32768 + 16384, (&V), 0, 0, page_head_1, kv_full_addr + 8);
                tma_3d_gmem2smem(kv8_addr + 32768 + 24576, (&V), 0, 64, page_head_1, kv_full_addr + 8);
            }
        }
    }
    int qpos1 = qpos0 + 1;
    unsigned int prev_bl = 15;
    unsigned int dk0[32];
    unsigned int dk1[32];
    unsigned int dv0[32];
    unsigned int dv1[32];
    unsigned int ak0[32];
    unsigned int ak1[32];
    unsigned int av0[32];
    unsigned int av1[32];
    float d_s0[16];
    float d_s1[16];
    float d_o0[16];
    float d_o1[16];
    d_o0[0] = 0.0f;
    d_o1[0] = 0.0f;
    d_o0[1] = 0.0f;
    d_o1[1] = 0.0f;
    d_o0[2] = 0.0f;
    d_o1[2] = 0.0f;
    d_o0[3] = 0.0f;
    d_o1[3] = 0.0f;
    d_o0[4] = 0.0f;
    d_o1[4] = 0.0f;
    d_o0[5] = 0.0f;
    d_o1[5] = 0.0f;
    d_o0[6] = 0.0f;
    d_o1[6] = 0.0f;
    d_o0[7] = 0.0f;
    d_o1[7] = 0.0f;
    d_o0[8] = 0.0f;
    d_o1[8] = 0.0f;
    d_o0[9] = 0.0f;
    d_o1[9] = 0.0f;
    d_o0[10] = 0.0f;
    d_o1[10] = 0.0f;
    d_o0[11] = 0.0f;
    d_o1[11] = 0.0f;
    d_o0[12] = 0.0f;
    d_o1[12] = 0.0f;
    d_o0[13] = 0.0f;
    d_o1[13] = 0.0f;
    d_o0[14] = 0.0f;
    d_o1[14] = 0.0f;
    d_o0[15] = 0.0f;
    d_o1[15] = 0.0f;
    float mrun[8];
    float lp[8];
    mrun[0] = -CAKE_INF;
    lp[0] = 0.0f;
    mrun[1] = -CAKE_INF;
    lp[1] = 0.0f;
    mrun[2] = -CAKE_INF;
    lp[2] = 0.0f;
    mrun[3] = -CAKE_INF;
    lp[3] = 0.0f;
    mrun[4] = -CAKE_INF;
    lp[4] = 0.0f;
    mrun[5] = -CAKE_INF;
    lp[5] = 0.0f;
    mrun[6] = -CAKE_INF;
    lp[6] = 0.0f;
    mrun[7] = -CAKE_INF;
    lp[7] = 0.0f;
    float tmax[8];
    float rv[4];
    int first_page = 1;
    uint64_t _wgmma_desc_1 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((q16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    uint64_t _wgmma_desc_3 = (((uint64_t)(((q16_addr + 4096)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
    uint64_t _wgmma_desc_4 = (((uint64_t)(((p16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_7 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
    uint64_t _wgmma_desc_5 = (((uint64_t)(((p16_addr + 4096)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_8 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
    #pragma unroll 1
    for (int j = 0; j < nblk_u; j++) {
        int stage = j % 2;
        int phase = j / 2 % 2;
        int blk_j = meta_blk[j];
        unsigned int mask_j = meta_mask[j];
        int pbase = blk_j * 128;
        unsigned int bl = 0;
        if ((mask_j & 1) != 0) {
            bl = bl | 1;
        }
        if ((mask_j >> 1 & 1) != 0) {
            bl = bl | 2;
        }
        if ((mask_j >> 2 & 1) != 0) {
            bl = bl | 4;
        }
        if ((mask_j >> 3 & 1) != 0) {
            bl = bl | 8;
        }
        mbarrier_wait(kv_full_addr + (stage) * 8, phase);
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_6 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk0[0]), "+r"(dk0[1]), "+r"(dk0[2]), "+r"(dk0[3]), "+r"(dk0[4]), "+r"(dk0[5]), "+r"(dk0[6]), "+r"(dk0[7])
            : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk0[8]), "+r"(dk0[(8) + 1]), "+r"(dk0[(8) + 2]), "+r"(dk0[(8) + 3]), "+r"(dk0[(8) + 4]), "+r"(dk0[(8) + 5]), "+r"(dk0[(8) + 6]), "+r"(dk0[(8) + 7])
            : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk0[16]), "+r"(dk0[(16) + 1]), "+r"(dk0[(16) + 2]), "+r"(dk0[(16) + 3]), "+r"(dk0[(16) + 4]), "+r"(dk0[(16) + 5]), "+r"(dk0[(16) + 6]), "+r"(dk0[(16) + 7])
            : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk0[24]), "+r"(dk0[(24) + 1]), "+r"(dk0[(24) + 2]), "+r"(dk0[(24) + 3]), "+r"(dk0[(24) + 4]), "+r"(dk0[(24) + 5]), "+r"(dk0[(24) + 6]), "+r"(dk0[(24) + 7])
            : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_7 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_7 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_7);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk1[0]), "+r"(dk1[1]), "+r"(dk1[2]), "+r"(dk1[3]), "+r"(dk1[4]), "+r"(dk1[5]), "+r"(dk1[6]), "+r"(dk1[7])
            : "l"(_wgmma_a_0_2), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk1[8]), "+r"(dk1[(8) + 1]), "+r"(dk1[(8) + 2]), "+r"(dk1[(8) + 3]), "+r"(dk1[(8) + 4]), "+r"(dk1[(8) + 5]), "+r"(dk1[(8) + 6]), "+r"(dk1[(8) + 7])
            : "l"(_wgmma_a_0_2 + 2), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk1[16]), "+r"(dk1[(16) + 1]), "+r"(dk1[(16) + 2]), "+r"(dk1[(16) + 3]), "+r"(dk1[(16) + 4]), "+r"(dk1[(16) + 5]), "+r"(dk1[(16) + 6]), "+r"(dk1[(16) + 7])
            : "l"(_wgmma_a_0_2 + 4), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
            : "+r"(dk1[24]), "+r"(dk1[(24) + 1]), "+r"(dk1[(24) + 2]), "+r"(dk1[(24) + 3]), "+r"(dk1[(24) + 4]), "+r"(dk1[(24) + 5]), "+r"(dk1[(24) + 6]), "+r"(dk1[(24) + 7])
            : "l"(_wgmma_a_0_2 + 6), "l"(_wgmma_b_0_1)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        ak0[0] = dk0[0] + zero_u32;
        ak0[1] = dk0[1] + zero_u32;
        ak0[2] = dk0[2] + zero_u32;
        ak0[3] = dk0[3] + zero_u32;
        ak0[4] = dk0[4] + zero_u32;
        ak0[5] = dk0[5] + zero_u32;
        ak0[6] = dk0[6] + zero_u32;
        ak0[7] = dk0[7] + zero_u32;
        ak0[8] = dk0[8] + zero_u32;
        ak0[9] = dk0[9] + zero_u32;
        ak0[10] = dk0[10] + zero_u32;
        ak0[11] = dk0[11] + zero_u32;
        ak0[12] = dk0[12] + zero_u32;
        ak0[13] = dk0[13] + zero_u32;
        ak0[14] = dk0[14] + zero_u32;
        ak0[15] = dk0[15] + zero_u32;
        ak0[16] = dk0[16] + zero_u32;
        ak0[17] = dk0[17] + zero_u32;
        ak0[18] = dk0[18] + zero_u32;
        ak0[19] = dk0[19] + zero_u32;
        ak0[20] = dk0[20] + zero_u32;
        ak0[21] = dk0[21] + zero_u32;
        ak0[22] = dk0[22] + zero_u32;
        ak0[23] = dk0[23] + zero_u32;
        ak0[24] = dk0[24] + zero_u32;
        ak0[25] = dk0[25] + zero_u32;
        ak0[26] = dk0[26] + zero_u32;
        ak0[27] = dk0[27] + zero_u32;
        ak0[28] = dk0[28] + zero_u32;
        ak0[29] = dk0[29] + zero_u32;
        ak0[30] = dk0[30] + zero_u32;
        ak0[31] = dk0[31] + zero_u32;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[0]), "r"(ak0[1]), "r"(ak0[2]), "r"(ak0[3]), "l"(_wgmma_b_0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[4]), "r"(ak0[(4) + 1]), "r"(ak0[(4) + 2]), "r"(ak0[(4) + 3]), "l"(_wgmma_b_0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[8]), "r"(ak0[(8) + 1]), "r"(ak0[(8) + 2]), "r"(ak0[(8) + 3]), "l"(_wgmma_b_0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[12]), "r"(ak0[(12) + 1]), "r"(ak0[(12) + 2]), "r"(ak0[(12) + 3]), "l"(_wgmma_b_0_3 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[16]), "r"(ak0[(16) + 1]), "r"(ak0[(16) + 2]), "r"(ak0[(16) + 3]), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[20]), "r"(ak0[(20) + 1]), "r"(ak0[(20) + 2]), "r"(ak0[(20) + 3]), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[24]), "r"(ak0[(24) + 1]), "r"(ak0[(24) + 2]), "r"(ak0[(24) + 3]), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0[0]), "+f"(d_s0[1]), "+f"(d_s0[2]), "+f"(d_s0[3]), "+f"(d_s0[4]), "+f"(d_s0[5]), "+f"(d_s0[6]), "+f"(d_s0[7]), "+f"(d_s0[8]), "+f"(d_s0[9]), "+f"(d_s0[10]), "+f"(d_s0[11]), "+f"(d_s0[12]), "+f"(d_s0[13]), "+f"(d_s0[14]), "+f"(d_s0[15])
            : "r"(ak0[28]), "r"(ak0[(28) + 1]), "r"(ak0[(28) + 2]), "r"(ak0[(28) + 3]), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        ak1[0] = dk1[0] + zero_u32;
        ak1[1] = dk1[1] + zero_u32;
        ak1[2] = dk1[2] + zero_u32;
        ak1[3] = dk1[3] + zero_u32;
        ak1[4] = dk1[4] + zero_u32;
        ak1[5] = dk1[5] + zero_u32;
        ak1[6] = dk1[6] + zero_u32;
        ak1[7] = dk1[7] + zero_u32;
        ak1[8] = dk1[8] + zero_u32;
        ak1[9] = dk1[9] + zero_u32;
        ak1[10] = dk1[10] + zero_u32;
        ak1[11] = dk1[11] + zero_u32;
        ak1[12] = dk1[12] + zero_u32;
        ak1[13] = dk1[13] + zero_u32;
        ak1[14] = dk1[14] + zero_u32;
        ak1[15] = dk1[15] + zero_u32;
        ak1[16] = dk1[16] + zero_u32;
        ak1[17] = dk1[17] + zero_u32;
        ak1[18] = dk1[18] + zero_u32;
        ak1[19] = dk1[19] + zero_u32;
        ak1[20] = dk1[20] + zero_u32;
        ak1[21] = dk1[21] + zero_u32;
        ak1[22] = dk1[22] + zero_u32;
        ak1[23] = dk1[23] + zero_u32;
        ak1[24] = dk1[24] + zero_u32;
        ak1[25] = dk1[25] + zero_u32;
        ak1[26] = dk1[26] + zero_u32;
        ak1[27] = dk1[27] + zero_u32;
        ak1[28] = dk1[28] + zero_u32;
        ak1[29] = dk1[29] + zero_u32;
        ak1[30] = dk1[30] + zero_u32;
        ak1[31] = dk1[31] + zero_u32;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[0]), "r"(ak1[1]), "r"(ak1[2]), "r"(ak1[3]), "l"(_wgmma_b_0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[4]), "r"(ak1[(4) + 1]), "r"(ak1[(4) + 2]), "r"(ak1[(4) + 3]), "l"(_wgmma_b_0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[8]), "r"(ak1[(8) + 1]), "r"(ak1[(8) + 2]), "r"(ak1[(8) + 3]), "l"(_wgmma_b_0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[12]), "r"(ak1[(12) + 1]), "r"(ak1[(12) + 2]), "r"(ak1[(12) + 3]), "l"(_wgmma_b_0_3 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[16]), "r"(ak1[(16) + 1]), "r"(ak1[(16) + 2]), "r"(ak1[(16) + 3]), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[20]), "r"(ak1[(20) + 1]), "r"(ak1[(20) + 2]), "r"(ak1[(20) + 3]), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[24]), "r"(ak1[(24) + 1]), "r"(ak1[(24) + 2]), "r"(ak1[(24) + 3]), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1[0]), "+f"(d_s1[1]), "+f"(d_s1[2]), "+f"(d_s1[3]), "+f"(d_s1[4]), "+f"(d_s1[5]), "+f"(d_s1[6]), "+f"(d_s1[7]), "+f"(d_s1[8]), "+f"(d_s1[9]), "+f"(d_s1[10]), "+f"(d_s1[11]), "+f"(d_s1[12]), "+f"(d_s1[13]), "+f"(d_s1[14]), "+f"(d_s1[15])
            : "r"(ak1[28]), "r"(ak1[(28) + 1]), "r"(ak1[(28) + 2]), "r"(ak1[(28) + 3]), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_8 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_8 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_8);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
            : "+r"(dv0[0]), "+r"(dv0[1]), "+r"(dv0[2]), "+r"(dv0[3]), "+r"(dv0[4]), "+r"(dv0[5]), "+r"(dv0[6]), "+r"(dv0[7]), "+r"(dv0[8]), "+r"(dv0[9]), "+r"(dv0[10]), "+r"(dv0[11]), "+r"(dv0[12]), "+r"(dv0[13]), "+r"(dv0[14]), "+r"(dv0[15])
            : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_5)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
            : "+r"(dv0[0]), "+r"(dv0[1]), "+r"(dv0[2]), "+r"(dv0[3]), "+r"(dv0[4]), "+r"(dv0[5]), "+r"(dv0[6]), "+r"(dv0[7]), "+r"(dv0[8]), "+r"(dv0[9]), "+r"(dv0[10]), "+r"(dv0[11]), "+r"(dv0[12]), "+r"(dv0[13]), "+r"(dv0[14]), "+r"(dv0[15])
            : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_5 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
            : "+r"(dv0[16]), "+r"(dv0[(16) + 1]), "+r"(dv0[(16) + 2]), "+r"(dv0[(16) + 3]), "+r"(dv0[(16) + 4]), "+r"(dv0[(16) + 5]), "+r"(dv0[(16) + 6]), "+r"(dv0[(16) + 7]), "+r"(dv0[(16) + 8]), "+r"(dv0[(16) + 9]), "+r"(dv0[(16) + 10]), "+r"(dv0[(16) + 11]), "+r"(dv0[(16) + 12]), "+r"(dv0[(16) + 13]), "+r"(dv0[(16) + 14]), "+r"(dv0[(16) + 15])
            : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_5 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
            : "+r"(dv0[16]), "+r"(dv0[(16) + 1]), "+r"(dv0[(16) + 2]), "+r"(dv0[(16) + 3]), "+r"(dv0[(16) + 4]), "+r"(dv0[(16) + 5]), "+r"(dv0[(16) + 6]), "+r"(dv0[(16) + 7]), "+r"(dv0[(16) + 8]), "+r"(dv0[(16) + 9]), "+r"(dv0[(16) + 10]), "+r"(dv0[(16) + 11]), "+r"(dv0[(16) + 12]), "+r"(dv0[(16) + 13]), "+r"(dv0[(16) + 14]), "+r"(dv0[(16) + 15])
            : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_5 + 6)
            : "memory");
        uint64_t _wgmma_desc_9 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 24576)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_9 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_9);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
            : "+r"(dv1[0]), "+r"(dv1[1]), "+r"(dv1[2]), "+r"(dv1[3]), "+r"(dv1[4]), "+r"(dv1[5]), "+r"(dv1[6]), "+r"(dv1[7]), "+r"(dv1[8]), "+r"(dv1[9]), "+r"(dv1[10]), "+r"(dv1[11]), "+r"(dv1[12]), "+r"(dv1[13]), "+r"(dv1[14]), "+r"(dv1[15])
            : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
            : "+r"(dv1[0]), "+r"(dv1[1]), "+r"(dv1[2]), "+r"(dv1[3]), "+r"(dv1[4]), "+r"(dv1[5]), "+r"(dv1[6]), "+r"(dv1[7]), "+r"(dv1[8]), "+r"(dv1[9]), "+r"(dv1[10]), "+r"(dv1[11]), "+r"(dv1[12]), "+r"(dv1[13]), "+r"(dv1[14]), "+r"(dv1[15])
            : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_6 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
            : "+r"(dv1[16]), "+r"(dv1[(16) + 1]), "+r"(dv1[(16) + 2]), "+r"(dv1[(16) + 3]), "+r"(dv1[(16) + 4]), "+r"(dv1[(16) + 5]), "+r"(dv1[(16) + 6]), "+r"(dv1[(16) + 7]), "+r"(dv1[(16) + 8]), "+r"(dv1[(16) + 9]), "+r"(dv1[(16) + 10]), "+r"(dv1[(16) + 11]), "+r"(dv1[(16) + 12]), "+r"(dv1[(16) + 13]), "+r"(dv1[(16) + 14]), "+r"(dv1[(16) + 15])
            : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_6 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
            : "+r"(dv1[16]), "+r"(dv1[(16) + 1]), "+r"(dv1[(16) + 2]), "+r"(dv1[(16) + 3]), "+r"(dv1[(16) + 4]), "+r"(dv1[(16) + 5]), "+r"(dv1[(16) + 6]), "+r"(dv1[(16) + 7]), "+r"(dv1[(16) + 8]), "+r"(dv1[(16) + 9]), "+r"(dv1[(16) + 10]), "+r"(dv1[(16) + 11]), "+r"(dv1[(16) + 12]), "+r"(dv1[(16) + 13]), "+r"(dv1[(16) + 14]), "+r"(dv1[(16) + 15])
            : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_6 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        int limv[2];
        if ((bl & 1) != 0) {
            int r_c = 2 * cq;
            int lim_ = qpos1 + (r_c >> 3) - pbase;
            if (lim_ > 128) {
                lim_ = 128;
            }
            if (lim_ < 0) {
                lim_ = 0;
            }
            if ((mask_j >> (unsigned int)(r_c >> 3) & 1) == 0) {
                lim_ = 0;
            }
            limv[0] = lim_;
            tmax[0] = -CAKE_INF;
            int r_c_0 = 2 * cq + 1;
            int lim__1 = qpos1 + (r_c_0 >> 3) - pbase;
            if (lim__1 > 128) {
                lim__1 = 128;
            }
            if (lim__1 < 0) {
                lim__1 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_0 >> 3) & 1) == 0) {
                lim__1 = 0;
            }
            limv[1] = lim__1;
            tmax[1] = -CAKE_INF;
            float v0 = -CAKE_INF;
            if (key0 < limv[0]) {
                v0 = d_s0[0] * softmax_scale_log2;
            }
            d_s0[0] = v0;
            float v1 = -CAKE_INF;
            if (limv[0] > 64 + key0) {
                v1 = d_s1[0] * softmax_scale_log2;
            }
            d_s1[0] = v1;
            float _max_0 = max_noftz(v0, v1);
            float _max_1 = max_noftz(tmax[0], _max_0);
            tmax[0] = _max_1;
            float v0_2 = -CAKE_INF;
            if (key0 < limv[1]) {
                v0_2 = d_s0[1] * softmax_scale_log2;
            }
            d_s0[1] = v0_2;
            float v1_3 = -CAKE_INF;
            if (limv[1] > 64 + key0) {
                v1_3 = d_s1[1] * softmax_scale_log2;
            }
            d_s1[1] = v1_3;
            float _max_2 = max_noftz(v0_2, v1_3);
            float _max_3 = max_noftz(tmax[1], _max_2);
            tmax[1] = _max_3;
            float v0_4 = -CAKE_INF;
            if (key1 < limv[0]) {
                v0_4 = d_s0[2] * softmax_scale_log2;
            }
            d_s0[2] = v0_4;
            float v1_5 = -CAKE_INF;
            if (limv[0] > 64 + key1) {
                v1_5 = d_s1[2] * softmax_scale_log2;
            }
            d_s1[2] = v1_5;
            float _max_4 = max_noftz(v0_4, v1_5);
            float _max_5 = max_noftz(tmax[0], _max_4);
            tmax[0] = _max_5;
            float v0_6 = -CAKE_INF;
            if (key1 < limv[1]) {
                v0_6 = d_s0[3] * softmax_scale_log2;
            }
            d_s0[3] = v0_6;
            float v1_7 = -CAKE_INF;
            if (limv[1] > 64 + key1) {
                v1_7 = d_s1[3] * softmax_scale_log2;
            }
            d_s1[3] = v1_7;
            float _max_6 = max_noftz(v0_6, v1_7);
            float _max_7 = max_noftz(tmax[1], _max_6);
            tmax[1] = _max_7;
            float t = tmax[0];
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, t, 4);
            float _max_8 = max_noftz(t, _shfl_xor_0);
            t = _max_8;
            float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, t, 8);
            float _max_9 = max_noftz(t, _shfl_xor_1);
            t = _max_9;
            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, t, 16);
            float _max_10 = max_noftz(t, _shfl_xor_2);
            t = _max_10;
            tmax[0] = t;
            float t_8 = tmax[1];
            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, t_8, 4);
            float _max_11 = max_noftz(t_8, _shfl_xor_3);
            t_8 = _max_11;
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, t_8, 8);
            float _max_12 = max_noftz(t_8, _shfl_xor_4);
            t_8 = _max_12;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, t_8, 16);
            float _max_13 = max_noftz(t_8, _shfl_xor_5);
            t_8 = _max_13;
            tmax[1] = t_8;
            if (lane_0 < 4) {
                int hh = 2 * cq;
                red[hh * 4 + warp_1] = tmax[0];
                int hh_0 = 2 * cq + 1;
                red[hh_0 * 4 + warp_1] = tmax[1];
            }
        }
        if ((bl >> 1 & 1) != 0) {
            int r_c_1 = 8 + 2 * cq;
            int lim__2 = qpos1 + (r_c_1 >> 3) - pbase;
            if (lim__2 > 128) {
                lim__2 = 128;
            }
            if (lim__2 < 0) {
                lim__2 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_1 >> 3) & 1) == 0) {
                lim__2 = 0;
            }
            limv[0] = lim__2;
            tmax[2] = -CAKE_INF;
            int r_c_0_1 = 8 + 2 * cq + 1;
            int lim__1_1 = qpos1 + (r_c_0_1 >> 3) - pbase;
            if (lim__1_1 > 128) {
                lim__1_1 = 128;
            }
            if (lim__1_1 < 0) {
                lim__1_1 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_0_1 >> 3) & 1) == 0) {
                lim__1_1 = 0;
            }
            limv[1] = lim__1_1;
            tmax[3] = -CAKE_INF;
            float v0_1 = -CAKE_INF;
            if (key0 < limv[0]) {
                v0_1 = d_s0[4] * softmax_scale_log2;
            }
            d_s0[4] = v0_1;
            float v1_1 = -CAKE_INF;
            if (limv[0] > 64 + key0) {
                v1_1 = d_s1[4] * softmax_scale_log2;
            }
            d_s1[4] = v1_1;
            float _max_14 = max_noftz(v0_1, v1_1);
            float _max_15 = max_noftz(tmax[2], _max_14);
            tmax[2] = _max_15;
            float v0_2_1 = -CAKE_INF;
            if (key0 < limv[1]) {
                v0_2_1 = d_s0[5] * softmax_scale_log2;
            }
            d_s0[5] = v0_2_1;
            float v1_3_1 = -CAKE_INF;
            if (limv[1] > 64 + key0) {
                v1_3_1 = d_s1[5] * softmax_scale_log2;
            }
            d_s1[5] = v1_3_1;
            float _max_16 = max_noftz(v0_2_1, v1_3_1);
            float _max_17 = max_noftz(tmax[3], _max_16);
            tmax[3] = _max_17;
            float v0_4_1 = -CAKE_INF;
            if (key1 < limv[0]) {
                v0_4_1 = d_s0[6] * softmax_scale_log2;
            }
            d_s0[6] = v0_4_1;
            float v1_5_1 = -CAKE_INF;
            if (limv[0] > 64 + key1) {
                v1_5_1 = d_s1[6] * softmax_scale_log2;
            }
            d_s1[6] = v1_5_1;
            float _max_18 = max_noftz(v0_4_1, v1_5_1);
            float _max_19 = max_noftz(tmax[2], _max_18);
            tmax[2] = _max_19;
            float v0_6_1 = -CAKE_INF;
            if (key1 < limv[1]) {
                v0_6_1 = d_s0[7] * softmax_scale_log2;
            }
            d_s0[7] = v0_6_1;
            float v1_7_1 = -CAKE_INF;
            if (limv[1] > 64 + key1) {
                v1_7_1 = d_s1[7] * softmax_scale_log2;
            }
            d_s1[7] = v1_7_1;
            float _max_20 = max_noftz(v0_6_1, v1_7_1);
            float _max_21 = max_noftz(tmax[3], _max_20);
            tmax[3] = _max_21;
            float t_1 = tmax[2];
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, t_1, 4);
            float _max_22 = max_noftz(t_1, _shfl_xor_6);
            t_1 = _max_22;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, t_1, 8);
            float _max_23 = max_noftz(t_1, _shfl_xor_7);
            t_1 = _max_23;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, t_1, 16);
            float _max_24 = max_noftz(t_1, _shfl_xor_8);
            t_1 = _max_24;
            tmax[2] = t_1;
            float t_8_1 = tmax[3];
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 4);
            float _max_25 = max_noftz(t_8_1, _shfl_xor_9);
            t_8_1 = _max_25;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 8);
            float _max_26 = max_noftz(t_8_1, _shfl_xor_10);
            t_8_1 = _max_26;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, t_8_1, 16);
            float _max_27 = max_noftz(t_8_1, _shfl_xor_11);
            t_8_1 = _max_27;
            tmax[3] = t_8_1;
            if (lane_0 < 4) {
                int hh_1 = 8 + 2 * cq;
                red[hh_1 * 4 + warp_1] = tmax[2];
                int hh_0_1 = 8 + 2 * cq + 1;
                red[hh_0_1 * 4 + warp_1] = tmax[3];
            }
        }
        if ((bl >> 2 & 1) != 0) {
            int r_c_2 = 16 + 2 * cq;
            int lim__3 = qpos1 + (r_c_2 >> 3) - pbase;
            if (lim__3 > 128) {
                lim__3 = 128;
            }
            if (lim__3 < 0) {
                lim__3 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_2 >> 3) & 1) == 0) {
                lim__3 = 0;
            }
            limv[0] = lim__3;
            tmax[4] = -CAKE_INF;
            int r_c_0_2 = 16 + 2 * cq + 1;
            int lim__1_2 = qpos1 + (r_c_0_2 >> 3) - pbase;
            if (lim__1_2 > 128) {
                lim__1_2 = 128;
            }
            if (lim__1_2 < 0) {
                lim__1_2 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_0_2 >> 3) & 1) == 0) {
                lim__1_2 = 0;
            }
            limv[1] = lim__1_2;
            tmax[5] = -CAKE_INF;
            float v0_3 = -CAKE_INF;
            if (key0 < limv[0]) {
                v0_3 = d_s0[8] * softmax_scale_log2;
            }
            d_s0[8] = v0_3;
            float v1_2 = -CAKE_INF;
            if (limv[0] > 64 + key0) {
                v1_2 = d_s1[8] * softmax_scale_log2;
            }
            d_s1[8] = v1_2;
            float _max_28 = max_noftz(v0_3, v1_2);
            float _max_29 = max_noftz(tmax[4], _max_28);
            tmax[4] = _max_29;
            float v0_2_2 = -CAKE_INF;
            if (key0 < limv[1]) {
                v0_2_2 = d_s0[9] * softmax_scale_log2;
            }
            d_s0[9] = v0_2_2;
            float v1_3_2 = -CAKE_INF;
            if (limv[1] > 64 + key0) {
                v1_3_2 = d_s1[9] * softmax_scale_log2;
            }
            d_s1[9] = v1_3_2;
            float _max_30 = max_noftz(v0_2_2, v1_3_2);
            float _max_31 = max_noftz(tmax[5], _max_30);
            tmax[5] = _max_31;
            float v0_4_2 = -CAKE_INF;
            if (key1 < limv[0]) {
                v0_4_2 = d_s0[10] * softmax_scale_log2;
            }
            d_s0[10] = v0_4_2;
            float v1_5_2 = -CAKE_INF;
            if (limv[0] > 64 + key1) {
                v1_5_2 = d_s1[10] * softmax_scale_log2;
            }
            d_s1[10] = v1_5_2;
            float _max_32 = max_noftz(v0_4_2, v1_5_2);
            float _max_33 = max_noftz(tmax[4], _max_32);
            tmax[4] = _max_33;
            float v0_6_2 = -CAKE_INF;
            if (key1 < limv[1]) {
                v0_6_2 = d_s0[11] * softmax_scale_log2;
            }
            d_s0[11] = v0_6_2;
            float v1_7_2 = -CAKE_INF;
            if (limv[1] > 64 + key1) {
                v1_7_2 = d_s1[11] * softmax_scale_log2;
            }
            d_s1[11] = v1_7_2;
            float _max_34 = max_noftz(v0_6_2, v1_7_2);
            float _max_35 = max_noftz(tmax[5], _max_34);
            tmax[5] = _max_35;
            float t_2 = tmax[4];
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, t_2, 4);
            float _max_36 = max_noftz(t_2, _shfl_xor_12);
            t_2 = _max_36;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, t_2, 8);
            float _max_37 = max_noftz(t_2, _shfl_xor_13);
            t_2 = _max_37;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, t_2, 16);
            float _max_38 = max_noftz(t_2, _shfl_xor_14);
            t_2 = _max_38;
            tmax[4] = t_2;
            float t_8_2 = tmax[5];
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 4);
            float _max_39 = max_noftz(t_8_2, _shfl_xor_15);
            t_8_2 = _max_39;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 8);
            float _max_40 = max_noftz(t_8_2, _shfl_xor_16);
            t_8_2 = _max_40;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, t_8_2, 16);
            float _max_41 = max_noftz(t_8_2, _shfl_xor_17);
            t_8_2 = _max_41;
            tmax[5] = t_8_2;
            if (lane_0 < 4) {
                int hh_2 = 16 + 2 * cq;
                red[hh_2 * 4 + warp_1] = tmax[4];
                int hh_0_2 = 16 + 2 * cq + 1;
                red[hh_0_2 * 4 + warp_1] = tmax[5];
            }
        }
        if ((bl >> 3 & 1) != 0) {
            int r_c_3 = 24 + 2 * cq;
            int lim__4 = qpos1 + (r_c_3 >> 3) - pbase;
            if (lim__4 > 128) {
                lim__4 = 128;
            }
            if (lim__4 < 0) {
                lim__4 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_3 >> 3) & 1) == 0) {
                lim__4 = 0;
            }
            limv[0] = lim__4;
            tmax[6] = -CAKE_INF;
            int r_c_0_3 = 24 + 2 * cq + 1;
            int lim__1_3 = qpos1 + (r_c_0_3 >> 3) - pbase;
            if (lim__1_3 > 128) {
                lim__1_3 = 128;
            }
            if (lim__1_3 < 0) {
                lim__1_3 = 0;
            }
            if ((mask_j >> (unsigned int)(r_c_0_3 >> 3) & 1) == 0) {
                lim__1_3 = 0;
            }
            limv[1] = lim__1_3;
            tmax[7] = -CAKE_INF;
            float v0_5 = -CAKE_INF;
            if (key0 < limv[0]) {
                v0_5 = d_s0[12] * softmax_scale_log2;
            }
            d_s0[12] = v0_5;
            float v1_4 = -CAKE_INF;
            if (limv[0] > 64 + key0) {
                v1_4 = d_s1[12] * softmax_scale_log2;
            }
            d_s1[12] = v1_4;
            float _max_42 = max_noftz(v0_5, v1_4);
            float _max_43 = max_noftz(tmax[6], _max_42);
            tmax[6] = _max_43;
            float v0_2_3 = -CAKE_INF;
            if (key0 < limv[1]) {
                v0_2_3 = d_s0[13] * softmax_scale_log2;
            }
            d_s0[13] = v0_2_3;
            float v1_3_3 = -CAKE_INF;
            if (limv[1] > 64 + key0) {
                v1_3_3 = d_s1[13] * softmax_scale_log2;
            }
            d_s1[13] = v1_3_3;
            float _max_44 = max_noftz(v0_2_3, v1_3_3);
            float _max_45 = max_noftz(tmax[7], _max_44);
            tmax[7] = _max_45;
            float v0_4_3 = -CAKE_INF;
            if (key1 < limv[0]) {
                v0_4_3 = d_s0[14] * softmax_scale_log2;
            }
            d_s0[14] = v0_4_3;
            float v1_5_3 = -CAKE_INF;
            if (limv[0] > 64 + key1) {
                v1_5_3 = d_s1[14] * softmax_scale_log2;
            }
            d_s1[14] = v1_5_3;
            float _max_46 = max_noftz(v0_4_3, v1_5_3);
            float _max_47 = max_noftz(tmax[6], _max_46);
            tmax[6] = _max_47;
            float v0_6_3 = -CAKE_INF;
            if (key1 < limv[1]) {
                v0_6_3 = d_s0[15] * softmax_scale_log2;
            }
            d_s0[15] = v0_6_3;
            float v1_7_3 = -CAKE_INF;
            if (limv[1] > 64 + key1) {
                v1_7_3 = d_s1[15] * softmax_scale_log2;
            }
            d_s1[15] = v1_7_3;
            float _max_48 = max_noftz(v0_6_3, v1_7_3);
            float _max_49 = max_noftz(tmax[7], _max_48);
            tmax[7] = _max_49;
            float t_3 = tmax[6];
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, t_3, 4);
            float _max_50 = max_noftz(t_3, _shfl_xor_18);
            t_3 = _max_50;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, t_3, 8);
            float _max_51 = max_noftz(t_3, _shfl_xor_19);
            t_3 = _max_51;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, t_3, 16);
            float _max_52 = max_noftz(t_3, _shfl_xor_20);
            t_3 = _max_52;
            tmax[6] = t_3;
            float t_8_3 = tmax[7];
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 4);
            float _max_53 = max_noftz(t_8_3, _shfl_xor_21);
            t_8_3 = _max_53;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 8);
            float _max_54 = max_noftz(t_8_3, _shfl_xor_22);
            t_8_3 = _max_54;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, t_8_3, 16);
            float _max_55 = max_noftz(t_8_3, _shfl_xor_23);
            t_8_3 = _max_55;
            tmax[7] = t_8_3;
            if (lane_0 < 4) {
                int hh_3 = 24 + 2 * cq;
                red[hh_3 * 4 + warp_1] = tmax[6];
                int hh_0_3 = 24 + 2 * cq + 1;
                red[hh_0_3 * 4 + warp_1] = tmax[7];
            }
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if ((bl & 1) != 0) {
            int hh2 = 2 * cq;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2 * 4 * 4)));
            float _max_56 = max_noftz(rv[0], rv[1]);
            float _max_57 = max_noftz(rv[2], rv[3]);
            float _max_58 = max_noftz(_max_56, _max_57);
            float page_max = _max_58;
            float _max_59 = max_noftz(mrun[0], page_max);
            float mnew = _max_59;
            float corr = 0.0f;
            if (mrun[0] != -CAKE_INF) {
                float _exp2_0 = approx_exp2(mrun[0] - mnew);
                corr = _exp2_0;
            }
            d_o0[0] = d_o0[0] * corr;
            d_o0[2] = d_o0[2] * corr;
            d_o1[0] = d_o1[0] * corr;
            d_o1[2] = d_o1[2] * corr;
            lp[0] = lp[0] * corr;
            mrun[0] = mnew;
            int hh2_0 = 2 * cq + 1;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_0 * 4 * 4)));
            float _max_60 = max_noftz(rv[0], rv[1]);
            float _max_61 = max_noftz(rv[2], rv[3]);
            float _max_62 = max_noftz(_max_60, _max_61);
            float page_max_1 = _max_62;
            float _max_63 = max_noftz(mrun[1], page_max_1);
            float mnew_2 = _max_63;
            float corr_3 = 0.0f;
            if (mrun[1] != -CAKE_INF) {
                float _exp2_1 = approx_exp2(mrun[1] - mnew_2);
                corr_3 = _exp2_1;
            }
            d_o0[1] = d_o0[1] * corr_3;
            d_o0[3] = d_o0[3] * corr_3;
            d_o1[1] = d_o1[1] * corr_3;
            d_o1[3] = d_o1[3] * corr_3;
            lp[1] = lp[1] * corr_3;
            mrun[1] = mnew_2;
        }
        if ((bl >> 1 & 1) != 0) {
            int hh2_1 = 8 + 2 * cq;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_1 * 4 * 4)));
            float _max_64 = max_noftz(rv[0], rv[1]);
            float _max_65 = max_noftz(rv[2], rv[3]);
            float _max_66 = max_noftz(_max_64, _max_65);
            float page_max_2 = _max_66;
            float _max_67 = max_noftz(mrun[2], page_max_2);
            float mnew_1 = _max_67;
            float corr_1 = 0.0f;
            if (mrun[2] != -CAKE_INF) {
                float _exp2_2 = approx_exp2(mrun[2] - mnew_1);
                corr_1 = _exp2_2;
            }
            d_o0[4] = d_o0[4] * corr_1;
            d_o0[6] = d_o0[6] * corr_1;
            d_o1[4] = d_o1[4] * corr_1;
            d_o1[6] = d_o1[6] * corr_1;
            lp[2] = lp[2] * corr_1;
            mrun[2] = mnew_1;
            int hh2_0_1 = 8 + 2 * cq + 1;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_0_1 * 4 * 4)));
            float _max_68 = max_noftz(rv[0], rv[1]);
            float _max_69 = max_noftz(rv[2], rv[3]);
            float _max_70 = max_noftz(_max_68, _max_69);
            float page_max_1_1 = _max_70;
            float _max_71 = max_noftz(mrun[3], page_max_1_1);
            float mnew_2_1 = _max_71;
            float corr_3_1 = 0.0f;
            if (mrun[3] != -CAKE_INF) {
                float _exp2_3 = approx_exp2(mrun[3] - mnew_2_1);
                corr_3_1 = _exp2_3;
            }
            d_o0[5] = d_o0[5] * corr_3_1;
            d_o0[7] = d_o0[7] * corr_3_1;
            d_o1[5] = d_o1[5] * corr_3_1;
            d_o1[7] = d_o1[7] * corr_3_1;
            lp[3] = lp[3] * corr_3_1;
            mrun[3] = mnew_2_1;
        }
        if ((bl >> 2 & 1) != 0) {
            int hh2_2 = 16 + 2 * cq;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_2 * 4 * 4)));
            float _max_72 = max_noftz(rv[0], rv[1]);
            float _max_73 = max_noftz(rv[2], rv[3]);
            float _max_74 = max_noftz(_max_72, _max_73);
            float page_max_3 = _max_74;
            float _max_75 = max_noftz(mrun[4], page_max_3);
            float mnew_3 = _max_75;
            float corr_2 = 0.0f;
            if (mrun[4] != -CAKE_INF) {
                float _exp2_4 = approx_exp2(mrun[4] - mnew_3);
                corr_2 = _exp2_4;
            }
            d_o0[8] = d_o0[8] * corr_2;
            d_o0[10] = d_o0[10] * corr_2;
            d_o1[8] = d_o1[8] * corr_2;
            d_o1[10] = d_o1[10] * corr_2;
            lp[4] = lp[4] * corr_2;
            mrun[4] = mnew_3;
            int hh2_0_2 = 16 + 2 * cq + 1;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_0_2 * 4 * 4)));
            float _max_76 = max_noftz(rv[0], rv[1]);
            float _max_77 = max_noftz(rv[2], rv[3]);
            float _max_78 = max_noftz(_max_76, _max_77);
            float page_max_1_2 = _max_78;
            float _max_79 = max_noftz(mrun[5], page_max_1_2);
            float mnew_2_2 = _max_79;
            float corr_3_2 = 0.0f;
            if (mrun[5] != -CAKE_INF) {
                float _exp2_5 = approx_exp2(mrun[5] - mnew_2_2);
                corr_3_2 = _exp2_5;
            }
            d_o0[9] = d_o0[9] * corr_3_2;
            d_o0[11] = d_o0[11] * corr_3_2;
            d_o1[9] = d_o1[9] * corr_3_2;
            d_o1[11] = d_o1[11] * corr_3_2;
            lp[5] = lp[5] * corr_3_2;
            mrun[5] = mnew_2_2;
        }
        if ((bl >> 3 & 1) != 0) {
            int hh2_3 = 24 + 2 * cq;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_3 * 4 * 4)));
            float _max_80 = max_noftz(rv[0], rv[1]);
            float _max_81 = max_noftz(rv[2], rv[3]);
            float _max_82 = max_noftz(_max_80, _max_81);
            float page_max_4 = _max_82;
            float _max_83 = max_noftz(mrun[6], page_max_4);
            float mnew_4 = _max_83;
            float corr_4 = 0.0f;
            if (mrun[6] != -CAKE_INF) {
                float _exp2_6 = approx_exp2(mrun[6] - mnew_4);
                corr_4 = _exp2_6;
            }
            d_o0[12] = d_o0[12] * corr_4;
            d_o0[14] = d_o0[14] * corr_4;
            d_o1[12] = d_o1[12] * corr_4;
            d_o1[14] = d_o1[14] * corr_4;
            lp[6] = lp[6] * corr_4;
            mrun[6] = mnew_4;
            int hh2_0_3 = 24 + 2 * cq + 1;
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
                : "r"(red_addr + (unsigned int)(hh2_0_3 * 4 * 4)));
            float _max_84 = max_noftz(rv[0], rv[1]);
            float _max_85 = max_noftz(rv[2], rv[3]);
            float _max_86 = max_noftz(_max_84, _max_85);
            float page_max_1_3 = _max_86;
            float _max_87 = max_noftz(mrun[7], page_max_1_3);
            float mnew_2_3 = _max_87;
            float corr_3_3 = 0.0f;
            if (mrun[7] != -CAKE_INF) {
                float _exp2_7 = approx_exp2(mrun[7] - mnew_2_3);
                corr_3_3 = _exp2_7;
            }
            d_o0[13] = d_o0[13] * corr_3_3;
            d_o0[15] = d_o0[15] * corr_3_3;
            d_o1[13] = d_o1[13] * corr_3_3;
            d_o1[15] = d_o1[15] * corr_3_3;
            lp[7] = lp[7] * corr_3_3;
            mrun[7] = mnew_2_3;
        }
        if ((bl & 1) != 0) {
            int hh3 = 2 * cq;
            int kk = key0;
            int pcol = hh3 * 128 + (kk * 2 ^ (hh3 & 7) << 4);
            float msub = mrun[0];
            if (msub == -CAKE_INF) {
                msub = 0.0f;
            }
            float _exp2_8 = approx_exp2(d_s0[0] - msub);
            float p0 = _exp2_8;
            float _exp2_9 = approx_exp2(d_s1[0] - msub);
            float p1 = _exp2_9;
            lp[0] = lp[0] + (p0 + p1);
            {
                __half _hval_10 = __float2half_rn(p0);
                uint16_t _bits_10 = *(uint16_t*)&_hval_10;
                uint32_t _addr_10 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_10), "h"(_bits_10) : "memory");
            }
            {
                __half _hval_11 = __float2half_rn(p1);
                uint16_t _bits_11 = *(uint16_t*)&_hval_11;
                uint32_t _addr_11 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_11), "h"(_bits_11) : "memory");
            }
            int hh3_0 = 2 * cq + 1;
            int kk_1 = key0;
            int pcol_2 = hh3_0 * 128 + (kk_1 * 2 ^ (hh3_0 & 7) << 4);
            float msub_3 = mrun[1];
            if (msub_3 == -CAKE_INF) {
                msub_3 = 0.0f;
            }
            float _exp2_10 = approx_exp2(d_s0[1] - msub_3);
            float p0_4 = _exp2_10;
            float _exp2_11 = approx_exp2(d_s1[1] - msub_3);
            float p1_5 = _exp2_11;
            lp[1] = lp[1] + (p0_4 + p1_5);
            {
                __half _hval_12 = __float2half_rn(p0_4);
                uint16_t _bits_12 = *(uint16_t*)&_hval_12;
                uint32_t _addr_12 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_2);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_12), "h"(_bits_12) : "memory");
            }
            {
                __half _hval_13 = __float2half_rn(p1_5);
                uint16_t _bits_13 = *(uint16_t*)&_hval_13;
                uint32_t _addr_13 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_13), "h"(_bits_13) : "memory");
            }
            int hh3_6 = 2 * cq;
            int kk_7 = key1;
            int pcol_8 = hh3_6 * 128 + (kk_7 * 2 ^ (hh3_6 & 7) << 4);
            float msub_9 = mrun[0];
            if (msub_9 == -CAKE_INF) {
                msub_9 = 0.0f;
            }
            float _exp2_12 = approx_exp2(d_s0[2] - msub_9);
            float p0_10 = _exp2_12;
            float _exp2_13 = approx_exp2(d_s1[2] - msub_9);
            float p1_11 = _exp2_13;
            lp[0] = lp[0] + (p0_10 + p1_11);
            {
                __half _hval_14 = __float2half_rn(p0_10);
                uint16_t _bits_14 = *(uint16_t*)&_hval_14;
                uint32_t _addr_14 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_8);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
            }
            {
                __half _hval_15 = __float2half_rn(p1_11);
                uint16_t _bits_15 = *(uint16_t*)&_hval_15;
                uint32_t _addr_15 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_8));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_15), "h"(_bits_15) : "memory");
            }
            int hh3_12 = 2 * cq + 1;
            int kk_13 = key1;
            int pcol_14 = hh3_12 * 128 + (kk_13 * 2 ^ (hh3_12 & 7) << 4);
            float msub_15 = mrun[1];
            if (msub_15 == -CAKE_INF) {
                msub_15 = 0.0f;
            }
            float _exp2_14 = approx_exp2(d_s0[3] - msub_15);
            float p0_16 = _exp2_14;
            float _exp2_15 = approx_exp2(d_s1[3] - msub_15);
            float p1_17 = _exp2_15;
            lp[1] = lp[1] + (p0_16 + p1_17);
            {
                __half _hval_16 = __float2half_rn(p0_16);
                uint16_t _bits_16 = *(uint16_t*)&_hval_16;
                uint32_t _addr_16 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_14);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_16), "h"(_bits_16) : "memory");
            }
            {
                __half _hval_17 = __float2half_rn(p1_17);
                uint16_t _bits_17 = *(uint16_t*)&_hval_17;
                uint32_t _addr_17 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_14));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_17), "h"(_bits_17) : "memory");
            }
        }
        if ((bl & 1) == 0) {
            if ((prev_bl & 1) != 0) {
                int hh3z = 2 * cq;
                int kkz = key0;
                int pcolz = hh3z * 128 + (kkz * 2 ^ (hh3z & 7) << 4);
                {
                    __half _hval_18 = __float2half_rn(0.0f);
                    uint16_t _bits_18 = *(uint16_t*)&_hval_18;
                    uint32_t _addr_18 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_18), "h"(_bits_18) : "memory");
                }
                {
                    __half _hval_19 = __float2half_rn(0.0f);
                    uint16_t _bits_19 = *(uint16_t*)&_hval_19;
                    uint32_t _addr_19 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_19), "h"(_bits_19) : "memory");
                }
                int hh3z_0 = 2 * cq + 1;
                int kkz_1 = key0;
                int pcolz_2 = hh3z_0 * 128 + (kkz_1 * 2 ^ (hh3z_0 & 7) << 4);
                {
                    __half _hval_20 = __float2half_rn(0.0f);
                    uint16_t _bits_20 = *(uint16_t*)&_hval_20;
                    uint32_t _addr_20 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_2);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_20), "h"(_bits_20) : "memory");
                }
                {
                    __half _hval_21 = __float2half_rn(0.0f);
                    uint16_t _bits_21 = *(uint16_t*)&_hval_21;
                    uint32_t _addr_21 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_21), "h"(_bits_21) : "memory");
                }
                int hh3z_3 = 2 * cq;
                int kkz_4 = key1;
                int pcolz_5 = hh3z_3 * 128 + (kkz_4 * 2 ^ (hh3z_3 & 7) << 4);
                {
                    __half _hval_22 = __float2half_rn(0.0f);
                    uint16_t _bits_22 = *(uint16_t*)&_hval_22;
                    uint32_t _addr_22 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_22), "h"(_bits_22) : "memory");
                }
                {
                    __half _hval_23 = __float2half_rn(0.0f);
                    uint16_t _bits_23 = *(uint16_t*)&_hval_23;
                    uint32_t _addr_23 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_23), "h"(_bits_23) : "memory");
                }
                int hh3z_6 = 2 * cq + 1;
                int kkz_7 = key1;
                int pcolz_8 = hh3z_6 * 128 + (kkz_7 * 2 ^ (hh3z_6 & 7) << 4);
                {
                    __half _hval_24 = __float2half_rn(0.0f);
                    uint16_t _bits_24 = *(uint16_t*)&_hval_24;
                    uint32_t _addr_24 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_8);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_24), "h"(_bits_24) : "memory");
                }
                {
                    __half _hval_25 = __float2half_rn(0.0f);
                    uint16_t _bits_25 = *(uint16_t*)&_hval_25;
                    uint32_t _addr_25 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_8));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_25), "h"(_bits_25) : "memory");
                }
            }
        }
        if ((bl >> 1 & 1) != 0) {
            int hh3_1 = 8 + 2 * cq;
            int kk_2 = key0;
            int pcol_1 = hh3_1 * 128 + (kk_2 * 2 ^ (hh3_1 & 7) << 4);
            float msub_1 = mrun[2];
            if (msub_1 == -CAKE_INF) {
                msub_1 = 0.0f;
            }
            float _exp2_16 = approx_exp2(d_s0[4] - msub_1);
            float p0_1 = _exp2_16;
            float _exp2_17 = approx_exp2(d_s1[4] - msub_1);
            float p1_1 = _exp2_17;
            lp[2] = lp[2] + (p0_1 + p1_1);
            {
                __half _hval_26 = __float2half_rn(p0_1);
                uint16_t _bits_26 = *(uint16_t*)&_hval_26;
                uint32_t _addr_26 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_1);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_26), "h"(_bits_26) : "memory");
            }
            {
                __half _hval_27 = __float2half_rn(p1_1);
                uint16_t _bits_27 = *(uint16_t*)&_hval_27;
                uint32_t _addr_27 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_1));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_27), "h"(_bits_27) : "memory");
            }
            int hh3_0_1 = 8 + 2 * cq + 1;
            int kk_1_1 = key0;
            int pcol_2_1 = hh3_0_1 * 128 + (kk_1_1 * 2 ^ (hh3_0_1 & 7) << 4);
            float msub_3_1 = mrun[3];
            if (msub_3_1 == -CAKE_INF) {
                msub_3_1 = 0.0f;
            }
            float _exp2_18 = approx_exp2(d_s0[5] - msub_3_1);
            float p0_4_1 = _exp2_18;
            float _exp2_19 = approx_exp2(d_s1[5] - msub_3_1);
            float p1_5_1 = _exp2_19;
            lp[3] = lp[3] + (p0_4_1 + p1_5_1);
            {
                __half _hval_28 = __float2half_rn(p0_4_1);
                uint16_t _bits_28 = *(uint16_t*)&_hval_28;
                uint32_t _addr_28 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_2_1);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_28), "h"(_bits_28) : "memory");
            }
            {
                __half _hval_29 = __float2half_rn(p1_5_1);
                uint16_t _bits_29 = *(uint16_t*)&_hval_29;
                uint32_t _addr_29 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_2_1));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_29), "h"(_bits_29) : "memory");
            }
            int hh3_6_1 = 8 + 2 * cq;
            int kk_7_1 = key1;
            int pcol_8_1 = hh3_6_1 * 128 + (kk_7_1 * 2 ^ (hh3_6_1 & 7) << 4);
            float msub_9_1 = mrun[2];
            if (msub_9_1 == -CAKE_INF) {
                msub_9_1 = 0.0f;
            }
            float _exp2_20 = approx_exp2(d_s0[6] - msub_9_1);
            float p0_10_1 = _exp2_20;
            float _exp2_21 = approx_exp2(d_s1[6] - msub_9_1);
            float p1_11_1 = _exp2_21;
            lp[2] = lp[2] + (p0_10_1 + p1_11_1);
            {
                __half _hval_30 = __float2half_rn(p0_10_1);
                uint16_t _bits_30 = *(uint16_t*)&_hval_30;
                uint32_t _addr_30 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_8_1);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_30), "h"(_bits_30) : "memory");
            }
            {
                __half _hval_31 = __float2half_rn(p1_11_1);
                uint16_t _bits_31 = *(uint16_t*)&_hval_31;
                uint32_t _addr_31 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_8_1));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_31), "h"(_bits_31) : "memory");
            }
            int hh3_12_1 = 8 + 2 * cq + 1;
            int kk_13_1 = key1;
            int pcol_14_1 = hh3_12_1 * 128 + (kk_13_1 * 2 ^ (hh3_12_1 & 7) << 4);
            float msub_15_1 = mrun[3];
            if (msub_15_1 == -CAKE_INF) {
                msub_15_1 = 0.0f;
            }
            float _exp2_22 = approx_exp2(d_s0[7] - msub_15_1);
            float p0_16_1 = _exp2_22;
            float _exp2_23 = approx_exp2(d_s1[7] - msub_15_1);
            float p1_17_1 = _exp2_23;
            lp[3] = lp[3] + (p0_16_1 + p1_17_1);
            {
                __half _hval_32 = __float2half_rn(p0_16_1);
                uint16_t _bits_32 = *(uint16_t*)&_hval_32;
                uint32_t _addr_32 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_14_1);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_32), "h"(_bits_32) : "memory");
            }
            {
                __half _hval_33 = __float2half_rn(p1_17_1);
                uint16_t _bits_33 = *(uint16_t*)&_hval_33;
                uint32_t _addr_33 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_14_1));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_33), "h"(_bits_33) : "memory");
            }
        }
        if ((bl >> 1 & 1) == 0) {
            if ((prev_bl >> 1 & 1) != 0) {
                int hh3z_1 = 8 + 2 * cq;
                int kkz_2 = key0;
                int pcolz_1 = hh3z_1 * 128 + (kkz_2 * 2 ^ (hh3z_1 & 7) << 4);
                {
                    __half _hval_34 = __float2half_rn(0.0f);
                    uint16_t _bits_34 = *(uint16_t*)&_hval_34;
                    uint32_t _addr_34 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_1);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_34), "h"(_bits_34) : "memory");
                }
                {
                    __half _hval_35 = __float2half_rn(0.0f);
                    uint16_t _bits_35 = *(uint16_t*)&_hval_35;
                    uint32_t _addr_35 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_1));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_35), "h"(_bits_35) : "memory");
                }
                int hh3z_0_1 = 8 + 2 * cq + 1;
                int kkz_1_1 = key0;
                int pcolz_2_1 = hh3z_0_1 * 128 + (kkz_1_1 * 2 ^ (hh3z_0_1 & 7) << 4);
                {
                    __half _hval_36 = __float2half_rn(0.0f);
                    uint16_t _bits_36 = *(uint16_t*)&_hval_36;
                    uint32_t _addr_36 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_2_1);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_36), "h"(_bits_36) : "memory");
                }
                {
                    __half _hval_37 = __float2half_rn(0.0f);
                    uint16_t _bits_37 = *(uint16_t*)&_hval_37;
                    uint32_t _addr_37 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_2_1));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_37), "h"(_bits_37) : "memory");
                }
                int hh3z_3_1 = 8 + 2 * cq;
                int kkz_4_1 = key1;
                int pcolz_5_1 = hh3z_3_1 * 128 + (kkz_4_1 * 2 ^ (hh3z_3_1 & 7) << 4);
                {
                    __half _hval_38 = __float2half_rn(0.0f);
                    uint16_t _bits_38 = *(uint16_t*)&_hval_38;
                    uint32_t _addr_38 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_1);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_38), "h"(_bits_38) : "memory");
                }
                {
                    __half _hval_39 = __float2half_rn(0.0f);
                    uint16_t _bits_39 = *(uint16_t*)&_hval_39;
                    uint32_t _addr_39 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_1));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_39), "h"(_bits_39) : "memory");
                }
                int hh3z_6_1 = 8 + 2 * cq + 1;
                int kkz_7_1 = key1;
                int pcolz_8_1 = hh3z_6_1 * 128 + (kkz_7_1 * 2 ^ (hh3z_6_1 & 7) << 4);
                {
                    __half _hval_40 = __float2half_rn(0.0f);
                    uint16_t _bits_40 = *(uint16_t*)&_hval_40;
                    uint32_t _addr_40 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_8_1);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_40), "h"(_bits_40) : "memory");
                }
                {
                    __half _hval_41 = __float2half_rn(0.0f);
                    uint16_t _bits_41 = *(uint16_t*)&_hval_41;
                    uint32_t _addr_41 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_8_1));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_41), "h"(_bits_41) : "memory");
                }
            }
        }
        if ((bl >> 2 & 1) != 0) {
            int hh3_2 = 16 + 2 * cq;
            int kk_3 = key0;
            int pcol_3 = hh3_2 * 128 + (kk_3 * 2 ^ (hh3_2 & 7) << 4);
            float msub_2 = mrun[4];
            if (msub_2 == -CAKE_INF) {
                msub_2 = 0.0f;
            }
            float _exp2_24 = approx_exp2(d_s0[8] - msub_2);
            float p0_2 = _exp2_24;
            float _exp2_25 = approx_exp2(d_s1[8] - msub_2);
            float p1_2 = _exp2_25;
            lp[4] = lp[4] + (p0_2 + p1_2);
            {
                __half _hval_42 = __float2half_rn(p0_2);
                uint16_t _bits_42 = *(uint16_t*)&_hval_42;
                uint32_t _addr_42 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_3);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_42), "h"(_bits_42) : "memory");
            }
            {
                __half _hval_43 = __float2half_rn(p1_2);
                uint16_t _bits_43 = *(uint16_t*)&_hval_43;
                uint32_t _addr_43 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_3));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_43), "h"(_bits_43) : "memory");
            }
            int hh3_0_2 = 16 + 2 * cq + 1;
            int kk_1_2 = key0;
            int pcol_2_2 = hh3_0_2 * 128 + (kk_1_2 * 2 ^ (hh3_0_2 & 7) << 4);
            float msub_3_2 = mrun[5];
            if (msub_3_2 == -CAKE_INF) {
                msub_3_2 = 0.0f;
            }
            float _exp2_26 = approx_exp2(d_s0[9] - msub_3_2);
            float p0_4_2 = _exp2_26;
            float _exp2_27 = approx_exp2(d_s1[9] - msub_3_2);
            float p1_5_2 = _exp2_27;
            lp[5] = lp[5] + (p0_4_2 + p1_5_2);
            {
                __half _hval_44 = __float2half_rn(p0_4_2);
                uint16_t _bits_44 = *(uint16_t*)&_hval_44;
                uint32_t _addr_44 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_2_2);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_44), "h"(_bits_44) : "memory");
            }
            {
                __half _hval_45 = __float2half_rn(p1_5_2);
                uint16_t _bits_45 = *(uint16_t*)&_hval_45;
                uint32_t _addr_45 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_2_2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_45), "h"(_bits_45) : "memory");
            }
            int hh3_6_2 = 16 + 2 * cq;
            int kk_7_2 = key1;
            int pcol_8_2 = hh3_6_2 * 128 + (kk_7_2 * 2 ^ (hh3_6_2 & 7) << 4);
            float msub_9_2 = mrun[4];
            if (msub_9_2 == -CAKE_INF) {
                msub_9_2 = 0.0f;
            }
            float _exp2_28 = approx_exp2(d_s0[10] - msub_9_2);
            float p0_10_2 = _exp2_28;
            float _exp2_29 = approx_exp2(d_s1[10] - msub_9_2);
            float p1_11_2 = _exp2_29;
            lp[4] = lp[4] + (p0_10_2 + p1_11_2);
            {
                __half _hval_46 = __float2half_rn(p0_10_2);
                uint16_t _bits_46 = *(uint16_t*)&_hval_46;
                uint32_t _addr_46 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_8_2);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_46), "h"(_bits_46) : "memory");
            }
            {
                __half _hval_47 = __float2half_rn(p1_11_2);
                uint16_t _bits_47 = *(uint16_t*)&_hval_47;
                uint32_t _addr_47 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_8_2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_47), "h"(_bits_47) : "memory");
            }
            int hh3_12_2 = 16 + 2 * cq + 1;
            int kk_13_2 = key1;
            int pcol_14_2 = hh3_12_2 * 128 + (kk_13_2 * 2 ^ (hh3_12_2 & 7) << 4);
            float msub_15_2 = mrun[5];
            if (msub_15_2 == -CAKE_INF) {
                msub_15_2 = 0.0f;
            }
            float _exp2_30 = approx_exp2(d_s0[11] - msub_15_2);
            float p0_16_2 = _exp2_30;
            float _exp2_31 = approx_exp2(d_s1[11] - msub_15_2);
            float p1_17_2 = _exp2_31;
            lp[5] = lp[5] + (p0_16_2 + p1_17_2);
            {
                __half _hval_48 = __float2half_rn(p0_16_2);
                uint16_t _bits_48 = *(uint16_t*)&_hval_48;
                uint32_t _addr_48 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_14_2);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_48), "h"(_bits_48) : "memory");
            }
            {
                __half _hval_49 = __float2half_rn(p1_17_2);
                uint16_t _bits_49 = *(uint16_t*)&_hval_49;
                uint32_t _addr_49 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_14_2));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_49), "h"(_bits_49) : "memory");
            }
        }
        if ((bl >> 2 & 1) == 0) {
            if ((prev_bl >> 2 & 1) != 0) {
                int hh3z_2 = 16 + 2 * cq;
                int kkz_3 = key0;
                int pcolz_3 = hh3z_2 * 128 + (kkz_3 * 2 ^ (hh3z_2 & 7) << 4);
                {
                    __half _hval_50 = __float2half_rn(0.0f);
                    uint16_t _bits_50 = *(uint16_t*)&_hval_50;
                    uint32_t _addr_50 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_3);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_50), "h"(_bits_50) : "memory");
                }
                {
                    __half _hval_51 = __float2half_rn(0.0f);
                    uint16_t _bits_51 = *(uint16_t*)&_hval_51;
                    uint32_t _addr_51 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_3));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_51), "h"(_bits_51) : "memory");
                }
                int hh3z_0_2 = 16 + 2 * cq + 1;
                int kkz_1_2 = key0;
                int pcolz_2_2 = hh3z_0_2 * 128 + (kkz_1_2 * 2 ^ (hh3z_0_2 & 7) << 4);
                {
                    __half _hval_52 = __float2half_rn(0.0f);
                    uint16_t _bits_52 = *(uint16_t*)&_hval_52;
                    uint32_t _addr_52 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_2_2);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_52), "h"(_bits_52) : "memory");
                }
                {
                    __half _hval_53 = __float2half_rn(0.0f);
                    uint16_t _bits_53 = *(uint16_t*)&_hval_53;
                    uint32_t _addr_53 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_2_2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_53), "h"(_bits_53) : "memory");
                }
                int hh3z_3_2 = 16 + 2 * cq;
                int kkz_4_2 = key1;
                int pcolz_5_2 = hh3z_3_2 * 128 + (kkz_4_2 * 2 ^ (hh3z_3_2 & 7) << 4);
                {
                    __half _hval_54 = __float2half_rn(0.0f);
                    uint16_t _bits_54 = *(uint16_t*)&_hval_54;
                    uint32_t _addr_54 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_2);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_54), "h"(_bits_54) : "memory");
                }
                {
                    __half _hval_55 = __float2half_rn(0.0f);
                    uint16_t _bits_55 = *(uint16_t*)&_hval_55;
                    uint32_t _addr_55 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_55), "h"(_bits_55) : "memory");
                }
                int hh3z_6_2 = 16 + 2 * cq + 1;
                int kkz_7_2 = key1;
                int pcolz_8_2 = hh3z_6_2 * 128 + (kkz_7_2 * 2 ^ (hh3z_6_2 & 7) << 4);
                {
                    __half _hval_56 = __float2half_rn(0.0f);
                    uint16_t _bits_56 = *(uint16_t*)&_hval_56;
                    uint32_t _addr_56 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_8_2);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_56), "h"(_bits_56) : "memory");
                }
                {
                    __half _hval_57 = __float2half_rn(0.0f);
                    uint16_t _bits_57 = *(uint16_t*)&_hval_57;
                    uint32_t _addr_57 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_8_2));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_57), "h"(_bits_57) : "memory");
                }
            }
        }
        if ((bl >> 3 & 1) != 0) {
            int hh3_3 = 24 + 2 * cq;
            int kk_4 = key0;
            int pcol_4 = hh3_3 * 128 + (kk_4 * 2 ^ (hh3_3 & 7) << 4);
            float msub_4 = mrun[6];
            if (msub_4 == -CAKE_INF) {
                msub_4 = 0.0f;
            }
            float _exp2_32 = approx_exp2(d_s0[12] - msub_4);
            float p0_3 = _exp2_32;
            float _exp2_33 = approx_exp2(d_s1[12] - msub_4);
            float p1_3 = _exp2_33;
            lp[6] = lp[6] + (p0_3 + p1_3);
            {
                __half _hval_58 = __float2half_rn(p0_3);
                uint16_t _bits_58 = *(uint16_t*)&_hval_58;
                uint32_t _addr_58 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_4);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_58), "h"(_bits_58) : "memory");
            }
            {
                __half _hval_59 = __float2half_rn(p1_3);
                uint16_t _bits_59 = *(uint16_t*)&_hval_59;
                uint32_t _addr_59 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_4));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_59), "h"(_bits_59) : "memory");
            }
            int hh3_0_3 = 24 + 2 * cq + 1;
            int kk_1_3 = key0;
            int pcol_2_3 = hh3_0_3 * 128 + (kk_1_3 * 2 ^ (hh3_0_3 & 7) << 4);
            float msub_3_3 = mrun[7];
            if (msub_3_3 == -CAKE_INF) {
                msub_3_3 = 0.0f;
            }
            float _exp2_34 = approx_exp2(d_s0[13] - msub_3_3);
            float p0_4_3 = _exp2_34;
            float _exp2_35 = approx_exp2(d_s1[13] - msub_3_3);
            float p1_5_3 = _exp2_35;
            lp[7] = lp[7] + (p0_4_3 + p1_5_3);
            {
                __half _hval_60 = __float2half_rn(p0_4_3);
                uint16_t _bits_60 = *(uint16_t*)&_hval_60;
                uint32_t _addr_60 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_2_3);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_60), "h"(_bits_60) : "memory");
            }
            {
                __half _hval_61 = __float2half_rn(p1_5_3);
                uint16_t _bits_61 = *(uint16_t*)&_hval_61;
                uint32_t _addr_61 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_2_3));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_61), "h"(_bits_61) : "memory");
            }
            int hh3_6_3 = 24 + 2 * cq;
            int kk_7_3 = key1;
            int pcol_8_3 = hh3_6_3 * 128 + (kk_7_3 * 2 ^ (hh3_6_3 & 7) << 4);
            float msub_9_3 = mrun[6];
            if (msub_9_3 == -CAKE_INF) {
                msub_9_3 = 0.0f;
            }
            float _exp2_36 = approx_exp2(d_s0[14] - msub_9_3);
            float p0_10_3 = _exp2_36;
            float _exp2_37 = approx_exp2(d_s1[14] - msub_9_3);
            float p1_11_3 = _exp2_37;
            lp[6] = lp[6] + (p0_10_3 + p1_11_3);
            {
                __half _hval_62 = __float2half_rn(p0_10_3);
                uint16_t _bits_62 = *(uint16_t*)&_hval_62;
                uint32_t _addr_62 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_8_3);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_62), "h"(_bits_62) : "memory");
            }
            {
                __half _hval_63 = __float2half_rn(p1_11_3);
                uint16_t _bits_63 = *(uint16_t*)&_hval_63;
                uint32_t _addr_63 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_8_3));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_63), "h"(_bits_63) : "memory");
            }
            int hh3_12_3 = 24 + 2 * cq + 1;
            int kk_13_3 = key1;
            int pcol_14_3 = hh3_12_3 * 128 + (kk_13_3 * 2 ^ (hh3_12_3 & 7) << 4);
            float msub_15_3 = mrun[7];
            if (msub_15_3 == -CAKE_INF) {
                msub_15_3 = 0.0f;
            }
            float _exp2_38 = approx_exp2(d_s0[15] - msub_15_3);
            float p0_16_3 = _exp2_38;
            float _exp2_39 = approx_exp2(d_s1[15] - msub_15_3);
            float p1_17_3 = _exp2_39;
            lp[7] = lp[7] + (p0_16_3 + p1_17_3);
            {
                __half _hval_64 = __float2half_rn(p0_16_3);
                uint16_t _bits_64 = *(uint16_t*)&_hval_64;
                uint32_t _addr_64 = static_cast<uint32_t>(p16_addr + (unsigned int)pcol_14_3);
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_64), "h"(_bits_64) : "memory");
            }
            {
                __half _hval_65 = __float2half_rn(p1_17_3);
                uint16_t _bits_65 = *(uint16_t*)&_hval_65;
                uint32_t _addr_65 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcol_14_3));
                asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_65), "h"(_bits_65) : "memory");
            }
        }
        if ((bl >> 3 & 1) == 0) {
            if ((prev_bl >> 3 & 1) != 0) {
                int hh3z_4 = 24 + 2 * cq;
                int kkz_5 = key0;
                int pcolz_4 = hh3z_4 * 128 + (kkz_5 * 2 ^ (hh3z_4 & 7) << 4);
                {
                    __half _hval_66 = __float2half_rn(0.0f);
                    uint16_t _bits_66 = *(uint16_t*)&_hval_66;
                    uint32_t _addr_66 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_4);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_66), "h"(_bits_66) : "memory");
                }
                {
                    __half _hval_67 = __float2half_rn(0.0f);
                    uint16_t _bits_67 = *(uint16_t*)&_hval_67;
                    uint32_t _addr_67 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_4));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_67), "h"(_bits_67) : "memory");
                }
                int hh3z_0_3 = 24 + 2 * cq + 1;
                int kkz_1_3 = key0;
                int pcolz_2_3 = hh3z_0_3 * 128 + (kkz_1_3 * 2 ^ (hh3z_0_3 & 7) << 4);
                {
                    __half _hval_68 = __float2half_rn(0.0f);
                    uint16_t _bits_68 = *(uint16_t*)&_hval_68;
                    uint32_t _addr_68 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_2_3);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_68), "h"(_bits_68) : "memory");
                }
                {
                    __half _hval_69 = __float2half_rn(0.0f);
                    uint16_t _bits_69 = *(uint16_t*)&_hval_69;
                    uint32_t _addr_69 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_2_3));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_69), "h"(_bits_69) : "memory");
                }
                int hh3z_3_3 = 24 + 2 * cq;
                int kkz_4_3 = key1;
                int pcolz_5_3 = hh3z_3_3 * 128 + (kkz_4_3 * 2 ^ (hh3z_3_3 & 7) << 4);
                {
                    __half _hval_70 = __float2half_rn(0.0f);
                    uint16_t _bits_70 = *(uint16_t*)&_hval_70;
                    uint32_t _addr_70 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_5_3);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_70), "h"(_bits_70) : "memory");
                }
                {
                    __half _hval_71 = __float2half_rn(0.0f);
                    uint16_t _bits_71 = *(uint16_t*)&_hval_71;
                    uint32_t _addr_71 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_5_3));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_71), "h"(_bits_71) : "memory");
                }
                int hh3z_6_3 = 24 + 2 * cq + 1;
                int kkz_7_3 = key1;
                int pcolz_8_3 = hh3z_6_3 * 128 + (kkz_7_3 * 2 ^ (hh3z_6_3 & 7) << 4);
                {
                    __half _hval_72 = __float2half_rn(0.0f);
                    uint16_t _bits_72 = *(uint16_t*)&_hval_72;
                    uint32_t _addr_72 = static_cast<uint32_t>(p16_addr + (unsigned int)pcolz_8_3);
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_72), "h"(_bits_72) : "memory");
                }
                {
                    __half _hval_73 = __float2half_rn(0.0f);
                    uint16_t _bits_73 = *(uint16_t*)&_hval_73;
                    uint32_t _addr_73 = static_cast<uint32_t>(p16_addr + (unsigned int)(4096 + pcolz_8_3));
                    asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_73), "h"(_bits_73) : "memory");
                }
            }
        }
        prev_bl = bl;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        av0[0] = dv0[0] + zero_u32;
        av1[0] = dv1[0] + zero_u32;
        av0[1] = dv0[1] + zero_u32;
        av1[1] = dv1[1] + zero_u32;
        av0[2] = dv0[2] + zero_u32;
        av1[2] = dv1[2] + zero_u32;
        av0[3] = dv0[3] + zero_u32;
        av1[3] = dv1[3] + zero_u32;
        av0[4] = dv0[4] + zero_u32;
        av1[4] = dv1[4] + zero_u32;
        av0[5] = dv0[5] + zero_u32;
        av1[5] = dv1[5] + zero_u32;
        av0[6] = dv0[6] + zero_u32;
        av1[6] = dv1[6] + zero_u32;
        av0[7] = dv0[7] + zero_u32;
        av1[7] = dv1[7] + zero_u32;
        av0[8] = dv0[8] + zero_u32;
        av1[8] = dv1[8] + zero_u32;
        av0[9] = dv0[9] + zero_u32;
        av1[9] = dv1[9] + zero_u32;
        av0[10] = dv0[10] + zero_u32;
        av1[10] = dv1[10] + zero_u32;
        av0[11] = dv0[11] + zero_u32;
        av1[11] = dv1[11] + zero_u32;
        av0[12] = dv0[12] + zero_u32;
        av1[12] = dv1[12] + zero_u32;
        av0[13] = dv0[13] + zero_u32;
        av1[13] = dv1[13] + zero_u32;
        av0[14] = dv0[14] + zero_u32;
        av1[14] = dv1[14] + zero_u32;
        av0[15] = dv0[15] + zero_u32;
        av1[15] = dv1[15] + zero_u32;
        av0[16] = dv0[16] + zero_u32;
        av1[16] = dv1[16] + zero_u32;
        av0[17] = dv0[17] + zero_u32;
        av1[17] = dv1[17] + zero_u32;
        av0[18] = dv0[18] + zero_u32;
        av1[18] = dv1[18] + zero_u32;
        av0[19] = dv0[19] + zero_u32;
        av1[19] = dv1[19] + zero_u32;
        av0[20] = dv0[20] + zero_u32;
        av1[20] = dv1[20] + zero_u32;
        av0[21] = dv0[21] + zero_u32;
        av1[21] = dv1[21] + zero_u32;
        av0[22] = dv0[22] + zero_u32;
        av1[22] = dv1[22] + zero_u32;
        av0[23] = dv0[23] + zero_u32;
        av1[23] = dv1[23] + zero_u32;
        av0[24] = dv0[24] + zero_u32;
        av1[24] = dv1[24] + zero_u32;
        av0[25] = dv0[25] + zero_u32;
        av1[25] = dv1[25] + zero_u32;
        av0[26] = dv0[26] + zero_u32;
        av1[26] = dv1[26] + zero_u32;
        av0[27] = dv0[27] + zero_u32;
        av1[27] = dv1[27] + zero_u32;
        av0[28] = dv0[28] + zero_u32;
        av1[28] = dv1[28] + zero_u32;
        av0[29] = dv0[29] + zero_u32;
        av1[29] = dv1[29] + zero_u32;
        av0[30] = dv0[30] + zero_u32;
        av1[30] = dv1[30] + zero_u32;
        av0[31] = dv0[31] + zero_u32;
        av1[31] = dv1[31] + zero_u32;
        int nxt = j + 2;
        if (nxt < nblk_u) {
            int page_head_2 = meta_pg[nxt];
            if (warp == 1) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768), (&K), 0, 0, page_head_2, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 8192, (&K), 0, 64, page_head_2, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 16384, (&V), 0, 0, page_head_2, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 24576, (&V), 0, 64, page_head_2, kv_full_addr + (stage) * 8);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av0[0]), "r"(av0[1]), "r"(av0[2]), "r"(av0[3]), "l"(_wgmma_b_0_7)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av0[16]), "r"(av0[(16) + 1]), "r"(av0[(16) + 2]), "r"(av0[(16) + 3]), "l"(_wgmma_b_0_7)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av1[0]), "r"(av1[1]), "r"(av1[2]), "r"(av1[3]), "l"(_wgmma_b_0_8)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av1[16]), "r"(av1[(16) + 1]), "r"(av1[(16) + 2]), "r"(av1[(16) + 3]), "l"(_wgmma_b_0_8)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av0[4]), "r"(av0[(4) + 1]), "r"(av0[(4) + 2]), "r"(av0[(4) + 3]), "l"(_wgmma_b_0_7 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av0[20]), "r"(av0[(20) + 1]), "r"(av0[(20) + 2]), "r"(av0[(20) + 3]), "l"(_wgmma_b_0_7 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av1[4]), "r"(av1[(4) + 1]), "r"(av1[(4) + 2]), "r"(av1[(4) + 3]), "l"(_wgmma_b_0_8 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av1[20]), "r"(av1[(20) + 1]), "r"(av1[(20) + 2]), "r"(av1[(20) + 3]), "l"(_wgmma_b_0_8 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av0[8]), "r"(av0[(8) + 1]), "r"(av0[(8) + 2]), "r"(av0[(8) + 3]), "l"(_wgmma_b_0_7 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av0[24]), "r"(av0[(24) + 1]), "r"(av0[(24) + 2]), "r"(av0[(24) + 3]), "l"(_wgmma_b_0_7 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av1[8]), "r"(av1[(8) + 1]), "r"(av1[(8) + 2]), "r"(av1[(8) + 3]), "l"(_wgmma_b_0_8 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av1[24]), "r"(av1[(24) + 1]), "r"(av1[(24) + 2]), "r"(av1[(24) + 3]), "l"(_wgmma_b_0_8 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av0[12]), "r"(av0[(12) + 1]), "r"(av0[(12) + 2]), "r"(av0[(12) + 3]), "l"(_wgmma_b_0_7 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av0[28]), "r"(av0[(28) + 1]), "r"(av0[(28) + 2]), "r"(av0[(28) + 3]), "l"(_wgmma_b_0_7 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3]), "+f"(d_o0[4]), "+f"(d_o0[5]), "+f"(d_o0[6]), "+f"(d_o0[7]), "+f"(d_o0[8]), "+f"(d_o0[9]), "+f"(d_o0[10]), "+f"(d_o0[11]), "+f"(d_o0[12]), "+f"(d_o0[13]), "+f"(d_o0[14]), "+f"(d_o0[15])
            : "r"(av1[12]), "r"(av1[(12) + 1]), "r"(av1[(12) + 2]), "r"(av1[(12) + 3]), "l"(_wgmma_b_0_8 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, {%16, %17, %18, %19}, %20, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3]), "+f"(d_o1[4]), "+f"(d_o1[5]), "+f"(d_o1[6]), "+f"(d_o1[7]), "+f"(d_o1[8]), "+f"(d_o1[9]), "+f"(d_o1[10]), "+f"(d_o1[11]), "+f"(d_o1[12]), "+f"(d_o1[13]), "+f"(d_o1[14]), "+f"(d_o1[15])
            : "r"(av1[28]), "r"(av1[(28) + 1]), "r"(av1[(28) + 2]), "r"(av1[(28) + 3]), "l"(_wgmma_b_0_8 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        first_page = 0;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    float t2 = lp[0];
    float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, t2, 4);
    t2 = t2 + _shfl_xor_24;
    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, t2, 8);
    t2 = t2 + _shfl_xor_25;
    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, t2, 16);
    t2 = t2 + _shfl_xor_26;
    lp[0] = t2;
    float t2_39 = lp[1];
    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, t2_39, 4);
    t2_39 = t2_39 + _shfl_xor_27;
    float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, t2_39, 8);
    t2_39 = t2_39 + _shfl_xor_28;
    float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, t2_39, 16);
    t2_39 = t2_39 + _shfl_xor_29;
    lp[1] = t2_39;
    float t2_40 = lp[2];
    float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, t2_40, 4);
    t2_40 = t2_40 + _shfl_xor_30;
    float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, t2_40, 8);
    t2_40 = t2_40 + _shfl_xor_31;
    float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, t2_40, 16);
    t2_40 = t2_40 + _shfl_xor_32;
    lp[2] = t2_40;
    float t2_41 = lp[3];
    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, t2_41, 4);
    t2_41 = t2_41 + _shfl_xor_33;
    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, t2_41, 8);
    t2_41 = t2_41 + _shfl_xor_34;
    float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, t2_41, 16);
    t2_41 = t2_41 + _shfl_xor_35;
    lp[3] = t2_41;
    float t2_42 = lp[4];
    float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 4);
    t2_42 = t2_42 + _shfl_xor_36;
    float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 8);
    t2_42 = t2_42 + _shfl_xor_37;
    float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, t2_42, 16);
    t2_42 = t2_42 + _shfl_xor_38;
    lp[4] = t2_42;
    float t2_43 = lp[5];
    float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 4);
    t2_43 = t2_43 + _shfl_xor_39;
    float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 8);
    t2_43 = t2_43 + _shfl_xor_40;
    float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, t2_43, 16);
    t2_43 = t2_43 + _shfl_xor_41;
    lp[5] = t2_43;
    float t2_44 = lp[6];
    float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 4);
    t2_44 = t2_44 + _shfl_xor_42;
    float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 8);
    t2_44 = t2_44 + _shfl_xor_43;
    float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, t2_44, 16);
    t2_44 = t2_44 + _shfl_xor_44;
    lp[6] = t2_44;
    float t2_45 = lp[7];
    float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 4);
    t2_45 = t2_45 + _shfl_xor_45;
    float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 8);
    t2_45 = t2_45 + _shfl_xor_46;
    float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, t2_45, 16);
    t2_45 = t2_45 + _shfl_xor_47;
    lp[7] = t2_45;
    if (lane_0 < 4) {
        int hh4 = 2 * cq;
        lred[hh4 * 4 + warp_1] = lp[0];
        int hh4_0 = 2 * cq + 1;
        lred[hh4_0 * 4 + warp_1] = lp[1];
        int hh4_1 = 8 + 2 * cq;
        lred[hh4_1 * 4 + warp_1] = lp[2];
        int hh4_2 = 8 + 2 * cq + 1;
        lred[hh4_2 * 4 + warp_1] = lp[3];
        int hh4_3 = 16 + 2 * cq;
        lred[hh4_3 * 4 + warp_1] = lp[4];
        int hh4_4 = 16 + 2 * cq + 1;
        lred[hh4_4 * 4 + warp_1] = lp[5];
        int hh4_5 = 24 + 2 * cq;
        lred[hh4_5 * 4 + warp_1] = lp[6];
        int hh4_6 = 24 + 2 * cq + 1;
        lred[hh4_6 * 4 + warp_1] = lp[7];
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    float lsum[8];
    int hh5 = 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5 * 4 * 4)));
    float ls = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[0] = 0.0f;
    if (ls != 0.0f) {
        float _rcp_0 = approx_rcp(ls);
        lsum[0] = _rcp_0;
    }
    int hh5_46 = 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_46 * 4 * 4)));
    float ls_47 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[1] = 0.0f;
    if (ls_47 != 0.0f) {
        float _rcp_1 = approx_rcp(ls_47);
        lsum[1] = _rcp_1;
    }
    int hh5_48 = 8 + 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_48 * 4 * 4)));
    float ls_49 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[2] = 0.0f;
    if (ls_49 != 0.0f) {
        float _rcp_2 = approx_rcp(ls_49);
        lsum[2] = _rcp_2;
    }
    int hh5_50 = 8 + 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_50 * 4 * 4)));
    float ls_51 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[3] = 0.0f;
    if (ls_51 != 0.0f) {
        float _rcp_3 = approx_rcp(ls_51);
        lsum[3] = _rcp_3;
    }
    int hh5_52 = 16 + 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_52 * 4 * 4)));
    float ls_53 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[4] = 0.0f;
    if (ls_53 != 0.0f) {
        float _rcp_4 = approx_rcp(ls_53);
        lsum[4] = _rcp_4;
    }
    int hh5_54 = 16 + 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_54 * 4 * 4)));
    float ls_55 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[5] = 0.0f;
    if (ls_55 != 0.0f) {
        float _rcp_5 = approx_rcp(ls_55);
        lsum[5] = _rcp_5;
    }
    int hh5_56 = 24 + 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_56 * 4 * 4)));
    float ls_57 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[6] = 0.0f;
    if (ls_57 != 0.0f) {
        float _rcp_6 = approx_rcp(ls_57);
        lsum[6] = _rcp_6;
    }
    int hh5_58 = 24 + 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_58 * 4 * 4)));
    float ls_59 = rv[0] + rv[1] + (rv[2] + rv[3]);
    lsum[7] = 0.0f;
    if (ls_59 != 0.0f) {
        float _rcp_7 = approx_rcp(ls_59);
        lsum[7] = _rcp_7;
    }
    int hh6 = 2 * cq;
    int dd = warp_1 * 16 + g;
    {
        uint32_t _addr_74 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6 * 128 + dd) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_74), "f"(d_o0[0] * lsum[0]) : "memory");
    }
    {
        uint32_t _addr_75 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6 * 128 + dd + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_75), "f"(d_o1[0] * lsum[0]) : "memory");
    }
    int hh6_60 = 2 * cq + 1;
    int dd_61 = warp_1 * 16 + g;
    {
        uint32_t _addr_76 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_60 * 128 + dd_61) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_76), "f"(d_o0[1] * lsum[1]) : "memory");
    }
    {
        uint32_t _addr_77 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_60 * 128 + dd_61 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_77), "f"(d_o1[1] * lsum[1]) : "memory");
    }
    int hh6_62 = 2 * cq;
    int dd_63 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_78 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_62 * 128 + dd_63) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_78), "f"(d_o0[2] * lsum[0]) : "memory");
    }
    {
        uint32_t _addr_79 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_62 * 128 + dd_63 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_79), "f"(d_o1[2] * lsum[0]) : "memory");
    }
    int hh6_64 = 2 * cq + 1;
    int dd_65 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_80 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_64 * 128 + dd_65) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_80), "f"(d_o0[3] * lsum[1]) : "memory");
    }
    {
        uint32_t _addr_81 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_64 * 128 + dd_65 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_81), "f"(d_o1[3] * lsum[1]) : "memory");
    }
    int hh6_66 = 8 + 2 * cq;
    int dd_67 = warp_1 * 16 + g;
    {
        uint32_t _addr_82 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_66 * 128 + dd_67) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_82), "f"(d_o0[4] * lsum[2]) : "memory");
    }
    {
        uint32_t _addr_83 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_66 * 128 + dd_67 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_83), "f"(d_o1[4] * lsum[2]) : "memory");
    }
    int hh6_68 = 8 + 2 * cq + 1;
    int dd_69 = warp_1 * 16 + g;
    {
        uint32_t _addr_84 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_68 * 128 + dd_69) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_84), "f"(d_o0[5] * lsum[3]) : "memory");
    }
    {
        uint32_t _addr_85 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_68 * 128 + dd_69 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_85), "f"(d_o1[5] * lsum[3]) : "memory");
    }
    int hh6_70 = 8 + 2 * cq;
    int dd_71 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_86 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_70 * 128 + dd_71) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_86), "f"(d_o0[6] * lsum[2]) : "memory");
    }
    {
        uint32_t _addr_87 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_70 * 128 + dd_71 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_87), "f"(d_o1[6] * lsum[2]) : "memory");
    }
    int hh6_72 = 8 + 2 * cq + 1;
    int dd_73 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_88 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_72 * 128 + dd_73) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_88), "f"(d_o0[7] * lsum[3]) : "memory");
    }
    {
        uint32_t _addr_89 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_72 * 128 + dd_73 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_89), "f"(d_o1[7] * lsum[3]) : "memory");
    }
    int hh6_74 = 16 + 2 * cq;
    int dd_75 = warp_1 * 16 + g;
    {
        uint32_t _addr_90 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_74 * 128 + dd_75) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_90), "f"(d_o0[8] * lsum[4]) : "memory");
    }
    {
        uint32_t _addr_91 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_74 * 128 + dd_75 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_91), "f"(d_o1[8] * lsum[4]) : "memory");
    }
    int hh6_76 = 16 + 2 * cq + 1;
    int dd_77 = warp_1 * 16 + g;
    {
        uint32_t _addr_92 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_76 * 128 + dd_77) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_92), "f"(d_o0[9] * lsum[5]) : "memory");
    }
    {
        uint32_t _addr_93 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_76 * 128 + dd_77 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_93), "f"(d_o1[9] * lsum[5]) : "memory");
    }
    int hh6_78 = 16 + 2 * cq;
    int dd_79 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_94 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_78 * 128 + dd_79) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_94), "f"(d_o0[10] * lsum[4]) : "memory");
    }
    {
        uint32_t _addr_95 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_78 * 128 + dd_79 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_95), "f"(d_o1[10] * lsum[4]) : "memory");
    }
    int hh6_80 = 16 + 2 * cq + 1;
    int dd_81 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_96 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_80 * 128 + dd_81) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_96), "f"(d_o0[11] * lsum[5]) : "memory");
    }
    {
        uint32_t _addr_97 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_80 * 128 + dd_81 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_97), "f"(d_o1[11] * lsum[5]) : "memory");
    }
    int hh6_82 = 24 + 2 * cq;
    int dd_83 = warp_1 * 16 + g;
    {
        uint32_t _addr_98 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_82 * 128 + dd_83) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_98), "f"(d_o0[12] * lsum[6]) : "memory");
    }
    {
        uint32_t _addr_99 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_82 * 128 + dd_83 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_99), "f"(d_o1[12] * lsum[6]) : "memory");
    }
    int hh6_84 = 24 + 2 * cq + 1;
    int dd_85 = warp_1 * 16 + g;
    {
        uint32_t _addr_100 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_84 * 128 + dd_85) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_100), "f"(d_o0[13] * lsum[7]) : "memory");
    }
    {
        uint32_t _addr_101 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_84 * 128 + dd_85 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_101), "f"(d_o1[13] * lsum[7]) : "memory");
    }
    int hh6_86 = 24 + 2 * cq;
    int dd_87 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_102 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_86 * 128 + dd_87) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_102), "f"(d_o0[14] * lsum[6]) : "memory");
    }
    {
        uint32_t _addr_103 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_86 * 128 + dd_87 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_103), "f"(d_o1[14] * lsum[6]) : "memory");
    }
    int hh6_88 = 24 + 2 * cq + 1;
    int dd_89 = warp_1 * 16 + g + 8;
    {
        uint32_t _addr_104 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_88 * 128 + dd_89) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_104), "f"(d_o0[15] * lsum[7]) : "memory");
    }
    {
        uint32_t _addr_105 = static_cast<uint32_t>(kv8_addr + (unsigned int)((hh6_88 * 128 + dd_89 + 64) * 4));
        asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_105), "f"(d_o1[15] * lsum[7]) : "memory");
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    float ov[8];
    int vid3 = tid_1;
    int on = vid3 / 16;
    int oseg = vid3 - on * 16;
    int od0 = oseg * 8;
    int tl_o = on >> 3;
    if (tl_o < n_tok) {
        int hg_o = on - (tl_o << 3);
        int orow = (cu_b + tok0 + tl_o) * num_q_heads + kv_head * 8 + hg_o;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
            : "r"(kv8_addr + (unsigned int)((on * 128 + od0) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
            : "r"(kv8_addr + (unsigned int)((on * 128 + od0 + 4) * 4)));
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (orow * 128 + od0)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
    int vid3_90 = tid_1 + 128;
    int on_91 = vid3_90 / 16;
    int oseg_92 = vid3_90 - on_91 * 16;
    int od0_93 = oseg_92 * 8;
    int tl_o_94 = on_91 >> 3;
    if (tl_o_94 < n_tok) {
        int hg_o_1 = on_91 - (tl_o_94 << 3);
        int orow_1 = (cu_b + tok0 + tl_o_94) * num_q_heads + kv_head * 8 + hg_o_1;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_91 * 128 + od0_93) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_91 * 128 + od0_93 + 4) * 4)));
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (orow_1 * 128 + od0_93)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
    int vid3_95 = tid_1 + 256;
    int on_96 = vid3_95 / 16;
    int oseg_97 = vid3_95 - on_96 * 16;
    int od0_98 = oseg_97 * 8;
    int tl_o_99 = on_96 >> 3;
    if (tl_o_99 < n_tok) {
        int hg_o_2 = on_96 - (tl_o_99 << 3);
        int orow_2 = (cu_b + tok0 + tl_o_99) * num_q_heads + kv_head * 8 + hg_o_2;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_96 * 128 + od0_98) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_96 * 128 + od0_98 + 4) * 4)));
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (orow_2 * 128 + od0_98)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }
    int vid3_100 = tid_1 + 384;
    int on_101 = vid3_100 / 16;
    int oseg_102 = vid3_100 - on_101 * 16;
    int od0_103 = oseg_102 * 8;
    int tl_o_104 = on_101 >> 3;
    if (tl_o_104 < n_tok) {
        int hg_o_3 = on_101 - (tl_o_104 << 3);
        int orow_3 = (cu_b + tok0 + tl_o_104) * num_q_heads + kv_head * 8 + hg_o_3;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_101 * 128 + od0_103) * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
            : "r"(kv8_addr + (unsigned int)((on_101 * 128 + od0_103 + 4) * 4)));
        {
            __nv_bfloat162 _pk[4];
            _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
            _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
            _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
            _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (orow_3 * 128 + od0_103)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
