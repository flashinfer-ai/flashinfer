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
#define SMEM_KV8_STAGE_BYTES 16384
#define SMEM_KV8_STRIDE 16384
#define SMEM_K16_OFF 66560
#define SMEM_K16_STAGE_BYTES 16384
#define SMEM_K16_STRIDE 16384
#define SMEM_V16T_OFF 99328
#define SMEM_V16T_STAGE_BYTES 16384
#define SMEM_V16T_STRIDE 16384
#define SMEM_IDENT8_OFF 132096
#define SMEM_IDENT8_STAGE_BYTES 8192
#define SMEM_IDENT8_STRIDE 8192
#define SMEM_MARK_OFF 66560
#define SMEM_MARK_STAGE_BYTES 4096
#define SMEM_MARK_STRIDE 4096
#define SMEM_PREF_OFF 140288
#define SMEM_PREF_STAGE_BYTES 512
#define SMEM_PREF_STRIDE 512
#define SMEM_CNT_OFF 140800
#define SMEM_CNT_STAGE_BYTES 32
#define SMEM_CNT_STRIDE 32
#define SMEM_HDR_OFF 140832
#define SMEM_HDR_STAGE_BYTES 32
#define SMEM_HDR_STRIDE 32
#define SMEM_META_PG_OFF 140864
#define SMEM_META_PG_STAGE_BYTES 512
#define SMEM_META_PG_STRIDE 512
#define SMEM_META_BLK_OFF 141376
#define SMEM_META_BLK_STAGE_BYTES 512
#define SMEM_META_BLK_STRIDE 512
#define SMEM_META_MASK_OFF 141888
#define SMEM_META_MASK_STAGE_BYTES 1024
#define SMEM_META_MASK_STRIDE 1024
#define SMEM_TOTAL 142976
#define THREADS 384
#define TRACE 0
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_c09efd3ff19d883d18ac(unsigned int* __restrict__ Q32, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ q2k_indices, int* __restrict__ cu_seqlens_q, int* __restrict__ page_table, int* __restrict__ seqused_k, int* __restrict__ q_offset, int total_q, int batch, int num_q_heads, int num_kv_heads, int max_pages, int use_q_offset, float softmax_scale_log2, unsigned int zero_u32, unsigned long long* __restrict__ trace)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define kv_full_addr (mbar_base + 0)
    #define f16_full_addr (mbar_base + 32)
    #define f16_empty_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* kv8 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int kv8_addr = smem + 1024;
    __half* k16 = reinterpret_cast<__half*>(smem_raw + 66560);
    const int k16_addr = smem + 66560;
    __half* v16t = reinterpret_cast<__half*>(smem_raw + 99328);
    const int v16t_addr = smem + 99328;
    uint8_t* ident8 = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int ident8_addr = smem + 132096;
    uint8_t* mark = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int mark_addr = smem + 66560;
    int* pref = reinterpret_cast<int*>(smem_raw + 140288);
    const int pref_addr = smem + 140288;
    int* cnt = reinterpret_cast<int*>(smem_raw + 140800);
    const int cnt_addr = smem + 140800;
    int* hdr = reinterpret_cast<int*>(smem_raw + 140832);
    const int hdr_addr = smem + 140832;
    int* meta_pg = reinterpret_cast<int*>(smem_raw + 140864);
    const int meta_pg_addr = smem + 140864;
    int* meta_blk = reinterpret_cast<int*>(smem_raw + 141376);
    const int meta_blk_addr = smem + 141376;
    unsigned int* meta_mask = reinterpret_cast<unsigned int*>(smem_raw + 141888);
    const int meta_mask_addr = smem + 141888;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // f16_full: 2 barriers, init_count=128
            mbarrier_init(smem + 32, 128);
            mbarrier_init(smem + 40, 128);
            // f16_empty: 2 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: conv ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 104;");
        { // conv_main
            unsigned long long stamps[14];
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
            stamps[12] = 0;
            stamps[13] = 0;
            if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
            if (warp == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }
            int gi = blockIdx.x;
            int kv_head = blockIdx.y;
            int tid_1 = threadIdx.x;
            int lane_0 = lane;
            int warp_1 = warp;
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
                            cnt_b = (cu_hi - cu_lo + 8 - 1) / 8;
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
                            sel_tok0 = lg * 8;
                            sel_qlen = hi_s - sel_cu;
                            sel_ntok = sel_qlen - sel_tok0;
                            if (sel_ntok > 8) {
                                sel_ntok = 8;
                            }
                            found = 1;
                        }
                        int _shfl_4 = __shfl_sync(0xFFFFFFFF, incl, 31);
                        base_g = base_g + _shfl_4;
                    }
                }
                if (lane_0 == 0) {
                    int qoff = 0;
                    int sk_sel = seqused_k[sel_b];
                    if (found != 0) {
                        if (use_q_offset != 0) {
                            qoff = q_offset[sel_b];
                        }
                        if (use_q_offset == 0) {
                            qoff = sk_sel - sel_qlen;
                        }
                    }
                    hdr[0] = sel_b;
                    hdr[1] = sel_tok0;
                    hdr[2] = sel_ntok;
                    hdr[3] = sel_cu;
                    hdr[4] = qoff + sel_tok0;
                    hdr[5] = sk_sel;
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
                int ich_lin = sid;
                if (ich_lin < 512) {
                    int irow = ich_lin / 8;
                    int ich = ich_lin - irow * 8;
                    int iword = irow / 4;
                    unsigned int ione = 56 << 8 * (irow - iword * 4);
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
                int ich_lin_0 = sid + 352;
                if (ich_lin_0 < 512) {
                    int irow_1 = ich_lin_0 / 8;
                    int ich_1 = ich_lin_0 - irow_1 * 8;
                    int iword_1 = irow_1 / 4;
                    unsigned int ione_1 = 56 << 8 * (irow_1 - iword_1 * 4);
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
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int tok0_u = hdr[1];
            int n_tok_u = hdr[2];
            int cu_u = hdr[3];
            int b_u = hdr[0];
            int blk_e[1];
            int e = tid_1;
            blk_e[0] = -1;
            if (e < 128) {
                int tl_e = e / 16;
                int k_e = e - tl_e * 16;
                if (tl_e < n_tok_u) {
                    int bv = q2k_indices[(kv_head * total_q + cu_u + tok0_u + tl_e) * 16 + k_e];
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
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
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
            int c_t = 0;
            int incl_t = 0;
            if (tid_1 < 128) {
                if (tid_1 < ngrp) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mw[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(0) + 3]))
                        : "r"(mark_addr + (unsigned int)(tid_1 * 32)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mw[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw[(4) + 3]))
                        : "r"(mark_addr + (unsigned int)(tid_1 * 32) + 16));
                }
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
                incl_t = c_t;
                int _shfl_up_5 = __shfl_up_sync(0xFFFFFFFF, incl_t, 1, 32);
                int up2 = _shfl_up_5;
                if (lane_0 >= 1) {
                    incl_t = incl_t + up2;
                }
                int _shfl_up_6 = __shfl_up_sync(0xFFFFFFFF, incl_t, 2, 32);
                int up2_0 = _shfl_up_6;
                if (lane_0 >= 2) {
                    incl_t = incl_t + up2_0;
                }
                int _shfl_up_7 = __shfl_up_sync(0xFFFFFFFF, incl_t, 4, 32);
                int up2_1 = _shfl_up_7;
                if (lane_0 >= 4) {
                    incl_t = incl_t + up2_1;
                }
                int _shfl_up_8 = __shfl_up_sync(0xFFFFFFFF, incl_t, 8, 32);
                int up2_2 = _shfl_up_8;
                if (lane_0 >= 8) {
                    incl_t = incl_t + up2_2;
                }
                int _shfl_up_9 = __shfl_up_sync(0xFFFFFFFF, incl_t, 16, 32);
                int up2_3 = _shfl_up_9;
                if (lane_0 >= 16) {
                    incl_t = incl_t + up2_3;
                }
                if (lane_0 == 31) {
                    cnt[warp_1] = incl_t;
                }
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (tid_1 < 128) {
                int pos_t = incl_t - c_t;
                int cw_ = cnt[0];
                if (warp_1 > 0) {
                    pos_t = pos_t + cw_;
                }
                int cw__0 = cnt[1];
                if (warp_1 > 1) {
                    pos_t = pos_t + cw__0;
                }
                int cw__1 = cnt[2];
                if (warp_1 > 2) {
                    pos_t = pos_t + cw__1;
                }
                int cw__2 = cnt[3];
                if (warp_1 > 3) {
                    pos_t = pos_t + cw__2;
                }
                pref[tid_1] = pos_t;
                if (c_t != 0) {
                    if ((mw[0] & 1) != 0) {
                        int blk_u = tid_1 * 32;
                        int pg_u = page_table[b_u * max_pages + blk_u];
                        int blk_st = blk_u;
                        if (pg_u < 0) {
                            pg_u = 0;
                            blk_st = 4194304;
                        }
                        meta_blk[pos_t] = blk_st;
                        meta_pg[pos_t] = pg_u * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[0] >> 8 & 1) != 0) {
                        int blk_u_1 = tid_1 * 32 + 1;
                        int pg_u_1 = page_table[b_u * max_pages + blk_u_1];
                        int blk_st_1 = blk_u_1;
                        if (pg_u_1 < 0) {
                            pg_u_1 = 0;
                            blk_st_1 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_1;
                        meta_pg[pos_t] = pg_u_1 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[0] >> 16 & 1) != 0) {
                        int blk_u_2 = tid_1 * 32 + 2;
                        int pg_u_2 = page_table[b_u * max_pages + blk_u_2];
                        int blk_st_2 = blk_u_2;
                        if (pg_u_2 < 0) {
                            pg_u_2 = 0;
                            blk_st_2 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_2;
                        meta_pg[pos_t] = pg_u_2 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[0] >> 24 & 1) != 0) {
                        int blk_u_3 = tid_1 * 32 + 3;
                        int pg_u_3 = page_table[b_u * max_pages + blk_u_3];
                        int blk_st_3 = blk_u_3;
                        if (pg_u_3 < 0) {
                            pg_u_3 = 0;
                            blk_st_3 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_3;
                        meta_pg[pos_t] = pg_u_3 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[1] & 1) != 0) {
                        int blk_u_4 = tid_1 * 32 + 4;
                        int pg_u_4 = page_table[b_u * max_pages + blk_u_4];
                        int blk_st_4 = blk_u_4;
                        if (pg_u_4 < 0) {
                            pg_u_4 = 0;
                            blk_st_4 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_4;
                        meta_pg[pos_t] = pg_u_4 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[1] >> 8 & 1) != 0) {
                        int blk_u_5 = tid_1 * 32 + 4 + 1;
                        int pg_u_5 = page_table[b_u * max_pages + blk_u_5];
                        int blk_st_5 = blk_u_5;
                        if (pg_u_5 < 0) {
                            pg_u_5 = 0;
                            blk_st_5 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_5;
                        meta_pg[pos_t] = pg_u_5 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[1] >> 16 & 1) != 0) {
                        int blk_u_6 = tid_1 * 32 + 4 + 2;
                        int pg_u_6 = page_table[b_u * max_pages + blk_u_6];
                        int blk_st_6 = blk_u_6;
                        if (pg_u_6 < 0) {
                            pg_u_6 = 0;
                            blk_st_6 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_6;
                        meta_pg[pos_t] = pg_u_6 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[1] >> 24 & 1) != 0) {
                        int blk_u_7 = tid_1 * 32 + 4 + 3;
                        int pg_u_7 = page_table[b_u * max_pages + blk_u_7];
                        int blk_st_7 = blk_u_7;
                        if (pg_u_7 < 0) {
                            pg_u_7 = 0;
                            blk_st_7 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_7;
                        meta_pg[pos_t] = pg_u_7 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[2] & 1) != 0) {
                        int blk_u_8 = tid_1 * 32 + 8;
                        int pg_u_8 = page_table[b_u * max_pages + blk_u_8];
                        int blk_st_8 = blk_u_8;
                        if (pg_u_8 < 0) {
                            pg_u_8 = 0;
                            blk_st_8 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_8;
                        meta_pg[pos_t] = pg_u_8 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[2] >> 8 & 1) != 0) {
                        int blk_u_9 = tid_1 * 32 + 8 + 1;
                        int pg_u_9 = page_table[b_u * max_pages + blk_u_9];
                        int blk_st_9 = blk_u_9;
                        if (pg_u_9 < 0) {
                            pg_u_9 = 0;
                            blk_st_9 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_9;
                        meta_pg[pos_t] = pg_u_9 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[2] >> 16 & 1) != 0) {
                        int blk_u_10 = tid_1 * 32 + 8 + 2;
                        int pg_u_10 = page_table[b_u * max_pages + blk_u_10];
                        int blk_st_10 = blk_u_10;
                        if (pg_u_10 < 0) {
                            pg_u_10 = 0;
                            blk_st_10 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_10;
                        meta_pg[pos_t] = pg_u_10 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[2] >> 24 & 1) != 0) {
                        int blk_u_11 = tid_1 * 32 + 8 + 3;
                        int pg_u_11 = page_table[b_u * max_pages + blk_u_11];
                        int blk_st_11 = blk_u_11;
                        if (pg_u_11 < 0) {
                            pg_u_11 = 0;
                            blk_st_11 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_11;
                        meta_pg[pos_t] = pg_u_11 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[3] & 1) != 0) {
                        int blk_u_12 = tid_1 * 32 + 12;
                        int pg_u_12 = page_table[b_u * max_pages + blk_u_12];
                        int blk_st_12 = blk_u_12;
                        if (pg_u_12 < 0) {
                            pg_u_12 = 0;
                            blk_st_12 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_12;
                        meta_pg[pos_t] = pg_u_12 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[3] >> 8 & 1) != 0) {
                        int blk_u_13 = tid_1 * 32 + 12 + 1;
                        int pg_u_13 = page_table[b_u * max_pages + blk_u_13];
                        int blk_st_13 = blk_u_13;
                        if (pg_u_13 < 0) {
                            pg_u_13 = 0;
                            blk_st_13 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_13;
                        meta_pg[pos_t] = pg_u_13 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[3] >> 16 & 1) != 0) {
                        int blk_u_14 = tid_1 * 32 + 12 + 2;
                        int pg_u_14 = page_table[b_u * max_pages + blk_u_14];
                        int blk_st_14 = blk_u_14;
                        if (pg_u_14 < 0) {
                            pg_u_14 = 0;
                            blk_st_14 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_14;
                        meta_pg[pos_t] = pg_u_14 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[3] >> 24 & 1) != 0) {
                        int blk_u_15 = tid_1 * 32 + 12 + 3;
                        int pg_u_15 = page_table[b_u * max_pages + blk_u_15];
                        int blk_st_15 = blk_u_15;
                        if (pg_u_15 < 0) {
                            pg_u_15 = 0;
                            blk_st_15 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_15;
                        meta_pg[pos_t] = pg_u_15 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[4] & 1) != 0) {
                        int blk_u_16 = tid_1 * 32 + 16;
                        int pg_u_16 = page_table[b_u * max_pages + blk_u_16];
                        int blk_st_16 = blk_u_16;
                        if (pg_u_16 < 0) {
                            pg_u_16 = 0;
                            blk_st_16 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_16;
                        meta_pg[pos_t] = pg_u_16 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[4] >> 8 & 1) != 0) {
                        int blk_u_17 = tid_1 * 32 + 16 + 1;
                        int pg_u_17 = page_table[b_u * max_pages + blk_u_17];
                        int blk_st_17 = blk_u_17;
                        if (pg_u_17 < 0) {
                            pg_u_17 = 0;
                            blk_st_17 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_17;
                        meta_pg[pos_t] = pg_u_17 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[4] >> 16 & 1) != 0) {
                        int blk_u_18 = tid_1 * 32 + 16 + 2;
                        int pg_u_18 = page_table[b_u * max_pages + blk_u_18];
                        int blk_st_18 = blk_u_18;
                        if (pg_u_18 < 0) {
                            pg_u_18 = 0;
                            blk_st_18 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_18;
                        meta_pg[pos_t] = pg_u_18 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[4] >> 24 & 1) != 0) {
                        int blk_u_19 = tid_1 * 32 + 16 + 3;
                        int pg_u_19 = page_table[b_u * max_pages + blk_u_19];
                        int blk_st_19 = blk_u_19;
                        if (pg_u_19 < 0) {
                            pg_u_19 = 0;
                            blk_st_19 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_19;
                        meta_pg[pos_t] = pg_u_19 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[5] & 1) != 0) {
                        int blk_u_20 = tid_1 * 32 + 20;
                        int pg_u_20 = page_table[b_u * max_pages + blk_u_20];
                        int blk_st_20 = blk_u_20;
                        if (pg_u_20 < 0) {
                            pg_u_20 = 0;
                            blk_st_20 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_20;
                        meta_pg[pos_t] = pg_u_20 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[5] >> 8 & 1) != 0) {
                        int blk_u_21 = tid_1 * 32 + 20 + 1;
                        int pg_u_21 = page_table[b_u * max_pages + blk_u_21];
                        int blk_st_21 = blk_u_21;
                        if (pg_u_21 < 0) {
                            pg_u_21 = 0;
                            blk_st_21 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_21;
                        meta_pg[pos_t] = pg_u_21 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[5] >> 16 & 1) != 0) {
                        int blk_u_22 = tid_1 * 32 + 20 + 2;
                        int pg_u_22 = page_table[b_u * max_pages + blk_u_22];
                        int blk_st_22 = blk_u_22;
                        if (pg_u_22 < 0) {
                            pg_u_22 = 0;
                            blk_st_22 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_22;
                        meta_pg[pos_t] = pg_u_22 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[5] >> 24 & 1) != 0) {
                        int blk_u_23 = tid_1 * 32 + 20 + 3;
                        int pg_u_23 = page_table[b_u * max_pages + blk_u_23];
                        int blk_st_23 = blk_u_23;
                        if (pg_u_23 < 0) {
                            pg_u_23 = 0;
                            blk_st_23 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_23;
                        meta_pg[pos_t] = pg_u_23 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[6] & 1) != 0) {
                        int blk_u_24 = tid_1 * 32 + 24;
                        int pg_u_24 = page_table[b_u * max_pages + blk_u_24];
                        int blk_st_24 = blk_u_24;
                        if (pg_u_24 < 0) {
                            pg_u_24 = 0;
                            blk_st_24 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_24;
                        meta_pg[pos_t] = pg_u_24 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[6] >> 8 & 1) != 0) {
                        int blk_u_25 = tid_1 * 32 + 24 + 1;
                        int pg_u_25 = page_table[b_u * max_pages + blk_u_25];
                        int blk_st_25 = blk_u_25;
                        if (pg_u_25 < 0) {
                            pg_u_25 = 0;
                            blk_st_25 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_25;
                        meta_pg[pos_t] = pg_u_25 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[6] >> 16 & 1) != 0) {
                        int blk_u_26 = tid_1 * 32 + 24 + 2;
                        int pg_u_26 = page_table[b_u * max_pages + blk_u_26];
                        int blk_st_26 = blk_u_26;
                        if (pg_u_26 < 0) {
                            pg_u_26 = 0;
                            blk_st_26 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_26;
                        meta_pg[pos_t] = pg_u_26 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[6] >> 24 & 1) != 0) {
                        int blk_u_27 = tid_1 * 32 + 24 + 3;
                        int pg_u_27 = page_table[b_u * max_pages + blk_u_27];
                        int blk_st_27 = blk_u_27;
                        if (pg_u_27 < 0) {
                            pg_u_27 = 0;
                            blk_st_27 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_27;
                        meta_pg[pos_t] = pg_u_27 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[7] & 1) != 0) {
                        int blk_u_28 = tid_1 * 32 + 28;
                        int pg_u_28 = page_table[b_u * max_pages + blk_u_28];
                        int blk_st_28 = blk_u_28;
                        if (pg_u_28 < 0) {
                            pg_u_28 = 0;
                            blk_st_28 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_28;
                        meta_pg[pos_t] = pg_u_28 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[7] >> 8 & 1) != 0) {
                        int blk_u_29 = tid_1 * 32 + 28 + 1;
                        int pg_u_29 = page_table[b_u * max_pages + blk_u_29];
                        int blk_st_29 = blk_u_29;
                        if (pg_u_29 < 0) {
                            pg_u_29 = 0;
                            blk_st_29 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_29;
                        meta_pg[pos_t] = pg_u_29 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[7] >> 16 & 1) != 0) {
                        int blk_u_30 = tid_1 * 32 + 28 + 2;
                        int pg_u_30 = page_table[b_u * max_pages + blk_u_30];
                        int blk_st_30 = blk_u_30;
                        if (pg_u_30 < 0) {
                            pg_u_30 = 0;
                            blk_st_30 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_30;
                        meta_pg[pos_t] = pg_u_30 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                    if ((mw[7] >> 24 & 1) != 0) {
                        int blk_u_31 = tid_1 * 32 + 28 + 3;
                        int pg_u_31 = page_table[b_u * max_pages + blk_u_31];
                        int blk_st_31 = blk_u_31;
                        if (pg_u_31 < 0) {
                            pg_u_31 = 0;
                            blk_st_31 = 4194304;
                        }
                        meta_blk[pos_t] = blk_st_31;
                        meta_pg[pos_t] = pg_u_31 * num_kv_heads + kv_head;
                        meta_mask[2 * pos_t] = 0;
                        meta_mask[2 * pos_t + 1] = 0;
                        pos_t = pos_t + 1;
                    }
                }
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
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
                        unsigned int lowmask = (1 << 8 * nb_) - 1;
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
                        unsigned int lowmask_1 = (1 << 8 * nb__0) - 1;
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
                        unsigned int lowmask_2 = (1 << 8 * nb__1) - 1;
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
                        unsigned int lowmask_3 = (1 << 8 * nb__2) - 1;
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
                        unsigned int lowmask_4 = (1 << 8 * nb__3) - 1;
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
                        unsigned int lowmask_5 = (1 << 8 * nb__4) - 1;
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
                        unsigned int lowmask_6 = (1 << 8 * nb__5) - 1;
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
                        unsigned int lowmask_7 = (1 << 8 * nb__6) - 1;
                        int _popc_23 = __popc(mw[7] & lowmask_7);
                        rank = rank + _popc_23;
                    }
                }
                int tl_i = tid_1 / 16;
                unsigned int bit = 1 << (tl_i & 31);
                atomicAdd_block(&meta_mask[2 * rank + (tl_i >> 5)], bit);
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int nblk_u = cnt[0] + cnt[1] + (cnt[2] + cnt[3]);
            int nhalf = 2 * nblk_u;
            int sk_b = hdr[5];
            int g_c = lane_0 / 4;
            int cq_c = lane_0 - g_c * 4;
            if (nhalf > 0) {
                int u_ = 0;
                int krow = 0;
                int page_head = meta_pg[u_];
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr, 16384);
                        tma_3d_gmem2smem(kv8_addr, (&K), 0, krow, page_head, kv_full_addr);
                        tma_3d_gmem2smem(kv8_addr + 8192, (&V), 0, krow, page_head, kv_full_addr);
                    }
                }
            }
            if (nhalf > 1) {
                int u__1 = 0;
                int krow_1 = 64;
                int page_head_1 = meta_pg[u__1];
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + 8, 16384);
                        tma_3d_gmem2smem(kv8_addr + 16384, (&K), 0, krow_1, page_head_1, kv_full_addr + 8);
                        tma_3d_gmem2smem(kv8_addr + 16384 + 8192, (&V), 0, krow_1, page_head_1, kv_full_addr + 8);
                    }
                }
            }
            if (nhalf > 2) {
                int u__2 = 1;
                int krow_2 = 0;
                int page_head_2 = meta_pg[u__2];
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + 16, 16384);
                        tma_3d_gmem2smem(kv8_addr + 32768, (&K), 0, krow_2, page_head_2, kv_full_addr + 16);
                        tma_3d_gmem2smem(kv8_addr + 32768 + 8192, (&V), 0, krow_2, page_head_2, kv_full_addr + 16);
                    }
                }
            }
            if (nhalf > 3) {
                int u__3 = 1;
                int krow_3 = 64;
                int page_head_3 = meta_pg[u__3];
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + 24, 16384);
                        tma_3d_gmem2smem(kv8_addr + 49152, (&K), 0, krow_3, page_head_3, kv_full_addr + 24);
                        tma_3d_gmem2smem(kv8_addr + 49152 + 8192, (&V), 0, krow_3, page_head_3, kv_full_addr + 24);
                    }
                }
            }
            int skey = warp_1 * 16 + 8 * (lane_0 >> 3 & 1) + (lane_0 & 7);
            int shalf = 8 * (lane_0 >> 4);
            unsigned int dk[32];
            unsigned int dv[32];
            int first_h = 1;
            uint64_t _wgmma_desc_1 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
            #pragma unroll 1
            for (int i = 0; i < nhalf; i++) {
                int stage = i % 4;
                int phase = i / 4 % 2;
                int fb = i % 2;
                mbarrier_wait(kv_full_addr + (stage) * 8, phase);
                asm volatile("" : "+r"(dk[0]) :: "memory");
                asm volatile("" : "+r"(dk[1]) :: "memory");
                asm volatile("" : "+r"(dk[2]) :: "memory");
                asm volatile("" : "+r"(dk[3]) :: "memory");
                asm volatile("" : "+r"(dk[4]) :: "memory");
                asm volatile("" : "+r"(dk[5]) :: "memory");
                asm volatile("" : "+r"(dk[6]) :: "memory");
                asm volatile("" : "+r"(dk[7]) :: "memory");
                asm volatile("" : "+r"(dk[8]) :: "memory");
                asm volatile("" : "+r"(dk[9]) :: "memory");
                asm volatile("" : "+r"(dk[10]) :: "memory");
                asm volatile("" : "+r"(dk[11]) :: "memory");
                asm volatile("" : "+r"(dk[12]) :: "memory");
                asm volatile("" : "+r"(dk[13]) :: "memory");
                asm volatile("" : "+r"(dk[14]) :: "memory");
                asm volatile("" : "+r"(dk[15]) :: "memory");
                asm volatile("" : "+r"(dk[16]) :: "memory");
                asm volatile("" : "+r"(dk[17]) :: "memory");
                asm volatile("" : "+r"(dk[18]) :: "memory");
                asm volatile("" : "+r"(dk[19]) :: "memory");
                asm volatile("" : "+r"(dk[20]) :: "memory");
                asm volatile("" : "+r"(dk[21]) :: "memory");
                asm volatile("" : "+r"(dk[22]) :: "memory");
                asm volatile("" : "+r"(dk[23]) :: "memory");
                asm volatile("" : "+r"(dk[24]) :: "memory");
                asm volatile("" : "+r"(dk[25]) :: "memory");
                asm volatile("" : "+r"(dk[26]) :: "memory");
                asm volatile("" : "+r"(dk[27]) :: "memory");
                asm volatile("" : "+r"(dk[28]) :: "memory");
                asm volatile("" : "+r"(dk[29]) :: "memory");
                asm volatile("" : "+r"(dk[30]) :: "memory");
                asm volatile("" : "+r"(dk[31]) :: "memory");
                asm volatile("" : "+r"(dv[0]) :: "memory");
                asm volatile("" : "+r"(dv[1]) :: "memory");
                asm volatile("" : "+r"(dv[2]) :: "memory");
                asm volatile("" : "+r"(dv[3]) :: "memory");
                asm volatile("" : "+r"(dv[4]) :: "memory");
                asm volatile("" : "+r"(dv[5]) :: "memory");
                asm volatile("" : "+r"(dv[6]) :: "memory");
                asm volatile("" : "+r"(dv[7]) :: "memory");
                asm volatile("" : "+r"(dv[8]) :: "memory");
                asm volatile("" : "+r"(dv[9]) :: "memory");
                asm volatile("" : "+r"(dv[10]) :: "memory");
                asm volatile("" : "+r"(dv[11]) :: "memory");
                asm volatile("" : "+r"(dv[12]) :: "memory");
                asm volatile("" : "+r"(dv[13]) :: "memory");
                asm volatile("" : "+r"(dv[14]) :: "memory");
                asm volatile("" : "+r"(dv[15]) :: "memory");
                asm volatile("" : "+r"(dv[16]) :: "memory");
                asm volatile("" : "+r"(dv[17]) :: "memory");
                asm volatile("" : "+r"(dv[18]) :: "memory");
                asm volatile("" : "+r"(dv[19]) :: "memory");
                asm volatile("" : "+r"(dv[20]) :: "memory");
                asm volatile("" : "+r"(dv[21]) :: "memory");
                asm volatile("" : "+r"(dv[22]) :: "memory");
                asm volatile("" : "+r"(dv[23]) :: "memory");
                asm volatile("" : "+r"(dv[24]) :: "memory");
                asm volatile("" : "+r"(dv[25]) :: "memory");
                asm volatile("" : "+r"(dv[26]) :: "memory");
                asm volatile("" : "+r"(dv[27]) :: "memory");
                asm volatile("" : "+r"(dv[28]) :: "memory");
                asm volatile("" : "+r"(dv[29]) :: "memory");
                asm volatile("" : "+r"(dv[30]) :: "memory");
                asm volatile("" : "+r"(dv[31]) :: "memory");
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                uint64_t _wgmma_desc_2 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
                    : "+r"(dk[0]), "+r"(dk[1]), "+r"(dk[2]), "+r"(dk[3]), "+r"(dk[4]), "+r"(dk[5]), "+r"(dk[6]), "+r"(dk[7])
                    : "l"(_wgmma_a_0_0), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
                    : "+r"(dk[8]), "+r"(dk[(8) + 1]), "+r"(dk[(8) + 2]), "+r"(dk[(8) + 3]), "+r"(dk[(8) + 4]), "+r"(dk[(8) + 5]), "+r"(dk[(8) + 6]), "+r"(dk[(8) + 7])
                    : "l"(_wgmma_a_0_0 + 2), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
                    : "+r"(dk[16]), "+r"(dk[(16) + 1]), "+r"(dk[(16) + 2]), "+r"(dk[(16) + 3]), "+r"(dk[(16) + 4]), "+r"(dk[(16) + 5]), "+r"(dk[(16) + 6]), "+r"(dk[(16) + 7])
                    : "l"(_wgmma_a_0_0 + 4), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 0, 1, 1;\n}\n"
                    : "+r"(dk[24]), "+r"(dk[(24) + 1]), "+r"(dk[(24) + 2]), "+r"(dk[(24) + 3]), "+r"(dk[(24) + 4]), "+r"(dk[(24) + 5]), "+r"(dk[(24) + 6]), "+r"(dk[(24) + 7])
                    : "l"(_wgmma_a_0_0 + 6), "l"(_wgmma_b_0_1)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                uint64_t _wgmma_desc_3 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 16384) + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t _wgmma_b_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
                    : "+r"(dv[0]), "+r"(dv[1]), "+r"(dv[2]), "+r"(dv[3]), "+r"(dv[4]), "+r"(dv[5]), "+r"(dv[6]), "+r"(dv[7]), "+r"(dv[8]), "+r"(dv[9]), "+r"(dv[10]), "+r"(dv[11]), "+r"(dv[12]), "+r"(dv[13]), "+r"(dv[14]), "+r"(dv[15])
                    : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
                    : "+r"(dv[0]), "+r"(dv[1]), "+r"(dv[2]), "+r"(dv[3]), "+r"(dv[4]), "+r"(dv[5]), "+r"(dv[6]), "+r"(dv[7]), "+r"(dv[8]), "+r"(dv[9]), "+r"(dv[10]), "+r"(dv[11]), "+r"(dv[12]), "+r"(dv[13]), "+r"(dv[14]), "+r"(dv[15])
                    : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 0, 1, 1;\n}\n"
                    : "+r"(dv[16]), "+r"(dv[(16) + 1]), "+r"(dv[(16) + 2]), "+r"(dv[(16) + 3]), "+r"(dv[(16) + 4]), "+r"(dv[(16) + 5]), "+r"(dv[(16) + 6]), "+r"(dv[(16) + 7]), "+r"(dv[(16) + 8]), "+r"(dv[(16) + 9]), "+r"(dv[(16) + 10]), "+r"(dv[(16) + 11]), "+r"(dv[(16) + 12]), "+r"(dv[(16) + 13]), "+r"(dv[(16) + 14]), "+r"(dv[(16) + 15])
                    : "l"(_wgmma_b_0_1), "l"(_wgmma_b_0_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k32.f16.e4m3.e4m3 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1;\n}\n"
                    : "+r"(dv[16]), "+r"(dv[(16) + 1]), "+r"(dv[(16) + 2]), "+r"(dv[(16) + 3]), "+r"(dv[(16) + 4]), "+r"(dv[(16) + 5]), "+r"(dv[(16) + 6]), "+r"(dv[(16) + 7]), "+r"(dv[(16) + 8]), "+r"(dv[(16) + 9]), "+r"(dv[(16) + 10]), "+r"(dv[(16) + 11]), "+r"(dv[(16) + 12]), "+r"(dv[(16) + 13]), "+r"(dv[(16) + 14]), "+r"(dv[(16) + 15])
                    : "l"(_wgmma_b_0_1 + 2), "l"(_wgmma_b_0_2 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                asm volatile("" : "+r"(dk[0]) :: "memory");
                asm volatile("" : "+r"(dk[1]) :: "memory");
                asm volatile("" : "+r"(dk[2]) :: "memory");
                asm volatile("" : "+r"(dk[3]) :: "memory");
                asm volatile("" : "+r"(dk[4]) :: "memory");
                asm volatile("" : "+r"(dk[5]) :: "memory");
                asm volatile("" : "+r"(dk[6]) :: "memory");
                asm volatile("" : "+r"(dk[7]) :: "memory");
                asm volatile("" : "+r"(dk[8]) :: "memory");
                asm volatile("" : "+r"(dk[9]) :: "memory");
                asm volatile("" : "+r"(dk[10]) :: "memory");
                asm volatile("" : "+r"(dk[11]) :: "memory");
                asm volatile("" : "+r"(dk[12]) :: "memory");
                asm volatile("" : "+r"(dk[13]) :: "memory");
                asm volatile("" : "+r"(dk[14]) :: "memory");
                asm volatile("" : "+r"(dk[15]) :: "memory");
                asm volatile("" : "+r"(dk[16]) :: "memory");
                asm volatile("" : "+r"(dk[17]) :: "memory");
                asm volatile("" : "+r"(dk[18]) :: "memory");
                asm volatile("" : "+r"(dk[19]) :: "memory");
                asm volatile("" : "+r"(dk[20]) :: "memory");
                asm volatile("" : "+r"(dk[21]) :: "memory");
                asm volatile("" : "+r"(dk[22]) :: "memory");
                asm volatile("" : "+r"(dk[23]) :: "memory");
                asm volatile("" : "+r"(dk[24]) :: "memory");
                asm volatile("" : "+r"(dk[25]) :: "memory");
                asm volatile("" : "+r"(dk[26]) :: "memory");
                asm volatile("" : "+r"(dk[27]) :: "memory");
                asm volatile("" : "+r"(dk[28]) :: "memory");
                asm volatile("" : "+r"(dk[29]) :: "memory");
                asm volatile("" : "+r"(dk[30]) :: "memory");
                asm volatile("" : "+r"(dk[31]) :: "memory");
                if (i >= 2) {
                    mbarrier_wait(f16_empty_addr + (fb) * 8, (i / 2 + 1) % 2);
                }
                int kbase = k16_addr + (unsigned int)(fb * 16384);
                int ka = kbase + skey * 128 + (shalf * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_4 = static_cast<uint32_t>(ka);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&dk[0])), "r"(*reinterpret_cast<const uint32_t*>(&dk[1])), "r"(*reinterpret_cast<const uint32_t*>(&dk[2])), "r"(*reinterpret_cast<const uint32_t*>(&dk[3]))
                    : "memory");
                int ka_0 = kbase + skey * 128 + ((16 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(ka_0);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&dk[4])), "r"(*reinterpret_cast<const uint32_t*>(&dk[5])), "r"(*reinterpret_cast<const uint32_t*>(&dk[6])), "r"(*reinterpret_cast<const uint32_t*>(&dk[7]))
                    : "memory");
                int ka_1 = kbase + skey * 128 + ((32 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_6 = static_cast<uint32_t>(ka_1);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&dk[8])), "r"(*reinterpret_cast<const uint32_t*>(&dk[9])), "r"(*reinterpret_cast<const uint32_t*>(&dk[10])), "r"(*reinterpret_cast<const uint32_t*>(&dk[11]))
                    : "memory");
                int ka_2 = kbase + skey * 128 + ((48 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_7 = static_cast<uint32_t>(ka_2);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&dk[12])), "r"(*reinterpret_cast<const uint32_t*>(&dk[13])), "r"(*reinterpret_cast<const uint32_t*>(&dk[14])), "r"(*reinterpret_cast<const uint32_t*>(&dk[15]))
                    : "memory");
                int ka_3 = kbase + 8192 + skey * 128 + (shalf * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_8 = static_cast<uint32_t>(ka_3);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&dk[16])), "r"(*reinterpret_cast<const uint32_t*>(&dk[17])), "r"(*reinterpret_cast<const uint32_t*>(&dk[18])), "r"(*reinterpret_cast<const uint32_t*>(&dk[19]))
                    : "memory");
                int ka_4 = kbase + 8192 + skey * 128 + ((16 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_9 = static_cast<uint32_t>(ka_4);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_9), "r"(*reinterpret_cast<const uint32_t*>(&dk[20])), "r"(*reinterpret_cast<const uint32_t*>(&dk[21])), "r"(*reinterpret_cast<const uint32_t*>(&dk[22])), "r"(*reinterpret_cast<const uint32_t*>(&dk[23]))
                    : "memory");
                int ka_5 = kbase + 8192 + skey * 128 + ((32 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_10 = static_cast<uint32_t>(ka_5);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&dk[24])), "r"(*reinterpret_cast<const uint32_t*>(&dk[25])), "r"(*reinterpret_cast<const uint32_t*>(&dk[26])), "r"(*reinterpret_cast<const uint32_t*>(&dk[27]))
                    : "memory");
                int ka_6 = kbase + 8192 + skey * 128 + ((48 + shalf) * 2 ^ (skey & 7) << 4);
                uint32_t _stmatrix_addr_11 = static_cast<uint32_t>(ka_6);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&dk[28])), "r"(*reinterpret_cast<const uint32_t*>(&dk[29])), "r"(*reinterpret_cast<const uint32_t*>(&dk[30])), "r"(*reinterpret_cast<const uint32_t*>(&dk[31]))
                    : "memory");
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                asm volatile("" : "+r"(dv[0]) :: "memory");
                asm volatile("" : "+r"(dv[1]) :: "memory");
                asm volatile("" : "+r"(dv[2]) :: "memory");
                asm volatile("" : "+r"(dv[3]) :: "memory");
                asm volatile("" : "+r"(dv[4]) :: "memory");
                asm volatile("" : "+r"(dv[5]) :: "memory");
                asm volatile("" : "+r"(dv[6]) :: "memory");
                asm volatile("" : "+r"(dv[7]) :: "memory");
                asm volatile("" : "+r"(dv[8]) :: "memory");
                asm volatile("" : "+r"(dv[9]) :: "memory");
                asm volatile("" : "+r"(dv[10]) :: "memory");
                asm volatile("" : "+r"(dv[11]) :: "memory");
                asm volatile("" : "+r"(dv[12]) :: "memory");
                asm volatile("" : "+r"(dv[13]) :: "memory");
                asm volatile("" : "+r"(dv[14]) :: "memory");
                asm volatile("" : "+r"(dv[15]) :: "memory");
                asm volatile("" : "+r"(dv[16]) :: "memory");
                asm volatile("" : "+r"(dv[17]) :: "memory");
                asm volatile("" : "+r"(dv[18]) :: "memory");
                asm volatile("" : "+r"(dv[19]) :: "memory");
                asm volatile("" : "+r"(dv[20]) :: "memory");
                asm volatile("" : "+r"(dv[21]) :: "memory");
                asm volatile("" : "+r"(dv[22]) :: "memory");
                asm volatile("" : "+r"(dv[23]) :: "memory");
                asm volatile("" : "+r"(dv[24]) :: "memory");
                asm volatile("" : "+r"(dv[25]) :: "memory");
                asm volatile("" : "+r"(dv[26]) :: "memory");
                asm volatile("" : "+r"(dv[27]) :: "memory");
                asm volatile("" : "+r"(dv[28]) :: "memory");
                asm volatile("" : "+r"(dv[29]) :: "memory");
                asm volatile("" : "+r"(dv[30]) :: "memory");
                asm volatile("" : "+r"(dv[31]) :: "memory");
                int nxt = i + 4;
                if (nxt < nhalf) {
                    int u__4 = nxt >> 1;
                    int krow_4 = (nxt & 1) * 64;
                    int page_head_4 = meta_pg[u__4];
                    if (warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 16384);
                            tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 16384), (&K), 0, krow_4, page_head_4, kv_full_addr + (stage) * 8);
                            tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 16384) + 8192, (&V), 0, krow_4, page_head_4, kv_full_addr + (stage) * 8);
                        }
                    }
                }
                int vlim_h = sk_b - meta_blk[i >> 1] * 128 - (i & 1) * 64;
                if (vlim_h < 64) {
                    int k0 = 2 * cq_c;
                    unsigned int mh = 4294967295;
                    if (vlim_h <= k0 + 1) {
                        mh = 65535;
                    }
                    if (k0 >= vlim_h) {
                        mh = 0;
                    }
                    dv[0] = dv[0] & mh;
                    dv[1] = dv[1] & mh;
                    dv[16] = dv[16] & mh;
                    dv[17] = dv[17] & mh;
                    int k0_0 = 8 + 2 * cq_c;
                    unsigned int mh_1 = 4294967295;
                    if (vlim_h <= k0_0 + 1) {
                        mh_1 = 65535;
                    }
                    if (k0_0 >= vlim_h) {
                        mh_1 = 0;
                    }
                    dv[2] = dv[2] & mh_1;
                    dv[3] = dv[3] & mh_1;
                    dv[18] = dv[18] & mh_1;
                    dv[19] = dv[19] & mh_1;
                    int k0_2 = 16 + 2 * cq_c;
                    unsigned int mh_3 = 4294967295;
                    if (vlim_h <= k0_2 + 1) {
                        mh_3 = 65535;
                    }
                    if (k0_2 >= vlim_h) {
                        mh_3 = 0;
                    }
                    dv[4] = dv[4] & mh_3;
                    dv[5] = dv[5] & mh_3;
                    dv[20] = dv[20] & mh_3;
                    dv[21] = dv[21] & mh_3;
                    int k0_4 = 24 + 2 * cq_c;
                    unsigned int mh_5 = 4294967295;
                    if (vlim_h <= k0_4 + 1) {
                        mh_5 = 65535;
                    }
                    if (k0_4 >= vlim_h) {
                        mh_5 = 0;
                    }
                    dv[6] = dv[6] & mh_5;
                    dv[7] = dv[7] & mh_5;
                    dv[22] = dv[22] & mh_5;
                    dv[23] = dv[23] & mh_5;
                    int k0_6 = 32 + 2 * cq_c;
                    unsigned int mh_7 = 4294967295;
                    if (vlim_h <= k0_6 + 1) {
                        mh_7 = 65535;
                    }
                    if (k0_6 >= vlim_h) {
                        mh_7 = 0;
                    }
                    dv[8] = dv[8] & mh_7;
                    dv[9] = dv[9] & mh_7;
                    dv[24] = dv[24] & mh_7;
                    dv[25] = dv[25] & mh_7;
                    int k0_8 = 40 + 2 * cq_c;
                    unsigned int mh_9 = 4294967295;
                    if (vlim_h <= k0_8 + 1) {
                        mh_9 = 65535;
                    }
                    if (k0_8 >= vlim_h) {
                        mh_9 = 0;
                    }
                    dv[10] = dv[10] & mh_9;
                    dv[11] = dv[11] & mh_9;
                    dv[26] = dv[26] & mh_9;
                    dv[27] = dv[27] & mh_9;
                    int k0_10 = 48 + 2 * cq_c;
                    unsigned int mh_11 = 4294967295;
                    if (vlim_h <= k0_10 + 1) {
                        mh_11 = 65535;
                    }
                    if (k0_10 >= vlim_h) {
                        mh_11 = 0;
                    }
                    dv[12] = dv[12] & mh_11;
                    dv[13] = dv[13] & mh_11;
                    dv[28] = dv[28] & mh_11;
                    dv[29] = dv[29] & mh_11;
                    int k0_12 = 56 + 2 * cq_c;
                    unsigned int mh_13 = 4294967295;
                    if (vlim_h <= k0_12 + 1) {
                        mh_13 = 65535;
                    }
                    if (k0_12 >= vlim_h) {
                        mh_13 = 0;
                    }
                    dv[14] = dv[14] & mh_13;
                    dv[15] = dv[15] & mh_13;
                    dv[30] = dv[30] & mh_13;
                    dv[31] = dv[31] & mh_13;
                }
                int vbase = v16t_addr + (unsigned int)(fb * 16384);
                int sdim = skey;
                int va = vbase + sdim * 128 + (shalf * 2 ^ (sdim & 7) << 4);
                uint32_t _stmatrix_addr_12 = static_cast<uint32_t>(va);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_12), "r"(*reinterpret_cast<const uint32_t*>(&dv[0])), "r"(*reinterpret_cast<const uint32_t*>(&dv[1])), "r"(*reinterpret_cast<const uint32_t*>(&dv[2])), "r"(*reinterpret_cast<const uint32_t*>(&dv[3]))
                    : "memory");
                int va_7 = vbase + sdim * 128 + ((16 + shalf) * 2 ^ (sdim & 7) << 4);
                uint32_t _stmatrix_addr_13 = static_cast<uint32_t>(va_7);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&dv[4])), "r"(*reinterpret_cast<const uint32_t*>(&dv[5])), "r"(*reinterpret_cast<const uint32_t*>(&dv[6])), "r"(*reinterpret_cast<const uint32_t*>(&dv[7]))
                    : "memory");
                int va_8 = vbase + sdim * 128 + ((32 + shalf) * 2 ^ (sdim & 7) << 4);
                uint32_t _stmatrix_addr_14 = static_cast<uint32_t>(va_8);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&dv[8])), "r"(*reinterpret_cast<const uint32_t*>(&dv[9])), "r"(*reinterpret_cast<const uint32_t*>(&dv[10])), "r"(*reinterpret_cast<const uint32_t*>(&dv[11]))
                    : "memory");
                int va_9 = vbase + sdim * 128 + ((48 + shalf) * 2 ^ (sdim & 7) << 4);
                uint32_t _stmatrix_addr_15 = static_cast<uint32_t>(va_9);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&dv[12])), "r"(*reinterpret_cast<const uint32_t*>(&dv[13])), "r"(*reinterpret_cast<const uint32_t*>(&dv[14])), "r"(*reinterpret_cast<const uint32_t*>(&dv[15]))
                    : "memory");
                int sdim_10 = 64 + skey;
                int va_11 = vbase + sdim_10 * 128 + (shalf * 2 ^ (sdim_10 & 7) << 4);
                uint32_t _stmatrix_addr_16 = static_cast<uint32_t>(va_11);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_16), "r"(*reinterpret_cast<const uint32_t*>(&dv[16])), "r"(*reinterpret_cast<const uint32_t*>(&dv[17])), "r"(*reinterpret_cast<const uint32_t*>(&dv[18])), "r"(*reinterpret_cast<const uint32_t*>(&dv[19]))
                    : "memory");
                int va_12 = vbase + sdim_10 * 128 + ((16 + shalf) * 2 ^ (sdim_10 & 7) << 4);
                uint32_t _stmatrix_addr_17 = static_cast<uint32_t>(va_12);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_17), "r"(*reinterpret_cast<const uint32_t*>(&dv[20])), "r"(*reinterpret_cast<const uint32_t*>(&dv[21])), "r"(*reinterpret_cast<const uint32_t*>(&dv[22])), "r"(*reinterpret_cast<const uint32_t*>(&dv[23]))
                    : "memory");
                int va_13 = vbase + sdim_10 * 128 + ((32 + shalf) * 2 ^ (sdim_10 & 7) << 4);
                uint32_t _stmatrix_addr_18 = static_cast<uint32_t>(va_13);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_18), "r"(*reinterpret_cast<const uint32_t*>(&dv[24])), "r"(*reinterpret_cast<const uint32_t*>(&dv[25])), "r"(*reinterpret_cast<const uint32_t*>(&dv[26])), "r"(*reinterpret_cast<const uint32_t*>(&dv[27]))
                    : "memory");
                int va_14 = vbase + sdim_10 * 128 + ((48 + shalf) * 2 ^ (sdim_10 & 7) << 4);
                uint32_t _stmatrix_addr_19 = static_cast<uint32_t>(va_14);
                asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_19), "r"(*reinterpret_cast<const uint32_t*>(&dv[28])), "r"(*reinterpret_cast<const uint32_t*>(&dv[29])), "r"(*reinterpret_cast<const uint32_t*>(&dv[30])), "r"(*reinterpret_cast<const uint32_t*>(&dv[31]))
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(f16_full_addr + (fb) * 8);
                first_h = 0;
            }
        }
    }
    // ---- Role: cons ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // cons_main
            unsigned long long cstamps[14];
            cstamps[0] = 0;
            cstamps[1] = 0;
            cstamps[2] = 0;
            cstamps[3] = 0;
            cstamps[4] = 0;
            cstamps[5] = 0;
            cstamps[6] = 0;
            cstamps[7] = 0;
            cstamps[8] = 0;
            cstamps[9] = 0;
            cstamps[10] = 0;
            cstamps[11] = 0;
            cstamps[12] = 0;
            cstamps[13] = 0;
            int gi_c = blockIdx.x;
            int kv_head_c = blockIdx.y;
            int tid_c = threadIdx.x;
            int lane_c = lane;
            int warp_c = warp;
            if (warp_c == 0) {
                int found_1 = 0;
                int base_g_1 = 0;
                int sel_b_1 = 0;
                int sel_tok0_1 = 0;
                int sel_ntok_1 = 0;
                int sel_cu_1 = 0;
                int sel_qlen_1 = 0;
                int nchunks_1 = (batch + 31) / 32;
                #pragma unroll 1
                for (int c_1 = 0; c_1 < nchunks_1; c_1++) {
                    if (found_1 == 0) {
                        int bb_1 = c_1 * 32 + lane_c;
                        int cnt_b_1 = 0;
                        int cu_lo_1 = 0;
                        int cu_hi_1 = 0;
                        if (bb_1 < batch) {
                            cu_lo_1 = cu_seqlens_q[bb_1];
                            cu_hi_1 = cu_seqlens_q[bb_1 + 1];
                            cnt_b_1 = (cu_hi_1 - cu_lo_1 + 8 - 1) / 8;
                        }
                        int incl_1 = cnt_b_1;
                        int _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, incl_1, 1, 32);
                        int up_4 = _shfl_up_10;
                        if (lane_c >= 1) {
                            incl_1 = incl_1 + up_4;
                        }
                        int _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, incl_1, 2, 32);
                        int up_0_1 = _shfl_up_11;
                        if (lane_c >= 2) {
                            incl_1 = incl_1 + up_0_1;
                        }
                        int _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, incl_1, 4, 32);
                        int up_1_1 = _shfl_up_12;
                        if (lane_c >= 4) {
                            incl_1 = incl_1 + up_1_1;
                        }
                        int _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, incl_1, 8, 32);
                        int up_2_1 = _shfl_up_13;
                        if (lane_c >= 8) {
                            incl_1 = incl_1 + up_2_1;
                        }
                        int _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, incl_1, 16, 32);
                        int up_3_1 = _shfl_up_14;
                        if (lane_c >= 16) {
                            incl_1 = incl_1 + up_3_1;
                        }
                        int excl_1 = incl_1 - cnt_b_1;
                        int hit_1 = 0;
                        if (bb_1 < batch) {
                            if (gi_c >= base_g_1 + excl_1) {
                                if (gi_c < base_g_1 + incl_1) {
                                    hit_1 = 1;
                                }
                            }
                        }
                        unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit_1 != 0);
                        unsigned int bal_1 = _vote_1;
                        if (bal_1 != 0) {
                            int _ffs_1 = __ffs(bal_1);
                            int src_l_1 = _ffs_1 - 1;
                            sel_b_1 = c_1 * 32 + src_l_1;
                            int _shfl_5 = __shfl_sync(0xFFFFFFFF, cu_lo_1, src_l_1);
                            sel_cu_1 = _shfl_5;
                            int _shfl_6 = __shfl_sync(0xFFFFFFFF, cu_hi_1, src_l_1);
                            int hi_s_1 = _shfl_6;
                            int _shfl_7 = __shfl_sync(0xFFFFFFFF, excl_1, src_l_1);
                            int ex_s_1 = _shfl_7;
                            int _shfl_8 = __shfl_sync(0xFFFFFFFF, cnt_b_1, src_l_1);
                            int cnt_s_1 = _shfl_8;
                            int lg_1 = gi_c - base_g_1 - ex_s_1;
                            lg_1 = cnt_s_1 - 1 - lg_1;
                            sel_tok0_1 = lg_1 * 8;
                            sel_qlen_1 = hi_s_1 - sel_cu_1;
                            sel_ntok_1 = sel_qlen_1 - sel_tok0_1;
                            if (sel_ntok_1 > 8) {
                                sel_ntok_1 = 8;
                            }
                            found_1 = 1;
                        }
                        int _shfl_9 = __shfl_sync(0xFFFFFFFF, incl_1, 31);
                        base_g_1 = base_g_1 + _shfl_9;
                    }
                }
                if (lane_c == 0) {
                    int qoff_1 = 0;
                    int sk_sel_1 = seqused_k[sel_b_1];
                    if (found_1 != 0) {
                        if (use_q_offset != 0) {
                            qoff_1 = q_offset[sel_b_1];
                        }
                        if (use_q_offset == 0) {
                            qoff_1 = sk_sel_1 - sel_qlen_1;
                        }
                    }
                    hdr[0] = sel_b_1;
                    hdr[1] = sel_tok0_1;
                    hdr[2] = sel_ntok_1;
                    hdr[3] = sel_cu_1;
                    hdr[4] = qoff_1 + sel_tok0_1;
                    hdr[5] = sk_sel_1;
                }
            }
            if (warp_c != 0) {
                int sid_1 = tid_c - 32;
                int ngrp_z_1 = (max_pages + 31) / 32;
                unsigned int zz_1[4];
                zz_1[0] = 0;
                zz_1[1] = 0;
                zz_1[2] = 0;
                zz_1[3] = 0;
                int grp_z_1 = sid_1;
                if (grp_z_1 < ngrp_z_1) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z_1 * 32)), "r"(zz_1[0]), "r"(zz_1[1]), "r"(zz_1[2]), "r"(zz_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(mark_addr + (unsigned int)(grp_z_1 * 32 + 16)), "r"(zz_1[0]), "r"(zz_1[1]), "r"(zz_1[2]), "r"(zz_1[3]) : "memory");
                }
                int ich_lin_1 = sid_1;
                if (ich_lin_1 < 512) {
                    int irow_2 = ich_lin_1 / 8;
                    int ich_2 = ich_lin_1 - irow_2 * 8;
                    int iword_2 = irow_2 / 4;
                    unsigned int ione_2 = 56 << 8 * (irow_2 - iword_2 * 4);
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
                int ich_lin_0_1 = sid_1 + 352;
                if (ich_lin_0_1 < 512) {
                    int irow_3 = ich_lin_0_1 / 8;
                    int ich_3 = ich_lin_0_1 - irow_3 * 8;
                    int iword_3 = irow_3 / 4;
                    unsigned int ione_3 = 56 << 8 * (irow_3 - iword_3 * 4);
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
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int tok0_u_1 = hdr[1];
            int n_tok_u_1 = hdr[2];
            int cu_u_1 = hdr[3];
            int b_u_1 = hdr[0];
            int blk_e_1[1];
            int e_1 = tid_c;
            blk_e_1[0] = -1;
            if (e_1 < 128) {
                int tl_e_1 = e_1 / 16;
                int k_e_1 = e_1 - tl_e_1 * 16;
                if (tl_e_1 < n_tok_u_1) {
                    int bv_1 = q2k_indices[(kv_head_c * total_q + cu_u_1 + tok0_u_1 + tl_e_1) * 16 + k_e_1];
                    if (bv_1 >= 0) {
                        if (bv_1 < max_pages) {
                            blk_e_1[0] = bv_1;
                        }
                    }
                }
            }
            if (blk_e_1[0] >= 0) {
                {
                    uint32_t _byte_0 = static_cast<uint32_t>(1) & 0xFFu;
                    uint32_t _addr_0 = static_cast<uint32_t>(mark_addr + (unsigned int)blk_e_1[0]);
                    asm volatile("st.shared.u8 [%0], %1;" :: "r"(_addr_0), "r"(_byte_0) : "memory");
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int ngrp_1 = (max_pages + 31) / 32;
            unsigned int mw_1[8];
            mw_1[0] = 0;
            mw_1[1] = 0;
            mw_1[2] = 0;
            mw_1[3] = 0;
            mw_1[4] = 0;
            mw_1[5] = 0;
            mw_1[6] = 0;
            mw_1[7] = 0;
            int c_t_1 = 0;
            int incl_t_1 = 0;
            if (tid_c < 128) {
                if (tid_c < ngrp_1) {
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 3]))
                        : "r"(mark_addr + (unsigned int)(tid_c * 32)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&mw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 3]))
                        : "r"(mark_addr + (unsigned int)(tid_c * 32) + 16));
                }
                int _popc_24 = __popc(mw_1[0]);
                c_t_1 = c_t_1 + _popc_24;
                int _popc_25 = __popc(mw_1[1]);
                c_t_1 = c_t_1 + _popc_25;
                int _popc_26 = __popc(mw_1[2]);
                c_t_1 = c_t_1 + _popc_26;
                int _popc_27 = __popc(mw_1[3]);
                c_t_1 = c_t_1 + _popc_27;
                int _popc_28 = __popc(mw_1[4]);
                c_t_1 = c_t_1 + _popc_28;
                int _popc_29 = __popc(mw_1[5]);
                c_t_1 = c_t_1 + _popc_29;
                int _popc_30 = __popc(mw_1[6]);
                c_t_1 = c_t_1 + _popc_30;
                int _popc_31 = __popc(mw_1[7]);
                c_t_1 = c_t_1 + _popc_31;
                incl_t_1 = c_t_1;
                int _shfl_up_15 = __shfl_up_sync(0xFFFFFFFF, incl_t_1, 1, 32);
                int up2_4 = _shfl_up_15;
                if (lane_c >= 1) {
                    incl_t_1 = incl_t_1 + up2_4;
                }
                int _shfl_up_16 = __shfl_up_sync(0xFFFFFFFF, incl_t_1, 2, 32);
                int up2_0_1 = _shfl_up_16;
                if (lane_c >= 2) {
                    incl_t_1 = incl_t_1 + up2_0_1;
                }
                int _shfl_up_17 = __shfl_up_sync(0xFFFFFFFF, incl_t_1, 4, 32);
                int up2_1_1 = _shfl_up_17;
                if (lane_c >= 4) {
                    incl_t_1 = incl_t_1 + up2_1_1;
                }
                int _shfl_up_18 = __shfl_up_sync(0xFFFFFFFF, incl_t_1, 8, 32);
                int up2_2_1 = _shfl_up_18;
                if (lane_c >= 8) {
                    incl_t_1 = incl_t_1 + up2_2_1;
                }
                int _shfl_up_19 = __shfl_up_sync(0xFFFFFFFF, incl_t_1, 16, 32);
                int up2_3_1 = _shfl_up_19;
                if (lane_c >= 16) {
                    incl_t_1 = incl_t_1 + up2_3_1;
                }
                if (lane_c == 31) {
                    cnt[warp_c] = incl_t_1;
                }
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (tid_c < 128) {
                int pos_t_1 = incl_t_1 - c_t_1;
                int cw__3 = cnt[0];
                if (warp_c > 0) {
                    pos_t_1 = pos_t_1 + cw__3;
                }
                int cw__0_1 = cnt[1];
                if (warp_c > 1) {
                    pos_t_1 = pos_t_1 + cw__0_1;
                }
                int cw__1_1 = cnt[2];
                if (warp_c > 2) {
                    pos_t_1 = pos_t_1 + cw__1_1;
                }
                int cw__2_1 = cnt[3];
                if (warp_c > 3) {
                    pos_t_1 = pos_t_1 + cw__2_1;
                }
                pref[tid_c] = pos_t_1;
                if (c_t_1 != 0) {
                    if ((mw_1[0] & 1) != 0) {
                        int blk_u_32 = tid_c * 32;
                        int pg_u_32 = page_table[b_u_1 * max_pages + blk_u_32];
                        int blk_st_32 = blk_u_32;
                        if (pg_u_32 < 0) {
                            pg_u_32 = 0;
                            blk_st_32 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_32;
                        meta_pg[pos_t_1] = pg_u_32 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[0] >> 8 & 1) != 0) {
                        int blk_u_33 = tid_c * 32 + 1;
                        int pg_u_33 = page_table[b_u_1 * max_pages + blk_u_33];
                        int blk_st_33 = blk_u_33;
                        if (pg_u_33 < 0) {
                            pg_u_33 = 0;
                            blk_st_33 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_33;
                        meta_pg[pos_t_1] = pg_u_33 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[0] >> 16 & 1) != 0) {
                        int blk_u_34 = tid_c * 32 + 2;
                        int pg_u_34 = page_table[b_u_1 * max_pages + blk_u_34];
                        int blk_st_34 = blk_u_34;
                        if (pg_u_34 < 0) {
                            pg_u_34 = 0;
                            blk_st_34 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_34;
                        meta_pg[pos_t_1] = pg_u_34 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[0] >> 24 & 1) != 0) {
                        int blk_u_35 = tid_c * 32 + 3;
                        int pg_u_35 = page_table[b_u_1 * max_pages + blk_u_35];
                        int blk_st_35 = blk_u_35;
                        if (pg_u_35 < 0) {
                            pg_u_35 = 0;
                            blk_st_35 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_35;
                        meta_pg[pos_t_1] = pg_u_35 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[1] & 1) != 0) {
                        int blk_u_36 = tid_c * 32 + 4;
                        int pg_u_36 = page_table[b_u_1 * max_pages + blk_u_36];
                        int blk_st_36 = blk_u_36;
                        if (pg_u_36 < 0) {
                            pg_u_36 = 0;
                            blk_st_36 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_36;
                        meta_pg[pos_t_1] = pg_u_36 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[1] >> 8 & 1) != 0) {
                        int blk_u_37 = tid_c * 32 + 4 + 1;
                        int pg_u_37 = page_table[b_u_1 * max_pages + blk_u_37];
                        int blk_st_37 = blk_u_37;
                        if (pg_u_37 < 0) {
                            pg_u_37 = 0;
                            blk_st_37 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_37;
                        meta_pg[pos_t_1] = pg_u_37 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[1] >> 16 & 1) != 0) {
                        int blk_u_38 = tid_c * 32 + 4 + 2;
                        int pg_u_38 = page_table[b_u_1 * max_pages + blk_u_38];
                        int blk_st_38 = blk_u_38;
                        if (pg_u_38 < 0) {
                            pg_u_38 = 0;
                            blk_st_38 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_38;
                        meta_pg[pos_t_1] = pg_u_38 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[1] >> 24 & 1) != 0) {
                        int blk_u_39 = tid_c * 32 + 4 + 3;
                        int pg_u_39 = page_table[b_u_1 * max_pages + blk_u_39];
                        int blk_st_39 = blk_u_39;
                        if (pg_u_39 < 0) {
                            pg_u_39 = 0;
                            blk_st_39 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_39;
                        meta_pg[pos_t_1] = pg_u_39 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[2] & 1) != 0) {
                        int blk_u_40 = tid_c * 32 + 8;
                        int pg_u_40 = page_table[b_u_1 * max_pages + blk_u_40];
                        int blk_st_40 = blk_u_40;
                        if (pg_u_40 < 0) {
                            pg_u_40 = 0;
                            blk_st_40 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_40;
                        meta_pg[pos_t_1] = pg_u_40 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[2] >> 8 & 1) != 0) {
                        int blk_u_41 = tid_c * 32 + 8 + 1;
                        int pg_u_41 = page_table[b_u_1 * max_pages + blk_u_41];
                        int blk_st_41 = blk_u_41;
                        if (pg_u_41 < 0) {
                            pg_u_41 = 0;
                            blk_st_41 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_41;
                        meta_pg[pos_t_1] = pg_u_41 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[2] >> 16 & 1) != 0) {
                        int blk_u_42 = tid_c * 32 + 8 + 2;
                        int pg_u_42 = page_table[b_u_1 * max_pages + blk_u_42];
                        int blk_st_42 = blk_u_42;
                        if (pg_u_42 < 0) {
                            pg_u_42 = 0;
                            blk_st_42 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_42;
                        meta_pg[pos_t_1] = pg_u_42 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[2] >> 24 & 1) != 0) {
                        int blk_u_43 = tid_c * 32 + 8 + 3;
                        int pg_u_43 = page_table[b_u_1 * max_pages + blk_u_43];
                        int blk_st_43 = blk_u_43;
                        if (pg_u_43 < 0) {
                            pg_u_43 = 0;
                            blk_st_43 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_43;
                        meta_pg[pos_t_1] = pg_u_43 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[3] & 1) != 0) {
                        int blk_u_44 = tid_c * 32 + 12;
                        int pg_u_44 = page_table[b_u_1 * max_pages + blk_u_44];
                        int blk_st_44 = blk_u_44;
                        if (pg_u_44 < 0) {
                            pg_u_44 = 0;
                            blk_st_44 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_44;
                        meta_pg[pos_t_1] = pg_u_44 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[3] >> 8 & 1) != 0) {
                        int blk_u_45 = tid_c * 32 + 12 + 1;
                        int pg_u_45 = page_table[b_u_1 * max_pages + blk_u_45];
                        int blk_st_45 = blk_u_45;
                        if (pg_u_45 < 0) {
                            pg_u_45 = 0;
                            blk_st_45 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_45;
                        meta_pg[pos_t_1] = pg_u_45 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[3] >> 16 & 1) != 0) {
                        int blk_u_46 = tid_c * 32 + 12 + 2;
                        int pg_u_46 = page_table[b_u_1 * max_pages + blk_u_46];
                        int blk_st_46 = blk_u_46;
                        if (pg_u_46 < 0) {
                            pg_u_46 = 0;
                            blk_st_46 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_46;
                        meta_pg[pos_t_1] = pg_u_46 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[3] >> 24 & 1) != 0) {
                        int blk_u_47 = tid_c * 32 + 12 + 3;
                        int pg_u_47 = page_table[b_u_1 * max_pages + blk_u_47];
                        int blk_st_47 = blk_u_47;
                        if (pg_u_47 < 0) {
                            pg_u_47 = 0;
                            blk_st_47 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_47;
                        meta_pg[pos_t_1] = pg_u_47 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[4] & 1) != 0) {
                        int blk_u_48 = tid_c * 32 + 16;
                        int pg_u_48 = page_table[b_u_1 * max_pages + blk_u_48];
                        int blk_st_48 = blk_u_48;
                        if (pg_u_48 < 0) {
                            pg_u_48 = 0;
                            blk_st_48 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_48;
                        meta_pg[pos_t_1] = pg_u_48 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[4] >> 8 & 1) != 0) {
                        int blk_u_49 = tid_c * 32 + 16 + 1;
                        int pg_u_49 = page_table[b_u_1 * max_pages + blk_u_49];
                        int blk_st_49 = blk_u_49;
                        if (pg_u_49 < 0) {
                            pg_u_49 = 0;
                            blk_st_49 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_49;
                        meta_pg[pos_t_1] = pg_u_49 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[4] >> 16 & 1) != 0) {
                        int blk_u_50 = tid_c * 32 + 16 + 2;
                        int pg_u_50 = page_table[b_u_1 * max_pages + blk_u_50];
                        int blk_st_50 = blk_u_50;
                        if (pg_u_50 < 0) {
                            pg_u_50 = 0;
                            blk_st_50 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_50;
                        meta_pg[pos_t_1] = pg_u_50 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[4] >> 24 & 1) != 0) {
                        int blk_u_51 = tid_c * 32 + 16 + 3;
                        int pg_u_51 = page_table[b_u_1 * max_pages + blk_u_51];
                        int blk_st_51 = blk_u_51;
                        if (pg_u_51 < 0) {
                            pg_u_51 = 0;
                            blk_st_51 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_51;
                        meta_pg[pos_t_1] = pg_u_51 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[5] & 1) != 0) {
                        int blk_u_52 = tid_c * 32 + 20;
                        int pg_u_52 = page_table[b_u_1 * max_pages + blk_u_52];
                        int blk_st_52 = blk_u_52;
                        if (pg_u_52 < 0) {
                            pg_u_52 = 0;
                            blk_st_52 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_52;
                        meta_pg[pos_t_1] = pg_u_52 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[5] >> 8 & 1) != 0) {
                        int blk_u_53 = tid_c * 32 + 20 + 1;
                        int pg_u_53 = page_table[b_u_1 * max_pages + blk_u_53];
                        int blk_st_53 = blk_u_53;
                        if (pg_u_53 < 0) {
                            pg_u_53 = 0;
                            blk_st_53 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_53;
                        meta_pg[pos_t_1] = pg_u_53 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[5] >> 16 & 1) != 0) {
                        int blk_u_54 = tid_c * 32 + 20 + 2;
                        int pg_u_54 = page_table[b_u_1 * max_pages + blk_u_54];
                        int blk_st_54 = blk_u_54;
                        if (pg_u_54 < 0) {
                            pg_u_54 = 0;
                            blk_st_54 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_54;
                        meta_pg[pos_t_1] = pg_u_54 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[5] >> 24 & 1) != 0) {
                        int blk_u_55 = tid_c * 32 + 20 + 3;
                        int pg_u_55 = page_table[b_u_1 * max_pages + blk_u_55];
                        int blk_st_55 = blk_u_55;
                        if (pg_u_55 < 0) {
                            pg_u_55 = 0;
                            blk_st_55 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_55;
                        meta_pg[pos_t_1] = pg_u_55 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[6] & 1) != 0) {
                        int blk_u_56 = tid_c * 32 + 24;
                        int pg_u_56 = page_table[b_u_1 * max_pages + blk_u_56];
                        int blk_st_56 = blk_u_56;
                        if (pg_u_56 < 0) {
                            pg_u_56 = 0;
                            blk_st_56 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_56;
                        meta_pg[pos_t_1] = pg_u_56 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[6] >> 8 & 1) != 0) {
                        int blk_u_57 = tid_c * 32 + 24 + 1;
                        int pg_u_57 = page_table[b_u_1 * max_pages + blk_u_57];
                        int blk_st_57 = blk_u_57;
                        if (pg_u_57 < 0) {
                            pg_u_57 = 0;
                            blk_st_57 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_57;
                        meta_pg[pos_t_1] = pg_u_57 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[6] >> 16 & 1) != 0) {
                        int blk_u_58 = tid_c * 32 + 24 + 2;
                        int pg_u_58 = page_table[b_u_1 * max_pages + blk_u_58];
                        int blk_st_58 = blk_u_58;
                        if (pg_u_58 < 0) {
                            pg_u_58 = 0;
                            blk_st_58 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_58;
                        meta_pg[pos_t_1] = pg_u_58 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[6] >> 24 & 1) != 0) {
                        int blk_u_59 = tid_c * 32 + 24 + 3;
                        int pg_u_59 = page_table[b_u_1 * max_pages + blk_u_59];
                        int blk_st_59 = blk_u_59;
                        if (pg_u_59 < 0) {
                            pg_u_59 = 0;
                            blk_st_59 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_59;
                        meta_pg[pos_t_1] = pg_u_59 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[7] & 1) != 0) {
                        int blk_u_60 = tid_c * 32 + 28;
                        int pg_u_60 = page_table[b_u_1 * max_pages + blk_u_60];
                        int blk_st_60 = blk_u_60;
                        if (pg_u_60 < 0) {
                            pg_u_60 = 0;
                            blk_st_60 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_60;
                        meta_pg[pos_t_1] = pg_u_60 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[7] >> 8 & 1) != 0) {
                        int blk_u_61 = tid_c * 32 + 28 + 1;
                        int pg_u_61 = page_table[b_u_1 * max_pages + blk_u_61];
                        int blk_st_61 = blk_u_61;
                        if (pg_u_61 < 0) {
                            pg_u_61 = 0;
                            blk_st_61 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_61;
                        meta_pg[pos_t_1] = pg_u_61 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[7] >> 16 & 1) != 0) {
                        int blk_u_62 = tid_c * 32 + 28 + 2;
                        int pg_u_62 = page_table[b_u_1 * max_pages + blk_u_62];
                        int blk_st_62 = blk_u_62;
                        if (pg_u_62 < 0) {
                            pg_u_62 = 0;
                            blk_st_62 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_62;
                        meta_pg[pos_t_1] = pg_u_62 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                    if ((mw_1[7] >> 24 & 1) != 0) {
                        int blk_u_63 = tid_c * 32 + 28 + 3;
                        int pg_u_63 = page_table[b_u_1 * max_pages + blk_u_63];
                        int blk_st_63 = blk_u_63;
                        if (pg_u_63 < 0) {
                            pg_u_63 = 0;
                            blk_st_63 = 4194304;
                        }
                        meta_blk[pos_t_1] = blk_st_63;
                        meta_pg[pos_t_1] = pg_u_63 * num_kv_heads + kv_head_c;
                        meta_mask[2 * pos_t_1] = 0;
                        meta_mask[2 * pos_t_1 + 1] = 0;
                        pos_t_1 = pos_t_1 + 1;
                    }
                }
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            if (blk_e_1[0] >= 0) {
                int grp_e_1 = blk_e_1[0] >> 5;
                int o_e_1 = blk_e_1[0] & 31;
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(0) + 3]))
                    : "r"(mark_addr + (unsigned int)(grp_e_1 * 32)));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&mw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mw_1[(4) + 3]))
                    : "r"(mark_addr + (unsigned int)(grp_e_1 * 32) + 16));
                int rank_1 = pref[grp_e_1];
                int nb__7 = o_e_1;
                if (nb__7 > 0) {
                    if (nb__7 >= 4) {
                        int _popc_32 = __popc(mw_1[0]);
                        rank_1 = rank_1 + _popc_32;
                    }
                    if (nb__7 < 4) {
                        unsigned int lowmask_8 = (1 << 8 * nb__7) - 1;
                        int _popc_33 = __popc(mw_1[0] & lowmask_8);
                        rank_1 = rank_1 + _popc_33;
                    }
                }
                int nb__0_1 = o_e_1 - 4;
                if (nb__0_1 > 0) {
                    if (nb__0_1 >= 4) {
                        int _popc_34 = __popc(mw_1[1]);
                        rank_1 = rank_1 + _popc_34;
                    }
                    if (nb__0_1 < 4) {
                        unsigned int lowmask_9 = (1 << 8 * nb__0_1) - 1;
                        int _popc_35 = __popc(mw_1[1] & lowmask_9);
                        rank_1 = rank_1 + _popc_35;
                    }
                }
                int nb__1_1 = o_e_1 - 8;
                if (nb__1_1 > 0) {
                    if (nb__1_1 >= 4) {
                        int _popc_36 = __popc(mw_1[2]);
                        rank_1 = rank_1 + _popc_36;
                    }
                    if (nb__1_1 < 4) {
                        unsigned int lowmask_10 = (1 << 8 * nb__1_1) - 1;
                        int _popc_37 = __popc(mw_1[2] & lowmask_10);
                        rank_1 = rank_1 + _popc_37;
                    }
                }
                int nb__2_1 = o_e_1 - 12;
                if (nb__2_1 > 0) {
                    if (nb__2_1 >= 4) {
                        int _popc_38 = __popc(mw_1[3]);
                        rank_1 = rank_1 + _popc_38;
                    }
                    if (nb__2_1 < 4) {
                        unsigned int lowmask_11 = (1 << 8 * nb__2_1) - 1;
                        int _popc_39 = __popc(mw_1[3] & lowmask_11);
                        rank_1 = rank_1 + _popc_39;
                    }
                }
                int nb__3_1 = o_e_1 - 16;
                if (nb__3_1 > 0) {
                    if (nb__3_1 >= 4) {
                        int _popc_40 = __popc(mw_1[4]);
                        rank_1 = rank_1 + _popc_40;
                    }
                    if (nb__3_1 < 4) {
                        unsigned int lowmask_12 = (1 << 8 * nb__3_1) - 1;
                        int _popc_41 = __popc(mw_1[4] & lowmask_12);
                        rank_1 = rank_1 + _popc_41;
                    }
                }
                int nb__4_1 = o_e_1 - 20;
                if (nb__4_1 > 0) {
                    if (nb__4_1 >= 4) {
                        int _popc_42 = __popc(mw_1[5]);
                        rank_1 = rank_1 + _popc_42;
                    }
                    if (nb__4_1 < 4) {
                        unsigned int lowmask_13 = (1 << 8 * nb__4_1) - 1;
                        int _popc_43 = __popc(mw_1[5] & lowmask_13);
                        rank_1 = rank_1 + _popc_43;
                    }
                }
                int nb__5_1 = o_e_1 - 24;
                if (nb__5_1 > 0) {
                    if (nb__5_1 >= 4) {
                        int _popc_44 = __popc(mw_1[6]);
                        rank_1 = rank_1 + _popc_44;
                    }
                    if (nb__5_1 < 4) {
                        unsigned int lowmask_14 = (1 << 8 * nb__5_1) - 1;
                        int _popc_45 = __popc(mw_1[6] & lowmask_14);
                        rank_1 = rank_1 + _popc_45;
                    }
                }
                int nb__6_1 = o_e_1 - 28;
                if (nb__6_1 > 0) {
                    if (nb__6_1 >= 4) {
                        int _popc_46 = __popc(mw_1[7]);
                        rank_1 = rank_1 + _popc_46;
                    }
                    if (nb__6_1 < 4) {
                        unsigned int lowmask_15 = (1 << 8 * nb__6_1) - 1;
                        int _popc_47 = __popc(mw_1[7] & lowmask_15);
                        rank_1 = rank_1 + _popc_47;
                    }
                }
                int tl_i_1 = tid_c / 16;
                unsigned int bit_1 = 1 << (tl_i_1 & 31);
                atomicAdd_block(&meta_mask[2 * rank_1 + (tl_i_1 >> 5)], bit_1);
            }
            asm volatile("barrier.sync 8, 384;" ::: "memory");
            int nblk_c = cnt[0] + cnt[1] + (cnt[2] + cnt[3]);
            int nhalf_c = 2 * nblk_c;
            int tok0 = hdr[1];
            int n_tok = hdr[2];
            int cu_b = hdr[3];
            int qpos1 = hdr[4] + 1;
            int cw = warp_c - 4;
            int wgi = cw / 4;
            int wi = cw - wgi * 4;
            int g = lane_c / 4;
            int cq = lane_c - g * 4;
            int r0 = wgi * 64 + wi * 16 + g;
            int r1 = r0 + 8;
            int tl0 = r0 >> 4;
            int tl1 = r1 >> 4;
            int hg0 = r0 - (tl0 << 4);
            int hg1 = r1 - (tl1 << 4);
            int mw0 = tl0 >> 5;
            int ms0 = tl0 & 31;
            int mw1 = tl1 >> 5;
            int ms1 = tl1 & 31;
            int tokbase = wgi * 4;
            int wword = tokbase >> 5;
            int wshift = tokbase & 31;
            int tokhi = tokbase + 3;
            int kcol = 2 * cq;
            unsigned int qf[32];
            qf[0] = 0;
            qf[1] = 0;
            qf[2] = 0;
            qf[3] = 0;
            qf[4] = 0;
            qf[5] = 0;
            qf[6] = 0;
            qf[7] = 0;
            qf[8] = 0;
            qf[9] = 0;
            qf[10] = 0;
            qf[11] = 0;
            qf[12] = 0;
            qf[13] = 0;
            qf[14] = 0;
            qf[15] = 0;
            qf[16] = 0;
            qf[17] = 0;
            qf[18] = 0;
            qf[19] = 0;
            qf[20] = 0;
            qf[21] = 0;
            qf[22] = 0;
            qf[23] = 0;
            qf[24] = 0;
            qf[25] = 0;
            qf[26] = 0;
            qf[27] = 0;
            qf[28] = 0;
            qf[29] = 0;
            qf[30] = 0;
            qf[31] = 0;
            if (tl0 < n_tok) {
                int qrow0 = (cu_b + tok0 + tl0) * num_q_heads + kv_head_c * 16 + hg0;
                int qb0 = qrow0 * 64 + cq;
                qf[0] = Q32[qb0];
                qf[2] = Q32[qb0 + 4];
                qf[4] = Q32[qb0 + 8];
                qf[6] = Q32[qb0 + 8 + 4];
                qf[8] = Q32[qb0 + 16];
                qf[10] = Q32[qb0 + 16 + 4];
                qf[12] = Q32[qb0 + 24];
                qf[14] = Q32[qb0 + 24 + 4];
                qf[16] = Q32[qb0 + 32];
                qf[18] = Q32[qb0 + 32 + 4];
                qf[20] = Q32[qb0 + 40];
                qf[22] = Q32[qb0 + 40 + 4];
                qf[24] = Q32[qb0 + 48];
                qf[26] = Q32[qb0 + 48 + 4];
                qf[28] = Q32[qb0 + 56];
                qf[30] = Q32[qb0 + 56 + 4];
            }
            if (tl1 < n_tok) {
                int qrow1 = (cu_b + tok0 + tl1) * num_q_heads + kv_head_c * 16 + hg1;
                int qb1 = qrow1 * 64 + cq;
                qf[1] = Q32[qb1];
                qf[3] = Q32[qb1 + 4];
                qf[5] = Q32[qb1 + 8];
                qf[7] = Q32[qb1 + 8 + 4];
                qf[9] = Q32[qb1 + 16];
                qf[11] = Q32[qb1 + 16 + 4];
                qf[13] = Q32[qb1 + 24];
                qf[15] = Q32[qb1 + 24 + 4];
                qf[17] = Q32[qb1 + 32];
                qf[19] = Q32[qb1 + 32 + 4];
                qf[21] = Q32[qb1 + 40];
                qf[23] = Q32[qb1 + 40 + 4];
                qf[25] = Q32[qb1 + 48];
                qf[27] = Q32[qb1 + 48 + 4];
                qf[29] = Q32[qb1 + 56];
                qf[31] = Q32[qb1 + 56 + 4];
            }
            uint32_t _bf16x2_to_f16x2_0;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_0) : "r"(qf[0]));
            qf[0] = _bf16x2_to_f16x2_0;
            uint32_t _bf16x2_to_f16x2_1;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_1) : "r"(qf[1]));
            qf[1] = _bf16x2_to_f16x2_1;
            uint32_t _bf16x2_to_f16x2_2;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_2) : "r"(qf[2]));
            qf[2] = _bf16x2_to_f16x2_2;
            uint32_t _bf16x2_to_f16x2_3;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_3) : "r"(qf[3]));
            qf[3] = _bf16x2_to_f16x2_3;
            uint32_t _bf16x2_to_f16x2_4;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_4) : "r"(qf[4]));
            qf[4] = _bf16x2_to_f16x2_4;
            uint32_t _bf16x2_to_f16x2_5;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_5) : "r"(qf[5]));
            qf[5] = _bf16x2_to_f16x2_5;
            uint32_t _bf16x2_to_f16x2_6;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_6) : "r"(qf[6]));
            qf[6] = _bf16x2_to_f16x2_6;
            uint32_t _bf16x2_to_f16x2_7;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_7) : "r"(qf[7]));
            qf[7] = _bf16x2_to_f16x2_7;
            uint32_t _bf16x2_to_f16x2_8;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_8) : "r"(qf[8]));
            qf[8] = _bf16x2_to_f16x2_8;
            uint32_t _bf16x2_to_f16x2_9;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_9) : "r"(qf[9]));
            qf[9] = _bf16x2_to_f16x2_9;
            uint32_t _bf16x2_to_f16x2_10;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_10) : "r"(qf[10]));
            qf[10] = _bf16x2_to_f16x2_10;
            uint32_t _bf16x2_to_f16x2_11;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_11) : "r"(qf[11]));
            qf[11] = _bf16x2_to_f16x2_11;
            uint32_t _bf16x2_to_f16x2_12;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_12) : "r"(qf[12]));
            qf[12] = _bf16x2_to_f16x2_12;
            uint32_t _bf16x2_to_f16x2_13;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_13) : "r"(qf[13]));
            qf[13] = _bf16x2_to_f16x2_13;
            uint32_t _bf16x2_to_f16x2_14;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_14) : "r"(qf[14]));
            qf[14] = _bf16x2_to_f16x2_14;
            uint32_t _bf16x2_to_f16x2_15;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_15) : "r"(qf[15]));
            qf[15] = _bf16x2_to_f16x2_15;
            uint32_t _bf16x2_to_f16x2_16;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_16) : "r"(qf[16]));
            qf[16] = _bf16x2_to_f16x2_16;
            uint32_t _bf16x2_to_f16x2_17;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_17) : "r"(qf[17]));
            qf[17] = _bf16x2_to_f16x2_17;
            uint32_t _bf16x2_to_f16x2_18;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_18) : "r"(qf[18]));
            qf[18] = _bf16x2_to_f16x2_18;
            uint32_t _bf16x2_to_f16x2_19;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_19) : "r"(qf[19]));
            qf[19] = _bf16x2_to_f16x2_19;
            uint32_t _bf16x2_to_f16x2_20;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_20) : "r"(qf[20]));
            qf[20] = _bf16x2_to_f16x2_20;
            uint32_t _bf16x2_to_f16x2_21;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_21) : "r"(qf[21]));
            qf[21] = _bf16x2_to_f16x2_21;
            uint32_t _bf16x2_to_f16x2_22;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_22) : "r"(qf[22]));
            qf[22] = _bf16x2_to_f16x2_22;
            uint32_t _bf16x2_to_f16x2_23;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_23) : "r"(qf[23]));
            qf[23] = _bf16x2_to_f16x2_23;
            uint32_t _bf16x2_to_f16x2_24;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_24) : "r"(qf[24]));
            qf[24] = _bf16x2_to_f16x2_24;
            uint32_t _bf16x2_to_f16x2_25;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_25) : "r"(qf[25]));
            qf[25] = _bf16x2_to_f16x2_25;
            uint32_t _bf16x2_to_f16x2_26;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_26) : "r"(qf[26]));
            qf[26] = _bf16x2_to_f16x2_26;
            uint32_t _bf16x2_to_f16x2_27;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_27) : "r"(qf[27]));
            qf[27] = _bf16x2_to_f16x2_27;
            uint32_t _bf16x2_to_f16x2_28;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_28) : "r"(qf[28]));
            qf[28] = _bf16x2_to_f16x2_28;
            uint32_t _bf16x2_to_f16x2_29;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_29) : "r"(qf[29]));
            qf[29] = _bf16x2_to_f16x2_29;
            uint32_t _bf16x2_to_f16x2_30;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_30) : "r"(qf[30]));
            qf[30] = _bf16x2_to_f16x2_30;
            uint32_t _bf16x2_to_f16x2_31;
            asm(
                "{ .reg .b16 _bf_lo, _bf_hi;             \n\t"
                "  .reg .f32 _fp_lo, _fp_hi;             \n\t"
                "  mov.b32 {_bf_lo, _bf_hi}, %1;         \n\t"
                "  cvt.f32.bf16 _fp_lo, _bf_lo;          \n\t"
                "  cvt.f32.bf16 _fp_hi, _bf_hi;          \n\t"
                "  cvt.rn.f16x2.f32 %0, _fp_hi, _fp_lo; }"
                : "=r"(_bf16x2_to_f16x2_31) : "r"(qf[31]));
            qf[31] = _bf16x2_to_f16x2_31;
            if (wgi == 1) {
                asm volatile("barrier.arrive 9, 256;" ::: "memory");
            }
            float d_s[32];
            float d_o[64];
            unsigned int pk[16];
            d_o[0] = 0.0f;
            d_o[1] = 0.0f;
            d_o[2] = 0.0f;
            d_o[3] = 0.0f;
            d_o[4] = 0.0f;
            d_o[5] = 0.0f;
            d_o[6] = 0.0f;
            d_o[7] = 0.0f;
            d_o[8] = 0.0f;
            d_o[9] = 0.0f;
            d_o[10] = 0.0f;
            d_o[11] = 0.0f;
            d_o[12] = 0.0f;
            d_o[13] = 0.0f;
            d_o[14] = 0.0f;
            d_o[15] = 0.0f;
            d_o[16] = 0.0f;
            d_o[17] = 0.0f;
            d_o[18] = 0.0f;
            d_o[19] = 0.0f;
            d_o[20] = 0.0f;
            d_o[21] = 0.0f;
            d_o[22] = 0.0f;
            d_o[23] = 0.0f;
            d_o[24] = 0.0f;
            d_o[25] = 0.0f;
            d_o[26] = 0.0f;
            d_o[27] = 0.0f;
            d_o[28] = 0.0f;
            d_o[29] = 0.0f;
            d_o[30] = 0.0f;
            d_o[31] = 0.0f;
            d_o[32] = 0.0f;
            d_o[33] = 0.0f;
            d_o[34] = 0.0f;
            d_o[35] = 0.0f;
            d_o[36] = 0.0f;
            d_o[37] = 0.0f;
            d_o[38] = 0.0f;
            d_o[39] = 0.0f;
            d_o[40] = 0.0f;
            d_o[41] = 0.0f;
            d_o[42] = 0.0f;
            d_o[43] = 0.0f;
            d_o[44] = 0.0f;
            d_o[45] = 0.0f;
            d_o[46] = 0.0f;
            d_o[47] = 0.0f;
            d_o[48] = 0.0f;
            d_o[49] = 0.0f;
            d_o[50] = 0.0f;
            d_o[51] = 0.0f;
            d_o[52] = 0.0f;
            d_o[53] = 0.0f;
            d_o[54] = 0.0f;
            d_o[55] = 0.0f;
            d_o[56] = 0.0f;
            d_o[57] = 0.0f;
            d_o[58] = 0.0f;
            d_o[59] = 0.0f;
            d_o[60] = 0.0f;
            d_o[61] = 0.0f;
            d_o[62] = 0.0f;
            d_o[63] = 0.0f;
            pk[0] = 0;
            pk[1] = 0;
            pk[2] = 0;
            pk[3] = 0;
            pk[4] = 0;
            pk[5] = 0;
            pk[6] = 0;
            pk[7] = 0;
            pk[8] = 0;
            pk[9] = 0;
            pk[10] = 0;
            pk[11] = 0;
            pk[12] = 0;
            pk[13] = 0;
            pk[14] = 0;
            pk[15] = 0;
            float mrun0 = -CAKE_INF;
            float mrun1 = -CAKE_INF;
            float l0 = 0.0f;
            float l1 = 0.0f;
            int first_c = 1;
            #pragma unroll 1
            for (int i_1 = 0; i_1 < nhalf_c; i_1++) {
                int u = i_1 >> 1;
                int fb_1 = i_1 % 2;
                int blk_j = meta_blk[u];
                int pbase = blk_j * 128 + (i_1 & 1) * 64;
                unsigned int mwd = meta_mask[2 * u + wword];
                unsigned int mb0 = meta_mask[2 * u + mw0];
                unsigned int mb1 = meta_mask[2 * u + mw1];
                int live = 0;
                if ((mwd >> (unsigned int)wshift & 15) != 0) {
                    if (qpos1 + tokhi - pbase > 0) {
                        live = 1;
                    }
                }
                int _shfl_10 = __shfl_sync(0xFFFFFFFF, live, 0);
                live = _shfl_10;
                mbarrier_wait(f16_full_addr + (fb_1) * 8, i_1 / 2 % 2);
                if (wgi == 0) {
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                }
                if (wgi == 1) {
                    asm volatile("barrier.sync 10, 256;" ::: "memory");
                }
                int qk_go = live;
                qk_go = 1;
                if (qk_go != 0) {
                    asm volatile("" : "+f"(d_s[0]) :: "memory");
                    asm volatile("" : "+f"(d_s[1]) :: "memory");
                    asm volatile("" : "+f"(d_s[2]) :: "memory");
                    asm volatile("" : "+f"(d_s[3]) :: "memory");
                    asm volatile("" : "+f"(d_s[4]) :: "memory");
                    asm volatile("" : "+f"(d_s[5]) :: "memory");
                    asm volatile("" : "+f"(d_s[6]) :: "memory");
                    asm volatile("" : "+f"(d_s[7]) :: "memory");
                    asm volatile("" : "+f"(d_s[8]) :: "memory");
                    asm volatile("" : "+f"(d_s[9]) :: "memory");
                    asm volatile("" : "+f"(d_s[10]) :: "memory");
                    asm volatile("" : "+f"(d_s[11]) :: "memory");
                    asm volatile("" : "+f"(d_s[12]) :: "memory");
                    asm volatile("" : "+f"(d_s[13]) :: "memory");
                    asm volatile("" : "+f"(d_s[14]) :: "memory");
                    asm volatile("" : "+f"(d_s[15]) :: "memory");
                    asm volatile("" : "+f"(d_s[16]) :: "memory");
                    asm volatile("" : "+f"(d_s[17]) :: "memory");
                    asm volatile("" : "+f"(d_s[18]) :: "memory");
                    asm volatile("" : "+f"(d_s[19]) :: "memory");
                    asm volatile("" : "+f"(d_s[20]) :: "memory");
                    asm volatile("" : "+f"(d_s[21]) :: "memory");
                    asm volatile("" : "+f"(d_s[22]) :: "memory");
                    asm volatile("" : "+f"(d_s[23]) :: "memory");
                    asm volatile("" : "+f"(d_s[24]) :: "memory");
                    asm volatile("" : "+f"(d_s[25]) :: "memory");
                    asm volatile("" : "+f"(d_s[26]) :: "memory");
                    asm volatile("" : "+f"(d_s[27]) :: "memory");
                    asm volatile("" : "+f"(d_s[28]) :: "memory");
                    asm volatile("" : "+f"(d_s[29]) :: "memory");
                    asm volatile("" : "+f"(d_s[30]) :: "memory");
                    asm volatile("" : "+f"(d_s[31]) :: "memory");
                    asm volatile("" : "+r"(qf[0]) :: "memory");
                    asm volatile("" : "+r"(qf[1]) :: "memory");
                    asm volatile("" : "+r"(qf[2]) :: "memory");
                    asm volatile("" : "+r"(qf[3]) :: "memory");
                    asm volatile("" : "+r"(qf[4]) :: "memory");
                    asm volatile("" : "+r"(qf[5]) :: "memory");
                    asm volatile("" : "+r"(qf[6]) :: "memory");
                    asm volatile("" : "+r"(qf[7]) :: "memory");
                    asm volatile("" : "+r"(qf[8]) :: "memory");
                    asm volatile("" : "+r"(qf[9]) :: "memory");
                    asm volatile("" : "+r"(qf[10]) :: "memory");
                    asm volatile("" : "+r"(qf[11]) :: "memory");
                    asm volatile("" : "+r"(qf[12]) :: "memory");
                    asm volatile("" : "+r"(qf[13]) :: "memory");
                    asm volatile("" : "+r"(qf[14]) :: "memory");
                    asm volatile("" : "+r"(qf[15]) :: "memory");
                    asm volatile("" : "+r"(qf[16]) :: "memory");
                    asm volatile("" : "+r"(qf[17]) :: "memory");
                    asm volatile("" : "+r"(qf[18]) :: "memory");
                    asm volatile("" : "+r"(qf[19]) :: "memory");
                    asm volatile("" : "+r"(qf[20]) :: "memory");
                    asm volatile("" : "+r"(qf[21]) :: "memory");
                    asm volatile("" : "+r"(qf[22]) :: "memory");
                    asm volatile("" : "+r"(qf[23]) :: "memory");
                    asm volatile("" : "+r"(qf[24]) :: "memory");
                    asm volatile("" : "+r"(qf[25]) :: "memory");
                    asm volatile("" : "+r"(qf[26]) :: "memory");
                    asm volatile("" : "+r"(qf[27]) :: "memory");
                    asm volatile("" : "+r"(qf[28]) :: "memory");
                    asm volatile("" : "+r"(qf[29]) :: "memory");
                    asm volatile("" : "+r"(qf[30]) :: "memory");
                    asm volatile("" : "+r"(qf[31]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_1 = (((uint64_t)(((k16_addr + (unsigned int)(fb_1 * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
                    uint64_t _wgmma_desc_2 = (((uint64_t)(((k16_addr + (unsigned int)(fb_1 * 16384) + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 0, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[0]), "r"(qf[1]), "r"(qf[2]), "r"(qf[3]), "l"(_wgmma_b_0_3)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[4]), "r"(qf[(4) + 1]), "r"(qf[(4) + 2]), "r"(qf[(4) + 3]), "l"(_wgmma_b_0_3 + 2)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[8]), "r"(qf[(8) + 1]), "r"(qf[(8) + 2]), "r"(qf[(8) + 3]), "l"(_wgmma_b_0_3 + 4)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[12]), "r"(qf[(12) + 1]), "r"(qf[(12) + 2]), "r"(qf[(12) + 3]), "l"(_wgmma_b_0_3 + 6)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[16]), "r"(qf[(16) + 1]), "r"(qf[(16) + 2]), "r"(qf[(16) + 3]), "l"(_wgmma_b_0_4)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[20]), "r"(qf[(20) + 1]), "r"(qf[(20) + 2]), "r"(qf[(20) + 3]), "l"(_wgmma_b_0_4 + 2)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[24]), "r"(qf[(24) + 1]), "r"(qf[(24) + 2]), "r"(qf[(24) + 3]), "l"(_wgmma_b_0_4 + 4)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_s[0]), "+f"(d_s[1]), "+f"(d_s[2]), "+f"(d_s[3]), "+f"(d_s[4]), "+f"(d_s[5]), "+f"(d_s[6]), "+f"(d_s[7]), "+f"(d_s[8]), "+f"(d_s[9]), "+f"(d_s[10]), "+f"(d_s[11]), "+f"(d_s[12]), "+f"(d_s[13]), "+f"(d_s[14]), "+f"(d_s[15]), "+f"(d_s[16]), "+f"(d_s[17]), "+f"(d_s[18]), "+f"(d_s[19]), "+f"(d_s[20]), "+f"(d_s[21]), "+f"(d_s[22]), "+f"(d_s[23]), "+f"(d_s[24]), "+f"(d_s[25]), "+f"(d_s[26]), "+f"(d_s[27]), "+f"(d_s[28]), "+f"(d_s[29]), "+f"(d_s[30]), "+f"(d_s[31])
                        : "r"(qf[28]), "r"(qf[(28) + 1]), "r"(qf[(28) + 2]), "r"(qf[(28) + 3]), "l"(_wgmma_b_0_4 + 6)
                        );
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("" : "+f"(d_s[0]) :: "memory");
                    asm volatile("" : "+f"(d_s[1]) :: "memory");
                    asm volatile("" : "+f"(d_s[2]) :: "memory");
                    asm volatile("" : "+f"(d_s[3]) :: "memory");
                    asm volatile("" : "+f"(d_s[4]) :: "memory");
                    asm volatile("" : "+f"(d_s[5]) :: "memory");
                    asm volatile("" : "+f"(d_s[6]) :: "memory");
                    asm volatile("" : "+f"(d_s[7]) :: "memory");
                    asm volatile("" : "+f"(d_s[8]) :: "memory");
                    asm volatile("" : "+f"(d_s[9]) :: "memory");
                    asm volatile("" : "+f"(d_s[10]) :: "memory");
                    asm volatile("" : "+f"(d_s[11]) :: "memory");
                    asm volatile("" : "+f"(d_s[12]) :: "memory");
                    asm volatile("" : "+f"(d_s[13]) :: "memory");
                    asm volatile("" : "+f"(d_s[14]) :: "memory");
                    asm volatile("" : "+f"(d_s[15]) :: "memory");
                    asm volatile("" : "+f"(d_s[16]) :: "memory");
                    asm volatile("" : "+f"(d_s[17]) :: "memory");
                    asm volatile("" : "+f"(d_s[18]) :: "memory");
                    asm volatile("" : "+f"(d_s[19]) :: "memory");
                    asm volatile("" : "+f"(d_s[20]) :: "memory");
                    asm volatile("" : "+f"(d_s[21]) :: "memory");
                    asm volatile("" : "+f"(d_s[22]) :: "memory");
                    asm volatile("" : "+f"(d_s[23]) :: "memory");
                    asm volatile("" : "+f"(d_s[24]) :: "memory");
                    asm volatile("" : "+f"(d_s[25]) :: "memory");
                    asm volatile("" : "+f"(d_s[26]) :: "memory");
                    asm volatile("" : "+f"(d_s[27]) :: "memory");
                    asm volatile("" : "+f"(d_s[28]) :: "memory");
                    asm volatile("" : "+f"(d_s[29]) :: "memory");
                    asm volatile("" : "+f"(d_s[30]) :: "memory");
                    asm volatile("" : "+f"(d_s[31]) :: "memory");
                }
                if (wgi == 0) {
                    asm volatile("barrier.arrive 10, 256;" ::: "memory");
                }
                if (wgi == 1) {
                    asm volatile("barrier.arrive 9, 256;" ::: "memory");
                }
                asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                asm volatile("" : "+f"(d_s[0]) :: "memory");
                asm volatile("" : "+f"(d_s[1]) :: "memory");
                asm volatile("" : "+f"(d_s[2]) :: "memory");
                asm volatile("" : "+f"(d_s[3]) :: "memory");
                asm volatile("" : "+f"(d_s[4]) :: "memory");
                asm volatile("" : "+f"(d_s[5]) :: "memory");
                asm volatile("" : "+f"(d_s[6]) :: "memory");
                asm volatile("" : "+f"(d_s[7]) :: "memory");
                asm volatile("" : "+f"(d_s[8]) :: "memory");
                asm volatile("" : "+f"(d_s[9]) :: "memory");
                asm volatile("" : "+f"(d_s[10]) :: "memory");
                asm volatile("" : "+f"(d_s[11]) :: "memory");
                asm volatile("" : "+f"(d_s[12]) :: "memory");
                asm volatile("" : "+f"(d_s[13]) :: "memory");
                asm volatile("" : "+f"(d_s[14]) :: "memory");
                asm volatile("" : "+f"(d_s[15]) :: "memory");
                asm volatile("" : "+f"(d_s[16]) :: "memory");
                asm volatile("" : "+f"(d_s[17]) :: "memory");
                asm volatile("" : "+f"(d_s[18]) :: "memory");
                asm volatile("" : "+f"(d_s[19]) :: "memory");
                asm volatile("" : "+f"(d_s[20]) :: "memory");
                asm volatile("" : "+f"(d_s[21]) :: "memory");
                asm volatile("" : "+f"(d_s[22]) :: "memory");
                asm volatile("" : "+f"(d_s[23]) :: "memory");
                asm volatile("" : "+f"(d_s[24]) :: "memory");
                asm volatile("" : "+f"(d_s[25]) :: "memory");
                asm volatile("" : "+f"(d_s[26]) :: "memory");
                asm volatile("" : "+f"(d_s[27]) :: "memory");
                asm volatile("" : "+f"(d_s[28]) :: "memory");
                asm volatile("" : "+f"(d_s[29]) :: "memory");
                asm volatile("" : "+f"(d_s[30]) :: "memory");
                asm volatile("" : "+f"(d_s[31]) :: "memory");
                asm volatile("" : "+f"(d_o[0]) :: "memory");
                asm volatile("" : "+f"(d_o[1]) :: "memory");
                asm volatile("" : "+f"(d_o[2]) :: "memory");
                asm volatile("" : "+f"(d_o[3]) :: "memory");
                asm volatile("" : "+f"(d_o[4]) :: "memory");
                asm volatile("" : "+f"(d_o[5]) :: "memory");
                asm volatile("" : "+f"(d_o[6]) :: "memory");
                asm volatile("" : "+f"(d_o[7]) :: "memory");
                asm volatile("" : "+f"(d_o[8]) :: "memory");
                asm volatile("" : "+f"(d_o[9]) :: "memory");
                asm volatile("" : "+f"(d_o[10]) :: "memory");
                asm volatile("" : "+f"(d_o[11]) :: "memory");
                asm volatile("" : "+f"(d_o[12]) :: "memory");
                asm volatile("" : "+f"(d_o[13]) :: "memory");
                asm volatile("" : "+f"(d_o[14]) :: "memory");
                asm volatile("" : "+f"(d_o[15]) :: "memory");
                asm volatile("" : "+f"(d_o[16]) :: "memory");
                asm volatile("" : "+f"(d_o[17]) :: "memory");
                asm volatile("" : "+f"(d_o[18]) :: "memory");
                asm volatile("" : "+f"(d_o[19]) :: "memory");
                asm volatile("" : "+f"(d_o[20]) :: "memory");
                asm volatile("" : "+f"(d_o[21]) :: "memory");
                asm volatile("" : "+f"(d_o[22]) :: "memory");
                asm volatile("" : "+f"(d_o[23]) :: "memory");
                asm volatile("" : "+f"(d_o[24]) :: "memory");
                asm volatile("" : "+f"(d_o[25]) :: "memory");
                asm volatile("" : "+f"(d_o[26]) :: "memory");
                asm volatile("" : "+f"(d_o[27]) :: "memory");
                asm volatile("" : "+f"(d_o[28]) :: "memory");
                asm volatile("" : "+f"(d_o[29]) :: "memory");
                asm volatile("" : "+f"(d_o[30]) :: "memory");
                asm volatile("" : "+f"(d_o[31]) :: "memory");
                asm volatile("" : "+f"(d_o[32]) :: "memory");
                asm volatile("" : "+f"(d_o[33]) :: "memory");
                asm volatile("" : "+f"(d_o[34]) :: "memory");
                asm volatile("" : "+f"(d_o[35]) :: "memory");
                asm volatile("" : "+f"(d_o[36]) :: "memory");
                asm volatile("" : "+f"(d_o[37]) :: "memory");
                asm volatile("" : "+f"(d_o[38]) :: "memory");
                asm volatile("" : "+f"(d_o[39]) :: "memory");
                asm volatile("" : "+f"(d_o[40]) :: "memory");
                asm volatile("" : "+f"(d_o[41]) :: "memory");
                asm volatile("" : "+f"(d_o[42]) :: "memory");
                asm volatile("" : "+f"(d_o[43]) :: "memory");
                asm volatile("" : "+f"(d_o[44]) :: "memory");
                asm volatile("" : "+f"(d_o[45]) :: "memory");
                asm volatile("" : "+f"(d_o[46]) :: "memory");
                asm volatile("" : "+f"(d_o[47]) :: "memory");
                asm volatile("" : "+f"(d_o[48]) :: "memory");
                asm volatile("" : "+f"(d_o[49]) :: "memory");
                asm volatile("" : "+f"(d_o[50]) :: "memory");
                asm volatile("" : "+f"(d_o[51]) :: "memory");
                asm volatile("" : "+f"(d_o[52]) :: "memory");
                asm volatile("" : "+f"(d_o[53]) :: "memory");
                asm volatile("" : "+f"(d_o[54]) :: "memory");
                asm volatile("" : "+f"(d_o[55]) :: "memory");
                asm volatile("" : "+f"(d_o[56]) :: "memory");
                asm volatile("" : "+f"(d_o[57]) :: "memory");
                asm volatile("" : "+f"(d_o[58]) :: "memory");
                asm volatile("" : "+f"(d_o[59]) :: "memory");
                asm volatile("" : "+f"(d_o[60]) :: "memory");
                asm volatile("" : "+f"(d_o[61]) :: "memory");
                asm volatile("" : "+f"(d_o[62]) :: "memory");
                asm volatile("" : "+f"(d_o[63]) :: "memory");
                if (live == 0) {
                    pk[0] = 0;
                    pk[1] = 0;
                    pk[2] = 0;
                    pk[3] = 0;
                    pk[4] = 0;
                    pk[5] = 0;
                    pk[6] = 0;
                    pk[7] = 0;
                    pk[8] = 0;
                    pk[9] = 0;
                    pk[10] = 0;
                    pk[11] = 0;
                    pk[12] = 0;
                    pk[13] = 0;
                    pk[14] = 0;
                    pk[15] = 0;
                }
                if (live != 0) {
                    int lim0 = qpos1 + tl0 - pbase;
                    if ((mb0 >> (unsigned int)ms0 & 1) == 0) {
                        lim0 = 0;
                    }
                    int lim1 = qpos1 + tl1 - pbase;
                    if ((mb1 >> (unsigned int)ms1 & 1) == 0) {
                        lim1 = 0;
                    }
                    float mx0 = -CAKE_INF;
                    float mx1 = -CAKE_INF;
                    int kk = kcol;
                    float v0 = -CAKE_INF;
                    if (kk < lim0) {
                        v0 = d_s[0];
                    }
                    d_s[0] = v0;
                    float v1 = -CAKE_INF;
                    if (kk < lim1) {
                        v1 = d_s[2];
                    }
                    d_s[2] = v1;
                    float _max_0 = max_noftz(mx0, v0);
                    mx0 = _max_0;
                    float _max_1 = max_noftz(mx1, v1);
                    mx1 = _max_1;
                    int kk_0 = kcol + 1;
                    float v0_1 = -CAKE_INF;
                    if (kk_0 < lim0) {
                        v0_1 = d_s[1];
                    }
                    d_s[1] = v0_1;
                    float v1_2 = -CAKE_INF;
                    if (kk_0 < lim1) {
                        v1_2 = d_s[3];
                    }
                    d_s[3] = v1_2;
                    float _max_2 = max_noftz(mx0, v0_1);
                    mx0 = _max_2;
                    float _max_3 = max_noftz(mx1, v1_2);
                    mx1 = _max_3;
                    int kk_3 = kcol + 8;
                    float v0_4 = -CAKE_INF;
                    if (kk_3 < lim0) {
                        v0_4 = d_s[4];
                    }
                    d_s[4] = v0_4;
                    float v1_5 = -CAKE_INF;
                    if (kk_3 < lim1) {
                        v1_5 = d_s[6];
                    }
                    d_s[6] = v1_5;
                    float _max_4 = max_noftz(mx0, v0_4);
                    mx0 = _max_4;
                    float _max_5 = max_noftz(mx1, v1_5);
                    mx1 = _max_5;
                    int kk_6 = kcol + 9;
                    float v0_7 = -CAKE_INF;
                    if (kk_6 < lim0) {
                        v0_7 = d_s[5];
                    }
                    d_s[5] = v0_7;
                    float v1_8 = -CAKE_INF;
                    if (kk_6 < lim1) {
                        v1_8 = d_s[7];
                    }
                    d_s[7] = v1_8;
                    float _max_6 = max_noftz(mx0, v0_7);
                    mx0 = _max_6;
                    float _max_7 = max_noftz(mx1, v1_8);
                    mx1 = _max_7;
                    int kk_9 = kcol + 16;
                    float v0_10 = -CAKE_INF;
                    if (kk_9 < lim0) {
                        v0_10 = d_s[8];
                    }
                    d_s[8] = v0_10;
                    float v1_11 = -CAKE_INF;
                    if (kk_9 < lim1) {
                        v1_11 = d_s[10];
                    }
                    d_s[10] = v1_11;
                    float _max_8 = max_noftz(mx0, v0_10);
                    mx0 = _max_8;
                    float _max_9 = max_noftz(mx1, v1_11);
                    mx1 = _max_9;
                    int kk_12 = kcol + 17;
                    float v0_13 = -CAKE_INF;
                    if (kk_12 < lim0) {
                        v0_13 = d_s[9];
                    }
                    d_s[9] = v0_13;
                    float v1_14 = -CAKE_INF;
                    if (kk_12 < lim1) {
                        v1_14 = d_s[11];
                    }
                    d_s[11] = v1_14;
                    float _max_10 = max_noftz(mx0, v0_13);
                    mx0 = _max_10;
                    float _max_11 = max_noftz(mx1, v1_14);
                    mx1 = _max_11;
                    int kk_15 = kcol + 24;
                    float v0_16 = -CAKE_INF;
                    if (kk_15 < lim0) {
                        v0_16 = d_s[12];
                    }
                    d_s[12] = v0_16;
                    float v1_17 = -CAKE_INF;
                    if (kk_15 < lim1) {
                        v1_17 = d_s[14];
                    }
                    d_s[14] = v1_17;
                    float _max_12 = max_noftz(mx0, v0_16);
                    mx0 = _max_12;
                    float _max_13 = max_noftz(mx1, v1_17);
                    mx1 = _max_13;
                    int kk_18 = kcol + 25;
                    float v0_19 = -CAKE_INF;
                    if (kk_18 < lim0) {
                        v0_19 = d_s[13];
                    }
                    d_s[13] = v0_19;
                    float v1_20 = -CAKE_INF;
                    if (kk_18 < lim1) {
                        v1_20 = d_s[15];
                    }
                    d_s[15] = v1_20;
                    float _max_14 = max_noftz(mx0, v0_19);
                    mx0 = _max_14;
                    float _max_15 = max_noftz(mx1, v1_20);
                    mx1 = _max_15;
                    int kk_21 = kcol + 32;
                    float v0_22 = -CAKE_INF;
                    if (kk_21 < lim0) {
                        v0_22 = d_s[16];
                    }
                    d_s[16] = v0_22;
                    float v1_23 = -CAKE_INF;
                    if (kk_21 < lim1) {
                        v1_23 = d_s[18];
                    }
                    d_s[18] = v1_23;
                    float _max_16 = max_noftz(mx0, v0_22);
                    mx0 = _max_16;
                    float _max_17 = max_noftz(mx1, v1_23);
                    mx1 = _max_17;
                    int kk_24 = kcol + 33;
                    float v0_25 = -CAKE_INF;
                    if (kk_24 < lim0) {
                        v0_25 = d_s[17];
                    }
                    d_s[17] = v0_25;
                    float v1_26 = -CAKE_INF;
                    if (kk_24 < lim1) {
                        v1_26 = d_s[19];
                    }
                    d_s[19] = v1_26;
                    float _max_18 = max_noftz(mx0, v0_25);
                    mx0 = _max_18;
                    float _max_19 = max_noftz(mx1, v1_26);
                    mx1 = _max_19;
                    int kk_27 = kcol + 40;
                    float v0_28 = -CAKE_INF;
                    if (kk_27 < lim0) {
                        v0_28 = d_s[20];
                    }
                    d_s[20] = v0_28;
                    float v1_29 = -CAKE_INF;
                    if (kk_27 < lim1) {
                        v1_29 = d_s[22];
                    }
                    d_s[22] = v1_29;
                    float _max_20 = max_noftz(mx0, v0_28);
                    mx0 = _max_20;
                    float _max_21 = max_noftz(mx1, v1_29);
                    mx1 = _max_21;
                    int kk_30 = kcol + 41;
                    float v0_31 = -CAKE_INF;
                    if (kk_30 < lim0) {
                        v0_31 = d_s[21];
                    }
                    d_s[21] = v0_31;
                    float v1_32 = -CAKE_INF;
                    if (kk_30 < lim1) {
                        v1_32 = d_s[23];
                    }
                    d_s[23] = v1_32;
                    float _max_22 = max_noftz(mx0, v0_31);
                    mx0 = _max_22;
                    float _max_23 = max_noftz(mx1, v1_32);
                    mx1 = _max_23;
                    int kk_33 = kcol + 48;
                    float v0_34 = -CAKE_INF;
                    if (kk_33 < lim0) {
                        v0_34 = d_s[24];
                    }
                    d_s[24] = v0_34;
                    float v1_35 = -CAKE_INF;
                    if (kk_33 < lim1) {
                        v1_35 = d_s[26];
                    }
                    d_s[26] = v1_35;
                    float _max_24 = max_noftz(mx0, v0_34);
                    mx0 = _max_24;
                    float _max_25 = max_noftz(mx1, v1_35);
                    mx1 = _max_25;
                    int kk_36 = kcol + 49;
                    float v0_37 = -CAKE_INF;
                    if (kk_36 < lim0) {
                        v0_37 = d_s[25];
                    }
                    d_s[25] = v0_37;
                    float v1_38 = -CAKE_INF;
                    if (kk_36 < lim1) {
                        v1_38 = d_s[27];
                    }
                    d_s[27] = v1_38;
                    float _max_26 = max_noftz(mx0, v0_37);
                    mx0 = _max_26;
                    float _max_27 = max_noftz(mx1, v1_38);
                    mx1 = _max_27;
                    int kk_39 = kcol + 56;
                    float v0_40 = -CAKE_INF;
                    if (kk_39 < lim0) {
                        v0_40 = d_s[28];
                    }
                    d_s[28] = v0_40;
                    float v1_41 = -CAKE_INF;
                    if (kk_39 < lim1) {
                        v1_41 = d_s[30];
                    }
                    d_s[30] = v1_41;
                    float _max_28 = max_noftz(mx0, v0_40);
                    mx0 = _max_28;
                    float _max_29 = max_noftz(mx1, v1_41);
                    mx1 = _max_29;
                    int kk_42 = kcol + 57;
                    float v0_43 = -CAKE_INF;
                    if (kk_42 < lim0) {
                        v0_43 = d_s[29];
                    }
                    d_s[29] = v0_43;
                    float v1_44 = -CAKE_INF;
                    if (kk_42 < lim1) {
                        v1_44 = d_s[31];
                    }
                    d_s[31] = v1_44;
                    float _max_30 = max_noftz(mx0, v0_43);
                    mx0 = _max_30;
                    float _max_31 = max_noftz(mx1, v1_44);
                    mx1 = _max_31;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, mx0, 1);
                    float _max_32 = max_noftz(mx0, _shfl_xor_0);
                    mx0 = _max_32;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, mx0, 2);
                    float _max_33 = max_noftz(mx0, _shfl_xor_1);
                    mx0 = _max_33;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, mx1, 1);
                    float _max_34 = max_noftz(mx1, _shfl_xor_2);
                    mx1 = _max_34;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, mx1, 2);
                    float _max_35 = max_noftz(mx1, _shfl_xor_3);
                    mx1 = _max_35;
                    float _max_36 = max_noftz(mrun0, mx0 * softmax_scale_log2);
                    float mnew0 = _max_36;
                    float _max_37 = max_noftz(mrun1, mx1 * softmax_scale_log2);
                    float mnew1 = _max_37;
                    float corr0 = 0.0f;
                    if (mrun0 != -CAKE_INF) {
                        float _exp2_0 = approx_exp2(mrun0 - mnew0);
                        corr0 = _exp2_0;
                    }
                    float corr1 = 0.0f;
                    if (mrun1 != -CAKE_INF) {
                        float _exp2_1 = approx_exp2(mrun1 - mnew1);
                        corr1 = _exp2_1;
                    }
                    d_o[0] = d_o[0] * corr0;
                    d_o[1] = d_o[1] * corr0;
                    d_o[2] = d_o[2] * corr1;
                    d_o[3] = d_o[3] * corr1;
                    d_o[4] = d_o[4] * corr0;
                    d_o[5] = d_o[5] * corr0;
                    d_o[6] = d_o[6] * corr1;
                    d_o[7] = d_o[7] * corr1;
                    d_o[8] = d_o[8] * corr0;
                    d_o[9] = d_o[9] * corr0;
                    d_o[10] = d_o[10] * corr1;
                    d_o[11] = d_o[11] * corr1;
                    d_o[12] = d_o[12] * corr0;
                    d_o[13] = d_o[13] * corr0;
                    d_o[14] = d_o[14] * corr1;
                    d_o[15] = d_o[15] * corr1;
                    d_o[16] = d_o[16] * corr0;
                    d_o[17] = d_o[17] * corr0;
                    d_o[18] = d_o[18] * corr1;
                    d_o[19] = d_o[19] * corr1;
                    d_o[20] = d_o[20] * corr0;
                    d_o[21] = d_o[21] * corr0;
                    d_o[22] = d_o[22] * corr1;
                    d_o[23] = d_o[23] * corr1;
                    d_o[24] = d_o[24] * corr0;
                    d_o[25] = d_o[25] * corr0;
                    d_o[26] = d_o[26] * corr1;
                    d_o[27] = d_o[27] * corr1;
                    d_o[28] = d_o[28] * corr0;
                    d_o[29] = d_o[29] * corr0;
                    d_o[30] = d_o[30] * corr1;
                    d_o[31] = d_o[31] * corr1;
                    d_o[32] = d_o[32] * corr0;
                    d_o[33] = d_o[33] * corr0;
                    d_o[34] = d_o[34] * corr1;
                    d_o[35] = d_o[35] * corr1;
                    d_o[36] = d_o[36] * corr0;
                    d_o[37] = d_o[37] * corr0;
                    d_o[38] = d_o[38] * corr1;
                    d_o[39] = d_o[39] * corr1;
                    d_o[40] = d_o[40] * corr0;
                    d_o[41] = d_o[41] * corr0;
                    d_o[42] = d_o[42] * corr1;
                    d_o[43] = d_o[43] * corr1;
                    d_o[44] = d_o[44] * corr0;
                    d_o[45] = d_o[45] * corr0;
                    d_o[46] = d_o[46] * corr1;
                    d_o[47] = d_o[47] * corr1;
                    d_o[48] = d_o[48] * corr0;
                    d_o[49] = d_o[49] * corr0;
                    d_o[50] = d_o[50] * corr1;
                    d_o[51] = d_o[51] * corr1;
                    d_o[52] = d_o[52] * corr0;
                    d_o[53] = d_o[53] * corr0;
                    d_o[54] = d_o[54] * corr1;
                    d_o[55] = d_o[55] * corr1;
                    d_o[56] = d_o[56] * corr0;
                    d_o[57] = d_o[57] * corr0;
                    d_o[58] = d_o[58] * corr1;
                    d_o[59] = d_o[59] * corr1;
                    d_o[60] = d_o[60] * corr0;
                    d_o[61] = d_o[61] * corr0;
                    d_o[62] = d_o[62] * corr1;
                    d_o[63] = d_o[63] * corr1;
                    l0 = l0 * corr0;
                    l1 = l1 * corr1;
                    mrun0 = mnew0;
                    mrun1 = mnew1;
                    float msub0 = mnew0;
                    if (msub0 == -CAKE_INF) {
                        msub0 = 0.0f;
                    }
                    float msub1 = mnew1;
                    if (msub1 == -CAKE_INF) {
                        msub1 = 0.0f;
                    }
                    float _exp2_2 = approx_exp2(d_s[0] * softmax_scale_log2 - msub0);
                    float p00 = _exp2_2;
                    float _exp2_3 = approx_exp2(d_s[1] * softmax_scale_log2 - msub0);
                    float p01 = _exp2_3;
                    float _exp2_4 = approx_exp2(d_s[2] * softmax_scale_log2 - msub1);
                    float p10 = _exp2_4;
                    float _exp2_5 = approx_exp2(d_s[3] * softmax_scale_log2 - msub1);
                    float p11 = _exp2_5;
                    l0 = l0 + (p00 + p01);
                    l1 = l1 + (p10 + p11);
                    uint32_t _f16x2_pack_0;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_0) : "f"(p01), "f"(p00));
                    pk[0] = _f16x2_pack_0;
                    uint32_t _f16x2_pack_1;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_1) : "f"(p11), "f"(p10));
                    pk[1] = _f16x2_pack_1;
                    float _exp2_6 = approx_exp2(d_s[4] * softmax_scale_log2 - msub0);
                    float p00_45 = _exp2_6;
                    float _exp2_7 = approx_exp2(d_s[5] * softmax_scale_log2 - msub0);
                    float p01_46 = _exp2_7;
                    float _exp2_8 = approx_exp2(d_s[6] * softmax_scale_log2 - msub1);
                    float p10_47 = _exp2_8;
                    float _exp2_9 = approx_exp2(d_s[7] * softmax_scale_log2 - msub1);
                    float p11_48 = _exp2_9;
                    l0 = l0 + (p00_45 + p01_46);
                    l1 = l1 + (p10_47 + p11_48);
                    uint32_t _f16x2_pack_2;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_2) : "f"(p01_46), "f"(p00_45));
                    pk[2] = _f16x2_pack_2;
                    uint32_t _f16x2_pack_3;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_3) : "f"(p11_48), "f"(p10_47));
                    pk[3] = _f16x2_pack_3;
                    float _exp2_10 = approx_exp2(d_s[8] * softmax_scale_log2 - msub0);
                    float p00_49 = _exp2_10;
                    float _exp2_11 = approx_exp2(d_s[9] * softmax_scale_log2 - msub0);
                    float p01_50 = _exp2_11;
                    float _exp2_12 = approx_exp2(d_s[10] * softmax_scale_log2 - msub1);
                    float p10_51 = _exp2_12;
                    float _exp2_13 = approx_exp2(d_s[11] * softmax_scale_log2 - msub1);
                    float p11_52 = _exp2_13;
                    l0 = l0 + (p00_49 + p01_50);
                    l1 = l1 + (p10_51 + p11_52);
                    uint32_t _f16x2_pack_4;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_4) : "f"(p01_50), "f"(p00_49));
                    pk[4] = _f16x2_pack_4;
                    uint32_t _f16x2_pack_5;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_5) : "f"(p11_52), "f"(p10_51));
                    pk[5] = _f16x2_pack_5;
                    float _exp2_14 = approx_exp2(d_s[12] * softmax_scale_log2 - msub0);
                    float p00_53 = _exp2_14;
                    float _exp2_15 = approx_exp2(d_s[13] * softmax_scale_log2 - msub0);
                    float p01_54 = _exp2_15;
                    float _exp2_16 = approx_exp2(d_s[14] * softmax_scale_log2 - msub1);
                    float p10_55 = _exp2_16;
                    float _exp2_17 = approx_exp2(d_s[15] * softmax_scale_log2 - msub1);
                    float p11_56 = _exp2_17;
                    l0 = l0 + (p00_53 + p01_54);
                    l1 = l1 + (p10_55 + p11_56);
                    uint32_t _f16x2_pack_6;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_6) : "f"(p01_54), "f"(p00_53));
                    pk[6] = _f16x2_pack_6;
                    uint32_t _f16x2_pack_7;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_7) : "f"(p11_56), "f"(p10_55));
                    pk[7] = _f16x2_pack_7;
                    float _exp2_18 = approx_exp2(d_s[16] * softmax_scale_log2 - msub0);
                    float p00_57 = _exp2_18;
                    float _exp2_19 = approx_exp2(d_s[17] * softmax_scale_log2 - msub0);
                    float p01_58 = _exp2_19;
                    float _exp2_20 = approx_exp2(d_s[18] * softmax_scale_log2 - msub1);
                    float p10_59 = _exp2_20;
                    float _exp2_21 = approx_exp2(d_s[19] * softmax_scale_log2 - msub1);
                    float p11_60 = _exp2_21;
                    l0 = l0 + (p00_57 + p01_58);
                    l1 = l1 + (p10_59 + p11_60);
                    uint32_t _f16x2_pack_8;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_8) : "f"(p01_58), "f"(p00_57));
                    pk[8] = _f16x2_pack_8;
                    uint32_t _f16x2_pack_9;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_9) : "f"(p11_60), "f"(p10_59));
                    pk[9] = _f16x2_pack_9;
                    float _exp2_22 = approx_exp2(d_s[20] * softmax_scale_log2 - msub0);
                    float p00_61 = _exp2_22;
                    float _exp2_23 = approx_exp2(d_s[21] * softmax_scale_log2 - msub0);
                    float p01_62 = _exp2_23;
                    float _exp2_24 = approx_exp2(d_s[22] * softmax_scale_log2 - msub1);
                    float p10_63 = _exp2_24;
                    float _exp2_25 = approx_exp2(d_s[23] * softmax_scale_log2 - msub1);
                    float p11_64 = _exp2_25;
                    l0 = l0 + (p00_61 + p01_62);
                    l1 = l1 + (p10_63 + p11_64);
                    uint32_t _f16x2_pack_10;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_10) : "f"(p01_62), "f"(p00_61));
                    pk[10] = _f16x2_pack_10;
                    uint32_t _f16x2_pack_11;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_11) : "f"(p11_64), "f"(p10_63));
                    pk[11] = _f16x2_pack_11;
                    float _exp2_26 = approx_exp2(d_s[24] * softmax_scale_log2 - msub0);
                    float p00_65 = _exp2_26;
                    float _exp2_27 = approx_exp2(d_s[25] * softmax_scale_log2 - msub0);
                    float p01_66 = _exp2_27;
                    float _exp2_28 = approx_exp2(d_s[26] * softmax_scale_log2 - msub1);
                    float p10_67 = _exp2_28;
                    float _exp2_29 = approx_exp2(d_s[27] * softmax_scale_log2 - msub1);
                    float p11_68 = _exp2_29;
                    l0 = l0 + (p00_65 + p01_66);
                    l1 = l1 + (p10_67 + p11_68);
                    uint32_t _f16x2_pack_12;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_12) : "f"(p01_66), "f"(p00_65));
                    pk[12] = _f16x2_pack_12;
                    uint32_t _f16x2_pack_13;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_13) : "f"(p11_68), "f"(p10_67));
                    pk[13] = _f16x2_pack_13;
                    float _exp2_30 = approx_exp2(d_s[28] * softmax_scale_log2 - msub0);
                    float p00_69 = _exp2_30;
                    float _exp2_31 = approx_exp2(d_s[29] * softmax_scale_log2 - msub0);
                    float p01_70 = _exp2_31;
                    float _exp2_32 = approx_exp2(d_s[30] * softmax_scale_log2 - msub1);
                    float p10_71 = _exp2_32;
                    float _exp2_33 = approx_exp2(d_s[31] * softmax_scale_log2 - msub1);
                    float p11_72 = _exp2_33;
                    l0 = l0 + (p00_69 + p01_70);
                    l1 = l1 + (p10_71 + p11_72);
                    uint32_t _f16x2_pack_14;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_14) : "f"(p01_70), "f"(p00_69));
                    pk[14] = _f16x2_pack_14;
                    uint32_t _f16x2_pack_15;
                    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_15) : "f"(p11_72), "f"(p10_71));
                    pk[15] = _f16x2_pack_15;
                }
                if (qk_go != 0) {
                    asm volatile("" : "+r"(pk[0]) :: "memory");
                    asm volatile("" : "+r"(pk[1]) :: "memory");
                    asm volatile("" : "+r"(pk[2]) :: "memory");
                    asm volatile("" : "+r"(pk[3]) :: "memory");
                    asm volatile("" : "+r"(pk[4]) :: "memory");
                    asm volatile("" : "+r"(pk[5]) :: "memory");
                    asm volatile("" : "+r"(pk[6]) :: "memory");
                    asm volatile("" : "+r"(pk[7]) :: "memory");
                    asm volatile("" : "+r"(pk[8]) :: "memory");
                    asm volatile("" : "+r"(pk[9]) :: "memory");
                    asm volatile("" : "+r"(pk[10]) :: "memory");
                    asm volatile("" : "+r"(pk[11]) :: "memory");
                    asm volatile("" : "+r"(pk[12]) :: "memory");
                    asm volatile("" : "+r"(pk[13]) :: "memory");
                    asm volatile("" : "+r"(pk[14]) :: "memory");
                    asm volatile("" : "+r"(pk[15]) :: "memory");
                    asm volatile("" : "+f"(d_o[0]) :: "memory");
                    asm volatile("" : "+f"(d_o[1]) :: "memory");
                    asm volatile("" : "+f"(d_o[2]) :: "memory");
                    asm volatile("" : "+f"(d_o[3]) :: "memory");
                    asm volatile("" : "+f"(d_o[4]) :: "memory");
                    asm volatile("" : "+f"(d_o[5]) :: "memory");
                    asm volatile("" : "+f"(d_o[6]) :: "memory");
                    asm volatile("" : "+f"(d_o[7]) :: "memory");
                    asm volatile("" : "+f"(d_o[8]) :: "memory");
                    asm volatile("" : "+f"(d_o[9]) :: "memory");
                    asm volatile("" : "+f"(d_o[10]) :: "memory");
                    asm volatile("" : "+f"(d_o[11]) :: "memory");
                    asm volatile("" : "+f"(d_o[12]) :: "memory");
                    asm volatile("" : "+f"(d_o[13]) :: "memory");
                    asm volatile("" : "+f"(d_o[14]) :: "memory");
                    asm volatile("" : "+f"(d_o[15]) :: "memory");
                    asm volatile("" : "+f"(d_o[16]) :: "memory");
                    asm volatile("" : "+f"(d_o[17]) :: "memory");
                    asm volatile("" : "+f"(d_o[18]) :: "memory");
                    asm volatile("" : "+f"(d_o[19]) :: "memory");
                    asm volatile("" : "+f"(d_o[20]) :: "memory");
                    asm volatile("" : "+f"(d_o[21]) :: "memory");
                    asm volatile("" : "+f"(d_o[22]) :: "memory");
                    asm volatile("" : "+f"(d_o[23]) :: "memory");
                    asm volatile("" : "+f"(d_o[24]) :: "memory");
                    asm volatile("" : "+f"(d_o[25]) :: "memory");
                    asm volatile("" : "+f"(d_o[26]) :: "memory");
                    asm volatile("" : "+f"(d_o[27]) :: "memory");
                    asm volatile("" : "+f"(d_o[28]) :: "memory");
                    asm volatile("" : "+f"(d_o[29]) :: "memory");
                    asm volatile("" : "+f"(d_o[30]) :: "memory");
                    asm volatile("" : "+f"(d_o[31]) :: "memory");
                    asm volatile("" : "+f"(d_o[32]) :: "memory");
                    asm volatile("" : "+f"(d_o[33]) :: "memory");
                    asm volatile("" : "+f"(d_o[34]) :: "memory");
                    asm volatile("" : "+f"(d_o[35]) :: "memory");
                    asm volatile("" : "+f"(d_o[36]) :: "memory");
                    asm volatile("" : "+f"(d_o[37]) :: "memory");
                    asm volatile("" : "+f"(d_o[38]) :: "memory");
                    asm volatile("" : "+f"(d_o[39]) :: "memory");
                    asm volatile("" : "+f"(d_o[40]) :: "memory");
                    asm volatile("" : "+f"(d_o[41]) :: "memory");
                    asm volatile("" : "+f"(d_o[42]) :: "memory");
                    asm volatile("" : "+f"(d_o[43]) :: "memory");
                    asm volatile("" : "+f"(d_o[44]) :: "memory");
                    asm volatile("" : "+f"(d_o[45]) :: "memory");
                    asm volatile("" : "+f"(d_o[46]) :: "memory");
                    asm volatile("" : "+f"(d_o[47]) :: "memory");
                    asm volatile("" : "+f"(d_o[48]) :: "memory");
                    asm volatile("" : "+f"(d_o[49]) :: "memory");
                    asm volatile("" : "+f"(d_o[50]) :: "memory");
                    asm volatile("" : "+f"(d_o[51]) :: "memory");
                    asm volatile("" : "+f"(d_o[52]) :: "memory");
                    asm volatile("" : "+f"(d_o[53]) :: "memory");
                    asm volatile("" : "+f"(d_o[54]) :: "memory");
                    asm volatile("" : "+f"(d_o[55]) :: "memory");
                    asm volatile("" : "+f"(d_o[56]) :: "memory");
                    asm volatile("" : "+f"(d_o[57]) :: "memory");
                    asm volatile("" : "+f"(d_o[58]) :: "memory");
                    asm volatile("" : "+f"(d_o[59]) :: "memory");
                    asm volatile("" : "+f"(d_o[60]) :: "memory");
                    asm volatile("" : "+f"(d_o[61]) :: "memory");
                    asm volatile("" : "+f"(d_o[62]) :: "memory");
                    asm volatile("" : "+f"(d_o[63]) :: "memory");
                    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                    uint64_t _wgmma_desc_3 = (((uint64_t)(((v16t_addr + (unsigned int)(fb_1 * 16384))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                    uint64_t _wgmma_b_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]), "l"(_wgmma_b_0_5)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(pk[4]), "r"(pk[(4) + 1]), "r"(pk[(4) + 2]), "r"(pk[(4) + 3]), "l"(_wgmma_b_0_5 + 2)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(pk[8]), "r"(pk[(8) + 1]), "r"(pk[(8) + 2]), "r"(pk[(8) + 3]), "l"(_wgmma_b_0_5 + 4)
                        );
                    asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n"
                        : "+f"(d_o[0]), "+f"(d_o[1]), "+f"(d_o[2]), "+f"(d_o[3]), "+f"(d_o[4]), "+f"(d_o[5]), "+f"(d_o[6]), "+f"(d_o[7]), "+f"(d_o[8]), "+f"(d_o[9]), "+f"(d_o[10]), "+f"(d_o[11]), "+f"(d_o[12]), "+f"(d_o[13]), "+f"(d_o[14]), "+f"(d_o[15]), "+f"(d_o[16]), "+f"(d_o[17]), "+f"(d_o[18]), "+f"(d_o[19]), "+f"(d_o[20]), "+f"(d_o[21]), "+f"(d_o[22]), "+f"(d_o[23]), "+f"(d_o[24]), "+f"(d_o[25]), "+f"(d_o[26]), "+f"(d_o[27]), "+f"(d_o[28]), "+f"(d_o[29]), "+f"(d_o[30]), "+f"(d_o[31]), "+f"(d_o[32]), "+f"(d_o[33]), "+f"(d_o[34]), "+f"(d_o[35]), "+f"(d_o[36]), "+f"(d_o[37]), "+f"(d_o[38]), "+f"(d_o[39]), "+f"(d_o[40]), "+f"(d_o[41]), "+f"(d_o[42]), "+f"(d_o[43]), "+f"(d_o[44]), "+f"(d_o[45]), "+f"(d_o[46]), "+f"(d_o[47]), "+f"(d_o[48]), "+f"(d_o[49]), "+f"(d_o[50]), "+f"(d_o[51]), "+f"(d_o[52]), "+f"(d_o[53]), "+f"(d_o[54]), "+f"(d_o[55]), "+f"(d_o[56]), "+f"(d_o[57]), "+f"(d_o[58]), "+f"(d_o[59]), "+f"(d_o[60]), "+f"(d_o[61]), "+f"(d_o[62]), "+f"(d_o[63])
                        : "r"(pk[12]), "r"(pk[(12) + 1]), "r"(pk[(12) + 2]), "r"(pk[(12) + 3]), "l"(_wgmma_b_0_5 + 6)
                        );
                    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                    asm volatile("" : "+f"(d_o[0]) :: "memory");
                    asm volatile("" : "+f"(d_o[1]) :: "memory");
                    asm volatile("" : "+f"(d_o[2]) :: "memory");
                    asm volatile("" : "+f"(d_o[3]) :: "memory");
                    asm volatile("" : "+f"(d_o[4]) :: "memory");
                    asm volatile("" : "+f"(d_o[5]) :: "memory");
                    asm volatile("" : "+f"(d_o[6]) :: "memory");
                    asm volatile("" : "+f"(d_o[7]) :: "memory");
                    asm volatile("" : "+f"(d_o[8]) :: "memory");
                    asm volatile("" : "+f"(d_o[9]) :: "memory");
                    asm volatile("" : "+f"(d_o[10]) :: "memory");
                    asm volatile("" : "+f"(d_o[11]) :: "memory");
                    asm volatile("" : "+f"(d_o[12]) :: "memory");
                    asm volatile("" : "+f"(d_o[13]) :: "memory");
                    asm volatile("" : "+f"(d_o[14]) :: "memory");
                    asm volatile("" : "+f"(d_o[15]) :: "memory");
                    asm volatile("" : "+f"(d_o[16]) :: "memory");
                    asm volatile("" : "+f"(d_o[17]) :: "memory");
                    asm volatile("" : "+f"(d_o[18]) :: "memory");
                    asm volatile("" : "+f"(d_o[19]) :: "memory");
                    asm volatile("" : "+f"(d_o[20]) :: "memory");
                    asm volatile("" : "+f"(d_o[21]) :: "memory");
                    asm volatile("" : "+f"(d_o[22]) :: "memory");
                    asm volatile("" : "+f"(d_o[23]) :: "memory");
                    asm volatile("" : "+f"(d_o[24]) :: "memory");
                    asm volatile("" : "+f"(d_o[25]) :: "memory");
                    asm volatile("" : "+f"(d_o[26]) :: "memory");
                    asm volatile("" : "+f"(d_o[27]) :: "memory");
                    asm volatile("" : "+f"(d_o[28]) :: "memory");
                    asm volatile("" : "+f"(d_o[29]) :: "memory");
                    asm volatile("" : "+f"(d_o[30]) :: "memory");
                    asm volatile("" : "+f"(d_o[31]) :: "memory");
                    asm volatile("" : "+f"(d_o[32]) :: "memory");
                    asm volatile("" : "+f"(d_o[33]) :: "memory");
                    asm volatile("" : "+f"(d_o[34]) :: "memory");
                    asm volatile("" : "+f"(d_o[35]) :: "memory");
                    asm volatile("" : "+f"(d_o[36]) :: "memory");
                    asm volatile("" : "+f"(d_o[37]) :: "memory");
                    asm volatile("" : "+f"(d_o[38]) :: "memory");
                    asm volatile("" : "+f"(d_o[39]) :: "memory");
                    asm volatile("" : "+f"(d_o[40]) :: "memory");
                    asm volatile("" : "+f"(d_o[41]) :: "memory");
                    asm volatile("" : "+f"(d_o[42]) :: "memory");
                    asm volatile("" : "+f"(d_o[43]) :: "memory");
                    asm volatile("" : "+f"(d_o[44]) :: "memory");
                    asm volatile("" : "+f"(d_o[45]) :: "memory");
                    asm volatile("" : "+f"(d_o[46]) :: "memory");
                    asm volatile("" : "+f"(d_o[47]) :: "memory");
                    asm volatile("" : "+f"(d_o[48]) :: "memory");
                    asm volatile("" : "+f"(d_o[49]) :: "memory");
                    asm volatile("" : "+f"(d_o[50]) :: "memory");
                    asm volatile("" : "+f"(d_o[51]) :: "memory");
                    asm volatile("" : "+f"(d_o[52]) :: "memory");
                    asm volatile("" : "+f"(d_o[53]) :: "memory");
                    asm volatile("" : "+f"(d_o[54]) :: "memory");
                    asm volatile("" : "+f"(d_o[55]) :: "memory");
                    asm volatile("" : "+f"(d_o[56]) :: "memory");
                    asm volatile("" : "+f"(d_o[57]) :: "memory");
                    asm volatile("" : "+f"(d_o[58]) :: "memory");
                    asm volatile("" : "+f"(d_o[59]) :: "memory");
                    asm volatile("" : "+f"(d_o[60]) :: "memory");
                    asm volatile("" : "+f"(d_o[61]) :: "memory");
                    asm volatile("" : "+f"(d_o[62]) :: "memory");
                    asm volatile("" : "+f"(d_o[63]) :: "memory");
                    asm volatile("" : "+r"(pk[0]) :: "memory");
                    asm volatile("" : "+r"(pk[1]) :: "memory");
                    asm volatile("" : "+r"(pk[2]) :: "memory");
                    asm volatile("" : "+r"(pk[3]) :: "memory");
                    asm volatile("" : "+r"(pk[4]) :: "memory");
                    asm volatile("" : "+r"(pk[5]) :: "memory");
                    asm volatile("" : "+r"(pk[6]) :: "memory");
                    asm volatile("" : "+r"(pk[7]) :: "memory");
                    asm volatile("" : "+r"(pk[8]) :: "memory");
                    asm volatile("" : "+r"(pk[9]) :: "memory");
                    asm volatile("" : "+r"(pk[10]) :: "memory");
                    asm volatile("" : "+r"(pk[11]) :: "memory");
                    asm volatile("" : "+r"(pk[12]) :: "memory");
                    asm volatile("" : "+r"(pk[13]) :: "memory");
                    asm volatile("" : "+r"(pk[14]) :: "memory");
                    asm volatile("" : "+r"(pk[15]) :: "memory");
                    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
                }
                if (wi == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(f16_empty_addr + (fb_1) * 8);
                    }
                }
                first_c = 0;
            }
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            asm volatile("" : "+f"(d_o[0]) :: "memory");
            asm volatile("" : "+f"(d_o[1]) :: "memory");
            asm volatile("" : "+f"(d_o[2]) :: "memory");
            asm volatile("" : "+f"(d_o[3]) :: "memory");
            asm volatile("" : "+f"(d_o[4]) :: "memory");
            asm volatile("" : "+f"(d_o[5]) :: "memory");
            asm volatile("" : "+f"(d_o[6]) :: "memory");
            asm volatile("" : "+f"(d_o[7]) :: "memory");
            asm volatile("" : "+f"(d_o[8]) :: "memory");
            asm volatile("" : "+f"(d_o[9]) :: "memory");
            asm volatile("" : "+f"(d_o[10]) :: "memory");
            asm volatile("" : "+f"(d_o[11]) :: "memory");
            asm volatile("" : "+f"(d_o[12]) :: "memory");
            asm volatile("" : "+f"(d_o[13]) :: "memory");
            asm volatile("" : "+f"(d_o[14]) :: "memory");
            asm volatile("" : "+f"(d_o[15]) :: "memory");
            asm volatile("" : "+f"(d_o[16]) :: "memory");
            asm volatile("" : "+f"(d_o[17]) :: "memory");
            asm volatile("" : "+f"(d_o[18]) :: "memory");
            asm volatile("" : "+f"(d_o[19]) :: "memory");
            asm volatile("" : "+f"(d_o[20]) :: "memory");
            asm volatile("" : "+f"(d_o[21]) :: "memory");
            asm volatile("" : "+f"(d_o[22]) :: "memory");
            asm volatile("" : "+f"(d_o[23]) :: "memory");
            asm volatile("" : "+f"(d_o[24]) :: "memory");
            asm volatile("" : "+f"(d_o[25]) :: "memory");
            asm volatile("" : "+f"(d_o[26]) :: "memory");
            asm volatile("" : "+f"(d_o[27]) :: "memory");
            asm volatile("" : "+f"(d_o[28]) :: "memory");
            asm volatile("" : "+f"(d_o[29]) :: "memory");
            asm volatile("" : "+f"(d_o[30]) :: "memory");
            asm volatile("" : "+f"(d_o[31]) :: "memory");
            asm volatile("" : "+f"(d_o[32]) :: "memory");
            asm volatile("" : "+f"(d_o[33]) :: "memory");
            asm volatile("" : "+f"(d_o[34]) :: "memory");
            asm volatile("" : "+f"(d_o[35]) :: "memory");
            asm volatile("" : "+f"(d_o[36]) :: "memory");
            asm volatile("" : "+f"(d_o[37]) :: "memory");
            asm volatile("" : "+f"(d_o[38]) :: "memory");
            asm volatile("" : "+f"(d_o[39]) :: "memory");
            asm volatile("" : "+f"(d_o[40]) :: "memory");
            asm volatile("" : "+f"(d_o[41]) :: "memory");
            asm volatile("" : "+f"(d_o[42]) :: "memory");
            asm volatile("" : "+f"(d_o[43]) :: "memory");
            asm volatile("" : "+f"(d_o[44]) :: "memory");
            asm volatile("" : "+f"(d_o[45]) :: "memory");
            asm volatile("" : "+f"(d_o[46]) :: "memory");
            asm volatile("" : "+f"(d_o[47]) :: "memory");
            asm volatile("" : "+f"(d_o[48]) :: "memory");
            asm volatile("" : "+f"(d_o[49]) :: "memory");
            asm volatile("" : "+f"(d_o[50]) :: "memory");
            asm volatile("" : "+f"(d_o[51]) :: "memory");
            asm volatile("" : "+f"(d_o[52]) :: "memory");
            asm volatile("" : "+f"(d_o[53]) :: "memory");
            asm volatile("" : "+f"(d_o[54]) :: "memory");
            asm volatile("" : "+f"(d_o[55]) :: "memory");
            asm volatile("" : "+f"(d_o[56]) :: "memory");
            asm volatile("" : "+f"(d_o[57]) :: "memory");
            asm volatile("" : "+f"(d_o[58]) :: "memory");
            asm volatile("" : "+f"(d_o[59]) :: "memory");
            asm volatile("" : "+f"(d_o[60]) :: "memory");
            asm volatile("" : "+f"(d_o[61]) :: "memory");
            asm volatile("" : "+f"(d_o[62]) :: "memory");
            asm volatile("" : "+f"(d_o[63]) :: "memory");
            if (wgi == 0) {
                asm volatile("barrier.sync 9, 256;" ::: "memory");
            }
            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, l0, 1);
            l0 = l0 + _shfl_xor_4;
            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, l0, 2);
            l0 = l0 + _shfl_xor_5;
            float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, l1, 1);
            l1 = l1 + _shfl_xor_6;
            float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, l1, 2);
            l1 = l1 + _shfl_xor_7;
            float inv0 = 0.0f;
            if (l0 != 0.0f) {
                float _rcp_0 = approx_rcp(l0);
                inv0 = _rcp_0;
            }
            float inv1 = 0.0f;
            if (l1 != 0.0f) {
                float _rcp_1 = approx_rcp(l1);
                inv1 = _rcp_1;
            }
            d_o[0] = d_o[0] * inv0;
            d_o[1] = d_o[1] * inv0;
            d_o[2] = d_o[2] * inv1;
            d_o[3] = d_o[3] * inv1;
            d_o[4] = d_o[4] * inv0;
            d_o[5] = d_o[5] * inv0;
            d_o[6] = d_o[6] * inv1;
            d_o[7] = d_o[7] * inv1;
            d_o[8] = d_o[8] * inv0;
            d_o[9] = d_o[9] * inv0;
            d_o[10] = d_o[10] * inv1;
            d_o[11] = d_o[11] * inv1;
            d_o[12] = d_o[12] * inv0;
            d_o[13] = d_o[13] * inv0;
            d_o[14] = d_o[14] * inv1;
            d_o[15] = d_o[15] * inv1;
            d_o[16] = d_o[16] * inv0;
            d_o[17] = d_o[17] * inv0;
            d_o[18] = d_o[18] * inv1;
            d_o[19] = d_o[19] * inv1;
            d_o[20] = d_o[20] * inv0;
            d_o[21] = d_o[21] * inv0;
            d_o[22] = d_o[22] * inv1;
            d_o[23] = d_o[23] * inv1;
            d_o[24] = d_o[24] * inv0;
            d_o[25] = d_o[25] * inv0;
            d_o[26] = d_o[26] * inv1;
            d_o[27] = d_o[27] * inv1;
            d_o[28] = d_o[28] * inv0;
            d_o[29] = d_o[29] * inv0;
            d_o[30] = d_o[30] * inv1;
            d_o[31] = d_o[31] * inv1;
            d_o[32] = d_o[32] * inv0;
            d_o[33] = d_o[33] * inv0;
            d_o[34] = d_o[34] * inv1;
            d_o[35] = d_o[35] * inv1;
            d_o[36] = d_o[36] * inv0;
            d_o[37] = d_o[37] * inv0;
            d_o[38] = d_o[38] * inv1;
            d_o[39] = d_o[39] * inv1;
            d_o[40] = d_o[40] * inv0;
            d_o[41] = d_o[41] * inv0;
            d_o[42] = d_o[42] * inv1;
            d_o[43] = d_o[43] * inv1;
            d_o[44] = d_o[44] * inv0;
            d_o[45] = d_o[45] * inv0;
            d_o[46] = d_o[46] * inv1;
            d_o[47] = d_o[47] * inv1;
            d_o[48] = d_o[48] * inv0;
            d_o[49] = d_o[49] * inv0;
            d_o[50] = d_o[50] * inv1;
            d_o[51] = d_o[51] * inv1;
            d_o[52] = d_o[52] * inv0;
            d_o[53] = d_o[53] * inv0;
            d_o[54] = d_o[54] * inv1;
            d_o[55] = d_o[55] * inv1;
            d_o[56] = d_o[56] * inv0;
            d_o[57] = d_o[57] * inv0;
            d_o[58] = d_o[58] * inv1;
            d_o[59] = d_o[59] * inv1;
            d_o[60] = d_o[60] * inv0;
            d_o[61] = d_o[61] * inv0;
            d_o[62] = d_o[62] * inv1;
            d_o[63] = d_o[63] * inv1;
            if (tl0 < n_tok) {
                int orow0 = (cu_b + tok0 + tl0) * num_q_heads + kv_head_c * 16 + hg0;
                int ob0 = orow0 * 128 + kcol;
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[0 + 0], d_o[0 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ob0))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[4 + 0], d_o[4 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 8)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[8 + 0], d_o[8 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 16)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[12 + 0], d_o[12 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 24)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[16 + 0], d_o[16 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 32)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[20 + 0], d_o[20 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 40)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[24 + 0], d_o[24 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 48)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[28 + 0], d_o[28 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 56)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[32 + 0], d_o[32 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 64)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[36 + 0], d_o[36 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 72)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[40 + 0], d_o[40 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 80)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[44 + 0], d_o[44 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 88)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[48 + 0], d_o[48 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 96)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[52 + 0], d_o[52 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 104)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[56 + 0], d_o[56 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 112)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[60 + 0], d_o[60 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob0 + 120)))[0]) = _pk;
                }
            }
            if (tl1 < n_tok) {
                int orow1 = (cu_b + tok0 + tl1) * num_q_heads + kv_head_c * 16 + hg1;
                int ob1 = orow1 * 128 + kcol;
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[2 + 0], d_o[2 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + ob1))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[6 + 0], d_o[6 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 8)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[10 + 0], d_o[10 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 16)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[14 + 0], d_o[14 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 24)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[18 + 0], d_o[18 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 32)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[22 + 0], d_o[22 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 40)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[26 + 0], d_o[26 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 48)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[30 + 0], d_o[30 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 56)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[34 + 0], d_o[34 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 64)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[38 + 0], d_o[38 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 72)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[42 + 0], d_o[42 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 80)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[46 + 0], d_o[46 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 88)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[50 + 0], d_o[50 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 96)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[54 + 0], d_o[54 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 104)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[58 + 0], d_o[58 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 112)))[0]) = _pk;
                }
                {
                    __nv_bfloat162 _pk = __floats2bfloat162_rn(d_o[62 + 0], d_o[62 + 1]);
                    *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (ob1 + 120)))[0]) = _pk;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
