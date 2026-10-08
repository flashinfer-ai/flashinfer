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
#define SMEM_Q16_STAGE_BYTES 2048
#define SMEM_Q16_STRIDE 2048
#define SMEM_P16_OFF 76800
#define SMEM_P16_STAGE_BYTES 1024
#define SMEM_P16_STRIDE 1024
#define SMEM_RED_OFF 80896
#define SMEM_RED_STAGE_BYTES 128
#define SMEM_RED_STRIDE 128
#define SMEM_LRED_OFF 81024
#define SMEM_LRED_STAGE_BYTES 128
#define SMEM_LRED_STRIDE 128
#define SMEM_FLAG_OFF 81152
#define SMEM_FLAG_STAGE_BYTES 16
#define SMEM_FLAG_STRIDE 16
#define SMEM_META_OFF 81168
#define SMEM_META_STAGE_BYTES 128
#define SMEM_META_STRIDE 128
#define SMEM_TOTAL 81408
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


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(128, LAUNCH_MIN_BLOCKS) void
kernel_cake_hopper_msa_74ca864f90af0cebf05f(unsigned int* __restrict__ Q32, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, int* __restrict__ q2k_indices, int* __restrict__ page_table, int* __restrict__ seqused_k, unsigned int* __restrict__ part_o, float* __restrict__ part_ml, unsigned int* __restrict__ counters, unsigned int* __restrict__ done, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, int max_pages, int num_chunks, float softmax_scale_log2, unsigned int zero_u32, unsigned long long* __restrict__ trace)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define kv_full_addr (mbar_base + 0)
    #define v_full_addr (mbar_base + 16)
    #define mfull_addr (mbar_base + 32)

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
    __half* p16 = reinterpret_cast<__half*>(smem_raw + 76800);
    const int p16_addr = smem + 76800;
    float* red = reinterpret_cast<float*>(smem_raw + 80896);
    const int red_addr = smem + 80896;
    float* lred = reinterpret_cast<float*>(smem_raw + 81024);
    const int lred_addr = smem + 81024;
    unsigned int* flag = reinterpret_cast<unsigned int*>(smem_raw + 81152);
    const int flag_addr = smem + 81152;
    int* meta = reinterpret_cast<int*>(smem_raw + 81168);
    const int meta_addr = smem + 81168;

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 5 barriers)
    // Mbarriers at smem_raw[0..40)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // kv_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // mfull: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
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
    int tok = blockIdx.y;
    int gx = blockIdx.x;
    int gc = gx / 4;
    int split = gx - gc * 4;
    int kv_head = gc / num_chunks;
    int chunk = gc - kv_head * num_chunks;
    int batch = tok / seqlen_q;
    int q_in_batch = tok - batch * seqlen_q;
    int kv_len = seqused_k[batch];
    int q_pos = kv_len - seqlen_q + q_in_batch;
    int group = num_q_heads / num_kv_heads;
    int head0 = kv_head * group + chunk * 8;
    int q_row0 = tok * num_q_heads + head0;
    int heads_valid = group - chunk * 8;
    if (heads_valid > 8) {
        heads_valid = 8;
    }
    int item = (tok * num_kv_heads + kv_head) * num_chunks + chunk;
    int tid_1 = threadIdx.x;
    int lane_0 = lane;
    int warp_1 = warp;
    int g = lane_0 / 4;
    int cq = lane_0 - g * 4;
    int key0 = warp_1 * 16 + g;
    int key1 = key0 + 8;
    if (warp_1 == 1) {
        int pte = lane_0 * 32;
        if (pte < max_pages) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(page_table + (batch * max_pages + pte))));
        }
        int pte_0 = (lane_0 + 32) * 32;
        if (pte_0 < max_pages) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(page_table + (batch * max_pages + pte_0))));
        }
        int pte_1 = (lane_0 + 64) * 32;
        if (pte_1 < max_pages) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(page_table + (batch * max_pages + pte_1))));
        }
        int pte_2 = (lane_0 + 96) * 32;
        if (pte_2 < max_pages) {
            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(page_table + (batch * max_pages + pte_2))));
        }
        if (lane_0 < 16) {
            int blk = q2k_indices[(kv_head * total_q + tok) * 16 + lane_0];
            int page = 0;
            if (blk >= 0) {
                page = page_table[batch * max_pages + blk];
                if (page < 0) {
                    blk = -1;
                    page = 0;
                }
            }
            meta[lane_0] = blk;
            meta[16 + lane_0] = page * num_kv_heads + kv_head;
        }
        __syncwarp();
        int slot = split * 4;
        int blk_1 = meta[slot];
        int page_head = meta[16 + slot];
        int valid = 0;
        if (blk_1 >= 0) {
            valid = q_pos + 1 - blk_1 * 128;
            if (valid > 128) {
                valid = 128;
            }
            if (valid < 0) {
                valid = 0;
            }
        }
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
    if (warp_1 != 1) {
        int sid = tid_1;
        if (warp_1 != 0) {
            sid = tid_1 - 32;
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
        int ich_lin_0 = sid + 96;
        if (ich_lin_0 < 512) {
            int irow_1 = ich_lin_0 / 8;
            int ich_1 = ich_lin_0 - irow_1 * 8;
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
        int ich_lin_1 = sid + 192;
        if (ich_lin_1 < 512) {
            int irow_2 = ich_lin_1 / 8;
            int ich_2 = ich_lin_1 - irow_2 * 8;
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
        int ich_lin_2 = sid + 288;
        if (ich_lin_2 < 512) {
            int irow_3 = ich_lin_2 / 8;
            int ich_3 = ich_lin_2 - irow_3 * 8;
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
        int ich_lin_3 = sid + 384;
        if (ich_lin_3 < 512) {
            int irow_4 = ich_lin_3 / 8;
            int ich_4 = ich_lin_3 - irow_4 * 8;
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
        int ich_lin_4 = sid + 480;
        if (ich_lin_4 < 512) {
            int irow_5 = ich_lin_4 / 8;
            int ich_5 = ich_lin_4 - irow_5 * 8;
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
        unsigned int qw[8];
        int vid = sid;
        int qn = vid / 16;
        int qv = vid - qn * 16;
        qw[0] = 0;
        qw[1] = 0;
        qw[2] = 0;
        qw[3] = 0;
        if (qn < heads_valid) {
            int qbase = (q_row0 + qn) * 64 + qv * 4;
            qw[0] = Q32[qbase];
            qw[1] = Q32[qbase + 1];
            qw[2] = Q32[qbase + 2];
            qw[3] = Q32[qbase + 3];
        }
        int vid_5 = sid + 96;
        int qn_6 = vid_5 / 16;
        int qv_7 = vid_5 - qn_6 * 16;
        qw[4] = 0;
        qw[5] = 0;
        qw[6] = 0;
        qw[7] = 0;
        if (qn_6 < heads_valid) {
            int qbase_1 = (q_row0 + qn_6) * 64 + qv_7 * 4;
            qw[4] = Q32[qbase_1];
            qw[5] = Q32[qbase_1 + 1];
            qw[6] = Q32[qbase_1 + 2];
            qw[7] = Q32[qbase_1 + 3];
        }
        int vid2 = sid;
        if (vid2 < 128) {
            int qn2 = vid2 / 16;
            int qv2 = vid2 - qn2 * 16;
            int qslab = qv2 / 8;
            int qcol = (qv2 - qslab * 8) * 16;
            int qoff = qslab * 1024 + qn2 * 128 + (qcol ^ (qn2 & 7) << 4);
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
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff), "r"(qw[0]), "r"(qw[1]), "r"(qw[2]), "r"(qw[3]) : "memory");
        }
        int vid2_8 = sid + 96;
        if (vid2_8 < 128) {
            int qn2_1 = vid2_8 / 16;
            int qv2_1 = vid2_8 - qn2_1 * 16;
            int qslab_1 = qv2_1 / 8;
            int qcol_1 = (qv2_1 - qslab_1 * 8) * 16;
            int qoff_1 = qslab_1 * 1024 + qn2_1 * 128 + (qcol_1 ^ (qn2_1 & 7) << 4);
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
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(q16_addr + (unsigned int)qoff_1), "r"(qw[4]), "r"(qw[5]), "r"(qw[6]), "r"(qw[7]) : "memory");
        }
    }
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int dk0[32];
    unsigned int dk1[32];
    unsigned int dv0[32];
    unsigned int dv1[32];
    unsigned int ak0[32];
    unsigned int ak1[32];
    unsigned int av0[32];
    unsigned int av1[32];
    float d_s0[4];
    float d_s1[4];
    float d_s0a[4];
    float d_s0b[4];
    float d_s1a[4];
    float d_s1b[4];
    float d_o0[4];
    float d_o1[4];
    float d_o0x[4];
    float d_o1x[4];
    d_o0[0] = 0.0f;
    d_o1[0] = 0.0f;
    d_o0x[0] = 0.0f;
    d_o1x[0] = 0.0f;
    d_o0[1] = 0.0f;
    d_o1[1] = 0.0f;
    d_o0x[1] = 0.0f;
    d_o1x[1] = 0.0f;
    d_o0[2] = 0.0f;
    d_o1[2] = 0.0f;
    d_o0x[2] = 0.0f;
    d_o1x[2] = 0.0f;
    d_o0[3] = 0.0f;
    d_o1[3] = 0.0f;
    d_o0x[3] = 0.0f;
    d_o1x[3] = 0.0f;
    float mrun[2];
    float lp[2];
    mrun[0] = -CAKE_INF;
    lp[0] = 0.0f;
    mrun[1] = -CAKE_INF;
    lp[1] = 0.0f;
    float sv0[4];
    float sv1[4];
    float tmax[2];
    float rv[4];
    float msub[2];
    int first_page = 1;
    uint64_t _wgmma_desc_0 = (((uint64_t)(((ident8_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_1 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_0 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_0);
    uint64_t _wgmma_desc_1 = (((uint64_t)(((q16_addr)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_3 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_1 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_1);
    uint64_t _wgmma_desc_2 = (((uint64_t)(((q16_addr + 1024)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
    uint64_t _wgmma_b_0_4 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_2 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_2);
    #pragma unroll 1
    for (int j = 0; j < 4; j++) {
        int stage = j % 2;
        int phase = j / 2 % 2;
        int par = j - j / 2 * 2;
        mbarrier_wait(kv_full_addr + (stage) * 8, phase);
        if (j == 0) {
            int slot_1 = split * 4 + 1;
            int blk_2 = meta[slot_1];
            int page_head_1 = meta[16 + slot_1];
            int valid_1 = 0;
            if (blk_2 >= 0) {
                valid_1 = q_pos + 1 - blk_2 * 128;
                if (valid_1 > 128) {
                    valid_1 = 128;
                }
                if (valid_1 < 0) {
                    valid_1 = 0;
                }
            }
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
        int slot_2 = split * 4 + j;
        int blk_3 = meta[slot_2];
        int page_head_2 = meta[16 + slot_2];
        int valid_2 = 0;
        if (blk_3 >= 0) {
            valid_2 = q_pos + 1 - blk_3 * 128;
            if (valid_2 > 128) {
                valid_2 = 128;
            }
            if (valid_2 < 0) {
                valid_2 = 0;
            }
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_3 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_a_0_0 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_3 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_3);
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
        uint64_t _wgmma_desc_4 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 8192)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_a_0_2 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_4 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_4);
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
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        ak0[0] = dk0[0] + zero_u32;
        ak1[0] = dk1[0] + zero_u32;
        ak0[1] = dk0[1] + zero_u32;
        ak1[1] = dk1[1] + zero_u32;
        ak0[2] = dk0[2] + zero_u32;
        ak1[2] = dk1[2] + zero_u32;
        ak0[3] = dk0[3] + zero_u32;
        ak1[3] = dk1[3] + zero_u32;
        ak0[4] = dk0[4] + zero_u32;
        ak1[4] = dk1[4] + zero_u32;
        ak0[5] = dk0[5] + zero_u32;
        ak1[5] = dk1[5] + zero_u32;
        ak0[6] = dk0[6] + zero_u32;
        ak1[6] = dk1[6] + zero_u32;
        ak0[7] = dk0[7] + zero_u32;
        ak1[7] = dk1[7] + zero_u32;
        ak0[8] = dk0[8] + zero_u32;
        ak1[8] = dk1[8] + zero_u32;
        ak0[9] = dk0[9] + zero_u32;
        ak1[9] = dk1[9] + zero_u32;
        ak0[10] = dk0[10] + zero_u32;
        ak1[10] = dk1[10] + zero_u32;
        ak0[11] = dk0[11] + zero_u32;
        ak1[11] = dk1[11] + zero_u32;
        ak0[12] = dk0[12] + zero_u32;
        ak1[12] = dk1[12] + zero_u32;
        ak0[13] = dk0[13] + zero_u32;
        ak1[13] = dk1[13] + zero_u32;
        ak0[14] = dk0[14] + zero_u32;
        ak1[14] = dk1[14] + zero_u32;
        ak0[15] = dk0[15] + zero_u32;
        ak1[15] = dk1[15] + zero_u32;
        ak0[16] = dk0[16] + zero_u32;
        ak1[16] = dk1[16] + zero_u32;
        ak0[17] = dk0[17] + zero_u32;
        ak1[17] = dk1[17] + zero_u32;
        ak0[18] = dk0[18] + zero_u32;
        ak1[18] = dk1[18] + zero_u32;
        ak0[19] = dk0[19] + zero_u32;
        ak1[19] = dk1[19] + zero_u32;
        ak0[20] = dk0[20] + zero_u32;
        ak1[20] = dk1[20] + zero_u32;
        ak0[21] = dk0[21] + zero_u32;
        ak1[21] = dk1[21] + zero_u32;
        ak0[22] = dk0[22] + zero_u32;
        ak1[22] = dk1[22] + zero_u32;
        ak0[23] = dk0[23] + zero_u32;
        ak1[23] = dk1[23] + zero_u32;
        ak0[24] = dk0[24] + zero_u32;
        ak1[24] = dk1[24] + zero_u32;
        ak0[25] = dk0[25] + zero_u32;
        ak1[25] = dk1[25] + zero_u32;
        ak0[26] = dk0[26] + zero_u32;
        ak1[26] = dk1[26] + zero_u32;
        ak0[27] = dk0[27] + zero_u32;
        ak1[27] = dk1[27] + zero_u32;
        ak0[28] = dk0[28] + zero_u32;
        ak1[28] = dk1[28] + zero_u32;
        ak0[29] = dk0[29] + zero_u32;
        ak1[29] = dk1[29] + zero_u32;
        ak0[30] = dk0[30] + zero_u32;
        ak1[30] = dk1[30] + zero_u32;
        ak0[31] = dk0[31] + zero_u32;
        ak1[31] = dk1[31] + zero_u32;
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
            : "r"(ak0[0]), "r"(ak0[1]), "r"(ak0[2]), "r"(ak0[3]), "l"(_wgmma_b_0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
            : "r"(ak1[0]), "r"(ak1[1]), "r"(ak1[2]), "r"(ak1[3]), "l"(_wgmma_b_0_3)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
            : "r"(ak0[16]), "r"(ak0[(16) + 1]), "r"(ak0[(16) + 2]), "r"(ak0[(16) + 3]), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 0, 1, 1, 0;\n}\n"
            : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
            : "r"(ak1[16]), "r"(ak1[(16) + 1]), "r"(ak1[(16) + 2]), "r"(ak1[(16) + 3]), "l"(_wgmma_b_0_4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
            : "r"(ak0[4]), "r"(ak0[(4) + 1]), "r"(ak0[(4) + 2]), "r"(ak0[(4) + 3]), "l"(_wgmma_b_0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
            : "r"(ak1[4]), "r"(ak1[(4) + 1]), "r"(ak1[(4) + 2]), "r"(ak1[(4) + 3]), "l"(_wgmma_b_0_3 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
            : "r"(ak0[20]), "r"(ak0[(20) + 1]), "r"(ak0[(20) + 2]), "r"(ak0[(20) + 3]), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
            : "r"(ak1[20]), "r"(ak1[(20) + 1]), "r"(ak1[(20) + 2]), "r"(ak1[(20) + 3]), "l"(_wgmma_b_0_4 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
            : "r"(ak0[8]), "r"(ak0[(8) + 1]), "r"(ak0[(8) + 2]), "r"(ak0[(8) + 3]), "l"(_wgmma_b_0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
            : "r"(ak1[8]), "r"(ak1[(8) + 1]), "r"(ak1[(8) + 2]), "r"(ak1[(8) + 3]), "l"(_wgmma_b_0_3 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
            : "r"(ak0[24]), "r"(ak0[(24) + 1]), "r"(ak0[(24) + 2]), "r"(ak0[(24) + 3]), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
            : "r"(ak1[24]), "r"(ak1[(24) + 1]), "r"(ak1[(24) + 2]), "r"(ak1[(24) + 3]), "l"(_wgmma_b_0_4 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0a[0]), "+f"(d_s0a[1]), "+f"(d_s0a[2]), "+f"(d_s0a[3])
            : "r"(ak0[12]), "r"(ak0[(12) + 1]), "r"(ak0[(12) + 2]), "r"(ak0[(12) + 3]), "l"(_wgmma_b_0_3 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1a[0]), "+f"(d_s1a[1]), "+f"(d_s1a[2]), "+f"(d_s1a[3])
            : "r"(ak1[12]), "r"(ak1[(12) + 1]), "r"(ak1[(12) + 2]), "r"(ak1[(12) + 3]), "l"(_wgmma_b_0_3 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s0b[0]), "+f"(d_s0b[1]), "+f"(d_s0b[2]), "+f"(d_s0b[3])
            : "r"(ak0[28]), "r"(ak0[(28) + 1]), "r"(ak0[(28) + 2]), "r"(ak0[(28) + 3]), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_s1b[0]), "+f"(d_s1b[1]), "+f"(d_s1b[2]), "+f"(d_s1b[3])
            : "r"(ak1[28]), "r"(ak1[(28) + 1]), "r"(ak1[(28) + 2]), "r"(ak1[(28) + 3]), "l"(_wgmma_b_0_4 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
        d_s0[0] = d_s0a[0] + d_s0b[0];
        d_s1[0] = d_s1a[0] + d_s1b[0];
        d_s0[1] = d_s0a[1] + d_s0b[1];
        d_s1[1] = d_s1a[1] + d_s1b[1];
        d_s0[2] = d_s0a[2] + d_s0b[2];
        d_s1[2] = d_s1a[2] + d_s1b[2];
        d_s0[3] = d_s0a[3] + d_s0b[3];
        d_s1[3] = d_s1a[3] + d_s1b[3];
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_5 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 16384)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_5 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_5 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_5);
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
        uint64_t _wgmma_desc_6 = (((uint64_t)(((kv8_addr + (unsigned int)(stage * 32768) + 24576)) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_6 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_6 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_6);
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
        tmax[0] = -CAKE_INF;
        tmax[1] = -CAKE_INF;
        sv0[0] = -CAKE_INF;
        if (key0 < valid_2) {
            sv0[0] = d_s0[0] * softmax_scale_log2;
        }
        sv1[0] = -CAKE_INF;
        if (valid_2 > 64 + key0) {
            sv1[0] = d_s1[0] * softmax_scale_log2;
        }
        float _max_0 = max_noftz(sv0[0], sv1[0]);
        float _max_1 = max_noftz(tmax[0], _max_0);
        tmax[0] = _max_1;
        sv0[1] = -CAKE_INF;
        if (key0 < valid_2) {
            sv0[1] = d_s0[1] * softmax_scale_log2;
        }
        sv1[1] = -CAKE_INF;
        if (valid_2 > 64 + key0) {
            sv1[1] = d_s1[1] * softmax_scale_log2;
        }
        float _max_2 = max_noftz(sv0[1], sv1[1]);
        float _max_3 = max_noftz(tmax[1], _max_2);
        tmax[1] = _max_3;
        sv0[2] = -CAKE_INF;
        if (key1 < valid_2) {
            sv0[2] = d_s0[2] * softmax_scale_log2;
        }
        sv1[2] = -CAKE_INF;
        if (valid_2 > 64 + key1) {
            sv1[2] = d_s1[2] * softmax_scale_log2;
        }
        float _max_4 = max_noftz(sv0[2], sv1[2]);
        float _max_5 = max_noftz(tmax[0], _max_4);
        tmax[0] = _max_5;
        sv0[3] = -CAKE_INF;
        if (key1 < valid_2) {
            sv0[3] = d_s0[3] * softmax_scale_log2;
        }
        sv1[3] = -CAKE_INF;
        if (valid_2 > 64 + key1) {
            sv1[3] = d_s1[3] * softmax_scale_log2;
        }
        float _max_6 = max_noftz(sv0[3], sv1[3]);
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
        float t_0 = tmax[1];
        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, t_0, 4);
        float _max_11 = max_noftz(t_0, _shfl_xor_3);
        t_0 = _max_11;
        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, t_0, 8);
        float _max_12 = max_noftz(t_0, _shfl_xor_4);
        t_0 = _max_12;
        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, t_0, 16);
        float _max_13 = max_noftz(t_0, _shfl_xor_5);
        t_0 = _max_13;
        tmax[1] = t_0;
        if (lane_0 < 4) {
            int hh = 2 * cq;
            red[hh * 4 + warp_1] = tmax[0];
            int hh_0 = 2 * cq + 1;
            red[hh_0 * 4 + warp_1] = tmax[1];
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        int hh2 = 2 * cq;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
            : "r"(red_addr + (unsigned int)(hh2 * 4 * 4)));
        float _max_14 = max_noftz(rv[0], rv[1]);
        float _max_15 = max_noftz(rv[2], rv[3]);
        float _max_16 = max_noftz(_max_14, _max_15);
        float page_max = _max_16;
        float _max_17 = max_noftz(mrun[0], page_max);
        float mnew = _max_17;
        float corr = 0.0f;
        if (mrun[0] != -CAKE_INF) {
            float _exp2_0 = approx_exp2(mrun[0] - mnew);
            corr = _exp2_0;
        }
        d_o0[0] = d_o0[0] * corr;
        d_o0[2] = d_o0[2] * corr;
        d_o1[0] = d_o1[0] * corr;
        d_o1[2] = d_o1[2] * corr;
        d_o0x[0] = d_o0x[0] * corr;
        d_o0x[2] = d_o0x[2] * corr;
        d_o1x[0] = d_o1x[0] * corr;
        d_o1x[2] = d_o1x[2] * corr;
        lp[0] = lp[0] * corr;
        mrun[0] = mnew;
        msub[0] = mnew;
        if (mnew == -CAKE_INF) {
            msub[0] = 0.0f;
        }
        int hh2_1 = 2 * cq + 1;
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
            : "r"(red_addr + (unsigned int)(hh2_1 * 4 * 4)));
        float _max_18 = max_noftz(rv[0], rv[1]);
        float _max_19 = max_noftz(rv[2], rv[3]);
        float _max_20 = max_noftz(_max_18, _max_19);
        float page_max_2 = _max_20;
        float _max_21 = max_noftz(mrun[1], page_max_2);
        float mnew_3 = _max_21;
        float corr_4 = 0.0f;
        if (mrun[1] != -CAKE_INF) {
            float _exp2_1 = approx_exp2(mrun[1] - mnew_3);
            corr_4 = _exp2_1;
        }
        d_o0[1] = d_o0[1] * corr_4;
        d_o0[3] = d_o0[3] * corr_4;
        d_o1[1] = d_o1[1] * corr_4;
        d_o1[3] = d_o1[3] * corr_4;
        d_o0x[1] = d_o0x[1] * corr_4;
        d_o0x[3] = d_o0x[3] * corr_4;
        d_o1x[1] = d_o1x[1] * corr_4;
        d_o1x[3] = d_o1x[3] * corr_4;
        lp[1] = lp[1] * corr_4;
        mrun[1] = mnew_3;
        msub[1] = mnew_3;
        if (mnew_3 == -CAKE_INF) {
            msub[1] = 0.0f;
        }
        int hh3 = 2 * cq;
        int kk = key0;
        int pcol = hh3 * 128 + (kk * 2 ^ (hh3 & 7) << 4);
        float p0 = 1.0f;
        float p1 = 1.0f;
        float _exp2_2 = approx_exp2(sv0[0] - msub[0]);
        p0 = _exp2_2;
        float _exp2_3 = approx_exp2(sv1[0] - msub[0]);
        p1 = _exp2_3;
        lp[0] = lp[0] + (p0 + p1);
        {
            __half _hval_7 = __float2half_rn(p0);
            uint16_t _bits_7 = *(uint16_t*)&_hval_7;
            uint32_t _addr_7 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_7), "h"(_bits_7) : "memory");
        }
        {
            __half _hval_8 = __float2half_rn(p1);
            uint16_t _bits_8 = *(uint16_t*)&_hval_8;
            uint32_t _addr_8 = static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_8), "h"(_bits_8) : "memory");
        }
        int hh3_5 = 2 * cq + 1;
        int kk_6 = key0;
        int pcol_7 = hh3_5 * 128 + (kk_6 * 2 ^ (hh3_5 & 7) << 4);
        float p0_8 = 1.0f;
        float p1_9 = 1.0f;
        float _exp2_4 = approx_exp2(sv0[1] - msub[1]);
        p0_8 = _exp2_4;
        float _exp2_5 = approx_exp2(sv1[1] - msub[1]);
        p1_9 = _exp2_5;
        lp[1] = lp[1] + (p0_8 + p1_9);
        {
            __half _hval_9 = __float2half_rn(p0_8);
            uint16_t _bits_9 = *(uint16_t*)&_hval_9;
            uint32_t _addr_9 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_7));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_9), "h"(_bits_9) : "memory");
        }
        {
            __half _hval_10 = __float2half_rn(p1_9);
            uint16_t _bits_10 = *(uint16_t*)&_hval_10;
            uint32_t _addr_10 = static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_7));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_10), "h"(_bits_10) : "memory");
        }
        int hh3_10 = 2 * cq;
        int kk_11 = key1;
        int pcol_12 = hh3_10 * 128 + (kk_11 * 2 ^ (hh3_10 & 7) << 4);
        float p0_13 = 1.0f;
        float p1_14 = 1.0f;
        float _exp2_6 = approx_exp2(sv0[2] - msub[0]);
        p0_13 = _exp2_6;
        float _exp2_7 = approx_exp2(sv1[2] - msub[0]);
        p1_14 = _exp2_7;
        lp[0] = lp[0] + (p0_13 + p1_14);
        {
            __half _hval_11 = __float2half_rn(p0_13);
            uint16_t _bits_11 = *(uint16_t*)&_hval_11;
            uint32_t _addr_11 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_12));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_11), "h"(_bits_11) : "memory");
        }
        {
            __half _hval_12 = __float2half_rn(p1_14);
            uint16_t _bits_12 = *(uint16_t*)&_hval_12;
            uint32_t _addr_12 = static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_12));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_12), "h"(_bits_12) : "memory");
        }
        int hh3_15 = 2 * cq + 1;
        int kk_16 = key1;
        int pcol_17 = hh3_15 * 128 + (kk_16 * 2 ^ (hh3_15 & 7) << 4);
        float p0_18 = 1.0f;
        float p1_19 = 1.0f;
        float _exp2_8 = approx_exp2(sv0[3] - msub[1]);
        p0_18 = _exp2_8;
        float _exp2_9 = approx_exp2(sv1[3] - msub[1]);
        p1_19 = _exp2_9;
        lp[1] = lp[1] + (p0_18 + p1_19);
        {
            __half _hval_13 = __float2half_rn(p0_18);
            uint16_t _bits_13 = *(uint16_t*)&_hval_13;
            uint32_t _addr_13 = static_cast<uint32_t>(p16_addr + (unsigned int)(par * 2 * 1024 + pcol_17));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_13), "h"(_bits_13) : "memory");
        }
        {
            __half _hval_14 = __float2half_rn(p1_19);
            uint16_t _bits_14 = *(uint16_t*)&_hval_14;
            uint32_t _addr_14 = static_cast<uint32_t>(p16_addr + (unsigned int)((par * 2 + 1) * 1024 + pcol_17));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_14), "h"(_bits_14) : "memory");
        }
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
        if (nxt < 4) {
            int slot_0 = split * 4 + nxt;
            int blk_1_1 = meta[slot_0];
            int page_head_2_1 = meta[16 + slot_0];
            int valid_3 = 0;
            if (blk_1_1 >= 0) {
                valid_3 = q_pos + 1 - blk_1_1 * 128;
                if (valid_3 > 128) {
                    valid_3 = 128;
                }
                if (valid_3 < 0) {
                    valid_3 = 0;
                }
            }
            if (warp == 1) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 32768);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768), (&K), 0, 0, page_head_2_1, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 8192, (&K), 0, 64, page_head_2_1, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 16384, (&V), 0, 0, page_head_2_1, kv_full_addr + (stage) * 8);
                    tma_3d_gmem2smem(kv8_addr + (unsigned int)(stage * 32768) + 24576, (&V), 0, 64, page_head_2_1, kv_full_addr + (stage) * 8);
                }
            }
        }
        asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
        uint64_t _wgmma_desc_15 = (((uint64_t)(((p16_addr + (unsigned int)(par * 2 * 1024))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_7 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_15 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_15);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
            : "r"(av0[0]), "r"(av0[1]), "r"(av0[2]), "r"(av0[3]), "l"(_wgmma_b_0_7)
            : "memory");
        uint64_t _wgmma_desc_16 = (((uint64_t)(((p16_addr + (unsigned int)((par * 2 + 1) * 1024))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
        uint64_t _wgmma_b_0_8 = ((uint64_t)make_warp_uniform((uint32_t)(_wgmma_desc_16 >> 32)) << 32) | (uint64_t)make_warp_uniform((uint32_t)_wgmma_desc_16);
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
            : "r"(av1[0]), "r"(av1[1]), "r"(av1[2]), "r"(av1[3]), "l"(_wgmma_b_0_8)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
            : "r"(av0[16]), "r"(av0[(16) + 1]), "r"(av0[(16) + 2]), "r"(av0[(16) + 3]), "l"(_wgmma_b_0_7)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
            : "r"(av1[16]), "r"(av1[(16) + 1]), "r"(av1[(16) + 2]), "r"(av1[(16) + 3]), "l"(_wgmma_b_0_8)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
            : "r"(av0[4]), "r"(av0[(4) + 1]), "r"(av0[(4) + 2]), "r"(av0[(4) + 3]), "l"(_wgmma_b_0_7 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
            : "r"(av1[4]), "r"(av1[(4) + 1]), "r"(av1[(4) + 2]), "r"(av1[(4) + 3]), "l"(_wgmma_b_0_8 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
            : "r"(av0[20]), "r"(av0[(20) + 1]), "r"(av0[(20) + 2]), "r"(av0[(20) + 3]), "l"(_wgmma_b_0_7 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
            : "r"(av1[20]), "r"(av1[(20) + 1]), "r"(av1[(20) + 2]), "r"(av1[(20) + 3]), "l"(_wgmma_b_0_8 + 2)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
            : "r"(av0[8]), "r"(av0[(8) + 1]), "r"(av0[(8) + 2]), "r"(av0[(8) + 3]), "l"(_wgmma_b_0_7 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
            : "r"(av1[8]), "r"(av1[(8) + 1]), "r"(av1[(8) + 2]), "r"(av1[(8) + 3]), "l"(_wgmma_b_0_8 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
            : "r"(av0[24]), "r"(av0[(24) + 1]), "r"(av0[(24) + 2]), "r"(av0[(24) + 3]), "l"(_wgmma_b_0_7 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
            : "r"(av1[24]), "r"(av1[(24) + 1]), "r"(av1[(24) + 2]), "r"(av1[(24) + 3]), "l"(_wgmma_b_0_8 + 4)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0[0]), "+f"(d_o0[1]), "+f"(d_o0[2]), "+f"(d_o0[3])
            : "r"(av0[12]), "r"(av0[(12) + 1]), "r"(av0[(12) + 2]), "r"(av0[(12) + 3]), "l"(_wgmma_b_0_7 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o0x[0]), "+f"(d_o0x[1]), "+f"(d_o0x[2]), "+f"(d_o0x[3])
            : "r"(av1[12]), "r"(av1[(12) + 1]), "r"(av1[(12) + 2]), "r"(av1[(12) + 3]), "l"(_wgmma_b_0_8 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1[0]), "+f"(d_o1[1]), "+f"(d_o1[2]), "+f"(d_o1[3])
            : "r"(av0[28]), "r"(av0[(28) + 1]), "r"(av0[(28) + 2]), "r"(av0[(28) + 3]), "l"(_wgmma_b_0_7 + 6)
            : "memory");
        asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, %3}, {%4, %5, %6, %7}, %8, 1, 1, 1, 0;\n}\n"
            : "+f"(d_o1x[0]), "+f"(d_o1x[1]), "+f"(d_o1x[2]), "+f"(d_o1x[3])
            : "r"(av1[28]), "r"(av1[(28) + 1]), "r"(av1[(28) + 2]), "r"(av1[(28) + 3]), "l"(_wgmma_b_0_8 + 6)
            : "memory");
        asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        first_page = 0;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    d_o0[0] = d_o0[0] + d_o0x[0];
    d_o1[0] = d_o1[0] + d_o1x[0];
    d_o0[1] = d_o0[1] + d_o0x[1];
    d_o1[1] = d_o1[1] + d_o1x[1];
    d_o0[2] = d_o0[2] + d_o0x[2];
    d_o1[2] = d_o1[2] + d_o1x[2];
    d_o0[3] = d_o0[3] + d_o0x[3];
    d_o1[3] = d_o1[3] + d_o1x[3];
    float t2 = lp[0];
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, t2, 4);
    t2 = t2 + _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, t2, 8);
    t2 = t2 + _shfl_xor_7;
    float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, t2, 16);
    t2 = t2 + _shfl_xor_8;
    lp[0] = t2;
    float t2_2 = lp[1];
    float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 4);
    t2_2 = t2_2 + _shfl_xor_9;
    float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 8);
    t2_2 = t2_2 + _shfl_xor_10;
    float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, t2_2, 16);
    t2_2 = t2_2 + _shfl_xor_11;
    lp[1] = t2_2;
    if (lane_0 < 4) {
        int hh4 = 2 * cq;
        lred[hh4 * 4 + warp_1] = lp[0];
        int hh4_0 = 2 * cq + 1;
        lred[hh4_0 * 4 + warp_1] = lp[1];
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    float lsum[2];
    int hh5 = 2 * cq;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5 * 4 * 4)));
    lsum[0] = rv[0] + rv[1] + (rv[2] + rv[3]);
    int hh5_3 = 2 * cq + 1;
    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
        : "=r"(*reinterpret_cast<uint32_t*>(&rv[0])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rv[(0) + 3]))
        : "r"(lred_addr + (unsigned int)(hh5_3 * 4 * 4)));
    lsum[1] = rv[0] + rv[1] + (rv[2] + rv[3]);
    int merge = 1;
    int pbase = item * 4 + split;
    int orow = (pbase * 128 + tid_1) * 4;
    float invl[2];
    invl[0] = 0.0f;
    if (lsum[0] != 0.0f) {
        float _rcp_0 = approx_rcp(lsum[0]);
        invl[0] = _rcp_0;
    }
    invl[1] = 0.0f;
    if (lsum[1] != 0.0f) {
        float _rcp_1 = approx_rcp(lsum[1]);
        invl[1] = _rcp_1;
    }
    unsigned int pk[4];
    uint32_t _f16x2_pack_0;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_0) : "f"(d_o0[1] * invl[1]), "f"(d_o0[0] * invl[0]));
    pk[0] = _f16x2_pack_0;
    uint32_t _f16x2_pack_1;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_1) : "f"(d_o0[3] * invl[1]), "f"(d_o0[2] * invl[0]));
    pk[1] = _f16x2_pack_1;
    uint32_t _f16x2_pack_2;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_2) : "f"(d_o1[1] * invl[1]), "f"(d_o1[0] * invl[0]));
    pk[2] = _f16x2_pack_2;
    uint32_t _f16x2_pack_3;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(_f16x2_pack_3) : "f"(d_o1[3] * invl[1]), "f"(d_o1[2] * invl[0]));
    pk[3] = _f16x2_pack_3;
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};" :: "l"(part_o + orow + (0)), "r"(pk[(0) + 0]), "r"(pk[(0) + 1]), "r"(pk[(0) + 2]), "r"(pk[(0) + 3]) : "memory");
    if (warp_1 == 0) {
        if (lane_0 < 4) {
            float mlv[4];
            mlv[0] = mrun[0];
            mlv[1] = lsum[0];
            mlv[2] = mrun[1];
            mlv[3] = lsum[1];
            {
                float4 _v4 = make_float4(mlv[0 + 0], mlv[0 + 1], mlv[0 + 2], mlv[0 + 3]);
                *reinterpret_cast<float4*>((part_ml + ((pbase * 4 + cq) * 4)) + 0) = _v4;
            }
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    if (warp == 0) {
        if (elect_sync()) {
            unsigned int _atomic_old_0;
            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                : "=r"(_atomic_old_0) : "l"(&counters[item]), "r"(static_cast<uint32_t>(1)) : "memory");
            unsigned int old = _atomic_old_0;
            flag[0] = old;
        }
    }
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int oldv = flag[0];
    merge = 0;
    if ((oldv & 3) == 3) {
        merge = 1;
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(mfull_addr, 8448);
                cp_async_bulk_gmem2smem(kv8_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(part_o) + ((unsigned long long)(item * 2048) * (unsigned long long)4)), 8192, mfull_addr);
                cp_async_bulk_gmem2smem(p16_addr, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(part_ml) + ((unsigned long long)(item * 64) * (unsigned long long)4)), 256, mfull_addr);
            }
        }
        mbarrier_wait(mfull_addr, 0);
        d_o0[0] = 0.0f;
        d_o1[0] = 0.0f;
        d_o0[1] = 0.0f;
        d_o1[1] = 0.0f;
        d_o0[2] = 0.0f;
        d_o1[2] = 0.0f;
        d_o0[3] = 0.0f;
        d_o1[3] = 0.0f;
        lsum[0] = 0.0f;
        lsum[1] = 0.0f;
        float mlall[16];
        unsigned int pall[16];
        float cw[8];
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mlall[0])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(0) + 3]))
            : "r"(p16_addr + (unsigned int)(cq * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mlall[4])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(4) + 3]))
            : "r"(p16_addr + (unsigned int)((4 + cq) * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mlall[8])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(8) + 3]))
            : "r"(p16_addr + (unsigned int)((8 + cq) * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&mlall[12])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&mlall[(12) + 3]))
            : "r"(p16_addr + (unsigned int)((12 + cq) * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&pall[0])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(0) + 3]))
            : "r"(kv8_addr + (unsigned int)(tid_1 * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&pall[4])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(4) + 3]))
            : "r"(kv8_addr + (unsigned int)((128 + tid_1) * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&pall[8])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(8) + 3]))
            : "r"(kv8_addr + (unsigned int)((256 + tid_1) * 4 * 4)));
        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(*reinterpret_cast<uint32_t*>(&pall[12])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pall[(12) + 3]))
            : "r"(kv8_addr + (unsigned int)((384 + tid_1) * 4 * 4)));
        float m0 = -CAKE_INF;
        float m1 = -CAKE_INF;
        float _max_22 = max_noftz(m0, mlall[0]);
        m0 = _max_22;
        float _max_23 = max_noftz(m1, mlall[2]);
        m1 = _max_23;
        float _max_24 = max_noftz(m0, mlall[4]);
        m0 = _max_24;
        float _max_25 = max_noftz(m1, mlall[6]);
        m1 = _max_25;
        float _max_26 = max_noftz(m0, mlall[8]);
        m0 = _max_26;
        float _max_27 = max_noftz(m1, mlall[10]);
        m1 = _max_27;
        float _max_28 = max_noftz(m0, mlall[12]);
        m0 = _max_28;
        float _max_29 = max_noftz(m1, mlall[14]);
        m1 = _max_29;
        cw[0] = 0.0f;
        if (mlall[0] != -CAKE_INF) {
            float _exp2_10 = approx_exp2(mlall[0] - m0);
            cw[0] = _exp2_10 * mlall[1];
        }
        cw[1] = 0.0f;
        if (mlall[2] != -CAKE_INF) {
            float _exp2_11 = approx_exp2(mlall[2] - m1);
            cw[1] = _exp2_11 * mlall[3];
        }
        lsum[0] = lsum[0] + cw[0];
        lsum[1] = lsum[1] + cw[1];
        cw[2] = 0.0f;
        if (mlall[4] != -CAKE_INF) {
            float _exp2_12 = approx_exp2(mlall[4] - m0);
            cw[2] = _exp2_12 * mlall[5];
        }
        cw[3] = 0.0f;
        if (mlall[6] != -CAKE_INF) {
            float _exp2_13 = approx_exp2(mlall[6] - m1);
            cw[3] = _exp2_13 * mlall[7];
        }
        lsum[0] = lsum[0] + cw[2];
        lsum[1] = lsum[1] + cw[3];
        cw[4] = 0.0f;
        if (mlall[8] != -CAKE_INF) {
            float _exp2_14 = approx_exp2(mlall[8] - m0);
            cw[4] = _exp2_14 * mlall[9];
        }
        cw[5] = 0.0f;
        if (mlall[10] != -CAKE_INF) {
            float _exp2_15 = approx_exp2(mlall[10] - m1);
            cw[5] = _exp2_15 * mlall[11];
        }
        lsum[0] = lsum[0] + cw[4];
        lsum[1] = lsum[1] + cw[5];
        cw[6] = 0.0f;
        if (mlall[12] != -CAKE_INF) {
            float _exp2_16 = approx_exp2(mlall[12] - m0);
            cw[6] = _exp2_16 * mlall[13];
        }
        cw[7] = 0.0f;
        if (mlall[14] != -CAKE_INF) {
            float _exp2_17 = approx_exp2(mlall[14] - m1);
            cw[7] = _exp2_17 * mlall[15];
        }
        lsum[0] = lsum[0] + cw[6];
        lsum[1] = lsum[1] + cw[7];
        float _cvt_f32_f16_0;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(pall[0])));
        d_o0[0] = d_o0[0] + cw[0] * _cvt_f32_f16_0;
        float _cvt_f32_f16_1;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(pall[0] >> 16)));
        d_o0[1] = d_o0[1] + cw[1] * _cvt_f32_f16_1;
        float _cvt_f32_f16_2;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(pall[1])));
        d_o0[2] = d_o0[2] + cw[0] * _cvt_f32_f16_2;
        float _cvt_f32_f16_3;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(pall[1] >> 16)));
        d_o0[3] = d_o0[3] + cw[1] * _cvt_f32_f16_3;
        float _cvt_f32_f16_4;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(pall[2])));
        d_o1[0] = d_o1[0] + cw[0] * _cvt_f32_f16_4;
        float _cvt_f32_f16_5;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(pall[2] >> 16)));
        d_o1[1] = d_o1[1] + cw[1] * _cvt_f32_f16_5;
        float _cvt_f32_f16_6;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(pall[3])));
        d_o1[2] = d_o1[2] + cw[0] * _cvt_f32_f16_6;
        float _cvt_f32_f16_7;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(pall[3] >> 16)));
        d_o1[3] = d_o1[3] + cw[1] * _cvt_f32_f16_7;
        float _cvt_f32_f16_8;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_8) : "h"((uint16_t)(pall[4])));
        d_o0[0] = d_o0[0] + cw[2] * _cvt_f32_f16_8;
        float _cvt_f32_f16_9;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_9) : "h"((uint16_t)(pall[4] >> 16)));
        d_o0[1] = d_o0[1] + cw[3] * _cvt_f32_f16_9;
        float _cvt_f32_f16_10;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_10) : "h"((uint16_t)(pall[5])));
        d_o0[2] = d_o0[2] + cw[2] * _cvt_f32_f16_10;
        float _cvt_f32_f16_11;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_11) : "h"((uint16_t)(pall[5] >> 16)));
        d_o0[3] = d_o0[3] + cw[3] * _cvt_f32_f16_11;
        float _cvt_f32_f16_12;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_12) : "h"((uint16_t)(pall[6])));
        d_o1[0] = d_o1[0] + cw[2] * _cvt_f32_f16_12;
        float _cvt_f32_f16_13;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_13) : "h"((uint16_t)(pall[6] >> 16)));
        d_o1[1] = d_o1[1] + cw[3] * _cvt_f32_f16_13;
        float _cvt_f32_f16_14;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_14) : "h"((uint16_t)(pall[7])));
        d_o1[2] = d_o1[2] + cw[2] * _cvt_f32_f16_14;
        float _cvt_f32_f16_15;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_15) : "h"((uint16_t)(pall[7] >> 16)));
        d_o1[3] = d_o1[3] + cw[3] * _cvt_f32_f16_15;
        float _cvt_f32_f16_16;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_16) : "h"((uint16_t)(pall[8])));
        d_o0[0] = d_o0[0] + cw[4] * _cvt_f32_f16_16;
        float _cvt_f32_f16_17;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_17) : "h"((uint16_t)(pall[8] >> 16)));
        d_o0[1] = d_o0[1] + cw[5] * _cvt_f32_f16_17;
        float _cvt_f32_f16_18;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_18) : "h"((uint16_t)(pall[9])));
        d_o0[2] = d_o0[2] + cw[4] * _cvt_f32_f16_18;
        float _cvt_f32_f16_19;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_19) : "h"((uint16_t)(pall[9] >> 16)));
        d_o0[3] = d_o0[3] + cw[5] * _cvt_f32_f16_19;
        float _cvt_f32_f16_20;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_20) : "h"((uint16_t)(pall[10])));
        d_o1[0] = d_o1[0] + cw[4] * _cvt_f32_f16_20;
        float _cvt_f32_f16_21;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_21) : "h"((uint16_t)(pall[10] >> 16)));
        d_o1[1] = d_o1[1] + cw[5] * _cvt_f32_f16_21;
        float _cvt_f32_f16_22;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_22) : "h"((uint16_t)(pall[11])));
        d_o1[2] = d_o1[2] + cw[4] * _cvt_f32_f16_22;
        float _cvt_f32_f16_23;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_23) : "h"((uint16_t)(pall[11] >> 16)));
        d_o1[3] = d_o1[3] + cw[5] * _cvt_f32_f16_23;
        float _cvt_f32_f16_24;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_24) : "h"((uint16_t)(pall[12])));
        d_o0[0] = d_o0[0] + cw[6] * _cvt_f32_f16_24;
        float _cvt_f32_f16_25;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_25) : "h"((uint16_t)(pall[12] >> 16)));
        d_o0[1] = d_o0[1] + cw[7] * _cvt_f32_f16_25;
        float _cvt_f32_f16_26;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_26) : "h"((uint16_t)(pall[13])));
        d_o0[2] = d_o0[2] + cw[6] * _cvt_f32_f16_26;
        float _cvt_f32_f16_27;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_27) : "h"((uint16_t)(pall[13] >> 16)));
        d_o0[3] = d_o0[3] + cw[7] * _cvt_f32_f16_27;
        float _cvt_f32_f16_28;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_28) : "h"((uint16_t)(pall[14])));
        d_o1[0] = d_o1[0] + cw[6] * _cvt_f32_f16_28;
        float _cvt_f32_f16_29;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_29) : "h"((uint16_t)(pall[14] >> 16)));
        d_o1[1] = d_o1[1] + cw[7] * _cvt_f32_f16_29;
        float _cvt_f32_f16_30;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_30) : "h"((uint16_t)(pall[15])));
        d_o1[2] = d_o1[2] + cw[6] * _cvt_f32_f16_30;
        float _cvt_f32_f16_31;
        asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_31) : "h"((uint16_t)(pall[15] >> 16)));
        d_o1[3] = d_o1[3] + cw[7] * _cvt_f32_f16_31;
    }
    if (merge != 0) {
        float inv = 0.0f;
        if (lsum[0] != 0.0f) {
            float _rcp_2 = approx_rcp(lsum[0]);
            inv = _rcp_2;
        }
        lsum[0] = inv;
        float inv_0 = 0.0f;
        if (lsum[1] != 0.0f) {
            float _rcp_3 = approx_rcp(lsum[1]);
            inv_0 = _rcp_3;
        }
        lsum[1] = inv_0;
        int hh6 = 2 * cq;
        int dd = warp_1 * 16 + g;
        {
            uint32_t _addr_17 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_17), "f"(d_o0[0] * lsum[0]) : "memory");
        }
        {
            uint32_t _addr_18 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6 * 128 + dd + 64) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_18), "f"(d_o1[0] * lsum[0]) : "memory");
        }
        int hh6_1 = 2 * cq + 1;
        int dd_2 = warp_1 * 16 + g;
        {
            uint32_t _addr_19 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_1 * 128 + dd_2) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_19), "f"(d_o0[1] * lsum[1]) : "memory");
        }
        {
            uint32_t _addr_20 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_1 * 128 + dd_2 + 64) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_20), "f"(d_o1[1] * lsum[1]) : "memory");
        }
        int hh6_3 = 2 * cq;
        int dd_4 = warp_1 * 16 + g + 8;
        {
            uint32_t _addr_21 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_21), "f"(d_o0[2] * lsum[0]) : "memory");
        }
        {
            uint32_t _addr_22 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_3 * 128 + dd_4 + 64) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_22), "f"(d_o1[2] * lsum[0]) : "memory");
        }
        int hh6_5 = 2 * cq + 1;
        int dd_6 = warp_1 * 16 + g + 8;
        {
            uint32_t _addr_23 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_23), "f"(d_o0[3] * lsum[1]) : "memory");
        }
        {
            uint32_t _addr_24 = static_cast<uint32_t>(ident8_addr + (unsigned int)((hh6_5 * 128 + dd_6 + 64) * 4));
            asm volatile("st.shared.f32 [%0], %1;" :: "r"(_addr_24), "f"(d_o1[3] * lsum[1]) : "memory");
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        float ov[8];
        int vid3 = tid_1;
        int on = vid3 / 16;
        int oseg = vid3 - on * 16;
        int od0 = oseg * 8;
        if (on < heads_valid) {
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&ov[0])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(0) + 3]))
                : "r"(ident8_addr + (unsigned int)((on * 128 + od0) * 4)));
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&ov[4])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&ov[(4) + 3]))
                : "r"(ident8_addr + (unsigned int)((on * 128 + od0 + 4) * 4)));
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(ov[0 + 0], ov[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(ov[0 + 2], ov[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(ov[0 + 4], ov[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(ov[0 + 6], ov[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + ((q_row0 + on) * 128 + od0)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
