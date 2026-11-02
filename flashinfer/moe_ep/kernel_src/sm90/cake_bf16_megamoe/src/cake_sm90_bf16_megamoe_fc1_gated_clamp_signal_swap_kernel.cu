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
#ifndef SW_C_SLOTS
#define SW_C_SLOTS 6
#endif
#define SW_T_STAGES 3
#define SW_T_STAGE_BYTES 16384
#define WCX_SLOT_STRIDE 28160
#define SW_RING_OFF (SW_T_STAGES * SW_T_STAGE_BYTES)
#define SW_PLANE_BYTES 4096
#define SW_HDR_OFF 24576
#define SW_PRE_OFF 25088
#define SW_SLOT_OFF 25152
#define SMEM_MBAR_OFF (SW_RING_OFF + SW_C_SLOTS * WCX_SLOT_STRIDE)
#define SMEM_TOTAL (SMEM_MBAR_OFF + 256)
static_assert(SMEM_TOTAL <= 232448, "swap unit smem budget");
#define THREADS 384
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



// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
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






__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

// ---- FC1 -> FC2 readiness counters, dynamic FC2 tile claim, C1 fc2_done ----
#ifndef FE_DELAY_NS
#define FE_DELAY_NS 0
#endif
#ifndef FE_NO_READY_WAIT
#define FE_NO_READY_WAIT 0
#endif
// sched (u32): [0] FC2 tile ticket, [1] FC2 CTA exit ticket, [2] FC1 go word, [3] FC1 CTAs that published every tile,
// [FE_READY_OFF + slot] FC1 CTA-tiles stored for m-tile slot (expert-major prefix of ceil(rows / 128) + m-tile).
#define FE_READY_OFF 8u

__device__ __forceinline__ void fe_wait_acq(int mbar_addr, unsigned int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "FE_WAIT_ACQ:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra FE_WAIT_ACQ_DONE;\n\t"
        "bra FE_WAIT_ACQ;\n\t"
        "FE_WAIT_ACQ_DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(10000000u) : "memory");
}

__device__ __forceinline__ unsigned int fe_ld_acquire_gpu(const unsigned int* p) {
    unsigned int v;
    asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void fe_wait_ge(const unsigned int* p, unsigned int target) {
    while (fe_ld_acquire_gpu(p) < target) {
        __nanosleep(64);
    }
}

// Release one CTA-tile: the CTA-scope acquire on `stored` made the math threads' stores visible to this thread;
// the gpu fence makes them (cumulatively) visible before the counter increment.
__device__ __forceinline__ void fe_signal_add(unsigned int* p) {
    asm volatile("fence.acq_rel.gpu;" ::: "memory");
    asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(p) : "memory");
}

__device__ __forceinline__ unsigned int fe_wait_slot(int addr, unsigned int seq) {
    unsigned long long v;
    asm volatile("ld.relaxed.cluster.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"(addr) : "memory");
    while ((unsigned int)(v >> 32) != seq) {
        __nanosleep(32);
        asm volatile("ld.relaxed.cluster.shared::cta.u64 %0, [%1];" : "=l"(v) : "r"(addr) : "memory");
    }
    return (unsigned int)v;
}

// Leader: tagged tile id into the same slot of both CTAs (one 8-byte relaxed store each; the tag is the tile sequence
// number, so no other data is ordered by it and no cluster-scope release is needed).
__device__ __forceinline__ void fe_publish(int addr, unsigned int seq, unsigned int info) {
    const unsigned long long v = ((unsigned long long)seq << 32) | (unsigned long long)info;
    asm volatile("st.relaxed.cluster.shared::cta.u64 [%0], %1;" :: "r"(addr), "l"(v) : "memory");
    unsigned int raddr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(raddr) : "r"(addr), "r"(1u));
    asm volatile("st.relaxed.cluster.shared::cluster.u64 [%0], %1;" :: "r"(raddr), "l"(v) : "memory");
}

struct FeWalk {
    unsigned int e, tb, mb, nte, mt;
};

__device__ __forceinline__ void fe_walk_load(FeWalk& w, const long long* offsets, unsigned int n_tiles, unsigned int n_pairs) {
    const unsigned int r0 = (unsigned int)offsets[w.e];
    const unsigned int r1 = (unsigned int)offsets[w.e + 1];
    const unsigned int mt = (r1 - r0 + 127u) / 128u;
    const unsigned int pp = mt / 2u;
    w.mt = mt;
    w.nte = pp * n_tiles + (mt - pp * 2u) * n_pairs;
}

// Global expert-major tile index g -> (e << 22) | t (t = tile within expert e, the original enumeration), or the
// sentinel past the last tile; mb = m-tile slot base of e.  g is non-decreasing per leader (one global ticket).
__device__ __forceinline__ unsigned int fe_decode(FeWalk& w, unsigned int g, unsigned int num_experts,
                                                    const long long* offsets, unsigned int n_tiles, unsigned int n_pairs,
                                                    unsigned int& mb) {
    while (w.e < num_experts && g >= w.tb + w.nte) {
        w.tb = w.tb + w.nte;
        w.mb = w.mb + w.mt;
        w.e = w.e + 1u;
        if (w.e < num_experts) {
            fe_walk_load(w, offsets, n_tiles, n_pairs);
        }
    }
    if (w.e >= num_experts) {
        return 0xFFFFFFFFu;
    }
    mb = w.mb;
    return (w.e << 22) | (g - w.tb);
}



// ---- swap-AB helpers
__device__ __forceinline__ uint64_t sw_desc(uint32_t saddr) {
    // the sealed descriptor form: start >> 4 | LBO 0 | SBO 1024 B (64) | SWIZZLE_128B
    return ((uint64_t)((saddr >> 4) & 0x3FFFu)) | ((uint64_t)0 << 16) | ((uint64_t)64 << 32) | (1ULL << 62);
}
__device__ __forceinline__ uint32_t sw_movt(uint32_t x) {
    uint32_t y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
__device__ __forceinline__ uint32_t sw_bf2(float a, float b) {
    __nv_bfloat162 v = __float22bfloat162_rn(make_float2(a, b));
    return reinterpret_cast<uint32_t*>(&v)[0];
}
// predicated (branch-free) mbarrier arrive: arrives iff p != 0
__device__ __forceinline__ void sw_arrive_if(int mbar_addr, uint32_t p) {
    asm volatile("{\n .reg .pred q;\n setp.ne.u32 q, %1, 0;\n @q mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n}"
                 :: "r"(mbar_addr), "r"(p) : "memory");
}
// PTX prmt (default mode: selector bit 3 replicates the sign of the selected byte), unlike __byte_perm
__device__ __forceinline__ uint32_t sw_prmt(uint32_t a, uint32_t b, uint32_t sel) {
    uint32_t d;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(sel));
    return d;
}
// consumer wait with a "memory" clobber: the stage's shared-memory loads are not hoisted above it by the compiler
__device__ __forceinline__ void sw_wait(int mbar_addr, uint32_t parity) {
    asm volatile("{\n .reg .pred P1;\n SW_WAIT_%=:\n mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
                 " @!P1 bra SW_WAIT_%=;\n}" :: "r"(mbar_addr), "r"(parity), "r"(10000000u) : "memory");
}
// the sealed gated-clamp epilogue's max (no flush-to-zero qualifier)
__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}

// one swap-AB CTA-tile with N = 16 token columns: k loop + epilogue (math WG h = math_wg_idx, 128 threads)
__device__ __forceinline__ void sw_tile_n16(const char* smem_raw, int smem, unsigned int& it, unsigned int num_k_stages,
        unsigned int math_wg_idx, unsigned int warp_in_wg, unsigned int lane, unsigned int m0, unsigned int n_sel,
        unsigned int row_lim, unsigned int n_out, __nv_bfloat16* __restrict__ D, int mbar_base, float clamp_limit) {
    float acc0[8], acc1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
    // operand fence: the zeroed accumulators are materialized before the first wgmma.fence (register writes are not
    // ordered by the asm "memory" clobber)
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    const unsigned int ct = math_wg_idx * 128u + warp_in_wg * 32u + lane;
    for (unsigned int k = 0; k < num_k_stages; ++k) {
        const unsigned int slot = it % SW_C_SLOTS, stage = it % SW_T_STAGES;
        const char* stg = smem_raw + SW_RING_OFF + slot * WCX_SLOT_STRIDE;
        sw_wait(mbar_base + 48 + slot * 8, it / SW_C_SLOTS & 1);
        // codec v2 register decode (branch-free; escape = select)
        const unsigned int t4 = ct & 3u;
        const uint2 hd = *reinterpret_cast<const uint2*>(stg + SW_HDR_OFF + (ct >> 2) * 8);
        const uint32_t pre = *reinterpret_cast<const uint8_t*>(stg + SW_PRE_OFF + (ct >> 2));
        // bit q of eb = chunk q of the thread's quad is escaped (header byte 0xFF)
        const uint32_t eb = ((((__vcmpeq4(hd.x, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu)
                          | (((((__vcmpeq4(hd.y, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu) << 4);
        uint4 p0, p1, p2, p3, p4, p5;
        uint32_t sw_000, sw_001, sw_010, sw_011, sw_100, sw_101, sw_110, sw_111;
        uint32_t fa0[4], fb0[4], fa1[4], fb1[4], fa2[4], fb2[4], fa3[4], fb3[4];
        p0 = *reinterpret_cast<const uint4*>(stg + 0 * SW_PLANE_BYTES + ct * 16);
        p4 = *reinterpret_cast<const uint4*>(stg + 4 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p4.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p0.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[2] = (sw_prmt(p4.x, p4.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p0.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[3] = (sw_prmt(p4.x, p4.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p0.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[2] = (sw_prmt(p4.y, p4.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p0.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[3] = (sw_prmt(p4.y, p4.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        sw_wait(mbar_base + 0 + stage * 8, it / SW_T_STAGES & 1);
        const uint32_t tb = (uint32_t)smem + stage * SW_T_STAGE_BYTES;
        const unsigned int prev = (it + SW_T_STAGES - 1) % SW_T_STAGES;
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa0[0]), "+r"(fa0[1]), "+r"(fa0[2]), "+r"(fa0[3]),
                         "+r"(fb0[0]), "+r"(fb0[1]), "+r"(fb0[2]), "+r"(fb0[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 0u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]) : "r"(fa0[0]), "r"(fa0[1]), "r"(fa0[2]), "r"(fa0[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]) : "r"(fb0[0]), "r"(fb0[1]), "r"(fb0[2]), "r"(fb0[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p1 = *reinterpret_cast<const uint4*>(stg + 1 * SW_PLANE_BYTES + ct * 16);
        sw_000 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 0) & 1u) ? (pre + __popc(eb & 0x0u)) * 16u + 4u * t4 : 0u));
        sw_001 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 2) & 1u) ? (pre + __popc(eb & 0x3u)) * 16u + 4u * t4 : 0u));
        sw_010 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 4) & 1u) ? (pre + __popc(eb & 0xFu)) * 16u + 4u * t4 : 0u));
        sw_011 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 6) & 1u) ? (pre + __popc(eb & 0x3Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p4.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p1.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[2] = (sw_000 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p1.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[3] = (sw_001 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p1.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[2] = (sw_010 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p1.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[3] = (sw_011 & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            // k-step k-1's groups are complete: release its token stage (one arrive per math warp)
            sw_arrive_if(mbar_base + 24 + prev * 8, (lane == 0u) & (k != 0u));
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa1[0]), "+r"(fa1[1]), "+r"(fa1[2]), "+r"(fa1[3]),
                         "+r"(fb1[0]), "+r"(fb1[1]), "+r"(fb1[2]), "+r"(fb1[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 32u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]) : "r"(fa1[0]), "r"(fa1[1]), "r"(fa1[2]), "r"(fa1[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]) : "r"(fb1[0]), "r"(fb1[1]), "r"(fb1[2]), "r"(fb1[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p2 = *reinterpret_cast<const uint4*>(stg + 2 * SW_PLANE_BYTES + ct * 16);
        p5 = *reinterpret_cast<const uint4*>(stg + 5 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p5.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p2.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[2] = (sw_prmt(p5.x, p5.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p2.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[3] = (sw_prmt(p5.x, p5.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p2.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[2] = (sw_prmt(p5.y, p5.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p2.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[3] = (sw_prmt(p5.y, p5.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa2[0]), "+r"(fa2[1]), "+r"(fa2[2]), "+r"(fa2[3]),
                         "+r"(fb2[0]), "+r"(fb2[1]), "+r"(fb2[2]), "+r"(fb2[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 64u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]) : "r"(fa2[0]), "r"(fa2[1]), "r"(fa2[2]), "r"(fa2[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]) : "r"(fb2[0]), "r"(fb2[1]), "r"(fb2[2]), "r"(fb2[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p3 = *reinterpret_cast<const uint4*>(stg + 3 * SW_PLANE_BYTES + ct * 16);
        sw_100 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 1) & 1u) ? (pre + __popc(eb & 0x1u)) * 16u + 4u * t4 : 0u));
        sw_101 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 3) & 1u) ? (pre + __popc(eb & 0x7u)) * 16u + 4u * t4 : 0u));
        sw_110 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 5) & 1u) ? (pre + __popc(eb & 0x1Fu)) * 16u + 4u * t4 : 0u));
        sw_111 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 7) & 1u) ? (pre + __popc(eb & 0x7Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p5.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p3.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[2] = (sw_100 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p3.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[3] = (sw_101 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p3.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[2] = (sw_110 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p3.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[3] = (sw_111 & em) | (dec & ~em);
                }
            }
        }
        // all LDS of this stage are done: release the compressed slot
        __syncwarp();
        sw_arrive_if(mbar_base + 96 + slot * 8, lane == 0u);
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa3[0]), "+r"(fa3[1]), "+r"(fa3[2]), "+r"(fa3[3]),
                         "+r"(fb3[0]), "+r"(fb3[1]), "+r"(fb3[2]), "+r"(fb3[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 96u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]) : "r"(fa3[0]), "r"(fa3[1]), "r"(fa3[2]), "r"(fa3[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, {%8, %9, %10, %11}, %12, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]) : "r"(fb3[0]), "r"(fb3[1]), "r"(fb3[2]), "r"(fb3[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        it = it + 1;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    // operand fence: the epilogue reads the accumulators only after the wait
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    sw_arrive_if(mbar_base + 24 + ((it + SW_T_STAGES - 1) % SW_T_STAGES) * 8, lane == 0u);
    // epilogue: element (W row 16 w + g + 8 rg of block b, token 8 j + 2 t + c) -> D[m0 + token][I col]
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int cb = n_sel * 128u + math_wg_idx * 64u + warp_in_wg * 16u;
#pragma unroll
    for (int j = 0; j < 2; ++j) {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
            // sealed fc1_gated_clamp_signal: g and u clamped to [-clamp_limit, clamp_limit] (max.f32, then fminf)
            const float g0 = fminf(max_noftz(acc0[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float g1 = fminf(max_noftz(acc0[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            const float u0 = fminf(max_noftz(acc1[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float u1 = fminf(max_noftz(acc1[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            float _exp2_0 = approx_exp2(g0 * -1.4426950408889634f);
            float _rcp_0 = approx_rcp(_exp2_0 + 1.0f);
            float _exp2_1 = approx_exp2(g1 * -1.4426950408889634f);
            float _rcp_1 = approx_rcp(_exp2_1 + 1.0f);
            const uint32_t pk = sw_bf2(g0 * _rcp_0 * u0, g1 * _rcp_1 * u1);
            const uint32_t tr = sw_movt(pk);
            const unsigned int row = m0 + 8u * j + g;
            if (row < row_lim) {
                *reinterpret_cast<uint32_t*>(D + ((size_t)row * n_out + cb + 8u * rg + 2u * t)) = tr;
            }
        }
    }
}

// one swap-AB CTA-tile with N = 64 token columns: k loop + epilogue (math WG h = math_wg_idx, 128 threads)
__device__ __forceinline__ void sw_tile_n64(const char* smem_raw, int smem, unsigned int& it, unsigned int num_k_stages,
        unsigned int math_wg_idx, unsigned int warp_in_wg, unsigned int lane, unsigned int m0, unsigned int n_sel,
        unsigned int row_lim, unsigned int n_out, __nv_bfloat16* __restrict__ D, int mbar_base, float clamp_limit) {
    float acc0[32], acc1[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
    // operand fence: the zeroed accumulators are materialized before the first wgmma.fence (register writes are not
    // ordered by the asm "memory" clobber)
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    asm volatile("" : "+f"(acc0[8]), "+f"(acc1[8]) :: "memory");
    asm volatile("" : "+f"(acc0[9]), "+f"(acc1[9]) :: "memory");
    asm volatile("" : "+f"(acc0[10]), "+f"(acc1[10]) :: "memory");
    asm volatile("" : "+f"(acc0[11]), "+f"(acc1[11]) :: "memory");
    asm volatile("" : "+f"(acc0[12]), "+f"(acc1[12]) :: "memory");
    asm volatile("" : "+f"(acc0[13]), "+f"(acc1[13]) :: "memory");
    asm volatile("" : "+f"(acc0[14]), "+f"(acc1[14]) :: "memory");
    asm volatile("" : "+f"(acc0[15]), "+f"(acc1[15]) :: "memory");
    asm volatile("" : "+f"(acc0[16]), "+f"(acc1[16]) :: "memory");
    asm volatile("" : "+f"(acc0[17]), "+f"(acc1[17]) :: "memory");
    asm volatile("" : "+f"(acc0[18]), "+f"(acc1[18]) :: "memory");
    asm volatile("" : "+f"(acc0[19]), "+f"(acc1[19]) :: "memory");
    asm volatile("" : "+f"(acc0[20]), "+f"(acc1[20]) :: "memory");
    asm volatile("" : "+f"(acc0[21]), "+f"(acc1[21]) :: "memory");
    asm volatile("" : "+f"(acc0[22]), "+f"(acc1[22]) :: "memory");
    asm volatile("" : "+f"(acc0[23]), "+f"(acc1[23]) :: "memory");
    asm volatile("" : "+f"(acc0[24]), "+f"(acc1[24]) :: "memory");
    asm volatile("" : "+f"(acc0[25]), "+f"(acc1[25]) :: "memory");
    asm volatile("" : "+f"(acc0[26]), "+f"(acc1[26]) :: "memory");
    asm volatile("" : "+f"(acc0[27]), "+f"(acc1[27]) :: "memory");
    asm volatile("" : "+f"(acc0[28]), "+f"(acc1[28]) :: "memory");
    asm volatile("" : "+f"(acc0[29]), "+f"(acc1[29]) :: "memory");
    asm volatile("" : "+f"(acc0[30]), "+f"(acc1[30]) :: "memory");
    asm volatile("" : "+f"(acc0[31]), "+f"(acc1[31]) :: "memory");
    const unsigned int ct = math_wg_idx * 128u + warp_in_wg * 32u + lane;
    for (unsigned int k = 0; k < num_k_stages; ++k) {
        const unsigned int slot = it % SW_C_SLOTS, stage = it % SW_T_STAGES;
        const char* stg = smem_raw + SW_RING_OFF + slot * WCX_SLOT_STRIDE;
        sw_wait(mbar_base + 48 + slot * 8, it / SW_C_SLOTS & 1);
        // codec v2 register decode (branch-free; escape = select)
        const unsigned int t4 = ct & 3u;
        const uint2 hd = *reinterpret_cast<const uint2*>(stg + SW_HDR_OFF + (ct >> 2) * 8);
        const uint32_t pre = *reinterpret_cast<const uint8_t*>(stg + SW_PRE_OFF + (ct >> 2));
        // bit q of eb = chunk q of the thread's quad is escaped (header byte 0xFF)
        const uint32_t eb = ((((__vcmpeq4(hd.x, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu)
                          | (((((__vcmpeq4(hd.y, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu) << 4);
        uint4 p0, p1, p2, p3, p4, p5;
        uint32_t sw_000, sw_001, sw_010, sw_011, sw_100, sw_101, sw_110, sw_111;
        uint32_t fa0[4], fb0[4], fa1[4], fb1[4], fa2[4], fb2[4], fa3[4], fb3[4];
        p0 = *reinterpret_cast<const uint4*>(stg + 0 * SW_PLANE_BYTES + ct * 16);
        p4 = *reinterpret_cast<const uint4*>(stg + 4 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p4.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p0.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[2] = (sw_prmt(p4.x, p4.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p0.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[3] = (sw_prmt(p4.x, p4.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p0.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[2] = (sw_prmt(p4.y, p4.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p0.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[3] = (sw_prmt(p4.y, p4.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        sw_wait(mbar_base + 0 + stage * 8, it / SW_T_STAGES & 1);
        const uint32_t tb = (uint32_t)smem + stage * SW_T_STAGE_BYTES;
        const unsigned int prev = (it + SW_T_STAGES - 1) % SW_T_STAGES;
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa0[0]), "+r"(fa0[1]), "+r"(fa0[2]), "+r"(fa0[3]),
                         "+r"(fb0[0]), "+r"(fb0[1]), "+r"(fb0[2]), "+r"(fb0[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 0u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]) : "r"(fa0[0]), "r"(fa0[1]), "r"(fa0[2]), "r"(fa0[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]) : "r"(fb0[0]), "r"(fb0[1]), "r"(fb0[2]), "r"(fb0[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p1 = *reinterpret_cast<const uint4*>(stg + 1 * SW_PLANE_BYTES + ct * 16);
        sw_000 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 0) & 1u) ? (pre + __popc(eb & 0x0u)) * 16u + 4u * t4 : 0u));
        sw_001 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 2) & 1u) ? (pre + __popc(eb & 0x3u)) * 16u + 4u * t4 : 0u));
        sw_010 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 4) & 1u) ? (pre + __popc(eb & 0xFu)) * 16u + 4u * t4 : 0u));
        sw_011 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 6) & 1u) ? (pre + __popc(eb & 0x3Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p4.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p1.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[2] = (sw_000 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p1.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[3] = (sw_001 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p1.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[2] = (sw_010 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p1.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[3] = (sw_011 & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            // k-step k-1's groups are complete: release its token stage (one arrive per math warp)
            sw_arrive_if(mbar_base + 24 + prev * 8, (lane == 0u) & (k != 0u));
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa1[0]), "+r"(fa1[1]), "+r"(fa1[2]), "+r"(fa1[3]),
                         "+r"(fb1[0]), "+r"(fb1[1]), "+r"(fb1[2]), "+r"(fb1[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 32u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]) : "r"(fa1[0]), "r"(fa1[1]), "r"(fa1[2]), "r"(fa1[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]) : "r"(fb1[0]), "r"(fb1[1]), "r"(fb1[2]), "r"(fb1[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p2 = *reinterpret_cast<const uint4*>(stg + 2 * SW_PLANE_BYTES + ct * 16);
        p5 = *reinterpret_cast<const uint4*>(stg + 5 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p5.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p2.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[2] = (sw_prmt(p5.x, p5.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p2.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[3] = (sw_prmt(p5.x, p5.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p2.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[2] = (sw_prmt(p5.y, p5.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p2.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[3] = (sw_prmt(p5.y, p5.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa2[0]), "+r"(fa2[1]), "+r"(fa2[2]), "+r"(fa2[3]),
                         "+r"(fb2[0]), "+r"(fb2[1]), "+r"(fb2[2]), "+r"(fb2[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 64u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]) : "r"(fa2[0]), "r"(fa2[1]), "r"(fa2[2]), "r"(fa2[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]) : "r"(fb2[0]), "r"(fb2[1]), "r"(fb2[2]), "r"(fb2[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p3 = *reinterpret_cast<const uint4*>(stg + 3 * SW_PLANE_BYTES + ct * 16);
        sw_100 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 1) & 1u) ? (pre + __popc(eb & 0x1u)) * 16u + 4u * t4 : 0u));
        sw_101 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 3) & 1u) ? (pre + __popc(eb & 0x7u)) * 16u + 4u * t4 : 0u));
        sw_110 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 5) & 1u) ? (pre + __popc(eb & 0x1Fu)) * 16u + 4u * t4 : 0u));
        sw_111 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 7) & 1u) ? (pre + __popc(eb & 0x7Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p5.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p3.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[2] = (sw_100 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p3.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[3] = (sw_101 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p3.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[2] = (sw_110 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p3.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[3] = (sw_111 & em) | (dec & ~em);
                }
            }
        }
        // all LDS of this stage are done: release the compressed slot
        __syncwarp();
        sw_arrive_if(mbar_base + 96 + slot * 8, lane == 0u);
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa3[0]), "+r"(fa3[1]), "+r"(fa3[2]), "+r"(fa3[3]),
                         "+r"(fb3[0]), "+r"(fb3[1]), "+r"(fb3[2]), "+r"(fb3[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 96u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]) : "r"(fa3[0]), "r"(fa3[1]), "r"(fa3[2]), "r"(fa3[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, {%32, %33, %34, %35}, %36, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]) : "r"(fb3[0]), "r"(fb3[1]), "r"(fb3[2]), "r"(fb3[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        it = it + 1;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    // operand fence: the epilogue reads the accumulators only after the wait
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    asm volatile("" : "+f"(acc0[8]), "+f"(acc1[8]) :: "memory");
    asm volatile("" : "+f"(acc0[9]), "+f"(acc1[9]) :: "memory");
    asm volatile("" : "+f"(acc0[10]), "+f"(acc1[10]) :: "memory");
    asm volatile("" : "+f"(acc0[11]), "+f"(acc1[11]) :: "memory");
    asm volatile("" : "+f"(acc0[12]), "+f"(acc1[12]) :: "memory");
    asm volatile("" : "+f"(acc0[13]), "+f"(acc1[13]) :: "memory");
    asm volatile("" : "+f"(acc0[14]), "+f"(acc1[14]) :: "memory");
    asm volatile("" : "+f"(acc0[15]), "+f"(acc1[15]) :: "memory");
    asm volatile("" : "+f"(acc0[16]), "+f"(acc1[16]) :: "memory");
    asm volatile("" : "+f"(acc0[17]), "+f"(acc1[17]) :: "memory");
    asm volatile("" : "+f"(acc0[18]), "+f"(acc1[18]) :: "memory");
    asm volatile("" : "+f"(acc0[19]), "+f"(acc1[19]) :: "memory");
    asm volatile("" : "+f"(acc0[20]), "+f"(acc1[20]) :: "memory");
    asm volatile("" : "+f"(acc0[21]), "+f"(acc1[21]) :: "memory");
    asm volatile("" : "+f"(acc0[22]), "+f"(acc1[22]) :: "memory");
    asm volatile("" : "+f"(acc0[23]), "+f"(acc1[23]) :: "memory");
    asm volatile("" : "+f"(acc0[24]), "+f"(acc1[24]) :: "memory");
    asm volatile("" : "+f"(acc0[25]), "+f"(acc1[25]) :: "memory");
    asm volatile("" : "+f"(acc0[26]), "+f"(acc1[26]) :: "memory");
    asm volatile("" : "+f"(acc0[27]), "+f"(acc1[27]) :: "memory");
    asm volatile("" : "+f"(acc0[28]), "+f"(acc1[28]) :: "memory");
    asm volatile("" : "+f"(acc0[29]), "+f"(acc1[29]) :: "memory");
    asm volatile("" : "+f"(acc0[30]), "+f"(acc1[30]) :: "memory");
    asm volatile("" : "+f"(acc0[31]), "+f"(acc1[31]) :: "memory");
    sw_arrive_if(mbar_base + 24 + ((it + SW_T_STAGES - 1) % SW_T_STAGES) * 8, lane == 0u);
    // epilogue: element (W row 16 w + g + 8 rg of block b, token 8 j + 2 t + c) -> D[m0 + token][I col]
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int cb = n_sel * 128u + math_wg_idx * 64u + warp_in_wg * 16u;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
            // sealed fc1_gated_clamp_signal: g and u clamped to [-clamp_limit, clamp_limit] (max.f32, then fminf)
            const float g0 = fminf(max_noftz(acc0[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float g1 = fminf(max_noftz(acc0[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            const float u0 = fminf(max_noftz(acc1[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float u1 = fminf(max_noftz(acc1[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            float _exp2_0 = approx_exp2(g0 * -1.4426950408889634f);
            float _rcp_0 = approx_rcp(_exp2_0 + 1.0f);
            float _exp2_1 = approx_exp2(g1 * -1.4426950408889634f);
            float _rcp_1 = approx_rcp(_exp2_1 + 1.0f);
            const uint32_t pk = sw_bf2(g0 * _rcp_0 * u0, g1 * _rcp_1 * u1);
            const uint32_t tr = sw_movt(pk);
            const unsigned int row = m0 + 8u * j + g;
            if (row < row_lim) {
                *reinterpret_cast<uint32_t*>(D + ((size_t)row * n_out + cb + 8u * rg + 2u * t)) = tr;
            }
        }
    }
}

// one swap-AB CTA-tile with N = 128 token columns: k loop + epilogue (math WG h = math_wg_idx, 128 threads)
__device__ __forceinline__ void sw_tile_n128(const char* smem_raw, int smem, unsigned int& it, unsigned int num_k_stages,
        unsigned int math_wg_idx, unsigned int warp_in_wg, unsigned int lane, unsigned int m0, unsigned int n_sel,
        unsigned int row_lim, unsigned int n_out, __nv_bfloat16* __restrict__ D, int mbar_base, float clamp_limit) {
    float acc0[64], acc1[64];
#pragma unroll
    for (int i = 0; i < 64; ++i) { acc0[i] = 0.0f; acc1[i] = 0.0f; }
    // operand fence: the zeroed accumulators are materialized before the first wgmma.fence (register writes are not
    // ordered by the asm "memory" clobber)
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    asm volatile("" : "+f"(acc0[8]), "+f"(acc1[8]) :: "memory");
    asm volatile("" : "+f"(acc0[9]), "+f"(acc1[9]) :: "memory");
    asm volatile("" : "+f"(acc0[10]), "+f"(acc1[10]) :: "memory");
    asm volatile("" : "+f"(acc0[11]), "+f"(acc1[11]) :: "memory");
    asm volatile("" : "+f"(acc0[12]), "+f"(acc1[12]) :: "memory");
    asm volatile("" : "+f"(acc0[13]), "+f"(acc1[13]) :: "memory");
    asm volatile("" : "+f"(acc0[14]), "+f"(acc1[14]) :: "memory");
    asm volatile("" : "+f"(acc0[15]), "+f"(acc1[15]) :: "memory");
    asm volatile("" : "+f"(acc0[16]), "+f"(acc1[16]) :: "memory");
    asm volatile("" : "+f"(acc0[17]), "+f"(acc1[17]) :: "memory");
    asm volatile("" : "+f"(acc0[18]), "+f"(acc1[18]) :: "memory");
    asm volatile("" : "+f"(acc0[19]), "+f"(acc1[19]) :: "memory");
    asm volatile("" : "+f"(acc0[20]), "+f"(acc1[20]) :: "memory");
    asm volatile("" : "+f"(acc0[21]), "+f"(acc1[21]) :: "memory");
    asm volatile("" : "+f"(acc0[22]), "+f"(acc1[22]) :: "memory");
    asm volatile("" : "+f"(acc0[23]), "+f"(acc1[23]) :: "memory");
    asm volatile("" : "+f"(acc0[24]), "+f"(acc1[24]) :: "memory");
    asm volatile("" : "+f"(acc0[25]), "+f"(acc1[25]) :: "memory");
    asm volatile("" : "+f"(acc0[26]), "+f"(acc1[26]) :: "memory");
    asm volatile("" : "+f"(acc0[27]), "+f"(acc1[27]) :: "memory");
    asm volatile("" : "+f"(acc0[28]), "+f"(acc1[28]) :: "memory");
    asm volatile("" : "+f"(acc0[29]), "+f"(acc1[29]) :: "memory");
    asm volatile("" : "+f"(acc0[30]), "+f"(acc1[30]) :: "memory");
    asm volatile("" : "+f"(acc0[31]), "+f"(acc1[31]) :: "memory");
    asm volatile("" : "+f"(acc0[32]), "+f"(acc1[32]) :: "memory");
    asm volatile("" : "+f"(acc0[33]), "+f"(acc1[33]) :: "memory");
    asm volatile("" : "+f"(acc0[34]), "+f"(acc1[34]) :: "memory");
    asm volatile("" : "+f"(acc0[35]), "+f"(acc1[35]) :: "memory");
    asm volatile("" : "+f"(acc0[36]), "+f"(acc1[36]) :: "memory");
    asm volatile("" : "+f"(acc0[37]), "+f"(acc1[37]) :: "memory");
    asm volatile("" : "+f"(acc0[38]), "+f"(acc1[38]) :: "memory");
    asm volatile("" : "+f"(acc0[39]), "+f"(acc1[39]) :: "memory");
    asm volatile("" : "+f"(acc0[40]), "+f"(acc1[40]) :: "memory");
    asm volatile("" : "+f"(acc0[41]), "+f"(acc1[41]) :: "memory");
    asm volatile("" : "+f"(acc0[42]), "+f"(acc1[42]) :: "memory");
    asm volatile("" : "+f"(acc0[43]), "+f"(acc1[43]) :: "memory");
    asm volatile("" : "+f"(acc0[44]), "+f"(acc1[44]) :: "memory");
    asm volatile("" : "+f"(acc0[45]), "+f"(acc1[45]) :: "memory");
    asm volatile("" : "+f"(acc0[46]), "+f"(acc1[46]) :: "memory");
    asm volatile("" : "+f"(acc0[47]), "+f"(acc1[47]) :: "memory");
    asm volatile("" : "+f"(acc0[48]), "+f"(acc1[48]) :: "memory");
    asm volatile("" : "+f"(acc0[49]), "+f"(acc1[49]) :: "memory");
    asm volatile("" : "+f"(acc0[50]), "+f"(acc1[50]) :: "memory");
    asm volatile("" : "+f"(acc0[51]), "+f"(acc1[51]) :: "memory");
    asm volatile("" : "+f"(acc0[52]), "+f"(acc1[52]) :: "memory");
    asm volatile("" : "+f"(acc0[53]), "+f"(acc1[53]) :: "memory");
    asm volatile("" : "+f"(acc0[54]), "+f"(acc1[54]) :: "memory");
    asm volatile("" : "+f"(acc0[55]), "+f"(acc1[55]) :: "memory");
    asm volatile("" : "+f"(acc0[56]), "+f"(acc1[56]) :: "memory");
    asm volatile("" : "+f"(acc0[57]), "+f"(acc1[57]) :: "memory");
    asm volatile("" : "+f"(acc0[58]), "+f"(acc1[58]) :: "memory");
    asm volatile("" : "+f"(acc0[59]), "+f"(acc1[59]) :: "memory");
    asm volatile("" : "+f"(acc0[60]), "+f"(acc1[60]) :: "memory");
    asm volatile("" : "+f"(acc0[61]), "+f"(acc1[61]) :: "memory");
    asm volatile("" : "+f"(acc0[62]), "+f"(acc1[62]) :: "memory");
    asm volatile("" : "+f"(acc0[63]), "+f"(acc1[63]) :: "memory");
    const unsigned int ct = math_wg_idx * 128u + warp_in_wg * 32u + lane;
    for (unsigned int k = 0; k < num_k_stages; ++k) {
        const unsigned int slot = it % SW_C_SLOTS, stage = it % SW_T_STAGES;
        const char* stg = smem_raw + SW_RING_OFF + slot * WCX_SLOT_STRIDE;
        sw_wait(mbar_base + 48 + slot * 8, it / SW_C_SLOTS & 1);
        // codec v2 register decode (branch-free; escape = select)
        const unsigned int t4 = ct & 3u;
        const uint2 hd = *reinterpret_cast<const uint2*>(stg + SW_HDR_OFF + (ct >> 2) * 8);
        const uint32_t pre = *reinterpret_cast<const uint8_t*>(stg + SW_PRE_OFF + (ct >> 2));
        // bit q of eb = chunk q of the thread's quad is escaped (header byte 0xFF)
        const uint32_t eb = ((((__vcmpeq4(hd.x, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu)
                          | (((((__vcmpeq4(hd.y, 0xFFFFFFFFu) & 0x01010101u) * 0x01020408u) >> 24) & 0xFu) << 4);
        uint4 p0, p1, p2, p3, p4, p5;
        uint32_t sw_000, sw_001, sw_010, sw_011, sw_100, sw_101, sw_110, sw_111;
        uint32_t fa0[4], fb0[4], fa1[4], fb1[4], fa2[4], fb2[4], fa3[4], fb3[4];
        p0 = *reinterpret_cast<const uint4*>(stg + 0 * SW_PLANE_BYTES + ct * 16);
        p4 = *reinterpret_cast<const uint4*>(stg + 4 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p4.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p0.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[2] = (sw_prmt(p4.x, p4.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p0.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa0[3] = (sw_prmt(p4.x, p4.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p0.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[2] = (sw_prmt(p4.y, p4.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p0.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb0[3] = (sw_prmt(p4.y, p4.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        sw_wait(mbar_base + 0 + stage * 8, it / SW_T_STAGES & 1);
        const uint32_t tb = (uint32_t)smem + stage * SW_T_STAGE_BYTES;
        const unsigned int prev = (it + SW_T_STAGES - 1) % SW_T_STAGES;
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa0[0]), "+r"(fa0[1]), "+r"(fa0[2]), "+r"(fa0[3]),
                         "+r"(fb0[0]), "+r"(fb0[1]), "+r"(fb0[2]), "+r"(fb0[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 0u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]), "+f"(acc0[32]), "+f"(acc0[33]), "+f"(acc0[34]), "+f"(acc0[35]), "+f"(acc0[36]), "+f"(acc0[37]), "+f"(acc0[38]), "+f"(acc0[39]), "+f"(acc0[40]), "+f"(acc0[41]), "+f"(acc0[42]), "+f"(acc0[43]), "+f"(acc0[44]), "+f"(acc0[45]), "+f"(acc0[46]), "+f"(acc0[47]), "+f"(acc0[48]), "+f"(acc0[49]), "+f"(acc0[50]), "+f"(acc0[51]), "+f"(acc0[52]), "+f"(acc0[53]), "+f"(acc0[54]), "+f"(acc0[55]), "+f"(acc0[56]), "+f"(acc0[57]), "+f"(acc0[58]), "+f"(acc0[59]), "+f"(acc0[60]), "+f"(acc0[61]), "+f"(acc0[62]), "+f"(acc0[63]) : "r"(fa0[0]), "r"(fa0[1]), "r"(fa0[2]), "r"(fa0[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]), "+f"(acc1[32]), "+f"(acc1[33]), "+f"(acc1[34]), "+f"(acc1[35]), "+f"(acc1[36]), "+f"(acc1[37]), "+f"(acc1[38]), "+f"(acc1[39]), "+f"(acc1[40]), "+f"(acc1[41]), "+f"(acc1[42]), "+f"(acc1[43]), "+f"(acc1[44]), "+f"(acc1[45]), "+f"(acc1[46]), "+f"(acc1[47]), "+f"(acc1[48]), "+f"(acc1[49]), "+f"(acc1[50]), "+f"(acc1[51]), "+f"(acc1[52]), "+f"(acc1[53]), "+f"(acc1[54]), "+f"(acc1[55]), "+f"(acc1[56]), "+f"(acc1[57]), "+f"(acc1[58]), "+f"(acc1[59]), "+f"(acc1[60]), "+f"(acc1[61]), "+f"(acc1[62]), "+f"(acc1[63]) : "r"(fb0[0]), "r"(fb0[1]), "r"(fb0[2]), "r"(fb0[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p1 = *reinterpret_cast<const uint4*>(stg + 1 * SW_PLANE_BYTES + ct * 16);
        sw_000 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 0) & 1u) ? (pre + __popc(eb & 0x0u)) * 16u + 4u * t4 : 0u));
        sw_001 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 2) & 1u) ? (pre + __popc(eb & 0x3u)) * 16u + 4u * t4 : 0u));
        sw_010 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 4) & 1u) ? (pre + __popc(eb & 0xFu)) * 16u + 4u * t4 : 0u));
        sw_011 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 6) & 1u) ? (pre + __popc(eb & 0x3Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p4.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 0) & 1u);
                const uint32_t smw = p1.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[2] = (sw_000 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 2) & 1u);
                const uint32_t smw = p1.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa1[3] = (sw_001 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p4.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 0) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 4) & 1u);
                const uint32_t smw = p1.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[2] = (sw_010 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 16) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 6) & 1u);
                const uint32_t smw = p1.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb1[3] = (sw_011 & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
            // k-step k-1's groups are complete: release its token stage (one arrive per math warp)
            sw_arrive_if(mbar_base + 24 + prev * 8, (lane == 0u) & (k != 0u));
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa1[0]), "+r"(fa1[1]), "+r"(fa1[2]), "+r"(fa1[3]),
                         "+r"(fb1[0]), "+r"(fb1[1]), "+r"(fb1[2]), "+r"(fb1[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 32u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]), "+f"(acc0[32]), "+f"(acc0[33]), "+f"(acc0[34]), "+f"(acc0[35]), "+f"(acc0[36]), "+f"(acc0[37]), "+f"(acc0[38]), "+f"(acc0[39]), "+f"(acc0[40]), "+f"(acc0[41]), "+f"(acc0[42]), "+f"(acc0[43]), "+f"(acc0[44]), "+f"(acc0[45]), "+f"(acc0[46]), "+f"(acc0[47]), "+f"(acc0[48]), "+f"(acc0[49]), "+f"(acc0[50]), "+f"(acc0[51]), "+f"(acc0[52]), "+f"(acc0[53]), "+f"(acc0[54]), "+f"(acc0[55]), "+f"(acc0[56]), "+f"(acc0[57]), "+f"(acc0[58]), "+f"(acc0[59]), "+f"(acc0[60]), "+f"(acc0[61]), "+f"(acc0[62]), "+f"(acc0[63]) : "r"(fa1[0]), "r"(fa1[1]), "r"(fa1[2]), "r"(fa1[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]), "+f"(acc1[32]), "+f"(acc1[33]), "+f"(acc1[34]), "+f"(acc1[35]), "+f"(acc1[36]), "+f"(acc1[37]), "+f"(acc1[38]), "+f"(acc1[39]), "+f"(acc1[40]), "+f"(acc1[41]), "+f"(acc1[42]), "+f"(acc1[43]), "+f"(acc1[44]), "+f"(acc1[45]), "+f"(acc1[46]), "+f"(acc1[47]), "+f"(acc1[48]), "+f"(acc1[49]), "+f"(acc1[50]), "+f"(acc1[51]), "+f"(acc1[52]), "+f"(acc1[53]), "+f"(acc1[54]), "+f"(acc1[55]), "+f"(acc1[56]), "+f"(acc1[57]), "+f"(acc1[58]), "+f"(acc1[59]), "+f"(acc1[60]), "+f"(acc1[61]), "+f"(acc1[62]), "+f"(acc1[63]) : "r"(fb1[0]), "r"(fb1[1]), "r"(fb1[2]), "r"(fb1[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p2 = *reinterpret_cast<const uint4*>(stg + 2 * SW_PLANE_BYTES + ct * 16);
        p5 = *reinterpret_cast<const uint4*>(stg + 5 * SW_PLANE_BYTES + ct * 16);
        {
            const uint32_t nw = p5.x;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p2.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[2] = (sw_prmt(p5.x, p5.z, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p2.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa2[3] = (sw_prmt(p5.x, p5.z, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.y;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p2.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[2] = (sw_prmt(p5.y, p5.w, 0x5410u) & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p2.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb2[3] = (sw_prmt(p5.y, p5.w, 0x7632u) & em) | (dec & ~em);
                }
            }
        }
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa2[0]), "+r"(fa2[1]), "+r"(fa2[2]), "+r"(fa2[3]),
                         "+r"(fb2[0]), "+r"(fb2[1]), "+r"(fb2[2]), "+r"(fb2[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 64u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]), "+f"(acc0[32]), "+f"(acc0[33]), "+f"(acc0[34]), "+f"(acc0[35]), "+f"(acc0[36]), "+f"(acc0[37]), "+f"(acc0[38]), "+f"(acc0[39]), "+f"(acc0[40]), "+f"(acc0[41]), "+f"(acc0[42]), "+f"(acc0[43]), "+f"(acc0[44]), "+f"(acc0[45]), "+f"(acc0[46]), "+f"(acc0[47]), "+f"(acc0[48]), "+f"(acc0[49]), "+f"(acc0[50]), "+f"(acc0[51]), "+f"(acc0[52]), "+f"(acc0[53]), "+f"(acc0[54]), "+f"(acc0[55]), "+f"(acc0[56]), "+f"(acc0[57]), "+f"(acc0[58]), "+f"(acc0[59]), "+f"(acc0[60]), "+f"(acc0[61]), "+f"(acc0[62]), "+f"(acc0[63]) : "r"(fa2[0]), "r"(fa2[1]), "r"(fa2[2]), "r"(fa2[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]), "+f"(acc1[32]), "+f"(acc1[33]), "+f"(acc1[34]), "+f"(acc1[35]), "+f"(acc1[36]), "+f"(acc1[37]), "+f"(acc1[38]), "+f"(acc1[39]), "+f"(acc1[40]), "+f"(acc1[41]), "+f"(acc1[42]), "+f"(acc1[43]), "+f"(acc1[44]), "+f"(acc1[45]), "+f"(acc1[46]), "+f"(acc1[47]), "+f"(acc1[48]), "+f"(acc1[49]), "+f"(acc1[50]), "+f"(acc1[51]), "+f"(acc1[52]), "+f"(acc1[53]), "+f"(acc1[54]), "+f"(acc1[55]), "+f"(acc1[56]), "+f"(acc1[57]), "+f"(acc1[58]), "+f"(acc1[59]), "+f"(acc1[60]), "+f"(acc1[61]), "+f"(acc1[62]), "+f"(acc1[63]) : "r"(fb2[0]), "r"(fb2[1]), "r"(fb2[2]), "r"(fb2[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
        // decode the next k16 step while this one is in flight
        p3 = *reinterpret_cast<const uint4*>(stg + 3 * SW_PLANE_BYTES + ct * 16);
        sw_100 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 1) & 1u) ? (pre + __popc(eb & 0x1u)) * 16u + 4u * t4 : 0u));
        sw_101 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 3) & 1u) ? (pre + __popc(eb & 0x7u)) * 16u + 4u * t4 : 0u));
        sw_110 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 5) & 1u) ? (pre + __popc(eb & 0x1Fu)) * 16u + 4u * t4 : 0u));
        sw_111 = *reinterpret_cast<const uint32_t*>(stg + SW_SLOT_OFF +
            (((eb >> 7) & 1u) ? (pre + __popc(eb & 0x7Fu)) * 16u + 4u * t4 : 0u));
        {
            const uint32_t nw = p5.z;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.x >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 1) & 1u);
                const uint32_t smw = p3.x;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[2] = (sw_100 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.x >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 3) & 1u);
                const uint32_t smw = p3.y;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fa3[3] = (sw_101 & em) | (dec & ~em);
                }
            }
        }
        {
            const uint32_t nw = p5.w;
            const uint32_t lo = nw & 0x0F0F0F0Fu, hi = (nw >> 4) & 0x0F0F0F0Fu;
            {
                const uint32_t ex7 = ((hd.y >> 8) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 5) & 1u);
                const uint32_t smw = p3.z;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8480u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[0] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8581u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[2] = (sw_110 & em) | (dec & ~em);
                }
            }
            {
                const uint32_t ex7 = ((hd.y >> 24) & 0xFFu) * 0x00800080u;
                const uint32_t em = 0u - ((eb >> 7) & 1u);
                const uint32_t smw = p3.w;
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x1100u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8682u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[1] = (smw & em) | (dec & ~em);
                }
                {
                    const uint32_t smp = sw_prmt(smw, 0u, 0x3322u);
                    const uint32_t x = sw_prmt(lo, hi, 0x8783u);
                    const uint32_t e = ex7 - x * 128u;  // (E - offset) << 7 per half; v2 has no zero code
                    const uint32_t dec = (smp & 0x807F807Fu) | e;
                    fb3[3] = (sw_111 & em) | (dec & ~em);
                }
            }
        }
        // all LDS of this stage are done: release the compressed slot
        __syncwarp();
        sw_arrive_if(mbar_base + 96 + slot * 8, lane == 0u);
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        {
            // the A fragments are operands of the fence: their register writes complete before it, the wgmma reads after it
            asm volatile("wgmma.fence.sync.aligned;" : "+r"(fa3[0]), "+r"(fa3[1]), "+r"(fa3[2]), "+r"(fa3[3]),
                         "+r"(fb3[0]), "+r"(fb3[1]), "+r"(fb3[2]), "+r"(fb3[3]) :: "memory");
            const uint64_t db = sw_desc(tb + 96u);
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc0[0]), "+f"(acc0[1]), "+f"(acc0[2]), "+f"(acc0[3]), "+f"(acc0[4]), "+f"(acc0[5]), "+f"(acc0[6]), "+f"(acc0[7]), "+f"(acc0[8]), "+f"(acc0[9]), "+f"(acc0[10]), "+f"(acc0[11]), "+f"(acc0[12]), "+f"(acc0[13]), "+f"(acc0[14]), "+f"(acc0[15]), "+f"(acc0[16]), "+f"(acc0[17]), "+f"(acc0[18]), "+f"(acc0[19]), "+f"(acc0[20]), "+f"(acc0[21]), "+f"(acc0[22]), "+f"(acc0[23]), "+f"(acc0[24]), "+f"(acc0[25]), "+f"(acc0[26]), "+f"(acc0[27]), "+f"(acc0[28]), "+f"(acc0[29]), "+f"(acc0[30]), "+f"(acc0[31]), "+f"(acc0[32]), "+f"(acc0[33]), "+f"(acc0[34]), "+f"(acc0[35]), "+f"(acc0[36]), "+f"(acc0[37]), "+f"(acc0[38]), "+f"(acc0[39]), "+f"(acc0[40]), "+f"(acc0[41]), "+f"(acc0[42]), "+f"(acc0[43]), "+f"(acc0[44]), "+f"(acc0[45]), "+f"(acc0[46]), "+f"(acc0[47]), "+f"(acc0[48]), "+f"(acc0[49]), "+f"(acc0[50]), "+f"(acc0[51]), "+f"(acc0[52]), "+f"(acc0[53]), "+f"(acc0[54]), "+f"(acc0[55]), "+f"(acc0[56]), "+f"(acc0[57]), "+f"(acc0[58]), "+f"(acc0[59]), "+f"(acc0[60]), "+f"(acc0[61]), "+f"(acc0[62]), "+f"(acc0[63]) : "r"(fa3[0]), "r"(fa3[1]), "r"(fa3[2]), "r"(fa3[3]), "l"(db) : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, {%64, %65, %66, %67}, %68, 1, 1, 1, 0;\n}\n" : "+f"(acc1[0]), "+f"(acc1[1]), "+f"(acc1[2]), "+f"(acc1[3]), "+f"(acc1[4]), "+f"(acc1[5]), "+f"(acc1[6]), "+f"(acc1[7]), "+f"(acc1[8]), "+f"(acc1[9]), "+f"(acc1[10]), "+f"(acc1[11]), "+f"(acc1[12]), "+f"(acc1[13]), "+f"(acc1[14]), "+f"(acc1[15]), "+f"(acc1[16]), "+f"(acc1[17]), "+f"(acc1[18]), "+f"(acc1[19]), "+f"(acc1[20]), "+f"(acc1[21]), "+f"(acc1[22]), "+f"(acc1[23]), "+f"(acc1[24]), "+f"(acc1[25]), "+f"(acc1[26]), "+f"(acc1[27]), "+f"(acc1[28]), "+f"(acc1[29]), "+f"(acc1[30]), "+f"(acc1[31]), "+f"(acc1[32]), "+f"(acc1[33]), "+f"(acc1[34]), "+f"(acc1[35]), "+f"(acc1[36]), "+f"(acc1[37]), "+f"(acc1[38]), "+f"(acc1[39]), "+f"(acc1[40]), "+f"(acc1[41]), "+f"(acc1[42]), "+f"(acc1[43]), "+f"(acc1[44]), "+f"(acc1[45]), "+f"(acc1[46]), "+f"(acc1[47]), "+f"(acc1[48]), "+f"(acc1[49]), "+f"(acc1[50]), "+f"(acc1[51]), "+f"(acc1[52]), "+f"(acc1[53]), "+f"(acc1[54]), "+f"(acc1[55]), "+f"(acc1[56]), "+f"(acc1[57]), "+f"(acc1[58]), "+f"(acc1[59]), "+f"(acc1[60]), "+f"(acc1[61]), "+f"(acc1[62]), "+f"(acc1[63]) : "r"(fb3[0]), "r"(fb3[1]), "r"(fb3[2]), "r"(fb3[3]), "l"(db) : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
        }
        it = it + 1;
    }
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    // operand fence: the epilogue reads the accumulators only after the wait
    asm volatile("" : "+f"(acc0[0]), "+f"(acc1[0]) :: "memory");
    asm volatile("" : "+f"(acc0[1]), "+f"(acc1[1]) :: "memory");
    asm volatile("" : "+f"(acc0[2]), "+f"(acc1[2]) :: "memory");
    asm volatile("" : "+f"(acc0[3]), "+f"(acc1[3]) :: "memory");
    asm volatile("" : "+f"(acc0[4]), "+f"(acc1[4]) :: "memory");
    asm volatile("" : "+f"(acc0[5]), "+f"(acc1[5]) :: "memory");
    asm volatile("" : "+f"(acc0[6]), "+f"(acc1[6]) :: "memory");
    asm volatile("" : "+f"(acc0[7]), "+f"(acc1[7]) :: "memory");
    asm volatile("" : "+f"(acc0[8]), "+f"(acc1[8]) :: "memory");
    asm volatile("" : "+f"(acc0[9]), "+f"(acc1[9]) :: "memory");
    asm volatile("" : "+f"(acc0[10]), "+f"(acc1[10]) :: "memory");
    asm volatile("" : "+f"(acc0[11]), "+f"(acc1[11]) :: "memory");
    asm volatile("" : "+f"(acc0[12]), "+f"(acc1[12]) :: "memory");
    asm volatile("" : "+f"(acc0[13]), "+f"(acc1[13]) :: "memory");
    asm volatile("" : "+f"(acc0[14]), "+f"(acc1[14]) :: "memory");
    asm volatile("" : "+f"(acc0[15]), "+f"(acc1[15]) :: "memory");
    asm volatile("" : "+f"(acc0[16]), "+f"(acc1[16]) :: "memory");
    asm volatile("" : "+f"(acc0[17]), "+f"(acc1[17]) :: "memory");
    asm volatile("" : "+f"(acc0[18]), "+f"(acc1[18]) :: "memory");
    asm volatile("" : "+f"(acc0[19]), "+f"(acc1[19]) :: "memory");
    asm volatile("" : "+f"(acc0[20]), "+f"(acc1[20]) :: "memory");
    asm volatile("" : "+f"(acc0[21]), "+f"(acc1[21]) :: "memory");
    asm volatile("" : "+f"(acc0[22]), "+f"(acc1[22]) :: "memory");
    asm volatile("" : "+f"(acc0[23]), "+f"(acc1[23]) :: "memory");
    asm volatile("" : "+f"(acc0[24]), "+f"(acc1[24]) :: "memory");
    asm volatile("" : "+f"(acc0[25]), "+f"(acc1[25]) :: "memory");
    asm volatile("" : "+f"(acc0[26]), "+f"(acc1[26]) :: "memory");
    asm volatile("" : "+f"(acc0[27]), "+f"(acc1[27]) :: "memory");
    asm volatile("" : "+f"(acc0[28]), "+f"(acc1[28]) :: "memory");
    asm volatile("" : "+f"(acc0[29]), "+f"(acc1[29]) :: "memory");
    asm volatile("" : "+f"(acc0[30]), "+f"(acc1[30]) :: "memory");
    asm volatile("" : "+f"(acc0[31]), "+f"(acc1[31]) :: "memory");
    asm volatile("" : "+f"(acc0[32]), "+f"(acc1[32]) :: "memory");
    asm volatile("" : "+f"(acc0[33]), "+f"(acc1[33]) :: "memory");
    asm volatile("" : "+f"(acc0[34]), "+f"(acc1[34]) :: "memory");
    asm volatile("" : "+f"(acc0[35]), "+f"(acc1[35]) :: "memory");
    asm volatile("" : "+f"(acc0[36]), "+f"(acc1[36]) :: "memory");
    asm volatile("" : "+f"(acc0[37]), "+f"(acc1[37]) :: "memory");
    asm volatile("" : "+f"(acc0[38]), "+f"(acc1[38]) :: "memory");
    asm volatile("" : "+f"(acc0[39]), "+f"(acc1[39]) :: "memory");
    asm volatile("" : "+f"(acc0[40]), "+f"(acc1[40]) :: "memory");
    asm volatile("" : "+f"(acc0[41]), "+f"(acc1[41]) :: "memory");
    asm volatile("" : "+f"(acc0[42]), "+f"(acc1[42]) :: "memory");
    asm volatile("" : "+f"(acc0[43]), "+f"(acc1[43]) :: "memory");
    asm volatile("" : "+f"(acc0[44]), "+f"(acc1[44]) :: "memory");
    asm volatile("" : "+f"(acc0[45]), "+f"(acc1[45]) :: "memory");
    asm volatile("" : "+f"(acc0[46]), "+f"(acc1[46]) :: "memory");
    asm volatile("" : "+f"(acc0[47]), "+f"(acc1[47]) :: "memory");
    asm volatile("" : "+f"(acc0[48]), "+f"(acc1[48]) :: "memory");
    asm volatile("" : "+f"(acc0[49]), "+f"(acc1[49]) :: "memory");
    asm volatile("" : "+f"(acc0[50]), "+f"(acc1[50]) :: "memory");
    asm volatile("" : "+f"(acc0[51]), "+f"(acc1[51]) :: "memory");
    asm volatile("" : "+f"(acc0[52]), "+f"(acc1[52]) :: "memory");
    asm volatile("" : "+f"(acc0[53]), "+f"(acc1[53]) :: "memory");
    asm volatile("" : "+f"(acc0[54]), "+f"(acc1[54]) :: "memory");
    asm volatile("" : "+f"(acc0[55]), "+f"(acc1[55]) :: "memory");
    asm volatile("" : "+f"(acc0[56]), "+f"(acc1[56]) :: "memory");
    asm volatile("" : "+f"(acc0[57]), "+f"(acc1[57]) :: "memory");
    asm volatile("" : "+f"(acc0[58]), "+f"(acc1[58]) :: "memory");
    asm volatile("" : "+f"(acc0[59]), "+f"(acc1[59]) :: "memory");
    asm volatile("" : "+f"(acc0[60]), "+f"(acc1[60]) :: "memory");
    asm volatile("" : "+f"(acc0[61]), "+f"(acc1[61]) :: "memory");
    asm volatile("" : "+f"(acc0[62]), "+f"(acc1[62]) :: "memory");
    asm volatile("" : "+f"(acc0[63]), "+f"(acc1[63]) :: "memory");
    sw_arrive_if(mbar_base + 24 + ((it + SW_T_STAGES - 1) % SW_T_STAGES) * 8, lane == 0u);
    // epilogue: element (W row 16 w + g + 8 rg of block b, token 8 j + 2 t + c) -> D[m0 + token][I col]
    const unsigned int g = lane >> 2, t = lane & 3u;
    const unsigned int cb = n_sel * 128u + math_wg_idx * 64u + warp_in_wg * 16u;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
#pragma unroll
        for (int rg = 0; rg < 2; ++rg) {
            // sealed fc1_gated_clamp_signal: g and u clamped to [-clamp_limit, clamp_limit] (max.f32, then fminf)
            const float g0 = fminf(max_noftz(acc0[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float g1 = fminf(max_noftz(acc0[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            const float u0 = fminf(max_noftz(acc1[4 * j + 2 * rg + 0], -clamp_limit), clamp_limit);
            const float u1 = fminf(max_noftz(acc1[4 * j + 2 * rg + 1], -clamp_limit), clamp_limit);
            float _exp2_0 = approx_exp2(g0 * -1.4426950408889634f);
            float _rcp_0 = approx_rcp(_exp2_0 + 1.0f);
            float _exp2_1 = approx_exp2(g1 * -1.4426950408889634f);
            float _rcp_1 = approx_rcp(_exp2_1 + 1.0f);
            const uint32_t pk = sw_bf2(g0 * _rcp_0 * u0, g1 * _rcp_1 * u1);
            const uint32_t tr = sw_movt(pk);
            const unsigned int row = m0 + 8u * j + g;
            if (row < row_lim) {
                *reinterpret_cast<uint32_t*>(D + ((size_t)row * n_out + cb + 8u * rg + 2u * t)) = tr;
            }
        }
    }
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_sm90_bf16_megamoe_fc1_gated_clamp_signal_swap(unsigned int num_experts, unsigned int shape_n, unsigned int shape_k, float clamp_limit, const __grid_constant__ CUtensorMap A, const uint8_t* __restrict__ Wc, unsigned int wcx_stage_bytes, long long* __restrict__ offsets, __nv_bfloat16* __restrict__ D, unsigned int* __restrict__ sched)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    // barriers: full_t[3] @0, empty_t[3] @24, full_c[SW_C_SLOTS] @48, empty_c[SW_C_SLOTS] @96 (room for 6), early-start @144
    static_assert(SW_C_SLOTS <= 6, "barrier map holds 6 compressed slots");
    const int mbar_base = smem + SMEM_MBAR_OFF;
    #define full_t_addr (mbar_base + 0)
    #define empty_t_addr (mbar_base + 24)
    #define full_c_addr (mbar_base + 48)
    #define empty_c_addr (mbar_base + 96)
    const int fe_stored_addr = smem + SMEM_MBAR_OFF + 144;
    const int fe_drained_addr = smem + SMEM_MBAR_OFF + 160;
    volatile unsigned int* fe_sig_info = reinterpret_cast<volatile unsigned int*>(smem_raw + SMEM_MBAR_OFF + 192);

    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    if (warp == 0) {
        if (elect_sync()) {
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
        }
    }
    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            for (int st = 0; st < SW_T_STAGES; ++st) {
                mbarrier_init(full_t_addr + st * 8, 1);
                mbarrier_init(empty_t_addr + st * 8, 8);
            }
            for (int st = 0; st < SW_C_SLOTS; ++st) {
                mbarrier_init(full_c_addr + st * 8, 1);
                mbarrier_init(empty_c_addr + st * 8, 8);
            }
            mbarrier_init(fe_stored_addr, 256);
            mbarrier_init(fe_stored_addr + 8, 256);
            mbarrier_init(fe_drained_addr, 1);
            mbarrier_init(fe_drained_addr + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    unsigned int cid = (unsigned int)cluster_id;
    unsigned int ncl = (unsigned int)num_clusters;
    unsigned int rank = (unsigned int)cta_rank;
    unsigned int n_tiles = shape_n / 256;
    unsigned int n_pairs = (n_tiles + 1) / 2;
    unsigned int num_k_stages = shape_k / 64;
    unsigned int n_out = shape_n / 2;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tid == 0) {
        // FC1's own griddepcontrol.wait has returned (compact's offsets are visible); FC2 acquires this word
        // before its first offsets read (FC2 runs without griddepcontrol.wait).
        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(sched + 2), "r"(1u) : "memory");
    }
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    if (static_cast<uint32_t>(warp) >= 8u) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
        if (warp == 8) {
            // token stages of this CTA's tile (no multicast): N / 16 TMA boxes {64, 16} per k-step
            if (elect_sync()) {
                unsigned int it_p = 0;
                unsigned int tile_base_p = 0;
                for (unsigned int e_p = 0; e_p < num_experts; e_p++) {
                    unsigned int row0_p = (unsigned int)offsets[e_p];
                    unsigned int row1_p = (unsigned int)offsets[e_p + 1];
                    unsigned int m_tiles_p = (row1_p - row0_p + 128 - 1) / 128;
                    unsigned int n_tiles_e_p = m_tiles_p * n_pairs;
                    unsigned int rem_p = (cid + ncl - tile_base_p % ncl) % ncl;
                    unsigned int _max_0 = ((n_tiles_e_p) > (rem_p) ? (n_tiles_e_p) : (rem_p));
                    unsigned int my_tiles_p = (_max_0 - rem_p + ncl - 1) / ncl;
                    for (unsigned int j_p = 0; j_p < my_tiles_p; j_p++) {
                        unsigned int t_p = rem_p + j_p * ncl;
                        unsigned int m0_p = row0_p + (t_p / n_pairs) * 128;
                        const unsigned int rv_p = (row1_p - m0_p) < 128u ? (row1_p - m0_p) : 128u;
                        const unsigned int nb_p = rv_p <= 16u ? 1u : (rv_p <= 64u ? 4u : 8u);
                        for (unsigned int k_p = 0; k_p < num_k_stages; k_p++) {
                            unsigned int stage_p = it_p % SW_T_STAGES;
                            mbarrier_wait_relaxed_hint(empty_t_addr + (stage_p) * 8, it_p / SW_T_STAGES + 1 & 1, 10000000);
                            mbarrier_arrive_expect_tx(full_t_addr + (stage_p) * 8, nb_p * 2048u);
                            for (unsigned int b_p = 0; b_p < nb_p; b_p++) {
                                const uint64_t _tma_a_desc = (uint64_t)((&A));
                                const uint32_t _tma_a_mbar = (uint32_t)(full_t_addr + (stage_p) * 8);
                                const uint32_t _tma_a_dst = (uint32_t)(smem + stage_p * SW_T_STAGE_BYTES + b_p * 2048u);
                                const int32_t _tma_a_c0 = (int32_t)(k_p * 64);
                                const int32_t _tma_a_c1 = (int32_t)(m0_p + 16u * b_p);
                                const uint64_t _tma_a_cache = (uint64_t)(0x1000000000000000ULL);
                                asm volatile(
                                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%3, %4}], [%2], %5;"
                                    :: "r"(_tma_a_dst), "l"(_tma_a_desc), "r"(_tma_a_mbar),
                                       "r"(_tma_a_c0), "r"(_tma_a_c1), "l"(_tma_a_cache) : "memory");
                            }
                            it_p = it_p + 1;
                        }
                    }
                    tile_base_p = tile_base_p + n_tiles_e_p;
                }
            }
        }
        else if (warp == 10) {
            // compressed W stages of this CTA's n-tile (2u + rank) into the SW_C_SLOTS-slot ring (one bulk copy each)
            if (elect_sync()) {
                unsigned int it_w = 0;
                unsigned int tile_base_w = 0;
                for (unsigned int e_w = 0; e_w < num_experts; e_w++) {
                    unsigned int row0_w = (unsigned int)offsets[e_w];
                    unsigned int row1_w = (unsigned int)offsets[e_w + 1];
                    unsigned int m_tiles_w = (row1_w - row0_w + 128 - 1) / 128;
                    unsigned int n_tiles_e_w = m_tiles_w * n_pairs;
                    unsigned int rem_w = (cid + ncl - tile_base_w % ncl) % ncl;
                    unsigned int _max_w = ((n_tiles_e_w) > (rem_w) ? (n_tiles_e_w) : (rem_w));
                    unsigned int my_tiles_w = (_max_w - rem_w + ncl - 1) / ncl;
                    for (unsigned int j_w = 0; j_w < my_tiles_w; j_w++) {
                        unsigned int t_w = rem_w + j_w * ncl;
                        unsigned int n_sel_w = (t_w % n_pairs) * 2 + rank;
                        // odd N: the peer of the last pair has no n-tile (its stores are masked); stream a valid one
                        unsigned int n_ld_w = n_sel_w < n_tiles ? n_sel_w : n_tiles - 1;
                        const uint8_t* src_w = Wc + ((unsigned long long)(e_w * n_tiles + n_ld_w) * num_k_stages) * wcx_stage_bytes;
                        for (unsigned int k_w = 0; k_w < num_k_stages; k_w++) {
                            unsigned int slot_w = it_w % SW_C_SLOTS;
                            mbarrier_wait_relaxed_hint(empty_c_addr + slot_w * 8, it_w / SW_C_SLOTS + 1 & 1, 10000000);
                            mbarrier_arrive_expect_tx(full_c_addr + slot_w * 8, wcx_stage_bytes);
                            asm volatile(
                                "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                                :: "r"((uint32_t)(smem + SW_RING_OFF + slot_w * WCX_SLOT_STRIDE)),
                                   "l"(src_w + (unsigned long long)k_w * wcx_stage_bytes), "r"(wcx_stage_bytes),
                                   "r"((uint32_t)(full_c_addr + slot_w * 8)) : "memory");
                            it_w = it_w + 1;
                        }
                    }
                    tile_base_w = tile_base_w + n_tiles_e_w;
                }
            }
        }
        else if (warp == 9) {
            // one release per stored FC1 CTA-tile on its m-tile slot, then the CTA count.
            if (lane == 0) {
                unsigned int j_s = 0;
                while (true) {
                    const unsigned int b_s = j_s & 1u;
                    fe_wait_acq(fe_stored_addr + b_s * 8, (j_s >> 1) & 1u);
                    const unsigned int slot_s = fe_sig_info[b_s];
                    if (slot_s == 0xFFFFFFFFu) {
                        break;
                    }
                    fe_signal_add(sched + FE_READY_OFF + slot_s);
                    mbarrier_arrive(fe_drained_addr + b_s * 8);
                    j_s = j_s + 1u;
                }
                asm volatile("red.release.gpu.global.add.u32 [%0], 1;" :: "l"(sched + 3) : "memory");
            }
        }
    }
    else {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
        unsigned int math_wg_idx = make_warp_uniform(warp_group_idx);
        unsigned int it_c = 0;
        unsigned int tile_base_c = 0;
        unsigned int mtile_base_c = 0;
        unsigned int j_sig = 0;
        unsigned int warp_in_wg = (unsigned int)warp % 4;
        for (unsigned int e_c = 0; e_c < num_experts; e_c++) {
            unsigned int row0_c = (unsigned int)offsets[e_c];
            unsigned int row1_c = (unsigned int)offsets[e_c + 1];
            unsigned int cnt_c = row1_c - row0_c;
            unsigned int m_tiles_c = (cnt_c + 128 - 1) / 128;
            unsigned int n_tiles_e_c = m_tiles_c * n_pairs;
            unsigned int rem_c = (cid + ncl - tile_base_c % ncl) % ncl;
            unsigned int _max_1 = ((n_tiles_e_c) > (rem_c) ? (n_tiles_e_c) : (rem_c));
            unsigned int my_tiles_c = (_max_1 - rem_c + ncl - 1) / ncl;
            for (unsigned int j_c = 0; j_c < my_tiles_c; j_c++) {
                unsigned int t_c = rem_c + j_c * ncl;
                unsigned int mt_c = t_c / n_pairs;
                unsigned int u_c = t_c - mt_c * n_pairs;
                unsigned int n_sel_c = u_c * 2 + rank;
                unsigned int m0_c = row0_c + mt_c * 128;
                unsigned int zero_c = 0;
                unsigned int row_lim_c = ((n_sel_c < n_tiles) ? row1_c : zero_c);
                const unsigned int rv_c = (row1_c - m0_c) < 128u ? (row1_c - m0_c) : 128u;
                if (rv_c <= 16u) {
                    sw_tile_n16(smem_raw, smem, it_c, num_k_stages, math_wg_idx, warp_in_wg, lane, m0_c, n_sel_c, row_lim_c, n_out, D, mbar_base, clamp_limit);
                } else if (rv_c <= 64u) {
                    sw_tile_n64(smem_raw, smem, it_c, num_k_stages, math_wg_idx, warp_in_wg, lane, m0_c, n_sel_c, row_lim_c, n_out, D, mbar_base, clamp_limit);
                } else {
                    sw_tile_n128(smem_raw, smem, it_c, num_k_stages, math_wg_idx, warp_in_wg, lane, m0_c, n_sel_c, row_lim_c, n_out, D, mbar_base, clamp_limit);
                }
                // this CTA-tile's h stores are issued -> warp 9 releases its m-tile slot
                {
                    const unsigned int b_sig = j_sig & 1u;
                    if (tid == 0) {
                        if (j_sig >= 2u) {
                            fe_wait_acq(fe_drained_addr + b_sig * 8, ((j_sig >> 1) - 1u) & 1u);
                        }
                        fe_sig_info[b_sig] = mtile_base_c + ((m0_c - row0_c) >> 7);
                    }
                    mbarrier_arrive(fe_stored_addr + b_sig * 8);
                }
                j_sig = j_sig + 1u;
            }
            mtile_base_c = mtile_base_c + m_tiles_c;
            tile_base_c = tile_base_c + n_tiles_e_c;
        }
        // no more tiles -> warp 9 publishes the CTA count and exits
        {
            const unsigned int b_sig = j_sig & 1u;
            if (tid == 0) {
                if (j_sig >= 2u) {
                    fe_wait_acq(fe_drained_addr + b_sig * 8, ((j_sig >> 1) - 1u) & 1u);
                }
                fe_sig_info[b_sig] = 0xFFFFFFFFu;
            }
            mbarrier_arrive(fe_stored_addr + b_sig * 8);
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
}

} // extern "C"
