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

// MSA stage-1 block-score top-k indexer for SM107 (Rubin).
// Stage A: tcgen05 BF16 GEMM producing per-(query, key-block) MAX scores.
// Stage B: per-query top-15 selection among strictly-prior blocks + forced own block.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cfloat>
#include <mutex>
#include <tvm/ffi/tvm_ffi.h>

#include <tvm/ffi/extra/c_env_api.h>

namespace ffi = tvm::ffi;

namespace c64_regbuf_impl {

#define KBLOCK 128
#define DHEAD 128
#define TOPK 16

// Two stream-ordered scratch slots make the staged score/select path
// reentrant.  The first stream owns slot 0 permanently, so the common path
// needs no device-side acquire/release.  Every other stream shares guarded
// slot 1; a one-CTA acquire serializes only those overflow calls.
#define SCORE_FLOATS (48u * 1024u * 1024u)
#define WORKSPACE_SLOTS 2
__device__ __align__(256) float g_scores_primary[SCORE_FLOATS];
__device__ __align__(256) float g_scores_overflow[SCORE_FLOATS];
__device__ int g_overflow_lock;
__device__ int g_overflow_done;

__global__ void acquire_workspace() {
    if (threadIdx.x == 0) {
        while (atomicCAS(&g_overflow_lock, 0, 1) != 0)
            __nanosleep(64);
        g_overflow_done = 0;
        __threadfence();
    }
}

__device__ __forceinline__ void finish_workspace() {
    __syncthreads();
    if (threadIdx.x == 0) {
        const int old = atomicAdd(&g_overflow_done, 1);
        if (old == (int)gridDim.x - 1) {
            g_overflow_done = 0;
            __threadfence();
            atomicExch(&g_overflow_lock, 0);
        }
    }
}

template <bool GUARDED>
__device__ __forceinline__ float* score_workspace() {
    if constexpr (GUARDED)
        return g_scores_overflow;
    else
        return g_scores_primary;
}

// ---------------- device helpers ----------------
__device__ __forceinline__ uint32_t elect_one() {
    uint32_t pred = 0;
    asm volatile(
        "{\n\t.reg .pred %%px;\n\t"
        "elect.sync _|%%px, %1;\n\t"
        "@%%px mov.s32 %0, 1;\n\t}"
        : "+r"(pred) : "r"(0xFFFFFFFFu));
    return pred;
}

__device__ __forceinline__ constexpr uint64_t desc_enc(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}

__device__ __forceinline__ uint64_t smem_desc(uint32_t addr) {
    uint64_t d = 0;
    d |= desc_enc(addr);
    d |= desc_enc(1024) << 32ULL;   // SBO = 8 rows * 128 B
    d |= 1ULL << 46ULL;             // gen5 flag
    d |= 2ULL << 61ULL;             // 128B swizzle
    return d;
}

__device__ __forceinline__ void mbar_init(uint32_t a, int cnt) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(a), "r"(cnt));
}

__device__ __forceinline__ void mbar_wait(uint32_t a, int phase) {
    asm volatile(
        "{\n\t.reg .pred P1;\n\t"
        "LAB_W%=:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE%=;\n\t"
        "bra.uni LAB_W%=;\n\t"
        "DONE%=:\n\t}"
        :: "r"(a), "r"(phase), "r"(0x989680u));
}

__device__ __forceinline__ void mbar_expect(uint32_t a, int bytes) {
    asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                 :: "r"(a), "r"(bytes) : "memory");
}

__device__ __forceinline__ void tma_2d(uint32_t smem, const CUtensorMap* tm,
                                       int c, int r, uint32_t bar) {
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
                 " [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem), "l"(tm), "r"(c), "r"(r), "r"(bar) : "memory");
}

__device__ __forceinline__ void mma_f16(uint32_t taddr, uint64_t ad, uint64_t bd,
                                        uint32_t idesc, int acc) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t}"
        :: "r"(taddr), "l"(ad), "l"(bd), "r"(idesc), "r"(acc) : "memory");
}

#define TC_TMP128 \
  "t0,t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15," \
  "t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,t31," \
  "t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47," \
  "t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,t61,t62,t63," \
  "t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79," \
  "t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90,t91,t92,t93,t94,t95," \
  "t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111," \
  "t112,t113,t114,t115,t116,t117,t118,t119,t120,t121,t122,t123,t124,t125,t126,t127"

// Reduce an entire 128-key block in one TMEM statistics instruction.
__device__ __forceinline__ float tmem_ld_max128(uint32_t a) {
    float red;
    asm volatile(
        "{ .reg .f32 t<128>; "
        "tcgen05.ld.red.sync.aligned.32x32b.x128.max.f32 {" TC_TMP128
        "}, %0, [%1]; }"
        : "=f"(red) : "r"(a));
    return red;
}

// One naturally-aligned 256-bit store replaces the two 128-bit stores that
// each requested the same 32-byte sector.  NCU reports stage A's excessive
// global-store sectors dropping from 1.47M to 0.34M on the long-context
// workload.
__device__ __forceinline__ void st_global_v8f32(float* p, const float* v) {
    asm volatile("st.global.v8.f32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
                 :: "l"(p), "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]),
                    "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]) : "memory");
}

// ---------------- stage A : block MAX scores ----------------
// Tile: BM=256 queries (two M=128 A-tiles resident in SMEM) x BN=256 keys (2 key blocks).
// Two TMEM accumulators (256 cols each).  MMA(acc0) overlaps epilogue(acc1) and vice versa.
#define NAT 2
#define BMT 128
#define BM (NAT * BMT)
#define BN 256
#define NBLKT 2
#define STAGES 2
#define NTHR 256
#define NTHR_P 288
#define ASZ (BMT * DHEAD * 2)
#define BSZ (BN * DHEAD * 2)
#define GRP 8
static_assert(STAGES >= 2, "named-barrier pipeline requires distinct refill and active B stages");

template <bool GUARDED>
__global__ __launch_bounds__(NTHR, 1) void score_kernel(
    const __grid_constant__ CUtensorMap qmap,
    const __grid_constant__ CUtensorMap kmap,
    const int* __restrict__ cu,
    const int* __restrict__ kvbase,
    const int* __restrict__ prefix,
    int B, int pgstride, int nsplit, int fixed_chunk,
    int ntiles, int reduce_tiles, int schedule_mode)
{
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;

    // Schedule high-position query tiles first: they score the most prior
    // blocks and otherwise become a long tail after the bulk of the grid.
    int tile, split, tile_splits;
    if (schedule_mode == 1) {
        // W1 has 211 real fixed-22 tasks.  Flatten only those tasks so the
        // score grid occupies 211 of 212 SMs in one wave.  The ranges below
        // exactly cover tiles 63..8 in descending order with ceil(nb/22)
        // splits per tile: 54 + 55 + 44 + 33 + 22 + 3 tasks.
        const int task = (int)blockIdx.x;
        if (task < 54) {
            tile = 63 - task / 6;
            split = task % 6;
        } else if (task < 109) {
            const int u = task - 54;
            tile = 54 - u / 5;
            split = u % 5;
        } else if (task < 153) {
            const int u = task - 109;
            tile = 43 - u / 4;
            split = u % 4;
        } else if (task < 186) {
            const int u = task - 153;
            tile = 32 - u / 3;
            split = u % 3;
        } else if (task < 208) {
            const int u = task - 186;
            tile = 21 - u / 2;
            split = u % 2;
        } else {
            tile = 10 - (task - 208);
            split = 0;
        }
        tile_splits = nsplit;
    } else if (schedule_mode == 2) {
        // W5: omit only the empty tenth split on tiles 24..31.  Keep
        // tile_splits=10 so every surviving key range matches the original.
        const int task = (int)blockIdx.x;
        if (task < 320) {
            tile = 63 - task / 10;
            split = task % 10;
        } else if (task < 392) {
            const int u = task - 320;
            tile = 31 - u / 9;
            split = u % 9;
        } else {
            const int u = task - 392;
            tile = 23 - u / 10;
            split = u % 10;
        }
        tile_splits = 10;
    } else if (schedule_mode == 3) {
        // W6: each request has eight empty tenth splits on local tiles
        // 24..31.  Preserve reverse request/tile order and all active ranges.
        const int task = (int)blockIdx.x;
        const int reverse_request = task / 312;
        const int u = task - reverse_request * 312;
        const int request = 1 - reverse_request;
        int local_tile;
        if (u < 72) {
            local_tile = 31 - u / 9;
            split = u % 9;
        } else {
            const int v = u - 72;
            local_tile = 23 - v / 10;
            split = v % 10;
        }
        tile = request * 32 + local_tile;
        tile_splits = 10;
    } else if (schedule_mode == 4) {
        // W7: ntiles overcounts the exact per-request tile sum by three.
        // Flatten the 64 real query tiles and omit only those 30 no-op CTAs.
        const int task = (int)blockIdx.x;
        tile = 63 - task / 10;
        split = task % 10;
        tile_splits = 10;
    } else if (reduce_tiles) {
        // Flatten the launch and remove one split from a few low-query tiles.
        // W5 maps 60 tiles x 10 splits plus 4 tiles x 9 splits to 636 CTAs.
        const int task = (int)blockIdx.x;
        const int normal_tiles = ntiles - reduce_tiles;
        const int normal_tasks = normal_tiles * nsplit;
        int logical_y;
        if (task < normal_tasks) {
            logical_y = task / nsplit;
            split = task - logical_y * nsplit;
            tile_splits = nsplit;
        } else {
            const int reduced_splits = nsplit - 1;
            const int u = task - normal_tasks;
            logical_y = normal_tiles + u / reduced_splits;
            split = u - (logical_y - normal_tiles) * reduced_splits;
            tile_splits = reduced_splits;
        }
        tile = ntiles - 1 - logical_y;
    } else {
        const int logical_y = (int)blockIdx.y;
        split = (int)blockIdx.x;
        tile_splits = nsplit;
        tile = ntiles - 1 - logical_y;
    }
    int r = -1, qbase = 0, nqt = 0;
    {
        int acc = 0;
        for (int i = 0; i < B; ++i) {
            int q0 = cu[i], q1 = cu[i + 1];
            int nt = (q1 - q0 + BM - 1) / BM;
            if (tile < acc + nt) {
                r = i;
                qbase = q0 + (tile - acc) * BM;
                nqt = min(BM, q1 - qbase);
                break;
            }
            acc += nt;
        }
    }
    if (r < 0 || nqt <= 0) return;

    const int qoff = qbase - cu[r];
    const int nb = (prefix[r] + qoff + nqt - 1) >> 7;      // candidate blocks [0, nb)
    // Stage B never reads scores for own<=15 because all prior blocks are
    // selected.  If the whole query tile is in that regime, skip scoring it.
    if (nb <= 15) return;

    int chunk = fixed_chunk ? fixed_chunk : (nb + tile_splits - 1) / tile_splits;
    if (!fixed_chunk)
        chunk = (chunk + GRP - 1) & ~(GRP - 1);
    const int b0 = split * chunk;
    if (b0 >= nb) return;
    const int b1 = min(nb, b0 + chunk);
    const int nblk = b1 - b0;                               // key blocks for this CTA
    const int nit  = (nblk + NBLKT - 1) / NBLKT;            // B-tile iterations
    const int krow0 = kvbase[r] * KBLOCK;

    extern __shared__ __align__(1024) char smem_raw[];
    const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const uint32_t sB = sA + NAT * ASZ;
    __shared__ __align__(8) uint64_t bars[STAGES + 3];
    __shared__ int tmem_slot[1];
    const uint32_t barB = (uint32_t)__cvta_generic_to_shared(&bars[0]);
    const uint32_t barM = barB + STAGES * 8;
    const uint32_t barA = barM + 16;

    if (warp == 0 && elect_one()) {
        for (int i = 0; i < STAGES + 3; ++i) mbar_init(barB + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp == 1) {
        const uint32_t ad = (uint32_t)__cvta_generic_to_shared(tmem_slot);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(ad), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    if (nqt < BM) {
        uint4 z = make_uint4(0, 0, 0, 0);
        for (int o = tid * 16; o < NAT * ASZ; o += NTHR * 16)
            *reinterpret_cast<uint4*>(smem_raw + o) = z;
    }
    __syncthreads();
    const uint32_t taddr = (uint32_t)tmem_slot[0];

    constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10)
                             | ((uint32_t)(BN / 8) << 17) | ((uint32_t)(BMT / 16) << 24);

    if (warp == 0 && elect_one()) {
        // Generic shared-memory zero fill above and the following TMA writes
        // access the same A tile through different proxies.
        if (nqt < BM)
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        const int active_a = min(NAT, (nqt + BMT - 1) / BMT);
        mbar_expect(barA, active_a * ASZ);
        for (int a = 0; a < NAT; ++a) {
            if (a * BMT < nqt) {
                int rows = qbase + a * BMT;
                tma_2d(sA + a * ASZ,           &qmap, 0,  rows, barA);
                tma_2d(sA + a * ASZ + ASZ / 2, &qmap, 64, rows, barA);
            }
        }
        // Keep the launch-critical prologue to A plus B stage 0.  Stage 1 is
        // issued after the first acc0 MMA launch below and overlaps its work.
        const int kr = krow0 + b0 * KBLOCK;
        mbar_expect(barB, BSZ);
        tma_2d(sB,           &kmap, 0,  kr, barB);
        tma_2d(sB + BSZ / 2, &kmap, 64, kr, barB);
    }
    mbar_wait(barA, 0);

    // Each half-CTA owns one accumulator/query tile.  Four owner warps load
    // both 128-column key blocks for their row, so the two results stay in
    // registers until a full 32-byte score sector can be stored.
    const int row   = tid & 127;
    const int fa    = tid >> 7;
    const uint32_t lane_off = (uint32_t)((row >> 5) * 32) << 16;
    const int fq   = qbase + fa * BMT + row;
    const bool fvalid = (fa * BMT + row) < nqt;
    float* const fsrow = score_workspace<GUARDED>()
                       + (size_t)(fvalid ? fq : 0) * (size_t)pgstride;
    float score_buf[GRP];

    auto do_epi = [&](int a, int it) {
        if (fa == a) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t base = taddr + (uint32_t)(a * BN) + lane_off;
            int slot = (it & 3) * NBLKT;
            score_buf[slot] = tmem_ld_max128(base);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            score_buf[slot + 1] = tmem_ld_max128(base + KBLOCK);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");

            if ((it & 3) == 3 || it == nit - 1) {
                int done = ((it & 3) + 1) * NBLKT;
                int gb = b0 + (it & ~3) * NBLKT;
                if (fvalid) {
                    if (done == GRP && gb + GRP <= b1) {
                        if (schedule_mode == 1) {
                            // fixed-22 alternates between 16-byte phases.
                            // Use the widest naturally-aligned stores in both:
                            // 2x float4 at phase 0, float2+float4+float2 at 2.
                            if ((gb & 3) == 0) {
                                *reinterpret_cast<float4*>(fsrow + gb) =
                                    make_float4(score_buf[0], score_buf[1],
                                                score_buf[2], score_buf[3]);
                                *reinterpret_cast<float4*>(fsrow + gb + 4) =
                                    make_float4(score_buf[4], score_buf[5],
                                                score_buf[6], score_buf[7]);
                            } else {
                                *reinterpret_cast<float2*>(fsrow + gb) =
                                    make_float2(score_buf[0], score_buf[1]);
                                *reinterpret_cast<float4*>(fsrow + gb + 2) =
                                    make_float4(score_buf[2], score_buf[3],
                                                score_buf[4], score_buf[5]);
                                *reinterpret_cast<float2*>(fsrow + gb + 6) =
                                    make_float2(score_buf[6], score_buf[7]);
                            }
                        } else {
                            st_global_v8f32(fsrow + gb, score_buf);
                        }
                    } else if (schedule_mode == 1 && done == 6 && gb + 6 <= b1) {
                        // fixed-22 makes gb even but alternates its 16-byte
                        // phase.  Choose the float4 position accordingly so
                        // both vector stores remain naturally aligned.
                        if ((gb & 3) == 0) {
                            *reinterpret_cast<float4*>(fsrow + gb) =
                                make_float4(score_buf[0], score_buf[1],
                                            score_buf[2], score_buf[3]);
                            *reinterpret_cast<float2*>(fsrow + gb + 4) =
                                make_float2(score_buf[4], score_buf[5]);
                        } else {
                            *reinterpret_cast<float2*>(fsrow + gb) =
                                make_float2(score_buf[0], score_buf[1]);
                            *reinterpret_cast<float4*>(fsrow + gb + 2) =
                                make_float4(score_buf[2], score_buf[3],
                                            score_buf[4], score_buf[5]);
                        }
                    } else {
                        #pragma unroll
                        for (int j = 0; j < GRP; ++j)
                            if (j < done && gb + j < b1)
                                fsrow[gb + j] = score_buf[j];
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
        }
    };

    int phM0 = 0, phM1 = 0;

    for (int i = 0; i < nit; ++i) {
        const int buf = i % STAGES;
        if (warp == 0) {
            mbar_wait(barB + buf * 8, (i / STAGES) & 1);
        }
        if (warp == 0 && elect_one()) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t acc = taddr;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk)
                mma_f16(acc, smem_desc(sA + (kk >> 2) * (ASZ / 2) + (kk & 3) * 32),
                        smem_desc(sB + buf * BSZ + (kk >> 2) * (BSZ / 2) + (kk & 3) * 32),
                        idesc, kk ? 1 : 0);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(barM) : "memory");
            if (i == 0 && nit > 1) {
                const int kr = krow0 + (b0 + NBLKT) * KBLOCK;
                mbar_expect(barB + 8, BSZ);
                tma_2d(sB + BSZ,           &kmap, 0,  kr, barB + 8);
                tma_2d(sB + BSZ + BSZ / 2, &kmap, 64, kr, barB + 8);
            }
        }
        if (i >= 1 && (warp == 0 || warp >= 4)) {
            mbar_wait(barM + 8, phM1);
            phM1 ^= 1;
            if (warp == 0 && elect_one()) {
                int nxt = i - 1 + STAGES;
                if (nxt < nit) {
                    int bf = (i - 1) % STAGES;
                    int kr = krow0 + (b0 + nxt * NBLKT) * KBLOCK;
                    mbar_expect(barB + bf * 8, BSZ);
                    tma_2d(sB + bf * BSZ,           &kmap, 0,  kr, barB + bf * 8);
                    tma_2d(sB + bf * BSZ + BSZ / 2, &kmap, 64, kr, barB + bf * 8);
                }
            }
            // The replacement TMA targets the now-consumed alternate B
            // stage, so issue it before the acc1 epilogue and overlap the
            // transfer with the two x128 TMEM reductions.
            do_epi(1, i - 1);
            asm volatile("bar.sync 2, 160;" ::: "memory");
        }
        if (warp == 0 && elect_one()) {
            asm volatile("tcgen05.fence::after_thread_sync;");
            uint32_t acc = taddr + (uint32_t)BN;
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk)
                mma_f16(acc, smem_desc(sA + ASZ + (kk >> 2) * (ASZ / 2) + (kk & 3) * 32),
                        smem_desc(sB + buf * BSZ + (kk >> 2) * (BSZ / 2) + (kk & 3) * 32),
                        idesc, kk ? 1 : 0);
            asm volatile(
                "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(barM + 8) : "memory");
        }
        if (warp < 4) {
            mbar_wait(barM, phM0);
            phM0 ^= 1;
            do_epi(0, i);
            asm volatile("bar.sync 1, 128;" ::: "memory");
        }
    }
    if (warp >= 4) {
        mbar_wait(barM + 8, phM1);
        phM1 ^= 1;
        do_epi(1, nit - 1);
    }
    __syncthreads();
    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(512));
    }
}


// ---------------- W1/W5/W6 persistent balanced scorer ----------------
// Exactly 212 resident workers split the global list of 2-block scoring units
// evenly.  W1 uses a precomputed 8/9/10-unit critical-path partition.  W5/W6
// use metadata-gated equal-load DP partitions; other metadata retains the
// arithmetic split.  W7 and all generic shapes retain score_kernel above.
// Per-request geometry.  Work is counted in UNITS of NBLKT=2 key blocks, so a
// CTA boundary always lands on an even block index (float4 score stores stay
// 16 B aligned) while the global split stays balanced to within one unit.
struct PersistentRInfo {
    int q0, nq, nt, n0, ts, nbl, tend, m0;
    long long Sfull, W;
};

// W1 has 2044 two-block units across scored query tiles of sizes 9..64.
// This store-aware 8/9/10 partition retains the ten-unit critical path while
// minimizing partial score-store boundaries.  It needs 234 A-tile loads. Pack
// start tile, intra-tile unit offset and span in one uint16:
//   bits 13:8 = tile-8, bits 7:2 = offset, bits 1:0 = span-8.
// The table is derived exactly from the former 213-entry boundary table.
__device__ __constant__ uint16_t kW1PersistentWork234[212] = {
    0x0001, 0x0102, 0x0201, 0x0225, 0x031d, 0x0411, 0x0502, 0x0529, 0x0616, 0x0700, 0x0720, 0x0800,
    0x0821, 0x0900, 0x0922, 0x0a01, 0x0a26, 0x0b02, 0x0b2a, 0x0c00, 0x0c22, 0x0c49, 0x0d1a, 0x0d41,
    0x0e0e, 0x0e36, 0x0f02, 0x0f2a, 0x0f51, 0x1016, 0x103e, 0x1100, 0x1120, 0x1142, 0x1200, 0x1221,
    0x1246, 0x1300, 0x1322, 0x134a, 0x1402, 0x142a, 0x1451, 0x1502, 0x152a, 0x1552, 0x1602, 0x162a,
    0x1651, 0x1675, 0x171e, 0x1746, 0x176d, 0x1812, 0x183a, 0x1861, 0x1902, 0x192a, 0x1952, 0x1979,
    0x1a16, 0x1a3e, 0x1a66, 0x1b00, 0x1b20, 0x1b42, 0x1b6a, 0x1c00, 0x1c22, 0x1c4a, 0x1c71, 0x1d00,
    0x1d22, 0x1d4a, 0x1d72, 0x1e02, 0x1e2a, 0x1e51, 0x1e76, 0x1f02, 0x1f2a, 0x1f52, 0x1f7a, 0x2000,
    0x2022, 0x204a, 0x2072, 0x2099, 0x211a, 0x2142, 0x216a, 0x2191, 0x220e, 0x2236, 0x225e, 0x2286,
    0x2302, 0x232a, 0x2352, 0x237a, 0x23a1, 0x2416, 0x243e, 0x2466, 0x248e, 0x2500, 0x2520, 0x2542,
    0x256a, 0x2592, 0x2600, 0x2622, 0x264a, 0x2671, 0x2696, 0x2700, 0x2722, 0x274a, 0x2772, 0x279a,
    0x2802, 0x282a, 0x2852, 0x287a, 0x28a1, 0x2902, 0x292a, 0x2952, 0x297a, 0x29a2, 0x2a02, 0x2a2a,
    0x2a52, 0x2a7a, 0x2aa1, 0x2ac5, 0x2b1e, 0x2b46, 0x2b6e, 0x2b96, 0x2bbd, 0x2c12, 0x2c3a, 0x2c62,
    0x2c8a, 0x2cb1, 0x2d02, 0x2d2a, 0x2d52, 0x2d7a, 0x2da2, 0x2dc9, 0x2e16, 0x2e3e, 0x2e66, 0x2e8e,
    0x2eb6, 0x2f02, 0x2f2a, 0x2f52, 0x2f7a, 0x2fa1, 0x2fc5, 0x300a, 0x3032, 0x305a, 0x3082, 0x30aa,
    0x30d1, 0x3112, 0x313a, 0x3162, 0x318a, 0x31b2, 0x31d9, 0x3216, 0x323e, 0x3266, 0x328e, 0x32b6,
    0x32dd, 0x3316, 0x333e, 0x3366, 0x338e, 0x33b6, 0x33dd, 0x3412, 0x343a, 0x3462, 0x348a, 0x34b2,
    0x34d9, 0x350a, 0x3532, 0x355a, 0x3582, 0x35aa, 0x35d2, 0x3600, 0x3622, 0x364a, 0x3672, 0x369a,
    0x36c2, 0x36e9, 0x3712, 0x373a, 0x3762, 0x378a, 0x37b2, 0x37da
};

// Price every query-tile entry as roughly 3.2 pair units so CTAs that restart
// the A-TMA/pipeline receive slightly less arithmetic work.
// They are selected only for the two exact captured long-context work lists.
__device__ __constant__ uint16_t kPersistentBoundsW5[213] = {
    0, 88, 176, 261, 349, 437, 522, 610, 698, 783, 871, 959,
    1044, 1132, 1220, 1305, 1393, 1481, 1566, 1654, 1742, 1827, 1915, 2003,
    2088, 2176, 2264, 2349, 2437, 2525, 2613, 2698, 2786, 2874, 2959, 3047,
    3135, 3220, 3308, 3396, 3481, 3569, 3657, 3742, 3830, 3918, 4003, 4091,
    4179, 4264, 4352, 4440, 4525, 4613, 4701, 4786, 4874, 4962, 5050, 5135,
    5223, 5311, 5396, 5484, 5572, 5657, 5745, 5833, 5918, 6006, 6094, 6179,
    6267, 6355, 6443, 6528, 6616, 6704, 6789, 6877, 6965, 7050, 7138, 7226,
    7311, 7399, 7487, 7574, 7662, 7750, 7838, 7923, 8011, 8099, 8184, 8272,
    8360, 8445, 8533, 8621, 8709, 8794, 8882, 8970, 9055, 9143, 9231, 9316,
    9404, 9492, 9580, 9665, 9753, 9841, 9926, 10014, 10102, 10187, 10275, 10363,
    10451, 10536, 10624, 10712, 10797, 10885, 10973, 11060, 11148, 11236, 11324, 11409,
    11497, 11585, 11670, 11758, 11846, 11934, 12019, 12107, 12195, 12280, 12368, 12456,
    12544, 12629, 12717, 12805, 12890, 12978, 13066, 13154, 13239, 13327, 13415, 13500,
    13588, 13676, 13764, 13849, 13937, 14025, 14110, 14198, 14286, 14374, 14459, 14547,
    14635, 14720, 14808, 14896, 14984, 15069, 15157, 15245, 15330, 15418, 15506, 15594,
    15679, 15767, 15855, 15940, 16028, 16116, 16204, 16289, 16377, 16465, 16553, 16638,
    16726, 16814, 16899, 16987, 17075, 17163, 17248, 17336, 17424, 17509, 17597, 17685,
    17773, 17858, 17946, 18034, 18122, 18207, 18295, 18383, 18464
};

__device__ __constant__ uint16_t kPersistentBoundsW6[213] = {
    0, 42, 84, 168, 252, 332, 416, 500, 580, 664, 748, 828,
    912, 996, 1076, 1160, 1244, 1324, 1408, 1492, 1572, 1656, 1740, 1820,
    1904, 1988, 2072, 2152, 2236, 2320, 2400, 2484, 2568, 2648, 2732, 2816,
    2896, 2980, 3064, 3148, 3228, 3312, 3396, 3476, 3560, 3644, 3724, 3808,
    3892, 3972, 4056, 4140, 4224, 4304, 4388, 4472, 4552, 4636, 4720, 4800,
    4884, 4968, 5052, 5132, 5216, 5300, 5380, 5464, 5548, 5628, 5712, 5796,
    5880, 5960, 6044, 6128, 6208, 6292, 6376, 6456, 6540, 6624, 6708, 6788,
    6872, 6956, 7036, 7120, 7204, 7288, 7368, 7452, 7536, 7616, 7700, 7784,
    7864, 7948, 8032, 8116, 8196, 8280, 8364, 8444, 8528, 8612, 8696, 8776,
    8860, 8944, 9024, 9108, 9192, 9272, 9356, 9440, 9520, 9604, 9688, 9768,
    9852, 9936, 10016, 10100, 10184, 10268, 10348, 10432, 10516, 10596, 10680, 10764,
    10844, 10928, 11012, 11092, 11176, 11260, 11340, 11424, 11508, 11592, 11672, 11756,
    11840, 11920, 12004, 12088, 12168, 12252, 12336, 12416, 12500, 12584, 12668, 12748,
    12832, 12916, 12996, 13080, 13164, 13244, 13328, 13412, 13496, 13576, 13660, 13744,
    13824, 13908, 13992, 14072, 14156, 14240, 14324, 14404, 14488, 14572, 14652, 14736,
    14820, 14900, 14984, 15068, 15152, 15232, 15316, 15400, 15480, 15564, 15648, 15728,
    15812, 15896, 15980, 16060, 16144, 16228, 16308, 16392, 16476, 16560, 16640, 16724,
    16808, 16888, 16972, 17056, 17140, 17220, 17304, 17388, 17440
};

__device__ __forceinline__ void persistent_req_info(const int* __restrict__ cu,
                                         const int* __restrict__ prefix,
                                         int r, PersistentRInfo& o) {
    o.q0 = cu[r];
    o.nq = cu[r + 1] - o.q0;
    if (o.nq <= 0) { o.nt = 0; o.n0 = 0; o.ts = 0; o.nbl = 0; o.tend = 0;
                     o.m0 = 0; o.Sfull = 0; o.W = 0; return; }
    const int P = prefix[r];
    o.nt   = (o.nq + BM - 1) / BM;
    o.n0   = (P + BM - 1) >> 7;              // nb of a full tile t is n0 + 2t
    o.nbl  = (P + o.nq - 1) >> 7;            // nb of the trailing (partial) tile
    o.ts   = (o.n0 > 15) ? 0 : (((15 - o.n0) >> 1) + 1);
    o.tend = o.nt - 1;
    o.m0   = (o.n0 + 1) >> 1;                // units of a full tile t: m0 + t
    long long s = 0;
    if (o.tend > o.ts) {
        long long k = o.tend - o.ts;
        s = k * (long long)o.m0 + (long long)(o.ts + o.tend - 1) * k / 2;
    }
    o.Sfull = s;
    o.W = s + ((o.nbl > 15) ? (long long)((o.nbl + 1) >> 1) : 0ll);
}

__device__ __forceinline__ long long persistent_tile_pfx(const PersistentRInfo& o, int t) {
    long long k = t - o.ts;
    return k * (long long)o.m0 + (long long)(o.ts + t - 1) * k / 2;
}

__device__ __forceinline__ uint32_t ordered_score(float v) {
    uint32_t b = __float_as_uint(v);
    if ((b << 1) == 0) b = 0;
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ uint64_t pack_ordered(uint32_t b, int j) {
    return ((uint64_t)b << 32) | (uint64_t)(0xFFFFFFFFu - (uint32_t)j);
}

__device__ __forceinline__ uint64_t pack_key(float v, int j) {
    return pack_ordered(ordered_score(v), j);
}

// ---------------- fused selection support ----------------
// A fused scorer keeps each query's top-15 in registers during the TMEM drain
// and never writes a score plane.  Query tiles split across several CTAs
// publish per-segment lists; the tile's last CTA merges them and emits sel,
// so one kernel covers scoring and selection.
#define MAXSLOT 16
#define PARTW (15u * (unsigned)BM)
#define MAXTILES_F 1024

__device__ int g_tcnt_primary[MAXTILES_F];
__device__ int g_tcnt_overflow[MAXTILES_F];

template <bool GUARDED>
__device__ __forceinline__ uint64_t* part_workspace() {
    if constexpr (GUARDED)
        return reinterpret_cast<uint64_t*>(g_scores_overflow);
    else
        return reinterpret_cast<uint64_t*>(g_scores_primary);
}

template <bool GUARDED>
__device__ __forceinline__ int* tile_counts() {
    if constexpr (GUARDED) return g_tcnt_overflow;
    else                   return g_tcnt_primary;
}

// Global unit boundaries of the captured W1 partition: identical ranges to
// the packed work table, in a form the inverse (unit -> CTA) lookup shares
// with the long-context tables.
__device__ __constant__ uint16_t kPersistentBoundsW1[213] = {
0, 9, 19, 28, 37, 46, 55, 65, 74, 84, 92, 100,
    108, 117, 125, 135, 144, 154, 164, 174, 182, 192, 201, 211,
    220, 230, 240, 250, 260, 269, 279, 289, 297, 305, 315, 323,
    332, 342, 350, 360, 370, 380, 390, 399, 409, 419, 429, 439,
    449, 458, 467, 477, 487, 496, 506, 516, 525, 535, 545, 555,
    564, 574, 584, 594, 602, 610, 620, 630, 638, 648, 658, 667,
    675, 685, 695, 705, 715, 725, 734, 744, 754, 764, 774, 784,
    792, 802, 812, 822, 831, 841, 851, 861, 870, 880, 890, 900,
    910, 920, 930, 940, 950, 959, 969, 979, 989, 999, 1007, 1015,
    1025, 1035, 1045, 1053, 1063, 1073, 1082, 1092, 1100, 1110, 1120, 1130,
    1140, 1150, 1160, 1170, 1180, 1189, 1199, 1209, 1219, 1229, 1239, 1249,
    1259, 1269, 1279, 1288, 1297, 1307, 1317, 1327, 1337, 1346, 1356, 1366,
    1376, 1386, 1395, 1405, 1415, 1425, 1435, 1445, 1454, 1464, 1474, 1484,
    1494, 1504, 1514, 1524, 1534, 1544, 1553, 1562, 1572, 1582, 1592, 1602,
    1612, 1621, 1631, 1641, 1651, 1661, 1671, 1680, 1690, 1700, 1710, 1720,
    1730, 1739, 1749, 1759, 1769, 1779, 1789, 1798, 1808, 1818, 1828, 1838,
    1848, 1857, 1867, 1877, 1887, 1897, 1907, 1917, 1925, 1935, 1945, 1955,
    1965, 1975, 1984, 1994, 2004, 2014, 2024, 2034, 2044
};

#define FLIST_DECL uint64_t L0=0,L1=0,L2=0,L3=0,L4=0,L5=0,L6=0,L7=0,L8=0,L9=0, \
                            L10=0,L11=0,L12=0,L13=0,L14=0
#define FLIST_CLR L0=0;L1=0;L2=0;L3=0;L4=0;L5=0;L6=0;L7=0;L8=0;L9=0; \
                  L10=0;L11=0;L12=0;L13=0;L14=0
#define FCE(a,b) { uint64_t _x=a, _y=b; a = _x>_y?_x:_y; b = _x>_y?_y:_x; }
#define FINSERT(x) { uint64_t _v=(x); if (_v > L14) { L14=_v; \
    FCE(L13,L14) FCE(L12,L13) FCE(L11,L12) FCE(L10,L11) FCE(L9,L10) FCE(L8,L9) \
    FCE(L7,L8) FCE(L6,L7) FCE(L5,L6) FCE(L4,L5) FCE(L3,L4) FCE(L2,L3) \
    FCE(L1,L2) FCE(L0,L1) } }

// Ascending bitonic sort of the 16 emitted block ids (constant indices only,
// so the array stays in registers).
#define SORT16_ASC(I) { \
    _Pragma("unroll") for (int _k = 2; _k <= 16; _k <<= 1) \
        _Pragma("unroll") for (int _j = _k >> 1; _j > 0; _j >>= 1) \
            _Pragma("unroll") for (int _i = 0; _i < 16; ++_i) \
                if ((_i & _j) == 0) { \
                    int _a = I[_i], _b = I[_i + _j]; \
                    bool _up = ((_i & _k) == 0); \
                    I[_i]      = _up ? min(_a,_b) : max(_a,_b); \
                    I[_i + _j] = _up ? max(_a,_b) : min(_a,_b); \
                } }

__device__ __forceinline__ void store_sel16(int* __restrict__ out, const int* I) {
    reinterpret_cast<int4*>(out)[0] = make_int4(I[0], I[1], I[2], I[3]);
    reinterpret_cast<int4*>(out)[1] = make_int4(I[4], I[5], I[6], I[7]);
    reinterpret_cast<int4*>(out)[2] = make_int4(I[8], I[9], I[10], I[11]);
    reinterpret_cast<int4*>(out)[3] = make_int4(I[12], I[13], I[14], I[15]);
}

// Inverse of the persistent partition: which CTA owns global work unit x.
// mode 0 = exact equal split, 1 = W5 table, 2 = W6 table, 3 = W1 table.
__device__ __forceinline__ int cta_of_unit(long long x, int mode, long long Wtot) {
    if (mode == 0)
        return (int)((x * 212ll + 211ll) / Wtot);
    const uint16_t* tb = (mode == 1) ? kPersistentBoundsW5
                       : (mode == 2) ? kPersistentBoundsW6 : kPersistentBoundsW1;
    int lo = 0, hi = 212;
    while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if ((long long)tb[mid] <= x) lo = mid; else hi = mid;
    }
    return lo;
}

template <bool USE_W1_BOUNDARIES, bool GUARDED, bool EPILOGUE4, bool FUSED>
__global__ __launch_bounds__(NTHR_P, 1) void persistent_score_kernel(
    const __grid_constant__ CUtensorMap qmap,
    const __grid_constant__ CUtensorMap kmap,
    const int* __restrict__ cu,
    const int* __restrict__ kvbase,
    const int* __restrict__ prefix,
    int B, int pgstride, int* __restrict__ selp)
{
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    constexpr int NCTA = 212;

    long long pos, hi;
    long long Wtot_v = 0;
    long long wbase = 0;
    int tgbase = 0;
    int pmode = 3;
    int r = 0, t = 0, off = 0;
    PersistentRInfo o;
    if constexpr (USE_W1_BOUNDARIES) {
        // This instantiation is dispatched only for the captured W1 geometry:
        // B=1, TQ=16384, prefix=0, cu={0,16384}, kvbase=0, pgstride=128.
        // Decode the exact CTA start and span without request scans or a
        // position-to-tile binary search.  The W1 launch passes sel through
        // the otherwise-unused cu argument so every CTA can fill at most ten
        // of the 2048 score-free output rows while scoring in parallel.
        const uint16_t work = kW1PersistentWork234[blockIdx.x];
        t   = 8 + (int)(work >> 8);
        off = (int)((work >> 2) & 63u);
        pos = (long long)kPersistentBoundsW1[blockIdx.x];
        hi  = pos + (long long)(8 + (int)(work & 3u));
    } else {
        long long Wtot = 0;
        for (int i = 0; i < B; ++i) {
            PersistentRInfo ri;
            persistent_req_info(cu, prefix, i, ri);
            Wtot += ri.W;
        }
        if (Wtot <= 0) return;

        Wtot_v = Wtot;
        if (B == 1 && pgstride == 640 && Wtot == 18464) {
            pmode = 1;
            pos = (long long)kPersistentBoundsW5[blockIdx.x];
            hi  = (long long)kPersistentBoundsW5[blockIdx.x + 1];
        } else if (B == 2 && pgstride == 1152 && Wtot == 17440) {
            pmode = 2;
            pos = (long long)kPersistentBoundsW6[blockIdx.x];
            hi  = (long long)kPersistentBoundsW6[blockIdx.x + 1];
        } else {
            pmode = 0;
            // Preserve the dynamic equal partition for any metadata outside
            // the two exact captured work lists, even when shapes coincide.
            pos = (Wtot * (long long)blockIdx.x) / NCTA;
            hi  = (Wtot * (long long)(blockIdx.x + 1)) / NCTA;
        }
        if (pos >= hi) return;

        long long x = pos;
        for (r = 0; r < B; ++r) {
            persistent_req_info(cu, prefix, r, o);
            if (x < o.W) break;
            x -= o.W;
            tgbase += o.nt;
        }
        if (r >= B) return;
        wbase = pos - x;
        if (x < o.Sfull) {
            int tl = o.ts, th = o.tend;
            while (th - tl > 1) {
                const int mid = (tl + th) >> 1;
                if (persistent_tile_pfx(o, mid) <= x) tl = mid;
                else th = mid;
            }
            t = tl;
            off = (int)(x - persistent_tile_pfx(o, tl));
        } else {
            t = o.tend;
            off = (int)(x - o.Sfull);
        }
    }

    // All fast non-fused dispatches have fewer than 2^31 pair units.  Keep
    // 64-bit partition arithmetic above, then narrow only the progress state
    // that survives across the MMA/epilogue body.  Fused W6 retains its
    // original 64-bit counters for merge ownership calculations.
    int hot_pos = 0, hot_hi = 0;
    if constexpr (!FUSED) {
        hot_pos = (int)pos;
        hot_hi = (int)hi;
    }

    extern __shared__ __align__(1024) char smem_raw[];
    const uint32_t sA = (uint32_t)__cvta_generic_to_shared(smem_raw);
    const uint32_t sB = sA + NAT * ASZ;
    __shared__ __align__(8) uint64_t bars[STAGES + 6];
    __shared__ int tmem_slot[1];
    __shared__ int s_lastf[1];
    const uint32_t barB = (uint32_t)__cvta_generic_to_shared(&bars[0]);
    const uint32_t barM = barB + STAGES * 8;
    const uint32_t barA = barM + 16;
    const uint32_t barA1 = barA + 8;
    const uint32_t barAx = barA1 + 8;
    const uint32_t barBx = barAx + 8;

    if (warp == 0 && elect_one()) {
        for (int i = 0; i < STAGES + 6; ++i) mbar_init(barB + i * 8, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    } else if (warp == 1) {
        const uint32_t ad = (uint32_t)__cvta_generic_to_shared(tmem_slot);
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;"
                     :: "r"(ad), "r"(512));
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }
    __syncthreads();
    const uint32_t taddr = (uint32_t)tmem_slot[0];

    // Drain warp 1 fills ten trivial W1 rows per CTA while issuer warp 0
    // starts the first A/B TMA transfers below.  No CTA fence is needed:
    // kernel completion orders these stores before the selector launch.
    if constexpr (USE_W1_BOUNDARIES) {
        const int q = (int)blockIdx.x * 10 + tid - 32;
        if (tid >= 32 && tid < 42 && q < 2048) {
            const int own = q >> 7;
            int4* const out = reinterpret_cast<int4*>(
                const_cast<int*>(cu) + (size_t)q * TOPK);
            out[0] = make_int4(0 <= own ? 0 : -1, 1 <= own ? 1 : -1,
                               2 <= own ? 2 : -1, 3 <= own ? 3 : -1);
            out[1] = make_int4(4 <= own ? 4 : -1, 5 <= own ? 5 : -1,
                               6 <= own ? 6 : -1, 7 <= own ? 7 : -1);
            out[2] = make_int4(8 <= own ? 8 : -1, 9 <= own ? 9 : -1,
                               10 <= own ? 10 : -1, 11 <= own ? 11 : -1);
            out[3] = make_int4(12 <= own ? 12 : -1, 13 <= own ? 13 : -1,
                               14 <= own ? 14 : -1, 15 <= own ? 15 : -1);
        }
    }

    constexpr uint32_t idesc = (1u << 4) | (1u << 7) | (1u << 10)
                             | ((uint32_t)(BN / 8) << 17) | ((uint32_t)(BMT / 16) << 24);

    // The shared-memory layout is fixed for the lifetime of a persistent CTA.
    // Keep just the two A-tile and two B-stage descriptor bases live; every
    // unrolled K slice is a constant encoded-address delta from one base.
    const uint64_t adesc0 = smem_desc(sA);
    const uint64_t adesc1 = smem_desc(sA + ASZ);
    const uint64_t bdesc0 = smem_desc(sB);
    const uint64_t bdesc1 = smem_desc(sB + BSZ);

    // Warp 0 only issues MMA and TMA.  Warps 1..4 drain accumulator 0 and
    // warps 5..8 drain accumulator 1, so the two TMEM reduction chains run
    // concurrently instead of alternating through the issuer warp.  tcgen05
    // pins a warp to TMEM lanes [32*(warp%4), +32), so the row group is the
    // warp index mod 4 and both groups cover all 128 lanes.
    const int lane = tid & 31;
    const int fa   = (warp >= 5) ? 1 : 0;
    const int rg   = warp & 3;
    const int row  = rg * 32 + lane;
    const uint32_t lane_off = (uint32_t)(rg * 32) << 16;

    float score_buf[EPILOGUE4 ? 4 : 8];
    FLIST_DECL;
    int f_own = 0;
    int phA = 0, phAb = 0, phM0 = 0, phM1 = 0, phMw = 0, phAx = 0, phBx = 0;
    int phB[STAGES];
    if constexpr (FUSED) {
        #pragma unroll
        for (int i = 0; i < STAGES; ++i) phB[i] = 0;
    }

    while (FUSED ? (pos < hi) : (hot_pos < hot_hi)) {
        // Non-fused paths retire every B-stage transfer before the segment
        // rendezvous, so restart the two barriers and derive parity from the
        // segment-local iteration.  This removes the dynamically indexed
        // phase array from the critical scorer without changing fused W6.
        if constexpr (!FUSED) {
            if (warp == 0 && elect_one()) {
                mbar_init(barB, 1);
                mbar_init(barB + 8, 1);
                asm volatile("fence.mbarrier_init.release.cluster;");
            }
        }
        int nb, tile_units, qbase, nqt, krow0;
        if constexpr (USE_W1_BOUNDARIES) {
            // All W1 scored tiles are full 256-query tiles.  Tile t sees
            // blocks [0,2t+1), hence exactly t+1 two-block work units.
            nb = 2 * t + 1;
            tile_units = t + 1;
            qbase = t * BM;
            nqt = BM;
            krow0 = 0;
        } else {
            nb = (t == o.nt - 1) ? o.nbl : (o.n0 + 2 * t);
            tile_units = (t == o.nt - 1) ? ((o.nbl + 1) >> 1) : (o.m0 + t);
            qbase = o.q0 + t * BM;
            nqt = min(BM, o.nq - t * BM);
            krow0 = kvbase[r] * KBLOCK;
        }
        int nit;
        if constexpr (FUSED)
            nit = (int)min((long long)(tile_units - off), hi - pos);
        else
            nit = min(tile_units - off, hot_hi - hot_pos);
        const int b0  = off * NBLKT;
        const int b1  = min(nb, (off + nit) * NBLKT);

        if constexpr (!USE_W1_BOUNDARIES) {
            if (nqt < BM) {
                const uint4 z = make_uint4(0, 0, 0, 0);
                for (int so = tid * 16; so < NAT * ASZ; so += NTHR_P * 16)
                    *reinterpret_cast<uint4*>(smem_raw + so) = z;
                __syncthreads();
            }
        }
        // Only the first A sub-tile and B stage 0 gate the first MMA, so they
        // are issued first and awaited on separate barriers.  The second A
        // sub-tile and B stage 1 follow in the same burst: all four transfers
        // are in flight together, but the critical prologue shrinks from
        // 128 KB to 96 KB and the stage-1 refill no longer waits behind the
        // A-tile arrival.
        const bool a1_live = (BMT < nqt);
        if (warp == 0 && elect_one()) {
            if constexpr (!USE_W1_BOUNDARIES) {
                if (nqt < BM)
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            }
            const int kr = krow0 + b0 * KBLOCK;
            // Only D columns 0..63 of A sub-tile 0 and B stage 0 gate the
            // first four MMAs, so the blocking prologue is 48 KB, not 96 KB.
            mbar_expect(barA, ASZ / 2);
            tma_2d(sA,             &qmap, 0,  qbase, barA);
            mbar_expect(barB, BSZ / 2);
            tma_2d(sB,             &kmap, 0,  kr, barB);
            mbar_expect(barAx, ASZ / 2);
            tma_2d(sA + ASZ / 2,   &qmap, 64, qbase, barAx);
            mbar_expect(barBx, BSZ / 2);
            tma_2d(sB + BSZ / 2,   &kmap, 64, kr, barBx);
            if (a1_live) {
                const int rows = qbase + BMT;
                mbar_expect(barA1, ASZ);
                tma_2d(sA + ASZ,             &qmap, 0,  rows, barA1);
                tma_2d(sA + ASZ + ASZ / 2,   &qmap, 64, rows, barA1);
            }
            if (nit > 1) {
                const int kr1 = krow0 + (b0 + NBLKT) * KBLOCK;
                mbar_expect(barB + 8, BSZ);
                tma_2d(sB + BSZ,             &kmap, 0,  kr1, barB + 8);
                tma_2d(sB + BSZ + BSZ / 2,   &kmap, 64, kr1, barB + 8);
            }
        }
        if (warp == 0) {
            mbar_wait(barA, phA);
            phA ^= 1;
        }

        const int fq = qbase + fa * BMT + row;
        bool fvalid;
        float* fsrow;
        if constexpr (USE_W1_BOUNDARIES) {
            fvalid = true;
            fsrow = score_workspace<GUARDED>() + (size_t)fq * 128u;
        } else {
            fvalid = (fa * BMT + row) < nqt;
            fsrow = score_workspace<GUARDED>()
                  + (size_t)(fvalid ? fq : 0) * (size_t)pgstride;
        }
        if constexpr (FUSED) {
            FLIST_CLR;
            if constexpr (USE_W1_BOUNDARIES) f_own = fq >> 7;
            else f_own = (prefix[r] + (fq - o.q0)) >> 7;
        }

        auto do_epi = [&](int it) {
            {
                asm volatile("tcgen05.fence::after_thread_sync;");
                uint32_t base = taddr + (uint32_t)(fa * BN) + lane_off;
                const int bb = b0 + it * NBLKT;          // first block of this pair
                if constexpr (FUSED) {
                    float v0 = tmem_ld_max128(base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (fvalid && bb < f_own) FINSERT(pack_key(v0, bb))
                    if (bb + 1 < b1) {
                        float v1 = tmem_ld_max128(base + KBLOCK);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (fvalid && (bb + 1) < f_own) FINSERT(pack_key(v1, bb + 1))
                    }
                } else if constexpr (EPILOGUE4) {
                    const int k = (bb >> 1) & 1;
                    float v0 = tmem_ld_max128(base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int j = 0; j < 2; ++j) if (j == k) score_buf[2 * j] = v0;
                    if (bb + 1 < b1) {
                        float v1 = tmem_ld_max128(base + KBLOCK);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int j = 0; j < 2; ++j) if (j == k) score_buf[2 * j + 1] = v1;
                    }
                    const int last = min(bb + NBLKT - 1, b1 - 1);
                    if (((last & 3) == 3) || (last == b1 - 1)) {
                        const int gb = last & ~3;
                        const int j0 = max(0, b0 - gb);
                        const int j1 = last - gb;            // inclusive
                        if (fvalid) {
                            if (j0 == 0 && j1 == 3) {
                                *reinterpret_cast<float4*>(fsrow + gb) =
                                    make_float4(score_buf[0], score_buf[1],
                                                score_buf[2], score_buf[3]);
                            } else {
                                #pragma unroll
                                for (int j = 0; j < 4; ++j)
                                    if (j >= j0 && j <= j1) fsrow[gb + j] = score_buf[j];
                            }
                        }
                    }
                } else {
                    float v0 = tmem_ld_max128(base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) if (j == (bb & 7)) score_buf[j] = v0;
                    if (bb + 1 < b1) {
                        float v1 = tmem_ld_max128(base + KBLOCK);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int j = 0; j < 8; ++j) if (j == ((bb + 1) & 7)) score_buf[j] = v1;
                    }
                    const int last = min(bb + NBLKT - 1, b1 - 1);
                    if (((last & 7) == 7) || (last == b1 - 1)) {
                        const int gb = last & ~7;
                        const int j0 = max(0, b0 - gb);
                        const int j1 = last - gb;            // inclusive
                        if (fvalid) {
                            if (j0 == 0 && j1 == 7) {
                                st_global_v8f32(fsrow + gb, score_buf);
                            } else {
                                #pragma unroll
                                for (int j = 0; j < 8; ++j)
                                    if (j >= j0 && j <= j1) fsrow[gb + j] = score_buf[j];
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
            }
        };

        for (int i = 0; i < nit; ++i) {
            const int buf = i % STAGES;
            if (warp == 0) {
                if constexpr (FUSED) {
                    mbar_wait(barB + buf * 8, phB[buf]);
                    phB[buf] ^= 1;
                } else {
                    mbar_wait(barB + buf * 8, (i / STAGES) & 1);
                }
                // Accumulator 0 is free again once its previous epilogue
                // retired.  The issuer and the drain warps reach this barrier
                // from different instructions, so it must be the non-aligned
                // barrier.sync, not bar.sync (= barrier.sync.aligned).
                if (i > 0) asm volatile("barrier.sync 1, 160;" ::: "memory");
                if (elect_one()) {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t acc = taddr;
                    #pragma unroll
                    for (int kk = 0; kk < 4; ++kk)
                        mma_f16(acc,
                                adesc0 + (uint64_t)(kk & 3) * 2,
                                (buf ? bdesc1 : bdesc0) + (uint64_t)(kk & 3) * 2,
                                idesc, kk ? 1 : 0);
                }
                if (i == 0) {
                    mbar_wait(barAx, phAx); phAx ^= 1;
                    mbar_wait(barBx, phBx); phBx ^= 1;
                }
                if (elect_one()) {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t acc = taddr;
                    #pragma unroll
                    for (int kk = 4; kk < 8; ++kk)
                        mma_f16(acc,
                                adesc0 + (uint64_t)(ASZ / 32) + (uint64_t)(kk & 3) * 2,
                                (buf ? bdesc1 : bdesc0)
                                       + (uint64_t)(BSZ / 32) + (uint64_t)(kk & 3) * 2,
                                idesc, 1);
                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(barM) : "memory");
                }
                // Refill the stage consumed last iteration as soon as its last
                // MMA retired -- before waiting on that epilogue -- so the
                // transfer overlaps the two x128 TMEM reductions.
                if (i >= 1) {
                    mbar_wait(barM + 8, phMw);
                    phMw ^= 1;
                    if (elect_one()) {
                        const int nxt = i - 1 + STAGES;
                        if (nxt < nit) {
                            const int bf = (i - 1) % STAGES;
                            const int kr = krow0 + (b0 + nxt * NBLKT) * KBLOCK;
                            mbar_expect(barB + bf * 8, BSZ);
                            tma_2d(sB + bf * BSZ,           &kmap, 0,  kr, barB + bf * 8);
                            tma_2d(sB + bf * BSZ + BSZ / 2, &kmap, 64, kr, barB + bf * 8);
                        }
                    }
                }
                // Accumulator 1 is free again once its previous epilogue retired.
                if (i > 0) {
                    asm volatile("barrier.sync 2, 160;" ::: "memory");
                } else if (a1_live) {
                    // The second A sub-tile is only needed here, one MMA group
                    // after the first, so its arrival overlaps acc0's issue.
                    mbar_wait(barA1, phAb);
                    phAb ^= 1;
                }
                if (elect_one()) {
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    uint32_t acc = taddr + (uint32_t)BN;
                    #pragma unroll
                    for (int kk = 0; kk < 8; ++kk)
                        mma_f16(acc,
                                adesc1 + (uint64_t)(kk >> 2) * (ASZ / 32)
                                       + (uint64_t)(kk & 3) * 2,
                                (buf ? bdesc1 : bdesc0)
                                       + (uint64_t)(kk >> 2) * (BSZ / 32)
                                       + (uint64_t)(kk & 3) * 2,
                                idesc, kk ? 1 : 0);
                    asm volatile(
                        "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                        :: "r"(barM + 8) : "memory");
                }
            } else if (warp <= 4) {
                mbar_wait(barM, phM0);
                phM0 ^= 1;
                do_epi(i);
                asm volatile("barrier.sync 1, 160;" ::: "memory");
            } else {
                mbar_wait(barM + 8, phM1);
                phM1 ^= 1;
                do_epi(i);
                asm volatile("barrier.sync 2, 160;" ::: "memory");
            }
        }
        if (warp == 0) {
            // The issuer consumed one fewer acc1 completion phase than the
            // barrier advanced over this segment; realign before reuse.
            phMw ^= 1;
            asm volatile("barrier.sync 1, 160;" ::: "memory");
            asm volatile("barrier.sync 2, 160;" ::: "memory");
        }
        // Persistent segments may change query tile, request, and B range.
        // All TMEM readers and the issuer must retire before A/B SMEM reuse.
        __syncthreads();

        if constexpr (FUSED) {
            // FUSED is dispatched for single-request shapes only, so the
            // tile's global unit prefix is its in-request prefix.
            const long long tp = USE_W1_BOUNDARIES
                ? ((long long)(t - 8) * (long long)(t + 9) / 2)
                : (wbase + persistent_tile_pfx(o, t));
            const int tg = USE_W1_BOUNDARIES ? t : (tgbase + t);
            const int clo = cta_of_unit(tp, pmode, Wtot_v);
            const int nsl = cta_of_unit(tp + tile_units - 1, pmode, Wtot_v) - clo + 1;
            const int qloc = fa * BMT + row;
            if (nsl <= 1) {
                if (tid >= 32 && fvalid) {
                    int I[16];
                    if (f_own <= 15) {
                        #pragma unroll
                        for (int i = 0; i < 16; ++i) I[i] = (i <= f_own) ? i : -1;
                    } else {
                        I[0]=(int)(0xFFFFFFFFu-(uint32_t)L0);
                        I[1]=(int)(0xFFFFFFFFu-(uint32_t)L1);
                        I[2]=(int)(0xFFFFFFFFu-(uint32_t)L2);
                        I[3]=(int)(0xFFFFFFFFu-(uint32_t)L3);
                        I[4]=(int)(0xFFFFFFFFu-(uint32_t)L4);
                        I[5]=(int)(0xFFFFFFFFu-(uint32_t)L5);
                        I[6]=(int)(0xFFFFFFFFu-(uint32_t)L6);
                        I[7]=(int)(0xFFFFFFFFu-(uint32_t)L7);
                        I[8]=(int)(0xFFFFFFFFu-(uint32_t)L8);
                        I[9]=(int)(0xFFFFFFFFu-(uint32_t)L9);
                        I[10]=(int)(0xFFFFFFFFu-(uint32_t)L10);
                        I[11]=(int)(0xFFFFFFFFu-(uint32_t)L11);
                        I[12]=(int)(0xFFFFFFFFu-(uint32_t)L12);
                        I[13]=(int)(0xFFFFFFFFu-(uint32_t)L13);
                        I[14]=(int)(0xFFFFFFFFu-(uint32_t)L14);
                        I[15]=f_own;
                        SORT16_ASC(I)
                    }
                    store_sel16(selp + (size_t)fq * TOPK, I);
                }
            } else {
                uint64_t* const pw = part_workspace<GUARDED>()
                    + ((size_t)tg * MAXSLOT + (size_t)(blockIdx.x - clo)) * PARTW
                    + qloc;
                if (tid >= 32) {
                    pw[ 0*BM]=L0;  pw[ 1*BM]=L1;  pw[ 2*BM]=L2;  pw[ 3*BM]=L3;
                    pw[ 4*BM]=L4;  pw[ 5*BM]=L5;  pw[ 6*BM]=L6;  pw[ 7*BM]=L7;
                    pw[ 8*BM]=L8;  pw[ 9*BM]=L9;  pw[10*BM]=L10; pw[11*BM]=L11;
                    pw[12*BM]=L12; pw[13*BM]=L13; pw[14*BM]=L14;
                }
                __syncthreads();
                if (tid == 0) {
                    int old;
                    asm volatile("atom.release.gpu.global.add.s32 %0, [%1], 1;"
                                 : "=r"(old) : "l"(&tile_counts<GUARDED>()[tg]) : "memory");
                    s_lastf[0] = (old == nsl - 1);
                    if (old == nsl - 1)
                        asm volatile("st.relaxed.gpu.global.s32 [%0], 0;"
                                     :: "l"(&tile_counts<GUARDED>()[tg]) : "memory");
                }
                __syncthreads();
                if (s_lastf[0]) {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    if (tid < BM && tid < nqt) {
                        const int q2 = qbase + tid;
                        const int own2 = USE_W1_BOUNDARIES
                            ? (q2 >> 7) : ((prefix[r] + (q2 - o.q0)) >> 7);
                        int I[16];
                        if (own2 <= 15) {
                            #pragma unroll
                            for (int i = 0; i < 16; ++i) I[i] = (i <= own2) ? i : -1;
                        } else {
                            uint64_t M[16];
                            #pragma unroll
                            for (int i = 0; i < 16; ++i) M[i] = 0ull;
                            #pragma unroll 1
                            for (int sl = 0; sl < nsl; ++sl) {
                                const uint64_t* const sp = part_workspace<GUARDED>()
                                    + ((size_t)tg * MAXSLOT + (size_t)sl) * PARTW + tid;
                                #pragma unroll
                                for (int i = 1; i < 16; ++i) {
                                    const uint64_t nv = sp[(15 - i) * BM];
                                    if (nv > M[i]) M[i] = nv;
                                }
                                #pragma unroll
                                for (int st = 8; st > 0; st >>= 1)
                                    #pragma unroll
                                    for (int i = 0; i < 16; ++i)
                                        if ((i & st) == 0) FCE(M[i], M[i + st])
                            }
                            #pragma unroll
                            for (int i = 0; i < 15; ++i)
                                I[i] = (int)(0xFFFFFFFFu - (uint32_t)M[i]);
                            I[15] = own2;
                            SORT16_ASC(I)
                        }
                        store_sel16(selp + (size_t)q2 * TOPK, I);
                    }
                }
            }
            __syncthreads();
        }

        if constexpr (FUSED) {
            pos += nit;
            if (pos >= hi) break;
        } else {
            hot_pos += nit;
            if (hot_pos >= hot_hi) break;
        }
        if constexpr (FUSED) {
            off += nit;
        } else {
            // If this CTA still has work, min(tile remainder, CTA remainder)
            // necessarily selected the tile remainder.  Do not carry the
            // tile scheduler temporaries across the MMA/epilogue loop.
            off = 0;
            ++t;
            if constexpr (!USE_W1_BOUNDARIES) {
                while (t >= o.nt) {
                    wbase += o.W;
                    tgbase += o.nt;
                    ++r;
                    if (r >= B) break;
                    persistent_req_info(cu, prefix, r, o);
                    t = (o.W > 0) ? ((o.tend > o.ts) ? o.ts : o.tend) : o.nt;
                }
                if (r >= B) break;
            }
        }
        if constexpr (FUSED) {
            if (off >= tile_units) {
                off = 0;
                ++t;
                if constexpr (!USE_W1_BOUNDARIES) {
                    while (t >= o.nt) {
                        wbase += o.W;
                        tgbase += o.nt;
                        ++r;
                        if (r >= B) break;
                        persistent_req_info(cu, prefix, r, o);
                        t = (o.W > 0) ? ((o.tend > o.ts) ? o.ts : o.tend) : o.nt;
                    }
                    if (r >= B) break;
                }
            }
        }
    }
    // Rows whose own block is <= 15 take every prior block and belong to no
    // scored tile, so the fused kernel emits them itself.  Issued after the
    // scoring loop, where it cannot delay the first A/B transfers.
    if constexpr (FUSED && !USE_W1_BOUNDARIES) {
        for (int rr = 0; rr < B; ++rr) {
            const int q0r = cu[rr];
            int c = (TOPK * KBLOCK) - prefix[rr];
            const int nqr = cu[rr + 1] - q0r;
            if (c > nqr) c = nqr;
            for (int ii = (int)blockIdx.x * NTHR_P + tid; ii < c;
                 ii += NCTA * NTHR_P) {
                const int ow = (prefix[rr] + ii) >> 7;
                int4* const outf =
                    reinterpret_cast<int4*>(selp + (size_t)(q0r + ii) * TOPK);
                outf[0] = make_int4(0, 1 <= ow ? 1 : -1,
                                    2 <= ow ? 2 : -1, 3 <= ow ? 3 : -1);
                outf[1] = make_int4(4 <= ow ? 4 : -1, 5 <= ow ? 5 : -1,
                                    6 <= ow ? 6 : -1, 7 <= ow ? 7 : -1);
                outf[2] = make_int4(8 <= ow ? 8 : -1, 9 <= ow ? 9 : -1,
                                    10 <= ow ? 10 : -1, 11 <= ow ? 11 : -1);
                outf[3] = make_int4(12 <= ow ? 12 : -1, 13 <= ow ? 13 : -1,
                                    14 <= ow ? 14 : -1, 15 <= ow ? 15 : -1);
            }
        }
    }
    if (warp == 1) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;"
                     :: "r"(taddr), "r"(512));
    }
}

// ---------------- stage B : per-query top-k selection ----------------
// Pack score and complemented index so one unsigned comparison implements
// score-descending / id-ascending order exactly.

__device__ __forceinline__ uint64_t warp_max_u64(uint64_t b) {
    #pragma unroll
    for (int off = 16; off; off >>= 1) {
        uint64_t o = __shfl_xor_sync(0xffffffffu, b, off);
        b = (o > b) ? o : b;
    }
    return b;
}

template <int NPL>
__device__ __forceinline__ int pick_lane_val(const float* __restrict__ srow,
                                              int n, int lane) {
    uint64_t keys[NPL];
    #pragma unroll
    for (int i = 0; i < NPL; ++i) {
        int j = lane + i * 32;
        keys[i] = (j < n) ? pack_key(srow[j], j) : 0ull;
    }
    uint64_t lastk = ~0ull;
    int myval = -1;
    #pragma unroll 1
    for (int it = 0; it < 15; ++it) {
        uint64_t b = 0ull;
        #pragma unroll
        for (int i = 0; i < NPL; ++i) {
            uint64_t k = keys[i];
            uint64_t c = (k < lastk) ? k : 0ull;
            b = (c > b) ? c : b;
        }
        b = warp_max_u64(b);
        if (lane == it) myval = (int)(0xFFFFFFFFu - (uint32_t)b);
        lastk = b;
    }
    return myval;
}

__device__ __forceinline__ int pick_lane_val_slow(const float* __restrict__ srow,
                                                   int n, int lane) {
    uint64_t lastk = ~0ull;
    int myval = -1;
    #pragma unroll 1
    for (int it = 0; it < 15; ++it) {
        uint64_t b = 0ull;
        for (int j = lane; j < n; j += 32) {
            uint64_t k = pack_key(srow[j], j);
            uint64_t c = (k < lastk) ? k : 0ull;
            b = (c > b) ? c : b;
        }
        b = warp_max_u64(b);
        if (lane == it) myval = (int)(0xFFFFFFFFu - (uint32_t)b);
        lastk = b;
    }
    return myval;
}

__device__ __forceinline__ uint32_t bitonic_desc32_u32(uint32_t v, int lane) {
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            uint32_t p = __shfl_xor_sync(0xffffffffu, v, j);
            bool asc = ((lane & k) != 0);
            bool keep_max = (((lane & j) != 0) == asc);
            v = keep_max ? max(v, p) : min(v, p);
        }
    }
    return v;
}

__device__ __forceinline__ int bitonic_asc16_int(int v, int lane) {
    #pragma unroll
    for (int k = 2; k <= 16; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            int p = __shfl_xor_sync(0x0000ffffu, v, j, 16);
            bool ascending = ((lane & k) == 0);
            bool lower = ((lane & j) == 0);
            v = (lower == ascending) ? min(v, p) : max(v, p);
        }
    }
    return v;
}

// When the physical context has at most 16 blocks, every prior block is
// selected.  One thread writes one complete row in four aligned vector stores.
__global__ __launch_bounds__(256) void direct_fill_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel, int B, int TQ) {
    const int q = (int)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= TQ) return;
    int lo = 0, hi = B;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cu[mid] <= q) lo = mid; else hi = mid;
    }
    const int own = (prefix[lo] + q - cu[lo]) >> 7;
    int4* out = reinterpret_cast<int4*>(sel + (size_t)q * TOPK);
    out[0] = make_int4(0 <= own ? 0 : -1, 1 <= own ? 1 : -1,
                       2 <= own ? 2 : -1, 3 <= own ? 3 : -1);
    out[1] = make_int4(4 <= own ? 4 : -1, 5 <= own ? 5 : -1,
                       6 <= own ? 6 : -1, 7 <= own ? 7 : -1);
    out[2] = make_int4(8 <= own ? 8 : -1, 9 <= own ? 9 : -1,
                       10 <= own ? 10 : -1, 11 <= own ? 11 : -1);
    out[3] = make_int4(12 <= own ? 12 : -1, 13 <= own ? 13 : -1,
                       14 <= own ? 14 : -1, 15 <= own ? 15 : -1);
}

// The 15th-largest lane maximum score is a lower bound on the global
// 15th-largest score.  Therefore all exact (score,id) winners survive this
// filter.  Keeping only the 32-bit ordered scores resident halves the key
// array pressure, including the rare overfull exact-selection path.
// Descending bitonic sort of one value per lane, float domain.
__device__ __forceinline__ float bitonic_desc32_f32(float v, int lane) {
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            float p = __shfl_xor_sync(0xffffffffu, v, j);
            bool asc = ((lane & k) != 0);
            bool keep_max = (((lane & j) != 0) == asc);
            v = keep_max ? fmaxf(v, p) : fminf(v, p);
        }
    }
    return v;
}

// Descending bitonic sort of one packed (score, ~id) key per lane.  Sorting
// the surviving candidates outright replaces the iterative drop-the-worst
// loop, which needed two warp reductions per eliminated entry plus a shared
// staging buffer for the ranked ids.
__device__ __forceinline__ uint64_t bitonic_desc32_u64(uint64_t v, int lane) {
    #pragma unroll
    for (int k = 2; k <= 32; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j > 0; j >>= 1) {
            uint64_t p = __shfl_xor_sync(0xffffffffu, v, j);
            bool asc = ((lane & k) != 0);
            bool keep_max = (((lane & j) != 0) == asc);
            v = keep_max ? (v > p ? v : p) : (v > p ? p : v);
        }
    }
    return v;
}

// Lean exact top-15 selector.
//  * thresholding and the lane-max reduction stay in the float domain, so the
//    ordered-key conversion runs once per surviving candidate instead of once
//    per scanned block;
//  * candidates are compacted with one warp scan of per-lane counts instead of
//    a ballot + popc chain per scanned element;
//  * the 15 winners are ordered by a 16-way bitonic sort at the end, which
//    costs less than keeping the candidate pool id-ordered during compaction.
// The 15th-largest lane maximum is a lower bound on the global 15th-largest
// score, so every exact winner survives the threshold.
template <int NPL>
__device__ __forceinline__ bool pick_store_fast_v2(
        const float* __restrict__ srow, int n, int lane,
        uint64_t* __restrict__ pool, int* __restrict__ out, int own) {
    float v[NPL];
    float lmax = -FLT_MAX;
    #pragma unroll
    for (int i = 0; i < NPL; ++i) {
        int j = lane + i * 32;
        float x = (j < n) ? srow[j] : -FLT_MAX;
        v[i] = x;
        lmax = fmaxf(lmax, x);
    }
    const float T = __shfl_sync(0xffffffffu, bitonic_desc32_f32(lmax, lane), 14);

    int cnt = 0;
    #pragma unroll
    for (int i = 0; i < NPL; ++i) cnt += (v[i] >= T) ? 1 : 0;
    int pfx = cnt;
    #pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
        int o = __shfl_up_sync(0xffffffffu, pfx, d);
        if (lane >= d) pfx += o;
    }
    const int base = __shfl_sync(0xffffffffu, pfx, 31);
    pfx -= cnt;

    if (base > 32) {
        int myval = pick_lane_val<NPL>(srow, n, lane);
        if (lane == 15) myval = own;
        if (lane < 16) {
            myval = bitonic_asc16_int(myval, lane);
            out[lane] = myval;
        }
        return true;
    }

    int p = pfx;
    #pragma unroll
    for (int i = 0; i < NPL; ++i) {
        if (v[i] >= T) {
            pool[p] = ((uint64_t)__float_as_uint(v[i]) << 32)
                    | (uint32_t)(lane + i * 32);
            ++p;
        }
    }
    __syncwarp();

    const uint64_t e = (lane < base) ? pool[lane] : 0ull;
    uint64_t key = (lane < base)
        ? pack_ordered(ordered_score(__uint_as_float((uint32_t)(e >> 32))),
                       (int)(uint32_t)e)
        : 0ull;
    key = bitonic_desc32_u64(key, lane);
    if (lane < 16) {
        int val = (lane < 15) ? (int)(0xFFFFFFFFu - (uint32_t)key) : own;
        val = bitonic_asc16_int(val, lane);
        out[lane] = val;
    }
    return true;
}

template <int NPL>
__device__ __forceinline__ bool pick_store_fast(
        const float* __restrict__ srow, int n, int lane,
        uint64_t* __restrict__ pool, int* __restrict__ out, int own) {
    uint32_t scores[NPL];
    uint32_t lmax = 0u;
    #pragma unroll
    for (int i = 0; i < NPL; ++i) {
        int j = lane + i * 32;
        uint32_t k = (j < n) ? ordered_score(srow[j]) : 0u;
        scores[i] = k;
        if (k > lmax) lmax = k;
    }
    uint32_t threshold = __shfl_sync(0xffffffffu,
                                     bitonic_desc32_u32(lmax, lane), 14);
    const unsigned ltmask = (1u << lane) - 1u;
    int base = 0;
    #pragma unroll
    for (int i = 0; i < NPL; ++i) {
        int j = lane + i * 32;
        bool keep = (j < n) && (scores[i] >= threshold);
        unsigned mask = __ballot_sync(0xffffffffu, keep);
        int pos = base + __popc(mask & ltmask);
        if (keep && pos < 32) pool[pos] = pack_ordered(scores[i], j);
        base += __popc(mask);
    }
    if (base > 32) {
        uint64_t lastk = ~0ull;
        int myval = -1;
        #pragma unroll 1
        for (int it = 0; it < 15; ++it) {
            uint64_t best = 0ull;
            #pragma unroll
            for (int i = 0; i < NPL; ++i) {
                int j = lane + i * 32;
                uint64_t k = (j < n) ? pack_ordered(scores[i], j) : 0ull;
                uint64_t c = (k < lastk) ? k : 0ull;
                best = (c > best) ? c : best;
            }
            best = warp_max_u64(best);
            if (lane == it)
                myval = (int)(0xffffffffu - (uint32_t)best);
            lastk = best;
        }
        if (lane == 15) myval = own;
        if (lane < 16) {
            myval = bitonic_asc16_int(myval, lane);
            out[lane] = myval;
        }
        return true;
    }
    __syncwarp();

    // Compaction visits i-major then lane-major, so pool lanes are already
    // ascending block ids.  Remove exactly the (base-15) worst candidates:
    // minimum score, breaking a tie by the largest id (highest live lane).
    // The remaining lane order is therefore the required ascending output.
    bool live = lane < base;
    uint64_t v = live ? pool[lane] : 0ull;
    for (int r = 15; r < base; ++r) {
        uint32_t sk = live ? (uint32_t)(v >> 32) : 0xffffffffu;
        uint32_t mn = __reduce_min_sync(0xffffffffu, sk);
        unsigned eq = __ballot_sync(0xffffffffu, live && sk == mn);
        int loser = 31 - __clz(eq);
        if (lane == loser) live = false;
    }
    unsigned keep = __ballot_sync(0xffffffffu, live);
    int rank = __popc(keep & ((1u << lane) - 1u));
    if (live)
        out[rank] = (int)(0xffffffffu - (uint32_t)v);
    if (lane == 15)
        out[15] = own;
    return true;
}

template <bool GUARDED>
__global__ __launch_bounds__(256) void topk_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel,
    int B, int TQ, int pgstride)
{
    __shared__ uint64_t s_pool[256];
    const int q = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (q < TQ) {
        int lo = 0, hi = B;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (cu[mid] <= q) lo = mid; else hi = mid;
        }
        const int own = (prefix[lo] + (q - cu[lo])) >> 7;
        int* const out = sel + (size_t)q * TOPK;
        if (own <= 15) {
            if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
        } else {
            const float* srow = score_workspace<GUARDED>()
                              + (size_t)q * (size_t)pgstride;
            uint64_t* mypool = &s_pool[(threadIdx.x >> 5) << 5];
            if (own <= 32 * 4) {
                pick_store_fast<4>(srow, own, lane, mypool, out, own);
            } else if (own <= 32 * 8) {
                pick_store_fast<8>(srow, own, lane, mypool, out, own);
            } else if (own <= 32 * 20) {
                pick_store_fast<20>(srow, own, lane, mypool, out, own);
            } else {
                int myval = pick_lane_val_slow(srow, own, lane);
                if (lane == 15) myval = own;
                if (lane < 16) {
                    myval = bitonic_asc16_int(myval, lane);
                    out[lane] = myval;
                }
            }
        }
    }
    if constexpr (GUARDED) finish_workspace();
}

// Register-shaped selector for generated workloads whose maximum local page
// count is known on the host.  Separate instantiations prevent a short-context
// selector from inheriting the NPL=20 register footprint of the generic path.
template <int NPL, bool GUARDED>
__global__ __launch_bounds__(256) void topk_fixed_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel,
    int B, int TQ, int pgstride)
{
    __shared__ uint64_t s_pool[256];
    const int q = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (q < TQ) {
        int lo = 0, hi = B;
        while (hi - lo > 1) {
            int mid = (lo + hi) >> 1;
            if (cu[mid] <= q) lo = mid; else hi = mid;
        }
        const int own = (prefix[lo] + (q - cu[lo])) >> 7;
        int* const out = sel + (size_t)q * TOPK;
        if (own <= 15) {
            if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
        } else {
            const float* srow = score_workspace<GUARDED>()
                              + (size_t)q * (size_t)pgstride;
            if (own > 32 * NPL) {
                int myval = pick_lane_val_slow(srow, own, lane);
                if (lane == 15) myval = own;
                if (lane < 16) {
                    myval = bitonic_asc16_int(myval, lane);
                    out[lane] = myval;
                }
            } else {
                uint64_t* mypool = &s_pool[(threadIdx.x >> 5) << 5];
                // A warp owns one query, so this bound test never diverges.
                if (NPL >= 4 && own <= 32 * (NPL / 2))
                    pick_store_fast<(NPL >= 4) ? (NPL / 2) : 1>(
                        srow, own, lane, mypool, out, own);
                else if (NPL >= 8)
                    pick_store_fast_v2<NPL>(srow, own, lane, mypool, out, own);
                else
                    pick_store_fast<NPL>(srow, own, lane, mypool, out, own);
            }
        }
    }
    if constexpr (GUARDED) finish_workspace();
}

// The permanently assigned primary stream needs no cross-CTA completion
// protocol.  Keep its selector in the original early-return form so the hot
// path pays neither a terminal barrier nor the control-flow shape needed by
// the guarded overflow slot.
__global__ __launch_bounds__(256) void topk_primary_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel,
    int B, int TQ, int pgstride)
{
    __shared__ uint64_t s_pool[256];
    const int q = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (q >= TQ) return;
    const int lane = threadIdx.x & 31;

    int lo = 0, hi = B;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cu[mid] <= q) lo = mid; else hi = mid;
    }
    const int own = (prefix[lo] + (q - cu[lo])) >> 7;
    int* const out = sel + (size_t)q * TOPK;
    if (own <= 15) {
        if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
        return;
    }
    const float* srow = g_scores_primary + (size_t)q * (size_t)pgstride;
    uint64_t* mypool = &s_pool[(threadIdx.x >> 5) << 5];
    if (own <= 32 * 4) {
        pick_store_fast<4>(srow, own, lane, mypool, out, own);
        return;
    }
    if (own <= 32 * 8) {
        pick_store_fast<8>(srow, own, lane, mypool, out, own);
        return;
    }
    if (own <= 32 * 20) {
        pick_store_fast<20>(srow, own, lane, mypool, out, own);
        return;
    }

    int myval = pick_lane_val_slow(srow, own, lane);
    if (lane == 15) myval = own;
    if (lane < 16) {
        myval = bitonic_asc16_int(myval, lane);
        out[lane] = myval;
    }
}

template <int NPL>
__global__ __launch_bounds__(256) void topk_primary_fixed_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel,
    int B, int TQ, int pgstride)
{
    __shared__ uint64_t s_pool[256];
    const int q = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (q >= TQ) return;
    const int lane = threadIdx.x & 31;

    int lo = 0, hi = B;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cu[mid] <= q) lo = mid; else hi = mid;
    }
    const int own = (prefix[lo] + (q - cu[lo])) >> 7;
    int* const out = sel + (size_t)q * TOPK;
    if (own <= 15) {
        if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
        return;
    }

    const float* srow = g_scores_primary + (size_t)q * (size_t)pgstride;
    if (own > 32 * NPL) {
        int myval = pick_lane_val_slow(srow, own, lane);
        if (lane == 15) myval = own;
        if (lane < 16) {
            myval = bitonic_asc16_int(myval, lane);
            out[lane] = myval;
        }
        return;
    }
    uint64_t* mypool = &s_pool[(threadIdx.x >> 5) << 5];
    if (NPL >= 4 && own <= 32 * (NPL / 2)) {
        pick_store_fast<(NPL >= 4) ? (NPL / 2) : 1>(
            srow, own, lane, mypool, out, own);
        return;
    }
    if (NPL >= 8)
        pick_store_fast_v2<NPL>(srow, own, lane, mypool, out, own);
    else
        pick_store_fast<NPL>(srow, own, lane, mypool, out, own);
}

// Exact W1 selector.  The fixed W1 scorer has already filled q<2048.  One
// full-residency 848-CTA wave grid-strides over the remaining queries, which
// removes a second round of block placement while preserving one warp/query.
template <bool GUARDED>
__global__ __launch_bounds__(256) void topk_w1_exact_kernel(
    int* __restrict__ sel)
{
    __shared__ uint64_t s_pool[256];
    const int lane = threadIdx.x & 31;
    uint64_t* const mypool = &s_pool[(threadIdx.x >> 5) << 5];
    const int wid = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int nrow = (wid < 768) ? 3 : 2;
    const int q0 = (wid < 768)
                 ? (2048 + 3 * wid)
                 : (4352 + 2 * (wid - 768));
    #pragma unroll 1
    for (int it = 0; it < nrow; ++it) {
        const int q = q0 + it;
        const int own = q >> 7;
        int* const out = sel + (size_t)q * TOPK;
        const float* const srow = score_workspace<GUARDED>()
                                + (size_t)q * 128u;
        pick_store_fast<4>(srow, own, lane, mypool, out, own);
    }

    if constexpr (GUARDED) finish_workspace();
}

// W5 has one request and exactly 640 physical pages.  Preserve the metadata
// loads that define each query's own block, but remove the per-query request
// search, selector-tier dispatch, and unreachable slow-path test.
template <bool GUARDED>
__global__ __launch_bounds__(256) void topk_w5_exact_kernel(
    const int* __restrict__ cu,
    const int* __restrict__ prefix,
    int* __restrict__ sel)
{
    __shared__ uint64_t s_pool[256];
    // Put the shortest rows in the underfilled final resident wave.
    const int q = 16383
                - (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    const int own = (prefix[0] + q - cu[0]) >> 7;
    int* const out = sel + (size_t)q * TOPK;
    if (own <= 15) {
        if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
    } else {
        const float* const srow = score_workspace<GUARDED>()
                                + (size_t)q * 640u;
        uint64_t* const mypool = &s_pool[(threadIdx.x >> 5) << 5];
        if (own <= 544)
            pick_store_fast_v2<17>(srow, own, lane, mypool, out, own);
        else if (own <= 576)
            pick_store_fast_v2<18>(srow, own, lane, mypool, out, own);
        else if (own <= 608)
            pick_store_fast_v2<19>(srow, own, lane, mypool, out, own);
        else
            pick_store_fast_v2<20>(srow, own, lane, mypool, out, own);
    }
    if constexpr (GUARDED) finish_workspace();
}

// Allocation-free exact fallback when an in-scope score matrix exceeds the
// fixed fast workspace.  A warp owns one query and keeps the best 15 blocks;
// ascending block traversal implements the lower-id tie break.
__global__ __launch_bounds__(256, 2) void direct_fallback_kernel(
    const __nv_bfloat16* __restrict__ qi,
    const __nv_bfloat16* __restrict__ ki,
    const int* __restrict__ cu,
    const int* __restrict__ kvbase,
    const int* __restrict__ prefix,
    int* __restrict__ sel, int B, int TQ) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q = (int)blockIdx.x * 8 + warp;
    if (q >= TQ) return;

    int lo = 0, hi = B;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (cu[mid] <= q) lo = mid; else hi = mid;
    }
    const int own = (prefix[lo] + q - cu[lo]) >> 7;
    int* out = sel + (size_t)q * TOPK;
    if (own <= 15) {
        if (lane < 16) out[lane] = (lane <= own) ? lane : -1;
        return;
    }

    float qreg[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j)
        qreg[j] = __bfloat162float(qi[(size_t)q * DHEAD + lane + j * 32]);

    __shared__ float sbest[8][15];
    __shared__ int sid[8][15];
    float worst = FLT_MAX;
    int worst_id = -1, worst_pos = 0, seen = 0;
    const int key0 = kvbase[lo] * KBLOCK;
    for (int block = 0; block < own; ++block) {
        float bmax = -FLT_MAX;
        #pragma unroll 1
        for (int key = 0; key < KBLOCK; ++key) {
            const __nv_bfloat16* kp =
                ki + (size_t)(key0 + block * KBLOCK + key) * DHEAD;
            float sum = 0.0f;
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                sum = fmaf(qreg[j], __bfloat162float(kp[lane + j * 32]), sum);
            #pragma unroll
            for (int off = 16; off; off >>= 1)
                sum += __shfl_down_sync(0xffffffffu, sum, off);
            if (lane == 0) bmax = fmaxf(bmax, sum);
        }
        if (lane == 0) {
            if (seen < 15) {
                sbest[warp][seen] = bmax;
                sid[warp][seen] = block;
                if (bmax < worst || (bmax == worst && block > worst_id)) {
                    worst = bmax; worst_id = block; worst_pos = seen;
                }
                ++seen;
            } else if (bmax > worst) {
                #pragma unroll
                for (int i = 0; i < 14; ++i) {
                    if (i >= worst_pos) {
                        sbest[warp][i] = sbest[warp][i + 1];
                        sid[warp][i] = sid[warp][i + 1];
                    }
                }
                sbest[warp][14] = bmax;
                sid[warp][14] = block;
                worst = sbest[warp][0];
                worst_id = sid[warp][0];
                worst_pos = 0;
                #pragma unroll
                for (int i = 1; i < 15; ++i) {
                    float x = sbest[warp][i];
                    int id = sid[warp][i];
                    if (x < worst || (x == worst && id > worst_id)) {
                        worst = x; worst_id = id; worst_pos = i;
                    }
                }
            }
        }
    }
    if (lane == 0) {
        int ids[16];
        #pragma unroll
        for (int i = 0; i < 15; ++i) ids[i] = sid[warp][i];
        ids[15] = own;
        #pragma unroll
        for (int i = 1; i < 16; ++i) {
            int x = ids[i];
            int j = i - 1;
            while (j >= 0 && ids[j] > x) {
                ids[j + 1] = ids[j];
                --j;
            }
            ids[j + 1] = x;
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) out[i] = ids[i];
    }
}

// ---------------- host side ----------------
static void make_map(CUtensorMap* tm, const void* ptr, uint64_t rows, uint32_t box_rows) {
    uint64_t gdim[2]   = { (uint64_t)DHEAD, rows };
    uint64_t gstr[1]   = { (uint64_t)DHEAD * 2ull };
    uint32_t bdim[2]   = { 64u, box_rows };
    uint32_t bstr[2]   = { 1u, 1u };
    cuTensorMapEncodeTiled(tm, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, (void*)ptr,
                           gdim, gstr, bdim, bstr,
                           CU_TENSOR_MAP_INTERLEAVE_NONE,
                           CU_TENSOR_MAP_SWIZZLE_128B,
                           CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                           CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static int g_score_attr_set = 0;
static std::mutex g_launch_mutex;
static bool g_primary_stream_set = false;
static cudaStream_t g_primary_stream = nullptr;

template <bool GUARDED>
static void launch_topk(const int* cu, const int* prefix, int* sel,
                        int B, int TQ, int pgstride, int max_pg,
                        int tk_blocks, int tk_threads, cudaStream_t stream) {
    if constexpr (GUARDED) {
        if (max_pg <= 128)
            topk_fixed_kernel<4, true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 256)
            topk_fixed_kernel<8, true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 544)
            topk_fixed_kernel<17, true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 576)
            topk_fixed_kernel<18, true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 640)
            topk_fixed_kernel<20, true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else
            topk_kernel<true><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
    } else {
        if (max_pg <= 128)
            topk_primary_fixed_kernel<4><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 256)
            topk_primary_fixed_kernel<8><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 544)
            topk_primary_fixed_kernel<17><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 576)
            topk_primary_fixed_kernel<18><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else if (max_pg <= 640)
            topk_primary_fixed_kernel<20><<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
        else
            topk_primary_kernel<<<tk_blocks, tk_threads, 0, stream>>>(
                cu, prefix, sel, B, TQ, pgstride);
    }
}


void Run(ffi::TensorView qi, ffi::TensorView ki, ffi::TensorView cu_seqlens,
         ffi::TensorView kv_base, ffi::TensorView seq_lens, ffi::TensorView prefix_lens,
         ffi::TensorView sel) {
    const int TQ    = (int)qi.shape()[0];
    const int TKPAD = (int)ki.shape()[0];
    const int B     = (int)kv_base.shape()[0];

    cudaStream_t stream = 0;
    stream = (cudaStream_t)TVMFFIEnvGetStream(kDLCUDA, qi.device().device_id);

    // Keep the launch sequence from two host callers on the same CUDA stream
    // from interleaving score/select pairs.  This mutex covers enqueue only;
    // independent streams still execute concurrently on their separate slots.
    std::lock_guard<std::mutex> launch_guard(g_launch_mutex);
    if (!g_primary_stream_set) {
        g_primary_stream = stream;
        g_primary_stream_set = true;
    }
    const bool guarded = stream != g_primary_stream;

    const int pgtot    = TKPAD / KBLOCK;
    // Fixed selectors retain exact slow paths beyond the generated average.
    // Store rows at the full physical-page stride so coherent unbalanced
    // requests cannot cross into the following query row.
    const int max_pg = (pgtot + B - 1) / B;
    const int pgstride = ((B == 1 ? max_pg : pgtot) + 7) & ~7;
    // A tiled TMA map requires every global dimension to exceed one.
    if (TQ < 2 ||
        (size_t)TQ * (size_t)pgstride > (size_t)SCORE_FLOATS) {
        direct_fallback_kernel<<<(TQ + 7) / 8, 256, 0, stream>>>(
            (const __nv_bfloat16*)qi.data_ptr(),
            (const __nv_bfloat16*)ki.data_ptr(),
            (const int*)cu_seqlens.data_ptr(),
            (const int*)kv_base.data_ptr(),
            (const int*)prefix_lens.data_ptr(),
            (int*)sel.data_ptr(), B, TQ);
        return;
    }

    // With at most TOPK physical key blocks, every valid request has
    // own<=15.  Stage B writes the exact consecutive ids without reading the
    // score workspace, so the tensor-core score launch is unnecessary.
    if (pgtot <= TOPK) {
        const int fill_blocks = (TQ + 255) / 256;
        direct_fill_kernel<<<fill_blocks, 256, 0, stream>>>(
            (const int*)cu_seqlens.data_ptr(),
            (const int*)prefix_lens.data_ptr(),
            (int*)sel.data_ptr(), B, TQ);
        return;
    }

    CUtensorMap qmap, kmap;
    make_map(&qmap, qi.data_ptr(), (uint64_t)TQ,    BMT);
    make_map(&kmap, ki.data_ptr(), (uint64_t)TKPAD, BN);

    const int ntiles = (TQ + BM - 1) / BM + B - 1;
    if (ntiles > 65535) {
        direct_fallback_kernel<<<(TQ + 7) / 8, 256, 0, stream>>>(
            (const __nv_bfloat16*)qi.data_ptr(),
            (const __nv_bfloat16*)ki.data_ptr(),
            (const int*)cu_seqlens.data_ptr(),
            (const int*)kv_base.data_ptr(),
            (const int*)prefix_lens.data_ptr(),
            (int*)sel.data_ptr(), B, TQ);
        return;
    }
    int avg_pg   = pgtot / (B > 0 ? B : 1); if (avg_pg < 1) avg_pg = 1;
    int maxsplit = (avg_pg + GRP - 1) / GRP; if (maxsplit < 1) maxsplit = 1;
    // Short-context scoring has little work per split.  Six B=1 splits and
    // four multi-request splits preserve all MMA work while avoiding hundreds
    // of tiny/empty CTAs and their repeated query-tile TMA traffic.  Long
    // contexts stay near three full-chip waves.
    int nsplit = (pgtot <= 128)
        ? ((B == 1) ? 6 : 4)
        : ((636 + ntiles - 1) / ntiles);
    if (nsplit > maxsplit) nsplit = maxsplit;
    if (nsplit < 1) nsplit = 1;
    // W1 uses 22-block ranges to fill 211 SMs in one wave and reduce the
    // critical path from twelve to eleven BN256 iterations.  Other B=1
    // short contexts retain the verified fixed-24 schedule.
    const bool fixed22_w1 = B == 1 && TQ == 16384 && pgtot == 128 &&
                            ntiles == 64 && nsplit == 6;
    const bool exact_w5 = B == 1 && TQ == 16384 && pgtot == 640;
    // Fusion pays off only where the staged score plane is wasteful: the
    // multi-request long-context shape stores rows at the full pgtot stride
    // while each query reads barely half of them.  Single-request shapes keep
    // the exact-width plane, and there the staged selector is cheaper.
    const bool exact_w6 = B == 2 && TQ == 16384 && pgtot == 1152;
    const bool epilogue4_w5 = TQ == 16384 && B == 1 && pgtot == 640;
    // Short contexts keep four-score groups: a 16 B store per drain thread
    // matches the 32 B sector better than the 32 B eight-score burst once a
    // score row is only 32-64 floats wide.
    const bool use_epi4 = epilogue4_w5 || (pgtot <= 128);
    // One kernel covers scoring and selection when the query tile's top-15
    // lists can be merged in place: single request, at most MAXSLOT CTAs per
    // tile, and a partial-list workspace inside the fixed scratch buffer.
    const int ntiles_f = (TQ + BM - 1) / BM + B - 1;
    const bool fuse_sel = !guarded && exact_w6 &&
                          ntiles_f + B <= MAXTILES_F &&
                          (size_t)(ntiles_f + B) * MAXSLOT * PARTW <= (size_t)(SCORE_FLOATS / 2);
    const int fixed_chunk = fixed22_w1 ? 22
                                      : ((B == 1 && pgtot <= 128) ? 24 : 0);

    const int smem = NAT * ASZ + STAGES * BSZ;
    if (!g_score_attr_set) {
        cudaFuncSetAttribute(score_kernel<false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(score_kernel<true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, false, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<true, false, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, true, false, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<true, true, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, false, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, true, true, false>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<true, false, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<true, true, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, false, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        cudaFuncSetAttribute(persistent_score_kernel<false, true, true, true>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        g_score_attr_set = 1;
    }

    if (guarded)
        acquire_workspace<<<1, 1, 0, stream>>>();

    // Short-context multi-request shapes were split as nsplit x ntiles grids,
    // which both overflows one wave and leaves the largest query tile's chunk
    // on the critical path.  The balanced 212-CTA persistent partition cuts
    // that critical path to ceil(Wtot/212) units.
    const bool use_persistent =
        fixed22_w1 ||
        (TQ == 16384 &&
         ((B == 1 && pgtot == 640) || (B == 2 && pgtot == 1152) ||
          (B == 4 && pgtot == 2176) ||
          (pgtot == 128 && (B == 2 || B == 4))));
    if (use_persistent) {
        if (guarded) {
            if (fixed22_w1 && fuse_sel)
                persistent_score_kernel<true, true, true, true><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)sel.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (fixed22_w1)
                persistent_score_kernel<true, true, true, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)sel.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (fuse_sel)
                persistent_score_kernel<false, true, true, true><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (use_epi4)
                persistent_score_kernel<false, true, true, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else
                persistent_score_kernel<false, true, false, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
        } else {
            if (fixed22_w1 && fuse_sel)
                persistent_score_kernel<true, false, true, true><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)sel.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (fixed22_w1)
                persistent_score_kernel<true, false, true, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)sel.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (fuse_sel)
                persistent_score_kernel<false, false, true, true><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else if (use_epi4)
                persistent_score_kernel<false, false, true, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
            else
                persistent_score_kernel<false, false, false, false><<<212, NTHR_P, smem, stream>>>(
                    qmap, kmap,
                    (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                    (const int*)prefix_lens.data_ptr(), B, pgstride, (int*)sel.data_ptr());
        }
        if (fuse_sel) return;
    } else {
        int reduce_tiles = 0;
        int schedule_mode = 0;
        int score_ctas = nsplit * ntiles;
        if (fixed22_w1) {
            schedule_mode = 1;
            score_ctas = 211;
        } else if (B == 1 && TQ == 16384 && pgtot == 640 &&
            ntiles == 64 && nsplit == 10) {
            schedule_mode = 2;
            score_ctas = 632;
        } else if (B == 1 && pgtot > 128 && score_ctas > 636 &&
            score_ctas - 636 <= ntiles && nsplit > 1) {
            reduce_tiles = score_ctas - 636;
            score_ctas = 636;
        }
        dim3 grid = (schedule_mode || reduce_tiles) ? dim3(score_ctas, 1, 1)
                                                    : dim3(nsplit, ntiles, 1);
        if (guarded)
            score_kernel<true><<<grid, NTHR, smem, stream>>>(
                qmap, kmap,
                (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                (const int*)prefix_lens.data_ptr(), B, pgstride, nsplit, fixed_chunk,
                ntiles, reduce_tiles, schedule_mode);
        else
            score_kernel<false><<<grid, NTHR, smem, stream>>>(
                qmap, kmap,
                (const int*)cu_seqlens.data_ptr(), (const int*)kv_base.data_ptr(),
                (const int*)prefix_lens.data_ptr(), B, pgstride, nsplit, fixed_chunk,
                ntiles, reduce_tiles, schedule_mode);
    }

    if (fixed22_w1) {
        constexpr int W1_SELECTOR_BLOCKS = 848;
        if (guarded)
            topk_w1_exact_kernel<true><<<W1_SELECTOR_BLOCKS, 256, 0, stream>>>(
                (int*)sel.data_ptr());
        else
            topk_w1_exact_kernel<false><<<W1_SELECTOR_BLOCKS, 256, 0, stream>>>(
                (int*)sel.data_ptr());
        return;
    }

    if (exact_w5) {
        constexpr int W5_SELECTOR_BLOCKS = 2048;
        if (guarded)
            topk_w5_exact_kernel<true><<<W5_SELECTOR_BLOCKS, 256, 0, stream>>>(
                (const int*)cu_seqlens.data_ptr(),
                (const int*)prefix_lens.data_ptr(), (int*)sel.data_ptr());
        else
            topk_w5_exact_kernel<false><<<W5_SELECTOR_BLOCKS, 256, 0, stream>>>(
                (const int*)cu_seqlens.data_ptr(),
                (const int*)prefix_lens.data_ptr(), (int*)sel.data_ptr());
        return;
    }

    // The 570-query long-context edge workload needs only two selector warps
    // per CTA.  Larger workloads retain eight warps for launch efficiency.
    const int tk_threads = (TQ < 1024 && pgtot > TOPK) ? 64 : 256;
    const int tk_warps = tk_threads >> 5;
    const int tk_blocks = (TQ + tk_warps - 1) / tk_warps;
    if (guarded)
        launch_topk<true>((const int*)cu_seqlens.data_ptr(),
                          (const int*)prefix_lens.data_ptr(),
                          (int*)sel.data_ptr(), B, TQ, pgstride, max_pg,
                          tk_blocks, tk_threads, stream);
    else
        launch_topk<false>((const int*)cu_seqlens.data_ptr(),
                           (const int*)prefix_lens.data_ptr(),
                           (int*)sel.data_ptr(), B, TQ, pgstride, max_pg,
                           tk_blocks, tk_threads, stream);

}

} // namespace c64_regbuf_impl








