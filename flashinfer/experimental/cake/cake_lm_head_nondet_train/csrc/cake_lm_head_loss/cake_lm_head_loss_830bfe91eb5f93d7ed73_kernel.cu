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


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_lm_head_loss_830bfe91eb5f93d7ed73(__nv_bfloat16* __restrict__ W, __nv_bfloat16* __restrict__ X, long long* __restrict__ labels, __nv_bfloat16* __restrict__ Z, float* __restrict__ PU, float* __restrict__ PMS, int V, int ldx, int rpc)
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
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* wbuf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int wbuf_addr = smem + 1024;
    float* red = reinterpret_cast<float*>(smem_raw + 197632);
    const int red_addr = smem + 197632;

    // Mbarrier init (8 barriers at smem_raw[0..64))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
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

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            int v0 = bid * rpc;
            int _min_0 = ((V - v0) < (rpc) ? (V - v0) : (rpc));
            int n_rows = ((_min_0) > (0) ? (_min_0) : (0));
            long long _max_1 = ((labels[0]) > (labels[1]) ? (labels[0]) : (labels[1]));
            int live = ((_max_1 >= 0ll) ? 1 : 0);
            int n_tiles = (n_rows + 3) / 4 * live;
            unsigned int slot = 0;
            unsigned int _phase_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int k = 0; k < n_tiles; k++) {
                    int _min_2 = ((n_rows - k * 4) < (4) ? (n_rows - k * 4) : (4));
                    int nr = ((_min_2) > (1) ? (_min_2) : (1));
                    mbarrier_wait(empty_addr + (slot) * 8, _phase_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive_expect_tx(full_addr + (slot) * 8, (unsigned int)(nr * 12288));
                    cp_async_bulk_gmem2smem(wbuf_addr + slot * 49152, reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(W) + (unsigned long long)(v0 + k * 4) * 12288ull), (unsigned int)(nr * 12288), full_addr + (slot) * 8);
                    slot += 1;
                    if (slot == 4) { slot = 0; _phase_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: work ----
    if (warp >= 1 && warp <= 8) {
        { // work_main
            int v0_1 = bid * rpc;
            int _min_0_1 = ((V - v0_1) < (rpc) ? (V - v0_1) : (rpc));
            int n_rows_1 = ((_min_0_1) > (0) ? (_min_0_1) : (0));
            long long _max_1_1 = ((labels[0]) > (labels[1]) ? (labels[0]) : (labels[1]));
            int live_1 = ((_max_1_1 >= 0ll) ? 1 : 0);
            int n_tiles_1 = (n_rows_1 + 3) / 4 * live_1;
            int c = tid - 32;
            int cw = warp - 1;
            unsigned int slot_1 = 0;
            float xf[48];
            #pragma unroll
            for (int t = 0; t < 2; t++) {
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    unsigned int xw[4];
                    {
                        uint4 _uv4_2 = *reinterpret_cast<const uint4*>(X + ((unsigned long long)t * (unsigned long long)ldx + (unsigned long long)(c + 256 * j) * 8ull) + 0);
                        xw[0 + 0] = _uv4_2.x;
                        xw[0 + 1] = _uv4_2.y;
                        xw[0 + 2] = _uv4_2.z;
                        xw[0 + 3] = _uv4_2.w;
                    }
                    #pragma unroll
                    for (int e = 0; e < 4; e++) {
                        xf[(t * 3 + j) * 8 + 2 * e] = __uint_as_float(xw[e] << 16);
                        xf[(t * 3 + j) * 8 + 2 * e + 1] = __uint_as_float(xw[e] & 4294901760u);
                    }
                }
            }
            float mrow[2];
            float srow[2];
            #pragma unroll
            for (int t_1 = 0; t_1 < 2; t_1++) {
                mrow[t_1] = -CUDART_INF_F;
                srow[t_1] = 0.0f;
            }
            float u[48];
            #pragma unroll
            for (int e_1 = 0; e_1 < 48; e_1++) {
                u[e_1] = 0.0f;
            }
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int k_1 = 0; k_1 < n_tiles_1; k_1++) {
                int _min_3 = ((n_rows_1 - k_1 * 4) < (4) ? (n_rows_1 - k_1 * 4) : (4));
                int nr_1 = ((_min_3) > (1) ? (_min_3) : (1));
                int buf = k_1 % 2;
                mbarrier_wait(full_addr + (slot_1) * 8, _phase_full);
                unsigned int wpk[48];
                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    #pragma unroll
                    for (int j_1 = 0; j_1 < 3; j_1++) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&wpk[(r * 3 + j_1) * 4])), "=r"(*reinterpret_cast<uint32_t*>(&wpk[((r * 3 + j_1) * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&wpk[((r * 3 + j_1) * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&wpk[((r * 3 + j_1) * 4) + 3]))
                            : "r"(wbuf_addr + slot_1 * 49152 + (unsigned int)(r * 12288 + (c + 256 * j_1) * 16)));
                    }
                }
                mbarrier_arrive(empty_addr + (slot_1) * 8);
                float pd[8];
                #pragma unroll
                for (int t_2 = 0; t_2 < 2; t_2++) {
                    #pragma unroll
                    for (int r_1 = 0; r_1 < 4; r_1++) {
                        float acc = 0.0f;
                        #pragma unroll
                        for (int q = 0; q < 12; q++) {
                            acc = __fmaf_rn(__uint_as_float(wpk[r_1 * 3 * 4 + q] << 16), xf[t_2 * 3 * 8 + 2 * q], acc);
                            acc = __fmaf_rn(__uint_as_float(wpk[r_1 * 3 * 4 + q] & 4294901760u), xf[t_2 * 3 * 8 + 2 * q + 1], acc);
                        }
                        pd[t_2 * 4 + r_1] = acc;
                    }
                }
                bool upper_0 = (lane & 16) != 0;
                float send_0 = ((upper_0) ? pd[0] : pd[4]);
                float keep_0 = ((upper_0) ? pd[4] : pd[0]);
                float _shfl_xor_sync_4 = __shfl_xor_sync(0xFFFFFFFF, send_0, 16);
                pd[0] = keep_0 + _shfl_xor_sync_4;
                float send_1 = ((upper_0) ? pd[1] : pd[5]);
                float keep_1 = ((upper_0) ? pd[5] : pd[1]);
                float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, send_1, 16);
                pd[1] = keep_1 + _shfl_xor_sync_5;
                float send_2 = ((upper_0) ? pd[2] : pd[6]);
                float keep_2 = ((upper_0) ? pd[6] : pd[2]);
                float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, send_2, 16);
                pd[2] = keep_2 + _shfl_xor_sync_6;
                float send_3 = ((upper_0) ? pd[3] : pd[7]);
                float keep_3 = ((upper_0) ? pd[7] : pd[3]);
                float _shfl_xor_sync_7 = __shfl_xor_sync(0xFFFFFFFF, send_3, 16);
                pd[3] = keep_3 + _shfl_xor_sync_7;
                bool upper_1 = (lane & 8) != 0;
                float send_4 = ((upper_1) ? pd[0] : pd[2]);
                float keep_4 = ((upper_1) ? pd[2] : pd[0]);
                float _shfl_xor_sync_8 = __shfl_xor_sync(0xFFFFFFFF, send_4, 8);
                pd[0] = keep_4 + _shfl_xor_sync_8;
                float send_5 = ((upper_1) ? pd[1] : pd[3]);
                float keep_5 = ((upper_1) ? pd[3] : pd[1]);
                float _shfl_xor_sync_9 = __shfl_xor_sync(0xFFFFFFFF, send_5, 8);
                pd[1] = keep_5 + _shfl_xor_sync_9;
                bool upper_2 = (lane & 4) != 0;
                float send_6 = ((upper_2) ? pd[0] : pd[1]);
                float keep_6 = ((upper_2) ? pd[1] : pd[0]);
                float _shfl_xor_sync_10 = __shfl_xor_sync(0xFFFFFFFF, send_6, 4);
                pd[0] = keep_6 + _shfl_xor_sync_10;
                float _shfl_xor_sync_11 = __shfl_xor_sync(0xFFFFFFFF, pd[0], 2);
                pd[0] = pd[0] + _shfl_xor_sync_11;
                float _shfl_xor_sync_12 = __shfl_xor_sync(0xFFFFFFFF, pd[0], 1);
                pd[0] = pd[0] + _shfl_xor_sync_12;
                if (lane % 4 == 0) {
                    red[(buf * 8 + cw) * 8 + lane / 4] = pd[0];
                }
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                float tot[8];
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    float a = red[buf * 8 * 8 + i];
                    #pragma unroll
                    for (int w = 1; w < 8; w++) {
                        a = a + red[(buf * 8 + w) * 8 + i];
                    }
                    tot[i] = (float)(__nv_bfloat16)a;
                }
                if (c < 8) {
                    if (c % 4 < nr_1) {
                        float zc = red[buf * 8 * 8 + c];
                        #pragma unroll
                        for (int w_1 = 1; w_1 < 8; w_1++) {
                            zc = zc + red[(buf * 8 + w_1) * 8 + c];
                        }
                        Z[(unsigned long long)(c / 4) * (unsigned long long)V + (unsigned long long)(v0_1 + k_1 * 4 + c % 4)] = (__nv_bfloat16)zc;
                    }
                }
                #pragma unroll
                for (int t_3 = 0; t_3 < 2; t_3++) {
                    float m_old = mrow[t_3];
                    float mx = m_old;
                    #pragma unroll
                    for (int r_2 = 0; r_2 < 4; r_2++) {
                        if (r_2 < nr_1) {
                            mx = max_noftz(mx, tot[t_3 * 4 + r_2]);
                        }
                    }
                    float sc = 1.0f;
                    if (mx != m_old) {
                        float _exp_13 = expf(m_old - mx);
                        sc = _exp_13;
                    }
                    float ps = 0.0f;
                    #pragma unroll
                    for (int r_3 = 0; r_3 < 4; r_3++) {
                        float p = 0.0f;
                        if (r_3 < nr_1) {
                            float _exp_14 = expf(tot[t_3 * 4 + r_3] - mx);
                            p = _exp_14;
                        }
                        tot[t_3 * 4 + r_3] = p;
                        ps = ps + p;
                    }
                    srow[t_3] = __fmaf_rn(srow[t_3], sc, ps);
                    mrow[t_3] = mx;
                    if (mx != m_old) {
                        #pragma unroll
                        for (int e_2 = 0; e_2 < 24; e_2++) {
                            u[t_3 * 3 * 8 + e_2] = u[t_3 * 3 * 8 + e_2] * sc;
                        }
                    }
                    #pragma unroll
                    for (int r_4 = 0; r_4 < 4; r_4++) {
                        if (r_4 < nr_1) {
                            #pragma unroll
                            for (int q_1 = 0; q_1 < 12; q_1++) {
                                u[t_3 * 3 * 8 + 2 * q_1] = __fmaf_rn(tot[t_3 * 4 + r_4], __uint_as_float(wpk[r_4 * 3 * 4 + q_1] << 16), u[t_3 * 3 * 8 + 2 * q_1]);
                                u[t_3 * 3 * 8 + 2 * q_1 + 1] = __fmaf_rn(tot[t_3 * 4 + r_4], __uint_as_float(wpk[r_4 * 3 * 4 + q_1] & 4294901760u), u[t_3 * 3 * 8 + 2 * q_1 + 1]);
                            }
                        }
                    }
                }
                slot_1 += 1;
                if (slot_1 == 4) { slot_1 = 0; _phase_full ^= 1; }
            }
            if (c == 0) {
                #pragma unroll
                for (int t_4 = 0; t_4 < 2; t_4++) {
                    PMS[(bid * 2 + t_4) * 2] = mrow[t_4];
                    PMS[(bid * 2 + t_4) * 2 + 1] = srow[t_4];
                }
            }
            #pragma unroll
            for (int t_5 = 0; t_5 < 2; t_5++) {
                #pragma unroll
                for (int j_2 = 0; j_2 < 3; j_2++) {
                    unsigned long long off = (unsigned long long)(bid * 2 + t_5) * 6144ull + (unsigned long long)(c + 256 * j_2) * 8ull;
                    {
                        float4 _v4 = make_float4(u[(t_5 * 3 + j_2) * 8 + 0], u[(t_5 * 3 + j_2) * 8 + 1], u[(t_5 * 3 + j_2) * 8 + 2], u[(t_5 * 3 + j_2) * 8 + 3]);
                        *reinterpret_cast<float4*>((PU + off) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(u[(t_5 * 3 + j_2) * 8 + 4 + 0], u[(t_5 * 3 + j_2) * 8 + 4 + 1], u[(t_5 * 3 + j_2) * 8 + 4 + 2], u[(t_5 * 3 + j_2) * 8 + 4 + 3]);
                        *reinterpret_cast<float4*>((PU + (off + 4ull)) + 0) = _v4;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
