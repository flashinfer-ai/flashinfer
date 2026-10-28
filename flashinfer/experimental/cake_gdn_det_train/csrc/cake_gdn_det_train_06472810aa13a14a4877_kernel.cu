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
#define NUM_ONE_STAGES 1
#define SMEM_S_Q_OFF 0
#define SMEM_S_Q_STAGE_BYTES 16384
#define SMEM_S_Q_STRIDE 16384
#define SMEM_S_K_OFF 16384
#define SMEM_S_K_STAGE_BYTES 16384
#define SMEM_S_K_STRIDE 16384
#define SMEM_TOTAL 32896
#define THREADS 288
#define LAUNCH_MIN_BLOCKS 6

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





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(288, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_det_train_06472810aa13a14a4877(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, float* __restrict__ g, float* __restrict__ beta, float* __restrict__ beta32_out, __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float* __restrict__ rstd_q, float* __restrict__ rstd_k, float* __restrict__ g_cs, int* __restrict__ chunk_start, int* __restrict__ chunk_len, int num_heads, int num_v_heads, int normalize_qk)
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

    const int mbar_base = smem + 32768;
    #define full_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* s_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int s_q_addr = smem + 0;
    __nv_bfloat16* s_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 16384);
    const int s_k_addr = smem + 16384;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[32768..32776)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // full: 1 barriers, init_count=1
            mbarrier_init(smem + 32768, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: rows ----
    if (warp <= 7) {
        { // rows_main
            int lin_r = blockIdx.x + blockIdx.y * gridDim.x;
            int chunk_r = lin_r / num_heads;
            int hq_r = lin_r - chunk_r * num_heads;
            int tok0_r = chunk_start[chunk_r];
            int n_valid_r = chunk_len[chunk_r];
            int lane_r = lane;
            mbarrier_wait(full_addr, 0);
            #pragma unroll
            for (int step = 0; step < 8; step++) {
                int row = warp + 8 * step;
                unsigned int off = (unsigned int)(row * 256 + lane_r * 8);
                unsigned int words[4];
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1]))
                    : "r"(s_q_addr + off));
                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(2) + 1]))
                    : "r"(s_k_addr + off));
                float q_frag[4];
                float k_frag[4];
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&q_frag[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[_pair]) << 16);
                    (&q_frag[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[_pair]) & 0xffff0000u);
                }
                #pragma unroll
                for (int _pair = 0; _pair < 2; _pair++) {
                    (&k_frag[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) << 16);
                    (&k_frag[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(words[2 + _pair]) & 0xffff0000u);
                }
                float q_sq = 0.0f;
                float k_sq = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; i++) {
                    q_sq = __fmaf_rn(q_frag[i], q_frag[i], q_sq);
                    k_sq = __fmaf_rn(k_frag[i], k_frag[i], k_sq);
                }
                float _warp_reduce_0 = q_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset);
                q_sq = _warp_reduce_0;
                float _warp_reduce_1 = k_sq;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                k_sq = _warp_reduce_1;
                float _rsqrt_2 = rsqrtf(q_sq + 1e-06f);
                float q_inv = ((normalize_qk != 0) ? _rsqrt_2 : 1.0f);
                float _rsqrt_3 = rsqrtf(k_sq + 1e-06f);
                float k_inv = ((normalize_qk != 0) ? _rsqrt_3 : 1.0f);
                #pragma unroll
                for (int i2 = 0; i2 < 4; i2++) {
                    q_frag[i2] = q_frag[i2] * q_inv;
                    k_frag[i2] = k_frag[i2] * k_inv;
                }
                unsigned int _packed_0[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_frag[_lp*2 + 0], q_frag[_lp*2+1 + 0]));
                    _packed_0[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(q_frag[_lp*2 + 2], q_frag[_lp*2+1 + 2]));
                    _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                }
                unsigned int _packed_1[2];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_frag[_lp*2 + 0], k_frag[_lp*2+1 + 0]));
                    _packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(k_frag[_lp*2 + 2], k_frag[_lp*2+1 + 2]));
                    _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                if (row < n_valid_r) {
                    long long rstd_index = (long long)(tok0_r + row) * (long long)num_heads + (long long)hq_r;
                    long long base = rstd_index * 128 + (long long)(lane_r * 4);
                    reinterpret_cast<int2*>(qn + (base))[0] = reinterpret_cast<int2*>(_packed_0)[0];
                    reinterpret_cast<int2*>(kn + (base))[0] = reinterpret_cast<int2*>(_packed_1)[0];
                    if (lane_r == 0) {
                        rstd_q[rstd_index] = q_inv;
                        rstd_k[rstd_index] = k_inv;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 8) {
        { // load_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&q_tma))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&k_tma))) : "memory");
            }
            int lin = blockIdx.x + blockIdx.y * gridDim.x;
            int chunk = lin / num_heads;
            int hq = lin - chunk * num_heads;
            int tok0 = chunk_start[chunk];
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(full_addr, 32768);
                tma_3d_gmem2smem(s_q_addr, (&q_tma), 0, hq, tok0, full_addr);
                tma_3d_gmem2smem(s_k_addr, (&k_tma), 0, hq, tok0, full_addr);
            }
            int n_valid = chunk_len[chunk];
            int group = num_v_heads / num_heads;
            int lane_0 = lane;
            #pragma unroll 1
            for (int j = 0; j < group; j += 2) {
                int hv_a = hq * group + j;
                int hv_b = hv_a;
                if (j + 1 < group) {
                    hv_b = hv_a + 1;
                }
                long long index_a_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv_a;
                long long index_a_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv_a;
                long long index_b_lo = (long long)(tok0 + lane_0) * (long long)num_v_heads + (long long)hv_b;
                long long index_b_hi = (long long)(tok0 + lane_0 + 32) * (long long)num_v_heads + (long long)hv_b;
                float ga_lo = 0.0f;
                float ga_hi = 0.0f;
                float gb_lo = 0.0f;
                float gb_hi = 0.0f;
                float ba_lo = 0.0f;
                float ba_hi = 0.0f;
                float bb_lo = 0.0f;
                float bb_hi = 0.0f;
                if (lane_0 < n_valid) {
                    ga_lo = g[index_a_lo];
                    gb_lo = g[index_b_lo];
                }
                if (n_valid > lane_0 + 32) {
                    ga_hi = g[index_a_hi];
                    gb_hi = g[index_b_hi];
                }
                #pragma unroll
                for (int delta = 0; delta < 5; delta++) {
                    const int shift = 1 << delta;
                    float up_a_lo = __shfl_up_sync(0xFFFFFFFF, ga_lo, shift, 32);
                    float up_a_hi = __shfl_up_sync(0xFFFFFFFF, ga_hi, shift, 32);
                    float up_b_lo = __shfl_up_sync(0xFFFFFFFF, gb_lo, shift, 32);
                    float up_b_hi = __shfl_up_sync(0xFFFFFFFF, gb_hi, shift, 32);
                    if (lane_0 >= shift) {
                        ga_lo = ga_lo + up_a_lo;
                        ga_hi = ga_hi + up_a_hi;
                        gb_lo = gb_lo + up_b_lo;
                        gb_hi = gb_hi + up_b_hi;
                    }
                }
                float total_a_lo = __shfl_sync(0xFFFFFFFF, ga_lo, 31);
                float total_b_lo = __shfl_sync(0xFFFFFFFF, gb_lo, 31);
                ga_hi = ga_hi + total_a_lo;
                gb_hi = gb_hi + total_b_lo;
                if (lane_0 < n_valid) {
                    g_cs[index_a_lo] = ga_lo * 1.442695f;
                    if (j + 1 < group) {
                        g_cs[index_b_lo] = gb_lo * 1.442695f;
                    }
                }
                if (n_valid > lane_0 + 32) {
                    g_cs[index_a_hi] = ga_hi * 1.442695f;
                    if (j + 1 < group) {
                        g_cs[index_b_hi] = gb_hi * 1.442695f;
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
