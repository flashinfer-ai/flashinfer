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
#define SMEM_SLOTS_OFF 1024
#define SMEM_SLOTS_STAGE_BYTES 256
#define SMEM_SLOTS_STRIDE 256
#define SMEM_TOTAL 1280
#define THREADS 64
#define LAUNCH_BOUNDS_THREADS 64

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



__device__ __forceinline__ __nv_bfloat162 __as_bf16x2(unsigned int v) {
    __nv_bfloat162_raw raw;
    raw.x = static_cast<unsigned short>(v);
    raw.y = static_cast<unsigned short>(v >> 16);
    return __nv_bfloat162(raw);
}

__device__ __forceinline__ float2 cake_bf16x2_to_f32x2(__nv_bfloat162 v) {
    return __bfloat1622float2(v);
}

extern "C" {

__global__ __launch_bounds__(LAUNCH_BOUNDS_THREADS) __cluster_dims__(16,1,1) void
kernel_cake_muse_spark_norm_cbd917375dce308daa21(const __nv_bfloat16* branch, const float* residual, const float* __restrict__ alpha, const float* __restrict__ beta, const __nv_bfloat16* __restrict__ next_weight, const __nv_bfloat16* __restrict__ post_weight, float* residual_out, __nv_bfloat16* ffn_input, int M, float post_eps, float next_eps, float post_gain_center, float next_gain_center, float alpha_value)
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
    #define bar_a_addr (mbar_base + 0)
    #define bar_b_addr (mbar_base + 8)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 16;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 16;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* slots = reinterpret_cast<float*>(smem_raw + 1024);
    const int slots_addr = smem + 1024;

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 2 barriers)
    // Mbarriers at smem_raw[0..16)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // bar_a: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // bar_b: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // === Task calls (declaration order) ===
    int warp_0 = warp;
    int lane_1 = lane;
    int rank = cta_rank;
    int first = cluster_id;
    int stride = num_clusters;
    int col0 = (rank * 64 + tid) * 8;
    long long col64 = (long long)col0;
    int peer = ((lane_1) < (15) ? (lane_1) : (15));
    uint32_t _mapa_0;
    asm volatile(
        "mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(_mapa_0) : "r"(slots_addr + (unsigned int)((rank * 2 + warp_0) * 4)), "r"(peer));
    float al[8];
    float be[8];
    unsigned int nw[4];
    unsigned int pw[4];
    {
        unsigned _ldv8_0_0;
        unsigned _ldv8_0_1;
        unsigned _ldv8_0_2;
        unsigned _ldv8_0_3;
        unsigned _ldv8_0_4;
        unsigned _ldv8_0_5;
        unsigned _ldv8_0_6;
        unsigned _ldv8_0_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(alpha + col0 + (0))) : "memory");
        al[0 + 0] = __uint_as_float(_ldv8_0_0);
        al[0 + 1] = __uint_as_float(_ldv8_0_1);
        al[0 + 2] = __uint_as_float(_ldv8_0_2);
        al[0 + 3] = __uint_as_float(_ldv8_0_3);
        al[0 + 4] = __uint_as_float(_ldv8_0_4);
        al[0 + 5] = __uint_as_float(_ldv8_0_5);
        al[0 + 6] = __uint_as_float(_ldv8_0_6);
        al[0 + 7] = __uint_as_float(_ldv8_0_7);
    }
    {
        unsigned _ldv8_1_0;
        unsigned _ldv8_1_1;
        unsigned _ldv8_1_2;
        unsigned _ldv8_1_3;
        unsigned _ldv8_1_4;
        unsigned _ldv8_1_5;
        unsigned _ldv8_1_6;
        unsigned _ldv8_1_7;
        asm volatile(
            "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
            : "=r"(_ldv8_1_0), "=r"(_ldv8_1_1), "=r"(_ldv8_1_2), "=r"(_ldv8_1_3), "=r"(_ldv8_1_4), "=r"(_ldv8_1_5), "=r"(_ldv8_1_6), "=r"(_ldv8_1_7) : "l"((const void*)(beta + col0 + (0))) : "memory");
        be[0 + 0] = __uint_as_float(_ldv8_1_0);
        be[0 + 1] = __uint_as_float(_ldv8_1_1);
        be[0 + 2] = __uint_as_float(_ldv8_1_2);
        be[0 + 3] = __uint_as_float(_ldv8_1_3);
        be[0 + 4] = __uint_as_float(_ldv8_1_4);
        be[0 + 5] = __uint_as_float(_ldv8_1_5);
        be[0 + 6] = __uint_as_float(_ldv8_1_6);
        be[0 + 7] = __uint_as_float(_ldv8_1_7);
    }
    {
        const uint4* _vptr_2 = reinterpret_cast<const uint4*>(next_weight + col0 + 0);
        uint4* _vdst_2 = reinterpret_cast<uint4*>(&nw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_2[_blk] = _vptr_2[_blk];
        }
    }
    {
        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(post_weight + col0 + 0);
        uint4* _vdst_3 = reinterpret_cast<uint4*>(&pw[0]);
        #pragma unroll
        for (int _blk = 0; _blk < 1; _blk++) {
            _vdst_3[_blk] = _vptr_3[_blk];
        }
    }
    unsigned int cur_b[4];
    float cur_r[8];
    unsigned int nxt_b[4];
    float nxt_r[8];
    long long fbase = (long long)first * 8192 + col64;
    if (first < M) {
        {
            const uint4* _vptr_4 = reinterpret_cast<const uint4*>(branch + fbase + 0);
            uint4* _vdst_4 = reinterpret_cast<uint4*>(&cur_b[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 1; _blk++) {
                _vdst_4[_blk] = _vptr_4[_blk];
            }
        }
        {
            unsigned _ldv8_5_0;
            unsigned _ldv8_5_1;
            unsigned _ldv8_5_2;
            unsigned _ldv8_5_3;
            unsigned _ldv8_5_4;
            unsigned _ldv8_5_5;
            unsigned _ldv8_5_6;
            unsigned _ldv8_5_7;
            asm volatile(
                "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_5_0), "=r"(_ldv8_5_1), "=r"(_ldv8_5_2), "=r"(_ldv8_5_3), "=r"(_ldv8_5_4), "=r"(_ldv8_5_5), "=r"(_ldv8_5_6), "=r"(_ldv8_5_7) : "l"((const void*)(residual + fbase + (0))) : "memory");
            cur_r[0 + 0] = __uint_as_float(_ldv8_5_0);
            cur_r[0 + 1] = __uint_as_float(_ldv8_5_1);
            cur_r[0 + 2] = __uint_as_float(_ldv8_5_2);
            cur_r[0 + 3] = __uint_as_float(_ldv8_5_3);
            cur_r[0 + 4] = __uint_as_float(_ldv8_5_4);
            cur_r[0 + 5] = __uint_as_float(_ldv8_5_5);
            cur_r[0 + 6] = __uint_as_float(_ldv8_5_6);
            cur_r[0 + 7] = __uint_as_float(_ldv8_5_7);
        }
    }
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    unsigned int _phase_bar_a_0 = 0;
    unsigned int _phase_bar_b_0 = 0;
    #pragma unroll 1
    for (int row0 = first; row0 < M; row0 += stride) {
        int row = row0;
        long long rbase = (long long)row * 8192 + col64;
        float bb[8];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float2 _cvt_f32_0 = cake_bf16x2_to_f32x2(__as_bf16x2(cur_b[k]));
            float2 _cvt_f32_1 = cake_bf16x2_to_f32x2(__as_bf16x2(pw[k]));
            bb[2 * k] = _cvt_f32_0.x * (_cvt_f32_1.x + post_gain_center);
            bb[2 * k + 1] = _cvt_f32_0.y * (_cvt_f32_1.y + post_gain_center);
        }
        float pa = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            pa = __fmaf_rn(bb[j], bb[j], pa);
        }
        float warp_reduce = pa;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce += __shfl_xor_sync(0xFFFFFFFF, warp_reduce, offset);
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(bar_a_addr, 128);
            }
        }
        if (lane_1 < 16) {
            uint32_t _mbar_peer_6;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_6) : "r"(bar_a_addr), "r"(peer));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_mapa_0), "f"(warp_reduce), "r"(_mbar_peer_6) : "memory");
        }
        int nrow = row + stride;
        if (nrow < M) {
            long long nbase = (long long)nrow * 8192 + col64;
            {
                const uint4* _vptr_7 = reinterpret_cast<const uint4*>(branch + nbase + 0);
                uint4* _vdst_7 = reinterpret_cast<uint4*>(&nxt_b[0]);
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vdst_7[_blk] = _vptr_7[_blk];
                }
            }
            {
                unsigned _ldv8_8_0;
                unsigned _ldv8_8_1;
                unsigned _ldv8_8_2;
                unsigned _ldv8_8_3;
                unsigned _ldv8_8_4;
                unsigned _ldv8_8_5;
                unsigned _ldv8_8_6;
                unsigned _ldv8_8_7;
                asm volatile(
                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(_ldv8_8_0), "=r"(_ldv8_8_1), "=r"(_ldv8_8_2), "=r"(_ldv8_8_3), "=r"(_ldv8_8_4), "=r"(_ldv8_8_5), "=r"(_ldv8_8_6), "=r"(_ldv8_8_7) : "l"((const void*)(residual + nbase + (0))) : "memory");
                nxt_r[0 + 0] = __uint_as_float(_ldv8_8_0);
                nxt_r[0 + 1] = __uint_as_float(_ldv8_8_1);
                nxt_r[0 + 2] = __uint_as_float(_ldv8_8_2);
                nxt_r[0 + 3] = __uint_as_float(_ldv8_8_3);
                nxt_r[0 + 4] = __uint_as_float(_ldv8_8_4);
                nxt_r[0 + 5] = __uint_as_float(_ldv8_8_5);
                nxt_r[0 + 6] = __uint_as_float(_ldv8_8_6);
                nxt_r[0 + 7] = __uint_as_float(_ldv8_8_7);
            }
        }
        #pragma unroll
        for (int j_1 = 0; j_1 < 8; j_1++) {
            cur_r[j_1] = al[j_1] * cur_r[j_1];
            bb[j_1] = be[j_1] * bb[j_1];
        }
        mbarrier_wait(bar_a_addr, _phase_bar_a_0);
        _phase_bar_a_0 ^= 1;
        int idx = ((lane_1) < (31) ? (lane_1) : (31));
        float value = slots[idx];
        float _warp_reduce_10 = value;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_10 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_10, offset);
        float _sqrt_9;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_9) : "f"(__fmaf_rn(_warp_reduce_10, 0.00012207031f, post_eps)));
        float r1 = __frcp_rn(_sqrt_9);
        float p2 = 0.0f;
        #pragma unroll
        for (int j_2 = 0; j_2 < 8; j_2++) {
            cur_r[j_2] = __fmaf_rn(bb[j_2], r1, cur_r[j_2]);
            p2 = __fmaf_rn(cur_r[j_2], cur_r[j_2], p2);
        }
        float warp_reduce_0 = p2;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            warp_reduce_0 += __shfl_xor_sync(0xFFFFFFFF, warp_reduce_0, offset);
        if (warp == 0) {
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(bar_b_addr, 128);
            }
        }
        if (lane_1 < 16) {
            uint32_t _mbar_peer_11;
            asm(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mbar_peer_11) : "r"(bar_b_addr), "r"(peer));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_mapa_0 + 128), "f"(warp_reduce_0), "r"(_mbar_peer_11) : "memory");
        }
        {
            unsigned _stv8_12_0 = __float_as_uint(cur_r[0 + 0]);
            unsigned _stv8_12_1 = __float_as_uint(cur_r[0 + 1]);
            unsigned _stv8_12_2 = __float_as_uint(cur_r[0 + 2]);
            unsigned _stv8_12_3 = __float_as_uint(cur_r[0 + 3]);
            unsigned _stv8_12_4 = __float_as_uint(cur_r[0 + 4]);
            unsigned _stv8_12_5 = __float_as_uint(cur_r[0 + 5]);
            unsigned _stv8_12_6 = __float_as_uint(cur_r[0 + 6]);
            unsigned _stv8_12_7 = __float_as_uint(cur_r[0 + 7]);
            asm volatile(
                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"((void*)(residual_out + rbase + (0))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
        }
        #pragma unroll
        for (int k_1 = 0; k_1 < 4; k_1++) {
            float2 _cvt_f32_2 = cake_bf16x2_to_f32x2(__as_bf16x2(nw[k_1]));
            bb[2 * k_1] = _cvt_f32_2.x + next_gain_center;
            bb[2 * k_1 + 1] = _cvt_f32_2.y + next_gain_center;
        }
        mbarrier_wait(bar_b_addr, _phase_bar_b_0);
        _phase_bar_b_0 ^= 1;
        int idx_0 = ((lane_1) < (31) ? (lane_1) : (31));
        float value_1 = slots[32 + idx_0];
        float _warp_reduce_14 = value_1;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            _warp_reduce_14 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_14, offset);
        float _sqrt_13;
        asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_13) : "f"(__fmaf_rn(_warp_reduce_14, 0.00012207031f, next_eps)));
        float r2 = __frcp_rn(_sqrt_13);
        #pragma unroll
        for (int j_3 = 0; j_3 < 8; j_3++) {
            bb[j_3] = cur_r[j_3] * r2 * bb[j_3];
        }
        unsigned int _packed_0[4];
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 0], bb[_lp*2+1 + 0]));
            _packed_0[_lp] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 2], bb[_lp*2+1 + 2]));
            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 4], bb[_lp*2+1 + 4]));
            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
        }
        #pragma unroll
        for (int _lp = 0; _lp < 1; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(bb[_lp*2 + 6], bb[_lp*2+1 + 6]));
            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
        }
        reinterpret_cast<int4*>(ffn_input + rbase)[0] = reinterpret_cast<int4*>(_packed_0)[0];
        #pragma unroll
        for (int k_2 = 0; k_2 < 4; k_2++) {
            cur_b[k_2] = nxt_b[k_2];
        }
        #pragma unroll
        for (int j_4 = 0; j_4 < 8; j_4++) {
            cur_r[j_4] = nxt_r[j_4];
        }
    }
}

} // extern "C"
