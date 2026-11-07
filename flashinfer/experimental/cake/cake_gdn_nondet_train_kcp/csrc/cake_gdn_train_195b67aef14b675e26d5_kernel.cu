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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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




__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ float mul_f32_rn_noftz(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

extern "C" {

__global__ __launch_bounds__(192, LAUNCH_MIN_BLOCKS) void
kernel_cake_gdn_train_195b67aef14b675e26d5(const __grid_constant__ CUtensorMap q_tma, const __grid_constant__ CUtensorMap k_tma, const __grid_constant__ CUtensorMap v_tma, float* __restrict__ g, float* __restrict__ beta, int* __restrict__ chunk_bos, int* __restrict__ chunk_len, __nv_bfloat16* __restrict__ w_out, __nv_bfloat16* __restrict__ u_out, __nv_bfloat16* __restrict__ aqk_out, __nv_bfloat16* __restrict__ akk, float* __restrict__ g_cum, float* __restrict__ rowscale, float* __restrict__ qscale, float* __restrict__ gamma, __nv_bfloat16* __restrict__ qn_out, __nv_bfloat16* __restrict__ kn_out, float* __restrict__ q_rstd, float* __restrict__ k_rstd, int rstd_tstride, int num_heads, int num_qk_heads, float scale)
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
    #define qk_full_addr (mbar_base + 0)
    #define gram_full_addr (mbar_base + 8)
    #define v_full_addr (mbar_base + 16)
    #define v_empty_addr (mbar_base + 32)
    #define op_ready_addr (mbar_base + 48)
    #define wu_full_addr (mbar_base + 56)
    #define qkn_ready_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Prologue
    __nv_bfloat16* q_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_p_addr = smem + 1024;
    __nv_bfloat16* k_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_p_addr = smem + 17408;
    __nv_bfloat16* v_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_p_addr = smem + 33792;
    __nv_bfloat16* q_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_k_addr = smem + 1024;
    __nv_bfloat16* k_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int k_k_addr = smem + 17408;
    __nv_bfloat16* kb_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int kb_mn_addr = smem + 1024;
    __nv_bfloat16* v_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_t_addr = smem + 33792;
    __nv_bfloat16* v_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int v_mn_addr = smem + 33792;
    __nv_bfloat16* akk_t = reinterpret_cast<__nv_bfloat16*>(smem_raw + 105472);
    const int akk_t_addr = smem + 105472;
    float* l_s = reinterpret_cast<float*>(smem_raw + 66560);
    const int l_s_addr = smem + 66560;
    float* a_s = reinterpret_cast<float*>(smem_raw + 83968);
    const int a_s_addr = smem + 83968;
    float* x_s = reinterpret_cast<float*>(smem_raw + 101376);
    const int x_s_addr = smem + 101376;
    float* gs_s = reinterpret_cast<float*>(smem_raw + 104448);
    const int gs_s_addr = smem + 104448;
    float* gc_s = reinterpret_cast<float*>(smem_raw + 104704);
    const int gc_s_addr = smem + 104704;
    float* eg_s = reinterpret_cast<float*>(smem_raw + 104960);
    const int eg_s_addr = smem + 104960;
    float* beta_s = reinterpret_cast<float*>(smem_raw + 105216);
    const int beta_s_addr = smem + 105216;

    // Kernel setup ops
    int chunk = blockIdx.x;
    int hq = blockIdx.y;
    int ext0 = chunk_bos[chunk];
    int clen = chunk_len[chunk];
    int group = num_heads / num_qk_heads;

    // Mbarrier init (9 barriers at smem_raw[0..72))

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'one' ---
            // qk_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // gram_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'vring' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'one' ---
            // op_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 48, 128);
            // wu_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // qkn_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 64, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 72);
    if (warp == 0) {
        int _tmem_hold = smem + 72;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const unsigned int taddr = tmem_addr_storage[0];
    const unsigned int tmem_acc = taddr;

    // ---- Role: compute ----
    if (warp <= 3) {
        { // compute_main
            int tid_0 = tid;
            int w_id = warp;
            int lane_1 = lane;
            int half = lane_1 >> 4;
            int t_row = w_id * 16 + (lane_1 & 15);
            int row_base = w_id * 32 << 16;
            int l_base = (int)l_s_addr;
            int a_base = (int)a_s_addr;
            int x_base = (int)x_s_addr;
            int gs_base = (int)gs_s_addr;
            int gc_base = (int)gc_s_addr;
            int eg_base = (int)eg_s_addr;
            int beta_base = (int)beta_s_addr;
            int irow0 = chunk * 64;
            unsigned int vst_c = 0;
            float zero16[16];
            zero16[0] = 0.0f;
            zero16[1] = 0.0f;
            zero16[2] = 0.0f;
            zero16[3] = 0.0f;
            zero16[4] = 0.0f;
            zero16[5] = 0.0f;
            zero16[6] = 0.0f;
            zero16[7] = 0.0f;
            zero16[8] = 0.0f;
            zero16[9] = 0.0f;
            zero16[10] = 0.0f;
            zero16[11] = 0.0f;
            zero16[12] = 0.0f;
            zero16[13] = 0.0f;
            zero16[14] = 0.0f;
            zero16[15] = 0.0f;
            unsigned int _phase_qk_full_0 = 0;
            unsigned int _phase_gram_full_0 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_wu_full_0 = 0;
            if (clen > 0) {
                float sqk[32];
                float gv_n = 0.0f;
                float bv_n = 0.0f;
                if (tid_0 < clen) {
                    long long tok0 = (long long)(ext0 + tid_0) * (long long)num_heads + (long long)(hq * group);
                    gv_n = g[tok0];
                    bv_n = beta[tok0];
                }
                mbarrier_wait(qk_full_addr, _phase_qk_full_0);
                _phase_qk_full_0 ^= 1;
                int l2n_c = lane_1 & 15;
                int l2n_rsub = w_id * 2 + half;
                unsigned int raw[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(0) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(8) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(12) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                float x[32];
                #pragma unroll
                for (int _pair = 0; _pair < 16; _pair++) {
                    (&x[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) << 16);
                    (&x[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw[_pair]) & 0xffff0000u);
                }
                float ss[4];
                ss[0] = mul_f32_rn_noftz(x[0], x[0]);
                float _fma_0;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(x[1]), "f"(x[1]), "f"(ss[0]));
                ss[0] = _fma_0;
                float _fma_1;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(x[2]), "f"(x[2]), "f"(ss[0]));
                ss[0] = _fma_1;
                float _fma_2;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(x[3]), "f"(x[3]), "f"(ss[0]));
                ss[0] = _fma_2;
                float _fma_3;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(x[4]), "f"(x[4]), "f"(ss[0]));
                ss[0] = _fma_3;
                float _fma_4;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(x[5]), "f"(x[5]), "f"(ss[0]));
                ss[0] = _fma_4;
                float _fma_5;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(x[6]), "f"(x[6]), "f"(ss[0]));
                ss[0] = _fma_5;
                float _fma_6;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(x[7]), "f"(x[7]), "f"(ss[0]));
                ss[0] = _fma_6;
                ss[1] = mul_f32_rn_noftz(x[8], x[8]);
                float _fma_7;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(x[9]), "f"(x[9]), "f"(ss[1]));
                ss[1] = _fma_7;
                float _fma_8;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(x[10]), "f"(x[10]), "f"(ss[1]));
                ss[1] = _fma_8;
                float _fma_9;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(x[11]), "f"(x[11]), "f"(ss[1]));
                ss[1] = _fma_9;
                float _fma_10;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(x[12]), "f"(x[12]), "f"(ss[1]));
                ss[1] = _fma_10;
                float _fma_11;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(x[13]), "f"(x[13]), "f"(ss[1]));
                ss[1] = _fma_11;
                float _fma_12;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(x[14]), "f"(x[14]), "f"(ss[1]));
                ss[1] = _fma_12;
                float _fma_13;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(x[15]), "f"(x[15]), "f"(ss[1]));
                ss[1] = _fma_13;
                ss[2] = mul_f32_rn_noftz(x[16], x[16]);
                float _fma_14;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(x[17]), "f"(x[17]), "f"(ss[2]));
                ss[2] = _fma_14;
                float _fma_15;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(x[18]), "f"(x[18]), "f"(ss[2]));
                ss[2] = _fma_15;
                float _fma_16;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(x[19]), "f"(x[19]), "f"(ss[2]));
                ss[2] = _fma_16;
                float _fma_17;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(x[20]), "f"(x[20]), "f"(ss[2]));
                ss[2] = _fma_17;
                float _fma_18;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(x[21]), "f"(x[21]), "f"(ss[2]));
                ss[2] = _fma_18;
                float _fma_19;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(x[22]), "f"(x[22]), "f"(ss[2]));
                ss[2] = _fma_19;
                float _fma_20;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(x[23]), "f"(x[23]), "f"(ss[2]));
                ss[2] = _fma_20;
                ss[3] = mul_f32_rn_noftz(x[24], x[24]);
                float _fma_21;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(x[25]), "f"(x[25]), "f"(ss[3]));
                ss[3] = _fma_21;
                float _fma_22;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(x[26]), "f"(x[26]), "f"(ss[3]));
                ss[3] = _fma_22;
                float _fma_23;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(x[27]), "f"(x[27]), "f"(ss[3]));
                ss[3] = _fma_23;
                float _fma_24;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(x[28]), "f"(x[28]), "f"(ss[3]));
                ss[3] = _fma_24;
                float _fma_25;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(x[29]), "f"(x[29]), "f"(ss[3]));
                ss[3] = _fma_25;
                float _fma_26;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(x[30]), "f"(x[30]), "f"(ss[3]));
                ss[3] = _fma_26;
                float _fma_27;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(x[31]), "f"(x[31]), "f"(ss[3]));
                ss[3] = _fma_27;
                float _shfl_xor_sync_29 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 8);
                float _add_28;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(ss[0]), "f"(_shfl_xor_sync_29));
                ss[0] = _add_28;
                float _shfl_xor_sync_31 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 8);
                float _add_30;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(ss[1]), "f"(_shfl_xor_sync_31));
                ss[1] = _add_30;
                float _shfl_xor_sync_33 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 8);
                float _add_32;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(ss[2]), "f"(_shfl_xor_sync_33));
                ss[2] = _add_32;
                float _shfl_xor_sync_35 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 8);
                float _add_34;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(ss[3]), "f"(_shfl_xor_sync_35));
                ss[3] = _add_34;
                float _shfl_xor_sync_37 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 4);
                float _add_36;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(ss[0]), "f"(_shfl_xor_sync_37));
                ss[0] = _add_36;
                float _shfl_xor_sync_39 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 4);
                float _add_38;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(ss[1]), "f"(_shfl_xor_sync_39));
                ss[1] = _add_38;
                float _shfl_xor_sync_41 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 4);
                float _add_40;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(ss[2]), "f"(_shfl_xor_sync_41));
                ss[2] = _add_40;
                float _shfl_xor_sync_43 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 4);
                float _add_42;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(ss[3]), "f"(_shfl_xor_sync_43));
                ss[3] = _add_42;
                float _shfl_xor_sync_45 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 2);
                float _add_44;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(ss[0]), "f"(_shfl_xor_sync_45));
                ss[0] = _add_44;
                float _shfl_xor_sync_47 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 2);
                float _add_46;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(ss[1]), "f"(_shfl_xor_sync_47));
                ss[1] = _add_46;
                float _shfl_xor_sync_49 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 2);
                float _add_48;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(ss[2]), "f"(_shfl_xor_sync_49));
                ss[2] = _add_48;
                float _shfl_xor_sync_51 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 2);
                float _add_50;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(ss[3]), "f"(_shfl_xor_sync_51));
                ss[3] = _add_50;
                float _shfl_xor_sync_53 = __shfl_xor_sync(0xFFFFFFFF, ss[0], 1);
                float _add_52;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(ss[0]), "f"(_shfl_xor_sync_53));
                ss[0] = _add_52;
                float _shfl_xor_sync_55 = __shfl_xor_sync(0xFFFFFFFF, ss[1], 1);
                float _add_54;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(ss[1]), "f"(_shfl_xor_sync_55));
                ss[1] = _add_54;
                float _shfl_xor_sync_57 = __shfl_xor_sync(0xFFFFFFFF, ss[2], 1);
                float _add_56;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(ss[2]), "f"(_shfl_xor_sync_57));
                ss[2] = _add_56;
                float _shfl_xor_sync_59 = __shfl_xor_sync(0xFFFFFFFF, ss[3], 1);
                float _add_58;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(ss[3]), "f"(_shfl_xor_sync_59));
                ss[3] = _add_58;
                float r[4];
                float _add_62;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(ss[0]), "f"(1e-06f));
                float _sqrt_61;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_61) : "f"(_add_62));
                float _fdiv_60 = __fdiv_rn(1.0f, _sqrt_61);
                r[0] = _fdiv_60;
                float _add_65;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(ss[1]), "f"(1e-06f));
                float _sqrt_64;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_64) : "f"(_add_65));
                float _fdiv_63 = __fdiv_rn(1.0f, _sqrt_64);
                r[1] = _fdiv_63;
                float _add_68;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(ss[2]), "f"(1e-06f));
                float _sqrt_67;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_67) : "f"(_add_68));
                float _fdiv_66 = __fdiv_rn(1.0f, _sqrt_67);
                r[2] = _fdiv_66;
                float _add_71;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(ss[3]), "f"(1e-06f));
                float _sqrt_70;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_70) : "f"(_add_71));
                float _fdiv_69 = __fdiv_rn(1.0f, _sqrt_70);
                r[3] = _fdiv_69;
                float y[32];
                y[0] = mul_f32_rn_noftz(x[0], r[0]);
                y[1] = mul_f32_rn_noftz(x[1], r[0]);
                y[2] = mul_f32_rn_noftz(x[2], r[0]);
                y[3] = mul_f32_rn_noftz(x[3], r[0]);
                y[4] = mul_f32_rn_noftz(x[4], r[0]);
                y[5] = mul_f32_rn_noftz(x[5], r[0]);
                y[6] = mul_f32_rn_noftz(x[6], r[0]);
                y[7] = mul_f32_rn_noftz(x[7], r[0]);
                y[8] = mul_f32_rn_noftz(x[8], r[1]);
                y[9] = mul_f32_rn_noftz(x[9], r[1]);
                y[10] = mul_f32_rn_noftz(x[10], r[1]);
                y[11] = mul_f32_rn_noftz(x[11], r[1]);
                y[12] = mul_f32_rn_noftz(x[12], r[1]);
                y[13] = mul_f32_rn_noftz(x[13], r[1]);
                y[14] = mul_f32_rn_noftz(x[14], r[1]);
                y[15] = mul_f32_rn_noftz(x[15], r[1]);
                y[16] = mul_f32_rn_noftz(x[16], r[2]);
                y[17] = mul_f32_rn_noftz(x[17], r[2]);
                y[18] = mul_f32_rn_noftz(x[18], r[2]);
                y[19] = mul_f32_rn_noftz(x[19], r[2]);
                y[20] = mul_f32_rn_noftz(x[20], r[2]);
                y[21] = mul_f32_rn_noftz(x[21], r[2]);
                y[22] = mul_f32_rn_noftz(x[22], r[2]);
                y[23] = mul_f32_rn_noftz(x[23], r[2]);
                y[24] = mul_f32_rn_noftz(x[24], r[3]);
                y[25] = mul_f32_rn_noftz(x[25], r[3]);
                y[26] = mul_f32_rn_noftz(x[26], r[3]);
                y[27] = mul_f32_rn_noftz(x[27], r[3]);
                y[28] = mul_f32_rn_noftz(x[28], r[3]);
                y[29] = mul_f32_rn_noftz(x[29], r[3]);
                y[30] = mul_f32_rn_noftz(x[30], r[3]);
                y[31] = mul_f32_rn_noftz(x[31], r[3]);
                unsigned int yb[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y[_lp*2 + 0], y[_lp*2+1 + 0]));
                    yb[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb[0]), "r"(yb[1]), "r"(yb[2]), "r"(yb[3]) : "memory");
                if (l2n_rsub < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb)[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub) * (long long)rstd_tstride + (long long)hq)) + (0)) = r[0];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb[4]), "r"(yb[5]), "r"(yb[6]), "r"(yb[7]) : "memory");
                if (l2n_rsub + 8 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 8) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb + (4))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 8) * (long long)rstd_tstride + (long long)hq)) + (0)) = r[1];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb[8]), "r"(yb[9]), "r"(yb[10]), "r"(yb[11]) : "memory");
                if (l2n_rsub + 16 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 16) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb + (8))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 16) * (long long)rstd_tstride + (long long)hq)) + (0)) = r[2];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb[12]), "r"(yb[13]), "r"(yb[14]), "r"(yb[15]) : "memory");
                if (l2n_rsub + 24 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 24) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb + (12))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 24) * (long long)rstd_tstride + (long long)hq)) + (0)) = r[3];
                    }
                }
                unsigned int raw_0[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(0) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(4) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(8) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_0[(12) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                float x_0[32];
                #pragma unroll
                for (int _pair = 0; _pair < 16; _pair++) {
                    (&x_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw_0[_pair]) << 16);
                    (&x_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw_0[_pair]) & 0xffff0000u);
                }
                float ss_0[4];
                ss_0[0] = mul_f32_rn_noftz(x_0[0], x_0[0]);
                float _fma_72;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_72) : "f"(x_0[1]), "f"(x_0[1]), "f"(ss_0[0]));
                ss_0[0] = _fma_72;
                float _fma_73;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_73) : "f"(x_0[2]), "f"(x_0[2]), "f"(ss_0[0]));
                ss_0[0] = _fma_73;
                float _fma_74;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_74) : "f"(x_0[3]), "f"(x_0[3]), "f"(ss_0[0]));
                ss_0[0] = _fma_74;
                float _fma_75;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_75) : "f"(x_0[4]), "f"(x_0[4]), "f"(ss_0[0]));
                ss_0[0] = _fma_75;
                float _fma_76;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_76) : "f"(x_0[5]), "f"(x_0[5]), "f"(ss_0[0]));
                ss_0[0] = _fma_76;
                float _fma_77;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_77) : "f"(x_0[6]), "f"(x_0[6]), "f"(ss_0[0]));
                ss_0[0] = _fma_77;
                float _fma_78;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_78) : "f"(x_0[7]), "f"(x_0[7]), "f"(ss_0[0]));
                ss_0[0] = _fma_78;
                ss_0[1] = mul_f32_rn_noftz(x_0[8], x_0[8]);
                float _fma_79;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_79) : "f"(x_0[9]), "f"(x_0[9]), "f"(ss_0[1]));
                ss_0[1] = _fma_79;
                float _fma_80;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_80) : "f"(x_0[10]), "f"(x_0[10]), "f"(ss_0[1]));
                ss_0[1] = _fma_80;
                float _fma_81;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_81) : "f"(x_0[11]), "f"(x_0[11]), "f"(ss_0[1]));
                ss_0[1] = _fma_81;
                float _fma_82;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_82) : "f"(x_0[12]), "f"(x_0[12]), "f"(ss_0[1]));
                ss_0[1] = _fma_82;
                float _fma_83;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_83) : "f"(x_0[13]), "f"(x_0[13]), "f"(ss_0[1]));
                ss_0[1] = _fma_83;
                float _fma_84;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_84) : "f"(x_0[14]), "f"(x_0[14]), "f"(ss_0[1]));
                ss_0[1] = _fma_84;
                float _fma_85;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_85) : "f"(x_0[15]), "f"(x_0[15]), "f"(ss_0[1]));
                ss_0[1] = _fma_85;
                ss_0[2] = mul_f32_rn_noftz(x_0[16], x_0[16]);
                float _fma_86;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_86) : "f"(x_0[17]), "f"(x_0[17]), "f"(ss_0[2]));
                ss_0[2] = _fma_86;
                float _fma_87;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_87) : "f"(x_0[18]), "f"(x_0[18]), "f"(ss_0[2]));
                ss_0[2] = _fma_87;
                float _fma_88;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_88) : "f"(x_0[19]), "f"(x_0[19]), "f"(ss_0[2]));
                ss_0[2] = _fma_88;
                float _fma_89;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_89) : "f"(x_0[20]), "f"(x_0[20]), "f"(ss_0[2]));
                ss_0[2] = _fma_89;
                float _fma_90;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_90) : "f"(x_0[21]), "f"(x_0[21]), "f"(ss_0[2]));
                ss_0[2] = _fma_90;
                float _fma_91;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_91) : "f"(x_0[22]), "f"(x_0[22]), "f"(ss_0[2]));
                ss_0[2] = _fma_91;
                float _fma_92;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_92) : "f"(x_0[23]), "f"(x_0[23]), "f"(ss_0[2]));
                ss_0[2] = _fma_92;
                ss_0[3] = mul_f32_rn_noftz(x_0[24], x_0[24]);
                float _fma_93;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_93) : "f"(x_0[25]), "f"(x_0[25]), "f"(ss_0[3]));
                ss_0[3] = _fma_93;
                float _fma_94;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_94) : "f"(x_0[26]), "f"(x_0[26]), "f"(ss_0[3]));
                ss_0[3] = _fma_94;
                float _fma_95;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_95) : "f"(x_0[27]), "f"(x_0[27]), "f"(ss_0[3]));
                ss_0[3] = _fma_95;
                float _fma_96;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_96) : "f"(x_0[28]), "f"(x_0[28]), "f"(ss_0[3]));
                ss_0[3] = _fma_96;
                float _fma_97;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_97) : "f"(x_0[29]), "f"(x_0[29]), "f"(ss_0[3]));
                ss_0[3] = _fma_97;
                float _fma_98;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_98) : "f"(x_0[30]), "f"(x_0[30]), "f"(ss_0[3]));
                ss_0[3] = _fma_98;
                float _fma_99;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_99) : "f"(x_0[31]), "f"(x_0[31]), "f"(ss_0[3]));
                ss_0[3] = _fma_99;
                float _shfl_xor_sync_101 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 8);
                float _add_100;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(ss_0[0]), "f"(_shfl_xor_sync_101));
                ss_0[0] = _add_100;
                float _shfl_xor_sync_103 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 8);
                float _add_102;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(ss_0[1]), "f"(_shfl_xor_sync_103));
                ss_0[1] = _add_102;
                float _shfl_xor_sync_105 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 8);
                float _add_104;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(ss_0[2]), "f"(_shfl_xor_sync_105));
                ss_0[2] = _add_104;
                float _shfl_xor_sync_107 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 8);
                float _add_106;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(ss_0[3]), "f"(_shfl_xor_sync_107));
                ss_0[3] = _add_106;
                float _shfl_xor_sync_109 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 4);
                float _add_108;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(ss_0[0]), "f"(_shfl_xor_sync_109));
                ss_0[0] = _add_108;
                float _shfl_xor_sync_111 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 4);
                float _add_110;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(ss_0[1]), "f"(_shfl_xor_sync_111));
                ss_0[1] = _add_110;
                float _shfl_xor_sync_113 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 4);
                float _add_112;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(ss_0[2]), "f"(_shfl_xor_sync_113));
                ss_0[2] = _add_112;
                float _shfl_xor_sync_115 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 4);
                float _add_114;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(ss_0[3]), "f"(_shfl_xor_sync_115));
                ss_0[3] = _add_114;
                float _shfl_xor_sync_117 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 2);
                float _add_116;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(ss_0[0]), "f"(_shfl_xor_sync_117));
                ss_0[0] = _add_116;
                float _shfl_xor_sync_119 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 2);
                float _add_118;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(ss_0[1]), "f"(_shfl_xor_sync_119));
                ss_0[1] = _add_118;
                float _shfl_xor_sync_121 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 2);
                float _add_120;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(ss_0[2]), "f"(_shfl_xor_sync_121));
                ss_0[2] = _add_120;
                float _shfl_xor_sync_123 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 2);
                float _add_122;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(ss_0[3]), "f"(_shfl_xor_sync_123));
                ss_0[3] = _add_122;
                float _shfl_xor_sync_125 = __shfl_xor_sync(0xFFFFFFFF, ss_0[0], 1);
                float _add_124;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(ss_0[0]), "f"(_shfl_xor_sync_125));
                ss_0[0] = _add_124;
                float _shfl_xor_sync_127 = __shfl_xor_sync(0xFFFFFFFF, ss_0[1], 1);
                float _add_126;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(ss_0[1]), "f"(_shfl_xor_sync_127));
                ss_0[1] = _add_126;
                float _shfl_xor_sync_129 = __shfl_xor_sync(0xFFFFFFFF, ss_0[2], 1);
                float _add_128;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_128) : "f"(ss_0[2]), "f"(_shfl_xor_sync_129));
                ss_0[2] = _add_128;
                float _shfl_xor_sync_131 = __shfl_xor_sync(0xFFFFFFFF, ss_0[3], 1);
                float _add_130;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_130) : "f"(ss_0[3]), "f"(_shfl_xor_sync_131));
                ss_0[3] = _add_130;
                float r_0[4];
                float _add_134;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_134) : "f"(ss_0[0]), "f"(1e-06f));
                float _sqrt_133;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_133) : "f"(_add_134));
                float _fdiv_132 = __fdiv_rn(1.0f, _sqrt_133);
                r_0[0] = _fdiv_132;
                float _add_137;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_137) : "f"(ss_0[1]), "f"(1e-06f));
                float _sqrt_136;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_136) : "f"(_add_137));
                float _fdiv_135 = __fdiv_rn(1.0f, _sqrt_136);
                r_0[1] = _fdiv_135;
                float _add_140;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_140) : "f"(ss_0[2]), "f"(1e-06f));
                float _sqrt_139;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_139) : "f"(_add_140));
                float _fdiv_138 = __fdiv_rn(1.0f, _sqrt_139);
                r_0[2] = _fdiv_138;
                float _add_143;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_143) : "f"(ss_0[3]), "f"(1e-06f));
                float _sqrt_142;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_142) : "f"(_add_143));
                float _fdiv_141 = __fdiv_rn(1.0f, _sqrt_142);
                r_0[3] = _fdiv_141;
                float y_0[32];
                y_0[0] = mul_f32_rn_noftz(x_0[0], r_0[0]);
                y_0[1] = mul_f32_rn_noftz(x_0[1], r_0[0]);
                y_0[2] = mul_f32_rn_noftz(x_0[2], r_0[0]);
                y_0[3] = mul_f32_rn_noftz(x_0[3], r_0[0]);
                y_0[4] = mul_f32_rn_noftz(x_0[4], r_0[0]);
                y_0[5] = mul_f32_rn_noftz(x_0[5], r_0[0]);
                y_0[6] = mul_f32_rn_noftz(x_0[6], r_0[0]);
                y_0[7] = mul_f32_rn_noftz(x_0[7], r_0[0]);
                y_0[8] = mul_f32_rn_noftz(x_0[8], r_0[1]);
                y_0[9] = mul_f32_rn_noftz(x_0[9], r_0[1]);
                y_0[10] = mul_f32_rn_noftz(x_0[10], r_0[1]);
                y_0[11] = mul_f32_rn_noftz(x_0[11], r_0[1]);
                y_0[12] = mul_f32_rn_noftz(x_0[12], r_0[1]);
                y_0[13] = mul_f32_rn_noftz(x_0[13], r_0[1]);
                y_0[14] = mul_f32_rn_noftz(x_0[14], r_0[1]);
                y_0[15] = mul_f32_rn_noftz(x_0[15], r_0[1]);
                y_0[16] = mul_f32_rn_noftz(x_0[16], r_0[2]);
                y_0[17] = mul_f32_rn_noftz(x_0[17], r_0[2]);
                y_0[18] = mul_f32_rn_noftz(x_0[18], r_0[2]);
                y_0[19] = mul_f32_rn_noftz(x_0[19], r_0[2]);
                y_0[20] = mul_f32_rn_noftz(x_0[20], r_0[2]);
                y_0[21] = mul_f32_rn_noftz(x_0[21], r_0[2]);
                y_0[22] = mul_f32_rn_noftz(x_0[22], r_0[2]);
                y_0[23] = mul_f32_rn_noftz(x_0[23], r_0[2]);
                y_0[24] = mul_f32_rn_noftz(x_0[24], r_0[3]);
                y_0[25] = mul_f32_rn_noftz(x_0[25], r_0[3]);
                y_0[26] = mul_f32_rn_noftz(x_0[26], r_0[3]);
                y_0[27] = mul_f32_rn_noftz(x_0[27], r_0[3]);
                y_0[28] = mul_f32_rn_noftz(x_0[28], r_0[3]);
                y_0[29] = mul_f32_rn_noftz(x_0[29], r_0[3]);
                y_0[30] = mul_f32_rn_noftz(x_0[30], r_0[3]);
                y_0[31] = mul_f32_rn_noftz(x_0[31], r_0[3]);
                unsigned int yb_0[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_0[_lp*2 + 0], y_0[_lp*2+1 + 0]));
                    yb_0[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + l2n_rsub * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_0[0]), "r"(yb_0[1]), "r"(yb_0[2]), "r"(yb_0[3]) : "memory");
                if (l2n_rsub < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_0)[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_0[0];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 8) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_0[4]), "r"(yb_0[5]), "r"(yb_0[6]), "r"(yb_0[7]) : "memory");
                if (l2n_rsub + 8 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 8) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_0 + (4))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 8) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_0[1];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 16) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_0[8]), "r"(yb_0[9]), "r"(yb_0[10]), "r"(yb_0[11]) : "memory");
                if (l2n_rsub + 16 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 16) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_0 + (8))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 16) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_0[2];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 24) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_0[12]), "r"(yb_0[13]), "r"(yb_0[14]), "r"(yb_0[15]) : "memory");
                if (l2n_rsub + 24 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 24) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_0 + (12))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 24) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_0[3];
                    }
                }
                unsigned int raw_1[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(0) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(8) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(12) + 3]))
                    : "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                float x_1[32];
                #pragma unroll
                for (int _pair = 0; _pair < 16; _pair++) {
                    (&x_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw_1[_pair]) << 16);
                    (&x_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw_1[_pair]) & 0xffff0000u);
                }
                float ss_1[4];
                ss_1[0] = mul_f32_rn_noftz(x_1[0], x_1[0]);
                float _fma_144;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_144) : "f"(x_1[1]), "f"(x_1[1]), "f"(ss_1[0]));
                ss_1[0] = _fma_144;
                float _fma_145;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_145) : "f"(x_1[2]), "f"(x_1[2]), "f"(ss_1[0]));
                ss_1[0] = _fma_145;
                float _fma_146;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_146) : "f"(x_1[3]), "f"(x_1[3]), "f"(ss_1[0]));
                ss_1[0] = _fma_146;
                float _fma_147;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_147) : "f"(x_1[4]), "f"(x_1[4]), "f"(ss_1[0]));
                ss_1[0] = _fma_147;
                float _fma_148;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_148) : "f"(x_1[5]), "f"(x_1[5]), "f"(ss_1[0]));
                ss_1[0] = _fma_148;
                float _fma_149;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_149) : "f"(x_1[6]), "f"(x_1[6]), "f"(ss_1[0]));
                ss_1[0] = _fma_149;
                float _fma_150;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_150) : "f"(x_1[7]), "f"(x_1[7]), "f"(ss_1[0]));
                ss_1[0] = _fma_150;
                ss_1[1] = mul_f32_rn_noftz(x_1[8], x_1[8]);
                float _fma_151;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_151) : "f"(x_1[9]), "f"(x_1[9]), "f"(ss_1[1]));
                ss_1[1] = _fma_151;
                float _fma_152;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_152) : "f"(x_1[10]), "f"(x_1[10]), "f"(ss_1[1]));
                ss_1[1] = _fma_152;
                float _fma_153;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_153) : "f"(x_1[11]), "f"(x_1[11]), "f"(ss_1[1]));
                ss_1[1] = _fma_153;
                float _fma_154;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_154) : "f"(x_1[12]), "f"(x_1[12]), "f"(ss_1[1]));
                ss_1[1] = _fma_154;
                float _fma_155;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_155) : "f"(x_1[13]), "f"(x_1[13]), "f"(ss_1[1]));
                ss_1[1] = _fma_155;
                float _fma_156;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_156) : "f"(x_1[14]), "f"(x_1[14]), "f"(ss_1[1]));
                ss_1[1] = _fma_156;
                float _fma_157;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_157) : "f"(x_1[15]), "f"(x_1[15]), "f"(ss_1[1]));
                ss_1[1] = _fma_157;
                ss_1[2] = mul_f32_rn_noftz(x_1[16], x_1[16]);
                float _fma_158;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_158) : "f"(x_1[17]), "f"(x_1[17]), "f"(ss_1[2]));
                ss_1[2] = _fma_158;
                float _fma_159;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_159) : "f"(x_1[18]), "f"(x_1[18]), "f"(ss_1[2]));
                ss_1[2] = _fma_159;
                float _fma_160;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_160) : "f"(x_1[19]), "f"(x_1[19]), "f"(ss_1[2]));
                ss_1[2] = _fma_160;
                float _fma_161;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_161) : "f"(x_1[20]), "f"(x_1[20]), "f"(ss_1[2]));
                ss_1[2] = _fma_161;
                float _fma_162;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_162) : "f"(x_1[21]), "f"(x_1[21]), "f"(ss_1[2]));
                ss_1[2] = _fma_162;
                float _fma_163;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_163) : "f"(x_1[22]), "f"(x_1[22]), "f"(ss_1[2]));
                ss_1[2] = _fma_163;
                float _fma_164;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_164) : "f"(x_1[23]), "f"(x_1[23]), "f"(ss_1[2]));
                ss_1[2] = _fma_164;
                ss_1[3] = mul_f32_rn_noftz(x_1[24], x_1[24]);
                float _fma_165;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_165) : "f"(x_1[25]), "f"(x_1[25]), "f"(ss_1[3]));
                ss_1[3] = _fma_165;
                float _fma_166;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_166) : "f"(x_1[26]), "f"(x_1[26]), "f"(ss_1[3]));
                ss_1[3] = _fma_166;
                float _fma_167;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_167) : "f"(x_1[27]), "f"(x_1[27]), "f"(ss_1[3]));
                ss_1[3] = _fma_167;
                float _fma_168;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_168) : "f"(x_1[28]), "f"(x_1[28]), "f"(ss_1[3]));
                ss_1[3] = _fma_168;
                float _fma_169;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_169) : "f"(x_1[29]), "f"(x_1[29]), "f"(ss_1[3]));
                ss_1[3] = _fma_169;
                float _fma_170;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_170) : "f"(x_1[30]), "f"(x_1[30]), "f"(ss_1[3]));
                ss_1[3] = _fma_170;
                float _fma_171;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_171) : "f"(x_1[31]), "f"(x_1[31]), "f"(ss_1[3]));
                ss_1[3] = _fma_171;
                float _shfl_xor_sync_173 = __shfl_xor_sync(0xFFFFFFFF, ss_1[0], 8);
                float _add_172;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_172) : "f"(ss_1[0]), "f"(_shfl_xor_sync_173));
                ss_1[0] = _add_172;
                float _shfl_xor_sync_175 = __shfl_xor_sync(0xFFFFFFFF, ss_1[1], 8);
                float _add_174;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_174) : "f"(ss_1[1]), "f"(_shfl_xor_sync_175));
                ss_1[1] = _add_174;
                float _shfl_xor_sync_177 = __shfl_xor_sync(0xFFFFFFFF, ss_1[2], 8);
                float _add_176;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_176) : "f"(ss_1[2]), "f"(_shfl_xor_sync_177));
                ss_1[2] = _add_176;
                float _shfl_xor_sync_179 = __shfl_xor_sync(0xFFFFFFFF, ss_1[3], 8);
                float _add_178;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_178) : "f"(ss_1[3]), "f"(_shfl_xor_sync_179));
                ss_1[3] = _add_178;
                float _shfl_xor_sync_181 = __shfl_xor_sync(0xFFFFFFFF, ss_1[0], 4);
                float _add_180;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_180) : "f"(ss_1[0]), "f"(_shfl_xor_sync_181));
                ss_1[0] = _add_180;
                float _shfl_xor_sync_183 = __shfl_xor_sync(0xFFFFFFFF, ss_1[1], 4);
                float _add_182;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_182) : "f"(ss_1[1]), "f"(_shfl_xor_sync_183));
                ss_1[1] = _add_182;
                float _shfl_xor_sync_185 = __shfl_xor_sync(0xFFFFFFFF, ss_1[2], 4);
                float _add_184;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_184) : "f"(ss_1[2]), "f"(_shfl_xor_sync_185));
                ss_1[2] = _add_184;
                float _shfl_xor_sync_187 = __shfl_xor_sync(0xFFFFFFFF, ss_1[3], 4);
                float _add_186;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_186) : "f"(ss_1[3]), "f"(_shfl_xor_sync_187));
                ss_1[3] = _add_186;
                float _shfl_xor_sync_189 = __shfl_xor_sync(0xFFFFFFFF, ss_1[0], 2);
                float _add_188;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_188) : "f"(ss_1[0]), "f"(_shfl_xor_sync_189));
                ss_1[0] = _add_188;
                float _shfl_xor_sync_191 = __shfl_xor_sync(0xFFFFFFFF, ss_1[1], 2);
                float _add_190;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_190) : "f"(ss_1[1]), "f"(_shfl_xor_sync_191));
                ss_1[1] = _add_190;
                float _shfl_xor_sync_193 = __shfl_xor_sync(0xFFFFFFFF, ss_1[2], 2);
                float _add_192;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_192) : "f"(ss_1[2]), "f"(_shfl_xor_sync_193));
                ss_1[2] = _add_192;
                float _shfl_xor_sync_195 = __shfl_xor_sync(0xFFFFFFFF, ss_1[3], 2);
                float _add_194;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_194) : "f"(ss_1[3]), "f"(_shfl_xor_sync_195));
                ss_1[3] = _add_194;
                float _shfl_xor_sync_197 = __shfl_xor_sync(0xFFFFFFFF, ss_1[0], 1);
                float _add_196;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_196) : "f"(ss_1[0]), "f"(_shfl_xor_sync_197));
                ss_1[0] = _add_196;
                float _shfl_xor_sync_199 = __shfl_xor_sync(0xFFFFFFFF, ss_1[1], 1);
                float _add_198;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_198) : "f"(ss_1[1]), "f"(_shfl_xor_sync_199));
                ss_1[1] = _add_198;
                float _shfl_xor_sync_201 = __shfl_xor_sync(0xFFFFFFFF, ss_1[2], 1);
                float _add_200;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_200) : "f"(ss_1[2]), "f"(_shfl_xor_sync_201));
                ss_1[2] = _add_200;
                float _shfl_xor_sync_203 = __shfl_xor_sync(0xFFFFFFFF, ss_1[3], 1);
                float _add_202;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_202) : "f"(ss_1[3]), "f"(_shfl_xor_sync_203));
                ss_1[3] = _add_202;
                float r_1[4];
                float _add_206;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_206) : "f"(ss_1[0]), "f"(1e-06f));
                float _sqrt_205;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_205) : "f"(_add_206));
                float _fdiv_204 = __fdiv_rn(1.0f, _sqrt_205);
                r_1[0] = _fdiv_204;
                float _add_209;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_209) : "f"(ss_1[1]), "f"(1e-06f));
                float _sqrt_208;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_208) : "f"(_add_209));
                float _fdiv_207 = __fdiv_rn(1.0f, _sqrt_208);
                r_1[1] = _fdiv_207;
                float _add_212;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_212) : "f"(ss_1[2]), "f"(1e-06f));
                float _sqrt_211;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_211) : "f"(_add_212));
                float _fdiv_210 = __fdiv_rn(1.0f, _sqrt_211);
                r_1[2] = _fdiv_210;
                float _add_215;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_215) : "f"(ss_1[3]), "f"(1e-06f));
                float _sqrt_214;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_214) : "f"(_add_215));
                float _fdiv_213 = __fdiv_rn(1.0f, _sqrt_214);
                r_1[3] = _fdiv_213;
                float y_1[32];
                y_1[0] = mul_f32_rn_noftz(x_1[0], r_1[0]);
                y_1[1] = mul_f32_rn_noftz(x_1[1], r_1[0]);
                y_1[2] = mul_f32_rn_noftz(x_1[2], r_1[0]);
                y_1[3] = mul_f32_rn_noftz(x_1[3], r_1[0]);
                y_1[4] = mul_f32_rn_noftz(x_1[4], r_1[0]);
                y_1[5] = mul_f32_rn_noftz(x_1[5], r_1[0]);
                y_1[6] = mul_f32_rn_noftz(x_1[6], r_1[0]);
                y_1[7] = mul_f32_rn_noftz(x_1[7], r_1[0]);
                y_1[8] = mul_f32_rn_noftz(x_1[8], r_1[1]);
                y_1[9] = mul_f32_rn_noftz(x_1[9], r_1[1]);
                y_1[10] = mul_f32_rn_noftz(x_1[10], r_1[1]);
                y_1[11] = mul_f32_rn_noftz(x_1[11], r_1[1]);
                y_1[12] = mul_f32_rn_noftz(x_1[12], r_1[1]);
                y_1[13] = mul_f32_rn_noftz(x_1[13], r_1[1]);
                y_1[14] = mul_f32_rn_noftz(x_1[14], r_1[1]);
                y_1[15] = mul_f32_rn_noftz(x_1[15], r_1[1]);
                y_1[16] = mul_f32_rn_noftz(x_1[16], r_1[2]);
                y_1[17] = mul_f32_rn_noftz(x_1[17], r_1[2]);
                y_1[18] = mul_f32_rn_noftz(x_1[18], r_1[2]);
                y_1[19] = mul_f32_rn_noftz(x_1[19], r_1[2]);
                y_1[20] = mul_f32_rn_noftz(x_1[20], r_1[2]);
                y_1[21] = mul_f32_rn_noftz(x_1[21], r_1[2]);
                y_1[22] = mul_f32_rn_noftz(x_1[22], r_1[2]);
                y_1[23] = mul_f32_rn_noftz(x_1[23], r_1[2]);
                y_1[24] = mul_f32_rn_noftz(x_1[24], r_1[3]);
                y_1[25] = mul_f32_rn_noftz(x_1[25], r_1[3]);
                y_1[26] = mul_f32_rn_noftz(x_1[26], r_1[3]);
                y_1[27] = mul_f32_rn_noftz(x_1[27], r_1[3]);
                y_1[28] = mul_f32_rn_noftz(x_1[28], r_1[3]);
                y_1[29] = mul_f32_rn_noftz(x_1[29], r_1[3]);
                y_1[30] = mul_f32_rn_noftz(x_1[30], r_1[3]);
                y_1[31] = mul_f32_rn_noftz(x_1[31], r_1[3]);
                unsigned int yb_1[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_1[_lp*2 + 0], y_1[_lp*2+1 + 0]));
                    yb_1[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_1[0]), "r"(yb_1[1]), "r"(yb_1[2]), "r"(yb_1[3]) : "memory");
                if (l2n_rsub + 32 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 32) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_1)[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 32) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_1[0];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_1[4]), "r"(yb_1[5]), "r"(yb_1[6]), "r"(yb_1[7]) : "memory");
                if (l2n_rsub + 40 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 40) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_1 + (4))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 40) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_1[1];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_1[8]), "r"(yb_1[9]), "r"(yb_1[10]), "r"(yb_1[11]) : "memory");
                if (l2n_rsub + 48 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 48) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_1 + (8))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 48) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_1[2];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_1[12]), "r"(yb_1[13]), "r"(yb_1[14]), "r"(yb_1[15]) : "memory");
                if (l2n_rsub + 56 < clen) {
                    reinterpret_cast<int4*>(qn_out + (((long long)(ext0 + l2n_rsub + 56) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_1 + (12))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(q_rstd + ((long long)(ext0 + l2n_rsub + 56) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_1[3];
                    }
                }
                unsigned int raw_2[16];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(0) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(4) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(8) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&raw_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_2[(12) + 3]))
                    : "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))));
                float x_2[32];
                #pragma unroll
                for (int _pair = 0; _pair < 16; _pair++) {
                    (&x_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(raw_2[_pair]) << 16);
                    (&x_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(raw_2[_pair]) & 0xffff0000u);
                }
                float ss_2[4];
                ss_2[0] = mul_f32_rn_noftz(x_2[0], x_2[0]);
                float _fma_216;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_216) : "f"(x_2[1]), "f"(x_2[1]), "f"(ss_2[0]));
                ss_2[0] = _fma_216;
                float _fma_217;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_217) : "f"(x_2[2]), "f"(x_2[2]), "f"(ss_2[0]));
                ss_2[0] = _fma_217;
                float _fma_218;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_218) : "f"(x_2[3]), "f"(x_2[3]), "f"(ss_2[0]));
                ss_2[0] = _fma_218;
                float _fma_219;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_219) : "f"(x_2[4]), "f"(x_2[4]), "f"(ss_2[0]));
                ss_2[0] = _fma_219;
                float _fma_220;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_220) : "f"(x_2[5]), "f"(x_2[5]), "f"(ss_2[0]));
                ss_2[0] = _fma_220;
                float _fma_221;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_221) : "f"(x_2[6]), "f"(x_2[6]), "f"(ss_2[0]));
                ss_2[0] = _fma_221;
                float _fma_222;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_222) : "f"(x_2[7]), "f"(x_2[7]), "f"(ss_2[0]));
                ss_2[0] = _fma_222;
                ss_2[1] = mul_f32_rn_noftz(x_2[8], x_2[8]);
                float _fma_223;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_223) : "f"(x_2[9]), "f"(x_2[9]), "f"(ss_2[1]));
                ss_2[1] = _fma_223;
                float _fma_224;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_224) : "f"(x_2[10]), "f"(x_2[10]), "f"(ss_2[1]));
                ss_2[1] = _fma_224;
                float _fma_225;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_225) : "f"(x_2[11]), "f"(x_2[11]), "f"(ss_2[1]));
                ss_2[1] = _fma_225;
                float _fma_226;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_226) : "f"(x_2[12]), "f"(x_2[12]), "f"(ss_2[1]));
                ss_2[1] = _fma_226;
                float _fma_227;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_227) : "f"(x_2[13]), "f"(x_2[13]), "f"(ss_2[1]));
                ss_2[1] = _fma_227;
                float _fma_228;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_228) : "f"(x_2[14]), "f"(x_2[14]), "f"(ss_2[1]));
                ss_2[1] = _fma_228;
                float _fma_229;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_229) : "f"(x_2[15]), "f"(x_2[15]), "f"(ss_2[1]));
                ss_2[1] = _fma_229;
                ss_2[2] = mul_f32_rn_noftz(x_2[16], x_2[16]);
                float _fma_230;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_230) : "f"(x_2[17]), "f"(x_2[17]), "f"(ss_2[2]));
                ss_2[2] = _fma_230;
                float _fma_231;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_231) : "f"(x_2[18]), "f"(x_2[18]), "f"(ss_2[2]));
                ss_2[2] = _fma_231;
                float _fma_232;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_232) : "f"(x_2[19]), "f"(x_2[19]), "f"(ss_2[2]));
                ss_2[2] = _fma_232;
                float _fma_233;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_233) : "f"(x_2[20]), "f"(x_2[20]), "f"(ss_2[2]));
                ss_2[2] = _fma_233;
                float _fma_234;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_234) : "f"(x_2[21]), "f"(x_2[21]), "f"(ss_2[2]));
                ss_2[2] = _fma_234;
                float _fma_235;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_235) : "f"(x_2[22]), "f"(x_2[22]), "f"(ss_2[2]));
                ss_2[2] = _fma_235;
                float _fma_236;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_236) : "f"(x_2[23]), "f"(x_2[23]), "f"(ss_2[2]));
                ss_2[2] = _fma_236;
                ss_2[3] = mul_f32_rn_noftz(x_2[24], x_2[24]);
                float _fma_237;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_237) : "f"(x_2[25]), "f"(x_2[25]), "f"(ss_2[3]));
                ss_2[3] = _fma_237;
                float _fma_238;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_238) : "f"(x_2[26]), "f"(x_2[26]), "f"(ss_2[3]));
                ss_2[3] = _fma_238;
                float _fma_239;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_239) : "f"(x_2[27]), "f"(x_2[27]), "f"(ss_2[3]));
                ss_2[3] = _fma_239;
                float _fma_240;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_240) : "f"(x_2[28]), "f"(x_2[28]), "f"(ss_2[3]));
                ss_2[3] = _fma_240;
                float _fma_241;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_241) : "f"(x_2[29]), "f"(x_2[29]), "f"(ss_2[3]));
                ss_2[3] = _fma_241;
                float _fma_242;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_242) : "f"(x_2[30]), "f"(x_2[30]), "f"(ss_2[3]));
                ss_2[3] = _fma_242;
                float _fma_243;
                asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(_fma_243) : "f"(x_2[31]), "f"(x_2[31]), "f"(ss_2[3]));
                ss_2[3] = _fma_243;
                float _shfl_xor_sync_245 = __shfl_xor_sync(0xFFFFFFFF, ss_2[0], 8);
                float _add_244;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_244) : "f"(ss_2[0]), "f"(_shfl_xor_sync_245));
                ss_2[0] = _add_244;
                float _shfl_xor_sync_247 = __shfl_xor_sync(0xFFFFFFFF, ss_2[1], 8);
                float _add_246;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_246) : "f"(ss_2[1]), "f"(_shfl_xor_sync_247));
                ss_2[1] = _add_246;
                float _shfl_xor_sync_249 = __shfl_xor_sync(0xFFFFFFFF, ss_2[2], 8);
                float _add_248;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_248) : "f"(ss_2[2]), "f"(_shfl_xor_sync_249));
                ss_2[2] = _add_248;
                float _shfl_xor_sync_251 = __shfl_xor_sync(0xFFFFFFFF, ss_2[3], 8);
                float _add_250;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_250) : "f"(ss_2[3]), "f"(_shfl_xor_sync_251));
                ss_2[3] = _add_250;
                float _shfl_xor_sync_253 = __shfl_xor_sync(0xFFFFFFFF, ss_2[0], 4);
                float _add_252;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_252) : "f"(ss_2[0]), "f"(_shfl_xor_sync_253));
                ss_2[0] = _add_252;
                float _shfl_xor_sync_255 = __shfl_xor_sync(0xFFFFFFFF, ss_2[1], 4);
                float _add_254;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_254) : "f"(ss_2[1]), "f"(_shfl_xor_sync_255));
                ss_2[1] = _add_254;
                float _shfl_xor_sync_257 = __shfl_xor_sync(0xFFFFFFFF, ss_2[2], 4);
                float _add_256;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_256) : "f"(ss_2[2]), "f"(_shfl_xor_sync_257));
                ss_2[2] = _add_256;
                float _shfl_xor_sync_259 = __shfl_xor_sync(0xFFFFFFFF, ss_2[3], 4);
                float _add_258;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_258) : "f"(ss_2[3]), "f"(_shfl_xor_sync_259));
                ss_2[3] = _add_258;
                float _shfl_xor_sync_261 = __shfl_xor_sync(0xFFFFFFFF, ss_2[0], 2);
                float _add_260;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_260) : "f"(ss_2[0]), "f"(_shfl_xor_sync_261));
                ss_2[0] = _add_260;
                float _shfl_xor_sync_263 = __shfl_xor_sync(0xFFFFFFFF, ss_2[1], 2);
                float _add_262;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_262) : "f"(ss_2[1]), "f"(_shfl_xor_sync_263));
                ss_2[1] = _add_262;
                float _shfl_xor_sync_265 = __shfl_xor_sync(0xFFFFFFFF, ss_2[2], 2);
                float _add_264;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_264) : "f"(ss_2[2]), "f"(_shfl_xor_sync_265));
                ss_2[2] = _add_264;
                float _shfl_xor_sync_267 = __shfl_xor_sync(0xFFFFFFFF, ss_2[3], 2);
                float _add_266;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_266) : "f"(ss_2[3]), "f"(_shfl_xor_sync_267));
                ss_2[3] = _add_266;
                float _shfl_xor_sync_269 = __shfl_xor_sync(0xFFFFFFFF, ss_2[0], 1);
                float _add_268;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_268) : "f"(ss_2[0]), "f"(_shfl_xor_sync_269));
                ss_2[0] = _add_268;
                float _shfl_xor_sync_271 = __shfl_xor_sync(0xFFFFFFFF, ss_2[1], 1);
                float _add_270;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_270) : "f"(ss_2[1]), "f"(_shfl_xor_sync_271));
                ss_2[1] = _add_270;
                float _shfl_xor_sync_273 = __shfl_xor_sync(0xFFFFFFFF, ss_2[2], 1);
                float _add_272;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_272) : "f"(ss_2[2]), "f"(_shfl_xor_sync_273));
                ss_2[2] = _add_272;
                float _shfl_xor_sync_275 = __shfl_xor_sync(0xFFFFFFFF, ss_2[3], 1);
                float _add_274;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_274) : "f"(ss_2[3]), "f"(_shfl_xor_sync_275));
                ss_2[3] = _add_274;
                float r_2[4];
                float _add_278;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_278) : "f"(ss_2[0]), "f"(1e-06f));
                float _sqrt_277;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_277) : "f"(_add_278));
                float _fdiv_276 = __fdiv_rn(1.0f, _sqrt_277);
                r_2[0] = _fdiv_276;
                float _add_281;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_281) : "f"(ss_2[1]), "f"(1e-06f));
                float _sqrt_280;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_280) : "f"(_add_281));
                float _fdiv_279 = __fdiv_rn(1.0f, _sqrt_280);
                r_2[1] = _fdiv_279;
                float _add_284;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_284) : "f"(ss_2[2]), "f"(1e-06f));
                float _sqrt_283;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_283) : "f"(_add_284));
                float _fdiv_282 = __fdiv_rn(1.0f, _sqrt_283);
                r_2[2] = _fdiv_282;
                float _add_287;
                asm("add.rn.f32 %0, %1, %2;" : "=f"(_add_287) : "f"(ss_2[3]), "f"(1e-06f));
                float _sqrt_286;
                asm volatile("sqrt.rn.f32 %0, %1;" : "=f"(_sqrt_286) : "f"(_add_287));
                float _fdiv_285 = __fdiv_rn(1.0f, _sqrt_286);
                r_2[3] = _fdiv_285;
                float y_2[32];
                y_2[0] = mul_f32_rn_noftz(x_2[0], r_2[0]);
                y_2[1] = mul_f32_rn_noftz(x_2[1], r_2[0]);
                y_2[2] = mul_f32_rn_noftz(x_2[2], r_2[0]);
                y_2[3] = mul_f32_rn_noftz(x_2[3], r_2[0]);
                y_2[4] = mul_f32_rn_noftz(x_2[4], r_2[0]);
                y_2[5] = mul_f32_rn_noftz(x_2[5], r_2[0]);
                y_2[6] = mul_f32_rn_noftz(x_2[6], r_2[0]);
                y_2[7] = mul_f32_rn_noftz(x_2[7], r_2[0]);
                y_2[8] = mul_f32_rn_noftz(x_2[8], r_2[1]);
                y_2[9] = mul_f32_rn_noftz(x_2[9], r_2[1]);
                y_2[10] = mul_f32_rn_noftz(x_2[10], r_2[1]);
                y_2[11] = mul_f32_rn_noftz(x_2[11], r_2[1]);
                y_2[12] = mul_f32_rn_noftz(x_2[12], r_2[1]);
                y_2[13] = mul_f32_rn_noftz(x_2[13], r_2[1]);
                y_2[14] = mul_f32_rn_noftz(x_2[14], r_2[1]);
                y_2[15] = mul_f32_rn_noftz(x_2[15], r_2[1]);
                y_2[16] = mul_f32_rn_noftz(x_2[16], r_2[2]);
                y_2[17] = mul_f32_rn_noftz(x_2[17], r_2[2]);
                y_2[18] = mul_f32_rn_noftz(x_2[18], r_2[2]);
                y_2[19] = mul_f32_rn_noftz(x_2[19], r_2[2]);
                y_2[20] = mul_f32_rn_noftz(x_2[20], r_2[2]);
                y_2[21] = mul_f32_rn_noftz(x_2[21], r_2[2]);
                y_2[22] = mul_f32_rn_noftz(x_2[22], r_2[2]);
                y_2[23] = mul_f32_rn_noftz(x_2[23], r_2[2]);
                y_2[24] = mul_f32_rn_noftz(x_2[24], r_2[3]);
                y_2[25] = mul_f32_rn_noftz(x_2[25], r_2[3]);
                y_2[26] = mul_f32_rn_noftz(x_2[26], r_2[3]);
                y_2[27] = mul_f32_rn_noftz(x_2[27], r_2[3]);
                y_2[28] = mul_f32_rn_noftz(x_2[28], r_2[3]);
                y_2[29] = mul_f32_rn_noftz(x_2[29], r_2[3]);
                y_2[30] = mul_f32_rn_noftz(x_2[30], r_2[3]);
                y_2[31] = mul_f32_rn_noftz(x_2[31], r_2[3]);
                unsigned int yb_2[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(y_2[_lp*2 + 0], y_2[_lp*2+1 + 0]));
                    yb_2[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 32) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_2[0]), "r"(yb_2[1]), "r"(yb_2[2]), "r"(yb_2[3]) : "memory");
                if (l2n_rsub + 32 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 32) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_2)[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 32) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_2[0];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 40) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_2[4]), "r"(yb_2[5]), "r"(yb_2[6]), "r"(yb_2[7]) : "memory");
                if (l2n_rsub + 40 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 40) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_2 + (4))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 40) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_2[1];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 48) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_2[8]), "r"(yb_2[9]), "r"(yb_2[10]), "r"(yb_2[11]) : "memory");
                if (l2n_rsub + 48 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 48) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_2 + (8))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 48) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_2[2];
                    }
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((k_k_addr + (unsigned int)(l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 ^ (l2n_c * 8 / 64 * 8192 + (l2n_rsub + 56) * 128 + l2n_c * 8 % 64 * 2 >> 7 & 7) << 4))), "r"(yb_2[12]), "r"(yb_2[13]), "r"(yb_2[14]), "r"(yb_2[15]) : "memory");
                if (l2n_rsub + 56 < clen) {
                    reinterpret_cast<int4*>(kn_out + (((long long)(ext0 + l2n_rsub + 56) * (long long)num_qk_heads + (long long)hq) * 128ll + (long long)(l2n_c * 8)))[0] = reinterpret_cast<int4*>(yb_2 + (12))[0];
                    if (l2n_c == 0) {
                        *(reinterpret_cast<float*>(k_rstd + ((long long)(ext0 + l2n_rsub + 56) * (long long)rstd_tstride + (long long)hq)) + (0)) = r_2[3];
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(qkn_ready_addr);
                mbarrier_wait(gram_full_addr, _phase_gram_full_0);
                _phase_gram_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sqk[0])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[1])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[2])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[3])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[4])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[5])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[6])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[7])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[8])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[9])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[10])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[11])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[12])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[13])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[14])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[15]))
                    : "r"(taddr + (unsigned int)row_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sqk[16])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[17])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[18])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[19])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[20])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[21])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[22])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[23])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[24])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[25])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[26])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[27])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[28])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[29])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[30])), "=r"(*reinterpret_cast<uint32_t*>(&sqk[31]))
                    : "r"(taddr + (unsigned int)row_base + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int row_valid = 0;
                if (t_row < clen) {
                    row_valid = 1;
                }
                #pragma unroll 1
                for (int j = 0; j < group + 1; j++) {
                    int hv = hq * group + j;
                    if (j > 1) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    if (group > j) {
                        if (tid_0 < 64) {
                            gs_s[tid_0] = gv_n;
                            beta_s[tid_0] = bv_n;
                            if (group > j + 1) {
                                if (tid_0 < clen) {
                                    long long tok_n = (long long)(ext0 + tid_0) * (long long)num_heads + (long long)(hv + 1);
                                    gv_n = g[tok_n];
                                    bv_n = beta[tok_n];
                                }
                            }
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (w_id == 0) {
                            float g0[1];
                            float g1[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&g0[0])) : "r"(gs_base + lane_1 * 4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&g1[0])) : "r"(gs_base + (32 + lane_1) * 4));
                            float x0 = g0[0];
                            float x1 = g1[0];
                            float y0_1_0 = __shfl_up_sync(0xFFFFFFFF, x0, 1, 32);
                            float y1_2_0 = __shfl_up_sync(0xFFFFFFFF, x1, 1, 32);
                            x0 = x0 + ((lane_1 >= 1) ? y0_1_0 : 0.0f);
                            x1 = x1 + ((lane_1 >= 1) ? y1_2_0 : 0.0f);
                            float y0_1_1 = __shfl_up_sync(0xFFFFFFFF, x0, 2, 32);
                            float y1_2_1 = __shfl_up_sync(0xFFFFFFFF, x1, 2, 32);
                            x0 = x0 + ((lane_1 >= 2) ? y0_1_1 : 0.0f);
                            x1 = x1 + ((lane_1 >= 2) ? y1_2_1 : 0.0f);
                            float y0_1_2 = __shfl_up_sync(0xFFFFFFFF, x0, 4, 32);
                            float y1_2_2 = __shfl_up_sync(0xFFFFFFFF, x1, 4, 32);
                            x0 = x0 + ((lane_1 >= 4) ? y0_1_2 : 0.0f);
                            x1 = x1 + ((lane_1 >= 4) ? y1_2_2 : 0.0f);
                            float y0_1_3 = __shfl_up_sync(0xFFFFFFFF, x0, 8, 32);
                            float y1_2_3 = __shfl_up_sync(0xFFFFFFFF, x1, 8, 32);
                            x0 = x0 + ((lane_1 >= 8) ? y0_1_3 : 0.0f);
                            x1 = x1 + ((lane_1 >= 8) ? y1_2_3 : 0.0f);
                            float y0_1_4 = __shfl_up_sync(0xFFFFFFFF, x0, 16, 32);
                            float y1_2_4 = __shfl_up_sync(0xFFFFFFFF, x1, 16, 32);
                            x0 = x0 + ((lane_1 >= 16) ? y0_1_4 : 0.0f);
                            x1 = x1 + ((lane_1 >= 16) ? y1_2_4 : 0.0f);
                            float tot0 = __shfl_sync(0xFFFFFFFF, x0, 31);
                            x1 = x1 + tot0;
                            gc_s[lane_1] = x0;
                            gc_s[32 + lane_1] = x1;
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (tid_0 < 64) {
                            float gt_r[1];
                            float gl_r[1];
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gt_r[0])) : "r"(gc_base + tid_0 * 4));
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gl_r[0])) : "r"(gc_base + (clen - 1) * 4));
                            float gt = gt_r[0];
                            float gl = gl_r[0];
                            float eg = approx_exp2(gt * 1.442695f);
                            float rs = approx_exp2((gl - gt) * 1.442695f);
                            int valid = 0;
                            if (tid_0 < clen) {
                                valid = 1;
                            }
                            float eg_v = ((valid != 0) ? eg : 0.0f);
                            float rs_v = ((valid != 0) ? rs : 0.0f);
                            float gt_v = ((valid != 0) ? gt : 0.0f);
                            eg_s[tid_0] = eg_v;
                            long long tape = (long long)(irow0 + tid_0) * (long long)num_heads + (long long)hv;
                            *(reinterpret_cast<float*>(g_cum + tape) + (0)) = gt_v;
                            *(reinterpret_cast<float*>(rowscale + tape) + (0)) = rs_v;
                            *(reinterpret_cast<float*>(qscale + tape) + (0)) = eg_v * scale;
                            if (tid_0 == 0) {
                                float gam = approx_exp2(gl * 1.442695f);
                                *(reinterpret_cast<float*>(gamma + ((long long)chunk * (long long)num_heads + (long long)hv)) + (0)) = gam;
                            }
                        }
                        float bt_r[1];
                        float gtr_r[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bt_r[0])) : "r"(beta_base + t_row * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&gtr_r[0])) : "r"(gc_base + t_row * 4));
                        float bt = bt_r[0];
                        float gtr = gtr_r[0];
                        long long aqk_base = ((long long)(irow0 + t_row) * (long long)num_heads + (long long)hv) * 64ll;
                        int s0_1_0 = half * 16;
                        float gsv_1_0[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(0) + 3]))
                            : "r"(gc_base + s0_1_0 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(4) + 3]))
                            : "r"(gc_base + (s0_1_0 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(8) + 3]))
                            : "r"(gc_base + (s0_1_0 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_0[(12) + 3]))
                            : "r"(gc_base + (s0_1_0 + 12) * 4));
                        float aq_1_0[16];
                        int s_1_0 = s0_1_0;
                        float fac_2_0 = approx_exp2(fminf(gtr - gsv_1_0[0], 0.0f) * 1.442695f);
                        int on_aqk_1_0 = 0;
                        if (s_1_0 <= t_row) {
                            on_aqk_1_0 = row_valid;
                        }
                        aq_1_0[0] = ((on_aqk_1_0 != 0) ? sqk[0] * scale * fac_2_0 : 0.0f);
                        int s_1_1 = s0_1_0 + 1;
                        float fac_2_1 = approx_exp2(fminf(gtr - gsv_1_0[1], 0.0f) * 1.442695f);
                        int on_aqk_1_1 = 0;
                        if (s_1_1 <= t_row) {
                            on_aqk_1_1 = row_valid;
                        }
                        aq_1_0[1] = ((on_aqk_1_1 != 0) ? sqk[1] * scale * fac_2_1 : 0.0f);
                        int s_1_2 = s0_1_0 + 2;
                        float fac_2_2 = approx_exp2(fminf(gtr - gsv_1_0[2], 0.0f) * 1.442695f);
                        int on_aqk_1_2 = 0;
                        if (s_1_2 <= t_row) {
                            on_aqk_1_2 = row_valid;
                        }
                        aq_1_0[2] = ((on_aqk_1_2 != 0) ? sqk[2] * scale * fac_2_2 : 0.0f);
                        int s_1_3 = s0_1_0 + 3;
                        float fac_2_3 = approx_exp2(fminf(gtr - gsv_1_0[3], 0.0f) * 1.442695f);
                        int on_aqk_1_3 = 0;
                        if (s_1_3 <= t_row) {
                            on_aqk_1_3 = row_valid;
                        }
                        aq_1_0[3] = ((on_aqk_1_3 != 0) ? sqk[3] * scale * fac_2_3 : 0.0f);
                        int s_1_4 = s0_1_0 + 4;
                        float fac_2_4 = approx_exp2(fminf(gtr - gsv_1_0[4], 0.0f) * 1.442695f);
                        int on_aqk_1_4 = 0;
                        if (s_1_4 <= t_row) {
                            on_aqk_1_4 = row_valid;
                        }
                        aq_1_0[4] = ((on_aqk_1_4 != 0) ? sqk[4] * scale * fac_2_4 : 0.0f);
                        int s_1_5 = s0_1_0 + 5;
                        float fac_2_5 = approx_exp2(fminf(gtr - gsv_1_0[5], 0.0f) * 1.442695f);
                        int on_aqk_1_5 = 0;
                        if (s_1_5 <= t_row) {
                            on_aqk_1_5 = row_valid;
                        }
                        aq_1_0[5] = ((on_aqk_1_5 != 0) ? sqk[5] * scale * fac_2_5 : 0.0f);
                        int s_1_6 = s0_1_0 + 6;
                        float fac_2_6 = approx_exp2(fminf(gtr - gsv_1_0[6], 0.0f) * 1.442695f);
                        int on_aqk_1_6 = 0;
                        if (s_1_6 <= t_row) {
                            on_aqk_1_6 = row_valid;
                        }
                        aq_1_0[6] = ((on_aqk_1_6 != 0) ? sqk[6] * scale * fac_2_6 : 0.0f);
                        int s_1_7 = s0_1_0 + 7;
                        float fac_2_7 = approx_exp2(fminf(gtr - gsv_1_0[7], 0.0f) * 1.442695f);
                        int on_aqk_1_7 = 0;
                        if (s_1_7 <= t_row) {
                            on_aqk_1_7 = row_valid;
                        }
                        aq_1_0[7] = ((on_aqk_1_7 != 0) ? sqk[7] * scale * fac_2_7 : 0.0f);
                        int s_1_8 = s0_1_0 + 8;
                        float fac_2_8 = approx_exp2(fminf(gtr - gsv_1_0[8], 0.0f) * 1.442695f);
                        int on_aqk_1_8 = 0;
                        if (s_1_8 <= t_row) {
                            on_aqk_1_8 = row_valid;
                        }
                        aq_1_0[8] = ((on_aqk_1_8 != 0) ? sqk[8] * scale * fac_2_8 : 0.0f);
                        int s_1_9 = s0_1_0 + 9;
                        float fac_2_9 = approx_exp2(fminf(gtr - gsv_1_0[9], 0.0f) * 1.442695f);
                        int on_aqk_1_9 = 0;
                        if (s_1_9 <= t_row) {
                            on_aqk_1_9 = row_valid;
                        }
                        aq_1_0[9] = ((on_aqk_1_9 != 0) ? sqk[9] * scale * fac_2_9 : 0.0f);
                        int s_1_10 = s0_1_0 + 10;
                        float fac_2_10 = approx_exp2(fminf(gtr - gsv_1_0[10], 0.0f) * 1.442695f);
                        int on_aqk_1_10 = 0;
                        if (s_1_10 <= t_row) {
                            on_aqk_1_10 = row_valid;
                        }
                        aq_1_0[10] = ((on_aqk_1_10 != 0) ? sqk[10] * scale * fac_2_10 : 0.0f);
                        int s_1_11 = s0_1_0 + 11;
                        float fac_2_11 = approx_exp2(fminf(gtr - gsv_1_0[11], 0.0f) * 1.442695f);
                        int on_aqk_1_11 = 0;
                        if (s_1_11 <= t_row) {
                            on_aqk_1_11 = row_valid;
                        }
                        aq_1_0[11] = ((on_aqk_1_11 != 0) ? sqk[11] * scale * fac_2_11 : 0.0f);
                        int s_1_12 = s0_1_0 + 12;
                        float fac_2_12 = approx_exp2(fminf(gtr - gsv_1_0[12], 0.0f) * 1.442695f);
                        int on_aqk_1_12 = 0;
                        if (s_1_12 <= t_row) {
                            on_aqk_1_12 = row_valid;
                        }
                        aq_1_0[12] = ((on_aqk_1_12 != 0) ? sqk[12] * scale * fac_2_12 : 0.0f);
                        int s_1_13 = s0_1_0 + 13;
                        float fac_2_13 = approx_exp2(fminf(gtr - gsv_1_0[13], 0.0f) * 1.442695f);
                        int on_aqk_1_13 = 0;
                        if (s_1_13 <= t_row) {
                            on_aqk_1_13 = row_valid;
                        }
                        aq_1_0[13] = ((on_aqk_1_13 != 0) ? sqk[13] * scale * fac_2_13 : 0.0f);
                        int s_1_14 = s0_1_0 + 14;
                        float fac_2_14 = approx_exp2(fminf(gtr - gsv_1_0[14], 0.0f) * 1.442695f);
                        int on_aqk_1_14 = 0;
                        if (s_1_14 <= t_row) {
                            on_aqk_1_14 = row_valid;
                        }
                        aq_1_0[14] = ((on_aqk_1_14 != 0) ? sqk[14] * scale * fac_2_14 : 0.0f);
                        int s_1_15 = s0_1_0 + 15;
                        float fac_2_15 = approx_exp2(fminf(gtr - gsv_1_0[15], 0.0f) * 1.442695f);
                        int on_aqk_1_15 = 0;
                        if (s_1_15 <= t_row) {
                            on_aqk_1_15 = row_valid;
                        }
                        aq_1_0[15] = ((on_aqk_1_15 != 0) ? sqk[15] * scale * fac_2_15 : 0.0f);
                        unsigned int _packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 0], aq_1_0[_lp*2+1 + 0]));
                            _packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 2], aq_1_0[_lp*2+1 + 2]));
                            _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 4], aq_1_0[_lp*2+1 + 4]));
                            _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 6], aq_1_0[_lp*2+1 + 6]));
                            _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 8], aq_1_0[_lp*2+1 + 8]));
                            _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 10], aq_1_0[_lp*2+1 + 10]));
                            _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 12], aq_1_0[_lp*2+1 + 12]));
                            _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_0[_lp*2 + 14], aq_1_0[_lp*2+1 + 14]));
                            _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_288 = reinterpret_cast<const unsigned*>(_packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0_1_0))), "r"(_raw_stv8_288[0]), "r"(_raw_stv8_288[1]), "r"(_raw_stv8_288[2]), "r"(_raw_stv8_288[3]), "r"(_raw_stv8_288[4]), "r"(_raw_stv8_288[5]), "r"(_raw_stv8_288[6]), "r"(_raw_stv8_288[7]) : "memory");
                        }
                        int s0_1_1 = 32 + half * 16;
                        float gsv_1_1[16];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(0) + 3]))
                            : "r"(gc_base + s0_1_1 * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(4) + 3]))
                            : "r"(gc_base + (s0_1_1 + 4) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(8) + 3]))
                            : "r"(gc_base + (s0_1_1 + 8) * 4));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&gsv_1_1[(12) + 3]))
                            : "r"(gc_base + (s0_1_1 + 12) * 4));
                        float aq_1_1[16];
                        int s_1_16 = s0_1_1;
                        float fac_2_16 = approx_exp2(fminf(gtr - gsv_1_1[0], 0.0f) * 1.442695f);
                        int on_aqk_1_16 = 0;
                        if (s_1_16 <= t_row) {
                            on_aqk_1_16 = row_valid;
                        }
                        aq_1_1[0] = ((on_aqk_1_16 != 0) ? sqk[16] * scale * fac_2_16 : 0.0f);
                        int s_1_17 = s0_1_1 + 1;
                        float fac_2_17 = approx_exp2(fminf(gtr - gsv_1_1[1], 0.0f) * 1.442695f);
                        int on_aqk_1_17 = 0;
                        if (s_1_17 <= t_row) {
                            on_aqk_1_17 = row_valid;
                        }
                        aq_1_1[1] = ((on_aqk_1_17 != 0) ? sqk[17] * scale * fac_2_17 : 0.0f);
                        int s_1_18 = s0_1_1 + 2;
                        float fac_2_18 = approx_exp2(fminf(gtr - gsv_1_1[2], 0.0f) * 1.442695f);
                        int on_aqk_1_18 = 0;
                        if (s_1_18 <= t_row) {
                            on_aqk_1_18 = row_valid;
                        }
                        aq_1_1[2] = ((on_aqk_1_18 != 0) ? sqk[18] * scale * fac_2_18 : 0.0f);
                        int s_1_19 = s0_1_1 + 3;
                        float fac_2_19 = approx_exp2(fminf(gtr - gsv_1_1[3], 0.0f) * 1.442695f);
                        int on_aqk_1_19 = 0;
                        if (s_1_19 <= t_row) {
                            on_aqk_1_19 = row_valid;
                        }
                        aq_1_1[3] = ((on_aqk_1_19 != 0) ? sqk[19] * scale * fac_2_19 : 0.0f);
                        int s_1_20 = s0_1_1 + 4;
                        float fac_2_20 = approx_exp2(fminf(gtr - gsv_1_1[4], 0.0f) * 1.442695f);
                        int on_aqk_1_20 = 0;
                        if (s_1_20 <= t_row) {
                            on_aqk_1_20 = row_valid;
                        }
                        aq_1_1[4] = ((on_aqk_1_20 != 0) ? sqk[20] * scale * fac_2_20 : 0.0f);
                        int s_1_21 = s0_1_1 + 5;
                        float fac_2_21 = approx_exp2(fminf(gtr - gsv_1_1[5], 0.0f) * 1.442695f);
                        int on_aqk_1_21 = 0;
                        if (s_1_21 <= t_row) {
                            on_aqk_1_21 = row_valid;
                        }
                        aq_1_1[5] = ((on_aqk_1_21 != 0) ? sqk[21] * scale * fac_2_21 : 0.0f);
                        int s_1_22 = s0_1_1 + 6;
                        float fac_2_22 = approx_exp2(fminf(gtr - gsv_1_1[6], 0.0f) * 1.442695f);
                        int on_aqk_1_22 = 0;
                        if (s_1_22 <= t_row) {
                            on_aqk_1_22 = row_valid;
                        }
                        aq_1_1[6] = ((on_aqk_1_22 != 0) ? sqk[22] * scale * fac_2_22 : 0.0f);
                        int s_1_23 = s0_1_1 + 7;
                        float fac_2_23 = approx_exp2(fminf(gtr - gsv_1_1[7], 0.0f) * 1.442695f);
                        int on_aqk_1_23 = 0;
                        if (s_1_23 <= t_row) {
                            on_aqk_1_23 = row_valid;
                        }
                        aq_1_1[7] = ((on_aqk_1_23 != 0) ? sqk[23] * scale * fac_2_23 : 0.0f);
                        int s_1_24 = s0_1_1 + 8;
                        float fac_2_24 = approx_exp2(fminf(gtr - gsv_1_1[8], 0.0f) * 1.442695f);
                        int on_aqk_1_24 = 0;
                        if (s_1_24 <= t_row) {
                            on_aqk_1_24 = row_valid;
                        }
                        aq_1_1[8] = ((on_aqk_1_24 != 0) ? sqk[24] * scale * fac_2_24 : 0.0f);
                        int s_1_25 = s0_1_1 + 9;
                        float fac_2_25 = approx_exp2(fminf(gtr - gsv_1_1[9], 0.0f) * 1.442695f);
                        int on_aqk_1_25 = 0;
                        if (s_1_25 <= t_row) {
                            on_aqk_1_25 = row_valid;
                        }
                        aq_1_1[9] = ((on_aqk_1_25 != 0) ? sqk[25] * scale * fac_2_25 : 0.0f);
                        int s_1_26 = s0_1_1 + 10;
                        float fac_2_26 = approx_exp2(fminf(gtr - gsv_1_1[10], 0.0f) * 1.442695f);
                        int on_aqk_1_26 = 0;
                        if (s_1_26 <= t_row) {
                            on_aqk_1_26 = row_valid;
                        }
                        aq_1_1[10] = ((on_aqk_1_26 != 0) ? sqk[26] * scale * fac_2_26 : 0.0f);
                        int s_1_27 = s0_1_1 + 11;
                        float fac_2_27 = approx_exp2(fminf(gtr - gsv_1_1[11], 0.0f) * 1.442695f);
                        int on_aqk_1_27 = 0;
                        if (s_1_27 <= t_row) {
                            on_aqk_1_27 = row_valid;
                        }
                        aq_1_1[11] = ((on_aqk_1_27 != 0) ? sqk[27] * scale * fac_2_27 : 0.0f);
                        int s_1_28 = s0_1_1 + 12;
                        float fac_2_28 = approx_exp2(fminf(gtr - gsv_1_1[12], 0.0f) * 1.442695f);
                        int on_aqk_1_28 = 0;
                        if (s_1_28 <= t_row) {
                            on_aqk_1_28 = row_valid;
                        }
                        aq_1_1[12] = ((on_aqk_1_28 != 0) ? sqk[28] * scale * fac_2_28 : 0.0f);
                        int s_1_29 = s0_1_1 + 13;
                        float fac_2_29 = approx_exp2(fminf(gtr - gsv_1_1[13], 0.0f) * 1.442695f);
                        int on_aqk_1_29 = 0;
                        if (s_1_29 <= t_row) {
                            on_aqk_1_29 = row_valid;
                        }
                        aq_1_1[13] = ((on_aqk_1_29 != 0) ? sqk[29] * scale * fac_2_29 : 0.0f);
                        int s_1_30 = s0_1_1 + 14;
                        float fac_2_30 = approx_exp2(fminf(gtr - gsv_1_1[14], 0.0f) * 1.442695f);
                        int on_aqk_1_30 = 0;
                        if (s_1_30 <= t_row) {
                            on_aqk_1_30 = row_valid;
                        }
                        aq_1_1[14] = ((on_aqk_1_30 != 0) ? sqk[30] * scale * fac_2_30 : 0.0f);
                        int s_1_31 = s0_1_1 + 15;
                        float fac_2_31 = approx_exp2(fminf(gtr - gsv_1_1[15], 0.0f) * 1.442695f);
                        int on_aqk_1_31 = 0;
                        if (s_1_31 <= t_row) {
                            on_aqk_1_31 = row_valid;
                        }
                        aq_1_1[15] = ((on_aqk_1_31 != 0) ? sqk[31] * scale * fac_2_31 : 0.0f);
                        unsigned int _packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 0], aq_1_1[_lp*2+1 + 0]));
                            _packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 2], aq_1_1[_lp*2+1 + 2]));
                            _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 4], aq_1_1[_lp*2+1 + 4]));
                            _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 6], aq_1_1[_lp*2+1 + 6]));
                            _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 8], aq_1_1[_lp*2+1 + 8]));
                            _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 10], aq_1_1[_lp*2+1 + 10]));
                            _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 12], aq_1_1[_lp*2+1 + 12]));
                            _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(aq_1_1[_lp*2 + 14], aq_1_1[_lp*2+1 + 14]));
                            _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_289 = reinterpret_cast<const unsigned*>(_packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (aqk_base + (long long)s0_1_1))), "r"(_raw_stv8_289[0]), "r"(_raw_stv8_289[1]), "r"(_raw_stv8_289[2]), "r"(_raw_stv8_289[3]), "r"(_raw_stv8_289[4]), "r"(_raw_stv8_289[5]), "r"(_raw_stv8_289[6]), "r"(_raw_stv8_289[7]) : "memory");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        int t2 = tid_0;
                        mbarrier_wait(v_full_addr + (vst_c) * 8, _phase_v_full);
                        int qi_1_0 = t2;
                        int srow_2_0 = qi_1_0 >> 4;
                        int d0_1_0 = (qi_1_0 & 15) * 8;
                        float bs_r_1_0[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_0[0])) : "r"(beta_base + srow_2_0 * 4));
                        float fvb_1_0 = bs_r_1_0[0];
                        unsigned int vaddr_1_0 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_0 / 64 * 8192 + srow_2_0 * 128 + d0_1_0 % 64 * 2 ^ (d0_1_0 / 64 * 8192 + srow_2_0 * 128 + d0_1_0 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w[0])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w[(0) + 3]))
                            : "r"(vaddr_1_0));
                        float unpack_bf16[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w[_pair]) << 16);
                            (&unpack_bf16[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w[_pair]) & 0xffff0000u);
                        }
                        float s[8];
                        s[0] = unpack_bf16[0] * fvb_1_0;
                        s[1] = unpack_bf16[1] * fvb_1_0;
                        s[2] = unpack_bf16[2] * fvb_1_0;
                        s[3] = unpack_bf16[3] * fvb_1_0;
                        s[4] = unpack_bf16[4] * fvb_1_0;
                        s[5] = unpack_bf16[5] * fvb_1_0;
                        s[6] = unpack_bf16[6] * fvb_1_0;
                        s[7] = unpack_bf16[7] * fvb_1_0;
                        unsigned int s_1_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s[_lp*2 + 0], s[_lp*2+1 + 0]));
                            s_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_0), "r"(s_1_bf16[0]), "r"(s_1_bf16[1]), "r"(s_1_bf16[2]), "r"(s_1_bf16[3]) : "memory");
                        int qi_1_1 = t2 + 128;
                        int srow_2_1 = qi_1_1 >> 4;
                        int d0_1_1 = (qi_1_1 & 15) * 8;
                        float bs_r_1_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_1[0])) : "r"(beta_base + srow_2_1 * 4));
                        float fvb_1_1 = bs_r_1_1[0];
                        unsigned int vaddr_1_1 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_1 / 64 * 8192 + srow_2_1 * 128 + d0_1_1 % 64 * 2 ^ (d0_1_1 / 64 * 8192 + srow_2_1 * 128 + d0_1_1 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_0[(0) + 3]))
                            : "r"(vaddr_1_1));
                        float unpack_bf16_0[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_0[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_0[_pair]) << 16);
                            (&unpack_bf16_0[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_0[_pair]) & 0xffff0000u);
                        }
                        float s_0[8];
                        s_0[0] = unpack_bf16_0[0] * fvb_1_1;
                        s_0[1] = unpack_bf16_0[1] * fvb_1_1;
                        s_0[2] = unpack_bf16_0[2] * fvb_1_1;
                        s_0[3] = unpack_bf16_0[3] * fvb_1_1;
                        s_0[4] = unpack_bf16_0[4] * fvb_1_1;
                        s_0[5] = unpack_bf16_0[5] * fvb_1_1;
                        s_0[6] = unpack_bf16_0[6] * fvb_1_1;
                        s_0[7] = unpack_bf16_0[7] * fvb_1_1;
                        unsigned int s_1_bf16_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_0[_lp*2 + 0], s_0[_lp*2+1 + 0]));
                            s_1_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_1), "r"(s_1_bf16_0[0]), "r"(s_1_bf16_0[1]), "r"(s_1_bf16_0[2]), "r"(s_1_bf16_0[3]) : "memory");
                        int qi_1_2 = t2 + 256;
                        int srow_2_2 = qi_1_2 >> 4;
                        int d0_1_2 = (qi_1_2 & 15) * 8;
                        float bs_r_1_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_2[0])) : "r"(beta_base + srow_2_2 * 4));
                        float fvb_1_2 = bs_r_1_2[0];
                        unsigned int vaddr_1_2 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_2 / 64 * 8192 + srow_2_2 * 128 + d0_1_2 % 64 * 2 ^ (d0_1_2 / 64 * 8192 + srow_2_2 * 128 + d0_1_2 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_1[(0) + 3]))
                            : "r"(vaddr_1_2));
                        float unpack_bf16_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) << 16);
                            (&unpack_bf16_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_1[_pair]) & 0xffff0000u);
                        }
                        float s_2[8];
                        s_2[0] = unpack_bf16_1[0] * fvb_1_2;
                        s_2[1] = unpack_bf16_1[1] * fvb_1_2;
                        s_2[2] = unpack_bf16_1[2] * fvb_1_2;
                        s_2[3] = unpack_bf16_1[3] * fvb_1_2;
                        s_2[4] = unpack_bf16_1[4] * fvb_1_2;
                        s_2[5] = unpack_bf16_1[5] * fvb_1_2;
                        s_2[6] = unpack_bf16_1[6] * fvb_1_2;
                        s_2[7] = unpack_bf16_1[7] * fvb_1_2;
                        unsigned int s_1_bf16_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_2[_lp*2 + 0], s_2[_lp*2+1 + 0]));
                            s_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_2), "r"(s_1_bf16_1[0]), "r"(s_1_bf16_1[1]), "r"(s_1_bf16_1[2]), "r"(s_1_bf16_1[3]) : "memory");
                        int qi_1_3 = t2 + 384;
                        int srow_2_3 = qi_1_3 >> 4;
                        int d0_1_3 = (qi_1_3 & 15) * 8;
                        float bs_r_1_3[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_3[0])) : "r"(beta_base + srow_2_3 * 4));
                        float fvb_1_3 = bs_r_1_3[0];
                        unsigned int vaddr_1_3 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_3 / 64 * 8192 + srow_2_3 * 128 + d0_1_3 % 64 * 2 ^ (d0_1_3 / 64 * 8192 + srow_2_3 * 128 + d0_1_3 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_2[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 3]))
                            : "r"(vaddr_1_3));
                        float unpack_bf16_2[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_2[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_2[_pair]) << 16);
                            (&unpack_bf16_2[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_2[_pair]) & 0xffff0000u);
                        }
                        float s_3[8];
                        s_3[0] = unpack_bf16_2[0] * fvb_1_3;
                        s_3[1] = unpack_bf16_2[1] * fvb_1_3;
                        s_3[2] = unpack_bf16_2[2] * fvb_1_3;
                        s_3[3] = unpack_bf16_2[3] * fvb_1_3;
                        s_3[4] = unpack_bf16_2[4] * fvb_1_3;
                        s_3[5] = unpack_bf16_2[5] * fvb_1_3;
                        s_3[6] = unpack_bf16_2[6] * fvb_1_3;
                        s_3[7] = unpack_bf16_2[7] * fvb_1_3;
                        unsigned int s_1_bf16_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_3[_lp*2 + 0], s_3[_lp*2+1 + 0]));
                            s_1_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_3), "r"(s_1_bf16_2[0]), "r"(s_1_bf16_2[1]), "r"(s_1_bf16_2[2]), "r"(s_1_bf16_2[3]) : "memory");
                        int qi_1_4 = t2 + 512;
                        int srow_2_4 = qi_1_4 >> 4;
                        int d0_1_4 = (qi_1_4 & 15) * 8;
                        float bs_r_1_4[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_4[0])) : "r"(beta_base + srow_2_4 * 4));
                        float fvb_1_4 = bs_r_1_4[0];
                        unsigned int vaddr_1_4 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_4 / 64 * 8192 + srow_2_4 * 128 + d0_1_4 % 64 * 2 ^ (d0_1_4 / 64 * 8192 + srow_2_4 * 128 + d0_1_4 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_3[(0) + 3]))
                            : "r"(vaddr_1_4));
                        float unpack_bf16_3[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_3[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_3[_pair]) << 16);
                            (&unpack_bf16_3[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_3[_pair]) & 0xffff0000u);
                        }
                        float s_4[8];
                        s_4[0] = unpack_bf16_3[0] * fvb_1_4;
                        s_4[1] = unpack_bf16_3[1] * fvb_1_4;
                        s_4[2] = unpack_bf16_3[2] * fvb_1_4;
                        s_4[3] = unpack_bf16_3[3] * fvb_1_4;
                        s_4[4] = unpack_bf16_3[4] * fvb_1_4;
                        s_4[5] = unpack_bf16_3[5] * fvb_1_4;
                        s_4[6] = unpack_bf16_3[6] * fvb_1_4;
                        s_4[7] = unpack_bf16_3[7] * fvb_1_4;
                        unsigned int s_1_bf16_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_4[_lp*2 + 0], s_4[_lp*2+1 + 0]));
                            s_1_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_4), "r"(s_1_bf16_3[0]), "r"(s_1_bf16_3[1]), "r"(s_1_bf16_3[2]), "r"(s_1_bf16_3[3]) : "memory");
                        int qi_1_5 = t2 + 640;
                        int srow_2_5 = qi_1_5 >> 4;
                        int d0_1_5 = (qi_1_5 & 15) * 8;
                        float bs_r_1_5[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_5[0])) : "r"(beta_base + srow_2_5 * 4));
                        float fvb_1_5 = bs_r_1_5[0];
                        unsigned int vaddr_1_5 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_5 / 64 * 8192 + srow_2_5 * 128 + d0_1_5 % 64 * 2 ^ (d0_1_5 / 64 * 8192 + srow_2_5 * 128 + d0_1_5 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_4[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_4[(0) + 3]))
                            : "r"(vaddr_1_5));
                        float unpack_bf16_4[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_4[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_4[_pair]) << 16);
                            (&unpack_bf16_4[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_4[_pair]) & 0xffff0000u);
                        }
                        float s_5[8];
                        s_5[0] = unpack_bf16_4[0] * fvb_1_5;
                        s_5[1] = unpack_bf16_4[1] * fvb_1_5;
                        s_5[2] = unpack_bf16_4[2] * fvb_1_5;
                        s_5[3] = unpack_bf16_4[3] * fvb_1_5;
                        s_5[4] = unpack_bf16_4[4] * fvb_1_5;
                        s_5[5] = unpack_bf16_4[5] * fvb_1_5;
                        s_5[6] = unpack_bf16_4[6] * fvb_1_5;
                        s_5[7] = unpack_bf16_4[7] * fvb_1_5;
                        unsigned int s_1_bf16_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_5[_lp*2 + 0], s_5[_lp*2+1 + 0]));
                            s_1_bf16_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_5), "r"(s_1_bf16_4[0]), "r"(s_1_bf16_4[1]), "r"(s_1_bf16_4[2]), "r"(s_1_bf16_4[3]) : "memory");
                        int qi_1_6 = t2 + 768;
                        int srow_2_6 = qi_1_6 >> 4;
                        int d0_1_6 = (qi_1_6 & 15) * 8;
                        float bs_r_1_6[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_6[0])) : "r"(beta_base + srow_2_6 * 4));
                        float fvb_1_6 = bs_r_1_6[0];
                        unsigned int vaddr_1_6 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_6 / 64 * 8192 + srow_2_6 * 128 + d0_1_6 % 64 * 2 ^ (d0_1_6 / 64 * 8192 + srow_2_6 * 128 + d0_1_6 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_5[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_5[(0) + 3]))
                            : "r"(vaddr_1_6));
                        float unpack_bf16_5[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_5[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_5[_pair]) << 16);
                            (&unpack_bf16_5[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_5[_pair]) & 0xffff0000u);
                        }
                        float s_6[8];
                        s_6[0] = unpack_bf16_5[0] * fvb_1_6;
                        s_6[1] = unpack_bf16_5[1] * fvb_1_6;
                        s_6[2] = unpack_bf16_5[2] * fvb_1_6;
                        s_6[3] = unpack_bf16_5[3] * fvb_1_6;
                        s_6[4] = unpack_bf16_5[4] * fvb_1_6;
                        s_6[5] = unpack_bf16_5[5] * fvb_1_6;
                        s_6[6] = unpack_bf16_5[6] * fvb_1_6;
                        s_6[7] = unpack_bf16_5[7] * fvb_1_6;
                        unsigned int s_1_bf16_5[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_6[_lp*2 + 0], s_6[_lp*2+1 + 0]));
                            s_1_bf16_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_6), "r"(s_1_bf16_5[0]), "r"(s_1_bf16_5[1]), "r"(s_1_bf16_5[2]), "r"(s_1_bf16_5[3]) : "memory");
                        int qi_1_7 = t2 + 896;
                        int srow_2_7 = qi_1_7 >> 4;
                        int d0_1_7 = (qi_1_7 & 15) * 8;
                        float bs_r_1_7[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_7[0])) : "r"(beta_base + srow_2_7 * 4));
                        float fvb_1_7 = bs_r_1_7[0];
                        unsigned int vaddr_1_7 = (v_t_addr + vst_c * 16384 + (unsigned int)(d0_1_7 / 64 * 8192 + srow_2_7 * 128 + d0_1_7 % 64 * 2 ^ (d0_1_7 / 64 * 8192 + srow_2_7 * 128 + d0_1_7 % 64 * 2 >> 7 & 7) << 4));
                        unsigned int w_6[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_6[(0) + 3]))
                            : "r"(vaddr_1_7));
                        float unpack_bf16_6[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_6[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) << 16);
                            (&unpack_bf16_6[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_6[_pair]) & 0xffff0000u);
                        }
                        float s_7[8];
                        s_7[0] = unpack_bf16_6[0] * fvb_1_7;
                        s_7[1] = unpack_bf16_6[1] * fvb_1_7;
                        s_7[2] = unpack_bf16_6[2] * fvb_1_7;
                        s_7[3] = unpack_bf16_6[3] * fvb_1_7;
                        s_7[4] = unpack_bf16_6[4] * fvb_1_7;
                        s_7[5] = unpack_bf16_6[5] * fvb_1_7;
                        s_7[6] = unpack_bf16_6[6] * fvb_1_7;
                        s_7[7] = unpack_bf16_6[7] * fvb_1_7;
                        unsigned int s_1_bf16_6[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_7[_lp*2 + 0], s_7[_lp*2+1 + 0]));
                            s_1_bf16_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(vaddr_1_7), "r"(s_1_bf16_6[0]), "r"(s_1_bf16_6[1]), "r"(s_1_bf16_6[2]), "r"(s_1_bf16_6[3]) : "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                    }
                    if (j > 0) {
                        mbarrier_wait(wu_full_addr, _phase_wu_full_0);
                        _phase_wu_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (group > j) {
                        int arow = tid_0 >> 1;
                        long long akk_base = ((long long)(irow0 + arow) * (long long)num_heads + (long long)hv) * 64ll;
                        int sa0_1_0 = (tid_0 & 1) * 4 * 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"((akk_t_addr + (unsigned int)(sa0_1_0 / 64 * 8192 + arow * 128 + sa0_1_0 % 64 * 2 ^ (sa0_1_0 / 64 * 8192 + arow * 128 + sa0_1_0 % 64 * 2 >> 7 & 7) << 4))), "l"(akk + (akk_base + (long long)sa0_1_0)));
                        int sa0_1_1 = ((tid_0 & 1) * 4 + 1) * 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"((akk_t_addr + (unsigned int)(sa0_1_1 / 64 * 8192 + arow * 128 + sa0_1_1 % 64 * 2 ^ (sa0_1_1 / 64 * 8192 + arow * 128 + sa0_1_1 % 64 * 2 >> 7 & 7) << 4))), "l"(akk + (akk_base + (long long)sa0_1_1)));
                        int sa0_1_2 = ((tid_0 & 1) * 4 + 2) * 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"((akk_t_addr + (unsigned int)(sa0_1_2 / 64 * 8192 + arow * 128 + sa0_1_2 % 64 * 2 ^ (sa0_1_2 / 64 * 8192 + arow * 128 + sa0_1_2 % 64 * 2 >> 7 & 7) << 4))), "l"(akk + (akk_base + (long long)sa0_1_2)));
                        int sa0_1_3 = ((tid_0 & 1) * 4 + 3) * 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                            :: "r"((akk_t_addr + (unsigned int)(sa0_1_3 / 64 * 8192 + arow * 128 + sa0_1_3 % 64 * 2 ^ (sa0_1_3 / 64 * 8192 + arow * 128 + sa0_1_3 % 64 * 2 >> 7 & 7) << 4))), "l"(akk + (akk_base + (long long)sa0_1_3)));
                        asm volatile("cp.async.commit_group;");
                        int qi_1_8 = tid_0;
                        int srow_2_8 = qi_1_8 >> 4;
                        int d0_1_8 = (qi_1_8 & 15) * 8;
                        float bs_r_1_8[1];
                        float es_r_1_0[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_8[0])) : "r"(beta_base + srow_2_8 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_0[0])) : "r"(eg_base + srow_2_8 * 4));
                        float fkb_1_0 = bs_r_1_8[0] * es_r_1_0[0];
                        unsigned int w_7[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_7[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 ^ (d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_7[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_7[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_7[_pair]) << 16);
                            (&unpack_bf16_7[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_7[_pair]) & 0xffff0000u);
                        }
                        float s_1[8];
                        s_1[0] = unpack_bf16_7[0] * fkb_1_0;
                        s_1[1] = unpack_bf16_7[1] * fkb_1_0;
                        s_1[2] = unpack_bf16_7[2] * fkb_1_0;
                        s_1[3] = unpack_bf16_7[3] * fkb_1_0;
                        s_1[4] = unpack_bf16_7[4] * fkb_1_0;
                        s_1[5] = unpack_bf16_7[5] * fkb_1_0;
                        s_1[6] = unpack_bf16_7[6] * fkb_1_0;
                        s_1[7] = unpack_bf16_7[7] * fkb_1_0;
                        unsigned int s_1_bf16_7[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_1[_lp*2 + 0], s_1[_lp*2+1 + 0]));
                            s_1_bf16_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 ^ (d0_1_8 / 64 * 8192 + srow_2_8 * 128 + d0_1_8 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_7[0]), "r"(s_1_bf16_7[1]), "r"(s_1_bf16_7[2]), "r"(s_1_bf16_7[3]) : "memory");
                        int qi_1_9 = tid_0 + 128;
                        int srow_2_9 = qi_1_9 >> 4;
                        int d0_1_9 = (qi_1_9 & 15) * 8;
                        float bs_r_1_9[1];
                        float es_r_1_1[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_9[0])) : "r"(beta_base + srow_2_9 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_1[0])) : "r"(eg_base + srow_2_9 * 4));
                        float fkb_1_1 = bs_r_1_9[0] * es_r_1_1[0];
                        unsigned int w_7_1[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_7_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_7_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_7_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_7_1[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 ^ (d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_7_1[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_7_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_7_1[_pair]) << 16);
                            (&unpack_bf16_7_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_7_1[_pair]) & 0xffff0000u);
                        }
                        float s_8[8];
                        s_8[0] = unpack_bf16_7_1[0] * fkb_1_1;
                        s_8[1] = unpack_bf16_7_1[1] * fkb_1_1;
                        s_8[2] = unpack_bf16_7_1[2] * fkb_1_1;
                        s_8[3] = unpack_bf16_7_1[3] * fkb_1_1;
                        s_8[4] = unpack_bf16_7_1[4] * fkb_1_1;
                        s_8[5] = unpack_bf16_7_1[5] * fkb_1_1;
                        s_8[6] = unpack_bf16_7_1[6] * fkb_1_1;
                        s_8[7] = unpack_bf16_7_1[7] * fkb_1_1;
                        unsigned int s_1_bf16_7_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_8[_lp*2 + 0], s_8[_lp*2+1 + 0]));
                            s_1_bf16_7_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 ^ (d0_1_9 / 64 * 8192 + srow_2_9 * 128 + d0_1_9 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_7_1[0]), "r"(s_1_bf16_7_1[1]), "r"(s_1_bf16_7_1[2]), "r"(s_1_bf16_7_1[3]) : "memory");
                        int qi_1_10 = tid_0 + 256;
                        int srow_2_10 = qi_1_10 >> 4;
                        int d0_1_10 = (qi_1_10 & 15) * 8;
                        float bs_r_1_10[1];
                        float es_r_1_2[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_10[0])) : "r"(beta_base + srow_2_10 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_2[0])) : "r"(eg_base + srow_2_10 * 4));
                        float fkb_1_2 = bs_r_1_10[0] * es_r_1_2[0];
                        unsigned int w_8[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_8[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 ^ (d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_8[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_8[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) << 16);
                            (&unpack_bf16_8[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_8[_pair]) & 0xffff0000u);
                        }
                        float s_9[8];
                        s_9[0] = unpack_bf16_8[0] * fkb_1_2;
                        s_9[1] = unpack_bf16_8[1] * fkb_1_2;
                        s_9[2] = unpack_bf16_8[2] * fkb_1_2;
                        s_9[3] = unpack_bf16_8[3] * fkb_1_2;
                        s_9[4] = unpack_bf16_8[4] * fkb_1_2;
                        s_9[5] = unpack_bf16_8[5] * fkb_1_2;
                        s_9[6] = unpack_bf16_8[6] * fkb_1_2;
                        s_9[7] = unpack_bf16_8[7] * fkb_1_2;
                        unsigned int s_1_bf16_8[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_9[_lp*2 + 0], s_9[_lp*2+1 + 0]));
                            s_1_bf16_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 ^ (d0_1_10 / 64 * 8192 + srow_2_10 * 128 + d0_1_10 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_8[0]), "r"(s_1_bf16_8[1]), "r"(s_1_bf16_8[2]), "r"(s_1_bf16_8[3]) : "memory");
                        int qi_1_11 = tid_0 + 384;
                        int srow_2_11 = qi_1_11 >> 4;
                        int d0_1_11 = (qi_1_11 & 15) * 8;
                        float bs_r_1_11[1];
                        float es_r_1_3[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_11[0])) : "r"(beta_base + srow_2_11 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_3[0])) : "r"(eg_base + srow_2_11 * 4));
                        float fkb_1_3 = bs_r_1_11[0] * es_r_1_3[0];
                        unsigned int w_9[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_9[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 ^ (d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_9[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_9[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_9[_pair]) << 16);
                            (&unpack_bf16_9[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_9[_pair]) & 0xffff0000u);
                        }
                        float s_10[8];
                        s_10[0] = unpack_bf16_9[0] * fkb_1_3;
                        s_10[1] = unpack_bf16_9[1] * fkb_1_3;
                        s_10[2] = unpack_bf16_9[2] * fkb_1_3;
                        s_10[3] = unpack_bf16_9[3] * fkb_1_3;
                        s_10[4] = unpack_bf16_9[4] * fkb_1_3;
                        s_10[5] = unpack_bf16_9[5] * fkb_1_3;
                        s_10[6] = unpack_bf16_9[6] * fkb_1_3;
                        s_10[7] = unpack_bf16_9[7] * fkb_1_3;
                        unsigned int s_1_bf16_9[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_10[_lp*2 + 0], s_10[_lp*2+1 + 0]));
                            s_1_bf16_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 ^ (d0_1_11 / 64 * 8192 + srow_2_11 * 128 + d0_1_11 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_9[0]), "r"(s_1_bf16_9[1]), "r"(s_1_bf16_9[2]), "r"(s_1_bf16_9[3]) : "memory");
                        int qi_1_12 = tid_0 + 512;
                        int srow_2_12 = qi_1_12 >> 4;
                        int d0_1_12 = (qi_1_12 & 15) * 8;
                        float bs_r_1_12[1];
                        float es_r_1_4[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_12[0])) : "r"(beta_base + srow_2_12 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_4[0])) : "r"(eg_base + srow_2_12 * 4));
                        float fkb_1_4 = bs_r_1_12[0] * es_r_1_4[0];
                        unsigned int w_10[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_10[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 ^ (d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_10[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_10[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_10[_pair]) << 16);
                            (&unpack_bf16_10[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_10[_pair]) & 0xffff0000u);
                        }
                        float s_11[8];
                        s_11[0] = unpack_bf16_10[0] * fkb_1_4;
                        s_11[1] = unpack_bf16_10[1] * fkb_1_4;
                        s_11[2] = unpack_bf16_10[2] * fkb_1_4;
                        s_11[3] = unpack_bf16_10[3] * fkb_1_4;
                        s_11[4] = unpack_bf16_10[4] * fkb_1_4;
                        s_11[5] = unpack_bf16_10[5] * fkb_1_4;
                        s_11[6] = unpack_bf16_10[6] * fkb_1_4;
                        s_11[7] = unpack_bf16_10[7] * fkb_1_4;
                        unsigned int s_1_bf16_10[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_11[_lp*2 + 0], s_11[_lp*2+1 + 0]));
                            s_1_bf16_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 ^ (d0_1_12 / 64 * 8192 + srow_2_12 * 128 + d0_1_12 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_10[0]), "r"(s_1_bf16_10[1]), "r"(s_1_bf16_10[2]), "r"(s_1_bf16_10[3]) : "memory");
                        int qi_1_13 = tid_0 + 640;
                        int srow_2_13 = qi_1_13 >> 4;
                        int d0_1_13 = (qi_1_13 & 15) * 8;
                        float bs_r_1_13[1];
                        float es_r_1_5[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_13[0])) : "r"(beta_base + srow_2_13 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_5[0])) : "r"(eg_base + srow_2_13 * 4));
                        float fkb_1_5 = bs_r_1_13[0] * es_r_1_5[0];
                        unsigned int w_11[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_11[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 ^ (d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_11[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_11[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_11[_pair]) << 16);
                            (&unpack_bf16_11[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_11[_pair]) & 0xffff0000u);
                        }
                        float s_12[8];
                        s_12[0] = unpack_bf16_11[0] * fkb_1_5;
                        s_12[1] = unpack_bf16_11[1] * fkb_1_5;
                        s_12[2] = unpack_bf16_11[2] * fkb_1_5;
                        s_12[3] = unpack_bf16_11[3] * fkb_1_5;
                        s_12[4] = unpack_bf16_11[4] * fkb_1_5;
                        s_12[5] = unpack_bf16_11[5] * fkb_1_5;
                        s_12[6] = unpack_bf16_11[6] * fkb_1_5;
                        s_12[7] = unpack_bf16_11[7] * fkb_1_5;
                        unsigned int s_1_bf16_11[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_12[_lp*2 + 0], s_12[_lp*2+1 + 0]));
                            s_1_bf16_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 ^ (d0_1_13 / 64 * 8192 + srow_2_13 * 128 + d0_1_13 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_11[0]), "r"(s_1_bf16_11[1]), "r"(s_1_bf16_11[2]), "r"(s_1_bf16_11[3]) : "memory");
                        int qi_1_14 = tid_0 + 768;
                        int srow_2_14 = qi_1_14 >> 4;
                        int d0_1_14 = (qi_1_14 & 15) * 8;
                        float bs_r_1_14[1];
                        float es_r_1_6[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_14[0])) : "r"(beta_base + srow_2_14 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_6[0])) : "r"(eg_base + srow_2_14 * 4));
                        float fkb_1_6 = bs_r_1_14[0] * es_r_1_6[0];
                        unsigned int w_12[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_12[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 ^ (d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_12[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_12[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_12[_pair]) << 16);
                            (&unpack_bf16_12[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_12[_pair]) & 0xffff0000u);
                        }
                        float s_13[8];
                        s_13[0] = unpack_bf16_12[0] * fkb_1_6;
                        s_13[1] = unpack_bf16_12[1] * fkb_1_6;
                        s_13[2] = unpack_bf16_12[2] * fkb_1_6;
                        s_13[3] = unpack_bf16_12[3] * fkb_1_6;
                        s_13[4] = unpack_bf16_12[4] * fkb_1_6;
                        s_13[5] = unpack_bf16_12[5] * fkb_1_6;
                        s_13[6] = unpack_bf16_12[6] * fkb_1_6;
                        s_13[7] = unpack_bf16_12[7] * fkb_1_6;
                        unsigned int s_1_bf16_12[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_13[_lp*2 + 0], s_13[_lp*2+1 + 0]));
                            s_1_bf16_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 ^ (d0_1_14 / 64 * 8192 + srow_2_14 * 128 + d0_1_14 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_12[0]), "r"(s_1_bf16_12[1]), "r"(s_1_bf16_12[2]), "r"(s_1_bf16_12[3]) : "memory");
                        int qi_1_15 = tid_0 + 896;
                        int srow_2_15 = qi_1_15 >> 4;
                        int d0_1_15 = (qi_1_15 & 15) * 8;
                        float bs_r_1_15[1];
                        float es_r_1_7[1];
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&bs_r_1_15[0])) : "r"(beta_base + srow_2_15 * 4));
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&es_r_1_7[0])) : "r"(eg_base + srow_2_15 * 4));
                        float fkb_1_7 = bs_r_1_15[0] * es_r_1_7[0];
                        unsigned int w_13[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w_13[(0) + 3]))
                            : "r"((k_k_addr + (unsigned int)(d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 ^ (d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 >> 7 & 7) << 4))));
                        float unpack_bf16_13[8];
                        #pragma unroll
                        for (int _pair = 0; _pair < 4; _pair++) {
                            (&unpack_bf16_13[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(w_13[_pair]) << 16);
                            (&unpack_bf16_13[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(w_13[_pair]) & 0xffff0000u);
                        }
                        float s_14[8];
                        s_14[0] = unpack_bf16_13[0] * fkb_1_7;
                        s_14[1] = unpack_bf16_13[1] * fkb_1_7;
                        s_14[2] = unpack_bf16_13[2] * fkb_1_7;
                        s_14[3] = unpack_bf16_13[3] * fkb_1_7;
                        s_14[4] = unpack_bf16_13[4] * fkb_1_7;
                        s_14[5] = unpack_bf16_13[5] * fkb_1_7;
                        s_14[6] = unpack_bf16_13[6] * fkb_1_7;
                        s_14[7] = unpack_bf16_13[7] * fkb_1_7;
                        unsigned int s_1_bf16_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(s_14[_lp*2 + 0], s_14[_lp*2+1 + 0]));
                            s_1_bf16_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((q_k_addr + (unsigned int)(d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 ^ (d0_1_15 / 64 * 8192 + srow_2_15 * 128 + d0_1_15 % 64 * 2 >> 7 & 7) << 4))), "r"(s_1_bf16_13[0]), "r"(s_1_bf16_13[1]), "r"(s_1_bf16_13[2]), "r"(s_1_bf16_13[3]) : "memory");
                    }
                    if (j > 0) {
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int wrow = l_base + t_row * 272;
                        int urow = a_base + t_row * 272;
                        int col_1_0 = half * 16;
                        float _tmem_load_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                            : "r"(taddr + (unsigned int)row_base));
                        float _tmem_load_8_0[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_0[15]))
                            : "r"(taddr + (unsigned int)row_base + 128));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_0[16];
                        float uo_1_0[16];
                        wo_2_0[0] = ((row_valid != 0) ? _tmem_load_0[0] : 0.0f);
                        uo_1_0[0] = ((row_valid != 0) ? _tmem_load_8_0[0] : 0.0f);
                        wo_2_0[1] = ((row_valid != 0) ? _tmem_load_0[1] : 0.0f);
                        uo_1_0[1] = ((row_valid != 0) ? _tmem_load_8_0[1] : 0.0f);
                        wo_2_0[2] = ((row_valid != 0) ? _tmem_load_0[2] : 0.0f);
                        uo_1_0[2] = ((row_valid != 0) ? _tmem_load_8_0[2] : 0.0f);
                        wo_2_0[3] = ((row_valid != 0) ? _tmem_load_0[3] : 0.0f);
                        uo_1_0[3] = ((row_valid != 0) ? _tmem_load_8_0[3] : 0.0f);
                        wo_2_0[4] = ((row_valid != 0) ? _tmem_load_0[4] : 0.0f);
                        uo_1_0[4] = ((row_valid != 0) ? _tmem_load_8_0[4] : 0.0f);
                        wo_2_0[5] = ((row_valid != 0) ? _tmem_load_0[5] : 0.0f);
                        uo_1_0[5] = ((row_valid != 0) ? _tmem_load_8_0[5] : 0.0f);
                        wo_2_0[6] = ((row_valid != 0) ? _tmem_load_0[6] : 0.0f);
                        uo_1_0[6] = ((row_valid != 0) ? _tmem_load_8_0[6] : 0.0f);
                        wo_2_0[7] = ((row_valid != 0) ? _tmem_load_0[7] : 0.0f);
                        uo_1_0[7] = ((row_valid != 0) ? _tmem_load_8_0[7] : 0.0f);
                        wo_2_0[8] = ((row_valid != 0) ? _tmem_load_0[8] : 0.0f);
                        uo_1_0[8] = ((row_valid != 0) ? _tmem_load_8_0[8] : 0.0f);
                        wo_2_0[9] = ((row_valid != 0) ? _tmem_load_0[9] : 0.0f);
                        uo_1_0[9] = ((row_valid != 0) ? _tmem_load_8_0[9] : 0.0f);
                        wo_2_0[10] = ((row_valid != 0) ? _tmem_load_0[10] : 0.0f);
                        uo_1_0[10] = ((row_valid != 0) ? _tmem_load_8_0[10] : 0.0f);
                        wo_2_0[11] = ((row_valid != 0) ? _tmem_load_0[11] : 0.0f);
                        uo_1_0[11] = ((row_valid != 0) ? _tmem_load_8_0[11] : 0.0f);
                        wo_2_0[12] = ((row_valid != 0) ? _tmem_load_0[12] : 0.0f);
                        uo_1_0[12] = ((row_valid != 0) ? _tmem_load_8_0[12] : 0.0f);
                        wo_2_0[13] = ((row_valid != 0) ? _tmem_load_0[13] : 0.0f);
                        uo_1_0[13] = ((row_valid != 0) ? _tmem_load_8_0[13] : 0.0f);
                        wo_2_0[14] = ((row_valid != 0) ? _tmem_load_0[14] : 0.0f);
                        uo_1_0[14] = ((row_valid != 0) ? _tmem_load_8_0[14] : 0.0f);
                        wo_2_0[15] = ((row_valid != 0) ? _tmem_load_0[15] : 0.0f);
                        uo_1_0[15] = ((row_valid != 0) ? _tmem_load_8_0[15] : 0.0f);
                        unsigned int pw_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 0], wo_2_0[_lp*2+1 + 0]));
                            pw_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 2], wo_2_0[_lp*2+1 + 2]));
                            pw_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 4], wo_2_0[_lp*2+1 + 4]));
                            pw_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 6], wo_2_0[_lp*2+1 + 6]));
                            pw_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 8], wo_2_0[_lp*2+1 + 8]));
                            pw_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 10], wo_2_0[_lp*2+1 + 10]));
                            pw_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 12], wo_2_0[_lp*2+1 + 12]));
                            pw_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_0[_lp*2 + 14], wo_2_0[_lp*2+1 + 14]));
                            pw_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_0 * 2), "r"(pw_0[0]), "r"(pw_0[1]), "r"(pw_0[2]), "r"(pw_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_0 * 2 + 16), "r"(pw_0[4]), "r"(pw_0[5]), "r"(pw_0[6]), "r"(pw_0[7]) : "memory");
                        unsigned int pu_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 0], uo_1_0[_lp*2+1 + 0]));
                            pu_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 2], uo_1_0[_lp*2+1 + 2]));
                            pu_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 4], uo_1_0[_lp*2+1 + 4]));
                            pu_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 6], uo_1_0[_lp*2+1 + 6]));
                            pu_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 8], uo_1_0[_lp*2+1 + 8]));
                            pu_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 10], uo_1_0[_lp*2+1 + 10]));
                            pu_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 12], uo_1_0[_lp*2+1 + 12]));
                            pu_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_0[_lp*2 + 14], uo_1_0[_lp*2+1 + 14]));
                            pu_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_0 * 2), "r"(pu_0[0]), "r"(pu_0[1]), "r"(pu_0[2]), "r"(pu_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_0 * 2 + 16), "r"(pu_0[4]), "r"(pu_0[5]), "r"(pu_0[6]), "r"(pu_0[7]) : "memory");
                        int col_1_1 = 32 + half * 16;
                        float _tmem_load_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                            : "r"(taddr + (unsigned int)row_base + 32));
                        float _tmem_load_8_1[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_1[15]))
                            : "r"(taddr + (unsigned int)row_base + 160));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_1[16];
                        float uo_1_1[16];
                        wo_2_1[0] = ((row_valid != 0) ? _tmem_load_1[0] : 0.0f);
                        uo_1_1[0] = ((row_valid != 0) ? _tmem_load_8_1[0] : 0.0f);
                        wo_2_1[1] = ((row_valid != 0) ? _tmem_load_1[1] : 0.0f);
                        uo_1_1[1] = ((row_valid != 0) ? _tmem_load_8_1[1] : 0.0f);
                        wo_2_1[2] = ((row_valid != 0) ? _tmem_load_1[2] : 0.0f);
                        uo_1_1[2] = ((row_valid != 0) ? _tmem_load_8_1[2] : 0.0f);
                        wo_2_1[3] = ((row_valid != 0) ? _tmem_load_1[3] : 0.0f);
                        uo_1_1[3] = ((row_valid != 0) ? _tmem_load_8_1[3] : 0.0f);
                        wo_2_1[4] = ((row_valid != 0) ? _tmem_load_1[4] : 0.0f);
                        uo_1_1[4] = ((row_valid != 0) ? _tmem_load_8_1[4] : 0.0f);
                        wo_2_1[5] = ((row_valid != 0) ? _tmem_load_1[5] : 0.0f);
                        uo_1_1[5] = ((row_valid != 0) ? _tmem_load_8_1[5] : 0.0f);
                        wo_2_1[6] = ((row_valid != 0) ? _tmem_load_1[6] : 0.0f);
                        uo_1_1[6] = ((row_valid != 0) ? _tmem_load_8_1[6] : 0.0f);
                        wo_2_1[7] = ((row_valid != 0) ? _tmem_load_1[7] : 0.0f);
                        uo_1_1[7] = ((row_valid != 0) ? _tmem_load_8_1[7] : 0.0f);
                        wo_2_1[8] = ((row_valid != 0) ? _tmem_load_1[8] : 0.0f);
                        uo_1_1[8] = ((row_valid != 0) ? _tmem_load_8_1[8] : 0.0f);
                        wo_2_1[9] = ((row_valid != 0) ? _tmem_load_1[9] : 0.0f);
                        uo_1_1[9] = ((row_valid != 0) ? _tmem_load_8_1[9] : 0.0f);
                        wo_2_1[10] = ((row_valid != 0) ? _tmem_load_1[10] : 0.0f);
                        uo_1_1[10] = ((row_valid != 0) ? _tmem_load_8_1[10] : 0.0f);
                        wo_2_1[11] = ((row_valid != 0) ? _tmem_load_1[11] : 0.0f);
                        uo_1_1[11] = ((row_valid != 0) ? _tmem_load_8_1[11] : 0.0f);
                        wo_2_1[12] = ((row_valid != 0) ? _tmem_load_1[12] : 0.0f);
                        uo_1_1[12] = ((row_valid != 0) ? _tmem_load_8_1[12] : 0.0f);
                        wo_2_1[13] = ((row_valid != 0) ? _tmem_load_1[13] : 0.0f);
                        uo_1_1[13] = ((row_valid != 0) ? _tmem_load_8_1[13] : 0.0f);
                        wo_2_1[14] = ((row_valid != 0) ? _tmem_load_1[14] : 0.0f);
                        uo_1_1[14] = ((row_valid != 0) ? _tmem_load_8_1[14] : 0.0f);
                        wo_2_1[15] = ((row_valid != 0) ? _tmem_load_1[15] : 0.0f);
                        uo_1_1[15] = ((row_valid != 0) ? _tmem_load_8_1[15] : 0.0f);
                        unsigned int pw_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 0], wo_2_1[_lp*2+1 + 0]));
                            pw_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 2], wo_2_1[_lp*2+1 + 2]));
                            pw_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 4], wo_2_1[_lp*2+1 + 4]));
                            pw_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 6], wo_2_1[_lp*2+1 + 6]));
                            pw_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 8], wo_2_1[_lp*2+1 + 8]));
                            pw_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 10], wo_2_1[_lp*2+1 + 10]));
                            pw_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 12], wo_2_1[_lp*2+1 + 12]));
                            pw_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_1[_lp*2 + 14], wo_2_1[_lp*2+1 + 14]));
                            pw_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_1 * 2), "r"(pw_1[0]), "r"(pw_1[1]), "r"(pw_1[2]), "r"(pw_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_1 * 2 + 16), "r"(pw_1[4]), "r"(pw_1[5]), "r"(pw_1[6]), "r"(pw_1[7]) : "memory");
                        unsigned int pu_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 0], uo_1_1[_lp*2+1 + 0]));
                            pu_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 2], uo_1_1[_lp*2+1 + 2]));
                            pu_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 4], uo_1_1[_lp*2+1 + 4]));
                            pu_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 6], uo_1_1[_lp*2+1 + 6]));
                            pu_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 8], uo_1_1[_lp*2+1 + 8]));
                            pu_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 10], uo_1_1[_lp*2+1 + 10]));
                            pu_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 12], uo_1_1[_lp*2+1 + 12]));
                            pu_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_1[_lp*2 + 14], uo_1_1[_lp*2+1 + 14]));
                            pu_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_1 * 2), "r"(pu_1[0]), "r"(pu_1[1]), "r"(pu_1[2]), "r"(pu_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_1 * 2 + 16), "r"(pu_1[4]), "r"(pu_1[5]), "r"(pu_1[6]), "r"(pu_1[7]) : "memory");
                        int col_1_2 = 64 + half * 16;
                        float _tmem_load_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                            : "r"(taddr + (unsigned int)row_base + 64));
                        float _tmem_load_8_2[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_2[15]))
                            : "r"(taddr + (unsigned int)row_base + 192));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_2[16];
                        float uo_1_2[16];
                        wo_2_2[0] = ((row_valid != 0) ? _tmem_load_2[0] : 0.0f);
                        uo_1_2[0] = ((row_valid != 0) ? _tmem_load_8_2[0] : 0.0f);
                        wo_2_2[1] = ((row_valid != 0) ? _tmem_load_2[1] : 0.0f);
                        uo_1_2[1] = ((row_valid != 0) ? _tmem_load_8_2[1] : 0.0f);
                        wo_2_2[2] = ((row_valid != 0) ? _tmem_load_2[2] : 0.0f);
                        uo_1_2[2] = ((row_valid != 0) ? _tmem_load_8_2[2] : 0.0f);
                        wo_2_2[3] = ((row_valid != 0) ? _tmem_load_2[3] : 0.0f);
                        uo_1_2[3] = ((row_valid != 0) ? _tmem_load_8_2[3] : 0.0f);
                        wo_2_2[4] = ((row_valid != 0) ? _tmem_load_2[4] : 0.0f);
                        uo_1_2[4] = ((row_valid != 0) ? _tmem_load_8_2[4] : 0.0f);
                        wo_2_2[5] = ((row_valid != 0) ? _tmem_load_2[5] : 0.0f);
                        uo_1_2[5] = ((row_valid != 0) ? _tmem_load_8_2[5] : 0.0f);
                        wo_2_2[6] = ((row_valid != 0) ? _tmem_load_2[6] : 0.0f);
                        uo_1_2[6] = ((row_valid != 0) ? _tmem_load_8_2[6] : 0.0f);
                        wo_2_2[7] = ((row_valid != 0) ? _tmem_load_2[7] : 0.0f);
                        uo_1_2[7] = ((row_valid != 0) ? _tmem_load_8_2[7] : 0.0f);
                        wo_2_2[8] = ((row_valid != 0) ? _tmem_load_2[8] : 0.0f);
                        uo_1_2[8] = ((row_valid != 0) ? _tmem_load_8_2[8] : 0.0f);
                        wo_2_2[9] = ((row_valid != 0) ? _tmem_load_2[9] : 0.0f);
                        uo_1_2[9] = ((row_valid != 0) ? _tmem_load_8_2[9] : 0.0f);
                        wo_2_2[10] = ((row_valid != 0) ? _tmem_load_2[10] : 0.0f);
                        uo_1_2[10] = ((row_valid != 0) ? _tmem_load_8_2[10] : 0.0f);
                        wo_2_2[11] = ((row_valid != 0) ? _tmem_load_2[11] : 0.0f);
                        uo_1_2[11] = ((row_valid != 0) ? _tmem_load_8_2[11] : 0.0f);
                        wo_2_2[12] = ((row_valid != 0) ? _tmem_load_2[12] : 0.0f);
                        uo_1_2[12] = ((row_valid != 0) ? _tmem_load_8_2[12] : 0.0f);
                        wo_2_2[13] = ((row_valid != 0) ? _tmem_load_2[13] : 0.0f);
                        uo_1_2[13] = ((row_valid != 0) ? _tmem_load_8_2[13] : 0.0f);
                        wo_2_2[14] = ((row_valid != 0) ? _tmem_load_2[14] : 0.0f);
                        uo_1_2[14] = ((row_valid != 0) ? _tmem_load_8_2[14] : 0.0f);
                        wo_2_2[15] = ((row_valid != 0) ? _tmem_load_2[15] : 0.0f);
                        uo_1_2[15] = ((row_valid != 0) ? _tmem_load_8_2[15] : 0.0f);
                        unsigned int pw_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 0], wo_2_2[_lp*2+1 + 0]));
                            pw_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 2], wo_2_2[_lp*2+1 + 2]));
                            pw_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 4], wo_2_2[_lp*2+1 + 4]));
                            pw_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 6], wo_2_2[_lp*2+1 + 6]));
                            pw_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 8], wo_2_2[_lp*2+1 + 8]));
                            pw_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 10], wo_2_2[_lp*2+1 + 10]));
                            pw_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 12], wo_2_2[_lp*2+1 + 12]));
                            pw_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_2[_lp*2 + 14], wo_2_2[_lp*2+1 + 14]));
                            pw_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_2 * 2), "r"(pw_2[0]), "r"(pw_2[1]), "r"(pw_2[2]), "r"(pw_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_2 * 2 + 16), "r"(pw_2[4]), "r"(pw_2[5]), "r"(pw_2[6]), "r"(pw_2[7]) : "memory");
                        unsigned int pu_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 0], uo_1_2[_lp*2+1 + 0]));
                            pu_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 2], uo_1_2[_lp*2+1 + 2]));
                            pu_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 4], uo_1_2[_lp*2+1 + 4]));
                            pu_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 6], uo_1_2[_lp*2+1 + 6]));
                            pu_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 8], uo_1_2[_lp*2+1 + 8]));
                            pu_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 10], uo_1_2[_lp*2+1 + 10]));
                            pu_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 12], uo_1_2[_lp*2+1 + 12]));
                            pu_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_2[_lp*2 + 14], uo_1_2[_lp*2+1 + 14]));
                            pu_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_2 * 2), "r"(pu_2[0]), "r"(pu_2[1]), "r"(pu_2[2]), "r"(pu_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_2 * 2 + 16), "r"(pu_2[4]), "r"(pu_2[5]), "r"(pu_2[6]), "r"(pu_2[7]) : "memory");
                        int col_1_3 = 96 + half * 16;
                        float _tmem_load_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                            : "r"(taddr + (unsigned int)row_base + 96));
                        float _tmem_load_8_3[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8_3[15]))
                            : "r"(taddr + (unsigned int)row_base + 224));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wo_2_3[16];
                        float uo_1_3[16];
                        wo_2_3[0] = ((row_valid != 0) ? _tmem_load_3[0] : 0.0f);
                        uo_1_3[0] = ((row_valid != 0) ? _tmem_load_8_3[0] : 0.0f);
                        wo_2_3[1] = ((row_valid != 0) ? _tmem_load_3[1] : 0.0f);
                        uo_1_3[1] = ((row_valid != 0) ? _tmem_load_8_3[1] : 0.0f);
                        wo_2_3[2] = ((row_valid != 0) ? _tmem_load_3[2] : 0.0f);
                        uo_1_3[2] = ((row_valid != 0) ? _tmem_load_8_3[2] : 0.0f);
                        wo_2_3[3] = ((row_valid != 0) ? _tmem_load_3[3] : 0.0f);
                        uo_1_3[3] = ((row_valid != 0) ? _tmem_load_8_3[3] : 0.0f);
                        wo_2_3[4] = ((row_valid != 0) ? _tmem_load_3[4] : 0.0f);
                        uo_1_3[4] = ((row_valid != 0) ? _tmem_load_8_3[4] : 0.0f);
                        wo_2_3[5] = ((row_valid != 0) ? _tmem_load_3[5] : 0.0f);
                        uo_1_3[5] = ((row_valid != 0) ? _tmem_load_8_3[5] : 0.0f);
                        wo_2_3[6] = ((row_valid != 0) ? _tmem_load_3[6] : 0.0f);
                        uo_1_3[6] = ((row_valid != 0) ? _tmem_load_8_3[6] : 0.0f);
                        wo_2_3[7] = ((row_valid != 0) ? _tmem_load_3[7] : 0.0f);
                        uo_1_3[7] = ((row_valid != 0) ? _tmem_load_8_3[7] : 0.0f);
                        wo_2_3[8] = ((row_valid != 0) ? _tmem_load_3[8] : 0.0f);
                        uo_1_3[8] = ((row_valid != 0) ? _tmem_load_8_3[8] : 0.0f);
                        wo_2_3[9] = ((row_valid != 0) ? _tmem_load_3[9] : 0.0f);
                        uo_1_3[9] = ((row_valid != 0) ? _tmem_load_8_3[9] : 0.0f);
                        wo_2_3[10] = ((row_valid != 0) ? _tmem_load_3[10] : 0.0f);
                        uo_1_3[10] = ((row_valid != 0) ? _tmem_load_8_3[10] : 0.0f);
                        wo_2_3[11] = ((row_valid != 0) ? _tmem_load_3[11] : 0.0f);
                        uo_1_3[11] = ((row_valid != 0) ? _tmem_load_8_3[11] : 0.0f);
                        wo_2_3[12] = ((row_valid != 0) ? _tmem_load_3[12] : 0.0f);
                        uo_1_3[12] = ((row_valid != 0) ? _tmem_load_8_3[12] : 0.0f);
                        wo_2_3[13] = ((row_valid != 0) ? _tmem_load_3[13] : 0.0f);
                        uo_1_3[13] = ((row_valid != 0) ? _tmem_load_8_3[13] : 0.0f);
                        wo_2_3[14] = ((row_valid != 0) ? _tmem_load_3[14] : 0.0f);
                        uo_1_3[14] = ((row_valid != 0) ? _tmem_load_8_3[14] : 0.0f);
                        wo_2_3[15] = ((row_valid != 0) ? _tmem_load_3[15] : 0.0f);
                        uo_1_3[15] = ((row_valid != 0) ? _tmem_load_8_3[15] : 0.0f);
                        unsigned int pw_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 0], wo_2_3[_lp*2+1 + 0]));
                            pw_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 2], wo_2_3[_lp*2+1 + 2]));
                            pw_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 4], wo_2_3[_lp*2+1 + 4]));
                            pw_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 6], wo_2_3[_lp*2+1 + 6]));
                            pw_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 8], wo_2_3[_lp*2+1 + 8]));
                            pw_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 10], wo_2_3[_lp*2+1 + 10]));
                            pw_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 12], wo_2_3[_lp*2+1 + 12]));
                            pw_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wo_2_3[_lp*2 + 14], wo_2_3[_lp*2+1 + 14]));
                            pw_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_3 * 2), "r"(pw_3[0]), "r"(pw_3[1]), "r"(pw_3[2]), "r"(pw_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(wrow + col_1_3 * 2 + 16), "r"(pw_3[4]), "r"(pw_3[5]), "r"(pw_3[6]), "r"(pw_3[7]) : "memory");
                        unsigned int pu_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 0], uo_1_3[_lp*2+1 + 0]));
                            pu_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 2], uo_1_3[_lp*2+1 + 2]));
                            pu_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 4], uo_1_3[_lp*2+1 + 4]));
                            pu_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 6], uo_1_3[_lp*2+1 + 6]));
                            pu_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 8], uo_1_3[_lp*2+1 + 8]));
                            pu_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 10], uo_1_3[_lp*2+1 + 10]));
                            pu_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 12], uo_1_3[_lp*2+1 + 12]));
                            pu_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(uo_1_3[_lp*2 + 14], uo_1_3[_lp*2+1 + 14]));
                            pu_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_3 * 2), "r"(pu_3[0]), "r"(pu_3[1]), "r"(pu_3[2]), "r"(pu_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(urow + col_1_3 * 2 + 16), "r"(pu_3[4]), "r"(pu_3[5]), "r"(pu_3[6]), "r"(pu_3[7]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            long long grow0 = ((long long)(irow0 + w_id * 16) * (long long)num_heads + (long long)(hv - 1)) * 128ll;
                            long long gpitch = (long long)num_heads * 128ll;
                            {
                                void* _cpbulk_dst_290 = reinterpret_cast<void*>(w_out + grow0);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_290), "r"(l_s_addr + (unsigned int)(w_id * 16 * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_291 = reinterpret_cast<void*>(w_out + (grow0 + gpitch));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_291), "r"(l_s_addr + (unsigned int)((w_id * 16 + 1) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_292 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 2ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_292), "r"(l_s_addr + (unsigned int)((w_id * 16 + 2) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_293 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 3ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_293), "r"(l_s_addr + (unsigned int)((w_id * 16 + 3) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_294 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 4ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_294), "r"(l_s_addr + (unsigned int)((w_id * 16 + 4) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_295 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 5ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_295), "r"(l_s_addr + (unsigned int)((w_id * 16 + 5) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_296 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 6ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_296), "r"(l_s_addr + (unsigned int)((w_id * 16 + 6) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_297 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 7ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_297), "r"(l_s_addr + (unsigned int)((w_id * 16 + 7) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_298 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 8ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_298), "r"(l_s_addr + (unsigned int)((w_id * 16 + 8) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_299 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 9ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_299), "r"(l_s_addr + (unsigned int)((w_id * 16 + 9) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_300 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 10ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_300), "r"(l_s_addr + (unsigned int)((w_id * 16 + 10) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_301 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 11ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_301), "r"(l_s_addr + (unsigned int)((w_id * 16 + 11) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_302 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 12ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_302), "r"(l_s_addr + (unsigned int)((w_id * 16 + 12) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_303 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 13ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_303), "r"(l_s_addr + (unsigned int)((w_id * 16 + 13) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_304 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 14ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_304), "r"(l_s_addr + (unsigned int)((w_id * 16 + 14) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_305 = reinterpret_cast<void*>(w_out + (grow0 + gpitch * 15ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_305), "r"(l_s_addr + (unsigned int)((w_id * 16 + 15) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_306 = reinterpret_cast<void*>(u_out + grow0);
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_306), "r"(a_s_addr + (unsigned int)(w_id * 16 * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_307 = reinterpret_cast<void*>(u_out + (grow0 + gpitch));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_307), "r"(a_s_addr + (unsigned int)((w_id * 16 + 1) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_308 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 2ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_308), "r"(a_s_addr + (unsigned int)((w_id * 16 + 2) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_309 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 3ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_309), "r"(a_s_addr + (unsigned int)((w_id * 16 + 3) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_310 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 4ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_310), "r"(a_s_addr + (unsigned int)((w_id * 16 + 4) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_311 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 5ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_311), "r"(a_s_addr + (unsigned int)((w_id * 16 + 5) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_312 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 6ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_312), "r"(a_s_addr + (unsigned int)((w_id * 16 + 6) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_313 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 7ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_313), "r"(a_s_addr + (unsigned int)((w_id * 16 + 7) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_314 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 8ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_314), "r"(a_s_addr + (unsigned int)((w_id * 16 + 8) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_315 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 9ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_315), "r"(a_s_addr + (unsigned int)((w_id * 16 + 9) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_316 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 10ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_316), "r"(a_s_addr + (unsigned int)((w_id * 16 + 10) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_317 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 11ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_317), "r"(a_s_addr + (unsigned int)((w_id * 16 + 11) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_318 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 12ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_318), "r"(a_s_addr + (unsigned int)((w_id * 16 + 12) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_319 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 13ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_319), "r"(a_s_addr + (unsigned int)((w_id * 16 + 13) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_320 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 14ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_320), "r"(a_s_addr + (unsigned int)((w_id * 16 + 14) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            {
                                void* _cpbulk_dst_321 = reinterpret_cast<void*>(u_out + (grow0 + gpitch * 15ll));
                                asm volatile(
                                    "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                                    :: "l"(_cpbulk_dst_321), "r"(a_s_addr + (unsigned int)((w_id * 16 + 15) * 272)), "r"((uint32_t)(256))
                                    : "memory");
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (group > j) {
                        asm volatile("cp.async.wait_group 0;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(op_ready_addr);
                        vst_c += 1;
                        if (vst_c == 2) { vst_c = 0; _phase_v_full ^= 1; }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                }
                asm volatile("cp.async.bulk.wait_group 0;");
            } else {
                #pragma unroll 1
                for (int j_1 = 0; j_1 < group; j_1++) {
                    int hv_p = hq * group + j_1;
                    if (tid_0 < 64) {
                        long long rbase = (long long)(irow0 + tid_0) * (long long)num_heads + (long long)hv_p;
                        unsigned int packed[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_322 = reinterpret_cast<const unsigned*>(packed);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + rbase * 128ll)), "r"(_raw_stv8_322[0]), "r"(_raw_stv8_322[1]), "r"(_raw_stv8_322[2]), "r"(_raw_stv8_322[3]), "r"(_raw_stv8_322[4]), "r"(_raw_stv8_322[5]), "r"(_raw_stv8_322[6]), "r"(_raw_stv8_322[7]) : "memory");
                        }
                        unsigned int packed_0[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_323 = reinterpret_cast<const unsigned*>(packed_0);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + rbase * 128ll)), "r"(_raw_stv8_323[0]), "r"(_raw_stv8_323[1]), "r"(_raw_stv8_323[2]), "r"(_raw_stv8_323[3]), "r"(_raw_stv8_323[4]), "r"(_raw_stv8_323[5]), "r"(_raw_stv8_323[6]), "r"(_raw_stv8_323[7]) : "memory");
                        }
                        unsigned int packed_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_324 = reinterpret_cast<const unsigned*>(packed_1);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 16ll))), "r"(_raw_stv8_324[0]), "r"(_raw_stv8_324[1]), "r"(_raw_stv8_324[2]), "r"(_raw_stv8_324[3]), "r"(_raw_stv8_324[4]), "r"(_raw_stv8_324[5]), "r"(_raw_stv8_324[6]), "r"(_raw_stv8_324[7]) : "memory");
                        }
                        unsigned int packed_2[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_325 = reinterpret_cast<const unsigned*>(packed_2);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 16ll))), "r"(_raw_stv8_325[0]), "r"(_raw_stv8_325[1]), "r"(_raw_stv8_325[2]), "r"(_raw_stv8_325[3]), "r"(_raw_stv8_325[4]), "r"(_raw_stv8_325[5]), "r"(_raw_stv8_325[6]), "r"(_raw_stv8_325[7]) : "memory");
                        }
                        unsigned int packed_3[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_326 = reinterpret_cast<const unsigned*>(packed_3);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 32ll))), "r"(_raw_stv8_326[0]), "r"(_raw_stv8_326[1]), "r"(_raw_stv8_326[2]), "r"(_raw_stv8_326[3]), "r"(_raw_stv8_326[4]), "r"(_raw_stv8_326[5]), "r"(_raw_stv8_326[6]), "r"(_raw_stv8_326[7]) : "memory");
                        }
                        unsigned int packed_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_327 = reinterpret_cast<const unsigned*>(packed_4);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 32ll))), "r"(_raw_stv8_327[0]), "r"(_raw_stv8_327[1]), "r"(_raw_stv8_327[2]), "r"(_raw_stv8_327[3]), "r"(_raw_stv8_327[4]), "r"(_raw_stv8_327[5]), "r"(_raw_stv8_327[6]), "r"(_raw_stv8_327[7]) : "memory");
                        }
                        unsigned int packed_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_328 = reinterpret_cast<const unsigned*>(packed_5);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 48ll))), "r"(_raw_stv8_328[0]), "r"(_raw_stv8_328[1]), "r"(_raw_stv8_328[2]), "r"(_raw_stv8_328[3]), "r"(_raw_stv8_328[4]), "r"(_raw_stv8_328[5]), "r"(_raw_stv8_328[6]), "r"(_raw_stv8_328[7]) : "memory");
                        }
                        unsigned int packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_329 = reinterpret_cast<const unsigned*>(packed_6);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 48ll))), "r"(_raw_stv8_329[0]), "r"(_raw_stv8_329[1]), "r"(_raw_stv8_329[2]), "r"(_raw_stv8_329[3]), "r"(_raw_stv8_329[4]), "r"(_raw_stv8_329[5]), "r"(_raw_stv8_329[6]), "r"(_raw_stv8_329[7]) : "memory");
                        }
                        unsigned int packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_330 = reinterpret_cast<const unsigned*>(packed_7);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 64ll))), "r"(_raw_stv8_330[0]), "r"(_raw_stv8_330[1]), "r"(_raw_stv8_330[2]), "r"(_raw_stv8_330[3]), "r"(_raw_stv8_330[4]), "r"(_raw_stv8_330[5]), "r"(_raw_stv8_330[6]), "r"(_raw_stv8_330[7]) : "memory");
                        }
                        unsigned int packed_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_331 = reinterpret_cast<const unsigned*>(packed_8);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 64ll))), "r"(_raw_stv8_331[0]), "r"(_raw_stv8_331[1]), "r"(_raw_stv8_331[2]), "r"(_raw_stv8_331[3]), "r"(_raw_stv8_331[4]), "r"(_raw_stv8_331[5]), "r"(_raw_stv8_331[6]), "r"(_raw_stv8_331[7]) : "memory");
                        }
                        unsigned int packed_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_332 = reinterpret_cast<const unsigned*>(packed_9);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 80ll))), "r"(_raw_stv8_332[0]), "r"(_raw_stv8_332[1]), "r"(_raw_stv8_332[2]), "r"(_raw_stv8_332[3]), "r"(_raw_stv8_332[4]), "r"(_raw_stv8_332[5]), "r"(_raw_stv8_332[6]), "r"(_raw_stv8_332[7]) : "memory");
                        }
                        unsigned int packed_10[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_333 = reinterpret_cast<const unsigned*>(packed_10);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 80ll))), "r"(_raw_stv8_333[0]), "r"(_raw_stv8_333[1]), "r"(_raw_stv8_333[2]), "r"(_raw_stv8_333[3]), "r"(_raw_stv8_333[4]), "r"(_raw_stv8_333[5]), "r"(_raw_stv8_333[6]), "r"(_raw_stv8_333[7]) : "memory");
                        }
                        unsigned int packed_11[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_334 = reinterpret_cast<const unsigned*>(packed_11);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 96ll))), "r"(_raw_stv8_334[0]), "r"(_raw_stv8_334[1]), "r"(_raw_stv8_334[2]), "r"(_raw_stv8_334[3]), "r"(_raw_stv8_334[4]), "r"(_raw_stv8_334[5]), "r"(_raw_stv8_334[6]), "r"(_raw_stv8_334[7]) : "memory");
                        }
                        unsigned int packed_12[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_335 = reinterpret_cast<const unsigned*>(packed_12);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 96ll))), "r"(_raw_stv8_335[0]), "r"(_raw_stv8_335[1]), "r"(_raw_stv8_335[2]), "r"(_raw_stv8_335[3]), "r"(_raw_stv8_335[4]), "r"(_raw_stv8_335[5]), "r"(_raw_stv8_335[6]), "r"(_raw_stv8_335[7]) : "memory");
                        }
                        unsigned int packed_13[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_336 = reinterpret_cast<const unsigned*>(packed_13);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(w_out + (rbase * 128ll + 112ll))), "r"(_raw_stv8_336[0]), "r"(_raw_stv8_336[1]), "r"(_raw_stv8_336[2]), "r"(_raw_stv8_336[3]), "r"(_raw_stv8_336[4]), "r"(_raw_stv8_336[5]), "r"(_raw_stv8_336[6]), "r"(_raw_stv8_336[7]) : "memory");
                        }
                        unsigned int packed_14[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_337 = reinterpret_cast<const unsigned*>(packed_14);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(u_out + (rbase * 128ll + 112ll))), "r"(_raw_stv8_337[0]), "r"(_raw_stv8_337[1]), "r"(_raw_stv8_337[2]), "r"(_raw_stv8_337[3]), "r"(_raw_stv8_337[4]), "r"(_raw_stv8_337[5]), "r"(_raw_stv8_337[6]), "r"(_raw_stv8_337[7]) : "memory");
                        }
                        unsigned int packed_15[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_15[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_15[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_15[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_15[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_15[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_15[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_338 = reinterpret_cast<const unsigned*>(packed_15);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + rbase * 64ll)), "r"(_raw_stv8_338[0]), "r"(_raw_stv8_338[1]), "r"(_raw_stv8_338[2]), "r"(_raw_stv8_338[3]), "r"(_raw_stv8_338[4]), "r"(_raw_stv8_338[5]), "r"(_raw_stv8_338[6]), "r"(_raw_stv8_338[7]) : "memory");
                        }
                        unsigned int packed_16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_16[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_16[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_16[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_16[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_16[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_16[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_16[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_339 = reinterpret_cast<const unsigned*>(packed_16);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64ll + 16ll))), "r"(_raw_stv8_339[0]), "r"(_raw_stv8_339[1]), "r"(_raw_stv8_339[2]), "r"(_raw_stv8_339[3]), "r"(_raw_stv8_339[4]), "r"(_raw_stv8_339[5]), "r"(_raw_stv8_339[6]), "r"(_raw_stv8_339[7]) : "memory");
                        }
                        unsigned int packed_17[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_17[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_17[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_17[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_17[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_17[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_17[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_17[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_340 = reinterpret_cast<const unsigned*>(packed_17);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64ll + 32ll))), "r"(_raw_stv8_340[0]), "r"(_raw_stv8_340[1]), "r"(_raw_stv8_340[2]), "r"(_raw_stv8_340[3]), "r"(_raw_stv8_340[4]), "r"(_raw_stv8_340[5]), "r"(_raw_stv8_340[6]), "r"(_raw_stv8_340[7]) : "memory");
                        }
                        unsigned int packed_18[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 0], zero16[_lp*2+1 + 0]));
                            packed_18[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 2], zero16[_lp*2+1 + 2]));
                            packed_18[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 4], zero16[_lp*2+1 + 4]));
                            packed_18[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 6], zero16[_lp*2+1 + 6]));
                            packed_18[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 8], zero16[_lp*2+1 + 8]));
                            packed_18[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 10], zero16[_lp*2+1 + 10]));
                            packed_18[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 12], zero16[_lp*2+1 + 12]));
                            packed_18[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(zero16[_lp*2 + 14], zero16[_lp*2+1 + 14]));
                            packed_18[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        {
                            const unsigned* _raw_stv8_341 = reinterpret_cast<const unsigned*>(packed_18);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(aqk_out + (rbase * 64ll + 48ll))), "r"(_raw_stv8_341[0]), "r"(_raw_stv8_341[1]), "r"(_raw_stv8_341[2]), "r"(_raw_stv8_341[3]), "r"(_raw_stv8_341[4]), "r"(_raw_stv8_341[5]), "r"(_raw_stv8_341[6]), "r"(_raw_stv8_341[7]) : "memory");
                        }
                        *(reinterpret_cast<float*>(g_cum + rbase) + (0)) = 0.0f;
                        *(reinterpret_cast<float*>(rowscale + rbase) + (0)) = 0.0f;
                        *(reinterpret_cast<float*>(qscale + rbase) + (0)) = 0.0f;
                    }
                    if (tid_0 == 0) {
                        *(reinterpret_cast<float*>(gamma + ((long long)chunk * (long long)num_heads + (long long)hv_p)) + (0)) = 1.0f;
                    }
                }
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (w_id == 0) {
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            unsigned int vst_l = 0;
#if __CUDA_ARCH__ == 1070
#else
            int lane_l = lane;
            int lin_n = chunk + gridDim.x * hq + 152;
            int c_n = lin_n % gridDim.x;
            int hq_n = lin_n / gridDim.x;
            int ext0_n = chunk_bos[c_n];
            int clen_n = chunk_len[c_n];
            int nrow_n = ((hq_n < gridDim.y) ? clen_n : 0);
            long long tokp0 = (long long)(ext0_n + lane_l) * (long long)num_heads + (long long)(hq_n * group);
            long long tokp1 = (long long)(ext0_n + 32 + lane_l) * (long long)num_heads + (long long)(hq_n * group);
            if (lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(g + tokp0))); }
            if (lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(beta + tokp0))); }
            if (32 + lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(g + tokp1))); }
            if (32 + lane_l < nrow_n) { asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(beta + tokp1))); }
#endif
            unsigned int _phase_v_empty = 1;
            if (clen > 0) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(qk_full_addr, 32768);
                    int c0_1_0 = 0;
                    tma_3d_gmem2smem(q_p_addr, (&q_tma), c0_1_0, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr, (&k_tma), c0_1_0, hq, ext0, qk_full_addr);
                    int c0_1_1 = 64;
                    tma_3d_gmem2smem(q_p_addr + 8192, (&q_tma), c0_1_1, hq, ext0, qk_full_addr);
                    tma_3d_gmem2smem(k_p_addr + 8192, (&k_tma), c0_1_1, hq, ext0, qk_full_addr);
                    #pragma unroll 1
                    for (int j_2 = 0; j_2 < group; j_2++) {
                        int hv_l = hq * group + j_2;
                        mbarrier_wait(v_empty_addr + (vst_l) * 8, _phase_v_empty);
                        mbarrier_arrive_expect_tx(v_full_addr + (vst_l) * 8, 16384);
                        int c0v_1_0 = 0;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384, (&v_tma), c0v_1_0, hv_l, ext0, v_full_addr + (vst_l) * 8);
                        int c0v_1_1 = 64;
                        tma_3d_gmem2smem(v_p_addr + vst_l * 16384 + 8192, (&v_tma), c0v_1_1, hv_l, ext0, v_full_addr + (vst_l) * 8);
                        vst_l += 1;
                        if (vst_l == 2) { vst_l = 0; _phase_v_empty ^= 1; }
                    }
#if __CUDA_ARCH__ == 1070
#else
                    if (nrow_n > 0) {
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(0)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(0)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&q_tma))), "r"((int)(64)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&k_tma))), "r"((int)(64)), "r"((int)(hq_n)), "r"((int)(ext0_n)) : "memory");
                        #pragma unroll 1
                        for (int jp = 0; jp < group; jp++) {
                            int hv_n = hq_n * group + jp;
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(0)), "r"((int)(hv_n)), "r"((int)(ext0_n)) : "memory");
                            asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&v_tma))), "r"((int)(64)), "r"((int)(hv_n)), "r"((int)(ext0_n)) : "memory");
                        }
                    }
#endif
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 5) {
        { // mma_main
            unsigned int vst_m = 0;
            unsigned int _phase_qk_full_0_1 = 0;
            unsigned int _phase_qkn_ready_0 = 0;
            unsigned int _phase_op_ready_0 = 0;
            if (clen > 0) {
                if (elect_sync()) {
                    mbarrier_wait(qk_full_addr, _phase_qk_full_0_1);
                    _phase_qk_full_0_1 ^= 1;
                    mbarrier_wait(qkn_ready_addr, _phase_qkn_ready_0);
                    _phase_qkn_ready_0 ^= 1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                    int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x7FFF;
                    int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x7FFF;
#else
                    int _mma_a_lo_0 = ((q_k_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = ((k_k_addr) >> 4) & 0x3FFF;
#endif
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 68158608;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_acc), "r"(0) : "memory");
                    tcgen05_commit(gram_full_addr);
                    #pragma unroll 1
                    for (int _j = 0; _j < group; _j++) {
                        mbarrier_wait(op_ready_addr, _phase_op_ready_0);
                        _phase_op_ready_0 ^= 1;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = ((akk_t_addr) >> 4) & 0x7FFF;
                        int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x7FFF) | 0x2000000;
#else
                        int _mma_a_lo_1 = ((akk_t_addr) >> 4) & 0x3FFF;
                        int _smem_desc_lo_0 = (((kb_mn_addr) >> 4) & 0x3FFF) | 0x2000000;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_smem_desc_lo_0), "r"(tmem_acc), "r"(0) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x7FFF) | 0x2000000) + (vst_m) * 1024;
#else
                        int _smem_desc_lo_1 = ((((v_mn_addr) >> 4) & 0x3FFF) | 0x2000000) + (vst_m) * 1024;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_smem_desc_lo_1), "r"((tmem_acc + (128))), "r"(0) : "memory");
                        tcgen05_commit(wu_full_addr);
                        tcgen05_commit(v_empty_addr + (vst_m) * 8);
                        vst_m += 1;
                        if (vst_m == 2) { vst_m = 0; }
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
