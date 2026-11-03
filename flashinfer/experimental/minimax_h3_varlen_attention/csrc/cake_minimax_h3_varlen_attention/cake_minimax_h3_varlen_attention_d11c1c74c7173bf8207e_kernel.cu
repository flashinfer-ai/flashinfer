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

template <typename T, int N>
struct CakeParamArray {
    T v[N];
    __device__ __forceinline__ const T& operator[](int i) const { return v[i]; }
};

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBS_OFFSET 32
#define TMEM_OUTPUT_OFFSET 128
#define NUM_MAIN_STAGES 1
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_K_OFF 33792
#define SMEM_SMEM_K_STAGE_BYTES 8192
#define SMEM_SMEM_K_STRIDE 8192
#define SMEM_SMEM_V_OFF 41984
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_TOTAL 50176
#define THREADS 128

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


__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


#if __CUDA_ARCH__ == 1030
#else

__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}
#endif



__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}




__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(128) void
kernel_cake_minimax_h3_varlen_attention_d11c1c74c7173bf8207e(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, const CakeParamArray<int, 65> cu_seqlens, int num_heads, float softmax_scale_log2)
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
    #define ld_full_addr (mbar_base + 0)
    #define s_full_addr (mbar_base + 8)
    #define o_full_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_k_addr = smem + 33792;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_addr = smem + 41984;
    if (warp == 1) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory"); }
    if (warp == 1) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory"); }
    if (warp == 1) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 3 barriers)
    // Mbarriers at smem_raw[0..24)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // ld_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel pre-TMEM ops
    int head = (int)blockIdx.x;
    int segment = (int)blockIdx.y;
    int seg_begin = cu_seqlens[segment];
    int seg_len = cu_seqlens[segment + 1] - seg_begin;
    if (warp == 1) {
        if (elect_sync()) {
            mbarrier_arrive_expect_tx(ld_full_addr, 24576);
            tma_4d_gmem2smem(smem_q_addr, (&Q), 0, seg_begin, head, 0, ld_full_addr);
            tma_4d_gmem2smem(smem_q_addr + 16384, (&Q), 0, seg_begin, head, 1, ld_full_addr);
            tma_4d_gmem2smem(smem_k_addr, (&K), 0, seg_begin, head, 0, ld_full_addr);
            tma_4d_gmem2smem(smem_v_addr, (&V), 0, seg_begin, head, 0, ld_full_addr);
        }
    }

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 24);
    if (warp == 0) {
        int _tmem_hold = smem + 24;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probs = taddr + 32;
    const int tmem_output = taddr + 128;

    // === Task calls (declaration order) ===
    unsigned int _phase_ld_full_0 = 0;
    unsigned int _phase_s_full_0 = 0;
    unsigned int _phase_o_full_0 = 0;
    if (warp == 0) {
        mbarrier_wait(ld_full_addr, _phase_ld_full_0);
        _phase_ld_full_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_a_lo_0 = make_warp_uniform(((smem_q_addr) >> 4) & 0x3FFF);
        int _mma_b_lo_0 = make_warp_uniform(((smem_k_addr) >> 4) & 0x3FFF);
        {
            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
            if (elect_sync()) {
                tcgen05_mma_f16(tmem_scores, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
            }
        }
        elect_commit(s_full_addr);
        mbarrier_wait(s_full_addr, _phase_s_full_0);
        _phase_s_full_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        float sv[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
            : "r"(taddr));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        float tile_max = -CUDART_INF_F;
        if (seg_len < 32) {
            uint32_t _thresh_mask_1;
            {
                int _lim_2 = seg_len;
                if (_lim_2 <= 0) { _thresh_mask_1 = 0u; }
                else if (_lim_2 >= 32) { _thresh_mask_1 = 0xFFFFFFFFu; }
                else {
                    asm volatile("{"
                        ".reg .u32 t;\n\t"
                        "shl.b32 t, 1, %1;\n\t"
                        "add.u32 %0, t, -1;\n\t"
                        "}" : "=r"(_thresh_mask_1) : "r"(_lim_2));
                }
            }
            if (!(_thresh_mask_1 & (1u << 0))) sv[0] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 1))) sv[1] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 2))) sv[2] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 3))) sv[3] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 4))) sv[4] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 5))) sv[5] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 6))) sv[6] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 7))) sv[7] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 8))) sv[8] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 9))) sv[9] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 10))) sv[10] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 11))) sv[11] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 12))) sv[12] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 13))) sv[13] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 14))) sv[14] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 15))) sv[15] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 16))) sv[16] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 17))) sv[17] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 18))) sv[18] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 19))) sv[19] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 20))) sv[20] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 21))) sv[21] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 22))) sv[22] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 23))) sv[23] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 24))) sv[24] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 25))) sv[25] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 26))) sv[26] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 27))) sv[27] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 28))) sv[28] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 29))) sv[29] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 30))) sv[30] = -CUDART_INF_F;
            if (!(_thresh_mask_1 & (1u << 31))) sv[31] = -CUDART_INF_F;
        }
        float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[0], sv[1]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[2], sv[3]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[4], sv[5]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[6], sv[7]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[8], sv[9]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[10], sv[11]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[12], sv[13]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[14], sv[15]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[16], sv[17]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[18], sv[19]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[20], sv[21]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[22], sv[23]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[24], sv[25]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[26], sv[27]));
        _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[28], sv[29]));
        _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[30], sv[31]));
        float _reg_array_reduce_3 = row_max_reduce(_reg_reduce_max2_4);
        tile_max = _reg_array_reduce_3;
        float row_max = -CUDART_INF_F;
        float row_sum = 0.0f;
        float new_max;
        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
        float new_max_scaled = safe_max * softmax_scale_log2;
        float acc_scale = 1.0f;
        float neg_max_scaled = -new_max_scaled;
        const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_6 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_5, _fma_c2_6);
        const float2 _fma_b2_7 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_8 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 2))[_lf], _fma_b2_7, _fma_c2_8);
        const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_10 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 4))[_lf], _fma_b2_9, _fma_c2_10);
        const float2 _fma_b2_11 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_12 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 6))[_lf], _fma_b2_11, _fma_c2_12);
        const float2 _fma_b2_13 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_14 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 8))[_lf], _fma_b2_13, _fma_c2_14);
        const float2 _fma_b2_15 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_16 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 10))[_lf], _fma_b2_15, _fma_c2_16);
#if __CUDA_ARCH__ == 1030
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le] = approx_exp2(sv[_le]);
        }
        const float2 _fma_b2_17 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_18 = {neg_max_scaled, neg_max_scaled};
#else
        float2 _exp2_pair_17 = ex2_emulation_f32x2_value(make_float2(sv[0], sv[1]));
        sv[0] = _exp2_pair_17.x;
        sv[1] = _exp2_pair_17.y;
        const float2 _fma_b2_18 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_19 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_17, _fma_c2_18);
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 2] = approx_exp2(sv[_le + 2]);
        }
        const float2 _fma_b2_19 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_20 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_19, _fma_c2_20);
        float2 psum0 = make_float2(sv[0], sv[1]);
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 4] = approx_exp2(sv[_le + 4]);
        }
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 12))[_lf], _fma_b2_18, _fma_c2_19);
        float2 _exp2_pair_20 = ex2_emulation_f32x2_value(make_float2(sv[2], sv[3]));
        sv[2] = _exp2_pair_20.x;
        sv[3] = _exp2_pair_20.y;
#endif
        const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_22 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_21, _fma_c2_22);
        psum0 = add_f32x2(psum0, make_float2(sv[2], sv[3]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 6] = approx_exp2(sv[_le + 6]);
        }
        const float2 _fma_b2_23 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_24 = {neg_max_scaled, neg_max_scaled};
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 14))[_lf], _fma_b2_21, _fma_c2_22);
        float2 psum0 = make_float2(sv[0], sv[1]);
        float2 _exp2_pair_23 = ex2_emulation_f32x2_value(make_float2(sv[4], sv[5]));
        sv[4] = _exp2_pair_23.x;
        sv[5] = _exp2_pair_23.y;
        const float2 _fma_b2_24 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_25 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_23, _fma_c2_24);
        psum0 = add_f32x2(psum0, make_float2(sv[4], sv[5]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 8] = approx_exp2(sv[_le + 8]);
        }
        const float2 _fma_b2_25 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_26 = {neg_max_scaled, neg_max_scaled};
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_24, _fma_c2_25);
        psum0 = add_f32x2(psum0, make_float2(sv[2], sv[3]));
        float2 _exp2_pair_26 = ex2_emulation_f32x2_value(make_float2(sv[6], sv[7]));
        sv[6] = _exp2_pair_26.x;
        sv[7] = _exp2_pair_26.y;
        const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_28 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_25, _fma_c2_26);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 18))[_lf], _fma_b2_27, _fma_c2_28);
        psum0 = add_f32x2(psum0, make_float2(sv[4], sv[5]));
        float2 _exp2_pair_29 = ex2_emulation_f32x2_value(make_float2(sv[8], sv[9]));
        sv[8] = _exp2_pair_29.x;
        sv[9] = _exp2_pair_29.y;
        const float2 _fma_b2_30 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_31 = {neg_max_scaled, neg_max_scaled};
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 20))[_lf], _fma_b2_30, _fma_c2_31);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[6], sv[7]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 10] = approx_exp2(sv[_le + 10]);
        }
#if __CUDA_ARCH__ == 1030
        const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_28 = {neg_max_scaled, neg_max_scaled};
#else
        const float2 _fma_b2_32 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_33 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_27, _fma_c2_28);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 22))[_lf], _fma_b2_32, _fma_c2_33);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[8], sv[9]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 12] = approx_exp2(sv[_le + 12]);
        }
#if __CUDA_ARCH__ == 1030
        const float2 _fma_b2_29 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_30 = {neg_max_scaled, neg_max_scaled};
#else
        const float2 _fma_b2_34 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_35 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_29, _fma_c2_30);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 24))[_lf], _fma_b2_34, _fma_c2_35);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[10], sv[11]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 14] = approx_exp2(sv[_le + 14]);
        }
#if __CUDA_ARCH__ == 1030
        const float2 _fma_b2_31 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_32 = {neg_max_scaled, neg_max_scaled};
#else
        const float2 _fma_b2_36 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_37 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_31, _fma_c2_32);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 26))[_lf], _fma_b2_36, _fma_c2_37);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[12], sv[13]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 16] = approx_exp2(sv[_le + 16]);
        }
#if __CUDA_ARCH__ == 1030
        const float2 _fma_b2_33 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_34 = {neg_max_scaled, neg_max_scaled};
#else
        const float2 _fma_b2_38 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_39 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_33, _fma_c2_34);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 28))[_lf], _fma_b2_38, _fma_c2_39);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[14], sv[15]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 18] = approx_exp2(sv[_le + 18]);
        }
#if __CUDA_ARCH__ == 1030
        const float2 _fma_b2_35 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_36 = {neg_max_scaled, neg_max_scaled};
#else
        const float2 _fma_b2_40 = {softmax_scale_log2, softmax_scale_log2};
        const float2 _fma_c2_41 = {neg_max_scaled, neg_max_scaled};
#endif
        #pragma unroll
        for (int _lf = 0; _lf < 1; _lf++)
#if __CUDA_ARCH__ == 1030
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_35, _fma_c2_36);
#else
            fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 30))[_lf], _fma_b2_40, _fma_c2_41);
#endif
        psum0 = add_f32x2(psum0, make_float2(sv[16], sv[17]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 20] = approx_exp2(sv[_le + 20]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[18], sv[19]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 22] = approx_exp2(sv[_le + 22]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[20], sv[21]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 24] = approx_exp2(sv[_le + 24]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[22], sv[23]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 26] = approx_exp2(sv[_le + 26]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[24], sv[25]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 28] = approx_exp2(sv[_le + 28]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[26], sv[27]));
        #pragma unroll
        for (int _le = 0; _le < 2; _le++) {
            sv[_le + 30] = approx_exp2(sv[_le + 30]);
        }
        psum0 = add_f32x2(psum0, make_float2(sv[28], sv[29]));
        psum0 = add_f32x2(psum0, make_float2(sv[30], sv[31]));
        unsigned int sv_bf16[16];
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
            sv_bf16[_lp] = *(uint32_t*)&_bf2;
        }
        asm volatile(
            "tcgen05.st.sync.aligned.32x32b.x16.b32"
            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(taddr + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[15])));
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        __syncwarp();
        asm volatile("tcgen05.fence::after_thread_sync;");
        int _mma_b_lo_1 = make_warp_uniform((((smem_v_addr) >> 4) & 0x3FFF) | 0x1000000);
        asm volatile(
            "{\n\t"
            ".reg .pred leader, p0, p1;\n\t"
            ".reg .b32 dhi, blo, ta, id;\n\t"
            ".reg .b64 db;\n\t"
            "elect.sync _|leader, 0xFFFFFFFF;\n\t"
            "setp.ne.b32 p0, %3, 0;\n\t"
            "setp.ne.b32 p1, 1, 0;\n\t"
            ""
            "mov.b32 dhi, 0x40004040;\n\t"
            "mov.b32 id, 136381584;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_output), "r"(_mma_b_lo_1), "r"(tmem_probs), "r"(0) : "memory");
        elect_commit(o_full_addr);
        float2 zpair = make_float2(0.0f, 0.0f);
        float2 sum01 = add_f32x2(psum0, zpair);
        float2 sum23 = add_f32x2(zpair, zpair);
        float2 total_sum = add_f32x2(sum01, sum23);
        row_sum = __fmaf_rn(row_sum, acc_scale, total_sum.x) + total_sum.y;
        float final_scale;
        if (row_sum != 0.0f && row_sum == row_sum) {
#if __CUDA_ARCH__ == 1030
            float _rcp_37 = approx_rcp(row_sum);
            final_scale = _rcp_37;
#else
            float _rcp_42 = approx_rcp(row_sum);
            final_scale = _rcp_42;
#endif
        } else {
            final_scale = 0.0f;
        }
        int out_row = (seg_begin + lane) * num_heads + head;
        mbarrier_wait(o_full_addr, _phase_o_full_0);
        _phase_o_full_0 ^= 1;
        asm volatile("tcgen05.fence::after_thread_sync;");
        float _tmem_load_0[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
            : "r"(taddr + 128));
        float _tmem_load_1[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
            : "r"(taddr + 160));
        float _tmem_load_2[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
            : "r"(taddr + 192));
        float _tmem_load_3[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
            : "r"(taddr + 224));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (seg_len > lane) {
            float _scaled_0[32];
#if __CUDA_ARCH__ == 1030
            const float2 _scale2_38 = {final_scale, final_scale};
#else
            const float2 _scale2_43 = {final_scale, final_scale};
#endif
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_38);
#else
                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_43);
#endif
            unsigned int _packed_0[16];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                _packed_0[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 4], _scaled_0[_lp*2+1 + 4]));
                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 6], _scaled_0[_lp*2+1 + 6]));
                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 8], _scaled_0[_lp*2+1 + 8]));
                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 10], _scaled_0[_lp*2+1 + 10]));
                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 12], _scaled_0[_lp*2+1 + 12]));
                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 14], _scaled_0[_lp*2+1 + 14]));
                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 16], _scaled_0[_lp*2+1 + 16]));
                _packed_0[_lp + 8] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 18], _scaled_0[_lp*2+1 + 18]));
                _packed_0[_lp + 9] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 20], _scaled_0[_lp*2+1 + 20]));
                _packed_0[_lp + 10] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 22], _scaled_0[_lp*2+1 + 22]));
                _packed_0[_lp + 11] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 24], _scaled_0[_lp*2+1 + 24]));
                _packed_0[_lp + 12] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 26], _scaled_0[_lp*2+1 + 26]));
                _packed_0[_lp + 13] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 28], _scaled_0[_lp*2+1 + 28]));
                _packed_0[_lp + 14] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 30], _scaled_0[_lp*2+1 + 30]));
                _packed_0[_lp + 15] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(O + ((long long)out_row * 128))[0] = reinterpret_cast<int4*>(_packed_0)[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 8))[0] = reinterpret_cast<int4*>(_packed_0 + (4))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 16))[0] = reinterpret_cast<int4*>(_packed_0 + (8))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 24))[0] = reinterpret_cast<int4*>(_packed_0 + (12))[0];
            float _scaled_1[32];
#if __CUDA_ARCH__ == 1030
            const float2 _scale2_39 = {final_scale, final_scale};
#else
            const float2 _scale2_44 = {final_scale, final_scale};
#endif
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_39);
#else
                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_44);
#endif
            unsigned int _packed_1[16];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                _packed_1[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 4], _scaled_1[_lp*2+1 + 4]));
                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 6], _scaled_1[_lp*2+1 + 6]));
                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 8], _scaled_1[_lp*2+1 + 8]));
                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 10], _scaled_1[_lp*2+1 + 10]));
                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 12], _scaled_1[_lp*2+1 + 12]));
                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 14], _scaled_1[_lp*2+1 + 14]));
                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 16], _scaled_1[_lp*2+1 + 16]));
                _packed_1[_lp + 8] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 18], _scaled_1[_lp*2+1 + 18]));
                _packed_1[_lp + 9] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 20], _scaled_1[_lp*2+1 + 20]));
                _packed_1[_lp + 10] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 22], _scaled_1[_lp*2+1 + 22]));
                _packed_1[_lp + 11] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 24], _scaled_1[_lp*2+1 + 24]));
                _packed_1[_lp + 12] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 26], _scaled_1[_lp*2+1 + 26]));
                _packed_1[_lp + 13] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 28], _scaled_1[_lp*2+1 + 28]));
                _packed_1[_lp + 14] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 30], _scaled_1[_lp*2+1 + 30]));
                _packed_1[_lp + 15] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 32))[0] = reinterpret_cast<int4*>(_packed_1)[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 32 + 8))[0] = reinterpret_cast<int4*>(_packed_1 + (4))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 32 + 16))[0] = reinterpret_cast<int4*>(_packed_1 + (8))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 32 + 24))[0] = reinterpret_cast<int4*>(_packed_1 + (12))[0];
            float _scaled_2[32];
#if __CUDA_ARCH__ == 1030
            const float2 _scale2_40 = {final_scale, final_scale};
#else
            const float2 _scale2_45 = {final_scale, final_scale};
#endif
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_40);
#else
                reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_45);
#endif
            unsigned int _packed_2[16];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                _packed_2[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 2], _scaled_2[_lp*2+1 + 2]));
                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 4], _scaled_2[_lp*2+1 + 4]));
                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 6], _scaled_2[_lp*2+1 + 6]));
                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 8], _scaled_2[_lp*2+1 + 8]));
                _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 10], _scaled_2[_lp*2+1 + 10]));
                _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 12], _scaled_2[_lp*2+1 + 12]));
                _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 14], _scaled_2[_lp*2+1 + 14]));
                _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 16], _scaled_2[_lp*2+1 + 16]));
                _packed_2[_lp + 8] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 18], _scaled_2[_lp*2+1 + 18]));
                _packed_2[_lp + 9] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 20], _scaled_2[_lp*2+1 + 20]));
                _packed_2[_lp + 10] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 22], _scaled_2[_lp*2+1 + 22]));
                _packed_2[_lp + 11] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 24], _scaled_2[_lp*2+1 + 24]));
                _packed_2[_lp + 12] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 26], _scaled_2[_lp*2+1 + 26]));
                _packed_2[_lp + 13] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 28], _scaled_2[_lp*2+1 + 28]));
                _packed_2[_lp + 14] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 30], _scaled_2[_lp*2+1 + 30]));
                _packed_2[_lp + 15] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 64))[0] = reinterpret_cast<int4*>(_packed_2)[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 64 + 8))[0] = reinterpret_cast<int4*>(_packed_2 + (4))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 64 + 16))[0] = reinterpret_cast<int4*>(_packed_2 + (8))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 64 + 24))[0] = reinterpret_cast<int4*>(_packed_2 + (12))[0];
            float _scaled_3[32];
#if __CUDA_ARCH__ == 1030
            const float2 _scale2_41 = {final_scale, final_scale};
#else
            const float2 _scale2_46 = {final_scale, final_scale};
#endif
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
#if __CUDA_ARCH__ == 1030
                reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_41);
#else
                reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_46);
#endif
            unsigned int _packed_3[16];
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                _packed_3[_lp] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 2], _scaled_3[_lp*2+1 + 2]));
                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 4], _scaled_3[_lp*2+1 + 4]));
                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 6], _scaled_3[_lp*2+1 + 6]));
                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 8], _scaled_3[_lp*2+1 + 8]));
                _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 10], _scaled_3[_lp*2+1 + 10]));
                _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 12], _scaled_3[_lp*2+1 + 12]));
                _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 14], _scaled_3[_lp*2+1 + 14]));
                _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 16], _scaled_3[_lp*2+1 + 16]));
                _packed_3[_lp + 8] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 18], _scaled_3[_lp*2+1 + 18]));
                _packed_3[_lp + 9] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 20], _scaled_3[_lp*2+1 + 20]));
                _packed_3[_lp + 10] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 22], _scaled_3[_lp*2+1 + 22]));
                _packed_3[_lp + 11] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 24], _scaled_3[_lp*2+1 + 24]));
                _packed_3[_lp + 12] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 26], _scaled_3[_lp*2+1 + 26]));
                _packed_3[_lp + 13] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 28], _scaled_3[_lp*2+1 + 28]));
                _packed_3[_lp + 14] = *(uint32_t*)&_bf2;
            }
            #pragma unroll
            for (int _lp = 0; _lp < 1; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 30], _scaled_3[_lp*2+1 + 30]));
                _packed_3[_lp + 15] = *(uint32_t*)&_bf2;
            }
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 96))[0] = reinterpret_cast<int4*>(_packed_3)[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 96 + 8))[0] = reinterpret_cast<int4*>(_packed_3 + (4))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 96 + 16))[0] = reinterpret_cast<int4*>(_packed_3 + (8))[0];
            reinterpret_cast<int4*>(O + ((long long)out_row * 128 + 96 + 24))[0] = reinterpret_cast<int4*>(_packed_3 + (12))[0];
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
