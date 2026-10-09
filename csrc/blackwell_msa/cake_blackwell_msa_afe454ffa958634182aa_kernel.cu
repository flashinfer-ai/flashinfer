/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
#define TMEM_NCOLS 512
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_OFFSET 256
#define NUM_Q_PIPE_STAGES 2
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 2
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 32768
#define SMEM_Q_SMEM_STRIDE 32768
#define SMEM_Q_STORE_SMEM_OFF 1024
#define SMEM_Q_STORE_SMEM_STAGE_BYTES 32768
#define SMEM_Q_STORE_SMEM_STRIDE 32768
#define SMEM_K_SMEM_OFF 66560
#define SMEM_K_SMEM_STAGE_BYTES 32768
#define SMEM_K_SMEM_STRIDE 32768
#define SMEM_V_SMEM_OFF 99328
#define SMEM_V_SMEM_STAGE_BYTES 32768
#define SMEM_V_SMEM_STRIDE 32768
#define SMEM_V_CONVERT_SMEM_OFF 99328
#define SMEM_V_CONVERT_SMEM_STAGE_BYTES 32768
#define SMEM_V_CONVERT_SMEM_STRIDE 32768
#define SMEM_FP8_SMEM_OFF 132096
#define SMEM_FP8_SMEM_STAGE_BYTES 16384
#define SMEM_FP8_SMEM_STRIDE 16384
#define SMEM_TOTAL 148480
#define THREADS 512

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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.

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



__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
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


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}


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

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
    }
}



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




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

__global__ __launch_bounds__(512, 1) void
kernel_cake_blackwell_msa_afe454ffa958634182aa(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, uint8_t* __restrict__ partial_o, float* __restrict__ partial_scale, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, __nv_bfloat16* __restrict__ out, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int q_group_segment_end_128, int q_group_segment_end_64, int q_group_segment_end_32, int q_group_segment_end_16, int q_group_segment_end_8, int q_group_segment_end_4, int q_group_segment_end_2, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int work_capacity, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 40)
    #define fp8_k_full_addr (mbar_base + 48)
    #define fp8_v_full_addr (mbar_base + 56)
    #define fp8_empty_addr (mbar_base + 64)
    #define s_full_addr (mbar_base + 72)
    #define s_empty_addr (mbar_base + 88)
    #define p_full_addr (mbar_base + 104)
    #define p_full_2_addr (mbar_base + 120)
    #define p_empty_addr (mbar_base + 136)
    #define o_full_addr (mbar_base + 152)
    #define o_empty_addr (mbar_base + 168)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* q_store_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_store_smem_addr = smem + 1024;
    __nv_bfloat16* k_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int k_smem_addr = smem + 66560;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int v_smem_addr = smem + 99328;
    __nv_bfloat16* v_convert_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int v_convert_smem_addr = smem + 99328;
    uint8_t* fp8_smem = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int fp8_smem_addr = smem + 132096;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=4
            mbarrier_init(smem + 0, 4);
            mbarrier_init(smem + 8, 4);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            // v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            // fp8_k_full: 1 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            // fp8_v_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // fp8_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            mbarrier_init(smem + 96, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            mbarrier_init(smem + 112, 128);
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 120, 128);
            mbarrier_init(smem + 128, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 0) {
        int _tmem_hold = smem + 184;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax_even ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_even_main
            int group_count = 1;
            if (blockIdx.x < q_group_segment_end_16) {
                if (blockIdx.x < q_group_segment_end_64) {
                    group_count = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                } else {
                    group_count = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                }
            } else if (blockIdx.x < q_group_segment_end_4) {
                group_count = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
            } else {
                group_count = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
            }
            int group_count_0 = group_count;
            int work_idx = blockIdx.x;
            int metadata_base = work_idx * 6;
            int head_kv = scheduler_metadata[metadata_base];
            int row_linear = scheduler_metadata[metadata_base + 1];
            int q_begin = scheduler_metadata[metadata_base + 2];
            int q_count = scheduler_metadata[metadata_base + 3];
            int batch = scheduler_metadata[metadata_base + 4];
            int kv_block = scheduler_metadata[metadata_base + 5];
            int row_ptr_base = head_kv * (total_rows + 1) + row_linear;
            int row_start = k2q_row_ptr[row_ptr_base] + q_begin;
            int q_batch_offset = cu_seqlens_q[batch];
            int k_batch_offset = cu_seqlens_k[batch];
            int kv_len = kv_lens[batch];
            if (max_pages == 0) {
                kv_len = cu_seqlens_k[batch + 1] - k_batch_offset;
            }
            int query_offset = q_offsets[batch];
            if (derive_q_offset != 0) {
                query_offset = kv_len - (cu_seqlens_q[batch + 1] - q_batch_offset);
            }
            int stage_warp = warp;
            int my_row = stage_warp * 32 + lane;
            int tmem_row_base = stage_warp * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration = 0; stage_iteration < (group_count_0 + 1) / 2; stage_iteration++) {
                int group = stage_iteration * 2;
                int softmax_phase = stage_iteration & 1;
                mbarrier_wait(s_full_addr, softmax_phase);
                int whole_group_valid = 1;
                int token_in_group = 0;
                int edge_in_work = 0;
                int packed_q = 0;
                int q_idx = 0;
                int valid_cols = 0;
                float row_max = -CAKE_INF;
                float score_bias = -CAKE_INF;
                float score_values[128];
                int score_base = taddr + (unsigned int)tmem_row_base;
                if (whole_group_valid != 0) {
                    token_in_group = my_row / 16;
                    edge_in_work = group * 8 + token_in_group;
                    int row_valid = ((edge_in_work < q_count) ? 1 : 0);
                    int owner_lane = lane / 16 * 16;
                    int owned_packed = -1;
                    if (lane == owner_lane && edge_in_work < q_count) {
                        owned_packed = k2q_qsplit_indices[head_kv * nnz_per_head + row_start + edge_in_work];
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, owned_packed, owner_lane);
                    packed_q = _shfl_0;
                    q_idx = packed_q & 16777215;
                    if (row_valid != 0) {
                        valid_cols = kv_len - kv_block * 128;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (causal != 0) {
                            int query_position = query_offset + q_idx;
                            int causal_cols = query_position - kv_block * 128 + 1;
                            if (valid_cols > causal_cols) {
                                valid_cols = causal_cols;
                            }
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values[0]), "=f"(score_values[1]), "=f"(score_values[2]), "=f"(score_values[3]), "=f"(score_values[4]), "=f"(score_values[5]), "=f"(score_values[6]), "=f"(score_values[7]), "=f"(score_values[8]), "=f"(score_values[9]), "=f"(score_values[10]), "=f"(score_values[11]), "=f"(score_values[12]), "=f"(score_values[13]), "=f"(score_values[14]), "=f"(score_values[15]), "=f"(score_values[16]), "=f"(score_values[17]), "=f"(score_values[18]), "=f"(score_values[19]), "=f"(score_values[20]), "=f"(score_values[21]), "=f"(score_values[22]), "=f"(score_values[23]), "=f"(score_values[24]), "=f"(score_values[25]), "=f"(score_values[26]), "=f"(score_values[27]), "=f"(score_values[28]), "=f"(score_values[29]), "=f"(score_values[30]), "=f"(score_values[31])
                        : "r"(score_base));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values[32]), "=f"(score_values[33]), "=f"(score_values[34]), "=f"(score_values[35]), "=f"(score_values[36]), "=f"(score_values[37]), "=f"(score_values[38]), "=f"(score_values[39]), "=f"(score_values[40]), "=f"(score_values[41]), "=f"(score_values[42]), "=f"(score_values[43]), "=f"(score_values[44]), "=f"(score_values[45]), "=f"(score_values[46]), "=f"(score_values[47]), "=f"(score_values[48]), "=f"(score_values[49]), "=f"(score_values[50]), "=f"(score_values[51]), "=f"(score_values[52]), "=f"(score_values[53]), "=f"(score_values[54]), "=f"(score_values[55]), "=f"(score_values[56]), "=f"(score_values[57]), "=f"(score_values[58]), "=f"(score_values[59]), "=f"(score_values[60]), "=f"(score_values[61]), "=f"(score_values[62]), "=f"(score_values[63])
                        : "r"(score_base + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values[64]), "=f"(score_values[65]), "=f"(score_values[66]), "=f"(score_values[67]), "=f"(score_values[68]), "=f"(score_values[69]), "=f"(score_values[70]), "=f"(score_values[71]), "=f"(score_values[72]), "=f"(score_values[73]), "=f"(score_values[74]), "=f"(score_values[75]), "=f"(score_values[76]), "=f"(score_values[77]), "=f"(score_values[78]), "=f"(score_values[79]), "=f"(score_values[80]), "=f"(score_values[81]), "=f"(score_values[82]), "=f"(score_values[83]), "=f"(score_values[84]), "=f"(score_values[85]), "=f"(score_values[86]), "=f"(score_values[87]), "=f"(score_values[88]), "=f"(score_values[89]), "=f"(score_values[90]), "=f"(score_values[91]), "=f"(score_values[92]), "=f"(score_values[93]), "=f"(score_values[94]), "=f"(score_values[95])
                        : "r"(score_base + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values[96]), "=f"(score_values[97]), "=f"(score_values[98]), "=f"(score_values[99]), "=f"(score_values[100]), "=f"(score_values[101]), "=f"(score_values[102]), "=f"(score_values[103]), "=f"(score_values[104]), "=f"(score_values[105]), "=f"(score_values[106]), "=f"(score_values[107]), "=f"(score_values[108]), "=f"(score_values[109]), "=f"(score_values[110]), "=f"(score_values[111]), "=f"(score_values[112]), "=f"(score_values[113]), "=f"(score_values[114]), "=f"(score_values[115]), "=f"(score_values[116]), "=f"(score_values[117]), "=f"(score_values[118]), "=f"(score_values[119]), "=f"(score_values[120]), "=f"(score_values[121]), "=f"(score_values[122]), "=f"(score_values[123]), "=f"(score_values[124]), "=f"(score_values[125]), "=f"(score_values[126]), "=f"(score_values[127])
                        : "r"(score_base + 96));
                    if (valid_cols < 128) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = valid_cols;
                            if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                            }
                        }
                        if (!(_slice_lo_mask_0 & (1u << 0))) score_values[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) score_values[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) score_values[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) score_values[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) score_values[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) score_values[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) score_values[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) score_values[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) score_values[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) score_values[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) score_values[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) score_values[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) score_values[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) score_values[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) score_values[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) score_values[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) score_values[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) score_values[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) score_values[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) score_values[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) score_values[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) score_values[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) score_values[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) score_values[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) score_values[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) score_values[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) score_values[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) score_values[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) score_values[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) score_values[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) score_values[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) score_values[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = valid_cols - 32;
                            if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_1 & (1u << 0))) score_values[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) score_values[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) score_values[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) score_values[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) score_values[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) score_values[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) score_values[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) score_values[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) score_values[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) score_values[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) score_values[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) score_values[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) score_values[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) score_values[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) score_values[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) score_values[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) score_values[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) score_values[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) score_values[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) score_values[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) score_values[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) score_values[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) score_values[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) score_values[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) score_values[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) score_values[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) score_values[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) score_values[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) score_values[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) score_values[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) score_values[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) score_values[63] = -CAKE_INF;
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_2 = valid_cols - 64;
                            if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
                            }
                        }
                        if (!(_slice_lo_mask_2 & (1u << 0))) score_values[64] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 1))) score_values[65] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 2))) score_values[66] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 3))) score_values[67] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 4))) score_values[68] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 5))) score_values[69] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 6))) score_values[70] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 7))) score_values[71] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 8))) score_values[72] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 9))) score_values[73] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 10))) score_values[74] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 11))) score_values[75] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 12))) score_values[76] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 13))) score_values[77] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 14))) score_values[78] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 15))) score_values[79] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 16))) score_values[80] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 17))) score_values[81] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 18))) score_values[82] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 19))) score_values[83] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 20))) score_values[84] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 21))) score_values[85] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 22))) score_values[86] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 23))) score_values[87] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 24))) score_values[88] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 25))) score_values[89] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 26))) score_values[90] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 27))) score_values[91] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 28))) score_values[92] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 29))) score_values[93] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 30))) score_values[94] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 31))) score_values[95] = -CAKE_INF;
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_3 = valid_cols - 96;
                            if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
                            }
                        }
                        if (!(_slice_lo_mask_3 & (1u << 0))) score_values[96] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 1))) score_values[97] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 2))) score_values[98] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 3))) score_values[99] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 4))) score_values[100] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 5))) score_values[101] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 6))) score_values[102] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 7))) score_values[103] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 8))) score_values[104] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 9))) score_values[105] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 10))) score_values[106] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 11))) score_values[107] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 12))) score_values[108] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 13))) score_values[109] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 14))) score_values[110] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 15))) score_values[111] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 16))) score_values[112] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 17))) score_values[113] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 18))) score_values[114] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 19))) score_values[115] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 20))) score_values[116] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 21))) score_values[117] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 22))) score_values[118] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 23))) score_values[119] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 24))) score_values[120] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 25))) score_values[121] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 26))) score_values[122] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 27))) score_values[123] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 28))) score_values[124] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 29))) score_values[125] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 30))) score_values[126] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 31))) score_values[127] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&score_values[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values[32], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values[64], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values[96], _reg_reduce_max2_4);
                    float score_values_max = row_max_reduce(_reg_reduce_max2_4);
                    row_max = score_values_max;
                    float safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                    score_bias = ((valid_cols > 0) ? (-safe_max) * softmax_scale_log2 : -CAKE_INF);
                }
                mbarrier_wait(p_empty_addr, softmax_phase ^ 1);
                float row_sum = 0.0f;
                if (whole_group_valid != 0) {
                    int p_base = taddr + 64 + (unsigned int)tmem_row_base;
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values)[_lf], _fma_b2_5, _fma_c2_6);
                    for (int probability_segment = 0; probability_segment < 2; probability_segment++) {
                        uint32_t score_values_bf16[16];
                        softmax_frag_exp2_cast(&(score_values + probability_segment * 32)[0], score_values_bf16, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base + probability_segment * 16), "r"(score_values_bf16[0]), "r"(score_values_bf16[1]), "r"(score_values_bf16[2]), "r"(score_values_bf16[3]), "r"(score_values_bf16[4]), "r"(score_values_bf16[5]), "r"(score_values_bf16[6]), "r"(score_values_bf16[7]), "r"(score_values_bf16[8]), "r"(score_values_bf16[9]), "r"(score_values_bf16[10]), "r"(score_values_bf16[11]), "r"(score_values_bf16[12]), "r"(score_values_bf16[13]), "r"(score_values_bf16[14]), "r"(score_values_bf16[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr);
                    uint32_t score_values_bf16_1[32];
                    softmax_frag_exp2_cast(&score_values[64], score_values_bf16_1, 0);
                    softmax_frag_exp2_cast(&score_values[96], &score_values_bf16_1[16], 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base + 32), "r"(score_values_bf16_1[0]), "r"(score_values_bf16_1[1]), "r"(score_values_bf16_1[2]), "r"(score_values_bf16_1[3]), "r"(score_values_bf16_1[4]), "r"(score_values_bf16_1[5]), "r"(score_values_bf16_1[6]), "r"(score_values_bf16_1[7]), "r"(score_values_bf16_1[8]), "r"(score_values_bf16_1[9]), "r"(score_values_bf16_1[10]), "r"(score_values_bf16_1[11]), "r"(score_values_bf16_1[12]), "r"(score_values_bf16_1[13]), "r"(score_values_bf16_1[14]), "r"(score_values_bf16_1[15]), "r"(score_values_bf16_1[16]), "r"(score_values_bf16_1[17]), "r"(score_values_bf16_1[18]), "r"(score_values_bf16_1[19]), "r"(score_values_bf16_1[20]), "r"(score_values_bf16_1[21]), "r"(score_values_bf16_1[22]), "r"(score_values_bf16_1[23]), "r"(score_values_bf16_1[24]), "r"(score_values_bf16_1[25]), "r"(score_values_bf16_1[26]), "r"(score_values_bf16_1[27]), "r"(score_values_bf16_1[28]), "r"(score_values_bf16_1[29]), "r"(score_values_bf16_1[30]), "r"(score_values_bf16_1[31]));
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values[64], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values[96], &_reg_reduce_sum2_7);
                    float score_values_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    row_sum = score_values_sum;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_2_addr);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr);
                mbarrier_wait(o_full_addr, softmax_phase);
                if (whole_group_valid != 0) {
                    int q_head_local = my_row - token_in_group * 16;
                    int output_valid = 0;
                    long long partial_row = 0;
                    long long final_row = 0;
                    float inv_sum = 0.0f;
                    int single_split = 0;
                    if (edge_in_work < q_count) {
                        int split_slot = packed_q >> 24 & 15;
                        if (split_slot >= 0 && split_slot < topk) {
                            output_valid = 1;
                            int q_abs = q_batch_offset + q_idx;
                            int q_head = head_kv * 16 + q_head_local;
                            final_row = (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
                            float _rcp_0 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_0 : 0.0f);
                        }
                    }
                    long long partial_base = partial_row * 128;
                    {
                        int output_row_addr = taddr + (unsigned int)TMEM_OUTPUT_OFFSET + (unsigned int)tmem_row_base;
                        long long partial_metadata_rows = (long long)topk * (long long)total_q * (long long)num_q_heads;
                        {
                            float row_min = CAKE_INF;
                            float row_max_0 = -CAKE_INF;
                            #pragma unroll 1
                            for (int output_segment = 0; output_segment < 4; output_segment++) {
                                float _tmem_load_1[16];
                                tmem_ld_x16(&_tmem_load_1[0], output_row_addr + output_segment * 16);
                                float _tmem_load_1_min = _tmem_load_1[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_1_min = fminf(_tmem_load_1_min, _tmem_load_1[_lr]);
                                }
                                float _min_0 = fminf(row_min, _tmem_load_1_min);
                                row_min = _min_0;
                                float _tmem_load_1_max = _tmem_load_1[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_1_max = max_noftz(_tmem_load_1_max, _tmem_load_1[_lr]);
                                }
                                float _max_0 = max_noftz(row_max_0, _tmem_load_1_max);
                                row_max_0 = _max_0;
                            }
                            float _tmem_load_2[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                                : "r"(output_row_addr + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                                : "r"(output_row_addr + 64 + 32));
                            float _tmem_load_2_min = _tmem_load_2[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 64; _lr++) {
                                _tmem_load_2_min = fminf(_tmem_load_2_min, _tmem_load_2[_lr]);
                            }
                            float _min_1 = fminf(row_min, _tmem_load_2_min);
                            row_min = _min_1;
                            float2 _reg_reduce_max2_8 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&_tmem_load_2[0], _reg_reduce_max2_8);
                            row_max_x32_accum(&_tmem_load_2[32], _reg_reduce_max2_8);
                            float _tmem_load_2_max = row_max_reduce(_reg_reduce_max2_8);
                            float _max_1 = max_noftz(row_max_0, _tmem_load_2_max);
                            row_max_0 = _max_1;
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float row_center = (row_max_0 + row_min) * 0.5f;
                            float residual_abs_max = (row_max_0 - row_min) * 0.5f;
                            float dequant_scale = 0.0f;
                            float quant_scale = 0.0f;
                            if (residual_abs_max > 0.0f && residual_abs_max == residual_abs_max) {
                                dequant_scale = residual_abs_max * inv_sum * 0.002232142857142857f;
                                quant_scale = 448.0f / residual_abs_max;
                            }
                            if (output_valid != 0) {
                                partial_scale[partial_row] = dequant_scale;
                                partial_scale[partial_metadata_rows + partial_row] = row_center * inv_sum;
                            }
                            #pragma unroll 1
                            for (int output_segment_1 = 0; output_segment_1 < 4; output_segment_1++) {
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], output_row_addr + output_segment_1 * 16);
                                const float2 _sub2_9 = {row_center, row_center};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _sub2_9);
                                if (output_valid != 0) {
                                    {
                                        const float2 _prescale2_10 = {quant_scale, quant_scale};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[0])[_ps], _prescale2_10);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_3[0 + _ps] *= quant_scale;
                                        #endif
                                        unsigned int _fp8_pk[4];
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[0]) : "f"(_tmem_load_3[0 + 0]), "f"(_tmem_load_3[0 + 1]), "f"(_tmem_load_3[0 + 2]), "f"(_tmem_load_3[0 + 3]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[1]) : "f"(_tmem_load_3[0 + 4]), "f"(_tmem_load_3[0 + 5]), "f"(_tmem_load_3[0 + 6]), "f"(_tmem_load_3[0 + 7]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[2]) : "f"(_tmem_load_3[0 + 8]), "f"(_tmem_load_3[0 + 9]), "f"(_tmem_load_3[0 + 10]), "f"(_tmem_load_3[0 + 11]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[3]) : "f"(_tmem_load_3[0 + 12]), "f"(_tmem_load_3[0 + 13]), "f"(_tmem_load_3[0 + 14]), "f"(_tmem_load_3[0 + 15]));
                                        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + (long long)output_segment_1 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                    }
                                }
                            }
                            const float2 _sub2_11 = {row_center, row_center};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _sub2_11);
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_12 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[0])[_ps], _prescale2_12);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_2[0 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_2[0 + 0]), "f"(_tmem_load_2[0 + 1]), "f"(_tmem_load_2[0 + 2]), "f"(_tmem_load_2[0 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_2[0 + 4]), "f"(_tmem_load_2[0 + 5]), "f"(_tmem_load_2[0 + 6]), "f"(_tmem_load_2[0 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_2[0 + 8]), "f"(_tmem_load_2[0 + 9]), "f"(_tmem_load_2[0 + 10]), "f"(_tmem_load_2[0 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_2[0 + 12]), "f"(_tmem_load_2[0 + 13]), "f"(_tmem_load_2[0 + 14]), "f"(_tmem_load_2[0 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 64)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_13 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[16])[_ps], _prescale2_13);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_2[16 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_2[16 + 0]), "f"(_tmem_load_2[16 + 1]), "f"(_tmem_load_2[16 + 2]), "f"(_tmem_load_2[16 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_2[16 + 4]), "f"(_tmem_load_2[16 + 5]), "f"(_tmem_load_2[16 + 6]), "f"(_tmem_load_2[16 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_2[16 + 8]), "f"(_tmem_load_2[16 + 9]), "f"(_tmem_load_2[16 + 10]), "f"(_tmem_load_2[16 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_2[16 + 12]), "f"(_tmem_load_2[16 + 13]), "f"(_tmem_load_2[16 + 14]), "f"(_tmem_load_2[16 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 80)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_14 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[32])[_ps], _prescale2_14);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_2[32 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_2[32 + 0]), "f"(_tmem_load_2[32 + 1]), "f"(_tmem_load_2[32 + 2]), "f"(_tmem_load_2[32 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_2[32 + 4]), "f"(_tmem_load_2[32 + 5]), "f"(_tmem_load_2[32 + 6]), "f"(_tmem_load_2[32 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_2[32 + 8]), "f"(_tmem_load_2[32 + 9]), "f"(_tmem_load_2[32 + 10]), "f"(_tmem_load_2[32 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_2[32 + 12]), "f"(_tmem_load_2[32 + 13]), "f"(_tmem_load_2[32 + 14]), "f"(_tmem_load_2[32 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 96)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid != 0) {
                                {
                                    const float2 _prescale2_15 = {quant_scale, quant_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[48])[_ps], _prescale2_15);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_2[48 + _ps] *= quant_scale;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_2[48 + 0]), "f"(_tmem_load_2[48 + 1]), "f"(_tmem_load_2[48 + 2]), "f"(_tmem_load_2[48 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_2[48 + 4]), "f"(_tmem_load_2[48 + 5]), "f"(_tmem_load_2[48 + 6]), "f"(_tmem_load_2[48 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_2[48 + 8]), "f"(_tmem_load_2[48 + 9]), "f"(_tmem_load_2[48 + 10]), "f"(_tmem_load_2[48 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_2[48 + 12]), "f"(_tmem_load_2[48 + 13]), "f"(_tmem_load_2[48 + 14]), "f"(_tmem_load_2[48 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base + 112)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                        }
                    }
                    if (output_valid != 0) {
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(row_sum));
                        float partial_lse_value = ((row_sum > 0.0f) ? row_max * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_INF);
                        partial_lse[partial_row] = partial_lse_value;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row] = partial_lse_value;
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr);
            }
        }
    }
    // ---- Role: softmax_odd ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_odd_main
            int group_count_1 = 1;
            if (blockIdx.x < q_group_segment_end_16) {
                if (blockIdx.x < q_group_segment_end_64) {
                    group_count_1 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                } else {
                    group_count_1 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                }
            } else if (blockIdx.x < q_group_segment_end_4) {
                group_count_1 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
            } else {
                group_count_1 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
            }
            int group_count_0_1 = group_count_1;
            int work_idx_1 = blockIdx.x;
            int metadata_base_1 = work_idx_1 * 6;
            int head_kv_1 = scheduler_metadata[metadata_base_1];
            int row_linear_1 = scheduler_metadata[metadata_base_1 + 1];
            int q_begin_1 = scheduler_metadata[metadata_base_1 + 2];
            int q_count_1 = scheduler_metadata[metadata_base_1 + 3];
            int batch_1 = scheduler_metadata[metadata_base_1 + 4];
            int kv_block_1 = scheduler_metadata[metadata_base_1 + 5];
            int row_ptr_base_1 = head_kv_1 * (total_rows + 1) + row_linear_1;
            int row_start_1 = k2q_row_ptr[row_ptr_base_1] + q_begin_1;
            int q_batch_offset_1 = cu_seqlens_q[batch_1];
            int k_batch_offset_1 = cu_seqlens_k[batch_1];
            int kv_len_1 = kv_lens[batch_1];
            if (max_pages == 0) {
                kv_len_1 = cu_seqlens_k[batch_1 + 1] - k_batch_offset_1;
            }
            int query_offset_1 = q_offsets[batch_1];
            if (derive_q_offset != 0) {
                query_offset_1 = kv_len_1 - (cu_seqlens_q[batch_1 + 1] - q_batch_offset_1);
            }
            int stage_warp_1 = warp - 4;
            int my_row_1 = stage_warp_1 * 32 + lane;
            int tmem_row_base_1 = stage_warp_1 * 32 << 16;
            #pragma unroll 1
            for (int stage_iteration_1 = 0; stage_iteration_1 < group_count_0_1 / 2; stage_iteration_1++) {
                int group_1 = stage_iteration_1 * 2 + 1;
                int softmax_phase_1 = stage_iteration_1 & 1;
                mbarrier_wait(s_full_addr + 8, softmax_phase_1);
                int whole_group_valid_1 = 1;
                int token_in_group_1 = 0;
                int edge_in_work_1 = 0;
                int packed_q_1 = 0;
                int q_idx_1 = 0;
                int valid_cols_1 = 0;
                float row_max_1 = -CAKE_INF;
                float score_bias_1 = -CAKE_INF;
                float score_values_1[128];
                int score_base_1 = taddr + 128 + (unsigned int)tmem_row_base_1;
                if (whole_group_valid_1 != 0) {
                    token_in_group_1 = my_row_1 / 16;
                    edge_in_work_1 = group_1 * 8 + token_in_group_1;
                    int row_valid_1 = ((edge_in_work_1 < q_count_1) ? 1 : 0);
                    int owner_lane_1 = lane / 16 * 16;
                    int owned_packed_1 = -1;
                    if (lane == owner_lane_1 && edge_in_work_1 < q_count_1) {
                        owned_packed_1 = k2q_qsplit_indices[head_kv_1 * nnz_per_head + row_start_1 + edge_in_work_1];
                    }
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, owned_packed_1, owner_lane_1);
                    packed_q_1 = _shfl_1;
                    q_idx_1 = packed_q_1 & 16777215;
                    if (row_valid_1 != 0) {
                        valid_cols_1 = kv_len_1 - kv_block_1 * 128;
                        if (valid_cols_1 > 128) {
                            valid_cols_1 = 128;
                        }
                        if (causal != 0) {
                            int query_position_1 = query_offset_1 + q_idx_1;
                            int causal_cols_1 = query_position_1 - kv_block_1 * 128 + 1;
                            if (valid_cols_1 > causal_cols_1) {
                                valid_cols_1 = causal_cols_1;
                            }
                        }
                        if (valid_cols_1 < 0) {
                            valid_cols_1 = 0;
                        }
                    }
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values_1[0]), "=f"(score_values_1[1]), "=f"(score_values_1[2]), "=f"(score_values_1[3]), "=f"(score_values_1[4]), "=f"(score_values_1[5]), "=f"(score_values_1[6]), "=f"(score_values_1[7]), "=f"(score_values_1[8]), "=f"(score_values_1[9]), "=f"(score_values_1[10]), "=f"(score_values_1[11]), "=f"(score_values_1[12]), "=f"(score_values_1[13]), "=f"(score_values_1[14]), "=f"(score_values_1[15]), "=f"(score_values_1[16]), "=f"(score_values_1[17]), "=f"(score_values_1[18]), "=f"(score_values_1[19]), "=f"(score_values_1[20]), "=f"(score_values_1[21]), "=f"(score_values_1[22]), "=f"(score_values_1[23]), "=f"(score_values_1[24]), "=f"(score_values_1[25]), "=f"(score_values_1[26]), "=f"(score_values_1[27]), "=f"(score_values_1[28]), "=f"(score_values_1[29]), "=f"(score_values_1[30]), "=f"(score_values_1[31])
                        : "r"(score_base_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values_1[32]), "=f"(score_values_1[33]), "=f"(score_values_1[34]), "=f"(score_values_1[35]), "=f"(score_values_1[36]), "=f"(score_values_1[37]), "=f"(score_values_1[38]), "=f"(score_values_1[39]), "=f"(score_values_1[40]), "=f"(score_values_1[41]), "=f"(score_values_1[42]), "=f"(score_values_1[43]), "=f"(score_values_1[44]), "=f"(score_values_1[45]), "=f"(score_values_1[46]), "=f"(score_values_1[47]), "=f"(score_values_1[48]), "=f"(score_values_1[49]), "=f"(score_values_1[50]), "=f"(score_values_1[51]), "=f"(score_values_1[52]), "=f"(score_values_1[53]), "=f"(score_values_1[54]), "=f"(score_values_1[55]), "=f"(score_values_1[56]), "=f"(score_values_1[57]), "=f"(score_values_1[58]), "=f"(score_values_1[59]), "=f"(score_values_1[60]), "=f"(score_values_1[61]), "=f"(score_values_1[62]), "=f"(score_values_1[63])
                        : "r"(score_base_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values_1[64]), "=f"(score_values_1[65]), "=f"(score_values_1[66]), "=f"(score_values_1[67]), "=f"(score_values_1[68]), "=f"(score_values_1[69]), "=f"(score_values_1[70]), "=f"(score_values_1[71]), "=f"(score_values_1[72]), "=f"(score_values_1[73]), "=f"(score_values_1[74]), "=f"(score_values_1[75]), "=f"(score_values_1[76]), "=f"(score_values_1[77]), "=f"(score_values_1[78]), "=f"(score_values_1[79]), "=f"(score_values_1[80]), "=f"(score_values_1[81]), "=f"(score_values_1[82]), "=f"(score_values_1[83]), "=f"(score_values_1[84]), "=f"(score_values_1[85]), "=f"(score_values_1[86]), "=f"(score_values_1[87]), "=f"(score_values_1[88]), "=f"(score_values_1[89]), "=f"(score_values_1[90]), "=f"(score_values_1[91]), "=f"(score_values_1[92]), "=f"(score_values_1[93]), "=f"(score_values_1[94]), "=f"(score_values_1[95])
                        : "r"(score_base_1 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(score_values_1[96]), "=f"(score_values_1[97]), "=f"(score_values_1[98]), "=f"(score_values_1[99]), "=f"(score_values_1[100]), "=f"(score_values_1[101]), "=f"(score_values_1[102]), "=f"(score_values_1[103]), "=f"(score_values_1[104]), "=f"(score_values_1[105]), "=f"(score_values_1[106]), "=f"(score_values_1[107]), "=f"(score_values_1[108]), "=f"(score_values_1[109]), "=f"(score_values_1[110]), "=f"(score_values_1[111]), "=f"(score_values_1[112]), "=f"(score_values_1[113]), "=f"(score_values_1[114]), "=f"(score_values_1[115]), "=f"(score_values_1[116]), "=f"(score_values_1[117]), "=f"(score_values_1[118]), "=f"(score_values_1[119]), "=f"(score_values_1[120]), "=f"(score_values_1[121]), "=f"(score_values_1[122]), "=f"(score_values_1[123]), "=f"(score_values_1[124]), "=f"(score_values_1[125]), "=f"(score_values_1[126]), "=f"(score_values_1[127])
                        : "r"(score_base_1 + 96));
                    if (valid_cols_1 < 128) {
                        uint32_t _slice_lo_mask_4;
                        {
                            int _lim_0 = valid_cols_1;
                            if (_lim_0 <= 0) { _slice_lo_mask_4 = 0u; }
                            else if (_lim_0 >= 32) { _slice_lo_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_0));
                            }
                        }
                        if (!(_slice_lo_mask_4 & (1u << 0))) score_values_1[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 1))) score_values_1[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 2))) score_values_1[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 3))) score_values_1[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 4))) score_values_1[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 5))) score_values_1[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 6))) score_values_1[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 7))) score_values_1[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 8))) score_values_1[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 9))) score_values_1[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 10))) score_values_1[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 11))) score_values_1[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 12))) score_values_1[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 13))) score_values_1[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 14))) score_values_1[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 15))) score_values_1[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 16))) score_values_1[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 17))) score_values_1[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 18))) score_values_1[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 19))) score_values_1[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 20))) score_values_1[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 21))) score_values_1[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 22))) score_values_1[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 23))) score_values_1[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 24))) score_values_1[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 25))) score_values_1[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 26))) score_values_1[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 27))) score_values_1[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 28))) score_values_1[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 29))) score_values_1[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 30))) score_values_1[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_4 & (1u << 31))) score_values_1[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_5;
                        {
                            int _lim_1 = valid_cols_1 - 32;
                            if (_lim_1 <= 0) { _slice_lo_mask_5 = 0u; }
                            else if (_lim_1 >= 32) { _slice_lo_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_1));
                            }
                        }
                        if (!(_slice_lo_mask_5 & (1u << 0))) score_values_1[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 1))) score_values_1[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 2))) score_values_1[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 3))) score_values_1[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 4))) score_values_1[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 5))) score_values_1[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 6))) score_values_1[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 7))) score_values_1[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 8))) score_values_1[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 9))) score_values_1[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 10))) score_values_1[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 11))) score_values_1[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 12))) score_values_1[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 13))) score_values_1[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 14))) score_values_1[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 15))) score_values_1[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 16))) score_values_1[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 17))) score_values_1[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 18))) score_values_1[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 19))) score_values_1[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 20))) score_values_1[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 21))) score_values_1[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 22))) score_values_1[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 23))) score_values_1[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 24))) score_values_1[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 25))) score_values_1[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 26))) score_values_1[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 27))) score_values_1[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 28))) score_values_1[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 29))) score_values_1[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 30))) score_values_1[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_5 & (1u << 31))) score_values_1[63] = -CAKE_INF;
                        uint32_t _slice_lo_mask_6;
                        {
                            int _lim_2 = valid_cols_1 - 64;
                            if (_lim_2 <= 0) { _slice_lo_mask_6 = 0u; }
                            else if (_lim_2 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_2));
                            }
                        }
                        if (!(_slice_lo_mask_6 & (1u << 0))) score_values_1[64] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 1))) score_values_1[65] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 2))) score_values_1[66] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 3))) score_values_1[67] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 4))) score_values_1[68] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 5))) score_values_1[69] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 6))) score_values_1[70] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 7))) score_values_1[71] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 8))) score_values_1[72] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 9))) score_values_1[73] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 10))) score_values_1[74] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 11))) score_values_1[75] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 12))) score_values_1[76] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 13))) score_values_1[77] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 14))) score_values_1[78] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 15))) score_values_1[79] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 16))) score_values_1[80] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 17))) score_values_1[81] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 18))) score_values_1[82] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 19))) score_values_1[83] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 20))) score_values_1[84] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 21))) score_values_1[85] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 22))) score_values_1[86] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 23))) score_values_1[87] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 24))) score_values_1[88] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 25))) score_values_1[89] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 26))) score_values_1[90] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 27))) score_values_1[91] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 28))) score_values_1[92] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 29))) score_values_1[93] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 30))) score_values_1[94] = -CAKE_INF;
                        if (!(_slice_lo_mask_6 & (1u << 31))) score_values_1[95] = -CAKE_INF;
                        uint32_t _slice_lo_mask_7;
                        {
                            int _lim_3 = valid_cols_1 - 96;
                            if (_lim_3 <= 0) { _slice_lo_mask_7 = 0u; }
                            else if (_lim_3 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_3));
                            }
                        }
                        if (!(_slice_lo_mask_7 & (1u << 0))) score_values_1[96] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 1))) score_values_1[97] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 2))) score_values_1[98] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 3))) score_values_1[99] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 4))) score_values_1[100] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 5))) score_values_1[101] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 6))) score_values_1[102] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 7))) score_values_1[103] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 8))) score_values_1[104] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 9))) score_values_1[105] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 10))) score_values_1[106] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 11))) score_values_1[107] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 12))) score_values_1[108] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 13))) score_values_1[109] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 14))) score_values_1[110] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 15))) score_values_1[111] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 16))) score_values_1[112] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 17))) score_values_1[113] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 18))) score_values_1[114] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 19))) score_values_1[115] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 20))) score_values_1[116] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 21))) score_values_1[117] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 22))) score_values_1[118] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 23))) score_values_1[119] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 24))) score_values_1[120] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 25))) score_values_1[121] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 26))) score_values_1[122] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 27))) score_values_1[123] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 28))) score_values_1[124] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 29))) score_values_1[125] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 30))) score_values_1[126] = -CAKE_INF;
                        if (!(_slice_lo_mask_7 & (1u << 31))) score_values_1[127] = -CAKE_INF;
                    }
                    float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&score_values_1[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values_1[32], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values_1[64], _reg_reduce_max2_4);
                    row_max_x32_accum(&score_values_1[96], _reg_reduce_max2_4);
                    float score_values_max_1 = row_max_reduce(_reg_reduce_max2_4);
                    row_max_1 = score_values_max_1;
                    float safe_max_1 = ((row_max_1 == -CAKE_INF) ? 0.0f : row_max_1);
                    score_bias_1 = ((valid_cols_1 > 0) ? (-safe_max_1) * softmax_scale_log2 : -CAKE_INF);
                }
                mbarrier_wait(p_empty_addr + 8, softmax_phase_1 ^ 1);
                float row_sum_1 = 0.0f;
                if (whole_group_valid_1 != 0) {
                    int p_base_1 = taddr + 128 + 64 + (unsigned int)tmem_row_base_1;
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values_1)[_lf], _fma_b2_5, _fma_c2_6);
                    for (int probability_segment_1 = 0; probability_segment_1 < 2; probability_segment_1++) {
                        uint32_t score_values_bf16_2[16];
                        softmax_frag_exp2_cast(&(score_values_1 + probability_segment_1 * 32)[0], score_values_bf16_2, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base_1 + probability_segment_1 * 16), "r"(score_values_bf16_2[0]), "r"(score_values_bf16_2[1]), "r"(score_values_bf16_2[2]), "r"(score_values_bf16_2[3]), "r"(score_values_bf16_2[4]), "r"(score_values_bf16_2[5]), "r"(score_values_bf16_2[6]), "r"(score_values_bf16_2[7]), "r"(score_values_bf16_2[8]), "r"(score_values_bf16_2[9]), "r"(score_values_bf16_2[10]), "r"(score_values_bf16_2[11]), "r"(score_values_bf16_2[12]), "r"(score_values_bf16_2[13]), "r"(score_values_bf16_2[14]), "r"(score_values_bf16_2[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(p_full_addr + 8);
                    uint32_t score_values_bf16_3[32];
                    softmax_frag_exp2_cast(&score_values_1[64], score_values_bf16_3, 0);
                    softmax_frag_exp2_cast(&score_values_1[96], &score_values_bf16_3[16], 0);
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base_1 + 32), "r"(score_values_bf16_3[0]), "r"(score_values_bf16_3[1]), "r"(score_values_bf16_3[2]), "r"(score_values_bf16_3[3]), "r"(score_values_bf16_3[4]), "r"(score_values_bf16_3[5]), "r"(score_values_bf16_3[6]), "r"(score_values_bf16_3[7]), "r"(score_values_bf16_3[8]), "r"(score_values_bf16_3[9]), "r"(score_values_bf16_3[10]), "r"(score_values_bf16_3[11]), "r"(score_values_bf16_3[12]), "r"(score_values_bf16_3[13]), "r"(score_values_bf16_3[14]), "r"(score_values_bf16_3[15]), "r"(score_values_bf16_3[16]), "r"(score_values_bf16_3[17]), "r"(score_values_bf16_3[18]), "r"(score_values_bf16_3[19]), "r"(score_values_bf16_3[20]), "r"(score_values_bf16_3[21]), "r"(score_values_bf16_3[22]), "r"(score_values_bf16_3[23]), "r"(score_values_bf16_3[24]), "r"(score_values_bf16_3[25]), "r"(score_values_bf16_3[26]), "r"(score_values_bf16_3[27]), "r"(score_values_bf16_3[28]), "r"(score_values_bf16_3[29]), "r"(score_values_bf16_3[30]), "r"(score_values_bf16_3[31]));
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values_1[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values_1[32], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values_1[64], &_reg_reduce_sum2_7);
                    softmax_block_sum(&score_values_1[96], &_reg_reduce_sum2_7);
                    float score_values_sum_1 = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    row_sum_1 = score_values_sum_1;
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(p_full_2_addr + 8);
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(s_empty_addr + 8);
                mbarrier_wait(o_full_addr + 8, softmax_phase_1);
                if (whole_group_valid_1 != 0) {
                    int q_head_local_1 = my_row_1 - token_in_group_1 * 16;
                    int output_valid_1 = 0;
                    long long partial_row_1 = 0;
                    long long final_row_1 = 0;
                    float inv_sum_1 = 0.0f;
                    int single_split_1 = 0;
                    if (edge_in_work_1 < q_count_1) {
                        int split_slot_1 = packed_q_1 >> 24 & 15;
                        if (split_slot_1 >= 0 && split_slot_1 < topk) {
                            output_valid_1 = 1;
                            int q_abs_1 = q_batch_offset_1 + q_idx_1;
                            int q_head_1 = head_kv_1 * 16 + q_head_local_1;
                            final_row_1 = (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            partial_row_1 = (long long)split_slot_1 * (long long)total_q * (long long)num_q_heads + (long long)q_abs_1 * (long long)num_q_heads + (long long)q_head_1;
                            float _rcp_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_1 : 0.0f);
                        }
                    }
                    long long partial_base_1 = partial_row_1 * 128;
                    {
                        int output_row_addr_1 = taddr + (unsigned int)TMEM_OUTPUT_OFFSET + 128 + (unsigned int)tmem_row_base_1;
                        long long partial_metadata_rows_1 = (long long)topk * (long long)total_q * (long long)num_q_heads;
                        {
                            float row_min_1 = CAKE_INF;
                            float row_max_0_1 = -CAKE_INF;
                            #pragma unroll 1
                            for (int output_segment_2 = 0; output_segment_2 < 4; output_segment_2++) {
                                float _tmem_load_6[16];
                                tmem_ld_x16(&_tmem_load_6[0], output_row_addr_1 + output_segment_2 * 16);
                                float _tmem_load_6_min = _tmem_load_6[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_6_min = fminf(_tmem_load_6_min, _tmem_load_6[_lr]);
                                }
                                float _min_2 = fminf(row_min_1, _tmem_load_6_min);
                                row_min_1 = _min_2;
                                float _tmem_load_6_max = _tmem_load_6[0];
                                #pragma unroll
                                for (int _lr = 1; _lr < 16; _lr++) {
                                    _tmem_load_6_max = max_noftz(_tmem_load_6_max, _tmem_load_6[_lr]);
                                }
                                float _max_2 = max_noftz(row_max_0_1, _tmem_load_6_max);
                                row_max_0_1 = _max_2;
                            }
                            float _tmem_load_7[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                                : "r"(output_row_addr_1 + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_7[32]), "=f"(_tmem_load_7[33]), "=f"(_tmem_load_7[34]), "=f"(_tmem_load_7[35]), "=f"(_tmem_load_7[36]), "=f"(_tmem_load_7[37]), "=f"(_tmem_load_7[38]), "=f"(_tmem_load_7[39]), "=f"(_tmem_load_7[40]), "=f"(_tmem_load_7[41]), "=f"(_tmem_load_7[42]), "=f"(_tmem_load_7[43]), "=f"(_tmem_load_7[44]), "=f"(_tmem_load_7[45]), "=f"(_tmem_load_7[46]), "=f"(_tmem_load_7[47]), "=f"(_tmem_load_7[48]), "=f"(_tmem_load_7[49]), "=f"(_tmem_load_7[50]), "=f"(_tmem_load_7[51]), "=f"(_tmem_load_7[52]), "=f"(_tmem_load_7[53]), "=f"(_tmem_load_7[54]), "=f"(_tmem_load_7[55]), "=f"(_tmem_load_7[56]), "=f"(_tmem_load_7[57]), "=f"(_tmem_load_7[58]), "=f"(_tmem_load_7[59]), "=f"(_tmem_load_7[60]), "=f"(_tmem_load_7[61]), "=f"(_tmem_load_7[62]), "=f"(_tmem_load_7[63])
                                : "r"(output_row_addr_1 + 64 + 32));
                            float _tmem_load_7_min = _tmem_load_7[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 64; _lr++) {
                                _tmem_load_7_min = fminf(_tmem_load_7_min, _tmem_load_7[_lr]);
                            }
                            float _min_3 = fminf(row_min_1, _tmem_load_7_min);
                            row_min_1 = _min_3;
                            float2 _reg_reduce_max2_8 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&_tmem_load_7[0], _reg_reduce_max2_8);
                            row_max_x32_accum(&_tmem_load_7[32], _reg_reduce_max2_8);
                            float _tmem_load_7_max = row_max_reduce(_reg_reduce_max2_8);
                            float _max_3 = max_noftz(row_max_0_1, _tmem_load_7_max);
                            row_max_0_1 = _max_3;
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float row_center_1 = (row_max_0_1 + row_min_1) * 0.5f;
                            float residual_abs_max_1 = (row_max_0_1 - row_min_1) * 0.5f;
                            float dequant_scale_1 = 0.0f;
                            float quant_scale_1 = 0.0f;
                            if (residual_abs_max_1 > 0.0f && residual_abs_max_1 == residual_abs_max_1) {
                                dequant_scale_1 = residual_abs_max_1 * inv_sum_1 * 0.002232142857142857f;
                                quant_scale_1 = 448.0f / residual_abs_max_1;
                            }
                            if (output_valid_1 != 0) {
                                partial_scale[partial_row_1] = dequant_scale_1;
                                partial_scale[partial_metadata_rows_1 + partial_row_1] = row_center_1 * inv_sum_1;
                            }
                            #pragma unroll 1
                            for (int output_segment_3 = 0; output_segment_3 < 4; output_segment_3++) {
                                float _tmem_load_8[16];
                                tmem_ld_x16(&_tmem_load_8[0], output_row_addr_1 + output_segment_3 * 16);
                                const float2 _sub2_9 = {row_center_1, row_center_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _sub2_9);
                                if (output_valid_1 != 0) {
                                    {
                                        const float2 _prescale2_10 = {quant_scale_1, quant_scale_1};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_8[0])[_ps], _prescale2_10);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 16; _ps++)
                                            _tmem_load_8[0 + _ps] *= quant_scale_1;
                                        #endif
                                        unsigned int _fp8_pk[4];
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[0]) : "f"(_tmem_load_8[0 + 0]), "f"(_tmem_load_8[0 + 1]), "f"(_tmem_load_8[0 + 2]), "f"(_tmem_load_8[0 + 3]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[1]) : "f"(_tmem_load_8[0 + 4]), "f"(_tmem_load_8[0 + 5]), "f"(_tmem_load_8[0 + 6]), "f"(_tmem_load_8[0 + 7]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[2]) : "f"(_tmem_load_8[0 + 8]), "f"(_tmem_load_8[0 + 9]), "f"(_tmem_load_8[0 + 10]), "f"(_tmem_load_8[0 + 11]));
                                        asm("{\n\t"
                                            ".reg .b16 _lo, _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}\n"
                                            : "=r"(_fp8_pk[3]) : "f"(_tmem_load_8[0 + 12]), "f"(_tmem_load_8[0 + 13]), "f"(_tmem_load_8[0 + 14]), "f"(_tmem_load_8[0 + 15]));
                                        *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + (long long)output_segment_3 * 16)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                    }
                                }
                            }
                            const float2 _sub2_11 = {row_center_1, row_center_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                sub_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _sub2_11);
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_12 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[0])[_ps], _prescale2_12);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_7[0 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_7[0 + 0]), "f"(_tmem_load_7[0 + 1]), "f"(_tmem_load_7[0 + 2]), "f"(_tmem_load_7[0 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_7[0 + 4]), "f"(_tmem_load_7[0 + 5]), "f"(_tmem_load_7[0 + 6]), "f"(_tmem_load_7[0 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_7[0 + 8]), "f"(_tmem_load_7[0 + 9]), "f"(_tmem_load_7[0 + 10]), "f"(_tmem_load_7[0 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_7[0 + 12]), "f"(_tmem_load_7[0 + 13]), "f"(_tmem_load_7[0 + 14]), "f"(_tmem_load_7[0 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 64)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_13 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[16])[_ps], _prescale2_13);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_7[16 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_7[16 + 0]), "f"(_tmem_load_7[16 + 1]), "f"(_tmem_load_7[16 + 2]), "f"(_tmem_load_7[16 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_7[16 + 4]), "f"(_tmem_load_7[16 + 5]), "f"(_tmem_load_7[16 + 6]), "f"(_tmem_load_7[16 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_7[16 + 8]), "f"(_tmem_load_7[16 + 9]), "f"(_tmem_load_7[16 + 10]), "f"(_tmem_load_7[16 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_7[16 + 12]), "f"(_tmem_load_7[16 + 13]), "f"(_tmem_load_7[16 + 14]), "f"(_tmem_load_7[16 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 80)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_14 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[32])[_ps], _prescale2_14);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_7[32 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_7[32 + 0]), "f"(_tmem_load_7[32 + 1]), "f"(_tmem_load_7[32 + 2]), "f"(_tmem_load_7[32 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_7[32 + 4]), "f"(_tmem_load_7[32 + 5]), "f"(_tmem_load_7[32 + 6]), "f"(_tmem_load_7[32 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_7[32 + 8]), "f"(_tmem_load_7[32 + 9]), "f"(_tmem_load_7[32 + 10]), "f"(_tmem_load_7[32 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_7[32 + 12]), "f"(_tmem_load_7[32 + 13]), "f"(_tmem_load_7[32 + 14]), "f"(_tmem_load_7[32 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 96)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                            if (output_valid_1 != 0) {
                                {
                                    const float2 _prescale2_15 = {quant_scale_1, quant_scale_1};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[48])[_ps], _prescale2_15);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 16; _ps++)
                                        _tmem_load_7[48 + _ps] *= quant_scale_1;
                                    #endif
                                    unsigned int _fp8_pk[4];
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[0]) : "f"(_tmem_load_7[48 + 0]), "f"(_tmem_load_7[48 + 1]), "f"(_tmem_load_7[48 + 2]), "f"(_tmem_load_7[48 + 3]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[1]) : "f"(_tmem_load_7[48 + 4]), "f"(_tmem_load_7[48 + 5]), "f"(_tmem_load_7[48 + 6]), "f"(_tmem_load_7[48 + 7]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[2]) : "f"(_tmem_load_7[48 + 8]), "f"(_tmem_load_7[48 + 9]), "f"(_tmem_load_7[48 + 10]), "f"(_tmem_load_7[48 + 11]));
                                    asm("{\n\t"
                                        ".reg .b16 _lo, _hi;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                        "mov.b32 %0, {_lo, _hi};\n\t"
                                        "}\n"
                                        : "=r"(_fp8_pk[3]) : "f"(_tmem_load_7[48 + 12]), "f"(_tmem_load_7[48 + 13]), "f"(_tmem_load_7[48 + 14]), "f"(_tmem_load_7[48 + 15]));
                                    *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(partial_o + (partial_base_1 + 112)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                                }
                            }
                        }
                    }
                    if (output_valid_1 != 0) {
                        float _log2_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(row_sum_1));
                        float partial_lse_value_1 = ((row_sum_1 > 0.0f) ? row_max_1 * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f : -CAKE_INF);
                        partial_lse[partial_row_1] = partial_lse_value_1;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row_1] = partial_lse_value_1;
                        }
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                mbarrier_arrive(o_empty_addr + 8);
            }
        }
    }
    // ---- Role: qload ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // qload_main
            int work_idx_2 = blockIdx.x;
            int metadata_base_2 = work_idx_2 * 6;
            int head_kv_2 = scheduler_metadata[metadata_base_2];
            int row_linear_2 = scheduler_metadata[metadata_base_2 + 1];
            int q_begin_2 = scheduler_metadata[metadata_base_2 + 2];
            int q_count_2 = scheduler_metadata[metadata_base_2 + 3];
            int batch_2 = scheduler_metadata[metadata_base_2 + 4];
            int kv_block_2 = scheduler_metadata[metadata_base_2 + 5];
            int row_ptr_base_2 = head_kv_2 * (total_rows + 1) + row_linear_2;
            int row_start_2 = k2q_row_ptr[row_ptr_base_2] + q_begin_2;
            int q_batch_offset_2 = cu_seqlens_q[batch_2];
            int k_batch_offset_2 = cu_seqlens_k[batch_2];
            int kv_len_2 = kv_lens[batch_2];
            if (max_pages == 0) {
                kv_len_2 = cu_seqlens_k[batch_2 + 1] - k_batch_offset_2;
            }
            int query_offset_2 = q_offsets[batch_2];
            if (derive_q_offset != 0) {
                query_offset_2 = kv_len_2 - (cu_seqlens_q[batch_2 + 1] - q_batch_offset_2);
            }
            int group_count_2 = 1;
            if (blockIdx.x < q_group_segment_end_16) {
                if (blockIdx.x < q_group_segment_end_64) {
                    group_count_2 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                } else {
                    group_count_2 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                }
            } else if (blockIdx.x < q_group_segment_end_4) {
                group_count_2 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
            } else {
                group_count_2 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
            }
            int group_count_0_2 = group_count_2;
            #pragma unroll 1
            for (int group_2 = 0; group_2 < group_count_0_2; group_2++) {
                int q_stage = group_2 & 1;
                int q_phase = group_2 / 2 & 1;
                mbarrier_wait(q_empty_addr + (q_stage) * 8, q_phase ^ 1);
                int q_stage_addr = q_store_smem_addr + (unsigned int)(q_stage * 32768);
                {
                    int qload_warp = warp - 8;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_stage) * 8, 8192);
                        int token_in_group_2 = qload_warp * 2;
                        int edge_in_work_2 = group_2 * 8 + token_in_group_2;
                        int edge_valid = ((edge_in_work_2 < q_count_2) ? 1 : 0);
                        int safe_edge = ((edge_valid != 0) ? edge_in_work_2 : 0);
                        int packed_q_2 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge];
                        int decoded_q_abs = q_batch_offset_2 + (packed_q_2 & 16777215);
                        int q_abs_2 = ((edge_valid != 0) ? decoded_q_abs : 0);
                        int dst_offset = token_in_group_2 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset, (&q), 0, head_kv_2 * 16, 0, q_abs_2, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset, (&q), 0, head_kv_2 * 16, 1, q_abs_2, q_full_addr + (q_stage) * 8);
                        int token_in_group_0 = qload_warp * 2 + 1;
                        int edge_in_work_1_1 = group_2 * 8 + token_in_group_0;
                        int edge_valid_2 = ((edge_in_work_1_1 < q_count_2) ? 1 : 0);
                        int safe_edge_3 = ((edge_valid_2 != 0) ? edge_in_work_1_1 : 0);
                        int packed_q_4 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_3];
                        int decoded_q_abs_5 = q_batch_offset_2 + (packed_q_4 & 16777215);
                        int q_abs_6 = ((edge_valid_2 != 0) ? decoded_q_abs_5 : 0);
                        int dst_offset_7 = token_in_group_0 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset_7, (&q), 0, head_kv_2 * 16, 0, q_abs_6, q_full_addr + (q_stage) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset_7, (&q), 0, head_kv_2 * 16, 1, q_abs_6, q_full_addr + (q_stage) * 8);
                    }
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int work_idx_3 = blockIdx.x;
            int metadata_base_3 = work_idx_3 * 6;
            int head_kv_3 = scheduler_metadata[metadata_base_3];
            int row_linear_3 = scheduler_metadata[metadata_base_3 + 1];
            int q_begin_3 = scheduler_metadata[metadata_base_3 + 2];
            int q_count_3 = scheduler_metadata[metadata_base_3 + 3];
            int batch_3 = scheduler_metadata[metadata_base_3 + 4];
            int kv_block_3 = scheduler_metadata[metadata_base_3 + 5];
            int row_ptr_base_3 = head_kv_3 * (total_rows + 1) + row_linear_3;
            int row_start_3 = k2q_row_ptr[row_ptr_base_3] + q_begin_3;
            int q_batch_offset_3 = cu_seqlens_q[batch_3];
            int k_batch_offset_3 = cu_seqlens_k[batch_3];
            int kv_len_3 = kv_lens[batch_3];
            if (max_pages == 0) {
                kv_len_3 = cu_seqlens_k[batch_3 + 1] - k_batch_offset_3;
            }
            int query_offset_3 = q_offsets[batch_3];
            if (derive_q_offset != 0) {
                query_offset_3 = kv_len_3 - (cu_seqlens_q[batch_3 + 1] - q_batch_offset_3);
            }
            int group_count_3 = 1;
            if (blockIdx.x < q_group_segment_end_16) {
                if (blockIdx.x < q_group_segment_end_64) {
                    group_count_3 = ((blockIdx.x < q_group_segment_end_128) ? 128 : 64);
                } else {
                    group_count_3 = ((blockIdx.x < q_group_segment_end_32) ? 32 : 16);
                }
            } else if (blockIdx.x < q_group_segment_end_4) {
                group_count_3 = ((blockIdx.x < q_group_segment_end_8) ? 8 : 4);
            } else {
                group_count_3 = ((blockIdx.x < q_group_segment_end_2) ? 2 : 1);
            }
            int group_count_0_3 = group_count_3;
            unsigned int _phase_k_full_0 = 0;
            mbarrier_wait(k_full_addr, _phase_k_full_0);
            _phase_k_full_0 ^= 1;
            int q_stage_1 = 0;
            int q_phase_1 = 0;
            mbarrier_wait(q_full_addr + (q_stage_1) * 8, q_phase_1);
            mbarrier_wait(s_empty_addr + (q_stage_1) * 8, q_phase_1 ^ 1);
            int _mma_a_lo_0 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_1) * 2048);
            int _mma_b_lo_0 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f16((tmem_scores + (q_stage_1 * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136316048, 1);
                }
            }
            elect_commit(s_full_addr + (q_stage_1) * 8);
            elect_commit(q_empty_addr + (q_stage_1) * 8);
            if (group_count_0_3 > 1) {
                int q_stage_0 = 1;
                int q_phase_1_1 = 0;
                mbarrier_wait(q_full_addr + (q_stage_0) * 8, q_phase_1_1);
                mbarrier_wait(s_empty_addr + (q_stage_0) * 8, q_phase_1_1 ^ 1);
                int _mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_0) * 2048);
                int _mma_b_lo_1 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0 * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136316048, 1);
                    }
                }
                elect_commit(s_full_addr + (q_stage_0) * 8);
                elect_commit(q_empty_addr + (q_stage_0) * 8);
            }
            unsigned int _phase_v_full_0 = 0;
            mbarrier_wait(v_full_addr, _phase_v_full_0);
            _phase_v_full_0 ^= 1;
            #pragma unroll 1
            for (int group_3 = 2; group_3 < group_count_0_3; group_3++) {
                int pv_group = group_3 - 2;
                int pv_stage = pv_group & 1;
                int pv_phase = pv_group / 2 & 1;
                mbarrier_wait(p_full_addr + (pv_stage) * 8, pv_phase);
                mbarrier_wait(o_empty_addr + (pv_stage) * 8, pv_phase ^ 1);
                int _mma_b_lo_2 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(0));
                mbarrier_wait(p_full_2_addr + (pv_stage) * 8, pv_phase);
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(_mma_b_lo_2), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(1));
                elect_commit(o_full_addr + (pv_stage) * 8);
                elect_commit(p_empty_addr + (pv_stage) * 8);
                int q_stage_0_1 = group_3 & 1;
                int q_phase_1_2 = group_3 / 2 & 1;
                mbarrier_wait(q_full_addr + (q_stage_0_1) * 8, q_phase_1_2);
                mbarrier_wait(s_empty_addr + (q_stage_0_1) * 8, q_phase_1_2 ^ 1);
                int _mma_a_lo_4 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_0_1) * 2048);
                int _mma_b_lo_4 = make_warp_uniform(((k_smem_addr) >> 4) & 0x3FFF);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 1018U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f16((tmem_scores + (q_stage_0_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                    }
                }
                elect_commit(s_full_addr + (q_stage_0_1) * 8);
                elect_commit(q_empty_addr + (q_stage_0_1) * 8);
            }
            int drain_start = ((group_count_0_3 == 1) ? 0 : group_count_0_3 - 2);
            #pragma unroll 1
            for (int pv_group_1 = drain_start; pv_group_1 < group_count_0_3; pv_group_1++) {
                int pv_stage_1 = pv_group_1 & 1;
                int pv_phase_1 = pv_group_1 / 2 & 1;
                mbarrier_wait(p_full_addr + (pv_stage_1) * 8, pv_phase_1);
                mbarrier_wait(o_empty_addr + (pv_stage_1) * 8, pv_phase_1 ^ 1);
                int _mma_b_lo_5 = make_warp_uniform((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000);
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
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_5), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(0));
                mbarrier_wait(p_full_2_addr + (pv_stage_1) * 8, pv_phase_1);
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
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(_mma_b_lo_5), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(1));
                elect_commit(o_full_addr + (pv_stage_1) * 8);
                elect_commit(p_empty_addr + (pv_stage_1) * 8);
            }
            #pragma unroll 1
            for (int completed_group = drain_start; completed_group < group_count_0_3; completed_group++) {
                int completed_stage = completed_group & 1;
                int completed_phase = completed_group / 2 & 1;
                mbarrier_wait(o_empty_addr + (completed_stage) * 8, completed_phase);
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: transform ----
    if (warp >= 13 && warp <= 14) {
        { // transform_main
            int work_idx_4 = blockIdx.x;
            int metadata_base_4 = work_idx_4 * 6;
            int head_kv_4 = scheduler_metadata[metadata_base_4];
            int row_linear_4 = scheduler_metadata[metadata_base_4 + 1];
            int q_begin_4 = scheduler_metadata[metadata_base_4 + 2];
            int q_count_4 = scheduler_metadata[metadata_base_4 + 3];
            int batch_4 = scheduler_metadata[metadata_base_4 + 4];
            int kv_block_4 = scheduler_metadata[metadata_base_4 + 5];
            int row_ptr_base_4 = head_kv_4 * (total_rows + 1) + row_linear_4;
            int row_start_4 = k2q_row_ptr[row_ptr_base_4] + q_begin_4;
            int q_batch_offset_4 = cu_seqlens_q[batch_4];
            int k_batch_offset_4 = cu_seqlens_k[batch_4];
            int kv_len_4 = kv_lens[batch_4];
            if (max_pages == 0) {
                kv_len_4 = cu_seqlens_k[batch_4 + 1] - k_batch_offset_4;
            }
            int query_offset_4 = q_offsets[batch_4];
            if (derive_q_offset != 0) {
                query_offset_4 = kv_len_4 - (cu_seqlens_q[batch_4 + 1] - q_batch_offset_4);
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            int work_idx_5 = blockIdx.x;
            int metadata_base_5 = work_idx_5 * 6;
            int head_kv_5 = scheduler_metadata[metadata_base_5];
            int row_linear_5 = scheduler_metadata[metadata_base_5 + 1];
            int q_begin_5 = scheduler_metadata[metadata_base_5 + 2];
            int q_count_5 = scheduler_metadata[metadata_base_5 + 3];
            int batch_5 = scheduler_metadata[metadata_base_5 + 4];
            int kv_block_5 = scheduler_metadata[metadata_base_5 + 5];
            int row_ptr_base_5 = head_kv_5 * (total_rows + 1) + row_linear_5;
            int row_start_5 = k2q_row_ptr[row_ptr_base_5] + q_begin_5;
            int q_batch_offset_5 = cu_seqlens_q[batch_5];
            int k_batch_offset_5 = cu_seqlens_k[batch_5];
            int kv_len_5 = kv_lens[batch_5];
            if (max_pages == 0) {
                kv_len_5 = cu_seqlens_k[batch_5 + 1] - k_batch_offset_5;
            }
            int query_offset_5 = q_offsets[batch_5];
            if (derive_q_offset != 0) {
                query_offset_5 = kv_len_5 - (cu_seqlens_q[batch_5 + 1] - q_batch_offset_5);
            }
            int token_base = k_batch_offset_5 + kv_block_5 * 128;
            int page_head = head_kv_5;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(k_full_addr, 32768);
                int token0 = token_base;
                int token1 = token_base + 64;
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr), "l"((&k)), "r"(0), "r"(token0), "r"(0), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 8192), "l"((&k)), "r"(0), "r"(token1), "r"(0), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 16384), "l"((&k)), "r"(0), "r"(token0), "r"(1), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(k_smem_addr + 24576), "l"((&k)), "r"(0), "r"(token1), "r"(1), "r"(page_head),
                       "r"(k_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                mbarrier_arrive_expect_tx(v_full_addr, 32768);
                int token0_0 = token_base;
                int token1_1 = token_base + 64;
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr), "l"((&v)), "r"(0), "r"(token0_0), "r"(0), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 8192), "l"((&v)), "r"(0), "r"(token1_1), "r"(0), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 16384), "l"((&v)), "r"(0), "r"(token0_0), "r"(1), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                    :: "r"(v_smem_addr + 24576), "l"((&v)), "r"(0), "r"(token1_1), "r"(1), "r"(page_head),
                       "r"(v_full_addr), "l"(0x12F0000000000000ULL) : "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
