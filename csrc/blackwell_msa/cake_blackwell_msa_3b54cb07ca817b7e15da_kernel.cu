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
        asm("add.rn.ftz.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_blackwell_msa_3b54cb07ca817b7e15da(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, int* __restrict__ scheduler_metadata, int* __restrict__ k2q_row_ptr, int* __restrict__ k2q_qsplit_indices, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, float* __restrict__ partial_temperature_lse, int* __restrict__ cu_seqlens_q, int* __restrict__ cu_seqlens_k, int* __restrict__ q_offsets, int* __restrict__ kv_lens, int* __restrict__ page_table, int total_q, int num_q_heads, int num_kv_heads, int total_rows, int nnz_per_head, int num_work_items, int topk, int max_pages, int causal, int derive_q_offset, float softmax_scale_log2, float lse_temperature_scale, int return_temperature_lse, int* __restrict__ split_counts, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, float* __restrict__ temperature_lse, unsigned int* __restrict__ grid_barrier, int return_softmax_lse)
{
    // Grid-dependency payloads: written while the kernel runs, read with coherent loads only.
    asm volatile("" : "+l"(partial_o));
    __builtin_assume(__isGlobal(partial_o));
    asm volatile("" : "+l"(partial_lse));
    __builtin_assume(__isGlobal(partial_lse));

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
    #define v_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 64)
    #define s_full_addr (mbar_base + 80)
    #define s_empty_addr (mbar_base + 96)
    #define p_full_addr (mbar_base + 112)
    #define p_full_2_addr (mbar_base + 128)
    #define p_empty_addr (mbar_base + 144)
    #define o_full_addr (mbar_base + 160)
    #define o_empty_addr (mbar_base + 176)

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
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int v_smem_addr = smem + 132096;
    uint8_t* partial_stage = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int partial_stage_addr = smem + 197632;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 24 barriers)
    // Mbarriers at smem_raw[0..192)

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
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 96, 128);
            mbarrier_init(smem + 104, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            mbarrier_init(smem + 120, 128);
            // p_full_2: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // o_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 192);
    if (warp == 0) {
        int _tmem_hold = smem + 192;
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
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax_even ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_even_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int my_row = warp * 32 + lane;
            int tmem_row_base = warp * 32 << 16;
            int persist_work = num_work_items;
            int persist_grid = gridDim.x;
            int persist_bid = blockIdx.x;
            int persist_waves = persist_work / persist_grid;
            int persist_tail = persist_work - persist_waves * persist_grid;
            int persist_items = persist_waves;
            if ((persist_waves & 1) == 0) {
                if (persist_bid < persist_tail) {
                    persist_items = persist_items + 1;
                }
            } else if (persist_tail > persist_grid - 1 - persist_bid) {
                persist_items = persist_items + 1;
            }
            int groups_done = 0;
            int stage_groups = 0;
            #pragma unroll 1
            for (int persist_iter = 0; persist_iter < persist_items; persist_iter++) {
                int persist_lane = persist_bid;
                if ((persist_iter & 1) != 0) {
                    persist_lane = persist_grid - 1 - persist_bid;
                }
                int work_idx = persist_iter * persist_grid + persist_lane;
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
                int group_count = (q_count + 8 - 1) / 8;
                int first_group = groups_done & 1 ^ 0;
                int stage_group_count = (group_count - first_group + 1) / 2;
                #pragma unroll 1
                for (int stage_iteration = 0; stage_iteration < stage_group_count; stage_iteration++) {
                    int group = first_group + stage_iteration * 2;
                    int _parity_s_full = (int)((unsigned int)stage_groups % 2);
                    mbarrier_wait(s_full_addr, _parity_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int token_in_group = my_row / 16;
                    int edge_in_work = group * 8 + token_in_group;
                    int row_valid = ((edge_in_work < q_count) ? 1 : 0);
                    int owner_lane = lane / 16 * 16;
                    int owned_packed = -1;
                    if (lane == owner_lane && edge_in_work < q_count) {
                        owned_packed = k2q_qsplit_indices[head_kv * nnz_per_head + row_start + edge_in_work];
                    }
                    int packed_q = __shfl_sync(0xFFFFFFFF, owned_packed, owner_lane);
                    int q_idx = packed_q & 16777215;
                    int valid_cols = 0;
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
                    float score_values[128];
                    int score_base = taddr + (unsigned int)tmem_row_base;
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (valid_cols < 128) {
                        uint32_t _thresh_mask_0;
                        {
                            int _lim_1 = valid_cols;
                            if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                            else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                            }
                        }
                        if (!(_thresh_mask_0 & (1u << 0))) score_values[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 1))) score_values[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 2))) score_values[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 3))) score_values[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 4))) score_values[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 5))) score_values[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 6))) score_values[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 7))) score_values[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 8))) score_values[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 9))) score_values[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 10))) score_values[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 11))) score_values[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 12))) score_values[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 13))) score_values[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 14))) score_values[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 15))) score_values[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 16))) score_values[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 17))) score_values[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 18))) score_values[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 19))) score_values[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 20))) score_values[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 21))) score_values[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 22))) score_values[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 23))) score_values[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 24))) score_values[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 25))) score_values[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 26))) score_values[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 27))) score_values[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 28))) score_values[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 29))) score_values[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 30))) score_values[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_0 & (1u << 31))) score_values[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_2;
                        {
                            int _lim_3 = valid_cols - 32;
                            if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                            else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                            }
                        }
                        if (!(_thresh_mask_2 & (1u << 0))) score_values[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 1))) score_values[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 2))) score_values[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 3))) score_values[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 4))) score_values[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 5))) score_values[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 6))) score_values[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 7))) score_values[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 8))) score_values[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 9))) score_values[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 10))) score_values[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 11))) score_values[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 12))) score_values[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 13))) score_values[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 14))) score_values[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 15))) score_values[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 16))) score_values[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 17))) score_values[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 18))) score_values[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 19))) score_values[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 20))) score_values[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 21))) score_values[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 22))) score_values[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 23))) score_values[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 24))) score_values[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 25))) score_values[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 26))) score_values[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 27))) score_values[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 28))) score_values[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 29))) score_values[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 30))) score_values[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_2 & (1u << 31))) score_values[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_4;
                        {
                            int _lim_5 = valid_cols - 64;
                            if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                            else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                            }
                        }
                        if (!(_thresh_mask_4 & (1u << 0))) score_values[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 1))) score_values[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 2))) score_values[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 3))) score_values[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 4))) score_values[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 5))) score_values[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 6))) score_values[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 7))) score_values[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 8))) score_values[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 9))) score_values[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 10))) score_values[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 11))) score_values[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 12))) score_values[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 13))) score_values[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 14))) score_values[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 15))) score_values[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 16))) score_values[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 17))) score_values[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 18))) score_values[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 19))) score_values[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 20))) score_values[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 21))) score_values[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 22))) score_values[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 23))) score_values[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 24))) score_values[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 25))) score_values[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 26))) score_values[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 27))) score_values[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 28))) score_values[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 29))) score_values[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 30))) score_values[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_4 & (1u << 31))) score_values[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_6;
                        {
                            int _lim_7 = valid_cols - 96;
                            if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                            else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
                            }
                        }
                        if (!(_thresh_mask_6 & (1u << 0))) score_values[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 1))) score_values[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 2))) score_values[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 3))) score_values[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 4))) score_values[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 5))) score_values[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 6))) score_values[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 7))) score_values[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 8))) score_values[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 9))) score_values[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 10))) score_values[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 11))) score_values[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 12))) score_values[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 13))) score_values[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 14))) score_values[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 15))) score_values[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 16))) score_values[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 17))) score_values[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 18))) score_values[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 19))) score_values[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 20))) score_values[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 21))) score_values[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 22))) score_values[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 23))) score_values[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 24))) score_values[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 25))) score_values[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 26))) score_values[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 27))) score_values[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 28))) score_values[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 29))) score_values[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 30))) score_values[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_6 & (1u << 31))) score_values[127] = -CUDART_INF_F;
                    }
                    float2 _reg_reduce_max2_8 = {-CUDART_INF_F, -CUDART_INF_F};
                    row_max_x32_accum(&score_values[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values[96], _reg_reduce_max2_8);
                    float row_max = row_max_reduce(_reg_reduce_max2_8);
                    float safe_max = ((row_max == -CUDART_INF_F) ? 0.0f : row_max);
                    float score_bias = ((valid_cols > 0) ? (-safe_max) * softmax_scale_log2 : -CUDART_INF_F);
                    int _parity_p_empty = (int)((unsigned int)stage_groups % 2) ^ 1;
                    mbarrier_wait(p_empty_addr, _parity_p_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int p_base = taddr + 64 + (unsigned int)tmem_row_base;
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {score_bias, score_bias};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values)[_lf], _fma_b2_9, _fma_c2_10);
                    for (int probability_segment = 0; probability_segment < 2; probability_segment++) {
                        unsigned int score_values_bf16[16];
                        softmax_frag_exp2_cast(&(score_values + probability_segment * 32)[0], score_values_bf16, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base + probability_segment * 16), "r"(score_values_bf16[0]), "r"(score_values_bf16[1]), "r"(score_values_bf16[2]), "r"(score_values_bf16[3]), "r"(score_values_bf16[4]), "r"(score_values_bf16[5]), "r"(score_values_bf16[6]), "r"(score_values_bf16[7]), "r"(score_values_bf16[8]), "r"(score_values_bf16[9]), "r"(score_values_bf16[10]), "r"(score_values_bf16[11]), "r"(score_values_bf16[12]), "r"(score_values_bf16[13]), "r"(score_values_bf16[14]), "r"(score_values_bf16[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr);
#if __CUDA_ARCH__ == 1030
#else
                    #pragma unroll
                    for (int _le = 0; _le < 24; _le++) {
                        score_values[_le + 64] = approx_exp2(score_values[_le + 64]);
                    }
                    float2 _exp2_pair_11 = ex2_emulation_f32x2_value(make_float2(score_values[88], score_values[89]));
                    score_values[88] = _exp2_pair_11.x;
                    score_values[89] = _exp2_pair_11.y;
                    float2 _exp2_pair_12 = ex2_emulation_f32x2_value(make_float2(score_values[90], score_values[91]));
                    score_values[90] = _exp2_pair_12.x;
                    score_values[91] = _exp2_pair_12.y;
                    float2 _exp2_pair_13 = ex2_emulation_f32x2_value(make_float2(score_values[92], score_values[93]));
                    score_values[92] = _exp2_pair_13.x;
                    score_values[93] = _exp2_pair_13.y;
                    float2 _exp2_pair_14 = ex2_emulation_f32x2_value(make_float2(score_values[94], score_values[95]));
                    score_values[94] = _exp2_pair_14.x;
                    score_values[95] = _exp2_pair_14.y;
                    #pragma unroll
                    for (int _le = 0; _le < 24; _le++) {
                        score_values[_le + 96] = approx_exp2(score_values[_le + 96]);
                    }
                    float2 _exp2_pair_15 = ex2_emulation_f32x2_value(make_float2(score_values[120], score_values[121]));
                    score_values[120] = _exp2_pair_15.x;
                    score_values[121] = _exp2_pair_15.y;
                    float2 _exp2_pair_16 = ex2_emulation_f32x2_value(make_float2(score_values[122], score_values[123]));
                    score_values[122] = _exp2_pair_16.x;
                    score_values[123] = _exp2_pair_16.y;
                    float2 _exp2_pair_17 = ex2_emulation_f32x2_value(make_float2(score_values[124], score_values[125]));
                    score_values[124] = _exp2_pair_17.x;
                    score_values[125] = _exp2_pair_17.y;
                    float2 _exp2_pair_18 = ex2_emulation_f32x2_value(make_float2(score_values[126], score_values[127]));
                    score_values[126] = _exp2_pair_18.x;
                    score_values[127] = _exp2_pair_18.y;
#endif
                    unsigned int score_values_bf16_1[32];
#if __CUDA_ARCH__ == 1030
                    softmax_frag_exp2_cast(&score_values[64], score_values_bf16_1, 0);
                    softmax_frag_exp2_cast(&score_values[96], &score_values_bf16_1[16], 0);
#else
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(score_values[_lp*2 + 64], score_values[_lp*2+1 + 64]));
                        score_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
#endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base + 32), "r"(score_values_bf16_1[0]), "r"(score_values_bf16_1[1]), "r"(score_values_bf16_1[2]), "r"(score_values_bf16_1[3]), "r"(score_values_bf16_1[4]), "r"(score_values_bf16_1[5]), "r"(score_values_bf16_1[6]), "r"(score_values_bf16_1[7]), "r"(score_values_bf16_1[8]), "r"(score_values_bf16_1[9]), "r"(score_values_bf16_1[10]), "r"(score_values_bf16_1[11]), "r"(score_values_bf16_1[12]), "r"(score_values_bf16_1[13]), "r"(score_values_bf16_1[14]), "r"(score_values_bf16_1[15]), "r"(score_values_bf16_1[16]), "r"(score_values_bf16_1[17]), "r"(score_values_bf16_1[18]), "r"(score_values_bf16_1[19]), "r"(score_values_bf16_1[20]), "r"(score_values_bf16_1[21]), "r"(score_values_bf16_1[22]), "r"(score_values_bf16_1[23]), "r"(score_values_bf16_1[24]), "r"(score_values_bf16_1[25]), "r"(score_values_bf16_1[26]), "r"(score_values_bf16_1[27]), "r"(score_values_bf16_1[28]), "r"(score_values_bf16_1[29]), "r"(score_values_bf16_1[30]), "r"(score_values_bf16_1[31]));
#if __CUDA_ARCH__ == 1030
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values[96], &_reg_reduce_sum2_11);
                    float row_sum = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
#else
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values[0], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values[32], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values[64], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values[96], &_reg_reduce_sum2_19);
                    float row_sum = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
#endif
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(s_empty_addr);
                    int _parity_o_full = (int)((unsigned int)stage_groups % 2);
                    mbarrier_wait(o_full_addr, _parity_o_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int q_head_local = my_row - token_in_group * 16;
                    int output_valid = 0;
                    long long partial_row = 0;
                    float inv_sum = 0.0f;
                    if (edge_in_work < q_count) {
                        int split_slot = packed_q >> 24 & 15;
                        if (split_slot >= 0 && split_slot < topk) {
                            output_valid = 1;
                            int q_abs = q_batch_offset + q_idx;
                            int q_head = head_kv * 16 + q_head_local;
                            partial_row = (long long)split_slot * (long long)total_q * (long long)num_q_heads + (long long)q_abs * (long long)num_q_heads + (long long)q_head;
#if __CUDA_ARCH__ == 1030
                            float _rcp_12 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_12 : 0.0f);
#else
                            float _rcp_20 = approx_rcp(row_sum);
                            inv_sum = ((row_sum > 0.0f && row_sum == row_sum) ? _rcp_20 : 0.0f);
#endif
                        }
                    }
                    long long partial_base = partial_row * 128;
                    int output_row_addr = taddr + (unsigned int)256 + (unsigned int)tmem_row_base;
                    float output_values[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values[0]), "=f"(output_values[1]), "=f"(output_values[2]), "=f"(output_values[3]), "=f"(output_values[4]), "=f"(output_values[5]), "=f"(output_values[6]), "=f"(output_values[7]), "=f"(output_values[8]), "=f"(output_values[9]), "=f"(output_values[10]), "=f"(output_values[11]), "=f"(output_values[12]), "=f"(output_values[13]), "=f"(output_values[14]), "=f"(output_values[15]), "=f"(output_values[16]), "=f"(output_values[17]), "=f"(output_values[18]), "=f"(output_values[19]), "=f"(output_values[20]), "=f"(output_values[21]), "=f"(output_values[22]), "=f"(output_values[23]), "=f"(output_values[24]), "=f"(output_values[25]), "=f"(output_values[26]), "=f"(output_values[27]), "=f"(output_values[28]), "=f"(output_values[29]), "=f"(output_values[30]), "=f"(output_values[31])
                        : "r"(output_row_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values[32]), "=f"(output_values[33]), "=f"(output_values[34]), "=f"(output_values[35]), "=f"(output_values[36]), "=f"(output_values[37]), "=f"(output_values[38]), "=f"(output_values[39]), "=f"(output_values[40]), "=f"(output_values[41]), "=f"(output_values[42]), "=f"(output_values[43]), "=f"(output_values[44]), "=f"(output_values[45]), "=f"(output_values[46]), "=f"(output_values[47]), "=f"(output_values[48]), "=f"(output_values[49]), "=f"(output_values[50]), "=f"(output_values[51]), "=f"(output_values[52]), "=f"(output_values[53]), "=f"(output_values[54]), "=f"(output_values[55]), "=f"(output_values[56]), "=f"(output_values[57]), "=f"(output_values[58]), "=f"(output_values[59]), "=f"(output_values[60]), "=f"(output_values[61]), "=f"(output_values[62]), "=f"(output_values[63])
                        : "r"(output_row_addr + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values[64]), "=f"(output_values[65]), "=f"(output_values[66]), "=f"(output_values[67]), "=f"(output_values[68]), "=f"(output_values[69]), "=f"(output_values[70]), "=f"(output_values[71]), "=f"(output_values[72]), "=f"(output_values[73]), "=f"(output_values[74]), "=f"(output_values[75]), "=f"(output_values[76]), "=f"(output_values[77]), "=f"(output_values[78]), "=f"(output_values[79]), "=f"(output_values[80]), "=f"(output_values[81]), "=f"(output_values[82]), "=f"(output_values[83]), "=f"(output_values[84]), "=f"(output_values[85]), "=f"(output_values[86]), "=f"(output_values[87]), "=f"(output_values[88]), "=f"(output_values[89]), "=f"(output_values[90]), "=f"(output_values[91]), "=f"(output_values[92]), "=f"(output_values[93]), "=f"(output_values[94]), "=f"(output_values[95])
                        : "r"(output_row_addr + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values[96]), "=f"(output_values[97]), "=f"(output_values[98]), "=f"(output_values[99]), "=f"(output_values[100]), "=f"(output_values[101]), "=f"(output_values[102]), "=f"(output_values[103]), "=f"(output_values[104]), "=f"(output_values[105]), "=f"(output_values[106]), "=f"(output_values[107]), "=f"(output_values[108]), "=f"(output_values[109]), "=f"(output_values[110]), "=f"(output_values[111]), "=f"(output_values[112]), "=f"(output_values[113]), "=f"(output_values[114]), "=f"(output_values[115]), "=f"(output_values[116]), "=f"(output_values[117]), "=f"(output_values[118]), "=f"(output_values[119]), "=f"(output_values[120]), "=f"(output_values[121]), "=f"(output_values[122]), "=f"(output_values[123]), "=f"(output_values[124]), "=f"(output_values[125]), "=f"(output_values[126]), "=f"(output_values[127])
                        : "r"(output_row_addr + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_13 = {inv_sum, inv_sum};
#else
                    const float2 _scale2_21 = {inv_sum, inv_sum};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(output_values)[_ls], _scale2_13);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(output_values)[_ls], _scale2_21);
#endif
                    unsigned int output_values_bf16[64];
                    #pragma unroll
                    for (int _lp = 0; _lp < 64; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 0], output_values[_lp*2+1 + 0]));
                        output_values_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int words_0[8];
                    words_0[0] = output_values_bf16[0];
                    words_0[1] = output_values_bf16[1];
                    words_0[2] = output_values_bf16[2];
                    words_0[3] = output_values_bf16[3];
                    words_0[4] = output_values_bf16[4];
                    words_0[5] = output_values_bf16[5];
                    words_0[6] = output_values_bf16[6];
                    words_0[7] = output_values_bf16[7];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + partial_base)), "r"(words_0[(0) + 0]), "r"(words_0[(0) + 1]), "r"(words_0[(0) + 2]), "r"(words_0[(0) + 3]), "r"(words_0[(0) + 4]), "r"(words_0[(0) + 5]), "r"(words_0[(0) + 6]), "r"(words_0[(0) + 7]) : "memory");
                    }
                    unsigned int words_1[8];
                    words_1[0] = output_values_bf16[8];
                    words_1[1] = output_values_bf16[9];
                    words_1[2] = output_values_bf16[10];
                    words_1[3] = output_values_bf16[11];
                    words_1[4] = output_values_bf16[12];
                    words_1[5] = output_values_bf16[13];
                    words_1[6] = output_values_bf16[14];
                    words_1[7] = output_values_bf16[15];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 16))), "r"(words_1[(0) + 0]), "r"(words_1[(0) + 1]), "r"(words_1[(0) + 2]), "r"(words_1[(0) + 3]), "r"(words_1[(0) + 4]), "r"(words_1[(0) + 5]), "r"(words_1[(0) + 6]), "r"(words_1[(0) + 7]) : "memory");
                    }
                    unsigned int words_2[8];
                    words_2[0] = output_values_bf16[16];
                    words_2[1] = output_values_bf16[17];
                    words_2[2] = output_values_bf16[18];
                    words_2[3] = output_values_bf16[19];
                    words_2[4] = output_values_bf16[20];
                    words_2[5] = output_values_bf16[21];
                    words_2[6] = output_values_bf16[22];
                    words_2[7] = output_values_bf16[23];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 32))), "r"(words_2[(0) + 0]), "r"(words_2[(0) + 1]), "r"(words_2[(0) + 2]), "r"(words_2[(0) + 3]), "r"(words_2[(0) + 4]), "r"(words_2[(0) + 5]), "r"(words_2[(0) + 6]), "r"(words_2[(0) + 7]) : "memory");
                    }
                    unsigned int words_3[8];
                    words_3[0] = output_values_bf16[24];
                    words_3[1] = output_values_bf16[25];
                    words_3[2] = output_values_bf16[26];
                    words_3[3] = output_values_bf16[27];
                    words_3[4] = output_values_bf16[28];
                    words_3[5] = output_values_bf16[29];
                    words_3[6] = output_values_bf16[30];
                    words_3[7] = output_values_bf16[31];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 48))), "r"(words_3[(0) + 0]), "r"(words_3[(0) + 1]), "r"(words_3[(0) + 2]), "r"(words_3[(0) + 3]), "r"(words_3[(0) + 4]), "r"(words_3[(0) + 5]), "r"(words_3[(0) + 6]), "r"(words_3[(0) + 7]) : "memory");
                    }
                    unsigned int words_4[8];
                    words_4[0] = output_values_bf16[32];
                    words_4[1] = output_values_bf16[33];
                    words_4[2] = output_values_bf16[34];
                    words_4[3] = output_values_bf16[35];
                    words_4[4] = output_values_bf16[36];
                    words_4[5] = output_values_bf16[37];
                    words_4[6] = output_values_bf16[38];
                    words_4[7] = output_values_bf16[39];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 64))), "r"(words_4[(0) + 0]), "r"(words_4[(0) + 1]), "r"(words_4[(0) + 2]), "r"(words_4[(0) + 3]), "r"(words_4[(0) + 4]), "r"(words_4[(0) + 5]), "r"(words_4[(0) + 6]), "r"(words_4[(0) + 7]) : "memory");
                    }
                    unsigned int words_5[8];
                    words_5[0] = output_values_bf16[40];
                    words_5[1] = output_values_bf16[41];
                    words_5[2] = output_values_bf16[42];
                    words_5[3] = output_values_bf16[43];
                    words_5[4] = output_values_bf16[44];
                    words_5[5] = output_values_bf16[45];
                    words_5[6] = output_values_bf16[46];
                    words_5[7] = output_values_bf16[47];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 80))), "r"(words_5[(0) + 0]), "r"(words_5[(0) + 1]), "r"(words_5[(0) + 2]), "r"(words_5[(0) + 3]), "r"(words_5[(0) + 4]), "r"(words_5[(0) + 5]), "r"(words_5[(0) + 6]), "r"(words_5[(0) + 7]) : "memory");
                    }
                    unsigned int words_6[8];
                    words_6[0] = output_values_bf16[48];
                    words_6[1] = output_values_bf16[49];
                    words_6[2] = output_values_bf16[50];
                    words_6[3] = output_values_bf16[51];
                    words_6[4] = output_values_bf16[52];
                    words_6[5] = output_values_bf16[53];
                    words_6[6] = output_values_bf16[54];
                    words_6[7] = output_values_bf16[55];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 96))), "r"(words_6[(0) + 0]), "r"(words_6[(0) + 1]), "r"(words_6[(0) + 2]), "r"(words_6[(0) + 3]), "r"(words_6[(0) + 4]), "r"(words_6[(0) + 5]), "r"(words_6[(0) + 6]), "r"(words_6[(0) + 7]) : "memory");
                    }
                    unsigned int words_7[8];
                    words_7[0] = output_values_bf16[56];
                    words_7[1] = output_values_bf16[57];
                    words_7[2] = output_values_bf16[58];
                    words_7[3] = output_values_bf16[59];
                    words_7[4] = output_values_bf16[60];
                    words_7[5] = output_values_bf16[61];
                    words_7[6] = output_values_bf16[62];
                    words_7[7] = output_values_bf16[63];
                    if (output_valid != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base + 112))), "r"(words_7[(0) + 0]), "r"(words_7[(0) + 1]), "r"(words_7[(0) + 2]), "r"(words_7[(0) + 3]), "r"(words_7[(0) + 4]), "r"(words_7[(0) + 5]), "r"(words_7[(0) + 6]), "r"(words_7[(0) + 7]) : "memory");
                    }
                    if (output_valid != 0) {
#if __CUDA_ARCH__ == 1030
                        float _log2_14;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_14) : "f"(row_sum));
                        float partial_lse_value = ((row_sum > 0.0f) ? __fmaf_rn(row_max * softmax_scale_log2, 0.6931472f, _log2_14 * 0.6931472f) : -CUDART_INF_F);
#else
                        float _log2_22;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22) : "f"(row_sum));
                        float partial_lse_value = ((row_sum > 0.0f) ? __fmaf_rn(row_max * softmax_scale_log2, 0.6931472f, _log2_22 * 0.6931472f) : -CUDART_INF_F);
#endif
                        partial_lse[partial_row] = partial_lse_value;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row] = partial_lse_value;
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr);
                    stage_groups += 1;
                }
                groups_done = groups_done + group_count;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            __threadfence();
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"(reinterpret_cast<unsigned int*>(grid_barrier)), "r"(static_cast<unsigned int>(1)) : "memory");
                    {
                        unsigned int* _gc_p = reinterpret_cast<unsigned int*>(grid_barrier) + (0);
                        unsigned int _gc_v;
                        do {
                            _gc_v = atomicAdd(_gc_p, 0u);
                        } while (_gc_v < (unsigned int)(num_bids));
                        __threadfence();
                    }
                }
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            __threadfence();
            int row_group = tid / 8;
            int lane_in_row = lane % 8;
            int source_lane_base = lane / 8 * 8;
            int total_rows_out = total_q * num_q_heads;
            int row_blocks = (total_rows_out + 31) / 32;
            #pragma unroll 1
            for (int row_block = bid; row_block < row_blocks; row_block += num_bids) {
                int row = row_block * 32 + row_group;
                int row_valid_1 = ((row < total_rows_out) ? 1 : 0);
                int split_count = 0;
                if (row_valid_1 != 0) {
                    int q_abs_1 = row / num_q_heads;
                    int q_head_1 = row - q_abs_1 * num_q_heads;
                    int kv_head = q_head_1 / 16;
                    split_count = split_counts[q_abs_1 * num_kv_heads + kv_head];
                    if (split_count > topk) {
                        split_count = topk;
                    }
                    if (split_count > 16) {
                        split_count = 16;
                    }
                    if (split_count < 0) {
                        split_count = 0;
                    }
                }
                int split0 = lane_in_row;
                int split1 = lane_in_row + 8;
                float lse0 = -CUDART_INF_F;
                float lse1 = -CUDART_INF_F;
                if (row_valid_1 != 0 && split0 < split_count) {
                    lse0 = partial_lse[(long long)split0 * (long long)total_rows_out + (long long)row];
                }
                if (row_valid_1 != 0 && split1 < split_count) {
                    lse1 = partial_lse[(long long)split1 * (long long)total_rows_out + (long long)row];
                }
                float lse_max;
                asm("max.f32 %0, %1, %2;" : "=f"(lse_max) : "f"(lse0), "f"(lse1));
                float peer_max_2_0 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 4);
                if (peer_max_2_0 > lse_max) {
                    lse_max = peer_max_2_0;
                }
                float peer_max_2_1 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 2);
                if (peer_max_2_1 > lse_max) {
                    lse_max = peer_max_2_1;
                }
                float peer_max_2_2 = __shfl_xor_sync(0xFFFFFFFF, lse_max, 1);
                if (peer_max_2_2 > lse_max) {
                    lse_max = peer_max_2_2;
                }
                float safe_lse_max = ((lse_max == -CUDART_INF_F) ? 0.0f : lse_max);
                float weight0 = 0.0f;
                float weight1 = 0.0f;
                if (split0 < split_count) {
#if __CUDA_ARCH__ == 1030
                    float _exp2_15 = approx_exp2((lse0 - safe_lse_max) * 1.442695f);
                    weight0 = _exp2_15;
#else
                    float _exp2_23 = approx_exp2((lse0 - safe_lse_max) * 1.442695f);
                    weight0 = _exp2_23;
#endif
                    if (lse0 == -CUDART_INF_F) {
                        weight0 = 0.0f;
                    }
                }
                if (split1 < split_count) {
#if __CUDA_ARCH__ == 1030
                    float _exp2_16 = approx_exp2((lse1 - safe_lse_max) * 1.442695f);
                    weight1 = _exp2_16;
#else
                    float _exp2_24 = approx_exp2((lse1 - safe_lse_max) * 1.442695f);
                    weight1 = _exp2_24;
#endif
                    if (lse1 == -CUDART_INF_F) {
                        weight1 = 0.0f;
                    }
                }
                float lse_sum = weight0 + weight1;
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_17 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
                lse_sum += _shfl_xor_sync_17;
                float _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
                lse_sum += _shfl_xor_sync_18;
                float _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
                lse_sum += _shfl_xor_sync_19;
                float _rcp_20 = approx_rcp(lse_sum);
                float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_20 : 0.0f);
#else
                float _shfl_xor_sync_25 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 4);
                lse_sum += _shfl_xor_sync_25;
                float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 2);
                lse_sum += _shfl_xor_sync_26;
                float _shfl_xor_sync_27 = __shfl_xor_sync(0xFFFFFFFF, lse_sum, 1);
                lse_sum += _shfl_xor_sync_27;
                float _rcp_28 = approx_rcp(lse_sum);
                float inv_lse_sum = ((lse_sum > 0.0f && lse_sum == lse_sum) ? _rcp_28 : 0.0f);
#endif
                weight0 *= inv_lse_sum;
                weight1 *= inv_lse_sum;
                if (row_valid_1 != 0 && lane_in_row == 0) {
#if __CUDA_ARCH__ == 1030
                    float _log2_21;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_21) : "f"(lse_sum));
                    float combined_lse = ((lse_sum > 0.0f) ? __fmaf_rn(_log2_21, 0.6931472f, safe_lse_max) : -CUDART_INF_F);
#else
                    float _log2_29;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_29) : "f"(lse_sum));
                    float combined_lse = ((lse_sum > 0.0f) ? __fmaf_rn(_log2_29, 0.6931472f, safe_lse_max) : -CUDART_INF_F);
#endif
                    if (return_softmax_lse != 0) {
                        lse[row] = combined_lse;
                    }
                    if (return_temperature_lse != 0) {
                        temperature_lse[row] = combined_lse;
                    }
                }
                float split_weights[16];
                #pragma unroll
                for (int split = 0; split < 16; split++) {
                    float source_weight = ((split < 8) ? weight0 : weight1);
#if __CUDA_ARCH__ == 1030
                    float _shfl_sync_22 = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
                    split_weights[split] = _shfl_sync_22;
#else
                    float _shfl_sync_30 = __shfl_sync(0xFFFFFFFF, source_weight, source_lane_base + split % 8);
                    split_weights[split] = _shfl_sync_30;
#endif
                }
                int col_segment = lane_in_row * 16;
                float accum[16];
                #pragma unroll
                for (int elem = 0; elem < 16; elem++) {
                    accum[elem] = 0.0f;
                }
                for (int preload_split = 0; preload_split < 4; preload_split++) {
                    for (int half = 0; half < 2; half++) {
                        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                            :: "r"(partial_stage_addr + (unsigned int)(preload_split * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half * 16)), "l"(partial_o + (((long long)preload_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment + (long long)(half * 8))), "r"((split_count > preload_split) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                }
                #pragma unroll
                for (int split_1 = 0; split_1 < 16; split_1++) {
                    float split_weight = split_weights[split_1];
                    asm volatile("cp.async.wait_group 3;");
                    unsigned int packed_values[8];
                    int stage_value_addr = partial_stage_addr + (unsigned int)(split_1 % 4 * 8192) + (unsigned int)(tid * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_values[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(0) + 3]))
                        : "r"(stage_value_addr));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_values[4])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values[(4) + 3]))
                        : "r"(stage_value_addr + 16));
                    float unpacked[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&unpacked[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_values[_pair]) << 16);
                        (&unpacked[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_values[_pair]) & 0xffff0000u);
                    }
                    if (split_count > split_1) {
                        #pragma unroll
                        for (int elem_1 = 0; elem_1 < 16; elem_1++) {
                            accum[elem_1] = __fmaf_rn(unpacked[elem_1], split_weight, accum[elem_1]);
                        }
                    }
                    int next_split = split_1 + 4;
                    int safe_next_split = ((next_split < 16) ? next_split : 0);
                    for (int half_1 = 0; half_1 < 2; half_1++) {
                        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                            :: "r"(partial_stage_addr + (unsigned int)(split_1 % 4 * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half_1 * 16)), "l"(partial_o + (((long long)safe_next_split * (long long)total_rows_out + (long long)row) * 128 + (long long)col_segment + (long long)(half_1 * 8))), "r"((next_split < 16 && next_split < split_count) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                }
                asm volatile("cp.async.wait_group 0;");
                unsigned int packed[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 0], accum[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 2], accum[_lp*2+1 + 2]));
                    packed[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 4], accum[_lp*2+1 + 4]));
                    packed[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 6], accum[_lp*2+1 + 6]));
                    packed[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 8], accum[_lp*2+1 + 8]));
                    packed[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 10], accum[_lp*2+1 + 10]));
                    packed[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 12], accum[_lp*2+1 + 12]));
                    packed[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum[_lp*2 + 14], accum[_lp*2+1 + 14]));
                    packed[_lp + 7] = *(uint32_t*)&_bf2;
                }
                if (row_valid_1 != 0) {
                    reinterpret_cast<int4*>(out + ((long long)row * 128 + (long long)col_segment))[0] = reinterpret_cast<int4*>(packed)[0];
                    reinterpret_cast<int4*>(out + ((long long)row * 128 + (long long)col_segment + 8))[0] = reinterpret_cast<int4*>(packed + (4))[0];
                }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    unsigned int departed;
                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                        : "=r"(departed) : "l"(&grid_barrier[1]), "r"(static_cast<uint32_t>(1)) : "memory");
                    if (departed == (unsigned int)num_bids - 1) {
                        *(reinterpret_cast<unsigned int*>(grid_barrier) + (0)) = 0;
                        *(reinterpret_cast<unsigned int*>(grid_barrier) + (1)) = 0;
                    }
                }
            }
        }
    }
    // ---- Role: softmax_odd ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_odd_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int my_row_1 = (warp - 4) * 32 + lane;
            int tmem_row_base_1 = (warp - 4) * 32 << 16;
            int persist_work_1 = num_work_items;
            int persist_grid_1 = gridDim.x;
            int persist_bid_1 = blockIdx.x;
            int persist_waves_1 = persist_work_1 / persist_grid_1;
            int persist_tail_1 = persist_work_1 - persist_waves_1 * persist_grid_1;
            int persist_items_1 = persist_waves_1;
            if ((persist_waves_1 & 1) == 0) {
                if (persist_bid_1 < persist_tail_1) {
                    persist_items_1 = persist_items_1 + 1;
                }
            } else if (persist_tail_1 > persist_grid_1 - 1 - persist_bid_1) {
                persist_items_1 = persist_items_1 + 1;
            }
            int groups_done_1 = 0;
            int stage_groups_1 = 0;
            #pragma unroll 1
            for (int persist_iter_1 = 0; persist_iter_1 < persist_items_1; persist_iter_1++) {
                int persist_lane_1 = persist_bid_1;
                if ((persist_iter_1 & 1) != 0) {
                    persist_lane_1 = persist_grid_1 - 1 - persist_bid_1;
                }
                int work_idx_1 = persist_iter_1 * persist_grid_1 + persist_lane_1;
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
                int group_count_1 = (q_count_1 + 8 - 1) / 8;
                int first_group_1 = groups_done_1 & 1 ^ 1;
                int stage_group_count_1 = (group_count_1 - first_group_1 + 1) / 2;
                #pragma unroll 1
                for (int stage_iteration_1 = 0; stage_iteration_1 < stage_group_count_1; stage_iteration_1++) {
                    int group_1 = first_group_1 + stage_iteration_1 * 2;
                    int _parity_s_full_1 = (int)((unsigned int)stage_groups_1 % 2);
                    mbarrier_wait(s_full_addr + 8, _parity_s_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int token_in_group_1 = my_row_1 / 16;
                    int edge_in_work_1 = group_1 * 8 + token_in_group_1;
                    int row_valid_2 = ((edge_in_work_1 < q_count_1) ? 1 : 0);
                    int owner_lane_1 = lane / 16 * 16;
                    int owned_packed_1 = -1;
                    if (lane == owner_lane_1 && edge_in_work_1 < q_count_1) {
                        owned_packed_1 = k2q_qsplit_indices[head_kv_1 * nnz_per_head + row_start_1 + edge_in_work_1];
                    }
                    int packed_q_1 = __shfl_sync(0xFFFFFFFF, owned_packed_1, owner_lane_1);
                    int q_idx_1 = packed_q_1 & 16777215;
                    int valid_cols_1 = 0;
                    if (row_valid_2 != 0) {
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
                    float score_values_1[128];
                    int score_base_1 = taddr + 128 + (unsigned int)tmem_row_base_1;
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
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (valid_cols_1 < 128) {
                        uint32_t _thresh_mask_0_1;
                        {
                            int _lim_1 = valid_cols_1;
                            if (_lim_1 <= 0) { _thresh_mask_0_1 = 0u; }
                            else if (_lim_1 >= 32) { _thresh_mask_0_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_0_1) : "r"(_lim_1));
                            }
                        }
                        if (!(_thresh_mask_0_1 & (1u << 0))) score_values_1[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 1))) score_values_1[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 2))) score_values_1[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 3))) score_values_1[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 4))) score_values_1[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 5))) score_values_1[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 6))) score_values_1[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 7))) score_values_1[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 8))) score_values_1[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 9))) score_values_1[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 10))) score_values_1[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 11))) score_values_1[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 12))) score_values_1[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 13))) score_values_1[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 14))) score_values_1[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 15))) score_values_1[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 16))) score_values_1[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 17))) score_values_1[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 18))) score_values_1[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 19))) score_values_1[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 20))) score_values_1[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 21))) score_values_1[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 22))) score_values_1[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 23))) score_values_1[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 24))) score_values_1[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 25))) score_values_1[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 26))) score_values_1[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 27))) score_values_1[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 28))) score_values_1[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 29))) score_values_1[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 30))) score_values_1[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_0_1 & (1u << 31))) score_values_1[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_2_1;
                        {
                            int _lim_3 = valid_cols_1 - 32;
                            if (_lim_3 <= 0) { _thresh_mask_2_1 = 0u; }
                            else if (_lim_3 >= 32) { _thresh_mask_2_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_2_1) : "r"(_lim_3));
                            }
                        }
                        if (!(_thresh_mask_2_1 & (1u << 0))) score_values_1[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 1))) score_values_1[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 2))) score_values_1[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 3))) score_values_1[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 4))) score_values_1[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 5))) score_values_1[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 6))) score_values_1[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 7))) score_values_1[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 8))) score_values_1[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 9))) score_values_1[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 10))) score_values_1[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 11))) score_values_1[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 12))) score_values_1[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 13))) score_values_1[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 14))) score_values_1[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 15))) score_values_1[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 16))) score_values_1[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 17))) score_values_1[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 18))) score_values_1[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 19))) score_values_1[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 20))) score_values_1[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 21))) score_values_1[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 22))) score_values_1[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 23))) score_values_1[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 24))) score_values_1[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 25))) score_values_1[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 26))) score_values_1[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 27))) score_values_1[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 28))) score_values_1[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 29))) score_values_1[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 30))) score_values_1[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_2_1 & (1u << 31))) score_values_1[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_4_1;
                        {
                            int _lim_5 = valid_cols_1 - 64;
                            if (_lim_5 <= 0) { _thresh_mask_4_1 = 0u; }
                            else if (_lim_5 >= 32) { _thresh_mask_4_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_4_1) : "r"(_lim_5));
                            }
                        }
                        if (!(_thresh_mask_4_1 & (1u << 0))) score_values_1[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 1))) score_values_1[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 2))) score_values_1[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 3))) score_values_1[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 4))) score_values_1[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 5))) score_values_1[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 6))) score_values_1[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 7))) score_values_1[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 8))) score_values_1[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 9))) score_values_1[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 10))) score_values_1[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 11))) score_values_1[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 12))) score_values_1[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 13))) score_values_1[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 14))) score_values_1[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 15))) score_values_1[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 16))) score_values_1[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 17))) score_values_1[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 18))) score_values_1[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 19))) score_values_1[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 20))) score_values_1[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 21))) score_values_1[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 22))) score_values_1[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 23))) score_values_1[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 24))) score_values_1[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 25))) score_values_1[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 26))) score_values_1[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 27))) score_values_1[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 28))) score_values_1[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 29))) score_values_1[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 30))) score_values_1[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_4_1 & (1u << 31))) score_values_1[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_6_1;
                        {
                            int _lim_7 = valid_cols_1 - 96;
                            if (_lim_7 <= 0) { _thresh_mask_6_1 = 0u; }
                            else if (_lim_7 >= 32) { _thresh_mask_6_1 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_6_1) : "r"(_lim_7));
                            }
                        }
                        if (!(_thresh_mask_6_1 & (1u << 0))) score_values_1[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 1))) score_values_1[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 2))) score_values_1[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 3))) score_values_1[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 4))) score_values_1[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 5))) score_values_1[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 6))) score_values_1[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 7))) score_values_1[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 8))) score_values_1[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 9))) score_values_1[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 10))) score_values_1[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 11))) score_values_1[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 12))) score_values_1[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 13))) score_values_1[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 14))) score_values_1[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 15))) score_values_1[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 16))) score_values_1[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 17))) score_values_1[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 18))) score_values_1[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 19))) score_values_1[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 20))) score_values_1[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 21))) score_values_1[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 22))) score_values_1[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 23))) score_values_1[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 24))) score_values_1[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 25))) score_values_1[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 26))) score_values_1[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 27))) score_values_1[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 28))) score_values_1[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 29))) score_values_1[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 30))) score_values_1[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_6_1 & (1u << 31))) score_values_1[127] = -CUDART_INF_F;
                    }
                    float2 _reg_reduce_max2_8 = {-CUDART_INF_F, -CUDART_INF_F};
                    row_max_x32_accum(&score_values_1[0], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[32], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[64], _reg_reduce_max2_8);
                    row_max_x32_accum(&score_values_1[96], _reg_reduce_max2_8);
                    float row_max_1 = row_max_reduce(_reg_reduce_max2_8);
                    float safe_max_1 = ((row_max_1 == -CUDART_INF_F) ? 0.0f : row_max_1);
                    float score_bias_1 = ((valid_cols_1 > 0) ? (-safe_max_1) * softmax_scale_log2 : -CUDART_INF_F);
                    int _parity_p_empty_1 = (int)((unsigned int)stage_groups_1 % 2) ^ 1;
                    mbarrier_wait(p_empty_addr + 8, _parity_p_empty_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int p_base_1 = taddr + 128 + 64 + (unsigned int)tmem_row_base_1;
                    const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_10 = {score_bias_1, score_bias_1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(score_values_1)[_lf], _fma_b2_9, _fma_c2_10);
                    for (int probability_segment_1 = 0; probability_segment_1 < 2; probability_segment_1++) {
                        unsigned int score_values_bf16_2[16];
                        softmax_frag_exp2_cast(&(score_values_1 + probability_segment_1 * 32)[0], score_values_bf16_2, 0);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base_1 + probability_segment_1 * 16), "r"(score_values_bf16_2[0]), "r"(score_values_bf16_2[1]), "r"(score_values_bf16_2[2]), "r"(score_values_bf16_2[3]), "r"(score_values_bf16_2[4]), "r"(score_values_bf16_2[5]), "r"(score_values_bf16_2[6]), "r"(score_values_bf16_2[7]), "r"(score_values_bf16_2[8]), "r"(score_values_bf16_2[9]), "r"(score_values_bf16_2[10]), "r"(score_values_bf16_2[11]), "r"(score_values_bf16_2[12]), "r"(score_values_bf16_2[13]), "r"(score_values_bf16_2[14]), "r"(score_values_bf16_2[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_addr + 8);
#if __CUDA_ARCH__ == 1030
#else
                    #pragma unroll
                    for (int _le = 0; _le < 24; _le++) {
                        score_values_1[_le + 64] = approx_exp2(score_values_1[_le + 64]);
                    }
                    float2 _exp2_pair_11 = ex2_emulation_f32x2_value(make_float2(score_values_1[88], score_values_1[89]));
                    score_values_1[88] = _exp2_pair_11.x;
                    score_values_1[89] = _exp2_pair_11.y;
                    float2 _exp2_pair_12 = ex2_emulation_f32x2_value(make_float2(score_values_1[90], score_values_1[91]));
                    score_values_1[90] = _exp2_pair_12.x;
                    score_values_1[91] = _exp2_pair_12.y;
                    float2 _exp2_pair_13 = ex2_emulation_f32x2_value(make_float2(score_values_1[92], score_values_1[93]));
                    score_values_1[92] = _exp2_pair_13.x;
                    score_values_1[93] = _exp2_pair_13.y;
                    float2 _exp2_pair_14 = ex2_emulation_f32x2_value(make_float2(score_values_1[94], score_values_1[95]));
                    score_values_1[94] = _exp2_pair_14.x;
                    score_values_1[95] = _exp2_pair_14.y;
                    #pragma unroll
                    for (int _le = 0; _le < 24; _le++) {
                        score_values_1[_le + 96] = approx_exp2(score_values_1[_le + 96]);
                    }
                    float2 _exp2_pair_15 = ex2_emulation_f32x2_value(make_float2(score_values_1[120], score_values_1[121]));
                    score_values_1[120] = _exp2_pair_15.x;
                    score_values_1[121] = _exp2_pair_15.y;
                    float2 _exp2_pair_16 = ex2_emulation_f32x2_value(make_float2(score_values_1[122], score_values_1[123]));
                    score_values_1[122] = _exp2_pair_16.x;
                    score_values_1[123] = _exp2_pair_16.y;
                    float2 _exp2_pair_17 = ex2_emulation_f32x2_value(make_float2(score_values_1[124], score_values_1[125]));
                    score_values_1[124] = _exp2_pair_17.x;
                    score_values_1[125] = _exp2_pair_17.y;
                    float2 _exp2_pair_18 = ex2_emulation_f32x2_value(make_float2(score_values_1[126], score_values_1[127]));
                    score_values_1[126] = _exp2_pair_18.x;
                    score_values_1[127] = _exp2_pair_18.y;
#endif
                    unsigned int score_values_bf16_3[32];
#if __CUDA_ARCH__ == 1030
                    softmax_frag_exp2_cast(&score_values_1[64], score_values_bf16_3, 0);
                    softmax_frag_exp2_cast(&score_values_1[96], &score_values_bf16_3[16], 0);
#else
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(score_values_1[_lp*2 + 64], score_values_1[_lp*2+1 + 64]));
                        score_values_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
#endif
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x32.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                        :: "r"(p_base_1 + 32), "r"(score_values_bf16_3[0]), "r"(score_values_bf16_3[1]), "r"(score_values_bf16_3[2]), "r"(score_values_bf16_3[3]), "r"(score_values_bf16_3[4]), "r"(score_values_bf16_3[5]), "r"(score_values_bf16_3[6]), "r"(score_values_bf16_3[7]), "r"(score_values_bf16_3[8]), "r"(score_values_bf16_3[9]), "r"(score_values_bf16_3[10]), "r"(score_values_bf16_3[11]), "r"(score_values_bf16_3[12]), "r"(score_values_bf16_3[13]), "r"(score_values_bf16_3[14]), "r"(score_values_bf16_3[15]), "r"(score_values_bf16_3[16]), "r"(score_values_bf16_3[17]), "r"(score_values_bf16_3[18]), "r"(score_values_bf16_3[19]), "r"(score_values_bf16_3[20]), "r"(score_values_bf16_3[21]), "r"(score_values_bf16_3[22]), "r"(score_values_bf16_3[23]), "r"(score_values_bf16_3[24]), "r"(score_values_bf16_3[25]), "r"(score_values_bf16_3[26]), "r"(score_values_bf16_3[27]), "r"(score_values_bf16_3[28]), "r"(score_values_bf16_3[29]), "r"(score_values_bf16_3[30]), "r"(score_values_bf16_3[31]));
#if __CUDA_ARCH__ == 1030
                    float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values_1[0], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[32], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[64], &_reg_reduce_sum2_11);
                    softmax_block_sum(&score_values_1[96], &_reg_reduce_sum2_11);
                    float row_sum_1 = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
#else
                    float2 _reg_reduce_sum2_19 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&score_values_1[0], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values_1[32], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values_1[64], &_reg_reduce_sum2_19);
                    softmax_block_sum(&score_values_1[96], &_reg_reduce_sum2_19);
                    float row_sum_1 = _reg_reduce_sum2_19.x + _reg_reduce_sum2_19.y;
#endif
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_2_addr + 8);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(s_empty_addr + 8);
                    int _parity_o_full_1 = (int)((unsigned int)stage_groups_1 % 2);
                    mbarrier_wait(o_full_addr + 8, _parity_o_full_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int q_head_local_1 = my_row_1 - token_in_group_1 * 16;
                    int output_valid_1 = 0;
                    long long partial_row_1 = 0;
                    float inv_sum_1 = 0.0f;
                    if (edge_in_work_1 < q_count_1) {
                        int split_slot_1 = packed_q_1 >> 24 & 15;
                        if (split_slot_1 >= 0 && split_slot_1 < topk) {
                            output_valid_1 = 1;
                            int q_abs_2 = q_batch_offset_1 + q_idx_1;
                            int q_head_2 = head_kv_1 * 16 + q_head_local_1;
                            partial_row_1 = (long long)split_slot_1 * (long long)total_q * (long long)num_q_heads + (long long)q_abs_2 * (long long)num_q_heads + (long long)q_head_2;
#if __CUDA_ARCH__ == 1030
                            float _rcp_12_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_12_1 : 0.0f);
#else
                            float _rcp_20_1 = approx_rcp(row_sum_1);
                            inv_sum_1 = ((row_sum_1 > 0.0f && row_sum_1 == row_sum_1) ? _rcp_20_1 : 0.0f);
#endif
                        }
                    }
                    long long partial_base_1 = partial_row_1 * 128;
                    int output_row_addr_1 = taddr + (unsigned int)256 + 128 + (unsigned int)tmem_row_base_1;
                    float output_values_1[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values_1[0]), "=f"(output_values_1[1]), "=f"(output_values_1[2]), "=f"(output_values_1[3]), "=f"(output_values_1[4]), "=f"(output_values_1[5]), "=f"(output_values_1[6]), "=f"(output_values_1[7]), "=f"(output_values_1[8]), "=f"(output_values_1[9]), "=f"(output_values_1[10]), "=f"(output_values_1[11]), "=f"(output_values_1[12]), "=f"(output_values_1[13]), "=f"(output_values_1[14]), "=f"(output_values_1[15]), "=f"(output_values_1[16]), "=f"(output_values_1[17]), "=f"(output_values_1[18]), "=f"(output_values_1[19]), "=f"(output_values_1[20]), "=f"(output_values_1[21]), "=f"(output_values_1[22]), "=f"(output_values_1[23]), "=f"(output_values_1[24]), "=f"(output_values_1[25]), "=f"(output_values_1[26]), "=f"(output_values_1[27]), "=f"(output_values_1[28]), "=f"(output_values_1[29]), "=f"(output_values_1[30]), "=f"(output_values_1[31])
                        : "r"(output_row_addr_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values_1[32]), "=f"(output_values_1[33]), "=f"(output_values_1[34]), "=f"(output_values_1[35]), "=f"(output_values_1[36]), "=f"(output_values_1[37]), "=f"(output_values_1[38]), "=f"(output_values_1[39]), "=f"(output_values_1[40]), "=f"(output_values_1[41]), "=f"(output_values_1[42]), "=f"(output_values_1[43]), "=f"(output_values_1[44]), "=f"(output_values_1[45]), "=f"(output_values_1[46]), "=f"(output_values_1[47]), "=f"(output_values_1[48]), "=f"(output_values_1[49]), "=f"(output_values_1[50]), "=f"(output_values_1[51]), "=f"(output_values_1[52]), "=f"(output_values_1[53]), "=f"(output_values_1[54]), "=f"(output_values_1[55]), "=f"(output_values_1[56]), "=f"(output_values_1[57]), "=f"(output_values_1[58]), "=f"(output_values_1[59]), "=f"(output_values_1[60]), "=f"(output_values_1[61]), "=f"(output_values_1[62]), "=f"(output_values_1[63])
                        : "r"(output_row_addr_1 + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values_1[64]), "=f"(output_values_1[65]), "=f"(output_values_1[66]), "=f"(output_values_1[67]), "=f"(output_values_1[68]), "=f"(output_values_1[69]), "=f"(output_values_1[70]), "=f"(output_values_1[71]), "=f"(output_values_1[72]), "=f"(output_values_1[73]), "=f"(output_values_1[74]), "=f"(output_values_1[75]), "=f"(output_values_1[76]), "=f"(output_values_1[77]), "=f"(output_values_1[78]), "=f"(output_values_1[79]), "=f"(output_values_1[80]), "=f"(output_values_1[81]), "=f"(output_values_1[82]), "=f"(output_values_1[83]), "=f"(output_values_1[84]), "=f"(output_values_1[85]), "=f"(output_values_1[86]), "=f"(output_values_1[87]), "=f"(output_values_1[88]), "=f"(output_values_1[89]), "=f"(output_values_1[90]), "=f"(output_values_1[91]), "=f"(output_values_1[92]), "=f"(output_values_1[93]), "=f"(output_values_1[94]), "=f"(output_values_1[95])
                        : "r"(output_row_addr_1 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(output_values_1[96]), "=f"(output_values_1[97]), "=f"(output_values_1[98]), "=f"(output_values_1[99]), "=f"(output_values_1[100]), "=f"(output_values_1[101]), "=f"(output_values_1[102]), "=f"(output_values_1[103]), "=f"(output_values_1[104]), "=f"(output_values_1[105]), "=f"(output_values_1[106]), "=f"(output_values_1[107]), "=f"(output_values_1[108]), "=f"(output_values_1[109]), "=f"(output_values_1[110]), "=f"(output_values_1[111]), "=f"(output_values_1[112]), "=f"(output_values_1[113]), "=f"(output_values_1[114]), "=f"(output_values_1[115]), "=f"(output_values_1[116]), "=f"(output_values_1[117]), "=f"(output_values_1[118]), "=f"(output_values_1[119]), "=f"(output_values_1[120]), "=f"(output_values_1[121]), "=f"(output_values_1[122]), "=f"(output_values_1[123]), "=f"(output_values_1[124]), "=f"(output_values_1[125]), "=f"(output_values_1[126]), "=f"(output_values_1[127])
                        : "r"(output_row_addr_1 + 96));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                    const float2 _scale2_13 = {inv_sum_1, inv_sum_1};
#else
                    const float2 _scale2_21 = {inv_sum_1, inv_sum_1};
#endif
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
#if __CUDA_ARCH__ == 1030
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(output_values_1)[_ls], _scale2_13);
#else
                        mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(output_values_1)[_ls], _scale2_21);
#endif
                    unsigned int output_values_bf16_1[64];
                    #pragma unroll
                    for (int _lp = 0; _lp < 64; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values_1[_lp*2 + 0], output_values_1[_lp*2+1 + 0]));
                        output_values_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    unsigned int words_8[8];
                    words_8[0] = output_values_bf16_1[0];
                    words_8[1] = output_values_bf16_1[1];
                    words_8[2] = output_values_bf16_1[2];
                    words_8[3] = output_values_bf16_1[3];
                    words_8[4] = output_values_bf16_1[4];
                    words_8[5] = output_values_bf16_1[5];
                    words_8[6] = output_values_bf16_1[6];
                    words_8[7] = output_values_bf16_1[7];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + partial_base_1)), "r"(words_8[(0) + 0]), "r"(words_8[(0) + 1]), "r"(words_8[(0) + 2]), "r"(words_8[(0) + 3]), "r"(words_8[(0) + 4]), "r"(words_8[(0) + 5]), "r"(words_8[(0) + 6]), "r"(words_8[(0) + 7]) : "memory");
                    }
                    unsigned int words_9[8];
                    words_9[0] = output_values_bf16_1[8];
                    words_9[1] = output_values_bf16_1[9];
                    words_9[2] = output_values_bf16_1[10];
                    words_9[3] = output_values_bf16_1[11];
                    words_9[4] = output_values_bf16_1[12];
                    words_9[5] = output_values_bf16_1[13];
                    words_9[6] = output_values_bf16_1[14];
                    words_9[7] = output_values_bf16_1[15];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 16))), "r"(words_9[(0) + 0]), "r"(words_9[(0) + 1]), "r"(words_9[(0) + 2]), "r"(words_9[(0) + 3]), "r"(words_9[(0) + 4]), "r"(words_9[(0) + 5]), "r"(words_9[(0) + 6]), "r"(words_9[(0) + 7]) : "memory");
                    }
                    unsigned int words_10[8];
                    words_10[0] = output_values_bf16_1[16];
                    words_10[1] = output_values_bf16_1[17];
                    words_10[2] = output_values_bf16_1[18];
                    words_10[3] = output_values_bf16_1[19];
                    words_10[4] = output_values_bf16_1[20];
                    words_10[5] = output_values_bf16_1[21];
                    words_10[6] = output_values_bf16_1[22];
                    words_10[7] = output_values_bf16_1[23];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 32))), "r"(words_10[(0) + 0]), "r"(words_10[(0) + 1]), "r"(words_10[(0) + 2]), "r"(words_10[(0) + 3]), "r"(words_10[(0) + 4]), "r"(words_10[(0) + 5]), "r"(words_10[(0) + 6]), "r"(words_10[(0) + 7]) : "memory");
                    }
                    unsigned int words_11[8];
                    words_11[0] = output_values_bf16_1[24];
                    words_11[1] = output_values_bf16_1[25];
                    words_11[2] = output_values_bf16_1[26];
                    words_11[3] = output_values_bf16_1[27];
                    words_11[4] = output_values_bf16_1[28];
                    words_11[5] = output_values_bf16_1[29];
                    words_11[6] = output_values_bf16_1[30];
                    words_11[7] = output_values_bf16_1[31];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 48))), "r"(words_11[(0) + 0]), "r"(words_11[(0) + 1]), "r"(words_11[(0) + 2]), "r"(words_11[(0) + 3]), "r"(words_11[(0) + 4]), "r"(words_11[(0) + 5]), "r"(words_11[(0) + 6]), "r"(words_11[(0) + 7]) : "memory");
                    }
                    unsigned int words_12[8];
                    words_12[0] = output_values_bf16_1[32];
                    words_12[1] = output_values_bf16_1[33];
                    words_12[2] = output_values_bf16_1[34];
                    words_12[3] = output_values_bf16_1[35];
                    words_12[4] = output_values_bf16_1[36];
                    words_12[5] = output_values_bf16_1[37];
                    words_12[6] = output_values_bf16_1[38];
                    words_12[7] = output_values_bf16_1[39];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 64))), "r"(words_12[(0) + 0]), "r"(words_12[(0) + 1]), "r"(words_12[(0) + 2]), "r"(words_12[(0) + 3]), "r"(words_12[(0) + 4]), "r"(words_12[(0) + 5]), "r"(words_12[(0) + 6]), "r"(words_12[(0) + 7]) : "memory");
                    }
                    unsigned int words_13[8];
                    words_13[0] = output_values_bf16_1[40];
                    words_13[1] = output_values_bf16_1[41];
                    words_13[2] = output_values_bf16_1[42];
                    words_13[3] = output_values_bf16_1[43];
                    words_13[4] = output_values_bf16_1[44];
                    words_13[5] = output_values_bf16_1[45];
                    words_13[6] = output_values_bf16_1[46];
                    words_13[7] = output_values_bf16_1[47];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 80))), "r"(words_13[(0) + 0]), "r"(words_13[(0) + 1]), "r"(words_13[(0) + 2]), "r"(words_13[(0) + 3]), "r"(words_13[(0) + 4]), "r"(words_13[(0) + 5]), "r"(words_13[(0) + 6]), "r"(words_13[(0) + 7]) : "memory");
                    }
                    unsigned int words_14[8];
                    words_14[0] = output_values_bf16_1[48];
                    words_14[1] = output_values_bf16_1[49];
                    words_14[2] = output_values_bf16_1[50];
                    words_14[3] = output_values_bf16_1[51];
                    words_14[4] = output_values_bf16_1[52];
                    words_14[5] = output_values_bf16_1[53];
                    words_14[6] = output_values_bf16_1[54];
                    words_14[7] = output_values_bf16_1[55];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 96))), "r"(words_14[(0) + 0]), "r"(words_14[(0) + 1]), "r"(words_14[(0) + 2]), "r"(words_14[(0) + 3]), "r"(words_14[(0) + 4]), "r"(words_14[(0) + 5]), "r"(words_14[(0) + 6]), "r"(words_14[(0) + 7]) : "memory");
                    }
                    unsigned int words_15[8];
                    words_15[0] = output_values_bf16_1[56];
                    words_15[1] = output_values_bf16_1[57];
                    words_15[2] = output_values_bf16_1[58];
                    words_15[3] = output_values_bf16_1[59];
                    words_15[4] = output_values_bf16_1[60];
                    words_15[5] = output_values_bf16_1[61];
                    words_15[6] = output_values_bf16_1[62];
                    words_15[7] = output_values_bf16_1[63];
                    if (output_valid_1 != 0) {
                        asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(partial_o + (partial_base_1 + 112))), "r"(words_15[(0) + 0]), "r"(words_15[(0) + 1]), "r"(words_15[(0) + 2]), "r"(words_15[(0) + 3]), "r"(words_15[(0) + 4]), "r"(words_15[(0) + 5]), "r"(words_15[(0) + 6]), "r"(words_15[(0) + 7]) : "memory");
                    }
                    if (output_valid_1 != 0) {
#if __CUDA_ARCH__ == 1030
                        float _log2_14_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_14_1) : "f"(row_sum_1));
                        float partial_lse_value_1 = ((row_sum_1 > 0.0f) ? __fmaf_rn(row_max_1 * softmax_scale_log2, 0.6931472f, _log2_14_1 * 0.6931472f) : -CUDART_INF_F);
#else
                        float _log2_22_1;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_22_1) : "f"(row_sum_1));
                        float partial_lse_value_1 = ((row_sum_1 > 0.0f) ? __fmaf_rn(row_max_1 * softmax_scale_log2, 0.6931472f, _log2_22_1 * 0.6931472f) : -CUDART_INF_F);
#endif
                        partial_lse[partial_row_1] = partial_lse_value_1;
                        if (return_temperature_lse != 0) {
                            partial_temperature_lse[partial_row_1] = partial_lse_value_1;
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(o_empty_addr + 8);
                    stage_groups_1 += 1;
                }
                groups_done_1 = groups_done_1 + group_count_1;
            }
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            __threadfence();
            asm volatile("barrier.sync 1, 256;" ::: "memory");
            __threadfence();
            int row_group_1 = tid / 8;
            int lane_in_row_1 = lane % 8;
            int source_lane_base_1 = lane / 8 * 8;
            int total_rows_out_1 = total_q * num_q_heads;
            int row_blocks_1 = (total_rows_out_1 + 31) / 32;
            #pragma unroll 1
            for (int row_block_1 = bid; row_block_1 < row_blocks_1; row_block_1 += num_bids) {
                int row_1 = row_block_1 * 32 + row_group_1;
                int row_valid_3 = ((row_1 < total_rows_out_1) ? 1 : 0);
                int split_count_1 = 0;
                if (row_valid_3 != 0) {
                    int q_abs_3 = row_1 / num_q_heads;
                    int q_head_3 = row_1 - q_abs_3 * num_q_heads;
                    int kv_head_1 = q_head_3 / 16;
                    split_count_1 = split_counts[q_abs_3 * num_kv_heads + kv_head_1];
                    if (split_count_1 > topk) {
                        split_count_1 = topk;
                    }
                    if (split_count_1 > 16) {
                        split_count_1 = 16;
                    }
                    if (split_count_1 < 0) {
                        split_count_1 = 0;
                    }
                }
                int split0_1 = lane_in_row_1;
                int split1_1 = lane_in_row_1 + 8;
                float lse0_1 = -CUDART_INF_F;
                float lse1_1 = -CUDART_INF_F;
                if (row_valid_3 != 0 && split0_1 < split_count_1) {
                    lse0_1 = partial_lse[(long long)split0_1 * (long long)total_rows_out_1 + (long long)row_1];
                }
                if (row_valid_3 != 0 && split1_1 < split_count_1) {
                    lse1_1 = partial_lse[(long long)split1_1 * (long long)total_rows_out_1 + (long long)row_1];
                }
                float lse_max_1;
                asm("max.f32 %0, %1, %2;" : "=f"(lse_max_1) : "f"(lse0_1), "f"(lse1_1));
                float peer_max_2_3 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 4);
                if (peer_max_2_3 > lse_max_1) {
                    lse_max_1 = peer_max_2_3;
                }
                float peer_max_2_4 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 2);
                if (peer_max_2_4 > lse_max_1) {
                    lse_max_1 = peer_max_2_4;
                }
                float peer_max_2_5 = __shfl_xor_sync(0xFFFFFFFF, lse_max_1, 1);
                if (peer_max_2_5 > lse_max_1) {
                    lse_max_1 = peer_max_2_5;
                }
                float safe_lse_max_1 = ((lse_max_1 == -CUDART_INF_F) ? 0.0f : lse_max_1);
                float weight0_1 = 0.0f;
                float weight1_1 = 0.0f;
                if (split0_1 < split_count_1) {
#if __CUDA_ARCH__ == 1030
                    float _exp2_15_1 = approx_exp2((lse0_1 - safe_lse_max_1) * 1.442695f);
                    weight0_1 = _exp2_15_1;
#else
                    float _exp2_23_1 = approx_exp2((lse0_1 - safe_lse_max_1) * 1.442695f);
                    weight0_1 = _exp2_23_1;
#endif
                    if (lse0_1 == -CUDART_INF_F) {
                        weight0_1 = 0.0f;
                    }
                }
                if (split1_1 < split_count_1) {
#if __CUDA_ARCH__ == 1030
                    float _exp2_16_1 = approx_exp2((lse1_1 - safe_lse_max_1) * 1.442695f);
                    weight1_1 = _exp2_16_1;
#else
                    float _exp2_24_1 = approx_exp2((lse1_1 - safe_lse_max_1) * 1.442695f);
                    weight1_1 = _exp2_24_1;
#endif
                    if (lse1_1 == -CUDART_INF_F) {
                        weight1_1 = 0.0f;
                    }
                }
                float lse_sum_1 = weight0_1 + weight1_1;
#if __CUDA_ARCH__ == 1030
                float _shfl_xor_sync_17_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 4);
                lse_sum_1 += _shfl_xor_sync_17_1;
                float _shfl_xor_sync_18_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 2);
                lse_sum_1 += _shfl_xor_sync_18_1;
                float _shfl_xor_sync_19_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 1);
                lse_sum_1 += _shfl_xor_sync_19_1;
                float _rcp_20_1 = approx_rcp(lse_sum_1);
                float inv_lse_sum_1 = ((lse_sum_1 > 0.0f && lse_sum_1 == lse_sum_1) ? _rcp_20_1 : 0.0f);
#else
                float _shfl_xor_sync_25_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 4);
                lse_sum_1 += _shfl_xor_sync_25_1;
                float _shfl_xor_sync_26_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 2);
                lse_sum_1 += _shfl_xor_sync_26_1;
                float _shfl_xor_sync_27_1 = __shfl_xor_sync(0xFFFFFFFF, lse_sum_1, 1);
                lse_sum_1 += _shfl_xor_sync_27_1;
                float _rcp_28_1 = approx_rcp(lse_sum_1);
                float inv_lse_sum_1 = ((lse_sum_1 > 0.0f && lse_sum_1 == lse_sum_1) ? _rcp_28_1 : 0.0f);
#endif
                weight0_1 *= inv_lse_sum_1;
                weight1_1 *= inv_lse_sum_1;
                if (row_valid_3 != 0 && lane_in_row_1 == 0) {
#if __CUDA_ARCH__ == 1030
                    float _log2_21_1;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_21_1) : "f"(lse_sum_1));
                    float combined_lse_1 = ((lse_sum_1 > 0.0f) ? __fmaf_rn(_log2_21_1, 0.6931472f, safe_lse_max_1) : -CUDART_INF_F);
#else
                    float _log2_29_1;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_29_1) : "f"(lse_sum_1));
                    float combined_lse_1 = ((lse_sum_1 > 0.0f) ? __fmaf_rn(_log2_29_1, 0.6931472f, safe_lse_max_1) : -CUDART_INF_F);
#endif
                    if (return_softmax_lse != 0) {
                        lse[row_1] = combined_lse_1;
                    }
                    if (return_temperature_lse != 0) {
                        temperature_lse[row_1] = combined_lse_1;
                    }
                }
                float split_weights_1[16];
                #pragma unroll
                for (int split_2 = 0; split_2 < 16; split_2++) {
                    float source_weight_1 = ((split_2 < 8) ? weight0_1 : weight1_1);
#if __CUDA_ARCH__ == 1030
                    float _shfl_sync_22_1 = __shfl_sync(0xFFFFFFFF, source_weight_1, source_lane_base_1 + split_2 % 8);
                    split_weights_1[split_2] = _shfl_sync_22_1;
#else
                    float _shfl_sync_30_1 = __shfl_sync(0xFFFFFFFF, source_weight_1, source_lane_base_1 + split_2 % 8);
                    split_weights_1[split_2] = _shfl_sync_30_1;
#endif
                }
                int col_segment_1 = lane_in_row_1 * 16;
                float accum_1[16];
                #pragma unroll
                for (int elem_2 = 0; elem_2 < 16; elem_2++) {
                    accum_1[elem_2] = 0.0f;
                }
                for (int preload_split_1 = 0; preload_split_1 < 4; preload_split_1++) {
                    for (int half_2 = 0; half_2 < 2; half_2++) {
                        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                            :: "r"(partial_stage_addr + (unsigned int)(preload_split_1 * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half_2 * 16)), "l"(partial_o + (((long long)preload_split_1 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1 + (long long)(half_2 * 8))), "r"((split_count_1 > preload_split_1) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                }
                #pragma unroll
                for (int split_3 = 0; split_3 < 16; split_3++) {
                    float split_weight_1 = split_weights_1[split_3];
                    asm volatile("cp.async.wait_group 3;");
                    unsigned int packed_values_1[8];
                    int stage_value_addr_1 = partial_stage_addr + (unsigned int)(split_3 % 4 * 8192) + (unsigned int)(tid * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(0) + 3]))
                        : "r"(stage_value_addr_1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_values_1[(4) + 3]))
                        : "r"(stage_value_addr_1 + 16));
                    float unpacked_1[16];
                    #pragma unroll
                    for (int _pair = 0; _pair < 8; _pair++) {
                        (&unpacked_1[_pair * 2])[0] = __uint_as_float(static_cast<uint32_t>(packed_values_1[_pair]) << 16);
                        (&unpacked_1[_pair * 2])[1] = __uint_as_float(static_cast<uint32_t>(packed_values_1[_pair]) & 0xffff0000u);
                    }
                    if (split_count_1 > split_3) {
                        #pragma unroll
                        for (int elem_3 = 0; elem_3 < 16; elem_3++) {
                            accum_1[elem_3] = __fmaf_rn(unpacked_1[elem_3], split_weight_1, accum_1[elem_3]);
                        }
                    }
                    int next_split_1 = split_3 + 4;
                    int safe_next_split_1 = ((next_split_1 < 16) ? next_split_1 : 0);
                    for (int half_3 = 0; half_3 < 2; half_3++) {
                        asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16, %2;"
                            :: "r"(partial_stage_addr + (unsigned int)(split_3 % 4 * 8192) + (unsigned int)(tid * 32) + (unsigned int)(half_3 * 16)), "l"(partial_o + (((long long)safe_next_split_1 * (long long)total_rows_out_1 + (long long)row_1) * 128 + (long long)col_segment_1 + (long long)(half_3 * 8))), "r"((next_split_1 < 16 && next_split_1 < split_count_1) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                }
                asm volatile("cp.async.wait_group 0;");
                unsigned int packed_1[8];
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 0], accum_1[_lp*2+1 + 0]));
                    packed_1[_lp] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 2], accum_1[_lp*2+1 + 2]));
                    packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 4], accum_1[_lp*2+1 + 4]));
                    packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 6], accum_1[_lp*2+1 + 6]));
                    packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 8], accum_1[_lp*2+1 + 8]));
                    packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 10], accum_1[_lp*2+1 + 10]));
                    packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 12], accum_1[_lp*2+1 + 12]));
                    packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(accum_1[_lp*2 + 14], accum_1[_lp*2+1 + 14]));
                    packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                }
                if (row_valid_3 != 0) {
                    reinterpret_cast<int4*>(out + ((long long)row_1 * 128 + (long long)col_segment_1))[0] = reinterpret_cast<int4*>(packed_1)[0];
                    reinterpret_cast<int4*>(out + ((long long)row_1 * 128 + (long long)col_segment_1 + 8))[0] = reinterpret_cast<int4*>(packed_1 + (4))[0];
                }
            }
        }
    }
    // ---- Role: qload ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // qload_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int persist_work_2 = num_work_items;
            int persist_grid_2 = gridDim.x;
            int persist_bid_2 = blockIdx.x;
            int persist_waves_2 = persist_work_2 / persist_grid_2;
            int persist_tail_2 = persist_work_2 - persist_waves_2 * persist_grid_2;
            int persist_items_2 = persist_waves_2;
            if ((persist_waves_2 & 1) == 0) {
                if (persist_bid_2 < persist_tail_2) {
                    persist_items_2 = persist_items_2 + 1;
                }
            } else if (persist_tail_2 > persist_grid_2 - 1 - persist_bid_2) {
                persist_items_2 = persist_items_2 + 1;
            }
            int groups_done_2 = 0;
            #pragma unroll 1
            for (int persist_iter_2 = 0; persist_iter_2 < persist_items_2; persist_iter_2++) {
                int persist_lane_2 = persist_bid_2;
                if ((persist_iter_2 & 1) != 0) {
                    persist_lane_2 = persist_grid_2 - 1 - persist_bid_2;
                }
                int work_idx_2 = persist_iter_2 * persist_grid_2 + persist_lane_2;
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
                int group_count_2 = (q_count_2 + 8 - 1) / 8;
                #pragma unroll 1
                for (int group_2 = 0; group_2 < group_count_2; group_2++) {
                    int q_stage = (groups_done_2 + group_2) % 2;
                    int _stage_q_empty = (int)((unsigned int)(groups_done_2 + group_2) % 2);
                    int _parity_q_empty = (int)((unsigned int)(groups_done_2 + group_2) / 2 % 2) ^ 1;
                    mbarrier_wait(q_empty_addr + (_stage_q_empty) * 8, _parity_q_empty);
                    int q_stage_addr = q_store_smem_addr + (unsigned int)(q_stage * 32768);
                    int qload_warp = warp - 8;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + ((int)((unsigned int)(groups_done_2 + group_2) % 2)) * 8, 8192);
                        int token_in_group_1_0 = qload_warp * 2;
                        int edge_in_work_2_0 = group_2 * 8 + token_in_group_1_0;
                        int edge_valid_1_0 = ((edge_in_work_2_0 < q_count_2) ? 1 : 0);
                        int safe_edge_1_0 = ((edge_valid_1_0 != 0) ? edge_in_work_2_0 : 0);
                        int packed_q_1_0 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_1_0];
                        int decoded_q_abs_1_0 = q_batch_offset_2 + (packed_q_1_0 & 16777215);
                        int q_abs_1_0 = ((edge_valid_1_0 != 0) ? decoded_q_abs_1_0 : 0);
                        int dst_offset_1_0 = token_in_group_1_0 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset_1_0, (&q), 0, head_kv_2 * 16, 0, q_abs_1_0, q_full_addr + ((int)((unsigned int)(groups_done_2 + group_2) % 2)) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset_1_0, (&q), 0, head_kv_2 * 16, 1, q_abs_1_0, q_full_addr + ((int)((unsigned int)(groups_done_2 + group_2) % 2)) * 8);
                        int token_in_group_1_1 = qload_warp * 2 + 1;
                        int edge_in_work_2_1 = group_2 * 8 + token_in_group_1_1;
                        int edge_valid_1_1 = ((edge_in_work_2_1 < q_count_2) ? 1 : 0);
                        int safe_edge_1_1 = ((edge_valid_1_1 != 0) ? edge_in_work_2_1 : 0);
                        int packed_q_1_1 = k2q_qsplit_indices[head_kv_2 * nnz_per_head + row_start_2 + safe_edge_1_1];
                        int decoded_q_abs_1_1 = q_batch_offset_2 + (packed_q_1_1 & 16777215);
                        int q_abs_1_1 = ((edge_valid_1_1 != 0) ? decoded_q_abs_1_1 : 0);
                        int dst_offset_1_1 = token_in_group_1_1 * 2048;
                        tma_4d_gmem2smem(q_stage_addr + dst_offset_1_1, (&q), 0, head_kv_2 * 16, 0, q_abs_1_1, q_full_addr + ((int)((unsigned int)(groups_done_2 + group_2) % 2)) * 8);
                        tma_4d_gmem2smem(q_stage_addr + 16384 + dst_offset_1_1, (&q), 0, head_kv_2 * 16, 1, q_abs_1_1, q_full_addr + ((int)((unsigned int)(groups_done_2 + group_2) % 2)) * 8);
                    }
                }
                groups_done_2 += group_count_2;
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int persist_work_3 = num_work_items;
            int persist_grid_3 = gridDim.x;
            int persist_bid_3 = blockIdx.x;
            int persist_waves_3 = persist_work_3 / persist_grid_3;
            int persist_tail_3 = persist_work_3 - persist_waves_3 * persist_grid_3;
            int persist_items_3 = persist_waves_3;
            if ((persist_waves_3 & 1) == 0) {
                if (persist_bid_3 < persist_tail_3) {
                    persist_items_3 = persist_items_3 + 1;
                }
            } else if (persist_tail_3 > persist_grid_3 - 1 - persist_bid_3) {
                persist_items_3 = persist_items_3 + 1;
            }
            int groups_done_3 = 0;
            #pragma unroll 1
            for (int persist_iter_3 = 0; persist_iter_3 < persist_items_3; persist_iter_3++) {
                int persist_lane_3 = persist_bid_3;
                if ((persist_iter_3 & 1) != 0) {
                    persist_lane_3 = persist_grid_3 - 1 - persist_bid_3;
                }
                int work_idx_3 = persist_iter_3 * persist_grid_3 + persist_lane_3;
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
                int group_count_3 = (q_count_3 + 8 - 1) / 8;
                int kv_slot = persist_iter_3 % 2;
                int _stage_k_full = (int)((unsigned int)persist_iter_3 % 2);
                int _parity_k_full = (int)((unsigned int)persist_iter_3 / 2 % 2);
                mbarrier_wait(k_full_addr + (_stage_k_full) * 8, _parity_k_full);
                int q_stage_1 = groups_done_3 % 2;
                int _stage_q_full = (int)((unsigned int)groups_done_3 % 2);
                int _parity_q_full = (int)((unsigned int)groups_done_3 / 2 % 2);
                mbarrier_wait(q_full_addr + (_stage_q_full) * 8, _parity_q_full);
                int _stage_s_empty = (int)((unsigned int)groups_done_3 % 2);
                int _parity_s_empty = (int)((unsigned int)groups_done_3 / 2 % 2) ^ 1;
                mbarrier_wait(s_empty_addr + (_stage_s_empty) * 8, _parity_s_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int mma_a_lo = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_1) * 2048);
                int mma_b_lo = make_warp_uniform((((k_smem_addr) >> 4) & 0x3FFF) + (kv_slot) * 2048);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo);
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
                elect_commit(s_full_addr + ((int)((unsigned int)groups_done_3 % 2)) * 8);
                elect_commit(q_empty_addr + ((int)((unsigned int)groups_done_3 % 2)) * 8);
                if (group_count_3 > 1) {
                    int q_stage_0 = (groups_done_3 + 1) % 2;
                    int _stage_q_full_1 = (int)((unsigned int)(groups_done_3 + 1) % 2);
                    int _parity_q_full_1 = (int)((unsigned int)(groups_done_3 + 1) / 2 % 2);
                    mbarrier_wait(q_full_addr + (_stage_q_full_1) * 8, _parity_q_full_1);
                    int _stage_s_empty_1 = (int)((unsigned int)(groups_done_3 + 1) % 2);
                    int _parity_s_empty_1 = (int)((unsigned int)(groups_done_3 + 1) / 2 % 2) ^ 1;
                    mbarrier_wait(s_empty_addr + (_stage_s_empty_1) * 8, _parity_s_empty_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_0 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_0) * 2048);
                    int mma_b_lo_0 = make_warp_uniform((((k_smem_addr) >> 4) & 0x3FFF) + (kv_slot) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_0);
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
                    elect_commit(s_full_addr + ((int)((unsigned int)(groups_done_3 + 1) % 2)) * 8);
                    elect_commit(q_empty_addr + ((int)((unsigned int)(groups_done_3 + 1) % 2)) * 8);
                }
                int _stage_v_full = (int)((unsigned int)persist_iter_3 % 2);
                int _parity_v_full = (int)((unsigned int)persist_iter_3 / 2 % 2);
                mbarrier_wait(v_full_addr + (_stage_v_full) * 8, _parity_v_full);
                #pragma unroll 1
                for (int group_3 = 2; group_3 < group_count_3; group_3++) {
                    int pv_stage = (groups_done_3 + group_3 - 2) % 2;
                    int _stage_p_full = (int)((unsigned int)(groups_done_3 + group_3 - 2) % 2);
                    int _parity_p_full = (int)((unsigned int)(groups_done_3 + group_3 - 2) / 2 % 2);
                    mbarrier_wait(p_full_addr + (_stage_p_full) * 8, _parity_p_full);
                    int _stage_o_empty = (int)((unsigned int)(groups_done_3 + group_3 - 2) % 2);
                    int _parity_o_empty = (int)((unsigned int)(groups_done_3 + group_3 - 2) / 2 % 2) ^ 1;
                    mbarrier_wait(o_empty_addr + (_stage_o_empty) * 8, _parity_o_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_1 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot) * 2048);
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
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(mma_b_lo_1), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(0) : "memory");
                    int _stage_p_full_2 = (int)((unsigned int)(groups_done_3 + group_3 - 2) % 2);
                    int _parity_p_full_2 = (int)((unsigned int)(groups_done_3 + group_3 - 2) / 2 % 2);
                    mbarrier_wait(p_full_2_addr + (_stage_p_full_2) * 8, _parity_p_full_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_1_0 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot) * 2048);
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
                    :: "r"((tmem_output + (pv_stage * 128))), "r"(mma_b_lo_1_0), "r"(tmem_scores + (pv_stage * 128 + 64)), "r"(1) : "memory");
                    elect_commit(o_full_addr + ((int)((unsigned int)(groups_done_3 + group_3 - 2) % 2)) * 8);
                    elect_commit(p_empty_addr + ((int)((unsigned int)(groups_done_3 + group_3 - 2) % 2)) * 8);
                    int q_stage_1_1 = (groups_done_3 + group_3) % 2;
                    int _stage_q_full_2 = (int)((unsigned int)(groups_done_3 + group_3) % 2);
                    int _parity_q_full_2 = (int)((unsigned int)(groups_done_3 + group_3) / 2 % 2);
                    mbarrier_wait(q_full_addr + (_stage_q_full_2) * 8, _parity_q_full_2);
                    int _stage_s_empty_2 = (int)((unsigned int)(groups_done_3 + group_3) % 2);
                    int _parity_s_empty_2 = (int)((unsigned int)(groups_done_3 + group_3) / 2 % 2) ^ 1;
                    mbarrier_wait(s_empty_addr + (_stage_s_empty_2) * 8, _parity_s_empty_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_a_lo_1 = make_warp_uniform((((q_smem_addr) >> 4) & 0x3FFF) + (q_stage_1_1) * 2048);
                    int mma_b_lo_2 = make_warp_uniform((((k_smem_addr) >> 4) & 0x3FFF) + (kv_slot) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 1018U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_scores + (q_stage_1_1 * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 136316048, 1);
                        }
                    }
                    elect_commit(s_full_addr + ((int)((unsigned int)(groups_done_3 + group_3) % 2)) * 8);
                    elect_commit(q_empty_addr + ((int)((unsigned int)(groups_done_3 + group_3) % 2)) * 8);
                }
                int drain_start = ((group_count_3 == 1) ? 0 : group_count_3 - 2);
                #pragma unroll 1
                for (int pv_group = drain_start; pv_group < group_count_3; pv_group++) {
                    int pv_stage_1 = (groups_done_3 + pv_group) % 2;
                    int _stage_p_full_1 = (int)((unsigned int)(groups_done_3 + pv_group) % 2);
                    int _parity_p_full_1 = (int)((unsigned int)(groups_done_3 + pv_group) / 2 % 2);
                    mbarrier_wait(p_full_addr + (_stage_p_full_1) * 8, _parity_p_full_1);
                    int _stage_o_empty_1 = (int)((unsigned int)(groups_done_3 + pv_group) % 2);
                    int _parity_o_empty_1 = (int)((unsigned int)(groups_done_3 + pv_group) / 2 % 2) ^ 1;
                    mbarrier_wait(o_empty_addr + (_stage_o_empty_1) * 8, _parity_o_empty_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot) * 2048);
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
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(mma_b_lo_3), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(0) : "memory");
                    int _stage_p_full_2_1 = (int)((unsigned int)(groups_done_3 + pv_group) % 2);
                    int _parity_p_full_2_1 = (int)((unsigned int)(groups_done_3 + pv_group) / 2 % 2);
                    mbarrier_wait(p_full_2_addr + (_stage_p_full_2_1) * 8, _parity_p_full_2_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int mma_b_lo_1_1 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot) * 2048);
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
                    :: "r"((tmem_output + (pv_stage_1 * 128))), "r"(mma_b_lo_1_1), "r"(tmem_scores + (pv_stage_1 * 128 + 64)), "r"(1) : "memory");
                    elect_commit(o_full_addr + ((int)((unsigned int)(groups_done_3 + pv_group) % 2)) * 8);
                    elect_commit(p_empty_addr + ((int)((unsigned int)(groups_done_3 + pv_group) % 2)) * 8);
                }
                elect_commit(kv_empty_addr + ((int)((unsigned int)persist_iter_3 % 2)) * 8);
                groups_done_3 += group_count_3;
            }
            int _stage_o_empty_2 = (int)((unsigned int)(groups_done_3 - 1) % 2);
            int _parity_o_empty_2 = (int)((unsigned int)(groups_done_3 - 1) / 2 % 2);
            mbarrier_wait(o_empty_addr + (_stage_o_empty_2) * 8, _parity_o_empty_2);
            if (groups_done_3 > 1) {
                int _stage_o_empty_3 = (int)((unsigned int)(groups_done_3 - 2) % 2);
                int _parity_o_empty_3 = (int)((unsigned int)(groups_done_3 - 2) / 2 % 2);
                mbarrier_wait(o_empty_addr + (_stage_o_empty_3) * 8, _parity_o_empty_3);
            }
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            if (elect_sync()) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int persist_work_4 = num_work_items;
            int persist_grid_4 = gridDim.x;
            int persist_bid_4 = blockIdx.x;
            int persist_waves_4 = persist_work_4 / persist_grid_4;
            int persist_tail_4 = persist_work_4 - persist_waves_4 * persist_grid_4;
            int persist_items_4 = persist_waves_4;
            if ((persist_waves_4 & 1) == 0) {
                if (persist_bid_4 < persist_tail_4) {
                    persist_items_4 = persist_items_4 + 1;
                }
            } else if (persist_tail_4 > persist_grid_4 - 1 - persist_bid_4) {
                persist_items_4 = persist_items_4 + 1;
            }
            #pragma unroll 1
            for (int persist_iter_4 = 0; persist_iter_4 < persist_items_4; persist_iter_4++) {
                int persist_lane_4 = persist_bid_4;
                if ((persist_iter_4 & 1) != 0) {
                    persist_lane_4 = persist_grid_4 - 1 - persist_bid_4;
                }
                int work_idx_4 = persist_iter_4 * persist_grid_4 + persist_lane_4;
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
                int group_count_4 = (q_count_4 + 8 - 1) / 8;
                int kv_slot_1 = persist_iter_4 % 2;
                int token_base = k_batch_offset_4 + kv_block_4 * 128;
                int page_head = head_kv_4;
                int _stage_kv_empty = (int)((unsigned int)persist_iter_4 % 2);
                int _parity_kv_empty = (int)((unsigned int)persist_iter_4 / 2 % 2) ^ 1;
                mbarrier_wait(kv_empty_addr + (_stage_kv_empty) * 8, _parity_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(k_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8, 32768);
                    int token0 = token_base;
                    int token1 = token_base + 64;
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + (unsigned int)(kv_slot_1 * 32768)), "l"((&k)), "r"(0), "r"(token0), "r"(0), "r"(page_head),
                           "r"(k_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 8192), "l"((&k)), "r"(0), "r"(token1), "r"(0), "r"(page_head),
                           "r"(k_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 16384), "l"((&k)), "r"(0), "r"(token0), "r"(1), "r"(page_head),
                           "r"(k_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(k_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 24576), "l"((&k)), "r"(0), "r"(token1), "r"(1), "r"(page_head),
                           "r"(k_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    mbarrier_arrive_expect_tx(v_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8, 32768);
                    int token0_0 = token_base;
                    int token1_0 = token_base + 64;
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + (unsigned int)(kv_slot_1 * 32768)), "l"((&v)), "r"(0), "r"(token0_0), "r"(0), "r"(page_head),
                           "r"(v_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 8192), "l"((&v)), "r"(0), "r"(token1_0), "r"(0), "r"(page_head),
                           "r"(v_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 16384), "l"((&v)), "r"(0), "r"(token0_0), "r"(1), "r"(page_head),
                           "r"(v_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                        :: "r"(v_smem_addr + (unsigned int)(kv_slot_1 * 32768) + 24576), "l"((&v)), "r"(0), "r"(token1_0), "r"(1), "r"(page_head),
                           "r"(v_full_addr + ((int)((unsigned int)persist_iter_4 % 2)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
